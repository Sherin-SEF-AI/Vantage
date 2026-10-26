"""Review a recorded segment or a finished video: frames every few seconds, a contact sheet, and the narration's
timing (gaps between lines longer than 1.5 s are flagged as dead air).

    python3 demo/youtube/review.py coverage                  the segment demo/build/youtube/seg/coverage
    python3 demo/youtube/review.py coverage --draft          ... its draft (demo/build/youtube/draft/seg/coverage)
    python3 demo/youtube/review.py path/to/video.mp4         any video
    options: --every 2.0 (seconds between frames), --out DIR (default: <segment>/review), --cols 6

Writes DIR/frame_<t>.jpg, DIR/contact.jpg (each tile stamped with its time and the line being spoken) and prints the
timeline report.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common  # noqa: E402

DEAD_AIR = 1.5


def duration(path):
    out = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', path],
                         capture_output=True, text=True).stdout.strip()
    return float(out or 0)


def frames(video, out, every, width=640):
    tmp = os.path.join(out, '_f')
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    rate = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=r_frame_rate',
                           '-of', 'csv=p=0', video], capture_output=True, text=True).stdout.strip() or '30/1'
    num, den = (rate.split('/') + ['1'])[:2]
    step = max(1, round(every * float(num) / float(den)))        # exact frames at k * every seconds
    subprocess.run(['ffmpeg', '-v', 'error', '-i', video, '-vf', f"select='not(mod(n\\,{step}))',scale={width}:-2",
                    '-fps_mode', 'passthrough', '-q:v', '3', os.path.join(tmp, '%05d.jpg')], check=True)
    files = sorted(glob.glob(os.path.join(tmp, '*.jpg')))
    res = []
    for i, f in enumerate(files):
        t = i * every
        dst = os.path.join(out, f'frame_{t:07.1f}.jpg')
        os.replace(f, dst)
        res.append((t, dst))
    shutil.rmtree(tmp, ignore_errors=True)
    return res


def contact_sheet(items, path, cols=6, label=lambda t: ''):
    from PIL import Image, ImageDraw, ImageFont
    if not items:
        return
    ims = [Image.open(f) for _, f in items]
    tw, th = ims[0].size
    rows = -(-len(ims) // cols)
    pad, cap = 6, 34
    sheet = Image.new('RGB', (cols * (tw + pad) + pad, rows * (th + cap + pad) + pad), (18, 18, 17))
    d = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype(os.path.join(common.ROOT, 'vantage', 'resources', 'fonts', 'IBMPlexMono-Regular.ttf'), 13)
    except OSError:
        font = ImageFont.load_default()
    for i, ((t, _), im) in enumerate(zip(items, ims)):
        r, c = divmod(i, cols)
        x, y = pad + c * (tw + pad), pad + r * (th + cap + pad)
        sheet.paste(im, (x, y))
        m, s = divmod(t, 60)
        d.text((x + 2, y + th + 3), f'{int(m):02d}:{s:04.1f}', fill=(57, 135, 229), font=font)
        d.text((x + 2, y + th + 18), label(t)[:int(tw / 7.8)], fill=(195, 194, 183), font=font)
    sheet.save(path, quality=88)


def timeline_report(tl):
    lines = tl.get('lines', [])
    total = tl.get('seconds', 0)
    print(f'{tl.get("id", "?")}: {total:.1f} s, {len(lines)} lines, {sum(x["seconds"] for x in lines):.1f} s of speech')
    prev_end = 0.0
    flags = []
    for x in lines:
        gap = x['start'] - prev_end
        mark = '  <-- dead air' if gap > DEAD_AIR else ''
        if gap > DEAD_AIR:
            flags.append((prev_end, gap))
        print(f'  {x["start"]:7.2f}  +{gap:4.1f}  {x["key"]:<16} {x["seconds"]:5.1f} s  {x["caption"][:64]}{mark}')
        prev_end = x['start'] + x['seconds']
    tail = total - prev_end
    print(f'  tail after the last line: {tail:.1f} s')
    for e in tl.get('events', []):
        if e['kind'] == 'wait':
            print(f'  wait at {e["start"]:.1f}s: {e["wall_s"]:.1f} s of work shown in {e["video_s"]:.1f} s'
                  f' (x{e["wall_s"] / max(1e-6, e["video_s"]):.1f})')
    for e in tl.get('events', []):
        if e['kind'] == 'path-leak':
            print(f'  PATH ON SCREEN at {e["start"]:.1f}s: {e["what"]}')
    if tl.get('unsaid'):
        print('  NEVER SAID:', ', '.join(tl['unsaid']))
    return flags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('what')
    ap.add_argument('--draft', action='store_true')
    ap.add_argument('--every', type=float, default=3.0)
    ap.add_argument('--cols', type=int, default=6)
    ap.add_argument('--out')
    a = ap.parse_args()
    if a.what.endswith('.mp4'):
        video, tl = a.what, None
        out = a.out or os.path.splitext(a.what)[0] + '-review'
    else:
        seg = os.path.join(common.DRAFT_DIR if a.draft else common.BUILD, 'seg', a.what)
        video = os.path.join(seg, 'video.mp4')
        tl = json.load(open(os.path.join(seg, 'timeline.json'), encoding='utf-8'))
        out = a.out or os.path.join(seg, 'review')
    os.makedirs(out, exist_ok=True)
    for f in glob.glob(os.path.join(out, 'frame_*.jpg')):
        os.remove(f)
    items = frames(video, out, a.every)

    def label(t):
        if tl is None:
            return ''
        for x in tl['lines']:
            if x['start'] <= t <= x['start'] + x['seconds']:
                return x['key'] + '  ' + x['caption']
        return ''
    contact_sheet(items, os.path.join(out, 'contact.jpg'), a.cols, label)
    print(f'{len(items)} frames, contact sheet: {os.path.join(out, "contact.jpg")}  ({duration(video):.1f} s video)')
    if tl is not None:
        timeline_report(tl)


if __name__ == '__main__':
    main()
