"""Assemble the tour from its recorded segments: the video (cards, transitions), the mix (voice, music ducked under
it, stings, clicks; loudness -14 LUFS, true peak under -1 dBTP), the subtitles, YouTube chapters, the description,
the thumbnail and a contact sheet. produce.py runs this; it can also run on its own:

    python3 demo/youtube/assemble.py [--only intro,projects,...] [--draft] [--name vantage-tour-draft] [--episode v020]

Reads demo/build/youtube/seg/<id>/ (draft: demo/build/youtube/draft/seg/<id>/); writes the deliverables to
demo/out/youtube/ (draft: demo/build/youtube/draft/out/). Segments that have not been recorded are left out.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np  # noqa: E402

import common  # noqa: E402

SR = 48000
DIP = 0.5               # seconds: a segment dips to the background colour before the next card
FADE_IN = 0.4           # the very first frames come up from black
FADE_OUT = 1.0          # ... and the last go down to black
# music levels (dB against the voice's reference level)
MUSIC_FEATURE = -2.5    # the cold open, the logo: music alone
MUSIC_CARD = -6.0       # chapter and part cards
MUSIC_BETWEEN = -16.0   # between narration lines
MUSIC_UNDER = -22.0     # under the voice
SFX = {'part': -5.0, 'card': -11.0, 'logo': -3.0, 'whoosh': -7.0, 'click': -26.0}
SFX_HIT = {'part': 0.95, 'card': 0.7, 'whoosh': 1.2, 'logo': 0.0, 'click': 0.0}


# ====================================================================== the timeline
def build_plan(only, seg_root, fps):
    import cards
    import chapters as C
    seq = C.sequence(only)
    items, t = [], 0.0
    for e in seq:
        if e['kind'] == 'segment':
            d = os.path.join(seg_root, e['id'])
            if not os.path.exists(os.path.join(d, 'video.mp4')):
                print(f'  (no recording of {e["id"]}: left out)')
                continue
            tl = json.load(open(os.path.join(d, 'timeline.json'), encoding='utf-8'))
            frames = round(tl['frames'] * fps / tl['fps'])
            items.append({'kind': 'segment', 'id': e['id'], 'dir': d, 'tl': tl, 'frames': frames})
        else:
            if e['kind'] == 'chapter-card' and not os.path.exists(os.path.join(seg_root, e['id'], 'video.mp4')):
                continue
            ch_dur, part_dur = cards.durations()
            dur = part_dur if e['kind'] == 'part' else ch_dur
            items.append({'kind': e['kind'], 'entry': e, 'frames': round(dur * fps)})
    # a part card whose chapters were all left out goes too
    items = [it for i, it in enumerate(items)
             if not (it['kind'] == 'part' and (i + 1 >= len(items) or items[i + 1]['kind'] != 'chapter-card'))]
    f = 0
    for it in items:
        it['f0'] = f
        it['start'] = f / fps
        it['dur'] = it['frames'] / fps
        f += it['frames']
    return items, f


# ====================================================================== video
def decode(path, w, h):
    vf = f'scale={w}:{h}:flags=area' if w != common.W else 'null'
    return subprocess.Popen(['ffmpeg', '-v', 'error', '-i', path, '-vf', vf, '-f', 'rawvideo', '-pix_fmt', 'bgra', '-'],
                            stdout=subprocess.PIPE)


def render_video(items, total, path, w, h, fps, seg_root, draft):
    import cv2
    import cards
    import gfx
    cards.app()
    enc = subprocess.Popen(
        ['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgra', '-s', f'{w}x{h}', '-r', str(fps), '-i', '-',
         '-c:v', 'libx264', '-preset', 'veryfast' if draft else 'slow', '-crf', '23' if draft else '18',
         '-profile:v', 'high', '-pix_fmt', 'yuv420p', '-g', str(fps * 2), '-movflags', '+faststart', path],
        stdin=subprocess.PIPE)
    base = gfx.base_np(h, w)
    black = np.zeros_like(base)
    black[..., 3] = 255
    n_out = 0
    t0 = time.time()
    last = None
    for k, it in enumerate(items):
        n = it['frames']
        is_last = k == len(items) - 1
        if it['kind'] == 'segment':
            dec = decode(os.path.join(it['dir'], 'video.mp4'), w, h)
            size = w * h * 4
            dip_n = 0 if is_last else round(DIP * fps)
            for i in range(n):
                raw = dec.stdout.read(size)
                if len(raw) == size:
                    a = np.frombuffer(raw, np.uint8).reshape(h, w, 4)
                    last = a
                else:
                    a = last if last is not None else base           # a frame short: hold the last one
                if dip_n and i >= n - dip_n:
                    a = gfx.mix_np(a, base, gfx.smoother((i - (n - dip_n) + 1) / dip_n))
                if k == 0 and i < FADE_IN * fps:
                    a = gfx.mix_np(black, a, gfx.smoother(i / (FADE_IN * fps)))
                if is_last and i >= n - FADE_OUT * fps:
                    a = gfx.mix_np(a, black, gfx.smoother((i - (n - FADE_OUT * fps)) / (FADE_OUT * fps)))
                enc.stdin.write(np.ascontiguousarray(a).tobytes())
                n_out += 1
            dec.stdout.close()
            dec.wait()
        else:
            card = cards.make(it['entry'], seg_root)
            for i in range(n):
                a = card.frame(i / fps)
                if (w, h) != (common.W, common.H):
                    a = cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA)
                enc.stdin.write(np.ascontiguousarray(a).tobytes())
                n_out += 1
        print(f'  video {it["kind"]:<12} {it.get("id", it.get("entry", {}).get("id", "")):<16} '
              f'{it["start"]:7.1f} s +{it["dur"]:5.1f} s   ({time.time() - t0:.0f} s)', flush=True)
    enc.stdin.close()
    enc.wait()
    return n_out


# ====================================================================== audio
def read_wav(path):
    from scipy.io import wavfile
    from scipy.signal import resample_poly
    sr, a = wavfile.read(path)
    if a.dtype == np.int16:
        a = a.astype(np.float32) / 32768.0
    elif a.dtype == np.int32:
        a = a.astype(np.float32) / 2147483648.0
    else:
        a = a.astype(np.float32)
    if sr != SR:
        g = math.gcd(sr, SR)
        a = resample_poly(a, SR // g, sr // g, axis=0).astype(np.float32)
    return a


def voice_events(items):
    out = []
    for it in items:
        if it['kind'] != 'segment':
            continue
        for x in it['tl']['lines']:
            out.append({'t': it['start'] + x['start'], 'file': common.voice_wav(x['hash']), 'key': x['key'],
                        'caption': x['caption'], 'seconds': x['seconds']})
    return out


def sfx_events(items):
    out = []
    for it in items:
        if it['kind'] == 'part':
            out.append((it['start'] + 0.5 - SFX_HIT['part'], 'part'))
        elif it['kind'] == 'chapter-card':
            out.append((it['start'] + 0.5 - SFX_HIT['card'], 'card'))
        elif it['kind'] == 'segment':
            for e in it['tl'].get('events', []):
                if e['kind'] == 'click':
                    out.append((it['start'] + e['start'], 'click'))
                elif e['kind'] == 'mark' and e.get('name') in ('logo', 'whoosh'):
                    out.append((it['start'] + e['start'] - SFX_HIT[e['name']], e['name']))
                elif e['kind'] == 'mark' and e.get('name') == 'sfx':
                    out.append((it['start'] + e['start'] - SFX_HIT.get(e['sound'], 0.0), e['sound']))
    return out


def music_levels(items, voices, total_s, rate=100):
    """The music's gain (dB) at `rate` Hz: a level per stretch, lower under the voice, eased in and out."""
    n = int(total_s * rate) + 2
    lvl = np.full(n, MUSIC_BETWEEN, np.float32)
    for it in items:
        a, b = int(it['start'] * rate), int((it['start'] + it['dur']) * rate)
        if it['kind'] != 'segment':
            lvl[a:b] = MUSIC_CARD
        else:
            for e in it['tl'].get('events', []):
                if e['kind'] == 'mark' and e.get('name') == 'music':
                    s0 = int((it['start'] + e['start']) * rate)
                    s1 = int((it['start'] + e['start'] + e.get('dur', 1.0)) * rate)
                    lvl[s0:s1] = e.get('db', MUSIC_FEATURE)
    for v in voices:
        a, b = int((v['t'] - 0.30) * rate), int((v['t'] + v['seconds'] + 0.25) * rate)
        lvl[max(0, a):max(0, b)] = np.minimum(lvl[max(0, a):max(0, b)], MUSIC_UNDER)
    # bridge short gaps between lines: no pumping in a 0.4 s pause
    under = lvl <= MUSIC_UNDER + 0.01
    i = 0
    while i < n:
        if not under[i]:
            j = i
            while j < n and not under[j]:
                j += 1
            if 0 < i and j < n and (j - i) < 1.2 * rate:
                lvl[i:j] = MUSIC_UNDER
            i = j
        else:
            i += 1
    # ease: falls fast (0.15 s), rises slowly (0.7 s)
    out = np.empty_like(lvl)
    cur = lvl[0]
    fa, fr = 1 - math.exp(-1 / (0.15 * rate)), 1 - math.exp(-1 / (0.7 * rate))
    for i in range(n):
        cur += (lvl[i] - cur) * (fa if lvl[i] < cur else fr)
        out[i] = cur
    return out


def render_audio(items, total_frames, fps, raw_path):
    """The mix as raw 48 kHz stereo float32, rendered 30 s at a time."""
    music_dir = os.path.join(common.BUILD, 'music')
    import chapters as C
    import music
    music.build(music_dir)
    variant = C.episode().get('music')           # an episode with its own bed (as070); the others: bed.wav
    bed = read_wav(music.build_variant(music_dir, variant) if variant else os.path.join(music_dir, 'bed.wav'))
    L = len(bed)
    sfx = {k: read_wav(os.path.join(music_dir, f'{k}.wav')) for k in SFX}
    total_s = total_frames / fps
    N = int(round(total_s * SR))
    voices = voice_events(items)
    cache = {}
    for v in voices:
        if v['file'] not in cache:
            cache[v['file']] = read_wav(v['file'])
    # one gain for every line: their average RMS at -20 dBFS (the music's reference)
    rms = np.sqrt(np.mean(np.concatenate([a ** 2 for a in cache.values()]))) if cache else 0.1
    vgain = 10 ** (-20 / 20) / max(rms, 1e-6)
    levels = music_levels(items, voices, total_s)
    events = sfx_events(items)
    chunk = 30 * SR
    fade_n = int(1.5 * SR)
    with open(raw_path, 'wb') as fh:
        for c0 in range(0, N, chunk):
            c1 = min(N, c0 + chunk)
            m = c1 - c0
            idx = (np.arange(c0, c1) % L)
            mus = bed[idx]
            t = np.arange(c0, c1) / SR
            g = 10 ** (np.interp(t * 100, np.arange(len(levels)), levels) / 20)
            mus = mus * g[:, None]
            # the music fades in over the first moment and out over the last seconds
            if c0 < int(0.3 * SR):
                k = np.clip(np.arange(c0, c1) / (0.3 * SR), 0, 1)
                mus *= k[:, None]
            tail0 = N - int(3.0 * SR)
            if c1 > tail0:
                k = np.clip((N - np.arange(c0, c1)) / (3.0 * SR), 0, 1)
                mus *= (k ** 1.5)[:, None]
            out = mus.astype(np.float32)
            for v in voices:
                a = cache[v['file']]
                s = int(round(v['t'] * SR))
                e = s + len(a)
                if e <= c0 or s >= c1:
                    continue
                lo, hi = max(s, c0), min(e, c1)
                seg = a[lo - s:hi - s] * vgain
                out[lo - c0:hi - c0] += seg[:, None] if seg.ndim == 1 else seg
            for ts, name in events:
                a = sfx[name]
                s = int(round(ts * SR))
                e = s + len(a)
                if e <= c0 or s >= c1:
                    continue
                lo, hi = max(s, c0, 0), min(e, c1)
                if hi <= lo:
                    continue
                out[lo - c0:hi - c0] += a[lo - s:hi - s] * 10 ** (SFX[name] / 20)
            fh.write(np.ascontiguousarray(out, np.float32).tobytes())
    del fade_n
    return voices


def _loudness_raw(raw_path):
    """Integrated loudness (LUFS) and true peak (dBTP) of a raw 48 kHz stereo float32 file (EBU R128, ffmpeg)."""
    p = subprocess.run(['ffmpeg', '-hide_banner', '-nostats', '-f', 'f32le', '-ar', str(SR), '-ac', '2', '-i', raw_path,
                        '-af', 'ebur128=peak=true', '-f', 'null', '-'], capture_output=True, text=True)
    import re
    s = p.stderr[p.stderr.rfind('Summary:'):]
    i = re.search(r'I:\s+(-?[\d.]+) LUFS', s)
    tp = re.search(r'Peak:\s+(-?[\d.]+) dBFS', s)
    return float(i.group(1)), float(tp.group(1))


def _limit(raw_in, raw_out, gain_db, ceiling_db):
    """Linear gain, then a true-peak limiter: peaks estimated at 4x oversampling, the gain reduction held for 20 ms
    either side of a peak and smoothed with a 20 ms moving average (so it is in place before the peak and lets go
    gently); processed 30 s at a time with overlap."""
    from scipy.ndimage import minimum_filter1d, uniform_filter1d
    from scipy.signal import resample_poly
    g0 = 10 ** (gain_db / 20)
    ceil = 10 ** (ceiling_db / 20)
    n = os.path.getsize(raw_in) // 8
    hold = int(0.020 * SR)
    pad = 4 * hold
    chunk = 30 * SR
    src = np.memmap(raw_in, np.float32, 'r', shape=(n, 2))
    with open(raw_out, 'wb') as fh:
        for c0 in range(0, n, chunk):
            c1 = min(n, c0 + chunk)
            a0, a1 = max(0, c0 - pad), min(n, c1 + pad)
            x = np.array(src[a0:a1], np.float32) * g0
            up = resample_poly(x, 4, 1, axis=0)
            pk = np.abs(up[:4 * len(x)]).reshape(len(x), 4, 2).max(axis=(1, 2))
            g = np.minimum(1.0, ceil / np.maximum(pk, 1e-9))
            g = minimum_filter1d(g, size=2 * hold + 1)
            g = uniform_filter1d(g, size=hold)
            y = x * g[:, None]
            fh.write(np.ascontiguousarray(y[c0 - a0:c1 - a0], np.float32).tobytes())
    del src


def loudnorm(raw_path, wav_path, target=-14.0, ceiling=-1.6):
    """-14 LUFS integrated, true peak under the ceiling: a linear gain and the limiter, measured again and corrected
    once (the limiter takes a little loudness off the peaks)."""
    i0, tp0 = _loudness_raw(raw_path)
    tmp = raw_path + '.lim'
    gain = target - i0
    for _ in range(2):
        _limit(raw_path, tmp, gain, ceiling)
        i1, tp1 = _loudness_raw(tmp)
        if abs(i1 - target) < 0.15:
            break
        gain += target - i1
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-f', 'f32le', '-ar', str(SR), '-ac', '2', '-i', tmp, '-c:a',
                    'pcm_f32le', wav_path], check=True)
    os.remove(tmp)
    return {'input_i': i0, 'input_tp': tp0, 'input_lra': None}, {'output_i': i1, 'output_tp': tp1,
                                                                    'normalization_type': f'linear {gain:+.2f} dB + limiter'}


def measure(path):
    p = subprocess.run(['ffmpeg', '-hide_banner', '-nostats', '-i', path, '-map', '0:a', '-af', 'ebur128=peak=true',
                        '-f', 'null', '-'], capture_output=True, text=True)
    s = p.stderr[p.stderr.rfind('Summary:'):]
    import re
    i = re.search(r'I:\s+(-?[\d.]+) LUFS', s)
    tp = re.search(r'Peak:\s+(-?[\d.]+) dBFS', s)
    lra = re.search(r'LRA:\s+(-?[\d.]+) LU', s)
    return {'I': float(i.group(1)) if i else None, 'TP': float(tp.group(1)) if tp else None,
            'LRA': float(lra.group(1)) if lra else None}


# ====================================================================== texts
def hms(t, srt=False):
    if srt:
        from mix import ts
        return ts(t)
    t = int(t)
    h, r = divmod(t, 3600)
    m, s = divmod(r, 60)
    return f'{h}:{m:02d}:{s:02d}' if h else f'{m:02d}:{s:02d}'


def write_srt(voices, total_s, path):
    from mix import chunks, wrap2
    cues = []
    for v in voices:
        parts = chunks(v['caption'])
        n = sum(len(p) for p in parts)
        t = v['t']
        for p in parts:
            d = v['seconds'] * len(p) / n
            cues.append((t, t + d, wrap2(p)))
            t += d
    with open(path, 'w', encoding='utf-8') as fh:
        for i, (s, e, text) in enumerate(cues, 1):
            nxt = cues[i][0] - 0.02 if i < len(cues) else total_s
            fh.write(f'{i}\n{hms(s, True)} --> {hms(min(e + 0.15, nxt), True)}\n{text}\n\n')
    return len(cues)


def chapter_marks(items, ep=None):
    """[(seconds, label)]: 00:00 the intro, each chapter from its part card or chapter card, the outro (the
    episode's intro and outro and their labels)."""
    import chapters as C
    ep = ep or C.episode()
    marks = []
    pending_part = None
    for it in items:
        if it['kind'] == 'segment' and it['id'] == ep['intro']:
            marks.append((0.0, ep['marks'][0]))
        elif it['kind'] == 'part':
            pending_part = it['start']
        elif it['kind'] == 'chapter-card':
            e = it['entry']
            t = pending_part if pending_part is not None else it['start']
            pending_part = None
            marks.append((t, f'{e["nn"]:02d} {e["title"]}'))
        elif it['kind'] == 'segment' and it['id'] == ep['outro']:
            marks.append((it['start'], ep['marks'][1]))
    if marks and marks[0][0] > 0:
        marks[0] = (0.0, marks[0][1])
    # YouTube: every chapter at least 10 s
    clean = []
    for i, (t, lab) in enumerate(marks):
        if clean and t - clean[-1][0] < 10:
            continue
        clean.append((t, lab))
    return clean


def write_chapters(marks, path):
    with open(path, 'w', encoding='utf-8') as fh:
        for t, lab in marks:
            fh.write(f'{hms(t)} {lab}\n')


DESCRIPTION = """# {title}

## Description

Vantage designs the sensor rigs of data-collection vehicles: the cars that record the camera data self-driving models
are trained on. Import a 3D model of the vehicle, place the cameras, lidars, radars, GNSS antennas and IMUs on it, and
prove in simulation what the rig will record before any hardware is fitted: what each camera sees through its real
lens model, where the body, the rig and the other sensors' housings block it, how many pixels land on the ground,
where cameras overlap, how far the rig resolves cars, people and signs, and how sensitive it all is to mounting errors
and load. Then calibrate the real cameras, check the rig as built against the plan from a recorded drive, and hand
the calibration to the data pipeline (OpenCV, ROS 2, Kalibr, nuScenes, KITTI and more).

This tour shows every feature, recorded from the app itself (v{version}) on its Tata Tigor EV sample project.

Download (Windows, macOS, Linux x86-64 and arm64 / Jetson): https://github.com/Sherin-SEF-AI/Vantage/releases
Website and documentation: https://sherin-sef-ai.github.io/Vantage/
Source code: https://github.com/Sherin-SEF-AI/Vantage
Licence: GPL-3.0-or-later. Free and open source.

Notes: every figure on screen is the app's own output on the sample project; time-lapses are marked with their
speed-up; the assistant chapter's model replies are scripted for the video (its tools and figures are computed live).
Narration: Kokoro-82M (voice af_heart), rendered locally. Music: synthesised for this video.

## Chapters

{chapters}

## Tags

{tags}
"""

DESCRIPTIONS = {}         # episode -> its description template (the tour's is DESCRIPTION)
DESCRIPTIONS['v020'] = """# {title}

## Description

Vantage {version} adds seven studies for a data-collection rig on the road and in the workshop, and makes its
calibration more accurate. This video explains each one on the app's Tata Tigor EV sample: the problem it solves on a
data-collection vehicle, every control on its page, how to read each result, a worked example, its limits, and its
batch-mode and export hooks.

What is new in {version}:
- Critical targets and mounting envelopes (Analysis ▸ Critical targets): what the rig must resolve, by which sensor,
  and how far each camera may move before a target is lost.
- Road and body motion (Analysis ▸ Road and body motion): brake dive, squat and roll; ramps, crests, speed bumps and
  kerbs in the coverage analysis.
- Weather and night (Analysis ▸ Weather and night): overcast, night, rain, fog and snow in the perception range study
  and the rendered sample dataset.
- Materials, glass and wipers (Analysis ▸ Materials, glass and wipers): radar loss through fascia, paint and grilles;
  refraction, glare and the wiper sweep for cameras behind glass.
- Cable harness (Analysis ▸ Cable harness): every cable routed along the body against its link's length limit.
- Calibration observability (Calibrate ▸ Calibration observability): how firmly the planned boards pin each camera,
  the weak axis, and the board moves that help most.
- A more accurate calibration: the installation check's joint solve now converges every time and matches its own
  uncertainty; the lidar-camera calibration's precision is honest and its bias is gone.
- Automation and CI (Export ▸ Automation): calibration files regenerated on save, your own approved test command,
  --watch, and GitHub Actions / GitLab CI templates.

Download {version} (Windows, macOS, Linux x86-64 and arm64 / Jetson): https://github.com/Sherin-SEF-AI/Vantage/releases/tag/v{version}
All releases: https://github.com/Sherin-SEF-AI/Vantage/releases
Website and documentation: https://sherin-sef-ai.github.io/Vantage/
Source code: https://github.com/Sherin-SEF-AI/Vantage
Licence: GPL-3.0-or-later. Free and open source.
A full tour of every earlier feature is a separate video.

Notes: recorded from the app itself (v{version}) on its sample project; every figure on screen is the app's own output
on the sample, and nominal or typical values are labelled as such; time-lapses are marked with their speed-up; the
calibration chapter's before/after figures were measured for this video by running 0.1.1's and 0.2.0's solves on the
same simulated captures. Narration: Kokoro-82M (voice af_heart), rendered locally. Music: synthesised for this video.

## Chapters

{chapters}

## Tags

{tags}
"""

DESCRIPTIONS['gt040'] = """# {title}

## Description

Vantage {version} labels what a sensor rig records in the CARLA simulator: every camera frame gets a semantic, an
instance and a range image and its objects (tight and full 2D boxes, 3D boxes, occlusion, truncation, COCO), and every
lidar sweep its nuScenes-lidarseg and panoptic labels. This video shows them on real CARLA 0.9.16 drives of the app's
sample data rig (six fisheye cameras and a 128-beam roof lidar), frame by frame at real speed: how each label is made
through the fisheye's own lens model, what it is worth against the lidar, the built-in scenarios that replay the same
way for the same seed, and the drives opened in the app's Ground truth tab and read back in batch mode.

In this video:
- The rig in CARLA: each fisheye rendered as five pinhole views and joined through its lens model's remap; an instance
  and a depth image beside every view, read at the nearest view pixel, so labels are never blurred.
- Four layers on one camera: tight boxes from each object's own pixels and full boxes from its 3D box; CARLA's 29
  semantic classes plus the ego; one colour per object for the whole drive; the range along each pixel's ray.
- Every lidar point labelled: a semantic lidar on the same rays as the recorded one, checked on every step.
- Scenarios: a pedestrian crossing, a cyclist crossing and a cut-in, each with its events in run.json.
- In the app: Simulate ▸ CARLA ▸ Open run…, the Ground truth tab and its object table, the nuScenes and COCO files,
  and --ground-truth-summary.

Download {version} (Windows, macOS, Linux x86-64 and arm64 / Jetson): https://github.com/Sherin-SEF-AI/Vantage/releases/tag/v{version}
Website and documentation (Simulate in CARLA ▸ Ground truth, faults and scenarios): https://sherin-sef-ai.github.io/Vantage/
Source code: https://github.com/Sherin-SEF-AI/Vantage
Licence: GPL-3.0-or-later. Free and open source.

Notes: the footage is real CARLA 0.9.16 drives recorded by Vantage {version} (Town10HD, ClearNoon, seed 1), shown at
their real speed and held where marked "paused"; every figure narrated is read from the drives' own files, except the
range and lidar-box checks, which were measured on the same drive without walkers and are said so. The last chapter is
recorded from the app itself. CARLA is a separate open-source simulator; it is not part of Vantage. Narration:
Kokoro-82M (voice af_heart), rendered locally. Music: synthesised for this video.

## Chapters

{chapters}

## Tags

{tags}
"""

DESCRIPTIONS['as070'] = """# {title}

## Description

Vantage designs the sensor rigs of data-collection vehicles: the cars that record the camera, lidar and radar data
self-driving models are trained on. Its assistant is a chat about the open design. It reads the design only through
Vantage's own tools (coverage, blind ground, pixel density, the studies), so every figure it quotes is the app's own;
it simulates a change on a copy before anyone touches the design, and it proposes changes on a card that nothing
applies until you accept.

Every reply in this video is the real model's, GPT-OSS 120B on Groq, unedited, on the app's built-in Tata Tigor EV
sample (its data rig: six cameras, a roof lidar, five radars). Every figure on screen is read from the app while
recording.

In this video:
- Ask the design: where the rig is blind and why, from find_blind_regions; every lookup listed and open to inspect;
  consent before the first question, and what is (and is never) sent.
- What-ifs and proposals: a camera move simulated on a copy, then proposed on a card with its predicted effect;
  Apply as one undo step, the measured result beside the prediction, and Undo.
- Recommendations as you work: the live strip follows the selection and your last edit, worked out locally and
  instantly; nothing is sent until you click Ask.
- Whole-rig design: optimise_rig searches the whole layout against your targets (camera count, minimum density, blind
  ground) within a time budget; add_camera_options finds where one more camera would close a gap.
- The design log: a discarded proposal is recorded as rejected, the model reads the log and does not offer it again.
- The Copilot: a CARLA drive asked for in words becomes a plan card, every value with its source; it stays a draft
  until you approve it (nothing is run in this video).
- Automatic model choice and pictures: quick questions to a fast model, goals, plans and picture questions to Claude
  Opus; without a Claude key the panel falls back to GPT-OSS 120B and says so. The coverage picture the look tool
  draws is shown; with Groq nothing of it is sent.
- The benchmark: python -m vantage.assistant.evals asks 51 design questions and grades the answers against the app's
  own numbers, without a judge model. The full run on GPT-OSS 120B shown here scored 89 % over 50 scored questions.

Website and documentation: https://sherin-sef-ai.github.io/Vantage/
Download (Windows, macOS, Linux x86-64 and arm64 / Jetson): https://github.com/Sherin-SEF-AI/Vantage/releases
Source code: https://github.com/Sherin-SEF-AI/Vantage
Licence: GPL-3.0-or-later. Free and open source.

Notes: recorded from the app itself on its sample project. The assistant's replies come from GPT-OSS 120B on Groq with
the author's own key and are shown unedited; waits for the model and the whole-rig search are shown in real time or as
marked time-lapses. For chapter 4 the quarter cameras were tilted 12 degrees up, as said on screen. The benchmark run
was made before the recording and its output is replayed. The assistant needs your own Groq or Claude API key; it is
off until you add one. Narration: Kokoro-82M (voice af_heart), rendered locally. Music: synthesised for this video.

## Chapters

{chapters}

## Tags

{tags}
"""

TAGS = ['autonomous driving', 'self-driving', 'sensor rig', 'data collection', 'camera placement', 'lidar', 'radar',
        'sensor calibration', 'camera calibration', 'surround view', 'fisheye camera', 'GMSL', 'Jetson AGX Orin',
        'e-con Systems', 'ROS 2', 'nuScenes', 'KITTI', 'Kalibr', 'OpenCV', 'open source', 'PyQt', 'Vantage']
TAGS_BY = {'v020': ['Vantage 0.2.0', 'what is new', 'autonomous driving', 'data collection', 'sensor rig',
                    'sensor calibration', 'lidar camera calibration', 'calibration observability', 'radar fascia',
                    'cable harness', 'GMSL2', 'weather simulation', 'fog', 'night driving', 'speed bump',
                    'mounting tolerance', 'CI', 'GitHub Actions', 'GitLab CI', 'open source', 'Vantage'],
           'gt040': ['Vantage 0.4.0', 'CARLA', 'CARLA simulator', 'ground truth', 'auto-labelling', 'synthetic data',
                     'semantic segmentation', 'instance segmentation', 'depth', 'lidar segmentation',
                     'nuScenes-lidarseg', 'panoptic segmentation', 'COCO', 'fisheye camera', 'sensor rig',
                     'data collection', 'autonomous driving', 'scenarios', 'open source', 'Vantage'],
           'as070': ['AI assistant', 'LLM agent', 'tool use', 'GPT-OSS 120B', 'Groq', 'Copilot', 'sensor rig',
                     'camera placement', 'autonomous driving', 'data collection', 'design optimisation',
                     'lidar', 'radar', 'CARLA', 'LLM benchmark', 'human in the loop', 'open source', 'Vantage']}


def write_description(marks, path, title, ep=None):
    import chapters as C
    ep = ep or C.episode()
    tmpl = DESCRIPTIONS[ep['description']] if ep.get('description') else DESCRIPTION
    tags = TAGS_BY[ep['tags']] if ep.get('tags') else TAGS
    chapters = '\n'.join(f'{hms(t)} {lab}' for t, lab in marks)
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(tmpl.format(title=title, version=ep['version'], chapters=chapters, tags=', '.join(tags)))


def make_thumbnail(seg_root, path, ep=None):
    """1280 x 720: a frame of the cold open (the episode's intro still 'thumb') with the title, under 2 MB."""
    import cards
    import chapters as C
    import gfx
    from PyQt6.QtCore import QRectF
    from PyQt6.QtGui import QImage, QLinearGradient, QBrush
    cards.app()
    th = (ep or C.episode())['thumb']
    src = os.path.join(seg_root, th['still'], 'stills', 'thumb.png')
    if th.get('maker') == 'as070':               # the assistant episode draws its own (gfx_as.thumbnail)
        import gfx_as
        return gfx_as.thumbnail(QImage(src) if os.path.exists(src) else None, path)
    shot = QImage(src) if os.path.exists(src) else gfx.from_np(gfx.base_np())
    img = gfx.from_np(gfx.base_np())
    p = gfx.painter(img)
    p.drawImage(QRectF(th.get('shot_x', 400), 0, 1920, 1080), shot.scaled(1920, 1080))     # the car in the right half
    g = QLinearGradient(0, 0, 1920, 0)
    x0 = 0.22 if 'shot_x' not in th else (th['shot_x'] + 60) / 1920    # past the shifted shot's left edge (x = 400)
    g.setColorAt(0.0, gfx.col('#0b0b0a', 1.0))
    g.setColorAt(x0, gfx.col('#0b0b0a', 1.0))
    g.setColorAt(x0 + 0.18, gfx.col('#0b0b0a', 0.78))
    g.setColorAt(x0 + 0.42, gfx.col('#0b0b0a', 0.0))
    p.fillRect(QRectF(0, 0, 1920, 1080), QBrush(g))
    p.drawImage(QRectF(120, 150, 150, 150), gfx.icon_image())
    gfx.draw_text(p, 120, 400, th['eyebrow'], gfx.font(46, 'semibold', mono=True, spacing=6), gfx.ACCENT)
    gfx.draw_text(p, 110, 610, th['word'], gfx.font(th['word_px'], 'semibold'), gfx.INK)
    p.fillRect(QRectF(124, 650, 220, 6), gfx.col(gfx.ACCENT))
    gfx.draw_text(p, 120, 760, th['lines'][0], gfx.font(76, 'semibold'), gfx.INK)
    gfx.draw_text(p, 120, 850, th['lines'][1], gfx.font(76, 'semibold'), gfx.INK2)
    gfx.draw_text(p, 122, 960, th['foot'], gfx.font(34, 'medium', mono=True, spacing=4), gfx.MUTED)
    p.end()
    small = img.scaled(1280, 720, transformMode=__import__('PyQt6.QtCore', fromlist=['Qt']).Qt.TransformationMode.SmoothTransformation)
    small.save(path, 'PNG')
    if os.path.getsize(path) > 2_000_000:
        small.save(os.path.splitext(path)[0] + '.jpg', 'JPG', 92)
        os.replace(os.path.splitext(path)[0] + '.jpg', path.replace('.png', '.jpg'))
    return path


def contact_sheets(video, out_dir, total_s, name):
    import review
    every = 5.0 if total_s < 900 else 10.0
    tmp = os.path.join(out_dir, f'{name}-frames')
    os.makedirs(tmp, exist_ok=True)
    items = review.frames(video, tmp, every, width=480)
    per = 60
    paths = []
    for k in range(0, len(items), per):
        pth = os.path.join(out_dir, f'{name}-contact-{k // per + 1:02d}.jpg')
        review.contact_sheet(items[k:k + per], pth, 6)
        paths.append(pth)
    for _, f in items:
        os.remove(f)
    os.rmdir(tmp)
    return paths


# ====================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default='')
    ap.add_argument('--draft', action='store_true')
    ap.add_argument('--name')
    ap.add_argument('--episode')
    ap.add_argument('--remux', action='store_true',
                    help='keep the video of the deliverable already assembled (same pieces) and redo the sound, texts '
                         'and sheets')
    a = ap.parse_args()
    only = [x for x in a.only.split(',') if x] or None
    import chapters as C
    os.environ['VANTAGE_EPISODE'] = C.resolve(a.episode, only)
    ep = C.episode()
    seg_root = os.path.join(common.DRAFT_DIR if a.draft else common.BUILD, 'seg')
    out_dir = os.path.join(common.DRAFT_DIR, 'out') if a.draft else common.OUT
    os.makedirs(out_dir, exist_ok=True)
    fps = 15 if a.draft else common.FPS
    w, h = (960, 540) if a.draft else (common.W, common.H)
    complete = all(cid in C.files() for _, cid, _, _ in ep['plan']) and not only
    name = a.name or (ep['name'] if complete else ep['name'] + '-draft')
    t0 = time.time()
    items, total = build_plan(only, seg_root, fps)
    if not items:
        raise SystemExit('nothing recorded yet')
    total_s = total / fps
    print(f'{name}: {len(items)} pieces, {total_s:.1f} s ({total} frames at {fps} fps)', flush=True)
    work = os.path.join(common.DRAFT_DIR if a.draft else common.BUILD, 'assemble')
    os.makedirs(work, exist_ok=True)
    vid = os.path.join(work, f'{name}.video.mp4')
    final = os.path.join(out_dir, f'{name}.mp4')
    if a.remux:
        # the video stream of the deliverable already made from these pieces (frames and durations unchanged)
        subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', final, '-map', '0:v', '-c', 'copy', vid], check=True)
        n = total
    else:
        n = render_video(items, total, vid, w, h, fps, seg_root, a.draft)
    raw = os.path.join(work, f'{name}.mix.f32')
    voices = render_audio(items, total, fps, raw)
    wav = os.path.join(work, f'{name}.mix.wav')
    # ffmpeg's AAC encoder can overshoot a sharp transient by several dB (seen: a 0.83 peak encoded as 1.56); the mix
    # has next to nothing above 16 kHz, so a 17 kHz low-pass first. If the file still measures over -1 dBTP, the mix is
    # limited again to a lower true-peak ceiling (the loudness stays at -14 LUFS); last, the other coder, then half a
    # dB less
    tries = [(-1.6, ['-af', 'lowpass=f=17000']), (-2.4, ['-af', 'lowpass=f=17000']), (-3.2, ['-af', 'lowpass=f=17000']),
             (-3.2, ['-aac_coder', 'fast']), (-3.2, ['-aac_coder', 'fast', '-af', 'volume=-0.5dB'])]
    done_ceiling = None
    for ceiling, extra in tries:
        if ceiling != done_ceiling:
            m1, m2 = loudnorm(raw, wav, ceiling=ceiling)
            done_ceiling = ceiling
        subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', vid, '-i', wav, '-map', '0:v', '-map', '1:a', '-c:v', 'copy',
                        '-c:a', 'aac', '-b:a', '192k', '-ar', str(SR), '-ac', '2'] + extra +
                       ['-metadata', f'title={ep["video_title"]}', '-metadata:s:a:0', 'language=eng',
                        '-movflags', '+faststart', '-shortest', final], check=True)
        loud = measure(final)
        loud['aac'] = ' '.join(extra) + f' (mix limited to {ceiling} dBTP)'
        if loud['TP'] is not None and loud['TP'] <= -1.0:
            break
        print(f'  true peak {loud["TP"]} dBTP after AAC with {extra}, the mix limited to {ceiling} dBTP: trying again',
              flush=True)
    os.remove(raw)
    srt = os.path.join(out_dir, f'{name}.srt')
    ncues = write_srt(voices, total_s, srt)
    marks = chapter_marks(items, ep)
    f_ch, f_desc, f_th = (x.format(name=name) for x in ep['texts'])
    write_chapters(marks, os.path.join(out_dir, f_ch))
    write_description(marks, os.path.join(out_dir, f_desc), ep['title'], ep)
    thumb = make_thumbnail(seg_root, os.path.join(out_dir, f_th), ep)
    sheets = contact_sheets(final, out_dir, total_s, name)
    report = {'name': name, 'seconds': round(total_s, 2), 'frames': n, 'fps': fps, 'size': [w, h],
              'loudness_final': loud, 'loudnorm_pass1': {k: m1[k] for k in ('input_i', 'input_tp', 'input_lra')},
              'loudnorm_pass2': {k: m2.get(k) for k in ('output_i', 'output_tp', 'normalization_type')},
              'cues': ncues, 'chapters': marks, 'pieces': [(it['kind'], it.get('id') or it['entry'].get('id', it['entry'].get('part')),
                                                           round(it['start'], 2), round(it['dur'], 2)) for it in items],
              'wall_seconds': round(time.time() - t0, 1), 'thumbnail': thumb, 'contact_sheets': sheets}
    with open(os.path.join(out_dir, f'{name}.report.json'), 'w', encoding='utf-8') as fh:
        json.dump(report, fh, indent=1, ensure_ascii=False)
    os.remove(wav)
    os.remove(vid)
    print(f'{final}\n  {total_s:.1f} s, loudness I {loud["I"]} LUFS, true peak {loud["TP"]} dBTP, LRA {loud["LRA"]} LU\n'
          f'  {ncues} subtitle cues, {len(marks)} chapters, thumbnail {thumb}\n  contact sheets: {", ".join(sheets)}\n'
          f'  assembled in {time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    main()
