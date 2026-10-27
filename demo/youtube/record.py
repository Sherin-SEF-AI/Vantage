"""Record one segment of the tour (a chapter, the intro or the outro) in this process.

    python3 demo/youtube/record.py CHAPTER_ID [--draft] [--out DIR]

Writes DIR/video.mp4 (1920 x 1080, 30 fps, near-lossless H.264; --draft: 960 x 540, 15 fps, 1x pixels), timeline.json
(the narration lines with their start times, clicks, waits, marks), stills/ and meta.json (what it was recorded from,
so produce.py knows when to re-record). DIR defaults to demo/build/youtube/seg/<id> (draft: .../draft/seg/<id>). The
segment is written to DIR.tmp and moved into place only when the recording succeeded.

Run it under the memory cap, offscreen, with a throw-away settings folder (see AUTHORING.md); it uses a throw-away
home folder of its own and removes it at the end.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common  # noqa: E402


def seg_dir(cid, draft=False):
    return os.path.join(common.DRAFT_DIR if draft else common.BUILD, 'seg', cid)


def segment_key(cid, draft=False, style=None):
    """What a segment depends on: its script, its voice lines, the recorder's style version (style: the version of
    the studio module a recording actually ran with; else read from studio.py as it is on disk now)."""
    import chapters as C
    from common import line_hash
    lines = C.lines(cid)
    if style is None:
        style = 0
        try:
            src = open(os.path.join(HERE, 'studio.py'), encoding='utf-8').read()
            style = int(src.split('STYLE_VERSION = ', 1)[1].split()[0])
        except (OSError, ValueError, IndexError):
            pass
    return {'source': C.source_hash(cid), 'voice': [line_hash(s) for _, s in lines], 'style': style, 'draft': draft}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('chapter')
    ap.add_argument('--draft', action='store_true')
    ap.add_argument('--out')
    a = ap.parse_args()
    if not a.draft:
        os.environ['QT_SCALE_FACTOR'] = '2'          # grab at 3840 x 2160: crisp frames and crisp 2x zooms
    else:
        os.environ.pop('QT_SCALE_FACTOR', None)
    os.environ['HOME'] = tempfile.mkdtemp(prefix='h')   # throw-away home: what the scene writes lands here
    out = os.path.abspath(a.out or seg_dir(a.chapter, a.draft))
    tmp = out + '.tmp'
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    code = 1
    try:
        code = record(a.chapter, tmp, a.draft)
        if code == 0:
            shutil.rmtree(out, ignore_errors=True)
            os.replace(tmp, out)
            print(f'segment: {out}', flush=True)
    except Exception:
        traceback.print_exc()
    finally:
        shutil.rmtree(os.environ['HOME'], ignore_errors=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


def record(cid, out, draft):
    import chapters as C
    import studio
    from PyQt6.QtWidgets import QApplication
    t0 = time.time()
    # the key at the start: the script, voice and style this recording is made with (a file changed while it records
    # must not be credited to it)
    key = segment_key(cid, draft, style=studio.STYLE_VERSION)
    meta = C.meta(cid)
    lines = C.lines(cid)
    vlines = common.chapter_voice(cid, lines)
    mod = C.load(cid)
    from vantage.ui.viewport3d import configure_default_format
    configure_default_format()
    app = QApplication(sys.argv[:1])
    app.setApplicationName('Vantage')
    app.setOrganizationName('Vantage')
    from vantage.ui.theme import THEME, load_fonts
    load_fonts()
    THEME.apply(app, 'dark')
    studio.use_shader_renderer()
    fps = 15 if draft else common.FPS
    stage = studio.Stage(app, out, fps)
    ep_plan = C.episode(C.episode_of(cid))['plan']        # the chapter's own episode: its parts on the indicator
    plan = ep_plan if any(c[1] == cid for c in ep_plan) else None
    r = studio.Rec(app, out, cid, vlines, fps=fps, out_size=(960, 540) if draft else (common.W, common.H),
                   crf=20 if draft else 12, preset='veryfast' if draft else 'medium', plan=plan)
    r.meta = meta
    stage.rec = r
    try:
        mod.scene(stage, r)
    finally:
        info = r.close(meta)
        stage.close_window()
    with open(os.path.join(out, 'meta.json'), 'w', encoding='utf-8') as fh:
        json.dump({**key, 'id': cid, 'seconds': info['seconds'], 'frames': info['frames'], 'fps': fps,
                   'wall_seconds': round(time.time() - t0, 1), 'recorded': time.strftime('%Y-%m-%d %H:%M')}, fh,
                  indent=1)
    print(f'{cid}: {info["seconds"]:.1f} s of video in {time.time() - t0:.0f} s', flush=True)
    return 0


if __name__ == '__main__':
    main()
