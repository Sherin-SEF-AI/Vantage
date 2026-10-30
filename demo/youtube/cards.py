"""The chapter and part cards, painted frame by frame (no app needed) for assemble.py, and previewed on their own.

    python3 demo/youtube/cards.py preview coverage [--out DIR]      the chapter card of 09 (and its part card): an MP4
                                                                    and key frames as PNG

A chapter card fades in from the theme's background over a blurred, darkened copy of its chapter's first frame,
sets its type, and leaves by sharpening into that frame, so the cut to the recording is invisible. A part card sits on
the plain background and leaves by fading its type, so the chapter card that follows comes in from the same colour.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np  # noqa: E402

import common  # noqa: E402

CHAPTER_DUR = 3.6           # seconds
CHAPTER_EXIT = 0.8          # the last seconds: type out, frame sharpens in
PART_DUR = 4.8
PART_EXIT = 0.6


_APP = []


def app():
    """A QGuiApplication with the app's fonts (kept alive here: Qt needs it for every painter)."""
    from PyQt6.QtGui import QGuiApplication
    a = QGuiApplication.instance()
    if a is None:
        a = QGuiApplication(sys.argv[:1])
        if common.ROOT not in sys.path:
            sys.path.insert(0, common.ROOT)
        from vantage.ui.theme import load_fonts
        load_fonts()
    _APP.append(a)
    return a


def first_frame(video: str) -> np.ndarray | None:
    """The first frame of a segment as BGRA (1920 x 1080)."""
    if not os.path.exists(video):
        return None
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', video, '-frames:v', '1', '-vf',
                          f'scale={common.W}:{common.H}:flags=area', '-f', 'rawvideo', '-pix_fmt', 'bgra', '-'],
                         capture_output=True).stdout
    if len(raw) < common.W * common.H * 4:
        return None
    return np.frombuffer(raw, np.uint8)[:common.W * common.H * 4].reshape(common.H, common.W, 4).copy()


class ChapterCard:
    def __init__(self, entry, frame: np.ndarray | None, dur=CHAPTER_DUR):
        import gfx
        self.e, self.dur = entry, dur
        self.bd = gfx.Backdrop(frame)

    def frame(self, t) -> np.ndarray:
        import gfx
        e = self.e
        exit0 = self.dur - CHAPTER_EXIT
        if t < exit0:
            k = gfx.smoother(t / 0.6)
            bg = self.bd.at(0.0, 1 - 0.30 * k)                  # from the background colour to a faint, blurred frame
        else:
            k = gfx.smoother((t - exit0) / CHAPTER_EXIT)
            bg = self.bd.at(k, 0.70 * (1 - k))                  # sharpen into the chapter's first frame
        img = gfx.from_np(bg)
        if t < exit0 + CHAPTER_EXIT * 0.75:
            p = gfx.painter(img)
            gfx.vignette(p, 0.45 * (1 - gfx.smoother((t - exit0) / (CHAPTER_EXIT * 0.75))))
            p.end()
            gfx.chapter_card(img, t, e['nn'], e['part'], e['part_title'], e['title'], e['promise'], e['menu'],
                             self.dur, CHAPTER_EXIT)
        return gfx.to_np(img)


class PartCard:
    def __init__(self, entry, dur=PART_DUR):
        self.e, self.dur = entry, dur

    def frame(self, t) -> np.ndarray:
        import gfx
        img = gfx.from_np(gfx.base_np())
        p = gfx.painter(img)
        gfx.vignette(p, 0.35)
        p.end()
        e = self.e
        gfx.part_card(img, t, e['part'], e['title'], e['blurb'], e['items'], self.dur, PART_EXIT, e.get('of', 5))
        return gfx.to_np(img)


def durations(ep=None) -> tuple[float, float]:
    """(chapter card, part card) seconds for an episode: its own 'cards' durations, else CHAPTER_DUR and PART_DUR
    (every episode but as070 uses the defaults)."""
    import chapters as C
    d = (ep or C.episode()).get('cards') or {}
    return float(d.get('chapter', CHAPTER_DUR)), float(d.get('part', PART_DUR))


def make(entry, seg_root=None):
    """The card object for a sequence entry ('chapter-card' or 'part')."""
    ch, pt = durations()
    if entry['kind'] == 'part':
        return PartCard(entry, pt)
    seg_root = seg_root or common.SEG_DIR
    return ChapterCard(entry, first_frame(os.path.join(seg_root, entry['id'], 'video.mp4')), ch)


def preview(cid, out):
    import chapters as C
    import gfx
    app()
    seq = C.sequence(name=C.episode_of(cid))          # the chapter's own episode
    i = next(k for k, e in enumerate(seq) if e['kind'] == 'chapter-card' and e['id'] == cid)
    entries = [seq[i]]
    if i > 0 and seq[i - 1]['kind'] == 'part':
        entries.insert(0, seq[i - 1])
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f'card_{cid}.mp4')
    proc = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgra', '-s',
                             f'{common.W}x{common.H}', '-r', str(common.FPS), '-i', '-', '-c:v', 'libx264', '-crf', '18',
                             '-pix_fmt', 'yuv420p', path], stdin=subprocess.PIPE)
    for e in entries:
        c = make(e)
        n = round(c.dur * common.FPS)
        for f in range(n):
            a = c.frame(f / common.FPS)
            proc.stdin.write(a.tobytes())
            if f in (round(n * 0.12), round(n * 0.5), n - round(0.5 * common.FPS), n - 1):
                gfx.from_np(a).save(os.path.join(out, f'card_{e.get("id", "part")}_{f:03d}.png'))
    proc.stdin.close()
    proc.wait()
    print(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['preview'])
    ap.add_argument('chapter')
    ap.add_argument('--out', default=os.path.join(common.BUILD, 'cards'))
    a = ap.parse_args()
    preview(a.chapter, a.out)


if __name__ == '__main__':
    main()
