"""Record the website's still screenshots (the site/media JPEGs that are not clip posters) from the real application,
with the clips' stage and recorder (demo/clips.py): one 1600 x 900 frame each, dark theme, the 3D views drawn by the
shader renderer through EGL, a throw-away home folder and settings folder, the sample's reference photos never shown.

    python3 demo/stills.py OUT_DIR [--stills window,compare,export,suspension,prefs] [--quality 88]

window     the main window with the sample project, a camera selected
compare    Compare variants: the sample's layouts side by side, the difference map and table
export     the Export dialog previewing the bird's-eye remap tables' header
suspension Analysis ▸ Load cases… on the data rig, its Suspension tab (typical values); loadcases.jpg is the
           poster of the loadcases clip
prefs      Preferences ▸ Assistant (no key entered), zoomed to the dialog
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import time
import traceback

import numpy as np

import clips  # noqa: E402  (offscreen, EGL, the recorder; sets a throw-away settings folder through tour)
from clips import H, W, Clip, Stage  # noqa: E402
from PyQt6.QtCore import QPoint  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402


class Still(Clip):
    """The clips' recorder without a video: tick() composes frames, save() writes the current one."""

    def __init__(self, app, fps=30):
        self.app, self.fps, self.out, self.preview = app, fps, '', False
        self.dur, self.captions = {}, {}
        self.f = 0
        self.win = None
        self.cur = np.array([W * 0.6, H * 0.6])
        self.cursor_on = False
        self.ripples, self.tweens, self.timeline = [], [], []
        self.crop = np.array([0.0, 0.0, W, H])
        self.lower = self.badge = self.fade = self.card = self.backdrop = self.last = None
        self.timelapse = 0
        self.speech_end = 0.0
        self.scene = ''
        self.proc = None

    def _write(self, img):
        pass

    def close(self):
        pass

    def save(self, path, quality):
        img = self.compose()
        if not img.save(path, 'JPG', quality):
            raise OSError(f'could not write {path}')


def _wait(stage, r, cond, timeout=60):
    t = time.time()
    while not cond() and time.time() - t < timeout:
        r.tick()
        time.sleep(0.02)


# ====================================================================== the stills
def window(stage, r):
    w = stage.window()
    r.win = w
    st = w.store
    st.select_cams([st.variant.by_tag('ML').uid])
    stage.settle(4)
    r.hold(0.5)


def compare(stage, r):
    w = stage.window()
    r.win = w
    w.set_page('compare')
    stage.settle(4)
    pg = w.page('compare')
    uids = [v.uid for v in w.store.project.variants]
    _wait(stage, r, lambda: all(pg.results.get(u) is not None for u in uids), 120)    # every variant analysed
    pg.refresh()
    stage.settle(4)
    r.hold(1.0)


def export(stage, r):
    from vantage.ui.dialogs.export_dialog import ExportDialog
    w = stage.window()
    r.win = w
    d = ExportDialog(w.ctx, w)
    stage.tilde(d.out)                           # the default output folder in the working folder, from ~
    d.show()
    r.hold(0.2)
    d.move(w.mapToGlobal(QPoint(max(0, (w.width() - d.width()) // 2), max(0, (w.height() - d.height()) // 2))))
    d.out.setCursorPosition(0)
    d.select('bev_lut')
    r.hold(0.5)
    stage.settle(3)
    r.hold(2.5)
    stage._dialog = d


def suspension(stage, r):
    w = stage._rig_window()
    r.win = w
    d = w.load_cases()
    d.move(w.mapToGlobal(QPoint((w.width() - d.width()) // 2, max(36, (w.height() - d.height()) // 2))))
    d.tabs.setCurrentIndex(2)
    r.hold(1.0)


def prefs(stage, r):
    from vantage.ui.dialogs.misc import PreferencesDialog
    w = stage.window()
    r.win = w
    d = PreferencesDialog(w.ctx, w, tab='assistant')
    d.show()
    r.hold(0.2)
    d.move(w.mapToGlobal(QPoint(max(0, (w.width() - d.width()) // 2), max(0, (w.height() - d.height()) // 2))))
    r.hold(0.5)
    r.zoom(r.wrect(d), 0.05, margin=40)
    r.hold(0.2)
    stage._dialog = d


STILLS = ['window', 'compare', 'export', 'suspension', 'prefs']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--stills', default=','.join(STILLS))
    ap.add_argument('--quality', type=int, default=88)
    ap.add_argument('--theme', default='dark')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    a.out = os.path.abspath(a.out)
    os.environ['HOME'] = tempfile.mkdtemp(prefix='h')
    try:
        from vantage.ui.viewport3d import configure_default_format
        configure_default_format()
        app = QApplication(sys.argv[:1])
        from vantage.ui.theme import THEME, load_fonts
        load_fonts()
        THEME.apply(app, a.theme)
        clips.use_shader_renderer()
        stage = Stage(app, a.out, 30)
        failed = []
        for name in [s for s in a.stills.split(',') if s]:
            r = Still(app)
            stage._dialog = None
            try:
                globals()[name](stage, r)
                path = os.path.join(a.out, f'{name}.jpg')
                r.save(path, a.quality)
                print(f'{name}.jpg {os.path.getsize(path) / 1e3:.0f} kB', flush=True)
            except Exception:
                traceback.print_exc()
                failed.append(name)
            finally:
                if stage._dialog is not None:
                    stage._dialog.reject()
                stage.close_window()
        if failed:
            print('failed:', ', '.join(failed))
            sys.exit(1)
    finally:
        shutil.rmtree(os.environ['HOME'], ignore_errors=True)


if __name__ == '__main__':
    main()
