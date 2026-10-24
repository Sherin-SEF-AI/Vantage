"""Record the short feature clips for the website (site/media).

Drives the real application offscreen, one window per clip, with its 3D views drawn by the shader renderer through a
headless EGL context (what a desktop with OpenGL shows; Qt's offscreen platform has no OpenGL of its own). Each clip
is a silent 1600 x 900 video plus a poster frame. The sample's reference photos are never shown (third-party images).

    python3 demo/clips.py OUT_DIR [--clips hero,coverage,...] [--fps 30]

Needs EGL OpenGL (Mesa) and ffmpeg. The assistant clip talks to tests/fake_groq.py: its model replies are scripted
for the recording; the tools, figures and proposal are computed live by Vantage.

The recording runs with a throw-away home folder (removed at the end): recordings and datasets the clips write go
there, the working folder (~/Documents/Vantage) too, and paths the app shows in full are shown from ~ (see tilde()),
so no real folder of the machine that records appears in a clip.
"""
from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, HERE, os.path.join(ROOT, 'tests')]
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['VANTAGE_NO_KEYRING'] = '1'
os.environ.pop('GROQ_API_KEY', None)

import numpy as np  # noqa: E402
import tour  # noqa: E402  (sets a throw-away XDG_CONFIG_HOME; the recorder and its pointer, menus and zooms)
from tour import Recorder, ease  # noqa: E402

W, H = 1600, 900
tour.W, tour.H = W, H

from PyQt6.QtCore import QPoint, QPointF, QRect, QRectF, Qt  # noqa: E402
from PyQt6.QtGui import QColor, QImage, QPainter  # noqa: E402
from PyQt6.QtTest import QTest  # noqa: E402
from PyQt6.QtWidgets import QApplication, QDialog, QMenu  # noqa: E402


# ====================================================================== the throw-away home folder
def home() -> str:
    return os.environ['HOME']


def _from_home(text):
    h = home()
    return text.replace(h + os.sep, '~' + os.sep).replace(h, '~') if isinstance(text, str) else text


def tilde(*widgets):
    """Show paths under the (throw-away) home folder from ~ in these widgets, as the app itself does where it shortens
    a path (the dataset page's short_path): wraps their setText / setPlainText, and fixes what they show now."""
    for w in widgets:
        for name in ('setText', 'setPlainText', 'setToolTip'):
            f = getattr(w, name, None)
            if f is not None:
                setattr(w, name, lambda text, f=f: f(_from_home(text)))
        if hasattr(w, 'toPlainText'):
            w.setPlainText(w.toPlainText())
        elif hasattr(w, 'text'):
            w.setText(w.text())


def tilde_path_label(pg):
    """A recording page (drive-log check, as-built check) shows its recording's full path, elided, in path_label: show
    it from ~ instead, re-derived from the page's settings each time the page fills its fields."""
    sync = pg._sync_fields

    def synced(*a, **k):
        out = sync(*a, **k)
        p = pg.settings.path
        if p:
            pg.path_label.setText(_from_home(p))
            pg.path_label.setToolTip(_from_home(p))
        return out
    pg._sync_fields = synced
    synced()


# ====================================================================== the shader renderer in the recorded views
def use_shader_renderer():
    from gl_harness import egl_context
    egl_context()
    from vantage.ui import viewport3d as V
    from vantage.ui.render_gl import GLRenderer
    R = GLRenderer()
    if not R.init():
        raise SystemExit(f'shader renderer unavailable: {R.error}')
    frame = V.GLViewport._frame

    def paint_scene(self, p):
        w, h = self.width(), self.height()
        if w < 8 or h < 8:
            return
        self.renderer = R
        rgba = R.render_offscreen(frame(self, w, h, 1.0, clear=(0, 0, 0, 0)))
        p.drawImage(0, 0, QImage(rgba.tobytes(), w, h, w * 4, QImage.Format.Format_RGBA8888_Premultiplied))
    V.SoftViewport._paint_scene = paint_scene
    V.SoftViewport.gl_bowl = True
    V.SoftViewport.renderer = R


# ====================================================================== recording
class Clip(Recorder):
    """The tour's recorder without narration: one silent clip."""

    def __init__(self, app, path, fps=30):
        self.app, self.fps, self.out, self.preview = app, fps, os.path.dirname(path), False
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
        self.path = path
        self.proc = subprocess.Popen(
            ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgra', '-s', f'{W}x{H}',
             '-r', str(fps), '-i', '-', '-c:v', 'libx264', '-preset', 'medium', '-crf', '16', '-pix_fmt', 'yuv420p',
             path], stdin=subprocess.PIPE)

    def close(self):
        self.proc.stdin.close()
        self.proc.wait()

    def window_image(self) -> QImage:
        """The window with its open menus, dialogs and popups (a combo box's list) drawn where they are."""
        img = self.win.grab().toImage().convertToFormat(QImage.Format.Format_RGB32)
        if img.width() != W or img.height() != H:
            img = img.scaled(W, H)
        p = QPainter(img)
        o = self.win.mapToGlobal(QPoint(0, 0))
        tops = [w for w in QApplication.topLevelWidgets()
                if w is not self.win and w.isVisible() and w.width() > 4
                and (isinstance(w, (QMenu, QDialog)) or w.windowType() == Qt.WindowType.Popup)]
        tops.sort(key=lambda w: 0 if isinstance(w, QDialog) else 1)
        for w in tops:
            g = w.mapToGlobal(QPoint(0, 0))
            pos = QPoint(g.x() - o.x(), g.y() - o.y())
            for k, a in ((14, 18), (8, 26), (3, 40)):
                p.fillRect(QRect(pos.x() - k // 2, pos.y() + k // 2, w.width() + k, w.height() + k // 2), QColor(0, 0, 0, a))
            p.drawPixmap(pos, w.grab())
        p.end()
        return img


class Stage:
    def __init__(self, app, out, fps):
        self.app, self.out, self.fps = app, out, fps
        self.win = None

    # the throw-away home folder, for the clip modules
    home = property(lambda self: home())
    tilde = staticmethod(tilde)
    tilde_path_label = staticmethod(tilde_path_label)

    # ---------------------------------------------------------------- set-up
    def window(self, edit=None):
        from PyQt6.QtCore import QSettings
        from vantage.core import samples
        from vantage.ui.mainwindow import MainWindow
        QSettings('Vantage', 'Vantage').remove('overlay')     # each clip starts from the app's default overlays
        p = samples.tigor_sample_project()
        p.photos.source = ''                    # the studio photos are third-party: never in a recording
        if edit is not None:
            edit(p)
        w = MainWindow(p, soft_3d=True)
        w.resize(W, H)
        w.show()
        self.win = w
        self.settle(60)
        vp = self.vp
        vp.cam.az, vp.cam.el = -42.0, 24.0
        w.viewarea.frame_all()
        self.settle(3)
        return w

    def close_window(self):
        w = self.win
        if w is None:
            return
        w.store.undo.setClean()
        w.cancel_jobs()
        for d in QApplication.topLevelWidgets():         # a clip's dialogs and popups must not show in the next one
            if d is not w and d.isVisible() and (isinstance(d, (QDialog, QMenu)) or d.windowType() == Qt.WindowType.Popup):
                d.hide()
        w.hide()
        w.deleteLater()
        self.win = None
        for _ in range(5):
            self.app.processEvents()

    @property
    def vp(self):
        return self.win.viewarea.panes[0].vp3d

    def settle(self, timeout=20):
        t = time.time()
        a = self.win.ctx.analysis
        while time.time() - t < timeout:
            for _ in range(5):
                self.app.processEvents()
                time.sleep(0.01)
            if a.result is not None and not a._busy and not a.timer.isActive():
                break

    def orbit(self, r, vp, d_az, dur, d_el=0.0, linear=False):
        n = max(1, round(dur * self.fps))
        az0, el0 = vp.cam.az, vp.cam.el
        for k in range(1, n + 1):
            t = k / n
            s = t if linear else (t if 0.15 < t < 0.85 else ease(t))
            vp.cam.az = az0 + d_az * s
            vp.cam.el = el0 + d_el * s
            vp.update()
            r.tick()

    def record(self, name, fn):
        path = os.path.join(self.out, f'{name}.raw.mp4')
        r = Clip(self.app, path, self.fps)
        t0 = time.time()
        try:
            fn(r)
        finally:
            r.close()
            self.close_window()
        print(f'{name}: {r.f / self.fps:.1f} s of video in {time.time() - t0:.0f} s', flush=True)

    # ================================================================ clips
    def hero(self, r):
        """A slow orbit: the body lit by each camera's view, its frustums, shadows (no window chrome)."""
        from vantage.ui import viewport3d as V
        w = self.window()
        w.ctx.set_overlay('camlight', True)
        w.ctx.set_overlay('frustums', True)
        self.settle(5)
        hv = V.SoftViewport(self.vp.provider)
        hv.paint_overlay = lambda p, ui=True: None
        hv.resize(W, H)
        hv.show()
        hv.cam.az, hv.cam.el = -30.0, 24.0
        hv.frame_box(*self._car_box(w), 0.92)
        hv.cam.dist *= 0.7
        r.win = hv
        for _ in range(10):
            self.app.processEvents()
        self.orbit(r, hv, 360.0, 16.0, linear=True)
        hv.hide()

    def _car_box(self, w):
        vg = w.ctx.analysis.engine.vehicle_geom(w.store.project)
        lo, hi = vg.bbox
        return np.array([lo[0] - 1.6, lo[1] - 1.6, 0.0]), np.array([hi[0] + 1.6, hi[1] + 1.6, hi[2]])

    def coverage(self, r):
        w = self.window()
        r.win = w
        r.chapter('Analysis', 'Live coverage')
        r.cursor_on = False
        self.orbit(r, self.vp, -40, 2.4, d_el=8)
        r.cursor_on = True
        r.menu('View', 'Overlays', 'Heatmap', 'Density')
        r.cursor_on = False
        self.orbit(r, self.vp, -30, 2.6)
        r.cursor_on = True
        r.menu('View', 'Overlays', 'Heatmap', 'Redundancy')
        r.cursor_on = False
        self.orbit(r, self.vp, -25, 2.0)
        r.cursor_on = True
        w.bottom.show_tab('coverage')
        rect = r.wrect(w.docks['bottom'])
        r.move(QPointF(rect.x() + 200, rect.y() + 100), 0.8)
        r.zoom(QRectF(rect.x(), rect.y(), rect.width() * 0.5, rect.height()), 0.9)
        r.hold(1.8)
        r.zoom(None, 0.7)

    def placement(self, r):
        w = self.window()
        r.win = w
        st = w.store
        w.set_heatmap('density')
        self.settle(3)
        r.chapter('Placement', 'Edit a camera, watch the analysis follow')
        ml = st.variant.by_tag('ML')
        s, _, ok = self.vp.project([ml.pos()])
        r.click(self.vp, QPointF(*s[0]), 0.9)
        self.settle(3)
        insp = r.wrect(w.docks['inspector'])
        r.move(QPointF(insp.x() + insp.width() * 0.6, insp.y() + 200), 0.8)
        p0, y0 = st.cam(ml.uid).pitch, st.cam(ml.uid).yaw
        n = round(4.5 * self.fps)
        for k in range(1, n + 1):
            t = k / n
            st.set_cam(ml.uid, merge=True, pitch=round(p0 + 16 * math.sin(math.pi * t), 1),
                       yaw=round(y0 - 22 * math.sin(2 * math.pi * t) * 0.6, 1), text='Pose')
            r.cur = np.array([insp.x() + insp.width() * 0.6 + 50 * math.sin(math.pi * t), insp.y() + 200])
            r.tick()
        st.set_cam(ml.uid, merge=True, pitch=p0, yaw=y0, text='Pose')
        self.settle(3)
        r.hold(0.4)
        r.toolbar('split4')
        self.settle(3)
        r.cursor_on = False
        r.hold(2.6)

    def camlight(self, r):
        w = self.window()
        w.ctx.set_overlay('camlight', False)            # on by default: the clip turns it on from the menu
        self.settle(2)
        r.win = w
        st = w.store
        r.chapter('3D', 'Camera light: what each lens sees')
        r.menu('View', 'Overlays', 'Camera light')
        self.settle(3)
        r.cursor_on = False
        self.orbit(r, self.vp, 35, 2.5, d_el=6)
        ml = st.variant.by_tag('F')
        y0 = ml.yaw
        n = round(3.5 * self.fps)
        for k in range(1, n + 1):
            t = k / n
            st.set_cam(ml.uid, merge=True, yaw=round(y0 + 25 * math.sin(2 * math.pi * t), 1), text='Yaw')
            r.tick()
        st.set_cam(ml.uid, merge=True, yaw=y0, text='Yaw')
        self.orbit(r, self.vp, -60, 3.0, d_el=-4)

    def bowl(self, r):
        """The 3D surround view full-frame: the parking scene through every camera, projected onto the bowl."""
        from vantage.ui import viewport3d as V
        w = self.window()
        w.set_page('camviews')
        pg = w._page_widgets['camviews']
        ctl = pg.bowl_scene.bowl
        t = time.time()
        while ctl.draw is None and time.time() - t < 120:
            self.app.processEvents()
            time.sleep(0.05)
        self.settle(3)
        bv = V.SoftViewport(pg.bowl_scene)
        bv.paint_overlay = lambda p, ui=True: None
        bv.resize(W, H)
        bv.show()
        src = pg.bowl.cam
        bv.cam.target, bv.cam.dist, bv.cam.fov = src.target.copy(), src.dist * 0.82, src.fov
        bv.cam.az, bv.cam.el = 150.0, 34.0
        r.win = bv
        for _ in range(10):
            self.app.processEvents()
        r.chapter('Surround view', '3D bowl: what the in-car display shows')
        self.orbit(r, bv, 300.0, 15.0, d_el=-10, linear=True)
        bv.hide()

    def guidelines(self, r):
        w = self.window()
        r.win = w
        w.set_page('camviews')
        pg = w._page_widgets['camviews']
        self.settle(3)
        bar = pg.guide_bar
        r.chapter('Parking guidelines', 'Trajectory and swept path through every lens')
        if not bar.on.isChecked():
            r.click(bar.on, QPointF(8, bar.on.height() / 2), 0.8)
        self.settle(2)
        sl = bar.slider
        y = sl.height() / 2
        mid = QPointF(sl.width() / 2, y)
        r.drag(sl, mid, QPointF(sl.width() * 0.2, y), 2.0)
        r.hold(0.8)
        r.drag(sl, QPointF(sl.width() * 0.2, y), QPointF(sl.width() * 0.8, y), 3.0)
        r.hold(0.8)
        cams = r.wrect(pg.views[1]) if len(pg.views) > 1 else None
        if cams is not None:
            r.zoom(cams, 0.9)
            r.drag(sl, QPointF(sl.width() * 0.8, y), QPointF(sl.width() * 0.35, y), 2.0)
            r.hold(0.8)
            r.zoom(None, 0.7)
        r.cursor_on = False
        r.hold(0.8)

    def stitched(self, r):
        w = self.window()
        r.win = w
        pane = w.viewarea.active
        pane.set_kind('top')
        self.settle(3)
        r.chapter('Stitched view', 'Where standing people vanish at the seams')
        r.menu('View', 'Overlays', 'Heatmap', 'Objects')
        t = time.time()
        while w.ctx.stitched.result is None and time.time() - t < 30:
            self.app.processEvents()
            time.sleep(0.05)
        self.settle(3)
        plan = pane.plan
        plan.fit()
        r.hold(0.5)
        hs = w.ctx.stitched.result.hotspots if w.ctx.stitched.result else []
        if hs:
            h = hs[0]
            a = plan.w2s(h.x - 0.9, h.y - 0.5)
            b = plan.w2s(h.x + 0.5, h.y + 0.6)
            r.move(r.wpos(plan, a), 0.8)
            n = round(3.0 * self.fps)
            for k in range(1, n + 1):
                tt = ease(k / n)
                pt = QPointF(a.x() + (b.x() - a.x()) * tt, a.y() + (b.y() - a.y()) * tt)
                QTest.mouseMove(plan, pt.toPoint())
                r.cur = np.array([r.wpos(plan, pt).x(), r.wpos(plan, pt).y()])
                r.tick()
            r.hold(1.6)
        w.set_page('camviews')
        pg = w._page_widgets['camviews']
        self.settle(2)
        r.hold(0.4)
        r.click(pg.b_suggest)
        t = time.time()
        while w.ctx.stitched.advice is None and time.time() - t < 40:
            r.tick()
        r.hold(0.8)
        if pg.b_apply.isEnabled():
            r.click(pg.b_apply)
        self.settle(4)
        r.hold(1.5)

    def volume(self, r):
        w = self.window()
        r.win = w
        r.chapter('Analysis', '3D blind volume')
        r.menu('Analysis', 'Blind volume')
        pg = w._page_widgets.get('volume') or w.page('volume')
        ctl = w.ctx.volume
        t = time.time()
        while ctl.result is None and time.time() - t < 5:
            r.tick()
        if ctl.result is None:
            r.click(pg.b_run)
            r.wait_for(lambda: ctl.result is not None, timeout=60)
        self.settle(2)
        r.hold(0.8)
        sl = pg.slider
        y = sl.height() / 2
        x0 = (sl.value() - sl.minimum()) / max(1, sl.maximum() - sl.minimum()) * sl.width()
        r.drag(sl, QPointF(x0, y), QPointF(sl.width() * 0.08, y), 1.4)
        r.drag(sl, QPointF(sl.width() * 0.08, y), QPointF(sl.width() * 0.9, y), 3.2)
        r.hold(0.6)
        r.drag(sl, QPointF(sl.width() * 0.9, y), QPointF(sl.width() * 0.45, y), 1.4)
        r.cursor_on = False
        r.hold(1.6)

    def install(self, r):
        w = self.window()
        r.win = w
        r.chapter('Calibration', 'Installation check from one image per camera')
        r.menu('Calibrate', 'Installation check')
        pg = w.page('install')
        self.settle(3)
        r.move_to(pg.sim_ang, None, 0.8)
        r.hold(0.5)
        r.click(pg.b_run, real=True)
        r.wait_for(lambda: not pg.busy and bool(pg.results), timeout=240, speed=3)
        r.zoom(r.wrect(pg.table), 1.0)
        r.move_to(pg.table, QPointF(pg.table.width() * 0.55, 60), 0.8)
        r.hold(1.6)
        r.zoom(None, 0.7)
        bev = r.wrect(pg.bev)
        r.zoom(QRectF(bev.x() - 20, bev.y() - 40, bev.width() + 40, bev.height() + 60), 1.0)
        r.click(pg.which.buttons['planned'])
        r.hold(1.4)
        r.click(pg.which.buttons['measured'])
        r.hold(1.4)
        r.zoom(None, 0.7)

    def tolerance(self, r):
        w = self.window()
        r.win = w
        r.chapter('Studies', 'Monte Carlo tolerance study')
        r.menu('Analysis', 'Tolerance study')
        pg = w.page('tolerance')
        pg.samples.set_value(500)
        r.hold(0.4)
        r.click(pg.b_run, real=False)
        pg.run()
        r.wait_for(lambda: 'tolerance' not in w._jobs, timeout=240, speed=2)
        r.hold(0.4)
        hr, tr = r.wrect(pg.hist), r.wrect(pg.tornado)
        r.zoom(hr.united(tr), 0.9)
        r.move_to(pg.tornado, None, 0.8)
        r.hold(1.6)
        r.zoom(None, 0.7)
        r.hold(0.6)

    def assistant(self, r):
        from fake_groq import FakeGroq
        from vantage.assistant import secrets as K
        srv = FakeGroq().start()
        os.environ['VANTAGE_ASSISTANT_URL'] = srv.url
        try:
            K.save('test-key')

            def tilt(p):
                for tag in ('QR', 'QL'):
                    p.variant.by_tag(tag).pitch -= 12
            w = self.window(edit=tilt)
            r.win = w
            w.ctx.settings.setValue('assistant/enabled', True)
            w.ctx.assistant.reload()
            w.show_assistant()
            self.settle(2)
            panel = w.assistant_panel
            srv.delay = 0.012
            S = ['Find the blind ground and why', 'Search a better pose for the quarter cameras',
                 'Verify and propose the best option']

            def plan(st, cid):
                return srv.tool_call('update_plan', {'steps': [{'step': a, 'status': b} for a, b in zip(S, st)]}, cid)
            srv.script += [
                plan(['in_progress', 'pending', 'pending'], 'p1'),
                srv.tool_call('find_blind_regions', {}, 'a1', reasoning='Start with where the ground is blind and why.'),
                plan(['done', 'in_progress', 'pending'], 'p2'),
                srv.tool_call('optimise_camera', {'camera': 'QR', 'vary': ['pitch', 'yaw'], 'objective': 'blind',
                                                  'min_density': 3}, 'a2'),
                plan(['done', 'done', 'in_progress'], 'p3'),
                srv.tool_call('propose_changes', {'changes': [{'camera': 'QR', 'dpitch': 4}],
                                                  'title': 'Tilt the quarter cameras down 4°',
                                                  'rationale': 'Brings the rear wheel arches back into QR and QL.'}, 'a3'),
                plan(['done', 'done', 'done'], 'p4'),
                srv.answer('**Proposed: tilt QR and QL down 4°.** The blind patches behind the rear wheels close; the '
                           'minimum density stays at 3.1 px/cm.\n\n- 22 poses tried in half a second; yaw did not help.\n'
                           '- Nothing changes until you apply it.'),
            ]
            r.chapter('AI assistant', 'Agent mode: plan, measure, search, propose')
            dock = r.wrect(w.docks['assistant'])
            r.click(panel.composer.agent)
            r.move_to(panel.composer.edit, None, 0.6)
            for ch in 'Close every blind patch':
                QTest.keyClicks(panel.composer.edit, ch)
                r.tick(2)
            r.hold(0.3)
            QTest.keyClick(panel.composer.edit, Qt.Key.Key_Return)
            r.zoom(QRectF(dock.x() - 260, dock.y(), dock.width() + 260, dock.height()), 1.0, wait=False)
            r.wait_for(lambda: not w.ctx.assistant.busy, timeout=90, speed=1)
            r.hold(0.6)
            from vantage.ui.assistant import ProposalCard
            cards = panel.findChildren(ProposalCard)
            if cards:
                panel.area.ensureWidgetVisible(cards[-1].b_apply)
                r.hold(0.4)
                r.click(cards[-1].b_apply)
                r.wait_for(lambda: bool(w.ctx.assistant.proposals.get(1, {}).get('measured')), timeout=30)
            r.hold(2.2)
            r.zoom(None, 0.7)
            r.hold(0.6)
        finally:
            os.environ.pop('VANTAGE_ASSISTANT_URL', None)
            K.delete()
            srv.stop()

    def perception(self, r):
        """Perception range: how far out the rig resolves each object, against the ranges the dataset needs."""
        w = self.window()
        r.win = w
        r.chapter('Simulation', 'Perception range: how far the rig resolves objects')
        r.menu('Analysis', 'Perception range')
        pg = w.page('perception')
        t = time.time()
        while pg.ctl.result is None and time.time() - t < 8:
            r.tick()
        if pg.ctl.result is None:
            r.click(pg.b_run)
            r.wait_for(lambda: pg.ctl.result is not None, timeout=60)
        r.hold(1.6)
        for key in ('1', '3'):
            b = pg.pick.buttons.get(key) if hasattr(pg.pick, 'buttons') else None
            if b is not None:
                r.click(b)
                r.hold(1.4)
        b = pg.mode.buttons.get('cameras') if hasattr(pg.mode, 'buttons') else None
        if b is not None:
            r.click(b)
            r.hold(1.8)
        right = r.wrect(pg)
        r.zoom(QRectF(right.x() + right.width() * 0.64, right.y(), right.width() * 0.36, right.height()), 1.0)
        r.hold(2.2)
        r.zoom(None, 0.7)
        r.hold(0.5)

    def rig(self, r):
        """Rig parts: a roof rack and a sensor mast block and carry cameras."""
        from vantage.core.project import Camera
        w = self.window()
        w.ctx.set_overlay('camlight', False)            # on by default: the clip turns it on from the menu
        self.settle(2)
        r.win = w
        st = w.store
        r.chapter('Rig', 'Roof racks and masts that block and carry cameras')
        r.menu('Vehicle', 'Add rig part', 'Roof rack')
        self.settle(3)
        r.menu('Vehicle', 'Add rig part', 'Sensor mast')
        self.settle(3)
        head = max(st.project.rig, key=lambda part: part.z)

        def add(p):
            c = Camera(tag='RF', name='Roof front', x=head.x + head.length / 2 + 0.012, y=0.0, z=head.z, yaw=0.0, pitch=18.0,
                       slot=6, lens_id='h03r2')
            p.variant.cameras.append(c)
        st.edit('variant', add, 'Roof camera on the mast')
        self.settle(4)
        r.menu('View', 'Overlays', 'Camera light')
        self.settle(3)
        r.cursor_on = False
        self.orbit(r, self.vp, 70, 4.5, d_el=6)
        self.orbit(r, self.vp, -50, 3.5, d_el=-4)
        r.hold(0.6)

    def sensors(self, r):
        """Lidars and radars: a lidar on the sensor mast with its ground rings, radars with their fields."""
        from vantage.core.samples import data_rig_sensors
        from vantage.ui.sensorview import add_sensor, set_props
        w = self.window()
        r.win = w
        st = w.store
        r.chapter('Sensors', 'Lidars and radars on the rig')
        r.menu('Vehicle', 'Add rig part', 'Sensor mast')
        self.settle(3)
        r.menu('Vehicle', 'Add lidar', 'Ouster OS1-128')
        self.settle(3)
        r.menu('Vehicle', 'Add radar', 'Long-range radar')
        self.settle(2)
        for s in data_rig_sensors(st.project.vehicle.mesh)[2:]:          # the four corner radars
            q = add_sensor(w.ctx, 'radar', 'radar_corner')
            set_props(w.ctx, q.uid, f'Placed {q.tag}', x=s.x, y=s.y, z=s.z, yaw=s.yaw)
        lid = next(s for s in st.variant.sensors if s.kind == 'lidar')
        st.select_obj(('sensor', lid.uid))
        self.settle(3)
        r.cursor_on = False
        self.orbit(r, self.vp, 70, 4.5, d_el=8)
        r.hold(0.4)
        w.set_view('top')
        self.settle(2)
        plan = w.viewarea.panes[0].plan
        s0 = plan.scale
        n = round(2.4 * self.fps)
        for k in range(1, n + 1):                         # zoom out to the rings and the radar fields
            plan.scale = s0 * (1 - 0.72 * ease(k / n))
            plan._user_view = True
            plan.update()
            r.tick()
        r.hold(2.0)
        w.set_view('persp')
        r.hold(0.5)

    def _rig_window(self, edit=None):
        """The sample with its data-rig variant (six cameras, a roof lidar, five radars) active; edit(p) runs after."""
        def setup(p):
            p.active_uid = next(v.uid for v in p.variants if v.sensors)
            if edit is not None:
                edit(p)
        return self.window(setup)

    def _nav_window(self, edit=None):
        """The data rig with two GNSS antennas and a dual-antenna INS where Vehicle ▸ Add puts them; edit(p) runs
        after."""
        def setup(p):
            from vantage.core import sensors
            from vantage.core.project import Sensor
            v = p.variant
            for kind in ('gnss', 'gnss', 'imu'):
                m = p.sensor_models[sensors.default_model_id(kind)]
                v.sensors.append(Sensor(kind=kind, model_id=m.id, tag=sensors.next_tag(kind, v.tags()), name=m.name,
                                        **sensors.default_pose(kind, m, p.vehicle.mesh, p.rig, v.sensors)))
            if edit is not None:
                edit(p)
        return self._rig_window(setup)

    def sensorcoverage(self, r):
        """Lidar and radar coverage: what the roof blocks, the sector verdicts, then the lidar raised on a riser."""
        w = self._rig_window()
        r.win = w
        st = w.store
        r.chapter('Coverage', 'What the lidars and radars add to the cameras')
        r.menu('Analysis', 'Lidar and radar coverage')
        pg = w.page('coverage')
        r.wait_for(lambda: pg.ctl.result is not None, timeout=60)
        r.hold(2.0)
        r.click(pg.reach.buttons['60'] if '60' in pg.reach.buttons else next(iter(pg.reach.buttons.values())))
        r.hold(1.4)
        r.click(pg.layer.buttons['rings'])
        r.hold(1.8)
        r.zoom(r.wrect(pg.lidars), 0.9, margin=60)
        r.hold(2.0)
        r.zoom(None, 0.6)
        lid = next(s for s in st.variant.sensors if s.kind == 'lidar')
        from vantage.ui.sensorview import set_props
        before = pg.ctl.result
        set_props(w.ctx, lid.uid, f'Raised {lid.tag} 0.6 m', z=round(lid.z + 0.6, 3))
        r.wait_for(lambda: pg.ctl.result is not None and pg.ctl.result is not before, timeout=60)
        r.hold(2.4)
        r.click(pg.layer.buttons['density'])
        r.hold(2.0)
        r.zoom(r.wrect(pg.grid), 0.9, margin=60)
        r.hold(1.8)
        r.zoom(None, 0.6)
        r.hold(0.4)

    def timing(self, r):
        """Sync and timing: camera triggers against the lidar sweep, then the suggested trigger lines."""
        w = self._rig_window()
        r.win = w
        r.chapter('Sync', 'Camera triggers against the lidar sweep')
        r.menu('Analysis', 'Sync and timing')
        pg = w.page('timing')
        self.settle(2)
        r.hold(2.0)
        right = r.wrect(pg.table)
        r.zoom(QRectF(right.x() - 8, right.y() - 30, right.width() + 16, right.height() * 0.42), 0.9)
        r.hold(1.6)
        r.zoom(None, 0.6)
        r.click(pg.b_suggest)
        r.hold(1.8)
        r.click(pg.b_apply)
        self.settle(2)
        r.hold(2.4)
        r.zoom(r.wrect(pg.dial), 0.9)
        r.hold(1.6)
        r.zoom(None, 0.6)
        r.hold(0.4)

    def recording(self, r):
        """Recording budget: every sensor's data rate, the recorder's disks and the links, raw against H.265."""
        w = self._rig_window()
        r.win = w
        r.chapter('Recording', 'How much the rig records, and whether the recorder keeps up')
        r.menu('Analysis', 'Recording budget')
        pg = w.page('recording')
        self.settle(2)
        r.hold(2.2)
        r.click(pg.cam_fmt.buttons['h265'])
        self.settle(1)
        r.hold(2.2)
        r.click(pg.cam_fmt.buttons['raw'])
        self.settle(1)
        r.hold(1.2)
        top = r.wrect(pg)
        r.zoom(QRectF(top.x() + top.width() * 0.2, top.y(), top.width() * 0.8, top.height() * 0.62), 0.9)
        r.hold(2.0)
        r.zoom(None, 0.6)
        r.hold(0.4)

    def liverig(self, r):
        """Live rig health: every camera's stream with frame rate, drops, exposure, sharpness and drift."""
        w = self.window()
        r.win = w
        r.chapter('Live rig', 'Every stream, its health, and whether a camera moved')
        r.menu('Calibrate', 'Live rig')
        pg = w.page('rig')
        for c in w.store.variant.cameras:
            pg.set_kind(c.uid, 'synthetic')
        self.settle(2)
        r.hold(0.5)
        r.click(pg.b_start)
        t0 = time.time()
        while time.time() - t0 < 9.0:
            r.tick()
            time.sleep(0.02)
        r.hold(0.4)
        grid = r.wrect(pg)
        r.zoom(QRectF(grid.x(), grid.y() + 40, grid.width() * 0.5, grid.height() * 0.55), 1.0)
        t0 = time.time()
        while time.time() - t0 < 2.5:
            r.tick()
            time.sleep(0.02)
        r.zoom(None, 0.7)
        pg.stop()
        r.hold(0.4)

    def datasets(self, r):
        """Calibration for the data pipeline: nuScenes, KITTI, ROS camera_info and Kalibr, previewed live."""
        from vantage.ui.dialogs.export_dialog import ExportDialog
        w = self.window()
        r.win = w
        r.chapter('Export', "Calibration in your dataset's format")
        d = ExportDialog(w.ctx, w)
        self.tilde(d.out)                       # the default output folder (in the working folder), from ~
        d.show()
        r.hold(0.4)
        r.zoom(r.wrect(d), 0.9)
        for fid in ('nuscenes_calib', 'kitti_calib', 'ros_camera_info', 'kalibr_camchain'):
            r.click(d.rows[fid].cb, None, 0.6)            # tick it: it gets a preview tab and is written
            d.select(fid)
            r.hold(1.8)
        r.zoom(None, 0.6)
        d.close()
        r.hold(0.3)


CLIPS = ['hero', 'coverage', 'placement', 'camlight', 'bowl', 'guidelines', 'stitched', 'volume', 'install',
         'tolerance', 'assistant', 'perception', 'rig', 'liverig', 'datasets', 'sensors', 'sensorcoverage', 'timing',
         'recording', 'power', 'gnss', 'exposure', 'brackets', 'requirements', 'dataset', 'lidarcalib', 'drivelog',
         'fleet', 'housings', 'asbuilt', 'loadcases', 'library']
POSTER_AT = {'hero': 2.0, 'requirements': 18.0, 'housings': 4.0, 'asbuilt': 13.5}


def clip_fn(stage, name):
    """A clip: a Stage method, or clip(stage, r) in demo/clip_<name>.py."""
    fn = getattr(stage, name, None)
    if fn is not None:
        return fn
    import importlib
    mod = importlib.import_module(f'clip_{name}')
    return lambda r: mod.clip(stage, r)


def web(out, name, poster_at=None):
    """The web version (H.264, faststart, no audio) and a JPEG poster."""
    src = os.path.join(out, f'{name}.raw.mp4')
    dst = os.path.join(out, f'{name}.mp4')
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', src, '-c:v', 'libx264', '-preset', 'slow', '-crf', '27',
                    '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-an', dst], check=True)
    dur = float(subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', dst],
                               capture_output=True, text=True).stdout.strip() or 1)
    t = poster_at if poster_at is not None else dur * 0.72
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-ss', f'{t:.2f}', '-i', dst, '-frames:v', '1', '-q:v', '3',
                    os.path.join(out, f'{name}.jpg')], check=True)
    return os.path.getsize(dst), dur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--clips', default=','.join(CLIPS))
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--theme', default='dark')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    a.out = os.path.abspath(a.out)
    # a throw-away home folder: what the clips write lands there, and the app shows it as ~ (see tilde())
    os.environ['HOME'] = tempfile.mkdtemp(prefix='h')
    try:
        run(a)
    finally:
        shutil.rmtree(os.environ['HOME'], ignore_errors=True)


def run(a):
    from vantage.ui.viewport3d import configure_default_format
    configure_default_format()
    app = QApplication(sys.argv[:1])
    from vantage.ui.theme import THEME, load_fonts
    load_fonts()
    THEME.apply(app, a.theme)
    use_shader_renderer()
    st = Stage(app, a.out, a.fps)
    failed = []
    for name in [c for c in a.clips.split(',') if c]:
        try:
            st.record(name, clip_fn(st, name))
            size, dur = web(a.out, name, POSTER_AT.get(name))
            print(f'  -> {name}.mp4 {size / 1e6:.1f} MB, {dur:.1f} s', flush=True)
        except Exception:
            traceback.print_exc()
            failed.append(name)
    if failed:
        print('failed:', ', '.join(failed))
        sys.exit(1)


if __name__ == '__main__':
    main()
