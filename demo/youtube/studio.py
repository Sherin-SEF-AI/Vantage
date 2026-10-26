"""The tour's studio: the recorder (Rec), the stage (Stage) and the shader renderer set-up. A chapter's scene(stage, r)
drives the real application through these; see AUTHORING.md for the API.

The app runs offscreen at 1920 x 1080 logical pixels with a device pixel ratio of 2 (QT_SCALE_FACTOR=2, set by
record.py): every frame is grabbed at 3840 x 2160 and scaled down, so text is crisp in the full frame and still crisp
in a 2x zoom. The 3D views are drawn by the shader renderer through a headless EGL context at that resolution.
Overlays (lower thirds, callouts, figures, keycaps) are painted on the final 1920 x 1080 frame, after the zoom.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DEMO = os.path.dirname(HERE)
ROOT = os.path.dirname(DEMO)
for p_ in (os.path.join(ROOT, 'tests'), ROOT, DEMO, HERE):
    if p_ not in sys.path:
        sys.path.insert(0, p_)
# Qt's offscreen platform has an 800 × 800 virtual screen (400 × 400 logical at the 2× capture scale): menus,
# submenus, combo pop-ups and tooltips were clamped into its top-left corner. A 3840 × 2160 screen, 1920 × 1080
# logical, places them where a desktop does.
PLATFORM = 'offscreen:configfile=' + os.path.join(HERE, 'offscreen-screen.json')
os.environ['QT_QPA_PLATFORM'] = PLATFORM
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['VANTAGE_NO_KEYRING'] = '1'
os.environ.pop('GROQ_API_KEY', None)

import numpy as np  # noqa: E402
import clips  # noqa: E402  (imports tour: a throw-away XDG_CONFIG_HOME, the pointer, menus)
import tour  # noqa: E402
from common import FPS, GAP, H, W  # noqa: E402

os.environ['QT_QPA_PLATFORM'] = PLATFORM        # clips.py sets plain 'offscreen' when it is imported

clips.W, clips.H = W, H
tour.W, tour.H = W, H

from PyQt6.QtCore import QPoint, QPointF, QRect, QRectF, Qt  # noqa: E402
from PyQt6.QtGui import QColor, QImage, QPainter, QPainterPath  # noqa: E402
from PyQt6.QtTest import QTest  # noqa: E402
from PyQt6.QtWidgets import QApplication, QDialog, QMenu, QWidget  # noqa: E402

import gfx  # noqa: E402
from gfx import from_np, smoother, to_np  # noqa: E402

STYLE_VERSION = 2           # bump to re-record every segment after a change to the recorder or the overlays
                            # (2: a full-size offscreen screen, so menus and pop-ups open in place)
MIN_CROP_W = W / 2.6        # the deepest zoom: 2.6x (crisp to 2x)


# ====================================================================== the shader renderer at the device pixel ratio
def use_shader_renderer():
    """3D views drawn by the shader renderer (headless EGL) at the painter's device pixel ratio."""
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
        dev = p.device()
        dpr = dev.devicePixelRatioF() if dev is not None else 1.0
        pw, ph = int(round(w * dpr)), int(round(h * dpr))
        self.renderer = R
        rgba = R.render_offscreen(frame(self, pw, ph, dpr, clear=(0, 0, 0, 0)))
        img = QImage(rgba.tobytes(), pw, ph, pw * 4, QImage.Format.Format_RGBA8888_Premultiplied)
        img.setDevicePixelRatio(dpr)
        p.drawImage(QPointF(0, 0), img)
    V.SoftViewport._paint_scene = paint_scene
    V.SoftViewport.gl_bowl = True
    V.SoftViewport.renderer = R
    return R


# ====================================================================== the recorder
class Rec(tour.Recorder):
    """Records one segment: drives the pointer, menus and zooms (tour.Recorder's), starts narration lines, and paints
    the overlays. Every method that takes a `target` accepts a widget, a QRect/QRectF in window coordinates, a
    (x, y, w, h) tuple, or a callable returning one of those (evaluated every frame)."""

    def __init__(self, app, out_dir, cid, voice_lines, fps=FPS, out_size=(W, H), encode=True, crf=12,
                 preset='medium', plan=None):
        self.app, self.fps, self.out, self.preview = app, fps, out_dir, False
        self.cid = cid
        self.vlines = {v['key']: v for v in voice_lines}
        self.dur = {k: v['seconds'] for k, v in self.vlines.items()}
        self.captions = {k: v['caption'] for k, v in self.vlines.items()}
        self.f = 0
        self.win = None
        self.origin = QPointF(0, 0)          # where a smaller filmed widget (a dialog on its own) sits in the frame
        self.cur = np.array([W * 0.62, H * 0.6])
        self.cursor_on = False
        self.ripples, self.tweens = [], []
        self.crop = np.array([0.0, 0.0, W, H])
        self._zoom = None
        self.layers = []
        self.card = None                     # fn(rec) -> QImage (W x H): a full-frame painting instead of the app
        self.stage_bg = None                 # QImage (W x H) behind a filmed widget smaller than the frame
        self.fade = None
        self.last = None
        self.timeline, self.events = [], []
        self.speech_end = 0.0
        self.scene = cid
        self._badge = None
        self._said = []
        self.out_size = tuple(out_size)
        self.wall0 = time.time()
        os.makedirs(os.path.join(out_dir, 'stills'), exist_ok=True)
        self.proc = None
        if encode:
            ow, oh = self.out_size
            self.proc = subprocess.Popen(
                ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgra', '-s', f'{ow}x{oh}',
                 '-r', str(fps), '-i', '-', '-c:v', 'libx264', '-preset', preset, '-crf', str(crf),
                 '-pix_fmt', 'yuv420p', os.path.join(out_dir, 'video.mp4')], stdin=subprocess.PIPE)
        if plan is not None:
            self.add(gfx.Indicator(plan, cid))

    # ------------------------------------------------------------------ time
    def tick(self, n=1):
        for _ in range(max(0, int(n))):
            self.app.processEvents()
            self._advance()
            self._write(self.compose())
            if self.f % self.fps == 0:
                try:
                    self.check_paths()
                except Exception as ex:                  # the guard must never break a recording
                    if not getattr(self, '_guard_failed', False):
                        self._guard_failed = True
                        print(f'path guard failed: {ex!r}', flush=True)
            self.f += 1

    def check_paths(self):
        """Look through the text of every visible widget for a machine path (the throw-away home, /home/, /tmp/,
        the scratch folder): each hit is logged as a 'path-leak' event and printed (review.py reports them)."""
        from PyQt6.QtCore import Qt as _Qt
        from PyQt6.QtWidgets import (QAbstractButton, QAbstractItemView, QComboBox, QLabel, QLineEdit,
                                     QPlainTextEdit, QTextEdit)
        bad = [b for b in (os.environ.get('HOME', ''), '/home/', '/tmp/', 'claude-1000') if b and len(b) > 4]
        roots = [w for w in [self.win] + self._tops() if w is not None]
        hits = set()

        def look(text, where):
            if isinstance(text, str):
                for b_ in bad:
                    if b_ in text:
                        hits.add(f'{where}: {text[:120]!r}')
        for root in roots:
            for wd in [root] + root.findChildren(QWidget):
                if not wd.isVisible():
                    continue
                if isinstance(wd, (QLabel, QLineEdit, QAbstractButton)):
                    look(wd.text(), type(wd).__name__)
                elif isinstance(wd, QComboBox):
                    look(wd.currentText(), 'QComboBox')
                elif isinstance(wd, (QPlainTextEdit, QTextEdit)):
                    look(wd.toPlainText()[:20000], type(wd).__name__)
                elif isinstance(wd, QAbstractItemView) and wd.model() is not None:
                    m = wd.model()
                    try:
                        ncol = m.columnCount()
                    except TypeError:                   # list models keep columnCount() private
                        ncol = 1
                    for row in range(min(m.rowCount(), 300)):
                        for col_ in range(min(ncol, 12)):
                            ix = m.index(row, col_)
                            look(m.data(ix, _Qt.ItemDataRole.DisplayRole), type(wd).__name__)
                            ud = m.data(ix, _Qt.ItemDataRole.UserRole)
                            if isinstance(ud, dict):
                                for v in ud.values():
                                    look(v, type(wd).__name__ + ' data')
        for h in sorted(hits):
            key = ('path-leak', h)
            if key not in getattr(self, '_leaks', set()):
                self._leaks = getattr(self, '_leaks', set()) | {key}
                self.events.append({'kind': 'path-leak', 'start': round(self.now(), 3), 'what': h})
                print(f'WARNING path on screen at {self.now():.1f} s: {h}', flush=True)

    def _advance(self):
        for tw in list(self.tweens):
            obj, start, end, f0, nf = tw
            t = tour.ease((self.f - f0 + 1) / nf)
            setattr(self, obj, start + (end - start) * t)
            if self.f - f0 + 1 >= nf:
                self.tweens.remove(tw)
        if self._zoom is not None:
            c0, c1, f0, n = self._zoom
            s = smoother((self.f - f0 + 1) / n)
            self.crop = _crop_between(c0, c1, s)
            if self.f - f0 + 1 >= n:
                self.crop = c1.copy()
                self._zoom = None
        self.layers = [ly for ly in self.layers if not ly.done(self.f)]

    def wait_for(self, cond, timeout=180, speed=1.0):
        """Record while the app works (a background job) until cond() is true. The video keeps pace with the wall
        clock divided by `speed`: speed 1 is real time; above 1 a time-lapse, marked on screen with the factor
        measured from the clock (video seconds against wall seconds). Raises TimeoutError after `timeout` s."""
        t0, f0 = time.time(), self.f
        badge = None
        if speed > 1.01:
            badge = self._badge = self.add(gfx.Badge())
            badge.factor = speed
        img = None
        while not cond():
            wall = time.time() - t0
            if wall > timeout:
                raise TimeoutError(f'wait_for: still waiting after {timeout} s')
            want = int(wall / speed * self.fps)
            if self.f - f0 < want or self.f == f0:
                self.app.processEvents()
                self._advance()
                img = self.compose()
                for _ in range(max(1, want - (self.f - f0))):
                    self._write(img)
                    self.f += 1
                if badge is not None and self.f > f0:
                    badge.factor = (time.time() - t0) / ((self.f - f0) / self.fps)
            else:
                self.app.processEvents()
                time.sleep(0.005)
        wall = time.time() - t0
        behind = int(wall / speed * self.fps) - (self.f - f0)       # the job ended between two frames: catch up
        if img is not None and behind > 0:
            for _ in range(behind):
                self._write(img)
                self.f += 1
        self.events.append({'kind': 'wait', 'start': round(f0 / self.fps, 3), 'video_s': round((self.f - f0) / self.fps, 3),
                            'wall_s': round(wall, 3), 'speed': speed})
        if badge is not None:
            badge.factor = wall / max(1e-6, (self.f - f0) / self.fps)
            badge.close(self, 0.6)
            self._badge = None
        self.tick()
        return wall

    def timelapse(self, factor=None):
        """Show the time-lapse badge with this factor (the caller measured it), or hide it (None)."""
        if factor is None:
            if self._badge is not None:
                self._badge.close(self)
                self._badge = None
            return
        if self._badge is None:
            self._badge = self.add(gfx.Badge())
        self._badge.factor = float(factor)

    # ------------------------------------------------------------------ narration
    def say(self, *args):
        """say(i) or say(chapter_id, i): start line i now, after the previous line has finished (plus a short gap):
        the scene waits here if needed, so the action that follows starts with the words."""
        cid, i = (self.cid, args[0]) if len(args) == 1 else args
        key = f'{cid}_{i}'
        if key not in self.dur:
            raise KeyError(f'no voice line {key}')
        if key in self._said:
            raise ValueError(f'line {key} said twice')
        if self.timeline:
            self.until(self.speech_end + GAP)
        t = self.now()
        v = self.vlines[key]
        self.timeline.append({'key': key, 'start': round(t, 3), 'seconds': v['seconds'], 'caption': v['caption'],
                              'hash': v['hash']})
        self._said.append(key)
        self.speech_end = t + v['seconds']
        print(f'{t:7.2f}s  {key}  ({v["seconds"]:.1f} s)', flush=True)

    def line(self, i):
        """Duration (s) of line i of this chapter."""
        return self.dur[f'{self.cid}_{i}']

    def speech_left(self):
        return max(0.0, self.speech_end - self.now())

    def finish_speech(self, pad=0.3):
        self.until(self.speech_end + pad)

    def at_word(self, i, fragment):
        """Seconds from the start of line i (already started) to where `fragment` is spoken, estimated from its
        position in the spoken text (Kokoro speaks at an even pace)."""
        v = self.vlines[f'{self.cid}_{i}']
        s = v['spoken']
        k = s.find(fragment)
        if k < 0:
            raise ValueError(f'{fragment!r} not in line {i}')
        start = next(x['start'] for x in self.timeline if x['key'] == v['key'])
        return start + v['seconds'] * k / max(1, len(s))

    def until_word(self, i, fragment):
        self.until(self.at_word(i, fragment))

    # ------------------------------------------------------------------ overlays
    def add(self, layer):
        layer.start(self)
        self.layers.append(layer)
        return layer

    def has_layer(self, cls):
        return any(isinstance(ly, cls) and not ly.done(self.f) for ly in self.layers)

    def lower_third(self, title=None, sub=None, dur=4.5, eyebrow=''):
        meta = getattr(self, 'meta', {}) or {}
        return self.add(gfx.LowerThird(title if title is not None else meta.get('title', ''),
                                       sub if sub is not None else meta.get('menu', ''), dur, eyebrow))

    def chapter(self, num, title):
        """tour/clips compatibility: a lower third."""
        return self.lower_third(title, num, 3.6)

    def callout(self, target, text='', dur=3.0, style='box', dim=0.38, pad=6, side='auto', wait=False):
        ly = self.add(gfx.Callout(target, text, dur, style, dim, pad, side))
        if wait:
            self.hold(dur)
        return ly

    def arrow(self, a, b, text='', dur=2.5, bend=0.18):
        return self.add(gfx.Arrow(a, b, text, dur, bend))

    def key_combo(self, combo, dur=1.8, label='', at=None):
        self.events.append({'kind': 'keys', 'start': round(self.now(), 3), 'combo': combo})
        return self.add(gfx.Keys(combo, dur, at, label))

    def key_badge(self, text):
        return self.key_combo(text)

    def figure(self, label, value, unit='', decimals=None, dur=3.5, src=None, pos='auto', start=0.0, note=''):
        return self.add(gfx.Figure(label, value, unit, decimals, dur, src, pos, start, note))

    def statement(self, text, sub='', dur=3.5, eyebrow='', scrim=0.78):
        """A big line on the left over a dark scrim (openers, summaries over a full-frame view)."""
        return self.add(gfx.Statement(text, sub, dur, eyebrow, scrim))

    def agenda(self, items, title='', dur=None):
        """A numbered list on the left; reveal item i with layer.reveal(r, i)."""
        return self.add(gfx.Agenda(items, title, dur))

    def terminal(self, script, title='bash', backdrop=None):
        """Show a terminal typing `script` ([('cmd', text) | ('out', text[, tone])]) over the stage background (or
        a QImage backdrop) from now on; returns the seconds it needs. r.card = None ends it."""
        f0 = self.f
        bg = backdrop

        def card(rec):
            img = QImage(W, H, QImage.Format.Format_RGB32)
            p = QPainter(img)
            if bg is not None:
                p.drawImage(QRectF(0, 0, W, H), bg)
            else:
                p.fillRect(QRectF(0, 0, W, H), QColor(gfx.BASE))
                gfx.vignette(p, 0.5)
            p.end()
            gfx.terminal(img, (rec.f - f0) / rec.fps, script, title)
            return img
        self.card = card
        return 0.3 + sum(len(it[1]) / 38.0 + 0.45 if it[0] == 'cmd' else 0.12 for it in script)

    def note(self, text, dur=None):
        """A small chip at the top centre; dur None keeps it until note(None)."""
        if text is None:
            for ly in self.layers:
                if isinstance(ly, gfx.Note) and ly.end is None:
                    ly.close(self)
            return None
        return self.add(gfx.Note(text, dur))

    def close_layer(self, layer, after=0.0):
        if layer is not None:
            layer.close(self, after)

    def mark(self, name, **data):
        self.events.append({'kind': 'mark', 'name': name, 'start': round(self.now(), 3), **data})

    def still(self, name):
        """Save the current frame (as composed) to stills/<name>.png (thumbnails, reviews)."""
        self.compose().save(os.path.join(self.out, 'stills', f'{name}.png'))
        self.mark('still', file=f'stills/{name}.png')

    def snap(self, name):                    # tour.Recorder calls this from menu(): no stills there
        pass

    def expect(self, shown, narrated, what=''):
        """Honesty guard: the figure the app shows must be the one the narration says. Raises otherwise."""
        if str(shown).strip() != str(narrated).strip():
            raise AssertionError(f'{what or "figure"}: the app shows {shown!r}, the narration says {narrated!r}')

    # ------------------------------------------------------------------ geometry
    def wpos(self, widget, local=None) -> QPointF:
        """Frame (logical) coordinates of a point in a widget, its centre by default, before any zoom."""
        if local is None:
            local = QPointF(widget.width() / 2, widget.height() / 2)
        g = widget.mapToGlobal(QPointF(local).toPoint())
        o = self.win.mapToGlobal(QPoint(0, 0))
        return QPointF(g.x() - o.x() + self.origin.x(), g.y() - o.y() + self.origin.y())

    def wrect(self, widget) -> QRectF:
        tl = self.wpos(widget, QPointF(0, 0))
        return QRectF(tl.x(), tl.y(), widget.width(), widget.height())

    def target_rect(self, target) -> QRectF | None:
        """A target (widget, rect, tuple or callable) as a rectangle in window coordinates."""
        if callable(target) and not isinstance(target, QWidget):
            target = target()
        if target is None:
            return None
        if isinstance(target, QWidget):
            return self.wrect(target) if target.isVisible() else None
        if isinstance(target, (QRect, QRectF)):
            return QRectF(target)
        if isinstance(target, (tuple, list)) and len(target) == 4:
            return QRectF(*target)
        raise TypeError(f'not a target: {target!r}')

    def to_frame(self, x, y) -> QPointF:
        cx, cy, cw, ch = self.crop
        return QPointF((x - cx) * W / cw, (y - cy) * H / ch)

    def frame_rect(self, target) -> QRectF | None:
        r = self.target_rect(target)
        if r is None:
            return None
        a = self.to_frame(r.left(), r.top())
        b = self.to_frame(r.right(), r.bottom())
        return QRectF(a, b)

    def frame_point(self, target) -> QPointF | None:
        if isinstance(target, (QPoint, QPointF)):
            return self.to_frame(target.x(), target.y())
        if isinstance(target, (tuple, list)) and len(target) == 2:
            return self.to_frame(*target)
        r = self.frame_rect(target)
        return r.center() if r is not None else None

    def zoom_level(self):
        return W / self.crop[2]

    # ------------------------------------------------------------------ the pointer and input
    def ripple(self):
        self.ripples.append((self.cur.copy(), self.f))
        self.events.append({'kind': 'click', 'start': round(self.now(), 3)})

    def send_move(self, widget, pt):
        """A mouse-move (no button) straight to a widget, mouse tracking or not: hover read-outs and highlights."""
        from PyQt6.QtCore import QEvent
        from PyQt6.QtGui import QMouseEvent
        pt = QPointF(pt)
        g = widget.mapToGlobal(pt.toPoint())
        QApplication.sendEvent(widget, QMouseEvent(QEvent.Type.MouseMove, pt, QPointF(g), Qt.MouseButton.NoButton,
                                                   Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier))

    def leave(self, widget):
        """The pointer leaves a widget (its hover read-out or highlight goes)."""
        from PyQt6.QtCore import QEvent
        QApplication.sendEvent(widget, QEvent(QEvent.Type.Leave))

    def glide(self, widget, a, b, dur=1.5, send=True):
        """Move the pointer from a to b (widget coordinates) with mouse-move events, so hover read-outs follow."""
        self.move_to(widget, a, 0.6)
        n = max(2, round(dur * self.fps))
        for k in range(1, n + 1):
            t = smoother(k / n)
            pt = QPointF(a.x() + (b.x() - a.x()) * t, a.y() + (b.y() - a.y()) * t)
            if send:
                self.send_move(widget, pt)
            q = self.wpos(widget, pt)
            self.cur = np.array([q.x(), q.y()])
            self.tick()

    def hover(self, widget, local=None, dur=0.7):
        """Move the pointer onto a point of a widget (its centre by default) with mouse-move events on the way."""
        end = QPointF(local) if local is not None else QPointF(widget.width() / 2, widget.height() / 2)
        g = self.wpos(widget, end)
        start_local = QPointF(end.x() + (self.cur[0] - g.x()), end.y() + (self.cur[1] - g.y()))
        self.glide(widget, start_local, end, dur) if dur > 0 else self.send_move(widget, end)
        self.tick()

    def local_rect(self, widget, rect):
        """A rectangle in a widget's own coordinates, as a target (window coordinates, evaluated when used)."""
        return lambda: QRectF(self.wpos(widget, QRectF(rect).topLeft()), QRectF(rect).size()) if widget.isVisible() else None

    def tree_item(self, tree, prefix, column=0):
        """A QTreeWidget row whose text starts with `prefix` (the Outliner's rows), as a target."""
        def find():
            stack = [tree.topLevelItem(i) for i in range(tree.topLevelItemCount())]
            while stack:
                it = stack.pop(0)
                if it.text(column).startswith(prefix):
                    rr = tree.visualItemRect(it)
                    if rr.isEmpty():
                        return None
                    tl = self.wpos(tree.viewport(), QPointF(0, rr.top()))
                    return QRectF(tl.x(), tl.y(), tree.viewport().width(), rr.height())
                stack += [it.child(k) for k in range(it.childCount())]
            return None
        return find

    def type(self, widget, text, cps=16.0):
        """Type text into a widget at a natural pace."""
        for ch in text:
            QTest.keyClicks(widget, ch)
            self.hold(1.0 / cps * (1.6 if ch == ' ' else 1.0))

    def peek_menu(self, title, *path, hold=1.4):
        """Open a menu-bar menu (and submenus along `path`), rest on the last item, then close it without
        triggering anything: to show what a menu holds."""
        mb = self.win.menuBar()
        m = next(a.menu() for a in mb.actions() if a.menu() and a.text().replace('&', '') == title)
        rr = mb.actionGeometry(m.menuAction())
        self.move_to(mb, QPointF(rr.center()), 0.7)
        self.ripple()
        m.popup(mb.mapToGlobal(rr.bottomLeft()))
        self.hold(0.35)
        for name in path:
            act = next(a for a in m.actions() if a.text().replace('&', '').startswith(name))
            ar = m.actionGeometry(act)
            self.move_to(m, QPointF(ar.center().x() - ar.width() * 0.2, ar.center().y()), 0.55)
            m.setActiveAction(act)
            self.hold(0.25)
            if act.menu() is not None and name != path[-1]:
                act.menu().popup(m.mapToGlobal(ar.topRight()))
                self.hold(0.3)
                m = act.menu()
        self.hold(hold)
        for w in QApplication.topLevelWidgets():
            if isinstance(w, QMenu) and w.isVisible():
                w.hide()
        self.hold(0.2)

    def enter(self, field, text, cps=10.0):
        """Click a text or number field, select what it holds, type `text` and press Enter (the app's spin boxes
        commit on Enter)."""
        le = field.lineEdit() if hasattr(field, 'lineEdit') else field
        self.click(field, None, 0.6, real=True, hold=0.15)
        field.setFocus()
        le.selectAll()
        self.hold(0.2)
        for ch in text:
            QTest.keyClicks(le, ch)
            self.hold(1.0 / cps)
        self.hold(0.2)
        QTest.keyClick(le, Qt.Key.Key_Return)
        self.app.processEvents()
        self.tick()

    def choose(self, combo, item, dur=0.6):
        """Open a combo box, move to an item (index or the start of its text) in its list and pick it."""
        self.click(combo, None, dur, real=False, hold=0.1)
        combo.showPopup()
        self.hold(0.35)
        i = item if isinstance(item, int) else next(k for k in range(combo.count())
                                                     if combo.itemText(k).strip().startswith(item))
        view = combo.view()
        r = view.visualRect(view.model().index(i, 0))
        self.move_to(view.viewport(), QPointF(r.center().x() * 0.6, r.center().y()), 0.55)
        view.setCurrentIndex(view.model().index(i, 0))
        self.hold(0.25)
        self.ripple()
        self.hold(0.12)
        combo.hidePopup()
        combo.setCurrentIndex(i)
        combo.activated.emit(i)
        self.hold(0.2)

    # ------------------------------------------------------------------ zoom
    def zoom(self, target=None, dur=1.0, wait=True, margin=24):
        """Ease the frame to a target (None: the whole frame) with a quintic ease and a zoom that feels even (its
        width changes geometrically, the point being zoomed on stays put)."""
        if target is None:
            c1 = np.array([0.0, 0.0, W, H])
        else:
            r = self.target_rect(target)
            r = r.adjusted(-margin, -margin, margin, margin)
            w = max(r.width(), r.height() * W / H, MIN_CROP_W)
            w = min(w, W)
            h = w * H / W
            cx, cy = r.center().x(), r.center().y()
            c1 = np.array([min(max(0.0, cx - w / 2), W - w), min(max(0.0, cy - h / 2), H - h), w, h])
        if dur <= 0:                                   # a cut: no animation
            self.crop, self._zoom = c1, None
            return
        n = max(1, round(dur * self.fps))
        self._zoom = (self.crop.copy(), c1, self.f, n)
        if wait:
            self.tick(n)

    def zoom_to(self, target=None, dur=1.0, wait=True, margin=24):
        return self.zoom(target, dur, wait, margin)

    # ------------------------------------------------------------------ frames
    def _tops(self):
        win = self.win
        out = []
        for w in QApplication.topLevelWidgets():
            if w is win or not w.isVisible() or w.width() <= 4:
                continue
            kind = w.windowType()
            if isinstance(w, (QMenu, QDialog)) or kind in (Qt.WindowType.Popup, Qt.WindowType.ToolTip):
                out.append(w)
        out.sort(key=lambda w: 0 if isinstance(w, QDialog) else 2 if w.windowType() == Qt.WindowType.ToolTip else 1)
        return out

    def _stage_canvas(self, ss):
        img = QImage(int(W * ss), int(H * ss), QImage.Format.Format_RGB32)
        img.setDevicePixelRatio(ss)
        p = QPainter(img)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        if self.stage_bg is not None:
            p.drawImage(QRectF(0, 0, W, H), self.stage_bg)
        else:
            p.fillRect(QRectF(0, 0, W, H), QColor(gfx.BASE))
            gfx.glow(p, W * 0.5, H * 0.45, W * 0.6, gfx.ACCENT, 0.06)
            gfx.vignette(p, 0.5)
        p.end()
        return img

    def window_image(self) -> QImage:
        """The filmed widget at the device pixel ratio, with its open menus, dialogs, popups and tooltips drawn
        where they are; a widget smaller than the frame sits at self.origin on the stage background."""
        win = self.win
        pm = win.grab()
        ss = pm.devicePixelRatio()
        img = pm.toImage().convertToFormat(QImage.Format.Format_RGB32)
        img.setDevicePixelRatio(ss)
        if win.width() != W or win.height() != H or self.origin != QPointF(0, 0):
            canvas = self._stage_canvas(ss)
            p = QPainter(canvas)
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            o = self.origin
            for k, a in ((40, 0.10), (22, 0.14), (8, 0.22)):
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor(0, 0, 0, int(255 * a)))
                p.drawRoundedRect(QRectF(o.x() - k / 2, o.y() - k / 4, win.width() + k, win.height() + k * 0.8), 12, 12)
            p.drawImage(o, img)
            p.end()
            img = canvas
        p = QPainter(img)
        o = win.mapToGlobal(QPoint(0, 0))
        for w in self._tops():
            g = w.mapToGlobal(QPoint(0, 0))
            pos = QPointF(g.x() - o.x() + self.origin.x(), g.y() - o.y() + self.origin.y())
            for k, a in ((14, 18), (8, 26), (3, 40)):
                p.fillRect(QRectF(pos.x() - k // 2, pos.y() + k // 2, w.width() + k, w.height() + k // 2),
                           QColor(0, 0, 0, a))
            p.drawPixmap(pos, w.grab())
        p.end()
        return img

    def compose(self) -> QImage:
        if self.card is not None:
            img = self.card(self)
            if img.width() != W:
                img = img.scaled(W, H, Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation)
            a = to_np(img)
        else:
            src = self.window_image()
            self.paint_pointer(src)
            a = _crop_resize(to_np(src), self.crop, src.width() / W)
        frosts = []
        live = sorted([ly for ly in self.layers if ly.f0 is not None], key=lambda ly: ly.z)
        for ly in live:
            frosts += ly.frost(self, self.f)
        img = from_np(a)
        p = gfx.painter(img)
        for r, radius, alpha in frosts:
            if alpha > 0.01 and r.width() > 4 and r.height() > 4:
                _frost(p, a, r, radius, alpha)
        for ly in live:
            ly.paint(p, self, self.f)
        p.end()
        return img

    def _write(self, img: QImage):
        a = to_np(img)
        if self.fade is not None:
            src, f0, n = self.fade
            t = (self.f - f0 + 1) / n
            if t >= 1:
                self.fade = None
            else:
                a = gfx.mix_np(src, a, smoother(t))
        self.last = a
        if self.proc is not None:
            if self.out_size != (W, H):
                import cv2
                a = cv2.resize(a, self.out_size, interpolation=cv2.INTER_AREA)
            self.proc.stdin.write(np.ascontiguousarray(a).tobytes())

    def crossfade(self, dur=0.5):
        """Dissolve from the last frame into what follows (a page switch, a new window)."""
        if self.last is not None:
            self.fade = (self.last.copy(), self.f, max(1, round(dur * self.fps)))

    def close(self, meta=None):
        if self.proc is not None:
            self.proc.stdin.close()
            self.proc.wait()
        missing = [k for k in self.dur if k.startswith(f'{self.cid}_') and k not in self._said]
        info = {'id': self.cid, 'fps': self.fps, 'frames': self.f, 'seconds': round(self.now(), 3),
                'wall_seconds': round(time.time() - self.wall0, 1), 'lines': self.timeline, 'events': self.events,
                'unsaid': missing, 'meta': meta or {}}
        with open(os.path.join(self.out, 'timeline.json'), 'w', encoding='utf-8') as fh:
            json.dump(info, fh, indent=1, ensure_ascii=False)
        if missing:
            raise RuntimeError(f'lines never said: {", ".join(missing)}')
        return info


def _crop_between(c0, c1, s):
    """The crop rectangle a fraction s of the way from c0 to c1: width interpolated geometrically, the corner
    linearly in the zoom's own progress (a point both crops share stays fixed on screen)."""
    w0, w1 = c0[2], c1[2]
    if abs(w0 - w1) < 0.5:
        return c0 + (c1 - c0) * s
    w = w0 * (w1 / w0) ** s
    k = (w0 - w) / (w0 - w1)
    x = c0[0] + (c1[0] - c0[0]) * k
    y = c0[1] + (c1[1] - c0[1]) * k
    return np.array([x, y, w, w * H / W])


def _crop_resize(a: np.ndarray, crop, ss: float) -> np.ndarray:
    """The crop (logical coordinates) of a frame grabbed at ss x, as a W x H frame: an exact box filter for the full
    frame, otherwise a sub-pixel affine warp to twice the output size and a box filter down (no shimmer in slow
    zooms)."""
    import cv2
    x, y, w, h = crop
    if w >= W - 0.5 and abs(x) < 0.01 and abs(y) < 0.01:
        if a.shape[1] == W:
            return a
        return cv2.resize(a, (W, H), interpolation=cv2.INTER_AREA)
    k = 2 if ss >= 1.5 else 1
    ow, oh = W * k, H * k
    sx = ow / (w * ss)
    M = np.array([[sx, 0, -x * ss * sx], [0, sx, -y * ss * sx]], np.float32)
    big = cv2.warpAffine(a, M, (ow, oh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return cv2.resize(big, (W, H), interpolation=cv2.INTER_AREA) if k > 1 else big


def _frost(p: QPainter, a: np.ndarray, r: QRectF, radius, alpha):
    """Frosted glass under a panel: the frame blurred inside the panel's rounded rectangle."""
    import cv2
    m = 24
    x0, y0 = max(0, int(r.left()) - m), max(0, int(r.top()) - m)
    x1, y1 = min(a.shape[1], int(r.right()) + m), min(a.shape[0], int(r.bottom()) + m)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return
    region = np.ascontiguousarray(a[y0:y1, x0:x1])
    small = cv2.resize(region, ((x1 - x0) // 2, (y1 - y0) // 2), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 7)
    region = cv2.resize(small, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
    img = from_np(region)
    path = QPainterPath()
    path.addRoundedRect(r, radius, radius)
    p.save()
    p.setClipPath(path)
    p.setOpacity(alpha)
    p.drawImage(QPointF(x0, y0), img)
    p.restore()


# ====================================================================== the stage
class Stage(clips.Stage):
    """The website clips' stage at 1920 x 1080: window(edit) opens the sample, _rig_window(edit) its data rig,
    _nav_window(edit) the data rig with GNSS and IMU; home, tilde() and tilde_path_label() keep machine paths off
    screen. Adds the start window, saved sample projects, chromeless full-frame 3D views and orbits."""

    def __init__(self, app, out, fps=FPS):
        super().__init__(app, out, fps)
        self.extra = []

    def window(self, edit=None):
        w = super().window(edit)
        w.setMinimumSize(0, 0)
        w.resize(W, H)
        self.settle(5)
        self._last_result = w.ctx.analysis.result
        return w

    def close_window(self):
        for x in self.extra:
            try:
                x.hide()
                x.deleteLater()
            except RuntimeError:
                pass
        self.extra = []
        super().close_window()

    # ---------------------------------------------------------------- projects on disk (in the throw-away home)
    def projects_dir(self):
        d = os.path.join(self.home, 'Documents', 'Vantage')
        os.makedirs(d, exist_ok=True)
        return d

    def save_project(self, name, edit=None, thumb_from=None):
        """Save the sample (edited by edit(p)) as ~/Documents/Vantage/<name>.vantage with a thumbnail rendered from
        `thumb_from` (a 3D view) when given; returns the path."""
        from PyQt6.QtCore import QBuffer, QIODevice
        from vantage.core import samples
        p = samples.tigor_sample_project()
        p.photos.source = ''
        if edit is not None:
            edit(p)
        png = b''
        if thumb_from is not None:
            img = thumb_from.grab().toImage().scaled(320, 180, Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                                                     Qt.TransformationMode.SmoothTransformation)
            buf = QBuffer()
            buf.open(QIODevice.OpenModeFlag.WriteOnly)
            img.save(buf, 'PNG')
            png = bytes(buf.data())
        path = os.path.join(self.projects_dir(), f'{name}.vantage')
        p.save(path, png)
        return path

    def start_window(self, recent=()):
        """The start window with these recent projects (paths shown from ~), centred on the stage background."""
        from PyQt6.QtCore import QSettings
        from vantage.ui.dialogs.start import StartWindow
        s = QSettings('Vantage', 'Vantage')
        s.setValue('recent', list(recent))
        sw = StartWindow(s)
        sw.resize(1240, 780)
        for i in range(sw.list.count()):
            it = sw.list.item(i)
            d = dict(it.data(Qt.ItemDataRole.UserRole))
            d['path'] = clips._from_home(d['path'])
            it.setData(Qt.ItemDataRole.UserRole, d)
        sw.show()
        for _ in range(5):
            self.app.processEvents()
        self.extra.append(sw)
        return sw

    def film_centred(self, r, widget):
        """Film a widget smaller than the frame (a dialog on its own), centred on the stage background."""
        r.win = widget
        r.origin = QPointF(round((W - widget.width()) / 2), round((H - widget.height()) / 2))

    def film_window(self, r, w):
        r.win = w
        r.origin = QPointF(0, 0)

    # ---------------------------------------------------------------- cinematic 3D
    def full_view(self, provider, cam_from=None, overlay=False):
        """A chromeless 3D view filling the frame (no axes, no navigation control) on a provider (a pane's
        vp3d.provider, the bowl scene...): for cold opens and backgrounds. Film it with r.win = view."""
        from vantage.ui import viewport3d as V
        v = V.SoftViewport(provider)
        if not overlay:
            v.paint_overlay = lambda p, ui=True: None
        v.resize(W, H)
        v.show()
        if cam_from is not None:
            c = cam_from
            v.cam.target, v.cam.dist, v.cam.fov, v.cam.az, v.cam.el = c.target.copy(), c.dist, c.fov, c.az, c.el
        for _ in range(5):
            self.app.processEvents()
        self.extra.append(v)
        return v

    def fly(self, r, vp, az=None, el=None, dist=None, target=None, dur=1.5):
        """Ease a 3D view's camera to a new azimuth / elevation / distance / target (any of them) over dur seconds,
        recording as it goes (the shorter way round in azimuth)."""
        c = vp.cam
        a0, e0, d0, t0 = c.az, c.el, c.dist, np.array(c.target, float)
        a1 = a0 if az is None else a0 + ((az - a0 + 180.0) % 360.0 - 180.0)
        e1 = e0 if el is None else el
        d1 = d0 if dist is None else dist
        t1 = t0 if target is None else np.array(target, float)
        n = max(1, round(dur * self.fps))
        for k in range(1, n + 1):
            s = smoother(k / n)
            c.az, c.el = a0 + (a1 - a0) * s, e0 + (e1 - e0) * s
            c.dist = d0 * (d1 / d0) ** s
            c.target = t0 + (t1 - t0) * s
            vp.update()
            r.tick()

    def car_box(self, w):
        return self._car_box(w)

    def orbit(self, r, vp, d_az, dur, d_el=0.0, linear=False, d_dist=0.0):
        """Turn a 3D view's camera by d_az (and d_el) degrees over dur seconds, eased at both ends unless linear;
        d_dist moves it in (negative) or out by that fraction of its distance."""
        n = max(1, round(dur * self.fps))
        az0, el0, dist0 = vp.cam.az, vp.cam.el, vp.cam.dist
        for k in range(1, n + 1):
            t = k / n
            s = t if linear else smoother(t)
            vp.cam.az = az0 + d_az * s
            vp.cam.el = el0 + d_el * s
            if d_dist:
                vp.cam.dist = dist0 * (1 + d_dist * s)
            vp.update()
            r.tick()

    def wait_analysis(self, r, timeout=30, since='last'):
        """Record (real time) until the live analysis has caught up with the last edit: a result newer than `since`
        (by default the one the previous wait_analysis saw, or the one current at the window's opening) and idle."""
        a = self.win.ctx.analysis
        old = self._last_result if since == 'last' else since
        r.wait_for(lambda: a.result is not None and a.result is not old and not a._busy and not a.timer.isActive(),
                   timeout=timeout)
        self._last_result = a.result

    def mark_result(self):
        """Remember the current analysis result: the next wait_analysis waits for a newer one."""
        self._last_result = self.win.ctx.analysis.result

    _last_result = None
