"""Record the Vantage demo tour.

Drives the real application offscreen (software 3D renderer), frame by frame, and writes a silent 1920 x 1080 video
plus timeline.json with the start time of every narration line. mix.py adds the voice and subtitles.

    python tour.py VOICE_DIR OUT_DIR [--fps 30] [--preview] [--scenes intro,start,...]

VOICE_DIR is the output of voice.py (durations.json + WAVs). --preview saves a still at every narration line and
skips the video encode.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
# the tour must never touch the user's settings
os.environ['XDG_CONFIG_HOME'] = tempfile.mkdtemp(prefix='vantage-demo-cfg-')

import numpy as np  # noqa: E402
from PyQt6.QtCore import Qt, QPoint, QPointF, QRect, QRectF, QSettings  # noqa: E402
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPainterPath, QPen, QLinearGradient, QPolygonF  # noqa: E402
from PyQt6.QtTest import QTest  # noqa: E402
from PyQt6.QtWidgets import QApplication, QDialog, QMenu, QPushButton, QLabel  # noqa: E402

from narration import SCENES  # noqa: E402

W, H = 1920, 1080
ACCENT = QColor('#3987e5')
GAP = 0.35                      # silence between narration lines, seconds


def ease(t):
    t = min(1.0, max(0.0, t))
    return 4 * t ** 3 if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2


def font(px, weight=QFont.Weight.Normal, mono=False):
    f = QFont('IBM Plex Mono' if mono else 'IBM Plex Sans')
    f.setPixelSize(int(px))
    f.setWeight(weight)
    return f


def to_np(img: QImage) -> np.ndarray:
    img = img.convertToFormat(QImage.Format.Format_RGB32)
    ptr = img.constBits()
    ptr.setsize(img.sizeInBytes())
    a = np.frombuffer(ptr, np.uint8).reshape(img.height(), img.bytesPerLine() // 4, 4)[:, :img.width()]
    return a.copy()


def from_np(a: np.ndarray) -> QImage:
    a = np.ascontiguousarray(a)
    return QImage(a.data, a.shape[1], a.shape[0], a.shape[1] * 4, QImage.Format.Format_RGB32).copy()


# ====================================================================== recorder
class Recorder:
    def __init__(self, app, voice_dir, out_dir, fps=30, preview=False):
        self.app = app
        self.fps = fps
        self.out = out_dir
        self.preview = preview
        info = json.load(open(os.path.join(voice_dir, 'durations.json')))
        self.dur = {k: v['seconds'] for k, v in info['lines'].items()}
        self.captions = {k: v['caption'] for k, v in info['lines'].items()}
        self.f = 0
        self.win = None
        self.cur = np.array([W * 0.62, H * 0.6])
        self.cursor_on = False
        self.ripples = []
        self.crop = np.array([0.0, 0.0, W, H])
        self.tweens = []
        self.lower = None
        self.badge = None
        self.timelapse = 0
        self.fade = None
        self.card = None
        self.backdrop = None
        self.last = None
        self.timeline = []
        self.speech_end = 0.0
        self.scene = ''
        self.proc = None
        os.makedirs(out_dir, exist_ok=True)
        if preview:
            os.makedirs(os.path.join(out_dir, 'stills'), exist_ok=True)
        else:
            self.proc = subprocess.Popen(
                ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgra', '-s', f'{W}x{H}',
                 '-r', str(fps), '-i', '-', '-c:v', 'libx264', '-preset', 'medium', '-crf', '17', '-pix_fmt', 'yuv420p',
                 '-movflags', '+faststart', os.path.join(out_dir, 'video.mp4')], stdin=subprocess.PIPE)

    # ------------------------------------------------------------------ time
    def now(self):
        return self.f / self.fps

    def tick(self, n=1):
        for _ in range(max(0, int(n))):
            self.app.processEvents()
            for tw in list(self.tweens):
                obj, start, end, f0, nf = tw
                t = ease((self.f - f0 + 1) / nf)
                setattr(self, obj, start + (end - start) * t)
                if self.f - f0 + 1 >= nf:
                    self.tweens.remove(tw)
            self._write(self.compose())
            self.f += 1

    def hold(self, sec):
        self.tick(round(sec * self.fps))

    def until(self, t):
        self.hold(max(0.0, t - self.now()))

    def wait_for(self, cond, timeout=180, speed=1):
        """Record while a background job runs. Each frame covers at least speed/fps of wall time (and always a
        little, so the worker threads get the interpreter); when that is faster than real time the measured
        speed-up is shown on screen."""
        t0, f0 = time.time(), self.f
        while not cond() and time.time() - t0 < timeout:
            ts = time.time()
            self.tick()
            time.sleep(max(0.015, speed / self.fps - (time.time() - ts)))
            ratio = (time.time() - t0) / max(1e-6, (self.f - f0) / self.fps)
            self.timelapse = int(round(ratio)) if ratio >= 1.6 else 0
        self.timelapse = 0
        self.tick()

    def tween(self, attr, end, dur, wait=True):
        start = np.array(getattr(self, attr), float)
        n = max(1, round(dur * self.fps))
        self.tweens = [t for t in self.tweens if t[0] != attr]
        self.tweens.append((attr, start, np.array(end, float), self.f, n))
        if wait:
            self.tick(n)

    # ------------------------------------------------------------------ narration
    def say(self, sid, i):
        key = f'{sid}_{i}'
        self.until(self.speech_end + GAP if self.timeline else self.now())
        t = self.now()
        self.timeline.append({'key': key, 'start': round(t, 3), 'seconds': self.dur[key], 'caption': self.captions[key]})
        self.speech_end = t + self.dur[key]
        print(f'{t:7.2f}s  {key}', flush=True)
        if self.preview:
            self.compose().save(os.path.join(self.out, 'stills', f'{len(self.timeline):02d}_{key}.png'))

    def finish_speech(self, pad=0.3):
        self.until(self.speech_end + pad)

    def snap(self, name):
        if self.preview:
            self.compose().save(os.path.join(self.out, 'stills', f'{len(self.timeline):02d}_{name}.png'))

    # ------------------------------------------------------------------ overlays
    def chapter(self, num, title):
        self.lower = (num, title, self.f)

    def key_badge(self, text):
        self.badge = (text, self.f)

    def crossfade(self, dur=0.5):
        if self.last is not None:
            self.fade = (self.last.copy(), self.f, max(1, round(dur * self.fps)))

    # ------------------------------------------------------------------ geometry
    def wpos(self, widget, local=None) -> QPointF:
        """Window coordinates of a point in a widget (its centre by default)."""
        if local is None:
            local = QPointF(widget.width() / 2, widget.height() / 2)
        g = widget.mapToGlobal(QPointF(local).toPoint())
        o = self.win.mapToGlobal(QPoint(0, 0))
        return QPointF(g.x() - o.x(), g.y() - o.y())

    def wrect(self, widget) -> QRectF:
        tl = self.wpos(widget, QPointF(0, 0))
        return QRectF(tl.x(), tl.y(), widget.width(), widget.height())

    # ------------------------------------------------------------------ pointer
    def move(self, target, dur=0.7, wait=True):
        self.cursor_on = True
        if isinstance(target, (QPoint, QPointF)):
            target = (target.x(), target.y())
        d = float(np.linalg.norm(np.asarray(target, float) - self.cur))
        self.tween('cur', target, max(0.25, min(dur, 0.25 + d / 1400)), wait)

    def move_to(self, widget, local=None, dur=0.7):
        self.move(self.wpos(widget, local), dur)

    def ripple(self):
        self.ripples.append((self.cur.copy(), self.f))

    def click(self, widget, local=None, dur=0.7, real=True, hold=0.25):
        self.move_to(widget, local, dur)
        self.tick(3)
        self.ripple()
        if real:
            p = (QPointF(local) if local is not None else QPointF(widget.width() / 2, widget.height() / 2)).toPoint()
            QTest.mouseClick(widget, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, p)
        self.hold(hold)

    def drag(self, widget, a, b, dur=1.0):
        self.move_to(widget, a, 0.6)
        self.tick(3)
        self.ripple()
        QTest.mousePress(widget, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPointF(a).toPoint())
        n = max(2, round(dur * self.fps))
        for k in range(1, n + 1):
            t = ease(k / n)
            p = QPointF(a.x() + (b.x() - a.x()) * t, a.y() + (b.y() - a.y()) * t)
            QTest.mouseMove(widget, p.toPoint())
            self.cur = np.array([self.wpos(widget, p).x(), self.wpos(widget, p).y()])
            self.tick()
        QTest.mouseRelease(widget, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPointF(b).toPoint())
        self.tick(4)

    def menu(self, title, *path, then=None):
        """Open a menu-bar menu, walk to the item, and trigger it."""
        mb = self.win.menuBar()
        m = next(a.menu() for a in mb.actions() if a.menu() and a.text().replace('&', '') == title)
        r = mb.actionGeometry(m.menuAction())
        self.move_to(mb, QPointF(r.center()), 0.7)
        self.ripple()
        m.popup(mb.mapToGlobal(r.bottomLeft()))
        self.hold(0.35)
        self.snap(f'menu_{title}')
        act = None
        for name in path:
            act = next(a for a in m.actions() if a.text().replace('&', '').startswith(name))
            ar = m.actionGeometry(act)
            self.move_to(m, QPointF(ar.center().x() - ar.width() * 0.2, ar.center().y()), 0.55)
            m.setActiveAction(act)
            self.hold(0.25)
            if act.menu() is not None:
                sub = act.menu()
                sub.popup(m.mapToGlobal(ar.topRight()))
                self.hold(0.3)
                m = sub
        self.ripple()
        self.hold(0.15)
        for w in QApplication.topLevelWidgets():
            if isinstance(w, QMenu) and w.isVisible():
                w.hide()
        act.trigger()
        self.hold(0.3)
        return act

    def toolbar(self, action_id, dur=0.7):
        a = self.win.actions_by_id[action_id]
        w = self.win.toolbar.widgetForAction(a)
        self.click(w, dur=dur, real=False)
        a.trigger()
        self.hold(0.2)

    # ------------------------------------------------------------------ zoom
    def zoom(self, rect: QRectF | None, dur=0.8, wait=True, margin=24):
        if rect is None:
            target = np.array([0.0, 0.0, W, H])
        else:
            r = QRectF(rect).adjusted(-margin, -margin, margin, margin)
            w = max(r.width(), r.height() * W / H, 480)
            h = w * H / W
            cx, cy = r.center().x(), r.center().y()
            x = min(max(0, cx - w / 2), W - w)
            y = min(max(0, cy - h / 2), H - h)
            target = np.array([x, y, w, h])
        self.tween('crop', target, dur, wait)

    # ------------------------------------------------------------------ frames
    def window_image(self) -> QImage:
        img = self.win.grab().toImage().convertToFormat(QImage.Format.Format_RGB32)
        if img.width() != W or img.height() != H:
            img = img.scaled(W, H)
        p = QPainter(img)
        o = self.win.mapToGlobal(QPoint(0, 0))
        tops = [w for w in QApplication.topLevelWidgets()
                if w is not self.win and w.isVisible() and isinstance(w, (QMenu, QDialog)) and w.width() > 4]
        tops.sort(key=lambda w: 1 if isinstance(w, QMenu) else 0)
        for w in tops:
            g = w.mapToGlobal(QPoint(0, 0))
            pos = QPoint(g.x() - o.x(), g.y() - o.y())
            for k, a in ((14, 18), (8, 26), (3, 40)):
                p.fillRect(QRect(pos.x() - k // 2, pos.y() + k // 2, w.width() + k, w.height() + k // 2), QColor(0, 0, 0, a))
            p.drawPixmap(pos, w.grab())
        p.end()
        return img

    def paint_pointer(self, img):
        if not self.cursor_on:
            return
        p = QPainter(img)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        for pos, f0 in list(self.ripples):
            t = (self.f - f0) / (0.45 * self.fps)
            if t > 1:
                self.ripples.remove((pos, f0))
                continue
            c = QColor(ACCENT)
            c.setAlphaF(0.75 * (1 - t))
            p.setPen(QPen(c, 2.5))
            c2 = QColor(ACCENT)
            c2.setAlphaF(0.22 * (1 - t))
            p.setBrush(c2)
            r = 6 + 24 * ease(t)
            p.drawEllipse(QPointF(*pos), r, r)
        x, y = self.cur
        s = 1.25
        pts = [(0, 0), (0, 17), (4.2, 13.2), (7.2, 20.2), (10.2, 19), (7.4, 12.3), (12.6, 12.3)]
        shadow = QPolygonF([QPointF(x + 1.6 + u * s, y + 2.2 + v * s) for u, v in pts])
        poly = QPolygonF([QPointF(x + u * s, y + v * s) for u, v in pts])
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(0, 0, 0, 70))
        p.drawPolygon(shadow)
        p.setPen(QPen(QColor('#111111'), 1.3))
        p.setBrush(QColor('#ffffff'))
        p.drawPolygon(poly)
        p.end()

    def paint_overlays(self, img):
        p = QPainter(img)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        if self.lower is not None:
            num, title, f0 = self.lower
            t = (self.f - f0) / self.fps
            dur = 3.6
            if t > dur:
                self.lower = None
            else:
                a = min(1.0, t / 0.35, (dur - t) / 0.35)
                slide = (1 - ease(min(1.0, t / 0.45))) * 24
                p.setOpacity(max(0.0, a))
                x0, y0 = 56 - slide, H - 210
                f1, f2 = font(15, QFont.Weight.DemiBold, mono=True), font(34, QFont.Weight.DemiBold)
                p.setFont(f2)
                tw = p.fontMetrics().horizontalAdvance(title)
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor(18, 18, 17, 225))
                p.drawRoundedRect(QRectF(x0, y0, tw + 64, 92), 6, 6)
                p.fillRect(QRectF(x0, y0, 5, 92), ACCENT)
                p.setFont(f1)
                p.setPen(ACCENT)
                p.drawText(QPointF(x0 + 30, y0 + 32), num)
                p.setFont(f2)
                p.setPen(QColor('#f0efec'))
                p.drawText(QPointF(x0 + 30, y0 + 72), title)
                p.setOpacity(1.0)
        if self.badge is not None:
            text, f0 = self.badge
            t = (self.f - f0) / self.fps
            if t > 1.6:
                self.badge = None
            else:
                p.setOpacity(min(1.0, t / 0.15, (1.6 - t) / 0.3))
                p.setFont(font(26, QFont.Weight.DemiBold, mono=True))
                tw = p.fontMetrics().horizontalAdvance(text)
                r = QRectF(W / 2 - tw / 2 - 22, H - 150, tw + 44, 54)
                p.setPen(QPen(QColor(255, 255, 255, 60), 1))
                p.setBrush(QColor(18, 18, 17, 230))
                p.drawRoundedRect(r, 8, 8)
                p.setPen(QColor('#f0efec'))
                p.drawText(r, Qt.AlignmentFlag.AlignCenter, text)
                p.setOpacity(1.0)
        if self.timelapse:
            p.setFont(font(18, QFont.Weight.DemiBold, mono=True))
            r = QRectF(W - 150, 92, 112, 36)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(18, 18, 17, 220))
            p.drawRoundedRect(r, 6, 6)
            p.setPen(QColor('#fab219'))
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, f'▶▶ ×{self.timelapse}')
        p.end()

    def compose(self) -> QImage:
        if self.card is not None:
            img = self.card(self)
        else:
            img = self.backdrop(self) if self.backdrop is not None else self.window_image()
            self.paint_pointer(img)
            x, y, w, h = self.crop
            if w < W - 0.5:
                img = img.copy(QRect(int(round(x)), int(round(y)), int(round(w)), int(round(h)))).scaled(
                    W, H, Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.paint_overlays(img)
        return img

    def _write(self, img: QImage):
        a = to_np(img)
        if self.fade is not None:
            src, f0, n = self.fade
            t = (self.f - f0 + 1) / n
            if t >= 1:
                self.fade = None
            else:
                k = ease(t)
                a = (src.astype(np.float32) * (1 - k) + a.astype(np.float32) * k).astype(np.uint8)
        self.last = a
        if self.proc is not None:
            self.proc.stdin.write(a.tobytes())

    def close(self):
        if self.proc is not None:
            self.proc.stdin.close()
            self.proc.wait()
        with open(os.path.join(self.out, 'timeline.json'), 'w') as fh:
            json.dump({'fps': self.fps, 'frames': self.f, 'seconds': round(self.now(), 3), 'lines': self.timeline}, fh,
                      indent=1)


# ====================================================================== cards
def blurred(img: QImage, dim=0.38) -> QImage:
    import cv2
    a = to_np(img)
    small = cv2.resize(a, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 6)
    big = cv2.resize(small, (W, H), interpolation=cv2.INTER_CUBIC).astype(np.float32) * dim
    return from_np(np.clip(big, 0, 255).astype(np.uint8))


class Cards:
    def __init__(self, bg: QImage):
        self.bg = blurred(bg)

    def _base(self):
        img = self.bg.copy()
        p = QPainter(img)
        g = QLinearGradient(0, 0, 0, H)
        g.setColorAt(0, QColor(10, 10, 10, 120))
        g.setColorAt(1, QColor(10, 10, 10, 200))
        p.fillRect(QRect(0, 0, W, H), g)
        p.end()
        return img

    def title(self, rec, f0, eyebrow, title, sub, foot=''):
        t = (rec.f - f0) / rec.fps
        img = self._base()
        p = QPainter(img)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing)

        def text(s, f, y, color, delay):
            k = ease((t - delay) / 0.7)
            p.setOpacity(k)
            p.setFont(f)
            p.setPen(color)
            p.drawText(QRectF(0, y + (1 - k) * 18, W, 200), Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, s)
        text(eyebrow.upper(), font(20, QFont.Weight.DemiBold, mono=True), 348, ACCENT, 0.1)
        text(title, font(132, QFont.Weight.DemiBold), 392, QColor('#f0efec'), 0.25)
        k = ease((t - 0.5) / 0.8)
        p.setOpacity(k)
        p.fillRect(QRectF(W / 2 - 160 * k, 578, 320 * k, 2), ACCENT)
        text(sub, font(32), 606, QColor('#c3c2b7'), 0.6)
        if foot:
            text(foot, font(19, mono=True), 900, QColor('#898781'), 1.0)
        p.end()
        return img

    def terminal(self, rec, f0, script):
        """script: [(kind, text)] with kind 'cmd' (typed) or 'out' (printed)."""
        t = (rec.f - f0) / rec.fps
        img = self._base()
        p = QPainter(img)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        r = QRectF(210, 150, W - 420, H - 300)
        for k, a in ((24, 30), (12, 40), (4, 60)):
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(0, 0, 0, a))
            p.drawRoundedRect(r.adjusted(-k / 2, -k / 4, k / 2, k), 14, 14)
        p.setBrush(QColor('#0d0d0c'))
        p.setPen(QPen(QColor('#383835'), 1))
        p.drawRoundedRect(r, 12, 12)
        p.setBrush(QColor('#1a1a19'))
        path = QPainterPath()
        path.addRoundedRect(QRectF(r.x(), r.y(), r.width(), 44), 12, 12)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawPath(path)
        p.fillRect(QRectF(r.x(), r.y() + 30, r.width(), 14), QColor('#1a1a19'))
        for i, c in enumerate(('#e66767', '#fab219', '#0ca30c')):
            p.setBrush(QColor(c))
            p.drawEllipse(QPointF(r.x() + 26 + i * 22, r.y() + 22), 6.5, 6.5)
        p.setFont(font(16, mono=True))
        p.setPen(QColor('#898781'))
        p.drawText(QRectF(r.x(), r.y(), r.width(), 44), Qt.AlignmentFlag.AlignCenter, 'end-of-line station · bash')
        f = font(21, mono=True)
        p.setFont(f)
        lh = 33
        x0, y = r.x() + 34, r.y() + 82
        clock = 0.4
        for item in script:
            kind, s = item[0], item[1]
            tone = item[2] if len(item) > 2 else ''
            if kind == 'cmd':
                n = int(max(0, min(len(s), (t - clock) * 38)))
                if t < clock:
                    break
                p.setPen(QColor('#0ca30c'))
                p.drawText(QPointF(x0, y), '$')
                p.setPen(QColor('#f0efec'))
                p.drawText(QPointF(x0 + 26, y), s[:n])
                if n < len(s):
                    if int(t * 2.2) % 2 == 0:
                        cw = p.fontMetrics().horizontalAdvance(s[:n])
                        p.fillRect(QRectF(x0 + 28 + cw, y - 21, 12, 26), QColor('#f0efec'))
                    break
                clock += len(s) / 38 + 0.45
            else:
                if t < clock:
                    break
                color = {'ok': QColor('#5fc35f'), 'warn': QColor('#fab219'), 'bad': QColor('#ec835a')}.get(tone, QColor('#c3c2b7'))
                p.setPen(color)
                p.drawText(QPointF(x0, y), s)
                clock += 0.12
            y += lh
        p.end()
        return img


# ====================================================================== the tour
class Tour:
    def __init__(self, rec, workdir):
        self.r = rec
        self.work = workdir
        from vantage.core import samples
        from vantage.ui.mainwindow import MainWindow
        self.project = samples.tigor_sample_project()
        # a saved copy so the start window lists a real recent project
        self.saved = os.path.join(workdir, 'tigor-ev-svm.vantage')
        self.win = MainWindow(self.project, soft_3d=True)
        rec.win = self.win
        self.win.resize(W, H)
        self.win.show()
        self.settle(60)
        vp = self.vp
        vp.cam.az, vp.cam.el = -42.0, 24.0
        self.win.viewarea.frame_all()
        self.settle(2)
        thumb = self.vp.grab().toImage().scaled(320, 180, Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                                                Qt.TransformationMode.SmoothTransformation)
        from PyQt6.QtCore import QBuffer, QIODevice
        buf = QBuffer()
        buf.open(QIODevice.OpenModeFlag.WriteOnly)
        thumb.save(buf, 'PNG')
        self.project.save(self.saved, bytes(buf.data()))
        self.win.add_recent(self.saved)
        self.win.store.undo.setClean()
        self.cards = Cards(self.win.grab().toImage())

    @property
    def vp(self):
        return self.win.viewarea.panes[0].vp3d

    def settle(self, timeout=20):
        app = self.r.app
        t = time.time()
        a = self.win.ctx.analysis
        while time.time() - t < timeout:
            for _ in range(5):
                app.processEvents()
                time.sleep(0.01)
            if a.result is not None and not a._busy and not a.timer.isActive():
                break

    def orbit(self, d_az, dur, d_el=0.0):
        vp = self.vp
        n = max(1, round(dur * self.r.fps))
        az0, el0 = vp.cam.az, vp.cam.el
        for k in range(1, n + 1):
            t = k / n
            s = t if 0.15 < t < 0.85 else ease(t)
            vp.cam.az = az0 + d_az * s
            vp.cam.el = el0 + d_el * s
            vp.update()
            self.r.tick()

    def page_reset(self):
        w = self.win
        if w.ctx.shading != 'solid':
            w.set_shading('solid')
            w.actions_by_id['shade_solid'].setChecked(True)
        if w.page_name() != 'viewports':
            w.set_page('viewports')
        if w.viewarea.mode != 'single':
            w.set_split('single')
        pane = w.viewarea.active
        if pane.kind != 'persp':
            pane.set_kind('persp')
        self.r.app.processEvents()

    # ------------------------------------------------------------------ scenes
    def intro(self):
        r = self.r
        f0 = r.f
        r.card = lambda rec: self.cards.title(rec, f0, 'Surround-view camera planner', 'Vantage',
                                              'Plan  ·  Analyse  ·  Check as built',
                                              'Tata Tigor EV  ·  six STURDeCAM31 fisheye cameras  ·  Jetson AGX Orin')
        r.hold(0.8)
        r.say('intro', 0)
        r.hold(2.5)
        r.snap('title')
        r.finish_speech()
        r.say('intro', 1)
        r.finish_speech(0.5)

    def start(self):
        r = self.r
        from vantage.ui.dialogs.start import StartWindow
        sw = StartWindow(QSettings('Vantage', 'Vantage'))
        sw.resize(1240, 780)
        sw.show()
        r.app.processEvents()

        def backdrop(rec):
            img = self.cards._base()
            p = QPainter(img)
            pos = QPoint((W - sw.width()) // 2, (H - sw.height()) // 2)
            for k, a in ((28, 26), (14, 36), (4, 60)):
                p.fillRect(QRect(pos.x() - k // 2, pos.y() + k // 3, sw.width() + k, sw.height() + k // 2), QColor(0, 0, 0, a))
            p.drawPixmap(pos, sw.grab())
            p.end()
            return img
        self.sw_origin = QPoint((W - sw.width()) // 2, (H - sw.height()) // 2)
        r.card = None
        r.backdrop = backdrop
        r.crossfade(0.6)
        r.cursor_on = True
        r.cur = np.array([W * 0.55, H * 0.75])
        r.say('start', 0)

        def sw_pos(widget, local=None):
            g = widget.mapTo(sw, QPoint(widget.width() // 2, widget.height() // 2) if local is None else local)
            return QPointF(self.sw_origin.x() + g.x(), self.sw_origin.y() + g.y())
        r.hold(0.4)
        r.move(sw_pos(sw.list, QPoint(220, 40)), 0.9)
        r.hold(0.8)
        tigor = next(b for b in sw.findChildren(QPushButton)
                     if any('Tigor' in l.text() for l in b.findChildren(QLabel)))
        r.move(sw_pos(tigor), 0.9)
        r.finish_speech(0.1)
        r.ripple()
        r.hold(0.35)
        sw.close()
        r.backdrop = None
        r.crossfade(0.7)
        r.cur = np.array([W * 0.5, H * 0.5])
        r.cursor_on = False
        r.say('start', 1)
        self.orbit(55, r.dur['start_1'] + 0.2, d_el=4)
        r.say('start', 2)
        r.cursor_on = True
        r.move(r.wpos(self.win.outliner, QPointF(120, 130)), 0.8)
        r.zoom(r.wrect(self.win.docks['outliner']), 0.9)
        r.hold(1.3)
        r.zoom(None, 0.7)
        r.move(r.wpos(self.win.docks['inspector'], QPointF(150, 120)), 0.9)
        r.zoom(r.wrect(self.win.docks['inspector']), 0.9)
        r.finish_speech(0.0)
        r.zoom(None, 0.7)

    def coverage(self):
        r = self.r
        w = self.win
        self.page_reset()
        r.chapter('01', 'Live coverage analysis')
        r.say('coverage', 0)
        r.cursor_on = False
        self.orbit(-70, r.dur['coverage_0'] * 0.9, d_el=10)
        r.cursor_on = True
        r.say('coverage', 1)
        r.menu('View', 'Overlays', 'Heatmap', 'Density')
        r.hold(1.6)
        r.menu('View', 'Overlays', 'Heatmap', 'Redundancy')
        r.finish_speech(0.4)
        r.say('coverage', 2)
        w.bottom.show_tab('coverage')
        rect = r.wrect(w.docks['bottom'])
        r.move(QPointF(rect.x() + 220, rect.y() + 110), 0.8)
        r.zoom(QRectF(rect.x(), rect.y(), rect.width() * 0.48, rect.height()), 1.0)
        r.finish_speech(0.4)
        r.zoom(None, 0.8)
        w.set_heatmap('none')
        w.actions_by_id['heat_none'].setChecked(True)

    def edit(self):
        r = self.r
        w = self.win
        st = w.store
        self.page_reset()
        r.chapter('02', 'Placing cameras')
        ml = st.variant.by_tag('ML')
        r.say('edit', 0)
        from vantage.ui.camview import CamImageView  # noqa: F401  (imported for its side effects on first use)
        s, _, ok = self.vp.project([ml.pos()])
        r.click(self.vp, QPointF(*s[0]), 0.9)
        self.settle(3)
        r.finish_speech(0.1)
        r.say('edit', 1)
        page = w.inspector.pages['camera']
        field = page.rp if hasattr(page, 'rp') else None
        insp = r.wrect(w.docks['inspector'])
        if field is not None:
            r.move_to(field, None, 0.8)
        r.zoom(QRectF(insp.x() - 900, insp.y(), 900 + insp.width(), insp.height() * 0.62), 0.9, wait=False)
        p0 = st.cam(ml.uid).pitch
        n = round(r.dur['edit_1'] * r.fps * 0.9)
        st.dragging = False
        for k in range(1, n + 1):
            t = k / n
            v = p0 + 18 * math.sin(math.pi * t) * (1 if t < 1 else 0)
            st.set_cam(ml.uid, merge=True, pitch=round(v, 1), text='Pitch')
            if field is not None:
                c = r.wpos(field)
                r.cur = np.array([c.x() + 60 * math.sin(math.pi * t), c.y()])
            r.tick()
        st.set_cam(ml.uid, merge=True, pitch=p0, text='Pitch')
        self.settle(3)
        r.finish_speech(0.2)
        r.zoom(None, 0.6, wait=False)
        r.say('edit', 2)
        r.toolbar('split4')
        self.settle(3)
        r.hold(0.4)
        r.snap('quad')
        r.cursor_on = False
        r.finish_speech(0.5)
        r.cursor_on = True

    def seams(self):
        r = self.r
        w = self.win
        self.page_reset()
        pane = w.viewarea.active
        pane.set_kind('bev')
        r.crossfade(0.4)
        self.settle(3)
        plan = pane.plan
        plan.fit()
        r.tick(2)
        r.chapter('03', "Bird's-eye view and seams")
        r.say('seams', 0)
        r.hold(2.4)
        res = w.ctx.analysis.result
        s0 = max(res.seams, key=lambda s: abs(math.sin(math.radians(s.angle))))
        a_, b_ = plan._seam_segment(res, s0.angle)
        a = plan.w2s(*b_)
        b = QPointF(a.x() + 70, a.y())
        r.drag(plan, a, b, 1.3)
        self.settle(3)
        r.snap('seam_drag')
        r.finish_speech(0.3)
        r.say('seams', 1)
        w.set_page('compare')
        r.crossfade(0.5)
        self.settle(3)
        r.move(QPointF(W * 0.52, H * 0.45), 1.0)
        r.finish_speech(0.4)

    def tolerance(self):
        r = self.r
        w = self.win
        r.chapter('04', 'Tolerance study')
        r.say('tolerance', 0)
        r.menu('Analysis', 'Tolerance study')
        pg = w.page('tolerance')
        pg.samples.set_value(800)
        r.hold(0.5)
        r.click(pg.b_run, real=False)
        pg.run()
        r.wait_for(lambda: 'tolerance' not in w._jobs, timeout=240, speed=2)
        r.hold(0.4)
        r.snap('tolerance_done')
        r.say('tolerance', 1)
        hr, tr = r.wrect(pg.hist), r.wrect(pg.tornado)
        r.zoom(hr.united(tr), 0.9, wait=False)
        r.move_to(pg.hist, None, 0.8)
        r.hold(1.0)
        r.move_to(pg.tornado, None, 0.8)
        r.hold(1.0)
        r.zoom(None, 0.7, wait=False)
        if pg.table.rowCount() > 1:
            btn = pg.table.cellWidget(1, 5)
            if btn is not None:
                r.click(btn)
        r.finish_speech(0.5)

    def optimiser(self):
        r = self.r
        w = self.win
        w.set_page('optimiser')
        r.crossfade(0.5)
        pg = w.page('optimiser')
        pg.gens.set_value(12)
        pg.pop.set_value(28)
        r.chapter('05', 'Placement optimiser')
        r.say('optimiser', 0)
        r.click(pg.b_run, real=True)
        r.wait_for(lambda: 'optimiser' not in w._jobs, timeout=400, speed=4)
        r.snap('optimiser_done')
        r.finish_speech(0.6)

    def targets(self):
        r = self.r
        w = self.win
        w.menu_target = None
        r.chapter('06', 'Calibration targets')
        r.say('targets', 0)
        r.menu('Calibrate', 'Target planner')
        self.settle(3)
        r.move(QPointF(W * 0.45, H * 0.4), 1.0)
        r.finish_speech(0.5)

    def install(self):
        r = self.r
        w = self.win
        r.chapter('07', 'Installation check')
        r.say('install', 0)
        r.menu('Calibrate', 'Installation check')
        pg = w.page('install')
        self.settle(3)
        r.finish_speech(0.2)
        r.say('install', 1)
        r.move_to(pg.rows[w.store.variant.cameras[0].uid][0], None, 0.8)
        r.hold(0.6)
        r.move_to(pg.sim_ang, None, 0.8)
        r.hold(0.8)
        r.click(pg.b_run, real=True)
        r.wait_for(lambda: not pg.busy and bool(pg.results), timeout=240, speed=1)
        r.finish_speech(0.2)
        r.say('install', 2)
        r.zoom(r.wrect(pg.table), 1.0)
        r.move_to(pg.table, QPointF(pg.table.width() * 0.55, 60), 0.8)
        r.snap('install_table')
        r.finish_speech(0.4)
        r.zoom(None, 0.7)
        r.say('install', 3)
        bev = r.wrect(pg.bev)
        r.zoom(QRectF(bev.x() - 20, bev.y() - 40, bev.width() + 40, bev.height() + 60), 1.0)
        r.click(pg.which.buttons['planned'])
        r.hold(1.8)
        r.snap('install_planned')
        r.click(pg.which.buttons['measured'])
        r.hold(0.5)
        r.snap('install_measured')
        r.finish_speech(0.4)
        r.zoom(None, 0.7)
        r.say('install', 4)
        r.click(pg.b_apply)
        r.hold(0.6)
        r.move(r.wpos(w.var_combo) if hasattr(w, 'var_combo') else QPointF(W - 380, 41), 0.9)
        r.finish_speech(0.5)

    def loads(self):
        r = self.r
        w = self.win
        st = w.store
        if st.variant.name.endswith('as-built'):
            st.undo.undo()
            self.settle(5)
        w.set_page('viewports')
        r.crossfade(0.5)
        self.page_reset()
        r.chapter('08', 'Load cases and link budget')
        r.say('loads', 0)
        r.menu('Analysis', 'Load cases')
        d = getattr(w, '_load_dialog', None)
        r.hold(0.3)
        r.click(d.b_run, real=False)
        d.run()
        r.wait_for(lambda: d.results is not None, timeout=120)
        rr = r.wrect(d)
        r.zoom(rr, 0.9)
        r.snap('loads_dialog')
        r.finish_speech(0.4)
        r.zoom(None, 0.6)
        d.close()
        r.hold(0.2)
        r.say('loads', 1)
        ol = w.outliner
        from vantage.ui.outliner import ROLE_KIND
        it = None
        for i in range(ol.topLevelItemCount()):
            top = ol.topLevelItem(i)
            if top.data(0, ROLE_KIND) == 'ecu':
                it = top
        if it is not None:
            rect = ol.visualItemRect(it)
            r.click(ol.viewport(), QPointF(rect.center().x(), rect.center().y()), real=False)
        st.select_obj(('ecu',))
        self.settle(2)
        r.zoom(r.wrect(w.docks['inspector']), 1.0)
        r.move_to(w.inspector.pages['ecu'].bt, None, 0.8)
        r.snap('ecu')
        r.finish_speech(0.5)
        r.zoom(None, 0.6)

    def photo(self):
        r = self.r
        w = self.win
        r.chapter('09', 'Photo-matched view')
        r.say('photo', 0)
        r.menu('View', 'Shading', 'Photo-matched')
        r.cursor_on = False
        r.hold(2.0)
        r.snap('photo')
        r.key_badge(']')
        w.photo_step(1)
        r.crossfade(0.35)
        r.hold(1.8)
        r.key_badge(']')
        w.photo_step(1)
        r.crossfade(0.35)
        r.finish_speech(0.5)
        r.cursor_on = True

    def export(self):
        r = self.r
        w = self.win
        w.set_view('persp')
        w.viewarea.frame_all()
        r.crossfade(0.4)
        r.chapter('10', 'Export')
        r.say('export', 0)
        r.key_badge('Ctrl+E')
        from vantage.ui.dialogs.export_dialog import ExportDialog
        d = ExportDialog(w.ctx, w)
        d.show()
        r.hold(0.3)
        r.zoom(r.wrect(d), 0.9)
        for fid in ('extrinsics_json', 'opencv_yaml', 'ros2', 'jetson_json', 'bev_lut', 'mount_sheet_pdf'):
            row = d.rows[fid]
            r.click(row, QPointF(row.width() * 0.35, row.height() / 2), 0.7, real=False)
            d.select(fid)
            r.hold(min(1.6, max(0.8, r.dur['export_0'] / 8)))
            r.snap(f'export_{fid}')
        r.finish_speech(0.3)
        r.zoom(None, 0.5)
        d.close()
        r.hold(0.2)
        r.say('export', 1)
        f0 = r.f
        script = self.batch_script
        r.cursor_on = False
        r.card = lambda rec: self.cards.terminal(rec, f0, script)
        r.crossfade(0.5)
        total = 0.4 + sum(len(it[1]) / 38 + 0.45 if it[0] == 'cmd' else 0.12 for it in script)
        r.hold(max(total + 1.5, r.dur['export_1'] + 0.5))
        r.snap('terminal')
        r.finish_speech(0.6)

    def outro(self):
        r = self.r
        f0 = r.f
        r.card = lambda rec: self.cards.title(rec, f0, 'Surround-view camera planner', 'Vantage',
                                              'Plan it.  Prove it.  Check it as built.',
                                              'Live coverage  ·  Tolerance & optimiser  ·  Installation check  ·  '
                                              "Bird's-eye remap tables  ·  15 export formats")
        r.crossfade(0.8)
        r.hold(0.6)
        r.say('outro', 0)
        r.finish_speech(2.4)
        r.snap('outro_end')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('voice')
    ap.add_argument('out')
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--preview', action='store_true')
    ap.add_argument('--scenes', default='')
    ap.add_argument('--batch-script', help='JSON [[kind, text], ...] for the terminal scene')
    a = ap.parse_args()
    from vantage.ui.viewport3d import configure_default_format
    configure_default_format()
    app = QApplication(sys.argv[:1])
    app.setApplicationName('Vantage')
    app.setOrganizationName('Vantage')
    from vantage.ui.theme import THEME, load_fonts
    load_fonts()
    THEME.apply(app, 'dark')
    rec = Recorder(app, a.voice, a.out, a.fps, a.preview)
    work = tempfile.mkdtemp(prefix='vantage-demo-')
    tour = Tour(rec, work)
    tour.batch_script = [tuple(x) for x in json.load(open(a.batch_script))] if a.batch_script else [
        ('cmd', 'vantage tigor-ev-svm.vantage --install-check captures/ --report check.json'), ('out', 'wrote check.json')]
    order = [s[0] for s in SCENES]
    want = [s for s in a.scenes.split(',') if s] or order
    t0 = time.time()
    for sid in order:
        if sid in want:
            print(f'--- {sid}', flush=True)
            getattr(tour, sid)()
    rec.close()
    print(f'recorded {rec.now():.1f} s ({rec.f} frames) in {time.time() - t0:.0f} s', flush=True)
    tour.win.store.undo.setClean()
    for obj in (tour.win.store.undo, tour.win.store, tour.win.ctx.analysis, tour.win.ctx):
        obj.blockSignals(True)
    os._exit(0)


if __name__ == '__main__':
    main()
