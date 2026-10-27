"""The tour's motion graphics: one visual language for every overlay and card.

Drawn with QPainter on 1920 x 1080 frames, in IBM Plex Sans / Mono (the app's own fonts) and the app's dark theme tokens,
with ONE accent colour (the app's blue, camera 1 and the density scale). Restrained on purpose: short eased fades and
slides, no bounces, no overshoot.

Two kinds of things live here:

* overlay layers drawn over the recorded app (studio.Rec adds them): LowerThird, Callout, Arrow, Keys, Figure, Note,
  Badge and the persistent chapter Indicator;
* full-frame cards painted without the app (assemble.py and the intro/outro chapters use them): chapter_card,
  part_card, logo_reveal, end_card, plus the frame helpers (blur, dip) the transitions use.

Every layer is a function of its own clock (frames since it started), so a recording is deterministic.
"""
from __future__ import annotations

import math
import os

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import (QBrush, QColor, QFont, QFontMetricsF, QImage, QLinearGradient, QPainter, QPainterPath, QPen,
                         QRadialGradient)

W, H = 1920, 1080
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ICON = os.path.join(ROOT, 'vantage', 'resources', 'icons', 'vantage.png')

# the app's dark theme tokens (vantage/ui/theme.py TOKENS['dark']) and its blue
BASE = '#121211'
PANEL = '#1a1a19'
RAISED = '#222220'
HAIR = '#2c2c2a'
STRONG = '#383835'
INK = '#f0efec'
INK2 = '#c3c2b7'
MUTED = '#898781'
ACCENT = '#3987e5'
ACCENT_SOFT = '#86b6ef'

VERSION = '0.1.1'
SITE = 'sherin-sef-ai.github.io/Vantage'
RELEASES = 'github.com/Sherin-SEF-AI/Vantage/releases'
REPO = 'github.com/Sherin-SEF-AI/Vantage'
TAGLINE = 'Sensor rig planner for autonomous-driving data collection'
LICENCE = 'GPL-3.0-or-later'


def col(hexs, a=1.0) -> QColor:
    c = QColor(hexs)
    c.setAlphaF(max(0.0, min(1.0, a)))
    return c


# ====================================================================== easing (all overshoot-free)
def clamp(t):
    return 0.0 if t <= 0 else 1.0 if t >= 1 else float(t)


def smoother(t):
    """Quintic smootherstep: zero velocity and acceleration at both ends (camera moves, zooms)."""
    t = clamp(t)
    return t * t * t * (t * (6 * t - 15) + 10)


def out_cubic(t):
    t = clamp(t)
    return 1 - (1 - t) ** 3


def out_quint(t):
    t = clamp(t)
    return 1 - (1 - t) ** 5


def in_cubic(t):
    t = clamp(t)
    return t ** 3


def in_out(t):
    t = clamp(t)
    return 4 * t ** 3 if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2


def window(t, t0, d):
    """0..1 progress of a step that starts at t0 and lasts d (seconds)."""
    return clamp((t - t0) / d) if d > 0 else float(t >= t0)


# ====================================================================== type
def font(px, weight='regular', mono=False, spacing=0.0) -> QFont:
    f = QFont('IBM Plex Mono' if mono else 'IBM Plex Sans')
    f.setPixelSize(int(round(px)))
    f.setWeight({'regular': QFont.Weight.Normal, 'medium': QFont.Weight.Medium,
                 'semibold': QFont.Weight.DemiBold}[weight])
    if spacing:
        f.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, spacing)
    f.setHintingPreference(QFont.HintingPreference.PreferNoHinting)
    return f


def text_w(f: QFont, s: str) -> float:
    return QFontMetricsF(f).horizontalAdvance(s)


def draw_text(p: QPainter, x, y, s, f, color, alpha=1.0):
    """Text with its baseline at y."""
    p.setFont(f)
    p.setPen(col(color, alpha) if isinstance(color, str) else color)
    p.drawText(QPointF(x, y), s)


def painter(img: QImage) -> QPainter:
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
    p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
    return p


# ====================================================================== frames as numpy (BGRA, H x W x 4)
def to_np(img: QImage) -> np.ndarray:
    img = img.convertToFormat(QImage.Format.Format_RGB32)
    ptr = img.constBits()
    ptr.setsize(img.sizeInBytes())
    a = np.frombuffer(ptr, np.uint8).reshape(img.height(), img.bytesPerLine() // 4, 4)[:, :img.width()]
    return a.copy()


def from_np(a: np.ndarray) -> QImage:
    a = np.ascontiguousarray(a)
    return QImage(a.data, a.shape[1], a.shape[0], a.shape[1] * 4, QImage.Format.Format_RGB32).copy()


def blur_np(a: np.ndarray, sigma: float) -> np.ndarray:
    """A soft Gaussian blur, computed at a quarter size (cheap at 1080p) when sigma is large."""
    import cv2
    if sigma <= 0.3:
        return a
    h, w = a.shape[:2]
    k = 4 if sigma >= 8 else 2 if sigma >= 3 else 1
    small = cv2.resize(a, (w // k, h // k), interpolation=cv2.INTER_AREA) if k > 1 else a
    small = cv2.GaussianBlur(small, (0, 0), sigma / k)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR) if k > 1 else small


def base_np(h=H, w=W) -> np.ndarray:
    a = np.zeros((h, w, 4), np.uint8)
    c = QColor(BASE)
    a[..., 0], a[..., 1], a[..., 2], a[..., 3] = c.blue(), c.green(), c.red(), 255
    return a


def mix_np(a: np.ndarray, b: np.ndarray, k: float) -> np.ndarray:
    if k <= 0:
        return a
    if k >= 1:
        return b
    return (a.astype(np.float32) * (1 - k) + b.astype(np.float32) * k + 0.5).astype(np.uint8)


def dip(a: np.ndarray, k: float) -> np.ndarray:
    """The transition family: towards the theme's background colour (k = 0 the frame, 1 the background)."""
    return mix_np(a, base_np(a.shape[0], a.shape[1]), smoother(k))


class Backdrop:
    """A frame as a card background: blurred and darkened, with the blur levels precomputed so a card can dissolve
    from the blurred frame to the sharp one."""
    LEVELS = (28.0, 14.0, 7.0, 3.5, 0.0)

    def __init__(self, frame: np.ndarray | None):
        self.frame = frame if frame is not None else base_np()
        self.levels = [blur_np(self.frame, s) for s in self.LEVELS]

    def at(self, sharp: float, dim: float) -> np.ndarray:
        """sharp 0 = most blurred .. 1 = the frame itself; dim 0 = as is .. 1 = the background colour."""
        x = clamp(sharp) * (len(self.levels) - 1)
        i = min(int(x), len(self.levels) - 2)
        img = mix_np(self.levels[i], self.levels[i + 1], x - i)
        return mix_np(img, base_np(img.shape[0], img.shape[1]), dim) if dim > 0 else img


def vignette(p: QPainter, strength=0.55, w=W, h=H):
    g = QRadialGradient(QPointF(w * 0.42, h * 0.46), w * 0.75)
    g.setColorAt(0.0, col(BASE, 0.0))
    g.setColorAt(1.0, col('#000000', strength))
    p.fillRect(QRectF(0, 0, w, h), QBrush(g))


def glow(p: QPainter, cx, cy, r, color=ACCENT, a=0.10):
    g = QRadialGradient(QPointF(cx, cy), r)
    g.setColorAt(0.0, col(color, a))
    g.setColorAt(1.0, col(color, 0.0))
    p.fillRect(QRectF(cx - r, cy - r, 2 * r, 2 * r), QBrush(g))


def panel(p: QPainter, r: QRectF, a=1.0, radius=10.0, fill=None, border=True, shadow=True):
    """The tour's panel: near-black glass with a hairline border and a soft shadow."""
    if shadow:
        p.setPen(Qt.PenStyle.NoPen)
        for k, al in ((18, 0.10), (10, 0.12), (4, 0.16)):
            p.setBrush(col('#000000', al * a))
            p.drawRoundedRect(r.adjusted(-k / 2, -k / 4, k / 2, k * 0.75), radius + k / 2, radius + k / 2)
    p.setBrush(fill if fill is not None else col('#141413', 0.84 * a))
    p.setPen(QPen(col('#ffffff', 0.09 * a), 1.0) if border else Qt.PenStyle.NoPen)
    p.drawRoundedRect(r, radius, radius)


# ====================================================================== overlay layers
class Layer:
    """A timed overlay. dur None = until close(). t_in/t_out: the length of its entrance and exit."""
    z = 10
    t_in = 0.45
    t_out = 0.35

    def __init__(self, dur):
        self.dur = dur
        self.f0 = None
        self.end = None             # frame the exit finishes on
        self.fps = 30

    def start(self, rec):
        self.f0, self.fps = rec.f, rec.fps
        if self.dur is not None:
            self.end = rec.f + max(1, round((self.dur + 0.0) * rec.fps))
        return self

    def close(self, rec, after=0.0):
        """Begin the exit `after` seconds from now."""
        self.end = rec.f + round((after + self.t_out) * rec.fps)

    def t(self, f):
        return (f - self.f0) / self.fps

    def a_in(self, f):
        return out_cubic(self.t(f) / self.t_in) if self.t_in > 0 else 1.0

    def a_out(self, f):
        """1 while shown, falling to 0 over t_out before end."""
        if self.end is None:
            return 1.0
        left = (self.end - f) / self.fps
        return clamp(left / self.t_out) if self.t_out > 0 else float(left > 0)

    def done(self, f):
        return self.end is not None and f >= self.end

    def frost(self, rec, f):
        """[(QRectF in frame coordinates, radius, alpha)] to frost (blur) under this layer before it paints."""
        return []

    def paint(self, p: QPainter, rec, f):
        raise NotImplementedError


class LowerThird(Layer):
    """Bottom-left title: an accent bar draws up, the panel opens to the right, the title and the menu path rise in."""
    z = 30
    t_out = 0.4

    def __init__(self, title, sub='', dur=4.5, eyebrow=''):
        super().__init__(dur)
        self.title, self.sub, self.eyebrow = title, sub, eyebrow
        self.f_title = font(38, 'semibold')
        self.f_sub = font(18, 'regular', mono=True)
        self.f_eye = font(14, 'semibold', mono=True, spacing=2.0)
        tw = max(text_w(self.f_title, title), text_w(self.f_sub, sub), text_w(self.f_eye, eyebrow.upper()))
        self.w = tw + 74
        self.h = 112 if (sub or eyebrow) else 78
        if sub and eyebrow:
            self.h = 136
        self.x0, self.y0 = 64, H - 112 - self.h

    def _rect(self, f):
        t = self.t(f)
        o = 1 - self.a_out(f)
        open_k = out_quint(window(t, 0.12, 0.55))
        dx = -14 * in_cubic(o)
        return QRectF(self.x0 + dx, self.y0, max(6.0, self.w * open_k), self.h), o

    def frost(self, rec, f):
        r, o = self._rect(f)
        return [(r, 8.0, (1 - o) * out_cubic(window(self.t(f), 0.1, 0.3)))]

    def paint(self, p, rec, f):
        t = self.t(f)
        r, o = self._rect(f)
        a = 1 - o
        p.save()
        p.setOpacity(a)
        panel(p, r, a=out_cubic(window(t, 0.1, 0.3)), radius=8, fill=col('#121211', 0.72), shadow=True)
        bar_k = out_cubic(window(t, 0.0, 0.32))
        p.fillRect(QRectF(r.x(), r.y() + r.height() * (1 - bar_k), 4, r.height() * bar_k), col(ACCENT))
        p.setClipRect(r)
        x = r.x() + 34
        y = r.y() + 18
        if self.eyebrow:
            k = out_cubic(window(t, 0.22, 0.45))
            draw_text(p, x, y + 16 + (1 - k) * 10, self.eyebrow.upper(), self.f_eye, ACCENT, k)
            y += 26
        k = out_cubic(window(t, 0.2, 0.5))
        draw_text(p, x, y + 38 + (1 - k) * 14, self.title, self.f_title, INK, k)
        if self.sub:
            k = out_cubic(window(t, 0.32, 0.5))
            draw_text(p, x, y + 74 + (1 - k) * 10, self.sub, self.f_sub, INK2, k)
        p.restore()


def _round_rect_path(r: QRectF, radius) -> QPainterPath:
    path = QPainterPath()
    path.addRoundedRect(r, radius, radius)
    return path


class Callout(Layer):
    """Highlights a widget or rectangle: the rest of the frame dims a little, an accent outline draws on with a soft
    glow, and an optional label sits beside it on a leader line. `target` is resolved every frame (rec.target_rect),
    so it follows zooms and widgets that move."""
    z = 20
    t_in = 0.55
    t_out = 0.35

    def __init__(self, target, text='', dur=3.0, style='box', dim=0.38, pad=6, side='auto'):
        super().__init__(dur)
        self.target, self.text, self.style, self.dim, self.pad, self.side = target, text, style, dim, pad, side
        self.f_label = font(21, 'semibold')

    def _geom(self, rec):
        r = rec.frame_rect(self.target)
        if r is None:
            return None
        return r.adjusted(-self.pad, -self.pad, self.pad, self.pad)

    def _label_rect(self, r: QRectF):
        tw = text_w(self.f_label, self.text)
        lw, lh = tw + 44, 46
        gap = 34
        side = self.side
        if side == 'auto':
            room = {'above': r.top() - 70, 'below': H - 120 - r.bottom(), 'right': W - 40 - r.right(),
                    'left': r.left() - 40}
            fits = {k: v for k, v in room.items() if v >= (lw + gap if k in ('left', 'right') else lh + gap)}
            side = max(fits, key=fits.get) if fits else max(room, key=room.get)
            if 'above' in fits and room['above'] >= lh + gap + 10 and side in ('left', 'right') and r.height() < 300:
                side = 'above'
        if side == 'above':
            x = min(max(24, r.center().x() - lw / 2), W - 24 - lw)
            return QRectF(x, r.top() - gap - lh, lw, lh), QPointF(r.center().x(), r.top()), side
        if side == 'below':
            x = min(max(24, r.center().x() - lw / 2), W - 24 - lw)
            return QRectF(x, r.bottom() + gap, lw, lh), QPointF(r.center().x(), r.bottom()), side
        if side == 'right':
            y = min(max(70, r.center().y() - lh / 2), H - 120 - lh)
            return QRectF(r.right() + gap, y, lw, lh), QPointF(r.right(), r.center().y()), side
        y = min(max(70, r.center().y() - lh / 2), H - 120 - lh)
        return QRectF(r.left() - gap - lw, y, lw, lh), QPointF(r.left(), r.center().y()), side

    def frost(self, rec, f):
        r = self._geom(rec)
        if r is None or not self.text:
            return []
        lr, _, _ = self._label_rect(r)
        return [(lr, 8.0, self.a_out(f) * out_cubic(window(self.t(f), 0.3, 0.3)))]

    def paint(self, p, rec, f):
        r = self._geom(rec)
        if r is None:
            return
        t = self.t(f)
        a = self.a_out(f)
        k_in = self.a_in(f)
        radius = 10.0 if self.style != 'circle' else None
        if self.style == 'circle':
            d = max(r.width(), r.height()) * 1.15
            r = QRectF(r.center().x() - d / 2, r.center().y() - d / 2, d, d)
        shape = QPainterPath()
        if self.style == 'circle':
            shape.addEllipse(r)
        else:
            shape.addRoundedRect(r, radius, radius)
        p.save()
        if self.dim > 0:
            outside = QPainterPath()
            outside.addRect(QRectF(0, 0, W, H))
            p.fillPath(outside.subtracted(shape), col('#000000', self.dim * k_in * a))
        draw = out_cubic(window(t, 0.05, 0.55))
        breathe = 0.5 + 0.5 * math.sin(max(0.0, t - 0.6) * 2.4)
        length = shape.length()
        for wpx, al in ((14.0, 0.07 + 0.03 * breathe), (8.0, 0.12), (2.5, 0.95)):
            pen = QPen(col(ACCENT, al * a), wpx)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            if draw < 0.999:
                pen.setDashPattern([max(0.01, length * draw / wpx), length / wpx + 10])
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPath(shape)
        if self.text:
            lr, anchor, side = self._label_rect(r)
            k = out_cubic(window(t, 0.3, 0.4)) * a
            # leader line from the outline to the label
            if side == 'above':
                b = QPointF(anchor.x() if lr.left() < anchor.x() < lr.right() else lr.center().x(), lr.bottom())
            elif side == 'below':
                b = QPointF(anchor.x() if lr.left() < anchor.x() < lr.right() else lr.center().x(), lr.top())
            elif side == 'right':
                b = QPointF(lr.left(), lr.center().y())
            else:
                b = QPointF(lr.right(), lr.center().y())
            lk = out_cubic(window(t, 0.25, 0.3))
            pen = QPen(col(ACCENT, 0.9 * a), 1.6)
            p.setPen(pen)
            p.drawLine(anchor, QPointF(anchor.x() + (b.x() - anchor.x()) * lk, anchor.y() + (b.y() - anchor.y()) * lk))
            p.setBrush(col(ACCENT, a))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(anchor, 3.2 * lk, 3.2 * lk)
            p.setOpacity(k)
            dy = (1 - out_cubic(window(t, 0.3, 0.4))) * (8 if side != 'above' else -8)
            lr2 = lr.translated(0, dy)
            panel(p, lr2, a=1.0, radius=8, fill=col('#121211', 0.80))
            p.fillRect(QRectF(lr2.x() + 16, lr2.center().y() - 4, 8, 8), col(ACCENT))
            draw_text(p, lr2.x() + 32, lr2.center().y() + 7.5, self.text, self.f_label, INK)
        p.restore()


class Arrow(Layer):
    """An accent arrow drawn from a to b (points or widgets, resolved every frame), with an optional label at its
    tail."""
    z = 22
    t_in = 0.5

    def __init__(self, a, b, text='', dur=2.5, bend=0.18):
        super().__init__(dur)
        self.a, self.b, self.text, self.bend = a, b, text, bend
        self.f_label = font(20, 'semibold')

    def paint(self, p, rec, f):
        a = rec.frame_point(self.a)
        b = rec.frame_point(self.b)
        if a is None or b is None:
            return
        t = self.t(f)
        al = self.a_out(f)
        k = out_cubic(window(t, 0.0, 0.5))
        mx, my = (a.x() + b.x()) / 2, (a.y() + b.y()) / 2
        dx, dy = b.x() - a.x(), b.y() - a.y()
        c = QPointF(mx - dy * self.bend, my + dx * self.bend)
        path = QPainterPath(a)
        path.quadTo(c, b)
        L = path.length()
        p.save()
        for wpx, aa in ((9.0, 0.12), (3.0, 1.0)):
            pen = QPen(col(ACCENT, aa * al), wpx)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            if k < 0.999:
                pen.setDashPattern([max(0.01, L * k / wpx), L / wpx + 10])
            p.setPen(pen)
            p.drawPath(path)
        if k > 0.85:
            hk = (k - 0.85) / 0.15
            pt = path.pointAtPercent(0.999)
            ang = math.radians(-path.angleAtPercent(0.999))
            s = 16 * hk
            head = QPainterPath(pt)
            head.lineTo(pt.x() - s * math.cos(ang - 0.45), pt.y() - s * math.sin(ang - 0.45))
            head.lineTo(pt.x() - s * math.cos(ang + 0.45), pt.y() - s * math.sin(ang + 0.45))
            head.closeSubpath()
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(col(ACCENT, al))
            p.drawPath(head)
        if self.text:
            kk = out_cubic(window(t, 0.15, 0.4)) * al
            p.setOpacity(kk)
            tw = text_w(self.f_label, self.text)
            lr = QRectF(a.x() - tw / 2 - 18, a.y() + (14 if dy < 0 else -58), tw + 36, 44)
            panel(p, lr, radius=8, fill=col('#121211', 0.82))
            draw_text(p, lr.x() + 18, lr.center().y() + 7, self.text, self.f_label, INK)
        p.restore()


class Keys(Layer):
    """Keycaps for a shortcut ("Ctrl+K"), centred low in the frame, pressed once as they appear."""
    z = 40
    t_in = 0.22
    t_out = 0.3

    def __init__(self, combo, dur=1.8, at=None, label=''):
        super().__init__(dur)
        self.keys = [k.strip() for k in combo.replace(' + ', '+').split('+')] if combo != '+' else ['+']
        self.at, self.label = at, label
        self.f_key = font(26, 'semibold', mono=True)
        self.f_plus = font(22, 'regular')
        self.f_label = font(18, 'regular')

    def paint(self, p, rec, f):
        t = self.t(f)
        a = self.a_in(f) * self.a_out(f)
        sizes = [max(58.0, text_w(self.f_key, k) + 36) for k in self.keys]
        gap = 34
        total = sum(sizes) + gap * (len(sizes) - 1)
        cx, y = (W / 2, H - 230) if self.at is None else (self.at.x(), self.at.y())
        x = cx - total / 2
        h = 60
        press = 1.0 if 0.22 < t < 0.36 else 0.0
        sc = 0.94 + 0.06 * out_cubic(t / 0.22)
        p.save()
        p.setOpacity(a)
        p.translate(cx, y + h / 2)
        p.scale(sc, sc)
        p.translate(-cx, -(y + h / 2))
        if self.label:
            draw_text(p, cx - text_w(self.f_label, self.label) / 2, y - 18, self.label, self.f_label, INK2)
        for i, (k, w_) in enumerate(zip(self.keys, sizes)):
            dy = 3 * press
            body = QRectF(x, y + dy, w_, h - 5)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(col('#000000', 0.35))
            p.drawRoundedRect(QRectF(x - 2, y + 6, w_ + 4, h), 11, 11)
            p.setBrush(col('#0e0e0d'))
            p.drawRoundedRect(QRectF(x, y + 5, w_, h - 5), 10, 10)      # the key's side
            g = QLinearGradient(0, body.top(), 0, body.bottom())
            g.setColorAt(0, col('#34342f'))
            g.setColorAt(1, col('#262623'))
            p.setBrush(QBrush(g))
            p.setPen(QPen(col('#ffffff', 0.10), 1))
            p.drawRoundedRect(body, 10, 10)
            p.setFont(self.f_key)
            p.setPen(col(INK))
            p.drawText(body.adjusted(0, -1, 0, 0), Qt.AlignmentFlag.AlignCenter, k)
            x += w_
            if i < len(self.keys) - 1:
                p.setFont(self.f_plus)
                p.setPen(col(MUTED))
                p.drawText(QRectF(x, y, gap, h), Qt.AlignmentFlag.AlignCenter, '+')
                x += gap
        p.restore()


def fmt_number(v, decimals):
    s = f'{v:,.{decimals}f}'
    return s


class Figure(Layer):
    """A stat tile that counts up to a value the app shows. `src` (a widget, rectangle or callable) is where on screen
    the figure comes from: the tile sits beside it (pos='auto') with a short leader and the source gets a thin
    outline, so no figure stands without its source. pos may also be 'tr', 'tl', 'tc', 'br', 'bl' or a QPointF."""
    z = 25
    t_in = 0.5

    def __init__(self, label, value, unit='', decimals=None, dur=3.5, src=None, pos='auto', start=0.0, note=''):
        super().__init__(dur)
        self.label, self.value, self.unit, self.src, self.pos, self.start_v, self.note = \
            label, float(value), unit, src, pos, float(start), note
        if decimals is None:
            s = f'{value}'
            decimals = len(s.split('.')[1]) if '.' in s else 0
        self.decimals = decimals
        self.f_label = font(15, 'semibold', mono=True, spacing=1.5)
        self.f_big = font(72, 'semibold')
        self.f_unit = font(30, 'regular')
        self.f_note = font(15, 'regular', mono=True)
        wv = text_w(self.f_big, fmt_number(self.value, decimals)) + (text_w(self.f_unit, ' ' + unit) + 6 if unit else 0)
        self.w = max(300.0, wv + 64, text_w(self.f_label, label.upper()) + 64, text_w(self.f_note, note) + 64)
        self.h = 158 if note else 136
        self.side = None

    def _rect(self, rec=None):
        m = 56
        if isinstance(self.pos, QPointF):
            return QRectF(self.pos.x(), self.pos.y(), self.w, self.h)
        if self.pos == 'auto' and self.src is not None and rec is not None:
            sr = rec.frame_rect(self.src)
            if sr is not None:
                return self._beside(sr)
        pos = self.pos if self.pos != 'auto' else 'tr'
        x = {'tr': W - m - self.w, 'tl': m, 'br': W - m - self.w, 'bl': m, 'tc': (W - self.w) / 2}[pos]
        y = {'tr': 92, 'tl': 92, 'tc': 92, 'br': H - 140 - self.h, 'bl': H - 140 - self.h}[pos]
        return QRectF(x, y, self.w, self.h)

    def _beside(self, sr: QRectF) -> QRectF:
        gap = 40
        cands = {
            'right': QRectF(sr.right() + gap, sr.center().y() - self.h / 2, self.w, self.h),
            'above': QRectF(sr.center().x() - self.w / 2, sr.top() - gap - self.h, self.w, self.h),
            'left': QRectF(sr.left() - gap - self.w, sr.center().y() - self.h / 2, self.w, self.h),
            'below': QRectF(sr.center().x() - self.w / 2, sr.bottom() + gap, self.w, self.h),
        }

        def clampr(r):
            x = min(max(24.0, r.x()), W - 24 - r.width())
            y = min(max(64.0, r.y()), H - 40 - r.height())
            return QRectF(x, y, r.width(), r.height())
        if self.side is None:
            for k in ('right', 'above', 'left', 'below'):
                r = cands[k]
                c = clampr(r)
                if abs(c.x() - r.x()) < 1 and abs(c.y() - r.y()) < 60 and not c.intersects(sr):
                    self.side = k
                    break
            else:
                self.side = 'above'
        return clampr(cands[self.side])

    def frost(self, rec, f):
        return [(self._rect(rec).translated(0, (1 - out_cubic(window(self.t(f), 0.0, 0.45))) * 16), 12.0,
                 self.a_in(f) * self.a_out(f))]

    def paint(self, p, rec, f):
        t = self.t(f)
        a = self.a_out(f)
        k = out_cubic(window(t, 0.0, 0.45))
        r = self._rect(rec).translated(0, (1 - k) * 16)
        p.save()
        if self.src is not None:
            sr = rec.frame_rect(self.src)
            if sr is not None:
                sr = sr.adjusted(-4, -4, 4, 4)
                lk = out_cubic(window(t, 0.3, 0.4)) * a
                p.setPen(QPen(col(ACCENT, 0.9 * lk), 1.8))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawRoundedRect(sr, 6, 6)
                # a short leader between the nearest edges
                if sr.right() < r.left():
                    a_, b_ = QPointF(sr.right(), sr.center().y()), QPointF(r.left(), min(max(r.top() + 20, sr.center().y()), r.bottom() - 20))
                elif sr.left() > r.right():
                    a_, b_ = QPointF(sr.left(), sr.center().y()), QPointF(r.right(), min(max(r.top() + 20, sr.center().y()), r.bottom() - 20))
                elif sr.top() > r.bottom():
                    x = min(max(r.left() + 20, sr.center().x()), r.right() - 20)
                    a_, b_ = QPointF(x, sr.top()), QPointF(x, r.bottom())
                else:
                    x = min(max(r.left() + 20, sr.center().x()), r.right() - 20)
                    a_, b_ = QPointF(x, sr.bottom()), QPointF(x, r.top())
                p.setPen(QPen(col(ACCENT, 0.85 * lk), 1.6))
                p.drawLine(a_, QPointF(a_.x() + (b_.x() - a_.x()) * lk, a_.y() + (b_.y() - a_.y()) * lk))
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(col(ACCENT, lk))
                p.drawEllipse(a_, 3.2, 3.2)
        p.setOpacity(k * a)
        panel(p, r, radius=12, fill=col('#121211', 0.80))
        p.fillRect(QRectF(r.x(), r.y() + 18, 3, 34), col(ACCENT))
        draw_text(p, r.x() + 30, r.y() + 40, self.label.upper(), self.f_label, MUTED)
        c = out_quint(window(t, 0.15, 1.1))
        v = self.start_v + (self.value - self.start_v) * c
        draw_text(p, r.x() + 28, r.y() + 112, fmt_number(v, self.decimals), self.f_big, INK)
        if self.unit:
            draw_text(p, r.x() + 28 + text_w(self.f_big, fmt_number(self.value, self.decimals)) + 10, r.y() + 112,
                      self.unit, self.f_unit, INK2)
        if self.note:
            draw_text(p, r.x() + 30, r.y() + 142, self.note, self.f_note, MUTED)
        p.restore()


class Note(Layer):
    """A small information chip at the top centre (e.g. "Assistant replies scripted for this video")."""
    z = 35
    t_in = 0.4

    def __init__(self, text, dur=None, y=78):
        super().__init__(dur)
        self.text, self.y = text, y
        self.f = font(19, 'regular')

    def _rect(self):
        tw = text_w(self.f, self.text)
        return QRectF((W - tw - 64) / 2, self.y, tw + 64, 44)

    def frost(self, rec, f):
        return [(self._rect(), 22.0, self.a_in(f) * self.a_out(f))]

    def paint(self, p, rec, f):
        a = self.a_in(f) * self.a_out(f)
        r = self._rect().translated(0, (1 - self.a_in(f)) * -8)
        p.save()
        p.setOpacity(a)
        panel(p, r, radius=22, fill=col('#121211', 0.82))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(col(ACCENT))
        p.drawEllipse(QPointF(r.x() + 24, r.center().y()), 4.5, 4.5)
        draw_text(p, r.x() + 40, r.center().y() + 6.5, self.text, self.f, INK)
        p.restore()


class Badge(Layer):
    """The time-lapse badge, top right: "TIME-LAPSE ×4" with the measured factor (set every frame by the recorder)."""
    z = 36
    t_in = 0.3
    t_out = 0.3

    def __init__(self):
        super().__init__(None)
        self.factor = 1.0
        self.f = font(17, 'semibold', mono=True, spacing=1.5)

    def paint(self, p, rec, f):
        a = self.a_in(f) * self.a_out(f)
        s = f'TIME-LAPSE  ×{self.factor:.0f}' if self.factor >= 1.5 else 'REAL TIME'
        tw = text_w(self.f, s)
        r = QRectF(W - 56 - tw - 66, 84, tw + 66, 42)
        p.save()
        p.setOpacity(a)
        panel(p, r, radius=21, fill=col('#121211', 0.84))
        # two small play triangles
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(col(ACCENT))
        x, cy = r.x() + 20, r.center().y()
        for dx in (0, 9):
            tri = QPainterPath(QPointF(x + dx, cy - 6))
            tri.lineTo(x + dx + 9, cy)
            tri.lineTo(x + dx, cy + 6)
            tri.closeSubpath()
            p.drawPath(tri)
        draw_text(p, r.x() + 46, cy + 6, s, self.f, INK)
        p.restore()


class Indicator(Layer):
    """Persistent and quiet: a hairline map of the video along the bottom edge (one segment per chapter, grouped in
    the five parts; done chapters lit, this one in the accent colour) and, while the frame is not zoomed and no lower
    third is showing, the part and chapter in small type at the top centre."""
    z = 5
    t_in = 0.6

    def __init__(self, plan, chapter_id):
        super().__init__(None)
        self.plan = plan            # [(nn, id, part, title)] of the 38 chapters
        self.cid = chapter_id
        self.f = font(13, 'medium', mono=True, spacing=0.6)
        self.label_a = 0.0
        self.me = next((c for c in plan if c[1] == chapter_id), None)

    def paint(self, p, rec, f):
        a = self.a_in(f) * self.a_out(f)
        if self.me is None:
            return
        p.save()
        n = len(self.plan)
        gap_part, gap = 10.0, 2.0
        parts = sorted({c[2] for c in self.plan})
        x0, x1 = 0.0, float(W)
        usable = (x1 - x0) - gap_part * (len(parts) - 1) - gap * (n - len(parts))
        seg = usable / n
        x = x0
        y, h = H - 4.0, 4.0
        cur_i = self.plan.index(self.me)
        for i, c in enumerate(self.plan):
            if i > 0:
                x += gap_part if c[2] != self.plan[i - 1][2] else gap
            if i < cur_i:
                color = col(INK, 0.30 * a)
            elif i == cur_i:
                color = col(ACCENT, a)
            else:
                color = col(INK, 0.10 * a)
            p.fillRect(QRectF(x, y, seg, h), color)
            x += seg
        # the label: only on a full, un-zoomed frame without a lower third
        want = 1.0 if (rec.zoom_level() < 1.04 and not rec.has_layer(LowerThird)) else 0.0
        self.label_a += (want - self.label_a) * 0.12
        if self.label_a > 0.02:
            s = f'PART {self.me[2]}  ·  {self.me[0]:02d}  {self.me[3]}'
            tw = text_w(self.f, s)
            p.setOpacity(self.label_a * a * 0.75)
            draw_text(p, (W - tw) / 2 + 30, 17, s, self.f, MUTED)
            p.fillRect(QRectF((W - tw) / 2 + 14, 8, 6, 6), col(ACCENT))
        p.restore()


class Statement(Layer):
    """A big left-aligned line (and an optional smaller one) over a soft dark scrim on the left of the frame: for
    openers and summaries over a full-frame 3D view."""
    z = 28
    t_in = 0.7
    t_out = 0.5

    def __init__(self, text, sub='', dur=3.5, eyebrow='', scrim=0.78):
        super().__init__(dur)
        self.text, self.sub, self.eyebrow, self.scrim = text, sub, eyebrow, scrim
        self.f_text = font(68, 'semibold')
        self.f_sub = font(30, 'regular')
        self.f_eye = font(17, 'semibold', mono=True, spacing=3.0)

    def paint(self, p, rec, f):
        t = self.t(f)
        a = self.a_out(f)
        p.save()
        if self.scrim > 0:
            g = QLinearGradient(0, 0, W * 0.62, 0)
            k = out_cubic(window(t, 0, 0.6)) * a
            g.setColorAt(0.0, col('#0b0b0a', self.scrim * k))
            g.setColorAt(0.55, col('#0b0b0a', self.scrim * 0.6 * k))
            g.setColorAt(1.0, col('#0b0b0a', 0.0))
            p.fillRect(QRectF(0, 0, W, H), QBrush(g))
        x, y = 150, H * 0.5
        lift = -10 * (1 - a)
        if self.eyebrow:
            k = out_cubic(window(t, 0.1, 0.5)) * a
            draw_text(p, x, y - 92 + (1 - k) * 10 + lift, self.eyebrow.upper(), self.f_eye, ACCENT, k)
        k = out_cubic(window(t, 0.15, 0.6)) * a
        draw_text(p, x, y + (1 - k) * 18 + lift, self.text, self.f_text, INK, k)
        if self.sub:
            k = out_cubic(window(t, 0.35, 0.6)) * a
            draw_text(p, x, y + 62 + (1 - k) * 12 + lift, self.sub, self.f_sub, INK2, k)
        p.restore()


class Agenda(Layer):
    """A numbered list on the left, each item revealed by reveal(i) (the intro's five parts)."""
    z = 28
    t_in = 0.6
    t_out = 0.6

    def __init__(self, items, title='', dur=None, scrim=0.8):
        super().__init__(dur)
        self.items, self.title, self.scrim = items, title, scrim
        self.shown = {}
        self.f_t = font(17, 'semibold', mono=True, spacing=3.0)
        self.f_n = font(26, 'medium', mono=True)
        self.f_i = font(46, 'semibold')

    def reveal(self, rec, i):
        self.shown.setdefault(i, rec.f)

    def paint(self, p, rec, f):
        a = self.a_out(f)
        t = self.t(f)
        p.save()
        g = QLinearGradient(0, 0, W * 0.65, 0)
        k = out_cubic(window(t, 0, 0.6)) * a
        g.setColorAt(0.0, col('#0b0b0a', self.scrim * k))
        g.setColorAt(0.6, col('#0b0b0a', self.scrim * 0.55 * k))
        g.setColorAt(1.0, col('#0b0b0a', 0.0))
        p.fillRect(QRectF(0, 0, W, H), QBrush(g))
        x = 150
        y0 = H / 2 - (len(self.items) * 84) / 2 + 40
        if self.title:
            kk = out_cubic(window(t, 0.1, 0.5)) * a
            draw_text(p, x, y0 - 56, self.title.upper(), self.f_t, ACCENT, kk)
        for i, (num, text) in enumerate(self.items):
            if i not in self.shown:
                continue
            kk = out_cubic((f - self.shown[i]) / (0.5 * self.fps)) * a
            y = y0 + i * 84 + (1 - kk) * 14
            draw_text(p, x, y, f'{num}', self.f_n, ACCENT, kk)
            draw_text(p, x + 56, y + 2, text, self.f_i, INK, kk)
        p.restore()


# ====================================================================== full-frame cards
def _ghost_number(p, s, x, y, px, a):
    f = font(px, 'semibold')
    path = QPainterPath()
    path.addText(QPointF(x, y), f, s)
    p.setPen(QPen(col(INK, 0.10 * a), 1.6))
    p.setBrush(col(INK, 0.025 * a))
    p.drawPath(path)


def chapter_card(img: QImage, t: float, nn: int, part: int, part_title: str, title: str, promise: str, menu: str,
                 dur: float, exit_t: float = 0.7):
    """Paints the chapter card's type on img (the background is the caller's). t in seconds since the card began;
    the type leaves during the last exit_t seconds."""
    p = painter(img)
    out = clamp((t - (dur - exit_t)) / (exit_t * 0.6))
    a_out = 1 - in_cubic(out)
    lift = -18 * in_cubic(out)
    x = 176
    # big outlined chapter number, drifting slowly up on the right
    gk = out_cubic(window(t, 0.05, 0.9))
    _ghost_number(p, f'{nn:02d}', W - 690, 870 - 30 * (t / max(dur, 1)) - (1 - gk) * 40, 560, gk * a_out)
    # eyebrow
    k = out_cubic(window(t, 0.15, 0.5))
    f_eye = font(18, 'semibold', mono=True, spacing=3.0)
    draw_text(p, x, 382 + (1 - k) * 14 + lift, f'PART {part}  ·  {part_title.upper()}', f_eye, ACCENT, k * a_out)
    # number + title
    k = out_cubic(window(t, 0.25, 0.6))
    f_num = font(30, 'medium', mono=True)
    draw_text(p, x, 470 + (1 - k) * 22 + lift, f'{nn:02d}', f_num, MUTED, k * a_out)
    f_title = font(92 if text_w(font(92, 'semibold'), title) < 1500 else 76, 'semibold')
    draw_text(p, x + 64, 478 + (1 - k) * 22 + lift, title, f_title, INK, k * a_out)
    # accent line wipe
    k = out_quint(window(t, 0.45, 0.7))
    p.fillRect(QRectF(x, 530 + lift, 120 * k, 3), col(ACCENT, a_out))
    # promise and menu path
    k = out_cubic(window(t, 0.55, 0.6))
    f_pr = font(34, 'regular')
    draw_text(p, x, 600 + (1 - k) * 14 + lift, promise, f_pr, INK2, k * a_out)
    if menu:
        k = out_cubic(window(t, 0.7, 0.6))
        f_m = font(19, 'regular', mono=True)
        draw_text(p, x, 662 + (1 - k) * 10 + lift, menu, f_m, MUTED, k * a_out)
    p.end()


def part_card(img: QImage, t: float, part: int, title: str, blurb: str, items, dur: float, exit_t: float = 0.6,
              of: int = 5):
    """The part card: PART n OF m, its title, a line, and its chapters listed in order."""
    p = painter(img)
    out = clamp((t - (dur - exit_t)) / exit_t)
    a_out = 1 - in_cubic(out)
    lift = -14 * in_cubic(out)
    glow(p, 360, 300, 900, ACCENT, 0.07 * out_cubic(window(t, 0, 1.0)) * a_out)
    gk = out_cubic(window(t, 0.0, 1.0))
    _ghost_number(p, str(part), W - 640, 960 - 40 * t / max(dur, 1) - (1 - gk) * 40, 820, gk * a_out)
    x = 176
    k = out_cubic(window(t, 0.15, 0.5))
    draw_text(p, x, 386 + (1 - k) * 14 + lift, f'PART {part} OF {of}', font(20, 'semibold', mono=True, spacing=4.0),
              ACCENT, k * a_out)
    k = out_cubic(window(t, 0.25, 0.65))
    draw_text(p, x, 508 + (1 - k) * 26 + lift, title, font(118, 'semibold'), INK, k * a_out)
    k = out_quint(window(t, 0.5, 0.7))
    p.fillRect(QRectF(x, 556 + lift, 160 * k, 3), col(ACCENT, a_out))
    k = out_cubic(window(t, 0.6, 0.6))
    draw_text(p, x, 624 + (1 - k) * 12 + lift, blurb, font(32, 'regular'), INK2, k * a_out)
    # the chapters, in two columns when there are many
    f_n = font(19, 'medium', mono=True)
    f_t = font(23, 'regular')
    cols = 1 if len(items) <= 5 else 2
    per = math.ceil(len(items) / cols)
    for i, (nn, ttl) in enumerate(items):
        c, r = divmod(i, per)
        k = out_cubic(window(t, 0.85 + 0.07 * i, 0.45))
        xx = x + c * 760
        yy = 724 + r * 44 + (1 - k) * 10 + lift
        draw_text(p, xx, yy, f'{nn:02d}', f_n, ACCENT, k * a_out * 0.95)
        draw_text(p, xx + 50, yy, ttl, f_t, INK2, k * a_out)
    p.end()


_ICON = None


def icon_image() -> QImage:
    global _ICON
    if _ICON is None:
        _ICON = QImage(ICON)
    return _ICON


def logo_reveal(img: QImage, t: float, dur: float, a_out: float = 1.0, version: str | None = None):
    """The logo: the icon scales and sharpens in, the wordmark slides out from behind it, then the tagline and the
    licence line (with `version`, default VERSION)."""
    import cv2
    p = painter(img)
    glow(p, W / 2, H / 2 - 40, 760, ACCENT, 0.09 * out_cubic(window(t, 0.0, 1.2)) * a_out)
    ic = icon_image()
    size = 196
    k = out_quint(window(t, 0.0, 1.1))
    s = size * (0.86 + 0.14 * k)
    # wordmark geometry: icon on the left, "Vantage" on its right, the pair centred
    f_word = font(132, 'semibold')
    ww = text_w(f_word, 'Vantage')
    slide = out_quint(window(t, 0.55, 0.9))
    total = size + 46 + ww
    cx_pair = W / 2 - total / 2
    icon_x = W / 2 - size / 2 + (cx_pair - (W / 2 - size / 2)) * slide
    icon_y = H / 2 - 150
    # blur-in: draw a blurred copy whose blur falls to zero
    blur = (1 - k) * 18
    src = ic.scaled(int(s * 2), int(s * 2), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
    if blur > 0.5:
        a = to_np(src.convertToFormat(QImage.Format.Format_ARGB32))
        a = cv2.GaussianBlur(a, (0, 0), blur)
        src = QImage(np.ascontiguousarray(a).data, a.shape[1], a.shape[0], a.shape[1] * 4,
                     QImage.Format.Format_ARGB32).copy()
    p.setOpacity(k * a_out)
    p.drawImage(QRectF(icon_x + (size - s) / 2, icon_y + (size - s) / 2, s, s), src)
    # wordmark: revealed left to right from behind the icon
    tx = icon_x + size + 46
    p.save()
    p.setClipRect(QRectF(icon_x + size + 10, 0, W, H))
    p.setOpacity(slide * a_out)
    draw_text(p, tx - (1 - slide) * 80, icon_y + size / 2 + 46, 'Vantage', f_word, INK)
    p.restore()
    k2 = out_cubic(window(t, 1.25, 0.7))
    f_tag = font(36, 'regular')
    tw = text_w(f_tag, TAGLINE)
    draw_text(p, W / 2 - tw / 2, icon_y + size + 112 + (1 - k2) * 12, TAGLINE, f_tag, INK2, k2 * a_out)
    k3 = out_cubic(window(t, 1.6, 0.7))
    line = f'Free and open source  ·  v{version or VERSION}'
    f_l = font(21, 'medium', mono=True, spacing=1.0)
    lw = text_w(f_l, line)
    p.setOpacity(1.0)
    draw_text(p, W / 2 - lw / 2, icon_y + size + 170 + (1 - k3) * 10, line, f_l, MUTED, k3 * a_out)
    k4 = out_quint(window(t, 1.5, 0.8))
    p.fillRect(QRectF(W / 2 - 60 * k4, icon_y + size + 64, 120 * k4, 2), col(ACCENT, a_out))
    p.end()


def end_card(img: QImage, t: float, dur: float, version: str | None = None, rows=None):
    """The end screen: the left half says what to do next; the right half stays clear for YouTube's end-screen
    elements (two video tiles and the subscribe button). `rows`: [(heading, link, note)] instead of the download,
    website and source rows; `version` instead of VERSION."""
    p = painter(img)
    x = 150
    k = out_cubic(window(t, 0.0, 0.8))
    glow(p, 420, 420, 820, ACCENT, 0.08 * k)
    ic = icon_image()
    p.setOpacity(k)
    p.drawImage(QRectF(x, 214 + (1 - k) * 12, 96, 96), ic)
    draw_text(p, x + 122, 288 + (1 - k) * 12, 'Vantage', font(84, 'semibold'), INK, k)
    k = out_cubic(window(t, 0.25, 0.7))
    draw_text(p, x, 372 + (1 - k) * 10, TAGLINE, font(28, 'regular'), INK2, k)
    k = out_quint(window(t, 0.4, 0.7))
    p.fillRect(QRectF(x, 408, 120 * k, 3), col(ACCENT))
    rows = rows or [('DOWNLOAD', RELEASES, 'Windows · macOS · Linux x86-64 and arm64 (Jetson)'),
                    ('WEBSITE AND DOCS', SITE, 'feature videos, the full documentation, batch mode'),
                    ('SOURCE', REPO, f'{LICENCE} · free and open source')]
    f_h = font(16, 'semibold', mono=True, spacing=2.0)
    f_u = font(30, 'medium')
    f_s = font(20, 'regular')
    for i, (h_, u, s) in enumerate(rows):
        k = out_cubic(window(t, 0.6 + 0.18 * i, 0.6))
        y = 486 + i * 128 + (1 - k) * 12
        draw_text(p, x, y, h_, f_h, ACCENT, k)
        draw_text(p, x, y + 42, u, f_u, INK, k)
        draw_text(p, x, y + 76, s, f_s, MUTED, k)
    # a hairline dividing the halves, and the clear area's quiet frame
    k = out_cubic(window(t, 0.8, 1.0))
    p.fillRect(QRectF(W / 2 + 10, 150, 1, (H - 300) * k), col(INK, 0.08))
    k = out_cubic(window(t, 1.2, 0.8))
    f_v = font(16, 'medium', mono=True, spacing=1.5)
    draw_text(p, x, H - 96, f'v{version or VERSION}  ·  RECORDED FROM THE APP  ·  VOICE: KOKORO-82M, LOCAL', f_v, MUTED,
              0.8 * k)
    p.end()


def terminal(img: QImage, t: float, script, title='bash', cps=38.0):
    """A terminal window centred on the frame, typing `script` as time goes on: [('cmd', text) | ('out', text[, tone])],
    tone '' / 'ok' / 'warn' / 'bad'. Commands type at cps characters a second; output lines appear 0.12 s apart.
    Returns the time the script finishes (s)."""
    p = painter(img)
    r = QRectF(200, 130, W - 400, H - 260)
    for k, a in ((30, 0.10), (14, 0.16), (5, 0.22)):
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(col('#000000', a))
        p.drawRoundedRect(r.adjusted(-k / 2, -k / 4, k / 2, k), 16, 16)
    p.setBrush(col('#0d0d0c'))
    p.setPen(QPen(col(STRONG), 1))
    p.drawRoundedRect(r, 14, 14)
    bar = QPainterPath()
    bar.addRoundedRect(QRectF(r.x(), r.y(), r.width(), 48), 14, 14)
    p.fillPath(bar, col(PANEL))
    p.fillRect(QRectF(r.x(), r.y() + 34, r.width(), 14), col(PANEL))
    p.fillRect(QRectF(r.x(), r.y() + 48, r.width(), 1), col(HAIR))
    for i in range(3):
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(col(STRONG))
        p.drawEllipse(QPointF(r.x() + 28 + i * 22, r.y() + 24), 6.5, 6.5)
    p.setFont(font(16, 'regular', mono=True))
    p.setPen(col(MUTED))
    p.drawText(QRectF(r.x(), r.y(), r.width(), 48), Qt.AlignmentFlag.AlignCenter, title)
    f = font(21, 'regular', mono=True)
    p.setFont(f)
    lh = 32
    x0, y = r.x() + 36, r.y() + 92
    clock = 0.3
    tones = {'ok': '#5fc35f', 'warn': '#fab219', 'bad': '#ec835a', '': INK2}
    shown = []
    for item in script:
        kind, s = item[0], item[1]
        if kind == 'cmd':
            if t < clock:
                break
            n = int(max(0, min(len(s), (t - clock) * cps)))
            shown.append(('cmd', s[:n], n < len(s)))
            clock += len(s) / cps + 0.45
            if n < len(s):
                break
        else:
            if t < clock:
                break
            shown.append(('out', s, item[2] if len(item) > 2 else ''))
            clock += 0.12
    maxl = int((r.height() - 120) / lh)
    for kind, s, extra in shown[-maxl:]:
        if kind == 'cmd':
            p.setPen(col(ACCENT))
            p.drawText(QPointF(x0, y), '$')
            p.setPen(col(INK))
            p.drawText(QPointF(x0 + 26, y), s)
            if extra and int(t * 2.2) % 2 == 0:
                cw = QFontMetricsF(f).horizontalAdvance(s)
                p.fillRect(QRectF(x0 + 28 + cw, y - 20, 11, 25), col(INK))
        else:
            p.setPen(col(tones.get(extra, INK2)))
            p.drawText(QPointF(x0, y), s)
        y += lh
    p.end()
    total = 0.3 + sum(len(it[1]) / cps + 0.45 if it[0] == 'cmd' else 0.12 for it in script)
    return total
