"""3D viewport: orbit camera, GL renderer (PyOpenGL) with a QPainter fallback, overlays drawn with QPainter.

Scene content comes from a SceneProvider (see scene.py), so the same viewport serves the main
workspace, the bowl view and the import wizard preview.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import numpy as np
from PyQt6.QtCore import Qt, QPointF, QRectF, pyqtSignal, QTimer, QSize
from PyQt6.QtGui import (QPainter, QColor, QPen, QPolygonF, QTransform, QImage, QSurfaceFormat, QFont,
                         QPainterPath, QBrush, QGuiApplication)
from PyQt6.QtWidgets import QWidget

from .theme import THEME, mono, sans

# ---------------------------------------------------------------------------- GL availability
_GL_OK = None


def gl_available() -> bool:
    global _GL_OK
    if _GL_OK is not None:
        return _GL_OK
    if os.environ.get('VANTAGE_SOFTWARE_3D') or QGuiApplication.platformName() in ('offscreen', 'minimal', 'vnc'):
        _GL_OK = False
        return False
    try:
        from PyQt6.QtGui import QOpenGLContext, QOffscreenSurface
        from PyQt6.QtOpenGL import QOpenGLFramebufferObject
        ctx = QOpenGLContext()
        fmt = QSurfaceFormat()
        fmt.setVersion(2, 1)
        fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CompatibilityProfile)
        ctx.setFormat(fmt)
        if not ctx.create():
            raise RuntimeError('no context')
        surf = QOffscreenSurface()
        surf.setFormat(ctx.format())
        surf.create()
        if not ctx.makeCurrent(surf):
            raise RuntimeError('makeCurrent failed')
        fbo = QOpenGLFramebufferObject(8, 8)
        fbo.bind()
        from OpenGL import GL
        GL.glClearColor(1, 0, 0, 1)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        v = np.array([[-1, -1, 0], [1, -1, 0], [0, 1, 0]], np.float32)
        GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
        GL.glVertexPointer(3, GL.GL_FLOAT, 0, v)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)
        GL.glDisableClientState(GL.GL_VERTEX_ARRAY)
        GL.glFinish()
        fbo.release()
        ctx.doneCurrent()
        _GL_OK = True
    except Exception:
        _GL_OK = False
    return _GL_OK


def configure_default_format():
    fmt = QSurfaceFormat()
    fmt.setVersion(2, 1)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CompatibilityProfile)
    fmt.setDepthBufferSize(24)
    fmt.setSamples(4)
    QSurfaceFormat.setDefaultFormat(fmt)


# ---------------------------------------------------------------------------- orbit camera
class OrbitCam:
    def __init__(self):
        self.target = np.array([1.2, 0.0, 0.4])
        self.dist = 9.0
        self.az = 58.0          # degrees, eye direction from target
        self.el = 34.0
        self.fov = 32.0
        self.ortho = False
        self.ortho_h = 8.0
        self.custom = None      # (view 4x4, proj params) for photo-matched views

    def eye(self):
        a, e = math.radians(self.az), math.radians(self.el)
        return self.target + self.dist * np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])

    def view(self):
        if self.custom is not None:
            return self.custom['view']
        eye = self.eye()
        f = self.target - eye
        f /= np.linalg.norm(f)
        up = np.array([0, 0, 1.0])
        if abs(f @ up) > 0.999:
            up = np.array([-math.cos(math.radians(self.az)), -math.sin(math.radians(self.az)), 0.0])
        s = np.cross(f, up)
        s /= np.linalg.norm(s)
        u = np.cross(s, f)
        M = np.eye(4)
        M[0, :3], M[1, :3], M[2, :3] = s, u, -f
        M[:3, 3] = -M[:3, :3] @ eye
        return M

    def proj(self, aspect):
        near, far = 0.05, 400.0
        if self.custom is not None:
            c = self.custom
            # off-axis pinhole from photo intrinsics, fitted to the widget aspect
            return c['proj'](aspect)
        if self.ortho:
            h = self.ortho_h / 2
            w = h * aspect
            P = np.eye(4)
            P[0, 0], P[1, 1] = 1 / w, 1 / h
            P[2, 2] = -2 / (far - near)
            P[2, 3] = -(far + near) / (far - near)
            return P
        f = 1 / math.tan(math.radians(self.fov) / 2)
        P = np.zeros((4, 4))
        P[0, 0], P[1, 1] = f / aspect, f
        P[2, 2] = (far + near) / (near - far)
        P[2, 3] = 2 * far * near / (near - far)
        P[3, 2] = -1
        return P

    def copy(self):
        c = OrbitCam()
        c.__dict__.update({k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in self.__dict__.items()})
        return c


VIEW_PRESETS = {
    'persp': dict(az=58.0, el=34.0, ortho=False),
    'top': dict(az=-90.0, el=89.99, ortho=True),
    'front': dict(az=0.0, el=0.01, ortho=True),
    'rear': dict(az=180.0, el=0.01, ortho=True),
    'side': dict(az=90.0, el=0.01, ortho=True),
    'iso': dict(az=58.0, el=34.0, ortho=False),
}


# ---------------------------------------------------------------------------- draw items
@dataclass
class MeshDraw:
    tri: np.ndarray            # (M,3,3) float32
    normals: np.ndarray        # (M,3)
    base: np.ndarray           # (M,3) float 0..1
    alpha: np.ndarray          # (M,)
    version: int = 0
    wire: bool = False


@dataclass
class GroundDraw:
    image: QImage
    extent: tuple              # x0, y0, x1, y1 (world)
    z: float = 0.002
    nearest: bool = False
    version: int = 0


@dataclass
class LineDraw:
    pts: np.ndarray            # (N,3) polyline or (S,2,3) segments
    color: QColor
    width: float = 1.0
    dash: bool = False
    depth: bool = True
    segments: bool = False
    dash_len: float = 0.07


@dataclass
class TriDraw:
    tri: np.ndarray            # (K,3,3)
    colors: np.ndarray         # (K,4) 0..1
    depth: bool = True


LIGHT = np.array([0.35, 0.55, 0.76])
LIGHT /= np.linalg.norm(LIGHT)


def shade_faces(md: MeshDraw, view_dir):
    n = md.normals
    lam = np.clip(n @ LIGHT, 0, 1)
    head = np.abs(n @ view_dir)
    zc = md.tri[:, :, 2].mean(axis=1)
    ao = 0.72 + 0.28 * np.clip(zc / 0.9, 0, 1)
    s = (0.34 + 0.40 * lam + 0.20 * head) * ao
    c = md.base * s[:, None]
    return np.clip(c, 0, 1)


def dash_segments(pts, dash=0.07, gap=None):
    """Polyline (N,3) -> dashed segments (S,2,3) in world units."""
    pts = np.asarray(pts, float)
    if len(pts) < 2:
        return np.zeros((0, 2, 3))
    gap = gap if gap is not None else dash * 0.8
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    if not np.isfinite(total) or total < 1e-9:
        return np.zeros((0, 2, 3))
    period = max(dash + gap, 1e-6)
    n = int(total / period) + 1
    if n > 20000:  # keep pathological inputs bounded: stretch the pattern instead
        period = total / 20000
        dash = period * 0.55
        n = 20001
    starts = np.arange(n) * period
    starts = starts[starts < total]
    ends = np.minimum(starts + dash, total)

    def at(s):
        return np.stack([np.interp(s, cum, pts[:, k]) for k in range(pts.shape[1])], axis=1)
    return np.stack([at(starts), at(ends)], axis=1)


# ---------------------------------------------------------------------------- base behaviour
class ViewportBase:
    """Shared navigation, projection and overlay logic (mixed into the GL and software widgets)."""
    view_changed = None

    def _init_base(self, provider):
        self.provider = provider
        self.cam = OrbitCam()
        self._anim = None
        self._drag = None
        self._last = None
        self._press_pos = None
        self.nav_style = 'blender'
        self.hover_pos = None
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.setAcceptDrops(True)
        self.setMinimumSize(120, 90)
        provider.changed.connect(self.update)
        THEME.changed.connect(self._on_theme)

    def _on_theme(self, *_):
        self.update()

    # ---- matrices
    def matrices(self):
        w, h = max(self.width(), 1), max(self.height(), 1)
        V = self.cam.view()
        P = self.cam.proj(w / h)
        return V, P, P @ V

    def project(self, X):
        """World (N,3) -> screen (N,2), view depth (N,), in-front mask."""
        X = np.atleast_2d(np.asarray(X, float))
        V, P, M = self.matrices()
        h = np.c_[X, np.ones(len(X))] @ M.T
        w = h[:, 3]
        ok = w > 1e-6 if not self.cam.ortho else np.ones(len(X), bool)
        ww = np.where(np.abs(w) < 1e-9, 1e-9, w)
        sx = (h[:, 0] / ww + 1) * 0.5 * self.width()
        sy = (1 - h[:, 1] / ww) * 0.5 * self.height()
        depth = -(np.c_[X, np.ones(len(X))] @ V.T)[:, 2]
        if self.cam.custom is None and not self.cam.ortho:
            ok &= depth > 0.05
        return np.stack([sx, sy], 1), depth, ok

    def ray(self, sx, sy):
        V, P, M = self.matrices()
        inv = np.linalg.inv(M)
        nx = 2 * sx / max(self.width(), 1) - 1
        ny = 1 - 2 * sy / max(self.height(), 1)
        a = inv @ np.array([nx, ny, -1, 1.0])
        b = inv @ np.array([nx, ny, 1, 1.0])
        a, b = a[:3] / a[3], b[:3] / b[3]
        d = b - a
        return a, d / np.linalg.norm(d)

    def ground_at(self, sx, sy, z=0.0):
        o, d = self.ray(sx, sy)
        if abs(d[2]) < 1e-9:
            return None
        t = (z - o[2]) / d[2]
        if t < 0:
            return None
        return o + d * t

    def px_per_m(self, at):
        """Screen pixels per metre at a point, measured across the screen (a vertical metre is a dot from above)."""
        right = self.cam.view()[0, :3]
        p, _, ok = self.project([at, np.asarray(at, float) + right])
        return float(np.linalg.norm(p[1] - p[0])) if ok.all() else 50.0

    # ---- view control
    def _leave_custom(self):
        """Leave a photo-matched (custom) view properly: the provider also drops the photo and its label."""
        if self.cam.custom is None:
            return
        hook = getattr(self.provider, 'leave_custom_view', None)
        if hook is not None:
            hook(self)
        self.cam.custom = None

    def set_view(self, name, animate=False):
        pr = VIEW_PRESETS.get(name)
        if not pr:
            return
        self._leave_custom()
        if animate:
            self.animate_to(az=pr['az'], el=pr['el'], ortho=pr['ortho'])
            return
        self._stop_anim()
        self.cam.az, self.cam.el, self.cam.ortho = pr['az'], pr['el'], pr['ortho']
        self.update()
        self._emit_view()

    def frame_box(self, lo, hi, margin=1.25, animate=False):
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        c = (lo + hi) / 2
        r = float(np.linalg.norm(hi - lo)) / 2 * margin
        self._leave_custom()
        if self.cam.ortho:
            aspect = max(self.width(), 1) / max(self.height(), 1)
            ext = hi - lo
            if self.cam.el > 80:
                need = max(ext[1], ext[0] / aspect)
            else:
                need = max(ext[2], max(ext[0], ext[1]) / aspect)
            goal = dict(target=c, ortho_h=max(need * margin, 0.5))
        else:
            goal = dict(target=c, dist=max(r / math.tan(math.radians(self.cam.fov) / 2), 0.5))
        if animate:
            self.animate_to(**goal)
            return
        self._stop_anim()
        self.cam.target = goal['target']
        if 'ortho_h' in goal:
            self.cam.ortho_h = goal['ortho_h']
        else:
            self.cam.dist = goal['dist']
        self.update()
        self._emit_view()

    # ---- smooth view changes
    def _stop_anim(self):
        if self._anim is not None:
            self._anim['timer'].stop()
            self._anim = None

    def animate_to(self, az=None, el=None, dist=None, target=None, ortho=None, ortho_h=None, ms=280):
        """Ease the orbit camera to a new view. Perspective and orthographic do not interpolate, so the switch happens
        at the end (into ortho, with the matching height) or at the start (out of ortho, at the matching distance)."""
        from PyQt6.QtCore import QElapsedTimer
        self._stop_anim()
        cam = self.cam
        half = math.tan(math.radians(cam.fov) / 2)
        if ortho is False and cam.ortho:
            cam.ortho = False
            cam.dist = max(cam.ortho_h / (2 * half), 0.5)
        a0 = dict(az=cam.az, el=cam.el, dist=cam.dist, target=cam.target.copy(), ortho_h=cam.ortho_h)
        az1 = cam.az if az is None else cam.az + ((az - cam.az + 180.0) % 360.0 - 180.0)    # the short way round
        a1 = dict(az=az1, el=cam.el if el is None else el, dist=cam.dist if dist is None else dist,
                  target=cam.target.copy() if target is None else np.asarray(target, float),
                  ortho_h=cam.ortho_h if ortho_h is None else ortho_h)
        if ortho and not cam.ortho and dist is None:
            a1['dist'] = cam.dist
        clock = QElapsedTimer()
        clock.start()
        timer = QTimer(self)
        timer.setInterval(15)

        def step():
            t = min(1.0, clock.elapsed() / max(ms, 1))
            e = t * t * (3 - 2 * t)
            cam.az = a0['az'] + (a1['az'] - a0['az']) * e
            cam.el = a0['el'] + (a1['el'] - a0['el']) * e
            cam.dist = a0['dist'] + (a1['dist'] - a0['dist']) * e
            cam.target = a0['target'] + (a1['target'] - a0['target']) * e
            cam.ortho_h = a0['ortho_h'] + (a1['ortho_h'] - a0['ortho_h']) * e
            if t >= 1.0:
                cam.az = (a1['az'] + 180.0) % 360.0 - 180.0
                if ortho and not cam.ortho:
                    cam.ortho = True
                    if ortho_h is None:
                        cam.ortho_h = max(2 * cam.dist * half, 0.3)
                self._stop_anim()
            self.update()
            self._emit_view()
        timer.timeout.connect(step)
        self._anim = {'timer': timer}
        timer.start()
        step()

    def _emit_view(self):
        if hasattr(self, 'viewChanged'):
            self.viewChanged.emit()

    # ---- events
    def mousePressEvent(self, e):
        if self._drag is not None:
            # another button during a drag: ignore it, or the drag's release would never reach the provider
            return
        self._stop_anim()
        self.setFocus()
        self._btn = e.button()
        self._press_pos = e.position()
        self._last = e.position()
        if self.provider.mouse_press(self, e):
            self._drag = 'provider'
            return
        b = e.button()
        mods = e.modifiers()
        orbit_btn = Qt.MouseButton.MiddleButton if self.nav_style == 'blender' else Qt.MouseButton.RightButton
        pan_btn = Qt.MouseButton.MiddleButton if self.nav_style == 'cad' else None
        if b == orbit_btn and self.nav_style == 'blender' and mods & Qt.KeyboardModifier.ShiftModifier:
            self._drag = 'pan'
        elif b == orbit_btn:
            self._drag = 'orbit'
        elif pan_btn is not None and b == pan_btn:
            self._drag = 'pan'
        elif b == Qt.MouseButton.LeftButton:
            self._drag = 'maybe-orbit' if not (mods & Qt.KeyboardModifier.ShiftModifier) else 'maybe-pan'
        elif b == Qt.MouseButton.RightButton:
            self._drag = 'maybe-pan'

    def mouseMoveEvent(self, e):
        pos = e.position()
        if self._drag == 'provider':
            self.provider.mouse_move(self, e)
            return
        if self._drag in ('maybe-orbit', 'maybe-pan') and self._press_pos is not None:
            if (pos - self._press_pos).manhattanLength() > 4:
                self._drag = self._drag[6:]
        if self._drag == 'orbit' and self.cam.custom is None:
            d = pos - self._last
            self.cam.az -= d.x() * 0.4
            self.cam.el = max(-5.0, min(89.99, self.cam.el + d.y() * 0.35))
            if self.cam.ortho and self.cam.el < 89:
                self.cam.ortho = self.cam.ortho  # keep
            self.update()
            self._emit_view()
        elif self._drag == 'pan' and self.cam.custom is None:
            d = pos - self._last
            V = self.cam.view()
            s = self.cam.ortho_h / max(self.height(), 1) if self.cam.ortho else \
                self.cam.dist * 2 * math.tan(math.radians(self.cam.fov) / 2) / max(self.height(), 1)
            right, up = V[0, :3], V[1, :3]
            self.cam.target = self.cam.target - right * d.x() * s + up * d.y() * s
            self.update()
            self._emit_view()
        else:
            self.hover_pos = pos
            self.provider.hover(self, pos)
        self._last = pos

    def focusOutEvent(self, e):
        if self._drag == 'provider':
            self.provider.mouse_release(self, None)
        self._drag = None
        super().focusOutEvent(e)

    def mouseReleaseEvent(self, e):
        if self._drag is not None and e.button() != getattr(self, '_btn', e.button()):
            return                      # the release of a button pressed during the drag
        d = self._drag
        self._drag = None
        if d == 'provider':
            self.provider.mouse_release(self, e)
            return
        moved = self._press_pos is not None and (e.position() - self._press_pos).manhattanLength() > 4
        if not moved:
            if e.button() == Qt.MouseButton.RightButton:
                self.provider.context_menu(self, e.position(), e.globalPosition().toPoint())
            elif e.button() == Qt.MouseButton.LeftButton:
                self.provider.click(self, e)

    def mouseDoubleClickEvent(self, e):
        self.provider.double_click(self, e)

    def wheelEvent(self, e):
        if self.cam.custom is not None:
            return
        dy = e.angleDelta().y()
        if dy == 0:                     # horizontal scrolling is not a zoom
            return
        self._stop_anim()
        k = 0.88 ** (dy / 120.0)        # touchpads send many small steps
        sx, sy = e.position().x(), e.position().y()
        tgt = self.cam.target
        if self.cam.ortho:
            # the point under the cursor on the plane through the target, square to the view
            o, d = self.ray(sx, sy)
            g = o + d * float((tgt - o) @ d)
        else:
            g = self.ground_at(sx, sy, tgt[2])
            if g is not None and np.linalg.norm(g - tgt) > 3 * self.cam.dist:
                g = None                # a grazing ray: zoom about the target instead of flinging it away
        if self.cam.ortho:
            new_h = max(0.3, min(200, self.cam.ortho_h * k))
            k = new_h / self.cam.ortho_h
            self.cam.ortho_h = new_h
        else:
            new_d = max(0.4, min(200, self.cam.dist * k))
            k = new_d / self.cam.dist
            self.cam.dist = new_d
        if g is not None:
            self.cam.target = g + (tgt - g) * k
        self.update()
        self._emit_view()

    def leaveEvent(self, e):
        self.hover_pos = None
        self.provider.hover(self, None)

    def keyPressEvent(self, e):
        if self.provider.key_press(self, e):
            return
        k = e.key()
        num = bool(e.modifiers() & Qt.KeyboardModifier.KeypadModifier)
        if k == Qt.Key.Key_7 and num:
            self.set_view('top')
        elif k == Qt.Key.Key_1 and num:
            self.set_view('front')
        elif k == Qt.Key.Key_3 and num:
            self.set_view('side')
        elif k == Qt.Key.Key_5 and num:
            self._leave_custom()
            self.cam.ortho = not self.cam.ortho
            self.update()
        else:
            super().keyPressEvent(e)

    def dragEnterEvent(self, e):
        if self.provider.drag_enter(self, e):
            e.acceptProposedAction()

    def dragMoveEvent(self, e):
        self.provider.drag_move(self, e)
        e.acceptProposedAction()

    def dropEvent(self, e):
        self.provider.drop(self, e)

    # ---- painting helpers shared by both renderers
    def paint_background(self, p: QPainter):
        p.fillRect(self.rect(), self.provider.background_color())
        bg = self.provider.background_image(self)
        if bg is not None:
            img, rect = bg
            p.drawImage(rect, img)

    def paint_overlay(self, p: QPainter):
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.provider.overlay(p, self)
        self._paint_axes(p)
        self._paint_scale(p)
        label = self.provider.hud_label(self)
        if label:
            p.setFont(sans(11))
            if self.provider.background_image(self) is not None:
                # over a photo: a backing plate keeps the label readable on bright backgrounds
                w = min(self.width() - 12, p.fontMetrics().horizontalAdvance(label) + 12)
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(THEME.color('panel', 0.82))
                p.drawRoundedRect(QRectF(4, 4, w, 19), 3, 3)
            p.setPen(THEME.color('ink2'))
            p.drawText(QRectF(10, 6, self.width() - 20, 16), Qt.AlignmentFlag.AlignLeft, label)

    def _paint_axes(self, p: QPainter):
        V = self.cam.view()
        cx, cy = 22, self.height() - 22
        L = 18
        cols = [QColor('#e0533f'), QColor('#3fae49'), QColor('#3d86e6')]
        axes = []
        for k in range(3):
            v = V[:3, k]
            axes.append((v[2], k, v))
        for _, k, v in sorted(axes):
            p.setPen(QPen(cols[k], 1.4))
            p.drawLine(QPointF(cx, cy), QPointF(cx + v[0] * L, cy - v[1] * L))

    def _paint_scale(self, p: QPainter):
        tgt = self.cam.target
        ppm = self.px_per_m(np.array([tgt[0], tgt[1], 0.0]))
        if ppm <= 1:
            return
        meters = 1.0
        for m in (0.1, 0.25, 0.5, 1, 2, 5, 10, 20):
            if m * ppm >= 60:
                meters = m
                break
        w = meters * ppm
        x1, y = self.width() - 18, self.height() - 14
        x0 = x1 - w
        p.setPen(QPen(THEME.color('ink2'), 1))
        p.drawLine(QPointF(x0, y), QPointF(x1, y))
        p.drawLine(QPointF(x0, y - 3), QPointF(x0, y))
        p.drawLine(QPointF(x1, y - 3), QPointF(x1, y))
        p.setFont(mono(11))
        txt = f'{meters:g} m'
        p.drawText(QRectF(x0, y - 18, w, 14), Qt.AlignmentFlag.AlignRight, txt)

    def grab_image(self) -> QImage:
        return self.grabFramebuffer() if hasattr(self, 'grabFramebuffer') else self.grab().toImage()

    # ---- high-resolution renders
    def render_image(self, scale=2.0, transparent=False) -> QImage:
        """The view at scale x its size, overlays included (vectors stay crisp). transparent: no background."""
        w, h = max(1, int(round(self.width() * scale))), max(1, int(round(self.height() * scale)))
        img = QImage(w, h, QImage.Format.Format_ARGB32_Premultiplied)
        img.fill(Qt.GlobalColor.transparent)
        scene = self._scene_image(w, h, scale)          # GL: the 3D layer rendered off-screen; None: painted below
        p = QPainter(img)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.scale(scale, scale)
        if not transparent:
            self.paint_background(p)
        if scene is not None:
            p.save()
            p.resetTransform()
            p.drawImage(0, 0, scene)
            p.restore()
        else:
            self._paint_scene(p)
        self.paint_overlay(p)
        p.end()
        return img

    def _scene_image(self, w, h, scale):
        return None

    def _render_style(self):
        st = getattr(self.provider, 'render_style', None)
        return st(self) if st is not None else None


def soft_shadow(md, res=0.02, light=None):
    """(QImage, extent) of the vehicle's shadow along the key light plus contact darkening under the body, for the
    software renderer (the GL renderer uses real shadow maps)."""
    import cv2
    L = np.asarray(light if light is not None else LIGHT, float)
    T = md.tri.astype(np.float64)
    keep = md.alpha > 0.5
    T = T[keep] if keep.any() else T
    z = T[..., 2:3]
    proj = T[..., :2] - L[:2] * (z / max(L[2], 0.2))             # along the light onto z = 0
    pts = np.concatenate([proj.reshape(-1, 2), T[..., :2].reshape(-1, 2)])
    lo, hi = pts.min(axis=0) - 0.8, pts.max(axis=0) + 0.8
    nx, ny = int(np.ceil((hi[0] - lo[0]) / res)), int(np.ceil((hi[1] - lo[1]) / res))

    def raster(P, mask=None):
        img = np.zeros((ny, nx), np.uint8)
        q = np.empty_like(P)
        q[..., 0] = (P[..., 0] - lo[0]) / res
        q[..., 1] = (hi[1] - P[..., 1]) / res
        sel = q if mask is None else q[mask]
        cv2.fillPoly(img, np.round(sel * 4).astype(np.int32), 255, lineType=cv2.LINE_8, shift=2)
        return img.astype(np.float32) / 255.0
    sh = cv2.GaussianBlur(raster(proj), (0, 0), 2.5)
    low = T[..., 2].min(axis=1) < 0.55                           # parts near the ground darken it most
    ao = cv2.GaussianBlur(raster(T[..., :2], low), (0, 0), 9.0) if low.any() else np.zeros_like(sh)
    a = np.clip(np.maximum(sh * 0.30, ao * 0.55), 0, 1)
    rgba = np.zeros((ny, nx, 4), np.uint8)
    rgba[..., 3] = (a * 255).astype(np.uint8)                  # black, premultiplied
    img = QImage(rgba.tobytes(), nx, ny, nx * 4, QImage.Format.Format_RGBA8888_Premultiplied).copy()
    return img, (float(lo[0]), float(lo[1]), float(lo[0] + nx * res), float(hi[1]))


# ---------------------------------------------------------------------------- software renderer
class SoftViewport(QWidget, ViewportBase):
    viewChanged = pyqtSignal()

    def __init__(self, provider, parent=None):
        super().__init__(parent)
        self._init_base(provider)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)

    def paintEvent(self, e):
        p = QPainter(self)
        self.paint_background(p)
        self._paint_scene(p)
        self.paint_overlay(p)
        p.end()

    def _poly(self, pts):
        return QPolygonF([QPointF(float(a), float(b)) for a, b in pts])

    def _paint_scene(self, p: QPainter):
        prov = self.provider
        V, P, M = self.matrices()
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        # ground texture via projective transform
        g = prov.ground()
        if g is not None:
            x0, y0, x1, y1 = g.extent
            corners = np.array([[x0, y1, g.z], [x1, y1, g.z], [x1, y0, g.z], [x0, y0, g.z]])
            sc, dep, ok = self.project(corners)
            if ok.all():
                src = QPolygonF([QPointF(0, 0), QPointF(g.image.width(), 0), QPointF(g.image.width(), g.image.height()),
                                 QPointF(0, g.image.height())])
                dst = self._poly(sc)
                tr = QTransform()
                if QTransform.quadToQuad(src, dst, tr):
                    p.save()
                    p.setTransform(tr)
                    p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, not g.nearest)
                    p.drawImage(0, 0, g.image)
                    p.restore()
        lines = prov.lines(self)
        self._paint_lines(p, [l for l in lines if l.depth and self._on_ground(l)])
        md = prov.mesh()
        st = self._render_style()
        if md is not None and len(md.tri) and (st is None or st.shadows or st.ao):
            self._paint_soft_shadow(p, md)
        # triangles (mesh + extra), painter's algorithm
        items = []
        view_dir = -(V[2, :3])
        if md is not None and len(md.tri):
            tri = md.tri
            cols = shade_faces(md, view_dir)
            views = getattr(prov, 'camera_lights', lambda vp: [])(self)
            if views:
                cols = self._camlight_colors(md, cols, views, st.camlight_mix if st else 0.6)
            eye = np.linalg.inv(V)[:3, 3]
            cen = tri.mean(axis=1)
            facing = np.einsum('ij,ij->i', md.normals, (eye - cen) if not self.cam.ortho else np.tile(view_dir, (len(cen), 1)))
            keep = (facing > 0) | (md.alpha < 0.99)
            items.append((tri[keep], np.c_[cols[keep], md.alpha[keep]], md.wire))
        for td in prov.tris_extra(self):
            items.append((td.tri, td.colors, False))
        if items:
            T = np.concatenate([i[0] for i in items])
            C = np.concatenate([i[1] for i in items])
            W = np.concatenate([np.full(len(i[0]), i[2]) for i in items])
            flat = T.reshape(-1, 3)
            sc, dep, ok = self.project(flat)
            sc = sc.reshape(-1, 3, 2)
            dep = dep.reshape(-1, 3).mean(axis=1)
            ok = ok.reshape(-1, 3).all(axis=1)
            order = np.argsort(-dep)
            order = order[ok[order]]
            p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
            for i in order:
                c = C[i]
                col = QColor.fromRgbF(float(c[0]), float(c[1]), float(c[2]), float(c[3]))
                if W[i]:
                    p.setBrush(Qt.BrushStyle.NoBrush)
                    p.setPen(QPen(col, 1))
                else:
                    p.setBrush(col)
                    p.setPen(QPen(col, 0.8) if c[3] > 0.99 else Qt.PenStyle.NoPen)
                a = sc[i]
                p.drawPolygon(QPolygonF([QPointF(a[0, 0], a[0, 1]), QPointF(a[1, 0], a[1, 1]), QPointF(a[2, 0], a[2, 1])]))
            p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self._paint_lines(p, [l for l in lines if not (l.depth and self._on_ground(l))])

    def _on_ground(self, l):
        pts = l.pts.reshape(-1, 3)
        return len(pts) and float(np.abs(pts[:, 2]).max()) < 0.03

    def _paint_soft_shadow(self, p: QPainter, md):
        """The vehicle's shadow and contact darkening on the ground, as a soft texture computed once per mesh."""
        cache = getattr(self, '_shadow_cache', None)
        if cache is None or cache[0] is not md:
            cache = (md,) + soft_shadow(md)
            self._shadow_cache = cache
        _, img, (x0, y0, x1, y1) = cache
        corners = np.array([[x0, y1, 0.001], [x1, y1, 0.001], [x1, y0, 0.001], [x0, y0, 0.001]])
        sc, dep, ok = self.project(corners)
        if not ok.all():
            return
        src = QPolygonF([QPointF(0, 0), QPointF(img.width(), 0), QPointF(img.width(), img.height()), QPointF(0, img.height())])
        tr = QTransform()
        if QTransform.quadToQuad(src, self._poly(sc), tr):
            p.save()
            p.setTransform(tr, True)
            p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
            p.drawImage(0, 0, img)
            p.restore()

    def _camlight_colors(self, md, cols, views, mix):
        from ..core.visibility import seen_by, tint
        key = (id(md), tuple((v.uid, v.pos.tobytes(), v.W2O.tobytes(), id(v.intr), id(v.cube), v.color) for v in views), mix)
        c = getattr(self, '_camlight_cache', None)
        if c is None or c[0] != key:
            cen = md.tri.mean(axis=1)
            seen = seen_by(cen + md.normals * 0.03, views)
            c = (key, seen, cen)
            self._camlight_cache = c
        return tint(cols, c[1], views, mix, points=c[2])

    def _paint_lines(self, p: QPainter, lines):
        for l in lines:
            segs = l.pts if l.segments else np.stack([l.pts[:-1], l.pts[1:]], 1) if len(l.pts) > 1 else np.zeros((0, 2, 3))
            if l.dash and len(segs):
                segs = np.concatenate([dash_segments(s, l.dash_len) for s in segs]) if l.segments else dash_segments(l.pts, l.dash_len)
            if not len(segs):
                continue
            sc, dep, ok = self.project(segs.reshape(-1, 3))
            sc = sc.reshape(-1, 2, 2)
            ok = ok.reshape(-1, 2).all(axis=1)
            pen = QPen(l.color, l.width)
            pen.setCapStyle(Qt.PenCapStyle.FlatCap)
            p.setPen(pen)
            from PyQt6.QtCore import QLineF
            p.drawLines([QLineF(a[0, 0], a[0, 1], a[1, 0], a[1, 1]) for a, k in zip(sc, ok) if k])


# ---------------------------------------------------------------------------- GL renderer
try:
    from PyQt6.QtOpenGLWidgets import QOpenGLWidget
except Exception:  # pragma: no cover
    QOpenGLWidget = None


if QOpenGLWidget is not None:
    class GLViewport(QOpenGLWidget, ViewportBase):
        viewChanged = pyqtSignal()

        def __init__(self, provider, parent=None):
            super().__init__(parent)
            self._init_base(provider)
            self._tex = {}
            self._mesh_cache = None
            self.gl_failed = False
            self.renderer = None

        def initializeGL(self):
            from OpenGL import GL
            from .render_gl import GLRenderer
            self.GL = GL
            self._tex = {}                      # a new context (the widget was re-parented): old ids are gone
            r = GLRenderer()
            self.renderer = r if (not os.environ.get('VANTAGE_LEGACY_GL') and r.init()) else None
            if self.renderer is None and r.error:
                print('Shader renderer unavailable, using fixed-function GL:', r.error.splitlines()[0][:200])

        @property
        def gl_bowl(self):
            return self.renderer is not None

        def _frame(self, w, h, scale=1.0, clear=None):
            from .render_gl import Frame, Style
            prov = self.provider
            V, P, _ = self.matrices()
            f = Frame(V, P, w, h, target=self.cam.target.copy(), mesh=prov.mesh(), ground=prov.ground(),
                      lines=prov.lines(self), tris=prov.tris_extra(self), clear=clear, ortho=self.cam.ortho)
            f.bowl = getattr(prov, 'bowl_draw', lambda vp: None)(self)
            f.cams = getattr(prov, 'camera_lights', lambda vp: [])(self)
            f.style = self._render_style() or Style()
            self.renderer.dpr = scale
            return f

        def paintGL(self):
            p = QPainter(self)
            self.paint_background(p)
            p.beginNativePainting()
            try:
                if self.renderer is not None:
                    dpr = self.devicePixelRatioF()
                    self.renderer.render(self._frame(int(self.width() * dpr), int(self.height() * dpr), dpr),
                                         self.defaultFramebufferObject())
                else:
                    self._gl_scene()
            except Exception as ex:  # keep the overlay alive if a driver call fails
                self.gl_failed = True
                print('GL error:', ex)
                if self.renderer is not None:       # fall back to the fixed-function path from now on
                    self.renderer = None
            p.endNativePainting()
            self.paint_overlay(p)
            p.end()

        def _scene_image(self, w, h, scale):
            if self.renderer is None:
                return None
            self.makeCurrent()
            try:
                rgba = self.renderer.render_offscreen(self._frame(w, h, scale, clear=(0, 0, 0, 0)))
            finally:
                self.doneCurrent()
            return QImage(rgba.tobytes(), w, h, w * 4, QImage.Format.Format_RGBA8888_Premultiplied).copy()

        def render_image(self, scale=2.0, transparent=False) -> QImage:
            if self.renderer is None:       # fixed-function path: the framebuffer, scaled
                img = self.grabFramebuffer()
                return img.scaled(int(img.width() * scale), int(img.height() * scale), Qt.AspectRatioMode.IgnoreAspectRatio,
                                  Qt.TransformationMode.SmoothTransformation)
            return ViewportBase.render_image(self, scale, transparent)

        def _gl_scene(self):
            GL = self.GL
            dpr = self.devicePixelRatioF()
            GL.glViewport(0, 0, int(self.width() * dpr), int(self.height() * dpr))
            GL.glClear(GL.GL_DEPTH_BUFFER_BIT)
            V, P, M = self.matrices()
            GL.glMatrixMode(GL.GL_PROJECTION)
            GL.glLoadMatrixd(P.T.astype(np.float64).ravel())
            GL.glMatrixMode(GL.GL_MODELVIEW)
            GL.glLoadMatrixd(V.T.astype(np.float64).ravel())
            GL.glEnable(GL.GL_MULTISAMPLE)      # QPainter switches it off before native painting
            GL.glDisable(GL.GL_LIGHTING)
            GL.glDisable(GL.GL_CULL_FACE)
            GL.glEnable(GL.GL_BLEND)
            GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
            GL.glEnable(GL.GL_DEPTH_TEST)
            GL.glDepthFunc(GL.GL_LEQUAL)
            GL.glEnable(GL.GL_LINE_SMOOTH)
            prov = self.provider
            # ground texture (no depth write so lines on the ground stay visible)
            g = prov.ground()
            if g is not None:
                self._draw_ground(g)
            lines = prov.lines(self)
            for l in lines:
                if l.depth and self._is_ground(l):
                    self._draw_line(l)
            # mesh
            md = prov.mesh()
            if md is not None and len(md.tri):
                view_dir = -(V[2, :3])
                cols = shade_faces(md, view_dir)
                self._draw_tris(md.tri, np.c_[cols, md.alpha], wire=md.wire, offset=True)
            for td in prov.tris_extra(self):
                self._draw_tris(td.tri, td.colors, depth=td.depth)
            for l in lines:
                if not (l.depth and self._is_ground(l)):
                    self._draw_line(l)
            GL.glDisable(GL.GL_DEPTH_TEST)
            GL.glMatrixMode(GL.GL_PROJECTION)
            GL.glLoadIdentity()
            GL.glMatrixMode(GL.GL_MODELVIEW)
            GL.glLoadIdentity()

        def _is_ground(self, l):
            pts = l.pts.reshape(-1, 3)
            return len(pts) and float(np.abs(pts[:, 2]).max()) < 0.03

        def _draw_tris(self, tri, colors, wire=False, depth=True, offset=False):
            GL = self.GL
            n = len(tri)
            if n == 0:
                return
            v = np.ascontiguousarray(tri.reshape(-1, 3), np.float32)
            c = np.ascontiguousarray(np.repeat(colors, 3, axis=0), np.float32)
            opaque = colors[:, 3] > 0.99
            if depth:
                GL.glEnable(GL.GL_DEPTH_TEST)
            else:
                GL.glDisable(GL.GL_DEPTH_TEST)
            if offset:
                GL.glEnable(GL.GL_POLYGON_OFFSET_FILL)
                GL.glPolygonOffset(1.0, 1.0)
            GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
            GL.glEnableClientState(GL.GL_COLOR_ARRAY)
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE if wire else GL.GL_FILL)
            if opaque.all():
                GL.glVertexPointer(3, GL.GL_FLOAT, 0, v)
                GL.glColorPointer(4, GL.GL_FLOAT, 0, c)
                GL.glDrawArrays(GL.GL_TRIANGLES, 0, len(v))
            else:
                for mask, write in ((opaque, True), (~opaque, False)):
                    if not mask.any():
                        continue
                    vm = np.ascontiguousarray(tri[mask].reshape(-1, 3), np.float32)
                    cm = np.ascontiguousarray(np.repeat(colors[mask], 3, axis=0), np.float32)
                    GL.glDepthMask(GL.GL_TRUE if write else GL.GL_FALSE)
                    GL.glVertexPointer(3, GL.GL_FLOAT, 0, vm)
                    GL.glColorPointer(4, GL.GL_FLOAT, 0, cm)
                    GL.glDrawArrays(GL.GL_TRIANGLES, 0, len(vm))
                GL.glDepthMask(GL.GL_TRUE)
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
            GL.glDisableClientState(GL.GL_COLOR_ARRAY)
            GL.glDisableClientState(GL.GL_VERTEX_ARRAY)
            if offset:
                GL.glDisable(GL.GL_POLYGON_OFFSET_FILL)
            GL.glEnable(GL.GL_DEPTH_TEST)

        def _draw_line(self, l: LineDraw):
            GL = self.GL
            segs = l.pts if l.segments else (np.stack([l.pts[:-1], l.pts[1:]], 1) if len(l.pts) > 1 else np.zeros((0, 2, 3)))
            if l.dash and len(segs):
                segs = np.concatenate([dash_segments(s, l.dash_len) for s in segs]) if l.segments else dash_segments(l.pts, l.dash_len)
            if not len(segs):
                return
            v = np.ascontiguousarray(segs.reshape(-1, 3), np.float32)
            if l.depth:
                GL.glEnable(GL.GL_DEPTH_TEST)
            else:
                GL.glDisable(GL.GL_DEPTH_TEST)
            GL.glLineWidth(float(l.width) * self.devicePixelRatioF())
            c = l.color
            GL.glColor4f(c.redF(), c.greenF(), c.blueF(), c.alphaF())
            GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
            GL.glVertexPointer(3, GL.GL_FLOAT, 0, v)
            GL.glDrawArrays(GL.GL_LINES, 0, len(v))
            GL.glDisableClientState(GL.GL_VERTEX_ARRAY)
            GL.glEnable(GL.GL_DEPTH_TEST)

        def _draw_ground(self, g: GroundDraw):
            GL = self.GL
            key = id(self)
            img = g.image.convertToFormat(QImage.Format.Format_RGBA8888)
            tex_key = (g.version, img.width(), img.height(), g.nearest)
            tid = self._tex.get('ground')
            if tid is None:
                tid = GL.glGenTextures(1)
                self._tex['ground'] = tid
            GL.glBindTexture(GL.GL_TEXTURE_2D, tid)
            if self._tex.get('ground_key') != tex_key:
                ptr = img.constBits()
                ptr.setsize(img.sizeInBytes())
                data = np.frombuffer(ptr, np.uint8).reshape(img.height(), img.bytesPerLine())[:, :img.width() * 4].copy()
                GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
                GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, img.width(), img.height(), 0, GL.GL_RGBA,
                                GL.GL_UNSIGNED_BYTE, data)
                f = GL.GL_NEAREST if g.nearest else GL.GL_LINEAR
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, f)
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, f)
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
                self._tex['ground_key'] = tex_key
            GL.glEnable(GL.GL_TEXTURE_2D)
            GL.glDepthMask(GL.GL_FALSE)
            GL.glColor4f(1, 1, 1, 1)
            x0, y0, x1, y1 = g.extent
            z = g.z
            GL.glBegin(GL.GL_QUADS)
            GL.glTexCoord2f(0, 1); GL.glVertex3f(x0, y0, z)
            GL.glTexCoord2f(1, 1); GL.glVertex3f(x1, y0, z)
            GL.glTexCoord2f(1, 0); GL.glVertex3f(x1, y1, z)
            GL.glTexCoord2f(0, 0); GL.glVertex3f(x0, y1, z)
            GL.glEnd()
            GL.glDepthMask(GL.GL_TRUE)
            GL.glDisable(GL.GL_TEXTURE_2D)
            GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
else:
    GLViewport = None


def make_viewport(provider, parent=None):
    if GLViewport is not None and gl_available():
        return GLViewport(provider, parent)
    return SoftViewport(provider, parent)
