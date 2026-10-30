"""Getting around the 3D view.

- Orbit about the point under the cursor (the body, a rig part, a sensor or the ground where the drag starts), so
  what you look at stays put while the view turns round it; zoom towards the surface under the cursor.
- The navigation control in the view's top-right corner, for a touchpad or a mouse without a middle button: drag the
  axis ball to orbit, click an axis to look along it, drag the hand to pan and the magnifier to zoom, click the frame
  button to see everything and the cube to switch perspective and orthographic.
- Touchpad: two fingers orbit, with Shift they pan, with Ctrl (or a pinch) they zoom. A mouse wheel zooms.
- Keyboard (3D view focused): arrows orbit, Shift+arrows pan, + and − zoom; 1 front, Ctrl+1 rear, 3 side, 7 top,
  5 perspective/orthographic, 0 the default view; F frames the selection and Home everything (main window).
- The view bar along the top-left edge: buttons for the view, the shading, the overlays and the heat map, each opening
  the same choices as the View menu (the provider lists them: view_bar and view_bar_menu).
"""
from __future__ import annotations

import math

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QFontMetricsF, QPainter, QPen
from PyQt6.QtWidgets import QMenu

EL_MIN, EL_MAX = -5.0, 89.99      # elevation limits of the orbit, degrees (just below the horizon, not upside down)
ORBIT_DEG_PER_PX = 0.4
AXIS_COLORS = ('#e0533f', '#3fae49', '#3d86e6')      # x, y, z as the axes indicator and the move handles
BALL_R = 30.0                     # the axis ball's radius, logical px
BTN_R = 12.0                      # the buttons under it
# what clicking an axis bubble looks along: (axis, sign) -> (azimuth, elevation) of the eye
AXIS_VIEWS = {(0, 1): (0.0, 0.0), (0, -1): (180.0, 0.0), (1, 1): (90.0, 0.0), (1, -1): (-90.0, 0.0),
              (2, 1): (-90.0, 89.99)}


def orbit_about(cam, pivot, daz_deg: float, del_deg: float):
    """Turn the orbit camera about pivot by daz (about the world z axis) and del (up and down about the camera's
    horizontal right axis); target and eye move together, so the pivot stays where it is on the screen. The elevation
    stays within EL_MIN..EL_MAX."""
    pivot = np.asarray(pivot, float)
    el1 = min(max(cam.el + del_deg, EL_MIN), EL_MAX)
    E, T = cam.eye(), np.asarray(cam.target, float)
    a = math.radians(daz_deg)
    Rz = np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])
    E, T = pivot + (E - pivot) @ Rz.T, pivot + (T - pivot) @ Rz.T
    off = E - T
    dist = float(np.linalg.norm(off))
    az = math.degrees(math.atan2(off[1], off[0]))
    el0 = math.degrees(math.asin(max(-1.0, min(1.0, off[2] / max(dist, 1e-12)))))
    turn = el1 - el0
    if abs(turn) > 1e-9:
        # about the horizontal axis through the pivot, square to the azimuth: raising the eye lowers the target
        s = np.array([-math.sin(math.radians(az)), math.cos(math.radians(az)), 0.0])
        R = _axis_angle(s, -math.radians(turn))
        E2, T2 = pivot + (E - pivot) @ R.T, pivot + (T - pivot) @ R.T
        off2 = E2 - T2
        el2 = math.degrees(math.asin(max(-1.0, min(1.0, off2[2] / max(dist, 1e-12)))))
        if abs(el2 - el1) > abs((2 * el0 - el2) - el1):       # the other way round
            R = _axis_angle(s, math.radians(turn))
            E2, T2 = pivot + (E - pivot) @ R.T, pivot + (T - pivot) @ R.T
        E, T = E2, T2
        off = E - T
        az = math.degrees(math.atan2(off[1], off[0]))
    cam.target = T
    cam.dist = max(float(np.linalg.norm(off)), 1e-3)
    cam.az = az
    cam.el = math.degrees(math.asin(max(-1.0, min(1.0, off[2] / cam.dist))))


def _axis_angle(u, ang) -> np.ndarray:
    u = np.asarray(u, float) / max(np.linalg.norm(u), 1e-12)
    c, s = math.cos(ang), math.sin(ang)
    x, y, z = u
    return np.array([[c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
                     [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
                     [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)]])


def is_touchpad(e) -> bool:
    """A wheel event from a touchpad (precise pixel deltas in scroll phases), not a mouse wheel."""
    try:
        from PyQt6.QtGui import QInputDevice
        dev = e.device()
        if dev is not None and dev.type() == QInputDevice.DeviceType.TouchPad:
            return True
    except (AttributeError, RuntimeError):
        pass
    return not e.pixelDelta().isNull() and e.phase() != Qt.ScrollPhase.NoScrollPhase


# ============================================================================ the navigation control
class NavControl:
    """The axis ball and its buttons in a view's top-right corner: geometry, hit testing and painting. The viewport
    owns one and routes the mouse to it first."""

    BUTTONS = (('pan', 'hand', 'Drag to pan'), ('zoom', 'search', 'Drag up or down to zoom'),
               ('frame', 'frame', 'Frame everything (Home)'), ('ortho', 'cube', 'Perspective / orthographic (5)'))

    def __init__(self, vp):
        self.vp = vp
        self.hot = None            # what the pointer is over: 'ball', ('axis', k, sign), 'pan', 'zoom', 'frame', 'ortho'
        self.visible = True

    def centre(self) -> QPointF:
        return QPointF(self.vp.width() - BALL_R - 14.0, BALL_R + 34.0)

    def button_centre(self, i) -> QPointF:
        c = self.centre()
        return QPointF(c.x(), c.y() + BALL_R + 10.0 + BTN_R + i * (2 * BTN_R + 6.0))

    def bubbles(self):
        """[(depth, axis, sign, QPointF)] of the axis ends, back to front."""
        V = self.vp.cam.view()
        c = self.centre()
        out = []
        for k in range(3):
            v = V[:3, k]
            for sign in (1, -1):
                q = QPointF(c.x() + sign * v[0] * BALL_R * 0.78, c.y() - sign * v[1] * BALL_R * 0.78)
                out.append((sign * v[2], k, sign, q))
        out.sort(key=lambda r: r[0])
        return out

    def enabled(self) -> bool:
        vp = self.vp
        return self.visible and vp.cam.custom is None and vp.width() >= 260 and vp.height() >= 200

    def hit(self, pos):
        """What is under pos, or None."""
        b = self.vp.bar.hit(pos) if self.visible and hasattr(self.vp, 'bar') else None
        if b is not None:
            return ('bar', b)
        if not self.enabled():
            return None
        for i, (key, _, _) in enumerate(self.BUTTONS):
            b = self.button_centre(i)
            if math.hypot(pos.x() - b.x(), pos.y() - b.y()) <= BTN_R + 2:
                return key
        c = self.centre()
        if math.hypot(pos.x() - c.x(), pos.y() - c.y()) > BALL_R + 3:
            return None
        for _, k, sign, q in reversed(self.bubbles()):          # the front ones first
            if (k, sign) in AXIS_VIEWS and math.hypot(pos.x() - q.x(), pos.y() - q.y()) <= 8.5:
                return ('axis', k, sign)
        return 'ball'

    def tooltip(self, what) -> str:
        if isinstance(what, tuple) and what[0] == 'bar':
            return self.vp.bar.tooltip(what[1])
        if what == 'ball':
            return 'Drag to orbit · click an axis to look along it'
        if isinstance(what, tuple):
            name = {(0, 1): 'front', (0, -1): 'rear', (1, 1): 'left side', (1, -1): 'right side', (2, 1): 'top'}
            return f'Look at the {name[(what[1], what[2])]}'
        return next((t for k, _, t in self.BUTTONS if k == what), '')

    def cursor(self, what):
        if what in ('ball', 'pan'):
            return Qt.CursorShape.OpenHandCursor
        if what == 'zoom':
            return Qt.CursorShape.SizeVerCursor
        if what is not None:
            return Qt.CursorShape.PointingHandCursor
        return None

    def paint(self, p: QPainter):
        if not self.enabled():
            return
        from . import icons
        from .theme import THEME
        dark = THEME.name != 'light'
        c = self.centre()
        p.save()
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        on_ball = self.hot is not None
        bg = THEME.color('panel', 0.78 if on_ball else 0.42)
        p.setPen(QPen(THEME.color('strong', 0.9 if on_ball else 0.5), 1.0))
        p.setBrush(bg)
        p.drawEllipse(c, BALL_R, BALL_R)
        for depth, k, sign, q in self.bubbles():
            col = QColor(AXIS_COLORS[k])
            front = depth >= 0
            hot = self.hot == ('axis', k, sign)
            if sign > 0:
                p.setPen(QPen(col, 1.6))
                p.drawLine(c, q)
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(col.lighter(125) if hot else col)
                r = 8.0 if hot else 7.0
                p.drawEllipse(q, r, r)
                p.setPen(QColor(18, 18, 17) if dark else QColor(250, 250, 248))
                f = p.font()
                f.setPointSizeF(7.5)
                f.setBold(True)
                p.setFont(f)
                p.drawText(QRectF(q.x() - 7, q.y() - 7, 14, 14), Qt.AlignmentFlag.AlignCenter, 'XYZ'[k])
            else:
                col.setAlphaF(0.85 if front or hot else 0.5)
                p.setPen(QPen(col, 1.4))
                p.setBrush(col if hot else THEME.color('panel', 0.6))
                p.drawEllipse(q, 5.0, 5.0)
        for i, (key, icon, _) in enumerate(self.BUTTONS):
            b = self.button_centre(i)
            hot = self.hot == key
            p.setPen(QPen(THEME.color('strong', 0.9 if hot else 0.55), 1.0))
            p.setBrush(THEME.color('sel' if hot else 'panel', 0.9 if hot else 0.6))
            p.drawEllipse(b, BTN_R, BTN_R)
            col = THEME.tok['ink'] if hot else THEME.tok['ink2']
            if key == 'ortho' and self.vp.cam.ortho:
                col = THEME.tok['ink']
            pm = icons.pixmap(icon, col, 14)
            p.drawPixmap(QPointF(b.x() - 7, b.y() - 7), pm)
        p.restore()


class StayOpenMenu(QMenu):
    """A menu whose check boxes toggle without closing it (the overlays), so several can be changed in one go."""

    def mouseReleaseEvent(self, e):
        a = self.activeAction()
        if a is not None and a.isCheckable() and a.isEnabled() and a.menu() is None and a.actionGroup() is None:
            a.trigger()
            return
        QMenu.mouseReleaseEvent(self, e)


class ViewBar:
    """The buttons along a 3D view's top-left edge (the view, the shading, the overlays, the heat map), each opening
    a menu of the window's own actions. The provider lists them, view_bar(vp) -> [(key, text, tooltip, icon)], and
    builds the menus, view_bar_menu(vp, key) -> QMenu; a provider without them keeps its plain label."""

    H = 22.0
    GAP = 4.0
    PAD = 8.0
    ICON = 12

    def __init__(self, vp):
        self.vp = vp
        self.hot = None
        self.open = None           # the key whose menu is showing
        self.menu = None

    def items(self):
        f = getattr(self.vp.provider, 'view_bar', None)
        if f is None or self.vp.width() < 320:
            return []
        try:
            return f(self.vp) or []
        except Exception as ex:   # never let a provider's bar take the view down
            print('view bar:', ex)
            return []

    def layout(self):
        """[(key, QRectF, text, tooltip, icon)] left to right; buttons that would run into the navigation control are
        left out."""
        from .theme import sans
        fm = QFontMetricsF(sans(11))
        x, y = 6.0, 6.0
        limit = self.vp.width() - 2 * BALL_R - 40.0
        out = []
        for key, text, tip, icon in self.items():
            w = self.PAD + (self.ICON + 5 if icon else 0) + fm.horizontalAdvance(text) + 5 + 8 + self.PAD
            if x + w > limit:
                break
            out.append((key, QRectF(x, y, w, self.H), text, tip, icon))
            x += w + self.GAP
        return out

    def right(self) -> float:
        """x just past the last button (where a note can follow), or 0 without a bar."""
        lay = self.layout()
        return lay[-1][1].right() if lay else 0.0

    def hit(self, pos):
        for key, r, *_ in self.layout():
            if r.adjusted(-1, -2, 1, 2).contains(pos):
                return key
        return None

    def tooltip(self, key) -> str:
        return next((tip for k, _, _, tip, _ in self.layout() if k == key), '')

    def paint(self, p: QPainter, hot=None):
        from . import icons
        from .theme import THEME, sans
        lay = self.layout()
        if not lay:
            return
        p.save()
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.setFont(sans(11))
        fm = p.fontMetrics()
        for key, r, text, _, icon in lay:
            on = key == hot or key == self.open
            p.setPen(QPen(THEME.color('strong', 0.95 if on else 0.6), 1.0))
            p.setBrush(THEME.color('sel' if on else 'panel', 0.92 if on else 0.72))
            p.drawRoundedRect(r, 5, 5)
            x = r.left() + self.PAD
            col = THEME.tok['ink'] if on else THEME.tok['ink2']
            if icon:
                p.drawPixmap(QPointF(x, r.center().y() - self.ICON / 2), icons.pixmap(icon, col, self.ICON))
                x += self.ICON + 5
            p.setPen(THEME.color('ink' if on else 'ink2'))
            p.drawText(QPointF(x, r.center().y() + fm.ascent() / 2 - 1.5), text)
            x += fm.horizontalAdvance(text) + 5
            p.drawPixmap(QPointF(x, r.center().y() - 4), icons.pixmap('chev-d', col, 8))
        p.restore()

    def popup(self, key):
        """Open key's menu under its button."""
        lay = {k: r for k, r, *_ in self.layout()}
        f = getattr(self.vp.provider, 'view_bar_menu', None)
        if key not in lay or f is None:
            return None
        m = f(self.vp, key)
        if m is None:
            return None
        from PyQt6.QtCore import QPoint
        self.menu, self.open = m, key
        m.aboutToHide.connect(self._closed)
        r = lay[key]
        m.popup(self.vp.mapToGlobal(QPoint(int(r.left()), int(r.bottom() + 3))))
        self.vp.update()
        return m

    def _closed(self):
        self.open = None
        self.vp.update()
