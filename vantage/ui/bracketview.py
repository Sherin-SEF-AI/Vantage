"""3D preview of a mount bracket (Analysis ▸ Mount brackets): the printable part shaded, the camera housing drawn
see-through in its colour, and an inset of the camera's image with the share the bracket blocks. Drag to orbit, wheel
to zoom, double-click to reset. Software-rendered with a z-buffer (core.meshing.render): the same on every platform
and with no OpenGL context."""
from __future__ import annotations

import math

import numpy as np
from PyQt6.QtCore import QRectF, Qt, QTimer
from PyQt6.QtGui import QColor, QImage, QPainter, QPen
from PyQt6.QtWidgets import QWidget

from ..core import brackets as BR
from ..core import meshing as M
from .theme import THEME, STATUS, mono, sans

HOME = (-55.0, 28.0)          # azimuth, elevation of the starting view (degrees)


def _rgb(c: QColor):
    return (c.red(), c.green(), c.blue())


class BracketView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(240, 200)
        self.setMouseTracking(True)
        self.b = None
        self.slot = 0
        self.message = 'Generate the brackets'
        self.mesh = None             # display mesh (thinned), bracket frame mm
        self.fast = None             # coarser, while dragging
        self.yaw, self.elev = HOME
        self.zoom = 1.0
        self._img = None
        self._key = None
        self._drag = None
        self._settle = QTimer(self)
        self._settle.setSingleShot(True)
        self._settle.setInterval(160)
        self._settle.timeout.connect(self._refine)
        self._dragging = False
        THEME.changed.connect(self._retheme)

    def _retheme(self, *_):
        self._key = None
        self.update()

    # ------------------------------------------------------------------ content
    def set_bracket(self, b, slot=0, message=''):
        """Show bracket b (core.brackets.Bracket) in camera colour slot; without a mesh, the message."""
        same = b is not None and self.b is not None and b is self.b
        self.b, self.slot = b, slot
        self.message = message
        if not same:
            self.mesh = self.fast = None
            if b is not None and b.generated:
                self.mesh = b.display or BR.display_mesh(b, 30000)
                self.fast = b.display_fast or BR.display_mesh(b, 6000)
        self._key = None
        self.update()

    def reset_view(self):
        self.yaw, self.elev = HOME
        self.zoom = 1.0
        self._key = None
        self.update()

    # ------------------------------------------------------------------ drawing
    def _render(self, w, h, fast):
        b = self.b
        mesh = self.fast if fast and self.fast is not None else self.mesh
        V = mesh[0]
        HV, HF = BR.housing_mesh(b)
        allv = np.vstack([V, HV]) if len(HV) else V
        centre = (allv.min(axis=0) + allv.max(axis=0)) / 2
        rad = float(np.linalg.norm(allv - centre, axis=1).max()) or 1.0
        scale = 0.45 * min(w, h) / rad * self.zoom
        basis = M.view_basis(self.yaw, self.elev)
        bg = _rgb(THEME.color('viewport'))
        dark = THEME.color('viewport').lightness() < 128
        body = (176, 178, 182) if dark else (204, 206, 210)
        layers = [(V, mesh[1], body, 1.0), (HV, HF, _rgb(THEME.cam(self.slot)), 0.42)]
        img = M.render(layers, basis, (w, h), scale, centre, bg=bg, ss=1)        # ~70 ms at rest: no supersampling
        img = np.ascontiguousarray(img)
        q = QImage(img.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        return q

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = self.width(), self.height()
        p.fillRect(self.rect(), THEME.color('viewport'))
        if self.b is None or self.mesh is None:
            p.setFont(sans(12))
            p.setPen(THEME.color('muted'))
            p.drawText(QRectF(20, 0, W - 40, H), Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                       self.message or 'No bracket for this camera')
            p.end()
            return
        key = (W, H, round(self.yaw, 2), round(self.elev, 2), round(self.zoom, 3), self._dragging, id(self.mesh))
        if key != self._key or self._img is None:
            self._img = self._render(W, H, self._dragging)
            self._key = key
        p.drawImage(0, 0, self._img)
        self._paint_overlay(p, W, H)
        p.end()

    def _paint_overlay(self, p, W, H):
        b = self.b
        p.setFont(mono(11))
        p.setPen(THEME.color('ink2'))
        p.drawText(QRectF(10, 6, W - 20, 16), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                   f'{b.tag} · {b.face} face · {"tube clamp" if b.kind == "clamp" else "flat base"}')
        p.setFont(sans(11))
        p.setPen(THEME.color('muted'))
        p.drawText(QRectF(10, H - 20, W - 20, 16), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                   'Drag to orbit · wheel to zoom · double-click to reset')
        m = b.fov_mask
        if m is not None and m.size:
            hh, ww = m.shape
            iw = 128
            ih = max(1, int(round(iw * hh / ww)))
            rgb = np.zeros((hh, ww, 3), np.uint8)
            rgb[m == 0] = _rgb(THEME.color('panel'))
            rgb[m == 1] = _rgb(THEME.color('muted'))
            rgb[m == 2] = _rgb(QColor(STATUS['critical']))
            img = QImage(np.ascontiguousarray(rgb).data, ww, hh, 3 * ww, QImage.Format.Format_RGB888).copy()
            r = QRectF(W - iw - 10, 10, iw, ih)
            p.drawImage(r, img)
            p.setPen(QPen(THEME.color('hair'), 1))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(r)
            share = b.params.get('fov_blocked_pct', 0.0)
            p.setPen(THEME.color('ink2'))
            p.drawText(QRectF(r.left() - 40, r.bottom() + 2, iw + 40, 16),
                       Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                       f'Image: {share:.1f} % blocked' if share > 0 else 'Image: clear of the bracket')

    # ------------------------------------------------------------------ interaction
    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._drag = (e.position(), self.yaw, self.elev)

    def mouseMoveEvent(self, e):
        if self._drag is None or self.mesh is None:
            return
        pos, yaw, elev = self._drag
        d = e.position() - pos
        self.yaw = yaw - d.x() * 0.5
        self.elev = max(-89.0, min(89.0, elev + d.y() * 0.5))
        self._dragging = True
        self.update()

    def mouseReleaseEvent(self, e):
        if self._drag is not None:
            self._drag = None
            self._settle.start()

    def _refine(self):
        self._dragging = False
        self.update()

    def wheelEvent(self, e):
        if self.mesh is None:
            return
        step = e.angleDelta().y() / 120.0
        self.zoom = max(0.3, min(6.0, self.zoom * math.pow(1.15, step)))
        self.update()

    def mouseDoubleClickEvent(self, e):
        self.reset_view()
