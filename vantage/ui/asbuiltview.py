"""As-built check in the UI: one camera's frame (or the median of its frames) with the real outline of the vehicle
(what stayed the same while driving) and the simulated one (the plan's, or at the fitted pose) drawn over it, the
stretches the frames could not decide, the measured and the predicted image circle, and a legend."""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QWidget

from ..core import asbuilt as AB
from .theme import THEME, mono, sans

LEGEND = (('real', 'Real outline', AB.REAL_COLOR, Qt.PenStyle.SolidLine),
          ('sim', 'Simulated', AB.SIM_COLOR, Qt.PenStyle.SolidLine),
          ('undecided', 'Undecided', AB.UND_COLOR, Qt.PenStyle.DashLine),
          ('circle_real', 'Image circle', AB.CIRCLE_REAL, Qt.PenStyle.SolidLine),
          ('circle_sim', 'Lens model', AB.CIRCLE_SIM, Qt.PenStyle.DashLine))


class AsBuiltView(QWidget):
    """The picture (a JPEG) fitted into the widget with the layers of core.asbuilt.overlay_layers drawn over it in the
    picture's pixel coordinates, a caption on top and the legend under it."""
    TOP, BOTTOM = 22.0, 24.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 240)
        self.img: QImage | None = None
        self.layers = None
        self.caption = ''
        self.note = ''
        self.empty = 'Run the check to see each camera\'s real outline against the plan\'s.'
        THEME.changed.connect(self._theme_changed)      # a bound slot: disconnected when the widget goes

    def _theme_changed(self, *_):
        self.update()

    def clear(self, empty=None):
        self.img, self.layers, self.caption, self.note = None, None, '', ''
        if empty is not None:
            self.empty = empty
        self.update()

    def set_frame(self, jpeg: bytes, layers: dict | None, caption='', note=''):
        img = QImage.fromData(jpeg, 'JPG') if jpeg else QImage()
        self.img = None if img.isNull() else img
        self.layers, self.caption, self.note = layers, caption, note
        self.update()

    def image_rect(self) -> QRectF:
        """Where the picture is drawn: fitted under the caption, aspect kept."""
        if self.img is None:
            return QRectF()
        W, H = self.width(), self.height() - self.TOP - self.BOTTOM
        iw, ih = self.img.width(), self.img.height()
        s = min(W / max(iw, 1), H / max(ih, 1))
        w, h = iw * s, ih * s
        return QRectF((W - w) / 2, self.TOP + (H - h) / 2, w, h)

    def _path(self, line, r, s):
        path = QPainterPath()
        pts = np.asarray(line, float)
        if len(pts) < 2:
            return path
        X = r.left() + (pts[:, 0] + 0.5) * s
        Y = r.top() + (pts[:, 1] + 0.5) * s
        path.moveTo(QPointF(X[0], Y[0]))
        for x, y in zip(X[1:], Y[1:]):
            path.lineTo(QPointF(x, y))
        path.closeSubpath()
        return path

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), THEME.color('viewport'))
        if self.img is None:
            p.setFont(sans(12))
            p.setPen(THEME.color('muted'))
            p.drawText(QRectF(self.rect()).adjusted(20, 0, -20, 0), Qt.AlignmentFlag.AlignCenter |
                       Qt.TextFlag.TextWordWrap, self.empty)
            p.end()
            return
        r = self.image_rect()
        p.drawImage(r, self.img)
        p.fillRect(r, QColor(0, 0, 0, 50))                 # the lines read better on a slightly dimmed picture
        L = self.layers or {}
        s = r.width() / self.img.width()
        p.save()
        p.setClipRect(r)
        p.setBrush(Qt.BrushStyle.NoBrush)
        for key, _, col, style in LEGEND:
            if key.startswith('circle'):
                c = L.get(key)
                if c is None:
                    continue
                x, y, rr = c
                pen = QPen(QColor(col), 1.5, style)
                p.setPen(pen)
                p.drawEllipse(QPointF(r.left() + (x + 0.5) * s, r.top() + (y + 0.5) * s), rr * s, rr * s)
                continue
            width = 2.2 if key == 'real' else 1.4
            p.setPen(QPen(QColor(col), width, style, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
            for line in L.get(key) or []:
                p.drawPath(self._path(line, r, s))
        p.restore()
        p.setFont(sans(12, QFont.Weight.DemiBold))
        p.setPen(THEME.color('ink'))
        p.drawText(QRectF(4, 0, self.width() - 8, self.TOP - 2), Qt.AlignmentFlag.AlignLeft |
                   Qt.AlignmentFlag.AlignVCenter, self.caption)
        p.setFont(sans(11))
        p.setPen(THEME.color('muted'))
        p.drawText(QRectF(4, 0, self.width() - 8, self.TOP - 2), Qt.AlignmentFlag.AlignRight |
                   Qt.AlignmentFlag.AlignVCenter, self.note)
        # the legend
        y = self.height() - self.BOTTOM / 2
        x = 8.0
        p.setFont(mono(10))
        for key, text, col, style in LEGEND:
            if key.startswith('circle') and not L.get(key):
                continue
            p.setPen(QPen(QColor(col), 2.0, style))
            p.drawLine(QPointF(x, y), QPointF(x + 18, y))
            p.setPen(THEME.color('ink2'))
            tw = p.fontMetrics().horizontalAdvance(text)
            p.drawText(QRectF(x + 22, y - 8, tw + 4, 16), Qt.AlignmentFlag.AlignVCenter, text)
            x += 22 + tw + 16
        p.end()
