"""Sample dataset preview drawing: the camera thumbnails of one instant of the synthetic drive, and a top view of the
ego's surroundings at that instant (forward up, left to the left): the lidar points coloured by height, the radar
detections, the ground-truth boxes, the road's edges and the body's outline."""
from __future__ import annotations

import math

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPolygonF
from PyQt6.QtWidgets import QToolTip

from ..core import sensor_coverage as SC
from .charts import ChartBase
from .theme import SEQ, THEME, mono, sans

BOX_COLORS = {'vehicle.truck': '#9085e9', 'human.pedestrian.adult': '#e66767', 'vehicle.bicycle': '#c98500'}
LEGEND_H = 18
Z_LO, Z_HI = -0.5, 4.0                    # lidar height colour range, metres in the ego frame


def _qimage(img: np.ndarray) -> QImage:
    """A QImage that owns a copy of an RGB uint8 array."""
    a = np.ascontiguousarray(img)
    h, w = a.shape[:2]
    return QImage(a.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


def _ramp() -> np.ndarray:
    """(n, 3) uint8: the app's sequential ramp from low to high points, the end that stands out from the theme's
    background last (dark theme: light for high points; light theme: dark)."""
    seq = SEQ[:10][::-1] if THEME.color('base').lightness() < 128 else SEQ[3:]
    return np.array([[int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)] for c in seq], np.uint8)


def box_color(category) -> QColor:
    """Outline of a ground-truth box: cars in the theme's ink (the commonest, apart from the lidar's blues)."""
    return QColor(BOX_COLORS[category]) if category in BOX_COLORS else THEME.color('ink2')


class CameraGrid(ChartBase):
    """The preview's camera images in a grid (three across), each with its tag and channel."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.cams = []                        # [(tag, slot, channel, QImage)]
        self.empty = 'No preview yet.'
        self.setMinimumHeight(160)

    def set_cameras(self, cams):
        self.cams = [(c['tag'], c['slot'], c['channel'], _qimage(c['image'])) for c in cams]
        self.update()

    def set_empty(self, text):
        self.cams = []
        self.empty = text
        self.update()

    def _layout(self):
        n = len(self.cams)
        cols = min(3, max(n, 1))
        rows = max(1, math.ceil(n / cols))
        gap = 6
        W, H = self.width(), self.height()
        cw = (W - gap * (cols - 1)) / cols
        im = self.cams[0][3] if self.cams else None
        ar = im.height() / im.width() if im is not None and im.width() else 0.8
        ch = min(cw * ar, (H - gap * (rows - 1) - 16 * rows) / rows)
        cw = ch / ar
        out = []
        for i in range(n):
            r, c = divmod(i, cols)
            out.append(QRectF(c * (cw + gap), r * (ch + gap + 16) + 16, cw, ch))
        return out

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        if not self.cams:
            p.setFont(sans(11))
            p.setPen(THEME.color('muted'))
            p.drawText(QRectF(0, 0, self.width(), self.height()), Qt.AlignmentFlag.AlignCenter, self.empty)
            p.end()
            return
        for (tag, slot, channel, img), r in zip(self.cams, self._layout()):
            p.drawImage(r, img)
            p.setPen(QPen(THEME.color('hair'), 1))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(r)
            p.fillRect(QRectF(r.left(), r.top() - 13, 8, 8), THEME.cam(slot))
            p.setFont(mono(11, QFont.Weight.DemiBold))
            p.setPen(THEME.color('ink'))
            p.drawText(QPointF(r.left() + 12, r.top() - 5), tag)
            p.setFont(sans(10))
            p.setPen(THEME.color('muted'))
            p.drawText(QRectF(r.left() + 40, r.top() - 15, r.width() - 40, 12),
                       Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, channel)
        p.end()


class TopView(ChartBase):
    """The ego's surroundings at the preview instant, forward up, `reach` metres each way from the vehicle origin."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.data = None
        self.outline = None                  # body seen from above (N, 2), vehicle frame
        self.reach = 40.0
        self.empty = 'No preview yet.'
        self._img = None
        self._key = None
        self.setMinimumSize(220, 220)

    def set_data(self, data, outline=None):
        self.data, self.outline = data, outline
        self._key = None
        self.update()

    def set_reach(self, metres):
        self.reach = float(metres)
        self._key = None
        self.update()

    def set_empty(self, text):
        self.data = None
        self.empty = text
        self.update()

    def _geom(self):
        s = max(min(self.width(), self.height() - 20 - LEGEND_H), 10)
        k = s / (2 * self.reach)
        cx, cy = self.width() / 2, 20 + s / 2                     # the vehicle origin at the centre
        return k, cx, cy

    def _pt(self, x, y) -> QPointF:
        k, cx, cy = self._geom()
        return QPointF(cx - y * k, cy - x * k)

    def _points_image(self):
        """The lidar points rasterised (RGBA), coloured by height, cached per size and data."""
        key = (self.width(), self.height(), self.reach, id(self.data), THEME.color('base').name())
        if key == self._key:
            return self._img
        W, H = self.width(), self.height()
        rgba = np.zeros((H, W, 4), np.uint8)
        P = self.data['lidar'] if self.data is not None else np.zeros((0, 4))
        if len(P):
            k, cx, cy = self._geom()
            u = np.round(cx - P[:, 1] * k).astype(int)
            v = np.round(cy - P[:, 0] * k).astype(int)
            ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
            z = np.clip((P[ok, 2] - Z_LO) / (Z_HI - Z_LO), 0, 1)
            ramp = _ramp()
            order = np.argsort(z)                                # higher points drawn last
            col = ramp[np.round(z[order] * (len(ramp) - 1)).astype(int)]
            uu, vv = u[ok][order], v[ok][order]
            rgba[vv, uu, :3] = col
            rgba[vv, uu, 3] = 255
        self._img = QImage(rgba.data, W, H, 4 * W, QImage.Format.Format_RGBA8888).copy()
        self._key = key
        return self._img

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.title = 'Top view'
        self.paint_title(p, f'±{self.reach:.0f} m · forward up')
        if self.data is None:
            p.setFont(sans(11))
            p.setPen(THEME.color('muted'))
            p.drawText(QRectF(0, 20, self.width(), self.height() - 20), Qt.AlignmentFlag.AlignCenter, self.empty)
            p.end()
            return
        k, cx, cy = self._geom()
        s = 2 * self.reach * k
        frame = QRectF(cx - s / 2, cy - s / 2, s, s)
        p.fillRect(frame, THEME.color('raised'))
        p.setClipRect(frame)
        # range rings every 10 m
        p.setPen(QPen(THEME.color('hair'), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        for r in range(10, int(self.reach * 1.5) + 1, 10):
            p.drawEllipse(QPointF(cx, cy), r * k, r * k)
        # the road's edges
        p.setPen(QPen(THEME.color('strong'), 1.2))
        for line in self.data.get('road', []):
            p.drawPolyline(QPolygonF([self._pt(x, y) for x, y in line]))
        p.drawImage(QPointF(0, 0), self._points_image())
        # ground truth
        for b in self.data.get('boxes', []):
            col = box_color(b['category'])
            p.setPen(QPen(col, 1.6))
            fill = QColor(col)
            fill.setAlphaF(0.18)
            p.setBrush(fill)
            p.drawPolygon(QPolygonF([self._pt(x, y) for x, y in b['corners']]))
        # radar detections
        rc = QColor(SC.MODALITY_COLORS['radar'])
        p.setPen(QPen(rc, 1.5))
        p.setBrush(Qt.BrushStyle.NoBrush)
        for r in self.data.get('radar', []):
            for q in r['points']:
                p.drawEllipse(self._pt(q[0], q[1]), 3.5, 3.5)
        # the body
        if self.outline is not None and len(self.outline) > 2:
            p.setPen(QPen(THEME.color('ink'), 1.2))
            body = QColor(THEME.color('body'))
            body.setAlphaF(0.6)
            p.setBrush(body)
            p.drawPolygon(QPolygonF([self._pt(x, y) for x, y in self.outline]))
        p.setClipping(False)
        # legend
        p.setFont(sans(10))
        x = frame.left()
        y = frame.bottom() + 13
        ramp = _ramp()
        items = [('Lidar, low to high', None), ('Radar', rc)] + \
                [(n, box_color(c)) for n, c in (('Car', 'vehicle.car'), ('Truck', 'vehicle.truck'),
                                                ('Pedestrian', 'human.pedestrian.adult'),
                                                ('Cyclist', 'vehicle.bicycle'))]
        for text, col in items:
            if col is None:                       # the height ramp
                for i in range(len(ramp)):
                    p.fillRect(QRectF(x + 2 * i, y - 7, 2, 8), QColor(*map(int, ramp[i])))
                x += 2 * len(ramp) - 8
            else:
                p.fillRect(QRectF(x, y - 7, 8, 8), col)
            p.setPen(THEME.color('ink2'))
            p.drawText(QPointF(x + 11, y), text)
            x += 15 + p.fontMetrics().horizontalAdvance(text) + 8
        p.end()

    def mouseMoveEvent(self, e):
        if self.data is None:
            return
        k, cx, cy = self._geom()
        pos = e.position()
        xv, yv = (cy - pos.y()) / k, (cx - pos.x()) / k
        hit = None
        for b in self.data.get('boxes', []):
            c = np.asarray(b['corners'])
            if c[:, 0].min() - 0.3 <= xv <= c[:, 0].max() + 0.3 and c[:, 1].min() - 0.3 <= yv <= c[:, 1].max() + 0.3:
                hit = b
                break
        if hit is not None:
            QToolTip.showText(e.globalPosition().toPoint(), f'{hit["name"]} · {hit["category"]}'
                              + (' · moving' if hit['moving'] else ''), self)
        else:
            QToolTip.showText(e.globalPosition().toPoint(), f'x {xv:.1f} m, y {yv:.1f} m', self)
