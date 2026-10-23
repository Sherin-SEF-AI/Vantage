"""GNSS sky plots: one antenna's sky as a polar plot, forward (the vehicle's +x, or a tilted antenna's own +x) up the
page and left to the left as seen from above, the zenith in the middle and the antenna's horizon on the rim. Rings every
15° of elevation, the elevation mask dashed, the cells something of the vehicle blocks shaded in the colour of their
blocker (dimmer below the mask, where the receiver ignores the sky anyway), the skyline (the top of what rises from
the antenna's horizon) as a line, and a legend of the blockers with their share of the sky above the mask. Hovering a
cell names its blocker; a click picks the antenna."""
from __future__ import annotations

import math

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QToolTip

from ..core import gnss as G
from .charts import ChartBase
from .theme import THEME, mono, sans

TITLE_H = 20
LEG_H = 17
LEG_ROWS = 4
RING_STEP = 15


def blocker_color(b, alpha=None) -> QColor:
    """A blocker's colour: a camera its colour slot in the current theme, anything else its kind's hue family."""
    c = THEME.cam(b['slot']) if b['kind'] == 'camera' else QColor(b['color'])
    c = QColor(c)
    if alpha is not None:
        c.setAlphaF(alpha)
    return c


class SkyPlot(ChartBase):
    """One antenna's sky (core.gnss.SkyView)."""
    picked = pyqtSignal(str)                  # the antenna's uid

    def __init__(self, parent=None):
        super().__init__(parent)
        self.sky: G.SkyView | None = None
        self.selected = False
        self.hover_blk = None                 # index of the hovered blocker
        self._img = None
        self._img_key = None
        self.setMinimumSize(250, 320)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_sky(self, sky, selected=False):
        if sky is not self.sky:
            self.hover_blk = None
        self.sky, self.selected = sky, selected
        self.update()

    # ---------------------------------------------------------------- geometry
    def _legend(self):
        if self.sky is None:
            return []
        return [b for b in self.sky.blockers if b['share_all'] > 0][:LEG_ROWS * 2]

    def _disc(self):
        """(centre, radius) of the plot."""
        W, H = self.width(), self.height()
        rows = max(1, min(LEG_ROWS, math.ceil(len(self._legend()) / 2)))
        avail = H - TITLE_H - 22 - rows * LEG_H - 26
        R = max(min(W - 44, avail) / 2, 20.0)
        return QPointF(W / 2, TITLE_H + 22 + R), R

    def to_sky(self, pos):
        """(azimuth, elevation) in degrees under a widget point, or None outside the disc."""
        c, R = self._disc()
        dx, dy = pos.x() - c.x(), pos.y() - c.y()
        r = math.hypot(dx, dy) / R
        if r > 1.0:
            return None
        return math.degrees(math.atan2(-dx, -dy)) % 360.0, 90.0 * (1.0 - r)

    def point(self, az, el) -> QPointF:
        c, R = self._disc()
        rr = R * (90.0 - el) / 90.0
        a = math.radians(az)
        return QPointF(c.x() - math.sin(a) * rr, c.y() - math.cos(a) * rr)

    # ---------------------------------------------------------------- the cells
    def _image(self, R):
        """The blocked cells as an image of the disc (cached on the sky, size, theme and hover)."""
        sky = self.sky
        size = int(2 * R) + 2
        key = (id(sky), size, THEME.name, self.hover_blk)
        if self._img_key == key:
            return self._img
        yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
        dx, dy = xx + 0.5 - size / 2, yy + 0.5 - size / 2
        r = np.hypot(dx, dy) / R
        inside = r <= 1.0
        el = 90.0 * (1.0 - r)
        az = np.degrees(np.arctan2(-dx, -dy)) % 360.0
        A = len(sky.az)
        ai = np.rint(az / (360.0 / A)).astype(int) % A
        ei = np.clip(np.searchsorted(sky.el_edges, el, 'right') - 1, 0, len(sky.el_edges) - 2)
        who = np.where(inside, sky.who[ei, ai], -1)
        lut = np.zeros((len(sky.blockers) + 1, 4), np.float32)
        for i, b in enumerate(sky.blockers):
            c = blocker_color(b)
            dim = self.hover_blk is not None and self.hover_blk != i
            lut[i] = (c.red(), c.green(), c.blue(), 255 * (0.25 if dim else 0.9))
        img = lut[np.where(who >= 0, who, len(sky.blockers))]
        low = inside & (el < sky.mask_deg)
        img[..., 3] = np.where(low & (who >= 0), img[..., 3] * 0.5, img[..., 3])
        arr = np.ascontiguousarray(np.clip(img, 0, 255).astype(np.uint8))
        q = QImage(arr.data, size, size, 4 * size, QImage.Format.Format_RGBA8888).copy()
        self._img, self._img_key = q, key
        return q

    # ---------------------------------------------------------------- painting
    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = self.width(), self.height()
        p.setPen(QPen(THEME.color('ink') if self.selected else THEME.color('hair'), 1.5 if self.selected else 1))
        p.setBrush(THEME.color('raised'))
        p.drawRoundedRect(QRectF(0.75, 0.75, W - 1.5, H - 1.5), 4, 4)
        sky = self.sky
        if sky is None:
            p.end()
            return
        # ---- title
        p.setFont(mono(12, QFont.Weight.DemiBold))
        p.setPen(THEME.color('ink'))
        p.drawText(QRectF(10, 4, W - 20, TITLE_H), Qt.AlignmentFlag.AlignVCenter, sky.tag)
        tw = p.fontMetrics().horizontalAdvance(sky.tag)
        p.setFont(sans(11))
        p.setPen(THEME.color('ink2'))
        note = (f'{sky.visible * 100:.1f} % open above {sky.mask_deg:g}° · ~{sky.expected_sats:.0f} of '
                f'{sky.n_sats:.0f} satellites')
        p.drawText(QRectF(18 + tw, 4, W - 28 - tw, TITLE_H), Qt.AlignmentFlag.AlignVCenter,
                   p.fontMetrics().elidedText(note, Qt.TextElideMode.ElideRight, int(W - 28 - tw)))
        c, R = self._disc()
        # ---- the disc, the band below the mask, the blocked cells
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(THEME.color('panel'))
        p.drawEllipse(c, R, R)
        rm = R * (90.0 - sky.mask_deg) / 90.0
        if sky.mask_deg > 0:
            band = QPainterPath()
            band.addEllipse(c, R, R)
            inner = QPainterPath()
            inner.addEllipse(c, rm, rm)
            p.fillPath(band.subtracted(inner), THEME.color('sel'))
        img = self._image(R)
        p.drawImage(QPointF(c.x() - img.width() / 2, c.y() - img.height() / 2), img)
        # ---- rings, spokes, the mask, the horizon profile
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(THEME.color('strong'), 1))
        for el in range(0, 90, RING_STEP):
            r = R * (90 - el) / 90
            p.drawEllipse(c, r, r)
        for ang in range(0, 360, 45):
            p.drawLine(self.point(ang, 0), self.point(ang, 90))
        pen = QPen(THEME.color('ink2'), 1.2, Qt.PenStyle.DashLine)
        p.setPen(pen)
        p.drawEllipse(c, rm, rm)
        if np.isfinite(sky.skyline).any():
            path = QPainterPath()
            hz = np.where(np.isfinite(sky.skyline), sky.skyline, 0.0)
            for i, (az, h) in enumerate(zip(np.r_[sky.az, sky.az[:1]], np.r_[hz, hz[:1]])):
                q = self.point(float(az), float(h))
                if i == 0:
                    path.moveTo(q)
                else:
                    path.lineTo(q)
            p.setPen(QPen(THEME.color('ink'), 1.2))
            p.drawPath(path)
        p.setFont(mono(10))
        p.setPen(THEME.color('muted'))
        for el in range(RING_STEP, 90, RING_STEP):
            q = self.point(0, el)
            p.drawText(QPointF(q.x() + 3, q.y() + 11), f'{el}°')
        p.setFont(sans(11, QFont.Weight.DemiBold))
        p.setPen(THEME.color('ink2'))
        for text, ang in (('F', 0), ('L', 90), ('B', 180), ('R', 270)):
            q = self.point(ang, -8)
            p.drawText(QRectF(q.x() - 10, q.y() - 8, 20, 16), Qt.AlignmentFlag.AlignCenter, text)
        # ---- legend
        p.setFont(sans(11))
        leg = self._legend()
        y0 = c.y() + R + 20
        colw = (W - 20) / 2
        if not leg:
            p.setPen(THEME.color('muted'))
            p.drawText(QRectF(10, y0, W - 20, LEG_H), Qt.AlignmentFlag.AlignVCenter,
                       'Open sky: nothing of the vehicle blocks it')
        for i, b in enumerate(leg[:LEG_ROWS * 2]):
            x, y = 10 + (i % 2) * colw, y0 + (i // 2) * LEG_H
            j = sky.blockers.index(b)
            p.fillRect(QRectF(x, y + 4, 9, 9), blocker_color(b))
            p.setPen(THEME.color('ink' if self.hover_blk == j else 'ink2'))
            text = f'{b["name"]} {b["share"] * 100:.1f} %' if b['share'] >= 0.0005 else f'{b["name"]} below the mask'
            p.drawText(QRectF(x + 14, y, colw - 18, LEG_H), Qt.AlignmentFlag.AlignVCenter,
                       p.fontMetrics().elidedText(text, Qt.TextElideMode.ElideRight, int(colw - 18)))
        p.end()

    # ---------------------------------------------------------------- interaction
    def describe(self, az, el) -> str:
        sky = self.sky
        b, d = sky.at(az, el)
        where = f'{sky.tag} · azimuth {az:.0f}°, elevation {el:.0f}°'
        if b is None:
            return where + (' · below the mask' if el < sky.mask_deg else ' · open sky')
        kind = G.KIND_WORD.get(b['kind'], b['kind'])
        return (f'{where}\nBlocked by {b["name"]} ({kind}), {d:.2f} m away'
                f'{" · below the mask" if el < sky.mask_deg else ""}\n{b["name"]} blocks {b["share"] * 100:.1f} % of '
                f'the sky above the mask, up to {b["max_el"]:.0f}°')

    def mouseMoveEvent(self, e):
        if self.sky is None:
            return
        hit = self.to_sky(e.position())
        blk = None
        if hit is not None:
            b, _ = self.sky.at(*hit)
            blk = self.sky.blockers.index(b) if b is not None else None
            QToolTip.showText(e.globalPosition().toPoint(), self.describe(*hit), self)
        else:
            QToolTip.hideText()
        if blk != self.hover_blk:
            self.hover_blk = blk
            self.update()

    def leaveEvent(self, e):
        self.hover_blk = None
        super().leaveEvent(e)

    def mousePressEvent(self, e):
        if self.sky is not None and e.button() == Qt.MouseButton.LeftButton:
            self.picked.emit(self.sky.uid)
