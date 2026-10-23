"""Drive-log check in the UI: the overlay viewer (a sampled camera frame with the lidar points projected at the plan's or
the corrected rotation, coloured by range) and the timeline of each stream's data and gaps over the drive."""
from __future__ import annotations


import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QFont, QImage, QLinearGradient, QPainter, QPen
from PyQt6.QtWidgets import QToolTip, QWidget

from ..core import drivelog as DL
from .charts import ChartBase
from .theme import STATUS, THEME, mono, sans

SEV_TONE = {'fault': 'critical', 'warning': 'warning', 'ok': 'good'}


def sev_color(sev) -> QColor:
    return QColor(STATUS[SEV_TONE.get(sev, 'warning')])


class OverlayView(QWidget):
    """One camera frame (the thumbnail the check kept) with the lidar points drawn over it: each a 2 px dot coloured by
    its range from the camera (near yellow, far violet, on a log scale), a caption on top and the range scale."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 240)
        self.img: QImage | None = None
        self.pts = None                 # (x, y, rgb) in thumbnail pixels
        self.caption = ''
        self.note = ''
        self.empty = 'Run the check to see the lidar points over the camera images.'
        THEME.changed.connect(lambda *_: self.update())

    def clear(self, empty=None):
        self.img, self.pts, self.caption, self.note = None, None, '', ''
        if empty is not None:
            self.empty = empty
        self.update()

    def set_frame(self, jpeg: bytes, pts, caption='', note=''):
        img = QImage.fromData(jpeg, 'JPG') if jpeg else QImage()
        self.img = None if img.isNull() else img
        self.pts, self.caption, self.note = pts, caption, note
        self.update()

    def image_rect(self) -> QRectF:
        """Where the thumbnail is drawn: fitted into the widget under the caption, aspect kept."""
        if self.img is None:
            return QRectF()
        top, bottom = 22.0, 26.0
        W, H = self.width(), self.height() - top - bottom
        iw, ih = self.img.width(), self.img.height()
        s = min(W / max(iw, 1), H / max(ih, 1))
        w, h = iw * s, ih * s
        return QRectF((W - w) / 2, top + (H - h) / 2, w, h)

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
        if self.pts is not None and len(self.pts[0]):
            x, y, col = self.pts
            s = r.width() / self.img.width()
            X = r.left() + (np.asarray(x) + 0.5) * s
            Y = r.top() + (np.asarray(y) + 0.5) * s
            inside = (X >= r.left()) & (X <= r.right()) & (Y >= r.top()) & (Y <= r.bottom())
            p.setPen(Qt.PenStyle.NoPen)
            p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
            for xi, yi, c in zip(X[inside], Y[inside], np.asarray(col)[inside]):
                p.fillRect(QRectF(xi - 1.0, yi - 1.0, 2.0, 2.0), QColor(int(c[0]), int(c[1]), int(c[2])))
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setFont(sans(12, QFont.Weight.DemiBold))
        p.setPen(THEME.color('ink'))
        p.drawText(QRectF(4, 0, self.width() - 8, 20), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                   self.caption)
        p.setFont(sans(11))
        p.setPen(THEME.color('muted'))
        p.drawText(QRectF(4, 0, self.width() - 8, 20), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                   self.note)
        # the range scale
        y0 = self.height() - 18
        x0, w = 40.0, min(180.0, self.width() - 120.0)
        g = QLinearGradient(x0, 0, x0 + w, 0)
        lo, hi = 1.0, 40.0
        for k in range(9):
            t = k / 8
            rgb = DL.range_colors(np.array([lo * (hi / lo) ** t]), lo, hi)[0]
            g.setColorAt(t, QColor(int(rgb[0]), int(rgb[1]), int(rgb[2])))
        p.fillRect(QRectF(x0, y0, w, 8), QBrush(g))
        p.setFont(mono(10))
        p.setPen(THEME.color('ink2'))
        p.drawText(QRectF(4, y0 - 3, 34, 14), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, '1 m')
        p.drawText(QRectF(x0 + w + 4, y0 - 3, 260, 14), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                   '40 m   range from the camera')
        p.end()


class TimelineView(ChartBase):
    """Each stream over the drive: a bar from its first to its last message, its gaps (intervals over the drop
    factor × its period) in red, at least 2 px wide, a missing stream hatched. Hover a row for its figures."""
    LEFT, ROW, TOP, AXIS = 56, 20, 22, 26

    def __init__(self, parent=None):
        super().__init__(parent)
        self.title = 'Data and gaps per stream'
        self.rows = []                  # dicts: tag, color, first, last, gaps, missing, status, tip
        self.duration = 0.0
        self.setMinimumHeight(self.TOP + self.AXIS + self.ROW)

    def set_rows(self, rows, duration):
        self.rows, self.duration = rows, max(float(duration), 0.0)
        self.setMinimumHeight(self.TOP + self.AXIS + self.ROW * max(len(rows), 1))
        self.update()

    def _x(self, s):
        W = self.width() - self.LEFT - 12
        return self.LEFT + W * (s / self.duration if self.duration > 0 else 0.0)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.paint_title(p, f'{self.duration:.1f} s' if self.duration else '')
        if not self.rows:
            p.setFont(sans(11))
            p.setPen(THEME.color('muted'))
            p.drawText(QRectF(0, self.TOP, self.width(), self.ROW * 2), Qt.AlignmentFlag.AlignVCenter,
                       'No streams checked yet.')
            p.end()
            return
        x1 = self._x(self.duration)
        for i, r in enumerate(self.rows):
            y = self.TOP + i * self.ROW
            if self.hover == i:
                p.fillRect(QRectF(0, y, self.width(), self.ROW), THEME.color('raised'))
            p.setFont(mono(11, QFont.Weight.DemiBold))
            p.setPen(r['color'])
            p.drawText(QRectF(4, y, self.LEFT - 8, self.ROW), Qt.AlignmentFlag.AlignVCenter, r['tag'])
            if r['missing']:
                p.setPen(QPen(THEME.color('strong'), 1))
                xx = float(self.LEFT)
                while xx < x1:
                    p.drawLine(QPointF(xx, y + self.ROW - 5), QPointF(min(xx + 7, x1), y + 5))
                    xx += 9
                continue
            a, b = self._x(r['first']), self._x(r['last'])
            c = QColor(r['color'])
            c.setAlphaF(0.55)
            p.fillRect(QRectF(a, y + 7, max(b - a, 1.0), self.ROW - 14), c)
            gc = QColor(STATUS['critical'])
            for g0, g1 in r['gaps']:
                ga, gb = self._x(g0), self._x(g1)
                p.fillRect(QRectF(ga, y + 3, max(gb - ga, 2.0), self.ROW - 6), gc)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(sev_color(r['status']))
            p.drawEllipse(QPointF(self.width() - 6, y + self.ROW / 2), 3, 3)
        # the time axis
        ya = self.TOP + len(self.rows) * self.ROW + 4
        p.setPen(QPen(THEME.color('strong'), 1))
        p.drawLine(QPointF(self.LEFT, ya), QPointF(x1, ya))
        step = next((s for s in (0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 30, 60, 120, 300, 600, 1800, 3600)
                     if self.duration / s <= 10), 7200)
        p.setFont(mono(10))
        p.setPen(THEME.color('muted'))
        t = 0.0
        while t <= self.duration + 1e-9:
            x = self._x(t)
            p.drawLine(QPointF(x, ya), QPointF(x, ya + 4))
            p.drawText(QRectF(x - 30, ya + 4, 60, 14), Qt.AlignmentFlag.AlignHCenter, f'{t:g} s')
            t += step
        p.end()

    def mouseMoveEvent(self, e):
        pos = e.position()
        i = int((pos.y() - self.TOP) // self.ROW)
        hit = i if 0 <= i < len(self.rows) and pos.y() >= self.TOP else None
        if hit != self.hover:
            self.hover = hit
            self.update()
        if hit is None:
            QToolTip.hideText()
            return
        r = self.rows[hit]
        tip = r['tip']
        if self.duration > 0 and not r['missing']:
            t = (pos.x() - self.LEFT) / max(self.width() - self.LEFT - 12, 1) * self.duration
            g = next(((a, b) for a, b in r['gaps'] if a - 0.02 * self.duration <= t <= b + 0.02 * self.duration),
                     None)
            if g is not None:
                tip += f'\nGap from {g[0]:.3f} s to {g[1]:.3f} s ({(g[1] - g[0]) * 1000:.0f} ms)'
        QToolTip.showText(e.globalPosition().toPoint(), tip, self)


def timeline_rows(res, var) -> list:
    rows = []
    for e in res.streams:
        col = QColor(DL.stream_color(var, e))
        if e['kind'] == 'camera' and var is not None:
            c = next((c for c in var.cameras if c.uid == e['uid']), None)
            if c is not None:
                col = THEME.cam(c.slot)
        if e['topic'] is None:
            tip = f'{e["tag"]}: no topic in the log'
        else:
            tip = (f'{e["tag"]} · {e["topic"]}\n{e["messages"]} messages'
                   + (f' at {e["rate_hz"]:.2f} Hz' if e.get('rate_hz') else '')
                   + (f' (plan {e["expected_hz"]:g} Hz)' if e.get('expected_hz') else '')
                   + (f'\n{e["drops"]} dropped, longest gap {e["longest_gap_ms"]:.0f} ms' if e.get('drops') else ''))
        first, last = e.get('span_s') or (e.get('first_s', 0.0), e.get('last_s', 0.0))
        rows.append({'tag': e['tag'], 'color': col, 'first': first, 'last': last,
                     'gaps': e.get('gaps', []), 'missing': e['topic'] is None, 'status': e['status'] or 'ok',
                     'tip': tip})
    return rows
