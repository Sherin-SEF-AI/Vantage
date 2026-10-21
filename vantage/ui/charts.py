"""Plain QPainter charts: hairline axes, thin marks, hover tooltips."""
from __future__ import annotations

import math

import numpy as np
from PyQt6.QtCore import Qt, QPointF, QRectF, pyqtSignal, QLineF
from PyQt6.QtGui import QPainter, QColor, QPen, QFont, QPolygonF, QBrush
from PyQt6.QtWidgets import QWidget, QToolTip

from .theme import THEME, mono, sans, SEQ, STATUS, seq_color
from .widgets import paint_chip


class ChartBase(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.hover = None
        self.title = ''
        self.subtitle = ''
        THEME.changed.connect(lambda *_: self.update())

    def paint_title(self, p, right=''):
        if self.title:
            p.setFont(sans(12, QFont.Weight.DemiBold))
            p.setPen(THEME.color('ink'))
            p.drawText(QRectF(0, 0, self.width(), 18), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, self.title)
        if right or self.subtitle:
            p.setFont(sans(11))
            p.setPen(THEME.color('muted'))
            p.drawText(QRectF(0, 0, self.width(), 18), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, right or self.subtitle)

    def leaveEvent(self, e):
        self.hover = None
        self.update()


class BarList(ChartBase):
    """Rows: chip + tag | bar in identity colour | value. Click selects the camera."""
    picked = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.rows = []      # (uid, slot, tag, value)
        self.selected = set()
        self.unit = 'm²'
        self.vmax = None
        self.setMinimumHeight(120)

    def set_rows(self, rows, selected=()):
        self.rows = rows
        self.selected = set(selected)
        self.update()

    def _geom(self):
        top = 22
        rh = 22
        return top, rh

    def paintEvent(self, e):
        p = QPainter(self)
        self.paint_title(p, self.unit)
        top, rh = self._geom()
        vmax = self.vmax or max([r[3] for r in self.rows] + [1e-9])
        W = self.width()
        x_bar = 44
        x_val = W - 6
        bw = W - x_bar - 70
        for i, (uid, slot, tag, val) in enumerate(self.rows):
            y = top + i * rh
            if uid in self.selected or self.hover == i:
                p.fillRect(QRectF(0, y, W, rh), THEME.color('sel' if uid in self.selected else 'raised'))
                if uid in self.selected:
                    p.fillRect(QRectF(0, y, 2, rh), THEME.color('ink'))
            paint_chip(p, 6, y + rh / 2, slot, tag)
            w = max(0.0, val / vmax * bw)
            p.fillRect(QRectF(x_bar, y + 6, w, rh - 12), THEME.cam(slot))
            p.setFont(mono(12))
            p.setPen(THEME.color('ink'))
            p.drawText(QRectF(x_val - 70, y, 70, rh), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, f'{val:.3f}')
        p.end()

    def mouseMoveEvent(self, e):
        top, rh = self._geom()
        i = int((e.position().y() - top) // rh)
        self.hover = i if 0 <= i < len(self.rows) else None
        if self.hover is not None:
            uid, slot, tag, val = self.rows[self.hover]
            QToolTip.showText(e.globalPosition().toPoint(), f'{tag}: {val:.3f} {self.unit} seen only by this camera', self)
        self.update()

    def mousePressEvent(self, e):
        top, rh = self._geom()
        i = int((e.position().y() - top) // rh)
        if 0 <= i < len(self.rows):
            self.picked.emit(self.rows[i][0])


class DensityHistogram(ChartBase):
    """Area of the ROI per density bin, blue sequential bins, min/target markers."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.edges = None
        self.areas = None
        self.minv = 3.0
        self.target = 5.0
        self.title = 'Best-camera density in ROI'
        self.setMinimumHeight(120)

    def set_data(self, edges, areas, minv, target):
        self.edges, self.areas, self.minv, self.target = edges, areas, minv, target
        self.update()

    def _plot(self):
        return QRectF(40, 26, self.width() - 52, self.height() - 46)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.paint_title(p)
        # legend top-right
        p.setFont(sans(11))
        p.setPen(THEME.color('ink2'))
        x = self.width() - 6
        items = [('area per 1 px/cm bin, m²', None), (f'target {self.target:g}', 'solid'), (f'min {self.minv:g}', 'dash')]
        for txt, style in items:
            fw = p.fontMetrics().horizontalAdvance(txt)
            x -= fw
            p.setPen(THEME.color('ink2'))
            p.drawText(QPointF(x, 13), txt)
            if style:
                pen = QPen(THEME.color('ink'), 1, Qt.PenStyle.DashLine if style == 'dash' else Qt.PenStyle.SolidLine)
                p.setPen(pen)
                p.drawLine(QPointF(x - 22, 9), QPointF(x - 6, 9))
                x -= 26
            x -= 16
        if self.edges is None:
            p.end()
            return
        r = self._plot()
        n = len(self.areas)
        amax = max(float(np.max(self.areas)), 1e-6)
        # baseline + y ticks
        p.setPen(QPen(THEME.color('strong'), 1))
        p.drawLine(QPointF(r.left(), r.bottom()), QPointF(r.right(), r.bottom()))
        p.setFont(mono(10))
        for k in range(1, 4):
            v = amax * k / 3
            y = r.bottom() - r.height() * k / 3
            p.setPen(QPen(THEME.color('hair'), 1))
            p.drawLine(QPointF(r.left(), y), QPointF(r.right(), y))
            p.setPen(THEME.color('muted'))
            p.drawText(QRectF(0, y - 7, r.left() - 6, 14), Qt.AlignmentFlag.AlignRight, f'{v:.1f}')
        bw = r.width() / n
        for i in range(n):
            a = self.areas[i]
            h = a / amax * r.height()
            lo = self.edges[i]
            mid = (self.edges[i] + min(self.edges[i + 1], 20)) / 2
            col = seq_color(max(mid, 0.1)) if lo >= 3 else QColor(SEQ[1])
            if i == self.hover:
                col = col.lighter(118)
            p.fillRect(QRectF(r.left() + i * bw + 1, r.bottom() - h, bw - 2, h), col)
        # x labels
        p.setPen(THEME.color('muted'))
        for i in range(n):
            lo = self.edges[i]
            lab = '<3' if i == 0 else (f'{int(lo)}+' if i == n - 1 else f'{int(lo)}')
            p.drawText(QRectF(r.left() + i * bw, r.bottom() + 3, bw, 14), Qt.AlignmentFlag.AlignCenter, lab)
        # min / target markers
        for v, dash in ((self.minv, True), (self.target, False)):
            k = np.searchsorted(self.edges, v, side='right') - 1
            k = int(np.clip(k, 0, n - 1))
            frac = (v - self.edges[k]) / max(min(self.edges[k + 1], 30) - self.edges[k], 1e-6)
            x = r.left() + (k + frac) * bw
            pen = QPen(THEME.color('ink'), 1, Qt.PenStyle.DashLine if dash else Qt.PenStyle.SolidLine)
            p.setPen(pen)
            p.drawLine(QPointF(x, r.top()), QPointF(x, r.bottom()))
        p.end()

    def mouseMoveEvent(self, e):
        if self.edges is None:
            return
        r = self._plot()
        n = len(self.areas)
        i = int((e.position().x() - r.left()) // (r.width() / n))
        self.hover = i if 0 <= i < n else None
        if self.hover is not None:
            lo, hi = self.edges[i], self.edges[i + 1]
            rng = f'{lo:g}–{hi:g}' if hi < 100 else f'≥ {lo:g}'
            QToolTip.showText(e.globalPosition().toPoint(), f'{rng} px/cm: {self.areas[i]:.3f} m²', self)
        self.update()


class Histogram(ChartBase):
    """Generic histogram with optional log-count y axis and vertical markers."""

    def __init__(self, parent=None, log=True):
        super().__init__(parent)
        self.log = log
        self.counts = None
        self.edges = None
        self.markers = []      # (x, label, dashed)
        self.xlabel = ''
        self.bar = None
        self.setMinimumHeight(140)

    def set_data(self, values, bins=20, markers=(), xlabel=''):
        values = np.asarray(values, float)
        if len(values) == 0:
            self.counts = None
            self.update()
            return
        hi = max(float(values.max()), 1e-6)
        self.edges = np.linspace(0, hi * 1.0001, bins + 1)
        self.counts, _ = np.histogram(values, self.edges)
        self.markers = list(markers)
        self.xlabel = xlabel
        self.update()

    def _plot(self):
        return QRectF(44, 24, self.width() - 60, self.height() - 52)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.paint_title(p, 'log count' if self.log else 'count')
        if self.counts is None:
            p.end()
            return
        r = self._plot()
        c = self.counts.astype(float)
        f = (lambda v: math.log10(v + 1)) if self.log else (lambda v: v)
        cmax = max(f(float(c.max())), 1e-6)
        p.setPen(QPen(THEME.color('strong'), 1))
        p.drawLine(QPointF(r.left(), r.bottom()), QPointF(r.right(), r.bottom()))
        p.setFont(mono(10))
        for k in range(1, 4):
            y = r.bottom() - r.height() * k / 3
            p.setPen(QPen(THEME.color('hair'), 1))
            p.drawLine(QPointF(r.left(), y), QPointF(r.right(), y))
            v = cmax * k / 3
            lab = f'{10 ** v - 1:.0f}' if self.log else f'{v:.0f}'
            p.setPen(THEME.color('muted'))
            p.drawText(QRectF(0, y - 7, r.left() - 6, 14), Qt.AlignmentFlag.AlignRight, lab)
        n = len(c)
        bw = r.width() / n
        bar = QColor(THEME.tok['ink2'])
        for i in range(n):
            h = f(c[i]) / cmax * r.height()
            col = bar.lighter(115) if i == self.hover else bar
            p.fillRect(QRectF(r.left() + i * bw + 1, r.bottom() - h, bw - 2, h), col)
        hi = self.edges[-1]
        p.setPen(THEME.color('muted'))
        for k in range(5):
            v = hi * k / 4
            x = r.left() + r.width() * k / 4
            p.drawText(QRectF(x - 30, r.bottom() + 3, 60, 14), Qt.AlignmentFlag.AlignCenter, f'{v:.3f}')
        if self.xlabel:
            p.drawText(QRectF(r.left(), r.bottom() + 16, r.width(), 14), Qt.AlignmentFlag.AlignRight, self.xlabel)
        for x, lab, dashed in self.markers:
            xx = r.left() + (x / max(hi, 1e-9)) * r.width()
            p.setPen(QPen(THEME.color('ink'), 1, Qt.PenStyle.DashLine if dashed else Qt.PenStyle.SolidLine))
            p.drawLine(QPointF(xx, r.top()), QPointF(xx, r.bottom()))
        p.end()

    def mouseMoveEvent(self, e):
        if self.counts is None:
            return
        r = self._plot()
        n = len(self.counts)
        i = int((e.position().x() - r.left()) // (r.width() / n))
        self.hover = i if 0 <= i < n else None
        if self.hover is not None:
            QToolTip.showText(e.globalPosition().toPoint(),
                              f'{self.edges[i]:.4f}–{self.edges[i + 1]:.4f} m²: {int(self.counts[i])} samples', self)
        self.update()


class Tornado(ChartBase):
    """One-at-a-time sensitivity: Δ for the − and + perturbation per parameter."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.rows = []    # (label, minus, plus)
        self.left_label = '−'
        self.right_label = '+'
        self.setMinimumHeight(160)

    def set_rows(self, rows, left='−5° / −10 mm', right='+5° / +10 mm'):
        self.rows = rows
        self.left_label, self.right_label = left, right
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        self.paint_title(p, 'Δ blind area, m²')
        if not self.rows:
            p.end()
            return
        top = 40
        n = min(len(self.rows), max(1, (self.height() - top - 10) // 20))
        rows = self.rows[:n]
        vmax = max(max(abs(r[1]), abs(r[2])) for r in rows) or 1e-6
        cx = self.width() * 0.55
        half = min(cx - 60, self.width() - cx - 20)
        p.setFont(sans(11))
        p.setPen(THEME.color('ink2'))
        p.fillRect(QRectF(cx - half, 24, 9, 9), QColor(THEME.tok['ink2']))
        p.drawText(QPointF(cx - half + 14, 33), self.left_label)
        fw = p.fontMetrics().horizontalAdvance(self.right_label)
        p.fillRect(QRectF(cx + half - 9, 24, 9, 9), QColor(THEME.tok['muted']))
        p.drawText(QPointF(cx + half - fw - 14, 33), self.right_label)
        for i, (lab, m, pl) in enumerate(rows):
            y = top + i * 20
            if i == self.hover:
                p.fillRect(QRectF(0, y, self.width(), 20), THEME.color('raised'))
            p.setFont(mono(11))
            p.setPen(THEME.color('ink'))
            p.drawText(QRectF(4, y, cx - half - 10, 20), Qt.AlignmentFlag.AlignVCenter, lab)
            for v, col in ((m, QColor(THEME.tok['ink2'])), (pl, QColor(THEME.tok['muted']))):
                w = v / vmax * half
                p.fillRect(QRectF(cx + min(0, w), y + 4, abs(w), 12), col)
        p.setPen(QPen(THEME.color('strong'), 1))
        p.drawLine(QPointF(cx, top - 4), QPointF(cx, top + n * 20))
        p.end()

    def mouseMoveEvent(self, e):
        i = int((e.position().y() - 40) // 20)
        self.hover = i if 0 <= i < len(self.rows) else None
        if self.hover is not None:
            lab, m, pl = self.rows[i]
            QToolTip.showText(e.globalPosition().toPoint(), f'{lab}: {m:+.4f} / {pl:+.4f} m²', self)
        self.update()


class Scatter(ChartBase):
    """Every evaluated layout: x = worst-case blind area, y = min density; Pareto front line."""
    picked = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pts = np.zeros((0, 2))
        self.front = np.zeros((0, 2))
        self.labels = []          # (x, y, label) for candidates
        self.current = None
        self.selected = None
        self.xlabel = 'worst-case blind area under tolerance, m²'
        self.ylabel = 'min px/cm'
        self.setMinimumHeight(200)

    def set_data(self, pts, front, labels, current=None, selected=None):
        self.pts = np.asarray(pts, float).reshape(-1, 2)
        self.front = np.asarray(front, float).reshape(-1, 2)
        self.labels = labels
        self.current = current
        self.selected = selected
        self.update()

    def _plot(self):
        return QRectF(48, 34, self.width() - 64, self.height() - 64)

    def _range(self):
        allp = self.pts if len(self.pts) else np.array([[0, 0], [1, 1]])
        xs, ys = allp[:, 0], allp[:, 1]
        x1 = max(float(np.percentile(xs, 98)) if len(xs) > 5 else float(xs.max()), 1e-3) * 1.08
        y0, y1 = float(ys.min()), float(ys.max())
        if y1 - y0 < 0.2:
            y0, y1 = y0 - 0.2, y1 + 0.2
        return 0.0, x1, y0 - 0.05 * (y1 - y0), y1 + 0.1 * (y1 - y0)

    def _map(self, x, y):
        r = self._plot()
        x0, x1, y0, y1 = self._range()
        return QPointF(r.left() + (min(x, x1) - x0) / (x1 - x0) * r.width(), r.bottom() - (y - y0) / (y1 - y0) * r.height())

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.paint_title(p)
        p.setFont(sans(11))
        p.setPen(THEME.color('ink2'))
        x = self.width() - 6
        for txt, kind in (('current', 'circle'), ('Pareto front', 'line'), ('evaluated', 'dot')):
            fw = p.fontMetrics().horizontalAdvance(txt)
            x -= fw
            p.drawText(QPointF(x, 13), txt)
            if kind == 'circle':
                p.setPen(QPen(THEME.color('ink'), 1))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawEllipse(QPointF(x - 9, 9), 4, 4)
            elif kind == 'line':
                p.setPen(QPen(THEME.color('ink'), 1))
                p.drawLine(QPointF(x - 18, 9), QPointF(x - 4, 9))
            else:
                p.fillRect(QRectF(x - 9, 7, 4, 4), THEME.color('muted'))
            p.setPen(THEME.color('ink2'))
            x -= 26
        r = self._plot()
        p.setFont(sans(10))
        p.setPen(THEME.color('muted'))
        p.drawText(QRectF(r.left() - 40, r.top() - 20, 120, 14), Qt.AlignmentFlag.AlignLeft, self.ylabel)
        p.drawText(QRectF(r.left(), r.bottom() + 16, r.width(), 14), Qt.AlignmentFlag.AlignRight, self.xlabel)
        x0, x1, y0, y1 = self._range()
        p.setFont(mono(10))
        for k in range(5):
            gx = r.left() + r.width() * k / 4
            gy = r.bottom() - r.height() * k / 4
            p.setPen(QPen(THEME.color('hair'), 1))
            p.drawLine(QPointF(gx, r.top()), QPointF(gx, r.bottom()))
            p.drawLine(QPointF(r.left(), gy), QPointF(r.right(), gy))
            p.setPen(THEME.color('muted'))
            p.drawText(QRectF(gx - 30, r.bottom() + 2, 60, 12), Qt.AlignmentFlag.AlignCenter, f'{x0 + (x1 - x0) * k / 4:.3f}')
            p.drawText(QRectF(0, gy - 6, r.left() - 6, 12), Qt.AlignmentFlag.AlignRight, f'{y0 + (y1 - y0) * k / 4:.1f}')
        p.setPen(QPen(THEME.color('strong'), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(r)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(THEME.color('muted'))
        for x, y in self.pts:
            q = self._map(x, y)
            p.drawRect(QRectF(q.x() - 1.5, q.y() - 1.5, 3, 3))
        if len(self.front) > 1:
            fr = self.front[np.argsort(self.front[:, 0])]
            p.setPen(QPen(THEME.color('ink'), 1.2))
            p.drawPolyline(QPolygonF([self._map(x, y) for x, y in fr]))
        for i, (x, y, lab) in enumerate(self.labels):
            q = self._map(x, y)
            sel = self.selected == i
            p.setPen(QPen(THEME.color('ink'), 1))
            p.setBrush(THEME.color('ink') if sel else THEME.color('panel'))
            p.drawRect(QRectF(q.x() - 3.5, q.y() - 3.5, 7, 7))
            p.setFont(mono(10))
            p.drawText(QPointF(q.x() + 6, q.y() - 5), lab)
        if self.current is not None:
            q = self._map(*self.current)
            p.setPen(QPen(THEME.color('ink'), 1.2))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(q, 6, 6)
            p.setFont(mono(10))
            p.drawText(QPointF(q.x() + 8, q.y() + 14), 'current')
        p.end()

    def mousePressEvent(self, e):
        best = None
        for i, (x, y, lab) in enumerate(self.labels):
            q = self._map(x, y)
            d = math.hypot(q.x() - e.position().x(), q.y() - e.position().y())
            if d < 10 and (best is None or d < best[1]):
                best = (i, d)
        if best:
            self.picked.emit(best[0])

    def mouseMoveEvent(self, e):
        for i, (x, y, lab) in enumerate(self.labels):
            q = self._map(x, y)
            if math.hypot(q.x() - e.position().x(), q.y() - e.position().y()) < 9:
                QToolTip.showText(e.globalPosition().toPoint(), f'{lab}: worst {x:.3f} m² · min {y:.1f} px/cm', self)
                return


class ProjectionPlot(ChartBase):
    """Image height r (mm) against field angle θ for the projection models."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.curves = []         # (label, theta_deg array, r array, style) style: 'sel'|'other'
        self.points = []         # (theta_deg, r)
        self.hlines = []         # (r, label)
        self.vline = None        # (theta_deg, label)
        self.xmax = 90
        self.ymax = 5
        self.setMinimumHeight(260)

    def set_data(self, curves, points, hlines, vline, xmax, ymax):
        self.curves, self.points, self.hlines, self.vline = curves, points, hlines, vline
        self.xmax, self.ymax = xmax, ymax
        self.update()

    def _plot(self):
        return QRectF(46, 14, self.width() - 60, self.height() - 70)

    def _m(self, t, r):
        R = self._plot()
        return QPointF(R.left() + t / self.xmax * R.width(), R.bottom() - r / self.ymax * R.height())

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        R = self._plot()
        p.setFont(mono(10))
        for k in range(7):
            x = R.left() + R.width() * k / 6
            p.setPen(QPen(THEME.color('hair'), 1))
            p.drawLine(QPointF(x, R.top()), QPointF(x, R.bottom()))
            p.setPen(THEME.color('muted'))
            p.drawText(QRectF(x - 20, R.bottom() + 3, 40, 12), Qt.AlignmentFlag.AlignCenter, f'{self.xmax * k / 6:.0f}')
        for k in range(6):
            y = R.bottom() - R.height() * k / 5
            p.setPen(QPen(THEME.color('hair'), 1))
            p.drawLine(QPointF(R.left(), y), QPointF(R.right(), y))
            p.setPen(THEME.color('muted'))
            p.drawText(QRectF(0, y - 6, R.left() - 6, 12), Qt.AlignmentFlag.AlignRight, f'{self.ymax * k / 5:.1f}')
        p.setFont(sans(10))
        p.drawText(QRectF(R.left() - 40, 0, 60, 12), Qt.AlignmentFlag.AlignLeft, 'r, mm')
        p.drawText(QRectF(R.left(), R.bottom() + 16, R.width(), 12), Qt.AlignmentFlag.AlignRight, 'field angle θ, °')
        p.setClipRect(R.adjusted(-1, -1, 1, 1))
        for r, lab in self.hlines:
            pen = QPen(THEME.color('ink2'), 1, Qt.PenStyle.CustomDashLine)
            pen.setDashPattern([2, 3])
            p.setPen(pen)
            y = self._m(0, r).y()
            p.drawLine(QPointF(R.left(), y), QPointF(R.right(), y))
        for lab, th, r, style in self.curves:
            if style == 'sel':
                pen = QPen(THEME.color('ink'), 1.6)
            else:
                pen = QPen(THEME.color('muted'), 1, Qt.PenStyle.CustomDashLine)
                pen.setDashPattern([2, 3])
            p.setPen(pen)
            p.drawPolyline(QPolygonF([self._m(t, rr) for t, rr in zip(th, r) if np.isfinite(rr)]))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(THEME.color('ink'))
        for t, r in self.points:
            q = self._m(t, r)
            p.drawRect(QRectF(q.x() - 2.5, q.y() - 2.5, 5, 5))
        if self.vline:
            x = self._m(self.vline[0], 0).x()
            p.setPen(QPen(THEME.color('ink'), 1))
            p.drawLine(QPointF(x, R.top()), QPointF(x, R.bottom()))
        p.setClipping(False)
        p.end()
