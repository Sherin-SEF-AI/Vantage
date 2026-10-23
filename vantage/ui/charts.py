"""Plain QPainter charts: hairline axes, thin marks, hover tooltips.

Shared conventions: title sans 12 demibold, its unit or key at the right in sans 11 muted; tick labels mono 10 muted
at round values (nice_ticks); gridlines 1 px solid 'hair', baselines 'strong'; marks with a 2 px gap and a rounded
data end; a hovered mark lifts (lighter, ink outline) and a painted tooltip gives its exact value first and what it is
after (tip_text() returns the same text)."""
from __future__ import annotations

import math

import numpy as np
from PyQt6.QtCore import Qt, QObject, QPointF, QRectF, pyqtSignal
from PyQt6.QtGui import QPainter, QColor, QPen, QFont, QPolygonF, QBrush, QFontMetricsF, QPainterPath
from PyQt6.QtWidgets import QWidget

from . import colormaps
from .theme import THEME, mono, sans
from .widgets import paint_chip


class _CameraHover(QObject):
    """The camera under the pointer in any chart or plan view ('' = none): a hovered bar lights up its camera's ground
    in the plan views, a hovered camera in the plan lights up its bar."""
    changed = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.uid = ''

    def set(self, uid):
        uid = uid or ''
        if uid != self.uid:
            self.uid = uid
            self.changed.emit(uid)


CAMERA_HOVER = _CameraHover()


def nice_ticks(vmax, n=4):
    """Round tick values 0 … top (top >= vmax), about n steps (1, 2, 2.5 or 5 × 10^k apart)."""
    if not (vmax > 0) or not math.isfinite(vmax):
        return [0.0, 1.0]
    n = max(n, 1)
    mag = 10 ** math.floor(math.log10(vmax / n))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10, 20) if vmax / (m * mag) <= n + 1 + 1e-9)
    k = math.ceil(vmax / step - 1e-9)
    return [round(i * step, 10) for i in range(k + 1)]


def tick_label(v):
    return f'{round(v, 6):g}'


class ChartBase(QWidget):
    """Shared chart chrome: title row, hover state and a painted tooltip (value first, then what it is)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.hover = None
        self.title = ''
        self.subtitle = ''
        self.tip = None                 # (anchor QPointF, value text, [label lines]) while hovering a mark
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

    def set_tip(self, anchor, value, *lines):
        self.tip = (QPointF(anchor), value, [t for t in lines if t])

    def tip_text(self):
        """The hover tooltip as plain text ('' when nothing is hovered)."""
        if self.tip is None:
            return ''
        return ' · '.join([self.tip[1]] + self.tip[2])

    def paint_tip(self, p):
        if self.tip is None:
            return
        anchor, value, lines = self.tip
        f1, f2 = mono(12, QFont.Weight.DemiBold), sans(10)
        fm1, fm2 = QFontMetricsF(f1), QFontMetricsF(f2)
        w = max([fm1.horizontalAdvance(value)] + [fm2.horizontalAdvance(t) for t in lines]) + 18
        h = 8 + 17 + 14 * len(lines) + 5
        x, y = anchor.x() + 12, anchor.y() - h - 6
        if x + w > self.width() - 2:
            x = anchor.x() - w - 12
        if y < 2:
            y = anchor.y() + 14
        x = min(max(2.0, x), max(2.0, self.width() - w - 2))
        y = min(max(2.0, y), max(2.0, self.height() - h - 2))
        p.save()
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.setPen(QPen(THEME.color('strong'), 1))
        p.setBrush(THEME.color('raised'))
        p.drawRoundedRect(QRectF(x + 0.5, y + 0.5, w, h), 4, 4)
        p.setFont(f1)
        p.setPen(THEME.color('ink'))
        p.drawText(QRectF(x + 9, y + 6, w - 14, 17), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, value)
        p.setFont(f2)
        p.setPen(THEME.color('ink2'))
        for i, t in enumerate(lines):
            p.drawText(QRectF(x + 9, y + 23 + 14 * i, w - 14, 14), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, t)
        p.restore()

    def paint_grid_y(self, p, r, ticks, top, fmt=tick_label):
        """Hairline gridlines and right-aligned tick labels left of the plot rectangle r (0 … top)."""
        p.setFont(mono(10))
        for v in ticks:
            y = r.bottom() - r.height() * v / top
            if v > 0:
                p.setPen(QPen(THEME.color('hair'), 1))
                p.drawLine(QPointF(r.left(), y), QPointF(r.right(), y))
            p.setPen(THEME.color('muted'))
            p.drawText(QRectF(0, y - 7, r.left() - 6, 14), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, fmt(v))

    @staticmethod
    def bar_path(r, radius=3.0, end='top'):
        """A bar with a rounded data end and a square base (end: 'top' for columns, 'right' for bars)."""
        path = QPainterPath()
        rad = min(radius, r.width() / 2, r.height() / 2)
        if rad < 0.8:
            path.addRect(r)
            return path
        path.addRoundedRect(r, rad, rad)
        sq = QPainterPath()
        if end == 'top':
            sq.addRect(QRectF(r.left(), r.top() + rad, r.width(), r.height() - rad))
        else:
            sq.addRect(QRectF(r.left(), r.top(), r.width() - rad, r.height()))
        return path.united(sq)

    def leaveEvent(self, e):
        self.hover = None
        self.tip = None
        self.update()


class BarList(ChartBase):
    """Rows: chip + tag | bar in identity colour | value. Hovering a row lights up its camera in the plan views (and a
    camera hovered there lights up its row); click selects the camera."""
    picked = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.rows = []      # (uid, slot, tag, value)
        self.selected = set()
        self.unit = 'm²'
        self.what = 'seen only by this camera'
        self.vmax = None
        self.ext_hover = ''             # camera hovered elsewhere (plan view)
        self._own_hover = False
        self.setMinimumHeight(120)
        CAMERA_HOVER.changed.connect(self._on_cam_hover)

    def _on_cam_hover(self, uid):
        if uid != self.ext_hover:
            self.ext_hover = uid
            self.update()

    def set_rows(self, rows, selected=()):
        self.rows = rows
        self.selected = set(selected)
        if self.hover is not None and self.hover >= len(rows):
            self.hover, self.tip = None, None
        self.update()

    def _geom(self):
        top = 22
        rh = 22
        return top, rh

    def _bar_rect(self, i, val, vmax):
        top, rh = self._geom()
        x_bar, bw = 44, max(self.width() - 44 - 70, 10)
        return QRectF(x_bar + 1, top + i * rh + 6, max(0.0, val / vmax * bw), rh - 12)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.paint_title(p, self.unit)
        top, rh = self._geom()
        vmax = self.vmax or max([r[3] for r in self.rows] + [1e-9])
        W = self.width()
        x_val = W - 6
        for i, (uid, slot, tag, val) in enumerate(self.rows):
            y = top + i * rh
            hot = self.hover == i or (uid == self.ext_hover and self.hover is None)
            if uid in self.selected or hot:
                p.fillRect(QRectF(0, y, W, rh), THEME.color('sel' if uid in self.selected else 'raised'))
                if uid in self.selected:
                    p.fillRect(QRectF(0, y, 2, rh), THEME.color('ink'))
            paint_chip(p, 6, y + rh / 2, slot, tag)
            r = self._bar_rect(i, val, vmax)
            if r.width() > 0:
                col = THEME.cam(slot)
                p.setPen(QPen(THEME.color('ink'), 1) if hot else Qt.PenStyle.NoPen)
                p.setBrush(col.lighter(112) if hot else col)
                p.drawPath(self.bar_path(r, 3, 'right'))
            p.setFont(mono(12))
            p.setPen(THEME.color('ink'))
            p.drawText(QRectF(x_val - 70, y, 70, rh), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, f'{val:.3f}')
        if self.rows:
            p.setPen(QPen(THEME.color('strong'), 1))
            p.drawLine(QPointF(44.5, top + 3), QPointF(44.5, top + len(self.rows) * rh - 3))
        self.paint_tip(p)
        p.end()

    def mouseMoveEvent(self, e):
        top, rh = self._geom()
        i = int((e.position().y() - top) // rh)
        self.hover = i if 0 <= i < len(self.rows) and e.position().y() >= top else None
        if self.hover is not None:
            uid, slot, tag, val = self.rows[self.hover]
            vmax = self.vmax or max([r[3] for r in self.rows] + [1e-9])
            r = self._bar_rect(self.hover, val, vmax)
            self.set_tip(QPointF(min(r.right(), self.width() - 80), r.center().y()), f'{val:.3f} {self.unit}',
                         f'{tag} · {self.what}', 'click selects it')
            self._own_hover = True
            CAMERA_HOVER.set(uid)
        else:
            self.tip = None
            if self._own_hover:
                self._own_hover = False
                CAMERA_HOVER.set('')
        self.update()

    def leaveEvent(self, e):
        super().leaveEvent(e)
        if self._own_hover:
            self._own_hover = False
            CAMERA_HOVER.set('')

    def mousePressEvent(self, e):
        top, rh = self._geom()
        i = int((e.position().y() - top) // rh)
        if 0 <= i < len(self.rows):
            self.picked.emit(self.rows[i][0])


class DensityHistogram(ChartBase):
    """ROI area per best-camera density bin (1 px/cm bins), bars on the plan view's density scale, ground below the
    minimum grey and hatched, labelled minimum (dashed) and target (solid) lines."""

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
        if self.hover is not None and (areas is None or self.hover >= len(areas)):
            self.hover, self.tip = None, None
        self.update()

    def _plot(self):
        return QRectF(42, 36, self.width() - 54, self.height() - 56)

    def _bin_hi(self, i):
        """Upper edge of bin i for drawing (the open last bin is drawn 1 px/cm wide)."""
        lo, hi = self.edges[i], self.edges[i + 1]
        return hi if hi < 1e6 else lo + 1

    def x_of(self, v):
        """Plot x of a density value (bins are equally wide on screen)."""
        r = self._plot()
        n = len(self.areas)
        k = int(np.clip(np.searchsorted(self.edges, v, side='right') - 1, 0, n - 1))
        frac = float(np.clip((v - self.edges[k]) / max(self._bin_hi(k) - self.edges[k], 1e-6), 0, 1))
        return r.left() + (k + frac) * r.width() / n

    def _below(self, i):
        return (self.edges[i] + self._bin_hi(i)) / 2 < self.minv

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.paint_title(p, 'ROI area per 1 px/cm bin · m²')
        if self.edges is None or self.areas is None or not len(self.areas):
            p.end()
            return
        r = self._plot()
        n = len(self.areas)
        amax = max(float(np.max(self.areas)), 1e-6)
        ticks = nice_ticks(amax, 3 if r.height() < 110 else 4)
        top = ticks[-1]
        self.paint_grid_y(p, r, ticks, top)
        sc = colormaps.density_scale(self.minv, self.target)
        bw = r.width() / n
        for i in range(n):
            a = float(self.areas[i])
            h = a / top * r.height()
            if h <= 0:
                continue
            rr = QRectF(r.left() + i * bw + 1, r.bottom() - h, bw - 2, h)
            below = self._below(i)
            col = QColor(sc.below_hex) if below else sc.qcolor((self.edges[i] + self._bin_hi(i)) / 2)
            hot = i == self.hover
            path = self.bar_path(rr, 2, 'top')
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(col.lighter(112) if hot else col)
            p.drawPath(path)
            if below:
                p.setBrush(QBrush(QColor(sc.hatch_hex), Qt.BrushStyle.FDiagPattern))
                p.drawPath(path)
            if hot:
                p.setPen(QPen(THEME.color('ink'), 1))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawPath(path)
        p.setPen(QPen(THEME.color('strong'), 1))
        p.drawLine(QPointF(r.left(), r.bottom()), QPointF(r.right(), r.bottom()))
        # x ticks at the bin edges (bins are [lo, hi)); the open last bin is ≥ its lower edge
        p.setFont(mono(10))
        p.setPen(THEME.color('muted'))
        every = 1 if bw >= 20 else 2
        for i in range(n):
            if i % every:
                continue
            x = r.left() + i * bw
            p.drawText(QRectF(x - 16, r.bottom() + 3, 32, 13), Qt.AlignmentFlag.AlignCenter, tick_label(self.edges[i]))
        p.setFont(sans(10))
        p.drawText(QRectF(r.right() - 60, r.bottom() + 3, 60, 13), Qt.AlignmentFlag.AlignRight, 'px/cm')
        # minimum (dashed) and target (solid), labelled at the top: the minimum to the left of its line, the target
        # to the right of its line
        p.setFont(sans(10))
        fm = p.fontMetrics()
        for v, name, dash in ((self.minv, 'min', True), (self.target, 'target', False)):
            if not (v > 0):
                continue
            x = self.x_of(v)
            p.setPen(QPen(THEME.color('panel', 0.9), 3))
            p.drawLine(QPointF(x, r.top() - 4), QPointF(x, r.bottom()))
            pen = QPen(THEME.color('ink'), 1.2)
            if dash:
                pen.setStyle(Qt.PenStyle.CustomDashLine)
                pen.setDashPattern([3, 2])
            p.setPen(pen)
            p.drawLine(QPointF(x, r.top() - 4), QPointF(x, r.bottom()))
            txt = f'{name} {v:g}'
            tw = fm.horizontalAdvance(txt)
            left = name == 'min' and self.target > self.minv
            tx = x - tw - 4 if left else x + 4
            tx = min(max(0.0, tx), self.width() - tw)
            p.setPen(THEME.color('ink'))
            p.drawText(QRectF(tx, r.top() - 17, tw + 2, 13), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, txt)
        self.paint_tip(p)
        p.end()

    def bin_text(self, i):
        """(value, label lines) the tooltip shows for bin i."""
        lo, hi = self.edges[i], self.edges[i + 1]
        rng = f'{lo:g}–{hi:g} px/cm' if hi < 1e6 else f'≥ {lo:g} px/cm'
        if i == 0 and lo == 0:
            rng = f'below {hi:g} px/cm'
        tot = float(np.sum(self.areas))
        share = f'{self.areas[i] / tot * 100:.1f} % of the seen ROI' if tot > 0 else ''
        note = 'below the minimum' if self._below(i) else ('meets the target' if lo >= self.target else '')
        return f'{self.areas[i]:.3f} m²', [rng, share, note]

    def mouseMoveEvent(self, e):
        if self.edges is None or self.areas is None or not len(self.areas):
            return
        r = self._plot()
        n = len(self.areas)
        i = int((e.position().x() - r.left()) // (r.width() / n))
        self.hover = i if 0 <= i < n and e.position().x() >= r.left() else None
        if self.hover is not None:
            value, lines = self.bin_text(i)
            top = nice_ticks(max(float(np.max(self.areas)), 1e-6), 3 if r.height() < 110 else 4)[-1]
            y = r.bottom() - float(self.areas[i]) / top * r.height()
            self.set_tip(QPointF(r.left() + (i + 0.5) * r.width() / n, y), value, *lines)
        else:
            self.tip = None
        self.update()


class Histogram(ChartBase):
    """Generic histogram with optional log-count y axis and labelled vertical markers."""

    def __init__(self, parent=None, log=True):
        super().__init__(parent)
        self.log = log
        self.counts = None
        self.edges = None
        self.markers = []      # (x, label, dashed)
        self.xlabel = ''
        self.unit = 'm²'
        self.bar = None
        self.setMinimumHeight(140)

    def set_data(self, values, bins=20, markers=(), xlabel=''):
        values = np.asarray(values, float)
        values = values[np.isfinite(values)]
        if len(values) == 0:
            self.counts = None
            self.tip = None
            self.update()
            return
        hi = max(float(values.max()), 1e-6)
        self.edges = np.linspace(0, hi * 1.0001, bins + 1)
        self.counts, _ = np.histogram(values, self.edges)
        self.markers = list(markers)
        self.xlabel = xlabel
        self.update()

    def _plot(self):
        return QRectF(44, 38, self.width() - 60, self.height() - 66)

    def _yticks(self):
        cmax = float(self.counts.max()) if self.counts is not None and len(self.counts) else 1.0
        if self.log:
            top = 10 ** math.ceil(math.log10(cmax + 1))
            ticks = [0] + [10 ** k for k in range(0, int(round(math.log10(top))) + 1)]
            return ticks, top
        t = nice_ticks(cmax, 3)
        return t, t[-1]

    def _fy(self, v, top):
        return math.log10(v + 1) / math.log10(top + 1) if self.log else v / top

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.paint_title(p, 'count · log scale' if self.log else 'count')
        if self.counts is None:
            p.end()
            return
        r = self._plot()
        ticks, top = self._yticks()
        p.setFont(mono(10))
        for v in ticks:
            y = r.bottom() - r.height() * self._fy(v, top)
            if v > 0:
                p.setPen(QPen(THEME.color('hair'), 1))
                p.drawLine(QPointF(r.left(), y), QPointF(r.right(), y))
            p.setPen(THEME.color('muted'))
            p.drawText(QRectF(0, y - 7, r.left() - 6, 14), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, f'{v:g}')
        c = self.counts.astype(float)
        n = len(c)
        bw = r.width() / n
        bar = QColor(THEME.tok['ink2'])
        for i in range(n):
            h = self._fy(c[i], top) * r.height()
            if h <= 0:
                continue
            path = self.bar_path(QRectF(r.left() + i * bw + 1, r.bottom() - h, bw - 2, h), 2, 'top')
            p.setPen(QPen(THEME.color('ink'), 1) if i == self.hover else Qt.PenStyle.NoPen)
            p.setBrush(bar.lighter(125) if i == self.hover else bar)
            p.drawPath(path)
        p.setPen(QPen(THEME.color('strong'), 1))
        p.drawLine(QPointF(r.left(), r.bottom()), QPointF(r.right(), r.bottom()))
        hi = self.edges[-1]
        xt = [v for v in nice_ticks(hi, 4) if v <= hi * 1.0001]
        p.setFont(mono(10))
        p.setPen(THEME.color('muted'))
        for v in xt:
            x = r.left() + r.width() * v / hi
            p.drawText(QRectF(x - 30, r.bottom() + 3, 60, 13), Qt.AlignmentFlag.AlignCenter, tick_label(v))
        if self.xlabel:
            p.setFont(sans(10))
            p.drawText(QRectF(r.left(), r.bottom() + 16, r.width(), 13), Qt.AlignmentFlag.AlignRight, self.xlabel)
        # markers: a line each, labelled with name and value above the plot (staggered when close)
        p.setFont(sans(10))
        fm = p.fontMetrics()
        placed = []
        for x, lab, dashed in sorted(self.markers, key=lambda m: m[0]):
            if not np.isfinite(x):
                continue
            xx = r.left() + min(max(x / max(hi, 1e-9), 0), 1) * r.width()
            pen = QPen(THEME.color('ink'), 1.2)
            if dashed:
                pen.setStyle(Qt.PenStyle.CustomDashLine)
                pen.setDashPattern([3, 2])
            p.setPen(pen)
            p.drawLine(QPointF(xx, r.top() - 2), QPointF(xx, r.bottom()))
            txt = f'{lab} {x:.3g}'
            tw = fm.horizontalAdvance(txt)
            tx = min(max(0.0, xx - tw / 2), self.width() - tw)
            row = 0
            while any(rw == row and not (tx > b + 4 or tx + tw < a - 4) for a, b, rw in placed):
                row += 1
            placed.append((tx, tx + tw, row))
            p.setPen(THEME.color('ink'))
            p.drawText(QRectF(tx, r.top() - 15 - 12 * row, tw + 2, 12), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, txt)
        self.paint_tip(p)
        p.end()

    def mouseMoveEvent(self, e):
        if self.counts is None:
            return
        r = self._plot()
        n = len(self.counts)
        i = int((e.position().x() - r.left()) // (r.width() / n))
        self.hover = i if 0 <= i < n and e.position().x() >= r.left() else None
        if self.hover is not None:
            ticks, top = self._yticks()
            y = r.bottom() - self._fy(float(self.counts[i]), top) * r.height()
            k = int(self.counts[i])
            self.set_tip(QPointF(r.left() + (i + 0.5) * r.width() / n, y), f'{k} sample' + ('' if k == 1 else 's'),
                         f'{self.edges[i]:.4f}–{self.edges[i + 1]:.4f} {self.unit}',
                         f'{k / max(int(self.counts.sum()), 1) * 100:.1f} % of all samples')
        else:
            self.tip = None
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
        self.paint_tip(p)
        p.end()

    def mouseMoveEvent(self, e):
        i = int((e.position().y() - 40) // 20)
        self.hover = i if 0 <= i < len(self.rows) and e.position().y() >= 40 else None
        if self.hover is not None:
            lab, m, pl = self.rows[i]
            self.set_tip(QPointF(self.width() * 0.55, 40 + i * 20 + 10), f'{m:+.4f} / {pl:+.4f} m²', lab,
                         f'{self.left_label}  /  {self.right_label}')
        else:
            self.tip = None
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
        self.paint_tip(p)
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
        tip = None
        for i, (x, y, lab) in enumerate(self.labels):
            q = self._map(x, y)
            if math.hypot(q.x() - e.position().x(), q.y() - e.position().y()) < 12:
                tip = (q, f'{x:.3f} m² · {y:.1f} px/cm', lab, 'worst-case blind area · minimum density', 'click to select')
                break
        if tip is None and self.current is not None:
            q = self._map(*self.current)
            if math.hypot(q.x() - e.position().x(), q.y() - e.position().y()) < 12:
                tip = (q, f'{self.current[0]:.3f} m² · {self.current[1]:.1f} px/cm', 'current layout')
        old = self.tip_text()
        self.tip = None
        if tip is not None:
            self.set_tip(*tip)
        if self.tip_text() != old:
            self.update()


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
        p.setFont(mono(10))
        for r, lab in self.hlines:
            pen = QPen(THEME.color('ink2'), 1, Qt.PenStyle.CustomDashLine)
            pen.setDashPattern([2, 3])
            p.setPen(pen)
            y = self._m(0, r).y()
            p.drawLine(QPointF(R.left(), y), QPointF(R.right(), y))
            if lab:
                p.setPen(THEME.color('ink2'))
                p.drawText(QRectF(R.right() - 64, y - 14, 60, 12), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, lab)
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
            if len(self.vline) > 1 and self.vline[1]:
                p.setFont(mono(10))
                p.drawText(QRectF(x + 4, R.top() + 2, 80, 12), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, self.vline[1])
        p.setClipping(False)
        p.end()
