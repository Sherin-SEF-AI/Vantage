"""Analysis ▸ Experiments: many designs of the rig evaluated automatically — the factors to vary (poses, lenses,
sensor models, mount offsets, the ROI distance, a load case, the weather), the responses to compute (any metric of the
requirements scorecard), a design (full factorial, one at a time, Latin hypercube, Sobol, Morris, Monte Carlo) — and
what comes out: Sobol indices with intervals, Morris μ*/σ, main effects, two-factor surfaces, the Pareto front, the
best design (Add as variant), and the uncertainty of every figure under random inputs. The run is a background job
(cancel resumes from the points done); nothing runs by itself after an edit (the result is marked stale)."""
from __future__ import annotations

import json
import math
import threading
import traceback

import numpy as np
from PyQt6.QtCore import QObject, QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPen, QPolygonF
from PyQt6.QtWidgets import (QAbstractItemView, QComboBox, QFileDialog, QGridLayout, QHBoxLayout, QHeaderView,
                             QLineEdit, QListWidget, QListWidgetItem, QMenu, QMessageBox, QProgressBar, QScrollArea,
                             QSplitter, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget)

from ...core import experiments as EX
from ..charts import ChartBase
from ..theme import SEQ, THEME, mono, sans
from ..widgets import ROW_H, ScrubSpinBox, SegmentedControl, StatusLine, ToggleSwitch, button, hline, label
from .recording_ws import Tile

JOB = 'experiments'
JOB_EST = 'experiments_estimate'
LONG_S = 120.0                      # a run estimated longer than this asks first
MAX_ROWS = 500                      # the points table shows the best this many (the export has every point)
EMPTY = 'Add factors and responses, pick a design, then Run.'
DESIGN_ALL = tuple(EX.DESIGNS)
NUM_FIELDS = (('lo', 'Low', 'The low end of the range'), ('hi', 'High', 'The high end of the range'),
              ('levels', 'Levels', 'Levels across the range for the full factorial and one at a time (0: the '
                                   'design\'s)'),
              ('mean', 'Mean', 'The normal distribution\'s mean (Latin hypercube, Sobol, Monte Carlo)'),
              ('sd', 'SD', 'The normal distribution\'s standard deviation'))
DESIGN_FIELDS = (('samples', 'Samples', 0, 'Points of the Latin hypercube or the Monte Carlo', ('lhs', 'montecarlo')),
                 ('sobol_n', 'Base samples N', 0, 'Sobol: N(k + 2) points for k factors; a power of two keeps the '
                                                  'sequence balanced', ('sobol',)),
                 ('morris_r', 'Trajectories r', 0, 'Morris: r(k + 1) points', ('morris',)),
                 ('levels', 'Levels', 0, 'Numeric levels of the full factorial and one at a time (a factor may set its '
                                         'own)', ('factorial', 'oat')),
                 ('morris_levels', 'Grid levels p', 0, 'Morris: the grid each factor moves on', ('morris',)),
                 ('seed', 'Seed', 0, 'The random seed: the same seed gives the same design', ('lhs', 'sobol', 'morris',
                                                                                             'montecarlo')),
                 ('bootstrap', 'Bootstrap', 0, 'Resamples for the intervals (0: none)', DESIGN_ALL),
                 ('confidence', 'Confidence', 3, 'The intervals\' confidence level', DESIGN_ALL),
                 ('workers', 'Workers', 0, 'Points evaluated in parallel (0: as many as help, up to 4)', DESIGN_ALL))
TABS = ('Sensitivity', 'Effects', 'Surface', 'Pareto', 'Uncertainty', 'Points', 'Notes')


def _combo(min_chars=14) -> QComboBox:
    cb = QComboBox()
    cb.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
    cb.setMinimumContentsLength(min_chars)
    return cb


def _table(cols, stretch=0, select=True):
    t = QTableWidget(0, len(cols))
    for c, (h, tip) in enumerate(cols):
        it = QTableWidgetItem(h)
        if tip:
            it.setToolTip(tip)
        t.setHorizontalHeaderItem(c, it)
    t.verticalHeader().setVisible(False)
    t.verticalHeader().setDefaultSectionSize(ROW_H)
    t.horizontalHeader().setFixedHeight(ROW_H)
    t.setShowGrid(False)
    t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    if select:
        t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        t.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    else:
        t.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
    hh = t.horizontalHeader()
    for c in range(len(cols)):
        hh.setSectionResizeMode(c, QHeaderView.ResizeMode.Stretch if c == stretch else
                                QHeaderView.ResizeMode.ResizeToContents)
    return t


def _num(text, tip=None, color=None):
    it = QTableWidgetItem(text)
    it.setFont(mono(12))
    it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    if tip:
        it.setToolTip(tip)
    if color is not None:
        it.setForeground(color)
    return it


def _g(v, nd=4):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return '—'
    return f'{v:.{nd}g}'


def _ci(c, nd=3):
    if not c or c[0] is None:
        return '—'
    return f'{c[0]:.{nd}f} – {c[1]:.{nd}f}'


def _accent() -> QColor:
    return THEME.cam(0)


def _ticks(lo, hi, n=4):
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
        return [lo]
    raw = (hi - lo) / max(n, 1)
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if raw <= m * mag + 1e-12)
    t = math.ceil(lo / step - 1e-9) * step
    out = []
    while t <= hi + 1e-9 * step:
        out.append(round(t, 10))
        t += step
    return out


def _lim(vals, pad=0.06, zero=False):
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], float)
    if not len(v):
        return 0.0, 1.0
    lo, hi = float(v.min()), float(v.max())
    if zero:
        lo, hi = min(lo, 0.0), max(hi, 0.0)
    if hi - lo < 1e-12:
        lo, hi = lo - (abs(lo) * 0.1 or 0.5), hi + (abs(hi) * 0.1 or 0.5)
    d = (hi - lo) * pad
    return lo - d, hi + d


class _Plot(ChartBase):
    """Chart chrome shared by the experiment's charts: a plot rectangle with value axes and hover."""
    MARGIN = (54, 40, 16, 40)           # left, top, right, bottom

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(220)
        self.message = 'Run the experiment'

    def rect(self):
        l_, t_, r_, b_ = self.MARGIN
        return QRectF(l_, t_, max(self.width() - l_ - r_, 10), max(self.height() - t_ - b_, 10))

    def empty(self, p):
        p.setFont(sans(12))
        p.setPen(THEME.color('muted'))
        p.drawText(QRectF(0, 0, self.width(), self.height()), Qt.AlignmentFlag.AlignCenter, self.message)

    def axes(self, p, r, xlim, ylim, xlabel='', ylabel='', xticks=True, yticks=True):
        X = lambda v: r.left() + (v - xlim[0]) / ((xlim[1] - xlim[0]) or 1.0) * r.width()        # noqa: E731
        Y = lambda v: r.bottom() - (v - ylim[0]) / ((ylim[1] - ylim[0]) or 1.0) * r.height()     # noqa: E731
        p.setFont(mono(10))
        if yticks:
            for t in _ticks(*ylim):
                y = Y(t)
                p.setPen(QPen(THEME.color('hair'), 1))
                p.drawLine(QPointF(r.left(), y), QPointF(r.right(), y))
                p.setPen(THEME.color('muted'))
                p.drawText(QRectF(0, y - 7, r.left() - 6, 14), Qt.AlignmentFlag.AlignRight |
                           Qt.AlignmentFlag.AlignVCenter, f'{t:.4g}')
        if xticks:
            for t in _ticks(*xlim):
                x = X(t)
                if not yticks:
                    p.setPen(QPen(THEME.color('hair'), 1))
                    p.drawLine(QPointF(x, r.top()), QPointF(x, r.bottom()))
                p.setPen(THEME.color('muted'))
                p.drawText(QRectF(x - 32, r.bottom() + 3, 64, 13), Qt.AlignmentFlag.AlignCenter, f'{t:.4g}')
        p.setPen(QPen(THEME.color('strong'), 1))
        p.drawLine(QPointF(r.left(), r.bottom()), QPointF(r.right(), r.bottom()))
        p.setFont(sans(10))
        p.setPen(THEME.color('muted'))
        if xlabel:
            p.drawText(QRectF(r.left(), r.bottom() + 17, r.width(), 13), Qt.AlignmentFlag.AlignRight, xlabel)
        if ylabel:
            p.drawText(QRectF(4, r.top() - 17, self.width() - 8, 14), Qt.AlignmentFlag.AlignLeft, ylabel)
        return X, Y


# ============================================================================ charts
class IndexChart(_Plot):
    """What drives a response: Sobol indices (first order filled, total outlined, with their intervals), Morris μ*
    against σ, or the swing of each one-at-a-time sweep."""
    MARGIN = (130, 34, 18, 40)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.mode, self.rows, self.unit = '', [], ''
        self._hits = []

    def set_data(self, mode, rows, unit=''):
        """rows: sobol [(label, S1, S1ci, ST, STci)], morris [(label, mu*, sigma, mu*ci)], oat [(label, swing)]."""
        self.mode, self.rows, self.unit = mode, rows, unit
        self.tip = None
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.title = {'sobol': 'Sobol indices', 'morris': 'Morris screening', 'oat': 'One-at-a-time swing',
                      'range': 'Main-effect range'}.get(self.mode, 'Sensitivity')
        self.paint_title(p, {'sobol': 'first order ■ · total □ · interval', 'morris': f'μ* · σ, {self.unit}',
                             'oat': self.unit, 'range': self.unit}.get(self.mode, ''))
        if not self.rows:
            self.empty(p)
            p.end()
            return
        self._hits = []
        acc = _accent()
        if self.mode == 'morris':
            r = QRectF(60, 42, max(self.width() - 80, 10), max(self.height() - 84, 10))
            xs = [row[1] for row in self.rows if row[1] is not None]
            ys = [row[2] for row in self.rows if row[2] is not None]
            X, Y = self.axes(p, r, _lim(xs + [0.0], zero=True), _lim(ys + [0.0], zero=True),
                             f'μ*, {self.unit} per range', f'σ, {self.unit}')
            p.setFont(sans(11))
            fm = p.fontMetrics()
            placed = []
            for i, (lab, mu, sg, ci) in enumerate(self.rows):
                if mu is None:
                    continue
                q = QPointF(X(mu), Y(sg))
                p.setPen(QPen(THEME.color('ink'), 1) if i == self.hover else Qt.PenStyle.NoPen)
                p.setBrush(acc)
                p.drawEllipse(q, 4.5, 4.5)
                box = QRectF(q.x() + 7, q.y() - 16, fm.horizontalAdvance(lab) + 2, 14)
                while any(box.intersects(b) for b in placed):          # labels of close points stacked
                    box.translate(0, 14)
                placed.append(box)
                p.setPen(THEME.color('ink2'))
                p.drawText(box, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, lab)
                self._hits.append((QRectF(q.x() - 8, q.y() - 8, 16, 16), i))
        else:
            r = self.rect()
            n = len(self.rows)
            if self.mode == 'sobol':
                vals = [v for row in self.rows for v in (row[1], row[3], *(row[2] or (None,)), *(row[4] or (None,)))
                        if v is not None]
                xlim = (min(0.0, min(vals, default=0.0)), max(1.0, max(vals, default=1.0)))
                xlabel = 'share of the variance'
            else:
                vals = [row[1] for row in self.rows if row[1] is not None]
                xlim = (0.0, max(vals, default=1.0) * 1.08 or 1.0)
                word = 'swing' if self.mode == 'oat' else 'range of the means'
                xlabel = f'{word}, {self.unit}' if self.unit else word
            X, _ = self.axes(p, r, xlim, (0, 1), xlabel, yticks=False)
            rowh = r.height() / max(n, 1)
            for i, row in enumerate(self.rows):
                y0 = r.top() + i * rowh
                if i == self.hover:
                    p.fillRect(QRectF(0, y0, self.width(), rowh), THEME.color('raised'))
                p.setFont(sans(11))
                p.setPen(THEME.color('ink'))
                p.drawText(QRectF(4, y0, r.left() - 10, rowh), Qt.AlignmentFlag.AlignVCenter |
                           Qt.AlignmentFlag.AlignRight, row[0])
                self._hits.append((QRectF(0, y0, self.width(), rowh), i))
                if self.mode == 'sobol':
                    bh = min(rowh * 0.3, 9.0)
                    for val, ci, fill, off in ((row[1], row[2], True, rowh / 2 - bh - 1), (row[3], row[4], False,
                                                                                            rowh / 2 + 1)):
                        if val is None:
                            continue
                        x0, x1 = sorted((X(0.0), X(val)))
                        rr = QRectF(x0, y0 + off, max(x1 - x0, 1.0), bh)
                        if fill:
                            p.fillRect(rr, acc)
                        else:
                            p.setPen(QPen(acc, 1.2))
                            p.setBrush(Qt.BrushStyle.NoBrush)
                            p.drawRect(rr)
                        if ci and ci[0] is not None:
                            ym = y0 + off + bh / 2
                            p.setPen(QPen(THEME.color('ink'), 1))
                            p.drawLine(QPointF(X(ci[0]), ym), QPointF(X(ci[1]), ym))
                            for c in ci:
                                p.drawLine(QPointF(X(c), ym - 3), QPointF(X(c), ym + 3))
                elif row[1] is not None:
                    bh = min(rowh * 0.5, 12.0)
                    p.fillRect(QRectF(X(0.0), y0 + rowh / 2 - bh / 2, max(X(row[1]) - X(0.0), 1.0), bh), acc)
        self.paint_tip(p)
        p.end()

    def mouseMoveEvent(self, e):
        hit = next((i for rr, i in self._hits if rr.contains(e.position())), None)
        self.hover = hit
        self.tip = None
        if hit is not None:
            row = self.rows[hit]
            if self.mode == 'sobol':
                self.set_tip(e.position(), f'S1 {_g(row[1], 3)} · ST {_g(row[3], 3)}', row[0],
                             f'first order interval {_ci(row[2])}', f'total interval {_ci(row[4])}')
            elif self.mode == 'morris':
                self.set_tip(e.position(), f'μ* {_g(row[1])} · σ {_g(row[2])}', row[0],
                             f'μ* interval {_ci(row[3], 4)} {self.unit}')
            else:
                self.set_tip(e.position(), f'{_g(row[1])} {self.unit}'.strip(), row[0],
                             'from the lowest to the highest value over its levels' if self.mode == 'oat' else
                             'from the lowest to the highest mean over its levels or bins')
        self.update()


class EffectsChart(_Plot):
    """Main effects: the mean response per level of each factor with its interval, small multiples on one scale."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.panels, self.unit = [], ''         # (label, categorical, [(x, level, mean, lo, hi, n)])
        self._pts = []

    def set_data(self, panels, unit=''):
        self.panels, self.unit = panels, unit
        self.tip = None
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.title = 'Main effects'
        self.paint_title(p, f'mean and interval per level · {self.unit}'.rstrip(' ·'))
        if not self.panels:
            self.empty(p)
            p.end()
            return
        n = len(self.panels)
        cols = min(n, 3 if self.width() > 700 else 2)
        rows = int(math.ceil(n / cols))
        pw, ph = self.width() / cols, (self.height() - 24) / rows
        allv = [v for _, _, pts in self.panels for pt in pts for v in pt[2:5] if v is not None]
        ylim = _lim(allv)
        acc = _accent()
        self._pts = []
        for k, (lab, cat, pts) in enumerate(self.panels):
            cx, cy = (k % cols) * pw, 24 + (k // cols) * ph
            r = QRectF(cx + 48, cy + 22, max(pw - 64, 10), max(ph - 56, 10))
            p.setFont(sans(11, QFont.Weight.DemiBold))
            p.setPen(THEME.color('ink'))
            p.drawText(QRectF(cx + 48, cy + 2, pw - 56, 16), Qt.AlignmentFlag.AlignLeft, lab)
            pts = [pt for pt in pts if pt[2] is not None]
            if not pts:
                continue
            xs = list(range(len(pts))) if cat else [pt[0] for pt in pts]
            xlim = (-0.5, len(pts) - 0.5) if cat else _lim(xs, 0.04)
            X, Y = self.axes(p, r, xlim, ylim, xticks=not cat)
            if cat:
                p.setFont(sans(10))
                p.setPen(THEME.color('muted'))
                for x, pt in zip(xs, pts):
                    p.drawText(QRectF(X(x) - 50, r.bottom() + 3, 100, 13), Qt.AlignmentFlag.AlignCenter,
                               str(pt[1])[:16])
            qs = []
            for x, pt in zip(xs, pts):
                q = QPointF(X(x), Y(pt[2]))
                if pt[3] is not None:
                    p.setPen(QPen(THEME.color('strong'), 3))
                    p.drawLine(QPointF(q.x(), Y(pt[3])), QPointF(q.x(), Y(pt[4])))
                qs.append(q)
                self._pts.append((q, lab, pt))
            p.setPen(QPen(acc, 2))
            p.drawPolyline(QPolygonF(qs))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(acc)
            for q in qs:
                p.drawEllipse(q, 3, 3)
        self.paint_tip(p)
        p.end()

    def mouseMoveEvent(self, e):
        best = min(((math.hypot(q.x() - e.position().x(), q.y() - e.position().y()), q, lab, pt)
                    for q, lab, pt in self._pts), default=None, key=lambda t: t[0])
        self.tip = None
        if best is not None and best[0] < 12:
            _, q, lab, pt = best
            at = pt[1] if pt[1] is not None else f'{pt[0]:.4g}'
            self.set_tip(q, f'{_g(pt[2])} {self.unit}'.strip(), f'{lab} at {at}', f'{pt[5]} points' + (
                f' · interval {_g(pt[3])} – {_g(pt[4])}' if pt[3] is not None else ''))
        self.update()


class HeatMap(_Plot):
    """The response's mean on a two-factor grid; darker is better."""
    MARGIN = (70, 40, 70, 44)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.grid, self.xl, self.yl, self.unit, self.better_high = None, [], [], '', True
        self.xname, self.yname = '', ''

    def set_data(self, grid, xlabels, ylabels, unit, better_high, xname, yname):
        self.grid, self.xl, self.yl, self.unit, self.better_high = grid, xlabels, ylabels, unit, better_high
        self.xname, self.yname = xname, yname
        self.tip = None
        self.update()

    def _col(self, t):
        ramp = [QColor(h) for h in SEQ]
        t = min(max(t, 0.0), 1.0) * (len(ramp) - 1)
        i0 = int(math.floor(t))
        i1 = min(i0 + 1, len(ramp) - 1)
        f = t - i0
        a, b = ramp[i0], ramp[i1]
        return QColor(int(a.red() * (1 - f) + b.red() * f), int(a.green() * (1 - f) + b.green() * f),
                      int(a.blue() * (1 - f) + b.blue() * f))

    def _t(self, v, lo, hi):
        t = (v - lo) / ((hi - lo) or 1.0)
        return t if self.better_high else 1.0 - t

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.title = 'Response surface'
        self.paint_title(p, 'darker is better')
        M = self.grid
        if M is None or not np.isfinite(M).any():
            self.empty(p)
            p.end()
            return
        r = self.rect()
        nb, na = M.shape
        lo, hi = float(np.nanmin(M)), float(np.nanmax(M))
        cw, ch = r.width() / na, r.height() / nb
        p.setFont(mono(10))
        for ib in range(nb):
            for ia in range(na):
                v = M[ib, ia]
                rr = QRectF(r.left() + ia * cw + 1, r.bottom() - (ib + 1) * ch + 1, cw - 2, ch - 2)
                if not np.isfinite(v):
                    p.fillRect(rr, THEME.color('raised'))
                    continue
                t = self._t(v, lo, hi)
                p.fillRect(rr, self._col(t))
                if (ia, ib) == self.hover:
                    p.setPen(QPen(THEME.color('ink'), 1.5))
                    p.setBrush(Qt.BrushStyle.NoBrush)
                    p.drawRect(rr)
                if na * nb <= 64 and cw > 34:
                    p.setPen(QColor('#0b0b0b') if t < 0.55 else QColor('#f0efec'))
                    p.drawText(rr, Qt.AlignmentFlag.AlignCenter, f'{v:.3g}')
        p.setFont(sans(10))
        p.setPen(THEME.color('muted'))
        step_a = max(1, int(math.ceil(na / max(r.width() / 46, 1))))
        for ia, t in enumerate(self.xl):
            if ia % step_a == 0:
                p.drawText(QRectF(r.left() + ia * cw - 10, r.bottom() + 3, cw + 20, 13), Qt.AlignmentFlag.AlignCenter,
                           str(t)[:10])
        step_b = max(1, int(math.ceil(nb / max(r.height() / 16, 1))))
        for ib, t in enumerate(self.yl):
            if ib % step_b == 0:
                p.drawText(QRectF(0, r.bottom() - (ib + 1) * ch, r.left() - 6, ch), Qt.AlignmentFlag.AlignRight |
                           Qt.AlignmentFlag.AlignVCenter, str(t)[:10])
        p.drawText(QRectF(r.left(), r.bottom() + 18, r.width(), 13), Qt.AlignmentFlag.AlignRight, self.xname)
        p.drawText(QRectF(4, r.top() - 17, r.width(), 13), Qt.AlignmentFlag.AlignLeft, self.yname)
        # the colour scale
        sx = r.right() + 18
        for s in range(60):
            t = s / 59.0
            p.fillRect(QRectF(sx, r.bottom() - (s + 1) * r.height() / 60, 12, r.height() / 60 + 0.5), self._col(t))
        best, worst = (hi, lo) if self.better_high else (lo, hi)
        p.setFont(mono(10))
        p.setPen(THEME.color('ink2'))
        p.drawText(QRectF(sx - 8, r.top() - 15, 60, 12), Qt.AlignmentFlag.AlignLeft, f'{best:.3g}')
        p.drawText(QRectF(sx - 8, r.bottom() + 2, 60, 12), Qt.AlignmentFlag.AlignLeft, f'{worst:.3g}')
        self.paint_tip(p)
        p.end()

    def mouseMoveEvent(self, e):
        M = self.grid
        self.tip = None
        self.hover = None
        if M is not None:
            r = self.rect()
            nb, na = M.shape
            ia = int((e.position().x() - r.left()) // (r.width() / na))
            ib = int((r.bottom() - e.position().y()) // (r.height() / nb))
            if 0 <= ia < na and 0 <= ib < nb and r.contains(e.position()):
                self.hover = (ia, ib)
                v = M[ib, ia]
                self.set_tip(e.position(), f'{_g(v)} {self.unit}'.strip() if np.isfinite(v) else 'no point here',
                             f'{self.xname} {self.xl[ia]}', f'{self.yname} {self.yl[ib]}')
        self.update()


class ParetoChart(_Plot):
    """Every point on two responses; the Pareto front over all responses in the accent; the picked point ringed, the
    design as it is a square. A click picks the nearest point."""
    picked = pyqtSignal(int)
    MARGIN = (64, 40, 18, 44)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.x = self.y = np.zeros(0)
        self.front = set()
        self.sel = None
        self.base = None
        self.xname = self.yname = ''
        self._map = None

    def set_data(self, x, y, front, sel, base, xname, yname):
        self.x, self.y = np.asarray(x, float), np.asarray(y, float)
        self.front, self.sel, self.base, self.xname, self.yname = set(front), sel, base, xname, yname
        self.tip = None
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.title = 'Pareto front'
        self.paint_title(p, 'front ● · picked ○ · design □')
        ok = np.isfinite(self.x) & np.isfinite(self.y)
        if not ok.any():
            self._map = None
            self.empty(p)
            p.end()
            return
        r = self.rect()
        bx = [self.base[0]] if self.base is not None and np.isfinite(self.base[0]) else []
        by = [self.base[1]] if self.base is not None and np.isfinite(self.base[1]) else []
        X, Y = self.axes(p, r, _lim(list(self.x[ok]) + bx), _lim(list(self.y[ok]) + by), self.xname, self.yname)
        self._map = (X, Y)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(THEME.color('muted'))
        for i in np.nonzero(ok)[0]:
            if i not in self.front:
                p.drawRect(QRectF(X(self.x[i]) - 1.5, Y(self.y[i]) - 1.5, 3, 3))
        p.setBrush(_accent())
        for i in self.front:
            if ok[i]:
                p.drawEllipse(QPointF(X(self.x[i]), Y(self.y[i])), 4, 4)
        if bx and by:
            p.setPen(QPen(THEME.color('ink'), 1.2))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(QRectF(X(bx[0]) - 5, Y(by[0]) - 5, 10, 10))
        if self.sel is not None and 0 <= self.sel < len(self.x) and ok[self.sel]:
            p.setPen(QPen(THEME.color('ink'), 1.5))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(QPointF(X(self.x[self.sel]), Y(self.y[self.sel])), 8, 8)
        self.paint_tip(p)
        p.end()

    def _nearest(self, pos):
        if self._map is None:
            return None
        X, Y = self._map
        ok = np.isfinite(self.x) & np.isfinite(self.y)
        idx = np.nonzero(ok)[0]
        if not len(idx):
            return None
        d = np.hypot(np.array([X(v) for v in self.x[idx]]) - pos.x(), np.array([Y(v) for v in self.y[idx]]) - pos.y())
        k = int(np.argmin(d))
        return int(idx[k]) if d[k] < 12 else None

    def mousePressEvent(self, e):
        i = self._nearest(e.position())
        if i is not None:
            self.picked.emit(i)

    def mouseMoveEvent(self, e):
        i = self._nearest(e.position())
        self.tip = None
        if i is not None:
            self.set_tip(e.position(), f'point {i + 1}', f'{self.xname} {_g(self.x[i])}',
                         f'{self.yname} {_g(self.y[i])}', 'on the Pareto front' if i in self.front else '',
                         'click to pick it')
        self.update()


class DistChart(_Plot):
    """A response's Monte Carlo distribution with its 5th, 50th and 95th percentiles."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.counts, self.edges, self.marks, self.unit = None, None, [], ''
        self.setMinimumHeight(180)

    def set_data(self, values, marks, unit):
        v = np.asarray(values, float)
        v = v[np.isfinite(v)]
        self.unit, self.marks = unit, marks
        if not len(v):
            self.counts = None
        else:
            lo, hi = _lim(v, 0.02)
            self.counts, self.edges = np.histogram(v, bins=24, range=(lo, hi))
        self.tip = None
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.title = 'Distribution'
        self.paint_title(p, 'points per bin')
        if self.counts is None:
            self.empty(p)
            p.end()
            return
        r = self.rect()
        X, Y = self.axes(p, r, (self.edges[0], self.edges[-1]), (0, max(int(self.counts.max()), 1) * 1.12),
                         self.unit)
        col = _accent()
        for i, c in enumerate(self.counts):
            if c:
                p.fillRect(QRectF(X(self.edges[i]) + 1, Y(c), X(self.edges[i + 1]) - X(self.edges[i]) - 2,
                                  Y(0) - Y(c)), col.lighter(125) if i == self.hover else col)
        p.setFont(sans(10))
        for k, (x, lab) in enumerate(self.marks):
            if x is None:
                continue
            pen = QPen(THEME.color('ink'), 1)
            pen.setStyle(Qt.PenStyle.DashLine)
            p.setPen(pen)
            p.drawLine(QPointF(X(x), r.top()), QPointF(X(x), r.bottom()))
            p.setPen(THEME.color('ink'))
            p.drawText(QPointF(X(x) + 3, r.top() + 12 + 12 * k), f'{lab} {x:.4g}')
        self.paint_tip(p)
        p.end()

    def mouseMoveEvent(self, e):
        self.tip = None
        self.hover = None
        if self.counts is not None:
            r = self.rect()
            n = len(self.counts)
            i = int((e.position().x() - r.left()) // (r.width() / n))
            if 0 <= i < n and r.contains(e.position()):
                self.hover = i
                c = int(self.counts[i])
                self.set_tip(e.position(), f'{c} point' + ('' if c == 1 else 's'),
                             f'{self.edges[i]:.4g} – {self.edges[i + 1]:.4g} {self.unit}'.strip())
        self.update()


class ConvergenceChart(_Plot):
    """The running mean of a response with its interval against the number of Monte Carlo points."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.conv, self.unit = [], ''
        self.setMinimumHeight(180)

    def set_data(self, conv, unit):
        self.conv, self.unit = conv or [], unit
        self.tip = None
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.title = 'Convergence'
        self.paint_title(p, 'running mean and interval')
        if len(self.conv) < 2:
            self.empty(p)
            p.end()
            return
        r = self.rect()
        ns = [c['n'] for c in self.conv]
        lo = [c['mean'] - (c['half'] or 0.0) for c in self.conv]
        hi = [c['mean'] + (c['half'] or 0.0) for c in self.conv]
        X, Y = self.axes(p, r, (0, max(ns)), _lim(lo + hi), 'points', self.unit)
        band = QPolygonF([QPointF(X(n), Y(v)) for n, v in zip(ns, hi)] +
                         [QPointF(X(n), Y(v)) for n, v in reversed(list(zip(ns, lo)))])
        c = _accent()
        c.setAlphaF(0.25)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(c)
        p.drawPolygon(band)
        p.setPen(QPen(_accent(), 2))
        p.drawPolyline(QPolygonF([QPointF(X(cv['n']), Y(cv['mean'])) for cv in self.conv]))
        self.paint_tip(p)
        p.end()

    def mouseMoveEvent(self, e):
        self.tip = None
        if len(self.conv) >= 2:
            r = self.rect()
            ns = [c['n'] for c in self.conv]
            n = (e.position().x() - r.left()) / r.width() * max(ns)
            c = min(self.conv, key=lambda c_: abs(c_['n'] - n))
            if r.contains(e.position()):
                self.set_tip(e.position(), f'{_g(c["mean"])} {self.unit}'.strip(), f'after {c["n"]} points',
                             f'± {_g(c["half"])}' if c['half'] is not None else '')
        self.update()


class _Sig(QObject):
    done = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)
    progress = pyqtSignal(int, float)


# ============================================================================ the page
class ExperimentsPage(QWidget):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.result: EX.ExperimentResult | None = None
        self.estimate: dict | None = None
        self.busy = False
        self.uses_carla = False                 # the running experiment drives CARLA (detection responses)
        self.est_busy = False
        self.status_text = ''
        self.sel = None                         # the picked point
        self._fsel = ''                         # the factor in the editor
        self._rsel = -1                         # the response in the editor
        self._gen = 0
        self._est_gen = 0
        self._for = None                        # (project, key) of the result
        self._est_for = None
        self._then_run = False
        self._dirty = True
        self._shown = None
        self._syncing = False
        self.sig = _Sig()
        self.sig.done.connect(self._thread_done)
        self.sig.failed.connect(self._failed)
        self.sig.progress.connect(lambda g, f: self.bar.setValue(int(f * 1000)) if g == self._gen else None)
        self.setObjectName('base')
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setHandleWidth(2)
        split.addWidget(self._left())
        split.addWidget(self._right())
        split.setSizes([380, 1000])
        lay.addWidget(split)
        ctx.store.changed.connect(self._on_change)
        ctx.store.project_replaced.connect(self._on_replaced)
        THEME.changed.connect(self._retheme)
        self._sync_fields()
        self.refresh()

    # ------------------------------------------------------------------ the spec panel
    def _left(self):
        w = QWidget()
        w.setObjectName('panel')
        ll = QVBoxLayout(w)
        ll.setContentsMargins(8, 6, 8, 8)
        ll.setSpacing(4)
        top = QHBoxLayout()
        top.addWidget(label('Experiments'))
        top.addStretch(1)
        top.addWidget(label('designs · sensitivity · uncertainty', muted=True, small=True))
        ll.addLayout(top)
        ll.addWidget(hline())
        ll.addWidget(label('Vary design inputs over a designed set of points and compute any scorecard metric at '
                           'each: which inputs drive the figures (Sobol indices, Morris screening, main effects), '
                           'the best design found, and how uncertain the figures are when the inputs are (Monte '
                           'Carlo).', muted=True, small=True, wrap=True))
        row = QHBoxLayout()
        a = label('Name', ink2=True)
        a.setFixedWidth(60)
        row.addWidget(a)
        self.name = QLineEdit()
        self.name.setToolTip('The experiment\'s name (in its files and the variants it adds)')
        self.name.editingFinished.connect(self._name_edited)
        row.addWidget(self.name, 1)
        ll.addLayout(row)
        row = QHBoxLayout()
        a = label('Variant', ink2=True)
        a.setFixedWidth(60)
        row.addWidget(a)
        self.variant = _combo(16)
        self.variant.setToolTip('The variant the experiment varies (its cameras and sensors are the factors\' '
                                'targets)')
        self.variant.activated.connect(self._variant_picked)
        row.addWidget(self.variant, 1)
        ll.addLayout(row)
        # ---- factors
        ll.addSpacing(6)
        hdr = QHBoxLayout()
        hdr.addWidget(label('Factors', bold=True))
        hdr.addStretch(1)
        self.b_add_f = button('Add', 'ghost', 'plus', 'A design input to vary')
        self.fmenu = QMenu(self.b_add_f)
        self.fmenu.aboutToShow.connect(self._fill_factor_menu)
        self.b_add_f.setMenu(self.fmenu)
        self.b_del_f = button('Remove', 'ghost', 'minus', 'Remove the selected factor')
        self.b_del_f.clicked.connect(lambda: self.remove_factor())
        hdr.addWidget(self.b_add_f)
        hdr.addWidget(self.b_del_f)
        ll.addLayout(hdr)
        self.ftable = _table((('Factor', 'The design input and its id'), ('Range', 'The range or the levels'),
                              ('Draws', 'How the sampled designs draw it')), stretch=0)
        self.ftable.itemSelectionChanged.connect(self._factor_row)
        self.ftable.setFixedHeight(ROW_H * 6 + 6)
        ll.addWidget(self.ftable)
        self.fed = QWidget()
        fl = QVBoxLayout(self.fed)
        fl.setContentsMargins(0, 2, 0, 2)
        fl.setSpacing(3)
        self.fed_head = label('', ink2=True, small=True, wrap=True)
        fl.addWidget(self.fed_head)
        self.f_mode = SegmentedControl([('value', 'Value'), ('offset', 'Offset')], 'value')
        self.f_mode.setToolTip('Value: the input is set to the factor\'s value. Offset: added to the design\'s own')
        self.f_mode.changed.connect(lambda m: self.set_factor(mode=m))
        self.f_dist = SegmentedControl([('uniform', 'Uniform'), ('normal', 'Normal'), ('truncnormal', 'Truncated')],
                                       'uniform')
        self.f_dist.setToolTip('How Latin hypercube, Sobol and Monte Carlo draw it: uniform over the range, normal '
                               '(mean, SD), or normal truncated to the range')
        self.f_dist.changed.connect(lambda m: self.set_factor(dist=m))
        r1 = QHBoxLayout()
        r1.addWidget(self.f_mode)
        r1.addStretch(1)
        fl.addLayout(r1)
        grid = QGridLayout()
        grid.setVerticalSpacing(2)
        grid.setHorizontalSpacing(6)
        self.f_fields = {}
        for i, (key, text, tip) in enumerate(NUM_FIELDS):
            f = ScrubSpinBox('', 'plain', 0 if key == 'levels' else 4, ' ', step=1.0 if key == 'levels' else 0.01,
                             width=104)
            f.setToolTip(tip)
            f.committed.connect(lambda v, k=key: self.set_factor(**{k: (int(round(v)) if k == 'levels' else v)}))
            a = label(text, ink2=True)
            a.setToolTip(tip)
            grid.addWidget(a, i // 2, (i % 2) * 2)
            grid.addWidget(f, i // 2, (i % 2) * 2 + 1)
            self.f_fields[key] = f
        fl.addLayout(grid)
        fl.addWidget(self.f_dist)
        self.f_mirror = ToggleSwitch('Mirror partner follows')
        self.f_mirror.setToolTip('The camera\'s mirror partner gets the mirrored value (y, yaw and roll change sign)')
        self.f_mirror.clicked.connect(lambda on: self.set_factor(mirror=bool(on)))
        fl.addWidget(self.f_mirror)
        self.f_levels = QListWidget()
        self.f_levels.setToolTip('The levels the factor takes (tick to include)')
        self.f_levels.setFixedHeight(ROW_H * 5)
        self.f_levels.itemChanged.connect(self._levels_ticked)
        fl.addWidget(self.f_levels)
        ll.addWidget(self.fed)
        # ---- responses
        ll.addSpacing(6)
        hdr = QHBoxLayout()
        hdr.addWidget(label('Responses', bold=True))
        hdr.addStretch(1)
        self.b_add_r = button('Add', 'ghost', 'plus', 'A scorecard metric to compute at every point')
        self.rmenu = QMenu(self.b_add_r)
        self.rmenu.aboutToShow.connect(self._fill_response_menu)
        self.b_add_r.setMenu(self.rmenu)
        self.b_del_r = button('Remove', 'ghost', 'minus', 'Remove the selected response')
        self.b_del_r.clicked.connect(lambda: self.remove_response())
        hdr.addWidget(self.b_add_r)
        hdr.addWidget(self.b_del_r)
        ll.addLayout(hdr)
        self.rtable = _table((('Response', 'The metric and its parameters'), ('Better', 'More or less is better'),
                              ('Cost', 'cheap: no geometry; study: a geometric study per point')), stretch=0)
        self.rtable.itemSelectionChanged.connect(self._response_row)
        self.rtable.setFixedHeight(ROW_H * 5 + 6)
        ll.addWidget(self.rtable)
        self.red = QWidget()
        self.red_lay = QGridLayout(self.red)
        self.red_lay.setContentsMargins(0, 2, 0, 2)
        self.red_lay.setVerticalSpacing(2)
        self.red_lay.setHorizontalSpacing(6)
        self._param_boxes = []
        ll.addWidget(self.red)
        # ---- design
        ll.addSpacing(6)
        ll.addWidget(label('Design', bold=True))
        self.design = _combo(18)
        for k in EX.DESIGNS:
            self.design.addItem(EX.DESIGN_LABEL[k], k)
        self.design.setToolTip('Full factorial and one at a time cover the levels; Latin hypercube spreads samples; '
                               'Sobol measures variance shares; Morris screens many factors cheaply; Monte Carlo '
                               'propagates the factors\' distributions')
        self.design.activated.connect(lambda i: self.set_design(design=self.design.itemData(i)))
        ll.addWidget(self.design)
        grid = QGridLayout()
        grid.setVerticalSpacing(2)
        grid.setHorizontalSpacing(6)
        self.d_fields, self.d_labels = {}, {}
        for i, (key, text, dec, tip, _) in enumerate(DESIGN_FIELDS):
            f = ScrubSpinBox('', 'plain', dec, '', step=1.0 if not dec else 0.01, width=96)
            f.setToolTip(tip)
            f.committed.connect(lambda v, k=key: self.set_design(**{k: v}))
            a = label(text, ink2=True)
            a.setToolTip(tip)
            grid.addWidget(a, i, 0)
            grid.addWidget(f, i, 1)
            self.d_fields[key], self.d_labels[key] = f, a
        grid.setColumnStretch(2, 1)
        ll.addLayout(grid)
        self.count = label('', mono_=True)
        ll.addWidget(self.count)
        self.cost = label('', muted=True, small=True, wrap=True)
        ll.addWidget(self.cost)
        self.prob_host = QWidget()
        self.prob_lay = QVBoxLayout(self.prob_host)
        self.prob_lay.setContentsMargins(0, 2, 0, 2)
        self.prob_lay.setSpacing(3)
        self._probs = []
        ll.addWidget(self.prob_host)
        ll.addStretch(1)
        self.status = label('', muted=True, small=True, wrap=True)
        ll.addWidget(self.status)
        self.bar = QProgressBar()
        self.bar.setTextVisible(False)
        self.bar.setRange(0, 1000)
        self.bar.hide()
        ll.addWidget(self.bar)
        bl = QHBoxLayout()
        self.b_import = button('Import…', 'ghost', 'import', 'Load an experiment spec (experiment_spec.json) saved '
                                                             'with an earlier export')
        self.b_import.clicked.connect(lambda: self.import_spec())
        bl.addWidget(self.b_import)
        bl.addStretch(1)
        self.b_export = button('Export…', None, 'export', 'The spec (JSON, re-runnable), every point (CSV), the '
                                                           'analysis (JSON) and the plots (PNG)')
        self.b_export.clicked.connect(lambda: self.export())
        bl.addWidget(self.b_export)
        ll.addLayout(bl)
        bl2 = QHBoxLayout()
        self.b_cancel = button('Cancel', 'ghost', None, 'Stop the run: the points done stay cached, Run resumes')
        self.b_cancel.clicked.connect(self.cancel)
        bl2.addWidget(self.b_cancel)
        bl2.addStretch(1)
        self.b_est = button('Estimate', None, None, 'Time the design and the first points to estimate the run')
        self.b_est.clicked.connect(lambda: self.estimate_cost())
        self.b_run = button('Run', 'primary', None, 'Evaluate every point of the design (a background job)')
        self.b_run.clicked.connect(lambda: self.run())
        bl2.addWidget(self.b_est)
        bl2.addWidget(self.b_run)
        ll.addLayout(bl2)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        sc.setWidget(w)
        sc.setMinimumWidth(340)
        return sc

    # ------------------------------------------------------------------ the results panel
    def _right(self):
        right = QWidget()
        right.setObjectName('panel')
        rl = QVBoxLayout(right)
        rl.setContentsMargins(10, 8, 10, 6)
        rl.setSpacing(6)
        tiles = QHBoxLayout()
        tiles.setSpacing(8)
        self.tiles = {}
        for key, caption, tip in (('points', 'Points', 'Design points evaluated, and how many came from the cache'),
                                  ('time', 'Time', 'The run\'s wall time and the time per point'),
                                  ('best', 'Best', 'The best point on the first response (on the Pareto front, '
                                                   'meeting the targets where set)'),
                                  ('driver', 'Top driver', 'The factor the design finds most influential on the shown '
                                                           'response')):
            t = self.tiles[key] = Tile(caption, tip)
            t.setMinimumWidth(120)
            tiles.addWidget(t, 1)
        rl.addLayout(tiles)
        pick = QHBoxLayout()
        pick.addWidget(label('Response', ink2=True))
        self.r_pick = _combo(24)
        self.r_pick.activated.connect(lambda _: self._view_changed())
        pick.addWidget(self.r_pick, 2)
        pick.addSpacing(8)
        pick.addWidget(label('Surface', ink2=True))
        self.x_pick = _combo(10)
        self.y_pick = _combo(10)
        for cb in (self.x_pick, self.y_pick):
            cb.setToolTip('The two factors of the response surface')
            cb.activated.connect(lambda _: self._view_changed())
            pick.addWidget(cb, 1)
        rl.addLayout(pick)
        self.tabs = QTabWidget()
        sw = QWidget()
        sl = QVBoxLayout(sw)
        sl.setContentsMargins(4, 4, 4, 4)
        self.index_chart = IndexChart()
        sl.addWidget(self.index_chart, 3)
        self.index_table = _table((('Factor', ''), ('A', ''), ('A interval', ''), ('B', ''), ('B interval', '')),
                                  select=False)
        sl.addWidget(self.index_table, 2)
        self.tabs.addTab(sw, TABS[0])
        self.effects = EffectsChart()
        self.tabs.addTab(self.effects, TABS[1])
        self.heat = HeatMap()
        self.tabs.addTab(self.heat, TABS[2])
        pw = QWidget()
        pl = QVBoxLayout(pw)
        pl.setContentsMargins(4, 4, 4, 4)
        prow = QHBoxLayout()
        prow.addWidget(label('Against', ink2=True))
        self.p_pick = _combo(18)
        self.p_pick.setToolTip('The second response of the Pareto plot')
        self.p_pick.activated.connect(lambda _: self._view_changed())
        prow.addWidget(self.p_pick, 1)
        pl.addLayout(prow)
        self.pareto = ParetoChart()
        self.pareto.picked.connect(self.pick)
        pl.addWidget(self.pareto, 1)
        self.p_text = label('', ink2=True, small=True, wrap=True)
        pl.addWidget(self.p_text)
        prow2 = QHBoxLayout()
        self.b_best = button('Pick the best', 'ghost', None, 'Pick the best point found')
        self.b_best.clicked.connect(lambda: self.pick(self.result.analysis['best']['index'])
                                    if self.result is not None and self.result.analysis.get('best') else None)
        prow2.addWidget(self.b_best)
        prow2.addStretch(1)
        self.b_variant = button('Add as variant', 'primary', 'plus', 'A new variant with the picked point\'s cameras '
                                                                    'and sensors (undoable)')
        self.b_variant.clicked.connect(lambda: self.add_variant())
        prow2.addWidget(self.b_variant)
        pl.addLayout(prow2)
        self.tabs.addTab(pw, TABS[3])
        uw = QWidget()
        ul = QVBoxLayout(uw)
        ul.setContentsMargins(4, 4, 4, 4)
        urow = QHBoxLayout()
        self.dist = DistChart()
        self.conv = ConvergenceChart()
        urow.addWidget(self.dist, 1)
        urow.addWidget(self.conv, 1)
        ul.addLayout(urow, 3)
        self.u_table = _table((('Statistic', ''), ('Value', ''), ('Interval', 'Bootstrap (percentile) interval; the '
                                                                             'share meeting the target: Wilson')),
                              select=False)
        ul.addWidget(self.u_table, 2)
        self.tabs.addTab(uw, TABS[4])
        ptw = QWidget()
        ptl = QVBoxLayout(ptw)
        ptl.setContentsMargins(4, 4, 4, 4)
        self.pt_note = label('', muted=True, small=True, wrap=True)
        ptl.addWidget(self.pt_note)
        self.points = _table((('#', ''),), stretch=-1)
        self.points.itemSelectionChanged.connect(self._point_row)
        ptl.addWidget(self.points, 1)
        self.tabs.addTab(ptw, TABS[5])
        nw = QWidget()
        nl = QVBoxLayout(nw)
        nl.setContentsMargins(4, 4, 4, 4)
        self.find_host = QWidget()
        self.find_lay = QVBoxLayout(self.find_host)
        self.find_lay.setContentsMargins(0, 0, 0, 0)
        self.find_lay.setSpacing(4)
        self.find_lay.addStretch(1)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setWidget(self.find_host)
        nl.addWidget(sc)
        self.findings: list[StatusLine] = []
        self.tabs.addTab(nw, TABS[6])
        rl.addWidget(self.tabs, 1)
        return right

    # ------------------------------------------------------------------ settings
    @property
    def settings(self) -> EX.ExperimentSettings:
        return EX.settings_of(self.ctx.store.project)

    def _variant(self):
        p = self.ctx.store.project
        return p.variant_by_uid(self.settings.variant) or p.variant

    def _store(self, st: EX.ExperimentSettings, text) -> bool:
        new = EX.ExperimentSettings.from_dict(st.to_dict()).to_dict()
        if new == self.settings.to_dict():
            self._sync_fields()                 # a clamped value: show what is stored
            return False

        def fn(p):
            p.analysis.experiments = new
        self.ctx.store.edit('analysis', fn, f'Experiments · {text}')
        return True

    def _name_edited(self):
        t = self.name.text().strip()
        st = self.settings
        if t and t != st.name:
            st.name = t
            self._store(st, f'name {t}')
        else:
            self._sync_fields()

    def _variant_picked(self, i):
        uid = self.variant.itemData(i) or ''
        st = self.settings
        if uid == (st.variant or self.ctx.store.project.active_uid):
            return
        p = self.ctx.store.project
        old = p.variant_by_uid(st.variant) or p.variant
        new = p.variant_by_uid(uid)
        st.variant = uid
        if old is not None and new is not None:              # factors follow their targets by tag
            for f in st.factors:
                if f['kind'] == 'camera' and old.cam(f['target']) is not None:
                    c = new.by_tag(old.cam(f['target']).tag)
                    f['target'] = c.uid if c is not None else f['target']
                elif f['kind'] == 'sensor' and old.sensor(f['target']) is not None:
                    tag = old.sensor(f['target']).tag
                    s = next((x for x in new.sensors if x.tag == tag), None)
                    f['target'] = s.uid if s is not None else f['target']
        self._store(st, f'variant {new.name if new else uid}')

    # ---- factors
    def _fill_factor_menu(self):
        m = self.fmenu
        m.clear()
        p = self.ctx.store.project
        v = self._variant()
        groups = {}
        for t in EX.targets(p, v):
            groups.setdefault(t['group'], []).append(t)
        for g, items in groups.items():
            sub = m.addMenu(g)
            for t in items:
                a = sub.addAction(t['label'])
                a.triggered.connect(lambda _=False, t=t: self.add_factor(t['kind'], t['target'], t['attr']))

    def add_factor(self, kind, target='', attr='') -> str:
        p = self.ctx.store.project
        st = self.settings
        if len(st.factors) >= EX.MAX_FACTORS:
            self.ctx.msg(f'An experiment takes at most {EX.MAX_FACTORS} factors.', 4000)
            return ''
        d = EX.new_factor(p, self._variant(), kind, target, attr, [f['id'] for f in st.factors])
        st.factors = st.factors + [d]
        f = EX.Factor.from_dict(d)
        self._fsel = d['id']
        self._store(st, f'factor {d["id"]} {EX.factor_label(f, p, self._variant())}')
        return d['id']

    def remove_factor(self, fid=None) -> bool:
        st = self.settings
        fid = fid or self._fsel
        if not any(f['id'] == fid for f in st.factors):
            return False
        st.factors = [f for f in st.factors if f['id'] != fid]
        self._fsel = st.factors[0]['id'] if st.factors else ''
        return self._store(st, f'removed factor {fid}')

    def set_factor(self, fid=None, **kw) -> bool:
        if self._syncing:
            return False
        st = self.settings
        fid = fid or self._fsel
        d = next((f for f in st.factors if f['id'] == fid), None)
        if d is None:
            return False
        d.update(kw)
        what = ', '.join(f'{k} {v:g}' if isinstance(v, (int, float)) and not isinstance(v, bool) else
                         f'{k} {v}' for k, v in kw.items())
        return self._store(st, f'factor {fid} {what}')

    def _levels_ticked(self, item):
        if self._syncing:
            return
        st = self.settings
        d = next((f for f in st.factors if f['id'] == self._fsel), None)
        if d is None:
            return
        on = [self.f_levels.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.f_levels.count())
              if self.f_levels.item(i).checkState() == Qt.CheckState.Checked]
        self.set_factor(choices=on)

    def _factor_row(self):
        if self._syncing:
            return
        r = self.ftable.currentRow()
        st = self.settings
        if 0 <= r < len(st.factors):
            self._fsel = st.factors[r]['id']
            self._sync_factor_editor()

    # ---- responses
    def _fill_response_menu(self):
        m = self.rmenu
        m.clear()
        p = self.ctx.store.project
        groups = {}
        for r in EX.response_catalogue(p, self._variant()):
            groups.setdefault(r['study_label'], []).append(r)
        for g, items in groups.items():
            sub = m.addMenu(g)
            for r in items:
                a = sub.addAction(r['label'] + (f' ({r["unit"]})' if r['unit'] else ''))
                a.setToolTip(r['help'])
                a.triggered.connect(lambda _=False, mid=r['id']: self.add_response(mid))

    def add_response(self, mid, params=None) -> bool:
        st = self.settings
        r = EX.clean_response({'metric': mid, 'params': params or {}})
        if r is None:
            return False
        if any(EX.response_key(x) == EX.response_key(r) for x in st.responses):
            self.ctx.msg('That response is in the experiment already.', 3000)
            return False
        if len(st.responses) >= EX.MAX_RESPONSES:
            self.ctx.msg(f'An experiment takes at most {EX.MAX_RESPONSES} responses.', 4000)
            return False
        st.responses = st.responses + [r]
        self._rsel = len(st.responses) - 1
        return self._store(st, f'response {EX.metric(mid).label}')

    def remove_response(self, i=None) -> bool:
        st = self.settings
        i = self._rsel if i is None else i
        if not 0 <= i < len(st.responses):
            return False
        lab = EX.metric(st.responses[i]['metric']).label
        st.responses = [r for k, r in enumerate(st.responses) if k != i]
        self._rsel = min(i, len(st.responses) - 1)
        return self._store(st, f'removed response {lab}')

    def set_response(self, i=None, **kw) -> bool:
        if self._syncing:
            return False
        st = self.settings
        i = self._rsel if i is None else i
        if not 0 <= i < len(st.responses):
            return False
        r = dict(st.responses[i])
        if 'params' in kw:
            r['params'] = dict(r['params'], **kw.pop('params'))
        r.update(kw)
        st.responses = [r if k == i else x for k, x in enumerate(st.responses)]
        return self._store(st, f'response {EX.metric(r["metric"]).label}')

    def _response_row(self):
        if self._syncing:
            return
        r = self.rtable.currentRow()
        if 0 <= r < len(self.settings.responses):
            self._rsel = r
            self._sync_response_editor()

    # ---- design
    def set_design(self, **kw) -> bool:
        if self._syncing:
            return False
        st = self.settings
        for k, v in kw.items():
            setattr(st, k, v)
        what = ', '.join(f'{k} {v}' if isinstance(v, str) else f'{k} {v:g}' for k, v in kw.items())
        return self._store(st, what)

    # ------------------------------------------------------------------ syncing the fields
    def _sync_fields(self):
        st = self.settings
        p = self.ctx.store.project
        v = self._variant()
        self._syncing = True
        try:
            if not self.name.hasFocus():
                self.name.setText(st.name)
            self.variant.clear()
            for x in p.variants:
                self.variant.addItem(x.name + (' (active)' if x.uid == p.active_uid else ''), x.uid)
            self.variant.setCurrentIndex(max(0, next((i for i, x in enumerate(p.variants) if x.uid == v.uid), 0)))
            fs = EX.factors_of(st)
            if self._fsel not in {f.id for f in fs}:
                self._fsel = fs[0].id if fs else ''
            t = self.ftable
            t.setRowCount(len(fs))
            for r, f in enumerate(fs):
                lab = EX.factor_label(f, p, v)
                t.setItem(r, 0, QTableWidgetItem(f'{f.id} · {lab}'))
                if f.categorical:
                    rng = ', '.join(f.choices) or 'no level'
                elif f.unit == 'm' and f.mode == 'offset':
                    rng = f'{f.lo * 1000:+.0f} … {f.hi * 1000:+.0f} mm'
                else:
                    rng = f'{f.lo:.4g} … {f.hi:.4g} {f.unit}'.strip()
                t.setItem(r, 1, QTableWidgetItem(rng))
                draws = 'levels' if f.categorical else (EX.DIST_LABEL[f.dist] + (
                    f' {f.mean:.3g} ± {f.sd:.3g}' if f.dist != 'uniform' else ''))
                t.setItem(r, 2, QTableWidgetItem(draws))
                if f.id == self._fsel:
                    t.selectRow(r)
            self.b_del_f.setEnabled(bool(fs))
            rs = st.responses
            self._rsel = min(max(self._rsel, 0), len(rs) - 1) if rs else -1
            t = self.rtable
            t.setRowCount(len(rs))
            for r, resp in enumerate(EX.responses_of(st, p, v)):
                it = QTableWidgetItem(resp.label + (f' ({resp.unit})' if resp.unit else ''))
                it.setToolTip(resp.label)
                t.setItem(r, 0, it)
                t.setItem(r, 1, QTableWidgetItem(('more' if resp.maximise else 'less') + (
                    f' · {EX.RQ.OP_TEXT[resp.op]} {resp.target:g}' if resp.target is not None and resp.op else '')))
                t.setItem(r, 2, QTableWidgetItem(resp.cost))
                if r == self._rsel:
                    t.selectRow(r)
            self.b_del_r.setEnabled(bool(rs))
            self.design.setCurrentIndex(EX.DESIGNS.index(st.design))
            for key, _, dec, _, designs in DESIGN_FIELDS:
                f = self.d_fields[key]
                f.set_value(float(getattr(st, key)))
                on = st.design in designs
                f.setVisible(on)
                self.d_labels[key].setVisible(on)
            self.count.setText(EX.design_text(fs, st))
        finally:
            self._syncing = False
        self._sync_factor_editor()
        self._sync_response_editor()
        self._sync_problems()

    def _sync_factor_editor(self):
        st = self.settings
        d = next((f for f in st.factors if f['id'] == self._fsel), None)
        self.fed.setVisible(d is not None)
        if d is None:
            return
        f = EX.Factor.from_dict(d)
        p, v = self.ctx.store.project, self._variant()
        self._syncing = True
        try:
            nom = EX.nominal_of(f, p, v)
            self.fed_head.setText(f'{f.id} · {EX.factor_label(f, p, v)}' + (
                f' · as designed {nom if isinstance(nom, str) else f"{nom:.4g}"}' if nom not in (None, '') and
                not (f.mode == 'offset' and not f.categorical) else ''))
            numeric = not f.categorical
            self.f_mode.setVisible(numeric and f.kind in ('camera', 'sensor'))
            self.f_mode.set_value(f.mode)
            self.f_dist.setVisible(numeric)
            self.f_dist.set_value(f.dist)
            for key, fld in self.f_fields.items():
                fld.setVisible(numeric)
                fld.set_unit(f.unit if key in ('lo', 'hi', 'mean', 'sd') else '')
                fld.set_value(float(getattr(f, key)))
            for key in ('mean', 'sd'):
                self.f_fields[key].setEnabled(f.dist != 'uniform')
            for i in range(self.fed.layout().count()):
                item = self.fed.layout().itemAt(i)
                if item.layout() is not None:          # the field labels follow their fields
                    lay = item.layout()
                    for j in range(lay.count()):
                        w = lay.itemAt(j).widget()
                        if w is not None and w not in self.f_fields.values() and w is not self.f_mode:
                            w.setVisible(numeric)
            cam = v.cam(f.target) if f.kind == 'camera' else None
            self.f_mirror.setVisible(f.kind == 'camera')
            self.f_mirror.setEnabled(bool(cam is not None and cam.link_uid and v.cam(cam.link_uid)))
            self.f_mirror.setChecked(f.mirror)
            self.f_levels.setVisible(f.categorical)
            if f.categorical:
                self.f_levels.clear()
                opts = EX.choice_options(p, v, f.kind, f.target, f.attr)
                known = {k for k, _ in opts}
                for k, lab in opts + [(c, f'{c} (not in the project)') for c in f.choices if c not in known]:
                    it = QListWidgetItem(lab)
                    it.setData(Qt.ItemDataRole.UserRole, k)
                    it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    it.setCheckState(Qt.CheckState.Checked if k in f.choices else Qt.CheckState.Unchecked)
                    self.f_levels.addItem(it)
        finally:
            self._syncing = False

    def _sync_response_editor(self):
        for w in self._param_boxes:
            if isinstance(w, QLineEdit):
                w.editingFinished.emit()       # a target typed but not yet committed (committed after this sync)
            w.blockSignals(True)
            w.hide()
            self.red_lay.removeWidget(w)
            w.deleteLater()
        self._param_boxes = []
        st = self.settings
        if not 0 <= self._rsel < len(st.responses):
            self.red.hide()
            return
        self.red.show()
        r = st.responses[self._rsel]
        m = EX.metric(r['metric'])
        p, v = self.ctx.store.project, self._variant()
        row = 0
        for prm in m.params:
            a = label(prm.label, ink2=True)
            cb = _combo(14)
            for k, lab in prm.options(p, v):
                cb.addItem(lab, k)
            cur = r['params'].get(prm.key, prm.default)
            i = next((j for j in range(cb.count()) if str(cb.itemData(j)).lower() == str(cur).lower()), -1)
            if i < 0:
                cb.addItem(f'{cur} (not in the project)', cur)
                i = cb.count() - 1
            cb.setCurrentIndex(i)
            cb.activated.connect(lambda j, key=prm.key, c=cb: self.set_response(params={key: c.itemData(j)}))
            self.red_lay.addWidget(a, row, 0)
            self.red_lay.addWidget(cb, row, 1)
            self._param_boxes += [a, cb]
            row += 1
        a = label('Target', ink2=True)
        a.setToolTip('Optional: the value a point must reach (the best point is picked among those meeting every '
                     'target; Monte Carlo reports the share meeting it). Empty: none')
        ed = QLineEdit('' if r.get('target') is None else f'{r["target"]:g}')
        ed.setPlaceholderText(f'{EX.RQ.OP_TEXT[m.better]} value, or empty')
        # committed after the signal returns: a sync (another row picked) tears this field down while it signals
        ed.editingFinished.connect(lambda e=ed, k=EX.response_key(r): QTimer.singleShot(
            0, lambda t=e.text(): self._target_edited(k, t)))
        self.red_lay.addWidget(a, row, 0)
        self.red_lay.addWidget(ed, row, 1)
        self._param_boxes += [a, ed]

    def _target_edited(self, key, text) -> bool:
        """The target typed for the response key ('' clears it); not a number: the field shows the stored one."""
        st = self.settings
        i = next((j for j, r in enumerate(st.responses) if EX.response_key(r) == key), None)
        if i is None:
            return False
        t = str(text).strip()
        cur = st.responses[i].get('target')
        if not t:
            return self.set_response(i, target=None, op='') if cur is not None else False
        try:
            from ...core.mathutil import parse_number
            val = float(parse_number(t, 'plain'))
        except Exception:                      # parse_number raises on anything that is not a number
            self._sync_response_editor()
            return False
        if cur is None or abs(val - cur) > 1e-12:
            return self.set_response(i, target=val, op=EX.metric(st.responses[i]['metric']).better)
        return False

    def _sync_problems(self):
        for w in self._probs:
            w.hide()
            self.prob_lay.removeWidget(w)
            w.deleteLater()
        self._probs = []
        p = self.ctx.store.project
        st = self.settings
        probs = EX.validate(p, st) if (st.factors or st.responses) else []
        for sev, msg in probs[:6]:
            w = StatusLine('critical' if sev == 'error' else 'warning', msg)
            self.prob_lay.addWidget(w)
            self._probs.append(w)
        est = self.estimate if self._est_for == self._spec_key() else None
        if est is not None:
            txt = (f'About {_dur(est["total_s"])} for {est["points"] - est["cached"]:,} new points '
                   f'({est["per_point_s"] * 1000:.0f} ms a point, {est["workers"]} workers)' +
                   (f', {est["cached"]:,} cached' if est['cached'] else '') + '.')
            if est['heavy']:
                txt += ' Heavy: ' + ', '.join(est['heavy']) + '.'
            self.cost.setText(txt)
        else:
            heavy = sorted({r.study for r in EX.responses_of(st, p) if r.cost != 'cheap'})
            self.cost.setText(('Studies run per point: ' + ', '.join(EX.study_of(s).label.lower() for s in heavy) +
                               '. Estimate times them.') if heavy else
                              ('Cheap responses only (no geometry).' if st.responses else ''))

    # ------------------------------------------------------------------ state
    def _spec_key(self, p=None):
        p = p or self.ctx.store.project
        st = EX.settings_of(p)
        v = p.variant_by_uid(st.variant) or p.variant
        try:
            bk = EX.base_key(p, v.uid)
        except Exception:                      # noqa: BLE001 — a half-built project: never stale-check on it
            bk = ''
        d = dict(st.to_dict(), variant=v.uid)  # the variant it runs on: following the active one or pinned to it alike
        return json.dumps([d, bk], sort_keys=True, default=str)

    @property
    def stale(self) -> bool:
        return self.result is not None and self._for is not None and self._for[1] != self._spec_key()

    def _on_change(self, *_):
        if not self.isVisible():
            self._dirty = True
            return
        self._sync_fields()
        self.refresh()

    def _on_replaced(self, *_):
        self._gen += 1                          # a run still going is for the old project
        self._est_gen += 1
        self.result, self._for, self.busy, self.est_busy = None, None, False, False
        self.estimate, self._est_for, self.sel = None, None, None
        self._fsel, self._rsel = '', -1
        self.status_text = ''
        self._shown = False
        self._sync_fields()
        self.refresh()

    def _retheme(self, *_):
        self._shown = False
        self.refresh()

    def showEvent(self, e):
        super().showEvent(e)
        self.activate()

    def activate(self):
        if self._dirty:
            self._dirty = False
            self._sync_fields()
        self.refresh()

    # ------------------------------------------------------------------ run
    def _engine(self):
        win = self.ctx.window
        return win._shared_engine() if win is not None and hasattr(win, '_shared_engine') else \
            self.ctx.analysis.engine.share()

    def _start(self, kind, title, work, on_done, gen_attr):
        win = self.ctx.window
        gen = getattr(self, gen_attr)
        if win is not None and hasattr(win, 'run_job'):
            ok = win.run_job(kind, title, work, lambda out: on_done(gen, out), lambda m: self._failed(gen, m))
            return ok

        def thread():
            try:
                out = work(lambda f, *_: self.sig.progress.emit(self._gen, f), lambda: gen != getattr(self, gen_attr))
                self.sig.done.emit(gen, (kind, out))
            except RuntimeError:                # the page closed while computing
                pass
            except Exception as e:              # noqa: BLE001
                try:
                    self.sig.failed.emit(gen, f'{type(e).__name__}: {e}\n{traceback.format_exc()}')
                except RuntimeError:
                    pass
        threading.Thread(target=thread, daemon=True).start()
        return True

    def _thread_done(self, gen, payload):
        kind, out = payload
        if kind == JOB:
            self._done(gen, out)
        else:
            self._est_done(gen, out)

    def estimate_cost(self, then_run=False) -> bool:
        """Time the design and its first points (a background job); then_run: start the run after it (asking first
        when it will take long)."""
        if self.busy or self.est_busy:
            return False
        p = self.ctx.store.project
        bad = EX.errors(p, self.settings)
        if bad:
            self.status_text = bad[0]
            self.refresh()
            return False
        project = p.snapshot()
        st = EX.settings_of(project)
        eng = self._engine()
        self._est_gen += 1
        key = (p, self._spec_key(p))
        self.est_busy = True
        self._then_run = then_run
        self.status_text = 'Estimating the run…'

        def work(progress, cancel):
            return key, EX.estimate(project, eng, st, cancel=cancel)
        ok = self._start(JOB_EST, 'Experiments · estimate', work, self._est_done, '_est_gen')
        if not ok:
            self.est_busy = False
        self.refresh()
        return ok

    def _est_done(self, gen, out):
        if gen != self._est_gen:
            return
        self.est_busy = False
        key, est = out if out is not None else (None, None)
        if est is None or key[0] is not self.ctx.store.project:
            self.status_text = 'Estimate cancelled' if est is None else ''
            self.refresh()
            return
        self.estimate, self._est_for = est, key[1]
        self.status_text = f'{est["points"]:,} points · about {_dur(est["total_s"])}'
        self._sync_problems()
        self.refresh()
        if self._then_run and key[1] == self._spec_key():
            self._then_run = False
            if est['total_s'] > LONG_S:
                q = QMessageBox.question(self, 'Run the experiment',
                                         f'The run takes about {_dur(est["total_s"])} for '
                                         f'{est["points"] - est["cached"]:,} points' +
                                         (f' (heavy: {", ".join(est["heavy"])})' if est['heavy'] else '') +
                                         '. It runs in the background and can be cancelled (the points done stay '
                                         'cached). Run it?')
                if q != QMessageBox.StandardButton.Yes:
                    return
            self.run(confirm=False)

    def run(self, confirm=True, plan=None) -> bool:
        """Run the experiment (a background job). confirm: estimate first and ask when it will take long. plan: the
        Copilot's ticket when an approved plan runs it (no question asked: the plan was approved with its cost)."""
        win = self.ctx.window
        if self.busy:
            return False
        if isinstance(plan, dict):              # reached from a Copilot plan: only with its approval record
            from ...assistant import copilot as CP
            ok, why = CP.gate(self.ctx.store.project, plan, 'experiment', self.settings.to_dict())
            if not ok:
                self.status_text = why
                self.refresh()
                return False
        p = self.ctx.store.project
        st = self.settings
        bad = EX.errors(p, st)
        if bad:
            self.status_text = bad[0]
            self.refresh()
            return False
        self.uses_carla = any(r.study in EX.CARLA_STUDY_S for r in EX.responses_of(st, p))
        other = win.carla_user('experiments') if self.uses_carla and hasattr(win, 'carla_user') else ''
        if other:                                       # one client steps a synchronous CARLA world
            self.status_text = other
            self.refresh()
            return False
        if confirm and self._est_for != self._spec_key(p):
            return self.estimate_cost(then_run=True)
        if confirm and self.estimate is not None and self.estimate['total_s'] > LONG_S:
            q = QMessageBox.question(self, 'Run the experiment', f'The run takes about '
                                     f'{_dur(self.estimate["total_s"])}. Run it?')
            if q != QMessageBox.StandardButton.Yes:
                return False
        project = p.snapshot()
        st = EX.settings_of(project)
        eng = self._engine()
        self._gen += 1
        key = (p, self._spec_key(p))
        self.busy = True
        self.status_text = 'Running…'
        self.bar.setValue(0)

        def work(progress, cancel):
            return key, EX.run(project, eng, st, progress=progress, cancel=cancel)
        ok = self._start(JOB, f'Experiment · {st.name}', work, self._done, '_gen')
        if not ok:
            self.busy = False
            self.status_text = ''
        self.refresh()
        return ok

    def _done(self, gen, out):
        if gen != self._gen:
            return
        self.busy = False
        key, res = out if out is not None else (None, None)
        if res is None or key[0] is not self.ctx.store.project:
            self.status_text = 'Cancelled: the points done stay cached, Run resumes from them' if res is None else ''
            self.refresh()
            return
        self.result, self._for = res, key
        b = res.analysis.get('best')
        self.sel = b['index'] if b else None
        self.status_text = (f'{res.n:,} points' + (f' ({res.cached:,} from the cache)' if res.cached else '') +
                            f' in {_dur(res.seconds)}')
        self.ctx.store.project.add_log('experiments', f'{res.settings.name}: {EX.DESIGN_LABEL[res.design.method]}, '
                                                      f'{res.n} points in {res.seconds:.1f} s')
        self.ctx.msg(f'Experiment finished · {self.status_text}', 5000)
        self._shown = None
        self._fill_pickers(res)
        self.refresh()

    def _failed(self, gen, msg):
        if gen not in (self._gen, self._est_gen):
            return
        self.busy = self.est_busy = False
        self.status_text = 'Failed: ' + msg.splitlines()[0]
        self.refresh()

    def cancel(self):
        win = self.ctx.window
        for job in (JOB, JOB_EST):
            j = getattr(win, '_jobs', {}).get(job) if win is not None else None
            if j is not None:
                j['cancel'].set()
        if win is None or not hasattr(win, 'run_job'):
            self._gen += 1
            self._est_gen += 1
            self.busy = self.est_busy = False
            self.status_text = 'Cancelled: the points done stay cached, Run resumes from them'
            self.refresh()

    # ------------------------------------------------------------------ actions
    def pick(self, i):
        res = self.result
        if res is None or i is None or not 0 <= i < res.n:
            return
        self.sel = int(i)
        self._fill_pareto(res)
        self._select_point_row()

    def add_variant(self, i=None) -> str | None:
        """Add the picked point (or i) as a new variant and switch to it, as one undo step."""
        res = self.result
        i = self.sel if i is None else i
        if res is None or i is None:
            return None
        p = self.ctx.store.project
        try:
            v, note = EX.to_variant(res, i, p)
        except ValueError as e:
            self.ctx.msg(str(e), 5000)
            return None
        state = v.to_dict()
        stale = self.stale                      # before the edit, which makes the new variant the active one
        st = self.settings
        pin = not st.variant                    # an experiment that follows the active variant stays on the run's

        def fn(pj):
            from ...core.project import Variant
            pj.variants.append(Variant.from_dict(state))
            pj.active_uid = state['uid']
            if pin:
                pj.analysis.experiments = dict(st.to_dict(), variant=res.variant_uid)
        self.ctx.store.edit(['variants', 'analysis'] if pin else 'variants', fn, f'Experiments · new variant {v.name}')
        self.ctx.msg(f'Added {v.name}' + (' (the design changed since the run: the point\'s values are applied to the '
                                          'variant as it is now)' if stale else ''), 6000)
        return v.uid

    def export(self, folder=None):
        res = self.result
        if res is None:
            return []
        if folder is None:
            folder = QFileDialog.getExistingDirectory(self, 'Export the experiment into')
            if not folder:
                return []
        from ...core import provenance as PV          # where the JSON came from
        paths = PV.stamp_paths(EX.export(res, self.ctx.store.project, folder), self.ctx.store.project, folder)
        self.ctx.msg(f'Exported the experiment to {folder}', 5000)
        return paths

    def import_spec(self, path=None) -> bool:
        if path is None:
            path, _ = QFileDialog.getOpenFileName(self, 'Import an experiment spec', '', 'Experiment spec (*.json)')
            if not path:
                return False
        try:
            st = EX.load_spec(path, self.ctx.store.project)
        except (OSError, ValueError) as e:
            self.ctx.msg(f'Not an experiment spec: {e}', 5000)
            return False
        self._fsel, self._rsel = '', 0
        return self._store(st, f'imported {st.name}')

    def summary(self) -> dict | None:
        """The last run's compact figures (core.experiments.summary) for the assistant; None before a run."""
        if self.result is None:
            return None
        return dict(EX.summary(self.result), stale=self.stale)

    # ------------------------------------------------------------------ show
    def _fill_pickers(self, res):
        cur = self.r_pick.currentData()
        self.r_pick.clear()
        for r in res.responses:
            self.r_pick.addItem(r.label + (f' ({r.unit})' if r.unit else ''), r.key)
        i = max(0, self.r_pick.findData(cur))
        self.r_pick.setCurrentIndex(i)
        curp = self.p_pick.currentData()
        self.p_pick.clear()
        for r in res.responses:
            self.p_pick.addItem(r.label, r.key)
        j = self.p_pick.findData(curp)
        self.p_pick.setCurrentIndex(j if j >= 0 else (1 if len(res.responses) > 1 else 0))
        for cb, k in ((self.x_pick, 0), (self.y_pick, 1)):
            cur = cb.currentData()
            cb.clear()
            for f, lab in zip(res.factors, res.labels):
                cb.addItem(lab, f.id)
            j = cb.findData(cur)
            cb.setCurrentIndex(j if j >= 0 else min(k, cb.count() - 1))

    def _view_changed(self):
        self._shown = None
        self.refresh()

    def _rk(self):
        res = self.result
        if res is None or not res.responses:
            return None, -1
        key = self.r_pick.currentData() or res.responses[0].key
        j = res.response(key)
        return (key, j) if j >= 0 else (res.responses[0].key, 0)

    def refresh(self):
        win = self.ctx.window
        st = self.settings
        running = self.busy or self.est_busy
        ok = not EX.errors(self.ctx.store.project, st)
        self.b_run.setEnabled(ok and not running)
        self.b_est.setEnabled(ok and not running)
        self.b_cancel.setEnabled(running)
        self.b_export.setEnabled(self.result is not None)
        self.b_variant.setEnabled(self.result is not None and self.sel is not None)
        self.b_best.setEnabled(self.result is not None and bool(self.result.analysis.get('best')))
        self.bar.setVisible(running and (win is None or not hasattr(win, 'run_job')))
        text = self.status_text or ('Ready: Run evaluates the design.' if ok else EMPTY)
        self.status.setText(text + (' · stale: the design or the experiment changed since' if self.stale else ''))
        res = self.result
        if res is self._shown:
            return
        self._shown = res
        for c in (self.index_chart, self.effects, self.heat, self.pareto, self.dist, self.conv):
            c.message = 'Run the experiment' if res is None else 'Not for this design'
        self._fill_tiles(res)
        self._fill_sensitivity(res)
        self._fill_effects(res)
        self._fill_surface(res)
        self._fill_pareto(res)
        self._fill_uncertainty(res)
        self._fill_points(res)
        self._fill_findings(res)

    def _fill_tiles(self, res):
        if res is None:
            for t in self.tiles.values():
                t.set('—')
            return
        self.tiles['points'].set(f'{res.n:,}', '', f'{res.cached:,} from the cache' if res.cached else
                                 EX.DESIGN_LABEL[res.design.method])
        self.tiles['time'].set(_dur(res.seconds), '', f'{res.seconds / max(res.n - res.cached, 1) * 1000:.0f} ms a '
                                                      f'point · {res.workers} workers')
        b = res.analysis.get('best')
        if b is not None and res.responses:
            r = res.responses[0]
            v = b['values'][r.key]
            self.tiles['best'].set(_g(v), r.unit, f'point {b["index"] + 1}' + ('' if b['meets_targets'] else
                                                                                 ' · misses a target'),
                                   None if b['meets_targets'] else 'warning')
        else:
            self.tiles['best'].set('—', '', 'no point with every value')
        key, j = self._rk()
        rk = EX.ranking(res, key) if key else []
        if rk:
            lab = dict(zip([f.id for f in res.factors], res.labels))
            self.tiles['driver'].set(lab[rk[0][0]][:14], '', f'{rk[0][2]} {_g(rk[0][1], 3)}')
        else:
            self.tiles['driver'].set('—')

    def _fill_sensitivity(self, res):
        t = self.index_table
        t.setRowCount(0)
        if res is None:
            self.index_chart.set_data('', [])
            return
        key, j = self._rk()
        a = res.analysis['responses'][key]
        r = res.responses[j]
        m = res.design.method
        heads = {'sobol': ('First order S1', 'S1 interval', 'Total ST', 'ST interval'),
                 'morris': ('μ*', 'μ* interval', 'μ', 'σ'),
                 'oat': ('Swing', 'Lowest', 'Highest', 'Best at')}.get(m, ('Main-effect range', 'Lowest mean',
                                                                     'Highest mean', 'Levels'))
        for c, h in enumerate(heads, 1):
            t.horizontalHeaderItem(c).setText(h)
        rows = []
        t.setRowCount(len(res.factors))
        for jf, (f, lab) in enumerate(zip(res.factors, res.labels)):
            t.setItem(jf, 0, QTableWidgetItem(lab))
            if m == 'sobol':
                s = a['sobol']
                rows.append((lab, s['S1'][jf], s['S1_ci'][jf], s['ST'][jf], s['ST_ci'][jf]))
                vals = (_g(s['S1'][jf], 3), _ci(s['S1_ci'][jf]), _g(s['ST'][jf], 3), _ci(s['ST_ci'][jf]))
            elif m == 'morris':
                mo = a['morris']
                rows.append((lab, mo['mu_star'][jf], mo['sigma'][jf], mo['mu_star_ci'][jf]))
                vals = (_g(mo['mu_star'][jf]), _ci(mo['mu_star_ci'][jf], 4), _g(mo['mu'][jf]), _g(mo['sigma'][jf]))
            elif m == 'oat':
                sw = a['sweeps'][jf]
                rows.append((lab, sw['swing']))
                pairs = [(x, y) for x, y in zip(sw['x'], sw['y']) if y is not None]
                ys = [y for _, y in pairs]
                best_at = (max(pairs, key=lambda t_: t_[1]) if r.maximise else min(pairs, key=lambda t_: t_[1]))[0] \
                    if pairs else None
                vals = (_g(sw['swing']), _g(min(ys)) if ys else '—', _g(max(ys)) if ys else '—',
                        '—' if best_at is None else (best_at if isinstance(best_at, str) else f'{best_at:.4g}'))
            else:
                me = a['main_effects'][jf]
                ms = [x['mean'] for x in me['rows'] if x['mean'] is not None]
                vals = (_g(me['range']), _g(min(ms)) if ms else '—', _g(max(ms)) if ms else '—', str(len(me['rows'])))
            for c, v in enumerate(vals, 1):
                t.setItem(jf, c, _num(v))
        if m in ('sobol', 'morris', 'oat'):
            self.index_chart.set_data(m, rows, r.unit)
        else:
            rk = sorted(((lab, a['main_effects'][jf]['range']) for jf, lab in enumerate(res.labels)),
                        key=lambda t_: -(t_[1] or 0))
            self.index_chart.set_data('range', rk, r.unit)
        if m == 'sobol' and a['sobol'].get('note'):
            self.index_chart.message = a['sobol']['note']

    def _fill_effects(self, res):
        if res is None:
            self.effects.set_data([])
            return
        key, j = self._rk()
        a = res.analysis['responses'][key]
        panels = []
        for f, lab, me in zip(res.factors, res.labels, a['main_effects']):
            pts = [(x['x'], x['level'], x['mean'], x['lo'], x['hi'], x['n']) for x in me['rows']]
            panels.append((lab, f.categorical, pts))
        self.effects.set_data(panels, res.responses[j].unit)

    def _fill_surface(self, res):
        if res is None or len(res.factors) < 2:
            self.heat.set_data(None, [], [], '', True, '', '')
            return
        key, j = self._rk()
        ia = max(0, self.x_pick.currentIndex())
        ib = max(0, self.y_pick.currentIndex())
        if ia == ib:
            self.heat.message = 'Pick two different factors'
            self.heat.set_data(None, [], [], '', True, '', '')
            return
        X = res.X()
        fa, fb = res.factors[ia], res.factors[ib]
        g = EX.binned_grid(X[:, ia], X[:, ib], res.Y[:, j], fa.categorical, fb.categorical)
        M = np.array([[np.nan if v is None else v for v in row] for row in g['mean']], float).reshape(
            len(g['xb']), len(g['xa']))

        def labs(f, xs):
            return [f.choices[int(x)] if f.categorical and 0 <= int(x) < len(f.choices) else f'{x:.3g}' for x in xs]
        r = res.responses[j]
        self.heat.set_data(M, labs(fa, g['xa']), labs(fb, g['xb']), r.unit, r.maximise,
                           res.labels[ia] + (f' ({fa.unit})' if fa.unit else ''),
                           res.labels[ib] + (f' ({fb.unit})' if fb.unit else ''))

    def _fill_pareto(self, res):
        if res is None:
            self.pareto.set_data([], [], [], None, None, '', '')
            self.p_text.setText('')
            self.b_variant.setEnabled(False)
            return
        key, j = self._rk()
        jb = res.response(self.p_pick.currentData()) if self.p_pick.currentData() else -1
        if jb < 0 or jb == j:
            jb = next((k for k in range(len(res.responses)) if k != j), j)
        ra, rb = res.responses[j], res.responses[jb]
        self.pareto.set_data(res.Y[:, j], res.Y[:, jb], res.analysis['pareto'], self.sel,
                             (res.base['y'][j], res.base['y'][jb]),
                             ra.label + (f' ({ra.unit})' if ra.unit else ''),
                             rb.label + (f' ({rb.unit})' if rb.unit else ''))
        self.b_variant.setEnabled(self.sel is not None)
        if self.sel is not None:
            i = self.sel
            figs = ' · '.join(f'{r.label} {_g(res.Y[i, k])}' + (f' {r.unit}' if r.unit else '')
                              for k, r in enumerate(res.responses))
            on = ' (on the Pareto front)' if i in set(res.analysis['pareto']) else ''
            self.p_text.setText(f'Point {i + 1}{on}: {EX.describe_point(res, i)}. {figs}.')
        else:
            self.p_text.setText(f'{len(res.analysis["pareto"])} points on the Pareto front. Click one to pick it.')

    def _fill_uncertainty(self, res):
        t = self.u_table
        t.setRowCount(0)
        if res is None or res.design.method != 'montecarlo':
            self.dist.set_data([], [], '')
            self.conv.set_data([], '')
            return
        key, j = self._rk()
        u = res.analysis['responses'][key].get('uncertainty', {})
        r = res.responses[j]
        self.dist.set_data(res.Y[:, j], [(u.get('p5'), 'p5'), (u.get('p50'), 'p50'), (u.get('p95'), 'p95')], r.unit)
        self.conv.set_data(u.get('convergence'), r.unit)
        rows = [('Points with a value', str(u.get('n', 0)), '')]
        for k_, lab in (('mean', 'Mean'), ('sd', 'Standard deviation'), ('p5', '5th percentile'),
                        ('p50', 'Median'), ('p95', '95th percentile')):
            if u.get(k_) is not None:
                rows.append((lab, f'{_g(u[k_])} {r.unit}'.strip(), _ci(u.get(f'{k_}_ci'), 4)))
        if u.get('min') is not None:
            rows.append(('Range', f'{_g(u["min"])} … {_g(u["max"])} {r.unit}'.strip(), ''))
        if u.get('meets') is not None:
            rows.append((f'Meeting {EX.RQ.OP_TEXT[r.op]} {r.target:g} {r.unit}'.strip(),
                         f'{100 * u["meets"]:.1f} %', _ci([100 * c for c in u['meets_ci']], 1) + ' %'))
        b = res.base['y'][j]
        if np.isfinite(b):
            rows.append(('The design itself', f'{_g(b)} {r.unit}'.strip(), ''))
        t.setRowCount(len(rows))
        for i, (a_, b_, c_) in enumerate(rows):
            t.setItem(i, 0, QTableWidgetItem(a_))
            t.setItem(i, 1, _num(b_))
            t.setItem(i, 2, _num(c_))

    def _fill_points(self, res):
        t = self.points
        t.setRowCount(0)
        if res is None:
            self.pt_note.setText('')
            return
        key, j = self._rk()
        r = res.responses[j]
        y = res.Y[:, j]
        order = np.argsort(np.where(np.isfinite(y), -y if r.maximise else y, np.inf), kind='stable')[:MAX_ROWS]
        cols = ['#', 'Block'] + [f'{lab}' + (f' ({f.unit})' if f.unit else '') for f, lab in
                                 zip(res.factors, res.labels)] + \
               [x.label[:28] + (f' ({x.unit})' if x.unit else '') for x in res.responses]
        t.setColumnCount(len(cols))
        t.setHorizontalHeaderLabels(cols)
        hh = t.horizontalHeader()
        for c in range(len(cols)):
            hh.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        t.setRowCount(len(order))
        front = set(res.analysis['pareto'])
        self._row_of = {}
        for row, i in enumerate(order):
            i = int(i)
            self._row_of[i] = row
            it = _num(str(i + 1), 'on the Pareto front' if i in front else None,
                      _accent() if i in front else None)
            it.setData(Qt.ItemDataRole.UserRole, i)
            t.setItem(row, 0, it)
            t.setItem(row, 1, QTableWidgetItem(res.design.block(i)))
            for c, (f, v) in enumerate(zip(res.factors, res.values[i]), 2):
                t.setItem(row, c, _num(v if isinstance(v, str) else _g(v, 5)))
            for c, rr in enumerate(res.responses, 2 + len(res.factors)):
                yv = res.Y[i, c - 2 - len(res.factors)]
                why = res.reasons.get((i, c - 2 - len(res.factors)))
                t.setItem(row, c, _num(_g(yv, 5), why))
        self.pt_note.setText(f'Sorted by {r.label.lower()}, best first' + (
            f'; the best {MAX_ROWS:,} of {res.n:,} (the export has every point)' if res.n > MAX_ROWS else '') +
            '. Highlighted: on the Pareto front. Pick a row to add it as a variant.')
        self._select_point_row()

    def _select_point_row(self):
        row = getattr(self, '_row_of', {}).get(self.sel)
        if row is not None and self.points.currentRow() != row:
            self._syncing = True
            try:
                self.points.selectRow(row)
            finally:
                self._syncing = False

    def _point_row(self):
        if self._syncing:
            return
        it = self.points.item(self.points.currentRow(), 0)
        if it is not None:
            i = it.data(Qt.ItemDataRole.UserRole)
            if i is not None and i != self.sel:
                self.sel = int(i)
                self._fill_pareto(self.result)

    def _fill_findings(self, res):
        for w in self.findings:
            w.hide()
            self.find_lay.removeWidget(w)
            w.deleteLater()
        self.findings = []
        lines = []
        if res is not None:
            for w_ in res.warnings:
                lines.append(('warning', w_))
            for r in res.responses:
                a = res.analysis['responses'][r.key]
                for ne in a.get('not_evaluated', []):
                    lines.append(('warning', f'{r.label}: no value at {ne["points"]} of {res.n} points — '
                                             f'{ne["reason"]}'))
                if 'sobol' in a and a['sobol'].get('note'):
                    lines.append(('warning', f'{r.label}: {a["sobol"]["note"]}'))
                if 'sobol' in a and a['sobol']['S1'][0] is not None:
                    wide = [lab for lab, ci in zip(res.labels, a['sobol']['ST_ci']) if ci[0] is not None and
                            ci[1] - ci[0] > 0.2]
                    if wide:
                        lines.append(('warning', f'{r.label}: the total-index intervals of {", ".join(wide)} are '
                                                 f'wider than 0.2: more base samples settle them.'))
            if res.analysis.get('feasible') == 0:
                lines.append(('serious', 'No point meets every response\'s target: the best point is the Pareto '
                                         'front\'s best on the first response.'))
            lines.append(('good', f'Seed {res.settings.seed} · inputs {res.base_key} · Vantage {res.version}: the same '
                                  'spec on the same project gives the same points.'))
        lines.append(('', EX.ASSUMPTIONS))
        lines.append(('', EX.LIMITS_TEXT))
        for sev, text in lines:
            w = StatusLine(sev, text)
            self.find_lay.insertWidget(self.find_lay.count() - 1, w)
            self.findings.append(w)


def _dur(s) -> str:
    return EX._dur(s)
