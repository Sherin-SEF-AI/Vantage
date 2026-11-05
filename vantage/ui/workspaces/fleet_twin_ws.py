"""Calibrate ▸ Fleet twin: every vehicle's sensors followed over time from what the fleet already produces (the Fleet
page's records, online self-calibrations of recorded drives, rig-health drift), each axis's drift modelled with its
uncertainty, the date each sensor will leave its limit forecast, knocks flagged at once, the measured drift handed back
to the design's tolerances. A fleet table coloured by urgency, each sensor's history with the smoothed band, its knocks,
the forecast cone and its limits, the fleet's drift-rate distributions, watch folders per vehicle, the recalibration
schedule. The analysis runs in the background (a few seconds for a big fleet); every edit is one undo step
(project.analysis.fleet_twin)."""
from __future__ import annotations

import os
import threading
import traceback

import numpy as np
from PyQt6.QtCore import QObject, QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen, QPolygonF
from PyQt6.QtWidgets import (QAbstractItemView, QComboBox, QFileDialog, QGridLayout, QHBoxLayout, QHeaderView,
                             QLineEdit, QScrollArea, QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from ...core import fleet as FL
from ...core import fleet_twin as FT
from ..charts import ChartBase
from ..theme import STATUS, THEME, mono, sans
from ..widgets import ROW_H, ScrubSpinBox, StatusLine, ToggleSwitch, button, fit_width, hline, label

JOB = 'fleet_twin'
DRIVE_JOB = 'fleet_twin_drive'
WATCH_STATE_TEXT = {'arriving': 'arriving', 'quiet': 'waiting until it is quiet', 'incomplete': 'not finished',
                    'ready': 'ready'}
URG_COLOR = {'beyond': STATUS['critical'], 'due': STATUS['serious'], 'soon': STATUS['warning'], 'later': '#8fbf3a',
             'ok': STATUS['good'], 'unknown': '#6e7f96'}
SRC_COLOR = {'install_check': '#3987e5', 'extrinsics': '#199e70', 'nuscenes': '#199e70', 'kalibr': '#c98500',
             'variant': '#9085e9', 'manual': '#9085e9', 'online_calib': '#d55181'}
SEV_TONE = {'critical': 'critical', 'serious': 'serious', 'warning': 'warning', 'good': 'good'}
# (key, label, unit, decimals, step, tooltip)
FIELDS = (
    ('horizon_days', 'Horizon', 'days', 0, 30.0, 'Forecast this many days after the as-of date'),
    ('risk', 'Risk', '%', 1, 1.0, 'Recalibrate when the chance of being beyond the limit reaches this'),
    ('false_alarm', 'False alarms', 'per 10k', 2, 0.5,
     'A knock alarm on a step-free axis at most this often per observation (per ten thousand)'),
    ('glr_window', 'Knock window', 'obs', 0, 1.0, 'A knock is looked for among the last this many observations'),
    ('min_step', 'Smallest knock', '× limit', 2, 0.05,
     'A step smaller than this share of the limit is left to the drift'),
    ('outlier_z', 'Outlier', 'σ', 1, 0.5, 'A vehicle whose drift rate is this many σ off the others is named'),
    ('tol_horizon_days', 'Tolerance after', 'days', 0, 30.0,
     'The tolerance inputs give the pose spread of a new vehicle this many days after it was built'),
    ('user_mm', 'My limit, position', 'mm', 1, 1.0, 'Used with Limits: My limits'),
    ('user_deg', 'My limit, angle', '°', 2, 0.05, 'Used with Limits: My limits'),
)
SCALE = {'risk': 100.0, 'false_alarm': 1e4}             # shown as % and per ten thousand
SCHED_COLS = (('Recalibrate by', ''), ('Vehicle', ''), ('Sensor', ''), ('Urgency', ''),
              ('P now', 'Chance it is beyond its limit already'),
              ('Median', 'The date by which it is beyond with probability 0.5'),
              ('Axis', 'The axis nearest its limit'), ('Limit', 'Where the limit comes from'))


def _tbl(cols, select=True):
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
    t.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection if select else
                       QAbstractItemView.SelectionMode.NoSelection)
    hh = t.horizontalHeader()
    for c in range(len(cols)):
        hh.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
    hh.setSectionResizeMode(len(cols) - 1, QHeaderView.ResizeMode.Stretch)
    return t


def _it(text, num=False, tip=None, color=None, fill=None, data=None):
    it = QTableWidgetItem(text)
    it.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
    if num:
        it.setFont(mono(12))
        it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    if tip:
        it.setToolTip(tip)
    if color is not None:
        it.setForeground(color)
    if fill is not None:
        it.setBackground(fill)
    if data is not None:
        it.setData(Qt.ItemDataRole.UserRole, data)
    return it


def cell_tip(c, res) -> str:
    """The hover text of a fleet-table cell."""
    out = [f'{c["name"]} · {c["tag"]} · {FT.URGENCY_LABEL[c["urgency"]]}',
           f'{c["observations"]} observations, {FT.date_text(c["first"])} … {FT.date_text(c["last"])}']
    for ax, sd in c['axes'].items():
        u = sd['unit']
        d = 1 if u == 'mm' else 3
        out.append(f'{ax}: {sd["now"][0]:+.{d}f} ± {sd["now"][1]:.{d}f} {u} now, {sd["rate_now"][0] * 30:+.{d + 1}f} '
                   f'± {sd["rate_now"][1] * 30:.{d + 1}f} per 30 days; limit {-sd["lo"]:.3g} … {sd["hi"]:.3g} '
                   f'({c["limit_source"][ax]})')
    out.append(f'P(beyond) now {c["p_now"]:.3f}, in {res.settings.horizon_days:.0f} days {c["p_horizon"]:.3f}')
    if c['date_risk'] is not None:
        out.append(f'Recalibrate by {FT.date_text(c["date_risk"])} ({res.settings.risk:.0%} risk)'
                   + (f'; median {FT.date_text(c["date_50"])}' if c['date_50'] else '')
                   + (f'; 90 % {FT.date_text(c["date_90"])}' if c['date_90'] else ''))
    for k in c['knocks']:
        out.append(f'Knock between {k["after"]} and {k["date"]}: ' +
                   ', '.join(f'{a} {v:+.3g}' for a, v in k['size'].items()))
    return '\n'.join(out)


class SeriesChart(ChartBase):
    """One vehicle's sensor, one axis, over time: the observations with their σ (coloured by source), the smoothed
    level with its 2 σ band, the knocks, the forecast with its 2 σ cone and the limits; below, the probability of
    being beyond the limit by each date (this axis, and the sensor with every axis)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.title = 'Sensor over time'
        self.res = None
        self.cell = None
        self.axis = ''
        self._dots = []
        self.setMinimumHeight(300)

    def set_data(self, res, cell, axis):
        self.res, self.cell, self.axis = res, cell, axis
        self.title = f'{cell["name"]} · {cell["tag"]} {axis}' if cell else 'Sensor over time'
        self.tip = None
        self.update()

    def mouseMoveEvent(self, e):
        pos = e.position()
        hit = min(self._dots, key=lambda d: (d[0].x() - pos.x()) ** 2 + (d[0].y() - pos.y()) ** 2, default=None)
        if hit is not None and abs(hit[0].x() - pos.x()) < 10 and abs(hit[0].y() - pos.y()) < 30:
            self.set_tip(hit[0], hit[1], *hit[2])
        else:
            self.tip = None
        self.update()

    def leaveEvent(self, e):
        self.tip = None
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._dots = []
        c = self.cell
        sd = c['axes'].get(self.axis) if c else None
        self.paint_title(p, (FT.URGENCY_LABEL[c['urgency']] +
                             (f' · recalibrate by {FT.date_text(c["date_risk"])}' if c['date_risk'] else ''))
                         if c else '')
        if sd is None:
            p.setFont(sans(11))
            p.setPen(THEME.color('muted'))
            p.drawText(QRectF(0, 24, self.width(), 30), Qt.AlignmentFlag.AlignVCenter,
                       'Click a cell of the fleet table to see that sensor over time.')
            p.end()
            return
        res = self.res
        u = sd['unit']
        t = np.asarray(sd['t'])
        ft = np.asarray(sd['forecast']['t'])
        fm, fs = np.asarray(sd['forecast']['mean']), np.asarray(sd['forecast']['sd'])
        lvl, lsd = np.asarray(sd['level']), np.asarray(sd['level_sd'])
        y, s = np.asarray(sd['y']), np.asarray(sd['sigma'])
        t0, t1 = float(t[0]), float(max(ft[-1], t[-1] + 1.0))
        vals = np.r_[y + s, y - s, fm + 2 * fs, fm - 2 * fs, -sd['lo'], sd['hi']]
        vlo, vhi = float(vals.min()), float(vals.max())
        pad = 0.06 * max(vhi - vlo, 1e-9)
        vlo, vhi = vlo - pad, vhi + pad
        left, right, top = 54.0, 10.0, 26.0
        W = self.width() - left - right
        H1 = (self.height() - top - 40) * 0.68
        y2 = top + H1 + 26
        H2 = self.height() - y2 - 18
        X = lambda q: left + (q - t0) / max(t1 - t0, 1e-9) * W                   # noqa: E731
        Y = lambda q: top + H1 - (q - vlo) / (vhi - vlo) * H1                     # noqa: E731
        p.setPen(QPen(THEME.color('hair'), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(QRectF(left, top, W, H1))
        # the band, the cone, the limits
        band = QColor(THEME.color('ink2'))
        band.setAlphaF(0.14)
        poly = QPolygonF([QPointF(X(a), Y(b)) for a, b in zip(t, lvl + 2 * lsd)] +
                         [QPointF(X(a), Y(b)) for a, b in zip(t[::-1], (lvl - 2 * lsd)[::-1])])
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(band)
        p.drawPolygon(poly)
        cone = QColor('#9085e9')
        cone.setAlphaF(0.20)
        p.setBrush(cone)
        p.drawPolygon(QPolygonF([QPointF(X(a), Y(b)) for a, b in zip(ft, fm + 2 * fs)] +
                                [QPointF(X(a), Y(b)) for a, b in zip(ft[::-1], (fm - 2 * fs)[::-1])]))
        p.setPen(QPen(QColor(STATUS['critical']), 1, Qt.PenStyle.DashLine))
        for v in (-sd['lo'], sd['hi']):
            p.drawLine(QPointF(left, Y(v)), QPointF(left + W, Y(v)))
        p.setPen(QPen(THEME.color('strong'), 1))
        if vlo < 0 < vhi:
            p.drawLine(QPointF(left, Y(0)), QPointF(left + W, Y(0)))
        xa = X(res.as_of)
        if left <= xa <= left + W:
            p.setPen(QPen(THEME.color('muted'), 1, Qt.PenStyle.DotLine))
            p.drawLine(QPointF(xa, top), QPointF(xa, top + H1 + 26 + H2))
        # the smoothed level and the forecast
        p.setPen(QPen(QColor('#539bf5'), 2))
        for a0, b0, a1, b1 in zip(t, lvl, t[1:], lvl[1:]):
            p.drawLine(QPointF(X(a0), Y(b0)), QPointF(X(a1), Y(b1)))
        p.setPen(QPen(QColor('#9085e9'), 1.4, Qt.PenStyle.DashLine))
        for a0, b0, a1, b1 in zip(ft, fm, ft[1:], fm[1:]):
            p.drawLine(QPointF(X(a0), Y(b0)), QPointF(X(a1), Y(b1)))
        # knocks
        p.setFont(sans(10))
        for st_ in sd['steps']:
            xs = X(0.5 * (st_['after_days'] + st_['days']))
            p.setPen(QPen(QColor(STATUS['serious']), 1.2))
            p.drawLine(QPointF(xs, top), QPointF(xs, top + H1))
            p.drawText(QPointF(xs + 4, top + 12), f'knock {st_["size"]:+.3g} {u}')
        # observations
        for i, (a, b, e_) in enumerate(zip(t, y, s)):
            col = QColor(SRC_COLOR.get(sd['kinds'][i], '#888888'))
            p.setPen(QPen(col, 1))
            p.drawLine(QPointF(X(a), Y(b - e_)), QPointF(X(a), Y(b + e_)))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(col)
            pt = QPointF(X(a), Y(b))
            p.drawEllipse(pt, 3, 3)
            self._dots.append((pt, f'{b:+.4g} ± {e_:.2g} {u}',
                               [sd['dates'][i][:19], f'{FT.SOURCE_LABEL.get(sd["kinds"][i], sd["kinds"][i])}: '
                                                     f'{sd["sources"][i]}'[:90],
                                f'smoothed {sd["level"][i]:+.4g} ± {sd["level_sd"][i]:.2g}']))
        # ticks
        p.setFont(mono(10))
        p.setPen(THEME.color('muted'))
        for v in (vlo + pad, 0.5 * (vlo + vhi), vhi - pad):
            p.drawText(QRectF(0, Y(v) - 7, left - 5, 14), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                       f'{v:+.3g}')
        p.setFont(sans(10))
        p.drawText(QRectF(left, top + H1 + 2, 120, 14), Qt.AlignmentFlag.AlignLeft, FT.date_text(t0))
        p.drawText(QRectF(left + W - 120, top + H1 + 2, 120, 14), Qt.AlignmentFlag.AlignRight, FT.date_text(t1))
        p.drawText(QRectF(left + 130, top + H1 + 2, W - 260, 14), Qt.AlignmentFlag.AlignHCenter,
                   f'{u}: dots measured (1 σ), blue smoothed (2 σ), violet forecast (2 σ), red limits')
        # the probability panel
        p.setPen(QPen(THEME.color('hair'), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(QRectF(left, y2, W, H2))
        Yp = lambda q: y2 + H2 - q * H2                                           # noqa: E731
        pt_ = np.asarray(sd['p_exceed']['t'])
        pp = np.asarray(sd['p_exceed']['p'])
        p.setPen(QPen(QColor('#9085e9'), 1.2))
        for a0, b0, a1, b1 in zip(pt_, pp, pt_[1:], pp[1:]):
            p.drawLine(QPointF(X(a0), Yp(b0)), QPointF(X(a1), Yp(b1)))
        cd = np.asarray(c['p_curve']['days']) + res.as_of
        cp = np.asarray(c['p_curve']['p'])
        p.setPen(QPen(QColor(URG_COLOR[c['urgency']]), 2))
        for a0, b0, a1, b1 in zip(cd, cp, cd[1:], cp[1:]):
            p.drawLine(QPointF(X(a0), Yp(b0)), QPointF(X(a1), Yp(b1)))
        p.setPen(QPen(THEME.color('muted'), 1, Qt.PenStyle.DashLine))
        p.drawLine(QPointF(left, Yp(res.settings.risk)), QPointF(left + W, Yp(res.settings.risk)))
        p.setFont(mono(10))
        p.setPen(THEME.color('muted'))
        for v in (0.0, 1.0):
            p.drawText(QRectF(0, Yp(v) - 7, left - 5, 14), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                       f'{v:.0f}')
        p.setFont(sans(10))
        p.drawText(QPointF(left + 4, y2 - 4), f'P(beyond the limit by the date): {self.axis} (violet), the sensor '
                                              f'(all axes, colour of its urgency); dashed: {res.settings.risk:.0%}')
        self.paint_tip(p)
        p.end()


class RatesChart(ChartBase):
    """One sensor position and axis across the fleet: each vehicle's drift rate now (± 2 σ) and the fitted rate prior
    (mean ± 2 τ); outliers marked."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.title = 'Drift rates across the fleet'
        self.row = None
        self.names = {}
        self.setMinimumHeight(150)

    def set_data(self, row, names):
        self.row, self.names = row, names
        self.title = f'{row["tag"]} {row["axis"]}: drift per 30 days across the fleet' if row else \
            'Drift rates across the fleet'
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = self.row
        right = ''
        if r is not None:
            right = (f'prior {r["prior_rate_per_30d"]:+.3g} ± {r["prior_rate_sd_per_30d"]:.3g} {r["unit"]} '
                     f'({r["pooled"]})' if r['prior_rate_per_30d'] is not None else 'no fleet prior (too few vehicles)')
        self.paint_title(p, right)
        if r is None:
            p.setFont(sans(11))
            p.setPen(THEME.color('muted'))
            p.drawText(QRectF(0, 24, self.width(), 30), Qt.AlignmentFlag.AlignVCenter, 'No sensor selected.')
            p.end()
            return
        rates = np.asarray(r['rates_per_30d'])
        sds = np.asarray(r['rates_sd_per_30d'])
        vals = np.r_[rates + 2 * sds, rates - 2 * sds, 0.0]
        if r['prior_rate_per_30d'] is not None:
            vals = np.r_[vals, r['prior_rate_per_30d'] + 2 * r['prior_rate_sd_per_30d'],
                         r['prior_rate_per_30d'] - 2 * r['prior_rate_sd_per_30d']]
        lo, hi = float(vals.min()), float(vals.max())
        pad = 0.08 * max(hi - lo, 1e-12)
        lo, hi = lo - pad, hi + pad
        left, right_, top, bottom = 110.0, 14.0, 26.0, 20.0
        W = self.width() - left - right_
        X = lambda v: left + (v - lo) / (hi - lo) * W                             # noqa: E731
        n = len(rates)
        rh = max(min((self.height() - top - bottom) / max(n, 1), 18.0), 6.0)
        if r['prior_rate_per_30d'] is not None:
            band = QColor(THEME.color('ink2'))
            band.setAlphaF(0.12)
            m, s = r['prior_rate_per_30d'], r['prior_rate_sd_per_30d']
            p.fillRect(QRectF(X(m - 2 * s), top, X(m + 2 * s) - X(m - 2 * s), rh * n), band)
        p.setPen(QPen(THEME.color('strong'), 1))
        p.drawLine(QPointF(X(0), top), QPointF(X(0), top + rh * n))
        out = {o['vehicle'] for o in r['outliers']}
        p.setFont(sans(10))
        for i, (vid, a, s) in enumerate(zip(r['vehicle_ids'], rates, sds)):
            yy = top + rh * (i + 0.5)
            col = QColor(STATUS['serious'] if vid in out else '#539bf5')
            p.setPen(QPen(col, 1.4))
            p.drawLine(QPointF(X(a - 2 * s), yy), QPointF(X(a + 2 * s), yy))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(col)
            p.drawEllipse(QPointF(X(a), yy), 3, 3)
            p.setPen(THEME.color('ink2'))
            p.drawText(QRectF(0, yy - 7, left - 8, 14), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                       self.names.get(vid, vid)[:16])
        p.setFont(mono(10))
        p.setPen(THEME.color('muted'))
        yb = top + rh * n + 2
        for v in (lo + pad, 0.0, hi - pad):
            p.drawText(QRectF(X(v) - 40, yb, 80, 14), Qt.AlignmentFlag.AlignHCenter, f'{v:+.3g}')
        p.end()


class _Sig(QObject):
    done = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)
    drive_done = pyqtSignal(object)
    drive_failed = pyqtSignal(str)
    health_done = pyqtSignal(object)


class FleetTwinPage(QWidget):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.result: FT.TwinResult | None = None
        self._for = None                        # the inputs key the result is for
        self._gen = 0
        self.busy = False
        self.reading = None                     # (vehicle id, path, key) of the drive being calibrated
        self.health_busy = False                # the live streams are being read
        self.status_text = ''
        self._dirty = True
        self._shown = None
        self.sel = None                         # (vehicle id, tag)
        self.sel_axis = ''
        self.sel_vehicle = ''                   # the vehicle imports and the watch apply to
        self.watch_state = FT.WatchState()
        self.sig = _Sig()
        self.sig.done.connect(self._done)
        self.sig.failed.connect(self._failed)
        self.sig.drive_done.connect(self._drive_done)
        self.sig.drive_failed.connect(self._drive_failed)
        self.sig.health_done.connect(self._health_done)
        self._auto = QTimer(self)
        self._auto.setSingleShot(True)
        self._auto.setInterval(500)
        self._auto.timeout.connect(self._auto_run)
        self.poll_timer = QTimer(self)
        self.poll_timer.setInterval(2000)
        self.poll_timer.timeout.connect(self.poll_watch)
        self.setObjectName('base')
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setHandleWidth(2)
        split.addWidget(self._left())
        split.addWidget(self._right())
        split.setSizes([380, 1220])
        lay.addWidget(split)
        ctx.store.changed.connect(self._on_change)
        ctx.store.project_replaced.connect(self._on_replaced)
        THEME.changed.connect(self._retheme)
        self._sync_fields()
        self.refresh()

    # ------------------------------------------------------------------ left: settings, imports, watch
    def _left(self):
        w = QWidget()
        w.setObjectName('panel')
        ll = QVBoxLayout(w)
        ll.setContentsMargins(8, 6, 8, 8)
        ll.setSpacing(4)
        top = QHBoxLayout()
        top.addWidget(label('Fleet twin'))
        top.addStretch(1)
        top.addWidget(label('drift, forecasts and knocks', muted=True, small=True))
        ll.addLayout(top)
        ll.addWidget(hline())
        ll.addWidget(label('Every vehicle\'s sensors over time, from the Fleet page\'s records, online '
                           'self-calibrations of recorded drives and rig-health drift: each axis\'s drift with its '
                           'uncertainty, the date each sensor will leave its limit, knocks flagged at once, and the '
                           'measured drift as inputs to the design\'s tolerances.', muted=True, small=True, wrap=True))
        ll.addSpacing(4)
        vr = QHBoxLayout()
        a = label('Plan', ink2=True)
        a.setFixedWidth(70)
        vr.addWidget(a)
        self.variant = QComboBox()
        self.variant.setToolTip('The variant the sensors are followed against')
        self.variant.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.variant.setMinimumContentsLength(14)
        self.variant.activated.connect(self._variant_picked)
        vr.addWidget(self.variant, 1)
        ll.addLayout(vr)
        lr = QHBoxLayout()
        a = label('Limits', ink2=True)
        a.setFixedWidth(70)
        lr.addWidget(a)
        self.limit = QComboBox()
        for k in FT.LIMIT_SOURCES:
            self.limit.addItem(FT.LIMIT_LABEL[k], k)
        self.limit.setToolTip('Automatic: the critical targets\' inverse tolerances where a target sets the axis, else '
                              'the mounting tolerance (Analysis); or one of them, the Fleet page\'s tolerance, or '
                              'your own figures')
        self.limit.activated.connect(lambda i: self._set(f'limits: {self.limit.itemText(i).lower()}',
                                                         limit_source=self.limit.itemData(i)))
        lr.addWidget(self.limit, 1)
        ll.addLayout(lr)
        pr = QHBoxLayout()
        a = label('Rate prior', ink2=True)
        a.setFixedWidth(70)
        pr.addWidget(a)
        self.pool = QComboBox()
        self.pool.addItem('Per sensor position', 'sensor')
        self.pool.addItem('Per sensor kind', 'kind')
        self.pool.setToolTip('Which vehicles a sensor\'s drift rate borrows strength from: the same position on the '
                             'other vehicles, or every sensor of its kind')
        self.pool.activated.connect(lambda i: self._set(f'rate prior {self.pool.itemText(i).lower()}',
                                                        pool=self.pool.itemData(i)))
        pr.addWidget(self.pool, 1)
        ll.addLayout(pr)
        ar = QHBoxLayout()
        a = label('As of', ink2=True)
        a.setFixedWidth(70)
        ar.addWidget(a)
        self.as_of = QLineEdit()
        self.as_of.setPlaceholderText('today')
        self.as_of.setToolTip('The date "now" for the forecast and the schedule (YYYY-MM-DD; empty: today)')
        self.as_of.editingFinished.connect(self._as_of_edited)
        ar.addWidget(self.as_of, 1)
        ll.addLayout(ar)
        g = QGridLayout()
        g.setVerticalSpacing(2)
        g.setHorizontalSpacing(6)
        g.setColumnStretch(2, 1)
        self.fields = {}
        for i, (key, text, unit, dec, step, tip) in enumerate(FIELDS):
            f = ScrubSpinBox('', 'plain', dec, unit, step=step, width=110)
            f.setToolTip(tip)
            f.committed.connect(lambda v, k=key, t=text, u=unit: self._set(f'{t.lower()} {v:g} {u}',
                                                                           **{k: v / SCALE.get(k, 1.0)}))
            a = label(text, ink2=True)
            a.setFixedWidth(118)
            a.setToolTip(tip)
            g.addWidget(a, i, 0)
            g.addWidget(f, i, 1)
            self.fields[key] = f
        ll.addLayout(g)
        ll.addSpacing(6)
        ll.addWidget(label('Vehicle', bold=True))
        vh = QHBoxLayout()
        self.vehicle = QComboBox()
        self.vehicle.setToolTip('The vehicle imports and the watch folder apply to (the Fleet page\'s vehicles)')
        self.vehicle.activated.connect(self._vehicle_picked)
        vh.addWidget(self.vehicle, 1)
        ll.addLayout(vh)
        ir = QHBoxLayout()
        self.b_import = button('Import calibration…', None, 'import',
                               'An online self-calibration\'s JSON export: its segments become observations of the '
                               'vehicle (dated by the date below, else the drive\'s own clock)')
        self.b_import.clicked.connect(lambda: self.import_online())
        ir.addWidget(self.b_import)
        self.b_drive = button('Calibrate drive…', None, 'folder',
                              'A recorded drive (nuScenes folder or ROS 2 bag): calibrated in the background, its '
                              'segments added to the vehicle')
        self.b_drive.clicked.connect(lambda: self.calibrate_drive())
        ir.addWidget(self.b_drive)
        ir.addStretch(1)
        ll.addLayout(ir)
        dr = QHBoxLayout()
        a = label('Drive date', ink2=True)
        a.setFixedWidth(70)
        dr.addWidget(a)
        self.drive_date = QLineEdit()
        self.drive_date.setPlaceholderText('from the drive')
        self.drive_date.setToolTip('The date the next imported or calibrated drive was recorded (YYYY-MM-DD); empty: '
                                   'the drive\'s own clock, else its files\' time')
        dr.addWidget(self.drive_date, 1)
        ll.addLayout(dr)
        ll.addSpacing(4)
        ll.addWidget(label('Watch folder', bold=True))
        self.watch_label = label('No watch folder', mono_=True, small=True)
        self.watch_label.setMinimumWidth(10)
        ll.addWidget(self.watch_label)
        wr = QHBoxLayout()
        self.b_watch = button('Set folder…', None, 'folder', 'New drives arriving in this folder are calibrated in the '
                                                             'background and added to the vehicle, once each')
        self.b_watch.clicked.connect(lambda: self.set_watch_folder())
        wr.addWidget(self.b_watch)
        self.watch_on = ToggleSwitch('Watching')
        self.watch_on.setToolTip('Read new drives as they arrive')
        self.watch_on.toggled.connect(self._watch_toggled)
        wr.addWidget(self.watch_on)
        self.b_unwatch = button('Remove', 'ghost', None, 'Stop watching this vehicle\'s folder')
        self.b_unwatch.clicked.connect(lambda: self.set_watch_folder(''))
        wr.addWidget(self.b_unwatch)
        wr.addStretch(1)
        ll.addLayout(wr)
        self.watch_status = label('', muted=True, small=True, wrap=True)
        ll.addWidget(self.watch_status)
        hr = QHBoxLayout()
        self.b_health = button('Read rig health', None, 'sync',
                               'Watch the live camera streams of this vehicle (the Rig health page\'s sources) for a '
                               'few seconds: a body edge off the plan becomes a drift event on the timeline')
        self.b_health.clicked.connect(lambda: self.read_rig_health())
        hr.addWidget(self.b_health)
        hr.addStretch(1)
        ll.addLayout(hr)
        ll.addSpacing(4)
        self.status = label('', muted=True, small=True, wrap=True)
        ll.addWidget(self.status)
        bl = QHBoxLayout()
        b_def = button('Defaults', 'ghost', None, 'Back to the default settings')
        b_def.clicked.connect(self._defaults)
        bl.addWidget(b_def)
        bl.addStretch(1)
        self.b_cancel = button('Cancel', None, None, 'Stop the analysis that is running')
        self.b_cancel.clicked.connect(self.cancel)
        bl.addWidget(self.b_cancel)
        self.b_run = button('Run', 'primary', None, 'Fit the fleet, find the knocks and forecast every sensor')
        self.b_run.clicked.connect(lambda: self.run())
        bl.addWidget(self.b_run)
        ll.addLayout(bl)
        b2 = QHBoxLayout()
        self.b_export = button('Export…', None, 'export',
                               'fleet_twin.json (everything, with σ), fleet_twin.csv (the table), '
                               'fleet_twin_schedule.csv, fleet_twin.png and fleet_twin_tolerances.json (the measured '
                               'drift as tolerance inputs)')
        self.b_export.clicked.connect(lambda: self.export())
        b2.addWidget(self.b_export)
        self.b_sched = button('Schedule CSV…', None, 'export', 'The recalibration schedule: which vehicle and sensor '
                                                               'next, by date')
        self.b_sched.clicked.connect(lambda: self.export_schedule())
        b2.addWidget(self.b_sched)
        b2.addStretch(1)
        ll.addLayout(b2)
        self.b_tol = button('Use as mounting tolerance', None, 'tolerance',
                            'Set the project\'s mounting tolerance (Analysis: the tolerance studies) to the measured '
                            'fleet\'s spread after "Tolerance after" days: ± 2 σ, the largest over the cameras '
                            '(one undo step)')
        self.b_tol.clicked.connect(lambda: self.use_tolerance())
        ll.addWidget(self.b_tol)
        ll.addWidget(label(FT.ASSUMPTIONS, muted=True, small=True, wrap=True))
        ll.addStretch(1)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setWidget(w)
        fit_width(sc)
        sc.setMinimumWidth(340)
        return sc

    # ------------------------------------------------------------------ right: table, charts, schedule, findings
    def _right(self):
        w = QWidget()
        w.setObjectName('panel')
        rl = QVBoxLayout(w)
        rl.setContentsMargins(10, 8, 10, 8)
        rl.setSpacing(6)
        self.headline = label('', bold=True)
        rl.addWidget(self.headline)
        split = QSplitter(Qt.Orientation.Vertical)
        self.table = QTableWidget(0, 0)
        self.table.verticalHeader().setDefaultSectionSize(ROW_H)
        self.table.horizontalHeader().setFixedHeight(ROW_H)
        self.table.setShowGrid(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)
        self.table.cellClicked.connect(self._cell_clicked)
        split.addWidget(self.table)
        mid = QWidget()
        ml = QVBoxLayout(mid)
        ml.setContentsMargins(0, 0, 0, 0)
        ar = QHBoxLayout()
        ar.addWidget(label('Axis', ink2=True))
        self.axis_box = QComboBox()
        self.axis_box.activated.connect(self._axis_picked)
        ar.addWidget(self.axis_box)
        ar.addStretch(1)
        ml.addLayout(ar)
        self.chart = SeriesChart()
        ml.addWidget(self.chart, 3)
        self.rates = RatesChart()
        ml.addWidget(self.rates, 1)
        split.addWidget(mid)
        bottom = QWidget()
        bl = QVBoxLayout(bottom)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.addWidget(label('Recalibration schedule', bold=True))
        self.sched = _tbl(SCHED_COLS)
        self.sched.cellClicked.connect(self._sched_clicked)
        bl.addWidget(self.sched, 1)
        bl.addWidget(label('Findings', bold=True))
        fw = QWidget()
        self.find_lay = QVBoxLayout(fw)
        self.find_lay.setContentsMargins(0, 0, 0, 0)
        self.find_lay.addStretch(1)
        fs = QScrollArea()
        fs.setWidgetResizable(True)
        fs.setWidget(fw)
        bl.addWidget(fs, 1)
        split.addWidget(bottom)
        split.setSizes([260, 520, 300])
        rl.addWidget(split, 1)
        self.findings = []
        return w

    # ------------------------------------------------------------------ settings
    @property
    def settings(self) -> FT.FleetTwinSettings:
        return FT.settings_of(self.ctx.store.project)

    def _store(self, st: FT.FleetTwinSettings, text) -> bool:
        new = FT.FleetTwinSettings.from_dict(st.to_dict()).to_dict()
        if new == self.settings.to_dict():
            self._sync_fields()
            return False

        def fn(p):
            FT.ensure_slot(p)['settings'] = new
        self.ctx.store.edit('analysis', fn, f'Fleet twin · {text}')
        return True

    def _set(self, text, **kw):
        st = self.settings
        for k, v in kw.items():
            setattr(st, k, v)
        return self._store(st, text)

    def _defaults(self):
        self._store(FT.FleetTwinSettings(variant=self.settings.variant), 'default settings')

    def _as_of_edited(self):
        text = self.as_of.text().strip()
        if text and FL.parse_date(text) is None:
            self.ctx.msg(f'"{text}" is not a date: use YYYY-MM-DD.', 4000)
            self._sync_fields()
            return
        self._set(f'as of {text or "today"}', as_of=FL.normalise_date(text) if text else '')

    def _variant_picked(self, i):
        uid = self.variant.itemData(i) or ''
        if uid != self.settings.variant:
            v = self.ctx.store.project.variant_by_uid(uid)
            self._set(f'plan {v.name if v else "of the fleet"}', variant=uid)

    def _sync_fields(self):
        st = self.settings
        for k, f in self.fields.items():
            f.set_value(float(getattr(st, k)) * SCALE.get(k, 1.0))
        for box, val in ((self.limit, st.limit_source), (self.pool, st.pool)):
            box.setCurrentIndex(max(box.findData(val), 0))
        if not self.as_of.hasFocus():
            self.as_of.setText(st.as_of)
        p = self.ctx.store.project
        items = [('The fleet\'s plan', '')] + [(v.name, v.uid) for v in p.variants]
        if [(self.variant.itemText(i), self.variant.itemData(i)) for i in range(self.variant.count())] != items:
            self.variant.blockSignals(True)
            self.variant.clear()
            for text, uid in items:
                self.variant.addItem(text, uid)
            self.variant.blockSignals(False)
        self.variant.setCurrentIndex(max(self.variant.findData(st.variant), 0))
        vs = FL.vehicles(p)
        vitems = [(v['name'], v['id']) for v in vs]
        if [(self.vehicle.itemText(i), self.vehicle.itemData(i)) for i in range(self.vehicle.count())] != vitems:
            self.vehicle.blockSignals(True)
            self.vehicle.clear()
            for text, vid in vitems:
                self.vehicle.addItem(text, vid)
            self.vehicle.blockSignals(False)
        if self.sel_vehicle not in [v for _, v in vitems]:
            self.sel_vehicle = vitems[0][1] if vitems else ''
        self.vehicle.setCurrentIndex(max(self.vehicle.findData(self.sel_vehicle), 0))
        self._sync_watch()

    def _sync_watch(self):
        p = self.ctx.store.project
        w = FT.watches(p).get(self.sel_vehicle)
        has = self.sel_vehicle != ''
        for b in (self.b_import, self.b_drive, self.b_watch):
            b.setEnabled(has)
        self.b_health.setEnabled(has and self.health_busy is False and
                                 any(str(v).strip() for v in (getattr(p, 'streams', {}) or {}).values()))
        self.b_unwatch.setEnabled(w is not None)
        self.watch_on.setEnabled(w is not None)
        self.watch_on.blockSignals(True)
        self.watch_on.setChecked(bool(w and w['enabled']))
        self.watch_on.blockSignals(False)
        if w:
            self.watch_label.setText(self.watch_label.fontMetrics().elidedText(
                w['folder'], Qt.TextElideMode.ElideMiddle, max(self.watch_label.width(), 300)))
            self.watch_label.setToolTip(w['folder'])
        else:
            self.watch_label.setText('No watch folder' if has else 'Add vehicles on the Fleet page first')
            self.watch_label.setToolTip('')
        seen = FT.seen(p).get(self.sel_vehicle, {})
        lines = [f'{v.get("drive")}: {v.get("status")} · {v.get("message")}' for v in list(seen.values())[-4:]]
        lines += [f'{FT._drive_name(path)}: {WATCH_STATE_TEXT.get(s, s)} · {t}'
                  for (vid, path), (s, t) in sorted(self.watch_state.status.items())
                  if vid == self.sel_vehicle and s not in ('read', 'reading')][:6]
        if self.reading is not None:
            lines.append(f'Calibrating {os.path.basename(self.reading[1].rstrip(os.sep))}…')
        self.watch_status.setText('\n'.join(lines))
        any_on = any(x['enabled'] for x in FT.watches(p).values())
        if any_on and not self.poll_timer.isActive():
            self.poll_timer.start()
        elif not any_on and self.poll_timer.isActive():
            self.poll_timer.stop()

    def _vehicle_picked(self, i):
        self.sel_vehicle = self.vehicle.itemData(i) or ''
        self._sync_watch()

    # ------------------------------------------------------------------ imports and the watch
    def _vehicle_name(self, vid):
        return next((v['name'] for v in FL.vehicles(self.ctx.store.project) if v['id'] == vid), vid)

    def _add_obs(self, vid, obs, text, seen=None):
        def fn(p):
            n = FT.add_observations(p, obs)
            if seen is not None:
                FT.mark_seen(p, vid, seen[0], seen[1])
            return n
        return self.ctx.store.edit('analysis', fn, f'Fleet twin · {text}')

    def import_online(self, path=None, date=None):
        """Import an online self-calibration's JSON export for the selected vehicle (one undo step). Returns the
        number of observations added."""
        vid = self.sel_vehicle
        if not vid:
            self.ctx.msg('Add a vehicle on the Fleet page first.', 4000)
            return 0
        if path is None:
            path, _ = QFileDialog.getOpenFileName(self, 'Import an online self-calibration', '',
                                                  'Online self-calibration (*.json);;All files (*)')
            if not path:
                return 0
        date = self.drive_date.text().strip() if date is None else date
        try:
            doc = FT.read_online_file(path)
            if date and FL.parse_date(date) is None:
                raise FT.TwinError(f'"{date}" is not a date: use YYYY-MM-DD.')
            obs = FT.online_observations(doc, vid, FL.normalise_date(date) if date else '', self.settings.online_obs,
                                         drive=os.path.basename(path), source=os.path.basename(path))
        except FT.TwinError as e:
            self.ctx.msg(str(e), 6000)
            return 0
        n = self._add_obs(vid, obs, f'{os.path.basename(path)} for {self._vehicle_name(vid)}')
        self.ctx.msg(f'Added {n} observation{"s" * (n != 1)} of {os.path.basename(path)} to '
                     f'{self._vehicle_name(vid)}.', 5000)
        return n

    def calibrate_drive(self, path=None) -> bool:
        """Calibrate a recorded drive for the selected vehicle in the background; its segments are added when done."""
        vid = self.sel_vehicle
        if not vid:
            self.ctx.msg('Add a vehicle on the Fleet page first.', 4000)
            return False
        if path is None:
            path = QFileDialog.getExistingDirectory(self, 'A recorded drive (nuScenes folder or rosbag2 folder)')
            if not path:
                return False
        found = FT.drive_paths(os.path.dirname(path.rstrip(os.sep)))
        sub = os.path.join(path, 'nuscenes')
        if os.path.isdir(sub) and sub in found:
            path = sub
        date = self.drive_date.text().strip()
        return self._read_drive(vid, path, FT.drive_key(path), FL.normalise_date(date) if FL.parse_date(date) else '')

    def set_watch_folder(self, folder=None):
        vid = self.sel_vehicle
        if not vid:
            return False
        if folder is None:
            folder = QFileDialog.getExistingDirectory(self, 'Watch a folder for new drives of this vehicle')
            if not folder:
                return False

        def fn(p):
            FT.set_watch(p, vid, folder or None, True)
        self.ctx.store.edit('analysis', fn, f'Fleet twin · watch folder of {self._vehicle_name(vid)}'
                            + ('' if folder else ' removed'))
        self._sync_watch()
        return True

    def _watch_toggled(self, on):
        w = FT.watches(self.ctx.store.project).get(self.sel_vehicle)
        if w is None:
            return

        def fn(p):
            FT.set_watch(p, self.sel_vehicle, w['folder'], bool(on))
        self.ctx.store.edit('analysis', fn, f'Fleet twin · watching {"on" if on else "off"}')
        self._sync_watch()

    def poll_watch(self):
        """Look at the watch folders (cheap: file sizes and times); read the next ready drive in the background."""
        if self.reading is not None:
            return []
        ready = self.watch_state.poll(self.ctx.store.project)
        if ready:
            vid, path, key = ready[0]
            self._read_drive(vid, path, key)            # dated by the drive itself (the typed date is for one drive)
        else:
            self._sync_watch()
        return ready

    def _read_drive(self, vid, path, key, date='') -> bool:
        if self.reading is not None:
            self.ctx.msg('A drive is being calibrated: wait for it to finish.', 4000)
            return False
        project = self.ctx.store.project.snapshot()
        win = self.ctx.window
        eng = self._engine()
        self.reading = (vid, path, key)
        self.watch_state.busy.add((vid, key))
        self._sync_watch()

        def work(progress, cancel):
            return (vid, path, key, FT.read_drive(project, vid, path, eng, progress, cancel, date=date))
        if win is not None and hasattr(win, 'run_job'):
            ok = win.run_job(DRIVE_JOB, f'Calibrating {os.path.basename(path.rstrip(os.sep))}', work,
                             lambda out: self._drive_done(out), lambda m: self._drive_failed(m))
            if not ok:
                self.reading = None
                self.watch_state.busy.discard((vid, key))
            return ok

        def thread():
            try:
                self.sig.drive_done.emit(work(lambda f, t='': None, lambda: False))
            except RuntimeError:
                pass
            except Exception as e:                          # noqa: BLE001
                try:
                    self.sig.drive_failed.emit(f'{type(e).__name__}: {e}')
                except RuntimeError:
                    pass
        threading.Thread(target=thread, daemon=True).start()
        return True

    def read_rig_health(self, seconds=3.0, run_for=None) -> bool:
        """Read the live streams of the plan's cameras (project.streams) for a few seconds in the background
        (core.rig_health.run_for) and add the drift events it finds to the selected vehicle (one undo step)."""
        vid = self.sel_vehicle
        p = self.ctx.store.project
        if not vid or self.health_busy:
            return False
        from ...core import rig_health as RH
        run_for = run_for or RH.run_for
        var = FT.plan_variant(p)
        cams = [c for c in var.cameras if c.visible and str((p.streams or {}).get(c.tag, '')).strip()]
        if not cams:
            self.ctx.msg('No camera of the plan has a live stream: set them on the Rig health page.', 5000)
            return False
        project = p.snapshot()
        self.health_busy = True
        self._sync_watch()

        def thread():
            try:
                snap = run_for(project, cams, seconds)
                self.sig.health_done.emit((vid, snap))
            except RuntimeError:
                pass
            except Exception as e:                          # noqa: BLE001  a stream that cannot be read
                try:
                    self.sig.health_done.emit((vid, e))
                except RuntimeError:
                    pass
        threading.Thread(target=thread, daemon=True).start()
        return True

    def _health_done(self, out):
        self.health_busy = False
        vid, snap = out
        if isinstance(snap, Exception):
            self.ctx.msg(f'Rig health could not read the streams: {snap}', 6000)
            self._sync_watch()
            return
        p = self.ctx.store.project
        evs = FT.rig_health_events(p, snap, vid, variant=FT.plan_variant(p))
        if evs:
            self.ctx.store.edit('analysis', lambda q: FT.add_events(q, evs),
                                f'Fleet twin · {len(evs)} rig-health drift event{"s" * (len(evs) != 1)}')
        cams = snap.get('cameras', {}) if isinstance(snap, dict) else {}
        self.ctx.msg(f'Rig health: {len(evs)} of {len(cams)} camera{"s" * (len(cams) != 1)} with the body edge off '
                     'the plan.', 6000)
        self._sync_watch()

    def _drive_done(self, out):
        if self.reading is None:
            return
        vid, path, key = self.reading
        self.reading = None
        self.watch_state.busy.discard((vid, key))
        rep = out[3] if out else None
        if rep is None:
            self._sync_watch()
            return
        name = rep['drive']
        if rep['status'] == 'changed':                  # still being written: looked at afresh, nothing added
            self.watch_state.forget(vid, path)
            self.ctx.msg(f'{name}: {rep["message"]}', 6000)
            self._sync_watch()
            return
        info = dict(rep, observations=len(rep['observations']))
        self._add_obs(vid, rep['observations'], f'drive {name} for {self._vehicle_name(vid)}', (key, info))
        self.watch_state.mark_read(vid, path)
        self.ctx.msg(f'{name}: {rep["status"]} · {rep["message"]}', 6000)
        self._sync_watch()

    def _drive_failed(self, msg):
        if self.reading is not None:
            vid, path, key = self.reading
            self.reading = None
            self.watch_state.busy.discard((vid, key))

            def fn(p):
                FT.mark_seen(p, vid, key, {'drive': os.path.basename(path.rstrip(os.sep)), 'status': 'error',
                                           'message': msg.splitlines()[0][:200], 'observations': 0})
            self.ctx.store.edit('analysis', fn, 'Fleet twin · a drive could not be read')
        self.ctx.msg(f'The drive could not be calibrated: {msg.splitlines()[0]}', 6000)
        self._sync_watch()

    # ------------------------------------------------------------------ run
    def _engine(self):
        """An analysis engine a worker may use (the window's shared one; None: the worker makes its own)."""
        win = self.ctx.window
        if win is not None and hasattr(win, '_shared_engine'):
            return win._shared_engine()
        an = getattr(self.ctx, 'analysis', None)
        return an.engine.share() if an is not None and hasattr(an, 'engine') else None

    def _inputs_key(self):
        return FT._key_of(self.ctx.store.project, self.settings)

    @property
    def stale(self) -> bool:
        return self.result is not None and self._for != self._inputs_key()

    def _auto_run(self):
        if self.isVisible() and not self.busy and (self.result is None or self.stale):
            self.run()

    def run(self) -> bool:
        if self.busy:
            return False
        p = self.ctx.store.project
        res = FT.cached(p)
        if res is not None:                                # the same inputs ran already (here or elsewhere)
            self.result, self._for = res, self._inputs_key()
            self.status_text = self._status_of(res)
            self.refresh()
            return True
        project = p.snapshot()
        win = self.ctx.window
        eng = self._engine()
        self._gen += 1
        gen = self._gen
        self._cancel = threading.Event()
        cancel_ev = self._cancel
        key = (p, self._inputs_key())
        self.busy = True
        self.status_text = 'Fitting the fleet…'

        def work(progress, cancel):
            out = FT.run(project, eng, FT.settings_of(project), None, progress,
                         lambda: cancel() or cancel_ev.is_set())
            if out is not None:                            # cached for the live project's key too
                with FT._lock:
                    FT._CACHE[key[1]] = out
            return out
        if win is not None and hasattr(win, 'run_job'):
            ok = win.run_job(JOB, 'Fleet twin', work, lambda out: self._done(gen, (key, out)),
                             lambda m: self._failed(gen, m))
            if not ok:
                self.busy = False
                self.status_text = ''
                self.ctx.msg('The fleet twin is still running: cancel it first.', 4000)
            self.refresh()
            return ok

        def thread():
            try:
                out = work(lambda f, text='': None, lambda: gen != self._gen)
                self.sig.done.emit(gen, (key, out))
            except RuntimeError:
                pass
            except Exception as e:                          # noqa: BLE001
                try:
                    self.sig.failed.emit(gen, f'{type(e).__name__}: {e}\n{traceback.format_exc()}')
                except RuntimeError:
                    pass
        threading.Thread(target=thread, daemon=True).start()
        self.refresh()
        return True

    def cancel(self):
        if not self.busy:
            return
        self._cancel.set()
        win = self.ctx.window
        job = getattr(win, '_jobs', {}).get(JOB) if win is not None else None
        if job is not None:
            job['cancel'].set()
        self.status_text = 'Cancelling…'
        self.refresh()

    @staticmethod
    def _status_of(res):
        m = res.metrics
        return (f'{m["cells"]} sensors on {m["vehicles"]} vehicles from {m["observations"]} observations · '
                f'{m["knocks"]} knock{"s" * (m["knocks"] != 1)} · {m["seconds"]:.1f} s')

    def _done(self, gen, out):
        if gen != self._gen:
            return
        self.busy = False
        key, res = out if out is not None else (None, None)
        if res is None or key[0] is not self.ctx.store.project:
            self.status_text = 'Cancelled' if res is None else ''
            self.refresh()
            return
        self.result, self._for = res, key[1]
        self.status_text = self._status_of(res)
        self.refresh()
        self.ctx.msg(f'Fleet twin: {self.status_text}', 5000)
        if self.stale:
            self._auto.start()

    def _failed(self, gen, msg):
        if gen != self._gen:
            return
        self.busy = False
        self.status_text = 'Failed: ' + msg.splitlines()[0]
        self.refresh()

    # ------------------------------------------------------------------ show
    def _on_change(self, *_):
        if self.isVisible():
            self._sync_fields()
            self.refresh()
            self._auto.start()
        else:
            self._dirty = True

    def _on_replaced(self, *_):
        self._gen += 1
        self.result, self._for, self.busy, self.sel = None, None, False, None
        self.reading = None
        self.watch_state = FT.WatchState()
        self.status_text = ''
        self._shown = None
        self._sync_fields()
        self.refresh()
        if self.isVisible():
            self._auto.start()

    def _retheme(self, *_):
        self._shown = None
        self.refresh()

    def showEvent(self, e):
        super().showEvent(e)
        if self._dirty:
            self.activate()

    def activate(self):
        self._dirty = False
        self._sync_fields()
        self.refresh()
        self._auto.start()

    def refresh(self):
        res = self.result
        self.b_run.setEnabled(not self.busy)
        self.b_cancel.setEnabled(self.busy)
        for b in (self.b_export, self.b_sched, self.b_tol):
            b.setEnabled(res is not None and not self.busy)
        empty = not FL.vehicles(self.ctx.store.project)
        text = self.status_text or ('Add vehicles and their records on the Fleet page (Calibrate ▸ Fleet…), or '
                                    'import online self-calibrations here.' if empty else 'Ready: Run fits the fleet.')
        self.status.setText(text + (' · stale: the fleet or the settings changed since' if self.stale else ''))
        if res is self._shown:
            return
        self._shown = res
        self._fill_table(res)
        self._fill_sched(res)
        self._fill_findings(res)
        self._show_cell()

    def _fill_table(self, res):
        t = self.table
        t.clear()
        if res is None:
            t.setRowCount(0)
            t.setColumnCount(0)
            self.headline.setText('')
            return
        tags = [p['tag'] for p in res.plan]
        t.setColumnCount(len(tags))
        t.setHorizontalHeaderLabels(tags)
        t.setRowCount(len(res.vehicles))
        t.setVerticalHeaderLabels([v['name'] for v in res.vehicles])
        for r, v in enumerate(res.vehicles):
            for c, tag in enumerate(tags):
                cell = res.cells.get((v['id'], tag))
                if cell is None:
                    t.setItem(r, c, _it('–', color=THEME.color('muted'), tip='not measured'))
                    continue
                col = QColor(URG_COLOR[cell['urgency']])
                col.setAlphaF(0.75 if cell['urgency'] in ('beyond', 'due', 'soon') else 0.30)
                txt = {'beyond': 'beyond', 'due': 'now'}.get(cell['urgency']) or \
                    (FT.date_text(cell['date_risk'])[:7] if cell['date_risk'] else 'holds')
                if cell['knocks']:
                    txt += ' ●'
                t.setItem(r, c, _it(txt, num=True, tip=cell_tip(cell, res), fill=col, data=(v['id'], tag)))
        hh = t.horizontalHeader()
        for c in range(len(tags)):
            hh.setSectionResizeMode(c, QHeaderView.ResizeMode.Stretch)
        urg = {u: sum(c['urgency'] == u for c in res.cells.values()) for u in FT.URGENCY}
        self.headline.setText(f'{len(res.cells)} sensors on {len(res.vehicles)} vehicles as of '
                              f'{FT.date_text(res.as_of)}: {urg["beyond"]} beyond, {urg["due"]} due, '
                              f'{urg["soon"]} within {FT.SOON_DAYS:.0f} days, {urg["later"]} later · '
                              f'{len(res.knocks)} knock{"s" * (len(res.knocks) != 1)} · exit code {FT.exit_code(res)}')
        if self.sel not in res.cells:
            order = sorted(res.cells, key=lambda k: (FT.URGENCY.index(res.cells[k]['urgency']),
                                                     not res.cells[k]['knocks'], res.cells[k]['date_risk'] or 1e18))
            self.sel = order[0] if order else None
            self.sel_axis = ''

    def _fill_sched(self, res):
        t = self.sched
        t.setRowCount(0)
        if res is None:
            return
        t.setRowCount(len(res.schedule))
        for r, row in enumerate(res.schedule):
            col = QColor(URG_COLOR[row['urgency']])
            t.setItem(r, 0, _it(row['recalibrate_by'], num=True, data=(row['vehicle_id'], row['sensor'])))
            t.setItem(r, 1, _it(row['vehicle']))
            t.setItem(r, 2, _it(row['sensor']))
            t.setItem(r, 3, _it(FT.URGENCY_LABEL[row['urgency']], color=col))
            t.setItem(r, 4, _it(f'{row["p_now"]:.3f}', num=True))
            t.setItem(r, 5, _it(row['median'] or '—', num=True))
            t.setItem(r, 6, _it(row['axis']))
            t.setItem(r, 7, _it(row['limit']))

    def _fill_findings(self, res):
        for w in self.findings:
            self.find_lay.removeWidget(w)
            w.hide()
            w.deleteLater()
        self.findings = []
        if res is None:
            return
        for f in res.findings:
            w = StatusLine(SEV_TONE.get(f['severity']), f['message'])
            self.find_lay.insertWidget(self.find_lay.count() - 1, w)
            self.findings.append(w)

    def _cell_clicked(self, r, c):
        it = self.table.item(r, c)
        key = it.data(Qt.ItemDataRole.UserRole) if it is not None else None
        if key:
            self.select(tuple(key))

    def _sched_clicked(self, r, c):
        it = self.sched.item(r, 0)
        key = it.data(Qt.ItemDataRole.UserRole) if it is not None else None
        if key:
            self.select(tuple(key))

    def select(self, key, axis=''):
        """Show one vehicle's sensor (vehicle id, tag), optionally on one axis."""
        if self.result is None or key not in self.result.cells:
            return False
        self.sel, self.sel_axis = key, axis
        self._show_cell()
        return True

    def _axis_picked(self, i):
        self.sel_axis = self.axis_box.itemData(i) or ''
        self._show_cell()

    def _show_cell(self):
        res = self.result
        c = res.cells.get(self.sel) if res is not None and self.sel else None
        if c is None:
            self.chart.set_data(res, None, '')
            self.rates.set_data(None, {})
            self.axis_box.clear()
            return
        axes = list(c['axes'])
        if self.sel_axis not in axes:
            self.sel_axis = FT.chart_axis(c)
        self.axis_box.blockSignals(True)
        self.axis_box.clear()
        for a in axes:
            self.axis_box.addItem(f'{a} ({c["axes"][a]["unit"]})', a)
        self.axis_box.setCurrentIndex(axes.index(self.sel_axis))
        self.axis_box.blockSignals(False)
        self.chart.set_data(res, c, self.sel_axis)
        row = next((f for f in res.fleet if f['tag'] == c['tag'] and f['axis'] == self.sel_axis), None)
        self.rates.set_data(row, {v['id']: v['name'] for v in res.vehicles})

    def summary(self) -> dict | None:
        if self.result is None:
            return None
        return dict(FT.summary(self.result), stale=self.stale)

    # ------------------------------------------------------------------ exports and the hand-off
    def export(self, folder=None):
        res = self.result
        if res is None:
            return []
        if folder is None:
            folder = QFileDialog.getExistingDirectory(self, 'Export the fleet twin into')
            if not folder:
                return []
        try:
            paths = FT.export(res, self.ctx.store.project, folder, picks=[self.sel] if self.sel else None)
        except (OSError, ValueError) as e:
            self.ctx.msg(f'Export failed: {e}', 6000)
            return []
        self.ctx.msg(f'Exported the fleet twin to {folder}', 5000)
        return paths

    def export_schedule(self, path=None):
        res = self.result
        if res is None:
            return ''
        if path is None:
            path, _ = QFileDialog.getSaveFileName(self, 'Save the recalibration schedule', 'fleet_twin_schedule.csv',
                                                  'CSV (*.csv)')
            if not path:
                return ''
        try:
            with open(path, 'w', encoding='utf-8', newline='') as fh:
                fh.write(FT.schedule_csv(res))
        except OSError as e:
            self.ctx.msg(f'Cannot write {path}: {e}', 6000)
            return ''
        self.ctx.msg(f'Saved the recalibration schedule ({len(res.schedule)} rows) to {path}', 5000)
        return path

    def use_tolerance(self):
        """Set analysis.tol_deg / tol_mm to the measured fleet's ± 2 σ after tol_horizon_days (one undo step)."""
        res = self.result
        if res is None:
            return None
        ts = FT.tolerance_inputs(res, project=self.ctx.store.project)['tol_settings']
        deg = max(ts['yaw'] or 0.0, ts['pitch'] or 0.0, ts['roll'] or 0.0)
        mm = ts['pos_mm'] or 0.0
        if deg <= 0 and mm <= 0:
            self.ctx.msg('The fleet has no measured camera spread to use.', 4000)
            return None
        a = self.ctx.store.project.analysis
        deg = round(deg, 3) if deg > 0 else a.tol_deg
        mm = round(mm, 2) if mm > 0 else a.tol_mm

        def fn(p):
            p.analysis.tol_deg, p.analysis.tol_mm = float(deg), float(mm)
        self.ctx.store.edit('analysis', fn, f'Mounting tolerance from the fleet twin: ±{deg:g}° / ±{mm:g} mm')
        self.ctx.msg(f'Mounting tolerance set to ±{deg:g}° and ±{mm:g} mm (the measured fleet after '
                     f'{res.settings.tol_horizon_days:.0f} days).', 6000)
        return deg, mm
