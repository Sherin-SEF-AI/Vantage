"""Simulate ▸ Simulation campaigns: many CARLA drives of the rig, each varied in one known way (a fault of known size,
a weather, a rig variant, a seed), checked by Vantage's own checks and compared with what was injected: detection
curves with Wilson intervals, false alarms on clean drives, the smallest fault each check finds, estimation accuracy,
a weather sweep against Vantage's weather model, rigs compared. Built from a template or edited; the cost is
estimated before running; a run is a background job with progress and Cancel, and a cancelled or failed campaign
resumes where it stopped (drives and evaluations are kept per key)."""
from __future__ import annotations

import json
import math
import os
import threading
import traceback

from PyQt6.QtCore import QObject, QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen, QPolygonF
from PyQt6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QFileDialog, QGridLayout, QHBoxLayout,
                             QHeaderView, QLineEdit, QListWidget, QListWidgetItem, QPlainTextEdit, QScrollArea,
                             QSplitter, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget)

from ... import workdir
from ...core import simcampaign as SC
from ..charts import ChartBase
from ..theme import STATUS, THEME, mono, sans
from ..widgets import ROW_H, ScrubSpinBox, SegmentedControl, button, hline, label
from .recording_ws import Tile

JOB = 'campaign'
EMPTY = ('Pick a template (Fault detectability, Weather sweep, Rig A vs B), check the cost, then Run. CARLA must be '
         'running on the port below (Simulate ▸ CARLA ▸ Start), or choose "Start its own".')
BASE_FIELDS = (
    # key, label, unit, decimals, step, tip, default
    ('duration_s', 'Drive', 's', 1, 1.0, 'Seconds recorded per drive', 10.0),
    ('traffic', 'Traffic', 'cars', 0, 1.0, 'Other vehicles on the autopilot', 30),
    ('image_scale', 'Image scale', '', 2, 0.05, 'Recorded images against the sensor\'s size (the drive-log check '
                                                'needs pixels finer than its tolerance)', 0.5),
    ('tick_hz', 'Step rate', 'Hz', 0, 10.0, 'Simulation steps per second (0: chosen by core.simrig). The fault '
                                            'template uses 320 Hz: the sample\'s sync plan then rounds to ~1 ms', 0.0),
    ('port', 'Port', '', 0, 1.0, 'The CARLA server\'s RPC port (the traffic manager: its own port in the CARLA '
                                 'settings)', 2000),
)
FAULT_COLS = (('Kind', 'rotation, translation, clock_offset, drop, freeze, range_bias'),
              ('Sensors', 'Sensor tags, comma separated'),
              ('Axes', 'yaw, pitch, roll (rotation) or x, y, z (translation); empty: all'),
              ('Magnitudes', 'Comma separated, in the kind\'s unit: ° · cm · ms · % of frames · s frozen · cm'),
              ('Signs', 'alternate (by seed), positive, negative, random'))
TH_COLS = (('Check', ''), ('Sensor', ''), ('Fault', ''), ('Axis', 'any: every axis pooled'),
           ('False alarms', 'Share of clean drives the check flags this sensor on (the curve\'s floor)'),
           ('Clean', 'The check\'s own figure on the clean drives: its baseline (a plan rounding, an estimator\'s '
                     'bias, the body edge\'s distance)'),
           ('Empirical', 'The smallest tested size from which every larger one is found at least the target share'),
           ('Fit', 'From the logistic fit in log size, with its bootstrap interval'), ('Interval', ''))
FA_COLS = (('Check', ''), ('Sensor', ''), ('Set', 'Clean drives, or the sensor unfaulted in faulty drives '
                                                  '(cross-talk from another sensor\'s fault)'),
           ('Flagged', ''), ('Drives', ''), ('Rate', 'With its Wilson interval'), ('Undecided', 'The check gave no '
                                                                                               'answer'),
           ('On CARLA', 'Whether the check can judge this on a CARLA drive. Not applicable: its flags are shown but '
                        'are no false alarms (hover for why)'))
ACC_COLS = (('Check', ''), ('Sensor', ''), ('Fault', ''), ('Axis', ''), ('n', ''),
            ('Bias', 'Mean of estimate − injected'), ('RMS', 'Root mean square of estimate − injected'),
            ('Paired RMS', 'Against the clean drive of the same seed (the scene\'s own bias taken out)'))
WX_COLS = (('Weather', ''), ('Class', ''), ('Objects', 'Labelled objects not ignored, every camera, seeds pooled'),
           ('Recall', 'With its Wilson interval'), ('R50', 'Range at which recall falls through 50 %'),
           ('CARLA factor', 'R50 against the clear drive\'s'), ('Vantage factor', 'Vantage\'s perception range '
                                                                                   'under the condition against clear '
                                                                                   '(median over directions)'),
           ('Lidar', 'Lidar points per sweep against the clear drive\'s'))
DRIVE_COLS = (('Drive', ''), ('Role', 'reference: CARLA\'s own car per camera for the drift indicator'),
              ('Seed', ''), ('Setting', 'Variant · weather · scenario'), ('Conditions', 'The faults on this drive'),
              ('Status', ''), ('Wall', 'Seconds the drive took'))
TABS = ('Detectability', 'False alarms', 'Accuracy', 'Weather', 'Rigs', 'Drives', 'Notes')


def _table(cols, stretch=None):
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
    t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    t.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    hh = t.horizontalHeader()
    for c in range(len(cols)):
        hh.setSectionResizeMode(c, QHeaderView.ResizeMode.Stretch if c == stretch else
                                QHeaderView.ResizeMode.ResizeToContents)
    if stretch is None:
        hh.setStretchLastSection(True)
    return t


def _cell(t, r, c, text, num=False, tip=None, color=None):
    it = QTableWidgetItem(text)
    if num:
        it.setFont(mono(12))
        it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    if tip:
        it.setToolTip(tip)
    if color is not None:
        it.setForeground(color)
    t.setItem(r, c, it)
    return it


def _g(v, nd=2, unit=''):
    return '—' if v is None else f'{v:.{nd}f}{(" " + unit) if unit else ""}'


def _pc(v, ci=None) -> str:
    if v is None:
        return '—'
    s = f'{v * 100:.0f} %'
    if ci:
        s += f'  [{ci[0] * 100:.0f}–{ci[1] * 100:.0f}]'
    return s


def _parse_list(text, num=True) -> list:
    out = []
    for x in str(text).replace(';', ',').split(','):
        x = x.strip()
        if not x:
            continue
        if num:
            try:
                out.append(float(x))
            except ValueError:
                continue
        else:
            out.append(x)
    return out


# ============================================================================ charts
class CurveChart(ChartBase):
    """Detection probability (0–100 %) against fault size on a log axis, one line per curve with its Wilson
    intervals as whiskers, the false-alarm floor dashed, the target dotted; hover gives k/n and the interval."""
    MARGIN = (44, 26, 14, 34)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(230)
        self.curves: list = []             # [(name, QColor, unit, [point dict], floor)]
        self.target = 0.9
        self._marks = []

    def set_data(self, curves, target):
        self.curves, self.target = curves, target
        self.tip = None
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.paint_title(p, 'detected share · size (log)')
        L, T, R, B = self.MARGIN
        r = QRectF(L, T, max(10.0, self.width() - L - R), max(10.0, self.height() - T - B))
        self.paint_grid_y(p, r, [0, 0.25, 0.5, 0.75, 1.0], 1.0, lambda v: f'{v * 100:.0f}%')
        self._marks = []
        mags = [pt['magnitude'] for c in self.curves for pt in c[3] if pt['magnitude'] > 0]
        if not mags:
            p.setPen(THEME.color('muted'))
            p.setFont(sans(11))
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, 'No faulty drives evaluated yet')
            return
        lo, hi = math.log10(min(mags)), math.log10(max(mags))
        if hi - lo < 1e-9:
            lo, hi = lo - 0.5, hi + 0.5
        pad = 0.06 * (hi - lo)
        lo, hi = lo - pad, hi + pad

        def X(m):
            return r.left() + (math.log10(m) - lo) / (hi - lo) * r.width()

        def Y(v):
            return r.bottom() - v * r.height()
        p.setFont(mono(10))
        p.setPen(THEME.color('muted'))
        for m in sorted(set(mags)):
            p.drawText(QRectF(X(m) - 30, r.bottom() + 4, 60, 14), Qt.AlignmentFlag.AlignCenter, f'{m:g}')
        units = sorted({c[2] for c in self.curves})
        p.drawText(QRectF(r.right() - 80, r.bottom() + 18, 80, 14), Qt.AlignmentFlag.AlignRight, ' / '.join(units))
        pen = QPen(THEME.color('strong'), 1, Qt.PenStyle.DotLine)
        p.setPen(pen)
        p.drawLine(QPointF(r.left(), Y(self.target)), QPointF(r.right(), Y(self.target)))
        for name, col, unit, pts, floor in self.curves:
            if floor:
                p.setPen(QPen(col, 1, Qt.PenStyle.DashLine))
                p.drawLine(QPointF(r.left(), Y(floor)), QPointF(r.right(), Y(floor)))
            p.setPen(QPen(col, 2))
            prev = None
            for pt in pts:
                if pt['p'] is None:
                    continue
                q = QPointF(X(pt['magnitude']), Y(pt['p']))
                if pt.get('ci'):
                    p.setPen(QPen(col, 1))
                    p.drawLine(QPointF(q.x(), Y(pt['ci'][0])), QPointF(q.x(), Y(pt['ci'][1])))
                p.setPen(QPen(col, 2))
                if prev is not None:
                    p.drawLine(prev, q)
                p.setBrush(col)
                p.drawEllipse(q, 3, 3)
                prev = q
                self._marks.append((q, name, unit, pt))
        if self.tip is not None:
            self.paint_tip(p)

    def mouseMoveEvent(self, e):
        pos = e.position()
        best = None
        for q, name, unit, pt in self._marks:
            d = (q.x() - pos.x()) ** 2 + (q.y() - pos.y()) ** 2
            if d < 100 and (best is None or d < best[0]):
                best = (d, q, name, unit, pt)
        if best is None:
            self.tip = None
        else:
            _, q, name, unit, pt = best
            ci = pt.get('ci')
            self.set_tip(q, f'{pt["k"]}/{pt["n"]} · {pt["p"] * 100:.0f} %', f'{name} at {pt["magnitude"]:g} {unit}',
                         f'{ci[0] * 100:.0f}–{ci[1] * 100:.0f} % (Wilson)' if ci else '')
        self.update()

    def leaveEvent(self, e):
        self.tip = None
        self.update()


class FactorChart(ChartBase):
    """Per weather: CARLA's range factor (R50 against clear, filled) and Vantage's predicted one (ring) for a class,
    with the recall as a bar behind them."""
    MARGIN = (44, 26, 14, 40)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(220)
        self.rows: list = []                # [(weather, recall, r50 factor, vantage factor, recall ratio)]

    def set_data(self, rows):
        self.rows = rows
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.paint_title(p, 'bar: recall · ●: CARLA R50 against clear (◆: recall against clear, no R50) · ○: '
                            'Vantage\'s range factor')
        L, T, R, B = self.MARGIN
        r = QRectF(L, T, max(10.0, self.width() - L - R), max(10.0, self.height() - T - B))
        top = 1.25
        self.paint_grid_y(p, r, [0, 0.25, 0.5, 0.75, 1.0, 1.25], top, lambda v: f'{v:g}')
        if not self.rows:
            p.setPen(THEME.color('muted'))
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, 'No weather sweep')
            return
        n = len(self.rows)
        w = r.width() / n
        ink = QColor('#2a78d6')
        for i, (wx, rec, f_c, f_v, ratio) in enumerate(self.rows):
            x = r.left() + (i + 0.5) * w
            if rec is not None:
                bar = QRectF(x - w * 0.3, r.bottom() - rec / top * r.height(), w * 0.6, rec / top * r.height())
                p.fillPath(self.bar_path(bar), THEME.color('strong'))
            for val, kind in ((f_c, 'r50'), (ratio if f_c is None else None, 'ratio'), (f_v, 'vantage')):
                if val is None:
                    continue
                q = QPointF(x + (6 if kind == 'vantage' else -6), r.bottom() - min(val, top) / top * r.height())
                p.setPen(QPen(ink, 2))
                p.setBrush(Qt.BrushStyle.NoBrush if kind == 'vantage' else ink)
                if kind == 'ratio':
                    p.drawPolygon(QPolygonF([q + QPointF(0, -5), q + QPointF(5, 0), q + QPointF(0, 5),
                                             q + QPointF(-5, 0)]))
                else:
                    p.drawEllipse(q, 4, 4)
            p.setPen(THEME.color('muted'))
            p.setFont(sans(10))
            p.drawText(QRectF(x - w / 2, r.bottom() + 4, w, 28), Qt.AlignmentFlag.AlignHCenter | Qt.TextFlag.TextWordWrap,
                       wx.replace('vantage:', '').replace('_', ' '))


class _Sig(QObject):
    done = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)
    progress = pyqtSignal(int, float, str)


# ============================================================================ the page
class CampaignPage(QWidget):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.result: SC.CampaignResult | None = None
        self.runner = None                        # tests: a bridge runner (in-process against the fake carla)
        self.detector = None                      # tests: a stand-in detector
        self.cancel_when = None                   # tests: an extra cancel condition
        self.busy = False
        self.status_text = ''
        self._gen = 0
        self._dirty = True
        self._syncing = False
        self._for = None
        self.sig = _Sig()
        self.sig.done.connect(self._thread_done)
        self.sig.failed.connect(self._failed)
        self.sig.progress.connect(self._progress)
        self.setObjectName('base')
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setHandleWidth(2)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setWidget(self._left())
        sc.setMinimumWidth(380)
        split.addWidget(sc)
        split.addWidget(self._right())
        split.setSizes([420, 1000])
        lay.addWidget(split)
        ctx.store.changed.connect(self._on_change)
        ctx.store.project_replaced.connect(self._on_replaced)
        THEME.changed.connect(self._retheme)
        self._sync_fields()
        self.refresh()

    # ------------------------------------------------------------------ left: the design
    def _left(self):
        w = QWidget()
        w.setObjectName('panel')
        ll = QVBoxLayout(w)
        ll.setContentsMargins(8, 6, 8, 8)
        ll.setSpacing(4)
        top = QHBoxLayout()
        top.addWidget(label('Simulation campaigns'))
        top.addStretch(1)
        top.addWidget(label('faults · weather · rigs', muted=True, small=True))
        ll.addLayout(top)
        ll.addWidget(hline())
        ll.addWidget(label('Many CARLA drives of the rig, each varied in one known way, every drive checked by '
                           'Vantage\'s own checks: how small a fault each check finds, how often it cries wolf, how '
                           'well it measures what it finds, and what the weather does to detection.', muted=True,
                           small=True, wrap=True))
        row = QHBoxLayout()
        self.tmpl = QComboBox()
        for k, (name, about) in SC.TEMPLATES.items():
            self.tmpl.addItem(name, k)
            self.tmpl.setItemData(self.tmpl.count() - 1, about, Qt.ItemDataRole.ToolTipRole)
        self.tmpl.setToolTip('A ready design, fitted to the active variant\'s sensors')
        row.addWidget(self.tmpl, 1)
        b = button('Use template', None, None, 'Replace the campaign with this template (one undo step)')
        b.clicked.connect(lambda: self.apply_template(self.tmpl.currentData()))
        row.addWidget(b)
        ll.addLayout(row)
        g = QGridLayout()
        g.setHorizontalSpacing(6)
        g.setVerticalSpacing(3)
        a = label('Name', ink2=True)
        a.setFixedWidth(110)
        g.addWidget(a, 0, 0)
        self.name_edit = QLineEdit()
        self.name_edit.setToolTip('The campaign\'s name (its folder under the working folder\'s campaigns/)')
        self.name_edit.editingFinished.connect(lambda: self._set('name', name=self.name_edit.text()))
        g.addWidget(self.name_edit, 0, 1)
        g.addWidget(label('Seeds', ink2=True), 1, 0)
        self.seeds_edit = QLineEdit()
        self.seeds_edit.setFont(mono(12))
        self.seeds_edit.setToolTip('Each seed repeats the whole design (traffic, spawn point, drop choices), comma '
                                   'separated')
        self.seeds_edit.editingFinished.connect(
            lambda: self._set('seeds', seeds=[int(x) for x in _parse_list(self.seeds_edit.text())]))
        g.addWidget(self.seeds_edit, 1, 1)
        g.addWidget(label('Scenario', ink2=True), 2, 0)
        self.scen = QComboBox()
        from ...sim import carla_bridge as CB
        for k, (lab, about, _p) in CB.SCENARIOS.items():
            self.scen.addItem(lab, k)
            self.scen.setItemData(self.scen.count() - 1, about, Qt.ItemDataRole.ToolTipRole)
        self.scen.activated.connect(lambda _i: self._set_base_key('scenario', {'name': self.scen.currentData(),
                                                                               'params': {}}))
        g.addWidget(self.scen, 2, 1)
        ll.addLayout(g)
        ll.addWidget(label('Rig variants', ink2=True))
        self.var_list = QListWidget()
        self.var_list.setMaximumHeight(80)
        self.var_list.setToolTip('The variants driven (none ticked: the active one); the first is the reference of '
                                 'a rig comparison')
        self.var_list.itemChanged.connect(lambda _it: self._lists_changed('variants'))
        ll.addWidget(self.var_list)
        ll.addWidget(label('Weathers', ink2=True))
        self.wx_list = QListWidget()
        self.wx_list.setMaximumHeight(110)
        self.wx_list.setToolTip('Vantage conditions (mapped to CARLA\'s weather: fog by a measured calibration, light '
                                'and rain by documented heuristics; compared with Vantage\'s own weather model) or '
                                'CARLA presets; none: the CARLA settings\' weather')
        self.wx_list.itemChanged.connect(lambda _it: self._lists_changed('weathers'))
        ll.addWidget(self.wx_list)
        ll.addWidget(hline())
        ll.addWidget(label('Faults', ink2=True))
        self.faults = QTableWidget(0, len(FAULT_COLS))
        for c, (h, tip) in enumerate(FAULT_COLS):
            it = QTableWidgetItem(h)
            it.setToolTip(tip)
            self.faults.setHorizontalHeaderItem(c, it)
        self.faults.verticalHeader().setVisible(False)
        self.faults.verticalHeader().setDefaultSectionSize(ROW_H)
        self.faults.horizontalHeader().setStretchLastSection(True)
        self.faults.setMaximumHeight(150)
        self.faults.itemChanged.connect(lambda _it: self._faults_edited())
        ll.addWidget(self.faults)
        fb = QHBoxLayout()
        b = button('Add', 'ghost', 'plus', 'Add a fault set (a rotation of the first camera; edit its cells)')
        b.clicked.connect(lambda: self.add_fault_set())
        fb.addWidget(b)
        b = button('Remove', 'ghost', 'trash', 'Remove the selected fault set')
        b.clicked.connect(lambda: self.remove_fault_set())
        fb.addWidget(b)
        fb.addStretch(1)
        self.clean_cb = QCheckBox('Clean drives')
        self.clean_cb.setToolTip('A drive with no fault per seed and setting: the false-alarm rates and baselines')
        self.clean_cb.toggled.connect(lambda v: self._set('clean drives', clean=bool(v)))
        fb.addWidget(self.clean_cb)
        self.multi_cb = QCheckBox('Multiplex')
        self.multi_cb.setToolTip('Faults of different sensors ride in the same drive (each check judges each sensor '
                                 'on its own); off: one fault per drive')
        self.multi_cb.toggled.connect(lambda v: self._set('multiplex', multiplex=bool(v)))
        fb.addWidget(self.multi_cb)
        ll.addLayout(fb)
        ll.addWidget(hline())
        ll.addWidget(label('Drives', ink2=True))
        g = QGridLayout()
        g.setHorizontalSpacing(6)
        g.setVerticalSpacing(3)
        self.base_fields = {}
        for i, (key, text, unit, dec, step, tip, _d) in enumerate(BASE_FIELDS):
            f = ScrubSpinBox('', 'plain', dec, unit, step=step, width=110)
            f.setToolTip(tip)
            f.committed.connect(lambda v, k=key: self._set_base_key(k, int(round(v)) if k in ('traffic', 'port')
                                                                     else float(v)))
            a = label(text, ink2=True)
            a.setFixedWidth(110)
            a.setToolTip(tip)
            g.addWidget(a, i // 2, (i % 2) * 2)
            g.addWidget(f, i // 2, (i % 2) * 2 + 1)
            self.base_fields[key] = f
        ll.addLayout(g)
        ll.addWidget(label('Checks', ink2=True))
        cg = QGridLayout()
        cg.setHorizontalSpacing(8)
        cg.setVerticalSpacing(2)
        self.eval_box = {}
        for i, e in enumerate(SC.EVALUATORS):
            cb = QCheckBox(SC.EVALUATOR_LABEL[e])
            cb.toggled.connect(lambda _v: self._evals_changed())
            cg.addWidget(cb, i // 2, i % 2)
            self.eval_box[e] = cb
        ll.addLayout(cg)
        g = QGridLayout()
        g.setHorizontalSpacing(6)
        g.addWidget(label('As-built on', ink2=True), 0, 0)
        self.ab_on = SegmentedControl([('none', 'None'), ('clean', 'Clean drives'), ('all', 'All')], 'clean')
        self.ab_on.setToolTip('The as-built check is slow and on CARLA drives judges the two bodies\' difference, '
                              'not the mount: run it on the clean drives to see what it says')
        self.ab_on.changed.connect(lambda k: self._set(f'as-built on {k}', asbuilt_on=k))
        g.addWidget(self.ab_on, 0, 1, Qt.AlignmentFlag.AlignLeft)
        g.addWidget(label('Target', ink2=True), 1, 0)
        self.target = ScrubSpinBox('', 'plain', 2, '', step=0.05, width=110)
        self.target.setToolTip('The detection probability the smallest detectable fault is quoted at')
        self.target.committed.connect(lambda v: self._set(f'target {v:g}', target=float(v)))
        g.addWidget(self.target, 1, 1, Qt.AlignmentFlag.AlignLeft)
        g.addWidget(label('Server', ink2=True), 2, 0)
        self.server = SegmentedControl([('running', 'Running'), ('own', 'Start its own')], 'running')
        self.server.setToolTip('Use the CARLA server running on the port, or start one for the campaign and stop it '
                               'afterwards (capped, on the NVIDIA GPU)')
        self.server.changed.connect(lambda k: self._set(f'server {k}', server=k))
        g.addWidget(self.server, 2, 1, Qt.AlignmentFlag.AlignLeft)
        g.addWidget(label('Folder', ink2=True), 3, 0)
        fr = QHBoxLayout()
        self.out_edit = QLineEdit()
        self.out_edit.setPlaceholderText('<working folder>/<project>/campaigns/<name>')
        self.out_edit.setToolTip('Where the drives, their checks and the campaign files go (big: about 0.3 GB per '
                                 'drive for the sample rig)')
        self.out_edit.editingFinished.connect(lambda: self.set_out_dir(self.out_edit.text()))
        fr.addWidget(self.out_edit, 1)
        b = button('…', 'ghost', None, 'Choose the folder')
        b.clicked.connect(lambda: self._pick_out())
        fr.addWidget(b)
        g.addLayout(fr, 3, 1)
        ll.addLayout(g)
        ll.addWidget(hline())
        self.cost = label('', small=True, wrap=True)
        self.cost.setToolTip('Drives × the measured seconds per drive (this campaign\'s own once it has drives)')
        ll.addWidget(self.cost)
        self.problems = label('', small=True, wrap=True)
        self.problems.setStyleSheet(f'color: {STATUS["serious"]}')
        ll.addWidget(self.problems)
        self.status = label('', muted=True, small=True, wrap=True)
        ll.addWidget(self.status)
        bl = QHBoxLayout()
        self.b_export = button('Export…', None, 'export', 'campaign.json, the CSV tables, the figures (PNG) and an '
                                                          'HTML report')
        self.b_export.clicked.connect(lambda: self.export())
        bl.addWidget(self.b_export)
        bl.addStretch(1)
        self.b_cancel = button('Cancel', 'ghost', None, 'Stop after tidying CARLA up (what is done is kept)')
        self.b_cancel.clicked.connect(lambda: self.cancel())
        bl.addWidget(self.b_cancel)
        self.b_run = button('Run', 'primary', 'play', 'Drive what is not driven yet, check what is not checked, '
                                                      'analyse (a background job; resumes a stopped campaign)')
        self.b_run.clicked.connect(lambda: self.run())
        bl.addWidget(self.b_run)
        ll.addLayout(bl)
        ll.addStretch(1)
        return w

    # ------------------------------------------------------------------ right: the results
    def _right(self):
        w = QWidget()
        rl = QVBoxLayout(w)
        rl.setContentsMargins(8, 6, 8, 8)
        rl.setSpacing(6)
        tiles = QHBoxLayout()
        self.t_drives = Tile('Drives', 'Driven (or reused) of the planned drives')
        self.t_found = Tile('Smallest rotation found', 'The drive-log check\'s smallest rotation found the target '
                                                       'share of the time (empirical; the best camera)')
        self.t_fa = Tile('False alarms', 'Checks flagging a sensor on clean drives, of the checks that can judge on '
                                         'CARLA (the as-built check\'s mount verdict cannot: CARLA\'s car is another '
                                         'body; its flags are listed in the False alarms tab, marked)')
        self.t_time = Tile('CARLA time', 'Wall time of the drives done')
        for t in (self.t_drives, self.t_found, self.t_fa, self.t_time):
            tiles.addWidget(t)
        rl.addLayout(tiles)
        self.tabs = QTabWidget()
        # detectability
        d = QWidget()
        dl = QVBoxLayout(d)
        dl.setContentsMargins(0, 4, 0, 0)
        pick = QHBoxLayout()
        pick.addWidget(label('Check', ink2=True))
        self.check_pick = QComboBox()
        self.check_pick.currentIndexChanged.connect(lambda _i: self._show_curves())
        pick.addWidget(self.check_pick, 1)
        self.axis_pick = SegmentedControl([('any', 'Axes pooled'), ('each', 'Per axis')], 'any')
        self.axis_pick.changed.connect(lambda _k: self._show_curves())
        pick.addWidget(self.axis_pick)
        dl.addLayout(pick)
        self.chart = CurveChart()
        dl.addWidget(self.chart, 1)
        self.table_thresholds = _table(TH_COLS, 0)
        dl.addWidget(self.table_thresholds, 1)
        self.tabs.addTab(d, TABS[0])
        self.table_fa = _table(FA_COLS, 0)
        self.tabs.addTab(self.table_fa, TABS[1])
        self.table_acc = _table(ACC_COLS, 0)
        self.tabs.addTab(self.table_acc, TABS[2])
        wx = QWidget()
        wl = QVBoxLayout(wx)
        wl.setContentsMargins(0, 4, 0, 0)
        wp = QHBoxLayout()
        wp.addWidget(label('Class', ink2=True))
        self.wx_class = QComboBox()
        self.wx_class.currentIndexChanged.connect(lambda _i: self._show_weather_chart())
        wp.addWidget(self.wx_class, 1)
        wl.addLayout(wp)
        self.wx_chart = FactorChart()
        wl.addWidget(self.wx_chart, 1)
        self.table_wx = _table(WX_COLS, 0)
        wl.addWidget(self.table_wx, 1)
        self.tabs.addTab(wx, TABS[3])
        self.rigs_text = QPlainTextEdit()
        self.rigs_text.setReadOnly(True)
        self.rigs_text.setFont(mono(11))
        self.tabs.addTab(self.rigs_text, TABS[4])
        self.table_drives = _table(DRIVE_COLS, 4)
        self.tabs.addTab(self.table_drives, TABS[5])
        self.notes = QPlainTextEdit()
        self.notes.setReadOnly(True)
        self.tabs.addTab(self.notes, TABS[6])
        rl.addWidget(self.tabs, 1)
        return w

    # ------------------------------------------------------------------ settings
    @property
    def settings(self) -> SC.CampaignSettings:
        return SC.settings_of(self.ctx.store.project)

    def _store(self, st: SC.CampaignSettings, text) -> bool:
        new = SC.CampaignSettings.from_dict(st.to_dict()).to_dict()
        if new == (getattr(self.ctx.store.project.analysis, 'campaign', None) or {}):
            self._sync_fields()
            return False

        def fn(p):
            p.analysis.campaign = new
        self.ctx.store.edit('analysis', fn, f'Simulation campaign · {text}')
        return True

    def _set(self, text, **kw) -> bool:
        if self._syncing:
            return False
        d = self.settings.to_dict()
        d.update(kw)
        if 'template' not in kw:
            d['template'] = 'custom' if d.get('template') not in SC.TEMPLATES or kw else d['template']
        return self._store(SC.CampaignSettings.from_dict(d), text)

    def _set_base_key(self, key, value) -> bool:
        if self._syncing:
            return False
        base = dict(self.settings.base)
        base[key] = value
        return self._set(f'{key} {value}', base=base)

    def set_base(self, values: dict) -> bool:
        base = dict(self.settings.base)
        base.update(values)
        return self._set('drive settings', base=base)

    def set_fault_sets(self, sets: list) -> bool:
        return self._set('faults', fault_sets=list(sets))

    def set_out_dir(self, path) -> bool:
        return self._set('folder', out_dir=os.path.abspath(os.path.expanduser(path)) if str(path).strip() else '')

    def apply_template(self, name) -> bool:
        if name not in SC.TEMPLATES:
            return False
        st = SC.template(name, self.ctx.store.project)
        old = self.settings
        if old.out_dir and (getattr(self.ctx.store.project.analysis, 'campaign', None) or {}):
            st.out_dir = ''
        port = dict(old.base).get('port')
        if port:
            st.base['port'] = port
        return self._store(st, f'template {SC.TEMPLATES[name][0]}')

    def add_fault_set(self) -> bool:
        var = self.ctx.store.project.variant
        cams = [c.tag for c in var.cameras if c.visible]
        if not cams:
            return False
        sets = list(self.settings.fault_sets) + [{'kind': 'rotation', 'tags': cams[:1], 'axes': ['yaw', 'pitch',
                                                                                                  'roll'],
                                                  'magnitudes': [0.1, 0.2, 0.3, 0.5, 1.0], 'signs': 'alternate'}]
        return self.set_fault_sets(sets)

    def remove_fault_set(self, row=None) -> bool:
        row = self.faults.currentRow() if row is None else row
        sets = list(self.settings.fault_sets)
        if not 0 <= row < len(sets):
            return False
        sets.pop(row)
        return self.set_fault_sets(sets)

    def _faults_edited(self):
        if self._syncing:
            return
        sets = []
        for r in range(self.faults.rowCount()):
            def t(c):
                it = self.faults.item(r, c)
                return it.text().strip() if it is not None else ''
            d = {'kind': t(0), 'tags': _parse_list(t(1), False), 'axes': _parse_list(t(2), False) or None,
                 'magnitudes': _parse_list(t(3)), 'signs': t(4) or 'alternate'}
            if d['axes'] is None:
                d.pop('axes')
            sets.append(d)
        clean = [f for f in (SC.clean_fault_set(x) for x in sets) if f]
        if len(clean) != len(sets):
            self.status_text = 'A fault set was not valid and is left out (kind, sensors and sizes are needed).'
        self.set_fault_sets(clean)

    def _lists_changed(self, which):
        if self._syncing:
            return
        lw = self.var_list if which == 'variants' else self.wx_list
        vals = [lw.item(i).data(Qt.ItemDataRole.UserRole) for i in range(lw.count())
                if lw.item(i).checkState() == Qt.CheckState.Checked]
        self._set(which, **{which: vals})

    def _evals_changed(self):
        if self._syncing:
            return
        self._set('checks', evaluators=[e for e, cb in self.eval_box.items() if cb.isChecked()])

    def _pick_out(self):
        d = QFileDialog.getExistingDirectory(self, 'Campaign folder')
        if d:
            self.set_out_dir(d)

    def _sync_fields(self):
        st = self.settings
        p = self.ctx.store.project
        self._syncing = True
        try:
            self.name_edit.setText(st.name)
            self.seeds_edit.setText(', '.join(str(s) for s in st.seeds))
            i = self.tmpl.findData(st.template)
            if i >= 0:
                self.tmpl.setCurrentIndex(i)
            sim = SC.base_sim(p, st)
            k = self.scen.findData(sim.scenario.get('name', 'free_drive'))
            self.scen.setCurrentIndex(max(k, 0))
            self.var_list.clear()
            for v in p.variants:
                if not any(c.visible for c in v.cameras):
                    continue
                it = QListWidgetItem(v.name + (' · active' if v is p.variant else ''))
                it.setData(Qt.ItemDataRole.UserRole, v.uid)
                it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                it.setCheckState(Qt.CheckState.Checked if v.uid in st.variants else Qt.CheckState.Unchecked)
                self.var_list.addItem(it)
            self.wx_list.clear()
            from ...core import simrig as SR
            from ...core import weather as WX
            for c in WX.PRESETS:
                key = f'vantage:{c.id}'
                it = QListWidgetItem(f'{c.label()}  (Vantage)')
                it.setData(Qt.ItemDataRole.UserRole, key)
                it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                it.setCheckState(Qt.CheckState.Checked if key in st.weathers else Qt.CheckState.Unchecked)
                self.wx_list.addItem(it)
            for w in SR.CARLA_WEATHERS:
                it = QListWidgetItem(f'{w}  (CARLA preset)')
                it.setData(Qt.ItemDataRole.UserRole, w)
                it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                it.setCheckState(Qt.CheckState.Checked if w in st.weathers else Qt.CheckState.Unchecked)
                self.wx_list.addItem(it)
            self.faults.setRowCount(len(st.fault_sets))
            for r, f in enumerate(st.fault_sets):
                vals = (f['kind'], ', '.join(f['tags']), ', '.join(f.get('axes') or []),
                        ', '.join(f'{m:g}' for m in f['magnitudes']), f.get('signs', ''))
                for c, v in enumerate(vals):
                    self.faults.setItem(r, c, QTableWidgetItem(v))
            self.clean_cb.setChecked(st.clean)
            self.multi_cb.setChecked(st.multiplex)
            for key, _t, _u, _d, _s, _tip, dflt in BASE_FIELDS:
                v = st.base.get(key, getattr(sim, key, dflt))
                self.base_fields[key].set_value(float(v))
            for e, cb in self.eval_box.items():
                cb.setChecked(e in st.evaluators)
            self.ab_on.set_value(st.asbuilt_on)
            self.target.set_value(st.target)
            self.server.set_value(st.server)
            self.out_edit.setText(st.out_dir)
        finally:
            self._syncing = False
        self._show_plan()

    def _show_plan(self):
        p = self.ctx.store.project
        st = self.settings
        try:
            pl = SC.plan(p, st)
            est = SC.estimate(p, st, pl)
        except Exception as e:                    # noqa: BLE001 — a design that cannot be planned: say why
            self.cost.setText(f'Cannot plan: {e}')
            return
        self._plan = pl
        self.cost.setText(est['text'] + f' · cost basis: {est["basis"]}')
        self.problems.setText(' '.join(pl.problems))
        self.problems.setVisible(bool(pl.problems))
        self._fill_drives(pl)

    def _fill_drives(self, pl):
        res = self.result
        t = self.table_drives
        t.setRowCount(len(pl.drives))
        out = SC.campaign_dir(self.ctx.store.project, self.settings, workdir.get(self.ctx.settings))
        for r, d in enumerate(pl.drives):
            s = res.states.get(d['id']) if res is not None else None
            status = s.status if s else ('done' if SC._drive_done(out, d) is not None else 'to drive')
            _cell(t, r, 0, d['id'])
            _cell(t, r, 1, d['role'])
            _cell(t, r, 2, str(d['seed']), True)
            _cell(t, r, 3, f'{d["variant"]} · {d["weather"]} · {d["scenario"]["name"]}')
            _cell(t, r, 4, '; '.join(c['label'] for c in d['conditions']) or 'clean')
            col = QColor(STATUS['serious']) if status == 'failed' else None
            _cell(t, r, 5, status, tip=(s.error if s and s.error else None), color=col)
            _cell(t, r, 6, _g(s.wall_s, 0, 's') if s and s.wall_s else '—', True)

    # ------------------------------------------------------------------ state
    def _key(self, p=None):
        p = p or self.ctx.store.project
        return json.dumps(SC.settings_of(p).to_dict(), sort_keys=True)

    @property
    def stale(self) -> bool:
        return self.result is not None and self._for is not None and self._for[1] != self._key()

    def _on_change(self, *_):
        if not self.isVisible():
            self._dirty = True
            return
        self._sync_fields()
        self.refresh()

    def _on_replaced(self, *_):
        self._gen += 1
        self.result, self._for, self.busy = None, None, False
        self.status_text = ''
        self._sync_fields()
        self.refresh()

    def _retheme(self, *_):
        self.update()

    def showEvent(self, e):
        super().showEvent(e)
        self.activate()

    def activate(self):
        if self._dirty:
            self._dirty = False
            self._sync_fields()
        self.refresh()

    # ------------------------------------------------------------------ jobs
    def _engine(self):
        win = self.ctx.window
        if win is not None and hasattr(win, '_shared_engine'):
            try:
                return win._shared_engine()
            except Exception:                     # noqa: BLE001 — a page without the main window's engine
                pass
        return None

    def run(self, plan=None) -> bool:
        if self.busy:
            return False
        if isinstance(plan, dict):              # reached from a Copilot plan: only with its approval record
            from ...assistant import copilot as CP
            ok, why = CP.gate(self.ctx.store.project, plan, 'campaign', self.settings.to_dict())
            if not ok:
                self.status_text = why
                self.refresh()
                return False
        win = self.ctx.window
        other = win.carla_user('campaign') if win is not None and hasattr(win, 'carla_user') else ''
        if other:                                     # one client steps a synchronous CARLA world
            self.status_text = other
            self.refresh()
            return False
        p = self.ctx.store.project
        st = self.settings
        key = (p, self._key(p))
        runner, detector, extra = self.runner, self.detector, self.cancel_when
        eng = self._engine()
        self._gen += 1
        gen = self._gen
        self.busy = True
        self.status_text = 'Running…'

        def work(progress, cancel):
            def stop():
                return cancel() or (extra is not None and extra())
            return key, SC.run(p, eng, st, progress=progress, cancel=stop, runner=runner, detector=detector)
        win = self.ctx.window
        if win is not None and hasattr(win, 'run_job'):
            ok = win.run_job(JOB, 'Simulation campaign', work, lambda out: self._thread_done(gen, out),
                             lambda m: self._failed(gen, m))
        else:
            def thread():
                try:
                    out = work(lambda f, t='': self.sig.progress.emit(gen, f, t), lambda: gen != self._gen)
                    self.sig.done.emit(gen, out)
                except Exception as e:             # noqa: BLE001
                    try:
                        self.sig.failed.emit(gen, f'{type(e).__name__}: {e}\n{traceback.format_exc()}')
                    except RuntimeError:
                        pass
            threading.Thread(target=thread, daemon=True).start()
            ok = True
        if not ok:
            self.busy = False
            self.status_text = 'Another campaign is running.'
        self.refresh()
        return ok

    def cancel(self):
        win = self.ctx.window
        j = getattr(win, '_jobs', {}).get(JOB) if win is not None else None
        if j is not None:
            j['cancel'].set()
        elif self.busy:
            self._gen += 1
            self.busy = False
            self.status_text = 'Cancelled'
            self.refresh()

    def _progress(self, gen, f, text):
        if gen == self._gen and text:
            self.status.setText(f'{f * 100:.0f} % · {text}')

    def _thread_done(self, gen, out):
        if gen != self._gen:
            return
        self.busy = False
        key, res = out if out is not None else (None, None)
        if res is None:
            self.status_text = 'Cancelled: what was driven and checked is kept; Run resumes.'
            self._sync_fields()
            self.refresh()
            return
        if key[0] is not self.ctx.store.project:
            self.refresh()
            return
        self.result, self._for = res, key
        n_ok = sum(s.status in ('ok', 'reused') for s in res.states.values())
        self.status_text = (f'{n_ok} of {len(res.plan.drives)} drives, analysed in {res.seconds:.0f} s' +
                            ('' if res.complete else ' (incomplete: Run resumes)'))
        self.ctx.store.project.add_log('campaign', f'{res.settings.name}: {n_ok} drives')
        self.ctx.msg(f'Simulation campaign · {self.status_text}', 5000)
        self._sync_fields()
        self.refresh()

    def _failed(self, gen, msg):
        if gen != self._gen:
            return
        self.busy = False
        self.status_text = 'Failed: ' + msg.splitlines()[0]
        self.refresh()

    def export(self, folder=None) -> list:
        res = self.result
        if res is None:
            return []
        if folder is None:
            folder = QFileDialog.getExistingDirectory(self, 'Export the campaign into')
            if not folder:
                return []
        from ...core import provenance as PV
        paths = PV.stamp_paths(SC.export(res, self.ctx.store.project, folder), self.ctx.store.project, folder)
        self.ctx.msg(f'Exported the campaign to {folder}', 5000)
        return paths

    def summary(self) -> dict | None:
        return SC.summary(self.result) if self.result is not None else None

    # ------------------------------------------------------------------ showing the result
    def refresh(self):
        self.b_run.setEnabled(not self.busy)
        self.b_cancel.setVisible(self.busy)
        self.b_export.setEnabled(self.result is not None and not self.busy)
        txt = self.status_text
        if self.stale and not self.busy:
            txt = (txt + ' · ' if txt else '') + 'The design changed since the run: Run again (done drives are kept).'
        if self.result is None and not txt:
            txt = EMPTY
        self.status.setText(txt)
        self._show()

    def _show(self):
        res = self.result
        for t in (self.table_thresholds, self.table_fa, self.table_acc, self.table_wx):
            t.setRowCount(0)
        if res is None:
            for tile in (self.t_drives, self.t_found, self.t_fa, self.t_time):
                tile.set('—')
            self.chart.set_data([], 0.9)
            self.wx_chart.set_data([])
            self.rigs_text.setPlainText('')
            self.notes.setPlainText('\n'.join(SC.ASSUMPTIONS))
            self.check_pick.blockSignals(True)
            self.check_pick.clear()
            self.check_pick.blockSignals(False)
            return
        det = res.detectability
        n_ok = sum(s.status in ('ok', 'reused') for s in res.states.values())
        self.t_drives.set(f'{n_ok}/{len(res.plan.drives)}', '', 'complete' if res.complete else 'Run resumes',
                          'good' if res.complete else 'warning')
        rot = [t for t in det['thresholds'] if t['check'] == 'drivelog.extrinsics' and t['axis'] in ('', 'any')]
        best = min((t['empirical'] for t in rot if t['empirical'] is not None), default=None)
        fit = min((t['fit'] for t in rot if t['fit'] is not None and not t['extrapolated']), default=None)
        if best is not None:
            self.t_found.set(_g(best, 2), '°', f'{det["target"] * 100:.0f} % of the time (tested sizes)')
        elif fit is not None:
            self.t_found.set(_g(fit, 2), '°', f'{det["target"] * 100:.0f} % of the time (fit; no tested size reached '
                                              'it every time)', 'warning')
        else:
            self.t_found.set('—', '', 'none found reliably' if rot else 'no rotation faults')
        nfa = sum(1 for f in det['false_alarms'] if f['k'] and f.get('applicable', True))
        nna = sum(1 for f in det['false_alarms'] if f['k'] and not f.get('applicable', True))
        self.t_fa.set(str(nfa), 'checks', 'on clean drives' + (f' · {nna} not applicable on CARLA' if nna else ''),
                      'warning' if nfa else 'good')
        wall = sum(s.wall_s for s in res.states.values())
        self.t_time.set(f'{wall / 60:.0f}', 'min', f'{len(res.plan.drives)} drives')
        # detectability
        self.check_pick.blockSignals(True)
        cur = self.check_pick.currentData()
        self.check_pick.clear()
        keys = []
        for c in det['curves']:
            k = (c['check'], c['kind'])
            if k not in keys:
                keys.append(k)
        keys.sort(key=lambda k: (k[0] not in SC.RELEVANT.get(k[1], ()), k))
        for check, kind in keys:
            self.check_pick.addItem(f'{SC.CHECKS[check][0]} · {kind.replace("_", " ")}'
                                    + ('' if SC.applicable(check) else ' (not applicable on CARLA)'), f'{check}|{kind}')
            if not SC.applicable(check):
                self.check_pick.setItemData(self.check_pick.count() - 1, SC.NOT_APPLICABLE[check],
                                            Qt.ItemDataRole.ToolTipRole)
        i = self.check_pick.findData(cur)
        self.check_pick.setCurrentIndex(max(i, 0))
        self.check_pick.blockSignals(False)
        self._show_curves()
        t = self.table_thresholds
        rows = det['thresholds']
        t.setRowCount(len(rows))
        for r, th in enumerate(rows):
            _cell(t, r, 0, SC.CHECKS[th['check']][0])
            _cell(t, r, 1, th['tag'])
            _cell(t, r, 2, th['kind'].replace('_', ' '))
            _cell(t, r, 3, th['axis'] or '—')
            _cell(t, r, 4, f'{th["floor"] * 100:.0f} %', True)
            _cell(t, r, 5, _g(th.get('clean_stat'), 2, th.get('stat_unit') or ''), True)
            _cell(t, r, 6, _g(th['empirical'], 3, th['unit']) if th['empirical'] is not None else
                  f'> {max(th["tested"]):g} {th["unit"]}', True)
            _cell(t, r, 7, (_g(th['fit'], 3, th['unit']) + (' (extrapolated)' if th['extrapolated'] else ''))
                  if th['fit'] is not None else '—', True)
            ci = th['fit_ci']
            _cell(t, r, 8, '—' if not ci or ci[0] is None else
                  f'{_g(ci[0], 3)} – {_g(ci[1], 3) if ci[1] is not None else "beyond the sizes tried"}', True)
        # false alarms and cross-talk
        t = self.table_fa
        # clean drives, then cross-talk; in each the checks that can judge first, those that cannot on CARLA after
        # them, greyed, their flags no warning
        rows = [(f, 'clean drives') for f in det['false_alarms']] + [(f, 'unfaulted, faulty drives')
                                                                     for f in det['crosstalk']]
        rows.sort(key=lambda fw: (fw[1] != 'clean drives', not fw[0].get('applicable', True)))
        t.setRowCount(len(rows))
        muted = THEME.color('muted')
        for r, (f, what) in enumerate(rows):
            ok = f.get('applicable', True)
            why = None if ok else f.get('not_applicable') or SC.NOT_APPLICABLE.get(f['check'])
            col = None if ok else muted
            _cell(t, r, 0, SC.CHECKS[f['check']][0], tip=why, color=col)
            _cell(t, r, 1, f['tag'], color=col)
            _cell(t, r, 2, what, color=col)
            _cell(t, r, 3, str(f['k']), True, color=(QColor(STATUS['warning']) if f['k'] else None) if ok else col)
            _cell(t, r, 4, str(f['n']), True, color=col)
            _cell(t, r, 5, _pc(f['p'], f['ci']), True, color=col)
            _cell(t, r, 6, str(f['undecided']), True, color=col)
            _cell(t, r, 7, 'judges' if ok else 'not applicable', tip=why, color=col).setTextAlignment(
                Qt.AlignmentFlag.AlignCenter)
        # accuracy
        t = self.table_acc
        rows = det['accuracy']
        t.setRowCount(len(rows))
        for r, a in enumerate(rows):
            _cell(t, r, 0, SC.CHECKS[a['check']][0])
            _cell(t, r, 1, a['tag'])
            _cell(t, r, 2, a['kind'].replace('_', ' '))
            _cell(t, r, 3, a['axis'] or '—')
            _cell(t, r, 4, str(a['absolute']['n']), True)
            _cell(t, r, 5, _g(a['absolute']['bias'], 3, a['unit']), True)
            _cell(t, r, 6, _g(a['absolute']['rms'], 3, a['unit']), True)
            _cell(t, r, 7, _g(a['paired']['rms'], 3, a['unit']), True)
        # weather
        wx = res.weather
        self.wx_class.blockSignals(True)
        cur = self.wx_class.currentData()
        self.wx_class.clear()
        if wx:
            classes = sorted({c for r in wx['rows'] for c, v in r['classes'].items() if v['n']})
            for c in classes:
                self.wx_class.addItem(c.replace('_', ' '), c)
            with_r50 = [c for c in classes if (wx['rows'][0]['classes'].get(c) or {}).get('r50_m')]
            first = 'pedestrian' if 'pedestrian' in with_r50 else (with_r50[0] if with_r50 else
                                                                   ('car' if 'car' in classes else None))
            i = self.wx_class.findData(cur if cur else first)
            self.wx_class.setCurrentIndex(max(i, 0))
            rows = [(r, c, v) for r in wx['rows'] for c, v in r['classes'].items() if v['n']]
            t = self.table_wx
            t.setRowCount(len(rows))
            for k, (r, c, v) in enumerate(rows):
                lid = next((s.get('points_ratio') for s in r['returns'].values() if s['kind'] == 'lidar'), None)
                _cell(t, k, 0, r['weather'])
                _cell(t, k, 1, c.replace('_', ' '))
                _cell(t, k, 2, str(v['n']), True)
                _cell(t, k, 3, _pc(v['recall'], v['ci']), True)
                _cell(t, k, 4, _g(v.get('r50_m'), 1, 'm'), True)
                _cell(t, k, 5, _g(v.get('r50_factor'), 2), True)
                _cell(t, k, 6, _g(v.get('vantage_factor'), 2), True)
                _cell(t, k, 7, _g(lid, 3), True)
        self.wx_class.blockSignals(False)
        self._show_weather_chart()
        # rigs
        if res.rigs:
            self.rigs_text.setPlainText(json.dumps(res.rigs.get('comparison') or res.rigs, indent=1,
                                                   default=str)[:20000])
        else:
            self.rigs_text.setPlainText('One variant: no rig comparison (tick two variants, or use Rig A vs B).')
        self.notes.setPlainText('\n\n'.join(['\n'.join(SC.headline(res))] + list(res.notes) +
                                            ['Assumptions:'] + list(SC.ASSUMPTIONS) + ['References:'] +
                                            list(SC.REFERENCES)))
        self._fill_drives(res.plan)

    def _show_curves(self):
        res = self.result
        if res is None:
            self.chart.set_data([], 0.9)
            return
        key = self.check_pick.currentData()
        if not key:
            self.chart.set_data([], res.detectability['target'])
            return
        check, kind = key.split('|')
        pooled = self.axis_pick.value() == 'any'
        cs = [c for c in res.detectability['curves'] if c['check'] == check and c['kind'] == kind and
              ((c['axis'] in ('', 'any')) if pooled else (c['axis'] not in ('any',)))]
        palette = [QColor(x) for x in ('#3478f6', '#e8730c', '#2f9e44', '#c2255c', '#7048e8', '#0c8599', '#868e96')]
        curves = []
        for i, c in enumerate(cs):
            name = c['tag'] + (f' {c["axis"]}' if c['axis'] not in ('', 'any') else '')
            curves.append((name, palette[i % len(palette)], c['unit'], c['points'], c['false_alarm']))
        self.chart.title = SC.CHECKS[check][0]
        self.chart.set_data(curves, res.detectability['target'])

    def _show_weather_chart(self):
        res = self.result
        if res is None or not res.weather:
            self.wx_chart.set_data([])
            return
        c = self.wx_class.currentData()
        rows = []
        for r in res.weather['rows']:
            v = r['classes'].get(c) or {}
            rows.append((r['weather'], v.get('recall'), v.get('r50_factor'), v.get('vantage_factor'),
                         v.get('recall_ratio')))
        self.wx_chart.title = f'Weather · {c or ""}'
        self.wx_chart.set_data(rows)
