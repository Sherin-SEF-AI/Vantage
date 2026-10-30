"""Calibrate ▸ Online self-calibration: the time-varying extrinsics of every camera over a whole recorded drive (a ROS 2
bag or a nuScenes folder) from the drive's own content: depth edges against image edges, the own vehicle's outline and
the ground seen by two cameras, per window, joined by a random walk; permanent steps (a knock) told from reversible
flex (with speed, braking, turning or the time of day). A timeline per camera (yaw, pitch and roll with σ bands and the
events), an events table, the final segment applied as an as-built variant, and the time-indexed calibration exported
as JSON, a rewritten nuScenes folder and a ROS 2 /tf sidecar bag."""
from __future__ import annotations

import json
import os
import threading
import traceback

import numpy as np
from PyQt6.QtCore import QObject, QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPainterPath, QPen, QPolygonF
from PyQt6.QtWidgets import (QAbstractItemView, QComboBox, QFileDialog, QGridLayout, QHBoxLayout, QHeaderView,
                             QProgressBar, QScrollArea, QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout,
                             QWidget)

from ...core import drivelog as DL
from ...core import online_calib as OC
from ..theme import THEME, mono, sans
from ..widgets import ROW_H, ScrubSpinBox, StatusLine, ToggleSwitch, button, hline, label

JOB = 'online_calib'
EMPTY = 'Open a recorded drive (a ROS 2 bag or a nuScenes folder), then Run.'
SEV_TONE = {'critical': 'critical', 'serious': 'serious', 'warning': 'warning', 'note': None, 'good': 'good'}
# (key, label, unit, decimals, step, tip)
FIELDS = (
    ('window_s', 'Window', 's', 1, 1.0, 'Length of a window: the corrections are taken as constant within it'),
    ('stride_s', 'Stride', 's', 1, 1.0, 'A window every this many seconds (shorter than the window: they overlap)'),
    ('frames_per_window', 'Frames per window', '', 0, 1.0,
     'Instants sampled per window length: one image of every camera and the lidar sweeps around it (the outline '
     'needs 4)'),
    ('search_deg', 'Search range', '°', 1, 0.5, 'Corrections searched within ± this in each window'),
    ('tolerance_deg', 'Tolerance', '°', 2, 0.05, 'A knock beyond this needs a new calibration (batch exit code 3)'),
    ('min_step_deg', 'Smallest step', '°', 2, 0.05, 'Steps smaller than this are not reported'),
    ('flex_min_deg', 'Smallest flex', '°', 2, 0.05, 'Flex smaller than this (half its range over the drive) is not '
                                                     'reported'),
    ('walk_deg_per_min', 'Random walk', '°/√min', 3, 0.01,
     'How far the corrections may drift between windows: σ of the drift over one minute'),
    ('bic_margin', 'BIC margin', '', 1, 1.0,
     'A step or a flex term must lower the Bayesian information criterion by more than this (6: strong evidence)'),
    ('ground_range_m', 'Ground range', 'm', 1, 1.0, 'The ground factor uses ground up to this far from a camera'),
)
SWITCHES = (('edges', 'Depth edges', 'Each camera\'s rotation relative to the lidar from depth edges against image '
                                      'edges (yaw, pitch, roll where the edges pin them down)'),
            ('outline', 'Vehicle outline', 'Each camera\'s rotation relative to the body from the own vehicle\'s '
                                           'outline (where the body is in view and the vehicle moves). Only when the '
                                           'recorded vehicle is the project\'s body'),
            ('ground', 'Ground overlap', 'Two cameras\' relative rotation from the ground they both see, warped onto '
                                        'the lidar\'s ground plane. Needs a textured, flat road and a lens model that '
                                        'holds far off its axis'),
            ('fit_translation', 'Fit position', 'Also fit the cameras\' positions from the outline (weakly '
                                                'observable)'))
EVENT_COLS = (('Sensor', ''), ('Kind', 'Knock: a step that stays; excursion: a step that returns; flex: varies with '
                                       'speed, braking, turning or the time of day'),
              ('Axes', ''), ('Time', 'Drive time of the step (m:ss)'), ('Size', 'Step size, or flex amplitude (±)'),
              ('±', 'Standard error'), ('What', ''))


def _table(cols):
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
    t.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
    hh = t.horizontalHeader()
    for c in range(len(cols)):
        hh.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
    hh.setSectionResizeMode(len(cols) - 1, QHeaderView.ResizeMode.Stretch)
    return t


def _cell(t, r, c, text, num=False, color=None, tip=None):
    it = QTableWidgetItem(text)
    if num:
        it.setFont(mono(12))
        it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    if color is not None:
        it.setForeground(color)
    if tip:
        it.setToolTip(tip)
    t.setItem(r, c, it)
    return it


class TimelinePlot(QWidget):
    """One sensor over the drive: yaw, pitch and roll (and x, y, z when fitted), each the smoothed correction with its
    σ band, the windows' own estimates (dots, where the window constrains the axis), knocks (a red line), excursions
    (an amber span) and flex (labelled)."""

    ROW = 64

    def __init__(self, res, tag):
        super().__init__()
        self.res, self.tag = res, tag
        self.ser = res.series[tag]
        self.sensor = res.sensor(tag)
        self.setMinimumHeight(self.ROW * len(self.ser['axes']) + 26)

    def colour(self) -> QColor:
        s = self.sensor
        return THEME.cam(s['slot']) if s['kind'] == 'camera' else THEME.color('#0aa5b8')

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = self.width(), self.height()
        left, right = 92, 12
        x0, x1 = left, W - right
        dur = max(self.res.span['duration_s'], 1e-6)
        X = lambda t: x0 + (x1 - x0) * float(t) / dur                    # noqa: E731
        ser = self.ser
        tw = np.asarray(ser['t'], float)
        t, E, S = OC.plot_series(self.res, self.tag)
        Wn = np.asarray(ser['win'], float)
        C = np.asarray(ser['constrained'], bool)
        col = self.colour()
        p.setFont(sans(11))
        for i, ax in enumerate(ser['axes']):
            yt, yb = 4 + i * self.ROW, 4 + (i + 1) * self.ROW - 6
            unit = 'm' if ax in OC.TAXES else '°'
            k = 1000.0 if unit == 'm' else 1.0
            vals = np.r_[E[:, i] + S[:, i], E[:, i] - S[:, i], Wn[C[:, i], i]] * k
            vals = vals[np.isfinite(vals)]
            span = min(max(float(np.max(np.abs(vals))) if len(vals) else 0.0, 0.2 if unit == '°' else 5.0),
                       3.0 if unit == '°' else 100.0)
            Y = lambda v: 0.5 * (yt + yb) - (yb - yt) / 2 * float(np.clip(v * k / span, -1, 1))   # noqa: E731
            p.fillRect(QRectF(x0, yt, x1 - x0, yb - yt), THEME.color('raised'))
            p.setPen(QPen(THEME.color('strong'), 1))
            p.drawLine(QPointF(x0, Y(0.0)), QPointF(x1, Y(0.0)))
            p.setPen(col)
            p.drawText(QRectF(4, yt, left - 8, (yb - yt) / 2),
                       Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                       f'{self.tag} {ax}')
            p.setPen(THEME.color('muted'))
            p.setFont(mono(10))
            p.drawText(QRectF(4, 0.5 * (yt + yb), left - 8, (yb - yt) / 2),
                       Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                       f'±{span:g}{" mm" if unit == "m" else "°"}')
            p.setFont(sans(11))
            if len(t) > 1:
                band = QPolygonF([QPointF(X(a), Y(m + s)) for a, m, s in zip(t, E[:, i], S[:, i])] +
                                 [QPointF(X(a), Y(m - s)) for a, m, s in zip(t[::-1], E[::-1, i], S[::-1, i])])
                c2 = QColor(col)
                c2.setAlphaF(0.22)
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(c2)
                p.drawPolygon(band)
                path = QPainterPath(QPointF(X(t[0]), Y(E[0, i])))
                for a, m in zip(t[1:], E[1:, i]):
                    path.lineTo(QPointF(X(a), Y(m)))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.setPen(QPen(col, 1.6))
                p.drawPath(path)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(THEME.color('ink2'))
            for kk in range(len(tw)):
                if C[kk, i] and np.isfinite(Wn[kk, i]):
                    p.drawEllipse(QPointF(X(tw[kk]), Y(Wn[kk, i])), 2.0, 2.0)
            for ev in self.res.events:
                if ev['sensor'] != self.tag or ax not in ev['axes']:
                    continue
                if ev['kind'] == 'knock':
                    p.setPen(QPen(THEME.color('critical'), 2))
                    p.drawLine(QPointF(X(ev['t_s']), yt), QPointF(X(ev['t_s']), yb))
                elif ev['kind'] == 'excursion':
                    c3 = THEME.color('warning')
                    c3.setAlphaF(0.25)
                    p.setPen(Qt.PenStyle.NoPen)
                    p.setBrush(c3)
                    p.drawRect(QRectF(X(ev['t_s']), yt, max(X(ev['t_end_s']) - X(ev['t_s']), 1.0), yb - yt))
                else:
                    p.setPen(THEME.color('warning'))
                    p.drawText(QRectF(x0 + 4, yt, x1 - x0 - 8, 16), Qt.AlignmentFlag.AlignRight,
                               f'flex with {OC.COV_TEXT[ev["covariate"]]} ±{ev["amplitude"]:.2f}{unit}')
        yb = 4 + len(ser['axes']) * self.ROW
        p.setPen(THEME.color('muted'))
        p.setFont(mono(10))
        step = next(s for s in (0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 30, 60, 120, 300, 600, 1800, 3600, 1e9)
                    if dur / s <= 10)
        tt = 0.0
        while tt <= dur + 1e-9:
            p.drawLine(QPointF(X(tt), yb - 6), QPointF(X(tt), yb - 2))
            p.drawText(QRectF(X(tt) - 30, yb - 2, 60, 16), Qt.AlignmentFlag.AlignHCenter, OC.clock(tt))
            tt += step
        p.end()


class _Sig(QObject):
    done = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)
    progress = pyqtSignal(int, float, str)


class OnlineCalibPage(QWidget):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.result: OC.OnlineCalibResult | None = None
        self._scan = None
        self._for = None
        self._gen = 0
        self._cancel = threading.Event()
        self.busy = False
        self.status_text = ''
        self._dirty = True
        self._shown = False
        self.plots = []
        self.findings = []
        self.sig = _Sig()
        self.sig.done.connect(self._done)
        self.sig.failed.connect(self._failed)
        self.sig.progress.connect(lambda g, f, t: self._progress(f, t) if g == self._gen else None)
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

    # ------------------------------------------------------------------ left: the drive and the settings
    def _left(self):
        w = QWidget()
        w.setObjectName('panel')
        ll = QVBoxLayout(w)
        ll.setContentsMargins(8, 6, 8, 8)
        ll.setSpacing(4)
        top = QHBoxLayout()
        top.addWidget(label('Online self-calibration'))
        top.addStretch(1)
        top.addWidget(label('extrinsics over a whole drive', muted=True, small=True))
        ll.addLayout(top)
        ll.addWidget(hline())
        ll.addWidget(label('Follow every camera\'s rotation relative to the lidar through a recorded drive, window by '
                           'window, from what the drive itself shows: depth edges against image edges, the own '
                           'vehicle\'s outline and the ground two cameras share. Steps that stay (a knock) are told '
                           'from flex that comes and goes with speed, braking or turning.', muted=True, small=True,
                           wrap=True))
        ll.addSpacing(4)
        ll.addWidget(label('Drive', bold=True))
        self.path_label = label('No drive open', mono_=True, small=True)
        self.path_label.setMinimumWidth(10)
        ll.addWidget(self.path_label)
        row = QHBoxLayout()
        b = button('Open bag…', None, 'import', 'A ROS 2 bag file: .mcap or .db3 (for a rosbag2 folder use Open '
                                               'folder…)')
        b.clicked.connect(self._pick_file)
        row.addWidget(b)
        b = button('Open folder…', None, 'folder', 'A rosbag2 folder or a nuScenes folder')
        b.clicked.connect(self._pick_dir)
        row.addWidget(b)
        row.addStretch(1)
        ll.addLayout(row)
        vr = QHBoxLayout()
        a = label('Plan', ink2=True)
        a.setFixedWidth(60)
        vr.addWidget(a)
        self.variant = QComboBox()
        self.variant.setToolTip('The variant whose plan the corrections are relative to')
        self.variant.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.variant.setMinimumContentsLength(14)
        self.variant.activated.connect(self._variant_picked)
        vr.addWidget(self.variant, 1)
        ll.addLayout(vr)
        ll.addSpacing(4)
        ll.addWidget(label('Measurements', bold=True))
        self.switches = {}
        for key, text, tip in SWITCHES:
            sw = ToggleSwitch(text)
            sw.setToolTip(tip)
            sw.toggled.connect(lambda on, k=key, t=text: self._set(f'{t.lower()} {"on" if on else "off"}',
                                                                   **{k: bool(on)}))
            ll.addWidget(sw)
            self.switches[key] = sw
        ll.addSpacing(4)
        ll.addWidget(label('Windows and events', bold=True))
        g = QGridLayout()
        g.setVerticalSpacing(2)
        g.setHorizontalSpacing(6)
        g.setColumnStretch(2, 1)
        self.fields = {}
        for i, (key, text, unit, dec, step, tip) in enumerate(FIELDS):
            f = ScrubSpinBox('', 'plain', dec, unit, step=step, width=110)
            f.setToolTip(tip)
            f.committed.connect(lambda v, k=key, t=text, u=unit: self._set(f'{t.lower()} {v:g}{u}', **{k: v}))
            a = label(text, ink2=True)
            a.setFixedWidth(118)
            a.setToolTip(tip)
            g.addWidget(a, i, 0)
            g.addWidget(f, i, 1)
            self.fields[key] = f
        ll.addLayout(g)
        ll.addWidget(label(OC.ASSUMPTIONS, muted=True, small=True, wrap=True))
        self.status = label('', muted=True, small=True, wrap=True)
        ll.addWidget(self.status)
        self.bar = QProgressBar()
        self.bar.setTextVisible(False)
        self.bar.setRange(0, 1000)
        self.bar.hide()
        ll.addWidget(self.bar)
        bl = QHBoxLayout()
        b_def = button('Defaults', 'ghost', None, 'Back to the default settings (the drive and the plan stay)')
        b_def.clicked.connect(self._defaults)
        bl.addWidget(b_def)
        bl.addStretch(1)
        self.b_cancel = button('Cancel', None, None, 'Stop the calibration that is running')
        self.b_cancel.clicked.connect(self.cancel)
        bl.addWidget(self.b_cancel)
        self.b_run = button('Run', 'primary', None, 'Read the drive and follow the extrinsics through it')
        self.b_run.clicked.connect(lambda: self.run())
        bl.addWidget(self.b_run)
        ll.addLayout(bl)
        b2 = QHBoxLayout()
        self.b_apply = button('Apply final segment', None, None,
                              'A new variant with the last segment\'s precise corrections (one undo step)')
        self.b_apply.clicked.connect(lambda: self.apply_final())
        b2.addWidget(self.b_apply)
        b2.addStretch(1)
        self.b_export = button('Export…', None, 'export',
                               'The calibration as JSON (segments and interpolated, with σ), the timeline as CSV and '
                               'PNG, a nuScenes folder with one calibrated_sensor row per segment (nuScenes input) and '
                               'a ROS 2 sidecar bag with /tf over time')
        self.b_export.clicked.connect(lambda: self.export())
        b2.addWidget(self.b_export)
        ll.addLayout(b2)
        ll.addStretch(1)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        sc.setWidget(w)
        sc.setMinimumWidth(340)
        return sc

    # ------------------------------------------------------------------ right: timelines, events, findings
    def _right(self):
        w = QWidget()
        w.setObjectName('panel')
        rl = QVBoxLayout(w)
        rl.setContentsMargins(10, 8, 10, 8)
        rl.setSpacing(6)
        self.headline = label('', bold=True)
        rl.addWidget(self.headline)
        split = QSplitter(Qt.Orientation.Vertical)
        tl = QWidget()
        self.plot_lay = QVBoxLayout(tl)
        self.plot_lay.setContentsMargins(0, 0, 0, 0)
        self.plot_lay.setSpacing(8)
        self.plot_lay.addStretch(1)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setWidget(tl)
        split.addWidget(sc)
        bottom = QWidget()
        bl = QVBoxLayout(bottom)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.addWidget(label('Events', bold=True))
        self.events = _table(EVENT_COLS)
        bl.addWidget(self.events, 1)
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
        split.setSizes([620, 300])
        rl.addWidget(split, 1)
        return w

    # ------------------------------------------------------------------ settings
    @property
    def settings(self) -> OC.OnlineCalibSettings:
        return OC.settings_of(self.ctx.store.project)

    def _store(self, st: OC.OnlineCalibSettings, text) -> bool:
        new = OC.OnlineCalibSettings.from_dict(st.to_dict()).to_dict()
        if new == self.settings.to_dict():
            self._sync_fields()
            return False

        def fn(p):
            p.analysis.online_calib = new
        self.ctx.store.edit('analysis', fn, f'Online self-calibration · {text}')
        return True

    def _set(self, text, **kw):
        st = self.settings
        for k, v in kw.items():
            setattr(st, k, v)
        return self._store(st, text)

    def _defaults(self):
        st = self.settings
        self._store(OC.OnlineCalibSettings(path=st.path, variant_uid=st.variant_uid), 'default settings')

    def open_path(self, path) -> bool:
        """Open a bag or nuScenes folder: remember it in the settings (undoable)."""
        p = DL.norm_path(path)
        if not p or not os.path.exists(p):
            self.ctx.msg(f'Not found: {path}', 5000)
            return False
        self._set(f'drive {os.path.basename(p)}', path=p)
        self.refresh()
        return True

    def _pick_file(self):
        f, _ = QFileDialog.getOpenFileName(self, 'Open a ROS 2 bag', os.path.dirname(self.settings.path or ''),
                                           'ROS 2 bags (*.mcap *.db3);;All files (*)')
        if f:
            self.open_path(f)

    def _pick_dir(self):
        d = QFileDialog.getExistingDirectory(self, 'Open a rosbag2 or nuScenes folder',
                                             os.path.dirname(self.settings.path or ''))
        if d:
            self.open_path(d)

    def _variant_picked(self, i):
        uid = self.variant.itemData(i) or ''
        if uid != self.settings.variant_uid:
            v = self.ctx.store.project.variant_by_uid(uid)
            self._set(f'relative to {v.name if v else "the active variant"}', variant_uid=uid)

    def _sync_fields(self):
        st = self.settings
        for k, f in self.fields.items():
            f.set_value(float(getattr(st, k)))
        for k, sw in self.switches.items():
            sw.blockSignals(True)
            sw.setChecked(bool(getattr(st, k)))
            sw.blockSignals(False)
        p = self.ctx.store.project
        items = [(f'Active variant ({p.variant.name})', '')] + [(v.name, v.uid) for v in p.variants]
        if [(self.variant.itemText(i), self.variant.itemData(i)) for i in range(self.variant.count())] != items:
            self.variant.blockSignals(True)
            self.variant.clear()
            for text, uid in items:
                self.variant.addItem(text, uid)
            self.variant.blockSignals(False)
        self.variant.setCurrentIndex(next((i for i in range(self.variant.count())
                                           if self.variant.itemData(i) == st.variant_uid), 0))
        if st.path:
            self.path_label.setText(self.path_label.fontMetrics().elidedText(
                st.path, Qt.TextElideMode.ElideMiddle, max(self.path_label.width(), 300)))
            self.path_label.setToolTip(st.path)
        else:
            self.path_label.setText('No drive open')
            self.path_label.setToolTip('')

    def _inputs_key(self, p=None):
        p = p or self.ctx.store.project
        st = OC.settings_of(p)
        var = (p.variant_by_uid(st.variant_uid) if st.variant_uid else None) or p.variant
        return json.dumps({'s': st.to_dict(), 'v': var.to_dict(), 'lib': p.section('library'),
                           'dl': p.analysis.drivelog, 'ab': p.analysis.asbuilt, 'timing': p.analysis.timing},
                          sort_keys=True, default=str)

    @property
    def stale(self) -> bool:
        return self.result is not None and self._for is not None and self._for != self._inputs_key()

    def _on_change(self, *_):
        if self.isVisible():
            self._sync_fields()
            self.refresh()
        else:
            self._dirty = True

    def _on_replaced(self, *_):
        self._gen += 1
        self.result, self._scan, self._for, self.busy = None, None, None, False
        self.status_text = ''
        self._shown = False
        self._sync_fields()
        self.refresh()

    def _retheme(self, *_):
        self._shown = False
        self.refresh()

    def showEvent(self, e):
        super().showEvent(e)
        if self._dirty:
            self.activate()

    def activate(self):
        self._dirty = False
        self._sync_fields()
        self.refresh()

    # ------------------------------------------------------------------ run and cancel
    def run(self) -> bool:
        st = self.settings
        if not st.path:
            self.ctx.msg('Open a ROS 2 bag or a nuScenes folder first.', 4000)
            return False
        win = self.ctx.window
        if self.busy:
            return False
        p = self.ctx.store.project
        project = p.snapshot()
        eng = win._shared_engine() if win is not None and hasattr(win, '_shared_engine') else \
            self.ctx.analysis.engine.share()
        scan = self._scan if self._scan is not None and self._scan.key[0] == DL.norm_path(st.path) else None
        self._gen += 1
        gen = self._gen
        self._cancel = threading.Event()
        cancel_ev = self._cancel
        key = (p, self._inputs_key(p))
        self.busy = True
        self.status_text = 'Reading the drive…' if scan is None else 'Calibrating…'

        def work(progress, cancel):
            return OC.run(project, eng, OC.settings_of(project), None, progress,
                          lambda: cancel() or cancel_ev.is_set(), scan=scan)
        if win is not None and hasattr(win, 'run_job'):
            ok = win.run_job(JOB, 'Online self-calibration', work, lambda out: self._done(gen, (key, out)),
                             lambda m: self._failed(gen, m))
            if not ok:
                self.busy = False
                self.status_text = ''
                self.ctx.msg('An online self-calibration is still running: cancel it first.', 4000)
            self.refresh()
            return ok

        def thread():
            try:
                out = work(lambda f, text='': self.sig.progress.emit(gen, f, text), lambda: gen != self._gen)
                self.sig.done.emit(gen, (key, out))
            except RuntimeError:
                pass
            except Exception as e:
                try:
                    self.sig.failed.emit(gen, f'{type(e).__name__}: {e}\n{traceback.format_exc()}')
                except RuntimeError:
                    pass
        threading.Thread(target=thread, daemon=True).start()
        self.refresh()
        return True

    def cancel(self):
        """Stop the run in progress (its result is dropped)."""
        if not self.busy:
            return
        self._cancel.set()
        win = self.ctx.window
        job = getattr(win, '_jobs', {}).get(JOB) if win is not None else None
        if job is not None:
            job['cancel'].set()
        self.status_text = 'Cancelling…'
        self.refresh()

    def _progress(self, f, text):
        self.bar.setValue(int(f * 1000))
        if text:
            self.status_text = f'{text[:1].upper()}{text[1:]}…'
            self.status.setText(self.status_text)

    def _done(self, gen, out):
        if gen != self._gen:
            return
        self.busy = False
        key, res = out if out is not None else (None, None)
        if res is None or key[0] is not self.ctx.store.project:
            self.status_text = 'Cancelled' if res is None else ''
            self.refresh()
            return
        self.result, self._scan, self._for = res, getattr(res, '_scan', None), key[1]
        m = res.metrics
        self.status_text = (f'{m["knocks"]} knock{"s" * (m["knocks"] != 1)}, {m["excursions"]} '
                            f'excursion{"s" * (m["excursions"] != 1)}, {m["flex"]} flex · {m["windows"]} windows over '
                            f'{OC.clock(m["duration_s"])} · {m["seconds"]:.1f} s')
        self.refresh()
        self.ctx.msg(f'Online self-calibration: {self.status_text}', 5000)

    def _failed(self, gen, msg):
        if gen != self._gen:
            return
        self.busy = False
        self.status_text = 'Failed: ' + msg.splitlines()[0]
        self.refresh()

    # ------------------------------------------------------------------ show
    def refresh(self):
        st = self.settings
        self.b_run.setEnabled(bool(st.path) and not self.busy)
        self.b_cancel.setEnabled(self.busy)
        win = self.ctx.window
        self.bar.setVisible(self.busy and (win is None or not hasattr(win, 'run_job')))
        res = self.result
        self.b_export.setEnabled(res is not None)
        n = len(OC.final_corrections(res)) if res is not None else 0
        alive = res is not None and self.ctx.store.project.variant_by_uid(res.variant_uid) is not None
        self.b_apply.setEnabled(n > 0 and alive)
        self.b_apply.setText(f'Apply final segment ({n})' if n else 'Apply final segment')
        text = self.status_text or (EMPTY if not st.path else 'Ready: Run reads the drive and calibrates it.')
        self.status.setText(text + (' · stale: the plan or the settings changed since' if self.stale else ''))
        if res is self._shown:
            return
        self._shown = res
        self._fill_plots(res)
        self._fill_events(res)
        self._fill_findings(res)

    def _fill_plots(self, res):
        for w in self.plots:
            self.plot_lay.removeWidget(w)
            w.hide()
            w.deleteLater()
        self.plots = []
        if res is None:
            self.headline.setText('')
            return
        worst = res.worst()
        self.headline.setText(f'{len(res.windows)} windows, relative to {res.lidar or "the vehicle body"} · '
                              f'{os.path.basename(res.path)} · exit code {OC.exit_code(res)}')
        self.headline.setToolTip(worst)
        for s in res.sensors:
            pl = TimelinePlot(res, s['tag'])
            self.plot_lay.insertWidget(self.plot_lay.count() - 1, pl)
            self.plots.append(pl)

    def _fill_events(self, res):
        t = self.events
        t.setRowCount(0)
        if res is None:
            return
        t.setRowCount(len(res.events))
        kc = {'knock': THEME.color('critical'), 'excursion': THEME.color('warning'), 'flex': THEME.color('warning')}
        for r, e in enumerate(res.events):
            _cell(t, r, 0, e['sensor'])
            _cell(t, r, 1, e['kind'], color=kc.get(e['kind']))
            _cell(t, r, 2, ', '.join(e['axes']))
            if e['kind'] == 'flex':
                _cell(t, r, 3, '—', num=True)
                ax = e['axes'][0]
                u = 'm' if ax in OC.TAXES else '°'
                _cell(t, r, 4, f'±{e["amplitude"]:.3f}{u}', num=True)
                _cell(t, r, 5, f'{e["slope_sigma"]:.4f}', num=True, tip=f'σ of the slope ({e["slope_unit"]})')
            else:
                when = e['clock'] + (f' → {e["clock_end"]}' if e.get('clock_end') else '')
                _cell(t, r, 3, when, num=True, tip=f'resolution {e["resolution_s"]:.2f} s')
                _cell(t, r, 4, ', '.join(f'{v:+.3f}' for v in e['size'].values()), num=True)
                _cell(t, r, 5, ', '.join(f'{v:.3f}' for v in e['sigma'].values()), num=True)
            _cell(t, r, 6, e['message'], tip=e['message'])

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
        for n in res.notes:
            w = StatusLine(None, n)
            self.find_lay.insertWidget(self.find_lay.count() - 1, w)
            self.findings.append(w)

    def summary(self) -> dict | None:
        """The last run's compact figures (core.online_calib.summary) for the assistant; None before a run."""
        if self.result is None:
            return None
        return dict(OC.summary(self.result), stale=self.stale)

    # ------------------------------------------------------------------ actions
    def apply_final(self):
        """Add the last segment's corrections as a new as-built variant and switch to it, as one undo step."""
        res = self.result
        if res is None:
            return None
        v = OC.corrected_variant(self.ctx.store.project, res)
        if v is None:
            self.ctx.msg('No correction of the last segment is precise enough to apply.', 4000)
            return None
        state = v.to_dict()

        def fn(pj):
            from ...core.project import Variant
            pj.variants.append(Variant.from_dict(state))
            pj.active_uid = state['uid']
        self.ctx.store.edit('variants', fn, f'Online self-calibration · new variant {v.name}')
        self.ctx.msg(f'Added {v.name}: {v.note}', 6000)
        return v.uid

    def export(self, folder=None):
        res = self.result
        if res is None:
            return []
        if folder is None:
            folder = QFileDialog.getExistingDirectory(self, 'Export the online self-calibration into')
            if not folder:
                return []
        from ...core import provenance as PV
        try:
            paths = OC.export(res, self.ctx.store.project, folder)
        except (OSError, ValueError) as e:
            self.ctx.msg(f'Export failed: {e}', 6000)
            return []
        paths = PV.stamp_paths(paths, self.ctx.store.project, folder)
        self.ctx.msg(f'Exported the online self-calibration to {folder}', 5000)
        return paths
