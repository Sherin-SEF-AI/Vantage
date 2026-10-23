"""Calibrate ▸ Drive-log check: a recorded drive (a ROS 2 bag or a nuScenes folder) checked against the rig plan:
stream health, data rate against the recording budget, the measured sync against the timing plan, each camera's
rotation against the lidar from edge alignment, the GNSS and IMU; a new corrected variant from the confident
rotations. The log is read in a background job; editing the mapping or the thresholds checks again without reading
the log again."""
from __future__ import annotations

import json
import os
import threading
import traceback

import numpy as np
from PyQt6.QtCore import QObject, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (QAbstractItemView, QComboBox, QFileDialog, QGridLayout, QHBoxLayout, QHeaderView,
                             QProgressBar, QScrollArea, QSlider, QSplitter, QTableWidget, QTableWidgetItem, QTabWidget,
                             QVBoxLayout, QWidget)

from ...core import drivelog as DL
from ..drivelogview import SEV_TONE, OverlayView, TimelineView, sev_color, timeline_rows
from ..theme import STATUS, THEME, mono
from ..widgets import ROW_H, ScrubSpinBox, SegmentedControl, StatusLine, button, hline, label
from .recording_ws import Tile

JOB = 'drivelog'
EMPTY = 'Open a ROS 2 bag (a rosbag2 folder, an .mcap or a .db3 file) or a nuScenes folder, then Run.'
# (key, label, unit, decimals, step, tip)
FIELDS = (
    ('frames_per_camera', 'Frames per camera', '', 0, 1.0,
     'Instants spread over the drive at which one image of every camera and the lidar sweeps around it are kept for '
     'the rotation check (read again when changed)'),
    ('tolerance_deg', 'Rotation tolerance', '°', 2, 0.05, 'A camera rotated by more than this against the plan is '
                                                          'flagged'),
    ('search_deg', 'Search range', '°', 1, 0.5, 'Rotation corrections are searched within ± this'),
    ('drop_factor', 'Drop factor', '×', 2, 0.1, 'An interval this many times the nominal period counts as a drop'),
    ('sync_tol_ms', 'Sync tolerance', 'ms', 1, 0.5, 'Sync offsets and trigger skews beyond this are flagged'),
)
STREAM_COLS = (('Sensor', ''), ('Topic', ''), ('Msgs', 'Messages on the topic'),
               ('Hz', 'Measured rate: one over the median interval between header stamps'),
               ('Plan Hz', 'The module\'s frame rate or the model\'s rate'),
               ('Drops', 'Frames missing: intervals longer than the drop factor × the nominal period'),
               ('Gap ms', 'The longest interval'), ('Jitter ms', 'Standard deviation of the other intervals'),
               ('Latency ms', 'Log time minus header stamp (median); negative is a clock problem'),
               ('Size', 'Image size, or points per cloud'),
               ('Mbit/s', 'Data rate on the topic'), ('Plan', 'The recording budget\'s data rate for the sensor'),
               ('Status', ''))
SYNC_COLS = (('Camera', ''), ('Line', 'Trigger line (sync group)'), ('Frames', ''),
             ('Plan ms', 'Planned misalignment: exposure centre minus the sweep crossing the view'),
             ('Measured ms', 'Measured offset, the same way (median)'),
             ('Own ms', 'The deviation from the plan with the lidar\'s phase error taken out: the camera\'s own trigger '
                        'error'),
             ('p95 ms', '95th percentile of the absolute offset'), ('Spread ms', '5–95 % range of the deviation'),
             ('Skew ms', 'Stamp minus the nearest stamp of the line\'s first camera (median)'), ('Status', ''))
EXT_COLS = (('Camera', ''), ('Yaw °', 'Rotation correction added to the plan\'s yaw'), ('Pitch °', ''),
            ('Roll °', ''), ('± °', 'Largest standard error of the pinned-down axes (frames and image regions)'),
            ('Agree', 'Share of the frames whose own best correction is within the tolerance'),
            ('Frames', ''), ('Confidence', ''), ('Verdict', ''))


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
    return t


def _cell(t, r, c, text, num=False, color=None, tip=None, bold=False):
    it = QTableWidgetItem(text)
    if num:
        it.setFont(mono(12, QFont.Weight.DemiBold if bold else None))
        it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    if color is not None:
        it.setForeground(color)
    if tip:
        it.setToolTip(tip)
    t.setItem(r, c, it)
    return it


def _f(v, d=1, unit=''):
    if v is None:
        return '—'
    try:
        v = float(v)
    except (TypeError, ValueError):
        return '—'
    return f'{round(v, d) + 0.0:,.{d}f}{unit}' if np.isfinite(v) else '—'


class _Sig(QObject):
    done = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)
    progress = pyqtSignal(int, float, str)


class DriveLogPage(QWidget):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.result: DL.DriveLogResult | None = None
        self._scan = None                  # the last scan: used again while the log's files are unchanged
        self._info = None                  # (path, LogInfo or error text) of the open log, for the mapping table
        self._for = None                   # inputs key of the result
        self._gen = 0
        self.busy = False
        self.status_text = ''
        self._dirty = True
        self._syncing = False
        self._shown = False
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
        split.setSizes([390, 1210])
        lay.addWidget(split)
        ctx.store.changed.connect(self._on_change)
        ctx.store.project_replaced.connect(self._on_replaced)
        THEME.changed.connect(self._retheme)
        self._sync_fields()
        self.refresh()

    # ------------------------------------------------------------------ left: the log, the mapping, the settings
    def _left(self):
        w = QWidget()
        w.setObjectName('panel')
        ll = QVBoxLayout(w)
        ll.setContentsMargins(8, 6, 8, 8)
        ll.setSpacing(4)
        top = QHBoxLayout()
        top.addWidget(label('Drive-log check'))
        top.addStretch(1)
        top.addWidget(label('the plan against a recording', muted=True, small=True))
        ll.addLayout(top)
        ll.addWidget(hline())
        ll.addWidget(label('Load a recorded drive and check it against the rig plan: are the streams there at their '
                           'rates, how much data they made against the budget, are the cameras triggered in step '
                           'with the lidar sweep, and do the lidar points land on the right pixels (each camera\'s '
                           'rotation relative to the lidar).', muted=True, small=True, wrap=True))
        ll.addSpacing(4)
        ll.addWidget(label('Log', bold=True))
        self.path_label = label('No log open', mono_=True, small=True)
        self.path_label.setMinimumWidth(10)
        ll.addWidget(self.path_label)
        row = QHBoxLayout()
        b = self.b_bag = button('Open bag…', None, 'import', 'A ROS 2 bag file: .mcap or .db3 (for a rosbag2 folder '
                                                              'use Open folder…)')
        b.clicked.connect(self._pick_file)
        row.addWidget(b)
        b = self.b_dir = button('Open folder…', None, 'folder', 'A rosbag2 folder (metadata.yaml with .mcap or .db3 '
                                                                'files) or a nuScenes folder (its tables and '
                                                                'samples/sweeps)')
        b.clicked.connect(self._pick_dir)
        row.addWidget(b)
        row.addStretch(1)
        ll.addLayout(row)
        vr = QHBoxLayout()
        a = label('Plan', ink2=True)
        a.setFixedWidth(60)
        vr.addWidget(a)
        self.variant = QComboBox()
        self.variant.setToolTip('The variant the log is checked against')
        self.variant.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.variant.setMinimumContentsLength(14)             # long variant names are elided, the panel stays narrow
        self.variant.activated.connect(self._variant_picked)
        vr.addWidget(self.variant, 1)
        ll.addLayout(vr)
        ll.addSpacing(4)
        mh = QHBoxLayout()
        mh.addWidget(label('Topics', bold=True))
        mh.addStretch(1)
        self.b_reset = button('Automatic', 'ghost', None, 'Forget the edits of this log\'s mapping: every topic '
                                                          'mapped by the tag in its name or frame again')
        self.b_reset.clicked.connect(self._reset_mapping)
        mh.addWidget(self.b_reset)
        ll.addLayout(mh)
        self.topics = QTableWidget(0, 4)
        self.topics.setHorizontalHeaderLabels(['Topic', 'Type', 'Msgs', 'Sensor'])
        self.topics.verticalHeader().setVisible(False)
        self.topics.verticalHeader().setDefaultSectionSize(ROW_H + 2)
        self.topics.horizontalHeader().setFixedHeight(ROW_H)
        self.topics.setShowGrid(False)
        self.topics.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.topics.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        hh = self.topics.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for c in (1, 2, 3):
            hh.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        self.topics.setMinimumHeight(ROW_H * 6)
        ll.addWidget(self.topics, 1)
        ll.addSpacing(4)
        ll.addWidget(label('Settings', bold=True))
        g = QGridLayout()
        g.setVerticalSpacing(2)
        g.setHorizontalSpacing(6)
        g.setColumnStretch(2, 1)
        a = label('Camera stamps', ink2=True)
        a.setToolTip('What a camera\'s header stamp marks; the offset to the exposure centre uses the timing plan\'s '
                     'readout and exposure')
        g.addWidget(a, 0, 0, 1, 3)                                # each control under its label: a narrow panel
        self.cam_stamp = SegmentedControl([(k, DL.STAMP_LABEL[k]) for k in DL.STAMPS], 'mid_exposure')
        self.cam_stamp.changed.connect(lambda k: self._set(f'camera stamps mark the {DL.STAMP_LABEL[k].lower()}',
                                                           camera_stamp=k))
        g.addWidget(self.cam_stamp, 1, 0, 1, 3, Qt.AlignmentFlag.AlignLeft)
        a = label('Lidar stamps', ink2=True)
        a.setToolTip('Used only for clouds without a per-point time field: what a sweep\'s header stamp marks. Auto: '
                     'the start in a bag, the end in nuScenes')
        g.addWidget(a, 2, 0, 1, 3)
        self.lid_stamp = SegmentedControl([(k, DL.LIDAR_STAMP_LABEL[k]) for k in DL.LIDAR_STAMPS], 'auto')
        self.lid_stamp.changed.connect(lambda k: self._set(f'lidar stamps {DL.LIDAR_STAMP_LABEL[k].lower()}',
                                                           lidar_stamp=k))
        g.addWidget(self.lid_stamp, 3, 0, 1, 3, Qt.AlignmentFlag.AlignLeft)
        self.fields = {}
        for i, (key, text, unit, dec, step, tip) in enumerate(FIELDS, 4):
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
        ll.addWidget(label(DL.ASSUMPTIONS, muted=True, small=True, wrap=True))
        self.status = label('', muted=True, small=True, wrap=True)
        ll.addWidget(self.status)
        self.bar = QProgressBar()
        self.bar.setTextVisible(False)
        self.bar.setRange(0, 1000)
        self.bar.hide()
        ll.addWidget(self.bar)
        bl = QHBoxLayout()
        b_def = button('Defaults', 'ghost', None, 'Back to the default thresholds (the log and the mapping stay)')
        b_def.clicked.connect(self._defaults)
        bl.addWidget(b_def)
        bl.addStretch(1)
        self.b_export = button('Export…', None, 'export', 'The check as JSON, the streams as CSV, the timeline and '
                                                          'each camera\'s overlay (plan and corrected) as PNG')
        self.b_export.clicked.connect(lambda: self.export())
        bl.addWidget(self.b_export)
        self.b_run = button('Run', 'primary', None, 'Read the log and check it against the plan')
        self.b_run.clicked.connect(lambda: self.run())
        bl.addWidget(self.b_run)
        ll.addLayout(bl)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        sc.setWidget(w)
        sc.setMinimumWidth(340)
        return sc

    # ------------------------------------------------------------------ right: tiles and the sections
    def _right(self):
        w = QWidget()
        w.setObjectName('panel')
        rl = QVBoxLayout(w)
        rl.setContentsMargins(10, 8, 10, 8)
        rl.setSpacing(8)
        tiles = QHBoxLayout()
        tiles.setSpacing(8)
        self.tiles = {}
        for key, caption, tip in (
                ('streams', 'Streams', 'Visible sensors whose stream passes every check, of all'),
                ('drops', 'Dropped frames', 'Frames missing over all streams, and the worst stream'),
                ('sync', 'Worst sync offset', 'The largest camera trigger error against the sync plan (the lidar\'s '
                                              'phase error taken out)'),
                ('rotation', 'Worst camera rotation', 'The largest confident rotation correction of a camera against '
                                                      'the lidar'),
                ('data', 'Data rate', 'GB per hour in the log against the recording budget')):
            t = self.tiles[key] = Tile(caption, tip)
            tiles.addWidget(t, 1)
        rl.addLayout(tiles)
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        # streams
        sw = QSplitter(Qt.Orientation.Vertical)
        sw.setHandleWidth(2)
        self.streams = _table(STREAM_COLS)
        self.streams.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        sw.addWidget(self.streams)
        self.timeline = TimelineView()
        sw.addWidget(self.timeline)
        sw.setSizes([360, 240])
        self.tabs.addTab(sw, 'Streams')
        # sync
        yw = QWidget()
        yl = QVBoxLayout(yw)
        yl.setContentsMargins(0, 6, 0, 0)
        yl.setSpacing(6)
        self.lidar_line = label('', wrap=True)
        yl.addWidget(self.lidar_line)
        self.sync = _table(SYNC_COLS)
        yl.addWidget(self.sync, 1)
        self.tabs.addTab(yw, 'Sync')
        # extrinsics
        ew = QWidget()
        el = QVBoxLayout(ew)
        el.setContentsMargins(0, 6, 0, 0)
        el.setSpacing(6)
        ctl = QHBoxLayout()
        self.cam_pick = QComboBox()
        self.cam_pick.setToolTip('The camera shown')
        self.cam_pick.currentIndexChanged.connect(lambda *_: self._camera_changed())
        ctl.addWidget(self.cam_pick)
        self.mode = SegmentedControl([('plan', 'Plan'), ('corrected', 'Corrected')], 'plan')
        self.mode.setToolTip('The lidar points projected with the plan\'s rotation or with the correction found')
        self.mode.changed.connect(lambda *_: self._show_frame())
        ctl.addWidget(self.mode)
        self.frame_label = label('', mono_=True, small=True)
        ctl.addWidget(self.frame_label)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setToolTip('The sampled frames over the drive')
        self.slider.valueChanged.connect(lambda *_: self._show_frame())
        ctl.addWidget(self.slider, 1)
        el.addLayout(ctl)
        esplit = QSplitter(Qt.Orientation.Vertical)
        esplit.setHandleWidth(2)
        self.overlay = OverlayView()
        esplit.addWidget(self.overlay)
        bot = QWidget()
        bl = QVBoxLayout(bot)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(4)
        self.ext = _table(EXT_COLS)
        bl.addWidget(self.ext, 1)
        ar = QHBoxLayout()
        self.ext_note = label('', muted=True, small=True, wrap=True)
        ar.addWidget(self.ext_note, 1)
        self.b_apply = button('Apply rotation corrections', None, 'rotate',
                              'A copy of the checked variant, named "… · drive-log corrected", with the rotations of '
                              'the cameras found rotated beyond the tolerance with high or medium confidence '
                              'applied on the axes the edges pin down; it becomes the active variant (one undo step)')
        self.b_apply.clicked.connect(lambda: self.apply_corrections())
        ar.addWidget(self.b_apply)
        bl.addLayout(ar)
        esplit.addWidget(bot)
        esplit.setSizes([440, 220])
        el.addWidget(esplit, 1)
        self.tabs.addTab(ew, 'Extrinsics')
        # findings
        fw = QWidget()
        fl = QVBoxLayout(fw)
        fl.setContentsMargins(0, 6, 0, 0)
        self.find_host = QWidget()
        self.find_lay = QVBoxLayout(self.find_host)
        self.find_lay.setContentsMargins(0, 0, 0, 0)
        self.find_lay.setSpacing(4)
        self.find_lay.addStretch(1)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setWidget(self.find_host)
        fl.addWidget(sc, 1)
        self.findings: list[StatusLine] = []
        self.tabs.addTab(fw, 'Findings')
        rl.addWidget(self.tabs, 1)
        return w

    # ------------------------------------------------------------------ settings
    @property
    def settings(self) -> DL.DriveLogSettings:
        return DL.DriveLogSettings.from_dict(self.ctx.store.project.analysis.drivelog)

    def _store(self, st: DL.DriveLogSettings, text) -> bool:
        new = DL.DriveLogSettings.from_dict(st.to_dict()).to_dict()
        if new == self.settings.to_dict():
            self._sync_fields()                # a clamped value: show what is stored
            return False

        def fn(p):
            p.analysis.drivelog = new
        self.ctx.store.edit('analysis', fn, f'Drive-log check · {text}')
        return True

    def _set(self, text, **kw):
        st = self.settings
        for k, v in kw.items():
            setattr(st, k, v)
        return self._store(st, text)

    def _defaults(self):
        st = self.settings
        d = DL.DriveLogSettings(path=st.path, variant_uid=st.variant_uid, mappings=st.mappings)
        self._store(d, 'default thresholds')

    def open_path(self, path) -> bool:
        """Open a bag or nuScenes folder: remember it in the settings (undoable) and list its topics."""
        p = DL.norm_path(path)
        if not p or not os.path.exists(p):
            self.ctx.msg(f'Not found: {path}', 5000)
            return False
        self._info = None
        self._set(f'log {os.path.basename(p)}', path=p)
        self._load_info()
        self._fill_topics()
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
            self._set(f'checked against {v.name if v else "the active variant"}', variant_uid=uid)

    def _set_mapping(self, topic, tag):
        st = self.settings
        path = DL.norm_path(st.path)
        m = dict(st.mappings.get(path, {}))
        m[topic] = tag
        st.mappings = {k: v for k, v in st.mappings.items() if k != path}
        st.mappings[path] = m
        self._store(st, f'{topic} → {tag or "no sensor"}')

    def _reset_mapping(self):
        st = self.settings
        path = DL.norm_path(st.path)
        if path in st.mappings:
            st.mappings = {k: v for k, v in st.mappings.items() if k != path}
            self._store(st, 'automatic topic mapping')

    def _load_info(self):
        path = self.settings.path
        if not path:
            self._info = None
            return
        if self._info is not None and self._info[0] == DL.norm_path(path):
            return
        try:
            self._info = (DL.norm_path(path), DL.open_log(path))
        except Exception as e:                  # a damaged or foreign file: say why
            self._info = (DL.norm_path(path), f'{type(e).__name__}: {e}')

    def _variant(self):
        st = self.settings
        p = self.ctx.store.project
        return (p.variant_by_uid(st.variant_uid) if st.variant_uid else None) or p.variant

    def _sync_fields(self):
        st = self.settings
        self._syncing = True
        try:
            for k, f in self.fields.items():
                f.set_value(float(getattr(st, k)))
            self.cam_stamp.set_value(st.camera_stamp)
            self.lid_stamp.set_value(st.lidar_stamp)
            p = self.ctx.store.project
            items = [(f'Active variant ({p.variant.name})', '')] + [(v.name, v.uid) for v in p.variants]
            if [(self.variant.itemText(i), self.variant.itemData(i)) for i in range(self.variant.count())] != items:
                self.variant.blockSignals(True)
                self.variant.clear()
                for text, uid in items:
                    self.variant.addItem(text, uid)
                self.variant.blockSignals(False)
            i = next((i for i in range(self.variant.count()) if self.variant.itemData(i) == st.variant_uid), 0)
            self.variant.setCurrentIndex(i)
            if st.path:
                self.path_label.setText(self.path_label.fontMetrics().elidedText(
                    st.path, Qt.TextElideMode.ElideMiddle, max(self.path_label.width(), 300)))
                self.path_label.setToolTip(st.path)
            else:
                self.path_label.setText('No log open')
                self.path_label.setToolTip('')
            self.b_reset.setEnabled(bool(st.path) and DL.norm_path(st.path) in st.mappings)
        finally:
            self._syncing = False

    def _fill_topics(self):
        """The topics of the open log and the sensor each is mapped to (a drop-down of the sensors its messages
        fit)."""
        st = self.settings
        t = self.topics
        self._load_info()
        var = self._variant()
        res = self.result if self.result is not None and self.result.path == DL.norm_path(st.path) else None
        if res is not None and res.scan is not None:
            src = res.scan
            rows = [(r['topic'], r['type'], r['messages']) for r in res.topics]
        elif self._info is not None and not isinstance(self._info[1], str):
            src = self._info[1]
            rows = sorted((tp, ty, src.counts.get(tp, 0)) for tp, ty in src.topics.items())
        else:
            t.setRowCount(0)
            return
        mapping = DL.mapping_for(var, src, st.overrides(st.path))
        t.setRowCount(len(rows))
        for r, (topic, ty, n) in enumerate(rows):
            it = QTableWidgetItem(topic)
            it.setToolTip(topic)
            t.setItem(r, 0, it)
            it = QTableWidgetItem(ty.split('/')[-1])
            it.setToolTip(ty)
            it.setForeground(THEME.color('ink2'))
            t.setItem(r, 1, it)
            _cell(t, r, 2, f'{n:,}', True)
            tags = DL.compatible_tags(var, ty)
            if not tags:
                it = QTableWidgetItem('—')
                it.setForeground(THEME.color('muted'))
                it.setToolTip('No sensor of the variant records this kind of message' if DL.topic_kind(ty) == 'other'
                              else 'No visible sensor of a kind that records these messages')
                t.removeCellWidget(r, 3)
                t.setItem(r, 3, it)
                continue
            cb = t.cellWidget(r, 3)
            if not isinstance(cb, QComboBox):
                cb = QComboBox()
                t.setCellWidget(r, 3, cb)
            cb.blockSignals(True)
            cb.clear()
            cb.addItem('—', '')
            for tag in tags:
                cb.addItem(tag, tag)
            cur = mapping.get(topic, '')
            cb.setCurrentIndex(max(cb.findData(cur), 0))
            cb.setToolTip('The rig sensor this topic records (— : none)'
                          + (' · edited' if topic in st.overrides(st.path) else ' · automatic'))
            cb.blockSignals(False)
            try:
                cb.activated.disconnect()
            except TypeError:
                pass
            cb.activated.connect(lambda i, c=cb, tp=topic: self._set_mapping(tp, c.itemData(i) or ''))

    # ------------------------------------------------------------------ state
    def _inputs_key(self, p=None):
        p = p or self.ctx.store.project
        st = DL.DriveLogSettings.from_dict(p.analysis.drivelog)
        var = (p.variant_by_uid(st.variant_uid) if st.variant_uid else None) or p.variant
        return json.dumps({'s': st.to_dict(), 'v': var.to_dict(), 'lib': p.section('library'),
                           'timing': p.analysis.timing, 'rec': p.analysis.recording, 'ecu': p.ecu.to_dict()},
                          sort_keys=True, default=str)

    @property
    def stale(self) -> bool:
        return self.result is not None and self._for is not None and self._for != self._inputs_key()

    def _on_change(self, *_):
        if self.isVisible():
            self._sync_fields()
            self._fill_topics()
            self.refresh()
        else:
            self._dirty = True

    def _on_replaced(self, *_):
        self._gen += 1                          # a run still going is for the old project
        self.result, self._scan, self._info, self._for, self.busy = None, None, None, None, False
        self.status_text = ''
        self._shown = False                     # not None: the empty result is drawn again
        self._sync_fields()
        self._fill_topics()
        self.refresh()

    def _retheme(self, *_):
        self._shown = False
        self._fill_topics()
        self.refresh()

    def showEvent(self, e):
        super().showEvent(e)
        if self._dirty:
            self.activate()

    def activate(self):
        self._dirty = False
        self._sync_fields()
        self._fill_topics()
        self.refresh()

    # ------------------------------------------------------------------ run
    def run(self, quiet=False) -> bool:
        st = self.settings
        if not st.path:
            self.ctx.msg('Open a ROS 2 bag or a nuScenes folder first.', 4000)
            return False
        win = self.ctx.window
        if self.busy and win is not None and JOB in getattr(win, '_jobs', {}):
            return False
        p = self.ctx.store.project
        project = p.snapshot()
        eng = win._shared_engine() if win is not None and hasattr(win, '_shared_engine') else \
            self.ctx.analysis.engine.share()
        scan = self._scan if self._scan is not None and self._scan.key[0] == DL.norm_path(st.path) else None
        self._gen += 1
        gen = self._gen
        key = (p, self._inputs_key(p))
        self.busy = True
        self.status_text = 'Reading the log…' if scan is None else 'Checking…'

        def work(progress, cancel):
            return DL.run(project, st.path, DL.DriveLogSettings.from_dict(project.analysis.drivelog), progress,
                          cancel, eng, None, scan)
        if win is not None and hasattr(win, 'run_job'):
            ok = win.run_job(JOB, 'Drive-log check', work, lambda out: self._done(gen, (key, out)),
                             lambda m: self._failed(gen, m))
            if not ok:
                self.busy = False
                self.status_text = ''
                self.ctx.msg('A drive-log check is still running: cancel it first.', 4000)
            self.refresh()
            return ok

        def thread():
            try:
                out = work(lambda f, text='': self.sig.progress.emit(gen, f, text), lambda: gen != self._gen)
                self.sig.done.emit(gen, (key, out))
            except RuntimeError:                # the page closed while reading
                pass
            except Exception as e:
                try:
                    self.sig.failed.emit(gen, f'{type(e).__name__}: {e}\n{traceback.format_exc()}')
                except RuntimeError:
                    pass
        threading.Thread(target=thread, daemon=True).start()
        self.refresh()
        return True

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
        self.result, self._scan, self._for = res, res.scan, key[1]
        m = res.metrics
        self.status_text = (f'{m["faults"]} fault{"s" * (m["faults"] != 1)}, {m["warnings"]} '
                            f'warning{"s" * (m["warnings"] != 1)} · {res.span["duration_s"]:.1f} s of drive, '
                            f'{res.span["messages"]:,} messages · read in {m["scan_seconds"]:.1f} s, checked in '
                            f'{m["seconds"] - (m["scan_seconds"] if res.scan is not None else 0):.1f} s')
        self._fill_topics()
        self.refresh()
        self.ctx.msg(f'Drive-log check: {self.status_text}', 5000)

    def _failed(self, gen, msg):
        if gen != self._gen:
            return
        self.busy = False
        self.status_text = 'Failed: ' + msg.splitlines()[0]
        self.refresh()

    # ------------------------------------------------------------------ show
    def refresh(self):
        st = self.settings
        win = self.ctx.window
        running = self.busy and (win is None or JOB in getattr(win, '_jobs', {JOB: 1}))
        self.b_run.setEnabled(bool(st.path) and not running)
        self.bar.setVisible(running and (win is None or not hasattr(win, 'run_job')))
        res = self.result
        self.b_export.setEnabled(res is not None)
        if self._info is not None and isinstance(self._info[1], str) and res is None:
            text = f'Cannot read the log: {self._info[1]}'
        else:
            text = self.status_text or (EMPTY if not st.path else 'Ready: Run reads the log and checks it.')
        self.status.setText(text + (' · stale: the plan or the settings changed since' if self.stale else ''))
        if res is self._shown:
            self._sync_apply()
            return
        self._shown = res
        self._fill_tiles(res)
        self._fill_streams(res)
        self._fill_sync(res)
        self._fill_ext(res)
        self._fill_findings(res)

    def _var_of(self, res):
        return self.ctx.store.project.variant_by_uid(res.variant_uid) if res is not None else None

    def _fill_tiles(self, res):
        if res is None:
            for t in self.tiles.values():
                t.set('—', '', 'no check yet')
            return
        m = res.metrics
        self.tiles['streams'].set(f'{m["streams_ok"]}', f'of {m["streams"]}',
                                  (f'{m["streams_faulty"]} with problems' + (f', {len(m["missing"])} missing'
                                                                             if m['missing'] else '')),
                                  'good' if not m['streams_faulty'] else 'critical' if m['missing'] or any(
                                      e['status'] == 'fault' for e in res.streams) else 'warning')
        self.tiles['drops'].set(f'{m["drops"]}', 'frames', (f'worst {m["worst_drop"]} '
                                                           f'{m["worst_drop_share"] * 100:.1f} %' if m['drops'] else
                                                           'no gaps'), 'good' if not m['drops'] else
                                'critical' if m['worst_drop_share'] > 0.01 else 'warning')
        tol = res.settings.sync_tol_ms
        if m['worst_sync_ms'] is None:
            self.tiles['sync'].set('—', '', res.sync.get('reason') or 'not measured')
        else:
            L = res.sync.get('lidar')
            note = f'{m["worst_sync_tag"]}' + (f' · {L["tag"]} phase {L["phase_error_deg"]:+.1f}°' if L else '')
            self.tiles['sync'].set(f'{m["worst_sync_ms"]:.1f}', 'ms', note,
                                   'good' if m['worst_sync_ms'] <= tol and (not L or abs(L['late_ms']) <= tol)
                                   else 'critical')
        if m['worst_rotation_deg'] is None:
            self.tiles['rotation'].set('—', '', 'no confident result')
        else:
            bad = m['worst_rotation_deg'] > res.settings.tolerance_deg
            self.tiles['rotation'].set(f'{m["worst_rotation_deg"]:.2f}', '°', f'{m["worst_rotation_tag"]} · '
                                       f'tolerance {res.settings.tolerance_deg:g}°', 'critical' if bad else 'good')
        t = res.data['totals']
        if t.get('gb_h') is None:
            self.tiles['data'].set('—', '', 'nothing recorded')
        else:
            r = t.get('ratio')
            self.tiles['data'].set(f'{t["gb_h"]:,.0f}', 'GB/h', f'budget {t["planned_gb_h"]:,.0f} GB/h',
                                   'warning' if r is not None and r > 1.25 else 'good')

    def _fill_streams(self, res):
        t = self.streams
        var = self._var_of(res)
        rows = res.streams if res is not None else []
        data = {r['tag']: r for r in res.data['rows']} if res is not None else {}
        t.setRowCount(len(rows))
        for i, e in enumerate(rows):
            col = QColor(DL.stream_color(var, e))
            if e['kind'] == 'camera' and var is not None:
                c = next((c for c in var.cameras if c.uid == e['uid']), None)
                if c is not None:
                    col = THEME.cam(c.slot)
            _cell(t, i, 0, e['tag'], color=col, tip=f'{DL.KIND_WORD.get(e["kind"], e["kind"])}')
            _cell(t, i, 1, e['topic'] or 'no topic', color=None if e['topic'] else THEME.color('muted'))
            _cell(t, i, 2, f'{e["messages"]:,}', True)
            _cell(t, i, 3, _f(e.get('rate_hz'), 2), True)
            _cell(t, i, 4, _f(e.get('expected_hz'), 1), True, THEME.color('ink2'))
            drops = e.get('drops') or 0
            _cell(t, i, 5, f'{drops}' if e['topic'] else '—', True,
                  QColor(STATUS['critical' if e.get('drop_share', 0) > 0.01 else 'warning']) if drops else None,
                  f'{drops} frames in {e.get("drop_events", 0)} gaps, {e.get("missing_share", 0) * 100:.2f} % of '
                  'the time' if drops else None)
            _cell(t, i, 6, _f(e.get('longest_gap_ms'), 0), True)
            _cell(t, i, 7, _f(e.get('jitter_ms'), 2), True)
            lat = e.get('latency_ms')
            _cell(t, i, 8, _f(lat['median'], 1) if lat else '—', True,
                  QColor(STATUS['critical']) if lat and lat['min'] < -1 else None,
                  f'median {lat["median"]:.1f}, p95 {lat["p95"]:.1f}, min {lat["min"]:.1f} ms' if lat else None)
            if e.get('size'):
                w, h = e['size']
                W, H = e.get('expected_size') or (w, h)
                _cell(t, i, 9, f'{w}×{h}', True, QColor(STATUS['critical']) if (w, h) != (W, H) else None,
                      f'module {W} × {H}')
            elif e.get('points') is not None:
                exp = e.get('points_expected')
                _cell(t, i, 9, f'{e["points"]:,.0f}', True,
                      tip=(f'{e["points"] / exp * 100:.0f} % of the model\'s {exp:,.0f} per frame' if exp else
                           'points per cloud'))
            else:
                _cell(t, i, 9, '—', True)
            d = data.get(e['tag'], {})
            _cell(t, i, 10, _f(d.get('mbps'), 1), True,
                  tip=(f'{d["bpp"]:.2f} bit/px, {d["compression"]:.1f}× smaller than raw' if d.get('bpp') and
                       d.get('compression') else None))
            r = d.get('ratio')
            _cell(t, i, 11, _f(d.get('planned_mbps'), 1), True, QColor(STATUS['warning']) if r and r > 1.25 else
                  THEME.color('ink2'), f'{r:.2f}× the budget' if r else None)
            st = e['status'] if e['topic'] or e['kind'] != 'gnss' else 'ok'
            _cell(t, i, 12, {'ok': 'OK', 'warning': 'Warning', 'fault': 'Fault'}.get(st, st), color=sev_color(st),
                  tip='\n'.join(e['problems'] + e['notes']) or None)
        self.timeline.set_rows(timeline_rows(res, var) if res is not None else [],
                               res.span['duration_s'] if res is not None else 0.0)

    def _fill_sync(self, res):
        t = self.sync
        if res is None:
            t.setRowCount(0)
            self.lidar_line.setText('')
            return
        L = res.sync.get('lidar')
        parts = []
        for x in res.sync['lidars']:
            parts.append(f'{x["tag"]}: phase-locked at {x["phase_deg"]:.1f}° (plan {x["plan_phase_deg"]:.1f}°, '
                         f'{"late" if x["late_ms"] > 0 else "early"} by {abs(x["late_ms"]):.2f} ms), '
                         f'{x["rate_hz"]:.3f} Hz, spread {x["spread_deg"]:.2f}°, from {x["source"]} over '
                         f'{x["sweeps"]} sweeps')
        if not parts:
            parts.append(res.sync.get('reason') or 'No spinning lidar with a topic: the sync against the sweep is not '
                                                   'measured.')
        elif L is None and res.sync.get('reason'):
            parts.append(res.sync['reason'])
        self.lidar_line.setText('\n'.join(parts) + f'\nCamera stamps taken as the '
                                f'{DL.STAMP_LABEL[res.sync["stamp_convention"]].lower()} (readout '
                                f'{res.sync["readout_ms"]:g} ms, exposure {res.sync["exposure_ms"]:g} ms).')
        rows = [r for r in res.sync['cameras']]
        t.setRowCount(len(rows))
        var = self._var_of(res)
        for i, r in enumerate(rows):
            c = var.cam(r['uid']) if var is not None else None
            _cell(t, i, 0, r['tag'], color=THEME.cam(c.slot) if c is not None else None)
            _cell(t, i, 1, r['group'] or 'free', color=THEME.color('ink2'))
            _cell(t, i, 2, f'{r["frames"]}' if r['topic'] else 'no topic', True)
            _cell(t, i, 3, _f(r.get('planned_ms'), 2), True, THEME.color('ink2'))
            _cell(t, i, 4, _f(r.get('offset_ms'), 2), True)
            own = r.get('own_ms')
            tol = res.settings.sync_tol_ms
            _cell(t, i, 5, _f(own, 2), True, QColor(STATUS['critical']) if own is not None and abs(own) > tol else
                  None, (f'deviation {r["deviation_ms"]:+.2f} ms, of which the lidar\'s phase '
                         f'{r["lidar_part_ms"]:+.2f} ms') if r.get('deviation_ms') is not None else None)
            _cell(t, i, 6, _f(r.get('p95_ms'), 2), True)
            _cell(t, i, 7, _f(r.get('spread_ms'), 2), True)
            sk = r.get('skew_ms')
            _cell(t, i, 8, _f(sk, 2), True, QColor(STATUS['critical']) if sk is not None and abs(sk) > tol else None,
                  f'against {r["skew_ref"]}, p95 {r["skew_p95_ms"]:.2f} ms' if sk is not None else None)
            stt = r.get('status')
            _cell(t, i, 9, {'ok': 'OK', 'warning': 'Warning', 'fault': 'Fault', None: '—'}.get(stt, stt),
                  color=sev_color(stt) if stt else THEME.color('muted'))

    def _fill_ext(self, res):
        t = self.ext
        rows = res.extrinsics if res is not None else []
        t.setRowCount(len(rows))
        var = self._var_of(res)
        for i, r in enumerate(rows):
            c = var.cam(r['uid']) if var is not None else None
            _cell(t, i, 0, r['tag'], color=THEME.cam(c.slot) if c is not None else None)
            d = r.get('delta_deg')
            obs = r.get('observable') or [True] * 3
            se = r.get('stderr_deg') or [None] * 3
            for k in range(3):
                if d is None:
                    _cell(t, i, 1 + k, '—', True)
                    continue
                txt = f'{round(d[k], 2) + 0.0:+.2f}'
                big = obs[k] and abs(d[k]) > res.settings.tolerance_deg
                _cell(t, i, 1 + k, txt if obs[k] else f'({txt})', True,
                      QColor(STATUS['critical']) if big else None if obs[k] else THEME.color('muted'),
                      (f'± {se[k]:.3f}° standard error' if se[k] is not None else '') +
                      ('' if obs[k] else '\nNot pinned down by the edges in view: left as planned'))
            pinned = [s for s, o in zip(se, obs) if o and s is not None]
            _cell(t, i, 4, _f(max(pinned), 3) if pinned and d is not None else '—', True)
            _cell(t, i, 5, f'{r["agree"] * 100:.0f} %' if r.get('agree') is not None else '—', True)
            _cell(t, i, 6, f'{r.get("frames") or 0}', True)
            _cell(t, i, 7, (r.get('confidence') or '—').capitalize(), color=THEME.color('ink2'))
            sev = {'rotated': 'fault', 'ok': 'ok'}.get(r['verdict'], 'warning')
            _cell(t, i, 8, 'OK' if r['verdict'] == 'ok' else r['verdict'].capitalize(), color=sev_color(sev),
                  tip=r.get('reason') or None)
        # the camera picker
        cur = self.cam_pick.currentData()
        self.cam_pick.blockSignals(True)
        self.cam_pick.clear()
        if res is not None:
            for uid, ov in sorted(res.overlays.items(), key=lambda kv: kv[1]['tag']):
                if ov['frames']:
                    self.cam_pick.addItem(ov['tag'], uid)
        i = max(self.cam_pick.findData(cur), 0)
        self.cam_pick.setCurrentIndex(i)
        self.cam_pick.blockSignals(False)
        lid = next((r['lidar'] for r in rows if r.get('lidar')), None)
        comp = next((r['compensation'] for r in rows if r.get('compensation')), None)
        self.ext_note.setText((f'Rotations relative to {lid}; motion between the lidar points and the images: '
                               f'{comp}. Translation is not observable this way. Values in brackets: the axis is '
                               'not pinned down by the edges in view.') if lid else
                              ('No lidar of the variant has a topic: rotations not checked.' if res is not None
                               else ''))
        self._camera_changed()
        self._sync_apply()

    def _sync_apply(self):
        res = self.result
        n = len([r for r in res.extrinsics if r.get('apply')]) if res is not None else 0
        alive = res is not None and self.ctx.store.project.variant_by_uid(res.variant_uid) is not None
        self.b_apply.setEnabled(bool(n) and alive)
        self.b_apply.setText(f'Apply rotation corrections ({n})' if n else 'Apply rotation corrections')

    def _camera_changed(self):
        res = self.result
        uid = self.cam_pick.currentData()
        ov = res.overlays.get(uid) if res is not None and uid else None
        n = len(ov['frames']) if ov else 0
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(n - 1, 0))
        if self.slider.value() > n - 1:
            self.slider.setValue(n // 2)
        self.slider.blockSignals(False)
        self.slider.setEnabled(n > 1)
        self._show_frame()

    def _show_frame(self):
        res = self.result
        uid = self.cam_pick.currentData()
        ov = res.overlays.get(uid) if res is not None and uid else None
        if not ov or not ov['frames']:
            self.frame_label.setText('')
            self.overlay.clear('Run the check to see the lidar points over the camera images.' if res is None else
                               'No camera frame with lidar depth edges in view.')
            return
        k = min(self.slider.value(), len(ov['frames']) - 1)
        fr = ov['frames'][k]
        corrected = self.mode.value() == 'corrected'
        delta = ov.get('delta') if corrected else None
        var = self._var_of(res)
        if var is None:
            self.overlay.clear('The checked variant was deleted.')
            return
        x, y, col = DL.project_overlay(self.ctx.store.project, var, ov, fr, delta)
        ext = next((e for e in res.extrinsics if e['uid'] == uid), {})
        if corrected and delta is None:
            cap = f'{ov["tag"]} · no correction found'
        elif corrected:
            cap = f'{ov["tag"]} · corrected: yaw {delta[0]:+.2f}°, pitch {delta[1]:+.2f}°, roll {delta[2]:+.2f}°'
        else:
            cap = f'{ov["tag"]} · the plan\'s rotation'
        note = f'{len(x):,} points · {fr["edges"]} depth edges'
        if ext.get('score_plan') is not None:
            note += f' · edge score {ext["score_plan"]:.3f} → {ext["score_corrected"]:.3f}'
        self.frame_label.setText(f'{k + 1}/{len(ov["frames"])} · {fr["t_s"]:.2f} s')
        self.overlay.set_frame(fr['jpeg'], (x, y, col), cap, note)

    def _fill_findings(self, res):
        for w in self.findings:
            self.find_lay.removeWidget(w)
            w.hide()                    # until the deferred delete runs it would still paint over the new rows
            w.deleteLater()
        self.findings = []
        if res is None:
            return
        for f in res.findings:
            w = StatusLine(SEV_TONE.get(f['severity'], 'warning'), f['message'])
            self.find_lay.insertWidget(self.find_lay.count() - 1, w)
            self.findings.append(w)
        for n in res.notes:
            w = StatusLine(None, n)
            self.find_lay.insertWidget(self.find_lay.count() - 1, w)
            self.findings.append(w)

    def summary(self) -> dict | None:
        """The last check's compact figures (core.drivelog.summary) for the assistant; None before a run."""
        if self.result is None:
            return None
        return dict(DL.summary(self.result), stale=self.stale)

    # ------------------------------------------------------------------ actions
    def apply_corrections(self):
        """Add the corrected variant (confident cameras only) and switch to it, as one undo step."""
        res = self.result
        if res is None:
            return None
        p = self.ctx.store.project
        v = DL.corrected_variant(p, res)
        if v is None:
            self.ctx.msg('No confident rotation correction to apply.', 4000)
            return None
        state = v.to_dict()

        def fn(pj):
            from ...core.project import Variant
            pj.variants.append(Variant.from_dict(state))
            pj.active_uid = state['uid']
        self.ctx.store.edit('variants', fn, f'Drive-log check · new variant {v.name}')
        self.ctx.msg(f'Added {v.name}: {v.note}', 6000)
        return v.uid

    def export(self, folder=None):
        res = self.result
        if res is None:
            return []
        if folder is None:
            folder = QFileDialog.getExistingDirectory(self, 'Export the drive-log check into')
            if not folder:
                return []
        paths = DL.export(res, self.ctx.store.project, folder)
        self.ctx.msg(f'Exported the drive-log check to {folder}', 5000)
        return paths
