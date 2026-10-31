"""Calibrate ▸ As-built check: a recording of the real rig (a ROS 2 bag, a nuScenes folder or a folder of images per
camera) against the plan. Per camera the outline of the vehicle in its frames (what stayed the same while driving)
against the simulated one, the image circle against the lens model, the pose correction that lays one onto the other
(and the drive-log check's rotation against the lidar beside it), then the coverage of the rig as built against the
plan's and a verdict; a new as-built variant from the confident corrections. The recording is read in a background
job; editing the mapping or the thresholds checks again without reading it again."""
from __future__ import annotations

import json
import os
import threading
import traceback

import numpy as np
from PyQt6.QtCore import QObject, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QImage, QPixmap
from PyQt6.QtWidgets import (QAbstractItemView, QComboBox, QFileDialog, QGridLayout, QHBoxLayout, QHeaderView,
                             QLabel, QProgressBar, QScrollArea, QSplitter, QTableWidget, QTableWidgetItem, QTabWidget,
                             QVBoxLayout, QWidget)

from ...core import asbuilt as AB
from ..asbuiltview import AsBuiltView
from ..theme import STATUS, THEME, mono
from ..widgets import fit_width, ROW_H, ScrubSpinBox, SegmentedControl, StatusLine, ToggleSwitch, button, hline, label
from .recording_ws import Tile

JOB = 'asbuilt'
RECENT_KEY = 'asbuilt/recent'
RECENT_KEPT = 8
EMPTY = ('Open a recording of the real rig (a ROS 2 bag, a nuScenes folder or a folder of images per camera), then '
         'Run.')
SEV_TONE = {'critical': 'critical', 'serious': 'serious', 'warning': 'warning', 'good': 'good'}
VERDICT_TONE = {'holds': 'good', 'holds_with_corrections': 'serious', 'does_not_hold': 'critical',
                'not_checkable': 'warning'}
# (group, key, label, unit, decimals, step, tip)
FIELDS = (
    ('Reading', 'frames', 'Frames per camera', '', 0, 1.0,
     'Frames kept per camera, spread over the drive (the recording is read again when this changes)'),
    ('Reading', 'min_motion_m', 'Travel between', 'm', 2, 0.1,
     'With odometry or GNSS in the recording: a frame is kept only when the vehicle moved this far (or turned 3°) '
     'since the stream\'s last kept frame'),
    ('Reading', 'min_change', 'Picture change', '', 1, 0.5,
     'Without odometry or GNSS: a frame is kept only when its picture changed by this many grey levels on average'),
    ('Mask', 'static_sigma', 'Static within', 'σ', 1, 0.5,
     'A pixel that varies less than this many noise sigmas over the frames (90th percentile) is static'),
    ('Mask', 'edge_contrast', 'Static edge', 'grey', 0, 1.0,
     'Intensity step of a static edge that splits static areas: the vehicle from plain road or sky'),
    ('Mask', 'min_area_pct', 'Smallest piece', '%', 2, 0.01,
     'Vehicle pieces (and holes in them) smaller than this share of the image are dropped'),
    ('Mask', 'search_deg', 'Search band', '°', 1, 0.5,
     'Static areas within this angle of the planned silhouette can be the vehicle; the rotation is searched within ± '
     'this'),
    ('Tolerances', 'tol_rotation_deg', 'Rotation', '°', 2, 0.05,
     'A confident correction beyond this is a mounting error (applied to the as-built copy)'),
    ('Tolerances', 'tol_translation_m', 'Shift', 'm', 3, 0.005, 'The same for a fitted shift'),
    ('Tolerances', 'tol_boundary_px', 'Outline', 'px', 1, 0.5,
     'An outline still this far off on average after the best pose (and this many degrees) is not explained by the '
     'pose'),
    ('Tolerances', 'tol_boundary_deg', 'Outline', '°', 2, 0.05, 'The same in degrees'),
    ('Tolerances', 'tol_circle_pct', 'Image circle', '%', 1, 0.5,
     'Image circle centre or radius off the lens model by more than this share of the radius'),
    ('Tolerances', 'tol_blind_m2', 'Blind ground', 'm²', 2, 0.01, 'More blind ground than this over the plan\'s'),
    ('Tolerances', 'tol_multi_pct', 'Two-camera share', 'pt', 1, 0.5,
     'The share seen by two or more cameras dropping by more than this many points'),
    ('Tolerances', 'tol_density_pct', 'Density shares', 'pt', 1, 0.5,
     'The shares at the minimum and target density dropping by more than this many points'),
    ('Tolerances', 'tol_visible_pct', 'Visible ground', '%', 1, 0.5,
     'A camera\'s visible ground changing by more than this share of the plan\'s'),
)
TOGGLES = (('lidar_check', 'Drive-log estimate', 'Also run the drive-log check\'s rotation against the lidar on the same '
                                                 'recording (when it has a lidar topic): a second, independent '
                                                 'estimate, and the one used when a camera\'s outline cannot tell'),
           ('fit_translation', 'Fit a shift', 'Also fit a shift of each camera (weakly observable: a distant edge moves '
                                              'the same for a turn as for a shift); held at the plan when the outline '
                                              'does not constrain it'))
CAM_COLS = (('Camera', ''), ('Frames', 'Frames kept while moving'),
            ('IoU plan', 'Overlap of the real and the simulated mask at the plan\'s pose, over the pixels the frames '
                         'decide'),
            ('IoU fit', 'The same at the fitted pose'),
            ('Outline px', 'Mean distance between the real and the planned outline (where the real one is in reach)'),
            ('Outline °', 'The same in degrees'),
            ('Yaw °', 'Correction added to the plan\'s yaw (fitted from the outline)'), ('Pitch °', ''), ('Roll °', ''),
            ('± °', 'Largest standard error of the constrained axes'),
            ('Lidar °', 'The drive-log check\'s rotation against the lidar: yaw / pitch / roll'),
            ('Agree', 'Whether the two estimates agree within their errors'),
            ('Circle', 'Image circle: centre and radius error against the lens model'),
            ('Verdict', ''))
COV_COLS = (('Figure', ''), ('Plan', ''), ('As built', ''), ('Δ', 'As built minus plan'), ('Tolerance', ''),
            ('Holds', ''))
VIS_COLS = (('Camera', ''), ('Plan m²', 'Ground in the ROI the camera sees'), ('As built m²', ''), ('Δ %', ''),
            ('Holds', ''))
REQ_COLS = (('Requirement', ''), ('Plan', ''), ('As built', ''))


def _table(cols, select=False):
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


def _f(v, d=1, unit='', sign=False):
    if v is None:
        return '—'
    try:
        v = float(v)
    except (TypeError, ValueError):
        return '—'
    if not np.isfinite(v):
        return '—'
    return f'{round(v, d) + 0.0:{"+" if sign else ""},.{d}f}{unit}'


def verdict_color(v) -> QColor:
    return QColor(STATUS[VERDICT_TONE.get(v, 'warning')])


class _Sig(QObject):
    done = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)
    progress = pyqtSignal(int, float, str)


class AsBuiltPage(QWidget):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.result: AB.AsBuiltResult | None = None
        self._scan = None                  # the last scan: used again while the recording and frame settings stay
        self._info = None                  # (path, LogInfo or error text) of the open recording, for the mapping
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
        split.setSizes([400, 1200])
        lay.addWidget(split)
        ctx.store.changed.connect(self._on_change)
        ctx.store.project_replaced.connect(self._on_replaced)
        THEME.changed.connect(self._retheme)
        self._sync_fields()
        self.refresh()

    # ------------------------------------------------------------------ left: the recording, the mapping, settings
    def _left(self):
        w = QWidget()
        w.setObjectName('panel')
        ll = QVBoxLayout(w)
        ll.setContentsMargins(8, 6, 8, 8)
        ll.setSpacing(4)
        top = QHBoxLayout()
        top.addWidget(label('As-built check'))
        top.addStretch(1)
        top.addWidget(label('the plan against the real rig', muted=True, small=True))
        ll.addLayout(top)
        ll.addWidget(hline())
        ll.addWidget(label('Load a short drive recorded by the real rig: does each camera see the vehicle where the plan '
                           'says it does, how far is it off (the pose correction that lines the outlines up), and do '
                           'the plan\'s coverage figures still hold with the rig as built?', muted=True, small=True,
                           wrap=True))
        ll.addSpacing(4)
        ll.addWidget(label('Recording', bold=True))
        self.path_label = label('No recording open', mono_=True, small=True)
        self.path_label.setMinimumWidth(10)
        ll.addWidget(self.path_label)
        row = QHBoxLayout()
        b = self.b_bag = button('Open bag…', None, 'import', 'A ROS 2 bag file: .mcap or .db3 (for a rosbag2 folder '
                                                              'use Open folder…)')
        b.clicked.connect(self._pick_file)
        row.addWidget(b)
        b = self.b_dir = button('Open folder…', None, 'folder',
                                'A rosbag2 folder, a nuScenes folder, or a folder of images per camera (a subfolder '
                                'per camera named by its tag, or file names starting with the tag: F_000123.jpg)')
        b.clicked.connect(self._pick_dir)
        row.addWidget(b)
        row.addStretch(1)
        ll.addLayout(row)
        self.recent = QComboBox()
        self.recent.setToolTip('Recordings opened before')
        self.recent.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.recent.setMinimumContentsLength(14)
        self.recent.activated.connect(self._recent_picked)
        ll.addWidget(self.recent)
        vr = QHBoxLayout()
        a = label('Plan', ink2=True)
        a.setFixedWidth(60)
        vr.addWidget(a)
        self.variant = QComboBox()
        self.variant.setToolTip('The variant the recording is checked against')
        self.variant.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.variant.setMinimumContentsLength(14)
        self.variant.activated.connect(self._variant_picked)
        vr.addWidget(self.variant, 1)
        ll.addLayout(vr)
        ll.addSpacing(4)
        mh = QHBoxLayout()
        mh.addWidget(label('Streams', bold=True))
        mh.addStretch(1)
        self.b_reset = button('Automatic', 'ghost', None, 'Forget the edits of this recording\'s mapping: every image '
                                                          'stream mapped by the camera tag in its name again')
        self.b_reset.clicked.connect(self._reset_mapping)
        mh.addWidget(self.b_reset)
        ll.addLayout(mh)
        self.topics = QTableWidget(0, 4)
        self.topics.setHorizontalHeaderLabels(['Stream', 'Type', 'Msgs', 'Camera'])
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
        self.toggles = {}
        for key, text, tip in TOGGLES:
            t = ToggleSwitch(text)
            t.setToolTip(tip)
            t.clicked.connect(lambda on, k=key, x=text: self._set(f'{x.lower()} {"on" if on else "off"}', **{k: on}))
            ll.addWidget(t)
            self.toggles[key] = t
        self.fields = {}
        group = None
        g = None
        r = 0
        for grp, key, text, unit, dec, step, tip in FIELDS:
            if grp != group:
                group = grp
                ll.addSpacing(4)
                ll.addWidget(label(grp, bold=True))
                g = QGridLayout()
                g.setVerticalSpacing(2)
                g.setHorizontalSpacing(6)
                g.setColumnStretch(2, 1)
                ll.addLayout(g)
                r = 0
            f = ScrubSpinBox('', 'plain', dec, unit, step=step, width=110)
            f.setToolTip(tip)
            f.committed.connect(lambda v, k=key, t=text, u=unit: self._set(f'{t.lower()} {v:g}{u}', **{k: v}))
            a = label(text, ink2=True)
            a.setFixedWidth(118)
            a.setToolTip(tip)
            g.addWidget(a, r, 0)
            g.addWidget(f, r, 1)
            r += 1
            self.fields[key] = f
        ll.addSpacing(4)
        ll.addWidget(label(AB.ASSUMPTIONS, muted=True, small=True, wrap=True))
        self.status = label('', muted=True, small=True, wrap=True)
        ll.addWidget(self.status)
        self.bar = QProgressBar()
        self.bar.setTextVisible(False)
        self.bar.setRange(0, 1000)
        self.bar.hide()
        ll.addWidget(self.bar)
        bl = QHBoxLayout()
        b_def = button('Defaults', 'ghost', None, 'Back to the default thresholds (the recording and the mapping stay)')
        b_def.clicked.connect(self._defaults)
        bl.addWidget(b_def)
        bl.addStretch(1)
        self.b_export = button('Export…', None, 'export', 'The check as JSON, the cameras as CSV, the coverage '
                                                          'difference and each camera\'s overlay as PNG')
        self.b_export.clicked.connect(lambda: self.export())
        bl.addWidget(self.b_export)
        self.b_run = button('Run', 'primary', None, 'Read the recording and check it against the plan')
        self.b_run.clicked.connect(lambda: self.run())
        bl.addWidget(self.b_run)
        ll.addLayout(bl)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setWidget(w)
        fit_width(sc)
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
                ('verdict', 'The plan\'s coverage', 'Holds, holds with corrections (the rig is off but the coverage '
                                                    'figures stay within the tolerances), or does not hold'),
                ('cameras', 'Cameras checked', 'Cameras whose outline (or drive-log estimate) could be checked, of '
                                               'all visible'),
                ('worst', 'Largest correction', 'The largest confident rotation correction applied to the as-built '
                                                'copy'),
                ('iou', 'Mask overlap', 'Mean IoU of the real and the simulated mask at the fitted pose (the plan\'s '
                                        'in the note)'),
                ('blind', 'Blind ground', 'Blind ground in the ROI as built against the plan')):
            t = self.tiles[key] = Tile(caption, tip)
            tiles.addWidget(t, 1)
        rl.addLayout(tiles)
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        # cameras
        cw = QWidget()
        cl = QVBoxLayout(cw)
        cl.setContentsMargins(0, 6, 0, 0)
        cl.setSpacing(6)
        ctl = QHBoxLayout()
        self.cam_pick = QComboBox()
        self.cam_pick.setToolTip('The camera shown')
        self.cam_pick.currentIndexChanged.connect(lambda *_: self._camera_changed())
        ctl.addWidget(self.cam_pick)
        self.mode = SegmentedControl([('plan', 'Plan'), ('fitted', 'Fitted')], 'plan')
        self.mode.setToolTip('The simulated outline at the plan\'s pose or at the fitted one')
        self.mode.changed.connect(lambda *_: self._show_camera())
        ctl.addWidget(self.mode)
        self.backdrop = SegmentedControl([('frame', 'Frame'), ('median', 'Median')], 'frame')
        self.backdrop.setToolTip('A frame from the middle of the drive, or the median of the kept frames (what stays '
                                 'the same: the vehicle, and plain road or sky)')
        self.backdrop.changed.connect(lambda *_: self._show_camera())
        ctl.addWidget(self.backdrop)
        ctl.addStretch(1)
        self.b_add = button('Add as-built variant', None, 'plus',
                            'A copy of the checked variant, named "… · as built <date>", with every confident '
                            'correction applied; it becomes the active variant (one undo step)')
        self.b_add.clicked.connect(lambda: self.add_variant())
        ctl.addWidget(self.b_add)
        cl.addLayout(ctl)
        csplit = QSplitter(Qt.Orientation.Vertical)
        csplit.setHandleWidth(2)
        self.view = AsBuiltView()
        csplit.addWidget(self.view)
        bot = QWidget()
        bl = QVBoxLayout(bot)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(4)
        self.cams = _table(CAM_COLS, select=True)
        self.cams.itemSelectionChanged.connect(self._row_picked)
        bl.addWidget(self.cams, 1)
        self.cam_note = label('', muted=True, small=True, wrap=True)
        bl.addWidget(self.cam_note)
        csplit.addWidget(bot)
        csplit.setSizes([440, 240])
        cl.addWidget(csplit, 1)
        self.tabs.addTab(cw, 'Cameras')
        # coverage
        vw = QWidget()
        vl = QHBoxLayout(vw)
        vl.setContentsMargins(0, 6, 0, 0)
        vl.setSpacing(8)
        tl = QVBoxLayout()
        tl.setSpacing(6)
        self.cov_line = label('', wrap=True)
        tl.addWidget(self.cov_line)
        self.cov = _table(COV_COLS)
        tl.addWidget(self.cov, 2)
        self.vis = _table(VIS_COLS)
        tl.addWidget(self.vis, 2)
        self.req_line = label('', muted=True, small=True, wrap=True)
        tl.addWidget(self.req_line)
        self.req = _table(REQ_COLS)
        self.req.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        tl.addWidget(self.req, 1)
        vl.addLayout(tl, 3)
        pv = QVBoxLayout()
        self.cov_img = QLabel()
        self.cov_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cov_img.setMinimumSize(220, 220)
        pv.addWidget(self.cov_img, 1)
        pv.addWidget(label('Forward up. Blue: seen as built (darker: by two or more cameras); red: blind in both; dark '
                           'red: blind only as built; navy: seen only as built; grey: the body.', muted=True,
                           small=True, wrap=True))
        vl.addLayout(pv, 2)
        self.tabs.addTab(vw, 'Coverage')
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
    def settings(self) -> AB.AsBuiltSettings:
        return AB.settings_of(self.ctx.store.project)

    def _store(self, st: AB.AsBuiltSettings, text) -> bool:
        new = AB.AsBuiltSettings.from_dict(st.to_dict()).to_dict()
        if new == self.settings.to_dict():
            self._sync_fields()                # a clamped value: show what is stored
            return False

        def fn(p):
            p.analysis.asbuilt = new
        self.ctx.store.edit('analysis', fn, f'As-built check · {text}')
        return True

    def _set(self, text, **kw):
        st = self.settings
        for k, v in kw.items():
            setattr(st, k, v)
        return self._store(st, text)

    def _defaults(self):
        st = self.settings
        d = AB.AsBuiltSettings(path=st.path, variant_uid=st.variant_uid, mappings=st.mappings)
        self._store(d, 'default thresholds')

    def open_path(self, path) -> bool:
        """Open a recording: remember it in the settings (undoable) and among the recent ones, list its streams."""
        p = AB.norm_path(path)
        try:
            AB.recording_kind(p)
        except (OSError, ValueError) as e:
            self.ctx.msg(str(e), 6000)
            return False
        self._info = None
        self._set(f'recording {os.path.basename(p)}', path=p)
        self._remember_recent(p)
        self._load_info()
        self._fill_topics()
        self.refresh()
        return True

    def _recent_list(self) -> list:
        v = self.ctx.settings.value(RECENT_KEY, [])
        if isinstance(v, str):
            v = [v]
        return [str(x) for x in (v or []) if str(x)]

    def _remember_recent(self, p):
        lst = [p] + [x for x in self._recent_list() if x != p]
        self.ctx.settings.setValue(RECENT_KEY, lst[:RECENT_KEPT])
        self._fill_recent()

    def _fill_recent(self):
        st = self.settings
        paths = self._recent_list() + [k for k in st.mappings if k not in self._recent_list()]
        paths = [x for x in paths if x]
        self.recent.blockSignals(True)
        self.recent.clear()
        self.recent.addItem('Recent recordings…' if paths else 'No recent recordings', '')
        for x in paths:
            self.recent.addItem(os.path.basename(x.rstrip(os.sep)) or x, x)
            self.recent.setItemData(self.recent.count() - 1, x, Qt.ItemDataRole.ToolTipRole)
        self.recent.setCurrentIndex(0)
        self.recent.setEnabled(bool(paths))
        self.recent.blockSignals(False)

    def _recent_picked(self, i):
        p = self.recent.itemData(i) or ''
        self.recent.setCurrentIndex(0)
        if p:
            self.open_path(p)

    def _pick_file(self):
        f, _ = QFileDialog.getOpenFileName(self, 'Open a ROS 2 bag', os.path.dirname(self.settings.path or ''),
                                           'ROS 2 bags (*.mcap *.db3);;All files (*)')
        if f:
            self.open_path(f)

    def _pick_dir(self):
        d = QFileDialog.getExistingDirectory(self, 'Open a rosbag2, nuScenes or image folder',
                                             os.path.dirname(self.settings.path or ''))
        if d:
            self.open_path(d)

    def _variant_picked(self, i):
        uid = self.variant.itemData(i) or ''
        if uid != self.settings.variant_uid:
            v = self.ctx.store.project.variant_by_uid(uid)
            self._set(f'checked against {v.name if v else "the active variant"}', variant_uid=uid)

    def set_mapping(self, stream, tag):
        """Map a stream to a camera tag ('' = no camera) for the open recording (undoable)."""
        st = self.settings
        path = AB.norm_path(st.path)
        m = dict(st.mappings.get(path, {}))
        m[stream] = tag
        st.mappings = {k: v for k, v in st.mappings.items() if k != path}
        st.mappings[path] = m
        self._store(st, f'{stream} → {tag or "no camera"}')

    def _reset_mapping(self):
        st = self.settings
        path = AB.norm_path(st.path)
        if path in st.mappings:
            st.mappings = {k: v for k, v in st.mappings.items() if k != path}
            self._store(st, 'automatic stream mapping')

    def _load_info(self):
        path = self.settings.path
        if not path:
            self._info = None
            return
        if self._info is not None and self._info[0] == AB.norm_path(path):
            return
        try:
            self._info = (AB.norm_path(path), AB.open_recording(path))
        except Exception as e:                  # noqa: BLE001 (a damaged or foreign recording: say why)
            self._info = (AB.norm_path(path), f'{type(e).__name__}: {e}')

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
            for k, t in self.toggles.items():
                t.blockSignals(True)
                t.setChecked(bool(getattr(st, k)))
                t.blockSignals(False)
            self.fields['tol_translation_m'].setEnabled(st.fit_translation)
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
                self.path_label.setText('No recording open')
                self.path_label.setToolTip('')
            self.b_reset.setEnabled(bool(st.path) and AB.norm_path(st.path) in st.mappings)
            self._fill_recent()
        finally:
            self._syncing = False

    def _fill_topics(self):
        """The image streams of the open recording and the camera each is mapped to (a drop-down of the cameras)."""
        st = self.settings
        t = self.topics
        self._load_info()
        var = self._variant()
        res = self.result if self.result is not None and self.result.path == AB.norm_path(st.path) else None
        if res is not None and res.scan is not None:
            src = res.scan
            rows = sorted((tp, ty, n) for tp, (ty, n) in src.topics.items())
        elif self._info is not None and not isinstance(self._info[1], str):
            src = self._info[1]
            rows = sorted((tp, ty, src.counts.get(tp, 0)) for tp, ty in src.topics.items())
        else:
            t.setRowCount(0)
            return
        rows = [r for r in rows if AB.DL.topic_kind(r[1]) == 'image']
        mapping = AB.mapping_for(var, src, st.overrides(st.path))
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
            tags = AB.compatible_tags(var, ty)
            if not tags:
                it = QTableWidgetItem('—')
                it.setForeground(THEME.color('muted'))
                it.setToolTip('No visible camera in the variant')
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
            cb.setToolTip('The camera this stream records (— : none)'
                          + (' · edited' if topic in st.overrides(st.path) else ' · automatic'))
            cb.blockSignals(False)
            try:
                cb.activated.disconnect()
            except TypeError:
                pass
            cb.activated.connect(lambda i, c=cb, tp=topic: self.set_mapping(tp, c.itemData(i) or ''))

    # ------------------------------------------------------------------ state
    def _inputs_key(self, p=None):
        p = p or self.ctx.store.project
        st = AB.settings_of(p)
        var = (p.variant_by_uid(st.variant_uid) if st.variant_uid else None) or p.variant
        m = p.vehicle.mesh
        return json.dumps({'s': st.to_dict(), 'v': var.to_dict(), 'lib': p.section('library'),
                           'rig': [r.to_dict() for r in p.rig], 'mesh': [id(m), getattr(m, 'n_tris', 0)],
                           'roi': p.roi.to_dict(), 'dens': [p.analysis.min_density, p.analysis.target_density],
                           'req': p.analysis.requirements, 'dl': p.analysis.drivelog},
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
        self._shown = False
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
            self.ctx.msg('Open a recording first.', 4000)
            return False
        win = self.ctx.window
        if self.busy and win is not None and JOB in getattr(win, '_jobs', {}):
            return False
        p = self.ctx.store.project
        project = p.snapshot()
        eng = win._shared_engine() if win is not None and hasattr(win, '_shared_engine') else \
            self.ctx.analysis.engine.share()
        scan = self._scan if self._scan is not None and self._scan.key[0] == AB.norm_path(st.path) else None
        self._gen += 1
        gen = self._gen
        key = (p, self._inputs_key(p))
        self.busy = True
        self.status_text = 'Reading the recording…' if scan is None else 'Checking…'

        def work(progress, cancel):
            return AB.run(project, eng, AB.settings_of(project), None, progress, cancel, st.path, scan)
        if win is not None and hasattr(win, 'run_job'):
            ok = win.run_job(JOB, 'As-built check', work, lambda out: self._done(gen, (key, out)),
                             lambda m: self._failed(gen, m))
            if not ok:
                self.busy = False
                self.status_text = ''
                self.ctx.msg('An as-built check is still running: cancel it first.', 4000)
            self.refresh()
            return ok

        def thread():
            try:
                out = work(lambda f, text='': self.sig.progress.emit(gen, f, text), lambda: gen != self._gen)
                self.sig.done.emit(gen, (key, out))
            except RuntimeError:                # the page closed while reading
                pass
            except Exception as e:              # noqa: BLE001
                try:
                    self.sig.failed.emit(gen, f'{type(e).__name__}: {e}\n{traceback.format_exc()}')
                except RuntimeError:
                    pass
        threading.Thread(target=thread, daemon=True).start()
        self.refresh()
        return True

    def cancel(self):
        """Stop a run in progress (the job bar's Cancel does the same)."""
        win = self.ctx.window
        job = getattr(win, '_jobs', {}).get(JOB) if win is not None else None
        if job is not None:
            job['cancel'].set()
        else:
            self._gen += 1
            self.busy = False
            self.status_text = 'Cancelled'
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
        self.result, self._scan, self._for = res, res.scan, key[1]
        AB.remember(self.ctx.store.project, res)
        m = res.metrics
        self.status_text = (f'{AB.VERDICT_TEXT[res.verdict]} · {m["checked"]} of {m["cameras"]} cameras checked, '
                            f'{len(m["corrected"])} corrected · {res.recording["frames"]} frames of '
                            f'{res.recording["duration_s"]:.1f} s (motion from {res.recording["motion"]}) · '
                            f'{m["seconds"]:.1f} s')
        self._fill_topics()
        self.refresh()
        self.ctx.msg(f'As-built check: {self.status_text}', 5000)

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
            text = f'Cannot read the recording: {self._info[1]}'
        else:
            text = self.status_text or (EMPTY if not st.path else 'Ready: Run reads the recording and checks it.')
        self.status.setText(text + (' · stale: the plan or the settings changed since' if self.stale else ''))
        if res is self._shown:
            self._sync_add()
            return
        self._shown = res
        self._fill_tiles(res)
        self._fill_cameras(res)
        self._fill_coverage(res)
        self._fill_findings(res)

    def _var_of(self, res):
        return self.ctx.store.project.variant_by_uid(res.variant_uid) if res is not None else None

    def _fill_tiles(self, res):
        if res is None:
            for t in self.tiles.values():
                t.set('—', '', 'no check yet')
            return
        m = res.metrics
        tone = VERDICT_TONE[res.verdict]
        n_corr = len(m['corrected'])
        if res.verdict == 'holds_with_corrections':     # the long words go into the note: the tile is narrow
            self.tiles['verdict'].set('Holds', '', f'with corrections to {n_corr} camera{"s" * (n_corr != 1)}', tone)
        else:
            self.tiles['verdict'].set(AB.VERDICT_TEXT[res.verdict], '', res.variant, tone)
        self.tiles['cameras'].set(f'{m["checked"] + sum(1 for r in res.cameras if r.get("source") == "lidar")}',
                                  f'of {m["cameras"]}',
                                  f'{m["not_checkable"]} not checkable' if m['not_checkable'] else 'all checked',
                                  'good' if not m['not_checkable'] else 'warning')
        corr = [r for r in res.cameras if r.get('apply')]
        if corr:
            w = max(corr, key=lambda r: max(abs(v) for v in r['correction_deg']))
            v = max(abs(x) for x in w['correction_deg'])
            self.tiles['worst'].set(f'{v:.2f}', '°', f'{w["tag"]} · {len(corr)} camera{"s" * (len(corr) != 1)} '
                                                      'corrected', 'serious')
        else:
            self.tiles['worst'].set('—', '', 'no correction needed', 'good' if m['checked'] else None)
        if m.get('mean_iou_fit') is not None:
            self.tiles['iou'].set(f'{m["mean_iou_fit"]:.3f}', '', f'plan {_f(m["mean_iou_plan"], 3)}')
        else:
            self.tiles['iou'].set('—', '', 'no outline fitted')
        cov = res.coverage
        if cov is None:
            self.tiles['blind'].set('—', '', 'no coverage analysis')
        else:
            row = next(r for r in cov['rows'] if r['key'] == 'blind_m2')
            self.tiles['blind'].set(f'{row["as_built"]:.2f}', 'm²', f'plan {row["plan"]:.2f} m² ({row["delta"]:+.2f})',
                                    'good' if row['holds'] else 'critical')

    def _fill_cameras(self, res):
        t = self.cams
        rows = res.cameras if res is not None else []
        var = self._var_of(res)
        t.blockSignals(True)
        t.setRowCount(len(rows))
        for i, r in enumerate(rows):
            c = var.cam(r['uid']) if var is not None else None
            _cell(t, i, 0, r['tag'], color=THEME.cam(c.slot) if c is not None else None, tip=r.get('topic') or
                  'no image stream')
            _cell(t, i, 1, f'{r["frames"]}', True)
            _cell(t, i, 2, _f(r.get('iou_plan'), 3), True)
            _cell(t, i, 3, _f(r.get('iou_fit'), 3), True)
            op = r.get('outline_plan') or {}
            _cell(t, i, 4, _f(op.get('mean_px'), 2), True,
                  tip=(f'95 %: {op["p95_px"]:.2f} px; {op["observed_pct"]:.0f} % of the planned outline observed'
                       if op.get('p95_px') is not None else None))
            _cell(t, i, 5, _f(op.get('mean_deg'), 3), True)
            fit = r.get('fit') or {}
            d = fit.get('delta_deg') if r['check'] in ('fitted', 'uncertain') else None
            cons = fit.get('constrained') or [False] * 3
            se = fit.get('stderr_deg') or [None] * 3
            for k in range(3):
                if d is None:
                    _cell(t, i, 6 + k, '—', True)
                    continue
                txt = _f(d[k], 2, sign=True)
                big = cons[k] and r.get('apply') and r['correction_deg'][k] != 0.0
                _cell(t, i, 6 + k, txt if cons[k] else f'({txt})', True,
                      QColor(STATUS['serious']) if big else None if cons[k] else THEME.color('muted'),
                      (f'± {se[k]:.3f}° standard error' if se[k] is not None else '') +
                      ('' if cons[k] else '\nNot constrained by the outline in view: left as planned'))
            pinned = [s for s, o in zip(se, cons) if o and s is not None]
            _cell(t, i, 9, _f(max(pinned), 3) if pinned and d is not None else '—', True)
            lid = r.get('lidar') or {}
            ld = AB.lidar_shown(lid)            # only the axes the lidar's edges pin down: the rest is no measurement
            obs = lid.get('observable') or [False] * 3
            _cell(t, i, 10, ' / '.join(_f(v, 2, sign=True) if o else '–' for v, o in zip(ld, obs)) if ld else '—',
                  True,
                  tip=(f'{lid.get("confidence") or "—"} confidence against {lid.get("lidar")}'
                       + (f': {lid["reason"]}' if lid.get('reason') else '')
                       + ('' if ld is None or all(obs) else
                          '\nAn axis shown – is not pinned down by the lidar\'s edges: left as planned')
                       + ('\nNo axis pinned down by the lidar\'s edges: no estimate' if ld is None and
                          lid.get('delta_deg') is not None else '')) if lid else None)
            ag = r.get('agree')
            _cell(t, i, 11, {True: 'Yes', False: 'No', None: '—'}[ag],
                  color=QColor(STATUS['good' if ag else 'warning']) if ag is not None else THEME.color('muted'))
            cc = r.get('circle') or {}
            if cc.get('measured'):
                _cell(t, i, 12, f'{cc["centre_error_px"]:.1f} / {cc["radius_error_px"]:+.1f} px', True,
                      tip=(f'Centre {cc["centre_error_px"]:.2f} px off, radius {cc["radius_error_px"]:+.2f} px '
                           f'({cc["radius_error_pct"]:+.2f} %, {cc["radius_error_deg"]:+.3f}°) against the lens model'))
            else:
                _cell(t, i, 12, 'inside' if cc.get('visible_predicted') else 'covers', color=THEME.color('muted'),
                      tip='The lens model puts the image circle inside the image, but no dark edge was found'
                      if cc.get('visible_predicted') else 'The image circle covers the sensor: nothing to measure')
            vt = AB.VERDICT_TEXT[r['verdict']]
            if r.get('source') == 'lidar':
                vt += ' (lidar)'
            _cell(t, i, 13, vt, color=verdict_color(r['verdict']), tip=r.get('reason') or None)
        t.blockSignals(False)
        cur = self.cam_pick.currentData()
        self.cam_pick.blockSignals(True)
        self.cam_pick.clear()
        for r in rows:
            if r.get('jpeg'):
                self.cam_pick.addItem(r['tag'], r['tag'])
        self.cam_pick.setCurrentIndex(max(self.cam_pick.findData(cur), 0))
        self.cam_pick.blockSignals(False)
        if res is None:
            self.cam_note.setText('')
        else:
            self.cam_note.setText('Corrections are added to the plan\'s yaw, pitch and roll (pitch positive down); '
                                  'values in brackets: the outline in view does not constrain that axis (left as '
                                  'planned). Lidar: the drive-log check\'s rotation against the lidar.')
        self._camera_changed()
        self._sync_add()

    def _sync_add(self):
        res = self.result
        alive = res is not None and self.ctx.store.project.variant_by_uid(res.variant_uid) is not None
        n = len([r for r in res.cameras if r.get('apply')]) if res is not None else 0
        self.b_add.setEnabled(alive and res.asbuilt is not None)
        self.b_add.setText(f'Add as-built variant ({n})' if n else 'Add as-built variant')

    def _row_picked(self):
        res = self.result
        rows = self.cams.selectionModel().selectedRows() if self.cams.selectionModel() else []
        if res is None or not rows:
            return
        tag = res.cameras[rows[0].row()]['tag']
        i = self.cam_pick.findData(tag)
        if i >= 0:
            self.cam_pick.setCurrentIndex(i)

    def _camera_changed(self):
        self._show_camera()

    def _show_camera(self):
        res = self.result
        tag = self.cam_pick.currentData()
        r = res.camera(tag) if res is not None and tag else None
        if r is None or not r.get('jpeg'):
            self.view.clear('Run the check to see each camera\'s real outline against the plan\'s.' if res is None
                            else 'No camera frame in the recording.')
            return
        which = self.mode.value() or 'plan'
        median = self.backdrop.value() == 'median' and r.get('median_jpeg')
        jpeg = r['median_jpeg'] if median else r['jpeg']
        size = (r['ashape'][1], r['ashape'][0]) if (median and r.get('ashape')) else r['tsize']
        layers = AB.overlay_layers(r, which, size) if r.get('ashape') else None
        fit = r.get('fit') or {}
        if which == 'fitted' and r['check'] in ('fitted', 'uncertain'):
            d = fit['delta_deg']
            cap = f'{r["tag"]} · fitted: yaw {d[0]:+.2f}°, pitch {d[1]:+.2f}°, roll {d[2]:+.2f}°'
            note = f'IoU {r["iou_fit"]:.3f}' if r.get('iou_fit') is not None else ''
        elif which == 'fitted':
            cap = f'{r["tag"]} · not fitted'
            note = r.get('reason') or ''
        else:
            cap = f'{r["tag"]} · the plan\'s pose'
            note = f'IoU {r["iou_plan"]:.3f}' if r.get('iou_plan') is not None else (r.get('reason') or '')
        note = (note + f' · {AB.VERDICT_TEXT[r["verdict"]]}').strip(' ·')
        self.view.set_frame(jpeg, layers, cap, note)

    def _fill_coverage(self, res):
        cov = res.coverage if res is not None else None
        for t in (self.cov, self.vis, self.req):
            t.setRowCount(0)
        self.cov_img.clear()
        if res is None:
            self.cov_line.setText('')
            self.req_line.setText('')
            return
        if cov is None:
            self.cov_line.setText('No coverage analysis: the project needs a vehicle body and an ROI.')
            self.req_line.setText('')
            return
        a = res.asbuilt
        self.cov_line.setText((f'The plan ({res.variant}) against the rig as built ({a.name}): ' if a is not None else
                               f'Nothing to correct: the rig as built is the plan ({res.variant}). ')
                              + ('the coverage figures hold.' if cov['holds'] else
                                 'some coverage figures change beyond the tolerances.'))
        t = self.cov
        t.setRowCount(len(cov['rows']))
        for i, row in enumerate(cov['rows']):
            d = 2 if row['unit'] != 'm²' else 3
            _cell(t, i, 0, f'{row["label"]} ({row["unit"]})')
            _cell(t, i, 1, _f(row['plan'], d), True, THEME.color('ink2'))
            _cell(t, i, 2, _f(row['as_built'], d), True)
            _cell(t, i, 3, _f(row['delta'], d, sign=True), True)
            _cell(t, i, 4, _f(row['tolerance'], 2) if row['tolerance'] is not None else '—', True, THEME.color('ink2'))
            h = row['holds']
            _cell(t, i, 5, {True: 'Holds', False: 'Does not hold', None: '—'}[h],
                  color=QColor(STATUS['good' if h else 'critical']) if h is not None else THEME.color('muted'))
        t = self.vis
        items = list(cov['cameras'].items())
        t.setRowCount(len(items))
        var = self._var_of(res)
        for i, (tag, d) in enumerate(items):
            c = var.by_tag(tag) if var is not None else None
            _cell(t, i, 0, tag, color=THEME.cam(c.slot) if c is not None else None)
            _cell(t, i, 1, _f(d['plan_m2'], 2), True, THEME.color('ink2'))
            _cell(t, i, 2, _f(d['as_built_m2'], 2), True)
            _cell(t, i, 3, _f(d['delta_pct'], 1, sign=True), True)
            _cell(t, i, 4, 'Holds' if d['holds'] else 'Does not hold', color=QColor(STATUS['good' if d['holds']
                                                                                         else 'critical']))
        rq = cov.get('requirements')
        t = self.req
        if rq is None:
            self.req_line.setText('No coverage requirement in the scorecard or implied by the plan.')
        else:
            self.req_line.setText(('The scorecard\'s coverage requirements' if rq['origin'] == 'scorecard' else
                                   'The coverage requirements the plan implies (the scorecard has none)')
                                  + f': plan {rq["plan"]}, as built {rq["as_built"]}.')
            t.setRowCount(len(rq['items']))
            word = {'pass': 'Pass', 'fail': 'Fail', 'not_evaluated': 'Not evaluated', 'not_run': 'Not run'}
            for i, it in enumerate(rq['items']):
                _cell(t, i, 0, it['text'], tip=it['metric'])
                for k, key in ((1, 'plan'), (2, 'as_built')):
                    s = it[key]
                    _cell(t, i, k, word.get(s, s), color=QColor(STATUS['good' if s == 'pass' else 'critical'
                                                                       if s == 'fail' else 'warning']),
                          tip=f'{it[key + "_value"]:.4g}' if it.get(key + '_value') is not None else None)
        png = AB.coverage_png(res, 520)
        if png:
            img = QImage.fromData(png, 'PNG')
            if not img.isNull():
                self.cov_img.setPixmap(QPixmap.fromImage(img).scaled(
                    520, 520, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

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
        """The last check's compact figures (core.asbuilt.summary) for the assistant; None before a run."""
        if self.result is None:
            return None
        return dict(AB.summary(self.result), stale=self.stale)

    # ------------------------------------------------------------------ actions
    def add_variant(self):
        """Add the as-built copy (the confident corrections applied) and switch to it, as one undo step."""
        res = self.result
        if res is None:
            return None
        p = self.ctx.store.project
        v = AB.asbuilt_variant(p, res)
        if v is None:
            self.ctx.msg('No confident correction to apply: the rig as built is the plan.', 4000)
            return None
        state = v.to_dict()

        def fn(pj):
            from ...core.project import Variant
            pj.variants.append(Variant.from_dict(state))
            pj.active_uid = state['uid']
        self.ctx.store.edit('variants', fn, f'As-built check · new variant {v.name}')
        self.ctx.msg(f'Added {v.name}: {v.note}', 6000)
        return v.uid

    def export(self, folder=None):
        res = self.result
        if res is None:
            return []
        if folder is None:
            folder = QFileDialog.getExistingDirectory(self, 'Export the as-built check into')
            if not folder:
                return []
        from ...core import provenance as PV          # where the JSON came from
        paths = PV.stamp_paths(AB.export(res, self.ctx.store.project, folder), self.ctx.store.project, folder)
        self.ctx.msg(f'Exported the as-built check to {folder}', 5000)
        return paths
