"""Simulate ▸ Foundation models: how far a modern detector actually finds each class with each camera (P(detect) =
50 % and 90 %, with intervals) and how far monocular depth stays usable, measured on rendered frames with ground truth
(an existing CARLA recording, or a range sweep of objects stood at known distances in CARLA), beside the geometric
pixels-on-target range. The model manager downloads and removes the models (licence and size shown; models whose
licence does not allow commercial use only on an opt-in); the class list is the open-vocabulary queries. The download
and the run are background jobs with Cancel; nothing runs by itself after an edit (the result is marked stale)."""
from __future__ import annotations

import copy
import json
import math
import os
import threading
import traceback

from PyQt6.QtCore import QObject, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPixmap
from PyQt6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QFileDialog, QGridLayout, QHBoxLayout,
                             QHeaderView, QLabel, QLineEdit, QListWidget, QListWidgetItem, QProgressBar, QScrollArea,
                             QSplitter, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget)

from ...core import detection as DT
from ...core import foundation_models as FM
from ...core import foundation_range as FR
from ..theme import STATUS, THEME, mono
from ..widgets import ROW_H, ScrubSpinBox, SegmentedControl, StatusLine, button, fit_width, hline, label
from .recording_ws import Tile

JOB = 'foundation'
JOB_DL = 'foundation_download'
EMPTY = ('Download a detector (and a depth model), add a CARLA drive recorded with ground truth or choose the range '
         'sweep, then Run.')
MODEL_COLS = (('Use', 'Run this model in the study'), ('Model', ''), ('Task', 'Detection or depth'),
              ('Licence', 'The licence of the model files (hover for where it was checked)'),
              ('Size', 'Download size'), ('Status', 'Downloaded and checked against its published SHA-256'))
CLASS_COLS = (('Class', 'The class name (results and records use it)'),
              ('Query', 'The text the open-vocabulary detectors look for'),
              ('YOLOX', 'The COCO class YOLOX finds it as (empty: YOLOX does not know it)'),
              ('CARLA', 'The actor a range sweep stands in (empty: only ranged in recordings that have it)'),
              ('Height m', ''), ('Width m', ''), ('Min px', 'The pixel height the geometric rule asks for'))
RANGE_COLS = (('Camera', ''), ('Class', ''), ('Model', ''), ('Images', 'Fisheye images or pinhole views'),
              ('Trials', 'Objects (one per frame) the camera saw'), ('Found', ''),
              ('R50', 'Range where P(detect) = 50 % (logistic fit in log range) with its bootstrap interval'),
              ('R90', 'Range where P(detect) = 90 %'),
              ('Geometric', 'core.perception\'s pixels-on-target range along the same line'),
              ('R50 / geo', 'Measured over geometric: < 1 the pixel rule over-promises, > 1 it under-promises'),
              ('H50 px', 'The object\'s image height at which P(detect) = 50 %'))
DEPTH_COLS = (('Camera', ''), ('Depth bin', 'Ground-truth planar depth'), ('Pixels', ''), ('AbsRel', ''),
              ('δ<1.25', 'Share of pixels within 25 % of the truth'))
TABS = ('Ranges', 'P(detect)', 'Depth', 'Examples', 'Notes')
FIELDS = (
    ('iou', 'IoU', '', 2, 0.05, 'A detection matches an object when their boxes overlap at least this much'),
    ('iou_small', 'IoU, small objects', '', 2, 0.05, 'The IoU for objects less than 32 px tall'),
    ('max_occluded', 'Skip occluded over', '', 2, 0.05, 'Objects hidden more than this share are not trials'),
    ('max_truncated', 'Skip truncated over', '', 2, 0.05, 'Objects more than this share outside the image are not '
                                                          'trials'),
    ('frame_step', 'Every n-th frame', '', 0, 1.0, 'Recordings: evaluate every n-th camera frame'),
    ('abs_rel_max', 'Usable depth AbsRel', '', 2, 0.01, 'Depth is usable out to the farthest bin with AbsRel below '
                                                        'this'),
    ('bootstrap', 'Bootstrap', '', 0, 50.0, 'Resamples for the intervals (0: none)'),
)
SCORE_FIELDS = (('yolox', 'YOLOX score'), ('owlv2', 'OWLv2 score'), ('grounding_dino', 'Grounding DINO score'))
SWEEP_TEXTS = (('sweep_cameras', 'Cameras', 'The cameras the sweep renders, by tag, comma separated'),
               ('sweep_classes', 'Classes', 'The classes stood in CARLA (they need a CARLA actor), comma separated'),
               ('ranges_m', 'Ranges m', 'Distances along each line, comma separated'),
               ('offsets_deg', 'Off axis °', 'Lines off each camera\'s heading (0: along it), comma separated'),
               ('road_lateral_m', 'Road lines m', 'Lines along the vehicle at these lateral offsets (left +), '
                                                  'ahead and behind: the side cameras\' off-axis view'),
               ('seeds', 'Seeds', 'Each scene is driven at each seed (another straight road: another background)'),
               ('town', 'Town', 'The CARLA town (the small towns are quickest)'))


def _table(cols, stretch=None, editable=False):
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
    if not editable:
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


def _cell(t, r, c, text, num=False, tip=None, color=None, editable=False):
    it = QTableWidgetItem(text)
    if num:
        it.setFont(mono(12))
        it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    if tip:
        it.setToolTip(tip)
    if color is not None:
        it.setForeground(color)
    if not editable:
        it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
    t.setItem(r, c, it)
    return it


def _pix(data: bytes) -> QPixmap:
    img = QImage.fromData(data)
    return QPixmap.fromImage(img)



def farthest_measured(fits):
    """The fit with the longest measured R50 over every camera, class and model (the 'all' subset), and the fits whose
    R50 lies beyond the farthest distance tested: those have no R50, only a bound, so they never rank as the farthest
    (a "> 100 m" there would say only how far the test went)."""
    best, beyond = None, []
    for f in fits:
        if f['subset'] != 'all':
            continue
        rf = f['range']
        if rf.get('beyond', [False])[0] and rf.get('tested'):
            beyond.append(f)
            continue
        x = rf['x'][0]
        if x is None or rf.get('flat') or rf.get('below', [False])[0]:
            continue
        if best is None or x > best['range']['x'][0]:
            best = f
    return best, beyond

class _Sig(QObject):
    done = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)
    progress = pyqtSignal(int, float, str)


class FoundationPage(QWidget):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.result: FR.FoundationResult | None = None
        self.detectors = None                 # {model id: callable} (tests); None: the settings' models
        self.depth = None                     # a depth callable (tests)
        self.runner = None                    # the sweep's bridge (tests: the fake carla)
        self.sim_settings = None
        self.busy = False
        self.dl_busy = False
        self.status_text = ''
        self._gen = 0
        self._dl_gen = 0
        self._for = None
        self._dirty = True
        self._syncing = False
        self.sig = _Sig()
        self.sig.done.connect(self._thread_done)
        self.sig.failed.connect(self._failed)
        self.sig.progress.connect(self._progress)
        self.setObjectName('base')
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setHandleWidth(2)
        split.addWidget(self._left())
        split.addWidget(self._right())
        split.setSizes([420, 980])
        lay.addWidget(split)
        ctx.store.changed.connect(self._on_change)
        ctx.store.project_replaced.connect(self._on_replaced)
        THEME.changed.connect(self._retheme)
        self._sync_fields()
        self.refresh()

    # ------------------------------------------------------------------ left
    def _left(self):
        w = QWidget()
        w.setObjectName('panel')
        ll = QVBoxLayout(w)
        ll.setContentsMargins(8, 6, 8, 8)
        ll.setSpacing(4)
        top = QHBoxLayout()
        top.addWidget(label('Foundation models'))
        top.addStretch(1)
        top.addWidget(label('detection range · depth', muted=True, small=True))
        ll.addLayout(top)
        ll.addWidget(hline())
        ll.addWidget(label('How far a modern detector actually finds each class with each camera, and how far '
                           'monocular depth stays usable, measured on rendered frames with ground truth, beside the '
                           'geometric pixels-on-target range.', muted=True, small=True, wrap=True))
        # models
        ll.addWidget(label('Models', ink2=True))
        self.model_table = _table(MODEL_COLS, stretch=1)
        self.model_table.setMinimumHeight(ROW_H * 8)
        self.model_table.itemChanged.connect(self._model_item_changed)
        ll.addWidget(self.model_table)
        self.opt_in = QCheckBox('Offer models whose licence does not allow commercial use')
        self.opt_in.setToolTip('Depth Anything V2 Base is CC-BY-NC-4.0 (non-commercial); the metric outdoor model is '
                               'fine-tuned on Virtual KITTI 2 (CC BY-NC-SA 3.0). Tick only if your use allows it.')
        self.opt_in.toggled.connect(lambda on: self._set('licence opt-in ' + ('on' if on else 'off'), opt_in=on))
        ll.addWidget(self.opt_in)
        self.model_status = StatusLine('warning', '')
        ll.addWidget(self.model_status)
        bl = QHBoxLayout()
        self.b_download = button('Download', None, 'import', 'Download the selected model from its pinned revision '
                                                            '(every file checked against its SHA-256)')
        self.b_download.clicked.connect(lambda: self.download())
        bl.addWidget(self.b_download)
        self.b_remove = button('Remove', 'ghost', 'trash', 'Delete the selected model\'s files')
        self.b_remove.clicked.connect(lambda: self.remove_model())
        bl.addWidget(self.b_remove)
        bl.addStretch(1)
        self.device_label = label('', muted=True, small=True)
        bl.addWidget(self.device_label)
        ll.addLayout(bl)
        dl = QHBoxLayout()
        dl.addWidget(label('Depth model', ink2=True))
        self.depth_pick = QComboBox()
        self.depth_pick.setToolTip('The monocular depth model (relative models are scaled per frame from the ground '
                                   'plane)')
        self.depth_pick.currentIndexChanged.connect(self._depth_picked)
        dl.addWidget(self.depth_pick, 1)
        self.device_pick = SegmentedControl([('auto', 'Auto'), ('cuda', 'GPU'), ('cpu', 'CPU')], 'auto')
        self.device_pick.setToolTip('Where the models run: the GPU when there is one with enough free memory')
        self.device_pick.changed.connect(lambda k: self._set(f'device {k}', device=k))
        dl.addWidget(self.device_pick)
        ll.addLayout(dl)
        ll.addWidget(hline())
        # classes
        ll.addWidget(label('Classes and queries', ink2=True))
        self.class_table = _table(CLASS_COLS, stretch=1, editable=True)
        self.class_table.setMinimumHeight(ROW_H * 7)
        self.class_table.itemChanged.connect(self._class_item_changed)
        ll.addWidget(self.class_table)
        cb = QHBoxLayout()
        b = button('Add class', 'ghost', 'plus', 'Add a class: give it a name and a text query (open-vocabulary '
                                                 'detectors find what the query describes)')
        b.clicked.connect(lambda: self.add_class())
        cb.addWidget(b)
        b = button('Remove class', 'ghost', 'minus', 'Remove the selected class')
        b.clicked.connect(lambda: self.remove_class())
        cb.addWidget(b)
        b = button('Default classes', 'ghost', None, 'Back to the default class list')
        b.clicked.connect(lambda: self._set('default classes', classes=copy.deepcopy(FR.DEFAULT_CLASSES)))
        cb.addWidget(b)
        cb.addStretch(1)
        ll.addLayout(cb)
        ll.addWidget(hline())
        # source
        ll.addWidget(label('Frames', ink2=True))
        self.source = SegmentedControl([('recordings', 'Recordings'), ('sweep', 'Range sweep in CARLA')],
                                       'recordings')
        self.source.setToolTip('An existing CARLA drive with ground truth (no server needed), or objects stood at '
                               'known distances in CARLA and rendered through the rig\'s cameras')
        self.source.changed.connect(lambda k: self._set(f'frames from {k}', source=k))
        ll.addWidget(self.source, 0, Qt.AlignmentFlag.AlignLeft)
        self.recs = QListWidget()
        self.recs.setMinimumHeight(60)
        self.recs.setMaximumHeight(100)
        self.recs.setToolTip('CARLA drives recorded with ground truth')
        ll.addWidget(self.recs)
        rb = QHBoxLayout()
        self.b_add_rec = button('Add…', None, 'folder', 'Add a CARLA run folder (one with ground_truth/index.json)')
        self.b_add_rec.clicked.connect(lambda: self.add_recording())
        rb.addWidget(self.b_add_rec)
        self.b_rm_rec = button('Remove', 'ghost', 'trash', 'Remove the selected recording from the list (nothing is '
                                                         'deleted)')
        self.b_rm_rec.clicked.connect(lambda: self.remove_recording())
        rb.addWidget(self.b_rm_rec)
        rb.addStretch(1)
        ll.addLayout(rb)
        g = QGridLayout()
        g.setHorizontalSpacing(6)
        g.setVerticalSpacing(3)
        self.texts = {}
        for i, (key, text, tip) in enumerate(SWEEP_TEXTS):
            e = QLineEdit()
            e.setToolTip(tip)
            e.setFont(mono(12))
            e.editingFinished.connect(lambda k=key, ed=e, t=text: self._text_edited(k, ed, t))
            a = label(text, ink2=True)
            a.setToolTip(tip)
            g.addWidget(a, i, 0)
            g.addWidget(e, i, 1)
            self.texts[key] = e
        self.sweep_grid = QWidget()
        self.sweep_grid.setLayout(g)
        ll.addWidget(self.sweep_grid)
        self.sweep_note = label('', muted=True, small=True, wrap=True)
        ll.addWidget(self.sweep_note)
        ll.addWidget(hline())
        # matching
        ll.addWidget(label('Matching, depth and statistics', ink2=True))
        g = QGridLayout()
        g.setHorizontalSpacing(6)
        g.setVerticalSpacing(3)
        self.fields = {}
        for i, (key, text, unit, dec, step, tip) in enumerate(FIELDS):
            f = ScrubSpinBox('', 'plain', dec, unit, step=step, width=110)
            f.setToolTip(tip)
            f.committed.connect(lambda v, k=key, t=text: self._set(f'{t.lower()} {v:g}', **{k: v}))
            a = label(text, ink2=True)
            a.setFixedWidth(150)
            a.setToolTip(tip)
            g.addWidget(a, i, 0)
            g.addWidget(f, i, 1)
            self.fields[key] = f
        self.score_fields = {}
        for j, (fam, text) in enumerate(SCORE_FIELDS, len(FIELDS)):
            f = ScrubSpinBox('', 'plain', 2, '', step=0.05, width=110)
            f.setToolTip(f'The operating point of {text.split(" score")[0]}: detections scoring at least this count')
            f.committed.connect(lambda v, k=fam: self._score_set(k, v))
            g.addWidget(label(text, ink2=True), j, 0)
            g.addWidget(f, j, 1)
            self.score_fields[fam] = f
        ll.addLayout(g)
        ll.addWidget(label('Images', ink2=True))
        self.views = SegmentedControl([('fisheye', 'Fisheye'), ('pinhole', 'Pinhole views'), ('both', 'Both')],
                                      'both')
        self.views.setToolTip('Detect on the fisheye images as recorded, on rectified pinhole views cut from them, '
                              'or on both (depth always on the pinhole views)')
        self.views.changed.connect(lambda k: self._set(f'images {k}', views=k))
        ll.addWidget(self.views, 0, Qt.AlignmentFlag.AlignLeft)
        ll.addWidget(label(FR.ASSUMPTIONS, muted=True, small=True, wrap=True))
        self.status = label('', muted=True, small=True, wrap=True)
        ll.addWidget(self.status)
        self.bar = QProgressBar()
        self.bar.setTextVisible(False)
        self.bar.setRange(0, 1000)
        self.bar.hide()
        ll.addWidget(self.bar)
        bl = QHBoxLayout()
        b_def = button('Defaults', 'ghost', None, 'Back to the default settings (the recordings stay)')
        b_def.clicked.connect(self._defaults)
        bl.addWidget(b_def)
        bl.addStretch(1)
        self.b_export = button('Export…', None, 'export', 'The study as JSON and CSV, the plots as PNG, the example '
                                                          'frames and the perception records')
        self.b_export.clicked.connect(lambda: self.export())
        bl.addWidget(self.b_export)
        self.b_records = button('To rig model', 'ghost', 'send', 'Add this run\'s perception records (per region, '
                                                               'vantage.perception_record/1) to the learned rig '
                                                               'model\'s store (Analysis ▸ Rig model)')
        self.b_records.clicked.connect(lambda: self.send_records())
        bl.addWidget(self.b_records)
        self.b_cancel = button('Cancel', 'ghost', None, 'Stop the run or the download')
        self.b_cancel.clicked.connect(lambda: self.cancel())
        bl.addWidget(self.b_cancel)
        self.b_run = button('Run', 'primary', 'play', 'Detect, fit the ranges and score the depth (a background '
                                                      'job)')
        self.b_run.clicked.connect(lambda: self.run())
        bl.addWidget(self.b_run)
        ll.addLayout(bl)
        ll.addStretch(1)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setWidget(w)
        fit_width(sc)
        sc.setMinimumWidth(400)
        return sc

    # ------------------------------------------------------------------ right
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
                ('trials', 'Trials', 'Objects the cameras saw, one per frame, over every model and image kind'),
                ('r50', 'Farthest R50', 'The longest 50 % detection range of any camera, class and model'),
                ('geo', 'Geometry', 'Median measured R50 over the geometric range (1: the pixel rule is right)'),
                ('depth', 'Usable depth', 'The farthest usable monocular depth over the cameras'),
                ('device', 'Ran on', 'GPU or CPU')):
            t = self.tiles[key] = Tile(caption, tip)
            tiles.addWidget(t, 1)
        rl.addLayout(tiles)
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.range_table = _table(RANGE_COLS)
        self.tabs.addTab(self.range_table, 'Ranges')
        pw = QWidget()
        pl = QVBoxLayout(pw)
        pl.setContentsMargins(0, 6, 0, 0)
        pick = QHBoxLayout()
        self.cam_pick = QComboBox()
        self.cam_pick.setToolTip('The camera shown')
        self.cam_pick.currentIndexChanged.connect(lambda *_: self._show_plot())
        pick.addWidget(self.cam_pick, 1)
        self.cls_pick = QComboBox()
        self.cls_pick.setToolTip('The class shown')
        self.cls_pick.currentIndexChanged.connect(lambda *_: self._show_plot())
        pick.addWidget(self.cls_pick, 1)
        self.mode_pick = SegmentedControl([('fisheye', 'Fisheye'), ('pinhole', 'Views')], 'fisheye')
        self.mode_pick.changed.connect(lambda *_: self._show_plot())
        pick.addWidget(self.mode_pick)
        pl.addLayout(pick)
        self.plot = QLabel()
        self.plot.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.plot.setMinimumHeight(300)
        pl.addWidget(self.plot, 1)
        pl.addWidget(label('Dots: the share detected per range bin (size: how many trials; whiskers: Wilson 95 %); '
                           'line: the logistic fit; diamond: R50; dashed: the geometric range.', muted=True,
                           small=True, wrap=True))
        self.tabs.addTab(pw, 'P(detect)')
        dw = QSplitter(Qt.Orientation.Vertical)
        dw.setHandleWidth(2)
        self.depth_plot = QLabel()
        self.depth_plot.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.depth_plot.setMinimumHeight(240)
        dw.addWidget(self.depth_plot)
        self.depth_table = _table(DEPTH_COLS)
        dw.addWidget(self.depth_table)
        self.tabs.addTab(dw, 'Depth')
        ew = QWidget()
        el = QVBoxLayout(ew)
        el.setContentsMargins(0, 6, 0, 0)
        self.example_pick = QComboBox()
        self.example_pick.setToolTip('An example frame per model, image kind and camera (the one with the most '
                                     'objects)')
        self.example_pick.currentIndexChanged.connect(lambda *_: self._show_example())
        el.addWidget(self.example_pick)
        self.example = QLabel()
        self.example.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.example.setMinimumHeight(300)
        el.addWidget(self.example, 1)
        el.addWidget(label('Objects: found (green), missed (orange), not a trial (grey). Detections at the operating '
                           'point: matched (blue), false (magenta), other (grey).', muted=True, small=True, wrap=True))
        self.tabs.addTab(ew, 'Examples')
        self.notes = label('', muted=True, small=True, wrap=True)
        self.notes.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.notes.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setWidget(self.notes)
        self.tabs.addTab(sc, 'Notes')
        rl.addWidget(self.tabs, 1)
        return w

    # ------------------------------------------------------------------ settings
    @property
    def settings(self) -> FR.FoundationSettings:
        return FR.settings_of(self.ctx.store.project)

    def _store(self, st: FR.FoundationSettings, text) -> bool:
        new = FR.FoundationSettings.from_dict(st.to_dict()).to_dict()
        if new == self.settings.to_dict():
            self._sync_fields()
            return False

        def fn(p):
            p.analysis.foundation = new
        self.ctx.store.edit('analysis', fn, f'Foundation models · {text}')
        return True

    def _set(self, text, **kw):
        if self._syncing:
            return False
        st = self.settings
        for k, v in kw.items():
            setattr(st, k, v)
        return self._store(st, text)

    def _score_set(self, fam, v):
        if self._syncing:
            return False
        st = self.settings
        st.scores[fam] = float(v)
        return self._store(st, f'{fam} score {v:g}')

    def _defaults(self):
        st = self.settings
        self._store(FR.FoundationSettings(recordings=list(st.recordings), source=st.source), 'defaults')

    def _text_edited(self, key, ed, text):
        if self._syncing:
            return
        raw = ed.text().strip()
        if key == 'town':
            self._set(f'town {raw}', town=raw)
            return
        vals = [x.strip() for x in raw.replace(';', ',').split(',') if x.strip()]
        self._set(f'{text.lower()} {raw}', **{key: vals})

    def _model_item_changed(self, it):
        if self._syncing or it.column() != 0:
            return
        mid = it.data(Qt.ItemDataRole.UserRole)
        st = self.settings
        if mid in FM.DEPTH_MODELS:
            if it.checkState() == Qt.CheckState.Checked:
                self._set(f'depth {mid}', depth_model=mid)
            elif st.depth_model == mid:
                self._set('no depth model', depth_model='')
            return
        on = it.checkState() == Qt.CheckState.Checked
        dets = [m for m in st.detectors if m != mid] + ([mid] if on else [])
        self._set(f'{FM.model_label(mid)} {"on" if on else "off"}', detectors=dets)

    def _depth_picked(self, i):
        if self._syncing:
            return
        self._set('depth model', depth_model=self.depth_pick.itemData(i) or '')

    def _class_item_changed(self, it):
        if self._syncing:
            return
        st = self.settings
        r, c = it.row(), it.column()
        if r >= len(st.classes):
            return
        key = ('name', 'query', 'coco', 'carla', 'height', 'width', 'min_px')[c]
        val = it.text().strip()
        cl = st.classes[r]
        if key in ('height', 'width', 'min_px'):
            try:
                cl[key] = float(val)
            except ValueError:
                self._sync_fields()
                return
        else:
            cl[key] = val
        self._store(st, f'class {cl["name"]} {key}')

    def add_class(self, name=None, query=None) -> bool:
        st = self.settings
        n = name or f'class {len(st.classes) + 1}'
        st.classes.append({'name': n, 'query': query or f'a {n}', 'height': 1.0, 'width': 0.5, 'min_px': 16.0})
        return self._store(st, f'add class {n}')

    def remove_class(self, row=None) -> bool:
        st = self.settings
        r = self.class_table.currentRow() if row is None else row
        if not 0 <= r < len(st.classes) or len(st.classes) <= 1:
            return False
        name = st.classes.pop(r)['name']
        return self._store(st, f'remove class {name}')

    def add_recording(self, path=None) -> bool:
        if path is None:
            path = QFileDialog.getExistingDirectory(self, 'A CARLA run with ground truth')
            if not path:
                return False
        st = self.settings
        if path in st.recordings:
            return False
        st.recordings.append(path)
        return self._store(st, f'add {os.path.basename(path)}')

    def remove_recording(self, row=None) -> bool:
        st = self.settings
        r = self.recs.currentRow() if row is None else row
        if not 0 <= r < len(st.recordings):
            return False
        p = st.recordings.pop(r)
        return self._store(st, f'remove {os.path.basename(p)}')

    def _key(self, p=None):
        p = p or self.ctx.store.project
        return json.dumps(FR.settings_of(p).to_dict(), sort_keys=True)

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
        self._fill_pickers()
        self.refresh()

    def _retheme(self, *_):
        self._show()

    def showEvent(self, e):
        super().showEvent(e)
        self.activate()

    def activate(self):
        if self._dirty:
            self._dirty = False
            self._sync_fields()
        self.refresh()

    def _sync_fields(self):
        st = self.settings
        self._syncing = True
        try:
            self.opt_in.setChecked(st.opt_in)
            cat = FM.catalogue(st.opt_in)
            self.model_table.setRowCount(len(cat))
            for r, m in enumerate(cat):
                use = _cell(self.model_table, r, 0, '')
                use.setData(Qt.ItemDataRole.UserRole, m['id'])
                flags = Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
                if not m['allowed']:
                    flags &= ~Qt.ItemFlag.ItemIsEnabled
                use.setFlags(flags)
                on = (m['id'] in st.detectors) if m['task'] == 'detect' else (m['id'] == st.depth_model)
                use.setCheckState(Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)
                name = _cell(self.model_table, r, 1, m['name'], tip=f'{m["paper"]}\n{m["repo"]}')
                name.setData(Qt.ItemDataRole.UserRole, m['id'])
                task = 'detect' if m['task'] == 'detect' else f'depth ({m.get("depth_kind") or ""})'
                _cell(self.model_table, r, 2, task + (' · COCO' if m.get('closed_vocabulary') else ''))
                col = None if m['commercial'] else QColor(STATUS['warning'])
                _cell(self.model_table, r, 3, m['licence'] + ('' if m['commercial'] else ' · opt-in'),
                      tip=m['licence_note'], color=col)
                _cell(self.model_table, r, 4, f'{m["size"] / 1e6:.0f} MB', num=True)
                _cell(self.model_table, r, 5, 'ready' if m['verified'] else ('check failed' if m['present'] else
                                                                            'not downloaded'), tip=m['text'],
                      color=QColor(STATUS['good'] if m['verified'] else STATUS['warning']))
            self.depth_pick.clear()
            self.depth_pick.addItem('No depth', '')
            for mid in FM.DEPTH_MODELS:
                if FM.allowed(mid, st.opt_in):
                    self.depth_pick.addItem(FM.SPECS[mid].name, mid)
            self.depth_pick.setCurrentIndex(max(self.depth_pick.findData(st.depth_model), 0))
            self.device_pick.set_value(st.device)
            self.class_table.setRowCount(len(st.classes))
            for r, c in enumerate(st.classes):
                for k, key in enumerate(('name', 'query', 'coco', 'carla')):
                    _cell(self.class_table, r, k, str(c.get(key) or ''), editable=True)
                for k, key in enumerate(('height', 'width', 'min_px'), 4):
                    _cell(self.class_table, r, k, f'{c[key]:g}', num=True, editable=True)
            self.source.set_value(st.source)
            self.recs.clear()
            for p in st.recordings:
                it = QListWidgetItem(os.path.basename(p.rstrip('/')) or p)
                it.setToolTip(p)
                self.recs.addItem(it)
            for key, e in self.texts.items():
                v = getattr(st, key)
                e.setText(v if isinstance(v, str) else ', '.join(f'{x:g}' if isinstance(x, (int, float)) else str(x)
                                                                 for x in v))
            sweep = st.source == 'sweep'
            self.sweep_grid.setVisible(sweep)
            self.sweep_note.setVisible(sweep)
            for wdg in (self.recs, self.b_add_rec, self.b_rm_rec):
                wdg.setVisible(not sweep)
            if sweep:
                self.sweep_note.setText(self._sweep_text(st))
            for k, f in self.fields.items():
                f.set_value(getattr(st, k))
            for k, f in self.score_fields.items():
                f.set_value(st.scores.get(k, FR.DEFAULT_SCORES[k]))
            self.views.set_value(st.views)
        finally:
            self._syncing = False
        self._sync_model()

    def _sweep_text(self, st) -> str:
        try:
            pl = FR.sweep_plan(self.ctx.store.project, st)
        except Exception as e:                       # noqa: BLE001 (a plan that cannot be made says why)
            return f'The sweep cannot be planned: {e}'
        bits = pl['notes'][-1:] + pl['notes'][:-1]
        return ' '.join(bits) + ' Needs a CARLA server (Simulate ▸ CARLA).'

    def _sync_model(self):
        st = self.settings
        why = FM.torch_missing()
        torch_need = any(m in FM.SPECS for m in st.detectors) or bool(st.depth_model)
        missing = [FM.model_label(m) for m in list(st.detectors) + ([st.depth_model] if st.depth_model else [])
                   if not (DT.status(m) if m in DT.MODELS else FM.status(m, verify=False))['verified']]
        if why and torch_need and self.detectors is None:
            self.model_status.set('serious', why)
        elif missing and self.detectors is None:
            self.model_status.set('warning', f'Not downloaded: {", ".join(missing)}. Select a row and press '
                                             'Download.')
        else:
            self.model_status.set('good', 'The chosen models are ready.')
        self.device_label.setText(FM.device_info(st.device, st.gpu_mem_gb)['text'] if not why else 'no torch')
        self.b_download.setEnabled(not self.dl_busy)

    # ------------------------------------------------------------------ jobs
    def _start(self, kind, title, work, gen_attr):
        win = self.ctx.window
        gen = getattr(self, gen_attr)
        if win is not None and hasattr(win, 'run_job'):
            return win.run_job(kind, title, work, lambda out: self._thread_done(gen, (kind, out)),
                               lambda m: self._failed(gen, m))

        def thread():
            try:
                out = work(lambda f, t='': self.sig.progress.emit(gen, f, t), lambda: gen != getattr(self, gen_attr))
                self.sig.done.emit(gen, (kind, out))
            except Exception as e:              # noqa: BLE001
                try:
                    self.sig.failed.emit(gen, f'{type(e).__name__}: {e}\n{traceback.format_exc()}')
                except RuntimeError:
                    pass
        threading.Thread(target=thread, daemon=True).start()
        return True

    def _progress(self, gen, f, text):
        if gen in (self._gen, self._dl_gen):
            self.bar.show()
            self.bar.setValue(int(f * 1000))
            if text:
                self.status.setText(text)

    def selected_model(self) -> str | None:
        r = self.model_table.currentRow()
        it = self.model_table.item(r, 1) if r >= 0 else None
        return it.data(Qt.ItemDataRole.UserRole) if it is not None else None

    def download(self, mid=None, fetch=None) -> bool:
        mid = mid or self.selected_model()
        if not mid or self.dl_busy:
            return False
        st = self.settings
        if mid in FM.SPECS and not FM.allowed(mid, st.opt_in):
            self.status_text = f'{FM.SPECS[mid].name}: {FM.SPECS[mid].licence_note}'
            self.refresh()
            return False
        self._dl_gen += 1
        self.dl_busy = True
        self.status_text = f'Downloading {FM.model_label(mid)}…'
        opt = st.opt_in

        def work(progress, cancel):
            if mid in DT.MODELS:
                return DT.download(mid, progress, cancel)
            return FM.download(mid, progress, cancel, opt_in=opt, fetch=fetch)
        ok = self._start(JOB_DL, f'Download {FM.model_label(mid)}', work, '_dl_gen')
        if not ok:
            self.dl_busy = False
        self.refresh()
        return ok

    def remove_model(self, mid=None) -> bool:
        mid = mid or self.selected_model()
        if not mid:
            return False
        done = DT.remove(mid) if mid in DT.MODELS else FM.remove(mid)
        self.ctx.msg(f'Removed {FM.model_label(mid)}' if done else f'{FM.model_label(mid)} was not downloaded', 4000)
        self._sync_fields()
        return done

    def _thread_done(self, gen, payload):
        kind, out = payload
        if kind == JOB_DL:
            if gen != self._dl_gen:
                return
            self.dl_busy = False
            self.status_text = 'Download cancelled' if out is None else f'Downloaded {os.path.basename(out)}'
            self.ctx.msg(self.status_text, 5000)
            self._sync_fields()
            self.refresh()
            return
        if gen != self._gen:
            return
        self.busy = False
        key, res = out if out is not None else (None, None)
        if res is None or key[0] is not self.ctx.store.project:
            self.status_text = 'Cancelled' if res is None else ''
            self.refresh()
            return
        self.result, self._for = res, key
        self.status_text = f'{len(res.trials["hit"])} trials, {res.timing["images"]} frames in {res.seconds:.1f} s'
        self.ctx.store.project.add_log('foundation', f'{res.source}: {len(res.fits)} range fits, '
                                                     f'{len(res.records)} perception records')
        self.ctx.msg(f'Foundation models · {self.status_text}', 5000)
        self._fill_pickers()
        self.refresh()

    def _failed(self, gen, msg):
        if gen == self._dl_gen and self.dl_busy:
            self.dl_busy = False
            self.status_text = 'Download failed: ' + msg.splitlines()[0]
            self._sync_fields()
        elif gen == self._gen and self.busy:
            self.busy = False
            self.status_text = 'Failed: ' + msg.splitlines()[0]
        self.refresh()

    def run(self) -> bool:
        if self.busy:
            return False
        st = self.settings
        if st.source == 'recordings' and not st.recordings:
            self.status_text = 'Add a recording with ground truth first.'
            self.refresh()
            return False
        if self.detectors is None:
            try:
                FR._check_models(st, verify=False)       # sizes here; the run checks every SHA-256
            except (ValueError, FileNotFoundError) as e:
                self.status_text = str(e)
                self.refresh()
                return False
        p = self.ctx.store.project
        key = (p, self._key(p))
        dets, dep, runner, sim = self.detectors, self.depth, self.runner, self.sim_settings
        win = self.ctx.window
        eng = win._shared_engine() if win is not None and hasattr(win, '_shared_engine') else None
        self._gen += 1
        self.busy = True
        self.status_text = 'Running…'
        self.bar.setValue(0)
        gen, sig = self._gen, self.sig

        def work(progress, cancel):
            def both(f, text=''):
                progress(f, text)
                try:
                    sig.progress.emit(gen, f, text)
                except RuntimeError:
                    pass
            return key, FR.run(p, eng, st, progress=both, cancel=cancel, detectors=dets, depth=dep,
                               sim_settings=sim, runner=runner)
        ok = self._start(JOB, 'Foundation models', work, '_gen')
        if not ok:
            self.busy = False
            self.status_text = ''
        self.refresh()
        return ok

    def cancel(self):
        win = self.ctx.window
        for job in (JOB, JOB_DL):
            j = getattr(win, '_jobs', {}).get(job) if win is not None else None
            if j is not None:
                j['cancel'].set()
        if win is None or not hasattr(win, 'run_job'):
            if self.busy:
                self._gen += 1
                self.busy = False
                self.status_text = 'Cancelled'
            if self.dl_busy:
                self._dl_gen += 1
                self.dl_busy = False
                self.status_text = 'Download cancelled'
            self.refresh()

    def export(self, folder=None) -> list:
        res = self.result
        if res is None:
            return []
        if folder is None:
            folder = QFileDialog.getExistingDirectory(self, 'Export the foundation-model study into')
            if not folder:
                return []
        from ...core import provenance as PV
        paths = PV.stamp_paths(FR.export(res, self.ctx.store.project, folder), self.ctx.store.project, folder)
        self.ctx.msg(f'Exported the foundation-model study to {folder}', 5000)
        return paths

    def send_records(self, store=None) -> dict | None:
        """The result's perception records to core.perf_model's store (store: a path; None: the user's)."""
        if self.result is None or not self.result.records:
            return None
        out = FR.store_records(self.result, None, store)
        if out['how'].startswith('core.perf_model'):
            self.ctx.msg(f'{out["stored"]} perception records added to the rig model\'s store'
                         + (f', {out["duplicates"]} already there' if out['duplicates'] else '')
                         + (f'; {len(out["errors"])} refused' if out['errors'] else ''), 6000)
        else:
            self.ctx.msg('The rig model is not in this build: Export writes the records as JSON.', 6000)
        return out

    def summary(self) -> dict | None:
        return FR.summary(self.result) if self.result is not None else None

    # ------------------------------------------------------------------ showing the result
    def refresh(self):
        st = self.settings
        busy = self.busy or self.dl_busy
        ready = st.source == 'sweep' or bool(st.recordings)
        self.b_run.setEnabled(not self.busy and ready)
        self.b_cancel.setVisible(busy)
        self.b_export.setEnabled(self.result is not None and not self.busy)
        self.b_records.setEnabled(self.result is not None and bool(self.result.records) and not self.busy)
        self.bar.setVisible(busy)
        txt = self.status_text
        if self.stale and not self.busy:
            txt = (txt + ' · ' if txt else '') + 'The settings changed since the run: Run again.'
        if not ready and self.result is None and not txt:
            txt = EMPTY
        self.status.setText(txt)
        self._show()

    def _fill_pickers(self):
        res = self.result
        for cb in (self.cam_pick, self.cls_pick, self.example_pick):
            cb.blockSignals(True)
            cb.clear()
        if res is not None:
            for c in res.cameras():
                self.cam_pick.addItem(c, c)
            for c in res.classes():
                self.cls_pick.addItem(c, c)
            for i, e in enumerate(res.examples):
                self.example_pick.addItem(e['caption'], i)
        for cb in (self.cam_pick, self.cls_pick, self.example_pick):
            cb.blockSignals(False)

    def _show(self):
        res = self.result
        dark = THEME.name == 'dark'
        if res is None:
            for t in self.tiles.values():
                t.set('—')
            self.range_table.setRowCount(0)
            self.depth_table.setRowCount(0)
            self.plot.setText(EMPTY)
            self.depth_plot.setText('Run with a depth model to see its error against depth.')
            self.example.setText('')
            self.notes.setText(self._notes_text(None))
            return
        n = len(res.trials['hit'])
        self.tiles['trials'].set(f'{n}', '', f'{len(res.recordings)} drive(s)')
        best, beyond = farthest_measured(res.fits)
        far = f' · {len(beyond)} more beyond the tested {beyond[0]["range"]["tested"][1]:g} m' if beyond else ''
        if best is not None:
            self.tiles['r50'].set(FR.range_text(best, 0).split('  ')[0], '',
                                  f'{best["camera"]} · {best["class"]} · {FM.model_label(best["model"])}{far}')
        elif beyond:
            self.tiles['r50'].set('beyond', '', f'every fit runs past the tested {beyond[0]["range"]["tested"][1]:g} m: '
                                                 'no R50 measured', 'warning')
        else:
            self.tiles['r50'].set('—')
        ratios = [f['ratio50'] for f in res.fits if f['subset'] == 'all' and f.get('ratio50')]
        if ratios:
            import numpy as np
            med = float(np.median(ratios))
            self.tiles['geo'].set(f'{med:.2f}', '×', 'measured R50 / geometric',
                                  'good' if 0.8 <= med <= 1.25 else 'warning')
        else:
            self.tiles['geo'].set('—')
        cams = (res.depth or {}).get('cameras') or {}
        if cams:
            u = max(c['usable_m'] for c in cams.values())
            self.tiles['depth'].set('> ' + f'{res.depth["edges"][-1]:g}' if u == math.inf else f'{u:g}', 'm',
                                    res.depth.get('name') or '')
        else:
            self.tiles['depth'].set('—')
        self.tiles['device'].set(str(res.device.get('device', '—')).upper(), '', res.device.get('text', ''))
        rows = [f for f in res.fits if f['subset'] == 'all']
        self.range_table.setRowCount(len(rows))
        for r, f in enumerate(rows):
            vals = (f['camera'], f['class'], FM.model_label(f['model']), FR.MODE_LABEL.get(f['mode'], f['mode']),
                    f'{f["n"]}', f'{f["k"]}', FR.range_text(f, 0), FR.range_text(f, 1),
                    '—' if not f.get('geometric_m') else f'{f["geometric_m"]:.1f} m',
                    '—' if not f.get('ratio50') else f'{f["ratio50"]:.2f}',
                    '—' if f['height']['x'][0] is None else f'{f["height"]["x"][0]:.0f}')
            for c, v in enumerate(vals):
                col = None
                if c == 9 and f.get('ratio50'):
                    col = QColor(STATUS['good'] if 0.8 <= f['ratio50'] <= 1.25 else STATUS['warning'])
                _cell(self.range_table, r, c, v, num=c >= 4, color=col)
        modes = res.settings.modes()
        for k, b in self.mode_pick.buttons.items():
            b.setEnabled(k in modes)
        if self.mode_pick.value() not in modes:
            self.mode_pick.set_value(modes[0])
        self._show_plot()
        if res.depth:
            self.depth_plot.setPixmap(_pix(FR.depth_png(res, dark)))
            drows = [(tag, b) for tag, c in sorted(cams.items()) for b in c['bins'] if b['n_px']]
            self.depth_table.setRowCount(len(drows))
            for r, (tag, b) in enumerate(drows):
                for c, v in enumerate((tag, b['bin'], f'{b["n_px"]:,}',
                                       '—' if b['abs_rel'] is None else f'{b["abs_rel"] * 100:.1f} %',
                                       '—' if b['delta1'] is None else f'{b["delta1"] * 100:.1f} %')):
                    _cell(self.depth_table, r, c, v, num=c >= 2)
        else:
            self.depth_plot.setText('No depth model in this run.')
            self.depth_table.setRowCount(0)
        self._show_example()
        self.notes.setText(self._notes_text(res))

    def _show_plot(self):
        res = self.result
        if res is None or self.cam_pick.count() == 0:
            self.plot.setText(EMPTY if res is None else 'No range fits (no trials).')
            return
        cam, cls = self.cam_pick.currentData(), self.cls_pick.currentData()
        mode = self.mode_pick.value() or res.settings.modes()[0]
        self.plot.setPixmap(_pix(FR.pdetect_png(res, cam, cls, mode, THEME.name == 'dark')))

    def _show_example(self):
        res = self.result
        i = self.example_pick.currentData()
        if res is None or i is None or not 0 <= i < len(res.examples):
            self.example.setText('No example frames.' if res is not None else '')
            return
        pm = _pix(res.examples[i]['jpg'])
        if self.example.width() > 50:
            pm = pm.scaledToWidth(min(self.example.width(), pm.width()), Qt.TransformationMode.SmoothTransformation)
        self.example.setPixmap(pm)

    def _notes_text(self, res) -> str:
        bits = []
        if res is not None:
            bits += res.notes
            bits.append('Models: ' + '; '.join(f'{m.get("name", m.get("id"))} ({m.get("licence", "")})'
                                                for m in res.models))
            bits.append(f'Ran on: {res.device.get("text", "")}. Median ms per image: ' + ', '.join(
                f'{FM.model_label(k)} {v["median"]:.0f}' for k, v in res.timing.get('ms_per_image', {}).items()))
            bits.append(f'{len(res.records)} perception records (vantage.perception_record/1) in the export.')
        bits += [FR.ASSUMPTIONS, FR.LIMITS_TEXT] + [f'{k}: {v}' for k, v in FR.METHOD.items()] + list(FR.REFERENCES)
        return '\n\n'.join(bits)
