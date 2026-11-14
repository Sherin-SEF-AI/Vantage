"""Simulate ▸ Log replay: a recorded drive's street scene rebuilt as 3D Gaussians (or a 3DGS .ply imported), rendered
through the recorded rig and a proposed rig variant with Vantage's own lens and lidar models, and detection compared
between the two on the re-rendered drive (core.splat_reproject).

Building the scene, the preview and the A/B run are background jobs with Cancel; the scene is cached in the working
folder (<working folder>/<project>/splat/) and reused while the drive and the build settings stay the same. Nothing runs
by itself after an edit: a result made from other settings is marked stale."""
from __future__ import annotations

import json
import os
import threading
import traceback

import numpy as np
from PyQt6.QtCore import QObject, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPainter
from PyQt6.QtWidgets import (QAbstractItemView, QComboBox, QFileDialog, QGridLayout, QHBoxLayout, QHeaderView,
                             QProgressBar, QScrollArea, QSlider, QSplitter, QTableWidget, QTableWidgetItem, QTabWidget,
                             QVBoxLayout, QWidget)

from ... import workdir
from ...core import detection as DT
from ...core import neural_replay as NR
from ...core import redrive as RD
from ...core import relight as RL
from ...core import splat_reproject as SP
from ..theme import STATUS, THEME, sans
from ..widgets import fit_width, ScrubSpinBox, SegmentedControl, StatusLine, ToggleSwitch, button, hline, label

JOB = 'splat'
EMPTY = ('Open a recorded drive (a nuScenes folder or a ROS 2 bag, e.g. a CARLA run), build its scene, choose the rig '
         'to compare with, then Run A/B.')
# (key, label, unit, decimals, step, tip)
BUILD_FIELDS = (
    ('max_gaussians', 'Gaussians, at most', '', 0, 10000.0, 'The scene\'s size: the voxels grow until it fits'),
    ('voxel_m', 'Voxel near the path', 'm', 3, 0.01, 'The points are thinned to one per voxel this size within 10 m '
                                                     'of the driven path, doubling with the distance'),
    ('max_range_m', 'Lidar range used', 'm', 0, 5.0, 'Points farther from the lidar are left out'),
    ('holdout_every', 'Hold out every', '', 0, 1.0, 'Every n-th frame and sweep is kept out of the build and '
                                                    'validates it (0: none)'),
    ('iterations', 'Optimisation steps', '', 0, 50.0, 'Photometric steps against the recorded images (0: colours '
                                                      'from the images only)'),
    ('time_budget_s', 'Time budget', 's', 0, 10.0, 'The optimisation stops after this long'),
    ('train_scale', 'Training size', '', 2, 0.05, 'Images at this share of their recorded size for the colours and '
                                                  'the optimisation'),
)
RENDER_FIELDS = (
    ('render_scale', 'Render size', '', 2, 0.05, 'Rendered images at this share of each module\'s sensor size'),
    ('eval_instants', 'Instants', '', 0, 1.0, 'Instants of the drive both rigs are rendered and compared at'),
    ('lidar_min_opacity', 'Lidar opacity', '', 2, 0.05, 'A beam returns when the Gaussians it crosses add up to at '
                                                        'least this opacity'),
    ('lidar_max_incidence_deg', 'Lidar incidence', '°', 1, 1.0, 'Beams meeting a surface more obliquely than this '
                                                               'return nothing'),
    ('val_frames', 'Validation frames', '', 0, 1.0, 'Held-out frames the reconstruction is scored on'),
    ('val_sweeps', 'Validation sweeps', '', 0, 1.0, 'Held-out sweeps per lidar the reconstruction is scored on'),
)
ALIGN_FIELDS = (('x', 'm'), ('y', 'm'), ('z', 'm'), ('yaw', '°'), ('pitch', '°'), ('roll', '°'))


def _qimage(rgb) -> QImage:
    a = np.ascontiguousarray(rgb[..., :3].astype(np.uint8))
    h, w = a.shape[:2]
    return QImage(a.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


class ImageView(QWidget):
    """An RGB image fitted into the widget with a caption."""

    def __init__(self, empty='', parent=None):
        super().__init__(parent)
        self.setMinimumSize(240, 160)
        self.img: QImage | None = None
        self.caption = ''
        self.empty = empty
        THEME.changed.connect(self._theme_changed)

    def _theme_changed(self, *_):
        self.update()

    def set_image(self, rgb, caption=''):
        self.img = _qimage(rgb) if rgb is not None else None
        self.caption = caption
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.fillRect(self.rect(), THEME.color('viewport'))
        p.setFont(sans(11))
        p.setPen(THEME.color('ink2'))
        p.drawText(QRectF(8, 2, self.width() - 16, 18), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                   self.caption)
        if self.img is None:
            p.setPen(THEME.color('muted'))
            p.drawText(QRectF(self.rect()), Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, self.empty)
            return
        top = 22.0
        W, H = self.width(), self.height() - top - 4
        s = min(W / max(self.img.width(), 1), H / max(self.img.height(), 1))
        w, h = self.img.width() * s, self.img.height() * s
        p.drawImage(QRectF((W - w) / 2, top + (H - h) / 2, w, h), self.img)


def _table(cols):
    t = QTableWidget(0, len(cols))
    t.setHorizontalHeaderLabels([c for c, _ in cols])
    for i, (_, tip) in enumerate(cols):
        if tip:
            t.horizontalHeaderItem(i).setToolTip(tip)
    t.verticalHeader().hide()
    t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    t.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
    t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    t.horizontalHeader().setStretchLastSection(True)
    return t


def _cell(t, r, c, text, num=False, color=None):
    it = QTableWidgetItem(str(text))
    if num:
        it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    if color is not None:
        it.setForeground(color)
    t.setItem(r, c, it)


def _pc(v):
    return '–' if v is None else f'{100 * v:.0f} %'


class _Sig(QObject):
    done = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)
    progress = pyqtSignal(int, float, str)
    nudge = pyqtSignal(int, object)


class SplatPage(QWidget):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.drive: SP.Drive | None = None
        self.scene: SP.Scene | None = None
        self.result: SP.SplatResult | None = None
        self.preview_out: dict | None = None
        self.detector = None                      # a callable image -> detections (tests); None: the chosen model
        self.busy = ''                            # '' | 'build' | 'preview' | 'run'
        self.status_text = ''
        self.cache_root = None                    # None: <working folder>/<project>/splat
        self._gen = 0
        self._for = None
        self._syncing = False
        self.sig = _Sig()
        self.sig.done.connect(self._thread_done)
        self.sig.failed.connect(self._failed)
        self.sig.progress.connect(self._progress)
        self.sig.nudge.connect(self._nudge_done)
        self.nr_result: NR.NeuralResult | None = None    # the neural replay study (Neural replay tab)
        self.rd_result: RD.RedriveResult | None = None   # the data-reuse report (Re-drive tab)
        self.rd_prepared: dict | None = None             # its scene, evidence and calibration (the nudge renders on it)
        self.rd_export: dict | None = None               # the last re-shot export
        self.nudge_out: dict | None = None
        self._nudge_lock = threading.Lock()
        self._nudge_req = None                           # the latest nudge render asked for (older ones are dropped)
        self._nudge_running = False
        self._nudge_gen = 0
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

    # ------------------------------------------------------------------ left
    def _field_grid(self, fields, store):
        g = QGridLayout()
        g.setHorizontalSpacing(6)
        g.setVerticalSpacing(3)
        for i, (key, text, unit, dec, step, tip) in enumerate(fields):
            f = ScrubSpinBox('', 'plain', dec, unit, step=step, width=110)
            f.setToolTip(tip)
            f.committed.connect(lambda v, k=key, t=text, u=unit: self._set(f'{t.lower()} {v:g}{u}', **{k: v}))
            a = label(text, ink2=True)
            a.setFixedWidth(140)
            a.setToolTip(tip)
            g.addWidget(a, i, 0)
            g.addWidget(f, i, 1)
            store[key] = f
        return g

    def _left(self):
        w = QWidget()
        w.setObjectName('panel')
        ll = QVBoxLayout(w)
        ll.setContentsMargins(8, 6, 8, 8)
        ll.setSpacing(4)
        top = QHBoxLayout()
        top.addWidget(label('Log replay'))
        top.addStretch(1)
        top.addWidget(label('drive · Gaussians · rig A/B', muted=True, small=True))
        ll.addLayout(top)
        ll.addWidget(hline())
        ll.addWidget(label('Re-render a recorded drive through a proposed rig: the street scene rebuilt as 3D '
                           'Gaussians from the drive\'s lidar and images, seen through each rig\'s own lenses, body '
                           'and lidars, and detection compared on the same moments.', muted=True, small=True,
                           wrap=True))
        ll.addWidget(label('Drive', ink2=True))
        self.drive_label = label('', small=True, wrap=True)
        ll.addWidget(self.drive_label)
        row = QHBoxLayout()
        b = button('Folder…', None, 'folder', 'Open a nuScenes folder or a CARLA run folder')
        b.clicked.connect(lambda: self._pick_drive(False))
        row.addWidget(b)
        b = button('Bag…', 'ghost', 'file', 'Open a ROS 2 bag (.mcap or .db3)')
        b.clicked.connect(lambda: self._pick_drive(True))
        row.addWidget(b)
        row.addStretch(1)
        ll.addLayout(row)
        ll.addWidget(hline())
        ll.addWidget(label('Scene', ink2=True))
        self.source = SegmentedControl([('lidar', 'From the drive'), ('ply', 'Import .ply')], 'lidar')
        self.source.setToolTip('Rebuild the scene from the drive\'s lidar and images, or import a 3D Gaussian '
                               'splatting .ply (gsplat, nerfstudio, INRIA) placed by the alignment below')
        self.source.changed.connect(lambda k: self._set(f'scene {k}', source=k))
        ll.addWidget(self.source, 0, Qt.AlignmentFlag.AlignLeft)
        self.ply_box = QWidget()
        pl = QVBoxLayout(self.ply_box)
        pl.setContentsMargins(0, 0, 0, 0)
        pl.setSpacing(3)
        prow = QHBoxLayout()
        self.ply_label = label('', small=True, wrap=True)
        prow.addWidget(self.ply_label, 1)
        b = button('.ply…', 'ghost', 'file', 'Choose the .ply file')
        b.clicked.connect(lambda: self._pick_ply())
        prow.addWidget(b)
        pl.addLayout(prow)
        ag = QGridLayout()
        ag.setHorizontalSpacing(4)
        self.align = {}
        for i, (k, unit) in enumerate(ALIGN_FIELDS):
            f = ScrubSpinBox(k[0].upper(), 'plain', 3 if unit == 'm' else 2, unit, width=80)
            f.setToolTip(f'The .ply\'s frame in the drive\'s global frame: {k}')
            f.committed.connect(lambda v, kk=k: self._align_edit(kk, v))
            ag.addWidget(f, i // 3, i % 3)
            self.align[k] = f
        f = ScrubSpinBox('S', 'plain', 4, '', width=80)
        f.setToolTip('The .ply\'s scale to metres')
        f.committed.connect(lambda v: self._set(f'ply scale {v:g}', ply_scale=v))
        ag.addWidget(f, 2, 0)
        self.align['scale'] = f
        pl.addLayout(ag)
        ll.addWidget(self.ply_box)
        self.fields = {}
        ll.addLayout(self._field_grid(BUILD_FIELDS, self.fields))
        g = QGridLayout()
        g.setHorizontalSpacing(6)
        a = label('Moving objects', ink2=True)
        a.setFixedWidth(140)
        g.addWidget(a, 0, 0)
        self.dynamic = SegmentedControl([('boxes', 'From boxes'), ('ignore', 'Static')], 'boxes')
        self.dynamic.setToolTip('Rebuild annotated actors in their own boxes (ground truth, sample_annotation or the '
                                'bag\'s boxes), or treat everything as static (moving objects smear)')
        self.dynamic.changed.connect(lambda k: self._set(f'moving objects {k}', dynamic=k))
        g.addWidget(self.dynamic, 0, 1)
        a = label('Device', ink2=True)
        g.addWidget(a, 1, 0)
        self.device = SegmentedControl([('auto', 'GPU if free'), ('cpu', 'CPU')], 'auto')
        self.device.setToolTip('The GPU when it has room (at most 3 GB is used), else the CPU')
        self.device.changed.connect(lambda k: self._set(f'device {k}', device=k))
        g.addWidget(self.device, 1, 1)
        self.optimise = ToggleSwitch('Optimise against the images')
        self.optimise.setToolTip('Photometric optimisation of colour, opacity, shape and position (time-budgeted)')
        self.optimise.toggled.connect(lambda on: self._set(f'optimise {"on" if on else "off"}', optimise=on))
        g.addWidget(self.optimise, 2, 0, 1, 2)
        ll.addLayout(g)
        bl = QHBoxLayout()
        bl.addStretch(1)
        self.b_build = button('Build scene', None, 'play', 'Build (or load from the cache) the scene: a background job')
        self.b_build.clicked.connect(lambda: self.build())
        bl.addWidget(self.b_build)
        ll.addLayout(bl)
        self.scene_label = label('', muted=True, small=True, wrap=True)
        ll.addWidget(self.scene_label)
        ll.addWidget(hline())
        ll.addWidget(label('Rigs', ink2=True))
        g = QGridLayout()
        g.setHorizontalSpacing(6)
        a = label('A', ink2=True)
        a.setFixedWidth(20)
        g.addWidget(a, 0, 0)
        self.rig_a = QComboBox()
        self.rig_a.setToolTip('Rig A: normally the recorded rig itself (its calibration), which validates the '
                              'reconstruction')
        self.rig_a.currentIndexChanged.connect(lambda i: self._rig_picked('a', i))
        g.addWidget(self.rig_a, 0, 1)
        g.addWidget(label('B', ink2=True), 1, 0)
        self.rig_b = QComboBox()
        self.rig_b.setToolTip('Rig B: the proposed variant')
        self.rig_b.currentIndexChanged.connect(lambda i: self._rig_picked('b', i))
        g.addWidget(self.rig_b, 1, 1)
        ll.addLayout(g)
        ll.addLayout(self._field_grid(RENDER_FIELDS, self.fields))
        g = QGridLayout()
        g.setHorizontalSpacing(6)
        a = label('Rolling shutter', ink2=True)
        a.setFixedWidth(140)
        g.addWidget(a, 0, 0)
        self.rolling = SegmentedControl([('module', 'As the module'), ('off', 'Off')], 'module')
        self.rolling.setToolTip('Render rows at their own read-out times (core.timing\'s read-out), or one instant')
        self.rolling.changed.connect(lambda k: self._set(f'rolling shutter {k}', rolling_shutter=k))
        g.addWidget(self.rolling, 0, 1)
        self.detect = ToggleSwitch('Detect (YOLOX, when downloaded)')
        self.detect.setToolTip('Run the detection-quality study\'s detector on the re-rendered images (Simulate ▸ '
                               'Detection quality downloads it)')
        self.detect.toggled.connect(lambda on: self._set(f'detection {"on" if on else "off"}', detect=on))
        g.addWidget(self.detect, 1, 0, 1, 2)
        ll.addLayout(g)
        self.model_status = StatusLine('note', '')
        ll.addWidget(self.model_status)
        ll.addWidget(label(SP.ASSUMPTIONS, muted=True, small=True, wrap=True))
        self.status = label('', muted=True, small=True, wrap=True)
        ll.addWidget(self.status)
        self.bar = QProgressBar()
        self.bar.setTextVisible(False)
        self.bar.setRange(0, 1000)
        self.bar.hide()
        ll.addWidget(self.bar)
        bl = QHBoxLayout()
        self.b_export = button('Export…', 'ghost', 'export', 'The study as JSON, the recall and validation as CSV, '
                                                            'previews as PNG and the scene as a 3DGS .ply')
        self.b_export.clicked.connect(lambda: self.export())
        bl.addWidget(self.b_export)
        bl.addStretch(1)
        self.b_cancel = button('Cancel', 'ghost', None, 'Stop the job')
        self.b_cancel.clicked.connect(lambda: self.cancel())
        bl.addWidget(self.b_cancel)
        self.b_run = button('Run A/B', 'primary', 'play', 'Validate the scene and compare the two rigs (a background '
                                                          'job; builds the scene first when needed)')
        self.b_run.clicked.connect(lambda: self.run())
        bl.addWidget(self.b_run)
        ll.addLayout(bl)
        ll.addStretch(1)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setWidget(w)
        fit_width(sc)
        sc.setMinimumWidth(360)
        return sc

    # ------------------------------------------------------------------ right
    def _right(self):
        w = QWidget()
        w.setObjectName('panel')
        rl = QVBoxLayout(w)
        rl.setContentsMargins(10, 8, 10, 8)
        self.tabs = QTabWidget()
        # preview
        pv = QWidget()
        pvl = QVBoxLayout(pv)
        pvl.setContentsMargins(0, 4, 0, 0)
        row = QHBoxLayout()
        row.addWidget(label('Time', ink2=True))
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider.setToolTip('The drive time to preview')
        self.slider.valueChanged.connect(lambda _: self._time_label())
        row.addWidget(self.slider, 1)
        self.t_label = label('', mono_=True, small=True)
        row.addWidget(self.t_label)
        self.cam_pick = QComboBox()
        self.cam_pick.setToolTip('The camera shown (by tag; a rig without it shows its first)')
        self.cam_pick.currentIndexChanged.connect(lambda _: self._show_preview())
        row.addWidget(self.cam_pick)
        self.b_preview = button('Preview', None, 'eye', 'Render both rigs at this time (a background job)')
        self.b_preview.clicked.connect(lambda: self.preview())
        row.addWidget(self.b_preview)
        pvl.addLayout(row)
        grid = QGridLayout()
        self.view_a = ImageView('Build the scene, then Preview.')
        self.view_b = ImageView('')
        self.lidar_a = ImageView('')
        self.lidar_b = ImageView('')
        grid.addWidget(self.view_a, 0, 0)
        grid.addWidget(self.view_b, 0, 1)
        grid.addWidget(self.lidar_a, 1, 0)
        grid.addWidget(self.lidar_b, 1, 1)
        pvl.addLayout(grid, 1)
        self.tabs.addTab(pv, 'Preview')
        self.val_table = _table((('Camera', ''), ('Frames', 'Held-out frames compared'),
                                 ('PSNR', 'Peak signal-to-noise ratio, dB, in the field and off the recording '
                                          'vehicle'), ('SSIM', 'Structural similarity')))
        vw = QWidget()
        vl = QVBoxLayout(vw)
        vl.setContentsMargins(0, 4, 0, 0)
        vl.addWidget(self.val_table, 1)
        self.val_lidar = label('', small=True, wrap=True)
        vl.addWidget(self.val_lidar)
        self.tabs.addTab(vw, 'Validation')
        self.ab_table = _table((('Class', ''), ('Range', 'From the rear axle'),
                                ('Objects', 'Objects either rig shows large and clear enough to score'),
                                ('Shown A', ''), ('Shown B', ''), ('Recall A', 'At the detection settings\' score'),
                                ('Recall B', ''), ('B − A', 'Points, with the 95 % interval of a bootstrap over '
                                                            'instants'),
                                ('Lidar pts A', 'Median lidar points on an object'), ('Lidar pts B', '')))
        self.tabs.addTab(self.ab_table, 'A/B')
        fw = QWidget()
        self.find_lay = QVBoxLayout(fw)
        self.find_lay.setContentsMargins(4, 4, 4, 4)
        self.find_lay.addStretch(1)
        fs = QScrollArea()
        fs.setWidgetResizable(True)
        fs.setWidget(fw)
        self.tabs.addTab(fs, 'Findings')
        self.tabs.addTab(self._neural_tab(), 'Neural replay')
        self.tabs.addTab(self._redrive_tab(), 'Re-drive')
        rl.addWidget(self.tabs, 1)
        return w

    # ------------------------------------------------------------------ re-drive (core.redrive)
    def _redrive_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 4, 0, 0)
        lay.addWidget(label('If we change the rig, do we lose our data? The proposed rig (rig B) re-shoots the drive; '
                            'every re-rendered pixel gets an expected error calibrated on views the reconstruction '
                            'never saw.', muted=True, small=True, wrap=True))
        row = QHBoxLayout()
        row.addWidget(label('Budget', ink2=True))
        self.rd_budget = ScrubSpinBox('', 'plain', 1, 'grey', step=0.5, width=96)
        self.rd_budget.setToolTip('A pixel is reusable when its mean absolute error is within this many grey levels '
                                  '(0–255)')
        self.rd_budget.committed.connect(lambda v: self._rd_set(f'error budget {v:g}', error_budget=v))
        row.addWidget(self.rd_budget)
        self.rd_paired_label = label('', small=True)
        b = button('Second recording…', 'ghost', 'folder', 'A recording of the same drive by another rig: it helps '
                                                        'calibrate the confidence, and is tested with a calibration '
                                                        'fitted without it')
        b.clicked.connect(lambda: self._rd_pick_paired())
        row.addWidget(b)
        row.addWidget(self.rd_paired_label, 1)
        self.b_rd_run = button('Calibrate & report', 'primary', 'play', 'Calibrate the confidence and report what '
                                                                       'of the drive the proposed rig can reuse (a '
                                                                       'background job)')
        self.b_rd_run.clicked.connect(lambda: self.redrive())
        row.addWidget(self.b_rd_run)
        self.b_rd_export = button('Export re-shot drive…', 'ghost', 'export', 'Write the drive re-rendered through '
                                                                              'the proposed rig as a nuScenes folder '
                                                                              'with confidence sidecars')
        self.b_rd_export.clicked.connect(lambda: self.reshoot())
        row.addWidget(self.b_rd_export)
        lay.addLayout(row)
        self.rd_table = _table((('Camera', ''), ('Within budget', 'Expected share of the camera\'s field within '
                                                                    'the budget'),
                                ('Expected error', 'Mean expected error, grey levels'), ('Verdict', ''),
                                ('Mostly', 'The main reason for the rest')))
        self.rd_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.rd_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.rd_table.itemSelectionChanged.connect(lambda: self._rd_show_overlay())
        self.rd_table.setMaximumHeight(190)
        lay.addWidget(self.rd_table)
        self.rd_text = label('', small=True, wrap=True)
        lay.addWidget(self.rd_text)
        g = QGridLayout()
        self.rd_polar = ImageView('Calibrate & report to see the regions.')
        self.rd_lost = ImageView('')
        self.rd_overlay = ImageView('')
        g.addWidget(self.rd_polar, 0, 0)
        g.addWidget(self.rd_lost, 0, 1)
        g.addWidget(self.rd_overlay, 0, 2)
        lay.addLayout(g, 1)
        # the nudge
        lay.addWidget(hline())
        nrow = QHBoxLayout()
        nrow.addWidget(label('Nudge', ink2=True))
        self.nudge_cam = QComboBox()
        self.nudge_cam.setToolTip('A camera of the proposed rig, re-rendered at the preview\'s time')
        self.nudge_cam.currentIndexChanged.connect(lambda _: self._nudge_reset_fields())
        nrow.addWidget(self.nudge_cam)
        self.nudge_f = {}
        for k, axis, kind, unit in (('dx', 'X', 'plain', 'm'), ('dy', 'Y', 'plain', 'm'), ('dz', 'Z', 'plain', 'm'),
                                    ('dyaw', 'Yaw', 'plain', '°'), ('dpitch', 'Pitch', 'plain', '°')):
            f = ScrubSpinBox(axis, kind, 3 if unit == 'm' else 1, unit, step=0.01 if unit == 'm' else 0.5, width=86)
            f.setToolTip(f'Move the camera: {"Δ" + k[1:]} ({"pitch positive down" if k == "dpitch" else unit})')
            f.scrubbed.connect(lambda v, kk=k: self._nudge_changed(kk, v, True))
            f.committed.connect(lambda v, kk=k: self._nudge_changed(kk, v, False))
            f.scrub_finished.connect(lambda: self.nudge(full=True))
            f.set_value(0.0)
            nrow.addWidget(f)
            self.nudge_f[k] = f
        self.nudge_lens = QComboBox()
        self.nudge_lens.setToolTip('Another lens for the camera')
        self.nudge_lens.currentIndexChanged.connect(lambda _: self.nudge(full=True))
        nrow.addWidget(self.nudge_lens)
        self.b_nudge_apply = button('Apply to variant', None, 'check', 'Apply the nudge to the proposed variant\'s '
                                                                      'camera (one undoable edit)')
        self.b_nudge_apply.clicked.connect(lambda: self.apply_nudge())
        nrow.addWidget(self.b_nudge_apply)
        b = button('Reset', 'ghost', None, 'Back to the camera as it is')
        b.clicked.connect(lambda: self._nudge_reset_fields(render=True))
        nrow.addWidget(b)
        nrow.addStretch(1)
        lay.addLayout(nrow)
        ng = QGridLayout()
        self.nudge_view = ImageView('Calibrate & report, then move a camera.')
        self.nudge_conf = ImageView('')
        ng.addWidget(self.nudge_view, 0, 0)
        ng.addWidget(self.nudge_conf, 0, 1)
        lay.addLayout(ng, 1)
        self.nudge_label = label('', small=True, wrap=True)
        lay.addWidget(self.nudge_label)
        return w

    # ------------------------------------------------------------------ neural replay (core.neural_replay)
    def _neural_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 4, 0, 0)
        lay.addWidget(label('Each Gaussian projected through the camera\'s exact model (the Unscented Transform, as '
                            '3DGUT), a lidar opacity of its own, pedestrians as articulated bodies posed from the '
                            'recorded images, traffic lights as emissive heads with their recorded state, and the '
                            're-shot frames relit to weather that was never driven. The settings apply to the log '
                            'replay, the A/B and the re-drive; Measure compares each addition with the log replay.',
                            muted=True, small=True, wrap=True))
        g = QGridLayout()
        g.setHorizontalSpacing(8)
        g.setVerticalSpacing(4)
        self.nr_projection = SegmentedControl([('ut', 'Unscented (3DGUT)'), ('ewa', 'Affine (EWA)')])
        self.nr_projection.setToolTip('How each Gaussian is projected: the Unscented Transform through the exact lens '
                                      '(rolling shutter exact per row), or the affine approximation at its centre')
        self.nr_projection.changed.connect(lambda k: self.nr_set(f'projection {k.upper()}', projection=k))
        g.addWidget(label('Projection', ink2=True), 0, 0)
        g.addWidget(self.nr_projection, 0, 1)
        self.nr_walkers = SegmentedControl([('articulated', 'Articulated'), ('rigid', 'Rigid')])
        self.nr_walkers.setToolTip('Pedestrians as articulated bodies (a skeleton posed per instant) or as the rigid '
                                   'point clouds of their boxes')
        self.nr_walkers.changed.connect(lambda k: self.nr_set(f'walkers {k}', walkers=k))
        g.addWidget(label('Walkers', ink2=True), 1, 0)
        g.addWidget(self.nr_walkers, 1, 1)
        self.nr_lights = ToggleSwitch('Traffic lights as emissive heads')
        self.nr_lights.setToolTip('Each traffic-light head of the ground truth (or a detector) drawn as a housing and '
                                  'three lamps showing the recorded state over time')
        self.nr_lights.toggled.connect(lambda on: self.nr_set('traffic lights ' + ('on' if on else 'off'),
                                                              traffic_lights=bool(on)))
        g.addWidget(self.nr_lights, 0, 2)
        self.nr_lidar = ToggleSwitch('Lidar opacity of its own')
        self.nr_lidar.setToolTip('Fit a lidar opacity per Gaussian to the training sweeps (glass: see-through for '
                                 'the camera, a return for the lidar)')
        self.nr_lidar.toggled.connect(lambda on: self.nr_set('lidar opacity ' + ('on' if on else 'off'),
                                                             lidar_opacity=bool(on)))
        g.addWidget(self.nr_lidar, 1, 2)
        self.nr_body = SegmentedControl([('capsules', 'Capsule body'), ('anny', 'Anny (CC0)'),
                                         ('smpl', 'SMPL (yours)')])
        self.nr_body.setToolTip('Vantage\'s own capsule body; Anny\'s MakeHuman base mesh (CC0, downloaded on '
                                'request); or an SMPL model file you downloaded yourself (non-commercial licence: '
                                'never bundled)')
        self.nr_body.changed.connect(lambda k: self.nr_set(f'body {k}', body_model=k))
        g.addWidget(label('Body', ink2=True), 2, 0)
        g.addWidget(self.nr_body, 2, 1)
        brow = QHBoxLayout()
        b = button('SMPL file…', 'ghost', 'folder', 'Your SMPL model (.npz) from smpl.is.tue.mpg.de')
        b.clicked.connect(lambda: self._nr_pick_smpl())
        brow.addWidget(b)
        self.nr_smpl_accept = ToggleSwitch('I accept SMPL\'s non-commercial licence')
        self.nr_smpl_accept.setToolTip(NR.SMPL_LICENCE)
        self.nr_smpl_accept.toggled.connect(lambda on: self.nr_set('SMPL licence ' + ('accepted' if on else
                                                                                      'not accepted'),
                                                                   smpl_accept=bool(on)))
        brow.addWidget(self.nr_smpl_accept)
        brow.addStretch(1)
        g.addLayout(brow, 2, 2)
        self.nr_relight = SegmentedControl([('none', 'As recorded'), ('wet', 'Wet'), ('night', 'Night'),
                                            ('snow', 'Snow')])
        self.nr_relight.setToolTip('Relight the re-rendered frames (previews, A/B, re-shot drive): relit, not '
                                   'recorded')
        self.nr_relight.changed.connect(lambda k: self.nr_set(f'relight {k}', relight=dict(
            NR.settings_of(self.ctx.store.project).relight, condition=k)))
        g.addWidget(label('Relight', ink2=True), 3, 0)
        g.addWidget(self.nr_relight, 3, 1)
        self.nr_method = SegmentedControl([('physical', 'Physical'), ('diffusion', 'Diffusion')])
        self.nr_method.setToolTip('The physically based relighting (always available), or InstructPix2Pix (MIT; '
                                  'downloaded on request; needs diffusers and a GPU)')
        self.nr_method.changed.connect(lambda k: self.nr_set(f'relight method {k}', relight=dict(
            NR.settings_of(self.ctx.store.project).relight, method=k)))
        g.addWidget(self.nr_method, 3, 2)
        lay.addLayout(g)
        mrow = QHBoxLayout()
        self.nr_models = StatusLine()
        mrow.addWidget(self.nr_models, 1)
        self.b_nr_pose = button('Download keypoint model', 'ghost', 'import',
                                'ViTPose-B, 344 MB, Apache-2.0, from its pinned revision (SHA-256 checked)')
        self.b_nr_pose.clicked.connect(lambda: self.nr_download('pose'))
        mrow.addWidget(self.b_nr_pose)
        self.b_nr_anny = button('Download Anny body', 'ghost', 'import',
                                'Anny\'s MakeHuman base mesh and skinning weights, 3.4 MB, CC0, from a pinned commit '
                                'of github.com/naver/anny (SHA-256 checked)')
        self.b_nr_anny.clicked.connect(lambda: self.nr_download('anny'))
        mrow.addWidget(self.b_nr_anny)
        self.b_nr_diff = button('Download relighting model', 'ghost', 'import',
                                'InstructPix2Pix (fp16), 2.1 GB, MIT, from its pinned revision (SHA-256 checked)')
        self.b_nr_diff.clicked.connect(lambda: self.nr_download('diffusion'))
        mrow.addWidget(self.b_nr_diff)
        lay.addLayout(mrow)
        row = QHBoxLayout()
        self.b_nr_run = button('Measure', 'primary', 'play', 'Rebuild with both projections, fit the walkers and '
                                                           'lights, and score them against the log replay and the '
                                                           'recorded images (a background job; minutes on a GPU)')
        self.b_nr_run.clicked.connect(lambda: self.nr_measure())
        row.addWidget(self.b_nr_run)
        self.b_nr_export = button('Export…', 'ghost', 'export', 'neural_replay.json, the CSVs and the previews')
        self.b_nr_export.clicked.connect(lambda: self.nr_export())
        row.addWidget(self.b_nr_export)
        row.addStretch(1)
        lay.addLayout(row)
        self.nr_table = _table((('Measure', ''), ('Log replay', 'The affine projection, rigid walkers, no lights, '
                                                               'one opacity'),
                                ('Neural', 'This tab\'s settings'), ('Recorded', 'The recorded images at the same '
                                                                                 'instants')))
        self.nr_table.setMaximumHeight(230)
        lay.addWidget(self.nr_table)
        g2 = QGridLayout()
        self.nr_walk_view = ImageView('Measure to see the walkers and lights.')
        self.nr_relit_view = ImageView('')
        g2.addWidget(self.nr_walk_view, 0, 0)
        g2.addWidget(self.nr_relit_view, 0, 1)
        lay.addLayout(g2, 1)
        return w

    def nr_settings(self) -> NR.NeuralSettings:
        return NR.settings_of(self.ctx.store.project)

    def nr_set(self, text, **kw) -> bool:
        """Change the neural replay settings (one undoable edit)."""
        if self._syncing:
            return False
        st = self.nr_settings()
        for k, v in kw.items():
            setattr(st, k, v)
        new = NR.NeuralSettings.from_dict(st.to_dict()).to_dict()
        if new == self.nr_settings().to_dict():
            return False

        def fn(p):
            p.analysis.neural_replay = new
        self.ctx.store.edit('analysis', fn, f'Neural replay · {text}')
        return True

    def _nr_pick_smpl(self):
        f, _ = QFileDialog.getOpenFileName(self, 'Your SMPL model (.npz)', '', 'SMPL model (*.npz)')
        if f:
            self.nr_set(f'SMPL file {os.path.basename(f)}', smpl_path=f, body_model='smpl')

    def nr_status_text(self) -> str:
        st = self.nr_settings()
        ps = NR.pose_status(st.pose_model, verify=False)
        ds = RL.diffusion_status(verify=False)
        parts = [ps['text']]
        if st.relight.get('method') == 'diffusion' or ds['present']:
            miss = RL.diffusion_missing()
            parts.append(ds['text'] + (f' {miss}' if miss else ''))
        if st.body_model == 'anny':
            parts.append(NR.anny_status(verify=False)['text'])
        if st.body_model == 'smpl':
            why = NR.smpl_usable(st)
            parts.append('SMPL body ready (your file).' if why is None else why)
        return ' · '.join(parts)

    def _nr_sync(self):
        st = self.nr_settings()
        self.nr_projection.set_value(st.projection)
        self.nr_walkers.set_value(st.walkers)
        self.nr_lights.setChecked(st.traffic_lights)
        self.nr_lidar.setChecked(st.lidar_opacity)
        self.nr_body.set_value(st.body_model)
        self.nr_smpl_accept.setChecked(st.smpl_accept)
        rl = RL.RelightSettings.from_dict(st.relight)
        self.nr_relight.set_value(rl.condition)
        self.nr_method.set_value(rl.method)
        ps = NR.pose_status(st.pose_model, verify=False)
        self.nr_models.set('good' if ps['verified'] else 'note', self.nr_status_text())
        self.b_nr_pose.setVisible(not ps['verified'])
        self.b_nr_anny.setVisible(st.body_model == 'anny' and not NR.anny_status(verify=False)['verified'])
        self.b_nr_diff.setVisible(rl.method == 'diffusion' and not RL.diffusion_status(verify=False)['verified'])

    def nr_download(self, what) -> bool:
        """Download the keypoint model ('pose') or the relighting model ('diffusion'): a background job."""
        if self.busy:
            return False
        st = self.nr_settings()

        def work(progress, cancel):
            if what == 'pose':
                return ('pose', NR.download_pose_model(st.pose_model, progress, cancel))
            if what == 'anny':
                return ('anny', NR.download_anny(progress, cancel))
            return ('diffusion', RL.download_diffusion(progress, cancel))
        self._begin('nr_download', 'Downloading…')
        ok = self._start('nr_download', 'Neural replay · download', work)
        if not ok:
            self.busy = ''
        self.refresh()
        return ok

    def nr_measure(self) -> bool:
        """The neural replay study on the chosen drive: a background job."""
        if self.busy:
            return False
        if not self.settings.drive:
            self.status_text = 'Choose a drive first.'
            self.refresh()
            return False
        p, eng, root, det = self.ctx.store.project, self._engine(), self._cache_root(), self.detector
        drive = self.drive

        def work(progress, cancel):
            return NR.run(p, eng, None, None, progress, cancel, det, root, drive)
        self._begin('neural', 'Measuring the neural replay…')
        ok = self._start('neural', 'Log replay · neural replay', work)
        if not ok:
            self.busy = ''
        self.refresh()
        return ok

    def nr_export(self, folder=None) -> list:
        res = self.nr_result
        if res is None:
            return []
        if folder is None:
            folder = QFileDialog.getExistingDirectory(self, 'Export the neural replay study into')
            if not folder:
                return []
        paths = NR.export(res, self.ctx.store.project, folder)
        self.ctx.msg(f'Neural replay · {len(paths)} files written to {folder}', 5000)
        return paths

    def _nr_done(self, kind, out):
        if kind == 'nr_download':
            what, path = out if isinstance(out, tuple) else ('', out)
            self.status_text = 'Download cancelled' if path is None else f'Downloaded: {os.path.basename(str(path))}'
            self._sync_fields()
            return
        self.nr_result = out
        s = NR.summary(out)
        self.status_text = (f'Neural replay measured in {out.seconds:.0f} s' +
                            (f': PSNR {s["psnr_ut_db"]:.2f} dB (UT) against {s["psnr_ewa_db"]:.2f} dB (EWA)'
                             if s.get('psnr_ut_db') is not None and s.get('psnr_ewa_db') is not None else ''))
        self.ctx.store.project.add_log('neural_replay', self.status_text)
        self._nr_show()

    def _nr_show(self):
        res = self.nr_result
        t = self.nr_table
        t.setRowCount(0)
        if res is None:
            self.nr_walk_view.set_image(None)
            self.nr_relit_view.set_image(None)
            return
        rows = []
        pj = res.projection or {}
        f = lambda v, d=2, u='': '–' if v is None else f'{v:.{d}f}{u}'  # noqa: E731
        if pj:
            rows.append(('Held-out PSNR', f(pj.get('ewa', {}).get('psnr_db'), 2, ' dB'),
                         f(pj.get('ut', {}).get('psnr_db'), 2, ' dB'), ''))
            rows.append(('Held-out SSIM', f(pj.get('ewa', {}).get('ssim'), 3), f(pj.get('ut', {}).get('ssim'), 3), ''))
            rows.append(('Lidar median error', f((pj.get('ewa', {}).get('lidar_median_abs_error_m') or 0) * 100, 1,
                                                  ' cm'),
                         f((pj.get('ut', {}).get('lidar_median_abs_error_m') or 0) * 100, 1, ' cm'), ''))
        L = res.lidar or {}
        if L:
            rows.append(('Lidar return agreement (opacity)', _pc(L['camera_opacity'].get('return_agreement')),
                         _pc(L['lidar_opacity'].get('return_agreement')), ''))
        pc = res.perception or {}
        for cls in ('pedestrian', 'traffic_light', 'car', 'vehicle'):
            r = (pc.get('rigid') or {}).get(cls)
            n = (pc.get('neural') or {}).get(cls)
            real = (pc.get('recorded') or {}).get(cls)
            if not (r or n or real):
                continue
            g = lambda x: '–' if not x else (f'{_pc(x["recall"])} of {x["objects"]}' if x.get('recall') is not None  # noqa: E731
                                             else f'{x["objects"]} shown')
            rows.append((f'{cls.replace("_", " ").capitalize()} rig recall', g(r), g(n), g(real)))
        for cond, d in (res.relight or {}).items():
            ch = d.get('change') or {}
            rows.append((f'Relit {cond}: recall change', '', ', '.join(f'{c} {100 * v:+.0f} pt' for c, v in ch.items()
                                                                      if c in ('pedestrian', 'car')) or '–', ''))
        t.setRowCount(len(rows))
        for i, r in enumerate(rows):
            for j, v in enumerate(r):
                _cell(t, i, j, v, num=j > 0)
        import cv2
        for view, fn in ((self.nr_walk_view, NR.walkers_png), (self.nr_relit_view, NR.relight_png)):
            png = fn(res)
            if png:
                img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
                view.set_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            else:
                view.set_image(None)

    @property
    def rd_settings(self) -> RD.RedriveSettings:
        return RD.settings_of(self.ctx.store.project)

    def _rd_set(self, text, **kw):
        if self._syncing:
            return False
        st = self.rd_settings
        for k, v in kw.items():
            setattr(st, k, v)
        new = RD.RedriveSettings.from_dict(st.to_dict()).to_dict()
        if new == self.rd_settings.to_dict():
            return False

        def fn(p):
            p.analysis.redrive = new
        self.ctx.store.edit('analysis', fn, f'Re-drive · {text}')
        return True

    def _rd_pick_paired(self):
        d = QFileDialog.getExistingDirectory(self, 'A second recording of the same drive')
        if d:
            self.set_paired(d)

    def set_paired(self, path) -> bool:
        """The second recording of the drive (undoable); it becomes a calibration source as well."""
        p = os.path.abspath(os.path.expanduser(str(path)))
        if not SP.is_drive(p):
            self.ctx.msg(f'{os.path.basename(p)} is neither a nuScenes folder nor a ROS 2 bag.', 6000)
            return False
        src = list(self.rd_settings.sources)
        if 'paired' not in src:
            src.append('paired')
        return self._rd_set(f'second recording {os.path.basename(p)}', paired_drive=p, sources=src) or True

    def _proposed(self):
        p = self.ctx.store.project
        return RD.proposed_variant(p, self.rd_settings, self.settings)

    def redrive(self) -> bool:
        """Calibrate the confidence and report the proposed rig's data reuse: a background job."""
        if self.busy:
            return False
        st, sst = self.rd_settings, self.settings
        if not sst.drive:
            self.status_text = 'Choose a drive first.'
            self.refresh()
            return False
        p, eng, root = self.ctx.store.project, self._engine(), self._cache_root()
        drive, scene, bk = self.drive, self.scene, self._build_key()
        v = self._proposed()

        def work(progress, cancel):
            res = RD.run(p, eng, st, v, progress, cancel, root, drive, scene)
            if res is not None:
                res._prepared['scene'].info.setdefault('build_key', bk)
            return res
        self._begin('redrive', 'Calibrating the confidence…')
        ok = self._start('redrive', 'Log replay · re-drive', work)
        if not ok:
            self.busy = ''
        self.refresh()
        return ok

    def reshoot(self, folder=None) -> bool:
        """Export the re-shot drive into folder (a background job; asks for one when None)."""
        if self.busy:
            return False
        if not self.settings.drive:
            self.status_text = 'Choose a drive first.'
            self.refresh()
            return False
        if folder is None:
            folder = QFileDialog.getExistingDirectory(self, 'Write the re-shot drive into')
            if not folder:
                return False
            folder = os.path.join(folder, f'reshot-{workdir.slug(self._proposed().name)}')
        p, eng, root, st = self.ctx.store.project, self._engine(), self._cache_root(), self.rd_settings
        prepared, v = self.rd_prepared, self._proposed()

        def work(progress, cancel):
            return RD.export_dataset(p, folder, st, v, eng, prepared, root, progress, cancel)
        self._begin('reshoot', 'Re-shooting the drive…')
        ok = self._start('reshoot', 'Log replay · re-shot drive', work)
        if not ok:
            self.busy = ''
        self.refresh()
        return ok

    def _rd_done(self, kind, out):
        if kind == 'redrive':
            self.rd_result = out
            self.rd_prepared = out._prepared
            self.drive, self.scene = self.rd_prepared['drive'], self.rd_prepared['scene']
            rg = out.report.get('rig') or {}
            self.status_text = rg.get('text', 'Re-drive done')
            self.ctx.store.project.add_log('redrive', f'Re-drive {out.variant}: {rg.get("text", "")}')
            self._nudge_fill()
            self.nudge(full=True)
        elif kind == 'reshoot':
            self.rd_export = out
            self.status_text = f'Re-shot drive written: {out.get("out_dir", "")}'
            self.ctx.msg(f'Log replay · {self.status_text}', 6000)

    def _rd_show(self):
        res = self.rd_result
        st = self.rd_settings
        self.rd_budget.set_value(st.error_budget)
        self.rd_paired_label.setText(os.path.basename(os.path.normpath(st.paired_drive)) if st.paired_drive else
                                     'none')
        t = self.rd_table
        sel = t.currentRow()
        t.setRowCount(0)
        if res is None:
            self.rd_text.setText('')
            for v in (self.rd_polar, self.rd_lost, self.rd_overlay):
                v.set_image(None)
            return
        for c in res.report.get('cameras') or []:
            r = t.rowCount()
            t.insertRow(r)
            _cell(t, r, 0, c['camera'])
            _cell(t, r, 1, _pc(c['within_share']), True)
            _cell(t, r, 2, f'{c["mean_expected_error"]:.1f}', True)
            _cell(t, r, 3, c['verdict'])
            _cell(t, r, 4, RD.REASON_TEXT.get(c['top_reason'], c['top_reason']))
        if 0 <= sel < t.rowCount():
            t.selectRow(sel)
        rg = res.report.get('rig') or {}
        inv = res.report.get('inverse') or {}
        rel = (res.calibration or {}).get('reliability_out_of_fold') or {}
        bits = [rg.get('text', '')]
        if inv.get('old_pixels'):
            bits.append(f'{100 * inv["lost_share"]:.0f} % of what the recorded rig saw is no longer seen.')
        if rel:
            bits.append(f'Calibration (out of fold): predicted {rel["mean_pred"]:.1f}, measured {rel["mean_meas"]:.1f} '
                        f'grey levels; rank correlation {rel.get("spearman") or 0:.2f}.')
        pv = res.paired or {}
        if pv.get('reliability'):
            pr = pv['reliability']
            bits.append(f'Second recording ({"in-sample: it is the only calibration source" if pv.get("in_calibration") else "out of sample"}): '
                        f'within budget {_pc(pr.get("within_pred"))} predicted, {_pc(pr.get("within_meas"))} measured.')
        self.rd_text.setText(' '.join(b for b in bits if b))
        import cv2
        for view, inverse, cap in ((self.rd_polar, False, 'Share within budget per region'),
                                   (self.rd_lost, True, 'Share of the recorded rig\'s view lost')):
            png = RD.polar_png(res, 360, inverse)
            img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR) if png else None
            view.set_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None, cap)
        self._rd_show_overlay()

    def _rd_show_overlay(self):
        res = self.rd_result
        ov = (res.report.get('_overlays') or {}) if res is not None else {}
        if not ov:
            self.rd_overlay.set_image(None)
            return
        r = self.rd_table.currentRow()
        tag = self.rd_table.item(r, 0).text() if 0 <= r < self.rd_table.rowCount() else next(iter(ov))
        o = ov.get(tag) or next(iter(ov.values()))
        self.rd_overlay.set_image(RD.overlay_rgb(o['rgb'], o['reason'], o['valid']),
                                  f'{tag} · t = {o["t"]:.2f} s · outside budget by reason')

    # ---- the nudge
    def _nudge_fill(self):
        v = self._proposed()
        cur = self.nudge_cam.currentData()
        self.nudge_cam.blockSignals(True)
        self.nudge_cam.clear()
        for c in (v.cameras if v is not None else []):
            if c.visible:
                self.nudge_cam.addItem(c.tag, c.uid)
        self.nudge_cam.setCurrentIndex(max(self.nudge_cam.findData(cur), 0))
        self.nudge_cam.blockSignals(False)
        self._nudge_reset_fields()

    def _nudge_camera(self):
        v = self._proposed()
        uid = self.nudge_cam.currentData()
        c = v.cam(uid) if (v is not None and uid) else None
        return v, c

    def _nudge_reset_fields(self, render=False):
        v, c = self._nudge_camera()
        for f in self.nudge_f.values():
            f.set_value(0.0)
        self.nudge_lens.blockSignals(True)
        self.nudge_lens.clear()
        if c is not None:
            p = self.ctx.store.project
            for lid, l in p.lenses.items():
                if not l.module_id or l.module_id == c.module_id:
                    self.nudge_lens.addItem(f'{l.name} · {l.label}' if l.label else l.name, lid)
            self.nudge_lens.setCurrentIndex(max(self.nudge_lens.findData(c.lens_id), 0))
        self.nudge_lens.blockSignals(False)
        if render:
            self.nudge(full=True)

    def nudge_values(self) -> dict:
        return {k: float(f.value()) for k, f in self.nudge_f.items()}

    def _nudge_changed(self, k, v, dragging):
        self.nudge_f[k].set_value(v) if not dragging else None
        self.nudge(full=not dragging, override={k: v})

    def nudged(self, override=None):
        """(variant, the camera as nudged) or (None, None)."""
        v, c = self._nudge_camera()
        if c is None:
            return None, None
        d = self.nudge_values()
        d.update(override or {})
        lens = self.nudge_lens.currentData()
        return v, RD.nudged_camera(c, d['dx'], d['dy'], d['dz'], d['dyaw'], d['dpitch'], lens)

    def nudge(self, full=True, override=None) -> bool:
        """Re-render the nudged camera with its confidence (reduced scale while dragging, full on release); the
        newest request replaces any that has not started."""
        if self.rd_prepared is None:
            return False
        v, c = self.nudged(override)
        if c is None:
            return False
        t = self.slider.value() / 1000.0 * self.rd_prepared['drive'].duration()
        req = (self.ctx.store.project, v, c, float(t), self.rd_settings, 1.0 if full else 0.5, self._nudge_gen)
        with self._nudge_lock:
            self._nudge_req = req
            if self._nudge_running:
                return True
            self._nudge_running = True
        threading.Thread(target=self._nudge_worker, daemon=True).start()
        return True

    def _nudge_worker(self):
        while True:
            with self._nudge_lock:
                req = self._nudge_req
                self._nudge_req = None
                if req is None:
                    self._nudge_running = False
                    return
            p, v, c, t, st, scale, gen = req
            try:
                out = RD.nudge_view(p, v, c, self.rd_prepared, t, st, scale, self._engine())
                out['camera'], out['t'], out['scale'] = c, t, scale
            except Exception as e:                   # noqa: BLE001
                out = {'error': f'{type(e).__name__}: {e}'}
            try:
                self.sig.nudge.emit(gen, out)
            except RuntimeError:
                return

    def _nudge_done(self, gen, out):
        if gen != self._nudge_gen:
            return
        self.nudge_out = out
        if 'error' in out:
            self.nudge_label.setText('Nudge failed: ' + out['error'])
            return
        c = out['camera']
        self.nudge_view.set_image(out['rgb'], f'{c.tag} · t = {out["t"]:.2f} s · re-rendered'
                                              + (' (reduced while dragging)' if out['scale'] < 1 else ''))
        self.nudge_conf.set_image(RD.error_colour(out['expected'], self.rd_settings.error_budget),
                                  'expected error: green 0, yellow the budget, red 3 × the budget')
        w = out.get('within_share')
        self.nudge_label.setText(f'{c.tag}: {_pc(w)} of the field within {self.rd_settings.error_budget:g} grey levels'
                                 f' · {out["seconds"]:.2f} s to render at {out["size"][0]} × {out["size"][1]}'
                                 + (f' · {out["note"]}' if out.get('note') else ''))

    def apply_nudge(self) -> bool:
        """Apply the nudge to the proposed variant's camera as one undoable edit."""
        v, c0 = self._nudge_camera()
        _, c = self.nudged()
        if c is None:
            return False
        if c0.locked:
            self.ctx.msg(f'{c0.tag} is locked: unlock it to move it.', 5000)
            return False
        d = self.nudge_values()
        if all(abs(x) < 1e-12 for x in d.values()) and c.lens_id == c0.lens_id:
            return False
        uid, vuid = c0.uid, v.uid

        def fn(p):
            vv = p.variant_by_uid(vuid)
            cam = vv.cam(uid)
            for k in ('x', 'y', 'z', 'yaw', 'pitch', 'lens_id', 'calib'):
                setattr(cam, k, getattr(c, k))
        self.ctx.store.edit(f'variant:{vuid}', fn, f'Re-drive nudge {c0.tag}: Δ {d["dx"]:+.3f}, {d["dy"]:+.3f}, '
                                                   f'{d["dz"]:+.3f} m, yaw {d["dyaw"]:+.1f}°, pitch '
                                                   f'{d["dpitch"]:+.1f}°')
        self._nudge_reset_fields(render=True)
        return True

    # ------------------------------------------------------------------ settings
    @property
    def settings(self) -> SP.SplatSettings:
        return SP.settings_of(self.ctx.store.project)

    def _store(self, st: SP.SplatSettings, text) -> bool:
        new = SP.SplatSettings.from_dict(st.to_dict()).to_dict()
        if new == self.settings.to_dict():
            self._sync_fields()
            return False

        def fn(p):
            p.analysis.splat = new
        self.ctx.store.edit('analysis', fn, f'Log replay · {text}')
        return True

    def _set(self, text, **kw):
        if self._syncing:
            return False
        st = self.settings
        for k, v in kw.items():
            setattr(st, k, v)
        return self._store(st, text)

    def _align_edit(self, k, v):
        st = self.settings
        if k in ('x', 'y', 'z'):
            xyz = list(st.ply_xyz)
            xyz['xyz'.index(k)] = v
            self._set(f'ply {k} {v:g} m', ply_xyz=xyz)
        else:
            ypr = list(st.ply_ypr)
            ypr[('yaw', 'pitch', 'roll').index(k)] = v
            self._set(f'ply {k} {v:g}°', ply_ypr=ypr)

    def _rig_picked(self, side, i):
        if self._syncing:
            return
        box = self.rig_a if side == 'a' else self.rig_b
        key = box.itemData(i)
        if key is not None:
            self._set(f'rig {side.upper()} {box.itemText(i)}', **{f'variant_{side}': key})

    def open_drive(self, path) -> bool:
        """Choose the drive (undoable); False when it is not one."""
        p = os.path.abspath(os.path.expanduser(str(path)))
        if not SP.is_drive(p):
            self.ctx.msg(f'{os.path.basename(p)} is neither a nuScenes folder nor a ROS 2 bag.', 6000)
            return False
        self.drive, self.scene, self.preview_out = None, None, None
        return self._set(f'drive {os.path.basename(p)}', drive=p) or True

    def _pick_drive(self, bag):
        st = self.settings
        start = os.path.dirname(st.drive) if st.drive else workdir.project_dir(self.ctx.store.project,
                                                                               workdir.get(self.ctx.settings))
        if bag:
            f, _ = QFileDialog.getOpenFileName(self, 'Open a ROS 2 bag', start, 'ROS 2 bags (*.mcap *.db3)')
            if f:
                self.open_drive(os.path.dirname(f) if os.path.isfile(os.path.join(os.path.dirname(f),
                                                                                     'metadata.yaml')) else f)
        else:
            d = QFileDialog.getExistingDirectory(self, 'Open a recorded drive', start)
            if d:
                self.open_drive(d)

    def _pick_ply(self):
        f, _ = QFileDialog.getOpenFileName(self, 'Import a 3D Gaussian splatting .ply', '', 'PLY (*.ply)')
        if f:
            self._set(f'ply {os.path.basename(f)}', ply_path=f, source='ply')

    def _cache_root(self) -> str:
        return self.cache_root or workdir.project_dir(self.ctx.store.project, workdir.get(self.ctx.settings), 'splat')

    # ------------------------------------------------------------------ state
    def _key(self, p=None):
        p = p or self.ctx.store.project
        return json.dumps(SP.settings_of(p).to_dict(), sort_keys=True)

    @property
    def stale(self) -> bool:
        return self.result is not None and self._for is not None and self._for != self._key()

    def _on_change(self, *_):
        if self.drive is not None and os.path.abspath(self.settings.drive or '.') != os.path.abspath(self.drive.path):
            self.drive, self.scene, self.preview_out = None, None, None
        if self.scene is not None and self.scene.info.get('build_key') not in (None, self._build_key()):
            self.scene = None
        if self.rd_prepared is not None and (self.drive is None or self.scene is None
                                             or self.rd_prepared['scene'] is not self.scene):
            self.rd_prepared = None                       # another drive or scene: the confidence is not its own
        self._sync_fields()
        self.refresh()

    def _build_key(self):
        nst = NR.settings_of(self.ctx.store.project)
        return json.dumps({'splat': self.settings.build_key(), 'neural': nst.build_key(),
                           'projection': nst.projection, 'relight': nst.relight}, sort_keys=True, default=str)

    def _on_replaced(self, *_):
        self._gen += 1
        self.drive = self.scene = self.result = self.preview_out = None
        self.rd_result = self.rd_prepared = self.rd_export = self.nudge_out = None
        self.nr_result = None
        self._nudge_gen += 1
        self._for, self.busy, self.status_text = None, '', ''
        self._sync_fields()
        self.refresh()

    def _retheme(self, *_):
        self._show()

    def showEvent(self, e):
        super().showEvent(e)
        self._sync_fields()
        self.refresh()

    def _sync_fields(self):
        st = self.settings
        p = self.ctx.store.project
        self._syncing = True
        try:
            for k, f in self.fields.items():
                f.set_value(getattr(st, k))
            self.source.set_value(st.source)
            self.dynamic.set_value(st.dynamic)
            self.device.set_value(st.device if st.device in ('auto', 'cpu') else 'auto')
            self.rolling.set_value(st.rolling_shutter)
            self.optimise.setChecked(st.optimise)
            self.detect.setChecked(st.detect)
            for k, i in (('x', 0), ('y', 1), ('z', 2)):
                self.align[k].set_value(st.ply_xyz[i])
            for k, i in (('yaw', 0), ('pitch', 1), ('roll', 2)):
                self.align[k].set_value(st.ply_ypr[i])
            self.align['scale'].set_value(st.ply_scale)
            self.ply_box.setVisible(st.source == 'ply')
            self.ply_label.setText(os.path.basename(st.ply_path) if st.ply_path else 'No .ply chosen.')
            self.drive_label.setText(os.path.basename(os.path.normpath(st.drive)) if st.drive else 'No drive chosen.')
            self.drive_label.setToolTip(st.drive)
            for box, side in ((self.rig_a, 'a'), (self.rig_b, 'b')):
                box.clear()
                if side == 'a':
                    box.addItem('Recorded rig (the drive\'s calibration)', SP.RECORDED)
                for v in p.variants:
                    box.addItem(v.name, v.uid)
                want = st.variant_a if side == 'a' else (st.variant_b or p.variant.uid)
                box.setCurrentIndex(max(box.findData(want), 0))
            self._nr_sync()
            ds = DT.status(self._det_model())
            self.model_status.set('good' if ds['verified'] else 'note',
                                  ds['text'] if st.detect else 'Detection off: the figures are the objects each rig '
                                                               'shows clearly enough to score.')
        finally:
            self._syncing = False

    def _det_model(self):
        from ...core import detection_eval as DE
        return DE.settings_of(self.ctx.store.project).model

    # ------------------------------------------------------------------ jobs
    def _start(self, kind, title, work):
        why = SP.torch_missing()
        if why:
            self.status_text = why
            self.ctx.msg(why, 10000)
            return False
        win = self.ctx.window
        gen = self._gen
        if win is not None and hasattr(win, 'run_job'):
            return win.run_job(JOB, title, work, lambda out: self._thread_done(gen, (kind, out)),
                               lambda m: self._failed(gen, m))

        def thread():
            try:
                out = work(lambda f, t='': self.sig.progress.emit(gen, f, t), lambda: gen != self._gen)
                self.sig.done.emit(gen, (kind, out))
            except Exception as e:              # noqa: BLE001
                try:
                    self.sig.failed.emit(gen, f'{type(e).__name__}: {e}\n{traceback.format_exc()}')
                except RuntimeError:
                    pass
        threading.Thread(target=thread, daemon=True).start()
        return True

    def _engine(self):
        eng = getattr(getattr(self.ctx, 'analysis', None), 'engine', None)
        return eng.share() if eng is not None else None

    def _begin(self, kind, text):
        self._gen += 1
        self.busy = kind
        self.status_text = text
        self.bar.setValue(0)

    def build(self) -> bool:
        """Open the drive and build (or load) its scene: a background job."""
        if self.busy:
            return False
        st = self.settings
        if not st.drive:
            self.status_text = 'Choose a drive first.'
            self.refresh()
            return False
        p, eng, root, bk = self.ctx.store.project, self._engine(), self._cache_root(), self._build_key()
        drive = self.drive

        def work(progress, cancel):
            d = drive or SP.open_drive(st.drive, root, progress, cancel)
            try:
                sc = SP.build_scene(d, st, p, eng, progress, cancel, root)
            except SP.Cancelled:
                return None
            sc.info['build_key'] = bk
            return d, sc
        self._begin('build', 'Building the scene…')
        ok = self._start('build', 'Log replay · scene', work)
        if not ok:
            self.busy = ''
        self.refresh()
        return ok

    def _rigs(self, eng):
        st = self.settings
        return SP.resolve_rigs(self.ctx.store.project, self.drive, st, eng)

    def preview(self, t=None) -> bool:
        if self.busy or self.drive is None or self.scene is None:
            if self.scene is None and not self.busy:
                self.status_text = 'Build the scene first.'
                self.refresh()
            return False
        st = self.settings
        if t is None:
            t = self.slider.value() / 1000.0 * self.drive.duration()
        drive, scene, eng = self.drive, self.scene, self._engine()

        def work(progress, cancel):
            ra, rb = SP.resolve_rigs(self.ctx.store.project, drive, st, eng)
            try:
                out = SP.preview(drive, scene, {'a': ra, 'b': rb}, float(t), st, cancel)
            except SP.Cancelled:
                return None
            out['t'] = float(t)
            out['names'] = (ra.name, rb.name)
            return out
        self._begin('preview', f'Rendering t = {t:.2f} s…')
        ok = self._start('preview', 'Log replay · preview', work)
        if not ok:
            self.busy = ''
        self.refresh()
        return ok

    def run(self, plan=None) -> bool:
        if self.busy:
            return False
        st = self.settings
        if not st.drive:
            self.status_text = 'Choose a drive first.'
            self.refresh()
            return False
        p = self.ctx.store.project
        key, eng, root, det = self._key(p), self._engine(), self._cache_root(), self.detector
        drive, scene, bk = self.drive, self.scene, self._build_key()

        def work(progress, cancel):
            d = drive or SP.open_drive(st.drive, root, lambda f, t='': progress(0.05 * f, t), cancel)
            sc = scene
            if sc is None:
                try:
                    sc = SP.build_scene(d, st, p, eng, lambda f, t='': progress(0.05 + 0.45 * f, t), cancel, root)
                except SP.Cancelled:
                    return None
                sc.info['build_key'] = bk
            res = SP.run(p, eng, st, None, lambda f, t='': progress(0.5 + 0.5 * f, t), cancel, det, root, d, sc)
            return (key, d, sc, res) if res is not None else None
        self._begin('run', 'Running…')
        ok = self._start('run', 'Log replay · A/B', work)
        if not ok:
            self.busy = ''
        self.refresh()
        return ok

    def cancel(self):
        win = self.ctx.window
        j = getattr(win, '_jobs', {}).get(JOB) if win is not None else None
        if j is not None:
            j['cancel'].set()
        if win is None or not hasattr(win, 'run_job'):
            if self.busy:
                self._gen += 1
                self.busy = ''
                self.status_text = 'Cancelled'
            self.refresh()

    def _progress(self, gen, f, text):
        if gen == self._gen and self.busy:
            self.bar.show()
            self.bar.setValue(int(f * 1000))
            if text:
                self.status.setText(text)

    def _thread_done(self, gen, payload):
        if gen != self._gen:
            return
        kind, out = payload
        self.busy = ''
        if out is None:
            self.status_text = 'Cancelled'
        elif kind == 'build':
            self.drive, self.scene = out
            info = self.scene.info
            self.status_text = (f'Scene: {self.scene.n:,} Gaussians' + (' (cached)' if info.get('cached') else
                                                                         f' in {info.get("seconds", 0):.0f} s'))
            self.ctx.msg(f'Log replay · {self.status_text}', 5000)
        elif kind == 'preview':
            self.preview_out = out
            self.status_text = f'Preview at t = {out["t"]:.2f} s'
        elif kind in ('redrive', 'reshoot'):
            self._rd_done(kind, out)
        elif kind in ('neural', 'nr_download'):
            self._nr_done(kind, out)
        elif kind == 'run':
            key, self.drive, self.scene, self.result = out
            self._for = key
            self.status_text = f'A/B done in {self.result.seconds:.0f} s'
            r = self.result
            self.ctx.store.project.add_log('splat', f'Log replay {os.path.basename(str(self.settings.drive))}: '
                                                    f'{r.rig_a.get("name")} vs {r.rig_b.get("name")}')
            self.ctx.msg(f'Log replay · {self.status_text}', 5000)
        self.refresh()

    def _failed(self, gen, msg):
        if gen == self._gen and self.busy:
            self.busy = ''
            self.status_text = 'Failed: ' + msg.splitlines()[0]
        self.refresh()

    def export(self, folder=None) -> list:
        res = self.result
        if res is None:
            return []
        if folder is None:
            folder = QFileDialog.getExistingDirectory(self, 'Export the log replay into')
            if not folder:
                return []
        from ...core import provenance as PV
        paths = PV.stamp_paths(SP.export(res, self.ctx.store.project, folder), self.ctx.store.project, folder)
        self.ctx.msg(f'Exported the log replay to {folder}', 5000)
        return paths

    def summary(self) -> dict | None:
        return SP.summary(self.result) if self.result is not None else None

    # ------------------------------------------------------------------ showing
    def refresh(self):
        st = self.settings
        busy = bool(self.busy)
        self.b_run.setEnabled(not busy and bool(st.drive))
        self.b_build.setEnabled(not busy and bool(st.drive))
        self.b_preview.setEnabled(not busy and self.scene is not None)
        self.b_cancel.setVisible(busy)
        self.b_export.setEnabled(self.result is not None and not busy)
        self.b_rd_run.setEnabled(not busy and bool(st.drive))
        self.b_rd_export.setEnabled(not busy and bool(st.drive))
        self.b_nudge_apply.setEnabled(self.rd_prepared is not None)
        self.b_nr_run.setEnabled(not busy and bool(st.drive))
        self.b_nr_export.setEnabled(self.nr_result is not None and not busy)
        self.bar.setVisible(busy)
        txt = self.status_text
        if self.stale and not busy:
            txt = (txt + ' · ' if txt else '') + 'The settings changed since the run: Run again.'
        if not st.drive and self.result is None and not txt:
            txt = EMPTY
        self.status.setText(txt)
        if self.scene is not None:
            i = self.scene.info
            self.scene_label.setText(f'{self.scene.n:,} Gaussians, {len(self.scene.actor_ids)} moving objects' +
                                     (f', held-out PSNR {i["optimisation"]["held_out_psnr_before_db"]:.1f} → '
                                      f'{i["optimisation"]["held_out_psnr_after_db"]:.1f} dB'
                                      if (i.get('optimisation') or {}).get('held_out_psnr_after_db') is not None
                                      else ''))
        else:
            self.scene_label.setText('No scene built yet.' if st.drive else '')
        self._show()

    def _time_label(self):
        d = self.drive.duration() if self.drive is not None else 0.0
        self.t_label.setText(f'{self.slider.value() / 1000.0 * d:5.2f} s')

    def _show(self):
        self._time_label()
        self._show_preview()
        self._rd_show()
        res = self.result
        t = self.val_table
        t.setRowCount(0)
        self.ab_table.setRowCount(0)
        while self.find_lay.count() > 1:
            it = self.find_lay.takeAt(0)
            if it.widget() is not None:
                it.widget().hide()
                it.widget().deleteLater()
        if res is None:
            self.val_lidar.setText('')
            return
        cams = (res.validation or {}).get('cameras') or {}
        for tag, v in (cams.get('cameras') or {}).items():
            r = t.rowCount()
            t.insertRow(r)
            _cell(t, r, 0, tag)
            _cell(t, r, 1, v['frames'], True)
            _cell(t, r, 2, f'{v["psnr"]:.1f} dB', True)
            _cell(t, r, 3, f'{v["ssim"]:.3f}', True)
        lv = (res.validation or {}).get('lidar') or {}
        self.val_lidar.setText(
            f'Lidar, {lv["sweeps"]} held-out sweeps: median range error {100 * lv["median_abs_error_m"]:.1f} cm, '
            f'{_pc(lv["within_10cm"])} within 10 cm, {_pc(lv["return_agreement"])} of the beams agree on returning.'
            if lv else (res.validation or {}).get('note', ''))
        a = self.ab_table
        for row in res.ab.get('recall') or []:
            r = a.rowCount()
            a.insertRow(r)
            _cell(a, r, 0, DT.CLASS_LABEL.get(row['class'], row['class'].title()))
            _cell(a, r, 1, row['range'])
            _cell(a, r, 2, row['objects'], True)
            _cell(a, r, 3, _pc(row['visible_share_a']), True)
            _cell(a, r, 4, _pc(row['visible_share_b']), True)
            _cell(a, r, 5, _pc(row.get('recall_a')), True)
            _cell(a, r, 6, _pc(row.get('recall_b')), True)
            d = row.get('delta')
            ci = row.get('delta_ci')
            _cell(a, r, 7, '–' if d is None else f'{100 * d:+.0f}' + (f' [{100 * ci[0]:+.0f}, {100 * ci[1]:+.0f}]'
                                                                       if ci else ''), True)
            _cell(a, r, 8, '–' if row.get('lidar_points_a') is None else f'{row["lidar_points_a"]:g}', True)
            _cell(a, r, 9, '–' if row.get('lidar_points_b') is None else f'{row["lidar_points_b"]:g}', True)
        for f in res.findings + [{'severity': 'note', 'message': n} for n in res.notes]:
            sl = StatusLine(f['severity'] if f['severity'] in STATUS else 'note', f['message'])
            self.find_lay.insertWidget(self.find_lay.count() - 1, sl)

    def _show_preview(self):
        out = self.preview_out
        tags = []
        if out is not None:
            for side in ('a', 'b'):
                tags += [t for t in out[side]['cameras'] if t not in tags]
        cur = self.cam_pick.currentText()
        if [self.cam_pick.itemText(i) for i in range(self.cam_pick.count())] != tags:
            self.cam_pick.blockSignals(True)
            self.cam_pick.clear()
            self.cam_pick.addItems(tags)
            if cur in tags:
                self.cam_pick.setCurrentText(cur)
            self.cam_pick.blockSignals(False)
        if out is None:
            for v in (self.view_a, self.view_b, self.lidar_a, self.lidar_b):
                v.set_image(None)
            return
        tag = self.cam_pick.currentText()
        for side, view, lv, name in (('a', self.view_a, self.lidar_a, out['names'][0]),
                                     ('b', self.view_b, self.lidar_b, out['names'][1])):
            cams = out[side]['cameras']
            k = tag if tag in cams else next(iter(cams), None)
            view.set_image(cams[k] if k else None, f'{side.upper()} · {name} · {k or "no camera"} · '
                                                   f't = {out["t"]:.2f} s')
            lv.set_image(SP.lidar_topview(out[side]['lidar']), f'{side.upper()} · lidar from above, ±40 m')
