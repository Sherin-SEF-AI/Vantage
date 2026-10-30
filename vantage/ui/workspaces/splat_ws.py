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
from ...core import splat_reproject as SP
from ..theme import STATUS, THEME, sans
from ..widgets import ScrubSpinBox, SegmentedControl, StatusLine, ToggleSwitch, button, hline, label

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
        sc.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        sc.setWidget(w)
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
        rl.addWidget(self.tabs, 1)
        return w

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
        self._sync_fields()
        self.refresh()

    def _build_key(self):
        return json.dumps(self.settings.build_key(), sort_keys=True, default=str)

    def _on_replaced(self, *_):
        self._gen += 1
        self.drive = self.scene = self.result = self.preview_out = None
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
