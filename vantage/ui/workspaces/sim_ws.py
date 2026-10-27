"""Simulate ▸ CARLA: run the planned rig in the CARLA simulator (CARLA renders its own town, traffic and weather; every
camera, lidar, radar, GNSS antenna and IMU of the variant records as planned) and get the drive back as a nuScenes
folder and a ROS 2 bag in the sample dataset's layout, then open it in the drive-log check and the as-built check.

Where CARLA is (its folder and its client Python) is kept per machine in the settings; the scenario and the rig's
simulation settings are saved with the project (undoable). The server starts headless in a capped systemd scope; a
run is a background job: Vantage writes the rig (core.simrig) and runs the bridge (vantage.sim.carla_bridge) in
CARLA's Python, reading its progress; Cancel stops it and it removes what it wrote."""
from __future__ import annotations

import datetime as _dt
import json
import os
import signal
import subprocess
import threading
import time
import traceback

from PyQt6.QtCore import QObject, Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (QAbstractItemView, QApplication, QComboBox, QFileDialog, QGridLayout, QHBoxLayout,
                             QHeaderView, QPlainTextEdit, QProgressBar, QScrollArea, QSplitter, QTableWidget,
                             QTableWidgetItem, QVBoxLayout, QWidget)

from ... import workdir
from ...core import simrig as SR
from ...core import weather as WX
from ...sim import server as SV
from ..theme import THEME, mono
from ..widgets import ROW_H, ScrubSpinBox, SegmentedControl, StatusLine, ToggleSwitch, button, hline, label
from .recording_ws import Tile

JOB = 'carla'
KEY_ROOT = 'sim/carla_root'
KEY_PYTHON = 'sim/python'
# (group, key, label, unit, decimals, step, scale, tip); scale: stored value × scale = shown value
FIELDS = (
    ('Scenario', 'duration_s', 'Duration', 's', 1, 1.0, 1.0, 'Seconds of drive recorded (simulation time)'),
    ('Scenario', 'traffic', 'Traffic', 'vehicles', 0, 1.0, 1.0, 'Other vehicles on the traffic manager\'s autopilot'),
    ('Scenario', 'seed', 'Seed', '', 0, 1.0, 1.0, 'Spawn points and the traffic manager: the same seed, the same drive'),
    ('Scenario', 'warmup_s', 'Warm-up', 's', 1, 0.5, 1.0, 'Traffic settles this long before the sensors attach'),
    ('Cameras', 'image_scale', 'Image size', '%', 0, 5.0, 100.0,
     'Images at this share of each sensor\'s size, as the sample dataset (the intrinsics are scaled to it)'),
    ('Cameras', 'face_oversample', 'View detail', '×', 2, 0.1, 1.0,
     'Resolution of the pinhole views CARLA renders against the lens\'s own at the image centre'),
    ('Cameras', 'jpeg_quality', 'JPEG quality', '', 0, 5.0, 1.0, 'JPEG quality of the written images'),
    ('Cameras', 'exposure_ev', 'Exposure', 'EV', 1, 0.25, 1.0, 'CARLA\'s exposure compensation'),
    ('Lidar and radar', 'lidar_noise_cm', 'Range noise', 'cm', 1, 0.5, 1.0, 'Lidar range noise along the ray (1 σ)'),
    ('Lidar and radar', 'max_lidar_channels', 'Max channels', '', 0, 16.0, 1.0,
     'The most even CARLA channels a lidar\'s beams may be laid over (more: exact elevations, more rays)'),
    ('Lidar and radar', 'radar_points', 'Radar rays', '/scan', 0, 100.0, 1.0,
     'Rays CARLA casts per radar zone and scan, inside the zone\'s field'),
    ('Simulation', 'tick_hz', 'Step rate', 'Hz', 0, 10.0, 1.0,
     'Simulation steps per second (0: chosen for exact lidar azimuths, then the most sensors on whole steps)'),
    ('Simulation', 'min_tick_hz', 'Least', 'Hz', 0, 10.0, 1.0, 'The chosen step rate is at least this'),
    ('Simulation', 'max_tick_hz', 'Most', 'Hz', 0, 10.0, 1.0, 'The chosen step rate is at most this'),
    ('Simulation', 'keyframe_hz', 'Keyframes', 'Hz', 1, 0.5, 1.0, 'nuScenes samples (annotated keyframes) per second'),
    ('Server', 'memory_gb', 'Memory cap', 'GB', 0, 1.0, 1.0,
     'The server runs in its own systemd scope capped at this much memory (no swap)'),
    ('Server', 'port', 'Port', '', 0, 1.0, 1.0, 'The server\'s RPC port (streaming on the next one)'),
    ('Server', 'tm_port', 'Traffic port', '', 0, 1.0, 1.0, 'The traffic manager\'s port'),
)
TOGGLES = {
    'Cameras': (('undistort', 'Rectified images', 'Pinhole images whose nuScenes camera_intrinsic is exact; off: the '
                                                 'real lens through its remap'),),
    'Lidar and radar': (('drop_self_hits', 'Drop hits on the own vehicle',
                         'Leave out lidar returns inside the CARLA vehicle\'s box'),),
    'Simulation': (('nuscenes', 'nuScenes', 'Write the nuScenes version folder v1.0-vantage'),
                   ('rosbag', 'ROS 2 bag', 'Write a rosbag2 bag (MCAP)')),
}
SENSOR_COLS = (('Sensor', ''), ('Kind', ''), ('In CARLA', 'What CARLA casts or renders for it'),
               ('Hz', 'The rate recorded (the plan\'s when it fits the simulation step)'), ('Note', ''))


class _Sig(QObject):
    plan = pyqtSignal(int, object)
    plan_failed = pyqtSignal(int, str)
    check = pyqtSignal(object)
    progress = pyqtSignal(float, str)
    done = pyqtSignal(object)
    failed = pyqtSignal(str)
    stopped = pyqtSignal()


def short_path(path, n=60) -> str:
    home = os.path.expanduser('~')
    if path == home or path.startswith(home + os.sep):
        path = '~' + path[len(home):]
    return path if len(path) <= n else path[:n // 2 - 1] + '…' + path[-(n // 2):]


def _item(text, tip='', num=False):
    it = QTableWidgetItem(text)
    if tip:
        it.setToolTip(tip)
    if num:
        it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        it.setFont(mono(12))
    return it


class SimPage(QWidget):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.setObjectName('base')
        self.sig = _Sig()
        self.sig.plan.connect(self._plan_done)
        self.sig.plan_failed.connect(self._plan_failed)
        self.sig.check.connect(self._check_done)
        self.sig.progress.connect(self._progress)
        self.sig.done.connect(self._done)
        self.sig.failed.connect(self._failed)
        self.sig.stopped.connect(self._refresh_server)
        self.server: SV.Server | None = None
        self.busy = False
        self.rig = None                   # the last planned rig (summary, no tables)
        self.result = None                # run.json of the last run
        self.out_dir = ''
        self.extra_pythonpath = ''        # tests: a fake carla module ahead of the real one
        self._cancel = threading.Event()
        self._proc = None
        self._pgen = 0
        self._dirty = True
        self._syncing = False
        self._check = None
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(400)
        self.timer.timeout.connect(self.plan)
        self.srv_timer = QTimer(self)
        self.srv_timer.setInterval(1000)
        self.srv_timer.timeout.connect(self._refresh_server)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setHandleWidth(2)
        split.addWidget(self._left())
        split.addWidget(self._right())
        split.setSizes([360, 1240])
        lay.addWidget(split)
        ctx.store.changed.connect(self._on_change)
        ctx.store.project_replaced.connect(self._on_replaced)
        THEME.changed.connect(self._on_theme)
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self.stop_server_now)
        self._load_paths()
        self.refresh()

    # ------------------------------------------------------------------ left: CARLA, server, settings
    def _left(self):
        w = QWidget()
        w.setObjectName('panel')
        ll = QVBoxLayout(w)
        ll.setContentsMargins(8, 6, 8, 8)
        ll.setSpacing(4)
        top = QHBoxLayout()
        top.addWidget(label('CARLA'))
        top.addStretch(1)
        top.addWidget(label(f'simulator {SR.CARLA_VERSION}', muted=True, small=True))
        ll.addLayout(top)
        ll.addWidget(hline())
        ll.addWidget(label('Run the planned rig in the CARLA simulator: its own town, traffic and weather, every camera, '
                           'lidar, radar, GNSS antenna and IMU as planned, recorded as a nuScenes folder and a ROS 2 '
                           'bag in the sample dataset\'s layout.', muted=True, small=True, wrap=True))
        ll.addSpacing(6)
        ll.addWidget(label('Installation', bold=True))
        self.root_label = label('', small=True, wrap=True)
        ll.addWidget(self.root_label)
        row = QHBoxLayout()
        b = button('CARLA folder…', 'ghost', 'folder', 'The folder holding CarlaUE4.sh (CARLA 0.9.16 for Linux)')
        b.clicked.connect(self._choose_root)
        row.addWidget(b)
        b = button('Python…', 'ghost', 'file', 'CARLA\'s client Python: a venv with the carla wheel, numpy, OpenCV and '
                                             'PyYAML')
        b.clicked.connect(self._choose_python)
        row.addWidget(b)
        self.b_check = button('Check', 'ghost', 'check', 'Check that CARLA\'s Python imports what the bridge needs')
        self.b_check.clicked.connect(self.check_client)
        row.addWidget(self.b_check)
        row.addStretch(1)
        ll.addLayout(row)
        self.py_status = StatusLine(None, '')
        ll.addWidget(self.py_status)
        ll.addSpacing(6)
        ll.addWidget(label('Server', bold=True))
        srow = QHBoxLayout()
        self.b_start = button('Start', 'primary', 'play', 'Start the CARLA server headless (-RenderOffScreen) in its own '
                                                         'systemd scope, capped at the memory below')
        self.b_start.clicked.connect(self.start_server)
        srow.addWidget(self.b_start)
        self.b_stop = button('Stop', 'ghost', 'stop', 'Stop the server this page started')
        self.b_stop.clicked.connect(self.stop_server)
        srow.addWidget(self.b_stop)
        self.quality = SegmentedControl([('Epic', 'Epic'), ('Low', 'Low')], 'Epic')
        self.quality.setToolTip('Rendering quality. Low: no post-processing or shadows, 50 m draw distance')
        self.quality.changed.connect(lambda k: self._set(f'server quality {k}', quality=k))
        srow.addWidget(self.quality)
        srow.addStretch(1)
        ll.addLayout(srow)
        self.no_render = ToggleSwitch('No rendering (no cameras)')
        self.no_render.setToolTip('Start the server with -nullrhi: no GPU at all. Lidar, radar, GNSS and IMU work; the '
                                  'cameras are left out of the run')
        ll.addWidget(self.no_render)
        self.srv_status = StatusLine(None, '')
        ll.addWidget(self.srv_status)
        self.fields, self.toggles = {}, {}
        group, grid = None, None

        def start(g):
            nonlocal grid
            if grid is not None:
                ll.addLayout(grid)
            ll.addSpacing(6)
            ll.addWidget(label(g, bold=True))
            if g == 'Scenario':
                self.town = QComboBox()
                for t in SR.TOWNS:
                    self.town.addItem(t, t)
                self.town.setToolTip('The CARLA map (Town06 and the others need CARLA\'s AdditionalMaps package)')
                self.town.currentIndexChanged.connect(
                    lambda i: self._set(f'town {self.town.itemData(i)}', town=self.town.itemData(i)))
                ll.addWidget(self.town)
                self.weather = QComboBox()
                self.weather.addItem('Weather as on the perception page', 'project')
                for c in WX.PRESETS:
                    self.weather.addItem(f'{c.name} (mapped)', f'vantage:{c.id}')
                for n in SR.CARLA_WEATHERS:
                    self.weather.addItem(f'CARLA {n}', n)
                self.weather.setToolTip('A Vantage condition mapped to CARLA\'s weather by documented heuristics, or one '
                                        'of CARLA\'s presets. Fog and rain show in the camera images only')
                self.weather.currentIndexChanged.connect(
                    lambda i: self._set('weather ' + self.weather.itemText(i).lower(), weather=self.weather.itemData(i)))
                ll.addWidget(self.weather)
                self.vehicle = QComboBox()
                self.vehicle.addItem('Ego vehicle: nearest by size', 'auto')
                for k, v in SR.CARLA_VEHICLES.items():
                    self.vehicle.addItem(f'{k.split(".", 1)[1]}  ({v[1]:.2f} × {v[2]:.2f} × {v[3]:.2f} m)', k)
                self.vehicle.setToolTip('The CARLA vehicle the rig is mounted on (its body, not the project\'s, is in '
                                        'the images). ' + SR.VEHICLE_NOTE)
                self.vehicle.currentIndexChanged.connect(
                    lambda i: self._set('ego ' + str(self.vehicle.itemData(i)), vehicle=self.vehicle.itemData(i)))
                ll.addWidget(self.vehicle)
            elif g == 'Cameras':
                self.rate = SegmentedControl([('lidar', 'Lidar rate'), ('module', 'Module fps')], 'lidar')
                self.rate.setToolTip('Lidar rate: one frame per lidar rotation at the sync plan\'s trigger time. '
                                     'Module fps: the module\'s rate')
                self.rate.changed.connect(lambda k: self._set('cameras at the ' + ('lidar rate' if k == 'lidar' else
                                                                                   'module fps'), camera_rate=k))
                ll.addWidget(self.rate)
                self.exposure = SegmentedControl([('manual', 'Manual exposure'), ('histogram', 'Auto per view')],
                                                 'manual')
                self.exposure.setToolTip('Manual: every view of a camera exposed alike (no seams). Auto: CARLA\'s '
                                         'histogram exposure per view, which shows as seams between views')
                self.exposure.changed.connect(lambda k: self._set(f'exposure {k}', exposure=k))
                ll.addWidget(self.exposure)
            elif g == 'Simulation':
                self.names = SegmentedControl([('tags', 'Rig tags'), ('nuscenes', 'nuScenes names')], 'tags')
                self.names.setToolTip('nuScenes channel names, as the sample dataset names them')
                self.names.changed.connect(lambda k: self._set('channels ' + k, channel_names=k))
                ll.addWidget(self.names)
            for key, text, tip in TOGGLES.get(g, ()):
                t = ToggleSwitch(text)
                t.setToolTip(tip)
                t.clicked.connect(lambda on, k=key, x=text: self._set(f'{x.lower()} {"on" if on else "off"}',
                                                                      **{k: on}))
                ll.addWidget(t)
                self.toggles[key] = t
            grid = QGridLayout()
            grid.setVerticalSpacing(2)
            grid.setHorizontalSpacing(6)
            grid.setColumnStretch(2, 1)
            return grid
        for g, key, text, unit, dec, step, scale, tip in FIELDS:
            if g != group:
                group = g
                grid = start(g)
            f = ScrubSpinBox('', 'plain', dec, unit, step=step, width=118)
            f.setToolTip(tip)
            f.committed.connect(lambda v, k=key, s=scale, t=text, u=unit: self._set(f'{t.lower()} {v:g} {u}'.strip(),
                                                                                  **{k: v / s}))
            a = label(text, ink2=True)
            a.setFixedWidth(104)
            a.setToolTip(tip)
            i = grid.rowCount()
            grid.addWidget(a, i, 0)
            grid.addWidget(f, i, 1)
            self.fields[key] = (f, scale)
        ll.addLayout(grid)
        ll.addSpacing(6)
        ll.addWidget(label('Limits', bold=True))
        ll.addWidget(label(SR.ASSUMPTIONS, muted=True, small=True, wrap=True))
        ll.addStretch(1)
        bl = QHBoxLayout()
        b = button('Defaults', 'ghost', None, 'Back to the default simulation settings')
        b.clicked.connect(self._defaults)
        bl.addWidget(b)
        bl.addStretch(1)
        ll.addLayout(bl)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        sc.setWidget(w)
        sc.setMinimumWidth(320)
        return sc

    # ------------------------------------------------------------------ right: the plan, the run, the result
    def _right(self):
        right = QSplitter(Qt.Orientation.Vertical)
        right.setHandleWidth(2)
        tw = QWidget()
        tw.setObjectName('panel')
        tl = QVBoxLayout(tw)
        tl.setContentsMargins(10, 8, 10, 8)
        tl.setSpacing(6)
        tiles = QHBoxLayout()
        tiles.setSpacing(8)
        self.tiles = {}
        for key, caption, tip in (
                ('step', 'Simulation step', 'Steps per second of simulation (each lidar step casts a slice of its turn)'),
                ('render', 'Rendering', 'Megapixels CARLA renders per frame of all cameras (their pinhole views)'),
                ('lidar', 'Lidar rays', 'Rays per second CARLA casts for the lidars (channels between beams included)'),
                ('vehicle', 'Ego vehicle', 'The CARLA vehicle the rig is mounted on, and how far its size is from the '
                                           'project\'s')):
            t = self.tiles[key] = Tile(caption, tip)
            tiles.addWidget(t, 1)
        tl.addLayout(tiles)
        tl.addWidget(label('The rig in CARLA', bold=True))
        self.table = QTableWidget(0, len(SENSOR_COLS))
        for c, (h, tip) in enumerate(SENSOR_COLS):
            it = QTableWidgetItem(h)
            if tip:
                it.setToolTip(tip)
            self.table.setHorizontalHeaderItem(c, it)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(ROW_H)
        self.table.horizontalHeader().setFixedHeight(ROW_H)
        self.table.setShowGrid(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        hh = self.table.horizontalHeader()
        for c in range(len(SENSOR_COLS) - 1):
            hh.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(len(SENSOR_COLS) - 1, QHeaderView.ResizeMode.Stretch)
        tl.addWidget(self.table, 1)
        self.p_status = label('', muted=True, small=True, wrap=True)
        tl.addWidget(self.p_status)
        right.addWidget(tw)
        gw = QWidget()
        gw.setObjectName('panel')
        gl = QVBoxLayout(gw)
        gl.setContentsMargins(10, 6, 10, 8)
        gl.setSpacing(4)
        row = QHBoxLayout()
        self.b_run = button('Run', 'primary', 'play', 'Write the rig and record a drive in CARLA into a new folder in the '
                                                     'working folder (Run into… picks another)')
        self.b_run.clicked.connect(lambda: self.run())
        row.addWidget(self.b_run)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(6)
        row.addWidget(self.bar, 1)
        self.b_cancel = button('Cancel', 'ghost', 'stop', 'Stop the run: CARLA is tidied up and what was written removed')
        self.b_cancel.clicked.connect(self.cancel)
        row.addWidget(self.b_cancel)
        b = button('Run into…', 'ghost', 'folder', 'Record into a folder of your choice')
        b.clicked.connect(self._run_into)
        row.addWidget(b)
        self.b_export = button('Export rig…', 'ghost', 'export', 'Write sim_rig.json and sim_remap.npz to run the '
                                                                'bridge yourself')
        self.b_export.clicked.connect(self._export_rig)
        row.addWidget(self.b_export)
        gl.addLayout(row)
        orow = QHBoxLayout()
        self.b_drivelog = button('Drive-log check', 'ghost', 'checklist', 'Open the recording in the drive-log check '
                                                                         'and run it')
        self.b_drivelog.clicked.connect(lambda: self.open_in('drivelog'))
        orow.addWidget(self.b_drivelog)
        self.b_asbuilt = button('As-built check', 'ghost', 'compare', 'Open the recording in the as-built check and run '
                                                                     'it (CARLA\'s car body is not the project\'s: its '
                                                                     'outline is checked against the plan\'s)')
        self.b_asbuilt.clicked.connect(lambda: self.open_in('asbuilt'))
        orow.addWidget(self.b_asbuilt)
        self.b_copy = button('Copy command', 'ghost', 'copy', 'Copy the bridge\'s command line for this rig')
        self.b_copy.clicked.connect(self._copy_command)
        orow.addWidget(self.b_copy)
        orow.addStretch(1)
        self.where = label('', muted=True, small=True)
        orow.addWidget(self.where)
        gl.addLayout(orow)
        self.g_status = label('', muted=True, small=True, wrap=True)
        gl.addWidget(self.g_status)
        self.out = QPlainTextEdit()
        self.out.setReadOnly(True)
        self.out.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.out.setFont(mono(11))
        self.out.setPlaceholderText('The run\'s report appears here: what was recorded, how fast, and every note.')
        gl.addWidget(self.out, 1)
        right.addWidget(gw)
        right.setSizes([520, 360])
        return right

    # ------------------------------------------------------------------ settings
    @property
    def settings(self) -> SR.SimSettings:
        return SR.settings_of(self.ctx.store.project)

    def _set(self, text, **kw):
        """Change settings as one undo step (saved with the project); False when nothing changes."""
        if self._syncing:
            return False
        st = self.settings
        for k, v in kw.items():
            setattr(st, k, v)
        new = SR.SimSettings.from_dict(st.to_dict()).to_dict()
        if new == self.settings.to_dict():
            self._sync_fields()
            return False

        def fn(p):
            p.analysis.sim = new
        self.ctx.store.edit('analysis', fn, f'CARLA · {text}')
        return True

    def _defaults(self):
        if self.ctx.store.project.analysis.sim:
            def fn(p):
                p.analysis.sim = {}
            self.ctx.store.edit('analysis', fn, 'CARLA · defaults')

    def _sync_fields(self):
        self._syncing = True
        try:
            st = self.settings.to_dict()
            for key, (f, scale) in self.fields.items():
                f.set_value(st[key] * scale)
            for combo, key in ((self.town, 'town'), (self.weather, 'weather'), (self.vehicle, 'vehicle')):
                i = combo.findData(st[key])
                if i < 0 and key == 'town':
                    combo.addItem(st[key], st[key])
                    i = combo.findData(st[key])
                combo.blockSignals(True)
                combo.setCurrentIndex(max(i, 0))
                combo.blockSignals(False)
            self.rate.set_value(st['camera_rate'])
            self.exposure.set_value(st['exposure'])
            self.names.set_value(st['channel_names'])
            self.quality.set_value(st['quality'])
            for key, t in self.toggles.items():
                t.blockSignals(True)
                t.setChecked(bool(st[key]))
                t.blockSignals(False)
        finally:
            self._syncing = False

    # ------------------------------------------------------------------ the installation
    def _load_paths(self):
        s = self.ctx.settings
        root = str(s.value(KEY_ROOT, '') or '')
        if not SV.is_carla_root(root):
            found = SV.candidate_roots()
            root = found[0] if found else ''
        py = str(s.value(KEY_PYTHON, '') or '') or SV.client_python(root)
        self.carla_root, self.python = root, py
        self._show_paths()

    def _show_paths(self):
        if self.carla_root:
            v = SV.version(self.carla_root)
            self.root_label.setText(f'CARLA {v or "?"} in {short_path(self.carla_root)}\n'
                                    f'Python: {short_path(self.python) if self.python else "none chosen"}')
        else:
            self.root_label.setText('No CARLA folder found: choose the folder holding CarlaUE4.sh (CARLA 0.9.16 for '
                                    'Linux, carla.org). Its client Python is a venv with the carla wheel, numpy, '
                                    'OpenCV and PyYAML.')
        self.root_label.setToolTip(f'{self.carla_root}\n{self.python}')

    def _choose_root(self):
        d = QFileDialog.getExistingDirectory(self, 'CARLA folder (holding CarlaUE4.sh)', self.carla_root or
                                             os.path.expanduser('~'))
        if not d:
            return
        if not SV.is_carla_root(d):
            self.ctx.msg(f'{short_path(d)} is not a CARLA folder: {SV.BINARY} is missing.', 6000)
            return
        self.carla_root = d
        self.ctx.settings.setValue(KEY_ROOT, d)
        if not self.python:
            self.python = SV.client_python(d)
            self.ctx.settings.setValue(KEY_PYTHON, self.python)
        self._show_paths()
        self._update_buttons()

    def _choose_python(self):
        f, _ = QFileDialog.getOpenFileName(self, 'CARLA\'s client Python', os.path.dirname(self.python or
                                                                                         self.carla_root or ''))
        if f:
            self.python = f
            self.ctx.settings.setValue(KEY_PYTHON, f)
            self._show_paths()
            self.check_client()

    def check_client(self):
        py = self.python
        self.py_status.set(None, 'Checking CARLA\'s Python…')

        def work():
            try:
                res = SV.check_client(py)
            except Exception as e:                    # noqa: BLE001
                res = {'ok': False, 'text': f'{type(e).__name__}: {e}'}
            try:
                self.sig.check.emit(res)
            except RuntimeError:
                pass
        threading.Thread(target=work, daemon=True).start()

    def _check_done(self, res):
        self._check = res
        self.py_status.set('good' if res.get('ok') else 'serious', res.get('text', ''))

    # ------------------------------------------------------------------ the server
    def start_server(self) -> bool:
        if self.server is not None and self.server.running:
            return False
        if not SV.is_carla_root(self.carla_root):
            self.ctx.msg('Choose the CARLA folder first.', 5000)
            return False
        st = self.settings
        log = os.path.join(workdir.project_dir(self.ctx.store.project, self.working(), 'carla'), 'server.log')
        os.makedirs(os.path.dirname(log), exist_ok=True)
        try:
            self.server = SV.Server(self.carla_root, st.port, st.memory_gb, st.quality, self.no_render.isChecked(), log)
            self.server.start()
        except Exception as e:                        # noqa: BLE001 (shown to the user)
            self.server = None
            self.srv_status.set('serious', str(e))
            return False
        self.srv_timer.start()
        self._refresh_server()
        self.ctx.msg('CARLA server starting (headless, capped)…', 4000)
        return True

    def stop_server(self):
        srv = self.server
        if srv is None:
            return
        self.srv_status.set(None, 'Stopping the server…')

        def work():
            srv.stop()
            try:
                self.sig.stopped.emit()
            except RuntimeError:
                pass
        threading.Thread(target=work, daemon=True).start()

    def stop_server_now(self):
        """On quit: the server this page started goes with the app."""
        if self.server is not None:
            self.server.stop(10.0)

    def _refresh_server(self):
        srv, st = self.server, self.settings
        if srv is not None and srv.running:
            up = SV.port_open('127.0.0.1', srv.port)
            mode = 'no rendering' if srv.no_render else f'off-screen, {srv.quality}'
            self.srv_status.set('good' if up else 'warning',
                                f'Running ({mode}, capped at {srv.memory_gb:g} GB), '
                                + ('answering on port %d' % srv.port if up else
                                   f'starting… {time.time() - srv.started:.0f} s'))
        elif SV.port_open('127.0.0.1', st.port):
            self.srv_status.set('good', f'A CARLA server answers on port {st.port} (not started here).')
            self.srv_timer.stop()
        else:
            if srv is not None and srv.proc is not None and srv.proc.returncode not in (None, 0, -15, 143):
                self.srv_status.set('serious', f'The server stopped (exit {srv.proc.returncode}); see '
                                               f'{short_path(srv.log_path)}.')
            else:
                self.srv_status.set(None, f'No server on port {st.port}.')
            if srv is None or not srv.running:
                self.srv_timer.stop()
        self._update_buttons()

    # ------------------------------------------------------------------ the plan (background)
    def plan(self):
        p = self.ctx.store.project
        self._pgen += 1
        gen = self._pgen
        snap = p.snapshot()
        st = self.settings

        def work():
            try:
                rig = SR.build(snap, st, tables=False)
                self.sig.plan.emit(gen, rig)
            except RuntimeError:
                pass
            except Exception as e:                    # noqa: BLE001
                try:
                    self.sig.plan_failed.emit(gen, f'{type(e).__name__}: {e}')
                except RuntimeError:
                    pass
        threading.Thread(target=work, daemon=True).start()

    def _plan_done(self, gen, rig):
        if gen != self._pgen:
            return
        self.rig = rig
        self._fill_plan(rig)

    def _plan_failed(self, gen, msg):
        if gen != self._pgen:
            return
        self.rig = None
        self.table.setRowCount(0)
        for t in self.tiles.values():
            t.set('—', '', '')
        self.p_status.setText(msg)
        self._update_buttons()

    def _fill_plan(self, rig):
        d = rig.doc
        sim = d['simulation']
        cams, lids = d['cameras'], d['lidars']
        mpx = sum(c['render_mpx'] for c in cams)
        rays = sum(L['rays_per_s'] for L in lids)
        self.tiles['step'].set(f'{sim["tick_hz"]:g}', 'Hz', f'{sim["dt_s"] * 1000:.2f} ms · warm-up {sim["warmup_ticks"]} steps')
        self.tiles['render'].set(f'{mpx:.1f}', 'Mpx', f'{sum(len(c["views"]) for c in cams)} views of {len(cams)} '
                                                     f'camera{"s" * (len(cams) != 1)}' if cams else 'no camera')
        self.tiles['lidar'].set(f'{rays / 1e6:.2f}', 'M/s', ', '.join(f'{L["tag"]} {L["blueprint"]["channels"]} ch'
                                                                     for L in lids) or 'no lidar')
        v = d['vehicle']
        err = v.get('size_error')
        self.tiles['vehicle'].set(v['blueprint'].split('.', 1)[1][:16], '',
                                  f'size within {100 * err:.0f} % (mean)' if err is not None else v['how'],
                                  None if err is None or err < 0.08 else 'warning')
        rows = []
        for c in cams:
            cap = c['capture']
            rows.append((c['tag'], 'camera', f'{len(c["views"])} view{"s" * (len(c["views"]) != 1)} '
                                             f'{"/".join(x["name"] for x in c["views"])} · '
                                             f'{c["views"][0]["width"]} px, {c["views"][0]["fov_deg"]:.1f}° → '
                                             f'{c["image"]["width"]} × {c["image"]["height"]}',
                         cap['recorded_hz'], f'exposure centre at {1000 * cap["recorded_centre_s"]:.1f} ms after the '
                                             f'sync pulse (plan {1000 * cap["exposure_centre_s"]:.1f} ms, '
                                             f'{cap["offset_error_ms"]:+.1f} ms: whole steps)'))
        for L in lids:
            bpd = L['blueprint']
            rows.append((L['tag'], 'lidar', f'{bpd["channels"]} channels {bpd["lower_fov"]:g}° … {bpd["upper_fov"]:g}° '
                                            f'· {L["azimuth_step_deg"]:.4f}° × {L["oversample"]}',
                         L['rate_hz'], L['plan_note']))
        for r in d['radars']:
            cones = ' + '.join(f'{z["blueprint"]["horizontal_fov"]:.0f}° × {z["blueprint"]["vertical_fov"]:.0f}° to '
                               f'{z["range"]:g} m' for z in r['zones'])
            kept = '/'.join(f'±{z["az_half_deg"]:g}° × ±{z["el_half_deg"]:g}°' for z in r['zones'])
            rows.append((r['tag'], 'radar', cones, r['recorded_hz'], f'kept: {kept}'))
        for x in d['imus']:
            rows.append((x['tag'], 'IMU', 'sensor.other.imu (rate from the vehicle\'s poses)', x['recorded_hz'], ''))
        for x in d['gnss']:
            rows.append((x['tag'], 'GNSS', 'sensor.other.gnss', x['recorded_hz'],
                         f'noise {100 * x["sigma_h_m"]:.0f} cm'))
        self.table.setRowCount(len(rows))
        for i, (tag, kind, what, hz, note) in enumerate(rows):
            self.table.setItem(i, 0, _item(tag))
            self.table.setItem(i, 1, _item(kind))
            self.table.setItem(i, 2, _item(what, what))
            self.table.setItem(i, 3, _item(f'{hz:.4g}', num=True))
            self.table.setItem(i, 4, _item(note, note))
        self.p_status.setText(' '.join(rig.notes))
        self._update_buttons()

    # ------------------------------------------------------------------ state
    def _on_change(self, *_):
        if self.isVisible():
            self.refresh()
            self.timer.start()
        else:
            self._dirty = True

    def _on_replaced(self, *_):
        self._pgen += 1
        self.rig = None
        self.result = None
        self.out.setPlainText('')
        self.g_status.setText('')
        self._on_change()

    def _on_theme(self, *_):
        self.table.viewport().update()

    def showEvent(self, e):
        super().showEvent(e)
        self.activate()

    def activate(self):
        self._show_where()
        if self._dirty or self.rig is None:
            self.refresh()
            self.timer.start()
        if self._check is None and self.python:
            self.check_client()
        self._refresh_server()

    def refresh(self):
        self._dirty = False
        self._sync_fields()
        self._update_buttons()

    def _update_buttons(self):
        ok = self.rig is not None
        srv_up = self.server is not None and self.server.running
        self.b_start.setEnabled(SV.is_carla_root(self.carla_root) and not srv_up)
        self.b_stop.setEnabled(srv_up)
        self.b_run.setEnabled(ok and not self.busy and bool(self.python))
        self.b_cancel.setEnabled(self.busy)
        self.b_export.setEnabled(ok)
        self.b_copy.setEnabled(ok and bool(self.python))
        rec = self._recording()
        self.b_drivelog.setEnabled(bool(rec) and not self.busy)
        self.b_asbuilt.setEnabled(bool(rec) and not self.busy)

    def working(self) -> str:
        return workdir.get(self.ctx.settings)

    def _show_where(self):
        folder = workdir.project_dir(self.ctx.store.project, self.working(), 'carla')
        self.where.setText(f'Runs go to {short_path(folder)}')
        self.where.setToolTip(self.working())

    def _new_dir(self) -> str:
        p = self.ctx.store.project
        root = workdir.project_dir(p, self.working(), 'carla')
        stamp = _dt.datetime.now().strftime('%Y%m%d-%H%M%S')
        out = os.path.join(root, f'{workdir.slug(p.variant.name)}-{stamp}')
        k = 2
        while os.path.exists(out):
            out = os.path.join(root, f'{workdir.slug(p.variant.name)}-{stamp}-{k}')
            k += 1
        return out

    def _recording(self) -> str:
        """The last run's bag folder (else its nuScenes folder), or ''."""
        r = self.result
        if not r or r.get('status') != 'ok' or not self.out_dir:
            return ''
        bag = os.path.join(self.out_dir, 'rosbag')
        if os.path.isdir(bag):
            subs = [os.path.join(bag, x) for x in sorted(os.listdir(bag))]
            subs = [x for x in subs if os.path.isdir(x)]
            if subs:
                return subs[0]
        nus = os.path.join(self.out_dir, 'nuscenes')
        return nus if os.path.isdir(nus) else ''

    # ------------------------------------------------------------------ export, command
    def _export_rig(self):
        folder = QFileDialog.getExistingDirectory(self, 'Write sim_rig.json and sim_remap.npz into', self.working())
        if not folder:
            return
        p = self.ctx.store.project
        try:
            from ...core import provenance as PV      # where the JSON came from (the bridge ignores the block)
            rig = SR.build(p.snapshot(), self.settings)
            paths = PV.stamp_paths(SR.export(rig, p, folder), p, folder)
        except Exception as e:                        # noqa: BLE001 (shown)
            self.ctx.msg(f'Export failed: {e}', 6000)
            return
        self.ctx.msg(f'Wrote {", ".join(os.path.basename(x) for x in paths)} to {short_path(folder)}', 5000)

    def command_line(self, rig_path='sim_rig.json', out_dir='OUT') -> str:
        st = self.settings
        args = [self.python or 'python', '-m', 'vantage.sim.carla_bridge', rig_path, out_dir, '--town', st.town,
                '--duration', f'{st.duration_s:g}', '--traffic', str(st.traffic), '--seed', str(st.seed),
                '--port', str(st.port), '--tm-port', str(st.tm_port)]
        if st.weather in SR.CARLA_WEATHERS:
            args += ['--weather', st.weather]
        src = SV.vantage_source()
        return (f'PYTHONPATH={src} ' if src else '') + ' '.join(args)

    def _copy_command(self):
        QApplication.clipboard().setText(self.command_line())
        self.ctx.msg('Copied the bridge\'s command line', 3000)

    # ------------------------------------------------------------------ the run
    def _run_into(self):
        folder = QFileDialog.getExistingDirectory(self, 'Record the CARLA drive into', self.working())
        if folder:
            self.run(folder)

    def run(self, folder=None) -> bool:
        """Write the rig into <folder>/rig and run the bridge into folder as a background job."""
        if self.busy:
            return False
        if not self.python:
            self.ctx.msg('Choose CARLA\'s client Python first.', 5000)
            return False
        st = self.settings
        if not SV.port_open('127.0.0.1', st.port):
            self.g_status.setText(f'No CARLA server answers on port {st.port}: start one (Server ▸ Start).')
            self.ctx.msg(self.g_status.text(), 5000)
            return False
        folder = folder or self._new_dir()
        no_cams = self.server is not None and self.server.running and self.server.no_render
        p = self.ctx.store.project
        snap = p.snapshot()
        python, extra = self.python, self.extra_pythonpath
        self._for_project = p
        self._cancel.clear()
        self.busy = True
        self.result = None
        self.out_dir = folder
        self.bar.setValue(0)
        self.out.setPlainText('')
        self.g_status.setText(f'Recording into {short_path(folder)}…')
        self._update_buttons()
        cancel_me = self._cancel

        def work(progress, cancel):
            def prog(f, text=''):
                progress(f, text)
                try:
                    self.sig.progress.emit(f, text)
                except RuntimeError:
                    pass
            return self._work(snap, st, folder, python, extra, no_cams, prog,
                              lambda: (cancel is not None and cancel()) or cancel_me.is_set())
        win = self.ctx.window
        if win is not None and hasattr(win, 'run_job'):
            if win.run_job(JOB, 'CARLA drive', work, self._done, self._failed):
                return True
            self.busy = False
            self.g_status.setText('A CARLA run is already going.')
            self._update_buttons()
            return False

        def thread():
            try:
                res = work(lambda *a: None, None)
                self.sig.done.emit(res)
            except RuntimeError:
                pass
            except Exception as e:                    # noqa: BLE001
                try:
                    self.sig.failed.emit(f'{type(e).__name__}: {e}\n{traceback.format_exc()}')
                except RuntimeError:
                    pass
        threading.Thread(target=thread, daemon=True).start()
        return True

    def _work(self, snap, st, folder, python, extra, no_cams, progress, cancel):
        """(runs on a worker thread) the rig, then the bridge in CARLA's Python; run.json's document, or None when
        cancelled."""
        progress(0.0, 'writing the rig')
        rig = SR.build(snap, st, progress=lambda f, t='': progress(0.05 * f, t), cancel=cancel)
        if rig is None:
            return None
        rig_dir = os.path.join(folder, 'rig')
        SR.export(rig, snap, rig_dir)
        args = ['--port', str(st.port), '--tm-port', str(st.tm_port), '--timeout', f'{st.timeout_s:g}']
        if no_cams:
            args.append('--no-cameras')
        cmd, env = SV.bridge_command(python, os.path.join(rig_dir, 'sim_rig.json'), folder, args)
        if extra:
            env['PYTHONPATH'] = extra + os.pathsep + env['PYTHONPATH']
        log_path = os.path.join(folder, 'bridge.log')
        import queue
        lines = queue.Queue()
        with open(log_path, 'w', encoding='utf-8') as log:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=log, text=True, env=env,
                                    start_new_session=True, cwd=folder)
            self._proc = proc

            def reader():
                for line in proc.stdout:
                    lines.put(line)
                lines.put(None)
            threading.Thread(target=reader, daemon=True).start()
            stop_at = None
            while True:
                try:
                    line = lines.get(timeout=0.2)
                except queue.Empty:
                    line = ''
                if line is None:
                    break
                if line:
                    pr = SV.parse_progress(line)
                    if pr is not None:
                        progress(0.05 + 0.95 * pr[0], pr[1])
                    else:
                        log.write(line)
                if cancel() and stop_at is None:
                    _signal(proc, signal.SIGINT)        # the bridge tidies CARLA up and removes what it wrote
                    stop_at = time.monotonic()
                if stop_at is not None and time.monotonic() - stop_at > 120:
                    _signal(proc, signal.SIGKILL)
                    break
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                _signal(proc, signal.SIGKILL)
                proc.wait(10)
        self._proc = None
        doc = SR.read_run(folder) or self._no_report(folder, proc.returncode, st.port)
        if doc.get('status') == 'cancelled':
            return None
        return doc

    BRIDGE_OUTPUT = ('nuscenes', 'rosbag', 'dataset.json', 'README.md')

    def _no_report(self, folder, code, port) -> dict:
        """(worker thread) The bridge ended without its report: killed, or aborted inside CARLA's client library (it
        aborts when the server dies mid-run). What it wrote is removed, as the bridge itself does on an error, so no
        half-written recording is opened by mistake; the rig and bridge.log stay, with a run.json saying why."""
        import shutil
        srv = self.server
        died = (srv is not None and srv.proc is not None and srv.proc.poll() is not None) or \
            not SV.port_open('127.0.0.1', port)
        why = (f'the CARLA server stopped during the run (the bridge exited with {code})' if died else
               f'the bridge exited with {code} and wrote no report')
        for name in self.BRIDGE_OUTPUT:
            path = os.path.join(folder, name)
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                elif os.path.exists(path):
                    os.remove(path)
            except OSError:
                pass
        doc = {'schema': SR.RUN_SCHEMA, 'status': 'failed', 'out_dir': folder,
               'error': f'{why}: see {short_path(os.path.join(folder, "bridge.log"))}', 'exit_code': code,
               'server_stopped': bool(died)}
        try:
            with open(os.path.join(folder, 'run.json'), 'w', encoding='utf-8') as fh:
                json.dump(doc, fh, indent=1)
        except OSError:
            pass
        return doc

    def cancel(self):
        if self.busy:
            self._cancel.set()
            if self._proc is not None:
                _signal(self._proc, signal.SIGINT)
            self.g_status.setText('Cancelling: CARLA is being tidied up…')

    def _progress(self, f, text):
        if self.busy:
            self.bar.setValue(int(f * 1000))
            if not self._cancel.is_set():
                self.g_status.setText(f'Recording · {f * 100:.0f} %' + (f' · {text}' if text else ''))

    def _done(self, doc):
        self.busy = False
        self.bar.setValue(0 if doc is None else 1000)
        if doc is None:
            self.g_status.setText('Cancelled: CARLA was tidied up and what was written removed.')
            self.ctx.msg('CARLA run cancelled', 4000)
            self._update_buttons()
            return
        self.result = doc
        if doc.get('status') != 'ok':
            self.g_status.setText('Failed: ' + str(doc.get('error', 'see the report')).splitlines()[0])
            self.out.setPlainText(json.dumps(doc, indent=1, default=str)[:20000])
            self._update_buttons()
            return
        self.out.setPlainText(self.report_text(doc))
        perf = doc.get('performance', {})
        self.g_status.setText(f'Recorded {perf.get("simulated_s", 0):g} s of drive in {perf.get("wall_s", 0):.0f} s '
                              f'({perf.get("realtime_factor", 0):.2f}× real time) into {short_path(self.out_dir)}')
        if getattr(self, '_for_project', None) is self.ctx.store.project:
            self.ctx.store.project.add_log('carla', f'CARLA drive recorded to {self.out_dir}')
        self.ctx.msg(f'CARLA drive recorded to {short_path(self.out_dir)}', 6000)
        self._update_buttons()

    def _failed(self, msg):
        self.busy = False
        self.bar.setValue(0)
        self.g_status.setText('Failed: ' + msg.splitlines()[0])
        self._update_buttons()

    @staticmethod
    def report_text(doc) -> str:
        sc = doc.get('scene', {})
        perf = doc.get('performance', {})
        c = doc.get('counts', {})
        veh = doc.get('vehicle', {})
        L = [f'CARLA {doc.get("server_version", "?")} · {sc.get("town", "")} · weather {sc.get("weather", "")} · '
             f'{sc.get("traffic", 0)} vehicles · seed {sc.get("seed", "")}',
             f'Ego: {veh.get("blueprint", "")} {veh.get("bounding_box_lwh_m", "")} m; the rig\'s origin (rear axle on '
             f'the ground) at {veh.get("ego_origin_in_actor_m", "")} m in its frame',
             f'{c.get("samples", 0)} samples, {c.get("sample_data", 0)} sample_data, {c.get("annotations", 0)} '
             f'annotations of {c.get("instances", 0)} instances, {c.get("bag_messages", 0)} bag messages',
             f'{perf.get("steps", 0)} steps at {sc.get("tick_hz", 0):g} Hz in {perf.get("recording_wall_s", 0):.1f} s: '
             f'{perf.get("steps_per_wall_s", 0):.1f} steps/s, {perf.get("realtime_factor", 0):.2f}× real time', '']
        for ch in doc.get('channels', []):
            L.append(f'  {ch["channel"] or ch["topic"]:<14} {ch["kind"]:<7} {ch["frames"]:>5} of '
                     f'{ch.get("planned_frames", ch["frames"]):>5} frames'
                     + (f' · {ch["keyframes"]} keyframes' if ch['channel'] else ''))
        mounts = [m for m in doc.get('mounts', []) if (m.get('inside_body_m') or 0.0) > 0.005]
        if mounts:
            L += ['', 'Inside the CARLA vehicle\'s body:'] + [f'  {m["tag"]} ({m["kind"]}) by {100 * m["inside_body_m"]:.1f} cm'
                                                              for m in mounts]
        if doc.get('notes'):
            L += ['', 'Notes:'] + [f'  {n}' for n in doc['notes']]
        return '\n'.join(L)

    # ------------------------------------------------------------------ the checks
    def open_in(self, page: str) -> bool:
        """Open the last recording in the drive-log check or the as-built check and run it."""
        rec = self._recording()
        win = self.ctx.window
        if not rec or win is None or not hasattr(win, 'page'):
            return False
        w = win.page(page)
        if not w.open_path(rec):
            return False
        win.set_page(page)
        if hasattr(w, 'run'):
            w.run()
        return True


def _signal(proc, sig):
    try:
        os.killpg(proc.pid, sig)
    except (ProcessLookupError, PermissionError):
        try:
            proc.send_signal(sig)
        except ProcessLookupError:
            pass
