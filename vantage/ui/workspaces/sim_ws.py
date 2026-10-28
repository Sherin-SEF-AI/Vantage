"""Simulate ▸ CARLA: run the planned rig in the CARLA simulator (CARLA renders its own town, traffic and weather; every
camera, lidar, radar, GNSS antenna and IMU of the variant records as planned) and get the drive back as a nuScenes
folder and a ROS 2 bag in the sample dataset's layout, then open it in the drive-log check and the as-built check.
Open run… shows a drive recorded before (by this page, headless with --carla-run, or by the bridge itself) exactly as
after a run, with the same checks; the Ground truth tab shows a drive's labels frame by frame (ui.gtview).

Where CARLA is (its folder and its client Python) is kept per machine in the settings; the scenario and the rig's
simulation settings are saved with the project (undoable). The server starts headless in a capped systemd scope; a
run is a background job: Vantage writes the rig (core.simrig) and runs the bridge (vantage.sim.carla_bridge) in
CARLA's Python, reading its progress; Cancel stops it and it removes what it wrote."""
from __future__ import annotations

import datetime as _dt
import json
import os
import re
import signal
import subprocess
import threading
import time
import traceback

from PyQt6.QtCore import QObject, Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (QAbstractItemView, QApplication, QComboBox, QFileDialog, QGridLayout, QHBoxLayout,
                             QHeaderView, QLineEdit, QPlainTextEdit, QProgressBar, QScrollArea, QSplitter,
                             QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget)

from ... import workdir
from ...core import simrig as SR
from ...core import weather as WX
from ...sim import carla_bridge as CB
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
    ('Scenario', 'walkers', 'Walkers', '', 0, 1.0, 1.0,
     'Pedestrians walking the sidewalks on CARLA\'s walker AI (a free drive, a scenario\'s background)'),
    ('Scenario', 'seed', 'Seed', '', 0, 1.0, 1.0, 'Spawn points and the traffic manager: the same seed, the same drive'),
    ('Scenario', 'warmup_s', 'Warm-up', 's', 1, 0.5, 1.0, 'Traffic settles this long before the sensors attach'),
    ('Scenario', 'spawn_point', 'Spawn point', '', 0, 1.0, 1.0,
     'The ego\'s spawn point on the map (−1: chosen by the seed; on a straight road for a scenario or an actors scene)'),
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
GT_TOGGLES = (
    ('semantic', 'Semantic labels', 'A class per pixel of every camera frame (CARLA\'s 29 labels; the ego apart)'),
    ('instance', 'Objects', 'Instance masks and every object\'s boxes: tight from its pixels, full from its 3D box '
                            'through the lens; occlusion, truncation, range; COCO files'),
    ('range', 'Range', 'The distance along each pixel\'s own ray (uint16 PNG, 1/256 m)'),
    ('lidar_labels', 'Lidar labels', 'nuScenes-lidarseg and -panoptic for every lidar sweep (a semantic lidar beside '
                                     'each lidar, the same rays)'))
FAULT_DEFAULT = {'rotation': {'deg': [0.0, 0.5, 0.0]}, 'translation': {'m': [0.01, 0.0, 0.0]},
                 'clock_offset': {'ms': 5.0}, 'drop': {'rate': 0.05}, 'freeze': {'from_s': 2.0, 'to_s': 3.0},
                 'range_bias': {'m': 0.05}}
FAULT_HINT = {'rotation': 'yaw, pitch, roll (°, its own frame)', 'translation': 'x, y, z (m, vehicle frame)',
              'clock_offset': 'ms (stamped late +, early −)', 'drop': 'rate 0–0.9, or frames 3, 7, …',
              'freeze': 'from – to (s)', 'range_bias': 'm along every ray'}
SCENARIO_LABELS = {'speed_kmh': 'Speed', 'gap_m': 'Gap', 'brake_at_s': 'Brakes at', 'traffic': 'Other traffic',
                   'cutter_speed_kmh': 'Cutter speed', 'cutter_offset_m': 'Cutter starts', 'cut_at_s': 'Cuts in at',
                   'side': 'From the', 'ego_speed_kmh': 'Ego speed', 'distance_m': 'Distance',
                   'walk_speed_mps': 'Walking', 'ride_speed_mps': 'Riding', 'start_at_s': 'Starts at',
                   'walker': 'Walker'}
_NUM = re.compile(r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?')


def fault_text(f: dict) -> str:
    """A fault's values as the faults table shows them."""
    k = f['kind']
    if k in ('rotation', 'translation'):
        return ', '.join(f'{x:g}' for x in f['deg' if k == 'rotation' else 'm'])
    if k == 'clock_offset':
        return f'{f["ms"]:g}'
    if k == 'range_bias':
        return f'{f["m"]:g}'
    if k == 'drop':
        return ('frames ' + ', '.join(str(x) for x in f['frames'])) if 'frames' in f else f'{f["rate"]:g}'
    return f'{f["from_s"]:g} – {f["to_s"]:g}'


def parse_fault(tag: str, kind: str, text: str) -> dict:
    """The fault a table row means; ValueError (in words) when its values do not read."""
    nums = [float(x) for x in _NUM.findall(text or '')]
    need = {'rotation': 3, 'translation': 3, 'clock_offset': 1, 'range_bias': 1, 'freeze': 2}.get(kind)
    f = {'tag': tag, 'kind': kind}
    if kind == 'drop':
        if 'frame' in (text or '').lower():
            if not nums or any(x < 0 or x != int(x) for x in nums):
                raise ValueError(f'{tag} drop: frames are whole numbers from 0.')
            f['frames'] = sorted({int(x) for x in nums})
        elif len(nums) == 1:
            f['rate'] = nums[0]
        else:
            raise ValueError(f'{tag} drop: a rate (0–0.9) or "frames 3, 7".')
    elif need is None or len(nums) != need:
        raise ValueError(f'{tag} {kind.replace("_", " ")}: {need} number{"s" * (need != 1)} ({FAULT_HINT[kind]}).')
    elif kind == 'rotation':
        f['deg'] = nums
    elif kind == 'translation':
        f['m'] = nums
    elif kind == 'clock_offset':
        f['ms'] = nums[0]
    elif kind == 'range_bias':
        f['m'] = nums[0]
    else:
        f['from_s'], f['to_s'] = nums
    good, bad = CB.clean_faults([f])
    if bad:
        raise ValueError(bad[0])
    return good[0]


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
                self.weather.setToolTip('A Vantage condition mapped to CARLA\'s weather (fog by a calibration measured '
                                        'on CARLA, light and rain by documented heuristics), or one of CARLA\'s '
                                        'presets. Fog and rain show in the camera images only; CARLA\'s rain is water '
                                        'on the lens')
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
                self.scen = QComboBox()
                for name, (lab, about, _) in CB.SCENARIOS.items():
                    self.scen.addItem(lab, name)
                    self.scen.setItemData(self.scen.count() - 1, about, Qt.ItemDataRole.ToolTipRole)
                self.scen.setToolTip('A built-in scenario (deterministic with the seed); the free drive is the town\'s '
                                     'traffic on the autopilot')
                self.scen.currentIndexChanged.connect(lambda i: self.set_scenario(self.scen.itemData(i)))
                ll.addWidget(self.scen)
                self.scen_about = label('', muted=True, small=True, wrap=True)
                ll.addWidget(self.scen_about)
                self.scen_box = QWidget()
                self.scen_grid = QGridLayout(self.scen_box)
                self.scen_grid.setContentsMargins(0, 0, 0, 0)
                self.scen_grid.setVerticalSpacing(2)
                self.scen_grid.setHorizontalSpacing(6)
                self.scen_grid.setColumnStretch(2, 1)
                self.scen_fields = {}
                self._scen_shown = None
                ll.addWidget(self.scen_box)
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
        # ---- ground truth
        ll.addSpacing(6)
        ll.addWidget(label('Ground truth', bold=True))
        self.gt_toggles = {}
        for key, text, tip in GT_TOGGLES:
            t = ToggleSwitch(text)
            t.setToolTip(tip)
            t.clicked.connect(lambda on, k=key, x=text: self.set_ground_truth(k, on, x))
            ll.addWidget(t)
            self.gt_toggles[key] = t
        self.gt_cost = label('', muted=True, small=True, wrap=True)
        ll.addWidget(self.gt_cost)
        # ---- faults
        ll.addSpacing(6)
        frow = QHBoxLayout()
        frow.addWidget(label('Faults', bold=True))
        frow.addStretch(1)
        self.b_add_fault = button('Add', 'ghost', 'plus', 'Inject a fault with a known answer: the calibration keeps '
                                                         'stating the plan, CARLA records the fault')
        self.b_add_fault.clicked.connect(self.add_fault)
        frow.addWidget(self.b_add_fault)
        ll.addLayout(frow)
        self.faults = QTableWidget(0, 4)
        for c, h in enumerate(('Sensor', 'Fault', 'Values', '')):
            self.faults.setHorizontalHeaderItem(c, QTableWidgetItem(h))
        self.faults.verticalHeader().setVisible(False)
        self.faults.verticalHeader().setDefaultSectionSize(ROW_H + 2)
        self.faults.horizontalHeader().setFixedHeight(ROW_H)
        self.faults.setShowGrid(False)
        self.faults.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        fh = self.faults.horizontalHeader()
        fh.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        fh.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        fh.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        fh.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.faults.setMinimumHeight(4 * ROW_H)
        self.faults.setMaximumHeight(9 * ROW_H)
        ll.addWidget(self.faults)
        self.fault_status = label('No fault: the drive records the rig as planned.', muted=True, small=True, wrap=True)
        ll.addWidget(self.fault_status)
        # ---- an actors scene
        ll.addSpacing(6)
        arow = QHBoxLayout()
        arow.addWidget(label('Actors scene', bold=True))
        arow.addStretch(1)
        b = button('Actors…', 'ghost', 'file', 'Load an ACTORS.json: actors placed around the ego, which stands still '
                                              '(no traffic)')
        b.clicked.connect(self._choose_actors)
        arow.addWidget(b)
        self.b_clear_actors = button('Clear', 'ghost', None, 'Back to a drive without the actors scene')
        self.b_clear_actors.clicked.connect(lambda: self._set('actors cleared', actors=[]))
        arow.addWidget(self.b_clear_actors)
        ll.addLayout(arow)
        self.actors_label = label('', muted=True, small=True, wrap=True)
        ll.addWidget(self.actors_label)
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
        self.b_open = button('Open run…', 'ghost', 'folder', 'Open a drive recorded before (a folder holding the '
                                                          'bridge\'s run.json): its report, its ground truth and the '
                                                          'checks below, as after a run')
        self.b_open.clicked.connect(lambda: self.open_run())
        row.addWidget(self.b_open)
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
        self.b_detect = button('Detection quality', 'ghost', 'eye', 'Open the recording in Detection quality: the '
                                                                    'detector on its images against its ground truth '
                                                                    '(a run with ground truth on)')
        self.b_detect.clicked.connect(self.open_detection)
        orow.addWidget(self.b_detect)
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
        from ..gtview import GroundTruthView
        self.gt = GroundTruthView()                # the run's ground truth, frame by frame
        self.out_tabs = QTabWidget()
        self.out_tabs.addTab(self.out, 'Report')
        self.out_tabs.addTab(self.gt, 'Ground truth')
        self.out_tabs.setTabToolTip(1, 'The run\'s ground truth frame by frame: the semantic, instance and range '
                                       'images, the boxes and the objects, a lidar sweep from above')
        self.out_tabs.currentChanged.connect(self._out_tab)
        gl.addWidget(self.out_tabs, 1)
        right.addWidget(gw)
        from .sim_live import SimLivePanel
        self.live = SimLivePanel(self.ctx, self)          # Live rig: the cameras streamed live, faults injected
        right.addWidget(self.live)
        right.setSizes([440, 300, 460])
        self.right = right
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
            for key, t in self.gt_toggles.items():
                t.blockSignals(True)
                t.setChecked(bool(st['ground_truth'].get(key, True)))
                t.blockSignals(False)
            i = self.scen.findData(st['scenario']['name'])
            self.scen.blockSignals(True)
            self.scen.setCurrentIndex(max(i, 0))
            self.scen.blockSignals(False)
            self._sync_scenario(st['scenario'])
            self._sync_faults(st['faults'])
            self._sync_actors(st['actors'], st['spawn_point'])
            free = st['scenario']['name'] == 'free_drive' and not st['actors']
            tf = self.fields['traffic'][0]
            tf.setEnabled(free)
            tf.setToolTip('Other vehicles on the traffic manager\'s autopilot' if free else
                          'A free drive\'s traffic: a scenario has its own (Other traffic), an actors scene none')
        finally:
            self._syncing = False

    # ------------------------------------------------------------------ ground truth, the scenario, faults, actors
    def set_ground_truth(self, key, on, text=''):
        gt = dict(self.settings.ground_truth)
        gt[key] = bool(on)
        return self._set(f'{(text or key).lower()} {"on" if on else "off"}', ground_truth=gt)

    def set_scenario(self, name):
        if name == self.settings.scenario.get('name'):
            return False
        return self._set(f'scenario {CB.SCENARIOS.get(name, ("",))[0].lower()}', scenario={'name': name, 'params': {}})

    def set_scenario_param(self, key, value):
        sc = self.settings.scenario
        params = dict(sc.get('params', {}))
        params[key] = value
        return self._set(f'{key.replace("_", " ")} {value}', scenario={'name': sc['name'], 'params': params})

    def _sync_scenario(self, sc):
        """The scenario's parameter fields (rebuilt when the scenario changes), with the settings' values."""
        name = sc['name']
        spec = CB.SCENARIOS[name]
        self.scen_about.setText(spec[1])
        if self._scen_shown != name:
            for w in list(self.scen_fields.values()) + getattr(self, '_scen_labels', []):
                w.hide()
                self.scen_grid.removeWidget(w)
                w.deleteLater()
            self.scen_fields, self._scen_labels = {}, []
            for i, (k, ps) in enumerate(spec[2].items()):
                tip = ps[-1]
                a = label(SCENARIO_LABELS.get(k, k.replace('_', ' ').capitalize()), ink2=True)
                a.setFixedWidth(104)
                a.setToolTip(tip)
                if ps[0] == 'num':
                    f = ScrubSpinBox('', 'plain', 0 if ps[4] == 'vehicles' or ps[3] - ps[2] >= 200 else 1, ps[4],
                                     step=1.0, width=118)
                    f.committed.connect(lambda v, key=k: self.set_scenario_param(key, float(v)))
                else:
                    f = SegmentedControl([(o, o) for o in ps[2]], ps[1])
                    f.changed.connect(lambda v, key=k: self.set_scenario_param(key, v))
                f.setToolTip(tip)
                self.scen_grid.addWidget(a, i, 0)
                self.scen_grid.addWidget(f, i, 1)
                self.scen_fields[k] = f
                self._scen_labels.append(a)
            self._scen_shown = name
        full = CB.clean_scenario(name, sc.get('params'))[0]['params']
        for k, f in self.scen_fields.items():
            if isinstance(f, ScrubSpinBox):
                f.set_value(full[k])
            else:
                f.set_value(full[k])
        self.scen_box.setVisible(bool(self.scen_fields))

    def _sensor_kinds(self) -> dict:
        """{tag: kind} of the variant's visible cameras and sensors (what a fault can name)."""
        v = self.ctx.store.project.variant
        out = {c.tag: 'camera' for c in v.cameras if c.visible}
        out.update({s.tag: s.kind for s in v.sensors if s.visible})
        return out

    def _sync_faults(self, faults):
        kinds = self._sensor_kinds()
        tb = self.faults
        tb.setRowCount(0)
        tb.setRowCount(len(faults))
        for r, f in enumerate(faults):
            sc = QComboBox()
            tags = list(kinds) + ([f['tag']] if f['tag'] not in kinds else [])
            for t in tags:
                sc.addItem(t, t)
            sc.setCurrentIndex(tags.index(f['tag']))
            kc = QComboBox()
            allowed = [k for k, (applies, _) in CB.FAULT_KINDS.items() if kinds.get(f['tag']) in applies] or [f['kind']]
            if f['kind'] not in allowed:
                allowed.append(f['kind'])
            for k in allowed:
                kc.addItem(k.replace('_', ' '), k)
                kc.setItemData(kc.count() - 1, CB.FAULT_KINDS[k][1], Qt.ItemDataRole.ToolTipRole)
            kc.setCurrentIndex(allowed.index(f['kind']))
            ed = QLineEdit(fault_text(f))
            ed.setPlaceholderText(FAULT_HINT[f['kind']])
            ed.setToolTip(FAULT_HINT[f['kind']])
            rm = button('', 'ghost', 'close', 'Remove this fault')
            sc.currentIndexChanged.connect(lambda _i, row=r: self._fault_changed(row, 'tag'))
            kc.currentIndexChanged.connect(lambda _i, row=r: self._fault_changed(row, 'kind'))
            ed.editingFinished.connect(lambda row=r: self._fault_changed(row, 'values'))
            rm.clicked.connect(lambda _=False, row=r: self.remove_fault(row))
            for c, w in enumerate((sc, kc, ed, rm)):
                tb.setCellWidget(r, c, w)
        n = len(faults)
        self.fault_status.setText(f'{n} fault{"s" * (n != 1)}: CARLA records them, the calibration states the plan '
                                  '(run.json faults_injected).' if n else
                                  'No fault: the drive records the rig as planned.')

    def _row(self, r):
        tb = self.faults
        return tb.cellWidget(r, 0).currentData(), tb.cellWidget(r, 1).currentData(), tb.cellWidget(r, 2).text()

    def _fault_changed(self, row, what):
        if self._syncing:
            return False
        faults = [dict(f) for f in self.settings.faults]
        if row >= len(faults):
            return False
        tag, kind, text = self._row(row)
        kinds = self._sensor_kinds()
        if what == 'tag' and kinds.get(tag) not in CB.FAULT_KINDS[kind][0]:
            kind = next(k for k, (applies, _) in CB.FAULT_KINDS.items() if kinds.get(tag) in applies)
        if what in ('tag', 'kind') and kind != faults[row]['kind']:
            f = dict({'tag': tag, 'kind': kind}, **FAULT_DEFAULT[kind])        # a new kind: its default values
        else:
            try:
                f = parse_fault(tag, kind, text)
            except ValueError as e:
                self.fault_status.setText(str(e))
                return False
        faults[row] = f
        return self._set(f'fault {tag} {kind.replace("_", " ")} {fault_text(f)}', faults=faults)

    def add_fault(self):
        kinds = self._sensor_kinds()
        if not kinds:
            self.ctx.msg('The variant has no sensor to inject a fault into.', 4000)
            return False
        tag = next(iter(kinds))
        kind = 'rotation'
        f = dict({'tag': tag, 'kind': kind}, **FAULT_DEFAULT[kind])
        return self._set(f'fault added ({tag})', faults=list(self.settings.faults) + [f])

    def remove_fault(self, row):
        faults = list(self.settings.faults)
        if 0 <= row < len(faults):
            f = faults.pop(row)
            return self._set(f'fault removed ({f["tag"]} {f["kind"].replace("_", " ")})', faults=faults)
        return False

    def _choose_actors(self):
        f, _ = QFileDialog.getOpenFileName(self, 'An actors scene (ACTORS.json)', self.working(), 'JSON (*.json)')
        if f:
            self.load_actors(f)

    def load_actors(self, path) -> bool:
        """Read ACTORS.json into the settings (undoable); False with the reason shown when it does not read."""
        try:
            with open(path, encoding='utf-8') as fh:
                items = json.load(fh)
        except (OSError, ValueError) as e:
            self.ctx.msg(f'{short_path(path)} does not read: {e}', 6000)
            return False
        actors, bad = CB.clean_actors(items)
        if not actors:
            self.ctx.msg(f'No actor in {short_path(path)}' + (f': {bad[0]}' if bad else '.'), 6000)
            return False
        self._set(f'actors scene ({len(actors)} from {os.path.basename(path)})', actors=actors)
        if bad:
            self.ctx.msg(f'{len(bad)} entr{"ies" if len(bad) != 1 else "y"} left out: {bad[0]}', 6000)
        return True

    def _sync_actors(self, actors, spawn_point):
        self.b_clear_actors.setEnabled(bool(actors))
        if not actors:
            self.actors_label.setText('None: the scenario above runs. An actors scene puts the actors of an '
                                      'ACTORS.json around the ego, which stands still (no traffic).')
            return
        kinds = {}
        for a in actors:
            kinds[a['kind']] = kinds.get(a['kind'], 0) + 1
        self.actors_label.setText(f'{len(actors)} actors around a still ego (the scenario is not run): '
                                  + ', '.join(f'{n} {k}' for k, n in kinds.items())
                                  + ('; spawn point chosen by the seed' if spawn_point < 0 else
                                     f'; spawn point {spawn_point}'))

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
        if getattr(self, 'live', None) is not None and self.live.running:
            self.live.stop('the server is stopping')
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
        gt = d.get('ground_truth') or {}
        gt_mpx = float(gt.get('gt_render_mpx', 0.0))
        rays = sum(L['rays_per_s'] for L in lids)
        self.tiles['step'].set(f'{sim["tick_hz"]:g}', 'Hz', f'{sim["dt_s"] * 1000:.2f} ms · warm-up {sim["warmup_ticks"]} steps')
        self.tiles['render'].set(f'{mpx + gt_mpx:.1f}', 'Mpx',
                                 (f'{sum(len(c["views"]) for c in cams)} views of {len(cams)} '
                                  f'camera{"s" * (len(cams) != 1)}' + (f' · {gt_mpx:.1f} ground truth' if gt_mpx else ''))
                                 if cams else 'no camera')
        kinds = gt.get('camera_kinds') or []
        if cams and kinds:
            self.gt_cost.setText(f'Renders {gt_mpx:.1f} Mpx per frame beside the images\' {mpx:.1f} '
                                 f'({" and ".join(k.replace("_", " ") for k in kinds)} cameras beside every view)'
                                 + (f'; a semantic lidar beside {len(lids)} lidar{"s" * (len(lids) != 1)}'
                                    if gt.get('lidar_labels') and lids else '') + '.')
        elif gt.get('lidar_labels') and lids:
            self.gt_cost.setText(f'A semantic lidar beside {len(lids)} lidar{"s" * (len(lids) != 1)}; no camera '
                                 'ground truth.')
        else:
            self.gt_cost.setText('No ground truth.')
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
        self.out_dir = ''
        self.out.setPlainText('')
        self.g_status.setText('')
        self.gt.set_run('')
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
        self.b_open.setEnabled(not self.busy)
        rec = self._recording()
        self.b_drivelog.setEnabled(bool(rec) and not self.busy)
        self.b_asbuilt.setEnabled(bool(rec) and not self.busy)
        self.b_detect.setEnabled(bool(rec) and not self.busy and self._has_ground_truth())

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
        """The bridge's command line for the rig exported to rig_path (Export rig… writes faults.json and actors.json
        next to it when there are any): every setting the run uses, said on the line."""
        import shlex
        st = self.settings
        args = [self.python or 'python', '-m', 'vantage.sim.carla_bridge', rig_path, out_dir, '--town', st.town,
                '--duration', f'{st.duration_s:g}', '--traffic', str(st.traffic), '--seed', str(st.seed),
                '--port', str(st.port), '--tm-port', str(st.tm_port)]
        if st.weather in SR.CARLA_WEATHERS:
            args += ['--weather', st.weather]
        gt = [k for k in CB.GT_KEYS if st.ground_truth.get(k, True)]
        args += ['--ground-truth', ','.join(gt) if gt else 'none']
        here = os.path.dirname(rig_path)
        if st.actors:
            args += ['--actors', os.path.join(here, 'actors.json')]
        else:
            args += ['--scenario', st.scenario['name']]
            if st.scenario.get('params'):
                args += ['--scenario-params', json.dumps(st.scenario['params'], separators=(',', ':'))]
        if st.faults:
            args += ['--faults', os.path.join(here, 'faults.json')]
        if st.spawn_point >= 0:
            args += ['--spawn-point', str(st.spawn_point)]
        if st.walkers:
            args += ['--walkers', str(st.walkers)]
        src = SV.vantage_source()
        return (f'PYTHONPATH={shlex.quote(src)} ' if src else '') + ' '.join(shlex.quote(a) for a in args)

    def _copy_command(self):
        QApplication.clipboard().setText(self.command_line())
        self.ctx.msg('Copied the bridge\'s command line', 3000)

    # ------------------------------------------------------------------ the run
    def _run_into(self):
        folder = QFileDialog.getExistingDirectory(self, 'Record the CARLA drive into', self.working())
        if folder:
            self.run(folder)

    def runs_folder(self) -> str:
        """Where this page records the project's runs (its working folder's <project>/carla)."""
        return workdir.project_dir(self.ctx.store.project, self.working(), 'carla')

    def open_run(self, folder=None) -> bool:
        """Show a drive recorded before (this page's, a headless --carla-run's, the bridge's own: a folder holding
        run.json) as after a run: its report, its ground truth, and the Drive-log check, As-built check and Detection
        quality buttons on it. The dialog starts in the project's runs folder. False (said) when it is not one."""
        if self.busy:
            return False
        if folder is None:
            start = self.runs_folder()
            folder = QFileDialog.getExistingDirectory(self, 'Open a CARLA drive (a folder holding run.json)',
                                                      start if os.path.isdir(start) else self.working())
            if not folder:
                return False
        folder = os.path.normpath(folder)
        doc = SR.read_run(folder)
        if doc is None:
            why = ('its run.json is not a CARLA run (vantage.sim_run/1)'
                   if os.path.isfile(os.path.join(folder, 'run.json')) else 'it holds no run.json')
            self.g_status.setText(f'{short_path(folder)} is not a CARLA drive: {why}.')
            self.ctx.msg(self.g_status.text(), 6000)
            return False
        self.result, self.out_dir = doc, folder
        self.bar.setValue(0)
        if doc.get('status') != 'ok':
            self.out.setPlainText(json.dumps(doc, indent=1, default=str)[:20000])
            self.g_status.setText(f'Opened {short_path(folder)}: the run did not end well ({doc.get("status")}): '
                                  + str(doc.get('error', 'see the report')).splitlines()[0])
            self.gt.set_run('')
        else:
            self.out.setPlainText(self.report_text(doc))
            has = self.gt.set_run(folder)
            perf, c = doc.get('performance', {}), doc.get('counts', {})
            self.g_status.setText(f'Opened {short_path(folder)}: {perf.get("simulated_s", 0):g} s of drive, '
                                  + (f'{c.get("gt_frames", 0)} camera frames with ground truth' if has else
                                     'no ground truth'))
        self.ctx.msg(f'Opened the CARLA drive {os.path.basename(folder)}', 4000)
        self._update_buttons()
        return True

    def run(self, folder=None) -> bool:
        """Write the rig into <folder>/rig and run the bridge into folder as a background job."""
        if self.busy:
            return False
        if getattr(self, 'live', None) is not None and self.live.running:
            self.g_status.setText('Live streaming is on: stop it first (Live rig ▸ Stop live).')
            self.ctx.msg(self.g_status.text(), 5000)
            return False
        win = self.ctx.window
        other = win.carla_user('sim') if win is not None and hasattr(win, 'carla_user') else ''
        if other:
            self.g_status.setText(other)
            self.ctx.msg(other, 5000)
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
        self.gt.set_run('')
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
        mem = max(SV.BRIDGE_MEMORY_GB, float(rig.doc['simulation'].get('bridge_memory_gb') or 0.0) + 1.0)
        cmd, env = SV.bridge_command(python, os.path.join(rig_dir, 'sim_rig.json'), folder, args, memory_gb=mem)
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
        if not died and self.python:                   # a killed bridge leaves the server synchronous, its actors
            try:
                cmd, env = SV.tidy_command(self.python, port)
                if self.extra_pythonpath:
                    env['PYTHONPATH'] = self.extra_pythonpath + os.pathsep + env['PYTHONPATH']
                r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=120)
                doc['tidied'] = json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 else \
                    {'error': (r.stderr or '').strip()[-300:]}
            except Exception as e:                     # noqa: BLE001 (best effort, said)
                doc['tidied'] = {'error': f'{type(e).__name__}: {e}'}
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
        self.gt.set_run(self.out_dir)
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
        gt = doc.get('ground_truth') or {}
        if c.get('gt_frames') or c.get('lidarseg'):
            L.append('')
            L.append(f'Ground truth: {c.get("gt_frames", 0)} camera frames, {c.get("gt_objects", 0)} objects, '
                     f'{c.get("lidarseg", 0)} lidar sweeps labelled ({gt.get("index") or "ground_truth/index.json"}; '
                     f'{gt.get("seconds", 0):.0f} s of work)')
            for tag, lb in (gt.get('lidar_labels') or {}).items():
                L.append(f'  {tag}: the semantic lidar matched ray for ray on {100 * lb["matched_in_order_share"]:.2f} % '
                         f'of its steps (range RMS {lb["range_rms_diff_m"]} m, the lidar\'s noise)')
        sc = doc.get('scenario') or {}
        if sc.get('name') and sc['name'] != 'free_drive':
            L += ['', f'Scenario {sc["name"]}: ' + ', '.join(f'{k} {v}' for k, v in (sc.get('params') or {}).items())]
            L += [f'  {e["t_s"]:7.3f} s  {e["what"]}' + ''.join(f', {k} {v}' for k, v in e.items()
                                                                 if k not in ('t_s', 'what')) for e in sc.get('events', [])]
        if doc.get('actors'):
            L += ['', 'Actors scene:'] + [
                f'  {a["id"]:<12} {a["kind"]:<12} ' + (f'actor {a["actor_id"]} {a["blueprint"]}, placed within '
                                                      f'{1000 * a.get("error_m", 0):.0f} mm' if a.get('actor_id')
                                                      else f'not spawned: {a.get("error", "")}')
                for a in doc['actors']]
        if doc.get('faults_injected'):
            L += ['', 'Faults injected (the calibration states the plan):']
            for f in doc['faults_injected']:
                L.append(f'  {f["tag"]} {f["kind"]}: {fault_text(f)}'
                         + (f' → pose change {f["pose_delta_deg"]} °' if f.get('pose_delta_deg') else '')
                         + (f' ({len(f["dropped_frames"])} frames)' if 'dropped_frames' in f else ''))
        mounts = [m for m in doc.get('mounts', []) if (m.get('inside_body_m') or 0.0) > 0.005]
        if mounts:
            L += ['', 'Inside the CARLA vehicle\'s body:'] + [f'  {m["tag"]} ({m["kind"]}) by {100 * m["inside_body_m"]:.1f} cm'
                                                              for m in mounts]
        if doc.get('notes'):
            L += ['', 'Notes:'] + [f'  {n}' for n in doc['notes']]
        return '\n'.join(L)

    # ------------------------------------------------------------------ the checks
    def _has_ground_truth(self) -> bool:
        from ...core import groundtruth as GT
        return bool(self.out_dir) and os.path.isfile(os.path.join(self.out_dir, GT.INDEX))

    def open_detection(self) -> bool:
        """Open the last run (with its ground truth) in Simulate ▸ Detection quality."""
        win = self.ctx.window
        if not self._recording() or not self._has_ground_truth() or win is None or not hasattr(win, 'page'):
            return False
        w = win.page('detection')
        if not w.open_path(self.out_dir):
            return False
        win.set_page('detection')
        return True

    def show_ground_truth(self):
        """The Ground truth tab in view (given room in the splitter when it has little)."""
        if self.out_tabs.currentWidget() is self.gt:
            self._out_tab(self.out_tabs.currentIndex())
        self.out_tabs.setCurrentWidget(self.gt)

    def _out_tab(self, i):
        """The Ground truth tab shown: the run's panel takes most of the height when it has little (the live panel
        keeps a share while it streams; the splitter gives it back)."""
        if self.out_tabs.widget(i) is not self.gt:
            return
        sizes = self.right.sizes()
        total = sum(sizes)
        if len(sizes) == 3 and total > 0 and sizes[1] < 0.6 * total:
            live = int(total * 0.2) if self.live.running else 0
            top = int(total * 0.16)
            self.right.setSizes([top, total - top - live, live])

    def show_live(self):
        """The live panel in view (Simulate ▸ CARLA live rig): given room in the splitter when it has none."""
        sizes = self.right.sizes()
        if len(sizes) == 3 and sizes[2] < 200:
            total = sum(sizes)
            self.right.setSizes([int(total * 0.3), int(total * 0.25), total - int(total * 0.3) - int(total * 0.25)])
        self.live.setFocus()

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
