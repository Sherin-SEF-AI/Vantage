"""Export formats: documents, extrinsics, calibration and robotics configs, images and CAD mount points.

Every format renders to bytes from (project, analysis result, options), so the same code feeds the live
preview and the write to disk.

Conventions
-----------
* World frame: ISO 8855 (x forward, y left, z up), metres, origin at the rear-axle centre on the ground.
* Export frame: the same axes with the origin moved to ``ExportOptions.frame``; positions export as
  ``pos - origin_offset``, scaled to the chosen units. Angles are unchanged by the frame shift.
* Camera pose: intrinsic Z-Y-X yaw, pitch, roll (pitch positive down), body optical axis along +x,
  ``R = rot_zyx(yaw, pitch, roll)`` maps body to world.
* OpenCV optical frame: x right, y down, z forward. ``W2O = BODY_TO_OPTICAL @ R.T`` maps world to optical.
* Intrinsics: Kannala-Brandt (same model as ``cv::fisheye``); other projections are fitted by least squares.
  fx, fy, cx, cy are in OpenCV pixel coordinates (pixel centres on integers), so the sensor centre of a
  1920-wide image is cx = 959.5. A camera with a unit calibration (core.calib) exports its own intrinsics.
"""
from __future__ import annotations

import base64
import csv
import datetime as _dt
import html
import io
import json
import math
import os
import re
import struct
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from . import rules
from .lens import Intrinsics, Lens, Module, r_of_theta, resolve
from .mathutil import BODY_TO_OPTICAL, quat_from_matrix, rot_zyx
from .project import FRAME_PRESETS, Camera, Project, Variant

__all__ = ['ExportOptions', 'ExportFormat', 'FORMATS', 'GROUPS', 'format_by_id', 'preview', 'export',
           'default_out_dir', 'kb_coefficients', 'camera_model', 'CameraModel']

SCHEMA_EXTRINSICS = 'vantage.extrinsics/1'
SCHEMA_JETSON = 'vantage.jetson/1'
ROTATION_TEXT = 'intrinsic Z-Y-X (yaw, pitch, roll); pitch positive down'
AXES_TEXT = 'x forward, y left, z up'
OPTICAL_TEXT = 'OpenCV optical: x right, y down, z forward'
SENSOR_FRAME_TEXT = ('lidars and radars: x forward, y left, z up (REP 103) at the optical centre; GNSS antennas: the '
                     'same axes at the phase centre; IMUs: the IMU\'s own axes at their centre; T_world_sensor maps '
                     'sensor-frame points into this frame')
PLATFORM = 'Jetson AGX Orin'
AIM_LENGTH_M = 0.3

CAM_COLORS = ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#008300', '#9085e9', '#e66767']
CAM_ACI = [5, 30, 3, 40, 6, 94, 171, 1]          # nearest AutoCAD colour index per slot, for DXF
SEVERITY_STYLE = {                                # label, fill, text
    'critical': ('CRITICAL', '#d03b3b', '#ffffff'),
    'serious': ('SERIOUS', '#ec835a', '#1b1b1a'),
    'warning': ('WARNING', '#fab219', '#1b1b1a'),
    'good': ('GOOD', '#0ca30c', '#ffffff'),
}
INK, INK2, MUTED, HAIR, TINT, GRID = '#1b1b1a', '#52514e', '#898781', '#d6d4cc', '#f4f3ef', '#ecebe6'

_ORIGIN_TEXT = {
    'rear_axle': 'rear axle centre, on the ground',
    'front_axle': 'front axle centre, on the ground',
    'front_bumper': 'front of the front bumper, on the centreline, on the ground',
    'centre': 'midpoint between the bumpers, on the centreline, on the ground',
}
_FRAME_SHORT = {'rear_axle': 'rear axle', 'front_axle': 'front axle', 'front_bumper': 'front bumper',
                'centre': 'vehicle centre', 'custom': 'custom origin'}


# ====================================================================== public types
@dataclass
class ExportOptions:
    frame: str = 'rear_axle'
    units: str = 'm'
    angles: str = 'deg'
    out_dir: str = ''
    images: dict | None = None     # {'viewport', 'bev', 'coverage', 'cam:<uid>'} -> PNG bytes (from the UI)
    issues: list | None = None     # list[rules.Issue]
    worst_single: tuple | None = None   # (blind m2, cam uid, perturbation) or None
    lut: object | None = None           # stitch.StitchLUT for the bird's-eye remap tables (built on demand if None)
    engine: object | None = None        # analysis.Engine to build them with (a fresh one if None)
    bowl_shape: object | None = None    # bowl.BowlShape for the 3D surround-view bowl (the project's if None)
    bowl_texture: object | None = None  # RGBA texture for it (a synthetic parking scene is rendered if None)


@dataclass
class ExportFormat:
    id: str
    group: str
    label: str
    ext_label: str
    filename: Callable          # (project, variant) -> str | list[str], relative to out_dir
    kind: str                   # 'text' | 'binary'
    multi: bool = False


def _safe(s: str) -> str:
    t = re.sub(r'[^A-Za-z0-9_-]+', '_', s or '').strip('_')
    return t or 'cam'


def safe_tags(variant: Variant) -> dict:
    """uid -> file and frame safe tag, unique ignoring case."""
    out, used = {}, set()
    for c in variant.cameras:
        base = _safe(c.tag)
        t, k = base, 2
        while t.lower() in used:
            t, k = f'{base}_{k}', k + 1
        used.add(t.lower())
        out[c.uid] = t
    return out


def sensor_tags(variant: Variant) -> dict:
    """uid -> file and frame safe tag of each lidar and radar, unique ignoring case and apart from the cameras'."""
    out, used = {}, {t.lower() for t in safe_tags(variant).values()}
    for s in variant.sensors:
        base = _safe(s.tag)
        t, k = base, 2
        while t.lower() in used:
            t, k = f'{base}_{k}', k + 1
        used.add(t.lower())
        out[s.uid] = t
    return out


def _bracket_names(p, v):
    """An STL per visible camera that takes a bracket, then the JSON and the sheet. A camera the generator turns down
    (it needs a larger standoff, no surface is near) writes no STL."""
    from .brackets import safe_tag
    return [f'brackets/{safe_tag(c.tag)}.stl' for c in v.cameras if c.visible and c.bracket != 'None'] + \
        ['brackets/brackets.json', 'brackets/brackets_sheet.png']


def _view_cams(variant: Variant) -> list:
    return [c for c in variant.cameras if c.visible]


GROUPS = ['Documents', 'Extrinsics', 'Calibration and robotics', 'Datasets', 'Surround view', 'Images', 'Mount points']

FORMATS: list[ExportFormat] = [
    ExportFormat('mount_sheet_pdf', 'Documents', 'Mount sheet', 'PDF', lambda p, v: 'mount-sheet.pdf', 'binary'),
    ExportFormat('report_html', 'Documents', 'Report', 'HTML', lambda p, v: 'report.html', 'text'),
    ExportFormat('report_pdf', 'Documents', 'Report', 'PDF', lambda p, v: 'report.pdf', 'binary'),
    ExportFormat('blind_volume', 'Documents', 'Blind volume', 'JSON, OBJ',
                 lambda p, v: ['blind_volume.json', 'blind_volume.obj'], 'text', multi=True),
    ExportFormat('perception_range', 'Documents', 'Perception range', 'JSON, PNG',
                 lambda p, v: ['perception_range/perception_range.json', 'perception_range/perception_range.png'],
                 'binary', multi=True),
    ExportFormat('sensor_coverage', 'Documents', 'Lidar and radar coverage', 'JSON, PNG',
                 lambda p, v: ['sensor_coverage/sensor_coverage.json', 'sensor_coverage/sensor_coverage.png'],
                 'binary', multi=True),
    ExportFormat('sync_plan', 'Documents', 'Sync and timing plan', 'JSON, PNG',
                 lambda p, v: ['sync_plan/sync_plan.json', 'sync_plan/sync_plan.png'], 'binary', multi=True),
    ExportFormat('recording_budget', 'Documents', 'Recording budget', 'JSON, CSV',
                 lambda p, v: ['recording_budget/recording_budget.json', 'recording_budget/recording_budget.csv'],
                 'text', multi=True),
    ExportFormat('power_budget', 'Documents', 'Power budget', 'JSON, CSV',
                 lambda p, v: ['power_budget/power_budget.json', 'power_budget/power_budget.csv',
                               'power_budget/power_circuits.csv'], 'text', multi=True),
    ExportFormat('gnss_imu', 'Documents', 'GNSS and IMU', 'JSON, TXT, PNG',
                 lambda p, v: ['gnss_imu/gnss_imu.json', 'gnss_imu/ins_lever_arms.txt', 'gnss_imu/gnss_sky.png'],
                 'binary', multi=True),
    ExportFormat('exposure', 'Documents', 'Sun glare and dirt', 'JSON, PNG',
                 lambda p, v: ['exposure/exposure.json', 'exposure/exposure.png'], 'binary', multi=True),
    ExportFormat('brackets', 'Mount points', 'Mount brackets', 'STL, JSON, PNG', _bracket_names, 'binary', multi=True),
    ExportFormat('requirements', 'Documents', 'Requirements scorecard and review pack', 'JSON, CSV, PDF',
                 lambda p, v: ['requirements/requirements.json', 'requirements/requirements.csv',
                               'requirements/review_pack.pdf'], 'binary', multi=True),
    ExportFormat('fleet', 'Documents', 'Fleet', 'JSON, CSV, PNG',
                 lambda p, v: ['fleet/fleet.json', 'fleet/fleet.csv', 'fleet/fleet.png'], 'binary', multi=True),
    ExportFormat('extrinsics_json', 'Extrinsics', 'Extrinsics', 'JSON', lambda p, v: 'extrinsics.json', 'text'),
    ExportFormat('extrinsics_yaml', 'Extrinsics', 'Extrinsics', 'YAML', lambda p, v: 'extrinsics.yaml', 'text'),
    ExportFormat('extrinsics_csv', 'Extrinsics', 'Extrinsics', 'CSV', lambda p, v: 'extrinsics.csv', 'text'),
    ExportFormat('opencv_yaml', 'Calibration and robotics', 'OpenCV calibration, per camera', 'YAML ×N',
                 lambda p, v: [f'opencv/{t}.yaml' for t in safe_tags(v).values()], 'text', multi=True),
    ExportFormat('ros2', 'Calibration and robotics', 'ROS 2 URDF + static TF', 'URDF, PY',
                 lambda p, v: ['vantage_cameras.urdf', 'static_tf.launch.py'], 'text', multi=True),
    ExportFormat('jetson_json', 'Calibration and robotics', 'Jetson camera config', 'JSON',
                 lambda p, v: 'jetson_cameras.json', 'text'),
    ExportFormat('nuscenes_calib', 'Datasets', 'nuScenes sensor calibration', 'JSON',
                 lambda p, v: ['nuscenes/calibrated_sensor.json', 'nuscenes/sensor.json', 'nuscenes/README.md'], 'text',
                 multi=True),
    ExportFormat('kitti_calib', 'Datasets', 'KITTI calibration', 'TXT',
                 lambda p, v: ['kitti/calib_cam_to_cam.txt', 'kitti/calib_cams_to_vehicle.txt', 'kitti/README.txt']
                 + (['kitti/calib_velo_to_cam.txt'] if v.cameras and any(s.kind == 'lidar' for s in v.sensors) else [])
                 + (['kitti/calib_imu_to_velo.txt'] if any(s.kind == 'lidar' for s in v.sensors)
                    and any(s.kind == 'imu' for s in v.sensors) else []),
                 'text', multi=True),
    ExportFormat('ros_camera_info', 'Datasets', 'ROS camera_info, per camera', 'YAML ×N',
                 lambda p, v: [f'camera_info/{t}.yaml' for t in safe_tags(v).values()], 'text', multi=True),
    ExportFormat('kalibr_camchain', 'Datasets', 'Kalibr camchain', 'YAML', lambda p, v: 'kalibr/camchain.yaml', 'text'),
    ExportFormat('bev_lut', 'Surround view', 'Bird\'s-eye remap tables', 'NPZ, BIN',
                 lambda p, v: ['bev_lut/bev_lut.json', 'bev_lut/bev_lut.npz'] + [f'bev_lut/{t}_{k}.bin' for t in safe_tags(v).values()
                                                                                 for k in ('map', 'weight')] + ['bev_lut/body_mask.bin',
                                                                                                                'bev_lut/bev_lut.txt'],
                 'binary', multi=True),
    ExportFormat('bowl_mesh', 'Surround view', '3D surround-view bowl', 'OBJ, NPZ, PNG',
                 lambda p, v: ['bowl/bowl.obj', 'bowl/bowl.mtl', 'bowl/bowl.png', 'bowl/bowl_lut.npz', 'bowl/bowl_lut.json'],
                 'binary', multi=True),
    ExportFormat('guidelines', 'Surround view', 'Parking guidelines, per camera', 'JSON, PNG ×N',
                 lambda p, v: [f'guidelines/{safe_tags(v)[c.uid]}.{e}' for c in _view_cams(v) for e in ('json', 'png')],
                 'binary', multi=True),
    ExportFormat('stitched_objects', 'Surround view', 'Objects in the stitched view', 'JSON, PNG',
                 lambda p, v: ['stitched_objects/stitched_objects.json', 'stitched_objects/stitched_objects.png'],
                 'binary', multi=True),
    ExportFormat('bev_stitcher', 'Surround view', 'Reference stitcher', 'C++, PY',
                 lambda p, v: ['bev_lut/stitcher/bev_stitch.cpp', 'bev_lut/stitcher/CMakeLists.txt',
                               'bev_lut/stitcher/bev_stitch.py', 'bev_lut/stitcher/README.md'], 'text', multi=True),
    ExportFormat('viewport_png', 'Images', 'Viewport', 'PNG', lambda p, v: 'viewport.png', 'binary'),
    ExportFormat('bev_png', 'Images', "Bird's-eye", 'PNG', lambda p, v: 'birds-eye.png', 'binary'),
    ExportFormat('camera_views_png', 'Images', 'All camera views', 'PNG ×N',
                 lambda p, v: [f'views/{safe_tags(v)[c.uid]}.png' for c in _view_cams(v)], 'binary', multi=True),
    ExportFormat('dxf', 'Mount points', 'Mount points', 'DXF', lambda p, v: 'mount-points.dxf', 'text'),
    ExportFormat('step', 'Mount points', 'Mount points', 'STEP', lambda p, v: 'mount-points.step', 'text'),
]
_BY_ID = {f.id: f for f in FORMATS}
_IMAGE_KEYS = {'viewport_png': ('viewport', 'viewport capture'), 'bev_png': ('bev', "bird's-eye capture")}


def format_by_id(fmt_id: str) -> ExportFormat:
    try:
        return _BY_ID[fmt_id]
    except KeyError:
        raise ValueError(f'Unknown export format {fmt_id!r}. Known: {", ".join(_BY_ID)}') from None


def _names(fmt: ExportFormat, project: Project, variant: Variant) -> list:
    n = fmt.filename(project, variant)
    return [n] if isinstance(n, str) else list(n)


def _slug(s: str) -> str:
    t = re.sub(r'[^a-z0-9]+', '-', (s or '').lower()).strip('-')
    return t or 'variant'


def default_out_dir(project: Project, variant: Variant | None = None) -> str:
    """<project folder>/exports/<variant slug>; needs a saved project."""
    variant = variant or project.variant
    if not project.path:
        raise ValueError('No output folder: set ExportOptions.out_dir or save the project first.')
    return os.path.join(os.path.dirname(os.path.abspath(project.path)), 'exports', _slug(variant.name))


# ====================================================================== numbers
def _now() -> _dt.datetime:
    return _dt.datetime.now().astimezone().replace(microsecond=0)


def _fs(v, dp: int) -> str:
    """Fixed decimals, plain ASCII minus, never '-0.000'."""
    s = f'{float(v):.{dp}f}'
    if s.startswith('-') and s.strip('-0.') == '':
        s = s[1:]
    return s


def _fn(v, dp: int):
    """Rounded number for JSON/YAML (int when dp == 0, no negative zero)."""
    r = round(float(v), dp) + 0.0
    if r == 0:
        r = 0.0
    return int(r) if dp == 0 else r


def _sig(v, n: int = 10) -> float:
    return float(f'{float(v):.{n}g}') + 0.0


# ====================================================================== camera model
def kb_coefficients(intr: Intrinsics, samples: int = 256) -> tuple[tuple, float]:
    """Kannala-Brandt k1..k4 for an Intrinsics, and the fit RMS in pixels (0 for native KB lenses).

    theta_d = theta (1 + k1 theta^2 + k2 theta^4 + k3 theta^6 + k4 theta^8), r_px = f_px theta_d,
    i.e. the OpenCV cv::fisheye model. Non-KB projections are fitted by least squares to r_of_theta
    sampled over theta in [0, max_theta].
    """
    if intr.projection == 'kannala_brandt':
        return tuple(float(k) for k in intr.kb), 0.0
    th = np.linspace(0.0, max(float(intr.max_theta), 1e-3), samples + 1)[1:]
    target = np.asarray(r_of_theta(th, 1.0, intr.projection), float)
    A = np.stack([th ** 3, th ** 5, th ** 7, th ** 9], axis=1)
    k, *_ = np.linalg.lstsq(A, target - th, rcond=None)
    resid = th + A @ k - target
    return tuple(float(x) for x in k), float(np.sqrt(np.mean(resid ** 2))) * intr.f_px


def _rotvec(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> Rodrigues vector, robust near 180 degrees (via the quaternion)."""
    x, y, z, w = quat_from_matrix(R)
    if w < 0:
        x, y, z, w = -x, -y, -z, -w
    s = math.sqrt(x * x + y * y + z * z)
    if s < 1e-12:
        return np.zeros(3)
    return np.array([x, y, z]) / s * (2.0 * math.atan2(s, w))


@dataclass
class CameraModel:
    cam: Camera
    tag: str                 # file/frame safe tag
    module: Module
    lens: Lens
    intr: Intrinsics
    fx: float
    fy: float
    cx: float
    cy: float
    k: tuple
    fit_rms_px: float        # 0 when the lens is natively Kannala-Brandt
    R: np.ndarray            # body -> world
    W2O: np.ndarray          # world -> optical
    p: np.ndarray            # position in the export frame and units (unrounded)

    @property
    def fitted(self) -> bool:
        return self.intr.projection != 'kannala_brandt'

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])

    @property
    def O2W(self) -> np.ndarray:
        return self.W2O.T

    @property
    def rvec(self) -> np.ndarray:
        return _rotvec(self.W2O)

    @property
    def tvec(self) -> np.ndarray:
        return -self.W2O @ self.p

    @property
    def aim(self) -> np.ndarray:
        return self.R[:, 0]


def camera_model(project: Project, cam: Camera, origin=(0.0, 0.0, 0.0), scale: float = 1.0, tag: str = '') -> CameraModel:
    module, lens = project.module_for(cam), project.lens_for(cam)
    intr = resolve(module, lens)
    k, rms = kb_coefficients(intr)
    R = rot_zyx(cam.yaw, cam.pitch, cam.roll)
    W2O = BODY_TO_OPTICAL @ R.T
    p = (cam.pos() - np.asarray(origin, float)) * scale
    f = intr.f_px
    # OpenCV pixel coordinates put pixel centres on integers; Vantage's sensor coordinates on i + 0.5
    return CameraModel(cam, tag or _safe(cam.tag), module, lens, intr, f, f * intr.aspect, float(intr.cx) - 0.5,
                       float(intr.cy) - 0.5, k, rms, R, W2O, p)


# ====================================================================== context
@dataclass
class _Ctx:
    project: Project
    variant: Variant
    result: object
    opts: ExportOptions
    frame: str
    units: str
    angles: str
    origin: np.ndarray
    now: _dt.datetime
    models: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    _measured: dict | None = None
    _issues: tuple | None = None

    # -- units
    @property
    def scale(self):
        return 1000.0 if self.units == 'mm' else 1.0

    @property
    def pdp(self):
        return 0 if self.units == 'mm' else 3

    @property
    def adp(self):
        return 5 if self.angles == 'rad' else 1

    def L(self, metres):
        return _fn(metres * self.scale, self.pdp)

    def Ls(self, metres):
        return _fs(metres * self.scale, self.pdp)

    def A(self, deg):
        return _fn(math.radians(deg) if self.angles == 'rad' else deg, self.adp)

    def As(self, deg):
        return _fs(math.radians(deg) if self.angles == 'rad' else deg, self.adp)

    # machine-readable files keep the angles to 0.0001° (or 1 µrad), not the 0.1° shown on sheets
    def Ap(self, deg):
        return _fn(math.radians(deg) if self.angles == 'rad' else deg, 6 if self.angles == 'rad' else 4)

    def Aps(self, deg):
        return _fs(math.radians(deg) if self.angles == 'rad' else deg, 6 if self.angles == 'rad' else 4)

    @property
    def angle_unit(self):
        return 'rad' if self.angles == 'rad' else '°'

    # -- text
    @property
    def frame_label(self):
        return f'ISO 8855 · {_FRAME_SHORT.get(self.frame, self.frame)}'

    @property
    def origin_text(self):
        if self.frame == 'custom':
            o = self.origin
            return f'custom point ({_fs(o[0], 3)}, {_fs(o[1], 3)}, {_fs(o[2], 3)}) m from the rear axle centre'
        return _ORIGIN_TEXT.get(self.frame, self.frame)

    @property
    def stamp(self):
        return self.now.isoformat(timespec='seconds')

    @property
    def images(self) -> dict:
        return self.opts.images or {}

    @property
    def vehicle_model(self):
        veh = self.project.vehicle
        return veh.spec_name or veh.name or 'Vehicle'

    def measured(self) -> dict:
        if self._measured is None:
            try:
                self._measured = self.project.vehicle.measured() or {}
            except Exception:
                self._measured = {}
        return self._measured

    def issues(self) -> tuple[list, bool]:
        """(issues, computed_here). Falls back to the rule checks that need no mount geometry."""
        if self._issues is None:
            if self.opts.issues is not None:
                self._issues = (list(self.opts.issues), False)
            elif self.variant is self.project.variant:
                try:
                    self._issues = (rules.collect_issues(self.project, self.result, None, None), True)
                except Exception:
                    self._issues = ([], True)
            else:
                self._issues = ([], True)
        return self._issues


def _context(project: Project, result, opts: ExportOptions) -> _Ctx:
    opts = opts or ExportOptions()
    frame = opts.frame or project.frame
    if frame not in FRAME_PRESETS:
        raise ValueError(f'Unknown frame {frame!r}; use one of {", ".join(FRAME_PRESETS)}.')
    if opts.units not in ('m', 'mm'):
        raise ValueError(f"Unknown units {opts.units!r}; use 'm' or 'mm'.")
    if opts.angles not in ('deg', 'rad'):
        raise ValueError(f"Unknown angles {opts.angles!r}; use 'deg' or 'rad'.")
    variant = None
    if result is not None and getattr(result, 'variant_uid', ''):
        variant = project.variant_by_uid(result.variant_uid)
    variant = variant or project.variant
    import copy
    q = copy.copy(project)          # shallow: only the frame differs
    q.frame = frame
    origin = np.asarray(q.origin_offset(), float)
    ctx = _Ctx(project, variant, result, opts, frame, opts.units, opts.angles, origin, _now())
    tags = safe_tags(variant)
    ctx.models = [camera_model(project, c, origin, ctx.scale, tags[c.uid]) for c in variant.cameras]
    return ctx


def _port_parts(port: int):
    if not port or port <= 0:
        return None, None
    return ('A' if port <= 4 else 'B'), ((port - 1) % 4) + 1


def _port_text(port: int) -> str:
    d, link = _port_parts(port)
    return f'{port} · {d}{link}' if d else '—'


# ====================================================================== extrinsics
def _intrinsics_dict(cm: CameraModel) -> dict:
    d = {'model': 'kannala_brandt', 'width': int(cm.intr.width), 'height': int(cm.intr.height),
         'fx': _fn(cm.fx, 4), 'fy': _fn(cm.fy, 4), 'cx': _fn(cm.cx, 4), 'cy': _fn(cm.cy, 4),
         'k': [_sig(k) for k in cm.k]}
    if cm.fitted:
        d['fitted_from'] = cm.intr.projection
        d['fit_rms_px'] = _fn(cm.fit_rms_px, 4)
    cal = cm.cam.calib
    d['source'] = 'unit_calibration' if cal else 'library_lens'
    if cal:
        d['serial'] = cal.get('serial', '')
        d['calibration_date'] = cal.get('date', '')
        d['calibration_rms_px'] = _fn(float(cal.get('rms_px', 0)), 4)
    return d


def _T_world_optical(ctx: _Ctx, cm: CameraModel) -> list:
    O2W = cm.O2W
    rows = [[_fn(O2W[i, j], 6) for j in range(3)] + [_fn(cm.p[i], ctx.pdp)] for i in range(3)]
    return rows + [[0, 0, 0, 1]]


def _rig_parts(ctx: _Ctx) -> list:
    """(frame-safe link name, part) for the rig parts in the analysis (the visible ones)."""
    out, used = [], set()
    for q in getattr(ctx.project, 'rig', ()):
        if not q.visible:
            continue
        base = 'rig_' + _safe(q.name).lower()
        t, k = base, 2
        while t in used:
            t, k = f'{base}_{k}', k + 1
        used.add(t)
        out.append((t, q))
    return out


def _rig_doc(ctx: _Ctx) -> list:
    out = []
    for t, q in _rig_parts(ctx):
        size = {'box': {'length': ctx.L(q.length), 'width': ctx.L(q.width), 'height': ctx.L(q.height)},
                'cylinder': {'radius': ctx.L(q.radius), 'height': ctx.L(q.height)},
                'bar': {'radius': ctx.L(q.radius), 'length': ctx.L(q.length)}}[q.kind]
        out.append({'name': q.name, 'frame': t, 'kind': q.kind,
                    'centre': [ctx.L(q.x - ctx.origin[0]), ctx.L(q.y - ctx.origin[1]), ctx.L(q.z - ctx.origin[2])],
                    'yaw': ctx.Ap(q.yaw), **size, 'occludes': q.occludes, 'mountable': q.mountable})
    return out


def _sensors(ctx: _Ctx) -> list:
    """(safe tag, sensor, model) for the variant's lidars, radars, GNSS antennas and IMUs."""
    tags = sensor_tags(ctx.variant)
    return [(tags[s.uid], s, ctx.project.sensor_model_for(s)) for s in ctx.variant.sensors]


def _T_vehicle_sensor(ctx: _Ctx, s) -> np.ndarray:
    """4x4 mapping points in the sensor's frame (x forward, y left, z up; REP 103) into the export frame, metres."""
    from .sensors import rotation
    T = np.eye(4)
    T[:3, :3] = rotation(s)
    T[:3, 3] = s.pos() - ctx.origin
    return T


def _sensor_field(ctx: _Ctx, m) -> dict:
    if m.is_lidar:
        lo, hi = m.v_fov()
        d = {'scan': m.scan, 'channels': m.n_channels(), 'elevation_min': ctx.Ap(lo), 'elevation_max': ctx.Ap(hi),
             'horizontal_fov': ctx.Ap(m.h_fov), 'azimuth_step': ctx.Ap(m.h_res), 'rate_hz': _fn(m.rate_hz, 3),
             'range_10pct': ctx.L(m.range_m), 'range_min': ctx.L(m.range_min_m), 'points_per_s': int(m.pps())}
        if m.spinning:
            d['rotation'] = 'clockwise' if m.rotation == 'cw' else 'counter-clockwise'
        if m.beams:
            d['beam_elevations'] = [ctx.Ap(b) for b in m.elevations()]
        return d
    if m.kind == 'gnss':
        return {'position': 'phase centre', 'elevation_mask': ctx.Ap(m.mask_deg)}
    if m.kind == 'imu':
        d = {'position': 'centre of the IMU axes', 'rate_hz': _fn(m.rate_hz, 3), 'antenna_inputs': m.antennas}
        if m.heading_sigma_deg > 0:
            d.update(heading_sigma=ctx.Ap(m.heading_sigma_deg), heading_baseline=ctx.L(m.heading_baseline_m))
        return d
    return {'zones': [{'range_car': ctx.L(r), 'azimuth_half': ctx.Ap(a), 'elevation_half': ctx.Ap(e)}
                      for r, a, e in m.zones], 'rate_hz': _fn(m.rate_hz, 3), 'range_min': ctx.L(m.range_min_m)}


def _sensors_doc(ctx: _Ctx) -> list:
    out = []
    for t, s, m in _sensors(ctx):
        T = _T_vehicle_sensor(ctx, s)
        rows = [[_fn(T[i, j], 6) for j in range(3)] + [ctx.L(T[i, 3])] for i in range(3)] + [[0, 0, 0, 1]]
        d = {'tag': s.tag, 'frame': f'{t}_link', 'kind': s.kind, 'name': s.name, 'model': m.name, 'vendor': m.vendor,
             'position': [ctx.L(s.x - ctx.origin[0]), ctx.L(s.y - ctx.origin[1]), ctx.L(s.z - ctx.origin[2])],
             'yaw': ctx.Ap(s.yaw), 'pitch': ctx.Ap(s.pitch), 'roll': ctx.Ap(s.roll),
             'field': _sensor_field(ctx, m), 'T_world_sensor': rows, 'visible': s.visible}
        if m.spinning:
            d['sync_phase'] = ctx.Ap(s.phase_deg)
        if s.port:
            d['port'] = s.port
        out.append(d)
    return out


def _extrinsics_doc(ctx: _Ctx) -> dict:
    meas, spec = ctx.measured(), ctx.project.vehicle.spec or {}

    def dim(key):
        v = meas.get(key) or spec.get(key)
        return ctx.L(v) if v else None

    cams = []
    for cm in ctx.models:
        c = cm.cam
        cams.append({
            'tag': c.tag, 'name': c.location or c.name, 'gmsl_port': c.port or None,
            'module': cm.module.name, 'lens': cm.lens.name,
            'position': [ctx.L(c.x - ctx.origin[0]), ctx.L(c.y - ctx.origin[1]), ctx.L(c.z - ctx.origin[2])],
            'yaw': ctx.Ap(c.yaw), 'pitch': ctx.Ap(c.pitch), 'roll': ctx.Ap(c.roll),
            'intrinsics': _intrinsics_dict(cm),
            'T_world_optical': _T_world_optical(ctx, cm),
        })
    doc = {
        'schema': SCHEMA_EXTRINSICS,
        'project': ctx.project.name,
        'variant': ctx.variant.name,
        'exported': ctx.stamp,
        'frame': {
            'id': ctx.frame,
            'convention': 'ISO 8855',
            'origin': ctx.origin_text,
            'origin_from_rear_axle': [ctx.L(ctx.origin[0]), ctx.L(ctx.origin[1]), ctx.L(ctx.origin[2])],
            'axes': AXES_TEXT,
            'units': ctx.units,
            'angles': ctx.angles,
            'rotation': ROTATION_TEXT,
            'optical_frame': OPTICAL_TEXT,
            'T_world_optical': 'maps optical-frame points into this frame (camera pose); translation in frame units',
        },
        'vehicle': {'model': ctx.vehicle_model, 'length': dim('Length'), 'width': dim('Width'),
                    'wheelbase': dim('Wheelbase')},
        'cameras': cams,
    }
    sensors = _sensors_doc(ctx)
    if sensors:
        doc['frame']['sensor_frame'] = SENSOR_FRAME_TEXT
        doc['sensors'] = sensors    # lidars and radars: position of the optical centre, T_world_sensor its pose
    rig = _rig_doc(ctx)
    if rig:
        doc['rig'] = rig         # roof racks, masts, enclosures: centre in this frame, length along x, width along y
    return doc


_NUM_LIST = re.compile(r'\[\s+((?:-?[0-9][0-9.eE+-]*|null),(?:\s+(?:-?[0-9][0-9.eE+-]*|null),)*\s+(?:-?[0-9][0-9.eE+-]*|null))\s+\]')


def _json(doc) -> str:
    """Indented JSON with numeric vectors and matrix rows kept on one line."""
    text = json.dumps(doc, indent=2, ensure_ascii=False)
    return _NUM_LIST.sub(lambda m: '[' + re.sub(r',\s+', ', ', m.group(1)) + ']', text) + '\n'


def _extrinsics_json(ctx: _Ctx) -> str:
    return _json(_extrinsics_doc(ctx))


def _extrinsics_yaml(ctx: _Ctx) -> str:
    import yaml
    head = (f'# Vantage camera extrinsics ({SCHEMA_EXTRINSICS})\n'
            f'# {ctx.project.name} / {ctx.variant.name}, frame {ctx.frame_label}, {ctx.units}, {ctx.angles}\n')
    body = yaml.safe_dump(_extrinsics_doc(ctx), sort_keys=False, allow_unicode=True, default_flow_style=None,
                          width=120)
    return head + body


def _extrinsics_csv(ctx: _Ctx) -> str:
    u, a = ctx.units, ctx.angles
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator='\n')
    w.writerow(['tag', 'name', 'gmsl_port', 'module', 'lens', f'x_{u}', f'y_{u}', f'z_{u}', f'yaw_{a}', f'pitch_{a}',
                f'roll_{a}', 'width', 'height', 'fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'k3', 'k4', 'frame'])
    for cm in ctx.models:
        c = cm.cam
        w.writerow([c.tag, c.location or c.name, c.port or '', cm.module.name, cm.lens.name,
                    ctx.Ls(c.x - ctx.origin[0]), ctx.Ls(c.y - ctx.origin[1]), ctx.Ls(c.z - ctx.origin[2]),
                    ctx.Aps(c.yaw), ctx.Aps(c.pitch), ctx.Aps(c.roll), cm.intr.width, cm.intr.height,
                    _fs(cm.fx, 4), _fs(cm.fy, 4), _fs(cm.cx, 4), _fs(cm.cy, 4)] + [repr(_sig(k)) for k in cm.k]
                   + [ctx.frame])
    for _, s, m in _sensors(ctx):             # lidars and radars: the model in 'module', the kind in 'lens'
        w.writerow([s.tag, s.name, s.port, m.name, s.kind,
                    ctx.Ls(s.x - ctx.origin[0]), ctx.Ls(s.y - ctx.origin[1]), ctx.Ls(s.z - ctx.origin[2]),
                    ctx.Aps(s.yaw), ctx.Aps(s.pitch), ctx.Aps(s.roll)] + [''] * 10 + [ctx.frame])
    return buf.getvalue()


# ====================================================================== OpenCV FileStorage YAML
def _cvnum(v) -> str:
    v = float(v)
    if abs(v) < 1e-12:
        v = 0.0
    s = f'{v + 0.0:.12g}'
    if s in ('-0',):
        s = '0'
    if not any(ch in s for ch in '.en'):
        s += '.'
    return s


def _cvmat(name: str, M) -> str:
    M = np.atleast_2d(np.asarray(M, float))
    data = ', '.join(_cvnum(v) for v in M.ravel())
    return (f'{name}: !!opencv-matrix\n   rows: {M.shape[0]}\n   cols: {M.shape[1]}\n   dt: d\n'
            f'   data: [ {data} ]\n')


def _cvstr(s: str) -> str:
    return '"' + str(s).replace('\\', '\\\\').replace('"', '\\"').replace('\n', ' ') + '"'


def _opencv_yaml(ctx: _Ctx, cm: CameraModel) -> str:
    c = cm.cam
    lines = [
        '%YAML:1.0',
        '---',
        f'# Vantage fisheye (Kannala-Brandt) calibration for camera {c.tag}, {ctx.project.name} / {ctx.variant.name}',
        f'# World frame: {ctx.frame_label} ({AXES_TEXT}), origin at the {ctx.origin_text}; units {ctx.units}.',
        '# Load with cv::FileStorage; use with cv::fisheye (projectPoints, undistortImage, ...).',
        '# rvec/tvec map world points into the camera (OpenCV optical: x right, y down, z forward):',
        '#   X_cam = R(rvec) X_world + tvec. R_world_camera = R(rvec)^T rotates camera vectors into the world.',
    ]
    if cm.fitted:
        lines.append(f'# Lens projection {cm.intr.projection} fitted to Kannala-Brandt, RMS {cm.fit_rms_px:.3f} px.')
    if c.calib:
        lines.append(f'# Intrinsics: this unit\'s own calibration ({c.calib.get("serial") or "no serial"}, '
                     f'{c.calib.get("date", "")}, RMS {float(c.calib.get("rms_px", 0)):.3f} px), not the library lens.')
    out = '\n'.join(lines) + '\n'
    out += (f'camera_name: {_cvstr(c.tag)}\n'
            f'camera_location: {_cvstr(c.location or c.name)}\n'
            f'module: {_cvstr(cm.module.name)}\n'
            f'lens: {_cvstr(cm.lens.name)}\n'
            + (f'serial: {_cvstr(c.calib.get("serial", ""))}\n' if c.calib and c.calib.get('serial') else '') +
            f'gmsl_port: {int(c.port or 0)}\n'
            f'frame: {_cvstr(ctx.frame)}\n'
            f'units: {_cvstr(ctx.units)}\n'
            f'image_width: {int(cm.intr.width)}\n'
            f'image_height: {int(cm.intr.height)}\n')
    out += _cvmat('camera_matrix', cm.K)
    out += 'distortion_model: fisheye\n'
    out += _cvmat('distortion_coefficients', np.array(cm.k)[None, :])
    out += _cvmat('rvec', cm.rvec[:, None])
    out += _cvmat('tvec', cm.tvec[:, None])
    out += _cvmat('R_world_camera', cm.O2W)
    pos = np.array([ctx.L(c.x - ctx.origin[0]), ctx.L(c.y - ctx.origin[1]), ctx.L(c.z - ctx.origin[2])], float)
    out += _cvmat('camera_position', pos[:, None])
    return out


# ====================================================================== ROS 2
def _xml_comment(s: str) -> str:
    # XML forbids '--' anywhere in a comment and a '-' right before its end
    return re.sub(r'-{2,}', '-', str(s).replace('\n', ' ')).rstrip('-')


def _urdf(ctx: _Ctx) -> str:
    o = ctx.origin
    L = [
        '<?xml version="1.0"?>',
        f'<!-- Vantage camera rig: {_xml_comment(ctx.project.name)} / {_xml_comment(ctx.variant.name)} '
        f'({_xml_comment(ctx.stamp)}) -->',
        f'<!-- base_link origin: {_xml_comment(ctx.origin_text)}. REP 103 axes ({AXES_TEXT}), metres, radians. -->',
        '<!-- <tag>_link: camera body, x along the optical axis. <tag>_optical_frame: x right, y down, z forward. -->',
        '<robot name="vantage_cameras">',
        '  <link name="base_link"/>',
    ]
    for cm in ctx.models:
        c, t = cm.cam, cm.tag
        xyz = ' '.join(_fs(v, 3) for v in (c.x - o[0], c.y - o[1], c.z - o[2]))
        rpy = ' '.join(_fs(math.radians(v), 5) for v in (c.roll, c.pitch, c.yaw))
        L += [
            '',
            f'  <!-- {_xml_comment(c.tag)}: {_xml_comment(c.location or c.name)} -->',
            f'  <link name="{t}_link"/>',
            f'  <joint name="{t}_joint" type="fixed">',
            '    <parent link="base_link"/>',
            f'    <child link="{t}_link"/>',
            f'    <origin xyz="{xyz}" rpy="{rpy}"/>',
            '  </joint>',
            f'  <link name="{t}_optical_frame"/>',
            f'  <joint name="{t}_optical_joint" type="fixed">',
            f'    <parent link="{t}_link"/>',
            f'    <child link="{t}_optical_frame"/>',
            f'    <origin xyz="0 0 0" rpy="{_fs(-math.pi / 2, 5)} 0 {_fs(-math.pi / 2, 5)}"/>',
            '  </joint>',
        ]
    for t, s, m in _sensors(ctx):
        from .sensors import body_box
        _, half, _, rnd = body_box(s, m)
        geom = (f'<cylinder radius="{_fs(half[0], 4)}" length="{_fs(2 * half[2], 4)}"/>' if rnd else
                f'<box size="{_fs(2 * half[0], 4)} {_fs(2 * half[1], 4)} {_fs(2 * half[2], 4)}"/>')
        off = f'0 0 {_fs(half[2] - m.optical_z_mm / 1000.0, 4)}'          # the housing centre above the optical centre
        xyz = ' '.join(_fs(v, 3) for v in (s.x - o[0], s.y - o[1], s.z - o[2]))
        rpy = ' '.join(_fs(math.radians(v), 5) for v in (s.roll, s.pitch, s.yaw))
        L += [
            '',
            f'  <!-- {s.kind} {_xml_comment(s.tag)}: {_xml_comment((m.vendor + " " + m.name).strip())}; frame at the '
            f'optical centre, x forward, y left, z up -->',
            f'  <link name="{t}_link">',
            f'    <visual><origin xyz="{off}" rpy="0 0 0"/><geometry>{geom}</geometry></visual>',
            f'    <collision><origin xyz="{off}" rpy="0 0 0"/><geometry>{geom}</geometry></collision>',
            '  </link>',
            f'  <joint name="{t}_joint" type="fixed">',
            '    <parent link="base_link"/>',
            f'    <child link="{t}_link"/>',
            f'    <origin xyz="{xyz}" rpy="{rpy}"/>',
            '  </joint>',
        ]
    for t, q in _rig_parts(ctx):
        geom = {'box': f'<box size="{_fs(q.length, 4)} {_fs(q.width, 4)} {_fs(q.height, 4)}"/>',
                'cylinder': f'<cylinder radius="{_fs(q.radius, 4)}" length="{_fs(q.height, 4)}"/>',
                'bar': f'<cylinder radius="{_fs(q.radius, 4)}" length="{_fs(q.length, 4)}"/>'}[q.kind]
        turn = f'{_fs(math.pi / 2, 5)} 0 0' if q.kind == 'bar' else '0 0 0'      # a bar runs along the link's y
        xyz = ' '.join(_fs(v, 3) for v in (q.x - o[0], q.y - o[1], q.z - o[2]))
        L += [
            '',
            f'  <!-- rig: {_xml_comment(q.name)} ({q.kind}{", blocks cameras" if q.occludes else ""}) -->',
            f'  <link name="{t}">',
            f'    <visual><origin xyz="0 0 0" rpy="{turn}"/><geometry>{geom}</geometry></visual>',
            f'    <collision><origin xyz="0 0 0" rpy="{turn}"/><geometry>{geom}</geometry></collision>',
            '  </link>',
            f'  <joint name="{t}_joint" type="fixed">',
            '    <parent link="base_link"/>',
            f'    <child link="{t}"/>',
            f'    <origin xyz="{xyz}" rpy="0 0 {_fs(math.radians(q.yaw), 5)}"/>',
            '  </joint>',
        ]
    L.append('</robot>')
    return '\n'.join(L) + '\n'


def _quat(R) -> tuple:
    x, y, z, w = quat_from_matrix(R)
    if w < 0:
        x, y, z, w = -x, -y, -z, -w
    return x, y, z, w


def _launch(ctx: _Ctx) -> str:
    o = ctx.origin
    q_opt = _quat(BODY_TO_OPTICAL.T)          # optical axes expressed in the body frame
    one = lambda s: str(s).replace('\n', ' ')
    L = [
        '# ROS 2 launch file: static TF for the Vantage camera rig.',
        f'# Project: {one(ctx.project.name)} / variant: {one(ctx.variant.name)} / exported {ctx.stamp}',
        f'# base_link origin: {ctx.origin_text}; REP 103 axes ({AXES_TEXT}), metres.',
        '# <tag>_link is the camera body (x along the optical axis); <tag>_optical_frame is x right, y down,',
        '# z forward. Usage: ros2 launch static_tf.launch.py',
        'from launch import LaunchDescription',
        'from launch_ros.actions import Node',
        '',
        '',
        'def _static_tf(name, xyz, quat, parent, child):',
        "    return Node(package='tf2_ros', executable='static_transform_publisher', name=name, output='log',",
        "                arguments=['--x', xyz[0], '--y', xyz[1], '--z', xyz[2],",
        "                           '--qx', quat[0], '--qy', quat[1], '--qz', quat[2], '--qw', quat[3],",
        "                           '--frame-id', parent, '--child-frame-id', child])",
        '',
        '',
        'def generate_launch_description():',
        '    return LaunchDescription([',
    ]
    for cm in ctx.models:
        c, t = cm.cam, cm.tag
        xyz = tuple(_fs(v, 3) for v in (c.x - o[0], c.y - o[1], c.z - o[2]))
        qb = tuple(_fs(v, 6) for v in _quat(cm.R))
        qo = tuple(_fs(v, 6) for v in q_opt)
        node = 'tf_' + re.sub(r'[^a-z0-9_]', '_', t.lower())         # ROS node names: letters, digits and '_'
        L.append(f'        _static_tf({node!r}, {xyz!r}, {qb!r}, {"base_link"!r}, {t + "_link"!r}),')
        L.append(f'        _static_tf({(node + "_optical")!r}, {("0", "0", "0")!r}, {qo!r}, '
                 f'{t + "_link"!r}, {t + "_optical_frame"!r}),')
    for t, s, _ in _sensors(ctx):              # lidars and radars: x forward, y left, z up at the optical centre
        T = _T_vehicle_sensor(ctx, s)
        xyz = tuple(_fs(v, 3) for v in T[:3, 3])
        qs = tuple(_fs(v, 6) for v in _quat(T[:3, :3]))
        node = 'tf_' + re.sub(r'[^a-z0-9_]', '_', t.lower())
        L.append(f'        _static_tf({node!r}, {xyz!r}, {qs!r}, {"base_link"!r}, {t + "_link"!r}),')
    L.append('    ])')
    return '\n'.join(L) + '\n'


# ====================================================================== datasets
# The calibration in the layouts training datasets and their tools read. These files are always in metres (the
# formats' own convention, like URDF/TF); the export frame and origin apply as everywhere else.
DATASET_NOTE = ('Pixel coordinates follow OpenCV and ROS: pixel centres on integers (the centre of a 1920-wide image '
                'is cx = 959.5).')


def _pinhole(cm: CameraModel) -> bool:
    """A rectilinear lens is an exact pinhole (no distortion); every other lens is written as Kannala-Brandt."""
    return cm.intr.projection == 'rectilinear'


def _T_vehicle_camera(ctx: _Ctx, cm: CameraModel) -> np.ndarray:
    """4x4 mapping points in the camera's optical frame (x right, y down, z forward) into the export frame, metres."""
    T = np.eye(4)
    T[:3, :3] = cm.O2W
    T[:3, 3] = cm.cam.pos() - ctx.origin
    return T


def _T_inv(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def _r(v, n=9):
    """A clean float for machine-read files (no -0.0)."""
    x = round(float(v), n) + 0.0
    return 0.0 if x == 0 else x


def _frame_sentence(ctx: _Ctx) -> str:
    return f'{ctx.frame_label} ({AXES_TEXT}), origin at the {ctx.origin_text}; metres'


def _token(*parts) -> str:
    """A stable 32-hex token like nuScenes', so a re-export gives the same tokens."""
    import hashlib
    return hashlib.md5('|'.join(str(x) for x in parts).encode('utf-8')).hexdigest()


def _nuscenes(ctx: _Ctx) -> list:
    sensors, cal = [], []
    p, v = ctx.project, ctx.variant
    for cm in ctx.models:
        st = _token('vantage', 'sensor', p.name, v.name, cm.tag)
        sensors.append({'token': st, 'channel': f'CAM_{cm.tag.upper()}', 'modality': 'camera'})
        T = _T_vehicle_camera(ctx, cm)
        x, y, z, w = _quat(T[:3, :3])
        K = cm.K
        cal.append({
            'token': _token('vantage', 'calibrated_sensor', p.name, v.name, cm.tag),
            'sensor_token': st,
            'translation': [_r(t, 6) for t in T[:3, 3]],
            'rotation': [_r(w, 10), _r(x, 10), _r(y, 10), _r(z, 10)],
            'camera_intrinsic': [[_r(K[i, j], 6) for j in range(3)] for i in range(3)],
            'vantage_distortion': {
                'model': 'none (pinhole)' if _pinhole(cm) else 'kannala_brandt',
                'coefficients': [] if _pinhole(cm) else [_sig(k) for k in cm.k],
                'image_size': [int(cm.intr.width), int(cm.intr.height)],
                'camera': cm.cam.tag, 'name': cm.cam.location or cm.cam.name,
                'source': 'unit_calibration' if cm.cam.calib else 'library_lens',
            },
        })
    for t, s, m in _sensors(ctx):                 # lidars and radars: no intrinsics, as in nuScenes
        if not m.has_field:
            continue                              # GNSS antennas and IMUs are not nuScenes sensors (ego_pose is theirs)
        st = _token('vantage', 'sensor', p.name, v.name, t)
        sensors.append({'token': st, 'channel': f'{s.kind.upper()}_{t.upper()}', 'modality': s.kind})
        T = _T_vehicle_sensor(ctx, s)
        x, y, z, w = _quat(T[:3, :3])
        cal.append({'token': _token('vantage', 'calibrated_sensor', p.name, v.name, t), 'sensor_token': st,
                    'translation': [_r(q, 6) for q in T[:3, 3]],
                    'rotation': [_r(w, 10), _r(x, 10), _r(y, 10), _r(z, 10)], 'camera_intrinsic': [],
                    'vantage_sensor': {'tag': s.tag, 'model': f'{m.vendor} {m.name}'.strip(), 'kind': s.kind}})
    head = f'Vantage {p.name} / {v.name}, exported {ctx.stamp}'
    rah = ''
    wc = p.vehicle.wheel_centres or {}
    if 'RL' in wc:
        rah = (f' The rear axle is {float(wc["RL"][2]):.3f} m above the ground here: subtract that from z if your ego '
               f'frame is at axle height.')
    readme = f"""# nuScenes calibration · {head}

`sensor.json` and `calibrated_sensor.json` follow the nuScenes tables of the same names, one camera per row
(channel `CAM_<TAG>`), then the lidars and radars (channels `LIDAR_<TAG>`, `RADAR_<TAG>`; `camera_intrinsic` empty,
as nuScenes has it for them).

- **translation**: the camera's position in the ego frame, metres. Ego frame here: {_frame_sentence(ctx)}.
  nuScenes puts its ego frame at the rear axle with x forward, y left, z up, which is this export's default
  (rear-axle) frame.{rah}
- **rotation**: quaternion `[w, x, y, z]` rotating the camera frame into the ego frame, as nuScenes does:
  `p_ego = R(rotation) · p_camera + translation`. The camera frame is the optical frame: x right, y down, z forward.
  A lidar's or radar's frame is x forward, y left, z up at its optical centre (REP 103). Drivers differ (nuScenes'
  own Velodyne frame has x to the right, an Ouster's lidar frame has x towards the connector): if your point clouds
  are in the driver's frame, add the driver's fixed rotation.
- **camera_intrinsic**: the 3×3 pinhole matrix. {DATASET_NOTE}
- **vantage_distortion** (an extra field the nuScenes tools ignore): these are fisheye cameras, Kannala-Brandt
  (`cv2.fisheye`, k1..k4). nuScenes assumes pinhole images, so undistort the images with these coefficients and
  `camera_intrinsic` (for example `cv2.fisheye.undistortImage(img, K, D, Knew=K)`) before using the 3×3 alone.
  A rectilinear lens has no distortion and says `none (pinhole)`.

Tokens are stable hashes of the project, variant and camera, so a re-export keeps them.
"""
    return [('nuscenes/calibrated_sensor.json', _json(cal).encode('utf-8')),
            ('nuscenes/sensor.json', _json(sensors).encode('utf-8')),
            ('nuscenes/README.md', readme.encode('utf-8'))]


def _ke(vals) -> str:
    """KITTI's number style; rotation round-off (1e-16) is written as 0."""
    return ' '.join(f'{(float(v) if abs(v) > 1e-12 else 0.0):.6e}' for v in np.asarray(vals, float).ravel())


def _kitti(ctx: _Ctx) -> list:
    cams = ctx.models
    when = ctx.now.strftime('%d-%b-%Y %H:%M:%S')
    square = float(getattr(ctx.project.target_settings, 'square_mm', 100.0)) / 1000.0
    names = ' '.join(cm.tag for cm in cams)
    # KITTI readers parse "key: numbers" lines and skip values that are not numbers, so the notes are such lines
    a = [f'calib_time: {when}', f'corner_dist: {_ke([square])}', f'camera_names: {names}',
         'note: cameras are numbered 00.. in camera_names order; R_xx/T_xx map camera 00 coordinates into camera xx '
         '(X_xx = R_xx X_00 + T_xx), metres; images are not rectified, so S_rect = S, R_rect = I and P_rect = [K|0]; '
         'D_xx holds Kannala-Brandt k1 k2 k3 k4 (then 0) where DISTORTION_MODEL_xx is equidistant - standard KITTI '
         'tools assume radtan (k1 k2 p1 p2 k3)']
    b = [f'calib_time: {when}', f'camera_names: {names}',
         f'note: Tr_cam_xx_to_vehicle is the 3x4 [R|t] (row-major) mapping camera xx optical coordinates (x right, '
         f'y down, z forward) into the vehicle frame: {_frame_sentence(ctx)}']
    T0 = _T_vehicle_camera(ctx, cams[0]) if cams else np.eye(4)
    for i, cm in enumerate(cams):
        n = f'{i:02d}'
        T = _T_vehicle_camera(ctx, cm)
        rel = _T_inv(T) @ T0                       # camera 00 coordinates -> camera n
        K = cm.K
        D = [0.0] * 5 if _pinhole(cm) else list(cm.k) + [0.0]
        size = [cm.intr.width, cm.intr.height]
        a += [f'S_{n}: {_ke(size)}', f'K_{n}: {_ke(K)}', f'D_{n}: {_ke(D)}',
              f'DISTORTION_MODEL_{n}: {"plumb_bob" if _pinhole(cm) else "equidistant"}',
              f'R_{n}: {_ke(rel[:3, :3])}', f'T_{n}: {_ke(rel[:3, 3])}',
              f'S_rect_{n}: {_ke(size)}', f'R_rect_{n}: {_ke(np.eye(3))}',
              f'P_rect_{n}: {_ke(np.c_[K, np.zeros(3)])}']
        b.append(f'Tr_cam_{n}_to_vehicle: {_ke(T[:3, :4])}')
    sens = _sensors(ctx)
    for t, s, _ in sens:
        b.append(f'Tr_{t}_to_vehicle: {_ke(_T_vehicle_sensor(ctx, s)[:3, :4])}')
    lidars = [(t, s) for t, s, _ in sens if s.kind == 'lidar']
    imus = [(t, s) for t, s, _ in sens if s.kind == 'imu']
    imu_velo = None
    if lidars and imus:
        (lt, ls), (it, is_) = lidars[0], imus[0]
        rel = _T_inv(_T_vehicle_sensor(ctx, ls)) @ _T_vehicle_sensor(ctx, is_)     # IMU coordinates -> lidar
        imu_velo = [f'calib_time: {when}', f'R: {_ke(rel[:3, :3])}', f'T: {_ke(rel[:3, 3])}',
                    f'note: IMU {is_.tag} ({it}_link: its own axes) into lidar {ls.tag} ({lt}_link: x forward, y left, '
                    f'z up): X_lidar = R X_imu + T, metres']
    velo = None
    if lidars and cams:
        t, s = lidars[0]
        rel = _T_inv(T0) @ _T_vehicle_sensor(ctx, s)          # lidar coordinates -> camera 00
        velo = [f'calib_time: {when}', f'R: {_ke(rel[:3, :3])}', f'T: {_ke(rel[:3, 3])}',
                f'delta_f: {_ke([0, 0])}', f'delta_c: {_ke([0, 0])}',
                f'note: lidar {s.tag} ({t}_link: x forward, y left, z up) into camera 00 {cams[0].cam.tag} '
                f'(x right, y down, z forward): X_cam = R X_lidar + T, metres']
    order = '\n'.join(f'  {i:02d}  {cm.cam.tag:<6} {cm.cam.location or cm.cam.name}' for i, cm in enumerate(cams))
    readme = f"""KITTI-style calibration · Vantage {ctx.project.name} / {ctx.variant.name}, exported {ctx.stamp}

calib_cam_to_cam.txt follows the KITTI raw-data layout: calib_time, corner_dist, and per camera S, K, D, R, T,
S_rect, R_rect, P_rect. calib_cams_to_vehicle.txt gives every camera's pose in the vehicle frame.

Cameras:
{order or '  (none)'}

- R_xx, T_xx map camera 00 coordinates into camera xx: X_xx = R_xx X_00 + T_xx (metres), as in KITTI.
- Tr_cam_xx_to_vehicle maps camera xx coordinates into the vehicle frame ({_frame_sentence(ctx)}).
- Camera frames are optical: x right, y down, z forward. {DATASET_NOTE}
- These are fisheye cameras. KITTI's D is a radtan vector (k1 k2 p1 p2 k3); here D_xx holds the Kannala-Brandt
  coefficients k1 k2 k3 k4 followed by 0, and DISTORTION_MODEL_xx says equidistant. Tools that assume radtan will
  misread them: undistort with cv2.fisheye first. A rectilinear lens has D = 0 and plumb_bob.
- The images are not rectified: S_rect = S, R_rect = I, P_rect = [K | 0] (no stereo baseline term).
- corner_dist is the floor calibration board's square ({square:.3f} m).
- Lidars and radars: Tr_<tag>_to_vehicle in calib_cams_to_vehicle.txt maps the sensor's frame (x forward, y left,
  z up at its optical centre) into the vehicle frame. With a lidar, calib_velo_to_cam.txt gives the first lidar's
  frame in camera 00 (R, T as KITTI's velodyne-to-camera file).
- GNSS antennas and IMUs: Tr_<tag>_to_vehicle as well (an antenna's frame at its phase centre, an IMU's at the centre
  of its axes). With an IMU and a lidar, calib_imu_to_velo.txt gives the first IMU's frame in the first lidar's
  (R, T as KITTI's imu-to-velodyne file; KITTI's OXTS IMU frame is x forward, y left, z up).
- The extra lines (camera_names, note, DISTORTION_MODEL) are not numbers, so KITTI readers such as pykitti skip them.
"""
    out = [('kitti/calib_cam_to_cam.txt', ('\n'.join(a) + '\n').encode('utf-8')),
           ('kitti/calib_cams_to_vehicle.txt', ('\n'.join(b) + '\n').encode('utf-8')),
           ('kitti/README.txt', readme.encode('utf-8'))]
    if velo:
        out.append(('kitti/calib_velo_to_cam.txt', ('\n'.join(velo) + '\n').encode('utf-8')))
    if imu_velo:
        out.append(('kitti/calib_imu_to_velo.txt', ('\n'.join(imu_velo) + '\n').encode('utf-8')))
    return out


def _yaml_mat(name: str, M) -> str:
    M = np.atleast_2d(np.asarray(M, float))
    data = ', '.join(repr(_r(v, 10)) for v in M.ravel())
    return f'{name}:\n  rows: {M.shape[0]}\n  cols: {M.shape[1]}\n  data: [{data}]\n'


def _camera_info(ctx: _Ctx, cm: CameraModel) -> str:
    """ROS sensor_msgs/CameraInfo as camera_calibration_parsers writes it."""
    c = cm.cam
    K = cm.K
    pin = _pinhole(cm)
    D = [0.0] * 5 if pin else list(cm.k)
    lens = "Pinhole lens: plumb_bob with no distortion." if pin else \
        "Fisheye lens: equidistant = Kannala-Brandt (cv2.fisheye) k1..k4."
    src = "this unit's own calibration" if c.calib else f'the library lens {cm.lens.name}'
    out = (f'# ROS camera_info for camera {c.tag} ({c.location or c.name}), Vantage {ctx.project.name} / '
           f'{ctx.variant.name}, exported {ctx.stamp}\n'
           f'# {lens} {DATASET_NOTE}\n'
           f'# Intrinsics: {src}.\n'
           f'image_width: {int(cm.intr.width)}\n'
           f'image_height: {int(cm.intr.height)}\n'
           f'camera_name: {cm.tag}\n')
    out += _yaml_mat('camera_matrix', K)
    out += f'distortion_model: {"plumb_bob" if pin else "equidistant"}\n'
    out += _yaml_mat('distortion_coefficients', np.array(D)[None, :])
    out += _yaml_mat('rectification_matrix', np.eye(3))
    out += _yaml_mat('projection_matrix', np.c_[K, np.zeros(3)])
    return out


def _kalibr(ctx: _Ctx) -> str:
    import yaml
    cams = ctx.models
    index = {cm.cam.uid: i for i, cm in enumerate(cams)}
    doc, prev = {}, None
    for i, cm in enumerate(cams):
        T = _T_vehicle_camera(ctx, cm)
        pin = _pinhole(cm)
        d = {'camera_model': 'pinhole',
             'intrinsics': [_r(cm.fx, 6), _r(cm.fy, 6), _r(cm.cx, 6), _r(cm.cy, 6)],
             'distortion_model': 'radtan' if pin else 'equidistant',
             'distortion_coeffs': [0.0] * 4 if pin else [_sig(k) for k in cm.k],
             'resolution': [int(cm.intr.width), int(cm.intr.height)],
             'rostopic': f'/{cm.tag.lower()}/image_raw'}
        if ctx.result is not None:
            ov = ctx.result.per_cam.get(cm.cam.uid, {}).get('overlaps', {})
            d['cam_overlaps'] = sorted(index[u] for u, a in ov.items() if u in index and a > 0.05)
        if prev is not None:
            rel = _T_inv(T) @ prev                 # previous camera's coordinates -> this camera
            d['T_cn_cnm1'] = [[_r(rel[r, c], 10) for c in range(4)] for r in range(4)]
        doc[f'cam{i}'] = d
        prev = T
    order = ', '.join(f'cam{i} = {cm.cam.tag}' for i, cm in enumerate(cams)) or 'none'
    head = (f'# Kalibr camera chain · Vantage {ctx.project.name} / {ctx.variant.name}, exported {ctx.stamp}\n'
            f'# {order}. T_cn_cnm1 maps points from the previous camera into this one (p_n = T p_n-1), metres;\n'
            f'# camera frames are optical (x right, y down, z forward). intrinsics are [fu, fv, pu, pv].\n'
            f'# {DATASET_NOTE}\n'
            f'# Poses in the vehicle frame ({_frame_sentence(ctx)}): see the extrinsics export.\n')
    return head + (yaml.safe_dump(doc, sort_keys=False, default_flow_style=None, width=200) if doc else '{}\n')


# ====================================================================== Jetson
def _jetson_json(ctx: _Ctx) -> str:
    p = ctx.project
    cams = []
    for cm in sorted(ctx.models, key=lambda m: (not m.cam.port, m.cam.port or 0, m.cam.tag)):
        c = cm.cam
        d, link = _port_parts(c.port)
        rl = rules.route_length(p, c)
        cams.append({
            'port': c.port or None, 'deserializer': d, 'link': link,
            'link_label': p.ecu.deserializer(c.port) if c.port else None,
            'tag': c.tag, 'name': c.location or c.name,
            'module': cm.module.name, 'sensor': cm.module.sensor, 'interface': cm.module.interface,
            'lens': cm.lens.name,
            'resolution': {'width': int(cm.module.width_px), 'height': int(cm.module.height_px)},
            'fps': _fn(cm.module.fps, 2), 'sync_group': c.sync, 'hw_sync': bool(cm.module.hw_sync),
            'device': f'/dev/video{c.port - 1}' if c.port else None,
            'cable_m': _fn(c.cable_m, 2), 'routed_m': _fn(rl, 2), 'cable_ok': bool(rl <= c.cable_m + 1e-6),
        })
    doc = {
        'schema': SCHEMA_JETSON, 'platform': PLATFORM, 'project': p.name, 'variant': ctx.variant.name,
        'exported': ctx.stamp,
        'ecu': {'name': p.ecu.name, 'location': p.ecu.location, 'ports': int(p.ecu.ports)},
        'deserializers': [{'id': 'A', 'ports': [1, 2, 3, 4]}, {'id': 'B', 'ports': [5, 6, 7, 8]}],
        'notes': ['device is a guess: assumes /dev/videoN enumerates in GMSL port order (port N -> /dev/video<N-1>).',
                  'routed_m is the planned cable route including the service loop.'],
        'cameras': cams,
    }
    return _json(doc)


# ====================================================================== DXF (R12)
def _outline_polys(ctx: _Ctx) -> list:
    res = ctx.result
    if res is None or getattr(res, 'grid', None) is None:
        return []
    out = []
    for pl in res.grid.outline:
        a = np.asarray(pl, float)
        if len(a) >= 2:
            out.append(a[:, :2])
    return out


def _dxf(ctx: _Ctx) -> str:
    s, o, dp = ctx.scale, ctx.origin, ctx.pdp
    out = []

    def g(code, val):
        out.append(f'{code:>3}\n{val}\n')

    def num(v):
        return _fs(v, dp) if dp else _fs(v, 0)

    ents = []          # (kind, data) to compute extents first
    pts = []
    for cm in ctx.models:
        c = cm.cam
        p0 = (c.pos() - o) * s
        p1 = p0 + cm.aim * AIM_LENGTH_M * s
        tp = p0 + np.array([0.04, 0.04, 0.0]) * s
        pts += [p0, p1]
        ents.append(('cam', cm, p0, p1, tp))
    polys = []
    zg = -o[2] * s
    for a in _outline_polys(ctx):
        q = np.column_stack([(a[:, 0] - o[0]) * s, (a[:, 1] - o[1]) * s, np.full(len(a), zg)])
        closed = len(q) > 2 and np.allclose(q[0], q[-1])
        if closed:
            q = q[:-1]
        polys.append((q, closed))
        pts += list(q)
    P = np.array(pts) if pts else np.zeros((1, 3))
    lo, hi = P.min(axis=0), P.max(axis=0)

    g(0, 'SECTION'); g(2, 'HEADER')
    g(9, '$ACADVER'); g(1, 'AC1009')
    g(9, '$DWGCODEPAGE'); g(3, 'ANSI_1252')          # R12 text is 8-bit; the file is written in cp1252
    g(9, '$INSUNITS'); g(70, 4 if ctx.units == 'mm' else 6)
    g(9, '$MEASUREMENT'); g(70, 1)
    g(9, '$EXTMIN'); g(10, num(lo[0])); g(20, num(lo[1])); g(30, num(lo[2]))
    g(9, '$EXTMAX'); g(10, num(hi[0])); g(20, num(hi[1])); g(30, num(hi[2]))
    g(0, 'ENDSEC')

    g(0, 'SECTION'); g(2, 'TABLES')
    g(0, 'TABLE'); g(2, 'LTYPE'); g(70, 1)
    g(0, 'LTYPE'); g(2, 'CONTINUOUS'); g(70, 0); g(3, 'Solid line'); g(72, 65); g(73, 0); g(40, '0.0')
    g(0, 'ENDTAB')
    layers = [('CAMERAS', 7), ('AIM', 1), ('LABELS', 7), ('OUTLINE', 8)]
    g(0, 'TABLE'); g(2, 'LAYER'); g(70, len(layers))
    for name, col in layers:
        g(0, 'LAYER'); g(2, name); g(70, 0); g(62, col); g(6, 'CONTINUOUS')
    g(0, 'ENDTAB')
    g(0, 'TABLE'); g(2, 'STYLE'); g(70, 1)
    g(0, 'STYLE'); g(2, 'STANDARD'); g(70, 0); g(40, '0.0'); g(41, '1.0'); g(50, '0.0'); g(71, 0); g(42, '0.2')
    g(3, 'txt'); g(4, '')
    g(0, 'ENDTAB')
    g(0, 'ENDSEC')

    g(0, 'SECTION'); g(2, 'ENTITIES')
    th = 0.08 * s
    for _, cm, p0, p1, tp in ents:
        col = CAM_ACI[cm.cam.slot % len(CAM_ACI)]
        g(0, 'POINT'); g(8, 'CAMERAS'); g(62, col); g(10, num(p0[0])); g(20, num(p0[1])); g(30, num(p0[2]))
        g(0, 'LINE'); g(8, 'AIM'); g(62, col)
        g(10, num(p0[0])); g(20, num(p0[1])); g(30, num(p0[2]))
        g(11, num(p1[0])); g(21, num(p1[1])); g(31, num(p1[2]))
        g(0, 'TEXT'); g(8, 'LABELS'); g(62, col)
        g(10, num(tp[0])); g(20, num(tp[1])); g(30, num(tp[2])); g(40, num(th) if dp == 0 else _fs(th, 3))
        g(1, cm.cam.tag.replace('\n', ' '))
    for q, closed in polys:
        g(0, 'POLYLINE'); g(8, 'OUTLINE'); g(66, 1); g(10, '0.0'); g(20, '0.0'); g(30, num(zg)); g(70, 1 if closed else 0)
        for v in q:
            g(0, 'VERTEX'); g(8, 'OUTLINE'); g(10, num(v[0])); g(20, num(v[1])); g(30, num(v[2]))
        g(0, 'SEQEND'); g(8, 'OUTLINE')
    g(0, 'ENDSEC')
    g(0, 'EOF')
    return ''.join(out)


# ====================================================================== STEP (AP214)
def _step_str(s: str) -> str:
    out, ext = [], []

    def flush():
        if ext:
            out.append('\\X2\\' + ''.join(f'{ord(ch):04X}' for ch in ext) + '\\X0\\')
            ext.clear()

    for ch in str(s):
        o = ord(ch)
        if o > 0xFFFF:
            ch, o = '?', ord('?')
        if 32 <= o <= 126:
            flush()
            out.append("''" if ch == "'" else ('\\\\' if ch == '\\' else ch))
        else:
            ext.append(' ' if o < 32 else ch)
            if o < 32:
                flush()
    flush()
    return "'" + ''.join(out) + "'"


def _step_real(v, dp: int) -> str:
    s = _fs(v, dp)
    if '.' not in s:
        s += '.'
    return s


def _step(ctx: _Ctx) -> str:
    rows = []

    def add(text):
        rows.append(text)
        return f'#{len(rows)}'

    dp, s, o = ctx.pdp, ctx.scale, ctx.origin
    pt = lambda name, v: add(f"CARTESIAN_POINT({_step_str(name)},({','.join(_step_real(x, dp) for x in v)}))")
    dr = lambda name, v: add(f"DIRECTION({_step_str(name)},({','.join(_step_real(x, 6) for x in v)}))")

    app = add("APPLICATION_CONTEXT('core data for automotive mechanical design processes')")
    add(f"APPLICATION_PROTOCOL_DEFINITION('international standard','automotive_design',2000,{app})")
    pctx = add(f"PRODUCT_CONTEXT('',{app},'mechanical')")
    prod = add(f"PRODUCT('vantage_camera_mounts',{_step_str('Camera mount points')},"
               f"{_step_str(ctx.project.name + ' / ' + ctx.variant.name)},({pctx}))")
    add(f"PRODUCT_RELATED_PRODUCT_CATEGORY('part',$,({prod}))")
    pdf = add(f"PRODUCT_DEFINITION_FORMATION('','',{prod})")
    pdc = add(f"PRODUCT_DEFINITION_CONTEXT('part definition',{app},'design')")
    pd = add(f"PRODUCT_DEFINITION('design','',{pdf},{pdc})")
    pds = add(f"PRODUCT_DEFINITION_SHAPE('','',{pd})")
    if ctx.units == 'mm':
        lu = add('(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT(.MILLI.,.METRE.))')
    else:
        lu = add('(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT($,.METRE.))')
    au = add('(NAMED_UNIT(*)PLANE_ANGLE_UNIT()SI_UNIT($,.RADIAN.))')
    su = add('(NAMED_UNIT(*)SI_UNIT($,.STERADIAN.)SOLID_ANGLE_UNIT())')
    unc = add(f"UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE({'1.E-03' if ctx.units == 'mm' else '1.E-06'}),{lu},"
              "'distance_accuracy_value','confusion accuracy')")
    gctx = add(f"(GEOMETRIC_REPRESENTATION_CONTEXT(3)GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT(({unc}))"
               f"GLOBAL_UNIT_ASSIGNED_CONTEXT(({lu},{au},{su}))"
               "REPRESENTATION_CONTEXT('Context #1','3D Context with UNIT and UNCERTAINTY'))")
    o_pt = pt('origin', (0.0, 0.0, 0.0))
    o_z = dr('', (0.0, 0.0, 1.0))
    o_x = dr('', (1.0, 0.0, 0.0))
    o_ax = add(f"AXIS2_PLACEMENT_3D({_step_str(ctx.frame_label)},{o_pt},{o_z},{o_x})")
    members, places = [], []
    for cm in ctx.models:
        c = cm.cam
        p0 = (c.pos() - o) * s
        p1 = p0 + cm.aim * AIM_LENGTH_M * s
        cp = pt(c.tag, p0)
        z = dr(f'{c.tag} optical axis', cm.aim)
        x = dr(f'{c.tag} image right', -cm.R[:, 1])
        places.append(add(f"AXIS2_PLACEMENT_3D({_step_str(c.tag)},{cp},{z},{x})"))
        ep = pt(f'{c.tag} aim', p1)
        members += [cp, add(f"POLYLINE({_step_str(c.tag + ' aim')},({cp},{ep}))")]
    items = [o_ax]
    if members:
        items.append(add(f"GEOMETRIC_CURVE_SET('camera mount points',({','.join(members)}))"))
    items += places
    rep = add(f"GEOMETRICALLY_BOUNDED_WIREFRAME_SHAPE_REPRESENTATION('camera mounts',({','.join(items)}),{gctx})")
    add(f"SHAPE_DEFINITION_REPRESENTATION({pds},{rep})")

    head = [
        'ISO-10303-21;',
        'HEADER;',
        f"FILE_DESCRIPTION(({_step_str('Vantage camera mount points')},"
        f"{_step_str('frame ' + ctx.frame_label + '; ' + AXES_TEXT + '; z axis = optical axis, x = image right')}),"
        "'2;1');",
        f"FILE_NAME('mount-points.step',{_step_str(ctx.stamp)},('Vantage'),('Vantage'),'Vantage','Vantage','');",
        "FILE_SCHEMA(('AUTOMOTIVE_DESIGN { 1 0 10303 214 1 1 1 1 }'));",
        'ENDSEC;',
        'DATA;',
    ]
    body = [f'#{i + 1}={r};' for i, r in enumerate(rows)]
    return '\n'.join(head + body + ['ENDSEC;', 'END-ISO-10303-21;']) + '\n'


# ====================================================================== shared document text
def _tolerance_lines(ctx: _Ctx) -> list:
    a = ctx.project.analysis
    lines = [f'Install each camera within ±{a.tol_deg:.1f}° of the yaw, pitch and roll listed and within '
             f'±{a.tol_mm:.0f} mm of the position. Values are nominal; calibrate the rig after installation.']
    ws = ctx.opts.worst_single
    if ws:
        blind, uid, pert = (tuple(ws) + (None, None, None))[:3]
        cam = ctx.variant.cam(uid) if uid else None
        if not blind or blind <= 0.0005:
            lines.append('No single camera error within this tolerance opens blind ground in the ROI.')
        elif cam is None:
            lines.append(f'Worst single-camera case leaves {blind:.3f} m² of the ROI blind.')
        else:
            lines.append(f'Worst single-camera case: {cam.tag} {_describe_pert(pert)} leaves {blind:.3f} m² '
                         'of the ROI blind.')
    return lines


def _describe_pert(d) -> str:
    """(dyaw, dpitch, droll, dx, dy, dz) -> 'off by yaw +5.0°, pitch -5.0°' / 'moved 100 mm along the body'."""
    if not d:
        return 'off nominal'
    d = [float(x) for x in d] + [0.0] * max(0, 6 - len(d))
    rot = [f'{name} {"+" if val > 0 else "-"}{abs(val):.1f}°'
           for name, val in zip(('yaw', 'pitch', 'roll'), d[:3]) if abs(val) > 1e-9]
    shift = float(np.linalg.norm(d[3:6]))
    parts = []
    if rot:
        parts.append('off by ' + ', '.join(rot))
    if shift > 1e-9:
        parts.append(f'moved {shift * 1000:.0f} mm along the body')
    return ' and '.join(parts) if parts else 'off nominal'


def _conventions_text(ctx: _Ctx) -> str:
    unit = 'mm' if ctx.units == 'mm' else 'm'
    ang = 'radians' if ctx.angles == 'rad' else 'degrees'
    return (f'Positions: camera optical centre in the {ctx.frame_label} frame ({AXES_TEXT}), origin at the '
            f'{ctx.origin_text}, in {unit}. Angles: {ROTATION_TEXT}, in {ang}; yaw 0 looks forward, positive '
            'turns left.')


def _camera_rows(ctx: _Ctx) -> list:
    rows = []
    for cm in ctx.models:
        c = cm.cam
        rl = rules.route_length(ctx.project, c)
        rows.append({
            'cam': c, 'tag': c.tag, 'slot': c.slot, 'location': c.location or c.name,
            'x': ctx.Ls(c.x - ctx.origin[0]), 'y': ctx.Ls(c.y - ctx.origin[1]), 'z': ctx.Ls(c.z - ctx.origin[2]),
            'yaw': ctx.As(c.yaw), 'pitch': ctx.As(c.pitch), 'roll': ctx.As(c.roll),
            'module': cm.module.name, 'lens': cm.lens.name + (f' ({cm.lens.label})' if cm.lens.label else ''),
            'port': _port_text(c.port), 'cable': _fs(c.cable_m, 1), 'routed': _fs(rl, 2),
            'over': rl > c.cable_m + 1e-6, 'standoff': _fs(c.standoff_mm, 0), 'bracket': c.bracket,
        })
    return rows


def _png_size(data: bytes):
    if len(data) >= 24 and data[:8] == b'\x89PNG\r\n\x1a\n' and data[12:16] == b'IHDR':
        return struct.unpack('>II', data[16:24])
    return None


def _image_list(ctx: _Ctx) -> list:
    """[(key, caption, png bytes)] in report order."""
    imgs = ctx.images
    out = []
    for key, cap in (('viewport', '3D viewport'), ('bev', "Bird's-eye view"), ('coverage', 'Coverage map')):
        if imgs.get(key):
            out.append((key, cap, imgs[key]))
    for c in ctx.variant.cameras:
        data = imgs.get(f'cam:{c.uid}')
        if data:
            out.append((f'cam:{c.uid}', f'{c.tag} · {c.location or c.name}', data))
    return out


# ====================================================================== report (HTML, shared with the PDF)
_CSS_WEB = """
:root { color-scheme: light; --ink:#1b1b1a; --ink2:#52514e; --muted:#898781; --hair:#e1e0d9; --tint:#f6f5f2; }
* { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; }
body { margin: 0; background: #ffffff; color: var(--ink);
       font: 14px/1.45 "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; }
main { max-width: 1080px; margin: 0 auto; padding: 32px 16px 48px; }
.kicker { margin: 0 0 4px; color: var(--muted); font-size: 12px; letter-spacing: .06em; text-transform: uppercase; }
h1 { font-size: 26px; line-height: 1.2; margin: 0 0 4px; font-weight: 600; }
h2 { font-size: 17px; margin: 36px 0 12px; font-weight: 600; padding-bottom: 6px; border-bottom: 1px solid var(--hair); }
.sub { color: var(--ink2); margin: 0 0 4px; }
.note { color: var(--ink2); margin: 8px 0 0; }
.muted { color: var(--muted); }
.wrap { overflow-x: auto; -webkit-overflow-scrolling: touch; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { padding: 6px 8px; border-bottom: 1px solid var(--hair); text-align: left; vertical-align: top; }
th { font-weight: 600; color: var(--ink2); background: var(--tint); font-size: 12px; white-space: nowrap; }
td.n, th.n { text-align: right; white-space: nowrap; font-variant-numeric: tabular-nums;
             font-family: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12.5px; }
th.n { font-family: inherit; font-size: 12px; }
td.warn { color: #b8312f; font-weight: 600; }
table.meta { width: auto; margin-top: 16px; }
table.meta th { background: none; padding: 3px 16px 3px 0; border: 0; font-weight: 500; color: var(--muted); }
table.meta td { padding: 3px 24px 3px 0; border: 0; }
.tiles { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }
.tile { border: 1px solid var(--hair); border-radius: 8px; padding: 12px 14px; }
.tile .v { font-size: 22px; font-weight: 600; font-variant-numeric: tabular-nums; }
.tile .l { color: var(--ink2); font-size: 12px; }
.sw { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px; vertical-align: -1px; }
.pill { display: inline-block; min-width: 72px; text-align: center; font-size: 11px; font-weight: 600; letter-spacing: .04em;
        padding: 1px 8px; border-radius: 9px; }
ul.issues { list-style: none; padding: 0; margin: 0; }
ul.issues li { display: flex; gap: 12px; align-items: baseline; padding: 8px 0; border-bottom: 1px solid var(--hair); }
ul.issues .obj { color: var(--muted); font-size: 12px; }
figure { margin: 0 0 16px; }
figure img { width: 100%; height: auto; display: block; border: 1px solid var(--hair); border-radius: 6px; background: var(--tint); }
figcaption { color: var(--ink2); font-size: 12px; margin-top: 6px; }
.views { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(100%, 300px), 1fr)); gap: 12px; }
.views.c2 { grid-template-columns: repeat(2, minmax(0, 1fr)); }
.views.c3 { grid-template-columns: repeat(3, minmax(0, 1fr)); }
td.tag { white-space: nowrap; }
@media (max-width: 720px) {
  main { padding: 20px 16px 32px; }
  .tiles { grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }
  .tile .v { font-size: 18px; }
  .views.c2, .views.c3 { grid-template-columns: minmax(0, 1fr); }
}
h3 { font-size: 14px; margin: 20px 0 8px; font-weight: 600; }
td.barcell { width: 45%; min-width: 120px; }
.bar { height: 10px; border-radius: 2px; min-width: 1px; }
footer { margin-top: 40px; color: var(--muted); font-size: 12px; }
@media print { main { max-width: none; padding: 0; } h2 { break-after: avoid; } figure, tr, li { break-inside: avoid; } }
"""

_CSS_QT = """
body { font-family: 'IBM Plex Sans'; font-size: 9pt; color: #1b1b1a; }
h1 { font-size: 18pt; font-weight: 600; margin-top: 0px; margin-bottom: 2px; }
h2 { font-size: 11.5pt; font-weight: 600; margin-top: 18px; margin-bottom: 6px; }
p { margin-top: 0px; margin-bottom: 3px; }
.kicker { color: #898781; font-size: 7.5pt; }
.sub { color: #52514e; }
.note { color: #52514e; }
.muted { color: #898781; }
table.t { border-collapse: collapse; }
table.t th { background-color: #f3f2ef; color: #52514e; font-weight: 600; font-size: 7.5pt; text-align: left;
             border-bottom: 1px solid #c3c2b7; }
table.t td { font-size: 8pt; border-bottom: 1px solid #e6e5df; }
td.n { font-family: 'IBM Plex Mono'; font-size: 7.5pt; text-align: right; }
th.n { text-align: right; }
td.warn { color: #b8312f; font-weight: 600; }
table.meta td { font-size: 8.5pt; }
td.ml { color: #898781; }
td.tv { font-size: 15pt; font-weight: 600; }
td.tl { color: #52514e; font-size: 7.5pt; }
.obj { color: #898781; font-size: 7.5pt; }
"""


def _report_html(ctx: _Ctx, qt: bool = False, img_src: Callable | None = None) -> str:
    """The report as HTML. qt=True targets QTextDocument (report PDF): table attributes instead of modern CSS,
    images and histogram bars as document resources supplied through img_src."""
    E = lambda s: html.escape(str(s), quote=True)
    p, v, res = ctx.project, ctx.variant, ctx.result
    m = res.metrics if res is not None else None
    ua, u = ctx.angle_unit, ctx.units
    tags = {c.uid: c.tag for c in v.cameras}
    slot = {c.uid: c.slot for c in v.cameras}
    out = []
    w = out.append

    def table(heads):
        """heads: [(label, numeric)] -> opening tag and header row."""
        if qt:
            row = ''.join(f'<th align="right" nowrap>{E(h)}</th>' if n else f'<th align="left">{E(h)}</th>'
                          for h, n in heads)
            return f'<table class="t" width="100%" cellspacing="0" cellpadding="4"><thead><tr>{row}</tr></thead>\n'
        row = ''.join(f'<th class="n">{E(h)}</th>' if n else f'<th>{E(h)}</th>' for h, n in heads)
        return f'<div class="wrap"><table><thead><tr>{row}</tr></thead><tbody>\n'

    tbl_end = '</table>\n' if qt else '</tbody></table></div>\n'

    def num(val, warn=False):
        cls = 'n warn' if warn else 'n'
        return f'<td class="{cls}" align="right" nowrap>{val}</td>' if qt else f'<td class="{cls}">{val}</td>'

    def sw(s):
        col = CAM_COLORS[s % len(CAM_COLORS)]
        if qt:
            return f'<span style="color:{col}">&#9632;</span>&nbsp;'
        return f'<span class="sw" style="background:{col}"></span>'

    def tagcell(c):
        return f'<td nowrap>{sw(c.slot)}<b>{E(c.tag)}</b></td>' if qt else f'<td class="tag">{sw(c.slot)}<b>{E(c.tag)}</b></td>'

    def pill(sev):
        lab, bg, fg = SEVERITY_STYLE.get(sev, (sev.upper(), MUTED, '#ffffff'))
        if qt:
            return (f'<span style="background-color:{bg}; color:{fg}; font-size:7pt; font-weight:600">'
                    f'&nbsp;{lab}&nbsp;</span>')
        return f'<span class="pill" style="background:{bg};color:{fg}">{lab}</span>'

    def figure(key, cap, data, width):
        src, iw, ih = img_src(key, data) if img_src else ('', 0, 0)
        if qt:
            hgt = f' height="{int(round(width * ih / iw))}"' if iw else ''
            return f'<img src="{E(src)}" width="{width}"{hgt}><br><span class="muted">{E(cap)}</span>'
        dims = f' width="{iw}" height="{ih}"' if iw else ''
        return f'<figure><img src="{src}" alt="{E(cap)}"{dims}><figcaption>{E(cap)}</figcaption></figure>'

    def grid(items, cols, width):
        if not items:
            return
        if qt:
            w('<table width="100%" cellspacing="0" cellpadding="4">')
            for i in range(0, len(items), cols):
                cells = ''.join(f'<td width="{100 // cols}%" valign="top">{figure(*it, width)}</td>'
                                for it in items[i:i + cols])
                w(f'<tr>{cells}</tr>')
            w('</table>\n')
        else:
            w(f'<div class="views c{cols}">' + ''.join(figure(*it, width) for it in items) + '</div>\n')

    title = f'Coverage report · {p.name} · {v.name}'
    if not qt:
        w('<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
          '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
          f'<meta name="generator" content="Vantage">\n<title>{E(title)}</title>\n<style>{_CSS_WEB}</style>\n'
          '</head>\n<body>\n<main>\n')
    else:
        w(f'<html><head><style>{_CSS_QT}</style></head><body>\n')

    # ---------------------------------------------------------------- header
    w(f'<p class="kicker">VANTAGE · SURROUND-VIEW COVERAGE REPORT</p>\n<h1>{E(p.name)}</h1>\n')
    w(f'<p class="sub">{E(v.name)} · {E(ctx.vehicle_model)} · {len(v.cameras)} cameras</p>\n')
    if v.note:
        w(f'<p class="note">{E(v.note)}</p>\n')
    meta = [('Project', p.name), ('Variant', v.name), ('Vehicle', ctx.vehicle_model),
            ('Frame', f'{ctx.frame_label} ({AXES_TEXT})'), ('Units', f'{ctx.units}, {ctx.angles}'),
            ('Exported', ctx.now.strftime('%Y-%m-%d %H:%M %Z').strip())]
    if qt:
        w('<table class="meta" cellspacing="0" cellpadding="2" style="margin-top:8px">')
        for i in range(0, len(meta), 2):
            w('<tr>' + ''.join(f'<td class="ml" width="12%">{E(k)}</td><td width="38%">{E(val)}</td>'
                               for k, val in meta[i:i + 2]) + '</tr>')
        w('</table>\n')
    else:
        w('<table class="meta">' + ''.join(f'<tr><th>{E(k)}</th><td>{E(val)}</td></tr>' for k, val in meta)
          + '</table>\n')

    # ---------------------------------------------------------------- summary
    w('<h2>Summary</h2>\n')
    if m is None:
        w('<p class="muted">No analysis result: load a vehicle body to analyse coverage.</p>\n')
    else:
        a = p.analysis
        tiles = [(f'{float(m["covered_pct"]):.2f} %', 'ROI ground covered'),
                 (f'{float(m["blind_m2"]):.3f} m²', 'Blind ground in the ROI'),
                 (f'{float(m["min_density"]):.1f} px/cm', 'Minimum pixel density'),
                 (f'{float(m["multi_pct"]):.2f} %', 'Seen by two or more cameras'),
                 (f'{float(m["min_ok_pct"]):.2f} %', f'At or above {a.min_density:.1f} px/cm (minimum)'),
                 (f'{float(m["target_ok_pct"]):.2f} %', f'At or above {a.target_density:.1f} px/cm (target)'),
                 (f'{float(m["in_focus_pct"]):.2f} %', 'In focus'),
                 (f'{float(m["roi_area"]):.2f} m²', f'ROI area ({float(m["cell"]) * 100:.0f} cm cells, '
                                                    f'z = {_fs(m["z"], 2)} m)')]
        if qt:
            w('<table width="100%" cellspacing="6" cellpadding="6">')
            for i in range(0, len(tiles), 4):
                w('<tr>' + ''.join(f'<td width="25%" style="border:1px solid #e1e0d9">'
                                   f'<span style="font-size:14pt; font-weight:600">{E(val)}</span><br>'
                                   f'<span style="color:#52514e; font-size:7.5pt">{E(lab)}</span></td>'
                                   for val, lab in tiles[i:i + 4]) + '</tr>')
            w('</table>\n')
        else:
            w('<div class="tiles">' + ''.join(f'<div class="tile"><div class="v">{E(val)}</div>'
                                              f'<div class="l">{E(lab)}</div></div>' for val, lab in tiles)
              + '</div>\n')

    # ---------------------------------------------------------------- overview images
    imgs = _image_list(ctx)
    main_imgs = [t for t in imgs if not t[0].startswith('cam:')]
    cam_imgs = [t for t in imgs if t[0].startswith('cam:')]
    if main_imgs:
        w('<h2>Views</h2>\n')
        first = [t for t in main_imgs if t[0] == 'viewport']
        rest = [t for t in main_imgs if t[0] != 'viewport']
        if first:
            w(('<p>' + figure(*first[0], 680) + '</p>\n') if qt else figure(*first[0], 680) + '\n')
        grid(rest, 2, 330)

    # ---------------------------------------------------------------- cameras
    w('<h2>Cameras</h2>\n')
    w(f'<p class="note">{E(_conventions_text(ctx))}</p>\n')
    rows = _camera_rows(ctx)
    w(table([('Tag', False), ('Location', False), (f'X ({u})', True), (f'Y ({u})', True), (f'Z ({u})', True),
             (f'Yaw ({ua})', True), (f'Pitch ({ua})', True), (f'Roll ({ua})', True)]))
    for r in rows:
        w('<tr>' + tagcell(r['cam']) + f'<td>{E(r["location"])}</td>'
          + ''.join(num(r[k]) for k in ('x', 'y', 'z', 'yaw', 'pitch', 'roll')) + '</tr>\n')
    w(tbl_end)
    w('<h3>Hardware and cabling</h3>\n' if not qt else
      '<p style="margin-top:10px; font-weight:600">Hardware and cabling</p>\n')
    w(table([('Tag', False), ('Module', False), ('Lens', False), ('GMSL port', False), ('Cable (m)', True),
             ('Routed (m)', True), ('Standoff (mm)', True), ('Bracket', False)]))
    for r in rows:
        w('<tr>' + tagcell(r['cam']) + f'<td>{E(r["module"])}</td><td>{E(r["lens"])}</td><td>{E(r["port"])}</td>'
          + num(r['cable']) + num(r['routed'], r['over']) + num(r['standoff']) + f'<td>{E(r["bracket"])}</td></tr>\n')
    w(tbl_end)
    if any(r['over'] for r in rows):
        w('<p class="note">Routed lengths in red are longer than the module cable.</p>\n')
    rig = _rig_doc(ctx)
    if rig:
        w('<h3>Rig</h3>\n' if not qt else '<p style="margin-top:10px; font-weight:600">Rig</p>\n')
        w(table([('Part', False), ('Kind', False), (f'X ({u})', True), (f'Y ({u})', True), (f'Z ({u})', True),
                 (f'Size ({u})', True), ('Blocks cameras', False), ('Carries cameras', False)]))
        for r in rig:
            size = (' × '.join(_fs(r[k], ctx.pdp) for k in ('length', 'width', 'height')) if r['kind'] == 'box' else
                    f'⌀{_fs(2 * r["radius"], ctx.pdp)} × {_fs(r.get("height", r.get("length")), ctx.pdp)}')
            w(f'<tr><td>{E(r["name"])}</td><td>{E(r["kind"])}</td>' + ''.join(num(_fs(v, ctx.pdp)) for v in r['centre'])
              + num(size) + f'<td>{"yes" if r["occludes"] else "no"}</td><td>{"yes" if r["mountable"] else "no"}</td></tr>\n')
        w(tbl_end)
    sens = _sensors(ctx)
    if sens:
        head = 'Lidars and radars' if all(m.has_field for _, _, m in sens) else 'Lidars, radars, GNSS and IMU'
        w(f'<h3>{head}</h3>\n' if not qt else
          f'<p style="margin-top:10px; font-weight:600">{head}</p>\n')
        w(table([('Tag', False), ('Kind', False), ('Model', False), (f'X ({u})', True), (f'Y ({u})', True),
                 (f'Z ({u})', True), ('Yaw', True), ('Pitch', True), ('Field', False)]))
        for _, s, m in sens:
            w(f'<tr><td>{E(s.tag)}</td><td>{E(s.kind)}</td><td>{E((m.vendor + " " + m.name).strip())}</td>'
              + ''.join(num(_fs(ctx.L(v - o), ctx.pdp)) for v, o in zip((s.x, s.y, s.z), ctx.origin))
              + num(f'{s.yaw:.1f}°') + num(f'{s.pitch:.1f}°') + f'<td>{E(m.summary())}</td></tr>\n')
        w(tbl_end)

    # ---------------------------------------------------------------- per-camera coverage
    w('<h2>Coverage per camera</h2>\n')
    if res is None:
        w('<p class="muted">No analysis result.</p>\n')
    else:
        w('<p class="note">Ground areas in m² over the ROI. "If removed" columns show what the rig loses '
          'without that camera.</p>\n')
        w(table([('Camera', False), ('Sees', True), ('Only it sees', True), ('Best camera for', True),
                 ('Min density (px/cm)', True), ('Blind if removed', True),
                 (f'Below {p.analysis.min_density:.1f} px/cm if removed', True), ('Overlaps', False)]))
        for c in v.cameras:
            pc = res.per_cam.get(c.uid)
            if pc is None:
                w('<tr>' + tagcell(c) + '<td colspan="7" class="muted">Not analysed (hidden)</td></tr>\n')
                continue
            ov = ', '.join(f'{E(tags.get(k, "?"))} {val:.2f}' for k, val in
                           sorted(pc.get('overlaps', {}).items(), key=lambda kv: -kv[1])) or '—'
            w('<tr>' + tagcell(c) + num(f'{pc["seen_m2"]:.2f}') + num(f'{pc["unique_m2"]:.2f}')
              + num(f'{pc["best_m2"]:.2f}') + num(f'{pc["min_dens"]:.1f}') + num(f'{pc["if_removed_blind"]:.2f}')
              + num(f'{pc["if_removed_below"]:.2f}') + f'<td>{ov}</td></tr>\n')
        w(tbl_end)

    # ---------------------------------------------------------------- seams
    w('<h2>Seams</h2>\n')
    if res is None or not res.seams:
        w('<p class="muted">No seams (fewer than two analysed cameras).</p>\n')
    else:
        w(f'<p class="note">Stitching seams as azimuths from the body centre (0 = forward, positive to the left); '
          f'blend width {v.blend_m:.2f} m.</p>\n')
        w(table([('Between', False), (f'Azimuth ({ua})', True), ('Placed', False),
                 ('Median density on seam (px/cm)', True)]))
        for sm in res.seams:
            w(f'<tr><td>{sw(slot.get(sm.a, 0))}{E(tags.get(sm.a, "?"))} · {sw(slot.get(sm.b, 0))}'
              f'{E(tags.get(sm.b, "?"))}</td>' + num(ctx.As(sm.angle))
              + f'<td>{"auto" if sm.auto else "manual"}</td>' + num(f'{sm.density:.1f}') + '</tr>\n')
        w(tbl_end)

    # ---------------------------------------------------------------- histogram
    w('<h2>Pixel density</h2>\n')
    if res is None:
        w('<p class="muted">No analysis result.</p>\n')
    else:
        edges, areas, blind = res.hist
        roi_area = max(float(m['roi_area']), 1e-9)
        mx = max([float(x) for x in areas] + [float(blind), 1e-9])
        mind, tgt = p.analysis.min_density, p.analysis.target_density
        w(f'<p class="note">Best-camera pixel density over the ROI ground; minimum {mind:.1f} px/cm, '
          f'target {tgt:.1f} px/cm.</p>\n')
        w(table([('Density (px/cm)', False), ('Area (m²)', True), ('Share of ROI', True), ('', False)]))
        hrows = [('Blind', float(blind), '#d03b3b')]
        for i, ar in enumerate(areas):
            lo, hi = float(edges[i]), float(edges[i + 1])
            lab = f'below {hi:g}' if lo <= 0 else (f'{lo:g} and above' if hi > 1e6 else f'{lo:g} to {hi:g}')
            col = '#ec835a' if hi <= mind else ('#86b6ef' if lo < tgt else '#2a78d6')
            hrows.append((lab, float(ar), col))
        for lab, ar, col in hrows:
            pct = 100.0 * ar / roi_area
            wb = ar / mx
            bar = ''
            if ar > 0:
                if qt:
                    src, _, _ = img_src(f'bar:{col}', None) if img_src else ('', 0, 0)
                    bar = f'<img src="{E(src)}" width="{max(1, int(round(wb * 300)))}" height="9">'
                else:
                    bar = f'<div class="bar" style="width:{100 * wb:.1f}%;background:{col}"></div>'
            w(f'<tr><td>{E(lab)}</td>' + num(f'{ar:.3f}') + num(f'{pct:.2f} %')
              + (f'<td width="45%">{bar}</td>' if qt else f'<td class="barcell">{bar}</td>') + '</tr>\n')
        w(tbl_end)

    # ---------------------------------------------------------------- issues
    issues, computed = ctx.issues()
    w(f'<h2>Issues ({len(issues)})</h2>\n')
    if computed:
        w('<p class="note">Computed at export; mount-surface checks were not run.</p>\n')
    if not issues:
        w('<p class="muted">No open issues.</p>\n')
    elif qt:
        w('<table width="100%" cellspacing="0" cellpadding="4">')
        for it in issues:
            w(f'<tr><td width="12%" valign="top">{pill(it.severity)}</td><td>{E(it.message)}'
              + (f'<br><span class="obj">{E(it.objects)}</span>' if it.objects else '') + '</td></tr>')
        w('</table>\n')
    else:
        w('<ul class="issues">')
        for it in issues:
            w(f'<li>{pill(it.severity)}<div>{E(it.message)}'
              + (f'<div class="obj">{E(it.objects)}</div>' if it.objects else '') + '</div></li>')
        w('</ul>\n')

    # ---------------------------------------------------------------- tolerance
    w('<h2>Tolerance</h2>\n')
    for line in _tolerance_lines(ctx):
        w(f'<p>{E(line)}</p>\n')

    # ---------------------------------------------------------------- camera views
    if cam_imgs:
        w('<h2 style="page-break-before: always">Camera views</h2>\n' if qt else '<h2>Camera views</h2>\n')
        grid(cam_imgs, 3, 218)

    w(f'<p class="muted" style="margin-top:28px">Generated by Vantage · {E(ctx.stamp)}</p>\n')
    w('</main>\n</body>\n</html>\n' if not qt else '</body></html>\n')
    return ''.join(out)


def _report_html_file(ctx: _Ctx) -> str:
    def src(key, data):
        sz = _png_size(data) or (0, 0)
        return 'data:image/png;base64,' + base64.b64encode(data).decode('ascii'), sz[0], sz[1]
    return _report_html(ctx, qt=False, img_src=src)


# ====================================================================== Qt helpers (PDF)
_FONT_STATE: dict = {}


def _require_app():
    from PyQt6.QtGui import QGuiApplication
    if QGuiApplication.instance() is None:
        raise RuntimeError('PDF export needs a running QGuiApplication (fonts and painting).')


def _families() -> tuple[str, str]:
    """(sans, mono) family names: IBM Plex from vantage/resources/fonts if it loads, else the defaults."""
    if 'families' in _FONT_STATE:
        return _FONT_STATE['families']
    from PyQt6.QtGui import QFont, QFontDatabase
    fdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'resources', 'fonts')
    if os.path.isdir(fdir):
        for fn in sorted(os.listdir(fdir)):
            if fn.lower().endswith(('.ttf', '.otf')):
                QFontDatabase.addApplicationFont(os.path.join(fdir, fn))
    fams = set(QFontDatabase.families())
    sans = 'IBM Plex Sans' if 'IBM Plex Sans' in fams else QFont().defaultFamily()
    mono = 'IBM Plex Mono' if 'IBM Plex Mono' in fams else 'monospace'
    _FONT_STATE['families'] = (sans, mono)
    return sans, mono


class _PdfOut:
    """QPdfWriter into memory, A4, zero margins, painter in device units (resolution dpi)."""

    def __init__(self, landscape: bool, title: str, res: int = 300):
        from PyQt6.QtCore import QBuffer, QIODevice, QMarginsF
        from PyQt6.QtGui import QPageLayout, QPageSize, QPainter, QPdfWriter
        self.buf = QBuffer()
        self.buf.open(QIODevice.OpenModeFlag.WriteOnly)
        self.writer = QPdfWriter(self.buf)
        self.writer.setResolution(res)
        self.writer.setTitle(title)
        self.writer.setCreator('Vantage')
        orient = QPageLayout.Orientation.Landscape if landscape else QPageLayout.Orientation.Portrait
        self.writer.setPageLayout(QPageLayout(QPageSize(QPageSize.PageSizeId.A4), orient, QMarginsF(0, 0, 0, 0)))
        self.painter = QPainter(self.writer)
        self.painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        self.res = res
        self.pages = 1

    def new_page(self):
        self.writer.newPage()
        self.pages += 1

    def finish(self) -> bytes:
        self.painter.end()
        data = bytes(self.buf.data())
        self.buf.close()
        return data


class _Sheet:
    """Millimetre drawing helpers on top of a _PdfOut painter."""

    def __init__(self, out: _PdfOut):
        from PyQt6.QtCore import Qt
        self.out = out
        self.p = out.painter
        self.k = out.res / 25.4
        self.sans, self.mono = _families()
        self.Qt = Qt

    def r(self, x, y, w, h):
        from PyQt6.QtCore import QRectF
        return QRectF(x * self.k, y * self.k, w * self.k, h * self.k)

    def pt(self, x, y):
        from PyQt6.QtCore import QPointF
        return QPointF(x * self.k, y * self.k)

    def pen(self, color, wmm=0.2, dash=None):
        from PyQt6.QtGui import QColor, QPen
        pen = QPen(QColor(color))
        pen.setWidthF(wmm * self.k)
        pen.setCapStyle(self.Qt.PenCapStyle.FlatCap)
        pen.setJoinStyle(self.Qt.PenJoinStyle.MiterJoin)
        if dash:
            pen.setDashPattern(dash)
        return pen

    def font(self, size, weight=None, mono=False, caps=False):
        from PyQt6.QtGui import QFont
        f = QFont(self.mono if mono else self.sans)
        f.setPointSizeF(size)
        if weight:
            f.setWeight({'medium': QFont.Weight.Medium, 'semibold': QFont.Weight.DemiBold}.get(weight, QFont.Weight.Normal))
        if caps:
            f.setCapitalization(QFont.Capitalization.AllUppercase)
            f.setLetterSpacing(QFont.SpacingType.PercentageSpacing, 108)
        return f

    def flags(self, align='l', valign='c', wrap=False):
        Q = self.Qt
        a = {'l': Q.AlignmentFlag.AlignLeft, 'r': Q.AlignmentFlag.AlignRight, 'c': Q.AlignmentFlag.AlignHCenter}[align]
        b = {'t': Q.AlignmentFlag.AlignTop, 'c': Q.AlignmentFlag.AlignVCenter, 'b': Q.AlignmentFlag.AlignBottom}[valign]
        f = a.value | b.value
        if wrap:
            f |= Q.TextFlag.TextWordWrap.value
        return f

    def text(self, x, y, w, h, s, font, color=INK, align='l', valign='c', wrap=False, elide=False):
        from PyQt6.QtGui import QColor, QFontMetricsF
        self.p.setFont(font)
        self.p.setPen(QColor(color))
        if elide:
            fm = QFontMetricsF(font, self.out.writer)
            s = fm.elidedText(s, self.Qt.TextElideMode.ElideRight, w * self.k)
        self.p.drawText(self.r(x, y, w, h), self.flags(align, valign, wrap), s)

    def text_height(self, s, font, w):
        self.p.setFont(font)
        br = self.p.boundingRect(self.r(0, 0, w, 500), self.flags('l', 't', True), s)
        return br.height() / self.k

    def text_width(self, s, font):
        from PyQt6.QtGui import QFontMetricsF
        return QFontMetricsF(font, self.out.writer).horizontalAdvance(s) / self.k

    def line(self, x1, y1, x2, y2, color=HAIR, wmm=0.2, dash=None):
        self.p.setPen(self.pen(color, wmm, dash))
        self.p.drawLine(self.pt(x1, y1), self.pt(x2, y2))

    def rect(self, x, y, w, h, stroke=None, fill=None, wmm=0.2, radius=0.0):
        from PyQt6.QtGui import QBrush, QColor
        self.p.setPen(self.pen(stroke, wmm) if stroke else self.Qt.PenStyle.NoPen)
        self.p.setBrush(QBrush(QColor(fill)) if fill else self.Qt.BrushStyle.NoBrush)
        if radius:
            self.p.drawRoundedRect(self.r(x, y, w, h), radius * self.k, radius * self.k)
        else:
            self.p.drawRect(self.r(x, y, w, h))

    def label(self, x, y, s, w=80):
        self.text(x, y, w, 3.2, s, self.font(6, 'semibold', caps=True), MUTED, valign='c')


# ====================================================================== mount sheet (PDF)
_MS_COLS = [('Tag', 11, 'l'), ('Location', 38, 'l'), ('X', 15, 'r'), ('Y', 15, 'r'), ('Z', 15, 'r'),
            ('Yaw', 14, 'r'), ('Pitch', 14, 'r'), ('Roll', 13, 'r'), ('Module', 25, 'l'), ('Lens', 22, 'l'),
            ('Port', 14, 'l'), ('Cable', 13, 'r'), ('Routed', 15, 'r'), ('Standoff', 17, 'r'), ('Bracket', 32, 'l')]
_PAGE_W, _PAGE_H, _MX = 297.0, 210.0, 12.0


def _ms_issue(sh: _Sheet, x, y, w, it, draw=True) -> float:
    lab, bg, fg = SEVERITY_STYLE.get(it.severity, (it.severity.upper(), MUTED, '#ffffff'))
    fmsg, fobj = sh.font(7.2), sh.font(6.3)
    tw = w - 18
    h1 = sh.text_height(it.message, fmsg, tw)
    h2 = sh.text_height(it.objects, fobj, tw) if it.objects else 0.0
    h = max(h1 + h2 + 0.6, 4.0)
    if draw:
        sh.rect(x, y + 0.2, 15.5, 3.3, fill=bg, radius=1.65)
        sh.text(x, y + 0.2, 15.5, 3.3, lab, sh.font(5.3, 'semibold'), fg, align='c')
        sh.text(x + 18, y, tw, h1 + 0.5, it.message, fmsg, INK, valign='t', wrap=True)
        if it.objects:
            sh.text(x + 18, y + h1 + 0.2, tw, h2 + 0.5, it.objects, fobj, MUTED, valign='t', wrap=True)
    return h


def _ms_title_block(sh: _Sheet, ctx: _Ctx, page: int, pages: int):
    y, h = 184.0, 14.0
    x0 = _MX
    w_all = _PAGE_W - 2 * _MX
    cells = [('Project', ctx.project.name, 46), ('Variant', ctx.variant.name, 46), ('Vehicle', ctx.vehicle_model, 48),
             ('Frame', ctx.frame_label, 56),
             ('Units', f'{ctx.units} · {"rad" if ctx.angles == "rad" else "deg"}', 24),
             ('Date', ctx.now.strftime('%Y-%m-%d %H:%M'), 30), ('Sheet', f'{page} / {pages}', 23)]
    sh.rect(x0, y, w_all, h, stroke=INK, wmm=0.35)
    x = x0
    for i, (lab, val, cw) in enumerate(cells):
        if i:
            sh.line(x, y, x, y + h, INK2, 0.18)
        sh.label(x + 2, y + 1.6, lab, cw - 4)
        sh.text(x + 2, y + 6.0, cw - 4, 6.0, val, sh.font(8.2, 'medium'), INK, elide=True)
        x += cw


def _ms_header(sh: _Sheet, ctx: _Ctx, cont=False):
    x0, x1 = _MX, _PAGE_W - _MX
    sh.label(x0, 10.5, 'Camera mount sheet' + (' · continued' if cont else ''), 120)
    sh.text(x0, 14.0, 180, 8.5, f'{ctx.variant.name}', sh.font(15, 'semibold'), INK, valign='c', elide=True)
    sh.text(x1 - 120, 10.5, 120, 3.2, 'Vantage · surround-view rig planning', sh.font(6.5), MUTED, align='r')
    sh.text(x1 - 120, 14.5, 120, 4.0, f'{ctx.project.name} · {ctx.vehicle_model}', sh.font(8.5, 'medium'), INK2,
            align='r', elide=True)
    sh.text(x1 - 120, 18.6, 120, 3.6, f'{len(ctx.variant.cameras)} cameras · {ctx.frame_label}', sh.font(7), MUTED,
            align='r', elide=True)
    sh.line(x0, 24.5, x1, 24.5, INK, 0.35)


def _ms_plan(sh: _Sheet, ctx: _Ctx, X, Y, W, H):
    from PyQt6.QtCore import QPointF
    from PyQt6.QtGui import QBrush, QColor, QPainterPath, QPolygonF
    p = sh.p
    o = ctx.origin
    sh.rect(X, Y, W, H, stroke=HAIR, wmm=0.25)
    sh.label(X + 3, Y + 1.8, 'Plan view · from above, front to the right', 120)
    res = ctx.result
    outline = [a - o[:2] for a in _outline_polys(ctx)]
    roi = []
    if res is not None and getattr(res, 'grid', None) is not None:
        roi = [np.asarray(b, float)[:, :2] - o[:2] for b in res.grid.boundary if len(b) >= 2]
    cams = []
    for cm in ctx.models:
        c = cm.cam
        d = cm.aim
        hn = math.hypot(d[0], d[1])
        end = None
        if hn > 1e-6:
            dist = 1.5
            if d[2] < -1e-6:
                dist = min(c.z / -d[2] * hn, 1.5)
            end = np.array([c.x, c.y]) - o[:2] + d[:2] / hn * dist
        cams.append((cm, np.array([c.x, c.y]) - o[:2], end, d[:2] / hn if hn > 1e-6 else None))
    ecu = np.array([ctx.project.ecu.x, ctx.project.ecu.y]) - o[:2]
    pts = [np.zeros((1, 2)), ecu[None]]
    pts += outline + roi + [c[1][None] for c in cams] + [c[2][None] for c in cams if c[2] is not None]
    P = np.vstack(pts)
    lo, hi = P.min(axis=0) - 0.35, P.max(axis=0) + 0.35
    ix, iy, iw, ih = X + 6, Y + 8, W - 12, H - 17
    s = min(iw / max(hi[0] - lo[0], 1e-3), ih / max(hi[1] - lo[1], 1e-3))     # mm per metre
    cxw, cyw = (lo + hi) / 2
    cx, cy = ix + iw / 2, iy + ih / 2

    def M(x, y):
        return cx + (x - cxw) * s, cy - (y - cyw) * s

    def Q(x, y):
        a, b = M(x, y)
        return QPointF(a * sh.k, b * sh.k)

    # metre grid, labelled in export units
    p.save()
    p.setClipRect(sh.r(X + 0.3, Y + 6.5, W - 0.6, H - 14.5))
    gx0, gx1 = math.floor(cxw - (iw / 2 + 6) / s), math.ceil(cxw + (iw / 2 + 6) / s)
    gy0, gy1 = math.floor(cyw - (ih / 2 + 6) / s), math.ceil(cyw + (ih / 2 + 6) / s)
    for gx in range(gx0, gx1 + 1):
        a, _ = M(gx, 0)
        sh.line(a, Y + 6.5, a, Y + H - 8, GRID if gx else '#dddbd3', 0.15)
    for gy in range(gy0, gy1 + 1):
        _, b = M(0, gy)
        sh.line(X, b, X + W, b, GRID if gy else '#dddbd3', 0.15)
    p.restore()
    fl = sh.font(5.2, mono=True)
    for gx in range(gx0, gx1 + 1):
        a, _ = M(gx, 0)
        if X + 4 < a < X + W - 4:
            sh.text(a - 6, Y + H - 11.5, 12, 3, _fs(gx * ctx.scale, 0), fl, MUTED, align='c')
    for gy in range(gy0, gy1 + 1):
        _, b = M(0, gy)
        if Y + 9 < b < Y + H - 12:
            sh.text(X + 1, b - 1.5, 9, 3, _fs(gy * ctx.scale, 0), fl, MUTED, align='l')

    # ROI boundary
    p.setBrush(sh.Qt.BrushStyle.NoBrush)
    p.setPen(sh.pen(MUTED, 0.25, dash=[5, 4]))
    for b in roi:
        p.drawPolyline(QPolygonF([Q(*v) for v in b]))
    # car outline
    if outline:
        path = QPainterPath()
        for a in outline:
            path.addPolygon(QPolygonF([Q(*v) for v in a]))
            path.closeSubpath()
        p.setPen(sh.pen(INK2, 0.3))
        p.setBrush(QBrush(QColor(TINT)))
        p.drawPath(path)
    # origin marker (crosshair) and an axis triad in the top-right corner
    ox, oy = M(0, 0)
    p.setBrush(sh.Qt.BrushStyle.NoBrush)
    p.setPen(sh.pen(INK, 0.25))
    p.drawEllipse(sh.pt(ox, oy), 1.1 * sh.k, 1.1 * sh.k)
    sh.line(ox - 2.0, oy, ox + 2.0, oy, INK, 0.2)
    sh.line(ox, oy - 2.0, ox, oy + 2.0, INK, 0.2)
    sh.text(ox + 1.3, oy + 0.5, 14, 3, 'origin', sh.font(5.2), INK2)
    tx, ty = X + W - 15.0, Y + 18.0
    fax = sh.font(6, 'semibold')
    _arrow(sh, tx, ty, tx + 8, ty, INK, 0.25, 1.3)
    _arrow(sh, tx, ty, tx, ty - 8, INK, 0.25, 1.3)
    sh.text(tx + 8.6, ty - 1.6, 4, 3.2, 'x', fax, INK)
    sh.text(tx - 1.6, ty - 11.6, 4, 3.2, 'y', fax, INK)
    # ECU
    ex, ey = M(*ecu)
    sh.rect(ex - 1.2, ey - 1.2, 2.4, 2.4, stroke=INK2, fill='#ffffff', wmm=0.22)
    sh.text(ex + 1.8, ey - 1.5, 20, 3, 'ECU', sh.font(5.2, 'semibold'), INK2)
    # cameras: aim arrow, marker, tag
    for cm, xy, end, hdir in cams:
        col = CAM_COLORS[cm.cam.slot % len(CAM_COLORS)]
        px, py = M(*xy)
        u = None
        if end is not None:
            ex2, ey2 = M(*end)
            ln = math.hypot(ex2 - px, ey2 - py)
            u = ((ex2 - px) / ln, (ey2 - py) / ln) if ln > 1e-6 else (hdir[0], -hdir[1])
            if ln < 5.0:
                ex2, ey2 = px + u[0] * 5.0, py + u[1] * 5.0
            _arrow(sh, px, py, ex2, ey2, col, 0.35, 1.7)
        else:
            p.setPen(sh.pen(col, 0.3))
            p.setBrush(sh.Qt.BrushStyle.NoBrush)
            p.drawEllipse(sh.pt(px, py), 2.6 * sh.k, 2.6 * sh.k)
        sh.rect(px - 1.35, py - 1.35, 2.7, 2.7, stroke='#ffffff', fill=col, wmm=0.25)
        f = sh.font(6.4, 'semibold')
        tw = sh.text_width(cm.cam.tag, f) + 2.2
        if u is None:
            u = (px - cx, py - cy)
            n = math.hypot(*u) or 1.0
            u = (-u[0] / n, -u[1] / n)
        lx, ly = px - u[0] * (3.2 + tw / 2), py - u[1] * 4.4
        sh.rect(lx - tw / 2, ly - 1.75, tw, 3.5, stroke=col, fill='#ffffff', wmm=0.3, radius=0.6)
        sh.text(lx - tw / 2, ly - 1.75, tw, 3.5, cm.cam.tag, f, INK, align='c')

    # legend and scale bar
    ly = Y + H - 5.2
    lx = X + 4
    fleg = sh.font(5.8)
    sh.rect(lx, ly - 1.1, 2.2, 2.2, fill=CAM_COLORS[0])
    sh.text(lx + 3.2, ly - 1.6, 30, 3.2, 'Camera', fleg, INK2)
    lx += 16
    _arrow(sh, lx, ly, lx + 7, ly, INK2, 0.3, 1.3)
    sh.text(lx + 8.5, ly - 1.6, 50, 3.2, 'Optical axis to the ground', fleg, INK2)
    lx += 41
    sh.line(lx, ly, lx + 7, ly, MUTED, 0.25, dash=[5, 4])
    dist = ctx.project.roi.distance
    sh.text(lx + 8.5, ly - 1.6, 50, 3.2, f'ROI boundary ({dist:.1f} m)' if ctx.project.roi.mode == 'offset'
            else 'ROI boundary', fleg, INK2)
    lx += 36
    sh.text(lx, ly - 1.6, 60, 3.2, f'Grid 1 m, labels in {ctx.units}', fleg, INK2)
    bar = s * 1.0
    bx = X + W - 5 - bar
    sh.line(bx, ly, bx + bar, ly, INK, 0.35)
    for t in (0.0, 0.5, 1.0):
        sh.line(bx + bar * t, ly - (1.0 if t != 0.5 else 0.6), bx + bar * t, ly + 0.0, INK, 0.25)
    sh.text(bx - 14, ly - 1.6, 12.5, 3.2, '1 m' if ctx.units == 'm' else '1000 mm', fleg, INK2, align='r')


def _arrow(sh: _Sheet, x1, y1, x2, y2, color, wmm, head):
    from PyQt6.QtCore import QPointF
    from PyQt6.QtGui import QBrush, QColor, QPolygonF
    p = sh.p
    ln = math.hypot(x2 - x1, y2 - y1)
    if ln < 1e-6:
        return
    ux, uy = (x2 - x1) / ln, (y2 - y1) / ln
    bx, by = x2 - ux * head * 1.3, y2 - uy * head * 1.3
    p.setPen(sh.pen(color, wmm))
    p.drawLine(sh.pt(x1, y1), sh.pt(bx + ux * 0.2, by + uy * 0.2))
    p.setPen(sh.Qt.PenStyle.NoPen)
    p.setBrush(QBrush(QColor(color)))
    k = sh.k
    p.drawPolygon(QPolygonF([QPointF(x2 * k, y2 * k), QPointF((bx - uy * head * 0.55) * k, (by + ux * head * 0.55) * k),
                             QPointF((bx + uy * head * 0.55) * k, (by - ux * head * 0.55) * k)]))
    p.setBrush(sh.Qt.BrushStyle.NoBrush)


def _ms_table(sh: _Sheet, ctx: _Ctx, X, Y) -> float:
    u, ua = ctx.units, ctx.angle_unit
    units = {'X': u, 'Y': u, 'Z': u, 'Yaw': ua, 'Pitch': ua, 'Roll': ua, 'Cable': 'm', 'Routed': 'm',
             'Standoff': 'mm'}
    W = sum(c[1] for c in _MS_COLS)
    hh, rh = 6.5, 6.0
    sh.rect(X, Y, W, hh, fill=TINT)
    sh.line(X, Y, X + W, Y, INK, 0.3)
    sh.line(X, Y + hh, X + W, Y + hh, INK2, 0.2)
    fh, fu = sh.font(6.3, 'semibold'), sh.font(5.6)
    x = X
    for name, cw, al in _MS_COLS:
        un = units.get(name)
        if un:
            nw = sh.text_width(name, fh)
            uw = sh.text_width(f' {un}', fu)
            if al == 'r':
                sh.text(x + 1.2, Y, cw - 2.4 - uw, hh, name, fh, INK2, align='r')
                sh.text(x + cw - 1.2 - uw, Y + 0.15, uw, hh, f' {un}', fu, MUTED, align='l')
            else:
                sh.text(x + 1.2, Y, nw, hh, name, fh, INK2)
        else:
            sh.text(x + 1.2, Y, cw - 2.4, hh, name, fh, INK2, align=al)
        x += cw
    fm, ft, fb = sh.font(6.8, mono=True), sh.font(6.8), sh.font(7, 'semibold')
    y = Y + hh
    for r in _camera_rows(ctx):
        x = X
        col = CAM_COLORS[r['slot'] % len(CAM_COLORS)]
        vals = {'Tag': r['tag'], 'Location': r['location'], 'X': r['x'], 'Y': r['y'], 'Z': r['z'], 'Yaw': r['yaw'],
                'Pitch': r['pitch'], 'Roll': r['roll'], 'Module': r['module'], 'Lens': r['lens'], 'Port': r['port'],
                'Cable': r['cable'], 'Routed': r['routed'], 'Standoff': r['standoff'], 'Bracket': r['bracket']}
        for name, cw, al in _MS_COLS:
            v = vals[name]
            if name == 'Tag':
                sh.rect(x + 1.2, y + rh / 2 - 1.2, 2.4, 2.4, fill=col)
                sh.text(x + 4.6, y, cw - 5.2, rh, v, fb, INK, elide=True)
            else:
                mono = name in ('X', 'Y', 'Z', 'Yaw', 'Pitch', 'Roll', 'Cable', 'Routed', 'Standoff')
                color = '#b8312f' if (name == 'Routed' and r['over']) else INK
                sh.text(x + 1.2, y, cw - 2.4, rh, v, fm if mono else ft, color, align=al, elide=True)
            x += cw
        y += rh
        sh.line(X, y, X + W, y, HAIR, 0.15)
    sh.line(X, y, X + W, y, INK2, 0.25)
    if any(r['over'] for r in _camera_rows(ctx)):
        sh.text(X, y + 0.8, W, 3.2, 'Red routed length: longer than the module cable.', sh.font(5.8), '#b8312f')
    return y


def _mount_sheet_pdf(ctx: _Ctx) -> tuple[bytes, int]:
    _require_app()
    out = _PdfOut(True, f'Camera mount sheet · {ctx.project.name} · {ctx.variant.name}')
    sh = _Sheet(out)
    issues, computed = ctx.issues()

    # layout pass for the right column: notes anchored at the bottom, issues fill the space above
    RX, RW = 182.0, _PAGE_W - _MX - 182.0
    fnote = sh.font(6.7)
    notes = [('Conventions', [_conventions_text(ctx)]), ('Tolerance', _tolerance_lines(ctx))]
    blocks, total = [], 0.0
    for head, paras in notes:
        hs = [sh.text_height(t, fnote, RW) for t in paras]
        h = 4.2 + sum(hs) + 1.0 * (len(hs) - 1)
        blocks.append((head, paras, hs, h))
        total += h + 3.0
    notes_top = 124.0 - total + 3.0
    iss_top, iss_bot = 30.5, notes_top - 4.0
    heights = [_ms_issue(sh, 0, 0, RW, it, draw=False) for it in issues]
    fit, used = 0, 0.0
    reserve = 4.0 + (3.5 if computed else 0.0)
    for i, h in enumerate(heights):
        more = len(heights) - i - 1
        need = used + h + 1.8 + (4.0 if more else 0.0)
        if iss_top + reserve + need > iss_bot:
            break
        used += h + 1.8
        fit += 1
    overflow = issues[fit:]
    # continuation sheets: two columns of issues per sheet
    colw = (_PAGE_W - 2 * _MX - 10) / 2
    sheets, page, col, y = [], [], 0, 33.0
    for idx, it in enumerate(overflow):
        h = _ms_issue(sh, 0, 0, colw, it, draw=False)
        if y + h > 180.0 and page:
            if col == 0:
                col, y = 1, 33.0
            else:
                sheets.append(page)
                page, col, y = [], 0, 33.0
        page.append((idx, col, y))
        y += h + 2.0
    if page:
        sheets.append(page)
    pages = 1 + len(sheets)

    _ms_header(sh, ctx)
    _ms_plan(sh, ctx, _MX, 28.0, 164.0, 96.0)
    # issues
    sh.label(RX, iss_top - 2.4, f'Issues ({len(issues)})', RW)
    y = iss_top + 1.6
    if computed:
        sh.text(RX, y, RW, 3.0, 'Computed at export; mount-surface checks not run.', sh.font(5.8), MUTED)
        y += 3.5
    if not issues:
        sh.text(RX, y, RW, 4, 'No open issues.', sh.font(7.2), INK2)
    for it, h in zip(issues[:fit], heights[:fit]):
        _ms_issue(sh, RX, y, RW, it)
        y += h + 1.8
    if overflow:
        more = 'sheet 2' if pages == 2 else f'sheets 2 to {pages}'
        sh.text(RX, y, RW, 3.5, f'+ {len(overflow)} more on {more}', sh.font(6.5, 'semibold'), INK2)
    # conventions and tolerance notes
    y = notes_top
    for head, paras, hs, h in blocks:
        sh.line(RX, y - 1.6, RX + RW, y - 1.6, HAIR, 0.2)
        sh.label(RX, y, head, RW)
        yy = y + 4.2
        for t, th in zip(paras, hs):
            sh.text(RX, yy, RW, th + 0.5, t, fnote, INK, valign='t', wrap=True)
            yy += th + 1.0
        y += h + 3.0
    # camera table
    sh.label(_MX, 127.6, 'Cameras', 40)
    _ms_table(sh, ctx, _MX, 131.5)
    _ms_title_block(sh, ctx, 1, pages)

    for k, page in enumerate(sheets):
        out.new_page()
        _ms_header(sh, ctx, cont=True)
        sh.label(_MX, 28.0, f'Issues, continued ({len(overflow)} of {len(issues)})', 160)
        for idx, col, y in page:
            _ms_issue(sh, _MX + col * (colw + 10), y, colw, overflow[idx])
        _ms_title_block(sh, ctx, k + 2, pages)
    return out.finish(), out.pages


# ====================================================================== report (PDF via QTextDocument)
def _report_pdf(ctx: _Ctx) -> tuple[bytes, int]:
    _require_app()
    from PyQt6.QtCore import QRectF, QSizeF, QUrl
    from PyQt6.QtGui import QColor, QFont, QImage, QTextDocument
    sans, _ = _families()
    doc = QTextDocument()
    dev = QImage(8, 8, QImage.Format.Format_RGB32)      # fixed 96 dpi layout device
    dev.setDotsPerMeterX(3780)
    dev.setDotsPerMeterY(3780)
    doc.documentLayout().setPaintDevice(dev)
    f = QFont(sans)
    f.setPointSizeF(9)
    doc.setDefaultFont(f)
    n_img = [0]
    bars = set()

    def src(key, data):
        if key.startswith('bar:'):
            name = f'vbar://{key[5:]}'
            if name not in bars:
                img = QImage(4, 4, QImage.Format.Format_RGB32)
                img.fill(QColor(key[4:]))
                doc.addResource(QTextDocument.ResourceType.ImageResource.value, QUrl(name), img)
                bars.add(name)
            return name, 4, 4
        img = QImage.fromData(data)
        name = f'vimg://{_safe(key)}-{n_img[0]}'
        n_img[0] += 1
        doc.addResource(QTextDocument.ResourceType.ImageResource.value, QUrl(name), img)
        return name, img.width(), img.height()

    html_text = _report_html(ctx, qt=True, img_src=src)
    doc.setHtml(html_text)
    W, H = 794.0, 1123.0                                 # A4 at 96 dpi
    doc.setPageSize(QSizeF(W, H))
    fmt = doc.rootFrame().frameFormat()
    fmt.setLeftMargin(57)
    fmt.setRightMargin(57)
    fmt.setTopMargin(57)
    fmt.setBottomMargin(68)
    doc.rootFrame().setFrameFormat(fmt)
    n = max(doc.pageCount(), 1)

    out = _PdfOut(False, f'Coverage report · {ctx.project.name} · {ctx.variant.name}')
    p = out.painter
    scale = out.writer.width() / W
    sh = _Sheet(out)
    for i in range(n):
        if i:
            out.new_page()
        p.save()
        p.scale(scale, scale)
        p.translate(0, -i * H)
        doc.drawContents(p, QRectF(0, i * H, W, H))
        p.restore()
        fy = 297.0 - 12.0
        sh.line(15.0, fy - 1.5, 195.0, fy - 1.5, HAIR, 0.2)
        sh.text(15.0, fy - 1.0, 140, 4, f'Vantage · {ctx.project.name} · {ctx.variant.name}', sh.font(6.5), MUTED,
                elide=True)
        sh.text(155.0, fy - 1.0, 40, 4, f'Page {i + 1} of {n}', sh.font(6.5), MUTED, align='r')
    return out.finish(), out.pages


# ====================================================================== dispatch
def _image_bytes(ctx: _Ctx, key: str, what: str) -> bytes:
    data = ctx.images.get(key)
    if not data:
        raise ValueError(f'No {what} image to export: the UI did not supply opts.images[{key!r}].')
    data = bytes(data)
    if _png_size(data) is None:
        raise ValueError(f'opts.images[{key!r}] is not PNG data.')
    return data


def _camera_view_files(ctx: _Ctx) -> list:
    tags = safe_tags(ctx.variant)
    out, missing = [], []
    for c in _view_cams(ctx.variant):
        data = ctx.images.get(f'cam:{c.uid}')
        if not data:
            missing.append(c.tag)
            continue
        if _png_size(bytes(data)) is None:
            raise ValueError(f"opts.images['cam:{c.uid}'] ({c.tag}) is not PNG data.")
        out.append((f'views/{tags[c.uid]}.png', bytes(data)))
    if missing:
        raise ValueError(f'No camera view image for {", ".join(missing)}: the UI did not supply '
                         f"opts.images['cam:<uid>'] for {'that camera' if len(missing) == 1 else 'those cameras'}.")
    if not out:
        raise ValueError('No camera views to export: the variant has no visible cameras.')
    return out


def _render(fmt_id: str, ctx: _Ctx) -> list:
    """[(relative path, bytes)] for one format."""
    fmt = format_by_id(fmt_id)
    name = lambda: _names(fmt, ctx.project, ctx.variant)[0]
    t = lambda s: s.encode('utf-8')
    if fmt_id == 'mount_sheet_pdf':
        data, pages = _mount_sheet_pdf(ctx)
        ctx.meta[fmt_id] = {'pages': pages}
        return [(name(), data)]
    if fmt_id == 'report_html':
        return [(name(), t(_report_html_file(ctx)))]
    if fmt_id == 'report_pdf':
        data, pages = _report_pdf(ctx)
        ctx.meta[fmt_id] = {'pages': pages}
        return [(name(), data)]
    if fmt_id == 'extrinsics_json':
        return [(name(), t(_extrinsics_json(ctx)))]
    if fmt_id == 'extrinsics_yaml':
        return [(name(), t(_extrinsics_yaml(ctx)))]
    if fmt_id == 'extrinsics_csv':
        return [(name(), t(_extrinsics_csv(ctx)))]
    if fmt_id == 'opencv_yaml':
        return [(f'opencv/{cm.tag}.yaml', t(_opencv_yaml(ctx, cm))) for cm in ctx.models]
    if fmt_id == 'ros2':
        return [('vantage_cameras.urdf', t(_urdf(ctx))), ('static_tf.launch.py', t(_launch(ctx)))]
    if fmt_id == 'nuscenes_calib':
        return _nuscenes(ctx)
    if fmt_id == 'kitti_calib':
        return _kitti(ctx)
    if fmt_id == 'ros_camera_info':
        return [(f'camera_info/{cm.tag}.yaml', t(_camera_info(ctx, cm))) for cm in ctx.models]
    if fmt_id == 'kalibr_camchain':
        return [(name(), t(_kalibr(ctx)))]
    if fmt_id == 'jetson_json':
        return [(name(), t(_jetson_json(ctx)))]
    if fmt_id in _IMAGE_KEYS:
        key, what = _IMAGE_KEYS[fmt_id]
        return [(name(), _image_bytes(ctx, key, what))]
    if fmt_id == 'camera_views_png':
        return _camera_view_files(ctx)
    if fmt_id == 'bev_lut':
        from .stitch import lut_files
        return [(f'bev_lut/{n}', d) for n, d in lut_files(_lut(ctx), ctx.project, ctx.variant)]
    if fmt_id == 'bev_stitcher':
        from .stitch import runtime_files
        return [(f'bev_lut/stitcher/{n}', d) for n, d in runtime_files()]
    if fmt_id == 'bowl_mesh':
        return [(f'bowl/{n}', d) for n, d in _bowl(ctx)]
    if fmt_id == 'blind_volume':
        return [(n, t(d)) for n, d in _blind_volume(ctx)]
    if fmt_id == 'guidelines':
        return _guidelines(ctx)
    if fmt_id == 'stitched_objects':
        return [(f'stitched_objects/{n}', d) for n, d in _stitched(ctx)]
    if fmt_id == 'perception_range':
        return [(f'perception_range/{n}', d) for n, d in _perception(ctx)]
    if fmt_id == 'sensor_coverage':
        return [(f'sensor_coverage/{n}', d) for n, d in _sensor_coverage(ctx)]
    if fmt_id == 'sync_plan':
        return [(f'sync_plan/{n}', d) for n, d in _sync_plan(ctx)]
    if fmt_id == 'recording_budget':
        from . import recording as RB
        return [(f'recording_budget/{n}', d) for n, d in RB.files(RB.budget(ctx.project, ctx.variant), ctx.project)]
    if fmt_id == 'power_budget':
        from . import power as PW
        return [(f'power_budget/{n}', d) for n, d in PW.files(PW.run(ctx.project, variant=ctx.variant), ctx.project)]
    if fmt_id == 'requirements':
        from . import requirements as RQ
        res = RQ.run(ctx.project, ctx.opts.engine, variant=ctx.variant)
        return [(f'requirements/{n}', d) for n, d in RQ.files(res, ctx.project)]
    if fmt_id == 'brackets':
        from . import brackets as BR
        from .analysis import Engine
        res = BR.run(ctx.project, ctx.opts.engine or Engine(), variant=ctx.variant)
        return BR.files(res, ctx.project)            # names already 'brackets/…'
    if fmt_id == 'fleet':
        from . import fleet as FL
        return [(f'fleet/{n}', d) for n, d in FL.files(FL.run(ctx.project), ctx.project)]
    if fmt_id == 'exposure':
        from . import exposure as XP
        from .analysis import Engine
        res = XP.run(ctx.project, ctx.opts.engine or Engine(), variant=ctx.variant)
        return [(f'exposure/{n}', d) for n, d in XP.files(res, ctx.project)]
    if fmt_id == 'gnss_imu':
        from . import gnss as GN
        return [(f'gnss_imu/{n}', d) for n, d in GN.files(GN.run(ctx.project, ctx.opts.engine, variant=ctx.variant),
                                                          ctx.project)]
    if fmt_id == 'dxf':
        return [(name(), _dxf(ctx).encode('cp1252', errors='replace'))]
    if fmt_id == 'step':
        return [(name(), t(_step(ctx)))]
    raise ValueError(f'Unknown export format {fmt_id!r}')


def _blind_volume(ctx: _Ctx):
    from . import volume as vol
    from .analysis import Engine
    st = vol.VolumeSettings.from_dict(getattr(ctx.project.analysis, 'volume', None))
    vr = vol.analyse_volume(ctx.project, ctx.opts.engine or Engine(), ctx.result, st.height, st.dz, st.cell or None)
    if vr is None:                  # no vehicle model yet: say so instead of failing the other formats
        doc = {'schema': 'vantage.blind_volume/1', 'project': ctx.project.name, 'variant': ctx.variant.name,
               'note': 'No vehicle model: there is no blind volume to analyse.'}
        return [('blind_volume.json', json.dumps(doc, indent=1) + '\n'),
                ('blind_volume.obj', '# Vantage blind volume: no vehicle model\n')]
    return [('blind_volume.json', json.dumps(vol.to_doc(vr, ctx.project), indent=1) + '\n'),
            ('blind_volume.obj', vol.to_obj(vr))]


def _bowl(ctx: _Ctx):
    if 'bowl' in ctx.meta:
        return ctx.meta['bowl']
    if ctx.result is None:
        raise ValueError('The 3D surround-view bowl needs an analysis result.')
    from . import bowl as bowl_core
    from .analysis import Engine
    eng = ctx.opts.engine or Engine()
    shape = ctx.opts.bowl_shape or bowl_core.BowlShape.from_dict(getattr(ctx.project.analysis, 'bowl', None) or {})
    tex = ctx.opts.bowl_texture
    if tex is None:
        from .synth import parking_world, render_capture
        world = parking_world(ctx.project)
        cams = [c for c in ctx.variant.cameras if c.visible]
        images = {c.uid: render_capture(ctx.project, eng, c, 640, 512, world=world, seed=c.slot, noise=1.0) for c in cams}
        tex = bowl_core.bowl_texture(bowl_core.bowl_lut(ctx.project, eng, ctx.result, shape, 1024, 320), images)
    ctx.meta['bowl'] = bowl_core.export_files(ctx.project, eng, ctx.result, shape, tex)
    return ctx.meta['bowl']


def _json_compact(doc, keys=('pixels', 'ground')) -> str:
    """Like _json, with the (long) arrays under keys written on one line each."""
    store = []

    def walk(o):
        if isinstance(o, dict):
            out = {}
            for k, v in o.items():
                if k in keys and isinstance(v, list):
                    store.append(json.dumps(v, separators=(',', ':')))
                    out[k] = f'@@{len(store) - 1}@@'
                else:
                    out[k] = walk(v)
            return out
        if isinstance(o, list):
            return [walk(x) for x in o]
        return o
    return re.sub(r'"@@(\d+)@@"', lambda m: store[int(m.group(1))], _json(walk(doc)))


def _guidelines(ctx: _Ctx):
    """Per visible camera: the parking-guideline table for the ECU (JSON) and a preview of the current guidelines
    over the synthetic parking scene (PNG)."""
    if 'guidelines' in ctx.meta:
        return ctx.meta['guidelines']
    from . import guidelines as G
    from .synth import parking_world
    eng, st = _guideline_setup(ctx)
    tags = safe_tags(ctx.variant)
    world = parking_world(ctx.project)
    files = []
    for c in _view_cams(ctx.variant):
        doc = _guideline_doc(ctx, c, eng, st)
        files.append((f'guidelines/{tags[c.uid]}.json', _json_compact(doc).encode('utf-8')))
        files.append((f'guidelines/{tags[c.uid]}.png',
                      G.preview_png(ctx.project, eng, c, st, list(doc['directions']), world)))
    ctx.meta['guidelines'] = files
    return files


def _guideline_setup(ctx: _Ctx):
    from . import guidelines as G
    from .analysis import Engine
    return ctx.opts.engine or Engine(), G.GuidelineSettings.from_dict(getattr(ctx.project.analysis, 'guidelines', None))


def _guideline_doc(ctx: _Ctx, c: Camera, eng, st) -> dict:
    from .. import __version__
    from . import guidelines as G
    p = ctx.project
    o = G.vehicle_outline(p)
    ox, oy = float(ctx.origin[0]), float(ctx.origin[1])
    table = G.guideline_table(p, eng, c, settings=st, origin=ctx.origin, scale=ctx.scale, dp=ctx.pdp)
    intr = resolve(p.module_for(c), p.lens_for(c))
    return {
        'schema': G.SCHEMA, 'generator': f'Vantage {__version__}', 'created': ctx.stamp,
        'project': p.name, 'variant': ctx.variant.name,
        'camera': {'tag': c.tag, 'name': c.name, 'port': c.port, 'sensor_width': intr.width,
                   'sensor_height': intr.height, 'lens': p.lens_for(c).name, 'unit_calibration': bool(c.calib)},
        'pixel_coordinates': G.PIXEL_TEXT,
        'frame': f'{ctx.frame_label}; origin: {ctx.origin_text}; axes: {AXES_TEXT}', 'units': ctx.units,
        'vehicle': {'wheelbase': ctx.L(o.wheelbase), 'track': ctx.L(o.track), 'width': ctx.L(2 * o.half_width),
                    'front_bumper_x': ctx.L(o.front_x - ox), 'rear_bumper_x': ctx.L(o.rear_x - ox),
                    'rear_axle_x': ctx.L(o.rear_axle_x - ox), 'centre_y': ctx.L(o.centre_y - oy)},
        'steering': {'model': 'kinematic bicycle about the rear axle; turn radius of the rear-axle centre = '
                              'wheelbase / tan(road wheel angle)',
                     'positive': 'turning left (counter-clockwise seen from above)',
                     'road_wheel_angle_max_deg': st.max_deg, 'step_deg': st.step_deg,
                     'steering_ratio': st.ratio or None},
        'guidelines': {'distances': [ctx.L(d) for d in st.distances], 'path_length': ctx.L(st.length),
                       'margin': ctx.L(st.margin_m), 'colors': dict(G.COLORS),
                       'bands': 'red up to the first distance, yellow up to the second, green beyond',
                       'kinds': {'static': 'width lines and distance markers straight back or ahead (no steering)',
                                 'trajectory': 'paths of the bumper corners at this angle',
                                 'distance': 'markers across the trajectory at the travel distances',
                                 'swept': 'outline of the ground the whole body (with the mirrors) passes over'}},
        'directions_note': 'reverse: backing up; forward: driving ahead. Only directions with a guideline visible in '
                           'this camera are listed; a line\'s pixels are its visible pieces.',
        **table,
    }


def _stitched(ctx: _Ctx):
    """stitched_objects.json (map summary, hotspots, seam advice in the export frame and units) and the map PNG."""
    if 'stitched' in ctx.meta:
        return ctx.meta['stitched']
    if ctx.result is None:
        raise ValueError('The objects-in-the-stitched-view map needs an analysis result.')
    from . import stitched_objects as so
    from .analysis import Engine
    eng = ctx.opts.engine or Engine()
    sres = so.analyse(ctx.project, eng, ctx.result)
    doc = so.to_doc(sres, so.advise(ctx.project, eng, ctx.result), L=ctx.L, A=ctx.Ap, origin=ctx.origin)
    head = {'project': ctx.project.name, 'variant': ctx.variant.name, 'created': ctx.stamp, 'frame': ctx.frame,
            'units': ctx.units, 'angles': ctx.angles,
            'azimuth': 'seam azimuths are measured about the body centre, counter-clockwise from +x'}
    doc = {'schema': doc.pop('schema'), **head, **doc}
    ctx.meta['stitched'] = [('stitched_objects.json', _json(doc).encode('utf-8')), ('stitched_objects.png', so.png(sres))]
    return ctx.meta['stitched']


def _perception(ctx: _Ctx):
    """perception_range.json (ranges per direction and object, sector verdicts, stereo pairs in the export frame and
    units) and the plan PNG."""
    if 'perception' in ctx.meta:
        return ctx.meta['perception']
    from . import perception as PR
    from .analysis import Engine
    st = PR.PerceptionSettings.from_dict(getattr(ctx.project.analysis, 'perception', None))
    pr = PR.analyse(ctx.project, ctx.opts.engine or Engine(), ctx.result, st)
    doc = PR.to_doc(pr, ctx.project, L=ctx.L, A=ctx.Ap, origin=ctx.origin)
    head = {'project': ctx.project.name, 'variant': ctx.variant.name, 'created': ctx.stamp, 'frame': ctx.frame,
            'units': ctx.units, 'angles': ctx.angles}
    doc = {'schema': doc.pop('schema'), **head, **{k: v for k, v in doc.items() if k not in ('project', 'variant')}}
    ctx.meta['perception'] = [('perception_range.json', _json(doc).encode('utf-8')),
                              ('perception_range.png', PR.png(pr))]
    return ctx.meta['perception']


def sensor_coverage_doc(ctx: _Ctx, res=None) -> dict:
    """The lidar and radar coverage of the export's variant as JSON, in the export frame, units and angles."""
    from . import sensor_coverage as SC
    from .analysis import Engine
    res = res or SC.run(ctx.project, ctx.opts.engine or Engine(), ctx.variant)
    doc = SC.to_doc(res, ctx.project, L=ctx.L, A=ctx.Ap, origin=ctx.origin)
    head = {'project': ctx.project.name, 'variant': ctx.variant.name, 'created': ctx.stamp, 'frame': ctx.frame,
            'units': ctx.units, 'angles': ctx.angles}
    return {'schema': doc.pop('schema'), **head, **{k: v for k, v in doc.items() if k not in ('project', 'variant')}}


def _sensor_coverage(ctx: _Ctx):
    """sensor_coverage.json (lidar, radar and camera ranges per object and direction, sector verdicts, what the
    vehicle blocks, camera-lidar overlap, ground density) and the plan PNG."""
    if 'sensor_coverage' in ctx.meta:
        return ctx.meta['sensor_coverage']
    from . import sensor_coverage as SC
    from .analysis import Engine
    res = SC.run(ctx.project, ctx.opts.engine or Engine(), ctx.variant)
    ctx.meta['sensor_coverage'] = [('sensor_coverage.json', _json(sensor_coverage_doc(ctx, res)).encode('utf-8')),
                                   ('sensor_coverage.png', SC.png(res))]
    return ctx.meta['sensor_coverage']


def sync_plan_doc(ctx: _Ctx, res=None) -> dict:
    """The sync and timing plan of the export's variant as JSON, in the export frame, units and angles."""
    from . import timing as TM
    res = res or TM.plan(ctx.project, ctx.variant)
    doc = TM.to_doc(res, ctx.project, L=ctx.L, A=ctx.Ap, origin=ctx.origin)
    head = {'project': ctx.project.name, 'variant': ctx.variant.name, 'created': ctx.stamp, 'frame': ctx.frame,
            'units': ctx.units, 'angles': ctx.angles}
    return {'schema': doc.pop('schema'), **head, **{k: v for k, v in doc.items() if k not in ('project', 'variant')}}


def _sync_plan(ctx: _Ctx):
    """sync_plan.json (trigger offsets against the lidar sweep, groups, motion blur) and the dial PNG."""
    from . import timing as TM
    res = TM.plan(ctx.project, ctx.variant)
    return [('sync_plan.json', _json(sync_plan_doc(ctx, res)).encode('utf-8')), ('sync_plan.png', TM.png(res))]


def _lut(ctx: _Ctx):
    if ctx.opts.lut is not None:
        return ctx.opts.lut
    if 'bev_lut' not in ctx.meta:
        if ctx.result is None:
            raise ValueError('The bird\'s-eye remap tables need an analysis result.')
        from .analysis import Engine
        from .stitch import build_lut
        ctx.meta['bev_lut'] = build_lut(ctx.project, ctx.opts.engine or Engine(), ctx.result)
    return ctx.meta['bev_lut']


def _display_dir(opts: ExportOptions, variant: Variant) -> str:
    if opts and opts.out_dir:
        parts = [x for x in os.path.normpath(opts.out_dir).replace('\\', '/').split('/') if x]
        return '/'.join(parts[-2:]) if parts else '.'
    return f'exports/{_slug(variant.name)}'


def _note(path: str, text: str) -> str:
    if path.endswith(('.urdf', '.xml', '.html')):
        return f'<!-- {text} -->'
    return f'# {text}'


def _size_text(n: int) -> str:
    if n < 1024:
        return f'{n} B'
    if n < 1024 * 1024:
        return f'{n / 1024:.0f} KB'
    return f'{n / 1024 / 1024:.1f} MB'


def preview(fmt_id: str, project: Project, result, opts: ExportOptions) -> tuple[str, str, int]:
    """(relative path of the first file, exact text or a short description, size in bytes)."""
    fmt = format_by_id(fmt_id)
    ctx = _context(project, result, opts)
    base = _display_dir(ctx.opts, ctx.variant)
    names = _names(fmt, project, ctx.variant)
    first = f'{base}/{names[0]}' if names else f'{base}/'

    if fmt.kind == 'text':
        files = _render(fmt_id, ctx)
        if not files:
            return first, 'No cameras in this variant.', 0
        total = sum(len(b) for _, b in files)
        text = files[0][1].decode('utf-8')
        if len(files) > 1:
            rest = [n for n, _ in files[1:]]
            shown = ', '.join(rest[:6]) + (f' and {len(rest) - 6} more' if len(rest) > 6 else '')
            text += '\n' + _note(files[0][0], f'Preview shows {files[0][0]} only. Also written: {shown} '
                                              f'({len(files)} files, {total} bytes in all).') + '\n'
        return f'{base}/{files[0][0]}', text, total

    if fmt_id in _IMAGE_KEYS or fmt_id == 'camera_views_png':
        if fmt_id in _IMAGE_KEYS:
            key, what = _IMAGE_KEYS[fmt_id]
            data = ctx.images.get(key)
            if not data:
                return first, f'Not available: no {what} was supplied. Export will fail until there is one.', 0
            sz = _png_size(bytes(data))
            dims = f'{sz[0]} × {sz[1]} px' if sz else 'not PNG data'
            return first, f'PNG · {dims} · {_size_text(len(data))}', len(data)
        cams = _view_cams(ctx.variant)
        have = [(c, ctx.images.get(f'cam:{c.uid}')) for c in cams]
        missing = [c.tag for c, d in have if not d]
        total = sum(len(d) for _, d in have if d)
        sizes = {_png_size(bytes(d)) for _, d in have if d}
        dims = ' · ' + (f'{next(iter(sizes))[0]} × {next(iter(sizes))[1]} px each' if len(sizes) == 1 and
                        None not in sizes else 'mixed sizes') if total else ''
        desc = f'{len(cams)} PNG files (one per visible camera){dims}'
        if missing:
            desc += f' · missing: {", ".join(missing)}; export will fail until every view is supplied'
        return first, desc + (f' · {_size_text(total)}' if total else ''), total

    if fmt_id == 'bowl_mesh':
        if ctx.result is None:
            return first, 'Not available: the bowl needs an analysis result.', 0
        files = _render(fmt_id, ctx)
        total = sum(len(d) for _, d in files)
        meta = json.loads(dict(files)['bowl/bowl_lut.json'])
        return (f'{base}/bowl/bowl_lut.json', f'{len(files)} files · {meta["vertices"]} vertices, {meta["faces"]} triangles · '
                f'{len(meta["cameras"])} cameras · {_size_text(total)}\n\n' + dict(files)['bowl/bowl_lut.json'].decode('utf-8'),
                total)
    if fmt_id == 'guidelines':
        cams = _view_cams(ctx.variant)
        if not cams:
            return first, 'No visible cameras in this variant.', 0
        # only the first table: the previews render a parking scene through every camera, too slow to show live
        eng, st = _guideline_setup(ctx)
        doc = _guideline_doc(ctx, cams[0], eng, st)
        data = _json_compact(doc).encode('utf-8')
        head = {k: v for k, v in doc.items() if k not in ('static', 'directions')}
        dirs = ', '.join(f'{d} ({len(rows)} steering angles)' for d, rows in doc['directions'].items()) or \
            'no guideline in view'
        return (first, f'{2 * len(cams)} files: a table (JSON) and a preview (PNG) per visible camera · {names[0]}: '
                f'{dirs} · {_size_text(len(data))}\n\n' + _json(head), len(data))
    if fmt_id == 'stitched_objects':
        if ctx.result is None:
            return first, 'Not available: the map needs an analysis result.', 0
        files = _render(fmt_id, ctx)
        total = sum(len(d) for _, d in files)
        doc = json.loads(dict(files)['stitched_objects/stitched_objects.json'])
        m = doc['metrics']
        return (f'{base}/stitched_objects/stitched_objects.json',
                f'{doc["object"]["label"]}: vanishes on {m["vanish_m2"]} m², cut off on {m["cut_m2"]} m², '
                f'{len(doc["hotspots"])} hotspots · {_size_text(total)}\n\n'
                + dict(files)['stitched_objects/stitched_objects.json'].decode('utf-8'), total)
    if fmt_id == 'perception_range':
        files = _render(fmt_id, ctx)
        total = sum(len(d) for _, d in files)
        doc = json.loads(dict(files)['perception_range/perception_range.json'])
        passed = sum(v['pass'] for s in doc['sectors'] for v in s['classes'].values())
        checks = sum(len(s['classes']) for s in doc['sectors'])
        head = {k: v for k, v in doc.items() if k not in ('azimuth', 'per_class', 'per_camera')}
        return (f'{base}/perception_range/perception_range.json',
                f'{len(doc["classes"])} objects × {len(doc["azimuth"])} directions · {passed} of {checks} sector checks '
                f'pass · {len(doc["stereo"]["pairs"])} overlapping camera pairs · {_size_text(total)}\n\n'
                f'(the per-direction ranges are left out of this preview)\n' + _json(head), total)
    if fmt_id == 'sensor_coverage':
        files = _render(fmt_id, ctx)
        total = sum(len(d) for _, d in files)
        doc = json.loads(dict(files)['sensor_coverage/sensor_coverage.json'])
        head = {k: v for k, v in doc.items() if k not in ('azimuth', 'per_class', 'ground_density', 'lidars', 'radars',
                                                          'ground_blind_from_body')}
        verdicts = [c['verdict'] for s in doc['sectors'] for c in s['classes'].values()] \
            if doc.get('sectors') and isinstance(doc['sectors'][0].get('classes'), dict) else []
        counts = ', '.join(f'{verdicts.count(v)} {v}' for v in ('redundant', 'single', 'gap')) if verdicts else ''
        return (f'{base}/sensor_coverage/sensor_coverage.json',
                f'{len(doc.get("lidars", []))} lidars, {len(doc.get("radars", []))} radars'
                + (f' · sector checks: {counts}' if counts else '') + f' · {_size_text(total)}\n\n'
                f'(the per-direction ranges, per-lidar figures and the ground density are left out of this preview)\n'
                + _json(head), total)
    if fmt_id == 'bev_lut':
        if ctx.result is None and ctx.opts.lut is None:
            return first, 'Not available: the remap tables need an analysis result.', 0
        files = _render(fmt_id, ctx)
        lut = _lut(ctx)
        total = sum(len(d) for _, d in files)
        return (f'{base}/bev_lut/bev_lut.json',
                f'{len(files)} files · {lut.nx} × {lut.ny} px at {lut.res * 1000:.0f} mm/px · {len(lut.uids)} cameras · '
                f'{_size_text(total)}\n\n' + files[0][1].decode('utf-8'), total)
    data = _render(fmt_id, ctx)[0][1]
    pages = ctx.meta.get(fmt_id, {}).get('pages', 1)
    issues, _ = ctx.issues()
    ni = len(issues)
    iss = f'{ni} issue' + ('' if ni == 1 else 's')
    pg = f'{pages} page' + ('' if pages == 1 else 's')
    if fmt_id == 'mount_sheet_pdf':
        desc = f'A4 landscape · {pg} · camera table, plan view, {iss}'
    else:
        nimg = len(_image_list(ctx))
        nc = len(ctx.variant.cameras)
        parts = (['metrics'] if ctx.result is not None else ['no analysis result'])
        parts += [f'{nc} camera' + ('' if nc == 1 else 's')]
        parts += (['seams', 'density histogram'] if ctx.result is not None else []) + [iss]
        if nimg:
            parts.append(f'{nimg} image' + ('' if nimg == 1 else 's'))
        desc = f'A4 portrait · {pg} · ' + ', '.join(parts)
    return first, f'{desc} · {_size_text(len(data))}', len(data)


def export(fmt_ids, project: Project, result, opts: ExportOptions, progress=None) -> list[str]:
    """Write the formats into opts.out_dir; returns the absolute paths written, in order."""
    ids = [fmt_ids] if isinstance(fmt_ids, str) else list(fmt_ids)
    for fid in ids:
        format_by_id(fid)
    ctx = _context(project, result, opts)
    out_dir = os.path.abspath(ctx.opts.out_dir or default_out_dir(project, ctx.variant))
    # fail before writing anything when a UI-supplied image is missing
    for fid in ids:
        if fid in _IMAGE_KEYS:
            _image_bytes(ctx, *_IMAGE_KEYS[fid])
        elif fid == 'camera_views_png':
            _camera_view_files(ctx)
        elif fid == 'bev_lut':
            _lut(ctx)
        elif fid == 'bowl_mesh':
            _bowl(ctx)
        elif fid == 'stitched_objects':
            _stitched(ctx)
        elif fid == 'perception_range':
            _perception(ctx)
        elif fid == 'sensor_coverage':
            _sensor_coverage(ctx)
    written = []
    for i, fid in enumerate(ids):
        for rel, data in _render(fid, ctx):
            path = os.path.join(out_dir, *rel.split('/'))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = path + '.part'
            with open(tmp, 'wb') as fh:
                fh.write(data)
            os.replace(tmp, path)
            written.append(path)
        if progress:
            progress((i + 1) / len(ids))
    return written
