"""The dataset calibration exports (KITTI, Kalibr, ROS camera_info, nuScenes) checked against the tools that read them.

The offline tests always run, with no network and no extra packages. Each one reproduces what a tool does with the
files from the tool's own source, reimplemented here (nothing is copied from the tools, and nothing from Vantage's own
exporter maths): the KITTI raw devkit's MATLAB readers (loadCalibrationCamToCam.m / loadCalibrationRigid.m, 2013),
pykitti 0.3.1's raw calibration loading, the KITTI object devkit and the line-indexed readers of mmdetection3d and
OpenPCDet, Kalibr's ConfigReader.py checks (ethz-asl/kalibr 1f60227) and its aslam_cv camera models (pinhole with
equidistant or radtan, omni (Mei & Rives 2007), double sphere (Usenko et al. 2018), extended unified (Khomutenko et al.
2016)), ROS 2 camera_calibration_parsers' readCalibrationYml and the C++ image_geometry PinholeCameraModel
(image_common faf0db2, vision_opencv 50f6f1f), and the URDF / REP 103 conventions. Projections use OpenCV's own
cv2.fisheye and cv2.projectPoints where a model is OpenCV's.

Ground truth is Vantage's lens model (core.lens Intrinsics.project_dirs) and the rig geometry (mathutil.rot_zyx,
BODY_TO_OPTICAL), so a file is right when the tool's reading of it puts every point where Vantage does.

The optional tests (skipped unless VANTAGE_DEVKITS points at a folder holding them) run the tools themselves:
  python/pykitti           pykitti 0.3.1: pykitti.raw on a KITTI raw drive folder built from the export
  python/nuscenes          nuScenes devkit 1.2.0 + pyquaternion: its geometry_utils.view_points
  kalibr/ConfigReader.py   Kalibr's config reader, with stubs for its compiled aslam_cv / sm modules only
  ros/parse_yml_check      camera_calibration_parsers' parse_yml.cpp built with yaml-cpp (prints what it read)
  ros/cameramodels_*.py    image_geometry's Python PinholeCameraModel (humble, rolling)
Run with:  QT_QPA_PLATFORM=offscreen python3 -m pytest tests/test_devkits.py -q
"""
from __future__ import annotations

import datetime as dt
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import types
import xml.etree.ElementTree as ET

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtGui import QGuiApplication  # noqa: E402

from vantage.core import calib  # noqa: E402
from vantage.core import exporters as ex  # noqa: E402
from vantage.core import sensors as SN  # noqa: E402
from vantage.core.lens import Lens, resolve  # noqa: E402
from vantage.core.mathutil import BODY_TO_OPTICAL, rot_zyx  # noqa: E402
from vantage.core.project import Sensor  # noqa: E402
from vantage.core.samples import tigor_sample_project  # noqa: E402

cv2 = pytest.importorskip('cv2')

FIXED_NOW = dt.datetime(2026, 10, 21, 9, 30, 0, tzinfo=dt.timezone(dt.timedelta(hours=5, minutes=30)))
IDS = ['nuscenes_calib', 'kitti_calib', 'ros_camera_info', 'kalibr_camchain', 'ros2']
ORIGIN = np.array([1.0, 0.2, 0.3])          # a custom export origin: every file must honour it
TOL_PX = 0.01                               # what file round-off may cost (KITTI keeps 7 significant digits)
DEVKITS = os.environ.get('VANTAGE_DEVKITS', '')


def _kit(*parts):
    path = os.path.join(DEVKITS, *parts) if DEVKITS else ''
    return path if path and os.path.exists(path) else ''


def needs(*parts):
    return pytest.mark.skipif(not _kit(*parts), reason=f'set VANTAGE_DEVKITS to a folder with {"/".join(parts)}')


# ----------------------------------------------------------------------------- fixtures
@pytest.fixture(scope='module')
def qapp():
    yield QGuiApplication.instance() or QGuiApplication([])


@pytest.fixture(autouse=True)
def fixed_clock(monkeypatch):
    monkeypatch.setattr(ex, '_now', lambda: FIXED_NOW)


def _data_rig():
    p = tigor_sample_project()
    v = next(x for x in p.variants if x.sensors)
    p.active_uid = v.uid
    return p, v


def _add(p, v, kind, mid, **turn):
    m = p.sensor_models[mid]
    pose = SN.default_pose(kind, m, p.vehicle.mesh, p.rig, v.sensors)
    pose.update(turn)
    s = Sensor(kind=kind, model_id=mid, tag=SN.next_tag(kind, v.tags()), name=kind, **pose)
    v.sensors.append(s)
    return s


@pytest.fixture(scope='module')
def rig(qapp):
    """The sample's data rig (6 cameras, roof lidar, 5 radars) with an IMU turned and tilted against the vehicle (so
    its axes are not the vehicle's), a GNSS antenna, one unit-calibrated camera (fx != fy, offset principal point) and
    one rectilinear camera, exported in a custom frame."""
    p, v = _data_rig()
    _add(p, v, 'imu', 'imu_mems', yaw=90.0, pitch=-3.0, roll=2.0)
    _add(p, v, 'gnss', 'gnss_survey')
    cams = v.cameras
    base = ex.camera_model(p, cams[1])
    cams[1].calib = calib.make_record(1920, 1536, base.fx * 1.004, base.fx * 1.011, 963.2, 761.7,
                                      [base.k[0] + 0.002, base.k[1] - 0.004, base.k[2], base.k[3]],
                                      rms_px=0.21, serial='SN-R-0042', source='test')
    p.lenses['rect'] = Lens(id='rect', name='RECT60', focal_mm=4.2, projection='rectilinear',
                            sheet_fov=(60.0, 48.0, 75.0))
    cams[4].lens_id = 'rect'
    p.custom_origin = list(ORIGIN)
    return p


@pytest.fixture(scope='module')
def exported(rig, tmp_path_factory):
    out = tmp_path_factory.mktemp('devkits')
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ex, '_now', lambda: FIXED_NOW)
        paths = ex.export(IDS, rig, None, ex.ExportOptions(out_dir=str(out), frame='custom'))
    return out, paths


# lenses past 90° (no pinhole model reaches them), and one no Kalibr model represents
WIDE = {'fe200': Lens(id='fe200', name='FE200', focal_mm=1.4, projection='equidistant', sheet_fov=(0, 0, 200.0)),
        'es190': Lens(id='es190', name='ES190', focal_mm=1.6, projection='equisolid', sheet_fov=(0, 0, 190.0)),
        'st180': Lens(id='st180', name='ST180', focal_mm=1.2, projection='stereographic', sheet_fov=(0, 0, 180.0)),
        'kbodd': Lens(id='kbodd', name='KBODD', focal_mm=1.6, projection='kannala_brandt', kb=(0.05, -0.2, 0.05, 0.0),
                      max_half_deg=100.0),
        'es150': Lens(id='es150', name='ES150', focal_mm=2.0, projection='equisolid', sheet_fov=(0, 0, 150.0))}


@pytest.fixture(scope='module')
def wide(qapp, tmp_path_factory):
    p, v = _data_rig()
    p.lenses.update(WIDE)
    for c, lid in zip(v.cameras, ['fe200', 'es190', 'st180', 'kbodd', 'es150', 'h03r2']):
        c.lens_id = lid
    out = tmp_path_factory.mktemp('wide')
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ex, '_now', lambda: FIXED_NOW)
        ex.export(['kalibr_camchain', 'ros_camera_info', 'kitti_calib', 'nuscenes_calib'], p, None,
                  ex.ExportOptions(out_dir=str(out)))
    return p, out


@pytest.fixture(scope='module')
def stereo(qapp, tmp_path_factory):
    """KITTI's own layout: four forward cameras sharing one orientation, camera 01 0.54 m right of camera 00, camera
    02 0.06 m left of it and camera 03 0.54 m right of 02, a roof lidar and an IMU."""
    p, v = _data_rig()
    v.cameras = v.cameras[:4]
    y0 = 0.27
    for c, dy, lid in zip(v.cameras, (0.0, -0.54, 0.06, 0.06 - 0.54), ('h03r2', 'h03r2', 'h05r2', 'h05r2')):
        c.x, c.y, c.z, c.yaw, c.pitch, c.roll, c.lens_id = 1.6, y0 + dy, 1.65, 0.0, 4.0, 0.0, lid
    v.sensors = [s for s in v.sensors if s.kind == 'lidar']
    _add(p, v, 'imu', 'ins_dual', yaw=1.5)
    out = tmp_path_factory.mktemp('stereo')
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ex, '_now', lambda: FIXED_NOW)
        ex.export(['kitti_calib'], p, None, ex.ExportOptions(out_dir=str(out)))
    return p, out


# ----------------------------------------------------------------------------- ground truth (Vantage's geometry)
def _intr(p, cam):
    return resolve(p.module_for(cam), p.lens_for(cam))


def _W2O(cam):
    return BODY_TO_OPTICAL @ rot_zyx(cam.yaw, cam.pitch, cam.roll).T


def lens_points(p, cam, n=2500, seed=0):
    """3D points in the camera's optical frame over everything the lens images (directions uniform in solid angle
    inside the image circle and inside the image, 0.5 to 30 m away) and their OpenCV pixels through Vantage's lens."""
    intr = _intr(p, cam)
    rng = np.random.default_rng(seed)
    m = 8 * n
    th = np.arccos(1 - rng.uniform(0, 1, m) * (1 - math.cos(intr.max_theta)))
    ph = rng.uniform(0, 2 * math.pi, m)
    d = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], axis=1)
    u, v, _, ok = intr.project_dirs(d)
    idx = np.flatnonzero(ok)[:n]
    P = d[idx] * rng.uniform(0.5, 30.0, len(idx))[:, None]
    return P, np.stack([u[idx], v[idx]], axis=1) - 0.5        # Vantage puts pixel centres on i + 0.5, OpenCV on i


def to_cam(cam, P_world):
    return (np.asarray(P_world, float) - cam.pos()) @ _W2O(cam).T


def sensor_T(s):
    """4x4 sensor frame -> vehicle frame (rear axle), from the sensor's yaw/pitch/roll like a camera's body."""
    T = np.eye(4)
    T[:3, :3] = rot_zyx(s.yaw, s.pitch, s.roll)
    T[:3, 3] = s.pos()
    return T


def cam_T(cam):
    """4x4 optical frame -> vehicle frame (rear axle)."""
    T = np.eye(4)
    T[:3, :3] = _W2O(cam).T
    T[:3, 3] = cam.pos()
    return T


def pinhole(K, X):
    h = np.asarray(X, float) @ np.asarray(K, float).T
    return h[:, :2] / h[:, 2:]


def errs(a, b):
    e = np.linalg.norm(np.asarray(a) - np.asarray(b), axis=1)
    return float(np.sqrt(np.mean(e * e))), float(e.max())


# ----------------------------------------------------------------------------- the tools, reimplemented
def kitti_matlab_read(path, name, M, N):
    """The KITTI raw devkit's readVariable (loadCalibrationCamToCam.m, loadCalibrationRigid.m): scan the file's
    whitespace-separated words for '<name>:', then read M x N numbers in row order (across lines); any word that is
    not a number empties the result."""
    words = open(path, encoding='utf-8').read().split()
    try:
        i = words.index(name + ':')
    except ValueError:
        return None
    vals = []
    for w in words[i + 1:i + 1 + M * N]:
        try:
            vals.append(float(w))
        except ValueError:
            return None
    return np.array(vals).reshape(M, N) if len(vals) == M * N else None


def kitti_matlab_cam_to_cam(path):
    """loadCalibrationCamToCam.m: cameras 00, 01, ... until one lacks S, K, D, R or T."""
    cams = []
    for c in range(100):
        n = f'{c:02d}'
        got = {k: kitti_matlab_read(path, f'{k}_{n}', *s) for k, s in
               (('S', (1, 2)), ('K', (3, 3)), ('D', (1, 5)), ('R', (3, 3)), ('T', (3, 1)), ('S_rect', (1, 2)),
                ('R_rect', (3, 3)), ('P_rect', (3, 4)))}
        if any(got[k] is None for k in ('S', 'K', 'D', 'R', 'T')):
            break
        cams.append(got)
    return kitti_matlab_read(path, 'corner_dist', 1, 1), cams


def pykitti_read_calib_file(path):
    """pykitti.utils.read_calib_file: 'key: values' per line, keys whose values are not all numbers dropped."""
    data = {}
    with open(path) as fh:
        for line in fh.readlines():
            key, value = line.split(':', 1)
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass
    return data


def _rt(R, t):
    return np.vstack((np.hstack([np.reshape(R, (3, 3)), np.reshape(t, (3, 1))]), [0, 0, 0, 1]))


def pykitti_raw_calib(folder):
    """pykitti.raw._load_calib: KITTI's stereo reading of the raw calibration (rectified cameras parallel to camera 0,
    each P_rect = K [I | t] with the baseline in t)."""
    rd = lambda name: pykitti_read_calib_file(os.path.join(folder, name))
    imu, velo, cc = rd('calib_imu_to_velo.txt'), rd('calib_velo_to_cam.txt'), rd('calib_cam_to_cam.txt')
    out = {'T_velo_imu': _rt(imu['R'], imu['T'])}
    T0 = _rt(velo['R'], velo['T'])
    R00 = np.eye(4)
    R00[:3, :3] = cc['R_rect_00'].reshape(3, 3)
    for i in range(4):
        P = cc[f'P_rect_0{i}'].reshape(3, 4)
        Ti = np.eye(4)
        Ti[0, 3] = P[0, 3] / P[0, 0]
        out[f'T_cam{i}_velo'] = Ti @ R00 @ T0
        out[f'K_cam{i}'] = P[:3, :3]
        out[f'T_cam{i}_imu'] = out[f'T_cam{i}_velo'] @ out['T_velo_imu']
    o = [np.linalg.inv(out[f'T_cam{i}_velo']) @ [0, 0, 0, 1] for i in range(4)]
    out['b_gray'], out['b_rgb'] = np.linalg.norm(o[1] - o[0]), np.linalg.norm(o[3] - o[2])
    return out


def kitti_object_readers(path):
    """The KITTI object calib as three tools read it, by line position: mmdetection3d's kitti_converter
    (split(' ')[1:13]), OpenPCDet's calibration_kitti.get_calib_from_file (lines 2-5) and the object devkit's
    readCalibration.m (dlmread(file, ' ', 0, 1), row cam + 1 reshaped row-major)."""
    lines = open(path).readlines()
    mm = {'P%d' % i: np.array([float(x) for x in lines[i].split(' ')[1:13]]).reshape(3, 4) for i in range(4)}
    mm['R0_rect'] = np.array([float(x) for x in lines[4].split(' ')[1:10]]).reshape(3, 3)
    mm['Tr_velo_to_cam'] = np.array([float(x) for x in lines[5].split(' ')[1:13]]).reshape(3, 4)
    mm['Tr_imu_to_velo'] = np.array([float(x) for x in lines[6].split(' ')[1:13]]).reshape(3, 4)
    pcd = {'P2': np.array(lines[2].strip().split(' ')[1:], np.float32).reshape(3, 4),
           'R0': np.array(lines[4].strip().split(' ')[1:], np.float32).reshape(3, 3),
           'Tr_velo2cam': np.array(lines[5].strip().split(' ')[1:], np.float32).reshape(3, 4)}
    rows = [[float(x) for x in ln.rstrip('\n').split(' ')[1:]] for ln in lines]
    dev = {cam: np.array(rows[cam][:12]).reshape((4, 3), order='F').T for cam in range(4)}   # MATLAB's reshape
    return mm, pcd, dev


def _ext(M):
    out = np.eye(4)
    out[:M.shape[0], :M.shape[1]] = M
    return out


# Kalibr (ConfigReader.py): the models AslamCamera builds and the parameter checks of CameraParameters,
# CameraChainParameters and ImuParameters
KALIBR_CAMERAS = {'pinhole': 4, 'omni': 5, 'ds': 6, 'eucm': 6}
KALIBR_DISTORTIONS = {'radtan': 4, 'equidistant': 4, 'fov': 1, 'none': 0}
KALIBR_PAIRS = {('pinhole', 'radtan'), ('pinhole', 'equidistant'), ('pinhole', 'fov'), ('omni', 'radtan'),
                ('omni', 'none'), ('eucm', 'none'), ('ds', 'none')}   # pinhole-none: AslamCamera sets no geometry


def kalibr_check_camera(c):
    model, q = c['camera_model'], c['intrinsics']
    assert model in KALIBR_CAMERAS and len(q) == KALIBR_CAMERAS[model]
    f, pp = {'pinhole': (q[0:2], q[2:4]), 'omni': (q[1:3], q[3:5]), 'ds': (q[2:4], q[4:6]),
             'eucm': (q[2:4], q[4:6])}[model]
    if model == 'omni':
        assert q[0] >= 0
    if model == 'ds':
        assert 0 <= q[1] < 1
    if model == 'eucm':
        assert 0 <= q[0] < 1 and q[1] >= 0
    assert all(isinstance(x, float) and x >= 0 for x in list(f) + list(pp)), 'Kalibr wants floats'
    assert c['distortion_model'] in KALIBR_DISTORTIONS
    assert len(c['distortion_coeffs']) == KALIBR_DISTORTIONS[c['distortion_model']]
    assert (model, c['distortion_model']) in KALIBR_PAIRS
    r = c['resolution']
    assert len(r) == 2 and all(isinstance(x, int) for x in r)
    assert isinstance(c['rostopic'], str)


def kalibr_project(c, P):
    """aslam_cv's euclideanToKeypoint for a camchain camera, from the models' published equations: pixels and
    validity (the image-bounds check aside)."""
    P = np.asarray(P, float)
    x, y, z = P.T
    q, k = [float(a) for a in c['intrinsics']], [float(a) for a in c['distortion_coeffs']]
    m = c['camera_model']
    if m == 'pinhole':                       # x/z, y/z
        fu, fv, cu, cv_ = q
        valid = z > 0
        mx, my = x / z, y / z
    elif m == 'omni':                        # Mei & Rives: the unit sphere, shifted by xi
        xi, fu, fv, cu, cv_ = q
        n = np.linalg.norm(P, axis=1)
        valid = z / n > -(xi if xi <= 1 else 1 / xi)
        mx, my = x / (z + xi * n), y / (z + xi * n)
    elif m == 'ds':                          # Usenko, Demmel & Cremers 2018, eqs. (40)-(45)
        xi, al, fu, fv, cu, cv_ = q
        d1 = np.linalg.norm(P, axis=1)
        d2 = np.sqrt(x * x + y * y + (xi * d1 + z) ** 2)
        w1 = al / (1 - al) if al <= 0.5 else (1 - al) / al
        w2 = (w1 + xi) / np.sqrt(2 * w1 * xi + xi * xi + 1)
        valid = z > -w2 * d1
        den = al * d2 + (1 - al) * (xi * d1 + z)
        mx, my = x / den, y / den
    else:                                    # extended unified (Khomutenko, Garcia & Martinet 2016)
        al, be, fu, fv, cu, cv_ = q
        d = np.sqrt(be * (x * x + y * y) + z * z)
        w = al / (1 - al) if al <= 0.5 else (1 - al) / al
        valid = z > -w * d
        den = al * d + (1 - al) * z
        mx, my = x / den, y / den
    if c['distortion_model'] == 'equidistant':
        r = np.hypot(mx, my)
        th = np.arctan(r)
        thd = th * (1 + k[0] * th ** 2 + k[1] * th ** 4 + k[2] * th ** 6 + k[3] * th ** 8)
        s = np.where(r > 1e-8, thd / np.where(r > 1e-8, r, 1), 1.0)
        mx, my = mx * s, my * s
    elif c['distortion_model'] == 'radtan':
        k1, k2, p1, p2 = k
        r2 = mx * mx + my * my
        rad = k1 * r2 + k2 * r2 * r2
        mx, my = (mx + mx * rad + 2 * p1 * mx * my + p2 * (r2 + 2 * mx * mx),
                  my + my * rad + 2 * p2 * mx * my + p1 * (r2 + 2 * my * my))
    return np.stack([fu * mx + cu, fv * my + cv_], axis=1), valid


def rigid_ok(T, tol=1e-8):
    T = np.asarray(T, float)
    R = T[:3, :3]
    return (T.shape == (4, 4) and np.allclose(T[3], [0, 0, 0, 1]) and np.allclose(R @ R.T, np.eye(3), atol=tol)
            and abs(np.linalg.det(R) - 1) < tol)


# ROS 2 camera_calibration_parsers readCalibrationYml (parse_yml.cpp) and the distortion models image_geometry handles
ROS_MODELS = ('plumb_bob', 'rational_polynomial', 'equidistant')


def ros_read_calibration_yml(text):
    doc = yaml.safe_load(text)
    out = {'camera_name': str(doc.get('camera_name', 'unknown')), 'width': int(doc['image_width']),
           'height': int(doc['image_height'])}
    for key, name, shape in (('k', 'camera_matrix', (3, 3)), ('r', 'rectification_matrix', (3, 3)),
                             ('p', 'projection_matrix', (3, 4))):
        node = doc[name]
        assert (node['rows'], node['cols']) == shape, name               # parse_yml asserts the sizes
        out[key] = np.array([float(node['data'][i]) for i in range(shape[0] * shape[1])]).reshape(shape)
    out['distortion_model'] = doc.get('distortion_model', 'plumb_bob')
    dn = doc['distortion_coefficients']
    out['d'] = np.array([float(dn['data'][i]) for i in range(dn['rows'] * dn['cols'])])
    out['binning'] = (int(doc.get('binning_x', 0)), int(doc.get('binning_y', 0)))
    if 'roi' in doc:
        r = doc['roi']
        out['roi'] = (int(r['x_offset']), int(r['y_offset']), int(r['height']), int(r['width']), bool(r['do_rectify']))
    return out


def urdf_rpy(r, p, y):
    """URDF <origin rpy>: fixed-axis roll about x, pitch about y, yaw about z: R = Rz(y) Ry(p) Rx(r)."""
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def quat_xyzw(x, y, z, w):
    n = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


# ============================================================================= KITTI
def test_kitti_matlab_devkit_reads_every_camera(rig, exported):
    """loadCalibrationCamToCam.m finds all six cameras with KITTI's shapes; its token scan is not confused by the
    note lines (no note holds a '<key>:' word); loadCalibrationRigid.m reads both rigid files."""
    out, _ = exported
    cams = rig.variant.cameras
    corner, got = kitti_matlab_cam_to_cam(out / 'kitti/calib_cam_to_cam.txt')
    assert corner is not None and math.isclose(float(corner[0, 0]), rig.target_settings.square_mm / 1000.0)
    assert len(got) == len(cams)
    py = pykitti_read_calib_file(out / 'kitti/calib_cam_to_cam.txt')
    for i, g in enumerate(got):
        assert all(g[k] is not None for k in ('S_rect', 'R_rect', 'P_rect')), i      # rectification present
        for k, v in g.items():
            assert np.allclose(v.ravel(), py[f'{k}_{i:02d}']), (k, i)                  # MATLAB and pykitti agree
    for name in ('calib_velo_to_cam.txt', 'calib_imu_to_velo.txt'):
        R, T = kitti_matlab_read(out / 'kitti' / name, 'R', 3, 3), kitti_matlab_read(out / 'kitti' / name, 'T', 3, 1)
        assert R is not None and T is not None and rigid_ok(_rt(R, T), 1e-5), name
    text = (out / 'kitti/calib_cam_to_cam.txt').read_text()
    keys = {ln.split(':', 1)[0] for ln in text.splitlines()}
    for ln in text.splitlines():                     # past its own key, no word of a line is '<a key>:'
        assert not any(w.endswith(':') and w[:-1] in keys for w in ln.split()[1:]), ln[:40]


def test_kitti_devkit_projection_holds_for_every_camera(rig, exported):
    """The devkit's Y = P_rect_xx R_rect_00 (R|T)_velo_to_cam X (and with (R|T)_imu_to_velo for IMU points) lands
    lidar and IMU points where camera xx's rectified pinhole image has them, for every camera of the surround rig."""
    out, _ = exported
    v = rig.variant
    lid = next(s for s in v.sensors if s.kind == 'lidar')
    imu = next(s for s in v.sensors if s.kind == 'imu')
    _, cams = kitti_matlab_cam_to_cam(out / 'kitti/calib_cam_to_cam.txt')
    k = lambda n: (kitti_matlab_read(out / 'kitti' / n, 'R', 3, 3), kitti_matlab_read(out / 'kitti' / n, 'T', 3, 1))
    velo, imu_velo = _rt(*k('calib_velo_to_cam.txt')), _rt(*k('calib_imu_to_velo.txt'))
    R00 = _ext(cams[0]['R_rect'])
    rng = np.random.default_rng(3)
    for i, cam in enumerate(v.cameras):
        K = cams[i]['K']
        # points in front of camera i, expressed in the lidar's and in the IMU's frame
        Pc = np.c_[rng.uniform(-3, 3, (40, 2)), rng.uniform(2, 25, 40)]
        Pw = Pc @ _W2O(cam) + cam.pos()
        for T_s, sensor in ((velo, lid), (velo @ imu_velo, imu)):
            Xs = np.linalg.inv(sensor_T(sensor)) @ np.c_[Pw, np.ones(len(Pw))].T
            Y = cams[i]['P_rect'] @ R00 @ T_s @ Xs
            assert np.abs(pinhole(K, Pc) - (Y[:2] / Y[2]).T).max() < TOL_PX, (cam.tag, sensor.tag)
        assert np.allclose(cams[i]['R_rect'], np.eye(3)) and np.allclose(cams[i]['S_rect'], cams[i]['S'])


def test_kitti_rectification_remap_matches_the_lens(rig, exported):
    """The README's remap (cv2.fisheye.initUndistortRectifyMap with K_xx, D_xx, R_rect_xx, K_xx again as the new
    camera matrix, and S_rect_xx) sends every rectified pixel to the raw pixel Vantage's lens images its ray at; a
    rectilinear lens needs none (D = 0, plumb_bob)."""
    out, _ = exported
    data = pykitti_read_calib_file(out / 'kitti/calib_cam_to_cam.txt')
    raw = dict(ln.split(': ', 1) for ln in (out / 'kitti/calib_cam_to_cam.txt').read_text().splitlines())
    readme = (out / 'kitti/README.txt').read_text()
    assert 'cv2.fisheye.initUndistortRectifyMap(K_xx, D_xx[:4], R_rect_xx, K_xx, S_rect_xx' in readme
    for i, cam in enumerate(rig.variant.cameras):
        n = f'{i:02d}'
        K, D = data[f'K_{n}'].reshape(3, 3), data[f'D_{n}']
        W, H = (int(x) for x in data[f'S_rect_{n}'])
        intr = _intr(rig, cam)
        if raw[f'DISTORTION_MODEL_{n}'] == 'plumb_bob':
            assert not D.any() and intr.projection == 'rectilinear'
            continue
        assert raw[f'DISTORTION_MODEL_{n}'] == 'equidistant' and D[4] == 0
        m1, m2 = cv2.fisheye.initUndistortRectifyMap(K, D[:4], data[f'R_rect_{n}'].reshape(3, 3), K, (W, H),
                                                     cv2.CV_32FC1)
        ur, vr = np.meshgrid(np.arange(0, W, 37), np.arange(0, H, 31))
        ray = np.c_[ur.ravel(), vr.ravel(), np.ones(ur.size)] @ np.linalg.inv(K).T
        u, v, _, ok = intr.project_dirs(ray)
        got = np.stack([m1[vr.ravel(), ur.ravel()], m2[vr.ravel(), ur.ravel()]], axis=1)
        assert ok.mean() > 0.95, cam.tag
        assert np.abs(got[ok] - (np.stack([u, v], axis=1)[ok] - 0.5)).max() < 0.02, cam.tag   # float32 maps
        # and the README states what part of the lens the rectified image keeps
        assert re.search(rf'{n}  {cam.tag:<6} raw +[\d.]+° × +[\d.]+°, rectified +[\d.]+° × +[\d.]+°', readme)


def test_pykitti_reads_kittis_own_layout_exactly(stereo):
    """With KITTI's layout (rectified cameras parallel to camera 00, baselines along its x) every matrix pykitti
    derives is Vantage's geometry: K_cam0..3, T_cam0..3_velo, T_cam0..3_imu, T_velo_imu and both baselines."""
    p, out = stereo
    v = p.variant
    got = pykitti_raw_calib(out / 'kitti')
    lid = next(s for s in v.sensors if s.kind == 'lidar')
    imu = next(s for s in v.sensors if s.kind == 'imu')
    assert np.allclose(got['T_velo_imu'], np.linalg.inv(sensor_T(lid)) @ sensor_T(imu), atol=2e-6)
    for i, cam in enumerate(v.cameras):
        cm = ex.camera_model(p, cam)
        assert np.allclose(got[f'K_cam{i}'], cm.K, rtol=1e-6), cam.tag
        want = np.linalg.inv(cam_T(cam)) @ sensor_T(lid)
        assert np.allclose(got[f'T_cam{i}_velo'], want, atol=2e-6), cam.tag
        assert np.allclose(got[f'T_cam{i}_imu'], np.linalg.inv(cam_T(cam)) @ sensor_T(imu), atol=2e-6), cam.tag
    assert got['b_gray'] == pytest.approx(0.54, abs=1e-5) and got['b_rgb'] == pytest.approx(0.54, abs=1e-5)


def test_pykitti_on_a_surround_rig(rig, exported):
    """A surround rig's cameras are not parallel: pykitti's K_cam0, T_cam0_velo, T_cam0_imu and T_velo_imu are exact;
    its K_cam1..3 are K R (the README says to read K_xx instead), and P_rect still projects every camera right
    (test above)."""
    out, _ = exported
    v = rig.variant
    got = pykitti_raw_calib(out / 'kitti')
    lid = next(s for s in v.sensors if s.kind == 'lidar')
    imu = next(s for s in v.sensors if s.kind == 'imu')
    c0 = v.cameras[0]
    T = lambda s: np.linalg.inv(cam_T(c0)) @ sensor_T(s)
    assert np.allclose(got['K_cam0'], ex.camera_model(rig, c0).K, rtol=1e-6)
    assert np.allclose(got['T_cam0_velo'], T(lid), atol=2e-6) and np.allclose(got['T_cam0_imu'], T(imu), atol=2e-6)
    assert np.allclose(got['T_velo_imu'], np.linalg.inv(sensor_T(lid)) @ sensor_T(imu), atol=2e-6)
    c2 = v.cameras[2]
    R02 = _W2O(c2) @ _W2O(c0).T
    assert np.allclose(got['K_cam2'], ex.camera_model(rig, c2).K @ R02, rtol=1e-5, atol=1e-3)
    readme = (out / 'kitti/README.txt').read_text()
    assert 'pykitti.raw' in readme and 'K_cam0, T_cam0_velo and T_cam0_imu are right' in readme


def test_kitti_object_files_per_camera(rig, exported):
    """kitti/object/<TAG>.txt: seven lines read by position (mmdetection3d, OpenPCDet, the object devkit); the
    devkit's x = P2 R0_rect Tr_velo_to_cam y lands lidar points where the camera's rectified image has them;
    Tr_imu_to_velo maps the IMU into the lidar."""
    out, _ = exported
    v = rig.variant
    lid = next(s for s in v.sensors if s.kind == 'lidar')
    imu = next(s for s in v.sensors if s.kind == 'imu')
    tags = ex.safe_tags(v)
    rng = np.random.default_rng(5)
    for cam in v.cameras:
        path = out / 'kitti' / 'object' / f'{tags[cam.uid]}.txt'
        lines = path.read_text().splitlines()
        assert [ln.split(':')[0] for ln in lines] == ['P0', 'P1', 'P2', 'P3', 'R0_rect', 'Tr_velo_to_cam',
                                                       'Tr_imu_to_velo']
        assert all('  ' not in ln and not ln.endswith(' ') for ln in lines)          # split(' ') readers
        mm, pcd, dev = kitti_object_readers(path)
        K = ex.camera_model(rig, cam).K
        assert np.allclose(mm['P2'], np.c_[K, np.zeros(3)], rtol=1e-6) and np.allclose(dev[2], mm['P2'])
        assert np.allclose(pcd['P2'], mm['P2'], rtol=1e-6) and np.allclose(pcd['Tr_velo2cam'], mm['Tr_velo_to_cam'],
                                                                           atol=1e-5)
        Pc = np.c_[rng.uniform(-4, 4, (30, 2)), rng.uniform(1.5, 40, 30)]
        y = np.linalg.inv(sensor_T(lid)) @ np.c_[Pc @ _W2O(cam) + cam.pos(), np.ones(30)].T
        x = mm['P2'] @ _ext(mm['R0_rect']) @ _ext(mm['Tr_velo_to_cam']) @ y
        assert np.abs((x[:2] / x[2]).T - pinhole(K, Pc)).max() < TOL_PX, cam.tag
        assert np.allclose(_ext(mm['Tr_imu_to_velo']), np.linalg.inv(sensor_T(lid)) @ sensor_T(imu), atol=2e-6)


def test_kitti_object_without_an_imu_uses_the_vehicle_frame(qapp, tmp_path):
    p, v = _data_rig()
    ex.export(['kitti_calib'], p, None, ex.ExportOptions(out_dir=str(tmp_path)))
    lid = next(s for s in v.sensors if s.kind == 'lidar')
    mm, _, _ = kitti_object_readers(tmp_path / 'kitti' / 'object' / 'F.txt')
    assert np.allclose(_ext(mm['Tr_imu_to_velo']), np.linalg.inv(sensor_T(lid)), atol=2e-6)
    assert not (tmp_path / 'kitti' / 'calib_imu_to_velo.txt').exists()
    assert 'The rig has no IMU' in (tmp_path / 'kitti' / 'README.txt').read_text()
    names = ex.format_by_id('kitti_calib').filename(p, v)
    assert names[-len(v.cameras):] == [f'kitti/object/{t}.txt' for t in ex.safe_tags(v).values()]
    v.sensors = [s for s in v.sensors if s.kind != 'lidar']        # no lidar: no object files
    assert not any('object' in n for n in ex.format_by_id('kitti_calib').filename(p, v))


# ============================================================================= Kalibr
def test_kalibr_passes_the_config_reader_checks(rig, exported):
    """Every camera passes CameraParameters' checks (models, counts, float intrinsics, int resolution, string topic)
    and is a pair AslamCamera builds; the chain's T_cn_cnm1 / T_cam_imu are rigid 4x4, timeshift_cam_imu a float,
    cam_overlaps in range and mutual; imu.yaml passes ImuParameters' checks."""
    out, _ = exported
    doc = yaml.safe_load((out / 'kalibr/camchain.yaml').read_text())
    n = len(rig.variant.cameras)
    assert list(doc) == [f'cam{i}' for i in range(n)]
    for i in range(n):
        c = doc[f'cam{i}']
        kalibr_check_camera(c)
        if i:
            assert rigid_ok(c['T_cn_cnm1'])
        else:
            assert 'T_cn_cnm1' not in c
        assert rigid_ok(c['T_cam_imu']) and isinstance(c['timeshift_cam_imu'], float)
        assert all(0 <= j < n and j != i and i in doc[f'cam{j}']['cam_overlaps'] for j in c['cam_overlaps'])
    imu = yaml.safe_load((out / 'kalibr/imu.yaml').read_text())
    assert isinstance(imu['rostopic'], str) and imu['update_rate'] > 0
    for k in ('accelerometer_noise_density', 'accelerometer_random_walk', 'gyroscope_noise_density',
              'gyroscope_random_walk'):
        assert isinstance(imu[k], float) and imu[k] > 0, k
    # the library IMU has no noise figures: Kalibr's documented example (github.com/ethz-asl/kalibr/wiki/yaml-formats)
    assert (imu['accelerometer_noise_density'], imu['accelerometer_random_walk'], imu['gyroscope_noise_density'],
            imu['gyroscope_random_walk']) == (1.86e-03, 4.33e-04, 1.87e-04, 2.66e-05)


def test_kalibr_chain_and_imu_transforms(rig, exported):
    """T_cn_cnm1 takes camera n-1 coordinates to camera n (Kalibr's camchain page); T_cam_imu takes IMU coordinates
    to the camera, and the chain is consistent the way Kalibr builds it (T_cN_imu = T_cN_cN-1 ... T_c1_c0 T_c0_imu)."""
    out, _ = exported
    v = rig.variant
    doc = yaml.safe_load((out / 'kalibr/camchain.yaml').read_text())
    imu = next(s for s in v.sensors if s.kind == 'imu')
    chain = np.array(doc['cam0']['T_cam_imu'])
    for i, cam in enumerate(v.cameras):
        c = doc[f'cam{i}']
        want = np.linalg.inv(cam_T(cam)) @ sensor_T(imu)
        assert np.allclose(c['T_cam_imu'], want, atol=1e-9), cam.tag
        if i:
            prev = v.cameras[i - 1]
            assert np.allclose(c['T_cn_cnm1'], np.linalg.inv(cam_T(cam)) @ cam_T(prev), atol=1e-9)
            chain = np.array(c['T_cn_cnm1']) @ chain
            assert np.allclose(chain, c['T_cam_imu'], atol=1e-8)
        assert c['rostopic'] == f'/{ex._ros_name(ex.safe_tags(v)[cam.uid])}/image_raw'
    text = (out / 'kalibr/camchain.yaml').read_text()
    assert re.search(r'cam0 = F, cam1 = R\b', text)                  # the header fleet.read_kalibr names cameras by
    imu_doc = yaml.safe_load((out / 'kalibr/imu.yaml').read_text())
    assert imu_doc['rostopic'] == f'/{ex._ros_name(ex.sensor_tags(v)[imu.uid])}/imu'
    assert imu_doc['update_rate'] == rig.sensor_models[imu.model_id].rate_hz
    head = (out / 'kalibr/imu.yaml').read_text()
    assert "NOT this IMU's" in head and 'm/s^2/sqrt(Hz)' in head and 'rad/s^2/sqrt(Hz)' in head


def test_kalibr_models_reproduce_the_lens(rig, exported):
    """Each written camera, projected the way aslam_cv does (and with OpenCV for its pinhole models), puts 3D points
    over the whole field where Vantage's lens does: the library fisheye, the unit calibration and the rectilinear
    lens are exact."""
    out, _ = exported
    doc = yaml.safe_load((out / 'kalibr/camchain.yaml').read_text())
    for i, cam in enumerate(rig.variant.cameras):
        c = doc[f'cam{i}']
        P, uv = lens_points(rig, cam)
        got, ok = kalibr_project(c, P)
        assert ok.all(), cam.tag
        rms, mx = errs(got, uv)
        assert mx < 1e-3, (cam.tag, c['camera_model'], c['distortion_model'], mx)
        fu, fv, cu, cv_ = c['intrinsics']
        K = np.array([[fu, 0, cu], [0, fv, cv_], [0, 0, 1]])
        if c['distortion_model'] == 'equidistant':
            cvp, _ = cv2.fisheye.projectPoints(P.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K,
                                               np.array(c['distortion_coeffs']))
        else:
            cvp, _ = cv2.projectPoints(P, np.zeros(3), np.zeros(3), K, np.array(c['distortion_coeffs']))
        assert np.abs(cvp.reshape(-1, 2) - got).max() < 1e-6, cam.tag
        # the exporter's own projection agrees with this one
        mine, ok2 = ex.kalibr_project(c, P)
        assert ok2.all() and np.abs(mine - got).max() < 1e-9


def test_kalibr_lenses_past_90_degrees(wide):
    """Lenses that image rays 90° or more off axis get a model that reaches them (omni, ds or eucm), fitted within
    0.25 px RMS and 1 px max over every pixel, as the header states; a lens none can represent is written with its
    best model and a warning; a narrower non-Kannala-Brandt lens stays pinhole-equi."""
    p, out = wide
    text = (out / 'kalibr/camchain.yaml').read_text()
    doc = yaml.safe_load(text)
    for i, cam in enumerate(p.variant.cameras):
        c = doc[f'cam{i}']
        kalibr_check_camera(c)
        P, uv = lens_points(p, cam, n=4000, seed=i)
        got, ok = kalibr_project(c, P)
        line = re.search(rf'^#   cam{i} {cam.tag}: (\S+), (.*)$', text, re.M)
        assert line, cam.tag
        lens = p.lens_for(cam)
        if lens.id == 'kbodd':
            assert 'WARNING' in line.group(2) and errs(got[ok], uv[ok])[0] > ex.FIT_RMS_PX
            continue
        rms, mx = errs(got[ok], uv[ok])
        assert ok.all() and rms <= ex.FIT_RMS_PX and mx <= ex.FIT_MAX_PX, (cam.tag, c['camera_model'], rms, mx)
        stated = re.search(r'([\d.]+) px RMS, ([\d.]+) px max', line.group(2))
        if np.degrees(np.arccos(P[:, 2] / np.linalg.norm(P, axis=1))).max() >= 90.0:      # images rays past 90°
            assert c['camera_model'] in ('omni', 'ds', 'eucm'), cam.tag
            assert stated and abs(float(stated.group(1)) - rms) < 0.02 + 0.3 * rms, (cam.tag, stated.group(0), rms)
        else:
            assert (c['camera_model'], c['distortion_model']) == ('pinhole', 'equidistant'), cam.tag
        assert 'WARNING' not in line.group(2)
    # the formats that can only write a pinhole-based model say what it cannot reach
    info = (out / 'camera_info' / 'F.yaml').read_text()
    assert re.search(r'cannot image the outer [\d.]+ % of the pixels \(rays 90° or more off axis\)', info)
    readme = (out / 'kitti' / 'README.txt').read_text()
    assert 'F: lens projection equidistant, written as Kannala-Brandt' in readme and 'cannot image the outer' in readme
    assert 'cannot image the outer' in (out / 'nuscenes' / 'README.md').read_text()


def test_kalibr_overlaps_from_the_analysis(rig, qapp, tmp_path):
    """With an analysis result cam_overlaps are the cameras that share ground (over 0.05 m²), mutual; without one the
    field-of-view estimate agrees on the neighbours (front with the mirrors, rear with the quarters)."""
    from vantage.core.analysis import Engine
    res = Engine().analyse(rig)
    ex.export(['kalibr_camchain'], rig, res, ex.ExportOptions(out_dir=str(tmp_path / 'a')))
    ex.export(['kalibr_camchain'], rig, None, ex.ExportOptions(out_dir=str(tmp_path / 'b')))
    a = yaml.safe_load((tmp_path / 'a/kalibr/camchain.yaml').read_text())
    b = yaml.safe_load((tmp_path / 'b/kalibr/camchain.yaml').read_text())
    cams = rig.variant.cameras
    index = {c.uid: i for i, c in enumerate(cams)}
    for i, c in enumerate(cams):
        ov = res.per_cam.get(c.uid, {}).get('overlaps', {})
        assert a[f'cam{i}']['cam_overlaps'] == sorted(index[u] for u, s in ov.items() if s > 0.05 and u in index)
        assert set(a[f'cam{i}']['cam_overlaps']) <= set(b[f'cam{i}']['cam_overlaps'])
    tags = [c.tag for c in cams]                     # (QL carries the narrow rectilinear lens here)
    assert tags.index('ML') in b['cam0']['cam_overlaps'] and tags.index('QR') in b['cam1']['cam_overlaps']
    assert tags.index('R') not in b['cam0']['cam_overlaps']
    extra = sum(len(set(b[k]['cam_overlaps']) - set(a[k]['cam_overlaps'])) for k in a) // 2
    assert extra <= 2                                # pairs only the body keeps apart
    assert 'the vehicle occluding' in (tmp_path / 'a/kalibr/camchain.yaml').read_text()


# ============================================================================= ROS camera_info
def test_camera_info_reads_like_camera_calibration_parsers(rig, exported):
    """readCalibrationYml's keys and fixed sizes, the keys in the order writeCalibrationYml writes them, a distortion
    model image_geometry handles with the count cv::fisheye / cv::undistortPoints accept, and a camera name
    camera_info_manager accepts."""
    out, _ = exported
    order = ['image_width', 'image_height', 'camera_name', 'camera_matrix', 'distortion_model',
             'distortion_coefficients', 'rectification_matrix', 'projection_matrix', 'binning_x', 'binning_y', 'roi']
    for cam in rig.variant.cameras:
        text = (out / 'camera_info' / f'{ex.safe_tags(rig.variant)[cam.uid]}.yaml').read_text()
        assert list(yaml.safe_load(text)) == order
        ci = ros_read_calibration_yml(text)
        assert ci['distortion_model'] in ROS_MODELS
        assert len(ci['d']) == (4 if ci['distortion_model'] == 'equidistant' else 5)
        assert re.fullmatch(r'[A-Za-z0-9_]+', ci['camera_name'])
        assert ci['binning'] == (0, 0) and ci['roi'] == (0, 0, 0, 0, False)
        assert (ci['width'], ci['height']) == (1920, 1536)


def _image_geometry(ci):
    """The C++ image_geometry PinholeCameraModel (vision_opencv rolling, the equidistant branch) as OpenCV calls:
    rectifyPoint, unrectifyPoint (projectPixelTo3dRay through P, then the distortion with rvec = Rodrigues(R^T)),
    project3dToPixel (P), and the rectification maps."""
    K, D, R, P = ci['k'], ci['d'], ci['r'], ci['p']
    fish = ci['distortion_model'] == 'equidistant'
    none = not np.any(D)

    def rectify(uv):
        if none:
            return np.asarray(uv, float)
        src = np.asarray(uv, np.float32).reshape(-1, 1, 2)          # image_geometry rectifies in float32
        f = cv2.fisheye.undistortPoints if fish else cv2.undistortPoints
        return f(src, K, D, R=R, P=P).reshape(-1, 2).astype(float)

    def unrectify(uv):
        if none:
            return np.asarray(uv, float)
        uv = np.asarray(uv, float)
        ray = np.c_[(uv[:, 0] - P[0, 2] - P[0, 3]) / P[0, 0], (uv[:, 1] - P[1, 2] - P[1, 3]) / P[1, 1],
                    np.ones(len(uv))]
        rvec, _ = cv2.Rodrigues(R.T)
        if fish:
            got, _ = cv2.fisheye.projectPoints(ray.reshape(-1, 1, 3), rvec, np.zeros(3), K, D)
        else:
            got, _ = cv2.projectPoints(ray, rvec, np.zeros(3), K, D)
        return got.reshape(-1, 2)

    def project(X):
        h = np.c_[X, np.ones(len(X))] @ P.T
        return h[:, :2] / h[:, 2:]

    def maps():
        f = cv2.fisheye.initUndistortRectifyMap if fish else cv2.initUndistortRectifyMap
        return f(K, D, R, P, (ci['width'], ci['height']), cv2.CV_32FC1)
    return rectify, unrectify, project, maps


def test_camera_info_image_geometry_matches_the_lens(rig, exported):
    """image_geometry with our K, D, R, P: rectifyPoint of a point's raw pixel (Vantage's lens) is its rectified pixel
    project3dToPixel gives, unrectifyPoint inverts it, and the rectification map sends rectified pixels to the raw
    pixel of their ray."""
    out, _ = exported
    for cam in rig.variant.cameras:
        ci = ros_read_calibration_yml((out / 'camera_info' / f'{ex.safe_tags(rig.variant)[cam.uid]}.yaml').read_text())
        rectify, unrectify, project, maps = _image_geometry(ci)
        P, uv_raw = lens_points(rig, cam, n=800, seed=7)
        rect = project(P)
        inside = (rect[:, 0] > 0) & (rect[:, 0] < 1919) & (rect[:, 1] > 0) & (rect[:, 1] < 1535) & (P[:, 2] > 0)
        assert inside.sum() > 200, cam.tag
        assert np.abs(rectify(uv_raw[inside]) - rect[inside]).max() < 0.02, cam.tag        # float32 maths
        assert np.abs(unrectify(rect[inside]) - uv_raw[inside]).max() < 1e-3, cam.tag
        m1, m2 = maps()
        ur, vr = np.meshgrid(np.arange(3, 1920, 59), np.arange(5, 1536, 47))
        ray = np.c_[ur.ravel(), vr.ravel(), np.ones(ur.size)] @ np.linalg.inv(ci['k']).T
        u, v, _, ok = _intr(rig, cam).project_dirs(ray)
        got = np.stack([m1[vr.ravel(), ur.ravel()], m2[vr.ravel(), ur.ravel()]], axis=1)
        if ci['distortion_model'] == 'plumb_bob':
            assert np.abs(got - np.stack([ur.ravel(), vr.ravel()], axis=1)).max() < 1e-3
        else:
            assert np.abs(got[ok] - (np.stack([u, v], axis=1)[ok] - 0.5)).max() < 0.02, cam.tag


def test_dataset_bag_camera_info_matches_the_files(rig, exported):
    """The sensor_msgs/CameraInfo the sample dataset's bag writes for a camera (core.dataset) carries the camera_info
    file's K, D, R, P and model at full size; scaled images scale K about the pixel-centre convention; undistorted
    images are plumb_bob with D = 0 and the same K, which is P = [K | 0]'s rectified image."""
    from vantage.core import dataset as DSET
    from vantage.core import rosbag as RB
    out, _ = exported
    ctx = ex._context(rig, None, ex.ExportOptions(frame='rear_axle'))
    for cm in ctx.models:
        ci = ros_read_calibration_yml((out / 'camera_info' / f'{cm.tag}.yaml').read_text())
        for scale, undistort in ((1.0, False), (0.5, False), (1.0, True)):
            w, h, K, dmodel, dcoef = DSET._camera_setup(rig, cm, scale, undistort)
            msg = RB.camera_info(w, h, dmodel, dcoef, K, None, None, 0, f'{cm.tag}_optical_frame')
            k, p_ = np.array(msg['k']).reshape(3, 3), np.array(msg['p']).reshape(3, 4)
            assert np.allclose(p_, np.c_[k, np.zeros(3)]) and np.allclose(np.array(msg['r']).reshape(3, 3), np.eye(3))
            assert msg['header']['frame_id'] == f'{cm.tag}_optical_frame'
            s = w / ci['width']
            want = ci['k'].copy()
            want[0, :] *= s
            want[1, :] *= s
            want[0, 2] = (ci['k'][0, 2] + 0.5) * s - 0.5
            want[1, 2] = (ci['k'][1, 2] + 0.5) * s - 0.5
            assert np.allclose(k, want, atol=1e-6), (cm.tag, scale)
            if undistort or ci['distortion_model'] == 'plumb_bob':
                assert dmodel == 'plumb_bob' and not any(msg['d'])
            else:
                assert dmodel == ci['distortion_model'] and np.allclose(msg['d'], ci['d'], atol=1e-9)


# ============================================================================= URDF / TF (REP 103 / 105)
def test_urdf_and_tf_optical_frames(rig, exported):
    """<tag>_optical_frame is z forward along the camera's aim, x right, y down (REP 103 suffix frames), reached by
    rpy (-pi/2, 0, -pi/2) from <tag>_link; base_link to <tag>_optical_frame equals the nuScenes calibrated_sensor
    rotation, KITTI's Tr_cam_xx_to_vehicle and the static TF launch file's quaternions."""
    out, _ = exported
    v = rig.variant
    root = ET.parse(out / 'vantage_cameras.urdf').getroot()
    joints = {j.get('name'): j for j in root.findall('joint')}
    nus = json.loads((out / 'nuscenes/calibrated_sensor.json').read_text())
    veh = pykitti_read_calib_file(out / 'kitti/calib_cams_to_vehicle.txt')
    launch = (out / 'static_tf.launch.py').read_text()
    tf = {m[5]: (np.array(m[0:3], float), np.array(m[3], float), m[4]) for m in
          [(a, b, c, [float(x) for x in re.findall(r"'(-?[\d.]+)'", q)], par, ch) for a, b, c, q, par, ch in
           re.findall(r"_static_tf\('[^']+', \('([^']+)', '([^']+)', '([^']+)'\), \(([^)]*)\), '([^']+)', '([^']+)'\)",
                      launch)]}
    tags = ex.safe_tags(v)
    for i, cam in enumerate(v.cameras):
        t = tags[cam.uid]

        def origin(name):
            o = joints[name].find('origin')
            return np.array(o.get('xyz').split(), float), urdf_rpy(*map(float, o.get('rpy').split()))
        xyz, Rb = origin(f'{t}_joint')
        _, Ro = origin(f'{t}_optical_joint')
        R = Rb @ Ro                                          # optical frame axes in base_link
        body = rot_zyx(cam.yaw, cam.pitch, cam.roll)
        assert np.allclose(R[:, 2], body[:, 0], atol=2e-5) and np.allclose(R[:, 0], -body[:, 1], atol=2e-5)
        assert np.allclose(R[:, 1], -body[:, 2], atol=2e-5), cam.tag
        assert np.allclose(xyz, cam.pos() - ORIGIN, atol=1e-3)
        assert np.allclose(R, quat_xyzw(*nus[i]['rotation'][1:], nus[i]['rotation'][0]), atol=2e-5)
        assert np.allclose(R, veh[f'Tr_cam_{i:02d}_to_vehicle'].reshape(3, 4)[:, :3], atol=2e-5)
        p0, q0, par0 = tf[f'{t}_link']
        p1, q1, par1 = tf[f'{t}_optical_frame']
        assert par0 == 'base_link' and par1 == f'{t}_link' and np.allclose(p1, 0)
        assert np.allclose(quat_xyzw(*q0) @ quat_xyzw(*q1), R, atol=2e-5) and np.allclose(p0, xyz, atol=1e-3)


# ============================================================================= files, names, edge cases
def test_formats_and_names(rig, exported, qapp):
    out, paths = exported
    v = rig.variant
    n = len(v.cameras)
    names = {os.path.relpath(x, out).replace(os.sep, '/') for x in paths}
    want = set()
    for fid in ('kitti_calib', 'kalibr_camchain', 'ros_camera_info'):
        want |= set(ex._names(ex.format_by_id(fid), rig, v))
    assert want <= names and {'kalibr/imu.yaml', 'kitti/calib_imu_to_velo.txt', 'kitti/object/QR.txt'} <= names
    assert len([x for x in names if x.startswith('kitti/object/')]) == n
    for fid in ('kitti_calib', 'kalibr_camchain', 'ros_camera_info'):
        rel, text, size = ex.preview(fid, rig, None, ex.ExportOptions(out_dir=str(out), frame='custom'))
        first = ex._names(ex.format_by_id(fid), rig, v)[0]
        assert rel.endswith(first) and text.startswith((out / first).read_text())
    q, w = _data_rig()                             # no IMU: one Kalibr file, as before
    assert ex.format_by_id('kalibr_camchain').filename(q, w) == 'kalibr/camchain.yaml'


def test_units_and_frames_leave_the_calibration_alone(rig, tmp_path):
    a = ex.export(IDS[:4], rig, None, ex.ExportOptions(out_dir=str(tmp_path / 'm'), frame='custom'))
    b = ex.export(IDS[:4], rig, None, ex.ExportOptions(out_dir=str(tmp_path / 'mm'), frame='custom', units='mm',
                                                       angles='rad'))
    for x, y in zip(a, b):
        assert open(x, 'rb').read() == open(y, 'rb').read(), x


def test_hidden_and_odd_tags(qapp, tmp_path):
    """A hidden camera is still a camera of the chain; a tag with characters ROS forbids gets a valid camera name and
    topic; with two IMUs, the first switched off, the files describe the visible one; a variant without cameras
    writes the empty files."""
    p, v = _data_rig()
    v.cameras[2].visible = False
    v.cameras[3].tag = '3-B'
    _add(p, v, 'imu', 'ins_fog')
    ex.export(['kalibr_camchain', 'ros_camera_info'], p, None, ex.ExportOptions(out_dir=str(tmp_path)))
    doc = yaml.safe_load((tmp_path / 'kalibr/camchain.yaml').read_text())
    assert len(doc) == 6 and doc['cam3']['rostopic'] == '/s_3_b/image_raw'
    assert yaml.safe_load((tmp_path / 'camera_info/3-B.yaml').read_text())['camera_name'] == '3_B'
    # two IMUs, the first switched off: the files describe the visible one
    first = next(s for s in v.sensors if s.kind == 'imu')
    first.visible = False
    on = _add(p, v, 'imu', 'imu_mems', yaw=45.0)
    ex.export(['kalibr_camchain', 'kitti_calib'], p, None, ex.ExportOptions(out_dir=str(tmp_path / 'h')))
    doc = yaml.safe_load((tmp_path / 'h/kalibr/camchain.yaml').read_text())
    assert np.allclose(doc['cam0']['T_cam_imu'], np.linalg.inv(cam_T(v.cameras[0])) @ sensor_T(on), atol=1e-9)
    assert yaml.safe_load((tmp_path / 'h/kalibr/imu.yaml').read_text())['rostopic'] == f'/{on.tag.lower()}/imu'
    v.cameras = []
    ex.export(['kalibr_camchain', 'kitti_calib'], p, None, ex.ExportOptions(out_dir=str(tmp_path / 'e')))
    assert yaml.safe_load((tmp_path / 'e/kalibr/camchain.yaml').read_text()) == {}
    assert yaml.safe_load((tmp_path / 'e/kalibr/imu.yaml').read_text())['update_rate'] == 200.0     # the MEMS IMU's


def test_imu_noise_from_the_model_when_given(qapp, tmp_path, monkeypatch):
    """When the IMU model carries all four noise figures, imu.yaml writes them (Kalibr's units) instead of Kalibr's
    example values."""
    p, v = _data_rig()
    imu = _add(p, v, 'imu', 'imu_mems')
    m = p.sensor_models[imu.model_id]
    own = {'accel_noise_density': 1.2e-3, 'accel_random_walk': 3.0e-4, 'gyro_noise_density': 8.7e-5,
           'gyro_random_walk': 4.0e-6}
    for k, val in own.items():
        monkeypatch.setattr(m, k, val)
    ex.export(['kalibr_camchain'], p, None, ex.ExportOptions(out_dir=str(tmp_path)))
    text = (tmp_path / 'kalibr/imu.yaml').read_text()
    doc = yaml.safe_load(text)
    assert doc['gyroscope_noise_density'] == 8.7e-5 and doc['accelerometer_random_walk'] == 3.0e-4
    assert "NOT this IMU's" not in text and "the IMU model's own figures" in text
    # the figures are the sensor model's own fields: saved with the project, negatives cleared
    from vantage.core.sensors import SensorModel
    back = SensorModel.from_dict(m.to_dict())
    assert {k: getattr(back, k) for k in own} == own
    assert SensorModel.from_dict(dict(m.to_dict(), gyro_random_walk=-1.0)).gyro_random_walk == 0.0


def test_the_checks_catch_the_old_mistakes(rig, exported, wide):
    """Injected errors are caught: the old P_rect_xx = [K | 0] for every camera misplaces lidar points in the other
    cameras by hundreds of pixels; an inverted T_cn_cnm1 moves points by metres; pinhole-equi on a 200° lens cannot
    image its outer field."""
    out, _ = exported
    v = rig.variant
    _, cams = kitti_matlab_cam_to_cam(out / 'kitti/calib_cam_to_cam.txt')
    velo = _rt(kitti_matlab_read(out / 'kitti/calib_velo_to_cam.txt', 'R', 3, 3),
               kitti_matlab_read(out / 'kitti/calib_velo_to_cam.txt', 'T', 3, 1))
    lid = next(s for s in v.sensors if s.kind == 'lidar')
    cam = v.cameras[2]
    Pc = np.c_[np.zeros((5, 2)), np.linspace(3, 20, 5)]
    X = np.linalg.inv(sensor_T(lid)) @ np.c_[Pc @ _W2O(cam) + cam.pos(), np.ones(5)].T
    old = np.c_[cams[2]['K'], np.zeros(3)] @ velo @ X                    # the old export's P_rect_02
    assert np.abs((old[:2] / old[2]).T - pinhole(cams[2]['K'], Pc)).min() > 100
    doc = yaml.safe_load((out / 'kalibr/camchain.yaml').read_text())
    T = np.array(doc['cam2']['T_cn_cnm1'])                              # R -> ML (not a half turn: F -> R is)
    P0 = np.array([[0.0, 0.0, 5.0, 1.0]])
    good = (T @ P0.T).T[:, :3]
    assert np.linalg.norm((np.linalg.inv(T) @ P0.T).T[:, :3] - good) > 1.0
    assert np.allclose(good, to_cam(v.cameras[2], P0[:, :3] @ _W2O(v.cameras[1]) + v.cameras[1].pos()), atol=1e-8)
    p, _ = wide
    fe = next(c for c in p.variant.cameras if c.lens_id == 'fe200')
    cm = ex.camera_model(p, fe)
    P, uv = lens_points(p, fe, n=3000)
    base = {'camera_model': 'pinhole', 'intrinsics': [cm.fx, cm.fy, cm.cx, cm.cy], 'distortion_model': 'equidistant',
            'distortion_coeffs': list(cm.k)}
    _, ok = kalibr_project(base, P)
    assert 0.8 < ok.mean() < 0.97                                       # the rays at 90° and beyond are lost


# ============================================================================= the tools themselves (optional)
def _fake_drive(base, calib_dir):
    """A KITTI raw drive folder pykitti.raw opens: the date folder with the calibration, one OXTS packet."""
    date = '2026_10_21'
    root = base / date
    drive = root / f'{date}_drive_0001_sync' / 'oxts'
    (drive / 'data').mkdir(parents=True)
    for n in ('calib_cam_to_cam.txt', 'calib_velo_to_cam.txt', 'calib_imu_to_velo.txt'):
        (root / n).write_bytes((calib_dir / n).read_bytes())
    packet = [49.0, 8.4, 112.0, 0.01, -0.02, 1.2] + [0.0] * 17 + [0.5, 0.1] + [4, 10, 5, 5, 5]
    (drive / 'data' / '0000000000.txt').write_text(' '.join(str(x) for x in packet) + '\n')
    (drive / 'timestamps.txt').write_text('2026-10-21 09:30:00.000000000\n')
    return date


def _pykitti():
    sys.path.insert(0, _kit('python'))
    try:
        import pykitti
        return pykitti
    finally:
        sys.path.pop(0)


@needs('python', 'pykitti')
def test_real_pykitti_raw(rig, exported, stereo, tmp_path):
    """pykitti.raw itself on drive folders built from the exports: on KITTI's layout every derived matrix is
    Vantage's geometry; on the surround rig camera 0's are, and the rest equal the reimplementation above."""
    pykitti = _pykitti()
    for which, (proj, out) in (('stereo', stereo), ('surround', (rig, exported[0]))):
        base = tmp_path / which
        date = _fake_drive(base, out / 'kitti')
        data = pykitti.raw(str(base), date, '0001')
        mine = pykitti_raw_calib(out / 'kitti')
        for k, val in mine.items():
            assert np.allclose(getattr(data.calib, k), val), (which, k)
        v = proj.variant
        lid = next(s for s in v.sensors if s.kind == 'lidar')
        imu = next(s for s in v.sensors if s.kind == 'imu')
        cams = range(4) if which == 'stereo' else range(1)
        for i in cams:
            cam = v.cameras[i]
            assert np.allclose(getattr(data.calib, f'K_cam{i}'), ex.camera_model(proj, cam).K, rtol=1e-6)
            assert np.allclose(getattr(data.calib, f'T_cam{i}_velo'), np.linalg.inv(cam_T(cam)) @ sensor_T(lid),
                               atol=2e-6)
            assert np.allclose(getattr(data.calib, f'T_cam{i}_imu'), np.linalg.inv(cam_T(cam)) @ sensor_T(imu),
                               atol=2e-6)
        if which == 'stereo':
            assert data.calib.b_gray == pytest.approx(0.54, abs=1e-5) and data.calib.b_rgb == pytest.approx(0.54,
                                                                                                             abs=1e-5)
        assert len(data.oxts) == 1 and len(data) == 1


def _stub_module(name, **attrs):
    mod = types.ModuleType(name)
    mod.__dict__.update(attrs)
    return mod


class _Recorder:
    """Stands in for one compiled aslam_cv class: records the constructor's arguments."""
    def __init__(self, *args):
        self.args = args


def _aslam(name):
    mod = types.ModuleType(name)
    mod.__getattr__ = lambda attr: type(attr, (_Recorder,), {})
    return mod


class _Transformation:
    """Stands in for sm.Transformation: built from a 4x4 the way Schweizer-Messer does (rotation through a
    quaternion), so T() differs from the input unless its rotation is a proper rotation."""
    def __init__(self, T):
        T = np.asarray(T, float)
        assert T.shape == (4, 4)
        from scipy.spatial.transform import Rotation
        self._R = Rotation.from_matrix(T[:3, :3]).as_matrix()
        self._t = T[:3, 3].copy()

    def T(self):
        out = np.eye(4)
        out[:3, :3], out[:3, 3] = self._R, self._t
        return out

    def t(self):
        return self._t


def _kalibr_reader():
    stubs = {'aslam_cv': _aslam('aslam_cv'), 'aslam_cv_backend': _aslam('aslam_cv_backend'),
             'sm': _stub_module('sm', Transformation=_Transformation)}
    saved = {k: sys.modules.get(k) for k in stubs}
    sys.modules.update(stubs)
    try:
        spec = importlib.util.spec_from_file_location('kalibr_config_reader', _kit('kalibr', 'ConfigReader.py'))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        for k, val in saved.items():
            if val is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = val


@needs('kalibr', 'ConfigReader.py')
def test_real_kalibr_config_reader(rig, exported, wide):
    """Kalibr's own ConfigReader reads the camchain and imu.yaml: every getter's checks pass, AslamCamera builds a
    geometry for each camera with our numbers, and the transforms survive sm.Transformation unchanged."""
    kc = _kalibr_reader()
    for out in (exported[0], wide[1]):
        chain = kc.CameraChainParameters(str(out / 'kalibr/camchain.yaml'))
        doc = yaml.safe_load((out / 'kalibr/camchain.yaml').read_text())
        for i in range(chain.numCameras()):
            cp = chain.getCameraParameters(i)
            model, q = cp.getIntrinsics()
            dist, k = cp.getDistortion()
            assert cp.getResolution() == [1920, 1536] and cp.getRosTopic().endswith('/image_raw')
            cam = kc.AslamCamera.fromParameters(cp)
            assert hasattr(cam, 'geometry'), (model, dist)                   # pinhole-none would set .camera
            proj = cam.geometry.args[0]                          # the projection, built with our numbers in order
            assert list(proj.args[:len(q)]) == q and list(proj.args[len(q):len(q) + 2]) == [1920, 1536]
            if dist != 'none':
                assert list(proj.args[-1].args) == k                   # the distortion object
            cp.printDetails(open(os.devnull, 'w'))
            if i:
                T = chain.getExtrinsicsLastCamToHere(i).T()
                assert np.allclose(T, doc[f'cam{i}']['T_cn_cnm1'], atol=1e-9)
            chain.getCamOverlaps(i)
            if 'T_cam_imu' in doc[f'cam{i}']:
                assert np.allclose(chain.getExtrinsicsImuToCam(i).T(), doc[f'cam{i}']['T_cam_imu'], atol=1e-9)
                assert chain.getTimeshiftCamImu(i) == 0.0
    imu = kc.ImuParameters(str(exported[0] / 'kalibr/imu.yaml'))
    rate = imu.getUpdateRate()
    ad, arw, an = imu.getAccelerometerStatistics()
    gd, grw, gn = imu.getGyroStatistics()
    assert ad == pytest.approx(an * math.sqrt(rate)) and gd == pytest.approx(gn * math.sqrt(rate)) and arw > 0 < grw
    assert imu.getRosTopic().endswith('/imu')


@needs('ros', 'parse_yml_check')
def test_real_camera_calibration_parsers(rig, exported):
    """camera_calibration_parsers' readCalibrationYml (the ROS 2 C++ source, built) reads every camera_info file to
    exactly the numbers the YAML holds."""
    out, _ = exported
    files = sorted(str(x) for x in (out / 'camera_info').glob('*.yaml'))
    res = subprocess.run([_kit('ros', 'parse_yml_check')] + files, capture_output=True, text=True, timeout=60)
    assert res.returncode == 0, res.stderr
    for line, path in zip(res.stdout.splitlines(), files):
        got = json.loads(line)
        want = ros_read_calibration_yml(open(path).read())
        assert got['ok'] and got['camera_name'] == want['camera_name']
        assert got['distortion_model'] == want['distortion_model']
        assert (got['width'], got['height']) == (want['width'], want['height'])
        for k in ('k', 'r', 'p', 'd'):
            assert np.allclose(np.array(got[k]).reshape(np.shape(want[k])), want[k], rtol=0, atol=1e-12), (path, k)
        assert (got['binning_x'], got['binning_y']) == want['binning'] and tuple(got['roi']) == want['roi']
    assert 'WARN' not in res.stderr and 'ERROR' not in res.stderr


def _load_cameramodels(path):
    stubs = {'sensor_msgs': _stub_module('sensor_msgs'), 'sensor_msgs.msg': _stub_module('sensor_msgs.msg'),
             'deprecated': _stub_module('deprecated'),
             'deprecated.sphinx': _stub_module('deprecated.sphinx', deprecated=lambda **kw: (lambda f: f))}
    saved = {k: sys.modules.get(k) for k in stubs}
    sys.modules.update(stubs)
    try:
        spec = importlib.util.spec_from_file_location('image_geometry_' + os.path.basename(path)[:-3], path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        for k, val in saved.items():
            if val is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = val


@needs('ros', 'cameramodels_rolling.py')
def test_real_image_geometry_python(rig, exported):
    """image_geometry's Python PinholeCameraModel (Humble and Rolling) with our CameraInfo: exact for the pinhole
    camera; for an equidistant camera it applies plumb_bob maths (it ignores distortion_model), as the camera_info
    comment warns, so its rectified pixels are off by many pixels while the C++ path above is exact."""
    out, _ = exported
    for name in ('cameramodels_humble.py', 'cameramodels_rolling.py'):
        mod = _load_cameramodels(_kit('ros', name))
        for cam in rig.variant.cameras:
            tag = ex.safe_tags(rig.variant)[cam.uid]
            ci = ros_read_calibration_yml((out / 'camera_info' / f'{tag}.yaml').read_text())
            roi = types.SimpleNamespace(x_offset=0, y_offset=0, height=0, width=0, do_rectify=False)
            msg = types.SimpleNamespace(width=ci['width'], height=ci['height'], binning_x=0, binning_y=0, roi=roi,
                                        header=types.SimpleNamespace(frame_id=f'{cam.tag}_optical_frame', stamp=0),
                                        distortion_model=ci['distortion_model'], k=list(ci['k'].ravel()),
                                        d=list(ci['d']), r=list(ci['r'].ravel()), p=list(ci['p'].ravel()))
            m = mod.PinholeCameraModel()
            m.fromCameraInfo(msg)
            P, uv_raw = lens_points(rig, cam, n=200, seed=9)
            keep = P[:, 2] > 0.5 * np.linalg.norm(P, axis=1)             # within 60° of the axis
            P, uv_raw = P[keep], uv_raw[keep]
            rect = np.array([m.project3dToPixel(x) for x in P])
            assert np.abs(rect - pinhole(ci['k'], P)).max() < 1e-6
            py = np.array([np.asarray(m.rectifyPoint(tuple(u))).ravel() for u in uv_raw])
            err = np.abs(py - rect).max()
            if ci['distortion_model'] == 'plumb_bob':
                assert err < 1e-3, (name, cam.tag)
            else:
                assert err > 5.0, (name, cam.tag, err)                     # the documented limitation


@needs('python', 'nuscenes')
def test_real_nuscenes_devkit(rig, exported):
    """nuScenes devkit maths (pyquaternion's Quaternion, geometry_utils.view_points) with the calibrated_sensor
    rows: a pinhole camera projects where the lens does; a fisheye camera's points land on the image undistorted
    with cv2.fisheye and camera_intrinsic, as the README says."""
    out, _ = exported
    sys.path.insert(0, _kit('python'))
    try:
        from pyquaternion import Quaternion
        # the module alone: the package's __init__ pulls in the whole devkit (matplotlib, sklearn)
        spec = importlib.util.spec_from_file_location('nuscenes_geometry_utils',
                                                      _kit('python', 'nuscenes', 'utils', 'geometry_utils.py'))
        geo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(geo)
        view_points = geo.view_points
    finally:
        sys.path.pop(0)
    cal = json.loads((out / 'nuscenes/calibrated_sensor.json').read_text())
    for row, cam in zip(cal, rig.variant.cameras):
        P_cam, uv = lens_points(rig, cam, n=300, seed=11)
        Pw = P_cam @ _W2O(cam) + cam.pos() - ORIGIN                       # ego frame (the export's origin)
        q = Quaternion(row['rotation'])
        pc = np.array([q.inverse.rotate(x - np.array(row['translation'])) for x in Pw])
        assert np.allclose(pc, P_cam, atol=1e-5)
        front = pc[:, 2] > 0.5 * np.linalg.norm(pc, axis=1)
        K = np.array(row['camera_intrinsic'])
        pts = view_points(pc[front].T, K, normalize=True)[:2].T
        d = row['vantage_distortion']
        if d['model'] == 'kannala_brandt':
            und = cv2.fisheye.undistortPoints(uv[front].reshape(-1, 1, 2), K, np.array(d['coefficients']), P=K)
            assert np.abs(und.reshape(-1, 2) - pts).max() < 0.01, cam.tag
        else:
            assert np.abs(uv[front] - pts).max() < 0.01, cam.tag
