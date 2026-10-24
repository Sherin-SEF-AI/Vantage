"""Tests for the dataset calibration exports: nuScenes, KITTI, ROS camera_info and Kalibr.

Each file is read the way the dataset's own tools read it, and known vehicle-frame points are projected through the
exported numbers (cv2.fisheye for Kannala-Brandt, plain pinhole maths for a rectilinear lens). They must land where
Vantage's lens model puts them, within 0.05 px.
Run with:  QT_QPA_PLATFORM=offscreen python3 -m pytest tests/test_dataset_exports.py -q
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os
import sys

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtGui import QGuiApplication  # noqa: E402

from vantage.core import calib  # noqa: E402
from vantage.core import exporters as ex  # noqa: E402
from vantage.core.analysis import Engine  # noqa: E402
from vantage.core.lens import Lens, resolve  # noqa: E402
from vantage.core.mathutil import BODY_TO_OPTICAL, rot_zyx  # noqa: E402
from vantage.core.samples import blank_project, tigor_sample_project  # noqa: E402

FIXED_NOW = dt.datetime(2026, 10, 21, 9, 30, 0, tzinfo=dt.timezone(dt.timedelta(hours=5, minutes=30)))
IDS = ['nuscenes_calib', 'kitti_calib', 'ros_camera_info', 'kalibr_camchain']
ORIGIN = np.array([1.0, 0.2, 0.3])          # a custom export origin, so every file must honour it
TOL_PX = 0.05

cv2 = pytest.importorskip('cv2')


# ----------------------------------------------------------------------------- fixtures
@pytest.fixture(scope='module')
def qapp():
    app = QGuiApplication.instance() or QGuiApplication([])
    yield app


@pytest.fixture(autouse=True)
def fixed_clock(monkeypatch):
    monkeypatch.setattr(ex, '_now', lambda: FIXED_NOW)


@pytest.fixture(scope='module')
def sample(qapp):
    p = tigor_sample_project()
    res = Engine().analyse(p)
    assert res is not None
    return p, res


@pytest.fixture(scope='module')
def rig(sample):
    """The sample with a unit-calibrated camera (fx != fy, offset principal point) and a rectilinear one."""
    p, res = sample
    q = p.snapshot()
    cams = q.variant.cameras
    base = ex.camera_model(q, cams[1])
    cams[1].calib = calib.make_record(1920, 1536, base.fx * 1.004, base.fx * 1.011, 963.2, 761.7,
                                      [base.k[0] + 0.002, base.k[1] - 0.004, base.k[2], base.k[3]],
                                      rms_px=0.21, serial='SN-R-0042', source='test')
    q.lenses['rect'] = Lens(id='rect', name='RECT60', focal_mm=4.2, projection='rectilinear',
                            sheet_fov=(60.0, 48.0, 75.0))
    cams[4].lens_id = 'rect'
    q.custom_origin = list(ORIGIN)
    return q, res


@pytest.fixture(scope='module')
def exported(rig, tmp_path_factory):
    q, res = rig
    out = tmp_path_factory.mktemp('datasets')
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ex, '_now', lambda: FIXED_NOW)
        paths = ex.export(IDS, q, res, ex.ExportOptions(out_dir=str(out), frame='custom'))
    return out, paths


# ----------------------------------------------------------------------------- helpers
def _app_view(q, cam):
    """Vehicle-frame points spread over the camera's view (ground and above), the app's optical-frame coordinates
    of them and the app's pixels, OpenCV convention (pixel centres on integers)."""
    intr = resolve(q.module_for(cam), q.lens_for(cam))
    W2O = BODY_TO_OPTICAL @ rot_zyx(cam.yaw, cam.pitch, cam.roll).T
    pos = cam.pos()
    xs, ys, zs = np.meshgrid(np.linspace(-4, 4, 17), np.linspace(-4, 4, 17), [0.0, 0.6, 1.4])
    P = np.stack([pos[0] + xs.ravel(), pos[1] + ys.ravel(), zs.ravel()], axis=1)
    Pc = (P - pos) @ W2O.T
    u, v, theta, ok = intr.project_dirs(Pc)
    keep = ok & (theta < math.radians(80 if intr.projection != 'rectilinear' else 60)) & (Pc[:, 2] > 0.05)
    assert keep.sum() >= 12, cam.tag
    idx = np.flatnonzero(keep)[np.linspace(0, keep.sum() - 1, 24).astype(int)]
    # the app's sensor coordinates have pixel centres at i + 0.5, OpenCV's at i
    return P[idx], Pc[idx], np.stack([u[idx], v[idx]], axis=1) - 0.5


def _project(Pc, K, D, model):
    """Optical-frame points -> pixels the way a dataset loader does it."""
    K = np.asarray(K, float).reshape(3, 3)
    if model in ('equidistant', 'kannala_brandt'):
        uv, _ = cv2.fisheye.projectPoints(Pc.reshape(-1, 1, 3).astype(np.float64), np.zeros(3), np.zeros(3), K,
                                          np.asarray(D, float).ravel()[:4])
        return uv.reshape(-1, 2)
    assert model in ('plumb_bob', 'radtan', 'none (pinhole)')
    assert not np.any(np.asarray(D, float))
    h = Pc @ K.T
    return h[:, :2] / h[:, 2:]


def _quat_matrix(w, x, y, z):
    """Rotation matrix of a unit quaternion given w first (pyquaternion's / nuScenes' order)."""
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _kitti_read(path):
    """pykitti's read_calib_file: 'key: numbers' lines; anything that is not numbers is skipped."""
    data, raw = {}, {}
    with open(path) as fh:
        for line in fh:
            key, value = line.split(':', 1)
            raw[key] = value.strip()
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass
    return data, raw


def _model_of(q, cam):
    return 'plumb_bob' if q.lens_for(cam).projection == 'rectilinear' else 'equidistant'


# ----------------------------------------------------------------------------- registry
def test_registered_under_datasets(rig):
    q, _ = rig
    assert 'Datasets' in ex.GROUPS
    got = {f.id: f for f in ex.FORMATS if f.group == 'Datasets'}
    assert list(got) == IDS
    tags = list(ex.safe_tags(q.variant).values())
    assert got['nuscenes_calib'].filename(q, q.variant) == ['nuscenes/calibrated_sensor.json', 'nuscenes/sensor.json',
                                                             'nuscenes/README.md']
    assert got['kitti_calib'].filename(q, q.variant)[0] == 'kitti/calib_cam_to_cam.txt'
    assert got['ros_camera_info'].filename(q, q.variant) == [f'camera_info/{t}.yaml' for t in tags]
    assert got['kalibr_camchain'].filename(q, q.variant) == 'kalibr/camchain.yaml'


def test_every_file_written(rig, exported):
    q, _ = rig
    out, paths = exported
    n = len(q.variant.cameras)
    assert len(paths) == 3 + 3 + n + 1
    names = {os.path.relpath(x, out).replace(os.sep, '/') for x in paths}
    for f in ('nuscenes/sensor.json', 'nuscenes/calibrated_sensor.json', 'nuscenes/README.md',
              'kitti/calib_cam_to_cam.txt', 'kitti/calib_cams_to_vehicle.txt', 'kitti/README.txt',
              'camera_info/F.yaml', 'camera_info/QR.yaml', 'kalibr/camchain.yaml'):
        assert f in names
    for path in paths:
        assert os.path.getsize(path) > 0


def test_previews_match_the_files(rig, exported):
    q, res = rig
    out, _ = exported
    for fid in IDS:
        rel, text, size = ex.preview(fid, q, res, ex.ExportOptions(out_dir=str(out), frame='custom'))
        first = ex.format_by_id(fid).filename(q, q.variant)
        first = first if isinstance(first, str) else first[0]
        assert rel.endswith(first) and size > 0
        assert text.startswith((out / first).read_text('utf-8'))


# ----------------------------------------------------------------------------- nuScenes
def test_nuscenes_tables(rig, exported):
    q, _ = rig
    out, _ = exported
    sensors = json.loads((out / 'nuscenes/sensor.json').read_text('utf-8'))
    cal = json.loads((out / 'nuscenes/calibrated_sensor.json').read_text('utf-8'))
    cams = q.variant.cameras
    assert [s['channel'] for s in sensors] == [f'CAM_{c.tag.upper()}' for c in cams]
    assert {s['modality'] for s in sensors} == {'camera'}
    assert [c['sensor_token'] for c in cal] == [s['token'] for s in sensors]
    tokens = [s['token'] for s in sensors] + [c['token'] for c in cal]
    assert len(set(tokens)) == len(tokens) and all(len(t) == 32 and int(t, 16) >= 0 for t in tokens)
    for c, row in zip(cams, cal):
        assert np.allclose(row['translation'], c.pos() - ORIGIN, atol=1e-6)     # export origin honoured, metres
        assert np.array(row['camera_intrinsic']).shape == (3, 3)
        dist = row['vantage_distortion']
        assert dist['image_size'] == [1920, 1536] and dist['camera'] == c.tag
        if _model_of(q, c) == 'plumb_bob':
            assert dist['model'] == 'none (pinhole)' and dist['coefficients'] == []
        else:
            assert dist['model'] == 'kannala_brandt' and len(dist['coefficients']) == 4
    assert cal[1]['vantage_distortion']['source'] == 'unit_calibration'
    readme = (out / 'nuscenes/README.md').read_text('utf-8')
    assert '[w, x, y, z]' in readme and 'undistort' in readme
    # a re-export keeps the tokens
    again = ex._render('nuscenes_calib', ex._context(q, None, ex.ExportOptions(frame='custom')))
    assert json.loads(dict(again)['nuscenes/sensor.json']) == sensors


def test_nuscenes_quaternion_is_w_first_sensor_to_ego(rig, exported):
    q, _ = rig
    out, _ = exported
    cal = json.loads((out / 'nuscenes/calibrated_sensor.json').read_text('utf-8'))
    for c, row in zip(q.variant.cameras, cal):
        w, x, y, z = row['rotation']
        assert w >= 0 and math.isclose(w * w + x * x + y * y + z * z, 1.0, abs_tol=1e-8)
        R = _quat_matrix(w, x, y, z)
        body = rot_zyx(c.yaw, c.pitch, c.roll)
        # the optical axis (camera z) is the camera's aim, camera x is the body's right (-y), camera y is down (-z)
        assert np.allclose(R @ [0, 0, 1], body[:, 0], atol=1e-8)
        assert np.allclose(R @ [1, 0, 0], -body[:, 1], atol=1e-8)
        assert np.allclose(R @ [0, 1, 0], -body[:, 2], atol=1e-8)
        # the camera's own origin goes to its translation
        assert np.allclose(R @ np.zeros(3) + row['translation'], c.pos() - ORIGIN, atol=1e-6)


def test_nuscenes_reprojects_like_the_app(rig, exported):
    q, _ = rig
    out, _ = exported
    cal = json.loads((out / 'nuscenes/calibrated_sensor.json').read_text('utf-8'))
    for c, row in zip(q.variant.cameras, cal):
        P, _, uv_app = _app_view(q, c)
        # nuScenes devkit: translate by -translation, then rotate by the quaternion's matrix transposed
        Pc = (P - ORIGIN - np.array(row['translation'])) @ _quat_matrix(*row['rotation'])
        d = row['vantage_distortion']
        uv = _project(Pc, row['camera_intrinsic'], d['coefficients'] or [0.0] * 4, d['model'])
        err = np.linalg.norm(uv - uv_app, axis=1)
        assert err.max() < TOL_PX, (c.tag, err.max())


# ----------------------------------------------------------------------------- KITTI
def test_kitti_layout(rig, exported):
    q, _ = rig
    out, _ = exported
    data, raw = _kitti_read(out / 'kitti/calib_cam_to_cam.txt')
    cams = q.variant.cameras
    assert raw['calib_time'] == '21-Oct-2026 09:30:00'
    assert data['corner_dist'].shape == (1,)
    assert raw['camera_names'].split() == [c.tag for c in cams]
    for i, c in enumerate(cams):
        n = f'{i:02d}'
        for key, size in (('S', 2), ('K', 9), ('D', 5), ('R', 9), ('T', 3), ('S_rect', 2), ('R_rect', 9),
                          ('P_rect', 12)):
            assert data[f'{key}_{n}'].shape == (size,), (key, n)
        assert raw[f'DISTORTION_MODEL_{n}'] == _model_of(q, c)
        assert f'DISTORTION_MODEL_{n}' not in data                     # pykitti skips it
        K = data[f'K_{n}'].reshape(3, 3)
        R = data[f'R_{n}'].reshape(3, 3)
        # KITTI's P_rect_xx projects (rectified) camera 00 coordinates into rectified image xx: K_xx [R_xx | T_xx]
        assert np.allclose(data[f'P_rect_{n}'].reshape(3, 4), K @ np.c_[R, data[f'T_{n}']], rtol=1e-6, atol=1e-3)
        assert np.allclose(data[f'R_rect_{n}'], np.eye(3).ravel()) and np.allclose(data[f'S_rect_{n}'], [1920, 1536])
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-5) and math.isclose(np.linalg.det(R), 1.0, abs_tol=1e-5)
    assert np.allclose(data['R_00'], np.eye(3).ravel()) and np.allclose(data['T_00'], 0)
    assert np.allclose(data['P_rect_00'].reshape(3, 4), np.c_[data['K_00'].reshape(3, 3), np.zeros(3)])
    readme = (out / 'kitti/README.txt').read_text('utf-8')
    assert 'radtan' in readme and 'equidistant' in readme and 'cv2.fisheye.initUndistortRectifyMap' in readme


def test_kitti_reprojects_like_the_app(rig, exported):
    q, _ = rig
    out, _ = exported
    data, raw = _kitti_read(out / 'kitti/calib_cam_to_cam.txt')
    veh, _ = _kitti_read(out / 'kitti/calib_cams_to_vehicle.txt')
    T00 = np.r_[veh['Tr_cam_00_to_vehicle'].reshape(3, 4), [[0, 0, 0, 1]]]
    for i, c in enumerate(q.variant.cameras):
        n = f'{i:02d}'
        P, Pc_app, uv_app = _app_view(q, c)
        Pe = np.c_[P - ORIGIN, np.ones(len(P))]
        X0 = (np.linalg.inv(T00) @ Pe.T)[:3].T                         # vehicle -> camera 00
        X = X0 @ data[f'R_{n}'].reshape(3, 3).T + data[f'T_{n}']        # camera 00 -> camera n, KITTI's R, T
        assert np.allclose(X, Pc_app, atol=2e-5), c.tag                 # %.6e keeps ~1e-6 relative
        Tn = np.r_[veh[f'Tr_cam_{n}_to_vehicle'].reshape(3, 4), [[0, 0, 0, 1]]]
        assert np.allclose((np.linalg.inv(Tn) @ Pe.T)[:3].T, Pc_app, atol=1e-5)
        D, model = data[f'D_{n}'], raw[f'DISTORTION_MODEL_{n}']
        assert D[4] == 0.0
        uv = _project(X, data[f'K_{n}'], D[:4] if model == 'equidistant' else D, model)
        err = np.linalg.norm(uv - uv_app, axis=1)
        assert err.max() < TOL_PX, (c.tag, err.max())


# ----------------------------------------------------------------------------- ROS camera_info
def test_camera_info_reprojects_like_the_app(rig, exported):
    q, _ = rig
    out, _ = exported
    tags = ex.safe_tags(q.variant)
    for c in q.variant.cameras:
        doc = yaml.safe_load((out / 'camera_info' / f'{tags[c.uid]}.yaml').read_text('utf-8'))
        assert doc['image_width'] == 1920 and doc['image_height'] == 1536 and doc['camera_name'] == tags[c.uid]
        assert doc['distortion_model'] == _model_of(q, c)
        K = np.array(doc['camera_matrix']['data']).reshape(3, 3)
        D = doc['distortion_coefficients']['data']
        assert len(D) == doc['distortion_coefficients']['cols'] == (5 if _model_of(q, c) == 'plumb_bob' else 4)
        assert np.allclose(np.array(doc['rectification_matrix']['data']).reshape(3, 3), np.eye(3))
        assert np.allclose(np.array(doc['projection_matrix']['data']).reshape(3, 4), np.c_[K, np.zeros(3)])
        # the keys camera_calibration_parsers writes after P
        assert doc['binning_x'] == doc['binning_y'] == 0 and doc['roi'] == {
            'x_offset': 0, 'y_offset': 0, 'height': 0, 'width': 0, 'do_rectify': False}
        _, Pc, uv_app = _app_view(q, c)
        err = np.linalg.norm(_project(Pc, K, D, doc['distortion_model']) - uv_app, axis=1)
        assert err.max() < TOL_PX, (c.tag, err.max())


# ----------------------------------------------------------------------------- Kalibr
def test_kalibr_chain_reprojects_like_the_app(rig, exported):
    q, _ = rig
    out, _ = exported
    doc = yaml.safe_load((out / 'kalibr/camchain.yaml').read_text('utf-8'))
    cams = q.variant.cameras
    assert list(doc) == [f'cam{i}' for i in range(len(cams))]
    assert 'T_cn_cnm1' not in doc['cam0']
    P, Pc_prev, _ = _app_view(q, cams[0])
    for i, c in enumerate(cams):
        d = doc[f'cam{i}']
        assert d['camera_model'] == 'pinhole' and d['resolution'] == [1920, 1536]
        assert d['rostopic'] == f'/{c.tag.lower()}/image_raw'
        assert d['distortion_model'] == ('radtan' if _model_of(q, c) == 'plumb_bob' else 'equidistant')
        assert all(0 <= j < len(cams) and j != i for j in d['cam_overlaps'])
        assert all(i in doc[f'cam{j}']['cam_overlaps'] for j in d['cam_overlaps'])      # overlap is mutual
        P_n, Pc_app, uv_app = _app_view(q, c)
        if i:
            T = np.array(d['T_cn_cnm1'])
            assert T.shape == (4, 4) and np.allclose(T[3], [0, 0, 0, 1])
            # the previous camera's view of the same points, carried into this camera by the chain
            prev = cams[i - 1]
            W2O_prev = BODY_TO_OPTICAL @ rot_zyx(prev.yaw, prev.pitch, prev.roll).T
            Xp = (P_n - prev.pos()) @ W2O_prev.T
            X = Xp @ T[:3, :3].T + T[:3, 3]
            assert np.allclose(X, Pc_app, atol=1e-8), c.tag
        fu, fv, pu, pv = d['intrinsics']
        K = [[fu, 0, pu], [0, fv, pv], [0, 0, 1]]
        model = 'plumb_bob' if d['distortion_model'] == 'radtan' else 'equidistant'
        err = np.linalg.norm(_project(Pc_app, K, d['distortion_coeffs'], model) - uv_app, axis=1)
        assert err.max() < TOL_PX, (c.tag, err.max())


def test_kalibr_and_camera_info_round_trip_through_the_importer(rig, exported):
    """Read back with core.calib (the lens calibration importer) and applied as unit calibrations, each fisheye
    camera gets back the model it was exported with."""
    q, _ = rig
    out, _ = exported
    cams = q.variant.cameras
    fisheye = [c for c in cams if _model_of(q, c) == 'equidistant']
    with pytest.raises(ValueError, match='not fisheye'):
        calib.read_file(str(out / 'kalibr/camchain.yaml'))             # the rectilinear camera is radtan
    pin = [c for c in cams if c not in fisheye]
    assert pin
    tags = ex.safe_tags(q.variant)
    sources = [calib.read_file(str(out / 'camera_info' / f'{tags[c.uid]}.yaml'))[0] for c in fisheye]

    # Kalibr without the pinhole camera, as a user would trim it
    doc = yaml.safe_load((out / 'kalibr/camchain.yaml').read_text('utf-8'))
    keep = {f'cam{i}': d for i, (c, d) in enumerate(zip(cams, doc.values())) if c in fisheye}
    trimmed = out / 'kalibr_fisheye.yaml'
    trimmed.write_text(yaml.safe_dump(keep), 'utf-8')
    for recs in (sources, calib.read_file(str(trimmed))):
        assert len(recs) == len(fisheye)
        matched, left = calib.match(recs, cams)
        assert not left and set(matched) == {c.uid for c in fisheye}
        r = q.snapshot()
        for c in r.variant.cameras:
            if c.uid in matched:
                c.calib = calib.clean_record(matched[c.uid])
        for c in fisheye:
            a, b = ex.camera_model(q, c), ex.camera_model(r, r.variant.cam(c.uid))
            assert np.allclose([b.fx, b.fy, b.cx, b.cy], [a.fx, a.fy, a.cx, a.cy], atol=1e-5), c.tag
            assert np.allclose(b.k, a.k, rtol=1e-8, atol=1e-11), c.tag


# ----------------------------------------------------------------------------- frames, units, empty
def test_units_do_not_change_dataset_files(rig, tmp_path):
    q, res = rig
    a = ex.export(IDS, q, None, ex.ExportOptions(out_dir=str(tmp_path / 'm'), frame='custom'))
    b = ex.export(IDS, q, None, ex.ExportOptions(out_dir=str(tmp_path / 'mm'), frame='custom', units='mm',
                                                 angles='rad'))
    for x, y in zip(a, b):
        assert open(x, 'rb').read() == open(y, 'rb').read(), x


def test_rear_axle_frame(rig, tmp_path):
    q, _ = rig
    ex.export(['nuscenes_calib', 'kitti_calib'], q, None, ex.ExportOptions(out_dir=str(tmp_path)))
    cal = json.loads((tmp_path / 'nuscenes/calibrated_sensor.json').read_text('utf-8'))
    veh, _ = _kitti_read(tmp_path / 'kitti/calib_cams_to_vehicle.txt')
    for i, (c, row) in enumerate(zip(q.variant.cameras, cal)):
        assert np.allclose(row['translation'], c.pos(), atol=1e-6)
        assert np.allclose(veh[f'Tr_cam_{i:02d}_to_vehicle'].reshape(3, 4)[:, 3], c.pos(), atol=1e-5)


def test_empty_variant(tmp_path, qapp):
    b = blank_project()
    paths = ex.export(IDS, b, None, ex.ExportOptions(out_dir=str(tmp_path)))
    assert json.loads((tmp_path / 'nuscenes/sensor.json').read_text('utf-8')) == []
    assert yaml.safe_load((tmp_path / 'kalibr/camchain.yaml').read_text('utf-8')) == {}
    data, _ = _kitti_read(tmp_path / 'kitti/calib_cam_to_cam.txt')
    assert 'corner_dist' in data and not any(k.startswith('K_') for k in data)
    assert not (tmp_path / 'camera_info').exists() and len(paths) == 7
    _, text, size = ex.preview('ros_camera_info', b, None, ex.ExportOptions(out_dir=str(tmp_path)))
    assert size == 0 and 'No cameras' in text
