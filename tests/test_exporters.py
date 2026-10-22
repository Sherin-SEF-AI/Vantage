"""Tests for vantage.core.exporters.

Builds the Tata Tigor sample project, analyses it and exports every format into tmp_path.
Run with:  QT_QPA_PLATFORM=offscreen python3 -m pytest tests/test_exporters.py -q
"""
from __future__ import annotations

import csv
import datetime as dt
import io
import json
import math
import os
import re
import sys
import xml.etree.ElementTree as ET

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtCore import QBuffer, QIODevice  # noqa: E402
from PyQt6.QtGui import QColor, QGuiApplication, QImage  # noqa: E402

from vantage.core import exporters as ex  # noqa: E402
from vantage.core.analysis import Engine  # noqa: E402
from vantage.core.lens import Lens, r_of_theta, resolve  # noqa: E402
from vantage.core.mathutil import BODY_TO_OPTICAL, rot_zyx  # noqa: E402
from vantage.core.samples import tigor_sample_project  # noqa: E402

FIXED_NOW = dt.datetime(2026, 10, 21, 9, 30, 0, tzinfo=dt.timezone(dt.timedelta(hours=5, minutes=30)))


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


def _png(w, h, color='#336699') -> bytes:
    img = QImage(w, h, QImage.Format.Format_RGB32)
    img.fill(QColor(color))
    buf = QBuffer()
    buf.open(QIODevice.OpenModeFlag.WriteOnly)
    assert img.save(buf, 'PNG')
    return bytes(buf.data())


@pytest.fixture(scope='module')
def images(sample):
    p, _ = sample
    imgs = {'viewport': _png(320, 180), 'bev': _png(200, 200, '#445566'), 'coverage': _png(200, 200, '#556677')}
    for c in p.variant.cameras:
        imgs[f'cam:{c.uid}'] = _png(192, 154, '#222222')
    return imgs


def _opts(tmp_path, images, **kw):
    return ex.ExportOptions(out_dir=str(tmp_path), images=images, **kw)


@pytest.fixture(scope='module')
def exported(sample, images, tmp_path_factory, qapp):
    """Every format exported once with default options (rear axle, m, deg)."""
    p, res = sample
    out = tmp_path_factory.mktemp('all')
    steps = []
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ex, '_now', lambda: FIXED_NOW)
        paths = ex.export([f.id for f in ex.FORMATS], p, res, ex.ExportOptions(out_dir=str(out), images=images),
                          progress=steps.append)
    return out, paths, steps


def _cams(p):
    return p.variant.cameras


# ----------------------------------------------------------------------------- registry
def test_formats_registry():
    ids = [f.id for f in ex.FORMATS]
    assert len(ids) == len(set(ids)) == 25
    assert {f.group for f in ex.FORMATS} == set(ex.GROUPS)
    for f in ex.FORMATS:
        assert f.kind in ('text', 'binary')
        assert f.label and f.ext_label
    with pytest.raises(ValueError):
        ex.format_by_id('nope')


def test_export_writes_every_file(sample, exported):
    p, _ = sample
    out, paths, steps = exported
    n = len(_cams(p))
    # 11 single-file formats + N opencv + 2 ros + N views + (json, npz, txt, 2N bin, body mask) remap tables
    # + 4 reference stitcher sources + 5 bowl files (obj, mtl, png, npz, json) + 2 blind volume files (json, obj)
    # + N guideline tables and N previews + 2 stitched-objects files (json, png) + 2 perception range files (json, png)
    # + datasets: 3 nuScenes, 3 KITTI, N camera_info, 1 Kalibr
    assert len(paths) == 11 + 2 * n + 2 + (2 * n + 4) + 4 + 5 + 2 + 2 * n + 2 + 2 + 3 + 3 + n + 1
    for path in paths:
        assert os.path.isabs(path) and os.path.isfile(path) and os.path.getsize(path) > 0
        assert os.path.commonpath([path, str(out)]) == str(out)
    assert not [f for f in os.listdir(out) if f.endswith('.part')]
    assert steps and steps[-1] == pytest.approx(1.0) and steps == sorted(steps)
    names = {os.path.relpath(x, out).replace(os.sep, '/') for x in paths}
    for expected in ('mount-sheet.pdf', 'report.html', 'report.pdf', 'extrinsics.json', 'extrinsics.yaml',
                     'extrinsics.csv', 'vantage_cameras.urdf', 'static_tf.launch.py', 'jetson_cameras.json',
                     'viewport.png', 'birds-eye.png', 'mount-points.dxf', 'mount-points.step', 'opencv/F.yaml',
                     'views/QR.png', 'bev_lut/bev_lut.json', 'bev_lut/F_map.bin', 'bev_lut/body_mask.bin',
                     'bev_lut/bev_lut.txt', 'bev_lut/stitcher/bev_stitch.cpp', 'bev_lut/stitcher/bev_stitch.py',
                     'bowl/bowl.obj', 'bowl/bowl_lut.npz', 'bowl/bowl.png', 'guidelines/R.json', 'guidelines/R.png',
                     'stitched_objects/stitched_objects.json', 'stitched_objects/stitched_objects.png',
                     'perception_range/perception_range.json', 'perception_range/perception_range.png'):
        assert expected in names


# ----------------------------------------------------------------------------- extrinsics
def test_json_extrinsics(sample, exported):
    p, _ = sample
    out, _, _ = exported
    doc = json.loads((out / 'extrinsics.json').read_text('utf-8'))
    assert doc['schema'] == 'vantage.extrinsics/1'
    assert doc['project'] == p.name and doc['variant'] == p.variant.name
    assert dt.datetime.fromisoformat(doc['exported']) == FIXED_NOW
    fr = doc['frame']
    assert fr['units'] == 'm' and fr['angles'] == 'deg' and fr['axes'] == 'x forward, y left, z up'
    assert fr['rotation'] == 'intrinsic Z-Y-X (yaw, pitch, roll); pitch positive down'
    assert doc['vehicle']['wheelbase'] == pytest.approx(2.45, abs=1e-3)
    assert doc['vehicle']['length'] > 3.5
    cams = doc['cameras']
    assert len(cams) == 6
    for c, jc in zip(_cams(p), cams):
        assert jc['tag'] == c.tag and jc['name'] == c.location and jc['gmsl_port'] == c.port
        assert jc['position'] == pytest.approx([c.x, c.y, c.z], abs=1e-9)
        assert (jc['yaw'], jc['pitch'], jc['roll']) == pytest.approx((c.yaw, c.pitch, c.roll), abs=0.05)
        intr = jc['intrinsics']
        assert intr['model'] == 'kannala_brandt' and len(intr['k']) == 4
        assert intr['fx'] == pytest.approx(2.62 / 0.003, abs=1e-3) and intr['fx'] == intr['fy']
        assert (intr['width'], intr['height'], intr['cx'], intr['cy']) == (1920, 1536, 959.5, 767.5)   # OpenCV pixel centres
        T = np.array(jc['T_world_optical'])
        R = rot_zyx(c.yaw, c.pitch, c.roll)
        assert T.shape == (4, 4)
        assert np.allclose(T[:3, :3], (BODY_TO_OPTICAL @ R.T).T, atol=1e-6)
        assert np.allclose(T[:3, 3], [c.x, c.y, c.z]) and np.allclose(T[3], [0, 0, 0, 1])
        # optical z (column 2) is the camera's aim direction
        assert np.allclose(T[:3, 2], R[:, 0], atol=1e-6)


def test_yaml_matches_json(sample, exported):
    out, _, _ = exported
    y = yaml.safe_load((out / 'extrinsics.yaml').read_text('utf-8'))
    j = json.loads((out / 'extrinsics.json').read_text('utf-8'))
    assert y == j


def test_csv(sample, exported):
    p, _ = sample
    out, _, _ = exported
    rows = list(csv.DictReader(io.StringIO((out / 'extrinsics.csv').read_text('utf-8'))))
    assert len(rows) == 6
    for c, r in zip(_cams(p), rows):
        assert r['tag'] == c.tag
        assert float(r['x_m']) == pytest.approx(c.x) and float(r['y_m']) == pytest.approx(c.y)
        assert float(r['yaw_deg']) == pytest.approx(c.yaw, abs=1e-4) and len(r['yaw_deg'].split('.')[1]) == 4
        assert len(r['x_m'].split('.')[1]) == 3
    assert '−' not in (out / 'extrinsics.csv').read_text('utf-8')


def test_frame_option_shifts_by_front_bumper(sample, images, tmp_path):
    p, res = sample
    ex.export(['extrinsics_json'], p, res, _opts(tmp_path / 'ra', images))
    ex.export(['extrinsics_json'], p, res, _opts(tmp_path / 'fb', images, frame='front_bumper'))
    ra = json.loads((tmp_path / 'ra' / 'extrinsics.json').read_text('utf-8'))
    fb = json.loads((tmp_path / 'fb' / 'extrinsics.json').read_text('utf-8'))
    front_x = p.vehicle.measured()['front_x']
    assert front_x == pytest.approx(3.215, abs=0.01)
    assert p.frame == 'rear_axle'      # the option must not change the project
    assert fb['frame']['id'] == 'front_bumper'
    assert fb['frame']['origin_from_rear_axle'][0] == pytest.approx(front_x, abs=1e-3)
    for a, b in zip(ra['cameras'], fb['cameras']):
        assert b['position'][0] == pytest.approx(a['position'][0] - front_x, abs=1.5e-3)
        assert b['position'][1:] == a['position'][1:]
        assert (b['yaw'], b['pitch'], b['roll']) == (a['yaw'], a['pitch'], a['roll'])
        assert b['T_world_optical'][0][3] == pytest.approx(a['T_world_optical'][0][3] - front_x, abs=1.5e-3)


def test_mm_units_scale_positions(sample, images, tmp_path):
    p, res = sample
    ex.export(['extrinsics_json', 'extrinsics_csv', 'dxf', 'step'], p, res, _opts(tmp_path, images, units='mm'))
    doc = json.loads((tmp_path / 'extrinsics.json').read_text('utf-8'))
    assert doc['frame']['units'] == 'mm'
    for c, jc in zip(_cams(p), doc['cameras']):
        assert jc['position'] == [round(c.x * 1000), round(c.y * 1000), round(c.z * 1000)]
        assert all(isinstance(v, int) for v in jc['position'])
        assert jc['T_world_optical'][0][3] == round(c.x * 1000)
    rows = list(csv.DictReader(io.StringIO((tmp_path / 'extrinsics.csv').read_text('utf-8'))))
    assert rows[0]['x_mm'] == f'{_cams(p)[0].x * 1000:.0f}'
    dxf = (tmp_path / 'mount-points.dxf').read_text('utf-8')
    assert re.search(r'\$INSUNITS\n\s*70\n4\n', dxf)
    step = (tmp_path / 'mount-points.step').read_text('utf-8')
    assert 'SI_UNIT(.MILLI.,.METRE.)' in step
    assert f"CARTESIAN_POINT('F',(3230.,0.,620.))" in step


def test_radian_angles(sample, images, tmp_path):
    p, res = sample
    ex.export(['extrinsics_json'], p, res, _opts(tmp_path, images, angles='rad'))
    doc = json.loads((tmp_path / 'extrinsics.json').read_text('utf-8'))
    for c, jc in zip(_cams(p), doc['cameras']):
        assert jc['yaw'] == pytest.approx(math.radians(c.yaw), abs=1e-6)
        assert jc['pitch'] == pytest.approx(math.radians(c.pitch), abs=1e-6)


def test_bad_options(sample):
    p, res = sample
    for kw in ({'frame': 'roof'}, {'units': 'in'}, {'angles': 'grad'}):
        with pytest.raises(ValueError):
            ex.preview('extrinsics_json', p, res, ex.ExportOptions(**kw))


# ----------------------------------------------------------------------------- OpenCV
def _ground_points(cam, intr, W2O):
    """Ground points (z = 0) that the app projects inside the image, well inside the fisheye range."""
    pos = cam.pos()
    xs, ys = np.meshgrid(np.linspace(-4, 4, 33), np.linspace(-4, 4, 33))
    P = np.stack([pos[0] + xs.ravel(), pos[1] + ys.ravel(), np.zeros(xs.size)], axis=1)
    u, v, theta, ok = intr.project_dirs((P - pos) @ W2O.T)
    keep = ok & (theta < math.radians(80))
    return P[keep], np.stack([u[keep], v[keep]], axis=1)


def test_opencv_yaml_reprojects_like_the_app(sample, exported):
    cv2 = pytest.importorskip('cv2')
    p, _ = sample
    out, _, _ = exported
    for c in _cams(p):
        fs = cv2.FileStorage(str(out / 'opencv' / f'{c.tag}.yaml'), cv2.FILE_STORAGE_READ)
        assert fs.isOpened()
        K = fs.getNode('camera_matrix').mat()
        D = fs.getNode('distortion_coefficients').mat()
        rvec = fs.getNode('rvec').mat()
        tvec = fs.getNode('tvec').mat()
        Rwc = fs.getNode('R_world_camera').mat()
        assert fs.getNode('distortion_model').string() == 'fisheye'
        assert int(fs.getNode('image_width').real()) == 1920 and int(fs.getNode('image_height').real()) == 1536
        assert K.shape == (3, 3) and D.shape == (1, 4) and rvec.shape == (3, 1) and tvec.shape == (3, 1)
        fs.release()

        R = rot_zyx(c.yaw, c.pitch, c.roll)
        W2O = BODY_TO_OPTICAL @ R.T
        Rv, _ = cv2.Rodrigues(rvec)
        assert np.allclose(Rv, W2O, atol=1e-9)              # also covers the yaw-180 rear camera
        assert np.allclose(Rwc, W2O.T, atol=1e-9)
        assert np.allclose(tvec.ravel(), -W2O @ c.pos(), atol=1e-9)

        intr = resolve(p.module_for(c), p.lens_for(c))
        P, uv_app = _ground_points(c, intr, W2O)
        assert len(P) >= 8, c.tag
        idx = np.linspace(0, len(P) - 1, 12).astype(int)
        uv_cv, _ = cv2.fisheye.projectPoints(P[idx].reshape(-1, 1, 3).astype(np.float64), rvec, tvec, K, D)
        # the app's sensor coordinates have pixel centres at i + 0.5, OpenCV's at i
        err = np.linalg.norm(uv_cv.reshape(-1, 2) - (uv_app[idx] - 0.5), axis=1)
        assert err.max() < 0.05, (c.tag, err.max())


def test_kb_fit_for_other_projections(sample, images, tmp_path):
    """A non-KB lens exports k1..k4 fitted to r(theta); the fit reproduces the lens within a pixel."""
    p, res = sample
    q = p.snapshot()
    q.lenses['equi'] = Lens(id='equi', name='EQ190', focal_mm=1.8, projection='equisolid',
                            sheet_fov=(150.0, 120.0, 180.0))
    for c in q.variant.cameras:
        c.lens_id = 'equi'
    cm = ex.camera_model(q, q.variant.cameras[0])
    assert cm.fitted
    intr = cm.intr
    th = np.linspace(0, intr.max_theta, 200)
    k1, k2, k3, k4 = cm.k
    r_kb = cm.fx * th * (1 + k1 * th ** 2 + k2 * th ** 4 + k3 * th ** 6 + k4 * th ** 8)
    r_lens = r_of_theta(th, intr.f_mm, intr.projection) / intr.pixel_mm
    assert np.max(np.abs(r_kb - r_lens)) < 1.0
    assert cm.fit_rms_px < 0.5
    ex.export(['extrinsics_json'], q, None, _opts(tmp_path, images))
    doc = json.loads((tmp_path / 'extrinsics.json').read_text('utf-8'))
    ji = doc['cameras'][0]['intrinsics']
    assert ji['model'] == 'kannala_brandt' and ji['fitted_from'] == 'equisolid'
    assert ji['k'] == pytest.approx(list(cm.k), rel=1e-6)


# ----------------------------------------------------------------------------- ROS 2
def test_urdf(sample, exported):
    p, _ = sample
    out, _, _ = exported
    root = ET.parse(out / 'vantage_cameras.urdf').getroot()
    assert root.tag == 'robot'
    links = [ln.get('name') for ln in root.findall('link')]
    assert len(links) == 13 and links[0] == 'base_link'
    joints = {j.get('name'): j for j in root.findall('joint')}
    assert len(joints) == 12
    for c in _cams(p):
        j = joints[f'{c.tag}_joint']
        assert j.find('parent').get('link') == 'base_link'
        xyz = [float(v) for v in j.find('origin').get('xyz').split()]
        rpy = [float(v) for v in j.find('origin').get('rpy').split()]
        assert xyz == pytest.approx([c.x, c.y, c.z])
        assert rpy == pytest.approx([math.radians(c.roll), math.radians(c.pitch), math.radians(c.yaw)], abs=1e-5)
        oj = joints[f'{c.tag}_optical_joint']
        assert oj.find('child').get('link') == f'{c.tag}_optical_frame'
        assert [float(v) for v in oj.find('origin').get('rpy').split()] == pytest.approx(
            [-math.pi / 2, 0, -math.pi / 2], abs=1e-5)


def _rpy_matrix(r, p_, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p_), math.sin(p_), math.cos(y), math.sin(y)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


def test_urdf_optical_rpy_is_body_to_optical():
    assert np.allclose(_rpy_matrix(-math.pi / 2, 0, -math.pi / 2), BODY_TO_OPTICAL.T)


def _quat_matrix(x, y, z, w):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def test_launch_file(sample, exported):
    p, _ = sample
    out, _, _ = exported
    src = (out / 'static_tf.launch.py').read_text('utf-8')
    compile(src, 'static_tf.launch.py', 'exec')
    assert 'def generate_launch_description' in src and "executable='static_transform_publisher'" in src
    for flag in ('--x', '--y', '--z', '--qx', '--qy', '--qz', '--qw', '--frame-id', '--child-frame-id'):
        assert f"'{flag}'" in src
    calls = re.findall(r"_static_tf\('([^']+)', \(([^)]*)\), \(([^)]*)\), '([^']+)', '([^']+)'\)", src)
    assert len(calls) == 12
    by_child = {c[4]: c for c in calls}
    for c in _cams(p):
        name, xyz, quat, parent, child = by_child[f'{c.tag}_link']
        assert parent == 'base_link'
        assert [float(v.strip(" '")) for v in xyz.split(',')] == pytest.approx([c.x, c.y, c.z])
        q = [float(v.strip(" '")) for v in quat.split(',')]
        assert np.allclose(_quat_matrix(*q), rot_zyx(c.yaw, c.pitch, c.roll), atol=1e-5)
        _, xyz, quat, parent, _ = by_child[f'{c.tag}_optical_frame']
        assert parent == f'{c.tag}_link'
        assert np.allclose(_quat_matrix(*[float(v.strip(" '")) for v in quat.split(',')]), BODY_TO_OPTICAL.T,
                           atol=1e-5)


# ----------------------------------------------------------------------------- Jetson
def test_jetson_config(sample, exported):
    p, _ = sample
    out, _, _ = exported
    doc = json.loads((out / 'jetson_cameras.json').read_text('utf-8'))
    assert doc['platform'] == 'Jetson AGX Orin'
    cams = doc['cameras']
    assert [c['port'] for c in cams] == sorted(c.port for c in _cams(p))
    for jc in cams:
        port = jc['port']
        assert jc['deserializer'] == ('A' if port <= 4 else 'B')
        assert jc['link'] == (port - 1) % 4 + 1
        assert jc['link_label'] == p.ecu.deserializer(port)
        assert jc['device'] == f'/dev/video{port - 1}'
        assert jc['resolution'] == {'width': 1920, 'height': 1536}
        assert jc['cable_m'] == 3.0 and 0 < jc['routed_m'] < 5
        assert jc['sync_group'] == 'A'


# ----------------------------------------------------------------------------- CAD
def _dxf_pairs(text):
    lines = text.split('\n')
    if lines[-1] == '':
        lines.pop()
    assert len(lines) % 2 == 0
    return [(int(lines[i]), lines[i + 1]) for i in range(0, len(lines), 2)]


def test_dxf(sample, exported):
    p, res = sample
    out, _, _ = exported
    text = (out / 'mount-points.dxf').read_text('utf-8')
    pairs = _dxf_pairs(text)
    vals = [v for c, v in pairs if c == 0]
    assert vals[0] == 'SECTION' and vals[-1] == 'EOF'
    assert ('2', 'ENTITIES') in [(str(c), v) for c, v in pairs]
    assert vals.count('SECTION') == vals.count('ENDSEC') == 3
    assert re.search(r'\$INSUNITS\n\s*70\n6\n', text)
    layers = {v for c, v in pairs if c == 8}
    assert layers == {'CAMERAS', 'AIM', 'LABELS', 'OUTLINE'}
    assert vals.count('POINT') == vals.count('LINE') == vals.count('TEXT') == 6
    assert vals.count('POLYLINE') == len(res.grid.outline) == vals.count('SEQEND')
    # first camera: POINT at its position, aim LINE 0.3 m long along the optical axis
    c = _cams(p)[0]
    i = pairs.index((0, 'POINT'))
    pt = {code: float(v) for code, v in pairs[i + 1:i + 6] if code in (10, 20, 30)}
    assert (pt[10], pt[20], pt[30]) == pytest.approx((c.x, c.y, c.z))
    i = pairs.index((0, 'LINE'))
    ln = {code: float(v) for code, v in pairs[i + 1:i + 10] if code in (10, 20, 30, 11, 21, 31)}
    a = np.array([ln[10], ln[20], ln[30]])
    b = np.array([ln[11], ln[21], ln[31]])
    assert np.linalg.norm(b - a) == pytest.approx(0.3, abs=2e-3)
    assert np.allclose((b - a) / np.linalg.norm(b - a), rot_zyx(c.yaw, c.pitch, c.roll)[:, 0], atol=1e-2)


def test_step(sample, exported):
    p, _ = sample
    out, _, _ = exported
    text = (out / 'mount-points.step').read_text('utf-8')
    assert text.startswith('ISO-10303-21;')
    assert text.rstrip().endswith('END-ISO-10303-21;')
    assert "FILE_SCHEMA(('AUTOMOTIVE_DESIGN" in text
    for kw in ('FILE_DESCRIPTION(', 'FILE_NAME(', 'HEADER;', 'DATA;', 'GEOMETRIC_CURVE_SET(',
               'GEOMETRIC_REPRESENTATION_CONTEXT(3)', 'SI_UNIT($,.METRE.)'):
        assert kw in text
    assert all(ord(ch) < 128 for ch in text)
    data = text.split('DATA;', 1)[1].split('ENDSEC;', 1)[0]
    ents = re.findall(r'^#(\d+)=(.*);$', data.strip(), re.M)
    ids = [int(i) for i, _ in ents]
    assert ids == list(range(1, len(ids) + 1))
    for _, body in ents:
        assert body.count('(') == body.count(')')
        for ref in re.findall(r'#(\d+)', body.replace("'Context #1'", '')):
            assert 1 <= int(ref) <= len(ids)
    assert len(re.findall(r'=AXIS2_PLACEMENT_3D\(', data)) == 7      # 6 cameras + the frame origin
    assert len(re.findall(r'=CARTESIAN_POINT\(', data)) == 1 + 2 * 6
    c = _cams(p)[0]
    assert f"CARTESIAN_POINT('{c.tag}',({c.x:.3f},0.000,{c.z:.3f}))" in text


# ----------------------------------------------------------------------------- documents
def test_pdfs(exported):
    out, _, _ = exported
    for name in ('mount-sheet.pdf', 'report.pdf'):
        data = (out / name).read_bytes()
        assert data.startswith(b'%PDF')
        assert b'%%EOF' in data[-64:]


def test_pdf_pages_render(exported, qapp):
    QtPdf = pytest.importorskip('PyQt6.QtPdf')
    from PyQt6.QtCore import QSize
    out, _, _ = exported
    doc = QtPdf.QPdfDocument(None)
    doc.load(str(out / 'mount-sheet.pdf'))
    assert doc.status() == QtPdf.QPdfDocument.Status.Ready
    assert doc.pageCount() >= 1
    sz = doc.pagePointSize(0)
    assert sz.width() > sz.height()                      # A4 landscape
    assert abs(sz.width() - 841.9) < 2 and abs(sz.height() - 595.3) < 2
    img = doc.render(0, QSize(600, 424))
    assert not img.isNull()
    doc.close()


def test_report_html(sample, exported, images):
    p, res = sample
    out, _, _ = exported
    text = (out / 'report.html').read_text('utf-8')
    assert text.startswith('<!DOCTYPE html>')
    assert '<link' not in text and 'src="http' not in text and '<script' not in text
    assert text.count('data:image/png;base64,') == len(images)
    for c in _cams(p):
        assert f'<b>{c.tag}</b>' in text
    assert 'Coverage per camera' in text and 'Seams' in text and 'Pixel density' in text and 'Issues' in text
    assert f'{res.metrics["covered_pct"]:.2f} %' in text


def test_mount_sheet_lists_supplied_issues(sample, images, tmp_path, qapp):
    """Many issues overflow onto a second sheet, and the preview says so."""
    from vantage.core.rules import Issue
    p, res = sample
    issues = [Issue(['critical', 'serious', 'warning'][i % 3], f'Issue number {i} with a long enough message to '
                    'wrap across two lines in the right column of the mount sheet.', 'F, R') for i in range(14)]
    opts = _opts(tmp_path, images, issues=issues, worst_single=(0.42, _cams(p)[2].uid, (5.0, -5.0, 0, 0, 0, 0)))
    rel, desc, size = ex.preview('mount_sheet_pdf', p, res, opts)
    assert rel.endswith('/mount-sheet.pdf')
    assert desc.startswith('A4 landscape · 2 pages · camera table, plan view, 14 issues')
    assert size > 1000
    path, = ex.export(['mount_sheet_pdf'], p, res, opts)
    assert open(path, 'rb').read(4) == b'%PDF'


# ----------------------------------------------------------------------------- images
def test_missing_image_raises(sample, tmp_path):
    p, res = sample
    with pytest.raises(ValueError, match='viewport'):
        ex.export(['extrinsics_json', 'viewport_png'], p, res, ex.ExportOptions(out_dir=str(tmp_path), images={}))
    assert not os.listdir(tmp_path)                       # nothing written before the check fails
    with pytest.raises(ValueError, match='camera view'):
        ex.export(['camera_views_png'], p, res,
                  ex.ExportOptions(out_dir=str(tmp_path), images={'cam:' + _cams(p)[0].uid: _png(8, 8)}))
    with pytest.raises(ValueError, match='not PNG'):
        ex.export(['bev_png'], p, res, ex.ExportOptions(out_dir=str(tmp_path), images={'bev': b'GIF89a...'}))


def test_images_written_verbatim(sample, exported, images):
    out, _, _ = exported
    assert (out / 'viewport.png').read_bytes() == images['viewport']
    assert (out / 'birds-eye.png').read_bytes() == images['bev']
    p, _ = sample
    for c in _cams(p):
        assert (out / 'views' / f'{c.tag}.png').read_bytes() == images[f'cam:{c.uid}']


# ----------------------------------------------------------------------------- preview
def test_preview_text_matches_export(sample, images, tmp_path):
    p, res = sample
    opts = _opts(tmp_path / 'exports' / 'v3', images)
    for f in ex.FORMATS:
        rel, text, size = ex.preview(f.id, p, res, opts)
        assert rel.startswith('exports/v3/'), rel
        if f.kind != 'text':
            continue
        paths = ex.export([f.id], p, res, opts)
        first = open(paths[0], encoding='utf-8').read()
        assert size == sum(os.path.getsize(x) for x in paths)
        assert rel == 'exports/v3/' + os.path.relpath(paths[0], tmp_path / 'exports' / 'v3').replace(os.sep, '/')
        if f.multi:
            assert text.startswith(first) and len(text) > len(first)
            assert 'Also written' in text[len(first):]
        else:
            assert text == first


def test_preview_binary_descriptions(sample, images, tmp_path):
    p, res = sample
    opts = _opts(tmp_path, images)
    _, d, n = ex.preview('mount_sheet_pdf', p, res, opts)
    assert d.startswith('A4 landscape · 1 page · camera table, plan view, ') and n > 1000
    _, d, n = ex.preview('report_pdf', p, res, opts)
    assert d.startswith('A4 portrait · ') and 'pages' in d and n > 1000
    _, d, n = ex.preview('viewport_png', p, res, opts)
    assert d.startswith('PNG · 320 × 180 px') and n == len(images['viewport'])
    _, d, n = ex.preview('viewport_png', p, res, ex.ExportOptions(out_dir=str(tmp_path)))
    assert 'Not available' in d and n == 0
    rel, d, n = ex.preview('camera_views_png', p, res, opts)
    assert d.startswith('6 PNG files') and rel.endswith('/views/F.png')


def test_bev_lut_files(sample, exported):
    """Remap tables: sizes match the JSON header and the weights of all cameras sum to 255 where covered."""
    p, _ = sample
    out, _, _ = exported
    meta = json.loads((out / 'bev_lut' / 'bev_lut.json').read_text('utf-8'))
    w, h = meta['width'], meta['height']
    assert meta['schema'] == 'vantage.bev_lut/1' and len(meta['cameras']) == len(_cams(p))
    tot = np.zeros((h, w), int)
    for c in meta['cameras']:
        mp = np.fromfile(out / 'bev_lut' / c['map'], np.float32).reshape(h, w, 2)
        wt = np.fromfile(out / 'bev_lut' / c['weight'], np.uint8).reshape(h, w)
        assert ((mp[..., 0] >= 0) | (wt == 0)).all()
        assert mp[..., 0].max() <= c['sensor_width'] and mp[..., 1].max() <= c['sensor_height']
        tot += wt
    body = np.fromfile(out / 'bev_lut' / 'body_mask.bin', np.uint8).reshape(h, w).astype(bool)
    cov = tot > 0
    assert cov.mean() > 0.8 and not (cov & body).any()
    assert np.abs(tot[cov] - 255).max() <= len(meta['cameras'])        # per-camera rounding only
    npz = np.load(out / 'bev_lut' / 'bev_lut.npz')
    assert npz['body_mask'].shape == (h, w)
    _, d, n = ex.preview('bev_lut', p, sample[1], ex.ExportOptions(out_dir=str(out)))
    assert d.startswith(f'{len(meta["cameras"]) * 2 + 4} files · {w} × {h} px') and n > 0
    # the plain-text manifest the C++ reference stitcher reads agrees with the JSON
    man = [ln.split() for ln in (out / 'bev_lut' / 'bev_lut.txt').read_text('utf-8').splitlines() if not ln.startswith('#')]
    kv = {r[0]: r[1:] for r in man}
    assert int(kv['width'][0]) == w and int(kv['height'][0]) == h and kv['body'] == ['body_mask.bin']
    cams = [r[1:] for r in man if r[0] == 'camera']
    assert [c[1] for c in cams] == [c['map'] for c in meta['cameras']]
    assert [int(c[3]) for c in cams] == [c['sensor_width'] for c in meta['cameras']]


def test_number_formatting():
    assert ex._fs(-0.0001, 3) == '0.000'
    assert ex._fs(-1.23456, 1) == '-1.2'
    assert ex._fn(-0.0001, 3) == 0.0 and math.copysign(1, ex._fn(-0.0001, 3)) == 1
    assert ex._fn(1234.4, 0) == 1234 and isinstance(ex._fn(1234.4, 0), int)
    assert ex._step_str("it's · ok") == "'it''s \\X2\\00B7\\X0\\ ok'"
    assert ex._cvnum(960) == '960.' and ex._cvnum(-0.0) == '0.'


# ----------------------------------------------------------------------------- robustness
def test_exports_without_analysis_result(sample, tmp_path, qapp):
    """No mesh analysis (result None): every non-image format still exports."""
    p, _ = sample
    ids = [f.id for f in ex.FORMATS if f.group not in ('Images', 'Surround view')]
    paths = ex.export(ids, p, None, ex.ExportOptions(out_dir=str(tmp_path)))
    with pytest.raises(ValueError, match='analysis result'):
        ex.export(['bev_lut'], p, None, ex.ExportOptions(out_dir=str(tmp_path / 'lut')))
    assert not (tmp_path / 'lut').exists()
    # opencv and camera_info: one per camera; ros2, the blind volume and the perception range: two;
    # nuScenes and KITTI: three
    assert len(paths) == (len(ids) - 7) + 2 * len(_cams(p)) + 2 + 2 + 2 + 3 + 3
    html_text = (tmp_path / 'report.html').read_text('utf-8')
    assert 'No analysis result' in html_text
    _, desc, _ = ex.preview('report_pdf', p, None, ex.ExportOptions(out_dir=str(tmp_path)))
    assert 'no analysis result' in desc


def test_empty_variant(tmp_path, qapp):
    from vantage.core.samples import blank_project
    b = blank_project()
    ids = [f.id for f in ex.FORMATS if f.group not in ('Images', 'Surround view')]
    ex.export(ids, b, None, ex.ExportOptions(out_dir=str(tmp_path)))
    assert json.loads((tmp_path / 'extrinsics.json').read_text('utf-8'))['cameras'] == []
    assert len(ET.parse(tmp_path / 'vantage_cameras.urdf').getroot().findall('link')) == 1
    compile((tmp_path / 'static_tf.launch.py').read_text('utf-8'), 'static_tf.launch.py', 'exec')
    assert (tmp_path / 'mount-points.step').read_text('utf-8').rstrip().endswith('END-ISO-10303-21;')
    rel, text, size = ex.preview('opencv_yaml', b, None, ex.ExportOptions(out_dir=str(tmp_path)))
    assert size == 0 and 'No cameras' in text


def test_duplicate_tags_and_custom_origin(sample, tmp_path):
    p, _ = sample
    q = p.snapshot()
    q.variant.cameras[2].tag = 'F'
    q.variant.cameras[3].tag = 'M R/2'
    q.custom_origin = [1.0, 0.0, 0.3]
    names = ex.format_by_id('opencv_yaml').filename(q, q.variant)
    assert names == ['opencv/F.yaml', 'opencv/R.yaml', 'opencv/F_2.yaml', 'opencv/M_R_2.yaml', 'opencv/QL.yaml',
                     'opencv/QR.yaml']
    ex.export(['ros2', 'extrinsics_json', 'opencv_yaml'], q, None, ex.ExportOptions(out_dir=str(tmp_path),
                                                                                    frame='custom'))
    links = [ln.get('name') for ln in ET.parse(tmp_path / 'vantage_cameras.urdf').getroot().findall('link')]
    assert len(links) == len(set(links)) == 13 and 'M_R_2_optical_frame' in links
    doc = json.loads((tmp_path / 'extrinsics.json').read_text('utf-8'))
    c0 = q.variant.cameras[0]
    assert doc['cameras'][0]['position'] == pytest.approx([c0.x - 1.0, c0.y, c0.z - 0.3], abs=1e-9)
    assert doc['cameras'][2]['tag'] == 'F'          # the tag itself is kept in the data
    assert len(os.listdir(tmp_path / 'opencv')) == 6


def test_tolerance_wording():
    assert ex._describe_pert((5.0, -5.0, 0, 0, 0, 0)) == 'off by yaw +5.0°, pitch -5.0°'
    assert ex._describe_pert((0, 0, 0, 0.0, 0.1, 0.0)) == 'moved 100 mm along the body'
