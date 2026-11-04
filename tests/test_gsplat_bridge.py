"""The 3D Gaussian Splatting bridge (core/gsplat_bridge.py): the OpenCV-fisheye parameters reproduce Vantage's lens
(pixel -> ray -> pixel), the axis conventions on a known pose (Nerfstudio's OpenGL camera-to-world, COLMAP's OpenCV
world-to-camera), the timing and the files of Nerfstudio, 3DGRUT (COLMAP text) and NCore."""
import json
import math

import numpy as np
import pytest

from vantage.core import gsplat_bridge as GB
from vantage.core import samples
from vantage.core.lens import resolve
from vantage.core.project import Camera


@pytest.fixture(scope='module')
def rig():
    p = samples.tigor_sample_project()
    p.active_uid = next(v.uid for v in p.variants if v.sensors)
    return p


def test_fisheye_params_reproduce_the_lens(rig):
    cam = rig.variant.by_tag('F')
    par = GB.camera_params(rig, cam)
    intr = resolve(rig.module_for(cam), rig.lens_for(cam))
    assert par['fit_rms_px'] == 0.0 and par['cx'] == pytest.approx(intr.cx) and par['fx'] == pytest.approx(intr.f_px)
    rng = np.random.default_rng(1)
    u = rng.uniform(0, intr.width, 400)
    v = rng.uniform(0, intr.height, 400)
    d, th = intr.unproject(u, v)
    keep = (th < min(intr.max_theta, math.radians(85))) & (d[:, 2] > 0.05)
    uu, vv = GB.project_opencv_fisheye(par, d[keep])
    assert np.abs(uu - u[keep]).max() < 1e-6 and np.abs(vv - v[keep]).max() < 1e-6
    # against OpenCV itself, whose pixel centres are at integers (cx - 0.5)
    cv2 = pytest.importorskip('cv2')
    K = np.array([[par['fx'], 0, par['cx'] - 0.5], [0, par['fy'], par['cy'] - 0.5], [0, 0, 1]])
    D = np.array([par['k1'], par['k2'], par['k3'], par['k4']])
    pts, _ = cv2.fisheye.projectPoints(d[keep].reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, D)
    assert np.abs(pts[:, 0, 0] - (u[keep] - 0.5)).max() < 1e-4


def test_axis_conventions_on_a_known_pose():
    cam = Camera(tag='T', x=1.0, y=2.0, z=3.0, yaw=90.0, pitch=0.0, roll=0.0)      # looking along +y, level
    Rvo = GB.optical_to_vehicle(cam)
    assert np.allclose(Rvo[:, 2], [0, 1, 0], atol=1e-12)                       # optical z (forward) = vehicle +y
    assert np.allclose(Rvo[:, 1], [0, 0, -1], atol=1e-12)                      # optical y (down) = vehicle -z
    T = GB.c2w_opengl(Rvo, cam.pos())
    assert np.allclose(-T[:3, 2], [0, 1, 0]) and np.allclose(T[:3, 1], [0, 0, 1])   # looks down -Z, +Y up
    assert np.allclose(T[:3, 0], [1, 0, 0]) and np.allclose(T[:3, 3], [1, 2, 3])
    R, t = GB.w2c_opencv(Rvo, cam.pos())
    P = np.array([1.0, 7.0, 3.0])                                              # 5 m straight ahead
    assert np.allclose(R @ P + t, [0, 0, 5], atol=1e-12)
    # in a world where the ego has driven 10 m forward and turned 90° left
    Tw = np.eye(4)
    Tw[:3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float)
    Tw[:3, 3] = (10, 0, 0)
    R, t = GB.w2c_opencv(Rvo, cam.pos(), Tw)
    Pw = Tw[:3, :3] @ P + Tw[:3, 3]
    assert np.allclose(R @ Pw + t, [0, 0, 5], atol=1e-12)
    assert np.allclose(GB.c2w_opengl(Rvo, cam.pos(), Tw)[:3, 3], Tw[:3, :3] @ cam.pos() + Tw[:3, 3])


def test_trajectories():
    tr = GB.constant_velocity_trajectory(10.0, 0.0, 1.0, 10.0)
    assert len(tr) == 11 and tr[-1][0] == pytest.approx(1.0) and tr[-1][1][0, 3] == pytest.approx(10.0)
    T = GB.interp_pose(tr, 0.55)
    assert T[0, 3] == pytest.approx(5.5)
    tr = GB.constant_velocity_trajectory(10.0, 90.0, 1.0, 10.0)                # a quarter turn in 1 s
    T = GB.interp_pose(tr, 1.0)
    assert math.degrees(math.atan2(T[1, 0], T[0, 0])) == pytest.approx(90.0)


def test_rig_timing(rig):
    tm = GB.rig_timing(rig)
    assert len(tm['cameras']) == len([c for c in rig.variant.cameras if c.visible])
    c = tm['cameras'][0]
    assert c['shutter_type'] in ('ROLLING_TOP_TO_BOTTOM', 'GLOBAL')
    assert c['neurad']['rolling_shutter_time'] == pytest.approx(c['readout_ms'] / 1000)
    assert c['neurad']['time_to_center_pixel'] == pytest.approx((c['readout_ms'] + c['exposure_ms']) / 2000)
    lid = tm['lidars'][0]
    m = rig.sensor_model_for(next(s for s in rig.variant.sensors if s.kind == 'lidar'))
    assert lid['channels'] == len(m.elevations()) and lid['rate_hz'] == m.rate_hz
    assert lid['ncore']['spinning_direction'] == m.rotation
    assert len(lid['ncore']['column_azimuths_rad']) == round(360.0 / m.h_res)
    assert lid['neurad']['elevation_mapping']['0'] == pytest.approx(min(m.elevations()))


def test_export_files_parse(rig, tmp_path):
    paths = GB.export_gsplat_bridge(rig, None, str(tmp_path), speed_mps=8.0, duration_s=1.0, rate_hz=5.0)
    assert all(p.startswith(str(tmp_path)) for p in paths)
    ncam = len([c for c in rig.variant.cameras if c.visible])
    doc = json.loads((tmp_path / 'nerfstudio' / 'transforms.json').read_text())
    assert doc['camera_model'] == 'OPENCV_FISHEYE' and len(doc['frames']) == 6 * ncam
    f = doc['frames'][0]
    for k in ('fl_x', 'fl_y', 'cx', 'cy', 'w', 'h', 'k1', 'k2', 'k3', 'k4', 'transform_matrix', 'file_path'):
        assert all(k in g for g in doc['frames']), k                          # per-frame fields on every frame
    T = np.array(f['transform_matrix'])
    assert T.shape == (4, 4) and np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-8)
    # the frame of a later sample sits where the ego has driven at that camera's own time
    cam = rig.variant.by_tag(f['vantage_camera'])
    last = [g for g in doc['frames'] if g['vantage_camera'] == cam.tag][-1]
    assert np.array(last['transform_matrix'])[0, 3] == pytest.approx(cam.x + 8.0 * last['time'], abs=1e-4)   # time rounded to 1 µs
    sp = tmp_path / '3dgut' / 'sparse' / '0'
    cams = [l for l in (sp / 'cameras.txt').read_text().splitlines() if l and not l.startswith('#')]
    assert len(cams) == ncam and all(l.split()[1] == 'OPENCV_FISHEYE' and len(l.split()) == 12 for l in cams)
    imgs = [l for l in (sp / 'images.txt').read_text().splitlines() if not l.startswith('#')]
    data = [l for l in imgs[::2] if l]
    assert len(data) == 6 * ncam and all(len(l.split()) == 10 for l in data)
    q = np.array([float(x) for x in data[0].split()[1:5]])
    assert np.linalg.norm(q) == pytest.approx(1.0) and q[0] >= 0
    pts = [l for l in (sp / 'points3D.txt').read_text().splitlines() if l and not l.startswith('#')]
    assert len(pts) > 100 and len(pts[0].split()) == 8
    nc = json.loads((tmp_path / '3dgut' / 'ncore_sensors.json').read_text())
    assert len(nc['cameras']) == ncam and nc['lidars'][0]['model'] == 'RowOffsetStructuredSpinningLidarModel'
    tm = json.loads((tmp_path / 'rig_timing.json').read_text())
    assert tm['schema'] == GB.SCHEMA and tm['lidars']
