"""Tests for the synthetic sample dataset (core.drivescene, core.dataset) and its page (ui.workspaces.dataset_ws).

The maths is checked against independent references: finite differences of the drive for the IMU and the speed,
the meridian and prime-vertical radii for the geodetic conversion, cv2.fisheye with the exported Kannala-Brandt
coefficients for the rendered images, the nuscenes_calib export for the calibration, the sync plan for the camera
times. A short generation (1 s, small images) of the sample data rig with an IMU and two GNSS antennas is checked for
table integrity, file formats and the bag's topics (the nuScenes devkit itself is not needed here).
Run with:  QT_QPA_PLATFORM=offscreen python3 -m pytest tests/test_dataset.py -q
"""
from __future__ import annotations

import copy
import glob
import json
import math
import os
import re
import sys
import time

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np  # noqa: E402
import pytest  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vantage.core import dataset as DSET  # noqa: E402
from vantage.core import drivescene as DS  # noqa: E402
from vantage.core import exporters as EX  # noqa: E402
from vantage.core import sensors as SN  # noqa: E402
from vantage.core import timing as TM  # noqa: E402
from vantage.core.analysis import Engine  # noqa: E402
from vantage.core.project import Sensor  # noqa: E402
from vantage.core.samples import tigor_sample_project  # noqa: E402

from test_gui import app, win, pump, settle, _wait  # noqa: E402,F401  (fixtures)

cv2 = pytest.importorskip('cv2')
DUR, SCALE = 1.0, 0.15


# ----------------------------------------------------------------------------- fixtures
def _rig():
    """The sample's data rig (6 cameras, a roof lidar, 5 radars) with an INS and two GNSS antennas, active."""
    p = tigor_sample_project()
    v = next(v for v in p.variants if v.sensors)
    p.active_uid = v.uid
    models = {m.id: m for m in SN.builtin_sensor_models()}
    for kind, mid in (('imu', 'ins_dual'), ('gnss', 'gnss_survey'), ('gnss', 'gnss_survey')):
        pose = SN.default_pose(kind, models[mid], p.vehicle.mesh, p.rig, v.sensors)
        v.sensors.append(Sensor(kind=kind, model_id=mid, tag=SN.next_tag(kind, v.tags()), **pose))
    return p


@pytest.fixture(scope='module')
def engine():
    return Engine()


@pytest.fixture(scope='module')
def gen(tmp_path_factory, engine):
    """A 1 s drive at 30 km/h, images at 15 %, both outputs."""
    p = _rig()
    st = DSET.DatasetSettings.from_dict({'duration_s': DUR, 'image_scale': SCALE})
    out = str(tmp_path_factory.mktemp('dataset'))
    calls = []
    res = DSET.generate(p, engine, st, out, lambda f, text='': calls.append(f))
    assert res is not None
    root = os.path.join(out, 'nuscenes', DSET.VERSION)
    tables = {}
    for f in sorted(glob.glob(os.path.join(root, '*.json'))):
        tables[os.path.basename(f)[:-5]] = json.load(open(f))
    return {'p': p, 'st': st, 'res': res, 'out': out, 'nus': os.path.join(out, 'nuscenes'), 't': tables,
            'progress': calls}


@pytest.fixture(scope='module')
def at_rest(tmp_path_factory, engine):
    """The same rig standing still: a short bag only."""
    p = _rig()
    st = DSET.DatasetSettings.from_dict({'duration_s': 0.6, 'speed_kmh': 0, 'image_scale': 0.05, 'nuscenes': False})
    out = str(tmp_path_factory.mktemp('at_rest'))
    res = DSET.generate(p, engine, st, out)
    return {'p': p, 'st': st, 'res': res}


def _by_token(rows):
    return {r['token']: r for r in rows}


def _channel_of(t):
    cs = _by_token(t['calibrated_sensor'])
    se = _by_token(t['sensor'])
    return lambda sd: se[cs[sd['calibrated_sensor_token']]['sensor_token']]['channel']


# ----------------------------------------------------------------------------- settings
def test_settings_from_dict_is_tolerant():
    st = DSET.DatasetSettings.from_dict({
        'duration_s': '1e9', 'speed_kmh': -5, 'seed': 3.7, 'image_scale': 'abc', 'scene': 'moon',
        'camera_rate': 'module', 'nuscenes': 'no', 'rosbag': 0, 'jpeg_quality': float('nan'), 'bogus': 1,
        'origin_lat': 95, 'keyframe_hz': True, 'undistort': 'yes', 'channel_names': 'nuscenes'})
    assert st.duration_s == DSET.LIMITS['duration_s'][1] and st.speed_kmh == 0.0 and st.seed == 4
    assert st.image_scale == 0.5 and st.scene == 'urban' and st.camera_rate == 'module'
    assert not st.nuscenes and not st.rosbag and st.undistort and st.channel_names == 'nuscenes'
    assert st.jpeg_quality == 90 and st.origin_lat == 85.0 and st.keyframe_hz == 2.0
    assert isinstance(st.seed, int) and isinstance(st.jpeg_quality, int)
    assert DSET.DatasetSettings.from_dict(st.to_dict()) == st
    assert DSET.DatasetSettings.from_dict(None) == DSET.DatasetSettings()
    assert DSET.DatasetSettings.from_dict('junk') == DSET.DatasetSettings()


# ----------------------------------------------------------------------------- the scene and the drive
def test_road_round_trip_and_drive_derivatives():
    sc = DS.build_scene('urban', 30, 8, 1)
    rd = sc.road
    s = np.linspace(-50, rd.L1 + rd.Larc + 60, 400)
    d = np.linspace(-12, 12, 400)
    x, y, _ = rd.to_world(s, d)
    s2, d2 = rd.to_sd(x, y)
    assert np.allclose(s2, s, atol=1e-8) and np.allclose(d2, d, atol=1e-8)
    xa, ya, psi = rd.frame(s)
    xb, yb, _ = rd.frame(s + 1e-5)
    assert np.allclose(np.hypot(xb - xa, yb - ya), 1e-5, rtol=1e-4)            # s is arc length
    assert np.allclose(DS._wrap(np.arctan2(yb - ya, xb - xa) - psi), 0, atol=1e-4)
    dr = sc.drive
    t = np.linspace(0.05, 7.95, 400)
    h = 1e-4
    xa, ya, pa = dr.pose(t - h)
    xb, yb, pb = dr.pose(t + h)
    x, y, psi = dr.pose(t)
    assert np.allclose(np.hypot(xb - xa, yb - ya) / (2 * h), dr.speed(t), atol=1e-4)
    assert np.allclose(DS._wrap(np.arctan2(yb - ya, xb - xa) - psi), 0, atol=1e-5)
    k = dr.kinematics(t)
    assert np.allclose(DS._wrap(pb - pa) / (2 * h), k['omega'], atol=1e-5)
    assert np.allclose((dr.speed(t + h) - dr.speed(t - h)) / (2 * h), k['a'], atol=1e-5)
    # the ego stays in its lane, through the bend too
    _, dd = rd.to_sd(x, y)
    assert np.all(np.abs(dd + rd.p['lane_w'] / 2) < 0.25)
    assert sc.kind == 'urban' and any(o.category for o in sc.objects)
    assert 0 < sc.extent[2] and min(x.min(), y.min()) > 0                       # the scene at positive x, y


def test_scene_is_seeded_and_traffic_keeps_clear():
    a, b = DS.build_scene('urban', 30, 8, 5), DS.build_scene('urban', 30, 8, 5)
    c = DS.build_scene('urban', 30, 8, 6)
    pa, pb = a.prims_at(3.0), b.prims_at(3.0)
    assert pa.n == pb.n and np.array_equal(pa.bc, pb.bc)
    assert c.prims_at(3.0).n != pa.n or not np.array_equal(c.prims_at(3.0).bc, pa.bc)
    # no two moving actors (nor an actor and the ego) overlap during the drive
    tt = np.linspace(0, 8, 81)
    mov = [o for o in a.objects if o.moving]
    ex, ey, ep = a.drive.pose(tt)
    for o in mov:
        x, y, _ = a.actor_pose(o, tt)
        gap = np.hypot(x - (ex + 1.2 * np.cos(ep)), y - (ey + 1.2 * np.sin(ep)))
        assert gap.min() > 0.5 * o.size[1] + 0.8, o.name
    hw = DS.build_scene('highway', 100, 6, 2)
    assert hw.kind == 'highway' and sum(1 for o in hw.objects if o.category) >= 5


def test_geodetic_against_the_ellipsoid_radii():
    lat0, lon0, h0 = 13.0827, 80.2707, 7.0
    lat, lon, h = DS.enu_to_geodetic([[0, 1000, 0], [1000, 0, 0], [0, 0, 50]], lat0, lon0, h0)
    a, e2 = 6378137.0, 6.69437999014e-3
    sp = math.sin(math.radians(lat0))
    M = a * (1 - e2) / (1 - e2 * sp * sp) ** 1.5                                 # meridian radius
    N = a / math.sqrt(1 - e2 * sp * sp)                                          # prime vertical radius
    assert abs(lat[0] - lat0 - math.degrees(1000 / (M + h0))) < 1e-8
    assert abs(lon[1] - lon0 - math.degrees(1000 / ((N + h0) * math.cos(math.radians(lat0))))) < 1e-8
    assert abs(lat[2] - lat0) < 1e-12 and abs(lon[2] - lon0) < 1e-12 and abs(h[2] - h0 - 50) < 1e-6
    assert abs(h[0] - h0 - 1000 ** 2 / (2 * M)) < 2e-3                            # the tangent plane rises
    rng = np.random.default_rng(1)
    E = rng.uniform(-2000, 2000, (50, 3))
    back = DS.geodetic_to_enu(*DS.enu_to_geodetic(E, -33.9, 151.2, 40.0), -33.9, 151.2, 40.0)
    assert np.abs(back - E).max() < 1e-6


# ----------------------------------------------------------------------------- sensors
def _wall_scene(speed_kmh, x_face=28.0):
    """A straight road and a single static wall across it, its face x_face metres ahead of the ego's start."""
    road = DS.Road(10.0, 10.0, 0.0, 600.0, 50.0, 1, 0.5, DS.URBAN)
    drive = DS.EgoDrive(road, speed_kmh, 1.0, -1.75)
    x0, y0, _ = road.to_world(0.0, -1.75)
    wall = DS.building_object(1, (float(x0) + x_face + 0.5, float(y0), 0.0), 1.0, 40.0, 8.0, (0.7, 0.7, 0.7))
    return DS.Scene(road, drive, [wall]), float(x0) + x_face


def _lidar(p):
    return next(s for s in p.variant.sensors if s.kind == 'lidar')


def test_lidar_points_on_a_face_and_motion_skew(engine):
    p = _rig()
    lid = _lidar(p)
    ls = DS.LidarSim(p, engine, lid)
    sig = 0.015
    # standing still: the wall's points lie on its face within the range noise
    sc, xf = _wall_scene(0.0)
    sw = ls.sweep(sc, 0.0, sig, True, seed=3)
    m = sw['obj'] == 1
    assert m.sum() > 2000
    Pv = sw['xyz'][m].astype(float) @ ls.R_vs.T + ls.pos
    x0, y0, _ = (float(v) for v in sc.drive.pose(0.0))
    err = (Pv[:, 0] + x0) - xf
    assert np.mean(np.abs(err) < 3 * sig) > 0.99 and np.abs(err).max() < 5 * sig
    assert 0.7 * sig < np.sqrt(np.mean(err ** 2)) < 1.2 * sig
    assert np.allclose(sw['t'][:5], 0.0) and sw['t'].max() < ls.period
    # driving: each column fires from where the ego is at that instant (no motion compensation)
    sc, xf = _wall_scene(36.0)
    sw = ls.sweep(sc, 0.0, 0.0, True, seed=3)
    m = sw['obj'] == 1
    col = np.rint(sw['t'] / ls.period * ls.J).astype(int)
    first, last = m & (col < 4), m & (col >= ls.J - 4)
    assert first.sum() > 50 and last.sum() > 50
    Pv = sw['xyz'].astype(float) @ ls.R_vs.T + ls.pos
    tf, tl = sw['t'][first].astype(float), sw['t'][last].astype(float)
    xs, _, _ = sc.drive.pose(np.r_[tf, tl])
    own = Pv[:, 0][np.r_[np.nonzero(first)[0], np.nonzero(last)[0]]] + xs        # each point at its own pose
    assert np.abs(own - xf).max() < 1e-6
    x_end = float(sc.drive.pose(ls.period)[0])                                   # one pose for the sweep, as a reader
    skew = (Pv[first, 0] + x_end).mean() - (Pv[last, 0] + x_end).mean()
    expect = float(np.mean(sc.drive.travel(tl)) - np.mean(sc.drive.travel(tf)))
    assert expect > 0.7 and abs(skew - expect) < 0.01


def test_lidar_self_hits_flagged_and_dropped(engine):
    p = _rig()
    lid = _lidar(p)
    lid.z = 1.62                                       # 9 cm above the roof: the roof is past the minimum range
    ls = DS.LidarSim(p, engine, lid)
    sc, _ = _wall_scene(0.0)
    keep = ls.sweep(sc, 0.0, 0.0, False, seed=1)
    drop = ls.sweep(sc, 0.0, 0.0, True, seed=1)
    n_self = int(keep['self'].sum())
    assert n_self > 100 and np.all(keep['obj'][keep['self']] == DS.P_BODY)
    assert len(drop['xyz']) == len(keep['xyz']) - n_self and not drop['self'].any()
    r = np.linalg.norm(keep['xyz'][keep['self']], axis=1)
    assert r.min() >= ls.m.range_min_m and r.max() < 3.0


def test_imu_lever_arm_against_finite_differences():
    sc = DS.build_scene('urban', 30, 8, 1)
    dr = sc.drive
    imu = Sensor(kind='imu', model_id='imu_mems', tag='IMU9', x=1.5, y=0.4, z=0.8, yaw=30.0, pitch=5.0, roll=-3.0)
    R_vi = SN.rotation(imu)
    t = np.array([0.5, 1.4, 2.0, 2.6, 2.9, 3.4, 5.0, 7.5])
    f, w = DS.imu_truth(dr, imu, t)
    h = 1e-3
    for k, tk in enumerate(t):
        P = [DS.antenna_world(dr, imu, tk + d)[0] for d in (-h, 0.0, h)]
        a_w = (P[0] - 2 * P[1] + P[2]) / (h * h)
        _, _, psi = (float(v) for v in dr.pose(tk))
        f_ref = (DS._rz(psi) @ R_vi).T @ (a_w + np.array([0, 0, DS.G]))
        assert np.abs(f[k] - f_ref).max() < 2e-3, (tk, f[k], f_ref)
        _, _, pa = dr.pose(tk - h)
        _, _, pb = dr.pose(tk + h)
        w_ref = R_vi.T @ np.array([0, 0, float(DS._wrap(pb - pa)) / (2 * h)])
        assert np.abs(w[k] - w_ref).max() < 1e-6


def test_imu_at_rest_and_in_a_steady_turn():
    imu = Sensor(kind='imu', model_id='ins_dual', tag='IMU1', x=0.0, y=0.0, z=0.45)
    road = DS.Road(0.0, 0.0, 0.3, 0.0, 30.0, 1, math.pi, DS.URBAN)               # the bend from the start
    rest = DS.EgoDrive(road, 0.0, 2.0, -1.75)
    f, w = DS.imu_truth(rest, imu, np.linspace(0, 2, 21))
    assert np.allclose(f, [0, 0, DS.G], atol=1e-12) and np.allclose(w, 0, atol=1e-12)
    turn = DS.EgoDrive(road, 36.0, 4.0, -1.75)
    t = np.linspace(2.0, 3.5, 16)                                                # after the 1.6 s acceleration
    k = turn.kinematics(t)
    v, om = 10.0, 10.0 / (30.0 + 1.75)
    assert np.allclose(k['v'], v) and np.allclose(k['omega'], om, rtol=1e-9)
    f, w = DS.imu_truth(turn, imu, t)
    assert np.allclose(w, [0, 0, om], atol=1e-9)
    assert np.allclose(f, [0, v * om, DS.G], atol=1e-9)                          # centripetal v·ω, to the left


def test_imu_bag_at_rest(at_rest):
    RB = pytest.importorskip('vantage.core.rosbag')
    res = at_rest['res']
    r = RB.BagReader(res.bag_dir)
    acc, gyr = [], []
    for _, _, m in r.messages(['/imu1/imu']):
        acc.append([m['linear_acceleration'][k] for k in 'xyz'])
        gyr.append([m['angular_velocity'][k] for k in 'xyz'])
        assert m['header']['frame_id'] == 'IMU1_link' and m['orientation_covariance'][0] >= 0
    r.close()
    acc, gyr = np.array(acc), np.array(gyr)
    n = len(acc)
    assert n == int(0.6 * 100) + 1
    bias_a, bias_g = 5 * DS.IMU_NOISE['accel_bias'], 5 * DS.IMU_NOISE['gyro_bias']
    assert np.all(np.abs(acc.mean(0) - [0, 0, DS.G]) < bias_a + 4 * DS.IMU_NOISE['accel_sigma'] / math.sqrt(n))
    assert np.all(np.abs(gyr.mean(0)) < bias_g + 4 * DS.IMU_NOISE['gyro_sigma'] / math.sqrt(n))
    assert 0.5 < acc[:, 2].std() / DS.IMU_NOISE['accel_sigma'] < 1.6


def test_render_matches_the_exported_lens(engine):
    """A static box ahead of the front camera: its rendered pixels lie inside the box's corners projected with
    cv2.fisheye and the exported K and Kannala-Brandt coefficients (independent of the renderer's own code)."""
    p = _rig()
    cam = p.variant.by_tag('F')
    road = DS.Road(10.0, 10.0, 0.0, 600.0, 50.0, 1, 0.5, DS.URBAN)
    drive = DS.EgoDrive(road, 0.0, 1.0, -1.75)
    x0, y0, _ = (float(v) for v in drive.pose(0.0))
    box = DS.car_object(1, (2.0, 1.6, 1.5), (0.6, 0.2, 0.2), pose=(x0 + 9.0, y0 + 0.8, 0.3))
    sc = DS.Scene(road, drive, [box])
    cm = EX.camera_model(p, cam)
    w, h = 480, 384
    cs = DS.CameraSim(p, engine, cam, w, h)
    img, ids = cs.render(sc, 0.0, noise=0, want_ids=True)
    mask = ids == 1
    assert mask.sum() > 500
    L, W, H = box.size
    c, s = math.cos(0.3), math.sin(0.3)
    q = np.array([[sx * L / 2, sy * W / 2, z] for sx in (-1, 1) for sy in (-1, 1) for z in (0.0, H)])
    Pw = np.c_[x0 + 9.0 + q[:, 0] * c - q[:, 1] * s, y0 + 0.8 + q[:, 0] * s + q[:, 1] * c, q[:, 2]]
    Pv = Pw - [x0, y0, 0.0]                                                      # the ego at yaw 0
    Pc = (Pv - cam.pos()) @ cm.W2O.T
    uv, _ = cv2.fisheye.projectPoints(Pc[None].astype(np.float64), np.zeros(3), np.zeros(3), cm.K,
                                      np.array(cm.k, float))
    uv = uv[0] * [w / cm.intr.width, h / cm.intr.height] + [0.5 * w / cm.intr.width - 0.5,
                                                           0.5 * h / cm.intr.height - 0.5]
    hull = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(hull, np.round(cv2.convexHull(uv.astype(np.float32))[:, 0] * 4).astype(np.int32), 1, shift=2)
    hull = cv2.dilate(hull, np.ones((3, 3), np.uint8))
    assert hull[mask].mean() > 0.99
    assert mask.sum() > 0.8 * (hull > 0).sum() * 0.5                            # and fills a good part of it


def _straight(speed_kmh, objects, duration=2.0):
    road = DS.Road(10.0, 10.0, 0.0, 800.0, 50.0, 1, 0.5, DS.URBAN)
    drive = DS.EgoDrive(road, speed_kmh, duration, -1.75)
    x0, y0, _ = road.to_world(0.0, -1.75)
    return road, drive, float(x0), float(y0)


def test_rolling_shutter_bands_are_global_frames_at_their_times(engine):
    p = _rig()
    cam = p.variant.by_tag('ML')
    road, drive, x0, y0 = _straight(50.0, [])
    wall = DS.building_object(1, (x0 + 14.0, y0 + 7.0, 0.0), 12.0, 1.0, 5.0, (0.7, 0.7, 0.7))
    pole = DS.Obj(2, 'lamp', (0.2, 0.2, 6.0), [DS._cyl(0, 0, 0.1, 0.0, 6.0, DS.M_PLAIN, (0.5, 0.5, 0.5), 0.3)],
                  pose=(x0 + 6.0, y0 + 4.5, 0.0))
    car = DS.car_object(3, (4.4, 1.8, 1.5), (0.6, 0.1, 0.1))
    car.motion = DS.Motion(12.0, 1.75, -8.0)
    sc = DS.Scene(road, drive, [wall, pole, car])
    cs = DS.CameraSim(p, engine, cam, 240, 192)
    ro, ex = 0.03, 0.002
    rs = cs.render(sc, 1.0, ro, ex, True, noise=0).astype(int)
    tb, _ = cs.times(1.0, ro, ex, True)
    assert np.all(np.diff(tb) > 0) and abs(tb[0] - 1.0 - ro * 8 / 192 - ex / 2) < 1e-12
    for b in (0, len(tb) // 2, len(tb) - 1):
        gs = cs.render(sc, tb[b] - ro / 2 - ex / 2, ro, ex, False, noise=0).astype(int)
        rows = slice(b * DS.TILE, min((b + 1) * DS.TILE, 192))
        assert (np.abs(rs[rows] - gs[rows]) <= 1).mean() > 0.999, b
    gs = cs.render(sc, 1.0, ro, ex, False, noise=0).astype(int)
    assert (np.abs(rs - gs) > 10).mean() > 0.002          # the ego moved 0.4 m during the readout


def test_radar_ranges_doppler_and_occlusion(engine):
    p = _rig()
    rd1 = next(s for s in p.variant.sensors if s.tag == 'RD1')
    road, drive, x0, y0 = _straight(36.0, [])
    xr = x0 + 1.0 * 10.0                                   # roughly where the ego is at t = 1 s
    ahead = DS.car_object(1, (4.5, 1.8, 1.5), (0.5, 0.5, 0.5))
    hidden = DS.car_object(2, (4.5, 1.8, 1.5), (0.5, 0.5, 0.5))
    oncoming = DS.car_object(3, (4.5, 1.8, 1.5), (0.5, 0.5, 0.5))
    t = 1.0
    ex_, ey_, _ = (float(v) for v in drive.pose(t))
    radar_x = ex_ + rd1.x
    ahead.pose = (radar_x + 25.0 + 2.25, ey_ + rd1.y, 0.0)
    hidden.pose = (radar_x + 33.0 + 2.25, ey_ + rd1.y, 0.0)
    oncoming.motion = DS.Motion((radar_x - x0) + 40.0 + 10.0 * t, 1.75, -10.0)
    sc = DS.Scene(road, drive, [ahead, hidden, oncoming])
    det = DS.RadarSim(p, engine, rd1).scan(sc, t, seed=4, clutter=0)
    ids = det['obj'].tolist()
    assert 1 in ids and 3 in ids and 2 not in ids          # the car behind the first is hidden
    k = ids.index(1)
    r = math.sqrt(det['x'][k] ** 2 + det['y'][k] ** 2 + det['z'][k] ** 2)
    assert abs(r - 25.0) < 4 * DS.RADAR_NOISE['range_m']
    assert abs(det['v_r'][k] + 10.0) < 5 * DS.RADAR_NOISE['velocity_mps'] and abs(det['v_r_comp'][k]) < 1e-6
    assert det['dyn_prop'][k] == 1 and abs(det['rcs'][k] - 10.0) < 5 * DS.RADAR_NOISE['rcs_db']
    k = ids.index(3)
    u = math.cos(math.atan2(det['y'][k], det['x'][k]))
    assert abs(det['v_r'][k] + 20.0 * u) < 0.6 and abs(det['v_r_comp'][k] + 10.0 * u) < 0.2
    assert det['dyn_prop'][k] == 2


def test_visibility_in_the_open_and_behind_a_wall(engine):
    p = _rig()
    road, drive, x0, y0 = _straight(0.0, [])
    open_car = DS.car_object(1, (4.5, 1.8, 1.5), (0.5, 0.5, 0.5), pose=(x0 + 12.0, y0 + 3.5, 0.0))
    hid_car = DS.car_object(2, (4.5, 1.8, 1.5), (0.5, 0.5, 0.5), pose=(x0 + 1.0, y0 - 12.0, 0.0))
    wall = DS.building_object(3, (x0 + 1.0, y0 - 7.0, 0.0), 30.0, 0.5, 4.0, (0.7, 0.7, 0.7))
    sc = DS.Scene(road, drive, [open_car, hid_car, wall])
    cams = [DS.CameraSim(p, engine, c, 160, 128) for c in p.variant.cameras]
    boxes = {b['id']: b for b in sc.boxes_at(0.0)}
    assert DS.visibility(sc, 0.0, boxes[1], cams) > 0.8
    assert DS.visibility(sc, 0.0, boxes[2], cams) == 0.0
    assert DS.visibility_bin(0.39) == '1' and DS.visibility_bin(0.4) == '2' and DS.visibility_bin(0.95) == '4'


def test_cameras_only_rectified(engine, tmp_path):
    p = _rig()
    v = p.variant
    v.sensors = []
    v.cameras[5].visible = False
    st = DSET.DatasetSettings.from_dict({'duration_s': 0.5, 'image_scale': 0.08, 'undistort': True, 'rosbag': False,
                                         'keyframe_hz': 4})
    res = DSET.generate(p, engine, st, str(tmp_path))
    assert res.bag_dir == '' and not os.path.exists(tmp_path / 'rosbag')
    root = tmp_path / 'nuscenes' / DSET.VERSION
    t = {k: json.load(open(root / f'{k}.json')) for k in ('sensor', 'calibrated_sensor', 'sample', 'sample_data')}
    assert sorted(s['channel'] for s in t['sensor']) == sorted(f'CAM_{c.tag}' for c in v.cameras[:5])
    for row in t['calibrated_sensor']:
        vd = row['vantage_distortion']
        assert vd['model'].startswith('none') and vd['coefficients'] == []
        assert len(vd['rectified_from']['coefficients']) == 4
    samples = sorted(s['timestamp'] for s in t['sample'])
    assert np.allclose(np.diff(samples), round(30 / 4) / 30 * 1e6, atol=1)     # every 8th frame of the first camera
    for sd in t['sample_data']:
        img = cv2.imread(str(tmp_path / 'nuscenes' / sd['filename']))
        assert img.shape[:2] == (sd['height'], sd['width']) == (round(1536 * 0.08), round(1920 * 0.08))
    # a rectified image is black outside the lens's field only (the corners of a 95° pinhole view are inside it)
    assert img.mean() > 20


def test_nuscenes_channel_names(engine, tmp_path):
    p = _rig()
    st = DSET.DatasetSettings.from_dict({'duration_s': 0.5, 'image_scale': 0.05, 'rosbag': False,
                                         'channel_names': 'nuscenes'})
    plan = DSET._plan(p, p.variant, st)
    names = {c.sensor.tag: c.channel for c in plan['chans'] if c.channel}
    assert names == {'F': 'CAM_FRONT', 'R': 'CAM_BACK', 'ML': 'CAM_FRONT_LEFT', 'MR': 'CAM_FRONT_RIGHT',
                     'QL': 'CAM_BACK_LEFT', 'QR': 'CAM_BACK_RIGHT', 'L1': 'LIDAR_TOP', 'RD1': 'RADAR_FRONT',
                     'RD2': 'RADAR_FRONT_LEFT', 'RD3': 'RADAR_FRONT_RIGHT', 'RD4': 'RADAR_BACK_LEFT',
                     'RD5': 'RADAR_BACK_RIGHT'}
    assert not plan['notes']
    DSET.generate(p, engine, st, str(tmp_path))
    root = tmp_path / 'nuscenes' / DSET.VERSION
    sensor = json.load(open(root / 'sensor.json'))
    assert sorted(s_['channel'] for s_ in sensor) == sorted(names.values())
    # the calibration is still the export's (same tokens and poses), only the channel names differ
    ref = DSET.DatasetSettings.from_dict(dict(st.to_dict(), channel_names='tags'))
    plain = DSET._plan(p, p.variant, ref)
    ctx = EX._context(p, None, EX.ExportOptions(frame='rear_axle'))
    exp = {r['token']: r for r in json.loads(dict(EX._nuscenes(ctx))['nuscenes/calibrated_sensor.json'])}
    for row in json.load(open(root / 'calibrated_sensor.json')):
        assert row['translation'] == exp[row['token']]['translation']
    assert len(plain['chans']) == len(plan['chans'])
    for sd in json.load(open(root / 'sample_data.json')):
        assert sd['filename'].split('/')[1] in names.values()


# ----------------------------------------------------------------------------- the written dataset
def test_tables_resolve_and_chains_are_consistent(gen):
    t = gen['t']
    assert set(t) == {'attribute', 'calibrated_sensor', 'category', 'ego_pose', 'instance', 'log', 'map', 'sample',
                      'sample_annotation', 'sample_data', 'scene', 'sensor', 'visibility'}
    tok = {k: _by_token(v) for k, v in t.items()}
    ch = _channel_of(t)
    for sd in t['sample_data']:
        assert sd['sample_token'] in tok['sample'] and sd['ego_pose_token'] in tok['ego_pose']
        assert tok['ego_pose'][sd['ego_pose_token']]['timestamp'] == sd['timestamp']
        assert tok['calibrated_sensor'][sd['calibrated_sensor_token']]['sensor_token'] in tok['sensor']
        path = os.path.join(gen['nus'], sd['filename'])
        assert os.path.getsize(path) > 0
        assert sd['filename'].startswith('samples/' if sd['is_key_frame'] else 'sweeps/')
        assert f'/{ch(sd)}/' in sd['filename']
    for s in t['sample']:
        assert s['scene_token'] in tok['scene']
    sc = t['scene'][0]
    assert sc['log_token'] in tok['log'] and sc['nbr_samples'] == len(t['sample'])
    assert t['map'][0]['log_tokens'] == [t['log'][0]['token']]
    assert os.path.getsize(os.path.join(gen['nus'], t['map'][0]['filename'])) > 0
    for a in t['sample_annotation']:
        assert a['sample_token'] in tok['sample'] and a['instance_token'] in tok['instance']
        assert a['visibility_token'] in tok['visibility']
        assert all(x in tok['attribute'] for x in a['attribute_tokens'])
        assert len(a['size']) == 3 and len(a['rotation']) == 4
    for i in t['instance']:
        assert i['category_token'] in tok['category']
        chain, cur = [], i['first_annotation_token']
        while cur:
            chain.append(cur)
            cur = tok['sample_annotation'][cur]['next']
        assert chain[-1] == i['last_annotation_token'] and len(chain) == i['nbr_annotations']
        ts = [tok['sample'][tok['sample_annotation'][c]['sample_token']]['timestamp'] for c in chain]
        assert ts == sorted(ts)
    # one prev/next chain per channel, in time order, covering all of it
    by = {}
    for sd in t['sample_data']:
        by.setdefault(ch(sd), []).append(sd)
    assert len(by) == 12                       # 6 cameras, 1 lidar, 5 radars (GNSS and the IMU are not channels)
    for c, rows in by.items():
        heads = [r for r in rows if r['prev'] == '']
        assert len(heads) == 1, c
        n, cur, last = 0, heads[0], -1
        while True:
            n += 1
            assert cur['timestamp'] > last
            last = cur['timestamp']
            if not cur['next']:
                break
            nxt = tok['sample_data'][cur['next']]
            assert nxt['prev'] == cur['token']
            cur = nxt
        assert n == len(rows)
    # samples at keyframe_hz, each with one keyframe of every channel, the nearest one
    samples = sorted(t['sample'], key=lambda s: s['timestamp'])
    assert np.allclose(np.diff([s['timestamp'] for s in samples]), 1e6 / gen['st'].keyframe_hz, atol=1)
    for s in samples:
        keys = [sd for sd in t['sample_data'] if sd['sample_token'] == s['token'] and sd['is_key_frame']]
        assert sorted(ch(k) for k in keys) == sorted(by)
        for k in keys:
            near = min(by[ch(k)], key=lambda r: abs(r['timestamp'] - s['timestamp']))
            assert near['token'] == k['token']
    # the lidar keyframe is the sample itself
    lid = [sd for sd in t['sample_data'] if ch(sd) == 'LIDAR_L1' and sd['is_key_frame']]
    assert sorted(sd['timestamp'] for sd in lid) == [s['timestamp'] for s in samples]


def test_ego_poses_are_the_simulated_drive(gen):
    t = gen['t']
    sc = DSET.scene_for(gen['p'], gen['st'])
    for e in t['ego_pose'][::7]:
        ts = (e['timestamp'] - DSET.EPOCH_US) / 1e6
        x, y, psi = (float(v) for v in sc.drive.pose(ts))
        assert np.allclose(e['translation'], [x, y, 0.0], atol=1e-6)
        w, qx, qy, qz = e['rotation']
        assert abs(qx) < 1e-9 and abs(qy) < 1e-9 and abs(DS._wrap(2 * math.atan2(qz, w) - psi)) < 1e-8


def test_calibrated_sensor_equals_the_export(gen, tmp_path):
    p = gen['p']
    EX.export(['nuscenes_calib'], p, None, EX.ExportOptions(out_dir=str(tmp_path)))
    exp_cal = _by_token(json.load(open(tmp_path / 'nuscenes' / 'calibrated_sensor.json')))
    exp_sen = _by_token(json.load(open(tmp_path / 'nuscenes' / 'sensor.json')))
    t = gen['t']
    assert len(t['calibrated_sensor']) == 12
    models = {c.tag: EX.camera_model(p, c) for c in p.variant.cameras}
    sizes = {sd['calibrated_sensor_token']: (sd['width'], sd['height']) for sd in t['sample_data']}
    for row in t['calibrated_sensor']:
        e = exp_cal[row['token']]
        assert row['sensor_token'] == e['sensor_token']
        assert row['translation'] == e['translation'] and row['rotation'] == e['rotation']
        assert exp_sen[row['sensor_token']] == next(s for s in t['sensor'] if s['token'] == row['sensor_token'])
        if e['camera_intrinsic']:
            tag = e['vantage_distortion']['camera']
            cm = models[tag]
            w, h = sizes[row['token']]
            sx, sy = w / cm.intr.width, h / cm.intr.height
            assert (w, h) == (round(cm.intr.width * SCALE), round(cm.intr.height * SCALE))
            K = np.array(row['camera_intrinsic'])
            Ke = np.array(e['camera_intrinsic'])
            assert np.allclose(K, [[Ke[0, 0] * sx, 0, (Ke[0, 2] + 0.5) * sx - 0.5],
                                   [0, Ke[1, 1] * sy, (Ke[1, 2] + 0.5) * sy - 0.5], [0, 0, 1]], atol=2e-6)
            assert row['vantage_distortion']['coefficients'] == e['vantage_distortion']['coefficients']
        else:
            assert row['camera_intrinsic'] == []


def test_camera_times_follow_the_sync_plan(gen):
    p = gen['p']
    tp = TM.plan(p, p.variant)
    T = tp.period_ms / 1000.0
    mid = (tp.settings.readout_ms / 2 + tp.settings.exposure_ms / 2) / 1000.0
    t = gen['t']
    ch = _channel_of(t)
    tags = EX.safe_tags(p.variant)
    for c in p.variant.cameras:
        e = tp.cam(c.uid)
        ts = np.array(sorted(sd['timestamp'] for sd in t['sample_data'] if ch(sd) == f'CAM_{tags[c.uid].upper()}'))
        tt = (ts - DSET.EPOCH_US) / 1e6 - mid
        assert len(tt) >= 9
        assert np.allclose(np.diff(tt), T, atol=2e-6)                           # one frame per lidar rotation
        assert e['triggered'] and np.allclose(tt % T, e['trigger_ms'] / 1000.0, atol=2e-6)
    # at the module's rate: every pulse of the trigger line; without a lidar: free-running from t = 0
    st = DSET.DatasetSettings.from_dict({'duration_s': 1.0, 'camera_rate': 'module'})
    plan = DSET._plan(p, p.variant, st)
    for c in (c for c in plan['chans'] if c.kind == 'camera'):
        e = tp.cam(c.sensor.uid)
        fps = p.module_for(c.sensor).fps
        assert np.allclose(np.diff(c.times), 1 / fps) and np.allclose(c.times % (1 / fps), (e['trigger_ms'] / 1000.0)
                                                                      % (1 / fps), atol=1e-9)
        assert np.allclose(c.stamps - c.times, mid)
    q = copy.deepcopy(p)
    q.variant.sensors = [s for s in q.variant.sensors if s.kind != 'lidar']
    plan = DSET._plan(q, q.variant, DSET.DatasetSettings.from_dict({'duration_s': 1.0}))
    cam = next(c for c in plan['chans'] if c.kind == 'camera')
    assert cam.times[0] == 0.0 and np.allclose(np.diff(cam.times), 1 / 30.0)


def test_lidar_and_radar_files(gen):
    t = gen['t']
    ch = _channel_of(t)
    lid = [sd for sd in t['sample_data'] if ch(sd) == 'LIDAR_L1']
    m = gen['p'].sensor_model_for(_lidar(gen['p']))
    for sd in lid:
        raw = np.fromfile(os.path.join(gen['nus'], sd['filename']), np.float32)
        assert raw.size % 5 == 0 and raw.size > 5 * 10000 and sd['fileformat'] == 'pcd'
        pts = raw.reshape(-1, 5)
        r = np.linalg.norm(pts[:, :3], axis=1)
        assert r.min() >= m.range_min_m - 0.1 and pts[:, 3].min() >= 0 and pts[:, 3].max() <= 255
        assert np.all(pts[:, 4] == np.round(pts[:, 4])) and pts[:, 4].max() < m.n_channels()
    rad = [sd for sd in t['sample_data'] if ch(sd).startswith('RADAR_')]
    n_det = 0
    for sd in rad:
        data = open(os.path.join(gen['nus'], sd['filename']), 'rb').read()
        head, body = data.split(b'DATA binary\n', 1)
        lines = head.decode().splitlines()
        assert lines[0].startswith('#') and lines[1] == 'VERSION 0.7' and len(lines) == 10
        assert lines[2].split()[1:] == list(DSET.RADAR_DTYPE.names)
        n = int(lines[6].split()[1])
        assert DSET.RADAR_DTYPE.itemsize == 43 and len(body) == 43 * n + 1          # a byte after the last point
        a = np.frombuffer(body[:-1], DSET.RADAR_DTYPE)
        if np.isnan(a['x'][0]):
            continue
        n_det += n
        assert np.all(a['invalid_state'] == 0) and np.all(a['ambig_state'] == 3) and np.all(a['dyn_prop'] <= 2)
        assert np.all(np.hypot(a['x'], a['y']) < 300)
    assert n_det > 50
    anns = t['sample_annotation']
    assert sum(a['num_lidar_pts'] > 0 for a in anns) > 0.3 * len(anns)
    assert sum(a['num_radar_pts'] > 0 for a in anns) >= 5
    assert {a['visibility_token'] for a in anns} >= {'1', '4'}


def test_bag_topics_frames_and_consistency(gen):
    RB = pytest.importorskip('vantage.core.rosbag')
    res, t, p = gen['res'], gen['t'], gen['p']
    r = RB.BagReader(res.bag_dir)
    ch = _channel_of(t)
    n_sd = {}
    for sd in t['sample_data']:
        n_sd[ch(sd)] = n_sd.get(ch(sd), 0) + 1
    tags = {**EX.safe_tags(p.variant), **EX.sensor_tags(p.variant)}
    for c in p.variant.cameras:
        tg = DSET._ros_name(tags[c.uid])
        assert r.topics[f'/{tg}/image_raw/compressed'] == 'sensor_msgs/msg/CompressedImage'
        assert r.counts[f'/{tg}/image_raw/compressed'] == r.counts[f'/{tg}/camera_info'] == \
            n_sd[f'CAM_{tags[c.uid].upper()}']
    assert r.counts['/l1/points'] == n_sd['LIDAR_L1'] and r.topics['/l1/points'] == 'sensor_msgs/msg/PointCloud2'
    for k in range(1, 6):
        assert r.counts[f'/rd{k}/detections'] == n_sd[f'RADAR_RD{k}']
    assert r.counts['/imu1/imu'] == int(DUR * 100) + 1
    assert r.counts['/gn1/fix'] == r.counts['/gn2/fix'] == int(DUR * 10) + 1
    assert r.counts['/tf'] == r.counts['/odom'] == int(DUR * DSET.TF_HZ) + 1 and r.counts['/tf_static'] == 1
    assert r.counts['/ground_truth/markers'] == r.counts['/ground_truth/boxes'] == n_sd['LIDAR_L1']
    # frame ids are the URDF export's link names
    urdf = EX._urdf(EX._context(p, None, EX.ExportOptions(frame='rear_axle')))
    links = set(re.findall(r'<link name="([^"]+)"', urdf)) | {'odom'}
    frames = set()
    stamps = {}
    cams = {f'CAM_{tags[c.uid].upper()}': DSET._ros_name(tags[c.uid]) for c in p.variant.cameras}
    K_info = {}
    for topic, t_ns, m in r.messages():
        if topic in ('/tf_static', '/tf'):
            for x in m['transforms']:
                frames |= {x['header']['frame_id'], x['child_frame_id']}
        elif 'header' in m:
            frames.add(m['header']['frame_id'])
            assert RB.stamp_ns(m['header']['stamp']) == t_ns
        stamps.setdefault(topic, []).append(t_ns)
        if topic.endswith('/camera_info') and topic not in K_info:
            K_info[topic] = (m['width'], m['height'], np.array(m['k']).reshape(3, 3))
    r.close()
    assert frames <= links, frames - links
    cal = _by_token(t['calibrated_sensor'])
    for sd in t['sample_data']:
        c = ch(sd)
        if c in cams and sd['prev'] == '':
            w, h, K = K_info[f'/{cams[c]}/camera_info']
            assert (w, h) == (sd['width'], sd['height'])
            assert np.allclose(K, cal[sd['calibrated_sensor_token']]['camera_intrinsic'], atol=1e-5)
    # the same instants in both outputs
    lid_us = sorted(sd['timestamp'] for sd in t['sample_data'] if ch(sd) == 'LIDAR_L1')
    assert [s // 1000 for s in stamps['/l1/points']] == lid_us


def test_gnss_fixes_convert_back_to_the_path(gen):
    RB = pytest.importorskip('vantage.core.rosbag')
    p, st = gen['p'], gen['st']
    sc = DSET.scene_for(p, st)
    sig = p.sensor_model_for(next(s for s in p.variant.sensors if s.kind == 'imu')).pos_sigma_cm / 100.0
    r = RB.BagReader(gen['res'].bag_dir)
    errs = []
    for tag in ('GN1', 'GN2'):
        ant = next(s for s in p.variant.sensors if s.tag == tag)
        for _, t_ns, m in r.messages([f'/{tag.lower()}/fix']):
            ts = (t_ns - DSET.EPOCH_US * 1000) / 1e9
            E = DS.geodetic_to_enu(m['latitude'], m['longitude'], m['altitude'], st.origin_lat, st.origin_lon,
                                   st.origin_alt)
            errs.append(E - DS.antenna_world(sc.drive, ant, ts)[0])
            assert m['position_covariance_type'] == 2 and abs(m['position_covariance'][0] - sig ** 2) < 1e-12
    r.close()
    e = np.array(errs)
    assert len(e) == 2 * (int(DUR * 10) + 1)
    assert np.abs(e[:, :2]).max() < 5 * sig and np.abs(e[:, 2]).max() < 10 * sig
    assert 0.4 * sig < np.sqrt(np.mean(e[:, :2] ** 2)) < 1.6 * sig


def test_estimate_and_manifest(gen):
    est = DSET.estimate(gen['p'], gen['st'])
    t = gen['t']
    ch = _channel_of(t)
    for c, n in est['frames'].items():
        if c.startswith(('CAM_', 'LIDAR_', 'RADAR_')):
            assert n == sum(1 for sd in t['sample_data'] if ch(sd) == c)
    assert est['ok'] and est['samples'] == len(t['sample']) and est['bytes'] > 0 and est['seconds'] > 0
    doc = json.load(open(os.path.join(gen['out'], 'dataset.json')))
    assert doc['schema'] == DSET.SCHEMA and doc['counts']['samples'] == len(t['sample'])
    assert doc['written']['nuscenes'] == 'nuscenes'
    assert 'NuScenes(version=' in open(os.path.join(gen['out'], 'README.md')).read()
    s = DSET.summary(gen['res'])
    json.dumps(s)
    assert s['samples'] == len(t['sample']) and s['annotations'] == len(t['sample_annotation'])
    assert gen['progress'] and gen['progress'][-1] == 1.0 and gen['progress'] == sorted(gen['progress'])
    empty = copy.deepcopy(gen['p'])
    empty.variant.cameras, empty.variant.sensors = [], []
    assert not DSET.estimate(empty, gen['st'])['ok']
    with pytest.raises(ValueError):
        DSET.generate(empty, None, gen['st'], os.path.join(gen['out'], 'never'))


def test_preview(engine):
    p = _rig()
    st = DSET.DatasetSettings.from_dict({'duration_s': 4.0})
    d = DSET.preview(p, engine, st, 2.0, width=160)
    assert len(d['cameras']) == 6 and all(c['image'].shape[1] == 160 for c in d['cameras'])
    assert len(d['lidar']) > 10000 and d['boxes'] and d['road']
    # the lidar points in the ego frame: many on the ground (z = 0) around the car, none below it
    near = d['lidar'][np.hypot(d['lidar'][:, 0], d['lidar'][:, 1]) < 25]
    assert (np.abs(near[:, 2]) < 0.05).sum() > 500 and near[:, 2].min() > -0.1
    assert abs(d['speed_kmh'] - 30.0) < 1e-6


def test_cancel_and_foreign_folders(engine, tmp_path):
    p = _rig()
    st = DSET.DatasetSettings.from_dict({'duration_s': 1.0, 'image_scale': 0.05})
    n = [0]

    def prog(f, text=''):
        n[0] += 1
    res = DSET.generate(p, engine, st, str(tmp_path / 'c'), prog, cancel=lambda: n[0] > 3)
    assert res is None
    assert not os.path.exists(tmp_path / 'c' / 'nuscenes') and not os.listdir(tmp_path / 'c' / 'rosbag')
    foreign = tmp_path / 'f' / 'nuscenes'
    foreign.mkdir(parents=True)
    (foreign / 'v1.0-trainval').mkdir()
    with pytest.raises(ValueError):
        DSET.check_out_dir(str(tmp_path / 'f'), st, p)
    with pytest.raises(ValueError):
        DSET.generate(p, engine, st, str(tmp_path / 'f'))
    assert (foreign / 'v1.0-trainval').is_dir()                                   # left alone


# ----------------------------------------------------------------------------- the page
def _open(win, name, cls):
    if name not in win._page_widgets:
        pg = cls(win.ctx)
        win.pages.addWidget(pg)
        win._page_widgets[name] = pg
    win.set_page(name)
    return win._page_widgets[name]


def _until(app, cond, timeout=120):
    t = time.time()
    while not cond() and time.time() - t < timeout:
        pump(app, 0.05)
    return cond()


def test_page_preview_settings_generate_and_cancel(app, win, tmp_path):
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    from vantage.ui.workspaces.dataset_ws import DatasetPage
    st = win.store
    rig = next(v for v in st.project.variants if v.sensors)
    st.edit('variants', lambda p: setattr(p, 'active_uid', rig.uid), 'Data rig')
    settle(app, win)
    pg = _open(win, 'dataset', DatasetPage)
    pump(app, 0.3)
    # the estimate and a preview of the default drive
    assert pg.estimate['ok'] and pg.tiles['frames'].value != '—'
    assert _until(app, lambda: pg.preview_data is not None)
    assert len(pg.grid.cams) == 6 and len(pg.top.data['lidar']) > 1000
    # typed settings are undoable and saved with the project
    f, scale = pg.fields['duration_s']
    f.setFocus()
    f.lineEdit().selectAll()
    QTest.keyClicks(f.lineEdit(), '0.6')
    QTest.keyClick(f.lineEdit(), Qt.Key.Key_Return)
    pump(app, 0.2)
    assert st.project.analysis.dataset['duration_s'] == 0.6
    f, scale = pg.fields['image_scale']
    f.setFocus()
    f.lineEdit().selectAll()
    QTest.keyClicks(f.lineEdit(), '6')
    QTest.keyClick(f.lineEdit(), Qt.Key.Key_Return)
    pump(app, 0.2)
    assert abs(st.project.analysis.dataset['image_scale'] - 0.06) < 1e-9
    QTest.mouseClick(pg.toggles['rosbag'], Qt.MouseButton.LeftButton)
    pump(app, 0.2)
    assert st.project.analysis.dataset['rosbag'] is False
    st.undo.undo()
    pump(app, 0.2)
    assert st.project.analysis.dataset['rosbag'] is True and pg.toggles['rosbag'].isChecked()
    assert pg.slider.maximum() == 6
    # generate into a folder: progress, then the summary with the devkit snippet
    out = tmp_path / 'ds'
    assert pg.generate(str(out))
    assert pg.busy and not pg.b_generate.isEnabled()
    assert _until(app, lambda: not pg.busy, 300)
    assert pg.result is not None, pg.g_status.text()
    assert "NuScenes(version='v1.0-vantage'" in pg.out.toPlainText() and 'ros2 bag info' in pg.out.toPlainText()
    assert os.path.isdir(out / 'nuscenes' / 'v1.0-vantage') and os.path.exists(out / 'dataset.json')
    # a second run into the same folder replaces its own output; cancelling removes it again
    pg._set('longer drive', duration_s=3.0, image_scale=0.2)
    pump(app, 0.2)
    assert pg.generate(str(out))
    assert _until(app, lambda: pg.bar.value() > 0, 60)
    QTest.mouseClick(pg.b_cancel, Qt.MouseButton.LeftButton)
    assert _until(app, lambda: not pg.busy, 120)
    assert pg.result is None and 'Cancelled' in pg.g_status.text()
    assert not os.path.exists(out / 'nuscenes')
    # a folder holding someone else's nuScenes data is refused before anything starts
    (tmp_path / 'other' / 'nuscenes').mkdir(parents=True)
    (tmp_path / 'other' / 'nuscenes' / 'x.json').write_text('{}')
    assert not pg.generate(str(tmp_path / 'other')) and 'did not write' in pg.g_status.text()
    # the project replaced while the page is open
    from vantage.core import samples
    st.set_project(samples.tigor_sample_project())
    pump(app, 0.3)
    assert pg.result is None and pg.out.toPlainText() == ''
