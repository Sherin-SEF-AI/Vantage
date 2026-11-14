"""Coherent FMCW lidar (core.fmcw_lidar): the library, Doppler sign and size, the ego-motion fit, moving points, the
sun and cross-talk models, settings, a synthetic CARLA-style recording with a known truth, exports and the page."""
import json
import math
import os

import numpy as np
import pytest

from vantage.core import fmcw_lidar as FM
from vantage.core import recording as RBUD
from vantage.core import sensors as S
from vantage.core.project import Project, Sensor

from test_gui import app, win, pump, _wait  # noqa: F401 (fixtures)

REAL = os.environ.get('VANTAGE_FMCW_DRIVE', '')                 # a CARLA scenario drive with lidar and radar, if any


def _open(win, name, cls):
    if name not in win._page_widgets:
        pg = cls(win.ctx)
        win.pages.addWidget(pg)
        win._page_widgets[name] = pg
    win.set_page(name)
    return win._page_widgets[name]


def _project(model_id='aeva_aeries_ii', **pose):
    p = Project()
    d = dict(x=1.0, y=0.0, z=1.8)
    d.update(pose)
    p.variant.sensors.append(Sensor(kind='lidar', model_id=model_id, tag='L1', **d))
    return p


# =========================================================================== the library
def test_library_models_are_fmcw_with_sources():
    lib = FM.library()
    ids = {m.id for m in lib}
    assert ids == {'aeva_aeries_ii', 'aeva_atlas', 'voyant_carbon30', 'voyant_carbon60'}
    built = {m.id: m for m in S.builtin_sensor_models()}
    for m in lib:
        assert m.kind == 'lidar' and m.ranging == 'fmcw' and FM.is_fmcw(m) and m.builtin
        assert m.url.startswith('https://') and m.vendor and FM.READ in m.note and S.CHECKED in m.note
        assert built[m.id] == m                                        # through sensors' hook
        assert S.SensorModel.from_dict(json.loads(json.dumps(m.to_dict()))) == m
        assert set(m.params) <= set(FM.PARAM_KEYS), m.id
        assert m.pps() > 0 and m.h_fov <= 360 and m.data_bps() > 0
        assert RBUD.link_capacity_mbps(m.interface)                     # the recording budget can read the link
        assert m.id not in S.RETIRED and 'typical' not in m.name.lower()
    c30 = built['voyant_carbon30']
    assert c30.pps() == 977_000 and c30.h_fov == 120 and c30.n_channels() == 128
    sp = FM.spec_of(c30)
    assert sp.velocity_sigma == pytest.approx(0.028) and sp.velocity_max == 63.0
    assert sp.sources['velocity_sigma'] == 'datasheet' and sp.sources['wavelength_nm'] == 'assumed'
    a2 = FM.spec_of(built['aeva_aeries_ii'])                         # not stated: the documented defaults, said so
    assert a2.velocity_sigma == FM.DEFAULT_VEL_SIGMA and math.isinf(a2.velocity_max)
    assert a2.sources['velocity_sigma'] == 'assumed' and a2.sources['velocity_max'] == 'not stated'
    assert FM.spec_of(c30, 0.1).velocity_sigma == 0.1                 # a setting overrides
    assert 'velocity_precision_mps' in built['aeva_aeries_ii'].params['assumed']
    assert not FM.is_fmcw(built['ouster_os1_128'])
    # a project keeps the models and a file round trip keeps their ranging and params
    p = Project()
    assert FM.is_fmcw(p.sensor_models['aeva_atlas'])
    q = S.SensorModel.from_dict(dict(p.sensor_models['voyant_carbon60'].to_dict(), params='junk', ranging='mimo'))
    assert q.ranging == '' and q.params == {}


# =========================================================================== Doppler
def test_doppler_sign_and_size():
    sp = np.zeros(3)
    v_s = np.array([10.0, 0.0, 0.0])
    # a static point straight ahead closes at the car's speed; behind it recedes; abeam it reads 0
    vr, d = FM.radial_velocity([[20.0, 0, 0], [-20.0, 0, 0], [0, 15.0, 0]], sp, v_s)
    assert vr == pytest.approx([-10.0, 10.0, 0.0])
    assert np.allclose(np.linalg.norm(d, axis=1), 1.0)
    # a lead car at 15 m/s pulls away at 5 m/s; an oncoming one at 10 m/s closes at 20 m/s
    vr, _ = FM.radial_velocity([[20.0, 0, 0], [30.0, 0, 0]], sp, v_s, [[15.0, 0, 0], [-10.0, 0, 0]])
    assert vr == pytest.approx([5.0, -20.0])
    # off the axis: v_r = |v| cos(angle)
    a = math.radians(30)
    vr, _ = FM.radial_velocity([[10 * math.cos(a), 10 * math.sin(a), 0.0]], sp, v_s)
    assert vr[0] == pytest.approx(-10 * math.cos(a))
    # rigid rotation: a yawing body's points move w × r
    V = FM.rigid_velocity([[1.0, 0, 0], [0, 2.0, 0]], [0, 0, 0], [0, 0, 0], 0.5)
    assert V == pytest.approx(np.array([[0, 0.5, 0], [-1.0, 0, 0]]))
    # the sensor's own velocity on a turning car: ω × r_s added
    assert FM.sensor_velocity([10, 0, 0], 0.2, [2.0, 1.0, 1.5]) == pytest.approx([9.8, 0.4, 0.0])
    # the measurement: noise of sigma, NaN outside the stated range
    rng = np.random.default_rng(0)
    m = FM.measure(np.full(20000, 3.0), 0.05, 63.0, rng)
    assert np.mean(m) == pytest.approx(3.0, abs=0.002) and np.std(m) == pytest.approx(0.05, rel=0.03)
    assert np.isnan(FM.measure(np.array([70.0, -64.0, 62.0]), 0.0, 63.0, rng)[:2]).all()
    assert FM.measure(np.array([70.0]), 0.0, math.inf, rng)[0] == 70.0
    assert FM._check_box_doppler()[0] < 1e-9


def test_ego_motion_fit_recovers_the_velocity_and_rejects_movers():
    rng = np.random.default_rng(4)
    n = 3000
    az = rng.uniform(-math.pi, math.pi, n)
    el = rng.uniform(-0.35, 0.15, n)
    D = np.c_[np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)]
    v = np.array([20.0, -0.6, 0.1])
    vr = -D @ v + rng.normal(0, 0.03, n)
    mov = np.zeros(n, bool)
    mov[:600] = True
    vr[mov] += 8.0                                                     # 20 % movers, all far off the static model
    fit = FM.fit_sensor_velocity(D, vr, 0.03, seed=1)
    sd = np.sqrt(np.diag(fit['cov']))
    assert np.all(np.abs(fit['v'] - v) < 4 * sd) and np.all(sd < 0.01)
    assert not fit['inliers'][mov].any() and fit['inliers'][~mov].mean() > 0.99
    assert fit['rms'] == pytest.approx(0.03, rel=0.1)
    # plain least squares would be pulled off by the movers: RANSAC is what keeps them out
    ls = np.linalg.lstsq(-D, vr, rcond=None)[0]
    assert np.linalg.norm(ls - v) > 50 * np.linalg.norm(fit['v'] - v)
    # NaNs (outside the velocity range) are ignored; too few rays or rays in one line give None
    vr2 = vr.copy()
    vr2[::7] = np.nan
    assert np.linalg.norm(FM.fit_sensor_velocity(D, vr2, 0.03, seed=1)['v'] - v) < 0.01
    assert FM.fit_sensor_velocity(D[:2], vr[:2]) is None
    assert FM.fit_sensor_velocity(np.tile([1.0, 0, 0], (50, 1)), np.full(50, -3.0)) is None
    # forward speed and yaw rate from one lidar off the rear axle (no side slip at the axle)
    w = math.radians(12.0)
    r_s = np.array([2.2, 0.4, 1.9])
    vs = FM.sensor_velocity([15.0, 0, 0], w, r_s)
    eg = FM.ego_from_sensor_velocity(vs, r_s)
    assert eg['vx'] == pytest.approx(15.0) and eg['yaw_rate'] == pytest.approx(w)
    assert FM.ego_from_sensor_velocity(vs, [0.1, 0.0, 2.0])['yaw_rate'] is None
    err, _ = FM._check_ego_fit()
    assert err < 4.0


def test_scene_moving_points_and_across_the_ray():
    p = _project()
    st = FM.FmcwSettings(scene='urban', ego_speed_kmh=50.0)
    fr = FM.scene_frame(p, p.variant.sensors[0], p.sensor_models['aeva_aeries_ii'], st)
    seen = {o for o in np.unique(fr.inst) if o}
    assert {1, 2, 6} <= seen                                          # lead car, oncoming car, parked car
    cl, m = FM.evaluate_frame(fr, FM.spec_of(p.sensor_models['aeva_aeries_ii']), 0.3, 0)
    assert m['ego_error_mps'] < 0.01 and abs(m['speed_error_mps']) < 0.01
    assert m['recall'] > 0.95 and m['precision'] > 0.99
    objs = {o['name']: o for o in m['objects']}
    assert objs['parked car']['found'] == 0 and not objs['parked car']['moving']
    assert objs['parked car']['mean_v_r'] == pytest.approx(-13.2, abs=0.6)   # static: the car's own motion
    assert objs['oncoming car']['mean_own_v_r'] == pytest.approx(-50 / 3.6, abs=0.3)
    assert objs['lead car']['mean_own_v_r'] == pytest.approx(40 / 3.6, abs=0.3)
    assert len(cl.P) == len(cl.v_r) == m['points']
    # a car crossing straight across the sensor's line of sight: moving, but Doppler cannot see it
    fr2 = FM.Frame(np.array([[20.0, -0.5, 0.8], [20.0, 0.0, 0.8], [20.0, 0.5, 0.8]] + [[40.0, y, 1.0]
                                                                                     for y in range(-20, 21)]),
                   np.array([1, 1, 1] + [0] * 41), np.zeros(3), np.array([10.0, 0, 0]), 0.0,
                   {1: {'name': 'crossing', 'cls': 'car', 'c': np.array([20.0, 0.0, 0.8]), 'v': np.array([0, 8.0, 0]),
                        'w': 0.0}})
    fr2.P[:, 2] = 0.0
    _, m2 = FM.evaluate_frame(fr2, FM.FmcwSpec(1550, 0.01, math.inf, 0.0, {}), 0.3, 0)
    assert m2['moving_points'] == 3 and m2['across_ray_points'] >= 1 and m2['moving_found'] < 3


def test_highway_closing_speed_exceeds_a_velocity_range():
    p = _project('voyant_carbon30')
    p.sensor_models['voyant_carbon30'].range_m = 150.0                # its rated maximum: to see the oncoming car
    st = FM.FmcwSettings(scene='highway', ego_speed_kmh=120.0, max_rays=20000)
    res = FM.run(p, None, st)
    m = res.lidars[0]['metrics']
    assert m['outside_velocity_range'] > 0                             # oncoming at 130 + 120 km/h: 69 m/s > 63
    assert np.isnan(res.clouds['L1'].v_r).sum() == m['outside_velocity_range']
    p.analysis.fmcw = st.to_dict()
    iss = FM.issues(p)
    assert len(iss) == 1 and iss[0].severity == 'warning' and '63' in iss[0].message and iss[0].action == 'fmcw'
    p.analysis.fmcw = FM.FmcwSettings(scene='urban').to_dict()
    assert FM.issues(p) == []
    assert FM.issues(_project('ouster_os1_128')) == []


# =========================================================================== sun and cross-talk
def test_solar_tables_and_closed_forms():
    b = FM.band_irradiance(905, 905)
    assert b['etr'] == 0.918 and b['global'] == 0.81709 and b['direct'] == 0.76337
    b = FM.band_irradiance(1540, 1560)
    assert b['etr'] == pytest.approx(np.mean(FM.G173_ROWS[(FM.G173_ROWS[:, 0] >= 1540) &
                                                            (FM.G173_ROWS[:, 0] <= 1560), 1]))
    assert len(FM.G173_ROWS) == 82
    assert FM.air_mass(90.0) == pytest.approx(1.0, abs=1e-3)
    assert FM.air_mass(90 - 48.19) == pytest.approx(1.5, abs=0.01)   # G173's sun
    # the direct beam at G173's own air mass reproduces its direct column
    sk = FM.sky(('x', '', 90 - 48.19, 0.0, 'clear'), FM.band_irradiance(895, 915))
    assert sk['dn'] == pytest.approx(FM.band_irradiance(895, 915)['direct'], rel=1e-3)
    # Planck's occupation and the antenna-theorem photons per mode: a blackbody radiance gives back 1/(e^x − 1)
    lam, T = 1550e-9, 5772.0
    B = 2 * FM.H * FM.C ** 2 / lam ** 5 / math.expm1(FM.H * FM.C / (lam * FM.KB * T))
    assert FM.occupancy(B, lam) == pytest.approx(FM.planck_occupancy(lam, T), rel=1e-12)
    for c in FM.validation_checks():
        err, detail = c['fn']()
        assert err <= c['tolerance'], (c['id'], err, detail)
    tb = FM._check_solar_planck()[1]
    assert 6000 < tb['occupancy_1550']['brightness_temperature_K'] < 6700
    assert FM.illuminance_klx(next(c for c in FM.CONDITIONS if c[0] == 'noon_ahead')) == pytest.approx(95, abs=15)


def test_sun_tof_against_fmcw():
    """A forward lidar with beams at the low sun's elevation: the ToF range falls in sunlight (datasheet range in the
    dark), sunlit scenes raise a fixed threshold's false returns, the sun's disc blinds a few points; the FMCW receiver
    keeps its range on targets (n̄ ~ 1e-6) and loses a few per cent on the disc only."""
    m = S.SensorModel('t', 'T', 'lidar', 'X', 'solid', 64, -10.0, 10.0, h_fov=120.0, h_res=0.2, rate_hz=10.0,
                      range_m=150.0, range_min_m=0.5, points_per_s=0.0)
    p = Project()
    s = Sensor(kind='lidar', model_id='t', tag='L1', x=1.0, y=0.0, z=1.8)
    st = FM.FmcwSettings(tof_reference='dark', max_rays=20000)
    rows = {r['condition']: r for r in FM.sun_study(p, s, m, st)}
    assert set(rows) == {c[0] for c in FM.CONDITIONS}
    night, noon_b, low_a = rows['night'], rows['noon_behind'], rows['low_ahead']
    assert night['tof']['range_m'] == pytest.approx(150.0, rel=1e-6) and night['sun_points_per_frame'] == 0
    assert noon_b['tof']['range_ahead_m'] < 0.8 * 150.0                # front-lit targets ahead
    assert noon_b['tof']['false_per_s'] >= 0.99 * night['tof']['false_per_s']   # the threshold tracks the ambient
    assert low_a['tof']['false_per_s'] > 10 * night['tof']['false_per_s']      # the beams on the sun's disc fire
    assert low_a['tof']['false_per_s'] >= 0.9 * low_a['sun_points_per_frame'] * m.rate_hz
    assert noon_b['fmcw']['range_m'] > 0.999 * 150.0
    assert noon_b['fmcw']['photons_per_mode_target'] < 1e-4
    assert low_a['sun_points_per_frame'] > 0 and rows['low_behind']['sun_points_per_frame'] == 0
    assert low_a['tof']['blinded_range_m'] < 0.2 * 150.0
    assert 0.85 * 150 < low_a['fmcw']['blinded_range_m'] < 0.99 * 150
    assert 0.2 < low_a['fmcw']['photons_per_mode_sun'] < 0.4
    assert low_a['fmcw']['false_per_s'] < 2 * night['fmcw']['false_per_s']
    # a threshold that cannot track the ambient (one value per 360° of azimuth) fires on the sunlit beams
    wide = {r['condition']: r for r in FM.sun_study(p, s, m, FM.FmcwSettings(tof_reference='dark', max_rays=20000,
                                                                               tof_adapt_deg=360.0))}
    assert wide['noon_behind']['tof']['false_per_s'] >= noon_b['tof']['false_per_s']
    # APD receiver: an amplifier-noise floor makes it far less sensitive to sunlight than photon counting
    st2 = FM.FmcwSettings(tof_reference='dark', tof_receiver='apd', max_rays=20000)
    apd = {r['condition']: r for r in FM.sun_study(p, s, m, st2)}
    assert apd['noon_behind']['tof']['range_ahead_m'] > noon_b['tof']['range_ahead_m']
    # the datasheet's range stated in sunlight: the dark gives more (capped at the rated maximum)
    m.range_max_m = 250.0
    st3 = FM.FmcwSettings(tof_reference='sun', max_rays=20000)
    sun = {r['condition']: r for r in FM.sun_study(p, s, m, st3)}
    assert 150.0 < sun['night']['tof']['range_m'] <= 250.0
    # cross-talk: one test car in mutual view ahead, none behind a forward-only lidar
    ct = FM.crosstalk(s, m, FM.FmcwSettings(crosstalk_cars=1, crosstalk_distance_m=12.0))
    assert ct['in_mutual_view'] == 1 and ct['fmcw_points_per_s'] == 0.0
    assert ct['tof_points_per_frame'] == pytest.approx(0.015 * m.pps() / 10.0)
    ct2 = FM.crosstalk(s, m, FM.FmcwSettings(crosstalk_cars=4, crosstalk_distance_m=12.0))
    assert ct2['in_mutual_view'] == 2                                  # the two behind are outside its 120°
    assert FM.crosstalk(s, m, FM.FmcwSettings(crosstalk_cars=0))['tof_points_per_s'] == 0.0


# =========================================================================== settings
def test_settings_round_trip_and_junk():
    st = FM.FmcwSettings(source='recording', scene='highway', ego_speed_kmh=99.0, walls=False, recording='/x',
                         tof_receiver='apd', conditions=['night', 'low_ahead'], crosstalk_cars=3)
    assert FM.FmcwSettings.from_dict(json.loads(json.dumps(st.to_dict()))) == st
    j = FM.FmcwSettings.from_dict({'source': 'nope', 'scene': 7, 'ego_speed_kmh': 'fast', 'max_rays': 1e12,
                                   'moving_mps': -3, 'walls': 'off', 'conditions': 'night, junk', 'seed': 'x',
                                   'crosstalk_cars': 2.6, 'tof_receiver': 'pmt', 'p_false_ref': float('nan')})
    d = FM.FmcwSettings()
    assert j.source == d.source and j.scene == d.scene and j.ego_speed_kmh == d.ego_speed_kmh
    assert j.max_rays == 400000 and j.moving_mps == 0.05 and j.walls is False and j.conditions == ['night']
    assert j.crosstalk_cars == 3 and isinstance(j.crosstalk_cars, int) and j.tof_receiver == 'spad'
    assert j.p_false_ref == d.p_false_ref
    assert FM.FmcwSettings.from_dict(None) == d and FM.FmcwSettings.from_dict('junk') == d
    p = Project()
    assert FM.settings_of(p) == d
    p.analysis.fmcw = {'scene': 'highway'}
    assert FM.settings_of(p).scene == 'highway'
    k1 = FM.cache_key(p)
    p.analysis.fmcw = {'scene': 'urban'}
    assert FM.cache_key(p) != k1


# =========================================================================== a synthetic recording with a known truth
V_EGO = 10.0
V_CAR = 5.0


def _write_drive(root, with_velocity=True, n=12):
    """A tiny drive in the bridge's layout: nuScenes tables, lidar .pcd.bin sweeps, a radar's .pcd detections and a
    ROS 2 bag with /odom and /ground_truth/boxes. The ego drives along the global x axis at V_EGO; a car 25 m ahead
    in the next lane drives at V_CAR; a wall stands beside the road."""
    from vantage.core import rosbag as RB
    ver = os.path.join(root, 'nuscenes', 'v1.0-test')
    os.makedirs(ver)
    sensors = [{'token': 'sl', 'channel': 'LIDAR_L1', 'modality': 'lidar'},
               {'token': 'sr', 'channel': 'RADAR_RD1', 'modality': 'radar'}]
    cs = [{'token': 'cl', 'sensor_token': 'sl', 'translation': [1.0, 0.0, 1.8], 'rotation': [1, 0, 0, 0]},
          {'token': 'cr', 'sensor_token': 'sr', 'translation': [3.0, 0.0, 0.5], 'rotation': [1, 0, 0, 0]}]
    rows, poses = [], []
    t0 = 1_700_000_000_000_000
    rng = np.random.default_rng(5)
    bag = RB.BagWriter(os.path.join(root, 'rosbag', 'drive'))
    bag.add_topic('/odom', 'nav_msgs/msg/Odometry')
    bag.add_topic('/ground_truth/boxes', 'std_msgs/msg/String')
    wall = np.c_[rng.uniform(-30, 60, 3000), np.full(3000, 8.0), rng.uniform(0.2, 4, 3000)]
    ground = np.c_[rng.uniform(-20, 40, 3000), rng.uniform(-7, 7, 3000), np.zeros(3000)]
    for i in range(n):
        t = i * 0.1
        ts = t0 + int(round(t * 1e6))
        ego = np.array([V_EGO * t, 0.0, 0.0])
        car = np.array([25.0 + V_CAR * t, 3.5, 0.75])
        # car surface points (its rear and left faces), world frame, then vehicle frame = world − ego (no rotation)
        cy, cz = np.meshgrid(np.linspace(-0.85, 0.85, 12), np.linspace(-0.55, 0.7, 10))   # above 0.15 m
        rear = np.c_[np.full(cy.size, car[0] - 2.25), car[1] + cy.ravel(), car[2] + cz.ravel()]
        P = np.r_[ground + [ego[0], 0, 0], wall, rear] - ego
        is_car = np.r_[np.zeros(len(ground) + len(wall), bool), np.ones(len(rear), bool)]
        for kind, sd, mount, pts in (('lidar', 'l', np.array([1.0, 0, 1.8]), P),
                                     ('radar', 'r', np.array([3.0, 0, 0.5]), P[::7])):
            tok = f'{sd}{i:03d}'
            poses.append({'token': tok, 'timestamp': ts, 'rotation': [1, 0, 0, 0], 'translation': ego.tolist()})
            q = pts - mount
            if kind == 'lidar':
                fn = f'sweeps/LIDAR_L1/{tok}.pcd.bin'
                os.makedirs(os.path.join(root, 'nuscenes', 'sweeps', 'LIDAR_L1'), exist_ok=True)
                np.c_[q, np.zeros(len(q)), np.zeros(len(q))].astype(np.float32).tofile(
                    os.path.join(root, 'nuscenes', fn))
            else:
                fn = f'sweeps/RADAR_RD1/{tok}.pcd'
                os.makedirs(os.path.join(root, 'nuscenes', 'sweeps', 'RADAR_RD1'), exist_ok=True)
                V = np.zeros((len(pts), 3))
                V[is_car[::7]] = [V_CAR, 0, 0]
                vr, u = FM.radial_velocity(pts, mount, [V_EGO, 0, 0], V)
                a = np.zeros(len(q), np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('vx', '<f4'),
                                               ('vy', '<f4')]))
                a['x'], a['y'], a['z'] = q[:, 0], q[:, 1], q[:, 2]
                a['vx'], a['vy'] = vr * u[:, 0], vr * u[:, 1]
                hdr = ('VERSION 0.7\nFIELDS x y z vx vy\nSIZE 4 4 4 4 4\nTYPE F F F F F\nCOUNT 1 1 1 1 1\n'
                       f'WIDTH {len(a)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {len(a)}\nDATA binary\n')
                with open(os.path.join(root, 'nuscenes', fn), 'wb') as fh:
                    fh.write(hdr.encode() + a.tobytes())
            rows.append({'token': tok, 'ego_pose_token': tok, 'calibrated_sensor_token': 'c' + sd, 'timestamp': ts,
                         'filename': fn, 'is_key_frame': i % 5 == 0})
        t_ns = ts * 1000
        bag.write('/odom', t_ns, {'header': RB.header(t_ns, 'odom'), 'child_frame_id': 'base_link',
                                  'pose': {'pose': {'position': {'x': ego[0], 'y': 0.0, 'z': 0.0},
                                                    'orientation': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0}}},
                                  'twist': {'twist': {'linear': {'x': V_EGO, 'y': 0.0, 'z': 0.0},
                                                      'angular': {'x': 0.0, 'y': 0.0, 'z': 0.0}}}})
        box = {'id': 1, 'category': 'vehicle.car', 'name': 'vehicle.test', 'centre': car.tolist(),
               'size_lwh': [4.5, 1.7, 1.5], 'yaw': 0.0}
        if with_velocity:
            box['velocity'] = [V_CAR, 0.0, 0.0]
        bag.write('/ground_truth/boxes', t_ns, {'data': json.dumps({'stamp_ns': t_ns, 'boxes': [box]})})
    bag.close()
    for name, doc in (('sensor', sensors), ('calibrated_sensor', cs), ('ego_pose', poses), ('sample_data', rows)):
        with open(os.path.join(ver, f'{name}.json'), 'w') as fh:
            json.dump(doc, fh)
    return root


@pytest.fixture(scope='module')
def drive(tmp_path_factory):
    return _write_drive(str(tmp_path_factory.mktemp('drive')))


def test_recording_with_carla_velocities(drive):
    assert FM.recording_lidars(drive) == ['L1']
    rec = FM.Recording(drive)
    d = rec.describe()
    assert d['sweeps'] == 12 and d['velocities'].startswith('CARLA') and d['ego'] == 'bag /odom twist'
    assert d['labels'] == 'points in boxes'
    fr = rec.frame(5)
    assert fr.v_ego == pytest.approx([V_EGO, 0, 0]) and fr.t == pytest.approx(0.5)
    assert (fr.inst == 1).sum() == 120 and fr.objects[1]['v'] == pytest.approx([V_CAR, 0, 0])
    p = _project('voyant_carbon30', x=1.0, z=1.8)
    st = FM.FmcwSettings(source='recording', recording=drive, max_sweeps=6)
    res = FM.run(p, None, st)
    r = res.lidars[0]
    sm = r['summary']
    assert sm['sweeps'] == 6 and sm['ego_error_mean_mps'] < 0.01
    assert sm['recall'] == pytest.approx(1.0) and sm['precision'] == pytest.approx(1.0)
    car = next(o for o in r['metrics']['objects'] if o['id'] == 1)
    assert car['mean_own_v_r'] == pytest.approx(V_CAR, abs=0.1)        # the car's 5 m/s along the ray
    assert car['mean_v_r'] == pytest.approx(V_CAR - V_EGO, abs=0.2)
    assert len(res.sweeps['L1']) == 6
    rd = res.checks['radar_doppler']
    assert rd['n'] > 0 and rd['median_abs_mps'] < 1e-3 and rd['sign_agreement'] == 1.0 and rd['n_moving'] > 0
    assert res.checks['differenced_velocity']['median_mps'] < 1e-3   # straight-line boxes: differencing is exact
    # the ego from the poses when the bag has no /odom: the same speed
    rec.odom = []
    v, w = rec.ego_motion(5)
    assert v == pytest.approx([V_EGO, 0, 0], abs=1e-6) and w == pytest.approx(0.0)
    with pytest.raises(ValueError):
        FM.run(p, None, FM.FmcwSettings(source='recording', recording=''))


def test_recording_without_velocities_differences_the_boxes(tmp_path):
    root = _write_drive(str(tmp_path / 'nov'), with_velocity=False, n=8)
    rec = FM.Recording(root)
    assert rec.velocity_source == 'boxes differenced'
    o = rec.objects_at(4)[1]
    assert o['v'] == pytest.approx([V_CAR, 0, 0], abs=1e-6)
    v, w = FM.track_velocity([0, 0.1, 0.2, 0.3], [[0, 0, 0], [1, 0.5, 0], [2, 1, 0], [3, 1.5, 0]], [0, 0.1, 0.2, 0.3],
                             0.15)
    assert v == pytest.approx([10, 5, 0]) and w == pytest.approx(1.0)
    assert FM.track_velocity([0.0], [[0, 0, 0]])[0] == pytest.approx([0, 0, 0])


def test_exports_and_overlay(tmp_path):
    p = _project('aeva_aeries_ii')
    p.variant.sensors.append(Sensor(kind='lidar', model_id='ouster_os1_128', tag='L2', x=0.5, z=1.9))
    res = FM.run(p, None, FM.FmcwSettings(max_rays=8000))
    assert [r['tag'] for r in res.lidars] == ['L1', 'L2']
    l2 = res.lidar('L2')
    assert not l2['fmcw'] and l2['doppler_model'] == 'aeva_aeries_ii' and l2['points_per_s'] == 1_310_720
    names = [n for n, _ in FM.files(res, p)]
    assert names[:3] == ['fmcw.json', 'fmcw_lidars.csv', 'fmcw_sun.csv']
    assert {'fmcw_L1.pcd', 'fmcw_L1.ply', 'fmcw_L2.pcd', 'fmcw_L2.ply'} <= set(names)
    paths = FM.export(res, p, str(tmp_path))
    doc = json.load(open(tmp_path / 'fmcw.json'))
    assert doc['schema'] == FM.SCHEMA and doc['frame'].startswith('vehicle') and doc['sign'] == FM.SIGN
    assert len(doc['lidars']) == 2 and doc['lidars'][0]['sun'][0]['condition'] == 'night'
    assert len(open(tmp_path / 'fmcw_sun.csv').read().splitlines()) == 1 + 2 * len(FM.CONDITIONS)
    cl = res.clouds['L1']
    back = FM.read_ply(open(tmp_path / 'fmcw_L1.ply', 'rb').read())
    assert len(back) == len(cl.P) and np.allclose(back['velocity'], cl.v_r, equal_nan=True)
    assert np.array_equal(back['moving'].astype(bool), cl.moving)
    pcd = FM.read_pcd(str(tmp_path / 'fmcw_L1.pcd'))
    assert np.allclose(pcd['x'], cl.P[:, 0]) and np.allclose(pcd['velocity'], cl.v_r, equal_nan=True)
    assert all(os.path.isfile(x) for x in paths)
    # the ROS 2 bag: a PointCloud2 per lidar with a float32 'velocity' field
    from vantage.core import rosbag as RB
    bag = FM.write_bag(res, str(tmp_path))
    with RB.BagReader(bag) as b:
        msgs = list(b.messages())
    assert {m[0] for m in msgs} == {'/l1/points4d', '/l2/points4d'}
    arr = RB.pointcloud2_array(next(m for t, _, m in msgs if t == '/l1/points4d'))
    assert np.allclose(arr['velocity'], cl.v_r, equal_nan=True) and arr.dtype.itemsize == 20
    sm = FM.summary(res)
    assert sm['lidars'][1]['evaluated_as'] == 'Aeva Aeries II' and 'night' in sm['lidars'][0]['sun']
    json.dumps(sm)
    # the 3D view's lidar points: an FMCW lidar coloured by radial velocity, a ToF lidar left alone
    s1, s2 = p.variant.sensors
    pts = np.array([[20.0, 0, 0], [-20.0, 0, 0], [1.0, 15.0, 1.8]])
    rgba = FM.sweep_velocity_rgba(p, s1, pts)
    assert rgba.shape == (3, 4) and rgba[0, 2] > rgba[0, 0] and rgba[1, 0] > rgba[1, 2]    # closing blue, red away
    assert FM.sweep_velocity_rgba(p, s2, pts) is None
    assert np.allclose(FM.velocity_rgba(np.array([np.nan]))[0, :3], 0.25)


def test_empty_variant_and_cancel():
    p = Project()
    res = FM.run(p, None, FM.FmcwSettings())
    assert res.lidars == [] and res.clouds == {}
    assert FM.run(_project(), None, FM.FmcwSettings(), cancel=lambda: True) is None
    p = _project()
    p.variant.sensors[0].visible = False
    assert FM.run(p, None, FM.FmcwSettings()).lidars == []


# =========================================================================== the page
def test_page(app, win, tmp_path, drive):  # noqa: F811
    from vantage.ui.workspaces.fmcw_ws import FmcwPage, cloud_colours
    store = win.ctx.store
    win.switch_variant(next(v for v in store.project.variants if v.sensors).uid)
    pg = _open(win, 'fmcw', FmcwPage)
    pg.activate()
    _wait(app, lambda: pg.result is not None and not pg.busy, timeout=120)
    n = len(pg.result.lidars)
    assert n >= 1 and pg.t_lid.rowCount() == n and pg.t_sun.rowCount() == len(FM.CONDITIONS)
    assert 'ego velocity within' in pg.head.text() and pg.c_lidar.count() == n
    assert pg.shown_cloud() is not None and pg.view3d is not None
    draws = pg.scene3d.points(pg.view3d)
    assert len(draws) == 1 and len(draws[0].points) == len(pg.shown_cloud().P)
    for k, _ in (('radial', 0), ('own', 0), ('moving', 0)):
        pg.c_mode.set_value(k)
        pg._redraw()
        assert cloud_colours(pg.shown_cloud(), k).shape == (len(pg.shown_cloud().P), 4)
        pump(app, 0.05)
    # an edit is one undo step and re-runs the study
    assert pg.set_scene('highway')
    _wait(app, lambda: pg.result is not None and not pg.busy and pg.result.settings.scene == 'highway', timeout=120)
    assert store.project.analysis.fmcw['ego_speed_kmh'] == 120.0
    assert pg.fields['ego_speed_kmh'].value() == pytest.approx(120.0)
    store.undo.undo()
    pump(app, 0.2)
    assert FM.settings_of(store.project).scene == 'urban'
    _wait(app, lambda: pg.result is not None and not pg.busy and pg.result.settings.scene == 'urban', timeout=120)
    assert not pg.set_option('ego_speed_kmh', FM.settings_of(store.project).ego_speed_kmh)    # no change, no step
    # a recording
    assert not pg.choose_recording(str(tmp_path))                     # not a drive
    assert pg.choose_recording(drive)
    _wait(app, lambda: pg.result is not None and not pg.busy and pg.result.settings.source == 'recording',
          timeout=120)
    assert 'recording' in pg.head.text() and 'radar Doppler' in pg.checks.text()
    assert pg.rec_tag.count() == 1 and pg.rec_tag.isEnabled() and not pg.fields['ego_speed_kmh'].isEnabled()
    paths = pg.export(str(tmp_path / 'out'))
    assert any(x.endswith('fmcw.json') for x in paths) and any(x.endswith('fmcw_4d_bag') for x in paths)
    store.undo.undo()
    pump(app, 0.2)
    assert FM.settings_of(store.project).source == 'scene'
    _wait(app, lambda: pg.result is not None and not pg.busy and pg.result.settings.source == 'scene', timeout=120)


# =========================================================================== a real CARLA drive (when present)
@pytest.mark.skipif(not os.path.isdir(REAL), reason='needs the CARLA recordings of round 9')
def test_real_pedestrian_crossing():
    p = _project('voyant_carbon30', x=0.966, z=1.57)
    res = FM.run(p, None, FM.FmcwSettings(source='recording', recording=REAL, max_sweeps=12))
    sm = res.lidars[0]['summary']
    assert sm['ego_error_mean_mps'] < 0.01 and sm['precision'] > 0.99
    assert sm['across_ray_points'] > 0 and sm['recall'] < 0.95 and sm['recall_along_ray'] > 0.99
    rd = res.checks['radar_doppler']
    assert rd['n'] > 1000 and rd['median_abs_mps'] < 0.02 and rd['sign_agreement'] > 0.98
