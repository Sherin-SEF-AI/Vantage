"""GNSS antennas and the IMU: the sky view against an open roof and against walls of known analytic solid angle,
blockers attributed to the right body part, rig part, sensor or camera, near-field multipath and the ground plane, the
heading baseline and its σ, the lever arms in both frames (an IMU turned 90°), the pose error budget against a hand
calculation and a Monte Carlo of δθ × L, the cache, the Issues list, settings, the export, and the page (offscreen)."""
import json
import math

import numpy as np
import pytest
from scipy.integrate import quad

from vantage.core import gnss as G
from vantage.core import samples, sensors
from vantage.core.analysis import Engine
from vantage.core.mesh import Mesh
from vantage.core.project import Camera, Project, RigPart, Sensor, Variant

TOP = 1.5                      # roof of the test body
OZ = 0.05                      # the survey antenna's phase centre above its base


def _box(x0, x1, y0, y1, z0, z1):
    V = np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)], float)
    F = [[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4],
         [1, 5, 7], [1, 7, 3]]
    return Mesh(V, np.array(F))


def _car():
    """A box body 4 m × 1.8 m × 1.5 m with a flat roof."""
    p = Project()
    p.vehicle.mesh = _box(-0.5, 3.5, -0.9, 0.9, 0.0, TOP)
    return p


def _ant(tag='GN1', x=1.0, y=0.0, z=TOP + OZ, model='gnss_survey', **kw):
    return Sensor(kind='gnss', model_id=model, tag=tag, x=x, y=y, z=z, **kw)


def _imu(tag='IMU1', model='ins_dual', **pose):
    pose = {'x': 0.0, 'y': 0.0, 'z': 0.45, **pose}
    return Sensor(kind='imu', model_id=model, tag=tag, **pose)


def _wall(d, w, h, x0=1.0):
    """A thin wall whose near face is d in front of an antenna at x0 on the roof, w wide and rising h above it."""
    ztop = TOP + OZ + h
    return RigPart(name='Wall', kind='box', x=x0 + d + 0.005, y=0.0, z=(TOP + ztop) / 2, length=0.01, width=w,
                   height=ztop - TOP)


def _wall_share(d, w, h, mask):
    """Analytic share of the sky above the mask that the wall's front face hides: for each azimuth within ±atan(w/2d)
    the face's top edge stands at atan(h cos(az) / d)."""
    a = math.atan(w / 2 / d)
    f = lambda az: max(math.sin(math.atan(h * math.cos(az) / d)) - math.sin(math.radians(mask)), 0.0)  # noqa: E731
    return quad(f, -a, a, limit=200)[0] / (2 * math.pi * (1 - math.sin(math.radians(mask))))


def _st(**kw):
    return G.GnssSettings.from_dict(kw)


@pytest.fixture(autouse=True)
def _fresh_cache():
    G.clear_cache()
    yield


# ----------------------------------------------------------------------------- the sky view
def test_an_antenna_on_a_flat_roof_sees_the_whole_sky_above_its_mask():
    p = _car()
    p.variant.sensors.append(_ant())
    res = G.run(p, Engine())
    a = res.antennas[0]
    assert a.visible == pytest.approx(1.0) and a.visible_all == pytest.approx(1.0)
    assert a.expected_sats == pytest.approx(32.0) and a.mask_deg == 10.0 and not a.blockers and not a.sectors
    assert np.isnan(a.horizon).all() and not a.near and not a.own_frame
    assert a.ground['ok'] and a.ground['surface'] == 'Body'
    assert (a.who == -1).all() and a.el_edges[0] == 0.0 and a.el_edges[-1] == 90.0 and 10.0 in a.el_edges
    # the cells' solid angles add up to the hemisphere: the shares are true solid-angle shares
    A = len(a.az)
    w = math.radians(360.0 / A) * (np.sin(np.radians(a.el_edges[1:])) - np.sin(np.radians(a.el_edges[:-1])))
    assert w.sum() * A == pytest.approx(2 * math.pi)
    sky = next(f for f in res.findings if f['kind'] == 'sky')
    assert sky['severity'] == 'good' and '100.0 %' in sky['message']


@pytest.mark.parametrize('d,w,h', [(1.0, 1.2, 0.8), (0.6, 1.2, 0.8), (1.0, 0.6, 0.6), (0.5, 1.5, 1.0)])
def test_a_wall_hides_its_analytic_solid_angle(d, w, h):
    p = _car()
    p.variant.sensors.append(_ant())
    p.rig.append(_wall(d, w, h))
    e = Engine()
    want = _wall_share(d, w, h, 10.0)
    fine = G.run(p, e, _st(az_step_deg=0.5, el_step_deg=0.5)).antennas[0]
    assert fine.blocked == pytest.approx(want, rel=0.02)
    coarse = G.run(p, e, _st()).antennas[0]                  # the default 2° × 1° cells
    assert coarse.blocked == pytest.approx(want, rel=0.03)
    # the horizon straight ahead is the wall's top edge; the sector spans ±atan(w / 2d)
    assert fine.horizon[0] == pytest.approx(math.degrees(math.atan(h / d)), abs=0.6)
    assert fine.skyline[0] == fine.horizon[0] and np.isnan(fine.skyline[len(fine.az) // 2])   # the wall rises from it
    half = math.degrees(math.atan(w / 2 / d))
    s = fine.sectors[0]
    assert (s['az_from'] - 360.0) == pytest.approx(-half, abs=0.6) and s['az_to'] == pytest.approx(half, abs=0.6)
    assert s['blockers'] == ['Wall'] and s['share'] == pytest.approx(fine.blocked)
    b = fine.blockers[0]
    assert (b['name'], b['kind']) == ('Wall', 'rig') and b['share'] == pytest.approx(fine.blocked)
    assert b['nearest_m'] == pytest.approx(d, abs=0.01)
    assert fine.expected_sats == pytest.approx(32 * (1 - fine.blocked))


def test_blocked_rays_are_attributed_to_the_part_sensor_or_camera_they_meet():
    p = _car()
    v = p.variant
    a = _ant()
    v.sensors.append(a)
    p.rig.append(RigPart(name='Sensor mast', kind='cylinder', x=1.0, y=0.3, z=TOP + 0.3, height=0.6, radius=0.03))
    v.sensors.append(Sensor(kind='lidar', model_id='ouster_os1_128', tag='L1', x=0.4, y=0.0, z=1.9))
    v.cameras.append(Camera(tag='CAM', x=1.45, y=-0.3, z=1.70, yaw=180.0, pitch=0.0))
    res = G.run(p, Engine(), _st(az_step_deg=1.0, el_step_deg=0.5))
    sky = res.antennas[0]
    by = {b['name']: b for b in sky.blockers}
    assert by['Sensor mast']['kind'] == 'rig' and by['L1']['kind'] == 'lidar' and by['CAM']['kind'] == 'camera'

    def seen(P):
        d = np.asarray(P, float) - a.pos()
        return math.degrees(math.atan2(d[1], d[0])) % 360, math.degrees(math.atan2(d[2], math.hypot(d[0], d[1])))
    assert sky.at(90.0, 20.0)[0]['name'] == 'Sensor mast'
    assert sky.at(*seen([0.4, 0.0, 1.9]))[0]['name'] == 'L1'
    b, dist = sky.at(*seen([1.47, -0.3, 1.70]))
    assert b['name'] == 'CAM' and dist == pytest.approx(math.dist(a.pos(), [1.45, -0.3, 1.70]), abs=0.03)
    assert sky.at(0.0, 80.0)[0] is None                                     # the zenith is open
    # the mast 0.27 m away rises far above the antenna's horizon: multipath, named
    mp = next(f for f in res.findings if f['kind'] == 'multipath')
    assert mp['severity'] == 'warning' and 'Sensor mast' in mp['message']
    assert [n['name'] for n in sky.near] == ['Sensor mast'] and sky.near[0]['dist_m'] == pytest.approx(0.27, abs=0.01)
    # a hidden sensor blocks nothing
    v.sensors[1].visible = False
    again = G.run(p, Engine(), _st(az_step_deg=1.0, el_step_deg=0.5)).antennas[0]
    assert 'L1' not in {b['name'] for b in again.blockers}


def test_an_antenna_inside_or_under_something_has_no_sky():
    from vantage.core import rules
    p = _car()
    p.variant.sensors.append(_ant(z=1.0))                       # inside the body
    res = G.run(p, Engine())
    a = res.antennas[0]
    assert a.no_sky and a.visible == pytest.approx(0.0) and a.blockers[0]['name'] == 'Body'
    f = next(f for f in res.findings if f['kind'] == 'no_sky')
    assert f['severity'] == 'critical' and 'GN1 has no sky' in f['message']
    # the Issues list says it once: rules.sensor_issues finds it inside the body
    vg = Engine().vehicle_geom(p)
    assert not [i for i in G.issues(p, vg) if 'no sky' in i.message]
    assert [i.severity for i in rules.sensor_issues(p, vg)] == ['critical']
    p2 = _car()
    p2.variant.sensors.append(_ant(z=TOP + 0.2))
    p2.rig.append(RigPart(name='Enclosure', kind='box', x=1.0, y=0.0, z=TOP + 0.15, length=0.6, width=0.5, height=0.3))
    a2 = G.run(p2, Engine()).antennas[0]
    assert a2.no_sky and a2.blockers[0]['name'] == 'Enclosure'
    # under a roof box on the rack: not inside anything, yet no sky above the mask: the sky check says so
    p3 = _car()
    p3.variant.sensors.append(_ant())
    p3.rig.append(RigPart(name='Roof box', kind='box', x=1.0, y=0.0, z=TOP + OZ + 0.25, length=1.4, width=1.2,
                          height=0.3))
    res3 = G.run(p3, Engine())
    assert res3.antennas[0].no_sky and res3.antennas[0].blockers[0]['name'] == 'Roof box'
    # the box hangs over it: the horizon profile reaches the zenith, the skyline (what rises from the horizon) is open
    assert (res3.antennas[0].horizon == 90.0).all() and np.isnan(res3.antennas[0].skyline).all()
    crit = [i for i in G.issues(p3) if i.severity == 'critical']
    assert len(crit) == 1 and 'GN1 has no sky' in crit[0].message and crit[0].uids == [p3.variant.sensors[0].uid]
    assert not rules.sensor_issues(p3, Engine().vehicle_geom(p3))


def test_sky_severity_follows_the_blocked_share():
    got = []
    for d, w, h in ((1.0, 0.6, 0.6), (0.6, 1.2, 0.8), (0.5, 1.5, 1.5)):
        p = _car()
        p.variant.sensors.append(_ant())
        p.rig.append(_wall(d, w, h))
        a = G.run(p, Engine()).antennas[0]
        iss = [i for i in G.issues(p) if 'of the sky above' in i.message]
        got.append((a.blocked, iss[0].severity if iss else None))
    (b0, s0), (b1, s1), (b2, s2) = got
    assert b0 < 0.10 and s0 is None                    # under 10 %: fine
    assert 0.10 < b1 < 0.25 and s1 == 'warning'
    assert b2 > 0.25 and s2 == 'serious'


def test_tilted_antenna_takes_the_sky_about_its_own_axis():
    p = _car()
    p.variant.sensors.append(_ant(pitch=30.0))           # leaning forward: its own horizon dips into the roof behind
    res = G.run(p, Engine())
    a = res.antennas[0]
    assert a.own_frame and a.tilt_deg == pytest.approx(30.0)
    assert a.visible_all < 0.95 and a.blockers[0]['name'] == 'Body'
    assert any(f['kind'] == 'tilt' for f in res.findings)
    flat = _car()
    flat.variant.sensors.append(_ant(pitch=4.0))          # a few degrees: the vehicle's sky
    assert not G.run(flat, Engine()).antennas[0].own_frame


def test_ground_plane_on_the_roof_but_not_on_a_mast():
    p = _car()
    p.rig.append(RigPart(name='Sensor mast', kind='cylinder', x=2.0, y=0.0, z=TOP + 0.3, height=0.6, radius=0.03))
    p.rig.append(RigPart(name='Platform', kind='box', x=0.0, y=0.0, z=TOP + 0.05, length=0.6, width=0.6, height=0.1))
    p.variant.sensors += [_ant('GN1', x=1.0), _ant('GN2', x=2.0, z=TOP + 0.6 + OZ),
                          _ant('GN3', x=0.0, z=TOP + 0.1 + OZ)]
    res = G.run(p, Engine())
    g = {a.tag: a.ground for a in res.antennas}
    assert g['GN1']['ok'] and g['GN1']['surface'] == 'Body'
    assert not g['GN2']['ok'] and g['GN2']['surface'] == 'Sensor mast' and g['GN2']['missing'] == 8
    assert g['GN3']['ok'] and g['GN3']['surface'] == 'Platform'
    note = [f for f in res.findings if f['kind'] == 'ground_plane']
    assert len(note) == 1 and note[0]['severity'] == 'note' and note[0]['tags'] == ['GN2']
    assert not any('ground plane' in i.message for i in G.issues(p))              # a note, not an issue


# ----------------------------------------------------------------------------- heading, lever arms, budget
def test_heading_baseline_length_direction_and_sigma():
    p = _car()
    p.variant.sensors += [_imu(), _ant('GN1', x=0.5), _ant('GN2', x=1.7, y=0.3, z=TOP + OZ + 0.1)]
    res = G.run(p, Engine())
    b = res.baseline
    L = math.sqrt(1.2 ** 2 + 0.3 ** 2 + 0.1 ** 2)
    Lh = math.hypot(1.2, 0.3)
    assert (b['primary'], b['secondary'], b['ins']) == ('GN1', 'GN2', 'IMU1')
    assert b['length_m'] == pytest.approx(L) and b['horizontal_m'] == pytest.approx(Lh)
    assert b['azimuth_deg'] == pytest.approx(math.degrees(math.atan2(0.3, 1.2)))
    assert b['vertical_deg'] == pytest.approx(math.degrees(math.atan2(0.1, Lh)))
    assert b['heading_sigma_deg'] == pytest.approx(0.1 * 2.0 / Lh)                    # ins_dual: 0.1° at 2 m
    assert next(f for f in res.findings if f['kind'] == 'baseline')['severity'] == 'good'
    assert [r['role'] for r in res.lever_arms if r['kind'] == 'gnss'] == ['primary', 'secondary']
    # the chosen primary and secondary turn the baseline round
    r2 = G.run(p, Engine(), _st(primary='GN2', secondary='GN1'))
    assert r2.baseline['primary'] == 'GN2' and r2.baseline['azimuth_deg'] == pytest.approx(b['azimuth_deg'] - 180)
    # shorter baselines: a warning below 1 m, serious below 0.5 m
    for x2, sev in ((1.3, 'warning'), (0.9, 'serious')):
        p.variant.sensors[2].x, p.variant.sensors[2].y, p.variant.sensors[2].z = x2, 0.0, TOP + OZ
        f = next(f for f in G.run(p, Engine()).findings if f['kind'] == 'baseline')
        assert f['severity'] == sev and set(f['tags']) == {'GN1', 'GN2'}


def test_antenna_and_receiver_counts():
    def kinds(*sens):
        p = _car()
        p.variant.sensors += list(sens)
        return {i.message.split(':')[0]: i.severity for i in G.issues(p)}, G.run(p, Engine())
    iss, res = kinds(_ant('GN1'), _ant('GN2', x=2.2))                         # antennas, no receiver
    assert any(f['kind'] == 'no_receiver' and f['severity'] == 'warning' for f in res.findings)
    assert res.baseline is None and res.budget is None
    _, res = kinds(_imu(), _ant('GN1'))                                        # two inputs, one antenna
    f = next(f for f in res.findings if f['kind'] == 'no_heading')
    assert f['severity'] == 'warning' and 'needs motion' in f['message'] and res.baseline is None
    _, res = kinds(_imu())                                                     # an INS with no antenna
    assert any(f['kind'] == 'no_antennas' for f in res.findings)
    _, res = kinds(_imu(), _ant('GN1'), _ant('GN2', x=2.2), _ant('GN3', x=1.6, y=0.5))
    f = next(f for f in res.findings if f['kind'] == 'antenna_count')
    assert f['severity'] == 'note' and f['tags'] == ['GN3']
    _, res = kinds(_imu(model='imu_mems'), _ant('GN1'))                        # a plain IMU takes no antenna
    assert any(f['kind'] == 'no_receiver' for f in res.findings)
    assert any(f['kind'] == 'model_figures' for f in res.findings)             # and gives no accuracy figures
    p = _car()
    assert G.issues(p) == [] and G.run(p, Engine()).antennas == []             # nothing to study


def test_reference_imu_is_the_first_ins_or_the_chosen_one():
    p = _car()
    p.variant.sensors += [_imu('IMU1', model='imu_mems', x=1.0), _imu('IMU2', x=2.0), _ant()]
    res = G.run(p, Engine())
    assert res.imu['tag'] == 'IMU2'                                            # the one with antenna inputs
    assert G.run(p, Engine(), _st(imu='IMU1')).imu['tag'] == 'IMU1'
    assert G.run(p, Engine(), _st(imu='nope')).imu['tag'] == 'IMU2'
    assert any(f['kind'] == 'imu_extra' and 'IMU1' in f['message'] for f in res.findings)


def test_lever_arms_in_the_vehicle_and_a_turned_imu_frame():
    p = _car()
    p.variant.sensors += [_imu(x=0.5, y=0.2, z=0.4, yaw=90.0), _ant('GN1', x=1.0), _ant('GN2', x=2.5)]
    p.variant.cameras.append(Camera(tag='F', x=3.6, y=0.0, z=0.6))
    p.variant.sensors.append(Sensor(kind='lidar', model_id='ouster_os1_128', tag='L1', x=1.5, y=0.0, z=1.6))
    res = G.run(p, Engine())
    gn1 = res.lever('GN1')
    np.testing.assert_allclose(gn1['vehicle_mm'], [500.0, -200.0, 1150.0], atol=1e-9)
    # the IMU's x points to the vehicle's left, its y to the rear: the antenna 0.2 m right and 0.5 m ahead
    np.testing.assert_allclose(gn1['imu_mm'], [-200.0, -500.0, 1150.0], atol=1e-9)
    o = res.lever('Vehicle origin')
    np.testing.assert_allclose(o['vehicle_mm'], [-500.0, -200.0, -400.0], atol=1e-9)
    np.testing.assert_allclose(o['imu_mm'], [-200.0, 500.0, -400.0], atol=1e-9)
    assert [r['kind'] for r in res.lever_arms] == ['origin', 'gnss', 'gnss', 'camera', 'lidar']
    assert res.lever('F')['length_mm'] == pytest.approx(math.dist([0.5, 0.2, 0.4], [3.6, 0.0, 0.6]) * 1000)
    # the baseline straight ahead is 90° to the right of the turned IMU's x
    assert res.baseline['azimuth_deg'] == pytest.approx(0.0) and res.baseline['imu_azimuth_deg'] == pytest.approx(-90.0)
    assert any(f['kind'] == 'imu_axes' and 'yaw 90.00°' in f['message'] for f in res.findings)
    # the text for the INS: the antenna lever arms in FLU and FRD and the rotation
    txt = G.ins_text(res, p)
    row = next(l for l in txt.splitlines() if l.startswith('GN1'))
    vals = [float(x) for x in row.split()[2:]]
    np.testing.assert_allclose(vals, [-0.2, -0.5, 1.15, -0.2, 0.5, -1.15], atol=1e-4)
    assert 'yaw 90.0000°' in txt and 'yaw -90.0000°' in txt and 'primary' in row
    origin = next(l for l in txt.splitlines() if l.startswith('origin'))
    np.testing.assert_allclose([float(x) for x in origin.split()[1:]], [-0.2, 0.5, -0.4, -0.2, -0.5, 0.4], atol=1e-4)


def test_pose_error_budget_against_a_hand_calculation():
    p = _car()
    p.variant.sensors += [_imu(x=0.5, y=0.0, z=0.4), _ant('GN1', x=0.5), _ant('GN2', x=2.5)]
    p.variant.cameras.append(Camera(tag='C1', x=3.5, y=1.0, z=1.4))           # lever arm (3, 1, 1) m
    st = _st(pos_sigma_cm=2.0, vertical_ratio=2.0, attitude_sigma_deg=0.05, heading_sigma_deg=0.2,
             lever_arm_sigma_mm=5.0, label_range_m=50.0)
    res = G.run(p, Engine(), st)
    r = res.budget_row('C1')
    sr = sp = math.radians(0.05)
    sy = math.radians(0.2)
    Lx, Ly, Lz = 3.0, 1.0, 1.0
    rot = [1000 * math.sqrt((sp * Lz) ** 2 + (sy * Ly) ** 2), 1000 * math.sqrt((sy * Lx) ** 2 + (sr * Lz) ** 2),
           1000 * math.sqrt((sr * Ly) ** 2 + (sp * Lx) ** 2)]
    np.testing.assert_allclose(r['rotation_mm'], rot, rtol=1e-9)
    np.testing.assert_allclose(r['rotation_mm'], [3.598, 10.508, 2.760], atol=2e-3)       # by hand
    meas = 5.0 * math.sqrt(2)                       # the antenna's lever arm in the INS and the camera's own
    tot = [math.sqrt(20 ** 2 + rot[0] ** 2 + meas ** 2), math.sqrt(20 ** 2 + rot[1] ** 2 + meas ** 2),
           math.sqrt(40 ** 2 + rot[2] ** 2 + meas ** 2)]
    np.testing.assert_allclose(r['sigma_mm'], tot, rtol=1e-9)
    np.testing.assert_allclose(r['sigma_mm'], [21.516, 23.673, 40.714], atol=2e-3)
    assert r['horizontal_mm'] == pytest.approx(math.hypot(tot[0], tot[1]))
    assert r['far_mm'] == pytest.approx(math.hypot(math.hypot(tot[0], tot[1]), 50.0 * sy * 1000))
    assert r['roll_deg'] == 0.05 and r['heading_deg'] == 0.2
    # an independent check of δθ × L: sample the attitude errors and measure the spread
    rng = np.random.default_rng(3)
    dth = rng.normal(size=(400_000, 3)) * np.array([sr, sp, sy])
    mc = np.cross(dth, [Lx, Ly, Lz]).std(axis=0) * 1000
    np.testing.assert_allclose(r['rotation_mm'], mc, rtol=0.01)
    # the budget's own heading: from the baseline when not set; the INS model's position and attitude when not set
    r2 = G.run(p, Engine()).budget
    assert r2['inputs']['heading_source'] == 'baseline' and r2['inputs']['pos_sigma_cm'] == 2.0
    assert r2['inputs']['heading_sigma_deg'] == pytest.approx(0.1 * 2.0 / 2.0)
    assert {x['kind'] for x in r2['rows']} == {'origin', 'camera'}              # antennas are not budget rows


def test_budget_without_heading_or_accuracy_figures_says_so():
    p = _car()
    p.variant.sensors += [_imu(), _ant('GN1')]                                 # one antenna: no baseline
    res = G.run(p, Engine())
    inp = res.budget['inputs']
    assert inp['heading_source'] == 'none' and math.isnan(inp['heading_sigma_deg'])
    f = next(f for f in res.findings if f['kind'] == 'model_figures')
    assert 'heading σ' in f['message'] and f['severity'] == 'note'
    json.dumps(G.to_doc(res, p), allow_nan=False)                              # no NaN in the export
    json.dumps(G.summary(res), allow_nan=False)
    assert G.run(p, Engine(), _st(heading_sigma_deg=0.3)).budget['inputs']['heading_source'] == 'setting'


# ----------------------------------------------------------------------------- cache, settings, robustness
def test_the_sky_is_cached_until_something_it_depends_on_changes():
    p = _car()
    v = p.variant
    v.sensors += [_ant(), Sensor(kind='lidar', model_id='ouster_os1_128', tag='L1', x=0.5, y=0.0, z=1.9)]
    e = Engine()
    a1 = G.run(p, e).antennas[0]
    assert G.run(p, e).antennas[0] is a1
    import time
    t = time.perf_counter()
    G.issues(p, e.vehicle_geom(p))
    assert time.perf_counter() - t < 0.05                                       # cheap for the Issues list
    v.sensors[1].x = 0.9                                                         # another sensor moves
    a2 = G.run(p, e).antennas[0]
    assert a2 is not a1 and a2.blockers[0]['share'] > a1.blockers[0]['share']
    v.sensors[0].y = 0.2                                                         # the antenna moves
    a3 = G.run(p, e).antennas[0]
    assert a3 is not a2 and a3.pos[1] == 0.2
    p.rig.append(_wall(1.0, 1.2, 0.8))                                           # the rig changes
    a4 = G.run(p, e).antennas[0]
    assert a4 is not a3 and 'Wall' in {b['name'] for b in a4.blockers}
    p.rig[0].name = 'Board'                                                      # a rename: the names follow
    assert 'Board' in {b['name'] for b in G.run(p, e).antennas[0].blockers}
    a5 = G.run(p, e, _st(mask_override=True, mask_deg=20.0)).antennas[0]       # the mask changes
    assert a5.mask_deg == 20.0 and a5.visible != a4.visible


def test_settings_are_validated():
    s = G.GnssSettings.from_dict({'mask_deg': 'x', 'az_step_deg': 0, 'el_step_deg': float('nan'), 'n_sats': 1e9,
                                  'mask_override': 'yes', 'imu': 5, 'primary': ' GN2 ', 'near_m': -1,
                                  'lever_arm_sigma_mm': '7.5', 'nonsense': 1})
    assert s.mask_deg == 10.0 and s.az_step_deg == 0.5 and s.el_step_deg == 1.0 and s.n_sats == 200.0
    assert s.mask_override is True and s.imu == '' and s.primary == 'GN2' and s.near_m == 0.05
    assert s.lever_arm_sigma_mm == 7.5
    assert G.GnssSettings.from_dict(s.to_dict()) == s
    assert G.GnssSettings.from_dict(None) == G.GnssSettings() == G.GnssSettings.from_dict(['junk'])


def test_bad_poses_hidden_sensors_cancel_progress_and_another_variant():
    p = _car()
    v = p.variant
    bad = _ant('GN9')
    bad.x = float('nan')
    hidden = _ant('GN5', x=2.0)
    hidden.visible = False
    v.sensors += [_ant('GN1'), bad, hidden]
    res = G.run(p, Engine())
    assert [a.tag for a in res.antennas] == ['GN1']
    assert any(f['kind'] == 'pose' and f['severity'] == 'critical' and f['tags'] == ['GN9'] for f in res.findings)
    seen = []
    assert G.run(p, Engine(), progress=lambda f, t='': seen.append((f, t))) is not None
    assert seen[-1][0] == 1.0 and any('GN1' in t for _, t in seen)
    assert G.run(p, Engine(), cancel=lambda: True) is None
    v2 = Variant(name='Other')
    v2.sensors += [_ant('GN1', x=0.3), _ant('GN2', x=2.8), _imu()]
    p.variants.append(v2)
    r2 = G.run(p, Engine(), variant=v2)
    assert r2.variant == 'Other' and r2.baseline['length_m'] == pytest.approx(2.5) and p.variant is v


def test_the_sample_rig_with_default_antennas_and_an_ins():
    p = samples.tigor_sample_project()
    v = next(v for v in p.variants if v.sensors)
    p.active_uid = v.uid
    for kind, mid in (('gnss', 'gnss_survey'), ('gnss', 'gnss_survey'), ('imu', 'ins_dual')):
        m = p.sensor_models[mid]
        v.sensors.append(Sensor(kind=kind, model_id=mid, tag=sensors.next_tag(kind, v.tags()),
                                **sensors.default_pose(kind, m, p.vehicle.mesh, p.rig, v.sensors)))
    res = G.run(p, Engine())
    assert [a.tag for a in res.antennas] == ['GN1', 'GN2']
    assert all(a.visible > 0.9 and a.ground['ok'] for a in res.antennas)
    assert res.baseline['horizontal_m'] > 0.9 and math.isfinite(res.baseline['heading_sigma_deg'])
    assert not any(f['severity'] in ('critical', 'serious') for f in res.findings)
    assert {r['tag'] for r in res.budget['rows']} >= {'Vehicle origin', 'F', 'L1', 'RD1'}
    assert res.seconds < 2.0


# ----------------------------------------------------------------------------- output
def test_summary_doc_text_png_and_export(tmp_path):
    import cv2
    p = _car()
    p.name = 'gnss-test'
    p.variant.sensors += [_imu(), _ant('GN1', x=0.5), _ant('GN2', x=2.5)]
    p.rig.append(_wall(0.6, 1.2, 0.8, x0=0.5))
    res = G.run(p, Engine())
    s = G.summary(res)
    json.dumps(s)
    assert s['antennas'][0]['blockers'][0]['name'] == 'Wall' and s['baseline']['length_m'] == 2.0
    assert set(s['antenna_lever_arms_imu_mm']) == {'GN1', 'GN2'}
    doc = json.loads(json.dumps(G.to_doc(res, p), allow_nan=False))
    assert doc['schema'] == G.SCHEMA == 'vantage.gnss_imu/1' and doc['project'] == 'gnss-test'
    assert doc['frame'] == G.FRAME and 'mm' in doc['units'] and doc['assumptions'] == G.ASSUMPTIONS
    a = doc['antennas'][0]
    assert len(a['horizon_profile']['azimuth_deg']) == len(a['horizon_profile']['max_blocked_elevation_deg']) == 180
    assert len(a['horizon_profile']['skyline_deg']) == 180
    assert a['blockers'][0]['name'] == 'Wall' and a['blocked_sectors'][0]['blockers'] == ['Wall']
    assert a['sky_visible_pct'] == pytest.approx(res.antennas[0].visible * 100, abs=0.01)
    assert doc['baseline']['heading_sigma_deg'] == pytest.approx(0.1)
    lv = {r['target']: r for r in doc['lever_arms']}
    assert lv['GN2']['vehicle_frame_mm'] == [2500.0, 0.0, 1100.0] and lv['GN2']['imu_frame_mm'] == [2500.0, 0.0, 1100.0]
    assert doc['pose_error_budget']['rows'][0]['target'] == 'Vehicle origin'
    assert doc['imu']['rotation_imu_to_vehicle'] == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    names = [n for n, _ in G.files(res, p)]
    assert names == ['gnss_imu.json', 'ins_lever_arms.txt', 'gnss_sky.png']
    paths = G.export(res, p, str(tmp_path / 'out'))
    assert [path.split('/')[-1] for path in paths] == names
    img = cv2.imread(paths[2])
    assert img is not None and img.shape[1] == 2 * 440 and img.shape[0] > 440
    assert 'GN1' in (tmp_path / 'out' / 'ins_lever_arms.txt').read_text()
    # no antenna: the PNG says so; no IMU: the text says so
    q = _car()
    q.variant.sensors.append(_imu())
    fs = dict(G.files(G.run(q, Engine()), q))
    assert list(fs) == names and cv2.imdecode(np.frombuffer(fs['gnss_sky.png'], np.uint8), 1).shape[1] == 440
    q2 = _car()
    q2.variant.sensors.append(_ant())
    assert 'No IMU' in G.ins_text(G.run(q2, Engine()), q2)


# ----------------------------------------------------------------------------- the page
from test_gui import app, win, pump, settle, _wait     # noqa: E402,F401  (fixtures)


def _open(win, name, cls):
    if name not in win._page_widgets:
        pg = cls(win.ctx)
        win.pages.addWidget(pg)
        win._page_widgets[name] = pg
    win.set_page(name)
    return win._page_widgets[name]


def _type_into_field(app, f, text):
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    QTest.mouseClick(f.lineEdit(), Qt.MouseButton.LeftButton)
    QTest.keyClick(f.lineEdit(), Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
    QTest.keyClicks(f.lineEdit(), text)
    QTest.keyClick(f.lineEdit(), Qt.Key.Key_Return)
    pump(app, 0.1)


def test_page_picks_an_antenna_edits_the_mask_undoes_and_exports(app, win, tmp_path):
    from PyQt6.QtCore import QPoint, Qt
    from PyQt6.QtTest import QTest
    from vantage.ui.workspaces.gnss_ws import EMPTY, GnssPage
    store = win.store
    pg = _open(win, 'gnss', GnssPage)
    pump(app, 0.2)
    assert pg.stack.currentIndex() == 1 and pg.status.text() == EMPTY            # nothing to study yet

    def add(p):
        v = p.variant
        for kind, mid in (('gnss', 'gnss_survey'), ('gnss', 'gnss_survey'), ('imu', 'ins_dual')):
            m = p.sensor_models[mid]
            v.sensors.append(Sensor(kind=kind, model_id=mid, tag=sensors.next_tag(kind, v.tags()),
                                    **sensors.default_pose(kind, m, p.vehicle.mesh, p.rig, v.sensors)))
    store.edit('variant', add, 'Add GNSS antennas and an INS')
    _wait(app, lambda: pg.result is not None and not pg.busy)
    res = pg.result
    assert pg.stack.currentIndex() == 0
    assert [a.tag for a in res.antennas] == ['GN1', 'GN2'] and len(pg.plots) == 2 and pg.ants.rowCount() == 2
    assert pg.ants.item(0, 0).text() == 'GN1' and pg.levers.rowCount() == len(res.lever_arms)
    assert pg.budget.rowCount() == len(res.budget['rows']) and len(pg.findings) == len(res.findings)
    assert pg.tiles['heading'].value == f'{res.baseline["horizontal_m"]:.2f}'
    assert 'GN1 (primary) → GN2 (secondary)' in pg.base_text.text()
    # a click on GN2's sky plot picks it: highlighted here, selected in the views, its sectors listed
    gn2 = res.antennas[1]
    plot = pg.plots[1]
    QTest.mouseClick(plot, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                     QPoint(plot.width() // 2, plot.height() // 2))
    pump(app, 0.1)
    assert pg.sel_uid == gn2.uid and plot.selected and not pg.plots[0].selected
    assert store.sel_obj == ('sensor', gn2.uid) and 'GN2' in pg.sec_head.text() and pg.ants.currentRow() == 1
    # one mask for every antenna, typed: saved with the project, computed again, undoable
    _type_into_field(app, pg.fields['mask_deg'], '20')
    _wait(app, lambda: pg.result.antennas[0].mask_deg == 20.0)
    assert store.project.analysis.gnss['mask_override'] is True and store.project.analysis.gnss['mask_deg'] == 20.0
    assert pg.mask_on.isChecked() and pg.plots[0].sky.mask_deg == 20.0
    # the primary antenna chosen from its list
    cb = pg.combos['primary']
    cb.setCurrentIndex(cb.findData('GN2'))
    cb.activated.emit(cb.currentIndex())
    _wait(app, lambda: pg.result.baseline['primary'] == 'GN2')
    store.undo.undo()
    store.undo.undo()
    _wait(app, lambda: pg.result.antennas[0].mask_deg == 10.0 and pg.result.baseline['primary'] == 'GN1')
    assert store.project.analysis.gnss == {} and pg.fields['mask_deg'].value() == 10.0 and not pg.mask_on.isChecked()
    assert pg.sel_uid == gn2.uid                                                   # the pick survives the edits
    # export
    paths = pg.export(str(tmp_path))
    assert json.loads((tmp_path / 'gnss_imu.json').read_text())['schema'] == G.SCHEMA and len(paths) == 3
    # moving an antenna updates the page
    uid = res.antennas[0].uid
    store.edit('variant', lambda p: setattr(p.variant.sensor(uid), 'z', 1.0), 'Sink GN1 into the roof')
    _wait(app, lambda: pg.result.antennas[0].no_sky)
    assert pg.tiles['sky'].sev == 'critical'
    # leave the page: edits are no longer computed until it shows again
    win.set_page('viewports')
    pump(app, 0.2)
    shown = pg.result
    store.undo.undo()
    pump(app, 0.2)
    assert pg.result is shown and pg._dirty
    win.set_page('gnss')
    _wait(app, lambda: pg.result is not shown and not pg.result.antennas[0].no_sky)
    # undo the sensors away: the page says how to add them
    store.undo.undo()
    _wait(app, lambda: pg.stack.currentIndex() == 1)
    assert pg.result is None
    # another project while the page shows: its antenna, picked
    q = samples.tigor_sample_project()
    m = q.sensor_models['gnss_patch']
    q.variant.sensors.append(Sensor(kind='gnss', model_id='gnss_patch', tag='GN7',
                                    **sensors.default_pose('gnss', m, q.vehicle.mesh, q.rig, [])))
    store.set_project(q)
    _wait(app, lambda: pg.result is not None and [a.tag for a in pg.result.antennas] == ['GN7'])
    assert pg.stack.currentIndex() == 0 and pg.sel_uid == q.variant.sensors[-1].uid and len(pg.plots) == 1
    assert pg.tiles['heading'].value == '—' and any(f['kind'] == 'no_receiver' for f in pg.result.findings)
