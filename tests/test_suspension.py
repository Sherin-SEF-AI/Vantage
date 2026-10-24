"""Suspension and loads (core.suspension), the generalised load cases (core.loadcases) and the load cases dialog.

The statics are checked against independent hand calculations (springs in series, moment balance, the roll of a
lateral load with and without an anti-roll bar, bump stops, measured curves); the typed load cases against a copy of
the earlier implementation; the camera moves against an independent rigid transform (scipy)."""
import json
import math
import os
import time

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
import pytest
from PyQt6.QtCore import Qt
from scipy.spatial.transform import Rotation

from vantage.core import samples, suspension as SU, loadcases as LC
from vantage.core.analysis import Engine
from vantage.core.mathutil import rot_y

G = SU.G


# ----------------------------------------------------------------------------- helpers
def _square(k_w=25.0, k_t=200.0, arb=0.0, kerb=300.0, bump=False):
    """Four identical linear corners, no bump stops."""
    st = SU.SuspensionSettings()
    for c in SU.CORNERS:
        st.corners[c] = SU.Corner(spring_rate=k_w, motion_ratio=1.0, tyre_rate=k_t, kerb_load=kerb, bump_gap=70.0,
                                  bump_rate=0.0, bump_curve=[] if not bump else [[0, 0], [10, 800]])
    st.arb_front = st.arb_rear = arb
    return st


L, T = 2.5, 1.5
XY = {'FL': (L, T / 2), 'FR': (L, -T / 2), 'RL': (0.0, T / 2), 'RR': (0.0, -T / 2)}


def _series(a, b):
    return a * b / (a + b)


@pytest.fixture(scope='module')
def rig():
    """The sample's data rig variant (cameras, a lidar and radars), with GNSS and IMU added."""
    from vantage.core.project import Sensor
    p = samples.tigor_sample_project()
    v = next(v for v in p.variants if v.sensors)
    p.active_uid = v.uid
    v.sensors.append(Sensor(tag='GN1', name='GNSS', kind='gnss', model_id='gnss_survey', x=0.5, y=0.3, z=1.55))
    v.sensors.append(Sensor(tag='IMU1', name='IMU', kind='imu', model_id='ins_dual', x=0.3, y=0.0, z=0.4))
    return p


@pytest.fixture(scope='module')
def engine():
    return Engine()


# ----------------------------------------------------------------------------- units
def test_unit_conversions():
    q = SU.quantity
    assert q('25 N/mm', 'rate') == 25.0 and q('25 kN/m', 'rate') == 25.0 and q('25000 N/m', 'rate') == pytest.approx(25)
    assert q('140 lbf/in', 'rate') == pytest.approx(140 * 4.4482216152605 / 25.4)          # 24.52 N/mm
    assert q('2.5 kgf/mm', 'rate') == pytest.approx(2.5 * 9.80665)
    assert q(140, 'rate', 'lbf/in') == pytest.approx(24.5175, rel=1e-4)
    assert q('3.7 kN', 'mass') == pytest.approx(3700 / 9.80665)
    assert q('815 lb', 'mass') == pytest.approx(815 * 0.45359237)
    assert q('380 kg', 'force') == pytest.approx(380 * 9.80665)
    assert q('2.8 in', 'length') == pytest.approx(71.12) and q('0.07 m', 'length') == pytest.approx(70.0)
    assert q('3,5', 'length') == 3.5 and q('−12', 'length') == -12.0
    assert q('1490 mm', 'position') == pytest.approx(1.49) and q('60 %', 'percent') == 60.0
    for bad, kind, msg in (('25 N/m²', 'rate', 'unknown unit'), ('abc', 'rate', 'not a number'),
                           ('nan', 'mass', 'not a number'), ('12 kg', 'rate', 'spring rate'), (True, 'mass', 'number')):
        with pytest.raises(ValueError, match=msg):
            q(bad, kind, where='FL.spring_rate')
    with pytest.raises(ValueError, match='FL.spring_rate'):
        q('25 furlongs', 'rate', where='FL.spring_rate')


# ----------------------------------------------------------------------------- statics: known values
def test_symmetric_load_on_the_centre_drops_every_corner_equally():
    st = _square()
    m = 400.0
    s = SU.solve(st, XY, [('load', m, L / 2, 0.0, 0.8)])
    F = m * G / 4                                              # each corner carries a quarter
    drop = F / _series(25.0, 200.0)                            # spring and tyre in series
    for c in SU.CORNERS:
        cs = s.corners[c]
        assert cs.drop_mm == pytest.approx(drop, rel=1e-9)
        assert cs.travel_mm == pytest.approx(F / 25.0, rel=1e-9) and cs.tyre_mm == pytest.approx(F / 200.0, rel=1e-9)
        assert cs.added_n == pytest.approx(F, rel=1e-9) and cs.load_kg == pytest.approx(300 + m / 4, rel=1e-9)
    assert abs(s.pitch_deg) < 1e-9 and abs(s.roll_deg) < 1e-9 and s.heave_mm == pytest.approx(drop)
    assert s.converged and s.iterations == 1                   # linear: one Newton step
    assert s.cog == pytest.approx((L / 2, 0.0, 0.8)) and s.added_kg == m and s.total_kg == pytest.approx(1600)


def test_load_over_the_front_axle_drops_only_the_front():
    st = _square(k_w=22.0, k_t=180.0)
    st.corners['RL'].spring_rate = st.corners['RR'].spring_rate = 30.0     # the rear rates do not matter
    m = 150.0
    s = SU.solve(st, XY, [('front', m, L, 0.0, 0.5)])
    for c in ('RL', 'RR'):
        assert abs(s.corners[c].drop_mm) < 1e-9 and abs(s.corners[c].added_n) < 1e-6
    for c in ('FL', 'FR'):
        assert s.corners[c].drop_mm == pytest.approx(m * G / 2 / _series(22.0, 180.0), rel=1e-9)
    # nose down by the front drop over the wheelbase
    assert s.pitch_deg == pytest.approx(math.degrees(math.atan(s.corners['FL'].drop_mm / 1000 / L)), rel=1e-9)
    assert s.pitch_deg > 0 and abs(s.roll_deg) < 1e-9


@pytest.mark.parametrize('arb', [0.0, 15.0])
def test_load_on_the_left_rolls_left_by_the_hand_calculation(arb):
    k_w, k_t = 25.0, 200.0
    st = _square(k_w, k_t, arb=arb)
    m, y0 = 90.0, 0.45
    s = SU.solve(st, XY, [('left', m, L / 2, y0, 0.5)])
    W = m * G
    # hand calculation: the heave is a quarter of the weight per corner on the in-phase rate; the moment W·y0 goes to
    # the two axles equally (identical corners) as ±W·y0/(2T) per wheel, on the roll rate: the wheel rate plus the
    # bar, in series with the tyre
    heave = W / 4 / _series(k_w, k_t)
    dF = W * y0 / (2 * T)
    a = dF / _series(k_w + arb, k_t)
    for c, sign in (('FL', 1), ('RL', 1), ('FR', -1), ('RR', -1)):
        assert s.corners[c].drop_mm == pytest.approx(heave + sign * a, rel=1e-9)
        assert s.corners[c].added_n == pytest.approx(W / 4 + sign * dF, rel=1e-9)
    roll = -math.degrees(math.atan(2 * a / T / 1000))          # left side lower: negative (right-hand about +x)
    assert s.roll_deg == pytest.approx(roll, rel=1e-9) and s.roll_deg < 0
    assert abs(s.pitch_deg) < 1e-9


def test_anti_roll_bar_shifts_the_lateral_load_to_its_axle():
    st = _square()
    st.arb_front = 20.0
    s = SU.solve(st, XY, [('left', 80.0, L / 2, 0.5, 0.5)])
    dF_front = s.corners['FL'].added_n - s.corners['FR'].added_n
    dF_rear = s.corners['RL'].added_n - s.corners['RR'].added_n
    assert dF_front > dF_rear > 0                              # the stiffer axle in roll carries more of the moment
    assert (dF_front + dF_rear) * T / 2 == pytest.approx(80 * G * 0.5, rel=1e-9)      # the moment balances


def test_bump_stop_limits_travel_above_its_gap():
    k_w, k_b, gap = 25.0, 400.0, 20.0
    st = _square(k_w, k_t=1e4 / 2)                             # a stiff tyre keeps the numbers readable
    for c in SU.CORNERS:
        st.corners[c].bump_gap, st.corners[c].bump_rate, st.corners[c].bump_curve = gap, k_b, []
    m = 400.0
    F = m * G / 4
    free = SU.solve(_square(k_w, k_t=1e4 / 2), XY, [('l', m, L / 2, 0, 0.5)])
    s = SU.solve(st, XY, [('l', m, L / 2, 0, 0.5)])
    assert free.corners['FL'].travel_mm == pytest.approx(F / k_w) and F / k_w > gap
    want = gap + (F - k_w * gap) / (k_w + k_b)                 # the stop adds its rate beyond the gap
    for c in SU.CORNERS:
        cs = s.corners[c]
        assert cs.travel_mm == pytest.approx(want, rel=1e-9)
        assert cs.on_bump and cs.bump_mm == pytest.approx(want - gap, rel=1e-9)
    assert s.bump == list(SU.CORNERS) and s.converged
    # a progressive curve: the solved travel satisfies the curve's force exactly (interpolated)
    curve = [[0.0, 0.0], [5.0, 300.0], [10.0, 800.0], [15.0, 1600.0], [20.0, 3000.0]]
    for c in SU.CORNERS:
        st.corners[c].bump_rate, st.corners[c].bump_curve = 0.0, curve
    s = SU.solve(st, XY, [('l', m, L / 2, 0, 0.5)])
    u = s.corners['FL'].travel_mm - gap
    assert 0 < u < 20
    assert k_w * (u + gap) + np.interp(u, *zip(*curve)) == pytest.approx(F, rel=1e-9)
    # below the gap the stop does nothing
    light = SU.solve(st, XY, [('l', 50.0, L / 2, 0, 0.5)])
    assert not light.bump and light.corners['FL'].travel_mm == pytest.approx(50 * G / 4 / k_w, rel=1e-9)


def test_measured_curve_is_followed_instead_of_the_linear_rate():
    st = _square(k_w=5.0, k_t=200.0)                           # the spring rate must be ignored
    curve = [[2000.0, -40.0], [3000.0, -10.0], [3620.0, 0.0], [4500.0, 20.0], [6000.0, 40.0]]
    for c in SU.CORNERS:
        st.corners[c].curve = [list(p) for p in curve]
        st.corners[c].kerb_load = 3620.0 / G                   # kerb sits at travel 0 on the curve
    for target, travel in ((4500.0, 20.0), (5000.0, 20 + 500 / 1500 * 20), (3310.0, -5.0)):
        dF = target - 3620.0
        s = SU.solve(st, XY, [('l', 4 * dF / G, L / 2, 0, 0.5)])
        cs = s.corners['RR']
        assert cs.travel_mm == pytest.approx(travel, rel=1e-9, abs=1e-9)
        assert cs.drop_mm == pytest.approx(travel + dF / 200.0, rel=1e-9, abs=1e-9)
        assert not cs.beyond_curve and cs.load_kg == pytest.approx(target / G)
    s = SU.solve(st, XY, [('l', 4 * 3000.0 / G, L / 2, 0, 0.5)])     # beyond the last point: its slope continues
    assert s.corners['FL'].beyond_curve and s.corners['FL'].travel_mm == pytest.approx(40 + 620 / 1500 * 20, rel=1e-9)


def test_solve_balances_force_and_moments_and_the_drops_are_planar():
    st = SU.typical()
    st.corners['FL'].curve = [[2500.0, -30.0], [3600.0, 0.0], [5200.0, 30.0], [8000.0, 50.0]]
    st.corners['FL'].kerb_load = 3600.0 / G
    rng = np.random.default_rng(3)
    xy = {'FL': (2.6, 0.74), 'FR': (2.6, -0.74), 'RL': (0.0, 0.72), 'RR': (0.0, -0.72)}
    for _ in range(12):
        loads = [(f'm{i}', float(rng.uniform(5, 160)), float(rng.uniform(-0.8, 3.0)), float(rng.uniform(-0.6, 0.6)),
                  0.5) for i in range(int(rng.integers(1, 6)))]
        s = SU.solve(st, xy, loads)
        assert s.converged
        F = np.array([s.corners[c].added_n for c in SU.CORNERS])
        X = np.array([xy[c][0] for c in SU.CORNERS])
        Y = np.array([xy[c][1] for c in SU.CORNERS])
        W = np.array([m * G for _, m, *_ in loads])
        assert F.sum() == pytest.approx(W.sum(), rel=1e-7)
        assert (F * X).sum() == pytest.approx(sum(w * ld[2] for w, ld in zip(W, loads)), rel=1e-6, abs=1e-6)
        assert (F * Y).sum() == pytest.approx(sum(w * ld[3] for w, ld in zip(W, loads)), rel=1e-6, abs=1e-6)
        D = np.array([s.corners[c].drop_mm for c in SU.CORNERS])
        w, pitch, roll, rms = LC.plane_of_drops(dict(zip(SU.CORNERS, D)), xy)
        assert rms < 1e-9 and pitch == pytest.approx(s.pitch_deg, abs=1e-9) and roll == pytest.approx(s.roll_deg,
                                                                                                       abs=1e-9)
        # each corner's tyre force is its tyre rate times its deflection
        for c in SU.CORNERS:
            assert s.corners[c].added_n == pytest.approx(st.corners[c].tyre_rate * s.corners[c].tyre_mm, rel=1e-9,
                                                         abs=1e-9)


def test_no_load_and_kerb_distribution():
    s = SU.solve(_square(), XY, [])
    assert s.converged and s.iterations == 0 and all(s.corners[c].drop_mm == 0 for c in SU.CORNERS)
    st = SU.SuspensionSettings()
    st.set_kerb(1200, 60, 50)
    assert st.corners['FL'].kerb_load == pytest.approx(360) and st.corners['RR'].kerb_load == pytest.approx(240)
    assert st.kerb_mass == pytest.approx(1200) and st.front_pct == pytest.approx(60) and st.left_pct == 50
    st.set_kerb(front_pct=55)
    assert st.kerb_mass == pytest.approx(1200) and st.front_pct == pytest.approx(55)


def test_wheel_rate_from_spring_rate_and_motion_ratio():
    c = SU.Corner(spring_rate=40.0, motion_ratio=0.7)
    assert c.wheel_rate == pytest.approx(40 * 0.49)
    st = SU.SuspensionSettings.from_dict({'axles': {'rear': {'motion_ratio': 0.65, 'wheel_rate': '19 N/mm'}}})
    assert st.corners['RL'].wheel_rate == pytest.approx(19.0) and st.corners['RL'].spring_rate == pytest.approx(
        19 / 0.65 ** 2)


# ----------------------------------------------------------------------------- settings
def test_from_dict_is_tolerant_and_clamps():
    junk = {'source': 'nonsense', 'vehicle': {'track_front': 'wide', 'gross_mass': -5, 'driver_side': 'middle',
                                              'kerb_mass': float('nan')},
            'axles': {'front': {'spring_rate': 1e9, 'arb_rate': 'x'}, 'rear': 7},
            'corners': {'FL': {'tyre_rate': -3, 'curve': [[1, 2], [0, 5]], 'bump_curve': 'abc'}, 'ZZ': {}},
            'loads': {'items': [{'name': 'Box', 'mass': '12 kg', 'x': 'far'}, 5, {'id': 'box', 'mass': 1e9}],
                      'cases': [{'name': 'C', 'items': ['box', 'ghost']}, 'x'], 'calibrated_at': 'nope'},
            'tolerance': {'mm': 'x', 'deg': 50}}
    st = SU.SuspensionSettings.from_dict(junk)
    assert st.source == 'typical' and st.driver_side == 'left' and st.track_front == 0.0 and st.gross_mass == 0.0
    assert st.corners['FL'].spring_rate == SU.LIMITS['spring_rate'][1] and st.corners['FL'].tyre_rate == 10.0
    assert st.corners['FL'].curve == [] and st.corners['FL'].bump_curve                  # bad tables are skipped
    assert [i.id for i in st.items] == ['box', 'box_2'] and st.items[0].mass == 12.0 and st.items[1].mass == 10000.0
    assert st.cases[0].items == {'box': None} and st.calibrated_at == '' and st.tol_deg == 10.0
    with pytest.raises(ValueError) as e:
        SU.SuspensionSettings.from_dict(junk, strict=True)
    msg = str(e.value)
    assert 'vehicle.track_front' in msg and 'corners: "ZZ"' in msg and 'wheel load must rise' in msg
    assert SU.SuspensionSettings.from_dict(None).to_dict() == SU.SuspensionSettings().to_dict()


def test_settings_round_trip_and_old_projects(rig):
    st = SU.settings_for(rig)
    assert st.is_typical and len(st.cases) == 5 and st.calibrated_at == 'cal'
    assert SU.SuspensionSettings.from_dict(st.to_dict(), base=SU.typical(rig)).to_dict() == st.to_dict()
    # an old project file: no suspension settings at all
    from vantage.core.project import AnalysisSettings
    a = AnalysisSettings.from_dict({'cell_m': 0.05, 'load_cases': [['Kerb', 0, 0]]})
    assert a.suspension == {} and a.load_cases == [['Kerb', 0, 0]]
    q = samples.tigor_sample_project()
    q.analysis = a
    assert SU.issues(q) == [] and SU.settings_for(q).is_typical
    # the driver side of partial settings places the driver
    q.analysis.suspension = {'vehicle': {'driver_side': 'right'}}
    s2 = SU.settings_for(q)
    assert s2.item('driver').y < 0 < s2.item('passenger').y


def test_typical_values_follow_the_vehicle(rig):
    st = SU.typical(rig)
    g = SU.vehicle_geometry(rig)
    assert g['track']['front'] == pytest.approx(2 * 0.7455, abs=1e-3)       # the tyre centre lines of the model
    assert all(300 < st.corners[c].ride_height < 340 for c in SU.CORNERS)    # wheel centre to the arch on the model
    assert 1100 < st.kerb_mass < 1350 and st.front_pct == pytest.approx(60)
    # ride frequencies of the typical rates (sprung corner mass, in series with the tyre)
    for c in SU.CORNERS:
        co = st.corners[c]
        ride = _series(co.wheel_rate, co.tyre_rate) * 1000
        f = math.sqrt(ride / (co.kerb_load - SU.TYP_UNSPRUNG_KG)) / (2 * math.pi)
        assert f == pytest.approx(SU.TYP_RIDE_HZ[SU.AXLE[c]], abs=0.02)
    seats = SU.seat_positions(rig, 'left')
    assert seats['driver'][1] > 0 and seats['passenger'][1] < 0 and 0 < seats['rear_left'][0] < seats['driver'][0]
    assert seats['luggage'][0] < 0                                             # behind the rear axle
    blank = samples.blank_project()
    bst = SU.settings_for(blank)
    xy = SU.corner_positions(blank, bst)
    assert xy['FL'] == (SU.TYP_WHEELBASE, SU.TYP_TRACK / 2) and SU.solve_cases(blank).cases


def test_rig_component_masses(rig):
    from vantage.core.project import RigPart
    p = rig
    v = p.variant
    st = SU.settings_for(p)
    comps = {c['tag']: c for c in SU.rig_components(p, st)}
    f = v.by_tag('F')
    assert comps['F']['mass'] == pytest.approx(p.module_for(f).weight_g / 1000 + 0.015)
    lid = p.sensor_model_for(next(s for s in v.sensors if s.tag == 'L1'))
    vol = math.pi * (lid.size_mm[0] / 10) * (lid.size_mm[1] / 10) / 4 * lid.size_mm[2] / 10
    assert comps['L1']['mass'] == pytest.approx(vol * 1.2 / 1000 + 0.4, abs=1e-3) and comps['ECU']['mass'] == 2.5
    assert {'GN1', 'IMU1'} <= set(comps) and comps['L1']['source'].startswith('typical')
    st.rig_masses = {comps['L1']['uid']: 0.95, 'RD1': 0.5}                   # by uid and by tag
    comps2 = {c['tag']: c for c in SU.rig_components(p, st)}
    assert comps2['L1']['mass'] == 0.95 and comps2['RD1']['mass'] == 0.5 and comps2['L1']['source'] == 'entered'
    part = RigPart(name='Plate', kind='box', length=1.0, width=0.5, height=0.1)
    assert SU.rig_part_mass(part) == pytest.approx(2 * (0.5 + 0.1 + 0.05) * 0.003 * 2700)
    bar = RigPart(name='Bar', kind='bar', length=1.2, radius=0.02)
    assert SU.rig_part_mass(bar) == pytest.approx((2 * math.pi * 0.02 * 1.2 + 2 * math.pi * 0.0004) * 0.003 * 2700)
    s = next(s for s in v.sensors if s.tag == 'RD5')
    s.visible = False
    try:
        assert 'RD5' not in {c['tag'] for c in SU.rig_components(p, st)}      # hidden: left out, as the analysis does
    finally:
        s.visible = True
    case = SU.LoadCase('c', 'C', {'rig': 10.0})
    loads = SU.case_loads(p, st, case)
    assert sum(m for _, m, *_ in loads) == pytest.approx(10.0)               # a case's rig mass scales every part


# ----------------------------------------------------------------------------- files
def test_file_round_trips(tmp_path, rig):
    st = SU.settings_for(rig)
    st.corners['FL'].curve = [[2500.0, -30.0], [3600.0, 0.0], [5200.0, 30.0]]
    st.gross_mass = 1650.0
    st.driver_side = 'right'
    st.rig_masses = {'L1': 0.9}
    st.source = 'entered'
    for ext in ('json', 'yaml'):
        path = str(tmp_path / f's.{ext}')
        assert SU.save_file(st, path) == [path]
        back, what, warn = SU.load_file(path, SU.typical(rig))
        assert what == 'suspension file' and not warn and back.source == 'imported'
        a, b = back.to_dict(), st.to_dict()
        for d in (a, b):
            d.pop('source')
            d.pop('note')
        assert a == b
    # CSV: the parameter sheet plus the curves
    path = str(tmp_path / 's.csv')
    paths = SU.save_file(st, path)
    assert paths == [path, str(tmp_path / 's_curves.csv')]
    base = SU.typical(rig)
    back, what, _ = SU.load_file(paths[0], base)
    back, what2, _ = SU.load_file(paths[1], back)
    assert what == 'parameter sheet' and what2 == 'ride-height curve for FL'
    for c in SU.CORNERS:
        assert back.corners[c].to_dict() == st.corners[c].to_dict()
    assert (back.gross_mass, back.driver_side, back.arb_front) == (1650.0, 'right', st.arb_front)


def test_import_converts_units_and_reports_errors(tmp_path):
    base = SU.typical()
    y = tmp_path / 'k.yaml'
    y.write_text('schema: vantage.suspension_data/1\nunits: {rate: lbf/in}\n'
                 'vehicle: {kerb_mass: 2.7 kN, front_pct: 58}\n'
                 'axles:\n  front: {spring_rate: 140, tyre_rate: 200 N/mm, bump_gap: 2.5 in}\n'
                 '  rear: {wheel_rate: 20 kN/m, axle_load: 1100 lb}\n'
                 'corners:\n  FL: {ride_height: 0.36 m}\n', encoding='utf-8')
    st, _, warn = SU.load_file(str(y), base)
    assert st.corners['FR'].spring_rate == pytest.approx(140 * 4.4482216152605 / 25.4)
    assert st.corners['FL'].tyre_rate == 200.0 and st.corners['FL'].bump_gap == pytest.approx(63.5)
    assert st.corners['RL'].wheel_rate == pytest.approx(20.0)
    assert st.corners['RL'].kerb_load + st.corners['RR'].kerb_load == pytest.approx(1100 * 0.45359237)
    assert st.corners['FL'].kerb_load + st.corners['FR'].kerb_load == pytest.approx(2700 / G * 0.58)
    assert st.corners['FL'].ride_height == pytest.approx(360.0) and st.corners['FR'].ride_height == base.corners[
        'FR'].ride_height
    bad = tmp_path / 'bad.yaml'
    bad.write_text('corners:\n  FL: {spring_rate: 25 N/m², kerb_load: heavy}\n', encoding='utf-8')
    with pytest.raises(ValueError) as e:
        SU.load_file(str(bad), base)
    assert 'bad.yaml' in str(e.value) and 'corners.FL.spring_rate' in str(e.value) and 'kerb_load' in str(e.value)
    other = tmp_path / 'other.json'
    other.write_text(json.dumps({'schema': 'vantage.extrinsics/1', 'corners': {}}), encoding='utf-8')
    with pytest.raises(ValueError, match='not a suspension file'):
        SU.load_file(str(other), base)
    # ride-height tables: units in the header, ride height falling with load, a corner column or a chosen corner
    t = tmp_path / 'rh.csv'
    t.write_text('# K&C bounce\nload (kN),ride_height (mm)\n2.5,380\n3.6,350\n5.2,320\n', encoding='utf-8')
    with pytest.raises(ValueError, match='choose the corner'):
        SU.load_file(str(t), base)
    st, what, _ = SU.load_file(str(t), base, corner='rear')
    assert what == 'ride-height curves for RL, RR'
    assert st.corners['RL'].curve == [[2500.0, -380.0], [3600.0, -350.0], [5200.0, -320.0]]
    kg = tmp_path / 'kg.csv'
    kg.write_text('corner;load_kg;travel_mm\nFL;300;-10\nFL;400;12\nLR;250;-5\nLR;350;20\n', encoding='utf-8')
    st, what, _ = SU.load_file(str(kg), base)
    assert st.corners['FL'].curve[1] == pytest.approx([400 * G, 12.0]) and st.corners['RL'].curve
    nm = tmp_path / 'nm.csv'
    nm.write_text('corner,load_N,travel_mm\nFL,3000,0\nFL,2900,10\n', encoding='utf-8')
    with pytest.raises(ValueError, match='must rise'):
        SU.load_file(str(nm), base)
    sheet = tmp_path / 'p.csv'
    sheet.write_text('parameter,value,unit\nFL.spring_rate,24,N/mm\nfront.tyre_rate,190,N/mm\nkerb_mass,1300,kg\n'
                     'RR.warp_factor,9,\n', encoding='utf-8')
    with pytest.raises(ValueError, match='unknown field "warp_factor"'):
        SU.load_file(str(sheet), base)


# ----------------------------------------------------------------------------- load cases: typed, exactly as before
def _old_body_pose(front_mm, rear_mm, wheelbase):
    theta = math.atan2((front_mm - rear_mm) / 1000.0, max(wheelbase, 0.5))
    return rot_y(theta), np.array([0.0, 0.0, -rear_mm / 1000.0]), math.degrees(theta)


def _old_reground(cam_pos, P_world, R, t):
    Pb = (P_world - t) @ R
    d = Pb - cam_pos
    with np.errstate(divide='ignore', invalid='ignore'):
        s = -cam_pos[2] / d[:, 2]
    return cam_pos + d * s[:, None], s > 0


def _old_evaluate(engine, project, cases, cell=None):
    """loadcases.evaluate as it was before the suspension (round 4), verbatim."""
    wc = project.vehicle.wheel_centres
    wb = abs(wc['FL'][0] - wc['RL'][0]) if 'FL' in wc and 'RL' in wc else 2.6
    base = engine.analyse(project, cell=cell)
    cams = {c.uid: c for c in project.variant.cameras}
    g = base.grid
    jj, ii = np.nonzero(g.roi & (base.best >= 0))
    Pc = np.stack([g.xs[ii], g.ys[jj], np.zeros(len(ii))], 1)
    best = base.best[jj, ii]
    seam_pts = []
    cx, cy = base.centre
    for sm in base.seams:
        a = math.radians(sm.angle)
        rr = np.arange(0.2, 8.0, g.cell)
        X, Y = cx + rr * math.cos(a), cy + rr * math.sin(a)
        ia = base.cam_index(sm.a)
        ib = base.cam_index(sm.b)
        for x, y in zip(X, Y):
            cc = g.cell_of(x, y)
            if cc and g.roi[cc[1], cc[0]] and base.cams[ia].vis[cc[1], cc[0]] and base.cams[ib].vis[cc[1], cc[0]]:
                seam_pts.append((x, y, sm.a, sm.b))
    out = []
    for name, f, r in cases:
        R, t, pitch = _old_body_pose(float(f), float(r), wb)
        res = engine.analyse(project, cell=cell, body_pose=(R, t)) if (f or r) else base
        m = res.metrics
        shift = 0.0
        for k, ce in enumerate(base.cams):
            sel = best == k
            if not sel.any() or ce.uid not in cams:
                continue
            Q, ok = _old_reground(cams[ce.uid].pos(), Pc[sel], R, t)
            if ok.any():
                shift = max(shift, float(np.max(np.linalg.norm(Q[ok, :2] - Pc[sel][ok, :2], axis=1))))
        ghost = 0.0
        for x, y, ua, ub in seam_pts:
            P = np.array([[x, y, 0.0]])
            qa, oka = _old_reground(cams[ua].pos(), P, R, t)
            qb, okb = _old_reground(cams[ub].pos(), P, R, t)
            if oka[0] and okb[0]:
                ghost = max(ghost, float(np.linalg.norm(qa[0, :2] - qb[0, :2])))
        d = m['blind_m2'] - base.metrics['blind_m2']
        ghost_cm, shift_cm = ghost * 100, shift * 100
        status = 'good' if ghost_cm < 2 and d < 0.01 else ('warning' if ghost_cm < 5 and d < 0.05 else 'serious')
        out.append((name, float(f), float(r), pitch, m['blind_m2'], m['min_density'], m['multi_pct'], d, shift_cm,
                    ghost_cm, status))
    return out


def test_typed_cases_evaluate_exactly_as_before(engine):
    p = samples.tigor_sample_project()
    p.analysis.load_cases = [['Kerb', 0, 0], ['Driver', 8, 6], ['Heavy', 22, 55], ['Nose', 30, -5]]
    cases = LC.cases_of(p)
    old = _old_evaluate(engine, p, cases, cell=0.1)
    new = LC.evaluate(engine, p, cases, cell=0.1)
    assert len(new) == len(old) == 4
    for o, n in zip(old, new):
        assert (n.name, n.front_mm, n.rear_mm, n.pitch_deg, n.blind_m2, n.min_density, n.multi_pct, n.delta_blind,
                n.ground_shift_cm, n.seam_ghost_cm, n.status) == o
        assert n.source == 'typed' and n.roll_deg == 0.0 and n.corners == {'FL': o[1], 'FR': o[1], 'RL': o[2],
                                                                           'RR': o[2]}
    # the two-drop pose is unchanged; a roll adds to it
    R0, t0, p0 = LC.body_pose(30.0, 10.0, 2.5)
    Ro, to, po = _old_body_pose(30.0, 10.0, 2.5)
    assert np.array_equal(R0, Ro) and np.array_equal(t0, to) and p0 == po
    R1, _, _ = LC.body_pose(30.0, 10.0, 2.5, roll_deg=0.5)
    assert np.allclose(R1, Rotation.from_euler('YX', [p0, 0.5], degrees=True).as_matrix(), atol=1e-12)


# ----------------------------------------------------------------------------- load cases: roll, rigid moves
def test_evaluate_with_roll_moves_the_cameras_rigidly(engine):
    p = samples.tigor_sample_project()
    v = p.variant
    case = LC.CornerCase('rolled', heave_mm=14.0, pitch_deg=0.3, roll_deg=-0.6,
                         corners={'FL': 1, 'FR': 2, 'RL': 3, 'RR': 4})
    flat = LC.CornerCase('flat', heave_mm=14.0, pitch_deg=0.3, roll_deg=0.0)
    r, rf = LC.evaluate(engine, p, [case, flat], cell=0.1)
    assert r.source == 'suspension' and r.roll_deg == -0.6 and r.reference == 'kerb'
    # an independent rigid transform: intrinsic pitch about y then roll about x, translated down by the heave
    Rb = Rotation.from_euler('YX', [0.3, -0.6], degrees=True)
    t = np.array([0, 0, -0.014])
    devs = {d['tag']: d for d in r.devices}
    assert set(devs) == {c.tag for c in v.cameras if c.visible}
    for c in v.cameras:
        moved = Rb.apply(c.pos()) + t
        d = devs[c.tag]
        assert [d['dx_mm'], d['dy_mm'], d['dz_mm']] == pytest.approx(list((moved - c.pos()) * 1000), abs=1e-9)
        cam = Rotation.from_euler('ZYX', [c.yaw, c.pitch, c.roll], degrees=True)
        ypr = (Rb * cam).as_euler('ZYX', degrees=True)
        assert d['d_pitch_deg'] == pytest.approx(ypr[1] - c.pitch, abs=1e-9)
        assert d['d_yaw_deg'] == pytest.approx((ypr[0] - c.yaw + 180) % 360 - 180, abs=1e-9)
        assert d['turn_deg'] == pytest.approx(math.degrees(Rb.magnitude()), abs=1e-9)
    # small angles: a camera looking at azimuth yaw sees the body's pitch times cos(yaw) and its roll times -sin(yaw)
    # as its own pitch (a left-looking mirror camera tilts down when the left side drops: roll < 0)
    for tag in ('F', 'ML', 'MR', 'R'):
        psi = math.radians(v.by_tag(tag).yaw)
        assert devs[tag]['d_pitch_deg'] == pytest.approx(0.3 * math.cos(psi) + 0.6 * math.sin(psi), abs=0.01)
    # ground points at the corners drop by the plane: heave + x tan(pitch) - y tan(roll)
    for x, y in ((2.45, 0.75), (2.45, -0.75), (0.0, 0.75), (0.0, -0.75)):
        z = (Rb.apply([x, y, 0.0]) + t)[2]
        assert -z * 1000 == pytest.approx(14 + x * 1000 * math.tan(math.radians(0.3)) - y * 1000 *
                                          math.tan(math.radians(-0.6)), abs=0.01)
    # the roll is what makes the two sides' seams disagree more than the same pitch and heave alone
    assert r.seam_ghost_cm > rf.seam_ghost_cm and r.ground_shift_cm >= rf.ground_shift_cm * 0.99


def test_corner_cases_are_relative_to_the_calibration_load(rig, engine):
    p = rig
    res = SU.solve_cases(p)
    cal = res.case('cal')
    assert cal.worst_mm == 0.0 and cal.turn_deg == 0.0 and res.calibration.startswith('Rig')
    four = res.case('four')
    cs, fs = res.calibration_statics, four.statics
    R0, t0 = LC.pose(cs.heave_mm, cs.pitch_deg, cs.roll_deg)
    R1, t1 = LC.pose(fs.heave_mm, fs.pitch_deg, fs.roll_deg)
    for d in four.devices:
        o = next(x for x in list(p.variant.cameras) + list(p.variant.sensors) if x.uid == d['uid'])
        want = ((R1 @ o.pos() + t1) - (R0 @ o.pos() + t0)) * 1000
        assert [d['dx_mm'], d['dy_mm'], d['dz_mm']] == pytest.approx(list(want), abs=1e-9)
    assert four.rel['heave_mm'] == pytest.approx(fs.heave_mm - cs.heave_mm)
    assert four.worst_mm > 10 and fs.pitch_deg < cs.pitch_deg < 0                 # rear heavy: nose up
    # the evaluation: the calibration case reproduces the calibrated stitch exactly, heavier loads ghost more
    lr = LC.evaluate(engine, p, SU.corner_cases(res), cell=0.1)
    by = dict(zip([c.id for c in res.cases], lr))
    assert by['cal'].ground_shift_cm < 1e-6 and by['cal'].seam_ghost_cm < 1e-6 and by['cal'].delta_blind == 0
    assert by['cal'].reference.startswith('Rig') and all(d['move_mm'] < 1e-9 for d in by['cal'].devices)
    assert by['driver'].seam_ghost_cm < by['four'].seam_ghost_cm < by['full'].seam_ghost_cm
    assert by['full'].corners['RL'] == pytest.approx(res.case('full').statics.corners['RL'].drop_mm)
    # calibrated at kerb instead: the rig-only case now moves the cameras
    st = SU.settings_for(p)
    st.calibrated_at = ''
    res2 = SU.solve_cases(p, st)
    assert res2.case('cal').worst_mm > 5 and res2.calibration.startswith('kerb')
    assert SU.corner_cases(res2)[0].reference is None


# ----------------------------------------------------------------------------- the study: run, findings, issues, export
def test_run_findings_and_exports(rig, engine, tmp_path):
    p = rig
    st = SU.settings_for(p)
    st.gross_mass = 1500.0
    res = SU.run(p, engine, settings=st, cell=0.1)
    assert res.images and all(c.image is not None for c in res.cases)
    kinds = {f['kind'] for f in res.findings}
    assert {'typical', 'bump', 'calibration', 'gross', 'ghost'} <= kinds
    assert res.findings[0]['severity'] == 'serious'
    full = res.case('full')
    assert full.statics.bump == ['RL', 'RR'] and full.over_gross == pytest.approx(full.statics.total_kg - 1500)
    summ = SU.summary(res)
    json.dumps(summ)
    assert summ['suspension'] == 'typical values' and len(summ['cases']) == 5
    doc = SU.to_doc(res, p)
    assert doc['schema'] == SU.SCHEMA and doc['frame'].startswith('vehicle') and doc['typical_suspension']
    assert doc['cases'][4]['corners']['RL']['on_bump_stop'] and doc['cases'][1]['ground']['seam_ghost_cm'] > 0
    names = [n for n, _ in SU.files(res, p)]
    assert names == ['suspension_load_cases.json', 'suspension_load_cases.csv', 'suspension_devices.csv',
                     'suspension_data.json']
    paths = SU.export(res, p, str(tmp_path / 'out'))
    assert all(os.path.getsize(x) > 0 for x in paths)
    rows = (tmp_path / 'out' / 'suspension_devices.csv').read_text().splitlines()
    assert len(rows) == 1 + 5 * len(res.cases[0].devices)
    back, _, _ = SU.load_file(str(tmp_path / 'out' / 'suspension_data.json'), SU.typical(p))
    assert back.gross_mass == 1500.0 and len(back.cases) == 5
    # cancel and statics only
    assert SU.run(p, engine, settings=st, cancel=lambda: True, cell=0.1) is None
    quick = SU.run(p, settings=st, images=False)
    assert not quick.images and quick.cases[0].image is None


def test_issues_are_gated_cached_and_cheap(rig):
    p = rig
    p.analysis.suspension = {}
    assert SU.issues(p) == []
    st = SU.settings_for(p)
    p.analysis.suspension = st.to_dict()
    SU._ISSUE_CACHE.clear()
    t0 = time.perf_counter()
    iss = SU.issues(p)
    first = time.perf_counter() - t0
    t0 = time.perf_counter()
    again = SU.issues(p)
    cached = time.perf_counter() - t0
    try:
        assert first < 0.05 and cached < first and [i.message for i in again] == [i.message for i in iss]
        msgs = [i.message for i in iss]
        assert any('Bump stops touched' in m for m in msgs) and all(m.endswith('(typical suspension values)') for m in msgs)
        assert {i.severity for i in iss} <= {'critical', 'serious', 'warning'}
        cal = next(i for i in iss if 'calibration tolerance' in i.message)
        assert cal.uids and 'F' in cal.objects
        st.source = 'entered'
        st.tol_mm, st.tol_deg = 500.0, 10.0
        for c in SU.CORNERS:
            st.corners[c].bump_gap = 300.0
        p.analysis.suspension = st.to_dict()
        assert SU.issues(p) == []
    finally:
        p.analysis.suspension = {}


# ----------------------------------------------------------------------------- the dialog
from test_gui import app, win, pump, settle, _wait  # noqa: E402,F401  (fixtures)


def test_load_cases_dialog_drives_the_suspension(app, win, tmp_path):
    st = win.store
    st.project.active_uid = next(v for v in st.project.variants if v.sensors).uid
    st.project_replaced.emit()
    settle(app, win)
    d = win.load_cases()
    pump(app, 0.2)
    assert d.table.rowCount() == 4                               # the typed drops: as before
    assert d.ctable.rowCount() == 5 and 'Typical values' in d.typical_lbl.text()
    assert d.ctable.item(0, 0).font().bold() and not d.ctable.item(1, 0).font().bold()
    assert d.cal_combo.currentData() == 'cal'
    fl_before = float(d.ctable.item(3, 2).text().replace('−', '-'))
    n0 = st.undo.index()
    # import a ride-height curve for the front axle (stiffer than the typical spring)
    csv = tmp_path / 'front.csv'
    csv.write_text('corner,load_N,travel_mm\n' + ''.join(f'{c},{l},{t}\n' for c in ('FL', 'FR')
                                                         for l, t in ((2500, -40), (3620, 0), (6000, 30))),
                   encoding='utf-8')
    assert d.import_file(str(csv))
    pump(app, 0.1)
    s = d.settings()
    assert s.source == 'imported' and len(s.corners['FL'].curve) == 3 and not s.corners['RL'].curve
    assert d.stable.item(0, 10).text().startswith('3 points') and 'Typical' not in d.typical_lbl.text()
    assert float(d.ctable.item(3, 2).text().replace('−', '-')) < fl_before      # stiffer front: less drop
    assert st.undo.index() == n0 + 1
    # type a spring rate in lbf/in: both rear wheels take it
    d.stable.item(2, 1).setText('140 lbf/in')
    pump(app, 0.1)
    s = d.settings()
    assert s.corners['RR'].spring_rate == pytest.approx(24.517, abs=1e-3) and s.corners['RL'].spring_rate == \
        s.corners['RR'].spring_rate
    # add a roof load to the selected case ("Driver")
    d.ctable.setCurrentCell(1, 0)
    pump(app, 0.05)
    added0 = float(d.ctable.item(1, 1).text())
    rid = d.add_item('roof')
    pump(app, 0.1)
    assert rid == 'roof_2' and d.settings().case('driver').items.get(rid, 'missing') is None
    assert float(d.ctable.item(1, 1).text()) == pytest.approx(added0 + 30.0, abs=0.06)
    # tick the boot luggage into the case in the case's item list, then give it 25 kg for this case
    row = next(i for i in range(d.mtable.rowCount()) if d.mtable.item(i, 0).data(Qt.ItemDataRole.UserRole) == 'luggage')
    d.mtable.item(row, 0).setCheckState(Qt.CheckState.Checked)
    pump(app, 0.05)
    row = next(i for i in range(d.mtable.rowCount()) if d.mtable.item(i, 0).data(Qt.ItemDataRole.UserRole) == 'luggage')
    d.mtable.item(row, 1).setText('25')
    pump(app, 0.05)
    assert d.settings().case('driver').items['luggage'] == 25.0
    assert d.dtable.rowCount() == len([c for c in st.variant.cameras if c.visible]) + len(st.variant.sensors)
    # evaluate (background)
    _wait(app, lambda: win.ctx.analysis.result is not None)
    d.run()
    _wait(app, lambda: not d.busy and d.results is not None)
    assert len(d.results) == 4 and d.sres is not None and d.sres.images
    assert d.ctable.item(1, 13).text() and d.ctable.item(1, 14).text()
    assert d.table.item(3, 9).text()                            # the typed drops got their figures too
    # undo every edit: back to the project without suspension settings
    while st.undo.index() > n0:
        st.undo.undo()
        pump(app, 0.02)
    pump(app, 0.1)
    assert st.project.analysis.suspension == {} and d.ctable.rowCount() == 5
    assert 'Typical values' in d.typical_lbl.text() and not d.ctable.item(1, 13).text()
    st.undo.redo()
    pump(app, 0.1)
    assert d.settings().corners['FL'].curve
    # export and import again
    out = d.export_file(str(tmp_path / 'susp.yaml'))
    assert out and SU.load_file(out[0], SU.typical(st.project))[0].corners['FL'].curve
    # a bad file says why and changes nothing
    bad = tmp_path / 'bad.csv'
    bad.write_text('corner,load_N,travel_mm\nFL,3000,0\nFL,2000,5\n', encoding='utf-8')
    n1 = st.undo.index()
    assert not d.import_file(str(bad)) and 'must rise' in d.msg_lbl.text() and st.undo.index() == n1
    # vehicle fields read units; the driver side swaps the front occupants
    d.fields['kerb_mass'].setText('12 kN')
    d.fields['kerb_mass'].setModified(True)
    d.fields['kerb_mass'].editingFinished.emit()
    pump(app, 0.05)
    assert d.settings().kerb_mass == pytest.approx(12000 / G, abs=0.01)
    y = d.settings().item('driver').y
    d.side.buttons['right'].click()
    pump(app, 0.05)
    assert d.settings().driver_side == 'right' and d.settings().item('driver').y == pytest.approx(-abs(y))
    # the calibration load
    d.cal_combo.setCurrentIndex(0)
    pump(app, 0.05)
    assert d.settings().calibrated_at == '' and not d.ctable.item(0, 0).font().bold()
    d.ctable.item(2, 0).setText('Driver + data engineer')
    pump(app, 0.05)
    assert d.settings().case('crew').name == 'Driver + data engineer'
    # a new project replaces what the dialog shows
    from vantage.core import samples as S
    st.set_project(S.tigor_sample_project())
    settle(app, win)
    assert d.sres is None and d.settings().is_typical and d.ctable.rowCount() == 5
    d.close()

