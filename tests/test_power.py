"""Power budget: the data rig's budget against hand-computed figures (loads, the DC-DC converter, the alternator, the
battery, circuits and fuses), the power cables against copper resistance tables and the constant-power drop, the
settings, the findings and Issues, the export round trip and the page (offscreen)."""
import copy
import csv
import json
import math
import time

import pytest

from vantage.core import power as PW, rules, samples, sensors
from vantage.core.project import Sensor

# the data rig: 6 cameras (no module figure: the typical 1.5 W), an OS1-128 (20 W), a long-range radar (6.6 W) and
# four corner radars (4 W), plus the INS (12 W) and two survey antennas (0.2 W) the tests add; compute 60 W,
# recorder 40 W, switch 15 W, time server 5 W
CAM_W, OS1_W, LONG_W, CORNER_W, INS_W, ANT_W = 1.5, 20.0, 6.6, 4.0, 12.0, 0.2
FIXED_W = 60.0 + 40.0 + 15.0 + 5.0
STEADY = 6 * CAM_W + OS1_W + LONG_W + 4 * CORNER_W + INS_W + 2 * ANT_W + FIXED_W             # 184 W
PEAK = (6 * CAM_W + LONG_W + 4 * CORNER_W + INS_W + 2 * ANT_W + FIXED_W) * 1.2 + OS1_W * 2.0  # 236.8 W
CABLED = ('L1', 'RD1', 'RD2', 'RD3', 'RD4', 'RD5', 'IMU1')
# copper resistance at 20 °C, Ω/km (published AWG tables, solid conductor)
OHM_PER_KM = {10: 3.277, 12: 5.211, 14: 8.286, 16: 13.17, 18: 20.95, 20: 33.31, 22: 52.96, 24: 84.22}


@pytest.fixture(scope='module')
def base():
    return samples.tigor_sample_project()


def _add(p, kind, model_id, tag, **kw):
    m = p.sensor_models[model_id]
    s = Sensor(kind=kind, model_id=model_id, tag=tag,
               **sensors.default_pose(kind, m, p.vehicle.mesh, p.rig, p.variant.sensors))
    for k, v in kw.items():
        setattr(s, k, v)
    p.variant.sensors.append(s)
    return s


@pytest.fixture()
def p(base):
    """The data rig, active, with an INS and two antennas."""
    q = copy.deepcopy(base)
    q.active_uid = next(v for v in q.variants if v.sensors).uid
    _add(q, 'imu', 'ins_dual', 'IMU1')
    _add(q, 'gnss', 'gnss_survey', 'GN1')
    _add(q, 'gnss', 'gnss_survey', 'GN2')
    return q


def _sensor(v, tag):
    return next(s for s in v.sensors if s.tag == tag)


def _short(p, **kw):
    """Settings with every power cable 0 m long (no drop: currents are P / V exactly)."""
    d = {'cables': {s.uid: {'m': 0.0} for s in p.variant.sensors if s.kind != 'gnss'}}
    d.update(kw)
    return PW.PowerSettings.from_dict(d)


def _cv(v_bus, p_w, r):
    """Constant-power load through r: the voltage at the device (independent of the module)."""
    return (v_bus + math.sqrt(v_bus ** 2 - 4 * p_w * r)) / 2


# ----------------------------------------------------------------------------- the budget, by hand
def test_the_data_rig_by_hand(p):
    res = PW.run(p, settings=_short(p))
    assert [r['tag'] for r in res.rows] == ['F', 'R', 'ML', 'MR', 'QL', 'QR', 'L1', 'RD1', 'RD2', 'RD3', 'RD4', 'RD5',
                                           'IMU1', 'GN1', 'GN2', 'Compute', 'Recorder', 'Switch', 'PTP']
    f = res.row('F')
    assert (f['steady_w'], f['peak_w'], f['source'], f['circuit']) == (1.5, pytest.approx(1.8), 'typical', 'compute')
    l1 = res.row('L1')
    assert (l1['steady_w'], l1['peak_w'], l1['source'], l1['circuit']) == (20.0, 40.0, 'library', 'lidar')
    assert l1['steady_a'] == pytest.approx(20 / 12) and l1['peak_a'] == pytest.approx(40 / 12)
    assert res.row('RD1')['peak_w'] == pytest.approx(6.6 * 1.2) and res.row('GN1')['circuit'] == 'gnss_imu'
    assert res.row('GN1')['cable_m'] is None and res.row('IMU1')['cable_m'] == 0.0
    t = res.totals
    assert t['steady_w'] == pytest.approx(STEADY) and STEADY == pytest.approx(184.0)
    assert t['peak_w'] == pytest.approx(PEAK) and PEAK == pytest.approx(236.8)
    assert t['cable_loss_w'] == pytest.approx(0.0, abs=1e-9) and t['bus_a'] == pytest.approx(184 / 12)
    # the converter: 184 W out of 600 W, 200 W in at 92 %
    assert t['dcdc_output_w'] == pytest.approx(184.0) and t['dcdc_load'] == pytest.approx(184 / 600)
    assert t['dcdc_input_w'] == pytest.approx(200.0) and t['dcdc_loss_w'] == pytest.approx(16.0)
    assert t['dcdc_input_peak_w'] == pytest.approx(236.8 / 0.92) and t['input_a'] == pytest.approx(200 / 13.8)
    # the alternator: 50 A × 13.8 V = 690 W spare, 490 W left; the battery: 1000 Wh × 0.8 / 200 W = 4 h
    assert t['alternator_spare_w'] == pytest.approx(690.0) and t['alternator_margin_w'] == pytest.approx(490.0)
    assert math.isinf(t['battery_drain_min']) and t['runtime_min'] == pytest.approx(240.0)
    # the enclosure: compute, recorder, switch, time server and the converter's loss
    assert t['heat_w'] == pytest.approx(FIXED_W + 16.0) and t['heat_tags'] == ['Compute', 'Recorder', 'Switch', 'PTP']
    assert t['by_kind_w']['radar'] == pytest.approx(22.6) and t['counts']['camera'] == 6
    # nothing to fix on the sample rig: only notes (no camera power figure, the heat) and good lines
    assert res.worst() == 'good' and PW.issues(p) == []
    assert {f['kind'] for f in res.findings if f['severity'] == 'note'} == {'no_figure', 'heat'}
    assert next(f for f in res.findings if f['kind'] == 'no_figure')['tags'] == ['F', 'R', 'ML', 'MR', 'QL', 'QR']


def test_circuits_currents_and_automatic_fuses(p):
    res = PW.run(p, settings=_short(p))
    # (steady A, peak A, the smallest standard fuse with steady ≤ 80 % and peak ≤ the fuse)
    want = {'compute': ((60 + 9) / 12, (72 + 10.8) / 12, 7.5), 'lidar': (20 / 12, 40 / 12, 4.0),
            'radar': (22.6 / 12, 22.6 * 1.2 / 12, 3.0), 'gnss_imu': (12.4 / 12, 12.4 * 1.2 / 12, 2.0),
            'recorder': (40 / 12, 48 / 12, 5.0), 'network': (20 / 12, 24 / 12, 3.0)}
    assert [c['circuit'] for c in res.circuits] == list(want)
    for c in res.circuits:
        sa, pa, fuse = want[c['circuit']]
        assert c['steady_a'] == pytest.approx(sa) and c['peak_a'] == pytest.approx(pa)
        assert c['fuse_a'] == fuse and c['fuse_auto'] and c['severity'] == 'good'
        assert c['load'] == pytest.approx(sa / fuse)
    assert res.circuit('compute')['tags'] == ['F', 'R', 'ML', 'MR', 'QL', 'QR', 'Compute']
    assert res.circuit('compute')['uids'] == [c.uid for c in p.variant.cameras]
    assert res.totals['worst_circuit'] == 'compute'                     # 5.75 of 7.5 A: 77 %
    assert PW.auto_fuse(5.75, 6.9) == 7.5 and PW.auto_fuse(0, 0) == 1.0 and PW.auto_fuse(100, 100) is None
    # on a 24 V bus every current halves
    res24 = PW.run(p, settings=_short(p, bus_v=24))
    assert res24.circuit('compute')['steady_a'] == pytest.approx(69 / 24) and res24.circuit('compute')['fuse_a'] == 4.0
    # a camera moved to its own circuit; a fixed load to another
    st = _short(p, circuit_of={p.variant.by_tag('F').uid: 'cameras', 'recorder': 'compute'})
    res = PW.run(p, settings=st)
    assert res.row('F')['circuit'] == 'cameras' and res.circuit('cameras')['tags'] == ['F']
    assert res.circuit('recorder') is None and 'Recorder' in res.circuit('compute')['tags']


def test_fitted_fuses_and_their_findings(p):
    def circ(**fuses):
        res = PW.run(p, settings=_short(p, fuses=fuses))
        return res, [f for f in res.findings if f['kind'] == 'circuit']
    # 5.75 A steady through a 5 A fuse: it blows
    res, fs = circ(compute=5)
    c = res.circuit('compute')
    assert (c['fuse_a'], c['fuse_auto'], c['severity']) == (5.0, False, 'critical')
    assert fs[0]['severity'] == 'critical' and 'over its 5 A fuse' in fs[0]['message'] and '7.5 A' in fs[0]['message']
    p.analysis.power = _short(p, fuses={'compute': 5}).to_dict()
    iss = PW.issues(p)
    assert [(i.severity, i.objects) for i in iss] == [('critical', 'Compute circuit')]
    assert iss[0].uids == [c.uid for c in p.variant.cameras] and isinstance(iss[0], rules.Issue)
    # 82 % of a 7 A fuse: over the 80 % for continuous loads
    res, fs = circ(compute=7)
    assert res.circuit('compute')['load'] == pytest.approx(5.75 / 7) and fs[0]['severity'] == 'warning'
    assert '80 %' in fs[0]['message']
    # the lidar spins up at 3.33 A on a 3 A fuse: short inrush, a warning
    res, fs = circ(lidar=3)
    assert res.circuit('lidar')['peak_load'] == pytest.approx(40 / 12 / 3) and fs[0]['severity'] == 'warning'
    assert 'inrush' in fs[0]['message'] and fs[0]['tags'] == ['L1']
    # a 40 W heater on top: 6.67 A at switch-on, over twice the 3 A fuse
    res = PW.run(p, settings=_short(p, fuses={'lidar': 3}, lidar_heater_w=40))
    assert res.row('L1')['peak_w'] == pytest.approx(80.0) and res.circuit('lidar')['severity'] == 'critical'
    assert 'twice' in next(f for f in res.findings if f['kind'] == 'circuit')['message']
    # a generous fuse everywhere is good, and the good line names the busiest
    res, fs = circ(compute=15)
    assert [f['severity'] for f in fs] == ['good'] and 'recorder' in fs[0]['message']


def test_dcdc_alternator_and_battery_findings(p):
    def find(kind, **kw):
        res = PW.run(p, settings=_short(p, **kw))
        return res, next(f for f in res.findings if f['kind'] == kind)
    res, f = find('dcdc', dcdc_w=150)
    assert f['severity'] == 'critical' and '184 W' in f['message'] and res.worst() == 'critical'
    _, f = find('dcdc', dcdc_w=200)                                    # 184 W steady, 237 W at switch-on
    assert f['severity'] == 'warning' and 'switch-on' in f['message']
    _, f = find('dcdc')
    assert f['severity'] == 'good'
    # 10 A × 13.8 V = 138 W spare against 200 W: the battery drains while driving, 800 Wh / 62 W
    res, f = find('alternator', alternator_a=10)
    assert res.totals['alternator_margin_w'] == pytest.approx(-62.0) and f['severity'] == 'serious'
    assert res.totals['battery_drain_min'] == pytest.approx(800 / 62 * 60) and '774 min' in f['message']
    res, f = find('alternator', alternator_a=10, battery_wh=0)
    assert 'vehicle\'s own battery' in f['message']
    _, f = find('alternator', alternator_a=16)                         # 200 of 220.8 W: over 80 %
    assert f['severity'] == 'warning'
    # 100 Wh × 0.8 / 200 W = 24 min, short of 30: a 125 Wh battery would do
    res, f = find('runtime', battery_wh=100)
    assert res.totals['runtime_min'] == pytest.approx(24.0) and f['severity'] == 'warning'
    assert '125 Wh' in f['message']
    _, f = find('runtime', battery_wh=0)
    assert f['severity'] == 'warning' and 'no auxiliary battery' in f['message']
    p.analysis.power = _short(p, battery_wh=100, alternator_a=10, dcdc_w=150).to_dict()
    assert [i.severity for i in PW.issues(p)] == ['critical', 'serious', 'warning']


# ----------------------------------------------------------------------------- power cables
def test_copper_resistance_matches_the_tables():
    for awg, ohm in OHM_PER_KM.items():
        assert PW.awg_ohm_per_km(awg) == pytest.approx(ohm, rel=2e-3)
    assert PW.awg_diameter_mm(18) == pytest.approx(1.024, abs=1e-3)
    assert PW.awg_diameter_mm(10) == pytest.approx(2.588, abs=1e-3)
    # the constant-power supply: P = V × I at the device and Vbus = V + I × R
    v, i = PW.supply(12.0, 20.0, 0.419)
    assert v * i == pytest.approx(20.0) and v + i * 0.419 == pytest.approx(12.0)
    assert PW.supply(12.0, 100.0, 1.0) == (None, pytest.approx(2 * 100 / 12))     # 144 < 400: it collapses
    assert PW.supply(12.0, 0.0, 1.0) == (12.0, 0.0) and PW.supply(24.0, 48.0, 0.0) == (24.0, 2.0)


def test_cable_drop_for_a_known_length_and_gauge(p):
    uid = _sensor(p.variant, 'L1').uid

    def l1(**cable):
        return PW.run(p, settings=PW.PowerSettings.from_dict({'cables': {uid: cable}})).row('L1')
    # 10 m of 18 AWG: 2 × 10 m × 20.95 Ω/km = 0.419 Ω there and back
    r = l1(m=10, awg=18)
    R = 2 * 10 * OHM_PER_KM[18] / 1000
    assert r['cable_ohm'] == pytest.approx(R, rel=2e-3) and r['cable_set'] and r['awg'] == 18
    vs, vp = _cv(12, 20, R), _cv(12, 40, R)
    assert vs == pytest.approx(11.2554, abs=1e-3)
    assert r['v_steady'] == pytest.approx(vs, abs=2e-3) and r['v_peak'] == pytest.approx(vp, abs=2e-3)
    assert r['drop_pct'] == pytest.approx((12 - vs) / 12 * 100, abs=0.02) and r['steady_a'] == pytest.approx(20 / vs,
                                                                                                             rel=1e-3)
    assert r['loss_w'] == pytest.approx(r['steady_a'] ** 2 * r['cable_ohm'])          # I²R
    assert r['cable_severity'] == 'serious' and r['cable_problem'] == 'drop'          # 6.2 % > 5 %
    res = PW.run(p, settings=PW.PowerSettings.from_dict({'cables': {uid: {'m': 10, 'awg': 18}}}))
    f = next(f for f in res.findings if f['kind'] == 'cable')
    assert f['severity'] == 'serious' and f['tags'] == ['L1'] and 'Use 14 AWG' in f['message']
    assert PW.fix_awg(res.settings, 20, 40, 10) == 14
    # 14 AWG: 2.4 % steady, 4.8 % at spin-up: good; with a 11.5 V minimum input the spin-up dips below it
    r = l1(m=10, awg=14)
    assert r['drop_pct'] == pytest.approx((12 - _cv(12, 20, 2 * 10 * OHM_PER_KM[14] / 1000)) / 12 * 100, abs=0.02)
    assert r['cable_severity'] == 'good'
    st = PW.PowerSettings.from_dict({'cables': {uid: {'m': 10, 'awg': 14}}, 'min_input_v': 11.5})
    assert PW.run(p, settings=st).row('L1')['cable_problem'] == 'min_input'
    # 16 AWG: 3.8 % steady but 8 % for the seconds of spin-up: a warning
    r = l1(m=10, awg=16)
    assert r['cable_problem'] == 'peak_drop' and r['cable_severity'] == 'warning'
    # 24 AWG: at spin-up 4 × 40 W × 1.68 Ω > 12² V²: the voltage collapses
    r = l1(m=10, awg=24)
    assert r['cable_problem'] == 'collapse' and r['v_peak'] is None and r['peak_drop_pct'] is None
    # the default length is the Manhattan route from the ECU, as a camera's cable
    s, e = _sensor(p.variant, 'L1'), p.ecu
    r = PW.run(p).row('L1')
    want = (abs(s.x - e.x) + abs(s.y - e.y) + abs(s.z - e.z)) * e.route_factor + e.service_loop_m
    assert not r['cable_set'] and r['cable_m'] == pytest.approx(want) and r['awg'] == 18
    # routed cables: the converter carries the loads and the cables' I²R
    res = PW.run(p)
    loss = sum(r['steady_a'] ** 2 * r['cable_ohm'] for r in res.rows if r['cable_m'] is not None)
    assert res.totals['cable_loss_w'] == pytest.approx(loss) and loss > 0
    assert res.totals['dcdc_output_w'] == pytest.approx(STEADY + loss)


# ----------------------------------------------------------------------------- devices, variants, edge cases
def test_hidden_missing_and_odd_devices(p, base):
    _sensor(p.variant, 'RD2').visible = False
    p.variant.by_tag('F').visible = False
    p.sensor_models['radar_corner'] = copy.deepcopy(p.sensor_models['radar_corner'])
    p.sensor_models['radar_corner'].power_w = float('nan')               # a hand-edited library: no figure
    p.modules['sturdecam31'].power_w = 2.2
    res = PW.run(p, settings=_short(p))
    assert res.row('RD2') is None and res.row('F') is None
    assert res.row('RD3')['steady_w'] == 0.0 and res.row('RD3')['source'] == 'missing'
    assert res.row('R')['steady_w'] == 2.2 and res.row('R')['source'] == 'module'
    notes = [f for f in res.findings if f['kind'] == 'no_figure']
    assert [f['tags'] for f in notes] == [['RD3'], ['RD4'], ['RD5']] and all(f['severity'] == 'note' for f in notes)
    assert res.totals['steady_w'] == pytest.approx(5 * 2.2 + OS1_W + LONG_W + INS_W + 2 * ANT_W + FIXED_W)
    # other loads: their own peak and circuit, in the enclosure when ticked
    st = _short(p, other=[['Fan', 6, 9, 'recorder', True], ['Laptop', 45, 30, 'nonsense']])
    res = PW.run(p, settings=st)
    fan, lap = res.row('Fan'), res.row('Laptop')
    assert (fan['peak_w'], fan['circuit'], fan['enclosure']) == (9.0, 'recorder', True)
    assert (lap['peak_w'], lap['circuit'], lap['enclosure']) == (45.0, 'other', False)   # peak never under steady
    assert 'Fan' in res.totals['heat_tags'] and 'Fan' in res.circuit('recorder')['tags']
    # an empty variant with every fixed load off: nothing draws power, nothing breaks
    q = copy.deepcopy(base)
    q.variant.cameras.clear()
    q.analysis.power = {'compute_w': 0, 'recorder_w': 0, 'switch_w': 0, 'ptp_w': 0}
    res = PW.run(q)
    assert res.rows == [] and res.circuits == [] and res.totals['steady_w'] == 0.0
    assert math.isinf(res.totals['runtime_min']) and res.worst() == 'good' and PW.issues(q) == []
    doc = PW.to_doc(res, q)
    assert doc['totals']['engine_off_runtime_min'] is None and doc['devices'] == []
    json.dumps(doc)
    # another variant than the active one
    four = next(v for v in p.variants if v.name == '4-cam')
    res = PW.run(p, variant=four)
    assert res.variant == '4-cam' and [r['tag'] for r in res.rows[:4]] == [c.tag for c in four.cameras]


def test_settings_are_validated():
    d = PW.PowerSettings().to_dict()
    assert PW.PowerSettings.from_dict(d) == PW.PowerSettings()
    assert PW.PowerSettings.from_dict(None).to_dict() == d and PW.PowerSettings.from_dict('junk').to_dict() == d
    st = PW.PowerSettings.from_dict({
        'bus_v': 13.8, 'camera_w': 'lots', 'lidar_peak': 0.2, 'dcdc_eff': 1.5, 'dcdc_w': -3,
        'battery_dod': float('nan'),
        'alternator_a': '60', 'cable_awg': 40, 'max_drop_pct': 0, 'min_input_v': float('inf'), 'nonsense': 1,
        'other': [['Fan', '6', 'x', 'compute'], 'junk', [None, -5, 1e9, 'cameras', 'yes'], []] + [['L', 1, 1]] * 60,
        'fuses': {'compute': '10', 'lidar': -1, 'bogus': 5, 'radar': float('nan')},
        'circuit_of': {'abc': 'lidar', 'def': 'nowhere', 3: 'radar'},
        'cables': {'u1': {'m': -2, 'awg': 'x'}, 'u2': {'m': 'far'}, 'u3': 'junk', 'u4': {'awg': 2}},
        'enclosure': ['compute', 3, 'compute', 'u1'], 'dcdc_in_enclosure': 'no'})
    assert st.bus_v == 12.0 and st.camera_w == 1.5 and st.lidar_peak == 1.0 and st.dcdc_eff == 1.0
    assert st.dcdc_w == PW.LIMITS['dcdc_w'][0] and st.battery_dod == 0.8 and st.alternator_a == 60.0
    assert st.cable_awg == PW.AWG_RANGE[1] and st.max_drop_pct == 0.1 and st.min_input_v == 9.0
    assert 'nonsense' not in st.to_dict() and len(st.other) == PW.MAX_OTHER
    assert st.other[0] == ['Fan', 6.0, 6.0, 'compute', False]
    assert st.other[1] == ['None', 0.0, PW.LOAD_W[1], 'cameras', True]
    assert st.fuses == {'compute': 10.0} and st.circuit_of == {'abc': 'lidar'}
    assert st.cables == {'u1': {'m': 0.0}, 'u4': {'awg': PW.AWG_RANGE[0]}}
    assert st.enclosure == ['compute', 'u1'] and st.dcdc_in_enclosure is False
    assert PW.PowerSettings.from_dict({'bus_v': 30}).bus_v == 24.0
    assert PW.PowerSettings.from_dict({'bus_v': 99}).bus_v == 48.0
    assert PW.PowerSettings.from_dict({'enclosure': []}).enclosure == []
    # the project's own settings are used when none are given
    q = samples.tigor_sample_project()
    q.analysis.power = {'compute_w': 30, 'bus_v': 24}
    res = PW.run(q)
    assert res.settings.compute_w == 30.0 and res.row('Compute')['steady_a'] == pytest.approx(30 / 24)


def test_issues_are_cheap(p):
    PW.issues(p)
    t0 = time.perf_counter()
    for _ in range(200):
        PW.issues(p)
    assert (time.perf_counter() - t0) / 200 < 0.005


# ----------------------------------------------------------------------------- output
def test_summary_doc_csv_round_trip(p, tmp_path):
    p.analysis.power = {'fuses': {'lidar': 3}}
    res = PW.run(p)
    s = PW.summary(res)
    json.dumps(s)
    assert s['steady_w'] == round(STEADY, 1) and s['dcdc_rating_w'] == 600.0 and len(s['per_device']) == 19
    assert s['battery_drain_while_driving_min'] is None
    assert s['engine_off_runtime_min'] == round(res.totals['runtime_min'], 1)
    assert [f['severity'] for f in s['findings']][:1] == ['warning'] and 'assumptions' in s
    paths = PW.export(res, p, str(tmp_path / 'out'))
    assert [x.rsplit('/', 1)[-1] for x in paths] == ['power_budget.json', 'power_budget.csv', 'power_circuits.csv']
    assert [n for n, _ in PW.files(res, p)] == ['power_budget.json', 'power_budget.csv', 'power_circuits.csv']
    doc = json.loads((tmp_path / 'out' / 'power_budget.json').read_text())
    assert doc['schema'] == PW.SCHEMA == 'vantage.power_budget/1'
    assert doc['project'] == p.name and doc['variant'] == p.variant.name and doc['units']['energy'] == 'Wh'
    assert doc['settings']['fuses'] == {'lidar': 3.0} and doc['assumptions'] == PW.ASSUMPTIONS
    assert [d['tag'] for d in doc['devices']] == [r['tag'] for r in res.rows]
    for d, r in zip(doc['devices'], res.rows):
        assert d['steady_w'] == pytest.approx(r['steady_w'], abs=1e-3) and d['circuit'] == r['circuit']
        assert d['steady_a'] == pytest.approx(r['steady_a'], abs=1e-3)
        if r['cable_m'] is None:
            assert d['cable_m'] is None and d['device_v_peak'] is None
        else:
            assert d['drop_pct'] == pytest.approx(r['drop_pct'], abs=0.01) and d['awg'] == 18
    lid = next(c for c in doc['circuits'] if c['circuit'] == 'lidar')
    assert (lid['fuse_a'], lid['fuse_auto'], lid['status'], lid['devices']) == (3.0, False, 'warning', ['L1'])
    assert doc['totals']['dcdc_input_w'] == pytest.approx(res.totals['dcdc_input_w'], abs=1e-3)
    assert doc['totals']['battery_drain_while_driving_min'] is None
    assert doc['findings'][0]['severity'] == 'warning' and doc['findings'][0]['objects'] == 'Lidars circuit'
    with open(tmp_path / 'out' / 'power_budget.csv', newline='') as fh:
        rows = list(csv.DictReader(fh))
    assert [r['tag'] for r in rows] == [r['tag'] for r in res.rows] + ['total']
    for row, r in zip(rows, res.rows):
        assert float(row['steady_w']) == pytest.approx(r['steady_w'], abs=1e-3)
        assert float(row['peak_a']) == pytest.approx(r['peak_a'], abs=1e-3) and row['circuit'] == r['circuit']
        assert row['in_enclosure'] == str(r['enclosure'])
        assert (row['cable_m'] == '') == (r['cable_m'] is None)
    assert float(rows[-1]['steady_w']) == pytest.approx(STEADY) and float(rows[-1]['steady_a']) == pytest.approx(
        res.totals['bus_a'], abs=1e-3)
    with open(tmp_path / 'out' / 'power_circuits.csv', newline='') as fh:
        circs = list(csv.DictReader(fh))
    assert [c['circuit'] for c in circs] == [c['circuit'] for c in res.circuits]
    assert circs[0]['devices'] == 'F R ML MR QL QR Compute' and circs[1]['status'] == 'warning'
    assert float(circs[1]['peak_load_pct']) == pytest.approx(res.circuit('lidar')['peak_load'] * 100, abs=0.01)


# ----------------------------------------------------------------------------- the page, driven like a user
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


def _row_of(table, text, col=0):
    return next(i for i in range(table.rowCount()) if table.item(i, col) and table.item(i, col).text() == text)


def test_page_edits_undoes_adds_loads_and_exports(app, win, tmp_path):
    from PyQt6.QtCore import Qt
    from vantage.ui.workspaces.power_ws import PowerPage, C_CABLE, C_CIRCUIT, C_DROP, C_ENCL, C_FUSE
    store = win.store
    rig = next(v for v in store.project.variants if v.sensors)
    store.edit('variants', lambda q: setattr(q, 'active_uid', rig.uid), 'Switched to the data rig')

    def add(q):
        _add(q, 'imu', 'ins_dual', 'IMU1')
        _add(q, 'gnss', 'gnss_survey', 'GN1')
        _add(q, 'gnss', 'gnss_survey', 'GN2')
    store.edit('variant', add, 'Added the INS and antennas')
    pg = _open(win, 'power', PowerPage)
    pump(app, 0.2)
    assert win.page_name() == 'power' and pg.isVisible()
    res = pg.result
    assert res.variant == rig.name and len(res.rows) == 19
    assert pg.table.rowCount() == len(res.rows) + 1 and pg.table.item(6, 0).text() == 'L1'
    assert pg.tiles['load'].value == f'{res.totals["steady_w"]:,.0f}'
    assert pg.tiles['dcdc'].sev == 'good' and pg.tiles['alternator'].sev == 'good' and pg.tiles['circuit'].sev == 'good'
    assert pg.circ.rowCount() == len(res.circuits) and len(pg.findings) == len(res.findings)
    assert len(pg.bar.segs) == len(res.rows) + 2                      # every device, the cables, the converter
    steady = res.totals['steady_w']
    # a bigger compute module, typed into its field: saved, computed again at once, undoable
    _type_into_field(app, pg.fields['compute_w'][0], '400')
    assert store.project.analysis.power['compute_w'] == 400
    assert pg.result.totals['steady_w'] == pytest.approx(steady + 340)
    assert pg.tiles['load'].value == f'{steady + 340:,.0f}' and pg.tiles['circuit'].sev == 'good'
    # nonsense is clamped; the field shows what is stored
    _type_into_field(app, pg.fields['dcdc_eff'][0], '20')
    assert store.project.analysis.power['dcdc_eff'] == 0.5 and pg.fields['dcdc_eff'][0].value() == 50.0
    # a 48 V bus: every current falls to a quarter
    pg.bus.buttons['48'].click()
    pump(app, 0.1)
    assert pg.result.settings.bus_v == 48.0 and pg.result.row('Compute')['steady_a'] == pytest.approx(400 / 48)
    store.undo.undo()
    store.undo.undo()
    store.undo.undo()
    pump(app, 0.1)
    assert store.project.analysis.power == {} and pg.fields['compute_w'][0].value() == 60.0
    assert pg.result.totals['steady_w'] == pytest.approx(steady) and pg.bus.value() == '12'
    # a 5 A fuse on the compute circuit, typed into the circuits table: critical
    i = _row_of(pg.circ, 'Compute')
    assert pg.circ.item(i, C_FUSE).text() == '7.5 auto'
    pg.circ.item(i, C_FUSE).setText('5')
    pump(app, 0.1)
    assert store.project.analysis.power['fuses'] == {'compute': 5.0}
    assert pg.tiles['circuit'].sev == 'critical' and pg.result.worst() == 'critical'
    assert pg.circ.item(_row_of(pg.circ, 'Compute'), C_FUSE).text() == '5'
    # the lidar's cable set to 10 m, its circuit moved, a camera inside the enclosure
    j = _row_of(pg.table, 'L1')
    pg.table.item(j, C_CABLE).setText('10')
    pump(app, 0.1)
    uid = _sensor(rig, 'L1').uid
    assert store.project.analysis.power['cables'] == {uid: {'m': 10.0}} and pg.result.row('L1')['cable_set']
    assert pg.table.item(j, C_DROP).text() == f'{pg.result.row("L1")["drop_pct"]:.1f} %'
    cb = pg.table.cellWidget(j, C_CIRCUIT)
    cb.setCurrentIndex(cb.findData('network'))
    pump(app, 0.1)
    assert store.project.analysis.power['circuit_of'] == {uid: 'network'}
    assert pg.result.row('L1')['circuit'] == 'network'
    k = _row_of(pg.table, 'F')
    pg.table.item(k, C_ENCL).setCheckState(Qt.CheckState.Checked)
    pump(app, 0.1)
    assert rig.by_tag('F').uid in store.project.analysis.power['enclosure'] and 'F' in pg.result.totals['heat_tags']
    # the fuse cleared: sized automatically again
    pg.circ.item(_row_of(pg.circ, 'Compute'), C_FUSE).setText('')
    pump(app, 0.1)
    assert store.project.analysis.power['fuses'] == {} and pg.tiles['circuit'].sev == 'good'
    assert pg.circ.item(_row_of(pg.circ, 'Compute'), C_FUSE).text() == '7.5 auto'
    for _ in range(5):
        store.undo.undo()
    pump(app, 0.1)
    assert store.project.analysis.power == {} and pg.circ.item(i, C_FUSE).text() == '7.5 auto'
    # other loads: add one, edit its power, remove it
    pg.b_add.click()
    pump(app, 0.1)
    assert store.project.analysis.power['other'] == [['Load 1', 10.0, 12.0, 'other', False]]
    assert pg.other.rowCount() == 1 and pg.result.row('Load 1')['steady_w'] == 10.0
    pg.other.item(0, 1).setText('25')
    pump(app, 0.1)
    assert store.project.analysis.power['other'][0][1:3] == [25.0, 25.0]
    assert pg.result.totals['steady_w'] == pytest.approx(steady + 25)
    pg.other.item(0, 0).setText('Fan tray')
    pump(app, 0.1)
    assert pg.result.row('Fan tray') is not None and pg.table.item(len(pg.result.rows) - 1, 0).text() == 'Fan tray'
    pg.other.cellWidget(0, 3).setCurrentIndex(pg.other.cellWidget(0, 3).findData('recorder'))
    pump(app, 0.1)
    assert pg.result.row('Fan tray')['circuit'] == 'recorder'
    pg.other.selectRow(0)
    pg.b_remove.click()
    pump(app, 0.1)
    assert store.project.analysis.power['other'] == [] and pg.other.rowCount() == 0
    assert pg.result.totals['steady_w'] == pytest.approx(steady)
    store.undo.undo()
    pump(app, 0.1)
    assert pg.result.row('Fan tray') is not None and pg.other.rowCount() == 1
    # export
    paths = pg.export(str(tmp_path))
    assert len(paths) == 3 and json.loads((tmp_path / 'power_budget.json').read_text())['schema'] == PW.SCHEMA
    # leave the page: edits are no longer computed until it shows again
    win.set_page('viewports')
    pump(app, 0.2)
    assert not pg.isVisible()
    shown = pg.result
    store.edit('analysis', lambda q: setattr(q.analysis, 'power', {'battery_wh': 100.0}), 'test')
    pump(app, 0.1)
    assert pg.result is shown and pg._dirty
    _open(win, 'power', PowerPage)
    pump(app, 0.2)
    assert pg.result is not shown and pg.tiles['runtime'].sev == 'warning'
    assert pg.fields['battery_wh'][0].value() == 100.0
    # the project replaced while the page shows: it follows
    store.set_project(samples.tigor_sample_project())
    pump(app, 0.2)
    assert pg.result.variant == store.project.variant.name and store.project.analysis.power == {}
    assert pg.table.rowCount() == len(pg.result.rows) + 1
