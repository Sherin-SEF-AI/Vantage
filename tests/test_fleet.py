"""Fleet: the same rig on many vehicles. The maths against known values (deltas, rotation angles, trends, crossings,
robust z-scores), a synthetic fleet whose drifting camera, outlier vehicle, jump and systematic offset are found and
nothing else, every import round-tripped through the real exporters (extrinsics in mm and the front-bumper frame,
nuScenes, Kalibr) and the installation check's report from batch.install_check on synthetic captures, the editing
helpers, save/load and undo, the outputs, and the page driven like a user."""
import copy
import datetime as dt
import json
import math
import os
import re

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
import pytest

from vantage.core import exporters as ex
from vantage.core import fleet as FL
from vantage.core import samples
from vantage.core import sensors as SN
from vantage.core.mathutil import euler_zyx, rot_zyx
from vantage.core.project import Project, Sensor

START = dt.date(2026, 3, 2)
DAYS = (0, 60, 120, 180)              # four records over six months
DRIFT = ('V2', 'F', 4, 0.2 / 30.0)    # V2's front camera pitches down 0.2° a month
OUTLIER = ('V4', 'ML', 3, 2.0)        # V4's left mirror camera is yawed 2° on every record


def rig_project(extra=True):
    """The sample's data rig (6 cameras, a lidar, 5 radars), active, with a GNSS antenna and an IMU added."""
    p = samples.tigor_sample_project()
    v = next(x for x in p.variants if x.sensors)
    p.active_uid = v.uid
    if extra:
        for kind, mid in (('gnss', 'gnss_survey'), ('imu', 'imu_mems')):
            m = p.sensor_models[mid]
            pose = SN.default_pose(kind, m, p.vehicle.mesh, p.rig, v.sensors)
            v.sensors.append(Sensor(kind=kind, model_id=mid, tag=SN.next_tag(kind, v.tags()), name=kind, **pose))
    return p


def plus(pose, d):
    """A pose moved by d = [dx, dy, dz mm, dyaw, dpitch, droll °]."""
    return {'x': pose['x'] + d[0] / 1000.0, 'y': pose['y'] + d[1] / 1000.0, 'z': pose['z'] + d[2] / 1000.0,
            'yaw': pose['yaw'] + d[3], 'pitch': pose['pitch'] + d[4], 'roll': pose['roll'] + d[5]}


def synth_fleet(p, n=5, days=DAYS, special=(DRIFT, OUTLIER), extra=None, noise=(0.3, 0.01), seed=7):
    """n vehicles × len(days) records with known per-vehicle offsets (within ±3 mm / ±0.15°) and small measurement
    noise; special: (vehicle, tag, component, value): DRIFT's value is a rate per day, any other a constant offset;
    extra(vehicle name, tag, record index) -> an extra delta."""
    rng = np.random.default_rng(seed)
    plan = FL.plan_sensors(p, p.variant)
    fleet = {}
    offsets = {}
    for i in range(n):
        name = f'V{i + 1}'
        v = FL.add_vehicle(fleet, name, vin=f'MAT{i:05d}')
        off = {ps['tag']: np.r_[rng.uniform(-3, 3, 3), rng.uniform(-0.15, 0.15, 3)] for ps in plan}
        offsets[name] = off
        for j, day in enumerate(days):
            poses = {}
            for ps in plan:
                d = off[ps['tag']] + np.r_[rng.normal(0, noise[0], 3), rng.normal(0, noise[1], 3)]
                for sp in special:
                    vn, tag, comp, val = sp
                    if vn == name and tag == ps['tag']:
                        d[comp] += val * day if sp == DRIFT else val
                if extra is not None:
                    d = d + np.asarray(extra(name, ps['tag'], j), float)
                poses[ps['tag']] = plus(ps['pose'], d)
            FL.add_record(fleet, v['id'], {'kind': 'manual', 'date': (START + dt.timedelta(days=day)).isoformat(),
                                           'source': f'survey {j + 1}', 'poses': poses})
    return fleet, offsets


def vid_of(res, name):
    return next(v['id'] for v in res.vehicles if v['name'] == name)


def names(res, keys):
    n = {v['id']: v['name'] for v in res.vehicles}
    return {(n[v], t) for v, t in keys}


# ----------------------------------------------------------------------------- geometry against known values
def test_rotations_deltas_and_verdicts_known_values():
    rng = np.random.default_rng(0)
    ypr = rng.uniform(-80, 80, (20, 3))
    M = FL._rots(ypr)
    for a, R in zip(ypr, M):
        assert np.allclose(R, rot_zyx(*a), atol=1e-12)
    # the angle between two rotations: a pure 2.5° turn about any axis, exact at tiny angles too
    for ang in (2.5, 1e-4, 179.0):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        t = math.radians(ang)
        Rr = np.eye(3) + math.sin(t) * K + (1 - math.cos(t)) * K @ K
        assert float(FL._angle_between(M[0], M[0] @ Rr)) == pytest.approx(ang, rel=1e-6, abs=1e-9)
    plan = {'x': 1.0, 'y': 0.5, 'z': 1.2, 'yaw': 179.5, 'pitch': 20.0, 'roll': -179.0}
    meas = {'x': 1.004, 'y': 0.497, 'z': 1.2, 'yaw': -179.5, 'pitch': 20.3, 'roll': 179.5}
    d = FL.delta(plan, meas)
    assert np.allclose(d, [4.0, -3.0, 0.0, 1.0, 0.3, -1.5], atol=1e-9)          # yaw and roll wrapped
    st = FL.FleetSettings()
    # a yaw-only offset rotates by exactly that much; the ratio is the worse of the two parts
    assert float(FL.rotation_deg(plan, [0, 0, 0, 0.8, 0, 0])) == pytest.approx(0.8, abs=1e-9)
    assert float(FL.ratio_of([10, 0, 0, 0, 0, 0], 0.0, st)) == pytest.approx(1.0)
    assert FL.verdict_of(1.0) == 'ok' and FL.verdict_of(1.01) == 'adjust' and FL.verdict_of(3.0) == 'adjust'
    assert FL.verdict_of(3.01) == 're-mount' and FL.verdict_of(None) == 'missing'
    assert float(FL.ratio_of([2, -25, 0, 0, 0, 0], 1.2, st)) == pytest.approx(2.5)
    # words: yaw positive left, pitch positive down, z positive up
    assert FL.describe('yaw', 2.1, 'the fleet median') == 'is yawed 2.1° left of the fleet median'
    assert FL.describe('pitch', -1.0, 'the plan') == 'is pitched 1.0° up from the plan'
    assert FL.describe('z', 25.0, 'the plan') == 'sits 25 mm above the plan'


def test_dates():
    assert FL.normalise_date('2026-05-01') == '2026-05-01'
    assert FL.normalise_date('2026-05-01T14:30:00+02:00') == '2026-05-01T12:30:00'      # to UTC
    assert FL.normalise_date('01.05.2026') == '2026-05-01'
    assert FL.date_days('2026-05-02') - FL.date_days('2026-05-01') == 1.0
    assert FL.date_days('2026-05-01T12:00:00') - FL.date_days('2026-05-01') == pytest.approx(0.5)
    for junk in ('', 'soon', '2026-13-01', None, 42):
        assert FL.date_days(junk) is None
    with pytest.raises(ValueError, match='not a date'):
        FL.normalise_date('next tuesday')


def test_trend_crossing_and_jump_known_values():
    """A perfectly linear rise of Δz, 3 mm per 30 days from 0: the trend is 3 mm / 30 days, 9 mm after 90 days, and
    the 10 mm tolerance is crossed 10 days after the last record."""
    plan = {'x': 1.0, 'y': 0.0, 'z': 1.5, 'yaw': 0.0, 'pitch': 10.0, 'roll': 0.0}
    st = FL.FleetSettings(drift_window_days=30)
    recs = [{'id': f'r{i}', 'date': (START + dt.timedelta(days=30 * i)).isoformat(), 'source': 's', 'kind': 'manual',
             'poses': {'L1': plus(plan, [0, 0, 3.0 * i, 0, 0, 0])}, 'quality': {}} for i in range(4)]
    series = [(FL.date_days(r['date']), r, 'L1') for r in recs]
    d = FL._drift(plan, series, st)
    assert d['trend_per_30d'][2] == pytest.approx(3.0, abs=1e-9) and d['change'][2] == pytest.approx(9.0)
    assert d['significant'] == [False, False, True, False, False, False]
    assert d['days_to_tol'] == pytest.approx(10.0, abs=0.01)
    assert d['trend'] and not d['jump'] and d['rate_ratio'] == pytest.approx(0.3)
    assert FL._drift(plan, series, FL.FleetSettings(drift_window_days=5))['trend'] is False   # 9.5 mm in 5 days
    # the same rise in two records is not yet a trend (three dated records are needed), and 3 mm is no jump
    two = FL._drift(plan, series[:2], st)
    assert not two['trend'] and not two['jump'] and two['days_to_tol'] is None
    # a 6 mm knock on top of the course in the last record: a jump of 0.6 of the tolerance
    recs[3]['poses']['L1'] = plus(plan, [0, 0, 15.0, 0, 0, 0])
    j = FL._drift(plan, series, st)
    assert j['jump'] and j['jump_ratio'] == pytest.approx(0.6, abs=1e-6) and j['step_ratio'] == pytest.approx(0.9)


def test_t_table_against_the_distribution():
    stats = pytest.importorskip('scipy.stats')
    for dof in list(range(1, 61)) + [75, 90, 120, 200, 500, 5000]:
        assert FL._t_crit(dof) == pytest.approx(float(stats.t.ppf(0.975, dof)), abs=1e-3), dof


def test_robust_z_and_outlier_against_hand_computed():
    st = FL.FleetSettings()
    ps = {'tag': 'F', 'kind': 'camera', 'uid': 'u', 'pose': {'x': 0, 'y': 0, 'z': 1, 'yaw': 0, 'pitch': 0, 'roll': 0}}
    yaws = [0.10, -0.20, 0.05, 2.30, 0.00]
    cells = [{'vehicle': f'v{i}', 'd': [0, 0, 0, y, 0, 0], 'ratio': abs(y), 'drift': None, 'outlier': None}
             for i, y in enumerate(yaws)]
    f = FL._consistency(ps, cells, {f'v{i}': f'V{i + 1}' for i in range(5)}, st)
    med = float(np.median(yaws))
    sigma = 1.4826 * float(np.median(np.abs(np.array(yaws) - med)))
    spread = max(sigma, FL.SIGMA_FLOOR * st.ang_tol_deg)
    assert f['median'][3] == pytest.approx(med) and f['sigma'][3] == pytest.approx(sigma)
    assert [o['vehicle'] for o in f['outliers']] == ['v3']
    assert f['outliers'][0]['z'] == pytest.approx((2.30 - med) / spread) and f['outliers'][0]['component'] == 'yaw'
    assert not f['systematic']


# ----------------------------------------------------------------------------- the synthetic fleet
def test_synthetic_fleet_finds_exactly_the_drift_and_the_outlier():
    p = rig_project()
    p.fleet, _ = synth_fleet(p)
    res = FL.run(p)
    assert len(res.vehicles) == 5 and len(res.plan) == 14 and len(res.cells) == 70
    assert names(res, res.flagged('drift')) == {('V2', 'F')}
    assert res.flagged('jump') == []
    plain = {(o['name'], f['tag']) for f in res.fleet for o in f['outliers'] if o['cause'] is None}
    assert plain == {('V4', 'ML')}
    assert names(res, res.flagged('adjust')) == {('V2', 'F'), ('V4', 'ML')} and res.flagged('re-mount') == []
    assert not any(f['systematic'] for f in res.fleet)
    c = res.cell(vid_of(res, 'V2'), 'F')
    assert c['drift']['trend_per_30d'][4] == pytest.approx(0.2, abs=0.01)          # pitch, ° per 30 days
    tr = np.abs(c['drift']['trend_per_30d'])
    assert c['drift']['significant'][4] and np.argmax(tr / [10, 10, 10, 1, 1, 1]) == 4
    assert c['records'] == 4 and c['date'] == (START + dt.timedelta(days=180)).isoformat()
    o = res.cell(vid_of(res, 'V4'), 'ML')['outlier']
    assert o['component'] == 'yaw' and o['from_median'] == pytest.approx(2.0, abs=0.35) and o['z'] > 3.5
    kinds = [f['kind'] for f in res.findings if f['severity'] != 'good']
    assert sorted(kinds) == ['adjust', 'adjust', 'drift', 'outlier']
    msg = next(f['message'] for f in res.findings if f['kind'] == 'outlier')
    assert msg.startswith('V4: ML is yawed 2.') and 'left of the fleet median' in msg
    assert 'V2: F drifts +0.2' in next(f['message'] for f in res.findings if f['kind'] == 'drift')
    # earlier, after the first three records (within tolerance yet), the drift is seen coming with its date
    p.fleet = synth_fleet(p, days=DAYS[:3])[0]
    early = FL.run(p, settings=FL.FleetSettings(drift_window_days=90))
    c = early.cell(vid_of(early, 'V2'), 'F')
    assert c['verdict'] == 'ok' and c['drift']['trend']
    d, st = c['drift'], early.settings
    plan = next(ps for ps in early.plan if ps['tag'] == 'F')['pose']
    slope = np.where(d['significant'], np.array(d['trend_per_30d']) / 30.0, 0.0)
    ts = np.arange(0.0, 90.0, 0.005)                                 # brute force against the bisection
    D = np.array(d['series'][-1]['d']) + ts[:, None] * slope
    ang = [float(np.degrees(np.arccos(np.clip((np.trace(rot_zyx(plan['yaw'], plan['pitch'], plan['roll']).T @ rot_zyx(
        plan['yaw'] + x[3], plan['pitch'] + x[4], plan['roll'] + x[5])) - 1) / 2, -1, 1)))) for x in D[::20]]
    first = int(np.argmax(np.maximum(np.abs(D[::20, :3]).max(axis=1) / 10.0, np.array(ang) / 1.0) > 1.0))
    assert 0 < d['days_to_tol'] < 90 and d['days_to_tol'] == pytest.approx(ts[::20][first], abs=0.11)
    assert names(early, early.flagged('drift')) == {('V2', 'F')}


def test_a_knocked_sensor_is_a_jump():
    p = rig_project()
    p.fleet, _ = synth_fleet(p, special=(DRIFT,), extra=lambda v, t, j: [0, 0, 0, 1.5, 0, 0]
                             if (v, t, j) == ('V3', 'RD2', 3) else [0] * 6)
    res = FL.run(p)
    assert names(res, res.flagged('jump')) == {('V3', 'RD2')}
    assert names(res, res.flagged('drift')) == {('V2', 'F')}
    d = res.cell(vid_of(res, 'V3'), 'RD2')['drift']
    assert d['jump_ratio'] == pytest.approx(1.5, abs=0.15) and not d['trend']
    o = res.cell(vid_of(res, 'V3'), 'RD2')['outlier']
    assert o is not None and o['cause'] == 'jump'                     # named once, as the jump
    assert sum(f['kind'] == 'outlier' for f in res.findings) == 0
    assert 'V3: RD2 moved +1.5' in next(f['message'] for f in res.findings if f['kind'] == 'jump')


def test_systematic_offset_and_the_fleet_median_variant():
    """Every vehicle's lidar mast is built 25 mm taller than planned: the plan, not the vehicles, is off."""
    p = rig_project()
    p.fleet, offsets = synth_fleet(p, special=(), extra=lambda v, t, j: [0, 0, 25.0, 0, 0, 0] if t == 'L1' else [0] * 6)
    plan = p.variant
    res = FL.run(p)
    f = res.sensor('L1')
    assert f['systematic'] and f['near_median'] == 5 and f['median'][2] == pytest.approx(25.0, abs=3.5)
    assert sum(x['systematic'] for x in res.fleet) == 1 and res.sensor('L1')['outliers'] == []
    assert any(x['kind'] == 'systematic' and x['message'].startswith('L1 sits 2') for x in res.findings)
    assert len(res.flagged('adjust')) == 5
    before = plan.to_dict()
    n = len(p.variants)
    v = FL.median_variant(p, res)
    assert len(p.variants) == n + 1 and p.active_uid == v.uid and v.name.endswith('fleet median')
    assert plan.to_dict() == before                                     # the plan itself is untouched
    L1 = next(s for s in v.sensors if s.tag == 'L1')
    planned = next(s for s in plan.sensors if s.tag == 'L1')
    assert L1.z == pytest.approx(planned.z + f['median'][2] / 1000, abs=1e-4)
    assert {c.uid for c in v.cameras}.isdisjoint({c.uid for c in plan.cameras})
    assert {s.uid for s in v.sensors}.isdisjoint({s.uid for s in plan.sensors})
    assert all(c.link_uid == '' for c in v.cameras)                     # measured poses are not mirror images
    # against the new plan the fleet is within tolerance and nothing is systematic
    again = FL.run(p)
    assert again.variant_uid == v.uid and not any(x['systematic'] for x in again.fleet)
    assert again.flagged('adjust') == [] and again.flagged('re-mount') == []
    # the chosen plan variant stays the plan however the active one changes
    p.fleet['settings'] = {'variant': plan.uid}
    assert FL.run(p).variant_uid == plan.uid


def test_tags_matched_ignoring_case_unknown_hidden_and_missing_reported():
    p = rig_project()
    plan = {ps['tag']: ps['pose'] for ps in FL.plan_sensors(p, p.variant)}
    p.variant.by_tag('QR').visible = False
    fl = {}
    a = FL.add_vehicle(fl, 'V1')
    b = FL.add_vehicle(fl, 'V2')
    FL.add_record(fl, a['id'], {'date': '2026-04-01', 'kind': 'manual', 'poses': {
        'f': plan['F'], 'F': plus(plan['F'], [50, 0, 0, 0, 0, 0]), 'X9': plan['F'], 'qr': plan['QR'],
        'l1': plus(plan['L1'], [0, 0, 40, 0, 0, 0])}})
    FL.add_record(fl, b['id'], {'date': '2026-04-02', 'kind': 'manual', 'poses': {'F': plan['F'], 'R': plan['R']}})
    FL.add_vehicle(fl, 'V3')                                           # no records yet
    p.fleet = fl
    res = FL.run(p)
    va, vb = res.vehicle(a['id']), res.vehicle(b['id'])
    c = res.cell(a['id'], 'F')
    assert c['record_tag'] == 'F' and c['verdict'] == 're-mount'        # the exact case wins; 'f' is a duplicate
    assert [u['tag'] for u in va['unknown']] == ['X9', 'f'] and [h['tag'] for h in va['hidden']] == ['qr']
    assert res.cell(a['id'], 'L1')['record_tag'] == 'l1' and res.cell(a['id'], 'L1')['verdict'] == 're-mount'
    assert 'QR' not in [ps['tag'] for ps in res.plan]
    assert 'R' in va['missing'] and 'L1' in vb['missing']
    kinds = {(f['kind'], f['vehicle']) for f in res.findings}
    assert ('unknown', a['id']) in kinds and ('hidden', a['id']) in kinds and ('remount', a['id']) in kinds
    assert ('missing', a['id']) in kinds and ('missing', b['id']) in kinds        # each measured elsewhere
    assert any(f['kind'] == 'no_records' and f['message'] == 'V3 has no records yet.' for f in res.findings)
    assert any(f['kind'] == 'never' and 'RD1' in f['tags'] for f in res.findings)
    rep = FL.import_report(fl['vehicles'][0]['records'][0], p)
    assert rep['unknown'] == ['X9'] and rep['hidden'] == ['qr'] and rep['duplicates'] == ['f']
    assert 'not in the plan: X9' in rep['text']


def test_tolerant_of_junk_and_empty_cases():
    p = rig_project(extra=False)
    assert FL.run(p).findings[0]['kind'] == 'empty' and FL.issues(p) == []
    st = FL.FleetSettings.from_dict({'pos_tol_mm': 'x', 'ang_tol_deg': -4, 'drift_window_days': float('nan'),
                                     'outlier_z': 1e9, 'variant': 7})
    assert (st.pos_tol_mm, st.ang_tol_deg, st.drift_window_days, st.outlier_z, st.variant) == \
        (10.0, 0.01, 30.0, 50.0, '')
    plan = FL.plan_sensors(p, p.variant)[0]
    p.fleet = {'vehicles': [None, 'junk', {'name': '', 'records': [
        None, {'date': 'when?', 'poses': {'F': dict(plan['pose'], yaw=float('nan')), 'R': {'x': 1}}},
        {'date': '2026-01-01', 'kind': 'nonsense', 'poses': {plan['tag']: plan['pose']},
         'quality': {'F': {'rms_px': 0.2, 'x': [1]}}}]}],
               'settings': 'bad'}
    vs = FL.vehicles(p)
    assert len(vs) == 1 and vs[0]['name'] == 'Vehicle' and len(vs[0]['records']) == 2
    assert vs[0]['records'][0]['poses'] == {} and vs[0]['records'][1]['kind'] == 'manual'
    assert vs[0]['records'][1]['quality'] == {'F': {'rms_px': 0.2}}
    assert [v['id'] for v in FL.vehicles(p)] == [v['id'] for v in vs]           # ids are stable when missing
    res = FL.run(p)
    assert res.cell(vs[0]['id'], plan['tag'])['verdict'] == 'ok'
    json.dumps(FL.to_doc(res, p))
    # an empty plan variant
    from vantage.core.project import Variant
    empty = Variant(name='Empty')
    p.variants.append(empty)
    p.active_uid = empty.uid
    res = FL.run(p)
    assert res.plan == [] and res.cells == {} and res.findings[0]['kind'] == 'plan'
    assert FL.issues(p) == []
    FL.png(res)                                                         # draws an empty matrix
    assert FL.run(p, cancel=lambda: True) is None                     # cancelled


# ----------------------------------------------------------------------------- imports, round trips
def _built(p, rng_seed=3):
    """A copy of the project whose variant is 'as built': every camera and sensor moved by a known amount."""
    q = copy.deepcopy(p)
    rng = np.random.default_rng(rng_seed)
    truth = {}
    for o in list(q.variant.cameras) + list(q.variant.sensors):
        d = np.r_[rng.uniform(-15, 15, 3), rng.uniform(-1.5, 1.5, 3)]
        truth[o.tag] = d
        for k, val in plus({k: getattr(o, k) for k in FL.POSE_KEYS}, d).items():
            setattr(o, k, val)
    return q, truth


def _check(rec, p, truth, mm=0.6, deg=2e-3, tags=None):
    plan = {ps['tag']: ps['pose'] for ps in FL.plan_sensors(p, p.variant)}
    want = set(tags if tags is not None else truth)
    assert set(rec['poses']) == want
    for tag in want:
        d = FL.delta(plan[tag], rec['poses'][tag])
        assert np.abs(d[:3] - truth[tag][:3]).max() < mm, (tag, d, truth[tag])
        assert np.abs(d[3:] - truth[tag][3:]).max() < deg, (tag, d, truth[tag])


@pytest.mark.parametrize('frame,units,angles', [('front_bumper', 'mm', 'deg'), ('custom', 'm', 'rad'),
                                                ('rear_axle', 'm', 'deg')])
def test_extrinsics_round_trip(tmp_path, frame, units, angles):
    p = rig_project()
    q, truth = _built(p)
    q.frame, q.units, q.custom_origin = frame, units, [0.75, -0.2, 0.31]       # the project's own frame and units
    out = tmp_path / 'ex'
    paths = ex.export(['extrinsics_json', 'extrinsics_yaml', 'extrinsics_csv'], q, None,
                      ex.ExportOptions(out_dir=str(out), frame=q.frame, units=q.units, angles=angles))
    doc = json.loads((out / 'extrinsics.json').read_text('utf-8'))
    assert doc['frame']['id'] == frame and doc['frame']['units'] == units
    if frame != 'rear_axle':
        assert any(abs(v) > 0.1 for v in doc['frame']['origin_from_rear_axle'])   # the file really is shifted
    for path in paths:
        rec = FL.read_file(path, q)                 # the CSV needs the project for its frame origin
        assert rec['kind'] == 'extrinsics'
        _check(rec, p, truth, mm=0.6 if units == 'mm' else 0.6, deg=2e-3 if angles == 'deg' else 2e-3)
    rec = FL.read_file(str(out / 'extrinsics.json'))  # JSON and YAML carry their origin: no project needed
    _check(rec, p, truth)
    assert FL.normalise_date(doc['exported']) == rec['date']


def test_nuscenes_round_trip(tmp_path):
    p = rig_project()
    q, truth = _built(p, 5)
    q.custom_origin = [1.0, 0.2, 0.3]
    out = tmp_path / 'ns'
    ex.export(['nuscenes_calib'], q, None, ex.ExportOptions(out_dir=str(out), frame='custom'))
    cal = json.loads((out / 'nuscenes/calibrated_sensor.json').read_text('utf-8'))
    tags = [t for t in truth if t not in ('GN1', 'IMU1')]                # not nuScenes sensors (the ego pose is theirs)
    assert len(cal) == len(tags)
    for target in (out / 'nuscenes', out / 'nuscenes/calibrated_sensor.json', out / 'nuscenes/sensor.json', out):
        rec = FL.read_file(str(target), q)
        assert rec['kind'] == 'nuscenes'
        _check(rec, p, truth, mm=0.01, deg=1e-5, tags=tags)
    # without the README the ego origin must be given; otherwise it is taken as the rear axle and said so
    os.remove(out / 'nuscenes/README.md')
    _check(FL.read_nuscenes(str(out / 'nuscenes'), q, origin=[1.0, 0.2, 0.3]), p, truth, mm=0.01, deg=1e-5, tags=tags)
    rec = FL.read_nuscenes(str(out / 'nuscenes'), q)
    assert 'rear axle (assumed)' in rec['note']
    plan = {ps['tag']: ps['pose'] for ps in FL.plan_sensors(p, p.variant)}
    assert FL.delta(plan['F'], rec['poses']['F'])[0] == pytest.approx(truth['F'][0] - 1000.0, abs=0.01)
    # a channel in calibrated_sensor.json whose sensor.json row is missing is skipped and said so
    sen = json.loads((out / 'nuscenes/sensor.json').read_text('utf-8'))
    (out / 'nuscenes/sensor.json').write_text(json.dumps(sen[1:]), 'utf-8')
    rec = FL.read_nuscenes(str(out / 'nuscenes'), q, origin=[1.0, 0.2, 0.3])
    assert len(rec['poses']) == len(tags) - 1 and 'skipped' in rec['note']


def test_kalibr_is_placed_by_the_best_rigid_fit(tmp_path):
    """The camchain only holds camera-to-camera transforms: a whole-rig turn and shift disappears, one camera turned
    against the others shows."""
    p = rig_project()
    q = copy.deepcopy(p)
    G = rot_zyx(3.0, -1.0, 0.5)
    g = np.array([0.04, -0.03, 0.02])
    for c in q.variant.cameras:
        Rb = G @ rot_zyx(c.yaw, c.pitch, c.roll)
        pos = G @ c.pos() + g
        c.x, c.y, c.z = pos
        c.yaw, c.pitch, c.roll = euler_zyx(Rb)
    out = tmp_path / 'k'
    ex.export(['kalibr_camchain'], q, None, ex.ExportOptions(out_dir=str(out)))
    path = out / 'kalibr' / 'camchain.yaml'
    rec = FL.read_file(str(path), p)
    assert rec['kind'] == 'kalibr' and set(rec['poses']) == {c.tag for c in p.variant.cameras}
    _check(rec, p, {c.tag: np.zeros(6) for c in p.variant.cameras}, mm=0.01, deg=1e-5,
           tags=[c.tag for c in p.variant.cameras])
    # one camera turned 2° in yaw against the rest: it alone shows it (less what the fit shares out)
    ml = q.variant.by_tag('ML')
    ml.yaw += 2.0
    ex.export(['kalibr_camchain'], q, None, ex.ExportOptions(out_dir=str(out)))
    rec = FL.read_file(str(path), p)
    plan = {ps['tag']: ps['pose'] for ps in FL.plan_sensors(p, p.variant)}
    rots = {t: float(FL.rotation_deg(plan[t], FL.delta(plan[t], rec['poses'][t]))) for t in rec['poses']}
    assert max(rots, key=rots.get) == 'ML' and 1.4 < rots['ML'] < 2.0
    assert all(r < 0.45 for t, r in rots.items() if t != 'ML')
    with pytest.raises(FL.FleetImportError, match='open the project'):
        FL.read_kalibr({'cam0': {}}, None)


def test_installation_check_report_from_batch(tmp_path):
    """batch.install_check on synthetic captures of two cameras mounted with known errors: the report imports with
    their measured poses, RMS and status, the cameras without an image keep their status and are named."""
    from vantage import batch
    from vantage.core import capture
    from vantage.core.analysis import Engine
    from vantage.core.install import InstallSettings
    from vantage.core.synth import render_capture
    p = samples.tigor_sample_project()                          # its active variant has the floor boards planned
    e = Engine()
    caps = tmp_path / 'caps'
    caps.mkdir()
    errors = {'F': (0.6, -0.7, 0.4, 0.007, -0.006, 0.004), 'ML': (-1.6, 0.5, 0.9, -0.012, 0.01, 0.006)}
    for tag, d in errors.items():
        c = p.variant.by_tag(tag)
        pose = rot_zyx(c.yaw + d[0], c.pitch + d[1], c.roll + d[2]), c.pos() + np.asarray(d[3:])
        capture.save_image(str(caps / f'{tag}.png'), render_capture(p, e, c, 960, 768, pose=pose, seed=11))
    doc, _ = batch.install_check(p, e, None, str(caps), InstallSettings())
    path = tmp_path / 'install-check.json'
    path.write_text(json.dumps(doc, indent=1), 'utf-8')
    rec = FL.read_file(str(path), p)
    assert rec['kind'] == 'install_check' and set(rec['poses']) == {'F', 'ML'}
    assert rec['date'] == FL.normalise_date(doc['time']) and rec['source'] == 'install-check.json'
    truth = {t: np.r_[np.asarray(d[3:]) * 1000, d[:3]] for t, d in errors.items()}
    _check(rec, p, truth, mm=6.0, deg=0.3, tags=['F', 'ML'])
    for t in ('F', 'ML'):
        q = rec['quality'][t]
        assert q['status'] in ('good', 'warning', 'serious') and q['rms_px'] < 1.0 and q['corners'] > 20
        meas = next(r for r in doc['cameras'] if r['tag'] == t)['measured']
        assert rec['poses'][t] == pytest.approx(meas)
    others = {c.tag for c in p.variant.cameras} - {'F', 'ML'}
    assert all(rec['quality'][t]['status'] == 'serious' for t in others)
    assert all(t in rec['note'] for t in others) and 'no pose for' in rec['note']
    # into the fleet: the delta the fleet shows is the one the check measured
    fl = {}
    v = FL.add_vehicle(fl, 'V1')
    FL.add_record(fl, v['id'], rec)
    p.fleet = fl
    res = FL.run(p)
    row = next(r for r in doc['cameras'] if r['tag'] == 'ML')
    c = res.cell(v['id'], 'ML')
    assert np.allclose(c['d'][:3], row['delta_mm'], atol=0.15) and np.allclose(c['d'][3:], row['delta_deg'], atol=2e-3)
    assert c['quality']['rms_px'] == row['rms_px']


def test_read_file_says_why_not(tmp_path):
    p = rig_project()
    cases = {'broken.json': '{"cameras": [', 'other.json': json.dumps({'schema': 'vantage.recording_budget/1'}),
             'nothing.yaml': 'just: words\n', 'empty.csv': '', 'cols.csv': 'a,b\n1,2\n',
             'failed.json': json.dumps({'schema': 'vantage.install_check/1',
                                        'cameras': [{'tag': 'F', 'status': 'serious'}]}),
             'bad_units.json': json.dumps({'schema': 'vantage.extrinsics/1', 'frame': {'units': 'ft'}, 'cameras': []})}
    want = {'broken.json': 'not valid JSON', 'other.json': 'holds no measured poses', 'nothing.yaml': 'not a file',
            'empty.csv': 'empty', 'cols.csv': 'Not an extrinsics CSV', 'failed.json': 'no measured pose',
            'bad_units.json': 'Unknown length unit'}
    for name, text in cases.items():
        (tmp_path / name).write_text(text, 'utf-8')
        with pytest.raises(FL.FleetImportError, match=want[name]):
            FL.read_file(str(tmp_path / name), p)
    with pytest.raises(FL.FleetImportError, match='does not exist'):
        FL.read_file(str(tmp_path / 'nope.json'), p)
    (tmp_path / 'ns').mkdir()
    (tmp_path / 'ns' / 'calibrated_sensor.json').write_text('[]', 'utf-8')
    with pytest.raises(FL.FleetImportError, match='sensor.json is missing'):
        FL.read_file(str(tmp_path / 'ns'), p)
    (tmp_path / 'front.csv').write_text('tag,x_mm,y_mm,z_mm,yaw_deg,pitch_deg,roll_deg,frame\n'
                                        'F,1,2,3,0,0,0,front_bumper\n', 'utf-8')
    with pytest.raises(FL.FleetImportError, match='open the project'):
        FL.read_file(str(tmp_path / 'front.csv'))
    rec = FL.read_file(str(tmp_path / 'front.csv'), p)
    assert rec['poses']['F']['x'] == pytest.approx(3.215 + 0.001)          # the sample's front bumper is 3.215 m out


def test_batch_style_import_into_a_named_vehicle(tmp_path):
    p = rig_project()
    q, truth = _built(p, 9)
    out = tmp_path / 'e'
    ex.export(['extrinsics_json', 'nuscenes_calib'], q, None, ex.ExportOptions(out_dir=str(out)))
    (tmp_path / 'junk.json').write_text('{}', 'utf-8')
    doc = FL.import_files(p, 'Car 12', [str(out / 'extrinsics.json'), str(out / 'nuscenes'), str(tmp_path / 'junk.json')])
    json.dumps(doc)
    assert doc['vehicle'] == 'Car 12' and [d['kind'] for d in doc['imported']] == ['extrinsics', 'nuscenes']
    assert doc['imported'][0]['poses'] == 14 and doc['imported'][0]['unknown'] == []
    assert len(doc['errors']) == 1 and doc['errors'][0]['file'] == 'junk.json'
    again = FL.import_files(p, 'car 12', [str(out / 'extrinsics.json')])             # the same vehicle, any case
    assert again['vehicle'] == 'Car 12' and len(p.fleet['vehicles']) == 1
    assert len(p.fleet['vehicles'][0]['records']) == 3
    res = FL.run(p)
    vid = p.fleet['vehicles'][0]['id']
    c = res.cell(vid, 'F')
    assert np.allclose(c['d'][:3], truth['F'][:3], atol=0.6) and np.allclose(c['d'][3:], truth['F'][3:], atol=2e-3)


# ----------------------------------------------------------------------------- editing, save/load, undo, outputs
def test_editing_helpers_variant_and_manual_records():
    p = rig_project()
    fl = {}
    a = FL.add_vehicle(fl)
    b = FL.add_vehicle(fl, vin=' WVW1 ')
    assert (a['name'], b['name'], b['vin']) == ('V1', 'V2', 'WVW1')
    r = FL.add_record(fl, a['id'], FL.record_from_variant(p, p.variant, '2026-02-01'))
    assert r['kind'] == 'variant' and r['source'] == f'variant: {p.variant.name}' and len(r['poses']) == 14
    m = FL.add_record(fl, a['id'], FL.manual_record('2026-03-01'))
    plan = FL.plan_sensors(p, p.variant)[0]
    assert FL.set_pose(fl, a['id'], m['id'], 'f', plus(plan['pose'], [0, 0, 12, 0, 0, 0]))
    with pytest.raises(ValueError):
        FL.set_pose(fl, a['id'], m['id'], 'F', dict(plan['pose'], z=float('inf')))
    with pytest.raises(ValueError, match='not a date'):
        FL.set_record(fl, a['id'], m['id'], date='31/31/2026')
    assert FL.set_record(fl, a['id'], m['id'], date='2026-03-05T10:00:00')
    assert FL.record(fl, a['id'], m['id'])['date'] == '2026-03-05T10:00:00'
    assert FL.set_vehicle(fl, b['id'], name='  ', note='rig rev B') and FL.vehicle(fl, b['id'])['name'] == 'V3'
    p.fleet = fl
    res = FL.run(p)
    c = res.cell(a['id'], 'F')
    assert c['record'] == m['id'] and c['d'][2] == pytest.approx(12.0) and c['verdict'] == 'adjust'
    assert FL.set_pose(fl, a['id'], m['id'], 'F', None) and FL.record(fl, a['id'], m['id'])['poses'] == {}
    assert FL.remove_record(fl, a['id'], m['id']) and not FL.remove_record(fl, a['id'], m['id'])
    assert FL.remove_vehicle(fl, b['id']) and [v['name'] for v in fl['vehicles']] == ['V1']


def test_save_load_and_undo(tmp_path):
    from vantage.ui.store import ProjectStore
    p = rig_project()
    p.fleet, _ = synth_fleet(p)
    p.fleet['settings'] = {'pos_tol_mm': 5.0}
    path = tmp_path / 'f.vantage'
    p.save(str(path))
    q = Project.load(str(path))
    assert q.fleet == p.fleet and FL.settings_of(q).pos_tol_mm == 5.0
    a, b = FL.run(p), FL.run(q)
    assert FL.summary(a) == FL.summary(b)
    store = ProjectStore(q)
    vid = q.fleet['vehicles'][0]['id']
    n = len(q.fleet['vehicles'][0]['records'])
    store.edit('fleet', lambda pj: FL.add_record(pj.fleet, vid, FL.manual_record('2026-10-01')), 'Fleet · new record')
    store.edit('fleet', lambda pj: FL.remove_vehicle(pj.fleet, q.fleet['vehicles'][1]['id']), 'Fleet · deleted V2')
    assert len(q.fleet['vehicles']) == 4 and len(q.fleet['vehicles'][0]['records']) == n + 1
    store.undo.undo()
    store.undo.undo()
    assert q.fleet == p.fleet
    store.undo.redo()
    assert len(q.fleet['vehicles'][0]['records']) == n + 1 and len(q.fleet['vehicles']) == 5
    store.undo.cleanChanged.disconnect()        # no signal into a store Python has already let go of


def test_outputs_and_issues(tmp_path):
    import cv2
    p = rig_project()
    p.fleet, _ = synth_fleet(p, extra=lambda v, t, j: [0, 0, 0, 0, 0, 4.0] if (v, t) == ('V5', 'R') else [0] * 6)
    res = FL.run(p)
    s = FL.summary(res)
    json.dumps(s)
    assert s['vehicles'] == 5 and s['records'] == 20 and s['verdicts'] == {'ok': 67, 'adjust': 2, 're-mount': 1}
    assert [d['vehicle'] for d in s['drifting']] == ['V2'] and s['worst'][0]['sensor'] == 'R'
    doc = FL.to_doc(res, p)
    assert doc['schema'] == 'vantage.fleet/1' and 'rear axle' in doc['frame'] and doc['units']['offsets'] == 'mm'
    v2 = next(v for v in doc['vehicles'] if v['name'] == 'V2')
    assert len(v2['records']) == 4 and len(v2['sensors']) == 14
    f = next(x for x in v2['sensors'] if x['sensor'] == 'F')
    assert f['drift']['drifting'] and f['verdict'] == 'adjust'
    rows = FL.to_csv(res).strip().splitlines()
    assert len(rows) == 1 + 5 * 14 and rows[0].split(',')[:3] == ['vehicle', 'vin', 'sensor']
    paths = FL.export(res, p, str(tmp_path / 'out'))
    assert [os.path.basename(x) for x in paths] == ['fleet.json', 'fleet.csv', 'fleet.png']
    img = cv2.imread(paths[2])
    assert img is not None and img.shape[1] == 1200 and img.shape[0] > 600
    assert json.loads(open(paths[0], encoding='utf-8').read())['vehicles'][0]['name'] == 'V1'
    # the Issues list: cheap, latest records only
    import time
    t = time.perf_counter()
    iss = FL.issues(p)
    assert time.perf_counter() - t < 0.03
    got = {(i.severity, i.objects) for i in iss}
    assert got == {('warning', 'V2 · F'), ('warning', 'V4 · ML'), ('serious', 'V5 · R')}
    assert all(i.uids == [p.variant.by_tag(i.objects.split(' · ')[1]).uid] for i in iss)
    p.fleet['settings'] = {'pos_tol_mm': 10, 'ang_tol_deg': 5.0}
    assert {(i.severity, i.objects) for i in FL.issues(p)} == set()
    # the Issues list's lean path (newest records only) agrees with the analysis, also when the newest pose of a sensor
    # is broken and an older record has to stand in
    p.fleet['settings'] = {}
    for seed in (1, 2, 3):
        p.fleet, _ = synth_fleet(p, seed=seed, extra=lambda v, t, j: [0, 0, 30.0 * (j == 2), 0, 0, 0]
                                 if t == 'RD3' else [0] * 6)
        rec = p.fleet['vehicles'][seed]['records'][-1]
        rec['poses']['RD3']['z'] = float('nan')                      # V(seed+1)'s newest RD3 is unusable
        res = FL.run(p)
        want = {(c['vehicle'], c['tag'], c['verdict']) for c in res.cells.values()
                if c['verdict'] in ('adjust', 're-mount')}
        vid = {v['name']: v['id'] for v in res.vehicles}
        got = {(vid[i.objects.split(' · ')[0]], i.objects.split(' · ')[1],
                're-mount' if i.severity == 'serious' else 'adjust') for i in FL.issues(p)}
        assert got == want and any(t == 'RD3' for _, t, _ in want)


def test_page_analyses_a_big_fleet_on_a_worker(monkeypatch):
    """Over ASYNC_POSES poses the page copies the fleet and computes it on a thread; the result is the same, a newer
    refresh supersedes an older one, and a failure is reported on the page."""
    import time
    from types import SimpleNamespace
    from PyQt6.QtWidgets import QApplication
    from vantage.ui.store import ProjectStore
    from vantage.ui.workspaces import fleet_ws as FW
    p = rig_project()
    p.fleet, _ = synth_fleet(p)
    store = ProjectStore(p)
    ctx = SimpleNamespace(store=store, msg=lambda *a: None, window=None)
    monkeypatch.setattr(FW, 'ASYNC_POSES', 100)
    pg = FW.FleetPage(ctx)

    def wait():
        t = time.time()
        while pg.busy and time.time() - t < 30:
            QApplication.processEvents()
            time.sleep(0.01)
        assert not pg.busy
    wait()
    assert FL.summary(pg.result) == FL.summary(FL.run(p)) and pg.matrix.res is pg.result
    first = pg.result
    pg.refresh()
    pg.refresh()                                                     # supersedes the one before
    assert pg.busy and pg._gen == 3
    wait()
    assert pg.result is not first and FL.summary(pg.result) == FL.summary(first)
    monkeypatch.setattr(FL, 'run', lambda *a, **k: 1 / 0)
    pg.refresh()
    t = time.time()
    while pg.busy and time.time() - t < 10:
        QApplication.processEvents()
        time.sleep(0.01)
    assert 'failed: ZeroDivisionError' in pg.status.text.text()
    store.undo.cleanChanged.disconnect()


# ----------------------------------------------------------------------------- the page
from test_gui import app, win, pump  # noqa: E402,F401  (fixtures)


def _open(win, name, cls):
    if name not in win._page_widgets:
        pg = cls(win.ctx)
        win.pages.addWidget(pg)
        win._page_widgets[name] = pg
    win.set_page(name)
    return win._page_widgets[name]


def test_page_add_import_matrix_edit_and_undo(app, win, tmp_path):
    from PyQt6.QtCore import QPoint, Qt
    from PyQt6.QtTest import QTest
    from vantage.ui.workspaces.fleet_ws import FleetPage
    store = win.store
    p = store.project
    rig = next(v for v in p.variants if v.sensors)
    win.switch_variant(rig.uid)
    pump(app, 0.1)
    pg = _open(win, 'fleet', FleetPage)
    pump(app, 0.2)
    assert win.page_name() == 'fleet' and pg.isVisible()
    assert pg.result.vehicles == [] and pg.findings[0].text.text().startswith('No vehicles yet')
    # an as-built extrinsics file from a calibration session: F 15 mm low, ML yawed 2°
    q = copy.deepcopy(p)
    q.variant.by_tag('F').z -= 0.015
    q.variant.by_tag('ML').yaw += 2.0
    out = tmp_path / 'cal'
    ex.export(['extrinsics_json'], q, None, ex.ExportOptions(out_dir=str(out), frame='front_bumper', units='mm'))
    # add a vehicle with the button, name it, import the file into it
    QTest.mouseClick(pg.b_add_vehicle, Qt.MouseButton.LeftButton)
    pump(app, 0.1)
    assert len(p.fleet['vehicles']) == 1 and pg.vtable.rowCount() == 1
    e = pg.v_edits['name']
    e.setFocus()
    e.selectAll()
    QTest.keyClicks(e, 'Test car 7')
    QTest.keyClick(e, Qt.Key.Key_Return)
    pump(app, 0.1)
    assert p.fleet['vehicles'][0]['name'] == 'Test car 7' and pg.vtable.item(0, 0).text() == 'Test car 7'
    pg.import_files([str(out / 'extrinsics.json')])
    pump(app, 0.1)
    vid = p.fleet['vehicles'][0]['id']
    assert len(p.fleet['vehicles'][0]['records']) == 1 and pg.rtable.rowCount() == 1
    assert pg.status.text.text().startswith('Imported into Test car 7: 12 poses from extrinsics.json')
    res = pg.result
    assert res.cell(vid, 'F')['verdict'] == 'adjust' and res.cell(vid, 'F')['d'][2] == pytest.approx(-15, abs=0.6)
    assert res.cell(vid, 'ML')['verdict'] == 'adjust' and res.cell(vid, 'R')['verdict'] == 'ok'
    assert pg.tiles['off'].value == '2 · 0' and pg.tiles['off'].sev == 'warning'
    # the matrix: click the F cell, see its detail; the poses of the record are listed
    r = pg.matrix.cell_rect(vid, 'F')
    assert r is not None
    QTest.mouseClick(pg.matrix, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                     QPoint(int(r.center().x()), int(r.center().y())))
    pump(app, 0.1)
    assert pg.sel_cell == (vid, 'F') and pg.detail.text.text().startswith('Test car 7 · F · Adjust')
    assert re.search(r'Δz -1[45]\.\d mm', pg.detail.text.text())
    assert pg.ptable.rowCount() == 12
    # a typed-in correction to F's z in the record (a re-survey after fixing the bracket): back within tolerance
    row = next(i for i in range(pg.ptable.rowCount()) if pg.ptable.item(i, 0).text() == 'F')
    planned_z = rig.by_tag('F').z
    pg.ptable.item(row, 3).setText(f'{planned_z:.4f}')
    pump(app, 0.1)
    assert pg.result.cell(vid, 'F')['verdict'] == 'ok'
    # change the record's date in its table
    pg.rtable.item(0, 0).setText('2026-04-01')
    pump(app, 0.1)
    assert p.fleet['vehicles'][0]['records'][0]['date'] == '2026-04-01'
    pg.rtable.item(0, 0).setText('someday')                    # refused, the table shows the stored date again
    pump(app, 0.1)
    assert p.fleet['vehicles'][0]['records'][0]['date'] == '2026-04-01' and pg.rtable.item(0, 0).text() == '2026-04-01'
    # a stricter angle tolerance, typed into its field
    f = pg.fields['ang_tol_deg']
    QTest.mouseClick(f.lineEdit(), Qt.MouseButton.LeftButton)
    QTest.keyClick(f.lineEdit(), Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
    QTest.keyClicks(f.lineEdit(), '0.5')
    QTest.keyClick(f.lineEdit(), Qt.Key.Key_Return)
    pump(app, 0.1)
    assert p.fleet['settings']['ang_tol_deg'] == 0.5 and pg.result.settings.ang_tol_deg == 0.5
    # a second vehicle from a variant, then the fleet-median variant: one undo step each
    n_var = len(p.variants)
    pg.add_vehicle('V2')
    pg.import_variant(rig.uid)
    pump(app, 0.1)
    assert len(pg.result.vehicles) == 2 and pg.matrix.cell_rect(pg.sel_vehicle, 'L1') is not None
    # pick the first vehicle again in its table: its records and fields follow
    QTest.mouseClick(pg.vtable.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                     pg.vtable.visualItemRect(pg.vtable.item(0, 1)).center())
    pump(app, 0.1)
    assert pg.sel_vehicle == vid and pg.v_edits['name'].text() == 'Test car 7' and pg.rtable.rowCount() == 1
    v = pg.make_median_variant()
    pump(app, 0.1)
    assert len(p.variants) == n_var + 1 and p.active_uid == v.uid and win.variant_combo.count() == n_var + 1
    png = pg.export(str(tmp_path / 'export'))
    assert len(png) == 3 and os.path.getsize(png[2]) > 1000
    # undo everything, step by step, back to an empty fleet and the original variants
    store.undo.undo()                                          # the median variant
    pump(app, 0.1)
    assert len(p.variants) == n_var and p.active_uid == rig.uid
    for _ in range(7):                     # variant record, V2, tolerance, date, pose, extrinsics import, name
        store.undo.undo()
        pump(app, 0.05)
    assert [x['name'] for x in p.fleet['vehicles']] == ['V1'] and p.fleet['vehicles'][0]['records'] == []
    store.undo.undo()
    pump(app, 0.1)
    assert not p.fleet.get('vehicles') and pg.vtable.rowCount() == 0 and pg.result.vehicles == []
    store.undo.redo()
    pump(app, 0.1)
    assert pg.vtable.rowCount() == 1
    # a project opened while the page shows: its (empty) fleet replaces the old one
    store.set_project(samples.tigor_sample_project())
    pump(app, 0.1)
    assert pg.result.vehicles == [] and pg.vtable.rowCount() == 0 and pg.sel_vehicle == '' and pg.sel_cell is None
