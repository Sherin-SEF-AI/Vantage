"""Fleet twin (core.fleet_twin, ui.workspaces.fleet_twin_ws): the drift model against known truths, the knock test's
detection and false-alarm rate, the fleet's borrowing of strength, first passage against Monte Carlo, imports, the
watch folder, outputs, old files, and the page driven like a user."""
from __future__ import annotations

import copy
import datetime as dt
import json
import math
import os
import shutil
import sys
import time

import numpy as np
import pytest

from vantage.core import fleet as FL
from vantage.core import fleet_twin as FT
from vantage.core import samples
from vantage.core.project import Project

START = dt.date(2026, 1, 5)


def rig_project():
    p = samples.tigor_sample_project()
    v = next(x for x in p.variants if x.sensors)
    p.active_uid = v.uid
    p.fleet = {}
    return p


def plus(pose, d):
    return {'x': pose['x'] + d[0] / 1000.0, 'y': pose['y'] + d[1] / 1000.0, 'z': pose['z'] + d[2] / 1000.0,
            'yaw': pose['yaw'] + d[3], 'pitch': pose['pitch'] + d[4], 'roll': pose['roll'] + d[5]}


def fleet_of(p, n=6, records=8, every=30, sigma=(2.0, 0.05), drift=None, knock=None, short=None, seed=3,
             kind='install_check', tags=None):
    """n vehicles × records dated every `every` days; offsets per vehicle and sensor; the measurement noise of the
    stated σ of `kind`; drift(name, tag) -> [6] rate per day; knock: (name, tag, record index, [6] step); short:
    {name: number of records}."""
    rng = np.random.default_rng(seed)
    plan = FL.plan_sensors(p, p.variant)
    if tags is not None:
        plan = [ps for ps in plan if ps['tag'] in tags]
    for i in range(n):
        name = f'V{i + 1}'
        v = FL.add_vehicle(p.fleet, name)
        off = {ps['tag']: np.r_[rng.normal(0, 2, 3), rng.normal(0, 0.1, 3)] for ps in plan}
        for j in range((short or {}).get(name, records)):
            day = j * every
            poses = {}
            for ps in plan:
                d = off[ps['tag']] + np.r_[rng.normal(0, sigma[0], 3), rng.normal(0, sigma[1], 3)]
                if drift is not None:
                    d = d + np.asarray(drift(name, ps['tag']), float) * day
                if knock is not None and knock[0] == name and knock[1] == ps['tag'] and j >= knock[2]:
                    d = d + np.asarray(knock[3], float)
                poses[ps['tag']] = plus(ps['pose'], d)
            FL.add_record(p.fleet, v['id'], {'kind': kind, 'date': (START + dt.timedelta(days=day)).isoformat(),
                                             'source': f'check {j + 1}', 'poses': poses})
    return p


def cell(res, name, tag):
    vid = next(v['id'] for v in res.vehicles if v['name'] == name)
    return res.cells[(vid, tag)]


# =========================================================================== the model against known truths
def test_one_series_rate_equals_least_squares_slope_and_its_standard_error():
    """No process noise, a diffuse rate prior: the filter's rate at the end is the least-squares slope and its σ the
    slope's standard error for the known measurement σ (an independent closed form)."""
    rng = np.random.default_rng(1)
    t = np.sort(rng.uniform(0, 300, 9))
    sig = 0.04
    y = 0.3 + 0.002 * t + rng.normal(0, sig, len(t))
    pk = FT._Pack([{'t': t, 'y': y, 'v': np.full(len(t), sig ** 2), 'src': np.zeros(len(t), int)}])
    kf = FT._kf2(pk, np.zeros(1), np.full(1, 1e4), np.zeros(1), np.zeros(1), np.ones_like(pk.y), keep=True)
    X = np.c_[np.ones_like(t), t - t[-1]]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    cov = sig ** 2 * np.linalg.inv(X.T @ X)
    assert kf['x'][0, 1] == pytest.approx(beta[1], rel=1e-5)
    assert math.sqrt(kf['P'][0, 2]) == pytest.approx(math.sqrt(cov[1, 1]), rel=1e-4)
    assert kf['x'][0, 0] == pytest.approx(beta[0], abs=1e-6)                    # the level at the last time
    sm = FT._smooth(pk, kf)
    fit = X @ beta
    assert np.allclose(sm['XS'][0, :, 0], fit + 0.0, atol=1e-6)                 # the smoothed level is the line


def test_kalman_recovers_known_linear_drifts_with_calibrated_intervals():
    """300 vehicles drifting at known rates (a fleet spread of rates), 12 noisy records each: the fitted model's
    90 % intervals of each vehicle's rate and level now cover the truth about 90 % of the time, unbiased."""
    rng = np.random.default_rng(7)
    S, n, sig = 300, 12, 0.05
    rates = rng.normal(0.001, 0.0006, S)
    rows, truth_L = [], []
    for i in range(S):
        t = np.sort(rng.uniform(0, 330, n))
        L0 = rng.normal(0, 0.2)
        y = L0 + rates[i] * t + rng.normal(0, sig, n)
        rows.append({'t': t, 'y': y, 'v': np.full(n, sig ** 2), 'src': np.zeros(n, int)})
        truth_L.append(L0 + rates[i] * t[-1])
    pk = FT._Pack(rows)
    zeros = np.zeros(S, int)
    fit, pmap = FT.fit_hyper(pk, zeros, zeros, zeros, 1)
    assert fit.mu[pmap[0]] == pytest.approx(0.001, abs=1.5e-4)                  # the fleet's mean rate
    assert fit.tau[pmap[0]] == pytest.approx(0.0006, rel=0.25)                   # and its spread
    qr, ql, s2, off = FT._series_params(fit, pk, zeros)
    assert np.allclose(np.sqrt(s2[:, 0]), 1.0, atol=0.15)                       # the stated σ was right
    mu = np.full(S, fit.mu[pmap[0]])
    var = np.full(S, fit.tau[pmap[0]] ** 2)
    kf = FT._kf2(pk, mu, var, qr, ql, s2)
    r, rs = kf['x'][:, 1], np.sqrt(kf['P'][:, 2])
    L, Ls = kf['x'][:, 0], np.sqrt(kf['P'][:, 0])
    cov_r = np.mean(np.abs(r - rates) < 1.645 * rs)
    cov_L = np.mean(np.abs(L - np.array(truth_L)) < 1.645 * Ls)
    assert 0.84 <= cov_r <= 0.96 and 0.84 <= cov_L <= 0.96
    assert abs(np.mean(r - rates)) < 3 * np.std(r - rates) / math.sqrt(S)


# =========================================================================== knocks
def _sim(rng, S, n, sig, qr=1e-10, rate_sd=0.0005, step=None):
    """Series of the model with known parameters (internal units): level, rate random walk, measurement σ;
    step: (observation index, size) added from that observation on."""
    rows = []
    for _ in range(S):
        t = np.cumsum(rng.uniform(10, 40, n))
        t -= t[0]
        R = rng.normal(0, rate_sd)
        L = rng.normal(0, 0.3)
        y = np.empty(n)
        for k in range(n):
            if k:
                dt_ = t[k] - t[k - 1]
                L += R * dt_
                R += rng.normal(0, math.sqrt(qr * dt_))
            y[k] = L + (step[1] if step is not None and k >= step[0] else 0.0) + rng.normal(0, sig)
        rows.append({'t': t, 'y': y, 'v': np.full(n, sig ** 2), 'src': np.zeros(n, int)})
    return FT._Pack(rows)


def test_knock_found_within_two_observations_at_the_stated_false_alarm_rate():
    rng = np.random.default_rng(11)
    sig, alpha, W = 0.05, 1e-3, 6
    S0, n0 = 1500, 30
    pk = _sim(rng, S0, n0, sig)
    args = (np.zeros(S0), np.full(S0, 0.0005 ** 2), np.full(S0, 1e-10), np.zeros(S0), np.ones_like(pk.y))
    _, alarms = FT.detect_steps(pk, *args, np.zeros(S0), alpha, W)
    tested = S0 * (n0 - 1)
    rate = len(alarms) / tested
    assert len(alarms) <= alpha * tested + 3 * math.sqrt(alpha * tested)      # at most α per observation (Bonferroni)
    # a 0.5° knock (10 σ) at observation 15 of 200 series, and a 0.25° one (5 σ)
    for size, within, share in ((0.5, 1, 0.99), (0.25, 2, 0.9)):
        S = 200
        pk = _sim(rng, S, 25, sig, step=(15, size))
        args = (np.zeros(S), np.full(S, 0.0005 ** 2), np.full(S, 1e-10), np.zeros(S), np.ones_like(pk.y))
        steps, alarms = FT.detect_steps(pk, *args, np.zeros(S), alpha, W)
        first = {}
        for (i, j, k, b, sb, l, ok) in alarms:
            if k >= 15 and i not in first:
                first[i] = (j, k, b, sb)
        found = [i for i, (j, k, b, sb) in first.items() if j in (14, 15, 16) and k - 15 + 1 <= within]
        assert len(found) >= share * S, (size, len(found))
        errs = np.array([first[i][2] - size for i in found])
        assert abs(np.median(errs)) < 0.3 * sig * 3                                 # the size, unbiased
    assert rate < alpha


def test_a_knock_in_the_fleet_is_reported_with_its_size_date_and_delay():
    p = fleet_of(rig_project(), n=5, records=10, knock=('V3', 'F', 6, [0, 0, 0, 0.0, 0.6, 0.0]), tags=('F', 'ML'))
    res = FT.run(p, None, FT.FleetTwinSettings(as_of='2026-11-05', limit_source='fleet'))
    knocks = res.knocks
    assert len(knocks) == 1
    k = knocks[0]
    assert (k['name'], k['tag'], k['axes'], k['delay']) == ('V3', 'F', ['pitch'], 1)
    assert k['size']['pitch'] == pytest.approx(0.6, abs=4 * k['sigma']['pitch'])
    assert k['date'] == (START + dt.timedelta(days=180)).isoformat() and k['after'] == (
        START + dt.timedelta(days=150)).isoformat()
    assert any(f['kind'] == 'knock' and 'V3' in f['message'] for f in res.findings)
    sd = cell(res, 'V3', 'F')['axes']['pitch']
    assert sd['now'][0] == pytest.approx(np.mean(sd['y'][6:]), abs=0.05)        # the level after the knock


# =========================================================================== borrowing strength
def test_a_two_record_vehicle_borrows_its_rate_from_the_fleet():
    """Eight vehicles whose front camera pitches 0.06°/30 d (spread 0.01) and a ninth with two records 10 days apart
    whose raw slope is 0.18°/30 d (within its large uncertainty, σ 0.08°/30 d): its rate is pulled most of the way
    to the fleet's; alone (fewer than three vehicles to pool) it keeps its own slope."""
    rng = np.random.default_rng(5)
    p = rig_project()
    p.fleet = {}
    rates = {f'V{i + 1}': rng.normal(0.06, 0.01) / 30.0 for i in range(8)}
    fleet_of(p, n=8, records=8, sigma=(2.0, 0.02), drift=lambda name, tag: [0, 0, 0, 0, rates[name], 0] if tag == 'F'
             else [0] * 6, tags=('F',), kind='extrinsics')
    ps = next(x for x in FL.plan_sensors(p, p.variant) if x['tag'] == 'F')
    v9 = FL.add_vehicle(p.fleet, 'V9')
    for day, dp in ((0, 0.0), (10, 0.06)):
        FL.add_record(p.fleet, v9['id'], {'kind': 'extrinsics', 'date': (START + dt.timedelta(days=day)).isoformat(),
                                          'poses': {'F': plus(ps['pose'], [0, 0, 0, 0, dp, 0])}})
    res = FT.run(p, None, FT.FleetTwinSettings(as_of='2026-11-05', limit_source='fleet'))
    raw = 0.06 / 10 * 30
    r9 = cell(res, 'V9', 'F')['axes']['pitch']['rate_now'][0] * 30
    fleet = np.mean([cell(res, f'V{i + 1}', 'F')['axes']['pitch']['rate_now'][0] * 30 for i in range(8)])
    assert fleet == pytest.approx(0.06, abs=0.01)
    assert abs(r9 - fleet) < 0.25 * abs(raw - fleet)
    row = next(f for f in res.fleet if f['tag'] == 'F' and f['axis'] == 'pitch')
    assert row['pooled'] == 'sensor' and row['prior_rate_per_30d'] == pytest.approx(0.06, abs=0.02)
    # alone (the other vehicles removed) nothing to borrow from: its own slope
    p.fleet['vehicles'] = [v for v in p.fleet['vehicles'] if v['name'] == 'V9']
    res1 = FT.run(p, None, FT.FleetTwinSettings(as_of='2026-11-05', limit_source='fleet'))
    assert cell(res1, 'V9', 'F')['axes']['pitch']['rate_now'][0] * 30 == pytest.approx(raw, rel=0.05)


# =========================================================================== first passage
def _mc_first_passage(m, P, qr, ql, lo, hi, H, N=60000, dt_=0.5, seed=0):
    """Monte Carlo of the same model: exact Gaussian transitions (the integrated rate noise included) every dt_, and
    between them a crossing by the level noise with the Brownian bridge's probability exp(−2 a b / (q_l dt)) (a, b:
    the distances to the limit at either end), so the paths are watched continuously."""
    rng = np.random.default_rng(seed)
    C = np.linalg.cholesky(np.array([[P[0], P[1]], [P[1], P[2]]]) + 1e-18 * np.eye(2))
    X = (C @ rng.standard_normal((2, N))).T + np.asarray(m)
    Q = qr * np.array([[dt_ ** 3 / 3, dt_ ** 2 / 2], [dt_ ** 2 / 2, dt_]]) + ql * np.array([[dt_, 0], [0, 0]])
    Cq = np.linalg.cholesky(Q + 1e-30 * np.eye(2))
    out = (X[:, 0] <= -lo) | (X[:, 0] >= hi)
    steps = int(round(H / dt_))
    curve = [out.mean()]
    for _ in range(steps):
        x0 = X[:, 0]
        X = np.c_[X[:, 0] + dt_ * X[:, 1], X[:, 1]] + (Cq @ rng.standard_normal((2, N))).T
        x1 = X[:, 0]
        out |= (x1 <= -lo) | (x1 >= hi)
        if ql > 0:
            with np.errstate(over='ignore'):
                p_hi = np.exp(-2.0 * np.maximum(hi - x0, 0) * np.maximum(hi - x1, 0) / (ql * dt_))
                p_lo = np.exp(-2.0 * np.maximum(x0 + lo, 0) * np.maximum(x1 + lo, 0) / (ql * dt_))
            out |= rng.random(N) < p_hi + p_lo
        curve.append(out.mean())
    return np.linspace(0, H, steps + 1), np.array(curve)


@pytest.mark.parametrize('m, P, qr, ql, lo, hi', [
    ((0.2, 0.004), (0.01, 0.0, 0.002 ** 2), 1e-9, 0.0, 1.0, 1.0),              # drifting up, an uncertain rate
    ((0.0, 0.0), (0.04, 0.0, 0.003 ** 2), 2e-8, 1e-4, 1.0, 0.6),               # no mean drift, asymmetric limits
    ((-0.5, -0.001), (0.0025, 1e-5, 1e-6), 1e-10, 2e-3, 1.0, 1.0),             # level noise dominates
    ((0.0, 0.0), (0.0025, 0.0, 1e-12), 1e-12, 4e-3, 0.8, 0.8),                  # a random walk alone
])
def test_first_passage_matches_monte_carlo(m, P, qr, ql, lo, hi):
    H = 365.0
    t, pf = FT.first_passage(np.array(m), np.array(P), qr, ql, lo, hi, H)
    tm, pm = _mc_first_passage(m, P, qr, ql, lo, hi, H)
    assert np.max(np.abs(np.interp(tm, t, pf) - pm)) < 0.012
    marg = FT.marginal_exceed(np.array(m), np.array(P), qr, ql, lo, hi, t)
    assert np.all(pf >= marg - 0.005)                                           # first passage ≥ the marginal
    assert np.all(np.diff(pf) >= -1e-9)                                         # and never falls


# =========================================================================== imports
def _extrinsics_doc(pitch_corr=0.3, sig=0.02, start_ns=1767268800 * 10 ** 9, segments=1):
    segs = []
    for k in range(segments):
        segs.append({'t0_s': 5.0 * k, 't1_s': 5.0 * (k + 1),
                     'correction': {'yaw': 0.0, 'pitch': pitch_corr + 0.1 * k, 'roll': None},
                     'sigma': {'yaw': sig, 'pitch': sig, 'roll': None}})
    return {'schema': 'vantage.online_calib.extrinsics/1', 'drive': 'd1', 'reference': 'L1', 'start_ns': start_ns,
            'sensors': {'F': {'kind': 'camera', 'plan_yaw_pitch_roll_deg': [0.0, 40.0, 0.0],
                              'plan_position_m': [3.23, 0.0, 0.62], 'axes': ['yaw', 'pitch', 'roll'],
                              'relative_to': 'L1', 'segments': segs,
                              'interpolated': {'t_s': [1.0, 3.0, 5.0], 'correction': [[0, 0.3, 0.0]] * 3,
                                               'sigma': [[0.02, 0.02, 0.02]] * 3}}}}


def test_online_calibration_imports_against_the_twin_plan():
    p = rig_project()
    v = FL.add_vehicle(p.fleet, 'V1')
    obs = FT.online_observations(_extrinsics_doc(segments=2), v['id'])
    assert len(obs) == 2 and obs[0]['date'].startswith('2026-01-01T12:00:02')     # the drive's clock + the middle
    assert set(obs[0]['values']) == {'yaw', 'pitch'}                              # roll not observable: left out
    assert len(FT.online_observations(_extrinsics_doc(), v['id'], mode='windows')) == 3
    with pytest.raises(FT.TwinError):
        FT.online_observations(_extrinsics_doc(start_ns=12345), v['id'])         # no calendar clock, no date
    assert FT.add_observations(p, obs) == 2 and FT.add_observations(p, obs) == 0  # duplicates dropped
    cam = next(c for c in p.variant.cameras if c.tag == 'F')
    cam.pitch += 0.1                                                             # the twin's plan moved
    plan = FL.plan_sensors(p, p.variant)
    got = FT.stored_observations(p, plan)[(v['id'], 'F')]
    assert got[0]['d'][4] == pytest.approx(0.2) and np.isnan(got[0]['d'][5]) and np.isnan(got[0]['d'][0])
    assert FT.remove_observations(p, v['id']) == 2 and not FT.observations(p)


def test_record_sigma_defaults_and_install_rms():
    rec = FL.clean_record({'kind': 'install_check', 'poses': {}, 'quality': {'F': {'rms_px': 1.0}}})
    assert FT._record_sigma(rec, 'F')[:2] == pytest.approx((4.0, 0.1))
    rec = FL.clean_record({'kind': 'manual', 'poses': {}, 'quality': {'F': {'sigma_mm': 0.5, 'sigma_deg': 0.01}}})
    assert FT._record_sigma(rec, 'F')[:2] == (0.5, 0.01)
    assert FT._record_sigma(FL.clean_record({'kind': 'nuscenes', 'poses': {}}), 'F')[:2] == (1.0, 0.03)


def test_rig_health_drift_events_are_flagged_not_measured():
    p = fleet_of(rig_project(), n=3, records=4, tags=('F',))
    vid = FL.vehicles(p)[0]['id']
    snap = {'cameras': {'F': {'drift_px': 24.0}, 'R': {'drift_px': 3.0}, 'ML': {'drift_px': None}},
            'time': '2026-04-10T10:00:00'}
    evs = FT.rig_health_events(p, snap, vid)
    assert [e['tag'] for e in evs] == ['F']
    from vantage.core.lens import resolve
    cam = next(c for c in p.variant.cameras if c.tag == 'F')
    intr = resolve(p.module_for(cam), p.lens_for(cam))
    assert evs[0]['deg'] == pytest.approx(24.0 / (intr.f_px * math.pi / 180.0))
    FT.add_events(p, evs)
    res = FT.run(p, None, FT.FleetTwinSettings(as_of='2026-05-01', limit_source='fleet'))
    assert any(f['kind'] == 'rig_health' and 'F' in f['message'] for f in res.findings)
    n_obs = sum(len(sd['t']) for c in res.cells.values() for sd in c['axes'].values())
    assert n_obs == 3 * 4 * 6                                                    # the event is not an observation


def test_limits_from_critical_envelopes_else_the_mounting_tolerance():
    p = rig_project()
    plan = FL.plan_sensors(p, p.variant)
    f = next(x for x in plan if x['tag'] == 'F')

    class Env:
        lo = {'pitch': (0.4, 'child at 2 m', 'pixels'), 'yaw': (9.0, '', 'open'), 'z': (0.004, 'kerb', 'seen')}
        hi = {'pitch': (0.7, 'child at 2 m', 'pixels')}

    class Crit:
        def camera(self, key):
            return Env() if key == f['uid'] else None
    lim = FT.limits(p, plan, FT.FleetTwinSettings(), Crit())
    L = lim['F']
    a = p.analysis
    assert L['lo'][4] == 0.4 and L['hi'][4] == 0.7 and L['target'][4] == 'child at 2 m'
    assert L['lo'][2] == pytest.approx(4.0) and L['hi'][2] == a.tol_mm         # the kerb sets z's lower side (mm)
    assert L['lo'][3] == a.tol_deg and L['source'][3] == 'mounting tolerance'  # an open side has no target
    assert FT.limits(p, plan, FT.FleetTwinSettings(limit_source='user', user_deg=0.5), Crit())['F']['lo'][4] == 0.5
    other = lim['L1']
    assert list(other['lo']) == [a.tol_mm] * 3 + [a.tol_deg] * 3


def test_limits_from_a_real_critical_target():
    """A child in front of the car (core.critical's preset): the front camera's limits come from its envelope."""
    from vantage.core import critical as CR
    p = rig_project()
    p.analysis.critical = {'targets': [CR.preset(p, 'child')]}
    fleet_of(p, n=3, records=4, tags=('F',))
    res = FT.run(p, None, FT.FleetTwinSettings(as_of='2026-06-01'))
    assert res.metrics['critical']
    crit = CR.run(p, None, variant=p.variant, envelopes=True)
    env = crit.camera('F')
    c = next(c for c in res.cells.values() if c['tag'] == 'F')
    hit = 0
    for ax, sd in c['axes'].items():
        u = 1000.0 if ax in ('x', 'y', 'z') else 1.0
        for side, key in ((env.lo, 'lo'), (env.hi, 'hi')):
            e = side.get(ax)
            if e is not None and e[1]:
                assert sd[key] == pytest.approx(e[0] * u, rel=1e-6) and c['limit_target'][ax] == e[1]
                hit += 1
    assert hit >= 1


# =========================================================================== outputs, issues, tolerances, old files
def test_drift_forecast_schedule_outputs_and_issues(tmp_path):
    p = fleet_of(rig_project(), n=5, records=9,
                 drift=lambda name, tag: [0, 0, 0, 0, 0.3 / 30.0, 0] if (name, tag) == ('V2', 'ML') else [0] * 6)
    st = FT.FleetTwinSettings(as_of='2026-10-01', limit_source='user', user_deg=3.0, user_mm=20.0)
    FT.set_settings(p, **st.to_dict())
    res = FT.run(p, None, FT.settings_of(p))
    c = cell(res, 'V2', 'ML')
    sd = c['axes']['pitch']
    assert sd['rate_now'][0] * 30 == pytest.approx(0.3, abs=0.05)
    # 0.3°/30 d from about 2.4° on 2026-09-01 (plus its offset): the 3° limit about two months on
    assert c['urgency'] in ('due', 'soon', 'later', 'beyond') and c['date_risk'] is not None
    assert c['date_50'] is not None and c['date_risk'] <= c['date_50'] <= (c['date_90'] or 1e18)
    expect = sd['t_last'] + (3.0 - sd['now'][0]) / sd['rate_now'][0]
    assert c['date_50'] == pytest.approx(expect, abs=10.0)                       # the median: the mean line's crossing
    assert res.schedule[0]['vehicle'] == 'V2' and res.schedule[0]['sensor'] == 'ML'
    assert FT.exit_code(res) in (0, 3) and FT.exit_code(None) == 4
    doc = json.loads(json.dumps(FT.to_doc(res, p)))
    assert doc['schema'] == FT.SCHEMA and doc['cells'] and doc['tolerance_inputs']['schema'] == FT.TOLERANCE_SCHEMA
    names = [n for n, _ in FT.files(res, p)]
    assert names == ['fleet_twin.json', 'fleet_twin.csv', 'fleet_twin_schedule.csv', 'fleet_twin.png',
                     'fleet_twin_tolerances.json']
    paths = FT.export(res, p, str(tmp_path))
    assert all(os.path.getsize(x) > 100 for x in paths)
    head = open(paths[1]).readline().strip().split(',')
    assert tuple(head) == FT.CSV_COLUMNS
    assert open(paths[2]).read().splitlines()[1].split(',')[2] == 'V2'
    assert 'provenance' in json.load(open(paths[4]))
    t0 = time.perf_counter()
    iss = FT.issues(p)                                                           # cheap: from the cache
    assert time.perf_counter() - t0 < 0.1
    assert any('V2' in i.message and 'ML' in i.message for i in iss) == (c['urgency'] in ('due', 'soon', 'beyond'))
    s = FT.summary(res)
    assert s['schedule'][0]['sensor'] == 'ML' and json.dumps(s)
    tol = FT.tolerance_inputs(res, 180)
    a = tol['sensors']['ML']['axes']['pitch']
    row = next(f for f in res.fleet if f['tag'] == 'ML' and f['axis'] == 'pitch')
    assert a['drift']['mean'] == pytest.approx(row['prior_rate_per_30d'] * 6, rel=1e-3)     # the prior × 180 days
    assert a['drift']['sd'] >= abs(row['prior_rate_sd_per_30d']) * 6 * 0.999                 # V2 spreads the fleet
    assert a['total']['p95_abs'] >= abs(a['total']['mean'])
    ts = FT.tol_settings(res, 180)
    assert ts.distribution == 'normal' and ts.pitch == pytest.approx(tol['tol_settings']['pitch'])
    assert ts.pitch >= 2 * a['total']['sd'] - 1e-6
    FT.set_settings(p, horizon_days=400)                                         # a change: the cache no longer fits
    assert FT.issues(p) == []


def test_old_project_files_without_the_slot_and_junk_load(tmp_path):
    p = fleet_of(rig_project(), n=3, records=3, tags=('F',))
    import zipfile
    path = str(tmp_path / 'old.vantage')
    p.save(path)
    with zipfile.ZipFile(path) as z:                   # the file as an older version wrote it: no fleet_twin slot
        parts = {n: z.read(n) for n in z.namelist()}
    raw = json.loads(parts['project.json'])
    raw.get('analysis', {}).pop('fleet_twin', None)
    parts['project.json'] = json.dumps(raw).encode()
    with zipfile.ZipFile(path, 'w') as z:
        for n, b in parts.items():
            z.writestr(n, b)
    q = Project.load(path)
    assert FT.slot(q) == {} and FT.settings_of(q) == FT.FleetTwinSettings()
    res = FT.run(q, None, FT.FleetTwinSettings(limit_source='fleet'))
    assert res is not None and res.cells
    q.analysis.fleet_twin = {'settings': 'junk', 'observations': [1, {}, {'tag': 'F', 'date': 'x'}], 'watch': 5,
                             'events': [None], 'seen': []}
    assert FT.settings_of(q) == FT.FleetTwinSettings() and FT.observations(q) == [] and FT.watches(q) == {}
    assert FT.run(q, None, FT.FleetTwinSettings(limit_source='fleet')) is not None
    st = FT.FleetTwinSettings.from_dict({'horizon_days': 1e9, 'risk': -1, 'glr_window': 3.7, 'pool': 'x',
                                         'as_of': 'not a date', 'false_alarm': float('nan')})
    assert st.horizon_days == 3650 and st.risk == 0.001 and st.glr_window == 4 and st.pool == 'sensor'
    assert st.as_of == '' and st.false_alarm == 1e-4
    # stored observations survive a save and a load
    v = FL.vehicles(q)[0]
    FT.add_observations(q, FT.online_observations(_extrinsics_doc(), v['id']))
    q.analysis.fleet_twin['settings'] = {}
    q.save(path)
    assert len(FT.observations(Project.load(path))) == 1


def test_empty_cases():
    p = rig_project()
    res = FT.run(p, None, FT.FleetTwinSettings())
    assert res.cells == {} and any(f['kind'] == 'data' for f in res.findings) and FT.exit_code(res) == 0
    assert FT.files(res, p)
    v = next(x for x in p.variants if x.sensors)
    for o in list(v.cameras) + list(v.sensors):
        o.visible = False
    assert any(f['kind'] == 'plan' for f in FT.run(p, None, FT.FleetTwinSettings()).findings)
    p = fleet_of(rig_project(), n=1, records=1, tags=('F',))                     # one record: a level, no rate
    res = FT.run(p, None, FT.FleetTwinSettings(limit_source='fleet'))
    assert all(c['observations'] == 1 for c in res.cells.values())
    assert FT.run(p, None, FT.FleetTwinSettings(), cancel=lambda: True) is None


# =========================================================================== the watch folder
@pytest.fixture(scope='module')
def drive(tmp_path_factory):
    """A 2 s synthetic nuScenes drive of the data rig (F, MR, QR; core.dataset, as test_online_calib makes them)."""
    sys.path.insert(0, os.path.dirname(__file__))
    from test_online_calib import make_project, segment
    from vantage.core.analysis import Engine
    base = tmp_path_factory.mktemp('twin_drive')
    eng = Engine()
    nd = segment(str(base / 'src' / 'drive_a'), eng, None, dur=2.0)
    p, v = make_project()
    p.fleet = {}
    p.analysis.online_calib = {'window_s': 1.0, 'stride_s': 1.0}
    return {'nd': nd, 'src': os.path.dirname(nd), 'p': p, 'eng': eng, 'base': base}


def test_watch_folder_adds_observations_once_per_new_drive(drive):
    p = copy.deepcopy(drive['p'])
    veh = FL.add_vehicle(p.fleet, 'V1')
    folder = str(drive['base'] / 'watch')
    os.makedirs(folder, exist_ok=True)
    FT.set_watch(p, veh['id'], folder)
    FT.set_settings(p, watch_quiet_s=5.0)
    now = [100.0]
    ws = FT.WatchState(clock=lambda: now[0])
    assert ws.poll(p) == []
    shutil.copytree(drive['src'], os.path.join(folder, 'drive_a'))              # a simulator's folder: drive/nuscenes
    assert ws.poll(p) == []                                                      # just arrived: wait until quiet
    now[0] += 2.0
    assert ws.poll(p) == []
    now[0] += 4.0
    ready = ws.poll(p)
    assert len(ready) == 1 and ready[0][1].endswith(os.path.join('drive_a', 'nuscenes'))
    reps = FT.watch_once(p, ws, drive['eng'])
    assert len(reps) == 1 and reps[0][1]['status'] == 'ok'
    obs = FT.observations(p)
    assert {o['tag'] for o in obs} == {'F', 'MR', 'QR'} and all(o['vehicle'] == veh['id'] for o in obs)
    assert obs[0]['date'].startswith('2026-01-01')                               # the drive's own clock
    now[0] += 10.0
    assert FT.watch_once(p, ws, drive['eng']) == [] and len(FT.observations(p)) == len(obs)   # once per drive
    shutil.copytree(drive['src'], os.path.join(folder, 'drive_b'))
    now[0] += 1.0
    ws.poll(p)
    now[0] += 6.0
    assert len(FT.watch_once(p, ws, drive['eng'])) == 1 and len(FT.observations(p)) == 2 * len(obs)
    assert len(FT.seen(p)[veh['id']]) == 2
    FT.set_watch(p, veh['id'], folder, enabled=False)
    shutil.copytree(drive['src'], os.path.join(folder, 'drive_c'))
    now[0] += 100.0
    ws.poll(p)
    assert ws.poll(p) == []                                                      # switched off
    res = FT.run(p, drive['eng'], FT.FleetTwinSettings(limit_source='fleet'))
    c = res.cells[(veh['id'], 'F')]
    assert set(c['axes']) <= {'yaw', 'pitch', 'roll'} and c['observations'] == 2


# =========================================================================== the page
from test_gui import app, win, pump, settle, _wait   # noqa: E402,F401  (fixtures)


def _open(win, name, cls):
    if name not in win._page_widgets:
        pg = cls(win.ctx)
        win.pages.addWidget(pg)
        win._page_widgets[name] = pg
    win.set_page(name)
    return win._page_widgets[name]


def test_page_runs_shows_edits_imports_watches_exports_and_undoes(app, win, tmp_path):
    from vantage.ui.workspaces.fleet_twin_ws import FleetTwinPage
    store = win.ctx.store
    p = store.project
    v = next(x for x in p.variants if x.sensors)

    def fn(pr):
        pr.active_uid = v.uid
        pr.fleet = {}
        fleet_of(pr, n=4, records=7, knock=('V2', 'F', 4, [0, 0, 0, 0.0, 0.7, 0.0]), tags=('F', 'ML', 'L1'))
    store.edit(['fleet', 'variants'], fn, 'test fleet')
    pg = _open(win, 'fleet_twin', FleetTwinPage)
    pg.activate()
    _wait(app, lambda: pg.result is not None and not pg.busy, 120)
    res = pg.result
    assert len(res.cells) == 4 * 3 and pg.table.rowCount() == 4 and pg.table.columnCount() == len(res.plan)
    assert any(k['name'] == 'V2' and k['tag'] == 'F' for k in res.knocks)
    assert 'knock' in pg.headline.text()
    # the selected cell is the knocked one (most urgent first, knocks first): its chart paints
    pg.chart.grab()
    vid2 = next(x['id'] for x in res.vehicles if x['name'] == 'V2')
    r = [x['id'] for x in res.vehicles].index(vid2)
    col = [x['tag'] for x in res.plan].index('F')
    pg.table.setCurrentCell(r, col)
    pg._cell_clicked(r, col)
    assert pg.sel == (vid2, 'F') and pg.chart.axis == 'pitch'
    i = pg.axis_box.findData('yaw')
    pg.axis_box.setCurrentIndex(i)
    pg._axis_picked(i)
    assert pg.chart.axis == 'yaw' and pg.rates.row['axis'] == 'yaw'
    pg.chart.grab()
    pg.rates.grab()
    assert pg.sched.rowCount() == len(res.schedule) and pg.findings
    # a setting typed: one undo step, the analysis runs again
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    f = pg.fields['horizon_days']
    QTest.mouseClick(f.lineEdit(), Qt.MouseButton.LeftButton)
    QTest.keyClick(f.lineEdit(), Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
    QTest.keyClicks(f.lineEdit(), '500')
    QTest.keyClick(f.lineEdit(), Qt.Key.Key_Return)
    pump(app, 0.2)
    assert FT.settings_of(store.project).horizon_days == 500
    _wait(app, lambda: pg.result is not None and pg.result.settings.horizon_days == 500 and not pg.busy, 120)
    store.undo.undo()
    pump(app, 0.2)
    assert FT.settings_of(store.project).horizon_days == 365
    _wait(app, lambda: pg.result.settings.horizon_days == 365 and not pg.busy, 120)
    # import an online self-calibration for the selected vehicle, undo it
    pg.sel_vehicle = vid2
    path = str(tmp_path / 'oc.json')
    json.dump({'schema': 'vantage.online_calib/1', 'extrinsics': _extrinsics_doc()}, open(path, 'w'))
    assert pg.import_online(path) == 1 and len(FT.observations(store.project)) == 1
    store.undo.undo()
    assert FT.observations(store.project) == []
    # a watch folder (empty): set, polled, removed with undo
    folder = str(tmp_path / 'watch')
    os.makedirs(folder)
    assert pg.set_watch_folder(folder)
    assert FT.watches(store.project)[vid2]['folder'] == folder and pg.poll_timer.isActive()
    assert pg.poll_watch() == []
    store.undo.undo()
    assert FT.watches(store.project) == {}
    pump(app, 0.2)
    assert not pg.poll_timer.isActive()
    # rig health from the live streams (a fake reader: the body edge of F 30 px off)
    store.edit('streams', lambda pr: pr.streams.update({'F': 'synthetic'}), 'stream')
    pg._sync_watch()
    assert pg.b_health.isEnabled()
    assert pg.read_rig_health(0.1, run_for=lambda pr, cams, s: {'cameras': {'F': {'drift_px': 30.0}}})
    _wait(app, lambda: not pg.health_busy, 30)
    assert [e['tag'] for e in FT.events(store.project)] == ['F']
    store.undo.undo()
    assert FT.events(store.project) == []
    # exports and the tolerance hand-off
    _wait(app, lambda: pg.result is not None and not pg.busy and not pg.stale, 120)
    paths = pg.export(str(tmp_path / 'out'))
    assert len(paths) == 5 and all(os.path.exists(x) for x in paths)
    sched = pg.export_schedule(str(tmp_path / 'sched.csv'))
    assert open(sched).readline().startswith('recalibrate_by')
    before = (store.project.analysis.tol_deg, store.project.analysis.tol_mm)
    deg, mm = pg.use_tolerance()
    assert (store.project.analysis.tol_deg, store.project.analysis.tol_mm) == (deg, mm) and deg < before[0]
    store.undo.undo()
    assert (store.project.analysis.tol_deg, store.project.analysis.tol_mm) == before
    # the project replaced while the page shows
    q = samples.tigor_sample_project()
    store.set_project(q)
    pump(app, 0.3)
    assert pg.result is None or pg.result.cells == {}
    _wait(app, lambda: not pg.busy, 60)
