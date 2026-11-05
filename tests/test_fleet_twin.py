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
import threading
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
    # the drive's clock is the simulator's sync pulse (2026-01-01 12:00, core.dataset.EPOCH_US), not the calendar:
    # dated by its newest file (local time), the day it was written
    newest = max(os.path.getmtime(os.path.join(r, n)) for r, _, ns in os.walk(os.path.join(folder, 'drive_a'))
                 for n in ns)
    when = dt.datetime.fromisoformat(min(o['date'] for o in obs)).timestamp()   # the drive's start: its end less
    assert newest - 3.5 <= when <= newest                                         # its 2 s (whole seconds)
    now[0] += 10.0
    assert FT.watch_once(p, ws, drive['eng']) == [] and len(FT.observations(p)) == len(obs)   # once per drive
    shutil.copytree(drive['src'], os.path.join(folder, 'drive_dup'))            # the same drive again (times kept)
    now[0] += 1.0
    ws.poll(p)
    now[0] += 6.0
    assert FT.watch_once(p, ws, drive['eng']) == [] and len(FT.observations(p)) == len(obs)  # read before
    time.sleep(1.1)                                                              # (file times in whole seconds)
    shutil.copytree(drive['src'], os.path.join(folder, 'drive_b'), copy_function=shutil.copyfile)  # a new drive
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


# ----------------------------------------------------------------- drives arriving (round 15): a fake writer
def _files_of(src):
    """The drive's files relative to its root, in the CARLA recorder's order (vantage.sim.record): the marker and
    the empty version folder first, the images and point clouds as the drive runs, the tables at the end, then
    dataset.json, then the bridge's run.json."""
    rel = sorted(os.path.relpath(os.path.join(r, n), src) for r, _, ns in os.walk(src) for n in ns)
    data = [r for r in rel if os.sep + 'v1.0-' not in os.sep + r and r.startswith('nuscenes' + os.sep)
            and not r.endswith(FT.DRIVE_MARKER)]
    tables = [r for r in rel if os.sep + 'v1.0-' in os.sep + r]
    return data, tables


def _copy(src, dst, r):
    os.makedirs(os.path.dirname(os.path.join(dst, r)), exist_ok=True)
    shutil.copyfile(os.path.join(src, r), os.path.join(dst, r))       # new file times, as a writer makes them


class Writer(threading.Thread):
    """Writes a drive into dst over time as the CARLA bridge does (or stops half way: kill_after = 'data' / 'tables'
    / 'dataset'). pause: seconds between phases (longer than the watch's quiet time, as a slow tick or the world
    loading); log: [(monotonic time, event)]."""

    def __init__(self, src, dst, pause=0.8, kill_after=None, status='ok', per_file=0.0):
        super().__init__(daemon=True)
        self.src, self.dst, self.pause, self.kill_after, self.status = src, dst, pause, kill_after, status
        self.per_file = per_file
        self.log = []
        self.done = threading.Event()

    def note(self, what):
        self.log.append((time.monotonic(), what))

    def run(self):
        data, tables = _files_of(self.src)
        nus = os.path.join(self.dst, 'nuscenes')
        os.makedirs(os.path.join(nus, 'v1.0-vantage'), exist_ok=True)
        open(os.path.join(nus, FT.DRIVE_MARKER), 'w').write('Written by Vantage (test writer)\n')
        self.note('prepared')
        time.sleep(self.pause)                                     # the world loading: nothing written
        for i, r in enumerate(data):
            _copy(self.src, self.dst, r)
            if i == len(data) // 2:
                time.sleep(self.pause)                             # a slow stretch of the drive
            elif self.per_file:
                time.sleep(self.per_file)
        self.note('data written')
        if self.kill_after == 'data':
            return self.note('killed')
        for r in tables:
            _copy(self.src, self.dst, r)
        self.note('tables written')
        if self.kill_after == 'tables':
            return self.note('killed')
        time.sleep(self.pause)
        doc = json.load(open(os.path.join(self.src, 'dataset.json')))
        doc['generator'] = 'carla'
        json.dump(doc, open(os.path.join(self.dst, 'dataset.json'), 'w'))
        self.note('dataset.json written')
        if self.kill_after == 'dataset':
            return self.note('killed')
        time.sleep(self.pause)
        json.dump({'schema': 'vantage.sim_run/1', 'status': self.status}, open(os.path.join(self.dst, 'run.json'), 'w'))
        self.note('run.json written')
        self.done.set()


def _watch_project(drive, folder, quiet):
    p = copy.deepcopy(drive['p'])
    veh = FL.add_vehicle(p.fleet, 'V1')
    os.makedirs(folder, exist_ok=True)
    FT.set_watch(p, veh['id'], folder)
    FT.set_settings(p, watch_quiet_s=1.0)
    return p, veh['id']


def _poll_until(p, ws, seconds, quiet, stop=None):
    """Poll every 50 ms for `seconds` (or until stop()); [(time, ready list)] of every poll that returned something."""
    got = []
    t_end = time.monotonic() + seconds
    while time.monotonic() < t_end and not (stop and stop()):
        r = ws.poll(p, quiet)
        if r:
            got.append((time.monotonic(), r))
        time.sleep(0.05)
    return got


def test_a_drive_being_written_is_never_read_and_is_read_once_when_complete(drive):
    """The recorder's order (images over time with pauses longer than the quiet time, the tables, dataset.json, then
    run.json): never ready before run.json, ready within the quiet time after it, read once, dated by its files."""
    folder = str(drive['base'] / 'arrive')
    p, vid = _watch_project(drive, folder, 0.5)
    w = Writer(drive['src'], os.path.join(folder, 'drive_live'), pause=0.8)
    ws = FT.WatchState()
    states = set()
    w.start()
    first = None
    t_end = time.monotonic() + 30
    while time.monotonic() < t_end:
        r = ws.poll(p, 0.5)
        states |= {s for (_, _), (s, _) in ws.status.items()}
        if r:
            first = (time.monotonic(), r)
            break
        time.sleep(0.05)
    assert first is not None and w.done.is_set()                                  # only after run.json
    t_run = next(t for t, e in w.log if e == 'run.json written')
    assert first[0] - t_run < 2.0                                                 # the quiet time and a poll
    assert {'arriving', 'quiet', 'incomplete'} <= states                          # it said so while waiting
    reps = FT.watch_once(p, ws, drive['eng'], quiet=0.5)
    assert len(reps) == 1 and reps[0][1]['status'] == 'ok' and reps[0][1]['observations']
    assert all(o['vehicle'] == vid for o in FT.observations(p))
    newest = max(os.path.getmtime(os.path.join(r_, n)) for r_, _, ns in os.walk(os.path.join(folder, 'drive_live'))
                 for n in ns)
    when = dt.datetime.fromisoformat(reps[0][1]['date']).timestamp()            # its end less its 2 s
    assert newest - 3.5 <= when <= newest
    n = len(FT.observations(p))
    for _ in range(5):                                                            # read once
        time.sleep(0.15)
        assert FT.watch_once(p, ws, drive['eng'], quiet=0.5) == []
    assert len(FT.observations(p)) == n and len(FT.seen(p)[vid]) == 1
    assert ws.status[(vid, os.path.join(folder, 'drive_live', 'nuscenes'))][0] == 'read'


@pytest.mark.parametrize('kill_after, status, why', [
    ('data', 'ok', 'no sample_data.json yet'),                 # the bridge killed while driving: never tables
    ('tables', 'ok', 'dataset.json not written yet'),          # killed while writing its last files
    ('dataset', 'ok', 'run.json not written yet'),
    (None, 'failed', "the run ended 'failed'"),                 # a run that failed after the tables (left over)
])
def test_a_drive_aborted_half_way_is_never_read(drive, kill_after, status, why):
    folder = str(drive['base'] / f'abort-{kill_after}-{status}')
    p, vid = _watch_project(drive, folder, 0.4)
    w = Writer(drive['src'], os.path.join(folder, 'drive_x'), pause=0.6, kill_after=kill_after, status=status)
    ws = FT.WatchState()
    w.start()
    w.join(30)
    got = _poll_until(p, ws, 2.0, 0.4)
    assert got == []
    (state, text), = ws.status.values()
    assert state == 'incomplete' and why in text


def test_slow_copy_stall_rename_and_aborted_copy(drive, tmp_path):
    """A copy that puts the tables first and stalls longer than the quiet time with images missing is not read
    until every file is there; a copy cut short (half the images, or a JPEG cut) stays incomplete; a folder renamed
    into place from a hidden or .part name is read once, and renamed again it is not read again."""
    src = drive['src']
    data, tables = _files_of(src)
    folder = str(drive['base'] / 'copies')
    p, vid = _watch_project(drive, folder, 0.4)
    ws = FT.WatchState()
    # tables first, half the images, a stall
    dst = os.path.join(folder, 'drive_slow')
    for r in tables + ['dataset.json'] + data[:len(data) // 2]:
        _copy(src, dst, r)
    shutil.copyfile(os.path.join(src, 'nuscenes', FT.DRIVE_MARKER), os.path.join(dst, 'nuscenes', FT.DRIVE_MARKER))
    assert _poll_until(p, ws, 1.2, 0.4) == []
    (state, text), = ws.status.values()
    assert state == 'incomplete' and 'missing' in text
    # a JPEG cut short at the end of the copy
    for r in data[len(data) // 2:]:
        _copy(src, dst, r)
    jpg = next(r for r in data if r.endswith('.jpg'))
    with open(os.path.join(dst, jpg), 'r+b') as fh:
        fh.truncate(os.path.getsize(os.path.join(src, jpg)) // 2)
    assert _poll_until(p, ws, 1.2, 0.4) == []
    assert 'cut short' in ws.status[(vid, os.path.join(dst, 'nuscenes'))][1]
    _copy(src, dst, jpg)                                                          # the copy finished
    got = _poll_until(p, ws, 2.0, 0.4, stop=lambda: False)
    assert got and got[0][1][0][1] == os.path.join(dst, 'nuscenes')
    shutil.rmtree(dst)
    ws.poll(p, 0.4)
    assert ws.status == {}
    # renamed into place from a hidden name and from .part: never looked at before the rename
    for tmpname in ('.incoming-drive', 'drive_r.part'):
        q = os.path.join(folder, tmpname)
        shutil.copytree(src, q)
        assert FT.drive_paths(folder, partial=True) == [] and _poll_until(p, ws, 0.6, 0.4) == []
        shutil.rmtree(q)
    q = os.path.join(folder, '.incoming-drive')
    shutil.copytree(src, q)
    os.rename(q, os.path.join(folder, 'drive_r'))
    got = _poll_until(p, ws, 1.5, 0.4, stop=lambda: bool(ws.status) and
                      next(iter(ws.status.values()))[0] == 'ready')
    reps = FT.watch_once(p, ws, drive['eng'], quiet=0.4)
    assert len(reps) == 1 and reps[0][1]['status'] == 'ok'
    os.rename(os.path.join(folder, 'drive_r'), os.path.join(folder, 'drive_r2'))   # moved again: the same drive
    time.sleep(0.5)
    for _ in range(3):
        assert FT.watch_once(p, ws, drive['eng'], quiet=0.4) == []
        time.sleep(0.3)
    assert ws.status[(vid, os.path.join(folder, 'drive_r2', 'nuscenes'))] == ('read', 'read before')
    assert len(FT.seen(p)[vid]) == 1


def test_a_drive_changing_while_it_is_read_is_not_added(drive, monkeypatch):
    from vantage.core import online_calib as OC
    folder = str(drive['base'] / 'changing')
    p, vid = _watch_project(drive, folder, 0.3)
    shutil.copytree(drive['src'], os.path.join(folder, 'drive_c'))
    ws = FT.WatchState()
    assert _poll_until(p, ws, 1.0, 0.3, stop=lambda: ws.status and next(iter(ws.status.values()))[0] == 'ready')
    real = OC.run

    def writer_still_busy(*a, **k):
        open(os.path.join(folder, 'drive_c', 'nuscenes', 'late.bin'), 'wb').write(b'x' * 10)
        raise ValueError('a table is being rewritten')
    monkeypatch.setattr(OC, 'run', writer_still_busy)
    lines = []
    assert FT.watch_once(p, ws, drive['eng'], quiet=0.3, log=lines.append) == []
    assert FT.observations(p) == [] and FT.seen(p) == {} and 'changed while it was read' in lines[-1]
    monkeypatch.setattr(OC, 'run', real)
    assert ws.poll(p, 0.3) == [] and next(iter(ws.status.values()))[0] == 'arriving'   # looked at afresh
    time.sleep(0.4)
    reps = FT.watch_once(p, ws, drive['eng'], quiet=0.3)
    assert len(reps) == 1 and reps[0][1]['status'] == 'ok'


def test_bags_complete_only_when_closed(tmp_path):
    from vantage.core import rosbag as RB
    folder = tmp_path / 'bags'
    folder.mkdir()
    b = RB.BagWriter(str(folder / 'run1'), overwrite=True)
    b.add_topic('/odom', 'nav_msgs/msg/Odometry')
    assert FT.drive_paths(str(folder)) == [str(folder / 'run1')]
    ok, why = FT.drive_complete(str(folder / 'run1'))
    assert not ok and 'metadata.yaml' in why
    b.close()
    assert FT.drive_complete(str(folder / 'run1')) == (True, '')
    mcap = next(x for x in os.listdir(folder / 'run1') if x.endswith('.mcap'))
    single = folder / 'single.mcap'
    shutil.copyfile(folder / 'run1' / mcap, single)
    assert FT.drive_complete(str(single)) == (True, '')
    data = single.read_bytes()
    single.write_bytes(data[:-20])                                                # still being written
    assert FT.drive_complete(str(single))[0] is False
    (folder / 'run1' / 'metadata.yaml').unlink()
    assert not FT.drive_complete(str(folder / 'run1'))[0]
    # a simulator's drive recorded as a bag only: <drive>/rosbag/<name>/
    os.makedirs(folder / 'sim' / 'rosbag')
    shutil.copytree(folder / 'run1', folder / 'sim' / 'rosbag' / 'drv')
    assert str(folder / 'sim' / 'rosbag' / 'drv') in FT.drive_paths(str(folder))
    assert FT.drive_root(str(folder / 'sim' / 'rosbag' / 'drv')) == str(folder / 'sim')


def test_drive_dates_calendar_clock_or_files(tmp_path):
    ex_real = {'start_ns': int(dt.datetime(2026, 9, 14, 8, 30, tzinfo=dt.timezone.utc).timestamp() * 1e9)}
    assert FT.drive_date('/nonexistent', ex_real) == '2026-09-14T08:30:00'          # a real calendar clock
    assert not FT.synthetic_clock('/nonexistent', FT.SYNC_EPOCH_US * 1000)          # no Vantage marker: calendar
    nus = tmp_path / 'drv' / 'nuscenes'                                            # a simulated drive
    nus.mkdir(parents=True)
    (nus / FT.DRIVE_MARKER).write_text('x')
    (nus / 'a.bin').write_bytes(b'1')
    os.utime(nus / 'a.bin', (1_790_000_000, 1_790_000_000))
    (nus / FT.DRIVE_MARKER).touch()
    os.utime(nus / FT.DRIVE_MARKER, (1_789_000_000, 1_789_000_000))
    ex_sim = {'start_ns': FT.SYNC_EPOCH_US * 1000 + 50_000_000}
    assert FT.synthetic_clock(str(nus), ex_sim['start_ns'])
    assert FT.drive_date(str(nus), ex_sim) == dt.datetime.fromtimestamp(1_790_000_000).isoformat(timespec='seconds')
    (tmp_path / 'drv' / 'run.json').write_text(json.dumps({'status': 'ok', 'recorded_utc': {
        'start': '2026-11-05T05:00:00+00:00'}}))                                     # the bridge says when (proposed)
    want = dt.datetime(2026, 11, 5, 5, 0, tzinfo=dt.timezone.utc).astimezone().replace(tzinfo=None)
    assert FT.drive_date(str(nus), ex_sim) == want.isoformat(timespec='seconds')


def test_headless_watch_reads_a_drive_arriving_during_the_run(drive, tmp_path, capsys):
    """--fleet-twin-watch with --watch-runs N while a fake writer writes a drive: read once, saved, the twin run."""
    from vantage import batch
    folder = str(tmp_path / 'watch')
    p, vid = _watch_project(drive, folder, 1.0)
    path = str(tmp_path / 'twin.vantage')
    p.save(path)
    w = Writer(drive['src'], os.path.join(folder, 'drive_h'), pause=1.2)
    w.start()
    rc = batch.run(batch.parser().parse_args([path, '--fleet-twin-watch', '--watch-runs', '60',
                                               '--watch-interval', '0.25']))
    assert rc == 0 and w.done.is_set()
    err = capsys.readouterr().err
    assert 'drive_h: arriving' in err and 'drive_h: incomplete' in err and 'drive_h: ok' in err
    assert err.index('drive_h: ok') > err.index('drive_h: incomplete') and err.count('drive_h: ok') == 1
    q = Project.load(path)
    assert len(FT.seen(q)[vid]) == 1 and FT.observations(q)
    assert all(o['vehicle'] == vid for o in FT.observations(q))


@pytest.mark.skipif(not os.environ.get('VANTAGE_CARLA_DRIVE'), reason='needs VANTAGE_CARLA_DRIVE: a finished CARLA '
                    'drive folder (nuscenes/, dataset.json, run.json) of the sample\'s Data rig')
def test_real_carla_drive_arriving(tmp_path):
    """A real CARLA drive written into a watch folder by the fake writer in the bridge's order (and a stall), read
    once when complete, its observations on the right vehicle, dated by its files."""
    src = os.environ['VANTAGE_CARLA_DRIVE']
    p = rig_project()
    p.analysis.online_calib = {'window_s': 2.5, 'stride_s': 2.5}
    veh = FL.add_vehicle(p.fleet, 'V9')
    folder = str(tmp_path / 'watch')
    os.makedirs(folder)
    FT.set_watch(p, veh['id'], folder)
    w = Writer(src, os.path.join(folder, 'real'), pause=1.5, per_file=0.002)
    ws = FT.WatchState()
    w.start()
    t_end = time.monotonic() + 600
    while time.monotonic() < t_end and not ws.poll(p, 1.0):
        time.sleep(0.1)
    assert w.done.is_set()                                                        # ready only after run.json
    reps = FT.watch_once(p, ws, None, quiet=1.0)
    assert len(reps) == 1 and reps[0][1]['status'] == 'ok'
    obs = FT.observations(p)
    assert obs and all(o['vehicle'] == veh['id'] for o in obs) and 'F' in {o['tag'] for o in obs}
    assert FT.watch_once(p, ws, None, quiet=1.0) == []


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


def _page_watch_run(app, win, project, vid, folder, src, quiet=1.0, timeout=600):
    """The page watching `folder` while a Writer writes `src` into it: the page's own poll timer and job; returns
    (page, writer, the status lines shown while it waited, the first time a read started)."""
    from vantage.ui.workspaces.fleet_twin_ws import FleetTwinPage
    store = win.ctx.store
    store.set_project(project)
    pump(app, 0.3)
    pg = _open(win, 'fleet_twin', FleetTwinPage)
    pg.activate()
    pg.sel_vehicle = vid
    assert pg.set_watch_folder(folder)
    store.edit('analysis', lambda pr: FT.set_settings(pr, watch_quiet_s=quiet), 'quiet')
    pg.poll_timer.setInterval(100)
    w = Writer(src, os.path.join(folder, 'drive_p'), pause=quiet + 0.5)
    w.start()
    shown = set()
    t_end = time.monotonic() + timeout
    started = None
    while time.monotonic() < t_end:
        pump(app, 0.05)
        shown.add(pg.watch_status.text())
        if pg.reading is not None and started is None:
            started = time.monotonic()
            done_at_start = w.done.is_set()
        if started is not None and pg.reading is None:
            break
    assert started is not None and done_at_start                                  # read only once it was complete
    return pg, w, shown


def test_page_watch_reads_an_arriving_drive_once(app, win, drive, tmp_path):
    p = copy.deepcopy(drive['p'])
    veh = FL.add_vehicle(p.fleet, 'V1')
    folder = str(tmp_path / 'pwatch')
    os.makedirs(folder)
    pg, w, shown = _page_watch_run(app, win, p, veh['id'], folder, drive['src'])
    store = win.ctx.store
    text = '\n'.join(shown)
    assert 'arriving' in text and 'not finished' in text                          # what it said while waiting
    obs = FT.observations(store.project)
    assert obs and {o['vehicle'] for o in obs} == {veh['id']} and len(FT.seen(store.project)[veh['id']]) == 1
    assert 'drive_p: ok' in pg.watch_status.text()
    for _ in range(10):                                                           # never read again
        pump(app, 0.1)
        assert pg.reading is None
    assert len(FT.observations(store.project)) == len(obs)
    store.undo.undo()                                                             # one undo step
    assert FT.observations(store.project) == [] and FT.seen(store.project) == {}
    pg.poll_timer.setInterval(2000)
    store.set_project(samples.tigor_sample_project())
    pump(app, 0.3)


@pytest.mark.skipif(not os.environ.get('VANTAGE_CARLA_DRIVE'), reason='needs VANTAGE_CARLA_DRIVE (a finished CARLA '
                    'drive of the sample\'s Data rig) and VANTAGE_TWIN_PROJECT (a project whose fleet has its history)')
def test_real_page_reads_an_arriving_carla_drive(app, win, tmp_path):
    """The page with a real CARLA drive arriving in the bridge's order: read once when complete, the twin runs, its
    knocks are on the page and in the Issues list."""
    p = Project.load(os.environ['VANTAGE_TWIN_PROJECT'])
    for vid in list(FT.watches(p)):
        FT.set_watch(p, vid, None)
    veh = next(v for v in FL.vehicles(p) if v['name'] == os.environ.get('VANTAGE_TWIN_VEHICLE', 'V2'))
    folder = str(tmp_path / 'watch')
    os.makedirs(folder)
    before = {o['id'] for o in FT.observations(p)}
    pg, w, shown = _page_watch_run(app, win, p, veh['id'], folder, os.environ['VANTAGE_CARLA_DRIVE'], quiet=2.0)
    store = win.ctx.store
    assert 'not finished' in '\n'.join(shown)
    new = [o for o in FT.observations(store.project) if o['id'] not in before]
    assert new and {o['vehicle'] for o in new} == {veh['id']}
    _wait(app, lambda: pg.result is not None and not pg.busy and not pg.stale, 300)
    knocks = [k for k in pg.result.knocks if k['vehicle'] == veh['id']]
    assert any(k['tag'] == 'F' for k in knocks), knocks
    win.ctx.analysis.run_now()
    _wait(app, lambda: any(i.objects == f'{veh["name"]} · F' for i in win.ctx.analysis.issues), 120)
    pg.poll_timer.setInterval(2000)
    store.set_project(samples.tigor_sample_project())
    pump(app, 0.3)
