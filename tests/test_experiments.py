"""Designed experiments (core/experiments.py, Analysis ▸ Experiments): the designs (factorial, one at a time, Latin
hypercube, Sobol/Saltelli, Morris, Monte Carlo), the estimators against analytic references (Ishigami's Sobol indices,
Morris on linear and interacting functions, the bootstrap intervals' coverage), a real experiment on the sample checked
against direct evaluation, factors of every kind, the cache and the journal, exports, the batch helper and the page."""
import copy
import json
import math
import os

import numpy as np
import pytest

from vantage.core import experiments as EX
from vantage.core import samples
from vantage.core.analysis import Engine

A_ISH, B_ISH = 7.0, 0.1


def _ishigami(v):
    x1, x2, x3 = v['X1'], v['X2'], v['X3']
    return math.sin(x1) + A_ISH * math.sin(x2) ** 2 + B_ISH * x3 ** 4 * math.sin(x1)


def _ishigami_indices():
    """Ishigami and Homma (1990): the analytic first-order and total indices for x_i ~ U(−π, π)."""
    a, b, pi = A_ISH, B_ISH, math.pi
    V = a * a / 8 + b * pi ** 4 / 5 + b * b * pi ** 8 / 18 + 0.5
    V1 = 0.5 * (1 + b * pi ** 4 / 5) ** 2
    V2 = a * a / 8
    V13 = b * b * pi ** 8 * (1 / 18 - 1 / 50)
    return [V1 / V, V2 / V, 0.0], [(V1 + V13) / V, V2 / V, V13 / V]


def _params(n, lo=-math.pi, hi=math.pi, **kw):
    return [EX.Factor.from_dict(dict({'id': f'X{i + 1}', 'kind': 'param', 'attr': 'value', 'lo': lo, 'hi': hi}, **kw))
            for i in range(n)]


@pytest.fixture(scope='module')
def engine():
    return Engine()


@pytest.fixture()
def proj():
    EX.clear_session_cache()
    return samples.tigor_sample_project()


def _rig(p):
    v = next(v for v in p.variants if v.sensors)
    p.active_uid = v.uid
    return v


# ============================================================================ estimators against references
def test_ishigami_sobol_indices_within_their_intervals():
    """Saltelli/Jansen estimators on the Ishigami function: every analytic index inside its 95 % bootstrap interval,
    the estimates within 0.01 of it, and S1 ≤ ST."""
    s1, st_ = _ishigami_indices()
    assert s1[0] == pytest.approx(0.3139, abs=1e-4) and st_[2] == pytest.approx(0.2437, abs=1e-4)
    st = EX.ExperimentSettings(design='sobol', sobol_n=4096, seed=11, bootstrap=400)
    res = EX.run_function(_ishigami, _params(3), st)
    assert res.n == 4096 * 5 and res.design.block(0) == 'A' and res.design.block(4096) == 'B'
    assert res.design.block(3 * 4096) == 'AB2'
    s = res.analysis['responses']['y']['sobol']
    assert s['N'] == 4096 and s['dropped'] == 0
    for i in range(3):
        assert s['S1'][i] == pytest.approx(s1[i], abs=0.01)
        assert s['ST'][i] == pytest.approx(st_[i], abs=0.01)
        assert s['S1_ci'][i][0] <= s1[i] <= s['S1_ci'][i][1]
        assert s['ST_ci'][i][0] <= st_[i] <= s['ST_ci'][i][1]
        assert s['S1'][i] <= s['ST'][i] + 0.01
    # the variance is the function's own
    assert s['V'] == pytest.approx(A_ISH ** 2 / 8 + B_ISH * math.pi ** 4 / 5 + B_ISH ** 2 * math.pi ** 8 / 18 + 0.5,
                                   rel=0.03)


def test_sobol_intervals_cover_the_analytic_indices_over_seeds():
    """The bootstrap intervals are honest: over 16 seeds of a small design (N = 256) the analytic Ishigami indices
    fall inside them at least 85 % of the time (nominal 95 %; percentile intervals on a quasi-random sequence are
    conservative), and their width shrinks like 1/√N."""
    s1, st_ = _ishigami_indices()
    hit = tot = 0
    widths = {}
    for N in (256, 1024):
        for seed in range(16 if N == 256 else 4):
            res = EX.run_function(_ishigami, _params(3), EX.ExperimentSettings(design='sobol', sobol_n=N, seed=seed,
                                                                               bootstrap=200))
            s = res.analysis['responses']['y']['sobol']
            w = np.mean([s['ST_ci'][i][1] - s['ST_ci'][i][0] for i in range(3)])
            widths.setdefault(N, []).append(w)
            if N == 256:
                for i in range(3):
                    hit += s['S1_ci'][i][0] <= s1[i] <= s['S1_ci'][i][1]
                    hit += s['ST_ci'][i][0] <= st_[i] <= s['ST_ci'][i][1]
                    tot += 2
    assert hit / tot >= 0.85
    ratio = np.mean(widths[256]) / np.mean(widths[1024])
    assert 1.5 < ratio < 2.7                       # √4 = 2


def test_sobol_degenerate_cases():
    # a constant response: no index, said so
    res = EX.run_function(lambda v: 3.0, _params(2), EX.ExperimentSettings(design='sobol', sobol_n=64))
    s = res.analysis['responses']['y']['sobol']
    assert s['S1'] == [None, None] and 'does not vary' in s['note']
    # missing values: the rows are left out, the estimate stays right for an additive function
    f = lambda v: float('nan') if v['X1'] > 2.8 else 2 * v['X1'] + v['X2']      # noqa: E731
    res = EX.run_function(f, _params(2), EX.ExperimentSettings(design='sobol', sobol_n=1024, seed=3))
    s = res.analysis['responses']['y']['sobol']
    assert 0 < s['dropped'] < 1024 * 0.25 and s['N'] + s['dropped'] == 1024
    assert s['S1'][0] == pytest.approx(0.8, abs=0.06) and s['S1'][1] == pytest.approx(0.2, abs=0.06)
    assert res.analysis['responses']['y']['missing'] > 0


def test_latin_hypercube_stratifies_every_factor():
    for n in (7, 50):
        res = EX.run_function(lambda v: v['X1'], _params(4, 0.0, 1.0), EX.ExperimentSettings(design='lhs', samples=n,
                                                                                            seed=5))
        U = res.design.U
        assert U.shape == (n, 4)
        for j in range(4):
            assert sorted((U[:, j] * n).astype(int)) == list(range(n))       # one point per stratum
        # values follow the unit coordinates through the range
        X = res.X()
        assert np.allclose(X, U)
    a = EX.make_design(_params(3), EX.ExperimentSettings(design='lhs', samples=20, seed=1)).U
    b = EX.make_design(_params(3), EX.ExperimentSettings(design='lhs', samples=20, seed=1)).U
    c = EX.make_design(_params(3), EX.ExperimentSettings(design='lhs', samples=20, seed=2)).U
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_morris_on_known_functions():
    # linear: every elementary effect is the coefficient × the range; σ = 0; μ keeps the sign
    fs = _params(3, 0.0, 1.0)
    fs[1].hi = 2.0
    lin = lambda v: 2 * v['X1'] - 3 * v['X2'] + 0 * v['X3']                  # noqa: E731
    res = EX.run_function(lin, fs, EX.ExperimentSettings(design='morris', morris_r=12, morris_levels=4, seed=2))
    mo = res.analysis['responses']['y']['morris']
    assert res.n == 12 * 4 and mo['r'] == 12 and mo['effects'] == [12, 12, 12]
    assert mo['mu_star'] == pytest.approx([2.0, 6.0, 0.0], abs=1e-9)
    assert mo['mu'] == pytest.approx([2.0, -6.0, 0.0], abs=1e-9)
    assert max(mo['sigma']) < 1e-9
    # each trajectory moves one factor at a time by ±Δ on the grid
    U, d = res.design.U, res.design.info['delta']
    assert d == pytest.approx(4 / 6)
    for t in range(12):
        T = U[t * 4:(t + 1) * 4]
        steps = np.abs(np.diff(T, axis=0))
        assert ((steps > 1e-12).sum(axis=1) == 1).all() and np.allclose(steps.max(axis=1), d)
        assert sorted(np.argmax(steps, axis=1)) == [0, 1, 2]
        assert np.allclose(T * 3, np.round(T * 3))                           # on the p = 4 grid
    # an interaction: σ > 0 for both factors in it, the bystander 0
    res = EX.run_function(lambda v: v['X1'] * v['X2'], _params(3, 0.0, 1.0),
                          EX.ExperimentSettings(design='morris', morris_r=30, seed=4))
    mo = res.analysis['responses']['y']['morris']
    assert mo['sigma'][0] > 0.1 and mo['sigma'][1] > 0.1 and mo['mu_star'][2] == 0.0
    # the Sobol G function (Saltelli et al. 2008): μ* ranks the factors by their importance a = 0, 1, 9, 99
    a = [0.0, 1.0, 9.0, 99.0]
    g = lambda v: float(np.prod([(abs(4 * v[f'X{i + 1}'] - 2) + a[i]) / (1 + a[i]) for i in range(4)]))   # noqa: E731
    res = EX.run_function(g, _params(4, 0.0, 1.0), EX.ExperimentSettings(design='morris', morris_r=40, seed=1))
    ms = res.analysis['responses']['y']['morris']['mu_star']
    assert ms[0] > ms[1] > ms[2] > ms[3]
    assert [f for f, _, _ in EX.ranking(res, 'y')] == ['X1', 'X2', 'X3', 'X4']


def test_uncertainty_intervals_cover_a_linear_function():
    """Monte Carlo through y = 2 x1 + 3 x2 with x ~ N(0, 1): mean 0, sd √13; over 80 seeds the bootstrap intervals of
    the mean and the sd, and the Wilson interval of P(y ≤ 0) = 0.5, cover the truth about 95 % of the time."""
    fs = _params(2, -10.0, 10.0, dist='normal', mean=0.0, sd=1.0)
    resp = [{'key': 'y', 'label': 'y', 'better': '<=', 'op': '<=', 'target': 0.0}]
    cov = {'mean': 0, 'sd': 0, 'meets': 0}
    seeds = 80
    for seed in range(seeds):
        res = EX.run_function(lambda v: 2 * v['X1'] + 3 * v['X2'], fs,
                              EX.ExperimentSettings(design='montecarlo', samples=200, seed=seed, bootstrap=300),
                              responses=resp)
        u = res.analysis['responses']['y']['uncertainty']
        cov['mean'] += u['mean_ci'][0] <= 0.0 <= u['mean_ci'][1]
        cov['sd'] += u['sd_ci'][0] <= math.sqrt(13) <= u['sd_ci'][1]
        cov['meets'] += u['meets_ci'][0] <= 0.5 <= u['meets_ci'][1]
    for k, c in cov.items():
        assert 0.86 <= c / seeds <= 1.0, (k, c)
    # a big run converges on the analytic figures; the convergence record ends at every point
    res = EX.run_function(lambda v: 2 * v['X1'] + 3 * v['X2'], fs,
                          EX.ExperimentSettings(design='montecarlo', samples=20000, seed=1, bootstrap=100),
                          responses=resp)
    u = res.analysis['responses']['y']['uncertainty']
    assert abs(u['mean']) < 0.08 and u['sd'] == pytest.approx(math.sqrt(13), rel=0.02)
    assert u['p95'] == pytest.approx(1.6449 * math.sqrt(13), rel=0.03) and u['meets'] == pytest.approx(0.5, abs=0.015)
    conv = u['convergence']
    assert conv[-1]['n'] == 20000 and conv[-1]['mean'] == pytest.approx(u['mean'])
    assert conv[-1]['half'] < conv[0]['half']
    assert EX.wilson(0, 10)[0] == 0.0 and EX.wilson(10, 10)[1] == pytest.approx(1.0)


def test_distributions_levels_and_designs():
    f = EX.Factor.from_dict({'id': 'X1', 'kind': 'param', 'attr': 'value', 'lo': 1.0, 'hi': 3.0, 'dist': 'truncnormal',
                             'mean': 2.5, 'sd': 1.0})
    u = np.linspace(0.0005, 0.9995, 2000)
    x = np.array(f.values(u, True, 2.0))
    assert x.min() >= 1.0 and x.max() <= 3.0 and np.all(np.diff(x) >= -1e-12)
    from scipy.stats import truncnorm
    assert x.mean() == pytest.approx(truncnorm.mean(-1.5, 0.5, loc=2.5, scale=1.0), abs=0.01)
    assert f.value(0.5, True, 0) == pytest.approx(truncnorm.ppf(0.5, -1.5, 0.5, loc=2.5, scale=1.0))
    assert f.values([np.nan], True, 'nom') == ['nom'] and f.value(float('nan'), True, 7.0) == 7.0
    # ranges for the grid designs whatever the distribution
    assert f.values([0.0, 0.5, 1.0], False, 0) == [1.0, 2.0, 3.0]
    # categorical: equal shares
    c = EX.Factor.from_dict({'id': 'X2', 'kind': 'param', 'attr': 'level', 'choices': ['a', 'b', 'c', 'a', '']})
    assert c.choices == ['a', 'b', 'c'] and c.categorical
    assert c.values([0.0, 0.34, 0.99, 0.5, 1.0], True, 'z') == ['a', 'b', 'c', 'b', 'c']
    assert c.values(np.linspace(0, 1, 3), False, 'z') == ['a', 'b', 'c']
    # counts of every design
    fs = [f, c]
    st = EX.ExperimentSettings(levels=4)
    for design, n in (('factorial', 12), ('oat', 7), ('lhs', 64), ('montecarlo', 64), ('sobol', 128 * 4),
                      ('morris', 30)):
        st.design = design
        d = EX.make_design(fs, st)
        assert d.n == n == EX.point_count(fs, st), design
    st.design = 'factorial'
    vals = EX.design_values(EX.make_design(fs, st), fs, [2.0, 'a'])
    assert sorted(set(v[0] for v in vals)) == pytest.approx([1.0, 1 + 2 / 3, 1 + 4 / 3, 3.0])
    assert {v[1] for v in vals} == {'a', 'b', 'c'}
    st.design = 'oat'
    d = EX.make_design(fs, st)
    vals = EX.design_values(d, fs, [2.0, 'a'])
    assert [v[1] for v in vals[:4]] == ['a'] * 4 and [v[0] for v in vals[4:]] == [2.0] * 3
    assert d.block(0) == 'X1' and d.block(5) == 'X2'
    assert EX.design_text(fs, st) == '7 points = 4 + 3'
    with pytest.raises(ValueError):
        EX.make_design([], st)
    st = EX.ExperimentSettings(design='factorial', levels=21)
    many = _params(5)
    assert EX.point_count(many, st) > EX.MAX_POINTS
    with pytest.raises(ValueError, match='more than'):
        EX.make_design(many, st)


def test_pareto_front_and_best():
    Y = np.array([[1.0, 5.0], [2.0, 3.0], [3.0, 3.0], [2.0, 6.0], [4.0, 1.0], [np.nan, 0.0], [1.0, 5.0]])
    # minimise column 0, maximise column 1
    front = EX.pareto_front(Y, [False, True])
    assert list(front) == [0, 3, 6]                                         # equal rows both stay
    assert list(EX.pareto_front(Y, [False, False])) == [0, 1, 4, 6]
    rng = np.random.default_rng(0)
    Z = rng.random((400, 3))
    mx = [True, False, True]
    C = Z * np.where(mx, -1, 1)
    brute = [i for i in range(len(Z)) if not any((C[j] <= C[i]).all() and (C[j] < C[i]).any() for j in range(len(Z)))]
    assert list(EX.pareto_front(Z, mx)) == brute
    # the best point: on the front, best on the first response among those meeting the targets
    fs = _params(2, 0.0, 1.0)
    resp = [{'key': 'cost', 'better': '<=', 'op': '<=', 'target': 1.2}, {'key': 'gain', 'better': '>='}]
    res = EX.run_function(lambda v: {'cost': v['X1'] + v['X2'], 'gain': v['X1'] * 2}, fs,
                          EX.ExperimentSettings(design='factorial', levels=5), responses=resp)
    b = res.analysis['best']
    assert b['index'] in res.analysis['pareto'] and b['meets_targets']
    assert b['values']['cost'] == pytest.approx(0.0)
    assert res.analysis['feasible'] == sum(1 for v in res.values if v[0] + v[1] <= 1.2 + 1e-9)
    assert res.analysis['best_per_response']['gain']['value'] == pytest.approx(2.0)


# ============================================================================ factors on the project
def test_sample_pitch_by_lens_against_direct_evaluation(proj, engine):
    """A real small experiment: camera F's pitch × its lens → blind ground, every point equal to the coverage
    analysis run directly on a copy of the project set by hand; the project itself untouched."""
    p = proj
    v = p.variant
    F = v.by_tag('F')
    before = json.dumps(p.to_json()['variants'], sort_keys=True)
    f1 = EX.new_factor(p, v, 'camera', F.uid, 'pitch')
    f1.update(lo=20.0, hi=50.0, levels=3)
    f2 = EX.new_factor(p, v, 'camera', F.uid, 'lens', [f1['id']])
    assert f2['choices'][0] == F.lens_id and len(f2['choices']) >= 2
    st = EX.ExperimentSettings.from_dict({'factors': [f1, f2], 'design': 'factorial',
                                          'responses': [{'metric': 'coverage.blind_m2'},
                                                        {'metric': 'coverage.min_density'}]})
    assert EX.validate(p, st) == []
    res = EX.run(p, engine, st, cache=False, workers=1)
    assert res.n == 3 * len(f2['choices']) and res.labels == ['F pitch', 'F lens']
    assert json.dumps(p.to_json()['variants'], sort_keys=True) == before
    for i in range(res.n):
        q = p.snapshot()
        c = q.variant.by_tag('F')
        c.pitch, c.lens_id = res.values[i]
        direct = engine.analyse(q).metrics
        assert res.Y[i, 0] == pytest.approx(direct['blind_m2'], abs=1e-12)
        assert res.Y[i, 1] == pytest.approx(direct['min_density'], abs=1e-9)
    nominal = engine.analyse(p).metrics
    assert res.base['y'][0] == pytest.approx(nominal['blind_m2'], abs=1e-12)
    assert res.nominal == [F.pitch, F.lens_id]
    # the narrow lens leaves more ground blind at every pitch
    a = res.analysis['responses']['coverage.blind_m2']
    lens_me = a['main_effects'][1]['rows']
    by = {r['level']: r['mean'] for r in lens_me}
    assert by['h03r2'] > by['h05r2']
    assert res.analysis['best']['factors']['F2'] == 'h05r2'
    s = EX.summary(res)
    assert s['responses'][0]['drivers'][0]['factor'] == 'F lens' and s['points'] == res.n


def test_mirror_sensor_offset_roi_weather_and_load_case(proj, engine):
    p = proj
    v = p.variant
    ML, MR = v.by_tag('ML'), v.by_tag('MR')
    assert ML.link_uid == MR.uid
    f = EX.Factor.from_dict(EX.new_factor(p, v, 'camera', ML.uid, 'yaw'))
    assert f.mirror and EX.factor_label(f, p, v) == 'ML+MR yaw'
    q, pose, _ = EX.apply(p, v.uid, [f], [ML.yaw + 4.0])
    assert pose is None and q.variant.by_tag('ML').yaw == pytest.approx(ML.yaw + 4.0)
    assert q.variant.by_tag('MR').yaw == pytest.approx(-(ML.yaw + 4.0))
    assert p.variant.by_tag('ML').yaw == ML.yaw                              # the project is not changed
    fo = EX.Factor.from_dict(dict(EX.new_factor(p, v, 'camera', ML.uid, 'y'), mirror=True))
    assert fo.mode == 'offset'
    q, _, _ = EX.apply(p, v.uid, [fo], [0.02])
    assert q.variant.by_tag('ML').y == pytest.approx(ML.y + 0.02) and q.variant.by_tag('MR').y == pytest.approx(
        MR.y - 0.02)
    # every camera's z offset (a mount height offset)
    fz = EX.Factor.from_dict(EX.new_factor(p, v, 'cameras', '', 'z'))
    q, _, _ = EX.apply(p, v.uid, [fz], [-0.03])
    assert all(c2.z == pytest.approx(c1.z - 0.03) for c1, c2 in zip(v.cameras, q.variant.cameras))
    # the ROI distance: the farther ground is seen at a lower density, as the analysis of each ROI says
    fr = EX.new_factor(p, v, 'roi', '', 'distance')
    st = EX.ExperimentSettings.from_dict({'factors': [dict(fr, lo=1.0, hi=3.0, levels=3)], 'design': 'oat',
                                          'responses': [{'metric': 'coverage.target_ok_pct'}]})
    res = EX.run(p, engine, st, cache=False, workers=1)
    assert [r[0] for r in res.values] == [1.0, 2.0, 3.0]
    for i, d in enumerate((1.0, 2.0, 3.0)):
        q = p.snapshot()
        q.roi.distance = d
        assert res.Y[i, 0] == pytest.approx(engine.analyse(q).metrics['target_ok_pct'], abs=1e-9)
    assert res.Y[0, 0] > res.Y[2, 0]
    # the weather condition: the perception range at night, as the perception study gives it for that condition
    from vantage.core import perception as PR
    fw = EX.new_factor(p, v, 'weather', '', 'condition')
    assert 'clear' in fw['choices'] and 'night' in fw['choices']
    st = EX.ExperimentSettings.from_dict({'factors': [dict(fw, choices=['clear', 'night'])], 'design': 'factorial',
                                          'responses': [{'metric': 'perception.range',
                                                         'params': {'cls': 'Pedestrian', 'sector': 'front'}}]})
    res = EX.run(p, engine, st, cache=False, workers=1)
    assert res.values == [['clear'], ['night']] and res.Y[1, 0] < res.Y[0, 0]
    for i, cond in enumerate(('clear', 'night')):
        pr = PR.analyse(p, engine, None, PR.PerceptionSettings.from_dict(p.analysis.perception), weather=cond)
        front = next(s_ for s_ in pr.sectors if s_['name'] == 'front')
        assert res.Y[i, 0] == pytest.approx(front['classes']['Pedestrian']['min_m'], abs=1e-9)
    # a load case moves the body in the ground analysis exactly as Analysis ▸ Load cases does
    from vantage.core import loadcases as LC
    fl = EX.new_factor(p, v, 'load_case', '', 'case')
    names = [c[0] for c in LC.cases_of(p)]
    assert fl['choices'] == names
    heavy = names[-1]
    st = EX.ExperimentSettings.from_dict({'factors': [dict(fl, choices=[names[0], heavy])], 'design': 'factorial',
                                          'responses': [{'metric': 'coverage.blind_m2'},
                                                        {'metric': 'timing.blur_px'}]})
    probs = EX.validate(p, st)
    assert any('load case' in m and 'Worst motion blur' in m for s_, m in probs if s_ == 'warning')
    res = EX.run(p, engine, st, cache=False, workers=1)
    lc = {r.name: r for r in LC.evaluate(engine, p, cell=p.analysis.cell_m)}
    assert res.Y[1, 0] == pytest.approx(lc[heavy].blind_m2, abs=1e-9)
    assert res.Y[0, 0] == pytest.approx(lc[names[0]].blind_m2, abs=1e-9)
    assert res.Y[0, 1] == res.Y[1, 1]                                        # the timing does not see it
    # a sensor factor: the roof lidar raised puts fewer of its points on the vehicle
    rv = _rig(p)
    L1 = next(s for s in rv.sensors if s.tag == 'L1')
    fs = EX.new_factor(p, rv, 'sensor', L1.uid, 'z')
    assert fs['mode'] == 'offset'
    st = EX.ExperimentSettings.from_dict({'variant': rv.uid, 'factors': [dict(fs, lo=0.0, hi=0.3, levels=2)],
                                          'design': 'factorial',
                                          'responses': [{'metric': 'sensor_coverage.lidar_blocked_pct',
                                                         'params': {'lidar': 'L1'}}]})
    res = EX.run(p, engine, st, cache=False, workers=1)
    assert res.labels == ['L1 z offset'] and res.Y[1, 0] < res.Y[0, 0]


def test_cache_cancel_resume_journal_and_workers(proj, engine, tmp_path):
    p = proj
    v = p.variant
    fs = []
    for c in v.cameras:
        if c.tag in ('F', 'R', 'ML'):
            d = EX.new_factor(p, v, 'camera', c.uid, 'pitch', [f['id'] for f in fs])
            fs.append(dict(d, lo=c.pitch - 8, hi=c.pitch + 8))
    st = EX.ExperimentSettings.from_dict({'factors': fs, 'design': 'lhs', 'samples': 24, 'seed': 3,
                                          'responses': [{'metric': 'coverage.blind_m2'},
                                                        {'metric': 'coverage.multi_pct'},
                                                        {'metric': 'recording.gb_per_hour'}]})
    ref = EX.run(p, engine, st, cache=False, workers=1)
    assert ref.cached == 0 and np.isfinite(ref.Y).all()
    # the same design from several workers: the same numbers
    par = EX.run(p, engine, st, cache=False, workers=3)
    assert par.workers == 3 and np.array_equal(par.Y, ref.Y)
    # cancelled after some points: None, the points done stay in this session's cache, a rerun resumes
    cache = EX.session_cache(EX.base_key(p, v.uid))
    assert len(cache) == 0
    res = EX.run(p, engine, st, workers=1, cancel=lambda: len(cache) >= 8)
    assert res is None
    assert len(cache) == 8                                                   # the design itself and 7 points
    res = EX.run(p, engine, st, workers=1)
    assert res.cached == 7 and np.array_equal(res.Y, ref.Y)
    again = EX.run(p, engine, st, workers=2)
    assert again.cached == again.n and np.array_equal(again.Y, ref.Y)
    # a response added later: the cached points are evaluated again for it
    st2 = EX.ExperimentSettings.from_dict(dict(st.to_dict(), responses=st.responses + [{'metric': 'timing.blur_px'}]))
    res = EX.run(p, engine, st2, workers=1)
    assert res.cached == 0 and np.array_equal(res.Y[:, :3], ref.Y)
    # the journal: a new process (a fresh cache) resumes from it; other inputs start it afresh
    jr = str(tmp_path / 'points.jsonl')
    r1 = EX.run(p, engine, st, cache=EX.PointCache(), journal=jr, workers=1)
    lines = open(jr).read().splitlines()
    assert json.loads(lines[0])['schema'] == EX.JOURNAL_SCHEMA and len(lines) == 1 + 1 + r1.n
    with open(jr, 'a') as fh:
        fh.write('{"key": "cut sh')                                          # a line cut short by a kill
    r2 = EX.run(p, engine, st, cache=EX.PointCache(), journal=jr, workers=1)
    assert r2.cached == r2.n and np.array_equal(r2.Y, ref.Y)
    q = p.snapshot()
    q.analysis.min_density = 4.0                                             # other inputs
    r3 = EX.run(q, engine, st, cache=EX.PointCache(), journal=jr, workers=1)
    assert r3.cached == 0 and json.loads(open(jr).readline())['base_key'] == EX.base_key(q, v.uid)
    # the keys: the experiment's own settings do not change the inputs; an edit of the design does
    k0 = EX.base_key(p, v.uid)
    p.analysis.experiments = st.to_dict()
    assert EX.base_key(p, v.uid) == k0
    p.variant.by_tag('R').x += 0.01
    assert EX.base_key(p, v.uid) != k0
    q = p.snapshot()
    q.vehicle.mesh = copy.deepcopy(p.vehicle.mesh)
    assert EX.base_key(q, v.uid) == EX.base_key(p, v.uid)                    # the mesh's contents, not its id


def test_settings_tolerant_and_validation(proj):
    p = proj
    v = p.variant
    assert EX.settings_of(p).factors == [] and EX.settings_of(p).design == 'lhs'     # an old project
    F = v.by_tag('F')
    junk = {'name': '  Pitch study ', 'design': 'nonsense', 'samples': 1e12, 'seed': -5, 'confidence': 2,
            'bootstrap': 'many', 'workers': 99,
            'factors': [{'id': 'F1', 'kind': 'camera', 'target': F.uid, 'attr': 'pitch', 'lo': 70, 'hi': -200,
                         'levels': 2.4, 'dist': 'cauchy', 'sd': -1},
                        {'id': 'F1', 'kind': 'camera', 'target': F.uid, 'attr': 'lens', 'choices': ['h05r2', 7]},
                        {'kind': 'teleport', 'attr': 'x'}, {'kind': 'camera', 'attr': 'colour'}, 'junk', None],
            'responses': [{'metric': 'coverage.blind_m2'}, {'metric': 'coverage.blind_m2'}, {'metric': 'nope'},
                          {'metric': 'perception.range', 'params': {'cls': 'Car'}, 'target': 'x'}, 5]}
    st = EX.ExperimentSettings.from_dict(junk)
    assert st.name == 'Pitch study' and st.design == 'lhs' and st.samples == EX.MAX_POINTS and st.seed == 0
    assert st.confidence == 0.999 and st.bootstrap == 500 and st.workers == 16
    f1, f2 = (EX.Factor.from_dict(d) for d in st.factors)
    assert len(st.factors) == 2 and f2.id == 'F2'
    assert (f1.lo, f1.hi) == (-90.0, 70.0) and f1.levels == 2 and f1.dist == 'uniform' and f1.sd > 0
    assert f2.choices == ['h05r2', '7']
    assert [r['metric'] for r in st.responses] == ['coverage.blind_m2', 'perception.range']
    assert st.responses[1]['params'] == {'cls': 'Car', 'sector': ''} and st.responses[1]['target'] is None
    assert EX.ExperimentSettings.from_dict(st.to_dict()).to_dict() == st.to_dict()
    probs = EX.validate(p, st)
    errs = [m for s, m in probs if s == 'error']
    assert any("'7' is not one of the project's levels" in m for m in errs)
    # a target gone, two factors setting the same thing, a recording-only metric, a bad parameter, a param factor
    bad = EX.ExperimentSettings.from_dict({'factors': [{'id': 'F1', 'kind': 'camera', 'target': 'gone', 'attr': 'x'},
                                                       {'id': 'F2', 'kind': 'roi', 'attr': 'distance'},
                                                       {'id': 'F3', 'kind': 'roi', 'attr': 'distance'},
                                                       {'id': 'F4', 'kind': 'param', 'attr': 'value'}],
                                           'responses': [{'metric': 'drivelog.drop_pct'},
                                                         {'metric': 'perception.range', 'params': {'cls': 'Unicorn'}}]})
    errs = EX.errors(p, bad)
    assert any('F1: its camera is not in the variant' in m for m in errs)
    assert any('F3 and F2 set the same thing' in m for m in errs)
    assert any('read off recordings' in m for m in errs)
    assert any("'Unicorn' is not one of the project's" in m for m in errs)
    assert any('F4: a function' in m for m in errs)
    with pytest.raises(ValueError):
        EX.run(p, None, bad)
    assert EX.errors(p, EX.ExperimentSettings()) == ['Add a factor: a design input to vary.',
                                                     'Add a response: a figure to compute at each design point.']
    cat = EX.response_catalogue(p)
    ids = {r['id'] for r in cat}
    assert 'coverage.blind_m2' in ids and 'critical.unresolved' in ids and 'observability.worst_ratio' in ids
    assert not any(i.startswith(('drivelog.', 'fleet.', 'lidar_calib.delta')) for i in ids)
    rv = _rig(p)
    t = EX.targets(p, rv)
    assert {x['kind'] for x in t} == set(EX.PROJECT_KINDS)
    assert sum(1 for x in t if x['kind'] == 'camera') == 7 * len(rv.cameras)
    assert sum(1 for x in t if x['kind'] == 'sensor') == 7 * len(rv.sensors)


def test_extra_responses_critical_and_observability(proj, engine):
    from vantage.core import critical as CR
    p = proj
    v = p.variant
    F = v.by_tag('F')
    st = EX.ExperimentSettings.from_dict({'factors': [dict(EX.new_factor(p, v, 'camera', F.uid, 'pitch'), lo=40,
                                                           hi=60, levels=2)],
                                          'design': 'factorial',
                                          'responses': [{'metric': 'critical.unresolved'},
                                                        {'metric': 'observability.worst_deg'},
                                                        {'metric': 'observability.worst_mm'},
                                                        {'metric': 'observability.worst_ratio'}]})
    res = EX.run(p, engine, st, cache=False, workers=1)
    assert np.isnan(res.Y[:, 0]).all() and 'No critical target' in res.reasons[(0, 0)]
    assert np.isfinite(res.Y[:, 1:]).all() and (res.Y[:, 1:] > 0).all()
    # a child 3 m ahead of the bumper: F as designed (40°) resolves it, pitched down to 60° it does not
    child = CR.preset(p, 'child')
    p.analysis.critical = dict(CR.settings_of(p).to_dict(), targets=[child])
    res = EX.run(p, engine, st, cache=False, workers=1)
    direct = []
    for val in (40.0, 60.0):
        q = p.snapshot()
        q.variant.by_tag('F').pitch = val
        direct.append(sum(not r.met for r in CR.run(q, engine, envelopes=False).targets))
    assert list(res.Y[:, 0]) == direct == [0, 1]
    from vantage.core import observability as OB
    ob = OB.run(p, engine, OB.ObservabilitySettings.from_dict(dict(OB.settings_of(p).to_dict(), suggest=False,
                                                                   lidar_pairs=False, fusion=False)))
    assert res.base['y'][1] == pytest.approx(max(max(c['sigma'][k] for k in ('yaw_deg', 'pitch_deg', 'roll_deg'))
                                                 for c in ob.cameras))


def test_variant_export_spec_batch_and_estimate(proj, engine, tmp_path):
    p = proj
    v = p.variant
    ML = v.by_tag('ML')
    fs = [dict(EX.new_factor(p, v, 'camera', ML.uid, 'pitch'), lo=30, hi=60, levels=3),
          dict(EX.new_factor(p, v, 'roi', '', 'distance', ['F1']), lo=1.5, hi=2.5, levels=2)]
    st = EX.ExperimentSettings.from_dict({'name': 'Mirror pitch', 'factors': fs, 'design': 'factorial',
                                          'responses': [{'metric': 'coverage.blind_m2'},
                                                        {'metric': 'coverage.multi_pct', 'target': 40}]})
    res = EX.run(p, engine, st, cache=False, workers=1)
    i = int(np.argmax(res.Y[:, 1]))
    nv, note = EX.to_variant(res, i, p)
    assert nv.uid != v.uid and {c.uid for c in nv.cameras}.isdisjoint({c.uid for c in v.cameras})
    assert nv.by_tag('ML').pitch == res.values[i][0] and nv.by_tag('MR').pitch == res.values[i][0]
    assert nv.by_tag('ML').link_uid == nv.by_tag('MR').uid
    assert 'Not held by a variant: ROI distance' in note and nv.name == f'Mirror pitch · point {i + 1}'
    q = p.snapshot()
    q.variants.append(nv)
    q.active_uid = nv.uid
    q.roi.distance = res.values[i][1]
    assert engine.analyse(q).metrics['multi_pct'] == pytest.approx(res.Y[i, 1])
    # files: the spec, every point, the analysis, the plots
    out = tmp_path / 'exp'
    paths = EX.export(res, p, str(out))
    names = sorted(os.path.basename(x) for x in paths)
    assert names == sorted(['experiment_spec.json', 'experiment_points.csv', 'experiment_analysis.json',
                            'experiment_effects.png', 'experiment_surface.png', 'experiment_pareto.png'])
    import cv2
    for n in names:
        if n.endswith('.png'):
            img = cv2.imread(str(out / n))
            assert img is not None and img.shape[0] > 200 and img.std() > 5
    doc = json.load(open(out / 'experiment_analysis.json'))
    assert doc['schema'] == EX.SCHEMA and doc['points'] == res.n and doc['design'] == 'factorial'
    assert doc['factors'][0]['label'] == 'ML+MR pitch' and doc['responses'][1]['target'] == 40.0
    assert doc['best']['point'] == res.analysis['best']['index'] + 1 and doc['references']
    lines = open(out / 'experiment_points.csv').read().splitlines()
    assert len(lines) == 2 + res.n and lines[1].startswith('design,as designed')
    assert lines[0].split(',')[2].startswith('F1 ML+MR pitch')
    # the spec runs again: the same points, by tag when the uids differ (a copy of the project)
    spec = json.load(open(out / 'experiment_spec.json'))
    assert spec['schema'] == EX.SPEC_SCHEMA and spec['settings']['factors'][0]['tag'] == 'ML'
    st2 = EX.load_spec(str(out / 'experiment_spec.json'), p)
    assert st2.to_dict() == dict(st.to_dict(), variant=v.uid)
    q = p.snapshot()
    from vantage.core.project import new_uid
    for c in q.variant.cameras:
        c.uid = new_uid()
    st3 = EX.load_spec(spec, q)
    assert EX.Factor.from_dict(st3.factors[0]).target == q.variant.by_tag('ML').uid
    with pytest.raises(ValueError):
        EX.load_spec('[1, 2]')
    # the batch helper: writes the files, resumes from its journal, refuses a spec that does not fit
    spec_path = str(out / 'experiment_spec.json')
    doc, code = EX.batch(p, spec_path, str(tmp_path / 'b1'), engine)
    assert code == 0 and doc['points'] == res.n and doc['cached'] == 0 and 'experiment_points.csv' in doc['files']
    assert os.path.exists(tmp_path / 'b1' / 'experiment_points.jsonl')
    doc, code = EX.batch(p, spec_path, str(tmp_path / 'b1'), engine)
    assert code == 0 and doc['cached'] == res.n
    assert EX.batch(p, str(tmp_path / 'missing.json'), str(tmp_path / 'b2'))[1] == 4
    broken = dict(spec, settings=dict(spec['settings'], responses=[{'metric': 'drivelog.drop_pct'}]))
    assert EX.batch(p, broken, str(tmp_path / 'b3'))[1] == 4
    # the estimate: measured per study, its probe points cached
    EX.clear_session_cache()
    est = EX.estimate(p, engine, st)
    assert est['points'] == res.n and est['per_point_s'] > 0 and est['cached'] == 0
    assert any(s['id'] == 'coverage' for s in est['studies']) and est['heavy'] == []
    assert EX.estimate(p, engine, st)['cached'] == 2


def test_plots_for_every_design():
    fs = _params(3, 0.0, 1.0)
    f = lambda v: {'a': v['X1'] + 2 * v['X2'] ** 2, 'b': v['X3'] - v['X1']}     # noqa: E731
    resp = [{'key': 'a', 'label': 'A', 'better': '<='}, {'key': 'b', 'label': 'B', 'better': '>='}]
    import cv2
    for design, want in (('sobol', 'sensitivity'), ('morris', 'sensitivity'), ('oat', 'sensitivity'),
                         ('montecarlo', 'uncertainty'), ('lhs', 'surface')):
        res = EX.run_function(f, fs, EX.ExperimentSettings(design=design, sobol_n=64, samples=64, morris_r=6),
                              responses=resp)
        png = getattr(EX, f'{want}_png')(res)
        img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
        assert img is not None and img.shape[1] >= 900 and img.std() > 5, design
        assert EX.effects_png(res) and EX.pareto_png(res)
        doc = EX.to_doc(res)
        json.dumps(doc, allow_nan=False)                                     # no NaN in the JSON
        assert doc['design'] == design


# ============================================================================ the page
from test_gui import app, win, pump, _wait   # noqa: E402,F401 (fixtures)


def _open(win, name, cls):
    if name not in win._page_widgets:
        pg = cls(win.ctx)
        win.pages.addWidget(pg)
        win._page_widgets[name] = pg
    win.set_page(name)
    return win._page_widgets[name]


def test_page_builds_runs_picks_and_undoes(app, win, tmp_path):
    from vantage.ui.workspaces.experiments_ws import ExperimentsPage
    EX.clear_session_cache()
    pg = _open(win, 'experiments', ExperimentsPage)
    pump(app, 0.2)
    p = win.store.project
    v = p.variant
    F = v.by_tag('F')
    assert not pg.b_run.isEnabled() and pg.status.text().startswith('Add factors')
    # factors through the menu's actions, edited in the editor; each step undoable
    pg._fill_factor_menu()
    sub = next(a.menu() for a in pg.fmenu.actions() if a.text() == 'Camera F')
    next(a for a in sub.actions() if a.text() == 'F · pitch').trigger()
    pump(app, 0.1)
    st = pg.settings
    assert [f['attr'] for f in st.factors] == ['pitch'] and pg.ftable.rowCount() == 1
    assert pg.fed.isVisible() and pg.f_fields['lo'].isVisible() and not pg.f_levels.isVisible()
    pg.f_fields['lo'].set_value(25.0)
    pg.f_fields['lo'].committed.emit(25.0)
    pg.f_fields['hi'].set_value(45.0)
    pg.f_fields['hi'].committed.emit(45.0)
    pg.f_fields['levels'].set_value(3)
    pg.f_fields['levels'].committed.emit(3.0)
    f1 = EX.Factor.from_dict(pg.settings.factors[0])
    assert (f1.lo, f1.hi, f1.levels) == (25.0, 45.0, 3)
    win.store.undo.undo()
    pump(app, 0.05)
    assert EX.Factor.from_dict(pg.settings.factors[0]).levels == 0
    win.store.undo.redo()
    pump(app, 0.05)
    fid = pg.add_factor('camera', F.uid, 'lens')
    pump(app, 0.1)
    assert pg.f_levels.isVisible() and not pg.f_fields['lo'].isVisible()
    assert pg.f_levels.count() >= 2 and pg.ftable.rowCount() == 2
    checked = [pg.f_levels.item(i).data(Qt.ItemDataRole.UserRole) for i in range(pg.f_levels.count())
               if pg.f_levels.item(i).checkState() == Qt.CheckState.Checked]
    assert checked == EX.Factor.from_dict(pg.settings.factors[1]).choices
    keep = ['h05r2', 'h03r2']
    for i in range(pg.f_levels.count()):
        it = pg.f_levels.item(i)
        want = Qt.CheckState.Checked if it.data(Qt.ItemDataRole.UserRole) in keep else Qt.CheckState.Unchecked
        if it.checkState() != want:
            it.setCheckState(want)
            pump(app, 0.02)
    assert EX.Factor.from_dict(pg.settings.factors[1]).choices == keep
    # responses through the menu; a parameter through its combo
    pg._fill_response_menu()
    sub = next(a.menu() for a in pg.rmenu.actions() if a.text() == 'Ground coverage')
    next(a for a in sub.actions() if a.text().startswith('Blind ground in the ROI')).trigger()
    pump(app, 0.05)
    assert pg.add_response('perception.range')
    pump(app, 0.1)
    assert pg.rtable.rowCount() == 2 and pg._rsel == 1
    from PyQt6.QtWidgets import QComboBox
    boxes = [w for w in pg._param_boxes if isinstance(w, QComboBox)]
    assert len(boxes) == 2
    j = boxes[0].findData('Pedestrian')
    boxes[0].setCurrentIndex(j)
    boxes[0].activated.emit(j)
    pump(app, 0.05)
    assert pg.settings.responses[1]['params']['cls'] == 'Pedestrian'
    # a target typed for the perception range, then another row picked before it was committed: kept, one field
    from PyQt6.QtWidgets import QLineEdit
    ed = next(w for w in pg._param_boxes if isinstance(w, QLineEdit))
    ed.setText('35')
    pg.rtable.selectRow(0)
    pump(app, 0.1)
    assert pg.settings.responses[1]['target'] == 35.0 and pg.settings.responses[1]['op'] == '>='
    assert pg._rsel == 0 and sum(isinstance(w, QLineEdit) for w in pg._param_boxes) == 1
    assert sum(w.isVisible() for w in pg.red.findChildren(QLineEdit)) == 1
    assert 'more · ≥ 35' in pg.rtable.item(1, 1).text()
    ed = next(w for w in pg._param_boxes if isinstance(w, QLineEdit))
    ed.setText('not a number')
    ed.editingFinished.emit()
    pump(app, 0.1)
    assert pg.settings.responses[0]['target'] is None
    win.store.undo.undo()                                                    # the target again
    pump(app, 0.1)
    assert pg.settings.responses[1]['target'] is None
    # the design: a full factorial
    pg.design.setCurrentIndex(EX.DESIGNS.index('factorial'))
    pg.design.activated.emit(EX.DESIGNS.index('factorial'))
    pump(app, 0.05)
    assert pg.settings.design == 'factorial' and pg.count.text() == '6 points = 3 × 2'
    assert pg.d_fields['levels'].isVisible() and not pg.d_fields['sobol_n'].isVisible()
    assert pg.b_run.isEnabled() and not pg._probs
    # Run: estimates first (quick here, so no question), then runs as a background job
    assert pg.run()
    _wait(app, lambda: pg.result is not None and not pg.busy and not pg.est_busy, 240)
    res = pg.result
    assert res.n == 6 and 'Failed' not in pg.status.text() and not pg.stale
    assert pg.estimate is not None and 'About' in pg.cost.text()
    assert pg.tiles['points'].value == '6' and pg.points.rowCount() == 6
    assert pg.index_chart.mode == 'range' and pg.index_table.rowCount() == 2
    assert len(pg.effects.panels) == 2 and pg.heat.grid is not None and pg.heat.grid.shape == (2, 3)
    b = res.analysis['best']
    assert pg.sel == b['index'] and pg.b_variant.isEnabled()
    for i in range(6):
        q = p.snapshot()
        c = q.variant.by_tag('F')
        c.pitch, c.lens_id = res.values[i]
        assert res.Y[i, 0] == pytest.approx(win.ctx.analysis.engine.analyse(q).metrics['blind_m2'], abs=1e-12)
    # pick a point in the Pareto chart and in the points table
    pg.tabs.setCurrentIndex(3)
    pump(app, 0.1)
    pg.pareto.repaint()
    X, Y = pg.pareto._map
    other = next(i for i in range(6) if i != b['index'] and np.isfinite(res.Y[i, 0]) and np.isfinite(res.Y[i, 1]))
    from PyQt6.QtCore import QPoint
    from PyQt6.QtTest import QTest
    QTest.mouseClick(pg.pareto, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                     QPoint(int(round(X(res.Y[other, 0]))), int(round(Y(res.Y[other, 1])))))
    pump(app, 0.05)
    assert pg.sel == other or res.Y[pg.sel, 0] == res.Y[other, 0]
    pg.points.selectRow(pg._row_of[b['index']])
    pump(app, 0.05)
    assert pg.sel == b['index'] and f'Point {b["index"] + 1}' in pg.p_text.text()
    # the shown response and the surface factors
    pg.r_pick.setCurrentIndex(1)
    pg.r_pick.activated.emit(1)
    pump(app, 0.05)
    assert pg.tiles['driver'].value != '—'
    # Add as variant: one undo step
    n0 = len(p.variants)
    uid = pg.add_variant()
    pump(app, 0.1)
    assert len(win.store.project.variants) == n0 + 1 and win.store.project.active_uid == uid
    nv = win.store.project.variant
    assert nv.by_tag('F').pitch == res.values[b['index']][0] and nv.by_tag('F').lens_id == res.values[b['index']][1]
    # the experiment followed the active variant: it stays on the run's (the new one has other camera uids), so its
    # factors still find their cameras and the result is not stale
    assert pg.settings.variant == v.uid and not pg.stale and not EX.errors(win.store.project, pg.settings)
    assert win.msg_label.text().startswith('Added ') and 'design changed' not in win.msg_label.text()
    win.store.undo.undo()
    pump(app, 0.1)
    assert len(win.store.project.variants) == n0 and win.store.project.active_uid == v.uid
    assert pg.settings.variant == '' and not pg.stale
    # an edit makes the result stale (no run by itself)
    pg.set_factor(f1.id, hi=50.0)
    pump(app, 0.1)
    assert pg.stale and 'stale' in pg.status.text() and pg.result is res
    win.store.undo.undo()
    pump(app, 0.1)
    assert not pg.stale
    # export, then import the spec into a cleared experiment
    paths = pg.export(str(tmp_path / 'out'))
    assert any(x.endswith('experiment_spec.json') for x in paths)
    spec = str(tmp_path / 'out' / 'experiment_spec.json')
    pg.remove_factor(fid)
    pump(app, 0.05)
    assert len(pg.settings.factors) == 1
    assert pg.import_spec(spec)
    pump(app, 0.1)
    assert len(pg.settings.factors) == 2 and pg.settings.design == 'factorial'
    # a run in the background can be cancelled; Run again resumes from the cache
    pg.set_design(design='lhs', samples=40)
    pump(app, 0.05)
    assert pg.run(confirm=False)
    pump(app, 0.3)
    pg.cancel()
    _wait(app, lambda: not pg.busy, 120)
    assert pg.run(confirm=False)
    _wait(app, lambda: not pg.busy and pg.result is not res, 240)
    assert pg.result.n == 40 and not pg.stale
    # the summary for the assistant
    s = pg.summary()
    assert s['points'] == 40 and s['design'] == 'lhs' and s['stale'] is False
    # the project replaced: the page starts over
    win.store.set_project(samples.tigor_sample_project())
    pump(app, 0.2)
    assert pg.result is None and pg.ftable.rowCount() == 0 and not pg.b_run.isEnabled()


from PyQt6.QtCore import Qt   # noqa: E402
