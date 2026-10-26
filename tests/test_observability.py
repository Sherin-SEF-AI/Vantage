"""Calibration observability and camera–lidar fusion density (core.observability, core.sensor_coverage.camera_fusion)
and the Calibrate ▸ Calibration observability page.

Independent checks: the board corners against the installation check's own board model (core.install), the pixel
Jacobians against finite differences of core.install's projection and of a target moved on the floor, the relative
pose Jacobian against finite differences, the radar covariance against its closed form, the fusion density against a
hand calculation for a lidar sitting in the camera; and the predicted σ against Monte Carlo solves with the machinery
that solves the real captures: core.joint.joint_solve for the cameras (boards exact and placed by hand), the
lidar–camera calibration's own least squares (core.lidar_calib._solve) for a lidar–camera pair, a least-squares solve
for a radar; a wrong prediction must fail the same comparison."""
import copy
import json
import math
import os

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
import pytest

from vantage.core import samples, observability as OB, sensor_coverage as SC, sensors as S
from vantage.core.analysis import Engine
from vantage.core.install import inner_corners, _project as install_project
from vantage.core.mathutil import rot_zyx
from vantage.core.project import Target


# ----------------------------------------------------------------------------- fixtures and helpers
@pytest.fixture(scope='module')
def engine():
    return Engine()


@pytest.fixture(scope='module')
def sample():
    return samples.tigor_sample_project()


def _rig():
    """The sample with its data rig (6 cameras, a roof lidar, 5 radars) active."""
    p = samples.tigor_sample_project()
    v = next(v for v in p.variants if v.sensors)
    p.active_uid = v.uid
    return p, v


def _only(p, tags, sensors=None):
    """Only the cameras tagged tags (and the sensors tagged sensors) visible."""
    v = p.variant
    for c in v.cameras:
        c.visible = c.tag in tags
    if sensors is not None:
        for s in v.sensors:
            s.visible = s.tag in sensors
    return v


def _st(**kw):
    d = {'fusion': False, 'lidar_pairs': False, 'suggest': False}
    d.update(kw)
    return OB.ObservabilitySettings.from_dict(d)


# ============================================================================ settings
def test_settings_are_tolerant_and_clamped():
    st = OB.ObservabilitySettings.from_dict({'pixel_sigma': '0.7', 'board_sigma_mm': -5, 'need_deg': 'x',
                                             'mc_trials': 3.6e9, 'boards_placed': 'no', 'suggest': 0,
                                             'min_square_px': float('nan'),
                                             'reflectors': [{'name': 'A', 'x': 5, 'y': '1', 'edge_mm': 5},
                                                            {'x': 'bad'}, None, {'name': 'B', 'x': 1e9, 'y': 0}]})
    assert st.pixel_sigma == 0.7 and st.board_sigma_mm == OB.LIMITS['board_sigma_mm'][0]
    assert st.need_deg == OB.ObservabilitySettings().need_deg and st.min_square_px == 8.0
    assert st.mc_trials == OB.LIMITS['mc_trials'][1] and isinstance(st.mc_trials, int)
    assert st.boards_placed is False and st.suggest is False
    assert [r['name'] for r in st.reflectors] == ['A', 'B']
    assert st.reflectors[0]['edge_mm'] == OB.REFLECTOR_LIMITS['edge_mm'][0] and st.reflectors[1]['x'] == 200.0
    # a round trip and junk
    assert OB.ObservabilitySettings.from_dict(st.to_dict()) == st
    for junk in (None, [], 'x', {'reflectors': 'no'}):
        assert OB.ObservabilitySettings.from_dict(junk) == OB.ObservabilitySettings()
    # a corner reflector's cross-section: 4πa⁴/3λ² (a 100 mm trihedral is about 14 dBsm at 77 GHz)
    lam = 299792458.0 / 77e9
    assert OB.reflector_rcs(100.0) == pytest.approx(4 * math.pi * 0.1 ** 4 / (3 * lam ** 2))
    assert 10 * math.log10(OB.reflector_rcs(100.0)) == pytest.approx(14.4, abs=0.2)


# ============================================================================ boards
def test_board_corners_match_the_installation_check_and_the_patterns():
    t = Target(name='T', x=2.0, y=-1.0, yaw=30.0, rows=6, cols=7, square_mm=80.0)
    P, g, grid, pat = OB.board_corners(t)
    assert pat == 'checkerboard' and grid == (5, 6) and P.shape == (30, 3) and (g == 0).all()
    assert np.allclose(P, inner_corners(t).reshape(-1, 3))
    t.pattern = 'charuco'
    P2, g2, grid2, _ = OB.board_corners(t)
    assert np.allclose(P2, P) and len(np.unique(g2)) == 30
    t.pattern = 'aprilgrid'
    P3, g3, grid3, _ = OB.board_corners(t)
    assert grid3 is None and P3.shape == (6 * 7 * 4, 3) and len(np.unique(g3)) == 42
    # a tag is square / 1.3 wide and its four corners sit round the square's centre
    T = P3.reshape(42, 4, 3)
    side = np.linalg.norm(T[:, 1] - T[:, 0], axis=1)
    assert np.allclose(side, 0.08 / 1.3)
    centres = T.mean(1)
    assert np.allclose(centres.mean(0), [2.0, -1.0, 0.0])
    assert np.allclose(np.sort(np.linalg.norm(centres[:, :2] - [2.0, -1.0], axis=1))[:2], 0.04, atol=1e-9)


def test_jacobians_against_independent_finite_differences(sample):
    v = sample.variant
    c = v.by_tag('ML')
    intr = OB.resolve(sample.module_for(c), sample.lens_for(c))
    R = rot_zyx(c.yaw, c.pitch, c.roll)
    cm = OB._Cam(c, intr, R, c.pos(), OB.BODY_TO_OPTICAL @ R.T, None)
    t = next(t for t in sample.targets if t.name == 'T4')
    W = inner_corners(t).reshape(-1, 3)
    uv, Jc, Jb = OB._jacobians(cm, W, np.array([t.x, t.y, 0.0]))
    h = 1e-5

    def proj(Rx, pos, pts):
        return install_project(intr, Rx, pos, pts, 1.0, 1.0)[0].ravel()
    assert np.allclose(uv.ravel(), proj(R, c.pos(), W))
    for k in range(3):                                   # rotations about the camera's own axes
        d = np.zeros(3)
        d[k] = h
        fd = (proj(R @ OB._exp(d), c.pos(), W) - proj(R @ OB._exp(-d), c.pos(), W)) / (2 * h)
        assert np.allclose(Jc[:, k], fd, rtol=1e-5, atol=1e-3)
        e = np.zeros(3)
        e[k] = h
        fd = (proj(R, c.pos() + e, W) - proj(R, c.pos() - e, W)) / (2 * h)
        assert np.allclose(Jc[:, 3 + k], fd, rtol=1e-5, atol=1e-3)
    # the board moved and turned on the floor (the installation check's own board model)
    for k, (dx, dy, dyaw) in enumerate(((h, 0, 0), (0, h, 0), (0, 0, math.degrees(h)))):
        a, b = copy.copy(t), copy.copy(t)
        a.x, a.y, a.yaw = t.x + dx, t.y + dy, t.yaw + dyaw
        b.x, b.y, b.yaw = t.x - dx, t.y - dy, t.yaw - dyaw
        fd = (proj(R, c.pos(), inner_corners(a).reshape(-1, 3)) -
              proj(R, c.pos(), inner_corners(b).reshape(-1, 3))) / (2 * h)
        assert np.allclose(Jb[:, k], fd, rtol=1e-5, atol=1e-3)


def test_which_boards_count(sample, engine):
    res = OB.run(sample, engine, _st())
    cams = {c['tag']: c for c in res.cameras}
    assert sorted(b['name'] for b in cams['F']['boards']) == ['T1', 'T2', 'T3']
    assert sorted(b['name'] for b in cams['ML']['boards']) == ['T2', 'T4']
    assert all(c['corners'] == 25 * len(c['boards']) for c in res.cameras)
    t8 = next(b for b in res.boards if b['name'] == 'T8')
    assert [s['camera'] for s in t8['seen_by']] == ['R'] and t8['only_board_of'] == []
    assert t8['unseen']['F'] == 'out of view'
    # a board whose squares shrink below the minimum in the image drops out with that reason
    big = _st(min_square_px=30.0)
    r2 = OB.run(sample, engine, big)
    f = next(c for c in r2.cameras if c['tag'] == 'F')
    assert not f['boards'] and any('squares under 30' in u['reason'] for u in f['unseen'])
    assert f['verdict'] == 'critical' and f['sigma'] is None
    # a ChArUco board half out of view still counts with the corners in view; a checkerboard does not
    p = samples.tigor_sample_project()
    cam = p.variant.by_tag('F')
    intr = OB.resolve(p.module_for(cam), p.lens_for(cam))
    R = rot_zyx(cam.yaw, cam.pitch, cam.roll)
    cm = OB._Cam(cam, intr, R, cam.pos(), OB.BODY_TO_OPTICAL @ R.T, None)
    edge = None
    for x in np.arange(2.0, 8.0, 0.05):           # walk a board out along the image's right edge
        t = Target(name='E', x=float(x), y=-3.0, yaw=0.0, rows=6, cols=6, square_mm=100.0)
        uv, ok, _ = cm.project(OB._board(t).P)
        if 0.3 < ok.mean() < 0.7:
            edge = t
            break
    assert edge is not None
    o, why = OB._see(cm, 0, OB._board(edge), 0, OB.ObservabilitySettings(min_square_px=1.0))
    assert o is None and why == 'partly out of view'
    edge.pattern = 'charuco'
    o, why = OB._see(cm, 0, OB._board(edge), 0, OB.ObservabilitySettings(min_square_px=1.0))
    assert o is not None and 4 <= o.n < 25


# ============================================================================ the covariance
def test_estimator_covariance_known_cases(sample, engine):
    m = OB._Model(sample, engine, sample.variant, _st(pixel_sigma=0.3))
    # boards exact: σ² (JᵀJ)⁻¹ per camera, linear in the pixel noise
    C1, det = m.cov(placed=False)
    m.st = _st(pixel_sigma=0.6)
    C2, _ = m.cov(placed=False)
    assert det.all() and np.allclose(C2, 4 * C1, rtol=1e-9, atol=1e-18)
    for k in range(len(m.cams)):
        G = sum(o.Jc.T @ o.Jc for o in m.seen if o.cam == k)
        assert np.allclose(C1[6 * k:6 * k + 6, 6 * k:6 * k + 6], 0.09 * np.linalg.inv(G), rtol=1e-6)
    # boards placed, noise at the joint solve's own weight (0.5 px): the plain inverse information
    m.st = _st(pixel_sigma=OB.JOINT_PIXEL_SIGMA)
    C, _ = m.cov(placed=True)
    n = 6 * len(m.cams) + 3 * len(m.boards)
    H = np.zeros((n, n))
    for o in m.seen:
        J = np.zeros((2 * o.n, n))
        J[:, 6 * o.cam:6 * o.cam + 6] = o.Jc
        J[:, 6 * len(m.cams) + 3 * o.board:6 * len(m.cams) + 3 * o.board + 3] = o.Jb
        H += J.T @ J / 0.25
    sb, sy = m.st.board_sigma_mm / 1000.0, math.radians(m.st.board_sigma_deg)
    H[6 * len(m.cams):, 6 * len(m.cams):] += np.diag(np.tile([1 / sb ** 2, 1 / sb ** 2, 1 / sy ** 2], len(m.boards)))
    assert np.allclose(C, np.linalg.inv(H), rtol=1e-6, atol=1e-14)
    # the placement adds to what the corners give, and vanishes as the boards are laid exactly
    m.st = _st()
    Cp, _ = m.cov(placed=True)
    Ce, _ = m.cov(placed=False)
    Ct, _ = m.cov(placed=True, sb_mm=1e-3, sb_deg=1e-4)
    for k in range(len(m.cams)):
        s = slice(6 * k, 6 * k + 6)
        assert (np.diag(Cp[s, s]) >= np.diag(Ce[s, s]) - 1e-15).all()
        assert np.allclose(np.diag(Ct[s, s]), np.diag(Ce[s, s]), rtol=2e-2)


def test_relative_pose_jacobian_against_finite_differences():
    rng = np.random.default_rng(4)
    Ra, Rb = rot_zyx(20, 35, -4), rot_zyx(-70, 50, 3)
    pa, pb = np.array([3.0, 0.2, 0.6]), np.array([1.8, -0.9, 0.95])
    J = OB._rel_jac(Ra, pa, Rb, pb)
    R0, t0 = OB.relative_pose(Ra, pa, Rb, pb)
    h = 1e-6
    for k in range(12):
        d = np.zeros(12)
        d[k] = h

        def f(s):
            x = s * d
            R, t = OB.relative_pose(Ra @ OB._exp(x[:3]), pa + x[3:6], Rb @ OB._exp(x[6:9]), pb + x[9:12])
            return np.r_[OB._log(R @ R0.T), t - t0]
        assert np.allclose(J[:, k], (f(1) - f(-1)) / (2 * h), atol=1e-6)
    assert rng is not None


def test_camera_figures_on_the_sample(sample, engine):
    res = OB.run(sample, engine, _st(suggest=True))
    assert res.metrics['measured'] == 6 and res.metrics['observations'] == 14
    for c in res.cameras:
        s, s0 = c['sigma'], c['sigma_corners']
        assert all(math.isfinite(v) and v > 0 for v in s.values())
        # the boards laid by hand cost more than the corners alone give, mostly in the shifts and the yaw
        assert s['x_mm'] > 3 * s0['x_mm'] and s['yaw_deg'] > 2 * s0['yaw_deg']
        assert c['condition'] > 1 and c['weakest']['text']
        assert abs(sum(b['share'] for b in c['boards']) - 1.0) < 1e-9
        assert all(b['ratio_without'] >= c['ratio'] - 1e-9 for b in c['boards'])
        assert c['verdict'] == OB._verdict(c['ratio'])
    # mirror images give mirror figures
    ml, mr = res.camera('ML'), res.camera('MR')
    for k in OB.SIGMA_KEYS:
        assert ml['sigma'][k] == pytest.approx(mr['sigma'][k], rel=1e-6)
    # placement-limited: the explanation says so and the precise placement is suggested with its gain
    f = res.camera('F')
    assert f['verdict'] == 'warning' and f['worst_axis'] == 'yaw'
    assert any('rest' in r and 'where the boards lie' in r for r in f['reasons'])
    sug = [c['suggestions'] for c in res.cameras]
    assert all(any(s['kind'] == 'placement' for s in lst) for lst in sug)
    s = next(s for s in f['suggestions'] if s['kind'] == 'placement')
    assert s['after'] < s['before'] and s['board_sigma_mm'] == 5.0
    # pairs: shared boards tie the cameras far tighter than their poses in the vehicle frame
    assert len(res.pairs) == 6 and {tuple(p['tags']) for p in res.pairs} >= {('F', 'ML'), ('R', 'QL')}
    fm = next(p for p in res.pairs if p['tags'] == ['F', 'ML'])
    assert fm['shared'] == ['T2'] and fm['sigma']['y_mm'] < 0.5 * f['sigma']['y_mm']


def test_suggestions_move_and_add_boards_and_apply(engine):
    p = samples.tigor_sample_project()
    p.targets = [t for t in p.targets if t.name in ('T1', 'T4', 'T6', 'T8', 'T5')]
    st = _st(suggest=True, boards_placed=False, need_deg=0.02, need_mm=0.5)
    res = OB.run(p, engine, st)
    f = res.camera('F')
    assert [b['name'] for b in f['boards']] == ['T1'] and f['verdict'] == 'serious'
    assert any('One board only (T1)' in r for r in f['reasons'])
    assert any('cover only' in r for r in f['reasons'])
    kinds = [s['kind'] for s in f['suggestions']]
    assert 'add' in kinds and 'placement' not in kinds          # the boards are exact here
    add = next(s for s in f['suggestions'] if s['kind'] == 'add')
    assert add['after'] < 0.9 * add['before'] and 'Add a 6 × 6 board in F\'s view' in add['text']
    # apply it: the study run again finds exactly what the suggestion promised
    q = copy.deepcopy(p)
    assert OB.apply_suggestion(q, add)
    assert len(q.targets) == 6 and q.targets[-1].name == 'T7'      # T6 (the count) is taken: the next free one
    assert (q.targets[-1].x, q.targets[-1].y, q.targets[-1].rows) == (add['x'], add['y'], 6)
    r2 = OB.run(q, engine, st)
    assert r2.camera('F')['ratio'] == pytest.approx(add['after'], rel=1e-6)
    for tag, (_, after) in add['others'].items():
        assert r2.camera(tag)['ratio'] == pytest.approx(after, rel=1e-6)
    # a move suggestion applies to its board
    mv = next((s for c in res.cameras for s in c['suggestions'] if s['kind'] == 'move'), None)
    assert mv is not None
    q = copy.deepcopy(p)
    assert OB.apply_suggestion(q, mv)
    t = next(t for t in q.targets if t.uid == mv['board'])
    assert (t.x, t.y) == (mv['x'], mv['y'])
    r3 = OB.run(q, engine, st)
    assert r3.camera(mv['camera'])['ratio'] == pytest.approx(mv['after'], rel=1e-6)
    assert not OB.apply_suggestion(q, dict(mv, board='gone'))
    # no suggestion puts a board under the car or on top of another
    grid = engine.roi_grid(p)
    for c in res.cameras:
        for s in c['suggestions']:
            if s['kind'] in ('move', 'add'):
                cell = grid.cell_of(s['x'], s['y'])
                assert cell is None or not grid.body[cell[1], cell[0]]
                for t in p.targets:
                    if s.get('board') != t.uid:
                        assert math.hypot(t.x - s['x'], t.y - s['y']) > 0.8


# ============================================================================ Monte Carlo against the real solves
def test_monte_carlo_matches_the_joint_solve_with_exact_boards(engine):
    p = samples.tigor_sample_project()
    res = OB.run(p, engine, _st(boards_placed=False, pixel_sigma=0.4))
    mc = OB.monte_carlo(res, trials=60, seed=11, parts=('cameras',))
    cams = mc['cameras']
    assert cams['solved'] == 60 and cams['stalled'] == 0
    assert set(cams['cameras']) == {'F', 'R', 'ML', 'MR', 'QL', 'QR'} and len(cams['pairs']) == 6
    ratios = [a['ratio'] for g in ('cameras', 'pairs') for s in cams[g].values() for a in s['axes'].values()]
    assert 0.9 < float(np.median(ratios)) < 1.1
    assert all(0.65 < r < 1.5 for r in ratios)
    for s in list(cams['cameras'].values()) + list(cams['pairs'].values()):
        assert abs(s['mahalanobis2'] / s['dof'] - 1.0) < 4 * math.sqrt(2 / (s['dof'] * 60))
    assert mc['agree'] and mc['joint_agree']
    # a wrong prediction (the noise taken as half what it is) fails the same comparison
    wrong = OB.run(p, engine, _st(boards_placed=False, pixel_sigma=0.2))
    m = wrong._model
    C, _ = m.cov()
    bad = {}
    for k, cm in enumerate(m.cams):
        ce, _ = OB._to_euler(C[6 * k:6 * k + 6, 6 * k:6 * k + 6], cm.R)
        st_ = cams['cameras'][cm.cam.tag]
        bad[cm.cam.tag] = {'axes': {key: {'ratio': st_['axes'][key]['empirical'] / math.sqrt(ce[i, i])}
                                    for i, key in enumerate(OB.SIGMA_KEYS)},
                           'mahalanobis2': st_['mahalanobis2'] * 4.0, 'dof': 6}
    assert not OB._agreement([bad], 60)[0]


def test_monte_carlo_matches_the_joint_solve_with_boards_placed_by_hand(engine):
    """Two cameras sharing a board, boards laid off their plan by draws of the placement σ: the joint solve's answers
    (carried to the optimum of the same least squares where it stopped short) spread as predicted, and so does the
    pair's relative pose."""
    p = samples.tigor_sample_project()
    _only(p, ('F', 'ML'))
    res = OB.run(p, engine, _st(boards_placed=True))
    assert [pp['tags'] for pp in res.pairs] == [['F', 'ML']]
    mc = OB.monte_carlo(res, trials=40, seed=5, parts=('cameras',))
    cams = mc['cameras']
    assert cams['solved'] >= 38
    opt = cams['optimum']
    ratios = [a['ratio'] for g in ('cameras', 'pairs') for s in opt[g].values() for a in s['axes'].values()]
    assert 0.85 < float(np.median(ratios)) < 1.15 and all(0.6 < r < 1.6 for r in ratios), ratios
    assert mc['agree']
    # the placement shows: the solved positions scatter by the boards' placement, not by the corners
    f = opt['cameras']['F']['axes']
    assert f['x_mm']['empirical'] > 3 * res.camera('F')['sigma_corners']['x_mm']
    # the shared board ties the pair: its relative shift scatters far less than either camera's position
    fm = opt['pairs']['F–ML']['axes']
    assert fm['y_mm']['empirical'] < 0.6 * f['y_mm']['empirical']


def test_monte_carlo_tells_a_solve_that_stopped_short(engine, monkeypatch):
    """A joint solve that stops before its optimum shows: here core.joint.joint_solve is held to three evaluations per
    least-squares call, so it stops near where it started (each camera's own solve). Its spread is too wide; the answers
    carried to the optimum of the same least squares still match the prediction. (The joint solve itself used to stop
    short within its 600 evaluations, which this check found; it now converges.)"""
    import scipy.optimize
    real = scipy.optimize.least_squares

    def capped(*a, **kw):
        kw['max_nfev'] = 3
        return real(*a, **kw)
    monkeypatch.setattr(scipy.optimize, 'least_squares', capped)
    p = samples.tigor_sample_project()
    _only(p, ('F', 'ML'))
    res = OB.run(p, engine, _st(boards_placed=True))
    mc = OB.monte_carlo(res, trials=12, seed=3, parts=('cameras',))
    c = mc['cameras']
    assert c['stalled'] == c['solved'] == 12 and c['median_cost_left'] > 0.5
    assert not mc['joint_agree'] and mc['agree']
    y_joint = c['cameras']['F']['axes']['y_mm']['empirical']
    y_opt = c['optimum']['cameras']['F']['axes']['y_mm']['empirical']
    assert y_joint > 1.3 * y_opt
    assert c['optimum']['cameras']['F']['axes']['y_mm']['ratio'] == pytest.approx(1.0, abs=0.5)
    # uncapped, the same simulated captures are solved to the optimum: nothing stalls
    monkeypatch.setattr(scipy.optimize, 'least_squares', real)
    mc = OB.monte_carlo(res, trials=12, seed=3, parts=('cameras',))
    assert mc['cameras']['stalled'] == 0 and mc['joint_agree']


def test_monte_carlo_lidar_pair_matches_the_calibration_solve(engine):
    p, v = _rig()
    _only(p, ('F',), ('L1',))
    res = OB.run(p, engine, _st(lidar_pairs=True))
    assert len(res.lidar_pairs) == 1
    lp = res.lidar_pairs[0]
    assert lp['camera'] == 'F' and lp['lidar'] == 'L1' and len(lp['poses']) == 6
    assert all(n > 50 for n in lp['points']) and all(math.isfinite(x) for x in lp['sigma'].values())
    mc = OB.monte_carlo(res, trials=30, seed=2, parts=('lidar',))
    s = mc['lidar']['pairs']['F–L1']
    ratios = [a['ratio'] for a in s['axes'].values()]
    assert 0.8 < float(np.median(ratios)) < 1.2 and all(0.6 < r < 1.6 for r in ratios), ratios
    assert abs(s['mahalanobis2'] / 6 - 1.0) < 4 * math.sqrt(2 / (6 * 30))
    assert mc['agree']
    # the board planes and the camera's corners together: the camera's depth on a board at 4–8 m is what limits the
    # lidar's z. The lidar–camera calibration's own prediction (its planes against the camera's corners at 0.3 px)
    # now says so too (it took the camera as good to 1 mm and was 3–5× too tight): it agrees with this study's and
    # with the spread of the solve's answers
    from vantage.core import lidar_calib as LC
    d = res._model.lidar_models[0]
    quick = LC.predicted_sigma(p, d['cam'], d['lidar'], d['plan'], d['lst'])
    band = 4.5 / math.sqrt(2 * 29)
    for k, a in s['axes'].items():
        assert abs(math.log(a['empirical'] / quick[k])) < band, (k, a['empirical'], quick[k])
        assert quick[k] == pytest.approx(lp['sigma'][k], rel=0.15), (k, quick[k], lp['sigma'][k])


# ============================================================================ radars
def test_radar_covariance_closed_form_and_monte_carlo(engine):
    p, v = _rig()
    rd1 = v.sensor(next(s.uid for s in v.sensors if s.tag == 'RD1'))
    # reflectors straight ahead of RD1 at its own height, surveyed exactly: ranges give x alone, azimuths y and yaw
    rs = [5.0, 9.0, 16.0]
    refl = [{'name': f'CR{i + 1}', 'x': rd1.x + r, 'y': rd1.y, 'z': rd1.z, 'edge_mm': 150.0} for i, r in enumerate(rs)]
    st = _st(reflectors=refl, reflector_sigma_mm=0.0, radar_range_sigma_m=0.1, radar_az_sigma_deg=0.3)
    Q = np.array([[r['x'], r['y'], r['z']] for r in refl])
    C = OB._radar_cov(rd1, Q, st)
    sa = math.radians(0.3)
    assert math.sqrt(C[0, 0]) == pytest.approx(0.1 / math.sqrt(3), rel=1e-4)
    # az = atan2(Δy, Δx) − yaw: ∂az/∂y = −1/r, ∂az/∂yaw = −1
    I = sum(np.array([[1 / r ** 2, 1 / r], [1 / r, 1.0]]) for r in rs) / sa ** 2
    Cy = np.linalg.inv(I)
    assert C[1:, 1:] == pytest.approx(Cy, rel=1e-4)
    res = OB.run(p, engine, st)
    r1 = next(r for r in res.radars if r['tag'] == 'RD1')
    assert r1['reflectors'] == ['CR1', 'CR2', 'CR3']
    assert r1['sigma']['yaw_deg'] == pytest.approx(math.degrees(math.sqrt(Cy[1, 1])), rel=1e-4)
    # the corner radars do not see reflectors straight ahead of the long-range one... or only some
    rd4 = next(r for r in res.radars if r['tag'] == 'RD4')
    assert rd4['sigma'] is None and rd4['note']
    # with a survey error the reflectors' places enter; the Monte Carlo agrees either way
    st2 = _st(reflectors=refl, reflector_sigma_mm=10.0)
    res2 = OB.run(p, engine, st2)
    assert next(r for r in res2.radars if r['tag'] == 'RD1')['sigma']['x_mm'] > r1['sigma']['x_mm']
    mc = OB.monte_carlo(res, trials=200, seed=3, parts=('radar',))
    s = mc['radar']['radars']['RD1']
    assert all(0.8 < a['ratio'] < 1.25 for a in s['axes'].values())
    assert mc['agree']


# ============================================================================ fusion density
def test_lidar_counts_match_brute_force_ray_casting(engine):
    p, v = _rig()
    L1 = next(s for s in v.sensors if s.tag == 'L1')
    m = p.sensor_model_for(L1)
    Lt = SC._Lidar(L1, m, None)
    G = np.array([[L1.x + 12.0, L1.y + 1.0], [L1.x - 7.0, L1.y - 3.0], [L1.x + 30.0, L1.y]])
    lines, pts = Lt.counts(G, 0.0, 1.7, 0.5)
    assert np.allclose(pts, Lt.points(G, 0.0, 1.7, 0.5))
    D = S.directions(m.elevations(), S.scan_azimuths(m)) @ S.rotation(L1).T
    o = L1.pos()
    for g, ln, pt in zip(G, lines, pts):
        # a 0.5 m wide, 1.7 m tall plane square to the line of sight
        n = np.r_[g - o[:2], 0.0]
        n /= np.linalg.norm(n)
        den = D @ n
        with np.errstate(divide='ignore', invalid='ignore'):
            t = np.where(den > 1e-9, float(np.r_[g, 0.0] @ n - o @ n) / den, -1)
        X = o + D * t[..., None]
        side = np.cross([0, 0, 1.0], n)
        hit = (t > 0) & (np.abs((X - np.r_[g, 0.0]) @ side) <= 0.25) & (X[..., 2] >= 0) & (X[..., 2] <= 1.7)
        assert abs(hit.any(1).sum() - ln) <= 1                 # channels across it
        assert abs(hit.sum() - pt) <= max(0.12 * pt, 2 * ln)    # points: the angular width / h_res per channel


def test_points_per_pixel_against_a_hand_calculation(engine):
    """A lidar sitting in a rectilinear camera's centre and looking the same way: at the image centre the points per
    pixel are the lidar's returns per steradian over the camera's pixels per steradian, f_px² (the distances cancel)."""
    p, v = _rig()
    for s in v.sensors:
        s.visible = s.tag == 'L1'
    cam = _only(p, ('F',)).by_tag('F')
    from vantage.core.lens import Lens
    p.lenses['test_rect'] = Lens(id='test_rect', name='Rectilinear 4 mm', focal_mm=4.0, projection='rectilinear',
                                 focus_near_m=0.0)
    cam.lens_id = 'test_rect'
    cam.x, cam.y, cam.z, cam.yaw, cam.pitch, cam.roll = 6.0, 0.0, 1.5, 0.0, 0.0, 0.0     # clear of the body
    L1 = next(s for s in v.sensors if s.tag == 'L1')
    L1.x, L1.y, L1.z, L1.yaw, L1.pitch, L1.roll = 6.0, 0.0, 1.5, 0.0, 0.0, 0.0
    out = SC.camera_fusion(p, engine, at_m=20.0)
    f = out[0]
    A = f['image']['points_per_px']
    nu, nv = f['image']['grid']
    intr = OB.resolve(p.module_for(cam), p.lens_for(cam))
    m = p.sensor_model_for(L1)
    E = m.elevations()
    i = int(np.searchsorted(E, 0.0))
    de = E[i] - E[i - 1]
    want = (180 / math.pi) ** 2 / (de * m.h_res) / intr.f_px ** 2
    # the grid cell nearest the principal point
    gu, gv = int(intr.cx / intr.width * nu), int(intr.cy / intr.height * nv)
    assert A[gv, gu] == pytest.approx(want, rel=0.03)
    # beyond the lidar's vertical field nothing lands: the top rows of a camera looking level see above +22.5° only
    # with a wide lens; here every supported cell lies within the lidar's elevations
    d_opt, _ = intr.unproject((np.arange(nu) + 0.5) * intr.width / nu, np.full(nu, 0.5 * intr.height / nv))
    el_top = np.degrees(np.arcsin(-d_opt[:, 1]))
    if el_top.max() > E[-1] + 1:
        assert (A[0][el_top > E[-1] + 1] == 0).all()


def test_fusion_on_the_data_rig(engine):
    p, v = _rig()
    out = SC.camera_fusion(p, engine)
    assert [f['tag'] for f in out] == [c.tag for c in v.cameras if c.visible]
    for f in out:
        assert f['lidars'] == ['L1']
        assert 0.0 < f['image']['support_share'] < 1.0 and f['image']['median_points_per_px'] > 0
        names = [k['name'] for k in f['classes']]
        assert names == [c['name'] for c in SC.DEFAULT_CLASSES]
        ped = next(k for k in f['classes'] if k['name'] == 'Pedestrian')
        rows = {r['range_m']: r for r in ped['rows']}
        # nearer is denser: points and scan lines fall with range
        assert rows[10.0]['points_median'] > rows[20.0]['points_median'] > rows[30.0]['points_median']
        assert rows[10.0]['lines_median'] >= rows[30.0]['lines_median']
        assert 0 <= ped['roi_supported_share'] <= 1 and ped['roi_samples'] > 0
    # the same points as the coverage study's own count for a pedestrian ahead of F
    f = out[0]
    cam = v.by_tag('F')
    L1 = next(s for s in v.sensors if s.tag == 'L1')
    ped = SC.CoverageClass(**SC.DEFAULT_CLASSES[1])
    g = np.array([[cam.x + 20.0, cam.y]])
    want = SC.object_points(p, engine, L1, ped, g)[0]
    row = next(r for r in f['classes'][1]['rows'] if r['range_m'] == 20.0)
    assert row['points_min'] <= want + 1e-6
    # no lidar: every camera reports no support, with a finding
    for s in v.sensors:
        s.visible = False
    out2 = SC.camera_fusion(p, engine)
    assert all(f['image']['support_share'] == 0 and f['lidars'] == [] for f in out2)
    assert all(not f['findings'] for f in out2)                 # nothing to fuse is not a fusion finding


def test_fusion_cancel(engine):
    p, _ = _rig()
    assert SC.camera_fusion(p, engine, cancel=lambda: True) is None


# ============================================================================ the study end to end
def test_run_on_the_data_rig_exports_and_summary(engine, tmp_path):
    p, v = _rig()
    st = OB.ObservabilitySettings.from_dict({'reflectors': [{'name': 'CR1', 'x': 9.0, 'y': 0.5, 'z': 0.5},
                                                            {'name': 'CR2', 'x': 15.0, 'y': -2.0, 'z': 0.6}]})
    res = OB.run(p, engine, st, progress=lambda f, t='': None)
    assert len(res.lidar_pairs) == 6 and res.fusion is not None and len(res.radars) == 5
    rd1 = next(r for r in res.radars if r['tag'] == 'RD1')
    assert rd1['reflectors'] == ['CR1', 'CR2'] and rd1['sigma'] is not None
    s = OB.summary(res)
    json.dumps(s, allow_nan=False)
    assert s['cameras'][0]['tag'] == 'F' and s['lidar_pairs'] and s['fusion']
    paths = OB.export(res, p, str(tmp_path / 'obs'))
    names = sorted(os.path.basename(x) for x in paths)
    assert names == ['fusion_density.png', 'observability.csv', 'observability.json', 'observability.png']
    assert sorted(OB.file_names(p, v)) == names
    p.analysis.observability = {'fusion': False}
    assert 'fusion_density.png' not in OB.file_names(p, v)
    doc = json.loads(open(tmp_path / 'obs' / 'observability.json').read())
    assert doc['schema'] == OB.SCHEMA and doc['frame'].startswith('vehicle') and doc['variant'] == v.name
    assert len(doc['cameras']) == 6 and doc['cameras'][0]['sigma']['yaw_deg'] > 0
    assert doc['reflectors'][0]['rcs_m2'] > 0 and doc['fusion'][0]['image']['grid'] == list(SC.FUSION_GRID)
    rows = open(tmp_path / 'obs' / 'observability.csv').read().splitlines()
    assert rows[0].split(',') == list(OB.CSV_COLUMNS)
    kinds = [r.split(',')[0] for r in rows[1:]]
    assert kinds.count('camera') == 6 and kinds.count('lidar') == 6 and kinds.count('radar') == 5
    import cv2
    for n in ('observability.png', 'fusion_density.png'):
        img = cv2.imdecode(np.frombuffer(open(tmp_path / 'obs' / n, 'rb').read(), np.uint8), cv2.IMREAD_COLOR)
        assert img is not None and img.shape[0] > 100


def test_edge_cases(engine):
    # no targets: the cameras cannot be measured, said once
    p = samples.tigor_sample_project()
    p.targets = []
    res = OB.run(p, engine, _st())
    assert all(c['sigma'] is None and c['verdict'] == 'critical' for c in res.cameras)
    assert any(f['kind'] == 'targets' for f in res.findings) and not any(f['kind'] == 'camera' for f in res.findings)
    assert OB.issues(p) == []
    # hidden cameras and a camera with a pose that is not a number are left out; an empty variant runs
    p = samples.tigor_sample_project()
    v = p.variant
    v.by_tag('R').visible = False
    v.by_tag('QL').yaw = float('nan')
    res = OB.run(p, engine, _st())
    assert [c['tag'] for c in res.cameras] == ['F', 'ML', 'MR', 'QR']
    v.cameras = []
    res = OB.run(p, engine, _st(suggest=True, lidar_pairs=True, fusion=True))
    assert res.cameras == [] and res.pairs == [] and res.fusion is None
    OB.files(res, p)
    # a variant other than the active one, without touching the project
    p = samples.tigor_sample_project()
    other = p.variants[1]
    res = OB.run(p, engine, _st(), variant=other)
    assert res.variant_uid == other.uid and p.active_uid == p.variants[0].uid
    assert [c['tag'] for c in res.cameras] == [c.tag for c in other.cameras]
    # cancel
    assert OB.run(p, engine, _st(), cancel=lambda: True) is None
    # an old project without the settings
    p.analysis.observability = {}
    assert OB.settings_of(p) == OB.ObservabilitySettings()


def test_issues(engine):
    p = samples.tigor_sample_project()
    vg = engine.vehicle_geom(p)
    res = engine.analyse(p)
    # the sample's boards pin every camera within twice the requirement: nothing for the Issues list
    assert OB.issues(p, vg, res) == []
    # a camera that sees no board whole; the mirror cameras left with one board each pin their yaw only loosely
    p.targets = [t for t in p.targets if t.name not in ('T1', 'T2', 'T3')]
    out = OB.issues(p, vg, res)
    assert out[0].objects == 'F' and out[0].severity == 'serious' and 'sees no calibration board' in out[0].message
    assert {'ML', 'MR'} <= {i.objects for i in out[1:]} and all(i.severity == 'warning' for i in out[1:])
    assert all('yaw' in i.message for i in out[1:])
    # cached on the inputs; without the analysis result the cube maps are built once
    assert OB.issues(p, vg, res) is not out and [i.message for i in OB.issues(p, vg)] == [i.message for i in out]
    # a tight requirement: the cameras beyond twice it are warned about
    p.analysis.observability = {'need_deg': 0.01, 'need_mm': 1.0}
    out = OB.issues(p, vg)
    assert {i.objects for i in out if i.severity == 'warning'} == {'R', 'ML', 'MR', 'QL', 'QR'}
    # reflectors placed but a radar without two in its field
    p2, _ = _rig()
    p2.analysis.observability = {'reflectors': [{'name': 'CR1', 'x': 9.0, 'y': 0.0, 'z': 0.5}]}
    out = OB.issues(p2, engine.vehicle_geom(p2))
    assert {i.objects for i in out} == {'RD1', 'RD2', 'RD3', 'RD4', 'RD5'}


# ============================================================================ the page
from test_gui import app, win, pump, settle, _wait   # noqa: E402,F401 (fixtures)
from PyQt6.QtCore import Qt                           # noqa: E402
from PyQt6.QtTest import QTest                        # noqa: E402


def _open(win, name, cls):
    if name not in win._page_widgets:
        pg = cls(win.ctx)
        win.pages.addWidget(pg)
        win._page_widgets[name] = pg
    win.set_page(name)
    return win._page_widgets[name]


def _row(t, text, col=0):
    return next(r for r in range(t.rowCount()) if t.item(r, col) is not None and t.item(r, col).text() == text)


def test_page_runs_shows_applies_verifies_and_undoes(app, win, tmp_path):
    from vantage.ui.workspaces.observability_ws import ObservabilityPage
    p, v = _rig()
    _only(p, ('F', 'ML', 'MR', 'QL'), ('L1', 'RD1', 'RD2'))
    win.store.set_project(p)
    pump(app, 0.3)
    pg = _open(win, 'observability', ObservabilityPage)
    # it runs by itself when shown
    _wait(app, lambda: pg.result is not None and not pg.busy, 180)
    res = pg.result
    assert 'Failed' not in pg.status.text() and not pg.stale
    assert pg.cams.rowCount() == 4 and pg.pairs.rowCount() == len(res.pairs) > 0
    assert pg.lidars.rowCount() == len(res.lidar_pairs) > 0 and pg.radars.rowCount() == 2
    assert pg.fusion_table.rowCount() > 0 and len(pg.fusion_maps.maps) == 4
    assert pg.tiles['cams'].unit == 'of 4' and pg.plan.res is res
    assert len(pg.findings) == len(res.findings)
    # pick QL in the table: its details; in the plan by a click on it
    r = _row(pg.cams, 'QL')
    pg.cams.selectRow(r)
    pump(app, 0.1)
    ql = res.camera('QL')
    assert pg.sel == ql['uid'] and pg.det_head.text().startswith('QL')
    assert pg.det_boards.rowCount() == len(ql['boards'])
    pg.plan.repaint()
    f_uid = res.camera('F')['uid']
    q = next(q for q, uid, _ in pg.plan._cams if uid == f_uid)
    QTest.mouseClick(pg.plan, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, q.toPoint())
    pump(app, 0.1)
    assert pg.sel == f_uid and pg.cams.item(pg.cams.currentRow(), 0).text() == 'F'
    # a setting through its field: undoable, and the study runs again by itself
    sig0 = res.camera('F')['sigma_corners']['roll_deg']
    pg.fields['pixel_sigma'].set_value(0.6)
    pg.fields['pixel_sigma'].committed.emit(0.6)
    assert pg.settings.pixel_sigma == 0.6 and pg.stale
    _wait(app, lambda: pg.result is not res and not pg.busy and not pg.stale, 180)
    assert pg.result.camera('F')['sigma_corners']['roll_deg'] == pytest.approx(2 * sig0, rel=1e-6)
    win.store.undo.undo()
    pump(app, 0.1)
    assert pg.settings.pixel_sigma == 0.3 and pg.fields['pixel_sigma'].value() == pytest.approx(0.3)
    _wait(app, lambda: not pg.busy and not pg.stale, 180)
    # the toggles
    pg.toggles['boards_placed'].click()
    pump(app, 0.1)
    assert pg.settings.boards_placed is False and not pg.fields['board_sigma_mm'].isEnabled()
    win.store.undo.undo()
    pump(app, 0.1)
    assert pg.settings.boards_placed is True and pg.fields['board_sigma_mm'].isEnabled()
    # reflectors: add one ahead of the first radar, edit it, remove it; each undoable
    assert pg.add_reflector()
    pump(app, 0.1)
    st = pg.settings
    assert [r['name'] for r in st.reflectors] == ['CR1'] and pg.refl.rowCount() == 1
    assert pg.refl.item(0, 5).text() == f'{OB.reflector_rcs(100.0):.1f}'
    pg.refl.item(0, 1).setText('12.5')
    pump(app, 0.1)
    assert pg.settings.reflectors[0]['x'] == 12.5
    pg.refl.item(0, 1).setText('not a number')
    pump(app, 0.1)
    assert pg.settings.reflectors[0]['x'] == 12.5 and pg.refl.item(0, 1).text() == '12.50'
    pg.refl.selectRow(0)
    assert pg.remove_reflector()
    pump(app, 0.1)
    assert pg.settings.reflectors == []
    win.store.undo.undo()
    win.store.undo.undo()
    pump(app, 0.1)
    assert pg.settings.reflectors[0]['x'] != 12.5 and len(pg.settings.reflectors) == 1
    win.store.undo.undo()
    pump(app, 0.1)
    assert pg.settings.reflectors == []
    _wait(app, lambda: not pg.busy and not pg.stale, 180)
    # the placement suggestion: its button sets the placement precision, undoable
    pg.pick(res.camera('ML')['uid'])
    pump(app, 0.1)
    rows = [w for w in pg._sugg_buttons]
    place = next((w for w in rows if w.button.text().startswith('Use ±')), None)
    assert place is not None
    place.button.click()
    pump(app, 0.1)
    assert pg.settings.board_sigma_mm == 5.0
    win.store.undo.undo()
    pump(app, 0.1)
    assert pg.settings.board_sigma_mm == 20.0
    _wait(app, lambda: not pg.busy and not pg.stale, 180)
    # a tight requirement with the boards exact: board moves and added boards; Apply adds one (undoable)
    pg.toggles['boards_placed'].click()
    pg.fields['need_deg'].set_value(0.005)
    pg.fields['need_deg'].committed.emit(0.005)
    _wait(app, lambda: not pg.busy and not pg.stale and pg.result.settings.need_deg == 0.005, 180)
    cam = next(c for c in pg.result.cameras if any(s['kind'] == 'add' for s in c['suggestions']))
    pg.pick(cam['uid'])
    pump(app, 0.1)
    s_add = next(s for s in cam['suggestions'] if s['kind'] == 'add')
    j = cam['suggestions'].index(s_add)
    n_t = len(win.store.project.targets)
    pg._sugg_buttons[j].button.click()
    pump(app, 0.1)
    assert len(win.store.project.targets) == n_t + 1
    assert win.store.project.targets[-1].x == s_add['x']
    _wait(app, lambda: not pg.busy and not pg.stale, 180)
    assert pg.result.camera(cam['tag'])['ratio'] == pytest.approx(s_add['after'], rel=1e-6)
    win.store.undo.undo()
    pump(app, 0.1)
    assert len(win.store.project.targets) == n_t
    _wait(app, lambda: not pg.busy and not pg.stale, 180)
    # Run through the button (a main-window job), then Verify with a few trials
    QTest.mouseClick(pg.b_run, Qt.MouseButton.LeftButton)
    pump(app, 0.05)
    _wait(app, lambda: not pg.busy and 'observability' not in win._jobs, 180)
    assert pg.result is not None and pg.b_verify.isEnabled()
    assert pg.verify(trials=5)
    _wait(app, lambda: not pg.mc_busy, 600)
    mc = pg.result.mc
    assert mc is not None and mc['trials'] == 5 and pg.mc_table.rowCount() > 0
    assert set(mc['cameras']['cameras']) == {'F', 'ML', 'MR', 'QL'} and mc['lidar']['pairs']
    assert pg.tiles['mc'].value in ('Holds', 'Differs') and 'simulated captures' in pg.mc_note.text()
    # export
    paths = pg.export(str(tmp_path / 'exp'))
    assert sorted(os.path.basename(x) for x in paths) == ['fusion_density.png', 'observability.csv',
                                                          'observability.json', 'observability.png']
    doc = json.loads(open(tmp_path / 'exp' / 'observability.json').read())
    assert doc['monte_carlo']['trials'] == 5
    assert pg.summary()['monte_carlo']['trials'] == 5
    # a new project drops the result and runs again for it
    win.store.set_project(samples.tigor_sample_project())
    pump(app, 0.1)
    assert pg.result is None or pg.result.variant_uid == win.store.project.variant.uid
    _wait(app, lambda: pg.result is not None and not pg.busy, 180)
    assert pg.radars.rowCount() == 0 and pg.lidars.rowCount() == 0 and pg.cams.rowCount() == 6
    assert not pg.fusion_maps.maps


def test_page_without_targets_and_hidden(app, win):
    from vantage.ui.workspaces.observability_ws import ObservabilityPage
    p = samples.tigor_sample_project()
    p.targets = []
    win.store.set_project(p)
    pump(app, 0.2)
    pg = _open(win, 'observability', ObservabilityPage)
    _wait(app, lambda: pg.result is not None and not pg.busy, 120)
    assert all(pg.cams.item(r, 8).text() == 'Critical' for r in range(pg.cams.rowCount()))
    assert any('No calibration targets' in w.text.text() for w in pg.findings)
    # hidden: edits only mark it; shown again it catches up
    win.set_page('viewports')
    pump(app, 0.1)

    def add(pj):
        pj.targets = samples.tigor_targets()
    win.store.edit('targets', add, 'targets back')
    pump(app, 0.2)
    assert pg._dirty and pg.result.cameras[0]['sigma'] is None
    win.set_page('observability')
    _wait(app, lambda: not pg.busy and not pg.stale, 120)
    assert pg.result.cameras[0]['sigma'] is not None and pg.cams.item(0, 8).text() != 'Critical'
