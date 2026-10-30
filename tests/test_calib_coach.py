"""The live calibration coach (core.calib_coach) and its mode on Calibrate ▸ Live rig (ui.live_calib_coach).

Independent checks: the Kannala–Brandt projection against the app's lens model (core.lens.Intrinsics.project_dirs) and
its parameter Jacobian against finite differences of it; a view's information on the intrinsics against the Schur
complement of the full joint Fisher matrix (intrinsics and both views' board poses) built by finite differences
through Intrinsics.project_dirs; the lidar–camera σ against core.observability's own lidar–camera pair (all poses in
one matrix); the ghost board against the lens model and against the board the lens calibration's detector finds in a
rendered frame; the coached and random technicians counted with the same seeds."""
import json
import math
import os
import time
from dataclasses import replace

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
import pytest

from vantage.core import calib_coach as CC, lidar_calib as LC, observability as OB, samples
from vantage.core.analysis import Engine
from vantage.core.lens import resolve
from vantage.core.mathutil import rot_x, rot_y


# ----------------------------------------------------------------------------- fixtures
@pytest.fixture(scope='module')
def engine():
    return Engine()


@pytest.fixture(scope='module')
def rig():
    """The sample with its data rig (6 cameras, a roof lidar, 5 radars) active."""
    p = samples.tigor_sample_project()
    v = next(v for v in p.variants if v.sensors)
    p.active_uid = v.uid
    return p


@pytest.fixture(scope='module')
def lens_coach(rig, engine):
    return CC.Coach(rig, engine, CC.CoachSettings(camera='F'))


def _exp(w):
    return OB._exp(w)


# ----------------------------------------------------------------------------- settings
def test_settings_are_tolerant():
    s = CC.CoachSettings.from_dict({'mode': 'LIDAR', 'camera': ' F ', 'cols': '12.4', 'rows': 'x', 'square_mm': -3,
                                    'near_m': 4.0, 'far_m': 1.0, 'criterion': 'bogus', 'max_views': 1e9,
                                    'sharp_min': float('nan'), 'unknown': 5, 'seed': 3.6})
    assert s.mode == 'lidar' and s.camera == 'F' and s.cols == 12 and s.rows == 6 and s.square_mm == 5.0
    assert (s.near_m, s.far_m) == (1.0, 4.0) and s.criterion == 'weakest' and s.max_views == 200
    assert s.sharp_min == CC.CoachSettings().sharp_min and s.seed == 4
    assert CC.CoachSettings.from_dict(None) == CC.CoachSettings()
    assert CC.CoachSettings.from_dict(s.to_dict()) == s
    d = CC.CoachSettings().to_dict()
    assert set(d) >= {'mode', 'camera', 'cols', 'rows', 'square_mm', 'pixel_sigma', 'need_deg', 'criterion', 'seed'}


def test_requirement_from_observability_scorecard_and_settings(rig):
    p = samples.tigor_sample_project()
    q = CC.requirement(p, CC.CoachSettings())
    assert q['deg'] == OB.ObservabilitySettings().need_deg and 'observability' in q['source_deg']
    p.analysis.requirements = {'items': [{'id': 'R4', 'metric': 'observability.worst_deg', 'op': '<=', 'target': 0.25},
                                         {'id': 'R5', 'metric': 'observability.worst_deg', 'op': '>=', 'target': 0.01}]}
    q = CC.requirement(p, CC.CoachSettings())
    assert q['deg'] == 0.25 and 'scorecard (R4)' in q['source_deg']
    assert CC.requirement(p, CC.CoachSettings(need_deg=0.05))['deg'] == 0.05
    lst = LC.LidarCalibSettings()
    q = CC.requirement(p, CC.CoachSettings(mode='lidar'))
    assert (q['deg'], q['mm']) == (lst.ang_tol_deg, lst.pos_tol_mm)
    p.analysis.requirements = {'items': [{'id': 'R1', 'metric': 'lidar_calib.planned_sigma_mm', 'op': '<=',
                                          'target': 4.0}]}
    q = CC.requirement(p, CC.CoachSettings(mode='lidar'))
    assert q['mm'] == 4.0 and q['deg'] == lst.ang_tol_deg


# ----------------------------------------------------------------------------- the lens model
def _intr_with(intr, p):
    """The app's lens model (core.lens.Intrinsics) carrying Kannala–Brandt parameters p (an independent projection)."""
    return replace(intr, f_mm=p[0] * intr.pixel_mm, aspect=p[1] / p[0], cx=p[2] + 0.5, cy=p[3] + 0.5,
                   projection='kannala_brandt', kb=tuple(float(x) for x in p[4:]))


def test_kb_projection_matches_the_lens_model(lens_coach):
    c = lens_coach
    intr = c.intr
    rng = np.random.default_rng(0)
    d = rng.normal(size=(500, 3))
    d[:, 2] = np.abs(d[:, 2]) + 0.2
    P = d * rng.uniform(0.5, 5, (500, 1))
    u, v, th, _ = intr.project_dirs(P)
    uv, th2 = CC.kb_project(c.p, P)
    assert np.allclose(uv[:, 0] + 0.5, u, atol=1e-9) and np.allclose(uv[:, 1] + 0.5, v, atol=1e-9)
    assert np.allclose(th, th2)
    # the unprojection inverts it
    r = CC.kb_unproject(c.p, uv[:, 0], uv[:, 1])
    assert np.allclose(r, P / np.linalg.norm(P, axis=1, keepdims=True), atol=1e-9)
    # the analytic parameter Jacobian against finite differences of the lens model itself
    J = CC._kb_param_jac(c.p, P[:20])
    for i in range(8):
        h = 1e-4 if i < 4 else 1e-7
        a, b = c.p.copy(), c.p.copy()
        a[i] += h
        b[i] -= h
        ia, ib = _intr_with(intr, a), _intr_with(intr, b)
        ua, va, _, _ = ia.project_dirs(P[:20])
        ub, vb, _, _ = ib.project_dirs(P[:20])
        fd = np.stack([(ua - ub) / (2 * h), (va - vb) / (2 * h)], -1)
        assert np.allclose(J[:, :, i], fd, rtol=1e-5, atol=1e-5), CC.LENS_PARAMS[i]


def test_information_adds_up(lens_coach):
    """Two views' information on the intrinsics = the sum of each view's: checked against the Schur complement of the
    full joint Fisher matrix (8 intrinsics + 6 per board pose) built by finite differences through the lens model."""
    c = lens_coach
    cands = c.candidates()
    views = [cands[3], cands[len(cands) // 2]]
    I1, I2 = c.info_of(views[:1])[0], c.info_of(views[1:])[0]
    s = c.sigma_px
    rows = []
    for k, b in enumerate(views):
        Rb, cb = c.pose_optical(b)

        def uv(x, Rb=Rb, cb=cb):
            intr = _intr_with(c.intr, x[:8])
            R = _exp(x[8:11]) @ Rb
            P = (c.model @ R.T) + cb + x[11:14]
            u, v, _, _ = intr.project_dirs(P)
            return np.stack([u, v], -1).ravel()
        x0 = np.r_[c.p, np.zeros(6)]
        J = np.zeros((2 * len(c.model), 14))
        for i in range(14):
            h = 1e-4 if i < 4 else 1e-7
            a, bb = x0.copy(), x0.copy()
            a[i] += h
            bb[i] -= h
            J[:, i] = (uv(a) - uv(bb)) / (2 * h)
        full = np.zeros((2 * len(c.model), 8 + 6 * len(views)))
        full[:, :8] = J[:, :8]
        full[:, 8 + 6 * k:14 + 6 * k] = J[:, 8:]
        rows.append(full)
    J = np.vstack(rows)
    F = J.T @ J / s ** 2
    A, B, D = F[:8, :8], F[:8, 8:], F[8:, 8:]
    ref = A - B @ np.linalg.solve(D, B.T)
    got = I1 + I2
    assert np.allclose(got, ref, rtol=2e-4, atol=1e-6 * np.abs(ref).max())
    # the coach sums its accepted views' information the same way
    c.reset()
    for b in views:
        c.add_pose(b)
    assert np.allclose(c.information(), I1 + I2)
    # one view of a plane cannot pin eight intrinsics; information is symmetric and positive semi-definite
    lam = np.linalg.eigvalsh(I1)
    assert lam.min() > -1e-6 * lam.max() and np.allclose(I1, I1.T)
    c.reset()


def test_sigma_falls_monotonically(lens_coach):
    c = lens_coach
    c.reset()
    tech = CC.Technician(c, 'random', 4)
    prev = c.status()
    worst = [prev['worst']]
    for _ in range(12):
        b = tech.next_pose()
        assert b is not None and c.add_pose(b) is not None
        s = c.status()
        for k in CC.LENS_LOCKS:
            assert s['locks'][k]['ratio'] <= prev['locks'][k]['ratio'] * (1 + 1e-9), k
        for k in CC.LENS_PARAMS:
            assert s['params'][k] <= prev['params'][k] * (1 + 1e-9), k
        worst.append(s['worst'])
        prev = s
    assert all(b < a for a, b in zip(worst, worst[1:]))
    assert worst[0] > CC.UNDETERMINED and math.isinf(c.status(np.zeros((8, 8)))['locks']['rays']['sigma'])
    c.reset()


def test_lidar_coach_matches_observability(rig, engine):
    """The coach's lidar–camera σ after the lidar–camera calibration's planned poses equals core.observability's for
    the same pair (which puts every pose in one matrix); the marginal matrices add up pose by pose."""
    c = CC.Coach(rig, engine, CC.CoachSettings(mode='lidar'))
    assert c.lidar is not None and c.st.mode == 'lidar'
    entry, data = OB._lidar_pair(rig, engine, c.vg, c.cam, c.lidar, c.share,
                                 OB.ObservabilitySettings.from_dict({'fusion': False, 'suggest': False}))
    prev = c.status()
    for b in data['plan']:
        assert c.add_pose(b) is not None
        s = c.status()
        assert s['worst'] <= prev['worst'] * (1 + 1e-9)
        prev = s
    for k in CC.LIDAR_LOCKS:
        assert prev['locks'][k]['sigma'] == pytest.approx(entry['sigma'][k], rel=1e-6), k
    a, m = c.information()
    assert np.allclose(a, sum(v.info[0] for v in c.views)) and np.allclose(m, sum(v.info[1] for v in c.views))
    assert prev['locks']['x_mm']['need'] == LC.LidarCalibSettings().pos_tol_mm


# ----------------------------------------------------------------------------- the next-best view
def test_next_best_view_targets_the_weakest_axis(lens_coach):
    c = lens_coach
    c.reset()
    intr = c.intr
    cands = c.candidates()

    def off_centre(b):
        u, v, _, _ = intr.project_dirs(c.to_optical(b.centre[None]))
        return math.hypot(u[0] - intr.cx, v[0] - intr.cy) / math.hypot(intr.cx, intr.cy)
    central = sorted(cands, key=off_centre)
    taken = []
    for b in central:
        if not c.duplicate(b):
            c.add_pose(b)
            taken.append(b)
        if len(taken) == 6:
            break
    before = c.status()
    weakest = before['worst_key']
    assert weakest in ('rays', 'distortion')           # boards near the middle leave the image's edge to extrapolation
    adv = c.next_best()
    assert adv.target_lock == weakest and adv.worst_after < adv.worst_before
    # it is the candidate that leaves the smallest worst ratio, among every candidate left
    left, infos = zip(*[(b, i) for b, i in zip(*c._cand_infos()) if not c.duplicate(b)])
    _, worst = c.scores(list(left), list(infos))
    assert adv.worst_after == pytest.approx(worst.min(), rel=1e-9)
    # and it works on the weakest lock more than a typical candidate does
    gains = []
    for i in infos:
        s = c.status(c.information(i))
        gains.append(before['locks'][weakest]['ratio'] / s['locks'][weakest]['ratio'])
    best = before['locks'][weakest]['ratio'] / adv.after[weakest]
    assert best >= np.percentile(gains, 90)
    # it sends the board towards the edge the central views left out
    assert off_centre(adv.pose) > max(off_centre(b) for b in taken)
    # the D-optimal criterion picks the largest log det instead
    cd = CC.Coach(c.project, c.engine, CC.dataclass_replace(c.st, criterion='d_optimal'))
    for b in taken:
        cd.add_pose(b)
    ad = cd.next_best()
    I0 = cd.information() + cd.prior
    best_ld = max(np.linalg.slogdet(I0 + i)[1] for i in infos)
    k = next(j for j, b in enumerate(left) if np.allclose(b.centre, ad.pose.centre) and np.allclose(b.R, ad.pose.R))
    assert np.linalg.slogdet(I0 + infos[k])[1] == pytest.approx(best_ld)
    c.reset()


def test_held_poses_and_guidance(lens_coach):
    c = lens_coach
    C = c.C
    centre = C + np.array([1.6, 0.3, 0.5])
    b = CC.held_pose(centre, C, 20.0, -15.0)
    assert np.allclose(b.R.T @ b.R, np.eye(3)) and np.linalg.det(b.R) == pytest.approx(1.0)
    top, side = CC.tilt_of(b, C)
    assert (top, side) == (pytest.approx(20.0), pytest.approx(-15.0))
    # tilting the top towards the camera brings the top edge nearer it; a positive side turn the right edge
    W, H = c.board.board_size()
    flat = CC.held_pose(centre, C)
    assert np.linalg.norm(b.centre + b.R[:, 1] * H / 2 - C) < np.linalg.norm(flat.centre + flat.R[:, 1] * H / 2 - C)
    s = CC.held_pose(centre, C, 0.0, 20.0)
    assert np.linalg.norm(s.centre - s.R[:, 0] * W / 2 - C) < np.linalg.norm(flat.centre - flat.R[:, 0] * W / 2 - C)
    # the technician faces the camera: their left is the camera's right (+x in the image), here the vehicle's −y
    cur = CC.held_pose(C + np.array([2.0, 0.0, 0.4]), C)
    tgt = CC.held_pose(cur.centre + np.array([0.0, -0.6, 0.0]), C, 20.0, 0.0)
    text, tv, mv = c.guidance(tgt, cur)
    assert 'move 0.6 m to your left' in text.lower() and 'tilt the top of the board 20° towards the camera' in text
    assert mv['left_m'] == pytest.approx(0.6, abs=0.02) and 'Δy -0.60 m' in tv and 'vehicle frame' in tv
    ua, _, _, _ = c.intr.project_dirs(c.to_optical(cur.centre[None]))
    ub, _, _, _ = c.intr.project_dirs(c.to_optical(tgt.centre[None]))
    assert ub[0] > ua[0]                                   # to the right in the image
    assert c.guidance(cur, cur)[0] == 'Hold it there.'
    text, _, _ = c.guidance(tgt, None)
    assert text.startswith('Hold the board') and 'm from the camera' in text


# ----------------------------------------------------------------------------- the ghost board and acceptance
def test_ghost_board_through_the_lens_model(lens_coach):
    c = lens_coach
    c.reset()
    b = c.candidates()[len(c.candidates()) // 3]
    g = c.ghost(b)
    u, v, _, ok = c.intr.project_dirs(c.to_optical(b.centre + c.model @ b.R.T))
    assert g['ok'] and ok.all() and np.allclose(g['corners'], np.c_[u, v], atol=1e-9)
    uv, _ = CC.kb_project(c.p, c.to_optical(b.centre + c.model @ b.R.T))
    assert np.allclose(g['corners'], uv + 0.5, atol=1e-9)
    W, H = c.board.board_size()
    assert len(g['outline']) == len(LC.outline_local(W, H, 16)) and len(g['grid']) == c.board.cols + c.board.rows + 4
    # the board the lens calibration's detector finds in a frame rendered at that pose lies on the ghost
    tech = CC.Technician(c, 'coached', 1)
    det = c.detect(tech.render(b, 0.5, seed=3))
    assert det.found
    d = np.linalg.norm(g['corners'][:, None, :] - (det.uv + 0.5)[None], axis=-1).min(1)
    assert np.median(d) < 0.6 and d.max() < 1.5
    assert np.linalg.norm(det.pose.centre - b.centre) < 0.01 and CC._normal_angle(det.pose.R[:, 2], b.R[:, 2]) < 0.5


def test_duplicate_and_blur_rejection(lens_coach):
    import cv2
    c = lens_coach
    c.reset()
    tech = CC.Technician(c, 'coached', 1)
    a = c.candidates()[10]
    f = tech.render(a, 0.5, seed=1)
    d1 = c.offer(f, 1.0)
    assert d1.accepted and len(c.views) == 1 and d1.sharpness >= c.st.sharp_min
    d2 = c.offer(tech.render(a, 0.5, seed=2), 2.0)                # the same pose again, fresh noise
    assert d2.found and not d2.accepted and 'the same view as view_01' in d2.reason and len(c.views) == 1
    near = CC.held_pose(a.centre + a.R[:, 0] * 0.04, c.C, *CC.tilt_of(a, c.C))
    assert c.duplicate(near) == 'view_01'
    far = next(b for b in c.candidates() if np.linalg.norm(b.centre - a.centre) > 0.4)
    Rb, cb = c.pose_optical(far)
    uv, _ = CC.kb_project(c.p, c.model @ Rb.T + cb)
    sq = CC._feature_px_grid(uv[None] * 0.5, c.board.rows, c.board.cols)[0]       # square side in the frame, px
    blurred = cv2.GaussianBlur(tech.render(far, 0.5, seed=4), (0, 0), 0.35 * sq)  # still found, a third blurred
    d3 = c.offer(blurred, 3.0)
    assert d3.found and not d3.accepted and d3.reason.startswith('blurred') and len(c.views) == 1
    d4 = c.offer(tech.render(far, 0.5, seed=5), 4.0)
    assert d4.accepted and len(c.views) == 2
    d5 = c.offer(np.full_like(f, 120), 5.0)
    assert not d5.found and d5.reason == 'no board found'
    assert c.undo().name == 'view_02' and len(c.views) == 1
    c.reset()


# ----------------------------------------------------------------------------- coached against random
def test_coached_needs_fewer_views_than_random(lens_coach):
    """Seeded: the coach's advice (with the technician missing it by 5 cm and 3°) against random hand-held poses from
    the same feasible set, counted from the poses."""
    c = lens_coach
    coached = CC.simulate(c, 'coached', 1)
    rand = [CC.simulate(c, 'random', s) for s in (1, 2, 3)]
    assert coached['reached']
    for r in rand:
        assert CC._views_to(r) > coached['views']
    assert coached['curve'][-1] <= 1.0 and all(b <= a * (1 + 1e-9) for a, b in zip(coached['curve'], coached['curve'][1:]))
    # the same seed gives the same session, number for number
    again = CC.simulate(c, 'coached', 1)
    assert again['curve'] == coached['curve'] and again['poses'] == coached['poses']
    c.reset()


def test_coached_beats_random_with_rendered_frames(rig, engine):
    """The same comparison through frames: each pose rendered through the camera's lens, the board detected, the view
    accepted only when sharp and new (lidar–camera mode: the lidar–camera calibration's board and detector)."""
    c = CC.Coach(rig, engine, CC.CoachSettings(mode='lidar'))
    co = CC.simulate(c, 'coached', 1, max_views=20, render=True)
    ra = CC.simulate(c, 'random', 1, max_views=20, render=True)
    assert co['reached'] and co['views'] < CC._views_to(ra)
    assert co['views'] <= 12


# ----------------------------------------------------------------------------- the study
def test_run_doc_files_and_determinism(rig, engine):
    st = CC.CoachSettings(camera='ML', random_runs=2, max_views=40)
    r1 = CC.run(rig, engine, st)
    r2 = CC.run(rig, engine, st)
    assert r1.metrics['coached_views'] == r2.metrics['coached_views'] and r1.coached['curve'] == r2.coached['curve']
    assert [r['curve'] for r in r1.random] == [r['curve'] for r in r2.random]
    assert r1.camera == 'ML' and r1.candidates > 50 and 0.3 < r1.coverage <= 1.0
    f = r1.findings[0]
    assert f['severity'] == 'good' and 'Following the coach' in f['message'] and f'{r1.coached["views"]} views' in \
        f['message'] and f['uids'] == [rig.variant.by_tag('ML').uid]
    doc = CC.to_doc(r1, rig)
    assert doc['schema'] == CC.SCHEMA and doc['frame'] == CC.FRAME and doc['settings'] == st.to_dict()
    assert doc['coached']['views'] == r1.coached['views'] and len(doc['random']) == 2 and doc['plan']
    assert doc['plan'][0]['text'].startswith('Hold the board') and 'limits' in doc and 'assumptions' in doc
    json.dumps(doc, allow_nan=False)
    names = [n for n, _ in CC.files(r1, rig)]
    assert names == ['calib_coach.json', 'calib_coach_views.csv']
    csv_text = CC.files(r1, rig)[1][1].decode()
    assert csv_text.splitlines()[0].startswith('strategy,seed,view,worst_ratio,sigma_fx')
    # cancel
    assert CC.run(rig, engine, st, cancel=lambda: True) is None
    with pytest.raises(ValueError):
        q = samples.tigor_sample_project()
        for cam in q.variant.cameras:
            cam.visible = False
        CC.Coach(q, engine, CC.CoachSettings())


def test_no_unseeded_randomness():
    src = open(CC.__file__, encoding='utf-8').read()
    assert 'np.random.default_rng(' in src
    for bad in ('np.random.rand', 'np.random.normal', 'np.random.uniform', 'np.random.seed', 'import random'):
        assert bad not in src


# ----------------------------------------------------------------------------- the page
from test_gui import app, win, pump, _wait     # noqa: E402,F401  (fixtures)


def test_live_rig_coach_mode_on_a_synthetic_source(app, win, tmp_path):
    from vantage.ui.live_calib_coach import TechnicianSource, LockBars
    from vantage.core import rig_health as rh
    st = win.store
    st.edit('streams', lambda p: p.streams.update({'F': 'synthetic:10', 'R': 'synthetic'}), 'Streams')
    win.actions_by_id['live_rig'].trigger()
    pump(app, 0.2)
    pg = win.page('rig')
    f = st.variant.by_tag('F')
    assert pg.mode == 'health' and pg.health_box.isVisible() and not pg.coach.isVisible()
    pg.set_mode('coach')
    assert pg.coach.isVisible() and not pg.health_box.isVisible()
    _wait(app, lambda: pg.coach.state['built'], 60)
    assert pg.coach.state['cam_tag'] == 'F' and [t.isVisible() for t in pg.tiles.values()].count(True) == 1
    assert pg.tiles[f.uid].isVisible() and len(pg.coach.bars.locks) == len(CC.LENS_LOCKS)
    assert pg.coach.advice.text().startswith('Hold the board') and 'vehicle frame' in pg.coach.advice_v.text()
    assert pg.coach.overlay()['ghost'] is not None and not pg.coach.b_finish.isEnabled()
    # the simulated technician takes over F's synthetic stream and follows the advice
    pg.coach.set_technician('coached')
    pg.b_start.click()
    assert pg.running
    _wait(app, lambda: isinstance(pg.monitor.streams['F'].source, TechnicianSource), 10)
    worst = []

    def progressed():
        s = pg.coach.state.get('status')
        if s is not None and (not worst or worst[-1][0] != pg.coach.state['views']):
            worst.append((pg.coach.state['views'], s['worst']))
        return pg.coach.state['views'] >= 4
    _wait(app, progressed, 150)
    ws = [w for _, w in worst]
    assert all(b <= a * (1 + 1e-9) for a, b in zip(ws, ws[1:])) and ws[-1] < ws[0]
    assert pg.coach.line.text.text().startswith('View') or pg.coach.state['det'] is not None
    ov = pg.coach.overlay()
    assert len(ov['coverage']) >= 4 * 54 and ov['ghost']['ok']
    img = pg.tiles[f.uid].grab().toImage()
    assert img.width() > 100
    assert LockBars().fill_of(1.0) == 1.0 and LockBars().fill_of(100.0) == 0.0
    # the other cameras keep their health statistics meanwhile
    assert pg.snap['cameras']['R']['frames'] > 0 and pg.coach_state()['camera'] == 'F'
    # finish: the lens calibration opens with the accepted views as its input set
    pg.coach._wait_idle()
    n = pg.coach.coach.views.__len__()
    d = pg.coach.finish()
    assert d is not None and len(d.views) == n and d.cols.value() == 9 and d.rows.value() == 6 and \
        d.square.value() == 60 and d.cam.currentData() == f.uid and d.b_run.isEnabled()
    d.close()
    paths = pg.coach.save_session(str(tmp_path / 'session'))
    doc = json.loads((tmp_path / 'session' / 'calib_coach_session.json').read_text())
    assert doc['schema'] == CC.SCHEMA and len(doc['views']) == n and len(paths) == n + 1
    # undo and reset
    pg.coach.undo()
    _wait(app, lambda: pg.coach.state['views'] <= n - 1, 20)
    # a settings edit is undoable and keeps the views (same camera and board)
    pg.coach._set('need_deg', 0.5, 'Coach requirement 0.5°')
    _wait(app, lambda: (pg.coach.state.get('need') or {}).get('deg') == 0.5, 30)
    assert pg.coach.state['views'] >= 3 and 'coach' in st.undo.undoText().lower()
    st.undo.undo()
    _wait(app, lambda: (pg.coach.state.get('need') or {}).get('deg') == 0.1, 30)
    # back to health: F gets its own synthetic stream again and the page behaves as before
    pg.set_mode('health')
    pump(app, 0.3)
    assert type(pg.monitor.streams['F'].source) is rh.SyntheticSource and pg.running
    assert all(t.isVisible() for t in pg.tiles.values()) and pg.coach_state() is None
    pg.stop()
    assert not pg.running


def test_live_rig_coach_lidar_finish(app, win, tmp_path):
    """Lidar–camera mode: the simulated technician's views (and its lidar's scans of the same poses) written as
    image + point-cloud pairs and handed to Calibrate ▸ Lidar–camera calibration, which calibrates from them."""
    st = win.store
    v = next(v for v in st.project.variants if v.sensors)
    st.edit('variants', lambda p: setattr(p, 'active_uid', v.uid), 'The data rig')
    st.edit('analysis', lambda p: setattr(p.analysis, 'calib_coach', {'mode': 'lidar'}), 'Coach: lidar')
    cam_tag = CC.Coach(st.project, Engine(), CC.CoachSettings(mode='lidar')).tag
    st.edit('streams', lambda p: p.streams.update({cam_tag: 'synthetic'}), 'Streams')
    win.set_page('rig')
    pump(app, 0.2)
    pg = win.page('rig')
    pg.set_mode('coach')
    _wait(app, lambda: pg.coach.state['built'] and pg.coach.state['cam_tag'] == cam_tag, 60)
    assert pg.coach.state['lidar'] == 'L1' and len(pg.coach.bars.locks) == 6
    assert pg.coach.b_finish.text().startswith('Lidar')
    pg.coach.set_technician('coached')
    pg.start()
    _wait(app, lambda: pg.coach.state['views'] >= 4, 150)
    pg.stop()
    out = tmp_path / 'lc'
    pg.coach._wait_idle()
    assert pg.coach.finish(str(out), run=False) == str(out)
    _wait(app, lambda: win.page_name() == 'lidar_calib', 60)
    names = sorted(os.listdir(out))
    assert 'pose01.png' in names and 'pose01.pcd' in names and 'calib_coach_session.json' in names
    lc = LC.LidarCalibSettings.from_dict(st.project.analysis.lidar_calib)
    assert lc.source == 'folder' and lc.folder == str(out) and lc.camera == cam_tag and lc.lidar == 'L1'
    res = LC.run(st.project.snapshot(), None, lc)
    assert res.T is not None and res.verdict not in ('failed', 'unobservable')
    assert len(res.used) >= 3
