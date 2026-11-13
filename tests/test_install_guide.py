"""Calibrate ▸ Install guide (core.install_guide, assistant.install_coach, ui.workspaces.guide_ws): the mathematics
against independent references (the observability study, Monte Carlo, the truth of synthetic captures), the stages end
to end on a rehearsed rig, the coach's number guard and request against the fake Anthropic server, and the page."""
import copy
import json
import math
import os
import tempfile
import time

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
import pytest

from vantage.core import install_guide as G
from vantage.core.mathutil import rot_zyx


# ---------------------------------------------------------------------------------------------------------- fixtures
@pytest.fixture(scope='module')
def engine():
    from vantage.core.analysis import Engine
    return Engine()


def _v4(p):
    """The Tigor sample's six-camera variant with the narrow lenses moved to the front and rear windows and the mirror
    cameras turned back (the layout the AGX Orin rig uses)."""
    from vantage.core.project import Variant, new_uid
    v3 = next(v for v in p.variants if v.name.startswith('6-cam') and any(c.tag == 'QL' for c in v.cameras))
    v = Variant.from_dict(copy.deepcopy(v3.to_dict()))
    v.uid, v.name = new_uid(), '6-cam · narrow F/R'
    for c in v.cameras:
        c.uid, c.link_uid = new_uid(), ''
        if c.tag == 'ML':
            c.yaw, c.pitch = 105.0, 55.0
        if c.tag == 'MR':
            c.yaw, c.pitch = -105.0, 55.0
        if c.tag == 'QL':
            c.tag, c.lens_id, c.x, c.y, c.z, c.yaw, c.pitch, c.roll = 'FN', 'h03r2', 1.65, 0.0, 1.40, 0.0, 3.0, 0.0
        if c.tag == 'QR':
            c.tag, c.lens_id, c.x, c.y, c.z, c.yaw, c.pitch, c.roll = 'RN', 'h03r2', 0.25, 0.0, 1.38, 180.0, 3.0, 0.0
    p.variants.append(v)
    p.active_uid = v.uid
    return p


@pytest.fixture(scope='module')
def planned(engine):
    """The narrow F/R rig with its board plan applied."""
    from vantage.core import samples
    p = _v4(samples.tigor_sample_project())
    plan = G.plan_boards(p, engine, G.GuideSettings())
    G.apply_plan(p, plan)
    return p, plan


def _truth(p, seed, ang=0.6, pos=0.004):
    rng = np.random.default_rng(seed)
    out = {}
    for c in p.variant.cameras:
        d = rng.normal(0, ang, 3)
        out[c.uid] = (rot_zyx(c.yaw + d[0], c.pitch + d[1], c.roll + d[2]), c.pos() + rng.normal(0, pos, 3))
    return out


# ---------------------------------------------------------------------------------------------------------- formulas
def test_identify_covered():
    base = {'F': 120.0, 'R': 110.0, 'ML': 100.0}
    assert G.identify_covered(base, {'F': 15.0, 'R': 108.0, 'ML': 101.0})[0] == 'F'
    k, why = G.identify_covered(base, {'F': 15.0, 'R': 12.0, 'ML': 101.0})
    assert k is None and 'several' in why
    k, why = G.identify_covered(base, {'F': 118.0, 'R': 109.0, 'ML': 99.0})
    assert k is None and 'cover' in why
    k, why = G.identify_covered(base, {'F': 15.0, 'R': 70.0, 'ML': 100.0})
    assert k is None and 'light' in why


def test_shim_and_ground_shift():
    assert G.shim_mm(1.0, 40.0) == pytest.approx(0.698, abs=1e-3)
    # d = h / tan(alpha): |dd/dalpha| = (h² + d²) / h
    h, d, delta = 0.62, 2.0, 0.1
    alpha = math.atan2(h, d)
    d2 = h / math.tan(alpha - math.radians(delta))
    assert G.ground_shift_mm(h, d, delta) == pytest.approx(abs(d2 - d) * 1000, rel=0.01)


def test_aim_instructions_directions():
    st = G.GuideSettings()
    a = {'ok': True, 'd_yaw_deg': 2.0, 'd_pitch_deg': -1.5, 'd_roll_deg': 0.3,
         'determined': {'yaw': True, 'pitch': True, 'roll': True}}
    lines = G.aim_instructions(a, st)
    assert 'turn it 2.0° right' in lines[0]           # turned left (yaw +): back to the right
    assert 'tilt it 1.5° down' in lines[1]            # pitch − is tilted up: down again
    assert 'within' in lines[2]
    a['determined']['roll'] = False
    assert 'boards will measure' in G.aim_instructions(a, st)[2]


def test_roll_sign_is_right_side_down():
    from vantage.core.mathutil import BODY_TO_OPTICAL
    W2O = BODY_TO_OPTICAL @ rot_zyx(0, 0, 10).T
    assert W2O[0][2] < 0                               # the image's right axis points down: right side down


# ---------------------------------------------------------------------------------------------------------- covariance
def test_joint_covariance_matches_observability(engine):
    """The guide's covariance of the planned floor solve against the observability study's (independent code)."""
    import dataclasses
    from vantage.core import samples
    from vantage.core import observability as O
    p = samples.tigor_sample_project()
    v = next(v for v in p.variants if v.name.startswith('6-cam') and any(c.tag == 'QL' for c in v.cameras))
    p.active_uid = v.uid
    cams = [c for c in v.cameras if c.visible]          # all of them: shared boards tie the cameras together
    st = G.GuideSettings(safety=1.0, board_sigma_mm=5.0, board_sigma_deg=0.25, pixel_sigma=0.3, min_square_px=8.0)
    cov = G.planned_floor_covariance(p, engine, cams, st)
    ost = dataclasses.replace(O.settings_of(p), board_sigma_mm=5.0, board_sigma_deg=0.25, pixel_sigma=0.3,
                              suggest=False, fusion=False)
    res = O.run(p, engine, ost)
    for c in cams:
        theirs = res.camera(c.tag)['sigma']
        if theirs is None:
            assert cov['cams'][c.uid] is None, c.tag
            continue
        ours = G.sigmas(G.planned_pose(c), cov['cams'][c.uid])
        for k in ('yaw_deg', 'pitch_deg', 'roll_deg', 'x_mm', 'y_mm'):
            assert ours[k] == pytest.approx(theirs[k], rel=0.12), (c.tag, k, ours[k], theirs[k])


def test_chain_covariance_matches_monte_carlo(engine):
    """A child camera chained to a parent through a board on a stand: the predicted covariance (first-order error
    propagation with the parent's own error) against the spread of solve_chain on noisy corners and a parent drawn
    from its covariance."""
    from vantage.core import samples
    p = _v4(samples.tigor_sample_project())
    par = next(c for c in p.variant.cameras if c.tag == 'F')
    ch = next(c for c in p.variant.cameras if c.tag == 'FN')
    ip, ic = G._intr(p, par), G._intr(p, ch)
    pp, pc = G.planned_pose(par), G.planned_pose(ch)
    st = G.GuideSettings(pixel_sigma=0.3)
    b = G.StandBoard(x=7.0, y=0.0, z=0.55, yaw=180.0, rows=6, cols=8, square_mm=100.0, parent='F', child='FN')
    L = b.local_corners().reshape(-1, 3)
    cov_p = np.diag([2e-4, 2e-4, 2e-4, 2e-3, 2e-3, 2e-3]) ** 2
    pred = G.chain_covariance(ip, pp, cov_p, ic, pc, b.frame(), L, st.pixel_sigma)['child']
    rng = np.random.default_rng(4)
    uv_p = G._uv_board(ip, pp, b.frame(), L, (1, 1))
    uv_c = G._uv_board(ic, pc, b.frame(), L, (1, 1))
    draws = []
    Lc = np.linalg.cholesky(cov_p)
    for _ in range(80):
        xp = Lc @ rng.normal(size=6)
        p_est = G.perturb(pp, xp)                       # the parent as solved, off by its own error
        op = uv_p + rng.normal(0, st.pixel_sigma, uv_p.shape)
        oc = uv_c + rng.normal(0, st.pixel_sigma, uv_c.shape)
        rp, _ = ip.unproject(op[:, 0], op[:, 1])
        rc, _ = ic.unproject(oc[:, 0], oc[:, 1])
        r = G.solve_chain(ip, p_est, cov_p, (op, rp), ic, (oc, rc), b, st, init_c=pc)
        assert r['ok']
        R, pos = r['pose']
        # the child's error in perturb parameters (rotation vector, shift)
        import cv2
        dR = G.w2o_of(R) @ G.w2o_of(pc[0]).T
        w, _ = cv2.Rodrigues(dR)
        draws.append(np.r_[w.ravel(), pos - pc[1]])
    emp = np.cov(np.array(draws).T)
    # each axis' spread within the sampling error of 80 draws (and a little linearisation)
    for i in range(6):
        assert math.sqrt(emp[i, i]) == pytest.approx(math.sqrt(pred[i, i]), rel=0.35), i


def test_floor_plan_meets_requirement(planned):
    p, plan = planned
    rows = {r['camera']: r for r in plan['cameras']}
    assert set(rows) == {'F', 'R', 'ML', 'MR', 'FN', 'RN'}
    assert all(r['ok'] for r in rows.values()), rows
    assert plan['check'] and plan['check']['meets'] and len(plan['check']['seen_by']) >= 2
    st = G.GuideSettings.from_dict(p.install_guide['settings'])
    assert st.check_boards == [plan['check']['board']]


# ---------------------------------------------------------------------------------------------------------- images
def test_aim_from_the_body_outline(planned, engine):
    p, _ = planned
    for tag, d in (('F', (0.8, -0.6, 0.5)), ('MR', (-1.1, 0.4, -0.7))):
        c = next(c for c in p.variant.cameras if c.tag == tag)
        pose = (rot_zyx(c.yaw + d[0], c.pitch + d[1], c.roll + d[2]), c.pos())
        img = G.render_view(p, engine, c, pose, w=1920, seed=1)
        a = G.estimate_aim(img, G.aim_model(p, engine, c), G._intr(p, c))
        assert a['ok']
        est = (a['d_yaw_deg'], a['d_pitch_deg'], a['d_roll_deg'])
        for k, (e, t) in enumerate(zip(est, d)):
            if a['determined'][('yaw', 'pitch', 'roll')[k]]:
                assert abs(e - t) < 0.3, (tag, k, e, t)


def test_aim_says_when_the_body_is_not_in_view(planned, engine):
    p, _ = planned
    c = next(c for c in p.variant.cameras if c.tag == 'FN')
    img = G.render_view(p, engine, c, w=960, seed=1)
    a = G.estimate_aim(img, G.aim_model(p, engine, c), G._intr(p, c))
    assert not a['ok'] and 'boards' in a['why']


def test_detect_on_plane_corner_accuracy(planned, engine):
    """Boards found in a capture from a pose 0.6° off the one the plane view is made with; the corners against their
    true projection."""
    p, _ = planned
    truth = _truth(p, 3)
    errs, found = [], 0
    for tag in ('F', 'R', 'MR'):
        c = next(c for c in p.variant.cameras if c.tag == tag)
        intr = G._intr(p, c)
        img = G.render_view(p, engine, c, truth[c.uid], w=1920, seed=0)
        obs = G.detect_floor(img, p, c, G.planned_pose(c), p.targets)
        found += len(obs)
        for o in obs:
            tp, _ = G.project_pts(intr, truth[c.uid], o.world)
            errs.append(np.linalg.norm(o.image_pts - tp, axis=1))
    assert found >= 6
    e = np.concatenate(errs)
    assert np.median(e) < 0.3 and np.percentile(e, 95) < 0.8


def test_stand_board_render_and_detect(engine):
    """A board on a stand drawn into a capture and found on its own plane, both cameras."""
    from vantage.core import samples
    p = _v4(samples.tigor_sample_project())
    c = next(c for c in p.variant.cameras if c.tag == 'FN')
    b = G.StandBoard(x=7.5, y=0.0, z=0.8, yaw=180.0, rows=7, cols=9, square_mm=120.0, parent='F', child='FN')
    pose = G.planned_pose(c)
    img = G.render_view(p, engine, c, pose, w=1920, seed=2, stands=[b])
    intr = G._intr(p, c)
    loc = b.local_corners()[..., :2].reshape(-1, 2)
    r = G.detect_on_plane(img, intr, pose, b.frame(), loc, b.size_m, b.square_mm / 1000, (b.cols - 1, b.rows - 1))
    assert r is not None
    tp, _ = G.project_pts(intr, pose, b.world().reshape(-1, 3))
    assert np.median(np.linalg.norm(r[0] - tp, axis=1)) < 0.3


# ---------------------------------------------------------------------------------------------------------- end to end
def test_session_end_to_end(planned, engine):
    """A rehearsed installation: the cameras a little off their plan, every stage's work on synthetic captures; the
    solved poses within 4σ of the truth on every axis, the coverage robust where the plan covers, the checks pass."""
    p, _ = planned
    truth = _truth(p, 11)
    s = G.GuideSession(p, engine)
    frames = {c.tag: G.render_view(p, engine, c, truth[c.uid], w=1920, seed=c.slot) for c in s.cams}
    s.update_health({'cameras': {c.tag: {'frames': 300, 'fps': 30.0, 'status': 'ok'} for c in s.cams}, 'rig': {}})
    assert s.connect_status()['ok']
    s.identify_skip()
    s.lens_accept = True
    for c in s.cams:
        s.aim_camera(c.tag, frames[c.tag])
    for _ in range(2):
        for c in s.cams:
            s.feed(c.tag, frames[c.tag])
    sol = s.solve()
    assert all(r['solved'] for r in sol['cameras']), sol
    for c in s.cams:
        pose = sol['poses'][c.uid]
        sig = G.sigmas(pose, sol['covs'][c.uid])
        dt, de = G.pose_delta(truth[c.uid], c), G.pose_delta(pose, c)
        for k in G.SIGMA_KEYS:
            assert abs(de[k] - dt[k]) <= 4 * sig[k], (c.tag, k, de[k], dt[k], sig[k])
    cov = s.coverage()
    assert cov['ok'] and cov['robust_pct'] >= cov['plan_covered_pct'] - 1.0
    ver = s.verify()
    assert any(t['test'].startswith('seams') and t['pass'] for t in ver['tests'])
    chk = [t for t in ver['tests'] if t['test'].startswith('check board')]
    assert chk and chk[0]['pass'], chk
    doc = s.report()
    json.dumps(doc)
    assert doc['schema'] == G.SCHEMA and len(doc['stages']) == len(G.STAGES)
    built = s.as_built()
    assert {c.uid for c in built.cameras} == {c.uid for c in p.variant.cameras}


def test_robust_coverage_shrinks_with_uncertainty(planned, engine):
    p, _ = planned
    poses = {c.uid: G.planned_pose(c) for c in p.variant.cameras}
    st = G.GuideSettings()
    exact = G.robust_coverage(p, engine, poses, {}, st)
    loose = G.robust_coverage(p, engine, poses, {c.uid: np.eye(6) * (0.01 ** 2) for c in p.variant.cameras}, st)
    assert exact['uncertain_pct'] == 0.0
    assert exact['robust_pct'] == pytest.approx(exact['plan_covered_pct'], abs=0.2)
    assert loose['robust_pct'] < exact['robust_pct'] and loose['uncertain_pct'] > 0


def test_project_keeps_the_guide(planned):
    from vantage.core.project import Project
    p, plan = planned
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'rig.vantage')
        p.save(path)
        q = Project.load(path)
    assert q.install_guide['settings']['check_boards'] == p.install_guide['settings']['check_boards']
    assert len(q.targets) == len(plan['targets'])


# ---------------------------------------------------------------------------------------------------------- coach
def test_number_guard():
    from vantage.assistant.install_coach import number_guard, numbers_in
    state = {'solve': {'cameras': [{'camera': 'F', 'ratio': 1.137, 'sigma': {'yaw_deg': 0.0712}}]},
             'aim': {'cameras': [{'instructions': ['yaw: +2.3° from the plan: turn it 2.3° right (a 1.6 mm shim)']}]},
             'coverage': {'robust_pct': 99.637}}
    allowed = numbers_in(state)
    text = ('Turn F 2.3° right with a 1.6 mm shim. F is at 1.14× the requirement, yaw 0.07°. '
            'Coverage is 99.6%. Move T12 by 35 cm. Check the 3 boards.')
    clean, removed = number_guard(text, allowed)
    assert removed == ['Move T12 by 35 cm.']
    assert 'Turn F 2.3°' in clean and '99.6%' in clean and '3 boards' in clean


def test_should_ask_policy():
    from vantage.assistant.install_coach import Coach
    c = Coach(lambda: {}, lambda t: None, key='k')
    st = {'stage': 'aim', 'stages': [{'stage': 'aim', 'done': False}], 'connect': {'cameras': []}, 'solve': {}}
    assert c.should_ask(st, now=1000.0) == 'the guide started'
    c.last_sig = __import__('vantage.assistant.install_coach', fromlist=['signature']).signature(st)
    c.last_t = 1000.0
    st2 = dict(st, stages=[{'stage': 'aim', 'done': True}])
    assert c.should_ask(st2, now=1005.0) is None              # too soon
    assert c.should_ask(st2, now=1030.0) == 'stage done: aim'


def test_coach_against_fake_anthropic():
    """One exchange: the coach reads the state through its tool, looks at a camera (the JPEG goes in the tool
    result), answers; a figure the guide never gave is removed from the answer."""
    from fake_anthropic import FakeAnthropic
    from vantage.assistant.install_coach import Coach
    srv = FakeAnthropic(key='sk-ant-test').start()
    try:
        state = {'stage': 'aim', 'stages': [], 'aim': {'cameras': [{'camera': 'R', 'd_pitch_deg': 1.3}]}}
        frame = np.full((480, 640, 3), 90, np.uint8)
        srv.script.append(srv.reply(tools=[('get_state', {'section': 'aim'}), ('look', {'camera': 'R'})]))
        srv.script.append(srv.reply(text='Tilt R 1.3° up. Then raise it 4.5 cm.'))
        c = Coach(lambda: state, lambda t: frame, key='sk-ant-test', url=srv.url)
        r = c.ask(reason='stage done: connect')
        assert r['error'] == '', r
        assert r['text'] == 'Tilt R 1.3° up.' and r['removed'] == ['Then raise it 4.5 cm.']
        body = srv.requests[0]
        assert body['model'] == 'claude-opus-5-5' and body['output_config']['effort'] == 'low'
        assert {t['name'] for t in body['tools']} == {'get_state', 'look', 'coverage_whatif'}
        assert all(t.get('strict') for t in body['tools'])
        assert srv.images and srv.images[0]['media_type'] == 'image/jpeg'
    finally:
        srv.stop()


# ---------------------------------------------------------------------------------------------------------- page
def test_page_builds_and_walks_the_stages(planned):
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from vantage.ui.mainwindow import MainWindow
    p, _ = planned
    w = MainWindow(p, soft_3d=True)
    try:
        w.resize(1600, 1000)
        w.show()
        t = time.time()
        w.set_page('guide')
        pg = w.page('guide')
        assert time.time() - t < 10
        for s in G.STAGES:
            pg.set_stage(s)
            app.processEvents()
            assert pg.session.stage == s
            assert pg.detail.toHtml()
        assert 'guide' in w.actions_by_id.get('install_guide').text().lower()
        pg.stop()
    finally:
        w.store.undo.setClean()
        w.close()
        app.processEvents()
