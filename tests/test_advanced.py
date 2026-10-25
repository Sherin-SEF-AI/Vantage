"""Installation check, stitching, load cases, link budget, capture helpers and the batch command line."""
import argparse
import json
import math
import os

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
import pytest

from vantage.core import samples, capture, rules
from vantage.core.analysis import Engine
from vantage.core.install import InstallSettings, check_camera, detect_boards, inner_corners
from vantage.core.loadcases import body_pose, evaluate
from vantage.core.mathutil import rot_zyx
from vantage.core.stitch import build_lut, stitch, lut_files
from vantage.core.synth import render_capture


@pytest.fixture(scope='module')
def qapp():
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope='module')
def sample(qapp):
    p = samples.tigor_sample_project()
    e = Engine()
    return p, e, e.analyse(p)


def _err_pose(cam, d):
    return rot_zyx(cam.yaw + d[0], cam.pitch + d[1], cam.roll + d[2]), cam.pos() + np.asarray(d[3:])


ERRORS = {'F': (0.6, -0.7, 0.4, 0.007, -0.006, 0.004), 'ML': (-1.6, 0.5, 0.9, -0.012, 0.01, 0.006),
          'QR': (0.3, 1.5, -0.6, 0.006, 0.016, -0.009)}


@pytest.fixture(scope='module')
def captures(sample):
    """Half-resolution synthetic captures of three cameras mounted with known errors."""
    p, e, _ = sample
    out = {}
    for tag, d in ERRORS.items():
        c = p.variant.by_tag(tag)
        out[tag] = render_capture(p, e, c, 960, 768, pose=_err_pose(c, d), seed=11)
    return out


# ----------------------------------------------------------------------------- installation check
def test_inner_corners_geometry(sample):
    p, _, _ = sample
    t = p.targets[0]
    W = inner_corners(t)
    assert W.shape == (t.rows - 1, t.cols - 1, 3)
    sq = t.square_mm / 1000
    assert np.allclose(np.linalg.norm(W[0, 1] - W[0, 0]), sq) and np.allclose(np.linalg.norm(W[1, 0] - W[0, 0]), sq)
    assert np.allclose(W.reshape(-1, 3).mean(axis=0)[:2], (t.x, t.y), atol=1e-9)


@pytest.mark.parametrize('tag', list(ERRORS))
def test_install_check_recovers_mounting_error(sample, captures, tag):
    p, _, _ = sample
    c = p.variant.by_tag(tag)
    d = np.array(ERRORS[tag])
    r = check_camera(p, c, captures[tag])
    assert r.pos is not None, r.message
    assert len(r.boards) >= 2 and r.rms_px < 1.0
    assert np.abs(r.dang_deg - d[:3]).max() < 0.3
    assert np.abs(r.dpos_mm - d[3:] * 1000).max() < 6.0
    # F is inside the default 1° / 10 mm tolerance; ML and QR are outside it but within 3x
    assert r.status == ('good' if tag == 'F' else 'warning')


def test_install_check_status_levels(sample, captures):
    p, _, _ = sample
    c = p.variant.by_tag('F')
    assert check_camera(p, c, captures['F'], InstallSettings(pos_tol_mm=20, ang_tol_deg=2)).status == 'good'
    assert check_camera(p, c, captures['F'], InstallSettings(pos_tol_mm=2, ang_tol_deg=0.1)).status == 'serious'


def test_install_check_large_error_and_no_boards(sample):
    p, e, _ = sample
    c = p.variant.by_tag('R')
    img = render_capture(p, e, c, 960, 768, pose=_err_pose(c, (2.5, -3.5, 1.0, -0.02, -0.02, 0.01)), seed=3)
    r = check_camera(p, c, img)
    assert r.status == 'serious' and r.pos is not None
    assert abs(r.dang_deg[1] + 3.5) < 0.6
    blank = render_capture(p, e, c, 960, 768, targets=[], seed=3)
    r = check_camera(p, c, blank)
    assert r.status == 'serious' and r.pos is None and 'No calibration board' in r.message
    assert detect_boards(blank, p, c) == []


def test_install_check_accepts_rgb_and_other_resolutions(sample, captures):
    p, _, _ = sample
    c = p.variant.by_tag('ML')
    import cv2
    rgb = np.repeat(cv2.resize(captures['ML'], (1280, 1024))[..., None], 3, axis=2)
    r = check_camera(p, c, rgb)
    assert r.pos is not None and np.abs(r.dang_deg - np.array(ERRORS['ML'][:3])).max() < 0.3


def test_board_detection_does_not_depend_on_what_ran_before(sample, captures):
    """OpenCV's SB detector draws on OpenCV's global random generator: with OpenCV 5 (the downloads' OpenCV) a check
    of camera F gave other corners, for one board a grid 75 px off, after a check of another camera had run."""
    import copy
    import cv2
    p, e, _ = sample
    rng = np.random.default_rng(5)                  # the layout of test_joint_solve_ties_cameras_together_…
    laid = copy.deepcopy(p.targets)
    for t in laid:
        t.x += rng.uniform(-0.02, 0.02)
        t.y += rng.uniform(-0.02, 0.02)
        t.yaw += rng.uniform(-1, 1)
    f = p.variant.cameras[0]
    d = np.r_[rng.uniform(-1, 1, 3), rng.uniform(-0.01, 0.01, 3)]
    img = render_capture(p, e, f, 960, 768, pose=_err_pose(f, d), targets=laid, seed=f.slot)
    a = check_camera(p, f, img).detected
    ml = p.variant.by_tag('ML')
    check_camera(p, ml, np.repeat(cv2.resize(captures['ML'], (1280, 1024))[..., None], 3, axis=2))
    b = check_camera(p, f, img).detected
    assert [o.name for o in a] == [o.name for o in b] and len(a) == 3
    assert all(np.array_equal(x.image_pts, y.image_pts) for x, y in zip(a, b))


# ----------------------------------------------------------------------------- stitching
def test_stitch_measured_poses_remove_the_error(sample):
    """Stitched with the planned poses, captures from mis-mounted cameras ghost; with the measured poses
    they match the stitch of correctly mounted cameras again."""
    p, e, res = sample
    cams = [c for c in p.variant.cameras]
    rng = np.random.default_rng(4)
    ref, bad, poses = {}, {}, {}
    for c in cams:
        d = np.r_[rng.uniform(-1.2, 1.2, 3), rng.uniform(-0.01, 0.01, 3)]
        ref[c.uid] = render_capture(p, e, c, 960, 768, seed=1, noise=0)
        bad[c.uid] = render_capture(p, e, c, 960, 768, pose=_err_pose(c, d), seed=1, noise=0)
        r = check_camera(p, c, bad[c.uid])
        assert r.pos is not None, (c.tag, r.message)
        poses[c.uid] = (rot_zyx(r.yaw, r.pitch, r.roll), r.pos)
    lp = build_lut(p, e, res, res=0.04)
    lm = build_lut(p, e, res, res=0.04, poses=poses)
    for lut in (lp, lm):
        tot = sum(lut.weight.values())
        assert np.allclose(tot[lut.covered], 1, atol=1e-5) and not (lut.covered & lut.body).any()
    s_ref = stitch(lp, ref).astype(float)
    err_planned = np.abs(stitch(lp, bad).astype(float) - s_ref)[lp.covered].mean()
    err_measured = np.abs(stitch(lm, bad).astype(float) - s_ref)[lm.covered & lp.covered].mean()
    assert err_measured < 0.5 * err_planned, (err_measured, err_planned)


def test_lut_files_layout(sample):
    p, e, res = sample
    lut = build_lut(p, e, res, res=0.05)
    files = dict(lut_files(lut, p))
    meta = json.loads(files['bev_lut.json'])
    assert meta['width'] == lut.nx and meta['height'] == lut.ny
    for cam in meta['cameras']:
        assert len(files[cam['map']]) == lut.nx * lut.ny * 2 * 4
        assert len(files[cam['weight']]) == lut.nx * lut.ny
    assert len(files['body_mask.bin']) == lut.nx * lut.ny


# ----------------------------------------------------------------------------- load cases
def test_body_pose_sign():
    R, t, pitch = body_pose(30.0, 10.0, 2.5)
    assert pitch > 0                                  # front lower than rear: nose down
    nose = R @ np.array([2.5, 0, 0.5]) + t
    assert nose[2] < 0.5 - 0.019 and abs(t[2] + 0.01) < 1e-12


def test_load_cases_grow_with_load(sample):
    p, e, _ = sample
    rs = evaluate(e, p, [('kerb', 0, 0), ('light', 5, 5), ('heavy', 25, 60)], cell=0.1)
    assert rs[0].ground_shift_cm < 1e-6 and rs[0].seam_ghost_cm < 1e-6 and rs[0].status == "good"
    assert rs[1].ground_shift_cm < rs[2].ground_shift_cm and rs[1].seam_ghost_cm < rs[2].seam_ghost_cm
    assert rs[2].status == 'serious' and rs[2].pitch_deg < 0


# ----------------------------------------------------------------------------- link budget
def test_link_budget(sample):
    p, _, res = sample
    g = rules.camera_gbps(p, p.variant.cameras[0])
    assert g == pytest.approx(1920 * 1536 * 30 * 16 * 1.10 / 1e9, rel=1e-6)
    rows = {r[0]: r for r in rules.link_budget(p)}
    assert rows['A'][3] == pytest.approx(4 * g) and rows['A'][3] < rows['A'][4]
    assert not any('Gbps' in i.message for i in rules.collect_issues(p, res, None, None))
    import copy
    q = copy.deepcopy(p)
    q.ecu.csi_gbps_per_deser = 5.0
    msgs = [i for i in rules.collect_issues(q, res, None, None) if 'Gbps' in i.message]
    assert msgs and msgs[0].severity == 'serious'


# ----------------------------------------------------------------------------- capture helpers
def test_match_folder_and_image_io(tmp_path):
    img = (np.arange(48 * 64) % 251).astype(np.uint8).reshape(48, 64)
    for name in ('F.png', 'cam_QR.png', 'ML-capture.png', 'MR_old.png', 'QL.txt'):
        if name.endswith('.png'):
            capture.save_image(str(tmp_path / name), img)
        else:
            (tmp_path / name).write_text('x')
    found = capture.match_folder(str(tmp_path), ['F', 'R', 'ML', 'MR', 'QL', 'QR'])
    assert {k: os.path.basename(v) for k, v in found.items()} == {'F': 'F.png', 'QR': 'cam_QR.png', 'ML': 'ML-capture.png',
                                                                  'MR': 'MR_old.png'}
    assert np.array_equal(capture.load_image(found['F']), img)
    rgb = np.dstack([img, img // 2, img // 3])
    capture.save_image(str(tmp_path / 'c.png'), rgb)
    assert np.array_equal(capture.load_image(str(tmp_path / 'c.png')), rgb)
    with pytest.raises(ValueError):
        capture.load_image(str(tmp_path / 'QL.txt'))
    assert capture.default_spec(3) == '/dev/video2'


def test_live_source_reports_unopenable_device():
    ls = capture.LiveSource('/dev/video-does-not-exist')
    ls.start()
    ls._thread.join(5)
    assert not ls.running and ls.error and ls.latest() is None
    ls.stop()


# ----------------------------------------------------------------------------- batch mode
def _args(**kw):
    from vantage import batch
    ap = argparse.ArgumentParser()
    ap.add_argument('path', nargs='?')
    ap.add_argument('--sample', action='store_true')
    ap.add_argument('--template')
    batch.add_arguments(ap)
    a = ap.parse_args([])
    a.sample = True
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def test_batch_analyse_and_export(sample, tmp_path):
    from vantage import batch
    out = tmp_path / 'a.json'
    code = batch.run(_args(analyse=str(out), export=str(tmp_path / 'x'), formats='extrinsics_json,bev_lut'))
    doc = json.loads(out.read_text('utf-8'))
    assert code == 2                                   # the sample has a small blind patch (critical issue)
    assert doc['metrics']['blind_m2'] == pytest.approx(0.010, abs=0.003)
    assert [c['tag'] for c in doc['cameras']] == ['F', 'R', 'ML', 'MR', 'QL', 'QR']
    assert doc['link_budget'][0]['load_pct'] == pytest.approx(62.3, abs=0.2)
    assert (tmp_path / 'x' / 'extrinsics.json').is_file() and (tmp_path / 'x' / 'bev_lut' / 'F_map.bin').is_file()
    with pytest.raises(SystemExit):
        batch.run(_args(export=str(tmp_path / 'y'), formats='viewport_png'))


def test_batch_install_check_and_as_built(sample, tmp_path):
    from vantage import batch
    from vantage.core.project import Project
    p, e, _ = sample
    caps = tmp_path / 'caps'
    caps.mkdir()
    rng = np.random.default_rng(2)
    truth = {}
    for c in p.variant.cameras:
        d = np.r_[rng.uniform(-0.5, 0.5, 3), rng.uniform(-0.004, 0.004, 3)]
        truth[c.tag] = d
        capture.save_image(str(caps / f'{c.tag}.png'), render_capture(p, e, c, 960, 768, pose=_err_pose(c, d), seed=5))
    rep = tmp_path / 'install.json'
    built = tmp_path / 'built.vantage'
    code = batch.run(_args(install_check=str(caps), report=str(rep), lut_out=str(tmp_path / 'lut'), as_built=str(built)))
    doc = json.loads(rep.read_text('utf-8'))
    assert code == 0, [(c['tag'], c['status'], c['message']) for c in doc['cameras']]
    for row in doc['cameras']:
        assert row['status'] == 'good'
        assert np.abs(np.array(row['delta_deg']) - truth[row['tag']][:3]).max() < 0.3
    assert (tmp_path / 'lut' / 'bev_lut.json').is_file() and (tmp_path / 'lut' / 'birds-eye-measured.png').is_file()
    q = Project.load(str(built))
    assert q.variant.name.endswith('as-built') and len(q.variants) == len(p.variants) + 1
    ml = q.variant.by_tag('ML')
    meas = next(r for r in doc['cameras'] if r['tag'] == 'ML')['measured']
    assert ml.yaw == pytest.approx(meas['yaw'], abs=1e-3) and ml.x == pytest.approx(meas['x'], abs=1e-4)
    assert all(c.link_uid == '' for c in q.variant.cameras)
    assert doc['joint']['shared_corners'] > 0 and 'own_solve' in doc['cameras'][0]
    assert set(doc['exposure_gains']) == {c.tag for c in p.variant.cameras}
    # a camera with no image fails the check
    os.remove(caps / 'QR.png')
    assert batch.run(_args(install_check=str(caps), report=str(rep))) == 3


# ----------------------------------------------------------------------------- joint solve
def test_joint_solve_ties_cameras_together_despite_misplaced_boards(sample):
    """Boards laid up to 20 mm / 1° off their planned places: each camera's own solve absorbs the error of its boards
    and neighbours disagree at the seams by centimetres; the joint solve brings them back to the noise floor and
    finds where the boards really lie."""
    import copy
    from vantage.core.install import check_all
    from vantage.core.joint import seam_spread
    p, e, _ = sample
    rng = np.random.default_rng(5)
    laid = copy.deepcopy(p.targets)
    for t in laid:
        t.x += rng.uniform(-0.02, 0.02)
        t.y += rng.uniform(-0.02, 0.02)
        t.yaw += rng.uniform(-1, 1)
    cams = p.variant.cameras
    truth, imgs = {}, {}
    for c in cams:
        d = np.r_[rng.uniform(-1, 1, 3), rng.uniform(-0.01, 0.01, 3)]
        truth[c.uid] = _err_pose(c, d)
        imgs[c.uid] = render_capture(p, e, c, 960, 768, pose=truth[c.uid], targets=laid, seed=c.slot)
    res, jr = check_all(p, cams, imgs, InstallSettings(joint=True))
    assert jr is not None and jr.ok and jr.shared_corners > 50
    floor, _, _ = seam_spread(jr.used, truth)                      # detection noise with the true poses
    assert jr.seam_before_mm > 15 and jr.seam_after_mm < max(4.0, 2 * floor * 1000)
    assert all(res[c.uid].joint and res[c.uid].single is not None for c in cams)
    # a board that disagrees with every placement is set aside, not averaged in
    from vantage.core.joint import joint_solve
    obs = {u: list(r.detected) for u, r in res.items()}
    victim = cams[0].uid
    o = obs[victim][0]
    bad = copy.copy(o)
    bad.image_pts = o.image_pts[::-1].copy()                      # a grid matched the wrong way round
    obs[victim] = [bad] + obs[victim][1:]
    init = {u: r.single for u, r in res.items()}
    jr2 = joint_solve(p, cams, obs, init, {u: im.shape for u, im in imgs.items()})
    assert [(t, b) for t, b, _ in jr2.rejected] == [(cams[0].tag, o.name)]
    assert jr2.seam_after_mm < max(4.0, 2 * floor * 1000)


def test_joint_solve_finds_a_misplaced_board(sample):
    """One board 60 mm and 2° off its place, the rest within 3 mm: the joint solve names it."""
    import copy
    from vantage.core.install import check_all
    p, e, _ = sample
    rng = np.random.default_rng(4)
    laid = copy.deepcopy(p.targets)
    for t in laid:
        t.x += rng.uniform(-0.003, 0.003)
        t.y += rng.uniform(-0.003, 0.003)
        if t.name == 'T7':
            a = rng.uniform(0, 2 * np.pi)
            t.x, t.y, t.yaw = t.x + 0.06 * np.cos(a), t.y + 0.06 * np.sin(a), t.yaw + 2.0
    imgs = {}
    for c in p.variant.cameras:
        d = np.r_[rng.uniform(-1, 1, 3), rng.uniform(-0.01, 0.01, 3)]
        imgs[c.uid] = render_capture(p, e, c, 960, 768, pose=_err_pose(c, d), targets=laid, seed=c.slot)
    _, jr = check_all(p, p.variant.cameras, imgs, InstallSettings())
    assert jr.moved and jr.moved[0].name == 'T7' and jr.moved[0].shift_mm > 35
    assert jr.moved[0].dyaw_deg == pytest.approx(2.0, abs=0.6) and 'T7' in jr.message


def test_joint_solve_without_shared_boards(sample):
    from vantage.core.joint import joint_solve
    p, e, _ = sample
    c = p.variant.by_tag('F')
    img = render_capture(p, e, c, 960, 768, seed=1)
    r = check_camera(p, c, img, InstallSettings())
    jr = joint_solve(p, [c], {c.uid: r.boards}, {c.uid: (rot_zyx(r.yaw, r.pitch, r.roll), r.pos)}, {c.uid: img.shape})
    assert jr.shared_corners == 0 and 'not tied together' in jr.message
    assert np.allclose(jr.poses[c.uid][1], r.pos, atol=2e-3)


# ----------------------------------------------------------------------------- photometric stitching
@pytest.fixture(scope='module')
def exposed(sample):
    p, e, res = sample
    rng = np.random.default_rng(2)
    expo = {c.uid: float(rng.uniform(0.7, 1.3)) for c in p.variant.cameras}
    imgs = {c.uid: render_capture(p, e, c, 960, 768, seed=c.slot, exposure=expo[c.uid], vignette=0.3)
            for c in p.variant.cameras}
    return expo, imgs, build_lut(p, e, res, res=0.02)


def test_exposure_gains_balance_the_cameras(exposed):
    from vantage.core.stitch import estimate_gains, overlap_mismatch, warp
    expo, imgs, lut = exposed
    W = warp(lut, imgs)
    g = estimate_gains(lut, warped=W)
    before, after = overlap_mismatch(lut, W), overlap_mismatch(lut, W, g)
    assert before > 10 and after < before / 8
    prod = np.array([expo[u] * g[u][0] for u in lut.uids])
    assert prod.std() / prod.mean() < 0.03                        # gain x exposure is the same for every camera
    assert abs(np.mean([g[u][0] for u in lut.uids]) - 1) < 0.05   # and the overall level is kept
    assert all(np.allclose(v, v[0]) for v in g.values())         # grey images: equal channels


def test_multiband_blend(exposed):
    expo, imgs, lut = exposed
    lin = stitch(lut, imgs, gains='auto')
    mb = stitch(lut, imgs, gains='auto', blend='multiband')
    assert mb.shape == lin.shape == (lut.ny, lut.nx, 3) and mb.dtype == np.uint8
    assert (mb[lut.body] == (38, 38, 36)).all() and (mb[~lut.covered & ~lut.body] == (20, 20, 20)).all()
    free = lut.covered & ~lut.body
    # the same picture, with seams resolved differently
    assert np.abs(mb[free].astype(int) - lin[free].astype(int)).mean() < 6
    # brightness steps across the seams: gradient energy along seam lines drops with multi-band
    gy = np.abs(np.diff(mb.astype(float)[..., 0], axis=0))[free[1:] & free[:-1]].mean()
    gy_lin = np.abs(np.diff(lin.astype(float)[..., 0], axis=0))[free[1:] & free[:-1]].mean()
    assert gy <= gy_lin * 1.05


# ----------------------------------------------------------------------------- reference stitchers
def _frames_dir(lut, imgs, tmp_path, tags):
    import cv2
    d = tmp_path / 'frames'
    d.mkdir()
    for uid, tag in zip(lut.uids, lut.tags):
        cv2.imwrite(str(d / f'{tag}.pgm'), imgs[uid])
    return d


def test_reference_stitcher_cpp_matches(exposed, tmp_path):
    import shutil
    import subprocess
    import cv2
    from vantage.core.stitch import export_lut, runtime_files
    if not shutil.which('g++'):
        pytest.skip('no C++ compiler')
    expo, imgs, lut = exposed
    export_lut(lut, str(tmp_path / 'lut'))
    src = tmp_path / 'bev_stitch.cpp'
    src.write_bytes(dict(runtime_files())['bev_stitch.cpp'])
    exe = tmp_path / 'bev_stitch'
    cc = subprocess.run(['g++', '-O2', '-std=c++17', '-Wall', '-Wextra', '-Werror', str(src), '-o', str(exe)],
                        capture_output=True, text=True)
    assert cc.returncode == 0, cc.stderr
    frames = _frames_dir(lut, imgs, tmp_path, lut.tags)
    for balance in (False, True):
        out = tmp_path / f'out{int(balance)}.ppm'
        r = subprocess.run([str(exe), str(tmp_path / 'lut'), str(out), '--frames', str(frames)] + (['--balance'] if balance else []),
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        cpp = cv2.imread(str(out))[..., ::-1].astype(int)
        py = stitch(lut, imgs, gains='auto' if balance else None).astype(int)
        d = np.abs(cpp - py)
        assert d.mean() < 0.1 and np.percentile(d, 99.9) <= 2, (balance, d.mean(), d.max())
    bad = subprocess.run([str(exe), str(tmp_path / 'lut'), str(tmp_path / 'x.ppm'), str(frames / 'F.pgm')],
                         capture_output=True, text=True)
    assert bad.returncode == 1 and 'frames for' in bad.stderr


def test_reference_stitcher_py_matches(exposed, tmp_path):
    import subprocess
    import sys
    import cv2
    from vantage.core.stitch import export_lut, runtime_files
    expo, imgs, lut = exposed
    export_lut(lut, str(tmp_path / 'lut'))
    st = tmp_path / 'lut' / 'stitcher'
    st.mkdir()
    for n, data in runtime_files():
        (st / n).write_bytes(data)
    frames = _frames_dir(lut, imgs, tmp_path, lut.tags)
    out = tmp_path / 'bev.png'
    r = subprocess.run([sys.executable, str(st / 'bev_stitch.py'), '--frames', str(frames), '--out', str(out), '--balance'],
                       capture_output=True, text=True, cwd=str(tmp_path))
    assert r.returncode == 0, r.stderr
    got = cv2.imread(str(out))[..., ::-1].astype(int)
    py = stitch(lut, imgs, gains='auto').astype(int)
    assert np.abs(got - py).mean() < 0.1
    assert len([ln for ln in r.stdout.splitlines() if ln.startswith('gain ')]) == len(lut.uids)


# ----------------------------------------------------------------------------- batch: lens calibration
def test_batch_calibrate_lens_and_import(sample, tmp_path):
    from vantage import batch
    from vantage.core import calib
    from vantage.core.project import Project
    p, e, _ = sample
    f = p.variant.by_tag('F')
    m, lens = p.module_for(f), p.nominal_lens_for(f)
    truth = calib.simulated_unit(m, lens, seed=21)
    views = tmp_path / 'views'
    views.mkdir()
    for name, img in calib.board_views(m, lens, truth, (9, 6), 60.0, n=14, seed=2):
        capture.save_image(str(views / f'{name}.png'), img)
    rep, yml, saved = tmp_path / 'cal.json', tmp_path / 'F.yaml', tmp_path / 'cal.vantage'
    code = batch.run(_args(calibrate_lens=str(views), camera='F', board='9x6', square=60.0, serial='SN-F',
                           calib_out=str(yml), save=str(saved), report=str(rep)))
    doc = json.loads(rep.read_text('utf-8'))
    assert code == 0 and doc['ok'] and doc['camera'] == 'F', doc['message']
    assert doc['calibration']['serial'] == 'SN-F' and doc['calibration']['rms_px'] < 0.2
    assert doc['calibration']['fx'] * 2 == pytest.approx(truth['fx'], rel=3e-3)
    q = Project.load(str(saved))
    assert q.variant.by_tag('F').calib['serial'] == 'SN-F' and not q.variant.by_tag('R').calib
    # the written YAML imports onto F (its camera_name) in a fresh copy of the sample
    saved2 = tmp_path / 'imp.vantage'
    code = batch.run(_args(import_calib=[str(yml)], save=str(saved2), report=str(tmp_path / 'imp.json')))
    idoc = json.loads((tmp_path / 'imp.json').read_text('utf-8'))
    assert code == 0 and [r['camera'] for r in idoc['applied']] == ['F'] and not idoc['unmatched']
    q2 = Project.load(str(saved2))
    a, b = q.variant.by_tag('F').calib, q2.variant.by_tag('F').calib
    assert all(a[k] == pytest.approx(b[k], rel=1e-12) for k in ('fx', 'fy', 'cx', 'cy')) and a['k'] == pytest.approx(b['k'])
    # nothing usable: exit code 4
    assert batch.run(_args(calibrate_lens=str(tmp_path), camera='F', report=str(tmp_path / 'none.json'))) == 4
    with pytest.raises(SystemExit):
        batch.run(_args(calibrate_lens=str(views)))                # no --camera
