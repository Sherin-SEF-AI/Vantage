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
    # a camera with no image fails the check
    os.remove(caps / 'QR.png')
    assert batch.run(_args(install_check=str(caps), report=str(rep))) == 3
