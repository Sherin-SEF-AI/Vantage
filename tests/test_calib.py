"""Unit lens calibration: the lens model with fy != fx, calibration from board views, calibration files, matching
to cameras, persistence and the effect on the installation check."""
import copy
import json
import math

import numpy as np
import pytest

from vantage.core import calib, samples
from vantage.core import exporters as ex
from vantage.core.analysis import Engine
from vantage.core.install import InstallSettings, check_camera
from vantage.core.lens import builtin_lenses, builtin_modules, resolve
from vantage.core.mathutil import rot_zyx
from vantage.core.project import Project
from vantage.core.synth import render_capture

MOD = builtin_modules()[0]
WIDE = builtin_lenses()[0]


@pytest.fixture(scope='module')
def sample():
    p = samples.tigor_sample_project()
    return p, Engine()


@pytest.fixture(scope='module')
def unit():
    """A simulated unit and its calibration from 16 synthetic board views at half resolution."""
    truth = calib.simulated_unit(MOD, WIDE, seed=3)
    views = calib.board_views(MOD, WIDE, truth, (9, 6), 60.0, n=16, seed=1)
    return truth, views, calib.calibrate(views, (9, 6), 60.0, MOD, WIDE)


def test_library_lens_as_record_is_the_same_model():
    rec = calib.nominal_record(MOD, WIDE)
    a, b = resolve(MOD, WIDE), resolve(MOD, calib.unit_lens(WIDE, MOD, rec))
    assert rec['cx'] == pytest.approx(959.5) and rec['cy'] == pytest.approx(767.5)     # OpenCV pixel centres
    d = calib.model_difference(a, b)
    assert d['max_px'] < 1e-6 and d['centre_px'] < 1e-9


def test_aspect_projects_and_unprojects_consistently():
    rec = dict(calib.nominal_record(MOD, WIDE))
    rec['fy'] = rec['fx'] * 1.004
    intr = resolve(MOD, calib.unit_lens(WIDE, MOD, rec))
    assert intr.aspect == pytest.approx(1.004)
    u = np.array([100.0, 960.0, 1500.0, 1800.0])
    v = np.array([200.0, 768.0, 1400.0, 100.0])
    d, th = intr.unproject(u, v)
    u2, v2, th2, ok = intr.project_dirs(d)
    assert np.allclose(u2, u, atol=1e-6) and np.allclose(v2, v, atol=1e-6) and np.allclose(th, th2)


def test_unit_lens_scales_a_lower_resolution_calibration():
    rec = calib.nominal_record(MOD, WIDE)
    half = calib.make_record(960, 768, rec['fx'] / 2, rec['fy'] / 2, (rec['cx'] + 0.5) / 2 - 0.5, (rec['cy'] + 0.5) / 2 - 0.5,
                             rec['k'])
    a, b = resolve(MOD, WIDE), resolve(MOD, calib.unit_lens(WIDE, MOD, half))
    assert calib.model_difference(a, b)['max_px'] < 1e-6
    assert not calib.problems(half, MOD, WIDE)
    crop = dict(half, width=960, height=600)
    assert any('aspect' in m for m in calib.problems(crop, MOD, WIDE))
    other = dict(rec, fx=rec['fx'] * 1.4, fy=rec['fy'] * 1.4)
    assert any('focal length' in m for m in calib.problems(other, MOD, WIDE))


def test_calibration_recovers_the_unit(unit):
    truth, views, r = unit
    assert r.ok, r.message
    assert all(v.used for v in r.views) and r.rms < 0.2
    rec = r.record
    assert (rec['width'], rec['height']) == (960, 768)
    s = MOD.width_px / rec['width']
    assert rec['fx'] * s == pytest.approx(truth['fx'], rel=3e-3)
    assert abs((rec['cx'] + 0.5) * s - 0.5 - truth['cx']) < 1.5 and abs((rec['cy'] + 0.5) * s - 0.5 - truth['cy']) < 1.5
    fit = resolve(MOD, calib.unit_lens(WIDE, MOD, rec))
    true = resolve(MOD, calib.unit_lens(WIDE, MOD, truth))
    field = math.radians(65)
    err = calib.model_difference(fit, true, max_theta=field)
    nominal = calib.model_difference(resolve(MOD, WIDE), true, max_theta=field)
    assert err['rms_px'] < 1.0 and nominal['rms_px'] > 5 * err['rms_px']
    assert r.coverage is not None and r.coverage.sum() == sum(len(v.corners) for v in r.views)
    rows = calib.compare_rows(MOD, WIDE, rec)
    assert [x[0] for x in rows[:4]] == ['fx px', 'fy px', 'cx px', 'cy px'] and len(rows) == 8


def test_calibration_needs_enough_views(unit):
    _, views, _ = unit
    r = calib.calibrate(views[:3], (9, 6), 60.0, MOD, WIDE)
    assert not r.ok and r.record is None and 'at least 4' in r.message
    blank = [(f'b{i}', np.full((768, 960), 120, np.uint8)) for i in range(5)]
    r = calib.calibrate(blank, (9, 6), 60.0, MOD, WIDE)
    assert not r.ok and all(v.note == 'board not found' for v in r.views)


def test_find_board_in_rectified_views():
    """A board near the image edge, too distorted for the raw-image detector, is found through the lens model."""
    intr = resolve(MOD, WIDE)
    from vantage.core.calib import _random_poses, render_board, board_points
    rng = np.random.default_rng(11)
    poses = _random_poses(intr, (9, 6), 60.0, 40, rng, 960, 768)
    edge = max(poses, key=lambda P: np.linalg.norm(P[1][:2]) / P[1][2])      # the most off-axis board
    img = render_board(intr, edge[0], edge[1], (9, 6), 60.0, 960, 768)
    c = calib.find_board(img, (9, 6), intr)
    assert c is not None and c.shape == (54, 2)
    obj = board_points((9, 6), 60.0) - np.array([4 * 0.06, 2.5 * 0.06, 0])
    P = obj @ edge[0].T + edge[1]
    u, v, _, _ = intr.project_dirs(P / np.linalg.norm(P, axis=1, keepdims=True))
    truth = np.stack([u / 2 - 0.5, v / 2 - 0.5], -1)
    # the detector's corner order is a rotation of the board's; compare as point sets
    d = np.linalg.norm(c[:, None] - truth[None], axis=-1).min(1)
    assert d.max() < 0.5


# ----------------------------------------------------------------------------- files
def _write(path, text):
    path.write_text(text, encoding='utf-8')
    return str(path)


def test_read_calibration_files(tmp_path, unit):
    _, _, r = unit
    rec = r.record
    K = [[rec['fx'], 0, rec['cx']], [0, rec['fy'], rec['cy']], [0, 0, 1]]
    files = {
        'opencv': _write(tmp_path / 'F.yaml', calib.to_opencv_yaml(dict(rec, serial='SN-7'), 'F')),
        'ros': _write(tmp_path / 'ros.yaml', 'image_width: 960\nimage_height: 768\ncamera_name: ML\n'
                      f'camera_matrix:\n  rows: 3\n  cols: 3\n  data: {[x for row in K for x in row]}\n'
                      f'distortion_model: equidistant\ndistortion_coefficients:\n  rows: 1\n  cols: 4\n  data: {rec["k"]}\n'),
        'kalibr': _write(tmp_path / 'camchain.yaml', ''.join(
            f'cam{i}:\n  camera_model: pinhole\n  distortion_model: equidistant\n  distortion_coeffs: {rec["k"]}\n'
            f'  intrinsics: [{rec["fx"]}, {rec["fy"]}, {rec["cx"]}, {rec["cy"]}]\n  resolution: [960, 768]\n'
            f'  rostopic: /{t}/image_raw\n' for i, t in enumerate(('QL', 'QR')))),
        'json': _write(tmp_path / 'generic.json', json.dumps({'K': K, 'D': rec['k'], 'width': 960, 'height': 768, 'tag': 'R'})),
    }
    got = {k: calib.read_file(v) for k, v in files.items()}
    assert [len(v) for v in got.values()] == [1, 1, 2, 1]
    for recs in got.values():
        for x in recs:
            for k in ('fx', 'fy', 'cx', 'cy'):
                assert x[k] == pytest.approx(rec[k], abs=1e-9)
            assert x['k'] == pytest.approx(rec['k'], abs=1e-12) and (x['width'], x['height']) == (960, 768)
    assert got['opencv'][0]['serial'] == 'SN-7' and got['opencv'][0]['hint'] == 'F'
    assert [x['hint'] for x in got['kalibr']] == ['/QL/image_raw', '/QR/image_raw']
    bad = _write(tmp_path / 'pinhole.yaml', 'image_width: 960\nimage_height: 768\ncamera_matrix: {rows: 3, cols: 3, data: '
                 '[500, 0, 480, 0, 500, 384, 0, 0, 1]}\ndistortion_model: plumb_bob\n'
                 'distortion_coefficients: {rows: 1, cols: 5, data: [0.1, 0, 0, 0, 0]}\n')
    with pytest.raises(ValueError, match='not fisheye'):
        calib.read_file(bad)


def test_match_records_to_cameras(sample):
    p, _ = sample
    cams = p.variant.cameras
    base = calib.nominal_record(MOD, WIDE)
    recs = [dict(base, hint='/QR/image_raw', stem='camchain'), dict(base, hint='', stem='cam_ML_2026'),
            dict(base, hint='F', stem='x'), dict(base, hint='spare', stem='spare')]
    out, unmatched = calib.match(recs, cams)
    tags = {p.variant.cam(u).tag: r for u, r in out.items()}
    assert set(tags) == {'QR', 'ML', 'F'} and tags['F'] is recs[2]
    assert unmatched == [recs[3]]
    # a known serial wins over a weak name match
    q = copy.deepcopy(cams)
    q[1].calib = dict(base, serial='SN-R')
    out, _ = calib.match([dict(base, hint='cam', stem='cam', serial='SN-R')], q)
    assert list(out) == [q[1].uid]


def test_unit_calibration_is_saved_and_exported(sample, tmp_path, unit):
    p0, _ = sample
    p = copy.deepcopy(p0)
    _, _, r = unit
    f = p.variant.by_tag('F')
    f.calib = calib.clean_record(dict(r.record, serial='SN-F1'))
    path = tmp_path / 'p.vantage'
    p.save(str(path))
    q = Project.load(str(path))
    qf = q.variant.by_tag('F')
    assert qf.calib == f.calib and q.variant.by_tag('R').calib == {}
    lens = q.lens_for(qf)
    assert lens.label == 'unit' and lens.projection == 'kannala_brandt' and 'SN-F1' in lens.name
    assert q.nominal_lens_for(qf).id == qf.lens_id
    # exports carry the unit's own intrinsics, in OpenCV pixel coordinates at sensor resolution
    cm = ex.camera_model(q, qf)
    s = MOD.width_px / r.record['width']
    assert cm.fx == pytest.approx(r.record['fx'] * s) and cm.cx == pytest.approx((r.record['cx'] + 0.5) * s - 0.5)
    assert cm.k == pytest.approx(tuple(r.record['k']))
    doc = json.loads(ex._render('extrinsics_json', ex._context(q, None, ex.ExportOptions()))[0][1])
    jf = next(c for c in doc['cameras'] if c['tag'] == 'F')['intrinsics']
    assert jf['source'] == 'unit_calibration' and jf['serial'] == 'SN-F1'
    assert next(c for c in doc['cameras'] if c['tag'] == 'R')['intrinsics']['source'] == 'library_lens'
    yaml_text = ex._render('opencv_yaml', ex._context(q, None, ex.ExportOptions()))[0][1].decode()
    assert "this unit's own calibration (SN-F1" in yaml_text
    # the exported file reads back as the same calibration (at sensor resolution)
    back = calib.read_file(_write(tmp_path / 'F.yaml', yaml_text))[0]
    b = resolve(MOD, calib.unit_lens(WIDE, MOD, back))
    assert calib.model_difference(b, resolve(MOD, lens))['max_px'] < 1e-3
    # Vantage JSON round trip
    js = _write(tmp_path / 'units.json', calib.to_json(q, q.variant.cameras))
    recs = calib.read_file(js)
    assert len(recs) == 1 and recs[0]['hint'] == 'F' and recs[0]['serial'] == 'SN-F1'


def test_installation_check_needs_the_units_own_lens(sample):
    """A capture taken through a unit whose lens differs from the datasheet: checked with the library lens the pose
    absorbs the lens error; with the unit calibration it comes out right."""
    p0, e = sample
    p = copy.deepcopy(p0)
    cam = p.variant.by_tag('ML')
    truth = calib.simulated_unit(MOD, WIDE, seed=8, f_pct=2.0, pp_px=20.0)
    cam.calib = truth
    d = np.array([0.6, -0.5, 0.4, 0.006, -0.005, 0.004])
    pose = (rot_zyx(cam.yaw + d[0], cam.pitch + d[1], cam.roll + d[2]), cam.pos() + d[3:])
    img = render_capture(p, e, cam, 960, 768, pose=pose, seed=4)
    with_unit = check_camera(p, cam, img, InstallSettings())
    cam.calib = {}
    with_library = check_camera(p, cam, img, InstallSettings())
    err_u = np.abs(with_unit.dang_deg - d[:3]).max()
    err_l = np.abs(with_library.dang_deg - d[:3]).max()
    assert with_unit.pos is not None and err_u < 0.15, err_u
    assert with_library.pos is None or err_l > 2 * err_u
    assert with_unit.rms_px < with_library.rms_px or with_library.pos is None
