"""Regression tests for bugs found in the core engine, file I/O and exports."""
import json
import math
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from vantage.core import samples, photos as ph
from vantage.core import exporters as ex
from vantage.core.analysis import Engine
from vantage.core.project import PhotoSet, PhotoFrame


@pytest.fixture(scope='module')
def sample():
    p = samples.tigor_sample_project()
    e = Engine()
    return p, e, e.analyse(p)


def test_remap_tables_with_duplicate_tags(sample, tmp_path):
    """Two cameras whose tags clean up to the same name must not overwrite each other's tables."""
    import copy
    p0, e, res = sample
    p = copy.deepcopy(p0)
    p.variant.cameras[2].tag = 'F'                # same as cameras[0]
    p.variant.cameras[1].tag = 'Rück'             # cleans up to R_ck
    paths = ex.export(['bev_lut'], p, res, ex.ExportOptions(out_dir=str(tmp_path), engine=e))
    names = [x.replace(str(tmp_path) + '/', '') for x in paths]
    assert len(names) == len(set(names))
    assert set(names) == set(ex.format_by_id('bev_lut').filename(p, p.variant))
    meta = json.loads((tmp_path / 'bev_lut' / 'bev_lut.json').read_text('utf-8'))
    maps = [c['map'] for c in meta['cameras']]
    assert len(maps) == len(set(maps)) and 'F_2_map.bin' in maps and 'R_ck_map.bin' in maps
    w, h = meta['width'], meta['height']
    tot = sum(np.fromfile(tmp_path / 'bev_lut' / c['weight'], np.uint8).reshape(h, w).astype(int) for c in meta['cameras'])
    assert tot.max() <= 255 + len(maps)


def _photo_set(p, npts):
    ref = p.photos
    lm = {k: list(map(float, v)) for k, v in p.vehicle.landmarks.items()}
    names = list(lm)
    truth = PhotoSet(width=ref.width, height=ref.height, f_px=ref.f_px, cx=ref.width / 2, cy=ref.height / 2, landmarks=lm)
    ps = PhotoSet(width=ref.width, height=ref.height, f_px=ref.f_px * 1.1, cx=ref.width / 2, cy=ref.height / 2, landmarks=lm)
    for k, n in enumerate(npts):
        truth.frames.append(PhotoFrame(f'f{k}', ref.frames[k].pose, []))
        uv = ph.project_points(truth, k, [lm[nm] for nm in names])
        pts = [(nm, float(a), float(b)) for nm, (a, b) in zip(names, uv)
               if np.isfinite(a) and 0 <= a < ref.width and 0 <= b < ref.height][:n]
        ps.frames.append(PhotoFrame(f'f{k}', None, pts))
    return ps


@pytest.mark.parametrize('npts', [[8], [8, 5]])
def test_photo_frame_with_many_landmarks_is_solved_alone(sample, npts):
    """A frame with 6+ landmarks used to be left out of PnP when there was no second such frame for the joint fit."""
    p = sample[0]
    if not p.photos.frames or len(p.vehicle.landmarks) < 8:
        pytest.skip('sample has no photo poses')
    ps = _photo_set(p, npts)
    ph.calibrate(ps)
    assert all(f.pose is not None for f in ps.frames)


def test_ros_and_urdf_names(sample):
    import copy
    p0, _, res = sample
    p = copy.deepcopy(p0)
    p.name = 'rig---A--'
    p.variant.cameras[0].tag = 'F-L'
    files = dict(ex._render('ros2', ex._context(p, res, ex.ExportOptions())))
    ET.fromstring(files['vantage_cameras.urdf'])                 # well-formed despite '---' in names
    launch = files['static_tf.launch.py'].decode()
    assert "'tf_f_l'" in launch and "'tf_f_l_optical'" in launch and 'F-L_link' in launch
    compile(launch, 'static_tf.launch.py', 'exec')


def test_machine_readable_angles_keep_precision(sample):
    import copy
    p0, _, res = sample
    p = copy.deepcopy(p0)
    c = p.variant.cameras[0]
    c.yaw, c.pitch, c.roll = 65.2371, 44.9623, -0.4481
    for angles, scale, tol in (('deg', 1.0, 1e-4), ('rad', math.pi / 180, 1e-6)):
        ctx = ex._context(p, res, ex.ExportOptions(angles=angles))
        doc = json.loads(ex._render('extrinsics_json', ctx)[0][1])
        jc = doc['cameras'][0]
        assert (jc['yaw'], jc['pitch'], jc['roll']) == pytest.approx((65.2371 * scale, 44.9623 * scale, -0.4481 * scale),
                                                                     abs=tol)
        row = ex._render('extrinsics_csv', ctx)[0][1].decode().splitlines()[1]
        assert f'{65.2371 * scale:.4f}' in row or f'{65.2371 * scale:.6f}' in row


# ----------------------------------------------------------------------------- calibration files and calibration
def test_pinhole_calibration_is_not_taken_for_fisheye(tmp_path):
    from vantage.core import calib
    K = '[ 900., 0., 960., 0., 900., 768., 0., 0., 1. ]'
    base = ('%YAML:1.0\n---\nimage_width: 1920\nimage_height: 1536\ncamera_matrix: !!opencv-matrix\n   rows: 3\n   cols: 3\n'
            f'   dt: d\n   data: {K}\n')
    pin = tmp_path / 'pinhole.yaml'
    pin.write_text(base + 'distortion_coefficients: !!opencv-matrix\n   rows: 1\n   cols: 5\n   dt: d\n'
                   '   data: [ -0.28, 0.07, 0.0009, -0.0004, 0.0 ]\n')
    with pytest.raises(ValueError, match='pinhole'):
        calib.read_file(str(pin))
    flag = tmp_path / 'flag.yaml'
    four = 'distortion_coefficients: !!opencv-matrix\n   rows: 1\n   cols: 4\n   dt: d\n   data: [ 0.01, 0.1, -0.2, 0.05 ]\n'
    flag.write_text(base + four + 'fisheye_model: 0\n')
    with pytest.raises(ValueError, match='not fisheye'):
        calib.read_file(str(flag))
    flag.write_text(base + four + 'fisheye_model: 1\n')
    assert calib.read_file(str(flag))[0]['k'] == pytest.approx([0.01, 0.1, -0.2, 0.05])


def test_calibration_refuses_images_of_another_aspect():
    from vantage.core import calib
    from vantage.core.lens import builtin_lenses, builtin_modules
    m, lens = builtin_modules()[0], builtin_lenses()[0]
    views = calib.board_views(m, lens, None, n=5, seed=3)
    cropped = [(n, img[:540, :960]) for n, img in views]
    r = calib.calibrate(cropped, (9, 6), 60.0, m, lens)
    assert not r.ok and r.record is None and 'aspect' in r.message
    rec = calib.make_record(960, 540, 430, 430, 480, 270, [0, 0, 0, 0])
    assert calib.fatal_problems(rec, m)


def test_match_tags_made_of_several_words():
    from types import SimpleNamespace as NS
    from vantage.core import calib
    cams = [NS(uid=t, tag=t, calib={}) for t in ('FL', 'FL_1', 'front', 'front-left')]
    base = calib.make_record(1920, 1536, 870, 870, 959.5, 767.5, [0, 0, 0, 0])
    recs = [dict(base, hint='calib_FL_1', stem='calib_FL_1'), dict(base, hint='/cam/front-left/image_raw', stem='chain')]
    out, rest = calib.match(recs, cams)
    assert out['FL_1'] is recs[0] and out['front-left'] is recs[1] and not rest


# ----------------------------------------------------------------------------- installation check
def test_rotation_verdict_for_a_camera_looking_straight_down(sample):
    import copy
    from vantage.core.install import PoseResult, InstallSettings, _grade
    from vantage.core.mathutil import rot_zyx, rot_x
    p0 = sample[0]
    cam = copy.deepcopy(p0.variant.cameras[0])
    cam.yaw, cam.pitch, cam.roll = 30.0, 90.0, 5.0
    R = rot_zyx(cam.yaw, cam.pitch, cam.roll) @ rot_x(math.radians(0.1))      # a 0.1° twist about the optical axis
    r = PoseResult(cam.uid, cam.tag)
    _grade(r, cam, R, cam.pos(), 0.1, InstallSettings())
    assert r.rot_err_deg == pytest.approx(0.1, abs=1e-6) and r.status == 'good'


def test_board_corners_have_no_half_pixel_bias(sample):
    from vantage.core.install import detect_boards, board_rms
    from vantage.core.mathutil import rot_zyx
    from vantage.core.synth import render_capture
    p, e, _ = sample
    c = p.variant.by_tag('R')
    img = render_capture(p, e, c, noise=0)                      # full resolution, at the planned pose
    obs = detect_boards(img, p, c)
    assert obs
    pose = (rot_zyx(c.yaw, c.pitch, c.roll), c.pos())
    assert max(board_rms(p, c, obs, img.shape, pose)) < 0.25


def test_joint_solve_resolves_after_rejecting(sample):
    import copy
    from vantage.core.install import check_all, InstallSettings
    from vantage.core.joint import joint_solve
    from vantage.core.synth import render_capture
    p, e, _ = sample
    cams = p.variant.cameras
    imgs = {c.uid: render_capture(p, e, c, 960, 768, seed=c.slot) for c in cams}
    res, _ = check_all(p, cams, imgs, InstallSettings(joint=False))
    obs = {u: list(r.detected) for u, r in res.items()}
    from vantage.core.mathutil import rot_zyx
    init = {u: (rot_zyx(r.yaw, r.pitch, r.roll), r.pos) for u, r in res.items()}
    shapes = {u: im.shape for u, im in imgs.items()}
    bad = {u: list(v) for u, v in obs.items()}
    without = {u: list(v) for u, v in obs.items()}
    flipped = []
    for c in cams[:3]:
        if len(bad[c.uid]) > 1:
            o = copy.copy(bad[c.uid][0])
            o.image_pts = o.image_pts[::-1].copy()      # a grid matched the wrong way round
            bad[c.uid][0] = o
            without[c.uid] = without[c.uid][1:]
            flipped.append(c.tag)
    jr = joint_solve(p, cams, bad, init, shapes)
    assert set(flipped) <= {t for t, _, _ in jr.rejected}
    # the reported poses are a solve without the rejected observations, not one still pulled by them
    ref = joint_solve(p, cams, without, init, shapes)
    for c in cams:
        assert np.linalg.norm(jr.poses[c.uid][1] - ref.poses[c.uid][1]) < 5e-4, c.tag


# ----------------------------------------------------------------------------- stitching, captures, live sources
def test_stitch_with_a_camera_missing(sample):
    from vantage.core.stitch import build_lut, stitch
    from vantage.core.synth import render_capture
    p, e, res = sample
    lut = build_lut(p, e, res, res=0.04)
    imgs = {c.uid: render_capture(p, e, c, 480, 384, seed=c.slot, noise=0) for c in p.variant.cameras}
    full = stitch(lut, imgs)
    part = dict(imgs)
    gone = p.variant.cameras[0].uid
    part.pop(gone)
    for blend in ('linear', 'multiband'):
        out = stitch(lut, part, blend=blend)
        others = sum(lut.weight[u] for u in part) > 0
        free = others & ~lut.body
        assert not (out[free] == 0).all(axis=1).any(), blend          # no black holes where another camera sees
        assert abs(float(out[free].mean()) - float(full[free].mean())) < 6, blend


def test_pixel_rays_cache_keeps_pixel_size_apart():
    import dataclasses
    from vantage.core.lens import builtin_lenses, builtin_modules, resolve
    from vantage.core.synth import pixel_rays
    m, lens = builtin_modules()[0], builtin_lenses()[0]
    a = resolve(m, lens)
    b = resolve(dataclasses.replace(m, pixel_um=2.0), lens)
    _, ta = pixel_rays(a, 64, 51)
    _, tb = pixel_rays(b, 64, 51)
    assert not np.allclose(ta, tb)


def test_sixteen_bit_images_keep_their_black_level(tmp_path):
    import cv2
    from vantage.core import capture
    img = np.zeros((4, 4, 3), np.uint16)
    img[:] = (1000, 20000, 60000)                  # BGR on disk
    path = tmp_path / 'x.png'
    cv2.imwrite(str(path), img)
    out = capture.load_image(str(path))
    assert out.dtype == np.uint8 and tuple(out[0, 0]) == (round(60000 * 255 / 65535), round(20000 * 255 / 65535),
                                                          round(1000 * 255 / 65535))
    flat = np.full((4, 4), 30000, np.uint16)
    cv2.imwrite(str(path), flat)
    assert capture.load_image(str(path))[0, 0] == round(30000 * 255 / 65535)


def test_live_source_restart_while_a_read_is_stuck(monkeypatch):
    import threading
    import time
    from vantage.core import capture

    class Slow:
        def isOpened(self):
            return True

        def read(self):
            time.sleep(2.2)
            return True, np.zeros((4, 4, 3), np.uint8)

        def release(self):
            pass
    monkeypatch.setattr(capture, '_open', lambda spec: Slow())
    ls = capture.LiveSource('fake')
    ls.start()
    time.sleep(0.2)
    ls.stop()                                       # gives up after 1.5 s while read() is still stuck
    ls.start()
    time.sleep(3.0)
    readers = [t for t in threading.enumerate() if t.name == 'live:fake']
    assert len(readers) == 1
    ls.stop()


# ----------------------------------------------------------------------------- core engine
def test_number_fields_do_arithmetic_in_the_display_unit():
    from vantage.core.mathutil import parse_number
    assert parse_number('1200/2', 'length', 'mm') == pytest.approx(600)
    assert parse_number('2*3', 'length', 'mm') == pytest.approx(6)
    assert parse_number('1200mm/2', 'length', 'mm') == pytest.approx(600)
    assert parse_number('1 m / 4', 'length', 'mm') == pytest.approx(250)
    assert parse_number('950 mm', 'length', 'm') == pytest.approx(0.95)
    assert parse_number('1.88+0.05', 'length', 'm') == pytest.approx(1.93)
    assert parse_number('2 cm', 'length', 'mm') == pytest.approx(20)
    assert parse_number('1 rad', 'angle') == pytest.approx(57.29578, abs=1e-4)


def test_mesh_caches_follow_mask_contents(sample):
    p = sample[0]
    m = p.vehicle.mesh
    m._cache.clear()
    for _ in range(5):
        m.crease_segments(35, m.occluder_faces())
        m.raycast(np.array([[1.0, 0, 3]]), np.array([[0, 0, -1.0]]), m.occluder_faces())
    assert len(m._cache) <= 6
    mask = m.occluder_faces().copy()
    a = m.crease_segments(35, mask)
    mask2 = mask.copy()
    mask2[np.nonzero(mask2)[0][:200]] = False
    b = m.crease_segments(35, mask2)
    assert len(a) != len(b)


def test_target_suggestions_with_no_room(sample):
    import copy
    from vantage.core import targets
    p0, e, _ = sample
    p = copy.deepcopy(p0)
    p.roi.distance = 0.3
    eng = Engine()
    res = eng.analyse(p)
    assert targets.auto_place(eng, p, res) == []


def test_short_lens_image_circle_stays_in_front():
    import dataclasses
    from vantage.core.lens import builtin_lenses, builtin_modules, resolve
    m = builtin_modules()[0]
    lens = dataclasses.replace(builtin_lenses()[0], projection='equisolid', focal_mm=1.8, sheet_fov=(0, 0, 0), max_half_deg=0)
    intr = resolve(m, lens)
    assert math.degrees(intr.max_theta) <= 95.0
    u, v, th, ok = intr.project_dirs(np.array([[0.0, 0.0, -1.0]]))
    assert not ok[0]


def test_nested_matrix_lists_in_opencv_yaml(tmp_path):
    from vantage.core import calib
    f = tmp_path / 'nested.yaml'
    f.write_text('%YAML:1.0\n---\nimage_width: 1920\nimage_height: 1536\ndistortion_model: fisheye\n'
                 'camera_matrix: [[870.0, 0.0, 959.5], [0.0, 871.0, 767.5], [0.0, 0.0, 1.0]]\n'
                 'distortion_coefficients: [0.01, 0.1, -0.2, 0.05]\n')
    r = calib.read_file(str(f))[0]
    assert (r['fx'], r['fy'], r['cx'], r['cy']) == (870.0, 871.0, 959.5, 767.5)


def test_dxf_is_cp1252_with_its_codepage(sample, tmp_path):
    import copy
    p0, _, res = sample
    p = copy.deepcopy(p0)
    p.variant.cameras[0].tag = 'Fü'
    data = dict(ex._render('dxf', ex._context(p, res, ex.ExportOptions())))['mount-points.dxf']
    text = data.decode('cp1252')
    assert '$DWGCODEPAGE' in text and 'ANSI_1252' in text and 'Fü' in text
    with pytest.raises(UnicodeDecodeError):
        data.decode('utf-8')                                     # really 8-bit, not UTF-8


def test_target_candidates_are_symmetric(sample):
    from vantage.core import targets
    p, e, res = sample
    placed = targets.auto_place(e, p, res)
    assert len(placed) == len(targets.adjacent_pairs(res))       # every adjacent pair gets a board
