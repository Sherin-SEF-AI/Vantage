"""Regression tests for bugs found in the core engine, file I/O and exports."""
import copy
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


def test_obstacles_under_a_load_case_are_seen_from_the_moved_body(sample):
    """With the body moved (a load case), obstacle points must go into the body frame like the ground grid does."""
    import copy
    from vantage.core.project import Obstacle
    p0, e, res = sample
    p = copy.deepcopy(p0)
    ob = Obstacle(name='Pole', x=4.5, y=0.8, height=1.0)
    p.obstacles = [ob]
    t = np.array([0.6, 0.0, 0.0])                       # the body 0.6 m further forward in the world
    moved = e.analyse(p, body_pose=(np.eye(3), t))
    q = copy.deepcopy(p)
    q.obstacles[0].x -= 0.6                              # the same pole, relative to the body
    cams = [c for c in q.variant.cameras if c.visible]
    expect = e.obstacle_visibility(q, cams, res.cubes)
    still = e.obstacle_visibility(p, cams, res.cubes)
    assert expect[ob.uid] and expect[ob.uid] != still[ob.uid]
    got = moved.obstacles[ob.uid]
    assert got.keys() == expect[ob.uid].keys()
    assert all(np.allclose(got[k], expect[ob.uid][k]) for k in got)


@pytest.mark.parametrize('projection', ['equidistant', 'equisolid', 'rectilinear'])
@pytest.mark.parametrize('aspect', [1.0, 1.21])
def test_density_counts_the_pixels_a_patch_covers(projection, aspect):
    """Density is sqrt(pixels per cm^2) on a patch facing the camera, including a calibrated fy/fx aspect."""
    from vantage.core.lens import Intrinsics
    intr = Intrinsics(width=1920, height=1280, pixel_mm=0.003, f_mm=1.3 if projection != 'rectilinear' else 4.0,
                      projection=projection, kb=(0, 0, 0, 0), cx=960, cy=640, max_theta=math.radians(95),
                      focus_near=0.1, focus_far=100.0, aspect=aspect)
    th, ph, rng, h = math.radians(25), math.radians(40), 2.0, 1e-4     # a 0.1 mm square patch at 2 m
    d = np.array([math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th)])
    e1 = np.cross(d, [0.0, 0.0, 1.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(d, e1)
    P = rng * d + h * np.array([[0, 0], [1, 0], [0, 1]]) @ np.stack([e1, e2])
    u, v, _, ok = intr.project_dirs(P / np.linalg.norm(P, axis=1)[:, None])
    assert ok.all()
    px_area = abs((u[1] - u[0]) * (v[2] - v[0]) - (u[2] - u[0]) * (v[1] - v[0]))
    expect = math.sqrt(px_area / (h * 100) ** 2)
    assert float(intr.density_px_per_cm(th, rng)) == pytest.approx(expect, rel=2e-3)


def test_ui_thread_on_another_car_does_not_disturb_a_running_analysis(sample):
    """The UI asks the shared engine about the live project while the worker analyses an older snapshot (e.g. just
    after a model import): each thread must keep its own car, and neither may rebuild the other's geometry."""
    import threading
    import time
    from vantage.core import analysis
    p0, _, clean = sample
    other = samples.template_project('suv')
    e = Engine()
    e.vehicle_geom(p0)
    builds = analysis._vg_counter[0]
    stop = threading.Event()

    def ui():
        n = 0
        while not stop.is_set():
            e.vehicle_geom(other if n % 2 else p0)
            n += 1
            time.sleep(0.002)
    t = threading.Thread(target=ui)
    t.start()
    try:
        res = e.analyse(p0.snapshot())
    finally:
        stop.set()
        t.join()
    assert res.metrics['blind_m2'] == pytest.approx(clean.metrics['blind_m2'], abs=1e-9)
    assert res.metrics['covered_pct'] == pytest.approx(clean.metrics['covered_pct'], abs=1e-9)
    assert analysis._vg_counter[0] - builds <= 1                    # at most the other car, once
    assert e.vehicle_geom(p0) is e.vg and e.vg.mesh is p0.vehicle.mesh


def test_shared_engine_reuses_the_geometry_on_its_worker_thread(sample):
    import threading
    from vantage.core import analysis
    p0, e, _ = sample
    vg = e.vehicle_geom(p0)
    w = e.share()
    builds = analysis._vg_counter[0]
    seen = []
    t = threading.Thread(target=lambda: seen.append((w.vg, w.vehicle_geom(p0))))
    t.start()
    t.join()
    assert seen == [(vg, vg)] and analysis._vg_counter[0] == builds
    assert w.cubes is e.cubes


def test_snapshot_keeps_its_own_vehicle_fields(sample):
    p0, _, _ = sample
    import copy
    p = copy.deepcopy(p0)
    q = p.snapshot()
    p.vehicle.occluder_target = 1234
    p.vehicle.wheel_centres['FL'] = [9.0, 9.0, 9.0]
    assert q.vehicle.occluder_target != 1234 and q.vehicle.wheel_centres['FL'] != [9.0, 9.0, 9.0]
    assert q.vehicle.mesh is p.vehicle.mesh


def test_frustum_cache_survives_more_poses_than_it_keeps():
    """A drag through more than 200 camera poses cleared the cache and then read the pose just stored (KeyError in
    the 3D view's paint)."""
    from vantage.ui.scene import FrustumCache
    p = samples.tigor_sample_project()
    cam = copy.deepcopy(p.variant.cameras[0])
    fc = FrustumCache()
    for k in range(450):
        cam.yaw = -80.0 + 0.1 * k
        rim, rays, axis = fc.get(p, cam)
        assert np.allclose(rays[0, 0], cam.pos()) and len(fc.d) <= 200


def test_width_leaves_out_mirrors_whatever_they_are_called():
    """The import wizard's spec check counted the mirrors in the width unless the mirror part was named exactly
    'Mirrors' (2.036 m measured against the Tigor's 1.677 m: 'check scale')."""
    from vantage.core import mesh as M
    assert all(M.is_mirror(n) for n in ('Mirrors', 'mirror_left', 'Door mirror R', 'ORVM_L', 'rear-view mirror'))
    assert not any(M.is_mirror(n) for n in ('Body', 'Glass', 'Wheels', 'Rear bumper', 'Roof rack'))
    m = samples.tigor_sample_project().vehicle.mesh
    keep = np.array([p.role != 'ignored' for p in m.parts])[m.face_part]
    tri = m.tri[keep]
    for names in (['Body', 'Glass', 'Wheels', 'Mirrors'], ['Body', 'Glass', 'Wheels', 'mirror_left']):
        mir = np.array([M.is_mirror(n) for n in names])[m.face_part][keep]
        assert np.ptp(M.body_width_points(tri, mir)[:, 1]) == pytest.approx(1.677, abs=0.01)
    # one part, nothing named: the width below the belt line
    assert np.ptp(M.body_width_points(tri, np.zeros(len(tri), bool))[:, 1]) == pytest.approx(1.677, abs=0.01)
    assert np.ptp(tri.reshape(-1, 3)[:, 1]) > 2.0                       # with the mirrors


def test_a_camera_on_the_roof_is_named_for_the_roof():
    """Place on surface named a camera on the roof 'SR' (side right) and aimed it as under a mirror."""
    from vantage.ui import ops
    p = samples.tigor_sample_project()
    up = (0.0, 0.0, 1.0)
    assert ops.location_tag(p, (1.55, 0.0, 1.53), up) == ('T', 'Roof', 'Roof, front centre')
    assert ops.default_aim(p, (1.55, 0.0, 1.53), up) == (0.0, 15.0)
    assert ops.location_tag(p, (0.6, 0.0, 1.5), up)[2] == 'Roof, rear centre'
    assert ops.default_aim(p, (0.6, 0.0, 1.5), up) == (180.0, 15.0)
    assert ops.location_tag(p, (1.2, 0.5, 1.5), up)[:2] == ('TL', 'Roof left')
    assert ops.location_tag(p, (1.2, -0.5, 1.5), up)[:2] == ('TR', 'Roof right')
    assert ops.default_aim(p, (1.2, -0.5, 1.5), up) == (-90.0, 15.0)
    assert ops.location_tag(p, (2.9, 0.0, 0.95), up)[2] == 'Bonnet'
    assert ops.location_tag(p, (-0.6, 0.0, 1.0), up)[2] == 'Boot lid, top'
    # the other surfaces are named as before
    assert ops.location_tag(p, (3.23, 0.0, 0.62), (1.0, 0.0, 0.0))[0] == 'F'
    assert ops.location_tag(p, (1.8, 0.95, 1.0), (0.0, 0.0, -1.0))[0] == 'ML'


def test_a_radar_deep_in_the_nose_is_flagged():
    """The depth of a radar was its distance to the nearest surface, so a radar deep in the nose beside a wheel arch
    passed; it is how much vehicle lies ahead along its boresight."""
    from vantage.core import rules
    from vantage.core.analysis import Engine
    p = samples.tigor_sample_project()
    p.active_uid = next(v for v in p.variants if v.sensors).uid
    vg = Engine().vehicle_geom(p)
    rd = next(s for s in p.variant.sensors if s.tag == 'RD1')
    x0 = rd.x

    def depth_issue(back):
        rd.x = x0 - back
        return [i for i in rules.sensor_issues(p, vg) if i.uids == [rd.uid] and 'boresight' in i.message]
    assert depth_issue(0.0) == [] and depth_issue(0.1) == []           # at or just behind the fascia
    import re
    for back in (0.25, 0.35, 0.5):                                     # 0.35 and 0.5 passed before
        iss = depth_issue(back)
        assert len(iss) == 1, back
        # RD1's optical centre sits 19 mm behind the fascia's face: the depth grows with the move
        assert float(re.search(r'sits (\d+) mm', iss[0].message).group(1)) == pytest.approx(back * 1000 - 19, abs=5)


def test_png_exports_spell_typographic_characters_in_ascii():
    """OpenCV's Hershey fonts are ASCII only: the cable harness PNG printed '??' for every '·', '×' and '→' of its
    titles (the sample's variants are named 'Data rig · 6-cam + lidar + radars')."""
    from vantage.core import pngtext
    assert pngtext.ascii_text('Cable harness · Data rig · 6-cam × 2 → ECU, 40° ±1') == \
        'Cable harness - Data rig - 6-cam x 2 -> ECU, 40 deg +/-1'
    assert pngtext.ascii_text('ok') == 'ok' and '?' not in pngtext.ascii_text('Δz ≤ 3 mm, σ … µs ▸')
    img = np.zeros((40, 400, 3), np.uint8)
    pngtext.put_text(img, 'a · b', (5, 25), 0, 0.6, (255, 255, 255), 1)
    assert img.any()
    import inspect
    from vantage.core import asbuilt, drivelog, harness, perception
    for m in (asbuilt, drivelog, harness, perception):
        assert 'cv2.putText(' not in inspect.getsource(m), m.__name__


def _occluder(tris):
    from vantage.core.occlusion import OccluderGeom
    tri = np.asarray(tris, float)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    return OccluderGeom(tri, n / np.linalg.norm(n, axis=1, keepdims=True))


def test_cube_map_depth_is_the_exact_distance_to_the_surface():
    """A cube map texel stored the distance to the centroid of the triangle drawn there (16-bit mm), not to the
    surface: under a camera 1 m above a 100 m ground plane of two triangles that was ~30 m wherever the ground was hit.
    Now every texel holds the exact distance along its centre ray to the plane of its triangle."""
    from vantage.core import occlusion as OC
    ground = [[[-50, -50, 0], [50, -50, 0], [50, 50, 0]], [[-50, -50, 0], [50, 50, 0], [-50, 50, 0]]]
    cube = OC.build_cube(_occluder(ground), (0.3, -0.2, 1.0))
    D = OC.face_dirs(OC.FACE_N)
    hit = np.isfinite(cube.depth)
    want = 1.0 / -D[..., 2]                                  # the ray (0.3, -0.2, 1) + t d meets z = 0 at t = 1 / -d_z
    down = D[..., 2] < -0.05                                 # (grazing rays run off the 50 m plane)
    assert hit[down].all() and not hit[D[..., 2] > 0].any()
    inside = down & (np.abs(D[..., 0] * want) < 45) & (np.abs(D[..., 1] * want) < 45)
    err = np.abs(cube.depth[inside] - want[inside]) / want[inside]
    assert inside.sum() > 6 * 256 * 256 // 4 and err.max() < 1e-5          # float32, nothing else
    # looked up off the texel centre: within the texel's size
    rng = np.random.default_rng(1)
    d = rng.normal(size=(2000, 3))
    d[:, 2] = -np.abs(d[:, 2]) - 2.0                         # well below the horizon (not grazing)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    got, _ = cube.lookup(d)
    assert np.allclose(got, 1.0 / -d[:, 2], rtol=0.02)


def test_cube_map_paints_a_near_slanted_triangle_over_a_farther_one():
    """The painter's order by centroid distance put a small board 1 m ahead over a large slanted panel the ray meets
    at 0.75 m (its centroid lies 2.4 m away), so the camera 'saw through' the panel. Ordered by each triangle's
    nearest point, the panel is drawn last and the texel holds its exact depth."""
    from vantage.core import occlusion as OC
    panel = [[0.4, -0.3, -0.2], [0.4, 0.3, -0.2], [6.0, 0.0, 3.0]]
    board = [[1.0, -0.05, -0.05], [1.0, 0.05, -0.05], [1.0, 0.0, 0.05]]
    occ = _occluder([panel, board])
    assert np.linalg.norm(occ.centroid[0]) > 2.4 > 1.0 > OC.tri_distance(occ.tri, np.zeros(3))[0]
    cube = OC.build_cube(occ, np.zeros(3))
    f, y, x = (int(v[0]) for v in OC.texel_index(np.array([[1.0, 0.0, 0.0]]), OC.FACE_N))
    d = OC.face_dirs(OC.FACE_N)[f, y, x]
    P = np.array(panel)
    n = np.cross(P[1] - P[0], P[2] - P[0])
    assert float(cube.depth[f, y, x]) == pytest.approx(float(n @ P[0] / (n @ d)), rel=1e-6)
    assert float(cube.lookup(np.array([[1.0, 0.0, 0.0]]))[0][0]) == pytest.approx(0.75, abs=0.01)


def test_tri_distance_against_a_brute_force_minimum():
    """The point–triangle distance (Ericson's regions) against the minimum over a fine barycentric grid."""
    from vantage.core.occlusion import tri_distance
    rng = np.random.default_rng(4)
    tri = rng.normal(size=(300, 3, 3))
    tri[0] = [[0, 0, 0], [1e-9, 0, 0], [2e-9, 0, 0]]                       # degenerate: its nearest corner
    P = rng.normal(size=3) * 2
    w = np.linspace(0, 1, 121)
    U, V = np.meshgrid(w, w)
    keep = (U + V) <= 1
    U, V = U[keep], V[keep]
    pts = tri[:, :1] + U[None, :, None] * (tri[:, 1:2] - tri[:, :1]) + V[None, :, None] * (tri[:, 2:3] - tri[:, :1])
    brute = np.linalg.norm(pts - P, axis=2).min(axis=1)
    got = tri_distance(tri, P)
    assert np.all(got <= brute + 1e-12) and np.allclose(got, brute, atol=2e-3 * np.abs(tri).max())


def test_observability_agreement_band_is_symmetric():
    """The Monte Carlo agreement allowed an empirical σ up to 1 + 4.5 se of the prediction but only down to
    1 / (1 + 4.5 se): a correct prediction failed at seed 61027 with 60 trials (σ̂/σ = 0.69). The band is on
    |log(σ̂/σ)| ≤ 4.5 / √(2(n − 1)), either way."""
    from vantage.core import observability as OB
    trials = 60
    band = 4.5 / math.sqrt(2 * (trials - 1))

    def group(ratio):
        return {'F': {'axes': {'yaw_deg': {'ratio': ratio}, 'x_mm': {'ratio': 1.0}}, 'mahalanobis2': 6.0, 'dof': 6}}
    for r in (0.69, 1 / 0.69, math.exp(band) * 0.999, math.exp(-band) * 1.001):
        ok, worst, b = OB._agreement([group(r)], trials)
        assert ok and b == pytest.approx(band) and worst == pytest.approx(abs(math.log(r))), r
    for r in (math.exp(band) * 1.01, math.exp(-band) / 1.01, 0.5, 2.0):
        assert not OB._agreement([group(r)], trials)[0], r


def test_drive_log_leaves_out_lidar_edges_the_body_mask_hides():
    """The drive-log rotation check zeroed image edges on the own vehicle (dilated) but kept the lidar edges that
    project there: at the plan pose they scored nothing, so a rotation that moved them off the vehicle scored higher.
    Exact cube-map depths grew the mirror cameras' vehicle mask to its true size (11 % of MR's image beyond 5 cm,
    not 6.6 %) and MR's pitch came out 0.25° off (16 standard errors). Lidar edges that project into the mask, or
    within three fine-map widths of it (where the smoothed image still feels the cut), are left out."""
    from vantage.core import drivelog as DL
    from vantage.core.lens import resolve
    from vantage.core.mathutil import BODY_TO_OPTICAL, rot_zyx
    p = samples.tigor_sample_project()
    cam = p.variant.cameras[0]
    intr = resolve(p.module_for(cam), p.lens_for(cam))
    W2O = BODY_TO_OPTICAL @ rot_zyx(cam.yaw, cam.pitch, cam.roll).T
    ha, wa = 96, 120
    sx, sy = wa / intr.width, ha / intr.height
    mask = np.zeros((ha, wa), bool)
    mask[60:, :] = True                                      # the lower part of the image: the own vehicle
    u = np.array([10.5, 60.5, 100.5, 60.5]) / sx
    v = np.array([20.5, 70.5, 90.5, 59.5]) / sy
    d, _ = intr.unproject(u, v)
    Q = (d @ W2O) * 5.0                                       # 5 m out along each pixel's ray, vehicle frame
    assert DL._on_mask(Q, W2O, intr, sx, sy, mask).tolist() == [False, True, True, False]
    assert DL._on_mask(np.zeros((0, 3)), W2O, intr, sx, sy, mask).tolist() == []


def test_computation_times_under_any_seconds_key_are_not_compared():
    """A result hash and --reproduce blanked only keys named 'seconds': an experiment's study_seconds and the drive-log
    and as-built checks' scan_seconds made a bit-for-bit re-run 'not reproduce'."""
    from vantage.core import provenance as PV
    a = {'schema': 'vantage.x/1', 'blind_m2': 0.01, 'study_seconds': {'coverage': 0.256, 'perception': 1.5},
         'metrics': {'scan_seconds': 3.1, 'seconds': 2.0, 'flag': True}}
    b = {'schema': 'vantage.x/1', 'blind_m2': 0.01, 'study_seconds': {'coverage': 0.162, 'perception': 0.9},
         'metrics': {'scan_seconds': 2.2, 'seconds': 1.0, 'flag': True}}
    assert PV.result_digest(a) == PV.result_digest(b)
    assert PV.compare(a, b)['verdict'] == 'reproduces'
    assert PV.neutral(a)['metrics']['flag'] is True                       # not a time: kept
    assert PV.result_digest(dict(b, blind_m2=0.02)) != PV.result_digest(a)


def test_drive_log_peak_width_does_not_depend_on_where_the_peak_falls_in_its_pixel():
    """The drive-log rotation check calls an axis pinned down when its edge-score peak is narrower than 1.5° (one sigma),
    from central differences of the log score at a fixed 0.15°. The score samples pixel maps bilinearly, so it is
    linear between pixel centres: in a quarter-size image (0.26° a pixel) a 0.15° step measured where the peak sat
    within its pixel, not its width, and MR of the sample's 0.25-scale drive came out 2.4–3.0° wide, 'uncertain',
    though every frame agreed (1.1–1.6° at a pixel's step). The step is now at least one pixel."""
    from vantage.core import drivelog as DL
    px, sig, base = 0.262, 0.6, 0.2
    grid = np.arange(-40, 41) * px                           # pixel centres (deg)
    smooth, by_px, fixed = [], [], []
    for phase in (0.0, 0.25, 0.5, 0.75):                      # the peak at this share of a pixel from a centre
        x0 = np.full(3, phase * px)
        g = base + np.exp(-0.5 * ((grid - x0[0]) / sig) ** 2)

        def sampled(d):                                      # linear between pixel centres, per axis
            return float(np.prod([np.interp(v, grid, g) for v in d]))

        def exact(d):
            return float(np.prod([base + math.exp(-0.5 * ((v - x0[0]) / sig) ** 2) for v in d]))
        smooth.append(DL._peak_width(exact, x0, px)[0][0])
        by_px.append(DL._peak_width(sampled, x0, px)[0][0])
        fixed.append(DL._peak_width(sampled, x0, 0.0)[0][0])          # the old fixed 0.15° step
    assert np.allclose(by_px, smooth, rtol=0.03), (by_px, smooth)
    assert max(fixed) / min(fixed) > 1.5                     # 0.50° to 1.04° for the same 0.66° peak
    w, ok = DL._peak_width(lambda d: float(np.prod([base + math.exp(-0.5 * (v / 0.3) ** 2) for v in d])),
                           np.zeros(3), 0.1)
    assert all(ok) and np.allclose(w, w[0]) and w[0] < 0.6    # finer pixels: the 0.15° step, a narrow peak passes


def test_campaign_finds_channels_by_the_rigs_names_and_ignoring_case():
    """Campaign evaluators looked a sensor up only as CAM_<tag> etc., case-sensitively: drives written with nuScenes
    channel names (CAM_FRONT) or a lower-case tag (channels carry it upper-cased) got no rig health, image statistics
    or returns. The rig's own channel for the tag comes first now, then the tag patterns ignoring case."""
    from vantage.core.simcampaign import _channel_of
    streams = {'CAM_FRONT': {}, 'CAM_QL': {}, 'LIDAR_TOP': {}, 'RADAR_RD1': {}}
    rig = {'cameras': [{'tag': 'F', 'channel': 'CAM_FRONT'}, {'tag': 'ql', 'channel': 'CAM_QL'}],
           'lidars': [{'tag': 'L1', 'channel': 'LIDAR_TOP'}]}
    assert _channel_of('F', streams, rig) == 'CAM_FRONT'
    assert _channel_of('L1', streams, rig) == 'LIDAR_TOP'
    assert _channel_of('ql', streams) == 'CAM_QL'                  # no rig: the tag pattern, ignoring case
    assert _channel_of('rd1', streams) == 'RADAR_RD1'
    assert _channel_of('MR', streams, rig) is None
    assert _channel_of('F', {'CAM_F': {}}, {'cameras': [{'tag': 'F', 'channel': 'CAM_FRONT'}]}) == 'CAM_F'


def test_critical_carla_uses_the_actual_mount_of_a_faulted_camera():
    """cam_pose read 'pose_actual' from rig/sim_rig.json, which the bridge never writes back (it moves an in-memory
    copy); the actual mount is in run.json faults_injected. with_actual_mounts puts it on the rig."""
    import numpy as np
    from vantage.core.critical_carla import cam_pose, with_actual_mounts
    plan = {'x': 3.6, 'y': 0.0, 'z': 0.6, 'yaw': 0.0, 'pitch': 30.0, 'roll': 0.0}
    rig = {'cameras': [{'tag': 'F', 'pose_vehicle': dict(plan)}, {'tag': 'R', 'pose_vehicle': dict(plan)}]}
    run = {'faults_injected': [
        {'kind': 'rotation', 'tag': 'F', 'actual_pose': dict(plan, pitch=30.5)},
        {'kind': 'translation', 'tag': 'F', 'actual_pose': dict(plan, pitch=30.5, z=0.61)},
        {'kind': 'drop', 'tag': 'R'}]}
    got = with_actual_mounts(rig, run)
    f = next(c for c in got['cameras'] if c['tag'] == 'F')
    R, pos = cam_pose(f)
    R0, _ = cam_pose(rig['cameras'][0])
    assert pos[2] == 0.61 and not np.allclose(R, R0)               # the last row: where the mount ended up
    assert 'pose_actual' not in rig['cameras'][0]                   # the caller's rig is not changed
    assert 'pose_actual' not in next(c for c in got['cameras'] if c['tag'] == 'R')
    assert with_actual_mounts(rig, {'faults_injected': []}) is rig and with_actual_mounts(rig, None) is rig
