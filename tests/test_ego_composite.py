"""core.ego_composite: the user's own body composited into CARLA's images, lidar sweeps and ground truth (recorded
drives and the live server), and the checks on it. No CARLA, no GPU, no network: synthetic depth, the fake carla."""
import dataclasses
import json
import math
import os
import threading

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))


def _sample():
    from vantage.core import samples
    p = samples.tigor_sample_project()
    p.active_uid = next(v for v in p.variants if v.sensors).uid
    return p


def _box(x0, x1, y0, y1, z0, z1):
    s = np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)], float)
    F = [[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4],
         [1, 5, 7], [1, 7, 3]]
    return s[np.array(F)]


def _brute(tri, O, D, near=0.0):
    v0 = tri[:, 0]
    e1, e2 = tri[:, 1] - v0, tri[:, 2] - v0
    out = []
    for d in D:
        p = np.cross(d, e2)
        det = (p * e1).sum(1)
        inv = 1 / np.where(abs(det) < 1e-14, np.nan, det)
        tv = O - v0
        u = (tv * p).sum(1) * inv
        q = np.cross(tv, e1)
        v = (q @ d) * inv
        t = (e2 * q).sum(1) * inv
        ok = (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9) & (t > near)
        out.append(np.where(ok, t, np.inf).min())
    return np.array(out)


# =========================================================================== the pieces
def test_settings_tolerate_junk_and_clamp():
    from vantage.core import ego_composite as EC
    s = EC.EgoSettings.from_dict({'live': 'no', 'neighbours': 99, 'paint': 'red', 'carla_car': 'paint it',
                                  'heading_step_deg': float('nan'), 'inpaint_px': -4, 'jpeg_quality': '95'})
    assert not s.live and s.neighbours == 20 and s.paint == '#b8bcc2' and s.carla_car == 'inpaint'
    assert s.heading_step_deg == EC.HEADING_STEP and s.inpaint_px == 1.0 and s.jpeg_quality == 95
    assert EC.EgoSettings.from_dict(None) == EC.EgoSettings()
    assert EC.EgoSettings.from_dict(EC.EgoSettings(paint='#112233').to_dict()).paint == '#112233'
    p = _sample()
    assert EC.settings_of(p) == EC.EgoSettings()            # an old project: no slot content
    from vantage.core import asbuilt as AB                  # the as-built check's new switch (default: the cube)
    assert AB.AsBuiltSettings().exact_body is False and AB.AsBuiltSettings.from_dict({'exact_body': 'yes'}).exact_body
    assert AB.AsBuiltSettings.from_dict({'exact_body': 3.5}).exact_body is False


def test_lens_unprojection_is_core_lens_for_every_projection():
    """The numpy-only lens (CARLA's Python) against core.lens on the rig's lens blocks, every projection."""
    from vantage.core import ego_composite as EC, lens as L, simrig as SR
    p = _sample()
    v = p.variant
    rig = SR.build(p, SR.SimSettings.from_dict({'image_scale': 0.1}))
    for proj in ('kannala_brandt', 'equidistant', 'equisolid', 'stereographic', 'orthographic', 'rectilinear'):
        x = np.linspace(0, 1.2, 50)
        kb = (0.01, -0.02, 0.003, -0.0004)
        assert np.allclose(EC.theta_of_r(x, proj, kb), L.theta_of_r(x, 1.0, proj, kb), atol=1e-9), proj
    for cam in rig.doc['cameras']:
        lens = cam['lens']
        w, h = cam['image']['width'], cam['image']['height']
        u, vv = EC.pixel_grid(w, h)
        d, th, ok = EC.lens_unproject(lens, u, vv)
        c = v.by_tag(cam['tag'])
        intr = L.resolve(p.module_for(c), p.lens_for(c))
        d2, th2 = intr.unproject(u * intr.width / w, vv * intr.height / h)
        assert np.degrees(np.arccos(np.clip((d * d2).sum(1), -1, 1))).max() < 1e-4
        # and back through core.groundtruth.project (the bridge's projection)
        uu, vv2, _, ok2 = EC.lens_project(lens, d[ok])
        assert np.abs(uu - u[ok]).max() < 1e-6 and np.abs(vv2 - vv[ok]).max() < 1e-6


def test_the_caster_is_exact_against_brute_force():
    from vantage.core import ego_composite as EC
    rng = np.random.default_rng(3)
    T = np.concatenate([_box(1, 2, -0.5, 0.5, 0, 1), _box(-0.3, 0.3, -0.3, 0.3, -0.3, 0.3) + [0, 0, 0],
                        rng.normal(0, 2, (60, 3, 3))])
    c = EC.Caster(T)
    for O in (np.zeros(3), np.array([1.5, 0.0, 0.5]), np.array([0.0, 0.31, 0.0])):
        D = rng.normal(size=(800, 3))
        D /= np.linalg.norm(D, axis=1, keepdims=True)
        t, i = c.cast(O, D, 0.0, grid=24)
        b = _brute(T, O, D)
        assert np.array_equal(np.isfinite(t), np.isfinite(b))
        f = np.isfinite(b)
        assert np.abs(t[f] - b[f]).max() < 1e-9
        assert (i[~f] == -1).all() and (i[f] >= 0).all()
    # shadows: parallel rays from many points towards s, against brute force; under the first box: blocked
    P = np.r_[[[1.5, 0.0, -0.5]], rng.uniform(-4, 4, (300, 3))]
    for s in (np.array([0.0, 0.0, 1.0]), np.array([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8])):
        got = c.blocked(P, s, near=1e-3)
        ref = np.array([np.isfinite(_brute(T, q, s[None], 1e-3)[0]) for q in P])
        assert np.array_equal(got, ref)
    assert c.blocked(P[:1], np.array([0.0, 0.0, 1.0]))[0]


def test_a_cameras_layer_is_the_occlusion_cube_and_the_exact_asbuilt_mask():
    """The front camera's body through its lens: the analysis's occlusion cube (0.35° texels) agrees but at a texel
    of its outline; core.asbuilt's exact mask (exact_body) is the same cast, pixel for pixel; the range is the
    exact distance to the hit triangle."""
    from vantage.core import asbuilt as AB, ego_composite as EC, simrig as SR
    from vantage.core.analysis import Engine
    from vantage.core.lens import resolve
    p = _sample()
    e = Engine()
    mesh = EC.ego_mesh(p, engine=e)
    assert (mesh.owner >= 0).any() and len(mesh.anchors) >= 6        # the housings are in
    c = EC.Caster(mesh.tri)
    rig = SR.build(p, SR.SimSettings.from_dict({'image_scale': 0.15}))
    for tag in ('F', 'ML'):
        cam = next(x for x in rig.doc['cameras'] if x['tag'] == tag)
        w, h = cam['image']['width'], cam['image']['height']
        L = EC.ego_layer(mesh, c, cam['lens'], cam['pose_vehicle'], w, h)
        c0 = p.variant.by_tag(tag)
        intr = resolve(p.module_for(c0), p.lens_for(c0))
        body, circle, bd = AB.sim_mask(p, c0, intr, (h, w), e)
        iou = (body & L.mask).sum() / (body | L.mask).sum()
        assert iou > 0.95, (tag, iou)
        both = body & L.mask
        assert np.median(np.abs(bd[both] - L.range[both])) < 0.005
        ex, _, exd = AB.sim_mask(p, c0, intr, (h, w), e, exact=True)
        assert np.array_equal(ex, L.mask) and np.allclose(exd[ex], L.range[ex], atol=1e-5)
        # the hit point lies on its triangle's plane
        P = L.points()
        tri = mesh.tri[L.tri[L.mask]]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        assert np.abs(((P - tri[:, 0]) * n).sum(1)).max() < 1e-6


def test_a_turned_layer_is_the_planned_one_turned():
    """A camera turned by 2° pitch: its layer equals the layer of a camera planned 2° further down."""
    from vantage.core import ego_composite as EC, simrig as SR
    p = _sample()
    mesh = EC.ego_mesh(p)
    c = EC.Caster(mesh.tri)
    cam = SR.build(p, SR.SimSettings.from_dict({'image_scale': 0.1})).doc['cameras'][0]
    w, h = cam['image']['width'], cam['image']['height']
    A = EC.ego_layer(mesh, c, cam['lens'], cam['pose_vehicle'], w, h, deg=[0, 2, 0])
    pose = dict(cam['pose_vehicle'], pitch=cam['pose_vehicle']['pitch'] + 2)
    B = EC.ego_layer(mesh, c, cam['lens'], pose, w, h)
    assert np.array_equal(A.mask, B.mask) and np.allclose(A.range[A.mask], B.range[B.mask])
    assert A.mount == ((0.0, 2.0, 0.0), None)
    P = EC.ego_layer(mesh, c, cam['lens'], cam['pose_vehicle'], w, h)
    assert not np.array_equal(P.mask, A.mask)


def test_compositing_on_synthetic_depth():
    """Body nearer than CARLA's range: painted; farther (a pedestrian between): CARLA's pixel kept; CARLA's car not
    covered by the body: a hole; CARLA's car under the body: covered."""
    from vantage.core import ego_composite as EC
    h, w = 4, 6
    rng = np.full((h, w), np.inf, np.float32)
    rng[2:, :] = 1.0                                       # the body: the two bottom rows at 1 m
    L = EC.Layer(w, h, rng, np.zeros((h, w), np.int32), np.ones((h, w), bool), np.zeros((h * w, 3), np.float32),
                 np.zeros(3), np.eye(3))
    img = np.full((h, w, 3), 100, np.uint8)
    car = np.zeros((h, w), bool)
    car[1:3, 0:3] = True                                   # CARLA's car: rows 1-2, columns 0-2
    carla_rng = np.full((h, w), 20.0, np.float32)
    carla_rng[3, 5] = 0.5                                  # something nearer than the body
    body_bgr = np.tile(np.array([[10, 20, 30]], np.uint8), (int(L.mask.sum()), 1))
    out, body, holes = EC.composite(img, L, body_bgr, car, carla_rng, True)
    assert body[2:, :].sum() == 2 * w - 1 and not body[3, 5] and not body[:2].any()
    assert (out[body] == [10, 20, 30]).all() and (out[3, 5] == 100).all()
    assert holes.tolist() == [[False] * w, [True, True, True, False, False, False], [False] * w, [False] * w]
    # live: no range image, the body wherever it is
    out2, body2, holes2 = EC.composite(img, L, body_bgr, car, None, False)
    assert body2.sum() == 2 * w and holes2.sum() == 3
    filled = EC.inpaint(out, holes, 2.0)
    assert (filled[holes] != 0).all()


def test_holes_are_filled_from_neighbouring_frames():
    """A camera over a ground plane painted with a ramp, moving forward: the pixels a hole hides in frame 1 are taken
    from frame 0's view of the same ground, at the right place (to a pixel: the ramp's colour agrees)."""
    from vantage.core import ego_composite as EC
    w, h = 64, 48
    lens = {'model': 'pinhole', 'K': [[40.0, 0, 31.5], [0, 40.0, 23.5], [0, 0, 1]], 'max_theta_deg': 60.0,
            'image_size': [w, h]}
    pose = {'x': 0.0, 'y': 0.0, 'z': 2.0, 'yaw': 0.0, 'pitch': 60.0, 'roll': 0.0}
    R = EC.mount_rotation(pose)
    pos = EC.mount_position(pose)
    rp = EC._Reprojector(lens, w, h, R, pos)
    d_v = rp.d_opt @ rp.W2O

    def render(x_ego):
        t = -pos[2] / d_v[:, 2]
        G = pos + d_v * t[:, None]
        gx = G[:, 0] + x_ego
        col = np.clip(40 + 25 * gx + 15 * (G[:, 1] + 3), 0, 255).astype(np.uint8)       # a smooth ramp on the road
        img = np.repeat(col[:, None], 3, 1).reshape(h, w, 3)
        return img, (t * np.linalg.norm(d_v, axis=1)).reshape(h, w).astype(np.float32)
    poses = [(np.eye(3), np.array([0.0, 0, 0])), (np.eye(3), np.array([0.3, 0, 0]))]
    i0, r0 = render(0.0)
    i1, r1 = render(0.3)
    sem = np.full((h, w), 7, np.uint8)
    holes = np.zeros((h, w), bool)
    holes[30:40, 20:44] = True
    Pg, idx = rp.world_points(r0, sem, np.zeros((h, w), bool), poses[0])
    col, fsem, frng, filled = rp.fill(holes, poses[1], [(Pg, idx, i0, sem, r0)])
    assert filled.sum() > 0.95 * holes.sum()
    err = np.abs(col[filled].astype(int) - i1[filled]).max(axis=1)
    assert np.median(err) <= 3 and err.max() <= 12, (np.median(err), err.max())   # nearest sampling: < 1 pixel
    assert np.abs(frng[filled] - r1[filled]).max() < 0.15
    assert (fsem[filled] == 7).all()


def test_lidar_points_behind_the_body_are_removed_or_moved():
    from vantage.core import ego_composite as EC
    T = _box(1.0, 2.0, -1.0, 1.0, 0.0, 1.0)
    c = EC.Caster(T)
    P = np.array([[5.0, 0.0, 0.5], [5.0, 8.0, 0.5], [-3.0, 0.0, 0.5]])     # behind the box, beside it, behind us
    keep, newP, moved = EC.composite_points(P, np.eye(3), np.zeros(3), c, True)
    assert keep.tolist() == [False, True, True] and not moved.any()
    keep, newP, moved = EC.composite_points(P, np.eye(3), np.zeros(3), c, False)
    assert keep.all() and moved.tolist() == [True, False, False] and np.allclose(newP[0], [1.0, 0.0, 0.1])


def test_sun_and_heading():
    from vantage.core import ego_composite as EC
    assert np.allclose(EC.sun_world(90, 0), [0, 0, 1], atol=1e-12)
    s = EC.sun_world(0, 0)                                  # on the horizon at azimuth 0: CARLA's −x
    assert np.allclose(s, [-1, 0, 0], atol=1e-12)
    assert np.allclose(EC.to_vehicle([1, 0, 0], math.pi / 2), [0, -1, 0], atol=1e-12)   # heading +y: x is to the right
    assert EC.sun_angles({'preset': 'ClearSunset'})[0] == 15
    assert EC.sun_angles({'params': {'sun_altitude_angle': 30, 'sun_azimuth_angle': 120, 'cloudiness': 40}}) == \
        (30.0, 120.0, 40.0)
    assert EC.sun_angles({'preset': 'ClearNoon', 'measured_azimuth': 77.0})[1] == 77.0
    assert EC.irradiance(-10, 0)[0] == 0.0 and EC.irradiance(45, 100)[0] < EC.irradiance(45, 0)[0]
    q = [math.cos(0.3), 0, 0, math.sin(0.3)]                 # 0.6 rad about z
    assert abs(EC.heading_of_quat(q) - 0.6) < 1e-12 and abs(EC.heading_of_quat_R(EC._quat_R(q)) - 0.6) < 1e-12
    assert EC.tone(np.array([[0.3, 0.3, 0.3]]))[0, 0] == pytest.approx(0.3)
    assert EC.tone(np.array([[5.0, 5.0, 5.0]])).max() <= 1.0


def test_shading_is_lit_by_the_sun_and_exposed_like_the_scene():
    from vantage.core import ego_composite as EC, simrig as SR
    p = _sample()
    mesh = EC.ego_mesh(p)
    c = EC.Caster(mesh.tri)
    cam = SR.build(p, SR.SimSettings.from_dict({'image_scale': 0.1})).doc['cameras'][2]   # ML: sees the side
    L = EC.ego_layer(mesh, c, cam['lens'], cam['pose_vehicle'], cam['image']['width'], cam['image']['height'])
    noon = EC.shade(mesh, L, np.array([0, 0, 1.0]), 90, 0)
    night = EC.shade(mesh, L, None, -90, 0)
    assert noon.mean() > 5 * night.mean() and np.isfinite(noon).all()
    shd = EC.ShadowCache(mesh, c).get(L, 'k', np.array([0, 0, 1.0]), 0.0)
    assert shd.dtype == bool and len(shd) == L.mask.sum()
    assert EC.exposure_gain(0.18 * sum(EC.irradiance(90, 0)), 90, 0) == pytest.approx(1.0)


# =========================================================================== a recorded drive (fake carla)
@pytest.fixture(scope='module')
def fake_drive(tmp_path_factory):
    from test_sim import _mini_run
    out = tmp_path_factory.mktemp('ego') / 'drive'
    r = _mini_run(out, {'image_scale': 0.1, 'duration_s': 1.0, 'traffic': 2}, cams=('F', 'ML'))
    assert r['report']['status'] == 'ok'
    return r


def test_a_recorded_drive_is_composited(fake_drive, tmp_path):
    """The fake drive (CARLA's box car) composited with the Tigor body: the body's pixels are the project's layer,
    the ground truth says ego there (instance EGO_INSTANCE, the body's range), CARLA's car is gone from the labels,
    the lidar lost the points the body hides, the original drive is untouched (a copy with hard links)."""
    import cv2
    from vantage.core import ego_composite as EC, groundtruth as GT
    p = fake_drive['project']
    src = fake_drive['out']
    before = {f: os.path.getmtime(os.path.join(dp, f)) for dp, _, fs in os.walk(src) for f in fs}
    out = str(tmp_path / 'ego')
    st = EC.EgoSettings(neighbours=3)
    rep = EC.composite_drive(p, src, out, st)
    assert rep['schema'] == EC.SCHEMA and set(rep['cameras']) == {'F', 'ML'}
    assert {f: os.path.getmtime(os.path.join(dp, f)) for dp, _, fs in os.walk(src) for f in fs} == before
    gt = GT.read(out)
    assert gt['conventions']['ego_instance'] == EC.EGO_INSTANCE
    for tag in ('F', 'ML'):
        c = rep['cameras'][tag]
        assert c['frames'] > 3 and c['body_px'] > 0
        fr = gt['cameras'][tag]['frames'][2]
        sem, inst, rng = GT.load_semantic(gt, fr), GT.load_instance(gt, fr), GT.load_range(gt, fr)
        body = cv2.imread(os.path.join(out, EC.EGO_DIR, f'CAM_{tag}.body.png'), cv2.IMREAD_GRAYSCALE) == 255
        ego = sem == EC.EGO_CLASS
        assert (inst[ego] == EC.EGO_INSTANCE).all()
        # every ego pixel is the project's body; the body's pixels are ego unless something was nearer
        assert not (ego & ~body).any() and (ego & body).sum() >= 0.9 * body.sum()
        lay = cv2.imread(os.path.join(out, EC.EGO_DIR, f'CAM_{tag}.range.png'), cv2.IMREAD_UNCHANGED) / 256.0
        assert np.abs(rng[ego] - lay[ego]).max() < 2 / 256
        osem = GT.load_semantic(GT.read(src), fr)
        hole = (osem == EC.EGO_CLASS) & ~body
        if hole.any():                                     # CARLA's car outside the body: filled or unlabelled
            assert not (sem[hole] == EC.EGO_CLASS).any()
    assert c['hole_px'] >= c['filled_px'] >= 0
    lid = rep['lidars']
    if lid:
        L = next(iter(lid.values()))
        assert L['points_out'] == L['points_in'] - L['blocked_by_body'] - L['carla_car_removed']
    run = json.load(open(os.path.join(out, 'run.json')))
    assert run['ego_composite']['schema'] == EC.SCHEMA and not os.path.exists(os.path.join(out, 'rosbag'))
    coco = json.load(open(os.path.join(out, 'ground_truth', 'coco', 'CAM_F.json')))
    assert coco['info']['ego_composite']
    # a second run into the same folder replaces it; a folder that is not a drive is refused
    EC.composite_drive(p, src, out, EC.EgoSettings(neighbours=0, lidar=False, carla_car='invalid'))
    with pytest.raises(ValueError, match='another folder'):
        EC.composite_drive(p, src, src)
    other = tmp_path / 'other'
    other.mkdir()
    (other / 'x.txt').write_text('mine')
    with pytest.raises(ValueError, match='not a composited drive'):
        EC.composite_drive(p, src, str(other))


def test_a_drive_without_ground_truth_is_refused(fake_drive, tmp_path):
    import shutil
    from vantage.core import ego_composite as EC
    d = tmp_path / 'nogt'
    shutil.copytree(fake_drive['out'], d, ignore=shutil.ignore_patterns('ground_truth'))
    with pytest.raises(ValueError, match='no ground truth'):
        EC.composite_drive(fake_drive['project'], str(d), str(tmp_path / 'o'))


def test_a_turned_drive_records_its_fault_and_the_composite_follows_it(fake_drive, tmp_path):
    """turn_drive turns F by 2° pitch exactly (the rim black), run.json records it, actual_mounts reads it back, and
    the composited body follows the turned mount: its mask is the layer of the turned camera."""
    import cv2
    from vantage.core import ego_composite as EC
    p = fake_drive['project']
    t = EC.turn_drive(fake_drive['out'], str(tmp_path / 'turned'), 'F', [0, 2, 0])
    assert t['deg'] == [0.0, 2.0, 0.0] and 0 <= t['rim_share'] < 0.2
    rig, run = EC.drive_rig(t['out_dir'])
    pose, deg, m = EC.actual_mounts(rig, run)['F']
    assert deg == [0.0, 2.0, 0.0] and m is None
    rep = EC.composite_drive(p, t['out_dir'], str(tmp_path / 'ego'), EC.EgoSettings(lidar=False, neighbours=0),
                             cameras=['F'])
    assert rep['cameras']['F']['mount']['deg'] == [0.0, 2.0, 0.0]
    body = cv2.imread(str(tmp_path / 'ego' / EC.EGO_DIR / 'CAM_F.body.png'), cv2.IMREAD_GRAYSCALE) == 255
    cam = next(c for c in rig['cameras'] if c['tag'] == 'F')
    L = EC.ego_layer(EC.ego_mesh(p), None, cam['lens'], cam['pose_vehicle'], body.shape[1], body.shape[0],
                     deg=[0, 2, 0])
    assert np.array_equal(L.mask, body)
    # a second turn composes with the first
    t2 = EC.turn_drive(t['out_dir'], str(tmp_path / 'turned2'), 'F', [0, 1, 0])
    assert t2['deg'] == pytest.approx([0.0, 3.0, 0.0], abs=1e-9)


def test_rotation_errors_and_the_asbuilt_parametrisation():
    from vantage.core import ego_composite as EC
    plan = {'yaw': 0.0, 'pitch': 40.0, 'roll': 0.0}
    assert EC.rotation_error_deg(plan, [0, 1, 0], [0, 1, 0]) == pytest.approx(0, abs=1e-9)
    added = EC.euler_added(plan, [1, 0, 0])                   # a yaw about the camera's own (tilted) axis
    assert EC.rotation_error_deg(plan, added, [1, 0, 0]) == pytest.approx(0, abs=1e-6)
    assert abs(added[2]) > 0.3                                # it is not a plain yaw in the plan's angles
    assert EC.rotation_error_deg(plan, [0, 0, 0], [0, 0, 2]) == pytest.approx(2.0, abs=1e-9)


def test_validation_checks_are_spec_dicts():
    from vantage.core import ego_composite as EC, usd_export as UX
    for c in EC.validation_checks() + UX.validation_checks():
        assert {'id', 'title', 'model', 'reference', 'tolerance', 'why'} <= set(c)


def test_the_validation_check_runs():
    from vantage.core import ego_composite as EC
    r = EC.check_layer_against_cube()
    assert r['ok'], r


# =========================================================================== live (fake carla)
def _box_project(L_extra=0.25, pitch=40.0):
    """A body that is the fake ego's box made longer at the front (so the composited body differs from CARLA's car),
    a fisheye in front of it looking down."""
    from vantage.core.project import Camera, Project, Variant
    from vantage.core.mesh import Mesh
    from test_sim_live import _fake, BOX_BP
    _, L, W, H, rear, _ = _fake().VEHICLES[BOX_BP]
    x0, x1 = -L / 2 - rear, L / 2 - rear + L_extra
    p = Project()
    p.name = 'Box rig'
    T = _box(x0, x1, -W / 2, W / 2, 0.0, H)
    V = T.reshape(-1, 3)
    p.vehicle.mesh = Mesh(V, np.arange(len(V)).reshape(-1, 3))
    v = Variant(name='Box')
    p.variants = [v]
    p.active_uid = v.uid
    v.cameras = [Camera(tag='F', name='Front', x=x1 + 0.04, y=0.0, z=0.8, yaw=0.0, pitch=pitch, roll=0.0)]
    p.analysis.sim = {'traffic': 0, 'warmup_s': 0.1, 'vehicle': BOX_BP, 'port': 2300, 'tm_port': 8400}
    return p


def _live(p, tmp, args=()):
    import contextlib
    from vantage.core import carla_live as CL, simrig as SR
    from vantage.sim import carla_bridge as CB, live as LV
    from test_sim_live import _fake

    @contextlib.contextmanager
    def ctx():
        fc = _fake()
        rig = CL.build_rig(p, scale=0.25)
        out = os.path.join(str(tmp), 'rig')
        SR.export(rig, p, out)
        rp = os.path.join(out, 'sim_rig.json')
        assert CL.write_ego(rp) == os.path.join(out, CL.EGO_FILE)
        doc, remaps = CB.load_rig(rp)
        a = LV.resolve_args(LV.parser().parse_args([rp] + list(args)), doc)
        fc.reset()
        live = LV.LiveRig(fc, doc, remaps, a, lambda d: None)
        srv = LV.serve(live)
        th = threading.Thread(target=live.run, daemon=True)
        th.start()
        try:
            yield live, LV.LiveClient(srv.server_address[1])
        finally:
            live.request_stop('test over')
            th.join(60)
            LV.close_server(srv, live)
    return ctx()


def test_live_frames_carry_the_project_body_and_the_toggle(tmp_path):
    """The live server loads sim_ego.npz: frames show the project's (longer) body where CARLA's box car is not,
    /F.body.png is the project's body, /F.carla_body.png CARLA's car; POST /ego off gives CARLA's frames back; a 2°
    turn casts the layer again and the drift indicator against the project's body follows the turn."""
    import cv2
    from vantage.core import carla_live as CL, ego_composite as EC, rig_health as rh
    from vantage.core.analysis import Engine
    from vantage.sim import live as LV
    from test_sim_live import _streaming, _frame, _applied
    p = _box_project()
    with _live(p, tmp_path, ['--still', '--noise', '0']) as (live, client):
        _streaming(client)
        st = client.status()
        assert st['ego_composite'] == {'loaded': True, 'on': True, 'path': 'sim_ego.npz', 'error': '',
                                       'cameras': ['F'], 'skipped': []}
        c = st['cameras']['F']
        assert c['body_mask']['of'] == 'project' and c['body_mask']['carla']['url'] == '/F.carla_body.png'
        own = cv2.imdecode(np.frombuffer(client.body('F'), np.uint8), cv2.IMREAD_GRAYSCALE)
        carla = cv2.imdecode(np.frombuffer(LV.LiveClient(client.port)._open(
            f'http://127.0.0.1:{client.port}/F.carla_body.png', timeout=5).read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        assert (own == 255).sum() > (carla == 255).sum() > 0               # the longer body covers more
        on = _frame(client)
        assert client.status()['cameras']['F']['ego']['frames'] > 0
        assert client.ego(False)['on'] is False
        assert client.status()['cameras']['F']['body_mask']['of'] == 'carla'
        off = _frame(client, after=3)
        body = cv2.resize(own, (on.shape[1], on.shape[0]), interpolation=cv2.INTER_NEAREST) == 255
        diff = np.abs(on.astype(int) - off).sum(2)
        assert diff[body].mean() > 10 * max(diff[~body].mean(), 0.1)        # the change is the body
        client.ego(True)
        ev = CL.body_view(client.body('F'), c['size'], ('own', 'F'))
        evp = rh.expected_view(p, Engine(), p.variant.cameras[0])
        assert (ev.body & evp.body).sum() / (ev.body | evp.body).sum() > 0.9   # the project's body, as planned
        f0 = _frame(client, after=2)
        client.fault({'tag': 'F', 'deg': [0, 2, 0]})
        assert _applied(client, 'F', 'rotation')
        f2 = _frame(client, after=3)
        e = client.status()['cameras']['F']['ego']
        assert e['casts'] >= 1 and e['ms_median'] is not None
        d0 = rh.alignment(cv2.cvtColor(f0, cv2.COLOR_BGR2RGB), ev)
        d2 = rh.alignment(cv2.cvtColor(f2, cv2.COLOR_BGR2RGB), ev)
        assert d0 is not None and d2 is not None and d0 <= 3.0 and d2 - d0 >= 8.0, (d0, d2)
    with _live(p, tmp_path / 'off', ['--no-ego']) as (live, client):
        _streaming(client)
        assert client.status()['ego_composite']['on'] is False
        with pytest.raises(LV.LiveError, match='No body layers'):
            client.ego(True)
        with pytest.raises(LV.LiveError, match='true or false'):
            client._req('POST', '/ego', {'on': 'yes'})


def test_layers_of_another_rig_are_not_used(tmp_path):
    """A sim_ego.npz left beside a rig whose camera has moved: the live server skips that camera's layers."""
    from vantage.core import carla_live as CL, ego_composite as EC, simrig as SR
    from vantage.sim import carla_bridge as CB
    p = _box_project()
    rig = CL.build_rig(p, scale=0.1)
    out = str(tmp_path / 'rig')
    p.variant.cameras[0].pitch += 3.0
    rig2 = CL.build_rig(p, scale=0.1, ego=False)
    SR.export(rig2, p, out)
    CL.write_ego(os.path.join(out, 'sim_rig.json'), rig.ego_npz)
    doc, _ = CB.load_rig(os.path.join(out, 'sim_rig.json'))
    le = EC.LiveEgo.load(os.path.join(out, CL.EGO_FILE), doc)
    assert le.skipped == ['F'] and not le.planned


def test_the_setting_off_builds_no_layers(tmp_path):
    from vantage.core import carla_live as CL
    p = _box_project()
    p.analysis.ego_composite = {'live': False}
    rig = CL.build_rig(p, scale=0.1)
    assert getattr(rig, 'ego_npz', None) is None and any('not composited' in n for n in rig.notes)
    p.analysis.ego_composite = {}
    rig = CL.build_rig(p, scale=0.1)
    z = np.load(__import__('io').BytesIO(rig.ego_npz))
    meta = json.loads(bytes(z['meta']).decode())
    assert meta['schema'] == 'vantage.ego_layers/1' and set(meta['cameras']) == {'F'}


def test_the_layer_is_opengls_depth_buffer(tmp_path):
    """The independent reference: OpenGL (EGL, headless) rasterises the same triangles; skipped without EGL."""
    from vantage.core import ego_composite as EC
    r = EC.check_layer_against_gl(str(tmp_path))
    if r['ok'] is None:
        pytest.skip(r['skip'])
    assert r['ok'], r
    for row in r['rows']:
        assert row['coverage_disagree_share'] < 0.01 and row['range_median_mm'] < 1.0


def test_the_module_contract_on_a_fake_drive(fake_drive, tmp_path):
    from vantage.core import ego_composite as EC
    p = fake_drive['project']
    with pytest.raises(ValueError, match='Choose a recorded'):
        EC.run(p, settings=EC.EgoSettings())
    st = EC.EgoSettings(drive=fake_drive['out'], out_dir=str(tmp_path / 'c'), lidar=False, neighbours=1)
    r = EC.run(p, settings=st, check=False)
    s = EC.summary(r)
    assert json.dumps(s) and set(s['cameras']) == {'F', 'ML'} and 'check' not in s
    paths = EC.export(r, p, str(tmp_path / 'x'))
    assert [os.path.basename(x) for x in paths] == ['ego_composite.json']
    doc = json.load(open(paths[0]))
    assert doc['schema'] == EC.SCHEMA and doc['composite']['cameras']
    assert EC.run(p, settings=st, cancel=lambda: True) is None


def test_a_batch_session_with_the_users_body(tmp_path):
    """run_session(ego=True): the batch mode's live session composites the project's body and measures the drift
    against it (the default stays CARLA's car, as before)."""
    import sys
    from vantage.core import carla_live as CL
    from test_sim_live import FAKE_DIR
    p = _box_project()
    res = CL.run_session(p, 2.0, sys.executable, str(tmp_path / 'run'), scale=0.1, extra_path=FAKE_DIR, scope=False,
                         check_server=False, ego=True)
    assert res['status'] == 'ok'
    c = res['rig_health']['cameras']['F']
    assert c['drift_checked'] and c['drift_against'] == 'project_body'
    assert c['ego_composite']['frames'] > 0 and c['ego_composite']['ms_median'] is not None
    assert (tmp_path / 'run' / 'rig' / CL.EGO_FILE).is_file()
