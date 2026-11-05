"""Rig perception (core.rig_perception, core.bev_backends, core.perception_scenes, the page): the monocular lift on
projected boxes of known objects (centres within the predicted σ, through a pinhole and the exact fisheye model), the
lens round trip, ego-motion compensation, cross-camera fusion recovering a triangulated point, nuScenes matching and
AP on hand-made cases worked out by hand, the vectorised weighted AP against the reference, region assignment, the
block bootstrap's pairing (a rig against itself: zero difference, an interval holding zero; a rig without its left
camera: a significant loss on the left only), the external backend with a fake detector script and the approval of
its command, perception records against core.perf_model's validator, scene sets through simcampaign's machinery
against the fake carla, the page driven like a user, and (test_real_*) the CARLA recordings of round 8."""
import json
import math
import os
import shutil
import sys

import numpy as np
import pytest

from vantage.core import bev_backends as BB
from vantage.core import detection as DT
from vantage.core import perception_scenes as PS
from vantage.core import regions as RG
from vantage.core import rig_perception as RP
from vantage.core.mathutil import BODY_TO_OPTICAL, rot_zyx

HERE = os.path.dirname(os.path.abspath(__file__))
REAL = os.environ.get('VANTAGE_CARLA_DRIVES', '/nonexistent') + '/'   # the round-8 paired CARLA drives with GT
REAL_A = REAL + 'data-rig-6-cam-lidar-radars'
REAL_B = REAL + 'data-rig-cameras-20-up'
T0 = 1_767_268_800_000_000


# ======================================================================= cameras and projected boxes
def cam(tag, x, y, z, yaw, pitch, W=640, H=400, f=320.0, lens=None):
    R = rot_zyx(yaw, pitch, 0.0) @ BODY_TO_OPTICAL.T          # optical -> vehicle
    K = np.array([[f, 0, (W - 1) / 2], [0, f, (H - 1) / 2], [0, 0, 1.0]])
    c = BB.CameraCal(tag, W, H, R, np.array([x, y, z], float), 'pinhole', K, None, None, math.radians(80.0),
                     'CAM_' + tag)
    if lens is not None:
        c.model, c.lens, c.max_theta = 'exact', dict(lens), math.radians(lens['max_theta_deg'])
    return c


FISHEYE = {'model': 'lens', 'projection': 'kannala_brandt', 'f_px': 300.0, 'kb': [0.02, -0.01, 0.002, 0.0],
           'cx': 640.0, 'cy': 400.0, 'aspect': 1.0, 'max_theta_deg': 95.0, 'sensor_size': [1280, 800],
           'image_size': [640, 400]}


def cuboid_points(c_xy, L, W, H, yaw=0.0, n=9):
    """Points on a box's surface (vehicle frame), standing on z = 0."""
    s = np.linspace(-0.5, 0.5, n)
    a, b = np.meshgrid(s, s)
    a, b = a.ravel(), b.ravel()
    faces = []
    for sign in (-0.5, 0.5):
        faces += [np.stack([np.full_like(a, sign) * L, a * W, (b + 0.5) * H], 1),
                  np.stack([a * L, np.full_like(a, sign) * W, (b + 0.5) * H], 1),
                  np.stack([a * L, b * W, np.full_like(a, sign + 0.5) * H], 1)]
    P = np.concatenate(faces)
    c, s_ = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1.0]])
    return P @ Rz.T + np.array([c_xy[0], c_xy[1], 0.0])


def box_of(cal, P):
    """The tight image box (pixel edges) of vehicle-frame points as the camera sees them, None when it sees none;
    and the share of points outside the image."""
    Po = (P - cal.t) @ cal.R
    u, v, ok = cal.project(Po)
    ok &= Po[:, 2] > 0.05
    inside = ok & (u >= 0) & (u <= cal.width) & (v >= 0) & (v <= cal.height)
    if inside.sum() < 3:
        return None, 1.0
    return [float(u[inside].min()), float(v[inside].min()), float(u[inside].max()), float(v[inside].max())], \
        1.0 - inside.mean()


# ======================================================================= the lift
@pytest.mark.parametrize('lens', [None, FISHEYE])
def test_lens_round_trip(lens):
    c = cam('F', 2.0, 0.0, 1.4, 0.0, 10.0, lens=lens)
    rng = np.random.default_rng(3)
    a, rr = rng.uniform(0, 2 * np.pi, 200), 190.0 * np.sqrt(rng.uniform(0, 1, 200))     # inside the image circle
    u = c.width / 2 + rr * np.cos(a)
    v = c.height / 2 + rr * np.sin(a)
    d, ok = c.rays(u, v)
    assert ok.all() and np.allclose(np.linalg.norm(d, axis=1), 1.0)
    u2, v2, ok2 = c.project(d * 7.0)
    assert ok2.all() and np.abs(u2 - u).max() < 1e-6 and np.abs(v2 - v).max() < 1e-6


@pytest.mark.parametrize('lens', [None, FISHEYE])
def test_lift_projected_boxes_within_predicted_sigma(lens):
    """Cars and pedestrians at known places, their projected tight boxes lifted: noise-free within 1 σ (the σ holds
    the unknown heading's reach), and with box edges perturbed by BOX_PX_SIGMA the radial error within 2 σ for ≥ 85 %
    of the cases (a Gaussian gives 95 %)."""
    c = cam('F', 2.0, 0.0, 1.4, 0.0, 8.0, lens=lens)
    rng = np.random.default_rng(11)
    cases = []
    for cls in ('car', 'pedestrian'):
        L, W, H, _sh = BB.CLASS_PRIORS[cls]
        for r in (6.0, 10.0, 16.0, 25.0):
            for bearing in (-25.0, 0.0, 20.0):
                yaw = float(rng.uniform(0, np.pi))
                xy = (2.0 + r * math.cos(math.radians(bearing)), r * math.sin(math.radians(bearing)))
                box, out = box_of(c, cuboid_points(xy, L, W, H, yaw))
                if box is not None and out == 0.0:              # wholly in the image (a cut box is tested apart)
                    cases.append((cls, xy, box))
    assert len(cases) >= 18
    within1 = []
    for cls, xy, box in cases:
        d = BB.lift_boxes([box], [cls], [0.9], c)
        assert len(d) == 1 and d[0].info['method'] == 'ground+size'
        e = d[0].centre[:2] - np.array(xy)
        u = np.array([xy[0] - 2.0, xy[1]]) / math.hypot(xy[0] - 2.0, xy[1])
        within1.append(abs(e @ u) <= d[0].info['sigma_range_m'] * 1.0 + 1e-9)
        assert abs(e @ np.array([-u[1], u[0]])) <= 2 * d[0].info['sigma_bearing_m']      # the bearing is right
    assert np.mean(within1) >= 0.9
    hits, n = 0, 0
    for k in range(4):
        for cls, xy, box in cases:
            b = np.array(box) + rng.normal(0, BB.BOX_PX_SIGMA, 4)
            d = BB.lift_boxes([b], [cls], [0.9], c)
            if not d:
                continue
            u = np.array([xy[0] - 2.0, xy[1]]) / math.hypot(xy[0] - 2.0, xy[1])
            n += 1
            hits += abs((d[0].centre[:2] - xy) @ u) <= 2 * d[0].info['sigma_range_m']
    assert n >= 0.95 * 4 * len(cases) and hits / n >= 0.85


def test_lift_sigma_grows_as_range_squared():
    """The ground range's σ is the geometry's: far away ≈ r² · σ_pitch / h (pinhole, pitch error only)."""
    c = cam('F', 0.0, 0.0, 1.5, 0.0, 0.0, f=800.0, W=1600, H=1000)
    st = BB.LiftSettings(box_px_sigma=1e-3, pitch_sigma_deg=0.5, use_size=False)
    sig = []
    for r in (20.0, 40.0):
        L, W, H, _ = BB.CLASS_PRIORS['pedestrian']
        box, _ = box_of(c, cuboid_points((r, 0.0), 0.05, 0.05, H))
        d = BB.lift_boxes([box], ['pedestrian'], [1.0], c, st=st)[0]
        sig.append(d.info['sigma_ground_m'])
        assert abs(d.info['sigma_ground_m'] - (r * r + 1.5 ** 2) / 1.5 * math.radians(0.5)) / \
            d.info['sigma_ground_m'] < 0.1
    assert 3.6 < sig[1] / sig[0] < 4.4


def test_lift_cut_boxes_and_classes_without_prior():
    c = cam('F', 2.0, 0.0, 1.4, 0.0, 8.0)
    d = BB.lift_boxes([[100, 0.5, 140, 180]], ['pedestrian'], [0.9], c)       # top cut by the image: ground only
    assert d and d[0].info['method'] == 'ground'
    d = BB.lift_boxes([[100, 100, 140, 400]], ['pedestrian'], [0.9], c)       # bottom cut: no range at all
    assert d == [] or d[0].info['method'] == 'size'
    assert BB.lift_boxes([[100, 100, 140, 180]], ['traffic_light'], [0.9], c) == []


def test_ego_motion_compensation():
    """A detection made 0.5 m of ego travel before the reference time lands 0.5 m further back."""
    c = cam('F', 2.0, 0.0, 1.4, 0.0, 8.0)
    box, _ = box_of(c, cuboid_points((14.0, 1.0), 0.6, 0.6, 1.65))
    T = np.eye(4)
    T[0, 3] = -0.5                                    # capture frame -> reference frame (the ego moved +0.5 m)
    a = BB.lift_boxes([box], ['pedestrian'], [0.9], c)[0]
    b = BB.lift_boxes([box], ['pedestrian'], [0.9], c, T)[0]
    assert np.allclose(b.centre[:2], a.centre[:2] + [-0.5, 0.0], atol=1e-9)


def test_triangulate_and_fusion_recover_the_point():
    p = np.array([15.0, 0.4])
    cams_xy = [np.array([2.0, 0.6]), np.array([2.0, -0.6]), np.array([1.0, 0.0])]
    rays = [(c, math.atan2(*(p - c)[::-1]), 0.003, float(np.hypot(*(p - c)))) for c in cams_xy]
    q, cov, ang = BB.triangulate(rays)
    assert np.allclose(q, p, atol=1e-9) and ang > 4 and np.all(np.linalg.eigvalsh(cov) > 0)
    assert BB.triangulate([(cams_xy[0], 0.0, 0.003, 10.0), (cams_xy[0] + [0, 0.1], 0.0, 0.003, 10.0)]) is None
    # two cameras each lift a pedestrian taller than the prior (biased ranges); fusion triangulates
    cl = cam('L', 2.0, 0.6, 1.4, 25.0, 8.0)
    cr = cam('R', 2.0, -0.6, 1.4, -25.0, 8.0)
    P = cuboid_points(p, 0.5, 0.5, 1.95)
    mono = []
    for c in (cl, cr):
        box, _ = box_of(c, P)
        mono += BB.lift_boxes([box], ['pedestrian'], [0.8 if c is cl else 0.7], c,
                              st=BB.LiftSettings(use_ground=False))
    assert len(mono) == 2 and all(np.hypot(*(m.centre[:2] - p)) > 0.6 for m in mono)
    fused = BB.fuse(mono, BB.LiftSettings(min_triangulation_deg=2.0))
    assert len(fused) == 1 and fused[0].info['fusion'] == 'triangulated' and set(fused[0].sources) == {'L', 'R'}
    assert np.hypot(*(fused[0].centre[:2] - p)) < 0.35 and fused[0].score == 0.8
    # far apart: not associated
    far = BB.lift_boxes([box_of(cr, cuboid_points((14.0, -9.0), .5, .5, 1.65))[0]], ['pedestrian'], [0.9], cr)
    assert len(BB.fuse(mono[:1] + far)) == 2


# ======================================================================= nuScenes matching and AP
def test_match_centres_hand_made():
    gt = [[10.0, 0.0], [20.0, 0.0]]
    det = [[10.3, 0.0], [20.0, 1.5], [30.0, 0.0]]
    sc = [0.9, 0.8, 0.7]
    s, g, e = RP.match_centres(gt, [False, False], det, sc, 0.5)
    assert s.tolist() == [1, 0, 0] and g.tolist() == [0, -1, -1] and abs(e[0] - 0.3) < 1e-12
    s, g, e = RP.match_centres(gt, [False, False], det, sc, 2.0)
    assert s.tolist() == [1, 1, 0] and g.tolist() == [0, 1, -1] and abs(e[1] - 1.5) < 1e-12
    # the higher score takes the object; the other is a false detection even though it is closer
    s, g, _ = RP.match_centres([[5.0, 0.0]], [False], [[5.9, 0.0], [5.1, 0.0]], [0.9, 0.5], 1.0)
    assert s.tolist() == [1, 0]
    # nearest untaken object, then the threshold (strict <)
    s, _, _ = RP.match_centres([[0.0, 0.0]], [False], [[1.0, 0.0]], [0.5], 1.0)
    assert s.tolist() == [0]
    # an ignore region: a detection on it is ignored, not false
    s, g, _ = RP.match_centres([[0.0, 0.0], [8.0, 0.0]], [False, True], [[8.2, 0.0], [0.1, 0.0]], [0.9, 0.8], 1.0)
    assert s.tolist() == [-1, 1] and g.tolist() == [1, 0]


def test_ap_nuscenes_worked_by_hand():
    """Two objects; detections TP (0.9), FP (0.8), TP (0.7): precision 1, 1/2, 2/3 at recall 1/2, 1/2, 1. The devkit
    interpolates precision at recall k/100 (np.interp: 1 below 0.5; at 0.5 the last of the tied points, 1/2; then
    linear to 2/3 at 1), drops k ≤ 10, subtracts 0.1, clips at 0 and divides by 0.9."""
    exp = []
    for k in range(11, 101):
        r = k / 100
        p = 1.0 if r < 0.5 else (0.5 if r == 0.5 else 0.5 + (r - 0.5) / 0.5 * (2 / 3 - 0.5))
        exp.append(max(p - 0.1, 0.0))
    want = sum(exp) / len(exp) / 0.9
    got = RP.ap_nuscenes([0.9, 0.8, 0.7], [1, 0, 1], 2)
    assert abs(got - want) < 1e-12
    assert RP.ap_nuscenes([0.9], [1], 1) == pytest.approx(1.0)            # one object, found first: AP 1
    assert RP.ap_nuscenes([0.9, 0.8], [0, 0], 3) == 0.0
    assert RP.ap_nuscenes([], [], 2) == 0.0 and RP.ap_nuscenes([0.5], [1], 0) is None


def test_ap_weighted_matches_reference_and_duplication():
    rng = np.random.default_rng(5)
    for _ in range(40):
        n = int(rng.integers(1, 40))
        sc = rng.random(n)
        tp = (rng.random(n) < 0.6).astype(float)
        ngt = int(tp.sum() + rng.integers(0, 5)) or 1
        order = np.lexsort((-np.arange(n), -sc))
        ref = RP.ap_nuscenes(sc, tp, ngt)
        assert abs(RP.ap_weighted(tp[order], np.ones((1, n)), [ngt])[0] - ref) < 1e-9
        w = rng.integers(0, 3, n).astype(float)
        if w.sum() == 0:
            continue
        rep = np.repeat(np.arange(n), w.astype(int))
        ref_w = RP.ap_nuscenes(sc[rep], tp[rep], ngt * 2)
        got = RP.ap_weighted(tp[order], w[order][None], [ngt * 2])[0]
        assert abs(got - ref_w) < 1e-9


def test_region_assignment_and_evaluate_frames():
    grid = RG.RegionGrid()
    pts = {(30.0, 0.0): 'S0R2', (0.0, 15.0): 'S2R1', (-50.0, -5.0): 'S4R3', (5.0, -5.0): 'S7R0',
           (0.0, -70.0): 'S6R4', (-90.0, 1.0): 'S4R5'}
    s, r = grid.assign([p[0] for p in pts], [p[1] for p in pts])
    assert [grid.region_id(a, b) for a, b in zip(s, r)] == list(pts.values())
    gt = {'xy': np.array(list(pts)), 'cls': ['car'] * len(pts)}
    dets = [BB.Detection3D('car', 0.9, np.array([x + 0.2, y, 0.75]), np.array([4.5, 1.8, 1.5]))
            for x, y in list(pts)[:4]]
    dets.append(BB.Detection3D('car', 0.8, np.array([20.0, -20.0, 0.75]), np.array([4.5, 1.8, 1.5])))   # false
    m = RP.evaluate_frames([(gt, dets)], grid)
    assert m['car']['S0R2']['recall'] == 1.0 and m['car']['S6R4']['recall'] == 0.0
    assert m['car']['all']['n_gt'] == 6 and m['car']['all']['tp'] == 4 and m['car']['all']['fp'] == 1
    fp_region = grid.region_id(*[int(x[0]) for x in grid.assign([20.0], [-20.0])])
    assert m['car'][fp_region]['fp'] == 1 and m['car'][fp_region]['precision'] == 0.0
    assert m['vehicle']['all']['tp'] == 4 and abs(m['car']['S0R2']['ate_m'] - 0.2) < 1e-9


# ======================================================================= a synthetic recording
OBJ = [('car', 'vehicle.car', (20.0, 3.5)), ('car', 'vehicle.car', (32.0, -4.0)),
       ('pedestrian', 'human.pedestrian.adult', (12.0, -3.0)), ('car', 'vehicle.car', (10.0, 9.0)),
       ('pedestrian', 'human.pedestrian.adult', (6.0, 7.0)), ('car', 'vehicle.car', (16.0, 12.0)),
       ('pedestrian', 'human.pedestrian.adult', (9.0, 5.5)), ('car', 'vehicle.car', (-9.0, 0.0))]
CAMS = {'F': (2.0, 0.0, 1.4, 0.0, 8.0, 20_000), 'L': (1.0, 0.9, 1.3, 90.0, 12.0, 50_000)}
V_EGO = 5.0


def _cal(tag):
    x, y, z, yaw, pitch, _off = CAMS[tag]
    return cam(tag, x, y, z, yaw, pitch)


def make_rec(root, rig, cams=('F', 'L'), n=30, key_every=5, seed=1, scenario='free_drive'):
    """A nuScenes folder with ground truth to the bridge's contract: a lidar at 10 Hz (the instants), cameras firing
    20 and 50 ms after it, the ego driving +x at 5 m/s, objects standing still (annotated at keyframes), camera labels
    (who sees what), images whose first pixel codes (camera, frame) for the fake 2D detector, run.json and the rig
    file. Returns {(camera index, frame): boxes, classes}."""
    import cv2
    from vantage.core.mathutil import quat_from_matrix
    V = os.path.join(root, 'nuscenes', 'v1.0-vantage')
    os.makedirs(V, exist_ok=True)
    sens, cal, sd, ego, samples, ann, inst = [], [], [], [], [], [], []
    gt_doc = {'schema': 'vantage.ground_truth/1', 'classes': [], 'cameras': {}, 'lidars': {}, 'notes': []}
    cats = [{'token': 'c_car', 'name': 'vehicle.car'}, {'token': 'c_ped', 'name': 'human.pedestrian.adult'}]
    sens.append({'token': 's_lid', 'channel': 'LIDAR_TOP', 'modality': 'lidar'})
    cal.append({'token': 'cal_lid', 'sensor_token': 's_lid', 'translation': [1.0, 0.0, 1.8],
                'rotation': [1, 0, 0, 0], 'camera_intrinsic': []})
    rig_cams = []
    for ci, tag in enumerate(cams):
        c = _cal(tag)
        x, y, z, yaw, pitch, _off = CAMS[tag]
        qx, qy, qz, qw = quat_from_matrix(c.R)
        sens.append({'token': f's_{tag}', 'channel': c.channel, 'modality': 'camera'})
        cal.append({'token': f'cal_{tag}', 'sensor_token': f's_{tag}', 'translation': c.t.tolist(),
                    'rotation': [qw, qx, qy, qz], 'camera_intrinsic': c.K.tolist(),
                    'vantage_distortion': {'model': 'none (pinhole)', 'image_size': [c.width, c.height]}})
        rig_cams.append({'tag': tag, 'channel': c.channel, 'pose_vehicle': {'x': x, 'y': y, 'z': z, 'yaw': yaw,
                                                                            'pitch': pitch, 'roll': 0.0},
                         'image': {'width': c.width, 'height': c.height, 'max_theta_deg': 80.0}})
        gt_doc['cameras'][tag] = {'channel': c.channel, 'width': c.width, 'height': c.height, 'frames': []}
    for k, (cls, catn, xy) in enumerate(OBJ):
        inst.append({'token': f'inst{k}', 'category_token': 'c_car' if cls == 'car' else 'c_ped'})
    lookup = {}
    for i in range(n):
        t = T0 + i * 100_000
        stok = f'samp{i}' if i % key_every == 0 else ''
        ego.append({'token': f'e_lid{i}', 'timestamp': t, 'rotation': [1, 0, 0, 0],
                    'translation': [V_EGO * (t - T0) / 1e6, 0.0, 0.0]})
        sd.append({'token': f'e_lid{i}', 'sample_token': stok, 'ego_pose_token': f'e_lid{i}',
                   'calibrated_sensor_token': 'cal_lid', 'timestamp': t, 'fileformat': 'pcd',
                   'is_key_frame': bool(stok), 'filename': f'samples/LIDAR_TOP/l{i}.pcd.bin'})
        if stok:
            samples.append({'token': stok, 'timestamp': t, 'scene_token': 'scene'})
            for k, (cls, catn, xy) in enumerate(OBJ):
                Lk, Wk, Hk, _ = BB.CLASS_PRIORS[cls]
                ann.append({'token': f'a{i}_{k}', 'sample_token': stok, 'instance_token': f'inst{k}',
                            'visibility_token': '4', 'attribute_tokens': [], 'translation': [xy[0], xy[1], Hk / 2],
                            'size': [Wk, Lk, Hk], 'rotation': [1, 0, 0, 0], 'num_lidar_pts': 10,
                            'num_radar_pts': 0})
        for ci, tag in enumerate(cams):
            c = _cal(tag)
            tc = t + CAMS[tag][5]
            ex = V_EGO * (tc - T0) / 1e6
            ego.append({'token': f'e_{tag}{i}', 'timestamp': tc, 'rotation': [1, 0, 0, 0],
                        'translation': [ex, 0.0, 0.0]})
            fn = f'samples/{c.channel}/{tag}_{i}.png'
            sd.append({'token': f'e_{tag}{i}', 'sample_token': '', 'ego_pose_token': f'e_{tag}{i}',
                       'calibrated_sensor_token': f'cal_{tag}', 'timestamp': tc, 'fileformat': 'png',
                       'is_key_frame': False, 'filename': fn})
            boxes, names, objs = [], [], []
            for k, (cls, catn, xy) in enumerate(OBJ):
                Lk, Wk, Hk, _ = BB.CLASS_PRIORS[cls]
                b, out = box_of(c, cuboid_points((xy[0] - ex, xy[1]), Lk, Wk, Hk))
                if b is None or out > 0.0:
                    continue
                boxes.append(b)
                names.append(cls)
                objs.append({'id': k + 1, 'actor_id': k + 1, 'instance_token': f'inst{k}', 'category': catn,
                             'class': cls, 'bbox': [round(v) for v in b], 'pixels': 500, 'bbox_full': b,
                             'truncated': 0.0, 'occluded': 0.0, 'range_m': float(np.hypot(xy[0] - ex, xy[1])),
                             'box3d': {'center': [xy[0] - ex, xy[1], Hk / 2], 'size': [Lk, Wk, Hk], 'yaw_deg': 0}})
            lookup[(ci, i)] = (np.array(boxes, float).reshape(-1, 4), names)
            img = np.zeros((c.height, c.width, 3), np.uint8)
            img[0, 0] = (i % 256, ci, i // 256)
            os.makedirs(os.path.join(root, 'nuscenes', 'samples', c.channel), exist_ok=True)
            cv2.imwrite(os.path.join(root, 'nuscenes', fn), img)
            gt_doc['cameras'][tag]['frames'].append({'sample_data_token': f'e_{tag}{i}', 'timestamp': tc,
                                                     'image': 'nuscenes/' + fn, 'objects': objs})
    for name, tab in (('sensor', sens), ('calibrated_sensor', cal), ('sample_data', sd), ('ego_pose', ego),
                      ('sample', samples), ('sample_annotation', ann), ('instance', inst), ('category', cats),
                      ('attribute', []), ('scene', [{'token': 'scene'}])):
        with open(os.path.join(V, name + '.json'), 'w') as fh:
            json.dump(tab, fh)
    os.makedirs(os.path.join(root, 'ground_truth'), exist_ok=True)
    with open(os.path.join(root, 'ground_truth', 'index.json'), 'w') as fh:
        json.dump(gt_doc, fh)
    os.makedirs(os.path.join(root, 'rig'), exist_ok=True)
    with open(os.path.join(root, 'rig', 'sim_rig.json'), 'w') as fh:
        json.dump({'variant': rig, 'variant_uid': '', 'output': {'gt_range_m': 60.0}, 'cameras': rig_cams,
                   'lidars': [{'tag': 'L1', 'model': 'test-lidar', 'pose_vehicle': {'x': 1.0, 'y': 0.0, 'z': 1.8,
                                                                                     'yaw': 0, 'pitch': 0,
                                                                                     'roll': 0}}],
                   'radars': [], 'vehicle': {'name': 'Test car', 'project': {'length': 4.5, 'width': 1.8,
                                                                            'height': 1.5}}}, fh)
    with open(os.path.join(root, 'run.json'), 'w') as fh:
        json.dump({'schema': 'vantage.sim_run/1', 'status': 'ok', 'variant': rig, 'project': 'Test',
                   'scenario': {'name': scenario, 'params': {}},
                   'scene': {'town': 'TestTown', 'seed': seed, 'weather': 'ClearNoon', 'duration_s': n / 10.0,
                             'traffic': 0}}, fh)
    return lookup


class FakeDetector:
    """The 2D detector of the synthetic recordings: reads the (camera, frame) code in the image's first pixel and
    returns that image's projected boxes, score 0.9."""

    def __init__(self, lookup):
        self.lookup = lookup
        self.calls = 0

    def __call__(self, img):
        self.calls += 1
        i = int(img[0, 0, 0]) + 256 * int(img[0, 0, 2])
        b, names = self.lookup.get((int(img[0, 0, 1]), i), (np.zeros((0, 4)), []))
        return {'boxes': b.astype(np.float32), 'scores': np.full(len(b), 0.9, np.float32),
                'cls': np.array([DT.CLASSES.index(c) for c in names], np.int8),
                'coco': np.zeros(len(b), np.int16), 'ms': 1.0}

    def describe(self):
        return {'model': 'fake', 'name': 'fake projected boxes'}


@pytest.fixture(scope='module')
def recs(tmp_path_factory):
    root = tmp_path_factory.mktemp('rigperc')
    a, b = str(root / 'rig-a'), str(root / 'rig-b')
    la = make_rec(a, 'Rig A', ('F', 'L'))
    lb = make_rec(b, 'Rig B', ('F',))
    return {'a': a, 'b': b, 'la': la, 'lb': lb, 'root': str(root)}


def _st(paths, **kw):
    d = {'recordings': [{'path': p, 'rig': r} for p, r in paths], 'bootstrap': 300, 'block_s': 0.5, 'cache': 'off',
         'min_gt': 5}
    d.update(kw)
    return RP.RigPerceptionSettings.from_dict(d)


def test_recording_reader_and_ground_truth(recs):
    r = RP.PerceptionRecording(recs['a'])
    assert sorted(r.cams) == ['F', 'L'] and r.lidar_channel == 'LIDAR_TOP' and len(r.samples) == 6
    assert len(r.instants('all')) == 30 and len(r.instants('keyframes')) == 6
    t = T0 + 1_200_000                                    # between keyframes 10 and 15: interpolated
    gt = r.gt_at(t, r.ego_at(t))
    ex = V_EGO * 1.2
    k = gt['token'].index('inst0')
    assert np.allclose(gt['xy'][k], [20.0 - ex, 3.5], atol=1e-9) and not gt['ignore'].any()
    gt_end = r.gt_at(T0 + 2_900_000, r.ego_at(T0 + 2_900_000))      # past the last keyframe (2.5 s) by 0.4 s
    assert len(gt_end['token']) == 0
    assert r.de_recording().same_scene(RP.PerceptionRecording(recs['b']).de_recording())
    info = RP.recording_info(recs['a'])
    assert info['ok'] and info['instances'] == len(OBJ)


def test_study_end_to_end_lifted_rig_without_left_camera(recs):
    det = FakeDetector(recs['la'])
    st = _st([(recs['a'], 'Rig A'), (recs['b'], 'Rig B')], min_objects=1)   # the fixture has 8 objects in all
    # rig B's images code camera 0 too (its only camera, F): the same boxes as rig A's F
    res = RP.run(None, settings=st, detector=det)
    assert res is not None and [r['name'] for r in res.rigs] == ['Rig A', 'Rig B']
    a, b = res.cell('Rig A', 'car'), res.cell('Rig B', 'car')
    assert a['n_gt'] == b['n_gt'] > 0                   # the same objects: the scene's, not the rig's
    assert a['recall'] > b['recall'] and a['visible_share'] > b['visible_share']
    assert a['ate_m'] is not None and a['ate_m'] < 1.0
    # the car behind (−9 m) is seen by neither rig: a miss for both
    rear = [c for c in res.comparisons if c['class'] == 'car' and c['region'].startswith('S4')]
    assert rear and all(c['recall'] == c['recall_reference'] == 0.0 for c in rear)
    # the left camera's objects: rig B loses them, significantly, paired
    left = [f for f in res.findings if f['significant'] and f['metric'] == 'recall']
    assert left and all(f['difference'] < 0 for f in left) and all(f['paired'] for f in left)
    assert any(res.grid.parse_id(f['region'])[0] in (1, 2) for f in left if f['region'] != 'all')
    assert all('loses' in f['text'] and 'paired over the same scenes' in f['text'] for f in left)
    assert res.findings[-1]['metric'] == 'note' and 'by chance' in res.findings[-1]['text']
    # pinhole-free: no front-region finding (both rigs see the front with the same camera)
    assert not any(f['significant'] and f['region'] != 'all' and res.grid.parse_id(f['region'])[0] == 0
                   for f in res.findings if f['metric'] != 'note')
    s = RP.summary(res)
    assert s['rigs'] == ['Rig A', 'Rig B'] and s['findings']
    doc = RP.to_doc(res)
    assert doc['schema'] == RP.SCHEMA and doc['frame'].startswith('vehicle') and recs['root'] not in json.dumps(doc)
    names = [n for n, _ in RP.files(res)]
    assert 'rig_perception.json' in names and 'perception_records.jsonl' in names
    assert any(n.startswith('map_car_rig-b_vs_rig-a') for n in names)
    png = dict(RP.files(res))['map_car_rig-a.png']
    assert png[:8] == b'\x89PNG\r\n\x1a\n'
    assert det.calls > 0
    # with the default guard (3 distinct objects) a region one or two objects fill is no finding; the total stays
    res3 = RP.run(None, settings=_st([(recs['a'], 'Rig A'), (recs['b'], 'Rig B')], bootstrap=100), detector=det)
    sig3 = [f for f in res3.findings if f['significant']]
    assert sig3 and all(f['region'] == 'all' or f['n'] >= 5 for f in sig3)
    assert len(sig3) < len([f for f in res.findings if f['significant']])


def test_rig_against_itself_gives_zero_difference(recs):
    c = os.path.join(recs['root'], 'rig-a-copy')
    if not os.path.isdir(c):
        shutil.copytree(recs['a'], c)
    det = FakeDetector(recs['la'])
    res = RP.run(None, settings=_st([(recs['a'], 'A'), (c, 'A again')]), detector=det)
    assert res.comparisons
    for row in res.comparisons:
        assert row['paired']
        if row['recall_difference'] is not None:
            assert row['recall_difference'] == 0.0
            ci = row['recall_difference_ci']
            assert ci is None or (ci[0] <= 0.0 <= ci[1])
    assert not any(f['significant'] for f in res.findings)


def test_unpaired_scenes_are_said(recs):
    other = os.path.join(recs['root'], 'rig-b-seed2')
    if not os.path.isdir(other):
        make_rec(other, 'Rig B', ('F',), seed=2)
    res = RP.run(None, settings=_st([(recs['a'], 'Rig A'), (other, 'Rig B')], bootstrap=100),
                 detector=FakeDetector(recs['la']))
    assert all(not c['paired'] for c in res.comparisons)
    assert any('not paired' in n for n in res.notes)


def test_settings_tolerant():
    st = RP.RigPerceptionSettings.from_dict({'recordings': ['/x', {'path': '/y', 'rig': 'B'}, {'path': '/x'}, 5],
                                             'score': 7, 'backend': 'external:bad name', 'views': 'nope',
                                             'classes': ['car', 'dog'], 'bootstrap': -3, 'view_yaws': ['a', 30]})
    assert [r['path'] for r in st.recordings] == ['/x', '/y'] and st.recordings[1]['rig'] == 'B'
    assert st.score == 0.99 and st.backend == 'lifted' and st.views == 'fisheye' and st.classes == ['car']
    assert st.bootstrap == 0 and st.view_yaws == [30.0]
    assert RP.RigPerceptionSettings.from_dict(None).to_dict() == RP.RigPerceptionSettings().to_dict()
    assert RP.eval_classes(st) == ['car', 'vehicle']
    assert RP.RigPerceptionSettings.from_dict({'backend': 'external:my-det_1'}).backend == 'external:my-det_1'


# ======================================================================= the external backend
class DictSettings:
    def __init__(self):
        self.d = {}

    def value(self, k, default=None):
        return self.d.get(k, default)

    def setValue(self, k, v):
        self.d[k] = v

    def remove(self, k):
        self.d.pop(k, None)


FAKE_EXTERNAL = r'''
import json, sys, os
args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
frames = json.load(open(args['--frames']))
V = os.path.join(args['--dataroot'], args['--version'])
ann = json.load(open(os.path.join(V, 'sample_annotation.json')))
inst = {r['token']: r['category_token'] for r in json.load(open(os.path.join(V, 'instance.json')))}
cat = {r['token']: r['name'] for r in json.load(open(os.path.join(V, 'category.json')))}
name = {'vehicle.car': 'car', 'human.pedestrian.adult': 'pedestrian'}
want = {s['sample_token'] for s in frames['samples']}
res = {t: [] for t in want}
for a in ann:
    if a['sample_token'] in want:
        res[a['sample_token']].append({'sample_token': a['sample_token'], 'translation': a['translation'],
                                       'size': a['size'], 'rotation': a['rotation'], 'velocity': [0, 0],
                                       'detection_name': name[cat[inst[a['instance_token']]]],
                                       'detection_score': 0.8, 'attribute_name': ''})
json.dump({'meta': {'use_camera': True}, 'results': res}, open(args['--out'], 'w'))
'''


def test_external_backend_with_a_fake_detector(recs, tmp_path):
    script = tmp_path / 'fake_det.py'
    script.write_text(FAKE_EXTERNAL)
    qs = DictSettings()
    es = BB.ExternalStore(qs)
    argv = BB.python_argv(script, '--dataroot', '{dataroot}', '--version', '{version}', '--frames', '{frames}',
                          '--out', '{out}')
    es.put('fake-det', argv)
    st = _st([(recs['a'], 'Rig A')], backend='external:fake-det')
    with pytest.raises(RuntimeError, match='not approved'):
        RP.run(None, settings=st, qsettings=qs, work_dir=str(tmp_path / 'w'))
    assert not es.approve('fake-det', argv[:-1])              # only the exact list
    assert es.approve('fake-det', argv) and es.approved('fake-det')
    res = RP.run(None, settings=st, qsettings=qs, work_dir=str(tmp_path / 'w'))
    assert res.timing['frames'] == 6 and any('keyframes' in n for n in res.notes)
    c = res.cell('Rig A', 'car')
    assert c['recall'] == 1.0 and c['ap'] == pytest.approx(1.0) and c['ate_m'] < 1e-6
    assert res.cell('Rig A', 'pedestrian')['recall'] == 1.0
    frames = json.load(open(next((tmp_path / 'w').glob('frames-*.json'))))
    assert frames['schema'] == BB.SCHEMA_FRAMES and len(frames['samples']) == 6
    es.put('fake-det', argv + ['--extra', '1'])                # a changed command withdraws the approval
    assert es.approved('fake-det') is None


def test_read_submission_converts_through_the_ego_pose():
    T = np.eye(4)
    T[:3, 3] = [100.0, 50.0, 0.0]
    yaw = math.radians(90)
    T[:3, :3] = rot_zyx(90.0, 0.0, 0.0)
    doc = {'results': {'s1': [{'translation': [100.0, 60.0, 0.8], 'size': [1.8, 4.5, 1.5],
                               'rotation': [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)], 'detection_name': 'car',
                               'detection_score': 0.7}, {'translation': [0, 0, 0], 'detection_name': 'barrier',
                                                         'detection_score': 0.9}]}}
    out = BB.read_submission(doc, {'s1': T})
    d = out['s1']
    assert len(d) == 1 and np.allclose(d[0].centre, [10.0, 0.0, 0.8], atol=1e-9) and abs(d[0].yaw) < 1e-9
    assert np.allclose(d[0].size, [4.5, 1.8, 1.5])


# ======================================================================= records
def test_records_validate_and_hold_no_paths(recs):
    res = RP.run(None, settings=_st([(recs['a'], 'Rig A'), (recs['b'], 'Rig B')], bootstrap=50),
                 detector=FakeDetector(recs['la']))
    out = RP.to_records(res, add=False)
    assert len(out) == 2
    for r in out:
        assert r['schema'] == 'vantage.perception_record/1' and r['detector']['backend'] == 'lifted'
        assert recs['root'] not in json.dumps(r)
        assert set(r['metrics']) <= set(RP.EVAL_CLASSES)
        for cells in r['metrics'].values():
            for rid, c in cells.items():
                assert res.grid.parse_id(rid) is not None and c['tp'] + c['fn'] == c['n_gt']
    try:
        from vantage.core import perf_model as PM
    except ImportError:
        pytest.skip('core.perf_model (the record store) is not there')
    for r in out:
        assert PM.validate_record(r) == [], PM.validate_record(r)[:5]
    store = PM.store_for(path=os.path.join(recs['root'], 'records.jsonl'))
    RP.to_records(res, add=True, store=store)
    assert len(store.records()) == 2 and not any(n.startswith('Record of') for n in res.notes)
    RP.to_records(res, add=True, store=store)                   # the same records again: duplicates, not added
    assert len(store.records()) == 2


# ======================================================================= scene sets (fake carla)
def test_scene_set_plan_and_settings():
    from vantage.core import samples
    p = samples.tigor_sample_project()
    v = next(v for v in p.variants if v.sensors)
    st = PS.SceneSetSettings.from_dict({'set_id': 'standard', 'variants': [v.uid, p.variants[0].uid],
                                        'seeds': [3, 3, 'x', 4], 'traffic': -5, 'duration_s': 0})
    assert st.seeds == [3, 4] and st.traffic == -1 and st.duration_s == 0.0
    pl = PS.plan(p, st)
    n = len(PS.scene_set('standard')['scenes'])
    drives = [d for d in pl.drives if d['role'] == 'drive']
    assert len(drives) == n * 2 * 2 and all(d['gt'] == 'semantic,instance,range' and not d['faults'] for d in drives)
    for seed in (3, 4):                                   # every variant drives the same scenes with the same seed
        a = [(d['scenario']['name'], json.dumps(d['scenario']['params'], sort_keys=True)) for d in drives
             if d['seed'] == seed and d['variant_uid'] == v.uid]
        b = [(d['scenario']['name'], json.dumps(d['scenario']['params'], sort_keys=True)) for d in drives
             if d['seed'] == seed and d['variant_uid'] != v.uid]
        assert a == b and len(a) == n
    assert PS.describe(st)[0]['puts']
    cs = PS.campaign_settings(p, st)
    assert cs.base['town'] == 'Town03' and cs.base['walkers'] == 40 and cs.evaluators == ['detection']


def test_scene_set_drives_through_simcampaign_fake_carla(tmp_path):
    sys.path.insert(0, HERE)
    from test_simcampaign import Runner, _project, BASE
    p = _project(cams=('F', 'ML'))
    p.analysis.sim = dict(BASE)
    v = p.variant
    import copy
    from vantage.core import project as PJ
    w = copy.deepcopy(v)
    w.uid, w.name = PJ.new_uid(), 'Front only'
    w.cameras = [c for c in w.cameras if c.tag == 'F']
    p.variants.append(w)
    st = PS.SceneSetSettings.from_dict({'set_id': 'quick', 'variants': [v.uid, w.uid], 'seeds': [2],
                                        'duration_s': 1.2, 'traffic': 6, 'walkers': 0,
                                        'out_dir': str(tmp_path / 'set')})
    st.duration_s = 1.2
    runner = Runner()
    out = PS.run(p, st, runner=runner)
    assert len(runner.runs) == 6 and all(s == 'ok' for s in out['states'].values())
    recs_ = out['recordings']
    assert [r['rig'] for r in recs_] == [v.name] * 3 + ['Front only'] * 3
    again = Runner()
    PS.run(p, st, runner=again)
    assert again.runs == []                              # every drive reused by its key
    y = PS.yields(recs_, RG.default_grid(p))
    assert y['schema'] == PS.SCHEMA and isinstance(y['object_instants'], dict)
    # the study on the fake drives with an oracle backend: paired over the same scenes
    res = RP.run(p, settings=RP.RigPerceptionSettings.from_dict({'recordings': recs_, 'bootstrap': 50}),
                 backend=Oracle())
    assert res is not None and [r['name'] for r in res.rigs] == [v.name, 'Front only']
    assert all(c['paired'] for c in res.comparisons)


class Oracle(BB.Backend):
    """Detections at the ground truth's places (a perfect 3D detector), for the plumbing."""
    id = 'oracle'
    name = 'Oracle'

    def prepare(self, recording):
        self.rec = recording

    def detect(self, frame):
        gt = self.rec.gt_at(frame.timestamp, frame.ego_pose)
        return [BB.Detection3D(c, 0.9, np.array([x, y, 0.5]), np.ones(3)) for c, (x, y) in zip(gt['cls'], gt['xy'])]


# ======================================================================= the page
from test_gui import app, win, pump, settle, _wait   # noqa: E402,F401 (fixtures)


def _open(win, name, cls):
    if name not in win._page_widgets:
        pg = cls(win.ctx)
        win.pages.addWidget(pg)
        win._page_widgets[name] = pg
    win.set_page(name)
    return win._page_widgets[name]


def test_page_like_a_user(app, win, recs, tmp_path):
    from vantage.ui.workspaces.rig_perception_ws import RigPerceptionPage
    pg = _open(win, 'rig_perception', RigPerceptionPage)
    pump(app)
    assert not pg.b_run.isEnabled()
    pg.detector = FakeDetector(recs['la'])
    assert pg.add_recording(recs['a'])
    assert pg.add_recording(recs['b'])
    assert not pg.add_recording(recs['a'])               # not twice
    pump(app)
    assert pg.recs.count() == 2 and pg.b_run.isEnabled()
    assert [r['rig'] for r in pg.settings.recordings] == ['Rig A', 'Rig B']
    # a field edit is undoable
    f = pg.fields['bootstrap']
    f.set_value(200)
    f.committed.emit(200.0)
    assert pg.settings.bootstrap == 200
    win.store.undo.undo()
    pump(app)
    assert pg.settings.bootstrap == 1000
    f.committed.emit(200.0)
    pg.fields['min_gt'].committed.emit(5.0)
    assert pg.run()
    _wait(app, lambda: pg.result is not None and not pg.busy, 120)
    assert pg.result is not None and pg.tiles['findings'].value != '—'
    assert pg.map_a.grid is not None and pg.map_d.values
    assert pg.region_table.rowCount() > 0 and pg.cmp_table.rowCount() > 0 and pg.find_list.count() > 0
    pg.cls_pick.setCurrentIndex(pg.cls_pick.findData('pedestrian'))
    pg.metric_pick.changed.emit('ap')
    pump(app)
    pg.tabs.setCurrentIndex(TABS_FRAMES)
    pg.slider.setValue(5)
    pump(app)
    assert pg.bev.frame is not None
    paths = pg.export(str(tmp_path / 'exp'))
    assert any(p.endswith('rig_perception.json') for p in paths)
    assert pg.compute_yield()
    _wait(app, lambda: pg.yield_doc is not None, 60)
    assert pg.yield_table.rowCount() > 0
    # rename a rig, make B the reference
    assert pg.rename_rig(1, 'Front only')
    assert pg.make_reference(1)
    assert pg.settings.recordings[0]['rig'] == 'Front only'
    assert pg.stale
    assert pg.remove_recording(0)
    # the external command: saved, approved, never in the project
    pg.backend_pick.changed.emit('external')
    pg.new_external('my-det')
    pg.ext_argv.setPlainText('/bin/true\n{out}')
    assert pg.save_external()
    assert pg.settings.backend == 'external:my-det' and pg.ext_status.text.text().startswith('Not approved')
    assert pg.approve_external() and pg.ext_status.text.text().startswith('Approved')
    assert '/bin/true' not in json.dumps(win.store.project.analysis.rig_perception)
    # project replaced while the page is open
    from vantage.core import samples
    win.store.set_project(samples.tigor_sample_project())
    pump(app)
    assert pg.result is None and pg.recs.count() == 0


TABS_FRAMES = 5


# ======================================================================= real recordings (round 8)
REAL_MODELS = os.environ.get('VANTAGE_DETECT_MODELS', '')
real_recs = pytest.mark.skipif(not (os.path.isdir(REAL_A) and os.path.isdir(REAL_B)),
                               reason='the round-8 CARLA recordings are not on this machine')
real = pytest.mark.skipif(not (os.path.isdir(REAL_A) and os.path.isdir(REAL_B) and REAL_MODELS and
                               os.path.isfile(os.path.join(REAL_MODELS, 'yolox_s.onnx'))),
                          reason='the round-8 CARLA recordings or YOLOX-S (VANTAGE_DETECT_MODELS) are not here')


@real
def test_real_paired_recordings(monkeypatch):
    monkeypatch.setenv(DT.ENV_DIR, REAL_MODELS)
    from vantage.core import samples
    p = samples.tigor_sample_project()
    st = RP.RigPerceptionSettings.from_dict({'recordings': [REAL_A, REAL_B], 'bootstrap': 200, 'cache': 'read',
                                             'instants': 'keyframes'})
    res = RP.run(p, settings=st)
    a, b = res.rigs[0]['name'], res.rigs[1]['name']
    assert res.cell(a, 'vehicle')['n_gt'] == res.cell(b, 'vehicle')['n_gt'] > 100
    assert all(c['paired'] for c in res.comparisons)
    assert 0.05 < res.cell(a, 'vehicle')['recall'] < 0.95 and res.cell(a, 'vehicle')['ate_m'] < 2.0
    assert res.timing['cache_hits'] > 0


@real_recs
def test_real_lift_of_ground_truth_boxes():
    """The lift on the recording's own tight boxes (a perfect 2D detector): median centre error at 10–20 m under
    1.5 m (1.1 m measured), through the exact fisheye model."""
    from vantage.core import detection as DT_
    rec = RP.PerceptionRecording(REAL_A)
    err = []
    for (tag, _t), f in list(rec.cam_gt.items())[::4]:
        cal = rec.cams[tag]
        assert cal.model == 'exact'
        for o in f['objects']:
            b = o.get('box3d')
            cls = DT_.GT_TO_CLASS.get(o['class'])
            if not b or cls not in BB.CLASS_PRIORS or o.get('truncated', 0) > 0.01 or o.get('occluded', 0) > 0.3:
                continue
            c = np.array(b['center'])
            r = math.hypot(c[0] - cal.t[0], c[1] - cal.t[1])
            if not 10 <= r < 20:
                continue
            d = BB.lift_boxes(np.array([o['bbox']], float), [cls], [1.0], cal)
            if d:
                err.append(math.hypot(*(d[0].centre[:2] - c[:2])))
    assert len(err) > 30 and np.median(err) < 1.5
