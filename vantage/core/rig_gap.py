"""Cross-rig domain gap: how far apart two camera rigs are geometrically, as an index a perception team can compare
before training or evaluating a multi-camera model (BEVDet, BEVFusion, PETR, ...) on a rig it was not trained on.

A rig is a list of cameras (centre, optical-to-vehicle rotation, real lens model, optional occlusion by the body):
a variant of a Vantage project (rig_from_variant: the project's lenses and the engine's occlusion cube maps), or a
rig read from a nuScenes calibration folder (calibrated_sensor.json + sensor.json; read_nuscenes_rig) or a KITTI
calibration folder (read_kitti_rig). Every figure is computed from those cameras; nothing is a published number.

The Rig Contrastive Distance (RigGap.total, 0 identical … 1 unrelated) is a weighted mean of six components, each in
0..1 and reported on its own:

- coverage (weight 0.30): the Jensen–Shannon distance (base 2: 0..1) between the two rigs' distributions of pixels
  over viewing directions. Directions are a Fibonacci sphere in the vehicle frame (default 4096, ~10 square degrees
  each); each rig puts on a direction the sum over its cameras that see it (in the image, in the image circle, not
  hidden by the body within 30 m) of the lens's pixels per steradian there ((px/rad)²). JS distance is symmetric,
  bounded and zero only for the same distribution: a rig looking the same way with the same share of pixels.
- resolution (0.20): where both rigs see a direction, |log2| of the ratio of their best px/rad; the median over
  those directions / 2 (two octaves of resolution difference = 1). Same layout, half the resolution: 0.5.
- height_pitch (0.15): the 1-Wasserstein distance between the rigs' camera heights (m) / 1 m and between their
  optical-axis depressions (deg) / 30 deg, averaged (each capped at 1). BEV lifting is sensitive to both.
- fov (0.10): the 1-Wasserstein distance between the rigs' horizontal fields of view (deg) / 90 deg.
- topology (0.10): the mean of |nA − nB| / max(nA, nB) and the total-variation difference between the share of
  directions seen by 0, 1, 2, 3+ cameras (overlap structure).
- bev (0.15): 1 − IoU of the ground cells (1 m, ±extent_m about each rig's origin) each rig sees.

Cameras are matched one to one (Hungarian assignment on optical-axis angle / 90°, centre distance / 2 m and |ΔHFOV| /
90°) to say which camera of B stands in for which of A and how far it is off.

What it is not: a predicted mAP. RigGap.transfer_band ('small', 'moderate', 'large' at total < 0.15, < 0.35, else) is a
heuristic band on this index, not calibrated against any benchmark. Two rigs on different vehicles are compared in
their own vehicle frames (ISO 8855-like: x forward, y left, z up, origin on the ground under the rear axle); an
external rig has no body, so nothing hides its views.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field

import numpy as np

from .analysis import camera_frames
from .lens import Intrinsics, resolve, theta_of_r

WEIGHTS = {'coverage': 0.30, 'resolution': 0.20, 'height_pitch': 0.15, 'fov': 0.10, 'topology': 0.10, 'bev': 0.15}
BANDS = ((0.15, 'small'), (0.35, 'moderate'), (1.01, 'large'))
OCCLUDE_M = 30.0            # a direction is hidden when the body is nearer than this along it
KITTI_VELO_HEIGHT_M = 1.73  # the KITTI recording platform's Velodyne height above ground (Geiger et al., IJRR 2013)


# ---------------------------------------------------------------- rigs
@dataclass
class RigCam:
    tag: str
    C: np.ndarray                   # centre, vehicle frame (m)
    R: np.ndarray                   # optical (x right, y down, z forward) -> vehicle rotation
    intr: Intrinsics
    cube: object = None             # occlusion cube map at C (CubeDepth.lookup(dirs) -> (depth, shade)) or None

    @property
    def axis(self) -> np.ndarray:
        return self.R[:, 2]

    @property
    def height(self) -> float:
        return float(self.C[2])

    @property
    def depression_deg(self) -> float:
        return float(math.degrees(math.asin(max(-1.0, min(1.0, -self.axis[2])))))

    @property
    def hfov_deg(self) -> float:
        """The horizontal field across the image's middle row, within the image circle."""
        i = self.intr
        u = np.array([0.0, float(i.width)])
        d, th = i.unproject(u, np.full(2, i.cy))
        th = np.minimum(th, i.max_theta)
        return float(math.degrees(th[0] + th[1]))


@dataclass
class Rig:
    name: str
    cams: list
    source: str = ''
    occlusion: bool = False

    def summary(self) -> dict:
        return {'name': self.name, 'source': self.source, 'cameras': len(self.cams), 'occlusion': self.occlusion,
                'per_camera': [{'tag': c.tag, 'height_m': round(c.height, 3),
                                'depression_deg': round(c.depression_deg, 2), 'hfov_deg': round(c.hfov_deg, 1),
                                'yaw_deg': round(math.degrees(math.atan2(c.axis[1], c.axis[0])), 1)}
                               for c in self.cams]}


def rig_from_variant(project, variant=None, engine=None, occlusion=True) -> Rig:
    """A Vantage variant's visible cameras with their real lenses and (with an engine) the body, rig and housings as
    occlusion (the coverage analysis's cube maps)."""
    v = variant or project.variant
    cams = []
    vg = engine.vehicle_geom(project, v) if (engine is not None and occlusion) else None
    for c in v.cameras:
        if not c.visible:
            continue
        intr = resolve(project.module_for(c), project.lens_for(c))
        _, W2O = camera_frames(c)
        cube = engine.cube(c.pos(), vg) if vg is not None else None
        cams.append(RigCam(c.tag, np.asarray(c.pos(), float), W2O.T.copy(), intr, cube))
    return Rig(f'{project.name} · {v.name}', cams, 'vantage', vg is not None)


def _intr_from_K(K, size, model='pinhole', k=(0.0, 0.0, 0.0, 0.0), cv_centres=True) -> Intrinsics:
    """An Intrinsics from an OpenCV-style K (fx, fy, cx, cy; pixel centres at integers when cv_centres) and image
    size; 'pinhole' is rectilinear, 'kannala_brandt' / 'equidistant' the OpenCV fisheye model (θd = θ(1 + k1θ² ...))."""
    K = np.asarray(K, float)
    w, h = int(size[0]), int(size[1])
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]) + (0.5 if cv_centres else 0.0), float(K[1, 2]) + (0.5 if cv_centres else 0.0)
    pm = 1e-3
    fisheye = model in ('kannala_brandt', 'equidistant', 'fisheye')
    proj = 'kannala_brandt' if fisheye else 'rectilinear'
    kb = tuple(float(x) for x in (list(k) + [0, 0, 0, 0])[:4]) if fisheye else (0.0, 0.0, 0.0, 0.0)
    r_corner = math.hypot(max(cx, w - cx), max(cy, h - cy) * fx / max(fy, 1e-9)) * pm
    th = float(theta_of_r(r_corner, fx * pm, proj, kb)) if fisheye else math.atan(r_corner / (fx * pm))
    max_theta = min(th + math.radians(1.0), math.radians(100.0 if fisheye else 89.0))
    return Intrinsics(width=w, height=h, pixel_mm=pm, f_mm=fx * pm, projection=proj, kb=kb, cx=cx, cy=cy,
                      max_theta=max_theta, focus_near=0.0, focus_far=math.inf, aspect=fy / fx)


def _quat_R(w, x, y, z) -> np.ndarray:
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def read_nuscenes_rig(folder, image_size=(1600, 900), name=None) -> Rig:
    """The cameras of a nuScenes calibration folder (calibrated_sensor.json and sensor.json: the dataset's own tables
    or Vantage's nuscenes_calib export). Each camera row's translation and rotation (w, x, y, z) map the camera's
    optical frame into the ego frame; camera_intrinsic is its K (pinhole), with Vantage's vantage_distortion block (a
    Kannala–Brandt model and the image size) used when present. Images without a stated size take image_size (the
    dataset's 1600 × 900). Lidars and radars are left out."""
    cs_path = os.path.join(folder, 'calibrated_sensor.json')
    s_path = os.path.join(folder, 'sensor.json')
    with open(cs_path, encoding='utf-8') as fh:
        cal = json.load(fh)
    chan = {}
    if os.path.exists(s_path):
        with open(s_path, encoding='utf-8') as fh:
            for s in json.load(fh):
                chan[s['token']] = (s.get('channel', ''), s.get('modality', ''))
    cams = []
    for row in cal:
        ch, mod = chan.get(row.get('sensor_token'), ('', ''))
        K = row.get('camera_intrinsic') or []
        if (mod and mod != 'camera') or len(K) != 3:
            continue
        vd = row.get('vantage_distortion') or {}
        size = vd.get('image_size') or image_size
        kb = vd.get('coefficients') or []
        model = 'kannala_brandt' if str(vd.get('model', '')).startswith('kannala') and kb else 'pinhole'
        intr = _intr_from_K(K, size, model, kb)
        w, x, y, z = (float(q) for q in row['rotation'])
        tag = vd.get('camera') or (ch[4:] if ch.startswith('CAM_') else ch) or row.get('token', '')[:6]
        cams.append(RigCam(str(tag), np.asarray(row['translation'], float), _quat_R(w, x, y, z), intr, None))
    return Rig(name or f'nuScenes calibration · {os.path.basename(os.path.abspath(folder))}', cams, 'nuscenes', False)


def _kitti_file(path) -> dict:
    out = {}
    with open(path, encoding='utf-8', errors='replace') as fh:
        for ln in fh:
            if ':' not in ln:
                continue
            k, v = ln.split(':', 1)
            try:
                out[k.strip()] = np.array([float(t) for t in v.split()], float)
            except ValueError:
                out[k.strip()] = v.strip()
    return out


def read_kitti_rig(folder, velo_height_m=KITTI_VELO_HEIGHT_M, name=None) -> Rig:
    """The cameras of a KITTI calibration folder. With Vantage's calib_cams_to_vehicle.txt each camera's pose is its
    Tr_cam_xx_to_vehicle; with KITTI's own raw files (calib_cam_to_cam.txt + calib_velo_to_cam.txt) the vehicle frame
    is the Velodyne's (x forward, y left, z up) lowered to the ground by velo_height_m (1.73 m on the KITTI car).
    Intrinsics from K_xx, S_xx and D_xx (DISTORTION_MODEL_xx equidistant: Kannala–Brandt; else pinhole)."""
    c2c = _kitti_file(os.path.join(folder, 'calib_cam_to_cam.txt'))
    names = str(c2c.get('camera_names', '')).split()
    ids = sorted({k[2:] for k in c2c if k.startswith('K_') and len(k) == 4})
    veh_path = os.path.join(folder, 'calib_cams_to_vehicle.txt')
    veh = _kitti_file(veh_path) if os.path.exists(veh_path) else {}
    T_v_c0 = None
    if not veh:
        v2c = _kitti_file(os.path.join(folder, 'calib_velo_to_cam.txt'))
        T_c0_v = np.eye(4)
        T_c0_v[:3, :3] = np.asarray(v2c['R'], float).reshape(3, 3)
        T_c0_v[:3, 3] = np.asarray(v2c['T'], float).reshape(3)
        T_v_c0 = np.linalg.inv(T_c0_v)
        T_v_c0[2, 3] += velo_height_m
    cams = []
    for i, n in enumerate(ids):
        K = np.asarray(c2c[f'K_{n}'], float).reshape(3, 3)
        S = np.asarray(c2c.get(f'S_{n}', [1242, 375]), float).reshape(2)
        D = np.asarray(c2c.get(f'D_{n}', [0, 0, 0, 0, 0]), float).reshape(-1)
        model = 'kannala_brandt' if str(c2c.get(f'DISTORTION_MODEL_{n}', '')) == 'equidistant' else 'pinhole'
        intr = _intr_from_K(K, S, model, D[:4])
        if veh:
            T = np.eye(4)
            T[:3, :4] = np.asarray(veh[f'Tr_cam_{n}_to_vehicle'], float).reshape(3, 4)
        else:
            T_cn_c0 = np.eye(4)
            T_cn_c0[:3, :3] = np.asarray(c2c.get(f'R_{n}', np.eye(3).ravel()), float).reshape(3, 3)
            T_cn_c0[:3, 3] = np.asarray(c2c.get(f'T_{n}', np.zeros(3)), float).reshape(3)
            T = T_v_c0 @ np.linalg.inv(T_cn_c0)
        tag = names[i] if i < len(names) else f'cam{n}'
        cams.append(RigCam(tag, T[:3, 3].copy(), T[:3, :3].copy(), intr, None))
    return Rig(name or f'KITTI calibration · {os.path.basename(os.path.abspath(folder))}', cams, 'kitti', False)


def load_rig(path, **kw) -> Rig:
    """A rig from a calibration folder: nuScenes (calibrated_sensor.json) or KITTI (calib_cam_to_cam.txt)."""
    for sub in ('', 'nuscenes', 'kitti'):
        d = os.path.join(path, sub)
        if os.path.exists(os.path.join(d, 'calibrated_sensor.json')):
            return read_nuscenes_rig(d, **kw)
        if os.path.exists(os.path.join(d, 'calib_cam_to_cam.txt')):
            return read_kitti_rig(d, **kw)
    raise FileNotFoundError(f'no nuScenes or KITTI calibration in {path}')


# ---------------------------------------------------------------- sampling
def fibonacci_sphere(n) -> np.ndarray:
    i = np.arange(n) + 0.5
    z = 1 - 2 * i / n
    r = np.sqrt(np.maximum(0, 1 - z * z))
    ph = math.pi * (3 - math.sqrt(5)) * i
    return np.stack([r * np.cos(ph), r * np.sin(ph), z], 1)


def _px_per_rad(intr, theta):
    return intr.density_px_per_cm(np.asarray(theta, float), 1.0) * 100.0


def _cam_dirs(cam: RigCam, D):
    """For unit vehicle-frame directions D: seen (bool) and px/rad of this camera."""
    opt = D @ cam.R                                   # vehicle -> optical: R^T d, row-wise
    u, v, th, ok = cam.intr.project_dirs(opt)
    if cam.cube is not None:
        depth, _ = cam.cube.lookup(D)
        ok = ok & (depth > OCCLUDE_M)
    return ok, np.where(ok, _px_per_rad(cam.intr, th), 0.0)


def _cam_ground(cam: RigCam, P):
    d = P - cam.C
    dist = np.linalg.norm(d, axis=1)
    opt = d @ cam.R
    u, v, th, ok = cam.intr.project_dirs(opt)
    if cam.cube is not None:
        depth, _ = cam.cube.lookup(d)
        ok = ok & (depth > dist - (0.02 + 0.01 * dist))
    return ok


def direction_profile(rig: Rig, D) -> dict:
    """Per direction: pixels per steradian summed over the cameras that see it, the best px/rad, cameras seeing it."""
    w = np.zeros(len(D))
    best = np.zeros(len(D))
    count = np.zeros(len(D), np.int16)
    for c in rig.cams:
        ok, pr = _cam_dirs(c, D)
        w += pr ** 2
        best = np.maximum(best, pr)
        count += ok
    return {'px_sr': w, 'best_px_rad': best, 'count': count}


def ground_seen(rig: Rig, extent_m=50.0, cell_m=1.0) -> np.ndarray:
    n = int(round(2 * extent_m / cell_m))
    xs = -extent_m + (np.arange(n) + 0.5) * cell_m
    X, Y = np.meshgrid(xs, xs)
    P = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], 1)
    seen = np.zeros(len(P), bool)
    for c in rig.cams:
        seen |= _cam_ground(c, P)
    return seen.reshape(n, n)


# ---------------------------------------------------------------- distances
def js_distance(p, q) -> float:
    """Jensen–Shannon distance (base 2, 0..1) between two non-negative weight vectors."""
    p = np.asarray(p, float)
    q = np.asarray(q, float)
    if p.sum() <= 0 or q.sum() <= 0:
        return 1.0 if (p.sum() > 0) != (q.sum() > 0) else 0.0
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)

    def kl(a, b):
        nz = a > 0
        return float(np.sum(a[nz] * np.log2(a[nz] / b[nz])))
    return float(math.sqrt(max(0.0, 0.5 * kl(p, m) + 0.5 * kl(q, m))))


def wasserstein1(a, b) -> float:
    """1-Wasserstein distance between two empirical 1-D distributions (equal-weight samples)."""
    a, b = np.sort(np.asarray(a, float)), np.sort(np.asarray(b, float))
    if not len(a) or not len(b):
        return 0.0 if len(a) == len(b) else float('inf')
    q = np.linspace(0, 1, 201)[1:-1]
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


def match_cameras(A: Rig, B: Rig) -> list:
    """One-to-one camera matches (Hungarian) on optical-axis angle / 90°, centre distance / 2 m and |ΔHFOV| / 90°."""
    from scipy.optimize import linear_sum_assignment
    if not A.cams or not B.cams:
        return []
    C = np.zeros((len(A.cams), len(B.cams)))
    for i, a in enumerate(A.cams):
        for j, b in enumerate(B.cams):
            ang = math.degrees(math.acos(float(np.clip(a.axis @ b.axis, -1, 1))))
            C[i, j] = ang / 90 + float(np.linalg.norm(a.C - b.C)) / 2 + abs(a.hfov_deg - b.hfov_deg) / 90
    ri, cj = linear_sum_assignment(C)
    out = []
    for i, j in zip(ri, cj):
        a, b = A.cams[i], B.cams[j]
        out.append({'a': a.tag, 'b': b.tag, 'cost': round(float(C[i, j]), 4),
                    'axis_angle_deg': round(math.degrees(math.acos(float(np.clip(a.axis @ b.axis, -1, 1)))), 2),
                    'centre_distance_m': round(float(np.linalg.norm(a.C - b.C)), 3),
                    'height_diff_m': round(b.height - a.height, 3),
                    'hfov_diff_deg': round(b.hfov_deg - a.hfov_deg, 1)})
    return out


@dataclass
class RigGap:
    a: str
    b: str
    components: dict
    weights: dict
    total: float
    transfer_band: str
    matches: list
    figures: dict
    findings: list
    ms: float = 0.0
    maps: dict = field(default_factory=dict, repr=False)      # direction and BEV arrays for the figure


def rig_gap(A: Rig, B: Rig, n_dirs=4096, extent_m=50.0, cell_m=1.0, weights=None) -> RigGap:
    """The Rig Contrastive Distance between rigs A and B (see the module docstring) with its components, the camera
    matches and plain-language findings."""
    import time
    t0 = time.perf_counter()
    W = dict(WEIGHTS, **(weights or {}))
    D = fibonacci_sphere(int(n_dirs))
    pa, pb = direction_profile(A, D), direction_profile(B, D)
    comp, fig = {}, {}
    comp['coverage'] = js_distance(pa['px_sr'], pb['px_sr'])
    both = (pa['count'] > 0) & (pb['count'] > 0)
    if both.any():
        lr = np.log2(pa['best_px_rad'][both] / pb['best_px_rad'][both])
        med = float(np.median(np.abs(lr)))
        fig['resolution_median_ratio_b_over_a'] = float(2 ** -np.median(lr))
    else:
        med = 2.0
        fig['resolution_median_ratio_b_over_a'] = None
    comp['resolution'] = min(1.0, med / 2.0)
    ha, hb = [c.height for c in A.cams], [c.height for c in B.cams]
    da, db = [c.depression_deg for c in A.cams], [c.depression_deg for c in B.cams]
    wh, wp = wasserstein1(ha, hb), wasserstein1(da, db)
    comp['height_pitch'] = 0.5 * (min(1.0, wh / 1.0) + min(1.0, wp / 30.0))
    fa, fb = [c.hfov_deg for c in A.cams], [c.hfov_deg for c in B.cams]
    wf = wasserstein1(fa, fb)
    comp['fov'] = min(1.0, wf / 90.0)
    na, nb = len(A.cams), len(B.cams)

    def shares(cnt):
        return np.array([(cnt == 0).mean(), (cnt == 1).mean(), (cnt == 2).mean(), (cnt >= 3).mean()])
    tv = 0.5 * float(np.abs(shares(pa['count']) - shares(pb['count'])).sum())
    comp['topology'] = 0.5 * (abs(na - nb) / max(na, nb, 1) + tv)
    ga, gb = ground_seen(A, extent_m, cell_m), ground_seen(B, extent_m, cell_m)
    uni = (ga | gb).sum()
    iou = float((ga & gb).sum() / uni) if uni else 1.0
    comp['bev'] = 1.0 - iou
    comp = {k: float(np.clip(v, 0.0, 1.0)) for k, v in comp.items()}
    total = float(sum(W[k] * comp[k] for k in comp) / sum(W[k] for k in comp))
    band = next(b for t, b in BANDS if total < t)
    fig.update({'height_w1_m': wh, 'depression_w1_deg': wp, 'hfov_w1_deg': wf, 'cameras': [na, nb],
                'bev_iou': iou, 'directions_seen': [float((pa['count'] > 0).mean()), float((pb['count'] > 0).mean())],
                'directions_both': float(both.mean()), 'multi_view_share': [float((pa['count'] >= 2).mean()),
                                                                            float((pb['count'] >= 2).mean())]})
    matches = match_cameras(A, B)
    f = []
    f.append(f'Rig Contrastive Distance {total:.3f} ({band} geometric shift; a heuristic band, not a predicted mAP).')
    big = max(comp, key=comp.get)
    f.append(f'The largest component is {big} ({comp[big]:.2f}).')
    if fig['resolution_median_ratio_b_over_a']:
        r = fig['resolution_median_ratio_b_over_a']
        f.append(f'Where both see, {B.name} has {r:.2f}× the angular resolution of {A.name} (median).')
    if wh > 0.1:
        f.append(f'Camera heights differ by {wh:.2f} m (1-Wasserstein): BEV lifting from one to the other shifts.')
    if wp > 3:
        f.append(f'Camera depressions differ by {wp:.1f}° (1-Wasserstein).')
    if na != nb:
        f.append(f'{A.name} has {na} cameras, {B.name} {nb}.')
    f.append(f'They see {100 * iou:.0f} % of the same ground within {extent_m:g} m (IoU of the cells seen).')
    worst = max(matches, key=lambda m: m['cost']) if matches else None
    if worst:
        f.append(f'The least alike matched pair is {worst["a"]} ↔ {worst["b"]}: axes {worst["axis_angle_deg"]:.0f}° '
                 f'apart, {worst["centre_distance_m"]:.2f} m between centres, HFOV {worst["hfov_diff_deg"]:+.0f}°.')
    return RigGap(A.name, B.name, comp, W, total, band, matches, fig, f, (time.perf_counter() - t0) * 1000,
                  {'dirs': D, 'a': pa, 'b': pb, 'ground_a': ga, 'ground_b': gb, 'extent_m': extent_m})


# ---------------------------------------------------------------- output
SCHEMA = 'vantage.rig_gap/1'


def to_doc(g: RigGap, A: Rig | None = None, B: Rig | None = None) -> dict:
    doc = {'schema': SCHEMA, 'a': g.a, 'b': g.b, 'total': round(g.total, 4), 'transfer_band': g.transfer_band,
           'transfer_band_note': 'a heuristic band on the index (< 0.15 small, < 0.35 moderate, else large); not '
                                 'calibrated against any benchmark and not a predicted mAP',
           'components': {k: round(v, 4) for k, v in g.components.items()}, 'weights': g.weights,
           'figures': json.loads(json.dumps(g.figures, default=float)), 'matches': g.matches, 'findings': g.findings,
           'frame': 'each rig in its own vehicle frame: x forward, y left, z up, origin on the ground under the rear '
                    'axle (KITTI: the Velodyne lowered to the ground)', 'ms': round(g.ms, 1)}
    if A is not None:
        doc['rig_a'] = A.summary()
    if B is not None:
        doc['rig_b'] = B.summary()
    return doc


def figure(g: RigGap, size=(1400, 520)):
    """A BGR comparison image: each rig's pixels over viewing directions (equirectangular, log scale) and the ground
    each sees (A only red, B only blue, both grey)."""
    import cv2
    W, H = size
    out = np.full((H, W, 3), 24, np.uint8)
    D = g.maps['dirs']
    az = np.degrees(np.arctan2(D[:, 1], D[:, 0]))
    el = np.degrees(np.arcsin(np.clip(D[:, 2], -1, 1)))
    pw, ph = 420, 210

    def panel(vals, x0, y0, title):
        img = np.zeros((ph, pw), np.float32)
        cnt = np.zeros((ph, pw), np.float32)
        px = ((180 - az) / 360 * (pw - 1)).astype(int)
        py = ((90 - el) / 180 * (ph - 1)).astype(int)
        np.add.at(img, (py, px), np.log10(1 + vals))
        np.add.at(cnt, (py, px), 1)
        img = np.where(cnt > 0, img / np.maximum(cnt, 1), 0)
        img = cv2.dilate(img, np.ones((5, 5), np.uint8))
        hi = max(float(img.max()), 1e-6)
        col = cv2.applyColorMap((img / hi * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        out[y0:y0 + ph, x0:x0 + pw] = col
        cv2.putText(out, title, (x0, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
    panel(g.maps['a']['px_sr'], 20, 40, f'A: pixels by viewing direction (az +180..-180, el)')
    panel(g.maps['b']['px_sr'], 20, 40 + ph + 40, f'B: pixels by viewing direction')
    ga, gb = g.maps['ground_a'], g.maps['ground_b']
    n = ga.shape[0]
    bev = np.full((n, n, 3), 40, np.uint8)
    bev[ga & ~gb] = (60, 60, 220)
    bev[gb & ~ga] = (220, 120, 40)
    bev[ga & gb] = (170, 170, 170)
    # rows of the arrays are y, columns x: up the image is forward (+x), right is the vehicle's right (-y)
    bev = cv2.resize(np.ascontiguousarray(bev[::-1, ::-1].transpose(1, 0, 2)), (440, 440),
                     interpolation=cv2.INTER_NEAREST)
    out[40:480, 480:920] = bev
    cv2.putText(out, 'ground seen: A red, B blue, both grey', (480, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (230, 230, 230), 1, cv2.LINE_AA)
    y = 60
    for k, v in g.components.items():
        cv2.putText(out, f'{k:13s} {v:.3f}', (950, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
        y += 26
    cv2.putText(out, f'total {g.total:.3f} ({g.transfer_band})', (950, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


def export_rig_gap(A: Rig, B: Rig, out_dir, gap: RigGap | None = None, **kw) -> list:
    """rig_gap.json (to_doc) and rig_gap.png (figure) in out_dir; returns the paths."""
    import cv2
    g = gap or rig_gap(A, B, **kw)
    os.makedirs(out_dir, exist_ok=True)
    pj = os.path.join(out_dir, 'rig_gap.json')
    with open(pj, 'w', encoding='utf-8') as fh:
        json.dump(to_doc(g, A, B), fh, indent=2, allow_nan=False)
    pp = os.path.join(out_dir, 'rig_gap.png')
    cv2.imwrite(pp, figure(g))
    return [pj, pp]
