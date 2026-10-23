"""Lidar–camera extrinsic calibration with a checkerboard held up at several poses, seen by both sensors.

What it measures: T_camera_lidar, the rigid transform that maps lidar points into the camera's optical frame (OpenCV:
x right, y down, z forward), and through the camera's pose in the variant where the lidar sits in the vehicle, against
the plan. The installation check (core.install) measures where each camera sits from floor boards; this measures a
lidar from a board both sensors see.

Method
- Pair: a camera and a lidar whose fields overlap, ranked by the share of the camera image whose scene 5–10 m out lies
  in the lidar's field and in its sight (pairs()).
- Board: a printed checkerboard (inner corners cols × rows, square size) with a white margin of equal width all round:
  the camera finds its corners, the lidar sees its plane, its outer edge and (when its reflectivity shows them) its
  squares.
- Pose plan (plan_poses()): N boards in the pair's common field near_m–far_m from the camera, where both sensors see
  all of the board, at least MIN_RINGS lidar channels cross it (so the farthest pose shrinks with a coarse lidar) and
  its squares stay at least MIN_SQUARE_PX wide in the image; each turned (yaw ±30°, pitch ±20°, roll ±15°) and placed
  for what its plane adds to pinning the lidar's six degrees of freedom: the predicted 1σ of the worst axis, in units
  of the tolerances, from the lidar points expected on each board and the range noise (greedy, then exchanged).
- Camera, per pose: the chessboard corners in a rectified pinhole view (exact lens model) around where the board is
  expected, else found by searching the whole image and virtual pinhole views tiling it, then refined in tight
  rectified views (the view whose corners fit best is kept); the board's pose in the optical frame from the corners'
  bearing rays (PnP in a virtual pinhole frame, then least squares on the reprojection error in image pixels).
- Lidar, per pose: the points around where the current extrinsic puts the camera-measured board, in a crop that grows
  when nothing is found (with a poor first guess: the largest planar cluster of board size at the board's range);
  RANSAC plane, least-squares refit, the connected cluster of inliers.
- Board edge: wherever a board point's neighbouring beam (the next azimuth in its ring, the ring above or below)
  misses the board, the edge lies between the two beams: the point must lie inside the camera-measured outline and the
  neighbour's crossing of the plane outside (a bracket, not a guess at where in between: along an edge parallel to the
  scan lines every bracket has the same phase, and treating their middles as independent measurements would bias the
  answer by up to half a beam step). A rolled board's slanted edges cut the rings at many phases and pin the board
  in its plane like a vernier.
- Checker in the reflectivity: when a board's reflectivity is clearly two-tone with about the checker's share of dark,
  each clearly dark point must lie in a dark square and each bright one in a white square or the margin (the camera
  says which squares are dark): the square edges bracket the board in its plane, again at many phases.
- Solve: one robust least-squares problem (scipy least_squares, soft-L1) over T_camera_lidar and every board's pose:
  the camera's corner reprojection errors (pixels), the lidar board points' distances to the board plane, and the
  edge and checker brackets; then again with the checker from there (close enough not to slip a square). The board
  poses are refined with the extrinsic, so the camera's own uncertainty in each board enters the answer. 1σ: the
  inverse of JᵀJ (a-priori noise: corner RMS, plane RMS) scaled by the residual variance; brackets that hold add
  nothing to JᵀJ, so the 1σ is conservative. Start: the plan, or the closed-form plane-to-plane alignment when that
  fits better.
- Observability: the extrinsic's information from the board planes alone, the boards' poses marginalised out; a
  direction the planes do not pin within 10 × the tolerance is named in words ("the boards were all tilted the same
  way: the translation along y (left–right) is not determined"), one within 3–10 × is reported as resting on the edges
  and the reflectivity.
- A pose whose residuals sit far above the others' (a board moved between the two captures, frames from different
  moments) is set aside and the solve repeated.

Synthetic capture (simulate()), for planning, demos and tests: the camera image through the real lens model
(per-pixel rays, ray–plane intersection, antialiased checker and margin over a plain floor and wall or the parking
scene rendered by core.synth, image noise); the lidar scan with the model's beams (core.sensors) against the board,
the floor, the scene and the own vehicle, with Gaussian range noise. The lidar can carry an injected mounting error
(Δx Δy Δz in the vehicle frame, Δyaw Δpitch Δroll added to its angles) so the solve's recovery shows.

Real data (load_folder()): image + point-cloud pairs matched by file stem (pose01.png + pose01.pcd); clouds as PCD
(ascii, binary, binary_compressed), KITTI .bin (float32 × 4), nuScenes .pcd.bin (× 5), PLY (ascii, binary), .npy and
CSV/TXT, in the lidar's own frame.
"""
from __future__ import annotations

import copy
import io
import json
import math
import os
import re
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from . import sensors as S
from .lens import resolve, dr_dtheta
from .mathutil import BODY_TO_OPTICAL, euler_zyx, quat_from_matrix, rodrigues, rot_x, rot_y, rot_z, rot_zyx, wrap180

SCHEMA = 'vantage.lidar_camera_calibration/1'
FRAME = 'vehicle (ISO 8855, rear axle)'
OPTICAL = 'camera optical frame (OpenCV: x right, y down, z forward)'
ASSUMPTIONS = (
    'The camera\'s lens model (its unit calibration when it has one) is taken as exact and the camera stays where '
    'the variant puts it: the lidar is measured relative to that camera. Lidar points are taken from the lidar\'s '
    'origin at one instant (no motion during the sweep, no beam-origin offset). The board is flat and rigid with an equal margin '
    'all round; its stand is not modelled. A real lidar returns the board for a beam that only partly hits it, so the '
    'board can look a few millimetres larger, and some lidars range dark and bright surfaces differently: the edge and '
    'checker brackets allow 4 mm, and the observability check relies on the board planes alone. The checker term needs '
    'a print whose dark squares are dark at the lidar\'s wavelength (matte black toner usually is; some inks are not). '
    'Synthetic captures show what the plan can achieve with ideal hardware, not a measurement.')
SCENES = ('plain', 'parking')
SOURCES = ('synthetic', 'folder')
BIN_FIELDS = ('auto', '4', '5')
SEVERITIES = ('critical', 'serious', 'warning', 'good')
VERDICT_SEVERITY = {'good': 'good', 'warning': 'warning', 'serious': 'serious', 'unobservable': 'critical',
                    'failed': 'critical'}
MIN_RINGS = 4                  # lidar channels that must cross a planned board
MIN_SQUARE_PX = 8.0            # a square's width in the image at the farthest planned pose
MIN_BOARD_POINTS = 20
EDGE_FLOOR_M = 0.004           # slack of the edge and checker brackets (beam divergence, range noise)
OWN_MOUNT_M = 0.05             # lidar: occluder hits this close to its optical centre are its own mount
NEAR_CLIP = 0.05               # camera: body surface this close is the lens housing
BOARD_MIN_Z, BOARD_MAX_Z = 0.2, 2.6    # a held or tripod-mounted board: bottom and top edge heights, m
MAX_POINTS_PER_POSE = 2500     # lidar board points per pose in the solve (a random subset beyond)
# the tilts a planned board may take: yaw (about its vertical), pitch (top away +), roll (in its plane), degrees
TILT_SET = tuple((y, p, r) for y in (-30.0, 0.0, 30.0) for p in (-20.0, 0.0, 20.0) for r in (-15.0, 0.0, 15.0))
MIN_BOARD_GAP_M = 0.7          # planned boards stand at least this far apart
PLAN_CAMERA_MM, PLAN_CAMERA_DEG = 1.0, 0.03     # the plan's allowance for the camera's measurement of a board plane
OUTLIER_COST = 3.0             # a pose whose mean whitened residual² is this far above the others' is set aside
REFLECT = {'board_dark': 6.0, 'board_white': 80.0, 'floor': 12.0, 'wall': 35.0, 'box': 25.0, 'cylinder': 30.0,
           'body': 40.0}                            # reflectivity, %, of the synthetic scan's surfaces
INT_FIELDS = ('cols', 'rows', 'poses', 'seed')
LIMITS = {'cols': (2, 30), 'rows': (2, 30), 'square_mm': (10.0, 500.0), 'margin_mm': (0.0, 500.0), 'poses': (3, 20),
          'near_m': (1.0, 60.0), 'far_m': (1.0, 80.0), 'range_sigma_mm': (0.0, 200.0), 'image_noise': (0.0, 40.0),
          'seed': (0, 999999), 'err_x_mm': (-500.0, 500.0), 'err_y_mm': (-500.0, 500.0),
          'err_z_mm': (-500.0, 500.0), 'err_yaw_deg': (-20.0, 20.0), 'err_pitch_deg': (-20.0, 20.0),
          'err_roll_deg': (-20.0, 20.0), 'pos_tol_mm': (0.1, 1000.0), 'ang_tol_deg': (0.01, 30.0)}
ERR_KEYS = ('err_x_mm', 'err_y_mm', 'err_z_mm', 'err_yaw_deg', 'err_pitch_deg', 'err_roll_deg')


# =========================================================================== settings
@dataclass
class LidarCalibSettings:
    camera: str = ''               # camera tag ('' = the best pair)
    lidar: str = ''                # lidar tag ('' = the best partner)
    cols: int = 7                  # inner corners along the board's width
    rows: int = 5                  # inner corners along its height
    square_mm: float = 100.0
    margin_mm: float = 50.0        # white border round the checker
    poses: int = 6
    near_m: float = 3.0            # board distance from the camera
    far_m: float = 8.0
    source: str = 'synthetic'      # synthetic | folder
    folder: str = ''               # real data: image + cloud pairs
    bin_fields: str = 'auto'       # a .bin cloud: auto (.pcd.bin = 5, else by size) | 4 (KITTI) | 5 (nuScenes)
    scene: str = 'plain'           # synthetic background: plain (floor and wall) | parking
    range_sigma_mm: float = 15.0   # synthetic lidar range noise, 1σ
    image_noise: float = 2.0       # synthetic image noise, grey levels 1σ
    seed: int = 1
    err_x_mm: float = 0.0          # injected lidar mounting error (synthetic): vehicle frame
    err_y_mm: float = 0.0
    err_z_mm: float = 0.0
    err_yaw_deg: float = 0.0       # added to the lidar's yaw, pitch and roll
    err_pitch_deg: float = 0.0
    err_roll_deg: float = 0.0
    edges: bool = True             # use the board-boundary term
    reflectivity: bool = True      # use the checker squares in the lidar's reflectivity, when it shows them
    pos_tol_mm: float = 10.0
    ang_tol_deg: float = 0.3

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d: dict | None) -> 'LidarCalibSettings':
        s = LidarCalibSettings()
        d = d if isinstance(d, dict) else {}
        for k in ('camera', 'lidar'):
            if isinstance(d.get(k), str):
                setattr(s, k, d[k].strip()[:32])
        if isinstance(d.get('folder'), str):
            s.folder = d['folder'].strip()
        for k, opts in (('source', SOURCES), ('scene', SCENES), ('bin_fields', BIN_FIELDS)):
            v = str(d.get(k, '')).strip().lower()
            if v in opts:
                setattr(s, k, v)
        for k in ('edges', 'reflectivity'):
            v = d.get(k)
            if isinstance(v, (bool, int)):
                setattr(s, k, bool(v))
            elif isinstance(v, str):
                setattr(s, k, v.strip().lower() in ('1', 'true', 'yes', 'on'))
        for k, (lo, hi) in LIMITS.items():
            if k in d:
                try:
                    v = float(d[k])
                except (TypeError, ValueError):
                    continue
                if math.isfinite(v):
                    v = min(max(v, lo), hi)
                    setattr(s, k, int(round(v)) if k in INT_FIELDS else v)
        if s.far_m < s.near_m:
            s.near_m, s.far_m = s.far_m, s.near_m
        return s

    def board_size(self) -> tuple[float, float]:
        """Outer width and height of the board, metres (checker squares plus the margin both sides)."""
        sq, mg = self.square_mm / 1000.0, self.margin_mm / 1000.0
        return (self.cols + 1) * sq + 2 * mg, (self.rows + 1) * sq + 2 * mg

    def error(self) -> np.ndarray:
        return np.array([getattr(self, k) for k in ERR_KEYS], float)


# =========================================================================== geometry helpers
def _exp(w) -> np.ndarray:
    """Rotation vector -> matrix."""
    w = np.asarray(w, float)
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def _T(R, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _inv(T) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    return _T(R.T, -R.T @ t)


def _rot_deg(R) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, (float(np.trace(R)) - 1) / 2))))


def _unit(v):
    v = np.asarray(v, float)
    return v / max(float(np.linalg.norm(v)), 1e-12)


def _w2o(cam) -> np.ndarray:
    """World -> optical rotation of a camera."""
    return BODY_TO_OPTICAL @ rot_zyx(cam.yaw, cam.pitch, cam.roll).T


def plan_T(cam, lidar) -> np.ndarray:
    """T_camera_lidar (4×4, optical frame) from the variant's poses of the camera and the lidar."""
    W2O = _w2o(cam)
    return _T(W2O @ S.rotation(lidar), W2O @ (lidar.pos() - cam.pos()))


def lidar_pose(cam, T) -> tuple[np.ndarray, np.ndarray]:
    """(R lidar -> vehicle, position) of the lidar from the camera's pose and T_camera_lidar."""
    W2O = _w2o(cam)
    return W2O.T @ T[:3, :3], cam.pos() + W2O.T @ T[:3, 3]


def pose_dict(R, pos) -> dict:
    y, p, r = euler_zyx(np.asarray(R, float))
    return {'x': float(pos[0]), 'y': float(pos[1]), 'z': float(pos[2]), 'yaw': y, 'pitch': p, 'roll': r}


def model_corners(st: LidarCalibSettings) -> np.ndarray:
    """Board-frame inner corners (rows*cols, 3), row-major; the board frame has its origin at the board's centre,
    x along its width, y along its height, z its normal."""
    sq = st.square_mm / 1000.0
    a = (np.arange(st.cols) - (st.cols - 1) / 2) * sq
    b = (np.arange(st.rows) - (st.rows - 1) / 2) * sq
    A, B = np.meshgrid(a, b)
    return np.stack([A.ravel(), B.ravel(), np.zeros(A.size)], 1)


def outline_local(W, H, per_edge=1) -> np.ndarray:
    """Points round the board's outer edge in the board frame: the corners and per_edge points along each edge."""
    c = np.array([[-W / 2, -H / 2], [W / 2, -H / 2], [W / 2, H / 2], [-W / 2, H / 2]])
    out = []
    for i in range(4):
        a, b = c[i], c[(i + 1) % 4]
        for k in range(per_edge + 1):
            out.append(a + (b - a) * k / (per_edge + 1))
    return np.c_[np.array(out), np.zeros(len(out))]


def _box_sdf(a, b, hw, hh):
    """Signed distance of in-plane points (a, b) to the rectangle |a| <= hw, |b| <= hh (positive outside)."""
    dx = np.abs(a) - hw
    dy = np.abs(b) - hh
    return np.hypot(np.maximum(dx, 0), np.maximum(dy, 0)) + np.minimum(np.maximum(dx, dy), 0)


@dataclass
class BoardPose:
    """Where a board stands: its centre and its axes in the vehicle frame (columns: width u, height v, normal n
    towards the sensors)."""
    centre: np.ndarray
    R: np.ndarray
    tilt: tuple = (0.0, 0.0, 0.0)          # yaw, pitch, roll from facing the sensors, degrees
    distance: float = 0.0                  # from the camera, m
    rings: int = 0                         # lidar channels crossing it (plan)

    def to_dict(self):
        return {'centre_m': [round(float(v), 4) for v in self.centre],
                'normal': [round(float(v), 5) for v in self.R[:, 2]],
                'width_axis': [round(float(v), 5) for v in self.R[:, 0]],
                'tilt_deg': {'yaw': self.tilt[0], 'pitch': self.tilt[1], 'roll': self.tilt[2]},
                'distance_from_camera_m': round(float(self.distance), 3), 'lidar_rings': int(self.rings)}


def _facing(centre, target, tilt=(0.0, 0.0, 0.0)) -> np.ndarray:
    """Board axes standing upright at centre, its normal level and pointing at target, then turned by tilt."""
    d = np.asarray(target, float) - np.asarray(centre, float)
    n0 = np.array([d[0], d[1], 0.0])
    n0 = _unit(n0) if np.linalg.norm(n0) > 1e-6 else np.array([1.0, 0, 0])
    v0 = np.array([0.0, 0.0, 1.0])
    u0 = np.cross(v0, n0)
    R0 = np.stack([u0, v0, n0], 1)
    y, p, r = (math.radians(a) for a in tilt)
    return R0 @ rot_y(y) @ rot_x(p) @ rot_z(r)


# =========================================================================== pairs
def _variant_project(project, variant):
    """The project with variant active (a shallow copy when it is not the active one)."""
    if variant is None or variant is project.variant:
        return project
    q = copy.copy(project)
    q.variants = list(project.variants) + ([] if any(v is variant for v in project.variants) else [variant])
    q.active_uid = variant.uid
    return q


def _cubes(project, engine):
    vg = engine.vehicle_geom(project) if engine is not None else None
    return (lambda pos: engine.cube(np.asarray(pos, float), vg)) if vg is not None else None


def pairs(project, variant=None, engine=None) -> list[dict]:
    """Every visible camera with every visible lidar, best first: share = the part of the camera's image (within its
    lens circle) whose scene point 5, 7.5 and 10 m out (on the ground where that is nearer) lies in the lidar's field
    and range, with the own vehicle in the way of neither (when an engine is given)."""
    v = variant or project.variant
    cams = [c for c in v.cameras if c.visible]
    lids = [s for s in v.sensors if s.visible and s.kind == 'lidar']
    cube = _cubes(_variant_project(project, v), engine)
    out = []
    nu, nv = 32, 24
    for c in cams:
        intr = resolve(project.module_for(c), project.lens_for(c))
        U, V = np.meshgrid((np.arange(nu) + 0.5) * intr.width / nu, (np.arange(nv) + 0.5) * intr.height / nv)
        d_opt, th = intr.unproject(U.ravel(), V.ravel())
        dw = d_opt @ _w2o(c)
        ok = th <= intr.max_theta
        C = c.pos()
        with np.errstate(divide='ignore', invalid='ignore'):
            tg = np.where(dw[:, 2] < -1e-9, -C[2] / dw[:, 2], np.inf)
        depth = cube(C).lookup(dw)[0] if cube is not None else np.full(len(dw), np.inf)
        for s in lids:
            m = project.sensor_model_for(s)
            cl = cube(s.pos()) if cube is not None else None
            hit = tot = 0
            for D in (5.0, 7.5, 10.0):
                t = np.minimum(D, tg)
                vis = ok & ((depth > t - 0.02) | (depth <= NEAR_CLIP))
                P = C + dw * t[:, None]
                inn = vis & S.in_field(s, m, P)
                if cl is not None and inn.any():
                    q = P[inn] - s.pos()
                    dist = np.linalg.norm(q, axis=1)
                    dl = cl.lookup(q)[0]
                    inn[inn] = (dl > dist - (0.02 + 0.01 * dist)) | (dl <= OWN_MOUNT_M)
                hit += int(inn.sum())
                tot += int(ok.sum())
            out.append({'camera': c.tag, 'camera_uid': c.uid, 'lidar': s.tag, 'lidar_uid': s.uid,
                        'share': hit / tot if tot else 0.0})
    out.sort(key=lambda r: -r['share'])
    return out


def pick_pair(project, variant=None, st: LidarCalibSettings | None = None, engine=None):
    """(camera, lidar, share) to calibrate: the tags in the settings, else the best-overlapping pair. Raises
    ValueError with what is missing."""
    st = st or LidarCalibSettings()
    v = variant or project.variant
    lids = [s for s in v.sensors if s.kind == 'lidar']
    if not v.cameras:
        raise ValueError(f'{v.name} has no camera.')
    if not lids:
        raise ValueError(f'{v.name} has no lidar: add one with Vehicle ▸ Add lidar.')
    cam = lid = None
    if st.camera:
        cam = next((c for c in v.cameras if c.tag.lower() == st.camera.lower()), None)
        if cam is None:
            raise ValueError(f'No camera tagged {st.camera} in {v.name}.')
    if st.lidar:
        lid = next((s for s in lids if s.tag.lower() == st.lidar.lower()), None)
        if lid is None:
            raise ValueError(f'No lidar tagged {st.lidar} in {v.name}.')
    ranked = pairs(project, v, engine)
    for r in ranked:
        if (cam is None or r['camera_uid'] == cam.uid) and (lid is None or r['lidar_uid'] == lid.uid):
            return v.cam(r['camera_uid']), v.sensor(r['lidar_uid']), r['share']
    cam = cam or next((c for c in v.cameras if c.visible), v.cameras[0])
    lid = lid or next((s for s in lids if s.visible), lids[0])
    return cam, lid, 0.0


# =========================================================================== the pose plan
def _lidar_frame_angles(L, RL, P):
    """(range, azimuth deg, elevation deg) of points (..., 3) in the lidar's frame."""
    q = (P - L) @ RL
    r = np.linalg.norm(q, axis=-1)
    az = np.degrees(np.arctan2(q[..., 1], q[..., 0]))
    el = np.degrees(np.arcsin(np.clip(q[..., 2] / np.maximum(r, 1e-12), -1, 1)))
    return r, az, el


def _check_poses(ctx, centres, Rs):
    """Validity and margin (0..1) of boards at centres (K,3) with axes Rs (K,3,3); also the rings crossing each."""
    W, H = ctx['W'], ctx['H']
    loc = outline_local(W, H, 1)                                   # (8, 3)
    P = centres[:, None, :] + np.einsum('kij,nj->kni', Rs, loc)     # (K, 8, 3)
    n = Rs[:, :, 2]
    C, L, RL, intr, W2O, m = ctx['C'], ctx['L'], ctx['RL'], ctx['intr'], ctx['W2O'], ctx['m']
    # camera: every outline point in the image with a margin, the board facing it
    d = (P - C) @ W2O.T
    u, v, th, ok = intr.project_dirs(d.reshape(-1, 3))
    mx, my = 0.03 * intr.width, 0.03 * intr.height
    inimg = ok & (u >= mx) & (u <= intr.width - mx) & (v >= my) & (v <= intr.height - my) & \
        (th <= intr.max_theta - math.radians(3))
    cam_ok = inimg.reshape(P.shape[:2]).all(1)
    cm = np.minimum(np.minimum(u - mx, intr.width - mx - u) / intr.width,
                    np.minimum(v - my, intr.height - my - v) / intr.height).reshape(P.shape[:2]).min(1)
    tc = C - centres
    cam_ok &= np.sum(n * tc, 1) / np.linalg.norm(tc, axis=1) >= math.cos(math.radians(70))
    # lidar: in its field and range, facing it, enough channels across
    r, az, el = _lidar_frame_angles(L, RL, P)
    lo, hi = m.v_fov()
    pad = max(0.25 * m.v_step(), 0.2)
    lid_ok = ((el >= lo + pad) & (el <= hi - pad) & (r >= m.range_min_m + 0.3) & (r <= 0.8 * m.range_m)).all(1)
    if m.az_half() < 180.0:
        lid_ok &= (np.abs(az) <= m.az_half() - 1.0).all(1)
    tl = L - centres
    lid_ok &= np.sum(n * tl, 1) / np.linalg.norm(tl, axis=1) >= math.cos(math.radians(70))
    lm = np.minimum(el.min(1) - lo, hi - el.max(1)) / max(hi - lo, 1e-6)
    # channels between the bottom and the top edge's midpoints (seen from the lidar)
    mid = centres[:, None, :] + np.einsum('kij,nj->kni', Rs, np.array([[0, -H / 2, 0], [0, H / 2, 0.0]]))
    _, _, em = _lidar_frame_angles(L, RL, mid)
    E = m.elevations()
    rings = (np.searchsorted(E, em.max(1), 'right') - np.searchsorted(E, em.min(1), 'left')).astype(int)
    need = MIN_RINGS if m.scan != 'nonrepetitive' else 1
    ok_all = cam_ok & lid_ok & (rings >= need)
    z = P[:, :, 2]
    ok_all &= (z.min(1) >= BOARD_MIN_Z) & (z.max(1) <= BOARD_MAX_Z)
    # nothing in the way: the vehicle (occlusion cubes) and the scene's objects
    if ok_all.any() and (ctx['cube_c'] is not None or ctx['world'] is not None):
        idx = np.flatnonzero(ok_all)
        Q = np.concatenate([P[idx], centres[idx, None, :]], 1).reshape(-1, 3)
        clear = np.ones(len(Q), bool)
        for o, cube, near in ((C, ctx['cube_c'], NEAR_CLIP), (L, ctx['cube_l'], OWN_MOUNT_M)):
            dv = Q - o
            dist = np.linalg.norm(dv, axis=1)
            dirs = dv / dist[:, None]
            if cube is not None:
                dep = cube.lookup(dirs)[0]
                clear &= (dep > dist - 0.03) | (dep <= near)
            if ctx['world'] is not None:
                st, _ = _scene_hits(o, dirs, ctx['world'], None, near, floor=False)
                clear &= st > dist - 0.03
        ok_all[idx] = clear.reshape(len(idx), -1).all(1)
    return ok_all, np.clip(np.minimum(cm, lm) * 4, 0, 1), rings


def _facing_many(centres, target, tilt) -> np.ndarray:
    """_facing for many centres at once: (K, 3, 3)."""
    d = np.asarray(target, float) - centres
    n0 = np.c_[d[:, 0], d[:, 1], np.zeros(len(d))]
    ln = np.linalg.norm(n0, axis=1, keepdims=True)
    n0 = np.where(ln > 1e-6, n0 / np.maximum(ln, 1e-12), [1.0, 0.0, 0.0])
    v0 = np.tile([0.0, 0.0, 1.0], (len(d), 1))
    u0 = np.cross(v0, n0)
    y, p, r = (math.radians(a) for a in tilt)
    return np.stack([u0, v0, n0], 2) @ (rot_y(y) @ rot_x(p) @ rot_z(r))


def plane_information(centres, Rs, L, W, H, m, range_sigma):
    """Predicted information (K, 6, 6) about the lidar's pose, rotation (rad, about its origin) then position (m) in the
    vehicle frame, from each board plane alone: the lidar points expected on it (its solid angle over the beam grid),
    their range noise along the normal and their spread over the board give the plane's offset and two tilts as the
    lidar measures them; a floor (PLAN_CAMERA_MM, PLAN_CAMERA_DEG) stands for the camera's own measurement of the
    board."""
    n = Rs[:, :, 2]
    rel = centres - L
    r = np.linalg.norm(rel, axis=1)
    cosi = np.abs(np.sum(n * rel, 1)) / np.maximum(r, 1e-9)
    cell = math.radians(max(m.h_res, 1e-3)) * math.radians(max(m.v_step(), 1e-3) if m.n_channels() > 1 else 1.0)
    N = np.maximum(W * H * cosi / np.maximum(r, 0.1) ** 2 / cell, 3.0)
    s2 = (max(range_sigma, 0.002) * np.maximum(cosi, 0.2)) ** 2
    K = len(centres)
    Sg = np.zeros((K, 3, 3))
    Sg[:, 0, 0] = s2 / N + (PLAN_CAMERA_MM / 1000.0) ** 2
    Sg[:, 1, 1] = s2 / (N * W * W / 12) + math.radians(PLAN_CAMERA_DEG) ** 2
    Sg[:, 2, 2] = s2 / (N * H * H / 12) + math.radians(PLAN_CAMERA_DEG) ** 2
    J = np.zeros((K, 3, 6))
    J[:, 0, :3] = np.cross(rel, n)
    J[:, 0, 3:] = n
    J[:, 1, :3] = Rs[:, :, 1]
    J[:, 2, :3] = -Rs[:, :, 0]
    return np.einsum('kai,kab,kbj->kij', J, np.linalg.inv(Sg), J)


def predicted_sigma(project, cam, lidar, poses, st: LidarCalibSettings) -> dict:
    """1σ of the lidar's pose the board planes of a plan would give (x y z mm, roll pitch yaw deg, vehicle frame):
    a prediction from the expected point counts and the range noise (plane_information), not a measurement; the
    edges and the checker in the reflectivity improve on it."""
    if not poses:
        return {}
    m = project.sensor_model_for(lidar)
    W, H = st.board_size()
    I = plane_information(np.array([b.centre for b in poses]), np.array([b.R for b in poses]),
                          lidar.pos().astype(float), W, H, m, st.range_sigma_mm / 1000.0).sum(0)
    cov = np.linalg.pinv(I + np.eye(6) * 1e-12)
    sd = np.sqrt(np.maximum(np.diag(cov), 0))
    return {'x_mm': sd[3] * 1e3, 'y_mm': sd[4] * 1e3, 'z_mm': sd[5] * 1e3, 'roll_deg': math.degrees(sd[0]),
            'pitch_deg': math.degrees(sd[1]), 'yaw_deg': math.degrees(sd[2])}


def plan_poses(project, cam, lidar, st: LidarCalibSettings | None = None, engine=None, world=None):
    """([BoardPose], notes): st.poses boards both sensors see whole, near_m–far_m from the camera, chosen one at a time
    from every place and tilt (TILT_SET) for what their planes add to pinning the lidar's six degrees of freedom: the
    predicted 1σ of its worst axis, in units of the tolerances (plane_information). Raises ValueError when no board
    fits anywhere."""
    st = st or LidarCalibSettings()
    m = project.sensor_model_for(lidar)
    intr = resolve(project.module_for(cam), project.lens_for(cam))
    W2O = _w2o(cam)
    C, L, RL = cam.pos().astype(float), lidar.pos().astype(float), S.rotation(lidar)
    W, H = st.board_size()
    notes = []
    near, far = float(st.near_m), float(st.far_m)
    # the squares must stay big enough to find in the image: their size at the farthest pose (lens centre focal)
    far_sq = (st.square_mm / 1000.0) * intr.f_px / MIN_SQUARE_PX
    if far_sq < far:
        notes.append(f'Squares of {st.square_mm:g} mm are smaller than {MIN_SQUARE_PX:g} px in {cam.tag}\'s image '
                     f'beyond {far_sq:.1f} m: the farthest pose is kept nearer.')
        far = max(far_sq, near)
    cube = _cubes(project, engine)
    ctx = {'W': W, 'H': H, 'C': C, 'L': L, 'RL': RL, 'intr': intr, 'W2O': W2O, 'm': m, 'world': world,
           'cube_c': cube(C) if cube else None, 'cube_l': cube(L) if cube else None}
    zc = W2O[2]                                   # the optical axis in the vehicle frame
    head = math.atan2(zc[1], zc[0]) if math.hypot(zc[0], zc[1]) > 0.2 else math.radians(cam.yaw)
    alphas = np.radians(np.arange(-80.0, 80.1, 5.0))
    dists = np.linspace(near, far, 8)
    hs = np.arange(0.55, 2.21, 0.15)
    A, D, Z = (g.ravel() for g in np.meshgrid(alphas, dists, hs, indexing='ij'))
    centres = np.stack([C[0] + D * np.cos(head + A), C[1] + D * np.sin(head + A), Z], 1)
    if world is not None:
        keep = np.hypot(centres[:, 0] - world.centre[0], centres[:, 1] - world.centre[1]) < world.wall_r - 0.6
        centres, D = centres[keep], D[keep]
    mid = (C + L) / 2
    cand_c, cand_R, cand_t, cand_rings = [], [], [], []
    for tilt in TILT_SET:
        Rs = _facing_many(centres, mid, tilt)
        ok, _, rings = _check_poses(ctx, centres, Rs) if len(centres) else (np.zeros(0, bool), None, None)
        if ok.any():
            cand_c.append(centres[ok])
            cand_R.append(Rs[ok])
            cand_t += [tilt] * int(ok.sum())
            cand_rings.append(rings[ok])
    if not cand_c:
        raise ValueError(f'{cam.tag} and {lidar.tag} share no field where a {W:.2f} × {H:.2f} m board fits whole '
                         f'{near:g}–{far:g} m from {cam.tag}, {BOARD_MIN_Z:g}–{BOARD_MAX_Z:g} m above the ground, '
                         f'with {MIN_RINGS} or more of {lidar.tag}\'s channels across it.')
    cc, cR, cr = np.concatenate(cand_c), np.concatenate(cand_R), np.concatenate(cand_rings)
    dist = np.linalg.norm(cc - C, axis=1)
    if dist.max() < far - 0.5 and far > near:
        notes.append(f'No board fits beyond {dist.max():.1f} m (the common field, {lidar.tag}\'s channels or the '
                     'scene): the poses stay nearer.')
    # the information each candidate adds, in units of the tolerances (rotation, position)
    info = plane_information(cc, cR, L, W, H, m, st.range_sigma_mm / 1000.0)
    sc = np.r_[[math.radians(st.ang_tol_deg)] * 3, [st.pos_tol_mm / 1000.0] * 3]
    info = info * sc[None, :, None] * sc[None, None, :]
    prior = np.eye(6) * 1e-6                     # a weak prior keeps the first boards' sums invertible

    def pick(base, free, full):
        """The free candidate that best completes base: (criterion, index)."""
        idx = np.flatnonzero(free)
        best = (np.inf, -1)
        for s0 in range(0, len(idx), 4096):
            part = idx[s0:s0 + 4096]
            dg = np.diagonal(np.linalg.inv(base[None] + info[part]), axis1=1, axis2=2)
            # until the set can pin every direction: as much in every direction as it gives; then the worst axis
            crit = dg.max(1) + 1e-3 * dg.sum(1) if full else np.sum(np.log(dg), 1)
            j = int(np.argmin(crit))
            if crit[j] < best[0]:
                best = (float(crit[j]), int(part[j]))
        return best

    def apart(sel, skip=-1):
        free = np.ones(len(cc), bool)
        for i, j in enumerate(sel):
            if i != skip:
                free &= np.linalg.norm(cc - cc[j], axis=1) >= MIN_BOARD_GAP_M
        return free
    sel: list[int] = []
    for k in range(int(st.poses)):
        free = apart(sel)
        if not free.any():
            notes.append(f'Pose {k + 1}: no place left that is not next to another board.')
            break
        sel.append(pick(prior + info[sel].sum(0), free, k >= 3)[1])
    # exchange: each board in turn swapped for the candidate that best completes the others
    if len(sel) >= 4:
        for _ in range(2):
            changed = False
            for i in range(len(sel)):
                rest = [j for k, j in enumerate(sel) if k != i]
                base = prior + info[rest].sum(0)
                cur = float(np.diagonal(np.linalg.inv(base + info[sel[i]])).max())
                crit, j = pick(base, apart(sel, i), True)
                if j >= 0 and j != sel[i] and crit < cur * 0.999:
                    sel[i], changed = j, True
            if not changed:
                break
    chosen = [BoardPose(cc[j].copy(), cR[j].copy(), cand_t[j], float(dist[j]), int(cr[j])) for j in sel]
    chosen.sort(key=lambda b: (b.distance, b.centre[1]))
    return chosen, notes


# =========================================================================== the synthetic scene
def make_world(project, scene='plain', reach=8.0):
    """The synthetic surroundings: the parking scene of core.synth, or (plain) a floor and a wall all round far
    enough out for boards `reach` metres from the cameras."""
    from . import synth
    if scene == 'parking':
        return synth.parking_world(project)
    meas = project.vehicle.measured() if project.vehicle.mesh is not None else {}
    cx = (meas.get('front_x', 3.2) + meas.get('rear_x', -0.8)) / 2
    half = max(meas.get('Length', 4.0) / 2, 1.0)
    return synth.World(centre=(cx, 0.0), markings=[], boxes=[], cylinders=[], wall_r=half + reach + 5.0, wall_h=4.0)


def _scene_hits(o, d, world, cube=None, near=OWN_MOUNT_M, floor=True):
    """(distance, reflectivity %) of rays from o along unit d (K,3) to the nearest surface of the scene: the floor, the
    wall all round, the scene's boxes and cylinders and the own vehicle (cube); inf where nothing."""
    from .synth import _ray_box, _ray_cyl
    o = np.asarray(o, float)
    K = len(d)
    best = np.full(K, np.inf)
    refl = np.zeros(K)

    def take(t, r):
        sel = t < best
        best[sel] = t[sel]
        refl[sel] = r
    with np.errstate(divide='ignore', invalid='ignore'):
        if floor:
            take(np.where(d[:, 2] < -1e-9, -o[2] / d[:, 2], np.inf), REFLECT['floor'])
        if world is not None:
            ox, oy = o[0] - world.centre[0], o[1] - world.centre[1]
            a = d[:, 0] ** 2 + d[:, 1] ** 2
            b = 2 * (ox * d[:, 0] + oy * d[:, 1])
            c = ox * ox + oy * oy - world.wall_r ** 2
            tw = (-b + np.sqrt(np.maximum(b * b - 4 * a * c, 0))) / (2 * np.maximum(a, 1e-12))
            zw = o[2] + tw * d[:, 2]
            take(np.where((tw > 0) & (zw >= 0) & (zw <= world.wall_h) & (a > 1e-9), tw, np.inf), REFLECT['wall'])
            for bx in world.boxes:
                take(_ray_box(o, d, bx)[0], REFLECT['box'])
            for x, y, r, h, _, _ in world.cylinders:
                take(_ray_cyl(o, d, x, y, r, h)[0], REFLECT['cylinder'])
    if cube is not None:
        dep = cube.lookup(d)[0].astype(float)
        take(np.where((dep > near) & np.isfinite(dep), dep, np.inf), REFLECT['body'])
    return best, refl


def _checker_value(a, b, st, fw):
    """Board brightness 0 (black) .. 1 (white) at board-frame points (a, b); fw = footprint (m) for antialiasing."""
    sq = st.square_mm / 1000.0
    nc, nr = st.cols + 1, st.rows + 1
    u = (a + nc * sq / 2) / sq
    v = (b + nr * sq / 2) / sq
    f = np.maximum(fw / sq, 1e-4)
    du = np.abs(u - np.round(u))
    dv = np.abs(v - np.round(v))
    sign = np.where((np.floor(u) + np.floor(v)) % 2 == 0, -1.0, 1.0)      # a dark square at the (0, 0) corner
    val = 0.5 + 0.5 * sign * np.clip(np.minimum(du, dv) / f, 0, 1)
    inside = (u >= 0) & (u <= nc) & (v >= 0) & (v <= nr)
    border = np.clip(np.minimum(np.minimum(u, nc - u), np.minimum(v, nr - v)) / f + 0.5, 0, 1)
    return np.where(inside, val * border + (1 - border), 1.0)


def _paint_board(img, intr, W2O, C, bp: BoardPose, st, depth_fn):
    """Composite one board into img (h, w, 3) float, in place: ray–plane intersection per pixel, antialiased checker
    and outer edge, hidden where the scene or the vehicle is nearer."""
    from .synth import pixel_rays
    h, w = img.shape[:2]
    sx, sy = w / intr.width, h / intr.height
    W, H = st.board_size()
    P = bp.centre + outline_local(W, H, 6) @ bp.R.T
    u, v, _, ok = intr.project_dirs((P - C) @ W2O.T)
    if ok.all():
        x0, x1 = max(int(u.min() * sx) - 4, 0), min(int(u.max() * sx) + 5, w)
        y0, y1 = max(int(v.min() * sy) - 4, 0), min(int(v.max() * sy) + 5, h)
    else:
        x0, x1, y0, y1 = 0, w, 0, h
    if x1 <= x0 or y1 <= y0:
        return
    d_all, th_all = pixel_rays(intr, w, h)
    d_opt = d_all.reshape(h, w, 3)[y0:y1, x0:x1].reshape(-1, 3)
    th = th_all.reshape(h, w)[y0:y1, x0:x1].ravel()
    dw = d_opt @ W2O
    n = bp.R[:, 2]
    den = dw @ n
    with np.errstate(divide='ignore', invalid='ignore'):
        t = np.where(np.abs(den) > 1e-9, float((bp.centre - C) @ n) / den, -1.0)
    X = C + dw * np.maximum(t, 0)[:, None]
    a = (X - bp.centre) @ bp.R[:, 0]
    b = (X - bp.centre) @ bp.R[:, 1]
    pa = intr.pixel_mm / np.maximum(dr_dtheta(np.maximum(th, 1e-4), intr.f_mm, intr.projection, intr.kb), 1e-9) * \
        (intr.width / w)
    fw = np.maximum(t, 0) * pa / np.maximum(np.abs(den), 0.15)
    cov = np.clip(np.minimum(W / 2 - np.abs(a), H / 2 - np.abs(b)) / np.maximum(fw, 1e-6) + 0.5, 0, 1)
    sel = (t > 0) & (th <= intr.max_theta) & (cov > 0)
    if not sel.any():
        return
    idx = np.flatnonzero(sel)
    vis = t[idx] < depth_fn(dw[idx]) - 1e-3
    idx = idx[vis]
    if not len(idx):
        return
    val = _checker_value(a[idx], b[idx], st, fw[idx])
    val = np.where(den[idx] < 0, val, 0.45)                       # the back of the board is plain
    sun = _unit([0.32, 0.42, 0.85])
    grey = (0.07 + 0.82 * val) * (0.84 + 0.16 * abs(float(n @ sun))) * 255.0
    k = cov[idx][:, None]
    patch = img[y0:y1, x0:x1].reshape(-1, img.shape[2])
    patch[idx] = k * grey[:, None] + (1 - k) * patch[idx]
    img[y0:y1, x0:x1] = patch.reshape(y1 - y0, x1 - x0, -1)


@dataclass
class Capture:
    """One frame pair: the camera image and the lidar cloud (lidar frame: x y z [reflectivity])."""
    name: str
    image: np.ndarray | None = None
    points: np.ndarray | None = None
    image_path: str = ''
    cloud_path: str = ''
    board: BoardPose | None = None         # synthetic: where the board stood
    error: str = ''                        # what went wrong reading it


def true_lidar(lidar, st: LidarCalibSettings):
    """The lidar as mounted in a synthetic capture: the plan plus the injected error."""
    e = st.error()
    t = copy.copy(lidar)
    t.x, t.y, t.z = lidar.x + e[0] / 1000, lidar.y + e[1] / 1000, lidar.z + e[2] / 1000
    t.yaw, t.pitch, t.roll = lidar.yaw + e[3], lidar.pitch + e[4], lidar.roll + e[5]
    return t


def simulate(project, engine, cam, lidar, poses, st: LidarCalibSettings, progress=None, cancel=None):
    """[Capture] for the planned board poses: the camera image (RGB, the sensor's full resolution) and the scan of
    the lidar mounted with st's injected error. None when cancelled."""
    from .synth import render_world
    if engine is None:
        from .analysis import Engine
        engine = Engine()
    intr = resolve(project.module_for(cam), project.lens_for(cam))
    W2O = _w2o(cam)
    C = cam.pos().astype(float)
    m = project.sensor_model_for(lidar)
    reach = max([b.distance for b in poses] + [st.far_m])
    world = make_world(project, st.scene, reach)
    cube = _cubes(project, engine)
    w, h = int(intr.width), int(intr.height)

    def prog(f, text):
        if progress is not None:
            progress(min(max(f, 0.0), 1.0), text)
    prog(0.0, f'Rendering {cam.tag}\'s view')
    if cancel is not None and cancel():
        return None
    bg = render_world(project, engine, cam, world, w, h, noise=0.0).astype(np.float32)
    cc = cube(C) if cube else None

    def cam_depth(dirs):
        return _scene_hits(C, dirs, world, cc, NEAR_CLIP)[0]
    true = true_lidar(lidar, st)
    o = true.pos().astype(float)
    Ds = S.directions(m.elevations(), S.scan_azimuths(m)).reshape(-1, 3)
    Dw = Ds @ S.rotation(true).T
    scene_t, scene_r = _scene_hits(o, Dw, world, cube(o) if cube else None, OWN_MOUNT_M)
    W, H = st.board_size()
    out = []
    for k, bp in enumerate(poses):
        if cancel is not None and cancel():
            return None
        rng = np.random.default_rng(int(st.seed) * 1009 + k)
        img = bg.copy()
        _paint_board(img, intr, W2O, C, bp, st, cam_depth)
        if st.image_noise > 0:
            img += rng.standard_normal(img.shape, dtype=np.float32) * np.float32(st.image_noise)
        img = np.clip(img, 0, 255).astype(np.uint8)
        # the lidar: the board in front of whatever the beam would meet
        n = bp.R[:, 2]
        den = Dw @ n
        with np.errstate(divide='ignore', invalid='ignore'):
            tb = np.where(np.abs(den) > 1e-9, float((bp.centre - o) @ n) / den, -1.0)
        X = o + Dw * np.maximum(tb, 0)[:, None]
        a = (X - bp.centre) @ bp.R[:, 0]
        b = (X - bp.centre) @ bp.R[:, 1]
        on = (tb > 0) & (np.abs(a) <= W / 2) & (np.abs(b) <= H / 2) & (tb < scene_t)
        T = np.where(on, tb, scene_t)
        refl = scene_r.copy()
        if on.any():
            val = _checker_value(a[on], b[on], st, np.full(int(on.sum()), 1e-4))
            refl[on] = REFLECT['board_dark'] + (REFLECT['board_white'] - REFLECT['board_dark']) * val
        ok = np.isfinite(T) & (T >= m.range_min_m) & (T <= m.range_m)
        Tn = T[ok] + rng.normal(0, st.range_sigma_mm / 1000.0, int(ok.sum()))
        pts = np.c_[Ds[ok] * Tn[:, None], refl[ok]].astype(np.float32)
        out.append(Capture(f'pose{k + 1:02d}', img, pts, board=bp))
        prog((k + 1) / max(len(poses), 1), f'Captured pose {k + 1} of {len(poses)}')
    return out


# =========================================================================== point clouds: read and write
CLOUD_EXT = ('.pcd.bin', '.pcd', '.bin', '.ply', '.npy', '.csv', '.txt')
IMAGE_EXT = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
_INTENSITY = ('intensity', 'i', 'reflectivity', 'signal', 'remission')
_PCD_TYPES = {('F', 4): '<f4', ('F', 8): '<f8', ('I', 1): '<i1', ('I', 2): '<i2', ('I', 4): '<i4', ('I', 8): '<i8',
              ('U', 1): '<u1', ('U', 2): '<u2', ('U', 4): '<u4', ('U', 8): '<u8'}
_PLY_TYPES = {'char': 'i1', 'int8': 'i1', 'uchar': 'u1', 'uint8': 'u1', 'short': 'i2', 'int16': 'i2',
              'ushort': 'u2', 'uint16': 'u2', 'int': 'i4', 'int32': 'i4', 'uint': 'u4', 'uint32': 'u4',
              'float': 'f4', 'float32': 'f4', 'double': 'f8', 'float64': 'f8'}


def split_ext(name: str) -> tuple[str, str]:
    """(stem, extension) with the two-part '.pcd.bin' kept whole; extension lower case ('' when unknown)."""
    low = name.lower()
    for ext in CLOUD_EXT + IMAGE_EXT:
        if low.endswith(ext):
            return name[:-len(ext)], ext
    return os.path.splitext(name)[0], ''


def _columns(names, cols: dict, path) -> np.ndarray:
    """(N, 3|4) x y z [intensity] from named columns; finite, non-zero points only (a driver writes 0 0 0 or NaN for
    no return)."""
    low = {str(k).lower(): k for k in names}
    miss = [a for a in 'xyz' if a not in low]
    if miss:
        raise ValueError(f'{os.path.basename(path)}: no {"/".join(miss)} field (fields: {", ".join(map(str, names))}).')
    xyz = [np.asarray(cols[low[a]], float).reshape(-1) for a in 'xyz']
    it = next((low[k] for k in _INTENSITY if k in low), None)
    if it is not None:
        xyz.append(np.asarray(cols[it], float).reshape(-1))
    return _clean(np.stack(xyz, 1))


def _clean(P) -> np.ndarray:
    P = np.asarray(P, float)
    if P.ndim != 2 or P.shape[1] < 3:
        raise ValueError('not an N × 3 point array')
    keep = np.isfinite(P[:, :3]).all(1) & (np.abs(P[:, :3]).sum(1) > 1e-6)
    return P[keep]


def _lzf_decompress(data: bytes, size: int) -> bytes:
    """LZF (as PCL's binary_compressed uses it)."""
    out = bytearray(size)
    i = o = 0
    n = len(data)
    while i < n:
        ctrl = data[i]
        i += 1
        if ctrl < 32:                         # a literal run
            ln = ctrl + 1
            out[o:o + ln] = data[i:i + ln]
            i += ln
            o += ln
        else:                                 # a back reference
            ln = ctrl >> 5
            ref = o - ((ctrl & 0x1f) << 8) - 1
            if ln == 7:
                ln += data[i]
                i += 1
            ref -= data[i]
            i += 1
            ln += 2
            if ref < 0:
                raise ValueError('corrupt LZF data')
            for _ in range(ln):              # may overlap itself
                out[o] = out[ref]
                o += 1
                ref += 1
    if o != size:
        raise ValueError('LZF data ends early')
    return bytes(out)


def read_pcd(path) -> np.ndarray:
    with open(path, 'rb') as f:
        data = f.read()
    head, pos = {}, 0
    while True:
        end = data.find(b'\n', pos)
        if end < 0:
            raise ValueError(f'{os.path.basename(path)}: the PCD header has no DATA line.')
        line = data[pos:end].decode('ascii', 'replace').strip()
        pos = end + 1
        if not line or line.startswith('#'):
            continue
        key, *vals = line.split()
        head[key.upper()] = vals
        if key.upper() == 'DATA':
            break
    try:
        names = head['FIELDS']
        sizes = [int(s) for s in head['SIZE']]
        types = [t.upper() for t in head['TYPE']]
        counts = [int(c) for c in head.get('COUNT', ['1'] * len(names))]
        npts = int(head.get('POINTS', [int(head['WIDTH'][0]) * int(head['HEIGHT'][0])])[0])
    except (KeyError, ValueError, IndexError) as e:
        raise ValueError(f'{os.path.basename(path)}: unreadable PCD header ({e}).') from None
    kind = head['DATA'][0].lower() if head['DATA'] else ''
    uniq = []
    for i, nm in enumerate(names):
        uniq.append(nm if nm not in uniq else f'{nm}_{i}')
    if kind == 'ascii':
        text = data[pos:].decode('ascii', 'replace')
        arr = np.loadtxt(io.StringIO(text), ndmin=2) if text.strip() else np.zeros((0, sum(counts)))
        cols, c0 = {}, 0
        for nm, cnt in zip(uniq, counts):
            cols[nm] = arr[:, c0]
            c0 += cnt
        return _columns(uniq, cols, path)
    try:
        dt = np.dtype([(nm, _PCD_TYPES[(t, s)], (c,)) if c > 1 else (nm, _PCD_TYPES[(t, s)])
                       for nm, t, s, c in zip(uniq, types, sizes, counts)])
    except KeyError:
        raise ValueError(f'{os.path.basename(path)}: unsupported PCD field type.') from None
    if kind == 'binary':
        rec = np.frombuffer(data, dt, count=npts, offset=pos)
        return _columns(uniq, {nm: rec[nm] for nm in uniq}, path)
    if kind == 'binary_compressed':
        csize, usize = np.frombuffer(data, '<u4', 2, pos)
        raw = _lzf_decompress(data[pos + 8:pos + 8 + int(csize)], int(usize))
        cols, off = {}, 0                    # field by field (structure of arrays)
        for nm, t, s, c in zip(uniq, types, sizes, counts):
            a = np.frombuffer(raw, _PCD_TYPES[(t, s)], npts * c, off).reshape(npts, c)
            cols[nm] = a[:, 0]
            off += npts * c * s
        return _columns(uniq, cols, path)
    raise ValueError(f'{os.path.basename(path)}: unknown PCD data kind "{kind}".')


def read_bin(path, fields='auto') -> np.ndarray:
    """KITTI (float32 x y z intensity) or nuScenes (x y z intensity ring) .bin."""
    size = os.path.getsize(path)
    if fields not in ('4', '5'):
        if path.lower().endswith('.pcd.bin'):
            fields = '5'
        elif size % 16 == 0:
            fields = '4'
        elif size % 20 == 0:
            fields = '5'
        else:
            raise ValueError(f'{os.path.basename(path)}: {size} bytes is not a whole number of 4- or 5-float points '
                             '(KITTI or nuScenes .bin).')
    k = int(fields)
    if size % (4 * k):
        raise ValueError(f'{os.path.basename(path)}: {size} bytes is not a whole number of {k}-float points.')
    a = np.fromfile(path, '<f4').reshape(-1, k)
    return _clean(a[:, :4].astype(float))


def read_ply(path) -> np.ndarray:
    with open(path, 'rb') as f:
        data = f.read()
    end = data.find(b'end_header')
    if not data.startswith(b'ply') or end < 0:
        raise ValueError(f'{os.path.basename(path)}: not a PLY file.')
    nl = data.find(b'\n', end)
    head = data[:end].decode('ascii', 'replace').splitlines()
    body = nl + 1
    fmt, elements = None, []
    for line in head:
        p = line.split()
        if not p:
            continue
        if p[0] == 'format':
            fmt = p[1]
        elif p[0] == 'element':
            elements.append([p[1], int(p[2]), []])
        elif p[0] == 'property' and elements:
            if p[1] == 'list':
                elements[-1][2].append((p[-1], None))
            else:
                elements[-1][2].append((p[-1], p[1]))
    if not elements or elements[0][0] != 'vertex':
        raise ValueError(f'{os.path.basename(path)}: the first PLY element is not "vertex".')
    _, n, props = elements[0]
    if any(t is None for _, t in props):
        raise ValueError(f'{os.path.basename(path)}: list properties on the vertices are not supported.')
    names = [nm for nm, _ in props]
    if fmt == 'ascii':
        lines = data[body:].decode('ascii', 'replace').splitlines()[:n]
        arr = np.loadtxt(lines, ndmin=2) if n else np.zeros((0, len(props)))
        return _columns(names, {nm: arr[:, i] for i, nm in enumerate(names)}, path)
    if fmt not in ('binary_little_endian', 'binary_big_endian'):
        raise ValueError(f'{os.path.basename(path)}: unknown PLY format {fmt}.')
    bo = '<' if fmt == 'binary_little_endian' else '>'
    try:
        dt = np.dtype([(nm, bo + _PLY_TYPES[t]) for nm, t in props])
    except KeyError as e:
        raise ValueError(f'{os.path.basename(path)}: unknown PLY type {e}.') from None
    rec = np.frombuffer(data, dt, count=n, offset=body)
    return _columns(names, {nm: rec[nm] for nm in names}, path)


def read_npy(path) -> np.ndarray:
    a = np.load(path, allow_pickle=False)
    if a.dtype.names:
        return _columns(a.dtype.names, {nm: a[nm] for nm in a.dtype.names}, path)
    if a.ndim != 2 or a.shape[1] < 3:
        raise ValueError(f'{os.path.basename(path)}: expected an N × 3 (or more) array, got {a.shape}.')
    return _clean(a[:, :4].astype(float))


def read_text(path) -> np.ndarray:
    with open(path, encoding='utf-8-sig', errors='replace') as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.lstrip().startswith(('#', '//'))]
    if not lines:
        raise ValueError(f'{os.path.basename(path)}: no points.')
    first = lines[0]
    delim = ',' if ',' in first else ';' if ';' in first else None
    split = (lambda s: [x.strip() for x in s.split(delim)]) if delim else (lambda s: s.split())   # noqa: E731
    names = None
    if re.search(r'[A-Za-z_]', first.replace('e+', '').replace('e-', '').replace('E', '').replace('nan', '')):
        names = split(first)
        lines = lines[1:]
    try:
        arr = np.array([[float(x) for x in split(ln)] for ln in lines], float)
    except ValueError as e:
        raise ValueError(f'{os.path.basename(path)}: not a table of numbers ({e}).') from None
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 3:
        raise ValueError(f'{os.path.basename(path)}: expected x, y, z columns.')
    if names and len(names) == arr.shape[1]:
        return _columns(names, {nm: arr[:, i] for i, nm in enumerate(names)}, path)
    return _clean(arr[:, :4])


def read_cloud(path, bin_fields='auto') -> np.ndarray:
    """A point cloud (N, 3 or 4: x y z [intensity]) in the lidar's frame, any supported format. Raises ValueError with
    what is wrong."""
    _, ext = split_ext(os.path.basename(path))
    if ext == '.pcd':
        return read_pcd(path)
    if ext in ('.bin', '.pcd.bin'):
        return read_bin(path, bin_fields)
    if ext == '.ply':
        return read_ply(path)
    if ext == '.npy':
        return read_npy(path)
    if ext in ('.csv', '.txt'):
        return read_text(path)
    raise ValueError(f'{os.path.basename(path)}: not a point cloud format Vantage reads '
                     f'({", ".join(CLOUD_EXT)}).')


def write_cloud(path, pts, binary=True):
    """Write (N, 3|4) points: .pcd (binary or ascii), .bin (KITTI float32 × 4), .pcd.bin (nuScenes × 5, ring 0),
    .ply (binary little-endian or ascii), .npy, .csv or .txt (x y z intensity)."""
    P = np.asarray(pts, float)
    if P.shape[1] < 4:
        P = np.c_[P[:, :3], np.zeros(len(P))]
    P = P[:, :4]
    _, ext = split_ext(os.path.basename(path))
    n = len(P)
    if ext == '.pcd':
        head = ('# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z intensity\nSIZE 4 4 4 4\n'
                f'TYPE F F F F\nCOUNT 1 1 1 1\nWIDTH {n}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {n}\n'
                f'DATA {"binary" if binary else "ascii"}\n')
        with open(path, 'wb') as f:
            f.write(head.encode('ascii'))
            if binary:
                f.write(P.astype('<f4').tobytes())
            else:
                np.savetxt(f, P, fmt='%.6f')
    elif ext == '.pcd.bin':
        np.c_[P, np.zeros(n)].astype('<f4').tofile(path)
    elif ext == '.bin':
        P.astype('<f4').tofile(path)
    elif ext == '.ply':
        head = (f'ply\nformat {"binary_little_endian" if binary else "ascii"} 1.0\nelement vertex {n}\n'
                'property float x\nproperty float y\nproperty float z\nproperty float intensity\nend_header\n')
        with open(path, 'wb') as f:
            f.write(head.encode('ascii'))
            if binary:
                f.write(P.astype('<f4').tobytes())
            else:
                np.savetxt(f, P, fmt='%.6f')
    elif ext == '.npy':
        np.save(path, P.astype(np.float32))
    elif ext in ('.csv', '.txt'):
        sep = ',' if ext == '.csv' else ' '
        np.savetxt(path, P, fmt='%.6f', delimiter=sep, header=sep.join('x y z intensity'.split()), comments='')
    else:
        raise ValueError(f'Cannot write a point cloud as {ext or os.path.basename(path)}.')


def read_image(path) -> np.ndarray:
    """RGB (or grey) uint8 image; 16-bit images are scaled down."""
    import cv2
    buf = np.fromfile(path, np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f'{os.path.basename(path)}: not an image OpenCV can read.')
    if img.dtype != np.uint8:
        img = (img.astype(float) / max(float(img.max()), 1.0) * 255).astype(np.uint8)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB if img.shape[2] == 4 else cv2.COLOR_BGR2RGB)
    return img


def _natural(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]


def load_folder(folder, st: LidarCalibSettings | None = None):
    """([Capture], notes): image + cloud pairs matched by file stem (pose01.png + pose01.pcd). A pair that cannot be
    read is kept with its error. Raises ValueError when the folder holds no pair."""
    st = st or LidarCalibSettings()
    if not folder or not os.path.isdir(folder):
        raise ValueError(f'Not a folder: {folder or "(none chosen)"}')
    imgs, clouds = {}, {}
    for name in sorted(os.listdir(folder)):
        stem, ext = split_ext(name)
        if ext in IMAGE_EXT:
            imgs.setdefault(stem, os.path.join(folder, name))
        elif ext in CLOUD_EXT:
            if stem in clouds:
                continue                                     # the first (by CLOUD_EXT order) of the same stem wins
            clouds[stem] = os.path.join(folder, name)
    both = sorted(set(imgs) & set(clouds), key=_natural)
    notes = []
    lone = sorted(set(imgs) ^ set(clouds), key=_natural)
    if lone:
        notes.append(f'{len(lone)} file{"s" if len(lone) > 1 else ""} without a partner of the same name: '
                     + ', '.join(lone[:6]) + (' …' if len(lone) > 6 else ''))
    if not both:
        raise ValueError(f'No image + point cloud pairs in {folder}: name them pose01.png + pose01.pcd (images: '
                         f'{", ".join(IMAGE_EXT)}; clouds: {", ".join(CLOUD_EXT)}).')
    out = []
    for stem in both:
        cap = Capture(stem, image_path=imgs[stem], cloud_path=clouds[stem])
        try:
            cap.image = read_image(imgs[stem])
            cap.points = read_cloud(clouds[stem], st.bin_fields).astype(np.float32)
        except (ValueError, OSError) as e:
            cap.error = str(e)
        out.append(cap)
    return out, notes


def save_captures(captures, folder, cloud_ext='.pcd', binary=True) -> list[str]:
    """Write captures as image + cloud pairs (pose01.png + pose01.pcd …) a folder calibration reads back."""
    import cv2
    os.makedirs(folder, exist_ok=True)
    paths = []
    for c in captures:
        if c.image is None or c.points is None:
            continue
        ip = os.path.join(folder, c.name + '.png')
        img = c.image if c.image.ndim == 2 else cv2.cvtColor(c.image, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode('.png', img)
        buf.tofile(ip)
        cp = os.path.join(folder, c.name + cloud_ext)
        write_cloud(cp, c.points, binary)
        paths += [ip, cp]
    return paths


# =========================================================================== camera: the board in the image
@dataclass
class CamBoard:
    """The board as the camera measured it, in the optical frame."""
    uv: np.ndarray               # (rows*cols, 2) corners in image pixels, model order
    rays: np.ndarray             # (rows*cols, 3) their bearing rays
    R: np.ndarray                # board -> optical (normal towards the camera)
    c: np.ndarray                # board centre
    rms_px: float
    cov: np.ndarray              # (6, 6) of (rotation vector, translation), optical frame
    how: str = ''
    phase: int = -1              # 0: the square at the board frame's (−x, −y) corner is dark, 1: bright, −1: unknown
    scale: tuple = (1.0, 1.0)    # image pixels per sensor pixel (an image may be scaled down)


def _gray(img):
    import cv2
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img


def _rays_from_patch(det, Rv, f, n):
    rv = np.c_[(det.reshape(-1, 2) - n / 2 + 0.5) / f, np.ones(det.size // 2)]
    ro = rv @ Rv.T
    return ro / np.linalg.norm(ro, axis=1, keepdims=True)


def _to_model_grid(G, st):
    """A detected (r, c, ...) grid in the model's (rows, cols) shape (transposed when found the other way round)."""
    if G.shape[:2] == (st.rows, st.cols):
        return G
    if G.shape[:2] == (st.cols, st.rows):
        return np.transpose(G, (1, 0) + tuple(range(2, G.ndim)))
    return None


def _view(gray, intr, sx, sy, Rv, f, n):
    import cv2
    xs = (np.arange(n) + 0.5 - n / 2) / f
    X, Y = np.meshgrid(xs, xs)
    ro = np.stack([X, Y, np.ones_like(X)], -1).reshape(-1, 3) @ Rv.T
    u, v, th, ok = intr.project_dirs(ro)
    mx = np.where(ok, u * sx - 0.5, -1).reshape(n, n).astype(np.float32)
    my = np.where(ok, v * sy - 0.5, -1).reshape(n, n).astype(np.float32)
    return cv2.remap(gray, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _search(gray, intr, sx, sy, st):
    """Corner rays (rows, cols, 3) of a board found anywhere: the whole image first, then virtual pinhole views tiling
    the lens circle. (rays, how) or (None, '')."""
    import cv2
    from .install import _find, _virtual_basis
    pattern = (st.cols, st.rows)
    h, w = gray.shape[:2]
    small = 1.0 if max(h, w) <= 2200 else 2200.0 / max(h, w)
    g = gray if small == 1.0 else cv2.resize(gray, (int(w * small), int(h * small)), interpolation=cv2.INTER_AREA)
    det = _find(g, pattern)
    if det is not None:
        uv = (det.reshape(-1, 2) + 0.5) / small
        d, _ = intr.unproject(uv[:, 0] / sx, uv[:, 1] / sy)
        G = _to_model_grid(d.reshape(det.shape[0], det.shape[1], 3), st)
        if G is not None:
            return G, 'whole image'
    n, half = 640, math.radians(32)
    f = n / 2 / math.tan(half)
    dirs = [(0.0, 0.0)] + [(38.0, a) for a in range(0, 360, 60)] + [(70.0, a) for a in range(0, 360, 36)]
    for th, az in dirs:
        if math.radians(th) > intr.max_theta:
            continue
        t, a = math.radians(th), math.radians(az)
        c = np.array([math.sin(t) * math.cos(a), math.sin(t) * math.sin(a), math.cos(t)])
        Rv = _virtual_basis(c)
        det = _find(_view(gray, intr, sx, sy, Rv, f, n), pattern)
        if det is None:
            continue
        G = _to_model_grid(_rays_from_patch(det, Rv, f, n).reshape(det.shape[0], det.shape[1], 3), st)
        if G is not None:
            return G, 'view search'
    return None, ''


def _refine_in_view(gray, intr, sx, sy, st, rays_grid, outline_rays=None, fills=(0.72, 0.45)):
    """The board found again in tight rectified views round expected corner rays (rows, cols, 3), one per fill; the
    found corners ordered like the expected ones. Returns a list of rays (rows, cols, 3), one per view that found it."""
    from .install import rectify, _find, _match_grid
    flat = rays_grid.reshape(-1, 3)
    allr = flat if outline_rays is None else np.vstack([flat, outline_rays])
    out = []
    for fill in fills:
        patch, Rv, f, n = rectify(gray, intr, allr, sx, sy, fill=fill)
        det = _find(patch, (st.cols, st.rows))
        if det is None:
            continue
        pr = flat @ Rv
        pred = (pr[:, :2] / pr[:, 2:3] * f + n / 2 - 0.5).reshape(st.rows, st.cols, 2)
        m = _match_grid(det, pred)
        if m is None or m[0] > n * 0.15:
            continue
        out.append(_rays_from_patch(m[1], Rv, f, n).reshape(st.rows, st.cols, 3))
    return out


def _board_pnp(intr, sx, sy, model, rays, uv):
    """Board pose (R board -> optical, centre, rms px, cov 6×6) from corner rays and image points."""
    import cv2
    from scipy.optimize import least_squares
    from .install import _virtual_basis
    Rv = _virtual_basis(rays.mean(0))
    loc = rays @ Rv
    norm = (loc[:, :2] / loc[:, 2:3]).astype(np.float64)
    cands = []
    try:
        n_sol, rvecs, tvecs, _ = cv2.solvePnPGeneric(model.astype(np.float64), norm, np.eye(3), None,
                                                     flags=cv2.SOLVEPNP_IPPE)
        cands = [(cv2.Rodrigues(rv)[0], tv.ravel()) for rv, tv in zip(rvecs, tvecs)]
    except cv2.error:
        pass
    if not cands:
        ok, rv, tv = cv2.solvePnP(model.astype(np.float64), norm, np.eye(3), None, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return None
        cands = [(cv2.Rodrigues(rv)[0], tv.ravel())]
    best = None
    for Rvb, tvb in cands:
        R0, c0 = Rv @ Rvb, Rv @ tvb

        def res(x, R0=R0, c0=c0):
            R = _exp(x[:3]) @ R0
            P = model @ R.T + c0 + x[3:]
            u, v, _, _ = intr.project_dirs(P)
            return (np.c_[u * sx, v * sy] - uv).ravel()
        sol = least_squares(res, np.zeros(6), method='lm', x_scale='jac', xtol=1e-12, ftol=1e-12)
        if not np.all(np.isfinite(sol.fun)):
            continue
        rms = float(np.sqrt(np.mean(sol.fun.reshape(-1, 2) ** 2) * 2))
        if best is None or rms < best[2]:
            best = (_exp(sol.x[:3]) @ R0, c0 + sol.x[3:], rms, sol.jac, sol.fun)
    if best is None:
        return None
    R, c, rms, J, r = best
    m_ = len(r)
    s2 = max(float(r @ r) / max(m_ - 6, 1), 0.03 ** 2)
    try:
        cov = s2 * np.linalg.inv(J.T @ J)
    except np.linalg.LinAlgError:
        cov = s2 * np.linalg.pinv(J.T @ J)
    return R, c, rms, cov


def square_centres(st: LidarCalibSettings):
    """Board-frame centres (n, 3) of the checker squares and their parity ((i + j) % 2, 0 at the (−x, −y) corner)."""
    sq = st.square_mm / 1000.0
    nc, nr = st.cols + 1, st.rows + 1
    i, j = np.meshgrid(np.arange(nc), np.arange(nr))
    a = (i.ravel() + 0.5) * sq - nc * sq / 2
    b = (j.ravel() + 0.5) * sq - nr * sq / 2
    return np.c_[a, b, np.zeros(a.size)], (i.ravel() + j.ravel()) % 2


def _square_phase(gray, intr, sx, sy, R, c, st) -> int:
    """Which squares are dark in the camera's view of the board (the board frame's (−x, −y) corner square dark: 0)."""
    P, par = square_centres(st)
    u, v, _, ok = intr.project_dirs(P @ R.T + c)
    x = np.clip(np.round(u * sx - 0.5).astype(int), 0, gray.shape[1] - 1)
    y = np.clip(np.round(v * sy - 0.5).astype(int), 0, gray.shape[0] - 1)
    g = gray[y, x].astype(float)
    if not ok.all() or not (par == 0).any() or not (par == 1).any():
        return -1
    m0, m1 = g[par == 0].mean(), g[par == 1].mean()
    if abs(m0 - m1) < 20:
        return -1
    return 0 if m0 < m1 else 1


def detect_board(project, cam, img, st: LidarCalibSettings, hint: BoardPose | None = None) -> CamBoard | None:
    """The board in one camera image: corners and its pose in the optical frame. hint: where it should stand (the
    plan); without one (real data) the whole image is searched."""
    from .install import _scale
    gray = _gray(img)
    intr = resolve(project.module_for(cam), project.lens_for(cam))
    sx, sy = _scale(intr, gray.shape)
    model = model_corners(st)
    W2O = _w2o(cam)
    C = cam.pos()
    cands, how = [], ''
    W, H = st.board_size()
    if hint is not None:
        P = hint.centre + model @ hint.R.T
        Po = hint.centre + outline_local(W, H, 1) @ hint.R.T
        r = (np.vstack([P, Po]) - C) @ W2O.T
        _, _, th, ok = intr.project_dirs(r)
        if ok.all():
            r /= np.linalg.norm(r, axis=1, keepdims=True)
            cands = _refine_in_view(gray, intr, sx, sy, st, r[:len(P)].reshape(st.rows, st.cols, 3), r[len(P):])
            how = 'expected place'
    if not cands:
        coarse, how = _search(gray, intr, sx, sy, st)
        if coarse is None:
            return None
        cands = [coarse]
        # a first pose from the coarse corners gives the board's outline: refine as round an expected board
        flat = coarse.reshape(-1, 3)
        u0, v0, _, ok0 = intr.project_dirs(flat)
        first = _board_pnp(intr, sx, sy, model, flat, np.c_[u0 * sx, v0 * sy]) if ok0.all() else None
        if first is not None:
            Rb, cb_ = first[0], first[1]
            r = np.vstack([model @ Rb.T + cb_, outline_local(W, H, 1) @ Rb.T + cb_])
            r /= np.linalg.norm(r, axis=1, keepdims=True)
            cands += _refine_in_view(gray, intr, sx, sy, st, r[:len(model)].reshape(st.rows, st.cols, 3),
                                     r[len(model):])
    # every view that found the board gives its corners: keep the set the board's pose fits best (a corner the
    # detector misplaced in one view shows as a worse fit)
    best = None
    for rays in cands:
        rays = rays.reshape(-1, 3)
        u, v, _, ok = intr.project_dirs(rays)
        if not ok.all():
            continue
        uv = np.c_[u * sx, v * sy]
        sol = _board_pnp(intr, sx, sy, model, rays, uv)
        if sol is not None and (best is None or sol[2] < best[0][2]):
            best = (sol, rays, uv)
    if best is None:
        return None
    (R, c, rms, cov), rays, uv = best
    if R[:, 2] @ c > 0:                     # normal towards the camera: turn the board frame 180° about its width axis
        R = R @ np.diag([1.0, -1.0, -1.0])
        uv = uv.reshape(st.rows, st.cols, 2)[::-1].reshape(-1, 2)
        rays = rays.reshape(st.rows, st.cols, 3)[::-1].reshape(-1, 3)
    return CamBoard(uv, rays, R, c, rms, cov, how, _square_phase(gray, intr, sx, sy, R, c, st), (sx, sy))


# =========================================================================== lidar: the board in the cloud
@dataclass
class LidarBoard:
    points: np.ndarray           # (M, 3) board points, lidar frame
    normal: np.ndarray           # plane n·p = d, normal towards the lidar
    d: float
    rms_m: float
    edge_in: np.ndarray          # (K, 3) board points at its edge (on the fitted plane): inside the outline
    edge_out: np.ndarray         # (K2, 3) where the neighbouring beams that missed it cross the plane: outside
    rings: int
    how: str = ''
    ck_pts: np.ndarray | None = None     # board points (on the plane) whose reflectivity is clearly dark or bright
    ck_dark: np.ndarray | None = None
    ck_note: str = ''                    # why the reflectivity is not used


def _fit_plane(P):
    c = P.mean(0)
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    n = vt[2]
    if n @ c > 0:
        n = -n
    return n, float(n @ c)


def _ransac(P, thr, rng, ref_n=None, max_ang=None, ref_c=None, max_off=None, iters=300):
    """Plane (n, d) with most inliers within thr, optionally near a reference normal and through near a point."""
    if len(P) < 3:
        return None
    i = rng.integers(0, len(P), (iters, 3))
    a, b, c = P[i[:, 0]], P[i[:, 1]], P[i[:, 2]]
    n = np.cross(b - a, c - a)
    nn = np.linalg.norm(n, axis=1)
    good = nn > 1e-9
    n = n[good] / nn[good, None]
    a = a[good]
    if ref_n is not None and max_ang is not None:
        keep = np.abs(n @ ref_n) >= math.cos(math.radians(max_ang))
        n, a = n[keep], a[keep]
    d = np.sum(n * a, 1)
    if ref_c is not None and max_off is not None:
        keep = np.abs(n @ ref_c - d) <= max_off
        n, d = n[keep], d[keep]
    if not len(n):
        return None
    cnt = np.zeros(len(n), int)
    for s in range(0, len(n), 64):                # bounded memory: 64 planes × the points at a time
        cnt[s:s + 64] = (np.abs(P @ n[s:s + 64].T - d[s:s + 64]) < thr).sum(0)
    j = int(np.argmax(cnt))
    return n[j], float(d[j]), int(cnt[j])


def _largest_cluster(P, radius, near=None):
    """Indices of the connected cluster (points closer than radius link) with the most points (or containing the point
    nearest `near`, when it holds at least half as many)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    if len(P) < 2:
        return np.arange(len(P))
    pr = cKDTree(P).query_pairs(radius, output_type='ndarray')
    g = coo_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])), shape=(len(P), len(P))) if len(pr) else \
        coo_matrix((len(P), len(P)))
    _, lab = connected_components(g, directed=False)
    counts = np.bincount(lab)
    best = int(np.argmax(counts))
    if near is not None:
        j = int(lab[np.argmin(np.linalg.norm(P - near, axis=1))])
        if counts[j] >= counts[best] / 2:
            best = j
    return np.flatnonzero(lab == best)


def _spacing(m, r):
    """Largest beam spacing (m) of the lidar at range r."""
    step = max(m.h_res, m.v_step() if m.n_channels() > 1 else m.h_res, 0.05)
    return r * math.radians(step)


def _accept(P, n, W, H, sp):
    """Whether a planar cluster could be the board: no wider than its diagonal, not a sliver."""
    if len(P) < MIN_BOARD_POINTS:
        return False
    q = P - P.mean(0)
    q = q - np.outer(q @ n, n)
    _, s, vt = np.linalg.svd(q, full_matrices=False)
    e1 = np.ptp(q @ vt[0])
    e2 = np.ptp(q @ vt[1])
    diag = math.hypot(W, H)
    return e1 <= diag * 1.1 + 2 * sp and e2 >= 0.25 * min(W, H) and e1 * e2 <= W * H * 1.6 + diag * 2 * sp


def _refine_plane(P, n, d, thr):
    for _ in range(3):
        inl = np.abs(P @ n - d) < thr
        if inl.sum() < 3:
            break
        n, d = _fit_plane(P[inl])
        r = P[inl] @ n - d
        thr = max(3.0 * float(np.sqrt(np.mean(r * r))), 0.01)
    inl = np.abs(P @ n - d) < thr
    return n, d, inl


def _ring_groups(el, gap):
    """Ring index per point from elevations (radians): sorted, split where neighbours differ by more than gap."""
    order = np.argsort(el)
    cut = np.r_[0, np.cumsum(np.diff(el[order]) > gap)]
    ring = np.empty(len(el), int)
    ring[order] = cut
    return ring


def _boundary(Pb, n, d, m, full_dirs=None, full_r=None):
    """Brackets of the board's edge: wherever a board point's neighbouring beam (the next azimuth in its ring, the ring
    above or below at its azimuth) misses the board, the edge lies between the two beams. Returns (inside (K,3),
    outside (K2,3), rings): the board point's and (where the cloud shows the neighbour passed the board: a return
    beyond it) the neighbour beam's intersections with the fitted plane, lidar frame. A neighbour that hit something
    nearer (an occluder) or returned nothing gives no outside point."""
    from scipy.spatial import cKDTree
    none = (np.zeros((0, 3)), np.zeros((0, 3)), 0)
    if m.scan == 'nonrepetitive' or len(Pb) < 3:
        return none
    r = np.linalg.norm(Pb, axis=1)
    el = np.arcsin(np.clip(Pb[:, 2] / r, -1, 1))
    az = np.arctan2(Pb[:, 1], Pb[:, 0])
    ac = math.atan2(np.sin(az).mean(), np.cos(az).mean())
    azr = (az - ac + math.pi) % (2 * math.pi) - math.pi
    vstep = math.radians(m.v_step()) if m.n_channels() > 1 else math.radians(1.0)
    ring = _ring_groups(el, max(0.3 * vstep, 1e-4))
    nr = int(ring.max()) + 1
    if nr < 2:
        return none[:2] + (nr,)
    rel = np.array([el[ring == k].mean() for k in range(nr)])
    gaps = np.diff(rel)
    g_med = float(np.median(gaps)) if len(gaps) else vstep
    E = np.radians(m.elevations())
    hstep_m = math.radians(m.h_res)
    half_az = math.radians(m.az_half())
    tree = cKDTree(full_dirs) if full_dirs is not None and len(full_dirs) else None

    def on_plane(azv, elv):
        dv = np.stack([np.cos(elv) * np.cos(azv), np.cos(elv) * np.sin(azv), np.sin(elv)], -1)
        den = dv @ n
        with np.errstate(divide='ignore', invalid='ignore'):
            t = np.where(np.abs(den) > 1e-6, d / den, np.nan)
        return dv, t

    def passed(azv, elv, tol):
        """Outside points of neighbour beams the cloud shows went past the board's plane."""
        dv, t = on_plane(azv, elv)
        ok = np.isfinite(t) & (t > 0)
        if tree is None:
            return dv[ok] * t[ok, None]
        dist, j = tree.query(dv, distance_upper_bound=tol)
        hit = np.isfinite(dist)
        rr = np.where(hit, full_r[np.minimum(j, len(full_r) - 1)], -1.0)
        ok &= hit & (rr > t - 0.05)
        return dv[ok] * t[ok, None]
    hsteps = np.full(nr, hstep_m)
    for k in range(nr):
        idx = np.flatnonzero(ring == k)
        if len(idx) >= 3:
            dd = np.diff(np.sort(azr[idx]))
            hsteps[k] = float(np.median(dd)) or hstep_m
    edge = np.zeros(len(Pb), bool)
    outs = []
    for k in range(nr):
        idx = np.flatnonzero(ring == k)
        o = idx[np.argsort(azr[idx])]
        hs = hsteps[k]
        tol = 0.35 * min(hs, vstep)
        for j, sgn in ((o[0], -1.0), (o[-1], 1.0)):
            if m.az_half() < 180 and abs(az[j] + sgn * hs) > half_az:
                continue                                       # the field's edge, not the board's
            edge[j] = True
            outs.append(passed(np.array([az[j] + sgn * hs]), el[j:j + 1], tol))
        for dk in (1, -1):
            kk = k + dk
            if 0 <= kk < nr:
                g = abs(rel[kk] - rel[k])
                nb = np.flatnonzero(ring == kk)
                miss = np.abs(azr[idx][:, None] - azr[nb][None, :]).min(1) > 0.75 * max(hsteps[k], hsteps[kk])
            else:
                nxt = E[E > rel[k] + 0.3 * vstep] if dk > 0 else E[E < rel[k] - 0.3 * vstep]
                if not len(nxt):
                    continue                                   # the lidar's top or bottom channel: the field ends
                g = float(nxt.min() - rel[k]) if dk > 0 else float(rel[k] - nxt.max())
                g = g if g <= 3 * g_med else g_med
                miss = np.ones(len(idx), bool)
            j = idx[miss]
            if not len(j):
                continue
            edge[j] = True
            outs.append(passed(az[j], el[j] + dk * g, 0.35 * min(hs, g)))
    dv, t = on_plane(az[edge], el[edge])
    ok = np.isfinite(t) & (t > 0)
    inside = dv[ok] * t[ok, None]
    outside = np.vstack(outs) if outs else np.zeros((0, 3))
    return inside, outside, nr


def _reflectivity(Pb, Ib, n, d, st: LidarCalibSettings):
    """(points on the plane (K,3), dark (K,), note): the board points whose reflectivity is clearly dark or clearly
    bright, when the board's reflectivity is two-tone like a checker (two well-separated levels, about as much dark as
    the checker has); points between the levels (beams on a square's edge) are left out."""
    if not st.reflectivity:
        return None, None, 'off'
    if Ib is None or len(Ib) < 30 or not np.isfinite(Ib).all():
        return None, None, 'the cloud has no reflectivity'
    v = np.asarray(Ib, float)
    lo, hi = np.percentile(v, [2, 98])
    if hi - lo <= 1e-9:
        return None, None, 'the board\'s reflectivity is flat'
    # Otsu's threshold
    hist, edges = np.histogram(v, 64, (lo, hi))
    mids = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * mids) / np.maximum(w0, 1)
    m1 = (np.sum(hist * mids) - np.cumsum(hist * mids)) / np.maximum(w1, 1)
    k = int(np.argmax(w0 * w1 * (m0 - m1) ** 2))
    dark, bright = v[v < mids[k]], v[v >= mids[k]]
    if len(dark) < 5 or len(bright) < 5:
        return None, None, 'the board\'s reflectivity is not two-tone'
    gap = bright.mean() - dark.mean()
    thr = (bright.mean() + dark.mean()) / 2          # Otsu splits the two levels; the cut lies midway between them
    spread = max(float(np.median(np.abs(dark - np.median(dark)))), float(np.median(np.abs(bright - np.median(bright)))))
    W, H = st.board_size()
    sq = st.square_mm / 1000.0
    want = ((st.cols + 1) * (st.rows + 1) // 2) * sq * sq / (W * H)
    frac = len(dark) / len(v)
    if gap < 6 * 1.4826 * spread or not 0.35 * want <= frac <= 1.8 * want:
        return None, None, 'the board\'s reflectivity does not show the checker'
    keep = np.abs(v - thr) > 0.2 * gap
    dirs = Pb[keep] / np.linalg.norm(Pb[keep], axis=1, keepdims=True)
    den = dirs @ n
    ok = np.abs(den) > 1e-6
    pts = dirs[ok] * (d / den[ok])[:, None]
    return pts, (v[keep] < thr)[ok], ''


def find_lidar_board(points, T_guess, cb: CamBoard, st: LidarCalibSettings, m, seed=0) -> LidarBoard | None:
    """The board in one cloud: points round where T_guess puts the camera-measured board, a margin that grows when
    nothing is found; else the largest planar cluster of board size near the board's range."""
    P4 = np.asarray(points, float)
    fin = np.isfinite(P4[:, :3]).all(1)
    P = P4[fin, :3]
    I = P4[fin, 3] if P4.shape[1] > 3 else None
    if len(P) < MIN_BOARD_POINTS:
        return None
    rng = np.random.default_rng(seed)
    W, H = st.board_size()
    Rg, tg = T_guess[:3, :3], T_guess[:3, 3]
    cl = Rg.T @ (cb.c - tg)                      # the board in the lidar frame, by the guess
    A = Rg.T @ cb.R
    rr = float(np.linalg.norm(cl))
    sp = _spacing(m, rr)
    thr0 = max(0.04, 3 * st.range_sigma_mm / 1000.0)
    found, how = None, ''
    for grow in (0.25, 0.5, 0.9, 1.5):
        q = (P - cl) @ A
        sel = (np.abs(q[:, 0]) <= W / 2 + grow) & (np.abs(q[:, 1]) <= H / 2 + grow) & (np.abs(q[:, 2]) <= 0.15 + grow)
        Q = P[sel]
        if len(Q) < MIN_BOARD_POINTS:
            continue
        pl = _ransac(Q, thr0, rng, A[:, 2], 15.0 + 25.0 * grow, cl, 0.1 + grow)
        if pl is None or pl[2] < MIN_BOARD_POINTS:
            continue
        n, d, inl = _refine_plane(Q, pl[0], pl[1], thr0)
        Qi = Q[inl]
        keep = _largest_cluster(Qi, 2.5 * sp + 0.05, cl)
        Qc = Qi[keep]
        if _accept(Qc, n, W, H, sp):
            found, how = Qc, 'expected place' if grow <= 0.25 else f'search {grow:g} m round the expected place'
            break
    if found is None:                            # a poor first guess: a board-sized plane at about the right range
        r_all = np.linalg.norm(P, axis=1)
        pool = P[(r_all > max(rr - 2.0, 0.3)) & (r_all < rr + 2.0)]
        best = None
        for _ in range(8):
            if len(pool) < MIN_BOARD_POINTS:
                break
            pl = _ransac(pool, thr0, rng, iters=400)
            if pl is None or pl[2] < MIN_BOARD_POINTS:
                break
            n, d, inl = _refine_plane(pool, pl[0], pl[1], thr0)
            plane_pts = pool[inl]
            rest = plane_pts
            for _ in range(4):
                if len(rest) < MIN_BOARD_POINTS:
                    break
                keep = _largest_cluster(rest, 2.5 * _spacing(m, float(np.linalg.norm(rest.mean(0)))) + 0.05)
                Qc = rest[keep]
                if _accept(Qc, n, W, H, sp):
                    score = abs(float(np.linalg.norm(Qc.mean(0))) - rr) + float(np.linalg.norm(Qc.mean(0) - cl)) * 0.5
                    if best is None or score < best[0]:
                        best = (score, Qc)
                rest = np.delete(rest, keep, axis=0)
            pool = pool[~inl]
        if best is not None:
            found, how = best[1], 'board-sized plane'
    if found is None:
        return None
    n, d = _fit_plane(found)
    res = found @ n - d
    rms = float(np.sqrt(np.mean(res ** 2)))
    # the board's beams for the edge test: every point near the plane next to the cluster, so a noisy point the
    # plane fit left out does not make an edge inside the board
    from scipy.spatial import cKDTree
    near = np.flatnonzero(np.abs(P @ n - d) < max(6 * rms, 0.05))
    dist, _ = cKDTree(found).query(P[near], distance_upper_bound=1.5 * sp + 0.02)
    li = near[np.isfinite(dist)]
    loose = P[li] if len(li) else found
    full = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-9)
    ein, eout, rings = _boundary(loose, n, d, m, full, np.linalg.norm(P, axis=1))
    ck = _reflectivity(loose, I[li] if I is not None and len(li) else None, n, d, st)
    return LidarBoard(found, n, d, rms, ein, eout, rings, how, *ck)


# =========================================================================== the solve
@dataclass
class PoseObs:
    name: str
    status: str = 'used'               # used | outlier | no_camera | no_lidar | unreadable
    message: str = ''
    cam: CamBoard | None = None
    lid: LidarBoard | None = None
    planned: BoardPose | None = None
    stats: dict = field(default_factory=dict)      # after the solve


def _closed_form(obs):
    """T_camera_lidar from the board planes alone (rotation aligning the normals, translation from the offsets);
    None when the normals do not span three directions."""
    Nc = np.array([o.cam.R[:, 2] for o in obs])
    Nl = np.array([o.lid.normal for o in obs])
    if len(obs) < 3 or np.linalg.svd(Nc, compute_uv=False)[-1] < 0.15:
        return None
    U, _, Vt = np.linalg.svd(Nl.T @ Nc)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    b = np.array([o.cam.R[:, 2] @ o.cam.c - o.cam.R[:, 2] @ (R @ o.lid.points.mean(0)) for o in obs])
    t, *_ = np.linalg.lstsq(Nc, b, rcond=None)
    return _T(R, t)


def _brackets(Ei, Eo, R, t, Rb, cb, W, H):
    """How far (m) the board's edge brackets break the camera-measured outline: an inside point beyond it, an outside
    point within it; 0 where the bracket holds."""
    out = []
    for E, sgn in ((Ei, 1.0), (Eo, -1.0)):
        if len(E):
            q = (E @ R.T + t - cb) @ Rb
            out.append(np.maximum(sgn * _box_sdf(q[:, 0], q[:, 1], W / 2, H / 2), 0.0))
    return np.concatenate(out) if out else np.zeros(0)


def _checker_miss(Pc, dark, R, t, Rb, cb, st, phase):
    """How far (m) each reflectivity-classified board point lies inside a square of the other colour (camera-measured
    board, the camera's colouring): the distance to the nearest square edge that would put it right; 0 where it
    agrees. A dark point outside the checker (on the white margin) must reach the checker."""
    if Pc is None or not len(Pc) or phase < 0:
        return np.zeros(0)
    q = (Pc @ R.T + t - cb) @ Rb
    sq = st.square_mm / 1000.0
    nc, nr = st.cols + 1, st.rows + 1
    u = (q[:, 0] + nc * sq / 2) / sq
    v = (q[:, 1] + nr * sq / 2) / sq
    inside = (u >= 0) & (u <= nc) & (v >= 0) & (v <= nr)
    fu, fv = np.floor(u), np.floor(v)
    miss = (inside & ((fu + fv + phase) % 2 == 0)) != dark
    inf = np.inf
    # a dark point cannot fix itself by leaving the checker for the white margin
    e = np.stack([np.where(dark & (fu <= 0), inf, u - fu), np.where(dark & (fu + 1 >= nc), inf, fu + 1 - u),
                  np.where(dark & (fv <= 0), inf, v - fv), np.where(dark & (fv + 1 >= nr), inf, fv + 1 - v)], 1)
    d_in = np.min(e, 1) * sq
    d_out = np.maximum(_box_sdf(q[:, 0], q[:, 1], nc * sq / 2, nr * sq / 2), 0.0)
    out = np.where(inside, d_in, d_out)
    return np.where(miss & np.isfinite(out), out, 0.0)


def _solve(intr, model, W, H, obs, T0, use_edges=True, seed=0, st=None, use_checker=False):
    """Joint robust least squares over T_camera_lidar and the boards' poses. Returns a dict with the solution, the
    whitened Jacobian's information, the per-pose figures and the residual variance."""
    from scipy.optimize import least_squares
    from scipy.sparse import lil_matrix
    rng = np.random.default_rng(seed)
    N = len(obs)
    R0, t0 = T0[:3, :3], T0[:3, 3]
    data = []
    for o in obs:
        pts = o.lid.points
        if len(pts) > MAX_POINTS_PER_POSE:
            pts = pts[rng.choice(len(pts), MAX_POINTS_PER_POSE, replace=False)]
        s_px = max(o.cam.rms_px, 0.05)
        s_r = max(o.lid.rms_m, 0.002)
        Ei = o.lid.edge_in if use_edges else np.zeros((0, 3))
        Eo = o.lid.edge_out if use_edges else np.zeros((0, 3))
        ck = use_checker and o.lid.ck_pts is not None and o.cam.phase >= 0
        Ck = o.lid.ck_pts if ck else np.zeros((0, 3))
        Cd = o.lid.ck_dark if ck else np.zeros(0, bool)
        data.append((o.cam.R, o.cam.c, o.cam.uv, s_px, pts, s_r, Ei, Eo, Ck, Cd, o.cam.phase, o.cam.scale))
    rows = []                        # (pose, kind, count): 0 corners (2 per), 1 plane, 2 edge brackets, 3 checker
    for k, (_, _, uv, _, pts, _, Ei, Eo, Ck, _, _, _) in enumerate(data):
        rows += [(k, 0, 2 * len(uv)), (k, 1, len(pts)), (k, 2, len(Ei) + len(Eo)), (k, 3, len(Ck))]

    def unpack(x):
        R = _exp(x[:3]) @ R0
        t = t0 + x[3:6]
        return R, t

    def board(x, k):
        Rb0, cb0 = data[k][0], data[k][1]
        j = 6 + 6 * k
        return _exp(x[j:j + 3]) @ Rb0, cb0 + x[j + 3:j + 6]

    def residuals(x):
        R, t = unpack(x)
        out = []
        for k, (_, _, uv, s_px, pts, s_r, Ei, Eo, Ck, Cd, ph, (sx, sy)) in enumerate(data):
            Rb, cb = board(x, k)
            u, v, _, _ = intr.project_dirs(model @ Rb.T + cb)
            out.append(((np.c_[u * sx, v * sy] - uv) / s_px).ravel())
            out.append(((pts @ R.T + t - cb) @ Rb[:, 2]) / s_r)
            out.append(_brackets(Ei, Eo, R, t, Rb, cb, W, H) / EDGE_FLOOR_M)
            out.append(_checker_miss(Ck, Cd, R, t, Rb, cb, st, ph) / EDGE_FLOOR_M)
        return np.concatenate(out)
    n_par = 6 + 6 * N
    n_res = sum(c for _, _, c in rows)
    Sp = lil_matrix((n_res, n_par), dtype=int)
    r0 = 0
    kinds = np.zeros(n_res, int)
    poses = np.zeros(n_res, int)
    for k, kind, cnt in rows:
        if kind:
            Sp[r0:r0 + cnt, :6] = 1
        Sp[r0:r0 + cnt, 6 + 6 * k:12 + 6 * k] = 1
        kinds[r0:r0 + cnt] = kind
        poses[r0:r0 + cnt] = k
        r0 += cnt
    x0 = np.zeros(n_par)
    sol = least_squares(residuals, x0, jac_sparsity=Sp, method='trf', loss='soft_l1', f_scale=3.0, x_scale='jac',
                        ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=200)
    x = sol.x
    r = residuals(x)
    J = sol.jac.toarray() if hasattr(sol.jac, 'toarray') else np.asarray(sol.jac)
    # the residual variance from the inliers (the robust loss let the others count less)
    inl = np.abs(r) < 4 * 3.0
    dof = max(int(inl.sum()) - n_par, 1)
    s2 = float(np.sum(r[inl] ** 2) / dof)
    R, t = unpack(x)
    per = []
    for k, (Rb0, cb0, uv, s_px, pts, s_r, Ei, Eo, Ck, Cd, ph, (sx, sy)) in enumerate(data):
        Rb, cb = board(x, k)
        u, v, _, _ = intr.project_dirs(model @ Rb.T + cb)
        cr = np.linalg.norm(np.c_[u * sx, v * sy] - uv, axis=1)
        Q = pts @ R.T + t
        pr = (Q - cb) @ Rb[:, 2]
        off = (Q - cb0) @ Rb0[:, 2]                         # against the camera's own measurement of the board
        nl = R @ obs[k].lid.normal
        ang = math.degrees(math.acos(min(1.0, abs(float(nl @ Rb0[:, 2])))))
        er = _brackets(Ei, Eo, R, t, Rb, cb, W, H)
        cm = _checker_miss(Ck, Cd, R, t, Rb, cb, st, ph)
        sel = poses == k
        per.append({'corner_rms_px': float(np.sqrt(np.mean(cr ** 2))),
                    'plane_rms_mm': float(np.sqrt(np.mean(pr ** 2))) * 1e3,
                    'offset_mm': float(np.mean(off)) * 1e3, 'angle_deg': ang,
                    'edge_mm': float(np.sqrt(np.mean(er ** 2))) * 1e3 if len(er) else float('nan'),
                    'edge_points': int(len(er)), 'checker_points': int(len(cm)),
                    'checker_wrong': float(np.mean(cm > 0.002)) if len(cm) else float('nan'),
                    'cost': float(np.mean(np.minimum(r[sel] ** 2, 100.0)))})
    return {'x': x, 'T': _T(R, t), 'J': J, 'kinds': kinds, 'poses': poses, 's2': s2, 'per': per,
            'status': int(sol.status), 'nfev': int(sol.nfev), 'n_res': n_res, 'n_par': n_par}


def _marginal(J, rows):
    """Information of the extrinsic (6×6) with the boards' poses marginalised, from the rows of J selected."""
    Jr = J[rows]
    Hm = Jr.T @ Jr
    A, B, Cm = Hm[:6, :6], Hm[:6, 6:], Hm[6:, 6:]
    if Cm.size == 0:
        return A
    try:
        return A - B @ np.linalg.solve(Cm, B.T)
    except np.linalg.LinAlgError:
        return A - B @ np.linalg.pinv(Cm) @ B.T


_AXES = ('x (forward–back)', 'y (left–right)', 'z (up–down)')
_ROT = ('roll (about x)', 'pitch (about y)', 'yaw (about z)')


def _words(vec, names) -> tuple[str, int]:
    """(words, the dominant axis) for a direction: one axis when it is nearly along it, else the two largest."""
    a = np.abs(vec) / max(float(np.linalg.norm(vec)), 1e-12)
    i = int(np.argmax(a))
    if a[i] > 0.85:
        return names[i], i
    j = [int(k) for k in np.argsort(a)[::-1][:2]]
    return f'{names[j[0]].split()[0]} and {names[j[1]].split()[0]} together', i


def _sigma_text(w) -> str:
    if 'sigma_mm' in w:
        return 'no constraint at all' if w['sigma_mm'] > 1000 else f'±{w["sigma_mm"]:.0f} mm'
    return 'no constraint at all' if w['sigma_deg'] > 30 else f'±{w["sigma_deg"]:.2f}°'


def observability(info, W2O, L, tol_m, tol_rad, normals_vehicle, n_used, weak_at=3.0, none_at=10.0):
    """Directions of the extrinsic the board planes alone do not pin: a translation more uncertain than weak_at × tol_m,
    a rotation than weak_at × tol_rad (directions scaled to metres at the boards' distance L to sort them apart);
    beyond none_at × the tolerance the direction is not determined at all (a degenerate set of boards). Returns
    {'sigma_m': [...], 'weak': [{'what', 'kind', 'level' ('none' | 'weak'), 'sigma_mm' | 'sigma_deg'}], 'reason',
    'message', 'degenerate'}."""
    D = np.diag([1 / L, 1 / L, 1 / L, 1, 1, 1])
    Iy = D @ info @ D
    Iy = (Iy + Iy.T) / 2
    lam, V = np.linalg.eigh(Iy)
    sig = 1.0 / np.sqrt(np.maximum(lam, 1e-30))
    weak, seen = [], set()
    for i in np.argsort(-sig):
        v = V[:, i]
        rot, tra = v[:3], v[3:]                    # rot in metres at the boards (rad × L)
        if np.linalg.norm(rot) > np.linalg.norm(tra):
            ratio = sig[i] / L / tol_rad
            words, ax = _words(W2O.T @ rot, _ROT)
            what = 'the rotation about ' + words
            item = {'kind': 'rotation', 'sigma_deg': float(min(math.degrees(sig[i] / L), 1e6))}
        else:
            ratio = sig[i] / tol_m
            words, ax = _words(W2O.T @ tra, _AXES)
            what = 'the translation along ' + words
            item = {'kind': 'translation', 'sigma_mm': float(min(sig[i], 1e6) * 1000)}
        if ratio <= weak_at or (item['kind'], ax) in seen:
            continue
        seen.add((item['kind'], ax))
        weak.append(dict(item, what=what, level='none' if ratio > none_at else 'weak'))
    none = [w for w in weak if w['level'] == 'none']
    reason = ''
    if weak:
        Nv = np.asarray(normals_vehicle, float)
        spread = max((math.degrees(math.acos(min(1.0, abs(float(a @ b))))) for a in Nv for b in Nv), default=0.0)
        if n_used < 3:
            reason = 'fewer than three boards were seen by both sensors'
        elif spread < 5.0:
            reason = 'the boards were all tilted the same way'
        elif np.linalg.svd(Nv, compute_uv=False)[-1] < 0.09:
            reason = 'the boards were only turned about one axis (their normals lie in one plane)'
        else:
            reason = 'the boards are too few or too alike'

    def listing(ws):
        names = [w['what'] for w in ws]
        return ', '.join(names[:-1]) + (' and ' if len(names) > 1 else '') + names[-1]
    msg = ''
    if none:
        msg = (f'{reason[0].upper() + reason[1:]}: {listing(none)} {"are" if len(none) > 1 else "is"} not determined '
               f'by the board planes ({_sigma_text(none[0])}). Add boards turned other ways.')
    elif weak:
        msg = (f'The board planes pin {listing(weak)} only to {_sigma_text(weak[0])}: the answer there rests on the '
               'board\'s edges and its checker in the reflectivity. Boards turned other ways would make it firmer.')
    return {'sigma_m': [float(s) for s in np.sort(sig)], 'weak': weak, 'reason': reason, 'message': msg,
            'degenerate': bool(none)}


# =========================================================================== the result
@dataclass
class LidarCalibResult:
    settings: LidarCalibSettings
    camera_uid: str
    camera_tag: str
    lidar_uid: str
    lidar_tag: str
    variant: str
    variant_uid: str
    as_built: bool
    share: float
    source: str                              # 'synthetic' or the folder
    plan: list                               # [BoardPose] (synthetic, or the plan for display)
    captures: list                           # [Capture]
    poses: list                              # [PoseObs]
    T_plan: np.ndarray
    T: np.ndarray | None = None              # solved T_camera_lidar (optical frame)
    cov: np.ndarray | None = None            # 6×6, (rotation vector, translation) in the optical frame
    sigma: dict = field(default_factory=dict)
    delta: dict = field(default_factory=dict)
    measured: dict = field(default_factory=dict)     # the lidar's pose in the vehicle
    planned: dict = field(default_factory=dict)
    obs: dict = field(default_factory=dict)          # observability
    verdict: str = 'failed'                  # good | warning | serious | unobservable | failed
    message: str = ''
    findings: list = field(default_factory=list)
    truth: dict | None = None                # synthetic: the true pose and the solve's error
    notes: list = field(default_factory=list)
    fit: dict = field(default_factory=dict)
    seconds: float = 0.0
    camera_pose: dict = field(default_factory=dict)  # the camera's pose the lidar is placed from

    @property
    def used(self):
        return [o for o in self.poses if o.status == 'used']

    def worst(self) -> str:
        return VERDICT_SEVERITY.get(self.verdict, 'critical')


def _sigmas(cam, T, cov):
    """1σ of the solve: translation (mm) and rotation (deg) about the optical axes, and the lidar's pose in the vehicle
    (x y z mm, yaw pitch roll deg), propagated numerically."""
    sc = np.sqrt(np.maximum(np.diag(cov), 0))
    out = {'camera': {'x_mm': sc[3] * 1e3, 'y_mm': sc[4] * 1e3, 'z_mm': sc[5] * 1e3,
                      'rx_deg': math.degrees(sc[0]), 'ry_deg': math.degrees(sc[1]), 'rz_deg': math.degrees(sc[2])}}
    R0, t0 = T[:3, :3], T[:3, 3]

    def f(x):
        R, p = lidar_pose(cam, _T(_exp(x[:3]) @ R0, t0 + x[3:]))
        y, pi, r = euler_zyx(R)
        return np.array([p[0], p[1], p[2], y, pi, r])
    base = f(np.zeros(6))
    Jf = np.zeros((6, 6))
    for i in range(6):
        e = np.zeros(6)
        e[i] = 1e-6
        dfx = f(e) - f(-e)
        dfx[3:] = (dfx[3:] + 180) % 360 - 180
        Jf[:, i] = dfx / 2e-6
    cv = Jf @ cov @ Jf.T
    sv = np.sqrt(np.maximum(np.diag(cv), 0))
    out['vehicle'] = {'x_mm': sv[0] * 1e3, 'y_mm': sv[1] * 1e3, 'z_mm': sv[2] * 1e3, 'yaw_deg': sv[3],
                      'pitch_deg': sv[4], 'roll_deg': sv[5]}
    out['position_mm'] = float(max(sv[:3]) * 1e3)
    out['angle_deg'] = float(math.degrees(max(sc[:3])))
    del base
    return out


def _delta(cam, T, T_plan):
    """T against the plan: in the optical frame (Δt mm, rotation vector deg) and the lidar's pose in the vehicle."""
    dR = T[:3, :3] @ T_plan[:3, :3].T
    rv = np.degrees(rodrigues(dR))
    dt = (T[:3, 3] - T_plan[:3, 3]) * 1e3
    Rm, pm = lidar_pose(cam, T)
    Rp, pp = lidar_pose(cam, T_plan)
    ym, pim, rm = euler_zyx(Rm)
    yp, pip, rp = euler_zyx(Rp)
    dv = (pm - pp) * 1e3
    return {'camera': {'x_mm': dt[0], 'y_mm': dt[1], 'z_mm': dt[2], 'rx_deg': rv[0], 'ry_deg': rv[1], 'rz_deg': rv[2]},
            'vehicle': {'x_mm': dv[0], 'y_mm': dv[1], 'z_mm': dv[2], 'yaw_deg': wrap180(ym - yp),
                        'pitch_deg': pim - pip, 'roll_deg': wrap180(rm - rp)},
            'position_mm': float(np.max(np.abs(dv))), 'distance_mm': float(np.linalg.norm(dv)),
            'rotation_deg': _rot_deg(dR)}


def calibrate(project, cam, lidar, captures, st: LidarCalibSettings, plan=None, progress=None, cancel=None,
              truth_lidar=None, source='synthetic', share=0.0, notes=None, variant=None):
    """Detect the board in every capture, solve T_camera_lidar and grade it. None when cancelled."""
    t0 = time.perf_counter()
    v = variant or project.variant
    m = project.sensor_model_for(lidar)
    intr = resolve(project.module_for(cam), project.lens_for(cam))
    T_plan = plan_T(cam, lidar)
    W, H = st.board_size()
    model = model_corners(st)
    res = LidarCalibResult(st, cam.uid, cam.tag, lidar.uid, lidar.tag, v.name, v.uid, 'as-built' in v.name.lower(),
                           share, source, list(plan or []), list(captures), [], T_plan, notes=list(notes or []))
    res.camera_pose = pose_dict(rot_zyx(cam.yaw, cam.pitch, cam.roll), cam.pos())
    Rp, pp = lidar_pose(cam, T_plan)
    res.planned = pose_dict(Rp, pp)

    def prog(f, text):
        if progress is not None:
            progress(min(max(f, 0.0), 1.0), text)
    obs = []
    for k, cp in enumerate(captures):
        if cancel is not None and cancel():
            return None
        o = PoseObs(cp.name, planned=cp.board)
        obs.append(o)
        if cp.error or cp.image is None or cp.points is None:
            o.status, o.message = 'unreadable', cp.error or 'image or cloud missing'
            continue
        try:
            o.cam = detect_board(project, cam, cp.image, st, cp.board)
        except Exception as e:                       # a malformed image
            o.status, o.message = 'no_camera', f'Could not read the image: {e}'
            continue
        if o.cam is None:
            o.status, o.message = 'no_camera', f'{cam.tag} did not find the {st.cols} × {st.rows} board'
            continue
        o.lid = find_lidar_board(cp.points, T_plan, o.cam, st, m, seed=k)
        if o.lid is None:
            o.status, o.message = 'no_lidar', (f'{lidar.tag} did not see the board (outside its field, hidden, or '
                                               'too far from where the plan puts it)')
            continue
        prog(0.8 * (k + 1) / max(len(captures), 1), f'Found the board in {cp.name}')
    good = [o for o in obs if o.status == 'used']
    res.poses = obs
    if len(good) < 3:
        res.verdict = 'failed'
        res.message = (f'{len(good)} of {len(obs)} poses have the board in both {cam.tag} and {lidar.tag}: three or '
                       'more are needed.')
        _findings(res)
        res.seconds = time.perf_counter() - t0
        return res
    # start: the plan, or the planes' closed form when it fits the boards better
    T0 = T_plan

    def plane_cost(T):
        return sum(float(np.mean(((o.lid.points @ T[:3, :3].T + T[:3, 3] - o.cam.c) @ o.cam.R[:, 2]) ** 2))
                   for o in good)
    cf = _closed_form(good)
    if cf is not None and plane_cost(cf) < plane_cost(T_plan):
        T0 = cf
    # a pose whose lidar board was found far from where the start puts it is looked for again round the start
    if T0 is not T_plan:
        for k, o in enumerate(obs):
            if o.status == 'no_lidar' and o.cam is not None:
                lb = find_lidar_board(captures[k].points, T0, o.cam, st, m, seed=k)
                if lb is not None:
                    o.lid, o.status, o.message = lb, 'used', ''
        good = [o for o in obs if o.status == 'used']
    prog(0.85, 'Solving')
    sol = _solve(intr, model, W, H, good, T0, st.edges, int(st.seed), st)
    # set aside a pose that disagrees with the rest (its residuals, whitened, far above the others'), then solve
    # again; at most a third of them
    for _ in range(max(len(good) // 3, 0)):
        if cancel is not None and cancel():
            return None
        costs = np.array([q['cost'] for q in sol['per']])
        worst = int(np.argmax(costs))
        if len(good) <= 3 or costs[worst] <= max(OUTLIER_COST, OUTLIER_COST * float(np.median(costs))):
            break
        o = good[worst]
        o.status = 'outlier'
        o.stats = sol['per'][worst]
        o.message = (f'Set aside: it disagrees with the other poses ({cam.tag}\'s board and {lidar.tag}\'s '
                     f'{o.stats["offset_mm"]:+.0f} mm, {o.stats["angle_deg"]:.2f}° apart after the solve): the board '
                     'moved between the two captures, or frames from different moments')
        good = [g for g in good if g.status == 'used']
        sol = _solve(intr, model, W, H, good, sol['T'], st.edges, int(st.seed), st)
    # then the checker squares in the lidar's reflectivity, from there (close enough not to slip a square)
    ck = [o for o in good if o.lid.ck_pts is not None and o.cam.phase >= 0]
    if ck:
        prog(0.92, 'Solving with the checker squares the lidar sees')
        sol = _solve(intr, model, W, H, good, sol['T'], st.edges, int(st.seed), st, True)
    for o, p in zip(good, sol['per']):
        o.stats = p
    T = sol['T']
    J, kinds = sol['J'], sol['kinds']
    s2 = sol['s2']
    info_all = _marginal(J, np.ones(len(kinds), bool))
    try:
        cov = s2 * np.linalg.inv(info_all)
    except np.linalg.LinAlgError:
        cov = s2 * np.linalg.pinv(info_all)
    info_planes = _marginal(J, kinds <= 1)
    Lm = float(np.mean([np.linalg.norm(o.lid.points.mean(0)) for o in good]))
    W2O = _w2o(cam)
    normals = [W2O.T @ o.cam.R[:, 2] for o in good]
    res.obs = observability(info_planes, W2O, Lm, st.pos_tol_mm / 1000.0, math.radians(st.ang_tol_deg), normals,
                            len(good))
    res.obs['edges'] = bool(st.edges and any(len(o.lid.edge_in) for o in good))
    res.T, res.cov = T, cov
    res.sigma = _sigmas(cam, T, cov)
    res.delta = _delta(cam, T, T_plan)
    Rm, pm = lidar_pose(cam, T)
    res.measured = pose_dict(Rm, pm)
    res.fit = {'residual_variance': s2, 'iterations': sol['nfev'], 'residuals': sol['n_res'],
               'board_distance_m': Lm, 'poses_with_reflectivity_checker': len(ck)}
    if truth_lidar is not None:
        Tt = plan_T(cam, truth_lidar)
        e = _delta(cam, T, Tt)
        res.truth = {'pose': pose_dict(S.rotation(truth_lidar), truth_lidar.pos()),
                     'injected': {k: getattr(st, k) for k in ERR_KEYS},
                     'error': e['vehicle'], 'error_position_mm': e['position_mm'],
                     'error_rotation_deg': e['rotation_deg']}
    _grade(res)
    _findings(res)
    res.seconds = time.perf_counter() - t0
    prog(1.0, res.message)
    return res


def _grade(res: LidarCalibResult):
    st = res.settings
    d = res.delta
    worst_p, worst_a = d['position_mm'], d['rotation_deg']
    if res.obs.get('degenerate'):
        res.verdict = 'unobservable'
        res.message = res.obs['message']
        return
    what = f'{res.lidar_tag} is {worst_p:.0f} mm / {worst_a:.2f}° from its planned pose'
    if worst_p <= st.pos_tol_mm and worst_a <= st.ang_tol_deg:
        res.verdict = 'good'
        res.message = f'{res.lidar_tag} sits where planned: within {st.pos_tol_mm:g} mm / {st.ang_tol_deg:g}° ' \
                      f'({worst_p:.1f} mm / {worst_a:.2f}°).'
    elif worst_p <= 3 * st.pos_tol_mm and worst_a <= 3 * st.ang_tol_deg:
        res.verdict = 'warning'
        res.message = f'{what}: adjust the bracket, or apply the measured pose to the variant.'
    else:
        res.verdict = 'serious'
        res.message = f'{what}: re-mount it, or apply the measured pose to the variant.'


def _findings(res: LidarCalibResult):
    st = res.settings
    out = []

    def add(sev, msg):
        out.append({'severity': sev, 'message': msg})
    add(VERDICT_SEVERITY.get(res.verdict, 'critical'), res.message)
    for o in res.poses:
        if o.status in ('no_camera', 'no_lidar', 'unreadable'):
            add('warning', f'{o.name}: {o.message}.')
        elif o.status == 'outlier':
            add('warning', f'{o.name}: {o.message}.')
    if res.obs.get('weak') and not res.obs.get('degenerate'):
        add('warning', res.obs['message'])
    if res.T is not None and res.verdict != 'unobservable':
        sg = res.sigma
        if sg['position_mm'] > st.pos_tol_mm / 2 or sg['angle_deg'] > st.ang_tol_deg / 2:
            add('warning', f'Uncertain: ±{sg["position_mm"]:.1f} mm / ±{sg["angle_deg"]:.2f}° (1σ) against a '
                           f'tolerance of {st.pos_tol_mm:g} mm / {st.ang_tol_deg:g}°: add board poses.')
        else:
            add('good', f'Determined to ±{sg["position_mm"]:.1f} mm / ±{sg["angle_deg"]:.3f}° (1σ) from '
                        f'{len(res.used)} poses.')
    for n in res.notes:
        add('warning', n)
    if res.truth is not None:
        add('good', f'Synthetic: the solve is {res.truth["error_position_mm"]:.1f} mm / '
                    f'{res.truth["error_rotation_deg"]:.3f}° from the true mounting.')
    order = {s: i for i, s in enumerate(SEVERITIES)}
    out.sort(key=lambda f: order.get(f['severity'], 9))
    res.findings = out


# =========================================================================== the study
def run(project, engine=None, settings=None, variant=None, progress=None, cancel=None, poses=None, captures=None):
    """Lidar–camera calibration of a variant (the active one by default): synthetic captures of the planned board
    poses (or the given poses), or the folder of real pairs (settings.source == 'folder'), or the given captures.
    Returns a LidarCalibResult (None when cancelled); raises ValueError when the variant has no usable pair."""
    st = settings or LidarCalibSettings.from_dict(getattr(project.analysis, 'lidar_calib', None))
    p = _variant_project(project, variant)
    v = p.variant
    if engine is None and captures is None and st.source == 'synthetic':
        from .analysis import Engine
        engine = Engine()
    cam, lid, share = pick_pair(p, v, st, engine)

    def sub(a, b):
        if progress is None:
            return None
        return lambda f, text='': progress(a + (b - a) * f, text)
    notes = []
    truth = None
    source = 'synthetic'
    if captures is None and st.source == 'folder':
        captures, notes = load_folder(st.folder, st)
        source = st.folder
        plan = []
    elif captures is None:
        if poses is None:
            poses, notes = plan_poses(p, cam, lid, st, engine, make_world(p, st.scene, st.far_m)
                                      if st.scene == 'parking' else None)
        plan = list(poses)
        captures = simulate(p, engine, cam, lid, plan, st, sub(0.0, 0.5), cancel)
        if captures is None:
            return None
        truth = true_lidar(lid, st)
    else:
        plan = [c.board for c in captures if c.board is not None]
        source = 'captures'
    return calibrate(p, cam, lid, captures, st, plan, sub(0.5 if source == 'synthetic' else 0.0, 1.0), cancel,
                     truth, source, share, notes, v)


def calibrate_folder(project, cam_tag, lidar_tag, folder, settings=None, progress=None, cancel=None, variant=None):
    """Batch entry: calibrate camera cam_tag to lidar lidar_tag from a folder of image + cloud pairs."""
    st = LidarCalibSettings.from_dict(dict((settings or LidarCalibSettings()).to_dict() if isinstance(
        settings, LidarCalibSettings) else (settings or {}), camera=cam_tag or '', lidar=lidar_tag or '',
        source='folder', folder=folder))
    return run(project, None, st, variant, progress, cancel)


# =========================================================================== apply
def apply_to_variant(variant, res: LidarCalibResult) -> bool:
    """Set the lidar's pose in variant to the measured one; False when there is nothing to apply."""
    if res.T is None or res.verdict in ('failed',):
        return False
    s = variant.sensor(res.lidar_uid) or next((x for x in variant.sensors if x.tag == res.lidar_tag and
                                               x.kind == 'lidar'), None)
    if s is None:
        return False
    m = res.measured
    s.x, s.y, s.z = (round(float(m[k]), 4) for k in ('x', 'y', 'z'))
    s.yaw, s.pitch, s.roll = round(wrap180(m['yaw']), 3), round(m['pitch'], 3), round(wrap180(m['roll']), 3)
    return True


def calibrated_variant(project, res: LidarCalibResult, base=None):
    """A copy of the variant (new uids) named '<name> · lidar calibrated' with the lidar at its measured pose,
    appended to the project and made active. None when there is nothing to apply."""
    from .project import Variant, new_uid
    cur = base or project.variant_by_uid(res.variant_uid) or project.variant
    v = Variant.from_dict(cur.to_dict())
    v.uid = new_uid()
    v.name = f'{cur.name} · lidar calibrated'
    cmap = {}
    for c in v.cameras:
        cmap[c.uid] = new_uid()
        c.uid = cmap[c.uid]
    for c in v.cameras:
        c.link_uid = cmap.get(c.link_uid, '')
    v.seams = {'|'.join(cmap.get(u, u) for u in k.split('|')): a for k, a in v.seams.items()}
    lid = v.sensor(res.lidar_uid) or next((x for x in v.sensors if x.tag == res.lidar_tag and x.kind == 'lidar'), None)
    for s in v.sensors:
        s.uid = new_uid()
    probe = copy.copy(res)
    probe.lidar_uid = lid.uid if lid is not None else ''
    if lid is None or not apply_to_variant(v, probe):
        return None
    project.variants.append(v)
    project.active_uid = v.uid
    return v


# =========================================================================== overlays
_RAMP = np.array([[0.27, 0.51, 0.97], [0.10, 0.80, 0.90], [0.10, 0.90, 0.55], [0.64, 0.99, 0.24],
                  [0.98, 0.80, 0.22], [0.96, 0.45, 0.10], [0.86, 0.15, 0.10]])     # far (blue) .. near (red)


def range_colors(r, lo, hi) -> np.ndarray:
    """RGB 0..255 (K, 3) for ranges r on a near (warm) .. far (blue) ramp between lo and hi."""
    t = 1.0 - np.clip((np.asarray(r, float) - lo) / max(hi - lo, 1e-6), 0, 1)
    x = t * (len(_RAMP) - 1)
    i = np.minimum(x.astype(int), len(_RAMP) - 2)
    f = (x - i)[:, None]
    return (_RAMP[i] * (1 - f) + _RAMP[i + 1] * f) * 255


def project_points(project, res: LidarCalibResult, k, which='solved'):
    """The lidar points of capture k in its image: (uv (K,2) image pixels, range (K,), index into the cloud) through
    the camera's lens model with the plan's or the solved extrinsic."""
    cap = res.captures[k]
    T = res.T if which == 'solved' and res.T is not None else res.T_plan
    cam = next((c for v in project.variants for c in v.cameras if c.uid == res.camera_uid), None)
    if cap.points is None or cap.image is None or cam is None:
        return np.zeros((0, 2)), np.zeros(0), np.zeros(0, int)
    intr = resolve(project.module_for(cam), project.lens_for(cam))
    h, w = cap.image.shape[:2]
    sx, sy = w / intr.width, h / intr.height
    P = cap.points[:, :3].astype(float)
    Q = P @ T[:3, :3].T + T[:3, 3]
    u, v, _, ok = intr.project_dirs(Q)
    idx = np.flatnonzero(ok)
    return np.c_[u[idx] * sx, v[idx] * sy], np.linalg.norm(P[idx], axis=1), idx


def _board_box(res, k, shape):
    """Square crop (x0, y0, size) round the board in capture k's image."""
    o = res.poses[k] if k < len(res.poses) else None
    h, w = shape[:2]
    if o is not None and o.cam is not None:
        uv = o.cam.uv
        cx, cy = uv.mean(0)
        span = max(np.ptp(uv[:, 0]), np.ptp(uv[:, 1])) * (res.settings.cols + 1) / max(res.settings.cols - 1, 1)
        size = int(min(max(span * 2.0, 160), min(h, w)))
    else:
        cx, cy, size = w / 2, h / 2, min(h, w)
    x0 = int(min(max(cx - size / 2, 0), w - size))
    y0 = int(min(max(cy - size / 2, 0), h - size))
    return x0, y0, size


def overlay_image(project, res: LidarCalibResult, k, which='solved', crop=True, out=512) -> np.ndarray:
    """RGB uint8: capture k's image (dimmed grey) with the lidar points drawn through the plan's or the solved
    extrinsic, coloured by range; cropped round the board (out × out) or the whole image."""
    import cv2
    cap = res.captures[k]
    img = cap.image
    g = _gray(img).astype(np.float32) * 0.6
    base = np.repeat(g[..., None], 3, 2)
    uv, rng, _ = project_points(project, res, k, which)
    if crop:
        x0, y0, size = _board_box(res, k, img.shape)
        sc = out / size
        base = cv2.resize(base[y0:y0 + size, x0:x0 + size], (out, out), interpolation=cv2.INTER_AREA)
        uv = (uv - [x0, y0]) * sc
    else:
        sc = 1.0
    H_, W_ = base.shape[:2]
    inb = (uv[:, 0] >= 0) & (uv[:, 0] < W_ - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < H_ - 1)
    uv, rng = uv[inb], rng[inb]
    o = res.poses[k] if k < len(res.poses) else None
    if len(rng):
        ref = float(np.linalg.norm(o.lid.points.mean(0))) if o is not None and o.lid is not None else \
            float(np.median(rng))
        col = range_colors(rng, ref - 2.0, ref + 4.0)
        order = np.argsort(-rng)                          # near points drawn last (on top)
        xi = np.round(uv[order, 0]).astype(int)
        yi = np.round(uv[order, 1]).astype(int)
        cc = col[order]
        r = 1 if crop else 0
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                base[np.clip(yi + dy, 0, H_ - 1), np.clip(xi + dx, 0, W_ - 1)] = cc
    if o is not None and o.cam is not None:
        W, H = res.settings.board_size()
        cam = next((c for v in project.variants for c in v.cameras if c.uid == res.camera_uid), None)
        if cam is not None:
            intr = resolve(project.module_for(cam), project.lens_for(cam))
            P = o.cam.c + outline_local(W, H, 12) @ o.cam.R.T
            u, v, _, ok = intr.project_dirs(P)
            sx, sy = img.shape[1] / intr.width, img.shape[0] / intr.height
            q = np.c_[u * sx, v * sy]
            if crop:
                q = (q - [x0, y0]) * sc
            pts = np.round(np.vstack([q, q[:1]]) * 4).astype(np.int32)
            cv2.polylines(base, [pts.reshape(-1, 1, 2)], False, (255, 255, 255), 1, cv2.LINE_AA, shift=2)
    return np.clip(base, 0, 255).astype(np.uint8)


def overlay_png(project, res: LidarCalibResult, k) -> bytes:
    """Plan | solved side by side, cropped round the board, with a caption: the visual proof for pose k."""
    import cv2
    panels = []
    for which, title in (('plan', 'Plan'), ('solved', 'Solved')):
        im = overlay_image(project, res, k, which, True, 512)
        head = np.full((30, 512, 3), 24, np.uint8)
        cv2.putText(head, f'{title} - {res.captures[k].name} - {res.lidar_tag} in {res.camera_tag}', (10, 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 230), 1, cv2.LINE_AA)
        panels.append(np.vstack([head, im]))
    gap = np.full((542, 6, 3), 24, np.uint8)
    out = np.hstack([panels[0], gap, panels[1]])
    ok, buf = cv2.imencode('.png', cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    return buf.tobytes()


# =========================================================================== output
def _r(v, nd=3):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if math.isfinite(v) else None


def _rd(d: dict, nd=3) -> dict:
    return {k: _r(v, nd) for k, v in d.items()}


def _T_doc(T):
    x, y, z, w = quat_from_matrix(T[:3, :3])
    return {'matrix': [[round(float(v), 9) for v in row] for row in T],
            'translation_m': [round(float(v), 6) for v in T[:3, 3]],
            'quaternion_xyzw': [round(v, 9) for v in (x, y, z, w)]}


def summary(res: LidarCalibResult) -> dict:
    """Compact figures (the assistant, the batch log)."""
    out = {'camera': res.camera_tag, 'lidar': res.lidar_tag, 'variant': res.variant, 'source': res.source,
           'verdict': res.verdict, 'message': res.message,
           'poses': {'total': len(res.poses), 'used': len(res.used),
                     'set_aside': [o.name for o in res.poses if o.status == 'outlier'],
                     'not_found': [f'{o.name} ({o.status})' for o in res.poses
                                   if o.status in ('no_camera', 'no_lidar', 'unreadable')]},
           'tolerance': {'position_mm': res.settings.pos_tol_mm, 'angle_deg': res.settings.ang_tol_deg}}
    if res.T is not None:
        out['delta_vehicle'] = _rd(res.delta['vehicle'], 2)
        out['delta_position_mm'] = _r(res.delta['position_mm'], 2)
        out['delta_rotation_deg'] = _r(res.delta['rotation_deg'], 3)
        out['sigma_vehicle'] = _rd(res.sigma['vehicle'], 3)
        out['lidar_pose_measured'] = _rd(res.measured, 4)
        out['lidar_pose_planned'] = _rd(res.planned, 4)
        out['not_determined'] = [w['what'] for w in res.obs.get('weak', []) if w['level'] == 'none']
        out['weakly_determined'] = [w['what'] for w in res.obs.get('weak', []) if w['level'] == 'weak']
    if res.truth is not None:
        out['synthetic_error'] = {'position_mm': _r(res.truth['error_position_mm'], 2),
                                  'rotation_deg': _r(res.truth['error_rotation_deg'], 4)}
    out['findings'] = [f['message'] for f in res.findings if f['severity'] != 'good']
    return out


def _pose_doc(o: PoseObs):
    d = {'name': o.name, 'status': o.status, 'message': o.message}
    if o.cam is not None:
        d['camera'] = {'corners': int(len(o.cam.uv)), 'corner_rms_px': _r(o.cam.rms_px, 3), 'found': o.cam.how,
                       'board_centre_m': [_r(v, 4) for v in o.cam.c],
                       'board_normal': [_r(v, 5) for v in o.cam.R[:, 2]],
                       'distance_m': _r(np.linalg.norm(o.cam.c), 3)}
    if o.lid is not None:
        d['lidar'] = {'points': int(len(o.lid.points)), 'plane_rms_mm': _r(o.lid.rms_m * 1e3, 2),
                      'rings': int(o.lid.rings), 'edge_points_inside': int(len(o.lid.edge_in)),
                      'edge_points_outside': int(len(o.lid.edge_out)), 'found': o.lid.how}
    if o.stats:
        d['residuals'] = {'corner_rms_px': _r(o.stats['corner_rms_px'], 3),
                          'plane_rms_mm': _r(o.stats['plane_rms_mm'], 2),
                          'offset_from_camera_board_mm': _r(o.stats['offset_mm'], 2),
                          'angle_to_camera_board_deg': _r(o.stats['angle_deg'], 3),
                          'edge_bracket_rms_mm': _r(o.stats['edge_mm'], 2)}
    if o.planned is not None:
        d['planned_board'] = o.planned.to_dict()
    return d


def to_doc(res: LidarCalibResult, project=None) -> dict:
    """The JSON export: the pair, the board, every pose, the solution with its 1σ and Δ against the plan, the
    verdict."""
    st = res.settings
    W, H = st.board_size()
    doc = {
        'schema': SCHEMA, 'project': getattr(project, 'name', ''), 'variant': res.variant,
        'frame': FRAME, 'camera_frame': OPTICAL, 'lidar_frame': 'the lidar\'s own frame (x forward, y left, z up)',
        'units': 'metres and degrees unless a key says mm or px',
        'pair': {'camera': res.camera_tag, 'lidar': res.lidar_tag, 'image_overlap_share': _r(res.share, 3),
                 'camera_pose': _rd(res.camera_pose, 4),
                 'camera_pose_is': 'the camera\'s measured as-built pose' if res.as_built else
                 'the camera\'s planned pose'},
        'board': {'inner_corners': [st.cols, st.rows], 'square_mm': st.square_mm, 'margin_mm': st.margin_mm,
                  'outer_size_m': [round(W, 4), round(H, 4)]},
        'source': res.source,
        'settings': st.to_dict(),
        'assumptions': ASSUMPTIONS,
        'poses': [_pose_doc(o) for o in res.poses],
        'verdict': res.verdict, 'message': res.message,
        'tolerance': {'position_mm': st.pos_tol_mm, 'angle_deg': st.ang_tol_deg},
        'T_camera_lidar_plan': _T_doc(res.T_plan),
        'lidar_pose_planned': _rd(res.planned, 5),
        'findings': res.findings, 'notes': res.notes,
    }
    if res.T is not None:
        doc['solution'] = {
            'T_camera_lidar': dict(_T_doc(res.T), maps='lidar points into the ' + OPTICAL),
            'T_lidar_camera': _T_doc(_inv(res.T)),
            'sigma_1': {'camera_frame': _rd(res.sigma['camera'], 4), 'vehicle_frame': _rd(res.sigma['vehicle'], 4)},
            'covariance': {'parameters': 'rotation vector (rad) then translation (m), optical frame',
                           'matrix': [[float(f'{v:.6g}') for v in row] for row in res.cov]},
            'delta_vs_plan': {'camera_frame': _rd(res.delta['camera'], 3),
                              'vehicle_frame': _rd(res.delta['vehicle'], 3),
                              'largest_position_mm': _r(res.delta['position_mm'], 3),
                              'rotation_deg': _r(res.delta['rotation_deg'], 4)},
            'lidar_pose_measured': _rd(res.measured, 5),
            'observability': {'weak_directions_of_the_planes': res.obs.get('weak', []),
                              'message': res.obs.get('message', ''),
                              'sigma_by_direction_mm': [_r(s * 1e3, 3) for s in res.obs.get('sigma_m', [])],
                              'edges_used': res.obs.get('edges', False)},
            'fit': {k: _r(v, 4) if isinstance(v, float) else v for k, v in res.fit.items()},
        }
    if res.truth is not None:
        doc['synthetic_truth'] = {'lidar_pose': _rd(res.truth['pose'], 5), 'injected_error': res.truth['injected'],
                                  'solve_error_vehicle': _rd(res.truth['error'], 4),
                                  'solve_error_position_mm': _r(res.truth['error_position_mm'], 3),
                                  'solve_error_rotation_deg': _r(res.truth['error_rotation_deg'], 5)}
    return doc


def files(res: LidarCalibResult, project=None) -> list:
    """[(name, bytes)]: lidar_camera_calibration.json and <pose>_overlay.png per pose with the board found."""
    out = [('lidar_camera_calibration.json', (json.dumps(to_doc(res, project), indent=1, ensure_ascii=False)
                                              + '\n').encode('utf-8'))]
    if project is not None:
        for k, o in enumerate(res.poses):
            if o.cam is not None and k < len(res.captures) and res.captures[k].image is not None:
                out.append((f'{o.name}_overlay.png', overlay_png(project, res, k)))
    return out


def export(res: LidarCalibResult, project, out_dir) -> list[str]:
    """Write the JSON and the overlays into out_dir; returns their paths."""
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for name, data in files(res, project):
        path = os.path.join(out_dir, name)
        with open(path, 'wb') as fh:
            fh.write(data)
        paths.append(path)
    return paths
