"""Calibration observability and camera–lidar fusion: before anything is captured, how firmly the planned calibration
pins every sensor's pose, which directions stay weak and why, what moving or adding a board would gain; and how densely
the lidars cover what each camera sees.

Cameras
- Boards: the project's calibration targets (Calibrate ▸ Targets), flat on the floor where the plan puts them. The
  corners a detector finds: a checkerboard's inner corners, the whole board in view (the detector needs the full grid,
  as the installation check's does); a ChArUco board's inner corners one by one (its markers make every corner
  identifiable, so a board partly out of view still counts, from four corners); an AprilGrid (Kalibr) rows × cols tags
  of square / (1 + 0.3), four corners each, tag by tag. A corner counts when it lies in the image and the lens circle,
  nothing of the body, the rig or a housing hides it (the engine's occlusion maps) and the squares (tags) round it are
  at least min_square_px wide in the image: the foreshortened side, so a board seen at a grazing angle drops out.
- Fisher information: every counted corner's two pixel coordinates through the camera's lens model (its unit
  calibration when it has one), with independent pixel noise σ on each, differentiated with respect to the camera's
  pose (three small rotations about its own axes, three shifts) and, with the boards placed by hand, each board's x, y
  and yaw on the floor, held near the plan by the placement σ: the model of the installation check's joint solve
  (core.joint), one information matrix over every camera and board. That solve weights a corner as 0.5 px whatever
  the real noise, so the covariance is the one of that estimator, H⁻¹ M H⁻¹ (H at its weights, M at the real noise;
  the same as H⁻¹ when σ is 0.5 px). With the boards taken as exact the cameras separate and the weights cancel.
- Reported per camera: 1σ of yaw, pitch and roll (degrees: the variant's Euler angles) and x, y, z (mm, vehicle frame);
  the same from the corners alone (the boards exact); the weakest direction (the covariance's longest axis with every
  rotation counted as the shift it makes at the boards' median distance) and the condition number of the information
  on that scale; per board its share of the camera's information and the camera's worst axis without it. A camera is
  judged by its worst axis against the requirement (1σ need_deg and need_mm).
- Camera pairs: the relative pose of every two cameras that see a common board (camera b in camera a's own axes) from
  the joint covariance: a shared board ties two cameras even where their poses in the vehicle frame are loose.
- Why: the weak axes explained from the geometry (corners near the image centre leave the roll loose; boards that
  cover a narrow part of the view let a turn and a shift look alike; the placement σ dominating what the corners give)
  and suggestions computed by trying moves: each weak camera's boards (and the boards near its view) moved closer,
  farther, sideways or turned, a board added where the camera sees a whole one, and the boards laid four times more
  precisely, each evaluated again over every camera; kept when it helps the camera by 10 % or more and makes no other
  camera more than 5 % worse.
- Monte Carlo (verify): noisy detections of exactly the corners counted, boards laid off their plan by draws of the
  placement σ, solved with the installation check's joint solve (core.joint.joint_solve, its default weights); the
  spread of the solved poses (and of the pairs' relative poses) against the predicted σ.

Lidars: each camera–lidar pair whose views overlap (core.lidar_calib.pairs) with the lidar–camera calibration's own
board plan (core.lidar_calib.plan_poses with the project's lidar–camera settings: boards held up at several poses
that both see). Information: the camera's corners on each board (σ px) and the lidar's points on it, every beam of
the model that meets the board (range noise along the beam, so σ_r × the cosine of the incidence across the plane),
over the extrinsic T_camera_lidar and every board's pose, the boards marginalised; as the calibration's solve weights
(a board's points at their RMS noise, at most 2 500 a pose). The board edges and the checker in the reflectivity are
left out (they only add). 1σ of the lidar's pose in the vehicle frame against the lidar–camera tolerances; verified
by Monte Carlo with the calibration's own solve (core.lidar_calib._solve) on noisy corners and points.

Radars: corner reflectors you place (a trihedral of edge a: RCS 4πa⁴ / 3λ² at 77 GHz). A radar measures each
reflector in its field (its zones for that cross-section, nothing of the vehicle in between) as a range and an
azimuth (typical noise, settings); its x, y and yaw follow (z, pitch and roll are not observable without elevation);
the reflectors' surveyed positions carry their own σ. Verified by Monte Carlo with a least-squares solve.

Fusion density (core.sensor_coverage.camera_fusion): per camera, lidar points and scan lines on each object class in
the camera's view out to the required range, points per camera pixel, and where the camera has no lidar support.
"""
from __future__ import annotations

import copy
import csv
import io
import json
import math
import os
import threading
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from . import sensors as S
from .lens import resolve
from .mathutil import BODY_TO_OPTICAL, euler_zyx, rot_z, rot_zyx, wrap180

SCHEMA = 'vantage.observability/1'
FRAME = 'vehicle (ISO 8855, rear axle)'
PATTERNS = ('checkerboard', 'charuco', 'aprilgrid')
APRIL_SPACING = 0.3                  # Kalibr AprilGrid: the gap between tags over the tag's size (its default)
CHARUCO_MIN = 4                      # corners a ChArUco board needs to count
JOINT_PIXEL_SIGMA = 0.5              # the weight core.joint.joint_solve gives a corner (its default), px
AXES = ('yaw', 'pitch', 'roll', 'x', 'y', 'z')
SIGMA_KEYS = ('yaw_deg', 'pitch_deg', 'roll_deg', 'x_mm', 'y_mm', 'z_mm')
REL_KEYS = ('roll_deg', 'pitch_deg', 'yaw_deg', 'x_mm', 'y_mm', 'z_mm')   # about / along camera a's own axes
SEVERITIES = ('critical', 'serious', 'warning', 'good')
UNDETERMINED = 100.0                 # a σ this many times the requirement: not determined at all
LAMBDA_77GHZ = 299792458.0 / 77e9    # radar wavelength, m
FASCIA_M = 0.15                      # radar: occluder hits this close are the fascia it sits behind
MOVE_STEPS = (('closer', 0.5), ('closer', 0.25), ('farther', 0.5), ('left', 0.5), ('right', 0.5), ('left', 1.0),
              ('right', 1.0), ('turn', 45.0), ('turn', -45.0))
GAIN_MIN = 0.10                      # a suggestion must improve its camera's worst axis by this share
ISSUE_RATIO = 2.0                    # the Issues list names a camera whose worst axis is beyond this × the requirement
MC_LIDAR_PAIRS = 2                   # lidar–camera pairs the Monte Carlo verifies: the best overlap and the weakest
COST_MAX = 0.05                      # and make no other camera worse by more than this share
BOARD_GAP_M = 0.05                   # free floor between two boards
ASSUMPTIONS = (
    'Predictions from the Fisher information of the planned geometry, not measurements: the lens models are taken as '
    'exact, corner noise as independent and equal on every corner (a corner far out in a fisheye is noisier in '
    'practice), boards as flat and printed true. Boards lie on the floor where the plan puts them; with "boards placed '
    'by hand" each may sit off by the placement σ in x, y and yaw (the installation check\'s joint solve model). The '
    'installation check itself detects checkerboards; ChArUco and AprilGrid corners are counted as their own '
    'detectors find them (partial boards), for a toolchain that uses them. Lidar–camera figures leave out the board '
    'edges and the reflectivity checker (they only improve them). Radar noise figures are typical for 77 GHz '
    'automotive radars (range about 0.1 m, azimuth 0.1–0.3° near boresight, worse at the edge of the field), not a '
    'product\'s datasheet: enter yours.')
# accepted range of every number in the settings (values outside are clamped)
LIMITS = {'pixel_sigma': (0.01, 10.0), 'board_sigma_mm': (0.1, 500.0), 'board_sigma_deg': (0.01, 30.0),
          'min_square_px': (1.0, 100.0), 'need_mm': (0.01, 1000.0), 'need_deg': (0.001, 30.0),
          'min_share': (0.0, 1.0), 'radar_range_sigma_m': (0.001, 10.0), 'radar_az_sigma_deg': (0.001, 20.0),
          'reflector_sigma_mm': (0.0, 500.0), 'radar_need_mm': (0.1, 5000.0), 'radar_need_deg': (0.001, 30.0),
          'mc_trials': (5, 1000), 'seed': (0, 999999), 'fusion_at_m': (2.0, 200.0)}
INT_FIELDS = ('mc_trials', 'seed')
BOOL_FIELDS = ('boards_placed', 'lidar_pairs', 'suggest', 'fusion')
REFLECTOR_LIMITS = {'x': (-200.0, 200.0), 'y': (-200.0, 200.0), 'z': (-5.0, 20.0), 'edge_mm': (10.0, 1000.0)}
MAX_REFLECTORS = 32


# =========================================================================== settings
def clean_reflector(r) -> dict | None:
    """A corner reflector as stored ({name, x, y, z, edge_mm}), or None when it is not a usable one."""
    try:
        o = {'name': str(r.get('name') or 'CR')[:24], 'x': float(r['x']), 'y': float(r['y']),
             'z': float(r.get('z', 0.5)), 'edge_mm': float(r.get('edge_mm', 100.0))}
    except (KeyError, TypeError, ValueError, AttributeError):
        return None
    for k, (lo, hi) in REFLECTOR_LIMITS.items():
        if not math.isfinite(o[k]):
            return None
        o[k] = min(max(o[k], lo), hi)
    return o


def reflector_rcs(edge_mm: float) -> float:
    """Peak radar cross-section of a triangular trihedral corner reflector of inner edge a at 77 GHz: 4πa⁴ / 3λ², m²."""
    a = float(edge_mm) / 1000.0
    return 4.0 * math.pi * a ** 4 / (3.0 * LAMBDA_77GHZ ** 2)


@dataclass
class ObservabilitySettings:
    pixel_sigma: float = 0.3           # corner noise, px 1σ per image axis (sub-pixel corner refinement, typical)
    boards_placed: bool = True         # boards laid by hand: x, y, yaw known within the placement σ (joint solve)
    board_sigma_mm: float = 20.0       # the installation check's default placement allowance
    board_sigma_deg: float = 1.0
    min_square_px: float = 8.0         # a square (tag) narrower than this in the image is not found
    need_mm: float = 10.0              # 1σ a camera's position must reach (a typical data-rig target)
    need_deg: float = 0.1              # 1σ its angles must reach
    lidar_pairs: bool = True           # the lidar–camera pairs (the lidar–camera calibration's board plan)
    min_share: float = 0.05            # a camera–lidar pair overlapping less of the image is left out
    radar_range_sigma_m: float = 0.10  # typical 77 GHz radar range noise, 1σ
    radar_az_sigma_deg: float = 0.3    # typical azimuth noise near boresight, 1σ
    reflector_sigma_mm: float = 5.0    # how well the reflectors' positions are surveyed, 1σ per axis
    radar_need_mm: float = 20.0
    radar_need_deg: float = 0.25
    reflectors: list = field(default_factory=list)     # [{name, x, y, z, edge_mm}] corner reflectors, vehicle frame
    suggest: bool = True               # try moves and added boards for the weak cameras
    fusion: bool = True                # the camera–lidar fusion density
    fusion_at_m: float = 20.0          # the image map's scene distance
    mc_trials: int = 30
    seed: int = 1

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d: dict | None) -> 'ObservabilitySettings':
        s = ObservabilitySettings()
        d = d if isinstance(d, dict) else {}
        for k in BOOL_FIELDS:
            v = d.get(k)
            if isinstance(v, (bool, int)) and not isinstance(v, float):
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
        refl = d.get('reflectors')
        if isinstance(refl, list):
            s.reflectors = [r for r in map(clean_reflector, refl[:MAX_REFLECTORS]) if r is not None]
        return s


def settings_of(project) -> ObservabilitySettings:
    return ObservabilitySettings.from_dict(getattr(project.analysis, 'observability', None))


# =========================================================================== small geometry
def _exp(w) -> np.ndarray:
    """Rotation vector -> matrix."""
    w = np.asarray(w, float)
    th = float(np.linalg.norm(w))
    if th < 1e-15:
        return np.eye(3)
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def _log(R) -> np.ndarray:
    """Matrix -> rotation vector (angles below 180°)."""
    c = max(-1.0, min(1.0, (float(np.trace(R)) - 1) / 2))
    th = math.acos(c)
    if th < 1e-12:
        return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return v * (th / (2 * math.sin(th)))


def finite_pose(o) -> bool:
    """True when a camera's or sensor's mounting pose (x y z yaw pitch roll) is all finite numbers."""
    return all(math.isfinite(float(getattr(o, k))) for k in ('x', 'y', 'z', 'yaw', 'pitch', 'roll'))


def _euler_jac(R) -> np.ndarray | None:
    """d(yaw, pitch, roll) degrees / d(rotation vector about the body's own axes) radians at R; None within 2° of
    looking straight up or down (yaw and roll are then one rotation)."""
    e0 = euler_zyx(R)
    if abs(e0[1]) > 88.0:
        return None
    J = np.zeros((3, 3))
    h = 1e-6
    for k in range(3):
        d = np.zeros(3)
        d[k] = h
        a = np.array(euler_zyx(R @ _exp(d)))
        b = np.array(euler_zyx(R @ _exp(-d)))
        J[:, k] = ((a - b + 180.0) % 360.0 - 180.0) / (2 * h)
    return J


def _to_euler(cov6, R) -> tuple[np.ndarray, bool]:
    """A camera pose covariance (body rotation vector rad, position m) as (yaw pitch roll deg, x y z mm), and whether
    the angles are the Euler angles (False: rotations about the camera's own z, y, x axes, near straight down)."""
    Je = _euler_jac(R)
    T = np.zeros((6, 6))
    if Je is None:
        T[0, 2] = T[1, 1] = T[2, 0] = math.degrees(1.0)
    else:
        T[:3, :3] = Je
    T[3:, 3:] = np.eye(3) * 1000.0
    return T @ cov6 @ T.T, Je is not None


def _sig(cov, keys=SIGMA_KEYS) -> dict:
    """1σ per axis; beyond a million (degrees or millimetres) a direction nothing pins: inf."""
    d = np.sqrt(np.maximum(np.diag(cov), 0.0))
    return {k: (float(v) if math.isfinite(v) and v < 1e6 else math.inf) for k, v in zip(keys, d)}


def _ratio(sig: dict, need_deg, need_mm) -> tuple[float, str]:
    """(worst σ in units of the requirement, the axis)."""
    best = (0.0, '')
    for k, v in sig.items():
        r = v / (need_deg if k.endswith('_deg') else need_mm)
        if not math.isfinite(r):
            return math.inf, k.split('_')[0]
        if r > best[0]:
            best = (r, k.split('_')[0])
    return best


def _verdict(ratio) -> str:
    if not math.isfinite(ratio) or ratio >= UNDETERMINED:
        return 'critical'
    return 'good' if ratio <= 1.0 + 1e-9 else 'warning' if ratio <= 2.0 else 'serious'


def _inv(H) -> tuple[np.ndarray, bool]:
    """Inverse of a symmetric information matrix; pinv (and False) when it is singular."""
    H = (H + H.T) / 2
    try:
        c = np.linalg.cond(H)
        if math.isfinite(c) and c < 1e14:
            return np.linalg.inv(H), True
    except np.linalg.LinAlgError:
        pass
    return np.linalg.pinv(H, rcond=1e-12), False


# =========================================================================== boards
@dataclass
class _Board:
    t: object                          # core.project.Target (a copy)
    P: np.ndarray                      # (N, 3) corners on the floor
    groups: np.ndarray                 # (N,) detection group: a corner counts only with its whole group
    grid: tuple | None                 # (rows, cols) of a corner grid (checkerboard, ChArUco); None: AprilGrid tags
    pattern: str
    centre: np.ndarray

    @property
    def diag(self) -> float:
        w, h = self.t.size_m
        return math.hypot(w, h)


def board_corners(t) -> tuple[np.ndarray, np.ndarray, tuple | None, str]:
    """(corners (N, 3) on the floor, detection group per corner, (rows, cols) of a corner grid or None, pattern) of
    a target: inner corners for a checkerboard (one group: the whole board) and a ChArUco board (a group per corner);
    four corners per tag for an AprilGrid (a group per tag)."""
    pattern = t.pattern if t.pattern in PATTERNS else 'checkerboard'
    c, s = math.cos(math.radians(t.yaw)), math.sin(math.radians(t.yaw))
    sq = t.square_mm / 1000.0
    rows, cols = max(int(t.rows), 2), max(int(t.cols), 2)
    if pattern == 'aprilgrid':
        tag = sq / (1.0 + APRIL_SPACING)
        a = (np.arange(cols) - (cols - 1) / 2) * sq
        b = (np.arange(rows) - (rows - 1) / 2) * sq
        A, B = np.meshgrid(a, b)
        off = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * tag / 2
        U = (A.ravel()[:, None] + off[None, :, 0]).ravel()
        V = (B.ravel()[:, None] + off[None, :, 1]).ravel()
        groups = np.repeat(np.arange(rows * cols), 4)
        grid = None
    else:
        w, h = cols * sq, rows * sq
        U, V = np.meshgrid(np.arange(1, cols) * sq - w / 2, np.arange(1, rows) * sq - h / 2)
        U, V = U.ravel(), V.ravel()
        groups = np.zeros(len(U), int) if pattern == 'checkerboard' else np.arange(len(U))
        grid = (rows - 1, cols - 1)
    P = np.stack([t.x + U * c - V * s, t.y + U * s + V * c, np.zeros(len(U))], 1)
    return P, groups, grid, pattern


def _board(t) -> _Board:
    P, g, grid, pat = board_corners(t)
    return _Board(copy.copy(t), P, g, grid, pat, np.array([float(t.x), float(t.y), 0.0]))


def _outline(t) -> np.ndarray:
    w, h = t.size_m
    c, s = math.cos(math.radians(t.yaw)), math.sin(math.radians(t.yaw))
    pts = np.array([(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)])
    return np.c_[t.x + pts[:, 0] * c - pts[:, 1] * s, t.y + pts[:, 0] * s + pts[:, 1] * c, np.zeros(4)]


# =========================================================================== cameras
@dataclass
class _Cam:
    cam: object
    intr: object
    R: np.ndarray                      # body -> vehicle
    pos: np.ndarray
    W2O: np.ndarray
    cube: object = None

    def project(self, P, W2O=None, pos=None):
        W2O = self.W2O if W2O is None else W2O
        pos = self.pos if pos is None else pos
        u, v, th, ok = self.intr.project_dirs((np.asarray(P, float) - pos) @ W2O.T)
        return np.stack([u, v], -1), ok, th


@dataclass
class _Seen:
    """A board as one camera sees it."""
    cam: int
    board: int
    idx: np.ndarray                    # the counted corners (indices into the board's corners)
    uv: np.ndarray                     # (n, 2) their pixels
    Jc: np.ndarray                     # (2n, 6) d pixels / d camera (body rotation vector rad, position m)
    Jb: np.ndarray                     # (2n, 3) d pixels / d board (dx, dy m, dyaw rad)
    dist: float                        # median corner distance, m
    feature_px: float                  # the smallest square (tag) side among them, px

    @property
    def n(self):
        return len(self.idx)


def _feature_px(uv, b: _Board) -> np.ndarray:
    """Per corner, the shortest side of the squares (tags) round it in the image, px."""
    if b.grid is not None:
        r, c = b.grid
        G = uv.reshape(r, c, 2)
        out = np.full((r, c), np.inf)
        if c > 1:
            e = np.linalg.norm(np.diff(G, axis=1), axis=-1)
            out[:, :-1] = np.minimum(out[:, :-1], e)
            out[:, 1:] = np.minimum(out[:, 1:], e)
        if r > 1:
            e = np.linalg.norm(np.diff(G, axis=0), axis=-1)
            out[:-1] = np.minimum(out[:-1], e)
            out[1:] = np.minimum(out[1:], e)
        return out.ravel()
    T = uv.reshape(-1, 4, 2)
    e = np.linalg.norm(T - np.roll(T, -1, axis=1), axis=-1).min(1)
    return np.repeat(e, 4)


def _jacobians(cm: _Cam, P, centre):
    """(uv, Jc (2n, 6), Jb (2n, 3)) of corners P: pixel derivatives with respect to the camera's pose (rotation vector
    about its own axes, then its position) and the board's placement on the floor (dx, dy, dyaw about centre)."""
    h = 1e-6
    uv, _, _ = cm.project(P)
    cols = []
    for k in range(3):
        d = np.zeros(3)
        d[k] = h
        a, _, _ = cm.project(P, BODY_TO_OPTICAL @ (cm.R @ _exp(d)).T)
        b, _, _ = cm.project(P, BODY_TO_OPTICAL @ (cm.R @ _exp(-d)).T)
        cols.append(((a - b) / (2 * h)).ravel())
    dpos = []
    for k in range(3):
        d = np.zeros(3)
        d[k] = h
        a, _, _ = cm.project(P, pos=cm.pos + d)
        b, _, _ = cm.project(P, pos=cm.pos - d)
        dpos.append((a - b) / (2 * h))               # (n, 2): d uv / d camera position = - d uv / d corner
    Jc = np.c_[np.stack(cols, 1), np.stack([x.ravel() for x in dpos], 1)]
    q = P - centre
    dyaw = -(dpos[0] * (-q[:, 1:2]) + dpos[1] * q[:, 0:1])          # corner moves by (−qy, qx) per radian
    Jb = np.stack([-dpos[0].ravel(), -dpos[1].ravel(), dyaw.ravel()], 1)
    return uv, Jc, Jb


def _see(cm: _Cam, ci, b: _Board, bi, st: ObservabilitySettings):
    """(_Seen or None, reason): the corners of board b that camera cm counts."""
    uv, ok, _ = cm.project(b.P)
    if not ok.any():
        return None, 'out of view'
    d = b.P - cm.pos
    dist = np.linalg.norm(d, axis=1)
    if cm.cube is not None:
        depth = cm.cube.lookup(d)[0]
        clear = depth > dist - (0.02 + 0.01 * dist)
    else:
        clear = np.ones(len(d), bool)
    feat = _feature_px(uv, b)
    use = ok & clear & (feat >= st.min_square_px)
    # a corner counts only with its whole group (the board, a tag)
    ng = int(b.groups.max()) + 1
    bad = np.bincount(b.groups, weights=(~use).astype(float), minlength=ng) > 0
    use &= ~bad[b.groups]
    if b.pattern == 'checkerboard' and use.all():
        _, ok2, _ = cm.project(_outline(b.t))
        if not ok2.all():
            use[:] = False
    need = CHARUCO_MIN if b.pattern == 'charuco' else 1
    if use.sum() < need:
        if not (ok & clear).any():
            return None, 'hidden by the vehicle' if ok.any() and not clear[ok].any() else 'out of view'
        if not ok.all() and b.pattern == 'checkerboard':
            return None, 'partly out of view'
        if (ok & ~clear).any() and b.pattern == 'checkerboard':
            return None, 'partly hidden by the vehicle'
        if (feat[ok & clear] < st.min_square_px).any():
            return None, f'squares under {st.min_square_px:g} px'
        return None, 'too few corners in view'
    idx = np.flatnonzero(use)
    uvi, Jc, Jb = _jacobians(cm, b.P[idx], b.centre)
    return _Seen(ci, bi, idx, uvi, Jc, Jb, float(np.median(dist[idx])), float(feat[idx].min())), ''


# =========================================================================== the joint covariance
def _joint(seen: list, n_cams, n_boards, st: ObservabilitySettings, placed=None, sb_mm=None, sb_deg=None,
           drop=None):
    """Covariance over every camera's pose (6 each: rotation vector about its own axes rad, position m) and, with the
    boards placed by hand, every board's (dx, dy m, dyaw rad): the joint solve's estimator, H⁻¹ M H⁻¹. drop: a
    (camera, board) observation to leave out. Returns (C, determined per camera)."""
    placed = st.boards_placed if placed is None else placed
    sb = (st.board_sigma_mm if sb_mm is None else sb_mm) / 1000.0
    sy = math.radians(st.board_sigma_deg if sb_deg is None else sb_deg)
    nb = n_boards if placed else 0
    n = 6 * n_cams + 3 * nb
    if n == 0:
        return np.zeros((0, 0)), np.zeros(0, bool)
    G = np.zeros((n, n))
    for o in seen:
        if drop is not None and (o.cam, o.board) == drop:
            continue
        c = slice(6 * o.cam, 6 * o.cam + 6)
        G[c, c] += o.Jc.T @ o.Jc
        if placed:
            b = slice(6 * n_cams + 3 * o.board, 6 * n_cams + 3 * o.board + 3)
            B = o.Jc.T @ o.Jb
            G[c, b] += B
            G[b, c] += B.T
            G[b, b] += o.Jb.T @ o.Jb
    det = np.array([G[6 * k:6 * k + 6, 6 * k:6 * k + 6].trace() > 0 for k in range(n_cams)], bool)
    P = np.zeros(n)
    if placed:
        P[6 * n_cams:] = np.tile([1 / sb ** 2, 1 / sb ** 2, 1 / sy ** 2], nb)
    w = JOINT_PIXEL_SIGMA ** 2 if placed else 1.0
    s2 = st.pixel_sigma ** 2
    H = G / w + np.diag(P)
    M = G * (s2 / w ** 2) + np.diag(P)
    for k in np.flatnonzero(~det):                       # a camera without corners stays out: a unit placeholder
        sl = slice(6 * k, 6 * k + 6)
        H[sl, sl] = np.eye(6)
        M[sl, sl] = np.eye(6)
    Hi, ok = _inv(H)
    C = Hi @ M @ Hi
    C = (C + C.T) / 2
    if not ok:                                           # singular: what the data does not pin is undetermined
        lam, V = np.linalg.eigh(H)
        null = V[:, lam < lam.max() * 1e-13]
        C = C + null @ null.T * 1e12
    return C, det


# =========================================================================== the study
@dataclass
class ObservabilityResult:
    settings: ObservabilitySettings
    variant: str
    variant_uid: str
    cameras: list                      # per visible camera: dict (see _camera_doc)
    boards: list                       # per target: dict
    pairs: list                        # camera pairs that share a board
    lidar_pairs: list                  # camera–lidar pairs
    radars: list                       # per visible radar
    fusion: list | None                # core.sensor_coverage.camera_fusion, None when off or no lidar
    findings: list                     # [{'severity', 'kind', 'message', 'tags', 'uids'}], worst first
    metrics: dict
    outline: list = field(default_factory=list)       # body outline polylines in plan (N, 2), for drawing
    mc: dict | None = None             # the last Monte Carlo verification (monte_carlo)
    _model: object = None              # the geometry the suggestions and the Monte Carlo need

    def camera(self, tag):
        return next((c for c in self.cameras if c['tag'] == tag), None)

    def camera_by_uid(self, uid):
        return next((c for c in self.cameras if c['uid'] == uid), None)

    def worst(self) -> str:
        return self.findings[0]['severity'] if self.findings else 'good'


class _Model:
    """Everything the study derived from the geometry, kept for the suggestions and the Monte Carlo."""

    def __init__(self, project, engine, v, st):
        self.project, self.engine, self.v, self.st = project, engine, v, st
        self.vg = engine.vehicle_geom(project, v) if engine is not None else None
        self.cams: list[_Cam] = []
        for c in v.cameras:
            if not c.visible or not finite_pose(c):
                continue
            intr = resolve(project.module_for(c), project.lens_for(c))
            R = rot_zyx(c.yaw, c.pitch, c.roll)
            pos = c.pos().astype(float)
            cube = engine.cube(pos, self.vg) if self.vg is not None else None
            self.cams.append(_Cam(c, intr, R, pos, BODY_TO_OPTICAL @ R.T, cube))
        self.boards = [_board(t) for t in project.targets]
        self.seen: list[_Seen] = []
        self.reasons = {}                      # (camera, board) -> why it does not count
        for ci, cm in enumerate(self.cams):
            for bi, b in enumerate(self.boards):
                o, why = _see(cm, ci, b, bi, st)
                if o is None:
                    self.reasons[(ci, bi)] = why
                else:
                    self.seen.append(o)
        self.grid = engine.roi_grid(project, None, v) if (engine is not None and self.vg is not None) else None

    def cov(self, seen=None, **kw):
        return _joint(self.seen if seen is None else seen, len(self.cams), len(self.boards), self.st, **kw)

    def ratios(self, C, det) -> list:
        st = self.st
        out = []
        for ci, cm in enumerate(self.cams):
            if not det[ci]:
                out.append(math.inf)
                continue
            ce, _ = _to_euler(C[6 * ci:6 * ci + 6, 6 * ci:6 * ci + 6], cm.R)
            out.append(_ratio(_sig(ce), st.need_deg, st.need_mm)[0])
        return out


def project_for(project, variant):
    """The project with variant active (a shallow copy when it is not the active one; the project itself when variant
    is None or already active). The copy shares everything but its variant list and active uid."""
    if variant is None or variant is project.variant:
        return project
    q = copy.copy(project)
    q.variants = list(project.variants) + ([] if any(v is variant for v in project.variants) else [variant])
    q.active_uid = variant.uid
    return q


def _fmt_sig(v, unit):
    if not math.isfinite(v):
        return 'not determined'
    return f'{v:.2f}°' if unit == 'deg' else f'{v:.1f} mm'


AXIS_WORDS = {'yaw': 'yaw', 'pitch': 'pitch', 'roll': 'roll', 'x': 'x (forward)', 'y': 'y (left)', 'z': 'z (up)'}


def _weakest(C6, R, dist) -> tuple[dict, float]:
    """The covariance's longest axis with rotations counted as the shift they make at dist, in natural units, and the
    condition number of the information on that scale."""
    D = np.diag([dist, dist, dist, 1.0, 1.0, 1.0])
    Cs = D @ C6 @ D
    lam, V = np.linalg.eigh((Cs + Cs.T) / 2)
    lam = np.maximum(lam, 1e-30)
    cond = float(lam[-1] / lam[0])
    v = V[:, -1] * math.sqrt(lam[-1])
    rot = v[:3] / dist
    Je = _euler_jac(R)
    ang = Je @ rot if Je is not None else np.degrees(rot[::-1])
    comp = {'yaw_deg': float(ang[0]), 'pitch_deg': float(ang[1]), 'roll_deg': float(ang[2]),
            'x_mm': float(v[3] * 1000), 'y_mm': float(v[4] * 1000), 'z_mm': float(v[5] * 1000)}
    # the parts that matter, in metres at the boards: biggest first
    mag = {k: abs(math.radians(comp[k])) * dist for k in ('yaw_deg', 'pitch_deg', 'roll_deg')}
    mag.update(x_mm=abs(v[3]), y_mm=abs(v[4]), z_mm=abs(v[5]))
    top = max(mag.values()) or 1.0
    parts = [k for k in sorted(mag, key=lambda k: -mag[k]) if mag[k] >= 0.3 * top][:3]
    words = []
    for k in parts:
        val = comp[k]
        name = AXIS_WORDS[k.split('_')[0]]
        words.append(f'{name} {val:+.2f}°' if k.endswith('_deg') else f'{name} {val:+.1f} mm')
    text = ' with '.join([', '.join(words[:1])] + ([', '.join(words[1:])] if len(words) > 1 else []))
    return {'components': {k: round(comp[k], 4) for k in parts}, 'text': text,
            'one_sigma_m': float(math.sqrt(lam[-1]))}, cond


def _explain(m: _Model, ci, sig, sig_c, ratio_axes, st) -> list:
    """Plain words for each weak axis, from the geometry of the corners the camera counts."""
    cm = m.cams[ci]
    mine = [o for o in m.seen if o.cam == ci]
    out = []
    if not mine:
        return out
    uv = np.concatenate([o.uv for o in mine])
    intr = cm.intr
    half = math.hypot(intr.width, intr.height) / 2
    r = np.hypot(uv[:, 0] - intr.cx, uv[:, 1] - intr.cy) / half
    rays, _ = intr.unproject(uv[:, 0], uv[:, 1])
    spread = math.degrees(math.acos(max(-1.0, min(1.0, float(np.min(rays @ rays.mean(0) / np.linalg.norm(
        rays.mean(0)))))))) * 2
    names = ', '.join(m.boards[o.board].t.name for o in mine)
    weak = [a for a, rr in ratio_axes.items() if rr > 1.0]
    placed_limited = [a for a in weak if sig_c[a] < 0.5 * sig[a]]
    if placed_limited:
        words = ', '.join(AXIS_WORDS[a.split('_')[0]] for a in placed_limited)
        alone = ', '.join(_fmt_sig(sig_c[a], 'deg' if a.endswith('deg') else 'mm') for a in placed_limited)
        out.append(f'{words} rest{"s" if len(placed_limited) == 1 else ""} on where the boards lie (placed to '
                   f'±{st.board_sigma_mm:g} mm, ±{st.board_sigma_deg:g}°): the corners alone give {alone}. Lay the '
                   'boards against a jig or measure where they lie, or let this camera share more boards.')
    rest = [a for a in weak if a not in placed_limited]
    if len(mine) == 1:
        out.append(f'One board only ({names}): everything rests on its {mine[0].n} corners.')
    if 'roll_deg' in rest and float(r.max()) < 0.45:
        out.append(f'Roll (about the optical axis) is loose: the corners lie within {100 * float(r.max()):.0f} % '
                   'of the way from the image centre to its corner. Move a board towards the edge of the image.')
    if any(a in rest for a in ('yaw_deg', 'pitch_deg', 'x_mm', 'y_mm', 'z_mm')) and spread < 40.0:
        out.append(f'The boards ({names}) cover only {spread:.0f}° of the view, so turning the camera and shifting it '
                   'look alike in the image. Spread the boards apart or bring one closer.')
    far = float(np.median([o.dist for o in mine]))
    small = min(o.feature_px for o in mine)
    if rest and small < 2 * st.min_square_px:
        out.append(f'The squares are small in the image ({small:.0f} px at {far:.1f} m): bring a board closer or use '
                   'larger squares.')
    if rest and len(out) == (1 if placed_limited else 0) + (1 if len(mine) == 1 else 0):
        words = [AXIS_WORDS[a.split('_')[0]] for a in rest]
        listing = ', '.join(words[:-1]) + (' and ' if len(words) > 1 else '') + words[-1]
        out.append(f'The corners of {names} do not pin {listing} to the requirement. Add a board in another part of '
                   'the image.')
    if (uv[:, 0] > intr.cx).all() or (uv[:, 0] < intr.cx).all():
        side = 'right' if (uv[:, 0] > intr.cx).all() else 'left'
        if rest:
            out.append(f'Every counted corner lies in the {side} half of the image.')
    return out


def _camera_entry(m: _Model, ci, C, C0, det, det0, st) -> dict:
    cm = m.cams[ci]
    c = cm.cam
    mine = [o for o in m.seen if o.cam == ci]
    entry = {'uid': c.uid, 'tag': c.tag, 'slot': c.slot, 'boards': [], 'unseen': [], 'corners': 0,
             'sigma': None, 'sigma_corners': None, 'euler': True, 'ratio': math.inf, 'worst_axis': '',
             'verdict': 'critical', 'weakest': None, 'condition': None, 'reasons': [], 'suggestions': [],
             'distance_m': None}
    for bi, b in enumerate(m.boards):
        if (ci, bi) in m.reasons:
            entry['unseen'].append({'uid': b.t.uid, 'name': b.t.name, 'reason': m.reasons[(ci, bi)]})
    if not det[ci]:
        entry['reasons'] = ['It sees no board whole: the installation check cannot measure it. Move a board into its '
                            'view (see the suggestions).' if m.boards else
                            'There are no calibration targets: plan them in Calibrate ▸ Targets.']
        return entry
    ce, is_euler = _to_euler(C[6 * ci:6 * ci + 6, 6 * ci:6 * ci + 6], cm.R)
    ce0, _ = _to_euler(C0[6 * ci:6 * ci + 6, 6 * ci:6 * ci + 6], cm.R)
    sig, sig0 = _sig(ce), _sig(ce0)
    ratio, axis = _ratio(sig, st.need_deg, st.need_mm)
    axes = {k: v / (st.need_deg if k.endswith('_deg') else st.need_mm) for k, v in sig.items()}
    dist = float(np.median(np.concatenate([np.linalg.norm(m.boards[o.board].P[o.idx] - cm.pos, axis=1)
                                           for o in mine])))
    weak, cond = _weakest(C[6 * ci:6 * ci + 6, 6 * ci:6 * ci + 6], cm.R, dist)
    entry.update(sigma=sig, sigma_corners=sig0, euler=is_euler, ratio=ratio, worst_axis=axis,
                 verdict=_verdict(ratio), weakest=weak, condition=cond, distance_m=dist,
                 corners=int(sum(o.n for o in mine)))
    # each board's share of the camera's information (rotations scaled to the boards' distance) and the worst axis
    # without it
    D = np.diag([dist, dist, dist, 1.0, 1.0, 1.0])
    tr = [float(np.trace(D @ (o.Jc.T @ o.Jc) @ D)) for o in mine]
    total = sum(tr) or 1.0
    for o, t_ in zip(mine, tr):
        Cd, dd = m.cov(drop=(ci, o.board))
        r_without = m.ratios(Cd, dd)[ci]
        b = m.boards[o.board]
        entry['boards'].append({'uid': b.t.uid, 'name': b.t.name, 'corners': o.n, 'distance_m': round(o.dist, 3),
                                'square_px': round(o.feature_px, 1), 'share': t_ / total,
                                'ratio_without': r_without})
    entry['boards'].sort(key=lambda d: -d['share'])
    entry['reasons'] = _explain(m, ci, sig, sig0, axes, st)
    return entry


def _rel_jac(Ra, pa, Rb, pb) -> np.ndarray:
    """d(relative pose of b in a's own axes: rotation vector, translation) / d(a's 6, b's 6), to first order."""
    Rab = Ra.T @ Rb
    t = Ra.T @ (pb - pa)
    J = np.zeros((6, 12))
    J[:3, :3] = -np.eye(3)
    J[:3, 6:9] = Rab
    J[3:, :3] = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    J[3:, 3:6] = -Ra.T
    J[3:, 9:12] = Ra.T
    return J


def relative_pose(Ra, pa, Rb, pb) -> tuple[np.ndarray, np.ndarray]:
    """(R, t) of camera b in camera a's own axes."""
    return Ra.T @ Rb, Ra.T @ (np.asarray(pb, float) - np.asarray(pa, float))


def _pair_cov(C, a, b, Ra, pa, Rb, pb) -> np.ndarray:
    idx = np.r_[6 * a:6 * a + 6, 6 * b:6 * b + 6]
    J = _rel_jac(Ra, pa, Rb, pb)
    cv = J @ C[np.ix_(idx, idx)] @ J.T
    T = np.diag([math.degrees(1)] * 3 + [1000.0] * 3)
    return T @ cv @ T.T


def _pairs(m: _Model, C, C0, det, st) -> list:
    by_cam = {}
    for o in m.seen:
        by_cam.setdefault(o.cam, set()).add(o.board)
    out = []
    n = len(m.cams)
    for a in range(n):
        for b in range(a + 1, n):
            shared = sorted(by_cam.get(a, set()) & by_cam.get(b, set()))
            if not shared or not (det[a] and det[b]):
                continue
            A, B = m.cams[a], m.cams[b]
            sig = _sig(_pair_cov(C, a, b, A.R, A.pos, B.R, B.pos), REL_KEYS)
            sig0 = _sig(_pair_cov(C0, a, b, A.R, A.pos, B.R, B.pos), REL_KEYS)
            ratio, axis = _ratio(sig, st.need_deg, st.need_mm)
            out.append({'a': A.cam.uid, 'b': B.cam.uid, 'tags': [A.cam.tag, B.cam.tag], 'ia': a, 'ib': b,
                        'shared': [m.boards[k].t.name for k in shared], 'sigma': sig, 'sigma_corners': sig0,
                        'ratio': ratio, 'worst_axis': axis, 'verdict': _verdict(ratio),
                        'baseline_m': float(np.linalg.norm(A.pos - B.pos))})
    out.sort(key=lambda p: -p['ratio'])
    return out


# =========================================================================== suggestions
def _with_board(m: _Model, bi, t):
    """The observations with board bi replaced by target t (bi = len(boards): t added), and the board list."""
    boards = list(m.boards)
    nb = _board(t)
    if bi == len(boards):
        boards.append(nb)
    else:
        boards[bi] = nb
    seen = [o for o in m.seen if o.board != bi]
    lost = set()
    for ci, cm in enumerate(m.cams):
        o, _ = _see(cm, ci, nb, bi, m.st)
        if o is not None:
            seen.append(o)
    have = {o.cam for o in seen}
    for ci in range(len(m.cams)):
        if ci not in have and any(o.cam == ci for o in m.seen):
            lost.add(ci)
    return seen, boards, lost


def _floor_free(m: _Model, t, bi) -> bool:
    """Whether target t lies on free floor: not under the body and clear of every other board."""
    b = _board(t)
    for k, o in enumerate(m.boards):
        if k != bi and math.hypot(o.centre[0] - b.centre[0], o.centre[1] - b.centre[1]) < \
                (o.diag + b.diag) / 2 + BOARD_GAP_M:
            return False
    g = m.grid
    if g is not None:
        for x, y, _ in np.vstack([_outline(t), b.centre[None]]):
            cell = g.cell_of(x, y)
            if cell is not None and g.body[cell[1], cell[0]]:
                return False
    return True


def _evaluate(m: _Model, seen, n_boards, before, ci, **kw):
    C, det = _joint(seen, len(m.cams), n_boards, m.st, **kw)
    after = m.ratios(C, det)
    gain = (before[ci] - after[ci]) / before[ci] if math.isfinite(before[ci]) else \
        (1.0 if math.isfinite(after[ci]) else 0.0)
    # how much worse another camera gets: an undetermined one cannot get worse, a determined one lost is infinitely so
    worse = max(((a / b - 1.0) if math.isfinite(a) else math.inf) if math.isfinite(b) and b > 0 else 0.0
                for k, (a, b) in enumerate(zip(after, before)) if k != ci) if len(after) > 1 else 0.0
    return after, gain, worse


def _suggest(m: _Model, ratios: list, max_cams=6) -> dict:
    """Per weak camera index, up to three suggestions (dicts), best first: the best board move, the best added
    board (for the max_cams weakest cameras) and laying the boards more precisely (one evaluation for all)."""
    from .project import Target
    st = m.st
    out = {}
    weak = [int(ci) for ci in np.argsort([-r if math.isfinite(r) else -1e18 for r in ratios]) if ratios[ci] > 1.0]
    tags = [cm.cam.tag for cm in m.cams]
    ts = m.project.target_settings
    placement = None
    if st.boards_placed and weak:
        C, det = _joint(m.seen, len(m.cams), len(m.boards), st, sb_mm=st.board_sigma_mm / 4,
                        sb_deg=st.board_sigma_deg / 4)
        placement = m.ratios(C, det)
    for rank, ci in enumerate(weak):
        cm = m.cams[ci]
        g = cm.pos[:2]
        cands = []
        # ---- move or turn a board (the camera's own and those near its view)
        for bi, b in enumerate(m.boards if rank < max_cams else ()):
            mine = any(o.cam == ci and o.board == bi for o in m.seen)
            d = b.centre[:2] - g
            L = float(np.linalg.norm(d))
            if L < 1e-6 or (not mine and L > 6.0):
                continue
            u = d / L
            left = np.array([-u[1], u[0]])
            for kind, amount in MOVE_STEPS:
                t = copy.copy(b.t)
                if kind == 'closer':
                    if L - amount < 0.6:
                        continue
                    t.x, t.y = (b.centre[:2] - u * amount).tolist()
                elif kind == 'farther':
                    t.x, t.y = (b.centre[:2] + u * amount).tolist()
                elif kind in ('left', 'right'):
                    t.x, t.y = (b.centre[:2] + left * amount * (1 if kind == 'left' else -1)).tolist()
                else:
                    t.yaw = float(wrap180(b.t.yaw + amount))
                # evaluated where it will be laid: the place as the suggestion states it
                t.x, t.y, t.yaw = round(float(t.x), 3), round(float(t.y), 3), round(float(t.yaw), 2)
                if not _floor_free(m, t, bi):
                    continue
                o, _ = _see(cm, ci, _board(t), bi, st)
                if o is None and not mine:
                    continue                    # a board this camera does not see either way
                seen, boards, lost = _with_board(m, bi, t)
                if lost:
                    continue
                after, gain, worse = _evaluate(m, seen, len(boards), ratios, ci)
                if gain >= GAIN_MIN and worse <= COST_MAX:
                    cands.append(('move', gain, after, {'board': b.t.uid, 'name': b.t.name, 'how': kind,
                                                        'amount': amount, 'x': round(t.x, 3), 'y': round(t.y, 3),
                                                        'yaw': round(t.yaw, 2)}))
        # ---- add a board where the camera sees a whole one
        head = math.atan2(cm.R[1, 0], cm.R[0, 0])
        proto = Target(name='new', rows=ts.rows, cols=ts.cols, square_mm=ts.square_mm, pattern=ts.pattern)
        best_add = None
        for dist in (np.arange(0.8, 4.01, 0.4) if rank < max_cams else ()):
            for a in np.radians(np.arange(-75.0, 75.1, 15.0)):
                t = copy.copy(proto)
                t.x = round(float(g[0] + dist * math.cos(head + a)), 3)
                t.y = round(float(g[1] + dist * math.sin(head + a)), 3)
                t.yaw = round(float(wrap180(math.degrees(head + a))), 2)
                if not _floor_free(m, t, -1):
                    continue
                if _see(cm, ci, _board(t), len(m.boards), st)[0] is None:
                    continue
                seen, boards, _ = _with_board(m, len(m.boards), t)
                after, gain, worse = _evaluate(m, seen, len(boards), ratios, ci)
                if gain >= GAIN_MIN and worse <= COST_MAX and (best_add is None or after[ci] < best_add[2][ci]):
                    best_add = ('add', gain, after, {'x': round(t.x, 3), 'y': round(t.y, 3), 'yaw': round(t.yaw, 2),
                                                     'rows': ts.rows, 'cols': ts.cols, 'square_mm': ts.square_mm,
                                                     'pattern': ts.pattern})
        if best_add is not None:
            cands.append(best_add)
        # ---- lay the boards more precisely
        if placement is not None:
            gain = (ratios[ci] - placement[ci]) / ratios[ci] if math.isfinite(ratios[ci]) else 0.0
            if gain >= GAIN_MIN:
                cands.append(('placement', gain, placement, {'board_sigma_mm': st.board_sigma_mm / 4,
                                                             'board_sigma_deg': st.board_sigma_deg / 4}))
        # the best of each kind (a board moved, a board added, the placement), best first
        keep = {}
        for kind, gain, after, info in sorted(cands, key=lambda c: c[2][ci]):
            keep.setdefault(kind, (kind, gain, after, info))
        res = []
        for kind, gain, after, info in sorted(keep.values(), key=lambda c: c[2][ci]):
            others = {tags[k]: (round(ratios[k], 3), round(after[k], 3)) for k in range(len(after))
                      if k != ci and abs(after[k] - ratios[k]) > 0.01 * max(ratios[k], 1e-9)}
            res.append(dict(info, kind=kind, camera=tags[ci], before=ratios[ci], after=after[ci], gain=gain,
                            others=others, text=_suggestion_text(kind, info, tags[ci], ratios[ci], after[ci], others,
                                                                st)))
        out[ci] = res[:3]
    return out


def _need_text(need, unit):
    return f'{need:g}°' if unit == 'deg' else f'{need:g} mm'


def _times(r):
    return 'not determined' if not math.isfinite(r) else f'{r:.1f}×'


def _suggestion_text(kind, info, tag, before, after, others, st) -> str:
    if kind == 'move':
        how = info['how']
        where = {'closer': f'towards {tag}', 'farther': f'away from {tag}', 'left': f'to the left (seen from {tag})',
                 'right': f'to the right (seen from {tag})'}
        what = (f'Turn {info["name"]} by {info["amount"]:+.0f}°' if how == 'turn' else
                f'Move {info["name"]} {info["amount"]:g} m {where[how]}')
    elif kind == 'add':
        what = f'Add a {info["cols"]} × {info["rows"]} board in {tag}\'s view'
    else:
        what = (f'Lay the boards to ±{info["board_sigma_mm"]:g} mm and ±{info["board_sigma_deg"]:g}° (a jig, or '
                'measure where they lie)')
    text = f'{what}: {tag}\'s worst axis {_times(before)} → {_times(after)} the requirement'
    if others and kind == 'placement':
        lo = min(a for _, a in others.values())
        hi = max(a for _, a in others.values())
        text += (f' (the other cameras to {_times(lo)}' + (f'–{_times(hi)}' if hi - lo > 0.05 else '') + ')')
    elif others:
        text += ' (' + ', '.join(f'{k} {_times(b)} → {_times(a)}' for k, (b, a) in others.items()) + ')'
    return text + '.'


def apply_suggestion(project, s: dict) -> bool:
    """Carry out a 'move' or 'add' suggestion on the project's targets; False when it no longer applies."""
    if s.get('kind') == 'move':
        t = next((t for t in project.targets if t.uid == s.get('board')), None)
        if t is None:
            return False
        t.x, t.y, t.yaw = float(s['x']), float(s['y']), float(s['yaw'])
        return True
    if s.get('kind') == 'add':
        from .project import Target
        names = {t.name for t in project.targets}
        k = len(project.targets) + 1
        while f'T{k}' in names:
            k += 1
        project.targets.append(Target(name=f'T{k}', x=float(s['x']), y=float(s['y']), yaw=float(s['yaw']),
                                      pattern=s.get('pattern', 'checkerboard'), rows=int(s.get('rows', 6)),
                                      cols=int(s.get('cols', 6)), square_mm=float(s.get('square_mm', 100.0))))
        return True
    return False


# =========================================================================== lidar–camera pairs
def lidar_model(project, engine, vg, cam, lidar, lst, poses, pixel_sigma):
    """The calibration's least squares at the plan: per pose the corners (model order) and the lidar's points on the
    board, and the information over T_camera_lidar and the boards' poses: (H, M, data).

    vg: the vehicle's geometry (the lidar's occlusion); lst: core.lidar_calib.LidarCalibSettings; poses:
    lidar_calib.BoardPose list (world); pixel_sigma: corner noise (px). H (n×n, n = 6 + 6 per pose: T_camera_lidar's
    rotation vector and translation, then each board's): the information under the calibration's weights; M: the
    middle of the sandwich covariance H⁻¹ M H⁻¹ (the weights where the range noise is not the fit's); data: {'poses'
    (per pose Rb, cb optical, uv corners, beams, s ranges, rms, points, rings), 'T0' (the planned T_camera_lidar),
    'model', 'W', 'H' (board size m), 'Ds' (the lidar's beam directions), 'intr', 'Hcam' (the corners' part of H)}."""
    from . import lidar_calib as LC
    m = project.sensor_model_for(lidar)
    intr = resolve(project.module_for(cam), project.lens_for(cam))
    W2O = LC.world_to_optical(cam)
    C = cam.pos().astype(float)
    T0 = LC.plan_T(cam, lidar)
    R0, t0 = T0[:3, :3], T0[:3, 3]
    model = LC.model_corners(lst)
    Wb, Hb = lst.board_size()
    Ds = S.directions(m.elevations(), S.scan_azimuths(m)).reshape(-1, 3)
    Dopt = Ds @ R0.T
    RL = S.rotation(lidar)
    cube = engine.cube(lidar.pos(), vg) if (engine is not None and vg is not None) else None
    sr = lst.range_sigma_mm / 1000.0
    K = len(poses)
    n = 6 + 6 * K
    H = np.zeros((n, n))
    M = np.zeros((n, n))
    Hcam = np.zeros((n, n))                              # the corners' part (for what limits the pair)
    data = []
    h = 1e-6
    for k, bp in enumerate(poses):
        Rb = W2O @ bp.R
        cb = W2O @ (bp.centre - C)

        def uv_of(Rx, cx):
            u, v, _, _ = intr.project_dirs(model @ Rx.T + cx)
            return np.stack([u, v], -1)
        uv = uv_of(Rb, cb)
        Jy = np.zeros((2 * len(model), 6))
        for j in range(3):
            d = np.zeros(3)
            d[j] = h
            Jy[:, j] = ((uv_of(_exp(d) @ Rb, cb) - uv_of(_exp(-d) @ Rb, cb)) / (2 * h)).ravel()
            Jy[:, 3 + j] = ((uv_of(Rb, cb + d) - uv_of(Rb, cb - d)) / (2 * h)).ravel()
        cols = slice(6 + 6 * k, 12 + 6 * k)
        Hc = Jy.T @ Jy / pixel_sigma ** 2
        H[cols, cols] += Hc
        M[cols, cols] += Hc
        Hcam[cols, cols] += Hc
        # the lidar's beams that meet the board
        nrm = Rb[:, 2]
        den = Dopt @ nrm
        with np.errstate(divide='ignore', invalid='ignore'):
            s = np.where(np.abs(den) > 1e-9, float(nrm @ (cb - t0)) / den, -1.0)
        X = t0 + Dopt * s[:, None]
        a = (X - cb) @ Rb[:, 0]
        b = (X - cb) @ Rb[:, 1]
        on = (s >= m.range_min_m) & (s <= m.range_m) & (np.abs(a) <= Wb / 2) & (np.abs(b) <= Hb / 2)
        if cube is not None and on.any():
            dw = Ds[on] @ RL.T
            dep = cube.lookup(dw)[0]
            keep = (dep > s[on] - 0.03) | (dep <= 0.05)
            on[np.flatnonzero(on)[~keep]] = False
        p = Ds[on] * s[on, None]                        # lidar frame
        cosi = np.abs(Dopt[on] @ nrm)
        N = len(p)
        rms = max(sr * float(np.sqrt(np.mean(cosi ** 2))) if N else sr, 0.002)
        if N:
            q = p @ R0.T
            Jp = np.zeros((N, n))
            Jp[:, 0:3] = np.cross(q, nrm)
            Jp[:, 3:6] = nrm
            Jp[:, 6 + 6 * k:9 + 6 * k] = np.cross(nrm, q + t0 - cb)
            Jp[:, 9 + 6 * k:12 + 6 * k] = -nrm
            f = min(1.0, LC.MAX_POINTS_PER_POSE / N)     # the solve keeps a random 2 500 of a pose's points
            H += f * Jp.T @ Jp / rms ** 2
            M += f * (Jp * (sr * cosi)[:, None] ** 2).T @ Jp / rms ** 4
        data.append({'Rb': Rb, 'cb': cb, 'uv': uv, 'beams': np.flatnonzero(on), 's': s[on], 'rms': rms,
                     'points': N, 'rings': int(len(np.unique(np.flatnonzero(on) // max(len(S.scan_azimuths(m)), 1))))})
    return H, M, {'poses': data, 'T0': T0, 'model': model, 'W': Wb, 'H': Hb, 'Ds': Ds, 'intr': intr, 'Hcam': Hcam}


def _lidar_pair(project, engine, vg, cam, lidar, share, st: ObservabilitySettings):
    from . import lidar_calib as LC
    lst = LC.LidarCalibSettings.from_dict(getattr(project.analysis, 'lidar_calib', None))
    lst.camera, lst.lidar = cam.tag, lidar.tag
    entry = {'camera': cam.tag, 'camera_uid': cam.uid, 'lidar': lidar.tag, 'lidar_uid': lidar.uid,
             'share': float(share), 'poses': [], 'notes': [], 'sigma': None, 'sigma_camera': None,
             'ratio': math.inf, 'worst_axis': '', 'verdict': 'critical', 'points': [], 'limited_by': '',
             'if_firmer': {},
             'tolerance': {'mm': lst.pos_tol_mm, 'deg': lst.ang_tol_deg},
             'board': {'cols': lst.cols, 'rows': lst.rows, 'square_mm': lst.square_mm}}
    try:
        poses, notes = LC.plan_poses(project, cam, lidar, lst, engine)
    except ValueError as e:
        entry['notes'] = [str(e)]
        return entry, None
    entry['notes'] = list(notes)
    entry['poses'] = [b.to_dict() for b in poses]
    H, M, data = lidar_model(project, engine, vg, cam, lidar, lst, poses, st.pixel_sigma)
    Hi, ok = _inv(H)
    Cv = Hi @ M @ Hi
    cov = (Cv[:6, :6] + Cv[:6, :6].T) / 2
    sg = LC.pose_sigmas(cam, data['T0'], cov)
    veh = sg['vehicle']
    sig = {'yaw_deg': veh['yaw_deg'], 'pitch_deg': veh['pitch_deg'], 'roll_deg': veh['roll_deg'],
           'x_mm': veh['x_mm'], 'y_mm': veh['y_mm'], 'z_mm': veh['z_mm']}
    sig = {k: (float(v) if ok and math.isfinite(v) else math.inf) for k, v in sig.items()}
    ratio, axis = _ratio(sig, lst.ang_tol_deg, lst.pos_tol_mm)
    entry.update(sigma=sig, sigma_camera={k: float(v) for k, v in sg['camera'].items()}, ratio=ratio,
                 worst_axis=axis, verdict=_verdict(ratio), points=[d['points'] for d in data['poses']])
    # what limits it: the worst axis with the camera's corners, then the lidar's points, made 100 times firmer
    Hc = data['Hcam']
    lim = {}
    for who, f_c, f_l in (('camera', 100.0, 1.0), ('lidar', 1.0, 100.0)):
        Hx = Hc * f_c + (H - Hc) * f_l
        Mx = Hc * f_c + (M - Hc) * f_l                  # noise / 10: weights and variances scale alike
        Hxi, _ = _inv(Hx)
        cx = Hxi @ Mx @ Hxi
        sx = LC.pose_sigmas(cam, data['T0'], (cx[:6, :6] + cx[:6, :6].T) / 2)['vehicle']
        lim[who] = _ratio({'yaw_deg': sx['yaw_deg'], 'pitch_deg': sx['pitch_deg'], 'roll_deg': sx['roll_deg'],
                           'x_mm': sx['x_mm'], 'y_mm': sx['y_mm'], 'z_mm': sx['z_mm']},
                          lst.ang_tol_deg, lst.pos_tol_mm)[0]
    entry['limited_by'] = 'camera' if lim['camera'] < lim['lidar'] else 'lidar'
    entry['if_firmer'] = {k: float(v) for k, v in lim.items()}
    if ratio > 1.0:
        if entry['limited_by'] == 'camera':
            entry['notes'].append(f'The camera\'s corners limit it: where {cam.tag} puts each board (its distance '
                                  'above all) is the loose part. Bigger squares, nearer poses or a narrower lens '
                                  f'would firm it up ({_times(ratio)} → {_times(lim["camera"])} with corners 10× '
                                  'sharper).')
        else:
            entry['notes'].append(f'The lidar\'s points limit it: more poses, boards turned further or nearer '
                                  f'would firm it up ({_times(ratio)} → {_times(lim["lidar"])} with points 10× '
                                  'sharper).')
    data.update(cam=cam, lidar=lidar, lst=lst, cov=cov, plan=poses)
    return entry, data


# =========================================================================== radars
def _radar_meas(s, Q, pose=None):
    """(range, azimuth rad) of points Q (n, 3) from radar s at pose (x, y, yaw deg) (its plan when None)."""
    x, y, yaw = (s.x, s.y, s.yaw) if pose is None else pose
    R = rot_zyx(yaw, s.pitch, s.roll)
    q = (np.asarray(Q, float) - np.array([x, y, s.z])) @ R
    return np.linalg.norm(q, axis=1), np.arctan2(q[:, 1], q[:, 0])


def _radar_jac(s, Q):
    """d(range, azimuth) / d(radar x m, y m, yaw rad) and d / d(reflector x, y, z m), stacked per reflector."""
    h = 1e-6
    n = len(Q)
    Jr = np.zeros((2 * n, 3))
    for k in range(3):
        d = np.zeros(3)
        d[k] = h if k < 2 else math.degrees(h)
        a = np.c_[_radar_meas(s, Q, (s.x + d[0], s.y + d[1], s.yaw + d[2]))]
        b = np.c_[_radar_meas(s, Q, (s.x - d[0], s.y - d[1], s.yaw - d[2]))]
        df = a - b
        df[:, 1] = (df[:, 1] + math.pi) % (2 * math.pi) - math.pi
        Jr[:, k] = (df / (2 * h)).ravel()
    Jq = np.zeros((2 * n, 3 * n))
    for k in range(3):
        d = np.zeros(3)
        d[k] = h
        a = np.c_[_radar_meas(s, Q + d)]
        b = np.c_[_radar_meas(s, Q - d)]
        df = a - b
        df[:, 1] = (df[:, 1] + math.pi) % (2 * math.pi) - math.pi
        dd = df / (2 * h)                                # (n, 2): each reflector moves only its own row pair
        for i in range(n):
            Jq[2 * i:2 * i + 2, 3 * i + k] = dd[i]
    return Jr, Jq


def _radar_cov(s, Q, st: ObservabilitySettings):
    """Covariance of the radar's (x m, y m, yaw rad) from reflectors Q (n, 3): range and azimuth noise, the
    reflectors' survey σ marginalised."""
    n = len(Q)
    Jr, Jq = _radar_jac(s, Q)
    w = np.tile([1 / st.radar_range_sigma_m ** 2, 1 / math.radians(st.radar_az_sigma_deg) ** 2], n)
    sq = st.reflector_sigma_mm / 1000.0
    if sq > 0:
        J = np.c_[Jr, Jq]
        H = J.T @ (J * w[:, None])
        H[3:, 3:] += np.eye(3 * n) / sq ** 2
    else:
        H = Jr.T @ (Jr * w[:, None])
    Hi, ok = _inv(H)
    C = Hi[:3, :3]
    if not ok:
        lam, V = np.linalg.eigh((H + H.T) / 2)
        null = V[:3, lam < lam.max() * 1e-13]
        C = C + null @ null.T * 1e12
    return C


def _radars(project, engine, vg, v, st: ObservabilitySettings) -> list:
    out = []
    refl = st.reflectors
    for s in v.sensors:
        if not s.visible or s.kind != 'radar' or not finite_pose(s):
            continue
        m = project.sensor_model_for(s)
        entry = {'uid': s.uid, 'tag': s.tag, 'model': m.name, 'reflectors': [], 'sigma': None, 'ratio': math.inf,
                 'worst_axis': '', 'verdict': 'critical', 'note': ''}
        if refl:
            Q = np.array([[r['x'], r['y'], r['z']] for r in refl], float)
            ok = np.zeros(len(Q), bool)
            for i, r in enumerate(refl):
                ok[i] = bool(S.in_field(s, m, Q[i:i + 1], reflector_rcs(r['edge_mm']))[0])
            if vg is not None and ok.any():
                cube = engine.cube(s.pos(), vg)
                d = Q[ok] - s.pos()
                dist = np.linalg.norm(d, axis=1)
                dep = cube.lookup(d)[0]
                clear = (dep > dist - (0.02 + 0.01 * dist)) | (dep <= FASCIA_M)
                ok[np.flatnonzero(ok)[~clear]] = False
            entry['reflectors'] = [refl[i]['name'] for i in np.flatnonzero(ok)]
            if ok.sum() >= 2:
                C = _radar_cov(s, Q[ok], st)
                sd = np.sqrt(np.maximum(np.diag(C), 0))
                sig = {'yaw_deg': math.degrees(sd[2]), 'x_mm': sd[0] * 1000, 'y_mm': sd[1] * 1000}
                sig = {k: (float(x) if math.isfinite(x) and x < 1e7 else math.inf) for k, x in sig.items()}
                ratio, axis = _ratio(sig, st.radar_need_deg, st.radar_need_mm)
                entry.update(sigma=sig, ratio=ratio, worst_axis=axis, verdict=_verdict(ratio))
            else:
                entry['note'] = (f'{int(ok.sum())} reflector in its field: two or more pin its x, y and yaw.'
                                 if ok.sum() == 1 else 'No reflector in its field (its zones for the reflector\'s '
                                                       'cross-section, nothing of the vehicle in between).')
        else:
            entry['note'] = 'No corner reflectors placed.'
        out.append(entry)
    return out


# =========================================================================== run
def _findings(res: ObservabilityResult) -> list:
    st = res.settings
    out = []

    def add(sev, kind, msg, tags=(), uids=()):
        out.append({'severity': sev, 'kind': kind, 'message': msg, 'tags': list(tags), 'uids': list(uids)})
    if not res._model.boards:
        add('warning', 'targets', 'No calibration targets are planned: the cameras\' poses cannot be measured. Plan '
                                  'them in Calibrate ▸ Targets.')
    for c in res.cameras:
        if res._model.boards and c['sigma'] is None:
            add('serious', 'camera', f'{c["tag"]} sees no board whole: the installation check cannot measure it.',
                [c['tag']], [c['uid']])
        elif c['sigma'] is not None and c['verdict'] != 'good':
            s = c['sigma'][f'{c["worst_axis"]}_{"deg" if c["worst_axis"] in ("yaw", "pitch", "roll") else "mm"}']
            unit = 'deg' if c['worst_axis'] in ('yaw', 'pitch', 'roll') else 'mm'
            need = st.need_deg if unit == 'deg' else st.need_mm
            add(c['verdict'] if c['verdict'] != 'critical' else 'serious', 'camera',
                f'{c["tag"]}: the boards pin its {AXIS_WORDS[c["worst_axis"]]} to ±{_fmt_sig(s, unit)} '
                f'({_times(c["ratio"])} the {_need_text(need, unit)} requirement).'
                + (f' {c["suggestions"][0]["text"]}' if c['suggestions'] else ''), [c['tag']], [c['uid']])
    good = [c for c in res.cameras if c['verdict'] == 'good']
    if good and len(good) == len(res.cameras):
        w = max(good, key=lambda c: c['ratio'])
        add('good', 'camera', f'Every camera\'s pose is pinned within the requirement: the loosest, {w["tag"]}, at '
                              f'{w["ratio"]:.2f}× on its {AXIS_WORDS[w["worst_axis"]]}.')
    for p in res.pairs:
        if p['verdict'] in ('serious', 'critical'):
            add('warning', 'pair', f'{p["tags"][0]}–{p["tags"][1]}: their relative pose (through '
                                   f'{", ".join(p["shared"])}) is pinned only to {_times(p["ratio"])} the requirement '
                                   f'on {p["worst_axis"]}.', p['tags'], [p['a'], p['b']])
    for lp in res.lidar_pairs:
        if lp['sigma'] is None:
            add('warning', 'lidar', f'{lp["camera"]}–{lp["lidar"]}: no lidar–camera board pose fits. '
                                    + (lp['notes'][0] if lp['notes'] else ''), [lp['camera'], lp['lidar']],
                [lp['camera_uid'], lp['lidar_uid']])
        elif lp['verdict'] != 'good':
            add('warning', 'lidar', f'{lp["camera"]}–{lp["lidar"]}: the planned lidar–camera boards pin the lidar\'s '
                                    f'{lp["worst_axis"]} to {_times(lp["ratio"])} the tolerance. '
                                    + (lp['notes'][-1] if lp['notes'] else ''), [lp['camera'], lp['lidar']],
                [lp['camera_uid'], lp['lidar_uid']])
    for r in res.radars:
        if st.reflectors and r['sigma'] is None:
            add('warning', 'radar', f'{r["tag"]}: {r["note"]}', [r['tag']], [r['uid']])
        elif r['sigma'] is not None and r['verdict'] != 'good':
            add('warning', 'radar', f'{r["tag"]}: the reflectors pin its {r["worst_axis"]} to {_times(r["ratio"])} '
                                    'the requirement. Add a reflector farther out or at another bearing.',
                [r['tag']], [r['uid']])
    for f in (res.fusion or []):
        for msg in f.get('findings', []):
            add(msg['severity'], 'fusion', msg['message'], [f['tag']], [f['uid']])
    order = {s: i for i, s in enumerate(SEVERITIES)}
    out.sort(key=lambda f: order.get(f['severity'], 9))
    return out


def run(project, engine=None, settings: ObservabilitySettings | None = None, variant=None, progress=None,
        cancel=None) -> ObservabilityResult | None:
    """The calibration observability of a variant (the active one by default): every visible camera from the planned
    boards, the camera pairs, the camera–lidar pairs, the radars from the reflectors, and the camera–lidar fusion
    density. Returns None when cancelled."""
    from .analysis import Engine
    t0 = time.perf_counter()
    st = settings or settings_of(project)
    p = project_for(project, variant)
    v = p.variant
    eng = engine or Engine()

    def prog(f, text=''):
        if progress is not None:
            progress(min(max(float(f), 0.0), 1.0), text)

    def stop():
        return cancel is not None and cancel()
    prog(0.02, 'Boards in every camera')
    m = _Model(p, eng, v, st)
    if stop():
        return None
    C, det = m.cov()
    C0, det0 = m.cov(placed=False)
    ratios = m.ratios(C, det)
    prog(0.15, 'Cameras')
    cams = [_camera_entry(m, ci, C, C0, det, det0, st) for ci in range(len(m.cams))]
    if stop():
        return None
    pairs = _pairs(m, C, C0, det, st)
    if st.suggest and m.cams:
        prog(0.2, 'Trying board moves')
        sugg = _suggest(m, ratios)
        for ci, lst in sugg.items():
            cams[ci]['suggestions'] = lst
    if stop():
        return None
    # ---- boards
    boards = []
    for bi, b in enumerate(m.boards):
        seen_by = [o for o in m.seen if o.board == bi]
        only = [m.cams[o.cam].cam.tag for o in seen_by if sum(1 for q in m.seen if q.cam == o.cam) == 1]
        ties = [f'{a["tags"][0]}–{a["tags"][1]}' for a in pairs if b.t.name in a['shared']]
        per = []
        for o in seen_by:
            ce = next(c for c in cams if c['uid'] == m.cams[o.cam].cam.uid)
            share = next((x['share'] for x in ce['boards'] if x['uid'] == b.t.uid), 0.0)
            per.append({'camera': m.cams[o.cam].cam.tag, 'corners': o.n, 'share': share,
                        'square_px': round(o.feature_px, 1), 'distance_m': round(o.dist, 3)})
        boards.append({'uid': b.t.uid, 'name': b.t.name, 'x': b.t.x, 'y': b.t.y, 'yaw': b.t.yaw,
                       'pattern': b.pattern, 'rows': b.t.rows, 'cols': b.t.cols, 'square_mm': b.t.square_mm,
                       'seen_by': per, 'only_board_of': only, 'ties': ties,
                       'unseen': {m.cams[ci].cam.tag: why for (ci, bi_), why in m.reasons.items() if bi_ == bi}})
    prog(0.35, 'Lidar–camera pairs')
    vg = m.vg
    lidar_pairs, lidar_models = [], []
    if st.lidar_pairs and any(s.visible and s.kind == 'lidar' for s in v.sensors) and m.cams:
        from . import lidar_calib as LC
        ranked = [r for r in LC.pairs(p, v, eng) if r['share'] >= st.min_share]
        for i, r in enumerate(ranked):
            if stop():
                return None
            cam, lid = v.cam(r['camera_uid']), v.sensor(r['lidar_uid'])
            if cam is None or lid is None or not finite_pose(cam) or not finite_pose(lid):
                continue
            entry, data = _lidar_pair(p, eng, vg, cam, lid, r['share'], st)
            lidar_pairs.append(entry)
            lidar_models.append(data)
            prog(0.35 + 0.3 * (i + 1) / max(len(ranked), 1), f'{cam.tag}–{lid.tag}')
    prog(0.65, 'Radars')
    radars = _radars(p, eng, vg, v, st)
    fusion = None
    if st.fusion and any(s.visible and s.kind == 'lidar' for s in v.sensors) and m.cams:
        from . import sensor_coverage as SC
        prog(0.7, 'Camera–lidar fusion')
        fusion = SC.camera_fusion(p, eng, None, at_m=st.fusion_at_m,
                                  progress=lambda f: prog(0.7 + 0.29 * f, 'Camera–lidar fusion'), cancel=cancel)
        if fusion is None and stop():
            return None
    outline = []
    if m.grid is not None:
        outline = [np.asarray(o, float)[:, :2].copy() for o in m.grid.outline]
    metrics = {'seconds': time.perf_counter() - t0, 'cameras': len(cams), 'boards': len(m.boards),
               'observations': len(m.seen), 'pairs': len(pairs), 'lidar_pairs': len(lidar_pairs),
               'radars': len(radars), 'reflectors': len(st.reflectors),
               'measured': sum(1 for c in cams if c['sigma'] is not None),
               'good': sum(1 for c in cams if c['verdict'] == 'good')}
    res = ObservabilityResult(st, v.name, v.uid, cams, boards, pairs, lidar_pairs, radars, fusion, [], metrics,
                              outline)
    m.lidar_models = lidar_models
    res._model = m
    res.findings = _findings(res)
    prog(1.0, 'Done')
    return res


# =========================================================================== Monte Carlo
def _mc_stats(errs: np.ndarray, cov_pred: np.ndarray, keys) -> dict:
    """Spread of simulated errors (trials, k) against the predicted covariance (k, k): per axis the empirical and the
    predicted σ and their ratio; the mean squared Mahalanobis distance (k for a right prediction)."""
    e = np.asarray(errs, float)
    sd = e.std(0, ddof=1) if len(e) > 1 else np.full(e.shape[1], np.nan)
    ps = np.sqrt(np.maximum(np.diag(cov_pred), 0))
    try:
        Ci = np.linalg.inv(cov_pred)
        d2 = float(np.mean(np.einsum('ni,ij,nj->n', e, Ci, e)))
    except np.linalg.LinAlgError:
        d2 = float('nan')
    out = {'trials': int(len(e)), 'mahalanobis2': d2, 'dof': int(e.shape[1]), 'axes': {}}
    for k, key in enumerate(keys):
        out['axes'][key] = {'predicted': float(ps[k]), 'empirical': float(sd[k]),
                            'ratio': float(sd[k] / ps[k]) if ps[k] > 0 else float('nan'),
                            'bias': float(e[:, k].mean())}
    return out


def _cam_errors(cm: _Cam, R, pos) -> np.ndarray:
    """(yaw, pitch, roll deg, x, y, z mm) of a solved pose against the plan (body axes near straight down)."""
    if _euler_jac(cm.R) is None:
        w = np.degrees(_log(cm.R.T @ np.asarray(R, float)))
        ang = [w[2], w[1], w[0]]
    else:
        y, p_, r = euler_zyx(np.asarray(R, float))
        c = cm.cam
        ang = [wrap180(y - c.yaw), p_ - c.pitch, wrap180(r - c.roll)]
    return np.r_[ang, (np.asarray(pos, float) - cm.pos) * 1000.0]


def _polish(m: _Model, cams, obs_data, poses, boards, iters=12):
    """Gauss–Newton from a solve's answer to the least-squares optimum of the same problem (the joint solve's model:
    corners weighted at 0.5 px, the boards' placement priors), with the plan's Jacobians: tells a solve that stopped
    short of its optimum. obs_data: [(camera index, board index, observed uv (n, 2), _Seen)]; poses: camera index ->
    (R, pos); boards: board index -> (dx, dy, dyaw rad). Returns (poses, cost drop)."""
    st = m.st
    placed = st.boards_placed
    nc, nb = len(cams), (len(m.boards) if placed else 0)
    ci_of = {k: i for i, k in enumerate(cams)}
    sw = JOINT_PIXEL_SIGMA if placed else 1.0
    pri = np.tile([1 / (st.board_sigma_mm / 1000.0), 1 / (st.board_sigma_mm / 1000.0),
                   1 / math.radians(st.board_sigma_deg)], nb) if placed else np.zeros(0)
    x = np.zeros(6 * nc + 3 * nb)
    for k in cams:
        R, pos = poses[k]
        cm = m.cams[k]
        i = ci_of[k]
        x[6 * i:6 * i + 3] = _log(cm.R.T @ np.asarray(R, float))
        x[6 * i + 3:6 * i + 6] = np.asarray(pos, float) - cm.pos
    if placed:
        for bi, val in boards.items():
            x[6 * nc + 3 * bi:6 * nc + 3 * bi + 3] = val
    n = len(x)
    JTJ = np.zeros((n, n))
    rows = []
    for k, bi, uv, o in obs_data:
        i = ci_of[k]
        cs = slice(6 * i, 6 * i + 6)
        bs = slice(6 * nc + 3 * bi, 6 * nc + 3 * bi + 3) if placed else None
        rows.append((k, i, bi, uv, o, cs, bs))
        JTJ[cs, cs] += o.Jc.T @ o.Jc / sw ** 2
        if placed:
            B = o.Jc.T @ o.Jb / sw ** 2
            JTJ[cs, bs] += B
            JTJ[bs, cs] += B.T
            JTJ[bs, bs] += o.Jb.T @ o.Jb / sw ** 2
    if placed:
        JTJ[6 * nc:, 6 * nc:] += np.diag(pri ** 2)
    Hi, _ = _inv(JTJ)

    def grad_cost(x):
        g = np.zeros(n)
        cost = 0.0
        for k, i, bi, uv, o, cs, bs in rows:
            cm, b = m.cams[k], m.boards[bi]
            R = cm.R @ _exp(x[6 * i:6 * i + 3])
            pos = cm.pos + x[6 * i + 3:6 * i + 6]
            W = b.P[o.idx]
            if placed:
                dx, dy, dyaw = x[6 * nc + 3 * bi:6 * nc + 3 * bi + 3]
                W = (W - b.centre) @ rot_z(dyaw).T + b.centre + np.array([dx, dy, 0.0])
            pr, _, _ = cm.project(W, BODY_TO_OPTICAL @ R.T, pos)
            r = ((pr - uv) / sw).ravel()
            cost += float(r @ r)
            g[cs] += o.Jc.T @ r / sw
            if placed:
                g[bs] += o.Jb.T @ r / sw
        if placed:
            rb = x[6 * nc:] * pri
            cost += float(rb @ rb)
            g[6 * nc:] += rb * pri
        return g, cost
    g, c0 = grad_cost(x)
    cost = c0
    for _ in range(iters):
        step = Hi @ g
        x_new = x - step
        g_new, c_new = grad_cost(x_new)
        if c_new > cost:
            break
        x, g, cost = x_new, g_new, c_new
        if np.max(np.abs(step)) < 1e-10:
            break
    out = {}
    for k in cams:
        i = ci_of[k]
        out[k] = (m.cams[k].R @ _exp(x[6 * i:6 * i + 3]), m.cams[k].pos + x[6 * i + 3:6 * i + 6])
    return out, 0.5 * (c0 - cost)


def _mc_cameras(res: ObservabilityResult, trials, rng, progress, cancel) -> dict | None:
    """Simulate the installation check's capture (noisy corners, boards laid off their plan) and solve it with
    core.joint.joint_solve, its default weights; compare the spread with the prediction. Each answer is also carried
    to the optimum of the same least squares (_polish): where the joint solve stops short of it, its own spread is
    wider than the prediction while the optimum's is not."""
    from .install import BoardObs, solve_pose
    from .joint import joint_solve
    m: _Model = res._model
    st = m.st
    cams = [k for k in range(len(m.cams)) if any(o.cam == k for o in m.seen)]
    if not cams:
        return None
    C, det = m.cov()
    idx = {k: i for i, k in enumerate(cams)}
    sb, sy = st.board_sigma_mm / 1000.0, math.radians(st.board_sigma_deg)
    errs = {w: {k: [] for k in cams} for w in ('joint', 'optimum')}
    pair_list = [p for p in res.pairs if p['ia'] in idx and p['ib'] in idx]
    perr = {w: {(p['ia'], p['ib']): [] for p in pair_list} for w in ('joint', 'optimum')}
    proj = m.project
    camobj = [m.cams[k].cam for k in cams]
    shapes = {m.cams[k].cam.uid: (m.cams[k].intr.height, m.cams[k].intr.width) for k in cams}
    bindex = {b.t.uid: bi for bi, b in enumerate(m.boards)}
    drops, stalled = [], 0
    t0 = time.perf_counter()
    for trial in range(trials):
        if cancel is not None and cancel():
            return None
        off = {bi: ((rng.normal(0, sb), rng.normal(0, sb), rng.normal(0, sy)) if st.boards_placed else (0, 0, 0))
               for bi in range(len(m.boards))}
        obs, data = {}, []
        for o in m.seen:
            if o.cam not in idx:
                continue
            cm, b = m.cams[o.cam], m.boards[o.board]
            W = b.P[o.idx]
            dx, dy, dyaw = off[o.board]
            Wt = (W - b.centre) @ rot_z(dyaw).T + b.centre + np.array([dx, dy, 0.0])
            uv, _, _ = cm.project(Wt)
            uv = uv + rng.normal(0, st.pixel_sigma, uv.shape)
            rays, _ = cm.intr.unproject(uv[:, 0], uv[:, 1])
            obs.setdefault(cm.cam.uid, []).append(BoardObs(b.t.uid, b.t.name, W, rays, uv, uv.copy(),
                                                           np.zeros((4, 2))))
            data.append((o.cam, o.board, uv, o))
        # the joint solve starts from each camera's own solve (core.install.solve_pose, the boards where the plan
        # puts them), as the installation check does
        init = {}
        for k in cams:
            cm = m.cams[k]
            try:
                R0, p0, _ = solve_pose(proj, cm.cam, obs[cm.cam.uid], shapes[cm.cam.uid])
            except Exception:                    # noqa: BLE001 - PnP did not converge: start from the plan
                R0, p0 = cm.R, cm.pos
            init[cm.cam.uid] = (R0, p0)
        jr = joint_solve(proj, camobj, obs, init, shapes, st.board_sigma_mm, st.board_sigma_deg,
                         solve_boards=st.boards_placed)
        if not jr.poses or jr.rejected:
            continue
        poses = {k: jr.poses[m.cams[k].cam.uid] for k in cams}
        bvals = {bindex[b.uid]: (b.dx_mm / 1000.0, b.dy_mm / 1000.0, math.radians(b.dyaw_deg))
                 for b in jr.boards if b.uid in bindex}
        best, drop = _polish(m, cams, data, poses, bvals)
        drops.append(drop)
        stalled += drop > 0.5                     # half a unit of χ² left on the table
        for w, P in (('joint', poses), ('optimum', best)):
            for k in cams:
                errs[w][k].append(_cam_errors(m.cams[k], *P[k]))
            for p in pair_list:
                a, b = p['ia'], p['ib']
                A, B = m.cams[a], m.cams[b]
                Rr, tr = relative_pose(np.asarray(P[a][0]), P[a][1], np.asarray(P[b][0]), P[b][1])
                R0, t0_ = relative_pose(A.R, A.pos, B.R, B.pos)
                perr[w][(a, b)].append(np.r_[np.degrees(_log(Rr @ R0.T)), (tr - t0_) * 1000.0])
        if progress is not None:
            progress((trial + 1) / trials)
    out = {'solver': 'core.joint.joint_solve (the installation check\'s joint solve, default weights)',
           'seconds': time.perf_counter() - t0, 'cameras': {}, 'pairs': {}, 'optimum': {'cameras': {}, 'pairs': {}},
           'solved': len(drops), 'stalled': int(stalled),
           'median_cost_left': float(np.median(drops)) if drops else 0.0}
    for k in cams:
        if len(errs['joint'][k]) < 3:
            continue
        ce, _ = _to_euler(C[6 * k:6 * k + 6, 6 * k:6 * k + 6], m.cams[k].R)
        out['cameras'][m.cams[k].cam.tag] = _mc_stats(np.array(errs['joint'][k]), ce, SIGMA_KEYS)
        out['optimum']['cameras'][m.cams[k].cam.tag] = _mc_stats(np.array(errs['optimum'][k]), ce, SIGMA_KEYS)
    for p in pair_list:
        key = (p['ia'], p['ib'])
        if len(perr['joint'][key]) < 3:
            continue
        A, B = m.cams[p['ia']], m.cams[p['ib']]
        cv = _pair_cov(C, p['ia'], p['ib'], A.R, A.pos, B.R, B.pos)
        name = f'{p["tags"][0]}–{p["tags"][1]}'
        out['pairs'][name] = _mc_stats(np.array(perr['joint'][key]), cv, REL_KEYS)
        out['optimum']['pairs'][name] = _mc_stats(np.array(perr['optimum'][key]), cv, REL_KEYS)
    return out


def _mc_lidar(res: ObservabilityResult, trials, rng, progress, cancel) -> dict | None:
    """Noisy corners and lidar points of the planned lidar–camera poses, solved with the calibration's own solve
    (core.lidar_calib._solve, edges and checker off)."""
    from . import lidar_calib as LC
    m: _Model = res._model
    st = m.st
    models = [d for d in getattr(m, 'lidar_models', []) if d is not None]
    if len(models) > MC_LIDAR_PAIRS:            # the best-overlapping pair (the one calibrated first) and the weakest
        ratio = {id(d): next((lp['ratio'] for lp in res.lidar_pairs if lp['camera_uid'] == d['cam'].uid and
                              lp['lidar_uid'] == d['lidar'].uid), 0.0) for d in models}
        keep = [models[0]] + sorted(models[1:], key=lambda d: -ratio[id(d)])[:MC_LIDAR_PAIRS - 1]
        models = keep
    out = {'solver': 'core.lidar_calib._solve (the lidar–camera calibration\'s least squares, edges and checker '
                     'off)', 'pairs': {}}
    t0 = time.perf_counter()
    for pi, d in enumerate(models):
        cam, lidar, lst = d['cam'], d['lidar'], d['lst']
        intr, model, T0 = d['intr'], d['model'], d['T0']
        sr = lst.range_sigma_mm / 1000.0
        errs = []
        Rl0, pl0 = LC.lidar_pose(cam, T0)
        e0 = euler_zyx(Rl0)
        for trial in range(trials):
            if cancel is not None and cancel():
                return None
            obs = []
            for k, pd in enumerate(d['poses']):
                Rb, cb = pd['Rb'], pd['cb']
                u, v, _, _ = intr.project_dirs(model @ Rb.T + cb)
                uv = np.stack([u, v], -1) + rng.normal(0, st.pixel_sigma, (len(model), 2))
                rays, _ = intr.unproject(uv[:, 0], uv[:, 1])
                # the camera's own estimate of the board starts the solve
                Rs = _exp(rng.normal(0, math.radians(0.3), 3)) @ Rb
                cs = cb + rng.normal(0, 0.005, 3)
                cbd = LC.CamBoard(uv, rays, Rs, cs, st.pixel_sigma, np.eye(6))
                beams = pd['beams']
                rr = pd['s'] + rng.normal(0, sr, len(beams))
                pts = d['Ds'][beams] * rr[:, None]
                nl = T0[:3, :3].T @ Rb[:, 2]
                lb = LC.LidarBoard(pts, nl, float(nl @ pts.mean(0)) if len(pts) else 0.0, pd['rms'],
                                   np.zeros((0, 3)), np.zeros((0, 3)), pd['rings'])
                obs.append(LC.PoseObs(f'pose{k + 1:02d}', cam=cbd, lid=lb))
            Ts = LC._T(_exp(rng.normal(0, math.radians(0.3), 3)) @ T0[:3, :3], T0[:3, 3] + rng.normal(0, 0.01, 3))
            sol = LC._solve(intr, model, d['W'], d['H'], obs, Ts, use_edges=False, seed=trial, st=lst,
                            use_checker=False)
            Rl, pl = LC.lidar_pose(cam, sol['T'])
            e = euler_zyx(Rl)
            errs.append([wrap180(e[0] - e0[0]), e[1] - e0[1], wrap180(e[2] - e0[2]), *((pl - pl0) * 1000.0)])
            if progress is not None:
                progress((pi * trials + trial + 1) / (trials * len(models)))
        # the predicted covariance of the lidar's vehicle pose (as pose_sigmas propagates it)
        Jf = np.zeros((6, 6))
        for i in range(6):
            dd = np.zeros(6)
            dd[i] = 1e-6

            def f(x):
                R, p_ = LC.lidar_pose(cam, LC._T(_exp(x[:3]) @ T0[:3, :3], T0[:3, 3] + x[3:]))
                y, pp, r = euler_zyx(R)
                return np.array([y, pp, r, p_[0] * 1000, p_[1] * 1000, p_[2] * 1000])
            df = f(dd) - f(-dd)
            df[:3] = (df[:3] + 180) % 360 - 180
            Jf[:, i] = df / 2e-6
        cv = Jf @ d['cov'] @ Jf.T
        out['pairs'][f'{cam.tag}–{lidar.tag}'] = _mc_stats(np.array(errs), cv, SIGMA_KEYS)
    out['seconds'] = time.perf_counter() - t0
    return out


def _mc_radars(res: ObservabilityResult, trials, rng, progress, cancel) -> dict | None:
    """Noisy ranges and azimuths of the reflectors, the reflectors off their surveyed place, solved by least squares
    for the radar's x, y and yaw."""
    from scipy.optimize import least_squares
    m: _Model = res._model
    st = m.st
    v = m.v
    out = {'solver': 'scipy least squares on range and azimuth (x, y, yaw)', 'radars': {}}
    entries = [r for r in res.radars if r['sigma'] is not None]
    refl = {r['name']: r for r in st.reflectors}
    for ri, r in enumerate(entries):
        s = v.sensor(r['uid'])
        Q = np.array([[refl[n]['x'], refl[n]['y'], refl[n]['z']] for n in r['reflectors']], float)
        C = _radar_cov(s, Q, st)
        errs = []
        sq = st.reflector_sigma_mm / 1000.0
        for trial in range(trials):
            if cancel is not None and cancel():
                return None
            Qt = Q + rng.normal(0, sq, Q.shape) if sq > 0 else Q
            rr, az = _radar_meas(s, Qt)
            rr = rr + rng.normal(0, st.radar_range_sigma_m, len(rr))
            az = az + rng.normal(0, math.radians(st.radar_az_sigma_deg), len(az))
            wr, wa = 1 / st.radar_range_sigma_m, 1 / math.radians(st.radar_az_sigma_deg)

            def resid(x):
                pr, pa = _radar_meas(s, Q, (s.x + x[0], s.y + x[1], s.yaw + math.degrees(x[2])))
                da = (pa - az + math.pi) % (2 * math.pi) - math.pi
                out_ = np.r_[(pr - rr) * wr, da * wa]
                return out_
            x0 = np.array([rng.normal(0, 0.02), rng.normal(0, 0.02), rng.normal(0, math.radians(0.5))])
            sol = least_squares(resid, x0, method='lm', xtol=1e-12, ftol=1e-12)
            errs.append(sol.x)
            if progress is not None:
                progress((ri * trials + trial + 1) / (trials * len(entries)))
        e = np.array(errs) * np.array([1000.0, 1000.0, math.degrees(1)])
        T = np.diag([1000.0, 1000.0, math.degrees(1)])
        out['radars'][r['tag']] = _mc_stats(e, T @ C @ T, ('x_mm', 'y_mm', 'yaw_deg'))
        if sq > 0:
            out['radars'][r['tag']]['note'] = ('the reflector survey error enters as a prior in the prediction; '
                                               'the solve holds the reflectors at their surveyed place')
    return out


def monte_carlo(res: ObservabilityResult, trials=None, seed=None, parts=('cameras', 'lidar', 'radar'), progress=None,
                cancel=None) -> dict | None:
    """Verify the predicted σ by simulation: noisy detections solved with the machinery that will solve the real
    ones. Returns {'cameras', 'lidar', 'radar', 'trials', 'seconds', 'agree'} (None when cancelled) and keeps it on
    the result."""
    st = res.settings
    trials = int(trials or st.mc_trials)
    rng = np.random.default_rng(int(st.seed if seed is None else seed))
    t0 = time.perf_counter()
    weights = {'cameras': 0.6, 'lidar': 0.3, 'radar': 0.1}
    use = [p for p in parts if p in weights]
    total = sum(weights[p] for p in use) or 1.0
    done = 0.0
    out = {'trials': trials, 'seed': int(st.seed if seed is None else seed)}

    def sub(w):
        base = done

        def f(x):
            if progress is not None:
                progress(min(1.0, (base + w * x) / total))
        return f
    for part, fn in (('cameras', _mc_cameras), ('lidar', _mc_lidar), ('radar', _mc_radars)):
        if part not in use:
            continue
        r = fn(res, trials, rng, sub(weights[part]), cancel)
        if r is None and cancel is not None and cancel():
            return None
        out[part] = r
        done += weights[part]
    out['seconds'] = time.perf_counter() - t0
    cam = out.get('cameras') or {}
    groups = [(cam.get('optimum') or {}).get('cameras'), (cam.get('optimum') or {}).get('pairs'),
              (out.get('lidar') or {}).get('pairs'), (out.get('radar') or {}).get('radars')]
    out['agree'], out['worst_log_ratio'], out['band'] = _agreement([g for g in groups if g], trials)
    if cam.get('cameras'):
        out['joint_agree'], out['joint_worst_log_ratio'], _ = _agreement([cam['cameras'], cam.get('pairs') or {}],
                                                                          trials)
    res.mc = out
    return out


def _agreement(groups, trials) -> tuple[bool, float, float]:
    """Whether simulated spreads match the predictions: every group's (a camera's, a pair's) mean squared
    Mahalanobis distance within four standard deviations of its number of axes, and every axis's empirical σ within
    4.5 standard errors of the predicted one, either way: |log(σ̂ / σ)| ≤ 4.5 / √(2(n − 1)) (log σ̂ from n trials
    scatters by 1 / √(2(n − 1)); dozens of correlated axes are checked at once). Returns (agree, worst |log ratio|, the
    per-axis band on |log ratio|)."""
    se = 1.0 / math.sqrt(2.0 * max(trials - 1, 1))
    band = 4.5 * se
    worst, ok = 0.0, True
    for g in groups:
        for s in g.values():
            for ax in s['axes'].values():
                if math.isfinite(ax['ratio']):
                    worst = max(worst, abs(math.log(max(ax['ratio'], 1e-9))))
            k = s['dof']
            band_k = 4.0 * math.sqrt(2.0 / (k * trials))
            if math.isfinite(s['mahalanobis2']) and abs(s['mahalanobis2'] / k - 1.0) > band_k:
                ok = False
    ok = ok and worst <= band + 1e-12          # |log ratio| within 4.5 standard errors of log σ̂, either way
    return bool(ok), worst, band


# =========================================================================== the Issues list
_ISSUE_CACHE = {}
_ISSUE_LOCK = threading.Lock()
_CUBES = None


def _issue_key(project, v):
    return json.dumps([project.analysis.observability, [t.to_dict() for t in project.targets],
                       [c.to_dict() for c in v.cameras], [s.to_dict() for s in v.sensors],      # housings block too
                       id(project.vehicle.mesh), [r.to_dict() for r in project.rig],
                       project.section('library'), getattr(project.analysis, 'housings', None)],
                      sort_keys=True, default=str)


class _CubeSource:
    """engine.cube for the Issues list: the analysis result's cube maps when it has them, else its own small cache."""

    def __init__(self, vg, cubes):
        self.vg, self.cubes = vg, list(cubes or [])        # [(position, cube map)]

    def vehicle_geom(self, project, variant=None):
        return self.vg

    def cube(self, pos, vg=None):
        global _CUBES
        for p, c in self.cubes:
            if np.allclose(p, pos, atol=1e-4):
                return c
        from .occlusion import CubeCache
        with _ISSUE_LOCK:
            if _CUBES is None:
                _CUBES = CubeCache(24)
        return _CUBES.get((vg or self.vg).occ, pos)

    def roi_grid(self, project, cell=None, variant=None):
        return None


def issues(project, vg=None, result=None) -> list:
    """The cameras the planned boards cannot measure (serious) or pin only to more than ISSUE_RATIO × the
    requirement (warning), and the radars short of reflectors, for the Issues list; only once calibration targets are
    planned (the page shows every figure). result: the analysis
    result, whose cube maps are used (else they are built once, cached). Cached on the inputs."""
    from .rules import Issue
    if not project.targets:
        return []
    v = project.variant
    try:
        key = _issue_key(project, v)
    except (TypeError, ValueError):
        key = None
    with _ISSUE_LOCK:
        hit = _ISSUE_CACHE.get(key) if key is not None else None
    if hit is not None:
        return list(hit)
    st = dataclass_replace(settings_of(project), suggest=False, lidar_pairs=False, fusion=False)
    cubes = []
    if result is not None and isinstance(getattr(result, 'cubes', None), dict):
        for cb in result.cubes.values():          # each map knows where it was built
            if getattr(cb, 'pos', None) is not None:
                cubes.append((np.asarray(cb.pos, float), cb))
    src = _CubeSource(vg, cubes) if vg is not None else None
    m = _Model(project, src, v, st)
    C, det = m.cov()
    out = []
    for ci, cm in enumerate(m.cams):
        c = cm.cam
        if not det[ci]:
            out.append(Issue('serious', f'{c.tag} sees no calibration board whole: the installation check cannot '
                                        'measure it. Move a board into its view (Calibrate ▸ Calibration '
                                        'observability suggests where).', c.tag, [c.uid], where=(c.x, c.y)))
            continue
        ce, _ = _to_euler(C[6 * ci:6 * ci + 6, 6 * ci:6 * ci + 6], cm.R)
        sig = _sig(ce)
        ratio, axis = _ratio(sig, st.need_deg, st.need_mm)
        if ratio > ISSUE_RATIO:
            unit = 'deg' if axis in ('yaw', 'pitch', 'roll') else 'mm'
            need = st.need_deg if unit == 'deg' else st.need_mm
            out.append(Issue('warning', f'{c.tag}: the planned boards pin its {AXIS_WORDS[axis]} only to '
                                        f'±{_fmt_sig(sig[axis + "_" + unit], unit)}, {_times(ratio)} the '
                                        f'{_need_text(need, unit)} requirement.', c.tag, [c.uid], where=(c.x, c.y)))
    if st.reflectors:
        for r in _radars(project, src, vg, v, st):
            if r['sigma'] is None:
                out.append(Issue('warning', f'{r["tag"]}: {r["note"]}', r['tag'], [r['uid']]))
    if key is not None:
        with _ISSUE_LOCK:
            while len(_ISSUE_CACHE) > 32:
                _ISSUE_CACHE.pop(next(iter(_ISSUE_CACHE)))
            _ISSUE_CACHE[key] = out
    return list(out)


def dataclass_replace(st: ObservabilitySettings, **kw) -> ObservabilitySettings:
    d = st.to_dict()
    d.update(kw)
    return ObservabilitySettings.from_dict(d)


# =========================================================================== output
def _r(v, nd=3):
    return None if v is None or not math.isfinite(float(v)) else round(float(v), nd)


def _sig_doc(sig, nd=4):
    return None if sig is None else {k: _r(v, nd) for k, v in sig.items()}


def summary(res: ObservabilityResult) -> dict:
    """Compact figures (the assistant, the batch report header)."""
    st = res.settings
    return _clean({
        'variant': res.variant,
        'requirement': {'deg': st.need_deg, 'mm': st.need_mm},
        'boards_placed': {'on': st.boards_placed, 'sigma_mm': st.board_sigma_mm, 'sigma_deg': st.board_sigma_deg},
        'pixel_sigma': st.pixel_sigma,
        'cameras': [{'tag': c['tag'], 'boards': [b['name'] for b in c['boards']], 'corners': c['corners'],
                     'sigma': _sig_doc(c['sigma'], 3), 'worst_axis': c['worst_axis'],
                     'ratio': _r(c['ratio'], 2), 'verdict': c['verdict'],
                     'weakest': c['weakest']['text'] if c['weakest'] else None,
                     'suggestion': c['suggestions'][0]['text'] if c['suggestions'] else None}
                    for c in res.cameras],
        'pairs': [{'pair': '–'.join(p['tags']), 'shared': p['shared'], 'ratio': _r(p['ratio'], 2),
                   'verdict': p['verdict']} for p in res.pairs],
        'lidar_pairs': [{'pair': f'{lp["camera"]}–{lp["lidar"]}', 'sigma': _sig_doc(lp['sigma'], 3),
                         'ratio': _r(lp['ratio'], 2), 'verdict': lp['verdict']} for lp in res.lidar_pairs],
        'radars': [{'radar': r['tag'], 'reflectors': r['reflectors'], 'sigma': _sig_doc(r['sigma'], 3),
                    'verdict': r['verdict'] if r['sigma'] is not None else None} for r in res.radars],
        'fusion': [{'camera': f['tag'], 'lidar_support_pct': _r(100 * f['image']['support_share'], 1),
                    'points_per_px': _r(f['image']['median_points_per_px'], 4)} for f in (res.fusion or [])],
        'monte_carlo': None if res.mc is None else {'trials': res.mc['trials'], 'agree': res.mc['agree'],
                                                     'worst_log_ratio': _r(res.mc['worst_log_ratio'], 3)},
        'findings': [{'severity': f['severity'], 'message': f['message']} for f in res.findings
                     if f['severity'] != 'good'],
        'assumptions': ASSUMPTIONS,
    })


def _camera_doc(c) -> dict:
    return {'tag': c['tag'], 'verdict': c['verdict'], 'ratio': _r(c['ratio'], 3), 'worst_axis': c['worst_axis'],
            'corners': c['corners'], 'median_corner_distance_m': _r(c['distance_m'], 3),
            'sigma': _sig_doc(c['sigma']), 'sigma_corners_only': _sig_doc(c['sigma_corners']),
            'angles': 'Euler yaw/pitch/roll (pitch positive down)' if c['euler'] else
                      'rotations about the camera\'s own z, y, x axes (it looks nearly straight down)',
            'weakest_direction': c['weakest'], 'condition_number': _r(c['condition'], 1),
            'boards': [{'name': b['name'], 'corners': b['corners'], 'distance_m': b['distance_m'],
                        'square_px': b['square_px'], 'information_share': _r(b['share'], 4),
                        'ratio_without': _r(b['ratio_without'], 3)} for b in c['boards']],
            'not_counted': c['unseen'], 'why': c['reasons'],
            'suggestions': [{k: (_r(v, 4) if isinstance(v, float) else v) for k, v in s.items()}
                            for s in c['suggestions']]}


def to_doc(res: ObservabilityResult, project=None) -> dict:
    """The JSON export (strict JSON: a σ nothing pins is null)."""
    st = res.settings
    return _clean({
        'schema': SCHEMA,
        'project': getattr(project, 'name', ''), 'variant': res.variant, 'frame': FRAME,
        'units': {'angles': 'degrees', 'positions': 'mm (σ), m (places)', 'pixels': 'px'},
        'settings': st.to_dict(),
        'assumptions': ASSUMPTIONS,
        'method': {'cameras': 'Fisher information of the counted corners (lens model, pixel noise), the boards\' '
                              'placement as priors when placed by hand; the covariance of the installation '
                              'check\'s joint solve (corners weighted at 0.5 px): H⁻¹ M H⁻¹',
                   'pairs': 'relative pose of camera b in camera a\'s own axes (x forward, y left, z up; roll about '
                            'x, pitch about y, yaw about z), from the joint covariance',
                   'lidar': 'the lidar–camera calibration\'s planned board poses: corners and lidar points on each '
                            'board, the boards marginalised; 1σ of the lidar\'s pose in the vehicle frame',
                   'radar': 'range and azimuth of the corner reflectors in each radar\'s field; x, y, yaw'},
        'cameras': [_camera_doc(c) for c in res.cameras],
        'boards': [{k: (v if not isinstance(v, float) else _r(v, 4)) for k, v in b.items()} for b in res.boards],
        'pairs': [{'cameras': p['tags'], 'shared_boards': p['shared'], 'baseline_m': _r(p['baseline_m'], 3),
                   'sigma': _sig_doc(p['sigma']), 'sigma_corners_only': _sig_doc(p['sigma_corners']),
                   'ratio': _r(p['ratio'], 3), 'worst_axis': p['worst_axis'], 'verdict': p['verdict']}
                  for p in res.pairs],
        'lidar_pairs': [{'camera': lp['camera'], 'lidar': lp['lidar'], 'image_share': _r(lp['share'], 4),
                         'board': lp['board'], 'poses': lp['poses'], 'points_per_pose': lp['points'],
                         'sigma_vehicle': _sig_doc(lp['sigma']),
                         'sigma_camera_frame': _sig_doc(lp['sigma_camera']), 'tolerance': lp['tolerance'],
                         'ratio': _r(lp['ratio'], 3), 'worst_axis': lp['worst_axis'], 'verdict': lp['verdict'],
                         'limited_by': lp['limited_by'],
                         'ratio_if_10x_sharper': {k: _r(v, 3) for k, v in lp['if_firmer'].items()},
                         'notes': lp['notes']} for lp in res.lidar_pairs],
        'radars': [{'radar': r['tag'], 'model': r['model'], 'reflectors': r['reflectors'],
                    'sigma': _sig_doc(r['sigma']), 'ratio': _r(r['ratio'], 3), 'verdict': r['verdict'],
                    'note': r['note']} for r in res.radars],
        'reflectors': [dict(r, rcs_m2=round(reflector_rcs(r['edge_mm']), 2)) for r in st.reflectors],
        'fusion': None if res.fusion is None else [_fusion_doc(f) for f in res.fusion],
        'monte_carlo': res.mc,
        'findings': [{'severity': f['severity'], 'kind': f['kind'], 'message': f['message'], 'objects': f['tags']}
                     for f in res.findings],
    })


def _fusion_doc(f) -> dict:
    img = f['image']
    return {'camera': f['tag'], 'lidars': f['lidars'],
            'classes': [{'name': k['name'], 'min_points': k['min_points'], 'min_px': k['min_px'],
                         'roi_samples': k['roi_samples'], 'roi_supported_share': _r(k['roi_supported_share'], 4),
                         'roi_without_points_share': _r(k['roi_zero_share'], 4),
                         'gaps_deg': [[_r(a, 1), _r(b, 1)] for a, b in k['gaps']],
                         'at_ranges': [{kk: (_r(vv, 4) if isinstance(vv, float) else vv) for kk, vv in row.items()}
                                       for row in k['rows']]} for k in f['classes']],
            'image': {'scene_at_m': img['at_m'], 'lidar_support_share': _r(img['support_share'], 4),
                      'median_points_per_px': _r(img['median_points_per_px'], 5),
                      'median_px_per_point': _r(img['median_px_per_point'], 2), 'unsupported': img['unsupported'],
                      'grid': list(img['grid']),
                      'points_per_px': [[_r(x, 5) for x in row] for row in img['points_per_px']]}}


CSV_COLUMNS = ('kind', 'name', 'verdict', 'ratio', 'worst_axis', 'yaw_deg', 'pitch_deg', 'roll_deg', 'x_mm', 'y_mm',
               'z_mm', 'boards', 'corners', 'condition', 'weakest', 'suggestion')


def to_csv(res: ObservabilityResult) -> str:
    """One row per camera, camera pair, lidar–camera pair and radar: the 1σ per axis and the verdict."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator='\n')
    w.writerow(CSV_COLUMNS)

    def sig(s, k):
        if s is None or k not in s:
            return ''
        return '' if not math.isfinite(s[k]) else round(s[k], 4)
    for c in res.cameras:
        w.writerow(['camera', c['tag'], c['verdict'], _r(c['ratio'], 3) if c['sigma'] else '', c['worst_axis']]
                   + [sig(c['sigma'], k) for k in SIGMA_KEYS]
                   + [' '.join(b['name'] for b in c['boards']), c['corners'], _r(c['condition'], 1) or '',
                      c['weakest']['text'] if c['weakest'] else '',
                      c['suggestions'][0]['text'] if c['suggestions'] else ''])
    for p in res.pairs:
        s = p['sigma']
        w.writerow(['pair', '–'.join(p['tags']), p['verdict'], _r(p['ratio'], 3), p['worst_axis'],
                    sig(s, 'yaw_deg'), sig(s, 'pitch_deg'), sig(s, 'roll_deg'), sig(s, 'x_mm'), sig(s, 'y_mm'),
                    sig(s, 'z_mm'), ' '.join(p['shared']), '', '', 'about and along ' + p['tags'][0] + '\'s axes', ''])
    for lp in res.lidar_pairs:
        w.writerow(['lidar', f'{lp["camera"]}–{lp["lidar"]}', lp['verdict'], _r(lp['ratio'], 3), lp['worst_axis']]
                   + [sig(lp['sigma'], k) for k in SIGMA_KEYS]
                   + [f'{len(lp["poses"])} planned poses', '', '', '', ' '.join(lp['notes'])])
    for r in res.radars:
        s = r['sigma']
        w.writerow(['radar', r['tag'], r['verdict'] if s else '', _r(r['ratio'], 3) if s else '', r['worst_axis'],
                    sig(s, 'yaw_deg'), '', '', sig(s, 'x_mm'), sig(s, 'y_mm'), '', ' '.join(r['reflectors']), '', '',
                    '', r['note']])
    return buf.getvalue()


# =========================================================================== the plan image
def _hex(h):
    return int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)


STATUS_BGR = {'good': (12, 163, 12), 'warning': (25, 178, 250), 'serious': (90, 131, 236), 'critical': (59, 59, 208)}
SEQ_UI = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6', '#256abf',
          '#1c5cab', '#184f95', '#104281', '#0d366b']       # the app's sequential ramp (ui.theme.SEQ)


def _ascii(text) -> str:
    """Text OpenCV's Hershey fonts can draw (they have no ·, –, σ, ° or ×)."""
    for a, b in (('·', '-'), ('–', '-'), ('—', '-'), ('σ', 'sigma'), ('°', ' deg'), ('×', 'x'), ('±', '+/-'),
                 ('▸', '>'), ('²', '2'), ('’', "'")):
        text = text.replace(a, b)
    return text.encode('ascii', 'replace').decode('ascii')


def png(res: ObservabilityResult, size=900) -> bytes:
    """The plan (forward right, left up): the body outline, the boards (light: one camera counts it, dark: two or
    more, grey: none), each camera with its verdict and lines to the boards it counts, the suggested moves and added
    boards (dashed), the radars and the reflectors (those far out at the edge)."""
    import cv2
    m: _Model = res._model
    head = 56
    img = np.full((size + head, size, 3), (30, 29, 28), np.uint8)
    pts = [o for o in res.outline] + [np.array([[b.centre[0], b.centre[1]]]) for b in m.boards] + \
          [np.array([[cm.pos[0], cm.pos[1]]]) for cm in m.cams] + \
          [np.array([[s['x'], s['y']]]) for c in res.cameras for s in c['suggestions'] if 'x' in s]
    allp = np.vstack(pts) if pts else np.zeros((1, 2))
    lo, hi = allp.min(0) - 1.0, allp.max(0) + 1.0
    sc = (size - 40) / max(hi[0] - lo[0], hi[1] - lo[1], 1.0)
    cx, cy = (lo + hi) / 2

    def w2s(x, y):
        return int(round(size / 2 + (x - cx) * sc)), int(round(head + size / 2 - (y - cy) * sc))
    font, aa = cv2.FONT_HERSHEY_SIMPLEX, cv2.LINE_AA
    amber = STATUS_BGR['warning']

    def dashed(poly, closed=True):
        P = list(poly) + ([poly[0]] if closed else [])
        for a, e in zip(P[:-1], P[1:]):
            n = max(int(math.hypot(e[0] - a[0], e[1] - a[1]) / 6), 1)
            for k in range(0, n, 2):
                p0 = (int(a[0] + (e[0] - a[0]) * k / n), int(a[1] + (e[1] - a[1]) * k / n))
                p1 = (int(a[0] + (e[0] - a[0]) * (k + 1) / n), int(a[1] + (e[1] - a[1]) * (k + 1) / n))
                cv2.line(img, p0, p1, amber, 1, aa)
    for o in res.outline:
        P = np.array([w2s(x, y) for x, y in o], np.int32)
        cv2.polylines(img, [P], False, (150, 148, 142), 1, aa)
    tags = {c['uid']: c for c in res.cameras}
    for o in m.seen:
        cv2.line(img, w2s(*m.cams[o.cam].pos[:2]), w2s(*m.boards[o.board].centre[:2]), (80, 78, 74), 1, aa)
    light, dark = _hex(SEQ_UI[2])[::-1], _hex(SEQ_UI[7])[::-1]
    for bi, b in enumerate(m.boards):
        n = sum(1 for o in m.seen if o.board == bi)
        col = (70, 70, 70) if n == 0 else light if n == 1 else dark
        P = np.array([w2s(x, y) for x, y, _ in _outline(b.t)], np.int32)
        cv2.fillPoly(img, [P], col, aa)
        cv2.putText(img, _ascii(b.t.name), (int(P[:, 0].max()) + 3, int(P[:, 1].min()) + 10), font, 0.4,
                    (220, 219, 214), 1, aa)
    for c in res.cameras:
        for s in c['suggestions']:
            if s['kind'] == 'move':
                b = next((b for b in m.boards if b.t.uid == s['board']), None)
                if b is None:
                    continue
                t = copy.copy(b.t)
                t.x, t.y, t.yaw = s['x'], s['y'], s['yaw']
                dashed([w2s(*b.centre[:2]), w2s(s['x'], s['y'])], closed=False)
            elif s['kind'] == 'add':
                from .project import Target
                t = Target(x=s['x'], y=s['y'], yaw=s['yaw'], rows=s['rows'], cols=s['cols'], square_mm=s['square_mm'])
            else:
                continue
            dashed([w2s(x, y) for x, y, _ in _outline(t)])
            q = w2s(s['x'], s['y'])
            cv2.putText(img, _ascii(f'for {c["tag"]}'), (q[0] - 18, q[1] + 4), font, 0.35, amber, 1, aa)
    for cm in m.cams:
        c = tags[cm.cam.uid]
        p = w2s(*cm.pos[:2])
        cv2.circle(img, p, 5, STATUS_BGR[c['verdict']], -1, aa)
        cv2.putText(img, _ascii(c['tag']), (p[0] + 7, p[1] - 5), font, 0.45, (236, 235, 230), 1, aa)
    brown = _hex('#a0743c')[::-1]
    for r in res.settings.reflectors:
        p = w2s(r['x'], r['y'])
        name = r['name']
        if not (8 <= p[0] <= size - 8 and head + 8 <= p[1] <= head + size - 8):
            name += f' at {r["x"]:.0f}, {r["y"]:.0f} m'
            p = (min(max(p[0], 12), size - 12), min(max(p[1], head + 12), head + size - 12))
        tri = np.array([(p[0], p[1] - 6), (p[0] - 5, p[1] + 4), (p[0] + 5, p[1] + 4)], np.int32)
        cv2.fillPoly(img, [tri], brown, aa)
        tw = cv2.getTextSize(_ascii(name), font, 0.4, 1)[0][0]
        x = p[0] + 7 if p[0] + 7 + tw < size - 4 else p[0] - 7 - tw
        cv2.putText(img, _ascii(name), (x, p[1] + 4), font, 0.4, (220, 219, 214), 1, aa)
    for s in m.v.sensors:
        if s.visible and s.kind == 'radar':
            p = w2s(s.x, s.y)
            cv2.rectangle(img, (p[0] - 3, p[1] - 3), (p[0] + 3, p[1] + 3), brown, -1)
    good = res.metrics['good']
    st = res.settings
    cv2.putText(img, _ascii(f'Calibration observability - {res.variant}'), (14, 20), font, 0.55, (236, 235, 230), 1,
                aa)
    cv2.putText(img, _ascii(f'{good} of {len(res.cameras)} cameras within the requirement ({st.need_deg:g} deg, '
                            f'{st.need_mm:g} mm, 1 sigma); boards placed to +/-{st.board_sigma_mm:g} mm'
                            if st.boards_placed else
                            f'{good} of {len(res.cameras)} cameras within the requirement ({st.need_deg:g} deg, '
                            f'{st.need_mm:g} mm, 1 sigma); boards taken as exact'), (14, 38), font, 0.42,
                (200, 199, 194), 1, aa)
    cv2.putText(img, 'boards: light one camera, dark two or more, grey none; dot: verdict; dashed: suggested', (14, 52),
                font, 0.38, (160, 160, 155), 1, aa)
    ok, buf = cv2.imencode('.png', img)
    return buf.tobytes()


def fusion_png(res: ObservabilityResult, cell=6) -> bytes | None:
    """Every camera's map of lidar points per camera pixel (the scene at fusion_at_m), side by side: black where no
    lidar point lands, the app's sequential ramp from 1/1000 to 1 point per pixel."""
    import cv2
    if not res.fusion:
        return None
    from .sensor_coverage import SEQ
    lut = np.array([_hex(c)[::-1] for c in SEQ], np.uint8)
    tiles = []
    for f in res.fusion:
        A = np.asarray(f['image']['points_per_px'], float)
        nv, nu = A.shape
        fin = np.isfinite(A)
        t = np.clip((np.log10(np.maximum(np.where(fin, A, 1e-9), 1e-9)) + 3.0) / 3.0, 0, 1)
        idx = np.rint(t * (len(SEQ) - 1)).astype(int)
        tile = lut[idx]
        tile[~fin] = (52, 51, 49)
        tile[fin & (np.where(fin, A, 0) <= 0)] = (0, 0, 0)
        tile = cv2.resize(tile, (nu * cell, nv * cell), interpolation=cv2.INTER_NEAREST)
        lab = np.full((22, nu * cell, 3), (30, 29, 28), np.uint8)
        cv2.putText(lab, _ascii(f'{f["tag"]}  {100 * f["image"]["support_share"]:.0f} % with lidar'), (4, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (236, 235, 230), 1, cv2.LINE_AA)
        tiles.append(np.vstack([lab, tile]))
    w = max(t.shape[1] for t in tiles)
    h = max(t.shape[0] for t in tiles)
    cols = min(3, len(tiles))
    rows = int(math.ceil(len(tiles) / cols))
    img = np.full((rows * (h + 8) + 8, cols * (w + 8) + 8, 3), (30, 29, 28), np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        img[8 + r * (h + 8):8 + r * (h + 8) + t.shape[0], 8 + c * (w + 8):8 + c * (w + 8) + t.shape[1]] = t
    ok, buf = cv2.imencode('.png', img)
    return buf.tobytes()


def file_names(project, variant=None) -> list:
    """The names files() writes for a variant (fusion_density.png only with a visible lidar and the fusion on)."""
    v = variant or project.variant
    st = settings_of(project)
    out = ['observability.json', 'observability.csv', 'observability.png']
    if st.fusion and any(s.visible and s.kind == 'lidar' for s in v.sensors) and any(c.visible for c in v.cameras):
        out.append('fusion_density.png')
    return out


def files(res: ObservabilityResult, project=None) -> list:
    """[(name, bytes)]: observability.json, observability.csv, observability.png and, with lidars,
    fusion_density.png."""
    out = [('observability.json', (json.dumps(to_doc(res, project), indent=1, ensure_ascii=False, allow_nan=False)
                                   + '\n').encode('utf-8')),
           ('observability.csv', to_csv(res).encode('utf-8')),
           ('observability.png', png(res))]
    fp = fusion_png(res)
    if fp is not None:
        out.append(('fusion_density.png', fp))
    return out


def _clean(o):
    """A document safe for strict JSON: numpy scalars and arrays as Python numbers and lists, inf and nan as null,
    tuples as lists."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if math.isfinite(float(o)) else None
    if o is None or isinstance(o, str):
        return o
    return str(o)


def export(res: ObservabilityResult, project, out_dir) -> list[str]:
    """Write the files into out_dir; returns their paths."""
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for name, data in files(res, project):
        path = os.path.join(out_dir, name)
        with open(path, 'wb') as fh:
            fh.write(data)
        paths.append(path)
    return paths
