"""Live calibration coach: while a technician holds a calibration board in front of a camera, how firmly the views
taken so far pin what the calibration measures, where to hold the board next to firm up the loosest part, and when
to stop.

What a hand-held board can pin
- A board held by hand stands at an unknown pose. On its own it measures the camera's intrinsics, the lens
  calibration's Kannala–Brandt model (fx, fy, cx, cy, k1–k4, as core.calib fits it with cv::fisheye), not where the
  camera sits on the vehicle: each view's board pose is a nuisance the solve estimates too. Mode 'lens'.
- A board both a camera and a lidar see pins the lidar's pose relative to the camera (T_camera_lidar, core.lidar_calib);
  the board's pose is again a nuisance. Mode 'lidar', with the lidar–camera calibration's board and tolerances.
- Boards laid on the floor where the plan puts them pin the camera's pose in the vehicle (the installation check):
  core.observability plans and judges those (with its own board-move suggestions); nothing to coach live.

Information
- Lens: each accepted view's corners (cols × rows inner corners, every one, as the lens calibration's detector needs
  the whole board) with independent pixel noise σ on each axis (the observability settings' pixel σ, typical 0.3 px
  for sub-pixel corners), differentiated through the Kannala–Brandt projection u = fx θd x/ρ + cx, v = fy θd y/ρ + cy,
  θd = θ (1 + k1 θ² + k2 θ⁴ + k3 θ⁶ + k4 θ⁸) (OpenCV pixel coordinates at the sensor's resolution) with respect to the
  eight intrinsics and the view's board pose (rotation and shift, 6). The board pose is marginalised per view (Schur
  complement), so the views' information on the intrinsics adds up: I = Σ_views (A − B D⁻¹ Bᵀ) / σ². The linearisation
  point is the camera's current lens model (its unit calibration when it has one, else the library lens; a lens that is
  not Kannala–Brandt is fitted with core.exporters.kb_coefficients, as the lens calibration's start).
- Lidar–camera: core.observability's model of that calibration (observability._lidar_model, the calibration's own
  weights: corners at σ, a board's lidar points at their RMS range noise across the plane, at most 2 500 a pose) per
  accepted board pose, the board's pose marginalised per view; the covariance is that estimator's, H⁻¹ M H⁻¹, where
  both the marginal H and the marginal M of the extrinsic add up view by view.
- A vague prior (each lens parameter ±10 rad of ray angle at the image edge; the lidar ±1000 rad and ±1000 m) keeps
  the inverse finite before the views pin everything; what it leaves is reported as not determined (beyond 100× the
  requirement, as core.observability).

What is reported (lock indicators)
- Lens, in degrees of ray angle (the lens model turns a pixel into a ray; its error tilts that ray): fx, fy, cx, cy each
  as its own 1σ times the largest change of a ray per unit of it over the reachable image; 'distortion' the 1σ ray
  error from k1–k4 together (their covariance block, so the correlated k's count as the curve they make, not one by
  one); 'rays' the 1σ ray error from all eight, the worst over the reachable image. The reachable image: pixels whose
  ray meets a place a hand-held board can be (its centre 0.5–2 m high, near_m–far_m from the lens, nothing of the
  vehicle in between); elsewhere (the body, the floor beneath a downward camera) the lens model can only extrapolate,
  and the coverage says how much of the image that is. Requirement: the scorecard's 'observability.worst_deg' target
  when there is one, else the observability settings' need_deg: the lens's share of a ray's error is held to the same
  angle as the camera's pose (a heuristic budget).
- Lidar–camera: 1σ of the lidar's yaw, pitch, roll (degrees) and x, y, z (mm) in the vehicle frame against the
  scorecard's 'lidar_calib.planned_sigma_deg' / 'planned_sigma_mm' targets, else the lidar–camera calibration's
  tolerances.

Next-best view
- Candidate poses a person can hold: the board's centre on the ray of a grid of image points (9 × 7 over the image),
  at near_m–far_m (4 distances, geometric), its centre 0.5–2 m above the ground and its lower edge above it, facing the
  lens and then tilted (top towards or away, a side towards or away: −45, −25, 0, 25, 45°, at most max_tilt_deg in
  all); kept when the whole board lies in the image (1.5 % margin, 3° inside the lens circle), its squares are at
  least min_square_px wide (the foreshortened side), nothing of the vehicle hides a corner and (lidar) at least four
  lidar channels cross it with 20 points. A candidate within the duplicate distance of a view already taken is left
  out.
- Criterion 'weakest' (default): the candidate whose information leaves the smallest worst lock ratio (the 1σ of the
  loosest indicator over its requirement); 'd_optimal': the largest log det of the information. Ties: the first
  candidate in grid order.
- Guidance from the board's current pose (the last detection) or, without one, from the lens: moves along the
  technician's own axes (they face the camera: their left is the camera's right), height and distance, then the tilts
  from facing the lens squarely; also the target in the vehicle frame.

Accepting a view
- Detection: the lens calibration's detector (core.calib.find_board: the whole image, then rectified pinhole views
  through the lens model) for both boards, on a copy at most 1280 px wide (lidar board: 1600 px); its time stays
  bounded on a frame without a board, which the lidar–camera calibration's exhaustive search does not (that one runs
  later, on the saved frames). The board's pose from its corners (lidar_calib._board_pnp: PnP on the bearing rays,
  then least squares in pixels).
- Sharp: rig_health's sharpness (variance of the Laplacian) of the board's region, scaled so that its squares are
  12 px wide (shrunk or enlarged: a blur counts in proportion to the squares, whatever the board's size in the
  image), at least sharp_min (60, a heuristic threshold measured on rendered boards: sharp ones, native or enlarged
  from half size, 106–3 700; a Gaussian blur of 0.3 square 9–35, of 0.6 square 2–4.4).
- New: no accepted view within dup_m of its centre with its normal within dup_deg.

Simulated technician (testing and planning without hardware): a seeded sequence of board poses, either the coach's
advice (with the technician's own error: jitter_m, jitter_deg, Gaussian) or random hand-held poses from the same
feasible set; rendered through the camera's lens model as frames (core.calib.render_board for a lens board,
core.lidar_calib._paint_board for a lidar board) and detected, or counted from the poses directly (run()).

Limits: predictions from the Fisher information at the current lens model and the estimated board poses, not a solve:
corners equally noisy everywhere (a corner far out in a fisheye is noisier in practice), boards flat and printed true,
the lens's pixels square to the model's precision. The ray-angle budget is a heuristic, not a standard. The lidar part
needs the lidar's scan of each accepted pose: the live rig streams cameras only, so the frames are saved for pairing
with the lidar's recording (or simulated).
"""
from __future__ import annotations

import copy
import csv
import io
import json
import math
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from . import lidar_calib as LC
from . import observability as OB
from .lens import resolve
from .mathutil import rot_x, rot_y

SCHEMA = 'vantage.calib_coach/1'
FRAME = 'vehicle (ISO 8855, rear axle)'
OPTICAL = LC.OPTICAL
PIXELS = ('sensor pixels at the module\'s full resolution, OpenCV convention (pixel centre at integer i) for the '
          'corners and the lens parameters; the overlay draws in the lens model\'s (centre at i + 0.5)')
MODES = ('lens', 'lidar')
CRITERIA = ('weakest', 'd_optimal')
TECHNICIANS = ('coached', 'random')
LENS_PARAMS = ('fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'k3', 'k4')
LENS_LOCKS = ('fx', 'fy', 'cx', 'cy', 'distortion', 'rays')
LIDAR_LOCKS = OB.SIGMA_KEYS
LOCK_LABELS = {'fx': 'Focal x', 'fy': 'Focal y', 'cx': 'Centre x', 'cy': 'Centre y', 'distortion': 'Distortion',
               'rays': 'Rays (all)', 'yaw_deg': 'Yaw', 'pitch_deg': 'Pitch', 'roll_deg': 'Roll', 'x_mm': 'x',
               'y_mm': 'y', 'z_mm': 'z'}
UNDETERMINED = OB.UNDETERMINED
PRIOR_RAD = 10.0                    # lens prior: each parameter ±this much ray angle at the edge
PRIOR_LIDAR = 1000.0                # lidar prior: ±this many rad and m
TILTS = (-45.0, -25.0, 0.0, 25.0, 45.0)
GRID_U, GRID_V = 9, 7               # candidate image points
N_DIST = 4
EDGE_MARGIN = 0.015                 # a candidate board keeps this share of the image width from the border
CIRCLE_MARGIN_DEG = 3.0             # and this far inside the lens circle
LOCK_GRID = (32, 24)                # pixels the lens indicators are evaluated on
SHARP_SQUARE_PX = 12.0              # the board's region is judged for sharpness with its squares this wide
GROUND_CLEAR_M = 0.05               # a board's lower edge at least this high
MAX_LIDAR_CANDIDATES = 160          # lidar candidates evaluated (deterministic stride over the feasible ones)
MOVE_MIN_M = 0.05                   # guidance leaves out smaller moves
TURN_MIN_DEG = 5.0                  # and smaller turns
ASSUMPTIONS = (
    'Fisher information of the views at the camera\'s current lens model and the board poses estimated from the '
    'corners: corner noise independent and equal everywhere (a corner far out in a fisheye is noisier in practice), '
    'boards flat and printed true. Lens: the Kannala-Brandt model the lens calibration fits (cv::fisheye), each view\'s '
    'board pose marginalised. Lidar-camera: core.observability\'s model of that calibration (the board planes from '
    'the lidar\'s beams, the corners from the camera; board edges and the reflectivity checker left out, they only '
    'add). The ray-angle budget for the lens (its error held to the same angle as the camera\'s pose) is a heuristic.')
LIMITS = {'cols': (3, 30), 'rows': (3, 30), 'square_mm': (5.0, 500.0), 'pixel_sigma': (0.0, 10.0),
          'need_deg': (0.0, 30.0), 'need_mm': (0.0, 1000.0), 'near_m': (0.2, 20.0), 'far_m': (0.3, 30.0),
          'low_m': (0.0, 3.0), 'high_m': (0.2, 4.0), 'max_tilt_deg': (0.0, 70.0), 'min_square_px': (0.0, 100.0),
          'dup_m': (0.0, 2.0), 'dup_deg': (0.0, 90.0), 'sharp_min': (0.0, 10000.0), 'detect_hz': (0.1, 30.0),
          'max_views': (1, 200), 'jitter_m': (0.0, 1.0), 'jitter_deg': (0.0, 30.0), 'random_runs': (1, 50),
          'seed': (0, 999999)}
INT_FIELDS = ('cols', 'rows', 'max_views', 'random_runs', 'seed')
CHOICES = {'mode': MODES, 'criterion': CRITERIA, 'technician': TECHNICIANS}


# =========================================================================== settings
@dataclass
class CoachSettings:
    mode: str = 'lens'                 # lens (intrinsics) | lidar (lidar–camera extrinsic)
    camera: str = ''                   # camera tag ('' = the first visible camera; lidar mode: the best pair's)
    lidar: str = ''                    # lidar tag ('' = the camera's best partner)
    cols: int = 9                      # lens board: inner corners across (the lens calibration's defaults)
    rows: int = 6
    square_mm: float = 60.0
    pixel_sigma: float = 0.0           # corner noise px 1σ; 0: the observability settings' (typical 0.3 px)
    need_deg: float = 0.0              # requirement; 0: the scorecard's, else the observability / lidar–camera one
    need_mm: float = 0.0               # lidar mode
    near_m: float = 0.5                # lens board distance from the lens (lidar mode: the lidar–camera plan's)
    far_m: float = 3.0
    low_m: float = 0.5                 # board centre height above the ground a person holds it at
    high_m: float = 2.0
    max_tilt_deg: float = 50.0
    min_square_px: float = 0.0         # 0: the observability settings' (8 px)
    criterion: str = 'weakest'         # weakest | d_optimal
    dup_m: float = 0.10                # a view this close (centre) ...
    dup_deg: float = 8.0               # ... and turned this little from an accepted one is a duplicate
    sharp_min: float = 60.0            # rig_health sharpness of the board's region, squares 12 px (heuristic)
    detect_hz: float = 2.0             # detections a second on the live page (one at a time, in the background)
    max_views: int = 40                # the simulated technician gives up after this many views
    technician: str = 'coached'        # the live page's simulated technician: coached | random
    jitter_m: float = 0.05             # how far the simulated technician misses the advice, 1σ
    jitter_deg: float = 3.0
    random_runs: int = 5               # run(): random technicians compared (seeds seed, seed + 1, …)
    seed: int = 1

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d: dict | None) -> 'CoachSettings':
        s = CoachSettings()
        d = d if isinstance(d, dict) else {}
        for k in ('camera', 'lidar'):
            if isinstance(d.get(k), str):
                setattr(s, k, d[k].strip()[:32])
        for k, opts in CHOICES.items():
            v = str(d.get(k, '')).strip().lower()
            if v in opts:
                setattr(s, k, v)
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
        if s.high_m < s.low_m:
            s.low_m, s.high_m = s.high_m, s.low_m
        return s


def settings_of(project) -> CoachSettings:
    return CoachSettings.from_dict(getattr(project.analysis, 'calib_coach', None))


def _scorecard_target(project, metrics) -> tuple[float, str] | None:
    """(target, requirement id) of the tightest 'at most' scorecard requirement on one of these metrics."""
    try:
        from . import requirements as RQ
        items = RQ.settings_of(project).items
    except Exception:                    # noqa: BLE001  (a scorecard that cannot be read: none)
        return None
    best = None
    for it in items:
        if it.get('metric') in metrics and it.get('op') in ('<=', '<'):
            try:
                t = float(it.get('target'))
            except (TypeError, ValueError):
                continue
            if math.isfinite(t) and t > 0 and (best is None or t < best[0]):
                best = (t, str(it.get('id', '')))
    return best


def requirement(project, st: CoachSettings, lst=None) -> dict:
    """{'deg', 'mm', 'source_deg', 'source_mm'}: the 1σ the locks are held to."""
    ost = OB.settings_of(project)
    out = {}
    if st.mode == 'lidar':
        lst = lst or LC.LidarCalibSettings.from_dict(getattr(project.analysis, 'lidar_calib', None))
        pairs = (('deg', st.need_deg, ('lidar_calib.planned_sigma_deg',), lst.ang_tol_deg),
                 ('mm', st.need_mm, ('lidar_calib.planned_sigma_mm', 'lidar_calib.sigma_mm'), lst.pos_tol_mm))
        fallback = 'the lidar–camera calibration\'s tolerance'
    else:
        pairs = (('deg', st.need_deg, ('observability.worst_deg',), ost.need_deg),)
        fallback = 'the calibration observability requirement'
    for unit, own, metrics, default in pairs:
        if own > 0:
            out[unit], out[f'source_{unit}'] = float(own), 'the coach\'s settings'
            continue
        sc = _scorecard_target(project, metrics)
        if sc is not None:
            out[unit], out[f'source_{unit}'] = sc[0], f'the scorecard ({sc[1]})'
        else:
            out[unit], out[f'source_{unit}'] = float(default), fallback
    return out


# =========================================================================== the Kannala–Brandt lens model
def lens_params(intr) -> np.ndarray:
    """[fx, fy, cx, cy, k1..k4] of a resolved lens (OpenCV pixel coordinates, sensor resolution): the lens
    calibration's model of it (core.calib.nominal_record's)."""
    from .exporters import kb_coefficients
    k, _ = kb_coefficients(intr)
    return np.array([intr.f_px, intr.f_px * intr.aspect, intr.cx - 0.5, intr.cy - 0.5, *k], float)


def kb_project(p, P) -> tuple[np.ndarray, np.ndarray]:
    """(uv (..., 2) OpenCV pixels, θ) of optical-frame points P (..., 3) through the Kannala–Brandt parameters p."""
    P = np.asarray(P, float)
    x, y, z = P[..., 0], P[..., 1], P[..., 2]
    rho = np.hypot(x, y)
    th = np.arctan2(rho, z)
    t2 = th * th
    td = th * (1 + t2 * (p[4] + t2 * (p[5] + t2 * (p[6] + t2 * p[7]))))
    s = np.where(rho > 1e-12, td / np.where(rho > 1e-12, rho, 1.0), 0.0)
    return np.stack([p[0] * s * x + p[2], p[1] * s * y + p[3]], -1), th


def _kb_param_jac(p, P) -> np.ndarray:
    """d uv / d p (..., 2, 8), analytic."""
    P = np.asarray(P, float)
    x, y, z = P[..., 0], P[..., 1], P[..., 2]
    rho = np.hypot(x, y)
    th = np.arctan2(rho, z)
    t2 = th * th
    td = th * (1 + t2 * (p[4] + t2 * (p[5] + t2 * (p[6] + t2 * p[7]))))
    ok = rho > 1e-12
    ix = np.where(ok, x / np.where(ok, rho, 1.0), 0.0)
    iy = np.where(ok, y / np.where(ok, rho, 1.0), 0.0)
    J = np.zeros(P.shape[:-1] + (2, 8))
    J[..., 0, 0] = td * ix
    J[..., 1, 1] = td * iy
    J[..., 0, 2] = 1.0
    J[..., 1, 3] = 1.0
    for i in range(4):
        tp = th ** (2 * i + 3)
        J[..., 0, 4 + i] = p[0] * tp * ix
        J[..., 1, 4 + i] = p[1] * tp * iy
    return J


def _kb_point_jac(p, P, h=1e-6) -> np.ndarray:
    """d uv / d P (..., 2, 3), central differences."""
    P = np.asarray(P, float)
    out = np.zeros(P.shape[:-1] + (2, 3))
    for k in range(3):
        d = np.zeros(3)
        d[k] = h
        a, _ = kb_project(p, P + d)
        b, _ = kb_project(p, P - d)
        out[..., :, k] = (a - b) / (2 * h)
    return out


def kb_unproject(p, u, v) -> np.ndarray:
    """Optical-frame unit rays (..., 3) of OpenCV pixels (u, v) through the Kannala–Brandt parameters p."""
    a = (np.asarray(u, float) - p[2]) / p[0]
    b = (np.asarray(v, float) - p[3]) / p[1]
    td = np.hypot(a, b)
    th = td.copy()
    for _ in range(20):
        t2 = th * th
        g = th * (1 + t2 * (p[4] + t2 * (p[5] + t2 * (p[6] + t2 * p[7])))) - td
        dg = 1 + t2 * (3 * p[4] + t2 * (5 * p[5] + t2 * (7 * p[6] + t2 * 9 * p[7])))
        th = np.clip(th - g / np.where(np.abs(dg) < 1e-9, 1e-9, dg), 0.0, math.pi)
    s = np.where(td > 1e-12, np.sin(th) / np.where(td > 1e-12, td, 1.0), 0.0)
    return np.stack([a * s, b * s, np.cos(th)], -1)


def _skew(q) -> np.ndarray:
    """(..., 3, 3) cross-product matrices."""
    q = np.asarray(q, float)
    S = np.zeros(q.shape[:-1] + (3, 3))
    S[..., 0, 1], S[..., 0, 2] = -q[..., 2], q[..., 1]
    S[..., 1, 0], S[..., 1, 2] = q[..., 2], -q[..., 0]
    S[..., 2, 0], S[..., 2, 1] = -q[..., 1], q[..., 0]
    return S


def lens_view_info(p, model, Rb, cb, sigma) -> tuple[np.ndarray, np.ndarray]:
    """Information on the eight lens parameters from boards at poses (Rb (N, 3, 3) board -> optical, centres cb
    (N, 3) optical) with corners model (n, 3) in the board frame, every corner's u and v at σ px, the board pose
    marginalised per view. Returns (I (N, 8, 8), uv (N, n, 2) OpenCV pixels)."""
    Rb = np.asarray(Rb, float).reshape(-1, 3, 3)
    cb = np.asarray(cb, float).reshape(-1, 3)
    P = np.einsum('kj,nij->nki', model, Rb) + cb[:, None, :]                  # (N, n, 3)
    uv, _ = kb_project(p, P)
    Jp = _kb_param_jac(p, P)                                                   # (N, n, 2, 8)
    JP = _kb_point_jac(p, P)                                                   # (N, n, 2, 3)
    Jw = JP @ (-_skew(P - cb[:, None, :]))                                     # d / d rotation about the centre
    N, n = P.shape[:2]
    A = Jp.reshape(N, 2 * n, 8)
    B = np.concatenate([Jw, JP], -1).reshape(N, 2 * n, 6)
    AA = np.einsum('nri,nrj->nij', A, A)
    AB = np.einsum('nri,nrj->nij', A, B)
    BB = np.einsum('nri,nrj->nij', B, B)
    I = (AA - AB @ np.linalg.solve(BB, np.swapaxes(AB, 1, 2))) / sigma ** 2
    return (I + np.swapaxes(I, 1, 2)) / 2, uv


def _tangent_basis(d) -> tuple[np.ndarray, np.ndarray]:
    a = np.where(np.abs(d[..., 1:2]) < 0.9, np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0]))
    e1 = np.cross(d, a)
    e1 /= np.linalg.norm(e1, axis=-1, keepdims=True)
    return e1, np.cross(d, e1)


def ray_jacobian(p, u, v) -> np.ndarray:
    """d(ray angle, rad, along two orthogonal tangent directions) / d p at OpenCV pixels (u, v): (G, 2, 8)."""
    d0 = kb_unproject(p, u, v)
    e1, e2 = _tangent_basis(d0)
    steps = np.array([1e-3, 1e-3, 1e-3, 1e-3, 1e-6, 1e-6, 1e-6, 1e-6]) * np.r_[np.abs(p[:2]), 1, 1, 1, 1, 1, 1]
    D = np.zeros((len(d0), 2, 8))
    for i, h in enumerate(steps):
        a = p.copy()
        b = p.copy()
        a[i] += h
        b[i] -= h
        dd = (kb_unproject(a, u, v) - kb_unproject(b, u, v)) / (2 * h)
        D[:, 0, i] = np.sum(dd * e1, -1)
        D[:, 1, i] = np.sum(dd * e2, -1)
    return D


def lens_locks(Sigma, D) -> np.ndarray:
    """Lock values (degrees) of covariances Sigma (..., 8, 8) over ray Jacobians D (G, 2, 8), in LENS_LOCKS order."""
    Sigma = np.asarray(Sigma, float)
    eff = np.linalg.norm(D, axis=1).max(0)                                     # (8,) rad per unit, worst pixel
    out = [np.sqrt(np.maximum(Sigma[..., i, i], 0)) * eff[i] for i in range(4)]

    def worst(Dx, S):
        a = np.einsum('gi,...ij,gj->...g', Dx[:, 0], S, Dx[:, 0])
        b = np.einsum('gi,...ij,gj->...g', Dx[:, 0], S, Dx[:, 1])
        c = np.einsum('gi,...ij,gj->...g', Dx[:, 1], S, Dx[:, 1])
        lam = (a + c) / 2 + np.sqrt(np.maximum(((a - c) / 2) ** 2 + b * b, 0.0))
        return np.sqrt(np.maximum(lam, 0)).max(-1)
    out.append(worst(D[:, :, 4:], Sigma[..., 4:, 4:]))
    out.append(worst(D, Sigma))
    return np.degrees(np.stack(out, -1))


# =========================================================================== board poses
def held_pose(centre, cam_pos, top_deg=0.0, side_deg=0.0) -> LC.BoardPose:
    """A board held at centre (vehicle frame), facing the lens at cam_pos squarely (its normal at the lens, its height
    axis as close to up as that allows), then its top tilted top_deg towards the lens and its right edge (as the
    technician holds it: the camera's left) side_deg towards the lens. Columns of R: width (the technician's left,
    the camera's right), height, normal towards the lens."""
    c = np.asarray(centre, float)
    n = np.asarray(cam_pos, float) - c
    dist = float(np.linalg.norm(n))
    n = n / max(dist, 1e-9)
    up = np.array([0.0, 0.0, 1.0])
    v = up - (up @ n) * n
    if np.linalg.norm(v) < 1e-6:
        v = np.array([1.0, 0.0, 0.0]) - n[0] * n
    v /= np.linalg.norm(v)
    u = np.cross(v, n)
    R = np.stack([u, v, n], 1) @ rot_y(math.radians(side_deg)) @ rot_x(math.radians(top_deg))
    return LC.BoardPose(c, R, (float(side_deg), float(top_deg), 0.0), dist)


def tilt_of(pose: LC.BoardPose, cam_pos) -> tuple[float, float]:
    """(top towards the lens, right edge towards the lens) degrees of a board pose: held_pose's angles."""
    base = held_pose(pose.centre, cam_pos).R
    Rl = base.T @ pose.R                         # = rot_y(side) @ rot_x(top)
    n = Rl[:, 2]                                 # (cos t sin s, −sin t, cos t cos s)
    top = math.degrees(math.asin(max(-1.0, min(1.0, -n[1]))))
    side = math.degrees(math.atan2(n[0], n[2]))
    return top, side


def _normal_angle(a, b) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, float(np.asarray(a) @ np.asarray(b))))))


def _feature_px_grid(uv, rows, cols) -> np.ndarray:
    """Per board (N), the shortest square side in the image among its corners (..., rows*cols, 2)."""
    G = uv.reshape(uv.shape[:-2] + (rows, cols, 2))
    e1 = np.linalg.norm(np.diff(G, axis=-2), axis=-1).min((-1, -2))
    e2 = np.linalg.norm(np.diff(G, axis=-3), axis=-1).min((-1, -2))
    return np.minimum(e1, e2)


# =========================================================================== views and advice
@dataclass
class View:
    """An accepted view."""
    name: str
    pose: LC.BoardPose                 # vehicle frame
    uv: np.ndarray | None              # (n, 2) corners, OpenCV pixels at the sensor's resolution
    info: object                       # lens: (8, 8); lidar: (a (6, 6), m (6, 6))
    frame: np.ndarray | None = None
    t: float = 0.0
    sharpness: float = float('nan')
    detected: bool = True              # from a detection (False: added from its pose, run())


@dataclass
class Detection:
    """What one frame showed."""
    found: bool
    uv: np.ndarray | None = None       # (n, 2) OpenCV pixels at the sensor's resolution
    pose: LC.BoardPose | None = None
    sharpness: float = float('nan')
    accepted: bool = False
    reason: str = ''                   # why not accepted
    rms_px: float = float('nan')
    seconds: float = 0.0


@dataclass
class Advice:
    """The next-best view."""
    pose: LC.BoardPose
    before: dict                       # lock -> ratio now
    after: dict                        # lock -> ratio with it
    worst_before: float
    worst_after: float
    target_lock: str                   # the lock the advice works on (the loosest now)
    text: str                          # plain guidance (the technician's frame)
    text_vehicle: str
    move: dict                         # numbers behind the text
    ghost: dict                        # outline / grid / corners in sensor pixels
    candidates: int = 0
    seconds: float = 0.0


class Coach:
    """The coach of one camera (lens mode) or one camera–lidar pair (lidar mode). Not thread-safe: one worker at a
    time (the page runs it on one background thread)."""

    def __init__(self, project, engine=None, settings: CoachSettings | None = None, variant=None):
        self.st = st = settings or settings_of(project)
        self.project = p = OB._project_for(project, variant)
        self.v = p.variant
        self.engine = engine
        self.ost = OB.settings_of(p)
        self.sigma_px = st.pixel_sigma if st.pixel_sigma > 0 else self.ost.pixel_sigma
        self.min_sq = st.min_square_px if st.min_square_px > 0 else self.ost.min_square_px
        cams = [c for c in self.v.cameras if c.visible and OB._finite_pose(c)]
        self.lidar = None
        self.share = 0.0
        if st.mode == 'lidar':
            self.lst = LC.LidarCalibSettings.from_dict(getattr(p.analysis, 'lidar_calib', None))
            self.lst.camera = st.camera or self.lst.camera
            self.lst.lidar = st.lidar or self.lst.lidar
            cam, lid, share = LC.pick_pair(p, self.v, self.lst, engine)       # raises ValueError without a pair
            self.cam, self.lidar, self.share = cam, lid, float(share)
            self.board = self.lst
            self.near, self.far = self.lst.near_m, self.lst.far_m
        else:
            self.cam = next((c for c in cams if c.tag == st.camera), cams[0] if cams else None)
            if self.cam is None:
                raise ValueError('No visible camera in the variant.')
            self.lst = None
            self.board = LC.LidarCalibSettings(cols=st.cols, rows=st.rows, square_mm=st.square_mm, margin_mm=0.0)
            self.near, self.far = st.near_m, st.far_m
        self.need = requirement(p, st, self.lst)
        cam = self.cam
        self.intr = resolve(p.module_for(cam), p.lens_for(cam))
        self.W2O = LC._w2o(cam)
        self.C = cam.pos().astype(float)
        self.model = LC.model_corners(self.board)
        self.vg = engine.vehicle_geom(p, self.v) if engine is not None else None
        self.cube = engine.cube(self.C, self.vg) if self.vg is not None else None
        self.views: list[View] = []
        self.last: Detection | None = None
        self._cands = None
        self._cand_info = None
        if st.mode == 'lens':
            self.p = lens_params(self.intr)
            self.mask, self.lock_uv = self._reachable()
            self.D = ray_jacobian(self.p, self.lock_uv[:, 0], self.lock_uv[:, 1])
            eff = np.maximum(np.linalg.norm(self.D, axis=1).max(0), 1e-12)
            self.prior = np.diag((eff / PRIOR_RAD) ** 2)
        else:
            self.vgl = self.vg
            self.T0 = LC.plan_T(cam, self.lidar)
            self.prior = np.eye(6) / PRIOR_LIDAR ** 2

    # ------------------------------------------------------------------ geometry
    @property
    def tag(self) -> str:
        return self.cam.tag

    def keys(self) -> tuple:
        return LENS_LOCKS if self.st.mode == 'lens' else LIDAR_LOCKS

    def need_of(self, key) -> float:
        if self.st.mode == 'lens' or key.endswith('_deg'):
            return self.need['deg']
        return self.need['mm']

    def to_optical(self, P) -> np.ndarray:
        return (np.asarray(P, float) - self.C) @ self.W2O.T

    def pose_optical(self, pose: LC.BoardPose) -> tuple[np.ndarray, np.ndarray]:
        """(R board -> optical, centre in the optical frame)."""
        return self.W2O @ pose.R, self.W2O @ (pose.centre - self.C)

    def pose_from_optical(self, R, c) -> LC.BoardPose:
        centre = self.C + self.W2O.T @ np.asarray(c, float)
        Rv = self.W2O.T @ np.asarray(R, float)
        top, side = tilt_of(LC.BoardPose(centre, Rv), self.C)
        return LC.BoardPose(centre, Rv, (round(side, 2), round(top, 2), 0.0), float(np.linalg.norm(centre - self.C)))

    def _reachable(self):
        """(mask (gv, gu) of the lock grid, OpenCV pixels (G, 2) of its reachable cells): pixels whose ray meets a
        place a hand-held board's corners can be."""
        intr = self.intr
        gu, gv = LOCK_GRID
        us = (np.arange(gu) + 0.5) / gu * intr.width
        vs = (np.arange(gv) + 0.5) / gv * intr.height
        U, V = np.meshgrid(us, vs)
        d, th = intr.unproject(U.ravel(), V.ravel())
        ok = th <= intr.max_theta - math.radians(CIRCLE_MARGIN_DEG)
        dw = d @ self.W2O
        W, H = self.board.board_size()
        half = 0.5 * math.hypot(W, H)
        ts = np.linspace(self.near, self.far, 24)
        Z = self.C[2] + dw[:, 2:3] * ts[None, :]
        hit = ((Z >= max(self.st.low_m - half, GROUND_CLEAR_M)) & (Z <= self.st.high_m + half)).any(1)
        if self.cube is not None:
            depth = self.cube.lookup(dw)[0]
            hit &= depth > self.near
        mask = (ok & hit).reshape(gv, gu)
        if not mask.any():
            mask = ok.reshape(gv, gu)
        sel = mask.ravel()
        return mask, np.c_[U.ravel()[sel] - 0.5, V.ravel()[sel] - 0.5]

    def coverage(self) -> float:
        """Share of the image (inside the lens circle) a hand-held board can reach."""
        if self.st.mode != 'lens':
            return float('nan')
        intr = self.intr
        gu, gv = LOCK_GRID
        U, V = np.meshgrid((np.arange(gu) + 0.5) / gu * intr.width, (np.arange(gv) + 0.5) / gv * intr.height)
        _, th = intr.unproject(U.ravel(), V.ravel())
        inside = th <= intr.max_theta
        return float(self.mask.ravel()[inside].mean()) if inside.any() else 0.0

    def feasible(self, poses) -> np.ndarray:
        """Which board poses (list of BoardPose) a hand-held board can take and the camera sees whole (lidar mode: the
        lidar's part is checked when the information is computed)."""
        if not poses:
            return np.zeros(0, bool)
        st = self.st
        intr = self.intr
        W, H = self.board.board_size()
        Rs = np.stack([b.R for b in poses])
        cs = np.stack([b.centre for b in poses])
        P = np.einsum('kj,nij->nki', self.model, Rs) + cs[:, None, :]
        O = np.einsum('kj,nij->nki', LC.outline_local(W, H, 3), Rs) + cs[:, None, :]
        A = np.concatenate([P, O], 1)
        u, v, th, ok = intr.project_dirs(self.to_optical(A))
        m = EDGE_MARGIN * intr.width
        ok = ok & (th <= intr.max_theta - math.radians(CIRCLE_MARGIN_DEG)) & (u >= m) & (u <= intr.width - m) & \
            (v >= m) & (v <= intr.height - m)
        good = ok.all(1)
        n = len(self.model)
        uv = np.stack([u[:, :n], v[:, :n]], -1)
        sq = _feature_px_grid(uv, self.board.rows, self.board.cols)
        good &= sq >= self.min_sq
        good &= O[:, :, 2].min(1) >= GROUND_CLEAR_M
        h = cs[:, 2]
        good &= (h >= st.low_m - 1e-9) & (h <= st.high_m + 1e-9)
        dist = np.linalg.norm(cs - self.C, axis=1)
        good &= (dist >= self.near - 1e-9) & (dist <= self.far + 1e-9)
        to_cam = (self.C - cs) / np.maximum(dist[:, None], 1e-9)
        good &= np.degrees(np.arccos(np.clip(np.sum(Rs[:, :, 2] * to_cam, 1), -1, 1))) <= st.max_tilt_deg + 1e-6
        if self.cube is not None and good.any():
            idx = np.flatnonzero(good)
            d = (P[idx] - self.C).reshape(-1, 3)
            dd = np.linalg.norm(d, axis=1)
            depth = self.cube.lookup(d)[0]
            clear = (depth > dd - (0.02 + 0.01 * dd)).reshape(len(idx), -1).all(1)
            good[idx[~clear]] = False
        return good

    def candidates(self) -> list:
        """The feasible hand-held poses the next-best view is chosen from (computed once)."""
        if self._cands is not None:
            return self._cands
        intr = self.intr
        us = (np.arange(GRID_U) + 0.5) / GRID_U * intr.width
        vs = (np.arange(GRID_V) + 0.5) / GRID_V * intr.height
        U, V = np.meshgrid(us, vs)
        d, th = intr.unproject(U.ravel(), V.ravel())
        dists = np.geomspace(self.near, self.far, N_DIST)
        out = []
        for k in range(len(d)):
            if th[k] > intr.max_theta - math.radians(CIRCLE_MARGIN_DEG):
                continue
            dw = self.W2O.T @ d[k]
            for dist in dists:
                c = self.C + dw * dist
                if not (self.st.low_m <= c[2] <= self.st.high_m):
                    continue
                for top in TILTS:
                    for side in TILTS:
                        b = held_pose(c, self.C, top, side)
                        if _normal_angle(b.R[:, 2], (self.C - c) / np.linalg.norm(self.C - c)) > \
                                self.st.max_tilt_deg + 1e-6:
                            continue
                        out.append(b)
        good = self.feasible(out)
        self._cands = [b for b, g in zip(out, good) if g]
        return self._cands

    # ------------------------------------------------------------------ information
    def info_of(self, poses) -> list:
        """Each pose's information (lens: (8, 8); lidar: (a, m) or None when the lidar does not see it)."""
        if not poses:
            return []
        if self.st.mode == 'lens':
            R, c = zip(*(self.pose_optical(b) for b in poses))
            I, _ = lens_view_info(self.p, self.model, np.stack(R), np.stack(c), self.sigma_px)
            return list(I)
        return [self._lidar_info(b) for b in poses]

    def _lidar_info(self, pose):
        """(a, m): pose's marginal information on T_camera_lidar under the calibration's weights and the matching
        middle of the sandwich (observability._lidar_model with one pose), or None when too few lidar channels or
        points meet the board."""
        H, M, data = OB._lidar_model(self.project, self.engine, self.vgl, self.cam, self.lidar, self.lst, [pose],
                                     self.sigma_px)
        d = data['poses'][0]
        if d['rings'] < LC.MIN_RINGS or d['points'] < LC.MIN_BOARD_POINTS:
            return None
        A, B, Dm = H[:6, :6], H[:6, 6:], H[6:, 6:]
        Mt, Mb, Md = M[:6, :6], M[:6, 6:], M[6:, 6:]
        G = B @ np.linalg.inv(Dm)                    # B D⁻¹
        a = A - G @ B.T
        m = Mt - Mb @ G.T - G @ Mb.T + G @ Md @ G.T
        return (a + a.T) / 2, (m + m.T) / 2

    def information(self, extra=None):
        """The information of the accepted views (plus extra, one more view's): lens (8, 8); lidar (a, m)."""
        infos = [v.info for v in self.views] + ([extra] if extra is not None else [])
        if self.st.mode == 'lens':
            return sum(infos, np.zeros((8, 8)))
        a = sum((x[0] for x in infos), np.zeros((6, 6)))
        m = sum((x[1] for x in infos), np.zeros((6, 6)))
        return a, m

    def covariance(self, info=None) -> np.ndarray:
        info = self.information() if info is None else info
        if self.st.mode == 'lens':
            return np.linalg.inv(info + self.prior)
        a, m = info
        Si = np.linalg.inv(a + self.prior)
        C = Si @ (m + self.prior) @ Si
        return (C + C.T) / 2

    def sigmas(self, cov=None) -> dict:
        """lock -> 1σ (degrees; lidar positions mm)."""
        cov = self.covariance() if cov is None else cov
        if self.st.mode == 'lens':
            vals = lens_locks(cov, self.D)
            return dict(zip(LENS_LOCKS, (float(x) for x in vals)))
        sg = LC._sigmas(self.cam, self.T0, cov)['vehicle']
        return {k: float(sg[k]) for k in LIDAR_LOCKS}

    def status(self, info=None) -> dict:
        """{'locks': {key: {'sigma', 'need', 'ratio', 'locked', 'unit', 'label'}}, 'worst', 'worst_key', 'locked',
        'views', 'params' (lens: 1σ of fx px … k4)}."""
        cov = self.covariance(info)
        sg = self.sigmas(cov)
        locks = {}
        for k in self.keys():
            need = self.need_of(k)
            r = sg[k] / need
            locks[k] = {'sigma': sg[k] if r < UNDETERMINED else math.inf, 'need': need, 'ratio': r,
                        'locked': bool(r <= 1.0), 'unit': 'mm' if k.endswith('_mm') else 'deg',
                        'label': LOCK_LABELS[k]}
        wk = max(locks, key=lambda k: locks[k]['ratio'])
        out = {'locks': locks, 'worst': locks[wk]['ratio'], 'worst_key': wk,
               'locked': all(x['locked'] for x in locks.values()), 'views': len(self.views)}
        if self.st.mode == 'lens':
            sd = np.sqrt(np.maximum(np.diag(cov), 0))
            out['params'] = {k: float(x) for k, x in zip(LENS_PARAMS, sd)}
        return out

    # ------------------------------------------------------------------ views
    def duplicate(self, pose: LC.BoardPose) -> str:
        """The name of an accepted view this pose repeats, or ''."""
        for v in self.views:
            if np.linalg.norm(v.pose.centre - pose.centre) < self.st.dup_m and \
                    _normal_angle(v.pose.R[:, 2], pose.R[:, 2]) < self.st.dup_deg:
                return v.name
        return ''

    def add_pose(self, pose: LC.BoardPose, frame=None, uv=None, name=None, t=0.0, sharpness=float('nan'),
                 detected=False) -> View | None:
        """Accept a view at pose (its information from the pose); None when the lidar does not see it."""
        info = self.info_of([pose])[0]
        if info is None:
            return None
        if uv is None:
            Rb, cb = self.pose_optical(pose)
            uv, _ = kb_project(self.p if self.st.mode == 'lens' else lens_params(self.intr), self.model @ Rb.T + cb)
        v = View(name or f'view_{len(self.views) + 1:02d}', pose, np.asarray(uv, float), info, frame, t, sharpness,
                 detected)
        self.views.append(v)
        return v

    def undo(self) -> View | None:
        return self.views.pop() if self.views else None

    def reset(self):
        self.views = []
        self.last = None

    # ------------------------------------------------------------------ detection
    def detect(self, frame, hint: LC.BoardPose | None = None) -> Detection:
        """The board in a camera frame (RGB or grey, any size with the sensor's aspect): its corners and pose. hint:
        unused (kept for callers that know where the board should be)."""
        import cv2
        t0 = time.perf_counter()
        img = frame
        h, w = img.shape[:2]
        intr = self.intr
        from . import calib
        # the lens calibration's detector (whole image, then rectified pinhole views through the lens model) for both
        # boards: its time is bounded when no board is in view (the lidar–camera calibration's own search is exhaustive
        # and can take minutes on an empty noisy frame); a lidar board is looked for at a finer scale (smaller squares)
        scale = min(1.0, (1280.0 if self.st.mode == 'lens' else 1600.0) / w)
        small = img if scale == 1.0 else cv2.resize(img, (int(round(w * scale)), int(round(h * scale))),
                                                    interpolation=cv2.INTER_AREA)
        corners = calib.find_board(small, (self.board.cols, self.board.rows), intr)
        if corners is None:
            return Detection(False, reason='no board found', seconds=time.perf_counter() - t0)
        sh, sw = small.shape[:2]
        sx, sy = sw / intr.width, sh / intr.height
        uvs = corners + 0.5                                      # the lens model's pixel convention, image scale
        rays, _ = intr.unproject(uvs[:, 0] / sx, uvs[:, 1] / sy)
        sol = LC._board_pnp(intr, sx, sy, self.model, rays, uvs)
        if sol is None:
            return Detection(False, reason='board pose not found', seconds=time.perf_counter() - t0)
        R, c, rms, _ = sol
        if R[:, 2] @ c > 0:                                      # normal towards the camera
            R = R @ np.diag([1.0, -1.0, -1.0])
        uv = uvs / np.array([sx, sy]) - 0.5
        pose = self.pose_from_optical(R, c)
        sharp = self.sharpness(img, uv)
        return Detection(True, uv, pose, sharp, rms_px=float(rms), seconds=time.perf_counter() - t0)

    def sharpness(self, img, uv) -> float:
        """rig_health's sharpness (variance of the Laplacian) of the board's region in a frame, the region scaled so
        that a square is SHARP_SQUARE_PX wide: a blur counts in proportion to the squares, whatever the board's size
        in the image. uv: the corners, OpenCV pixels at the sensor's resolution."""
        import cv2
        from .rig_health import image_stats
        h, w = img.shape[:2]
        sx, sy = w / self.intr.width, h / self.intr.height
        x0, x1 = int(max(uv[:, 0].min() * sx - 8, 0)), int(min(uv[:, 0].max() * sx + 9, w))
        y0, y1 = int(max(uv[:, 1].min() * sy - 8, 0)), int(min(uv[:, 1].max() * sy + 9, h))
        if x1 - x0 <= 8 or y1 - y0 <= 8:
            return float(image_stats(img)['sharpness'])
        crop = img[y0:y1, x0:x1]
        sq = float(np.median(_feature_px_grid(uv[None] * np.array([sx, sy]), self.board.rows, self.board.cols)))
        f = min(SHARP_SQUARE_PX / max(sq, 1e-3), 8.0)
        crop = cv2.resize(crop, (max(8, int(round(crop.shape[1] * f))), max(8, int(round(crop.shape[0] * f)))),
                          interpolation=cv2.INTER_AREA if f < 1.0 else cv2.INTER_LINEAR)
        return float(image_stats(crop)['sharpness'])

    def offer(self, frame, t=0.0, hint=None, accept=True) -> Detection:
        """Detect the board in a frame and accept the view when it is sharp and new (accept=False: judge only)."""
        det = self.detect(frame, hint)
        self.last = det if det.found else self.last
        if not det.found:
            return det
        if not (det.sharpness >= self.st.sharp_min):
            det.reason = f'blurred (sharpness {det.sharpness:.0f}, needs {self.st.sharp_min:g}): hold the board still'
            return det
        dup = self.duplicate(det.pose)
        if dup:
            det.reason = f'the same view as {dup}: move the board'
            return det
        if not self.feasible([det.pose])[0] and self.st.mode == 'lidar':
            pass                                          # a pose the plan would not choose still counts if seen
        if accept:
            v = self.add_pose(det.pose, frame, det.uv, t=t, sharpness=det.sharpness, detected=True)
            if v is None:
                det.reason = 'too few lidar channels cross the board here'
                return det
            det.accepted = True
        return det

    # ------------------------------------------------------------------ next-best view
    def _cand_infos(self):
        if self._cand_info is None:
            cands = self.candidates()
            if self.st.mode == 'lens':
                self._cand_info = (cands, self.info_of(cands))
            else:
                if len(cands) > MAX_LIDAR_CANDIDATES:
                    step = len(cands) / MAX_LIDAR_CANDIDATES
                    cands = [cands[int(i * step)] for i in range(MAX_LIDAR_CANDIDATES)]
                infos = [self._lidar_info(b) for b in cands]
                keep = [i for i, x in enumerate(infos) if x is not None]
                self._cand_info = ([cands[i] for i in keep], [infos[i] for i in keep])
        return self._cand_info

    def scores(self, cands, infos) -> tuple[np.ndarray, np.ndarray]:
        """(score, worst ratio after) per candidate: lower score is better."""
        need = np.array([self.need_of(k) for k in self.keys()])
        if self.st.mode == 'lens':
            I0 = self.information() + self.prior
            In = I0[None] + np.stack(infos)
            S = np.linalg.inv(In)
            worst = (lens_locks(S, self.D) / need).max(-1)
            if self.st.criterion == 'd_optimal':
                return -np.linalg.slogdet(In)[1], worst
            return worst, worst
        a0, m0 = self.information()
        worst, ld = [], []
        for a, m in infos:
            cov = self.covariance((a0 + a, m0 + m))
            sg = self.sigmas(cov)
            worst.append(max(sg[k] / self.need_of(k) for k in self.keys()))
            ld.append(np.linalg.slogdet(a0 + a + self.prior)[1])
        worst = np.array(worst)
        return (-np.array(ld) if self.st.criterion == 'd_optimal' else worst), worst

    def next_best(self, current: LC.BoardPose | None = None) -> Advice | None:
        """The candidate pose that firms up the calibration most (st.criterion), with its guidance; None when no
        candidate is left."""
        t0 = time.perf_counter()
        cands, infos = self._cand_infos()
        keep = [i for i, b in enumerate(cands) if not self.duplicate(b)]
        if not keep:
            return None
        cs = [cands[i] for i in keep]
        score, worst = self.scores(cs, [infos[i] for i in keep])
        k = int(np.argmin(score))
        before = self.status()
        after = self.status(self.information(infos[keep[k]]))
        pose = cs[k]
        cur = current if current is not None else (self.last.pose if self.last is not None and self.last.found
                                                    else None)
        text, text_v, move = self.guidance(pose, cur)
        return Advice(pose, {x: before['locks'][x]['ratio'] for x in self.keys()},
                      {x: after['locks'][x]['ratio'] for x in self.keys()}, before['worst'], after['worst'],
                      before['worst_key'], text, text_v, move, self.ghost(pose), len(cs), time.perf_counter() - t0)

    # ------------------------------------------------------------------ words and drawings
    def where_in_image(self, pose) -> str:
        u, v, _, _ = self.intr.project_dirs(self.to_optical(pose.centre[None]))
        fu, fv = float(u[0]) / self.intr.width, float(v[0]) / self.intr.height
        col = 'left' if fu < 0.36 else 'right' if fu > 0.64 else ''
        row = 'top' if fv < 0.36 else 'bottom' if fv > 0.64 else ''
        if not col and not row:
            return 'the middle of the image'
        return f'the {"-".join(x for x in (row, col) if x)} of the image'

    def guidance(self, target: LC.BoardPose, current: LC.BoardPose | None = None) -> tuple[str, str, dict]:
        """(text in the technician's frame, text in the vehicle frame, the numbers): how to get the board from current
        (None: from scratch) to target. The technician faces the camera: their left is the camera's right."""
        C = self.C
        f = target.centre - C
        fb = (current.centre if current is not None else target.centre) - C      # where the technician stands now
        fh = np.array([fb[0], fb[1], 0.0])
        fh = fh / np.linalg.norm(fh) if np.linalg.norm(fh) > 1e-6 else np.array([1.0, 0.0, 0.0])
        left = np.cross([0.0, 0.0, 1.0], -fh)                    # the technician's left (they face the lens)
        top, side = tilt_of(target, C)
        dist = float(np.linalg.norm(f))
        move = {'target': [round(float(x), 3) for x in target.centre], 'distance_m': round(dist, 3),
                'height_m': round(float(target.centre[2]), 3), 'top_deg': round(top, 1), 'side_deg': round(side, 1),
                'in_image': self.where_in_image(target)}
        parts = []
        if current is not None:
            d = target.centre - current.centre
            lat = float(d @ left)
            rng = float(np.linalg.norm((target.centre - C)[:2]) - np.linalg.norm((current.centre - C)[:2]))
            up = float(d[2])
            move.update(left_m=round(lat, 3), farther_m=round(rng, 3), up_m=round(up, 3),
                        delta_vehicle=[round(float(x), 3) for x in d])
            if abs(lat) >= MOVE_MIN_M:
                parts.append(f'move {abs(lat):.1f} m to your {"left" if lat > 0 else "right"}')
            if abs(rng) >= MOVE_MIN_M:
                parts.append(f'step {abs(rng):.1f} m {"back from" if rng > 0 else "towards"} the camera')
            if abs(up) >= MOVE_MIN_M:
                parts.append(f'{"raise" if up > 0 else "lower"} the board {abs(up):.1f} m')
        else:
            parts.append(f'hold the board {dist:.1f} m from the camera, its centre {target.centre[2]:.1f} m above '
                         f'the ground, in {move["in_image"]}')
        tilt = []
        if abs(top) >= TURN_MIN_DEG:
            tilt.append(f'tilt the top of the board {abs(top):.0f}° {"towards" if top > 0 else "away from"} the camera')
        if abs(side) >= TURN_MIN_DEG:
            tilt.append(f'turn its right edge {abs(side):.0f}° {"towards" if side > 0 else "away from"} the camera')
        if current is not None:
            ct, cs_ = tilt_of(current, C)
            if abs(ct - top) < TURN_MIN_DEG and abs(cs_ - side) < TURN_MIN_DEG:
                tilt = []
            elif not tilt:
                tilt.append('face the board squarely at the camera')
        if not tilt and current is None:
            tilt.append('facing the camera squarely')
        if current is not None and not parts and not tilt:
            text = 'Hold it there.'
        else:
            body = parts + ([('then ' if parts else '') + ', '.join(tilt)] if tilt else [])
            text = ', '.join(body)
            text = text[0].upper() + text[1:] + '.'
        tv = (f'Board centre to x {target.centre[0]:.2f} m, y {target.centre[1]:.2f} m, z {target.centre[2]:.2f} m '
              f'(vehicle frame), its face {_normal_angle(target.R[:, 2], (C - target.centre) / dist):.0f}° from facing '
              f'{self.tag}')
        if current is not None:
            d = target.centre - current.centre
            tv += f'; move Δx {d[0]:+.2f} m, Δy {d[1]:+.2f} m, Δz {d[2]:+.2f} m'
        return text, tv + '.', move

    def ghost(self, pose: LC.BoardPose, per_edge=16) -> dict:
        """The board at pose drawn through the camera's exact lens model, sensor pixels (centre at i + 0.5):
        {'outline': (M, 2) closed, 'grid': [polylines], 'corners': (n, 2), 'ok': all inside}."""
        W, H = self.board.board_size()
        sq = self.board.square_mm / 1000.0
        intr = self.intr

        def proj(L):
            u, v, _, ok = intr.project_dirs(self.to_optical(pose.centre + np.asarray(L) @ pose.R.T))
            return np.c_[u, v], ok
        outline, ok1 = proj(LC.outline_local(W, H, per_edge))
        corners, ok2 = proj(self.model)
        lines = []
        nc, nr = self.board.cols + 1, self.board.rows + 1
        xs = (np.arange(nc + 1) - nc / 2) * sq
        ys = (np.arange(nr + 1) - nr / 2) * sq
        s = np.linspace(0, 1, per_edge + 1)
        for x in xs:
            lines.append(proj(np.c_[np.full_like(s, x), ys[0] + (ys[-1] - ys[0]) * s, np.zeros_like(s)])[0])
        for y in ys:
            lines.append(proj(np.c_[xs[0] + (xs[-1] - xs[0]) * s, np.full_like(s, y), np.zeros_like(s)])[0])
        return {'outline': outline, 'grid': lines, 'corners': corners, 'ok': bool(ok1.all() and ok2.all())}

    # ------------------------------------------------------------------ outputs
    def lens_views(self) -> list:
        """[(name, frame)] of the accepted views that carry a frame: the lens calibration's input set."""
        return [(v.name, v.frame) for v in self.views if v.frame is not None]

    def session_doc(self) -> dict:
        """The live session as JSON: settings, requirement, accepted views (poses in the vehicle and optical frames)
        and the current σ."""
        stt = self.status()
        views = []
        for v in self.views:
            Rb, cb = self.pose_optical(v.pose)
            top, side = tilt_of(v.pose, self.C)
            views.append({'name': v.name, 'time_s': round(float(v.t), 3), 'centre_m': _r(v.pose.centre, 4),
                          'normal': _r(v.pose.R[:, 2], 5), 'centre_optical_m': _r(cb, 4),
                          'normal_optical': _r(Rb[:, 2], 5), 'top_deg': round(top, 2), 'side_deg': round(side, 2),
                          'distance_m': round(float(np.linalg.norm(v.pose.centre - self.C)), 3),
                          'sharpness': _r(v.sharpness, 1), 'detected': v.detected})
        return {'schema': SCHEMA, 'kind': 'session', 'mode': self.st.mode, 'camera': self.tag,
                'lidar': self.lidar.tag if self.lidar is not None else None, 'frame': FRAME, 'optical': OPTICAL,
                'pixels': PIXELS, 'units': {'angles': 'degrees', 'lengths': 'metres, lidar 1σ in mm'},
                'board': {'cols': self.board.cols, 'rows': self.board.rows, 'square_mm': self.board.square_mm,
                          'margin_mm': self.board.margin_mm},
                'requirement': self.need, 'pixel_sigma': self.sigma_px, 'settings': self.st.to_dict(),
                'status': _clean(stt), 'views': views, 'assumptions': ASSUMPTIONS}


def _r(v, nd=3):
    if isinstance(v, (list, tuple, np.ndarray)):
        return [_r(x, nd) for x in np.asarray(v, float).tolist()]
    v = float(v)
    return round(v, nd) if math.isfinite(v) else None


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        v = float(o)
        return round(v, 6) if math.isfinite(v) else None
    return o


# =========================================================================== the simulated technician
class Technician:
    """A deterministic technician: holds the board where the coach asks (missing by jitter) or at random hand-held
    poses, and shows it to the camera as frames rendered through its lens model."""

    def __init__(self, coach: Coach, kind='coached', seed=1, jitter_m=None, jitter_deg=None):
        self.coach = coach
        self.kind = kind if kind in TECHNICIANS else 'coached'
        self.rng = np.random.default_rng(int(seed))
        self.jm = coach.st.jitter_m if jitter_m is None else float(jitter_m)
        self.jd = coach.st.jitter_deg if jitter_deg is None else float(jitter_deg)
        self.pose: LC.BoardPose | None = None
        self.advice: Advice | None = None

    def _random_pose(self) -> LC.BoardPose | None:
        c = self.coach
        intr = c.intr
        st = c.st
        for _ in range(400):
            u, v = self.rng.uniform(0, intr.width), self.rng.uniform(0, intr.height)
            d, th = intr.unproject(np.array([u]), np.array([v]))
            dist = self.rng.uniform(c.near, c.far)
            top, side = self.rng.uniform(-st.max_tilt_deg, st.max_tilt_deg, 2)
            if th[0] > intr.max_theta or math.hypot(top, side) > st.max_tilt_deg * 1.2:
                continue
            centre = c.C + c.W2O.T @ d[0] * dist
            b = held_pose(centre, c.C, top, side)
            if c.feasible([b])[0] and not c.duplicate(b):
                return b
        return None

    def _jittered(self, b: LC.BoardPose) -> LC.BoardPose:
        c = self.coach
        if self.jm <= 0 and self.jd <= 0:
            return b
        top, side = tilt_of(b, c.C)
        for _ in range(20):
            centre = b.centre + self.rng.normal(0, self.jm, 3)
            nb = held_pose(centre, c.C, top + self.rng.normal(0, self.jd), side + self.rng.normal(0, self.jd))
            if c.feasible([nb])[0]:
                return nb
        return b

    def next_pose(self) -> LC.BoardPose | None:
        """Where the board goes next (and, coached, the advice that sent it there)."""
        if self.kind == 'random':
            self.advice = None
            self.pose = self._random_pose()
            return self.pose
        self.advice = self.coach.next_best(self.pose)
        self.pose = None if self.advice is None else self._jittered(self.advice.pose)
        return self.pose

    def render(self, pose: LC.BoardPose, scale=0.5, seed=0, noise=2.0) -> np.ndarray:
        """RGB frame (uint8) of the board at pose through the camera's lens model, scale × the sensor's size."""
        c = self.coach
        intr = c.intr
        w, h = int(round(intr.width * scale)), int(round(intr.height * scale))
        Rb, cb = c.pose_optical(pose)
        if c.st.mode == 'lens':
            from .calib import render_board
            g = render_board(intr, Rb, cb, (c.board.cols, c.board.rows), c.board.square_mm, w, h, seed=seed,
                             noise=noise)
            return np.repeat(g[..., None], 3, -1)
        from .synth import pixel_rays
        d, th = pixel_rays(intr, w, h)
        img = np.full((h * w, 3), 40.0, np.float32)
        inside = th <= intr.max_theta
        az = np.arctan2(d[:, 1], d[:, 0])
        img[inside] = (120 + 18 * np.sin(az[inside] * 3) * np.cos(th[inside] * 5))[:, None]
        img = img.reshape(h, w, 3)
        LC._paint_board(img, intr, c.W2O, c.C, pose, c.board, lambda dw: np.full(len(dw), np.inf))
        if noise:
            img += np.random.default_rng(seed).normal(0, noise, img.shape).astype(np.float32)
        return np.clip(img, 0, 255).astype(np.uint8)


def simulate(coach: Coach, kind='coached', seed=1, max_views=None, render=False, scale=0.5, progress=None,
             cancel=None) -> dict | None:
    """A simulated technician's session: views until every lock holds or max_views. render: each pose drawn as a
    frame and detected (else counted from the pose). Returns {'kind', 'seed', 'views' (accepted), 'reached',
    'curve' [worst ratio after each view, starting with none], 'locks' [per view], 'poses', 'texts', 'rejected'}."""
    coach.reset()
    tech = Technician(coach, kind, seed)
    n_max = int(max_views or coach.st.max_views)
    s0 = coach.status()
    out = {'kind': tech.kind, 'seed': int(seed), 'views': 0, 'reached': s0['locked'], 'curve': [s0['worst']],
           'locks': [{k: x['sigma'] for k, x in s0['locks'].items()}], 'poses': [], 'texts': [], 'rejected': 0,
           'tries': 0}
    tries = 0
    while len(coach.views) < n_max and not coach.status()['locked'] and tries < 3 * n_max:
        if cancel is not None and cancel():
            return None
        tries += 1
        pose = tech.next_pose()
        if pose is None:
            break
        if render:
            frame = tech.render(pose, scale, seed=int(seed) * 1009 + tries)
            det = coach.offer(frame, t=float(tries))
            ok = det.accepted
        else:
            ok = not coach.duplicate(pose) and coach.add_pose(pose) is not None
        if not ok:
            out['rejected'] += 1
            continue
        if not render:
            coach.last = Detection(True, None, pose)
        stt = coach.status()
        out['curve'].append(stt['worst'])
        out['locks'].append({k: x['sigma'] for k, x in stt['locks'].items()})
        top, side = tilt_of(pose, coach.C)
        out['poses'].append({'centre_m': _r(pose.centre, 3), 'top_deg': round(top, 1), 'side_deg': round(side, 1),
                             'in_image': coach.where_in_image(pose)})
        out['texts'].append(tech.advice.text if tech.advice is not None else '')
        if progress is not None:
            progress(len(coach.views) / n_max, f'{tech.kind}: {len(coach.views)} views')
    out['views'] = len(coach.views)
    out['reached'] = bool(coach.status()['locked'])
    out['tries'] = tries
    return out


# =========================================================================== the study
@dataclass
class CoachResult:
    settings: CoachSettings
    variant: str
    variant_uid: str
    mode: str
    camera: str
    lidar: str | None
    requirement: dict
    coverage: float
    candidates: int
    coached: dict
    random: list
    plan: list                         # the coached session's first poses with their guidance
    findings: list
    metrics: dict
    camera_uid: str = ''
    lidar_uid: str = ''


def _views_to(run_: dict) -> float:
    return float(run_['views']) if run_['reached'] else math.inf


def run(project, engine=None, settings: CoachSettings | None = None, variant=None, progress=None,
        cancel=None) -> CoachResult | None:
    """How many hand-held views the camera's calibration needs to reach its requirement following the coach, against
    random hand-held views (st.random_runs technicians, seeds seed, seed + 1, …), counted from the poses (no rendering).
    Returns None when cancelled; raises ValueError without a camera (or, lidar mode, a camera–lidar pair)."""
    from .analysis import Engine
    t0 = time.perf_counter()
    st = settings or settings_of(project)
    eng = engine or Engine()
    coach = Coach(project, eng, st, variant)

    def sub(a, b):
        return None if progress is None else (lambda f, text='': progress(a + (b - a) * f, text))
    if progress is not None:
        progress(0.02, 'Candidate board poses')
    coach._cand_infos()
    coached = simulate(coach, 'coached', st.seed, progress=sub(0.1, 0.5), cancel=cancel)
    if coached is None:
        return None
    rand = []
    for i in range(st.random_runs):
        r = simulate(coach, 'random', st.seed + i, progress=sub(0.5 + 0.5 * i / st.random_runs,
                                                                 0.5 + 0.5 * (i + 1) / st.random_runs),
                     cancel=cancel)
        if r is None:
            return None
        rand.append(r)
    coach.reset()
    plan = [dict(pose, text=t) for pose, t in zip(coached['poses'][:12], coached['texts'][:12])]
    res = CoachResult(st, coach.v.name, coach.v.uid, st.mode, coach.tag,
                      coach.lidar.tag if coach.lidar is not None else None, dict(coach.need), coach.coverage(),
                      len(coach._cand_info[0]), coached, rand, plan, [], {}, coach.cam.uid,
                      coach.lidar.uid if coach.lidar is not None else '')
    rv = [_views_to(r) for r in rand]
    res.metrics = {'seconds': time.perf_counter() - t0, 'coached_views': _views_to(coached),
                   'random_views_median': float(np.median(rv)) if rv else math.inf,
                   'random_reached': sum(1 for r in rand if r['reached']), 'random_runs': len(rand),
                   'candidates': res.candidates}
    res.findings = _findings(res, coach)
    if progress is not None:
        progress(1.0, 'Done')
    return res


def _what(res) -> str:
    return (f'{res.camera}\'s lens calibration' if res.mode == 'lens' else
            f'the {res.camera}–{res.lidar} lidar–camera calibration')


def _need_text(res) -> str:
    q = res.requirement
    return f'{q["deg"]:g}°' if res.mode == 'lens' else f'{q["deg"]:g}° and {q["mm"]:g} mm'


def _findings(res: CoachResult, coach: Coach) -> list:
    out = []
    uids = [x for x in (res.camera_uid, res.lidar_uid) if x]

    def add(sev, kind, msg):
        out.append({'severity': sev, 'kind': kind, 'message': msg, 'uids': list(uids)})
    c, m = res.coached, res.metrics
    rv = [_views_to(r) for r in res.random]
    if not res.candidates:
        add('serious', 'candidates', f'No hand-held board pose fits {res.camera}\'s view whole with squares of '
                                     f'{coach.min_sq:g} px or more ({coach.near:g}–{coach.far:g} m from the lens, '
                                     f'{res.settings.low_m:g}–{res.settings.high_m:g} m high).')
        return out
    if c['reached']:
        med = m['random_views_median']
        rnd = (f'random hand-held poses need {med:.0f} (median of {len(rv)})' if math.isfinite(med) else
               f'random hand-held poses do not reach it within {res.settings.max_views} views in '
               f'{len(rv) - m["random_reached"]} of {len(rv)} runs')
        add('good', 'coach', f'Following the coach, {_what(res)} reaches its {_need_text(res)} requirement in '
                             f'{c["views"]} views; {rnd}.')
    else:
        last = c['locks'][-1]
        loose = [k for k in coach.keys() if not (last[k] <= coach.need_of(k))]
        add('warning', 'coach', f'Even following the coach, {_what(res)} does not reach its {_need_text(res)} '
                                f'requirement in {res.settings.max_views} views: '
                                + ', '.join(LOCK_LABELS[k] for k in loose) + ' stay loose (worst '
                                f'{c["curve"][-1]:.1f}× the requirement). Larger squares, a nearer board or a '
                                'looser requirement would help.')
    if res.mode == 'lens' and res.coverage < 0.6:
        add('note', 'coverage', f'A board held {res.settings.low_m:g}–{res.settings.high_m:g} m high reaches '
                                f'{100 * res.coverage:.0f} % of {res.camera}\'s image; the lens model is extrapolated '
                                'over the rest (the body, the ground beneath the camera).')
    return out


# =========================================================================== documents
def to_doc(res: CoachResult, project=None) -> dict:
    def run_doc(r):
        return {'kind': r['kind'], 'seed': r['seed'], 'views': r['views'], 'reached': r['reached'],
                'rejected_as_duplicates': r['rejected'], 'worst_ratio_after_each_view': _r(r['curve'], 4),
                'sigma_after_each_view': _clean(r['locks'])}
    return {'schema': SCHEMA, 'project': getattr(project, 'name', ''), 'variant': res.variant, 'mode': res.mode,
            'camera': res.camera, 'lidar': res.lidar, 'frame': FRAME, 'optical': OPTICAL, 'pixels': PIXELS,
            'units': {'lens_locks': 'degrees of ray angle, 1σ', 'lidar_locks': 'degrees and mm, 1σ, vehicle frame',
                      'positions': 'metres, vehicle frame'},
            'settings': res.settings.to_dict(), 'requirement': res.requirement,
            'reachable_image_share': _r(res.coverage, 4), 'candidates': res.candidates,
            'coached': run_doc(res.coached), 'random': [run_doc(r) for r in res.random],
            'plan': _clean(res.plan), 'findings': res.findings, 'metrics': _clean(res.metrics),
            'assumptions': ASSUMPTIONS,
            'limits': ('Predictions from the Fisher information at the current lens model, not solves; the simulated '
                       'technician\'s poses are counted from the poses themselves (no rendering, no detection). The '
                       'lens indicators are ray angles over the image a hand-held board can reach, held to the '
                       'camera-pose angle requirement (a heuristic budget). Lidar-camera: the lidar\'s scan of each '
                       'pose is modelled (core.observability), the board edges and reflectivity left out.')}


CSV_COLUMNS = ('strategy', 'seed', 'view', 'worst_ratio')


def to_csv(res: CoachResult) -> str:
    keys = list(res.coached['locks'][0]) if res.coached['locks'] else []
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(list(CSV_COLUMNS) + [f'sigma_{k}' for k in keys])
    for r in [res.coached] + list(res.random):
        for i, (ratio, lk) in enumerate(zip(r['curve'], r['locks'])):
            w.writerow([r['kind'], r['seed'], i, _r(ratio, 5)] + [_r(lk[k], 6) for k in keys])
    return buf.getvalue()


def files(res: CoachResult, project=None) -> list:
    return [('calib_coach.json', json.dumps(to_doc(res, project), indent=1, ensure_ascii=False).encode('utf-8')),
            ('calib_coach_views.csv', to_csv(res).encode('utf-8'))]


def dataclass_replace(st: CoachSettings, **kw) -> CoachSettings:
    d = st.to_dict()
    d.update(kw)
    return CoachSettings.from_dict(d)


def copy_settings(st: CoachSettings) -> CoachSettings:
    return copy.deepcopy(st)
