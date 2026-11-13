"""Install guide: a camera rig fitted to a car, taken live from its streams (GMSL on a Jetson, or any source
capture.LiveSource opens) to a measured, verified calibration and coverage, one stage at a time. No Qt here: the page
(ui.workspaces.guide_ws), the batch mode and the assistant's coach (assistant.install_coach) read GuideSession.

Stages, each closed by a test rather than a judgement:

1. connect   every stream delivers at nominal_fraction of its frame rate or more (core.rig_health).
2. identify  the installer covers one lens at a time; the stream that goes dark is that camera's (identify_covered):
             a wrongly wired port is found before anything is measured.
3. lens      every camera has a unit calibration (Calibrate ▸ Lens calibration or the calibration coach), or the
             installer accepts the library model for it.
4. aim       the camera's rotation on its mount, from the car's own body outline in the live image against the
             outline the plan predicts (estimate_aim; no target needed), with the turn and the shim that corrects it.
5. boards    the planned boards (floor boards and boards on stands) are found where the plan puts them.
6. solve     every pose from the boards: the floor cameras and the floor boards' placement in one least-squares
             (core.joint), the cameras that see no floor board chained to a solved neighbour through a board on a
             stand both see (solve_chain); 1σ of every axis × safety within the requirement, and a χ² test of the
             residuals against the corner noise.
7. coverage  the coverage the measured rig gives, every ground cell graded robust, uncertain or blind by the pose
             uncertainty (robust_coverage), against the plan.
8. verify    tests that use nothing the solve used: a check board (left out of the solve) seen from two cameras must
             land on the same floor points within their predicted spread; the seams agree within tolerance.

The mathematics, and what is proven
- Pose covariance. Corners with independent Gaussian noise σ (px) on each coordinate; the Jacobian J of the corner
  pixels with respect to the unknowns (each camera's rotation vector and position, each floor board's x, y, yaw);
  board placement as a Gaussian prior P. The least-squares estimate is the maximum-likelihood one, and its covariance
  to first order is Σ = (JᵀJ/σ² + P)⁻¹, the Cramér–Rao bound of the linearised problem (joint_covariance). A
  camera's block of Σ is its pose covariance with every board marginalised (block inversion = Schur complement).
- Chained cameras (a board on a stand seen by parent p and child c). Unknowns x = (board pose B, child pose C), the
  parent's pose P fixed at its solve with covariance Σ_P. With J_x the Jacobian of both cameras' corners in x and A
  that of the parent's corners in P: Σ_x = H⁻¹ + G Σ_P Gᵀ, H = J_xᵀJ_x/σ², G = −H⁻¹ J_xᵀ A/σ² (first-order error
  propagation of the parent's error through the chain; exact for the linearised problem).
- Robust coverage. A ground point seen by a camera at pixel u with margin m (px) to the nearest image edge (sensor
  edge or the lens's image circle), n that edge's normal: its pixel moves with the pose error δ by J δ, so along n with
  σ_n² = nᵀ J Σ Jᵀ n. With m ≥ k σ_n the probability that the point really lies outside the image past that edge is
  at most Φ(−k) (0.13 % for k = 3), to first order.
- Ground error from a pitch error δ at height h, ground distance d: Δd = δ (h² + d²)/h, since d = h/tan α.
- χ² tests: Σ r²/σ² over the residuals of a least-squares fit with n parameters and N residuals follows χ²(N − n)
  when the noise model holds; a value in the upper 1 % says the corners are noisier than σ (blur, a bent board, a
  wrong lens model) and the covariance is inflated by s² = χ²/(N − n). Floor points of a check board from two cameras
  differ by D with covariance Σ_D; D ᵀ Σ_D⁻¹ D follows χ²(2) per corner.
Assumptions (each can fail on a real car, and the tests above are how it shows): a flat floor, flat true-printed
boards, the lens models (unit calibrations, else the library), corner noise independent between corners, small errors
(linearisation). Aim estimates from the body outline are about ±0.5° (a mount shifted by millimetres turns close body
edges too): good for mounting, never a calibration.
"""
from __future__ import annotations

import copy
import math
import threading
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from .install import BoardObs, inner_corners, _find, _match_grid
from .joint import _rotvec_to_matrix, joint_solve, floor_points
from .lens import resolve
from .mathutil import BODY_TO_OPTICAL, euler_zyx, rot_zyx, wrap180
from .project import Camera, Target, Variant, new_uid

SCHEMA = 'vantage.install_guide/1'
STAGES = ('connect', 'identify', 'lens', 'aim', 'boards', 'solve', 'coverage', 'verify')
STAGE_TITLES = {'connect': 'Connect the cameras', 'identify': 'Identify the ports', 'lens': 'Lens calibrations',
                'aim': 'Mount and aim', 'boards': 'Lay the boards', 'solve': 'Measure the poses',
                'coverage': 'Coverage', 'verify': 'Verify'}
SIGMA_KEYS = ('yaw_deg', 'pitch_deg', 'roll_deg', 'x_mm', 'y_mm', 'z_mm')
FD = 1e-6                     # finite-difference step (rad, m)
UNDETERMINED = 1e6            # a 1σ beyond this (deg or mm) is a direction nothing pins


@dataclass
class GuideSettings:
    need_deg: float = 0.1             # 1σ requirement per rotation axis
    need_mm: float = 10.0             # 1σ requirement per position axis
    safety: float = 1.5               # predicted σ × safety must meet the requirement (the Monte Carlo of the
                                      # observability study found the joint solve's yaw up to 1.47× its prediction)
    pixel_sigma: float = 0.3          # corner noise, px (sub-pixel corners)
    board_sigma_mm: float = 5.0       # how precisely the floor boards are laid (a mat or a jig)
    board_sigma_deg: float = 0.25
    aim_tol_deg: float = 1.0          # aim stage: every axis within this of the plan
    shim_base_mm: float = 40.0        # mounting base length the shims are computed for
    k_sigma: float = 3.0              # robust coverage: margin ≥ k σ
    nominal_fraction: float = 0.9     # connect: a stream at this fraction of its nominal frame rate or more
    frames_per_board: int = 5         # detections averaged per board
    coverage_tol_pct: float = 0.5     # coverage: robust coverage may fall this far below the plan's
    seam_tol_mm: float = 20.0         # verify: 90th percentile of the seams' floor disagreement
    chi2_alpha: float = 0.01          # χ² tests at this level
    min_square_px: float = 12.0       # a board's squares at least this wide (foreshortened side) for the detector
                                      # to find it: detect_on_plane found boards down to 9 px and missed every one
                                      # below 12 px in a synthetic six-camera test (the margin is the 12)
    check_boards: list = field(default_factory=list)      # floor board names left out of the solve, for verify

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        d = d or {}
        return GuideSettings(**{k: v for k, v in d.items() if k in GuideSettings.__dataclass_fields__})


# =========================================================================== BLAS threads
_blas_libs = None
_blas_lock = threading.Lock()


def _openblas():
    """[(set, get)] of every OpenBLAS the process has loaded (numpy's and scipy's wheels each bring one)."""
    global _blas_libs
    if _blas_libs is not None:
        return _blas_libs
    import ctypes
    out = []
    try:
        with open('/proc/self/maps') as f:
            paths = sorted({ln.split()[-1] for ln in f if 'openblas' in ln.lower() and '.so' in ln.split()[-1]})
    except OSError:
        paths = []
    for path in paths:
        try:
            lib = ctypes.CDLL(path)
        except OSError:
            continue
        for suffix in ('64_', '_64_', ''):
            st, gt = getattr(lib, f'openblas_set_num_threads{suffix}', None), \
                getattr(lib, f'openblas_get_num_threads{suffix}', None)
            if st is not None and gt is not None:
                gt.restype = ctypes.c_int
                out.append((st, gt))
                break
    _blas_libs = out
    return out


def _load_scipy():
    """scipy's linear algebra loaded before single_blas looks for the OpenBLAS libraries (its wheel may bring its own)."""
    global _blas_libs
    import importlib
    import sys
    if 'scipy.linalg' not in sys.modules:
        importlib.import_module('scipy.linalg')
        _blas_libs = None


class single_blas:
    """OpenBLAS on one thread while the block runs. The guide's linear algebra is many small matrices (6 × 6 to a
    few hundred square); on a Jetson whose cores the camera pipelines keep busy, OpenBLAS's worker threads spin-wait
    for each other and a 36 × 36 inverse took 62 ms instead of 0.04 ms (AGX Orin, six streams and a desktop running).
    A no-op where no OpenBLAS is loaded."""

    def __enter__(self):
        _blas_lock.acquire()
        self.saved = []
        for st, gt in _openblas():
            try:
                self.saved.append((st, int(gt())))
                st(1)
            except Exception:
                pass
        return self

    def __exit__(self, *exc):
        for st, n in self.saved:
            try:
                st(n)
            except Exception:
                pass
        _blas_lock.release()
        return False


# =========================================================================== pose helpers
def w2o_of(R) -> np.ndarray:
    return BODY_TO_OPTICAL @ np.asarray(R, float).T


def R_of_w2o(W2O) -> np.ndarray:
    return (BODY_TO_OPTICAL.T @ np.asarray(W2O, float)).T


def planned_pose(cam: Camera):
    return rot_zyx(cam.yaw, cam.pitch, cam.roll), cam.pos()


def perturb(pose, x6):
    """The pose moved by x6 = (rotation vector in the camera's optical frame, applied on the left of world→optical;
    shift in the world frame), core.joint's parametrisation."""
    R, pos = pose
    W2O = _rotvec_to_matrix(np.asarray(x6[:3], float)) @ w2o_of(R)
    return R_of_w2o(W2O), np.asarray(pos, float) + np.asarray(x6[3:], float)


def project_pts(intr, pose, P, scale=(1.0, 1.0)):
    """World points (K,3) -> (pixels (K,2) at the image's scale, inside the image and in front)."""
    R, pos = pose
    d = (np.asarray(P, float) - np.asarray(pos, float)) @ w2o_of(R).T
    u, v, th, ok = intr.project_dirs(d)
    return np.stack([u * scale[0], v * scale[1]], -1), ok


def proj_jac(intr, pose, P, scale=(1.0, 1.0)) -> np.ndarray:
    """(K, 2, 6): pixels of P per unit of the pose parameters (perturb)."""
    P = np.asarray(P, float)
    J = np.zeros((len(P), 2, 6))
    for k in range(6):
        e = np.zeros(6)
        e[k] = FD
        a, _ = project_pts(intr, perturb(pose, e), P, scale)
        b, _ = project_pts(intr, perturb(pose, -e), P, scale)
        J[:, :, k] = (a - b) / (2 * FD)
    return J


def euler_jac(pose) -> np.ndarray:
    """∂(yaw, pitch, roll)/∂(rotation vector), degrees per radian (3×3)."""
    J = np.zeros((3, 3))
    for k in range(3):
        e = np.zeros(6)
        e[k] = FD
        a = np.array(euler_zyx(perturb(pose, e)[0]))
        b = np.array(euler_zyx(perturb(pose, -e)[0]))
        J[:, k] = [wrap180(x) for x in (a - b)] / np.array(2 * FD)
    return J


def sigmas(pose, cov6) -> dict:
    """1σ of yaw, pitch, roll (deg) and x, y, z (mm) from a 6×6 covariance in the perturb parameters; a direction
    nothing pins is inf."""
    if cov6 is None:
        return {k: math.inf for k in SIGMA_KEYS}
    T = np.zeros((6, 6))
    T[:3, :3] = euler_jac(pose)
    T[3:, 3:] = np.eye(3) * 1000.0
    C = T @ np.asarray(cov6, float) @ T.T
    d = np.sqrt(np.maximum(np.diag(C), 0.0))
    return {k: (float(v) if math.isfinite(v) and v < UNDETERMINED else math.inf) for k, v in zip(SIGMA_KEYS, d)}


def ratio(sig: dict, st: GuideSettings) -> tuple[float, str]:
    """(worst σ × safety in units of the requirement, its axis)."""
    best = (0.0, '')
    for k, v in sig.items():
        r = v * st.safety / (st.need_deg if k.endswith('_deg') else st.need_mm)
        if not math.isfinite(r):
            return math.inf, k.split('_')[0]
        if r > best[0]:
            best = (r, k.split('_')[0])
    return best


def pose_doc(pose) -> dict:
    R, pos = pose
    y, p, r = euler_zyx(np.asarray(R, float))
    return {'x': float(pos[0]), 'y': float(pos[1]), 'z': float(pos[2]), 'yaw': float(y), 'pitch': float(p),
            'roll': float(r)}


def pose_delta(pose, cam: Camera) -> dict:
    """Measured minus planned: angles in degrees, positions in mm."""
    d = pose_doc(pose)
    return {'yaw_deg': wrap180(d['yaw'] - cam.yaw), 'pitch_deg': wrap180(d['pitch'] - cam.pitch),
            'roll_deg': wrap180(d['roll'] - cam.roll), 'x_mm': (d['x'] - cam.x) * 1000,
            'y_mm': (d['y'] - cam.y) * 1000, 'z_mm': (d['z'] - cam.z) * 1000}


def _scale(intr, shape):
    h, w = shape[:2]
    return w / intr.width, h / intr.height


def _intr(project, cam):
    return resolve(project.module_for(cam), project.lens_for(cam))


def chi2_sf(x, dof) -> float:
    """P(χ²(dof) ≥ x)."""
    from scipy.stats import chi2
    return float(chi2.sf(max(0.0, float(x)), max(1, int(dof))))


def chi2_test(stat, dof, alpha) -> dict:
    p = chi2_sf(stat, dof)
    return {'chi2': float(stat), 'dof': int(dof), 'p': p, 'pass': p >= alpha,
            'scale': float(math.sqrt(max(1.0, stat / max(dof, 1))))}


# =========================================================================== stage 2: identify the ports
def identify_covered(baseline: dict, current: dict, drop=0.5, steady=0.8) -> tuple[str | None, str]:
    """The stream the installer covered: the one whose mean brightness fell below `drop` of its baseline while every
    other stayed above `steady` of its own. Returns (key, why); key None when no stream or several fell (the hand
    shaded two lenses, the lights changed)."""
    fell, moved = [], []
    for k, b in baseline.items():
        c = current.get(k)
        if c is None or not b or b <= 1e-6:
            continue
        r = c / b
        if r < drop:
            fell.append((r, k))
        elif r < steady:
            moved.append(k)
    if len(fell) == 1 and not moved:
        return fell[0][1], f'{fell[0][1]} fell to {fell[0][0]:.0%} of its brightness'
    if not fell:
        return None, 'no stream went dark yet: cover the lens completely'
    if len(fell) > 1:
        return None, 'several streams went dark at once (' + ', '.join(k for _, k in sorted(fell)) + '): cover one'
    return None, 'other streams changed too (' + ', '.join(moved) + '): the light changed; hold still and try again'


def mean_brightness(frame) -> float:
    f = np.asarray(frame)
    if f.ndim == 3:
        f = f[::8, ::8].mean(axis=2)
    else:
        f = f[::8, ::8]
    return float(f.mean())


# =========================================================================== stage 3: lenses
def lens_status(project, cam: Camera) -> dict:
    rec = cam.calib or {}
    if rec:
        from .calib import problems
        probs = problems(rec, project.module_for(cam), project.nominal_lens_for(cam))
        return {'camera': cam.tag, 'source': 'unit calibration', 'serial': rec.get('serial', ''),
                'rms_px': rec.get('rms_px'), 'views': rec.get('views'), 'problems': probs, 'ok': not probs}
    lens = project.nominal_lens_for(cam)
    return {'camera': cam.tag, 'source': 'library model', 'lens': lens.name, 'problems': [], 'ok': False}


# =========================================================================== stage 4: aim from the body outline
@dataclass
class AimModel:
    """The planned body outline of one camera, as rays: what estimate_aim turns to meet the live image."""
    tag: str
    w: int
    h: int
    sensor: tuple
    dirs: np.ndarray           # (K, 3) optical-frame unit rays of the planned outline
    normals: np.ndarray        # (K, 2) unit normals (working pixels), into the body
    valid: np.ndarray          # (h, w) inside the image circle
    pose: tuple                # the planned (R, position)
    n_edge: int = 0


def aim_model(project, engine, cam: Camera, width=960, max_points=3000) -> AimModel:
    from .rig_health import expected_view
    ev = expected_view(project, engine, cam, width=width)
    intr = _intr(project, cam)
    W, H = ev.sensor
    pts = ev.edge.astype(float)
    nrm = ev.normal.astype(float)
    if len(pts) > max_points:
        k = np.linspace(0, len(pts) - 1, max_points).astype(int)
        pts, nrm = pts[k], nrm[k]
    # the outline pixels are the body's own; the edge in the image lies half a pixel outwards (normals point in)
    u = (pts[:, 0] + 0.5 - 0.5 * nrm[:, 0]) * W / ev.w
    v = (pts[:, 1] + 0.5 - 0.5 * nrm[:, 1]) * H / ev.h
    dirs, _ = intr.unproject(u, v)
    return AimModel(cam.tag, ev.w, ev.h, (W, H), dirs, nrm, ev.valid, planned_pose(cam), len(ev.edge))


def estimate_aim(frame, am: AimModel, intr, search_deg=6.0) -> dict:
    """The camera's rotation on its mount from one live frame: the rotation Rc (optical frame) that best lays the
    planned body outline on edges of the image running the same way. The image shows a fixed direction d0 (planned
    optical frame) at Rcᵀ d0, so the outline's rays are turned by Rc and projected; the score is the mean oriented
    gradient across the outline (each point capped at the image's 95th percentile, as rig_health.alignment). Search:
    pan and tilt on a 0.75° grid over ±search_deg, then roll, then all three finer (0.25°, then 0.05°).

    Returns the measured yaw/pitch/roll against the plan (degrees, the Vantage frame: pitch positive down), the
    support (share of the outline meeting an edge), and per axis whether the outline pins it (the score falls by at
    least 3 % a degree either side); None values when the outline is too short or meets too few edges."""
    import cv2
    out = {'camera': am.tag, 'ok': False, 'support': 0.0, 'points': int(len(am.dirs)),
           'd_yaw_deg': None, 'd_pitch_deg': None, 'd_roll_deg': None,
           'determined': {'yaw': False, 'pitch': False, 'roll': False}, 'why': ''}
    if len(am.dirs) < 40:
        out['why'] = 'too little of the body is in this camera\'s view to aim by it: aim it with the boards'
        return out
    g0 = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    g0 = cv2.resize(g0, (am.w, am.h), interpolation=cv2.INTER_AREA).astype(np.float32)
    sx, sy = am.w / am.sensor[0], am.h / am.sensor[1]
    nx, ny = am.normals[:, 0], am.normals[:, 1]

    def maps(blur):
        g = cv2.GaussianBlur(g0, (0, 0), blur)
        gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
        G = np.hypot(gx, gy)[am.valid]
        floor = max(8.0 / blur, 3.0 * float(np.median(G)))
        cap = max(floor, float(np.percentile(G, 95)))
        gx[~am.valid] = 0
        gy[~am.valid] = 0
        return gx, gy, floor, cap
    coarse_maps, fine_maps = maps(3.0), maps(1.0)

    def oriented(rv, m=fine_maps):
        gx, gy = m[0], m[1]
        Rc = _rotvec_to_matrix(np.radians(rv))
        d = am.dirs @ Rc
        u, v, _, ok = intr.project_dirs(d)
        x = np.clip(np.rint(u * sx - 0.5).astype(int), 0, am.w - 1)
        y = np.clip(np.rint(v * sy - 0.5).astype(int), 0, am.h - 1)
        val = np.abs(gx[y, x] * nx + gy[y, x] * ny)
        return np.where(ok, val, 0.0)

    def score(rv, m=fine_maps):
        return float(np.minimum(oriented(rv, m), m[3]).mean())
    S = float(search_deg)
    # coarse: all three axes on a 1° grid over a smoothed image (a wide basin round the answer), then finer on the
    # sharp image (0.25°, then 0.05°)
    grid = np.arange(-S, S + 1e-9, 1.0)
    rgrid = np.arange(-min(S, 4.0), min(S, 4.0) + 1e-9, 1.0)
    coarse = sorted(((score((t, p_, r), coarse_maps), t, p_, r) for t in grid for p_ in grid for r in rgrid),
                    reverse=True)
    # the smoothed image's best few basins (at least 1.5° apart), each refined on the sharp image: a straight body
    # edge makes a pan and a roll look alike from afar, and the sharp image tells them apart
    seeds = []
    for sc_, t, p_, r in coarse:
        pt = np.array([t, p_, r], float)
        if all(np.abs(pt - q).max() >= 1.5 for q in seeds):
            seeds.append(pt)
        if len(seeds) >= 6:
            break
    results = []
    for cur in seeds:
        for step, span, m in ((0.5, 1.0, coarse_maps), (0.25, 0.75, fine_maps), (0.05, 0.25, fine_maps)):
            g_ = np.arange(-span, span + 1e-9, step)
            cands = [(score(cur + (a, b, c), m), a, b, c) for a in g_ for b in g_ for c in g_]
            _, a, b, c = max(cands)
            cur = cur + (a, b, c)
        results.append((score(cur), cur))
    results.sort(key=lambda x: -x[0])
    cur = results[0][1]
    # a second basin nearly as good and far off: the outline cannot choose between them
    rival = next((r for r in results[1:] if np.abs(r[1] - cur).max() > 1.0), None)
    ambiguous = rival is not None and rival[0] >= 0.97 * results[0][0]
    floor = fine_maps[2]
    top = score(cur)
    support = float((oriented(cur) > floor).mean())
    out['support'] = support
    if support < 0.3:
        out['why'] = (f'only {support:.0%} of the planned body outline meets an edge in the image (the lens is '
                      'covered, the body and what is behind it look alike, or the camera is far off its plan)')
        return out
    # How firmly the outline pins each axis: the score's curvature at the optimum (central differences, 0.5°) as a
    # quadratic drop s(δ) ≈ ½ δᵀ A δ (relative to the peak); the turn of one Euler angle, the others free, that costs
    # a 3 % drop is √(2 · 0.03 · [Je A⁻¹ Jeᵀ]ᵢᵢ). An axis is pinned when that is within 1°: a straight outline (a
    # bumper's top edge) cannot tell a pan from a roll, and both are then left to the boards.
    h = 0.5
    A = np.zeros((3, 3))
    for i in range(3):
        for j in range(i, 3):
            ei, ej = np.eye(3)[i] * h, np.eye(3)[j] * h
            if i == j:
                v_ = (score(cur + ei) - 2 * top + score(cur - ei)) / h ** 2
            else:
                v_ = (score(cur + ei + ej) - score(cur + ei - ej) - score(cur - ei + ej) + score(cur - ei - ej)) / (4 * h * h)
            A[i, j] = A[j, i] = -v_ / max(top, 1e-9)
    Rc = _rotvec_to_matrix(np.radians(cur))
    R0, pos0 = am.pose
    Ra = R_of_w2o(Rc.T @ w2o_of(R0))
    y, p, r = euler_zyx(Ra)
    y0, p0_, r0_ = euler_zyx(R0)
    Je = euler_jac((R0, pos0)) * math.pi / 180.0          # degrees of Euler angle per degree of optical rotation
    try:
        lam, V = np.linalg.eigh((A + A.T) / 2)
        lam = np.maximum(lam, 1e-9)
        Ai = (V / lam) @ V.T
        spread = np.sqrt(np.maximum(2 * 0.03 * np.diag(Je @ Ai @ Je.T), 0.0))
    except np.linalg.LinAlgError:
        spread = np.full(3, np.inf)
    determined = {name: bool(spread[i] <= 1.0) for i, name in enumerate(('yaw', 'pitch', 'roll'))}
    if ambiguous:
        # the axes the two answers differ on are not pinned
        Rr = _rotvec_to_matrix(np.radians(rival[1]))
        yr, pr, rr = euler_zyx(R_of_w2o(Rr.T @ w2o_of(R0)))
        Rb_ = _rotvec_to_matrix(np.radians(cur))
        yb, pb, rb = euler_zyx(R_of_w2o(Rb_.T @ w2o_of(R0)))
        for name, a1, a2 in (('yaw', yb, yr), ('pitch', pb, pr), ('roll', rb, rr)):
            if abs(wrap180(a1 - a2)) > 0.5:
                determined[name] = False
    out.update(ok=True, d_yaw_deg=wrap180(y - y0), d_pitch_deg=wrap180(p - p0_), d_roll_deg=wrap180(r - r0_),
               determined=determined, score=top, optical_deg=[float(c) for c in cur],
               spread_deg={k: float(v) for k, v in zip(('yaw', 'pitch', 'roll'), spread)}, ambiguous=bool(ambiguous),
               pose=(Ra, np.asarray(pos0, float)))
    return out


def shim_mm(angle_deg, base_mm) -> float:
    """Shim thickness that turns a mounting base of length base_mm by angle_deg: base × tan(angle)."""
    return abs(base_mm * math.tan(math.radians(angle_deg)))


def aim_instructions(aim: dict, st: GuideSettings) -> list[str]:
    """What to do with the mount, from the camera's own point of view (standing behind it, looking where it looks).
    Vantage angles: yaw positive turns left (towards +y), pitch positive tilts down, roll positive turns the image
    clockwise as the camera sees it (its right side down)."""
    if not aim.get('ok'):
        return [aim.get('why') or 'no aim estimate']
    out = []
    tol = st.aim_tol_deg
    for key, neg, pos in (('d_yaw_deg', 'right', 'left'), ('d_pitch_deg', 'up', 'down'),
                          ('d_roll_deg', 'its left side down', 'its right side down')):
        v = aim[key]
        axis = key.split('_')[1]
        if not aim['determined'].get(axis, True):
            out.append(f'{axis}: not visible in the body outline; the boards will measure it')
            continue
        if abs(v) <= tol:
            out.append(f'{axis}: {v:+.1f}° from the plan, within ±{tol:g}°')
            continue
        # the correction undoes the error: a camera turned left (yaw +) goes back right
        word = neg if v > 0 else pos
        verb = 'turn' if axis == 'yaw' else 'tilt' if axis == 'pitch' else 'roll'
        out.append(f'{axis}: {v:+.1f}° from the plan: {verb} it {abs(v):.1f}° {word} '
                   f'(a {shim_mm(v, st.shim_base_mm):.1f} mm shim across a {st.shim_base_mm:g} mm base)')
    return out


def ground_shift_mm(h, d, delta_deg) -> float:
    """How far a ground point at distance d moves when a camera at height h tilts by delta: δ (h² + d²)/h."""
    return abs(math.radians(delta_deg)) * (h * h + d * d) / max(h, 1e-6) * 1000.0


# =========================================================================== boards on stands
@dataclass
class StandBoard:
    """A checkerboard upright on a stand: its centre at (x, y, z), its printed face looking towards azimuth yaw
    (degrees, vehicle frame). Seen by a solved camera (parent) and one the floor boards cannot reach (child), it ties
    the child to the parent; its own place is solved, never measured."""
    uid: str = field(default_factory=new_uid)
    name: str = 'S1'
    x: float = 6.0
    y: float = 0.0
    z: float = 0.9
    yaw: float = 180.0
    rows: int = 7               # squares
    cols: int = 9
    square_mm: float = 120.0
    parent: str = ''            # camera tags
    child: str = ''

    @property
    def size_m(self):
        return self.cols * self.square_mm / 1000.0, self.rows * self.square_mm / 1000.0

    def frame(self):
        """(R board→world, centre): board axes u (along, to its right as seen from the front), v up, n out of the
        printed face."""
        a = math.radians(self.yaw)
        n = np.array([math.cos(a), math.sin(a), 0.0])
        vv = np.array([0.0, 0.0, 1.0])
        uu = np.cross(vv, n)
        return np.stack([uu, vv, n], 1), np.array([self.x, self.y, self.z])

    def local_corners(self) -> np.ndarray:
        """(rows-1, cols-1, 3) inner corners in the board's frame, row-major."""
        sq = self.square_mm / 1000
        w, h = self.size_m
        i = np.arange(1, self.cols) * sq - w / 2
        j = h / 2 - np.arange(1, self.rows) * sq
        U, V = np.meshgrid(i, j)
        return np.stack([U, V, np.zeros_like(U)], -1)

    def local_outline(self) -> np.ndarray:
        w, h = self.size_m
        return np.array([[-w / 2, h / 2, 0], [w / 2, h / 2, 0], [w / 2, -h / 2, 0], [-w / 2, -h / 2, 0]], float)

    def world(self, pose=None):
        """Inner corners (rows-1, cols-1, 3) in the world: at the planned place, or at pose = (R, centre)."""
        R, c = self.frame() if pose is None else pose
        return self.local_corners() @ np.asarray(R).T + np.asarray(c)

    def outline(self, pose=None):
        R, c = self.frame() if pose is None else pose
        return self.local_outline() @ np.asarray(R).T + np.asarray(c)

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        return StandBoard(**{k: v for k, v in (d or {}).items() if k in StandBoard.__dataclass_fields__})


def stands_of(project) -> list[StandBoard]:
    g = getattr(project, 'install_guide', None) or {}
    return [StandBoard.from_dict(s) for s in g.get('stands', [])]


def board_in_view(intr, pose, P, outline, margin=0.03, edge_deg=3.0, min_square_px=0.0, grid_shape=None) -> dict:
    """Whether a board's corners and outline all lie in the image (a margin of the frame and edge_deg inside the
    image circle) in front of the camera, and its smallest square in pixels (from neighbouring corners)."""
    R, pos = pose
    pts = np.vstack([np.asarray(P, float).reshape(-1, 3), np.asarray(outline, float)])
    d = (pts - np.asarray(pos)) @ w2o_of(R).T
    u, v, th, ok = intr.project_dirs(d)
    m = margin
    inside = (ok & (d[:, 2] > 0) & (u > m * intr.width) & (u < (1 - m) * intr.width) & (v > m * intr.height)
              & (v < (1 - m) * intr.height) & (th <= intr.max_theta - math.radians(edge_deg)))
    sq = math.inf
    if grid_shape is not None:
        uv = np.stack([u, v], -1)[:np.prod(grid_shape)].reshape(*grid_shape, 2)
        a = np.linalg.norm(np.diff(uv, axis=0), axis=-1).min() if grid_shape[0] > 1 else math.inf
        b = np.linalg.norm(np.diff(uv, axis=1), axis=-1).min() if grid_shape[1] > 1 else math.inf
        sq = float(min(a, b))
    return {'inside': bool(inside.all()), 'square_px': sq, 'ok': bool(inside.all()) and sq >= min_square_px}


_inside_cache: dict = {}


def inside_body(engine, vg, pos) -> bool:
    """Whether a camera sits inside the body's shell (behind the windscreen or the rear window): it looks out through
    the shell round it, glass or not (many body models have no separate windscreen)."""
    if engine is None or vg is None:
        return False
    key = (getattr(vg, 'version', id(vg)), tuple(np.round(np.asarray(pos, float), 4)))
    r = _inside_cache.get(key)
    if r is None:
        from .materials import _inside
        r = _inside(vg.occ_mesh, np.asarray(pos, float))
        if len(_inside_cache) > 256:
            _inside_cache.clear()
        _inside_cache[key] = r
    return r


def _clear(engine, vg, pos, P, inside=False) -> bool:
    """Nothing of the vehicle between a camera at pos and the points. inside: the camera sits inside the shell, which
    it looks out through: the first surface a ray crosses is its window, anything after that blocks."""
    if engine is None or vg is None:
        return True
    d = np.asarray(P, float) - np.asarray(pos, float)
    dist = np.linalg.norm(d, axis=1)
    if inside:
        from .materials import _hits
        T = _hits(np.asarray(pos, float), d / dist[:, None], np.asarray(vg.occ_mesh.tri), float(dist.max()))
        T = np.sort(T, axis=1)
        second = T[:, 1] if T.shape[1] > 1 else np.full(len(T), np.inf)
        return bool(np.all(second > dist - 0.02))
    depth, _ = engine.cube(np.asarray(pos, float), vg).lookup(d)
    return bool(np.all(~np.isfinite(depth) | (depth > dist - 0.02 - 0.01 * dist)))


def _glass_of(engine, vg, pose) -> bool:
    return inside_body(engine, vg, pose[1])


def _board_param_pose(board_pose, x6):
    """Board pose moved by x6 = (rotation vector in the world, applied on the left; shift in the world)."""
    R, c = board_pose
    return _rotvec_to_matrix(np.asarray(x6[:3], float)) @ np.asarray(R, float), np.asarray(c, float) + x6[3:]


def _uv_board(intr, cam_pose, board_pose, L, scale):
    P = L @ np.asarray(board_pose[0]).T + np.asarray(board_pose[1])
    uv, _ = project_pts(intr, cam_pose, P, scale)
    return uv


def chain_covariance(intr_p, pose_p, cov_p, intr_c, pose_c, board_pose, L, sigma_px, scale_p=(1, 1),
                     scale_c=(1, 1)) -> dict:
    """Covariance of a child camera chained to a parent through one board both see (module docstring): the board's
    pose and the child's from both cameras' corners, the parent's error propagated through. L: (N, 3) the board's
    corners in its own frame. Returns {'child': 6×6, 'board': 6×6, 'child_given_parent': 6×6 (the parent exact),
    'cross': 6×6 Cov(child, parent)}."""
    L = np.asarray(L, float).reshape(-1, 3)
    N = len(L)

    def fp(xb, xp):
        return _uv_board(intr_p, perturb(pose_p, xp), _board_param_pose(board_pose, xb), L, scale_p).ravel()

    def fc(xb, xc):
        return _uv_board(intr_c, perturb(pose_c, xc), _board_param_pose(board_pose, xb), L, scale_c).ravel()
    z = np.zeros(6)
    JpB, JpP, JcB, JcC = (np.zeros((2 * N, 6)) for _ in range(4))
    for k in range(6):
        e = np.zeros(6)
        e[k] = FD
        JpB[:, k] = (fp(e, z) - fp(-e, z)) / (2 * FD)
        JpP[:, k] = (fp(z, e) - fp(z, -e)) / (2 * FD)
        JcB[:, k] = (fc(e, z) - fc(-e, z)) / (2 * FD)
        JcC[:, k] = (fc(z, e) - fc(z, -e)) / (2 * FD)
    Jx = np.block([[JpB, np.zeros((2 * N, 6))], [JcB, JcC]]) / sigma_px
    A = np.vstack([JpP, np.zeros((2 * N, 6))]) / sigma_px
    H = Jx.T @ Jx
    Hi = np.linalg.pinv(H)
    G = -Hi @ Jx.T @ A
    cp = np.zeros((6, 6)) if cov_p is None else np.asarray(cov_p, float)
    Sx = Hi + G @ cp @ G.T
    Sx = (Sx + Sx.T) / 2
    return {'child': Sx[6:, 6:], 'board': Sx[:6, :6], 'child_given_parent': Hi[6:, 6:], 'cross': G[6:] @ cp}


# =========================================================================== joint covariance of the floor solve
def floor_blocks(project, cams, obs: dict, poses: dict, shapes: dict, targets=None):
    """Observation blocks for joint_covariance from core.install BoardObs: [(cam index, board uid, world (N,3),
    image (N,2))]."""
    tl = {t.uid: t for t in (project.targets if targets is None else targets)}
    out = []
    for ci, c in enumerate(cams):
        for o in obs.get(c.uid, []):
            if o.target_uid in tl:
                out.append((ci, o.target_uid, np.asarray(o.world, float), np.asarray(o.image_pts, float)))
    return out


def joint_covariance(project, cams, blocks, poses: dict, shapes: dict, st: GuideSettings, targets=None) -> dict:
    """Covariance of the joint floor solve at its answer (module docstring): per camera uid its 6×6 (perturb
    parameters), every board's placement marginalised. blocks: floor_blocks(...). Returns {'cams': {uid: 6×6 or None
    when nothing pins it}, 'full': the whole matrix, 'index': {uid: column}, 'boards': [uid]}."""
    tl = {t.uid: t for t in (project.targets if targets is None else targets)}
    board_ids = []
    for b in blocks:
        if b[1] not in board_ids:
            board_ids.append(b[1])
    nc, nb = len(cams), len(board_ids)
    n = 6 * nc + 3 * nb
    H = np.zeros((n, n))
    sig_b = max(st.board_sigma_mm, 1e-3) / 1000.0
    sig_y = math.radians(max(st.board_sigma_deg, 1e-4))
    for bi in range(nb):
        H[6 * nc + 3 * bi, 6 * nc + 3 * bi] = 1 / sig_b ** 2
        H[6 * nc + 3 * bi + 1, 6 * nc + 3 * bi + 1] = 1 / sig_b ** 2
        H[6 * nc + 3 * bi + 2, 6 * nc + 3 * bi + 2] = 1 / sig_y ** 2
    intr = [_intr(project, c) for c in cams]
    for ci, buid, W, _uv in blocks:
        c = cams[ci]
        pose = poses[c.uid]
        sc = _scale(intr[ci], shapes[c.uid])
        bi = board_ids.index(buid)
        t = tl[buid]
        ctr = np.array([t.x, t.y, 0.0])
        Jc = proj_jac(intr[ci], pose, W, sc).reshape(-1, 6)
        Jb = np.zeros((2 * len(W), 3))
        for k in range(3):
            e = np.zeros(3)
            e[k] = FD

            def moved(x):
                cz, sz = math.cos(x[2]), math.sin(x[2])
                Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
                return (W - ctr) @ Rz.T + ctr + np.array([x[0], x[1], 0.0])
            a, _ = project_pts(intr[ci], pose, moved(e), sc)
            b, _ = project_pts(intr[ci], pose, moved(-e), sc)
            Jb[:, k] = ((a - b) / (2 * FD)).ravel()
        J = np.zeros((2 * len(W), n))
        J[:, 6 * ci:6 * ci + 6] = Jc
        J[:, 6 * nc + 3 * bi:6 * nc + 3 * bi + 3] = Jb
        H += J.T @ J / st.pixel_sigma ** 2
    full = np.linalg.pinv(H)
    covs = {}
    seen = {b[0] for b in blocks}
    for ci, c in enumerate(cams):
        if ci not in seen:
            covs[c.uid] = None
            continue
        C = full[6 * ci:6 * ci + 6, 6 * ci:6 * ci + 6]
        covs[c.uid] = (C + C.T) / 2
    return {'cams': covs, 'full': full, 'index': {c.uid: 6 * ci for ci, c in enumerate(cams)}, 'boards': board_ids}


# =========================================================================== planning
def visible_floor_corners(project, engine, cam: Camera, pose=None, targets=None, min_square_px=12.0, shape=None):
    """Floor boards the camera sees whole at a pose (planned when None): [(target, world (N,3), pixels (N,2))]."""
    intr = _intr(project, cam)
    pose = planned_pose(cam) if pose is None else pose
    sc = (1.0, 1.0) if shape is None else _scale(intr, shape)
    vg = engine.vehicle_geom(project) if engine is not None else None
    out = []
    for t in (project.targets if targets is None else targets):
        W = inner_corners(t)
        flat = W.reshape(-1, 3)
        from .install import board_outline
        # what the detector needs (detect_on_plane, measured on synthetic captures): every corner and the board's
        # outline in the image, and squares of min_square_px or more
        vis = board_in_view(intr, pose, flat, board_outline(t), margin=0.0, edge_deg=0.0,
                            min_square_px=min_square_px, grid_shape=W.shape[:2])
        if not vis['ok'] or not _clear(engine, vg, pose[1], flat, _glass_of(engine, vg, pose)):
            continue
        uv, _ = project_pts(intr, pose, flat, sc)
        out.append((t, flat, uv))
    return out


def planned_floor_covariance(project, engine, cams, st: GuideSettings, targets=None) -> dict:
    """joint_covariance at the plan (every camera at its planned pose, every visible board's corners exact)."""
    poses = {c.uid: planned_pose(c) for c in cams}
    shapes = {}
    blocks = []
    for ci, c in enumerate(cams):
        intr = _intr(project, c)
        shapes[c.uid] = (intr.height, intr.width)
        for t, W, uv in visible_floor_corners(project, engine, c, targets=targets, min_square_px=st.min_square_px):
            blocks.append((ci, t.uid, W, uv))
    return joint_covariance(project, cams, blocks, poses, shapes, st, targets) | {'blocks': blocks}


STAND_SIZES = ((7, 9, 120.0), (8, 11, 100.0), (6, 8, 150.0), (6, 8, 100.0))     # (rows, cols, square mm)


def plan_stand(project, engine, parent: Camera, child: Camera, cov_parent, st: GuideSettings,
               sizes=STAND_SIZES) -> dict | None:
    """The board on a stand that pins the child best through the parent: candidates along the child's view (3–10 m
    out, ±1 m aside, its centre 0.5–1.4 m high with the lower edge at least 0.15 m off the floor, facing between the
    two cameras), kept when the whole board lies in both images with squares of min_square_px or more and nothing of
    the car between; scored by the child's chained covariance (worst axis × safety over the requirement)."""
    ip, ic = _intr(project, parent), _intr(project, child)
    pp, pc = planned_pose(parent), planned_pose(child)
    vg = engine.vehicle_geom(project) if engine is not None else None
    gp, gc = _glass_of(engine, vg, pp), _glass_of(engine, vg, pc)
    fwd = w2o_of(pc[0])[2]
    az = math.atan2(fwd[1], fwd[0])
    ax, ay = math.cos(az), math.sin(az)
    best = None
    tried = 0
    for rows, cols, sq in sizes:
        for dist in (3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0):
            for lat in (0.0, -0.5, 0.5, -1.0, 1.0):
                for zc in (0.45, 0.6, 0.8, 1.1, 1.4):
                    b = StandBoard(rows=rows, cols=cols, square_mm=sq, parent=parent.tag, child=child.tag)
                    if zc - b.size_m[1] / 2 < 0.15:
                        continue
                    x = pc[1][0] + ax * dist - ay * lat
                    y = pc[1][1] + ay * dist + ax * lat
                    mid = (pp[1] + pc[1]) / 2
                    b.x, b.y, b.z = x, y, zc
                    b.yaw = math.degrees(math.atan2(mid[1] - y, mid[0] - x))
                    P = b.world().reshape(-1, 3)
                    gs = b.local_corners().shape[:2]
                    vp = board_in_view(ip, pp, P, b.outline(), margin=0.015, edge_deg=1.5,
                                       min_square_px=st.min_square_px, grid_shape=gs)
                    vc = board_in_view(ic, pc, P, b.outline(), margin=0.015, edge_deg=1.5,
                                       min_square_px=st.min_square_px, grid_shape=gs)
                    if not (vp['ok'] and vc['ok']):
                        continue
                    if not (_clear(engine, vg, pp[1], P, gp) and _clear(engine, vg, pc[1], P, gc)):
                        continue
                    if dist < ic.focus_near:
                        continue
                    tried += 1
                    cc = chain_covariance(ip, pp, cov_parent, ic, pc, b.frame(), b.local_corners(), st.pixel_sigma)
                    sig = sigmas(pc, cc['child'])
                    r, axis = ratio(sig, st)
                    if best is None or r < best['ratio']:
                        best = {'board': b, 'sigma': sig, 'ratio': r, 'axis': axis, 'distance_m': dist,
                                'square_px': {'parent': vp['square_px'], 'child': vc['square_px']}}
    if best is not None:
        best['candidates'] = tried
    return best


def _floor_reach(project, engine, cam, targets=None, min_square_px=12.0) -> bool:
    """Whether a board laid on the floor can be seen whole by the camera somewhere within 6 m."""
    from .project import Target as T
    pose = planned_pose(cam)
    fwd = w2o_of(pose[0])[2]
    if fwd[2] > 0.05:                        # looks up
        return False
    for d in (1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0):
        for lat in (-1.0, 0.0, 1.0):
            n = np.hypot(fwd[0], fwd[1]) or 1.0
            ax, ay = fwd[0] / n, fwd[1] / n
            t = T(name='probe', x=pose[1][0] + ax * d - ay * lat, y=pose[1][1] + ay * d + ax * lat,
                  yaw=math.degrees(math.atan2(ay, ax)), rows=6, cols=6, square_mm=100.0)
            if visible_floor_corners(project, engine, cam, targets=[t], min_square_px=min_square_px):
                return True
    return False


def parent_for(project, engine, child: Camera, floor_cams: list) -> Camera | None:
    """The floor-solved camera whose view overlaps the child's most (the angle between their optical axes)."""
    fc = w2o_of(planned_pose(child)[0])[2]
    best = None
    for c in floor_cams:
        f = w2o_of(planned_pose(c)[0])[2]
        ang = math.degrees(math.acos(max(-1.0, min(1.0, float(fc @ f)))))
        if best is None or ang < best[0]:
            best = (ang, c)
    return best[1] if best else None


FLOOR_BOARDS = ((6, 6, 100.0), (6, 6, 150.0))      # rows, cols (squares), square mm: the boards the planner adds
                                                   # (0.6 m and 0.9 m: both print on A0)


def _board_blocks(project, engine, cams, t: Target, st: GuideSettings):
    """The information one floor board gives: per camera that sees it whole, (cam index, Hcc 6×6, Hcb 6×3, Hbb 3×3)
    with the corner noise σ (the board's own placement prior is added once by the caller)."""
    out = []
    ctr = np.array([t.x, t.y, 0.0])
    for ci, c in enumerate(cams):
        vis = visible_floor_corners(project, engine, c, targets=[t], min_square_px=st.min_square_px)
        if not vis:
            continue
        _, W, _ = vis[0]
        intr = _intr(project, c)
        pose = planned_pose(c)
        Jc = proj_jac(intr, pose, W).reshape(-1, 6)
        Jb = np.zeros((2 * len(W), 3))
        for k in range(3):
            e = np.zeros(3)
            e[k] = FD

            def moved(x):
                cz, sz = math.cos(x[2]), math.sin(x[2])
                Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
                return (W - ctr) @ Rz.T + ctr + np.array([x[0], x[1], 0.0])
            a, _ = project_pts(intr, pose, moved(e))
            b, _ = project_pts(intr, pose, moved(-e))
            Jb[:, k] = ((a - b) / (2 * FD)).ravel()
        s2 = st.pixel_sigma ** 2
        out.append((ci, Jc.T @ Jc / s2, Jc.T @ Jb / s2, Jb.T @ Jb / s2))
    return out


def _board_schur(nc, blocks, st: GuideSettings) -> np.ndarray:
    """One floor board's information on the cameras with its own placement (x, y, yaw; held near the plan by the
    placement prior) eliminated: Σᵢ Hccᵢ − (Σᵢ Hcbᵢ) (P + Σᵢ Hbbᵢ)⁻¹ (Σⱼ Hcbⱼ)ᵀ over the cameras i, j that see it,
    a 6nc × 6nc matrix. Boards share no parameters, so the cameras' information with every board marginalised is
    the sum of these (exactly the Schur complement of the whole information matrix), and adding a board to a plan
    is adding its term: a 36 × 36 system for six cameras instead of one that grows by three with every board."""
    S = np.zeros((6 * nc, 6 * nc))
    sig_b = max(st.board_sigma_mm, 1e-3) / 1000.0
    sig_y = math.radians(max(st.board_sigma_deg, 1e-4))
    Hbb = np.diag([1 / sig_b ** 2, 1 / sig_b ** 2, 1 / sig_y ** 2])
    for ci, Hcc, Hcb, Hbb_i in blocks:
        S[6 * ci:6 * ci + 6, 6 * ci:6 * ci + 6] += Hcc
        Hbb = Hbb + Hbb_i
    B = np.linalg.inv(Hbb)
    for ci, _, Hcb_i, _ in blocks:
        for cj, _, Hcb_j, _ in blocks:
            S[6 * ci:6 * ci + 6, 6 * cj:6 * cj + 6] -= Hcb_i @ B @ Hcb_j.T
    return S


_T_cache: dict = {}


def _sigma_T(cam):
    """The 6×6 map from the perturb parameters to (yaw, pitch, roll deg; x, y, z mm) at the camera's planned pose."""
    key = (cam.uid, cam.yaw, cam.pitch, cam.roll)
    T = _T_cache.get(key)
    if T is None:
        T = np.zeros((6, 6))
        T[:3, :3] = euler_jac(planned_pose(cam))
        T[3:, 3:] = np.eye(3) * 1000.0
        if len(_T_cache) > 512:
            _T_cache.clear()
        _T_cache[key] = T
    return T


def _ratios(S, cams, st):
    """Each camera's worst σ × safety over the requirement, from the cameras' marginal information S (sum of
    _board_schur terms). A camera no board reaches is inf."""
    nc = len(cams)
    seen = [ci for ci in range(nc) if S[6 * ci, 6 * ci] > 0]
    out = [math.inf] * nc
    if not seen:
        return out, None
    idx = np.concatenate([np.arange(6 * ci, 6 * ci + 6) for ci in seen])
    Ss = S[np.ix_(idx, idx)]
    try:
        C = np.linalg.inv(Ss)
    except np.linalg.LinAlgError:
        C = np.linalg.pinv(Ss)
    for k, ci in enumerate(seen):
        T = _sigma_T(cams[ci])
        d = np.sqrt(np.maximum(np.diag(T @ C[6 * k:6 * k + 6, 6 * k:6 * k + 6] @ T.T), 0.0))
        sig = {key: (float(v) if v < UNDETERMINED else math.inf) for key, v in zip(SIGMA_KEYS, d)}
        out[ci] = ratio(sig, st)[0]
    return out, C


def floor_candidates(project, engine, spacing=0.5, near=0.4, far=4.0, boards=FLOOR_BOARDS) -> list[Target]:
    """Places a board could be laid: a grid round the body, near–far metres outside its outline, two turns."""
    vg = engine.vehicle_geom(project)
    lo, hi = vg.bbox
    out = []
    for x in np.arange(lo[0] - far, hi[0] + far + 1e-9, spacing):
        for y in np.arange(lo[1] - far, hi[1] + far + 1e-9, spacing):
            dx = max(lo[0] - x, 0.0, x - hi[0])
            dy = max(lo[1] - y, 0.0, y - hi[1])
            dd = math.hypot(dx, dy)
            if not (near <= dd <= far):
                continue
            for rows, cols, sq in boards:      # laid square on to the car (yaw 0): easy to lay true by a tape
                out.append(Target(name='cand', x=float(x), y=float(y), yaw=0.0, rows=rows, cols=cols, square_mm=sq))
    return out


def plan_floor(project, engine, cams, st: GuideSettings, max_add=10, progress=None) -> dict:
    """Floor boards for the cameras: the project's boards that some camera sees, then greedily the candidate that
    lowers the cameras' excess over the requirement most (Σ max(0, ratio − 1)², ties to the worst ratio), until every
    camera meets the requirement, nothing helps or max_add boards were added. Each candidate's information is computed
    once, reduced to the cameras (_board_schur); adding it is a sum (the same model as joint_covariance)."""
    nc = len(cams)
    keep, keep_blocks = [], []
    for t in project.targets:
        b = _board_blocks(project, engine, cams, t, st)
        if b:
            keep.append(t)
            keep_blocks.append(b)
    cands = floor_candidates(project, engine)
    cand_blocks = []
    for k, t in enumerate(cands):
        if progress and k % 50 == 0:
            progress(0.6 * k / max(len(cands), 1), 'Where boards could lie')
        b = _board_blocks(project, engine, cams, t, st)
        if b:
            cand_blocks.append((t, b, _board_schur(nc, b, st)))

    def score(r):
        return sum(max(0.0, x - 1.0) ** 2 if math.isfinite(x) else 1e6 for x in r), max(r) if r else 0.0
    S = sum((_board_schur(nc, b, st) for b in keep_blocks), np.zeros((6 * nc, 6 * nc)))
    rat, _ = _ratios(S, cams, st)
    log = []
    added = 0
    names = {t.name for t in project.targets}
    while added < max_add and score(rat)[0] > 0:
        best = None
        for t, b, Sb in cand_blocks:
            w = max(t.size_m) + 0.1
            if any(math.hypot(t.x - u.x, t.y - u.y) < w for u in keep):
                continue
            r, _ = _ratios(S + Sb, cams, st)
            sc = score(r)
            if best is None or sc < best[0]:
                best = (sc, t, b, r, Sb)
        if best is None or best[0] >= score(rat):
            break
        _, t, b, r, Sb = best
        S = S + Sb
        k = len(names) + 1
        while f'T{k}' in names:
            k += 1
        t = Target(name=f'T{k}', x=round(t.x, 3), y=round(t.y, 3), yaw=t.yaw, rows=t.rows, cols=t.cols,
                   square_mm=t.square_mm)
        names.add(t.name)
        keep.append(t)
        keep_blocks.append(b)
        seen = ', '.join(cams[ci].tag for ci, *_ in b)
        log.append(f'Add {t.name} at x {t.x:.2f} m, y {t.y:.2f} m (seen by {seen}): worst camera '
                   f'{max(rat):.2f}× → {max(r):.2f}× the requirement')
        rat = r
        added += 1
        if progress:
            progress(0.6 + 0.3 * added / max_add, log[-1])
    unused = [t.name for t in project.targets if t not in keep]
    return {'targets': keep, 'ratios': {c.tag: r for c, r in zip(cams, rat)}, 'log': log, 'unused': unused,
            'candidates': cand_blocks}


def extra_check_board(project, engine, cams, keep: list, cand_blocks: list, st: GuideSettings) -> Target | None:
    """A board for the verify stage only, where two cameras see it best (the most information in its second
    camera), clear of the boards already planned, with squares of 1.25 × min_square_px or more in both (the test must
    not hinge on a board at the detector's limit)."""
    best = None
    for t, b, _ in cand_blocks:
        if len(b) < 2:
            continue
        w = max(t.size_m) + 0.1
        if any(math.hypot(t.x - u.x, t.y - u.y) < w for u in keep):
            continue
        big = [c for c in cams if visible_floor_corners(project, engine, c, targets=[t],
                                                        min_square_px=1.25 * st.min_square_px)]
        if len(big) < 2:
            continue
        # information on the pose as a proxy for corners: the trace of each camera's block, second largest
        tr = sorted((float(np.trace(Hcc)) for _, Hcc, _, _ in b), reverse=True)
        if best is None or tr[1] > best[0]:
            best = (tr[1], t, [cams[ci].tag for ci, *_ in b])
    if best is None:
        return None
    t = best[1]
    return Target(name='TC', x=round(t.x, 3), y=round(t.y, 3), yaw=t.yaw, rows=t.rows, cols=t.cols,
                  square_mm=t.square_mm)


def choose_check_board(project, engine, cams, st: GuideSettings, targets=None) -> dict | None:
    """The floor board to leave out of the solve for the verify stage: seen by two or more cameras (so the test does
    not depend on where it lies), and the one whose absence costs the cameras least (every camera still meeting the
    requirement when possible)."""
    tl = list(project.targets if targets is None else targets)
    nc = len(cams)
    blocks = [_board_blocks(project, engine, cams, t, st) for t in tl]
    terms = [_board_schur(nc, b, st) if b else np.zeros((6 * nc, 6 * nc)) for b in blocks]
    S = sum(terms, np.zeros((6 * nc, 6 * nc)))
    best = None
    for i, (t, b) in enumerate(zip(tl, blocks)):
        if len({ci for ci, *_ in b}) < 2:
            continue
        r, _ = _ratios(S - terms[i], cams, st)
        worst = max(r) if r else math.inf
        if best is None or worst < best['worst']:
            best = {'board': t.name, 'worst': worst, 'seen_by': [cams[ci].tag for ci, *_ in b],
                    'meets': worst <= 1.0}
    return best


def plan_boards(project, engine, st: GuideSettings | None = None, max_add=10, progress=None) -> dict:
    """A board plan for the active variant: floor boards for every camera a floor board can reach (plan_floor), then
    a board on a stand for every camera the floor cannot reach, chained to the floor camera that looks most the same
    way (plan_stand). The project is not changed: apply_plan writes the result."""
    _load_scipy()
    with single_blas():
        return _plan_boards(project, engine, st or GuideSettings(), max_add, progress)


def _plan_boards(project, engine, st, max_add, progress) -> dict:
    q = project.snapshot()
    cams = [c for c in q.variant.cameras if c.visible]
    floor = [c for c in cams if _floor_reach(q, engine, c, min_square_px=st.min_square_px)]
    stand_cams = [c for c in cams if c not in floor]
    fl = plan_floor(q, engine, floor, st, max_add=max_add, progress=progress)
    q.targets = fl['targets']
    if progress:
        progress(0.92, 'Boards on stands')
    cov = planned_floor_covariance(q, engine, floor, st)
    stands, stand_rows = [], []
    for i, c in enumerate(stand_cams):
        par = parent_for(q, engine, c, floor)
        if par is None:
            stand_rows.append({'camera': c.tag, 'ok': False, 'why': 'no floor camera to chain it to'})
            continue
        best = plan_stand(q, engine, par, c, cov['cams'].get(par.uid), st)
        if best is None:
            stand_rows.append({'camera': c.tag, 'parent': par.tag, 'ok': False,
                               'why': f'no place for a board both {par.tag} and {c.tag} see whole'})
            continue
        b = best['board']
        b.name = f'S{i + 1}'
        stands.append(b)
        stand_rows.append({'camera': c.tag, 'parent': par.tag, 'ok': best['ratio'] <= 1.0, 'board': b.name,
                           'sigma': best['sigma'], 'ratio': best['ratio'], 'axis': best['axis'],
                           'distance_m': best['distance_m'], 'square_px': best['square_px']})
    check = choose_check_board(q, engine, floor, st)
    if not (check and check.get('meets')):
        tc = extra_check_board(q, engine, floor, q.targets, fl['candidates'], st)
        if tc is not None:
            names = {t.name for t in q.targets}
            k = len(names) + 1
            while f'T{k}' in names:
                k += 1
            tc.name = f'T{k}'
            q.targets.append(tc)
            seen = [c.tag for c in floor if visible_floor_corners(q, engine, c, targets=[tc])]
            check = {'board': tc.name, 'seen_by': seen, 'meets': True, 'extra': True,
                     'worst': max(fl['ratios'].values()) if fl['ratios'] else None}
    if progress:
        progress(1.0, 'Done')
    rows = []
    for c in floor:
        sig = sigmas(planned_pose(c), cov['cams'].get(c.uid))
        r, axis = ratio(sig, st)
        rows.append({'camera': c.tag, 'kind': 'floor', 'sigma': sig, 'ratio': r, 'axis': axis, 'ok': r <= 1.0,
                     'boards': [t.name for t, _, _ in visible_floor_corners(q, engine, c)]})
    for row in stand_rows:
        rows.append({'camera': row['camera'], 'kind': 'stand', **{k: v for k, v in row.items() if k != 'camera'}})
    return {'targets': [t.to_dict() for t in q.targets], 'stands': [s.to_dict() for s in stands], 'cameras': rows,
            'floor_cameras': [c.tag for c in floor], 'stand_cameras': [c.tag for c in stand_cams], 'log': fl['log'],
            'unused': fl['unused'], 'check': check, 'settings': st.to_dict()}


def apply_plan(project, plan: dict):
    """Write a plan_boards result into the project: its floor targets and stands (project.install_guide), and the
    check board into the guide's settings."""
    project.targets = [Target.from_dict(t) for t in plan['targets']]
    g = dict(getattr(project, 'install_guide', None) or {})
    g['stands'] = [dict(s) for s in plan['stands']]
    st = GuideSettings.from_dict(g.get('settings'))
    chk = plan.get('check')
    st.check_boards = [chk['board']] if chk and chk.get('meets') else []
    g['settings'] = st.to_dict()
    project.install_guide = g


class _NoBody:
    """An engine stand-in without a vehicle (synthetic views of a camera inside the cabin: the windscreen is open)."""
    def vehicle_geom(self, *a, **k):
        return None


class _CubeAt:
    """An engine whose occlusion is looked up from the camera's mount point, whatever position it is asked for: a
    camera a few millimetres off its plan sees the same body parts in its way (and a body model's surface is not
    accurate to millimetres: a camera on a bumper would otherwise drop inside it)."""
    def __init__(self, engine, pos):
        self.engine, self.pos = engine, np.asarray(pos, float)

    def vehicle_geom(self, *a, **k):
        return self.engine.vehicle_geom(*a, **k)

    def cube(self, pos, vg=None):
        return self.engine.cube(self.pos, vg) if vg is not None else self.engine.cube(self.pos)


def render_view(project, engine, cam: Camera, pose=None, w=960, h=None, seed=0, noise=1.5, stands=None):
    """A synthetic frame of what the camera sees of the calibration floor (core.synth.render_capture: the floor, the
    floor boards, the body) with the boards on stands drawn in; a camera inside the cabin sees out through it."""
    from .synth import render_capture
    intr = _intr(project, cam)
    h = h or max(1, round(w * intr.height / intr.width))
    pose = planned_pose(cam) if pose is None else pose
    vg = engine.vehicle_geom(project) if engine is not None else None
    eng = _NoBody() if inside_body(engine, vg, cam.pos()) else _CubeAt(engine, cam.pos())
    img = render_capture(project, eng, cam, w, h, pose=pose, seed=seed, noise=0.0)
    sb = stands_of(project) if stands is None else stands
    if sb:
        img = render_stand_boards(img, intr, pose, sb)
    if noise:
        rng = np.random.default_rng(seed + 1000)
        img = np.clip(img.astype(np.float32) + rng.normal(0, noise, img.shape), 0, 255).astype(np.uint8)
    return img


# =========================================================================== boards in images
PLANE_PX = 40.0             # a square's width in the board-plane view the detector is given


def floor_frame(t: Target):
    """(R board→world, centre) of a floor board: u along the board's x (its yaw), v across it, n up; its inner
    corners at u = i·sq − w/2, v = j·sq − h/2 (core.install.inner_corners)."""
    c, s_ = math.cos(math.radians(t.yaw)), math.sin(math.radians(t.yaw))
    R = np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])
    return R, np.array([t.x, t.y, 0.0])


def detect_on_plane(img, intr, pose, frame, local, size_m, sq_m, pattern, margin_sq=2.0):
    """Find a checkerboard by resampling the image onto the board's own plane (the pose known roughly: planned,
    aimed or solved). Every board then looks square-on and evenly sized to OpenCV's detector, however obliquely and
    far out in the fisheye the camera sees it: a floor board at a grazing angle that a pinhole patch round it shows
    3:1 foreshortened, and a neighbour board, no longer defeat it. A corner found at plane-view pixel (x, y) was
    sampled from image pixel M(x, y), the map that made the view, so that is where it is in the image, exactly,
    whatever the error of the pose the map was made with (it only has to put the board inside the view).

    frame: (R board→world, centre); local: (N, 2) the inner corners' plane coordinates (m), in the order to return
    them; pattern: (corners per row, rows). Returns (image pixels (N, 2) at the image's scale, unit optical rays
    (N, 3)) or None."""
    import cv2
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sx, sy = _scale(intr, g.shape)
    R, c = frame
    w, h = size_m
    k = PLANE_PX / sq_m                                   # plane-view pixels per metre
    mu = w / 2 + margin_sq * sq_m
    mv = h / 2 + margin_sq * sq_m
    nu, nv = int(round(2 * mu * k)), int(round(2 * mv * k))
    us = (np.arange(nu) + 0.5) / k - mu
    vs = (np.arange(nv) + 0.5) / k - mv
    U, V = np.meshgrid(us, vs)
    P = c + U.ravel()[:, None] * R[:, 0] + V.ravel()[:, None] * R[:, 1]
    uv, ok = project_pts(intr, pose, P, (sx, sy))
    d = (P - np.asarray(pose[1])) @ w2o_of(pose[0]).T
    ok = ok & (d[:, 2] > -1e9)
    if ok.mean() < 0.9:
        return None
    mapx = np.where(ok, uv[:, 0] - 0.5, -10).reshape(nv, nu).astype(np.float32)
    mapy = np.where(ok, uv[:, 1] - 0.5, -10).reshape(nv, nu).astype(np.float32)
    view = cv2.remap(g, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=128)
    det = _find(view, pattern)
    if det is None:
        return None
    pred = ((np.asarray(local, float) + [mu, mv]) * k - 0.5)
    rows_, cols_ = pattern[1], pattern[0]
    pred_g = pred.reshape(rows_, cols_, 2)
    # the detector returns the whole grid, so the only question is which way round it lies (8 symmetries); the
    # board may sit up to a couple of squares off the prediction when the pose the view was made with is a degree or
    # two out, and a quarter turn moves the corners by the board's size, so two squares is a safe limit
    m = _match_grid(det, pred_g)
    if m is None or m[0] > 2.0 * PLANE_PX:
        return None
    xy = m[1].reshape(-1, 2)
    # the corners back through the map (bilinear in the plane view, as cv2.remap sampled it)
    fx = np.clip(xy[:, 0], 0, nu - 1.001)
    fy = np.clip(xy[:, 1], 0, nv - 1.001)
    x0, y0 = np.floor(fx).astype(int), np.floor(fy).astype(int)
    ax, ay = fx - x0, fy - y0

    def bil(Mp):
        return ((1 - ax) * (1 - ay) * Mp[y0, x0] + ax * (1 - ay) * Mp[y0, x0 + 1] + (1 - ax) * ay * Mp[y0 + 1, x0]
                + ax * ay * Mp[y0 + 1, x0 + 1])
    pix = np.stack([bil(mapx) + 0.5, bil(mapy) + 0.5], -1)
    rays, _ = intr.unproject(pix[:, 0] / sx, pix[:, 1] / sy)
    rays = rays / np.linalg.norm(rays, axis=1, keepdims=True)
    return pix, rays


def detect_floor(img, project, cam, pose, targets) -> list:
    """core.install.BoardObs for every floor board found by detect_on_plane, at the pose given."""
    from .install import board_outline
    intr = _intr(project, cam)
    sc = _scale(intr, img.shape)
    out = []
    for t in targets:
        W = inner_corners(t)
        flat = W.reshape(-1, 3)
        pred, ok = project_pts(intr, pose, np.vstack([flat, board_outline(t)]), sc)
        if not ok.all():
            continue
        R, c = floor_frame(t)
        local = (flat - c) @ R[:, :2]
        r = detect_on_plane(img, intr, pose, (R, c), local, t.size_m, t.square_mm / 1000, (t.cols - 1, t.rows - 1))
        if r is None:
            continue
        out.append(BoardObs(t.uid, t.name, flat, r[1], r[0], pred[:len(flat)], pred[len(flat):]))
    return out


# =========================================================================== stand boards in images
def render_stand_boards(img, intr, pose, boards, margin_m=0.05, ss=2):
    """Draw boards on stands into a synthetic image (grey or RGB, any scale of the sensor): each sample ray meets the
    board's plane; black and white squares inside, a white border round them; ss × ss samples a pixel (anti-aliased,
    as a lens and a sensor's pixel area average), then a little blur for the optics. The car body is not drawn in
    front of them (the planner keeps them clear of it)."""
    import cv2
    out = np.array(img, copy=True)
    h, w = out.shape[:2]
    sx, sy = w / intr.width, h / intr.height
    R, pos = pose
    O2W = w2o_of(R).T
    acc = np.zeros(h * w)
    cov = np.zeros(h * w)
    offs = (np.arange(ss) + 0.5) / ss
    for oy in offs:
        for ox in offs:
            U, V = np.meshgrid((np.arange(w) + ox) / sx, (np.arange(h) + oy) / sy)
            d_opt, th = intr.unproject(U.ravel(), V.ravel())
            dw = d_opt @ O2W.T
            val = np.full(len(dw), np.nan)
            depth = np.full(len(dw), np.inf)
            for b in boards:
                Rb, c = b.frame()
                nrm = Rb[:, 2]
                den = dw @ nrm
                with np.errstate(divide='ignore', invalid='ignore'):
                    t = ((c - pos) @ nrm) / den
                hit = np.isfinite(t) & (t > 0) & (th <= intr.max_theta)
                loc = (pos + dw * np.where(hit, t, 0.0)[:, None] - c) @ Rb
                bw, bh = b.size_m
                inside = hit & (np.abs(loc[:, 0]) <= bw / 2 + margin_m) & (np.abs(loc[:, 1]) <= bh / 2 + margin_m)
                sq = b.square_mm / 1000
                iu = np.floor((loc[:, 0] + bw / 2) / sq)
                iv = np.floor((bh / 2 - loc[:, 1]) / sq)
                on = inside & (np.abs(loc[:, 0]) <= bw / 2) & (np.abs(loc[:, 1]) <= bh / 2)
                v_ = np.where(on, np.where((iu + iv) % 2 == 0, 25.0, 235.0), 240.0)
                closer = inside & (t < depth)
                val[closer] = v_[closer]
                depth[closer] = t[closer]
            m = np.isfinite(val)
            acc[m] += val[m]
            cov[m] += 1
    if not cov.any():
        return out
    k = cov.reshape(h, w) / (ss * ss)                     # the share of the pixel the boards cover
    lay = np.where(cov > 0, acc / np.maximum(cov, 1), 0).reshape(h, w).astype(np.float32)
    lay = cv2.GaussianBlur(lay, (0, 0), 0.5)
    k = cv2.GaussianBlur(k.astype(np.float32), (0, 0), 0.5)
    base = out.astype(np.float32)
    if base.ndim == 3:
        lay, k = lay[..., None], k[..., None]
    return (base * (1 - k) + lay * k).clip(0, 255).astype(np.uint8)


def _pnp(Pw, rays, init=None):
    """Camera pose from world points and unit optical rays (normalised coordinates, K = I)."""
    import cv2
    norm = rays[:, :2] / rays[:, 2:3]
    Pw = np.asarray(Pw, np.float64)
    if init is not None:
        W2O = w2o_of(init[0])
        rv0, _ = cv2.Rodrigues(W2O)
        tv0 = (-W2O @ np.asarray(init[1], float)).reshape(3, 1)
        ok, rv, tv = cv2.solvePnP(Pw, norm.astype(np.float64), np.eye(3), None, rv0, tv0, useExtrinsicGuess=True,
                                  flags=cv2.SOLVEPNP_ITERATIVE)
    else:
        ok, rv, tv = cv2.solvePnP(Pw, norm.astype(np.float64), np.eye(3), None, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        return None
    rv, tv = cv2.solvePnPRefineLM(Pw, norm.astype(np.float64), np.eye(3), None, rv, tv)
    W2O, _ = cv2.Rodrigues(rv)
    return R_of_w2o(W2O), (-W2O.T @ tv).ravel()


def solve_chain(intr_p, pose_p, cov_p, obs_p, intr_c, obs_c, board: StandBoard, st: GuideSettings,
                init_c=None, scale_p=(1, 1), scale_c=(1, 1)) -> dict:
    """The child's pose through a board on a stand: obs_p / obs_c = (pixels (N,2), rays (N,3)) of the board's inner
    corners in the parent's and the child's image (detect_on_plane), the parent at its solved pose. The board's pose
    from the parent's rays (PnP), the child's from the board (PnP), then both together by least squares on the
    pixels of both images (the parent held). Covariance by chain_covariance at the answer, scaled by the fit's χ²
    when the corners are noisier than σ."""
    from scipy.optimize import least_squares
    L = board.local_corners().reshape(-1, 3)
    # board pose in the parent's optical frame, as a camera-pose problem with the board's frame as the world
    bp = _pnp(L, obs_p[1])
    if bp is None:
        return {'ok': False, 'why': f'the board {board.name} could not be placed from {board.parent}'}
    Rbo_body, pbo = bp                         # pose of the parent camera in the board frame
    W2O_b = w2o_of(Rbo_body)                   # board frame → parent optical
    O2W_p = w2o_of(pose_p[0]).T                # parent optical → world
    Rb = O2W_p @ W2O_b                         # board frame → world
    cb = O2W_p @ (W2O_b @ (-pbo)) + np.asarray(pose_p[1], float)
    board_pose = (Rb, cb)
    Pw = L @ Rb.T + cb
    c0 = _pnp(Pw, obs_c[1], init_c)
    if c0 is None:
        return {'ok': False, 'why': f'{board.child} could not be placed from the board {board.name}'}

    def resid(x):
        bpz = _board_param_pose(board_pose, x[:6])
        cz = perturb(c0, x[6:])
        a = _uv_board(intr_p, pose_p, bpz, L, scale_p) - obs_p[0]
        b = _uv_board(intr_c, cz, bpz, L, scale_c) - obs_c[0]
        return np.concatenate([a.ravel(), b.ravel()]) / st.pixel_sigma
    sol = least_squares(resid, np.zeros(12), method='lm', xtol=1e-12, ftol=1e-12)
    board_pose = _board_param_pose(board_pose, sol.x[:6])
    child = perturb(c0, sol.x[6:])
    r = sol.fun
    stat = float(r @ r)
    dof = len(r) - 12
    test = chi2_test(stat, dof, st.chi2_alpha)
    cc = chain_covariance(intr_p, pose_p, cov_p, intr_c, child, board_pose, L, st.pixel_sigma, scale_p, scale_c)
    cov = cc['child'] * test['scale'] ** 2
    rms_p = float(np.sqrt(np.mean(np.sum((_uv_board(intr_p, pose_p, board_pose, L, scale_p) - obs_p[0]) ** 2, 1))))
    rms_c = float(np.sqrt(np.mean(np.sum((_uv_board(intr_c, child, board_pose, L, scale_c) - obs_c[0]) ** 2, 1))))
    return {'ok': True, 'pose': child, 'cov': cov, 'board_pose': board_pose, 'chi2': test,
            'rms_px': {'parent': rms_p, 'child': rms_c}, 'corners': len(L)}


# =========================================================================== robust coverage
def _boundary(intr, u, v, theta):
    """Margin (px) of each pixel to the nearest image edge (the sensor's four or the image circle) and that edge's
    inward normal."""
    from .lens import r_of_theta
    m = np.stack([u, intr.width - u, v, intr.height - v], -1)
    nrm = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    k = np.argmin(m, axis=1)
    marg = m[np.arange(len(u)), k]
    n = nrm[k]
    rmax = r_of_theta(np.array([intr.max_theta]), intr.f_mm, intr.projection, intr.kb)[0] / intr.pixel_mm
    r = r_of_theta(theta, intr.f_mm, intr.projection, intr.kb) / intr.pixel_mm
    cm = rmax - r
    use = cm < marg
    rad = np.stack([u - intr.cx, (v - intr.cy) / max(intr.aspect, 1e-9)], -1)
    rn = rad / np.maximum(np.linalg.norm(rad, axis=1, keepdims=True), 1e-9)
    n = np.where(use[:, None], -rn, n)
    marg = np.where(use, cm, marg)
    return marg, n


def as_built(variant: Variant, poses: dict, name=None) -> Variant:
    """A copy of the variant with the measured poses (uid -> (R, pos)); cameras without one keep their plan."""
    v = Variant.from_dict(copy.deepcopy(variant.to_dict()))
    v.uid = new_uid()
    v.name = name or f'{variant.name} · as built'
    for c in v.cameras:
        p = poses.get(c.uid)
        if p is None:
            continue
        d = pose_doc(p)
        c.x, c.y, c.z, c.yaw, c.pitch, c.roll = d['x'], d['y'], d['z'], d['yaw'], d['pitch'], d['roll']
    return v


def robust_coverage(project, engine, poses: dict, covs: dict, st: GuideSettings, variant=None) -> dict:
    """Coverage of the measured rig with every ground cell graded (module docstring): robust (some camera sees it with
    a margin of k σ), uncertain (seen, but only within k σ of an image edge) or blind; against the plan's coverage.
    poses / covs: camera uid -> measured (R, pos) / 6×6 covariance (None: the plan's pose, exact)."""
    from scipy import ndimage
    from .analysis import eval_points
    variant = variant or project.variant
    plan = engine.analyse(project, variant)
    if plan is None:
        return {'ok': False, 'why': 'no ground region to analyse'}
    g = plan.grid
    vg = engine.vehicle_geom(project, variant)
    roi = g.roi & ~g.body
    P = g.points(plan.z).reshape(-1, 3)
    robust = np.zeros(roi.shape, np.int16)
    seen = np.zeros(roi.shape, np.int16)
    per = {}
    for cam in [c for c in variant.cameras if c.visible]:
        pose = poses.get(cam.uid) or planned_pose(cam)
        cov = covs.get(cam.uid)
        intr = _intr(project, cam)
        # what the body hides is looked up from the mount point (_CubeAt), the image from the measured pose
        cube = engine.cube(cam.pos(), vg)
        R, pos = pose
        vis, _, _, _ = eval_points(P + (cam.pos() - np.asarray(pos, float)), cam, intr, cube,
                                   R_override=np.asarray(R, float), pos_override=cam.pos())
        vis = vis.reshape(roi.shape) & roi
        seen += vis
        idx = np.flatnonzero(vis.ravel())
        if not len(idx):
            per[cam.tag] = {'seen_m2': 0.0, 'robust_m2': 0.0}
            continue
        Pi = P[idx]
        d = (Pi - pose[1]) @ w2o_of(pose[0]).T
        u, v, th, _ = intr.project_dirs(d)
        marg, n = _boundary(intr, u, v, th)
        if cov is None:
            ok = np.ones(len(idx), bool)
            sn = np.zeros(len(idx))
        else:
            J = proj_jac(intr, pose, Pi)                         # (K, 2, 6)
            Jn = np.einsum('ki,kij->kj', n, J)                   # (K, 6)
            sn = np.sqrt(np.maximum(np.einsum('kj,jl,kl->k', Jn, np.asarray(cov, float), Jn), 0.0))
            ok = marg >= st.k_sigma * sn
        rb = np.zeros(roi.size, bool)
        rb[idx[ok]] = True
        robust += rb.reshape(roi.shape)
        a = g.cell ** 2
        per[cam.tag] = {'seen_m2': float(len(idx) * a), 'robust_m2': float(ok.sum() * a),
                        'median_edge_sigma_px': float(np.median(sn)) if len(sn) else 0.0}
    a = g.cell ** 2
    n_roi = int(roi.sum())
    rob = roi & (robust > 0)
    unc = roi & (robust == 0) & (seen > 0)
    blind = roi & (seen == 0)
    pv = np.zeros(roi.shape, bool)
    for ce in plan.cams:
        pv |= ce.vis
    plan_cov = float((roi & pv).sum() / max(n_roi, 1) * 100)
    holes = []
    lab, nl = ndimage.label(blind)
    X, Y = np.meshgrid(g.xs(), g.ys()) if callable(getattr(g, 'xs', None)) else np.meshgrid(g.xs, g.ys)
    for i in range(1, nl + 1):
        m = lab == i
        area = float(m.sum() * a)
        if area < 0.05:
            continue
        holes.append({'area_m2': area, 'x': float(X[m].mean()), 'y': float(Y[m].mean()),
                      'x_range': [float(X[m].min()), float(X[m].max())],
                      'y_range': [float(Y[m].min()), float(Y[m].max())]})
    holes.sort(key=lambda h: -h['area_m2'])
    status = np.zeros(roi.shape, np.int8)                    # 0 outside, 1 robust, 2 uncertain, 3 blind
    status[rob] = 1
    status[unc] = 2
    status[blind] = 3
    robust_pct = float(rob.sum() / max(n_roi, 1) * 100)
    return {'ok': True, 'robust_pct': robust_pct, 'uncertain_pct': float(unc.sum() / max(n_roi, 1) * 100),
            'blind_pct': float(blind.sum() / max(n_roi, 1) * 100), 'blind_m2': float(blind.sum() * a),
            'plan_covered_pct': plan_cov, 'roi_m2': float(n_roi * a), 'holes': holes[:12], 'per_camera': per,
            'k_sigma': st.k_sigma, 'meets_plan': robust_pct >= plan_cov - st.coverage_tol_pct,
            'grid': {'x0': float(g.x0), 'y0': float(g.y0), 'cell': float(g.cell), 'nx': int(g.nx), 'ny': int(g.ny)},
            'status': status}


def coverage_whatif(project, engine, tag, d_yaw=0.0, d_pitch=0.0, d_roll=0.0, variant=None) -> dict:
    """Coverage of the plan with one camera turned (degrees): how much a turn of the mount gains or loses."""
    from . import observability as O
    variant = variant or project.variant
    v2 = Variant.from_dict(copy.deepcopy(variant.to_dict()))
    cam = next((c for c in v2.cameras if c.tag == tag), None)
    if cam is None:
        return {'ok': False, 'why': f'no camera {tag}'}
    cam.yaw += d_yaw
    cam.pitch += d_pitch
    cam.roll += d_roll
    base = engine.analyse(project, variant)
    new = engine.analyse(O.project_for(project, v2), v2)
    return {'ok': True, 'camera': tag, 'turn_deg': {'yaw': d_yaw, 'pitch': d_pitch, 'roll': d_roll},
            'covered_pct': {'now': float(base.metrics['covered_pct']), 'turned': float(new.metrics['covered_pct'])},
            'blind_m2': {'now': float(base.metrics['blind_m2']), 'turned': float(new.metrics['blind_m2'])}}


def far_field(intr, widths=(('car', 1.8), ('pedestrian', 0.5)), px=20.0) -> dict:
    """The range at which an object of each width is px pixels wide on the optical axis (f_px × width / px)."""
    return {name: float(intr.f_px * w / px) for name, w in widths}


# =========================================================================== verification
def floor_point_cov(intr, pose, cov6, uv, sigma_px, scale=(1, 1)):
    """Floor points of pixels uv (K,2) and their 2×2 covariances from the pose's covariance and the pixel noise."""
    R, pos = pose
    d, _ = intr.unproject(uv[:, 0] / scale[0], uv[:, 1] / scale[1])

    def fp(pz, uvz):
        dd, _ = intr.unproject(uvz[:, 0] / scale[0], uvz[:, 1] / scale[1])
        return floor_points(pz[0], pz[1], dd)
    base = fp(pose, uv)
    K = len(uv)
    Jp = np.zeros((K, 2, 6))
    for k in range(6):
        e = np.zeros(6)
        e[k] = FD
        Jp[:, :, k] = (fp(perturb(pose, e), uv) - fp(perturb(pose, -e), uv)) / (2 * FD)
    Ju = np.zeros((K, 2, 2))
    for k in range(2):
        e = np.zeros(2)
        e[k] = 1e-3
        Ju[:, :, k] = (fp(pose, uv + e) - fp(pose, uv - e)) / 2e-3
    C = np.einsum('kij,jl,kml->kim', Jp, np.zeros((6, 6)) if cov6 is None else cov6, Jp)
    C = C + sigma_px ** 2 * np.einsum('kij,kmj->kim', Ju, Ju)
    return base, C


def cross_check(intr_a, pose_a, cov_a, uv_a, intr_b, pose_b, cov_b, uv_b, sigma_px, scale_a=(1, 1),
                scale_b=(1, 1), cross=None) -> dict:
    """The same floor corners from two cameras: per corner D = floor point(a) − floor point(b), its covariance from
    both poses and both pixels' noise, and d² = Dᵀ Σ_D⁻¹ D against χ²(2). cross: Cov(pose a, pose b) when the two
    were solved together (6×6), else independent."""
    pa, Ca = floor_point_cov(intr_a, pose_a, cov_a, uv_a, sigma_px, scale_a)
    pb, Cb = floor_point_cov(intr_b, pose_b, cov_b, uv_b, sigma_px, scale_b)
    D = pa - pb
    C = Ca + Cb
    if cross is not None:
        # Cov(fa, fb) = Ja X Jbᵀ, counted twice with a minus in Cov(fa − fb)
        def jac(intr, pose, uv, sc):
            K = len(uv)
            J = np.zeros((K, 2, 6))
            for k in range(6):
                e = np.zeros(6)
                e[k] = FD
                d1, _ = intr.unproject(uv[:, 0] / sc[0], uv[:, 1] / sc[1])
                J[:, :, k] = (floor_points(*perturb(pose, e), d1) - floor_points(*perturb(pose, -e), d1)) / (2 * FD)
            return J
        Ja, Jb = jac(intr_a, pose_a, uv_a, scale_a), jac(intr_b, pose_b, uv_b, scale_b)
        X = np.einsum('kij,jl,kml->kim', Ja, cross, Jb)
        C = C - X - np.transpose(X, (0, 2, 1))
    d2 = np.einsum('ki,kij,kj->k', D, np.linalg.inv(C + np.eye(2) * 1e-12), D)
    lim = float(-2 * math.log(0.01))                     # χ²(2) 99 %: 9.21
    pass_share = float((d2 <= lim).mean()) if len(d2) else 0.0
    return {'corners': int(len(D)), 'median_mm': float(np.median(np.linalg.norm(D, axis=1)) * 1000) if len(D) else None,
            'max_mm': float(np.max(np.linalg.norm(D, axis=1)) * 1000) if len(D) else None,
            'predicted_mm': float(np.median(np.sqrt(np.trace(C, axis1=1, axis2=2))) * 1000) if len(D) else None,
            'median_d2': float(np.median(d2)) if len(d2) else None, 'limit_d2': lim, 'pass_share': pass_share,
            'pass': pass_share >= 0.95}


# =========================================================================== the session
def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items() if k not in ('pose', 'cov', 'board_pose', 'used')}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return round(v, 4) if math.isfinite(v) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return None
    return o


class GuideSession:
    """One installation, stage by stage (module docstring). Thread use: the page feeds frames from a worker thread;
    every method takes the session's lock, and state() hands out a JSON-safe copy."""

    def __init__(self, project, engine=None, settings: GuideSettings | None = None):
        from .analysis import Engine
        self.project = project.snapshot()
        self.project.install_guide = copy.deepcopy(getattr(project, 'install_guide', None) or {})
        self.engine = engine or Engine()
        self.st = settings or GuideSettings.from_dict(self.project.install_guide.get('settings'))
        self.variant = self.project.variant
        self.cams = [c for c in self.variant.cameras if c.visible]
        self.by_tag = {c.tag: c for c in self.cams}
        self.stands = stands_of(self.project)
        self.lock = threading.RLock()
        self.stage = 'connect'
        self.events: list = []
        self.health: dict = {}
        self.nominal_fps: dict = {}           # tag -> the rate a stream is judged against (the module's when absent)
        self.ident = {'active': False, 'order': [c.tag for c in self.cams], 'done': {}, 'baseline': {},
                      'asking': None, 'why': '', 'remap': {}}
        self.lens_accept = False
        self.aim: dict = {}
        self.aim_models: dict = {}
        self.det: dict = {c.uid: {} for c in self.cams}         # uid -> board uid -> list of (pixels, rays, shape)
        self.det_stand: dict = {c.uid: {} for c in self.cams}   # uid -> stand uid -> list of (pixels, rays, shape)
        self.shapes: dict = {}
        self.solution: dict = {}
        self.cover: dict = {}
        self.verification: dict = {}
        self.log('start', f'{len(self.cams)} cameras in {self.variant.name}')

    # ------------------------------------------------------------------ bookkeeping
    def log(self, kind, text):
        self.events.append({'t': time.time(), 'kind': kind, 'text': text})
        del self.events[:-200]

    def cam(self, tag):
        return self.by_tag.get(tag)

    def floor_targets(self):
        chk = set(self.st.check_boards)
        return [t for t in self.project.targets if t.name not in chk]

    def check_targets(self):
        chk = set(self.st.check_boards)
        return [t for t in self.project.targets if t.name in chk]

    def floor_cams(self):
        stand_children = {s.child for s in self.stands}
        return [c for c in self.cams if c.tag not in stand_children]

    def current_pose(self, cam):
        """Best pose known now: solved, else aimed, else planned."""
        s = self.solution.get('poses', {}).get(cam.uid)
        if s is not None:
            return s
        a = self.aim.get(cam.tag)
        if a and a.get('ok') and 'pose' in a:
            return a['pose']
        return planned_pose(cam)

    # ------------------------------------------------------------------ stage 1
    def update_health(self, snap: dict):
        with self.lock:
            self.health = snap or {}

    def connect_status(self) -> dict:
        cams = (self.health or {}).get('cameras', {})
        rows, bad = [], []
        for c in self.cams:
            h = cams.get(c.tag, {})
            fps = h.get('fps')
            nominal = float(self.nominal_fps.get(c.tag) or self.project.module_for(c).fps or 30.0)
            ok = bool(h.get('frames')) and fps is not None and fps >= self.st.nominal_fraction * nominal and \
                h.get('status') in ('ok', 'warning')
            rows.append({'camera': c.tag, 'fps': fps, 'nominal_fps': nominal, 'status': h.get('status', 'off'),
                         'reasons': h.get('reasons', []), 'ok': ok})
            if not ok:
                bad.append(c.tag)
        rig = (self.health or {}).get('rig', {})
        return {'cameras': rows, 'ok': not bad and bool(rows), 'waiting_for': bad,
                'sync_skew_ms': rig.get('sync_skew_ms'), 'sync_status': rig.get('sync_status')}

    # ------------------------------------------------------------------ stage 2
    def identify_start(self, brightness: dict):
        """brightness: stream key (camera tag whose stream it is now) -> mean. The installer is then asked to cover
        each camera in turn."""
        with self.lock:
            self.ident.update(active=True, baseline=dict(brightness), done={}, remap={},
                              asking=self.ident['order'][0] if self.ident['order'] else None, why='')
            self.log('identify', f'cover the lens of {self.ident["asking"]}')

    def identify_feed(self, brightness: dict) -> dict:
        """Feed the current brightness of every stream while the installer covers the camera asked for. When exactly
        one stream went dark it is that camera's: recorded, and the next camera asked for."""
        with self.lock:
            idn = self.ident
            if not idn['active'] or idn['asking'] is None:
                return self.identify_status()
            key, why = identify_covered(idn['baseline'], brightness)
            idn['why'] = why
            if key is not None and key not in idn['done'].values():
                tag = idn['asking']
                idn['done'][tag] = key
                if key != tag:
                    idn['remap'][tag] = key
                    self.log('identify', f'{tag} is wired to the stream configured as {key}')
                else:
                    self.log('identify', f'{tag} is on its own port')
                rest = [t for t in idn['order'] if t not in idn['done']]
                idn['asking'] = rest[0] if rest else None
                if idn['asking'] is None:
                    idn['active'] = False
            return self.identify_status()

    def identify_skip(self):
        """The ports are known to be right (each camera labelled at its port)."""
        with self.lock:
            self.ident.update(active=False, asking=None, done={t: t for t in self.ident['order']}, remap={},
                              why='taken as wired to plan')
            self.log('identify', 'ports taken as wired to the plan')

    def identify_status(self) -> dict:
        idn = self.ident
        return {'asking': idn['asking'], 'done': dict(idn['done']), 'remap': dict(idn['remap']), 'why': idn['why'],
                'ok': len(idn['done']) == len(idn['order']) and not idn['active']}

    def stream_remap(self, streams: dict) -> dict:
        """The project's stream table corrected by the identification: camera tag -> the spec of the stream that went
        dark when that camera was covered."""
        out = dict(streams)
        for tag, key in self.ident['done'].items():
            if key in streams:
                out[tag] = streams[key]
        return out

    # ------------------------------------------------------------------ stage 3
    def lens_rows(self) -> list:
        return [lens_status(self.project, c) for c in self.cams]

    def lens_ok(self) -> bool:
        rows = self.lens_rows()
        return all(r['ok'] for r in rows) or (self.lens_accept and not any(r['problems'] for r in rows))

    # ------------------------------------------------------------------ stage 4
    def aim_camera(self, tag, frame) -> dict:
        cam = self.cam(tag)
        if cam is None:
            return {}
        am = self.aim_models.get(tag)
        if am is None:
            am = aim_model(self.project, self.engine, cam)
            self.aim_models[tag] = am
        a = estimate_aim(frame, am, _intr(self.project, cam))
        a['instructions'] = aim_instructions(a, self.st)
        a['within'] = bool(a.get('ok')) and all(
            abs(a[f'd_{k}_deg']) <= self.st.aim_tol_deg for k in ('yaw', 'pitch', 'roll') if a['determined'][k])
        with self.lock:
            self.aim[tag] = a
        return a

    def aim_status(self) -> dict:
        rows = []
        for c in self.cams:
            a = self.aim.get(c.tag)
            rows.append({'camera': c.tag, 'measured': bool(a and a.get('ok')), 'within': bool(a and a.get('within')),
                         'd_yaw_deg': a.get('d_yaw_deg') if a else None, 'd_pitch_deg': a.get('d_pitch_deg') if a
                         else None, 'd_roll_deg': a.get('d_roll_deg') if a else None,
                         'determined': a.get('determined') if a else None,
                         'instructions': a.get('instructions', []) if a else [], 'why': a.get('why', '') if a else ''})
        # a camera the body outline cannot aim (none in view) is left to the boards
        ok = all(r['within'] or (r['why'] and not r['measured']) for r in rows) and any(r['measured'] for r in rows)
        return {'cameras': rows, 'ok': ok}

    # ------------------------------------------------------------------ stage 5
    def feed(self, tag, frame) -> dict:
        """Find the camera's planned boards (floor and stand) in a frame, at the best pose known; keep the last
        frames_per_board detections of each."""
        cam = self.cam(tag)
        if cam is None or frame is None:
            return {}
        intr = _intr(self.project, cam)
        pose = self.current_pose(cam)
        obs = detect_floor(frame, self.project, cam, pose, self.project.targets)
        found = {'floor': [], 'stand': []}
        with self.lock:
            self.shapes[cam.uid] = frame.shape[:2]
            for o in obs:
                lst = self.det[cam.uid].setdefault(o.target_uid, [])
                lst.append(o)
                del lst[:-self.st.frames_per_board]
                found['floor'].append(o.name)
        for b in self.stands:
            if tag not in (b.parent, b.child):
                continue
            loc = b.local_corners()[..., :2].reshape(-1, 2)
            r = detect_on_plane(frame, intr, pose, b.frame(), loc, b.size_m, b.square_mm / 1000,
                                (b.cols - 1, b.rows - 1))
            if r is None:
                continue
            with self.lock:
                lst = self.det_stand[cam.uid].setdefault(b.uid, [])
                lst.append((r[0], r[1], frame.shape[:2]))
                del lst[:-self.st.frames_per_board]
            found['stand'].append(b.name)
        return found

    def _averaged(self, uid) -> list:
        """Each board's detections averaged over the frames kept (the board does not move between them)."""
        out = []
        for buid, lst in self.det.get(uid, {}).items():
            if not lst:
                continue
            o = lst[-1]
            pts = np.mean([x.image_pts for x in lst if x.image_pts.shape == o.image_pts.shape], axis=0)
            rays = np.mean([x.rays for x in lst if x.rays.shape == o.rays.shape], axis=0)
            rays /= np.linalg.norm(rays, axis=1, keepdims=True)
            out.append(BoardObs(o.target_uid, o.name, o.world, rays, pts, o.predicted, o.outline, o.picked))
        return out

    def _stand_avg(self, uid, suid):
        lst = self.det_stand.get(uid, {}).get(suid) or []
        if not lst:
            return None
        pts = np.mean([x[0] for x in lst], axis=0)
        rays = np.mean([x[1] for x in lst], axis=0)
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)
        return pts, rays, lst[-1][2], len(lst)

    def expected_boards(self) -> dict:
        """Which planned boards each camera should see (at its plan): tag -> [board names]. Computed once."""
        if getattr(self, '_expected', None) is not None:
            return self._expected
        out = {}
        for c in self.cams:
            names = [t.name for t, _, _ in visible_floor_corners(self.project, self.engine, c,
                                                                 min_square_px=self.st.min_square_px)]
            names += [b.name for b in self.stands if c.tag in (b.parent, b.child)]
            out[c.tag] = names
        self._expected = out
        return out

    def boards_status(self, compute=False) -> dict:
        """Boards expected and found per camera. The expected ones take a few seconds the first time (every board
        through every camera, the body in the way): without compute, before they are known, only what was found."""
        exp = self.expected_boards() if (compute or getattr(self, '_expected', None) is not None) else {}
        rows = []
        for c in self.cams:
            have = {o[-1].name for o in self.det[c.uid].values() if o}
            have |= {b.name for b in self.stands if self.det_stand[c.uid].get(b.uid)}
            need = exp.get(c.tag, [])
            rows.append({'camera': c.tag, 'expected': need, 'found': sorted(have),
                         'missing': [n for n in need if n not in have]})
        stands_ok = all(self.det_stand[self.cam(t).uid].get(b.uid) for b in self.stands for t in (b.parent, b.child)
                        if self.cam(t) is not None)
        return {'cameras': rows, 'ok': bool(rows) and all(r['found'] for r in rows) and stands_ok}

    # ------------------------------------------------------------------ stage 6
    def solve(self) -> dict:
        """Floor cameras jointly (core.joint, the check boards left out), their covariance at the answer, then each
        stand-board child through its parent."""
        _load_scipy()
        with single_blas():
            return self._solve()

    def _solve(self) -> dict:
        with self.lock:
            st = self.st
            fcams = self.floor_cams()
            chk = {t.uid for t in self.check_targets()}
            obs = {c.uid: [o for o in self._averaged(c.uid) if o.target_uid not in chk] for c in fcams}
            init = {c.uid: self.current_pose(c) for c in fcams}
            shapes = dict(self.shapes)
        solved = [c for c in fcams if obs.get(c.uid)]
        out = {'ok': False, 'poses': {}, 'covs': {}, 'cameras': [], 'chains': [], 'why': ''}
        if not solved:
            out['why'] = 'no floor camera has found its boards yet'
            with self.lock:
                self.solution = out
            return out
        jr = joint_solve(self.project, solved, obs, init, shapes, st.board_sigma_mm, st.board_sigma_deg,
                         pixel_sigma=st.pixel_sigma)
        if not jr.poses:
            out['why'] = jr.message
            with self.lock:
                self.solution = out
            return out
        blocks = floor_blocks(self.project, solved, jr.used, jr.poses, shapes)
        jc = joint_covariance(self.project, solved, blocks, jr.poses, shapes, st)
        # χ² of the floor fit: pixel residuals at the answer (the board moves are in the prior)
        r2, n_res = 0.0, 0
        for c in solved:
            intr = _intr(self.project, c)
            sc = _scale(intr, shapes[c.uid])
            for o in jr.used.get(c.uid, []):
                fix = next((b for b in jr.boards if b.uid == o.target_uid), None)
                W = np.asarray(o.world, float)
                if fix is not None:
                    t = next(t for t in self.project.targets if t.uid == o.target_uid)
                    ctr = np.array([t.x, t.y, 0.0])
                    a = math.radians(fix.dyaw_deg)
                    Rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
                    W = (W - ctr) @ Rz.T + ctr + np.array([fix.dx_mm, fix.dy_mm, 0.0]) / 1000
                uv, _ = project_pts(intr, jr.poses[c.uid], W, sc)
                r2 += float(np.sum((uv - o.image_pts) ** 2)) / st.pixel_sigma ** 2
                n_res += 2 * len(W)
        dof = max(1, n_res - 6 * len(solved) - 3 * len(jc['boards']))
        test = chi2_test(r2, dof, st.chi2_alpha)
        s2 = test['scale'] ** 2
        for c in solved:
            cov = jc['cams'].get(c.uid)
            cov = None if cov is None else cov * s2
            out['poses'][c.uid] = jr.poses[c.uid]
            out['covs'][c.uid] = cov
        out['floor_chi2'] = test
        out['joint'] = {'message': jr.message, 'seam_p90_mm': jr.seam_after_p90_mm, 'seam_median_mm':
                        jr.seam_after_mm, 'shared_corners': jr.shared_corners,
                        'moved': [{'board': b.name, 'shift_mm': b.shift_mm, 'dyaw_deg': b.dyaw_deg} for b in jr.moved],
                        'rejected': [{'camera': t, 'board': b, 'rms_px': e} for t, b, e in jr.rejected],
                        'rms_px': {self.cams_by_uid(u).tag: v for u, v in jr.rms_px.items()}}
        out['used'] = jr.used
        # chains
        for b in self.stands:
            par, ch = self.cam(b.parent), self.cam(b.child)
            if par is None or ch is None:
                continue
            row = {'board': b.name, 'parent': b.parent, 'child': b.child, 'ok': False}
            if par.uid not in out['poses']:
                row['why'] = f'{b.parent} is not solved yet'
                out['chains'].append(row)
                continue
            ap, ac = self._stand_avg(par.uid, b.uid), self._stand_avg(ch.uid, b.uid)
            if ap is None or ac is None:
                row['why'] = f'{b.name} not found in ' + ' and '.join(t for t, a in ((b.parent, ap), (b.child, ac))
                                                                       if a is None)
                out['chains'].append(row)
                continue
            ip, ic = _intr(self.project, par), _intr(self.project, ch)
            r = solve_chain(ip, out['poses'][par.uid], out['covs'][par.uid], (ap[0], ap[1]), ic, (ac[0], ac[1]), b,
                            st, init_c=self.current_pose(ch), scale_p=_scale(ip, ap[2]), scale_c=_scale(ic, ac[2]))
            if not r['ok']:
                row['why'] = r['why']
                out['chains'].append(row)
                continue
            out['poses'][ch.uid] = r['pose']
            out['covs'][ch.uid] = r['cov']
            row.update(ok=True, chi2=r['chi2'], rms_px=r['rms_px'], corners=r['corners'])
            out['chains'].append(row)
        for c in self.cams:
            pose = out['poses'].get(c.uid)
            if pose is None:
                out['cameras'].append({'camera': c.tag, 'solved': False})
                continue
            sig = sigmas(pose, out['covs'].get(c.uid))
            rr, axis = ratio(sig, st)
            out['cameras'].append({'camera': c.tag, 'solved': True, 'pose': pose_doc(pose),
                                   'delta': pose_delta(pose, c), 'sigma': sig, 'ratio': rr, 'axis': axis,
                                   'meets': rr <= 1.0})
        out['ok'] = all(r.get('solved') and r.get('meets') for r in out['cameras']) and test['p'] >= st.chi2_alpha \
            and all(ch.get('ok') for ch in out['chains'])
        if not out['ok']:
            why = [f"{r['camera']} not solved" for r in out['cameras'] if not r.get('solved')]
            why += [f"{r['camera']}: {r['axis']} {r['ratio']:.1f}× the requirement" for r in out['cameras']
                    if r.get('solved') and not r['meets']]
            if test['p'] < st.chi2_alpha:
                why.append(f"the corners fit {test['scale']:.1f}× worse than {st.pixel_sigma:g} px noise "
                           '(blur, a bent or wrongly printed board, or a lens model off)')
            out['why'] = '; '.join(why)
        with self.lock:
            self.solution = out
            self.log('solve', 'all cameras measured within the requirement' if out['ok'] else out['why'])
        return out

    def cams_by_uid(self, uid):
        return next(c for c in self.cams if c.uid == uid)

    # ------------------------------------------------------------------ stage 7
    def coverage(self) -> dict:
        with single_blas():
            return self._coverage()

    def _coverage(self) -> dict:
        sol = self.solution
        if not sol.get('poses'):
            return {'ok': False, 'why': 'measure the poses first'}
        cov = robust_coverage(self.project, self.engine, sol['poses'], sol['covs'], self.st, self.variant)
        far = {}
        for c in self.cams:
            lens = self.project.nominal_lens_for(c)
            if getattr(lens, 'label', '') == 'narrow':
                far[c.tag] = far_field(_intr(self.project, c))
        cov['far_field_m'] = far
        with self.lock:
            self.cover = cov
            self.log('coverage', f"robust {cov.get('robust_pct', 0):.1f}% of the ground region "
                                 f"(plan {cov.get('plan_covered_pct', 0):.1f}%)" if cov.get('ok') else cov.get('why'))
        return cov

    # ------------------------------------------------------------------ stage 8
    def verify(self) -> dict:
        sol = self.solution
        st = self.st
        out = {'tests': [], 'ok': False}
        if not sol.get('poses'):
            out['why'] = 'measure the poses first'
            return out
        # 1. check boards seen by two solved cameras: floor points must agree
        for t in self.check_targets():
            seen = []
            for c in self.cams:
                lst = self.det.get(c.uid, {}).get(t.uid)
                if lst and c.uid in sol['poses']:
                    o = self._averaged(c.uid)
                    o = next(x for x in o if x.target_uid == t.uid)
                    seen.append((c, o))
            if len(seen) >= 2:
                (ca, oa), (cb, ob) = seen[:2]
                ia, ib = _intr(self.project, ca), _intr(self.project, cb)
                r = cross_check(ia, sol['poses'][ca.uid], sol['covs'].get(ca.uid), oa.image_pts, ib,
                                sol['poses'][cb.uid], sol['covs'].get(cb.uid), ob.image_pts, st.pixel_sigma,
                                _scale(ia, self.shapes[ca.uid]), _scale(ib, self.shapes[cb.uid]))
                out['tests'].append({'test': f'check board {t.name} from {ca.tag} and {cb.tag}', **r})
            elif len(seen) == 1:
                c, o = seen[0]
                ii = _intr(self.project, c)
                pts, C = floor_point_cov(ii, sol['poses'][c.uid], sol['covs'].get(c.uid), o.image_pts, st.pixel_sigma,
                                         _scale(ii, self.shapes[c.uid]))
                W = np.asarray(o.world)[:, :2]
                C = C + np.eye(2) * (st.board_sigma_mm / 1000) ** 2
                D = pts - W
                d2 = np.einsum('ki,kij,kj->k', D, np.linalg.inv(C), D)
                lim = float(-2 * math.log(0.01))
                share = float((d2 <= lim).mean())
                out['tests'].append({'test': f'check board {t.name} from {c.tag} against its laid place',
                                     'corners': len(D), 'median_mm': float(np.median(np.linalg.norm(D, axis=1)) * 1000),
                                     'pass_share': share, 'pass': share >= 0.95})
        # 2. seams
        p90 = (sol.get('joint') or {}).get('seam_p90_mm')
        if p90 is not None and math.isfinite(p90):
            out['tests'].append({'test': 'seams between floor cameras', 'p90_mm': p90, 'limit_mm': st.seam_tol_mm,
                                 'pass': p90 <= st.seam_tol_mm})
        # 3. residuals
        fc = sol.get('floor_chi2')
        if fc:
            out['tests'].append({'test': 'floor fit residuals (χ²)', **fc})
        for ch in sol.get('chains', []):
            if ch.get('ok'):
                out['tests'].append({'test': f"chain {ch['parent']}→{ch['child']} residuals (χ²)", **ch['chi2']})
        # 4. coverage
        if self.cover.get('ok'):
            out['tests'].append({'test': 'robust coverage against the plan', 'robust_pct': self.cover['robust_pct'],
                                 'plan_pct': self.cover['plan_covered_pct'], 'pass': self.cover['meets_plan']})
        for t in self.check_targets():
            if not any(x['test'].startswith(f'check board {t.name} ') for x in out['tests']):
                out['tests'].append({'test': f'check board {t.name}', 'pass': False,
                                     'why': f'{t.name} was not found in any solved camera: is it laid where the plan '
                                            'says, flat and unshaded?'})
        if not any(t['test'].startswith('check board') for t in out['tests']):
            out['note'] = ('No check board: name one floor board as a check board (left out of the solve) where two '
                           'cameras see it, for a test independent of the solve.')
        out['ok'] = bool(out['tests']) and all(t['pass'] for t in out['tests'])
        with self.lock:
            self.verification = out
            self.log('verify', 'verified' if out['ok'] else 'failed: ' + ', '.join(t['test'] for t in out['tests']
                                                                              if not t['pass']))
        return out

    # ------------------------------------------------------------------ summary
    def stage_rows(self) -> list:
        con = self.connect_status()
        idn = self.identify_status()
        aim = self.aim_status()
        brd = self.boards_status()
        res = {'connect': con['ok'], 'identify': idn['ok'], 'lens': self.lens_ok(), 'aim': aim['ok'],
               'boards': brd['ok'], 'solve': bool(self.solution.get('ok')),
               'coverage': bool(self.cover.get('ok') and self.cover.get('meets_plan')),
               'verify': bool(self.verification.get('ok'))}
        return [{'stage': s, 'title': STAGE_TITLES[s], 'done': res[s], 'current': s == self.stage} for s in STAGES]

    def next_stage(self):
        """The first stage not done."""
        for r in self.stage_rows():
            if not r['done']:
                return r['stage']
        return 'verify'

    def set_stage(self, s):
        if s in STAGES:
            with self.lock:
                self.stage = s

    def state(self) -> dict:
        """Everything the page and the coach read, JSON-safe."""
        with self.lock:
            sol = self.solution
            doc = {'schema': SCHEMA, 'project': self.project.name, 'variant': self.variant.name,
                   'stage': self.stage, 'stages': self.stage_rows(), 'settings': self.st.to_dict(),
                   'connect': self.connect_status(), 'identify': self.identify_status(),
                   'lens': {'cameras': self.lens_rows(), 'accept_library': self.lens_accept, 'ok': self.lens_ok()},
                   'aim': self.aim_status(), 'boards': self.boards_status(),
                   'stands': [s.to_dict() for s in self.stands],
                   'solve': {k: v for k, v in sol.items() if k not in ('poses', 'covs', 'used')},
                   'coverage': {k: v for k, v in self.cover.items() if k != 'status'},
                   'verify': self.verification, 'events': self.events[-20:]}
        return _jsonable(doc)

    def report(self) -> dict:
        doc = self.state()
        doc['time'] = time.strftime('%Y-%m-%dT%H:%M:%S')
        doc['method'] = {'poses': 'joint least squares of the floor cameras and the floor boards (core.joint), the '
                                  'cameras the floor cannot reach chained through boards on stands; covariance '
                                  '(JᵀJ/σ² + P)⁻¹ at the answer, inflated by the fit\'s χ² when the corners are '
                                  'noisier than σ',
                         'coverage': 'cells robust when a camera sees them with a margin of k σ to its image edge '
                                     '(σ the pixel spread along the edge normal from the pose covariance)',
                         'verify': 'check boards left out of the solve; seams; χ² of the residuals'}
        return doc

    def as_built(self, name=None) -> Variant:
        return as_built(self.variant, self.solution.get('poses', {}), name or
                        f'{self.variant.name} · as built {time.strftime("%Y-%m-%d")}')
