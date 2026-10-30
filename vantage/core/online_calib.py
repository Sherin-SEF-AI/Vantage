"""Online self-calibration: the time-varying extrinsics T_cam^lidar(t) of every camera over a whole recorded drive, from
the drive's own content (no targets), with the permanent steps (a knock) told from the reversible flex.

Input: a recorded drive as the drive-log check reads it (core.drivelog: a ROS 2 bag or a nuScenes folder; topics mapped
to the variant's sensors as that check maps them, with its mapping edits and its camera / lidar stamp conventions).
One streaming pass (core.drivelog.scan_log) keeps, at instants spread evenly over the drive, one image of every camera
and the lidar sweeps around it; the instants are frames_per_window per window length (at most max_frames over the
drive: a long drive is sampled more thinly, which the notes say).

Windows. The drive is cut into windows of window_s seconds every stride_s seconds (they overlap when the stride is
shorter; the information of a window is then scaled by stride / window so a frame shared by two windows is not counted
twice). In each window, three kinds of measurement, each with its covariance and the degrees of freedom it constrains:
1. Depth edges against image edges (core.drivelog's targetless rotation check, its search, standard error and
   observability, run on the window's frames alone): the camera's rotation relative to the lidar, the yaw, pitch and
   roll the edges in view pin down (an axis whose score peak is flat is not constrained). Translation is not observable
   this way at road distances.
2. The own vehicle's outline (core.asbuilt: the static ego mask of the window's frames against the simulated one, the
   robust outline fit and its cluster-robust standard error): the camera's rotation (and, when fit_translation is on,
   its position) relative to the vehicle body, on the axes the outline in view constrains. It needs at least four
   frames in the window and a moving vehicle (the static mask is what stays still while the world moves).
3. Ground-plane consistency between two cameras that overlap on the ground: the lidar's ground plane in the window
   (RANSAC over the swept points near the road, z = 0 without a lidar), each camera's image warped onto it through its
   lens model and planned pose (inverse perspective mapping), the second camera's grid moved by the vehicle's motion
   between the two exposures (odometry; the IMU's turning; GNSS speed along x; else none), patches of the first warped
   image found in the second by normalised cross-correlation (cells where the lidar sees something standing above the
   ground left out). A displacement Δ at ground point g is Δ = J_A δ_A − J_B δ_B (J the ground point's motion per
   degree of each camera's yaw, pitch and roll, from the lens model and the plane): a binary factor on the two cameras'
   rotations relative to the lidar's ground plane. Which combinations of axes it pins down follows from its information
   matrix and is stated per window.
A window whose measurements constrain no axis of a sensor says so, with the reason (no depth edges, no outline, no
ground overlap). Only the depth edges are on by default: on the CARLA drives (another car than the project's body,
fisheye images remapped from CARLA's cameras) the outline read −2.3 … +3.1° and the ground factor a mirror camera's yaw
1–1.8° off, window after window, where the edges read a front camera's injected 0.5° pitch as +0.19 … +0.34°; on Vantage's
own synthetic drives (the project's body, the exact lens model) both are sound and switch on.

Estimator. State per window: per camera its rotation correction (yaw, pitch, roll added to the plan's, deg) relative
to the lidar (and its position correction, m, when fit_translation is on), and the lidar's own rotation relative to the
vehicle body (constrained only where an outline and the edges of the same camera meet). Measurement factors are linear
in these small corrections (drivelog and as-built report their corrections in the same Euler convention; the outline's
body-relative correction is the camera's lidar-relative one plus the lidar's, mapped through the two rotations'
Jacobians) and robust (Huber weights on each factor's whitened residual, iteratively reweighted). Between windows a
random walk (walk_deg_per_min, its variance growing with the time between window centres) links the states; the chain
is solved exactly by a Kalman filter and Rauch-Tung-Striebel smoother (the block-tridiagonal normal equations of the
sliding-window least squares). A weak prior (PRIOR_DEG, PRIOR_M) holds a state nothing constrains at the plan.

Slip and flex. Each sensor axis's per-window estimates (each window solved on its own, with its σ) are explained by
models chosen with the Bayesian information criterion: a constant, plus steps (a level change at a window boundary:
two parameters, the size and its location), plus linear terms in the vehicle's speed, longitudinal acceleration (from
the speed, so braking is negative), yaw rate (odometry, or the IMU's z rate) and the time of day (UTC hours, a proxy for
temperature: only over a drive long enough to span TOD_SPAN_H). Terms are added greedily (a pair of steps at once when
no single one pays: a shift that comes back is invisible to one step at a time) while one lowers the BIC by more than
bic_margin (6: "strong" evidence after Kass and Raftery 1995); the residual scale is estimated (the stated σ
give the relative weights; the scale is kept from falling below half of them), observations are Huber-weighted, and with
overlapping windows the effective number of observations is scaled by stride / window. Why BIC over piecewise models
rather than CUSUM: CUSUM needs the pre-change level and a drift parameter fixed in advance and finds one change at a
time; the BIC search fits the unknown levels, takes the per-window σ as weights, finds several changes and, in the
same comparison, whether a regression on speed, acceleration, yaw rate or time explains the variation with fewer
parameters than steps would (a flex that follows braking is one parameter; the steps that would mimic it are many).
A step's time is refined from the depth-edge frames' own optima around it (the frame spacing is its resolution). A
step followed by one back to the earlier level is a temporary excursion; a step that stays is a knock (slip). A
significant regression term is flex: reversible, it returns to the baseline when the speed, acceleration or yaw rate
does. Every event comes with its size, σ and time. The smoother is then run again with the random walk released at
each step and the flex terms added, so a step stays sharp and the flex shows in the timeline.

Exports. JSON: the piecewise extrinsics (one pose per segment between steps, with σ: the pose at rest, flex terms at
zero) and the interpolated series (smoothed, at every window centre, with σ; linear between centres, a jump at a step).
nuScenes (a nuScenes input only): a new folder with the tables copied, one calibrated_sensor row per camera segment
(the input's row with the segment's correction applied in the vehicle frame) and each camera sample_data re-pointed to
its segment's row; the data files are linked, never modified. ROS 2: a sidecar MCAP bag with /tf (base_link → each
estimated camera's <tag>_link at tf_hz, two stamps 1 ms apart at a step) and /tf_static (everything else of the plan).

Frames: the vehicle frame (ISO 8855, rear axle); angles in degrees, intrinsic Z-Y-X, pitch positive down; a camera's
correction is relative to the lidar (the lidar is the reference, as in the drive-log check); times in seconds after the
drive's first message.
"""
from __future__ import annotations

import copy
import csv
import dataclasses
import hashlib
import io
import json
import math
import os
import shutil
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from . import drivelog as DL
from .mathutil import BODY_TO_OPTICAL, euler_zyx, quat_from_matrix, rot_zyx
from .pngtext import put_text

SCHEMA = 'vantage.online_calib/1'
EXTRINSICS_SCHEMA = 'vantage.online_calib.extrinsics/1'
METHOD_REVISION = 1
AXES = ('yaw', 'pitch', 'roll')
TAXES = ('x', 'y', 'z')
VERB = {'yaw': 'yawed', 'pitch': 'pitched', 'roll': 'rolled', 'x': 'shifted in x', 'y': 'shifted in y',
        'z': 'shifted in z'}
KINDS = ('edges', 'outline', 'ground')
KIND_TEXT = {'edges': 'depth edges against image edges', 'outline': 'the own vehicle\'s outline',
             'ground': 'ground-plane consistency between overlapping cameras'}
COVARIATES = ('speed', 'accel', 'yaw_rate', 'time_of_day')
COV_TEXT = {'speed': 'speed', 'accel': 'braking and acceleration', 'yaw_rate': 'yaw rate (turning)',
            'time_of_day': 'the time of day (a temperature proxy)'}
COV_UNIT = {'speed': 'm/s', 'accel': 'm/s²', 'yaw_rate': '°/s', 'time_of_day': 'h'}
COV_MIN_STD = {'speed': 0.5, 'accel': 0.15, 'yaw_rate': 1.0, 'time_of_day': 0.0}
TOD_SPAN_H = 0.25           # the time of day is a candidate only over a drive at least this long (h)
PRIOR_DEG = 3.0             # the weak prior holding an unconstrained rotation at the plan (1 σ, deg)
PRIOR_M = 0.1               # … a position (m)
CONSTRAINED = 0.5           # a window constrains an axis when its σ there is under this share of the prior's
SCALE_FLOOR = 0.5           # the model's residual scale is kept from falling below this share of the stated σ
MIN_SEG = 3                 # windows on either side of a step
SIGMA_SCENE = 0.03          # deg: added (in quadrature) to a window's σ in the models: what the scene in view
                            # moves the measurements by from window to window (heuristic, from the CARLA drives)
MAX_TERMS = 6               # terms a model may add to its constant
GROUND_CELL = (0.02, 0.2)   # m: the IPM grid's cell size is the ground footprint of a pixel, within these
IPM_BLOCK_CELLS = 16        # cells per IPM patch side …
IPM_STRIDE = 8              # … one every this many cells (they overlap: each σ is scaled by block / stride)
IPM_MIN_NCC = 0.55          # a patch match below this correlation is dropped
IPM_DISTINCT = 0.08         # … and one whose second peak (2 cells away or more) comes within this of the best
IPM_EFF = 8                 # a pair's patches in one window count as at most this many independent ones
IPM_RIDGE = 0.15            # a peak this much flatter one way than the other is a ridge (a line): pinned across only
IPM_MIN_ELEV = 10.0         # deg: ground seen at a lower angle below the horizon is left out (grazing)
OUTLINE_TRAVEL_M = 2.5      # m: the outline needs the vehicle to travel this far over the window's frames
OBSTACLE_M = 0.15           # m: lidar points this far above the ground plane stand on it: their cells are left out
LIMITS = {'window_s': (0.2, 3600.0), 'stride_s': (0.1, 3600.0), 'frames_per_window': (1, 30),
          'max_frames': (10, 3000), 'search_deg': (0.2, 5.0), 'walk_deg_per_min': (0.0001, 10.0),
          'huber_k': (0.5, 20.0), 'tolerance_deg': (0.01, 10.0), 'min_step_deg': (0.005, 10.0),
          'flex_min_deg': (0.005, 10.0), 'bic_margin': (0.0, 100.0), 'ground_range_m': (2.0, 30.0),
          'tf_hz': (0.1, 200.0), 'seed': (0, 999999)}
INTS = ('frames_per_window', 'max_frames', 'seed')
BOOLS = ('edges', 'outline', 'ground', 'fit_translation', 'lidar_state')
ASSUMPTIONS = (
    'Only the depth edges are on by default: the outline needs the recorded vehicle to be the project\'s body and the '
    'ground factor a lens model true far off its axis. Corrections are small (within the search range) and constant '
    'within a window; a change inside a window is seen '
    'as a mix of the two poses. The lidar is the reference: a camera\'s correction is relative to it, and the lidar\'s '
    'own rotation against the body is constrained only where a camera\'s outline and its depth edges meet. Depth '
    'edges need structure in view (open road gives none) and take moving objects as static; the outline needs the '
    'vehicle to move and the body to be in view; the ground factor needs a textured, flat road between two cameras, '
    'takes the lidar\'s plane over the window as the road and the cameras\' positions as planned (a lidar rotation '
    'moves the cameras\' positions in its frame slightly, which is not modelled). Translation is observable only '
    'through the outline (fit_translation) and weakly. Flex is modelled as linear in speed, acceleration, yaw rate or '
    'the time of day (temperature is not measured; the time of day only stands in for it over long drives); other '
    'causes (road roughness, load) are not modelled. The steps\' and flex terms\' significance assumes the per-window '
    'errors are independent apart from the overlap between windows.')
DOWNSTREAM = {
    'nuscenes': ('The nuScenes devkit and loaders built on it (mmdetection3d\'s and OpenPCDet\'s converters, '
                 'nuscenes2bag) read a sample_data\'s extrinsics through its calibrated_sensor_token, so each frame '
                 'gets the extrinsics of its segment without any change to them. A loader that caches one '
                 'calibrated_sensor per channel (from a scene\'s first sample) sees only the first segment.'),
    'ros2': ('tf2 listeners (RViz, Foxglove, projection and fusion nodes using image_geometry or message_filters) look '
             'up base_link to <tag>_link at each message\'s stamp and get the interpolated extrinsics. Play the '
             'sidecar '
             'beside the drive without the drive\'s own /tf_static: the sidecar carries every static transform of the '
             'plan except the corrected cameras\', which it publishes on /tf over time.'),
    'json': ('A custom pipeline looks up a time in the segments (constant extrinsics between steps: use them for '
             'knocks) or in the interpolated series (use it where flex matters), and can weight or gate by the σ.')}


# =========================================================================== settings
@dataclass
class OnlineCalibSettings:
    path: str = ''                     # the bag or nuScenes folder last calibrated
    variant_uid: str = ''              # the plan; '' = the active variant
    window_s: float = 10.0             # window length (s)
    stride_s: float = 10.0             # a window every this many seconds (shorter than the window: they overlap)
    frames_per_window: int = 4         # instants sampled per window length
    max_frames: int = 400              # instants over the whole drive at most (a long drive is sampled thinner)
    search_deg: float = 1.5            # corrections searched within ± this per window
    edges: bool = True                 # depth edges against image edges
    outline: bool = False              # the own vehicle's outline (needs the recorded vehicle to be the project's body)
    ground: bool = False               # ground-plane consistency between overlapping cameras (needs a textured flat
                                       # road and a lens model true far off its axis)
    fit_translation: bool = False      # also fit the cameras' positions from the outline (weakly observable)
    lidar_state: bool = False          # also estimate the lidar's own rotation against the body (outline vs edges)
    walk_deg_per_min: float = 0.05     # random walk between windows: σ of the drift over one minute (deg)
    huber_k: float = 1.5               # Huber threshold on a factor's whitened residual
    tolerance_deg: float = 0.3         # a knock beyond this needs a new calibration (exit code 3)
    min_step_deg: float = 0.1          # steps smaller than this are not reported
    flex_min_deg: float = 0.1          # flex smaller than this (half its range over the drive) is not reported
    bic_margin: float = 6.0            # a model term must lower the BIC by more than this
    ground_range_m: float = 8.0        # the ground factor uses ground up to this far from a camera (horizontally)
    tf_hz: float = 10.0                # rate of the ROS 2 sidecar's /tf
    seed: int = 0                      # RANSAC of the ground plane

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d: dict | None) -> 'OnlineCalibSettings':
        """Tolerant of junk: unknown keys dropped, numbers clamped to LIMITS (NaN ignored), switches checked."""
        s = OnlineCalibSettings()
        d = d if isinstance(d, dict) else {}
        if isinstance(d.get('path'), str):
            s.path = d['path'][:4096]
        if isinstance(d.get('variant_uid'), str):
            s.variant_uid = d['variant_uid'][:64]
        for k in BOOLS:
            v = d.get(k)
            if isinstance(v, bool) or (isinstance(v, int) and not isinstance(v, float)):
                setattr(s, k, bool(v))
            elif isinstance(v, str):
                setattr(s, k, v.strip().lower() in ('1', 'true', 'yes', 'on'))
        for k, (lo, hi) in LIMITS.items():
            if k not in d or isinstance(d[k], bool):
                continue
            try:
                v = float(d[k])
            except (TypeError, ValueError):
                continue
            if math.isfinite(v):
                v = min(max(v, lo), hi)
                setattr(s, k, int(round(v)) if k in INTS else v)
        return s


def settings_of(project) -> OnlineCalibSettings:
    return OnlineCalibSettings.from_dict(getattr(project.analysis, 'online_calib', None))


# =========================================================================== small maths
def _rotvec(R) -> np.ndarray:
    return DL._R_rotvec(np.asarray(R, float))


def jac_rot(ypr) -> np.ndarray:
    """3×3: the vehicle-frame rotation vector (deg) of rot_zyx(ypr + δ) · rot_zyx(ypr)ᵀ per degree of δ (yaw, pitch,
    roll), by central differences: how an Euler correction of a sensor turns it in the vehicle frame."""
    ypr = np.asarray(ypr, float)
    R0 = rot_zyx(*ypr)
    h = 0.01
    J = np.zeros((3, 3))
    for a in range(3):
        e = np.zeros(3)
        e[a] = h
        J[:, a] = np.degrees(_rotvec(rot_zyx(*(ypr + e)) @ R0.T) - _rotvec(rot_zyx(*(ypr - e)) @ R0.T)) / (2 * h)
    return J


def huber_w(e, k):
    """Huber weights of whitened residual norms e (per factor of dimension dim: threshold k·√dim is applied by the
    caller through e)."""
    e = np.asarray(e, float)
    return np.where(e <= k, 1.0, k / np.maximum(e, 1e-12))


def clock(t_s) -> str:
    """A drive time as m:ss.s (under ten minutes), mm:ss, or h:mm:ss."""
    t = max(0.0, float(t_s))
    if t >= 3600:
        return f'{int(t // 3600)}:{int(t % 3600 // 60):02d}:{int(t % 60):02d}'
    if t >= 600:
        return f'{int(t // 60):02d}:{int(t % 60):02d}'
    return f'{int(t // 60):02d}:{t % 60:04.1f}'


def _r(v, d=4):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return round(f, d) + 0.0 if math.isfinite(f) else None


def _rl(v, d=4):
    return None if v is None else [_r(x, d) for x in v]


# =========================================================================== the state
@dataclass
class Slot:
    tag: str
    uid: str
    kind: str          # 'camera' | 'lidar'
    axis: str          # yaw | pitch | roll | x | y | z
    index: int

    @property
    def unit(self) -> str:
        return 'm' if self.axis in TAXES else '°'

    @property
    def prior(self) -> float:
        return PRIOR_M if self.axis in TAXES else PRIOR_DEG


class Layout:
    """Where each sensor's corrections sit in the per-window state vector."""

    def __init__(self, cams, lidar, translation, lidar_state=False):
        self.slots: list[Slot] = []
        self.rot, self.trans = {}, {}
        for c in cams:
            self.rot[c.tag] = self._add(c, 'camera', AXES)
            if translation:
                self.trans[c.tag] = self._add(c, 'camera', TAXES)
        self.lidar = lidar.tag if lidar is not None and lidar_state else None
        self.reference = lidar.tag if lidar is not None else None
        if self.lidar is not None:
            self.rot[lidar.tag] = self._add(lidar, 'lidar', AXES)
        self.n = len(self.slots)
        self.prior = np.array([s.prior for s in self.slots])

    def _add(self, obj, kind, axes):
        idx = []
        for a in axes:
            idx.append(len(self.slots))
            self.slots.append(Slot(obj.tag, obj.uid, kind, a, len(self.slots)))
        return np.array(idx)

    def sensors(self) -> list:
        out = []
        for s in self.slots:
            if s.tag not in out:
                out.append(s.tag)
        return out


class Rows:
    """Whitened linear measurement rows of a window: A x ≈ b, grouped per factor (a group shares one robust weight)."""

    def __init__(self, n):
        self.n = n
        self.A, self.b, self.g = [], [], []
        self.ng = 0

    def add(self, H, z, sigma, group=None):
        """One factor: rows H (m × n) and z (m) with standard errors sigma (m). Returns its group id; group: add the
        rows to that existing group (one robust weight for them all)."""
        H, z, s = np.atleast_2d(np.asarray(H, float)), np.atleast_1d(np.asarray(z, float)), np.atleast_1d(sigma)
        ok = np.isfinite(s) & (s > 0) & np.isfinite(z)
        if not ok.any():
            return None
        if group is None:
            gid = self.ng
            self.ng += 1
        else:
            gid = group
        self.A.append(H[ok] / s[ok, None])
        self.b.append(z[ok] / s[ok])
        self.g.append(np.full(int(ok.sum()), gid))
        return gid

    def arrays(self):
        if not self.A:
            return np.zeros((0, self.n)), np.zeros(0), np.zeros(0, int)
        return np.vstack(self.A), np.concatenate(self.b), np.concatenate(self.g)


def _group_weights(A, b, g, x, ng, k):
    if not len(b):
        return np.ones(ng)
    r = b - A @ x
    e2 = np.bincount(g, weights=r * r, minlength=ng)
    dim = np.maximum(np.bincount(g, minlength=ng), 1)
    return huber_w(np.sqrt(e2) / np.sqrt(dim), k)


def solve_window(A, b, g, ng, prior, k, iters=10):
    """The window on its own: x minimising Σ Huber(factor residuals) + |x / prior|², by iteratively reweighted least
    squares. Returns (x, covariance, the factors' weights)."""
    n = len(prior)
    P0 = np.diag(1.0 / np.asarray(prior, float) ** 2)
    w = np.ones(ng)
    x = np.zeros(n)
    for _ in range(iters):
        wr = w[g] if len(g) else np.zeros(0)
        Lam = A.T @ (A * wr[:, None]) + P0
        eta = A.T @ (b * wr)
        x_new = np.linalg.solve(Lam, eta)
        w_new = _group_weights(A, b, g, x_new, ng, k)
        done = np.abs(x_new - x).max() < 1e-6 and np.abs(w_new - w).max() < 1e-4
        x, w = x_new, w_new
        if done:
            break
    wr = w[g] if len(g) else np.zeros(0)
    Lam = A.T @ (A * wr[:, None]) + P0
    return np.linalg.solve(Lam, A.T @ (b * wr)), np.linalg.inv(Lam), w


def smooth_chain(infos, t, prior, q_rate, breaks=None, offsets=None, scale=1.0):
    """The sliding-window least squares over the whole drive, solved exactly as a linear Gaussian chain (Kalman filter
    forward, Rauch-Tung-Striebel smoother back). infos: per window (Λ, η) of its measurements (information form, on the
    full state x_k = s_k + offsets_k); s_{k+1} = s_k + w with Cov w = diag(q_rate² · Δt) (q_rate per state per √s);
    breaks: {k: [state indices]} whose random walk is released between window k−1 and k (a step); scale multiplies
    every Λ (overlapping windows). Returns (s (K × n), covariance (K × n × n))."""
    K = len(infos)
    n = len(prior)
    P0 = np.diag(np.asarray(prior, float) ** 2)
    off = np.zeros((K, n)) if offsets is None else np.asarray(offsets, float)
    m_f = np.zeros((K, n))
    P_f = np.zeros((K, n, n))
    m_p = np.zeros((K, n))
    P_p = np.zeros((K, n, n))
    big = (10.0 * np.asarray(prior, float)) ** 2
    for k in range(K):
        if k == 0:
            mp, Pp = np.zeros(n), P0.copy()
        else:
            dt = max(float(t[k] - t[k - 1]), 1e-6)
            Q = np.diag(np.asarray(q_rate, float) ** 2 * dt)
            for i in (breaks or {}).get(k, []):
                Q[i, i] = big[i]
            mp, Pp = m_f[k - 1], P_f[k - 1] + Q
        m_p[k], P_p[k] = mp, Pp
        Lam, eta = infos[k]
        Lam = Lam * scale
        eta = (eta - infos[k][0] @ off[k]) * scale
        Pi = np.linalg.inv(Pp)
        Pk = np.linalg.inv(Pi + Lam)
        Pk = 0.5 * (Pk + Pk.T)
        m_f[k] = Pk @ (Pi @ mp + eta)
        P_f[k] = Pk
    m_s, P_s = m_f.copy(), P_f.copy()
    for k in range(K - 2, -1, -1):
        G = P_f[k] @ np.linalg.inv(P_p[k + 1])
        m_s[k] = m_f[k] + G @ (m_s[k + 1] - m_p[k + 1])
        P = P_f[k] + G @ (P_s[k + 1] - P_p[k + 1]) @ G.T
        P_s[k] = 0.5 * (P + P.T)
    return m_s, P_s


# =========================================================================== slip and flex: model selection
def _wls(X, y, w):
    sw = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    r = y - X @ beta
    return beta, float(np.sum(w * r * r)), r


def select_model(t, y, sigma, covs=None, rho=1.0, margin=6.0, min_seg=MIN_SEG, max_terms=MAX_TERMS, huber_k=1.5,
                 min_step=0.0, min_amp=0.0):
    """The per-window estimates y (with σ) of one sensor axis at window centres t explained by a constant plus steps
    and linear covariate terms, chosen greedily by the Bayesian information criterion (see the module docstring), then
    pruned: a step smaller than min_step or within 3 σ of zero, a covariate term whose amplitude (half its effect over
    the covariate's 5–95 % range) is under min_amp or whose slope is within 3 σ of zero, with any one window left out,
    is dropped (the weakest first) and the rest refitted. covs: {name: values at the windows, already relative to
    their reference}. Returns {'steps':
    [{'j', 'tau', 'size', 'sigma', 'col'}], 'flex': [{'cov', 'slope', 'sigma', 'col'}], 'level', 'level_sigma', 'X',
    'beta', 'cov', 'scale', 'fit', 'n', 'terms', 'bic', 'dropped'}."""
    t, y, s = (np.asarray(a, float) for a in (t, y, sigma))
    n = len(y)
    covs = {k: np.asarray(v, float) for k, v in (covs or {}).items()}
    w0 = 1.0 / np.maximum(s, 1e-9) ** 2
    ne = max(rho * n, 1.0)

    def bic(rss, p):
        return ne * math.log(max(rss / n, SCALE_FLOOR ** 2)) + p * math.log(max(ne, 1.0 + 1e-9))

    def col_of(term):
        kind, j = term
        if kind == 'level':
            return np.ones(n)
        if kind == 'step':
            return (np.arange(n) >= j).astype(float)
        return covs[j]

    def n_par(terms):
        return sum(2 if k == 'step' else 1 for k, _ in terms)

    def fit(terms, rw):
        X = np.stack([col_of(tm) for tm in terms], 1)
        beta, rss, r = _wls(X, y, w0 * rw)
        return X, beta, rss, r

    def robust(r, rss, p_):
        if n < 3:
            return np.ones(n)
        sc = math.sqrt(max(rss / max(n - p_, 1), SCALE_FLOOR ** 2))
        return huber_w(np.abs(r) * np.sqrt(w0) / sc, huber_k)

    terms = [('level', None)]
    rw = np.ones(n)
    X, beta, rss, r = fit(terms, rw)
    trace = [bic(rss, 1)]
    while n >= 3 and len(terms) - 1 < max_terms:
        # candidates are scored on plain weighted least squares: Huber weights taken from a model that lacks the
        # structure would mark its windows as outliers and hide it (a step's windows under a constant); robustness
        # comes from MIN_SEG (one window cannot make a step), the leave-one-out check and the robust final fit
        X, _, rss_c, _ = fit(terms, rw)
        base = bic(rss_c, n_par(terms))
        best = None
        taken = [j for k_, j in terms if k_ == 'step']
        for j in range(min_seg, n - min_seg + 1):
            if any(abs(j - jj) < min_seg for jj in taken):
                continue
            _, rss_j, _ = _wls(np.column_stack([X, col_of(('step', j))]), y, w0 * rw)
            b = bic(rss_j, n_par(terms) + 2)
            if best is None or b < best[0]:
                best = (b, ('step', j))
        for name, v in covs.items():
            if ('cov', name) in terms or n < 4 or float(np.std(v)) <= 0:
                continue
            _, rss_c2, _ = _wls(np.column_stack([X, v]), y, w0 * rw)
            b = bic(rss_c2, n_par(terms) + 1)
            if best is None or b < best[0]:
                best = (b, ('cov', name))
        if (best is None or best[0] > base - margin) and n >= 3 * min_seg:
            # no single step pays: a pair (a shift that returns, which one step cannot fit) may; the grid is
            # thinned on long series so the search stays near 10⁴ fits
            g = max(1, n // 100)
            js = [j for j in range(min_seg, n - min_seg + 1, g) if all(abs(j - jj) >= min_seg for jj in taken)]
            for a_ in js:
                ca = col_of(('step', a_))
                for b_ in js:
                    if b_ - a_ < min_seg:
                        continue
                    _, rss_p, _ = _wls(np.column_stack([X, ca, col_of(('step', b_))]), y, w0 * rw)
                    b = bic(rss_p, n_par(terms) + 4)
                    if best is None or b < best[0]:
                        best = (b, ('pair', (a_, b_)))
        if best is None or best[0] > base - margin:
            break
        if best[1][0] == 'pair':
            terms += [('step', best[1][1][0]), ('step', best[1][1][1])]
            X, beta, rss, r = fit(terms, rw)
            trace.append(bic(rss, n_par(terms)))
            continue
        terms.append(best[1])
        X, beta, rss, r = fit(terms, rw)
        trace.append(bic(rss, n_par(terms)))

    def final(terms):
        X, beta, rss, r = fit(terms, np.ones(n))
        rw = robust(r, rss, len(terms))
        X, beta, rss, r = fit(terms, rw)
        s2 = max(rss / max(n - len(terms), 1), SCALE_FLOOR ** 2)
        C = np.linalg.pinv(X.T @ (X * (w0 * rw)[:, None])) * s2 / max(min(rho, 1.0), 1e-6)
        return X, beta, rss, C, rw, s2
    X, beta, rss, C, rw, s2 = final(terms)
    dropped = []
    while len(terms) > 1:                                # prune what would not be reported, the weakest first
        weak = []
        for i, (kind, j) in enumerate(terms):
            if kind == 'level':
                continue
            se = math.sqrt(max(C[i, i], 0.0))
            z = abs(beta[i]) / max(se, 1e-12)
            if kind == 'step':
                bad = abs(beta[i]) < min_step or z <= 3.0
            else:
                lo, hi = np.percentile(covs[j], [5, 95]) if len(covs[j]) else (0.0, 0.0)
                bad = 0.5 * abs(beta[i]) * (hi - lo) < min_amp or z <= 3.0
                if not bad:                              # a slope must not rest on one window: leave each one out
                    z = min(z, _loo_z(X, y, w0 * rw, i, rho))
                    bad = z <= 3.0
            if bad:
                weak.append((z, i))
        if not weak:
            break
        _, i = min(weak)
        dropped.append(terms[i])
        terms = terms[:i] + terms[i + 1:]
        X, beta, rss, C, rw, s2 = final(terms)
    out = {'steps': [], 'flex': [], 'level': float(beta[0]), 'level_sigma': float(math.sqrt(max(C[0, 0], 0.0))),
           'X': X, 'beta': beta, 'cov': C, 'scale': math.sqrt(s2), 'fit': X @ beta, 'n': n, 'terms': terms,
           'bic': trace, 'weights': rw, 'dropped': dropped}
    for i, (kind, j) in enumerate(terms):
        if kind == 'step':
            out['steps'].append({'j': int(j), 'tau': float(0.5 * (t[j - 1] + t[j])), 'size': float(beta[i]),
                                 'sigma': float(math.sqrt(max(C[i, i], 0.0))), 'col': i})
        elif kind == 'cov':
            out['flex'].append({'cov': j, 'slope': float(beta[i]), 'sigma': float(math.sqrt(max(C[i, i], 0.0))),
                                'col': i})
    out['steps'].sort(key=lambda st: st['j'])
    return out


def _loo_z(X, y, w, i, rho):
    """The smallest |β_i| / σ_i over the fits that leave one window out (an influence check: a flex slope two outlying
    windows make up does not survive it)."""
    n, p_ = X.shape
    if n - 1 <= p_:
        return 0.0
    zs = []
    for k in range(n):
        wk = w.copy()
        wk[k] = 0.0
        beta, rss, _ = _wls(X, y, wk)
        s2 = max(rss / max(n - 1 - p_, 1), SCALE_FLOOR ** 2)
        C = np.linalg.pinv(X.T @ (X * wk[:, None])) * s2 / max(min(rho, 1.0), 1e-6)
        zs.append(abs(beta[i]) / max(math.sqrt(max(C[i, i], 0.0)), 1e-12))
    return float(min(zs))


def refine_step(frames_t, frames_y, tau, before, after, lo, hi):
    """The step time refined from per-frame estimates (t, y) between lo and hi: the split of the frames into before
    (level `before`) and after (level `after`) with the least robust cost; (tau, resolution s). The coarse tau is kept
    when fewer than two frames lie on either side."""
    t = np.asarray(frames_t, float)
    y = np.asarray(frames_y, float)
    m = (t >= lo) & (t <= hi) & np.isfinite(y)
    t, y = t[m], y[m]
    if len(t) < 4:
        return tau, None
    o = np.argsort(t)
    t, y = t[o], y[o]
    mad = 1.4826 * float(np.median(np.abs(y - np.where(np.abs(y - before) < np.abs(y - after), before, after))))
    c = max(mad, 0.25 * abs(after - before), 1e-6)

    def rho(e):
        a = np.abs(e) / c
        return np.where(a <= 1.5, 0.5 * a * a, 1.5 * a - 0.5 * 1.5 ** 2)
    best, cand = None, []
    for i in range(2, len(t) - 1):
        cost = float(np.sum(rho(y[:i] - before)) + np.sum(rho(y[i:] - after)))
        cand.append((cost, i))
        if best is None or cost < best[0]:
            best = (cost, i)
    if best is None:
        return tau, None
    ties = [i for cst, i in cand if cst <= best[0] + 1e-9]
    i = ties[len(ties) // 2]
    res = float(np.median(np.diff(t))) if len(t) > 1 else None
    return float(0.5 * (t[i - 1] + t[i])), res


# =========================================================================== covariates from the motion
def _speed_fn(motion):
    if motion.speed is None:
        return None
    t, v = motion.speed
    if not len(t):
        return None
    return lambda x: np.interp(x, t, v)


def covariates_at(motion, times, start_ns) -> dict:
    """{name: values at the times (s after the drive start)}: speed (m/s), accel (m/s², the speed's derivative over
    0.5 s, braking negative), yaw_rate (°/s: odometry heading over 0.5 s, or the IMU's z rate) and time_of_day (UTC
    hours). A covariate the log cannot give is left out."""
    tt = np.asarray(times, float)
    out = {}
    sp = _speed_fn(motion)
    if sp is not None:
        out['speed'] = sp(tt)
        out['accel'] = (sp(tt + 0.25) - sp(tt - 0.25)) / 0.5
    if motion.kind == 'odometry' and motion.t is not None and len(motion.t) > 2:
        yaw = motion.ypr[:, 0]
        out['yaw_rate'] = np.degrees(np.interp(tt + 0.25, motion.t, yaw) - np.interp(tt - 0.25, motion.t, yaw)) / 0.5
    elif motion.kind == 'imu' and motion.omega is not None and len(motion.t) > 2:
        out['yaw_rate'] = np.degrees(np.interp(tt, motion.t, motion.omega[:, 2]))
    out['time_of_day'] = ((start_ns / 1e9 + tt) % 86400.0) / 3600.0
    return out


def _cov_reference(name, values) -> float:
    """The covariate's value at which the flex term is zero: rest (speed, accel and yaw rate 0) or, when the drive never
    comes near it, the nearest value it reached (the slowest speed driven: never an extrapolation); the drive's start
    for the time of day."""
    v = np.asarray(values, float)
    if not len(v):
        return 0.0
    if name == 'time_of_day':
        return float(v[0])
    return float(np.clip(0.0, v.min(), v.max()))


# =========================================================================== the ground factor (IPM)
def fit_ground_plane(P, rng, max_h=0.4, r_lo=2.0, r_hi=25.0, iters=120, tol=0.05):
    """(unit normal n with n_z > 0, d) of the plane n·p + d = 0 through the road points among P (vehicle frame, m):
    points within max_h of z = 0 and r_lo … r_hi m away horizontally, RANSAC then least squares on the inliers. None
    with fewer than 50 such points."""
    P = np.asarray(P, float)
    if not len(P):
        return None
    r = np.hypot(P[:, 0], P[:, 1])
    Q = P[(np.abs(P[:, 2]) < max_h) & (r >= r_lo) & (r <= r_hi)]
    if len(Q) < 50:
        return None
    if len(Q) > 20000:
        Q = Q[np.linspace(0, len(Q) - 1, 20000).astype(int)]
    best, best_n = None, -1
    for _ in range(iters):
        i = rng.choice(len(Q), 3, replace=False)
        nrm = np.cross(Q[i[1]] - Q[i[0]], Q[i[2]] - Q[i[0]])
        L = np.linalg.norm(nrm)
        if L < 1e-9:
            continue
        nrm = nrm / L
        if nrm[2] < 0:
            nrm = -nrm
        if nrm[2] < 0.95:
            continue
        d = -float(nrm @ Q[i[0]])
        cnt = int(np.count_nonzero(np.abs(Q @ nrm + d) < tol))
        if cnt > best_n:
            best, best_n = (nrm, d), cnt
    if best is None:
        return None
    nrm, d = best
    inl = Q[np.abs(Q @ nrm + d) < tol]
    c = inl.mean(0)
    _, _, Vt = np.linalg.svd(inl - c, full_matrices=False)
    nrm = Vt[-1]
    if nrm[2] < 0:
        nrm = -nrm
    return nrm, -float(nrm @ c), int(len(inl))


class _CamGeo:
    """A camera's lens and planned pose for the ground factor (analysis pixels as core.drivelog uses them)."""

    def __init__(self, project, cam, ashape):
        from .lens import resolve
        mod = project.module_for(cam)
        self.cam = cam
        self.intr = resolve(mod, project.lens_for(cam))
        self.W, self.H = int(mod.width_px), int(mod.height_px)
        self.ha, self.wa = ashape
        self.sx, self.sy = self.wa / self.W, self.ha / self.H
        self.ypr = np.array([cam.yaw, cam.pitch, cam.roll], float)
        self.pos = cam.pos()
        self.W2O = BODY_TO_OPTICAL @ rot_zyx(*self.ypr).T

    def project(self, G, W2O=None):
        """Analysis pixel (x, y) of vehicle-frame points G and whether each lies in view (in front, inside the lens
        circle's THETA_SHARE and the image)."""
        V = np.asarray(G, float) - self.pos
        u, v, th, ok = self.intr.project_dirs(V @ (self.W2O if W2O is None else W2O).T)
        x, y = u * self.sx - 0.5, v * self.sy - 0.5
        ok = ok & (th <= DL.THETA_SHARE * self.intr.max_theta) & (x >= 1) & (y >= 1) & (x <= self.wa - 2) & \
            (y <= self.ha - 2)
        return x, y, ok

    def ground_shift(self, G, plane, h=0.05):
        """∂(ground point)/∂(yaw, pitch, roll correction) at G (N × 3 on the plane): (N, 2, 3) m per degree, x and y.
        The pixel that sees G at the plan sees, with the camera turned by δ, the plane where its turned ray meets it."""
        nrm, d = plane
        V = np.asarray(G, float) - self.pos
        o = (V / np.linalg.norm(V, axis=1)[:, None]) @ self.W2O.T            # the pixels' optical rays
        out = np.zeros((len(G), 2, 3))
        for a in range(3):
            pts = []
            for sgn in (1.0, -1.0):
                e = np.zeros(3)
                e[a] = sgn * h
                Dv = o @ (BODY_TO_OPTICAL @ rot_zyx(*(self.ypr + e)).T)     # optical ray → vehicle, turned camera
                den = Dv @ nrm
                s = -(self.pos @ nrm + d) / np.where(np.abs(den) > 1e-9, den, np.nan)
                pts.append(self.pos + Dv * s[:, None])
            out[:, :, a] = (pts[0][:, :2] - pts[1][:, :2]) / (2 * h)
        return out


def _plane_z(plane, x, y):
    nrm, d = plane
    return -(d + nrm[0] * x + nrm[1] * y) / nrm[2]


def ground_pairs(project, geos: dict, engine, var, range_m, cell=None) -> list:
    """Camera pairs that overlap on the ground at the plan: their common grid (vehicle frame, z = 0), the cells both
    see (in view, at least IPM_MIN_ELEV° below the horizon, within range_m horizontally, not behind the own vehicle)
    and the patches (IPM_BLOCK_CELLS square) mostly inside the overlap. cell: the grid's cell (m); by default the
    coarser camera's ground footprint of one analysis pixel over the overlap, within GROUND_CELL."""
    tags = sorted(geos)
    has_body = engine is not None and engine.vehicle_geom(project, var) is not None
    out = []
    for i, a in enumerate(tags):
        for b in tags[i + 1:]:
            A, B = geos[a], geos[b]
            lo = np.minimum(A.pos[:2], B.pos[:2]) - range_m
            hi = np.maximum(A.pos[:2], B.pos[:2]) + range_m
            c0 = 0.1
            nx, ny = int((hi[0] - lo[0]) / c0), int((hi[1] - lo[1]) / c0)
            gx, gy = np.meshgrid(lo[0] + (np.arange(nx) + 0.5) * c0, lo[1] + (np.arange(ny) + 0.5) * c0)
            G = np.c_[gx.ravel(), gy.ravel(), np.zeros(gx.size)]
            ok = _sees(A, G, range_m, engine if has_body else None) & _sees(B, G, range_m, engine if has_body else None)
            if ok.sum() < 200:
                continue
            Go = G[ok]
            c = cell
            if c is None:
                fp = []
                for X in (A, B):
                    V = Go - X.pos
                    r = np.linalg.norm(V, axis=1)
                    ppd = X.intr.f_px * X.sx * math.pi / 180.0
                    elev = np.abs(V[:, 2]) / r
                    fp.append(np.median(r * math.radians(1.0) / ppd / np.maximum(elev, 0.2)))
                c = float(np.clip(max(fp), *GROUND_CELL))
            lo2, hi2 = Go[:, :2].min(0) - 2 * c, Go[:, :2].max(0) + 2 * c
            nx, ny = int(math.ceil((hi2[0] - lo2[0]) / c)), int(math.ceil((hi2[1] - lo2[1]) / c))
            if nx * ny > 4_000_000:
                continue
            gx, gy = np.meshgrid(lo2[0] + (np.arange(nx) + 0.5) * c, lo2[1] + (np.arange(ny) + 0.5) * c)
            G = np.c_[gx.ravel(), gy.ravel(), np.zeros(gx.size)]
            ok = (_sees(A, G, range_m, engine if has_body else None) &
                  _sees(B, G, range_m, engine if has_body else None)).reshape(ny, nx)
            bc = IPM_BLOCK_CELLS
            blocks = []
            for i0 in range(0, ny - bc + 1, IPM_STRIDE):
                for j0 in range(0, nx - bc + 1, IPM_STRIDE):
                    if ok[i0:i0 + bc, j0:j0 + bc].mean() >= 0.8:
                        blocks.append((i0, j0))
            if not blocks:
                continue
            out.append({'a': a, 'b': b, 'cell': c, 'x0': float(lo2[0]), 'y0': float(lo2[1]), 'nx': nx, 'ny': ny,
                        'ok': ok, 'blocks': blocks, 'area_m2': float(ok.sum() * c * c)})
    return out


def _sees(X: _CamGeo, G, range_m, engine):
    V = G - X.pos
    r = np.linalg.norm(V, axis=1)
    _, _, ok = X.project(G)
    ok &= np.hypot(V[:, 0], V[:, 1]) <= range_m
    ok &= -V[:, 2] / np.maximum(r, 1e-9) >= math.sin(math.radians(IPM_MIN_ELEV))
    if engine is not None and ok.any():
        bd = engine.cube(X.pos).lookup(V[ok] / r[ok, None])[0]
        hid = np.isfinite(bd) & (bd > 0.05) & (bd < r[ok] - 0.05)
        idx = np.nonzero(ok)[0]
        ok[idx[hid]] = False
    return ok


def _rel_motion(motion, ta, tb):
    """(R, p): the vehicle's pose at tb in its own frame at ta (a point X_a of the frame at ta is R.T (X_a − p) in the
    frame at tb). Odometry: both; IMU: the turn only; else the speed along x when known; else none."""
    if motion.kind == 'odometry':
        Ra, pa = motion._pose(ta)
        Rb, pb = motion._pose(tb)
        return Ra.T @ Rb, Ra.T @ (pb - pa)
    R = np.eye(3)
    if motion.kind == 'imu':
        sel = (motion.t >= min(ta, tb) - 0.05) & (motion.t <= max(ta, tb) + 0.05)
        w = motion.omega[sel].mean(0) if sel.any() else motion.omega[np.argmin(np.abs(motion.t - ta))]
        R = DL._rotvec_R(w * (tb - ta))
    v = motion.speed_at(0.5 * (ta + tb)) if motion.speed is not None else None
    return R, np.array([(v or 0.0) * (tb - ta), 0.0, 0.0])


def _decode_grey(smp):
    import cv2
    return cv2.imdecode(np.frombuffer(smp['grey'], np.uint8), cv2.IMREAD_GRAYSCALE)


def ground_factor(pair, A: _CamGeo, B: _CamGeo, imgA, imgB, ta, tb, plane, motion, obstacles=None, search_deg=1.5):
    """The patches of one frame pair: [(g (2,), Δ (2,) m, σ (2,) m, ncc, J_A (2×3), J_B (2×3), V (2×2))], Δ the
    displacement of A's warped ground texture in B's (see the module docstring), σ its standard errors along the
    columns of V (the correlation peak's principal directions: along a line marking infinite); J_B already turned into
    A's frame. obstacles: vehicle-frame
    points (at ta) whose cells are left out when they stand OBSTACLE_M above the plane."""
    import cv2
    c, nx, ny = pair['cell'], pair['nx'], pair['ny']
    gx, gy = np.meshgrid(pair['x0'] + (np.arange(nx) + 0.5) * c, pair['y0'] + (np.arange(ny) + 0.5) * c)
    G = np.c_[gx.ravel(), gy.ravel(), _plane_z(plane, gx.ravel(), gy.ravel())]
    R, p = _rel_motion(motion, ta, tb)
    Gb = (G - p) @ R                                     # the same ground points in the vehicle frame at tb
    xa, ya, oa = A.project(G)
    xb, yb, ob = B.project(Gb)
    if imgA.shape != (A.ha, A.wa) or imgB.shape != (B.ha, B.wa):
        return []
    f32 = np.float32
    IA = cv2.remap(imgA.astype(f32), xa.reshape(ny, nx).astype(f32), ya.reshape(ny, nx).astype(f32), cv2.INTER_LINEAR)
    IB = cv2.remap(imgB.astype(f32), xb.reshape(ny, nx).astype(f32), yb.reshape(ny, nx).astype(f32), cv2.INTER_LINEAR)
    valid = (oa & ob).reshape(ny, nx) & pair['ok']
    if obstacles is not None and len(obstacles):
        Pq = np.asarray(obstacles, float)
        hgt = Pq @ plane[0] + plane[1]
        Pq = Pq[hgt > OBSTACLE_M]
        if len(Pq):
            j = np.floor((Pq[:, 0] - pair['x0']) / c).astype(int)
            i = np.floor((Pq[:, 1] - pair['y0']) / c).astype(int)
            m = (i >= 0) & (i < ny) & (j >= 0) & (j < nx)
            ob_m = np.zeros((ny, nx), np.uint8)
            ob_m[i[m], j[m]] = 1
            k = 2 * int(math.ceil(0.3 / c)) + 1
            valid &= ~(cv2.dilate(ob_m, np.ones((k, k), np.uint8)) > 0)
    IA = cv2.GaussianBlur(IA, (0, 0), 1.0)
    IB = cv2.GaussianBlur(IB, (0, 0), 1.0)
    bc = IPM_BLOCK_CELLS
    out = []
    for i0, j0 in pair['blocks']:
        vb = valid[i0:i0 + bc, j0:j0 + bc]
        if vb.mean() < 0.8:
            continue
        ci, cj = i0 + bc // 2, j0 + bc // 2
        g = np.array([gx[ci, cj], gy[ci, cj], _plane_z(plane, gx[ci, cj], gy[ci, cj])])
        JA = A.ground_shift(g[None], plane)[0]
        gb = (g - p) @ R
        JB = (R[:2, :2] @ B.ground_shift(gb[None], plane)[0])
        reach = (np.abs(JA).sum(1).max() + np.abs(JB).sum(1).max()) * search_deg
        r = int(np.clip(math.ceil(reach / c), 2, (bc - 8) // 2))
        T = IA[i0 + r:i0 + bc - r, j0 + r:j0 + bc - r]
        S = IB[i0:i0 + bc, j0:j0 + bc]
        if not vb[r:bc - r, r:bc - r].all() or float(T.std()) < 1.0 or float(S.std()) < 1.0:
            continue
        res = cv2.matchTemplate(S, T, cv2.TM_CCOEFF_NORMED)
        _, mx, _, loc = cv2.minMaxLoc(res)
        if not math.isfinite(mx) or mx < IPM_MIN_NCC:
            continue
        px, py = loc
        if px in (0, res.shape[1] - 1) or py in (0, res.shape[0] - 1):
            continue                                     # at the search's edge: the match lies beyond it
        # the correlation peak's curvature: a line marking pins the shift across it only (its 2 × 2 information)
        hxx = res[py, px - 1] - 2 * mx + res[py, px + 1]
        hyy = res[py - 1, px] - 2 * mx + res[py + 1, px]
        hxy = 0.25 * (res[py + 1, px + 1] - res[py + 1, px - 1] - res[py - 1, px + 1] + res[py - 1, px - 1])
        lam, V = np.linalg.eigh(-np.array([[hxx, hxy], [hxy, hyy]], float))
        if lam[1] <= 1e-6:
            continue
        ridge = lam[0] < IPM_RIDGE * lam[1]
        far = res.copy()
        far[max(py - 2, 0):py + 3, max(px - 2, 0):px + 3] = -1.0
        if ridge:                                        # along the marking the correlation stays high: not a rival
            yy, xx = np.mgrid[0:res.shape[0], 0:res.shape[1]]
            across = np.abs((xx - px) * V[0, 1] + (yy - py) * V[1, 1])
            far[across < 1.5] = -1.0
        if far.size and float(far.max()) > mx - IPM_DISTINCT:
            continue                                     # a second peak as good: repeated texture
        A2 = -np.array([[hxx, hxy], [hxy, hyy]], float)
        try:
            sub = -np.linalg.solve(A2, [0.5 * (res[py, px + 1] - res[py, px - 1]),
                                        0.5 * (res[py + 1, px] - res[py - 1, px])])
        except np.linalg.LinAlgError:
            continue
        if ridge:                                        # the sub-cell offset across the ridge only
            sub = V[:, 1] * float(V[:, 1] @ sub)
        sub = np.clip(sub, -0.75, 0.75)
        dlt = np.array([(px + sub[0] - r) * c, (py + sub[1] - r) * c])
        sig = c * (0.3 + 3.0 * (1.0 - mx)) * bc / IPM_STRIDE     # heuristic: a fraction of a cell for a clean
                                                                    # peak, scaled for the patches' overlap
        sig2 = np.array([sig * math.sqrt(lam[1] / lam[0]) if not ridge else np.inf, sig])
        out.append((g[:2], dlt, sig2, float(mx), JA, JB, V))
    return out


# =========================================================================== results
@dataclass
class OnlineCalibResult:
    settings: OnlineCalibSettings
    path: str
    source: str
    variant: str
    variant_uid: str
    lidar: str | None
    span: dict                       # start_ns, end_ns, duration_s
    sensors: list                    # [{'tag', 'uid', 'kind', 'slot', 'topic', 'axes', 'plan_deg', 'plan_pos'}]
    windows: list                    # per window: t0, t1, t, frames, covariates, measurements, constrained, nothing
    series: dict                     # tag -> {'t', 'est', 'sigma', 'win', 'win_sigma', 'constrained', 'axes'}
    models: dict                     # tag -> {axis: model doc}
    segments: dict                   # tag -> [{'t0', 't1', 'delta', 'sigma', ...}]
    events: list
    findings: list
    metrics: dict
    pairs: list = field(default_factory=list)       # ground pairs: a, b, area, cell
    notes: list = field(default_factory=list)
    mapping: dict = field(default_factory=dict)     # topic -> tag
    frames: dict = field(default_factory=dict)      # tag -> {'t': [...], 'y': [[...]]}: the edge frames' own optima
    seconds: float = 0.0

    def sensor(self, tag):
        return next((s for s in self.sensors if s['tag'] == tag), None)

    def worst(self) -> str:
        order = {s: i for i, s in enumerate(('critical', 'serious', 'warning', 'note', 'good'))}
        return min((f['severity'] for f in self.findings), key=lambda s: order.get(s, 9), default='good')


def _finding(out, sev, kind, message, uids=()):
    out.append({'severity': sev, 'kind': kind, 'message': message, 'uids': list(uids)})


def _variant(project, variant, st):
    if variant is not None and not isinstance(variant, str):
        return variant
    uid = variant if isinstance(variant, str) else st.variant_uid
    return (project.variant_by_uid(uid) if uid else None) or project.variant


def window_spans(duration, window, stride) -> list:
    """[(t0, t1)] of the windows over a drive of duration s: every stride s, window s long; the last one is stretched
    to the drive's end when less than half a stride is left over, else one more window ends at the drive's end. One
    window over the whole drive when it is shorter than a window."""
    if duration <= window:
        return [(0.0, float(duration))]
    out = []
    k = 0
    while k * stride + window <= duration + 1e-9:
        out.append([k * stride, k * stride + window])
        k += 1
    left = duration - out[-1][1]
    if left > 1e-9:
        if left < 0.5 * stride:
            out[-1][1] = duration
        else:
            out.append([duration - window, duration])
    return [(float(a), float(b)) for a, b in out]


# =========================================================================== run
def run(project, engine=None, settings: OnlineCalibSettings | None = None, variant=None, progress=None, cancel=None,
        path=None, scan=None) -> OnlineCalibResult | None:
    """The online self-calibration of a drive (the settings' path by default) against a variant's plan (see the module
    docstring). engine: the analysis engine (the own vehicle for the outline and the edge masks; made when None). scan:
    a core.drivelog scan of the same files read with the same instants, used again. None when cancelled; ValueError
    when there is nothing to calibrate; FileNotFoundError when the drive is not there."""
    t_start = time.perf_counter()
    st = settings or settings_of(project)
    path = DL.norm_path(path or st.path)
    if not path:
        raise ValueError('No drive given: open a ROS 2 bag or a nuScenes folder.')
    if engine is None:
        from .analysis import Engine
        engine = Engine()
    var = _variant(project, variant, st)

    def prog(lo, hi):
        if progress is None:
            return None
        return lambda f, text='': progress(lo + (hi - lo) * max(0.0, min(1.0, f)), text)

    def stopped():
        return cancel is not None and cancel()
    info = DL.open_log(path)
    duration = max((info.end_ns - info.start_ns) / 1e9, 0.0)
    notes = []
    frame_step = st.window_s / max(st.frames_per_window, 1)
    n_frames = int(min(max(math.ceil(duration / frame_step), 2), st.max_frames))
    if math.ceil(duration / frame_step) > st.max_frames:
        notes.append(f'The drive ({duration:.0f} s) is sampled at {n_frames} instants (Max frames), one every '
                     f'{duration / n_frames:.2f} s instead of every {frame_step:.2f} s.')
    if scan is None or scan.key != DL.log_key(path) or scan.n_frames != n_frames:
        scan = DL.scan_log(path, n_frames, 20, prog(0.0, 0.35), cancel)
        if scan is None:
            return None
    if stopped():
        return None
    dl = DL.DriveLogSettings.from_dict(getattr(project.analysis, 'drivelog', None))
    dls = DL.DriveLogSettings(path=path, camera_stamp=dl.camera_stamp, lidar_stamp=dl.lidar_stamp,
                              tolerance_deg=st.tolerance_deg, search_deg=st.search_deg, mappings=dl.mappings)
    mapping = DL.mapping_for(var, scan, dls.overrides(path))
    rev = {g: t for t, g in mapping.items()}
    from . import timing as TM
    plan = TM.plan(project, var)
    tm = TM.TimingSettings.from_dict(getattr(project.analysis, 'timing', None))
    conv_ms = DL.stamp_offset_ms(dls.camera_stamp, tm.readout_ms, tm.exposure_ms)
    lid = None
    if plan.lidar is not None:
        lid = var.sensor(plan.lidar.uid)
    if lid is None or lid.tag not in rev:
        lid = next((s for s in var.sensors if s.visible and s.kind == 'lidar' and s.tag in rev), None)
    cams = [c for c in var.cameras if c.visible and c.tag in rev]
    if not cams:
        raise ValueError('No camera of the variant has a topic in the drive: nothing to calibrate.')
    motion = DL.Motion.from_scan(project, var, scan, mapping)
    lay = Layout(cams, lid, st.fit_translation, st.lidar_state)
    duration = scan.duration_s
    spans = window_spans(duration, st.window_s, st.stride_s)
    tgt_t = np.array([(tg['t_ns'] - scan.start_ns) / 1e9 for tg in scan.targets])
    rho = min(1.0, st.stride_s / st.window_s)
    rng = np.random.default_rng(st.seed)
    out = []
    # ---- per-camera geometry, the ground pairs, the lidar's context
    geos = {}
    for c in cams:
        sh = next((tg['images'][rev[c.tag]]['ashape'] for tg in scan.targets if rev[c.tag] in tg['images']), None)
        if sh is not None:
            geos[c.tag] = _CamGeo(project, c, sh)
    pairs = ground_pairs(project, geos, engine, var, st.ground_range_m) if st.ground else []
    lctx = _lidar_ctx(project, lid, rev, scan, dls) if lid is not None else None
    Jc = {c.tag: jac_rot((c.yaw, c.pitch, c.roll)) for c in cams}
    JL = jac_rot((lid.yaw, lid.pitch, lid.roll)) if lid is not None else None
    from . import asbuilt as AB
    abst = AB.settings_of(project)
    abst.fit_translation = st.fit_translation
    pv = project                                         # the outline sees the variant's own rig and housings
    if var is not project.variant:
        pv = copy.copy(project)
        pv.active_uid = var.uid
    has_body = engine.vehicle_geom(project, var) is not None
    if st.outline and not has_body:
        notes.append('The project has no vehicle body: the outline measurement is off.')
    windows, infos, rows_all = [], [], []
    frames_opt = {c.tag: {} for c in cams}
    wp = prog(0.35, 0.9)
    for k, (t0, t1) in enumerate(spans):
        if stopped():
            return None
        if wp is not None:
            wp(k / max(len(spans), 1), f'window {k + 1} of {len(spans)}')
        idx = [i for i, tv in enumerate(tgt_t) if t0 - 1e-9 <= tv < t1 + 1e-9 and (tv < t1 or k == len(spans) - 1)]
        rows = Rows(lay.n)
        meas = []
        win = {'k': k, 't0': t0, 't1': t1, 't': 0.5 * (t0 + t1), 'frames': len(idx)}
        sub = dataclasses.replace(scan, targets=[scan.targets[i] for i in idx])
        if st.edges and lid is not None:
            for c in cams:
                if stopped():
                    return None
                m = _edge_measurement(project, var, c, rev, lid, sub, motion, dls, conv_ms, engine, lay, rows,
                                      frames_opt[c.tag])
                if m is not None:
                    meas.append(m)
        if st.outline and has_body:
            for c in cams:
                if stopped():
                    return None
                m = _outline_measurement(pv, c, rev, sub, motion, abst, engine, lay, rows, Jc, JL)
                if m is not None:
                    meas.append(m)
        plane, plane_doc = None, None
        if pairs:
            plane, plane_doc, obst = _window_plane(sub, lctx, motion, rng, scan.start_ns)
            for pr in pairs:
                if stopped():
                    return None
                m = _ground_measurement(pr, geos, rev, sub, motion, plane, obst, conv_ms, scan.start_ns, lay, rows,
                                        st.search_deg)
                if m is not None:
                    meas.append(m)
        A, b, g = rows.arrays()
        x, P, w = solve_window(A, b, g, rows.ng, lay.prior, st.huber_k)
        wr = w[g] if len(g) else np.zeros(0)
        infos_k = (A.T @ (A * wr[:, None]), A.T @ (b * wr))
        sig = np.sqrt(np.maximum(np.diag(P), 0.0))
        con = sig < CONSTRAINED * lay.prior
        for m in meas:
            gids = m.pop('_groups', [])
            if gids:
                m['weight'] = _r(float(np.mean(w[gids])), 3)
        win.update({'x': x, 'sigma': sig, 'con': con, 'measurements': meas, 'plane': plane_doc})
        win['constrained'] = {tag: [s.axis for s in lay.slots if s.tag == tag and con[s.index]]
                              for tag in lay.sensors()}
        win['nothing'] = _nothing(win, meas, lay, st, lid, pairs, has_body)
        windows.append(win)
        infos.append(infos_k)
        rows_all.append((A, b, g, rows.ng))
    if stopped():
        return None
    if progress is not None:
        progress(0.92, 'solving')
    t_mid = np.array([w['t'] for w in windows])
    cov_all = covariates_at(motion, t_mid, scan.start_ns)
    for k, w in enumerate(windows):
        w['covariates'] = {n: float(v[k]) for n, v in cov_all.items()}
    # ---- slip and flex per sensor axis
    cands = {}
    for name, v in cov_all.items():
        if name == 'time_of_day':
            if duration / 3600.0 < TOD_SPAN_H:
                continue
        elif float(np.std(v)) < COV_MIN_STD[name]:
            continue
        cands[name] = v
    models, offsets, breaks = {}, np.zeros((len(windows), lay.n)), {}
    for s in lay.slots:
        X = np.array([w['x'][s.index] for w in windows])
        S = np.array([w['sigma'][s.index] for w in windows])
        C = np.array([bool(w['con'][s.index]) for w in windows])
        if C.sum() < 2:
            continue
        ii = np.nonzero(C)[0]
        covs = {n: v[ii] - _cov_reference(n, v) for n, v in cands.items()}
        unit_k = 1.0 if s.axis in AXES else PRIOR_M / PRIOR_DEG
        floor = SIGMA_SCENE * (1.0 if s.axis in AXES else PRIOR_M / PRIOR_DEG)
        mdl = select_model(t_mid[ii], X[ii], np.hypot(S[ii], floor), covs, rho, st.bic_margin, MIN_SEG, MAX_TERMS,
                           st.huber_k, st.min_step_deg * unit_k, st.flex_min_deg * unit_k)
        mdl['idx'] = ii
        models.setdefault(s.tag, {})[s.axis] = mdl
        for fx in mdl['flex']:
            v = cands[fx['cov']]
            offsets[:, s.index] += fx['slope'] * (v - _cov_reference(fx['cov'], v))
    events = _events(models, lay, frames_opt, t_mid, st, windows, cands, cams, lid)
    for ev in events:
        if ev['kind'] in ('knock', 'excursion'):
            for ax, tau in ev['_taus']:
                s = next(sl for sl in lay.slots if sl.tag == ev['sensor'] and sl.axis == ax)
                k = int(np.searchsorted(t_mid, tau))
                if 0 < k < len(windows):
                    breaks.setdefault(k, []).append(s.index)
    # ---- the smoother: random walk released at the steps, the flex terms as known offsets; robust weights again
    q = np.array([(st.walk_deg_per_min if s.axis in AXES else st.walk_deg_per_min * PRIOR_M / PRIOR_DEG)
                  / math.sqrt(60.0) for s in lay.slots])
    for _ in range(4):
        m_s, P_s = smooth_chain(infos, t_mid, lay.prior, q, breaks, offsets, rho)
        new = []
        for k, (A, b, g, ng) in enumerate(rows_all):
            w = _group_weights(A, b, g, m_s[k] + offsets[k], ng, st.huber_k)
            wr = w[g] if len(g) else np.zeros(0)
            new.append((A.T @ (A * wr[:, None]), A.T @ (b * wr)))
        infos = new
    m_s, P_s = smooth_chain(infos, t_mid, lay.prior, q, breaks, offsets, rho)
    est = m_s + offsets
    sig_s = np.sqrt(np.maximum(np.einsum('kii->ki', P_s), 0.0))
    for s in lay.slots:                                   # the flex term's own uncertainty
        mdl = models.get(s.tag, {}).get(s.axis)
        if mdl is None:
            continue
        for fx in mdl['flex']:
            v = cands[fx['cov']]
            sig_s[:, s.index] = np.hypot(sig_s[:, s.index], fx['sigma'] * (v - _cov_reference(fx['cov'], v)))
    # ---- series, segments, findings
    series = {}
    for tag in lay.sensors():
        sl = [s for s in lay.slots if s.tag == tag]
        ix = [s.index for s in sl]
        series[tag] = {'axes': [s.axis for s in sl], 't': t_mid.tolist(),
                       'est': est[:, ix].tolist(), 'sigma': sig_s[:, ix].tolist(),
                       'win': [[float(w['x'][i]) for i in ix] for w in windows],
                       'win_sigma': [[float(w['sigma'][i]) for i in ix] for w in windows],
                       'constrained': [[bool(w['con'][i]) for i in ix] for w in windows],
                       'observable': [bool(any(w['con'][i] for w in windows)) for i in ix]}
    segments = _segments(models, lay, events, duration, series)
    sensors = []
    for c in cams:
        sensors.append({'tag': c.tag, 'uid': c.uid, 'kind': 'camera', 'slot': int(getattr(c, 'slot', 0)),
                        'topic': rev.get(c.tag), 'axes': series[c.tag]['axes'],
                        'plan_deg': [c.yaw, c.pitch, c.roll], 'plan_pos': [c.x, c.y, c.z]})
    if lay.lidar is not None:
        sensors.append({'tag': lid.tag, 'uid': lid.uid, 'kind': 'lidar', 'slot': -1, 'topic': rev.get(lid.tag),
                        'axes': list(AXES), 'plan_deg': [lid.yaw, lid.pitch, lid.roll],
                        'plan_pos': [lid.x, lid.y, lid.z]})
    if lid is None:
        notes.append('No lidar of the variant has a topic: no depth edges; the cameras are relative to the vehicle '
                     'body (outline) and to the flat ground z = 0 (ground factor).')
    if motion.kind == 'none':
        notes.append('The drive has no odometry or IMU: the ground factor\'s second camera is moved by the GNSS speed '
                     'along x only (or not at all), and the flex regression has no speed, acceleration or yaw rate.')
    for n in scan.notes:
        notes.append(n)
    findings = _findings(windows, events, series, lay, st, cams, lid, pairs, duration)
    knocks = [e for e in events if e['kind'] == 'knock']
    metrics = {'windows': len(windows), 'frames': len(scan.targets), 'duration_s': duration,
               'measurements': {k_: sum(1 for w in windows for m in w['measurements'] if m['kind'] == k_)
                                for k_ in KINDS},
               'empty_windows': {tag: sum(1 for w in windows if not w['constrained'].get(tag))
                                 for tag in lay.sensors()},
               'knocks': len(knocks), 'excursions': sum(e['kind'] == 'excursion' for e in events),
               'flex': sum(e['kind'] == 'flex' for e in events),
               'knocks_beyond_tolerance': sum(bool(e.get('beyond_tolerance')) for e in knocks),
               'worst_knock_deg': max((max(abs(v) for a, v in e['size'].items() if a in AXES) for e in knocks
                                       if any(a in AXES for a in e['size'])), default=None),
               'covariates': sorted(cands), 'motion': motion.label, 'scan_seconds': scan.seconds}
    res = OnlineCalibResult(st, path, scan.source, var.name, var.uid, lid.tag if lid is not None else None,
                            {'start_ns': int(scan.start_ns), 'end_ns': int(scan.end_ns), 'duration_s': duration},
                            sensors, windows, series, _models_doc(models, lay), segments, events, findings, metrics,
                            [{'a': p_['a'], 'b': p_['b'], 'cell_m': p_['cell'], 'area_m2': p_['area_m2'],
                              'patches': len(p_['blocks'])} for p_ in pairs], notes, mapping,
                            {tag: {'t': sorted(d), 'y': [d[t_] for t_ in sorted(d)]} for tag, d in frames_opt.items()})
    res.seconds = time.perf_counter() - t_start
    res.metrics['seconds'] = res.seconds
    res._scan = scan
    if progress is not None:
        progress(1.0, 'done')
    return res


def _lidar_ctx(project, lid, rev, scan, dls) -> dict:
    m = project.sensor_model_for(lid)
    topic = rev[lid.tag]
    lts = scan.topics[topic]
    return {'topic': topic, 'T_s': 1.0 / m.rate_hz if m.rate_hz > 0 else 0.1,
            'ref': dls.lidar_stamp if dls.lidar_stamp != 'auto' else lts.stamp_ref,
            'R': rot_zyx(lid.yaw, lid.pitch, lid.roll), 'p': lid.pos(), 'sense': 1 if m.rotation == 'ccw' else -1,
            'rate': m.rate_hz if m.rate_hz > 0 else 10.0}


def _window_plane(sub, lctx, motion, rng, base_ns):
    """The window's ground plane from the lidar points of its instants (vehicle frame, at each instant), and those
    points for the obstacle mask (per instant). Flat z = 0 without a lidar or with too few road points."""
    obst = {}
    if lctx is None:
        return (np.array([0.0, 0.0, 1.0]), 0.0), {'normal': [0.0, 0.0, 1.0], 'height_m': 0.0, 'source': 'flat'}, obst
    Ps = []
    for i, tg in enumerate(sub.targets):
        got = DL._frame_points(tg, lctx['topic'], tg['t_ns'], lctx['T_s'], lctx['ref'], lctx['R'], lctx['p'], motion,
                               base_ns, lctx['sense'], lctx['rate'])
        if got is None:
            continue
        E, K, _ = got
        P = np.vstack([K, E]) if len(E) else K
        obst[i] = (tg['t_ns'], P)
        Ps.append(K)
    P = np.vstack(Ps) if Ps else np.zeros((0, 3))
    fit = fit_ground_plane(P, rng)
    if fit is None:
        return (np.array([0.0, 0.0, 1.0]), 0.0), {'normal': [0.0, 0.0, 1.0], 'height_m': 0.0, 'source': 'flat'}, obst
    nrm, d, n_in = fit
    return (nrm, d), {'normal': _rl(nrm, 5), 'height_m': _r(-d / nrm[2], 4), 'points': n_in, 'source': 'lidar',
                      'tilt_deg': _r(math.degrees(math.acos(min(1.0, float(nrm[2])))), 3)}, obst


def _edge_measurement(project, var, cam, rev, lid, sub, motion, dls, conv_ms, engine, lay, rows, fo):
    if not sub.targets:
        return None
    ov = {}
    try:
        r = DL._extrinsic(project, var, cam, rev[cam.tag], lid, rev[lid.tag], sub, motion, dls, conv_ms, engine, ov)
    except (ValueError, np.linalg.LinAlgError, IndexError) as e:
        return {'kind': 'edges', 'sensor': cam.tag, 'constrains': [], 'note': f'{type(e).__name__}: {e}'}
    doc = {'kind': 'edges', 'sensor': cam.tag, 'relative_to': lid.tag, 'frames': r.get('frames', 0),
           'points': r.get('points', 0), 'confidence': r.get('confidence'), 'constrains': [],
           'delta_deg': None, 'sigma_deg': None, 'note': r.get('reason') or ''}
    if r.get('delta_deg') is None:
        return doc
    d = np.asarray(r['delta_deg'], float)
    se = np.array([v if v is not None else np.inf for v in r['stderr_deg']], float)
    obs = np.asarray(r.get('observable') or [False] * 3, bool)
    split = r.get('split')
    if split is not None:
        se = se * (3.0 if split['share'] >= DL.SPLIT_EVEN else 1.5)
    sig = np.where(obs & np.isfinite(se), np.maximum(se, 0.005), np.inf)
    doc.update({'delta_deg': _rl(d), 'sigma_deg': [(_r(v) if np.isfinite(v) else None) for v in sig],
                'constrains': [a for a, o in zip(AXES, np.isfinite(sig)) if o]})
    if np.isfinite(sig).any():
        H = np.zeros((3, lay.n))
        H[np.arange(3), lay.rot[cam.tag]] = 1.0
        gid = rows.add(H, d, sig)
        doc['_groups'] = [gid] if gid is not None else []
    per = r.get('per_frame_deg') or []
    frs = (ov.get(cam.uid) or {}).get('frames') or []
    if len(per) == len(frs):
        for f_, y_ in zip(frs, per):
            fo[round(float(f_['t_s']), 4)] = [float(v) for v in y_]
    return doc


def _outline_measurement(project, cam, rev, sub, motion, abst, engine, lay, rows, Jc, JL):
    from . import asbuilt as AB
    topic = rev[cam.tag]
    smps = [tg['images'][topic] for tg in sub.targets if topic in tg['images']]
    doc = {'kind': 'outline', 'sensor': cam.tag, 'relative_to': 'vehicle body', 'frames': len(smps),
           'constrains': [], 'delta_deg': None, 'sigma_deg': None, 'note': ''}
    if len(smps) < AB.MIN_FRAMES:
        doc['note'] = f'{len(smps)} frames in the window (the outline needs {AB.MIN_FRAMES})'
        return doc
    smps.sort(key=lambda s_: s_['stamp_ns'])
    ts = [(s['stamp_ns'] - sub.start_ns) / 1e9 for s in smps]
    if motion.kind == 'odometry':                     # the ego mask needs the world to move past: frames kept only
        need = max(abst.min_motion_m, 0.5)            # once the vehicle moved since the last kept one
        keep, last, total = [], None, 0.0
        for s_, t_ in zip(smps, ts):
            pos = motion._pose(t_)[1][:2]
            if last is None or float(np.linalg.norm(pos - last)) >= need:
                total += 0.0 if last is None else float(np.linalg.norm(pos - last))
                keep.append((s_, t_))
                last = pos
        if len(keep) < AB.MIN_FRAMES or total < OUTLINE_TRAVEL_M:
            doc['note'] = (f'{len(keep)} frames {need:g} m apart over {total:.1f} m of travel in the window (the ego '
                           f'mask needs {AB.MIN_FRAMES} and {OUTLINE_TRAVEL_M:g} m)')
            return doc
        smps, ts = [k_[0] for k_ in keep], [k_[1] for k_ in keep]
        doc['frames'] = len(smps)
    ss = AB.StreamScan(topic, 'image')
    for s, t in zip(smps, ts):
        g = _decode_grey(s)
        if g is not None:
            ss.frames.append(AB.Frame(float(t), g, tuple(s['size']), None))
    try:
        row = AB.check_camera(project, cam, ss, abst, engine)
    except (ValueError, np.linalg.LinAlgError, IndexError) as e:
        doc['note'] = f'{type(e).__name__}: {e}'
        return doc
    fit = row.get('fit')
    if not fit or not fit.get('converged'):
        doc['note'] = row.get('reason') or 'no outline fit'
        return doc
    d = np.array([v if v is not None else 0.0 for v in fit['delta_deg']], float)
    se = np.array([v if v is not None else np.inf for v in fit['stderr_deg']], float)
    con = np.asarray(fit['constrained'], bool) & np.isfinite(se)
    sig = np.where(con, np.maximum(se, 0.005), np.inf)
    doc.update({'delta_deg': _rl(d), 'sigma_deg': [(_r(v) if np.isfinite(v) else None) for v in sig],
                'constrains': [a for a, o in zip(AXES, con) if o], 'points': fit.get('points'),
                'note': row.get('reason') or ''})
    gids = []
    if con.any():
        H = np.zeros((3, lay.n))
        H[np.arange(3), lay.rot[cam.tag]] = 1.0
        if lay.lidar is not None:
            H[:, lay.rot[lay.lidar]] = np.linalg.solve(Jc[cam.tag], JL)
        gid = rows.add(H, d, sig)
        if gid is not None:
            gids.append(gid)
    if cam.tag in lay.trans and fit.get('shift_m') is not None:
        dt_ = np.array([v if v is not None else 0.0 for v in fit['shift_m']], float)
        se_t = np.array([v if v is not None else np.inf for v in fit['stderr_m']], float)
        ct = np.asarray(fit['constrained_t'], bool) & np.isfinite(se_t)
        sig_t = np.where(ct, np.maximum(se_t, 1e-4), np.inf)
        doc.update({'shift_m': _rl(dt_, 5), 'sigma_m': [(_r(v, 5) if np.isfinite(v) else None) for v in sig_t]})
        doc['constrains'] += [a for a, o in zip(TAXES, ct) if o]
        if ct.any():
            H = np.zeros((3, lay.n))
            H[np.arange(3), lay.trans[cam.tag]] = 1.0
            gid = rows.add(H, dt_, sig_t)
            if gid is not None:
                gids.append(gid)
    doc['_groups'] = gids
    return doc


def _ground_measurement(pr, geos, rev, sub, motion, plane, obst, conv_ms, base_ns, lay, rows, search_deg):
    a, b = pr['a'], pr['b']
    A, B = geos[a], geos[b]
    ta_, tb_ = rev[a], rev[b]
    doc = {'kind': 'ground', 'sensors': [a, b], 'relative_to': 'the lidar\'s ground plane' if lay.reference else
           'the flat ground', 'patches': 0, 'frames': 0, 'constrains': [], 'note': ''}
    gids, Hs, pend = [], [], []
    for i, tg in enumerate(sub.targets):
        sa, sb = tg['images'].get(ta_), tg['images'].get(tb_)
        if sa is None or sb is None:
            continue
        ia, ib = _decode_grey(sa), _decode_grey(sb)
        if ia is None or ib is None:
            continue
        t_a = (sa['stamp_ns'] + conv_ms * 1e6 - base_ns) / 1e9
        t_b = (sb['stamp_ns'] + conv_ms * 1e6 - base_ns) / 1e9
        P = None
        if i in obst:
            t_l, P = obst[i]
            R, p = _rel_motion(motion, (t_l - base_ns) / 1e9, t_a)
            P = (P - p) @ R
        got = ground_factor(pr, A, B, ia, ib, t_a, t_b, plane, motion, P, search_deg)
        if got:
            doc['frames'] += 1
        for g, dlt, sig, ncc, JA, JB, V in got:
            H = np.zeros((2, lay.n))
            H[:, lay.rot[a]] = JA
            H[:, lay.rot[b]] = -JB
            pend.append((V.T @ H, V.T @ dlt, sig))        # whitened along the peak's principal directions
    # the patches of one pair and window share their systematic errors (the road's shape against the plane, the lens
    # model far off its axis, the cameras' timing at speed): one factor, one robust weight, its information capped at
    # IPM_EFF patches' worth (heuristic)
    inflate = math.sqrt(len(pend) / IPM_EFF) if len(pend) > IPM_EFF else 1.0
    gid = None
    for Hh, zz, sig in pend:
        got_g = rows.add(Hh, zz, sig * inflate, gid)
        if got_g is None:
            continue
        gid = got_g
        gids.append(gid)
        ok = np.isfinite(sig)
        Hs.append(Hh[ok] / (sig[ok, None] * inflate))
    doc['patches'] = len(gids)
    gids = sorted(set(gids))
    if not gids:
        doc['note'] = 'no ground patch matched (plain road, obstacles or no overlap in these frames)'
        return doc
    Hm = np.vstack(Hs)
    idx = np.r_[lay.rot[a], lay.rot[b]]
    Inf = Hm[:, idx].T @ Hm[:, idx]
    cond = 1.0 / np.sqrt(np.maximum(np.diag(Inf), 1e-12))          # σ of an axis with the others known
    ev, V = np.linalg.eigh(Inf)
    names = [f'{a} {x}' for x in AXES] + [f'{b} {x}' for x in AXES]
    dirs = []
    for j in np.argsort(-ev):
        if ev[j] <= 0 or 1.0 / math.sqrt(ev[j]) > 0.5:
            continue
        v = V[:, j]
        top = [(names[i], float(v[i])) for i in np.argsort(-np.abs(v)) if abs(v[i]) >= 0.3]
        dirs.append({'sigma_deg': _r(1.0 / math.sqrt(ev[j]), 4),
                     'combination': ' '.join(f'{"+" if w > 0 else "−"}{abs(w):.2f}·{n}' for n, w in top)})
    doc.update({'constrains': [names[i] for i in range(6) if cond[i] < 0.5],
                'conditional_sigma_deg': {names[i]: _r(cond[i], 4) for i in range(6)}, 'directions': dirs,
                '_groups': gids})
    return doc


def _nothing(win, meas, lay, st, lid, pairs, has_body) -> dict:
    """{tag: reason} for each sensor the window constrains on no axis."""
    out = {}
    for tag in lay.sensors():
        if win['constrained'].get(tag):
            continue
        if tag == lay.lidar:
            out[tag] = 'the lidar\'s own rotation needs a camera whose outline and depth edges both fit in the window'
            continue
        why = []
        if not win['frames']:
            why.append('no sampled frame')
        else:
            e = next((m for m in meas if m['kind'] == 'edges' and m['sensor'] == tag), None)
            if st.edges and lid is not None:
                why.append(f'edges: {e["note"] or "no axis pinned down"}' if e is not None else 'edges: not measured')
            o = next((m for m in meas if m['kind'] == 'outline' and m['sensor'] == tag), None)
            if st.outline and has_body:
                why.append(f'outline: {o["note"] or "no axis constrained"}' if o is not None else
                           'outline: not measured')
            if st.ground:
                if not any(tag in (p_['a'], p_['b']) for p_ in pairs):
                    why.append('ground: no other camera overlaps it on the ground')
                else:
                    gm = [m for m in meas if m['kind'] == 'ground' and tag in m['sensors']]
                    why.append('ground: ' + (gm[0]['note'] if gm and gm[0]['note'] else 'too few patches'))
        out[tag] = '; '.join(why) or 'no measurement is switched on'
    return out


# =========================================================================== events and segments
def _events(models, lay, frames_opt, t_mid, st, windows, cands, cams, lid) -> list:
    raw = []
    tol = st.tolerance_deg
    for tag, axes in models.items():
        for axis, mdl in axes.items():
            unit = 'm' if axis in TAXES else '°'
            min_step = st.min_step_deg if unit == '°' else st.min_step_deg * PRIOR_M / PRIOR_DEG
            steps = [s_ for s_ in mdl['steps'] if abs(s_['size']) >= min_step and abs(s_['size']) > 3 * s_['sigma']]
            tt = t_mid[mdl['idx']]
            beta = mdl['beta']
            ai = list(AXES + TAXES).index(axis) % 3
            fo = frames_opt.get(tag, {})
            ft = np.array(sorted(fo)) if fo and axis in AXES else np.zeros(0)
            fy = np.array([fo[t_][ai] for t_ in ft]) if len(ft) else np.zeros(0)
            levels = []
            for s_ in mdl['steps']:
                before = beta[0] + sum(o['size'] for o in mdl['steps'] if o['j'] < s_['j'])
                after = before + s_['size']
                levels.append((s_, before, after))
            used = set()
            for i, (s_, before, after) in enumerate(levels):
                if s_ not in steps or id(s_) in used:
                    continue
                j = s_['j']
                lo = tt[max(j - 1, 0)] - 0.5 * st.window_s
                hi = tt[min(j, len(tt) - 1)] + 0.5 * st.window_s
                tau, resn = refine_step(ft, fy, s_['tau'], before, after, lo, hi) if len(ft) else (s_['tau'], None)
                ev = {'sensor': tag, 'axis': axis, 'tau': tau, 'res': resn if resn is not None else
                      float(np.median(np.diff(t_mid))) if len(t_mid) > 1 else st.window_s,
                      'size': s_['size'], 'sigma': s_['sigma'], 'before': before, 'after': after, 'unit': unit}
                back = None                              # a later step back to the level before: an excursion
                for s2, b2, a2 in levels[i + 1:]:
                    if s2 not in steps or np.sign(s2['size']) == np.sign(s_['size']):
                        break
                    if abs(a2 - before) <= max(2 * math.hypot(s_['sigma'], s2['sigma']), min_step):
                        back = (s2, b2, a2)
                    break
                if back is not None:
                    s2 = back[0]
                    used.add(id(s2))
                    j2 = s2['j']
                    lo2 = tt[max(j2 - 1, 0)] - 0.5 * st.window_s
                    hi2 = tt[min(j2, len(tt) - 1)] + 0.5 * st.window_s
                    tau2, _ = refine_step(ft, fy, s2['tau'], back[1], back[2], lo2, hi2) if len(ft) else \
                        (s2['tau'], None)
                    ev.update({'kind': 'excursion', 'tau_end': tau2})
                else:
                    ev['kind'] = 'knock'
                raw.append(ev)
            for fx in mdl['flex']:
                v = cands[fx['cov']]
                lo_, hi_ = np.percentile(v, [5, 95])
                amp = 0.5 * abs(fx['slope']) * (hi_ - lo_)
                min_amp = st.flex_min_deg if unit == '°' else st.flex_min_deg * PRIOR_M / PRIOR_DEG
                if amp < min_amp or abs(fx['slope']) <= 3 * fx['sigma']:
                    continue
                raw.append({'kind': 'flex', 'sensor': tag, 'axis': axis, 'cov': fx['cov'], 'slope': fx['slope'],
                            'sigma': fx['sigma'], 'amplitude': amp, 'range': (float(lo_), float(hi_)), 'unit': unit})
    uid = {c.tag: c.uid for c in cams}
    if lid is not None:
        uid[lid.tag] = lid.uid
    rel = lid.tag if lid is not None else 'the vehicle body'
    events = []
    for kind in ('knock', 'excursion'):                  # steps of one sensor at one time: one event
        evs = sorted([e for e in raw if e['kind'] == kind], key=lambda e: (e['sensor'], e['tau']))
        groups = []
        for e in evs:
            gap = max(e['res'], st.stride_s)
            if groups and groups[-1][0]['sensor'] == e['sensor'] and abs(groups[-1][0]['tau'] - e['tau']) <= gap and \
                    all(x['axis'] != e['axis'] for x in groups[-1]):
                groups[-1].append(e)
            else:
                groups.append([e])
        for grp in groups:
            lead = max(grp, key=lambda e: abs(e['size']) / max(e['sigma'], 1e-9))
            tag = lead['sensor']
            rot = [e for e in grp if e['axis'] in AXES]
            beyond = any(abs(e['size']) > tol for e in rot)
            parts = ' and '.join(f'{VERB[e["axis"]]} {_fmt_v(e["size"], e["unit"])}' for e in grp)
            who = tag if tag != (lid.tag if lid is not None else None) else f'the lidar {tag}'
            if kind == 'knock':
                msg = (f'{who} {parts} at {clock(lead["tau"])} and stayed (knock): '
                       + '; '.join(f'{e["axis"]} {_fmt_v(e["before"], e["unit"])} → {_fmt_v(e["after"], e["unit"])} '
                                   f'± {_fmt_abs(e["sigma"], e["unit"])}' for e in grp)
                       + (f', relative to {rel}' if tag != (lid.tag if lid is not None else None) else
                          ', relative to the vehicle body') + '.')
            else:
                t_end = lead['tau_end']
                msg = (f'{who} {parts} at {clock(lead["tau"])} and returned at {clock(t_end)} (a temporary shift, '
                       f'not a knock): ' + '; '.join(f'{e["axis"]} ± {_fmt_abs(e["sigma"], e["unit"])}' for e in grp)
                       + '.')
            ev = {'kind': kind, 'sensor': tag, 'uid': uid.get(tag, ''), 'axes': [e['axis'] for e in grp],
                  't_s': float(lead['tau']), 'clock': clock(lead['tau']), 'resolution_s': float(lead['res']),
                  'size': {e['axis']: float(e['size']) for e in grp}, 'sigma': {e['axis']: float(e['sigma'])
                                                                              for e in grp},
                  'before': {e['axis']: float(e['before']) for e in grp},
                  'after': {e['axis']: float(e['after']) for e in grp},
                  'beyond_tolerance': bool(beyond and kind == 'knock'), 'relative_to': rel, 'message': msg,
                  '_taus': [(e['axis'], lead['tau']) for e in grp] +       # one time for the event's axes
                           [(e['axis'], lead['tau_end']) for e in grp if kind == 'excursion']}
            if kind == 'excursion':
                ev['t_end_s'] = float(lead['tau_end'])
                ev['clock_end'] = clock(ev['t_end_s'])
            events.append(ev)
    for e in [x for x in raw if x['kind'] == 'flex']:
        tag = e['sensor']
        unit = e['unit']
        per = f'{e["slope"]:+.4f}{unit if unit == "°" else " m"} per {COV_UNIT[e["cov"]]}'
        lo_, hi_ = e['range']
        rest = ('it returns to its baseline at rest' if e['cov'] in ('speed', 'accel', 'yaw_rate') else
                'it follows the time of day')
        msg = (f'{tag if tag != (lid.tag if lid is not None else None) else "the lidar " + tag} {e["axis"]} varies '
               f'±{_fmt_abs(e["amplitude"], unit)} with {COV_TEXT[e["cov"]]} (flex): {per} '
               f'± {_fmt_abs(e["sigma"], unit, 4)}, over {lo_:.1f} … {hi_:.1f} {COV_UNIT[e["cov"]]} (5–95 %); {rest}.')
        events.append({'kind': 'flex', 'sensor': tag, 'uid': uid.get(tag, ''), 'axes': [e['axis']],
                       'covariate': e['cov'], 'slope': float(e['slope']), 'slope_sigma': float(e['sigma']),
                       'slope_unit': f'{unit} per {COV_UNIT[e["cov"]]}', 'amplitude': float(e['amplitude']),
                       'covariate_range': [float(lo_), float(hi_)], 'relative_to': rel, 'message': msg,
                       't_s': None, 'clock': '', '_taus': []})
    order = {'knock': 0, 'excursion': 1, 'flex': 2}
    events.sort(key=lambda e: (order[e['kind']], e['t_s'] if e['t_s'] is not None else 0.0, e['sensor']))
    return events


def _fmt_v(v, unit, d=2):
    return f'{v * 1000:+.0f} mm' if unit == 'm' else f'{v:+.{d}f}°'


def _fmt_abs(v, unit, d=2):
    return f'{abs(v) * 1000:.0f} mm' if unit == 'm' else f'{abs(v):.{d}f}°'


def _segments(models, lay, events, duration, series) -> dict:
    """Per sensor: the pieces between its steps (knocks and the ends of excursions), each with its pose correction at
    rest (the models' levels, flex terms at zero) and σ; an axis without a model is None (not observable)."""
    out = {}
    for tag in lay.sensors():
        cuts = sorted({float(t) for e in events if e['sensor'] == tag and e['kind'] in ('knock', 'excursion')
                       for _, t in e['_taus']})
        bounds = [0.0] + [c for c in cuts if 0.0 < c < duration] + [duration]
        axes = series[tag]['axes']
        segs = []
        for i in range(len(bounds) - 1):
            t0, t1 = bounds[i], bounds[i + 1]
            tm = 0.5 * (t0 + t1)
            delta, sigma = [], []
            for ax in axes:
                mdl = models.get(tag, {}).get(ax)
                if mdl is None:
                    delta.append(None)
                    sigma.append(None)
                    continue
                c = np.zeros(len(mdl['beta']))
                c[0] = 1.0
                for s_ in mdl['steps']:                 # the model's level at the segment's middle
                    if s_['tau'] <= tm:
                        c[s_['col']] = 1.0
                delta.append(float(c @ mdl['beta']))
                sigma.append(float(math.sqrt(max(c @ mdl['cov'] @ c, 0.0))))
            segs.append({'t0': t0, 't1': t1, 'clock0': clock(t0), 'clock1': clock(t1), 'axes': axes,
                         'delta': delta, 'sigma': sigma})
        out[tag] = segs
    return out


def _models_doc(models, lay) -> dict:
    out = {}
    for tag, axes in models.items():
        out[tag] = {}
        for ax, m in axes.items():
            out[tag][ax] = {'windows': int(m['n']), 'level': _r(m['level'], 5), 'level_sigma': _r(m['level_sigma'], 5),
                            'scale': _r(m['scale'], 3),
                            'steps': [{'t_s': _r(s_['tau'], 3), 'size': _r(s_['size'], 5), 'sigma': _r(s_['sigma'], 5)}
                                      for s_ in m['steps']],
                            'flex': [{'covariate': f['cov'], 'slope': _r(f['slope'], 6), 'sigma': _r(f['sigma'], 6)}
                                     for f in m['flex']],
                            'terms': [k if k == 'level' else f'{k}:{j}' for k, j in m['terms']],
                            'bic': [_r(b, 3) for b in m['bic']], '_m': m}
    return out


def _findings(windows, events, series, lay, st, cams, lid, pairs, duration) -> list:
    out = []
    uid = {c.tag: c.uid for c in cams}
    if lid is not None:
        uid[lid.tag] = lid.uid
    for e in events:
        if e['kind'] == 'knock':
            sev = 'serious' if e['beyond_tolerance'] else 'warning'
            tail = (f' Beyond the {st.tolerance_deg:g}° tolerance: recalibrate, or use the segments\' extrinsics.'
                    if e['beyond_tolerance'] else f' Within the {st.tolerance_deg:g}° tolerance.')
            _finding(out, sev, 'knock', e['message'] + tail, [e['uid']])
        elif e['kind'] == 'excursion':
            _finding(out, 'warning', 'excursion', e['message'], [e['uid']])
        else:
            _finding(out, 'warning' if e['amplitude'] > st.tolerance_deg / 2 else 'note', 'flex', e['message'],
                     [e['uid']])
    for tag, s in series.items():
        never = [a for a, o in zip(s['axes'], s['observable']) if not o]
        if tag == lay.lidar:
            if len(never) == len(s['axes']):
                _finding(out, 'note', 'observability', f'The lidar {tag}\'s own rotation against the body is not '
                                                       'observable on this drive (no camera\'s outline and depth edges '
                                                       'met in a window): the cameras are relative to it.', [uid[tag]])
            continue
        empty = sum(1 for w in windows if not w['constrained'].get(tag))
        if never:
            _finding(out, 'note', 'observability',
                     f'{tag}: {", ".join(never)} constrained in no window: held at the plan (with the prior\'s '
                     f'±{PRIOR_DEG:g}°).', [uid.get(tag, '')])
        if empty:
            reasons = {}
            for w in windows:
                r = w['nothing'].get(tag)
                if r:
                    reasons[r] = reasons.get(r, 0) + 1
            top = max(reasons, key=reasons.get) if reasons else ''
            _finding(out, 'note', 'observability',
                     f'{tag}: {empty} of {len(windows)} windows constrain nothing ({top}).', [uid.get(tag, '')])
    if not any(e['kind'] in ('knock', 'excursion', 'flex') for e in events):
        obs = [t for t, s in series.items() if any(s['observable']) and t != lay.lidar]
        if obs:
            _finding(out, 'good', 'stable', f'{", ".join(obs)}: no step and no flex over the {clock(duration)} drive '
                                            f'(steps of {st.min_step_deg:g}° or more and flex of '
                                            f'±{st.flex_min_deg:g}° would show).')
    if st.ground and not pairs:
        _finding(out, 'note', 'ground', 'No two cameras overlap on the ground within '
                                        f'{st.ground_range_m:g} m: no ground-plane factor.')
    order = {s: i for i, s in enumerate(('critical', 'serious', 'warning', 'note', 'good'))}
    out.sort(key=lambda f: order.get(f['severity'], 9))
    return out


# =========================================================================== lookups, the corrected variant
def delta_at(res: OnlineCalibResult, tag, t_s):
    """(correction, σ) of a sensor at t_s from the interpolated series: linear between window centres, held before the
    first and after the last, and a jump at the sensor's steps (never interpolated across one)."""
    s = res.series[tag]
    t = np.asarray(s['t'], float)
    E, S = np.asarray(s['est'], float), np.asarray(s['sigma'], float)
    if len(t) == 1:
        return E[0], S[0]
    cuts = sorted(float(x) for e in res.events if e['sensor'] == tag and e['kind'] in ('knock', 'excursion')
                  for x in ([e['t_s']] + ([e['t_end_s']] if e.get('t_end_s') is not None else [])))
    k = int(np.searchsorted(t, t_s))
    if k <= 0:
        return E[0], S[0]
    if k >= len(t):
        return E[-1], S[-1]
    if any(t[k - 1] < c <= t[k] for c in cuts):
        c = next(c for c in cuts if t[k - 1] < c <= t[k])
        return (E[k - 1], S[k - 1]) if t_s < c else (E[k], S[k])
    a = (t_s - t[k - 1]) / max(t[k] - t[k - 1], 1e-9)
    return E[k - 1] * (1 - a) + E[k] * a, S[k - 1] * (1 - a) + S[k] * a


def plot_series(res: OnlineCalibResult, tag):
    """(t, estimates, σ) of a sensor for drawing: the window centres plus, at each of its steps, the values just before
    and just after it (so a plot shows the jump where it happened, not a ramp between window centres)."""
    s = res.series[tag]
    t = list(s['t'])
    cuts = sorted({float(x) for e in res.events if e['sensor'] == tag and e['kind'] in ('knock', 'excursion')
                   for x in [e['t_s']] + ([e['t_end_s']] if e.get('t_end_s') is not None else [])})
    pts = [(tt, 0) for tt in t] + [(c - 1e-6, -1) for c in cuts] + [(c, 1) for c in cuts]
    pts.sort()
    T, E, S = [], [], []
    for tt, _ in pts:
        d, sg = delta_at(res, tag, tt)
        T.append(tt)
        E.append(np.asarray(d, float))
        S.append(np.asarray(sg, float))
    return np.asarray(T), np.vstack(E), np.vstack(S)


def final_corrections(res: OnlineCalibResult, precise=True) -> dict:
    """{tag: (delta yaw/pitch/roll deg, shift m or None)} of the last segment, on the axes measured to within the
    tolerance / 1.5 whose correction is beyond 3 σ and the smallest reported step (precise=True: the others stay as
    planned), or on every observable axis."""
    out = {}
    tol = res.settings.tolerance_deg
    for tag, segs in res.segments.items():
        if not segs:
            continue
        sg = segs[-1]
        rot, sh = [0.0, 0.0, 0.0], None
        any_ = False
        for ax, d, s in zip(sg['axes'], sg['delta'], sg['sigma']):
            if d is None or s is None:
                continue
            k = 1.0 if ax in AXES else PRIOR_M / PRIOR_DEG
            if precise and (s > tol / 1.5 * k or abs(d) <= max(3.0 * s, res.settings.min_step_deg * k)):
                continue
            if ax in AXES:
                rot[AXES.index(ax)] = d
            else:
                sh = sh or [0.0, 0.0, 0.0]
                sh[TAXES.index(ax)] = d
            any_ = True
        if any_:
            out[tag] = (rot, sh)
    return out


def corrected_variant(project, res: OnlineCalibResult, name_suffix=None):
    """A copy of the calibrated variant named '<name> · online calib from <time>' with the last segment's precise
    corrections applied (each camera relative to the lidar; the lidar's own correction when it is precise, the cameras
    then turned with it): new uids, a corrected camera and its mirror partner unlinked. None when nothing is precise
    enough or the variant is gone."""
    from .project import Variant, new_uid
    src = project.variant_by_uid(res.variant_uid)
    if src is None:
        return None
    fc = final_corrections(res)
    lidc = fc.pop(res.lidar, None) if res.lidar else None
    if not fc and lidc is None:
        return None
    v = Variant.from_dict(src.to_dict())
    v.uid = new_uid()
    seg_t = max((segs[-1]['t0'] for segs in res.segments.values() if segs), default=0.0)
    v.name = f'{src.name} · {name_suffix or "online calib from " + clock(seg_t)}'
    umap = {c.uid: new_uid() for c in v.cameras}
    for c in v.cameras:
        c.uid = umap[c.uid]
        c.link_uid = umap.get(c.link_uid, '')
    for s in v.sensors:
        s.uid = new_uid()
    v.seams = {'|'.join(umap.get(u, u) for u in k.split('|')): a for k, a in v.seams.items()}
    lid = next((s for s in v.sensors if s.tag == res.lidar), None) if res.lidar else None
    phi = np.zeros(3)
    done = []
    if lid is not None and lidc is not None:
        JLm = jac_rot((lid.yaw, lid.pitch, lid.roll))
        phi = JLm @ np.asarray(lidc[0], float)
        lid.yaw, lid.pitch, lid.roll = (round(a, 4) for a in euler_zyx(
            rot_zyx(lid.yaw + lidc[0][0], lid.pitch + lidc[0][1], lid.roll + lidc[0][2])))
        done.append(f'{lid.tag} yaw {lidc[0][0]:+.2f}°, pitch {lidc[0][1]:+.2f}°, roll {lidc[0][2]:+.2f}°')
    for c in v.cameras:
        d = fc.get(c.tag)
        if d is None and not np.any(phi):
            continue
        rot, sh = d if d is not None else ([0.0, 0.0, 0.0], None)
        R = rot_zyx(c.yaw + rot[0], c.pitch + rot[1], c.roll + rot[2])
        if np.any(phi):
            R = DL._rotvec_R(np.radians(phi)) @ R
        c.yaw, c.pitch, c.roll = (round(a, 4) for a in euler_zyx(R))
        if sh is not None:
            c.x, c.y, c.z = round(c.x + sh[0], 5), round(c.y + sh[1], 5), round(c.z + sh[2], 5)
        if c.link_uid:
            partner = next((x for x in v.cameras if x.uid == c.link_uid), None)
            if partner is not None:
                partner.link_uid = ''
            c.link_uid = ''
        if d is not None:
            txt = f'{c.tag} yaw {rot[0]:+.2f}°, pitch {rot[1]:+.2f}°, roll {rot[2]:+.2f}°'
            if sh is not None:
                txt += f', shift {sh[0] * 1000:+.0f}/{sh[1] * 1000:+.0f}/{sh[2] * 1000:+.0f} mm'
            done.append(txt)
    v.note = (f'Online self-calibration of {os.path.basename(res.path)}, the last segment (from {clock(seg_t)}), '
              f'relative to {res.lidar or "the vehicle body"}: ' + '; '.join(done) + '.')
    return v


def exit_code(res: OnlineCalibResult | None) -> int:
    """Batch exit code: 3 when a knock beyond the tolerance was found (a sensor needs a new calibration), else 0."""
    if res is None:
        return 0
    return 3 if any(e['kind'] == 'knock' and e.get('beyond_tolerance') for e in res.events) else 0


# =========================================================================== documents
def summary(res: OnlineCalibResult) -> dict:
    return {'path': os.path.basename(res.path), 'variant': res.variant, 'lidar': res.lidar,
            'duration_s': _r(res.span['duration_s'], 2), 'windows': len(res.windows),
            'knocks': [e['message'] for e in res.events if e['kind'] == 'knock'],
            'excursions': [e['message'] for e in res.events if e['kind'] == 'excursion'],
            'flex': [e['message'] for e in res.events if e['kind'] == 'flex'],
            'exit_code': exit_code(res), 'worst': res.worst()}


def _measure_doc(m) -> dict:
    return {k: v for k, v in m.items() if not k.startswith('_')}


def _segment_pose(res, project, tag, seg):
    """Absolute pose of a camera segment in the vehicle frame (the plan plus the correction relative to the lidar)."""
    s = res.sensor(tag)
    d = [0.0, 0.0, 0.0]
    sh = [0.0, 0.0, 0.0]
    for ax, v in zip(seg['axes'], seg['delta']):
        if v is None:
            continue
        if ax in AXES:
            d[AXES.index(ax)] = v
        else:
            sh[TAXES.index(ax)] = v
    R = rot_zyx(*(np.asarray(s['plan_deg'], float) + d))
    ypr = euler_zyx(R)
    pos = np.asarray(s['plan_pos'], float) + sh
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = pos
    x, y, z, w = quat_from_matrix(R)
    return {'yaw_pitch_roll_deg': _rl(ypr, 5), 'position_m': _rl(pos, 5), 'quaternion_wxyz': _rl([w, x, y, z], 9),
            'T_vehicle_body': [[_r(T[i, j], 9) for j in range(4)] for i in range(4)]}


def extrinsics_doc(res: OnlineCalibResult, project=None) -> dict:
    """The time-indexed calibration: per sensor its segments (pose at rest, σ) and its interpolated series (σ)."""
    sens = {}
    for s in res.sensors:
        tag = s['tag']
        segs = []
        for sg in res.segments.get(tag, []):
            row = {'t0_s': _r(sg['t0'], 3), 't1_s': _r(sg['t1'], 3), 'from': sg['clock0'], 'to': sg['clock1'],
                   'correction': {ax: _r(v, 5) for ax, v in zip(sg['axes'], sg['delta'])},
                   'sigma': {ax: _r(v, 5) for ax, v in zip(sg['axes'], sg['sigma'])}}
            row.update(_segment_pose(res, project, tag, sg))
            segs.append(row)
        ser = res.series.get(tag)
        sens[tag] = {'kind': s['kind'], 'topic': s['topic'], 'plan_yaw_pitch_roll_deg': _rl(s['plan_deg'], 4),
                     'plan_position_m': _rl(s['plan_pos'], 4), 'axes': s['axes'],
                     'relative_to': (res.lidar or 'the vehicle body') if s['kind'] == 'camera' else 'the vehicle body',
                     'segments': segs,
                     'interpolated': ({'t_s': _rl(ser['t'], 3), 'correction': [_rl(e, 5) for e in ser['est']],
                                       'sigma': [_rl(e, 5) for e in ser['sigma']],
                                       'observable': ser['observable']} if ser else None)}
    return {'schema': EXTRINSICS_SCHEMA, 'drive': os.path.basename(res.path), 'variant': res.variant,
            'reference': res.lidar, 'start_ns': res.span['start_ns'], 'duration_s': _r(res.span['duration_s'], 3),
            'frame': 'vehicle (ISO 8855, rear axle): x forward, y left, z up; a camera body\'s x forward',
            'units': {'angle': 'deg (yaw, pitch, roll: intrinsic Z-Y-X, pitch positive down)', 'position': 'm',
                      'time': 's after the drive\'s first message (start_ns + t_s · 1e9 = the log clock, ns)'},
            'correction': 'added to the plan\'s yaw, pitch, roll (deg) or position (m); a camera\'s is relative to '
                          'the reference sensor (the lidar), whose own is relative to the vehicle body',
            'interpolation': 'linear between window centres, held before the first and after the last; a jump at '
                             'each step (segments\' bounds), never interpolated across one',
            'segments_note': 'a segment\'s pose is the pose at rest: flex terms at zero acceleration and yaw rate, at '
                             'zero speed (or the slowest speed driven when the drive never stops) and at the drive\'s '
                             'start time of day',
            'sensors': sens, 'downstream': DOWNSTREAM}


def to_doc(res: OnlineCalibResult, project=None) -> dict:
    """The JSON export: settings with every default, the windows and their measurements (with the axes they
    constrain), the per-sensor series, models, segments and events, the findings, the limits."""
    wins = []
    for w in res.windows:
        wins.append({'t0_s': _r(w['t0'], 3), 't1_s': _r(w['t1'], 3), 't_s': _r(w['t'], 3), 'frames': w['frames'],
                     'covariates': {k: _r(v, 4) for k, v in (w.get('covariates') or {}).items()},
                     'ground_plane': w.get('plane'),
                     'measurements': [_measure_doc(m) for m in w['measurements']],
                     'constrained': w['constrained'], 'constrains_nothing': w['nothing']})
    models = {tag: {ax: {k: v for k, v in m.items() if k != '_m'} for ax, m in axes.items()}
              for tag, axes in res.models.items()}
    return {
        'schema': SCHEMA, 'method_revision': METHOD_REVISION, 'project': getattr(project, 'name', ''),
        'drive': os.path.basename(res.path), 'source': res.source, 'variant': res.variant, 'reference': res.lidar,
        'frame': 'vehicle (ISO 8855, rear axle)',
        'units': {'angle': 'deg', 'position': 'm', 'time': 's after the drive\'s first message', 'speed': 'm/s',
                  'accel': 'm/s²', 'yaw_rate': '°/s', 'time_of_day': 'h (UTC)'},
        'settings': res.settings.to_dict(), 'limits': ASSUMPTIONS, 'downstream': DOWNSTREAM,
        'measurement_kinds': {k: KIND_TEXT[k] for k in KINDS},
        'span': res.span, 'mapping': res.mapping, 'ground_pairs': res.pairs, 'windows': wins, 'models': models,
        'events': [{k: v for k, v in e.items() if not k.startswith('_')} for e in res.events],
        'extrinsics': extrinsics_doc(res, project),
        'findings': [{'severity': f['severity'], 'kind': f['kind'], 'message': f['message']} for f in res.findings],
        'metrics': res.metrics, 'notes': res.notes, 'exit_code': exit_code(res)}


CSV_COLUMNS = ('t_s', 'sensor', 'axis', 'estimate', 'sigma', 'window_estimate', 'window_sigma', 'constrained',
               'speed', 'accel', 'yaw_rate')


def to_csv(res: OnlineCalibResult) -> str:
    """One row per window, sensor and axis: the smoothed estimate and its σ, the window's own estimate and σ."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator='\n')
    w.writerow(CSV_COLUMNS)
    for tag, s in res.series.items():
        for k, t in enumerate(s['t']):
            cv = res.windows[k].get('covariates') or {}
            for i, ax in enumerate(s['axes']):
                w.writerow([_r(t, 3), tag, ax, _r(s['est'][k][i], 5), _r(s['sigma'][k][i], 5), _r(s['win'][k][i], 5),
                            _r(s['win_sigma'][k][i], 5), int(s['constrained'][k][i]), _r(cv.get('speed'), 3),
                            _r(cv.get('accel'), 3), _r(cv.get('yaw_rate'), 3)])
    return buf.getvalue()


CAM_COLORS = ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#008300', '#9085e9', '#e66767']


def _bgr(hex_):
    h = hex_.lstrip('#')
    return int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)


def timeline_png(res: OnlineCalibResult, width=1200) -> bytes:
    """Per sensor, yaw, pitch and roll over the drive: the smoothed estimate with its σ band, the windows' own
    estimates, the knocks (red), excursions (amber) and the axes with flex (labelled)."""
    import cv2
    tags = [s['tag'] for s in res.sensors]
    rh, head, left = 70, 30, 120
    H = head + sum(rh * 3 + 18 for _ in tags) + 30
    img = np.full((H, width, 3), (30, 29, 28), np.uint8)
    font, aa = cv2.FONT_HERSHEY_SIMPLEX, cv2.LINE_AA
    dur = max(res.span['duration_s'], 1e-6)
    x0, x1 = left, width - 16
    X = lambda s: int(round(x0 + (x1 - x0) * s / dur))          # noqa: E731
    put_text(img, f'Online self-calibration  {os.path.basename(res.path)}  {clock(dur)}  relative to '
                  f'{res.lidar or "the body"}', (12, 20), font, 0.5, (236, 235, 230), 1, aa)
    y = head
    for si, s in enumerate(res.sensors):
        tag = s['tag']
        ser = res.series[tag]
        col = _bgr(CAM_COLORS[s['slot'] % 8]) if s['kind'] == 'camera' else (184, 165, 10)
        tw = np.asarray(ser['t'], float)
        t, E, S = plot_series(res, tag)
        Wn = np.asarray(ser['win'])
        C = np.asarray(ser['constrained'], bool)
        for i, ax in enumerate(ser['axes'][:3]):
            yt, yb = y + i * rh + 4, y + (i + 1) * rh - 4
            vals = np.r_[E[:, i] + S[:, i], E[:, i] - S[:, i], Wn[C[:, i], i]]
            vals = vals[np.isfinite(vals)]
            span = max(float(np.max(np.abs(vals))) if len(vals) else 0.0, 0.2)
            span = min(span, 3.0)
            Y = lambda v: int(round(0.5 * (yt + yb) - (yb - yt) / 2 * np.clip(v / span, -1, 1)))   # noqa: E731
            cv2.rectangle(img, (x0, yt), (x1, yb), (40, 39, 38), -1)
            cv2.line(img, (x0, Y(0.0)), (x1, Y(0.0)), (70, 69, 66), 1)
            put_text(img, f'{tag} {ax}', (12, (yt + yb) // 2 + 5), font, 0.42, col, 1, aa)
            put_text(img, f'{span:.2f}', (x0 - 44, yt + 10), font, 0.32, (129, 135, 137), 1, aa)
            if len(t) > 1:
                band = np.array([[X(tt), Y(e + s_)] for tt, e, s_ in zip(t, E[:, i], S[:, i])] +
                                [[X(tt), Y(e - s_)] for tt, e, s_ in zip(t[::-1], E[::-1, i], S[::-1, i])], np.int32)
                ov = img.copy()
                cv2.fillPoly(ov, [band], tuple(int(0.5 * c) for c in col))
                cv2.addWeighted(ov, 0.5, img, 0.5, 0, img)
                cv2.polylines(img, [np.array([[X(tt), Y(e)] for tt, e in zip(t, E[:, i])], np.int32)], False, col, 1,
                              aa)
            for k in range(len(tw)):
                if C[k, i]:
                    cv2.circle(img, (X(tw[k]), Y(Wn[k, i])), 2, (200, 199, 194), -1, aa)
            for e in res.events:
                if e['sensor'] != tag or ax not in e['axes']:
                    continue
                if e['kind'] == 'knock':
                    cv2.line(img, (X(e['t_s']), yt), (X(e['t_s']), yb), (59, 59, 208), 2)
                elif e['kind'] == 'excursion':
                    cv2.rectangle(img, (X(e['t_s']), yt), (X(e['t_end_s']), yb), (25, 178, 250), 1)
                else:
                    put_text(img, f'flex: {e["covariate"]}', (x1 - 130, yt + 12), font, 0.36, (25, 178, 250), 1, aa)
        y += rh * 3 + 18
    cv2.line(img, (x0, y), (x1, y), (129, 135, 137), 1)
    step = next(s for s in (0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 30, 60, 120, 300, 600, 1800, 3600, 1e9) if dur / s <= 12)
    tt = 0.0
    while tt <= dur + 1e-9:
        cv2.line(img, (X(tt), y), (X(tt), y + 4), (129, 135, 137), 1)
        put_text(img, clock(tt), (X(tt) - 16, y + 18), font, 0.36, (129, 135, 137), 1, aa)
        tt += step
    ok, buf = cv2.imencode('.png', img)
    return buf.tobytes()


def files(res: OnlineCalibResult, project=None) -> list:
    """[(name, bytes)]: online_calib.json (everything), online_calib_extrinsics.json (the time-indexed calibration),
    online_calib_timeline.csv and online_calib_timeline.png."""
    return [('online_calib.json', (json.dumps(to_doc(res, project), indent=1, ensure_ascii=False, default=_jd) + '\n')
             .encode('utf-8')),
            ('online_calib_extrinsics.json', (json.dumps(extrinsics_doc(res, project), indent=1, ensure_ascii=False,
                                                         default=_jd) + '\n').encode('utf-8')),
            ('online_calib_timeline.csv', to_csv(res).encode('utf-8')),
            ('online_calib_timeline.png', timeline_png(res))]


def _jd(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


# =========================================================================== nuScenes and ROS 2 exports
def _tok(*parts) -> str:
    return hashlib.md5('|'.join(str(p) for p in parts).encode('utf-8')).hexdigest()


def _inside(child, parent) -> bool:
    c, p = os.path.realpath(child), os.path.realpath(parent)
    return c == p or c.startswith(p.rstrip(os.sep) + os.sep)


def export_nuscenes(res: OnlineCalibResult, project, out_dir, src=None) -> dict:
    """A new nuScenes folder in out_dir: the input's tables copied, one calibrated_sensor row per camera segment (the
    input's row of that channel with the segment's correction applied in the vehicle frame: rotation ΔR · R_row with
    ΔR = rot_zyx(plan + δ) · rot_zyx(plan)ᵀ, translation + the shift) and every camera sample_data re-pointed to its
    segment's row; the data folders (samples, sweeps, maps …) linked to the input's (symbolic links, else hard links,
    else copies). The input is never modified. Returns {'path', 'version', 'rows', 'repointed', 'linked', 'notes'}."""
    nd = DL._nus_dirs(src or res.path)
    if nd is None:
        raise ValueError('The drive is not a nuScenes folder: there is no nuScenes calibration to rewrite.')
    ver, root = nd
    out_dir = os.path.abspath(out_dir)
    if _inside(out_dir, root) or _inside(root, out_dir):
        raise ValueError('The output folder must lie outside the input nuScenes folder (the input is never modified).')
    vname = os.path.basename(ver)
    if os.path.exists(os.path.join(out_dir, vname)) and os.listdir(os.path.join(out_dir, vname)):
        raise FileExistsError(f'{os.path.join(out_dir, vname)} already holds tables.')
    tables = {}
    for n in sorted(os.listdir(ver)):
        if n.endswith('.json'):
            with open(os.path.join(ver, n), encoding='utf-8') as fh:
                tables[n[:-5]] = json.load(fh)
    cal, sens, sds = tables.get('calibrated_sensor', []), tables.get('sensor', []), tables.get('sample_data', [])
    chan = {s['token']: s.get('channel', '') for s in sens}
    cs_by = {c['token']: c for c in cal}
    topic_tag = {t: g for t, g in res.mapping.items()}
    notes, new_rows, repointed = [], [], 0
    for s in res.sensors:
        if s['kind'] != 'camera' or not s['topic']:
            continue
        tag, ch = s['tag'], s['topic']
        mine = [sd for sd in sds if chan.get(cs_by.get(sd.get('calibrated_sensor_token'), {}).get('sensor_token'))
                == ch]
        if not mine or topic_tag.get(ch) != tag:
            continue
        counts = {}
        for sd in mine:
            counts[sd['calibrated_sensor_token']] = counts.get(sd['calibrated_sensor_token'], 0) + 1
        row0 = cs_by[max(counts, key=counts.get)]
        q = row0['rotation']
        R_row = _quat_wxyz_R(q)
        R_plan = rot_zyx(*s['plan_deg']) @ BODY_TO_OPTICAL.T
        dev = math.degrees(np.linalg.norm(_rotvec(R_row @ R_plan.T)))
        if dev > 0.05:
            notes.append(f'{ch}: the input\'s calibrated_sensor differs from the plan by {dev:.2f}°; the correction '
                         '(relative to the plan) is applied to the input\'s row.')
        segs = res.segments.get(tag, [])
        toks = []
        for i, sg in enumerate(segs):
            d = [0.0, 0.0, 0.0]
            sh = [0.0, 0.0, 0.0]
            for ax, v in zip(sg['axes'], sg['delta']):
                if v is None:
                    continue
                if ax in AXES:
                    d[AXES.index(ax)] = v
                else:
                    sh[TAXES.index(ax)] = v
            p = np.asarray(s['plan_deg'], float)
            dR = rot_zyx(*(p + d)) @ rot_zyx(*p).T
            Rn = dR @ R_row
            x, y, z, w = quat_from_matrix(Rn)
            tok = _tok('vantage', 'online_calib', row0['token'], i, json.dumps(_rl(d, 6)), json.dumps(_rl(sh, 6)))
            row = dict(row0, token=tok, rotation=[round(w, 10), round(x, 10), round(y, 10), round(z, 10)],
                       translation=[round(float(a + b), 6) for a, b in zip(row0['translation'], sh)])
            row['vantage_online_calib'] = {'segment': i, 't0_s': _r(sg['t0'], 3), 't1_s': _r(sg['t1'], 3),
                                           'correction': {ax: _r(v, 5) for ax, v in zip(sg['axes'], sg['delta'])},
                                           'sigma': {ax: _r(v, 5) for ax, v in zip(sg['axes'], sg['sigma'])},
                                           'relative_to': res.lidar or 'vehicle body', 'from_row': row0['token']}
            new_rows.append(row)
            toks.append((sg['t0'], tok))
        for sd in mine:
            t_s = (int(sd['timestamp']) * 1000 - res.span['start_ns']) / 1e9
            k = max([i for i, (t0, _) in enumerate(toks) if t0 <= t_s] or [0])
            sd['calibrated_sensor_token'] = toks[k][1]
            repointed += 1
    tables['calibrated_sensor'] = cal + new_rows
    os.makedirs(os.path.join(out_dir, vname), exist_ok=True)
    for n, rows in tables.items():
        with open(os.path.join(out_dir, vname, f'{n}.json'), 'w', encoding='utf-8') as fh:
            fh.write(json.dumps(rows, indent=1) + '\n')
    linked = {}
    data_root = root if os.path.realpath(ver) != os.path.realpath(root) else None
    if data_root is not None:
        for e in sorted(os.listdir(data_root)):
            q = os.path.join(data_root, e)
            if os.path.realpath(q) == os.path.realpath(ver) or not os.path.isdir(q) or e.startswith('.'):
                continue
            dst = os.path.join(out_dir, e)
            if os.path.exists(dst):
                continue
            linked[e] = _link_tree(q, dst)
    else:
        notes.append('The tables and the data share one folder: the data files were not linked.')
    return {'path': out_dir, 'version': vname, 'rows': len(new_rows), 'repointed': repointed, 'linked': linked,
            'notes': notes}


def _quat_wxyz_R(q):
    w, x, y, z = (float(v) for v in q)
    return DL._quat_R((x, y, z, w))


def _link_tree(src, dst) -> str:
    """Make dst show src's files: a symbolic link, else hard links, else copies. Returns which."""
    try:
        os.symlink(os.path.abspath(src), dst, target_is_directory=True)
        return 'symlink'
    except (OSError, NotImplementedError):
        pass
    how = 'hardlink'
    for r, _, names in os.walk(src):
        rel = os.path.relpath(r, src)
        os.makedirs(os.path.join(dst, rel), exist_ok=True)
        for n in names:
            a, b = os.path.join(r, n), os.path.join(dst, rel, n)
            try:
                os.link(a, b)
            except OSError:
                shutil.copy2(a, b)
                how = 'copy'
    return how


def export_tf_bag(res: OnlineCalibResult, project, out_path, hz=None) -> dict:
    """A sidecar rosbag2 (MCAP) at out_path: /tf with base_link → <tag>_link of every estimated camera at hz over the
    drive (the interpolated extrinsics; at a step two stamps 1 ms apart), /tf_static with everything else of the plan
    (the other sensors' links, each camera's optical frame). Stamps on the drive's own clock. Returns the writer's
    summary."""
    from . import rosbag as RB
    from .sensors import rotation as sensor_rotation
    var = project.variant_by_uid(res.variant_uid) if project is not None else None
    if var is None:
        raise ValueError('The calibrated variant is no longer in the project.')
    hz = float(hz or res.settings.tf_hz)
    dur = res.span['duration_s']
    base = int(res.span['start_ns'])
    est = {s['tag'] for s in res.sensors if s['kind'] == 'camera'}
    times = set(np.round(np.arange(0.0, dur + 1e-9, 1.0 / hz), 6).tolist())
    times.add(round(dur, 6))
    for e in res.events:
        if e['kind'] in ('knock', 'excursion'):
            for x in [e['t_s']] + ([e['t_end_s']] if e.get('t_end_s') is not None else []):
                times.update({round(max(x - 0.0005, 0.0), 6), round(min(x + 0.0005, dur), 6)})
    times = sorted(times)
    statics = []
    for c in var.cameras:
        if not c.visible:
            continue
        To = np.eye(4)
        To[:3, :3] = BODY_TO_OPTICAL.T
        statics.append(RB.transform_stamped(To, base, f'{c.tag}_link', f'{c.tag}_optical_frame'))
        if c.tag not in est:
            T = np.eye(4)
            T[:3, :3] = rot_zyx(c.yaw, c.pitch, c.roll)
            T[:3, 3] = c.pos()
            statics.append(RB.transform_stamped(T, base, 'base_link', f'{c.tag}_link'))
    for s in var.sensors:
        if not s.visible:
            continue
        T = np.eye(4)
        T[:3, :3] = sensor_rotation(s)
        T[:3, 3] = s.pos()
        statics.append(RB.transform_stamped(T, base, 'base_link', f'{s.tag}_link'))
    with RB.BagWriter(out_path, compression='none') as bw:
        bw.add_topic('/tf_static', 'tf2_msgs/msg/TFMessage')
        bw.add_topic('/tf', 'tf2_msgs/msg/TFMessage')
        bw.write('/tf_static', base, {'transforms': statics})
        cams = [s for s in res.sensors if s['kind'] == 'camera']
        for t in times:
            ns = base + int(round(t * 1e9))
            trs = []
            for s in cams:
                d, _ = delta_at(res, s['tag'], t)
                ax = res.series[s['tag']]['axes']
                rot = np.array([d[ax.index(a)] if a in ax else 0.0 for a in AXES])
                sh = np.array([d[ax.index(a)] if a in ax else 0.0 for a in TAXES])
                T = np.eye(4)
                T[:3, :3] = rot_zyx(*(np.asarray(s['plan_deg'], float) + rot))
                T[:3, 3] = np.asarray(s['plan_pos'], float) + sh
                trs.append(RB.transform_stamped(T, ns, 'base_link', f'{s["tag"]}_link'))
            bw.write('/tf', ns, {'transforms': trs})
        out = bw.close()
    out['stamps'] = len(times)
    return out


def export(res: OnlineCalibResult, project, out_dir, nuscenes=True, ros=True) -> list[str]:
    """Write files(res) into out_dir, and with a nuScenes input the rewritten nuScenes folder (out_dir/nuscenes), and
    the ROS 2 sidecar bag (out_dir/<drive>_online_tf). Returns the paths written."""
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for name, data in files(res, project):
        p = os.path.join(out_dir, name)
        with open(p, 'wb') as fh:
            fh.write(data)
        paths.append(p)
    if nuscenes and res.source == 'nuscenes':
        info = export_nuscenes(res, project, os.path.join(out_dir, 'nuscenes'))
        paths.append(info['path'])
    if ros and project is not None:
        stem = os.path.splitext(os.path.basename(res.path.rstrip(os.sep)))[0] or 'drive'
        bag = os.path.join(out_dir, f'{stem}_online_tf')
        if os.path.exists(os.path.join(bag, 'metadata.yaml')):
            shutil.rmtree(bag)
        export_tf_bag(res, project, bag)
        paths.append(bag)
    return paths


def batch(project, drive, out_dir, settings=None, progress=None) -> tuple:
    """Batch mode (--online-calib DRIVE OUT_DIR): run on drive, write export() into out_dir. Returns (result, exit
    code): 3 when a knock beyond the tolerance was found, 4 when the drive cannot be read, else 0."""
    st = settings or settings_of(project)
    try:
        res = run(project, None, st, progress=progress, path=drive)
    except (FileNotFoundError, ValueError, OSError):
        return None, 4
    if res is None:
        return None, 0
    export(res, project, out_dir)
    return res, exit_code(res)
