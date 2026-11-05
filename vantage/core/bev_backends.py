"""3D (bird's-eye) detectors for the rig-perception study (core.rig_perception): what turns one instant of a rig's
cameras into objects placed around the car, behind one small interface so a team's own detector can stand in.

The interface
-------------
``Backend.detect(frame) -> list[Detection3D]``. A Frame is one instant of one rig: every camera's image taken nearest to
that instant (and the lidar sweep's file when the recording has one), each with the recording's own calibration
(intrinsics and lens: the exact lens model the simulator used when the recording says it, else its Kannala-Brandt fit or
a pinhole; extrinsics in the vehicle frame) and the transform from the vehicle frame at the image's capture time to the
vehicle frame at the frame's reference time (ego motion: cameras on different trigger lines fire at different moments,
so a real stack compensates the ego's motion before it fuses them; so does this one). A Detection3D is a class, a
score, a centre (x, y, z in the vehicle frame at the reference time, ISO 8855: x forward, y left, z up, metres, rear
axle on the ground), a size [length, width, height], a yaw (or None) and the cameras (sources) it came from, with its
bird's-eye covariance where the backend knows one.

Backends
--------
* lifted (the default; no new dependency): YOLOX-S (core.detection, cv2.dnn) on each camera's image, or on virtual
  pinhole views cut from it (core.detection_eval.make_views), each 2D box lifted to 3D:
  - ground contact: the class's upright cylinder (CLASS_PRIORS) standing on the ground plane z = 0 of the vehicle
    frame at the azimuth of the box's centre ray (unprojected through the camera's lens model:
    core.lens.Intrinsics.unproject for the exact model) is moved along that azimuth until its projection's lowest row
    meets the box's bottom edge. On a fisheye the box's bottom-centre pixel is not the ground contact (an object's
    straight bottom edge bends; its lowest pixel is at a corner): measured on data-rig-6-cam-lidar-radars with the
    ground truth's own tight boxes, the bottom-centre ray landed a median 4.1 m short at 10–20 m and 9.2 m at
    20–40 m; the projected-cylinder fit removes that. CARLA's
    ground truth on its flat towns puts every object's box bottom on that plane (the annotated box centres stand at
    half their height: measured 0.93 m for a 1.86 m walker), so the plane is right there; on a slope, a crest or a
    dip it is not, and the range is then wrong by about range² · slope change / camera height (stated as a limit).
    The range's standard deviation is the image error over the projected row's slope with range: the box edge's
    pixel error and the pitch error σ_ε (the body's pitch on its springs plus the calibration's, PITCH_SIGMA_DEG) in
    pixels; for a pinhole that is r = h / tan(−ε), dr/dε = (r² + h²) / h, so it grows as r² · σ_ε / h;
  - size: the same cylinder's projected height against the box's (an upright object of height H at distance r spans
    about H / r radians), a second range with a relative error from the class's height spread and the box edges'
    pixel errors;
  - the two ranges combined by inverse variance (Gauss–Markov: the minimum-variance unbiased combination of two
    independent unbiased estimates); a range the box cannot give (its bottom outside the image or the lens's field, or
    no solution along the ray; its top cut by the image's edge) is left out;
  - cross-camera fusion: detections of one class from different sources (cameras, views) are associated in the
    bird's-eye view by a gated Hungarian assignment (Kuhn 1955; scipy.optimize.linear_sum_assignment) on the squared
    Mahalanobis distance of their positions (gate: the χ² quantile of 2 degrees of freedom at GATE_P, 9.21 at 99 %), one
    source after the other; a group of two or more is triangulated from its bearing rays (weighted least squares on
    the lines' normals, the camera positions and azimuths in the reference frame) when the rays cross at
    MIN_TRIANGULATION_DEG or more and the object is at least MIN_TRIANGULATION_SIZES of its own lengths away from
    every camera (nearer, two cameras see different faces of it and their box centres are not one point), which is
    free of the ground-plane and size-prior biases; otherwise the group takes its tightest member's estimate (fusion
    'best': the fullest view) or the inverse-covariance mean ('mean'). The score is the group's highest. Measured on
    the round-8 pair and the round-14 scene set (rig_perception's report): the mean pulls a good view towards a cut
    one (vehicle mAP 6.8 % against 7.8 % with 'best'), triangulating near objects costs up to 2.7 points of vehicle
    mAP, and no fusion at all keeps duplicates (precision 28 % against 47 % for pedestrians) but finds 1–5 points more
    vehicles (a duplicate is a second try).
  Honest about what it is: a monocular ground-plane lift with a size prior and a geometric fusion, not a learned 3D
  detector; its measured centre error (ATE) per region is part of every result.
* external: any team's own detector (BEVFusion, BEVFormer, …) run as a command the user sets on this computer: an
  argument list (never a shell string, never from a project file) kept in the user's own settings (QSettings, like
  core.automation's command) and run only after the user approved that exact list for that name. Vantage hands it
  the recording's nuScenes folder (the CARLA bridge writes one: samples, sweeps, calibrated_sensor, ego_pose; a
  re-shot export is one too) and a frames file (vantage.bev_frames/1: each keyframe's sample token, time, every
  camera's image, intrinsics, lens and extrinsics, the ego pose), and reads back the nuScenes detection submission
  format (Caesar et al. 2020; nuscenes-devkit eval/detection): {"results": {sample_token: [{"translation" (global,
  m), "size" [w, l, h], "rotation" (w, x, y, z), "detection_name", "detection_score", …}]}}, converted into the
  vehicle frame through the sample's ego pose. Its frames are the recording's keyframes (the samples it has tokens
  for).
* torch BEV model: none qualifies (see TORCH_NOTE). Simple-BEV (Harley et al., ICRA 2023) is pure PyTorch and takes any
  cameras' intrinsics and extrinsics, its code is MIT, its weights download (rgb_checkpoint.tar.gz, 461 749 880 bytes,
  Dropbox, read 2026-11-05), but the weights carry no licence of their own and are trained on nuScenes, whose terms
  allow non-commercial use only; every other pretrained multi-camera BEV model found is likewise trained on nuScenes,
  Waymo or Argoverse 2 (all non-commercial) or needs compiled CUDA operators. A team that holds a licence runs its own
  model through the external backend.

References: Ge et al. 2021 (YOLOX, arXiv:2107.08430); Kannala and Brandt 2006 (the lens model, IEEE TPAMI 28(8));
Kuhn 1955 (the Hungarian method, Naval Research Logistics Quarterly 2); Hartley and Zisserman 2004 (Multiple View
Geometry, ch. 12, triangulation); Caesar et al. 2020 (nuScenes: A multimodal dataset for autonomous driving, CVPR; the
detection submission format and its matching by centre distance).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field

import numpy as np

from . import detection as DT

SCHEMA_FRAMES = 'vantage.bev_frames/1'
FRAME = 'vehicle (ISO 8855, rear axle) at the frame\'s reference time'
BACKENDS = ('lifted', 'external')
BACKEND_LABEL = {'lifted': 'Lifted YOLOX (monocular ground plane + size prior, fused)',
                 'external': 'External command (nuScenes submission)'}
VERSION = 'lifted-1'                   # the lifted backend's method version (records name it)
GATE_P = 0.99
GATE_CHI2 = 9.2103                     # χ²(2 dof) at 0.99
MIN_TRIANGULATION_DEG = 4.0            # rays crossing at less than this: no triangulation
MIN_TRIANGULATION_SIZES = 8.0          # nor nearer than this many object lengths (two cameras see different faces)
PITCH_SIGMA_DEG = 0.5                  # body pitch on its springs + calibration (1 σ, typical)
YAW_SIGMA_DEG = 0.3                    # calibration yaw (1 σ, typical)
BOX_PX_SIGMA = 2.0                     # a detected box edge's error (1 σ, px; typical of COCO detectors' box noise)

# Typical sizes per class (length, width, height, σ height; metres). Typical figures, not a dataset's statistics:
# adult height from NCD-RisC (2016) eLife 5:e13410 (world means 1.71 m men, 1.59 m women, so 1.65 ± 0.1 m), a car and
# a bus from EU type-approval limits and common models (bus height up to 4 m: Directive 96/53/EC), a bicycle with its
# rider about 1.7 m tall (a 1.65 m rider seated raises little), a motorcycle with its rider about 1.5 m. CARLA's
# walkers are taller (its adult blueprints' boxes are 1.86 m: the ground truth), which a ground-contact range does not
# care about and the size range does: one reason the two are combined, not the size alone.
CLASS_PRIORS = {
    'pedestrian': (0.6, 0.6, 1.65, 0.12),
    'cyclist': (1.8, 0.6, 1.70, 0.15),
    'motorcyclist': (2.1, 0.8, 1.50, 0.15),
    'car': (4.5, 1.85, 1.50, 0.15),
    'truck': (8.0, 2.5, 3.20, 0.60),
    'bus': (11.0, 2.55, 3.20, 0.30),
}
BEV_CLASSES = tuple(CLASS_PRIORS)
# nuScenes detection names (and Vantage's own) -> Vantage's classes, for an external detector's output
NUSCENES_NAME_TO_CLASS = {'pedestrian': 'pedestrian', 'bicycle': 'cyclist', 'motorcycle': 'motorcyclist',
                          'car': 'car', 'truck': 'truck', 'bus': 'bus', 'trailer': 'truck',
                          'construction_vehicle': 'truck', 'cyclist': 'cyclist', 'motorcyclist': 'motorcyclist'}
TORCH_NOTE = (
    'No pretrained multi-camera BEV model qualifies for a built-in backend. Simple-BEV (Harley, Fang, Li, Ambrus, '
    'Fragkiadaki, Bohg: "Simple-BEV: What Really Matters for Multi-Sensor BEV Perception?", ICRA 2023; '
    'github.com/aharley/simple_bev) runs in pure PyTorch and takes any cameras\' intrinsics and extrinsics, and its '
    'code is MIT-licensed, but its released weights (get_rgb_model.sh: rgb_checkpoint.tar.gz, 461 749 880 bytes) carry '
    'no licence of their own and are trained on nuScenes, whose terms of use allow non-commercial use only. The other '
    'candidates (BEVFormer, BEVFusion, Lift-Splat-Shoot, CVT) are trained on nuScenes, Waymo Open or Argoverse 2 (all '
    'non-commercial terms), and BEVFormer and BEVFusion need compiled CUDA operators. A team with its own licensed '
    'model runs it through the external backend (an approved command reading the recording\'s nuScenes folder).')
LIMITS_TEXT = (
    'The lifted backend is monocular geometry: the ground contact assumes the flat ground plane z = 0 of the vehicle '
    'frame (right on CARLA\'s flat towns; on a slope of grade g the range is off by about r² · g / camera height), the '
    'size range assumes a typical class height (CLASS_PRIORS), and a box whose bottom is hidden (another object in '
    'front, the image\'s edge) is ranged by its size only. Fusion is geometric (gated nearest assignment), not tracking: '
    'nothing is carried from one instant to the next, so there are no velocities. Orientation (yaw) is not '
    'estimated; sizes are the class\'s typical ones.')


# =========================================================================== the data a backend sees
@dataclass
class CameraCal:
    """One camera (or virtual view) as a recording calibrates it. Image coordinates are pixel edges: pixel (i, j)
    covers [i, i+1) × [j, j+1), as the ground truth's boxes."""
    tag: str
    width: int
    height: int
    R: np.ndarray                         # optical frame -> vehicle frame
    t: np.ndarray                         # optical centre in the vehicle frame (m)
    model: str = 'pinhole'                # 'exact' (the simulator's lens block) | 'kb' | 'pinhole'
    K: np.ndarray | None = None           # 'kb' / 'pinhole': pixel-centre intrinsics (OpenCV)
    D: np.ndarray | None = None           # 'kb': k1..k4 (cv2.fisheye)
    lens: dict | None = None              # 'exact': core.simrig's lens block (projection, f_px, kb, cx, cy, …)
    max_theta: float = math.radians(89.0)
    channel: str = ''
    source: str = ''                      # the camera a view is cut from ('' = itself)

    def _intrinsics(self):
        from . import lens as LZ
        L = self.lens
        W, H = (L.get('sensor_size') or [self.width, self.height])[:2]
        return LZ.Intrinsics(width=int(W), height=int(H), pixel_mm=1.0, f_mm=float(L['f_px']),
                             projection=str(L['projection']), kb=tuple((list(L.get('kb') or []) + [0.0] * 4)[:4]),
                             cx=float(L['cx']), cy=float(L['cy']), max_theta=self.max_theta, focus_near=0.0,
                             focus_far=math.inf, aspect=float(L.get('aspect', 1.0) or 1.0))

    def rays(self, u, v) -> tuple[np.ndarray, np.ndarray]:
        """Image points (pixel edges) -> unit optical-frame rays (n × 3) and ok (inside the lens's field)."""
        u = np.asarray(u, float).reshape(-1)
        v = np.asarray(v, float).reshape(-1)
        if self.model == 'exact' and self.lens:
            W, H = (self.lens.get('sensor_size') or [self.width, self.height])[:2]
            w, h = (self.lens.get('image_size') or [self.width, self.height])[:2]
            d, th = self._intrinsics().unproject(u * (W / w), v * (H / h))
            return d / np.linalg.norm(d, axis=1, keepdims=True), th <= self.max_theta + 1e-9
        K = self.K
        mx = (u - 0.5 - K[0, 2]) / K[0, 0]
        my = (v - 0.5 - K[1, 2]) / K[1, 1]
        if self.model == 'kb' and self.D is not None:
            td = np.hypot(mx, my)
            k1, k2, k3, k4 = (float(x) for x in self.D)
            th = td.copy()
            for _ in range(20):                      # Newton on θ_d = θ (1 + k1 θ² + k2 θ⁴ + k3 θ⁶ + k4 θ⁸)
                t2 = th * th
                f = th * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4)))) - td
                df = 1 + t2 * (3 * k1 + t2 * (5 * k2 + t2 * (7 * k3 + t2 * 9 * k4)))
                th = th - f / np.where(np.abs(df) > 1e-9, df, 1e-9)
            s = np.where(td > 1e-12, np.sin(th) / np.where(td > 1e-12, td, 1.0), 1.0)
            d = np.stack([mx * s, my * s, np.cos(th)], -1)
            return d, (th >= 0) & (th <= self.max_theta)
        d = np.stack([mx, my, np.ones_like(mx)], -1)
        d = d / np.linalg.norm(d, axis=1, keepdims=True)
        return d, np.arccos(np.clip(d[:, 2], -1, 1)) <= self.max_theta

    def project(self, P_opt) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Optical-frame points (n × 3) -> image points (pixel edges) and ok."""
        P = np.asarray(P_opt, float).reshape(-1, 3)
        x, y, z = P[:, 0], P[:, 1], P[:, 2]
        rho = np.hypot(x, y)
        theta = np.arctan2(rho, z)
        if self.model == 'exact' and self.lens:
            from . import groundtruth as GT
            lens = dict(self.lens)
            lens.setdefault('max_theta_deg', math.degrees(self.max_theta))
            u, v, _th, ok = GT.project(lens, P)
            return u, v, ok & (theta <= self.max_theta)
        K = self.K
        if self.model == 'kb' and self.D is not None:
            k1, k2, k3, k4 = (float(q) for q in self.D)
            t2 = theta * theta
            td = theta * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))
            s = np.where(rho > 1e-12, td / np.where(rho > 1e-12, rho, 1.0), 0.0)
            return K[0, 0] * x * s + K[0, 2] + 0.5, K[1, 1] * y * s + K[1, 2] + 0.5, theta <= self.max_theta
        ok = (z > 1e-6) & (theta <= self.max_theta)
        zz = np.where(z > 1e-6, z, 1.0)
        return K[0, 0] * x / zz + K[0, 2] + 0.5, K[1, 1] * y / zz + K[1, 2] + 0.5, ok

    def hfov_vfov_deg(self) -> tuple[float, float]:
        """The field across the image's middle row and column (ray to ray, degrees)."""
        out = []
        for (u0, v0), (u1, v1) in (((0.5, self.height / 2), (self.width - 0.5, self.height / 2)),
                                   ((self.width / 2, 0.5), (self.width / 2, self.height - 0.5))):
            d, ok = self.rays([u0, self.width / 2, u1], [v0, self.height / 2, v1])
            a = [math.degrees(math.acos(float(np.clip(d[i] @ d[1], -1, 1)))) for i in (0, 2)]
            lim = math.degrees(self.max_theta)
            out.append(float(sum(min(x, lim) if ok[i] else lim for x, i in zip(a, (0, 2)))))
        return out[0], out[1]

    def describe(self) -> dict:
        d = {'tag': self.tag, 'channel': self.channel, 'width': int(self.width), 'height': int(self.height),
             'model': self.model, 'max_theta_deg': round(math.degrees(self.max_theta), 4),
             'R_vehicle_optical': np.round(self.R, 9).tolist(), 't_vehicle_m': np.round(self.t, 6).tolist()}
        if self.K is not None:
            d['K'] = np.round(self.K, 6).tolist()
        if self.D is not None:
            d['kb'] = [float(x) for x in self.D]
        if self.lens:
            d['lens'] = {k: v for k, v in self.lens.items() if k != 'note'}
        return d


@dataclass
class FrameImage:
    """One camera's (or view's) image of a frame."""
    cal: CameraCal
    timestamp: int                         # µs, capture
    T_ref: np.ndarray                      # 4 × 4: vehicle frame at capture -> vehicle frame at the reference time
    loader: object = None                  # () -> BGR image
    path: str = ''
    dets: dict | None = None               # 2D detections (core.detection's dict), when known already
    key: str = ''                          # cache key of the image (its path relative to the recording)

    def load(self):
        if self.loader is None:
            raise OSError(f'{self.cal.tag}: no image')
        return self.loader()


@dataclass
class Frame:
    """One instant of one rig."""
    token: str                             # a nuScenes sample token, or the recording's own id of the instant
    timestamp: int                         # µs, the reference time
    images: list                           # [FrameImage]
    lidar: str = ''                        # the lidar sweep file nearest the reference time, when there is one
    recording: str = ''
    keyframe: bool = False
    ego_pose: np.ndarray | None = None     # 4 × 4: vehicle frame at the reference time -> global


@dataclass
class Detection3D:
    cls: str
    score: float
    centre: np.ndarray                     # x, y, z (m), vehicle frame at the frame's reference time
    size: np.ndarray                       # length, width, height (m)
    yaw: float | None = None               # rad, left positive (None: not estimated)
    sources: tuple = ()
    cov: np.ndarray | None = None          # 2 × 2 bird's-eye covariance of (x, y), m²
    info: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {'class': self.cls, 'score': round(float(self.score), 5),
             'centre': [round(float(x), 4) for x in self.centre], 'size': [round(float(x), 3) for x in self.size],
             'yaw_deg': None if self.yaw is None else round(math.degrees(self.yaw), 3), 'sources': list(self.sources)}
        if self.cov is not None:
            d['sigma_xy_m'] = [round(math.sqrt(max(float(self.cov[0, 0]), 0)), 4),
                               round(math.sqrt(max(float(self.cov[1, 1]), 0)), 4)]
        d.update({k: v for k, v in self.info.items() if isinstance(v, (int, float, str, bool)) or v is None})
        return d


class Backend:
    """A 3D detector. detect(frame) -> [Detection3D]; prepare(recording) once before a recording's frames (an external
    command runs there); describe() -> what records name it by; frames: 'all' (any instant) or 'keyframes'."""
    id = 'backend'
    name = 'Backend'
    version = '1'
    frames = 'all'

    def prepare(self, recording) -> None:
        return None

    def detect(self, frame: Frame) -> list:
        raise NotImplementedError

    def settings(self) -> dict:
        return {}

    def settings_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.settings(), sort_keys=True, default=str).encode()).hexdigest()[:16]

    def describe(self) -> dict:
        return {'id': self.id, 'name': self.name, 'version': self.version, 'settings': self.settings(),
                'settings_hash': self.settings_hash()}


# =========================================================================== geometry helpers
def apply_T(T, P) -> np.ndarray:
    P = np.asarray(P, float).reshape(-1, 3)
    return P @ T[:3, :3].T + T[:3, 3]


def _yaw_of(T) -> float:
    return math.atan2(T[1, 0], T[0, 0])


def _rot2(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]])


def _cov_polar(az, sr, st):
    Rz = _rot2(az)
    return Rz @ np.diag([sr * sr, st * st]) @ Rz.T


def rad_per_px(cal: CameraCal, u, v) -> np.ndarray:
    """The angle between the rays of neighbouring pixels at (u, v) (radial, the larger of the two axes): the lens's
    local angular scale, rad per pixel."""
    u = np.asarray(u, float)
    v = np.asarray(v, float)
    d0, _ = cal.rays(u, v)
    d1, _ = cal.rays(u + 1.0, v)
    d2, _ = cal.rays(u, v + 1.0)
    a1 = np.arccos(np.clip(np.sum(d0 * d1, 1), -1, 1))
    a2 = np.arccos(np.clip(np.sum(d0 * d2, 1), -1, 1))
    return np.maximum(np.maximum(a1, a2), 1e-6)


@dataclass
class LiftSettings:
    pitch_sigma_deg: float = PITCH_SIGMA_DEG
    yaw_sigma_deg: float = YAW_SIGMA_DEG
    box_px_sigma: float = BOX_PX_SIGMA
    min_depression_deg: float = 0.3        # a ground ray less steep than this below the horizon gives no range
    edge_px: float = 2.0                   # a box edge this close to the image's edge is cut by it
    max_range_m: float = 150.0
    gate_chi2: float = GATE_CHI2
    min_triangulation_deg: float = MIN_TRIANGULATION_DEG
    min_triangulation_sizes: float = MIN_TRIANGULATION_SIZES
    fusion: str = 'best'                   # a group not triangulated: 'best' (its tightest member) | 'mean'
    fuse: bool = True
    use_size: bool = True
    use_ground: bool = True

    def to_dict(self):
        return dict(self.__dict__)


RANGE_GRID = np.geomspace(0.4, 200.0, 220)       # centre ranges tried by the fit (m)
N_RIM = 16                                        # points on each rim of the class cylinder


def _rims(cal: CameraCal, az, rad, H, ranges):
    """The class's upright cylinder (radius rad, height H, standing on z = 0) at each centre range along azimuth az
    from the camera, projected: (bottom-rim lowest image row, top-rim highest row, ok) per range (NaN where the lens
    does not see the rim)."""
    phi = np.linspace(0, 2 * np.pi, N_RIM, endpoint=False)
    cx = cal.t[0] + ranges[:, None] * math.cos(az) + rad * np.cos(phi)[None]
    cy = cal.t[1] + ranges[:, None] * math.sin(az) + rad * np.sin(phi)[None]
    n = cx.size
    P = np.concatenate([np.stack([cx.ravel(), cy.ravel(), np.zeros(n)], 1),
                        np.stack([cx.ravel(), cy.ravel(), np.full(n, H)], 1)])
    Po = (P - cal.t) @ cal.R
    u, v, ok = cal.project(Po)
    ok = ok & np.isfinite(v) & (Po[:, 2] > -1e9)
    v = np.where(ok, v, np.nan).reshape(2, len(ranges), N_RIM)
    vb = np.where(np.isnan(v[0]).any(1), np.nan, np.where(np.isnan(v[0]), -np.inf, v[0]).max(1))
    vt = np.where(np.isnan(v[1]).any(1), np.nan, np.where(np.isnan(v[1]), np.inf, v[1]).min(1))
    return vb, vt


def _solve(f, target, ranges):
    """The first range (from near to far) where f crosses target, linearly interpolated, with df/dr there; None."""
    g = f - target
    ok = np.isfinite(g[:-1]) & np.isfinite(g[1:])
    k = np.flatnonzero(ok & (np.sign(g[:-1]) != np.sign(g[1:])) & (g[:-1] != 0))
    if not len(k):
        return None
    i = int(k[0])
    a = g[i] / (g[i] - g[i + 1])
    r = ranges[i] + a * (ranges[i + 1] - ranges[i])
    dfdr = (f[i + 1] - f[i]) / (ranges[i + 1] - ranges[i])
    return float(r), float(dfdr)


def lift_boxes(boxes, classes, scores, cal: CameraCal, T_ref=None, st: LiftSettings | None = None) -> list:
    """2D boxes (n × 4, pixel edges) of one image with their Vantage classes (names) and scores -> monocular
    Detection3D, in the vehicle frame at the reference time (T_ref: capture -> reference; None = the same). Classes
    without a 3D prior (traffic lights, signs) are left out.

    The object is the class's upright cylinder (radius (length + width) / 4, its typical height) standing on z = 0 at
    the azimuth of the box's centre ray; its centre range is solved twice through the camera's own projection (exact on
    a fisheye, where the box's bottom-centre pixel is not the object's ground contact: a straight edge bends): so the
    projected cylinder's lowest row meets the box's bottom (ground contact), and so its projected height meets the
    box's (size). Each range's σ is the image error over the slope of the projected row with range: the box edge's
    BOX_PX_SIGMA with the pitch σ in pixels for the ground (σ_r ≈ r² · σ_pitch / h far away), the two edges' errors
    and the class's height spread for the size."""
    st = st or LiftSettings()
    T = np.eye(4) if T_ref is None else np.asarray(T_ref, float)
    boxes = np.asarray(boxes, float).reshape(-1, 4)
    out = []
    if not len(boxes):
        return out
    uc = (boxes[:, 0] + boxes[:, 2]) / 2.0
    vc = (boxes[:, 1] + boxes[:, 3]) / 2.0
    dc, okc = cal.rays(uc, vc)
    dcv = dc @ cal.R.T
    rpp = rad_per_px(cal, uc, np.clip(boxes[:, 3], 0, cal.height - 1.0))
    rppc = rad_per_px(cal, uc, vc)
    sp = math.radians(st.pitch_sigma_deg)
    sy = math.radians(st.yaw_sigma_deg)
    yaw_T = _yaw_of(T)
    ranges = RANGE_GRID[RANGE_GRID <= st.max_range_m]
    for i in range(len(boxes)):
        name = classes[i]
        if name not in CLASS_PRIORS or not okc[i]:
            continue
        L, Wd, H, sH = CLASS_PRIORS[name]
        rad = (L + Wd) / 4.0
        az = math.atan2(dcv[i, 1], dcv[i, 0])
        cut_bottom = boxes[i, 3] >= cal.height - st.edge_px
        cut_top = boxes[i, 1] <= st.edge_px
        vb, vt = _rims(cal, az, rad, H, ranges)
        est = []                                   # (range of the centre from the camera, sigma, kind)
        if st.use_ground and not cut_bottom and cal.t[2] > 0.05:
            sol = _solve(vb, boxes[i, 3], ranges)
            if sol is not None and sol[1] < 0:
                spx = math.hypot(st.box_px_sigma, sp / rpp[i])
                est.append((sol[0], spx / abs(sol[1]), 'ground'))
        if st.use_size and not cut_top and not cut_bottom:
            hp = vb - vt
            sol = _solve(hp, boxes[i, 3] - boxes[i, 1], ranges)
            if sol is not None and sol[1] < 0:
                s_img = math.sqrt(2.0) * st.box_px_sigma / abs(sol[1])
                est.append((sol[0], math.hypot(s_img, sol[0] * sH / H), 'size'))
        if not est:
            continue
        wsum = sum(1.0 / (s * s) for _r, s, _k in est)
        rc = sum(rr / (s * s) for rr, s, _k in est) / wsum
        sr = math.hypot(math.sqrt(1.0 / wsum), abs(L - Wd) / 4.0)   # the heading is unknown: so is the reach
        # across the ray: the box centre's pixel and the yaw calibration, and the unknown heading (perspective puts
        # the box's middle off the object's middle by up to a fraction of its extent)
        st_m = math.hypot(rc * math.hypot(sy, st.box_px_sigma * rppc[i]), abs(L - Wd) / 4.0)
        cx = float(cal.t[0]) + rc * math.cos(az)
        cy = float(cal.t[1]) + rc * math.sin(az)
        P = apply_T(T, [cx, cy, H / 2.0])[0]
        c_ref = apply_T(T, cal.t)[0]
        az_ref = az + yaw_T
        cov = _cov_polar(az_ref, sr, st_m)
        info = {'method': '+'.join(k for _r, _s, k in est), 'range_m': round(rc, 3), 'sigma_range_m': round(sr, 3),
                'sigma_bearing_m': round(st_m, 4)}
        for rr, s, k in est:
            info[f'range_{k}_m'] = round(rr, 3)
            info[f'sigma_{k}_m'] = round(s, 3)
        d = Detection3D(name, float(scores[i]), np.array([P[0], P[1], H / 2.0]), np.array([L, Wd, H]), None,
                        (cal.tag,), cov, info)
        d._ray = (c_ref[:2].copy(), az_ref, st_m / max(rc, 1e-3))   # camera xy, azimuth, bearing σ (rad)
        out.append(d)
    return out


def triangulate(rays) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Weighted least-squares crossing of bird's-eye bearing lines [(camera xy, azimuth, σ bearing rad, range m)]:
    (point, 2 × 2 covariance, the largest crossing angle in degrees), or None when they are (near) parallel or the
    point is behind a camera."""
    A = np.zeros((2, 2))
    b = np.zeros(2)
    dirs = []
    for c, az, sb, rng in rays:
        u = np.array([math.cos(az), math.sin(az)])
        n = np.array([-u[1], u[0]])
        w = 1.0 / max(sb * max(rng, 1.0), 1e-4) ** 2      # the line's perpendicular error at the object's range
        A += w * np.outer(n, n)
        b += w * np.outer(n, n) @ np.asarray(c, float)
        dirs.append(u)
    ang = 0.0
    for i in range(len(dirs)):
        for j in range(i + 1, len(dirs)):
            ang = max(ang, math.degrees(math.acos(min(1.0, abs(float(dirs[i] @ dirs[j]))))))
    if np.linalg.cond(A) > 1e8:
        return None
    cov = np.linalg.inv(A)
    p = cov @ b
    for (c, az, _sb, _r), u in zip(rays, dirs):
        if float((p - np.asarray(c, float)) @ u) <= 0:
            return None
    return p, cov, ang


def _fuse_group(members, st: LiftSettings) -> Detection3D:
    if len(members) == 1:
        return members[0]
    if st.fusion == 'mean':
        infos = [np.linalg.inv(m.cov + np.eye(2) * 1e-6) for m in members]
        cov = np.linalg.inv(sum(infos))
        mean = cov @ sum(I @ m.centre[:2] for I, m in zip(infos, members))
        how = 'mean'
    else:                                             # the member with the tightest estimate (the fullest view)
        sharp = min(members, key=lambda m: float(np.linalg.det(m.cov)))
        mean, cov, how = sharp.centre[:2].copy(), sharp.cov.copy(), 'best view'
    rays = [(m._ray[0], m._ray[1], m._ray[2], float(np.hypot(*(m.centre[:2] - m._ray[0])))) for m in members
            if hasattr(m, '_ray')]
    cams = {tuple(np.round(r[0], 3)) for r in rays}
    far = min(r[3] for r in rays) >= st.min_triangulation_sizes * float(max(members[0].size[:2])) if rays else False
    if len(rays) >= 2 and len(cams) >= 2 and far:
        tri = triangulate(rays)
        if tri is not None and tri[2] >= st.min_triangulation_deg:
            p, c2, _ang = tri
            # a crossing far from every monocular estimate's reach is a wrong association's: keep the mean then
            if all(float((p - m.centre[:2]) @ np.linalg.solve(m.cov + c2, p - m.centre[:2])) <= 4 * st.gate_chi2
                   for m in members):
                mean, cov, how = p, c2, 'triangulated'
    best = max(members, key=lambda m: m.score)
    src = tuple(dict.fromkeys(s for m in members for s in m.sources))
    d = Detection3D(best.cls, max(m.score for m in members), np.array([mean[0], mean[1], best.centre[2]]),
                    best.size.copy(), None, src, cov, {'method': best.info.get('method', ''), 'fusion': how,
                                                       'fused': len(members)})
    if hasattr(best, '_ray'):
        d._ray = best._ray
    return d


def fuse(dets: list, st: LiftSettings | None = None) -> list:
    """Cross-source fusion of monocular detections (module doc): per class, source by source, a gated Hungarian
    assignment of the new source's detections to the groups so far on the squared Mahalanobis distance."""
    from scipy.optimize import linear_sum_assignment
    st = st or LiftSettings()
    out = []
    for cls in dict.fromkeys(d.cls for d in dets):
        groups = []                                   # [members]
        fused = []                                    # their fused Detection3D
        by_src = {}
        for d in dets:
            if d.cls == cls:
                by_src.setdefault(d.sources[0] if d.sources else '', []).append(d)
        for src, ds in by_src.items():
            if not groups:
                for d in ds:
                    groups.append([d])
                    fused.append(d)
                continue
            C = np.full((len(fused), len(ds)), 1e9)
            for i, g in enumerate(fused):
                for j, d in enumerate(ds):
                    if any(src in m.sources for m in groups[i]):
                        continue
                    S = g.cov + d.cov
                    e = d.centre[:2] - g.centre[:2]
                    m2 = float(e @ np.linalg.solve(S, e))
                    if m2 <= st.gate_chi2:
                        C[i, j] = m2
            rows, cols = linear_sum_assignment(C)
            taken = set()
            for i, j in zip(rows, cols):
                if C[i, j] < 1e9:
                    groups[i].append(ds[j])
                    fused[i] = _fuse_group(groups[i], st)
                    taken.add(j)
            for j, d in enumerate(ds):
                if j not in taken:
                    groups.append([d])
                    fused.append(d)
        out.extend(fused)
    for d in out:
        d.info.setdefault('fusion', 'single')
        d.info.setdefault('fused', 1)
    return out


# =========================================================================== the lifted backend
class LiftedBackend(Backend):
    """YOLOX per image, lifted (lift_boxes) and fused across sources (fuse). detector: a callable image -> 2D
    detections (core.detection's dict); None: core.detection.load(detector_settings) on first use."""
    id = 'lifted'
    name = 'Lifted YOLOX'
    version = VERSION
    frames = 'all'

    def __init__(self, detector=None, detector_settings: DT.DetectorSettings | None = None,
                 lift: LiftSettings | None = None, score_floor: float = 0.05):
        self.detector = detector
        self.det_settings = detector_settings or DT.DetectorSettings()
        self.lift = lift or LiftSettings()
        self.score_floor = float(score_floor)
        self._info = None
        self.images = 0
        self.detector_s = 0.0

    def _det(self):
        if self.detector is None:
            d = DT.load(self.det_settings)
            self.detector, self._info = d.detect, d.describe()
        return self.detector

    def detector_info(self) -> dict:
        if self._info is None:
            d = self.detector
            self._info = dict(getattr(d, 'describe', lambda: {})() or {}) if d is not None else {}
            if not self._info:
                sp = DT.MODELS.get(self.det_settings.model)
                self._info = {'model': self.det_settings.model, 'name': sp.name if sp else 'custom',
                              'sha256': sp.sha256 if sp else ''} if d is None else \
                    {'model': 'custom', 'name': 'custom detector'}
        return self._info

    def settings(self) -> dict:
        return {'detector': self.det_settings.to_dict(), 'lift': self.lift.to_dict(), 'score_floor': self.score_floor,
                'version': self.version}

    def detections_2d(self, item: FrameImage) -> dict:
        if item.dets is None:
            t0 = time.perf_counter()
            d = self._det()(item.load())
            self.detector_s += time.perf_counter() - t0
            self.images += 1
            item.dets = {k: d[k] for k in ('boxes', 'scores', 'coco', 'cls', 'ms') if k in d}
        return item.dets

    def detect(self, frame: Frame) -> list:
        mono = []
        for item in frame.images:
            d = self.detections_2d(item)
            sc = np.asarray(d['scores'], float)
            keep = sc >= self.score_floor
            if not keep.any():
                continue
            # core.detection's boxes taken as pixel edges: against the ground truth's tight boxes (pixel edges) on
            # data-rig-6-cam-lidar-radars, YOLOX-S's bottom edges sit a median 0.1 px above them (305 matches)
            boxes = np.asarray(d['boxes'], float).reshape(-1, 4)[keep]
            names = [DT.CLASSES[int(c)] if 0 <= int(c) < len(DT.CLASSES) else '' for c in np.asarray(d['cls'])[keep]]
            mono += lift_boxes(boxes, names, sc[keep], item.cal, item.T_ref, self.lift)
        return fuse(mono, self.lift) if self.lift.fuse else mono


# =========================================================================== nuScenes submission (external backend)
def quat_matrix(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def read_submission(doc, ego_poses: dict) -> dict:
    """A nuScenes detection submission ({'results': {sample_token: [box]}}, or the results dict itself) -> {sample_token:
    [Detection3D]} in the vehicle frame (ego_poses: sample_token -> 4 × 4 vehicle -> global at the sample). Boxes of
    unknown classes or samples are left out; nuScenes' size is [width, length, height]."""
    res = doc.get('results', doc) if isinstance(doc, dict) else {}
    out = {}
    if not isinstance(res, dict):
        return out
    for tok, boxes in res.items():
        T = ego_poses.get(tok)
        if T is None or not isinstance(boxes, list):
            continue
        Ti = np.linalg.inv(T)
        lst = out.setdefault(tok, [])
        for b in boxes:
            if not isinstance(b, dict):
                continue
            cls = NUSCENES_NAME_TO_CLASS.get(str(b.get('detection_name', '')))
            try:
                tr = np.asarray(b['translation'], float)[:3]
                sz = np.asarray(b.get('size') or [0, 0, 0], float)[:3]
                sc = float(b.get('detection_score', 0.0))
                q = b.get('rotation') or [1, 0, 0, 0]
                Rg = quat_matrix(q)
            except (KeyError, TypeError, ValueError):
                continue
            if cls is None or tr.shape != (3,) or not np.all(np.isfinite(tr)) or not math.isfinite(sc):
                continue
            p = apply_T(Ti, tr)[0]
            Rv = Ti[:3, :3] @ Rg
            lst.append(Detection3D(cls, sc, p, np.array([sz[1], sz[0], sz[2]]) if sz.shape == (3,) else
                                   np.zeros(3), math.atan2(Rv[1, 0], Rv[0, 0]), ('external',), None,
                                   {'method': 'external'}))
    return out


class ExternalStore:
    """The user's external detector commands, by name, in the user's own settings (a QSettings or any object with
    value / setValue / remove), never in a project: {'argv': [...], 'timeout_min', 'approved'}. A changed argument list
    is not approved until approve(name, argv) with that exact list."""
    GROUP = 'rig_perception/external'

    def __init__(self, settings):
        self.settings = settings

    @staticmethod
    def digest(name, argv) -> str:
        return hashlib.sha256(json.dumps(['vantage-bev-external', str(name), list(argv)]).encode()).hexdigest()

    def _key(self, name):
        return f'{self.GROUP}/{hashlib.sha1(str(name).encode()).hexdigest()[:16]}'

    def names(self) -> list:
        raw = self.settings.value(f'{self.GROUP}/_names', '')
        try:
            v = json.loads(raw) if isinstance(raw, str) and raw else []
        except ValueError:
            v = []
        return [str(x) for x in v if isinstance(x, str)] if isinstance(v, list) else []

    def get(self, name) -> dict | None:
        raw = self.settings.value(self._key(name), '')
        try:
            d = json.loads(raw) if isinstance(raw, str) and raw else None
        except ValueError:
            d = None
        if not isinstance(d, dict) or d.get('name') != name or not isinstance(d.get('argv'), list):
            return None
        d['argv'] = [str(a) for a in d['argv']]
        return d

    def put(self, name, argv, timeout_min=30.0) -> dict:
        name = clean_name(name)
        if not name:
            raise ValueError('Give the detector a name (letters, digits, - and _).')
        argv = [str(a) for a in argv if str(a) != '']
        if not argv:
            raise ValueError('The command is empty.')
        old = self.get(name)
        d = {'name': name, 'argv': argv, 'timeout_min': float(min(max(float(timeout_min or 30.0), 0.1), 1440.0)),
             'approved': old['approved'] if old and old['argv'] == argv else ''}
        self.settings.setValue(self._key(name), json.dumps(d))
        names = self.names()
        if name not in names:
            self.settings.setValue(f'{self.GROUP}/_names', json.dumps(names + [name]))
        if hasattr(self.settings, 'sync'):
            self.settings.sync()
        return d

    def approve(self, name, argv) -> bool:
        d = self.get(name)
        if d is None or list(argv) != d['argv']:
            return False
        d['approved'] = self.digest(name, d['argv'])
        self.settings.setValue(self._key(name), json.dumps(d))
        if hasattr(self.settings, 'sync'):
            self.settings.sync()
        return True

    def approved(self, name) -> dict | None:
        d = self.get(name)
        return d if d is not None and d['approved'] == self.digest(name, d['argv']) else None

    def remove(self, name):
        self.settings.remove(self._key(name))
        self.settings.setValue(f'{self.GROUP}/_names', json.dumps([n for n in self.names() if n != name]))


def clean_name(name) -> str:
    s = ''.join(ch for ch in str(name or '') if ch.isalnum() or ch in '-_')[:40]
    return s


PLACEHOLDERS = ('{dataroot}', '{version}', '{frames}', '{out}')


class ExternalBackend(Backend):
    """An approved command per recording (module doc). argv: the approved list; its elements may hold the
    placeholders {dataroot} (the recording's nuScenes folder), {version} (its table folder's name), {frames} (the
    frames file Vantage writes) and {out} (where the command writes its submission JSON)."""
    id = 'external'
    frames = 'keyframes'

    def __init__(self, name: str, argv: list, work_dir: str, timeout_s: float = 1800.0, cancel=None):
        self.name = f'External: {name}'
        self.ext_name = name
        self.version = hashlib.sha256(json.dumps(list(argv)).encode()).hexdigest()[:12]
        self.argv = list(argv)
        self.work_dir = work_dir
        self.timeout_s = float(timeout_s)
        self.cancel = cancel or (lambda: False)
        self.results = {}
        self.log = []

    def settings(self) -> dict:
        return {'name': self.ext_name, 'argv_sha256': self.version}

    def prepare(self, recording) -> None:
        """recording: core.rig_perception's PerceptionRecording (its nuScenes folder, frames and ego poses)."""
        os.makedirs(self.work_dir, exist_ok=True)
        slug = hashlib.sha1(recording.path.encode()).hexdigest()[:10]
        frames_path = os.path.join(self.work_dir, f'frames-{slug}.json')
        out_path = os.path.join(self.work_dir, f'results-{slug}.json')
        if os.path.exists(out_path):
            os.remove(out_path)
        with open(frames_path, 'w', encoding='utf-8') as fh:
            json.dump(recording.frames_doc(), fh, indent=1)
        sub = {'{dataroot}': recording.nus_root, '{version}': recording.nus_version, '{frames}': frames_path,
               '{out}': out_path}
        cmd = []
        for a in self.argv:
            for k, v in sub.items():
                a = a.replace(k, v)
            cmd.append(a)
        t0 = time.monotonic()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                cwd=self.work_dir, shell=False)
        try:
            while proc.poll() is None:
                if self.cancel():
                    proc.terminate()
                    proc.wait(10)
                    raise RuntimeError('cancelled')
                if time.monotonic() - t0 > self.timeout_s:
                    proc.kill()
                    raise RuntimeError(f'{self.ext_name} ran over {self.timeout_s:g} s and was stopped.')
                time.sleep(0.05)
        finally:
            text = proc.stdout.read() if proc.stdout else ''
            self.log.append(text[-20000:])
        if proc.returncode != 0:
            raise RuntimeError(f'{self.ext_name} exited with {proc.returncode}: {text.strip()[-400:]}')
        try:
            with open(out_path, encoding='utf-8') as fh:
                doc = json.load(fh)
        except (OSError, ValueError) as e:
            raise RuntimeError(f'{self.ext_name} wrote no readable results ({os.path.basename(out_path)}: {e}).')
        self.results.update(read_submission(doc, recording.sample_poses()))

    def detect(self, frame: Frame) -> list:
        return list(self.results.get(frame.token, []))


def external_from_settings(qsettings, name, work_dir, cancel=None) -> ExternalBackend:
    """The approved command of that name (RuntimeError when it is missing or not approved)."""
    st = ExternalStore(qsettings)
    d = st.get(name)
    if d is None:
        raise RuntimeError(f'No external detector named {name!r} on this computer: set its command on the '
                           'Rig perception page.')
    if st.approved(name) is None:
        raise RuntimeError(f'The command of {name!r} is not approved: approve it on the Rig perception page.')
    return ExternalBackend(name, d['argv'], work_dir, d['timeout_min'] * 60.0, cancel)


def python_argv(script, *args) -> list:
    """An argument list running a Python script with this Python (for tests and simple adapters)."""
    return [sys.executable, str(script), *args]
