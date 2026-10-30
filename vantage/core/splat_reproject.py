"""Log replay through a proposed rig: a recorded drive's street scene rebuilt as 3D Gaussians, rendered through another
rig variant's cameras and lidars with Vantage's own sensor models, and detection compared between the two rigs on the
re-rendered drive (Simulate ▸ Log replay).

The drive
---------
A nuScenes folder as Vantage's CARLA bridge and sample dataset write it (docs/simulator-bridge.md: the tables, the
samples/ and sweeps/ files, optionally ground_truth/index.json and rig/sim_rig.json beside it) or a ROS 2 bag
(core.rosbag: compressed or raw images, camera_info, PointCloud2 sweeps, /tf_static, /odom or /tf, and
/ground_truth/boxes when the bag has them; the messages are copied once into the cache folder). Time t is seconds from
the drive's first ego pose. The ego pose is interpolated between the recorded poses (positions linearly, rotations by
normalised linear interpolation of the quaternions: the poses are 50–60 Hz apart, where nlerp and slerp differ by
less than 10⁻⁶ rad). Every position is kept relative to the drive's first ego position (the scene origin) so float32
keeps millimetres hundreds of metres away.

Reconstruction (source 'lidar')
-------------------------------
1. Gaussians from the lidar sweeps: every point goes into the world through the ego pose at the instant it was fired
   (a spinning lidar's firing time from its azimuth and turning sense, or the point's own time field in a bag; a
   recorded point lies where the sensor was when it fired, as the bridge and the sample dataset write them), then the
   points are thinned to voxel centroids whose size grows with the distance from the driven path (voxel_m within
   10 m, doubling every doubling of distance, at most 8×), the base size grown until the count fits max_gaussians.
2. Shape: each Gaussian's axes from the principal components of its 8 nearest neighbours: a flat neighbourhood
   (smallest eigenvalue under a tenth of the middle one) gives a disc along the surface, σ along the surface half the
   mean distance to the 3 nearest neighbours (at most three quarters of its voxel), across it a tenth of that;
   otherwise a sphere of that σ (vegetation, poles, edges). Opacity 0.9.
3. Colour: every Gaussian is projected into the training camera frames through the camera's calibration and lens
   model (the rig file's exact lens block when the drive has one, else the Kannala-Brandt calibration of its
   calibrated_sensor rows or camera_info), kept where it is the nearest surface (a z-buffer of the projected centres on
   4 × 4 pixel cells, eroded over 3 × 3 cells, within 2 % + 0.2 m), and coloured by the weighted mean of the image
   there (weight 1 / (1 + range / 20 m)); one never seen takes its nearest seen neighbour's colour. Pixels of the
   recording vehicle are left out: the drive's own ego labels (ground truth class 29) when it has them, else the
   project's vehicle seen from the camera through the occlusion engine (core.analysis.Engine). Sky: what the images
   show where no Gaussian projects, in a 16 × 64 equirectangular texture in the world frame (azimuth, elevation −30°
   to 90°).
4. Optimisation (optimise on): colour, opacity, scale, rotation, the sky and, with optimise_positions, the centres,
   by Adam on 0.8 L1 + 0.2 (1 − SSIM) against the recorded images at train_scale of their size, through the
   differentiable rasteriser of core.splat_raster (the lens model itself, not a pinhole), every held-out frame left
   out (holdout_every: frame k of each camera with k mod n = n // 2, and lidar sweep k likewise). Time-budgeted and
   cancellable; Gaussians left under 1 % opacity are removed after it. The lidar decides where the Gaussians are: a
   step may not move one more than 0.25 m from its point or grow it beyond 3 times its first size (the images alone
   would let them stretch into the sky and into fog in front of surfaces), and no densification or splitting (3DGS's
   adaptive density control) is done.
Import (source 'ply'): a standard 3D Gaussian splatting .ply (the INRIA layout gsplat, nerfstudio and the INRIA trainer
write: x, y, z, f_dc_0..2, opacity as a logit, scale_0..2 as logarithms, rot_0..3 as w, x, y, z) placed into the
drive's global frame by the alignment given (p_global = s·R(yaw, pitch, roll)·p + t); view-dependent colour (f_rest)
is dropped.

Moving objects
--------------
The scene is static: anything that moved during the drive smears along its path. Where the drive has 3D boxes (a
CARLA drive's ground truth: core.groundtruth's objects per camera frame, its lidar panoptic labels; a sample dataset's
or nuScenes' sample_annotation boxes; a bag's /ground_truth/boxes), every annotated actor is cut out of the static
scene and rebuilt in its own box frame from the points inside its box at each sweep (a rigid body: a walking
pedestrian's limbs smear inside the box), then placed at its box pose (interpolated between the annotations) whenever
a view is rendered. An actor is shown only within 0.11 s of an annotation and across gaps of at most 0.6 s. Without
boxes (dynamic 'ignore', or a drive without them) everything stays in the static scene and a finding says so.

Rendering through a rig
-----------------------
* Cameras: every pixel through the variant's camera model (core.lens.resolve of its module and lens, or its unit
  calibration: Kannala-Brandt fisheyes included) at render_scale of the sensor's size; the recorded rig at its recorded
  image size scaled alike. Gaussians composited front to back (core.splat_raster). The variant's own vehicle, rig parts
  and housings: the pixels whose ray meets them (the occlusion engine's cube map from the camera's position) show the
  engine's shading of the body in grey, not the scene: the recording vehicle is never in the scene (its lidar points
  are dropped, its pixels masked out of the colours and the loss). Rolling shutter (rolling_shutter 'module', modules
  not stated global): the image is rendered in up to 8 row bands, each with the ego pose at its rows' mid-exposure time
  (row y at t + (y / h − ½)·readout, the frame stamped at its middle row), readout core.timing's readout_ms; 'off'
  renders a global shutter. The recorded rig is rendered by the same rule: with the recording's own readout when the
  drive states one (a real log, a sample dataset written with a rolling shutter), else (frames stamped as single
  instants, as CARLA renders them, or a log that does not say) with the readout of the drive variant's camera of the
  same tag, so the same cameras in both rigs give the same images (the reconstruction and its validation always use
  the recording's own timing).
* Lidars: each beam of the variant's lidar model (core.sensors: its ring elevations and azimuth columns, turning
  sense, rate, phase) is cast against the Gaussians: the beam grid is rendered in 16 azimuth sectors, each with the ego
  pose at its firing time, as an image whose pixel centres are the beams, through the same rasteriser (each Gaussian's
  footprint in azimuth × elevation from the spherical projection's Jacobian, dilated by 0.05 px² instead of an
  image's 0.3: a beam is one ray). Range (lidar_depth): 'median', the rᵢ of the Gaussian at which the beam's
  transmittance falls through ½ (default: semi-transparent Gaussians left by the optimisation do not pull it towards
  what lies behind them), or 'expected', Σ αᵢTᵢ rᵢ / Σ αᵢTᵢ; each rᵢ the range along the beam to the plane of
  Gaussian i (its thinnest axis the normal, to first order; core.splat_raster); no return where the
  accumulated opacity Σ αᵢTᵢ stays under lidar_min_opacity, where the surface's incidence (the opacity-weighted
  |cos| between the beam and the Gaussians' thinnest axes) is beyond lidar_max_incidence_deg, outside the model's
  range, and where the beam meets the variant's own vehicle (self hits are dropped, as the recordings do). Points are
  written in the sensor frame at its firing pose.

Validation (rig A = the recorded rig)
-------------------------------------
Held-out frames rendered through the recorded calibration at train_scale against the recorded images: PSNR and SSIM
over the pixels that are inside the lens's field and not on the recording vehicle, before and after the optimisation;
held-out lidar sweeps rendered through the recorded lidar (ring elevations measured from the recorded points, columns
on the model's azimuth grid) against the recorded points beam by beam: range error where both return (and where the
beam meets the surface within 70° of its normal: at grazing incidence a centimetre of height is metres of range),
returns only recorded, returns only rendered (beams that returned in none of the compared sweeps, the recording
vehicle's own body and open sky, left out).

The A/B comparison
------------------
At eval_instants instants spread over the drive (the first camera's frame times) both rigs' cameras and lidars are
rendered. The ground truth of each image: every annotated object's box (vehicle frame at the instant) through that
camera: its visible pixels in the re-rendered image (the object's own Gaussians are the front-most), their tight box,
the share of the box's rays that something nearer hides (occluded), the share of its surface outside the image
(truncated), its range. core.detection_eval then does the rest as for recorded drives: ignore regions (min_px,
min_pixels, max_occluded, max_truncated), COCO's greedy matching (evaluate_image, match) of the detector's boxes
(core.detection: YOLOX through OpenCV, skipped with a note when the model is not downloaded), recall at the operating
point per class and range bin (the project's detection settings: classes, score, IoU, range_edges). The rig counts an
object (an actor at an instant) once, detected when any of its cameras detects it, ranged from the rear axle; the
comparison takes the objects that either rig shows unignored in at least one camera, so both rigs are scored on the
same objects. Without the detector the figures are the objects each rig shows large and clear enough to be scored.
Lidar: returns per sweep, the share of beams on the own vehicle, and the points on each annotated object per class and
range.
Coverage (rig A the recorded rig, a drive with per-frame ground truth such as a CARLA run): the objects the recording's
own ground truth shows unignored at the same instants (each camera's frame nearest the instant, within ACTOR_PAD_S),
per class and range, against those the re-rendered recorded rig shows; what the reconstruction leaves out (far, small
and thin objects the lidar caught too few points of) is not scored by either rig. Traffic lights, thin and high, are
mostly not rebuilt: their recall is not assessed (their rows stay in the table, but no finding is drawn from them).

Limits
------
The re-rendered images are not real images: Gaussians from a lidar's points and a few hundred photometric steps blur
fine texture, leave holes where the lidar never reached (above its field: tall buildings show sky), and show nothing
the recording never saw from where the proposed rig looks. A detector's absolute recall on them is therefore not its
recall on the recorded images: on the same objects it is a little lower (fine detail such as a traffic light's lamps is
lost), while the objects the lidar did not rebuild finely enough (far, small and thin ones) are not shown, so not
scored, which leaves an easier set; what carries over is the difference between rigs on the same reconstruction.
Colour has no view dependence and no exposure model (the recording's own exposure is baked in). The recorded rig and
the proposed rig see the same moment of the same drive (the ego's path is the recording's).
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from . import detection as DT
from . import detection_eval as DE
from . import splat_raster as SR
from .lens import resolve, theta_of_r, r_of_theta
from .mathutil import BODY_TO_OPTICAL, rot_zyx

SCHEMA = 'vantage.splat/1'
KEY = 'splat'
FRAME = 'vehicle (ISO 8855, rear axle on the ground); world: the drive\'s global frame (nuScenes global, bag odom)'
SOURCES = ('lidar', 'ply')
DYNAMIC = ('boxes', 'ignore')
ROLLING = ('module', 'off')
DEVICES = ('auto', 'cpu')
RECORDED = 'recorded'                      # variant_a: the drive's own rig (its calibration)
SKY_H, SKY_W = 16, 64
SKY_EL_LO = -30.0
NEAR_M = 10.0                              # the base voxel holds within this distance of the driven path
EGO_CLASS = 29
SECTORS = 16
MAX_BANDS = 8
ACTOR_PAD_S, ACTOR_GAP_S = 0.11, 0.6
STEEP_DEG = 70.0                             # validation: beams meeting their surface within this of its normal
LIDAR_LOWPASS = 0.05                         # px² of dilation on the beam grid (a beam is one ray, not a pixel's area)
SCENE_VERSION = 2                            # in the cache key: a scene built by older code is built again
MAX_SHIFT_M, MAX_GROWTH = 0.25, 3.0          # the optimisation's leash on each Gaussian (position, size)
LIMITS = {'max_gaussians': (2000, 300000), 'voxel_m': (0.02, 1.0), 'max_range_m': (10.0, 200.0),
          'sweep_step': (1, 50), 'holdout_every': (0, 50), 'iterations': (0, 20000), 'time_budget_s': (1.0, 7200.0),
          'train_scale': (0.05, 1.0), 'colour_step': (1, 50), 'eval_instants': (1, 200), 'render_scale': (0.05, 1.0),
          'lidar_min_opacity': (0.05, 0.99), 'lidar_max_incidence_deg': (30.0, 89.9), 'gpu_mem_gb': (0.5, 3.0),
          'seed': (0, 2 ** 31 - 1), 'min_psnr_db': (5.0, 50.0), 'ply_scale': (1e-4, 1e4), 'val_frames': (1, 500),
          'val_sweeps': (1, 100)}
INTS = ('max_gaussians', 'sweep_step', 'holdout_every', 'iterations', 'colour_step', 'eval_instants', 'seed',
        'val_frames', 'val_sweeps')
BOOLS = ('optimise', 'optimise_positions', 'detect', 'export_ply')
ASSUMPTIONS = (
    'The street scene is rebuilt from the drive\'s lidar sweeps as 3D Gaussians coloured (and optimised) from its '
    'camera images, and re-rendered through Vantage\'s lens and lidar models: the re-rendered images are not real '
    'images (blurred texture, holes where the lidar never reached, nothing the recording never saw). Absolute '
    'detection figures on them are not those of recorded images (on the same objects a little lower; objects the '
    'lidar did not rebuild finely enough, far, small and thin ones, are not scored at all); the difference between '
    'two rigs on the same '
    'reconstruction is what the study measures. Moving objects are rebuilt as rigid bodies in their annotated boxes '
    'where the drive has boxes; otherwise the scene is static and anything that moved smears. Rolling shutter '
    '(rolling_shutter module): a variant\'s camera gets its module\'s readout (core.timing; none for a global-shutter '
    'module); the recorded rig\'s camera gets the recording\'s own readout when the drive states one (a real log), '
    'otherwise (a simulator\'s single-instant frames, or a log that does not say) the readout of the drive variant\'s '
    'camera of the same tag, so the same cameras in both rigs render the same images. Traffic lights are mostly not '
    'rebuilt and their recall is not assessed.')


def _f(v, default):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


# =========================================================================== settings
@dataclass
class SplatSettings:
    drive: str = ''                    # the recording: a nuScenes folder (or the run folder holding nuscenes/) or a bag
    source: str = 'lidar'              # lidar: rebuilt from the drive; ply: a 3DGS .ply imported
    ply_path: str = ''
    ply_xyz: list = field(default_factory=lambda: [0.0, 0.0, 0.0])   # m: the .ply's origin in the drive's global frame
    ply_ypr: list = field(default_factory=lambda: [0.0, 0.0, 0.0])   # deg: its rotation (yaw, pitch down, roll)
    ply_scale: float = 1.0
    max_gaussians: int = 300000
    voxel_m: float = 0.05
    max_range_m: float = 80.0
    sweep_step: int = 1                # every n-th lidar sweep
    holdout_every: int = 8             # frame k of each camera (and sweep k) with k mod n = n // 2 held out; 0: none
    dynamic: str = 'boxes'             # boxes: annotated actors rebuilt in their boxes; ignore: all static
    colour_step: int = 2               # every n-th training frame colours the Gaussians
    optimise: bool = True
    optimise_positions: bool = True
    iterations: int = 600
    time_budget_s: float = 120.0
    train_scale: float = 0.25          # images at this share of their recorded size for colours and the optimisation
    variant_a: str = RECORDED          # 'recorded' or a variant uid
    variant_b: str = ''                # a variant uid ('' : the active variant)
    eval_instants: int = 12
    render_scale: float = 0.5          # rendered images at this share of the module's sensor size
    rolling_shutter: str = 'module'    # module: the module's readout (core.timing) unless it is global; off
    lidar_min_opacity: float = 0.5
    lidar_max_incidence_deg: float = 85.0
    lidar_depth: str = 'median'        # median: where the beam's transmittance falls through ½; expected: Σ αTr / Σ αT
    detect: bool = True                # run the detector when it is downloaded
    device: str = 'auto'               # auto: CUDA when present (≤ gpu_mem_gb), else the CPU; cpu
    gpu_mem_gb: float = 3.0
    seed: int = 0
    min_psnr_db: float = 18.0          # held-out PSNR below this: the reconstruction does not validate
    val_frames: int = 48               # held-out frames scored (spread over the drive)
    val_sweeps: int = 8                # held-out sweeps per lidar scored
    export_ply: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d) -> 'SplatSettings':
        """Tolerant: unknown keys dropped, numbers clamped to LIMITS, words checked, junk back to the default."""
        s = SplatSettings()
        d = d if isinstance(d, dict) else {}
        for k in ('drive', 'ply_path', 'variant_a', 'variant_b'):
            if isinstance(d.get(k), str):
                setattr(s, k, d[k].strip()[:4096])
        if not s.variant_a:
            s.variant_a = RECORDED
        for k, opts in (('source', SOURCES), ('dynamic', DYNAMIC), ('rolling_shutter', ROLLING),
                        ('lidar_depth', ('median', 'expected')),
                        ('device', DEVICES + ('cuda',))):
            if d.get(k) in opts:
                setattr(s, k, d[k])
        for k in BOOLS:
            v = d.get(k)
            if isinstance(v, bool) or (isinstance(v, int) and not isinstance(v, float)):
                setattr(s, k, bool(v))
        for k, (lo, hi) in LIMITS.items():
            if k not in d or isinstance(d[k], bool):
                continue
            v = _f(d[k], None)
            if v is None:
                continue
            v = min(max(v, lo), hi)
            setattr(s, k, int(round(v)) if k in INTS else float(v))
        for k, lim in (('ply_xyz', 1e7), ('ply_ypr', 360.0)):
            v = d.get(k)
            if isinstance(v, (list, tuple)) and len(v) == 3:
                vals = [_f(x, None) for x in v]
                if all(x is not None for x in vals):
                    setattr(s, k, [min(max(x, -lim), lim) for x in vals])
        return s

    def build_key(self) -> dict:
        """The settings the reconstruction depends on (the cache key)."""
        keys = ('version', 'source', 'max_gaussians', 'voxel_m', 'max_range_m', 'sweep_step', 'holdout_every',
                'dynamic',
                'colour_step', 'optimise', 'optimise_positions', 'iterations', 'time_budget_s', 'train_scale', 'seed')
        d = {k: getattr(self, k, None) for k in keys}
        d['version'] = SCENE_VERSION
        if self.source == 'ply':
            d.update(ply_path=self.ply_path, ply_xyz=self.ply_xyz, ply_ypr=self.ply_ypr, ply_scale=self.ply_scale)
        return d


def settings_of(project) -> SplatSettings:
    return SplatSettings.from_dict(getattr(project.analysis, KEY, None) if project is not None else None)


def _prog(progress, f, text=''):
    if progress is not None:
        try:
            progress(min(max(float(f), 0.0), 1.0), text)
        except TypeError:
            progress(min(max(float(f), 0.0), 1.0))


class Cancelled(Exception):
    pass


# =========================================================================== camera models
@dataclass
class CamModel:
    """A camera's projection onto an image of width × height (pixel-edge coordinates: pixel (i, j) covers
    [i, i+1) × [j, j+1)): u = cx + fx·x·r(θ)/ρ, v = cy + fy·y·r(θ)/ρ in the optical frame (x right, y down, z forward),
    r the normalised image height of the projection (core.lens), θ the angle off the axis, max_theta the field."""
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    projection: str = 'kannala_brandt'
    kb: tuple = (0.0, 0.0, 0.0, 0.0)
    max_theta: float = math.pi / 2

    @staticmethod
    def from_intrinsics(intr, w: int, h: int) -> 'CamModel':
        """Vantage's resolved intrinsics (core.lens.Intrinsics, sensor pixels) on a w × h image of the whole sensor."""
        sx, sy = w / intr.width, h / intr.height
        return CamModel(int(w), int(h), intr.f_px * sx, intr.f_px * intr.aspect * sy, intr.cx * sx, intr.cy * sy,
                        intr.projection, tuple(float(k) for k in intr.kb), float(intr.max_theta))

    @staticmethod
    def from_lens_block(lens: dict) -> 'CamModel':
        """A rig file's camera 'lens' block (docs/simulator-bridge.md)."""
        w, h = (int(x) for x in lens['image_size'])
        mt = math.radians(float(lens.get('max_theta_deg', 90.0)))
        if lens.get('model') == 'pinhole':
            K = np.asarray(lens['K'], float)
            return CamModel(w, h, K[0, 0], K[1, 1], K[0, 2] + 0.5, K[1, 2] + 0.5, 'rectilinear', (0, 0, 0, 0),
                            min(mt, math.radians(89.0)))
        W, H = (float(x) for x in lens['sensor_size'])
        f = float(lens['f_px'])
        return CamModel(w, h, f * w / W, f * float(lens.get('aspect', 1.0)) * h / H, float(lens['cx']) * w / W,
                        float(lens['cy']) * h / H, str(lens['projection']),
                        tuple(float(k) for k in (list(lens.get('kb') or []) + [0.0] * 4)[:4]), mt)

    @staticmethod
    def from_opencv(K, D, w: int, h: int, max_theta: float | None = None) -> 'CamModel':
        """An OpenCV calibration (K with pixel centres on integers; D: cv2.fisheye's k1..k4, or None for a pinhole)."""
        K = np.asarray(K, float).reshape(3, 3)
        if D is None:
            return CamModel(int(w), int(h), K[0, 0], K[1, 1], K[0, 2] + 0.5, K[1, 2] + 0.5, 'rectilinear',
                            (0.0, 0.0, 0.0, 0.0), max_theta or math.radians(89.0))
        kb = tuple(float(k) for k in (list(np.asarray(D, float).ravel()) + [0.0] * 4)[:4])
        m = CamModel(int(w), int(h), K[0, 0], K[1, 1], K[0, 2] + 0.5, K[1, 2] + 0.5, 'kannala_brandt', kb, math.pi)
        if max_theta is None:
            # the image's farthest corner, plus 2°, kept where the model still grows (a KB fit folds back past it)
            r = max(math.hypot(x - m.cx, y - m.cy) for x in (0, w) for y in (0, h)) / min(m.fx, m.fy)
            th = np.linspace(0.0, math.radians(120.0), 1201)
            rr = r_of_theta(th, 1.0, 'kannala_brandt', kb)
            grow = np.flatnonzero(np.diff(rr) <= 0)
            th_fold = th[grow[0]] if len(grow) else th[-1]
            max_theta = min(float(theta_of_r(r, 1.0, 'kannala_brandt', kb)) + math.radians(2.0), th_fold)
        m.max_theta = float(max_theta)
        return m

    def scaled(self, w: int, h: int) -> 'CamModel':
        sx, sy = w / self.width, h / self.height
        return CamModel(int(w), int(h), self.fx * sx, self.fy * sy, self.cx * sx, self.cy * sy, self.projection,
                        self.kb, self.max_theta)

    def project(self, P):
        """Optical-frame points (K, 3) -> (u, v, theta, ok): ok where the model holds (within the field)."""
        P = np.asarray(P, float).reshape(-1, 3)
        x, y, z = P[:, 0], P[:, 1], P[:, 2]
        rho = np.hypot(x, y)
        th = np.arctan2(rho, z)
        r = r_of_theta(th, 1.0, self.projection, self.kb)
        s = np.where(rho > 1e-12, r / np.where(rho > 1e-12, rho, 1.0), 0.0)
        ok = th <= self.max_theta
        if self.projection == 'rectilinear':
            ok &= z > 1e-6
        return self.cx + self.fx * x * s, self.cy + self.fy * y * s, th, ok

    def rays(self):
        """(h, w, 3) unit optical-frame rays of the pixel centres and (h, w) where they are within the field."""
        uu, vv = np.meshgrid(np.arange(self.width) + 0.5, np.arange(self.height) + 0.5)
        x = (uu - self.cx) / self.fx
        y = (vv - self.cy) / self.fy
        r = np.hypot(x, y)
        th = theta_of_r(r, 1.0, self.projection, self.kb)
        s = np.where(r > 1e-12, np.sin(th) / np.where(r > 1e-12, r, 1.0), 0.0)
        d = np.stack([x * s, y * s, np.cos(th)], -1)
        d /= np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-12)
        return d, (th <= self.max_theta) & np.isfinite(th)

    def projector(self):
        return SR.LensProjector(self.fx, self.fy, self.cx, self.cy, self.projection, self.kb, self.max_theta)

    def to_dict(self) -> dict:
        return {'width': self.width, 'height': self.height, 'fx': round(self.fx, 4), 'fy': round(self.fy, 4),
                'cx': round(self.cx, 4), 'cy': round(self.cy, 4), 'projection': self.projection,
                'kb': [float(k) for k in self.kb], 'max_theta_deg': round(math.degrees(self.max_theta), 3)}


# =========================================================================== poses
def quat_to_rot(q) -> np.ndarray:
    return SR.quat_to_rot_np(q)


def rot_to_quat(R) -> np.ndarray:
    return SR.rot_to_quat_np(R)


def _nlerp(q0, q1, f):
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0, -q1, q1)
    q = (1 - f)[..., None] * q0 + f[..., None] * q1
    return q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)


class Trajectory:
    """The ego pose world ← vehicle over the drive: t (s), positions (local frame: global − origin), quaternions w, x,
    y, z. Interpolated (positions linearly, rotations by nlerp), held at the ends."""

    def __init__(self, t, pos, quat):
        o = np.argsort(np.asarray(t, float), kind='stable')
        t = np.asarray(t, float)[o]
        keep = np.r_[True, np.diff(t) > 1e-9]
        self.t = t[keep]
        self.pos = np.asarray(pos, float)[o][keep]
        q = np.asarray(quat, float)[o][keep]
        self.quat = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)

    def at(self, t):
        """(R (k, 3, 3), p (k, 3)) at the times t (k,)."""
        t = np.atleast_1d(np.asarray(t, float))
        if len(self.t) == 1:
            return np.repeat(quat_to_rot(self.quat[:1]), len(t), 0), np.repeat(self.pos[:1], len(t), 0)
        i = np.clip(np.searchsorted(self.t, t) - 1, 0, len(self.t) - 2)
        f = np.clip((t - self.t[i]) / (self.t[i + 1] - self.t[i]), 0.0, 1.0)
        p = self.pos[i] + f[:, None] * (self.pos[i + 1] - self.pos[i])
        return quat_to_rot(_nlerp(self.quat[i], self.quat[i + 1], f)), p

    def pose(self, t):
        R, p = self.at([t])
        return R[0], p[0]

    def length_m(self) -> float:
        return float(np.sum(np.linalg.norm(np.diff(self.pos, axis=0), axis=1))) if len(self.pos) > 1 else 0.0


@dataclass
class Track:
    """An annotated object over the drive: box poses world ← box (local frame) at the annotation times."""
    id: int
    cls: str                            # core.groundtruth class name or a nuScenes-derived one ('car', 'pedestrian', …)
    category: str
    dynamic: bool
    size: tuple                         # (length, width, height) m
    t: np.ndarray
    centre: np.ndarray                  # (k, 3) box centres
    quat: np.ndarray                    # (k, 4)

    def pose(self, t):
        """(R world ← box, centre) at t, or None when the object is not shown then (more than ACTOR_PAD_S from an
        annotation, or inside a gap longer than ACTOR_GAP_S). An object that does not move is where it was annotated
        at any time."""
        ts = self.t
        if not len(ts):
            return None
        if not self.dynamic:
            k = int(np.argmin(np.abs(ts - t)))
            return quat_to_rot(self.quat[k:k + 1])[0], self.centre[k].copy()
        j = int(np.searchsorted(ts, t))
        if j < len(ts) and abs(ts[j] - t) < 1e-9:
            return quat_to_rot(self.quat[j:j + 1])[0], self.centre[j].copy()
        if 0 < j < len(ts) and ts[j] - ts[j - 1] <= ACTOR_GAP_S:
            f = np.array([(t - ts[j - 1]) / (ts[j] - ts[j - 1])])
            q = _nlerp(self.quat[j - 1:j], self.quat[j:j + 1], f)
            return quat_to_rot(q)[0], self.centre[j - 1] + f[0] * (self.centre[j] - self.centre[j - 1])
        k = j - 1 if j > 0 else 0
        if j >= len(ts):
            k = len(ts) - 1
        elif j > 0 and abs(ts[j] - t) < abs(ts[j - 1] - t):
            k = j
        if abs(ts[k] - t) <= ACTOR_PAD_S:
            return quat_to_rot(self.quat[k:k + 1])[0], self.centre[k].copy()
        return None


_CAT_CLASS = (('human.pedestrian', 'pedestrian'), ('vehicle.bicycle', 'bicycle'), ('vehicle.motorcycle', 'motorcycle'),
              ('vehicle.bus', 'bus'), ('vehicle.truck', 'truck'), ('vehicle.trailer', 'truck'),
              ('vehicle.construction', 'truck'), ('vehicle.emergency', 'car'), ('vehicle.car', 'car'))


def class_of_category(cat: str) -> str:
    """A nuScenes category -> the ground truth's class names that core.detection maps (car, truck, bus, pedestrian,
    bicycle, motorcycle); '' for the rest."""
    for pre, cls in _CAT_CLASS:
        if str(cat).startswith(pre):
            return cls
    return ''


def _build_tracks(obs: dict, info: dict, by_motion: bool = False) -> dict:
    """{id: Track} from {id: [(t, centre, quat), …]} and {id: (cls, category, dynamic, size)}. by_motion: an object
    is moving when it is a person or its box moved more than 0.3 m over the drive (else it stays in the static
    scene)."""
    out = {}
    for i, rows in obs.items():
        rows = sorted(rows, key=lambda r: r[0])
        t = np.array([r[0] for r in rows])
        keep = np.r_[True, np.diff(t) > 1e-6]
        cls, cat, dyn, size = info[i]
        c = np.array([r[1] for r in rows])[keep]
        if by_motion:
            moved = float(np.max(np.linalg.norm(c - c[0], axis=1))) if len(c) else 0.0
            dyn = moved > 0.3 or str(cat).startswith('human.')
        out[i] = Track(int(i), cls, cat, bool(dyn), tuple(float(x) for x in size), t[keep], c,
                       np.array([r[2] for r in rows])[keep])
    return out


def _yaw_quat(yaw_rad: float) -> np.ndarray:
    return np.array([math.cos(yaw_rad / 2), 0.0, 0.0, math.sin(yaw_rad / 2)])


# =========================================================================== the drive
@dataclass
class DriveCamera:
    tag: str
    channel: str
    model: CamModel                     # at the recorded image size
    R: np.ndarray                       # optical -> vehicle
    t: np.ndarray                       # optical centre, vehicle frame
    frames: list                        # [(t s, image path, ground-truth frame dict or None)]
    readout_s: float = 0.0              # rolling shutter of the recording (0: one instant per frame)


@dataclass
class DriveLidar:
    tag: str
    channel: str
    R: np.ndarray                       # sensor -> vehicle
    t: np.ndarray
    sweeps: list                        # [(t_end s, path, kind 'pcdbin' | 'npy', ground-truth sweep dict or None)]
    period_s: float = 0.1
    sense: int = -1                     # -1 clockwise seen from above (azimuth falls with time), +1 counter-clockwise
    spinning: bool = True
    columns: int = 1024
    model: str = ''
    elevations: np.ndarray | None = None   # per ring, degrees (measured from the points)


@dataclass
class Drive:
    path: str                           # the folder or bag given
    root: str                           # the run folder (holding nuscenes/, ground_truth/, rig/), or the bag's cache
    source: str                         # nuscenes | rosbag
    t0_us: int
    origin: np.ndarray                  # global position of the scene origin (the first ego position), m
    traj: Trajectory
    cameras: dict
    lidars: dict
    tracks: dict
    gt: dict | None = None
    project: str = ''
    variant: str = ''
    image_scale: float = 1.0
    box_source: str = ''
    notes: list = field(default_factory=list)
    key: str = ''

    def duration(self) -> float:
        ts = [f[0] for c in self.cameras.values() for f in c.frames] + list(self.traj.t)
        return float(max(ts) - min(ts)) if ts else 0.0

    def info(self) -> dict:
        return {'path': self.path, 'source': self.source, 'project': self.project, 'variant': self.variant,
                'duration_s': round(self.duration(), 3), 'path_length_m': round(self.traj.length_m(), 2),
                'cameras': {k: len(c.frames) for k, c in self.cameras.items()},
                'lidars': {k: len(s.sweeps) for k, s in self.lidars.items()},
                'objects': len(self.tracks), 'moving_objects': sum(1 for t in self.tracks.values() if t.dynamic),
                'boxes': self.box_source or 'none', 'ground_truth': self.gt is not None,
                'image_scale': self.image_scale, 'notes': list(self.notes)}


def _load_json(path):
    try:
        with open(path, encoding='utf-8') as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _nus_dirs(path):
    """(version folder, dataroot, run folder) of a nuScenes recording given as the run folder, the dataroot or the
    version folder; None when it is not one."""
    p = os.path.abspath(os.path.expanduser(str(path)))
    if not os.path.isdir(p):
        return None
    cands = [p, os.path.join(p, 'nuscenes')]
    for c in cands:
        if os.path.isfile(os.path.join(c, 'sample_data.json')):
            data = os.path.dirname(c)
            return c, data, os.path.dirname(data) if os.path.basename(data) == 'nuscenes' else data
        if os.path.isdir(c):
            for n in sorted(os.listdir(c)):
                q = os.path.join(c, n)
                if os.path.isfile(os.path.join(q, 'sample_data.json')):
                    return q, c, os.path.dirname(c) if os.path.basename(c) == 'nuscenes' else c
    return None


def drive_key(path) -> str:
    """A hash of the drive's tables, rig and ground-truth index (names, sizes, times): a cache made from another drive
    or a changed one is not reused."""
    p = os.path.abspath(os.path.expanduser(str(path)))
    h = hashlib.sha256(p.encode())
    for root, dirs, names in os.walk(p):
        dirs[:] = sorted(d for d in dirs if d not in ('samples', 'sweeps', 'lidarseg', 'panoptic', 'maps'))
        for n in sorted(names):
            if n.endswith(('.json', '.mcap', '.db3', '.yaml')):
                try:
                    st = os.stat(os.path.join(root, n))
                except OSError:
                    continue
                h.update(f'{os.path.relpath(os.path.join(root, n), p)}:{st.st_size}:{int(st.st_mtime)}'.encode())
    return h.hexdigest()[:16]


def is_drive(path) -> bool:
    p = os.path.abspath(os.path.expanduser(str(path or '')))
    if _nus_dirs(p) is not None:
        return True
    if os.path.isfile(p) and p.endswith(('.mcap', '.db3')):
        return True
    return os.path.isdir(p) and (os.path.isfile(os.path.join(p, 'metadata.yaml')) or
                                 any(n.endswith(('.mcap', '.db3')) for n in os.listdir(p)))


def open_drive(path, cache_dir=None, progress=None, cancel=None) -> Drive:
    """Read a recorded drive (ValueError when it is not one). A bag's messages are copied into cache_dir (needed)."""
    p = os.path.abspath(os.path.expanduser(str(path or '')))
    if not p or not os.path.exists(p):
        raise ValueError(f'Not found: {os.path.basename(p) or "no drive chosen"}.')
    nd = _nus_dirs(p)
    if nd is not None:
        return _open_nuscenes(p, *nd)
    if is_drive(p):
        if cache_dir is None:
            raise ValueError('Reading a bag needs a cache folder.')
        return _open_bag(p, cache_dir, progress, cancel)
    raise ValueError(f'{os.path.basename(p)} is neither a nuScenes folder nor a ROS 2 bag.')


def _quat_wxyz_R(q) -> np.ndarray:
    return quat_to_rot(np.asarray(q, float).reshape(1, 4))[0]


def _open_nuscenes(path, ver, dataroot, run_dir) -> Drive:
    names = ('sensor', 'calibrated_sensor', 'sample_data', 'ego_pose')
    tab = {}
    for n in names:
        tab[n] = _load_json(os.path.join(ver, f'{n}.json'))
        if not isinstance(tab[n], list):
            raise ValueError(f'{os.path.basename(ver)}/{n}.json is missing or unreadable.')
    sens = {s['token']: s for s in tab['sensor'] if isinstance(s, dict)}
    cal = {c['token']: c for c in tab['calibrated_sensor'] if isinstance(c, dict)}
    ego = {e['token']: e for e in tab['ego_pose'] if isinstance(e, dict)}
    if not ego:
        raise ValueError('The drive has no ego poses (ego_pose.json is empty).')
    t0_us = min(int(e['timestamp']) for e in ego.values())
    first = min(ego.values(), key=lambda e: int(e['timestamp']))
    origin = np.asarray(first['translation'], float)
    et = np.array([(int(e['timestamp']) - t0_us) * 1e-6 for e in ego.values()])
    ep = np.array([e['translation'] for e in ego.values()], float) - origin
    eq = np.array([e['rotation'] for e in ego.values()], float)
    traj = Trajectory(et, ep, eq)
    rig = _load_json(os.path.join(run_dir, 'rig', 'sim_rig.json')) or {}
    rig_cams = {c.get('channel'): c for c in rig.get('cameras', []) if isinstance(c, dict)}
    rig_lids = {c.get('channel'): c for c in rig.get('lidars', []) if isinstance(c, dict)}
    ds = _load_json(os.path.join(run_dir, 'dataset.json')) or {}
    dset = ds.get('settings') or {}
    gt = None
    try:
        from . import groundtruth as GT
        gt = GT.read(run_dir)
    except ImportError:                                   # pragma: no cover
        gt = None
    gt_frames, gt_sweeps = {}, {}
    if gt is not None:
        for c in (gt.get('cameras') or {}).values():
            for fr in c.get('frames') or []:
                gt_frames[fr.get('sample_data_token')] = fr
        for c in (gt.get('lidars') or {}).values():
            for sw in c.get('sweeps') or []:
                gt_sweeps[sw.get('sample_data_token')] = sw
    by_cal = {}
    for sd in tab['sample_data']:
        by_cal.setdefault(sd.get('calibrated_sensor_token'), []).append(sd)
    cams, lids = {}, {}
    notes = []
    readout = 0.0
    if dset.get('rolling_shutter'):
        ro = (ds.get('scene') or {}).get('camera_readout_ms')
        readout = _f(ro, 0.0) / 1000.0
    for ctok, rows in by_cal.items():
        c = cal.get(ctok)
        if c is None:
            continue
        s = sens.get(c.get('sensor_token')) or {}
        ch, mod = str(s.get('channel', '')), str(s.get('modality', ''))
        rows = sorted(rows, key=lambda r: int(r['timestamp']))
        R = _quat_wxyz_R(c.get('rotation', (1, 0, 0, 0)))
        tr = np.asarray(c.get('translation', (0, 0, 0)), float)
        tag = (c.get('vantage_distortion') or {}).get('camera') or (c.get('vantage_sensor') or {}).get('tag') or \
            re.sub(r'^(CAM|LIDAR|RADAR)_', '', ch)
        if mod == 'camera' and c.get('camera_intrinsic'):
            w, h = int(rows[0].get('width') or 0), int(rows[0].get('height') or 0)
            rc = rig_cams.get(ch)
            if rc is not None and isinstance(rc.get('lens'), dict):
                model = CamModel.from_lens_block(rc['lens'])
                if (model.width, model.height) != (w, h) and w and h:
                    model = model.scaled(w, h)
            else:
                vd = c.get('vantage_distortion') or {}
                D = None
                if str(vd.get('model', '')) in ('kannala_brandt', 'equidistant'):
                    D = (list(vd.get('coefficients') or []) + [0.0] * 4)[:4]
                model = CamModel.from_opencv(c['camera_intrinsic'], D, w, h)
            frames = [((int(r['timestamp']) - t0_us) * 1e-6, os.path.join(dataroot, r['filename']),
                       gt_frames.get(r['token'])) for r in rows]
            cams[tag] = DriveCamera(tag, ch, model, R, tr, frames, readout)
        elif mod == 'lidar':
            rl = rig_lids.get(ch) or {}
            sw = [((int(r['timestamp']) - t0_us) * 1e-6, os.path.join(dataroot, r['filename']), 'pcdbin',
                   gt_sweeps.get(r['token'])) for r in rows]
            ts = np.array([x[0] for x in sw])
            rate = _f(rl.get('rate_hz'), 0.0) or (1.0 / float(np.median(np.diff(ts))) if len(ts) > 1 else 10.0)
            L = DriveLidar(tag, ch, R, tr, sw, 1.0 / max(rate, 0.1),
                           1 if str(rl.get('rotation', 'clockwise')).startswith('counter') else -1,
                           str(rl.get('scan', 'spinning')) == 'spinning', int(rl.get('columns') or 0),
                           str((c.get('vantage_sensor') or {}).get('model', '')))
            lids[tag] = L
    if not cams:
        raise ValueError('The drive has no camera images.')
    # boxes: the ground truth's (every camera frame), else nuScenes' sample_annotation
    obs, info = {}, {}
    box_source = ''
    if gt is not None:
        for cam in cams.values():
            for t, _, fr in cam.frames:
                if fr is None:
                    continue
                Rw, pw = traj.pose(t)
                for o in fr.get('objects') or ():
                    b = o.get('box3d') if isinstance(o, dict) else None
                    if not isinstance(b, dict) or o.get('id') is None:
                        continue
                    i = int(o['id'])
                    rows = obs.setdefault(i, [])
                    if any(abs(r[0] - t) < 1e-6 for r in rows):
                        continue
                    cv = np.asarray(b['center'], float)
                    Rb = Rw @ rot_zyx(float(b.get('yaw_deg', 0.0)), 0.0, 0.0)
                    rows.append((t, Rw @ cv + pw, rot_to_quat(Rb[None])[0]))
                    info[i] = (str(o.get('class') or ''), str(o.get('category') or ''), i < 1000,
                               tuple(b.get('size') or (1, 1, 1)))
        box_source = 'ground truth'
    else:
        ann = _load_json(os.path.join(ver, 'sample_annotation.json'))
        smp = _load_json(os.path.join(ver, 'sample.json'))
        inst = _load_json(os.path.join(ver, 'instance.json'))
        cats = _load_json(os.path.join(ver, 'category.json'))
        if isinstance(ann, list) and ann and isinstance(smp, list) and isinstance(inst, list):
            st_of = {s['token']: (int(s['timestamp']) - t0_us) * 1e-6 for s in smp}
            cat_of = {c['token']: c.get('name', '') for c in (cats or [])}
            inst_ix = {r['token']: k + 1 for k, r in enumerate(inst)}
            inst_cat = {r['token']: cat_of.get(r.get('category_token'), '') for r in inst}
            for a in ann:
                i = inst_ix.get(a.get('instance_token'))
                t = st_of.get(a.get('sample_token'))
                if i is None or t is None:
                    continue
                cat = inst_cat.get(a['instance_token'], '')
                w_, l_, h_ = (float(x) for x in a.get('size', (1, 1, 1)))
                obs.setdefault(i, []).append((t, np.asarray(a['translation'], float) - origin,
                                              np.asarray(a.get('rotation', (1, 0, 0, 0)), float)))
                info[i] = (class_of_category(cat), cat, cat.startswith(('vehicle.', 'human.', 'animal')),
                           (l_, w_, h_))
            if obs:
                box_source = 'sample_annotation'
    tracks = _build_tracks(obs, info, by_motion=box_source == 'sample_annotation')
    d = Drive(path, run_dir, 'nuscenes', t0_us, origin, traj, cams, lids, tracks, gt,
              str(ds.get('project') or (gt or {}).get('project') or rig.get('project') or ''),
              str(ds.get('variant') or (gt or {}).get('variant') or rig.get('variant') or ''),
              _f(dset.get('image_scale'), 1.0) or 1.0, box_source, notes)
    d.key = drive_key(run_dir)
    for L in d.lidars.values():
        _lidar_layout(d, L)
    return d


# ---- a bag: copied into the cache once
def _tag_of(frame_id: str, topic: str) -> str:
    f = str(frame_id or '')
    for suf in ('_optical_frame', '_link', '_frame'):
        if f.endswith(suf):
            return f[:-len(suf)]
    parts = [x for x in str(topic).split('/') if x]
    return parts[0].upper() if parts else f


def _open_bag(path, cache_dir, progress=None, cancel=None) -> Drive:
    from . import rosbag as RB
    key = drive_key(path)
    out = os.path.join(cache_dir, f'bag-{key}')
    index = os.path.join(out, 'index.json')
    doc = _load_json(index)
    if not (isinstance(doc, dict) and doc.get('key') == key):
        doc = _extract_bag(path, out, key, progress, cancel)
    t0_ns = int(doc['t0_ns'])
    poses = np.asarray(doc['poses'], float)
    if not len(poses):
        raise ValueError('The bag has no ego poses (/odom or /tf odom → base_link).')
    origin = poses[0, 1:4].copy()
    traj = Trajectory(poses[:, 0], poses[:, 1:4] - origin, poses[:, 4:8])
    cams, lids = {}, {}
    tf = {k: np.asarray(v, float) for k, v in doc['tf'].items()}

    def chain(frame):
        """4×4 base_link ← frame through /tf_static."""
        T = np.eye(4)
        f, n = frame, 0
        while f in tf and n < 8:
            T = tf[f] @ T
            f = doc['tf_parent'][f]
            n += 1
        return T if f == 'base_link' else None
    for tag, c in doc['cameras'].items():
        T = chain(c['frame_id'])
        if T is None or not c.get('info'):
            continue
        ci = c['info']
        D = ci['d'] if ci.get('distortion_model') == 'equidistant' else None
        model = CamModel.from_opencv(np.asarray(ci['k'], float).reshape(3, 3), D, int(ci['width']), int(ci['height']))
        frames = [((t - t0_ns) * 1e-9, os.path.join(out, f), None) for t, f in c['frames']]
        cams[tag] = DriveCamera(tag, c['topic'], model, T[:3, :3], T[:3, 3], frames, 0.0)
    for tag, s in doc['lidars'].items():
        T = chain(s['frame_id'])
        if T is None:
            continue
        sw = [((t - t0_ns) * 1e-9, os.path.join(out, f), 'npy', None) for t, f in s['sweeps']]
        ts = np.array([x[0] for x in sw])
        period = float(np.median(np.diff(ts))) if len(ts) > 1 else 0.1
        lids[tag] = DriveLidar(tag, s['topic'], T[:3, :3], T[:3, 3], sw, period, -1, True, 0, '')
    if not cams:
        raise ValueError('The bag has no camera images with camera_info and a /tf_static chain to base_link.')
    obs, info = {}, {}
    for t_ns, boxes in doc['boxes']:
        t = (t_ns - t0_ns) * 1e-9
        for b in boxes:
            i = int(b.get('id', 0))
            cat = str(b.get('category', ''))
            obs.setdefault(i, []).append((t, np.asarray(b['centre'], float) - origin, _yaw_quat(_f(b.get('yaw'), 0.0))))
            info[i] = (class_of_category(cat), cat, cat.startswith(('vehicle.', 'human.', 'animal')),
                       tuple(b.get('size_lwh') or (1, 1, 1)))
    d = Drive(path, out, 'rosbag', t0_ns // 1000, origin, traj, cams, lids, _build_tracks(obs, info, True), None,
              '', '', 1.0, 'bag boxes' if obs else '', list(doc.get('notes') or []))
    d.key = key
    for L in d.lidars.values():
        _lidar_layout(d, L)
    return d


def _extract_bag(path, out, key, progress=None, cancel=None) -> dict:
    from . import rosbag as RB
    os.makedirs(out, exist_ok=True)
    rd = RB.BagReader(path)
    try:
        topics = dict(rd.topics)
        want = [t for t, ty in topics.items() if ty.split('/')[-1] in (
            'CompressedImage', 'Image', 'CameraInfo', 'PointCloud2', 'TFMessage', 'Odometry', 'String')]
        total = max(sum(rd.counts.get(t, 0) for t in want), 1)
        doc = {'key': key, 'cameras': {}, 'lidars': {}, 'tf': {}, 'tf_parent': {}, 'poses': [], 'boxes': [],
               'notes': [], 't0_ns': None}
        n = 0
        odom_seen = False
        for topic, t_ns, msg in rd.messages(want):
            n += 1
            if n % 200 == 0:
                _prog(progress, 0.3 * n / total, 'Reading the bag')
                if cancel is not None and cancel():
                    raise Cancelled()
            if not isinstance(msg, dict):
                continue
            ty = topics[topic].split('/')[-1]
            hdr = msg.get('header') or {}
            stamp = RB.stamp_ns(hdr.get('stamp') or {}) or int(t_ns)
            if ty in ('CompressedImage', 'Image'):
                tag = _tag_of(hdr.get('frame_id', ''), topic)
                c = doc['cameras'].setdefault(tag, {'topic': topic.rsplit('/', 2)[0], 'frame_id': hdr.get('frame_id'),
                                                    'frames': [], 'info': None})
                rel = f'{tag}_{stamp}.' + ('jpg' if ty == 'CompressedImage' and 'jp' in
                                           str(msg.get('format', '')).lower() else 'png')
                if ty == 'CompressedImage':
                    with open(os.path.join(out, rel), 'wb') as fh:
                        fh.write(bytes(msg['data']))
                else:
                    import cv2
                    img = RB.image_array(msg, to_rgb=True)
                    if img.ndim == 3:
                        img = cv2.cvtColor(img[..., :3], cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(out, rel), img)
                c['frames'].append((stamp, rel))
            elif ty == 'CameraInfo':
                tag = _tag_of(hdr.get('frame_id', ''), topic)
                c = doc['cameras'].setdefault(tag, {'topic': topic.rsplit('/', 1)[0], 'frame_id': hdr.get('frame_id'),
                                                    'frames': [], 'info': None})
                if c['info'] is None:
                    c['info'] = {'width': int(msg['width']), 'height': int(msg['height']),
                                 'distortion_model': str(msg.get('distortion_model', '')),
                                 'd': [float(x) for x in msg.get('d') or []], 'k': [float(x) for x in msg['k']]}
            elif ty == 'PointCloud2':
                if '/detections' in topic:
                    continue                                   # radar
                a = RB.pointcloud2_array(msg)
                names = a.dtype.names or ()
                if not all(k in names for k in ('x', 'y', 'z')):
                    continue
                tag = _tag_of(hdr.get('frame_id', ''), topic)
                s = doc['lidars'].setdefault(tag, {'topic': topic, 'frame_id': hdr.get('frame_id'), 'sweeps': []})
                arr = np.zeros((len(a), 5), np.float32)
                arr[:, 0], arr[:, 1], arr[:, 2] = a['x'], a['y'], a['z']
                arr[:, 3] = a['ring'] if 'ring' in names else -1
                tt = None
                for k in ('t', 'time', 'timestamp'):
                    if k in names:
                        tt = np.asarray(a[k], float)
                        break
                if tt is not None and len(tt) and np.nanmax(np.abs(tt)) < 1.0:
                    arr[:, 4] = tt
                else:
                    arr[:, 4] = np.nan
                rel = f'{tag}_{stamp}.npy'
                np.save(os.path.join(out, rel), arr)
                s['sweeps'].append((stamp, rel))
            elif ty == 'TFMessage':
                for tr in msg.get('transforms') or []:
                    M = RB.transform_matrix(tr)
                    par = tr['header']['frame_id'].lstrip('/')
                    ch = tr['child_frame_id'].lstrip('/')
                    if topic.endswith('tf_static'):
                        doc['tf'][ch] = M.tolist()
                        doc['tf_parent'][ch] = par
                    elif ch == 'base_link' and not odom_seen:
                        st = RB.stamp_ns(tr['header'].get('stamp') or {}) or int(t_ns)
                        qw = rot_to_quat(M[:3, :3][None])[0]
                        doc['poses'].append([st, *M[:3, 3].tolist(), *[float(x) for x in qw]])
            elif ty == 'Odometry':
                if not odom_seen:
                    doc['poses'] = []
                    odom_seen = True
                po = msg['pose']['pose']
                o = po['orientation']
                doc['poses'].append([stamp, po['position']['x'], po['position']['y'], po['position']['z'],
                                     o['w'], o['x'], o['y'], o['z']])
            elif ty == 'String' and topic.endswith('/ground_truth/boxes'):
                try:
                    bx = json.loads(msg.get('data', ''))
                    doc['boxes'].append([int(bx.get('stamp_ns', stamp)), bx.get('boxes') or []])
                except (ValueError, TypeError):
                    pass
    finally:
        rd.close()
    stamps = [p[0] for p in doc['poses']] + [f[0] for c in doc['cameras'].values() for f in c['frames']]
    doc['t0_ns'] = int(min(p[0] for p in doc['poses'])) if doc['poses'] else int(min(stamps, default=0))
    doc['poses'] = [[(p[0] - doc['t0_ns']) * 1e-9] + list(p[1:]) for p in doc['poses']]
    with open(os.path.join(out, 'index.json'), 'w', encoding='utf-8') as fh:
        json.dump(doc, fh)
    return doc


def _lidar_layout(d: Drive, L: DriveLidar):
    """Measure a recorded lidar's ring elevations (the median elevation of each ring's points over a few sweeps),
    its columns (the rig file's, else the most points a ring has in a sweep) and turning sense (from the azimuth
    order of the first sweep)."""
    pick = L.sweeps[:: max(1, len(L.sweeps) // 4)][:4]
    el, rings, cnt = [], [], 0
    sense_votes = []
    for sw in pick:
        try:
            a = load_sweep(sw)
        except (OSError, ValueError):
            continue
        P, ring = a[:, :3], a[:, 3]
        if (ring < 0).all():
            continue
        h = np.hypot(P[:, 0], P[:, 1])
        el.append(np.degrees(np.arctan2(P[:, 2], h)))
        rings.append(ring.astype(int))
        if len(P) > 100:
            dd = np.diff(np.degrees(np.arctan2(P[:, 1], P[:, 0])))
            sense_votes.append(np.sign(np.sum((dd + 180.0) % 360.0 - 180.0)))
            cnt = max(cnt, int(np.bincount(ring.astype(int)).max()))
    if el:
        e = np.concatenate(el)
        r = np.concatenate(rings)
        nr = int(r.max()) + 1
        elev = np.full(nr, np.nan)
        for k in range(nr):
            m = r == k
            if m.any():
                elev[k] = float(np.median(e[m]))
        good = np.isfinite(elev)
        idx = np.arange(nr)
        if good.sum() >= 2:
            # rings without points (beams on the own vehicle, dropped as self hits): an even fan when the measured
            # rings are one (residuals under 0.05°), else interpolated and extended with the end spacings
            a, b = np.polyfit(idx[good], elev[good], 1)
            if np.max(np.abs(a * idx[good] + b - elev[good])) < 0.05:
                elev = a * idx + b
            else:
                gi = idx[good]
                e = np.interp(idx, gi, elev[good])
                lo, hi = gi[0], gi[-1]
                s0 = (elev[gi[1]] - elev[gi[0]]) / max(gi[1] - gi[0], 1)
                s1 = (elev[gi[-1]] - elev[gi[-2]]) / max(gi[-1] - gi[-2], 1)
                e[:lo] = elev[lo] - s0 * (lo - idx[:lo])
                e[hi + 1:] = elev[hi] + s1 * (idx[hi + 1:] - hi)
                elev = e
        L.elevations = elev
    if not L.columns:
        L.columns = max(cnt, 64)
    if sense_votes and abs(float(np.mean(sense_votes))) > 0.5:
        L.sense = 1 if np.mean(sense_votes) > 0 else -1


def load_sweep(sw) -> np.ndarray:
    """A recorded sweep as (n, 5) float32: x, y, z (sensor frame), ring (−1 unknown), time from the sweep's start (s,
    NaN unknown)."""
    _, path, kind, _ = sw
    if kind == 'npy':
        return np.load(path)
    a = np.fromfile(path, np.float32)
    n = len(a) // 5
    a = a[:n * 5].reshape(n, 5)
    out = np.empty((n, 5), np.float32)
    out[:, :3] = a[:, :3]
    out[:, 3] = a[:, 4]
    out[:, 4] = np.nan
    return out


def firing_times(L: DriveLidar, sw, a: np.ndarray) -> np.ndarray:
    """Seconds (drive time) each point of a sweep was fired: its own time field, else (a spinning lidar) its azimuth
    behind the sweep's first point in the turning sense, else the sweep's stamp."""
    t_end = sw[0]
    tt = a[:, 4]
    if np.isfinite(tt).all() and len(tt):
        return t_end - L.period_s + tt.astype(float)
    if not L.spinning or not len(a):
        return np.full(len(a), t_end)
    az = np.degrees(np.arctan2(a[:, 1], a[:, 0]))
    a0 = float(np.median(az[:min(32, len(az))]))
    frac = np.mod((az - a0) * L.sense, 360.0) / 360.0
    return t_end - L.period_s + frac * L.period_s


# =========================================================================== the scene
@dataclass
class Scene:
    """Gaussians in the local frame (global − origin): static ones (actor −1) in it, an actor's in its box frame."""
    means: np.ndarray                   # (N, 3) float32
    scales: np.ndarray                  # (N, 3) σ, m
    quats: np.ndarray                   # (N, 4) w, x, y, z
    opacity: np.ndarray                 # (N,) [0, 1]
    colours: np.ndarray                 # (N, 3) [0, 1] RGB
    actor: np.ndarray                   # (N,) int32: index into actor_ids, −1 static
    obj: np.ndarray                     # (N,) int32: annotated object id, 0 none
    actor_ids: list
    sky: np.ndarray                     # (SKY_H, SKY_W, 3)
    origin: np.ndarray
    source: str = 'lidar'
    info: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(len(self.means))

    def save(self, folder) -> str:
        os.makedirs(folder, exist_ok=True)
        p = os.path.join(folder, 'scene.npz')
        tmp = p + '.part.npz'
        np.savez_compressed(tmp, means=self.means, scales=self.scales, quats=self.quats, opacity=self.opacity,
                            colours=self.colours, actor=self.actor, obj=self.obj,
                            actor_ids=np.asarray(self.actor_ids, np.int64), sky=self.sky, origin=self.origin)
        os.replace(tmp, p)
        with open(os.path.join(folder, 'scene.json'), 'w', encoding='utf-8') as fh:
            json.dump({'schema': SCHEMA + '#scene', 'source': self.source, 'info': self.info}, fh, indent=1,
                      default=float)
        return p

    @staticmethod
    def load(folder) -> 'Scene | None':
        p = os.path.join(folder, 'scene.npz')
        meta = _load_json(os.path.join(folder, 'scene.json'))
        if not os.path.isfile(p) or not isinstance(meta, dict):
            return None
        try:
            with np.load(p, allow_pickle=False) as z:
                return Scene(z['means'], z['scales'], z['quats'], z['opacity'], z['colours'], z['actor'], z['obj'],
                             [int(x) for x in z['actor_ids']], z['sky'], z['origin'], str(meta.get('source', 'lidar')),
                             dict(meta.get('info') or {}))
        except (OSError, ValueError, KeyError):
            return None


def _voxel_centroids(P, size):
    """Centroids of the points in each voxel of the given size (per point sizes allowed): (centroids, counts)."""
    size = np.broadcast_to(np.asarray(size, float), (len(P),))
    key = np.floor(P / size[:, None]).astype(np.int64)
    key = np.concatenate([key, np.round(np.log2(size) * 4).astype(np.int64)[:, None]], 1)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    out = np.zeros((len(cnt), 3))
    np.add.at(out, inv, P)
    return out / cnt[:, None], cnt


def _adaptive_voxels(P, path_pts, base, budget):
    """Thin P to voxel centroids, the voxel base within NEAR_M of the path and doubling with every doubling of the
    distance (at most 8×), the base grown by 20 % steps until the count fits the budget. Returns (centroids, base,
    the voxel size of each centroid)."""
    from scipy.spatial import cKDTree
    if not len(P):
        return P, base, np.zeros(0)
    tree = cKDTree(path_pts) if len(path_pts) else None

    def lev(Q):
        d = tree.query(Q, k=1)[0] if tree is not None else np.zeros(len(Q))
        return np.clip(np.floor(np.log2(np.maximum(d, NEAR_M) / NEAR_M)), 0, 3)
    level = lev(P)
    for _ in range(40):
        C, _ = _voxel_centroids(P, base * 2.0 ** level)
        if len(C) <= budget:
            break
        base *= 1.2
    C = C[:budget]
    return C, base, base * 2.0 ** lev(C)


def _shapes(P, voxel):
    """(scales (n, 3), quats (n, 4)) from the 8-neighbour principal components (module docstring, step 2); voxel: the
    thinning voxel size (one, or one per point), which caps σ at three quarters of it (an isolated point stays small,
    not a blob across the street)."""
    from scipy.spatial import cKDTree
    n = len(P)
    if n == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 4), np.float32)
    k = min(9, n)
    tree = cKDTree(P)
    dist, idx = tree.query(P, k=k)
    dist, idx = np.atleast_2d(dist), np.atleast_2d(idx)
    vox = np.broadcast_to(np.asarray(voxel, float), (n,))
    nn = dist[:, 1:min(4, k)].mean(1) if k > 1 else vox.copy()
    sig = np.clip(0.5 * nn, 0.004, np.maximum(0.75 * vox, 0.004))
    scales = np.repeat(sig[:, None], 3, 1)
    quats = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1))
    if k >= 4:
        Q = P[idx] - P[idx].mean(1, keepdims=True)
        C = np.einsum('nki,nkj->nij', Q, Q) / k
        lam, vec = np.linalg.eigh(C)                     # ascending
        flat = lam[:, 0] < 0.1 * np.maximum(lam[:, 1], 1e-12)
        R = vec.copy()
        R[np.linalg.det(R) < 0, :, 0] *= -1              # right-handed, the normal last
        R = R[:, :, [1, 2, 0]]
        scales[flat, 2] = np.maximum(0.1 * sig[flat], 0.003)
        quats[flat] = rot_to_quat(R[flat])
    return scales.astype(np.float32), quats.astype(np.float32)


def _ego_mask_fn(drive: Drive, project=None, engine=None):
    """A function (camera, frame, (h, w)) -> bool mask of the recording vehicle's pixels at that size, or None."""
    from . import groundtruth as GT
    import cv2
    bodies = {}
    var = _variant_named(project, drive.variant) if project is not None else None

    def fn(cam: DriveCamera, fr, size):
        h, w = size
        if fr[2] is not None and fr[2].get('semantic') and drive.gt is not None:
            try:
                sem = GT.load_semantic(drive.gt, fr[2])
                return cv2.resize((sem == EGO_CLASS).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
            except OSError:
                pass
        if project is None or engine is None or var is None:
            return None
        key = (cam.tag, w, h)
        if key not in bodies:
            bodies[key] = _body(project, engine, var, cam.R, cam.t, cam.model.scaled(w, h))[0]
        return bodies[key]
    return fn


def _variant_named(project, name):
    if project is None:
        return None
    for v in project.variants:
        if v.name == name:
            return v
    return None


def _load_rgb(path, size=None) -> np.ndarray:
    import cv2
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise OSError(f'cannot read {os.path.basename(path)}')
    if size is not None and (img.shape[1], img.shape[0]) != tuple(size):
        img = cv2.resize(img, tuple(size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def held_out(k: int, every: int) -> bool:
    return every > 0 and k % every == every // 2


def _sky_dir_to_uv(D):
    az = np.degrees(np.arctan2(D[..., 1], D[..., 0]))
    el = np.degrees(np.arcsin(np.clip(D[..., 2], -1, 1)))
    u = (az + 180.0) / 360.0 * SKY_W
    v = (90.0 - np.maximum(el, SKY_EL_LO)) / (90.0 - SKY_EL_LO) * SKY_H
    return u, v


def reconstruct(drive: Drive, st: SplatSettings, project=None, engine=None, progress=None, cancel=None) -> Scene:
    """The drive's street scene as Gaussians (module docstring); raises Cancelled when cancelled."""
    t_start = time.perf_counter()
    stop = cancel or (lambda: False)
    rng = np.random.default_rng(st.seed)
    if not drive.lidars:
        raise ValueError('The drive has no lidar: a scene cannot be rebuilt from it (import a .ply instead).')
    use_boxes = st.dynamic == 'boxes' and bool(drive.tracks)
    dyn = {i: tr for i, tr in drive.tracks.items() if tr.dynamic} if use_boxes else {}
    # ---- 1. lidar points into the world
    static_pts, actor_pts = [], {}
    n_in, n_ego, n_dyn_drop = 0, 0, 0
    path = drive.traj.pos
    jobs = []
    for L in drive.lidars.values():
        for k, sw in enumerate(L.sweeps):
            if held_out(k, st.holdout_every) or k % st.sweep_step:
                continue
            jobs.append((L, sw))
    try:
        from . import groundtruth as GT
        ego_idx = GT.NUSC_INDEX.get('vehicle.ego', 31)
    except ImportError:                                   # pragma: no cover
        GT, ego_idx = None, 31
    for j, (L, sw) in enumerate(jobs):
        if stop():
            raise Cancelled()
        _prog(progress, 0.25 * j / max(len(jobs), 1), f'Lidar sweeps {j + 1}/{len(jobs)}')
        a = load_sweep(sw)
        if not len(a):
            continue
        rr = np.linalg.norm(a[:, :3], axis=1)
        keep = (rr > 0.5) & (rr <= st.max_range_m) & np.isfinite(rr)
        lab = None
        if sw[3] is not None and drive.gt is not None and GT is not None and sw[3].get('panoptic'):
            try:
                pan = GT.load_panoptic(drive.gt, sw[3]).astype(np.int64)
                if len(pan) == len(a):
                    lab = pan
            except (OSError, ValueError, KeyError):
                lab = None
        if lab is not None:
            ego = (lab // 1000) == ego_idx
            n_ego += int((ego & keep).sum())
            keep &= ~ego
        a = a[keep]
        lab = lab[keep] if lab is not None else None
        n_in += len(a)
        tf = firing_times(L, sw, a)
        Rw, pw = drive.traj.at(tf)
        Pv = a[:, :3].astype(float) @ L.R.T + L.t
        Pw = np.einsum('nij,nj->ni', Rw, Pv) + pw
        is_dyn = np.zeros(len(a), bool)
        if dyn:
            if lab is not None:
                inst = lab % 1000
                spawned = (inst > 0) & (inst < 500)
                is_dyn |= spawned
            ts = 0.5 * (tf.min() + tf.max())
            for i, tr in dyn.items():
                po = tr.pose(ts)
                if po is None:
                    continue
                Rb, cb = po
                if lab is not None:
                    sel = np.flatnonzero((lab % 1000) == i)
                else:
                    q = (Pw - cb) @ Rb
                    sel = np.flatnonzero(np.all(np.abs(q) <= np.asarray(tr.size) / 2 + 0.6, 1))
                if not len(sel):
                    continue
                # the box at the points' own firing time (an object spans a few milliseconds of a sweep)
                po = tr.pose(float(np.median(tf[sel])))
                if po is None:
                    continue
                Rb, cb = po
                loc = (Pw[sel] - cb) @ Rb
                inside = np.all(np.abs(loc) <= np.asarray(tr.size) / 2 + 0.25, 1)
                actor_pts.setdefault(i, []).append(loc[inside])
                is_dyn[sel] = True
            n_dyn_drop += int(is_dyn.sum())
        static_pts.append(_voxel_centroids(Pw[~is_dyn], st.voxel_m * 0.5)[0])
    if not static_pts:
        raise ValueError('No lidar points could be used.')
    P = np.concatenate(static_pts)
    _prog(progress, 0.27, 'Thinning the points')
    # actors first (at most a quarter of the budget, 4000 each), then the static scene in what is left
    a_means, a_sc, a_qs, a_act, a_obj, actor_ids = [], [], [], [], [], []
    keys = [i for i in sorted(actor_pts) if sum(len(x) for x in actor_pts[i]) >= 5]
    cap = max(50, min(4000, (st.max_gaussians // 4) // max(len(keys), 1)))
    for i in keys:
        A = np.concatenate(actor_pts[i])
        vox = 0.04
        C, _ = _voxel_centroids(A, vox)
        while len(C) > cap:
            vox *= 1.25
            C, _ = _voxel_centroids(A, vox)
        s_, q_ = _shapes(C, vox)
        a_means.append(C)
        a_sc.append(s_)
        a_qs.append(q_)
        a_act.append(np.full(len(C), len(actor_ids), np.int32))
        a_obj.append(np.full(len(C), i, np.int32))
        actor_ids.append(int(i))
    n_act = sum(len(c) for c in a_means)
    Ps, base, vsize = _adaptive_voxels(P, path, st.voxel_m, st.max_gaussians - n_act)
    if stop():
        raise Cancelled()
    scales, quats = _shapes(Ps, vsize)
    means = [Ps] + a_means
    sc = [scales] + a_sc
    qs = [quats] + a_qs
    actor = [np.full(len(Ps), -1, np.int32)] + a_act
    obj = [np.zeros(len(Ps), np.int32)] + a_obj
    means = np.concatenate(means).astype(np.float32)
    scales = np.concatenate(sc).astype(np.float32)
    quats = np.concatenate(qs).astype(np.float32)
    actor = np.concatenate(actor)
    obj = np.concatenate(obj)
    # static annotated objects (parked cars, lights, signs) own the static Gaussians inside their boxes
    for i, tr in drive.tracks.items():
        if (tr.dynamic and use_boxes) or not len(tr.t):
            continue
        Rb, cb = quat_to_rot(tr.quat[:1])[0], tr.centre[0]
        q = (means.astype(float) - cb) @ Rb
        m = (actor < 0) & np.all(np.abs(q) <= np.asarray(tr.size) / 2 + 0.05, 1) & (obj == 0)
        obj[m] = i
    n = len(means)
    scene = Scene(means, scales, quats, np.full(n, 0.9, np.float32), np.full((n, 3), 0.5, np.float32), actor, obj,
                  actor_ids, np.full((SKY_H, SKY_W, 3), 0.6, np.float32), drive.origin.copy(), 'lidar')
    _prog(progress, 0.3, 'Colours from the images')
    seen = colour_scene(scene, drive, st, project, engine, progress=lambda f, t='': _prog(progress, 0.3 + 0.15 * f, t),
                        cancel=stop)
    scene.info = {'points_used': int(n_in), 'ego_points_dropped': int(n_ego), 'actor_points': int(n_dyn_drop),
                  'voxel_base_m': round(float(base), 4), 'gaussians': n, 'static': int((actor < 0).sum()),
                  'actors': len(actor_ids), 'coloured_share': round(float(seen), 4), 'sweeps': len(jobs),
                  'dynamic': 'boxes' if use_boxes else ('ignore' if st.dynamic == 'ignore' else 'no boxes'),
                  'box_source': drive.box_source}
    if st.optimise and st.iterations > 0:
        opt = optimise(scene, drive, st, project, engine, rng,
                       progress=lambda f, t='': _prog(progress, 0.45 + 0.5 * f, t), cancel=stop)
        scene.info['optimisation'] = opt
    scene.info['seconds'] = round(time.perf_counter() - t_start, 2)
    scene.info['drive_key'] = drive.key
    return scene


def _frames_for(drive: Drive, st: SplatSettings, train=True, step=1):
    """[(camera, frame index, frame)] of the training (or held-out) frames, every step-th."""
    out = []
    for cam in drive.cameras.values():
        k_used = 0
        for k, fr in enumerate(cam.frames):
            ho = held_out(k, st.holdout_every)
            if ho == train:
                continue
            if k_used % step == 0:
                out.append((cam, k, fr))
            k_used += 1
    return out


def _train_size(cam: DriveCamera, scale: float):
    return max(16, int(round(cam.model.width * scale))), max(16, int(round(cam.model.height * scale)))


def _posed_means(scene: Scene, drive: Drive, t: float):
    """World (local frame) centres of every Gaussian at time t and whether each is shown (actors out of their span
    are not)."""
    P = scene.means.astype(float).copy()
    shown = np.ones(scene.n, bool)
    for a, i in enumerate(scene.actor_ids):
        m = scene.actor == a
        po = drive.tracks[i].pose(t) if i in drive.tracks else None
        if po is None:
            shown[m] = False
            continue
        Rb, cb = po
        P[m] = P[m] @ Rb.T + cb
    return P, shown


def colour_scene(scene: Scene, drive: Drive, st: SplatSettings, project=None, engine=None, progress=None,
                 cancel=None) -> float:
    """Colour the Gaussians (and the sky) from the training frames (module docstring, step 3); returns the share of
    Gaussians seen by some frame."""
    import cv2
    from scipy.spatial import cKDTree
    stop = cancel or (lambda: False)
    acc = np.zeros((scene.n, 3))
    wsum = np.zeros(scene.n)
    sky = np.zeros((SKY_H, SKY_W, 3))
    sky_w = np.zeros((SKY_H, SKY_W))
    ego_fn = _ego_mask_fn(drive, project, engine)
    work = _frames_for(drive, st, True, st.colour_step)
    for j, (cam, k, fr) in enumerate(work):
        if stop():
            raise Cancelled()
        _prog(progress, j / max(len(work), 1), f'Colours · {cam.tag} frame {k + 1}')
        w, h = _train_size(cam, min(st.train_scale * 2, 1.0))
        try:
            img = _load_rgb(fr[1], (w, h))
        except OSError:
            continue
        m = cam.model.scaled(w, h)
        cell = max(1, int(round(min(w, h) / 120)))           # about 4 px on a 480 px image
        P, shown = _posed_means(scene, drive, fr[0])
        Rw, pw = drive.traj.pose(fr[0])
        o = Rw @ cam.t + pw
        O2W = Rw @ cam.R
        Q = (P - o) @ O2W
        rng_ = np.linalg.norm(Q, axis=1)
        u, v, th, ok = m.project(Q)
        ok &= shown & (rng_ > 0.3) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        ix = np.flatnonzero(ok)
        if not len(ix):
            continue
        cw, ch = (w + cell - 1) // cell, (h + cell - 1) // cell
        ci = (v[ix] // cell).astype(int) * cw + (u[ix] // cell).astype(int)
        zb = np.full(cw * ch, np.inf, np.float32)
        np.minimum.at(zb, ci, rng_[ix].astype(np.float32))
        zb2 = cv2.erode(zb.reshape(ch, cw), np.ones((3, 3), np.uint8), borderType=cv2.BORDER_REPLICATE)
        vis = rng_[ix] <= zb2.ravel()[ci] * 1.02 + 0.2
        ego = ego_fn(cam, fr, (h, w))
        if ego is not None:
            vis &= ~ego[v[ix].astype(int), u[ix].astype(int)]
        ix = ix[vis]
        col = cv2.remap(img, u[ix].astype(np.float32)[:, None] - 0.5, v[ix].astype(np.float32)[:, None] - 0.5,
                        cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)[:, 0, :]
        wt = 1.0 / (1.0 + rng_[ix] / 20.0)
        np.add.at(acc, ix, col * wt[:, None])
        np.add.at(wsum, ix, wt)
        # sky: pixels of cells nothing projects into (after the erosion), above the horizon
        empty = cv2.dilate(np.isfinite(zb.reshape(ch, cw)).astype(np.uint8), np.ones((3, 3), np.uint8)) == 0
        D, okr = m.rays()
        Dw = D @ O2W.T
        emp = cv2.resize(empty.astype(np.uint8), (cw * cell, ch * cell), interpolation=cv2.INTER_NEAREST)[:h, :w] > 0
        sel = emp & okr & (Dw[..., 2] > -0.05)
        if ego is not None:
            sel &= ~ego
        if sel.any():
            su, sv = _sky_dir_to_uv(Dw[sel])
            si = np.clip(sv.astype(int), 0, SKY_H - 1) * SKY_W + np.clip(su.astype(int), 0, SKY_W - 1)
            sky_flat = sky.reshape(-1, 3)
            np.add.at(sky_flat, si, img[sel])
            np.add.at(sky_w.reshape(-1), si, 1.0)
    seen = wsum > 0
    cols = np.where(seen[:, None], acc / np.maximum(wsum, 1e-12)[:, None], 0.5)
    if seen.any() and (~seen).any():
        tree = cKDTree(scene.means[seen])
        _, nn = tree.query(scene.means[~seen], k=1)
        cols[~seen] = cols[seen][nn]
    scene.colours = np.clip(cols, 0, 1).astype(np.float32)
    if sky_w.sum() > 0:
        mean = sky.reshape(-1, 3).sum(0) / sky_w.sum()
        rows = np.where((sky_w.sum(1) > 0)[:, None], sky.sum(1) / np.maximum(sky_w.sum(1), 1e-9)[:, None], mean)
        filled = np.where(sky_w[..., None] > 0, sky / np.maximum(sky_w, 1e-9)[..., None], rows[:, None, :])
        scene.sky = np.clip(filled, 0, 1).astype(np.float32)
    return float(seen.mean()) if scene.n else 0.0


# =========================================================================== the scene in torch
class TorchScene:
    """The scene's parameters as torch tensors on one device (optimisable), and its Gaussians posed at an instant."""

    def __init__(self, scene: Scene, drive: Drive | None, dev, grad: bool = False, positions: bool = False):
        import torch
        self.scene, self.drive, self.dev = scene, drive, dev
        T = lambda a, dt=torch.float32: torch.tensor(np.asarray(a), dtype=dt, device=dev)  # noqa: E731
        self.means = T(scene.means).requires_grad_(grad and positions)
        self.log_s = torch.log(T(scene.scales).clamp(min=1e-4)).requires_grad_(grad)
        self.quat = T(scene.quats).requires_grad_(grad)
        self.op_logit = torch.logit(T(scene.opacity).clamp(1e-4, 1 - 1e-4)).requires_grad_(grad)
        self.col_logit = torch.logit(T(scene.colours).clamp(0.01, 0.99)).requires_grad_(grad)
        self.sky_logit = torch.logit(T(scene.sky).clamp(0.01, 0.99)).permute(2, 0, 1)[None].contiguous() \
            .requires_grad_(grad)
        self.actor = T(scene.actor, torch.long)
        self.obj = T(scene.obj, torch.long)
        self.na = len(scene.actor_ids)

    def params(self) -> list:
        return [self.means, self.log_s, self.quat, self.op_logit, self.col_logit, self.sky_logit]

    def write_back(self):
        """The optimised values into the numpy scene."""
        import torch
        with torch.no_grad():
            sc = self.scene
            sc.means = self.means.detach().cpu().numpy().astype(np.float32)
            sc.scales = torch.exp(self.log_s).detach().cpu().numpy().astype(np.float32)
            q = self.quat.detach()
            sc.quats = (q / q.norm(dim=1, keepdim=True).clamp(min=1e-12)).cpu().numpy().astype(np.float32)
            sc.opacity = torch.sigmoid(self.op_logit).detach().cpu().numpy().astype(np.float32)
            sc.colours = torch.sigmoid(self.col_logit).detach().cpu().numpy().astype(np.float32)
            sc.sky = torch.sigmoid(self.sky_logit)[0].permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)

    def posed(self, t: float):
        """(world centres (N, 3), world rotations (N, 3, 3), shown (N,) bool) at time t (actors at their box poses)."""
        import torch
        Rq = SR.quat_to_rot(self.quat)
        if self.na == 0 or self.drive is None:
            return self.means, Rq, torch.ones(len(self.means), dtype=torch.bool, device=self.dev)
        Rt = np.tile(np.eye(3), (self.na + 1, 1, 1))
        ct = np.zeros((self.na + 1, 3))
        alive = np.zeros(self.na + 1, bool)
        alive[-1] = True
        for a, i in enumerate(self.scene.actor_ids):
            tr = self.drive.tracks.get(i)
            po = tr.pose(t) if tr is not None else None
            if po is not None:
                Rt[a], ct[a] = po
                alive[a] = True
        ai = torch.where(self.actor < 0, torch.full_like(self.actor, self.na), self.actor)
        Ra = torch.tensor(Rt, dtype=torch.float32, device=self.dev)[ai]
        ca = torch.tensor(ct, dtype=torch.float32, device=self.dev)[ai]
        P = torch.einsum('nij,nj->ni', Ra, self.means) + ca
        return P, Ra @ Rq, torch.tensor(alive, device=self.dev)[ai]

    def sky_at(self, Dw):
        """Background colours (K, 3) for world directions (K, 3) (torch) from the sky texture."""
        import torch
        import torch.nn.functional as Fn
        az = torch.rad2deg(torch.atan2(Dw[:, 1], Dw[:, 0]))
        el = torch.rad2deg(torch.asin(Dw[:, 2].clamp(-1, 1)))
        x = (az + 180.0) / 360.0 * 2 - 1
        y = (90.0 - el.clamp(min=SKY_EL_LO)) / (90.0 - SKY_EL_LO) * 2 - 1
        g = torch.stack([x, y], -1)[None, None]
        return Fn.grid_sample(torch.sigmoid(self.sky_logit), g, mode='bilinear', padding_mode='border',
                              align_corners=False)[0, :, 0].T


_RAYS = {}


def _normals(Rc, s, Pc):
    """Each Gaussian's thinnest axis in the frame of Rc (its rotation there); the ray towards it for a Gaussian not
    flatter than half its middle scale (a sphere: no plane)."""
    import torch
    j = s.argmin(1)
    n = torch.gather(Rc, 2, j[:, None, None].expand(-1, 3, 1))[..., 0]
    iso = s.min(1).values > 0.5 * s.median(1).values
    rr = torch.linalg.norm(Pc, dim=1, keepdim=True).clamp(min=1e-6)
    return torch.where(iso[:, None], Pc / rr, n)


def _rays_of(model: CamModel):
    key = (model.width, model.height, round(model.fx, 6), round(model.fy, 6), round(model.cx, 6), round(model.cy, 6),
           model.projection, model.kb, round(model.max_theta, 8))
    r = _RAYS.get(key)
    if r is None:
        if len(_RAYS) > 64:
            _RAYS.clear()
        r = _RAYS[key] = model.rays()
    return r


def render_view(ts: TorchScene, traj: Trajectory, model: CamModel, R_cv, t_cv, t: float, readout_s: float = 0.0,
                want_index: bool = False, depth: bool = False, grad: bool = False) -> dict:
    """One camera image at drive time t (the middle row's exposure centre): {'rgb' (h, w, 3) torch, 'alpha' (h, w),
    'depth' (h, w) expected range in m (depth), 'index' (h, w) the dominant Gaussian (want_index), 'cut'}. A rolling
    shutter (readout_s > 0) renders up to MAX_BANDS row bands, each at its own ego pose."""
    import torch
    w, h = model.width, model.height
    dev = ts.dev
    TH = (h + SR.TILE - 1) // SR.TILE
    nb = 1 if readout_s <= 0 else max(1, min(MAX_BANDS, TH))
    D, okr = _rays_of(model)
    Dt = torch.tensor(D.reshape(-1, 3), dtype=torch.float32, device=dev)
    proj = model.projector()
    img = alpha = dep = idx = None
    cut = 0
    with torch.set_grad_enabled(grad):
        s = torch.exp(ts.log_s)
        op = torch.sigmoid(ts.op_logit)
        col = torch.sigmoid(ts.col_logit)
        for b in range(nb):
            r0, r1 = b * TH // nb, (b + 1) * TH // nb
            yc = 0.5 * (r0 + r1) * SR.TILE
            tb = t + (min(yc, h) / h - 0.5) * readout_s if nb > 1 else t
            Rw, pw = traj.pose(tb)
            o = Rw @ np.asarray(t_cv, float) + pw
            W2O = (Rw @ np.asarray(R_cv, float)).T
            Wt = torch.tensor(W2O, dtype=torch.float32, device=dev)
            P, Rg, shown = ts.posed(tb)
            Pc = (P - torch.tensor(o, dtype=torch.float32, device=dev)) @ Wt.T
            cov = SR.covariance(Wt[None] @ Rg, s)
            feats = col
            nrm = None
            if depth:
                feats = torch.cat([col, torch.linalg.norm(Pc, dim=1, keepdim=True).detach()], 1)
                nrm = _normals(Wt[None] @ Rg, s, Pc).detach()
            bg = ts.sky_at(Dt @ Wt)
            if depth:
                bg = torch.cat([bg, torch.zeros_like(bg[:, :1])], 1)
            out = SR.splat(Pc, cov, feats, torch.where(shown, op, torch.zeros_like(op)), proj, w, h, bg=bg,
                           rows=(r0, r1), want_index=want_index, normals=nrm, depth_col=3 if depth else None)
            cut += out['cut']
            m = out['rendered'][:, None]
            img = out['image'] if img is None else torch.where(m, out['image'], img)
            alpha = out['alpha'] if alpha is None else torch.where(m[:, 0], out['alpha'], alpha)
            if want_index:
                idx = out['index'] if idx is None else torch.where(m[:, 0], out['index'], idx)
    res = {'rgb': img[:, :3].reshape(h, w, 3), 'alpha': alpha.reshape(h, w), 'cut': cut, 'field': okr}
    if depth:
        res['depth'] = (img[:, 3] / alpha.clamp(min=1e-6)).reshape(h, w)
    if want_index:
        res['index'] = idx.reshape(h, w)
    return res


# =========================================================================== optimisation
def optimise(scene: Scene, drive: Drive, st: SplatSettings, project=None, engine=None, rng=None, progress=None,
             cancel=None) -> dict:
    """Photometric optimisation of the scene against the training frames (module docstring, step 4)."""
    import torch
    stop = cancel or (lambda: False)
    rng = rng if rng is not None else np.random.default_rng(st.seed)
    torch.manual_seed(int(st.seed))
    dev = SR.device_for(st.device, st.gpu_mem_gb)
    ego_fn = _ego_mask_fn(drive, project, engine)
    frames = _frames_for(drive, st, True, 1)
    held = _frames_for(drive, st, False, 1)
    data = []
    for j, (cam, k, fr) in enumerate(frames):
        if stop():
            raise Cancelled()
        if j % 20 == 0:
            _prog(progress, 0.15 * j / max(len(frames), 1), 'Loading the training frames')
        w, h = _train_size(cam, st.train_scale)
        try:
            img = _load_rgb(fr[1], (w, h))
        except OSError:
            continue
        m = cam.model.scaled(w, h)
        valid = _rays_of(m)[1].copy()
        ego = ego_fn(cam, fr, (h, w))
        if ego is not None:
            valid &= ~ego
        data.append((cam, m, fr[0], torch.tensor((img * 255).astype(np.uint8), device=dev),
                     torch.tensor(valid, device=dev)))
    if not data:
        return {'iterations': 0, 'note': 'no training frames'}
    ts = TorchScene(scene, drive, dev, grad=True, positions=st.optimise_positions)
    psnr_before = _image_scores(ts, drive, st, held, ego_fn, limit=st.val_frames)
    groups = [{'params': [ts.log_s], 'lr': 5e-3}, {'params': [ts.quat], 'lr': 1e-3},
              {'params': [ts.op_logit], 'lr': 0.05}, {'params': [ts.col_logit], 'lr': 0.02},
              {'params': [ts.sky_logit], 'lr': 0.02}]
    if st.optimise_positions:
        groups.append({'params': [ts.means], 'lr': 2e-3})
    opt = torch.optim.Adam(groups, eps=1e-15)
    # the lidar placed the Gaussians: the photometric steps may not move one more than MAX_SHIFT_M from its point or
    # grow it past MAX_GROWTH times its first size (without this they stretch into the sky and into fog in front of
    # surfaces, which the images do not forbid but the lidar beams would hit)
    with torch.no_grad():
        means0 = ts.means.detach().clone()
        log_s_max = ts.log_s.detach().clone() + math.log(MAX_GROWTH)
    t0 = time.perf_counter()
    losses = []
    it = 0
    cut = 0
    order = rng.permutation(len(data))
    while it < st.iterations:
        if stop():
            raise Cancelled()
        el = time.perf_counter() - t0
        if el > st.time_budget_s:
            break
        if st.optimise_positions:
            groups[-1]['lr'] = 2e-3 * (0.01 ** (it / max(st.iterations, 1)))
        cam, m, t, img, valid = data[order[it % len(data)]]
        if it and it % len(data) == 0:
            order = rng.permutation(len(data))
        out = render_view(ts, drive.traj, m, cam.R, cam.t, t, cam.readout_s, grad=True)
        cut += out['cut']
        tgt = img.float() / 255.0
        pred = out['rgb']
        msk = valid.float()[..., None]
        n = msk.sum().clamp(min=1.0) * 3
        l1 = ((pred - tgt).abs() * msk).sum() / n
        ss = (SR.ssim_map(pred, tgt) * msk[..., 0]).sum() / msk.sum().clamp(min=1.0)
        loss = 0.8 * l1 + 0.2 * (1 - ss)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            ts.log_s.copy_(torch.minimum(ts.log_s, log_s_max))
            if st.optimise_positions:
                dm = ts.means - means0
                n_ = torch.linalg.norm(dm, dim=1, keepdim=True)
                ts.means.copy_(means0 + dm * torch.clamp(MAX_SHIFT_M / n_.clamp(min=1e-9), max=1.0))
        if it % 10 == 0:
            losses.append(round(float(loss.detach()), 5))
            _prog(progress, 0.2 + 0.7 * min(it / max(st.iterations, 1), el / st.time_budget_s),
                  f'Optimising · step {it} · loss {float(loss.detach()):.4f}')
        it += 1
    secs = time.perf_counter() - t0
    ts.write_back()
    keep = scene.opacity >= 0.01
    pruned = int((~keep).sum())
    if pruned:
        for k in ('means', 'scales', 'quats', 'opacity', 'colours', 'actor', 'obj'):
            setattr(scene, k, getattr(scene, k)[keep])
    ts2 = TorchScene(scene, drive, dev)
    psnr_after = _image_scores(ts2, drive, st, held, ego_fn, limit=st.val_frames)
    if dev.type == 'cuda':
        torch.cuda.empty_cache()
    return {'iterations': it, 'seconds': round(secs, 2), 'device': str(dev), 'frames': len(data),
            'loss_first': losses[0] if losses else None, 'loss_last': losses[-1] if losses else None,
            'loss_curve': losses, 'pruned': pruned, 'tile_entries_cut': int(cut),
            'held_out_psnr_before_db': psnr_before.get('psnr'), 'held_out_psnr_after_db': psnr_after.get('psnr'),
            'held_out_ssim_before': psnr_before.get('ssim'), 'held_out_ssim_after': psnr_after.get('ssim'),
            'stopped_by': 'iterations' if it >= st.iterations else 'time budget'}


def _image_scores(ts: TorchScene, drive: Drive, st: SplatSettings, frames, ego_fn, limit=None) -> dict:
    """Mean PSNR (dB) and SSIM of renders of frames against the recorded images, at train_scale, over the pixels in
    the field and off the recording vehicle; per camera too."""
    import torch
    if limit and len(frames) > limit:
        frames = [frames[int(i)] for i in np.linspace(0, len(frames) - 1, limit).round().astype(int)]
    per = {}
    with torch.no_grad():
        for cam, k, fr in frames:
            w, h = _train_size(cam, st.train_scale)
            try:
                img = _load_rgb(fr[1], (w, h))
            except OSError:
                continue
            m = cam.model.scaled(w, h)
            valid = _rays_of(m)[1].copy()
            ego = ego_fn(cam, fr, (h, w)) if ego_fn is not None else None
            if ego is not None:
                valid &= ~ego
            out = render_view(ts, drive.traj, m, cam.R, cam.t, fr[0], cam.readout_s)
            pred = out['rgb'].clamp(0, 1)
            tgt = torch.tensor(img, device=ts.dev)
            ss = SR.ssim_map(pred, tgt).cpu().numpy()
            p = SR.psnr(pred.cpu().numpy(), img, valid)
            if p is None:
                continue
            per.setdefault(cam.tag, []).append((p, float(ss[valid].mean())))
    if not per:
        return {}
    allp = [x for v in per.values() for x in v]
    return {'psnr': round(float(np.mean([x[0] for x in allp])), 3),
            'ssim': round(float(np.mean([x[1] for x in allp])), 4),
            'frames': len(allp),
            'cameras': {k: {'psnr': round(float(np.mean([x[0] for x in v])), 3),
                            'ssim': round(float(np.mean([x[1] for x in v])), 4), 'frames': len(v)}
                        for k, v in sorted(per.items())}}


# =========================================================================== rigs
@dataclass
class CamSpec:
    tag: str
    name: str
    model: CamModel
    R: np.ndarray                       # optical -> vehicle
    t: np.ndarray
    readout_s: float = 0.0
    body: np.ndarray | None = None      # (h, w) the own vehicle, rig parts and housings
    shade: np.ndarray | None = None     # (h, w) uint8, the engine's shading there
    slot: int = 0


@dataclass
class LidarSpec:
    tag: str
    name: str
    R: np.ndarray                       # sensor -> vehicle
    t: np.ndarray
    elevations: np.ndarray              # per ring, degrees, ring 0 the lowest
    az: np.ndarray                      # per column, degrees, sensor frame, in firing order
    frac: np.ndarray                    # per column, its firing time as a share of the period
    period_s: float = 0.1
    spinning: bool = True
    range_min: float = 0.5
    range_max: float = 120.0
    body: np.ndarray | None = None      # (rings, columns): beams that meet the own vehicle
    model: str = ''


@dataclass
class RigSpec:
    key: str                            # 'recorded' or the variant's uid
    name: str
    cameras: list
    lidars: list
    notes: list = field(default_factory=list)

    def info(self) -> dict:
        return {'key': self.key, 'name': self.name,
                'cameras': [{'tag': c.tag, 'size': [c.model.width, c.model.height], 'model': c.model.to_dict(),
                             'position_m': [round(float(x), 4) for x in c.t], 'readout_ms': round(c.readout_s * 1e3, 3),
                             'body_share': round(float(c.body.mean()), 4) if c.body is not None else None}
                            for c in self.cameras],
                'lidars': [{'tag': L.tag, 'model': L.model, 'rings': int(len(L.elevations)), 'columns': int(len(L.az)),
                            'position_m': [round(float(x), 4) for x in L.t],
                            'self_hit_share': round(float(L.body.mean()), 4) if L.body is not None else None}
                           for L in self.lidars],
                'notes': list(self.notes)}


def _body(project, engine, var, R_ov, t_v, model: CamModel):
    """(mask (h, w), shade (h, w) uint8) of what a camera at t_v (vehicle frame, rotation optical -> vehicle R_ov)
    sees of the vehicle with var's rig and housings, through the engine's cube map; (None, None) without a body."""
    if project is None or engine is None:
        return None, None
    try:
        vg = engine.vehicle_geom(project, var)
    except Exception:                                    # noqa: BLE001 — a project without a usable mesh
        vg = None
    if vg is None:
        return None, None
    cube = engine.cube(np.asarray(t_v, float))
    D, ok = _rays_of(model)
    depth, shade = cube.lookup(D.reshape(-1, 3) @ np.asarray(R_ov, float).T)
    mask = (np.isfinite(depth) & (depth > 0.05)).reshape(model.height, model.width) & ok
    return mask, np.asarray(shade).reshape(model.height, model.width).astype(np.uint8)


def _lidar_body(project, engine, var, spec: LidarSpec):
    if project is None or engine is None:
        return None
    try:
        vg = engine.vehicle_geom(project, var)
    except Exception:                                    # noqa: BLE001
        vg = None
    if vg is None:
        return None
    from . import sensors as SN
    dirs = SN.directions(spec.elevations, spec.az) @ np.asarray(spec.R, float).T
    depth, _ = engine.cube(np.asarray(spec.t, float)).lookup(dirs.reshape(-1, 3))
    return (np.isfinite(depth) & (depth > 0.05)).reshape(len(spec.elevations), len(spec.az))


def _readout(project, cam, st: SplatSettings) -> float:
    if st.rolling_shutter == 'off':
        return 0.0
    if project.module_for(cam).shutter == 'global':
        return 0.0
    from . import timing as TM
    st_t = TM.TimingSettings.from_dict(getattr(project.analysis, 'timing', None))
    return max(float(st_t.readout_ms), 0.0) / 1000.0


def variant_rig(project, variant, st: SplatSettings, engine=None) -> RigSpec:
    """A variant's visible cameras (at render_scale of each module's sensor) and lidars, with its body masks."""
    from . import sensors as SN
    cams, lids = [], []
    for c in variant.cameras:
        if not c.visible:
            continue
        intr = resolve(project.module_for(c), project.lens_for(c))
        w = max(16, int(round(intr.width * st.render_scale)))
        h = max(16, int(round(intr.height * st.render_scale)))
        m = CamModel.from_intrinsics(intr, w, h)
        R = rot_zyx(c.yaw, c.pitch, c.roll) @ BODY_TO_OPTICAL.T
        body, shade = _body(project, engine, variant, R, c.pos(), m)
        cams.append(CamSpec(c.tag, c.name, m, R, c.pos(), _readout(project, c, st), body, shade, int(c.slot)))
    for s in variant.sensors:
        if s.kind != 'lidar' or not s.visible:
            continue
        md = project.sensor_model_for(s)
        el = md.elevations()
        if md.spinning:
            C = md.columns()
            sense = -1.0 if md.rotation == 'cw' else 1.0
            k = np.arange(C)
            az = (s.phase_deg + sense * k * 360.0 / C + 180.0) % 360.0 - 180.0
            frac = k / C
        else:
            az = SN.scan_azimuths(md)
            frac = np.full(len(az), 0.5)
        spec = LidarSpec(s.tag, s.name, SN.rotation(s), s.pos(), np.asarray(el, float), np.asarray(az, float),
                         np.asarray(frac, float), 1.0 / max(md.rate_hz, 0.1), md.spinning, float(md.range_min_m),
                         float(max(md.range_m, md.range_max_m)), None, md.name)
        spec.body = _lidar_body(project, engine, variant, spec)
        lids.append(spec)
    return RigSpec(variant.uid, variant.name, cams, lids)


def recorded_rig(drive: Drive, st: SplatSettings, project=None, engine=None) -> RigSpec:
    """The drive's own rig from its calibration: cameras at render_scale / the drive's image_scale of the recorded
    size, lidars as measured; the body of the project's vehicle with the drive's variant (the active one when the
    project has no variant of that name)."""
    notes = []
    var = _variant_named(project, drive.variant)
    if project is not None and var is None:
        var = project.variant
        if drive.variant:
            notes.append(f'The project has no variant named "{drive.variant}": the recorded rig is drawn with the '
                         f'vehicle and housings of "{var.name}".')
    f = min(st.render_scale / max(drive.image_scale, 1e-6), 2.0)
    cams, lids = [], []
    slots = {c.tag: c.slot for c in (var.cameras if var is not None else [])}
    by_tag = {c.tag: c for c in (var.cameras if var is not None else [])}
    from_module = []
    for cam in drive.cameras.values():
        w = max(16, int(round(cam.model.width * f)))
        h = max(16, int(round(cam.model.height * f)))
        m = cam.model.scaled(w, h)
        body, shade = _body(project, engine, var, cam.R, cam.t, m) if var is not None else (None, None)
        ro = recorded_readout(cam, by_tag.get(cam.tag), project, st)
        if st.rolling_shutter == 'module' and cam.readout_s <= 0 and ro > 0:
            from_module.append(cam.tag)
        cams.append(CamSpec(cam.tag, cam.channel, m, cam.R, cam.t, ro, body, shade, int(slots.get(cam.tag, 0))))
    if from_module:
        notes.append(f'The drive does not state a rolling shutter for its camera frames (a simulator renders each '
                     f'frame at one instant): the recorded rig\'s cameras ({", ".join(from_module)}) are re-rendered '
                     f'with their modules\' rolling shutter, as a variant\'s are, so identical cameras compare alike.')
    for L in drive.lidars.values():
        spec = recorded_lidar_spec(L)
        if var is not None:
            spec.body = _lidar_body(project, engine, var, spec)
        lids.append(spec)
    return RigSpec(RECORDED, 'Recorded rig', cams, lids, notes)


def recorded_readout(cam: DriveCamera, vcam, project, st: SplatSettings) -> float:
    """The rolling-shutter readout (s) the recorded rig's camera is re-rendered with in the A/B: 0 with
    rolling_shutter 'off'; the recording's own readout when the drive states one (cam.readout_s > 0: a real log or a
    sample dataset written with a rolling shutter); otherwise (frames stamped as single instants, as CARLA renders
    them, or a log that does not say) the readout a variant's camera gets (_readout: core.timing's readout_ms unless
    the module is global shutter) for the same-tagged camera of the drive's variant, 0 without one. Both rigs of an A/B
    are thus rendered alike, and the same cameras give the same images."""
    if st.rolling_shutter != 'module':
        return 0.0
    if cam.readout_s > 0:
        return float(cam.readout_s)
    if vcam is None or project is None:
        return 0.0
    return _readout(project, vcam, st)


def recorded_lidar_spec(L: DriveLidar) -> LidarSpec:
    el = L.elevations if L.elevations is not None else np.linspace(-15.0, 15.0, 32)
    C = max(int(L.columns or 1024), 8)
    a0 = 0.0
    try:
        a = load_sweep(L.sweeps[0])
        if len(a):
            a0 = float(np.median(np.degrees(np.arctan2(a[:32, 1], a[:32, 0]))))
            step = 360.0 / C
            a0 = round(a0 / step) * step
    except (OSError, ValueError, IndexError):
        pass
    k = np.arange(C)
    az = (a0 + L.sense * k * 360.0 / C + 180.0) % 360.0 - 180.0
    return LidarSpec(L.tag, L.channel, L.R, L.t, np.asarray(el, float), az, k / C, L.period_s, L.spinning, 0.5, 200.0,
                     None, L.model or 'recorded')


# =========================================================================== lidar rendering
def render_lidar(ts: TorchScene, traj: Trajectory, spec: LidarSpec, t_start: float, st: SplatSettings) -> dict:
    """One sweep of a lidar starting at drive time t_start (module docstring: Rendering through a rig). Returns
    {'range' (rings, columns) m, NaN where no return, 'acc', 'cos', 'points' (n, 5) x y z (sensor frame at the firing
    pose) intensity (0: not modelled) ring, 'col' (n,) column of each point, 'time' (columns,) firing times,
    'returns', 'beams', 'self_hits', 'dropped_opacity', 'dropped_incidence'}."""
    import torch
    dev = ts.dev
    el = np.asarray(spec.elevations, float)
    R, C = len(el), len(spec.az)
    e_step = float(min(np.min(np.diff(np.sort(el))) if R > 1 else 0.5, 0.5))
    e_step = max(e_step, 0.05)
    e_hi = float(el.max()) + 2.5 * e_step              # the top beam on a pixel centre, two rows below the edge
    height = int(math.ceil((e_hi - (el.min() - 2 * e_step)) / e_step))
    rows = np.clip(np.floor((e_hi - el) / e_step).astype(int), 0, height - 1)
    # rings off the even grid are read from the row they fall in (at most half a row, ≤ 0.25°, from its centre)
    step = 360.0 / C if spec.spinning else (float(np.median(np.abs(np.diff(spec.az)))) if C > 1 else 1.0)
    nsec = SECTORS if (spec.spinning and C >= SECTORS * 4) else 1
    rng_g = np.full((R, C), np.nan)
    acc_g = np.zeros((R, C))
    cos_g = np.zeros((R, C))
    s = torch.exp(ts.log_s)
    op = torch.sigmoid(ts.op_logit)
    smin = s.min(1).values
    jmin = s.argmin(1)
    iso = smin > 0.5 * s.median(1).values
    with torch.no_grad():
        for k in range(nsec):
            cols = np.arange(k * C // nsec, (k + 1) * C // nsec)
            if not len(cols):
                continue
            tm = t_start + float(np.mean(spec.frac[cols])) * spec.period_s
            Rw, pw = traj.pose(tm)
            o = Rw @ np.asarray(spec.t, float) + pw
            W2S = (Rw @ np.asarray(spec.R, float)).T
            Wt = torch.tensor(W2S, dtype=torch.float32, device=dev)
            P, Rg, shown = ts.posed(tm)
            Ps = (P - torch.tensor(o, dtype=torch.float32, device=dev)) @ Wt.T
            Rs = Wt[None] @ Rg
            cov = SR.covariance(Rs, s)
            rr = torch.linalg.norm(Ps, dim=1)
            nrm = torch.gather(Rs, 2, jmin[:, None, None].expand(-1, 3, 1))[..., 0]
            cs = torch.where(iso, torch.ones_like(rr), (nrm * Ps).sum(1).abs() / rr.clamp(min=1e-6))
            nrm = torch.where(iso[:, None], Ps / rr[:, None].clamp(min=1e-6), nrm)
            azc = np.degrees(np.arctan2(np.mean(np.sin(np.radians(spec.az[cols]))),
                                        np.mean(np.cos(np.radians(spec.az[cols])))))
            rel = (spec.az[cols] - azc + 180.0) % 360.0 - 180.0
            m = 2
            a_lo = float(rel.min()) - (m + 0.5) * step
            width = int(math.ceil((rel.max() - rel.min()) / step)) + 2 * m + 2
            proj = SR.SphereProjector(azc, a_lo, step, e_hi, e_step, width, height)
            out = SR.splat(Ps, cov, torch.stack([rr, cs], 1), torch.where(shown, op, torch.zeros_like(op)), proj,
                           width, height, normals=nrm, depth_col=0, lowpass=LIDAR_LOWPASS)
            ci = np.clip(np.floor((rel - a_lo) / step).astype(int), 0, width - 1)
            img = out['image'].reshape(height, width, 2).cpu().numpy()
            if st.lidar_depth == 'median':
                img[..., 0] = out['median'].reshape(height, width).cpu().numpy() * \
                    out['alpha'].reshape(height, width).cpu().numpy()
            al = out['alpha'].reshape(height, width).cpu().numpy()
            a_ = al[rows[:, None], ci[None, :]]
            acc_g[:, cols] = a_
            with np.errstate(invalid='ignore', divide='ignore'):
                rng_g[:, cols] = img[rows[:, None], ci[None, :], 0] / np.maximum(a_, 1e-6)
                cos_g[:, cols] = img[rows[:, None], ci[None, :], 1] / np.maximum(a_, 1e-6)
    ok_op = acc_g >= st.lidar_min_opacity
    ok_inc = cos_g >= math.cos(math.radians(st.lidar_max_incidence_deg))
    ok_rng = (rng_g >= spec.range_min) & (rng_g <= spec.range_max)
    body = spec.body if spec.body is not None else np.zeros((R, C), bool)
    ret = ok_op & ok_inc & ok_rng & ~body
    out_rng = np.where(ret, rng_g, np.nan)
    from . import sensors as SN
    dirs = SN.directions(el, spec.az)
    rr_, cc_ = np.nonzero(ret)
    P = dirs[rr_, cc_] * out_rng[rr_, cc_][:, None]
    pts = np.zeros((len(P), 5), np.float32)
    pts[:, :3] = P
    pts[:, 4] = rr_
    return {'range': out_rng, 'acc': acc_g, 'cos': cos_g, 'points': pts, 'col': cc_, 'ring': rr_,
            'time': t_start + spec.frac * spec.period_s, 'returns': int(ret.sum()), 'beams': int(R * C),
            'self_hits': int(body.sum()), 'dropped_opacity': int((~ok_op & ~body).sum()),
            'dropped_incidence': int((ok_op & ~ok_inc & ~body).sum())}


def lidar_points_vehicle(spec: LidarSpec, sweep: dict, traj: Trajectory, t_ref: float) -> np.ndarray:
    """A rendered sweep's points in the vehicle frame at t_ref (each from its firing pose through the world)."""
    P = sweep['points'][:, :3].astype(float)
    if not len(P):
        return np.zeros((0, 3))
    tf = sweep['time'][sweep['col']]
    Rw, pw = traj.at(tf)
    Pv = P @ np.asarray(spec.R, float).T + spec.t
    Pw = np.einsum('nij,nj->ni', Rw, Pv) + pw
    R0, p0 = traj.pose(t_ref)
    return (Pw - p0) @ R0


# =========================================================================== validation
def validate(ts: TorchScene, drive: Drive, st: SplatSettings, project=None, engine=None, progress=None,
             cancel=None) -> dict:
    """The reconstruction against the held-out recorded frames and sweeps (module docstring: Validation)."""
    stop = cancel or (lambda: False)
    out = {'cameras': {}, 'lidar': {}}
    held = _frames_for(drive, st, False, 1)
    if not held:
        out['note'] = 'No frame is held out (holdout_every 0): the reconstruction is not validated.'
        return out
    ego_fn = _ego_mask_fn(drive, project, engine)
    _prog(progress, 0.0, 'Validation · held-out frames')
    sc = _image_scores(ts, drive, st, held, ego_fn, limit=st.val_frames)
    out['cameras'] = sc
    if stop():
        raise Cancelled()
    errs, coss, both, rec_only, ren_only, n_sw = [], [], 0, 0, 0, 0
    for L in drive.lidars.values():
        ever = None
        pairs = []
        spec = recorded_lidar_spec(L)
        ho = [sw for k, sw in enumerate(L.sweeps) if held_out(k, st.holdout_every)]
        ho = [ho[int(i)] for i in np.unique(np.linspace(0, len(ho) - 1, min(st.val_sweeps, len(ho))).round())] \
            if ho else []
        for j, sw in enumerate(ho):
            if stop():
                raise Cancelled()
            _prog(progress, 0.6 + 0.4 * j / max(len(ho), 1), f'Validation · lidar {L.tag} sweep {j + 1}')
            a = load_sweep(sw)
            if not len(a) or (a[:, 3] < 0).all():
                continue
            ren = render_lidar(ts, drive.traj, spec, sw[0] - L.period_s, st)
            ring = a[:, 3].astype(int)
            ok = (ring >= 0) & (ring < len(spec.elevations))
            az = np.degrees(np.arctan2(a[:, 1], a[:, 0]))
            C = len(spec.az)
            col = np.round(((az - spec.az[0]) * (1 if L.sense > 0 else -1)) % 360.0 / (360.0 / C)).astype(int) % C
            rr = np.linalg.norm(a[:, :3], axis=1)
            rec = np.zeros((len(spec.elevations), C), bool)
            rec[ring[ok], col[ok]] = True
            r_ren = ren['range'][ring[ok], col[ok]]
            m = np.isfinite(r_ren)
            errs.append(r_ren[m] - rr[ok][m])
            coss.append(ren['cos'][ring[ok], col[ok]][m])
            pairs.append((rec, np.isfinite(ren['range'])))
            ever = rec.copy() if ever is None else (ever | rec)
            n_sw += 1
        for rec, got in pairs:
            # beams that returned in none of the compared sweeps (the recording vehicle's own body, its points
            # dropped as self hits, and open sky) are left out of the return counts
            both += int((rec & got).sum())
            rec_only += int((rec & ~got).sum())
            ren_only += int((~rec & got & ever).sum())
    if errs:
        e = np.concatenate(errs)
        ae = np.abs(e)
        steep = np.concatenate(coss) >= math.cos(math.radians(STEEP_DEG))   # beams meeting the surface within 70°
        out['lidar'] = {'sweeps': n_sw, 'compared_points': int(len(e)),
                        'median_abs_error_m': round(float(np.median(ae)), 4),
                        'mean_error_m': round(float(np.mean(e)), 4),
                        'p90_abs_error_m': round(float(np.percentile(ae, 90)), 4),
                        'within_10cm': round(float(np.mean(ae <= 0.10)), 4),
                        'within_50cm': round(float(np.mean(ae <= 0.50)), 4),
                        'median_abs_error_steep_m': (round(float(np.median(ae[steep])), 4) if steep.any() else None),
                        'steep_share': round(float(np.mean(steep)), 4),
                        'returns_both': both, 'recorded_only': rec_only, 'rendered_only': ren_only,
                        'return_agreement': round(both / max(both + rec_only + ren_only, 1), 4)}
    return out


# =========================================================================== the A/B comparison
def _objects_in_view(drive: Drive, t: float, cam: CamSpec, inst: np.ndarray, Rw, pw, rays) -> list:
    """The annotated objects a rendered camera image shows, as ground-truth objects (core.groundtruth's keys)."""
    from . import groundtruth as GT
    h, w = inst.shape
    out = []
    W2O = np.asarray(cam.R, float).T
    D, okr = rays
    for i, tr in drive.tracks.items():
        if tr.cls not in DT.GT_TO_CLASS:
            continue
        po = tr.pose(t)
        if po is None:
            continue
        Rb, cb = po
        cv = Rw.T @ (cb - pw)
        Rbv = Rw.T @ Rb
        if np.linalg.norm(cv - cam.t) > 250:
            continue
        E = GT.box_edges(cv, tr.size, Rbv, n=9)
        u, v, _, ok = cam.model.project((E - cam.t) @ W2O.T)
        if not ok.any():
            continue
        x0, x1 = float(u[ok].min()), float(u[ok].max())
        y0, y1 = float(v[ok].min()), float(v[ok].max())
        if x1 < 0 or y1 < 0 or x0 > w or y0 > h:
            continue
        m = inst == i
        pix = int(m.sum())
        if pix == 0:
            continue
        ys, xs = np.nonzero(m)
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
        F = GT.box_faces(cv, tr.size, Rbv, n=7)
        fu, fv, _, fok = cam.model.project((F - cam.t) @ W2O.T)
        trunc = 1.0 - float(np.mean(fok & (fu >= 0) & (fu < w) & (fv >= 0) & (fv < h)))
        xa, xb = int(max(math.floor(x0), 0)), int(min(math.ceil(x1), w))
        ya, yb = int(max(math.floor(y0), 0)), int(min(math.ceil(y1), h))
        n_box = pix
        if xb > xa and yb > ya:
            dirs = D[ya:yb, xa:xb].reshape(-1, 3) @ np.asarray(cam.R, float).T
            tin, _ = GT.ray_box(np.asarray(cam.t, float), dirs, cv, tr.size, Rbv)
            n_box = max(int((np.isfinite(tin) & okr[ya:yb, xa:xb].ravel()).sum()), pix)
        yaw = math.degrees(math.atan2(Rbv[1, 0], Rbv[0, 0]))
        out.append({'id': int(i), 'actor_id': int(i) if tr.dynamic else None, 'instance_token': f'track:{i}',
                    'category': tr.category, 'class': tr.cls, 'bbox': bbox, 'pixels': pix,
                    'truncated': round(trunc, 4), 'occluded': round(max(0.0, 1.0 - pix / max(n_box, 1)), 4),
                    'range_m': round(float(np.linalg.norm(cv - cam.t)), 3),
                    'box3d': {'center': [round(float(x), 4) for x in cv], 'size': list(tr.size),
                              'yaw_deg': round(yaw, 3)}})
    return out


def _render_rig_frame(ts: TorchScene, drive: Drive, rig: RigSpec, t: float, scene: Scene):
    """Every camera of a rig at t: {tag: (rgb uint8 (h, w, 3), object ids (h, w) int32, range (h, w) float32)}."""
    import torch
    out = {}
    obj = torch.tensor(scene.obj, dtype=torch.long, device=ts.dev)
    with torch.no_grad():
        for c in rig.cameras:
            r = render_view(ts, drive.traj, c.model, c.R, c.t, t, c.readout_s, want_index=True, depth=True)
            rgb = (r['rgb'].clamp(0, 1) * 255 + 0.5).to(torch.uint8).cpu().numpy()
            idx = r['index']
            ids = torch.where((idx >= 0) & (r['alpha'] > 0.5), obj[idx.clamp(min=0)], torch.zeros_like(idx))
            ids = ids.cpu().numpy().astype(np.int32)
            rng = r['depth'].cpu().numpy().astype(np.float32)
            rng[r['alpha'].cpu().numpy() < 0.5] = 0.0
            fld = r['field']
            rgb[~fld] = 0
            ids[~fld] = 0
            rng[~fld] = 0
            if c.body is not None:
                g = c.shade[c.body] if c.shade is not None else np.full(int(c.body.sum()), 96, np.uint8)
                rgb[c.body] = np.repeat(np.clip(g, 30, 220)[:, None], 3, 1)
                ids[c.body] = 0
                rng[c.body] = 0
            out[c.tag] = (rgb, ids, rng)
    return out


def _thumb(img, width=320):
    import cv2
    h, w = img.shape[:2]
    if w <= width:
        return img.copy()
    return cv2.resize(img, (width, max(1, int(round(h * width / w)))), interpolation=cv2.INTER_AREA)


def _boot_ci(per_instant_a, per_instant_b, n_inst, B, seed, conf=0.95):
    """Percentile interval of the recall difference B − A from a bootstrap over instants (paired: the same instants
    drawn for both rigs). per_instant_*: (detected, objects) arrays per instant."""
    if n_inst < 2 or B <= 0:
        return None
    rng = np.random.default_rng(seed)
    da, na = per_instant_a
    db, nb = per_instant_b
    W = rng.multinomial(n_inst, np.full(n_inst, 1.0 / n_inst), size=B).astype(float)
    with np.errstate(invalid='ignore', divide='ignore'):
        ra = (W @ da) / (W @ na)
        rb = (W @ db) / (W @ nb)
    d = (rb - ra)[np.isfinite(rb - ra)]
    if len(d) < B // 2:
        return None
    lo, hi = np.percentile(d, [(1 - conf) / 2 * 100, (1 + conf) / 2 * 100])
    return [round(float(lo), 4), round(float(hi), 4)]


def _coverage(drive: Drive, rig_a: RigSpec, instants, rows_a: dict, de_st, classes, edges) -> list:
    """Per class and range bin: the objects the recording's own ground truth shows unignored at the instants
    ('recorded': each camera's frame nearest the instant, within ACTOR_PAD_S; an object counts once per instant), those
    the re-rendered recorded rig shows ('rendered'), and both. [] unless rig A is the recorded rig and the drive has
    per-frame ground truth."""
    if rig_a.key != RECORDED or drive.gt is None:
        return []
    rec = {}
    for k, t in enumerate(instants):
        for c in rig_a.cameras:
            dc = drive.cameras.get(c.tag)
            if dc is None or not dc.frames:
                continue
            j = int(np.argmin([abs(fr[0] - t) for fr in dc.frames]))
            tf, _, gfr = dc.frames[j]
            if gfr is None or abs(tf - t) > ACTOR_PAD_S:
                continue
            g = DE.gt_objects(gfr, de_st)
            for i in range(len(g['id'])):
                if g['ignore'][i] == DE.IGN_NONE and g['cls'][i] >= 0:
                    rec.setdefault((k, int(g['id'][i])), (int(g['cls'][i]), float(g['dist'][i])))
    if not rec:
        return []
    out = []
    labels = DE.bin_labels(edges)
    for ec in classes:
        cidx = [DT.CLASSES.index(m) for m in DT.GROUPS.get(ec, (ec,))]
        for b in range(len(edges) + 1):
            kr = {key for key, (cl, d) in rec.items() if cl in cidx and DE.bin_of(np.array([d]), edges)[0] == b}
            kq = {key for key, r in rows_a.items() if r['vis'] and r['cls'] in cidx
                  and DE.bin_of(np.array([r['dist']]), edges)[0] == b}
            if kr or kq:
                out.append({'class': ec, 'range': labels[b], 'bin': b, 'recorded': len(kr), 'rendered': len(kq),
                            'both': len(kr & kq)})
    return out


def compare(ts: TorchScene, drive: Drive, scene: Scene, rig_a: RigSpec, rig_b: RigSpec, st: SplatSettings,
            de_st, detector=None, progress=None, cancel=None) -> dict:
    """The A/B comparison (module docstring). detector: a callable BGR image -> detections, or None (visibility
    figures only)."""
    stop = cancel or (lambda: False)
    first = max(drive.cameras.values(), key=lambda c: (len(c.frames), c.tag == 'F'))
    times = [fr[0] for fr in first.frames]
    n = min(st.eval_instants, len(times))
    picks = np.unique(((np.arange(n) + 0.5) / max(n, 1) * len(times)).astype(int)) if n else np.zeros(0, int)
    instants = [float(times[i]) for i in picks]
    classes = [c for c in DT.EVAL_CLASSES if c in de_st.classes or (c in DT.GROUPS and
                                                                     any(m in de_st.classes for m in DT.GROUPS[c]))]
    edges = list(de_st.range_edges)
    rigs = {'a': rig_a, 'b': rig_b}
    rows = {'a': {}, 'b': {}}                  # (instant, obj id) -> {'cls', 'dist', 'vis', 'det'}
    per_cam = {'a': {}, 'b': {}}
    lid = {'a': {}, 'b': {}}
    lid_obj = {'a': {}, 'b': {}}               # (instant, obj id) -> points
    previews = []
    det_ms = []
    total = max(len(instants) * 2, 1)
    import cv2
    for k, t in enumerate(instants):
        Rw, pw = drive.traj.pose(t)
        prev = {'t': round(t, 4), 'a': {}, 'b': {}, 'lidar_a': None, 'lidar_b': None}
        for side, rig in rigs.items():
            if stop():
                raise Cancelled()
            _prog(progress, (2 * k + (side == 'b')) / total,
                  f'A/B · instant {k + 1}/{len(instants)} · rig {side.upper()}')
            renders = _render_rig_frame(ts, drive, rig, t, scene)
            for c in rig.cameras:
                rgb, ids, rng_img = renders[c.tag]
                prev[side][c.tag] = _thumb(rgb)
                objs = _objects_in_view(drive, t, c, ids, Rw, pw, _rays_of(c.model))
                frame = {'sample_data_token': f'{side}:{c.tag}:{k}', 'objects': objs}
                g = DE.gt_objects(frame, de_st)
                pc = per_cam[side].setdefault(c.tag, {'objects': 0, 'detected': 0, 'images': 0,
                                                     'body_share': round(float(c.body.mean()), 4)
                                                     if c.body is not None else 0.0})
                pc['images'] += 1
                score = np.full(len(g['id']), -np.inf)
                if detector is not None:
                    t0 = time.perf_counter()
                    dets = detector(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                    det_ms.append((time.perf_counter() - t0) * 1000)
                    ev = DE.evaluate_image(g, dets, rng_img, de_st, classes)
                    for e, i, sc in ev['gt']:
                        score[i] = max(score[i], sc)
                for i in range(len(g['id'])):
                    key = (k, int(g['id'][i]))
                    r = rows[side].setdefault(key, {'cls': int(g['cls'][i]), 'dist': float(g['dist'][i]),
                                                    'vis': False, 'det': False})
                    if g['ignore'][i] == DE.IGN_NONE and g['cls'][i] >= 0:
                        r['vis'] = True
                        pc['objects'] += 1
                        if score[i] >= de_st.score:
                            r['det'] = True
                            pc['detected'] += 1
            # lidars: one sweep ending at t
            pts_all = []
            for L in rig.lidars:
                sw = render_lidar(ts, drive.traj, L, t - L.period_s, st)
                d = lid[side].setdefault(L.tag, {'sweeps': 0, 'returns': 0, 'beams': 0, 'self_hits': 0,
                                                 'dropped_opacity': 0, 'dropped_incidence': 0, 'model': L.model})
                d['sweeps'] += 1
                for kk in ('returns', 'beams', 'self_hits', 'dropped_opacity', 'dropped_incidence'):
                    d[kk] += sw[kk]
                pts_all.append(lidar_points_vehicle(L, sw, drive.traj, t))
            P = np.concatenate(pts_all) if pts_all else np.zeros((0, 3))
            prev['lidar_' + side] = P[:: max(1, len(P) // 60000)].astype(np.float32)
            for i, tr in drive.tracks.items():
                if tr.cls not in DT.GT_TO_CLASS:
                    continue
                po = tr.pose(t)
                if po is None:
                    continue
                Rb, cb = po
                cv_ = Rw.T @ (cb - pw)
                Rbv = Rw.T @ Rb
                if np.hypot(cv_[0], cv_[1]) > 120 or not len(P):
                    continue
                q = (P - cv_) @ Rbv
                lid_obj[side][(k, int(i))] = int(np.all(np.abs(q) <= np.asarray(tr.size) / 2 + 0.05, 1).sum())
        previews.append(prev)
    # ---- recall per class and range: the union of the objects either rig shows
    out_rows = []
    nb = len(edges) + 1
    labels = DE.bin_labels(edges)
    coverage = _coverage(drive, rig_a, instants, rows['a'], de_st, classes, edges)
    for ec in classes:
        members = DT.GROUPS.get(ec, (ec,))
        cidx = [DT.CLASSES.index(m) for m in members]
        for b in range(nb):
            keys = set()
            for side in ('a', 'b'):
                for key, r in rows[side].items():
                    if r['vis'] and r['cls'] in cidx and DE.bin_of(np.array([r['dist']]), edges)[0] == b:
                        keys.add(key)
            if not keys:
                continue
            per = {}
            for side in ('a', 'b'):
                vis = np.array([rows[side].get(key, {}).get('vis', False) for key in sorted(keys)])
                det = np.array([rows[side].get(key, {}).get('det', False) for key in sorted(keys)])
                inst = np.array([key[0] for key in sorted(keys)])
                per[side] = (vis, det, inst)
            nU = len(keys)
            row = {'class': ec, 'range': labels[b], 'bin': b, 'objects': nU,
                   'visible_a': int(per['a'][0].sum()), 'visible_b': int(per['b'][0].sum()),
                   'visible_share_a': round(float(per['a'][0].mean()), 4),
                   'visible_share_b': round(float(per['b'][0].mean()), 4)}
            if detector is not None:
                ra, rb = float(per['a'][1].mean()), float(per['b'][1].mean())
                ni = len(instants)
                da = np.bincount(per['a'][2], weights=per['a'][1].astype(float), minlength=ni)
                db = np.bincount(per['b'][2], weights=per['b'][1].astype(float), minlength=ni)
                cnt = np.bincount(per['a'][2], minlength=ni).astype(float)
                row.update({'recall_a': round(ra, 4), 'recall_b': round(rb, 4), 'delta': round(rb - ra, 4),
                            'delta_ci': _boot_ci((da, cnt), (db, cnt), ni, 500, st.seed + b)})
            pa = [lid_obj['a'].get(key) for key in sorted(keys) if lid_obj['a'].get(key) is not None]
            pb = [lid_obj['b'].get(key) for key in sorted(keys) if lid_obj['b'].get(key) is not None]
            row['lidar_points_a'] = round(float(np.median(pa)), 1) if pa else None
            row['lidar_points_b'] = round(float(np.median(pb)), 1) if pb else None
            out_rows.append(row)
    cams = {side: [{'camera': tag, **v, 'recall': (round(v['detected'] / v['objects'], 4)
                                                     if (detector is not None and v['objects']) else None)}
                   for tag, v in per_cam[side].items()] for side in ('a', 'b')}
    return {'instants': [round(x, 4) for x in instants], 'classes': classes, 'range_edges': edges,
            'recall': out_rows, 'coverage': coverage, 'cameras': cams, 'lidar': lid, 'previews': previews,
            'detector_ms': round(float(np.median(det_ms)), 1) if det_ms else None}


# =========================================================================== .ply (3D Gaussian splatting)
_PLY_TYPES = {'char': 'i1', 'uchar': 'u1', 'short': '<i2', 'ushort': '<u2', 'int': '<i4', 'uint': '<u4',
              'float': '<f4', 'double': '<f8', 'int8': 'i1', 'uint8': 'u1', 'int16': '<i2', 'uint16': '<u2',
              'int32': '<i4', 'uint32': '<u4', 'float32': '<f4', 'float64': '<f8'}


def read_ply(path) -> dict:
    """The vertex element of a .ply file (binary little-endian or ASCII) as {property: array}."""
    with open(path, 'rb') as fh:
        head = b''
        while not head.endswith(b'end_header\n'):
            line = fh.readline()
            if not line:
                raise ValueError(f'{os.path.basename(path)}: no PLY header')
            head += line.replace(b'\r\n', b'\n')
        lines = head.decode('ascii', 'replace').splitlines()
        if not lines or lines[0].strip() != 'ply':
            raise ValueError(f'{os.path.basename(path)} is not a PLY file')
        fmt = next((ln.split()[1] for ln in lines if ln.startswith('format')), '')
        n, props, cur, before = 0, [], None, 0
        for ln in lines:
            p = ln.split()
            if p and p[0] == 'element':
                cur = p[1]
                if cur == 'vertex':
                    n = int(p[2])
                elif n == 0:
                    raise ValueError('elements before the vertices are not supported')
            elif p and p[0] == 'property' and cur == 'vertex':
                if p[1] == 'list':
                    raise ValueError('list properties on the vertices are not supported')
                props.append((p[2], _PLY_TYPES[p[1]]))
        dt = np.dtype(props)
        if fmt == 'binary_little_endian':
            a = np.frombuffer(fh.read(n * dt.itemsize), dt, count=n)
        elif fmt == 'ascii':
            a = np.loadtxt(fh, max_rows=n, ndmin=2)
            a = np.rec.fromarrays([a[:, k].astype(dt[k]) for k in range(len(props))], dtype=dt)
        else:
            raise ValueError(f'PLY format {fmt} is not supported (binary_little_endian or ascii)')
    return {k: np.asarray(a[k]) for k in dt.names}


def ply_scene(path, st: SplatSettings, origin) -> Scene:
    """A standard 3DGS .ply as a Scene in the drive's local frame (alignment from the settings)."""
    v = read_ply(path)
    if not all(k in v for k in ('x', 'y', 'z')):
        raise ValueError(f'{os.path.basename(path)} has no x, y, z.')
    P = np.stack([v['x'], v['y'], v['z']], 1).astype(float)
    n = len(P)
    sR = rot_zyx(*st.ply_ypr)
    k = float(st.ply_scale)
    Pg = k * P @ sR.T + np.asarray(st.ply_xyz, float)
    if all(f'scale_{i}' in v for i in range(3)):
        scales = np.exp(np.stack([v[f'scale_{i}'] for i in range(3)], 1).astype(float)) * k
    else:
        scales = _shapes(P, 0.05)[0] * k
    if all(f'rot_{i}' in v for i in range(4)):
        q = np.stack([v[f'rot_{i}'] for i in range(4)], 1).astype(float)
        Rg = np.einsum('ij,njk->nik', sR, quat_to_rot(q))
        quats = rot_to_quat(Rg)
    else:
        quats = np.tile([1.0, 0, 0, 0], (n, 1))
    opacity = 1 / (1 + np.exp(-v['opacity'].astype(float))) if 'opacity' in v else np.full(n, 0.9)
    if all(f'f_dc_{i}' in v for i in range(3)):
        col = 0.5 + SR.C0 * np.stack([v[f'f_dc_{i}'] for i in range(3)], 1).astype(float)
    elif all(c in v for c in ('red', 'green', 'blue')):
        col = np.stack([v['red'], v['green'], v['blue']], 1).astype(float)
        col = col / (255.0 if col.max() > 1.5 else 1.0)
    else:
        col = np.full((n, 3), 0.5)
    keep = np.all(np.isfinite(Pg), 1) & np.all(np.isfinite(scales), 1) & (opacity > 0.004)
    if keep.sum() > st.max_gaussians:
        order = np.argsort(-opacity[keep], kind='stable')[:st.max_gaussians]
        idx = np.flatnonzero(keep)[np.sort(order)]
    else:
        idx = np.flatnonzero(keep)
    m = len(idx)
    return Scene((Pg[idx] - origin).astype(np.float32), np.clip(scales[idx], 1e-4, 50).astype(np.float32),
                 quats[idx].astype(np.float32), np.clip(opacity[idx], 0, 1).astype(np.float32),
                 np.clip(col[idx], 0, 1).astype(np.float32), np.full(m, -1, np.int32), np.zeros(m, np.int32), [],
                 np.full((SKY_H, SKY_W, 3), 0.6, np.float32), np.asarray(origin, float).copy(), 'ply',
                 {'gaussians_in_file': int(n), 'gaussians': int(m), 'file': os.path.basename(path),
                  'alignment': {'xyz_m': list(st.ply_xyz), 'ypr_deg': list(st.ply_ypr), 'scale': st.ply_scale}})


def ply_bytes(scene: Scene, drive: Drive | None = None, t: float | None = None) -> bytes:
    """The scene as a standard 3DGS .ply (INRIA layout, degree-0 colour, f_rest zero), positions in the drive's
    global frame; actors placed at time t (left out when not shown then, or without a drive)."""
    P = scene.means.astype(float)
    Rq = quat_to_rot(scene.quats)
    keep = scene.actor < 0
    if drive is not None and t is not None:
        for a, i in enumerate(scene.actor_ids):
            m = scene.actor == a
            po = drive.tracks[i].pose(t) if i in drive.tracks else None
            if po is None:
                continue
            Rb, cb = po
            P[m] = P[m] @ Rb.T + cb
            Rq[m] = np.einsum('ij,njk->nik', Rb, Rq[m])
            keep |= m
    P, Rq = P[keep] + scene.origin, Rq[keep]
    n = len(P)
    names = (['x', 'y', 'z', 'nx', 'ny', 'nz', 'f_dc_0', 'f_dc_1', 'f_dc_2'] + [f'f_rest_{i}' for i in range(45)] +
             ['opacity', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3'])
    a = np.zeros((n, len(names)), np.float32)
    a[:, 0:3] = P
    a[:, 6:9] = (scene.colours[keep] - 0.5) / SR.C0
    op = np.clip(scene.opacity[keep].astype(float), 1e-6, 1 - 1e-6)
    a[:, 54] = np.log(op / (1 - op))
    a[:, 55:58] = np.log(np.maximum(scene.scales[keep], 1e-8))
    a[:, 58:62] = rot_to_quat(Rq)
    head = ('ply\nformat binary_little_endian 1.0\ncomment Vantage log replay (vantage.splat/1): 3D Gaussians, '
            'global frame of the drive, metres\nelement vertex %d\n' % n +
            ''.join(f'property float {k}\n' for k in names) + 'end_header\n')
    return head.encode('ascii') + a.tobytes()


# =========================================================================== building or loading the scene
def cache_folder(cache_root, drive: Drive, st: SplatSettings) -> str:
    key = hashlib.sha256(json.dumps({'drive': drive.key, 'build': st.build_key()}, sort_keys=True,
                                    default=str).encode()).hexdigest()[:12]
    name = re.sub(r'[^a-z0-9]+', '-', os.path.basename(os.path.normpath(drive.root)).lower()).strip('-') or 'drive'
    return os.path.join(cache_root, f'{name}-{key}')


def build_scene(drive: Drive, st: SplatSettings, project=None, engine=None, progress=None, cancel=None,
                cache_root=None, rebuild=False) -> Scene:
    """The scene for the drive and settings: from the cache when it was built with the same inputs, else built (or
    imported) and cached."""
    folder = cache_folder(cache_root, drive, st) if cache_root else None
    if folder and not rebuild:
        sc = Scene.load(folder)
        if sc is not None and sc.info.get('drive_key') == drive.key:
            sc.info['cached'] = True
            return sc
    if st.source == 'ply':
        if not st.ply_path or not os.path.isfile(st.ply_path):
            raise ValueError('Choose the .ply file to import.')
        sc = ply_scene(st.ply_path, st, drive.origin)
        _prog(progress, 0.3, 'Sky from the images')
        colour_sky_only(sc, drive, st, project, engine, cancel)
        sc.info['drive_key'] = drive.key
    else:
        sc = reconstruct(drive, st, project, engine, progress, cancel)
    sc.info['cached'] = False
    if folder:
        sc.save(folder)
    return sc


def colour_sky_only(scene: Scene, drive: Drive, st: SplatSettings, project=None, engine=None, cancel=None):
    """The sky texture of an imported scene, from the drive's images (its Gaussians keep their own colours)."""
    keep = scene.colours.copy()
    colour_scene(scene, drive, st, project, engine, cancel=cancel)
    scene.colours = keep


# =========================================================================== the study
@dataclass
class SplatResult:
    settings: SplatSettings
    drive: dict
    scene: dict
    validation: dict
    rig_a: dict
    rig_b: dict
    ab: dict
    detector: dict | None
    findings: list
    notes: list
    timing: dict
    seconds: float = 0.0

    @property
    def previews(self) -> list:
        return self.ab.get('previews') or []


def _finding(out, sev, kind, msg, uids=()):
    out.append({'severity': sev, 'kind': kind, 'message': msg, 'uids': list(uids)})


def _variant_by(project, key):
    if project is None or not key:
        return None
    return project.variant_by_uid(key) or next((v for v in project.variants if v.name == key), None)


def resolve_rigs(project, drive: Drive, st: SplatSettings, engine=None, variant=None) -> tuple:
    """(rig A, rig B): A the recorded rig or a variant (variant_a), B the variant given, else variant_b, else the
    active variant."""
    if st.variant_a == RECORDED or project is None or _variant_by(project, st.variant_a) is None:
        ra = recorded_rig(drive, st, project, engine)
    else:
        ra = variant_rig(project, _variant_by(project, st.variant_a), st, engine)
    vb = variant or _variant_by(project, st.variant_b) or (project.variant if project is not None else None)
    rb = variant_rig(project, vb, st, engine) if vb is not None else recorded_rig(drive, st, project, engine)
    return ra, rb


def run(project, engine=None, settings: SplatSettings | None = None, variant=None, progress=None, cancel=None,
        detector=None, cache_root=None, drive: Drive | None = None, scene: Scene | None = None) -> SplatResult | None:
    """The log replay study (None when cancelled). detector: a callable BGR image -> detections (tests, or one
    already loaded); by default the detection settings' model when it is downloaded and detect is on, else none.
    cache_root: where scenes and a bag's messages are cached (None: the working folder's <project>/splat)."""
    t_start = time.perf_counter()
    st = settings or settings_of(project)
    stop = cancel or (lambda: False)
    if engine is None and project is not None:
        from .analysis import Engine
        engine = Engine()
    if cache_root is None:
        from .. import workdir
        cache_root = workdir.project_dir(project, None, 'splat') if project is not None else None
    timing = {}
    try:
        t0 = time.perf_counter()
        if drive is None:
            drive = open_drive(st.drive, cache_root, lambda f, t='': _prog(progress, 0.05 * f, t), stop)
        timing['open_s'] = round(time.perf_counter() - t0, 2)
        t0 = time.perf_counter()
        if scene is None:
            scene = build_scene(drive, st, project, engine, lambda f, t='': _prog(progress, 0.05 + 0.45 * f, t),
                                stop, cache_root)
        timing['scene_s'] = round(time.perf_counter() - t0, 2)
        dev = SR.device_for(st.device, st.gpu_mem_gb)
        ts = TorchScene(scene, drive, dev)
        t0 = time.perf_counter()
        val = validate(ts, drive, st, project, engine, lambda f, t='': _prog(progress, 0.5 + 0.1 * f, t), stop)
        timing['validation_s'] = round(time.perf_counter() - t0, 2)
        ra, rb = resolve_rigs(project, drive, st, engine, variant)
        de_st = DE.settings_of(project) if project is not None else DE.DetectionSettings()
        det_info, notes = None, []
        det_fn = detector
        if det_fn is None and st.detect:
            s = DT.status(de_st.model)
            if s['verified']:
                d = DT.load(de_st.detector())
                det_fn, det_info = d.detect, d.describe()
            else:
                notes.append(f'Detection skipped: {s["text"]} (Simulate ▸ Detection quality ▸ Download). The '
                             'comparison counts the objects each rig shows large and clear enough to be scored.')
        elif det_fn is not None:
            det_info = dict(getattr(det_fn, 'describe', lambda: {})() or {'model': 'custom'})
        elif not st.detect:
            notes.append('Detection is off: the comparison counts the objects each rig shows large and clear enough '
                         'to be scored.')
        t0 = time.perf_counter()
        ab = compare(ts, drive, scene, ra, rb, st, de_st, det_fn,
                     lambda f, t='': _prog(progress, 0.6 + 0.4 * f, t), stop)
        timing['ab_s'] = round(time.perf_counter() - t0, 2)
    except Cancelled:
        return None
    findings = _findings(drive, scene, val, ra, rb, ab, st, de_st, det_fn is not None)
    notes += ra.notes + rb.notes + list(drive.notes)
    res = SplatResult(st, drive.info(), _scene_doc(scene), val, ra.info(), rb.info(), ab, det_info, findings, notes,
                      timing, round(time.perf_counter() - t_start, 2))
    res._scene = scene                                  # noqa: SLF001 — kept for the export of the .ply
    res._drive = drive
    _prog(progress, 1.0, 'Done')
    return res


def _scene_doc(scene: Scene) -> dict:
    d = dict(scene.info)
    d.update({'source': scene.source, 'gaussians': scene.n, 'actors': len(scene.actor_ids),
              'origin_global_m': [round(float(x), 4) for x in scene.origin],
              'extent_m': [round(float(x), 2) for x in (scene.means.max(0) - scene.means.min(0))] if scene.n else []})
    return d


def _pc(x):
    return f'{100 * x:.0f} %'


def _findings(drive, scene, val, ra, rb, ab, st, de_st, detected) -> list:
    out = []
    cams = (val or {}).get('cameras') or {}
    p = cams.get('psnr')
    if p is not None:
        if p < st.min_psnr_db:
            _finding(out, 'serious', 'validation',
                     f'Re-rendered held-out frames reach {p:.1f} dB PSNR (SSIM {cams.get("ssim", 0):.2f}) against '
                     f'the recorded images, under {st.min_psnr_db:g} dB: the reconstruction does not validate, and the '
                     'A/B figures are not to be trusted.')
        else:
            _finding(out, 'good', 'validation',
                     f'Re-rendered held-out frames reach {p:.1f} dB PSNR (SSIM {cams.get("ssim", 0):.2f}) against '
                     f'the recorded images ({cams.get("frames", 0)} frames).')
    lv = (val or {}).get('lidar') or {}
    if lv:
        ref = lv.get('median_abs_error_steep_m')
        ref = lv['median_abs_error_m'] if ref is None else ref
        sev = 'good' if ref <= 0.1 else ('note' if ref <= 0.3 else 'warning')
        _finding(out, sev, 'validation',
                 f'Re-rendered held-out lidar sweeps: median range error {100 * lv["median_abs_error_m"]:.1f} cm '
                 f'({_pc(lv["within_10cm"])} within 10 cm; '
                 + (f'{100 * lv["median_abs_error_steep_m"]:.1f} cm where the beam meets the surface within '
                    f'{STEEP_DEG:g}° of square, {_pc(lv["steep_share"])} of the beams; '
                    if lv.get('median_abs_error_steep_m') is not None else '')
                 + f'grazing beams are where a centimetre of height is metres of range), '
                 f'{_pc(lv["return_agreement"])} of the beams agree on returning or not.')
    if drive.tracks and st.dynamic == 'boxes':
        n_dyn = len(scene.actor_ids)
        _finding(out, 'note', 'dynamic',
                 f'{n_dyn} moving object{"s" if n_dyn != 1 else ""} rebuilt as rigid bodies in their annotated boxes '
                 f'({drive.box_source}); the rest of the scene is static.')
    else:
        _finding(out, 'warning', 'dynamic',
                 'Everything is treated as static' + (' (Moving objects: ignore)' if drive.tracks else
                                                      ': the drive has no 3D boxes') +
                 ': vehicles and pedestrians that moved during the drive are smeared along their paths.')
    rows = ab.get('recall') or []
    cov = ab.get('coverage') or []
    classes = ab.get('classes') or []
    if 'traffic_light' in classes and (any(r['class'] == 'traffic_light' for r in rows + cov) or
                                       any(t.cls == 'traffic_light' for t in drive.tracks.values())):
        n_rec = sum(r['recorded'] for r in cov if r['class'] == 'traffic_light')
        n_ren = sum(r['rendered'] for r in cov if r['class'] == 'traffic_light')
        _finding(out, 'note', 'coverage',
                 'Traffic lights are mostly not rebuilt (thin and high, they catch few lidar points)'
                 + (f': the re-rendered recorded rig shows {n_ren} of the {n_rec} the recording\'s ground truth shows '
                    f'at these instants' if n_rec else '')
                 + ', so their recall is not assessed (their rows are in the table only for completeness).')
    in_group = {m for g in classes if g in DT.GROUPS for m in DT.GROUPS[g]}
    for ec in classes:
        if ec == 'traffic_light' or ec in in_group:
            continue
        short = [r for r in cov if r['class'] == ec and r['recorded'] >= max(de_st.min_gt, 3)
                 and r['rendered'] < 0.8 * r['recorded']]
        if short:
            _finding(out, 'note', 'coverage',
                     f'{DT.CLASS_LABEL.get(ec, ec)}: the re-rendered recorded rig shows '
                     + ', '.join(f'{r["rendered"]} of the {r["recorded"]} at {r["range"]}' for r in short)
                     + ' that the recording\'s ground truth shows at these instants: objects the lidar did not rebuild '
                       'finely enough are not scored, so recall there is on an easier set than the recorded images\'.')
    for r in rows:
        if r['objects'] < max(de_st.min_gt, 3) or r['class'] == 'traffic_light':
            continue
        if detected and r.get('delta') is not None and abs(r['delta']) >= 0.05:
            sev = 'warning' if r['delta'] <= -0.10 else 'note'
            ci = r.get('delta_ci')
            cis = f' (95 % interval {100 * ci[0]:+.0f} to {100 * ci[1]:+.0f} points)' if ci else ''
            _finding(out, sev, 'recall',
                     f'{DT.CLASS_LABEL.get(r["class"], r["class"])} recall at {r["range"]}: {rb.name} '
                     f'{_pc(r["recall_b"])} against {ra.name} {_pc(r["recall_a"])}, '
                     f'{100 * r["delta"]:+.0f} points{cis}, {r["objects"]} objects (re-rendered images).')
        elif not detected and abs(r['visible_share_b'] - r['visible_share_a']) >= 0.05:
            _finding(out, 'note', 'visible',
                     f'{DT.CLASS_LABEL.get(r["class"], r["class"])} at {r["range"]}: {rb.name} shows '
                     f'{_pc(r["visible_share_b"])} of the {r["objects"]} objects large and clear enough to score, '
                     f'{ra.name} {_pc(r["visible_share_a"])}.')
    for side, rig in (('a', ra), ('b', rb)):
        for c in rig.cameras:
            if c.body is not None and c.body.mean() > 0.25:
                _finding(out, 'note', 'body',
                         f'{rig.name} camera {c.tag}: {_pc(float(c.body.mean()))} of the image is the own vehicle.')
    la = ab.get('lidar', {}).get('a', {})
    lb = ab.get('lidar', {}).get('b', {})
    if la and lb:
        ra_ret = sum(v['returns'] for v in la.values()) / max(sum(v['sweeps'] for v in la.values()), 1)
        rb_ret = sum(v['returns'] for v in lb.values()) / max(sum(v['sweeps'] for v in lb.values()), 1)
        _finding(out, 'note', 'lidar',
                 f'Lidar returns per sweep: {rb.name} {rb_ret:,.0f}, {ra.name} {ra_ret:,.0f} (re-rendered).')
    _finding(out, 'note', 'caveat',
             'Re-rendered images are not real images: on the same objects recall on them is a little lower than on '
             'recorded images, and objects the lidar did not rebuild finely enough (far, small and thin ones, such as '
             'traffic lights) are not scored at all, so absolute recall is not a recorded image\'s; compare the rigs, '
             'not the figures with other studies.')
    return out


def summary(res: SplatResult) -> dict:
    cams = (res.validation or {}).get('cameras') or {}
    lv = (res.validation or {}).get('lidar') or {}
    return {'drive': os.path.basename(str(res.drive.get('path', ''))), 'gaussians': res.scene.get('gaussians'),
            'actors': res.scene.get('actors'), 'held_out_psnr_db': cams.get('psnr'), 'held_out_ssim': cams.get('ssim'),
            'lidar_median_error_m': lv.get('median_abs_error_m'), 'rig_a': res.rig_a.get('name'),
            'rig_b': res.rig_b.get('name'), 'instants': len(res.ab.get('instants') or []),
            'detector': (res.detector or {}).get('name') or (res.detector or {}).get('model'),
            'findings': [f['message'] for f in res.findings if f['severity'] in ('warning', 'serious', 'critical')]}


def to_doc(res: SplatResult, project=None) -> dict:
    st = res.settings
    ab = {k: v for k, v in res.ab.items() if k != 'previews'}
    return {'schema': SCHEMA, 'project': getattr(project, 'name', ''), 'frame': FRAME,
            'units': {'length': 'm', 'angle': 'deg', 'time': 's from the drive\'s first ego pose', 'psnr': 'dB',
                      'recall': 'share 0–1'},
            'settings': st.to_dict(), 'detection_settings': (DE.settings_of(project).to_dict()
                                                             if project is not None else None),
            'drive': res.drive, 'scene': res.scene, 'validation': res.validation, 'rig_a': res.rig_a,
            'rig_b': res.rig_b, 'ab': ab, 'detector': res.detector, 'findings': res.findings, 'notes': res.notes,
            'timing': res.timing, 'seconds': res.seconds, 'assumptions': ASSUMPTIONS,
            'limits': ('Static scene except annotated actors (rigid bodies); no view-dependent colour or exposure '
                       'model; holes where the lidar never reached; the affine (EWA) projection of each Gaussian; '
                       'the depth order per Gaussian, not per pixel; the A/B instants are the recording\'s camera '
                       'times; detector figures on re-rendered images, not real ones.')}


def recall_csv(res: SplatResult) -> str:
    cols = ('class', 'range', 'objects', 'visible_a', 'visible_b', 'visible_share_a', 'visible_share_b', 'recall_a',
            'recall_b', 'delta', 'delta_ci', 'lidar_points_a', 'lidar_points_b')
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator='\n')
    w.writerow(cols)
    for r in res.ab.get('recall') or []:
        w.writerow(['' if r.get(c) is None else (' '.join(f'{x:g}' for x in r[c]) if isinstance(r.get(c), list)
                                                   else r[c]) for c in cols])
    return buf.getvalue()


def validation_csv(res: SplatResult) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator='\n')
    w.writerow(('camera', 'frames', 'psnr_db', 'ssim'))
    for tag, v in ((res.validation or {}).get('cameras') or {}).get('cameras', {}).items():
        w.writerow((tag, v['frames'], v['psnr'], v['ssim']))
    return buf.getvalue()


def preview_png(res: SplatResult, k: int = 0, width: int = 320) -> bytes | None:
    """Rig A's cameras over rig B's at instant k, labelled."""
    import cv2
    from .pngtext import put_text
    pv = res.previews
    if not pv:
        return None
    p = pv[min(max(k, 0), len(pv) - 1)]
    rows = []
    order = list(p['a']) + [t for t in p['b'] if t not in p['a']]
    for side, name in (('a', res.rig_a.get('name', 'A')), ('b', res.rig_b.get('name', 'B'))):
        tiles = []
        for tag in [t for t in order if t in p[side]]:
            img = p[side][tag]
            im = cv2.cvtColor(_thumb(img, width), cv2.COLOR_RGB2BGR)
            im = cv2.copyMakeBorder(im, 22, 2, 2, 2, cv2.BORDER_CONSTANT, value=(24, 24, 24))
            put_text(im, f'{side.upper()} {tag}', (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 235), 1,
                     cv2.LINE_AA)
            tiles.append(im)
        if tiles:
            hmax = max(t.shape[0] for t in tiles)
            tiles = [cv2.copyMakeBorder(t, 0, hmax - t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(24, 24, 24))
                     for t in tiles]
            row = np.concatenate(tiles, 1)
            head = np.full((24, row.shape[1], 3), 16, np.uint8)
            put_text(head, f'{side.upper()}: {name} · t = {p["t"]:.2f} s (re-rendered)', (6, 17),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
            rows.append(np.concatenate([head, row], 0))
    if not rows:
        return None
    wmax = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 0, 0, wmax - r.shape[1], cv2.BORDER_CONSTANT, value=(16, 16, 16)) for r in rows]
    ok, buf = cv2.imencode('.png', np.concatenate(rows, 0))
    return bytes(buf) if ok else None


def lidar_topview(P, half_m: float = 40.0, px: int = 480) -> np.ndarray:
    """(px, px, 3) uint8 RGB: points (vehicle frame) seen from above, coloured by height, x up the image."""
    img = np.full((px, px, 3), 18, np.uint8)
    if P is None or not len(P):
        return img
    P = np.asarray(P, float)
    s = px / (2 * half_m)
    u = (px / 2 - P[:, 1] * s).astype(int)
    v = (px / 2 - P[:, 0] * s).astype(int)
    ok = (u >= 0) & (u < px) & (v >= 0) & (v < px)
    z = np.clip((P[ok, 2] + 0.5) / 4.0, 0, 1)
    col = np.stack([60 + 195 * z, 200 - 80 * z, 255 - 195 * z], 1).astype(np.uint8)
    img[v[ok], u[ok]] = col
    c = px // 2
    img[c - 6:c + 6, c - 2:c + 2] = (240, 240, 240)
    return img


def lidar_png(res: SplatResult, k: int = 0) -> bytes | None:
    import cv2
    from .pngtext import put_text
    pv = res.previews
    if not pv:
        return None
    p = pv[min(max(k, 0), len(pv) - 1)]
    tiles = []
    for side, name in (('a', res.rig_a.get('name', 'A')), ('b', res.rig_b.get('name', 'B'))):
        im = cv2.cvtColor(lidar_topview(p.get('lidar_' + side)), cv2.COLOR_RGB2BGR)
        put_text(im, f'{side.upper()}: {name}', (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1,
                 cv2.LINE_AA)
        tiles.append(im)
    ok, buf = cv2.imencode('.png', np.concatenate(tiles, 1))
    return bytes(buf) if ok else None


def files(res: SplatResult, project=None) -> list:
    """[(name, bytes)]: the JSON, the recall and validation CSVs, the previews and (export_ply) the scene's .ply."""
    out = [('log_replay.json', (json.dumps(to_doc(res, project), indent=1, default=float) + '\n').encode('utf-8')),
           ('log_replay_recall.csv', recall_csv(res).encode('utf-8')),
           ('log_replay_validation.csv', validation_csv(res).encode('utf-8'))]
    png = preview_png(res)
    if png:
        out.append(('log_replay_cameras.png', png))
    png = lidar_png(res)
    if png:
        out.append(('log_replay_lidar.png', png))
    sc = getattr(res, '_scene', None)
    if res.settings.export_ply and sc is not None:
        dr = getattr(res, '_drive', None)
        t = (res.ab.get('instants') or [None])[0]
        out.append(('log_replay_scene.ply', ply_bytes(sc, dr, t)))
    return out


def export(res: SplatResult, project, out_dir) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for name, data in files(res, project):
        p = os.path.join(out_dir, name)
        with open(p, 'wb') as fh:
            fh.write(data)
        paths.append(p)
    return paths


def preview(drive: Drive, scene: Scene, rigs: dict, t: float, st: SplatSettings, cancel=None) -> dict:
    """Re-rendered views of each rig at drive time t: {side: {'cameras': {tag: RGB uint8}, 'lidar': (n, 3) vehicle
    frame}} for rigs {side: RigSpec}."""
    dev = SR.device_for(st.device, st.gpu_mem_gb)
    ts = TorchScene(scene, drive, dev)
    out = {}
    for side, rig in rigs.items():
        if cancel is not None and cancel():
            raise Cancelled()
        cams = {tag: v[0] for tag, v in _render_rig_frame(ts, drive, rig, t, scene).items()}
        pts = [lidar_points_vehicle(L, render_lidar(ts, drive.traj, L, t - L.period_s, st), drive.traj, t)
               for L in rig.lidars]
        out[side] = {'cameras': cams, 'lidar': np.concatenate(pts) if pts else np.zeros((0, 3))}
    return out
