"""High-fidelity neural log replay (Simulate ▸ Log replay ▸ Neural replay): the log replay's Gaussian scene
(core.splat_reproject) projected through each camera's exact model by the Unscented Transform, with a lidar opacity
of its own per Gaussian, pedestrians as articulated bodies posed from the recorded images, traffic lights as emissive
planes showing their recorded state over time, and the re-shot frames relit to conditions that were never driven
(core.relight). Every addition is measured against the log replay it extends, on the same drive.

1. The Unscented Transform (setting projection 'ut'; 'ewa' is the log replay's affine projection)
------------------------------------------------------------------------------------------------
Wu, Martinez Esturo, Mirzaei, Moenne-Loccoz and Gojcic, "3DGUT: Enabling Distorted Cameras and Secondary Rays in
Gaussian Splatting", CVPR 2025 (arXiv:2412.12507). Each Gaussian's seven sigma points go through the camera's own
projection (Kannala-Brandt fisheyes far beyond 90°, any core.lens model, a lidar's spherical beam grid); their weighted
mean and covariance say where it lands and which tiles it touches, and its response at a pixel is evaluated in 3D,
the Gaussian's maximum along that pixel's ray, so nothing about the projection is approximated
(core.splat_raster.splat_ut). Rolling shutter is exact per row: every pixel's ray leaves the camera where it was when
its row was read, and each sigma point is moved by the camera's motion at the row it lands on (two fixed-point passes;
the ego pose tabulated at 33 rows). Lidars: every beam is cast along its exact direction from where the sensor was
when its column fired. NVIDIA's reference implementation (github.com/nv-tlabs/3dgrut) is Apache-2.0 (its LICENSE, read
2026-11-11); none of its code is used: the transform is written into Vantage's own torch rasteriser.

2. Two opacities (lidar_opacity)
--------------------------------
Every Gaussian has a camera opacity (the photometric optimisation's) and a lidar opacity, fitted after it to the
training sweeps alone: per beam, the binary cross-entropy of the accumulated opacity against whether the recorded
beam returned, plus the range error where both return (Adam on the lidar opacities only, the geometry fixed). Glass
is the case it is for: see-through for the camera, a return where the lidar sees one. Validation: held-out sweeps
with each opacity.

3. Articulated walkers (walkers 'articulated')
----------------------------------------------
* The body: Vantage's own articulated model (permissive: part of Vantage): a kinematic tree of 13 segments (pelvis,
  torso, head, upper arms, forearms, thighs, shanks, feet) with 22 joint angles and offsets per instant and one scale
  per walker; segment lengths and joint heights as fractions of the stature from Winter, "Biomechanics and Motor
  Control of Human Movement", 4th ed., Wiley 2009, Fig. 4.1 (after Drillis and Contini 1966); the stature is the
  annotated box's height. Its surface is Gaussians on capsules and ellipsoids around the segments, each bound to its
  segment and posed by the skeleton (3D Gaussians attached to a skeleton), coloured from the recorded views that saw
  the walker (its own pixels by the ground truth's instance image when there is one).
* The pose per instant: a gait prior from the track alone (the box's speed drives a walking cycle: sagittal hip,
  knee, shoulder and elbow angles shaped like Winter 2009's normal-gait curves, Ch. 4; amplitudes scale with speed,
  standing below 0.2 m/s), then, when the keypoint model is downloaded, fitted to 2D keypoints of every recorded view
  of the walker (ViTPose-B, Apache-2.0, top-down in the box the ground truth or the projected 3D box gives): robust
  reprojection error through each camera's exact model, plus the prior, temporal smoothness and joint limits.
* Parametric body models: Anny (NAVER LABS Europe, Apache-2.0, github.com/naver/anny, licence read 2026-11-11) is a
  permissively licensed parametric body (MakeHuman-based); SMPL (Loper et al. 2015) is licensed for non-commercial
  use only. Neither is bundled. body_model 'anny' downloads, on request, Anny's MakeHuman base mesh (base.obj,
  released CC0 by the MakeHuman copyright holders) and its default skinning weights (weights.default.json, CC0) from
  a pinned commit of github.com/naver/anny, SHA-256 checked; its body surface, scaled to the walker's stature,
  becomes the walker's Gaussians, each vertex bound to the skeleton segment its weights favour (the MakeHuman rig's
  163 bones grouped into the 13 segments). Anny's phenotype blend shapes (age, weight, …) are not used: the shape is
  the template's, scaled by the stature. body_model 'smpl' reads a model file the user downloads from smpl.is.tue.mpg.de
  after accepting its licence (opt-in: the setting says so and smpl_accept must be on; the .npz form, read without
  pickles); its template surface, scaled to the walker's stature, becomes the walker's Gaussians, each vertex bound to
  the skeleton segment of its largest SMPL skinning weight (rigid per segment, not SMPL's blend skinning or its pose
  correctives) and posed by the same fitted skeleton. Without it, the capsule body.

4. Traffic lights (traffic_lights)
----------------------------------
Each traffic-light head of the drive's ground truth (class traffic_light; a CARLA run's heads are their own objects)
becomes an emissive plane on its face: a housing panel and three lamps (red over amber over green, the usual
vertical head), the lit lamp's colour as recorded. The face is the side of the head the recorded cameras saw lamps on
(else the side facing the driven path). The state over time is the ground truth's when it states one ('state' on the
object), else read from the recorded images: in every frame that sees the face from the front (≥ min_light_px pixels)
the three lamp regions' brightness and saturation, the brightest clearly above the others is the state, held to the
nearest observation in time and shared by the heads of one pole (one actor). Never seen: dark lamps (state unknown).
Without boxes (light_source 'detector', or 'auto' on a drive without traffic-light boxes), the detector's traffic
lights (core.detection, every third frame) are lifted to 3D along their box centre's ray at the range of the static
Gaussians inside the box, merged across frames (within 1.5 m, at least two detections) and faced towards the
cameras that saw them (lights_from_detections); those heads are drawn but, not being the drive's objects, not
scored.

5. Relighting (core.relight)
----------------------------
The re-rendered frames under wet, night or snow, by a diffusion model on request (InstructPix2Pix, MIT) or the
physically based fallback (always available); relit frames are labelled relit, not recorded, wherever they go.

The study (run) and what it measures, on one drive
--------------------------------------------------
* Projection: the scene rebuilt twice with the same settings and seed, once per projection; held-out PSNR and SSIM per
  camera (fisheyes especially) and held-out lidar range error, EWA against UT.
* Lidar opacity: the held-out lidar validation with the camera's opacity against the fitted lidar opacity.
* Walkers and lights: at eval instants the recorded rig re-rendered with the rigid walkers and no light planes, and
  with articulated walkers and emissive lights; the detector's rig-level recall per class (an object counted once
  per instant, found when any camera finds it) on both, and on the recorded images at the same instants (the real
  reference); traffic lights shown and found against the recording's ground truth.
* Relighting: the detector's recall on the relit re-renders against the unrelit ones, per condition.

Limits: the walker's body is a capsule model unless SMPL is supplied; clothing, hair and carried objects are only
colour; keypoints come from fisheye crops a model trained on rectilinear photos has not seen (scored by their
confidence); moving actors are posed at a frame's middle-row time; a traffic light's state is read from a few pixels
when the recording does not state it; relit frames are not recordings of that weather.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from . import splat_reproject as SP
from . import splat_raster as SR

SCHEMA = 'vantage.neural_replay/1'
KEY = 'neural_replay'
EXT_VERSION = 1
FRAME = SP.FRAME
PROJECTIONS = SR.PROJECTIONS
WALKERS = ('articulated', 'rigid')
BODY_MODELS = ('capsules', 'anny', 'smpl')
LIGHT_SOURCES = ('auto', 'boxes', 'detector')
STATES = ('red', 'amber', 'green')
STATE_RGB = {'red': (1.0, 0.16, 0.10), 'amber': (1.0, 0.72, 0.08), 'green': (0.25, 1.0, 0.55)}
LIMITS = {'lidar_steps': (0, 5000), 'lidar_lr': (1e-3, 1.0), 'walker_gaussians': (200, 6000),
          'fit_steps': (0, 5000), 'min_kp_score': (0.05, 0.95), 'min_walker_px': (20, 100000),
          'min_light_px': (2, 10000), 'eval_instants': (1, 200), 'gpu_mem_gb': (0.5, 5.5), 'seed': (0, 2 ** 31 - 1),
          'max_pose_crops': (1, 5000), 'kp_sigma_rel': (0.005, 0.5), 'prior_weight': (0.0, 100.0),
          'smooth_weight': (0.0, 1000.0)}
INTS = ('lidar_steps', 'walker_gaussians', 'fit_steps', 'min_walker_px', 'min_light_px', 'eval_instants', 'seed',
        'max_pose_crops')
BOOLS = ('lidar_opacity', 'traffic_lights', 'keypoints', 'smpl_accept', 'compare_projection', 'compare_walkers',
         'compare_lidar', 'real_reference')
ASSUMPTIONS = (
    'Neural replay extends the log replay\'s reconstruction: the Unscented Transform projects every Gaussian through '
    'the camera\'s exact model (3DGUT, Wu et al. 2025) with the rolling shutter exact per row; a lidar opacity per '
    'Gaussian is fitted to the training sweeps alone; pedestrians are articulated bodies (Vantage\'s capsule model, '
    'segment proportions from Winter 2009) posed by a gait prior from their track and, with the keypoint model, by '
    'the 2D keypoints of their recorded views; traffic-light heads are emissive planes with the state the recording '
    'shows (or states) over time. The images it renders are still not recordings: compare rigs and conditions on '
    'the same reconstruction, not the absolute figures with recorded ones.')


def _f(v, default):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def _prog(progress, f, text=''):
    SP._prog(progress, f, text)


class Cancelled(Exception):
    pass


# =========================================================================== settings
@dataclass
class NeuralSettings:
    projection: str = 'ut'              # ut: the Unscented Transform (measured better); ewa: the affine projection
    lidar_opacity: bool = True          # a lidar opacity per Gaussian, fitted to the training sweeps
    lidar_steps: int = 300
    lidar_lr: float = 0.05
    walkers: str = 'articulated'        # articulated | rigid (the log replay's)
    body_model: str = 'capsules'        # capsules (Vantage's own) | smpl (opt-in, the user's file)
    smpl_path: str = ''
    smpl_accept: bool = False           # the user accepted SMPL's non-commercial licence
    keypoints: bool = True              # fit the walkers to 2D keypoints when the pose model is downloaded
    pose_model: str = 'vitpose_b'
    walker_gaussians: int = 1500
    fit_steps: int = 400
    min_kp_score: float = 0.3
    min_walker_px: int = 150            # a recorded view of a walker this large or larger gives keypoints
    max_pose_crops: int = 600
    kp_sigma_rel: float = 0.04          # keypoint error scale: this share of the walker's box height in the image
    prior_weight: float = 1.0
    smooth_weight: float = 20.0
    traffic_lights: bool = True
    light_source: str = 'auto'          # auto: the ground truth's boxes when there are some, else the detector
    min_light_px: int = 6
    relight: dict = field(default_factory=dict)   # core.relight.RelightSettings
    eval_instants: int = 24             # the study: instants of the drive scored
    compare_projection: bool = True     # the study: rebuild with the other projection and validate both
    compare_walkers: bool = True        # the study: rigid against articulated, lights off against on
    compare_lidar: bool = True          # the study: the camera's opacity against the lidar's on held-out sweeps
    real_reference: bool = True         # the study: the detector on the recorded images at the same instants
    relight_conditions: list = field(default_factory=lambda: ['wet', 'night', 'snow'])
    gpu_mem_gb: float = 5.0             # each GPU job stays under 6 GB of VRAM
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d) -> 'NeuralSettings':
        """Tolerant: unknown keys dropped, numbers clamped to LIMITS, words checked, junk back to the default."""
        s = NeuralSettings()
        d = d if isinstance(d, dict) else {}
        for k, opts in (('projection', PROJECTIONS), ('walkers', WALKERS), ('body_model', BODY_MODELS),
                        ('light_source', LIGHT_SOURCES), ('pose_model', tuple(POSE_MODELS))):
            if d.get(k) in opts:
                setattr(s, k, d[k])
        if isinstance(d.get('smpl_path'), str):
            s.smpl_path = d['smpl_path'].strip()[:4096]
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
        if isinstance(d.get('relight'), dict):
            from . import relight as RL
            s.relight = RL.RelightSettings.from_dict(d['relight']).to_dict()
        rc = d.get('relight_conditions')
        if isinstance(rc, (list, tuple)):
            from . import relight as RL
            s.relight_conditions = [c for c in dict.fromkeys(rc) if c in RL.CONDITIONS and c != 'none']
        return s

    def build_key(self) -> dict:
        """What the scene's extension depends on (its cache key)."""
        keys = ('lidar_opacity', 'lidar_steps', 'lidar_lr', 'walkers', 'body_model', 'smpl_path', 'keypoints',
                'pose_model', 'walker_gaussians', 'fit_steps', 'min_kp_score', 'min_walker_px', 'max_pose_crops',
                'kp_sigma_rel', 'prior_weight', 'smooth_weight', 'traffic_lights', 'light_source', 'min_light_px',
                'seed')
        d = {k: getattr(self, k) for k in keys}
        d['version'] = EXT_VERSION
        d['pose_ready'] = bool(self.keypoints and pose_status(self.pose_model, verify=False)['verified'])
        d['smpl'] = smpl_usable(self) is None
        d['anny'] = self.body_model == 'anny' and anny_status(verify=False)['verified']
        return d


def settings_of(project) -> NeuralSettings:
    return NeuralSettings.from_dict(getattr(project.analysis, KEY, None) if project is not None else None)


# =========================================================================== the keypoint model
def _pose_specs():
    from . import foundation_models as FM
    return {
        'vitpose_b': FM.FoundationSpec(
            'vitpose_b', 'ViTPose-B (COCO keypoints)', 'pose', 'vitpose', 'usyd-community/vitpose-base-simple',
            'a93ac0c67e0b7e2c55287d21d4c460c8f3c54d45',
            FM._f(('config.json', 1798, '72fc2f0fc29eef43e87aa0ddd4c1269c5226ae157a99d0057019dd40129fed7c'),
                  ('preprocessor_config.json', 363,
                   '9b11cadc98c30b968a70cc1658ce1fbd74b721b0f21402a2ff8bc1dc9d1474a0'),
                  ('model.safetensors', 343673076,
                   '85375373893ddd3641f3912821073e53f5435f9e966e1dca59d004454bfe4fdf')),
            'Apache-2.0', 'https://huggingface.co/usyd-community/vitpose-base-simple', True,
            'Apache-2.0 (the model card\'s licence field, read 2026-11-11)',
            'Xu, Zhang, Zhang and Tao: ViTPose: Simple Vision Transformer Baselines for Human Pose Estimation, '
            'NeurIPS 2022, arXiv:2204.12484 (COCO keypoints; trained on COCO, CC-BY-4.0 annotations)', 86.0,
            note='top-down: 17 COCO keypoints in each walker\'s box; 256 × 192 input'),
    }


POSE_MODELS = ('vitpose_b',)


def pose_spec(model='vitpose_b'):
    return _pose_specs()[model if model in POSE_MODELS else 'vitpose_b']


def pose_status(model='vitpose_b', folder=None, verify=True) -> dict:
    from . import foundation_models as FM
    return FM.status(pose_spec(model), folder, verify=verify)


def download_pose_model(model='vitpose_b', progress=None, cancel=None, folder=None, fetch=None):
    """Download the keypoint model from its pinned revision (core.foundation_models.download: size and SHA-256
    checked, Apache-2.0)."""
    from . import foundation_models as FM
    return FM.download(pose_spec(model), progress, cancel, folder, fetch=fetch)


class PoseModel:
    """ViTPose through Hugging Face Transformers (optional imports): keypoints(image RGB uint8, boxes [[x0, y0, x1,
    y1], …]) -> (n, 17, 3) image coordinates (pixel-edge convention) and scores."""

    def __init__(self, model='vitpose_b', folder=None, device='auto', mem_gb=3.0):
        from . import foundation_models as FM
        sp = pose_spec(model)
        path = FM._load_check(sp, folder)
        import torch
        from transformers import VitPoseForPoseEstimation, VitPoseImageProcessor
        self.dev, self.device_info = FM._torch_device(device, mem_gb)
        self.proc = VitPoseImageProcessor.from_pretrained(path)
        self.model = VitPoseForPoseEstimation.from_pretrained(path).to(self.dev).eval()
        self.torch = torch
        self.spec = sp

    def describe(self) -> dict:
        return {'id': self.spec.id, 'name': self.spec.name, 'revision': self.spec.revision,
                'licence': self.spec.licence, 'device': str(self.dev)}

    def keypoints(self, img, boxes) -> np.ndarray:
        if not len(boxes):
            return np.zeros((0, 17, 3))
        torch = self.torch
        coco = [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in boxes]
        inp = self.proc(images=img, boxes=[coco], return_tensors='pt')
        with torch.no_grad():
            out = self.model(pixel_values=inp['pixel_values'].to(self.dev))
        res = self.proc.post_process_pose_estimation(out, boxes=[coco])[0]
        k = np.zeros((len(boxes), 17, 3))
        for i, r in enumerate(res):
            k[i, :, :2] = np.asarray(r['keypoints'].cpu() if hasattr(r['keypoints'], 'cpu') else r['keypoints'],
                                     float)[:17]                     # in the boxes' own image coordinates
            k[i, :, 2] = np.asarray(r['scores'].cpu() if hasattr(r['scores'], 'cpu') else r['scores'], float)[:17]
        return k


# =========================================================================== the articulated body
# Joint heights and segment lengths as fractions of the stature H: Winter, "Biomechanics and Motor Control of Human
# Movement", 4th ed., Wiley 2009, Fig. 4.1 (after Drillis and Contini 1966): head top 1.000, eye 0.936, chin 0.870,
# shoulder 0.818, elbow 0.630, wrist 0.485, knuckle 0.377, hip (greater trochanter) 0.530, knee 0.285, ankle 0.039;
# shoulder width 0.259, hip width 0.191, foot length 0.152. The face's keypoints (eyes, ears, nose) are placed from
# the eye height and the head's size; radii of the segments' surfaces are this model's own (a lean adult).
WINTER = {'chin': 0.870, 'shoulder': 0.818, 'elbow': 0.630, 'wrist': 0.485, 'knuckle': 0.377, 'hip': 0.530,
          'knee': 0.285, 'ankle': 0.039, 'shoulder_w': 0.259, 'hip_w': 0.191, 'foot': 0.152, 'eye': 0.936}
DOF = ('tx', 'ty', 'tz', 'yaw', 'lean', 'torso_flex', 'torso_roll', 'torso_twist', 'head_yaw', 'head_pitch',
       'l_sh_flex', 'l_sh_abd', 'r_sh_flex', 'r_sh_abd', 'l_elbow', 'r_elbow', 'l_hip_flex', 'l_hip_abd',
       'r_hip_flex', 'r_hip_abd', 'l_knee', 'r_knee')
NDOF = len(DOF)
D_ = {k: i for i, k in enumerate(DOF)}
# joint limits (rad; m for the offsets): a person's range, loose
DOF_LO = np.array([-0.4, -0.4, -0.15, -math.pi, -0.35, -0.35, -0.35, -0.7, -1.3, -0.7, -1.2, -0.25, -1.2, -0.25,
                   0.0, 0.0, -0.6, -0.35, -0.6, -0.35, 0.0, 0.0])
DOF_HI = np.array([0.4, 0.4, 0.15, math.pi, 0.35, 0.9, 0.35, 0.7, 1.3, 0.7, 2.6, 1.7, 2.6, 1.7, 2.5, 2.5, 1.9, 0.6,
                   1.9, 0.6, 2.5, 2.5])
# the prior's spread per joint (rad): how far a fit may stray from the gait prior at little cost
DOF_SIG = np.array([0.15, 0.15, 0.05, 0.6, 0.15, 0.25, 0.15, 0.3, 0.6, 0.4, 0.6, 0.4, 0.6, 0.4, 0.6, 0.6, 0.4, 0.2,
                    0.4, 0.2, 0.5, 0.5])
SEGMENTS = ('pelvis', 'torso', 'head', 'l_upperarm', 'l_forearm', 'r_upperarm', 'r_forearm', 'l_thigh', 'l_shank',
            'l_foot', 'r_thigh', 'r_shank', 'r_foot')
NSEG = len(SEGMENTS)
KEYPOINTS = ('nose', 'l_eye', 'r_eye', 'l_ear', 'r_ear', 'l_shoulder', 'r_shoulder', 'l_elbow', 'r_elbow',
             'l_wrist', 'r_wrist', 'l_hip', 'r_hip', 'l_knee', 'r_knee', 'l_ankle', 'r_ankle')   # COCO's order
FLIP = (0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15)                          # left <-> right


def _rx(a):
    import torch
    c, s = torch.cos(a), torch.sin(a)
    o, z = torch.ones_like(a), torch.zeros_like(a)
    return torch.stack([o, z, z, z, c, -s, z, s, c], -1).reshape(*a.shape, 3, 3)


def _ry(a):
    import torch
    c, s = torch.cos(a), torch.sin(a)
    o, z = torch.ones_like(a), torch.zeros_like(a)
    return torch.stack([c, z, s, z, o, z, -s, z, c], -1).reshape(*a.shape, 3, 3)


def _rz(a):
    import torch
    c, s = torch.cos(a), torch.sin(a)
    o, z = torch.ones_like(a), torch.zeros_like(a)
    return torch.stack([c, -s, z, s, c, z, z, z, o], -1).reshape(*a.shape, 3, 3)


def _mv(R, v):
    import torch
    return torch.einsum('...ij,...j->...i', R, v)


def forward_kinematics(theta, stature, box_h, scale=None):
    """The skeleton in its box's frame (x along the box's heading, y left, z up, origin the box's centre, the ground
    at z = −box_h / 2): theta (T, NDOF) torch, stature H (m), scale (T,) or None. Returns (segment rotations (T, NSEG,
    3, 3), segment origins (T, NSEG, 3), COCO keypoints (T, 17, 3)). Flexions rotate a limb forward (a knee's
    backward), abductions outward; the yaw turns the whole body about the vertical."""
    import torch
    th = theta
    T = th.shape[0]
    dt = th.dtype
    H = torch.as_tensor(float(stature), dtype=dt, device=th.device)
    if scale is not None:
        H = H * scale.reshape(-1)
    H = H.expand(T) if H.dim() else H.repeat(T)
    a = lambda k: th[:, D_[k]]  # noqa: E731

    def v(x, y, z):
        return torch.stack([H * x, H * y, H * z], -1)
    W = WINTER
    R0 = _rz(a('yaw')) @ _ry(a('lean'))
    o0 = torch.stack([a('tx'), a('ty'), -box_h / 2 + W['hip'] * H + a('tz')], -1)
    Rt = R0 @ _rz(a('torso_twist')) @ _ry(-a('torso_flex')) @ _rx(a('torso_roll'))
    up = W['shoulder'] - W['hip']
    sh_w = W['shoulder_w'] / 2
    chin = o0 + _mv(Rt, v(0.0, 0.0, W['chin'] - W['hip']))
    Rh = Rt @ _rz(a('head_yaw')) @ _ry(-a('head_pitch'))
    sh_l = o0 + _mv(Rt, v(0.0, sh_w, up))
    sh_r = o0 + _mv(Rt, v(0.0, -sh_w, up))
    Rua_l = Rt @ _ry(-a('l_sh_flex')) @ _rx(a('l_sh_abd'))
    Rua_r = Rt @ _ry(-a('r_sh_flex')) @ _rx(-a('r_sh_abd'))
    el_l = sh_l + _mv(Rua_l, v(0.0, 0.0, -(W['shoulder'] - W['elbow'])))
    el_r = sh_r + _mv(Rua_r, v(0.0, 0.0, -(W['shoulder'] - W['elbow'])))
    Rfa_l = Rua_l @ _ry(-a('l_elbow'))
    Rfa_r = Rua_r @ _ry(-a('r_elbow'))
    wr_l = el_l + _mv(Rfa_l, v(0.0, 0.0, -(W['elbow'] - W['wrist'])))
    wr_r = el_r + _mv(Rfa_r, v(0.0, 0.0, -(W['elbow'] - W['wrist'])))
    hw = 0.9 * W['hip_w'] / 2                         # the hip joints' centres, inside the trochanters
    hip_l = o0 + _mv(R0, v(0.0, hw, -0.02))
    hip_r = o0 + _mv(R0, v(0.0, -hw, -0.02))
    Rth_l = R0 @ _ry(-a('l_hip_flex')) @ _rx(a('l_hip_abd'))
    Rth_r = R0 @ _ry(-a('r_hip_flex')) @ _rx(-a('r_hip_abd'))
    thigh = W['hip'] - 0.02 - W['knee']
    kn_l = hip_l + _mv(Rth_l, v(0.0, 0.0, -thigh))
    kn_r = hip_r + _mv(Rth_r, v(0.0, 0.0, -thigh))
    Rsh_l = Rth_l @ _ry(a('l_knee'))
    Rsh_r = Rth_r @ _ry(a('r_knee'))
    shank = W['knee'] - W['ankle']
    an_l = kn_l + _mv(Rsh_l, v(0.0, 0.0, -shank))
    an_r = kn_r + _mv(Rsh_r, v(0.0, 0.0, -shank))
    eye_dz = W['eye'] - W['chin']
    face = [_mv(Rh, v(0.055, 0.0, eye_dz - 0.03)), _mv(Rh, v(0.045, 0.018, eye_dz)),
            _mv(Rh, v(0.045, -0.018, eye_dz)), _mv(Rh, v(-0.005, 0.042, eye_dz - 0.01)),
            _mv(Rh, v(-0.005, -0.042, eye_dz - 0.01))]
    kp = torch.stack([chin + f for f in face] + [sh_l, sh_r, el_l, el_r, wr_l, wr_r, hip_l, hip_r, kn_l, kn_r,
                                                   an_l, an_r], 1)
    Rs = torch.stack([R0, Rt, Rh, Rua_l, Rfa_l, Rua_r, Rfa_r, Rth_l, Rsh_l, Rsh_l, Rth_r, Rsh_r, Rsh_r], 1)
    os_ = torch.stack([o0, o0, chin, sh_l, el_l, sh_r, el_r, hip_l, kn_l, an_l, hip_r, kn_r, an_r], 1)
    return Rs, os_, kp


def _seg_shapes(H: float):
    """Each segment's surface in its own frame (origin at its proximal joint, the limb along −z): (kind, centre,
    semi-axes or (length, radius)). Proportions: Winter 2009 Fig. 4.1 lengths; radii this model's own."""
    W = WINTER
    ua = W['shoulder'] - W['elbow']
    fa = W['elbow'] - W['knuckle']
    th = W['hip'] - 0.02 - W['knee']
    sk = W['knee'] - W['ankle']
    return {
        'pelvis': ('ellipsoid', (0.0, 0.0, -0.01 * H), (0.075 * H, 0.105 * H, 0.065 * H)),
        'torso': ('ellipsoid', (0.0, 0.0, 0.15 * H), (0.07 * H, 0.105 * H, 0.17 * H)),
        'head': ('ellipsoid', (0.012 * H, 0.0, 0.07 * H), (0.058 * H, 0.046 * H, 0.068 * H)),
        'l_upperarm': ('capsule', ua * H, 0.027 * H), 'r_upperarm': ('capsule', ua * H, 0.027 * H),
        'l_forearm': ('capsule', fa * H, 0.021 * H), 'r_forearm': ('capsule', fa * H, 0.021 * H),
        'l_thigh': ('capsule', th * H, 0.045 * H), 'r_thigh': ('capsule', th * H, 0.045 * H),
        'l_shank': ('capsule', sk * H, 0.032 * H), 'r_shank': ('capsule', sk * H, 0.032 * H),
        'l_foot': ('ellipsoid', (0.045 * H, 0.0, -0.022 * H), (0.085 * H, 0.032 * H, 0.025 * H)),
        'r_foot': ('ellipsoid', (0.045 * H, 0.0, -0.022 * H), (0.085 * H, 0.032 * H, 0.025 * H)),
    }


def _fib_sphere(n: int) -> np.ndarray:
    """n nearly even unit vectors (the Fibonacci lattice: deterministic)."""
    k = np.arange(n) + 0.5
    z = 1 - 2 * k / n
    r = np.sqrt(np.maximum(1 - z * z, 0))
    phi = math.pi * (3 - math.sqrt(5)) * k
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], 1)


def _frame_of(n: np.ndarray) -> np.ndarray:
    """(k, 3, 3) right-handed rotations whose last column is the normal n."""
    a = np.where(np.abs(n[:, 2:3]) < 0.9, np.array([[0.0, 0.0, 1.0]]), np.array([[1.0, 0.0, 0.0]]))
    t1 = np.cross(a, n)
    t1 /= np.maximum(np.linalg.norm(t1, axis=1, keepdims=True), 1e-12)
    t2 = np.cross(n, t1)
    return np.stack([t1, t2, n], 2)


def body_gaussians(stature: float, total: int = 1500):
    """The capsule body's surface as Gaussians in segment frames: (segment (n,), local centres (n, 3), scales (n, 3),
    quats (n, 4)), the count per segment in proportion to its area. Flat discs tangent to the surface, σ along it
    0.6 of the spacing, a tenth of that across."""
    H = float(stature)
    shapes = _seg_shapes(H)
    areas = {}
    for name, sh in shapes.items():
        if sh[0] == 'ellipsoid':
            a, b, c = sh[2]
            p = 1.6075
            areas[name] = 4 * math.pi * (((a * b) ** p + (a * c) ** p + (b * c) ** p) / 3) ** (1 / p)
        else:
            L, r = sh[1], sh[2]
            areas[name] = 2 * math.pi * r * L + 4 * math.pi * r * r
    tot = sum(areas.values())
    seg, P, S, Q = [], [], [], []
    for si, name in enumerate(SEGMENTS):
        sh = shapes[name]
        n = max(12, int(round(total * areas[name] / tot)))
        spacing = math.sqrt(areas[name] / n)
        if sh[0] == 'ellipsoid':
            u = _fib_sphere(n)
            ax = np.asarray(sh[2], float)
            p = np.asarray(sh[1], float) + u * ax
            nn = u / ax ** 2
        else:
            L, r = sh[1], sh[2]
            n_cap = max(4, int(round(n * (4 * math.pi * r * r) / areas[name])))
            n_cyl = max(4, n - n_cap)
            rings = max(2, int(round(math.sqrt(n_cyl * L / (2 * math.pi * r)))))
            around = max(3, int(math.ceil(n_cyl / rings)))
            z = -(np.arange(rings) + 0.5) / rings * L
            ang = np.arange(around) * 2 * math.pi / around
            zz, aa = np.meshgrid(z, ang, indexing='ij')
            aa = aa + (np.arange(rings)[:, None] % 2) * math.pi / around
            pc = np.stack([r * np.cos(aa), r * np.sin(aa), zz], -1).reshape(-1, 3)
            nc = np.stack([np.cos(aa), np.sin(aa), np.zeros_like(aa)], -1).reshape(-1, 3)
            us = _fib_sphere(n_cap)
            ps = us * r + np.where(us[:, 2:3] >= 0, 0.0, -L) * np.array([[0.0, 0.0, 1.0]])
            p = np.concatenate([pc, ps])
            nn = np.concatenate([nc, us])
        nn /= np.maximum(np.linalg.norm(nn, axis=1, keepdims=True), 1e-12)
        sig = 0.6 * spacing
        seg.append(np.full(len(p), si, np.int16))
        P.append(p)
        S.append(np.tile([sig, sig, max(0.1 * sig, 0.002)], (len(p), 1)))
        Q.append(SR.rot_to_quat_np(_frame_of(nn)))
    return (np.concatenate(seg), np.concatenate(P).astype(np.float32), np.concatenate(S).astype(np.float32),
            np.concatenate(Q).astype(np.float32))


# --------------------------------------------------------------------------- Anny's base mesh (CC0, on request)
ANNY_COMMIT = '8f35dc2df1c4062fd2fae86734f9705de50c0c2b'
ANNY_FILES = (('src/anny/data/mpfb2/3dobjs/base.obj', 'base.obj', 1749303,
               '8e761e6624b8f54536409135d1636da63b32486a90d4897f84e121d144f6fb4c'),
              ('src/anny/data/mpfb2/rigs/standard/weights.default.json', 'weights.default.json', 1633269,
               'c3df45e2f2165663b483b7fa3dfed6f1bd5e2dd48d8825308de1a8453a88b111'),
              ('LICENSE', 'LICENSE', 10875, '1d459b13c9650c763d6351ce21ad3b02433c80131195cf9d908de4fbc213cf27'))
ANNY_URL = 'https://raw.githubusercontent.com/naver/anny/{commit}/{path}'


def _anny_spec():
    from . import foundation_models as FM
    return FM.FoundationSpec(
        'anny_base', 'Anny / MakeHuman base mesh', 'body', 'mesh', 'naver/anny', ANNY_COMMIT,
        FM._f(*((n, sz, h) for _, n, sz, h in ANNY_FILES)), 'CC0 (the assets); Apache-2.0 (the repository)',
        'https://github.com/naver/anny', True,
        'base.obj: released CC0 by the MakeHuman copyright holders (its header); weights.default.json: CC0 (its '
        'licence field); the repository Apache-2.0 (read 2026-11-11)',
        'Brégier et al., Human Mesh Modeling for Anny Body, arXiv:2511.03589 (2025)', 0.0,
        note='the template body (13 380 vertices) and its skinning weights')


def anny_status(folder=None, verify=True) -> dict:
    from . import foundation_models as FM
    return FM.status(_anny_spec(), folder, verify=verify)


def download_anny(progress=None, cancel=None, folder=None, fetch=None, timeout=60.0):
    """Download Anny's base mesh and weights from the pinned commit (raw files; sizes and SHA-256 checked: a wrong
    file is not kept); returns the folder, None when cancelled."""
    import urllib.request
    from . import detection as DT
    from . import foundation_models as FM
    from .. import __version__
    d = FM.model_dir(_anny_spec(), folder)
    os.makedirs(d, exist_ok=True)
    stop = cancel or (lambda: False)
    total = sum(f[2] for f in ANNY_FILES)
    done = 0
    for path, name, size, sha in ANNY_FILES:
        final = os.path.join(d, name)
        if os.path.isfile(final) and os.path.getsize(final) == size and DT.file_sha256(final) == sha:
            done += size
            continue
        url = ANNY_URL.format(commit=ANNY_COMMIT, path=path)
        if fetch is not None:
            r = fetch(url, timeout)
        else:
            req = urllib.request.Request(url, headers={'User-Agent': f'Vantage/{__version__}'})
            r = urllib.request.urlopen(req, timeout=timeout, context=DT._ssl_context())
        h = hashlib.sha256()
        part = final + '.part'
        n = 0
        try:
            with r, open(part, 'wb') as fh:
                while True:
                    if stop():
                        raise Cancelled()
                    b = r.read(1 << 20)
                    if not b:
                        break
                    fh.write(b)
                    h.update(b)
                    n += len(b)
                    if n > size + (1 << 20):
                        raise ValueError(f'{name} is larger than the published {size:,} bytes.')
                    if progress is not None:
                        progress(min((done + n) / total, 1.0), f'{name} · {DT._mb(done + n)}')
            if n != size or h.hexdigest() != sha:
                raise ValueError(f'{name}: {n:,} bytes, SHA-256 {h.hexdigest()[:16]}…, not the pinned file: not kept.')
            os.replace(part, final)
        except Cancelled:
            DT._rm(part)
            return None
        except BaseException:
            DT._rm(part)
            raise
        done += size
    return d


def _anny_segment(bone: str) -> str | None:
    """The skeleton segment a MakeHuman default-rig bone belongs to (.L is the person's left)."""
    b = bone.lower()
    side = 'l_' if b.endswith('.l') else 'r_' if b.endswith('.r') else ''
    for keys, seg in ((('upperarm',), 'upperarm'), (('lowerarm', 'wrist', 'finger', 'metacarpal', 'thumb'), 'forearm'),
                      (('upperleg',), 'thigh'), (('lowerleg',), 'shank'), (('foot', 'toe'), 'foot')):
        if any(k in b for k in keys):
            return side + seg if side else None
    if any(k in b for k in ('head', 'neck', 'jaw', 'eye', 'tongue', 'special', 'levator', 'oris', 'orbicularis',
                            'temporalis', 'risorius', 'mandible', 'teeth', 'nose', 'cheek', 'brow', 'ear')):
        return 'head'
    if any(k in b for k in ('pelvis', 'root')):
        return 'pelvis'
    return 'torso'                                   # spine, clavicles, shoulders, breasts


def load_anny(folder=None) -> dict:
    """Anny's body (the 'body' group of base.obj, triangulated) and its weights grouped into the skeleton's segments:
    {'v' (V, 3) in MakeHuman's frame (x the person's left, y up, z forward), 'f' (F, 3), 'segw' (V, NSEG)}."""
    s = anny_status(folder, verify=False)
    if not s['verified']:
        raise FileNotFoundError('Anny\'s base mesh is not downloaded (Log replay ▸ Neural replay ▸ Body: Anny).')
    d = s['path']
    V, F, group = [], [], None
    with open(os.path.join(d, 'base.obj'), encoding='utf-8') as fh:
        for ln in fh:
            if ln.startswith('v '):
                V.append([float(x) for x in ln.split()[1:4]])
            elif ln.startswith('g '):
                group = ln.split()[1]
            elif ln.startswith('f ') and group == 'body':
                ix = [int(t.split('/')[0]) - 1 for t in ln.split()[1:]]
                for k in range(1, len(ix) - 1):
                    F.append([ix[0], ix[k], ix[k + 1]])
    V = np.asarray(V, float)
    F = np.asarray(F, np.int64)
    with open(os.path.join(d, 'weights.default.json'), encoding='utf-8') as fh:
        W = json.load(fh).get('weights') or {}
    segw = np.zeros((len(V), NSEG))
    for bone, rows in W.items():
        sg = _anny_segment(bone)
        if sg is None or not rows:
            continue
        a = np.asarray(rows, float)
        np.add.at(segw[:, SEGMENTS.index(sg)], a[:, 0].astype(int), a[:, 1])
    used = np.unique(F)
    remap = np.full(len(V), -1)
    remap[used] = np.arange(len(used))
    return {'v': V[used], 'f': remap[F], 'segw': segw[used]}


PARENT = {'torso': 'pelvis', 'head': 'torso', 'l_upperarm': 'torso', 'r_upperarm': 'torso',
          'l_forearm': 'l_upperarm', 'r_forearm': 'r_upperarm', 'l_thigh': 'pelvis', 'r_thigh': 'pelvis',
          'l_shank': 'l_thigh', 'r_shank': 'r_thigh', 'l_foot': 'l_shank', 'r_foot': 'r_shank'}
CHILD = {'l_upperarm': 'l_forearm', 'r_upperarm': 'r_forearm', 'l_thigh': 'l_shank', 'r_thigh': 'r_shank',
         'l_shank': 'l_foot', 'r_shank': 'r_foot'}


def _blend_joint(v, segw, a: str, b: str):
    """A template's joint between segments a and b: the weighted mean of the vertices both influence (min of their
    weights), None when they share none."""
    w = np.minimum(segw[:, SEGMENTS.index(a)], segw[:, SEGMENTS.index(b)])
    if w.sum() <= 1e-9:
        return None
    return (v * w[:, None]).sum(0) / w.sum()


def _rot_between(a, b) -> np.ndarray:
    """The smallest rotation taking direction a to direction b."""
    a = a / max(np.linalg.norm(a), 1e-12)
    b = b / max(np.linalg.norm(b), 1e-12)
    c = float(a @ b)
    if c > 1 - 1e-12:
        return np.eye(3)
    if c < -1 + 1e-12:
        ax = np.cross(a, [1.0, 0.0, 0.0] if abs(a[0]) < 0.9 else [0.0, 1.0, 0.0])
        ax /= np.linalg.norm(ax)
        return 2 * np.outer(ax, ax) - np.eye(3)
    k = np.cross(a, b)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + K + K @ K / (1 + c)


def to_rest_pose(v, segw, seg, stature: float, box_h: float):
    """A template body (v in the box frame, standing on the ground, any pose: SMPL's and Anny's T or A pose) moved
    segment by segment onto the skeleton's rest pose (arms hanging, legs straight): each limb rotated about its
    proximal joint (the blend region with its parent) so its axis (to the joint with its child, or to its centroid)
    hangs as the skeleton's does, then its joint placed on the skeleton's; the pelvis and torso by the hips' midpoint,
    the head by the neck. Returns the vertices in the box frame (origin the box's centre)."""
    import torch
    Rs, os_, _ = forward_kinematics(torch.zeros(1, NDOF, dtype=torch.float64), stature, box_h)
    o_rest = os_[0].numpy() + np.array([0.0, 0.0, box_h / 2])          # joints with the ground at z = 0
    out = v.copy()
    hips = [_blend_joint(v, segw, 'pelvis', s_) for s_ in ('l_thigh', 'r_thigh')]
    hips = [h for h in hips if h is not None]
    mid_t = np.mean(hips, 0) if hips else np.array([0.0, 0.0, WINTER['hip'] * stature])
    mid_r = 0.5 * (o_rest[SEGMENTS.index('l_thigh')] + o_rest[SEGMENTS.index('r_thigh')])
    for k, name in enumerate(SEGMENTS):
        m = seg == k
        if not m.any():
            continue
        if name in ('pelvis', 'torso'):
            out[m] = v[m] - mid_t + mid_r
            continue
        j_t = _blend_joint(v, segw, PARENT[name], name)
        if j_t is None:
            out[m] = v[m] - mid_t + mid_r
            continue
        j_r = o_rest[k]
        R = np.eye(3)
        if name in CHILD:
            c_t = _blend_joint(v, segw, name, CHILD[name])
            if c_t is not None:
                R = _rot_between(c_t - j_t, o_rest[SEGMENTS.index(CHILD[name])] - j_r)
        elif name.endswith('forearm'):
            R = _rot_between(v[m].mean(0) - j_t, np.array([0.0, 0.0, -1.0]))
        out[m] = (v[m] - j_t) @ R.T + j_r
    return out - np.array([0.0, 0.0, box_h / 2])


def template_gaussians(v_box, f, segw, stature: float, total: int = 1500):
    """A body template's surface as segment-bound Gaussians (as body_gaussians): v_box (V, 3) in the box's axes (x
    forward, y left, z up), scaled to the stature and standing on the ground of a box of that height; each vertex
    bound to the segment of its largest weight (segw (V, NSEG)), its local position in that segment's rest frame;
    discs tangent to the surface (the faces' vertex normals), about total of the vertices, one per cube of a side
    that spreads them evenly over the surface."""
    import torch
    v = np.asarray(v_box, float).copy()
    v[:, 2] -= v[:, 2].min()
    v *= float(stature) / max(float(v[:, 2].max()), 1e-6)
    v[:, :2] -= np.median(v[:, :2], axis=0)
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    vn = np.zeros_like(v)
    for k in range(3):
        np.add.at(vn, f[:, k], fn)
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
    segw = np.asarray(segw, float)
    seg = segw.argmax(1)
    H = float(stature)
    vr = to_rest_pose(v, segw, seg, H, H)                         # T/A pose -> the skeleton's rest, box frame
    # the normals follow each segment's rotation: from the rested faces
    fn = np.cross(vr[f[:, 1]] - vr[f[:, 0]], vr[f[:, 2]] - vr[f[:, 0]])
    vn = np.zeros_like(vr)
    for k in range(3):
        np.add.at(vn, f[:, k], fn)
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
    # an even spread over the surface: one vertex per cube of the side that gives about total of them
    area = 0.5 * float(np.linalg.norm(fn, axis=1).sum())
    cell = math.sqrt(area / max(total, 1))
    _, pick = np.unique(np.floor(vr / cell).astype(np.int64), axis=0, return_index=True)
    pick = np.sort(pick)
    v_box, vn, seg = vr[pick], vn[pick], seg[pick]
    Rs, os_, _ = forward_kinematics(torch.zeros(1, NDOF, dtype=torch.float64), H, H)
    Rs, os_ = Rs[0].numpy(), os_[0].numpy()
    loc = np.einsum('nji,nj->ni', Rs[seg], v_box - os_[seg])
    nl = np.einsum('nji,nj->ni', Rs[seg], vn)
    sig = 0.6 * cell                                               # σ along the surface: 0.6 of the spacing
    sc = np.tile([sig, sig, max(0.1 * sig, 0.002)], (len(v_box), 1))
    return (seg.astype(np.int16), loc.astype(np.float32), sc.astype(np.float32),
            SR.rot_to_quat_np(_frame_of(nl)).astype(np.float32))


def anny_gaussians(model: dict, stature: float, total: int = 1500):
    return template_gaussians(model['v'][:, [2, 0, 1]], model['f'], model['segw'], stature, total)


# --------------------------------------------------------------------------- SMPL (opt-in, the user's file)
SMPL_TO_SEG = {0: 'pelvis', 1: 'l_thigh', 2: 'r_thigh', 3: 'torso', 4: 'l_shank', 5: 'r_shank', 6: 'torso',
               7: 'l_foot', 8: 'r_foot', 9: 'torso', 10: 'l_foot', 11: 'r_foot', 12: 'head', 13: 'torso',
               14: 'torso', 15: 'head', 16: 'l_upperarm', 17: 'r_upperarm', 18: 'l_forearm', 19: 'r_forearm',
               20: 'l_forearm', 21: 'r_forearm', 22: 'l_forearm', 23: 'r_forearm'}
SMPL_KEYS = ('v_template', 'weights', 'f')
SMPL_LICENCE = ('SMPL (Loper, Mahmood, Romero, Pons-Moll and Black, SIGGRAPH Asia 2015) is licensed for '
                'non-commercial scientific research only (smpl.is.tue.mpg.de). Vantage never bundles or downloads it: '
                'register there, accept its licence, download the model (.npz) yourself and turn on "I accept".')


def smpl_usable(nst: NeuralSettings) -> str | None:
    """None when the SMPL body can be used (chosen, its licence accepted, a readable .npz), else why not."""
    if nst.body_model != 'smpl':
        return 'The capsule body is chosen.'
    if not nst.smpl_accept:
        return 'SMPL is non-commercial: accept its licence to use your copy. ' + SMPL_LICENCE
    p = os.path.expanduser(nst.smpl_path or '')
    if not p or not os.path.isfile(p):
        return 'Choose your SMPL model file (.npz).'
    try:
        load_smpl(p)
    except (OSError, ValueError, KeyError) as e:
        return f'The SMPL file cannot be read: {e}'
    return None


def load_smpl(path) -> dict:
    """An SMPL model's template ('v_template' (V, 3), 'weights' (V, 24), 'f' (F, 3)) from its .npz, without pickles,
    in SMPL's frame (y up, the body facing +z, x its left)."""
    with np.load(os.path.expanduser(str(path)), allow_pickle=False) as z:
        miss = [k for k in SMPL_KEYS if k not in z.files]
        if miss:
            raise KeyError(f'missing {", ".join(miss)}')
        v = np.asarray(z['v_template'], float)
        w = np.asarray(z['weights'], float)
        f = np.asarray(z['f'], np.int64)
    if v.ndim != 2 or v.shape[1] != 3 or w.shape[0] != v.shape[0] or w.shape[1] < 24 or f.ndim != 2:
        raise ValueError('not an SMPL template (v_template V × 3, weights V × 24, f F × 3)')
    return {'v': v, 'w': w[:, :24], 'f': f}


def smpl_gaussians(model: dict, stature: float, total: int = 1500):
    """The SMPL template as segment-bound Gaussians (template_gaussians): SMPL's frame (y up, facing +z, x its left)
    to the box's, each vertex to the segment its 24 joints' weights favour (SMPL_TO_SEG)."""
    segw = np.zeros((len(model['v']), NSEG))
    for j, name in SMPL_TO_SEG.items():
        segw[:, SEGMENTS.index(name)] += model['w'][:, j]
    return template_gaussians(model['v'][:, [2, 0, 1]], model['f'], segw, stature, total)


# --------------------------------------------------------------------------- the gait prior
def _bump(x, mu, sd):
    d = (x - mu + 0.5) % 1.0 - 0.5
    return np.exp(-0.5 * (d / sd) ** 2)


def gait_prior(times, centres, yaws, stature: float) -> np.ndarray:
    """(T, NDOF) joint angles of a walking cycle driven by a track (times (T,), box centres (T, 3), box yaws (T,)):
    the stride length 0.83·H·√(v / 1.4 m/s) (a step ≈ 0.415 of the stature at a normal 1.4 m/s; shorter when slower),
    the cycle's phase the distance walked over the stride. Sagittal angles shaped like Winter 2009's normal-gait
    curves (Ch. 4): hip flexion 10° ± 20° (most at heel strike), knee two bumps (≈ 15° at loading, ≈ 60° in swing at
    72 % of the cycle), the arms swinging against the legs (± 15°), elbows ≈ 20°; the right leg half a cycle on.
    Amplitudes scale with min(v / 1.4, 1.2); under 0.2 m/s the person stands. The box's heading is the walking
    direction (yaw 0). A shape, not measured data: the keypoint fit refines it."""
    t = np.asarray(times, float)
    T = len(t)
    th = np.zeros((T, NDOF))
    if T == 0:
        return th
    c = np.asarray(centres, float)
    if T > 1:
        dt = np.maximum(np.gradient(t), 1e-3)
        v = np.linalg.norm(np.gradient(c[:, :2], axis=0), axis=1) / dt
        v = np.convolve(np.pad(v, 2, mode='edge'), np.ones(5) / 5, mode='valid')
        dist = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(c[:, :2], axis=0), axis=1))])
    else:
        v = np.zeros(1)
        dist = np.zeros(1)
    H = float(stature)
    stride = 0.83 * H * np.sqrt(np.clip(v / 1.4, 0.25, 1.5))
    phase = np.concatenate([[0.0], np.cumsum(np.diff(dist) / np.maximum(0.5 * (stride[1:] + stride[:-1]), 1e-3))])
    amp = np.clip(v / 1.4, 0.0, 1.2) * (v >= 0.2)
    for side, off in (('l', 0.0), ('r', 0.5)):
        x = (phase + off) % 1.0
        hip = 10.0 + 20.0 * np.cos(2 * math.pi * x)
        knee = 5.0 + 12.0 * _bump(x, 0.15, 0.07) + 55.0 * _bump(x, 0.72, 0.11)
        sh = -15.0 * np.cos(2 * math.pi * x)
        th[:, D_[f'{side}_hip_flex']] = np.radians(amp * hip)
        th[:, D_[f'{side}_knee']] = np.radians(amp * knee + (1 - amp) * 3.0)
        th[:, D_[f'{side}_sh_flex']] = np.radians(amp * sh)
        th[:, D_[f'{side}_elbow']] = np.radians(15.0 + 10.0 * amp * (1 + np.cos(2 * math.pi * x)) / 2)
        th[:, D_[f'{side}_sh_abd']] = np.radians(6.0)
        th[:, D_[f'{side}_hip_abd']] = np.radians(2.0)
    th[:, D_['torso_flex']] = np.radians(3.0 * amp)
    th[:, D_['torso_twist']] = np.radians(5.0 * amp * np.cos(2 * math.pi * phase))
    return th


# --------------------------------------------------------------------------- walkers: observations and the fit
@dataclass
class Walker:
    """An articulated pedestrian: its track, actor slot, stature, and the joint angles at knot times (box frame)."""
    track: int
    slot: int
    stature: float
    box_h: float
    times: np.ndarray                   # (K,) knot times, s
    theta: np.ndarray                   # (K, NDOF)
    scale: float = 1.0
    method: str = 'gait'                # gait: from the track alone; keypoints: fitted to the recorded views
    fit: dict = field(default_factory=dict)

    def theta_at(self, t: float) -> np.ndarray:
        ts = self.times
        if len(ts) == 1:
            return self.theta[0].copy()
        j = int(np.clip(np.searchsorted(ts, t) - 1, 0, len(ts) - 2))
        f = float(np.clip((t - ts[j]) / max(ts[j + 1] - ts[j], 1e-9), 0.0, 1.0))
        return (1 - f) * self.theta[j] + f * self.theta[j + 1]


def pedestrian_tracks(drive: SP.Drive, scene: SP.Scene) -> list:
    """[(track id, actor slot)] of the pedestrians the scene rebuilt as moving actors."""
    out = []
    for a, i in enumerate(scene.actor_ids):
        tr = drive.tracks.get(i)
        if tr is not None and tr.cls == 'pedestrian' and tr.dynamic and len(tr.t):
            out.append((int(i), a))
    return out


def _box_at(tr: SP.Track, t: float):
    po = tr.pose(t)
    if po is None:
        k = int(np.argmin(np.abs(tr.t - t)))
        po = SP.quat_to_rot(tr.quat[k:k + 1])[0], tr.centre[k].copy()
    return po


def knot_times(tr: SP.Track, step: float = 0.1) -> np.ndarray:
    t0, t1 = float(tr.t.min()), float(tr.t.max())
    n = max(1, int(math.floor((t1 - t0) / step + 1e-9)) + 1)
    return t0 + np.arange(n) * step if n > 1 else np.array([t0])


def _yaw_of(R) -> float:
    return math.atan2(R[1, 0], R[0, 0])


def walker_prior(tr: SP.Track) -> tuple:
    """(knot times, gait-prior angles) of a pedestrian track."""
    tk = knot_times(tr)
    poses = [_box_at(tr, t) for t in tk]
    c = np.array([p[1] for p in poses])
    yaw = np.unwrap([_yaw_of(p[0]) for p in poses])
    return tk, gait_prior(tk, c, yaw, float(tr.size[2]))


def _crop_box(b, w, h, grow=1.25):
    """A walker's 2D box grown by grow about its centre, at least 3:4 (the pose model's input shape), in the image."""
    x0, y0, x1, y1 = (float(v) for v in b[:4])
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    bw, bh = max(x1 - x0, 1.0) * grow, max(y1 - y0, 1.0) * grow
    bw = max(bw, 0.75 * bh)
    return [max(cx - bw / 2, 0.0), max(cy - bh / 2, 0.0), min(cx + bw / 2, float(w)), min(cy + bh / 2, float(h))]


def walker_views(drive: SP.Drive, tracks: dict, st: NeuralSettings, min_px: int | None = None) -> list:
    """[(camera, frame, [(track id, 2D box)])] of the recorded frames showing the walkers large enough (the ground
    truth's visible box where the frame has objects, else the projected 3D box within 40 m)."""
    from . import groundtruth as GT
    min_px = st.min_walker_px if min_px is None else min_px
    out = []
    for cam in drive.cameras.values():
        W2O = np.asarray(cam.R, float).T
        for fr in cam.frames:
            boxes = []
            gfr = fr[2]
            if gfr is not None and gfr.get('objects') is not None:
                for o in gfr['objects']:
                    i = int(o.get('id') or -1)
                    if i in tracks and o.get('bbox') and float(o.get('pixels') or 0) >= min_px \
                            and float(o.get('occluded') or 0) < 0.7:
                        boxes.append((i, list(o['bbox'][:4])))
            else:
                Rw, pw = drive.traj.pose(fr[0])
                for i, tr in tracks.items():
                    po = tr.pose(fr[0])
                    if po is None:
                        continue
                    Rb, cb = po
                    cv = Rw.T @ (cb - pw)
                    if np.linalg.norm(cv - cam.t) > 40:
                        continue
                    E = GT.box_edges(cv, tr.size, Rw.T @ Rb, n=5)
                    u, v, _, ok = cam.model.project((E - cam.t) @ W2O.T)
                    if ok.sum() < len(ok) // 2:
                        continue
                    b = [u[ok].min(), v[ok].min(), u[ok].max(), v[ok].max()]
                    if (b[2] - b[0]) * (b[3] - b[1]) < min_px * 2 or b[2] < 0 or b[3] < 0 or \
                            b[0] > cam.model.width or b[1] > cam.model.height:
                        continue
                    boxes.append((i, b))
            if boxes:
                out.append((cam, fr, boxes))
    return out


def walker_keypoints(drive: SP.Drive, tracks: dict, st: NeuralSettings, pose_model, progress=None,
                     cancel=None) -> dict:
    """{track id: [(t, camera tag, keypoints (17, 3): u, v, score in the recorded image, box height px)]} from the
    pose model on every view of walker_views (at most max_pose_crops crops, spread evenly over the views)."""
    stop = cancel or (lambda: False)
    views = walker_views(drive, tracks, st)
    n_crops = sum(len(v[2]) for v in views)
    if n_crops > st.max_pose_crops:
        keep = np.unique(np.linspace(0, len(views) - 1, max(1, int(len(views) * st.max_pose_crops / n_crops)))
                         .round().astype(int))
        views = [views[int(k)] for k in keep]
    out = {}
    for j, (cam, fr, boxes) in enumerate(views):
        if stop():
            raise Cancelled()
        if j % 10 == 0:
            _prog(progress, j / max(len(views), 1), f'Keypoints · {cam.tag} · view {j + 1}/{len(views)}')
        try:
            img = (SP._load_rgb(fr[1]) * 255 + 0.5).astype(np.uint8)
        except OSError:
            continue
        h, w = img.shape[:2]
        crops = [_crop_box(b, w, h) for _, b in boxes]
        kp = pose_model.keypoints(img, crops)
        for (i, b), k in zip(boxes, kp):
            out.setdefault(i, []).append((float(fr[0]), cam.tag, k, float(b[3] - b[1])))
    return out


def _robust(r2):
    import torch
    return torch.log1p(r2)                                  # Cauchy's ρ: outliers (a swapped limb) cost little


def fit_walker(tr: SP.Track, obs: list, cameras: dict, traj: SP.Trajectory, st: NeuralSettings, dev=None,
               tk=None, prior=None) -> tuple:
    """Fit a walker's joint angles to its keypoints: (knot times, angles (K, NDOF), scale, stats). obs as
    walker_keypoints gives; cameras {tag: object with .model (CamModel at the recorded size), .R, .t}. Minimises the
    robust (Cauchy) reprojection error of the 17 keypoints through each camera's exact model, weighted by the
    model's scores (under min_kp_score: left out), the error scale kp_sigma_rel × the box height in the image, plus
    prior_weight × the squared distance to the gait prior (per joint in DOF_SIG units), smooth_weight × the squared
    second difference over the knots, a penalty outside the joint limits and the scale's prior (1 ± 5 %). The body's
    facing is tried both ways (the box's heading and its opposite); the lower cost is kept."""
    import torch
    dev = dev or torch.device('cpu')
    if tk is None or prior is None:
        tk, prior = walker_prior(tr)
    rows = [(t, tag, k, bh) for t, tag, k, bh in obs if tag in cameras]
    K = len(tk)
    if not rows:
        return tk, prior, 1.0, {'observations': 0}
    dt = torch.float32
    tj = np.array([r[0] for r in rows])
    j = np.clip(np.searchsorted(tk, tj) - 1, 0, max(K - 2, 0))
    f = np.clip((tj - tk[j]) / np.maximum(tk[np.minimum(j + 1, K - 1)] - tk[j], 1e-9), 0, 1) if K > 1 else \
        np.zeros(len(tj))
    jn = np.minimum(j + 1, K - 1)
    # each observation's box pose and camera (world -> optical) at its time
    Rb, cb, W2O, oc = [], [], [], []
    for t, tag, _, _ in rows:
        r_, c_ = _box_at(tr, t)
        Rb.append(r_)
        cb.append(c_)
        Rw, pw = traj.pose(t)
        cam = cameras[tag]
        W2O.append((Rw @ np.asarray(cam.R, float)).T)
        oc.append(Rw @ np.asarray(cam.t, float) + pw)
    T_ = lambda a: torch.tensor(np.asarray(a), dtype=dt, device=dev)  # noqa: E731
    Rb, cb, W2O, oc = T_(Rb), T_(cb), T_(W2O), T_(oc)
    kp = T_(np.stack([r[2] for r in rows]))
    sig = T_(np.maximum(np.array([r[3] for r in rows]) * st.kp_sigma_rel, 1.0))
    wt = kp[..., 2] * (kp[..., 2] >= st.min_kp_score)
    tags = [r[1] for r in rows]
    projs = {tag: cameras[tag].model.projector() for tag in set(tags)}
    groups = {tag: torch.tensor([i for i, x in enumerate(tags) if x == tag], device=dev) for tag in projs}
    P0 = T_(prior)
    sd = T_(DOF_SIG)
    lo, hi = T_(DOF_LO), T_(DOF_HI)
    jt, jnt, ft = torch.tensor(j, device=dev), torch.tensor(jn, device=dev), T_(f)[:, None]

    def reproj(theta, scale):
        th = theta[jt] * (1 - ft) + theta[jnt] * ft
        _, _, kb = forward_kinematics(th, float(tr.size[2]), float(tr.size[2]), scale.expand(len(rows)))
        Xw = torch.einsum('nij,nkj->nki', Rb, kb) + cb[:, None, :]
        Xc = torch.einsum('nij,nkj->nki', W2O, Xw - oc[:, None, :])
        uv = torch.zeros(len(rows), 17, 2, dtype=dt, device=dev)
        okk = torch.zeros(len(rows), 17, dtype=torch.bool, device=dev)
        for tag, gi in groups.items():
            u, v, ok = projs[tag](Xc[gi].reshape(-1, 3))
            uv = uv.index_copy(0, gi, torch.stack([u, v], -1).reshape(len(gi), 17, 2))
            okk = okk.index_copy(0, gi, ok.reshape(len(gi), 17))
        return uv, okk

    def run(yaw_off):
        delta = torch.zeros(K, NDOF, dtype=dt, device=dev, requires_grad=True)
        lsc = torch.zeros(1, dtype=dt, device=dev, requires_grad=True)
        base = P0.clone()
        base[:, D_['yaw']] += yaw_off
        opt = torch.optim.Adam([delta, lsc], lr=0.03)
        last = None
        for it in range(max(st.fit_steps, 0) + 1):
            theta = base + delta
            sc = torch.exp(lsc)
            uv, okk = reproj(theta, sc)
            r2 = (((uv - kp[..., :2]) / sig[:, None, None]) ** 2).sum(-1)
            w_ = wt * okk
            data = (w_ * _robust(r2)).sum() / w_.sum().clamp(min=1e-6)
            pri = ((delta / sd) ** 2).mean()
            sm = (((theta[2:] - 2 * theta[1:-1] + theta[:-2]) / sd) ** 2).mean() if K > 2 else theta.sum() * 0
            lim = (torch.relu(theta - hi) ** 2 + torch.relu(lo - theta) ** 2).mean() * 100.0
            scp = ((sc - 1.0) / 0.05) ** 2
            loss = data + st.prior_weight * pri + st.smooth_weight * sm + lim + scp.sum() * 0.1
            last = float(loss.detach())
            if it == st.fit_steps:
                break
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        return last, (base + delta).detach(), float(torch.exp(lsc).detach())

    with torch.no_grad():
        uv0, ok0 = reproj(P0, torch.ones(1, dtype=dt, device=dev))
    best = None
    for yo in (0.0, math.pi):
        res = run(yo)
        if best is None or res[0] < best[0]:
            best = res + (yo,)
    loss, theta, scale, yo = best
    with torch.no_grad():
        uv1, ok1 = reproj(theta, torch.tensor([scale], dtype=dt, device=dev))
    m = (wt > 0).cpu().numpy()

    def err(uv, ok):
        e = torch.linalg.norm(uv - kp[..., :2], dim=-1).cpu().numpy()
        mm = m & ok.cpu().numpy()
        return e[mm], (e / sig.cpu().numpy()[:, None])[mm]
    e0, r0 = err(uv0, ok0)
    e1, r1 = err(uv1, ok1)
    hpx = np.array([r[3] for r in rows])[:, None].repeat(17, 1)
    stats = {'observations': len(rows), 'keypoints_used': int(m.sum()), 'cameras': sorted(set(tags)),
             'median_px_prior': round(float(np.median(e0)), 2) if len(e0) else None,
             'median_px_fit': round(float(np.median(e1)), 2) if len(e1) else None,
             'within_10pc_prior': round(float(np.mean(e0 <= 0.1 * hpx[m & ok0.cpu().numpy()])), 4) if len(e0) else None,
             'within_10pc_fit': round(float(np.mean(e1 <= 0.1 * hpx[m & ok1.cpu().numpy()])), 4) if len(e1) else None,
             'scale': round(scale, 4), 'facing_flipped': bool(yo != 0.0), 'cost': round(loss, 4)}
    th = theta.cpu().numpy().astype(float)
    th[:, D_['yaw']] = np.unwrap(th[:, D_['yaw']])
    return tk, th, scale, stats


def posed_body(w: Walker, t: float, seg, local, quats=None):
    """A walker's Gaussians at time t in its box frame (numpy): (centres (n, 3), rotations (n, 3, 3) or None)."""
    import torch
    th = torch.tensor(w.theta_at(t)[None], dtype=torch.float64)
    Rs, os_, _ = forward_kinematics(th, w.stature, w.box_h, torch.tensor([w.scale], dtype=torch.float64))
    Rs, os_ = Rs[0].numpy(), os_[0].numpy()
    P = np.einsum('nij,nj->ni', Rs[seg], local) * 1.0 + os_[seg]
    if quats is None:
        return P, None
    return P, Rs[seg] @ SR.quat_to_rot_np(quats)


def colour_walker(w: Walker, tr: SP.Track, seg, local, quats, drive: SP.Drive, st_splat, max_views: int = 120):
    """Colours (n, 3) of a walker's Gaussians from the recorded frames that see it: each Gaussian posed at the
    frame's time, kept where it faces the camera and lands on the walker's own pixels (the ground truth's instance
    image; without one, inside its projected box), the image there averaged (weight: the cosine of facing). Never
    seen: the nearest seen Gaussian of the same segment (else of any), else mid grey. Returns (colours, share seen)."""
    import cv2
    from scipy.spatial import cKDTree
    from . import groundtruth as GT
    n = len(local)
    acc = np.zeros((n, 3))
    wsum = np.zeros(n)
    views = walker_views(drive, {w.track: tr}, NeuralSettings(), min_px=30)
    if len(views) > max_views:
        views = [views[int(k)] for k in np.linspace(0, len(views) - 1, max_views).round().astype(int)]
    _, R0 = posed_body(w, float(tr.t[0]), seg, local, quats)
    for cam, fr, boxes in views:
        t = float(fr[0])
        sc = min(st_splat.train_scale * 2, 1.0)
        W_, H_ = SP._train_size(cam, sc)
        try:
            img = SP._load_rgb(fr[1], (W_, H_))
        except OSError:
            continue
        m = cam.model.scaled(W_, H_)
        Pb, Rg = posed_body(w, t, seg, local, quats)
        Rb, cb = _box_at(tr, t)
        Pw = Pb @ Rb.T + cb
        nw = (Rb @ Rg)[:, :, 2]
        Rw, pw = drive.traj.pose(t)
        o = Rw @ cam.t + pw
        O2W = Rw @ cam.R
        Q = (Pw - o) @ O2W
        u, v, _, ok = m.project(Q)
        view = o - Pw
        view /= np.maximum(np.linalg.norm(view, axis=1, keepdims=True), 1e-9)
        cosf = (nw * view).sum(1)
        ok &= (u >= 0) & (u < W_) & (v >= 0) & (v < H_) & (cosf > 0.15)
        inst = None
        if fr[2] is not None and fr[2].get('instance') and drive.gt is not None:
            try:
                inst = cv2.resize(GT.load_instance(drive.gt, fr[2]).astype(np.int32), (W_, H_),
                                  interpolation=cv2.INTER_NEAREST)
            except (OSError, ValueError, cv2.error):
                inst = None
        ix = np.flatnonzero(ok)
        if not len(ix):
            continue
        if inst is not None:
            ix = ix[inst[v[ix].astype(int), u[ix].astype(int)] == w.track]
        else:
            b = [x * W_ / cam.model.width if k % 2 == 0 else x * H_ / cam.model.height
                 for k, x in enumerate(next(bb for i, bb in boxes if i == w.track))]
            ix = ix[(u[ix] >= b[0]) & (u[ix] <= b[2]) & (v[ix] >= b[1]) & (v[ix] <= b[3])]
        if not len(ix):
            continue
        col = cv2.remap(img, u[ix].astype(np.float32)[:, None] - 0.5, v[ix].astype(np.float32)[:, None] - 0.5,
                        cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)[:, 0, :]
        np.add.at(acc, ix, col * cosf[ix, None])
        np.add.at(wsum, ix, cosf[ix])
    seen = wsum > 0
    cols = np.full((n, 3), 0.45)
    cols[seen] = acc[seen] / wsum[seen, None]
    if seen.any() and (~seen).any():
        for s_ in np.unique(seg):
            ms = seg == s_
            src = ms & seen
            tgt = ms & ~seen
            if not tgt.any():
                continue
            pool = src if src.any() else seen
            tree = cKDTree(local[pool] if src.any() else np.zeros((1, 3)))
            if src.any():
                _, nn = tree.query(local[tgt], k=1)
                cols[tgt] = cols[pool][nn]
            else:
                cols[tgt] = cols[seen].mean(0)
    return np.clip(cols, 0, 1).astype(np.float32), float(seen.mean()) if n else 0.0


def build_walkers(scene: SP.Scene, drive: SP.Drive, sst, nst: NeuralSettings, keypoints: dict | None = None,
                  dev=None, progress=None, cancel=None) -> tuple:
    """Articulated walkers for the scene's moving pedestrians: (walkers, arrays to append {'means' (segment-local),
    'scales', 'quats', 'opacity', 'colours', 'actor', 'obj', 'wid', 'wseg'}, the actor slots they replace, info)."""
    stop = cancel or (lambda: False)
    peds = pedestrian_tracks(drive, scene)
    walkers, parts, info = [], [], {'count': 0, 'method': 'gait prior from the track', 'fits': [],
                                    'body': 'capsules (Vantage\'s own)'}
    smpl = anny = None
    if nst.body_model == 'anny':
        try:
            anny = load_anny()
            info['body'] = 'Anny\'s MakeHuman base mesh (CC0), scaled to each walker\'s stature'
        except (FileNotFoundError, OSError, ValueError) as e:
            info['body_note'] = f'{e} The capsule body is used.'
    if nst.body_model == 'smpl':
        why = smpl_usable(nst)
        if why is None:
            smpl = load_smpl(nst.smpl_path)
            info['body'] = 'SMPL template (the user\'s file, non-commercial licence accepted)'
        else:
            info['body_note'] = why
    for k, (i, slot) in enumerate(peds):
        if stop():
            raise Cancelled()
        _prog(progress, k / max(len(peds), 1), f'Walker {k + 1}/{len(peds)}')
        tr = drive.tracks[i]
        tk, prior = walker_prior(tr)
        obs = (keypoints or {}).get(i) or []
        if obs:
            tk, th, scale, stats = fit_walker(tr, obs, drive.cameras, drive.traj, nst, dev, tk, prior)
            method = 'keypoints'
        else:
            th, scale, stats, method = prior, 1.0, {'observations': 0}, 'gait'
        w = Walker(i, slot, float(tr.size[2]), float(tr.size[2]), tk, th, scale, method, stats)
        if smpl is not None:
            seg, loc, sc, qs = smpl_gaussians(smpl, w.stature * w.scale, nst.walker_gaussians)
        elif anny is not None:
            seg, loc, sc, qs = anny_gaussians(anny, w.stature * w.scale, nst.walker_gaussians)
        else:
            seg, loc, sc, qs = body_gaussians(w.stature * w.scale, nst.walker_gaussians)
        # the surface was laid out for the scaled stature; the skeleton applies the scale to the joints
        col, seen = colour_walker(w, tr, seg, loc, qs, drive, sst)
        stats['coloured_share'] = round(seen, 4)
        stats.update({'track': i, 'method': method, 'gaussians': int(len(seg))})
        info['fits'].append(stats)
        walkers.append(w)
        n = len(seg)
        parts.append({'means': loc, 'scales': sc, 'quats': qs, 'opacity': np.full(n, 0.95, np.float32),
                      'colours': col, 'actor': np.full(n, slot, np.int32), 'obj': np.full(n, i, np.int32),
                      'wid': np.full(n, len(walkers) - 1, np.int16), 'wseg': seg.astype(np.int16)})
    info['count'] = len(walkers)
    if any(w.method == 'keypoints' for w in walkers):
        info['method'] = 'fitted to 2D keypoints of the recorded views (ViTPose-B) with a gait prior'
    return walkers, parts, [w.slot for w in walkers], info


# =========================================================================== traffic lights
@dataclass
class Light:
    """An emissive traffic-light head: its box (local frame), the face it shows, colours, and its state over time."""
    track: int
    group: str
    centre: np.ndarray
    R: np.ndarray
    size: tuple
    normal: np.ndarray                  # the face's outward normal (world, horizontal)
    housing: np.ndarray                 # RGB
    lit: dict                           # state -> RGB of the lit lamp as recorded (or STATE_RGB)
    times: np.ndarray = field(default_factory=lambda: np.zeros(0))
    states: list = field(default_factory=list)
    source: str = 'none'                # ground truth | images | none
    facing: str = 'path'                # images: lamps seen on that side; path: the side facing the driven path

    def state_at(self, t: float) -> str:
        if not len(self.times):
            return ''
        k = int(np.argmin(np.abs(self.times - t)))
        return self.states[k]

    def to_dict(self) -> dict:
        return {'track': self.track, 'group': self.group, 'centre': [round(float(x), 3) for x in self.centre],
                'size': [round(float(x), 3) for x in self.size], 'normal': [round(float(x), 3) for x in self.normal],
                'housing': [round(float(x), 3) for x in self.housing], 'source': self.source, 'facing': self.facing,
                'observations': len(self.times),
                'states': sorted(set(s for s in self.states if s))}


def _face_geometry(Rb, size, n_axis: int, sign: float):
    """(normal, face centre offset along it, face width, up) of one vertical face of a head's box."""
    ax = Rb[:, n_axis] * sign
    nrm = np.array([ax[0], ax[1], 0.0])
    nrm /= max(np.linalg.norm(nrm), 1e-9)
    depth = float(size[n_axis])
    width = float(size[1 - n_axis])
    return nrm, depth / 2, width, np.array([0.0, 0.0, 1.0])


def lamp_centres(L: Light) -> np.ndarray:
    """(3, 3) world centres of the red, amber and green lamps on the face (top to bottom, a third of the head's
    height apart), a centimetre proud of it."""
    h = float(L.size[2])
    off = _face_offset(L)
    c = L.centre + L.normal * (off + 0.01)
    return np.stack([c + np.array([0, 0, h / 3.2]), c, c - np.array([0, 0, h / 3.2])])


def _face_offset(L: Light) -> float:
    ex = L.R[:, 0]
    n_axis = 0 if abs(float(ex[:2] @ L.normal[:2])) >= 0.7 else 1
    return float(L.size[n_axis]) / 2


def _lamp_score(img, u, v, r):
    """(brightness·(½ + saturation) in a lamp's disc, its mean RGB): how lit a lamp looks."""
    h, w = img.shape[:2]
    rr = max(1, int(round(r)))
    x0, x1 = int(max(u - rr, 0)), int(min(u + rr + 1, w))
    y0, y1 = int(max(v - rr, 0)), int(min(v + rr + 1, h))
    if x1 <= x0 or y1 <= y0:
        return None, None
    patch = img[y0:y1, x0:x1].reshape(-1, 3)
    top = patch[np.argsort(patch.max(1))[-max(1, len(patch) // 3):]]      # the brightest third: the lamp itself
    rgb = top.mean(0)
    mx, mn = float(rgb.max()), float(rgb.min())
    sat = (mx - mn) / max(mx, 1e-6)
    return mx * (0.5 + sat), rgb


def _light_tracks(drive: SP.Drive) -> dict:
    return {i: tr for i, tr in drive.tracks.items() if tr.cls == 'traffic_light' and len(tr.t)}


def _actor_groups(drive: SP.Drive) -> tuple:
    """({object id: actor id} of the ground truth's objects, {object id: [(t, state)]} where an object states one)."""
    grp, states = {}, {}
    for cam in drive.cameras.values():
        for t, _, gfr in cam.frames:
            for o in (gfr or {}).get('objects') or ():
                i = o.get('id')
                if i is None:
                    continue
                if o.get('actor_id') is not None:
                    grp.setdefault(int(i), str(o['actor_id']))
                st_ = str(o.get('state') or '').lower()
                st_ = {'yellow': 'amber', 'orange': 'amber'}.get(st_, st_)
                if st_ in STATES or st_ == 'off':
                    states.setdefault(int(i), []).append((float(t), st_ if st_ != 'off' else ''))
    return grp, states


DETECTED_ID0 = 1_000_000                 # ids of heads found by the detector (not in the drive's tracks: never scored)


def lights_from_detections(drive: SP.Drive, scene: SP.Scene, detector, nst: NeuralSettings, step: int = 3,
                           min_score: float = 0.4, min_px: float = 8.0, merge_m: float = 1.5, progress=None,
                           cancel=None) -> dict:
    """Traffic-light heads of a drive without boxes (light_source 'detector'): {id: Track} of static heads. The
    detector (a callable BGR image -> core.detection's dict) runs on every step-th frame of each camera; each traffic
    light it finds (score >= min_score, at least min_px tall) is lifted to 3D along the ray through its box's centre
    at the range of the nearest static Gaussians that project inside the box (their 20th percentile), sized by the
    box's angular height x that range; detections within merge_m of a head are merged into it, and a head needs two
    detections. A head's box faces the mean camera position that saw it; its state is read from the images as a
    box's would be. Ids from DETECTED_ID0 (not the drive's objects: not scored)."""
    import cv2
    from . import detection as DT
    from .lens import theta_of_r
    stop = cancel or (lambda: False)
    tl = DT.CLASSES.index('traffic_light')
    Ps = scene.means[scene.actor < 0].astype(float)
    heads = []                                           # [sum xyz, n, sum height, [camera centres]]
    frames = [(cam, fr) for cam in drive.cameras.values() for k, fr in enumerate(cam.frames) if k % step == 0]
    for j, (cam, fr) in enumerate(frames):
        if stop():
            raise Cancelled()
        if j % 20 == 0:
            _prog(progress, j / max(len(frames), 1), f'Traffic lights · detector · {cam.tag}')
        img = cv2.imread(fr[1], cv2.IMREAD_COLOR)
        if img is None:
            continue
        dets = detector(img)
        cls = np.asarray(dets.get('cls', []), int).reshape(-1)
        sc_ = np.asarray(dets.get('scores', []), float).reshape(-1)
        boxes = np.asarray(dets.get('boxes', []), float).reshape(-1, 4)
        keep = np.flatnonzero((cls == tl) & (sc_ >= min_score) & (boxes[:, 3] - boxes[:, 1] >= min_px))
        if not len(keep):
            continue
        Rw, pw = drive.traj.pose(fr[0])
        o = Rw @ cam.t + pw
        O2W = Rw @ cam.R
        Q = (Ps - o) @ O2W
        u, v, _, ok = cam.model.project(Q)
        rng = np.linalg.norm(Q, axis=1)
        for b in boxes[keep]:
            m = ok & (u >= b[0]) & (u <= b[2]) & (v >= b[1]) & (v <= b[3]) & (rng > 1.0)
            if m.sum() < 2:
                continue
            r = float(np.percentile(rng[m], 20))
            uu = np.full(3, 0.5 * (b[0] + b[2]))
            vv = np.array([b[1], 0.5 * (b[1] + b[3]), b[3]])
            x = (uu - cam.model.cx) / cam.model.fx
            y = (vv - cam.model.cy) / cam.model.fy
            rr = np.hypot(x, y)
            th = theta_of_r(rr, 1.0, cam.model.projection, cam.model.kb)
            s_ = np.where(rr > 1e-12, np.sin(th) / np.where(rr > 1e-12, rr, 1.0), 0.0)
            dirs = np.stack([x * s_, y * s_, np.cos(th)], 1)
            dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
            X = o + O2W @ (dirs[1] * r)
            hgt = float(np.arccos(np.clip(dirs[0] @ dirs[2], -1, 1))) * r
            for h in heads:
                if np.linalg.norm(h[0] / h[1] - X) <= merge_m:
                    h[0] = h[0] + X
                    h[1] += 1
                    h[2] += hgt
                    h[3].append(o)
                    break
            else:
                heads.append([X.copy(), 1, hgt, [o]])
    out = {}
    for k, h in enumerate(sorted((h for h in heads if h[1] >= 2), key=lambda h: -h[1])):
        c = h[0] / h[1]
        hh = float(np.clip(h[2] / h[1], 0.4, 1.6))
        look = np.mean(h[3], 0) - c
        i = DETECTED_ID0 + k
        out[i] = SP.Track(i, 'traffic_light', 'static.manmade', False, (0.35, 0.40, hh), np.array([0.0]), c[None],
                          SP._yaw_quat(math.atan2(look[1], look[0]))[None])
    return out


def build_lights(scene: SP.Scene, drive: SP.Drive, nst: NeuralSettings, progress=None, cancel=None,
                 tracks: dict | None = None) -> tuple:
    """Emissive heads for the drive's traffic lights (module docstring, 4): (lights, info). Facing and state from the
    recorded images unless the ground truth states them. tracks: the heads (default: the drive's traffic-light
    tracks; lights_from_detections' for a drive without them)."""
    stop = cancel or (lambda: False)
    tracks = _light_tracks(drive) if tracks is None else tracks
    grp, gt_states = _actor_groups(drive)
    lights = []
    path = drive.traj.pos
    obs_by_group = {}
    for k, (i, tr) in enumerate(sorted(tracks.items())):
        if stop():
            raise Cancelled()
        _prog(progress, k / max(len(tracks), 1), f'Traffic light {k + 1}/{len(tracks)}')
        Rb, cb = SP.quat_to_rot(tr.quat[:1])[0], tr.centre[0].copy()
        size = tuple(float(x) for x in tr.size)
        m = scene.obj == i
        housing = scene.colours[m].mean(0) if m.any() else np.array([0.25, 0.25, 0.22])
        cands = [(ax, sg) for ax in (0, 1) for sg in (1.0, -1.0)]
        evid = {c: [] for c in cands}
        samples = []
        for cam in drive.cameras.values():
            W2O = np.asarray(cam.R, float).T
            for t, path_img, gfr in cam.frames:
                objs = {int(o['id']): o for o in (gfr or {}).get('objects') or () if o.get('id') is not None}
                if i < DETECTED_ID0 and gfr is not None and objs and \
                        (i not in objs or float(objs[i].get('pixels') or 0) < nst.min_light_px):
                    continue
                Rw, pw = drive.traj.pose(t)
                o_w = Rw @ cam.t + pw
                if np.linalg.norm(o_w - cb) > 80:
                    continue
                for c in cands:
                    nrm, off, width, up = _face_geometry(Rb, size, *c)
                    fc = cb + nrm * off
                    to_cam = o_w - fc
                    if to_cam @ nrm < 0.3 * np.linalg.norm(to_cam):
                        continue
                    L_ = Light(i, '', cb, Rb, size, nrm, housing, {})
                    lc = lamp_centres(L_)
                    Q = (np.vstack([lc, fc + up * size[2] / 2, fc - up * size[2] / 2]) - pw) @ Rw
                    u, v, _, ok = cam.model.project((Q - cam.t) @ W2O.T)
                    if not ok.all() or abs(v[3] - v[4]) < 9:
                        continue
                    samples.append((c, cam, t, path_img, u[:3], v[:3], abs(v[3] - v[4]) / 3.2 * 0.35 * 2))
        # read the lamps where the face is seen large enough (each image read once per light)
        cache = {}
        for c, cam, t, pth, u, v, rpx in samples:
            img = cache.get(pth)
            if img is None:
                try:
                    img = SP._load_rgb(pth)
                except OSError:
                    continue
                if len(cache) > 8:
                    cache.clear()
                cache[pth] = img
            sc = [_lamp_score(img, u[j], v[j], rpx) for j in range(3)]
            if any(s_[0] is None for s_ in sc):
                continue
            vals = np.array([s_[0] for s_ in sc])
            o_ = np.argsort(vals)[::-1]
            margin = float(vals[o_[0]] - vals[o_[1]])
            evid[c].append(margin)
            if margin >= 0.08:
                samples_ok = (t, STATES[int(o_[0])], sc[int(o_[0])][1], margin)
                evid.setdefault(('obs', c), []).append(samples_ok)
        best, facing = None, 'path'
        scores = {c: float(np.mean(v)) for c, v in evid.items() if isinstance(c[0], int) and v}
        if scores and max(scores.values()) >= 0.05:
            best = max(scores, key=scores.get)
            facing = 'images'
        if best is None:
            d = path[np.argmin(np.linalg.norm(path[:, :2] - cb[:2], axis=1))] - cb
            best = max(cands, key=lambda c: float(_face_geometry(Rb, size, *c)[0] @ d))
        nrm = _face_geometry(Rb, size, *best)[0]
        g = grp.get(i, f'track:{i}')
        L = Light(i, g, cb, Rb, size, nrm, np.clip(housing, 0, 1), {}, facing=facing)
        if i in gt_states:
            tt = np.array([x[0] for x in gt_states[i]])
            L.times, L.states, L.source = tt, [x[1] for x in gt_states[i]], 'ground truth'
        else:
            for t, state, rgb, mg in evid.get(('obs', best), []):
                obs_by_group.setdefault(g, []).append((t, state, rgb, mg))
        lights.append(L)
    for L in lights:
        if L.source == 'ground truth':
            continue
        ob = sorted(obs_by_group.get(L.group, []), key=lambda x: x[0])
        if ob:
            L.times = np.array([x[0] for x in ob])
            L.states = [x[1] for x in ob]
            L.source = 'images'
            for s_ in STATES:
                cs = [x[2] for x in ob if x[1] == s_]
                if cs:
                    L.lit[s_] = np.clip(np.mean(cs, 0) * 1.15, 0, 1)
    info = {'count': len(lights), 'with_state': sum(1 for L in lights if len(L.times)),
            'state_source': ', '.join(sorted(set(L.source for L in lights if L.source != 'none'))) or 'none',
            'faced_by_images': sum(1 for L in lights if L.facing == 'images'),
            'groups': len(set(L.group for L in lights))}
    return lights, info


def light_gaussians(lights: list) -> dict:
    """The heads as Gaussians: per head a housing panel on its face and its back (housing colour) and three lamps of
    seven discs each. {'means', 'scales', 'quats', 'opacity', 'colours', 'actor', 'obj', 'lamp' (−1 or 0–2),
    'light' (head index or −1)} in the local frame."""
    out = {k: [] for k in ('means', 'scales', 'quats', 'opacity', 'colours', 'actor', 'obj', 'lamp', 'light')}
    for li, L in enumerate(lights):
        nrm = L.normal
        up = np.array([0.0, 0.0, 1.0])
        side = np.cross(up, nrm)
        off = _face_offset(L)
        width = float(L.size[1] if abs(float(L.R[:, 0][:2] @ nrm[:2])) >= 0.7 else L.size[0])
        h = float(L.size[2])
        nx, nz = 5, 14
        sp = max(width / nx, h / nz)
        xs = (np.arange(nx) + 0.5) / nx - 0.5
        zs = (np.arange(nz) + 0.5) / nz - 0.5
        X, Z = np.meshgrid(xs * width, zs * h)
        for sgn in (1.0, -1.0):
            n_ = nrm * sgn
            P = L.centre + n_ * off + X.reshape(-1, 1) * side + Z.reshape(-1, 1) * up
            k = len(P)
            Rm = np.stack([side, up, n_], 1)
            out['means'].append(P)
            out['scales'].append(np.tile([0.6 * sp, 0.6 * sp, 0.006], (k, 1)))
            out['quats'].append(np.repeat(SR.rot_to_quat_np(Rm[None]), k, 0))
            out['colours'].append(np.repeat(L.housing[None], k, 0))
            out['lamp'].append(np.full(k, -1))
        lc = lamp_centres(L)
        r = min(width * 0.3, h / 8)
        hexo = np.array([[0, 0]] + [[math.cos(a), math.sin(a)] for a in np.arange(6) * math.pi / 3]) * r * 0.55
        Rm = np.stack([side, up, nrm], 1)
        for j in range(3):
            P = lc[j] + hexo[:, :1] * side + hexo[:, 1:] * up
            out['means'].append(P)
            out['scales'].append(np.tile([0.5 * r, 0.5 * r, 0.004], (7, 1)))
            out['quats'].append(np.repeat(SR.rot_to_quat_np(Rm[None]), 7, 0))
            out['colours'].append(np.full((7, 3), 0.1))
            out['lamp'].append(np.full(7, j))
        n = sum(len(m) for m in out['means'][-5:])
        out['opacity'].append(np.full(n, 0.98))
        out['actor'].append(np.full(n, -1))
        out['obj'].append(np.full(n, L.track))
        out['light'].append(np.full(n, li))
    if not lights:
        return {k: np.zeros((0, 3) if k in ('means', 'scales', 'colours') else (0, 4) if k == 'quats' else 0)
                for k in out}
    res = {k: np.concatenate(v) for k, v in out.items()}
    return {'means': res['means'].astype(np.float32), 'scales': res['scales'].astype(np.float32),
            'quats': res['quats'].astype(np.float32), 'opacity': res['opacity'].astype(np.float32),
            'colours': res['colours'].astype(np.float32), 'actor': res['actor'].astype(np.int32),
            'obj': res['obj'].astype(np.int32), 'lamp': res['lamp'].astype(np.int8),
            'light': res['light'].astype(np.int32)}


def lamp_colours(L: Light, t: float) -> np.ndarray:
    """(3, 3): the three lamps' colours at time t: the lit one as recorded, the others dark (a tenth of theirs)."""
    s = L.state_at(t)
    out = np.zeros((3, 3))
    for j, name in enumerate(STATES):
        lit = np.asarray(L.lit.get(name, STATE_RGB[name]), float)
        out[j] = lit if s == name else 0.1 * lit + 0.03
    return out


# =========================================================================== the lidar opacity
def recorded_grid(L: SP.DriveLidar, spec: SP.LidarSpec, sw) -> tuple:
    """A recorded sweep on the lidar's beam grid (as core.splat_reproject.validate maps it): (returned (rings,
    columns) bool, range (rings, columns) m, NaN where none)."""
    a = SP.load_sweep(sw)
    R, C = len(spec.elevations), len(spec.az)
    rec = np.zeros((R, C), bool)
    rr_g = np.full((R, C), np.nan)
    if not len(a) or (a[:, 3] < 0).all():
        return rec, rr_g
    ring = a[:, 3].astype(int)
    ok = (ring >= 0) & (ring < R)
    az = np.degrees(np.arctan2(a[:, 1], a[:, 0]))
    col = np.round(((az - spec.az[0]) * (1 if L.sense > 0 else -1)) % 360.0 / (360.0 / C)).astype(int) % C
    rr = np.linalg.norm(a[:, :3], axis=1)
    rec[ring[ok], col[ok]] = True
    rr_g[ring[ok], col[ok]] = rr[ok]
    return rec, rr_g


def fit_lidar_opacity(ts, drive: SP.Drive, sst, nst: NeuralSettings, progress=None, cancel=None) -> tuple:
    """The lidar opacity of every Gaussian, fitted to the training sweeps (module docstring, 2): (opacities (N,),
    stats). Each step renders one sector of one training sweep with gradients and minimises the binary
    cross-entropy of each beam's accumulated opacity against whether the recorded beam returned, plus 0.1 × the
    Huber range error (relative to 2 % of the range, at least 5 cm) where both return; Adam on the logits only
    (the geometry and the camera's opacity are fixed). Starts from the camera's opacity."""
    import torch
    stop = cancel or (lambda: False)
    rng = np.random.default_rng(nst.seed)
    logit = ts.op_logit.detach().clone().requires_grad_(True)
    jobs = []
    for L in drive.lidars.values():
        spec = SP.recorded_lidar_spec(L)
        for k, sw in enumerate(L.sweeps):
            if SP.held_out(k, sst.holdout_every) or k % sst.sweep_step:
                continue
            jobs.append((L, spec, sw))
    if not jobs or nst.lidar_steps <= 0:
        return torch.sigmoid(logit).detach().cpu().numpy(), {'steps': 0}
    grids = {}
    opt = torch.optim.Adam([logit], lr=nst.lidar_lr)
    losses = []
    t0 = time.perf_counter()
    nsec = SP._lidar_grid(jobs[0][1])[-1]
    for it in range(nst.lidar_steps):
        if stop():
            raise Cancelled()
        j = int(rng.integers(len(jobs)))
        L, spec, sw = jobs[j]
        if j not in grids:
            grids[j] = recorded_grid(L, spec, sw)
        rec, rr = grids[j]
        k = int(rng.integers(nsec))
        out = SP.lidar_beams(ts, drive.traj, spec, sw[0] - L.period_s, sst, sectors=[k], grad=True,
                             opacity=torch.sigmoid(logit))
        if not out:
            continue
        cols, acc, r_exp, _, _ = out[0]
        tgt = torch.tensor(rec[:, cols], dtype=acc.dtype, device=acc.device)
        a = acc.clamp(1e-4, 1 - 1e-4)
        bce = -(tgt * torch.log(a) + (1 - tgt) * torch.log(1 - a)).mean()
        rref = torch.tensor(np.nan_to_num(rr[:, cols], nan=0.0), dtype=acc.dtype, device=acc.device)
        both = (tgt > 0) & (acc > 0.3)
        if bool(both.any()):
            sc = torch.clamp(0.02 * rref[both], min=0.05)
            rl = torch.nn.functional.huber_loss(r_exp[both] / sc, rref[both] / sc, delta=1.0)
        else:
            rl = acc.sum() * 0
        loss = bce + 0.1 * rl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if it % 10 == 0:
            losses.append(round(float(loss.detach()), 5))
            _prog(progress, it / nst.lidar_steps, f'Lidar opacity · step {it} · loss {float(loss.detach()):.4f}')
    op = torch.sigmoid(logit).detach().cpu().numpy().astype(np.float32)
    cam = torch.sigmoid(ts.op_logit).detach().cpu().numpy()
    return op, {'steps': nst.lidar_steps, 'seconds': round(time.perf_counter() - t0, 2),
                'loss_first': losses[0] if losses else None, 'loss_last': losses[-1] if losses else None,
                'mean_abs_change': round(float(np.mean(np.abs(op - cam))), 4),
                'share_changed_0_2': round(float(np.mean(np.abs(op - cam) > 0.2)), 4),
                'see_through_for_camera': int(np.sum((cam < 0.2) & (op > 0.5))),
                'see_through_for_lidar': int(np.sum((cam > 0.5) & (op < 0.2)))}


# =========================================================================== the scene's extension
@dataclass
class NeuralExt:
    """What neural replay adds to a scene (aligned with its Gaussians): walkers, emissive lights, lidar opacities.
    core.splat_reproject.TorchScene attaches it (attach) and then poses, colours and casts beams through it."""
    walkers: list
    wid: np.ndarray                     # (N,) int16: walker index, −1 none (its centre and rotation are segment-local)
    wseg: np.ndarray                    # (N,) int16: the segment it is bound to
    lights: list
    lamp: np.ndarray                    # (N,) int8: lamp 0–2 (red, amber, green), −1 none
    light: np.ndarray                   # (N,) int32: head index, −1 none
    opacity_lidar: np.ndarray | None    # (N,) or None (the camera's)
    info: dict = field(default_factory=dict)
    relight: dict | None = None         # core.relight settings applied when rendering (not cached)

    def attach(self, ts):
        return _Bound(self, ts)

    def emissive(self) -> np.ndarray:
        return self.lamp >= 0

    def save(self, folder):
        os.makedirs(folder, exist_ok=True)
        W = self.walkers
        np.savez_compressed(os.path.join(folder, 'neural.npz'), wid=self.wid, wseg=self.wseg, lamp=self.lamp,
                            light=self.light,
                            opacity_lidar=(self.opacity_lidar if self.opacity_lidar is not None
                                           else np.zeros(0, np.float32)),
                            **{f'w{k}_times': w.times for k, w in enumerate(W)},
                            **{f'w{k}_theta': w.theta for k, w in enumerate(W)},
                            **{f'l{k}_times': L.times for k, L in enumerate(self.lights)})
        meta = {'schema': SCHEMA + '#ext', 'version': EXT_VERSION, 'info': self.info,
                'walkers': [{'track': w.track, 'slot': w.slot, 'stature': w.stature, 'box_h': w.box_h,
                             'scale': w.scale, 'method': w.method, 'fit': w.fit} for w in W],
                'lights': [dict(L.to_dict(), R=L.R.tolist(), centre=L.centre.tolist(), normal=L.normal.tolist(),
                                housing=L.housing.tolist(), lit={k: list(map(float, v)) for k, v in L.lit.items()},
                                states_seq=L.states, size=list(L.size)) for L in self.lights]}
        with open(os.path.join(folder, 'neural.json'), 'w', encoding='utf-8') as fh:
            json.dump(meta, fh, indent=1, default=float)

    @staticmethod
    def load(folder) -> 'NeuralExt | None':
        p, j = os.path.join(folder, 'neural.npz'), os.path.join(folder, 'neural.json')
        if not (os.path.isfile(p) and os.path.isfile(j)):
            return None
        try:
            with open(j, encoding='utf-8') as fh:
                meta = json.load(fh)
            if meta.get('version') != EXT_VERSION:
                return None
            with np.load(p, allow_pickle=False) as z:
                W = [Walker(int(m['track']), int(m['slot']), float(m['stature']), float(m['box_h']),
                            z[f'w{k}_times'], z[f'w{k}_theta'], float(m['scale']), str(m['method']),
                            dict(m.get('fit') or {})) for k, m in enumerate(meta['walkers'])]
                Ls = []
                for k, m in enumerate(meta['lights']):
                    Ls.append(Light(int(m['track']), str(m['group']), np.asarray(m['centre'], float),
                                    np.asarray(m['R'], float), tuple(m['size']), np.asarray(m['normal'], float),
                                    np.asarray(m['housing'], float), {a: np.asarray(b, float)
                                                                      for a, b in (m.get('lit') or {}).items()},
                                    z[f'l{k}_times'], list(m.get('states_seq') or []), str(m['source']),
                                    str(m.get('facing', 'path'))))
                opl = z['opacity_lidar']
                return NeuralExt(W, z['wid'], z['wseg'], Ls, z['lamp'], z['light'], opl if len(opl) else None,
                                 dict(meta.get('info') or {}))
        except (OSError, ValueError, KeyError, TypeError):
            return None


class _Bound:
    """A NeuralExt bound to a TorchScene: posed() and colours_at() for core.splat_reproject."""

    def __init__(self, ext: NeuralExt, ts):
        import torch
        self.ext, self.ts = ext, ts
        dev = ts.dev
        self.rows = [torch.tensor(np.flatnonzero(ext.wid == k), device=dev) for k in range(len(ext.walkers))]
        self.seg = torch.tensor(ext.wseg.astype(np.int64), device=dev)
        lr = np.flatnonzero(ext.lamp >= 0)
        self.lamp_rows = torch.tensor(lr, device=dev)
        self.lamp_light = ext.light[lr]
        self.lamp_k = ext.lamp[lr].astype(int)
        if ext.opacity_lidar is not None and len(ext.opacity_lidar) == len(ext.wid):
            ts.op_lidar_logit = torch.logit(torch.tensor(ext.opacity_lidar, dtype=torch.float32, device=dev)
                                            .clamp(1e-4, 1 - 1e-4))
        if ext.relight:
            from . import relight as RL
            ts.relight = RL.relighter(ext.relight, ts, ext.emissive())

    def posed(self, t: float):
        import torch
        ts = self.ts
        P, Rg, shown = ts.posed_rigid(t)
        if not self.ext.walkers or ts.drive is None:
            return P, Rg, shown
        dev = ts.dev
        for k, w in enumerate(self.ext.walkers):
            rows = self.rows[k]
            if not len(rows):
                continue
            tr = ts.drive.tracks.get(w.track)
            po = tr.pose(t) if tr is not None else None
            if po is None:
                continue                                  # hidden already: its actor slot is not shown
            th = torch.tensor(w.theta_at(t)[None], dtype=torch.float32, device=dev)
            Rs, os_, _ = forward_kinematics(th, w.stature, w.box_h, torch.tensor([w.scale], device=dev))
            sg = self.seg[rows]
            Rseg, oseg = Rs[0][sg], os_[0][sg]
            Rb = torch.tensor(po[0], dtype=torch.float32, device=dev)
            cb = torch.tensor(po[1], dtype=torch.float32, device=dev)
            loc = ts.means[rows]
            Pb = torch.einsum('nij,nj->ni', Rseg, loc) + oseg
            P = P.index_copy(0, rows, Pb @ Rb.T + cb)
            Rq = SR.quat_to_rot(ts.quat[rows])
            Rg = Rg.index_copy(0, rows, Rb[None] @ Rseg @ Rq)
        return P, Rg, shown

    def colours_at(self, t: float, col):
        import torch
        if not len(self.lamp_rows):
            return col
        lc = np.stack([lamp_colours(L, t) for L in self.ext.lights])        # (heads, 3, 3)
        c = torch.tensor(lc[self.lamp_light, self.lamp_k], dtype=col.dtype, device=col.device)
        return col.index_copy(0, self.lamp_rows, c)


def _append(scene: SP.Scene, parts: list, drop_slots=()) -> tuple:
    """A new Scene: scene's Gaussians without those of the actor slots dropped, then parts appended; with the
    extension arrays ('wid', 'wseg', 'lamp', 'light') for every Gaussian."""
    keep = ~np.isin(scene.actor, np.asarray(list(drop_slots), np.int32)) if drop_slots else np.ones(scene.n, bool)
    n0 = int(keep.sum())
    arr = {k: [getattr(scene, k)[keep]] for k in ('means', 'scales', 'quats', 'opacity', 'colours', 'actor', 'obj')}
    ext = {'wid': [np.full(n0, -1, np.int16)], 'wseg': [np.full(n0, -1, np.int16)],
           'lamp': [np.full(n0, -1, np.int8)], 'light': [np.full(n0, -1, np.int32)]}
    for p in parts:
        n = len(p['means'])
        for k in arr:
            arr[k].append(np.asarray(p[k]).astype(getattr(scene, k).dtype))
        ext['wid'].append(np.asarray(p.get('wid', np.full(n, -1)), np.int16))
        ext['wseg'].append(np.asarray(p.get('wseg', np.full(n, -1)), np.int16))
        ext['lamp'].append(np.asarray(p.get('lamp', np.full(n, -1)), np.int8))
        ext['light'].append(np.asarray(p.get('light', np.full(n, -1)), np.int32))
    sc = SP.Scene(*(np.concatenate(arr[k]) for k in ('means', 'scales', 'quats', 'opacity', 'colours', 'actor',
                                                      'obj')),
                  list(scene.actor_ids), scene.sky.copy(), scene.origin.copy(), scene.source, dict(scene.info),
                  scene.projection)
    return sc, {k: np.concatenate(v) for k, v in ext.items()}


def _active(nst: NeuralSettings, sst) -> bool:
    return nst.walkers == 'articulated' or nst.traffic_lights or (nst.lidar_opacity and sst.optimise)


def ext_key(scene: SP.Scene, drive: SP.Drive, sst, nst: NeuralSettings) -> str:
    h = hashlib.sha256(json.dumps({'drive': drive.key, 'n': scene.n, 'projection': scene.projection,
                                   'build': sst.build_key(), 'neural': nst.build_key(),
                                   'opacity_sum': round(float(scene.opacity.sum()), 3)}, sort_keys=True,
                                  default=str).encode())
    return h.hexdigest()[:12]


def augment(scene: SP.Scene, drive: SP.Drive, sst, project, engine=None, progress=None, cancel=None, folder=None,
            settings: NeuralSettings | None = None, pose_model=None, keypoints=None,
            light_detector=None) -> SP.Scene:
    """The scene with this module's extension (walkers, lights, lidar opacity) as the settings ask, built or read
    from <folder>/neural-<key>; the scene itself when nothing is asked. Relighting is attached as set (not cached)."""
    nst = settings or settings_of(project)
    from . import relight as RL
    rl = RL.RelightSettings.from_dict(nst.relight)
    rl_d = rl.to_dict() if rl.condition != 'none' else None
    if not _active(nst, sst):
        if rl_d:
            ext = NeuralExt([], np.full(scene.n, -1, np.int16), np.full(scene.n, -1, np.int16), [],
                            np.full(scene.n, -1, np.int8), np.full(scene.n, -1, np.int32), None,
                            {'relight': rl_d}, rl_d)
            scene.neural = ext
        else:
            scene.neural = None
        return scene
    key = ext_key(scene, drive, sst, nst)
    sub = os.path.join(folder, f'neural-{key}') if folder else None
    if sub and os.path.isdir(sub):
        sc = SP.Scene.load(sub)
        ext = NeuralExt.load(sub)
        if sc is not None and ext is not None and len(ext.wid) == sc.n:
            sc.info = dict(scene.info)                    # the base scene's (cached, timings, optimisation)
            ext.relight = rl_d
            ext.info['cached'] = True
            if rl_d:
                ext.info['relight'] = rl_d
            sc.neural = ext
            return sc
    t0 = time.perf_counter()
    stop = cancel or (lambda: False)
    dev = SR.device_for(sst.device, min(sst.gpu_mem_gb, nst.gpu_mem_gb))
    info = {'projection': scene.projection, 'walkers': {'count': 0}, 'lights': {'count': 0}, 'lidar_opacity': None}
    parts, drops, walkers, lights = [], [], [], []
    if nst.walkers == 'articulated':
        peds = dict(pedestrian_tracks(drive, scene))
        kps = keypoints
        if kps is None and peds and nst.keypoints:
            pm = pose_model
            if pm is None and pose_status(nst.pose_model, verify=False)['verified']:
                try:
                    pm = PoseModel(nst.pose_model, device=sst.device, mem_gb=min(nst.gpu_mem_gb, 3.0))
                except (ValueError, FileNotFoundError, ImportError, OSError) as e:
                    info['walkers']['note'] = f'Keypoint model not loaded ({e}): walkers posed from their tracks.'
            if pm is not None:
                kps = walker_keypoints(drive, {i: drive.tracks[i] for i in peds}, nst, pm,
                                       lambda f, t='': _prog(progress, 0.3 * f, t), stop)
                info['walkers']['pose_model'] = pm.describe() if hasattr(pm, 'describe') else {}
            elif 'note' not in info['walkers']:
                info['walkers']['note'] = ('The keypoint model (ViTPose-B) is not downloaded: walkers are posed by a '
                                           'gait prior from their tracks.')
        walkers, wparts, drops, winfo = build_walkers(scene, drive, sst, nst, kps, dev,
                                                      lambda f, t='': _prog(progress, 0.3 + 0.2 * f, t), stop)
        info['walkers'].update(winfo)
        parts += wparts
    if nst.traffic_lights:
        heads, how = None, 'boxes'
        if nst.light_source == 'detector' or (nst.light_source == 'auto' and not _light_tracks(drive)):
            det = light_detector
            if det is None:
                from . import detection as DT
                from . import detection_eval as DE
                de_st = DE.settings_of(project) if project is not None else DE.DetectionSettings()
                if DT.status(de_st.model)['verified']:
                    det = DT.load(de_st.detector()).detect
            if det is not None:
                heads = lights_from_detections(drive, scene, det, nst,
                                               progress=lambda f, t='': _prog(progress, 0.5 + 0.05 * f, t),
                                               cancel=stop)
                how = 'detector'
            else:
                heads = {}
                how = 'none: the drive has no traffic-light boxes and the detector is not downloaded'
        lights, linfo = build_lights(scene, drive, nst, lambda f, t='': _prog(progress, 0.55 + 0.05 * f, t), stop,
                                     heads)
        linfo['heads_from'] = how
        info['lights'] = linfo
        if lights:
            parts.append(light_gaussians(lights))
    sc, ex = _append(scene, parts, drops)
    ext = NeuralExt(walkers, ex['wid'], ex['wseg'], lights, ex['lamp'], ex['light'], None, info)
    sc.neural = ext
    if nst.lidar_opacity and sst.optimise and drive.lidars:
        ts = SP.TorchScene(sc, drive, dev)
        op, lst = fit_lidar_opacity(ts, drive, sst, nst, lambda f, t='': _prog(progress, 0.6 + 0.4 * f, t), stop)
        ext.opacity_lidar = op
        info['lidar_opacity'] = lst
        del ts
    info['seconds'] = round(time.perf_counter() - t0, 2)
    info['cached'] = False
    if sub:
        sc.save(sub)
        ext.save(sub)
    ext.relight = rl_d
    if rl_d:
        info['relight'] = rl_d
    sc.neural = ext
    if dev.type == 'cuda':
        import torch
        torch.cuda.empty_cache()
    return sc


# =========================================================================== the study
@dataclass
class NeuralResult:
    settings: NeuralSettings
    drive: dict
    scene: dict
    projection: dict
    lidar: dict
    perception: dict
    relight: dict
    detector: dict | None
    findings: list
    notes: list
    timing: dict
    seconds: float = 0.0
    previews: dict = field(default_factory=dict)


def _classes(de_st) -> list:
    from . import detection as DT
    return [c for c in DT.EVAL_CLASSES if c in de_st.classes or (c in DT.GROUPS and
                                                                 any(m in de_st.classes for m in DT.GROUPS[c]))]


def _instants(drive: SP.Drive, n: int) -> list:
    first = max(drive.cameras.values(), key=lambda c: (len(c.frames), c.tag == 'F'))
    times = [fr[0] for fr in first.frames]
    n = min(n, len(times))
    picks = np.unique(((np.arange(n) + 0.5) / max(n, 1) * len(times)).astype(int)) if n else np.zeros(0, int)
    return [float(times[i]) for i in picks]


def _summarise(rows: dict, classes, edges, kind: str, detected: bool = True) -> dict:
    """Per class (rig level: an object once per instant, found when any camera finds it): objects scored, found,
    recall; per range bin too. Without a detector (detected False) only the objects shown: recall None."""
    from . import detection as DT
    from . import detection_eval as DE
    labels = DE.bin_labels(edges)
    out = {}
    for ec in classes:
        cidx = [DT.CLASSES.index(m) for m in DT.GROUPS.get(ec, (ec,))]
        keys = [k for k, r in rows.items() if r['vis'] and r['cls'] in cidx]
        if not keys:
            continue
        det = [rows[k]['det'] for k in keys]
        bins = DE.bin_of(np.array([rows[k]['dist'] for k in keys]), edges)
        by = []
        for b in range(len(edges) + 1):
            m = bins == b
            if m.any():
                by.append({'range': labels[b], 'objects': int(m.sum()),
                           'recall': round(float(np.mean(np.asarray(det)[m])), 4) if detected else None})
        out[ec] = {'objects': len(keys), 'found': int(sum(det)) if detected else None,
                   'recall': round(float(np.mean(det)), 4) if detected else None, 'by_range': by, 'source': kind}
    return out


def evaluate_rig(ts, drive: SP.Drive, scene: SP.Scene, rig, instants, de_st, detector=None, progress=None,
                 cancel=None, keep=None) -> dict:
    """The recorded (or any) rig re-rendered at the instants: per object and instant whether some camera shows it
    unignored and whether the detector finds it (core.detection_eval's ignore rules and matching, as the log replay's
    A/B). keep: {(instant, camera tag)} whose images to return. Returns {'rows', 'images'}."""
    import cv2
    from . import detection_eval as DE
    stop = cancel or (lambda: False)
    classes = _classes(de_st)
    rows, images = {}, {}
    for k, t in enumerate(instants):
        if stop():
            raise Cancelled()
        _prog(progress, k / max(len(instants), 1), f'Instant {k + 1}/{len(instants)}')
        Rw, pw = drive.traj.pose(t)
        renders = SP._render_rig_frame(ts, drive, rig, t, scene)
        for c in rig.cameras:
            rgb, ids, rng_img = renders[c.tag]
            if keep is not None and (k, c.tag) in keep:
                images[(k, c.tag)] = rgb
            objs = SP._objects_in_view(drive, t, c, ids, Rw, pw, SP._rays_of(c.model))
            g = DE.gt_objects({'sample_data_token': f'n:{c.tag}:{k}', 'objects': objs}, de_st)
            score = np.full(len(g['id']), -np.inf)
            if detector is not None and len(g['id']):
                dets = detector(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                for e, i, sc in DE.evaluate_image(g, dets, rng_img, de_st, classes)['gt']:
                    score[i] = max(score[i], sc)
            _accumulate(rows, k, g, score, de_st)
    return {'rows': rows, 'images': images}


def _accumulate(rows, k, g, score, de_st):
    from . import detection_eval as DE
    for i in range(len(g['id'])):
        key = (k, int(g['id'][i]))
        r = rows.setdefault(key, {'cls': int(g['cls'][i]), 'dist': float(g['dist'][i]), 'vis': False,
                                  'det': False})
        if g['ignore'][i] == DE.IGN_NONE and g['cls'][i] >= 0:
            r['vis'] = True
            if score[i] >= de_st.score:
                r['det'] = True


def evaluate_recorded(drive: SP.Drive, instants, de_st, detector=None, progress=None, cancel=None,
                      keep=None, scale: float = 1.0) -> dict:
    """The recorded images at the instants (each camera's frame nearest each instant, within ACTOR_PAD_S) with their
    own ground truth: the real reference, scored as evaluate_rig scores re-renders. scale: the images resized by it
    (and their boxes), so the detector sees the size the re-renders have."""
    import cv2
    from . import detection_eval as DE
    stop = cancel or (lambda: False)
    classes = _classes(de_st)
    rows, images = {}, {}
    for k, t in enumerate(instants):
        if stop():
            raise Cancelled()
        _prog(progress, k / max(len(instants), 1), f'Recorded images · instant {k + 1}/{len(instants)}')
        for cam in drive.cameras.values():
            if not cam.frames:
                continue
            j = int(np.argmin([abs(fr[0] - t) for fr in cam.frames]))
            tf, path, gfr = cam.frames[j]
            if gfr is None or abs(tf - t) > SP.ACTOR_PAD_S:
                continue
            frame = dict(gfr)
            if abs(scale - 1.0) > 1e-6:
                frame['objects'] = [dict(o, bbox=[float(v) * scale for v in o['bbox'][:4]],
                                         pixels=float(o.get('pixels') or 0) * scale * scale)
                                    for o in gfr.get('objects') or () if o.get('bbox')]
            g = DE.gt_objects(frame, de_st)
            score = np.full(len(g['id']), -np.inf)
            img = None
            if detector is not None or (keep is not None and (k, cam.tag) in keep):
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                if img is not None and abs(scale - 1.0) > 1e-6:
                    img = cv2.resize(img, (max(16, int(round(img.shape[1] * scale))),
                                           max(16, int(round(img.shape[0] * scale)))), interpolation=cv2.INTER_AREA)
            if keep is not None and (k, cam.tag) in keep and img is not None:
                images[(k, cam.tag)] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if detector is not None and img is not None and len(g['id']):
                dets = detector(img)
                for e, i, sc in DE.evaluate_image(g, dets, None, de_st, classes)['gt']:
                    score[i] = max(score[i], sc)
            _accumulate(rows, k, g, score, de_st)
    return {'rows': rows, 'images': images}


def _validation_row(val: dict) -> dict:
    cams = (val or {}).get('cameras') or {}
    lv = (val or {}).get('lidar') or {}
    return {'psnr_db': cams.get('psnr'), 'ssim': cams.get('ssim'), 'frames': cams.get('frames'),
            'cameras': {k: {'psnr_db': v['psnr'], 'ssim': v['ssim'], 'frames': v['frames']}
                        for k, v in (cams.get('cameras') or {}).items()},
            'lidar_median_abs_error_m': lv.get('median_abs_error_m'), 'lidar_p90_abs_error_m':
                lv.get('p90_abs_error_m'), 'lidar_within_10cm': lv.get('within_10cm'),
            'lidar_return_agreement': lv.get('return_agreement'),
            'lidar_median_abs_error_steep_m': lv.get('median_abs_error_steep_m')}


def run(project, engine=None, settings: NeuralSettings | None = None, variant=None, progress=None, cancel=None,
        detector=None, cache_root=None, drive: SP.Drive | None = None, pose_model=None) -> NeuralResult | None:
    """The neural replay study on the log replay's drive (module docstring: The study); None when cancelled.
    detector: a callable BGR image -> detections (by default the detection settings' model when downloaded)."""
    t_start = time.perf_counter()
    if SP.torch_missing():
        raise ValueError(SP.torch_missing())
    nst = settings or settings_of(project)
    sst = SP.settings_of(project)
    stop = cancel or (lambda: False)
    if engine is None and project is not None:
        from .analysis import Engine
        engine = Engine()
    if cache_root is None and project is not None:
        from .. import workdir
        cache_root = workdir.project_dir(project, None, 'splat')
    from . import detection as DT
    from . import detection_eval as DE
    from . import relight as RL
    de_st = DE.settings_of(project) if project is not None else DE.DetectionSettings()
    timing, notes = {}, []
    det_info = None
    if detector is None and sst.detect:
        s = DT.status(de_st.model)
        if s['verified']:
            d = DT.load(de_st.detector())
            detector, det_info = d.detect, d.describe()
        else:
            notes.append(f'Detection skipped: {s["text"]}: the comparison counts what each render shows.')
    elif detector is not None:
        det_info = dict(getattr(detector, 'describe', lambda: {})() or {'model': 'custom'})
    P = lambda a, b: (lambda f, t='': _prog(progress, a + (b - a) * f, t))  # noqa: E731
    try:
        t0 = time.perf_counter()
        if drive is None:
            drive = SP.open_drive(sst.drive, cache_root, P(0.0, 0.02), stop)
        timing['open_s'] = round(time.perf_counter() - t0, 2)
        other = 'ewa' if nst.projection == 'ut' else 'ut'
        t0 = time.perf_counter()
        base = SP.build_scene(drive, sst, project, engine, P(0.02, 0.25), stop, cache_root,
                              projection=nst.projection, neural=False)
        timing['scene_s'] = round(time.perf_counter() - t0, 2)
        dev = SR.device_for(sst.device, min(sst.gpu_mem_gb, nst.gpu_mem_gb))
        proj = {}
        if nst.compare_projection:
            t0 = time.perf_counter()
            alt = SP.build_scene(drive, sst, project, engine, P(0.25, 0.45), stop, cache_root,
                                 projection=other, neural=False)
            timing['scene_other_s'] = round(time.perf_counter() - t0, 2)
            t0 = time.perf_counter()
            for name, sc in ((nst.projection, base), (other, alt)):
                ts = SP.TorchScene(sc, drive, dev)
                v = SP.validate(ts, drive, sst, project, engine, None, stop)
                proj[name] = dict(_validation_row(v), gaussians=sc.n,
                                  optimisation=(sc.info.get('optimisation') or {}))
                del ts
            timing['validation_s'] = round(time.perf_counter() - t0, 2)
        t0 = time.perf_counter()
        aug = SP.build_scene(drive, sst, project, engine, P(0.45, 0.6), stop, cache_root,
                             projection=nst.projection, neural=nst) if _active(nst, sst) else base
        timing['extension_s'] = round(time.perf_counter() - t0, 2)
        ext = getattr(aug, 'neural', None)
        lid = {}
        if nst.compare_lidar and ext is not None and ext.opacity_lidar is not None:
            t0 = time.perf_counter()
            ts = SP.TorchScene(aug, drive, dev)
            with_l = (SP.validate(ts, drive, sst, project, engine, None, stop) or {}).get('lidar') or {}
            ts.op_lidar_logit = None
            cam_l = (SP.validate(ts, drive, sst, project, engine, None, stop) or {}).get('lidar') or {}
            lid = {'camera_opacity': cam_l, 'lidar_opacity': with_l, 'fit': (ext.info or {}).get('lidar_opacity')}
            timing['lidar_s'] = round(time.perf_counter() - t0, 2)
            del ts
        perc, prev = {}, {}
        instants = _instants(drive, nst.eval_instants)
        classes = _classes(de_st)
        edges = list(de_st.range_edges)
        rig = SP.recorded_rig(drive, sst, project, engine)
        f_img = min(sst.render_scale / max(drive.image_scale, 1e-6), 2.0)
        if nst.compare_walkers:
            t0 = time.perf_counter()
            keep = _preview_keys(drive, instants)
            ts_r = SP.TorchScene(base, drive, dev)
            ev_r = evaluate_rig(ts_r, drive, base, rig, instants, de_st, detector, P(0.6, 0.7), stop, keep)
            del ts_r
            ts_n = SP.TorchScene(aug, drive, dev)
            ts_n.relight = None
            ev_n = evaluate_rig(ts_n, drive, aug, rig, instants, de_st, detector, P(0.7, 0.8), stop, keep)
            ev_real = evaluate_recorded(drive, instants, de_st, detector if nst.real_reference else None,
                                        P(0.8, 0.85), stop, keep, f_img) if drive.gt is not None else None
            dd = detector is not None
            perc = {'instants': [round(x, 4) for x in instants], 'classes': classes, 'range_edges': edges,
                    'rigid': _summarise(ev_r['rows'], classes, edges, 'rigid walkers, no light planes', dd),
                    'neural': _summarise(ev_n['rows'], classes, edges, 'articulated walkers, emissive lights', dd),
                    'recorded': (_summarise(ev_real['rows'], classes, edges, 'recorded images',
                                            dd and nst.real_reference) if ev_real is not None else {}),
                    'image_scale': round(f_img, 4), 'detected': detector is not None}
            prev['walkers'] = {'rigid': ev_r['images'], 'neural': ev_n['images'],
                               'recorded': (ev_real or {}).get('images', {})}
            timing['perception_s'] = round(time.perf_counter() - t0, 2)
        rel = {}
        if nst.relight_conditions:
            t0 = time.perf_counter()
            ts_n = SP.TorchScene(aug, drive, dev)
            em = ext.emissive() if ext is not None else None
            base_rows = None
            if perc:
                base_rows = perc['neural']
            else:
                ts_n.relight = None
                base_rows = _summarise(evaluate_rig(ts_n, drive, aug, rig, instants, de_st, detector, None,
                                                    stop)['rows'], classes, edges, 'unrelit', detector is not None)
            rl0 = RL.RelightSettings.from_dict(nst.relight)
            keep = _preview_keys(drive, instants)
            for j, cond in enumerate(nst.relight_conditions):
                rs = RL.RelightSettings.from_dict(dict(rl0.to_dict(), condition=cond))
                ts_n.relight = RL.relighter(rs.to_dict(), ts_n, em)
                ev = evaluate_rig(ts_n, drive, aug, rig, instants, de_st, detector,
                                  P(0.85 + 0.15 * j / len(nst.relight_conditions),
                                    0.85 + 0.15 * (j + 1) / len(nst.relight_conditions)), stop, keep)
                summ = _summarise(ev['rows'], classes, edges, f'relit {cond}', detector is not None)
                rel[cond] = {'label': getattr(ts_n.relight, 'label', rs.label()), 'settings': rs.to_dict(),
                             'classes': summ,
                             'change': {c: round(summ[c]['recall'] - base_rows[c]['recall'], 4)
                                        for c in summ if c in base_rows and summ[c]['recall'] is not None
                                        and base_rows[c]['recall'] is not None}}
                prev.setdefault('relight', {})[cond] = ev['images']
            timing['relight_s'] = round(time.perf_counter() - t0, 2)
    except (Cancelled, SP.Cancelled):
        return None
    res = NeuralResult(nst, drive.info(), SP._scene_doc(aug), proj, lid, perc, rel, det_info, [], notes, timing,
                       round(time.perf_counter() - t_start, 2), prev)
    res.findings = _findings(res)
    _prog(progress, 1.0, 'Done')
    return res


def _preview_keys(drive: SP.Drive, instants, k: int = 3) -> set:
    """The (instant, camera) pairs kept for the previews: the k (at most one per instant) whose recorded frame shows
    the most pedestrian pixels (the ground truth's objects), else the most pedestrians within 30 m in front."""
    best = []
    for j, t in enumerate(instants):
        Rw, pw = drive.traj.pose(t)
        for tag, cam in drive.cameras.items():
            score = 0.0
            if cam.frames:
                i = int(np.argmin([abs(fr[0] - t) for fr in cam.frames]))
                gfr = cam.frames[i][2]
                if gfr is not None and abs(cam.frames[i][0] - t) <= SP.ACTOR_PAD_S:
                    score = float(sum(float(o.get('pixels') or 0) for o in gfr.get('objects') or ()
                                      if o.get('class') == 'pedestrian'))
            if score == 0.0:
                for tr in drive.tracks.values():
                    po = tr.pose(t) if tr.cls == 'pedestrian' else None
                    if po is None:
                        continue
                    cv = Rw.T @ (po[1] - pw) - cam.t
                    if (cam.R.T @ cv)[2] > 0 and np.linalg.norm(cv) < 30:
                        score += 1e-3
            best.append((score, -j, tag, j))
    best.sort(reverse=True)
    out, used = set(), set()
    for score, _, tag, j in best:
        if j in used:
            continue
        out.add((j, tag))
        used.add(j)
        if len(out) >= k:
            break
    return out


def _pc(x):
    return '–' if x is None else f'{100 * x:.0f} %'


def _findings(res: NeuralResult) -> list:
    out = []
    pj = res.projection
    if len(pj) == 2:
        a, b = pj.get('ewa', {}), pj.get('ut', {})
        if a.get('psnr_db') is not None and b.get('psnr_db') is not None:
            d = b['psnr_db'] - a['psnr_db']
            out.append({'severity': 'good' if d > 0 else 'note', 'kind': 'projection',
                        'message': f'Held-out frames: {b["psnr_db"]:.2f} dB PSNR through the Unscented Transform '
                                   f'against {a["psnr_db"]:.2f} dB through the affine (EWA) projection ({d:+.2f} dB; '
                                   f'SSIM {b.get("ssim") or 0:.3f} against {a.get("ssim") or 0:.3f}).'})
        if a.get('lidar_median_abs_error_m') is not None and b.get('lidar_median_abs_error_m') is not None:
            out.append({'severity': 'note', 'kind': 'projection',
                        'message': f'Held-out lidar median range error {100 * b["lidar_median_abs_error_m"]:.1f} cm '
                                   f'(UT) against {100 * a["lidar_median_abs_error_m"]:.1f} cm (EWA).'})
    L = res.lidar
    if L.get('camera_opacity') and L.get('lidar_opacity'):
        c, l_ = L['camera_opacity'], L['lidar_opacity']
        out.append({'severity': 'note', 'kind': 'lidar',
                    'message': f'Lidar opacity: return agreement {_pc(l_.get("return_agreement"))} against '
                               f'{_pc(c.get("return_agreement"))} with the camera\'s opacity; median range error '
                               f'{100 * (l_.get("median_abs_error_m") or 0):.1f} cm against '
                               f'{100 * (c.get("median_abs_error_m") or 0):.1f} cm (held-out sweeps).'})
    pc = res.perception or {}
    if pc:
        for cls in ('pedestrian', 'traffic_light'):
            r, n, real = (pc.get('rigid') or {}).get(cls), (pc.get('neural') or {}).get(cls), \
                (pc.get('recorded') or {}).get(cls)
            if not (r or n or real):
                continue
            what = 'recall' if pc.get('detected') else 'objects shown'
            txt = (f'{cls.replace("_", " ").capitalize()} rig {what}: '
                   f'{_pc((n or {}).get("recall"))} of {(n or {}).get("objects", 0)} re-rendered with '
                   f'{"articulated walkers" if cls == "pedestrian" else "emissive lights"}, '
                   f'{_pc((r or {}).get("recall"))} of {(r or {}).get("objects", 0)} with the log replay\'s '
                   f'{"rigid walkers" if cls == "pedestrian" else "reconstruction alone"}')
            if real:
                txt += f'; {_pc(real.get("recall"))} of {real.get("objects", 0)} on the recorded images'
            out.append({'severity': 'note', 'kind': 'perception', 'message': txt + '.'})
    for cond, d in (res.relight or {}).items():
        ch = d.get('change') or {}
        if ch:
            parts = ', '.join(f'{c} {100 * v:+.0f} pt' for c, v in ch.items() if c in ('pedestrian', 'car', 'vehicle'))
            out.append({'severity': 'note', 'kind': 'relight',
                        'message': f'Relit to {cond} ({d.get("label")}): recall change against the unrelit re-render '
                                   f'{parts or "—"} (relit, not recorded).'})
    out.append({'severity': 'note', 'kind': 'caveat', 'message': ASSUMPTIONS})
    return out


def summary(res: NeuralResult) -> dict:
    pj = res.projection
    pc = res.perception or {}
    g = lambda side, c: ((pc.get(side) or {}).get(c) or {}).get('recall')  # noqa: E731
    return {'drive': os.path.basename(str(res.drive.get('path', ''))), 'projection': res.settings.projection,
            'psnr_ewa_db': (pj.get('ewa') or {}).get('psnr_db'), 'psnr_ut_db': (pj.get('ut') or {}).get('psnr_db'),
            'lidar_error_ewa_m': (pj.get('ewa') or {}).get('lidar_median_abs_error_m'),
            'lidar_error_ut_m': (pj.get('ut') or {}).get('lidar_median_abs_error_m'),
            'walkers': ((res.scene.get('neural') or {}).get('walkers') or {}).get('count'),
            'lights': ((res.scene.get('neural') or {}).get('lights') or {}).get('count'),
            'pedestrian_recall_rigid': g('rigid', 'pedestrian'), 'pedestrian_recall_articulated': g('neural',
                                                                                                     'pedestrian'),
            'pedestrian_recall_recorded': g('recorded', 'pedestrian'),
            'traffic_light_recall_rigid': g('rigid', 'traffic_light'),
            'traffic_light_recall_neural': g('neural', 'traffic_light'),
            'relight': {c: d.get('change') for c, d in (res.relight or {}).items()},
            'findings': [f['message'] for f in res.findings if f['severity'] in ('warning', 'serious', 'critical')]}


def to_doc(res: NeuralResult, project=None) -> dict:
    return {'schema': SCHEMA, 'project': getattr(project, 'name', ''), 'frame': FRAME,
            'units': {'length': 'm', 'psnr': 'dB', 'recall': 'share 0–1', 'time': 's from the drive\'s first pose'},
            'settings': res.settings.to_dict(), 'drive': res.drive, 'scene': res.scene,
            'projection': res.projection, 'lidar': res.lidar, 'perception': res.perception, 'relight': res.relight,
            'detector': res.detector, 'findings': res.findings, 'notes': res.notes, 'timing': res.timing,
            'seconds': res.seconds, 'assumptions': ASSUMPTIONS,
            'references': ['Wu et al., 3DGUT, CVPR 2025, arXiv:2412.12507',
                           'Winter, Biomechanics and Motor Control of Human Movement, 4th ed., Wiley 2009',
                           'Xu et al., ViTPose, NeurIPS 2022, arXiv:2204.12484',
                           'Brooks et al., InstructPix2Pix, CVPR 2023, arXiv:2211.09800'],
            'relit_not_recorded': bool(res.relight)}


def perception_csv(res: NeuralResult) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator='\n')
    w.writerow(('render', 'class', 'objects', 'found', 'recall'))
    for side in ('recorded', 'rigid', 'neural'):
        for c, r in ((res.perception or {}).get(side) or {}).items():
            w.writerow((side, c, r['objects'], r['found'], r['recall']))
    for cond, d in (res.relight or {}).items():
        for c, r in (d.get('classes') or {}).items():
            w.writerow((f'relit {cond}', c, r['objects'], r['found'], r['recall']))
    return buf.getvalue()


def projection_csv(res: NeuralResult) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator='\n')
    w.writerow(('projection', 'camera', 'frames', 'psnr_db', 'ssim'))
    for name, r in (res.projection or {}).items():
        w.writerow((name, 'all', r.get('frames'), r.get('psnr_db'), r.get('ssim')))
        for tag, c in (r.get('cameras') or {}).items():
            w.writerow((name, tag, c['frames'], c['psnr_db'], c['ssim']))
    return buf.getvalue()


def _tiles(rows: list, labels: list, width: int = 300):
    import cv2
    from .pngtext import put_text
    out = []
    for row, lab in zip(rows, labels):
        tiles = []
        for img, t in row:
            if img is None:
                continue
            im = cv2.cvtColor(SP._thumb(img, width), cv2.COLOR_RGB2BGR)
            im = cv2.copyMakeBorder(im, 20, 2, 2, 2, cv2.BORDER_CONSTANT, value=(24, 24, 24))
            put_text(im, t, (5, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (235, 235, 235), 1, cv2.LINE_AA)
            tiles.append(im)
        if not tiles:
            continue
        hmax = max(x.shape[0] for x in tiles)
        tiles = [cv2.copyMakeBorder(x, 0, hmax - x.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(24, 24, 24))
                 for x in tiles]
        r = np.concatenate(tiles, 1)
        head = np.full((22, r.shape[1], 3), 16, np.uint8)
        put_text(head, lab, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (235, 235, 235), 1, cv2.LINE_AA)
        out.append(np.concatenate([head, r], 0))
    if not out:
        return None
    wmax = max(x.shape[1] for x in out)
    out = [cv2.copyMakeBorder(x, 0, 0, 0, wmax - x.shape[1], cv2.BORDER_CONSTANT, value=(16, 16, 16)) for x in out]
    ok, buf = cv2.imencode('.png', np.concatenate(out, 0))
    return bytes(buf) if ok else None


def walkers_png(res: NeuralResult) -> bytes | None:
    pv = (res.previews or {}).get('walkers') or {}
    keys = sorted(set(pv.get('neural', {})) | set(pv.get('rigid', {})))
    rows, labels = [], []
    for k in keys:
        rows.append([(pv.get('recorded', {}).get(k), f'recorded {k[1]}'), (pv.get('rigid', {}).get(k), 'rigid'),
                     (pv.get('neural', {}).get(k), 'articulated + lights')])
        labels.append(f'instant {k[0] + 1} · camera {k[1]} (re-rendered except the first)')
    return _tiles(rows, labels)


def relight_png(res: NeuralResult) -> bytes | None:
    pv = (res.previews or {}).get('relight') or {}
    base = ((res.previews or {}).get('walkers') or {}).get('neural', {})
    keys = sorted(set(k for d in pv.values() for k in d))
    rows, labels = [], []
    for k in keys:
        rows.append([(base.get(k), 'unrelit')] + [(pv[c].get(k), f'relit: {c}') for c in pv])
        labels.append(f'instant {k[0] + 1} · camera {k[1]} · relit, not recorded')
    return _tiles(rows, labels)


def files(res: NeuralResult, project=None) -> list:
    out = [('neural_replay.json', (json.dumps(to_doc(res, project), indent=1, default=float) + '\n').encode()),
           ('neural_replay_perception.csv', perception_csv(res).encode()),
           ('neural_replay_projection.csv', projection_csv(res).encode())]
    for name, fn in (('neural_replay_walkers.png', walkers_png), ('neural_replay_relight.png', relight_png)):
        png = fn(res)
        if png:
            out.append((name, png))
    return out


def export(res: NeuralResult, project, out_dir) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for name, data in files(res, project):
        p = os.path.join(out_dir, name)
        with open(p, 'wb') as fh:
            fh.write(data)
        paths.append(p)
    return paths


# =========================================================================== validation
def _outcome(error, detail, ok=True, note=''):
    """vantage.validation's Outcome when it is importable, else the same fields as a dict."""
    try:
        from ..validation.suite import Outcome
        return Outcome(float(error), detail, ok=bool(ok), note=note)
    except ImportError:                                   # pragma: no cover
        return {'error': float(error), 'detail': detail, 'ok': bool(ok), 'note': note}


def validation_checks() -> list:
    """Checks for vantage/validation as dicts with suite.Spec's fields (id, model, title, quantity, reference,
    tolerance, unit, refs, fn, cost); each fn(ctx) returns a suite.Outcome. Proposed reference keys:
    'wu2025_3dgut', 'winter2009'. The lead registers them."""
    def ut_mc(ctx=None):
        r = ut_vs_monte_carlo([1.5, 0.3, 1.2], [0.3, 0.2, 0.25])
        return _outcome(r['ut_mean_err_px'], {k: round(v, 4) for k, v in r.items() if isinstance(v, float)},
                        ok=r['ut_cov_rel_err'] <= 0.15 and r['ewa_mean_err_px'] > r['ut_mean_err_px'],
                        note='the UT\'s mean against 400 000 samples; its covariance within 15 %')

    def ut_axis(ctx=None):
        c = check_ut_axis()
        return _outcome(c['value'], c)

    def fit(ctx=None):
        c = check_skeleton_fit()
        return _outcome(c['median_joint_error_deg'], c,
                        ok=c['median_joint_error_deg'] < c['prior_median_joint_error_deg'])
    return [
        {'id': 'neural.ut_monte_carlo', 'model': 'neural_replay',
         'title': 'The Unscented Transform places a Gaussian through a fisheye as a Monte Carlo does',
         'quantity': 'distance between the UT\'s projected mean and the mean of 400 000 samples of the Gaussian '
                     '(52° off the axis of the CARLA drives\' Kannala-Brandt fisheye, 1.9 m away), px',
         'reference': 'Monte Carlo through the same projection', 'tolerance': 0.5, 'unit': 'px',
         'refs': ('wu2025_3dgut', 'kannala2006'), 'fn': ut_mc, 'cost': 'cheap'},
        {'id': 'neural.ut_axis', 'model': 'neural_replay',
         'title': 'The 3D response along each ray equals the closed form on an equidistant lens\'s axis',
         'quantity': 'largest alpha difference over a 64 × 64 image', 'reference': 'the closed form of EWA with the '
         '0.3 px² dilation (exact on the axis)', 'tolerance': 2e-4, 'unit': 'alpha',
         'refs': ('wu2025_3dgut',), 'fn': ut_axis, 'cost': 'cheap'},
        {'id': 'neural.skeleton_fit', 'model': 'neural_replay',
         'title': 'The walker fit recovers known joint angles from noisy keypoints',
         'quantity': 'median joint-angle error of a synthetic 2 s walk seen by three fisheyes (keypoints with '
                     '1.5 px noise), below the gait prior\'s', 'reference': 'the synthetic walk\'s own angles',
         'tolerance': 6.0, 'unit': '°', 'refs': ('winter2009',), 'fn': fit, 'cost': 'study'},
    ]


def _fisheye(k: float = 1.0):
    """The CARLA drives' fisheye (Kannala-Brandt, 87° half field) at k × 320 × 256 px."""
    return SP.CamModel(int(320 * k), int(256 * k), 95.0 * k, 95.0 * k, 160.0 * k, 128.0 * k, 'kannala_brandt',
                       (0.007, 0.132, -0.193, 0.057), math.radians(87.0))


def ut_vs_monte_carlo(P, sigma, rot=None, n: int = 400000, seed: int = 0, model=None) -> dict:
    """The UT's and EWA's projected mean and covariance of one Gaussian (centre P, σ per axis) through a fisheye
    against a Monte Carlo of n samples (the reference)."""
    import torch
    m = model or _fisheye()
    proj = m.projector()
    dt = torch.float64
    Pt = torch.tensor([P], dtype=dt)
    R = torch.tensor(np.eye(3) if rot is None else rot, dtype=dt)[None]
    s = torch.tensor([sigma], dtype=dt)
    u, v, cov, _ = SR.ut_project(Pt, R, s, proj)
    ue, ve, cove, _, _ = SR.ewa_project(Pt, SR.covariance(R, s), proj)
    rng = np.random.default_rng(seed)
    X = np.asarray(P, float) + (rng.standard_normal((n, 3)) * np.asarray(sigma, float)) @ R[0].numpy().T
    uu, vv, ok = proj(torch.tensor(X, dtype=dt))
    uv = np.stack([uu.numpy(), vv.numpy()], 1)[ok.numpy()]
    mc_m = uv.mean(0)
    mc_c = np.cov(uv.T)
    ut_m = np.array([float(u[0]), float(v[0])])
    ew_m = np.array([float(ue[0]), float(ve[0])])
    fro = lambda c: float(np.linalg.norm(c - mc_c) / np.linalg.norm(mc_c))  # noqa: E731
    return {'mc_mean': mc_m, 'mc_cov': mc_c, 'ut_mean': ut_m, 'ut_cov': cov[0].numpy(), 'ewa_mean': ew_m,
            'ewa_cov': cove[0].numpy(), 'ut_mean_err_px': float(np.linalg.norm(ut_m - mc_m)),
            'ewa_mean_err_px': float(np.linalg.norm(ew_m - mc_m)), 'ut_cov_rel_err': fro(cov[0].numpy()),
            'ewa_cov_rel_err': fro(cove[0].numpy()), 'samples_in_field': int(len(uv))}


def check_ut_monte_carlo() -> dict:
    r = ut_vs_monte_carlo([1.5, 0.3, 1.2], [0.3, 0.2, 0.25])         # 52° off the axis, 1.9 m away
    ok = r['ut_mean_err_px'] <= 0.5 and r['ut_cov_rel_err'] <= 0.15 and r['ewa_mean_err_px'] > r['ut_mean_err_px']
    return {'passed': bool(ok), 'value': {k: round(v, 4) for k, v in r.items() if isinstance(v, float)}}


def check_ut_axis() -> dict:
    import torch
    f, z, sig, op = 100.0, 10.0, 0.08, 0.7
    proj = SR.LensProjector(f, f, 32.0, 32.0, 'equidistant', max_theta=math.radians(80))
    dt = torch.float64
    D, _ = SR.projector_rays(proj, 64, 64, dtype=dt)
    out = SR.splat_ut(torch.tensor([[0, 0, z]], dtype=dt), torch.eye(3, dtype=dt)[None],
                      torch.tensor([[sig] * 3], dtype=dt), torch.ones(1, 1, dtype=dt), torch.tensor([op], dtype=dt),
                      proj, 64, 64, D)
    a = out['alpha'].reshape(64, 64).numpy()
    s2 = (f * sig / z) ** 2 + SR.LOWPASS
    yy, xx = np.mgrid[0:64, 0:64] + 0.5
    ref = op * np.exp(-0.5 * ((xx - 32) ** 2 + (yy - 32) ** 2) / s2)
    ref = np.where(ref >= SR.ALPHA_MIN, np.minimum(ref, SR.ALPHA_MAX), 0.0)
    e = float(np.abs(a - ref).max())
    return {'passed': e <= 2e-4, 'value': e, 'tolerance': 2e-4}


def synthetic_walk(seconds: float = 2.0, rate: float = 10.0, seed: int = 0) -> tuple:
    """A pedestrian walking 1.4 m/s along +x for the given time with known joint angles (the gait prior plus seeded
    perturbations): (track, true angles (K, NDOF) at its knot times)."""
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, seconds + 1e-9, 1.0 / rate)
    c = np.stack([3.5 + 1.4 * t, np.full_like(t, 1.2), np.full_like(t, 0.9)], 1)
    q = np.tile(SP._yaw_quat(0.0), (len(t), 1))
    tr = SP.Track(1, 'pedestrian', 'human.pedestrian.adult', True, (0.4, 0.4, 1.8), t, c, q)
    tk, prior = walker_prior(tr)
    pert = rng.normal(0, 1, (1, NDOF)) * DOF_SIG * 0.8
    pert[:, :3] = 0.0
    pert[:, D_['yaw']] = 0.0
    truth = np.clip(prior + pert, DOF_LO, DOF_HI)
    return tr, tk, truth


def check_skeleton_fit(seed: int = 0) -> dict:
    """Fit a synthetic walk seen by three 1280 × 1024 fisheyes from 3.5–6.3 m (keypoints with 1.5 px noise) and
    compare the joint angles with the truth, and the gait prior's with it."""
    import torch
    tr, tk, truth = synthetic_walk(seed=seed)
    rng = np.random.default_rng(seed + 1)
    m = _fisheye(4.0)
    cams = {}
    for tag, yaw in (('F', 0.0), ('ML', 60.0), ('MR', -60.0)):
        from .mathutil import BODY_TO_OPTICAL, rot_zyx
        cams[tag] = SP.DriveCamera(tag, tag, m, rot_zyx(yaw, 5.0, 0.0) @ BODY_TO_OPTICAL.T, np.array([0.0, 0.0, 1.5]),
                                   [])
    traj = SP.Trajectory([0.0, 10.0], [[0, 0, 0], [0, 0, 0]], [[1, 0, 0, 0], [1, 0, 0, 0]])
    obs = []
    for j, t in enumerate(tr.t):
        th = torch.tensor(truth[min(j, len(truth) - 1)][None], dtype=torch.float64)
        _, _, kp = forward_kinematics(th, 1.8, 1.8)
        Rb, cb = _box_at(tr, t)
        Xw = kp[0].numpy() @ Rb.T + cb
        for tag, cam in cams.items():
            Xc = (Xw - cam.t) @ cam.R
            u, v, _, ok = m.project(Xc)
            if not ok.all() or u.min() < 0 or v.min() < 0 or u.max() > m.width or v.max() > m.height:
                continue
            k = np.stack([u + rng.normal(0, 1.5, 17), v + rng.normal(0, 1.5, 17), np.full(17, 0.9)], 1)
            obs.append((float(t), tag, k, float(v.max() - v.min())))
    st = NeuralSettings(fit_steps=400, prior_weight=0.2)
    _, fit, _, stats = fit_walker(tr, obs, cams, traj, st, None, tk, walker_prior(tr)[1])
    joints = [D_[k] for k in DOF if k not in ('tx', 'ty', 'tz', 'yaw')]
    err = np.degrees(np.abs(fit[:, joints] - truth[:, joints]))
    err0 = np.degrees(np.abs(walker_prior(tr)[1][:, joints] - truth[:, joints]))
    return {'passed': float(np.median(err)) <= 6.0 and float(np.median(err)) < float(np.median(err0)),
            'median_joint_error_deg': round(float(np.median(err)), 3),
            'prior_median_joint_error_deg': round(float(np.median(err0)), 3), 'observations': len(obs),
            'stats': stats}
