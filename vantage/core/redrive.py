"""Re-driving a recorded drive through a proposed rig, and how much of the existing data survives the change
(Simulate ▸ Log replay ▸ Re-drive). Built on the log replay's reconstruction (core.splat_reproject: the drive rebuilt
as 3D Gaussians from its lidar and images, moving actors in their boxes, rendered through any rig with Vantage's own
lens, body, rolling-shutter and lidar models); nothing of that is rebuilt here.

What it answers: "if we change the rig, do we lose our existing data?"

1. Re-shot dataset (export_dataset). The whole drive (or a span, at the recording's own frame times or at a chosen
   rate) re-rendered through the proposed variant and written as a nuScenes folder in the layout the CARLA bridge and
   the sample dataset write (core.dataset, vantage.sim.record): nuscenes/v1.0-vantage with the 13 tables, samples/ and
   sweeps/ per channel, the calibrated_sensor rows of the nuscenes_calib export (the variant's exact calibration,
   camera_intrinsic scaled to the image as the dataset does) and rig/sim_rig.json with each camera's exact lens block
   (what core.splat_reproject and core.detection_eval read), ego poses from the drive (the drive's own trajectory,
   interpolated at every stamp, and every recorded pose of the span as well), camera images (rolling shutter as the
   variant's modules) and lidar sweeps (each beam of the variant's lidar model cast against the Gaussians). The drive's
   3D boxes are in the world frame, so they carry over: nuScenes sample_annotation rows at the keyframes, and, where the
   source has a ground-truth index (core.groundtruth, a CARLA run) or ground_truth='always', a ground_truth/index.json
   with every frame's objects (the tight box of each object's re-rendered pixels, its occlusion and truncation, its 3D
   box in the vehicle frame), instance, range and semantic images (semantic: the annotated objects' classes and the own
   vehicle; everything else 0, unlabelled) and COCO files, so detection_eval scores the re-shot drive as it scores a
   CARLA run. Optionally a ROS 2 bag (core.rosbag) with the same frames. Every camera image gets its confidence map
   as a sidecar (confidence/<CHANNEL>/<name>.confidence.png: expected error, and .reason.png), indexed in
   confidence/index.json. Frames are rendered and written one at a time into <out>.part, renamed to <out> only when
   complete (a cancelled or failed export leaves nothing that looks complete).
2. Confidence of every re-rendered pixel, calibrated on evidence. Per pixel of a re-rendered image (at conf_scale of
   the image), features of how well the recording constrains it:
   * angle_deg: the novel-view angle, between this pixel's ray to its surface point and the nearest ray of a recorded
     training frame that saw the same point (a moving object's point in its own box frame);
   * extrap_m: the distance from this camera's centre to the nearest recorded camera centre that saw it (relative to
     the object for a moving one), or to any recorded centre when none saw it;
   * res_log2: log2 of the finest recorded pixel footprint on that point over this pixel's (positive: this camera
     sees it finer than the recording ever did);
   * views_log: ln(1 + the number of recorded frames that saw it); unseen: 1 when none did (the open sky included:
     a direction no recorded camera looked);
   * actor: the share of the pixel's opacity on a moving object; sky: 1 − its accumulated opacity (the sky texture);
   * spread: the opacity-weighted spread of the ranges along the ray over the range (semi-transparent or straddling
     Gaussians); lidar_log: ln(1 + the lidar points per sweep within 3σ of its Gaussians).
   Visibility of a point from a recorded frame: inside that frame's field, off its recording vehicle, and no farther
   than the frame's own re-rendered depth there (its 3 × 3 maximum, at most 128 px across) × 1.03 + 0.25 m (the
   training frames' depth maps are rendered once per scene and cached).
   Calibration (calibrate): views whose real image is known but which the reconstruction never saw: the held-out
   frames of the drive ('holdout'), every camera of a second recording of the same drive with another rig ('paired':
   the same scene, the cameras moved by a known offset, e.g. CARLA's same scenario and seed), and a recorded camera
   left out of a second reconstruction ('loo': its frames become novel views of the overlapping cameras). Per pixel,
   the measured error is the mean absolute difference of the image as an export writes it (rendered at the export's
   size, JPEG at its quality) and the real image, both area-averaged to conf_scale, in grey levels 0–255. The model: error ≈ c + Σₖ gₖ(featureₖ), each gₖ a monotone step
   function on quantile bins of its feature (additive isotonic regression by backfitting: Bacchetti 1989, "Additive
   isotonic models", JASA 84(405): 289–294; pool-adjacent-violators per feature), the directions those of the features'
   meaning (a larger novel-view angle never lowers the expected error; sky's direction is the one the data fits better),
   then the sum mapped to the measured error by one more isotonic regression (so E[error | prediction] = prediction on
   the calibration pixels), and per bin of the prediction the histogram of the measured errors (P(error ≤ b) for any
   budget b, made monotone the same way). Every bin edge and step is the fit's, stored with the result; nothing is a
   chosen weight. The fit is also cross-fitted over two folds of 1 s time blocks: the out-of-fold predictions are what
   the reliability table reports (predicted against measured per decile of the prediction).
3. "Do we lose our data?" (run): at report_instants instants the proposed rig's cameras are rendered with their
   features; per pixel P(error ≤ error_budget); per camera the expected share of its field within budget (the mean of
   P), the rest split by the dominant reason of each pixel ('unseen' where no recorded frame saw its point, else the
   feature group with the largest excess of its step function over its best step: unseen ('never seen by the old rig'), viewpoint ('too far from any recorded viewpoint':
   angle, distance, resolution), moving ('moving objects'), sky, geometry (thin or poorly rebuilt geometry: spread,
   lidar support) and floor (the reconstruction's own floor: no feature is worse than its best step)), so the shares
   sum to one. Per region of regions.default_grid (each pixel's surface point in the vehicle frame) the same, as a polar
   map, and image overlays. The verdict per camera: reuse (share ≥ reuse_share), re-collect (share < collect_share) or
   partial, with the reason. The inverse: the share of what the recorded rig saw at the same instants (its pixels on the
   Gaussians) that no camera of the proposed rig sees at any of them, per region.
4. Nudge (nudge_view): one camera of the proposed rig moved (Δx, Δy, Δz, Δyaw, Δpitch, another lens) and re-rendered
   with its confidence on the cached scene; the page renders at a reduced scale while dragging.

Limits: the confidence is calibrated on the views the calibration had: novelty beyond what they span (a camera
metres away from every recorded one) is extrapolated by the last step of each feature's fit, which is optimistic; it
says how well the reconstruction reproduces the recording's own look (exposure, colour, blur), not whether a detector
would behave the same; the error is of images area-averaged to conf_scale (fine texture the Gaussians blur is in the
floor); the confidence maps are of a global-shutter render at the frame's middle-row time. Radars are not re-rendered
(the reconstruction has no radar model): the re-shot dataset has no radar channels.
"""
from __future__ import annotations

import copy
import csv
import dataclasses
import datetime as _dt
import hashlib
import io
import json
import math
import os
import re
import shutil
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from . import splat_reproject as SP
from .mathutil import BODY_TO_OPTICAL, rot_zyx

SCHEMA = 'vantage.redrive/1'
KEY = 'redrive'
CONF_SCHEMA = 'vantage.redrive_confidence/1'
CALIB_SCHEMA = 'vantage.redrive_calibration/1'
EXPORT_SCHEMA = 'vantage.redrive_export/1'
VERSION = 'v1.0-vantage'
MARKER = '.vantage-dataset'
FRAME = 'vehicle (ISO 8855, rear axle on the ground); world: the drive\'s global frame'
FAR_M = 1e4                                   # the surface point of a sky pixel (a direction)
VIS_REL, VIS_ABS = 1.03, 0.25                 # a recorded frame sees a point no farther than its depth × 1.03 + 0.25 m
EV_SIDE = 128                                 # evidence depth maps: at most this many pixels across
CONF_SCALE_PNG = 256.0                        # confidence PNG: uint16 = round(expected error · 256)
BLOCK_S = 1.0                                 # cross-fitting folds: 1 s time blocks, alternating
ERR_EDGES = np.array([0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 64, 96, 128, 256], float)
# (feature, monotone direction: +1 error grows with it, −1 falls, 0 the better fitting one, unit, reason group, text)
FEATURES = (
    ('angle_deg', +1, '°', 'viewpoint', 'novel-view angle to the nearest recorded ray that saw the point'),
    ('extrap_m', +1, 'm', 'viewpoint', 'distance to the nearest recorded camera centre that saw the point'),
    ('res_log2', +1, 'log2', 'viewpoint', 'finest recorded pixel footprint over this pixel\'s'),
    ('views_log', -1, 'ln(1+n)', 'unseen', 'recorded frames that saw the point'),
    ('unseen', +1, 'share', 'unseen', 'no recorded frame saw the point (or the sky direction)'),
    ('actor', +1, 'share', 'moving', 'share of the pixel on a moving object'),
    ('sky', 0, 'share', 'sky', 'share of the pixel that is the sky texture'),
    ('spread', +1, 'relative', 'geometry', 'spread of the ranges along the ray over the range'),
    ('lidar_log', -1, 'ln(1+pts)', 'geometry', 'lidar points per sweep on the pixel\'s Gaussians'),
)
FEATURE_NAMES = tuple(f[0] for f in FEATURES)
REASONS = ('ok', 'unseen', 'viewpoint', 'moving', 'sky', 'geometry', 'floor')
REASON_TEXT = {'ok': 'within budget', 'unseen': 'never seen by the old rig',
               'viewpoint': 'too far from any recorded viewpoint', 'moving': 'moving objects', 'sky': 'sky',
               'geometry': 'thin or poorly rebuilt geometry', 'floor': 'the reconstruction\'s own floor'}
REASON_RGB = {'ok': (40, 160, 90), 'unseen': (220, 60, 60), 'viewpoint': (240, 150, 40), 'moving': (170, 90, 220),
              'sky': (90, 160, 240), 'geometry': (200, 200, 60), 'floor': (140, 140, 140)}
SOURCES = ('holdout', 'paired', 'loo')
SCHEDULES = ('recorded', 'rate')
GT_MODES = ('auto', 'always', 'never')
LIMITS = {'t_start': (0.0, 1e5), 't_end': (0.0, 1e5), 'rate_hz': (0.1, 100.0), 'image_scale': (0.0, 1.0),
          'jpeg_quality': (30, 100), 'keyframe_hz': (0.1, 20.0), 'conf_scale': (0.05, 1.0),
          'error_budget': (0.5, 128.0),
          'reuse_share': (0.0, 1.0), 'collect_share': (0.0, 1.0), 'calib_views': (4, 2000),
          'calib_pixels': (100, 100000), 'report_instants': (1, 200), 'max_views': (8, 5000), 'bins': (4, 64),
          'seed': (0, 2 ** 31 - 1), 'val_frames': (1, 2000)}
INTS = ('jpeg_quality', 'calib_views', 'calib_pixels', 'report_instants', 'max_views', 'bins', 'seed', 'val_frames')
BOOLS = ('lidar', 'rosbag')
ASSUMPTIONS = (
    'A re-shot drive is the log replay\'s reconstruction re-rendered through the proposed rig: not a real recording. '
    'Its confidence is the expected error of each re-rendered pixel (grey levels 0–255, images area-averaged to the '
    'confidence scale), calibrated on views whose real image is known and which the reconstruction never saw '
    '(held-out frames, a second recording of the same drive, a camera left out); beyond the novelty those views span '
    'it is extrapolated by the last step of each feature\'s fit. It measures how well the recording\'s own look is '
    'reproduced, not how a detector behaves. Radars are not re-rendered.')


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
class RedriveSettings:
    variant: str = ''                  # the proposed variant (uid or name; '' : the log replay's rig B, else active)
    t_start: float = 0.0               # s from the drive's first ego pose
    t_end: float = 0.0                 # 0: to the end of the drive
    schedule: str = 'recorded'         # recorded: the drive's own frame and sweep times; rate: every 1 / rate_hz
    rate_hz: float = 10.0
    image_scale: float = 0.0           # share of each module's sensor size; 0: the drive's own (its dataset's)
    rolling_shutter: str = 'module'    # module: the variant's modules' readout; off: one instant per frame
    lidar: bool = True
    rosbag: bool = False
    ground_truth: str = 'auto'         # auto: where the source has a ground-truth index; always; never
    jpeg_quality: int = 90
    keyframe_hz: float = 2.0
    conf_scale: float = 0.25           # confidence maps and the report at this share of the image size
    error_budget: float = 12.0         # grey levels (0–255): a pixel is reusable within this mean absolute error
    reuse_share: float = 0.8           # a camera is reusable when at least this share of its field is within budget
    collect_share: float = 0.5         # … and needs re-collecting below this share
    sources: list = field(default_factory=lambda: ['holdout', 'paired'])
    paired_drive: str = ''             # a second recording of the same drive (another rig)
    loo_camera: str = ''               # a recorded camera left out of a second reconstruction ('' : none)
    calib_views: int = 160             # calibration views at most (spread over the sources)
    calib_pixels: int = 3000           # pixels sampled per calibration view
    bins: int = 20                     # quantile bins per feature, at most
    max_views: int = 720               # recorded training frames kept as evidence, at most
    report_instants: int = 12
    val_frames: int = 60               # frames of the paired recording scored, at most
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d) -> 'RedriveSettings':
        """Tolerant: unknown keys dropped, numbers clamped to LIMITS, words checked, junk back to the default."""
        s = RedriveSettings()
        d = d if isinstance(d, dict) else {}
        for k in ('variant', 'paired_drive', 'loo_camera'):
            if isinstance(d.get(k), str):
                setattr(s, k, d[k].strip()[:4096])
        for k, opts in (('schedule', SCHEDULES), ('rolling_shutter', SP.ROLLING), ('ground_truth', GT_MODES)):
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
        src = d.get('sources')
        if isinstance(src, (list, tuple)):
            s.sources = [x for x in SOURCES if x in src]
        if s.collect_share > s.reuse_share:
            s.collect_share = s.reuse_share
        if s.t_end and s.t_end < s.t_start:
            s.t_end = 0.0
        return s


def settings_of(project) -> RedriveSettings:
    return RedriveSettings.from_dict(getattr(project.analysis, KEY, None) if project is not None else None)


def proposed_variant(project, st: RedriveSettings, sst: SP.SplatSettings | None = None):
    """The variant re-shot: settings.variant, else the log replay's rig B, else the active one."""
    if project is None:
        return None
    for key in (st.variant, (sst.variant_b if sst is not None else '')):
        v = SP._variant_by(project, key) if key else None
        if v is not None:
            return v
    return project.variant


# =========================================================================== evidence: the recorded training frames
@dataclass
class Evidence:
    """The recorded training frames as the confidence sees them: per frame its camera, time, centre (local world
    frame), rotation optical → world, focal length at its recorded size, its re-rendered depth map (inf: open sky) and
    its own vehicle's pixels; per Gaussian the lidar points per sweep within 3σ."""
    key: str
    tags: np.ndarray                    # (V,) camera tag per view
    t: np.ndarray                       # (V,)
    o: np.ndarray                       # (V, 3)
    R: np.ndarray                       # (V, 3, 3) optical -> world
    f: np.ndarray                       # (V,) fx at the recorded image size
    models: dict                        # tag -> CamModel at the depth-map size
    depth: dict                         # tag -> (n, h, w) float32, 3 × 3 maximum of the range, inf: sky
    ego: dict                           # tag -> (n, h, w) bool
    index: dict                         # tag -> (n,) indices into the views
    lidar: np.ndarray                   # (N,) points per sweep on each Gaussian
    info: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(len(self.t))

    def save(self, path):
        arrs = {'tags': np.asarray(self.tags, str), 't': self.t, 'o': self.o, 'R': self.R, 'f': self.f,
                'lidar': self.lidar.astype(np.float32)}
        for tag in self.models:
            arrs[f'depth__{tag}'] = self.depth[tag].astype(np.float16)
            arrs[f'ego__{tag}'] = np.packbits(self.ego[tag], axis=-1)
            arrs[f'index__{tag}'] = self.index[tag]
        meta = {'key': self.key, 'info': self.info, 'ego_w': {t: int(self.ego[t].shape[-1]) for t in self.ego},
                'models': {t: dataclasses.asdict(m) for t, m in self.models.items()}}
        tmp = path + '.part.npz'
        np.savez_compressed(tmp, meta=np.frombuffer(json.dumps(meta, default=float).encode(), np.uint8), **arrs)
        os.replace(tmp, path)

    @staticmethod
    def load(path) -> 'Evidence | None':
        try:
            with np.load(path, allow_pickle=False) as z:
                meta = json.loads(bytes(z['meta']).decode())
                models = {t: SP.CamModel(**{**m, 'kb': tuple(m['kb'])}) for t, m in meta['models'].items()}
                dep, ego, idx = {}, {}, {}
                for t in models:
                    dep[t] = z[f'depth__{t}'].astype(np.float32)
                    ego[t] = np.unpackbits(z[f'ego__{t}'], axis=-1)[..., :meta['ego_w'][t]].astype(bool)
                    idx[t] = z[f'index__{t}']
                return Evidence(meta['key'], z['tags'].astype(str), z['t'], z['o'], z['R'], z['f'], models, dep, ego,
                                idx, z['lidar'].astype(np.float32), dict(meta.get('info') or {}))
        except (OSError, ValueError, KeyError):
            return None


def _ev_size(model: SP.CamModel):
    s = min(1.0, EV_SIDE / max(model.width, model.height))
    return max(8, int(round(model.width * s))), max(8, int(round(model.height * s)))


def evidence_key(drive: SP.Drive, sst: SP.SplatSettings, scene: SP.Scene, max_views: int) -> str:
    h = hashlib.sha256(json.dumps({'drive': drive.key, 'build': sst.build_key(), 'n': scene.n,
                                   'cams': sorted(drive.cameras), 'max_views': int(max_views), 'v': 1},
                                  sort_keys=True, default=str).encode())
    h.update(np.ascontiguousarray(scene.means[:: max(1, scene.n // 997)]).tobytes())
    return h.hexdigest()[:16]


def build_evidence(scene: SP.Scene, drive: SP.Drive, sst: SP.SplatSettings, project=None, engine=None, dev=None,
                   max_views: int = 720, progress=None, cancel=None) -> Evidence:
    """The evidence of the scene's training frames (module docstring, 2)."""
    import torch
    from scipy.ndimage import maximum_filter
    stop = cancel or (lambda: False)
    dev = dev or SP.SR.device_for(sst.device, sst.gpu_mem_gb)
    frames = SP._frames_for(drive, sst, True, 1)
    if len(frames) > max_views:
        frames = [frames[int(i)] for i in np.linspace(0, len(frames) - 1, max_views).round().astype(int)]
    ego_fn = SP._ego_mask_fn(drive, project, engine)
    ts = SP.TorchScene(scene, drive, dev)
    tags, tt, oo, RR, ff = [], [], [], [], []
    models, dep, ego, idx = {}, {}, {}, {}
    with torch.no_grad():
        for j, (cam, k, fr) in enumerate(frames):
            if stop():
                raise Cancelled()
            if j % 10 == 0:
                _prog(progress, 0.8 * j / max(len(frames), 1), f'Evidence · {cam.tag} frame {k + 1}')
            w, h = _ev_size(cam.model)
            m = cam.model.scaled(w, h)
            r = SP.render_view(ts, drive.traj, m, cam.R, cam.t, fr[0], 0.0, depth=True)
            d = r['depth'].cpu().numpy().astype(np.float32)
            d[r['alpha'].cpu().numpy() < 0.5] = np.inf
            d[~r['field']] = np.inf
            d = maximum_filter(d, size=3, mode='nearest')
            e = ego_fn(cam, fr, (h, w))
            e = np.zeros((h, w), bool) if e is None else np.asarray(e, bool)
            Rw, pw = drive.traj.pose(fr[0])
            models.setdefault(cam.tag, m)
            dep.setdefault(cam.tag, []).append(d)
            ego.setdefault(cam.tag, []).append(e)
            idx.setdefault(cam.tag, []).append(len(tags))
            tags.append(cam.tag)
            tt.append(fr[0])
            oo.append(Rw @ cam.t + pw)
            RR.append(Rw @ cam.R)
            ff.append(cam.model.fx)
    if not tags:
        raise ValueError('The drive has no training frames: nothing to calibrate the confidence on.')
    lid = lidar_support(scene, drive, sst, cancel=stop,
                        progress=lambda f, t='': _prog(progress, 0.8 + 0.2 * f, t))
    ev = Evidence(evidence_key(drive, sst, scene, max_views), np.asarray(tags), np.asarray(tt, float),
                  np.asarray(oo, float), np.asarray(RR, float), np.asarray(ff, float), models,
                  {t: np.stack(v) for t, v in dep.items()}, {t: np.stack(v) for t, v in ego.items()},
                  {t: np.asarray(v, np.int64) for t, v in idx.items()}, lid,
                  {'views': len(tags), 'per_camera': {t: len(v) for t, v in idx.items()},
                   'depth_map_px': {t: [m.width, m.height] for t, m in models.items()}})
    _prog(progress, 1.0, 'Evidence')
    return ev


def lidar_support(scene: SP.Scene, drive: SP.Drive, sst: SP.SplatSettings, max_sweeps: int = 30, progress=None,
                  cancel=None) -> np.ndarray:
    """(N,) lidar points per training sweep within 3σ (largest axis, at least 2 cm) of each Gaussian's centre: static
    Gaussians from the points in the world, an actor's from the points inside its box in its box frame."""
    from scipy.spatial import cKDTree
    stop = cancel or (lambda: False)
    n = scene.n
    out = np.zeros(n, float)
    jobs = [(L, sw) for L in drive.lidars.values() for k, sw in enumerate(L.sweeps)
            if not SP.held_out(k, sst.holdout_every) and not k % sst.sweep_step]
    if not jobs or not n:
        return out.astype(np.float32)
    jobs = [jobs[int(i)] for i in np.unique(np.linspace(0, len(jobs) - 1, min(max_sweeps, len(jobs))).round())]
    smax = np.maximum(3.0 * scene.scales.max(1).astype(float), 0.02)
    static = np.flatnonzero(scene.actor < 0)
    tree_s = cKDTree(scene.means[static].astype(float)) if len(static) else None
    trees = {}
    for a, i in enumerate(scene.actor_ids):
        ix = np.flatnonzero(scene.actor == a)
        if len(ix):
            trees[i] = (ix, cKDTree(scene.means[ix].astype(float)))
    for j, (L, sw) in enumerate(jobs):
        if stop():
            raise Cancelled()
        _prog(progress, j / len(jobs), f'Lidar support · sweep {j + 1}/{len(jobs)}')
        try:
            a = SP.load_sweep(sw)
        except (OSError, ValueError):
            continue
        rr = np.linalg.norm(a[:, :3], axis=1)
        a = a[(rr > 0.5) & (rr <= sst.max_range_m) & np.isfinite(rr)]
        if not len(a):
            continue
        tf = SP.firing_times(L, sw, a)
        Rw, pw = drive.traj.at(tf)
        Pw = np.einsum('nij,nj->ni', Rw, a[:, :3].astype(float) @ L.R.T + L.t) + pw
        taken = np.zeros(len(Pw), bool)
        tm = float(np.median(tf))
        for i, (ix, tree) in trees.items():
            tr = drive.tracks.get(i)
            po = tr.pose(tm) if tr is not None else None
            if po is None:
                continue
            Rb, cb = po
            q = (Pw - cb) @ Rb
            inside = np.all(np.abs(q) <= np.asarray(tr.size) / 2 + 0.25, 1)
            if not inside.any():
                continue
            d, k = tree.query(q[inside], k=1)
            ok = d <= smax[ix[k]]
            np.add.at(out, ix[k[ok]], 1.0)
            taken |= inside
        if tree_s is not None:
            d, k = tree_s.query(Pw[~taken], k=1)
            ok = d <= smax[static[k]]
            np.add.at(out, static[k[ok]], 1.0)
    return (out / len(jobs)).astype(np.float32)


def evidence_for(scene, drive, sst, project=None, engine=None, dev=None, cache_root=None, max_views=720,
                 progress=None, cancel=None) -> Evidence:
    """The scene's evidence, from the scene's cache folder when it was built for the same scene and settings."""
    key = evidence_key(drive, sst, scene, max_views)
    path = None
    if cache_root:
        folder = SP.cache_folder(cache_root, drive, sst)
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f'redrive_evidence-{key}.npz')
        ev = Evidence.load(path) if os.path.isfile(path) else None
        if ev is not None and ev.key == key and len(ev.lidar) == scene.n:
            ev.info['cached'] = True
            return ev
    ev = build_evidence(scene, drive, sst, project, engine, dev, max_views, progress, cancel)
    if path:
        ev.save(path)
    ev.info['cached'] = False
    return ev


# =========================================================================== per-pixel features of one view
def _posed_actor_frames(scene: SP.Scene, drive: SP.Drive, t: float):
    """Per actor slot (scene.actor_ids order, then one for the static scene): (R world ← box, centre, shown) at t."""
    na = len(scene.actor_ids)
    R = np.tile(np.eye(3), (na + 1, 1, 1))
    c = np.zeros((na + 1, 3))
    shown = np.zeros(na + 1, bool)
    shown[-1] = True
    for a, i in enumerate(scene.actor_ids):
        tr = drive.tracks.get(i)
        po = tr.pose(t) if tr is not None else None
        if po is not None:
            R[a], c[a] = po
            shown[a] = True
    return R, c, shown


def render_features(ts: SP.TorchScene, scene: SP.Scene, ev: Evidence, drive: SP.Drive, model: SP.CamModel, R_cv,
                    t_cv, t: float, f_ref: float, body=None, chunk: int = 0) -> dict:
    """One camera at drive time t (global shutter at t, the frame's middle row) with every feature of the module
    docstring per pixel. R_cv: optical → vehicle, t_cv: its centre (vehicle frame), f_ref: the focal length (px) of
    the image the confidence is for (the pixel footprint's denominator), body: (h, w) own-vehicle pixels (no
    features: left out). Returns {'rgb' (h, w, 3) float [0, 1], 'alpha', 'depth', 'field', 'valid', 'X_v' (h, w, 3)
    surface points in the vehicle frame (NaN for the sky), 'index', 'actor_px', <feature> (h, w) float}."""
    import torch
    dev = ts.dev
    w, h = model.width, model.height
    D, okr = SP._rays_of(model)
    Rw, pw = drive.traj.pose(t)
    o = Rw @ np.asarray(t_cv, float) + pw
    O2W = Rw @ np.asarray(R_cv, float)
    lid = torch.tensor(np.log1p(ev.lidar), dtype=torch.float32, device=dev)
    act = (ts.actor >= 0).float()
    with torch.no_grad():
        Wt = torch.tensor(O2W.T, dtype=torch.float32, device=dev)
        P, Rg, shown = ts.posed(t)
        Pc = (P - torch.tensor(o, dtype=torch.float32, device=dev)) @ Wt.T
        s = torch.exp(ts.log_s)
        op = torch.sigmoid(ts.op_logit)
        col = torch.sigmoid(ts.col_logit)
        cov = SP.SR.covariance(Wt[None] @ Rg, s)
        rr = torch.linalg.norm(Pc, dim=1, keepdim=True)
        feats = torch.cat([col, rr, rr, rr * rr, act[:, None], lid[:, None]], 1)
        nrm = SP._normals(Wt[None] @ Rg, s, Pc)
        Dt = torch.tensor(D.reshape(-1, 3), dtype=torch.float32, device=dev)
        bg = torch.cat([ts.sky_at(Dt @ Wt), torch.zeros(len(Dt), 5, device=dev)], 1)
        out = SP.SR.splat(Pc, cov, feats, torch.where(shown, op, torch.zeros_like(op)), model.projector(), w, h,
                          bg=bg, want_index=True, normals=nrm, depth_col=3)
        img = out['image'].cpu().numpy().reshape(h, w, -1)
        alpha = out['alpha'].cpu().numpy().reshape(h, w)
        index = out['index'].cpu().numpy().reshape(h, w)
    a_ = np.maximum(alpha, 1e-6)
    depth = img[..., 3] / a_
    er, er2 = img[..., 4] / a_, img[..., 5] / a_
    spread = np.sqrt(np.maximum(er2 - er * er, 0.0)) / np.maximum(er, 1e-3)
    actor_sh = img[..., 6] / a_
    lid_px = img[..., 7] / a_
    surf = (alpha >= 0.5) & okr
    Dw = D @ O2W.T
    X = o + Dw * np.where(surf, depth, FAR_M)[..., None]
    # the pixel's object: its dominant Gaussian's actor slot (static: the last slot)
    na = len(scene.actor_ids)
    a_slot = np.full((h, w), na, np.int64)
    gi = np.where(surf & (index >= 0), index, -1)
    m = gi >= 0
    if m.any() and na:
        ga = scene.actor[gi[m]]
        a_slot[m] = np.where(ga >= 0, ga, na)
    valid = okr.copy()
    if body is not None:
        valid &= ~np.asarray(body, bool)
    feats_px = _evidence_features(scene, ev, drive, X[valid], a_slot[valid], o, t, f_ref, dev, chunk)
    res = {'rgb': np.clip(img[..., :3], 0, 1), 'alpha': alpha, 'depth': np.where(surf, depth, np.nan),
           'field': okr, 'valid': valid, 'index': index}
    for k in ('angle_deg', 'extrap_m', 'res_log2', 'views_log', 'unseen'):
        arr = np.full((h, w), np.nan, np.float32)
        arr[valid] = feats_px[k]
        res[k] = arr
    res['actor'] = np.where(valid, np.clip(actor_sh, 0, 1), np.nan).astype(np.float32)
    res['sky'] = np.where(valid, np.clip(1.0 - alpha, 0, 1), np.nan).astype(np.float32)
    res['spread'] = np.where(valid, np.clip(spread, 0, 10), np.nan).astype(np.float32)
    res['lidar_log'] = np.where(valid, np.where(surf, lid_px, 0.0), np.nan).astype(np.float32)
    Xv = (X - pw) @ Rw
    Xv[~surf] = np.nan
    res['X_v'] = Xv
    res['actor_px'] = (a_slot < na) & surf
    return res


def _evidence_features(scene, ev: Evidence, drive, X, a_slot, o, t, f_ref, dev, chunk=0) -> dict:
    """Per pixel (n): visibility from every evidence view and the novel-view angle, the extrapolation distance, the
    resolution ratio and the number of views (module docstring, 2)."""
    import torch
    n = len(X)
    out = {'angle_deg': np.full(n, 180.0), 'extrap_m': np.zeros(n), 'res_log2': np.zeros(n),
           'views_log': np.zeros(n), 'unseen': np.ones(n)}
    if n == 0:
        return out
    na = len(scene.actor_ids)
    Rt, ct, _ = _posed_actor_frames(scene, drive, t)
    # the point in its object's frame (world for the static scene), and this camera's centre there
    Xb = np.einsum('nji,nj->ni', Rt[a_slot], X - ct[a_slot])
    Cp = np.einsum('nji,nj->ni', Rt[a_slot], o - ct[a_slot])
    vp = Cp - Xb
    rp = np.linalg.norm(vp, axis=1)
    fp_p = rp / max(float(f_ref), 1e-6)
    T = lambda a, dt=torch.float32: torch.as_tensor(np.asarray(a), dtype=dt, device=dev)  # noqa: E731
    Xb_t, Cp_t, vp_t = T(Xb), T(Cp), T(vp / np.maximum(rp, 1e-9)[:, None])
    slot_t = T(a_slot, torch.long)
    present = [a for a in np.unique(a_slot) if a < na]
    nv = torch.zeros(n, device=dev)
    best = torch.full((n,), -1.0, device=dev)
    ex = torch.full((n,), float('inf'), device=dev)
    near_any = torch.full((n,), float('inf'), device=dev)
    fpmin = torch.full((n,), float('inf'), device=dev)
    if not chunk:
        chunk = max(1, min(64, int(3e6 // max(n, 1))))
    with torch.no_grad():
        for tag, vix in ev.index.items():
            m = ev.models[tag]
            proj = m.projector()
            dep_all = T(ev.depth[tag])
            ego_all = T(ev.ego[tag], torch.bool)
            for j0 in range(0, len(vix), chunk):
                vj = vix[j0:j0 + chunk]
                J = len(vj)
                # per view: every actor's pose at the view's time (the static scene: identity)
                Rv = np.tile(np.eye(3), (J, na + 1, 1, 1))
                cv = np.zeros((J, na + 1, 3))
                sv = np.ones((J, na + 1), bool)
                for q, jj in enumerate(vj):
                    if present:
                        R_, c_, s_ = _posed_actor_frames(scene, drive, float(ev.t[jj]))
                        Rv[q], cv[q], sv[q] = R_, c_, s_
                Xj = Xb_t[None].expand(J, n, 3).clone()
                for a in present:
                    sel = torch.nonzero(slot_t == int(a)).squeeze(1)
                    Ra, ca = T(Rv[:, a]), T(cv[:, a])
                    Xj[:, sel] = torch.einsum('jab,nb->jna', Ra, Xb_t[sel]) + ca[:, None]
                oj = T(ev.o[vj])
                d = Xj - oj[:, None]
                Q = torch.einsum('jnk,jkl->jnl', d, T(ev.R[vj]))
                u, v, ok = proj(Q.reshape(-1, 3))
                u, v, ok = u.reshape(J, n), v.reshape(J, n), ok.reshape(J, n)
                ok = ok & (u >= 0) & (u < m.width) & (v >= 0) & (v < m.height)
                ui = u.clamp(0, m.width - 1).long()
                vi = v.clamp(0, m.height - 1).long()
                jl = torch.as_tensor(np.arange(j0, j0 + J), device=dev)[:, None].expand(J, n)
                dep = dep_all[jl, vi, ui]
                eg = ego_all[jl, vi, ui]
                r = torch.linalg.norm(d, dim=-1)
                vis = ok & ~eg & (r <= dep * VIS_REL + VIS_ABS)
                if present:
                    vis = vis & T(sv, torch.bool)[:, slot_t]
                # the recorded camera's centre in the point's object frame
                Cobj = torch.einsum('jaki,jak->jai', T(Rv), oj[:, None, :] - T(cv))[:, slot_t] if present else \
                    oj[:, None, :].expand(J, n, 3)
                vjv = Cobj - Xb_t[None]
                rj = torch.linalg.norm(vjv, dim=-1).clamp(min=1e-9)
                cs = (vjv * vp_t[None]).sum(-1) / rj
                dist = torch.linalg.norm(Cobj - Cp_t[None], dim=-1)
                fpj = rj / T(ev.f[vj])[:, None]
                nv += vis.float().sum(0)
                best = torch.maximum(best, torch.where(vis, cs, torch.full_like(cs, -1.0)).max(0).values)
                inf = torch.full_like(dist, float('inf'))
                ex = torch.minimum(ex, torch.where(vis, dist, inf).min(0).values)
                near_any = torch.minimum(near_any, dist.min(0).values)
                fpmin = torch.minimum(fpmin, torch.where(vis, fpj, inf).min(0).values)
    nv = nv.cpu().numpy()
    seen = nv > 0
    out['angle_deg'] = np.where(seen, np.degrees(np.arccos(np.clip(best.cpu().numpy(), -1, 1))), 180.0)
    out['extrap_m'] = np.where(seen, ex.cpu().numpy(), near_any.cpu().numpy())
    fm = fpmin.cpu().numpy()
    with np.errstate(divide='ignore', invalid='ignore'):
        rl = np.log2(np.where(seen, fm, 1.0) / np.maximum(fp_p, 1e-12))
    out['res_log2'] = np.where(seen, np.clip(rl, -6, 6), 0.0)
    out['views_log'] = np.log1p(nv)
    out['unseen'] = (~seen).astype(float)
    return out


def feature_matrix(fe: dict, mask=None) -> np.ndarray:
    """(n, F) features of the pixels in mask (default: fe['valid'])."""
    m = fe['valid'] if mask is None else mask
    return np.stack([np.asarray(fe[k], float)[m] for k in FEATURE_NAMES], 1)


# =========================================================================== the error model
def pava(y, w, increasing=True) -> np.ndarray:
    """Weighted isotonic regression of y in its order (pool adjacent violators)."""
    y = np.asarray(y, float)
    w = np.asarray(w, float)
    if not increasing:
        return -pava(-y, w, True)
    vals, wts, cnt = [], [], []
    for yi, wi in zip(y, w):
        vals.append(yi)
        wts.append(wi)
        cnt.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v2, w2, c2 = vals.pop(), wts.pop(), cnt.pop()
            v1, w1, c1 = vals.pop(), wts.pop(), cnt.pop()
            ww = w1 + w2
            vals.append((v1 * w1 + v2 * w2) / ww if ww > 0 else 0.5 * (v1 + v2))
            wts.append(ww)
            cnt.append(c1 + c2)
    return np.repeat(vals, cnt)


def _quantile_edges(x, bins, min_count):
    """Inner bin edges: quantiles of x, merged until every bin holds at least min_count values."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return np.zeros(0)
    e = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)[1:-1]))
    while len(e):
        k = np.searchsorted(e, x, side='right')
        c = np.bincount(k, minlength=len(e) + 1)
        if c.min() >= min_count:
            break
        i = int(np.argmin(c))
        e = np.delete(e, min(max(i - (1 if i == len(e) else 0), 0), len(e) - 1))
    return e


@dataclass
class ErrorModel:
    """Expected per-pixel error from the features (module docstring, 2): an additive isotonic model, then an isotonic
    map of its sum to the measured error and the measured errors' histogram per bin of the sum."""
    features: list
    directions: list                    # per feature: +1 / −1 (as fitted)
    edges: list                         # per feature: inner bin edges
    steps: list                         # per feature: the step per bin (centred: their count-weighted mean is 0)
    c: float
    cal_edges: list                     # inner bin edges of the raw sum
    cal_mean: list                      # E[error | bin], isotonic
    cal_hist: list                      # per bin: counts of the errors in ERR_EDGES bins
    n: int = 0
    iterations: int = 0

    def _bins(self, X):
        X = np.asarray(X, float)
        return [np.searchsorted(np.asarray(self.edges[k]), np.nan_to_num(X[:, k], nan=0.0), side='right')
                for k in range(len(self.features))]

    def raw(self, X) -> np.ndarray:
        B = self._bins(X)
        return self.c + sum(np.asarray(self.steps[k])[B[k]] for k in range(len(B)))

    def _cal_bin(self, raw):
        return np.searchsorted(np.asarray(self.cal_edges), raw, side='right')

    def expected(self, X) -> np.ndarray:
        """Expected mean absolute error (grey levels) per row."""
        return np.asarray(self.cal_mean)[self._cal_bin(self.raw(X))]

    def p_within_table(self, budget: float) -> np.ndarray:
        """P(error ≤ budget) per bin of the raw sum, non-increasing over the bins (isotonic)."""
        H = np.asarray(self.cal_hist, float)
        e = ERR_EDGES
        # the share at or under the budget: whole ERR_EDGES bins below it, the bin it falls in linearly
        cum = np.concatenate([np.zeros((len(H), 1)), np.cumsum(H, 1)], 1)
        tot = np.maximum(cum[:, -1], 1e-9)
        k = int(np.clip(np.searchsorted(e, budget, side='right') - 1, 0, len(e) - 2))
        frac = (budget - e[k]) / (e[k + 1] - e[k]) if budget < e[-1] else 1.0
        p = (cum[:, k] + frac * H[:, k]) / tot
        p = np.clip(p, 0, 1)
        return pava(p, tot, increasing=False)

    def p_within(self, X, budget: float) -> np.ndarray:
        return self.p_within_table(budget)[self._cal_bin(self.raw(X))]

    def excess(self, X) -> np.ndarray:
        """(n, F): each feature's step above its best step (grey levels of expected error it adds)."""
        B = self._bins(X)
        return np.stack([np.asarray(self.steps[k])[B[k]] - np.min(self.steps[k]) for k in range(len(B))], 1)

    def reasons(self, X) -> np.ndarray:
        """Reason index (REASONS) per row: 'unseen' where no recorded frame saw the point, else the feature group
        with the largest excess, 'floor' when no feature is above its best step."""
        E = self.excess(X)
        groups = [g for g in REASONS if g not in ('ok', 'floor')]
        G = np.stack([E[:, [k for k, f in enumerate(FEATURES) if f[3] == g]].sum(1) for g in groups], 1)
        r = np.array([REASONS.index(g) for g in groups])[np.argmax(G, 1)]
        r = np.where(G.max(1) > 1e-9, r, REASONS.index('floor'))
        # a point no recorded frame saw is 'never seen by the old rig' by definition, whatever the fit's steps
        return np.where(np.asarray(X, float)[:, FEATURE_NAMES.index('unseen')] >= 0.5, REASONS.index('unseen'), r)

    def to_dict(self) -> dict:
        r = lambda a: [float(x) for x in a]  # noqa: E731  (full precision: an edge moved moves values across it)
        return {'features': list(self.features), 'directions': [int(d) for d in self.directions],
                'edges': [r(e) for e in self.edges], 'steps': [r(s) for s in self.steps], 'c': float(self.c),
                'cal_edges': r(self.cal_edges), 'cal_mean': r(self.cal_mean),
                'cal_hist': [[int(x) for x in h] for h in self.cal_hist], 'err_edges': r(ERR_EDGES),
                'n': int(self.n), 'iterations': int(self.iterations)}

    @staticmethod
    def from_dict(d) -> 'ErrorModel | None':
        try:
            if list(d['features']) != list(FEATURE_NAMES):
                return None
            return ErrorModel(list(d['features']), [int(x) for x in d['directions']],
                              [np.asarray(e, float) for e in d['edges']], [np.asarray(s, float) for s in d['steps']],
                              float(d['c']), list(map(float, d['cal_edges'])), list(map(float, d['cal_mean'])),
                              [list(map(int, h)) for h in d['cal_hist']], int(d.get('n', 0)),
                              int(d.get('iterations', 0)))
        except (KeyError, TypeError, ValueError):
            return None


def fit_error_model(X, y, bins: int = 20, iters: int = 40, min_count: int | None = None) -> ErrorModel:
    """Additive isotonic regression of y on the features X (n, F) by backfitting, then the calibration map
    (module docstring, 2)."""
    X = np.nan_to_num(np.asarray(X, float), nan=0.0)
    y = np.asarray(y, float)
    n, F = X.shape
    if n < 50:
        raise ValueError('Too few calibration pixels to fit the confidence (fewer than 50).')
    mc = min_count or max(20, n // (bins * 20))
    edges = [_quantile_edges(X[:, k], bins, mc) for k in range(F)]
    B = [np.searchsorted(edges[k], X[:, k], side='right') for k in range(F)]
    nb = [len(e) + 1 for e in edges]
    cnt = [np.bincount(B[k], minlength=nb[k]).astype(float) for k in range(F)]
    steps = [np.zeros(nb[k]) for k in range(F)]
    dirs = [FEATURES[k][1] for k in range(F)]
    c = float(y.mean())
    pred = np.zeros((F, n))
    it = 0
    for it in range(1, iters + 1):
        change = 0.0
        for k in range(F):
            r = y - c - (pred.sum(0) - pred[k])
            s = np.bincount(B[k], weights=r, minlength=nb[k])
            has = cnt[k] > 0
            mean = np.where(has, s / np.maximum(cnt[k], 1), 0.0)
            best = None
            for d in ((dirs[k],) if FEATURES[k][1] != 0 else (+1, -1)):
                g = np.zeros(nb[k])
                g[has] = pava(mean[has], cnt[k][has], increasing=d > 0)
                for i in np.flatnonzero(~has):            # empty bins: the nearest filled bin's step
                    fi = np.flatnonzero(has)
                    g[i] = g[fi[np.argmin(np.abs(fi - i))]] if len(fi) else 0.0
                sse = float(np.sum(cnt[k] * (mean - g) ** 2))
                if best is None or sse < best[0] - 1e-12:
                    best = (sse, g, d)
            g = best[1] - float(np.sum(cnt[k] * best[1]) / max(cnt[k].sum(), 1))
            if FEATURES[k][1] == 0:
                dirs[k] = best[2]
            change = max(change, float(np.max(np.abs(g - steps[k]))) if len(g) else 0.0)
            steps[k] = g
            pred[k] = g[B[k]]
        c = float(np.mean(y - pred.sum(0)))
        if change < 1e-3:
            break
    raw = c + pred.sum(0)
    ce = _quantile_edges(raw, 32, max(20, n // 128))
    cb = np.searchsorted(ce, raw, side='right')
    ncb = len(ce) + 1
    ccnt = np.bincount(cb, minlength=ncb).astype(float)
    cm = np.bincount(cb, weights=y, minlength=ncb) / np.maximum(ccnt, 1)
    has = ccnt > 0
    cal = np.zeros(ncb)
    cal[has] = pava(cm[has], ccnt[has], True)
    for i in np.flatnonzero(~has):
        fi = np.flatnonzero(has)
        cal[i] = cal[fi[np.argmin(np.abs(fi - i))]]
    eb = np.clip(np.searchsorted(ERR_EDGES, y, side='right') - 1, 0, len(ERR_EDGES) - 2)
    hist = np.zeros((ncb, len(ERR_EDGES) - 1), int)
    np.add.at(hist, (cb, eb), 1)
    return ErrorModel(list(FEATURE_NAMES), [int(d) for d in dirs], edges, steps, c, list(ce), list(cal),
                      hist.tolist(), n, it)


def reliability(pred, meas, budget: float, p_within=None, groups: int = 10) -> dict:
    """Predicted against measured error per decile of the prediction, the share within budget predicted (the mean of
    P) and measured, the rank correlation and the mean absolute error of the prediction against a constant."""
    pred = np.asarray(pred, float)
    meas = np.asarray(meas, float)
    if not len(pred):
        return {}
    q = np.unique(np.quantile(pred, np.linspace(0, 1, groups + 1)[1:-1]))
    b = np.searchsorted(q, pred, side='right')
    rows = []
    for k in range(len(q) + 1):
        m = b == k
        if m.any():
            rows.append({'pred_mean': round(float(pred[m].mean()), 3), 'meas_mean': round(float(meas[m].mean()), 3),
                         'meas_median': round(float(np.median(meas[m])), 3), 'pixels': int(m.sum()),
                         'within_meas': round(float(np.mean(meas[m] <= budget)), 4),
                         **({'within_pred': round(float(np.mean(p_within[m])), 4)} if p_within is not None else {})})
    from scipy.stats import spearmanr
    rho = float(spearmanr(pred, meas).statistic) if len(pred) > 2 and np.std(pred) > 0 else None
    out = {'deciles': rows, 'pixels': int(len(pred)), 'budget': budget,
           'mean_pred': round(float(pred.mean()), 3), 'mean_meas': round(float(meas.mean()), 3),
           'within_meas': round(float(np.mean(meas <= budget)), 4),
           'spearman': round(rho, 4) if rho is not None and math.isfinite(rho) else None,
           'mae_model': round(float(np.mean(np.abs(pred - meas))), 3),
           'mae_constant': round(float(np.mean(np.abs(meas - np.median(meas)))), 3),
           'calibration_error': round(float(np.sum([r['pixels'] * abs(r['pred_mean'] - r['meas_mean'])
                                                     for r in rows]) / max(len(pred), 1)), 3)}
    if p_within is not None:
        out['within_pred'] = round(float(np.mean(p_within)), 4)
    return out


# =========================================================================== calibration views
def _real_image(path, size):
    return SP._load_rgb(path, size)


def _pixel_error(rgb, real):
    return np.abs(np.asarray(rgb, float) - np.asarray(real, float)).mean(-1) * 255.0


def _sample(mask, k, rng):
    ix = np.flatnonzero(mask.ravel())
    if len(ix) > k:
        ix = np.sort(rng.choice(ix, k, replace=False))
    return ix


def _conf_size(model: SP.CamModel, scale: float):
    return max(16, int(round(model.width * scale))), max(16, int(round(model.height * scale)))


def drive_without(drive: SP.Drive, tag: str) -> SP.Drive:
    """The drive with one camera left out (its own cache key): the 'loo' calibration's reconstruction."""
    d = copy.copy(drive)
    d.cameras = {k: c for k, c in drive.cameras.items() if k != tag}
    d.key = f'{drive.key}-loo-{re.sub(r"[^A-Za-z0-9]+", "_", tag)}'
    return d


def align_paired(drive: SP.Drive, paired: SP.Drive, tol_m: float = 0.5) -> SP.Drive:
    """The paired recording on the drive's clock (its frame times moved by the difference of the two drives' first
    ego stamps); ValueError when its ego path is not the drive's (more than tol_m apart at its frame times)."""
    dt = (int(paired.t0_us) - int(drive.t0_us)) * 1e-6
    q = copy.copy(paired)
    q.cameras = {k: dataclasses.replace(c, frames=[(f[0] + dt, f[1], f[2]) for f in c.frames])
                 for k, c in paired.cameras.items()}
    ts = np.array([f[0] for c in q.cameras.values() for f in c.frames])
    if not len(ts):
        raise ValueError(f'{os.path.basename(os.path.normpath(paired.path))} has no camera frames.')
    ts = ts[(ts >= drive.traj.t[0]) & (ts <= drive.traj.t[-1])]
    if not len(ts):
        raise ValueError('The second recording does not overlap the drive in time.')
    ts = ts[:: max(1, len(ts) // 50)]
    a = drive.traj.at(ts)[1] + drive.origin
    b = paired.traj.at(ts - dt)[1] + paired.origin
    gap = float(np.max(np.linalg.norm(a - b, axis=1)))
    if gap > tol_m:
        raise ValueError(f'{os.path.basename(os.path.normpath(paired.path))} is not a recording of the same drive '
                         f'(its ego path is up to {gap:.1f} m from the drive\'s).')
    q.notes = list(paired.notes) + [f'Paired with the drive: clock offset {dt * 1e3:.3f} ms, ego paths within '
                                    f'{gap * 100:.1f} cm.']
    q.offset_s = dt
    return q


def calibration_views(drive: SP.Drive, sst: SP.SplatSettings, st: RedriveSettings, paired: SP.Drive | None = None,
                      loo_drive: SP.Drive | None = None) -> list:
    """[(source, DriveCamera, frame)] the calibration renders: held-out frames, the paired recording's frames, the
    left-out camera's frames (each source's share of calib_views in proportion to what it has)."""
    pools = {}
    if 'holdout' in st.sources:
        pools['holdout'] = [(c, fr) for c, _, fr in SP._frames_for(drive, sst, False, 1)]
    if 'paired' in st.sources and paired is not None:
        pools['paired'] = [(c, fr) for c in paired.cameras.values() for fr in c.frames]
    if 'loo' in st.sources and loo_drive is not None and st.loo_camera in drive.cameras:
        pools['loo'] = [(drive.cameras[st.loo_camera], fr) for fr in drive.cameras[st.loo_camera].frames]
    tot = sum(len(v) for v in pools.values())
    out = []
    for src, pool in pools.items():
        k = max(1, int(round(st.calib_views * len(pool) / max(tot, 1))))
        pick = np.unique(np.linspace(0, len(pool) - 1, min(k, len(pool))).round().astype(int)) if pool else []
        out += [(src, pool[i][0], pool[i][1]) for i in pick]
    return out


@dataclass
class Calibration:
    model: ErrorModel
    budget: float
    sources: dict                       # source -> {'views', 'pixels'}
    reliability: dict                   # out-of-fold, all sources
    per_source: dict                    # source -> out-of-fold reliability
    per_camera: dict                    # 'source:tag' -> {'views', 'psnr', 'ssim', 'mean_error'}
    feature_ranges: dict
    conf_scale: float
    key: str = ''
    seconds: float = 0.0

    def to_dict(self) -> dict:
        return {'schema': CALIB_SCHEMA, 'model': self.model.to_dict(), 'budget': self.budget, 'sources': self.sources,
                'reliability_out_of_fold': self.reliability, 'per_source': self.per_source,
                'per_camera': self.per_camera, 'feature_ranges': self.feature_ranges, 'conf_scale': self.conf_scale,
                'key': self.key, 'seconds': self.seconds,
                'units': {'error': 'mean absolute difference per pixel, grey levels 0–255, images area-averaged to '
                                   'conf_scale of their size'}}

    @staticmethod
    def from_dict(d) -> 'Calibration | None':
        if not isinstance(d, dict) or d.get('schema') != CALIB_SCHEMA:
            return None
        m = ErrorModel.from_dict(d.get('model') or {})
        if m is None:
            return None
        return Calibration(m, float(d.get('budget', 12.0)), dict(d.get('sources') or {}),
                           dict(d.get('reliability_out_of_fold') or {}), dict(d.get('per_source') or {}),
                           dict(d.get('per_camera') or {}), dict(d.get('feature_ranges') or {}),
                           float(d.get('conf_scale', 0.25)), str(d.get('key', '')), float(d.get('seconds', 0.0)))


def as_exported(ts, drive, model: SP.CamModel, R_cv, t_cv, t: float, size, quality: int, readout_s: float = 0.0):
    """The image an export writes for this camera (rendered at model's size, JPEG at quality), area-averaged to
    size (w, h), float RGB [0, 1]: what the confidence's error is measured on."""
    import cv2
    import torch
    with torch.no_grad():
        r = SP.render_view(ts, drive.traj, model, R_cv, t_cv, t, readout_s)
    rgb = (r['rgb'].clamp(0, 1) * 255 + 0.5).to(torch.uint8).cpu().numpy()
    ok, buf = cv2.imencode('.jpg', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, tuple(size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def _export_model(cam: SP.DriveCamera, f: float) -> SP.CamModel:
    """A recorded camera at the export's image size (f: the export's image scale over the recording's)."""
    f = min(max(f, 0.05), 2.0)
    return cam.model.scaled(max(16, int(round(cam.model.width * f))), max(16, int(round(cam.model.height * f))))


def _view_samples(ts, scene, ev, drive_scene, cam: SP.DriveCamera, fr, conf_scale, ego_fn, rng, k_px, f_exp=1.0,
                  quality=90):
    """Features, measured errors and the frame's PSNR / SSIM of one calibration view (the error of the image as
    an export writes it, area-averaged to the confidence's size)."""
    w, h = _conf_size(cam.model, conf_scale)
    m = cam.model.scaled(w, h)
    try:
        real = _real_image(fr[1], (w, h))
    except OSError:
        return None
    me = _export_model(cam, f_exp)
    fe = render_features(ts, scene, ev, drive_scene, m, cam.R, cam.t, fr[0], me.fx)
    valid = fe['valid'].copy()
    eg = ego_fn(cam, fr, (h, w)) if ego_fn is not None else None
    if eg is not None:
        valid &= ~np.asarray(eg, bool)
    if not valid.any():
        return None
    pred = as_exported(ts, drive_scene, me, cam.R, cam.t, fr[0], (w, h), quality)
    err = _pixel_error(pred, real)
    ix = _sample(valid, k_px, rng)
    X = np.stack([np.asarray(fe[f], float).ravel()[ix] for f in FEATURE_NAMES], 1)
    psnr = SP.SR.psnr(pred, real, valid)
    ssim = _ssim(pred, real, valid)
    return X, err.ravel()[ix], psnr, ssim, float(err[valid].mean())


def _ssim(a, b, mask) -> float | None:
    import torch
    s = SP.SR.ssim_map(torch.tensor(np.asarray(a, np.float32)), torch.tensor(np.asarray(b, np.float32))).numpy()
    return float(s[mask].mean()) if mask.any() else None


def calibrate(project, drive: SP.Drive, scene: SP.Scene, ev: Evidence, sst: SP.SplatSettings, st: RedriveSettings,
              engine=None, dev=None, paired: SP.Drive | None = None, loo=None, progress=None,
              cancel=None) -> Calibration:
    """Fit the error model on the calibration views (module docstring, 2). loo: (drive without the camera, its
    scene, its evidence) for the 'loo' source."""
    t0 = time.perf_counter()
    stop = cancel or (lambda: False)
    dev = dev or SP.SR.device_for(sst.device, sst.gpu_mem_gb)
    rng = np.random.default_rng(st.seed)
    views = calibration_views(drive, sst, st, paired, loo[0] if loo else None)
    if not views:
        raise ValueError('No calibration views: hold frames out (Hold out every > 0), give a second recording of the '
                         'drive, or leave a camera out.')
    ts = SP.TorchScene(scene, drive, dev)
    ts_loo = SP.TorchScene(loo[1], loo[0], dev) if loo else None
    ego_main = SP._ego_mask_fn(drive, project, engine)
    ego_pair = SP._ego_mask_fn(paired, project, engine) if paired is not None else None
    Xs, ys, srcs, times, vids = [], [], [], [], []
    f_exp = image_scale_of(drive, st) / max(drive.image_scale or 1.0, 1e-6)
    per_cam = {}
    for j, (src, cam, fr) in enumerate(views):
        if stop():
            raise Cancelled()
        _prog(progress, j / len(views), f'Calibration · {src} · {cam.tag} t = {fr[0]:.2f} s')
        if src == 'loo':
            r = _view_samples(ts_loo, loo[1], loo[2], loo[0], cam, fr, st.conf_scale, ego_main, rng, st.calib_pixels,
                              f_exp, st.jpeg_quality)
        else:
            r = _view_samples(ts, scene, ev, drive, cam, fr, st.conf_scale, ego_pair if src == 'paired' else ego_main,
                              rng, st.calib_pixels, f_exp, st.jpeg_quality)
        if r is None:
            continue
        X, y, p, s, me = r
        Xs.append(X)
        ys.append(y)
        srcs.append(np.full(len(y), SOURCES.index(src)))
        times.append(np.full(len(y), fr[0]))
        vids.append(np.full(len(y), j))
        pc = per_cam.setdefault(f'{src}:{cam.tag}', {'views': 0, 'psnr': [], 'ssim': [], 'mean_error': []})
        pc['views'] += 1
        if p is not None:
            pc['psnr'].append(p)
        if s is not None:
            pc['ssim'].append(s)
        pc['mean_error'].append(me)
    if not Xs:
        raise ValueError('No calibration view could be read (images missing).')
    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    S = np.concatenate(srcs)
    T_ = np.concatenate(times)
    V_ = np.concatenate(vids)
    _prog(progress, 0.95, 'Calibration · fitting')
    model = fit_error_model(X, y, st.bins)
    # out-of-fold: two folds of alternating time blocks (alternating views when the drive spans one block only)
    fold = (np.floor(T_ / BLOCK_S).astype(int) % 2)
    if min((fold == 0).sum(), (fold == 1).sum()) < 50:
        fold = V_ % 2
    oof = np.full(len(y), np.nan)
    oofp = np.full(len(y), np.nan)
    for f in (0, 1):
        tr, te = fold != f, fold == f
        if tr.sum() >= 50 and te.any():
            mf = fit_error_model(X[tr], y[tr], st.bins)
            oof[te] = mf.expected(X[te])
            oofp[te] = mf.p_within(X[te], st.error_budget)
    ok = np.isfinite(oof)
    rel = reliability(oof[ok], y[ok], st.error_budget, oofp[ok]) if ok.any() else {}
    per_src = {}
    for k, src in enumerate(SOURCES):
        m = ok & (S == k)
        if m.any():
            per_src[src] = reliability(oof[m], y[m], st.error_budget, oofp[m])
    sources = {src: {'views': int(sum(1 for v in views if v[0] == src)), 'pixels': int((S == k).sum())}
               for k, src in enumerate(SOURCES) if (S == k).any()}
    pcd = {k: {'views': v['views'], 'psnr_db': round(float(np.mean(v['psnr'])), 3) if v['psnr'] else None,
               'ssim': round(float(np.mean(v['ssim'])), 4) if v['ssim'] else None,
               'mean_error': round(float(np.mean(v['mean_error'])), 3)} for k, v in sorted(per_cam.items())}
    fr_ = {f: {'p5': round(float(np.percentile(X[:, k], 5)), 4), 'p50': round(float(np.percentile(X[:, k], 50)), 4),
               'p95': round(float(np.percentile(X[:, k], 95)), 4), 'max': round(float(X[:, k].max()), 4)}
           for k, f in enumerate(FEATURE_NAMES)}
    key = hashlib.sha256(json.dumps([ev.key, st.sources, st.paired_drive, st.loo_camera, st.calib_views,
                                     st.calib_pixels, st.bins, st.conf_scale, st.seed]).encode()).hexdigest()[:12]
    return Calibration(model, st.error_budget, sources, rel, per_src, pcd, fr_, st.conf_scale, key,
                       round(time.perf_counter() - t0, 2))


# =========================================================================== the proposed rig
def _splat_settings_for(sst: SP.SplatSettings, st: RedriveSettings) -> SP.SplatSettings:
    return SP.SplatSettings.from_dict(dict(sst.to_dict(), rolling_shutter=st.rolling_shutter))


def image_scale_of(drive: SP.Drive, st: RedriveSettings) -> float:
    return float(st.image_scale) if st.image_scale > 0 else float(drive.image_scale or 0.5)


def camera_spec(project, variant, cam, scale: float, sst: SP.SplatSettings, engine=None,
                size=None) -> SP.CamSpec:
    """A variant camera at scale of its module's sensor (or size): its lens model, pose, readout and body."""
    from .lens import resolve
    intr = resolve(project.module_for(cam), project.lens_for(cam))
    w, h = size or (max(16, int(round(intr.width * scale))), max(16, int(round(intr.height * scale))))
    m = SP.CamModel.from_intrinsics(intr, w, h)
    R = rot_zyx(cam.yaw, cam.pitch, cam.roll) @ BODY_TO_OPTICAL.T
    body, shade = SP._body(project, engine, variant, R, cam.pos(), m)
    return SP.CamSpec(cam.tag, cam.name, m, R, cam.pos(), SP._readout(project, cam, sst), body, shade, int(cam.slot))


def conf_view(ts, scene, ev, drive, spec: SP.CamSpec, t: float, f_ref: float, calib: Calibration | None,
              budget: float) -> dict:
    """A camera's re-rendered features with the calibrated expected error, P(within budget) and the reason per pixel
    (NaN / −1 outside the valid pixels)."""
    fe = render_features(ts, scene, ev, drive, spec.model, spec.R, spec.t, t, f_ref, body=spec.body)
    h, w = spec.model.height, spec.model.width
    E = np.full((h, w), np.nan, np.float32)
    P = np.full((h, w), np.nan, np.float32)
    Rn = np.full((h, w), -1, np.int8)
    v = fe['valid']
    if calib is not None and v.any():
        X = feature_matrix(fe)
        E[v] = calib.model.expected(X)
        P[v] = calib.model.p_within(X, budget)
        rs = calib.model.reasons(X)
        Rn[v] = np.where(P[v] >= 0.5, 0, rs)
    fe['expected'], fe['p_within'], fe['reason'] = E, P, Rn
    return fe


# =========================================================================== "do we lose our data?"
def _instants(drive: SP.Drive, st: RedriveSettings, n: int) -> list:
    first = max(drive.cameras.values(), key=lambda c: (len(c.frames), c.tag == 'F'))
    t1 = st.t_end if st.t_end > 0 else math.inf
    times = [fr[0] for fr in first.frames if st.t_start - 1e-9 <= fr[0] <= t1 + 1e-9]
    if not times:
        return []
    k = min(n, len(times))
    pick = np.unique(((np.arange(k) + 0.5) / k * len(times)).astype(int))
    return [float(times[i]) for i in pick]


def verdict(share: float, top_reason: str, st: RedriveSettings) -> tuple:
    pct = f'{100 * share:.0f} % of pixels within {st.error_budget:g} grey levels'
    if share >= st.reuse_share:
        return 'reuse', f'reuse: {pct}'
    why = REASON_TEXT.get(top_reason, top_reason)
    if share < st.collect_share:
        return 're-collect', f're-collect: {pct}; most of the rest is {why}'
    return 'partial', f'partly reusable: {pct}; the rest is mostly {why}'


def report(project, drive, scene, ev, calib: Calibration, variant, sst, st: RedriveSettings, engine=None, dev=None,
           progress=None, cancel=None) -> dict:
    """The data-reuse report of the proposed variant (module docstring, 3)."""
    from . import regions as RG
    stop = cancel or (lambda: False)
    dev = dev or SP.SR.device_for(sst.device, sst.gpu_mem_gb)
    ts = SP.TorchScene(scene, drive, dev)
    grid = RG.default_grid(project, variant)
    ssr = _splat_settings_for(sst, st)
    scale = image_scale_of(drive, st)
    inst = _instants(drive, st, st.report_instants)
    cams = [c for c in variant.cameras if c.visible]
    specs, fref = [], {}
    for c in cams:
        full = camera_spec(project, variant, c, scale, ssr, None)
        size = _conf_size(full.model, st.conf_scale)
        specs.append(camera_spec(project, variant, c, scale, ssr, engine, size))
        fref[c.tag] = full.model.fx
    nR = grid.n_ranges
    reg_px = np.zeros((grid.sectors, nR))
    reg_p = np.zeros((grid.sectors, nR))
    reg_reason = np.zeros((grid.sectors, nR, len(REASONS)))
    per_cam = {}
    overlays = {}
    new_seen = set()
    old = []
    rec_rig = _recorded_conf_rig(drive, st, project, engine)
    total = max(len(inst) * (len(specs) + len(rec_rig)), 1)
    step = 0
    for k, t in enumerate(inst):
        for sp_ in specs:
            if stop():
                raise Cancelled()
            _prog(progress, step / total, f'Report · t = {t:.2f} s · {sp_.tag}')
            step += 1
            fe = conf_view(ts, scene, ev, drive, sp_, t, fref[sp_.tag], calib, st.error_budget)
            v = fe['valid']
            pc = per_cam.setdefault(sp_.tag, {'pixels': 0, 'p_sum': 0.0, 'e_sum': 0.0,
                                              'reason': np.zeros(len(REASONS)), 'angle': [], 'extrap': []})
            P = fe['p_within'][v].astype(float)
            pc['pixels'] += int(v.sum())
            pc['p_sum'] += float(P.sum())
            pc['e_sum'] += float(np.nansum(fe['expected'][v]))
            # each pixel: P to 'ok', 1 − P to its reason (its dominant one even where P ≥ ½)
            rs = calib.model.reasons(feature_matrix(fe)) if v.any() else np.zeros(0, int)
            np.add.at(pc['reason'], np.zeros(len(P), int), P)
            np.add.at(pc['reason'], rs, 1.0 - P)
            pc['angle'].append(fe['angle_deg'][v])
            pc['extrap'].append(fe['extrap_m'][v])
            Xv = fe['X_v'][v]
            surf = np.isfinite(Xv[:, 0])
            if surf.any():
                s_, r_ = grid.assign(Xv[surf, 0], Xv[surf, 1])
                okr = r_ >= 0
                np.add.at(reg_px, (s_[okr], r_[okr]), 1.0)
                np.add.at(reg_p, (s_[okr], r_[okr]), P[surf][okr])
                np.add.at(reg_reason, (s_[okr], r_[okr], np.zeros(int(okr.sum()), int)), P[surf][okr])
                np.add.at(reg_reason, (s_[okr], r_[okr], rs[surf][okr]), 1.0 - P[surf][okr])
            ids = fe['index'][v & (fe['alpha'] >= 0.5)]
            new_seen.update(np.unique(ids[ids >= 0]).tolist())
            if k == len(inst) // 2:
                overlays[sp_.tag] = {'t': t, 'rgb': (fe['rgb'] * 255 + 0.5).astype(np.uint8),
                                     'expected': fe['expected'], 'reason': fe['reason'], 'p_within': fe['p_within'],
                                     'valid': v}
        # the recorded rig at the same instant: which Gaussians it sees, where
        for rc in rec_rig:
            if stop():
                raise Cancelled()
            _prog(progress, step / total, f'Report · recorded rig · {rc.tag}')
            step += 1
            old.append(_seen_by(ts, scene, drive, rc, t))
    reasons_rig = np.zeros(len(REASONS))
    cams_out = []
    for tag, pc in per_cam.items():
        npx = max(pc['pixels'], 1)
        share = pc['p_sum'] / npx
        rsh = pc['reason'] / npx
        top = REASONS[1 + int(np.argmax(rsh[1:]))] if rsh[1:].sum() > 0 else 'floor'
        vd, text = verdict(share, top, st)
        ang = np.concatenate(pc['angle']) if pc['angle'] else np.zeros(0)
        exd = np.concatenate(pc['extrap']) if pc['extrap'] else np.zeros(0)
        cams_out.append({'camera': tag, 'pixels': pc['pixels'], 'within_share': round(share, 4),
                         'mean_expected_error': round(pc['e_sum'] / npx, 3),
                         'reasons': {r: round(float(rsh[i]), 4) for i, r in enumerate(REASONS)},
                         'top_reason': top, 'verdict': vd, 'text': f'{tag}: {text}',
                         'median_angle_deg': round(float(np.median(ang)), 2) if len(ang) else None,
                         'median_extrap_m': round(float(np.median(exd)), 3) if len(exd) else None})
        reasons_rig += pc['reason']
    npx_all = max(sum(pc['pixels'] for pc in per_cam.values()), 1)
    share_all = float(sum(pc['p_sum'] for pc in per_cam.values()) / npx_all)
    rsh_all = reasons_rig / npx_all
    top_all = REASONS[1 + int(np.argmax(rsh_all[1:]))] if rsh_all[1:].sum() > 0 else 'floor'
    vd_all, text_all = verdict(share_all, top_all, st)
    regions = []
    for s, r, rid, name in grid.regions():
        if reg_px[s, r] > 0:
            regions.append({'region': rid, 'name': name, 'pixels': int(reg_px[s, r]),
                            'within_share': round(float(reg_p[s, r] / reg_px[s, r]), 4),
                            'reasons': {x: round(float(reg_reason[s, r, i] / reg_px[s, r]), 4)
                                        for i, x in enumerate(REASONS)}})
    inverse = _inverse(old, new_seen, grid)
    return {'instants': [round(x, 4) for x in inst], 'variant': variant.name, 'variant_uid': variant.uid,
            'budget': st.error_budget, 'grid': grid.to_dict(), 'cameras': cams_out,
            'rig': {'pixels': int(npx_all), 'within_share': round(share_all, 4),
                    'reasons': {r: round(float(rsh_all[i]), 4) for i, r in enumerate(REASONS)},
                    'top_reason': top_all, 'verdict': vd_all, 'text': f'Rig: {text_all}'},
            'regions': regions, 'inverse': inverse, '_overlays': overlays,
            '_polar': {'px': reg_px, 'p': reg_p}}


def _recorded_conf_rig(drive: SP.Drive, st: RedriveSettings, project=None, engine=None) -> list:
    """The recorded rig's cameras at conf_scale of their recorded image size (with the drive variant's body)."""
    var = SP._variant_named(project, drive.variant) if project is not None else None
    out = []
    for cam in drive.cameras.values():
        w, h = _conf_size(cam.model, st.conf_scale)
        m = cam.model.scaled(w, h)
        body = SP._body(project, engine, var, cam.R, cam.t, m)[0] if var is not None else None
        out.append(SP.CamSpec(cam.tag, cam.channel, m, cam.R, cam.t, 0.0, body, None, 0))
    return out


def _seen_by(ts, scene, drive, spec: SP.CamSpec, t: float) -> dict:
    """{'ids': Gaussian ids, 'px': pixels on each, 'xy': their centres in the vehicle frame at t} a camera sees."""
    import torch
    with torch.no_grad():
        r = SP.render_view(ts, drive.traj, spec.model, spec.R, spec.t, t, 0.0, want_index=True)
        idx = r['index'].cpu().numpy()
        al = r['alpha'].cpu().numpy()
    m = (al >= 0.5) & r['field'] & (idx >= 0)
    if spec.body is not None:
        m &= ~spec.body
    ids, cnt = np.unique(idx[m], return_counts=True)
    P, _ = SP._posed_means(scene, drive, t)
    Rw, pw = drive.traj.pose(t)
    xy = ((P[ids] - pw) @ Rw)[:, :2] if len(ids) else np.zeros((0, 2))
    return {'ids': ids, 'px': cnt, 'xy': xy}


def _inverse(old: list, new_seen: set, grid) -> dict:
    """What the recorded rig saw that the proposed one never sees at the report instants (pixels of the recorded
    rig on Gaussians no proposed camera shows), per region."""
    tot = np.zeros((grid.sectors, grid.n_ranges))
    lost = np.zeros_like(tot)
    all_px = lost_px = 0.0
    new = np.fromiter(new_seen, np.int64) if new_seen else np.zeros(0, np.int64)
    for o in old:
        if not len(o['ids']):
            continue
        gone = ~np.isin(o['ids'], new)
        s, r = grid.assign(o['xy'][:, 0], o['xy'][:, 1])
        ok = r >= 0
        np.add.at(tot, (s[ok], r[ok]), o['px'][ok])
        np.add.at(lost, (s[ok], r[ok]), (o['px'] * gone)[ok])
        all_px += float(o['px'].sum())
        lost_px += float((o['px'] * gone).sum())
    rows = []
    for s, r, rid, name in grid.regions():
        if tot[s, r] > 0:
            rows.append({'region': rid, 'name': name, 'old_pixels': int(tot[s, r]),
                         'lost_share': round(float(lost[s, r] / tot[s, r]), 4)})
    worst = sorted([x for x in rows if x['old_pixels'] >= 0.002 * max(all_px, 1)], key=lambda x: -x['lost_share'])
    return {'old_pixels': int(all_px), 'lost_share': round(lost_px / all_px, 4) if all_px else 0.0, 'regions': rows,
            'worst': worst[:5]}


# =========================================================================== a second recording of the drive
def paired_validation(project, drive, scene, ev, calib: Calibration, paired: SP.Drive, sst, st: RedriveSettings,
                      engine=None, dev=None, progress=None, cancel=None) -> dict:
    """The paired recording's own cameras re-rendered at its frame times and scored against its real images: PSNR,
    SSIM and the measured error against the confidence (per camera and over all; reliability)."""
    stop = cancel or (lambda: False)
    dev = dev or SP.SR.device_for(sst.device, sst.gpu_mem_gb)
    ts = SP.TorchScene(scene, drive, dev)
    ego_fn = SP._ego_mask_fn(paired, project, engine)
    frames = [(c, fr) for c in paired.cameras.values() for fr in c.frames]
    if len(frames) > st.val_frames:
        frames = [frames[int(i)] for i in np.linspace(0, len(frames) - 1, st.val_frames).round().astype(int)]
    rng = np.random.default_rng(st.seed + 1)
    per = {}
    E_all, M_all, P_all = [], [], []
    for j, (cam, fr) in enumerate(frames):
        if stop():
            raise Cancelled()
        _prog(progress, j / max(len(frames), 1), f'Paired recording · {cam.tag} t = {fr[0]:.2f} s')
        w, h = _conf_size(cam.model, st.conf_scale)
        m = cam.model.scaled(w, h)
        try:
            real = _real_image(fr[1], (w, h))
        except OSError:
            continue
        spec = SP.CamSpec(cam.tag, cam.channel, m, cam.R, cam.t, 0.0, None, None, 0)
        me = _export_model(cam, image_scale_of(drive, st) / max(paired.image_scale or 1.0, 1e-6))
        fe = conf_view(ts, scene, ev, drive, spec, fr[0], me.fx, calib, st.error_budget)
        v = fe['valid'].copy()
        eg = ego_fn(cam, fr, (h, w))
        if eg is not None:
            v &= ~np.asarray(eg, bool)
        if not v.any():
            continue
        fe['rgb'] = as_exported(ts, drive, me, cam.R, cam.t, fr[0], (w, h), st.jpeg_quality)
        err = _pixel_error(fe['rgb'], real)
        d = per.setdefault(cam.tag, {'frames': 0, 'psnr': [], 'ssim': [], 'meas': [], 'pred': [], 'pw': [],
                                     'within_meas': [], 'within_pred': []})
        d['frames'] += 1
        p = SP.SR.psnr(fe['rgb'], real, v)
        if p is not None:
            d['psnr'].append(p)
        s = _ssim(fe['rgb'], real, v)
        if s is not None:
            d['ssim'].append(s)
        d['meas'].append(float(err[v].mean()))
        d['pred'].append(float(np.nanmean(fe['expected'][v])))
        d['within_meas'].append(float(np.mean(err[v] <= st.error_budget)))
        d['within_pred'].append(float(np.nanmean(fe['p_within'][v])))
        ix = _sample(v, st.calib_pixels, rng)
        E_all.append(fe['expected'].ravel()[ix])
        M_all.append(err.ravel()[ix])
        P_all.append(fe['p_within'].ravel()[ix])
    cams = {}
    for tag, d in sorted(per.items()):
        cams[tag] = {'frames': d['frames'], 'psnr_db': round(float(np.mean(d['psnr'])), 3) if d['psnr'] else None,
                     'ssim': round(float(np.mean(d['ssim'])), 4) if d['ssim'] else None,
                     'mean_error_meas': round(float(np.mean(d['meas'])), 3),
                     'mean_error_pred': round(float(np.mean(d['pred'])), 3),
                     'within_meas': round(float(np.mean(d['within_meas'])), 4),
                     'within_pred': round(float(np.mean(d['within_pred'])), 4)}
    rel = reliability(np.concatenate(E_all), np.concatenate(M_all), st.error_budget, np.concatenate(P_all)) \
        if E_all else {}
    frame_corr = None
    fm = [(a, b) for d in per.values() for a, b in zip(d['pred'], d['meas'])]
    if len(fm) > 2:
        from scipy.stats import spearmanr
        frame_corr = round(float(spearmanr([a for a, _ in fm], [b for _, b in fm]).statistic), 4)
    return {'drive': os.path.basename(os.path.normpath(paired.path)), 'variant': paired.variant, 'cameras': cams,
            'reliability': rel, 'frames': sum(d['frames'] for d in per.values()),
            'frame_rank_correlation': frame_corr,
            'in_calibration': 'paired' in st.sources,
            'note': ('The paired recording is one of the calibration sources: these figures are in-sample.'
                     if 'paired' in st.sources else
                     'The paired recording is not a calibration source: these figures test the confidence on views '
                     'it was not fitted on.')}


# =========================================================================== the study
@dataclass
class RedriveResult:
    settings: RedriveSettings
    variant: str
    drive: dict
    evidence: dict
    calibration: dict
    report: dict
    paired: dict | None
    findings: list
    notes: list
    timing: dict
    seconds: float = 0.0


def _finding(out, sev, kind, msg):
    out.append({'severity': sev, 'kind': kind, 'message': msg})


def _open_and_build(project, sst, engine, cache_root, progress, cancel, drive=None, scene=None):
    if drive is None:
        drive = SP.open_drive(sst.drive, cache_root, lambda f, t='': _prog(progress, 0.05 * f, t), cancel)
    if scene is None:
        try:
            scene = SP.build_scene(drive, sst, project, engine, lambda f, t='': _prog(progress, 0.05 + 0.35 * f, t),
                                   cancel, cache_root)
        except SP.Cancelled:
            raise Cancelled()
    return drive, scene


def prepare(project, settings: RedriveSettings | None = None, engine=None, cache_root=None, progress=None,
            cancel=None, drive=None, scene=None, evidence=None, calibration: Calibration | None = None,
            paired: SP.Drive | None = None) -> dict:
    """Open the drive, build (or load) the scene, its evidence and the calibration: {'drive', 'scene', 'evidence',
    'calibration', 'paired', 'loo', 'dev', 'timing'}."""
    st = settings or settings_of(project)
    sst = SP.settings_of(project)
    stop = cancel or (lambda: False)
    if SP.torch_missing():
        raise ValueError(SP.torch_missing())
    if engine is None and project is not None:
        from .analysis import Engine
        engine = Engine()
    if cache_root is None and project is not None:
        from .. import workdir
        cache_root = workdir.project_dir(project, None, 'splat')
    timing = {}
    t0 = time.perf_counter()
    drive, scene = _open_and_build(project, sst, engine, cache_root, progress, stop, drive, scene)
    timing['scene_s'] = round(time.perf_counter() - t0, 2)
    dev = SP.SR.device_for(sst.device, sst.gpu_mem_gb)
    t0 = time.perf_counter()
    ev = evidence or evidence_for(scene, drive, sst, project, engine, dev, cache_root, st.max_views,
                                  lambda f, t='': _prog(progress, 0.4 + 0.1 * f, t), stop)
    timing['evidence_s'] = round(time.perf_counter() - t0, 2)
    if paired is None and st.paired_drive:
        paired = align_paired(drive, SP.open_drive(st.paired_drive, cache_root))
    loo = None
    if calibration is None and 'loo' in st.sources and st.loo_camera in drive.cameras:
        t0 = time.perf_counter()
        d2 = drive_without(drive, st.loo_camera)
        try:
            s2 = SP.build_scene(d2, sst, project, engine, lambda f, t='': _prog(progress, 0.5 + 0.15 * f,
                                                                                 'Left-out camera · ' + t),
                                stop, cache_root)
        except SP.Cancelled:
            raise Cancelled()
        e2 = evidence_for(s2, d2, sst, project, engine, dev, cache_root, st.max_views, None, stop)
        loo = (d2, s2, e2)
        timing['loo_s'] = round(time.perf_counter() - t0, 2)
    if calibration is None:
        t0 = time.perf_counter()
        calibration = calibrate(project, drive, scene, ev, sst, st, engine, dev, paired, loo,
                                lambda f, t='': _prog(progress, 0.65 + 0.15 * f, t), stop)
        timing['calibration_s'] = round(time.perf_counter() - t0, 2)
    return {'drive': drive, 'scene': scene, 'evidence': ev, 'calibration': calibration, 'paired': paired, 'loo': loo,
            'dev': dev, 'timing': timing, 'engine': engine, 'cache_root': cache_root}


def run(project, engine=None, settings: RedriveSettings | None = None, variant=None, progress=None, cancel=None,
        cache_root=None, drive=None, scene=None, evidence=None, calibration=None) -> RedriveResult | None:
    """The data-reuse study (None when cancelled): the scene, its evidence, the calibration, the report of the
    proposed variant and, with a paired recording, the confidence tested on it."""
    t_start = time.perf_counter()
    st = settings or settings_of(project)
    stop = cancel or (lambda: False)
    try:
        pr = prepare(project, st, engine, cache_root, progress, stop, drive, scene, evidence, calibration)
        sst = SP.settings_of(project)
        v = variant or proposed_variant(project, st, sst)
        if v is None:
            raise ValueError('No variant to re-shoot.')
        t0 = time.perf_counter()
        rep = report(project, pr['drive'], pr['scene'], pr['evidence'], pr['calibration'], v, sst, st, pr['engine'],
                     pr['dev'], lambda f, t='': _prog(progress, 0.8 + 0.15 * f, t), stop)
        pr['timing']['report_s'] = round(time.perf_counter() - t0, 2)
        pv = None
        if pr['paired'] is not None:
            t0 = time.perf_counter()
            pv = paired_validation(project, pr['drive'], pr['scene'], pr['evidence'], pr['calibration'], pr['paired'],
                                   sst, st, pr['engine'], pr['dev'], lambda f, t='': _prog(progress, 0.95 + 0.05 * f,
                                                                                           t), stop)
            pr['timing']['paired_s'] = round(time.perf_counter() - t0, 2)
    except (Cancelled, SP.Cancelled):
        return None
    cal = pr['calibration']
    res = RedriveResult(st, v.name, pr['drive'].info(), dict(pr['evidence'].info, key=pr['evidence'].key),
                        cal.to_dict(), rep, pv, [], [], pr['timing'], round(time.perf_counter() - t_start, 2))
    res.findings = _findings(res, st)
    res.notes = list(pr['drive'].notes)
    res._prepared = pr                                     # noqa: SLF001 — the scene, evidence for an export
    _prog(progress, 1.0, 'Done')
    return res


def _findings(res: RedriveResult, st: RedriveSettings) -> list:
    out = []
    rep = res.report
    for c in rep.get('cameras') or []:
        sev = {'reuse': 'good', 'partial': 'note', 're-collect': 'warning'}[c['verdict']]
        _finding(out, sev, 'camera', c['text'] + '.')
    rg = rep.get('rig') or {}
    if rg:
        _finding(out, {'reuse': 'good', 'partial': 'note', 're-collect': 'warning'}[rg['verdict']], 'rig',
                 rg['text'] + '.')
    inv = rep.get('inverse') or {}
    if inv.get('old_pixels'):
        worst = ', '.join(f'{w["name"]} ({100 * w["lost_share"]:.0f} %)' for w in inv.get('worst', [])[:3]
                          if w['lost_share'] >= 0.2)
        _finding(out, 'warning' if inv['lost_share'] >= 0.2 else 'note', 'inverse',
                 f'Of what the recorded rig saw at these instants, {100 * inv["lost_share"]:.0f} % is no longer seen '
                 f'by any camera of {res.variant}' + (f'; most lost at {worst}' if worst else '') + '.')
    rel = (res.calibration or {}).get('reliability_out_of_fold') or {}
    if rel:
        _finding(out, 'note', 'calibration',
                 f'Confidence out of fold: predicted {rel["mean_pred"]:.1f} against measured {rel["mean_meas"]:.1f} '
                 f'grey levels (rank correlation {rel.get("spearman") or 0:.2f}; within budget '
                 f'{100 * rel.get("within_pred", 0):.0f} % predicted, {100 * rel["within_meas"]:.0f} % measured).')
    pv = res.paired or {}
    prl = pv.get('reliability') or {}
    if prl:
        _finding(out, 'note', 'paired',
                 f'On the second recording ({pv.get("drive")}, {pv.get("frames")} frames): predicted '
                 f'{prl["mean_pred"]:.1f} against measured {prl["mean_meas"]:.1f} grey levels, within budget '
                 f'{100 * prl.get("within_pred", 0):.0f} % predicted against {100 * prl["within_meas"]:.0f} % measured'
                 + (' (in-sample: it is a calibration source).' if pv.get('in_calibration') else '.'))
    _finding(out, 'note', 'caveat', ASSUMPTIONS)
    return out


def summary(res: RedriveResult) -> dict:
    rep = res.report
    return {'variant': res.variant, 'drive': os.path.basename(str(res.drive.get('path', ''))),
            'budget': res.settings.error_budget, 'rig_within_share': (rep.get('rig') or {}).get('within_share'),
            'rig_verdict': (rep.get('rig') or {}).get('verdict'),
            'cameras': {c['camera']: [c['verdict'], c['within_share']] for c in rep.get('cameras') or []},
            'lost_share': (rep.get('inverse') or {}).get('lost_share'),
            'calibration_views': {k: v['views'] for k, v in (res.calibration.get('sources') or {}).items()},
            'findings': [f['message'] for f in res.findings if f['severity'] in ('warning', 'serious', 'critical')]}


def to_doc(res: RedriveResult, project=None) -> dict:
    rep = {k: v for k, v in res.report.items() if not k.startswith('_')}
    return {'schema': SCHEMA, 'project': getattr(project, 'name', ''), 'variant': res.variant, 'frame': FRAME,
            'units': {'error': 'grey levels 0–255, mean absolute difference per pixel at conf_scale',
                      'share': '0–1', 'length': 'm', 'angle': 'deg', 'time': 's from the drive\'s first ego pose'},
            'settings': res.settings.to_dict(), 'drive': res.drive, 'evidence': res.evidence,
            'calibration': res.calibration, 'report': rep, 'paired': res.paired, 'findings': res.findings,
            'notes': res.notes, 'timing': res.timing, 'seconds': res.seconds, 'assumptions': ASSUMPTIONS,
            'features': [{'name': f[0], 'direction': f[1], 'unit': f[2], 'reason': f[3], 'text': f[4]}
                         for f in FEATURES],
            'reasons': REASON_TEXT}


def cameras_csv(res: RedriveResult) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator='\n')
    w.writerow(['camera', 'pixels', 'within_share', 'mean_expected_error', 'verdict', 'top_reason',
                *[f'reason_{r}' for r in REASONS], 'median_angle_deg', 'median_extrap_m'])
    for c in res.report.get('cameras') or []:
        w.writerow([c['camera'], c['pixels'], c['within_share'], c['mean_expected_error'], c['verdict'],
                    c['top_reason'], *[c['reasons'][r] for r in REASONS], c['median_angle_deg'], c['median_extrap_m']])
    return buf.getvalue()


def regions_csv(res: RedriveResult) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator='\n')
    w.writerow(['region', 'name', 'pixels', 'within_share', *[f'reason_{r}' for r in REASONS], 'old_pixels',
                'lost_share'])
    inv = {r['region']: r for r in (res.report.get('inverse') or {}).get('regions') or []}
    for r in res.report.get('regions') or []:
        i = inv.get(r['region'], {})
        w.writerow([r['region'], r['name'], r['pixels'], r['within_share'], *[r['reasons'][x] for x in REASONS],
                    i.get('old_pixels', ''), i.get('lost_share', '')])
    return buf.getvalue()


# =========================================================================== pictures
def error_colour(E, budget: float) -> np.ndarray:
    """(h, w, 3) uint8: expected error from green (0) through yellow (the budget) to red (3 × budget); grey where
    there is none."""
    E = np.asarray(E, float)
    x = np.clip(np.nan_to_num(E, nan=-1.0) / max(budget, 1e-6), 0, 3) / 3.0
    stops = np.array([[30, 150, 80], [235, 200, 40], [220, 50, 40]], float)
    lo = np.clip((x * 2).astype(int), 0, 1)
    f = x * 2 - lo
    rgb = stops[lo] * (1 - f[..., None]) + stops[lo + 1] * f[..., None]
    rgb[~np.isfinite(E)] = (60, 60, 60)
    return rgb.astype(np.uint8)


def overlay_rgb(rgb, reason, valid=None, alpha: float = 0.55) -> np.ndarray:
    """The re-rendered image with the pixels outside budget tinted by their reason."""
    out = np.asarray(rgb, float).copy()
    reason = np.asarray(reason)
    for i, r in enumerate(REASONS):
        if r == 'ok':
            continue
        m = reason == i
        if m.any():
            out[m] = (1 - alpha) * out[m] + alpha * np.array(REASON_RGB[r], float)
    if valid is not None:
        out[~np.asarray(valid, bool)] *= 0.35
    return np.clip(out, 0, 255).astype(np.uint8)


def polar_png(res_or_report, px: int = 520, inverse: bool = False) -> bytes | None:
    """The regions as a polar map: the share within budget (or, inverse, the share lost) per sector and range bin,
    straight ahead up."""
    import cv2
    from .pngtext import put_text
    from . import regions as RG
    rep = res_or_report.report if hasattr(res_or_report, 'report') else res_or_report
    grid = RG.RegionGrid.from_dict(rep.get('grid') or {})
    rows = (rep.get('inverse') or {}).get('regions') if inverse else rep.get('regions')
    val = {r['region']: (r['lost_share'] if inverse else r['within_share']) for r in rows or []}
    img = np.full((px, px, 3), 24, np.uint8)
    c = px / 2
    nR = grid.n_ranges
    rad = [(k + 1) / nR * (c - 30) for k in range(nR)]
    w = 360.0 / grid.sectors
    for s in range(grid.sectors):
        for r in range(nR):
            v = val.get(grid.region_id(s, r))
            r0 = rad[r - 1] if r else 0.0
            a0, a1 = s * w - w / 2, s * w + w / 2
            th = np.radians(np.linspace(a0, a1, 16))
            pts = [(c - r0 * np.sin(t), c - r0 * np.cos(t)) for t in th] + \
                  [(c - rad[r] * np.sin(t), c - rad[r] * np.cos(t)) for t in th[::-1]]
            col = (55, 55, 55) if v is None else tuple(int(x) for x in error_colour(
                np.array([[(v if inverse else 1.0 - v) * 3 * 1.0]]), 1.0)[0, 0][::-1])
            cv2.fillPoly(img, [np.round(np.array(pts)).astype(np.int32)], col, cv2.LINE_AA)
            cv2.polylines(img, [np.round(np.array(pts)).astype(np.int32)], True, (20, 20, 20), 1, cv2.LINE_AA)
            if v is not None and r < nR:
                rm = (r0 + rad[r]) / 2
                tm = math.radians(s * w)
                put_text(img, f'{100 * v:.0f}', (int(c - rm * math.sin(tm)) - 9, int(c - rm * math.cos(tm)) + 4),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.33, (245, 245, 245), 1, cv2.LINE_AA)
    cv2.rectangle(img, (int(c) - 5, int(c) - 9), (int(c) + 5, int(c) + 9), (235, 235, 235), -1)
    names = grid.range_names()
    put_text(img, ('lost share (%)' if inverse else f'share within {rep.get("budget", 0):g} grey levels (%)')
             + f' · rings {", ".join(names)}', (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (235, 235, 235), 1,
             cv2.LINE_AA)
    ok, buf = cv2.imencode('.png', img)
    return bytes(buf) if ok else None


def overlays_png(res: RedriveResult, width: int = 320) -> bytes | None:
    """Every proposed camera at the middle instant: the re-render with the reasons tinted over it, and the expected
    error, labelled."""
    import cv2
    from .pngtext import put_text
    ov = res.report.get('_overlays') or {}
    if not ov:
        return None
    rows = []
    for tag, o in ov.items():
        a = overlay_rgb(o['rgb'], o['reason'], o['valid'])
        b = error_colour(o['expected'], res.settings.error_budget)
        tiles = []
        for im, cap in ((a, f'{tag} · reasons'), (b, f'{tag} · expected error')):
            h, w = im.shape[:2]
            im = cv2.resize(im, (width, max(1, int(round(h * width / w)))), interpolation=cv2.INTER_NEAREST)
            im = cv2.copyMakeBorder(cv2.cvtColor(im, cv2.COLOR_RGB2BGR), 20, 2, 2, 2, cv2.BORDER_CONSTANT,
                                    value=(24, 24, 24))
            put_text(im, cap, (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (235, 235, 235), 1, cv2.LINE_AA)
            tiles.append(im)
        hm = max(t.shape[0] for t in tiles)
        rows.append(np.concatenate([cv2.copyMakeBorder(t, 0, hm - t.shape[0], 0, 0, cv2.BORDER_CONSTANT,
                                                       value=(24, 24, 24)) for t in tiles], 1))
    wm = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 0, 0, wm - r.shape[1], cv2.BORDER_CONSTANT, value=(24, 24, 24)) for r in rows]
    items, x, y = [], 6, 0
    for r in REASONS[1:]:
        tw = cv2.getTextSize(REASON_TEXT[r], cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)[0][0] + 26
        if x + tw > wm and x > 6:
            x, y = 6, y + 1
        items.append((r, x, y))
        x += tw
    legend = np.full((22 * (y + 1), wm, 3), 16, np.uint8)
    for r, x, y in items:
        cv2.rectangle(legend, (x, 6 + 22 * y), (x + 10, 16 + 22 * y), REASON_RGB[r][::-1], -1)
        put_text(legend, REASON_TEXT[r], (x + 14, 15 + 22 * y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (235, 235, 235), 1,
                 cv2.LINE_AA)
    ok, buf = cv2.imencode('.png', np.concatenate([legend] + rows, 0))
    return bytes(buf) if ok else None


def files(res: RedriveResult, project=None) -> list:
    out = [('redrive.json', (json.dumps(to_doc(res, project), indent=1, default=float) + '\n').encode('utf-8')),
           ('redrive_cameras.csv', cameras_csv(res).encode('utf-8')),
           ('redrive_regions.csv', regions_csv(res).encode('utf-8'))]
    for name, png in (('redrive_polar.png', polar_png(res)), ('redrive_lost.png', polar_png(res, inverse=True)),
                      ('redrive_overlays.png', overlays_png(res))):
        if png:
            out.append((name, png))
    return out


def export(res: RedriveResult, project, out_dir) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for name, data in files(res, project):
        p = os.path.join(out_dir, name)
        with open(p, 'wb') as fh:
            fh.write(data)
        paths.append(p)
    return paths


# =========================================================================== nudge
def nudged_camera(cam, dx=0.0, dy=0.0, dz=0.0, dyaw=0.0, dpitch=0.0, lens_id=None):
    """A copy of a variant camera moved by the offsets (vehicle frame, m; degrees, pitch positive down) and with
    another lens."""
    from .project import Camera
    c = Camera.from_dict(cam.to_dict())
    c.x, c.y, c.z = c.x + dx, c.y + dy, c.z + dz
    c.yaw = (c.yaw + dyaw + 180.0) % 360.0 - 180.0
    c.pitch = float(np.clip(c.pitch + dpitch, -90.0, 90.0))
    if lens_id:
        c.lens_id = lens_id
        c.calib = {} if lens_id != cam.lens_id else c.calib
    return c


def nudge_view(project, variant, cam, prepared: dict, t: float, st: RedriveSettings, scale: float = 1.0,
               engine=None) -> dict:
    """One (possibly nudged) camera re-rendered with its confidence at scale × conf_scale of its image: {'rgb' uint8,
    'expected', 'p_within', 'reason', 'valid', 'within_share', 'seconds'}."""
    t0 = time.perf_counter()
    drive, scene, ev, calib = prepared['drive'], prepared['scene'], prepared['evidence'], prepared['calibration']
    sst = SP.settings_of(project)
    ssr = _splat_settings_for(sst, st)
    full = camera_spec(project, variant, cam, image_scale_of(drive, st), ssr, None)
    size = _conf_size(full.model, st.conf_scale * scale)
    spec = camera_spec(project, variant, cam, image_scale_of(drive, st), ssr, engine, size)
    ts = prepared.get('_ts')
    if ts is None:
        ts = prepared['_ts'] = SP.TorchScene(scene, drive, prepared['dev'])
    fe = conf_view(ts, scene, ev, drive, spec, t, full.model.fx, calib, st.error_budget)
    v = fe['valid']
    rgb = (fe['rgb'] * 255 + 0.5).astype(np.uint8)
    if spec.body is not None:
        rgb[spec.body] = 90
    body = float(spec.body[fe['field']].mean()) if spec.body is not None and fe['field'].any() else 0.0
    note = ('The camera is inside the vehicle body or a housing: its whole field is the own vehicle.' if body > 0.98
            else (f'{100 * body:.0f} % of the field is the own vehicle.' if body > 0.25 else ''))
    return {'rgb': rgb, 'expected': fe['expected'], 'p_within': fe['p_within'], 'reason': fe['reason'], 'valid': v,
            'within_share': float(np.nanmean(fe['p_within'][v])) if v.any() and calib is not None else None,
            'body_share': round(body, 4), 'note': note, 'size': size, 'seconds': round(time.perf_counter() - t0, 3)}


# =========================================================================== the re-shot dataset
def _token(*parts) -> str:
    return hashlib.md5('|'.join(str(x) for x in parts).encode('utf-8')).hexdigest()


def _rr(v, n=6):
    x = round(float(v), n) + 0.0
    return 0.0 if x == 0 else x


def _quat_wxyz(R):
    return [_rr(q, 10) for q in SP.rot_to_quat(np.asarray(R, float)[None])[0]]


def _times(drive: SP.Drive, st: RedriveSettings, tag: str | None, kind: str, period: float) -> np.ndarray:
    """The stamps a channel is rendered at within the span: the drive's own (a camera of the same tag, else its first
    camera; a lidar the drive's first lidar's sweep ends) or every 1 / rate_hz (a lidar every period)."""
    dur = drive.duration()
    t0 = st.t_start
    t1 = min(st.t_end, dur) if st.t_end > 0 else dur
    if st.schedule == 'rate':
        dt = period if kind == 'lidar' else 1.0 / st.rate_hz
        start = t0 + (dt if kind == 'lidar' else 0.0)
        return np.arange(start, t1 + 1e-9, dt)
    if kind == 'camera':
        src = drive.cameras.get(tag) or max(drive.cameras.values(), key=lambda c: (len(c.frames), c.tag == 'F'))
        ts = np.array([fr[0] for fr in src.frames])
    else:
        L = next(iter(drive.lidars.values()), None)
        ts = np.array([sw[0] for sw in L.sweeps]) if L is not None else np.arange(t0 + period, t1 + 1e-9, period)
    return ts[(ts >= t0 - 1e-9) & (ts <= t1 + 1e-9)]


def plan_export(project, variant, drive: SP.Drive, st: RedriveSettings) -> dict:
    """What an export writes: per channel its stamps and the keyframes (the dataset's rule: every n-th sweep of the
    first lidar, else frame of the first camera, n = its rate / keyframe_hz)."""
    from . import exporters as EX
    ctags = EX.safe_tags(variant)
    stags = EX.sensor_tags(variant)
    chans = []
    for c in variant.cameras:
        if c.visible:
            tag = ctags[c.uid]
            chans.append({'kind': 'camera', 'sensor': c, 'tag': tag, 'channel': f'CAM_{tag.upper()}',
                          'topic': '/' + _ros(tag), 'stamps': _times(drive, st, c.tag, 'camera', 0.1)})
    if st.lidar:
        for s in variant.sensors:
            if s.visible and s.kind == 'lidar':
                md = project.sensor_model_for(s)
                tag = stags[s.uid]
                chans.append({'kind': 'lidar', 'sensor': s, 'tag': tag, 'channel': f'LIDAR_{tag.upper()}',
                              'topic': '/' + _ros(tag),
                              'stamps': _times(drive, st, None, 'lidar', 1.0 / max(md.rate_hz, 0.1))})
    chans = [c for c in chans if len(c['stamps'])]
    if not chans:
        raise ValueError('Nothing to re-shoot: the variant has no visible camera (or lidar) with a frame in the span.')
    ref = next((c for c in chans if c['kind'] == 'lidar'), chans[0])
    st_ = ref['stamps']
    rate = 1.0 / float(np.median(np.diff(st_))) if len(st_) > 1 else st.keyframe_hz
    n = max(1, int(round(rate / st.keyframe_hz)))
    samples = st_[n - 1::n] if len(st_) >= n else st_[:1]
    for c in chans:
        j = np.abs(c['stamps'][None, :] - samples[:, None]).argmin(1)
        seen = {}
        for i, jj in enumerate(j):
            if jj not in seen or abs(c['stamps'][jj] - samples[i]) < abs(c['stamps'][jj] - samples[seen[jj]]):
                seen[int(jj)] = i
        c['keys'] = seen
    return {'chans': chans, 'samples': samples, 'keyframe_hz': rate / n}


def _ros(tag):
    from . import dataset as DSET
    return DSET._ros_name(tag)


def _out_dirs(out_dir):
    out = os.path.abspath(os.path.expanduser(str(out_dir)))
    if os.path.isfile(out):
        raise ValueError(f'{out} is a file: choose a folder.')
    if os.path.isdir(out) and os.listdir(out) and not os.path.isfile(os.path.join(out, 'redrive_export.json')):
        raise ValueError(f'{out} already holds files a re-shot export did not write: choose an empty folder.')
    return out, out + '.part'


def is_complete(out_dir) -> bool:
    """True when out_dir holds a finished re-shot export (its redrive_export.json says complete)."""
    d = SP._load_json(os.path.join(str(out_dir), 'redrive_export.json'))
    return isinstance(d, dict) and d.get('schema') == EXPORT_SCHEMA and d.get('complete') is True


def export_dataset(project, out_dir, settings: RedriveSettings | None = None, variant=None, engine=None,
                   prepared: dict | None = None, cache_root=None, progress=None, cancel=None) -> dict | None:
    """Write the re-shot drive (module docstring, 1). Returns the export's document (redrive_export.json), None when
    cancelled (nothing is left behind)."""
    import cv2
    from . import dataset as DSET
    from . import exporters as EX
    from . import groundtruth as GT
    st = settings or settings_of(project)
    stop = cancel or (lambda: False)
    t_start = time.perf_counter()
    out, part = _out_dirs(out_dir)
    try:
        pr = prepared or prepare(project, st, engine, cache_root, lambda f, t='': _prog(progress, 0.2 * f, t), stop)
    except (Cancelled, SP.Cancelled):
        return None
    drive, scene, ev, calib = pr['drive'], pr['scene'], pr['evidence'], pr['calibration']
    engine = pr.get('engine') or engine
    sst = SP.settings_of(project)
    ssr = _splat_settings_for(sst, st)
    v = variant or proposed_variant(project, st, sst)
    p = DSET._project_for(project, v)
    plan = plan_export(p, v, drive, st)
    scale = image_scale_of(drive, st)
    ts = SP.TorchScene(scene, drive, pr['dev'])
    if os.path.isdir(part):
        if os.listdir(part) and not os.path.isfile(os.path.join(part, 'nuscenes', MARKER)):
            raise ValueError(f'{part} holds files a re-shot export did not write: move it away first.')
        shutil.rmtree(part)
    os.makedirs(part)
    try:
        doc = _write_export(p, v, drive, scene, ev, calib, sst, ssr, st, plan, scale, ts, engine, part,
                            lambda f, t='': _prog(progress, 0.2 + 0.8 * f, t), stop, EX, DSET, GT, cv2)
    except (Cancelled, SP.Cancelled):
        shutil.rmtree(part, ignore_errors=True)
        return None
    except BaseException:
        shutil.rmtree(part, ignore_errors=True)
        raise
    doc['seconds'] = round(time.perf_counter() - t_start, 2)
    doc['complete'] = True
    with open(os.path.join(part, 'redrive_export.json'), 'w', encoding='utf-8') as fh:
        fh.write(json.dumps(doc, indent=1, default=float) + '\n')
    if os.path.isdir(out):
        shutil.rmtree(out)                                # only a finished re-shot export (checked above)
    os.replace(part, out)
    doc['out_dir'] = out
    _prog(progress, 1.0, 'Done')
    return doc


def _write_export(p, v, drive, scene, ev, calib, sst, ssr, st, plan, scale, ts, engine, root, progress, stop, EX,
                  DSET, GT, cv2) -> dict:
    from . import rosbag as RB
    from . import detection as DT
    nus = os.path.join(root, 'nuscenes')
    os.makedirs(os.path.join(nus, VERSION), exist_ok=True)
    with open(os.path.join(nus, MARKER), 'w') as fh:
        fh.write('Written by Vantage (core.redrive, a re-shot drive).\n')
    run_key = _token('vantage', 'redrive', drive.key, v.uid, json.dumps(v.to_dict(), sort_keys=True, default=str),
                     json.dumps(st.to_dict(), sort_keys=True))
    t0_us = int(drive.t0_us)
    us = lambda t: t0_us + int(round(float(t) * 1e6))  # noqa: E731
    logname = 'vantage-redrive-' + _dt.datetime.fromtimestamp(t0_us / 1e6, _dt.timezone.utc).strftime(
        '%Y-%m-%d-%H-%M-%S') + '+0000'
    write_gt = st.ground_truth == 'always' or (st.ground_truth == 'auto' and drive.gt is not None)
    write_gt = write_gt and bool(drive.tracks)
    # ---- calibration rows: the nuscenes_calib export's, intrinsics scaled to the images (as core.dataset)
    ctx = EX._context(p, None, EX.ExportOptions(frame='rear_axle'))
    models = {cm.cam.uid: cm for cm in ctx.models}
    calib_rows = dict(EX._nuscenes(ctx))
    cal_by_sensor = {r['sensor_token']: r for r in json.loads(calib_rows['nuscenes/calibrated_sensor.json'])}
    sens_by_chan = {r['channel']: r for r in json.loads(calib_rows['nuscenes/sensor.json'])}
    sensor_tab, cal_tab, rig_cams, rig_lids = [], [], [], []
    for c in plan['chans']:
        srow = dict(sens_by_chan[c['channel']])
        row = dict(cal_by_sensor[srow['token']])
        if c['kind'] == 'camera':
            cam = c['sensor']
            cm = models[cam.uid]
            w, h, K, dmodel, dcoef = DSET._camera_setup(p, cm, scale, False)
            row['camera_intrinsic'] = [[EX._r(K[i, j], 6) for j in range(3)] for i in range(3)]
            vd = dict(row.get('vantage_distortion', {}))
            vd['image_size'] = [w, h]
            vd['image_scale'] = [EX._r(w / cm.intr.width, 9), EX._r(h / cm.intr.height, 9)]
            row['vantage_distortion'] = vd
            spec = camera_spec(p, v, cam, scale, ssr, engine, (w, h))
            cspec = camera_spec(p, v, cam, scale, ssr, engine, _conf_size(spec.model, st.conf_scale))
            intr = cm.intr
            c.update(w=w, h=h, K=K, dmodel=dmodel, dcoef=dcoef, spec=spec, cspec=cspec, cs_token=row['token'])
            rig_cams.append({'uid': cam.uid, 'tag': cam.tag, 'safe_tag': c['tag'], 'name': cam.name,
                             'channel': c['channel'], 'topic': c['topic'], 'frame_id': f'{c["tag"]}_optical_frame',
                             'pose_vehicle': {k: float(getattr(cam, k)) for k in ('x', 'y', 'z', 'yaw', 'pitch',
                                                                                    'roll')},
                             'image': {'width': w, 'height': h, 'K': K.tolist(), 'distortion_model': dmodel,
                                       'D': list(dcoef), 'rectified': False,
                                       'sensor_size': [intr.width, intr.height], 'lens_projection': intr.projection,
                                       'max_theta_deg': math.degrees(intr.max_theta)},
                             'lens': {'model': 'lens', 'projection': intr.projection, 'f_px': float(intr.f_px),
                                      'kb': [float(x) for x in intr.kb], 'cx': float(intr.cx), 'cy': float(intr.cy),
                                      'aspect': float(intr.aspect), 'max_theta_deg': math.degrees(intr.max_theta),
                                      'sensor_size': [int(intr.width), int(intr.height)], 'image_size': [w, h]},
                             'readout_ms': round(spec.readout_s * 1000.0, 4),
                             'module': p.module_for(cam).name, 'lens_name': p.lens_for(cam).name})
        else:
            s = c['sensor']
            md = p.sensor_model_for(s)
            lspec = next(L for L in SP.variant_rig(p, v, ssr, None).lidars if L.tag == s.tag)
            lspec.body = SP._lidar_body(p, engine, v, lspec)
            c.update(lspec=lspec, cs_token=row['token'], period=1.0 / max(md.rate_hz, 0.1))
            rig_lids.append({'uid': s.uid, 'tag': s.tag, 'safe_tag': c['tag'], 'name': s.name,
                             'model': f'{md.vendor} {md.name}'.strip(), 'channel': c['channel'], 'topic': c['topic'],
                             'frame_id': f'{c["tag"]}_link',
                             'pose_vehicle': {k: float(getattr(s, k)) for k in ('x', 'y', 'z', 'yaw', 'pitch',
                                                                                  'roll')},
                             'rings': int(len(lspec.elevations)), 'columns': int(len(lspec.az)),
                             'rate_hz': float(md.rate_hz), 'range_min_m': float(md.range_min_m),
                             'scan': 'spinning' if md.spinning else 'solid_state',
                             'rotation': ('counter-clockwise' if getattr(md, 'rotation', 'cw') == 'ccw'
                                          else 'clockwise')})
        sensor_tab.append(srow)
        cal_tab.append(row)
    # ---- tokens
    samples = plan['samples']
    sample_us = [us(t) for t in samples]
    sample_tok = [_token('vantage', 'sample', run_key, x) for x in sample_us]
    for c in plan['chans']:
        c['tokens'] = [_token('vantage', 'sample_data', run_key, c['channel'], us(t)) for t in c['stamps']]
    # ---- the bag
    bag = None
    bag_rel = None
    if st.rosbag:
        name = f'vantage_redrive_{re.sub(r"[^a-z0-9]+", "_", (p.name + "_" + v.name).lower()).strip("_")}'
        bag_rel = os.path.join('rosbag', name)
        bag = RB.BagWriter(os.path.join(root, bag_rel), overwrite=True)
        for topic, ty in (('/tf_static', 'tf2_msgs/msg/TFMessage'), ('/tf', 'tf2_msgs/msg/TFMessage'),
                          ('/odom', 'nav_msgs/msg/Odometry'), ('/ground_truth/boxes', 'std_msgs/msg/String')):
            bag.add_topic(topic, ty)
        for c in plan['chans']:
            if c['kind'] == 'camera':
                bag.add_topic(c['topic'] + '/image_raw/compressed', 'sensor_msgs/msg/CompressedImage')
                bag.add_topic(c['topic'] + '/camera_info', 'sensor_msgs/msg/CameraInfo')
            else:
                bag.add_topic(c['topic'] + '/points', 'sensor_msgs/msg/PointCloud2')
        from types import SimpleNamespace
        tf_chans = [SimpleNamespace(kind=c['kind'], sensor=c['sensor'], tag=c['tag']) for c in plan['chans']]
        bag.write('/tf_static', us(0) * 1000, {'transforms': DSET._static_tfs(ctx, tf_chans, RB, us(0) * 1000)})
    try:
        return _write_frames(p, v, drive, scene, ev, calib, sst, ssr, st, plan, ts, root, nus, run_key, us, logname,
                             write_gt, sensor_tab, cal_tab, rig_cams, rig_lids, sample_us, sample_tok, bag, bag_rel,
                             progress, stop, DSET, GT, RB, DT, cv2)
    finally:
        if bag is not None and not bag.closed:
            try:
                bag.close()
            except Exception:                         # noqa: BLE001 — closing after a failure
                pass


def _write(root, rel, data) -> int:
    path = os.path.join(root, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as fh:
        fh.write(data)
    return len(data)


def _png(arr) -> bytes:
    import cv2
    ok, buf = cv2.imencode('.png', arr)
    if not ok:
        raise OSError('PNG encoding failed')
    return bytes(buf)


def _write_frames(p, v, drive, scene, ev, calib, sst, ssr, st, plan, ts, root, nus, run_key, us, logname, write_gt,
                  sensor_tab, cal_tab, rig_cams, rig_lids, sample_us, sample_tok, bag, bag_rel, progress, stop, DSET,
                  GT, RB, DT, cv2) -> dict:
    events = sorted([(float(t), ci, k) for ci, c in enumerate(plan['chans']) for k, t in enumerate(c['stamps'])],
                    key=lambda e: (e[0], plan['chans'][e[1]]['kind'] != 'camera', e[1]))
    nb = {'nuscenes': 0, 'ground_truth': 0, 'confidence': 0}
    sd_rows, ego_rows, gt_frames, coco, conf_frames = [], [], {}, {}, {}
    written = {ci: {} for ci in range(len(plan['chans']))}
    lidar_counts = {}                                   # (sample index, track id) -> rendered lidar points in its box
    per_chan = {c['channel']: {'frames': 0, 'bytes': 0} for c in plan['chans']}
    inst_tok = {i: _token('vantage', 'instance', run_key, i) for i in drive.tracks}
    jpeg = [int(cv2.IMWRITE_JPEG_QUALITY), int(st.jpeg_quality)]
    conf_err, conf_pw = [], []
    tf_next = 0.0
    gt_next = 0.0
    for n_ev, (t, ci, k) in enumerate(events):
        if stop():
            raise Cancelled()
        c = plan['chans'][ci]
        _prog(progress, n_ev / max(len(events), 1), f'Re-shooting · {c["channel"]} · t = {t:.2f} s')
        Rw, pw = drive.traj.pose(t)
        T_e = np.eye(4)
        T_e[:3, :3], T_e[:3, 3] = Rw, pw + drive.origin
        is_key = k in c['keys']
        sub = 'samples' if is_key else 'sweeps'
        tok = c['tokens'][k]
        t_us = us(t)
        if bag is not None:
            while tf_next <= t + 1e-9:
                tn = us(tf_next) * 1000
                Rt, pt = drive.traj.pose(tf_next)
                Tt = np.eye(4)
                Tt[:3, :3], Tt[:3, 3] = Rt, pt + drive.origin
                bag.write('/tf', tn, {'transforms': [RB.transform_stamped(Tt, tn, 'odom', 'base_link')]})
                bag.write('/odom', tn, {'header': RB.header(tn, 'odom'), 'child_frame_id': 'base_link',
                                        'pose': {'pose': {'position': {'x': float(Tt[0, 3]), 'y': float(Tt[1, 3]),
                                                                       'z': float(Tt[2, 3])},
                                                          'orientation': RB.quat_msg(Rt)}}})
                tf_next += 1.0 / DSET.TF_HZ
            while gt_next <= t + 1e-9 and drive.tracks:
                tn = us(gt_next) * 1000
                boxes = []
                for i, tr in drive.tracks.items():
                    po = tr.pose(gt_next)
                    if po is None:
                        continue
                    Rb, cb = po
                    boxes.append({'id': int(i), 'instance_token': inst_tok[i], 'category': tr.category,
                                  'name': tr.cls, 'centre': [round(float(q), 4) for q in cb + drive.origin],
                                  'size_lwh': [round(float(q), 4) for q in tr.size],
                                  'yaw': round(math.atan2(Rb[1, 0], Rb[0, 0]), 6)})
                bag.write('/ground_truth/boxes', tn, {'data': json.dumps({'stamp_ns': tn, 'frame_id': 'odom',
                                                                          'boxes': boxes}, separators=(',', ':'))})
                gt_next += 1.0 / DSET.GT_HZ
        if c['kind'] == 'camera':
            spec, cspec = c['spec'], c['cspec']
            rig = SP.RigSpec('proposed', v.name, [spec], [])
            rgb, ids, rng_img = SP._render_rig_frame(ts, drive, rig, t, scene)[spec.tag]
            ok, buf = cv2.imencode('.jpg', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), jpeg)
            data = bytes(buf)
            rel = f'{sub}/{c["channel"]}/{logname}__{c["channel"]}__{t_us}.jpg'
            nb['nuscenes'] += _write(nus, rel, data)
            per_chan[c['channel']]['bytes'] += len(data)
            # confidence
            fe = conf_view(ts, scene, ev, drive, cspec, t, spec.model.fx, calib, st.error_budget)
            base = os.path.splitext(os.path.basename(rel))[0]
            E = fe['expected']
            e16 = np.where(np.isfinite(E), np.clip(np.rint(E * CONF_SCALE_PNG), 1, 65535), 0).astype(np.uint16)
            r8 = np.where(fe['reason'] >= 0, fe['reason'] + 1, 0).astype(np.uint8)
            crel = f'confidence/{c["channel"]}/{base}.confidence.png'
            rrel = f'confidence/{c["channel"]}/{base}.reason.png'
            nb['confidence'] += _write(root, crel, _png(e16)) + _write(root, rrel, _png(r8))
            vv = fe['valid']
            conf_err.append(float(np.nanmean(E[vv])) if vv.any() else float('nan'))
            conf_pw.append(float(np.nanmean(fe['p_within'][vv])) if vv.any() else float('nan'))
            conf_frames.setdefault(spec.tag, []).append({
                'sample_data_token': tok, 'timestamp': t_us, 'image': f'nuscenes/{rel}', 'confidence': crel,
                'reason': rrel, 'size': [cspec.model.width, cspec.model.height],
                'mean_expected_error': round(conf_err[-1], 3) if math.isfinite(conf_err[-1]) else None,
                'within_share': round(conf_pw[-1], 4) if math.isfinite(conf_pw[-1]) else None})
            if write_gt:
                objs = SP._objects_in_view(drive, t, spec, ids, Rw, pw, SP._rays_of(spec.model))
                sem = np.zeros(ids.shape, np.uint8)
                if spec.body is not None:
                    sem[spec.body] = GT.EGO_CLASS
                rles = []
                for o in objs:
                    tr = drive.tracks[o['id']]
                    o['instance_token'] = inst_tok[o['id']]
                    o['bbox_full'] = _bbox_full(tr, spec, t, drive, GT)
                    mask = ids == o['id']
                    sem[mask] = GT.CLASS_ID.get(o['class'], 0)
                    rles.append(GT.rle_encode(mask))
                g = {}
                for kind, arr in (('semantic', sem), ('instance', np.clip(ids, 0, 65535).astype(np.uint16)),
                                  ('range', GT.range_png(rng_img))):
                    grel = f'ground_truth/{c["channel"]}/{base}.{kind}.png'
                    nb['ground_truth'] += _write(root, grel, _png(arr))
                    g[kind] = grel
                fr = {'sample_data_token': tok, 'timestamp': t_us, 'image': f'nuscenes/{rel}', **g,
                      'objects': objs, 'confidence': crel}
                if bag is not None:
                    fr['bag'] = {'topic': c['topic'] + '/image_raw/compressed', 'stamp_ns': t_us * 1000}
                gt_frames.setdefault(spec.tag, []).append(fr)
                coco.setdefault(spec.tag, []).append((fr, rles))
            if bag is not None:
                tn = t_us * 1000
                frame = f'{c["tag"]}_optical_frame'
                bag.write(c['topic'] + '/image_raw/compressed', tn, RB.compressed_image(data, tn, frame))
                bag.write(c['topic'] + '/camera_info', tn, RB.camera_info(c['w'], c['h'], c['dmodel'], c['dcoef'],
                                                                          c['K'], None, None, tn, frame))
            written[ci][k] = (t_us, rel, is_key, c['w'], c['h'], 'jpg')
        else:
            L = c['lspec']
            sw = SP.render_lidar(ts, drive.traj, L, t - c['period'], ssr)
            pts = sw['points']
            a = np.zeros((len(pts), 5), np.float32)
            a[:, :3] = pts[:, :3]
            a[:, 4] = pts[:, 4]
            rel = f'{sub}/{c["channel"]}/{logname}__{c["channel"]}__{t_us}.pcd.bin'
            data = a.tobytes()
            nb['nuscenes'] += _write(nus, rel, data)
            per_chan[c['channel']]['bytes'] += len(data)
            if is_key:
                Pv = SP.lidar_points_vehicle(L, sw, drive.traj, t)
                si = c['keys'][k]
                for i, tr in drive.tracks.items():
                    po = tr.pose(t)
                    if po is None or not len(Pv):
                        continue
                    Rb, cb = po
                    q = (Pv - Rw.T @ (cb - pw)) @ (Rw.T @ Rb)
                    lidar_counts[(si, i)] = lidar_counts.get((si, i), 0) + int(
                        np.all(np.abs(q) <= np.asarray(tr.size) / 2, 1).sum())
            if bag is not None:
                tn = t_us * 1000
                b = np.empty(len(pts), DSET.LIDAR_PC2)
                b['x'], b['y'], b['z'] = pts[:, 0], pts[:, 1], pts[:, 2]
                b['intensity'] = 0.0
                b['ring'] = pts[:, 4].astype(np.uint16)
                b['t'] = (sw['time'][sw['col']] - (t - c['period'])).astype(np.float32)
                bag.write(c['topic'] + '/points', tn, RB.pointcloud2(b, tn, f'{c["tag"]}_link'))
            written[ci][k] = (t_us, rel, is_key, 0, 0, 'pcd')
        per_chan[c['channel']]['frames'] += 1
        ego_rows.append({'token': tok, 'timestamp': t_us, 'rotation': _quat_wxyz(Rw),
                         'translation': [_rr(q) for q in pw + drive.origin]})
    # ---- the drive's own poses in the span (the trajectory a re-drive of this folder interpolates)
    t_lo = min(float(c['stamps'][0]) for c in plan['chans']) - 0.2
    t_hi = max(float(c['stamps'][-1]) for c in plan['chans']) + 0.2
    have = {r['timestamp'] for r in ego_rows}
    for t, pos, q in zip(drive.traj.t, drive.traj.pos, drive.traj.quat):
        if t_lo <= t <= t_hi and us(t) not in have:
            ego_rows.append({'token': _token('vantage', 'ego_pose', run_key, us(t)), 'timestamp': us(t),
                             'rotation': [_rr(x, 10) for x in q], 'translation': [_rr(x) for x in pos + drive.origin]})
    # ---- sample_data rows
    for ci, c in enumerate(plan['chans']):
        ks = sorted(written[ci])
        for i, k in enumerate(ks):
            t_us, rel, is_key, w, h, ext = written[ci][k]
            si = c['keys'].get(k)
            if si is None:
                si = int(min(np.searchsorted(sample_us, t_us, 'left'), len(sample_us) - 1))
            sd_rows.append({'token': c['tokens'][k], 'sample_token': sample_tok[si], 'ego_pose_token': c['tokens'][k],
                            'calibrated_sensor_token': c['cs_token'], 'timestamp': t_us, 'fileformat': ext,
                            'is_key_frame': bool(is_key), 'height': h, 'width': w, 'filename': rel,
                            'prev': c['tokens'][ks[i - 1]] if i else '',
                            'next': c['tokens'][ks[i + 1]] if i + 1 < len(ks) else ''})
    # ---- annotations at the keyframes: the drive's boxes (world frame) carried over
    cat_tok = lambda n: _token('vantage', 'category', n)  # noqa: E731
    cats = set()
    ann_rows, by_inst = [], {}
    for si, ts_ in enumerate(plan['samples']):
        for i, tr in drive.tracks.items():
            po = tr.pose(float(ts_))
            if po is None or not tr.category:
                continue
            Rb, cb = po
            l_, w_, h_ = tr.size
            a = {'token': _token('vantage', 'sample_annotation', run_key, si, i), 'sample_token': sample_tok[si],
                 'instance_token': inst_tok[i], 'visibility_token': '', 'attribute_tokens': [],
                 'translation': [_rr(q) for q in cb + drive.origin], 'size': [_rr(w_, 4), _rr(l_, 4), _rr(h_, 4)],
                 'rotation': _quat_wxyz(Rb), 'prev': '', 'next': '',
                 'num_lidar_pts': int(lidar_counts.get((si, i), 0)), 'num_radar_pts': 0}
            ann_rows.append(a)
            by_inst.setdefault(i, []).append(a)
            cats.add(tr.category)
    inst_rows = []
    for i, rows in by_inst.items():
        for j, a in enumerate(rows):
            a['prev'] = rows[j - 1]['token'] if j else ''
            a['next'] = rows[j + 1]['token'] if j + 1 < len(rows) else ''
        inst_rows.append({'token': inst_tok[i], 'category_token': cat_tok(drive.tracks[i].category),
                          'nbr_annotations': len(rows), 'first_annotation_token': rows[0]['token'],
                          'last_annotation_token': rows[-1]['token']})
    valid_inst = {r['token'] for r in inst_rows}
    log_tok = _token('vantage', 'log', run_key)
    scene_tok = _token('vantage', 'scene', run_key)
    sample_rows = [{'token': tok, 'timestamp': sample_us[i], 'prev': sample_tok[i - 1] if i else '',
                    'next': sample_tok[i + 1] if i + 1 < len(sample_tok) else '', 'scene_token': scene_tok}
                   for i, tok in enumerate(sample_tok)]
    map_rows = []
    src_map = _source_map(drive)
    if src_map is not None:
        mtok, mrel, mpath = src_map
        with open(mpath, 'rb') as fh:
            nb['nuscenes'] += _write(nus, mrel, fh.read())
        map_rows = [{'token': mtok, 'log_tokens': [log_tok], 'category': 'semantic_prior', 'filename': mrel}]
    extra = [(n, '') for n in sorted(cats) if n not in GT.NUSC_INDEX]
    tables = {
        'attribute': [{'token': _token('vantage', 'attribute', n), 'name': n, 'description': d}
                      for n, d in DSET.ATTRIBUTES],
        'calibrated_sensor': cal_tab,
        'category': GT.nuscenes_categories(cat_tok, extra),
        'ego_pose': sorted(ego_rows, key=lambda r: r['timestamp']),
        'instance': inst_rows,
        'log': [{'token': log_tok, 'logfile': logname, 'vehicle': f'{p.vehicle.name or "vehicle"} (re-shot rig '
                                                                   f'{v.name})',
                 'date_captured': _dt.datetime.fromtimestamp(us(0) / 1e6, _dt.timezone.utc).strftime('%Y-%m-%d'),
                 'location': 'redrive'}],
        'map': map_rows,
        'sample': sample_rows,
        'sample_annotation': ann_rows,
        'sample_data': sorted(sd_rows, key=lambda r: r['timestamp']),
        'scene': [{'token': scene_tok, 'log_token': log_tok, 'nbr_samples': len(sample_rows),
                   'first_sample_token': sample_tok[0], 'last_sample_token': sample_tok[-1], 'name': 'scene-0001',
                   'description': f'Re-shot drive: {os.path.basename(os.path.normpath(drive.path))} re-rendered '
                                  f'through {p.name} / {v.name}'}],
        'sensor': sensor_tab,
        'visibility': [{'token': t, 'level': lv, 'description': d} for t, lv, d in DSET.VISIBILITY],
    }
    for name, rows in tables.items():
        nb['nuscenes'] += _write(nus, f'{VERSION}/{name}.json', (json.dumps(rows, indent=1) + '\n').encode('utf-8'))
    # ---- ground truth, confidence index, rig file, dataset.json, README
    src_ds = SP._load_json(os.path.join(drive.root, 'dataset.json')) or {}
    if write_gt:
        for frames in gt_frames.values():
            for fr in frames:
                for o in fr['objects']:
                    if o.get('instance_token') not in valid_inst:
                        o['instance_token'] = None
        cams = {tag: {'channel': next(c['channel'] for c in plan['chans'] if c['kind'] == 'camera'
                                      and c['sensor'].tag == tag),
                      'width': next(c['w'] for c in plan['chans'] if c['kind'] == 'camera' and c['sensor'].tag == tag),
                      'height': next(c['h'] for c in plan['chans'] if c['kind'] == 'camera'
                                     and c['sensor'].tag == tag),
                      'frames': frames} for tag, frames in gt_frames.items()}
        gdoc = {'schema': GT.SCHEMA, 'project': p.name, 'variant': v.name, 'classes': GT.classes_doc(),
                'cameras': cams, 'lidars': {},
                'nuscenes_lidarseg_categories': [{'index': i, 'name': n} for i, n, _ in GT.NUSCENES_LIDARSEG],
                'conventions': {
                    'paths': 'relative to this drive\'s folder (the one holding ground_truth/ and nuscenes/)',
                    'pixels': 'output image pixel edges: pixel (i, j) covers [i, i+1) × [j, j+1); bbox [x0, y0, x1, '
                              'y1]',
                    'semantic': 'uint8 PNG: the annotated objects\' class ids and the own vehicle (29); everything '
                                'else 0 (unlabelled: the re-render knows no other classes)',
                    'instance': 'uint16 PNG, 0 = no object, else the object\'s id (the source drive\'s): the pixels '
                                'whose front-most Gaussian (of at least half the opacity) is the object\'s',
                    'range': f'uint16 PNG = round(range_m · {GT.RANGE_SCALE:g}) along the pixel\'s ray (the '
                             're-render\'s planar depth), 0 = nothing',
                    'box3d': 'vehicle frame (ISO 8855, rear axle on the ground) at the frame\'s time: centre, size '
                             '[length, width, height] m, yaw degrees (the source drive\'s boxes, carried over in the '
                             'world frame)',
                    'occluded': 'of the rays meeting the 3D box, the share hidden by something nearer in the '
                                're-render', 'truncated': 'share of the 3D box\'s surface outside the image',
                    'confidence': 'the frame\'s confidence sidecar (confidence/index.json)'},
                'notes': ['A re-shot drive (core.redrive): every image is the log replay\'s reconstruction re-rendered '
                          'through the proposed rig; objects the reconstruction did not rebuild are not in the images '
                          'and not listed.']}
        nb['ground_truth'] += _write(root, GT.INDEX, (json.dumps(gdoc, indent=1, default=_jdef) + '\n').encode())
        cats_c = [{'id': cc[0], 'name': cc[1], 'supercategory': cc[3].split('.')[0]} for cc in GT.CLASSES if cc[0] > 0]
        for tag, items in coco.items():
            images, anns = [], []
            for i, (f, rles) in enumerate(items):
                images.append({'id': i + 1, 'file_name': f['image'], 'width': cams[tag]['width'],
                               'height': cams[tag]['height'], 'sample_data_token': f['sample_data_token'],
                               'timestamp': f['timestamp']})
                for o, rle in zip(f['objects'], rles):
                    x0, y0, x1, y1 = o['bbox']
                    anns.append({'id': len(anns) + 1, 'image_id': i + 1, 'category_id': GT.CLASS_ID.get(o['class'], 0),
                                 'bbox': [x0, y0, x1 - x0, y1 - y0], 'area': o['pixels'], 'iscrowd': 0,
                                 'segmentation': rle, 'object_id': o['id'], 'actor_id': o['actor_id'],
                                 'instance_token': o['instance_token'], 'nuscenes_category': o['category'],
                                 'occluded': o['occluded'], 'truncated': o['truncated']})
            nb['ground_truth'] += _write(root, f'ground_truth/coco/{cams[tag]["channel"]}.json', (json.dumps(
                {'info': {'description': f'Vantage re-shot drive {p.name} / {v.name}, {cams[tag]["channel"]}',
                          'version': GT.SCHEMA}, 'images': images, 'annotations': anns, 'categories': cats_c},
                separators=(',', ':'), default=_jdef) + '\n').encode())
    cal = calib.to_dict() if calib is not None else None
    cdoc = {'schema': CONF_SCHEMA, 'units': {'confidence': f'uint16 PNG = round(expected error · {CONF_SCALE_PNG:g}),'
                                                           ' 0 = no value (outside the lens\'s field or on the own '
                                                           'vehicle); expected error: mean absolute difference per '
                                                           'pixel from the real image, grey levels 0–255, at '
                                                           'conf_scale of the image',
                                             'reason': 'uint8 PNG: 0 no value, else 1 + index into reasons (ok: '
                                                       'P(error ≤ budget) ≥ ½)'},
            'reasons': list(REASONS), 'reason_text': REASON_TEXT, 'budget': st.error_budget,
            'conf_scale': st.conf_scale, 'cameras': conf_frames, 'calibration': cal}
    nb['confidence'] += _write(root, 'confidence/index.json', (json.dumps(cdoc, indent=1, default=_jdef) + '\n')
                               .encode())
    rig_doc = {'schema': 'vantage.sim_rig/1', 'generator': 'vantage.redrive', 'project': p.name, 'variant': v.name,
               'variant_uid': v.uid, 'cameras': rig_cams, 'lidars': rig_lids, 'radars': [], 'imus': [], 'gnss': [],
               'notes': ['Written by a re-shot export (core.redrive), not by the CARLA bridge: the camera and lidar '
                         'blocks a reader of the calibration needs.']}
    _write(root, 'rig/sim_rig.json', (json.dumps(rig_doc, indent=1, default=_jdef) + '\n').encode())
    bag_res = None
    if bag is not None:
        bag_res = bag.close()
    readouts = sorted({rc['readout_ms'] for rc in rig_cams})
    scene_doc = dict(src_ds.get('scene') or {})
    span = round(float(max(c['stamps'][-1] for c in plan['chans']) - min(c['stamps'][0] for c in plan['chans'])), 6)
    scene_doc.setdefault('duration_s', span)               # the source scene's (it is the same scene)
    scene_doc.update({'source_drive': os.path.basename(os.path.normpath(drive.path)), 'source_variant': drive.variant,
                      'reshot_span_s': span,
                      'samples': len(sample_rows), 'keyframe_hz': plan['keyframe_hz'],
                      'camera_readout_ms': readouts[-1] if readouts else 0.0})
    if len(readouts) > 1:
        scene_doc['camera_readout_ms_per_camera'] = {rc['tag']: rc['readout_ms'] for rc in rig_cams}
    channels = [{'kind': c['kind'], 'tag': c['sensor'].tag, 'channel': c['channel'], 'topic': c['topic'],
                 'frames': per_chan[c['channel']]['frames'], 'keyframes': len(c['keys']),
                 'bytes': per_chan[c['channel']]['bytes'], 'planned_frames': len(c['stamps'])} for c in plan['chans']]
    counts = {'samples': len(sample_rows), 'sample_data': len(sd_rows), 'ego_poses': len(ego_rows),
              'annotations': len(ann_rows), 'instances': len(inst_rows),
              'bag_messages': int(bag_res['messages']) if bag_res else 0}
    if write_gt:
        counts['gt_frames'] = sum(len(f) for f in gt_frames.values())
        counts['gt_objects'] = sum(len(fr['objects']) for f in gt_frames.values() for fr in f)
    ds = {'schema': DSET.SCHEMA, 'generator': 'redrive', 'project': p.name, 'variant': v.name,
          'written': {'nuscenes': 'nuscenes', 'nuscenes_version': VERSION, 'rosbag': bag_rel},
          'settings': {'duration_s': span, 'image_scale': image_scale_of(drive, st),
                       'rolling_shutter': st.rolling_shutter == 'module' and any(r > 0 for r in readouts),
                       'keyframe_hz': st.keyframe_hz, 'jpeg_quality': st.jpeg_quality, 'nuscenes': True,
                       'rosbag': bool(st.rosbag), 'scene': 'redrive', 'schedule': st.schedule, 'rate_hz': st.rate_hz,
                       'span_s': [st.t_start, st.t_end]},
          'scene': scene_doc, 'counts': counts,
          'bytes': dict(nb, rosbag=int(bag_res['bytes']) if bag_res else 0),
          'ground_truth': ({'index': GT.INDEX, 'schema': GT.SCHEMA, 'cameras': sorted(gt_frames), 'lidars': []}
                           if write_gt else None),
          'confidence': {'index': 'confidence/index.json', 'schema': CONF_SCHEMA},
          'channels': channels,
          'conventions': {
              'global_frame': 'the source drive\'s global frame (nuScenes global / bag odom)',
              'ego_frame': 'vehicle, ISO 8855: x forward, y left, z up, origin at the rear-axle centre on the ground',
              'time': 'microseconds (nuScenes) / nanoseconds (bag) on the source drive\'s clock',
              'camera_timestamp': 'the middle row\'s exposure centre (rows at their own times with a rolling shutter)',
              'lidar_timestamp': 'end of the sweep; points in the sensor frame at their firing pose; .pcd.bin float32 '
                                 'x y z intensity ring (intensity 0: not modelled)',
              'images': 'the variant\'s real lens: camera_intrinsic + vantage_distortion (Kannala-Brandt) as the '
                        'nuscenes_calib export, scaled to the image size; the exact lens block in rig/sim_rig.json'},
          'assumptions': ASSUMPTIONS,
          'redrive': {'source': drive.path, 'source_key': drive.key, 'variant_uid': v.uid, 'run_key': run_key,
                      'scenario': (SP._load_json(os.path.join(drive.root, 'run.json')) or {}).get('scenario'),
                      'calibration_key': calib.key if calib is not None else None,
                      'budget': st.error_budget},
          'notes': ['Radars are not re-rendered: the re-shot drive has no radar channels.']}
    _write(root, 'dataset.json', (json.dumps(ds, indent=1, default=_jdef) + '\n').encode())
    _write(root, 'README.md', _readme(ds, p, v, drive).encode('utf-8'))
    pwm = [x for x in conf_pw if math.isfinite(x)]
    return {'schema': EXPORT_SCHEMA, 'complete': False, 'project': p.name, 'variant': v.name, 'variant_uid': v.uid,
            'source': drive.path, 'run_key': run_key, 'counts': counts, 'bytes': ds['bytes'], 'channels': channels,
            'confidence': {'budget': st.error_budget,
                           'mean_expected_error': round(float(np.nanmean(conf_err)), 3) if conf_err else None,
                           'within_share': round(float(np.mean(pwm)), 4) if pwm else None},
            'ground_truth': bool(write_gt), 'rosbag': bag_rel, 'settings': st.to_dict()}


def _jdef(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)


def _bbox_full(tr, spec, t, drive, GT):
    po = tr.pose(t)
    if po is None:
        return None
    Rb, cb = po
    Rw, pw = drive.traj.pose(t)
    cv = Rw.T @ (cb - pw)
    E = GT.box_edges(cv, tr.size, Rw.T @ Rb, n=9)
    u, v, _, ok = spec.model.project((E - spec.t) @ np.asarray(spec.R, float))
    if not ok.any():
        return None
    return [round(float(u[ok].min()), 2), round(float(v[ok].min()), 2), round(float(u[ok].max()), 2),
            round(float(v[ok].max()), 2)]


def _source_map(drive):
    """(token, relative file name, path) of the source drive's nuScenes map, or None."""
    nd = SP._nus_dirs(drive.path) if drive.source == 'nuscenes' else None
    if nd is None:
        return None
    rows = SP._load_json(os.path.join(nd[0], 'map.json'))
    if not isinstance(rows, list) or not rows or not isinstance(rows[0], dict):
        return None
    rel = rows[0].get('filename')
    path = os.path.join(nd[1], rel) if rel else None
    if not path or not os.path.isfile(path):
        return None
    return rows[0].get('token') or _token('map', rel), rel, path


def _readme(ds, p, v, drive) -> str:
    rows = '\n'.join(f'| {c["channel"]} | {c["kind"]} | {c["frames"]} | {c["keyframes"]} |' for c in ds['channels'])
    return f"""# Vantage re-shot drive · {p.name} / {v.name}

The recorded drive `{os.path.basename(os.path.normpath(drive.path))}` ({drive.variant or 'its own rig'}) re-rendered
through the proposed rig by Vantage's log replay (3D Gaussians rebuilt from the drive's lidar and images), written as
the sample dataset and the CARLA bridge write a drive. It is not a real recording.

- nuScenes: `nuscenes/` version `{VERSION}`: {ds['counts']['samples']} samples, {ds['counts']['sample_data']}
  sample_data, {ds['counts']['annotations']} annotations of {ds['counts']['instances']} instances (the drive's boxes).
- Confidence: `confidence/index.json`; per image `<name>.confidence.png` (expected error, uint16 = grey levels ×
  {CONF_SCALE_PNG:g}) and `<name>.reason.png`.
{('- Ground truth: `ground_truth/index.json` (vantage.ground_truth/1).' if ds.get('ground_truth') else '')}
{('- ROS 2 bag: `' + ds['written']['rosbag'] + '/`.' if ds['written'].get('rosbag') else '')}

| Channel | Kind | Frames | Keyframes |
|---|---|---:|---:|
{rows}

{ASSUMPTIONS}
"""


# =========================================================================== scoring a re-shot drive on a real one
def score_reshoot(reshot_dir, real_dir, budget: float | None = None, max_frames: int = 0, progress=None,
                  cancel=None) -> dict:
    """A re-shot drive against a real recording of the same drive by the same rig (the paired truth): per camera
    channel the frames at the same timestamps, PSNR and SSIM of the re-shot image against the real one (over the
    pixels inside the lens's field: the confidence's valid pixels), and its confidence sidecar against the measured
    error at the sidecar's scale (reliability)."""
    import cv2
    stop = cancel or (lambda: False)
    cidx = SP._load_json(os.path.join(reshot_dir, 'confidence', 'index.json'))
    if not isinstance(cidx, dict) or cidx.get('schema') != CONF_SCHEMA:
        raise ValueError(f'{os.path.basename(reshot_dir)} is not a re-shot drive (no confidence/index.json).')
    budget = float(budget if budget is not None else cidx.get('budget', 12.0))
    nd = SP._nus_dirs(real_dir)
    if nd is None:
        raise ValueError(f'{os.path.basename(real_dir)} is not a nuScenes folder.')
    sd = SP._load_json(os.path.join(nd[0], 'sample_data.json')) or []
    cal = {r['token']: r for r in SP._load_json(os.path.join(nd[0], 'calibrated_sensor.json')) or []}
    sens = {r['token']: r for r in SP._load_json(os.path.join(nd[0], 'sensor.json')) or []}
    real = {}
    for r in sd:
        c = cal.get(r.get('calibrated_sensor_token')) or {}
        ch = (sens.get(c.get('sensor_token')) or {}).get('channel')
        if ch and r.get('fileformat') == 'jpg':
            real[(ch, int(r['timestamp']))] = os.path.join(nd[1], r['filename'])
    per, E_all, M_all = {}, [], []
    jobs = [(tag, f) for tag, fr in (cidx.get('cameras') or {}).items() for f in fr]
    if max_frames and len(jobs) > max_frames:
        jobs = [jobs[int(i)] for i in np.linspace(0, len(jobs) - 1, max_frames).round().astype(int)]
    rng = np.random.default_rng(0)
    for j, (tag, f) in enumerate(jobs):
        if stop():
            return {}
        _prog(progress, j / max(len(jobs), 1), f'Scoring · {tag}')
        ch = os.path.basename(os.path.dirname(f['confidence']))
        rp = real.get((ch, int(f['timestamp'])))
        if rp is None:
            continue
        a = cv2.imread(os.path.join(reshot_dir, f['image']), cv2.IMREAD_COLOR)
        b = cv2.imread(rp, cv2.IMREAD_COLOR)
        cimg = cv2.imread(os.path.join(reshot_dir, f['confidence']), cv2.IMREAD_UNCHANGED)
        if a is None or b is None or cimg is None or a.shape != b.shape:
            continue
        a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        b = cv2.cvtColor(b, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        h, w = cimg.shape
        valid_c = cimg > 0
        valid_f = cv2.resize(valid_c.astype(np.uint8), (a.shape[1], a.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        d = per.setdefault(tag, {'frames': 0, 'psnr': [], 'ssim': [], 'meas': [], 'pred': [], 'wm': [], 'wp': []})
        d['frames'] += 1
        p = SP.SR.psnr(a, b, valid_f)
        if p is not None:
            d['psnr'].append(p)
        s = _ssim(a, b, valid_f)
        if s is not None:
            d['ssim'].append(s)
        ac = cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA)
        bc = cv2.resize(b, (w, h), interpolation=cv2.INTER_AREA)
        err = _pixel_error(ac, bc)
        E = cimg.astype(float) / CONF_SCALE_PNG
        d['meas'].append(float(err[valid_c].mean()))
        d['pred'].append(float(E[valid_c].mean()))
        d['wm'].append(float(np.mean(err[valid_c] <= budget)))
        if f.get('within_share') is not None:
            d['wp'].append(float(f['within_share']))
        ix = _sample(valid_c, 4000, rng)
        E_all.append(E.ravel()[ix])
        M_all.append(err.ravel()[ix])
    cams = {tag: {'frames': d['frames'], 'psnr_db': round(float(np.mean(d['psnr'])), 3) if d['psnr'] else None,
                  'ssim': round(float(np.mean(d['ssim'])), 4) if d['ssim'] else None,
                  'mean_error_meas': round(float(np.mean(d['meas'])), 3) if d['meas'] else None,
                  'mean_error_pred': round(float(np.mean(d['pred'])), 3) if d['pred'] else None,
                  'within_meas': round(float(np.mean(d['wm'])), 4) if d['wm'] else None,
                  'within_pred': round(float(np.mean(d['wp'])), 4) if d['wp'] else None}
            for tag, d in sorted(per.items())}
    rel = reliability(np.concatenate(E_all), np.concatenate(M_all), budget) if E_all else {}
    allp = [x for d in per.values() for x in d['psnr']]
    alls = [x for d in per.values() for x in d['ssim']]
    return {'reshot': os.path.basename(os.path.normpath(reshot_dir)), 'real': os.path.basename(os.path.normpath(
        real_dir)), 'budget': budget, 'frames': sum(d['frames'] for d in per.values()), 'cameras': cams,
            'psnr_db': round(float(np.mean(allp)), 3) if allp else None,
            'ssim': round(float(np.mean(alls)), 4) if alls else None, 'reliability': rel}

