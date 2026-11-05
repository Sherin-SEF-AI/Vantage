"""Rig perception: how well a 3D (bird's-eye) detector finds each class in each region around the car, for each rig, on
the same scenes, with intervals, and rig against rig, paired over the same scenes: "Rig B loses 6.1 points of pedestrian
recall at left · 40–60 m (95 % CI 2.3–9.8, paired over the same scenes)".

Inputs
------
One or more recordings with ground truth, each tagged with the rig it was recorded with (by default its variant's name):
CARLA runs as the bridge writes them (nuscenes/ tables, ground_truth/index.json, rig/sim_rig.json, run.json), or any
nuScenes folder with sample_annotation (a re-driven or re-shot export). Several recordings of one rig (a scene set:
core.perception_scenes) are pooled; the first rig is the reference every other is compared with.

Ground truth
------------
Every annotated object (nuScenes sample_annotation: the CARLA bridge annotates every actor and the map's parked cars
within its annotation range, output.gt_range_m, 60 m, at each keyframe) of the evaluated classes, whether or not any
camera of the rig sees it: an object a rig cannot see at all is a miss of that rig ("invisible to this rig" is the
point of comparing rigs). Between keyframes an object's box is interpolated linearly in the global frame (checked on
the recording data-rig-6-cam-lidar-radars: against the bridge's own per-frame boxes the interpolated centres differ by
a median of 0.05 mm, 95th percentile 4.7 cm, worst 26 cm, over 1552 camera-frame objects); an object annotated at only
one of the two keyframes around an instant (entering or leaving the annotated range), or standing on another level
(its box bottom more than LEVEL_M off the vehicle's ground plane: Town10HD annotates parked cars on a deck 24.6 m
up), is an ignore region there (a detection on it is neither right nor wrong; it is no miss). Objects and detections
farther from the grid's reference point than the evaluated range (the annotation range less the reference point's
distance from the rear axle, so every object counted is annotated; nuScenes likewise filters both by distance) are
left out. Each object at each instant is
also marked visible to the rig (some camera of the rig shows it, not an ignore region by core.detection_eval's rules:
min_px, min_pixels, max_occluded, max_truncated), marginal (shown, but too small, hidden or cut), or invisible (no
camera shows a pixel of it): the share visible per region is the rig's geometric coverage, measured.

Frames
------
The instants are the reference lidar's sweeps (every instant: instants 'all') or the nuScenes keyframes ('keyframes';
an external detector's are always the keyframes); at each, every camera contributes its image nearest the instant
(within half its frame period). Detections are compensated for the ego's motion between each image's capture and the
instant (ego_pose), as a real stack does before it fuses cameras that fire at different moments.

Matching and figures (nuScenes detection, Caesar et al. 2020; nuscenes-devkit eval/detection/algo.py)
--------------------------------------------------------------------------------------------------
* Per class, per instant: detections in descending score order (ties: the later one first, as the devkit sorts),
  each to the nearest untaken object by bird's-eye centre distance, a match when that distance is below the threshold
  (0.5, 1, 2 and 4 m); a detection whose nearest object is an ignore region is ignored. 'vehicle' (car, truck, bus
  without telling them apart) first suppresses a vehicle detection within group_nms_m of a higher-scoring one.
* AP per threshold as the devkit's calc_ap: precision against recall interpolated at 101 recall points (no
  envelope), recall below 0.1 and precision below 0.1 clipped, normalised; mean AP = their mean over the thresholds.
* Recall, precision and the centre error (ATE: mean bird's-eye distance of the matches, metres) at the operating score
  (score) and match_m (2 m, the devkit's true-positive threshold). ATE here is the plain mean of the matches at the
  operating point, not the devkit's recall-averaged figure.
* Regions: core.regions.default_grid(project) (8 sectors × 0–10, 10–20, 20–40, 40–60, 60–80, 80+ m about the middle of
  the vehicle's footprint); an object counts in its own region, a matched detection in its object's, a false
  detection in its own.

Uncertainty and comparisons
---------------------------
Percentile intervals of a block bootstrap over time (core.detection_eval.boot_weights: blocks of block_s seconds drawn
with replacement), stratified by scene: each scene's blocks are drawn within the scene. Recordings of the same scene
(the same scenario, seed, town and duration: core.detection_eval.Recording.same_scene) share the scene's draws, so a
rig compared with another on the same scenes is compared paired (the scene's difficulty cancels); scenes only one rig
drove are drawn independently, and the comparison says how many scenes were paired. Findings: every difference of
recall and of mean AP per class and region (and per class over every region) whose interval excludes zero, among
regions with at least min_gt object-instants of at least min_objects distinct objects (one object seen at 30
instants is still one object: a region it alone fills is no finding), ranked by size; the rest are "no difference
detectable with N objects". With many regions and classes some intervals exclude
zero by chance alone (about (1 − confidence) of the comparisons): the result says how many that is.

Records: to_records(result) gives one vantage.perception_record/1 per recording and detector (core.perf_model's
store), with point figures per class and region.
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

from . import bev_backends as BB
from . import detection as DT
from . import detection_eval as DE
from . import regions as RG

SCHEMA = 'vantage.rig_perception/1'
RECORD_SCHEMA = 'vantage.perception_record/1'
FRAME_V = 'vehicle (ISO 8855, rear axle)'
THRESHOLDS = (0.5, 1.0, 2.0, 4.0)
EVAL_CLASSES = BB.BEV_CLASSES + ('vehicle',)
GROUP = {'vehicle': ('car', 'truck', 'bus')}
CLASS_LABEL = dict(DT.CLASS_LABEL)
INSTANTS = ('all', 'keyframes')
VIEW_MODES = ('fisheye', 'pinhole')
CACHE_MODES = ('keep', 'read', 'off')
VIS_VISIBLE, VIS_MARGINAL, VIS_INVISIBLE, VIS_UNKNOWN = 0, 1, 2, 3
VIS_TEXT = {VIS_VISIBLE: 'visible', VIS_MARGINAL: 'marginal', VIS_INVISIBLE: 'invisible', VIS_UNKNOWN: 'unknown'}
MAX_RECORDINGS = 64
LEVEL_M = 2.0                          # an object whose box bottom is this far off the vehicle's ground: another level
DEFAULT_GT_RANGE_M = 50.0              # a nuScenes folder that does not say its annotation range (the devkit's car range)
MIN_RECALL, MIN_PRECISION = 0.1, 0.1   # the devkit's calc_ap clipping (detection_cvpr_2019)
# nuScenes categories -> Vantage's classes (the devkit's general_to_detection, with CARLA's emergency vehicles cars
# and its construction vehicles and trailers trucks; prams, wheelchairs and personal mobility left out as there)
CATEGORY_PREFIX = (('human.pedestrian.personal_mobility', None), ('human.pedestrian.stroller', None),
                   ('human.pedestrian.wheelchair', None), ('human.pedestrian', 'pedestrian'),
                   ('vehicle.bicycle', 'cyclist'), ('vehicle.motorcycle', 'motorcyclist'),
                   ('vehicle.bus', 'bus'), ('vehicle.car', 'car'), ('vehicle.emergency', 'car'),
                   ('vehicle.truck', 'truck'), ('vehicle.construction', 'truck'), ('vehicle.trailer', 'truck'))
ASSUMPTIONS = (
    'Ground truth is the simulator\'s annotation of every actor and parked car within its annotation range, '
    'interpolated between keyframes; objects no camera of a rig sees still count against it. Matching, AP and the '
    'distance thresholds are nuScenes\' (Caesar et al. 2020). The lifted detector is YOLOX (COCO-trained) on CARLA\'s '
    'renders, lifted to 3D by monocular geometry: absolute figures are that detector\'s on synthetic images; what '
    'carries over is the comparison of rigs on the same scenes. Intervals are percentile intervals of a block '
    'bootstrap over time, stratified by scene and paired between rigs on the same scenes; they cover the sampling of '
    'these drives\' moments, not other towns, weather or traffic.')
LIMITS_TEXT = BB.LIMITS_TEXT + ' ' + (
    'Regions beyond the annotated range are empty (no complete ground truth there). Few objects in a region give wide '
    'intervals: the yield table says where the statistics are thin. Between keyframes boxes are interpolated linearly '
    '(a turning vehicle\'s box cuts the corner by a few centimetres).')
REFERENCES = (
    'Caesar H. et al. (2020) nuScenes: A multimodal dataset for autonomous driving. CVPR (detection metrics: centre-'
    'distance matching at 0.5, 1, 2, 4 m, AP with min recall and precision 0.1; nuscenes-devkit eval/detection).',
    'Efron B., Tibshirani R. J. (1993) An Introduction to the Bootstrap. Chapman & Hall (percentile intervals).',
    'Künsch H. R. (1989) The jackknife and the bootstrap for general stationary observations. Annals of Statistics '
    '17(3) (block bootstrap).',
    'Ge Z. et al. (2021) YOLOX: Exceeding YOLO Series in 2021. arXiv:2107.08430.',
    'Kuhn H. W. (1955) The Hungarian method for the assignment problem. Naval Research Logistics Quarterly 2.',
    'Hartley R., Zisserman A. (2004) Multiple View Geometry in Computer Vision, 2nd ed., ch. 12 (triangulation).',
    'Harley A. W. et al. (2023) Simple-BEV: What Really Matters for Multi-Sensor BEV Perception? ICRA (the candidate '
    'learned backend: see bev_backends.TORCH_NOTE).')

LIMITS = {'score': (0.01, 0.99), 'match_m': (0.1, 10.0), 'frame_step': (1, 1000), 'max_range_m': (0.0, 500.0),
          'group_nms_m': (0.0, 5.0), 'pitch_sigma_deg': (0.01, 10.0), 'yaw_sigma_deg': (0.01, 10.0),
          'box_px_sigma': (0.1, 20.0), 'min_triangulation_deg': (0.5, 60.0), 'bootstrap': (0, 10000),
          'block_s': (0.0, 60.0), 'confidence': (0.5, 0.999), 'seed': (0, 2 ** 31 - 1), 'min_gt': (1, 10000), 'min_objects': (1, 1000),
          'view_fov_deg': (30.0, 150.0), 'view_pitch_deg': (-60.0, 89.0), 'view_width': (128, 2048),
          'view_aspect': (0.3, 1.5)}
INTS = ('frame_step', 'bootstrap', 'seed', 'min_gt', 'min_objects', 'view_width')


# =========================================================================== settings
@dataclass
class RigPerceptionSettings:
    recordings: list = field(default_factory=list)   # [{'path', 'rig'}]: the first rig is the reference
    backend: str = 'lifted'                          # 'lifted' | 'external:<name>' (a command on this computer)
    model: str = DT.DEFAULT_MODEL                    # the lifted backend's 2D detector
    views: str = 'fisheye'                           # fisheye images, or pinhole views cut from them
    view_fov_deg: float = 90.0
    view_pitch_deg: float = 10.0
    view_yaws: list = field(default_factory=lambda: [-60.0, 0.0, 60.0])
    view_width: int = 640
    view_aspect: float = 0.75
    score: float = 0.3                               # the operating point
    match_m: float = 2.0                             # the threshold recall, precision and ATE are quoted at
    instants: str = 'all'                            # all | keyframes
    frame_step: int = 1
    max_range_m: float = 0.0                         # 0 = the recordings' annotation range
    classes: list = field(default_factory=lambda: list(BB.BEV_CLASSES))
    group_nms_m: float = 1.5
    pitch_sigma_deg: float = BB.PITCH_SIGMA_DEG
    yaw_sigma_deg: float = BB.YAW_SIGMA_DEG
    box_px_sigma: float = BB.BOX_PX_SIGMA
    min_triangulation_deg: float = BB.MIN_TRIANGULATION_DEG
    fuse: bool = True
    bootstrap: int = 1000
    block_s: float = 1.0
    confidence: float = 0.95
    seed: int = 0
    min_gt: int = 10                                 # fewer object-instants in a region: shown, never a finding
    min_objects: int = 3                             # nor fewer distinct objects (one object seen 30 times is one)
    cache: str = 'keep'                              # 2D detections in the recording: keep (read + write) | read | off
    scene_set: dict = field(default_factory=dict)    # core.perception_scenes' settings (the page's drives)

    def to_dict(self):
        return asdict(self)

    def detector(self) -> DT.DetectorSettings:
        return DT.DetectorSettings.from_dict({'model': self.model})

    def lift(self) -> BB.LiftSettings:
        return BB.LiftSettings(pitch_sigma_deg=self.pitch_sigma_deg, yaw_sigma_deg=self.yaw_sigma_deg,
                               box_px_sigma=self.box_px_sigma, min_triangulation_deg=self.min_triangulation_deg,
                               fuse=self.fuse)

    def de_settings(self) -> DE.DetectionSettings:
        return DE.DetectionSettings.from_dict({'views': 'pinhole', 'view_fov_deg': self.view_fov_deg,
                                               'view_pitch_deg': self.view_pitch_deg, 'view_yaws': self.view_yaws,
                                               'view_width': self.view_width, 'view_aspect': self.view_aspect,
                                               'model': self.model})

    @staticmethod
    def from_dict(d) -> 'RigPerceptionSettings':
        s = RigPerceptionSettings()
        d = d if isinstance(d, dict) else {}
        recs = d.get('recordings')
        if isinstance(recs, (list, tuple)):
            out, seen = [], set()
            for r in recs:
                if isinstance(r, str):
                    r = {'path': r}
                if not isinstance(r, dict) or not isinstance(r.get('path'), str) or not r['path'].strip():
                    continue
                p = r['path'].strip()[:4096]
                if p in seen:
                    continue
                seen.add(p)
                rig = r.get('rig')
                out.append({'path': p, 'rig': rig.strip()[:120] if isinstance(rig, str) else ''})
            s.recordings = out[:MAX_RECORDINGS]
        b = d.get('backend')
        if isinstance(b, str) and (b == 'lifted' or (b.startswith('external:') and BB.clean_name(b[9:]) == b[9:]
                                                     and b[9:])):
            s.backend = b
        if d.get('model') in DT.MODELS:
            s.model = d['model']
        for k, opts in (('views', VIEW_MODES), ('instants', INSTANTS), ('cache', CACHE_MODES)):
            if d.get(k) in opts:
                setattr(s, k, d[k])
        v = d.get('fuse')
        if isinstance(v, bool):
            s.fuse = v
        for k, (lo, hi) in LIMITS.items():
            if k not in d or isinstance(d[k], bool):
                continue
            try:
                x = float(d[k])
            except (TypeError, ValueError):
                continue
            if math.isfinite(x):
                x = min(max(x, lo), hi)
                setattr(s, k, int(round(x)) if k in INTS else x)
        vy = d.get('view_yaws')
        if isinstance(vy, (list, tuple)):
            ys = []
            for x in vy[:8]:
                try:
                    f = float(x)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(f):
                    ys.append(min(max(f, -180.0), 180.0))
            s.view_yaws = ys or s.view_yaws
        cl = d.get('classes')
        if isinstance(cl, (list, tuple)):
            s.classes = [c for c in BB.BEV_CLASSES if c in cl]
        if isinstance(d.get('scene_set'), dict):
            s.scene_set = json.loads(json.dumps(d['scene_set'], default=str))
        return s


def settings_of(project) -> RigPerceptionSettings:
    a = getattr(project, 'analysis', None) if project is not None else None
    return RigPerceptionSettings.from_dict(getattr(a, 'rig_perception', None) if a is not None else None)


def eval_classes(st: RigPerceptionSettings) -> list:
    out = [c for c in BB.BEV_CLASSES if c in st.classes]
    if any(c in out for c in GROUP['vehicle']):
        out.append('vehicle')
    return out


def class_of_category(name: str) -> str | None:
    for pre, cls in CATEGORY_PREFIX:
        if str(name).startswith(pre):
            return cls
    return None


# =========================================================================== a recording
def _load_json(path):
    try:
        with open(path, encoding='utf-8') as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _pose(rot, trans) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = BB.quat_matrix(rot)
    T[:3, 3] = np.asarray(trans, float)[:3]
    return T


def _yaw_q(q) -> float:
    R = BB.quat_matrix(q)
    return math.atan2(R[1, 0], R[0, 0])


def nuscenes_root(path) -> tuple[str, str] | None:
    """(dataroot, version) of a recording folder (its nuscenes/ subfolder) or of a nuScenes folder itself."""
    for root in (os.path.join(path, 'nuscenes'), path):
        if not os.path.isdir(root):
            continue
        vs = sorted(d for d in os.listdir(root) if d.startswith('v1.0') and
                    os.path.isfile(os.path.join(root, d, 'sample_annotation.json')))
        if vs:
            return root, vs[0]
    return None


def rig_file(path) -> str | None:
    """The rig file a CARLA recording was made with: its own rig/sim_rig.json (a single run), else, for a drive of a
    core.simcampaign campaign (drives/<id>/ beside rigs/<rig id>/, as core.perception_scenes records them), the
    campaign's rig file its campaign_drive.json names. None when there is none (a nuScenes folder)."""
    own = os.path.join(path, 'rig', 'sim_rig.json')
    if os.path.isfile(own):
        return own
    doc = _load_json(os.path.join(path, 'campaign_drive.json'))
    rid = ((doc or {}).get('drive') or {}).get('rig') if isinstance(doc, dict) else None
    if isinstance(rid, str) and rid and os.sep not in rid and rid not in ('.', '..'):
        f = os.path.join(os.path.dirname(os.path.dirname(path)), 'rigs', rid, 'sim_rig.json')
        if os.path.isfile(f):
            return f
    return None


class PerceptionRecording:
    """A recording as the study reads it: its cameras (calibration, exact lens when the rig file has it), frames, ego
    poses, annotated objects and the cameras' per-frame ground truth (who sees what)."""

    def __init__(self, path, rig=''):
        path = os.path.abspath(os.path.expanduser(str(path)))
        if not os.path.isdir(path):
            raise ValueError(f'{path} is not a folder.')
        nr = nuscenes_root(path)
        if nr is None:
            raise ValueError(f'{os.path.basename(path)} has no nuScenes tables with annotations '
                             '(nuscenes/v1.0-*/sample_annotation.json).')
        self.path = path
        self.nus_root, self.nus_version = nr
        V = os.path.join(*nr)

        def table(n):
            t = _load_json(os.path.join(V, n + '.json'))
            return t if isinstance(t, list) else []
        sensors = {r['token']: r for r in table('sensor') if isinstance(r, dict) and r.get('token')}
        cals = {r['token']: r for r in table('calibrated_sensor') if isinstance(r, dict) and r.get('token')}
        self.ego = {r['token']: _pose(r['rotation'], r['translation']) for r in table('ego_pose')
                    if isinstance(r, dict) and r.get('token')}
        ego_ts = {r['token']: int(r['timestamp']) for r in table('ego_pose') if isinstance(r, dict)}
        order = sorted(self.ego, key=lambda k: ego_ts[k])
        self._ego_t = np.array([ego_ts[k] for k in order], np.int64)
        self._ego_T = [self.ego[k] for k in order]
        self.samples = sorted(((r['token'], int(r['timestamp'])) for r in table('sample') if isinstance(r, dict)),
                              key=lambda x: x[1])
        self.rig_doc = _load_json(rig_file(path) or '') or {}
        self.run_doc = _load_json(os.path.join(path, 'run.json')) or {}
        rig_cams = {c.get('channel'): c for c in self.rig_doc.get('cameras', []) if isinstance(c, dict)}
        # cameras and every sensor's frames
        self.cams = {}                                   # tag -> CameraCal
        self.channel_tag = {}
        self.frames = {}                                 # tag -> [(t, filename, ego token, is_key, sample token)]
        self.lidar_frames = []
        chan_of_cal = {}
        for c in cals.values():
            s = sensors.get(c.get('sensor_token')) or {}
            ch, mod = s.get('channel'), s.get('modality')
            if not ch:
                continue
            chan_of_cal[c['token']] = (ch, mod)
            if mod != 'camera' or not c.get('camera_intrinsic'):
                continue
            rc = rig_cams.get(ch) or {}
            tag = str(rc.get('tag') or (ch[4:] if ch.startswith('CAM_') else ch))
            K = np.asarray(c['camera_intrinsic'], float)
            vd = c.get('vantage_distortion') or {}
            im = rc.get('image') or {}
            if vd.get('image_size'):
                size = vd['image_size']
            elif im.get('width') and im.get('height'):
                size = [im['width'], im['height']]
            else:
                size = [int(round(2 * K[0, 2] + 1)), int(round(2 * K[1, 2] + 1))]
            cal = BB.CameraCal(tag, int(size[0]), int(size[1]), BB.quat_matrix(c.get('rotation', (1, 0, 0, 0))),
                               np.asarray(c.get('translation', (0, 0, 0)), float), 'pinhole', K, None, None,
                               math.radians(89.0), ch)
            model = str(vd.get('model', ''))
            if model in ('kannala_brandt', 'equidistant'):
                cal.model = 'kb'
                cal.D = np.asarray((list(vd.get('coefficients') or []) + [0.0] * 4)[:4], float)
                cal.max_theta = math.radians(89.0)
            lens = rc.get('lens') if isinstance(rc.get('lens'), dict) else None
            mt = (rc.get('image') or {}).get('max_theta_deg') or (lens or {}).get('max_theta_deg')
            if isinstance(mt, (int, float)) and math.isfinite(mt):
                cal.max_theta = math.radians(float(mt))
            if lens and lens.get('model') != 'pinhole' and all(k in lens for k in ('projection', 'f_px', 'cx', 'cy')):
                cal.model, cal.lens = 'exact', dict(lens)
            self.cams[tag] = cal
            self.channel_tag[ch] = tag
            self.frames[tag] = []
        lidars = sorted({ch for ch, mod in chan_of_cal.values() if mod == 'lidar'})
        self.lidar_channel = lidars[0] if lidars else ''
        for r in table('sample_data'):
            if not isinstance(r, dict):
                continue
            ch, _mod = chan_of_cal.get(r.get('calibrated_sensor_token'), (None, None))
            row = (int(r['timestamp']), str(r.get('filename', '')), r.get('ego_pose_token'),
                   bool(r.get('is_key_frame')), r.get('sample_token', ''))
            if ch in self.channel_tag:
                self.frames[self.channel_tag[ch]].append(row)
            elif ch == self.lidar_channel:
                self.lidar_frames.append(row)
        for k in self.frames:
            self.frames[k].sort()
        self.lidar_frames.sort()
        # annotated objects
        cat = {r['token']: r.get('name', '') for r in table('category') if isinstance(r, dict)}
        icat = {r['token']: cat.get(r.get('category_token'), '') for r in table('instance') if isinstance(r, dict)}
        attr = {r['token']: r.get('name', '') for r in table('attribute') if isinstance(r, dict)}
        st_of = {tok: t for tok, t in self.samples}
        tracks = {}
        for a in table('sample_annotation'):
            if not isinstance(a, dict) or a.get('sample_token') not in st_of:
                continue
            cls = class_of_category(icat.get(a.get('instance_token'), ''))
            if cls is None:
                continue
            if cls == 'cyclist' and any(attr.get(x) == 'cycle.without_rider' for x in a.get('attribute_tokens') or ()):
                continue                                 # a parked bicycle is no cyclist
            tr = tracks.setdefault(a['instance_token'], {'cls': cls, 'rows': []})
            tr['rows'].append((st_of[a['sample_token']], np.asarray(a['translation'], float),
                               np.asarray(a.get('size') or [0, 0, 0], float), _yaw_q(a.get('rotation') or [1, 0, 0, 0]),
                               str(a.get('visibility_token') or '')))
        self.tracks = {}
        for k, tr in tracks.items():
            rows = sorted(tr['rows'], key=lambda x: x[0])
            self.tracks[k] = {'cls': tr['cls'], 't': np.array([r[0] for r in rows], np.int64),
                              'p': np.array([r[1] for r in rows]), 'size': np.array([r[2] for r in rows]),
                              'yaw': np.unwrap(np.array([r[3] for r in rows])), 'vis': [r[4] for r in rows]}
        ts = np.array([t for _k, t in self.samples], np.int64)
        self.key_period_us = float(np.median(np.diff(ts))) if len(ts) > 1 else 5e5
        # the cameras' ground truth (who sees what) and the run's identity
        self.gt_index = None
        try:
            from . import groundtruth as GT
            self.gt_index = GT.read(path)
        except Exception:                                # noqa: BLE001 — a nuScenes folder without Vantage's labels
            self.gt_index = None
        self.cam_gt = {}                                 # (tag, t) -> frame dict of the ground-truth index
        if self.gt_index:
            for tag, c in (self.gt_index.get('cameras') or {}).items():
                for f in c.get('frames') or ():
                    if isinstance(f, dict) and 'timestamp' in f:
                        self.cam_gt[(str(tag), int(f['timestamp']))] = f
        out = self.rig_doc.get('output') or {}
        g = out.get('gt_range_m')
        self.gt_range_m = float(g) if isinstance(g, (int, float)) and g > 0 else (
            60.0 if self.rig_doc else DEFAULT_GT_RANGE_M)
        sc = self.run_doc.get('scene') or {}
        scen = self.run_doc.get('scenario')
        self.scenario = scen.get('name') if isinstance(scen, dict) else (scen if isinstance(scen, str) else '')
        self.scenario_params = (scen.get('params') if isinstance(scen, dict) else None) or {}
        self.variant = str(self.run_doc.get('variant') or self.rig_doc.get('variant') or '')
        self.variant_uid = str(self.rig_doc.get('variant_uid') or '')
        self.project = str(self.run_doc.get('project') or self.rig_doc.get('project') or '')
        self.seed = sc.get('seed') if isinstance(sc.get('seed'), int) else None
        self.town = str(sc.get('town') or self.run_doc.get('town') or '')
        w = sc.get('weather') or self.run_doc.get('weather')
        self.weather = str(w.get('preset') if isinstance(w, dict) else (w or ''))
        self.duration_s = float(sc['duration_s']) if isinstance(sc.get('duration_s'), (int, float)) else (
            (ts[-1] - ts[0]) / 1e6 + self.key_period_us / 1e6 if len(ts) > 1 else None)
        self.traffic = sc.get('traffic')
        self.walkers = sc.get('walkers', self.run_doc.get('walkers'))
        self.rig = rig or self.variant or os.path.basename(path)
        bits = [self.variant or os.path.basename(path)]
        if self.scenario:
            bits.append(self.scenario.replace('_', ' '))
        if self.seed is not None:
            bits.append(f'seed {self.seed}')
        self.label = ' · '.join(bits)
        self._de_cams = None

    # ---- identity
    def de_recording(self) -> DE.Recording:
        """The recording as core.detection_eval names it (its same_scene pairs scenes)."""
        return DE.Recording(self.path, self.label, self.gt_index or {}, {}, self.run_doc or None, self.project,
                            self.variant, self.scenario or '', self.seed, self.town, self.weather, self.duration_s)

    def scene_key(self) -> str:
        return json.dumps([self.scenario or '', self.seed, self.town, None if self.duration_s is None else
                           round(self.duration_s, 6)])

    def info(self) -> dict:
        return {'path': self.path, 'label': self.label, 'rig': self.rig, 'variant': self.variant,
                'scenario': self.scenario, 'seed': self.seed, 'town': self.town, 'weather': self.weather,
                'duration_s': self.duration_s, 'cameras': sorted(self.cams), 'samples': len(self.samples),
                'instances': len(self.tracks), 'annotation_range_m': self.gt_range_m,
                'camera_ground_truth': bool(self.cam_gt), 'lidar': self.lidar_channel}

    # ---- poses
    def ego_at(self, t) -> np.ndarray:
        """vehicle -> global at time t (µs): the nearest recorded pose (every sample_data has its own)."""
        if not len(self._ego_t):
            return np.eye(4)
        i = int(np.searchsorted(self._ego_t, t))
        if i >= len(self._ego_t) or (i > 0 and abs(self._ego_t[i - 1] - t) <= abs(self._ego_t[i] - t)):
            i -= 1
        return self._ego_T[max(i, 0)]

    def ego_of(self, token, t) -> np.ndarray:
        return self.ego.get(token) if token in self.ego else self.ego_at(t)

    def sample_poses(self) -> dict:
        return {tok: self.ego_at(t) for tok, t in self.samples}

    # ---- ground truth
    def gt_at(self, t, T_ref) -> dict:
        """The annotated objects at time t (µs) in the vehicle frame of T_ref (vehicle -> global): 'token', 'cls',
        'xy' (n × 2), 'z', 'size' (n × 3, l w h), 'yaw', 'ignore' (bool: annotated at only one keyframe around t)."""
        gap = 1.5 * self.key_period_us
        half = 0.5 * self.key_period_us + 1.0
        toks, cls, P, S, Y, ign = [], [], [], [], [], []
        for k, tr in self.tracks.items():
            ts = tr['t']
            j = int(np.searchsorted(ts, t))
            if j < len(ts) and ts[j] == t:
                p, s, y, bad = tr['p'][j], tr['size'][j], tr['yaw'][j], False
            elif 0 < j < len(ts) and ts[j] - ts[j - 1] <= gap:
                a = (t - ts[j - 1]) / (ts[j] - ts[j - 1])
                p = tr['p'][j - 1] * (1 - a) + tr['p'][j] * a
                s = tr['size'][j - 1] * (1 - a) + tr['size'][j] * a
                y, bad = tr['yaw'][j - 1] * (1 - a) + tr['yaw'][j] * a, False
            else:
                i = j - 1 if j > 0 and (j == len(ts) or t - ts[j - 1] < ts[j] - t) else min(j, len(ts) - 1)
                if i < 0 or abs(ts[i] - t) > half:
                    continue
                p, s, y, bad = tr['p'][i], tr['size'][i], tr['yaw'][i], True
            toks.append(k)
            cls.append(tr['cls'])
            P.append(p)
            S.append([s[1], s[0], s[2]])
            Y.append(y)
            ign.append(bad)
        n = len(toks)
        Ti = np.linalg.inv(T_ref)
        Pv = BB.apply_T(Ti, np.array(P).reshape(-1, 3)) if n else np.zeros((0, 3))
        yaw_ref = math.atan2(T_ref[1, 0], T_ref[0, 0])
        if n:                                            # another level (a bridge, a car park's deck): not here
            Sa = np.array(S).reshape(-1, 3)
            ign = list(np.asarray(ign, bool) | (np.abs(Pv[:, 2] - Sa[:, 2] / 2.0) > LEVEL_M))
        return {'token': toks, 'cls': cls, 'xy': Pv[:, :2], 'z': Pv[:, 2], 'size': np.array(S).reshape(-1, 3),
                'yaw': np.array(Y, float) - yaw_ref, 'ignore': np.array(ign, bool)}

    def camera_objects(self, tag, t, de_st) -> dict:
        """instance token -> 0 (shown, counted) or 1 (shown, an ignore region by detection_eval's rules) in that
        camera's frame at t (empty without camera ground truth)."""
        f = self.cam_gt.get((tag, int(t)))
        if f is None:
            return {}
        g = DE.gt_objects(f, de_st)
        out = {}
        for key, ig in zip(g['key'], g['ignore']):
            out[key] = min(out.get(key, 1), 0 if ig == DE.IGN_NONE else 1)
        return out

    # ---- frames
    def instants(self, mode='all', step=1) -> list:
        """[(token, t µs, keyframe)]: the reference lidar's sweeps (or the first camera's frames), or the keyframes."""
        if mode == 'keyframes' or not (self.lidar_frames or self.frames):
            out = [(tok, t, True) for tok, t in self.samples]
        elif self.lidar_frames:
            out = [(f'{self.lidar_channel}@{t}', t, k) for t, _f, _e, k, _s in self.lidar_frames]
            keys = {t: tok for tok, t in self.samples}
            out = [(keys.get(t, tok), t, k) for tok, t, k in out]
        else:
            tag = sorted(self.frames)[0]
            keys = {t: tok for tok, t in self.samples}
            out = [(keys.get(t, f'{tag}@{t}'), t, k) for t, _f, _e, k, _s in self.frames[tag]]
        return out[::max(int(step), 1)]

    def _period(self, tag) -> float:
        ts = np.array([r[0] for r in self.frames.get(tag, ())], np.int64)
        return float(np.median(np.diff(ts))) if len(ts) > 1 else 1e5

    def nearest_frame(self, tag, t):
        rows = self.frames.get(tag) or []
        if not rows:
            return None
        ts = [r[0] for r in rows]
        i = int(np.searchsorted(ts, t))
        cand = [j for j in (i - 1, i) if 0 <= j < len(rows)]
        j = min(cand, key=lambda j: abs(rows[j][0] - t))
        return rows[j] if abs(rows[j][0] - t) <= self._period(tag) / 2 + 1 else None

    def de_cameras(self) -> dict:
        """detection_eval's Camera objects (for its pinhole views), from the ground-truth index's recording."""
        if self._de_cams is None:
            self._de_cams = {}
            try:
                rec = DE.open_recording(self.path)
                self._de_cams = dict(rec.cameras)
            except ValueError:
                for tag, cal in self.cams.items():
                    if cal.K is None:
                        continue
                    self._de_cams[tag] = DE.Camera(tag, cal.channel, cal.width, cal.height, [], cal.K, cal.D,
                                                   'kb' if cal.D is not None else 'pinhole', cal.R, cal.t,
                                                   cal.max_theta)
        return self._de_cams

    def frames_doc(self) -> dict:
        """The keyframes as the external detector reads them (vantage.bev_frames/1)."""
        samples = []
        for tok, t in self.samples:
            T = self.ego_at(t)
            cams = []
            for tag, cal in sorted(self.cams.items()):
                fr = self.nearest_frame(tag, t)
                if fr is None:
                    continue
                Tc = self.ego_of(fr[2], fr[0])
                cams.append({'tag': tag, 'channel': cal.channel, 'file': os.path.join(self.nus_root, fr[1]),
                             'timestamp': fr[0], 'ego_pose': {'rotation': _quat_of(Tc[:3, :3]),
                                                              'translation': Tc[:3, 3].tolist()},
                             'calibration': cal.describe()})
            samples.append({'sample_token': tok, 'timestamp': t, 'ego_pose': {'rotation': _quat_of(T[:3, :3]),
                                                                            'translation': T[:3, 3].tolist()},
                            'cameras': cams})
        return {'schema': BB.SCHEMA_FRAMES, 'dataroot': self.nus_root, 'version': self.nus_version,
                'frame': 'nuScenes: global; calibration rotations optical -> vehicle (R) and the optical centre in '
                         'the vehicle frame (t)', 'samples': samples}


def _quat_of(R) -> list:
    from .mathutil import quat_from_matrix
    x, y, z, w = quat_from_matrix(np.asarray(R, float))
    return [float(w), float(x), float(y), float(z)]


def recording_info(path) -> dict:
    """What a folder holds, for the page's list: {'ok', 'text', ...info()}."""
    try:
        r = PerceptionRecording(path)
    except ValueError as e:
        return {'ok': False, 'path': str(path), 'text': str(e)}
    d = r.info()
    d['ok'] = True
    d['text'] = (f'{len(r.cams)} cameras, {len(r.samples)} keyframes, {len(r.tracks)} annotated objects' +
                 (f' · {r.town}' if r.town else '') + ('' if r.cam_gt else ' · no camera labels'))
    return d


# =========================================================================== frames of a recording
def build_frames(rec: PerceptionRecording, st: RigPerceptionSettings, mode='all') -> list:
    """The recording's Frame objects (images loaded lazily), in time order."""
    import cv2
    views = {}
    if st.views == 'pinhole':
        des = st.de_settings()
        for tag, c in rec.de_cameras().items():
            if tag in rec.cams:
                views[tag] = [(vw, 'view-' + hashlib.sha1(json.dumps(
                    [vw.yaw, des.view_fov_deg, des.view_pitch_deg, des.view_width, des.view_aspect]).encode()
                ).hexdigest()[:8]) for vw in DE.make_views(c, des)]
    out = []
    for tok, t, key in rec.instants(mode, st.frame_step):
        T_ref = rec.ego_at(t)
        Ti = np.linalg.inv(T_ref)
        items = []
        for tag, cal in sorted(rec.cams.items()):
            fr = rec.nearest_frame(tag, t)
            if fr is None:
                continue
            path = os.path.join(rec.nus_root, fr[1])
            T = Ti @ rec.ego_of(fr[2], fr[0])
            rel = os.path.relpath(path, rec.path).replace(os.sep, '/')
            memo = {}

            def load(path=path, memo=memo):
                if 'img' not in memo:
                    img = cv2.imread(path, cv2.IMREAD_COLOR)
                    if img is None:
                        raise OSError(f'cannot read {os.path.basename(path)}')
                    memo['img'] = img
                return memo['img']
            if st.views == 'pinhole' and tag in views:
                for vw, vkey in views[tag]:
                    vcal = BB.CameraCal(vw.name, vw.w, vw.h, cal.R @ vw.R, cal.t.copy(), 'pinhole', vw.K, None, None,
                                        math.radians(89.0), cal.channel, tag)
                    items.append(BB.FrameImage(vcal, fr[0], T, (lambda load=load, vw=vw: DE.remap(load(), vw)), path,
                                               None, f'{vkey}|{rel}'))
            else:
                items.append(BB.FrameImage(cal, fr[0], T, load, path, None, f'fisheye|{rel}'))
        lid = ''
        if rec.lidar_frames:
            lt = [r[0] for r in rec.lidar_frames]
            i = int(np.argmin(np.abs(np.asarray(lt) - t)))
            lid = os.path.join(rec.nus_root, rec.lidar_frames[i][1])
        out.append(BB.Frame(tok, int(t), items, lid, rec.path, bool(key), T_ref))
    return out


class _Caches:
    """core.detection_eval's detection caches of a recording (detections/<hash>-<CHANNEL>-<view>.npz): reused, and
    written back only when cache is 'keep'."""

    def __init__(self, rec: PerceptionRecording, det_key, mode):
        self.rec, self.det_key, self.mode = rec, det_key, mode
        self.c = {}
        self.hits = 0

    def _of(self, item):
        vkey, _rel = item.key.split('|', 1)
        ch = self.rec.cams[item.cal.source or item.cal.tag].channel
        k = (ch, vkey)
        if k not in self.c:
            self.c[k] = DE._Cache(self.rec.path, ch, vkey, self.det_key, self.mode != 'off')
        return self.c[k]

    def fill(self, item):
        if self.mode == 'off':
            return
        d = self._of(item).get(item.key.split('|', 1)[1], item.path)
        if d is not None:
            item.dets = d
            self.hits += 1

    def put(self, item):
        if self.mode == 'keep' and item.dets is not None:
            c = self._of(item)
            rel = item.key.split('|', 1)[1]
            if c.get(rel, item.path) is None:
                c.put(rel, item.path, item.dets)

    def save(self):
        if self.mode == 'keep':
            for c in self.c.values():
                c.save()


# =========================================================================== matching (nuScenes)
def match_centres(gt_xy, gt_ignore, det_xy, det_scores, thr):
    """nuScenes' greedy matching of one instant and class: detections in descending score order (ties: the later one
    first), each to the nearest untaken object, a match when its centre distance is < thr; a detection whose nearest
    object is an ignore region within thr is ignored (and takes it). Returns (status int8: 1 true, 0 false, -1
    ignored; det_gt: object index or -1; err: centre distance of a match, NaN otherwise)."""
    G = np.asarray(gt_xy, float).reshape(-1, 2)
    Dx = np.asarray(det_xy, float).reshape(-1, 2)
    sc = np.asarray(det_scores, float).reshape(-1)
    ig = np.asarray(gt_ignore, bool).reshape(-1)
    nd = len(Dx)
    status = np.zeros(nd, np.int8)
    det_gt = np.full(nd, -1, np.int64)
    err = np.full(nd, np.nan)
    if nd == 0 or len(G) == 0:
        return status, det_gt, err
    dist = np.hypot(Dx[:, None, 0] - G[None, :, 0], Dx[:, None, 1] - G[None, :, 1])
    taken = np.zeros(len(G), bool)
    order = np.lexsort((-np.arange(nd), -sc))
    for d in order:
        cand = ~taken & ~ig
        if cand.any():
            j = int(np.argmin(np.where(cand, dist[d], np.inf)))
            if dist[d, j] < thr:
                taken[j] = True
                status[d], det_gt[d], err[d] = 1, j, dist[d, j]
                continue
        cand = ~taken & ig
        if cand.any():
            j = int(np.argmin(np.where(cand, dist[d], np.inf)))
            if dist[d, j] < thr:
                taken[j] = True
                status[d], det_gt[d] = -1, j
    return status, det_gt, err


def ap_nuscenes(scores, tp, n_gt, min_recall=MIN_RECALL, min_precision=MIN_PRECISION) -> float | None:
    """The devkit's AP (accumulate + calc_ap) of one class's non-ignored detections: None without objects, 0 without a
    match. (Reference implementation: the vectorised one is checked against it.)"""
    if n_gt <= 0:
        return None
    s = np.asarray(scores, float)
    t = np.asarray(tp, float)
    if not len(t) or t.sum() == 0:
        return 0.0
    order = np.lexsort((-np.arange(len(s)), -s))
    t = t[order]
    ctp, cfp = np.cumsum(t), np.cumsum(1.0 - t)
    prec = ctp / (ctp + cfp)
    rec = ctp / float(n_gt)
    p = np.interp(np.linspace(0, 1, 101), rec, prec, right=0)
    p = p[int(round(100 * min_recall)) + 1:] - min_precision
    p[p < 0] = 0
    return min(float(np.mean(p)) / (1.0 - min_precision), 1.0)       # (1 + 2⁻⁵² from the division)


def ap_weighted(tp_sorted, wd, ngt_w, min_recall=MIN_RECALL, min_precision=MIN_PRECISION) -> np.ndarray:
    """ap_nuscenes for B resamples at once: tp_sorted (n,) in the devkit's order, wd (B, n) each detection's count in
    the resample (the bootstrap's block counts: whole numbers), ngt_w (B,) the resample's object count. Exactly the
    devkit's figure on each resample with its detections repeated as drawn (a detection drawn twice is two points of
    the precision-recall curve, as the devkit would see it): every row expanded into one flat array, each row's
    recall offset by twice its row index so one searchsorted serves np.interp for every row."""
    wd = np.atleast_2d(np.asarray(wd, float))
    B, n = wd.shape
    ngt_w = np.asarray(ngt_w, float).reshape(B)
    out = np.full(B, np.nan)
    ok = ngt_w > 0
    reps = np.rint(np.maximum(wd, 0)).astype(np.int64)
    m = reps.sum(1)
    out[ok & (m == 0)] = 0.0
    rows = np.flatnonzero(ok & (m > 0))
    if not len(rows):
        return out
    t = np.asarray(tp_sorted, float)
    R = reps[rows]
    k = len(rows)
    tot = R.sum(1)
    start = np.concatenate([[0], np.cumsum(tot)[:-1]])
    flat = np.repeat(np.tile(np.arange(n), k), R.ravel())
    row = np.repeat(np.arange(k), tot)
    tf = t[flat]
    ctp = np.cumsum(tf)
    cfp = np.cumsum(1.0 - tf)
    base_tp = np.concatenate([[0.0], ctp])[start]
    base_fp = np.concatenate([[0.0], cfp])[start]
    ctp = ctp - base_tp[row]
    cfp = cfp - base_fp[row]
    prec = ctp / (ctp + cfp)
    rec = ctp / ngt_w[rows][row]
    q = np.linspace(0, 1, 101)
    X = rec + 2.0 * row
    Q = (q[None] + 2.0 * np.arange(k)[:, None]).ravel()
    j = (np.searchsorted(X, Q, side='right') - 1).reshape(k, 101)
    first = start[:, None]
    last = (start + tot - 1)[:, None]
    x = np.broadcast_to(q, (k, 101))
    jc = np.clip(j, first, last)
    jn = np.clip(j + 1, first, last)
    x0, x1 = rec[jc], rec[jn]
    y0, y1 = prec[jc], prec[jn]
    with np.errstate(invalid='ignore', divide='ignore'):
        val = np.where(x1 > x0, y0 + (x - x0) * (y1 - y0) / np.where(x1 > x0, x1 - x0, 1.0), y0)
    val = np.where(j < first, prec[first], val)
    val = np.where(j >= last, np.where(x > rec[last], 0.0, prec[last]), val)
    p = np.maximum(val[:, int(round(100 * min_recall)) + 1:] - min_precision, 0.0)
    ap = np.minimum(p.mean(1) / (1.0 - min_precision), 1.0)
    anytp = ctp[start + tot - 1] > 0
    out[rows] = np.where(anytp, ap, 0.0)
    return out


def _group_nms(dets, m):
    """A vehicle group's detections: a vehicle within m metres of a higher-scoring one is the same vehicle."""
    if m <= 0 or len(dets) < 2:
        return dets
    dets = sorted(dets, key=lambda d: -d.score)
    keep = []
    for d in dets:
        if all(math.hypot(*(d.centre[:2] - k.centre[:2])) >= m for k in keep):
            keep.append(d)
    return keep


# =========================================================================== the result
@dataclass
class RigPerceptionResult:
    settings: RigPerceptionSettings
    grid: RG.RegionGrid
    rigs: list                           # [{'name', 'recordings': [index]}]
    recordings: list                     # [PerceptionRecording.info() + 'rig']
    backend: dict
    classes: list
    metrics: dict                        # rig -> class -> region id | 'all' -> figures (with intervals)
    comparisons: list                    # rig against the reference: class × region rows
    findings: list
    per_recording: list                  # [{class: {region id: point figures}}] per recording
    yields: dict                         # rig -> class -> region id -> objects (all instants)
    frames: dict                         # recording index -> [frame record for the page's frame view]
    timing: dict
    notes: list
    eval_range_m: float = 0.0
    seconds: float = 0.0
    variant: str = ''
    _samples: dict = field(default_factory=dict, repr=False)   # bootstrap resamples (not exported)

    def cell(self, rig, cls, rid='all'):
        return ((self.metrics.get(rig) or {}).get(cls) or {}).get(rid)


class _Cancelled(Exception):
    pass


def _rid_cells(grid):
    return [(s, r, grid.region_id(s, r)) for s in range(grid.sectors) for r in range(grid.n_ranges)]


# =========================================================================== run
def _backend_for(st, project, detector, qsettings, work_dir, cancel):
    if st.backend.startswith('external:'):
        if qsettings is None:
            try:
                from PyQt6.QtCore import QSettings
                qsettings = QSettings('Vantage', 'Vantage')
            except Exception as e:              # noqa: BLE001
                raise RuntimeError(f'No settings to read the external command from: {e}')
        from .. import workdir as WD
        wd = work_dir or (WD.project_dir(project, None, 'rig_perception', 'external') if project is not None else
                          os.path.join(os.path.expanduser('~'), '.cache', 'vantage-rig-perception'))
        return BB.external_from_settings(qsettings, st.backend[9:], wd, cancel)
    return BB.LiftedBackend(detector, st.detector(), st.lift(), score_floor=DT.DetectorSettings().keep_score)


def run(project=None, engine=None, settings: RigPerceptionSettings | None = None, variant=None, progress=None,
        cancel=None, backend: BB.Backend | None = None, detector=None, qsettings=None,
        work_dir=None) -> RigPerceptionResult | None:
    """Detect and evaluate every recording of the settings (None when cancelled). backend: a Backend instance (tests,
    a team's own in-process detector); detector: the lifted backend's 2D detector callable (tests); qsettings: where an
    external command is kept (default the user's)."""
    t0 = time.perf_counter()
    st = settings or settings_of(project)
    if not st.recordings:
        raise ValueError('Add a recording with ground truth (a CARLA run, or a nuScenes folder with annotations).')
    stop = cancel or (lambda: False)

    def prog(f, text=''):
        if progress is not None:
            try:
                progress(min(max(float(f), 0.0), 1.0), text)
            except TypeError:
                progress(min(max(float(f), 0.0), 1.0))
    recs = [PerceptionRecording(r['path'], r.get('rig') or '') for r in st.recordings]
    labels = [r.label for r in recs]
    for r in recs:
        if labels.count(r.label) > 1:
            r.label = f'{r.label} ({os.path.basename(r.path)})'
    grid = RG.default_grid(project) if project is not None else RG.RegionGrid()
    be = backend or _backend_for(st, project, detector, qsettings, work_dir, stop)
    if isinstance(be, BB.LiftedBackend) and be.detector is None:
        s = DT.status(st.model)
        if not s['verified']:
            raise FileNotFoundError(s['text'] + ' Download it on Simulate ▸ Detection quality.')
    classes = eval_classes(st)
    if not classes:
        raise ValueError('No class chosen.')
    notes = []
    gt_range = min(r.gt_range_m for r in recs)
    off = math.hypot(*grid.origin_xy)
    eval_range = (min(st.max_range_m, gt_range) if st.max_range_m > 0 else gt_range) - off
    if st.max_range_m > gt_range:
        notes.append(f'The evaluated range is capped at the recordings\' annotation range ({gt_range:g} m).')
    mode = 'keyframes' if (st.instants == 'keyframes' or getattr(be, 'frames', 'all') == 'keyframes') else 'all'
    if mode == 'keyframes' and st.instants != 'keyframes':
        notes.append(f'{be.name} gives detections at the keyframes only: every figure is over the keyframes.')
    de_st = DE.DetectionSettings()
    lift_be = isinstance(be, BB.LiftedBackend)
    det_key = ''
    if lift_be:
        info = be.detector_info()
        det_key = st.detector().key(info.get('sha256', info.get('model', '')))
    rig_names = list(dict.fromkeys(r.rig for r in recs))
    n_fr = sum(len(r.instants(mode, st.frame_step)) for r in recs) or 1
    done = 0
    G, D, frames_out = [], [], {}
    t_det = 0.0
    cache_hits = 0
    for ri, rec in enumerate(recs):
        if stop():
            return None
        frames = build_frames(rec, st, mode)          # images load lazily and go with their instant (memory)
        try:
            be.prepare(rec)
        except RuntimeError as e:
            if stop():
                return None
            raise RuntimeError(f'{rec.label}: {e}')
        caches = _Caches(rec, det_key, st.cache) if lift_be else None
        fo = frames_out.setdefault(ri, [])
        for fi, fr in enumerate(frames):
            if stop():
                return None
            if caches is not None:
                for item in fr.images:
                    caches.fill(item)
            t1 = time.perf_counter()
            try:
                dets = be.detect(fr)
            except OSError as e:
                notes.append(f'{rec.label}: {e}')
                dets = []
            t_det += time.perf_counter() - t1
            if caches is not None:
                for item in fr.images:
                    caches.put(item)
            gt = rec.gt_at(fr.timestamp, fr.ego_pose)
            vis = _visibility(rec, fr, gt, de_st)
            _evaluate(G, D, fo, ri, fi, fr, gt, vis, dets, grid, st, classes, eval_range)
            frames[fi] = None
            done += 1
            if done % 2 == 0 or done == n_fr:
                prog(0.9 * done / n_fr, f'{rec.label} · instant {fi + 1} of {len(frames)}')
        if caches is not None:
            cache_hits += caches.hits
            caches.save()
    prog(0.92, 'Statistics')
    res = _analyse(st, grid, recs, rig_names, classes, G, D, frames_out, be, notes, eval_range, stop)
    if res is None:
        return None
    res.timing = {'frames': n_fr, 'detector_s': round(t_det, 2), 'cache_hits': cache_hits,
                  'images_detected': getattr(be, 'images', None), 'wall_s': round(time.perf_counter() - t0, 2)}
    res.seconds = time.perf_counter() - t0
    res.variant = recs[0].variant
    prog(1.0, 'Done')
    return res


def _visibility(rec, fr, gt, de_st) -> np.ndarray:
    n = len(gt['token'])
    if not rec.cam_gt:
        return np.full(n, VIS_UNKNOWN, np.int8)
    best = {}
    seen_cams = set()
    for item in fr.images:
        tag = item.cal.source or item.cal.tag
        if tag in seen_cams:
            continue
        seen_cams.add(tag)
        for k, v in rec.camera_objects(tag, item.timestamp, de_st).items():
            best[k] = min(best.get(k, 9), v)
    return np.array([{0: VIS_VISIBLE, 1: VIS_MARGINAL}.get(best.get(k, 9), VIS_INVISIBLE) for k in gt['token']],
                    np.int8)


def _evaluate(G, D, fo, ri, fi, fr, gt, vis, dets, grid, st, classes, eval_range):
    """Match one instant per class at every threshold; append object rows to G and detection rows to D."""
    k_tp = THRESHOLDS.index(st.match_m) if st.match_m in THRESHOLDS else None
    thr_all = list(THRESHOLDS) + ([] if k_tp is not None else [st.match_m])
    kk = k_tp if k_tp is not None else len(THRESHOLDS)
    gxy = gt['xy']
    g_in = grid.range_m(gxy[:, 0], gxy[:, 1]) <= eval_range if len(gxy) else np.zeros(0, bool)
    gs, gr = grid.assign(gxy[:, 0], gxy[:, 1]) if len(gxy) else (np.zeros(0, int), np.zeros(0, int))
    dets = [d for d in dets if d.cls in BB.BEV_CLASSES]
    rec_gt, rec_det = [], []
    for ec in classes:
        members = GROUP.get(ec, (ec,))
        gi = np.flatnonzero(np.isin(np.asarray(gt['cls'], object), members) & g_in) if len(gxy) else np.zeros(0, int)
        dl = [d for d in dets if d.cls in members]
        if ec in GROUP:
            dl = _group_nms(dl, st.group_nms_m)
        dxy = np.array([d.centre[:2] for d in dl]).reshape(-1, 2)
        dsc = np.array([d.score for d in dl], float)
        d_in = grid.range_m(dxy[:, 0], dxy[:, 1]) <= eval_range if len(dl) else np.zeros(0, bool)
        dxy, dsc = dxy[d_in], dsc[d_in]
        dl = [d for d, k in zip(dl, d_in) if k]
        if not len(gi) and not len(dl):
            continue
        e = EVAL_CLASSES.index(ec)
        ds_, dr_ = grid.assign(dxy[:, 0], dxy[:, 1]) if len(dl) else (np.zeros(0, int), np.zeros(0, int))
        stat = np.zeros((len(dl), len(thr_all)), np.int8)
        dS = np.zeros((len(dl), len(thr_all)), np.int16)
        dR = np.zeros((len(dl), len(thr_all)), np.int16)
        msc = np.full((len(gi), len(thr_all)), -np.inf)
        g_err = np.full(len(gi), np.nan)
        d_err = np.full(len(dl), np.nan)
        for k, thr in enumerate(thr_all):
            s_, dg, er = match_centres(gxy[gi], gt['ignore'][gi], dxy, dsc, thr)
            stat[:, k] = s_
            for j in range(len(dl)):
                if dg[j] >= 0:
                    dS[j, k], dR[j, k] = gs[gi[dg[j]]], gr[gi[dg[j]]]
                    if s_[j] == 1:
                        msc[dg[j], k] = dsc[j]
                        if k == kk:
                            g_err[dg[j]] = er[j]
                            d_err[j] = er[j]
                else:
                    dS[j, k], dR[j, k] = ds_[j], dr_[j]
        for a, i in enumerate(gi):
            G.append((ri, fi, fr.timestamp, e, int(gs[i]), int(gr[i]), int(vis[i]) if len(vis) else VIS_UNKNOWN,
                      bool(gt['ignore'][i]), msc[a], g_err[a], f'{ri}:{gt["token"][i]}'))
        for j in range(len(dl)):
            D.append((ri, fi, fr.timestamp, e, float(dsc[j]), stat[j], dS[j], dR[j], d_err[j]))
        if ec not in GROUP:
            for a, i in enumerate(gi):
                rec_gt.append((float(gxy[i, 0]), float(gxy[i, 1]), e, int(vis[i]) if len(vis) else VIS_UNKNOWN,
                               float(msc[a, kk]) if np.isfinite(msc[a, kk]) else -1.0, bool(gt['ignore'][i])))
            for j, d in enumerate(dl):
                rec_det.append((float(dxy[j, 0]), float(dxy[j, 1]), e, float(dsc[j]), int(stat[j, kk]),
                                d.info.get('fusion', ''), '+'.join(d.sources)))
    fo.append({'t': int(fr.timestamp), 'token': fr.token, 'gt': rec_gt, 'det': rec_det})


# =========================================================================== statistics
def _scene_strata(recs) -> tuple[list, list]:
    """Each recording's scene index (recordings of one scene, by detection_eval's same_scene, share one) and each
    scene's start time (µs)."""
    scenes, starts, of = [], [], []
    for r in recs:
        dr = r.de_recording()
        k = next((i for i, (s0, _ri) in enumerate(scenes) if s0.same_scene(dr)), None)
        if k is None:
            scenes.append((dr, len(of)))
            starts.append(None)
            k = len(scenes) - 1
        of.append(k)
    return of, starts


def _analyse(st, grid, recs, rig_names, classes, G, D, frames_out, be, notes, eval_range, stop):
    conf = st.confidence
    nthr = len(THRESHOLDS) + (0 if st.match_m in THRESHOLDS else 1)
    kk = THRESHOLDS.index(st.match_m) if st.match_m in THRESHOLDS else len(THRESHOLDS)
    g = {'rec': np.array([x[0] for x in G], np.int64), 't': np.array([x[2] for x in G], np.int64),
         'e': np.array([x[3] for x in G], np.int64), 's': np.array([x[4] for x in G], np.int64),
         'r': np.array([x[5] for x in G], np.int64), 'vis': np.array([x[6] for x in G], np.int64),
         'ign': np.array([x[7] for x in G], bool), 'msc': np.array([x[8] for x in G]).reshape(-1, nthr),
         'err': np.array([x[9] for x in G], float), 'obj': np.array([x[10] for x in G], object)}
    d = {'rec': np.array([x[0] for x in D], np.int64), 't': np.array([x[2] for x in D], np.int64),
         'e': np.array([x[3] for x in D], np.int64), 'score': np.array([x[4] for x in D], float),
         'stat': np.array([x[5] for x in D], np.int8).reshape(-1, nthr),
         's': np.array([x[6] for x in D], np.int64).reshape(-1, nthr),
         'r': np.array([x[7] for x in D], np.int64).reshape(-1, nthr), 'err': np.array([x[8] for x in D], float)}
    # time blocks, stratified by scene (recordings of one scene share its blocks)
    scene_of, _starts = _scene_strata(recs)
    nsc = max(scene_of) + 1 if scene_of else 0
    t0s = {}
    for ri, fr in frames_out.items():
        if fr:
            k = scene_of[ri]
            t0s[k] = min(t0s.get(k, fr[0]['t']), min(f['t'] for f in fr))
    tmax = {}
    for ri, fr in frames_out.items():
        if fr:
            k = scene_of[ri]
            tmax[k] = max(tmax.get(k, 0), max(f['t'] for f in fr))
    block_us = st.block_s * 1e6
    nb_sc, offs = [], []
    o = 0
    for k in range(nsc):
        span = (tmax.get(k, 0) - t0s.get(k, 0))
        nb = int(span // block_us) + 1 if block_us > 0 else max(1, len({f['t'] for ri in frames_out for f in
                                                                         frames_out[ri] if scene_of[ri] == k}))
        nb_sc.append(nb)
        offs.append(o)
        o += nb
    nblocks = o
    tlists = {k: sorted({f['t'] for ri in frames_out for f in frames_out[ri] if scene_of[ri] == k}) for k in range(nsc)}

    def blocks_of(rec_idx, t):
        out = np.zeros(len(t), np.int64)
        for i, (ri, tt) in enumerate(zip(rec_idx, t)):
            k = scene_of[int(ri)]
            if block_us > 0:
                b = int((tt - t0s.get(k, tt)) // block_us)
            else:
                b = int(np.searchsorted(tlists[k], tt))
            out[i] = offs[k] + min(b, nb_sc[k] - 1)
        return out
    g_blk = blocks_of(g['rec'], g['t'])
    d_blk = blocks_of(d['rec'], d['t'])
    B = int(st.bootstrap)
    if B > 0 and nblocks:
        W = np.concatenate([DE.boot_weights(nb_sc[k], B, st.seed + 7919 * k) for k in range(nsc)], axis=1)
    else:
        W = np.zeros((0, nblocks))
    rig_of_rec = np.array([rig_names.index(r.rig) for r in recs], np.int64)
    g_rig = rig_of_rec[g['rec']] if len(G) else np.zeros(0, np.int64)
    d_rig = rig_of_rec[d['rec']] if len(D) else np.zeros(0, np.int64)
    g_keep = ~g['ign'] if len(G) else np.zeros(0, bool)
    g_hit = g['msc'][:, kk] >= st.score if len(G) else np.zeros(0, bool)
    d_op = d['score'] >= st.score if len(D) else np.zeros(0, bool)
    regions = _rid_cells(grid)

    def cell(gsel, dsel_k, ap=True):
        """Figures of objects gsel; dsel_k(k) -> the detections of the cell at threshold k."""
        gs = gsel & g_keep
        ngt = int(gs.sum())
        dk = dsel_k(kk) & (d['stat'][:, kk] >= 0) if len(D) else np.zeros(0, bool)
        do = dk & d_op
        tp = int(np.count_nonzero(gs & g_hit))
        tp_det = int(np.count_nonzero(do & (d['stat'][:, kk] == 1))) if len(D) else 0
        fp = int(np.count_nonzero(do)) - tp_det
        nvis = int(np.count_nonzero(gs & (g['vis'] == VIS_VISIBLE))) if len(G) else 0
        out = {'n_gt': ngt, 'n_objects': len(set(g['obj'][gs].tolist())) if ngt else 0, 'n_visible': nvis, 'tp': tp,
               'fn': ngt - tp, 'tp_det': tp_det, 'fp': fp}
        den_g = DE._per_block(g_blk[gs], np.ones(ngt), nblocks)
        rc, rcb = DE._ratio(DE._per_block(g_blk[gs], g_hit[gs], nblocks), den_g, W)
        out['recall'], out['recall_ci'] = rc, (DE._pct(rcb, conf) if rc is not None else None)
        vs = gs & (g['vis'] == VIS_VISIBLE) if len(G) else gs
        rv, _ = DE._ratio(DE._per_block(g_blk[vs], g_hit[vs], nblocks), DE._per_block(g_blk[vs], np.ones(int(vs.sum())),
                                                                                     nblocks), np.zeros((0, nblocks)))
        out['recall_visible'] = rv
        out['visible_share'] = (nvis / ngt) if ngt else None
        pr, prb = DE._ratio(DE._per_block(d_blk[do], d['stat'][do, kk] == 1, nblocks),
                            DE._per_block(d_blk[do], np.ones(int(do.sum())), nblocks), W)
        out['precision'], out['precision_ci'] = pr, (DE._pct(prb, conf) if pr is not None else None)
        mt = do & (d['stat'][:, kk] == 1) if len(D) else do
        at, atb = DE._ratio(DE._per_block(d_blk[mt], d['err'][mt], nblocks),
                            DE._per_block(d_blk[mt], np.ones(int(mt.sum())), nblocks), W)
        out['ate_m'], out['ate_ci'] = at, (DE._pct(atb, conf) if at is not None else None)
        out['_rcb'], out['_prb'] = rcb, prb
        if ap:
            aps, apb = [], []
            ngt_b = den_g
            for k in range(len(THRESHOLDS)):
                dv = dsel_k(k) & (d['stat'][:, k] >= 0) if len(D) else np.zeros(0, bool)
                idx = np.flatnonzero(dv)
                idx = idx[np.lexsort((-idx, -d['score'][idx]))] if len(idx) else idx
                tps = (d['stat'][idx, k] == 1).astype(float)
                aps.append(ap_nuscenes(d['score'][idx], tps, ngt) if ngt else None)
                if len(W) and ngt:
                    apb.append(ap_weighted(tps, W[:, d_blk[idx]] if len(idx) else np.zeros((len(W), 0)), W @ ngt_b))
            out['ap_by_threshold'] = {f'{t:g}': a for t, a in zip(THRESHOLDS, aps)}
            out['ap'] = float(np.mean(aps)) if ngt else None
            if apb:
                m = np.mean(np.vstack(apb), axis=0)
                out['ap_ci'], out['_apb'] = DE._pct(m, conf), m
            else:
                out['ap_ci'], out['_apb'] = None, np.zeros(0)
        return out

    metrics, samples, yields = {}, {}, {}
    for bi, rig in enumerate(rig_names):
        if stop():
            return None
        metrics[rig], yields[rig] = {}, {}
        for ec in classes:
            e = EVAL_CLASSES.index(ec)
            gk = (g_rig == bi) & (g['e'] == e) if len(G) else np.zeros(0, bool)
            dk = (d_rig == bi) & (d['e'] == e) if len(D) else np.zeros(0, bool)
            cells = {}
            ys = {}
            c = cell(gk, lambda k: dk)
            samples[(rig, ec, 'all')] = {k: c.pop(k) for k in ('_rcb', '_prb', '_apb')}
            cells['all'] = c
            for s, r, rid in regions:
                gsel = gk & (g['s'] == s) & (g['r'] == r) if len(G) else gk
                dany = bool((dk & (d['s'][:, kk] == s) & (d['r'][:, kk] == r)).any()) if len(D) else False
                if not (gsel.any() or dany):
                    continue
                c = cell(gsel, lambda k, s=s, r=r: dk & (d['s'][:, k] == s) & (d['r'][:, k] == r))
                samples[(rig, ec, rid)] = {k: c.pop(k) for k in ('_rcb', '_prb', '_apb')}
                cells[rid] = c
                ys[rid] = c['n_gt']
            metrics[rig][ec] = cells
            yields[rig][ec] = ys
    # comparisons with the reference rig
    comps, findings = [], []
    ref = rig_names[0]
    ref_scenes = {scene_of[i] for i, r in enumerate(recs) if r.rig == ref}
    for rig in rig_names[1:]:
        sc_b = {scene_of[i] for i, r in enumerate(recs) if r.rig == rig}
        shared = len(sc_b & ref_scenes)
        paired = shared == len(sc_b) == len(ref_scenes) and shared > 0
        how = ('paired over the same scenes' if paired else
               f'paired over {shared} of {len(sc_b | ref_scenes)} scenes' if shared else 'not paired: different scenes')
        if not paired:
            notes.append(f'{rig} and {ref}: {how}.' + ('' if shared else ' Their intervals are drawn independently.'))
        for ec in classes:
            for rid in ['all'] + [x[2] for x in regions]:
                a, b = metrics[ref].get(ec, {}).get(rid), metrics[rig].get(ec, {}).get(rid)
                if a is None or b is None:
                    continue
                sa, sb = samples[(ref, ec, rid)], samples[(rig, ec, rid)]
                row = {'reference': ref, 'rig': rig, 'class': ec, 'region': rid,
                       'region_name': 'every region' if rid == 'all' else grid.region_name(*grid.parse_id(rid)),
                       'paired': paired, 'pairing': how, 'n_gt_reference': a['n_gt'], 'n_gt': b['n_gt'],
                       'n_objects': min(a['n_objects'], b['n_objects'])}
                for key, kb in (('recall', '_rcb'), ('precision', '_prb'), ('ap', '_apb')):
                    va, vb = a.get(key), b.get(key)
                    row[f'{key}_reference'], row[key] = va, vb
                    if va is None or vb is None:
                        row[f'{key}_difference'] = row[f'{key}_difference_ci'] = None
                        continue
                    row[f'{key}_difference'] = vb - va
                    xa, xb = sa.get(kb, np.zeros(0)), sb.get(kb, np.zeros(0))
                    row[f'{key}_difference_ci'] = DE._pct(xb - xa, conf) if len(xa) and len(xa) == len(xb) else None
                va, vb = a.get('ate_m'), b.get('ate_m')
                row['ate_reference'], row['ate_m'] = va, vb
                row['ate_difference'] = (vb - va) if va is not None and vb is not None else None
                comps.append(row)
    findings = _findings(comps, st, conf)
    return RigPerceptionResult(st, grid, [{'name': n, 'recordings': [i for i, r in enumerate(recs) if r.rig == n]}
                                          for n in rig_names],
                               [dict(r.info(), index=i) for i, r in enumerate(recs)], be.describe(), list(classes),
                               metrics, comps, findings,
                               _per_recording(recs, classes, regions, g, d, g_keep, g_hit, d_op, kk),
                               yields, frames_out, {}, notes, eval_range, variant=recs[0].variant if recs else '',
                               _samples=samples)


def _per_recording(recs, classes, regions, g, d, g_keep, g_hit, d_op, kk) -> list:
    """Point figures per recording, class and region (the records' metrics: no intervals)."""
    out = []
    for ri in range(len(recs)):
        m = {}
        for ec in classes:
            e = EVAL_CLASSES.index(ec)
            gk = (g['rec'] == ri) & (g['e'] == e) & g_keep if len(g['rec']) else np.zeros(0, bool)
            dk = (d['rec'] == ri) & (d['e'] == e) if len(d['rec']) else np.zeros(0, bool)
            cells = {}
            for s, r, rid in regions:
                gs = gk & (g['s'] == s) & (g['r'] == r) if len(gk) else gk
                dm = (lambda k, s=s, r=r: dk & (d['s'][:, k] == s) & (d['r'][:, k] == r)) if len(dk) else \
                    (lambda k: dk)
                if not gs.any() and not (dm(kk).any() if len(dk) else False):
                    continue
                ngt = int(gs.sum())
                tp = int(np.count_nonzero(gs & g_hit))
                do = dm(kk) & (d['stat'][:, kk] >= 0) & d_op if len(dk) else dk
                td = int(np.count_nonzero(do & (d['stat'][:, kk] == 1))) if len(dk) else 0
                fp = int(np.count_nonzero(do)) - td
                aps = []
                for k in range(len(THRESHOLDS)):
                    dv = dm(k) & (d['stat'][:, k] >= 0) if len(dk) else dk
                    idx = np.flatnonzero(dv)
                    aps.append(ap_nuscenes(d['score'][idx], (d['stat'][idx, k] == 1).astype(float), ngt))
                mt = do & (d['stat'][:, kk] == 1) if len(dk) else dk
                cells[rid] = {'n_gt': ngt, 'tp': tp, 'fp': fp, 'fn': ngt - tp,
                              'recall': (tp / ngt) if ngt else None,
                              'precision': (td / (td + fp)) if (td + fp) else None,
                              'ap': float(np.mean(aps)) if ngt else None,
                              'ate_m': float(np.mean(d['err'][mt])) if mt.any() else None}
            m[ec] = cells
        out.append(m)
    return out


def _pt(v) -> str:
    return f'{abs(v) * 100:.1f}'


def _findings(comps, st, conf) -> list:
    """Significant differences (interval excluding zero) as claims, ranked by size; then the rest per class."""
    out = []
    tested = 0
    for c in comps:
        for key, word in (('recall', 'recall'), ('ap', 'mean AP')):
            dv, ci = c.get(f'{key}_difference'), c.get(f'{key}_difference_ci')
            n = min(c['n_gt'], c['n_gt_reference'])
            if dv is None or ci is None or n < st.min_gt or c.get('n_objects', st.min_objects) < st.min_objects:
                continue
            tested += 1
            sig = ci[0] > 0 or ci[1] < 0
            where = 'over every region' if c['region'] == 'all' else f'at {c["region_name"]}'
            cl = CLASS_LABEL.get(c['class'], c['class']).lower()
            if sig:
                lo, hi = (ci[0], ci[1]) if dv > 0 else (-ci[1], -ci[0])
                text = (f'{c["rig"]} {"gains" if dv > 0 else "loses"} {_pt(dv)} points of {cl} {word} {where} '
                        f'({conf * 100:g} % CI {lo * 100:.1f}–{hi * 100:.1f}, {c["pairing"]}).')
            else:
                text = f'{c["rig"]} against {c["reference"]}, {cl} {word} {where}: no difference detectable with ' \
                       f'{n} object-instants of {c.get("n_objects", "?")} objects ({conf * 100:g} % CI {ci[0] * 100:+.1f} to {ci[1] * 100:+.1f} points).'
            out.append({'rig': c['rig'], 'reference': c['reference'], 'class': c['class'], 'region': c['region'],
                        'region_name': c['region_name'], 'metric': key, 'difference': dv, 'ci': ci,
                        'n': n, 'significant': sig, 'paired': c['paired'], 'text': text})
    out.sort(key=lambda f: (not f['significant'], -abs(f['difference'])))
    if tested:
        k = sum(f['significant'] for f in out)
        out.append({'rig': '', 'reference': '', 'class': '', 'region': '', 'region_name': '', 'metric': 'note',
                    'difference': None, 'ci': None, 'n': tested, 'significant': False, 'paired': None,
                    'text': f'{k} of {tested} comparisons with at least {st.min_gt} object-instants of '
                            f'{st.min_objects} or more objects exclude zero; about '
                            f'{(1 - conf) * tested:.1f} would by chance alone at {conf * 100:g} % (no correction for '
                            'multiple comparisons).'})
    return out


def evaluate_frames(items, grid=None, settings: RigPerceptionSettings | None = None) -> dict:
    """Point figures (class -> region id | 'all' -> {n_gt, tp, fp, fn, recall, precision, ap, ate_m}) of instants
    given directly: items = [(gt, dets)], gt = {'xy' (n × 2, vehicle frame), 'cls' [names], 'ignore' (optional)},
    dets = [Detection3D]. No bootstrap (core.robust_perception and tests)."""
    st = settings or RigPerceptionSettings(bootstrap=0)
    grid = grid or RG.RegionGrid()
    classes = eval_classes(st)
    G, D, fo = [], [], []
    for fi, (gt, dets) in enumerate(items):
        n = len(gt['cls'])
        g = {'token': [str(i) for i in range(n)], 'cls': list(gt['cls']), 'xy': np.asarray(gt['xy'], float).reshape(-1, 2),
             'ignore': np.asarray(gt.get('ignore', np.zeros(n, bool)), bool)}
        fr = BB.Frame(str(fi), fi * 100000, [])
        _evaluate(G, D, fo, 0, fi, fr, g, np.full(n, VIS_UNKNOWN, np.int8), dets, grid, st, classes, math.inf)
    nthr = len(THRESHOLDS) + (0 if st.match_m in THRESHOLDS else 1)
    kk = THRESHOLDS.index(st.match_m) if st.match_m in THRESHOLDS else len(THRESHOLDS)
    g = {'rec': np.array([x[0] for x in G], np.int64), 'e': np.array([x[3] for x in G], np.int64),
         's': np.array([x[4] for x in G], np.int64), 'r': np.array([x[5] for x in G], np.int64),
         'msc': np.array([x[8] for x in G]).reshape(-1, nthr), 'ign': np.array([x[7] for x in G], bool)}
    d = {'rec': np.array([x[0] for x in D], np.int64), 'e': np.array([x[3] for x in D], np.int64),
         'score': np.array([x[4] for x in D], float), 'stat': np.array([x[5] for x in D], np.int8).reshape(-1, nthr),
         's': np.array([x[6] for x in D], np.int64).reshape(-1, nthr),
         'r': np.array([x[7] for x in D], np.int64).reshape(-1, nthr), 'err': np.array([x[8] for x in D], float)}
    keep = ~g['ign']
    hit = g['msc'][:, kk] >= st.score if len(G) else np.zeros(0, bool)
    dop = d['score'] >= st.score if len(D) else np.zeros(0, bool)
    m = _per_recording([None], classes, _rid_cells(grid), g, d, keep, hit, dop, kk)[0]
    for ec in classes:
        e = EVAL_CLASSES.index(ec)
        gk = (g['e'] == e) & keep if len(G) else np.zeros(0, bool)
        dk = (d['e'] == e) if len(D) else np.zeros(0, bool)
        ngt = int(gk.sum())
        tp = int(np.count_nonzero(gk & hit))
        do = dk & (d['stat'][:, kk] >= 0) & dop if len(D) else dk
        td = int(np.count_nonzero(do & (d['stat'][:, kk] == 1))) if len(D) else 0
        fp = int(np.count_nonzero(do)) - td
        aps = [ap_nuscenes(d['score'][dk & (d['stat'][:, k] >= 0)], (d['stat'][dk & (d['stat'][:, k] >= 0), k] == 1)
                           .astype(float), ngt) for k in range(len(THRESHOLDS))] if len(D) else \
            [0.0 if ngt else None] * len(THRESHOLDS)
        mt = do & (d['stat'][:, kk] == 1) if len(D) else dk
        m.setdefault(ec, {})['all'] = {'n_gt': ngt, 'tp': tp, 'fp': fp, 'fn': ngt - tp,
                                       'recall': tp / ngt if ngt else None,
                                       'precision': td / (td + fp) if td + fp else None,
                                       'ap': float(np.mean(aps)) if ngt else None,
                                       'ate_m': float(np.mean(d['err'][mt])) if len(D) and mt.any() else None}
    return m


# =========================================================================== records
def _rig_block_of(rec_info: dict, rec: PerceptionRecording, project=None) -> dict:
    """The recording's rig as a record's rig block: the project's variant when the rig file names one this project
    has (core.perf_model.rig_block), else the rig file's own cameras and sensors."""
    v = None
    if project is not None:
        v = project.variant_by_uid(rec.variant_uid) if rec.variant_uid else None
        if v is None and rec.variant:
            v = next((x for x in project.variants if x.name == rec.variant), None)
    if v is not None:
        try:
            from . import perf_model as PM
            sizes = {t: (c.width, c.height) for t, c in rec.cams.items()}
            return PM.rig_block(project, v, image_size=sizes)
        except ImportError:
            pass
    from .mathutil import euler_zyx
    cams = []
    by_tag = {c.get('tag'): c for c in rec.rig_doc.get('cameras', []) if isinstance(c, dict)}
    for tag, cal in sorted(rec.cams.items()):
        rc = by_tag.get(tag) or {}
        pv = rc.get('pose_vehicle') or {}
        if pv:
            x, y, z, yaw, pitch, roll = (float(pv.get(k, 0.0)) for k in ('x', 'y', 'z', 'yaw', 'pitch', 'roll'))
        else:
            from .mathutil import BODY_TO_OPTICAL
            Rb = cal.R @ BODY_TO_OPTICAL                       # body -> vehicle
            yaw, pitch, roll = euler_zyx(Rb)
            x, y, z = (float(q) for q in cal.t)
        hf, vf = cal.hfov_vfov_deg()
        proj = (cal.lens or {}).get('projection') or ('kannala_brandt' if cal.model == 'kb' else 'rectilinear')
        cams.append({'tag': tag, 'module': '', 'lens': '', 'width': int(cal.width), 'height': int(cal.height),
                     'x': round(x, 4), 'y': round(y, 4), 'z': round(z, 4), 'yaw': round(yaw, 3),
                     'pitch': round(pitch, 3), 'roll': round(roll, 3), 'hfov_deg': round(hf, 2),
                     'vfov_deg': round(vf, 2), 'max_theta_deg': round(math.degrees(cal.max_theta), 2),
                     'projection': proj})
    sens = []
    for kind, key in (('lidar', 'lidars'), ('radar', 'radars')):
        for s in rec.rig_doc.get(key, []) or []:
            if not isinstance(s, dict):
                continue
            pv = s.get('pose_vehicle') or {}
            sens.append({'tag': str(s.get('tag', '')), 'kind': kind, 'model': str(s.get('model') or s.get('model_id')
                                                                                 or ''),
                         **{k: round(float(pv.get(k, 0.0)), 4 if k in 'xyz' else 3)
                            for k in ('x', 'y', 'z', 'yaw', 'pitch', 'roll')}})
    veh = (rec.rig_doc.get('vehicle') or {})
    pj = veh.get('project') or (rec.run_doc.get('vehicle') or {}).get('project') or {}
    return {'vehicle': {'name': str(veh.get('name') or ''), 'length_m': pj.get('length'), 'width_m': pj.get('width'),
                        'height_m': pj.get('height')}, 'cameras': cams, 'sensors': sens}


def _scene_block(rec: PerceptionRecording, frames: int) -> dict:
    sc = {'kind': 'carla' if rec.rig_doc else 'nuscenes', 'town': rec.town, 'scenario': rec.scenario or '',
          'seed': rec.seed, 'weather': rec.weather, 'duration_s': rec.duration_s, 'frames': int(frames)}
    if rec.traffic is not None:
        sc['traffic'] = rec.traffic
    if rec.walkers is not None:
        sc['walkers'] = rec.walkers
    return sc


def record_source(path, scene_kind='carla') -> str:
    """A record's source: 'redrive' for a re-shot drive (its dataset.json says generator 'redrive', core.redrive),
    else 'carla' for a CARLA run, else 'redrive' (perf_model.SOURCES has no other kind for a nuScenes folder)."""
    ds = _load_json(os.path.join(str(path), 'dataset.json'))
    if isinstance(ds, dict) and ds.get('generator') == 'redrive':
        return 'redrive'
    return 'carla' if scene_kind == 'carla' else 'redrive'


def to_records(res: RigPerceptionResult, project=None, add=True, store=None) -> list:
    """One vantage.perception_record/1 per recording and detector (point figures per class and region). With add,
    each is written through core.perf_model.add_record when that module is there; the list is returned either way."""
    try:
        from . import perf_model as PM
    except ImportError:
        PM = None
    from . import provenance as PV
    out = []
    be = res.backend
    det_id = 'yolox_s'
    if be.get('id') == 'lifted':
        det_id = ((be.get('settings') or {}).get('detector') or {}).get('model') or DT.DEFAULT_MODEL
        backend = 'lifted'
    else:
        det_id = str(be.get('name', 'external')).replace('External: ', '')
        backend = f'external:{det_id}'
    detector = {'id': det_id, 'backend': backend, 'version': str(be.get('version', '')),
                'settings_hash': hashlib.sha256(json.dumps(
                    [be.get('settings_hash'), res.settings.score, res.settings.match_m, res.settings.instants,
                     res.settings.views, res.settings.group_nms_m], sort_keys=True).encode()).hexdigest()[:16]}
    for i, info in enumerate(res.recordings):
        try:
            rec = PerceptionRecording(info['path'], info.get('rig', ''))
        except ValueError:
            continue
        rig = _rig_block_of(info, rec, project)
        scene = _scene_block(rec, len(res.frames.get(i, [])))
        metrics = {}
        for ec, cells in (res.per_recording[i] if i < len(res.per_recording) else {}).items():
            m = {rid: {k: c[k] for k in ('n_gt', 'tp', 'fp', 'fn', 'recall', 'precision', 'ap', 'ate_m')}
                 for rid, c in cells.items() if rid != 'all'}
            if m:
                metrics[ec] = m
        if not metrics:
            continue
        prov = PV.block(None, {'metrics': metrics}, study='perception_record')
        if PM is not None:
            doc = PM.make_record(source=record_source(info['path'], scene['kind']), scene=scene,
                                 detector=detector, rig=rig, grid=res.grid, metrics=metrics, provenance=prov)
        else:
            from .. import __version__
            sc = dict(scene)
            sc['scene_hash'] = hashlib.sha256(PV.canonical({k: sc.get(k) for k in (
                'kind', 'town', 'scenario', 'seed', 'weather', 'duration_s', 'traffic', 'walkers', 'source_id')})
            ).hexdigest()
            doc = {'schema': RECORD_SCHEMA, 'id': '', 'created': PV._utc(), 'vantage': __version__,
                   'source': record_source(info['path'], scene['kind']), 'scene': sc, 'detector': detector,
                   'rig': rig, 'rig_hash': hashlib.sha256(PV.canonical(rig)).hexdigest(), 'grid': res.grid.to_dict(),
                   'metrics': metrics, 'provenance': prov}
            doc['id'] = hashlib.sha256(PV.canonical({k: v for k, v in doc.items()
                                                     if k not in ('id', 'created', 'provenance')})).hexdigest()
        if add and PM is not None:
            try:
                r = PM.add_record(doc, store)
                if r.get('errors'):
                    res.notes.append(f'Record of {info["label"]} not stored: {"; ".join(r["errors"][:3])}')
            except Exception as e:                        # noqa: BLE001 — the store is perf_model's
                res.notes.append(f'Record of {info["label"]} not stored: {e}')
        out.append(doc)
    return out


# =========================================================================== output
def _r(v, nd=4):
    if v is None:
        return None
    if isinstance(v, (float, np.floating)):
        return None if not math.isfinite(float(v)) else round(float(v), nd)
    if isinstance(v, (list, tuple)):
        return [_r(x, nd) for x in v]
    if isinstance(v, dict):
        return {k: _r(x, nd) for k, x in v.items()}
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


def summary(res: RigPerceptionResult) -> dict:
    out = {'schema': SCHEMA, 'rigs': [r['name'] for r in res.rigs], 'recordings': len(res.recordings),
           'backend': res.backend.get('name'), 'frames': res.timing.get('frames'), 'classes': res.classes,
           'evaluated_range_m': _r(res.eval_range_m, 2), 'per_rig': {}, 'findings': [], 'seconds': round(res.seconds, 2)}
    for rig in out['rigs']:
        out['per_rig'][rig] = {c: {k: _r((res.cell(rig, c) or {}).get(k)) for k in
                                   ('n_gt', 'recall', 'recall_ci', 'precision', 'ap', 'ap_ci', 'ate_m',
                                    'visible_share')} for c in res.classes}
    out['findings'] = [f['text'] for f in res.findings if f['significant']][:20]
    out['not_significant'] = sum(1 for f in res.findings if f['metric'] != 'note' and not f['significant'])
    return out


def to_doc(res: RigPerceptionResult, project=None) -> dict:
    st = res.settings
    return {'schema': SCHEMA, 'project': getattr(project, 'name', None), 'variant': res.variant,
            'frame': FRAME_V, 'units': {'distance': 'm', 'recall': 'share 0..1', 'ap': 'nuScenes AP 0..1',
                                        'ate_m': 'm'},
            'settings': dict({k: v for k, v in st.to_dict().items() if k not in ('scene_set', 'recordings')},
                             recordings=[{'file': os.path.basename(r['path']), 'rig': r.get('rig', '')}
                                         for r in st.recordings]),
            'grid': res.grid.to_dict(), 'regions': [{'id': rid, 'name': res.grid.region_name(s, r),
                                                     'area_m2': round(res.grid.area_m2(s, r), 2)}
                                                    for s, r, rid in _rid_cells(res.grid)],
            'evaluated_range_m': _r(res.eval_range_m, 3), 'thresholds_m': list(THRESHOLDS),
            'backend': res.backend, 'rigs': res.rigs,
            'recordings': [{k: (os.path.basename(v) if k == 'path' else v) for k, v in r.items()}
                           for r in res.recordings],
            'classes': res.classes, 'metrics': _r(res.metrics), 'comparisons': _r(res.comparisons),
            'findings': _r(res.findings), 'yields': res.yields, 'timing': res.timing, 'notes': res.notes,
            'assumptions': ASSUMPTIONS, 'limits': LIMITS_TEXT, 'references': list(REFERENCES)}


ROW_COLS = ('rig', 'class', 'region', 'region_name', 'n_gt', 'n_visible', 'visible_share', 'tp', 'fn', 'fp', 'recall',
            'recall_lo', 'recall_hi', 'recall_visible', 'precision', 'precision_lo', 'precision_hi', 'ap', 'ap_lo',
            'ap_hi', 'ap_0.5', 'ap_1', 'ap_2', 'ap_4', 'ate_m', 'ate_lo', 'ate_hi')
CMP_COLS = ('reference', 'rig', 'class', 'region', 'region_name', 'paired', 'n_gt_reference', 'n_gt',
            'recall_reference', 'recall', 'recall_difference', 'recall_difference_lo', 'recall_difference_hi',
            'ap_reference', 'ap', 'ap_difference', 'ap_difference_lo', 'ap_difference_hi', 'ate_reference', 'ate_m')


def _csv(rows, cols) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator='\n')
    w.writerow(cols)
    for r in rows:
        w.writerow(['' if r.get(c) is None else (f'{r[c]:.6g}' if isinstance(r.get(c), float) else r.get(c))
                    for c in cols])
    return buf.getvalue()


def region_rows(res: RigPerceptionResult) -> list:
    rows = []
    for rig, by in res.metrics.items():
        for ec, cells in by.items():
            for rid, c in cells.items():
                p = res.grid.parse_id(rid) if rid != 'all' else None
                row = {'rig': rig, 'class': ec, 'region': rid,
                       'region_name': res.grid.region_name(*p) if p else 'every region', **c}
                for k in ('recall', 'precision', 'ap', 'ate'):
                    ci = c.get(f'{k}_ci')
                    row[f'{k}_lo'], row[f'{k}_hi'] = (ci or (None, None))
                for t, a in (c.get('ap_by_threshold') or {}).items():
                    row[f'ap_{t}'] = a
                rows.append(row)
    return rows


def to_csv(res: RigPerceptionResult) -> str:
    return _csv(region_rows(res), ROW_COLS)


def comparisons_csv(res: RigPerceptionResult) -> str:
    rows = []
    for c in res.comparisons:
        r = dict(c)
        for k in ('recall', 'ap'):
            ci = c.get(f'{k}_difference_ci')
            r[f'{k}_difference_lo'], r[f'{k}_difference_hi'] = (ci or (None, None))
        rows.append(r)
    return _csv(rows, CMP_COLS)


# ---- the polar region map
def polar_cells(grid, inner=0.18, n_arc=12) -> list:
    """Each region's outline on a unit disc (front up, left on the left; rings of equal width, the car inside the
    inner radius): [(s, r, rid, [(x, y), ...])], x right, y up."""
    out = []
    nr = grid.n_ranges
    w = 360.0 / grid.sectors
    for s in range(grid.sectors):
        b0, b1 = math.radians(s * w - w / 2), math.radians(s * w + w / 2)
        for r in range(nr):
            r0 = inner + (1 - inner) * r / nr
            r1 = inner + (1 - inner) * (r + 1) / nr
            pts = []
            for k in range(n_arc + 1):
                b = b0 + (b1 - b0) * k / n_arc
                pts.append((-math.sin(b) * r1, math.cos(b) * r1))
            for k in range(n_arc, -1, -1):
                b = b0 + (b1 - b0) * k / n_arc
                pts.append((-math.sin(b) * r0, math.cos(b) * r0))
            out.append((s, r, grid.region_id(s, r), pts))
    return out


def map_values(res: RigPerceptionResult, cls, kind='recall', rig=None, other=None) -> dict:
    """region id -> (value, significant or thin flag, n): a rig's figure, or the difference other − rig."""
    out = {}
    if kind == 'difference':
        for c in res.comparisons:
            if c['class'] == cls and c['rig'] == other and c['region'] != 'all':
                dv, ci = c.get('recall_difference'), c.get('recall_difference_ci')
                n = min(c['n_gt'], c['n_gt_reference'])
                sig = bool(ci and (ci[0] > 0 or ci[1] < 0) and n >= res.settings.min_gt and
                           c.get('n_objects', 0) >= res.settings.min_objects)
                out[c['region']] = (dv, sig, n)
        return out
    for rid, c in ((res.metrics.get(rig) or {}).get(cls) or {}).items():
        if rid != 'all':
            out[rid] = (c.get(kind), c['n_gt'] >= res.settings.min_gt and
                        c.get('n_objects', 0) >= res.settings.min_objects, c['n_gt'])
    return out


def map_png(res: RigPerceptionResult, cls, kind='recall', rig=None, other=None, size=420, theme='light') -> bytes:
    """One polar region map as PNG (rings not to scale: equal width per range bin, labelled)."""
    import cv2
    from ..ui import colormaps as CM
    S = size
    img = np.full((S + 60, S, 3), 255, np.uint8)
    cx, cy, R = S / 2, S / 2 + 28, S / 2 - 14
    vals = map_values(res, cls, kind, rig, other)
    seq = CM.ramp(CM.BLUE, 256)[:, ::-1]                # recall 0 (pale) … 1 (dark blue), BGR
    div = CM.diverging('light')[:, ::-1]
    grey = tuple(int(x) for x in (CM.hex_rgb(CM.BELOW_MIN['light']) * 255)[::-1])
    hatch = tuple(int(x) for x in (CM.hex_rgb(CM.HATCH['light']) * 255)[::-1])
    for s, r, rid, pts in polar_cells(res.grid):
        poly = np.array([[cx + x * R, cy - y * R] for x, y in pts], np.int32)
        v = vals.get(rid)
        if v is None or v[0] is None:
            cv2.fillPoly(img, [poly], grey, cv2.LINE_AA)
            continue
        if kind == 'difference':
            t = 0.5 + 0.5 * max(-1.0, min(1.0, v[0] / 0.5))
            col = tuple(int(c) for c in div[int(round(t * (len(div) - 1)))])
        else:
            col = tuple(int(c) for c in seq[int(round(max(0.0, min(1.0, v[0])) * 255))])
        cv2.fillPoly(img, [poly], col, cv2.LINE_AA)
        if not v[1]:                                   # thin (fewer than min_gt) or not significant: hatched
            m = np.zeros(img.shape[:2], np.uint8)
            cv2.fillPoly(m, [poly], 255)
            hl = np.zeros_like(m)
            for k in range(-S, 2 * S, 9):
                cv2.line(hl, (k, 0), (k + S + 60, S + 60), 255, 1)
            img[(m > 0) & (hl > 0)] = hatch
        cv2.polylines(img, [poly], True, (255, 255, 255), 1, cv2.LINE_AA)
    cw, chh = int(R * 0.06), int(R * 0.13)
    cv2.rectangle(img, (int(cx - cw), int(cy - chh)), (int(cx + cw), int(cy + chh)), (60, 60, 60), -1)
    cv2.line(img, (int(cx), int(cy - chh)), (int(cx), int(cy - chh - 8)), (60, 60, 60), 2)
    lab = DT.CLASS_LABEL.get(cls, cls)
    title = (f'{lab}: {other} - {rig}, recall difference' if kind == 'difference' else f'{lab}: {rig}, {kind}')
    cv2.putText(img, title.encode('ascii', 'replace').decode()[:60], (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (40, 40, 40), 1, cv2.LINE_AA)
    names = res.grid.range_names()
    note = 'rings: ' + ', '.join(n.replace('–', '-').replace(' m', '') for n in names) + ' m; front up'
    cv2.putText(img, note[:70], (8, S + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 80), 1, cv2.LINE_AA)
    cv2.putText(img, 'hatched: ' + ('not significant' if kind == 'difference' else f'< {res.settings.min_gt} objects')
                + '; grey: no objects', (8, S + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 80), 1, cv2.LINE_AA)
    ok, buf = cv2.imencode('.png', img)
    return buf.tobytes() if ok else b''


def files(res: RigPerceptionResult, project=None, records=None) -> list:
    out = [('rig_perception.json', json.dumps(to_doc(res, project), indent=1, default=str).encode()),
           ('rig_perception_regions.csv', to_csv(res).encode()),
           ('rig_perception_comparisons.csv', comparisons_csv(res).encode()),
           ('rig_perception_findings.txt', ('\n'.join(f['text'] for f in res.findings) + '\n').encode())]
    rigs = [r['name'] for r in res.rigs]
    for ec in res.classes:
        for rig in rigs:
            out.append((f'map_{_slug(ec)}_{_slug(rig)}.png', map_png(res, ec, 'recall', rig)))
        for other in rigs[1:]:
            out.append((f'map_{_slug(ec)}_{_slug(other)}_vs_{_slug(rigs[0])}.png',
                        map_png(res, ec, 'difference', rigs[0], other)))
    recs = records if records is not None else to_records(res, project, add=False)
    if recs:
        out.append(('perception_records.jsonl', ''.join(json.dumps(r, sort_keys=True) + '\n' for r in recs).encode()))
    return out


def _slug(s) -> str:
    from .. import workdir as WD
    return WD.slug(s)[:40]


def export(res: RigPerceptionResult, project, out_dir) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for name, data in files(res, project):
        p = os.path.join(out_dir, name)
        with open(p, 'wb') as fh:
            fh.write(data)
        paths.append(p)
    return paths
