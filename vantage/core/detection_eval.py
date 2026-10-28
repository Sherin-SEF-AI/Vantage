"""Detection quality per rig: how well an off-the-shelf detector (core.detection: YOLOX through OpenCV's DNN module)
finds the objects in a rig's camera images, measured against the drive's own ground truth (a CARLA run with ground
truth: core.groundtruth, schema vantage.ground_truth/1), per camera, class and range, and compared between rigs.

What is measured
----------------
For every camera frame of a recording the detector runs on the image as it was recorded (the fisheye image itself;
optionally also on virtual pinhole views cut from it, below), and its detections are matched to the frame's
ground-truth objects:
* Classes: core.detection maps COCO's classes and the ground truth's onto Vantage's (pedestrian, cyclist,
  motorcyclist, car, truck, bus, traffic light, stop sign) and the group 'vehicle' (car, truck and bus without telling
  them apart; the group's detections pass one class-blind NMS first, so a box called both car and truck counts once).
* Boxes: a ground-truth box is the tight box of the object's visible pixels (its instance mask), as COCO boxes are.
* Matching per frame and class as COCO's evaluation does (Lin et al. 2014; pycocotools' COCOeval.evaluateImg):
  detections in descending score order, each to the unmatched object with the largest IoU at or above the threshold,
  objects to ignore only when no other is left; IoU ≥ iou (0.5), or ≥ iou_small (0.3) for objects less than small_px
  (32 px) tall, where a box a few pixels off loses most of its IoU.
* Ignore regions: an object less than min_px tall or with fewer than min_pixels visible pixels, more than max_occluded
  hidden or more than max_truncated outside the image (the ground truth's own shares), and a traffic sign not known to
  be a stop sign, is neither a miss nor a hit: a detection on it is ignored, and so is an unmatched detection less
  than min_px tall.
* Range: an object's range is the ground truth's (camera to box centre); a detection that matches nothing takes the
  range the range image gives inside it (the median over the box's central half), so precision is per range too.
  Bins: 0–10, 10–20, 20–40 and 40+ m by default (range_edges).
* Figures: recall and precision at the operating point (score ≥ score), and the average precision over every
  detection kept (keep_score): COCO's 101-point interpolated AP at the matching IoU above ('AP50'), and COCO's AP
  averaged over IoU 0.50:0.05:0.95 per camera and class. The detection range of a camera and class is the farthest
  range bin with recall ≥ 0.5 among bins with at least min_gt objects. Recall against the object's height in pixels
  (bins 16, 32, 64 … px) and the height where it crosses 50 % (interpolated in log height).
* The rig: an object (an actor, at one camera frame time) counts once for the whole rig, detected when any camera
  detects it, ranged by its distance from the vehicle (the box centre from the rear axle, ISO 8855), so rigs with
  different cameras compare on the same objects.
* Uncertainty: percentile intervals from a block bootstrap over time (Efron and Tibshirani 1993; Künsch 1989): the
  drive cut into blocks of block_s seconds (frames close in time see the same objects, so single frames are not
  independent), blocks drawn with replacement, every figure computed again; 1000 resamples by default.
* Comparing recordings (rig variants): every figure of each against the first, as differences with intervals. When
  the recordings ran the same scenario with the same seed, town and duration, the same time blocks are drawn for
  both (a paired bootstrap: the scene is the same, so its difficulty cancels); otherwise independently, and said so.
* Pinhole views (views = 'pinhole' or 'both'): virtual pinhole cameras sharing the fisheye's optical centre, level
  (no roll), pitched view_pitch_deg down, turned view_yaws about the vertical from the camera's heading, view_fov_deg
  wide, cut from the image through the recording's own calibration (camera_intrinsic and the Kannala-Brandt
  coefficients of calibrated_sensor's vantage_distortion, as cv2.fisheye uses them). The views' ground truth is the
  instance and range images remapped the same way (nearest pixel); an object's truncation in a view adds the share of
  its fisheye pixels outside the view. Per object, fisheye against views: detected in the fisheye image, detected in
  any view, the difference with a paired interval.
"""
from __future__ import annotations

import concurrent.futures as cf
import csv
import hashlib
import io
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from . import detection as DT

SCHEMA = 'vantage.detection_quality/1'
FRAME_V = 'vehicle (ISO 8855, rear axle)'
VIEW_MODES = ('fisheye', 'pinhole', 'both')
MODE_LABEL = {'fisheye': 'Fisheye images', 'pinhole': 'Pinhole views', 'both': 'Fisheye and pinhole views'}
CACHE_DIR = 'detections'               # in a recording: the detector's output, reused while its inputs stay the same
CACHE_SCHEMA = 'vantage.detections_cache/1'
AP_IOUS = tuple(round(0.5 + 0.05 * k, 2) for k in range(10))
IGN_NONE, IGN_SIZE, IGN_VIS, IGN_SIGN = 0, 1, 2, 3
IGN_TEXT = {IGN_SIZE: 'too small', IGN_VIS: 'mostly hidden or outside the image', IGN_SIGN: 'a sign of unknown type'}
MAX_RECORDINGS = 8
# evaluated by default: every class with complete ground truth (stop signs need the sign's type: see LIMITS_TEXT)
DEFAULT_CLASSES = ('pedestrian', 'cyclist', 'motorcyclist', 'car', 'truck', 'bus', 'traffic_light')
ASSUMPTIONS = (
    'The detector is YOLOX trained on COCO\'s photographs, run on CARLA\'s renders: absolute figures are the '
    'detector\'s on synthetic images, not a production detector\'s on real ones. What carries over is the comparison '
    'of rigs, cameras and image formats on the same scene. Ground-truth boxes are the tight boxes of each object\'s '
    'visible pixels (CARLA\'s instance camera through the rig\'s remap), as COCO\'s are. A false detection is ranged '
    'by the range image inside its box (the median over its central half). Intervals are percentile intervals of a '
    'block bootstrap over time; they cover the sampling of this drive\'s moments, not other towns, weather or traffic.')
LIMITS_TEXT = (
    'Stop signs are left out by default: COCO knows stop signs but no other sign, and the ground truth does not say '
    'which sign is which, so every sign is an ignore region for the stop-sign class (a detection on a sign is neither '
    'right nor wrong; there is no recall). On CARLA\'s Town10HD YOLOX also calls the banners on its street-light posts '
    'and some windows traffic lights: real false detections, counted against the traffic-light precision. Pinhole '
    'views use the recording\'s Kannala-Brandt coefficients (the dataset export\'s fit of the lens), not the lens '
    'model itself.')

LIMITS = {'score': (0.01, 0.99), 'iou': (0.1, 0.95), 'iou_small': (0.05, 0.95), 'small_px': (0.0, 512.0),
          'min_px': (0.0, 256.0), 'min_pixels': (0, 100000), 'max_occluded': (0.0, 1.0), 'max_truncated': (0.0, 1.0),
          'frame_step': (1, 1000), 'view_fov_deg': (30.0, 150.0), 'view_pitch_deg': (-60.0, 89.0),
          'view_width': (128, 2048), 'view_aspect': (0.3, 1.5), 'bootstrap': (0, 10000), 'block_s': (0.0, 60.0),
          'confidence': (0.5, 0.999), 'seed': (0, 2 ** 31 - 1), 'min_gt': (1, 10000)}
INTS = ('min_pixels', 'frame_step', 'view_width', 'bootstrap', 'seed', 'min_gt')
BOOLS = ('cache',)


# =========================================================================== settings
@dataclass
class DetectionSettings:
    recordings: list = field(default_factory=list)   # recording folders (CARLA runs with ground truth); the first is
                                                     # the reference the others are compared with
    # the detector (core.detection.DetectorSettings)
    model: str = DT.DEFAULT_MODEL
    input_px: int = 0
    max_input_px: int = 1280
    keep_score: float = 0.05
    nms_iou: float = 0.45
    max_dets: int = 100
    fuse_riders: bool = True
    backend: str = 'cpu'
    threads: int = 0
    # matching and ignore regions
    score: float = 0.3                 # the operating point: recall and precision count detections at or above it
    iou: float = 0.5
    iou_small: float = 0.3             # for objects less than small_px tall
    small_px: float = 32.0
    min_px: float = 10.0               # objects (and unmatched detections) less tall are ignored
    min_pixels: int = 20               # objects with fewer visible pixels are ignored
    max_occluded: float = 0.8
    max_truncated: float = 0.5
    range_edges: list = field(default_factory=lambda: [10.0, 20.0, 40.0])        # m: bins 0–10, 10–20, 20–40, 40+
    height_edges: list = field(default_factory=lambda: [16.0, 32.0, 64.0, 128.0, 256.0])   # px
    classes: list = field(default_factory=lambda: list(DEFAULT_CLASSES))    # stop signs on request
    cameras: list = field(default_factory=list)      # camera tags; [] = every camera of each recording
    frame_step: int = 1                # every n-th camera frame
    # pinhole views cut from the fisheye images
    views: str = 'fisheye'             # fisheye | pinhole | both
    view_fov_deg: float = 90.0         # horizontal
    view_pitch_deg: float = 10.0       # down
    view_yaws: list = field(default_factory=lambda: [-60.0, 0.0, 60.0])    # from the camera's heading, left positive
    view_width: int = 640
    view_aspect: float = 0.75          # height / width
    # uncertainty
    bootstrap: int = 1000
    block_s: float = 1.0
    confidence: float = 0.95
    seed: int = 0
    min_gt: int = 5
    cache: bool = True                 # keep the detections in the recording (detections/) for the next run

    def to_dict(self):
        return asdict(self)

    def detector(self) -> DT.DetectorSettings:
        return DT.DetectorSettings.from_dict(self.to_dict())

    @staticmethod
    def from_dict(d) -> 'DetectionSettings':
        s = DetectionSettings()
        d = d if isinstance(d, dict) else {}
        ds = DT.DetectorSettings.from_dict(d)
        for k in ds.to_dict():
            setattr(s, k, getattr(ds, k))
        recs = d.get('recordings')
        if isinstance(recs, (list, tuple)):
            out = []
            for r in recs:
                if isinstance(r, str) and r.strip() and r.strip() not in out:
                    out.append(r.strip()[:4096])
            s.recordings = out[:MAX_RECORDINGS]
        if d.get('views') in VIEW_MODES:
            s.views = d['views']
        for k in BOOLS:
            v = d.get(k)
            if isinstance(v, (bool, int)) and not isinstance(v, float):
                setattr(s, k, bool(v))
            elif isinstance(v, str):
                setattr(s, k, v.strip().lower() in ('1', 'true', 'yes', 'on'))
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
        for k, lo, hi, n in (('range_edges', 0.1, 1000.0, 8), ('height_edges', 1.0, 4096.0, 10)):
            v = _num_list(d.get(k), lo, hi, n)
            if v is not None:
                setattr(s, k, sorted(set(v)))
        v = _num_list(d.get('view_yaws'), -180.0, 180.0, 8)
        if v is not None and v:
            s.view_yaws = v
        cl = d.get('classes')
        if isinstance(cl, (list, tuple)):
            s.classes = [c for c in DT.CLASSES if c in cl]
        cams = d.get('cameras')
        if isinstance(cams, (list, tuple)):
            s.cameras = [str(c)[:16] for c in cams if isinstance(c, str) and c.strip()][:32]
        return s


def _num_list(v, lo, hi, n):
    if isinstance(v, str):
        v = [x for x in v.replace(';', ',').split(',') if x.strip()]
    if not isinstance(v, (list, tuple)):
        return None
    out = []
    for x in v[:n]:
        try:
            f = float(x)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            out.append(min(max(f, lo), hi))
    return out


def settings_of(project) -> DetectionSettings:
    a = getattr(project, 'analysis', None) if project is not None else None
    return DetectionSettings.from_dict(getattr(a, 'detection', None) if a is not None else None)


def bin_labels(edges, unit='m') -> list:
    e = [0.0] + [float(x) for x in edges]
    out = [f'{_fmt(e[i])}–{_fmt(e[i + 1])} {unit}' for i in range(len(e) - 1)]
    return out + [f'{_fmt(e[-1])}+ {unit}']


def height_labels(edges) -> list:
    e = [float(x) for x in edges]
    if not e:
        return ['every height']
    return [f'< {_fmt(e[0])} px'] + [f'{_fmt(e[i])}–{_fmt(e[i + 1])} px' for i in range(len(e) - 1)] + \
        [f'{_fmt(e[-1])}+ px']


def _fmt(x):
    return f'{x:g}'


def bin_of(values, edges) -> np.ndarray:
    """Bin index of each value (0 … len(edges)); -1 for NaN."""
    v = np.asarray(values, float)
    b = np.searchsorted(np.asarray(edges, float), v, side='right')
    return np.where(np.isfinite(v), b, -1).astype(np.int16)


# =========================================================================== a recording
@dataclass
class Camera:
    tag: str
    channel: str
    width: int
    height: int
    frames: list
    K: np.ndarray | None = None
    D: np.ndarray | None = None          # Kannala-Brandt k1..k4 (cv2.fisheye); None: a pinhole (or unknown)
    lens: str = ''                       # 'kb' | 'pinhole' | '' (no calibration found)
    R: np.ndarray | None = None          # optical frame -> vehicle frame
    t: np.ndarray | None = None          # camera position in the vehicle frame (m)
    max_theta: float | None = None       # rad, the lens's field (sim_rig.json), when known


@dataclass
class Recording:
    path: str
    label: str
    gt: dict
    cameras: dict                        # tag -> Camera
    run: dict | None = None
    project: str = ''
    variant: str = ''
    scenario: str = ''
    seed: int | None = None
    town: str = ''
    weather: str = ''
    duration_s: float | None = None

    def same_scene(self, other: 'Recording') -> bool:
        """Same scenario, seed, town and duration: their time blocks can be paired."""
        return bool(self.scenario == other.scenario and self.seed is not None and self.seed == other.seed and
                    self.town == other.town and self.duration_s is not None and other.duration_s is not None and
                    abs(self.duration_s - other.duration_s) < 1e-6)

    def info(self) -> dict:
        n = sum(len(c.frames) for c in self.cameras.values())
        objs = sum(len(f.get('objects') or ()) for c in self.cameras.values() for f in c.frames)
        return {'path': self.path, 'label': self.label, 'project': self.project, 'variant': self.variant,
                'scenario': self.scenario, 'seed': self.seed, 'town': self.town, 'weather': self.weather,
                'duration_s': self.duration_s, 'cameras': sorted(self.cameras), 'frames': n, 'objects': objs}


def _quat_matrix(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _load_json(path):
    try:
        with open(path, encoding='utf-8') as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _calibration(path) -> dict:
    """channel -> calibrated_sensor row of the recording's nuScenes tables."""
    vdir = os.path.join(path, 'nuscenes', 'v1.0-vantage')
    sens = _load_json(os.path.join(vdir, 'sensor.json')) or []
    cal = _load_json(os.path.join(vdir, 'calibrated_sensor.json')) or []
    chan = {r.get('token'): r.get('channel') for r in sens if isinstance(r, dict)}
    return {chan.get(r.get('sensor_token')): r for r in cal if isinstance(r, dict) and chan.get(r.get('sensor_token'))}


def _gt_doc(path) -> dict | None:
    try:
        from . import groundtruth as GT
    except ImportError:                                  # pragma: no cover — the reader ships with the app
        GT = None
    if GT is not None:
        return GT.read(path)
    doc = _load_json(os.path.join(path, 'ground_truth', 'index.json'))
    if isinstance(doc, dict) and doc.get('schema') == 'vantage.ground_truth/1':
        doc['_root'] = os.path.abspath(path)
        return doc
    return None


def open_recording(path) -> Recording:
    """A recording with ground truth (ValueError when the folder has none)."""
    path = os.path.abspath(os.path.expanduser(str(path)))
    if not os.path.isdir(path):
        raise ValueError(f'{path} is not a folder.')
    doc = _gt_doc(path)
    if doc is None:
        raise ValueError(f'{os.path.basename(path)} has no ground truth (ground_truth/index.json): run CARLA with '
                         'ground truth on (Simulate ▸ CARLA).')
    cal = _calibration(path)
    rig = _load_json(os.path.join(path, 'rig', 'sim_rig.json')) or {}
    rig_cams = {c.get('tag'): c for c in rig.get('cameras', []) if isinstance(c, dict)}
    cams = {}
    for tag, c in (doc.get('cameras') or {}).items():
        if not isinstance(c, dict):
            continue
        cam = Camera(str(tag), str(c.get('channel') or f'CAM_{tag}'), int(c.get('width') or 0),
                     int(c.get('height') or 0), [f for f in c.get('frames') or [] if isinstance(f, dict)])
        row = cal.get(cam.channel)
        if row is not None and row.get('camera_intrinsic'):
            cam.K = np.asarray(row['camera_intrinsic'], float)
            cam.R = _quat_matrix(row.get('rotation', (1, 0, 0, 0)))
            cam.t = np.asarray(row.get('translation', (0, 0, 0)), float)
            vd = row.get('vantage_distortion') or {}
            model = str(vd.get('model', ''))
            if model in ('kannala_brandt', 'equidistant'):
                cam.D = np.asarray((list(vd.get('coefficients') or []) + [0.0] * 4)[:4], float)
                cam.lens = 'kb'
            elif model.startswith('none'):
                cam.lens = 'pinhole'
        rc = rig_cams.get(tag) or {}
        mt = (rc.get('image') or {}).get('max_theta_deg')
        if isinstance(mt, (int, float)) and math.isfinite(mt):
            cam.max_theta = math.radians(float(mt))
        cams[cam.tag] = cam
    run = None
    try:
        from . import simrig as SR
        run = SR.read_run(path)
    except Exception:                                    # noqa: BLE001 — the run report is optional
        run = _load_json(os.path.join(path, 'run.json'))
    ds = _load_json(os.path.join(path, 'dataset.json')) or {}
    run = run if isinstance(run, dict) else None
    sc = (run or {}).get('scene') or {}
    scen = (run or {}).get('scenario')
    scen_name = scen.get('name') if isinstance(scen, dict) else (scen if isinstance(scen, str) else '')
    rec = Recording(path, '', doc, cams, run, str((run or {}).get('project') or ds.get('project') or ''),
                    str((run or {}).get('variant') or ds.get('variant') or ''), str(scen_name or ''),
                    sc.get('seed') if isinstance(sc.get('seed'), int) else None, str(sc.get('town') or ''),
                    str(sc.get('weather') or ''),
                    float(sc['duration_s']) if isinstance(sc.get('duration_s'), (int, float)) else None)
    bits = [rec.variant or os.path.basename(path)]
    if rec.scenario:
        bits.append(rec.scenario.replace('_', ' '))
    if rec.seed is not None:
        bits.append(f'seed {rec.seed}')
    rec.label = ' · '.join(bits)
    return rec


def recording_info(path) -> dict:
    """What a folder holds, for the page's list (cheap: the index only): {'ok', 'text', ...Recording.info()}."""
    try:
        rec = open_recording(path)
    except ValueError as e:
        return {'ok': False, 'path': str(path), 'text': str(e)}
    d = rec.info()
    d['ok'] = True
    d['text'] = (f'{len(rec.cameras)} cameras, {d["frames"]} frames, {d["objects"]} labelled objects' +
                 (f' · {rec.town}' if rec.town else '') + (f' · {rec.weather}' if rec.weather else ''))
    return d


# =========================================================================== ground truth of a frame
def gt_objects(frame: dict, st: DetectionSettings) -> dict:
    """The frame's objects as arrays: 'id', 'key' (actor or instance: the same object in every camera), 'cls' (index
    into CLASSES, -1 for classes not evaluated), 'box' (n × 4), 'height', 'pixels', 'occ', 'trunc', 'range', 'dist'
    (box centre from the rear axle, m), 'ignore' (IGN_*)."""
    objs = [o for o in frame.get('objects') or () if isinstance(o, dict) and o.get('bbox')]
    n = len(objs)
    out = {'id': np.zeros(n, np.int64), 'key': [], 'cls': np.full(n, -1, np.int16), 'box': np.zeros((n, 4)),
           'pixels': np.zeros(n), 'occ': np.zeros(n), 'trunc': np.zeros(n), 'range': np.full(n, np.nan),
           'dist': np.full(n, np.nan), 'xy': np.full((n, 2), np.nan), 'ignore': np.zeros(n, np.int8)}
    for i, o in enumerate(objs):
        out['id'][i] = int(o.get('id') or 0)
        key = o.get('instance_token') or (f'actor:{o["actor_id"]}' if o.get('actor_id') is not None else None)
        out['key'].append(str(key) if key else f'obj:{frame.get("sample_data_token", "")}:{o.get("id")}')
        name = DT.GT_TO_CLASS.get(str(o.get('class') or ''))
        if name is not None:
            out['cls'][i] = DT.CLASSES.index(name)
        out['box'][i] = [float(v) for v in o['bbox'][:4]]
        out['pixels'][i] = float(o.get('pixels') or 0)
        out['occ'][i] = float(o.get('occluded') or 0.0)
        out['trunc'][i] = float(o.get('truncated') or 0.0)
        if o.get('range_m') is not None:
            out['range'][i] = float(o['range_m'])
        b3 = o.get('box3d') or {}
        c = b3.get('center') if isinstance(b3, dict) else None
        out['dist'][i] = math.hypot(float(c[0]), float(c[1])) if c else out['range'][i]
        if c:
            out['xy'][i] = [float(c[0]), float(c[1])]
        if name == 'stop_sign' and not _is_stop(o):
            out['ignore'][i] = IGN_SIGN
    out['height'] = out['box'][:, 3] - out['box'][:, 1]
    _ignore(out, st)
    return out


def _is_stop(o: dict) -> bool:
    for k in ('type_id', 'blueprint', 'carla_type', 'sign'):
        v = o.get(k)
        if isinstance(v, str) and 'stop' in v.lower():
            return True
    return False


def _ignore(g: dict, st: DetectionSettings):
    ig = g['ignore']
    size = (g['height'] < st.min_px) | (g['pixels'] < st.min_pixels)
    vis = (g['occ'] > st.max_occluded) | (g['trunc'] > st.max_truncated)
    free = ig == IGN_NONE
    ig[free & vis] = IGN_VIS
    ig[(ig == IGN_NONE) & size] = IGN_SIZE


# =========================================================================== matching (COCO's rules)
def iou_matrix(a, b) -> np.ndarray:
    """IoU of every box of a (n × 4, x0 y0 x1 y1) with every box of b (m × 4): n × m."""
    a = np.asarray(a, float).reshape(-1, 4)
    b = np.asarray(b, float).reshape(-1, 4)
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    iw = np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0])
    ih = np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1])
    inter = np.clip(iw, 0, None) * np.clip(ih, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = aa[:, None] + ab[None, :] - inter
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(union > 0, inter / union, 0.0)


def match(gt_boxes, gt_ignore, gt_thr, det_boxes, det_scores, det_ignore=None):
    """Greedy matching as COCOeval.evaluateImg: detections in descending score order (stable), each to the unmatched
    object it overlaps most with IoU ≥ that object's threshold, preferring objects that are not ignored; a detection
    matched to an ignored object, or unmatched and flagged in det_ignore, is ignored.
    Returns (det_status int8: 1 true positive, 0 false positive, -1 ignored; det_gt: the object's index or -1;
    gt_score: the matched detection's score per object, -inf when unmatched)."""
    gt_boxes = np.asarray(gt_boxes, float).reshape(-1, 4)
    det_boxes = np.asarray(det_boxes, float).reshape(-1, 4)
    scores = np.asarray(det_scores, float).reshape(-1)
    ng, nd = len(gt_boxes), len(det_boxes)
    gig = np.asarray(gt_ignore, bool).reshape(-1) if ng else np.zeros(0, bool)
    thr = np.broadcast_to(np.asarray(gt_thr, float), (ng,)) if ng else np.zeros(0)
    status = np.zeros(nd, np.int8)
    det_gt = np.full(nd, -1, np.int64)
    gt_score = np.full(ng, -np.inf)
    if nd == 0:
        return status, det_gt, gt_score
    dig = np.zeros(nd, bool) if det_ignore is None else np.asarray(det_ignore, bool).reshape(-1)
    if ng == 0:
        status[dig] = -1
        return status, det_gt, gt_score
    ious = iou_matrix(det_boxes, gt_boxes)
    taken = np.zeros(ng, bool)
    for d in np.argsort(-scores, kind='mergesort'):
        ok = (ious[d] >= thr - 1e-12) & ~taken
        best = -1
        cand = ok & ~gig
        if cand.any():
            best = int(np.argmax(np.where(cand, ious[d], -1.0)))
        else:
            cand = ok & gig
            if cand.any():
                best = int(np.argmax(np.where(cand, ious[d], -1.0)))
        if best >= 0:
            taken[best] = True
            det_gt[d] = best
            gt_score[best] = scores[d]
            status[d] = -1 if gig[best] else 1
        else:
            status[d] = -1 if dig[d] else 0
    return status, det_gt, gt_score


def nms_agnostic(boxes, scores, thr) -> np.ndarray:
    """Indices kept by class-blind non-maximum suppression (highest score first)."""
    boxes = np.asarray(boxes, float).reshape(-1, 4)
    order = np.argsort(-np.asarray(scores, float), kind='mergesort')
    keep = []
    ious = iou_matrix(boxes, boxes)
    dead = np.zeros(len(boxes), bool)
    for i in order:
        if dead[i]:
            continue
        keep.append(i)
        dead |= ious[i] > thr
    return np.asarray(sorted(keep, key=lambda i: -float(scores[i])), int)


# =========================================================================== average precision
def ap_coco(scores, tp, n_gt) -> float | None:
    """COCO's 101-point interpolated average precision of one set of (non-ignored) detections: precision made
    monotone from the end, read at recall 0, 0.01 … 1 (the first detection reaching each), 0 beyond the last recall;
    None without objects. (Reference implementation: the tests check the vectorised one against it.)"""
    if n_gt <= 0:
        return None
    order = np.argsort(-np.asarray(scores, float), kind='mergesort')
    t = np.asarray(tp, float)[order]
    ctp, cfp = np.cumsum(t), np.cumsum(1.0 - t)
    rc = ctp / n_gt
    pr = ctp / np.maximum(ctp + cfp, np.spacing(1))
    pr = np.maximum.accumulate(pr[::-1])[::-1] if len(pr) else pr
    thr = np.arange(101) / 100.0
    idx = np.searchsorted(rc, thr - 1e-9, side='left')
    q = np.where(idx < len(pr), pr[np.minimum(idx, max(len(pr) - 1, 0))] if len(pr) else 0.0, 0.0)
    return float(np.mean(q))


def _ap_weighted(tp_sorted, wd, ngt_w, chunk_elems=4_000_000) -> np.ndarray:
    """_ap_rows over the resamples in chunks (memory: a few arrays of chunk_elems floats)."""
    wd = np.asarray(wd, float)
    if wd.ndim == 1:
        wd = wd[None]
    ngt_w = np.asarray(ngt_w, float).reshape(len(wd))
    n = max(wd.shape[1], 1)
    step = max(1, int(chunk_elems // n))
    if len(wd) <= step:
        return _ap_rows(tp_sorted, wd, ngt_w)
    return np.concatenate([_ap_rows(tp_sorted, wd[i:i + step], ngt_w[i:i + step]) for i in range(0, len(wd), step)])


def _ap_rows(tp_sorted, wd, ngt_w) -> np.ndarray:
    """101-point AP for B weightings at once: tp_sorted (n,) in descending score order, wd (B, n) each detection's
    weight, ngt_w (B,) the weighted object count. Each threshold t_k = k/100 takes the envelope at the first detection
    whose recall reaches it: detection i covers C(r_i) − C(r_{i−1}) thresholds, C(r) = ⌊100 r⌋ + 1."""
    wd = np.asarray(wd, float)
    if wd.ndim == 1:
        wd = wd[None]
    B, n = wd.shape
    ngt_w = np.asarray(ngt_w, float).reshape(B)
    out = np.full(B, np.nan)
    if n == 0:
        out[ngt_w > 0] = 0.0
        return out
    t = np.asarray(tp_sorted, float)[None]
    ctp = np.cumsum(wd * t, axis=1)
    cfp = np.cumsum(wd * (1.0 - t), axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        rc = ctp / ngt_w[:, None]
        pr = np.where(ctp + cfp > 0, ctp / np.maximum(ctp + cfp, 1e-300), 0.0)
    env = np.maximum.accumulate(pr[:, ::-1], axis=1)[:, ::-1]
    C = np.floor(np.clip(rc, 0, 1) * 100 + 1e-7) + 1.0
    C = np.where(np.isfinite(C), C, 0.0)
    n_i = np.diff(np.concatenate([np.zeros((B, 1)), C], axis=1), axis=1)
    ap = np.sum(env * n_i, axis=1) / 101.0
    ok = ngt_w > 0
    out[ok] = ap[ok]
    return out


# =========================================================================== pinhole views
@dataclass
class View:
    name: str
    yaw: float
    w: int
    h: int
    K: np.ndarray
    R: np.ndarray                         # view optical -> camera optical
    map_x: np.ndarray                     # view pixel -> fisheye pixel (cv2.remap)
    map_y: np.ndarray
    inside: np.ndarray                    # fisheye pixels whose ray lands in the view


def kb_project(cam: Camera, P) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Camera optical-frame directions (n × 3) -> fisheye pixel (u, v) and ok (within the lens's field)."""
    P = np.asarray(P, float).reshape(-1, 3)
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    rho = np.hypot(x, y)
    theta = np.arctan2(rho, z)
    K = cam.K
    if cam.lens == 'pinhole' or cam.D is None:
        ok = z > 1e-6
        zz = np.where(ok, z, 1.0)
        return K[0, 0] * x / zz + K[0, 2], K[1, 1] * y / zz + K[1, 2], ok
    k1, k2, k3, k4 = cam.D
    t2 = theta * theta
    td = theta * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))
    s = np.where(rho > 1e-12, td / np.where(rho > 1e-12, rho, 1.0), 0.0)
    ok = theta <= _theta_limit(cam)
    return K[0, 0] * x * s + K[0, 2], K[1, 1] * y * s + K[1, 2], ok


def _theta_limit(cam: Camera) -> float:
    """The largest ray angle the model maps one to one and inside the image's reach: the lens's field when the rig
    says it, the polynomial's first turning point, and the angle reaching past the image's farthest corner."""
    if cam.D is None:
        return math.radians(89.0)
    th = np.linspace(0, math.pi, 3601)
    k1, k2, k3, k4 = cam.D
    t2 = th * th
    r = th * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4)))) * cam.K[0, 0]
    dr = np.diff(r)
    lim = th[np.argmax(dr <= 0)] if (dr <= 0).any() else math.pi
    corner = max(math.hypot(px - cam.K[0, 2], py - cam.K[1, 2]) for px in (0, cam.width) for py in (0, cam.height))
    beyond = r > corner * 1.05
    if beyond.any():
        lim = min(lim, th[np.argmax(beyond)])
    if cam.max_theta is not None:
        lim = min(lim, cam.max_theta)
    return float(lim)


def kb_unproject(cam: Camera, u, v) -> tuple[np.ndarray, np.ndarray]:
    """Fisheye pixels -> unit optical-frame rays and ok (Newton on θ_d = θ(1 + k1 θ² + …))."""
    K = cam.K
    mx = (np.asarray(u, float) - K[0, 2]) / K[0, 0]
    my = (np.asarray(v, float) - K[1, 2]) / K[1, 1]
    td = np.hypot(mx, my)
    if cam.lens == 'pinhole' or cam.D is None:
        d = np.stack([mx, my, np.ones_like(mx)], -1)
        return d / np.linalg.norm(d, axis=-1, keepdims=True), np.ones(mx.shape, bool)
    k1, k2, k3, k4 = cam.D
    th = td.copy()
    for _ in range(20):
        t2 = th * th
        f = th * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4)))) - td
        df = 1 + t2 * (3 * k1 + t2 * (5 * k2 + t2 * (7 * k3 + t2 * 9 * k4)))
        th = th - f / np.where(np.abs(df) > 1e-9, df, 1e-9)
    t2 = th * th
    err = np.abs(th * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4)))) - td)
    ok = (err < 1e-6) & (th >= 0) & (th <= _theta_limit(cam))
    s = np.where(td > 1e-12, np.sin(th) / np.where(td > 1e-12, td, 1.0), 1.0)
    d = np.stack([mx * s, my * s, np.cos(th)], -1)
    return d, ok


def _heading(cam: Camera) -> float:
    """The camera's heading in the vehicle frame (rad, left positive): its optical axis, else its image's right."""
    z = cam.R[:, 2]
    if math.hypot(z[0], z[1]) > 1e-3:
        return math.atan2(z[1], z[0])
    x = cam.R[:, 0]                       # straight down: the heading its image's top points to
    return math.atan2(x[1], x[0]) + math.pi / 2


def make_views(cam: Camera, st: DetectionSettings) -> list:
    """The camera's virtual pinhole views (empty when its calibration is missing)."""
    if cam.K is None or cam.R is None or not cam.lens:
        return []
    out = []
    W = int(st.view_width)
    H = max(16, int(round(W * st.view_aspect)))
    f = (W / 2.0) / math.tan(math.radians(st.view_fov_deg) / 2)
    Kv = np.array([[f, 0, (W - 1) / 2.0], [0, f, (H - 1) / 2.0], [0, 0, 1.0]])
    psi0 = _heading(cam)
    pitch = math.radians(st.view_pitch_deg)
    vv, uu = np.mgrid[0:cam.height, 0:cam.width]
    fe_rays, fe_ok = kb_unproject(cam, uu.ravel().astype(float), vv.ravel().astype(float))
    fe_rays_v = fe_rays @ cam.R.T                                   # vehicle frame
    for yaw in st.view_yaws:
        psi = psi0 + math.radians(float(yaw))
        fwd = np.array([math.cos(pitch) * math.cos(psi), math.cos(pitch) * math.sin(psi), -math.sin(pitch)])
        right = np.array([math.sin(psi), -math.cos(psi), 0.0])
        down = np.cross(fwd, right)
        R_vv = np.column_stack([right, down, fwd])                  # view optical -> vehicle
        R_cv = cam.R.T @ R_vv                                       # view optical -> camera optical
        gy, gx = np.mgrid[0:H, 0:W].astype(float)
        d = np.stack([(gx - Kv[0, 2]) / f, (gy - Kv[1, 2]) / f, np.ones_like(gx)], -1).reshape(-1, 3) @ R_cv.T
        u, v, ok = kb_project(cam, d)
        ok &= (u >= -0.5) & (u <= cam.width - 0.5) & (v >= -0.5) & (v <= cam.height - 0.5)
        mx = np.where(ok, u, -10.0).reshape(H, W).astype(np.float32)
        my = np.where(ok, v, -10.0).reshape(H, W).astype(np.float32)
        q = fe_rays_v @ R_vv                                        # fisheye rays in the view's optical frame
        z = q[:, 2]
        zz = np.where(z > 1e-6, z, 1.0)
        pu = f * q[:, 0] / zz + Kv[0, 2]
        pv = f * q[:, 1] / zz + Kv[1, 2]
        inside = fe_ok & (z > 1e-6) & (pu >= -0.5) & (pu <= W - 0.5) & (pv >= -0.5) & (pv <= H - 0.5)
        name = f'{cam.tag}·{"+" if yaw > 0 else ""}{yaw:g}°' if yaw else f'{cam.tag}·0°'
        out.append(View(name, float(yaw), W, H, Kv, R_cv, mx, my, inside.reshape(cam.height, cam.width)))
    return out


def remap(img, view: View, nearest=False, fill=0):
    import cv2
    return cv2.remap(img, view.map_x, view.map_y, cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=fill)


def boxes_of_ids(ids_img) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(ids, boxes n × 4 [x0, y0, x1 + 1, y1 + 1], pixel counts) of every non-zero id of an instance image."""
    a = np.asarray(ids_img)
    flat = a.ravel()
    nz = np.flatnonzero(flat)
    if not len(nz):
        return np.zeros(0, np.int64), np.zeros((0, 4)), np.zeros(0)
    vals = flat[nz]
    order = np.argsort(vals, kind='stable')
    vals, nz = vals[order], nz[order]
    starts = np.flatnonzero(np.r_[True, vals[1:] != vals[:-1]])
    w = a.shape[1]
    xs, ys = nz % w, nz // w
    x0 = np.minimum.reduceat(xs, starts)
    x1 = np.maximum.reduceat(xs, starts) + 1
    y0 = np.minimum.reduceat(ys, starts)
    y1 = np.maximum.reduceat(ys, starts) + 1
    counts = np.diff(np.r_[starts, len(vals)])
    return vals[starts].astype(np.int64), np.column_stack([x0, y0, x1, y1]).astype(float), counts.astype(float)


def view_gt(g: dict, inst, view: View, st: DetectionSettings) -> dict:
    """The frame's objects as one view sees them: boxes from the remapped instance image, truncation adding the share
    of the object's fisheye pixels outside the view; objects the view does not see are left out."""
    vinst = remap(inst, view, nearest=True)
    ids, boxes, counts = boxes_of_ids(vinst)
    flat = np.asarray(inst).ravel()
    tot = np.bincount(flat, minlength=int(flat.max()) + 1 if flat.size else 1)
    ins = np.bincount(flat[view.inside.ravel()], minlength=len(tot))
    pos = {int(i): k for k, i in enumerate(g['id'])}
    rows = [(pos[int(i)], k) for k, i in enumerate(ids) if int(i) in pos]
    sel = np.array([r[0] for r in rows], int)
    vk = np.array([r[1] for r in rows], int)
    out = {k: (v[sel] if isinstance(v, np.ndarray) else [v[i] for i in sel]) for k, v in g.items()}
    if len(sel):
        out['box'] = boxes[vk]
        out['height'] = out['box'][:, 3] - out['box'][:, 1]
        out['pixels'] = counts[vk]
        oid = g['id'][sel]
        frac = np.where(tot[oid] > 0, ins[oid] / np.maximum(tot[oid], 1), 0.0)
        out['trunc'] = 1.0 - (1.0 - g['trunc'][sel]) * frac
        ig = np.where(g['ignore'][sel] == IGN_SIGN, IGN_SIGN, IGN_NONE).astype(np.int8)
        out['ignore'] = ig
        _ignore(out, st)
    else:
        out['height'] = np.zeros(0)
    return out


# =========================================================================== evaluating one image
def _det_range(boxes, rng) -> np.ndarray:
    """Range of each detection from the range image: the median of the valid ranges over the box's central half."""
    out = np.full(len(boxes), np.nan)
    if rng is None:
        return out
    h, w = rng.shape[:2]
    for i, b in enumerate(boxes):
        x0, y0, x1, y1 = b
        bw, bh = x1 - x0, y1 - y0
        xa, xb = int(max(math.floor(x0 + bw / 4), 0)), int(min(math.ceil(x1 - bw / 4), w))
        ya, yb = int(max(math.floor(y0 + bh / 4), 0)), int(min(math.ceil(y1 - bh / 4), h))
        if xb <= xa or yb <= ya:
            continue
        r = rng[ya:yb, xa:xb]
        r = r[r > 0]
        if r.size:
            out[i] = float(np.median(r))
    return out


def evaluate_image(g: dict, dets: dict, rng, st: DetectionSettings, classes) -> dict:
    """Match one image's detections to its objects, per evaluated class and group. Returns {'gt': rows, 'det': rows}
    with, per object and class: (class, index into g, matched score); per detection: (class, index into dets, score,
    status, range, status at each AP_IOUS)."""
    gt_rows, det_rows = [], []
    gcls = g['cls']
    dcls = np.asarray(dets['cls'], int)
    dbox = np.asarray(dets['boxes'], float).reshape(-1, 4)
    dsc = np.asarray(dets['scores'], float)
    dh = dbox[:, 3] - dbox[:, 1]
    drange = None
    thr = np.where(g['height'] < st.small_px, st.iou_small, st.iou) if len(gcls) else np.zeros(0)
    for ec in classes:
        members = DT.GROUPS.get(ec, (ec,))
        cidx = [DT.CLASSES.index(m) for m in members]
        gi = np.flatnonzero(np.isin(gcls, cidx))
        di = np.flatnonzero(np.isin(dcls, cidx))
        if ec in DT.GROUPS and len(di) > 1:
            di = di[nms_agnostic(dbox[di], dsc[di], st.nms_iou)]
        if not len(gi) and not len(di):
            continue
        ign = g['ignore'][gi] != IGN_NONE
        st_, dg, gs = match(g['box'][gi], ign, thr[gi], dbox[di], dsc[di], dh[di] < st.min_px)
        e = DT.EVAL_CLASSES.index(ec)
        for k, i in enumerate(gi):
            gt_rows.append((e, int(i), float(gs[k])))
        if len(di):
            if drange is None:
                drange = _det_range(dbox, rng)
            ap_st = np.zeros((len(di), len(AP_IOUS)), np.int8)
            for a, iou_t in enumerate(AP_IOUS):
                ap_st[:, a] = match(g['box'][gi], ign, iou_t, dbox[di], dsc[di], dh[di] < st.min_px)[0]
            for k, i in enumerate(di):
                r = g['range'][gi[dg[k]]] if dg[k] >= 0 else drange[i]
                det_rows.append((e, int(i), float(dsc[i]), int(st_[k]), float(r), ap_st[k]))
    return {'gt': gt_rows, 'det': det_rows}


# =========================================================================== the detection cache
class _Cache:
    """A camera's detections in a recording (detections/<model>-<settings hash>-<CHANNEL>-<view>.npz), by image path;
    entries are reused only for the same image (size and modification time) and the same detector settings."""

    def __init__(self, rec_path, channel, view_key, det_key, enabled=True):
        h = hashlib.sha1(det_key.encode()).hexdigest()[:12]
        self.path = os.path.join(rec_path, CACHE_DIR, f'{h}-{channel}-{view_key}.npz')
        self.enabled = enabled
        self.data = {}
        self.dirty = False
        self.det_key = det_key
        if enabled and os.path.isfile(self.path):
            try:
                with np.load(self.path, allow_pickle=False) as z:
                    meta = json.loads(str(z['meta']))
                    if meta.get('schema') == CACHE_SCHEMA and meta.get('key') == det_key:
                        b, s, c, k = z['boxes'], z['scores'], z['coco'], z['cls']
                        off = np.r_[0, np.cumsum(z['counts'])]
                        for i, (name, stamp) in enumerate(zip(meta['images'], meta['stamps'])):
                            sl = slice(off[i], off[i + 1])
                            self.data[name] = (stamp, {'boxes': b[sl], 'scores': s[sl], 'coco': c[sl], 'cls': k[sl],
                                                       'ms': float(z['ms'][i])})
            except (OSError, ValueError, KeyError):
                self.data = {}

    @staticmethod
    def stamp(path):
        try:
            s = os.stat(path)
            return f'{s.st_size}:{s.st_mtime_ns}'
        except OSError:
            return ''

    def get(self, rel, path):
        e = self.data.get(rel)
        return e[1] if e is not None and e[0] == self.stamp(path) else None

    def put(self, rel, path, dets):
        self.data[rel] = (self.stamp(path), dets)
        self.dirty = True

    def save(self) -> bool:
        if not (self.enabled and self.dirty and self.data):
            return False
        names = sorted(self.data)
        ent = [self.data[n][1] for n in names]
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            tmp = self.path + '.part.npz'
            np.savez_compressed(
                tmp, meta=json.dumps({'schema': CACHE_SCHEMA, 'key': self.det_key, 'images': names,
                                      'stamps': [self.data[n][0] for n in names]}),
                counts=np.array([len(e['scores']) for e in ent], np.int64),
                boxes=np.concatenate([np.asarray(e['boxes'], np.float32).reshape(-1, 4) for e in ent]),
                scores=np.concatenate([np.asarray(e['scores'], np.float32) for e in ent]),
                coco=np.concatenate([np.asarray(e['coco'], np.int16) for e in ent]),
                cls=np.concatenate([np.asarray(e['cls'], np.int8) for e in ent]),
                ms=np.array([float(e.get('ms', 0.0)) for e in ent]))
            os.replace(tmp, self.path)
            self.dirty = False
            return True
        except OSError:
            return False


# =========================================================================== the result
@dataclass
class DetectionEvalResult:
    settings: DetectionSettings
    recordings: list                     # [Recording.info()]
    detector: dict
    rows: list                           # per recording, mode, camera ('all' too), class, range bin ('all' too)
    ranges: list                         # detection range per recording, mode, camera, class
    heights: list                        # recall per object height, per recording, mode, class
    rig: list                            # rig-level object recall per recording, class, distance bin
    views: list                          # fisheye against pinhole views, per recording, class, range bin
    comparisons: list                    # each recording against the first
    frames: dict                         # (rec, mode, camera) -> [frame overlay records]
    timing: dict
    notes: list
    seconds: float = 0.0
    variant: str = ''

    def row(self, rec=0, mode='fisheye', camera='all', cls='vehicle', rbin='all'):
        return next((r for r in self.rows if r['rec'] == rec and r['mode'] == mode and r['camera'] == camera and
                     r['class'] == cls and r['range_bin'] == rbin), None)

    def modes(self) -> list:
        return {'fisheye': ['fisheye'], 'pinhole': ['pinhole'], 'both': ['fisheye', 'pinhole']}[self.settings.views]


class _Cancelled(Exception):
    pass


# =========================================================================== bootstrap helpers
def boot_weights(nblocks: int, B: int, seed: int) -> np.ndarray:
    """(B, nblocks) multinomial counts: blocks drawn with replacement, nblocks per resample."""
    if B <= 0 or nblocks <= 0:
        return np.zeros((0, max(nblocks, 0)))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, nblocks, (B, nblocks))
    W = np.zeros((B, nblocks))
    np.add.at(W, (np.repeat(np.arange(B), nblocks), idx.ravel()), 1.0)
    return W


def _pct(samples, conf) -> list | None:
    s = np.asarray(samples, float)
    s = s[np.isfinite(s)]
    if len(s) < 10:
        return None
    a = (1 - conf) / 2 * 100
    return [float(np.percentile(s, a)), float(np.percentile(s, 100 - a))]


def _ratio(num_b, den_b, W):
    """Value and resampled values of sum(num) / sum(den) with per-block counts."""
    num_b, den_b = np.asarray(num_b, float), np.asarray(den_b, float)
    d = den_b.sum()
    val = float(num_b.sum() / d) if d > 0 else None
    if not len(W):
        return val, np.zeros(0)
    with np.errstate(invalid='ignore', divide='ignore'):
        return val, (W @ num_b) / (W @ den_b)


def _per_block(blocks, values, nblocks):
    return np.bincount(blocks, weights=np.asarray(values, float), minlength=nblocks)[:nblocks]


def _h50(edges, recall, n):
    """The height (px) where recall first reaches 0.5, interpolated in log height between bin centres."""
    e = [float(x) for x in edges]
    lo = [e[0] / 2] + e
    hi = e + [e[-1] * 2]
    centres = [math.sqrt(max(a, 1e-3) * b) for a, b in zip(lo, hi)]
    pts = [(math.log(c), r) for c, r, k in zip(centres, recall, n) if r is not None and k > 0]
    for (x0, r0), (x1, r1) in zip(pts, pts[1:]):
        if r0 < 0.5 <= r1:
            return float(math.exp(x0 + (0.5 - r0) / (r1 - r0) * (x1 - x0)))
    if pts and pts[0][1] >= 0.5:
        return float(math.exp(pts[0][0]))
    return None


# =========================================================================== run
def _frames_of(rec: Recording, st: DetectionSettings):
    cams = [c for t, c in sorted(rec.cameras.items()) if not st.cameras or t in st.cameras]
    return [(c, c.frames[::max(int(st.frame_step), 1)]) for c in cams]


def _read_frame(rec, cam, fr, need_inst):
    import cv2
    from . import groundtruth as GT
    img = cv2.imread(GT.path_of(rec.gt, fr['image']), cv2.IMREAD_COLOR)
    if img is None:
        raise OSError(f'cannot read {fr["image"]}')
    try:
        rng = GT.load_range(rec.gt, fr) if fr.get('range') else None
    except OSError:
        rng = None
    inst = None
    if need_inst and fr.get('instance'):
        try:
            inst = GT.load_instance(rec.gt, fr)
        except OSError:
            inst = None
    return img, rng, inst


def run(project=None, engine=None, settings: DetectionSettings | None = None, variant=None, progress=None,
        cancel=None, detector=None) -> DetectionEvalResult | None:
    """Evaluate the settings' recordings (None when cancelled). detector: a callable image -> detections (tests, or a
    detector already loaded); by default the settings' model (FileNotFoundError when it is not downloaded)."""
    t0 = time.perf_counter()
    st = settings or settings_of(project)
    if not st.recordings:
        raise ValueError('Choose a recording with ground truth (a CARLA run with ground truth on).')
    stop = cancel or (lambda: False)

    def prog(f, text=''):
        if progress is not None:
            try:
                progress(min(max(float(f), 0.0), 1.0), text)
            except TypeError:
                progress(min(max(float(f), 0.0), 1.0))
    recs = [open_recording(p) for p in st.recordings]
    labels = [r.label for r in recs]
    for r in recs:
        if labels.count(r.label) > 1:
            r.label = f'{r.label} ({os.path.basename(r.path)})'
    if detector is None:
        det = DT.load(st.detector())
        det_fn, det_info = det.detect, det.describe()
    else:
        det_fn = detector
        det_info = dict(getattr(detector, 'describe', lambda: {})() or {'model': 'custom', 'name': 'custom detector'})
    det_key = st.detector().key(det_info.get('sha256', det_info.get('model', '')))
    modes = {'fisheye': ['fisheye'], 'pinhole': ['pinhole'], 'both': ['fisheye', 'pinhole']}[st.views]
    classes = [c for c in DT.EVAL_CLASSES if c in st.classes or (c in DT.GROUPS and
                                                                 any(m in st.classes for m in DT.GROUPS[c]))]
    if not classes:
        raise ValueError('No class chosen.')
    notes = []
    work = [(ri, c, frames) for ri, rec in enumerate(recs) for c, frames in _frames_of(rec, st)]
    n_img = sum(len(f) for _, _, f in work) or 1
    done = 0
    G, D, VO, frames_out = [], [], [], {}
    times = {'fisheye': [], 'pinhole': []}
    hits = {'fisheye': 0, 'pinhole': 0}               # detections taken from the cache
    t_det = 0.0
    from . import groundtruth as GT
    pool = cf.ThreadPoolExecutor(2)
    try:
        for ri, cam, frames in work:
            rec = recs[ri]
            if not frames:
                notes.append(f'{rec.label}: {cam.tag} has no frames.')
                continue
            views = []
            if 'pinhole' in modes:
                views = make_views(cam, st)
                if not views:
                    notes.append(f'{rec.label}: {cam.tag} has no lens calibration in the recording '
                                 '(nuscenes calibrated_sensor): no pinhole views.')
            caches = {'fisheye': _Cache(rec.path, cam.channel, 'fisheye', det_key, st.cache)}
            for k, vw in enumerate(views):
                vkey = 'view-' + hashlib.sha1(json.dumps([vw.yaw, st.view_fov_deg, st.view_pitch_deg, st.view_width,
                                                          st.view_aspect]).encode()).hexdigest()[:8]
                caches[k] = _Cache(rec.path, cam.channel, vkey, det_key, st.cache)
            need_inst = bool(views)
            futs = [pool.submit(_read_frame, rec, cam, fr, need_inst) for fr in frames[:3]]
            for fi, fr in enumerate(frames):
                if stop():
                    raise _Cancelled()
                if fi + 3 < len(frames):
                    futs.append(pool.submit(_read_frame, rec, cam, frames[fi + 3], need_inst))
                try:
                    img, rng, inst = futs[fi].result()
                except OSError as e:
                    notes.append(f'{rec.label}: {cam.tag}: {e}')
                    done += 1
                    continue
                futs[fi] = None
                g = gt_objects(fr, st)
                t_us = int(fr.get('timestamp') or 0)
                rel = fr['image']
                path = GT.path_of(rec.gt, rel)
                ev_fe = None
                if 'fisheye' in modes:
                    dets = caches['fisheye'].get(rel, path)
                    if dets is None:
                        t1 = time.perf_counter()
                        dets = det_fn(img)
                        t_det += time.perf_counter() - t1
                        dets = {k: dets[k] for k in ('boxes', 'scores', 'coco', 'cls', 'ms') if k in dets}
                        caches['fisheye'].put(rel, path, dets)
                    else:
                        hits['fisheye'] += 1
                    times['fisheye'].append(float(dets.get('ms', 0.0)))
                    ev = ev_fe = evaluate_image(g, dets, rng, st, classes)
                    _collect(G, D, ev, g, ri, cam.tag, -1, fi, t_us, 'fisheye')
                    _overlay(frames_out, (ri, 'fisheye', cam.tag), fi, rel, t_us, g, dets, ev, -1)
                vbest = {}
                for k, vw in enumerate(views):
                    vimg = remap(img, vw)
                    vrng = remap(rng, vw, nearest=True) if rng is not None else None
                    gv = view_gt(g, inst, vw, st) if inst is not None else gt_objects({'objects': []}, st)
                    vrel = f'{rel}#{vw.name}'
                    dets = caches[k].get(rel, path)
                    if dets is None:
                        t1 = time.perf_counter()
                        dets = det_fn(vimg)
                        t_det += time.perf_counter() - t1
                        dets = {kk: dets[kk] for kk in ('boxes', 'scores', 'coco', 'cls', 'ms') if kk in dets}
                        caches[k].put(rel, path, dets)
                    else:
                        hits['pinhole'] += 1
                    times['pinhole'].append(float(dets.get('ms', 0.0)))
                    ev = evaluate_image(gv, dets, vrng, st, classes)
                    _collect(G, D, ev, gv, ri, cam.tag, k, fi, t_us, 'pinhole')
                    _overlay(frames_out, (ri, 'pinhole', cam.tag), fi, vrel, t_us, gv, dets, ev, k)
                    for e, i, s in ev['gt']:
                        key = (e, int(gv['id'][i]))
                        vbest[key] = max(vbest.get(key, -np.inf), s)
                if views and ev_fe is not None:
                    for e, i, s in ev_fe['gt']:
                        if g['ignore'][i] != IGN_NONE:
                            continue
                        key = (e, int(g['id'][i]))
                        VO.append({'rec': ri, 'cam': cam.tag, 'frame': fi, 't': t_us, 'cls': e,
                                   'range': float(g['range'][i]), 'fe': s, 'pv': vbest.get(key, -np.inf),
                                   'in_view': key in vbest})
                done += 1
                if done % 2 == 0 or done == n_img:
                    prog(0.92 * done / n_img, f'{rec.label} · {cam.tag} · frame {fi + 1} of {len(frames)}')
            for c in caches.values():
                c.save()
    except _Cancelled:
        return None
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    prog(0.93, 'Statistics')
    res = _analyse(st, recs, classes, modes, G, D, VO, frames_out, det_info, notes, stop)
    if res is None:
        return None
    for m in ('fisheye', 'pinhole'):
        t = np.asarray(times[m], float)
        if len(t):
            res.timing[m] = {'images': int(len(t)), 'cached': hits[m], 'ms_mean': round(float(t.mean()), 2),
                             'ms_median': round(float(np.median(t)), 2),
                             'ms_p95': round(float(np.percentile(t, 95)), 2)}
    res.timing['detector_s'] = round(t_det, 2)
    res.seconds = time.perf_counter() - t0
    res.timing['wall_s'] = round(res.seconds, 2)
    prog(1.0, 'Done')
    return res


def _collect(G, D, ev, g, ri, cam, view, fi, t_us, mode):
    for e, i, s in ev['gt']:
        G.append({'rec': ri, 'mode': mode, 'cam': cam, 'view': view, 'frame': fi, 't': t_us, 'cls': e,
                  'id': int(g['id'][i]), 'key': g['key'][i], 'range': float(g['range'][i]),
                  'dist': float(g['dist'][i]), 'x': float(g['xy'][i, 0]), 'y': float(g['xy'][i, 1]),
                  'height': float(g['height'][i]), 'ignore': int(g['ignore'][i]), 'score': s})
    for e, i, s, status, r, aps in ev['det']:
        D.append({'rec': ri, 'mode': mode, 'cam': cam, 'view': view, 'frame': fi, 't': t_us, 'cls': e, 'score': s,
                  'status': status, 'range': r, 'ap': aps})


def _overlay(out, key, fi, rel, t_us, g, dets, ev, view):
    """One frame's boxes for the page's overlay (the classes only, not the groups)."""
    ng = len(g['cls'])
    gt = np.zeros((ng, 7))
    if ng:
        gt[:, :4] = g['box']
        gt[:, 4] = g['cls']
        gt[:, 5] = g['ignore']
        gt[:, 6] = -np.inf
    nd = len(dets['scores'])
    det = np.zeros((nd, 8))
    if nd:
        det[:, :4] = np.asarray(dets['boxes'], float).reshape(-1, 4)
        det[:, 4] = dets['cls']
        det[:, 5] = dets['scores']
        det[:, 6] = -2                                  # not evaluated (a class not chosen)
        det[:, 7] = dets['coco']
    for e, i, s in ev['gt']:
        if DT.EVAL_CLASSES[e] not in DT.GROUPS:
            gt[i, 6] = s
    for e, i, s, status, r, aps in ev['det']:
        if DT.EVAL_CLASSES[e] not in DT.GROUPS:
            det[i, 6] = status
    out.setdefault(key, []).append({'frame': fi, 'image': rel, 't': t_us, 'view': view, 'gt': gt, 'det': det})


# =========================================================================== statistics
def _arrays(rows, keys):
    return {k: np.array([r[k] for r in rows]) if rows else np.zeros(0) for k in keys}


def _analyse(st, recs, classes, modes, G, D, VO, frames_out, det_info, notes, stop) -> DetectionEvalResult | None:
    conf = st.confidence
    redges, hedges = list(st.range_edges), list(st.height_edges)
    rlabels, hlabels = bin_labels(redges), height_labels(hedges)
    nb = len(rlabels)
    block_us = st.block_s * 1e6
    g = _arrays(G, ('rec', 'mode', 'cam', 'view', 'frame', 't', 'cls', 'id', 'key', 'range', 'dist', 'x', 'y',
                    'height', 'ignore', 'score'))
    d = _arrays(D, ('rec', 'mode', 'cam', 'view', 'frame', 't', 'cls', 'score', 'status', 'range'))
    d_ap = np.array([r['ap'] for r in D], np.int8).reshape(-1, len(AP_IOUS))

    # time blocks counted from each recording's first frame (recordings of one scene start together: their blocks
    # are the same stretches of the drive)
    times = {}
    for r in G + D:
        times.setdefault(r['rec'], set()).add(r['t'])
    t_rec = {ri: np.array(sorted(ts), float) for ri, ts in times.items()}

    def blocks_of(t, recs_of):
        t = np.asarray(t, float)
        out = np.zeros(len(t), np.int64)
        for ri in np.unique(recs_of):
            m = np.asarray(recs_of) == ri
            tu = t_rec.get(int(ri), np.zeros(1))
            if block_us > 0:
                out[m] = np.floor((t[m] - tu[0]) / block_us).astype(np.int64)
            else:
                out[m] = np.searchsorted(tu, t[m])          # every frame time its own block
        return out
    g_blk = blocks_of(g['t'], g['rec']) if len(G) else np.zeros(0, np.int64)
    d_blk = blocks_of(d['t'], d['rec']) if len(D) else np.zeros(0, np.int64)
    nblocks = int(max(g_blk.max(initial=-1), d_blk.max(initial=-1)) + 1)
    B = int(st.bootstrap)
    W = boot_weights(nblocks, B, st.seed)
    Ws = [W] + [boot_weights(nblocks, B, st.seed + 1 + k) for k in range(len(recs) - 1)]   # unpaired draws
    g_bin = bin_of(g['range'], redges) if len(G) else np.zeros(0, np.int16)
    d_bin = bin_of(d['range'], redges) if len(D) else np.zeros(0, np.int16)
    g_hbin = bin_of(g['height'], hedges) if len(G) else np.zeros(0, np.int16)
    g_keep = (g['ignore'] == IGN_NONE) if len(G) else np.zeros(0, bool)
    g_det = (g['score'] >= st.score) if len(G) else np.zeros(0, bool)
    d_valid = (d['status'] >= 0) if len(D) else np.zeros(0, bool)
    d_op = d_valid & (d['score'] >= st.score) if len(D) else np.zeros(0, bool)

    def cell(gsel, dsel, Wr, ap=True):
        """Figures of one cell (objects gsel, detections dsel)."""
        ngt = int(np.count_nonzero(gsel & g_keep))
        nd = int(np.count_nonzero(dsel & d_op))
        tp = int(np.count_nonzero(dsel & d_op & (d['status'] == 1))) if len(D) else 0
        det_n = int(np.count_nonzero(gsel & g_keep & g_det))
        out = {'n_gt': ngt, 'detected': det_n, 'n_det': nd, 'tp': tp, 'fp': nd - tp}
        gs = gsel & g_keep
        rc, rcb = _ratio(_per_block(g_blk[gs], g_det[gs], nblocks), _per_block(g_blk[gs], np.ones(gs.sum()), nblocks),
                         Wr)
        ds = dsel & d_op
        pr, prb = _ratio(_per_block(d_blk[ds], d['status'][ds] == 1, nblocks), _per_block(d_blk[ds], np.ones(ds.sum()),
                                                                                          nblocks), Wr)
        out.update(recall=rc, recall_ci=_pct(rcb, conf) if rc is not None else None, precision=pr,
                   precision_ci=_pct(prb, conf) if pr is not None else None)
        out['_rcb'], out['_prb'] = rcb, prb
        if ap:
            dv = dsel & d_valid
            idx = np.flatnonzero(dv)
            idx = idx[np.argsort(-d['score'][idx], kind='mergesort')]
            tp_s = (d['status'][idx] == 1).astype(float)
            ngt_b = _per_block(g_blk[gs], np.ones(gs.sum()), nblocks)
            val = _ap_weighted(tp_s, np.ones((1, len(idx))), [ngt_b.sum()])[0]
            out['ap50'] = None if not np.isfinite(val) else float(val)
            if len(Wr) and out['ap50'] is not None:
                wd = Wr[:, d_blk[idx]] if len(idx) else np.zeros((len(Wr), 0))
                apb = _ap_weighted(tp_s, wd, Wr @ ngt_b)
                out['ap50_ci'] = _pct(apb, conf)
                out['_apb'] = apb
            else:
                out['ap50_ci'] = None
                out['_apb'] = np.zeros(0)
        return out

    rows, ranges, heights, rigs, views, comps = [], [], [], [], [], []
    cache = {}
    for ri, rec in enumerate(recs):
        if stop():
            return None
        Wr = Ws[0] if ri == 0 else (W if recs[0].same_scene(rec) else Ws[ri])
        for mode in modes:
            gm = (g['rec'] == ri) & (g['mode'] == mode) if len(G) else np.zeros(0, bool)
            dm = (d['rec'] == ri) & (d['mode'] == mode) if len(D) else np.zeros(0, bool)
            cams = ['all'] + sorted(c.tag for c, _f in _frames_of(rec, st))
            for cam in cams:
                gc = gm & (g['cam'] == cam) if cam != 'all' else gm
                dc = dm & (d['cam'] == cam) if cam != 'all' else dm
                for ec in classes:
                    e = DT.EVAL_CLASSES.index(ec)
                    gk = gc & (g['cls'] == e) if len(G) else gc
                    dk = dc & (d['cls'] == e) if len(D) else dc
                    per_bin = []
                    for b in [-1] + list(range(nb)):
                        gb = gk & (g_bin == b) if b >= 0 else gk
                        db = dk & (d_bin == b) if b >= 0 else dk
                        c = cell(gb, db, Wr)
                        row = {'rec': ri, 'recording': rec.label, 'mode': mode, 'camera': cam, 'class': ec,
                               'range_bin': rlabels[b] if b >= 0 else 'all', **{k: v for k, v in c.items()
                                                                             if not k.startswith('_')}}
                        if b < 0:
                            row['ap'] = _ap_iou_avg(d, d_ap, dk, gk & g_keep)
                        rows.append(row)
                        cache[(ri, mode, cam, ec, b)] = c
                        if b >= 0:
                            per_bin.append((b, c))
                    good = [(b, c) for b, c in per_bin if c['n_gt'] >= st.min_gt and c['recall'] is not None]
                    reach = [(b, c) for b, c in good if c['recall'] >= 0.5]
                    far = max(reach, key=lambda x: x[0]) if reach else None
                    ranges.append({'rec': ri, 'recording': rec.label, 'mode': mode, 'camera': cam, 'class': ec,
                                   'range_bin': rlabels[far[0]] if far else None,
                                   'range_m': (redges[far[0]] if far[0] < len(redges) else None) if far else None,
                                   'beyond_last_edge': bool(far and far[0] == nb - 1),
                                   'recall': far[1]['recall'] if far else None,
                                   'recall_ci': far[1]['recall_ci'] if far else None,
                                   'bins_with_data': len(good),
                                   'text': _range_text(far, rlabels, redges, good, st.min_gt)})
            # recall against the object's height (every camera; occluded and truncated objects left out)
            for ec in classes:
                e = DT.EVAL_CLASSES.index(ec)
                gk = gm & (g['cls'] == e) & ~np.isin(g['ignore'], (IGN_VIS, IGN_SIGN)) if len(G) else gm
                hb = []
                for b in range(len(hlabels)):
                    s_ = gk & (g_hbin == b)
                    val, vb = _ratio(_per_block(g_blk[s_], g_det[s_], nblocks),
                                     _per_block(g_blk[s_], np.ones(s_.sum()), nblocks), Wr)
                    hb.append({'bin': hlabels[b], 'n': int(s_.sum()), 'detected': int((s_ & g_det).sum()),
                               'recall': val, 'recall_ci': _pct(vb, conf) if val is not None else None})
                heights.append({'rec': ri, 'recording': rec.label, 'mode': mode, 'class': ec, 'bins': hb,
                                'h50_px': _h50(hedges, [x['recall'] for x in hb], [x['n'] for x in hb])})
        # the rig: an object at a time slot once, detected by any camera (fisheye images)
        rig_rows = _rig_objects(g, ri, 'fisheye' if 'fisheye' in modes else 'pinhole', g_blk, st.score, redges)
        for ec in classes:
            e = DT.EVAL_CLASSES.index(ec)
            for b in [-1] + list(range(nb)):
                sel = [o for o in rig_rows.values() if o['cls'] == e and o['seen'] and
                       (b < 0 or o['bin'] == b)]
                blk = np.array([o['blk'] for o in sel], np.int64)
                hit = np.array([o['det'] for o in sel], float)
                val, vb = _ratio(_per_block(blk, hit, nblocks), _per_block(blk, np.ones(len(sel)), nblocks), Wr)
                rigs.append({'rec': ri, 'recording': rec.label, 'class': ec,
                             'distance_bin': rlabels[b] if b >= 0 else 'all', 'n': len(sel),
                             'detected': int(hit.sum()), 'recall': val,
                             'recall_ci': _pct(vb, conf) if val is not None else None})
        # fisheye against pinhole views, object by object
        if VO:
            vo = [r for r in VO if r['rec'] == ri]
            for ec in classes:
                e = DT.EVAL_CLASSES.index(ec)
                for b in [-1] + list(range(nb)):
                    sel = [r for r in vo if r['cls'] == e and (b < 0 or bin_of([r['range']], redges)[0] == b)]
                    blk = blocks_of([r['t'] for r in sel], [ri] * len(sel)) if sel else np.zeros(0, np.int64)
                    fe = np.array([r['fe'] >= st.score for r in sel], float)
                    pv = np.array([r['pv'] >= st.score for r in sel], float)
                    one = np.ones(len(sel))
                    a, ab = _ratio(_per_block(blk, fe, nblocks), _per_block(blk, one, nblocks), Wr)
                    p_, pb = _ratio(_per_block(blk, pv, nblocks), _per_block(blk, one, nblocks), Wr)
                    diff = (p_ - a) if a is not None and p_ is not None else None
                    views.append({'rec': ri, 'recording': rec.label, 'class': ec,
                                  'range_bin': rlabels[b] if b >= 0 else 'all', 'n': len(sel),
                                  'in_a_view': int(sum(r['in_view'] for r in sel)),
                                  'recall_fisheye': a, 'recall_fisheye_ci': _pct(ab, conf) if a is not None else None,
                                  'recall_views': p_, 'recall_views_ci': _pct(pb, conf) if p_ is not None else None,
                                  'difference': diff,
                                  'difference_ci': _pct(pb - ab, conf) if diff is not None else None})
    # every recording against the first
    for ri in range(1, len(recs)):
        paired = recs[0].same_scene(recs[ri])
        for mode in modes:
            for ec in classes:
                for b in [-1] + list(range(nb)):
                    a, c = cache.get((0, mode, 'all', ec, b)), cache.get((ri, mode, 'all', ec, b))
                    if a is None or c is None:
                        continue
                    row = {'reference': recs[0].label, 'recording': recs[ri].label, 'rec': ri, 'mode': mode,
                           'class': ec, 'range_bin': rlabels[b] if b >= 0 else 'all', 'paired': paired,
                           'n_gt_reference': a['n_gt'], 'n_gt': c['n_gt']}
                    for k, kb in (('recall', '_rcb'), ('precision', '_prb'), ('ap50', '_apb')):
                        va, vc = a.get(k), c.get(k)
                        row[f'{k}_reference'], row[k] = va, vc
                        if va is None or vc is None:
                            row[f'{k}_difference'] = row[f'{k}_difference_ci'] = None
                            continue
                        row[f'{k}_difference'] = vc - va
                        sa, sc = a.get(kb, np.zeros(0)), c.get(kb, np.zeros(0))
                        row[f'{k}_difference_ci'] = _pct(sc - sa, conf) if len(sa) and len(sa) == len(sc) else None
                    comps.append(row)
        # the rig on the union of both recordings' objects
        cm = 'fisheye' if 'fisheye' in modes else 'pinhole'
        # both on one time grid: the coarser frame period, from the earlier start (one scene starts together)
        ia = np.flatnonzero((g['rec'] == 0) & (g['mode'] == cm)) if len(G) else np.zeros(0, int)
        ib = np.flatnonzero((g['rec'] == ri) & (g['mode'] == cm)) if len(G) else np.zeros(0, int)
        grid = max(_frame_period(g, ia), _frame_period(g, ib)) if len(ia) and len(ib) else None
        tg = float(min(g['t'][ia].min(), g['t'][ib].min())) if len(ia) and len(ib) else None
        ra = _rig_objects(g, 0, cm, g_blk, st.score, redges, grid, tg)
        rb = _rig_objects(g, ri, cm, g_blk, st.score, redges, grid, tg)
        if ra and rb:                                   # the time both drives cover
            lo = max(min(o['slot'] for o in ra.values()), min(o['slot'] for o in rb.values()))
            hi = min(max(o['slot'] for o in ra.values()), max(o['slot'] for o in rb.values()))
            ra = {k: o for k, o in ra.items() if lo <= o['slot'] <= hi}
            rb = {k: o for k, o in rb.items() if lo <= o['slot'] <= hi}
        Wp = W if paired else None
        union = _union(ra, rb) if paired else []
        for ec in classes:
            e = DT.EVAL_CLASSES.index(ec)
            for b in [-1] + list(range(nb)):
                if paired:
                    pairs = [(oa, ob) for oa, ob in union if (oa or ob)['cls'] == e and
                             ((oa is not None and oa['seen']) or (ob is not None and ob['seen'])) and
                             (b < 0 or min(o['bin'] for o in (oa, ob) if o is not None) == b)]
                    if not pairs:
                        continue
                    blk = np.array([(oa or ob)['blk'] for oa, ob in pairs], np.int64)
                    ha = np.array([float(oa is not None and oa['det']) for oa, ob in pairs])
                    hb_ = np.array([float(ob is not None and ob['det']) for oa, ob in pairs])
                    one = np.ones(len(pairs))
                    n_ref = n_rec = len(pairs)
                else:
                    sa_ = [o for o in ra.values() if o['cls'] == e and o['seen'] and (b < 0 or o['bin'] == b)]
                    sb_ = [o for o in rb.values() if o['cls'] == e and o['seen'] and (b < 0 or o['bin'] == b)]
                    if not sa_ and not sb_:
                        continue
                    n_ref, n_rec = len(sa_), len(sb_)
                if Wp is not None:
                    va, sa = _ratio(_per_block(blk, ha, nblocks), _per_block(blk, one, nblocks), Wp)
                    vb_, sb = _ratio(_per_block(blk, hb_, nblocks), _per_block(blk, one, nblocks), Wp)
                    ci = _pct(sb - sa, conf)
                else:
                    ba = np.array([o['blk'] for o in sa_], np.int64)
                    bb = np.array([o['blk'] for o in sb_], np.int64)
                    va, sa = _ratio(_per_block(ba, [float(o['det']) for o in sa_], nblocks),
                                    _per_block(ba, np.ones(len(sa_)), nblocks), Ws[0])
                    vb_, sb = _ratio(_per_block(bb, [float(o['det']) for o in sb_], nblocks),
                                     _per_block(bb, np.ones(len(sb_)), nblocks), Ws[ri])
                    ci = _pct(sb - sa, conf) if va is not None and vb_ is not None else None
                comps.append({'reference': recs[0].label, 'recording': recs[ri].label, 'rec': ri, 'mode': 'rig',
                              'class': ec, 'range_bin': rlabels[b] if b >= 0 else 'all', 'paired': paired,
                              'n_gt_reference': n_ref, 'n_gt': n_rec, 'recall_reference': va, 'recall': vb_,
                              'recall_difference': (vb_ - va) if va is not None and vb_ is not None else None,
                              'recall_difference_ci': ci,
                              'note': ('objects either rig saw (matched by place and time), by distance from the '
                                       'vehicle; detected by any camera') if paired else
                                      ('each rig\'s own objects, by distance from the vehicle; detected by any '
                                       'camera')})
        if not paired:
            notes.append(f'{recs[ri].label} and {recs[0].label} did not run the same scenario, seed, town and '
                         'duration: their intervals are drawn independently (not paired).')
    timing = {'opencv': det_info.get('opencv'), 'backend': det_info.get('backend')}
    return DetectionEvalResult(st, [r.info() for r in recs], det_info, rows, ranges, heights, rigs, views, comps,
                               frames_out, timing, notes, variant=recs[0].variant if recs else '')


def _ap_iou_avg(d, d_ap, dsel, gsel) -> float | None:
    """COCO's AP averaged over IoU 0.50:0.05:0.95 (no small-object threshold), no interval."""
    ngt = int(np.count_nonzero(gsel))
    if ngt == 0:
        return None
    idx = np.flatnonzero(dsel)
    vals = []
    for a in range(len(AP_IOUS)):
        st_ = d_ap[idx, a]
        ok = st_ >= 0
        ii = idx[ok]
        order = np.argsort(-d['score'][ii], kind='mergesort')
        vals.append(_ap_weighted((st_[ok][order] == 1).astype(float), np.ones((1, len(ii))), [ngt])[0])
    return float(np.mean(vals))


def _frame_period(g, sel) -> float:
    """The cameras' frame period (µs): the median of each camera's median frame interval (cameras on different trigger
    lines fire at different moments, so the drive's frame times together are no period)."""
    per = []
    for cam in np.unique(g['cam'][sel]):
        ts = np.unique(g['t'][sel][g['cam'][sel] == cam])
        if len(ts) > 1:
            per.append(float(np.median(np.diff(ts))))
    return max(float(np.median(per)), 1.0) if per else 1.0


def _rig_objects(g, ri, mode, g_blk, score, edges, slot_us=None, t0=None) -> dict:
    """(class, object key, time slot) -> {'cls', 'seen' (not ignored in some camera), 'det' (detected by some
    camera at the operating point), 'bin' (distance from the vehicle), 'blk', 'slot', 'xy'} for one recording. Time
    slots are slot_us long (default: the cameras' frame period) from t0; in each slot every camera contributes its
    one frame nearest the slot's centre, so a camera with more frames per second gets no extra tries."""
    out = {}
    if not len(g['rec']):
        return out
    sel = np.flatnonzero((g['rec'] == ri) & (g['mode'] == mode))
    if not len(sel):
        return out
    slot_us = slot_us or _frame_period(g, sel)
    t0 = g['t'][sel].min() if t0 is None else t0
    t = g['t'][sel].astype(float)
    slots = np.rint((t - t0) / slot_us).astype(np.int64)
    off = np.abs(t - t0 - slots * slot_us)
    best = {}
    for cam, sl, tt, o in zip(g['cam'][sel], slots, t, off):
        k = (cam, sl)
        if k not in best or o < best[k][1]:
            best[k] = (tt, o)
    for i, sl in zip(sel, slots):
        if g['t'][i] != best[(g['cam'][i], sl)][0]:
            continue
        slot = int(sl)
        k = (int(g['cls'][i]), str(g['key'][i]), slot)
        o = out.get(k)
        if o is None:
            o = out[k] = {'cls': int(g['cls'][i]), 'seen': False, 'det': False, 'dist': float(g['dist'][i]),
                          'blk': int(g_blk[i]), 'bin': -1, 'slot': slot, 'xy': (float(g['x'][i]), float(g['y'][i]))}
        o['seen'] |= bool(g['ignore'][i] == IGN_NONE)
        o['det'] |= bool(g['score'][i] >= score)
        if np.isfinite(g['dist'][i]):
            o['dist'] = min(o['dist'], float(g['dist'][i])) if np.isfinite(o['dist']) else float(g['dist'][i])
    for o in out.values():
        o['bin'] = int(bin_of([o['dist']], edges)[0])
    return out


def _union(ra: dict, rb: dict, tol_m: float = 1.0) -> list:
    """The objects of two recordings of the same scene as pairs (a, b), either None when only one rig saw it: the
    same class at the same time slot with box centres within tol_m (the vehicle frame), nearest first; objects
    without a 3D box pair by their key."""
    out = []
    by = {}
    for o in rb.values():
        by.setdefault((o['cls'], o['slot']), []).append(o)
    used = set()
    for oa in ra.values():
        best, bd = None, tol_m
        for ob in by.get((oa['cls'], oa['slot']), ()):
            if id(ob) in used:
                continue
            dx, dy = oa['xy'][0] - ob['xy'][0], oa['xy'][1] - ob['xy'][1]
            dist = math.hypot(dx, dy) if math.isfinite(dx) and math.isfinite(dy) else math.inf
            if dist <= bd:
                best, bd = ob, dist
        if best is not None:
            used.add(id(best))
        out.append((oa, best))
    out += [(None, ob) for ob in rb.values() if id(ob) not in used]
    return out


def _range_text(far, labels, edges, good, min_gt) -> str:
    if not good:
        return f'Too few objects (fewer than {min_gt} in every range bin).'
    if far is None:
        return 'Recall under 50 % in every range bin.'
    if far[0] == len(labels) - 1:
        return f'Recall ≥ 50 % beyond {edges[-1]:g} m.'
    return f'Recall ≥ 50 % up to {edges[far[0]]:g} m ({labels[far[0]]}).'


# =========================================================================== output
REFERENCES = (
    'Ge Z., Liu S., Wang F., Li Z., Sun J. (2021) YOLOX: Exceeding YOLO Series in 2021. arXiv:2107.08430.',
    'Lin T.-Y. et al. (2014) Microsoft COCO: Common Objects in Context. ECCV; the COCO detection evaluation '
    '(pycocotools COCOeval): greedy matching, ignore regions, 101-point interpolated average precision.',
    'Efron B., Tibshirani R. J. (1993) An Introduction to the Bootstrap. Chapman & Hall (percentile intervals).',
    'Künsch H. R. (1989) The jackknife and the bootstrap for general stationary observations. Annals of Statistics '
    '17(3) (resampling blocks of a time series).',
    'Kannala J., Brandt S. S. (2006) A generic camera model and calibration method for conventional, wide-angle, and '
    'fish-eye lenses. IEEE TPAMI 28(8) (the pinhole views\' lens model, as cv2.fisheye).')


def _r(v, nd=4):
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return round(f, nd) if math.isfinite(f) else None


def _ci(c, nd=4):
    return [_r(c[0], nd), _r(c[1], nd)] if c else None


def _clean_row(r: dict) -> dict:
    out = {}
    for k, v in r.items():
        if k.startswith('_'):
            continue
        if k.endswith('_ci'):
            out[k] = _ci(v)
        elif isinstance(v, float):
            out[k] = _r(v)
        elif isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating,)):
            out[k] = _r(float(v))
        else:
            out[k] = v
    return out


def summary(res: DetectionEvalResult) -> dict:
    """Compact figures (the assistant, the batch log): per class over every camera of each recording, the detection
    ranges, the rig-level recall and the differences against the first recording."""
    m0 = res.modes()[0]
    per = []
    for ri, info in enumerate(res.recordings):
        cls = []
        for r in res.rows:
            if r['rec'] == ri and r['mode'] == m0 and r['camera'] == 'all' and r['range_bin'] == 'all' and r['n_gt']:
                rg = next((x for x in res.ranges if x['rec'] == ri and x['mode'] == m0 and x['camera'] == 'all' and
                           x['class'] == r['class']), None)
                rig = next((x for x in res.rig if x['rec'] == ri and x['class'] == r['class'] and
                            x['distance_bin'] == 'all'), None)
                cls.append({'class': r['class'], 'objects': r['n_gt'], 'recall': _r(r['recall'], 3),
                            'recall_ci': _ci(r['recall_ci'], 3), 'precision': _r(r['precision'], 3),
                            'ap50': _r(r.get('ap50'), 3), 'ap': _r(r.get('ap'), 3),
                            'detection_range': rg['text'] if rg else None,
                            'rig_recall': _r(rig['recall'], 3) if rig else None})
        per.append({'recording': info['label'], 'variant': info['variant'], 'scenario': info['scenario'],
                    'seed': info['seed'], 'frames': info['frames'], 'classes': cls})
    comp = [{'recording': c['recording'], 'mode': c['mode'], 'class': c['class'],
             'recall_difference': _r(c.get('recall_difference'), 3), 'recall_difference_ci':
                 _ci(c.get('recall_difference_ci'), 3), 'paired': c['paired']}
            for c in res.comparisons if c['range_bin'] == 'all' and c.get('recall_difference') is not None]
    views = [{'recording': v['recording'], 'class': v['class'], 'recall_fisheye': _r(v['recall_fisheye'], 3),
              'recall_views': _r(v['recall_views'], 3), 'difference': _r(v['difference'], 3),
              'difference_ci': _ci(v['difference_ci'], 3)} for v in res.views
             if v['range_bin'] == 'all' and v['n']]
    return {'model': res.detector.get('name'), 'operating_score': res.settings.score, 'iou': res.settings.iou,
            'mode': res.settings.views, 'recordings': per, 'comparisons': comp, 'views': views,
            'timing': res.timing, 'notes': list(res.notes)[:10], 'assumptions': ASSUMPTIONS}


def to_doc(res: DetectionEvalResult, project=None) -> dict:
    st = res.settings
    return {
        'schema': SCHEMA, 'project': getattr(project, 'name', ''), 'variant': res.variant,
        'units': ('recall, precision and average precision as shares 0–1; ranges and distances in metres; heights '
                  'in output-image pixels; times in ms'),
        'frame': FRAME_V + ' for distances from the vehicle; ranges are camera to box centre',
        'settings': st.to_dict(), 'detector': res.detector, 'licence': DT.LICENCE_TEXT,
        'range_bins': bin_labels(st.range_edges), 'height_bins': height_labels(st.height_edges),
        'assumptions': ASSUMPTIONS, 'limits': LIMITS_TEXT, 'references': list(REFERENCES),
        'recordings': res.recordings,
        'per_camera': [_clean_row(r) for r in res.rows],
        'detection_ranges': [_clean_row(r) for r in res.ranges],
        'recall_by_height': [dict(_clean_row({k: v for k, v in h.items() if k != 'bins'}),
                                  bins=[_clean_row(b) for b in h['bins']]) for h in res.heights],
        'rig': [_clean_row(r) for r in res.rig],
        'pinhole_views': [_clean_row(r) for r in res.views],
        'comparisons': [_clean_row(r) for r in res.comparisons],
        'timing': res.timing, 'seconds': round(res.seconds, 2), 'notes': list(res.notes)}


ROW_COLS = ('recording', 'mode', 'camera', 'class', 'range_bin', 'n_gt', 'detected', 'recall', 'recall_lo', 'recall_hi',
            'n_det', 'tp', 'fp', 'precision', 'precision_lo', 'precision_hi', 'ap50', 'ap50_lo', 'ap50_hi', 'ap')


def _csv(rows, cols) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator='\n')
    w.writerow(cols)
    for r in rows:
        out = []
        for c in cols:
            if c.endswith('_lo') or c.endswith('_hi'):
                ci = r.get(c[:-3] + '_ci')
                v = (ci[0] if c.endswith('_lo') else ci[1]) if ci else None
            else:
                v = r.get(c)
            v = _r(v) if isinstance(v, (float, np.floating)) else v
            out.append('' if v is None else v)
        w.writerow(out)
    return buf.getvalue()


def to_csv(res: DetectionEvalResult) -> str:
    return _csv(res.rows, ROW_COLS)


def _slug(s) -> str:
    import re
    return re.sub(r'[^a-z0-9]+', '-', str(s).lower()).strip('-')[:60] or 'recording'


# ---------------------------------------------------------------- overlays
_COL = {'hit': (90, 180, 60), 'miss': (40, 110, 240), 'ignored': (150, 150, 150), 'tp': (230, 160, 40),
        'fp': (200, 60, 200)}
_VIEWS: dict = {}


def views_for(path, tag, st: DetectionSettings) -> list:
    """The pinhole views of a recording's camera (cached: they take a moment)."""
    key = (os.path.abspath(path), tag, st.view_fov_deg, st.view_pitch_deg, st.view_width, st.view_aspect,
           tuple(st.view_yaws))
    if key not in _VIEWS:
        rec = open_recording(path)
        _VIEWS[key] = make_views(rec.cameras[tag], st) if tag in rec.cameras else []
        while len(_VIEWS) > 8:
            _VIEWS.pop(next(iter(_VIEWS)))
    return _VIEWS[key]


def frame_image(res: DetectionEvalResult, rec: int, mode: str, cam: str, k: int):
    """The image of overlay record k (BGR), a pinhole view cut again when the record is a view's."""
    import cv2
    fr = res.frames[(rec, mode, cam)][k]
    path = res.recordings[rec]['path']
    img = cv2.imread(os.path.join(path, fr['image'].split('#')[0]), cv2.IMREAD_COLOR)
    if img is None:
        return None
    if fr['view'] >= 0:
        vs = views_for(path, cam, res.settings)
        if fr['view'] >= len(vs):
            return None
        img = remap(img, vs[fr['view']])
    return img


def overlay_image(res: DetectionEvalResult, rec: int, mode: str, cam: str, k: int, img=None):
    """Detections against ground truth on one frame: objects found (green), missed (orange), ignored (grey);
    detections at the operating point that match (blue) or match nothing (magenta)."""
    import cv2
    from .pngtext import put_text
    fr = res.frames[(rec, mode, cam)][k]
    if img is None:
        img = frame_image(res, rec, mode, cam, k)
    if img is None:
        return None
    out = img.copy()
    th = max(1, int(round(out.shape[1] / 700)))
    sc = res.settings.score
    for row in fr['gt']:
        x0, y0, x1, y1, c, ig, s = row
        if c < 0:
            continue
        col = _COL['ignored'] if ig else (_COL['hit'] if s >= sc else _COL['miss'])
        cv2.rectangle(out, (int(x0), int(y0)), (int(math.ceil(x1)) - 1, int(math.ceil(y1)) - 1), col,
                      th if ig else th + 1)
    for row in fr['det']:
        x0, y0, x1, y1, c, s, status, coco = row
        if s < sc or status < 0:
            continue
        col = _COL['tp'] if status == 1 else _COL['fp']
        cv2.rectangle(out, (int(round(x0)), int(round(y0))), (int(round(x1)), int(round(y1))), col, th)
        name = DT.CLASSES[int(c)] if 0 <= c < len(DT.CLASSES) else '?'
        put_text(out, f'{name} {s:.2f}', (int(x0) + 2, max(int(y0) - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX,
                 0.35 * th + 0.05, col, 1, cv2.LINE_AA)
    return out


def best_frame(res: DetectionEvalResult, rec: int, mode: str, cam: str) -> int:
    """The overlay record with the most objects that are not ignored (ties: the first)."""
    frs = res.frames.get((rec, mode, cam)) or []
    if not frs:
        return -1
    n = [int(np.count_nonzero((f['gt'][:, 5] == 0) & (f['gt'][:, 4] >= 0))) if len(f['gt']) else 0 for f in frs]
    return int(np.argmax(n))


# ---------------------------------------------------------------- the chart
_BG, _INK, _INK2, _MUTED, _HAIR = (255, 255, 255), (40, 40, 40), (90, 90, 90), (140, 140, 140), (225, 225, 225)
_SERIES = ((180, 110, 30), (40, 120, 230), (60, 160, 60), (160, 60, 160), (40, 40, 200), (140, 140, 20),
           (100, 100, 100), (30, 170, 200))


def chart_png(res: DetectionEvalResult, mode=None, camera='all') -> bytes:
    """Recall against range per class (one panel each, the classes with objects), one line per recording, with the
    intervals; PNG."""
    import cv2
    from .pngtext import ascii_text
    mode = mode or res.modes()[0]
    labels = bin_labels(res.settings.range_edges)
    cls = [c for c in DT.EVAL_CLASSES if any(r['class'] == c and r['n_gt'] and r['mode'] == mode and
                                             r['camera'] == camera for r in res.rows)]
    cols = 3
    rows_n = max(1, math.ceil(len(cls) / cols))
    pw, ph = 380, 250
    W, H = pw * cols, ph * rows_n + 70
    img = np.full((H, W, 3), _BG, np.uint8)

    def text(t, x, y, col=_INK, s=0.45, right=False, centre=False):
        t = ascii_text(t)
        (tw, _), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, s, 1)
        x = x - tw if right else (x - tw / 2 if centre else x)
        cv2.putText(img, t, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, s, col, 1, cv2.LINE_AA)
    text(f'Recall against range · {MODE_LABEL.get(mode, mode)} · camera {camera} · score >= {res.settings.score:g}, '
         f'IoU >= {res.settings.iou:g}', 12, 22, s=0.5)
    for i, rec in enumerate(res.recordings):
        col = _SERIES[i % len(_SERIES)]
        x = 12 + i * 300
        cv2.line(img, (x, 44), (x + 18, 44), col, 2, cv2.LINE_AA)
        text(rec['label'][:38], x + 24, 48, _INK2, 0.4)
    if not cls:
        text('No labelled objects of the chosen classes.', 12, 100, _MUTED)
    for k, c in enumerate(cls):
        ox, oy = (k % cols) * pw, 70 + (k // cols) * ph
        x0, y0, x1, y1 = ox + 50, oy + 26, ox + pw - 16, oy + ph - 40
        text(DT.CLASS_LABEL.get(c, c), x0, oy + 16, _INK, 0.45)
        for t in (0, 0.25, 0.5, 0.75, 1.0):
            y = y1 - t * (y1 - y0)
            cv2.line(img, (x0, int(y)), (x1, int(y)), _HAIR if t else _INK2, 1)
            text(f'{t * 100:.0f}%', x0 - 6, y + 4, _MUTED, 0.35, right=True)
        n = len(labels)
        X = lambda j: x0 + (j + 0.5) / n * (x1 - x0)          # noqa: E731
        for j, lab in enumerate(labels):
            text(lab, X(j), y1 + 16, _MUTED, 0.33, centre=True)
        for i in range(len(res.recordings)):
            col = _SERIES[i % len(_SERIES)]
            pts = []
            for j, lab in enumerate(labels):
                r = next((r for r in res.rows if r['rec'] == i and r['mode'] == mode and r['camera'] == camera and
                          r['class'] == c and r['range_bin'] == lab), None)
                if r is None or r['recall'] is None:
                    continue
                xx = X(j) + (i - (len(res.recordings) - 1) / 2) * 5
                y = y1 - r['recall'] * (y1 - y0)
                if r['recall_ci']:
                    cv2.line(img, (int(xx), int(y1 - r['recall_ci'][0] * (y1 - y0))),
                             (int(xx), int(y1 - r['recall_ci'][1] * (y1 - y0))), col, 1, cv2.LINE_AA)
                pts.append((int(xx), int(y)))
            for a, b in zip(pts, pts[1:]):
                cv2.line(img, a, b, col, 2, cv2.LINE_AA)
            for p in pts:
                cv2.circle(img, p, 3, col, -1, cv2.LINE_AA)
    ok, buf = cv2.imencode('.png', img)
    return buf.tobytes()


# ---------------------------------------------------------------- files
def _coco_ids(rec_path, channel) -> dict:
    """image file name -> image id of the recording's COCO file for a camera (core.groundtruth writes them)."""
    doc = _load_json(os.path.join(rec_path, 'ground_truth', 'coco', f'{channel}.json')) or {}
    out = {}
    for im in doc.get('images') or []:
        if isinstance(im, dict) and 'file_name' in im:
            out[os.path.basename(str(im['file_name']))] = im.get('id')
    return out


def coco_results(res: DetectionEvalResult, rec: int, cam: str, mode='fisheye') -> list:
    """The camera's detections (every one kept) in COCO's results format; image_id from the recording's COCO file
    (ground_truth/coco/CAM_<tag>.json) when it lists the image, else the image's path."""
    frs = res.frames.get((rec, mode, cam)) or []
    info = res.recordings[rec]
    gt = _gt_doc(info['path']) or {}
    channel = ((gt.get('cameras') or {}).get(cam) or {}).get('channel', f'CAM_{cam}')
    ids = _coco_ids(info['path'], channel)
    out = []
    for f in frs:
        name = os.path.basename(f['image'])
        iid = ids.get(name, f['image'])
        d = f['det']
        dets = {'boxes': d[:, :4], 'scores': d[:, 5], 'coco': d[:, 7].astype(int)} if len(d) else DT.empty()
        out += DT.to_coco_results(dets, iid)
    return out


def files(res: DetectionEvalResult, project=None) -> list:
    """[(name, bytes)]: detection_quality.json and .csv, detection_ranges.csv, detection_rig.csv,
    detection_comparison.csv (two or more recordings), detection_views.csv (pinhole views), detection_quality.png,
    detections/<recording>/<CHANNEL>.json (COCO results) and overlays/<recording>_<CHANNEL>.jpg (the frame with the
    most objects, each camera)."""
    import cv2
    out = [('detection_quality.json', (json.dumps(to_doc(res, project), indent=1) + '\n').encode('utf-8')),
           ('detection_quality.csv', to_csv(res).encode('utf-8')),
           ('detection_ranges.csv', _csv(res.ranges, ('recording', 'mode', 'camera', 'class', 'range_bin', 'range_m',
                                                      'beyond_last_edge', 'recall', 'recall_lo', 'recall_hi',
                                                      'bins_with_data', 'text')).encode('utf-8')),
           ('detection_rig.csv', _csv(res.rig, ('recording', 'class', 'distance_bin', 'n', 'detected', 'recall',
                                                'recall_lo', 'recall_hi')).encode('utf-8'))]
    if res.comparisons:
        out.append(('detection_comparison.csv', _csv(res.comparisons, (
            'reference', 'recording', 'mode', 'class', 'range_bin', 'paired', 'n_gt_reference', 'n_gt',
            'recall_reference', 'recall', 'recall_difference', 'recall_difference_lo', 'recall_difference_hi',
            'precision_reference', 'precision', 'precision_difference', 'precision_difference_lo',
            'precision_difference_hi', 'ap50_reference', 'ap50', 'ap50_difference', 'ap50_difference_lo',
            'ap50_difference_hi')).encode('utf-8')))
    if res.views:
        out.append(('detection_views.csv', _csv(res.views, (
            'recording', 'class', 'range_bin', 'n', 'in_a_view', 'recall_fisheye', 'recall_fisheye_lo',
            'recall_fisheye_hi', 'recall_views', 'recall_views_lo', 'recall_views_hi', 'difference', 'difference_lo',
            'difference_hi')).encode('utf-8')))
    out.append(('detection_quality.png', chart_png(res)))
    for ri, info in enumerate(res.recordings):
        slug = _slug(info['label'])
        for (r, mode, cam) in sorted(k for k in res.frames if k[0] == ri):
            if mode == 'fisheye':
                out.append((f'detections/{slug}/CAM_{cam}.json',
                            json.dumps(coco_results(res, ri, cam)).encode('utf-8')))
            k = best_frame(res, ri, mode, cam)
            img = overlay_image(res, ri, mode, cam, k) if k >= 0 else None
            if img is not None:
                ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 88])
                if ok:
                    out.append((f'overlays/{slug}_{mode}_CAM_{cam}.jpg', buf.tobytes()))
    return out


def export(res: DetectionEvalResult, project, out_dir) -> list[str]:
    paths = []
    for name, data in files(res, project):
        path = os.path.join(out_dir, *name.split('/'))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as fh:
            fh.write(data)
        paths.append(path)
    return paths


# =========================================================================== a variant through CARLA
def evaluate_variant(project, variant, sim_settings=None, det_settings=None, out_dir=None, python=None,
                     progress=None, cancel=None, detector=None, runner=None) -> DetectionEvalResult | None:
    """A design point for an experiment: the variant driven in CARLA with ground truth (core.simrig and the bridge in
    CARLA's Python, on the server already running on sim_settings.port), then evaluated. out_dir holds the drive
    (rig/, nuscenes/, ground_truth/, run.json, bridge.log); a drive already there from the same rig file and a run
    that ended well is reused (a cancelled experiment resumes). Raises RuntimeError when no server answers, CARLA's
    Python is missing or the run fails (run.json says why); None when cancelled. runner(rig_json, out_dir, progress,
    cancel) replaces the bridge subprocess (the tests run the bridge in-process against the fake carla)."""
    import shutil
    import signal
    import subprocess
    from . import simrig as SR
    from ..sim import server as SV
    stop = cancel or (lambda: False)
    sim = sim_settings or SR.settings_of(project)
    d = sim.to_dict()
    if 'ground_truth' in d:
        # the cameras' ground truth on; the lidar's labels are not needed here (they cost a semantic lidar per lidar)
        d['ground_truth'] = dict(d['ground_truth'] or {}, instance=True, range=True, semantic=True, lidar_labels=False)
    sim = SR.SimSettings.from_dict(d)
    dst = det_settings or settings_of(project)

    def prog(f, text=''):
        if progress is not None:
            progress(min(max(float(f), 0.0), 1.0), text)
    if out_dir is None:
        from .. import workdir as WD
        v = variant if variant is not None else project.variant
        out_dir = WD.project_dir(project, None, 'experiments', 'detection', WD.slug(v.name))
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    prog(0.01, 'The rig')
    rig = SR.build(project, sim, variant=variant)
    if rig is None:
        return None
    new_dir = os.path.join(out_dir, '.rig-new')
    SR.export(rig, project, new_dir)
    rig_dir = os.path.join(out_dir, 'rig')
    same = _same_files(os.path.join(new_dir, 'sim_rig.json'), os.path.join(rig_dir, 'sim_rig.json'))
    doc = SR.read_run(out_dir)
    reuse = same and doc is not None and doc.get('status') == 'ok' and _gt_doc(out_dir) is not None
    if not reuse:
        if os.path.isdir(rig_dir):
            shutil.rmtree(rig_dir)
        os.replace(new_dir, rig_dir)
        if os.path.isfile(os.path.join(out_dir, 'run.json')):
            os.remove(os.path.join(out_dir, 'run.json'))
    code = None
    if not reuse and runner is not None:
        runner(os.path.join(rig_dir, 'sim_rig.json'), out_dir, lambda f, t='': prog(0.02 + 0.6 * f, f'CARLA · {t}'),
               stop)
    elif not reuse:
        py = python or _carla_python()
        if not py or not os.path.isfile(py):
            raise RuntimeError('No Python for CARLA\'s client: choose it on Simulate ▸ CARLA.')
        if not SV.port_open('127.0.0.1', sim.port):
            raise RuntimeError(f'No CARLA server answers on port {sim.port}: start one on Simulate ▸ CARLA.')
        cmd, env = SV.bridge_command(py, os.path.join(rig_dir, 'sim_rig.json'), out_dir,
                                     ['--port', str(sim.port), '--tm-port', str(sim.tm_port),
                                      '--timeout', f'{sim.timeout_s:g}'])
        with open(os.path.join(out_dir, 'bridge.log'), 'w', encoding='utf-8') as log:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=log, text=True, env=env, cwd=out_dir)
            asked = False
            try:
                for line in proc.stdout:
                    pr = SV.parse_progress(line)
                    if pr is not None:
                        prog(0.02 + 0.6 * pr[0], f'CARLA · {pr[1]}')
                    else:
                        log.write(line)
                    if stop() and not asked:
                        proc.send_signal(signal.SIGINT)        # the bridge tidies CARLA up and removes its output
                        asked = True
                proc.wait()
            finally:
                if proc.poll() is None:
                    proc.send_signal(signal.SIGINT)
                    try:
                        proc.wait(60)
                    except subprocess.TimeoutExpired:
                        proc.kill()
            code = proc.returncode
    if not reuse:
        if stop():
            return None
        doc = SR.read_run(out_dir)
        if doc is None or doc.get('status') != 'ok':
            why = (doc or {}).get('error') or (f'the bridge exited with {code}; see bridge.log' if code is not None
                                               else 'it wrote no report')
            raise RuntimeError(f'The CARLA run failed: {str(why).splitlines()[0]}')
    else:
        shutil.rmtree(new_dir, ignore_errors=True)
    st = DetectionSettings.from_dict(dict(dst.to_dict(), recordings=[out_dir]))

    def sub(f, text=''):
        prog(0.62 + 0.38 * f, text)
    return run(project, settings=st, progress=sub, cancel=cancel, detector=detector)


def _same_files(a, b) -> bool:
    """Two sim_rig.json files describe the same run (everything but when they were written), and their remap tables
    are the same bytes."""
    da, db = _load_json(a), _load_json(b)
    if not isinstance(da, dict) or not isinstance(db, dict):
        return False
    da.pop('created', None)
    db.pop('created', None)
    if json.dumps(da, sort_keys=True) != json.dumps(db, sort_keys=True):
        return False
    ra, rb = (os.path.join(os.path.dirname(x), 'sim_remap.npz') for x in (a, b))
    try:
        with np.load(ra) as za, np.load(rb) as zb:
            return set(za.files) == set(zb.files) and all(np.array_equal(za[k], zb[k]) for k in za.files)
    except (OSError, ValueError):
        return not os.path.exists(ra) and not os.path.exists(rb)


def _carla_python() -> str:
    try:
        from PyQt6.QtCore import QSettings
        saved = str(QSettings('Vantage', 'Vantage').value('sim/python', '') or '')
    except Exception:                       # noqa: BLE001 — no settings: look for it
        saved = ''
    if saved and os.path.isfile(saved):
        return saved
    from ..sim import server as SV
    return next((py for py in (SV.client_python(r) for r in SV.candidate_roots()) if py), '')
