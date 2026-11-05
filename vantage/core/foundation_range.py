"""Detection and depth range of a rig's cameras with foundation models: for each camera and object class, the range at
which a modern detector actually finds the object (P(detect) = 50 % and 90 %, with intervals), and how far monocular
depth stays usable, measured on rendered frames with ground truth, beside the geometric prediction (core.perception's
pixels-on-target range of the same camera and class), so the user sees where geometry over- or under-promises.

The models (core.foundation_models)
-----------------------------------
Open-vocabulary detectors (OWLv2, Grounding DINO) answer text queries, one per class ("a pedestrian", "a child", "a
traffic cone", "a stroller": the class list is the user's), so classes a COCO detector does not have are ranged too;
YOLOX-S (core.detection, COCO) runs beside them as the baseline, on the classes COCO maps to (a child is a COCO person).
Depth: Depth Anything V2, metric (metres) or relative (inverse depth up to scale and shift, turned into metres per
frame from the ground plane, below).

The frames
----------
* Range sweeps (source 'sweep'): objects of each class stood at increasing distances (ranges_m) along each sweep
  camera's axis (and offsets_deg off it), and along lines parallel to the vehicle at road_lateral_m to its side (ahead
  and behind: the off-axis view of the side and rear cameras), on the ground, still, around the still ego in CARLA:
  the actor chosen and placed as core.critical_carla places a critical target (an adult or child walker by its
  measured rendered height facing the vehicle, the car nearest in size heading along the road, a two-wheeler with its
  rider, a cone). Objects that would overlap in any evaluated camera's image, or stand within MIN_SEP_M of each other
  or of the vehicle, go to separate drives (the core.critical_carla grouping). Each scene is driven at every seed
  (the bridge picks a straight road by the seed: a different background and light), one still frame measured per
  drive. Rendered through the rig's own cameras (only the sweep cameras; no lidar or radar), with ground truth.
* Existing recordings (source 'recordings'): a CARLA drive with ground truth (core.groundtruth): its objects at
  whatever ranges they occur, every frame_step-th camera frame. No server needed.
Each camera frame is evaluated as recorded (the fisheye image) and/or through rectified pinhole views cut from it
(core.detection_eval.make_views: level, view_pitch_deg down, view_yaws about the camera's heading); depth only on the
pinhole views (the models assume a pinhole camera).

Detection
---------
A trial is one object (one actor at one camera frame) the camera sees: in the image and not hidden more than
max_occluded or cut off more than max_truncated (the ground truth's own shares). Size never makes an object an ignore
region here: a far object a few pixels tall is exactly what the range study is about (a sweep object whose ground
truth shows no pixel at all, though its centre is in view and not hidden in the range image, counts as missed, its
height from the lens model). In a sweep only the placed objects are trials; the town's own objects are ignore regions
(a detection on them is neither right nor wrong). Matching per image and class as COCO's (detection_eval.match):
detections of the class (or of the classes it accepts, 'also': a child found as "a pedestrian" counts), at or above
the model's operating score, in descending score order, each to the object it overlaps most with IoU ≥ iou (0.5), or
≥ iou_small (0.3) for objects less than small_px tall. In the pinhole views an object counts as detected when any view
that shows it detects it. A detection overlapping no labelled object at all (IoU < 0.1 with every box, any class) is a
false detection (per image, for context and the perception records).

P(detect) against range
-----------------------
Per camera, class, model and image kind: the trials binned in range (bins RANGE_BIN_RATIO wide in log range), a
logistic regression of P(detect) on log10(range) fitted by maximum likelihood (simcampaign.logistic_fit, on 1/range so
the curve rises with the magnitude), and the ranges where it crosses 50 % and 90 % (R50, R90; simcampaign.fit_threshold).
Intervals: percentile intervals of a cluster bootstrap (Efron and Tibshirani 1993): trials resampled by object (a
sweep's placement in one drive, a recording's actor over its frames), every fit repeated (bootstrap resamples). A
crossing beyond the farthest range tested is reported as 'beyond' (extrapolated), one nearer than the nearest as
'below'. The same fit on the object's height in pixels gives H50 / H90 (the pixel height the detector needs).

Depth
-----
Per pinhole view frame, against the ground truth's planar depth (the range image along each pixel's ray over |(x, y,
1)| of the ray: the views share the fisheye's optical centre): AbsRel = mean |d − d*| / d* and δ < 1.25 = the share of
pixels with max(d / d*, d* / d) < 1.25 (Eigen, Puhrsch and Fergus 2014), per ground-truth depth bin (depth_edges) and
per region of core.regions' grid (the pixel's 3D point in the vehicle frame), over pixels with a ground-truth depth
(not the sky) and not the ego's own body (the ground truth's 'ego' class: a fixed mask a real rig knows from its
install). Relative models: per frame, inverse depth = α / z + β fitted to the pixels whose ray meets the ground plane
(z = 0 of the vehicle frame: the camera's mounted height and pose) between GROUND_NEAR_M and GROUND_FAR_M, by RANSAC
then least squares on the inliers (objects and kerbs in that band are outliers); then depth = α / (inverse depth − β).
Usable depth = the far edge of the farthest bin with AbsRel below abs_rel_max (10 % by default) and at least
depth_min_px pixels; contiguous_m: the same over the bins from the nearest without a gap. Per object: the median depth
over its own pixels against the median ground truth, binned by its range.

The geometric prediction
------------------------
core.perception.object_pixels (the object's bottom and top projected through the camera's lens model, body occlusion
when an engine is given) along the line from the camera at the median bearing of the camera's trials: the farthest
range where the object's height is at least min_px pixels in the image the detector sees (the sensor's pixels × the
recording's image scale), min_px from the project's perception classes of the same name, else the class's own. A
recording's camera missing from the project uses the recording's own calibration (Kannala-Brandt). The ratio R50 /
R_geo (and R90 / R_geo) says where the pixel rule over- or under-promises.
"""
from __future__ import annotations

import copy
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
from . import detection_eval as DE
from . import foundation_models as FM

SCHEMA = 'vantage.foundation_range/1'
RECORD_SCHEMA = 'vantage.perception_record/1'
FRAME = 'vehicle (ISO 8855, rear axle)'
SOURCES = ('recordings', 'sweep')
VIEW_MODES = ('fisheye', 'pinhole', 'both')
MODE_LABEL = {'fisheye': 'Fisheye images', 'pinhole': 'Pinhole views'}
RANGE_BIN_RATIO = 1.18            # range bins of the fit: each this many times the last (about 17 % wide)
HEIGHT_BIN_RATIO = 1.25
GROUND_NEAR_M, GROUND_FAR_M = 1.5, 40.0
MIN_SEP_M = 1.5                   # sweep objects nearer each other (footprints) or the vehicle than this: apart
OVERLAP_PAD = (0.15, 12.0)        # image boxes grown by 15 % and 12 px before the overlap test
FP_IOU = 0.1                      # a detection overlapping no labelled box this much is a false detection
MAX_EXAMPLES = 48
MIN_CLUSTERS = 5                  # fewer independent objects than this: no bootstrap interval (it would say nothing)
AXIS_DEG = 30.0                   # trials within this of a camera's axis (horizontally) are 'on axis'
MAX_RECORDINGS = 8

# The study's classes: name, text query (open-vocabulary models), YOLOX's class (core.detection.CLASSES) or '', the
# CARLA kind a sweep stands in (core.critical_carla: pedestrian, child, car, or an object named like it), ground-truth
# nuScenes categories and classes (core.groundtruth) that are this class, classes whose detections also count, size
# (m) and the pixel height the geometric rule asks for.
DEFAULT_CLASSES = [
    {'name': 'pedestrian', 'query': 'a pedestrian', 'coco': 'pedestrian', 'carla': 'pedestrian',
     'categories': ['human.pedestrian.adult', 'human.pedestrian.police_officer',
                    'human.pedestrian.construction_worker'], 'gt_classes': ['pedestrian'], 'also': ['child'],
     'height': 1.8, 'width': 0.5, 'min_px': 24.0},
    {'name': 'child', 'query': 'a child', 'coco': 'pedestrian', 'carla': 'child',
     'categories': ['human.pedestrian.child'], 'gt_classes': [], 'also': ['pedestrian'],
     'height': 1.15, 'width': 0.36, 'min_px': 24.0},
    {'name': 'cyclist', 'query': 'a cyclist', 'coco': 'cyclist', 'carla': 'cyclist',
     'categories': ['vehicle.bicycle'], 'gt_classes': ['bicycle', 'rider'], 'also': [],
     'height': 1.7, 'width': 0.6, 'min_px': 24.0},
    {'name': 'car', 'query': 'a car', 'coco': 'car', 'carla': 'car', 'categories': ['vehicle.car'],
     'gt_classes': ['car'], 'also': [], 'height': 1.5, 'width': 1.8, 'min_px': 20.0},
    {'name': 'truck', 'query': 'a truck', 'coco': 'truck', 'carla': 'truck', 'categories': ['vehicle.truck'],
     'gt_classes': ['truck'], 'also': [], 'height': 3.0, 'width': 2.5, 'min_px': 20.0},
    {'name': 'traffic cone', 'query': 'a traffic cone', 'coco': '', 'carla': 'cone',
     'categories': ['movable_object.trafficcone'], 'gt_classes': [], 'also': [], 'height': 0.6, 'width': 0.35,
     'min_px': 12.0},
    {'name': 'stroller', 'query': 'a stroller', 'coco': '', 'carla': '', 'categories': ['human.pedestrian.stroller'],
     'gt_classes': [], 'also': [], 'height': 1.0, 'width': 0.6, 'min_px': 16.0},
    {'name': 'wheelchair', 'query': 'a wheelchair', 'coco': '', 'carla': '',
     'categories': ['human.pedestrian.wheelchair'], 'gt_classes': [], 'also': [], 'height': 1.2, 'width': 0.7,
     'min_px': 16.0},
    {'name': 'debris', 'query': 'debris on the road', 'coco': '', 'carla': '', 'categories': ['movable_object.debris'],
     'gt_classes': [], 'also': [], 'height': 0.3, 'width': 0.5, 'min_px': 10.0},
    {'name': 'animal', 'query': 'an animal', 'coco': '', 'carla': '', 'categories': ['animal'], 'gt_classes': [],
     'also': [], 'height': 0.8, 'width': 1.0, 'min_px': 16.0},
]
CARLA_KINDS = ('pedestrian', 'child', 'car', 'truck', 'cyclist', 'motorcycle', 'cone', 'barrier', '')
DEFAULT_SCORES = {'yolox': 0.3, 'owlv2': 0.2, 'grounding_dino': 0.3}
DEFAULT_RANGES = [4.0, 6.0, 8.0, 10.0, 13.0, 16.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0]
LIMITS = {'iou': (0.1, 0.95), 'iou_small': (0.05, 0.95), 'small_px': (0.0, 512.0), 'max_occluded': (0.0, 1.0),
          'max_truncated': (0.0, 1.0), 'frame_step': (1, 1000), 'view_fov_deg': (30.0, 150.0),
          'view_pitch_deg': (-60.0, 89.0), 'view_width': (0, 2048), 'view_aspect': (0.3, 1.5),
          'abs_rel_max': (0.01, 1.0), 'depth_min_px': (1, 10 ** 7), 'bootstrap': (0, 5000),
          'confidence': (0.5, 0.999), 'seed': (0, 2 ** 31 - 1), 'gpu_mem_gb': (1.0, 15.0),
          'image_scale': (0.05, 1.0), 'sweep_frames': (1, 5), 'duration_s': (1.0, 10.0), 'max_frames': (0, 100000)}
INTS = ('frame_step', 'view_width', 'depth_min_px', 'bootstrap', 'seed', 'sweep_frames', 'max_frames')
ASSUMPTIONS = (
    'Detection ranges are measured on CARLA\'s renders with the models as published (OWLv2 and Grounding DINO '
    'trained on photographs, YOLOX on COCO): absolute ranges are these models\' on synthetic images; the comparison '
    'of cameras, classes, models and the geometric rule on the same frames is what carries over. A trial is one '
    'object at one frame, detected when a detection of its class (or a class it accepts) at the model\'s operating '
    'score overlaps it by the IoU threshold. R50 and R90 come from a logistic fit in log range with cluster-bootstrap '
    'intervals (resampling objects). Depth is scored on pinhole views against CARLA\'s exact depth; a relative model '
    'is scaled per frame from the ground plane under the camera\'s mounted pose.')
LIMITS_TEXT = (
    'A sweep stands still objects on the road of one CARLA town at the seeds\' spawn points: what lies behind them '
    '(buildings, a parked car) is CARLA\'s, and objects placed inside a building are not seen (they are not trials). '
    'Side cameras see down the road only off their axis; along their axis the town\'s buildings end the sweep. '
    'One still frame per drive: no motion blur, rolling shutter or compression beyond the recording\'s JPEG. The '
    'relative-depth scale assumes a flat ground under the camera\'s calibrated pose. Classes CARLA cannot stand in '
    '(stroller, wheelchair, debris, animal) are ranged only where a recording\'s ground truth has them. The frames '
    'carry the CARLA page\'s exposure: metered by default from round 15 (each camera\'s mean level at 118, fixed for '
    'the drive); drives from before were at CARLA\'s manual exposure, which renders a sunny road near white (the '
    'result\'s notes and each recording\'s exposure say which). The exposure is one fixed value per camera: no auto '
    'exposure lag, flicker or high dynamic range of a real camera.')
METHOD = {
    'trial': 'one object at one camera frame, seen (not hidden more than max_occluded nor cut off more than '
             'max_truncated); a pinhole-view trial is detected when any view showing it detects it',
    'match': 'COCO\'s greedy matching per image and class (detection_eval.match), IoU ≥ iou, ≥ iou_small below '
             'small_px tall',
    'fit': 'P(detect) = σ(a + b·log10(1/range)) by maximum likelihood on range bins (simcampaign.logistic_fit); '
           'R50, R90 where it crosses 0.5 and 0.9; cluster-bootstrap percentile intervals',
    'depth': 'AbsRel = mean |d − d*| / d*, δ1 = share with max(d/d*, d*/d) < 1.25 (Eigen et al. 2014), per '
             'ground-truth planar-depth bin and region; usable = far edge of the farthest bin with AbsRel < '
             'abs_rel_max',
    'geometric': 'core.perception.object_pixels along the camera\'s median trial bearing: the farthest range with '
                 'the object\'s image height ≥ min_px',
}
REFERENCES = (
    'Minderer, Gritsenko, Houlsby (2023) Scaling Open-Vocabulary Object Detection, NeurIPS (OWLv2), arXiv:2306.09683',
    'Liu et al. (2023) Grounding DINO, arXiv:2303.05499',
    'Yang et al. (2024) Depth Anything V2, arXiv:2406.09414',
    'Ge et al. (2021) YOLOX: Exceeding YOLO Series in 2021, arXiv:2107.08430',
    'Eigen, Puhrsch, Fergus (2014) Depth Map Prediction from a Single Image using a Multi-Scale Deep Network, NeurIPS',
    'Lin et al. (2014) Microsoft COCO: Common Objects in Context, ECCV (matching)',
    'Efron, Tibshirani (1993) An Introduction to the Bootstrap, Chapman & Hall (percentile intervals)')


# =========================================================================== classes
def _strs(v, n=12, ln=60) -> list:
    if isinstance(v, str):
        v = [x for x in v.replace(';', ',').split(',')]
    if not isinstance(v, (list, tuple)):
        return []
    out = []
    for x in v[:n]:
        if isinstance(x, str) and x.strip() and x.strip()[:ln] not in out:
            out.append(x.strip()[:ln])
    return out


def clean_class(c) -> dict | None:
    """A class as stored, or None when it is not usable (no name or query, a non-positive size)."""
    if not isinstance(c, dict):
        return None
    name = str(c.get('name') or '').strip()[:40]
    query = str(c.get('query') or '').strip()[:80] or (f'a {name}' if name else '')
    if not name:
        return None
    out = {'name': name, 'query': query, 'coco': c.get('coco') if c.get('coco') in DT.CLASSES else '',
           'carla': c.get('carla') if c.get('carla') in CARLA_KINDS else '',
           'categories': _strs(c.get('categories')), 'gt_classes': _strs(c.get('gt_classes')),
           'also': _strs(c.get('also'))}
    for k, d, lo, hi in (('height', 1.0, 0.05, 20.0), ('width', 0.5, 0.05, 20.0), ('min_px', 20.0, 1.0, 512.0)):
        try:
            x = float(c.get(k, d))
        except (TypeError, ValueError):
            x = d
        out[k] = min(max(x, lo), hi) if math.isfinite(x) else d
    return out


def class_index(classes, name) -> int:
    return next((i for i, c in enumerate(classes) if c['name'] == name), -1)


def class_of_object(o: dict, classes) -> int:
    """The study class of a ground-truth object: by its nuScenes category first, else by its class; -1 for none."""
    cat = str(o.get('category') or '')
    for i, c in enumerate(classes):
        if cat and cat in c['categories']:
            return i
    cl = str(o.get('class') or '')
    for i, c in enumerate(classes):
        if cl and cl in c['gt_classes']:
            return i
    return -1


def accepted(classes, i) -> list:
    """Label indices whose detections count for class i: its own and its 'also' classes."""
    out = [i]
    for n in classes[i]['also']:
        j = class_index(classes, n)
        if j >= 0 and j not in out:
            out.append(j)
    return out


# =========================================================================== settings
@dataclass
class FoundationSettings:
    source: str = 'recordings'                     # recordings | sweep
    recordings: list = field(default_factory=list)
    detectors: list = field(default_factory=lambda: list(FM.DEFAULT_DETECTORS))
    depth_model: str = FM.DEFAULT_DEPTH            # '' = no depth
    opt_in: bool = False                           # models whose licence does not allow commercial use
    classes: list = field(default_factory=lambda: copy.deepcopy(DEFAULT_CLASSES))
    scores: dict = field(default_factory=lambda: dict(DEFAULT_SCORES))   # the operating score per model family
    iou: float = 0.5
    iou_small: float = 0.3
    small_px: float = 32.0
    max_occluded: float = 0.5
    max_truncated: float = 0.3
    views: str = 'both'
    view_fov_deg: float = 90.0
    view_pitch_deg: float = 10.0
    view_yaws: list = field(default_factory=lambda: [-60.0, 0.0, 60.0])
    view_width: int = 0                            # px; 0 = the camera's own focal length (no resampling loss)
    view_aspect: float = 0.75
    depth_edges: list = field(default_factory=lambda: [5.0, 10.0, 20.0, 30.0, 40.0, 60.0, 80.0])
    abs_rel_max: float = 0.10
    depth_min_px: int = 2000
    frame_step: int = 5
    max_frames: int = 0                            # per camera (0: all)
    cameras: list = field(default_factory=list)    # [] = every camera
    bootstrap: int = 400
    confidence: float = 0.95
    seed: int = 0
    device: str = 'auto'                           # auto | cuda | cpu
    gpu_mem_gb: float = 6.0
    # the sweep
    sweep_cameras: list = field(default_factory=lambda: ['F', 'ML'])
    sweep_classes: list = field(default_factory=lambda: ['pedestrian', 'car'])
    ranges_m: list = field(default_factory=lambda: list(DEFAULT_RANGES))
    offsets_deg: list = field(default_factory=lambda: [0.0])
    road_lateral_m: list = field(default_factory=lambda: [3.5])
    seeds: list = field(default_factory=lambda: [1, 2])
    town: str = 'Town01'
    weather: str = 'ClearNoon'
    image_scale: float = 1.0
    duration_s: float = 1.0
    sweep_frames: int = 1

    def to_dict(self):
        return asdict(self)

    def view_width_for(self, cam=None) -> int:
        """The pinhole views' width: as set, or (0) the width at which a view's focal length equals the camera's at
        its centre, so the views keep the sensor's resolution on the axis (at most 2048 px)."""
        if self.view_width or cam is None or getattr(cam, 'K', None) is None:
            return int(self.view_width or 640)
        w = 2.0 * float(cam.K[0, 0]) * math.tan(math.radians(self.view_fov_deg) / 2.0)
        return int(min(max(round(w / 32.0) * 32, 128), 2048))

    def view_settings(self, cam=None) -> DE.DetectionSettings:
        """detection_eval's settings for the views (of cam) and the ignore rules: size never ignores an object."""
        return DE.DetectionSettings.from_dict({
            'view_fov_deg': self.view_fov_deg, 'view_pitch_deg': self.view_pitch_deg, 'view_yaws': self.view_yaws,
            'view_width': self.view_width_for(cam), 'view_aspect': self.view_aspect, 'min_px': 0.0, 'min_pixels': 0,
            'max_occluded': self.max_occluded, 'max_truncated': self.max_truncated, 'iou': self.iou,
            'iou_small': self.iou_small, 'small_px': self.small_px})

    def modes(self) -> list:
        return {'fisheye': ['fisheye'], 'pinhole': ['pinhole'], 'both': ['fisheye', 'pinhole']}[self.views]

    def score_of(self, mid: str) -> float:
        fam = 'yolox' if mid in DT.MODELS else FM.SPECS[mid].arch if mid in FM.SPECS else 'yolox'
        return float(self.scores.get(fam, DEFAULT_SCORES.get(fam, 0.3)))

    @staticmethod
    def from_dict(d) -> 'FoundationSettings':
        from . import simrig as SR
        s = FoundationSettings()
        d = d if isinstance(d, dict) else {}
        if d.get('source') in SOURCES:
            s.source = d['source']
        if d.get('views') in VIEW_MODES:
            s.views = d['views']
        recs = d.get('recordings')
        if isinstance(recs, (list, tuple)):
            s.recordings = _strs(recs, MAX_RECORDINGS, 4096)
        dets = d.get('detectors')
        if isinstance(dets, (list, tuple)):
            s.detectors = [m for m in dict.fromkeys(dets) if m in DT.MODELS or m in FM.DETECTORS][:6]
        if d.get('depth_model') in FM.DEPTH_MODELS or d.get('depth_model') == '':
            s.depth_model = d['depth_model']
        for k in ('opt_in',):
            v = d.get(k)
            if isinstance(v, bool):
                setattr(s, k, v)
        if d.get('device') in ('auto', 'cuda', 'cpu'):
            s.device = d['device']
        cl = d.get('classes')
        if isinstance(cl, (list, tuple)):
            out, seen = [], set()
            for c in cl[:24]:
                c = clean_class(c)
                if c is not None and c['name'] not in seen:
                    seen.add(c['name'])
                    out.append(c)
            if out:
                s.classes = out
        sc = d.get('scores')
        if isinstance(sc, dict):
            for k, v in sc.items():
                if k in DEFAULT_SCORES:
                    try:
                        x = float(v)
                    except (TypeError, ValueError):
                        continue
                    if math.isfinite(x):
                        s.scores[k] = min(max(x, 0.01), 0.99)
        for k, (lo, hi) in LIMITS.items():
            if k not in d or isinstance(d[k], bool) or not hasattr(s, k):
                continue
            try:
                x = float(d[k])
            except (TypeError, ValueError):
                continue
            if math.isfinite(x):
                x = min(max(x, lo), hi)
                setattr(s, k, int(round(x)) if k in INTS else x)
        for k, lo, hi, n in (('depth_edges', 0.5, 500.0, 12), ('ranges_m', 1.0, 300.0, 40),
                             ('road_lateral_m', -30.0, 30.0, 4), ('offsets_deg', -90.0, 90.0, 6),
                             ('view_yaws', -180.0, 180.0, 8)):
            v = DE._num_list(d.get(k), lo, hi, n)
            if v is not None and (v or k in ('road_lateral_m',)):
                setattr(s, k, sorted(set(v)) if k in ('depth_edges', 'ranges_m') else list(dict.fromkeys(v)))
        v = DE._num_list(d.get('seeds'), 0, 999999, 8)
        if v:
            s.seeds = list(dict.fromkeys(int(x) for x in v))
        for k in ('cameras', 'sweep_cameras'):
            v = d.get(k)
            if isinstance(v, (list, tuple, str)):
                setattr(s, k, [x[:16] for x in _strs(v, 32, 16)])
        v = d.get('sweep_classes')
        if isinstance(v, (list, tuple, str)):
            names = {c['name'] for c in s.classes}
            s.sweep_classes = [x for x in _strs(v, 12) if x in names]
        if isinstance(d.get('town'), str):
            s.town = d['town'].strip()[:64]
        if d.get('weather') in SR.CARLA_WEATHERS:
            s.weather = d['weather']
        return s


def settings_of(project) -> FoundationSettings:
    a = getattr(project, 'analysis', None) if project is not None else None
    return FoundationSettings.from_dict(getattr(a, 'foundation', None) if a is not None else None)


# =========================================================================== the range fit
def _bins(values, ratio):
    """Log bins over values: (bin index per value, geometric-mean centre per bin)."""
    v = np.asarray(values, float)
    lo = max(float(np.min(v)), 1e-3)
    edges = [lo]
    while edges[-1] <= float(np.max(v)):
        edges.append(edges[-1] * ratio)
    idx = np.clip(np.searchsorted(np.asarray(edges), v, side='right') - 1, 0, len(edges) - 2)
    return idx, np.asarray(edges)


def _binned(values, det, idx, nb):
    k = np.bincount(idx, weights=det.astype(float), minlength=nb)
    n = np.bincount(idx, minlength=nb).astype(float)
    with np.errstate(divide='ignore', invalid='ignore'):
        lg = np.bincount(idx, weights=np.log(np.maximum(values, 1e-6)), minlength=nb) / np.maximum(n, 1)
    c = np.exp(lg)
    ok = n > 0
    return c[ok], k[ok], n[ok]


def _fit_cross(mags, k, n, targets):
    """The logistic fit on magnitudes rising with P(detect): (fit, [the magnitude where it reaches each target])."""
    from . import simcampaign as SC
    fit = SC.logistic_fit(mags, k, n)
    return fit, [SC.fit_threshold(fit, t) for t in targets]


def detect_fit(values, detected, clusters=None, increasing=False, ratio=RANGE_BIN_RATIO, boot=400, conf=0.95,
               seed=0, targets=(0.5, 0.9)) -> dict:
    """P(detect) against a magnitude by logistic regression in its log: values are ranges (P falls with them:
    increasing False, fitted on 1/range) or pixel heights (increasing True). The values where it crosses each target,
    with cluster-bootstrap percentile intervals (clusters: an id per trial, resampled with replacement; None: every
    trial its own). {'n', 'k', 'x': [per target], 'ci': [per target], 'beyond': [...], 'below': [...], 'a', 'b',
    'tested': [lo, hi], 'bins': [{'x', 'k', 'n', 'p', 'lo', 'hi'}], 'separated'}."""
    from . import simcampaign as SC
    v = np.asarray(values, float)
    d = np.asarray(detected, bool)
    ok = np.isfinite(v) & (v > 0)
    v, d = v[ok], d[ok]
    cl = None if clusters is None else np.asarray(clusters, object)[ok]
    out = {'n': int(len(v)), 'k': int(d.sum()), 'x': [None] * len(targets), 'ci': [None] * len(targets),
           'beyond': [False] * len(targets), 'below': [False] * len(targets), 'a': None, 'b': None,
           'tested': None, 'bins': [], 'separated': False, 'targets': list(targets), 'increasing': bool(increasing)}
    if len(v) < 2:
        return out
    idx, edges = _bins(v, ratio)
    nb = len(edges) - 1
    c, k, n = _binned(v, d, idx, nb)
    out['tested'] = [float(v.min()), float(v.max())]
    for x_, k_, n_ in zip(c, k, n):
        w = SC.wilson(int(k_), int(n_), conf)
        out['bins'].append({'x': float(x_), 'k': int(k_), 'n': int(n_), 'p': float(k_ / n_),
                            'lo': w[0] if w else None, 'hi': w[1] if w else None})
    if len(c) < 2:
        return out
    to_m = (lambda x: x) if increasing else (lambda x: 1.0 / x)
    fit, ms = _fit_cross(to_m(c), k, n, targets)
    out['a'], out['b'], out['separated'] = fit.get('a'), fit.get('b'), bool(fit.get('separated'))
    lo_v, hi_v = float(v.min()), float(v.max())
    allk, nonek = bool(d.all()), not d.any()
    # a curve that does not fall with range (rise with height): b ≤ 0 with both outcomes seen, say so, no crossing
    out['flat'] = bool(not allk and not nonek and (out['b'] is None or out['b'] <= 0))
    for i, m in enumerate(ms):
        if out['flat']:
            continue
        never = (m is None and not allk) or nonek
        if allk or never:
            # every trial found: a range beyond the farthest tested (a height below the smallest); none, or a curve
            # that never rises to the target: a range nearer than the nearest (a height above the largest)
            out['beyond'][i] = (allk and not increasing) or (never and increasing)
            out['below'][i] = (allk and increasing) or (never and not increasing)
            continue
        x = float(to_m(m))
        out['x'][i] = x
        out['beyond'][i] = bool(x > hi_v * 1.0001)
        out['below'][i] = bool(x < lo_v * 0.9999)
    out['clusters'] = int(len(set(cl.tolist()))) if cl is not None else int(len(v))
    if boot and len(v) >= 4 and out['clusters'] >= MIN_CLUSTERS:
        rng = np.random.default_rng(seed)
        if cl is None:
            groups = [np.array([i]) for i in range(len(v))]
        else:
            uniq = {}
            for i, g in enumerate(cl):
                uniq.setdefault(g, []).append(i)
            groups = [np.array(x) for x in uniq.values()]
        G = len(groups)
        samples = [[] for _ in targets]
        big = 1e9
        for _ in range(int(boot)):
            pick = rng.integers(0, G, G)
            ii = np.concatenate([groups[j] for j in pick])
            cb, kb, nbb = _binned(v[ii], d[ii], idx[ii], nb)
            every, none_ = (0.0, big) if increasing else (big, 0.0)    # all found / none found
            if kb.sum() >= nbb.sum():
                for s in samples:
                    s.append(every)
                continue
            if kb.sum() <= 0 or len(cb) < 2:
                for s in samples:
                    s.append(none_)
                continue
            _, mb = _fit_cross(to_m(cb), kb, nbb, targets)
            for t, m in enumerate(mb):
                if m is None:
                    samples[t].append(big if increasing else 0.0)
                else:
                    x = float(to_m(m))
                    # far outside the values tested (10×): the fit says nothing about it there
                    samples[t].append(big if x > 10 * hi_v else (0.0 if x < lo_v / 10 else x))
        a = (1 - conf) / 2
        for t, s in enumerate(samples):
            s = np.asarray(s, float)
            s = s[np.isfinite(s)]
            if len(s) >= 10:
                q = np.quantile(s, [a, 1 - a])
                lo_q = float(q[0]) if q[0] < big / 2 else None       # None: beyond (or below) every value tested
                hi_q = float(q[1]) if q[1] < big / 2 else None
                # a point interval (every resample the same data, or the same crossing) says nothing: none is given
                if lo_q is not None and hi_q is not None and abs(hi_q - lo_q) <= 1e-9 * max(abs(hi_q), 1.0):
                    continue
                if lo_q is None and hi_q is None:
                    continue
                out['ci'][t] = [lo_q, hi_q]
    return out


def curve(fit: dict, xs) -> np.ndarray:
    """The fitted P(detect) at values xs (NaN without a fit)."""
    xs = np.asarray(xs, float)
    if fit.get('a') is None or fit.get('b') is None:
        return np.full(xs.shape, np.nan)
    m = xs if fit.get('increasing') else 1.0 / np.maximum(xs, 1e-9)
    z = fit['a'] + fit['b'] * np.log10(np.maximum(m, 1e-12))
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


# =========================================================================== depth
def depth_scores(pred, gt, valid):
    """Per pixel (AbsRel, δ<1.25) over valid pixels with a positive ground truth and prediction."""
    p = np.asarray(pred, float)
    g = np.asarray(gt, float)
    ok = np.asarray(valid, bool) & np.isfinite(p) & np.isfinite(g) & (g > 0) & (p > 0)
    pp, gg = p[ok], g[ok]
    rel = np.abs(pp - gg) / gg
    d1 = np.maximum(pp / gg, gg / pp) < 1.25
    return ok, rel, d1


def depth_bins(pred, gt, valid, edges):
    """(n, Σ AbsRel, n with δ<1.25) per ground-truth depth bin (0 … len(edges))."""
    ok, rel, d1 = depth_scores(pred, gt, valid)
    b = DE.bin_of(np.asarray(gt, float)[ok], edges).astype(int)
    nb = len(edges) + 1
    return (np.bincount(b, minlength=nb)[:nb].astype(float), np.bincount(b, weights=rel, minlength=nb)[:nb],
            np.bincount(b, weights=d1.astype(float), minlength=nb)[:nb])


def usable_range(edges, n, s_rel, thr, min_px) -> tuple:
    """(usable_m, contiguous_m): the far edge of the farthest depth bin with AbsRel < thr (bins with at least min_px
    pixels), and of the last bin of the run from the nearest without a failing bin; inf for the open last bin; 0 when
    none."""
    e = list(edges) + [math.inf]
    use, cont, run = 0.0, 0.0, True
    for i in range(len(n)):
        if n[i] < min_px:
            continue
        good = s_rel[i] / n[i] < thr
        if good:
            use = e[i]
            if run:
                cont = e[i]
        else:
            run = False
    return use, cont


def ground_scale(disp, rays_v, cam_z, mask, near=GROUND_NEAR_M, far=GROUND_FAR_M, iters=200, seed=0):
    """Relative inverse depth -> metres from the ground plane: disp ≈ α / z + β fitted on the pixels whose ray meets
    z = 0 of the vehicle frame at planar depth z between near and far (rays_v: H × W × 3 vehicle-frame directions of
    the view's (x, y, 1) rays, cam_z the camera's height), RANSAC then least squares on the inliers.
    Returns (depth H × W or None, info {'alpha', 'beta', 'inliers', 'candidates'})."""
    dz = rays_v[..., 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        zg = np.where(dz < -1e-6, -cam_z / dz, np.nan)
    cand = np.asarray(mask, bool) & np.isfinite(zg) & (zg >= near) & (zg <= far) & np.isfinite(disp)
    info = {'alpha': None, 'beta': None, 'inliers': 0.0, 'candidates': int(cand.sum())}
    if cand.sum() < 50:
        return None, info
    z = zg[cand]
    x = 1.0 / z
    y = np.asarray(disp, float)[cand]
    rng = np.random.default_rng(seed)
    # the near ground has most of the pixels: hypotheses from pairs in different depth strata (log z), so the far
    # ground, where a shift error matters most, is in every fit
    strata = np.minimum((np.log(z / near) / np.log(far / near) * 8).astype(int), 7)
    by = [np.flatnonzero(strata == k) for k in range(8)]
    by = [b_ for b_ in by if len(b_)]
    best, best_in, best_w = None, -1, -1.0
    n = len(x)
    w = z * z / np.mean(z * z)                   # relative depth error δz/z = z·δd/α: weight residuals by z²
    for _ in range(iters):
        if len(by) >= 2:
            s1, s2 = rng.choice(len(by), 2, replace=False)
            i, j = rng.choice(by[s1]), rng.choice(by[s2])
        else:
            i, j = rng.integers(0, n, 2)
        if abs(x[i] - x[j]) < 1e-4:
            continue
        a = (y[i] - y[j]) / (x[i] - x[j])
        if a <= 0:
            continue
        b = y[i] - a * x[i]
        inl = np.abs(y - (a * x + b)) < 0.1 * a * x
        c = float(w[inl].sum())
        if c > best_w:
            best, best_in, best_w = inl, int(inl.sum()), c
    if best is None or best_in < 20:
        return None, info
    sw = np.sqrt(w[best])
    A = np.column_stack([x[best], np.ones(best_in)]) * sw[:, None]
    a, b = np.linalg.lstsq(A, y[best] * sw, rcond=None)[0]
    if not (a > 0):
        return None, info
    with np.errstate(divide='ignore', invalid='ignore'):
        q = np.asarray(disp, float) - b
        depth = np.where(q > 1e-6 * a, a / q, np.nan)
    info.update(alpha=float(a), beta=float(b), inliers=float(best_in / n))
    return depth.astype(np.float32), info


# =========================================================================== the geometric prediction
def _bearing(dx, dy) -> float:
    return math.degrees(math.atan2(dy, dx))


def geometric_range(project, cam, cls: dict, bearing_deg: float, scale: float = 1.0, cube=None,
                    max_range=300.0, samples=400) -> float:
    """core.perception's pixel rule along the line from cam at bearing_deg: the farthest range (from the camera) with
    the object's height ≥ min_px in an image at scale × the sensor's pixels; 0 when it is never met."""
    from . import perception as PE
    from .lens import resolve
    intr = resolve(project.module_for(cam), project.lens_for(cam))
    oc = PE.ObjectClass(cls['name'], float(cls['height']), float(cls['width']), float(cls['min_px']) / max(scale, 1e-6))
    pos = cam.pos()
    r = np.geomspace(0.5, max_range, samples)
    b = math.radians(bearing_deg)
    G = np.c_[pos[0] + r * math.cos(b), pos[1] + r * math.sin(b)]
    h, _, seen = PE.object_pixels(cam, intr, G, oc, cube)
    ok = seen & (h >= oc.min_px)
    if not ok.any():
        return 0.0
    rows = np.arange(1)
    return float(PE._far_range(ok[None], seen[None], h[None], r, oc.min_px, rows)[0])


def geometric_range_kb(rcam: DE.Camera, cls: dict, bearing_deg: float, max_range=300.0, samples=400) -> float:
    """The same rule through a recording's own calibration (Kannala-Brandt), no body occlusion."""
    if rcam.K is None or rcam.R is None:
        return 0.0
    r = np.geomspace(0.5, max_range, samples)
    b = math.radians(bearing_deg)
    t = rcam.t
    base = 0.0
    P_bot = np.c_[t[0] + r * math.cos(b), t[1] + r * math.sin(b), np.full(len(r), base)]
    P_top = P_bot + [0, 0, float(cls['height'])]
    ub, vb, okb = DE.kb_project(rcam, (P_bot - t) @ rcam.R)
    ut, vt, okt = DE.kb_project(rcam, (P_top - t) @ rcam.R)
    inside = okb & okt & (ub >= 0) & (ub < rcam.width) & (vt >= 0) & (vb < rcam.height) & (ut >= 0) & (ut < rcam.width)
    inside &= (vb >= 0) & (vt < rcam.height)
    h = np.hypot(ut - ub, vt - vb)
    ok = inside & (h >= float(cls['min_px']))
    if not ok.any():
        return 0.0
    from . import perception as PE
    return float(PE._far_range(ok[None], inside[None], h[None], r, float(cls['min_px']), np.arange(1))[0])


# =========================================================================== the sweep
@dataclass
class Placement:
    id: str
    cls: str
    x: float
    y: float
    yaw: float
    line: str                       # 'F +0°' (a camera's axis) or 'road +3.5 m ahead'
    planned_m: float                # the line's own range (from the camera, or the vehicle's middle)
    actor: dict = field(default_factory=dict)        # critical_carla.actor_for

    def to_dict(self):
        return asdict(self)


def _box_xy(project):
    from . import critical as CR
    return CR.vehicle_box(project)


def _footprint(p: Placement, cls: dict):
    """Footprint half sizes (along x, along y) of an object at its heading."""
    a = p.actor or {}
    e = a.get('expected') or {}
    L = float(e.get('length') or cls['width'])
    W = float(e.get('width') or cls['width'])
    c, s = abs(math.cos(math.radians(p.yaw))), abs(math.sin(math.radians(p.yaw)))
    return (L * c + W * s) / 2.0, (L * s + W * c) / 2.0


def sweep_placements(project, st: FoundationSettings, variant=None) -> tuple[list, list]:
    """The placements of a sweep: (placements, notes). Every sweep class × line × range; lines along each sweep
    camera's heading (+ offsets) from the camera, and along the vehicle at road_lateral_m ahead and behind from the
    vehicle's middle. Placements inside the vehicle's footprint (with MIN_SEP_M) are left out."""
    from . import critical_carla as CC
    v = variant or project.variant
    cams = {c.tag: c for c in v.cameras if c.visible}
    notes = []
    box = _box_xy(project)
    vx0, vx1 = box['rear_x'], box['front_x']
    vy = box['half_w']
    mid = box['mid_x']
    lines = []
    for tag in st.sweep_cameras:
        c = cams.get(tag)
        if c is None:
            notes.append(f'Sweep camera {tag} is not a visible camera of {v.name}: left out.')
            continue
        for off in st.offsets_deg:
            lines.append((f'{tag} {off:+g}°', 'camera', (float(c.x), float(c.y)), float(c.yaw) + float(off)))
    # road lines ahead (behind) only when a sweep camera looks that way (its heading within 100° of it)
    heads = [float(cams[t].yaw) for t in st.sweep_cameras if t in cams]
    for lat in st.road_lateral_m:
        for word, brg in (('ahead', 0.0), ('behind', 180.0)):
            if any(abs((h - brg + 180.0) % 360.0 - 180.0) < 100.0 for h in heads):
                lines.append((f'road {lat:+g} m {word}', 'road', (mid, float(lat)), brg))
    out = []
    for cname in st.sweep_classes:
        ci = class_index(st.classes, cname)
        if ci < 0:
            continue
        cls = st.classes[ci]
        if not cls['carla']:
            notes.append(f'{cname}: CARLA has no actor for it: ranged only in recordings that have it.')
            continue
        for lname, kind, (ox, oy), brg in lines:
            for r in st.ranges_m:
                b = math.radians(brg)
                x, y = ox + r * math.cos(b), oy + r * math.sin(b)
                pid = f'{_slug(cname)}-{_slug(lname)}-{r:g}'
                t = _target(cls, x, y, mid)
                a = CC.actor_for(project, t, 'auto', box)
                if not a.get('spawnable'):
                    notes.append(f'{cname}: {a.get("why") or "not spawnable"}.')
                    break
                p = Placement(pid, cname, round(x, 3), round(y, 3), float(a['pose']['yaw']), lname, float(r), a)
                hx, hy = _footprint(p, cls)
                if (x + hx > vx0 - MIN_SEP_M and x - hx < vx1 + MIN_SEP_M and abs(y) - hy < vy + MIN_SEP_M):
                    continue                                        # on or next to the vehicle
                out.append(p)
    # the same place from two lines (a camera axis along the road): once
    uniq = {}
    for p in out:
        key = (p.cls, round(p.x, 1), round(p.y, 1))
        uniq.setdefault(key, p)
    return list(uniq.values()), list(dict.fromkeys(notes))


def _target(cls: dict, x, y, mid_x) -> dict:
    """A critical-target-like dict for critical_carla.actor_for (its kind, size, place)."""
    kind = cls['carla']
    t = {'uid': '', 'name': cls['name'], 'height': cls['height'], 'width': cls['width'], 'base': 0.0, 'x': x,
         'y': y, 'yaw': 0.0, 'enabled': True}
    if kind in ('pedestrian', 'child', 'car'):
        t['kind'] = kind
    else:
        t['kind'] = 'object'
        t['name'] = {'cyclist': 'cyclist', 'motorcycle': 'motorcycle', 'truck': 'truck', 'cone': 'traffic cone',
                     'barrier': 'barrier'}.get(kind, cls['name'])
    return t


def _slug(s) -> str:
    return ''.join(ch if ch.isalnum() else '-' for ch in str(s).lower()).strip('-').replace('--', '-')


def _image_boxes(project, cams, p: Placement, cls: dict) -> dict:
    """{camera uid: image box} of a placement's box (its footprint × height) in each camera (sensor px)."""
    from .analysis import camera_frames
    from .lens import resolve
    hx, hy = _footprint(p, cls)
    e = (p.actor or {}).get('expected') or {}
    h = float(e.get('rendered_height') or e.get('height') or cls['height'])
    xs, ys, zs = np.meshgrid([p.x - hx, p.x + hx], [p.y - hy, p.y + hy], [0.0, h / 2, h], indexing='ij')
    P = np.c_[xs.ravel(), ys.ravel(), zs.ravel()]
    out = {}
    for c in cams:
        intr = resolve(project.module_for(c), project.lens_for(c))
        _, W2O = camera_frames(c)
        u, v, _, ok = intr.project_dirs((P - c.pos()) @ W2O.T)
        if not ok.any():
            continue
        out[c.uid] = (float(u[ok].min()), float(v[ok].min()), float(u[ok].max()), float(v[ok].max()))
    return out


def _grown(bx):
    w, h = bx[2] - bx[0], bx[3] - bx[1]
    gx, gy = OVERLAP_PAD[0] * w + OVERLAP_PAD[1], OVERLAP_PAD[0] * h + OVERLAP_PAD[1]
    return bx[0] - gx, bx[1] - gy, bx[2] + gx, bx[3] + gy


def _overlap(a, b) -> bool:
    a, b = _grown(a), _grown(b)
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def sweep_plan(project, st: FoundationSettings, variant=None) -> dict:
    """{'placements', 'groups' [[placement id]], 'drives' [{'key', 'group', 'seed'}], 'notes'}: placements that no
    evaluated camera would see overlap (nor stand within MIN_SEP_M of each other) share a drive; every group is driven
    at every seed. Cheap: no CARLA."""
    v = variant or project.variant
    pls, notes = sweep_placements(project, st, v)
    cams = [c for c in v.cameras if c.visible and c.tag in st.sweep_cameras]
    cls_of = {c['name']: c for c in st.classes}
    boxes = {p.id: _image_boxes(project, cams, p, cls_of[p.cls]) for p in pls}
    foot = {p.id: _footprint(p, cls_of[p.cls]) for p in pls}

    def clash(a: Placement, b: Placement) -> bool:
        fa, fb = foot[a.id], foot[b.id]
        if abs(a.x - b.x) < fa[0] + fb[0] + MIN_SEP_M and abs(a.y - b.y) < fa[1] + fb[1] + MIN_SEP_M:
            return True
        ba, bb = boxes[a.id], boxes[b.id]
        return any(u in bb and _overlap(ba[u], bb[u]) for u in ba)
    groups = []
    # far objects first: they are small and pack densely; near ones need scenes of their own
    for p in sorted(pls, key=lambda q: -q.planned_m):
        for g in groups:
            if not any(clash(p, q) for q in g):
                g.append(p)
                break
        else:
            groups.append([p])
    gids = [[p.id for p in g] for g in groups]
    drives = [{'key': f'scene{gi + 1:02d}-seed{sd}', 'group': gi, 'seed': int(sd)}
              for gi in range(len(gids)) for sd in st.seeds]
    if pls:
        notes.append(f'{len(pls)} placements in {len(gids)} scenes × {len(st.seeds)} seed(s) = {len(drives)} drives.')
    return {'placements': pls, 'groups': gids, 'drives': drives, 'notes': notes}


def sweep_project(project, st: FoundationSettings, variant=None):
    """The project as a sweep drives it: the variant with only the sweep cameras, no lidar, radar or other sensor."""
    v = variant or project.variant
    q = copy.copy(project)
    v2 = copy.deepcopy(v)
    v2.cameras[:] = [c for c in v2.cameras if c.visible and c.tag in st.sweep_cameras]
    v2.sensors[:] = []
    q.variants = [v2]
    q.active_uid = v2.uid
    return q, v2


def run_sweep(project, st: FoundationSettings, variant=None, sim_settings=None, out_dir=None, python=None,
              runner=None, progress=None, cancel=None, memory_gb=5.0) -> tuple[list, dict] | None:
    """Every sweep drive through the CARLA bridge (core.critical_carla.drive: a drive already there from the same rig
    is reused). Returns ([{'key', 'group', 'seed', 'dir', 'reused', 'seconds'}], plan) or None when cancelled."""
    from . import critical_carla as CC
    from . import simrig as SR
    from .. import workdir as WD
    pl = sweep_plan(project, st, variant)
    if not pl['drives']:
        raise ValueError('Nothing to sweep: choose sweep cameras of the variant and classes CARLA can stand in.')
    q, v2 = sweep_project(project, st, variant)
    sim0 = sim_settings or SR.settings_of(project)
    root = os.path.abspath(out_dir or WD.project_dir(project, None, 'carla', 'foundation', WD.slug(v2.name)))
    by_id = {p.id: p for p in pl['placements']}
    done = []
    n = len(pl['drives'])
    for i, d in enumerate(pl['drives']):
        if cancel is not None and cancel():
            return None
        actors = [{'id': pid, 'kind': by_id[pid].actor['kind'], 'pose': dict(by_id[pid].actor['pose'])}
                  for pid in pl['groups'][d['group']]]
        cst = CC.CriticalCarlaSettings.from_dict({'town': st.town, 'weather': st.weather, 'seed': d['seed'],
                                                  'duration_s': st.duration_s, 'image_scale': st.image_scale})
        sim = CC.scene_settings(q, cst, actors, -1, sim0)

        def sub(f, text='', i=i, d=d):
            if progress is not None:
                progress(min((i + f) / n, 1.0), f'{d["key"]} ({i + 1} of {n}) · {text}')
        r = CC.drive(q, v2, sim, os.path.join(root, d['key']), python, sub, cancel, runner, memory_gb)
        if r is None:
            return None
        done.append(dict(d, dir=r['dir'], reused=r['reused'], seconds=round(r['seconds'], 2)))
    with open(os.path.join(root, 'foundation_sweep.json'), 'w', encoding='utf-8') as fh:
        json.dump({'schema': 'vantage.foundation_sweep/1', 'placements': [p.to_dict() for p in pl['placements']],
                   'groups': pl['groups'], 'drives': done, 'settings': st.to_dict()}, fh, indent=1)
    return done, pl


def recording_exposure(path) -> dict | None:
    """The camera exposure a CARLA drive was recorded with (core.simrig.exposure_of on its rig/sim_rig.json and
    run.json): drives from before round 15 read as manual at EV 0; None when the folder has no CARLA rig file (a
    recording from elsewhere)."""
    from . import simrig as SR

    def load(*parts):
        try:
            with open(os.path.join(str(path), *parts), encoding='utf-8') as fh:
                return json.load(fh)
        except (OSError, ValueError, TypeError):
            return None
    rig = load('rig', 'sim_rig.json') or load('sim_rig.json')
    if not isinstance(rig, dict):
        return None
    return SR.exposure_of(rig, load('run.json'))


def exposure_note(exps) -> str:
    """One line for the result's notes: what exposure the evaluated frames were recorded with."""
    exps = [e for e in exps if e]
    if not exps:
        return 'Exposure of the frames: unknown (not CARLA drives of this version\'s bridge).'
    parts = []
    for mode in ('metered', 'manual', 'histogram'):
        es = [e for e in exps if e['mode'] == mode]
        if not es:
            continue
        if mode == 'metered':
            evs = [v for e in es for v in (e.get('cameras') or {}).values()]
            rng = (f' (compensation {min(evs):+.2f} … {max(evs):+.2f} EV)' if evs and max(evs) - min(evs) > 0.005
                   else (f' (compensation {evs[0]:+.2f} EV)' if evs else ''))
            parts.append(f'{len(es)} metered{rng}')
        else:
            evs = sorted({round(float(e.get('exposure_ev', 0.0)), 2) for e in es})
            parts.append(f'{len(es)} {mode} at EV ' + ', '.join(f'{v:+.2f}' for v in evs))
    return 'Exposure of the frames: ' + '; '.join(parts) + ' (core.simrig.exposure_of).'


# =========================================================================== one image
def _objects(frame, classes, vst, sweep=None):
    """A frame's objects as detection_eval.gt_objects arrays plus 'scls' (study class, -1 none), 'trial' (counts
    as a trial), 'pid' (sweep placement id or ''). sweep: {actor id: (placement id, class index)} or None."""
    g = DE.gt_objects(frame, vst)
    objs = [o for o in frame.get('objects') or () if isinstance(o, dict) and o.get('bbox')]
    n = len(objs)
    scls = np.full(n, -1, np.int16)
    pid = []
    for i, o in enumerate(objs):
        if sweep is not None:
            hit = sweep.get(o.get('actor_id'))
            if hit is not None:
                scls[i] = hit[1]
                pid.append(hit[0])
                continue
            pid.append('')
            scls[i] = class_of_object(o, classes)
        else:
            pid.append('')
            scls[i] = class_of_object(o, classes)
    g['scls'] = scls
    g['pid'] = pid
    trial = (scls >= 0) & (g['ignore'] == DE.IGN_NONE)
    if sweep is not None:
        trial &= np.array([bool(p) for p in pid], bool) if n else np.zeros(0, bool)
    g['trial'] = trial
    return g


def match_image(g: dict, dets: dict, classes, score: float, st: FoundationSettings) -> dict:
    """One image's detections against its objects: {'hit': bool per object (trials only meaningful), 'score' per
    object, 'fp': indices of false detections (overlapping no labelled box), 'tp_det': indices of detections that
    matched a trial}."""
    n = len(g['scls'])
    hit = np.zeros(n, bool)
    best = np.full(n, -np.inf)
    boxes = np.asarray(dets['boxes'], float).reshape(-1, 4)
    sc = np.asarray(dets['scores'], float)
    lab = np.asarray(dets['labels'], int)
    on = sc >= score
    thr = np.where(g['height'] < st.small_px, st.iou_small, st.iou) if n else np.zeros(0)
    tp_det = set()
    for ci in range(len(classes)):
        gi = np.flatnonzero(g['scls'] == ci)
        if not len(gi):
            continue
        di = np.flatnonzero(on & np.isin(lab, accepted(classes, ci)))
        if not len(di):
            continue
        status, dg, gs = DE.match(g['box'][gi], ~g['trial'][gi], thr[gi], boxes[di], sc[di])
        hit[gi] |= np.isfinite(gs) & g['trial'][gi]
        best[gi] = np.maximum(best[gi], gs)
        for k, d in enumerate(di):
            if status[k] == 1:
                tp_det.add(int(d))
    fp = []
    if on.any():
        ious = DE.iou_matrix(boxes[on], g['box']) if n else np.zeros((int(on.sum()), 0))
        mx = ious.max(1) if n else np.zeros(int(on.sum()))
        fp = [int(i) for i, m in zip(np.flatnonzero(on), mx) if m < FP_IOU]
    return {'hit': hit, 'score': best, 'fp': fp, 'tp_det': sorted(tp_det)}


# =========================================================================== the result
@dataclass
class FoundationResult:
    settings: FoundationSettings
    source: str
    recordings: list                  # [Recording.info()] (sweep drives: one per drive)
    models: list                      # [describe()] of every model run
    device: dict
    trials: dict                      # numpy columns (see _TRIAL_KEYS)
    fits: list                        # per model, mode, camera, class (+ 'axis' subsets)
    geometric: dict                   # (camera, class) -> {'range_m', 'bearing_deg', 'min_px', 'how'}
    depth: dict                       # per depth model: per camera bins, usable range, per region, objects, scale fits
    false_det: list                   # per model, mode, camera: images, false detections
    examples: list                    # [{'model', 'mode', 'camera', 'caption', 'jpg'}]
    records: list
    timing: dict
    notes: list
    plan: dict | None = None
    seconds: float = 0.0
    variant: str = ''

    def fit(self, model, mode, camera, cls, subset='all'):
        return next((f for f in self.fits if f['model'] == model and f['mode'] == mode and f['camera'] == camera
                     and f['class'] == cls and f['subset'] == subset), None)

    def cameras(self) -> list:
        return sorted({f['camera'] for f in self.fits})

    def classes(self) -> list:
        return [c['name'] for c in self.settings.classes if any(f['class'] == c['name'] for f in self.fits)]


_TRIAL_KEYS = ('model', 'mode', 'rec', 'camera', 'cls', 'range', 'height', 'hit', 'score', 'cluster', 'x', 'y',
               'offaxis', 'slot', 'bearing', 'absent')


class _Cancelled(Exception):
    pass


def _heading(rcam: DE.Camera) -> float:
    return math.degrees(DE._heading(rcam)) if rcam.R is not None else 0.0


def _offaxis(rcam: DE.Camera, x, y) -> tuple[float, float]:
    """(off-axis angle, bearing from the camera) of a vehicle-frame point, degrees."""
    if rcam.t is None:
        return math.nan, math.nan
    b = _bearing(x - rcam.t[0], y - rcam.t[1])
    return abs((b - _heading(rcam) + 180.0) % 360.0 - 180.0), b


def _cam_key(cam: DE.Camera) -> tuple:
    """A camera's calibration as a key: the same camera in every drive of a sweep shares its views and rays."""
    arr = [np.asarray(a, float).round(9).tobytes() if a is not None else b'' for a in (cam.K, cam.D, cam.R, cam.t)]
    return (cam.tag, cam.width, cam.height, cam.lens, cam.max_theta, *arr)


def _view_rays(view: DE.View, rcam: DE.Camera):
    """(H × W × 3 vehicle-frame directions of the view's (x, y, 1) rays, |(x, y, 1)|)."""
    gy, gx = np.mgrid[0:view.h, 0:view.w].astype(float)
    d = np.stack([(gx - view.K[0, 2]) / view.K[0, 0], (gy - view.K[1, 2]) / view.K[1, 1], np.ones_like(gx)], -1)
    nrm = np.linalg.norm(d, axis=-1)
    dv = d @ (rcam.R @ view.R).T
    return dv, nrm


def _absent(rcam: DE.Camera, view, placements, present_pids, rng_img) -> list:
    """Sweep placements of this drive the image does not list: in view (centre inside the image and the lens's
    field) and not hidden (the range image at the centre's pixel not nearer than it) -> [(pid, class, range, px)]."""
    out = []
    if rcam.R is None or rcam.K is None:
        return out
    for pid, (p, cls) in placements.items():
        if pid in present_pids:
            continue
        e = (p.get('actor') or {}).get('expected') or {}
        h = float(e.get('rendered_height') or e.get('height') or cls['height'])
        pose = p.get('pose_vehicle') or {'x': p['x'], 'y': p['y'], 'z': 0.0}
        P = np.array([[pose['x'], pose['y'], pose['z'] + h / 2], [pose['x'], pose['y'], pose['z']],
                      [pose['x'], pose['y'], pose['z'] + h]])
        Pc = (P - rcam.t) @ rcam.R                         # camera optical frame
        if view is None:
            u, v, ok = DE.kb_project(rcam, Pc)
            W, H = rcam.width, rcam.height
        else:
            q = Pc @ view.R                                  # view optical (R: view -> camera)
            ok = q[:, 2] > 1e-6
            zz = np.where(ok, q[:, 2], 1.0)
            u = view.K[0, 0] * q[:, 0] / zz + view.K[0, 2]
            v = view.K[1, 1] * q[:, 1] / zz + view.K[1, 2]
            W, H = view.w, view.h
        if not (ok[0] and 0 <= u[0] < W and 0 <= v[0] < H):
            continue
        dist = float(np.linalg.norm(P[0] - rcam.t))
        if rng_img is not None:
            rr = float(rng_img[int(v[0]), int(u[0])])
            if 0 < rr < dist - (0.3 + 0.03 * dist):
                continue                                     # something nearer hides it
        px = float(np.hypot(u[2] - u[1], v[2] - v[1])) if ok[1] and ok[2] else 0.0
        out.append((pid, p['cls_index'], dist, px, pose['x'], pose['y']))
    return out


def _draw(img, g, mt, dets, classes, score, title=''):
    """Detections over ground truth: trials found (green), missed (orange), ignored (grey); detections at the
    operating point that match a trial (blue), false (magenta), other (thin grey)."""
    import cv2
    from .pngtext import put_text
    out = img.copy()
    th = max(1, int(round(out.shape[1] / 700)))
    for i in range(len(g['scls'])):
        x0, y0, x1, y1 = (int(round(v)) for v in g['box'][i])
        if g['scls'][i] < 0 and not g['trial'][i]:
            continue
        col = (150, 150, 150) if not g['trial'][i] else ((90, 180, 60) if mt['hit'][i] else (40, 110, 240))
        cv2.rectangle(out, (x0, y0), (x1 - 1, y1 - 1), col, th + (1 if g['trial'][i] else 0))
    fp = set(mt['fp'])
    tp = set(mt['tp_det'])
    for k, (b, s, lab) in enumerate(zip(dets['boxes'], dets['scores'], dets['labels'])):
        if s < score:
            continue
        col = (230, 160, 40) if k in tp else ((200, 60, 200) if k in fp else (170, 170, 170))
        x0, y0, x1, y1 = (int(round(v)) for v in b)
        cv2.rectangle(out, (x0, y0), (x1, y1), col, th)
        name = classes[int(lab)]['name'] if 0 <= lab < len(classes) else '?'
        put_text(out, f'{name} {s:.2f}', (x0 + 2, max(y0 - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.35 * th + 0.05, col,
                 1, cv2.LINE_AA)
    if title:
        put_text(out, title, (6, 16 * th), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * th, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _jpg(img, width=960) -> bytes:
    import cv2
    h, w = img.shape[:2]
    if w > width:
        img = cv2.resize(img, (width, int(round(h * width / w))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return bytes(buf) if ok else b''


def _load_models(st: FoundationSettings, detectors=None, depth=None):
    """{model id: detector} and the depth model (or None), loading what was not passed in."""
    names = [c['coco'] for c in st.classes]
    queries = [c['query'] for c in st.classes]
    dets = {}
    for mid in st.detectors:
        if detectors is not None and mid in detectors:
            dets[mid] = detectors[mid]
            continue
        if detectors is not None:
            continue
        if mid in DT.MODELS:
            dets[mid] = FM.YoloxAsQueries(names, DT.DetectorSettings.from_dict({'model': mid}))
        else:
            if not FM.allowed(mid, st.opt_in):
                raise ValueError(f'{FM.SPECS[mid].name}: {FM.SPECS[mid].licence_note}')
            dets[mid] = FM.OpenVocabDetector(mid, queries, FM.DetectorOptions(device=st.device,
                                                                              gpu_mem_gb=st.gpu_mem_gb))
    dm = None
    if depth is not None:
        dm = depth
    elif st.depth_model and detectors is None:
        if not FM.allowed(st.depth_model, st.opt_in):
            raise ValueError(f'{FM.SPECS[st.depth_model].name}: {FM.SPECS[st.depth_model].licence_note}')
        dm = FM.DepthModel(st.depth_model, st.device, st.gpu_mem_gb)
    return dets, dm


def _check_models(st: FoundationSettings, detectors=None, depth=None, verify=True):
    need_torch = any(m in FM.SPECS for m in st.detectors if detectors is None or m not in detectors) or (
        bool(st.depth_model) and depth is None and detectors is None)
    if need_torch:
        why = FM.torch_missing()
        if why:
            raise ValueError(why)
    if detectors is None:
        for mid in st.detectors:
            s = DT.status(mid) if mid in DT.MODELS else FM.status(mid, verify=verify)
            if not s['verified']:
                raise FileNotFoundError(s['text'] + ' Download it on Simulate ▸ Foundation models.')
        if st.depth_model and depth is None:
            s = FM.status(st.depth_model, verify=verify)
            if not s['verified']:
                raise FileNotFoundError(s['text'] + ' Download it on Simulate ▸ Foundation models.')


def _sweep_maps(rec: DE.Recording, pl: dict, classes) -> tuple[dict, dict]:
    """(actor id -> (placement id, class index), placement id -> (row, class)) of a sweep drive."""
    by_id = {p.id: p for p in pl['placements']}
    amap, prow = {}, {}
    for row in (rec.run or {}).get('actors') or []:
        p = by_id.get(row.get('id'))
        if p is None or row.get('actor_id') is None:
            continue
        ci = class_index(classes, p.cls)
        if ci < 0:
            continue
        amap[int(row['actor_id'])] = (p.id, ci)
        d = p.to_dict()
        d['pose_vehicle'] = row.get('pose_vehicle')
        d['cls_index'] = ci
        prow[p.id] = (d, classes[ci])
    return amap, prow


def run(project=None, engine=None, settings: FoundationSettings | None = None, variant=None, progress=None,
        cancel=None, detectors=None, depth=None, sim_settings=None, out_dir=None, python=None, runner=None,
        drives=None) -> FoundationResult | None:
    """The study. detectors: {model id: callable image -> {'boxes', 'scores', 'labels' (index into the classes)}} to
    use instead of loading the settings' models (tests), depth: a callable image -> depth with .kind 'metric' or
    'relative'. A sweep runs its drives first (sim_settings, out_dir, python, runner as core.critical_carla.run), or
    uses drives already made ([{'key', 'dir', 'seed', 'group'}] with plan in the settings' terms). Returns None when
    cancelled; ValueError for a missing input or torch, FileNotFoundError for a model not downloaded."""
    import cv2
    from . import groundtruth as GT
    from .regions import default_grid
    t0 = time.perf_counter()
    st = settings or settings_of(project)
    stop = cancel or (lambda: False)

    def prog(f, text=''):
        if progress is not None:
            progress(min(max(float(f), 0.0), 1.0), text)
    if not st.detectors and not st.depth_model:
        raise ValueError('Choose at least one detector or a depth model.')
    if not st.classes:
        raise ValueError('The class list is empty.')
    _check_models(st, detectors, depth)
    v = variant or (project.variant if project is not None else None)
    notes = []
    plan = None
    sources = []                                          # [(Recording, sweep maps or None)]
    if st.source == 'sweep':
        if project is None:
            raise ValueError('A sweep needs the project (its rig).')
        if drives is None:
            r = run_sweep(project, st, v, sim_settings, out_dir, python, runner,
                          lambda f, t='': prog(0.4 * f, t), cancel)
            if r is None:
                return None
            drives, plan = r
        else:
            plan = sweep_plan(project, st, v)
        notes += plan['notes']
        for d in drives:
            rec = DE.open_recording(d['dir'])
            rec.label = d['key']
            sources.append((rec, _sweep_maps(rec, plan, st.classes)))
    else:
        if not st.recordings:
            raise ValueError('Add a recording with ground truth (a CARLA drive), or choose the sweep.')
        for p in st.recordings:
            sources.append((DE.open_recording(p), None))
    base = 0.4 if st.source == 'sweep' else 0.0
    prog(base, 'Loading the models')
    dets, dm = _load_models(st, detectors, depth)
    infos = []
    for mid, d in dets.items():
        dd = d.describe() if hasattr(d, 'describe') else {'model': mid}
        dd['id'] = mid
        dd['score'] = st.score_of(mid)
        infos.append(dd)
    if dm is not None:
        dd = dm.describe() if hasattr(dm, 'describe') else {'model': st.depth_model or 'depth'}
        dd['id'] = st.depth_model or dd.get('model', 'depth')
        dd['kind'] = getattr(dm, 'kind', 'metric')
        infos.append(dd)
    device = FM.device_info(st.device, st.gpu_mem_gb) if FM.torch_missing() is None else \
        {'device': 'cpu', 'text': 'CPU (no torch: YOLOX through OpenCV)'}
    vst = st.view_settings()
    modes = st.modes()
    classes = st.classes
    rows = {k: [] for k in _TRIAL_KEYS}
    fdet = {}
    ms = {}
    examples = {}
    depth_acc = {}                                        # camera -> {'frames': [(n, s, d1)], 'regions'...}
    reg_acc = {}
    obj_acc = {}
    scale_info = []
    grid = default_grid(project) if project is not None else None
    if grid is None:
        from .regions import RegionGrid
        grid = RegionGrid()
    rig_objs = {}                                         # records: (src, slot, key) -> {...}
    jobs = []
    for ri, (rec, sw) in enumerate(sources):
        for tag, cam in sorted(rec.cameras.items()):
            if st.cameras and tag not in st.cameras:
                continue
            if sw is not None and tag not in st.sweep_cameras:
                continue
            frs = cam.frames[-max(st.sweep_frames, 1):] if sw is not None else cam.frames[::max(st.frame_step, 1)]
            if st.max_frames and sw is None:
                frs = frs[:st.max_frames]
            for fr in frs:
                jobs.append((ri, rec, sw, tag, cam, fr))
    if not jobs:
        raise ValueError('No camera frames to evaluate (check the cameras and the recordings).')
    views_of = {}
    rays_of = {}
    nj = len(jobs)
    t_inf = 0.0
    try:
        for j, (ri, rec, sw, tag, cam, fr) in enumerate(jobs):
            if stop():
                raise _Cancelled()
            prog(base + (1 - base) * 0.95 * j / nj, f'{rec.label or os.path.basename(rec.path)} · {tag} · frame '
                                                    f'{j + 1} of {nj}')
            img = cv2.imread(GT.path_of(rec.gt, fr['image']), cv2.IMREAD_COLOR)
            if img is None:
                notes.append(f'Cannot read {fr["image"]}: skipped.')
                continue
            try:
                rng = GT.load_range(rec.gt, fr) if fr.get('range') else None
            except OSError:
                rng = None
            need_inst = 'pinhole' in modes or dm is not None
            inst = sem = None
            if need_inst and fr.get('instance'):
                try:
                    inst = GT.load_instance(rec.gt, fr)
                except OSError:
                    inst = None
            if dm is not None and fr.get('semantic'):
                try:
                    sem = GT.load_semantic(rec.gt, fr)
                except OSError:
                    sem = None
            amap, prow = sw if sw is not None else (None, {})
            g = _objects(fr, classes, vst, amap)
            t_us = int(fr.get('timestamp') or 0)
            slot = int(round(t_us / 20000.0)) if sw is None else 0
            key_cam = _cam_key(cam)                         # drives of one rig share their views and rays
            if key_cam not in views_of:
                views_of[key_cam] = DE.make_views(cam, st.view_settings(cam)) \
                    if ('pinhole' in modes or dm is not None) else []
            views = views_of[key_cam]
            images = []                                     # (mode, view or None, image, objects, range image)
            if 'fisheye' in modes:
                images.append(('fisheye', None, img, g, rng))
            vimgs = []
            if views and inst is not None:
                for view in views:
                    vi = DE.remap(img, view)
                    vg = DE.view_gt(g, inst, view, vst)
                    has_pid = np.array([bool(q) for q in vg['pid']], bool)
                    vg['trial'] = (vg['scls'] >= 0) & (vg['ignore'] == DE.IGN_NONE) & (has_pid if sw is not None
                                                                                        else True)
                    vr = DE.remap(rng.astype(np.float32), view, nearest=True) if rng is not None else None
                    vimgs.append((view, vi, vg, vr))
                    if 'pinhole' in modes:
                        images.append(('pinhole', view, vi, vg, vr))
            # ---- detection
            for mid, det in dets.items():
                score = st.score_of(mid)
                per_obj = {}                                # (mode, cluster) -> trial row (pinhole: any view)
                for mode, view, im, gg, rr in images:
                    t1 = time.perf_counter()
                    out = det(im)
                    dt = time.perf_counter() - t1
                    t_inf += dt
                    ms.setdefault(mid, []).append(float(out.get('ms', dt * 1000.0)))
                    mt = match_image(gg, out, classes, score, st)
                    a = fdet.setdefault((mid, mode, tag), {'images': 0, 'false': 0, 'by_class': {}})
                    a['images'] += 1
                    a['false'] += len(mt['fp'])
                    for k in mt['fp']:
                        lab = int(out['labels'][k])
                        name = classes[lab]['name'] if 0 <= lab < len(classes) else '?'
                        a['by_class'][name] = a['by_class'].get(name, 0) + 1
                        _fp_record(rig_objs, ri, mid, mode, name, out['boxes'][k], rr, cam, view, grid)
                    for i in np.flatnonzero(gg['trial']):
                        ck = f'{ri}:{gg["pid"][i] or gg["key"][i]}'
                        xy = gg['xy'][i]
                        x_, y_ = (float(xy[0]), float(xy[1])) if np.isfinite(xy).all() else (math.nan, math.nan)
                        hpx = float(gg['height'][i])
                        row = per_obj.get((mode, ck))
                        if row is None or row.get('absent'):
                            off, brg = _offaxis(cam, x_, y_)
                            per_obj[(mode, ck)] = {
                                'cls': int(gg['scls'][i]), 'range': float(gg['range'][i]), 'height': hpx,
                                'hit': bool(mt['hit'][i]), 'score': float(mt['score'][i]), 'x': x_, 'y': y_,
                                'off': off, 'brg': brg, 'cluster': ck}
                        else:
                            row['hit'] |= bool(mt['hit'][i])
                            row['height'] = max(row['height'], hpx)
                            row['score'] = max(row['score'], float(mt['score'][i]))
                    if sw is not None:                       # placed, in view, not hidden, but not one pixel
                        present = {q for q in gg['pid'] if q}
                        for pid, ci, dist, px, px_x, px_y in _absent(cam, view, prow, present, rr):
                            ck = f'{ri}:{pid}'
                            if (mode, ck) in per_obj:
                                continue
                            off, brg = _offaxis(cam, px_x, px_y)
                            per_obj[(mode, ck)] = {'cls': ci, 'range': dist, 'height': px, 'hit': False,
                                                   'score': -math.inf, 'x': px_x, 'y': px_y, 'off': off, 'brg': brg,
                                                   'cluster': ck, 'absent': True}
                    ek = (mid, mode, tag)
                    ntr = int(np.count_nonzero(gg['trial']))
                    if ntr and (ek not in examples or ntr > examples[ek][0]):
                        cap = (f'{FM.model_label(mid)} · {tag}{"" if view is None else " " + view.name} · '
                               f'{rec.label} · {int(np.count_nonzero(mt["hit"] & gg["trial"]))} of {ntr} found')
                        examples[ek] = (ntr, {'model': mid, 'mode': mode, 'camera': tag, 'caption': cap,
                                              'jpg': _jpg(_draw(im, gg, mt, out, classes, score))})
                for (mode, ck), row in per_obj.items():
                    for k, val in (('model', mid), ('mode', mode), ('rec', ri), ('camera', tag), ('cls', row['cls']),
                                   ('range', row['range']), ('height', row['height']), ('hit', row['hit']),
                                   ('score', row['score']), ('cluster', ck), ('x', row['x']), ('y', row['y']),
                                   ('offaxis', row['off']), ('slot', slot), ('bearing', row['brg']),
                                   ('absent', bool(row.get('absent')))):
                        rows[k].append(val)
                    ro = rig_objs.setdefault(('obj', mid, mode, ri, slot, ck), {'cls': row['cls'], 'x': row['x'],
                                                                                'y': row['y'], 'hit': False})
                    ro['hit'] |= row['hit']
            # ---- depth
            if dm is not None:
                for view, vi, vg, vr in vimgs:
                    if vr is None:
                        continue
                    if key_cam + (view.name,) not in rays_of:
                        rays_of[key_cam + (view.name,)] = _view_rays(view, cam)
                    dv, nrm = rays_of[key_cam + (view.name,)]
                    gt = np.where(vr > 0, vr / nrm, np.nan).astype(np.float32)
                    valid = np.isfinite(gt)
                    if sem is not None:
                        vs = DE.remap(sem, view, nearest=True)
                        valid &= vs != GT.EGO_CLASS
                    t1 = time.perf_counter()
                    pred = dm(vi)
                    t_inf += time.perf_counter() - t1
                    ms.setdefault('depth', []).append((time.perf_counter() - t1) * 1000.0)
                    if getattr(dm, 'kind', 'metric') == 'relative':
                        pred, inf_ = ground_scale(pred, dv, float(cam.t[2]), valid, seed=st.seed)
                        inf_.update(camera=tag, view=view.name)
                        scale_info.append(inf_)
                        if pred is None:
                            continue
                    nb, sb, db = depth_bins(pred, gt, valid, st.depth_edges)
                    acc = depth_acc.setdefault(tag, [])
                    acc.append((nb, sb, db))
                    ok, rel, d1 = depth_scores(pred, gt, valid)
                    P = cam.t + dv[ok] * gt[ok][:, None]
                    s_, r_ = grid.assign(P[:, 0], P[:, 1])
                    good = r_ >= 0
                    rid = s_[good] * grid.n_ranges + r_[good]
                    nreg = grid.n_regions
                    ra = reg_acc.setdefault(tag, [np.zeros(nreg), np.zeros(nreg), np.zeros(nreg)])
                    ra[0] += np.bincount(rid, minlength=nreg)[:nreg]
                    ra[1] += np.bincount(rid, weights=rel[good], minlength=nreg)[:nreg]
                    ra[2] += np.bincount(rid, weights=d1[good].astype(float), minlength=nreg)[:nreg]
                    vinst = DE.remap(inst, view, nearest=True) if inst is not None else None
                    if vinst is not None:
                        for i, oid in enumerate(vg['id']):
                            m = (vinst == oid) & valid & np.isfinite(pred) & (pred > 0)
                            if m.sum() < 8:
                                continue
                            pg = float(np.median(gt[m]))
                            pp = float(np.median(pred[m]))
                            obj_acc.setdefault(tag, []).append((float(vg['range'][i]), abs(pp - pg) / pg,
                                                                int(vg['scls'][i])))
    except _Cancelled:
        _close(dets, dm, detectors, depth)
        return None
    _close(dets, dm, detectors, depth)
    if stop():
        return None
    prog(0.96, 'Fitting the ranges')
    trials = {k: np.array(v, dtype=object if k in ('model', 'mode', 'camera', 'cluster') else float)
              for k, v in rows.items()}
    for k in ('rec', 'cls', 'slot'):
        trials[k] = trials[k].astype(int) if len(trials[k]) else np.zeros(0, int)
    for k in ('hit', 'absent'):
        trials[k] = trials[k].astype(bool) if len(trials[k]) else np.zeros(0, bool)
    fits = _fits(trials, st, classes, stop)
    if fits is None:
        return None
    geo = _geometric(project, engine, sources, trials, st, classes, fits)
    for f in fits:
        gk = (f['camera'], f['class'], f['subset'])
        gr = geo.get(gk)
        f['geometric_m'] = gr['range_m'] if gr else None
        for t, key in ((0, 'ratio50'), (1, 'ratio90')):
            x = f['range']['x'][t]
            inside = not (f['range']['beyond'][t] or f['range']['below'][t])     # no ratio of an extrapolation
            f[key] = (x / gr['range_m']) if (gr and gr['range_m'] and x is not None and inside) else None
    dres = _depth_result(st, depth_acc, reg_acc, obj_acc, scale_info, grid, classes, infos, dm)
    fd = [{'model': k[0], 'mode': k[1], 'camera': k[2], 'images': a['images'], 'false': a['false'],
           'per_image': a['false'] / max(a['images'], 1), 'by_class': a['by_class']} for k, a in sorted(fdet.items())]
    timing = {'seconds_inference': round(t_inf, 2), 'images': len(jobs),
              'ms_per_image': {k: {'median': float(np.median(v)), 'n': len(v)} for k, v in ms.items()}}
    infos_src = []
    for rec, sw in sources:
        d = rec.info()
        d['kind'] = 'sweep drive' if sw is not None else 'recording'
        d['exposure'] = recording_exposure(rec.path)
        infos_src.append(d)
    notes.append(exposure_note([d['exposure'] for d in infos_src]))
    if any(not FM.allowed(m, False) for m in list(dets) + ([st.depth_model] if dm is not None and st.depth_model
                                                            else []) if m in FM.SPECS):
        notes.append('Models whose licence does not allow commercial use were run on your opt-in (see Models).')
    res = FoundationResult(st, st.source, infos_src, infos, device, trials, fits, {f'{k[0]}|{k[1]}|{k[2]}': v
                                                                                   for k, v in geo.items()},
                           dres, fd, [e for _, e in examples.values()][:MAX_EXAMPLES], [], timing, notes, None,
                           variant=getattr(v, 'name', '') if v is not None else '')
    if plan is not None:
        res.plan = {'placements': [p.to_dict() for p in plan['placements']], 'groups': plan['groups'],
                    'drives': [dict(d) for d in drives], 'notes': plan['notes']}
    res.records = records(res, project, sources, grid, rig_objs)
    res.seconds = time.perf_counter() - t0
    prog(1.0, 'Done')
    return res


def _close(dets, dm, detectors, depth):
    if detectors is None:
        for d in dets.values():
            if hasattr(d, 'close'):
                d.close()
    if depth is None and dm is not None and hasattr(dm, 'close'):
        dm.close()


def _fp_record(rig_objs, ri, mid, mode, name, box, rr, cam, view, grid):
    """A false detection placed in a region by the range image inside its box (its central half's median)."""
    if rr is None or cam.R is None:
        return
    r = DE._det_range(np.asarray(box, float).reshape(1, 4), rr)[0]
    if not np.isfinite(r):
        return
    u, v = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
    if view is None:
        d, ok = DE.kb_unproject(cam, np.array([u]), np.array([v]))
        if not ok[0]:
            return
        dc = d[0]
    else:
        q = np.array([(u - view.K[0, 2]) / view.K[0, 0], (v - view.K[1, 2]) / view.K[1, 1], 1.0])
        dc = view.R @ (q / np.linalg.norm(q))
    P = cam.t + cam.R @ dc * r
    s, rb = grid.assign(P[0], P[1])
    if int(rb) < 0:
        return
    k = ('fp', mid, mode, ri, name, grid.region_id(int(s), int(rb)))
    rig_objs[k] = rig_objs.get(k, 0) + 1


def _fits(trials, st, classes, stop) -> list | None:
    out = []
    if not len(trials['hit']):
        return out
    keys = sorted({(m, md, c, int(k)) for m, md, c, k in zip(trials['model'], trials['mode'], trials['camera'],
                                                              trials['cls'])}, key=str)
    for i, (mid, mode, cam, ci) in enumerate(keys):
        if stop():
            return None
        sel = (trials['model'] == mid) & (trials['mode'] == mode) & (trials['camera'] == cam) & (trials['cls'] == ci)
        for subset in ('all', 'axis'):
            s = sel & (trials['offaxis'] <= AXIS_DEG) if subset == 'axis' else sel
            if subset == 'axis' and (s.sum() < 6 or s.sum() == sel.sum()):
                continue
            rf = detect_fit(trials['range'][s], trials['hit'][s], trials['cluster'][s], False, RANGE_BIN_RATIO,
                            st.bootstrap, st.confidence, st.seed + i)
            hf = detect_fit(trials['height'][s], trials['hit'][s], trials['cluster'][s], True, HEIGHT_BIN_RATIO,
                            st.bootstrap, st.confidence, st.seed + i)
            out.append({'model': mid, 'mode': mode, 'camera': cam, 'class': classes[ci]['name'], 'subset': subset,
                        'n': rf['n'], 'k': rf['k'], 'range': rf, 'height': hf,
                        'bearing_deg': float(np.nanmedian(trials['bearing'][s]))
                        if np.isfinite(trials['bearing'][s]).any() else None,
                        'score': st.score_of(mid)})
    return out


def _geometric(project, engine, sources, trials, st, classes, fits) -> dict:
    """(camera, class, subset) -> {'range_m', 'bearing_deg', 'min_px', 'how'}."""
    out = {}
    pcams = {c.tag: c for c in project.variant.cameras} if project is not None else {}
    pe_min = {}
    if project is not None:
        from . import perception as PE
        ps = PE.PerceptionSettings.from_dict(getattr(project.analysis, 'perception', None))
        pe_min = {c['name'].lower(): c['min_px'] for c in ps.classes}
    rcams = {}
    for rec, _ in sources:
        for tag, c in rec.cameras.items():
            rcams.setdefault(tag, c)
    for f in fits:
        k = (f['camera'], f['class'], f['subset'])
        if k in out or f['bearing_deg'] is None or not math.isfinite(f['bearing_deg']):
            continue
        cls = dict(classes[class_index(classes, f['class'])])
        cls['min_px'] = float(pe_min.get(cls['name'].lower(), cls['min_px']))
        rc = rcams.get(f['camera'])
        pc = pcams.get(f['camera'])
        if pc is not None and rc is not None:
            from .lens import resolve
            intr = resolve(project.module_for(pc), project.lens_for(pc))
            scale = rc.width / max(intr.width, 1)
            cube = None
            if engine is not None:
                try:
                    engine.vehicle_geom(project)          # the body the cube is built from
                    cube = engine.cube(pc.pos())
                except Exception:                     # noqa: BLE001 (no vehicle: no body occlusion)
                    cube = None
            r = geometric_range(project, pc, cls, f['bearing_deg'], scale, cube)
            how = (f'core.perception.object_pixels, the project\'s camera {pc.tag} (image scale {scale:.2f}'
                   f'{", body occlusion" if cube is not None else ""})')
        elif rc is not None:
            r = geometric_range_kb(rc, cls, f['bearing_deg'])
            how = 'the recording\'s own calibration (Kannala-Brandt): the project has no camera of this tag'
        else:
            continue
        out[k] = {'range_m': r, 'bearing_deg': f['bearing_deg'], 'min_px': cls['min_px'], 'how': how}
    return out


def _depth_result(st, depth_acc, reg_acc, obj_acc, scale_info, grid, classes, infos, dm) -> dict:
    if dm is None:
        return {}
    labels = DE.bin_labels(st.depth_edges)
    rng = np.random.default_rng(st.seed)
    cams = {}
    for tag, frames in sorted(depth_acc.items()):
        N = np.array([f[0] for f in frames])
        S = np.array([f[1] for f in frames])
        D = np.array([f[2] for f in frames])
        n, s, d = N.sum(0), S.sum(0), D.sum(0)
        bins = []
        for i in range(len(n)):
            bins.append({'bin': labels[i], 'n_px': int(n[i]), 'abs_rel': float(s[i] / n[i]) if n[i] else None,
                         'delta1': float(d[i] / n[i]) if n[i] else None})
        use, cont = usable_range(st.depth_edges, n, s, st.abs_rel_max, st.depth_min_px)
        ci = None
        if st.bootstrap and len(frames) >= 5:
            vals = []
            for _ in range(min(int(st.bootstrap), 1000)):
                pick = rng.integers(0, len(frames), len(frames))
                u, _ = usable_range(st.depth_edges, N[pick].sum(0), S[pick].sum(0), st.abs_rel_max, st.depth_min_px)
                vals.append(u if math.isfinite(u) else 1e9)
            a = (1 - st.confidence) / 2
            q = np.quantile(vals, [a, 1 - a])
            ci = [float(q[0]) if q[0] < 1e8 else None, float(q[1]) if q[1] < 1e8 else None]
        ob = []
        rows = obj_acc.get(tag, [])
        if rows:
            r = np.array([x[0] for x in rows])
            e = np.array([x[1] for x in rows])
            b = DE.bin_of(r, st.depth_edges)
            for i in range(len(st.depth_edges) + 1):
                m = b == i
                ob.append({'bin': labels[i], 'n': int(m.sum()),
                           'abs_rel_median': float(np.median(e[m])) if m.any() else None})
        regs = {}
        ra = reg_acc.get(tag)
        if ra is not None:
            for sct, rb, rid, name in grid.regions():
                k = sct * grid.n_ranges + rb
                if ra[0][k] >= st.depth_min_px:
                    regs[rid] = {'name': name, 'n_px': int(ra[0][k]), 'abs_rel': float(ra[1][k] / ra[0][k]),
                                 'delta1': float(ra[2][k] / ra[0][k])}
        sc = [x for x in scale_info if x.get('camera') == tag]
        cams[tag] = {'bins': bins, 'usable_m': use, 'usable_ci': ci, 'contiguous_m': cont, 'frames': len(frames),
                     'objects': ob, 'regions': regs,
                     'ground_fit': {'views': len(sc), 'failed': sum(1 for x in sc if x.get('alpha') is None),
                                    'inliers_median': float(np.median([x['inliers'] for x in sc if x.get('alpha')]))
                                    if any(x.get('alpha') for x in sc) else None} if sc else None}
    dinfo = next((i for i in infos if i.get('kind') in ('metric', 'relative')), {})
    return {'model': dinfo.get('id'), 'name': dinfo.get('name', dinfo.get('id')), 'kind': dinfo.get('kind'),
            'scaled': 'per frame from the ground plane' if dinfo.get('kind') == 'relative' else 'metric output',
            'edges': list(st.depth_edges), 'abs_rel_max': st.abs_rel_max, 'cameras': cams}


# =========================================================================== perception records
def _canon(o) -> bytes:
    """core.provenance's canonical JSON (the hashes perf_model uses), else sorted compact JSON."""
    try:
        from .provenance import canonical
        return canonical(o)
    except ImportError:                                # pragma: no cover — provenance ships with the app
        return json.dumps(o, sort_keys=True, separators=(',', ':'), ensure_ascii=False, default=str).encode()


def _sha(o) -> str:
    return hashlib.sha256(_canon(o)).hexdigest()


RECORD_ID_EXCLUDE = ('id', 'created', 'provenance')


def record_id(doc: dict) -> str:
    """A record's id: perf_model.record_id when it is there, else the same rule (sha256 of the canonical JSON
    without id, created and provenance: a re-run gives the same id)."""
    try:
        from .perf_model import record_id as rid
        return rid(doc)
    except ImportError:
        return _sha({k: v for k, v in doc.items() if k not in RECORD_ID_EXCLUDE})


def rig_block(project, rec: DE.Recording | None = None) -> dict:
    """The record's rig block: from the recording's sim_rig.json when it has one, else the project's variant."""
    rig = None
    if rec is not None:
        rig = DE._load_json(os.path.join(rec.path, 'rig', 'sim_rig.json'))
    if isinstance(rig, dict) and rig.get('cameras'):
        veh = rig.get('vehicle') or {}
        meas = veh.get('measured') or {}
        cams = []
        for c in rig['cameras']:
            pv = c.get('pose_vehicle') or {}
            im = c.get('image') or {}
            K = im.get('K') or [[0, 0, 0], [0, 0, 0]]
            w, h = int(im.get('width') or 0), int(im.get('height') or 0)
            mt = im.get('max_theta_deg')
            cams.append({'tag': c.get('tag'), 'module': c.get('module', ''), 'lens': c.get('lens', ''), 'width': w,
                         'height': h, **{k: float(pv.get(k, 0.0)) for k in ('x', 'y', 'z', 'yaw', 'pitch', 'roll')},
                         'hfov_deg': round(2 * min(float(mt or 90.0), 90.0), 3) if mt else None,
                         'vfov_deg': None, 'max_theta_deg': mt, 'projection': im.get('lens_projection', '')})
            if K and K[0][0]:
                cams[-1]['f_px'] = float(K[0][0])
        sensors = []
        for kind in ('lidars', 'radars', 'imus', 'gnss'):
            for s in rig.get(kind) or []:
                pv = s.get('pose_vehicle') or {}
                sensors.append({'tag': s.get('tag'), 'kind': kind.rstrip('s'), 'model': s.get('model', ''),
                                **{k: float(pv.get(k, 0.0)) for k in ('x', 'y', 'z', 'yaw', 'pitch', 'roll')}})
        return {'vehicle': {'name': veh.get('name', ''), 'length_m': meas.get('Length'), 'width_m': meas.get('Width'),
                            'height_m': meas.get('Height')}, 'cameras': cams, 'sensors': sensors}
    from .lens import resolve
    v = project.variant
    cams = []
    for c in v.cameras:
        if not c.visible:
            continue
        intr = resolve(project.module_for(c), project.lens_for(c))
        mt = math.degrees(float(getattr(intr, 'max_theta', math.pi / 2)))
        cams.append({'tag': c.tag, 'module': c.module_id, 'lens': c.lens_id, 'width': int(intr.width),
                     'height': int(intr.height), 'x': float(c.x), 'y': float(c.y), 'z': float(c.z),
                     'yaw': float(c.yaw), 'pitch': float(c.pitch), 'roll': float(c.roll),
                     'hfov_deg': None, 'vfov_deg': None, 'max_theta_deg': round(mt, 3),
                     'projection': str(intr.projection)})
    sensors = [{'tag': s.tag, 'kind': s.kind, 'model': getattr(s, 'model_id', ''), 'x': float(s.x), 'y': float(s.y),
                'z': float(s.z), 'yaw': float(s.yaw), 'pitch': float(s.pitch), 'roll': float(s.roll)}
               for s in v.sensors if getattr(s, 'visible', True)]
    veh = project.vehicle
    try:
        vv = veh.mesh.vertices
        L, W, H = (float(vv[:, i].max() - vv[:, i].min()) for i in range(3))
    except Exception:                                  # noqa: BLE001
        L = W = H = None
    return {'vehicle': {'name': getattr(veh, 'name', ''), 'length_m': L, 'width_m': W, 'height_m': H},
            'cameras': cams, 'sensors': sensors}


def records(res: FoundationResult, project, sources, grid, rig_objs) -> list:
    """vantage.perception_record/1 per detector and image kind and scene set (a recording, or the whole sweep): an
    object at an instant counts once for the rig (detected when any camera detects it), in the region of its
    ground-truth box centre; false detections in the region the range image puts them."""
    from .. import __version__
    try:
        from . import provenance as PV
        prov = PV.block(project, None, study=RECORD_SCHEMA) if project is not None else None
    except Exception:                                  # noqa: BLE001 (a record without provenance is still a record)
        prov = None
    if isinstance(prov, dict):                         # no paths in a record: the project's digests, not its file
        prov = {k: v for k, v in prov.items() if k not in ('settings', 'output', 'command', 'inputs', 'cwd',
                                                           'created_utc')}   # (the record has its own time)
        if isinstance(prov.get('project'), dict):
            prov['project'] = {k: v for k, v in prov['project'].items() if k != 'file'}
    st = res.settings
    out = []
    groups = {}
    for k, val in rig_objs.items():
        if k[0] == 'obj':
            _, mid, mode, ri, slot, cluster = k
            sk = 0 if res.source == 'sweep' else ri
            groups.setdefault((mid, mode, sk), {'obj': [], 'fp': []})['obj'].append(val)
        else:
            _, mid, mode, ri, name, rid = k
            sk = 0 if res.source == 'sweep' else ri
            groups.setdefault((mid, mode, sk), {'obj': [], 'fp': []})['fp'].append((name, rid, val))
    for (mid, mode, sk), g in sorted(groups.items(), key=str):
        metrics = {}
        for o in g['obj']:
            if not (math.isfinite(o['x']) and math.isfinite(o['y'])):
                continue
            s, r = grid.assign(o['x'], o['y'])
            if int(r) < 0:
                continue
            name = st.classes[o['cls']]['name']
            m = metrics.setdefault(name, {}).setdefault(grid.region_id(int(s), int(r)),
                                                        {'n_gt': 0, 'tp': 0, 'fp': 0, 'fn': 0})
            m['n_gt'] += 1
            m['tp' if o['hit'] else 'fn'] += 1
        for name, rid, n in g['fp']:
            m = metrics.setdefault(name, {}).setdefault(rid, {'n_gt': 0, 'tp': 0, 'fp': 0, 'fn': 0})
            m['fp'] += n
        for cm in metrics.values():
            for m in cm.values():
                m['recall'] = round(m['tp'] / m['n_gt'], 6) if m['n_gt'] else None
                m['precision'] = round(m['tp'] / (m['tp'] + m['fp']), 6) if (m['tp'] + m['fp']) else None
                m['ap'] = None
                m['ate_m'] = None
        rec, _ = sources[sk] if res.source != 'sweep' else sources[0]
        if res.source == 'sweep':
            run0 = rec.run or {}
            sc0 = run0.get('scene') or {}
            scene = {'kind': 'carla', 'town': sc0.get('town') or st.town, 'scenario': 'foundation_range_sweep',
                     'seed': int(st.seeds[0]) if st.seeds else None, 'weather': sc0.get('weather') or st.weather,
                     'duration_s': st.duration_s, 'frames': len(sources)}
            scene['scene_hash'] = _sha({'placements': (res.plan or {}).get('placements'), 'seeds': st.seeds,
                                        'town': scene['town'], 'weather': scene['weather']})
        else:
            scene = {'kind': 'carla', 'town': rec.town, 'scenario': rec.scenario or 'free_drive', 'seed': rec.seed,
                     'weather': rec.weather, 'duration_s': rec.duration_s,
                     'frames': sum(len(c.frames) for c in rec.cameras.values())}
            scene['scene_hash'] = _sha({k: scene[k] for k in ('town', 'scenario', 'seed', 'weather', 'duration_s',
                                                              'frames')})
        rig = rig_block(project, rec) if (project is not None or rec is not None) else {}
        det_info = next((i for i in res.models if i.get('id') == mid), {})
        sett = {'mode': mode, 'score': st.score_of(mid), 'iou': st.iou, 'iou_small': st.iou_small,
                'small_px': st.small_px, 'max_occluded': st.max_occluded, 'max_truncated': st.max_truncated,
                'queries': [c['query'] for c in st.classes]}
        if mode == 'pinhole':
            sett.update(view_fov_deg=st.view_fov_deg, view_pitch_deg=st.view_pitch_deg, view_yaws=st.view_yaws,
                        view_width=st.view_width or 'focal length of the camera')
        doc = {'schema': RECORD_SCHEMA, 'vantage': __version__,
               'source': 'carla', 'scene': scene,
               'detector': {'id': mid, 'backend': f'external:foundation-2d-{mode}',
                            'version': det_info.get('revision') or det_info.get('sha256') or '',
                            'settings_hash': _sha(sett)},
               'rig': rig, 'rig_hash': _sha(rig), 'grid': grid.to_dict(), 'metrics': metrics,
               'provenance': prov}
        doc['id'] = record_id(doc)
        from datetime import datetime, timezone
        doc['created'] = datetime.now(timezone.utc).isoformat(timespec='seconds')
        out.append(doc)
    return out


def store_records(res: FoundationResult, out_dir=None, store=None) -> dict:
    """Hand the records to core.perf_model.add_record (store: its store or a path; None: the user's) when that module
    is there; else write them as JSON into out_dir. {'stored', 'duplicates', 'errors', 'written', 'how'}."""
    try:
        from . import perf_model as PM
        add = getattr(PM, 'add_record')
    except (ImportError, AttributeError):
        add = None
    if add is not None:
        n = dup = 0
        errors = []
        for r in res.records:
            out = add(r, store)
            n += bool(out.get('added'))
            dup += bool(out.get('duplicate'))
            errors += list(out.get('errors') or [])
        return {'stored': n, 'duplicates': dup, 'errors': errors, 'written': [], 'how': 'core.perf_model.add_record'}
    paths = []
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        for r in res.records:
            p = os.path.join(out_dir, f'perception_record-{r["id"][:16]}.json')
            with open(p, 'w', encoding='utf-8') as fh:
                json.dump(r, fh, indent=1)
            paths.append(p)
    return {'stored': 0, 'duplicates': 0, 'errors': [], 'written': paths,
            'how': 'JSON files (core.perf_model is not in this build)'}


# =========================================================================== output
def _r(v, nd=3):
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    return round(f, nd) if math.isfinite(f) else None


def _ci(c, nd=2):
    return None if not c else [_r(c[0], nd), _r(c[1], nd)]


def range_text(f: dict, t: int = 0) -> str:
    """'34.2 m [29.8–39.0]', '> 100 m' (beyond the farthest tested), '< 4 m', '—'."""
    rf = f['range']
    x = rf['x'][t]
    if rf.get('flat'):
        return 'no fall-off'
    if rf['beyond'][t] and rf['tested']:
        return f'> {rf["tested"][1]:.0f} m'
    if rf['below'][t] and rf['tested']:
        return f'< {rf["tested"][0]:.0f} m'
    if x is None:
        return '—'
    s = f'{x:.1f} m'
    c = rf['ci'][t]
    if c:
        s += f'  [{"0" if c[0] is None else f"{c[0]:.1f}"}–{"∞" if c[1] is None else f"{c[1]:.1f}"}]'
    elif rf.get('clusters', 0) < MIN_CLUSTERS:
        s += f'  (only {rf.get("clusters", 0)} objects: no interval)'
    return s


def _fit_doc(f: dict) -> dict:
    rf, hf = f['range'], f['height']
    return {'model': f['model'], 'mode': f['mode'], 'camera': f['camera'], 'class': f['class'], 'subset': f['subset'],
            'trials': f['n'], 'detected': f['k'], 'score': f['score'],
            'r50_m': _r(rf['x'][0], 2), 'r50_ci_m': _ci(rf['ci'][0]), 'r90_m': _r(rf['x'][1], 2),
            'r90_ci_m': _ci(rf['ci'][1]), 'r50_beyond': rf['beyond'][0], 'r90_beyond': rf['beyond'][1],
            'r50_below': rf['below'][0], 'r90_below': rf['below'][1],
            'tested_m': [_r(x, 2) for x in rf['tested']] if rf['tested'] else None,
            'objects': rf.get('clusters'), 'no_fall_off': bool(rf.get('flat')),
            'fit': {'a': _r(rf['a'], 5), 'b': _r(rf['b'], 5), 'separated': rf['separated'],
                    'model': 'P = 1 / (1 + exp(-(a + b·log10(1/range))))'},
            'h50_px': _r(hf['x'][0], 1), 'h50_ci_px': _ci(hf['ci'][0], 1), 'h90_px': _r(hf['x'][1], 1),
            'h90_ci_px': _ci(hf['ci'][1], 1), 'bearing_deg': _r(f['bearing_deg'], 1),
            'geometric_m': _r(f.get('geometric_m'), 2), 'r50_over_geometric': _r(f.get('ratio50'), 3),
            'r90_over_geometric': _r(f.get('ratio90'), 3),
            'bins': [{'range_m': _r(b['x'], 2), 'n': b['n'], 'detected': b['k'], 'p': _r(b['p'], 4),
                      'wilson': [_r(b['lo'], 4), _r(b['hi'], 4)]} for b in rf['bins']]}


def summary(res: FoundationResult) -> dict:
    """Compact figures: per camera, class and model R50 / R90 (main image kind, all trials), the geometric range and
    the usable depth per camera."""
    rows = []
    for f in res.fits:
        if f['subset'] != 'all':
            continue
        rows.append({'model': f['model'], 'mode': f['mode'], 'camera': f['camera'], 'class': f['class'],
                     'trials': f['n'], 'r50': range_text(f, 0), 'r90': range_text(f, 1),
                     'geometric_m': _r(f.get('geometric_m'), 1), 'r50_over_geometric': _r(f.get('ratio50'), 2)})
    dep = {}
    for tag, c in (res.depth.get('cameras') or {}).items():
        dep[tag] = {'usable_m': _r(c['usable_m'], 1), 'contiguous_m': _r(c['contiguous_m'], 1)}
    return {'schema': SCHEMA, 'source': res.source, 'recordings': len(res.recordings),
            'models': [m.get('id') for m in res.models], 'device': res.device.get('text'), 'ranges': rows,
            'depth': {'model': res.depth.get('model'), 'abs_rel_max': res.depth.get('abs_rel_max'), 'cameras': dep},
            'records': len(res.records), 'seconds': round(res.seconds, 1)}


def to_doc(res: FoundationResult, project=None) -> dict:
    st = res.settings
    dep = copy.deepcopy(res.depth) if res.depth else {}
    for c in (dep.get('cameras') or {}).values():
        for k in ('usable_m', 'contiguous_m'):
            c[k] = 'beyond the last bin' if c[k] == math.inf else _r(c[k], 2)
    return {'schema': SCHEMA, 'project': getattr(project, 'name', None), 'variant': res.variant,
            'source': res.source, 'units': {'range': 'm (camera to the object\'s box centre)', 'height': 'px',
                                            'depth': 'm (planar)'},
            'frame': FRAME, 'settings': st.to_dict(), 'recordings': res.recordings, 'models': res.models,
            'device': res.device, 'fits': [_fit_doc(f) for f in res.fits],
            'geometric': res.geometric, 'depth': dep, 'false_detections': res.false_det,
            'timing': res.timing, 'notes': res.notes, 'plan': res.plan, 'records': len(res.records),
            'assumptions': ASSUMPTIONS, 'limits': LIMITS_TEXT, 'method': METHOD, 'references': list(REFERENCES),
            'seconds': _r(res.seconds, 2)}


FIT_COLS = ('model', 'mode', 'camera', 'class', 'subset', 'trials', 'detected', 'score', 'r50_m', 'r50_lo', 'r50_hi',
            'r90_m', 'r90_lo', 'r90_hi', 'r50_beyond', 'r90_beyond', 'h50_px', 'h90_px', 'bearing_deg',
            'geometric_m', 'r50_over_geometric', 'r90_over_geometric')


def to_csv(res: FoundationResult) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(FIT_COLS)
    for f in res.fits:
        d = _fit_doc(f)
        c50, c90 = d['r50_ci_m'] or [None, None], d['r90_ci_m'] or [None, None]
        w.writerow([d['model'], d['mode'], d['camera'], d['class'], d['subset'], d['trials'], d['detected'],
                    d['score'], d['r50_m'], c50[0], c50[1], d['r90_m'], c90[0], c90[1], d['r50_beyond'],
                    d['r90_beyond'], d['h50_px'], d['h90_px'], d['bearing_deg'], d['geometric_m'],
                    d['r50_over_geometric'], d['r90_over_geometric']])
    return buf.getvalue()


def depth_csv(res: FoundationResult) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(('camera', 'bin', 'n_px', 'abs_rel', 'delta1'))
    for tag, c in (res.depth.get('cameras') or {}).items():
        for b in c['bins']:
            w.writerow((tag, b['bin'], b['n_px'], _r(b['abs_rel'], 5), _r(b['delta1'], 5)))
    return buf.getvalue()


def trials_csv(res: FoundationResult) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    T = res.trials
    w.writerow(('model', 'mode', 'recording', 'camera', 'class', 'range_m', 'height_px', 'detected', 'score',
                'object', 'x', 'y', 'off_axis_deg', 'no_pixels'))
    for i in range(len(T['hit'])):
        w.writerow((T['model'][i], T['mode'][i], int(T['rec'][i]), T['camera'][i],
                    res.settings.classes[int(T['cls'][i])]['name'], _r(T['range'][i], 3), _r(T['height'][i], 2),
                    int(bool(T['hit'][i])), _r(T['score'][i], 4) if np.isfinite(T['score'][i]) else '',
                    T['cluster'][i], _r(T['x'][i], 3), _r(T['y'][i], 3), _r(T['offaxis'][i], 2),
                    int(bool(T['absent'][i]))))
    return buf.getvalue()


# ---------------------------------------------------------------- charts (PNG)
_LIGHT = {'bg': (255, 255, 255), 'ink': (40, 40, 40), 'muted': (130, 130, 130), 'hair': (225, 225, 225)}
_DARK = {'bg': (32, 30, 28), 'ink': (225, 225, 225), 'muted': (150, 150, 150), 'hair': (70, 70, 70)}
SERIES = ((200, 120, 40), (40, 140, 230), (70, 170, 70), (170, 70, 170), (40, 60, 210), (150, 150, 30))


def _axes(img, x0, y0, x1, y1, xlo, xhi, pal, xlabel, ylabel, log=True):
    import cv2
    from .pngtext import put_text

    def X(v):
        f = (math.log(v) - math.log(xlo)) / (math.log(xhi) - math.log(xlo)) if log else (v - xlo) / (xhi - xlo)
        return int(round(x0 + f * (x1 - x0)))

    def Y(p):
        return int(round(y1 - p * (y1 - y0)))
    for p in (0.0, 0.25, 0.5, 0.75, 0.9, 1.0):
        cv2.line(img, (x0, Y(p)), (x1, Y(p)), pal['hair'], 1)
        put_text(img, f'{p * 100:.0f}%', (x0 - 40, Y(p) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, pal['muted'], 1,
                 cv2.LINE_AA)
    ticks = [t for t in (1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 300) if xlo <= t <= xhi]
    for t in ticks:
        cv2.line(img, (X(t), y0), (X(t), y1), pal['hair'], 1)
        put_text(img, f'{t:g}', (X(t) - 8, y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, pal['muted'], 1, cv2.LINE_AA)
    cv2.rectangle(img, (x0, y0), (x1, y1), pal['muted'], 1)
    put_text(img, xlabel, ((x0 + x1) // 2 - 40, y1 + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.42, pal['ink'], 1, cv2.LINE_AA)
    put_text(img, ylabel, (x0, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, pal['ink'], 1, cv2.LINE_AA)
    return X, Y


def pdetect_png(res: FoundationResult, camera: str, cls: str, mode: str | None = None, dark=False,
                size=(760, 380)) -> bytes:
    """P(detect) against range for one camera and class: per model the binned share detected (Wilson whiskers) and
    the fitted curve, R50 marked; the geometric range as a dashed line."""
    import cv2
    from .pngtext import put_text
    pal = _DARK if dark else _LIGHT
    mode = mode or res.settings.modes()[0]
    W, H = size
    img = np.full((H, W, 3), pal['bg'], np.uint8)
    fs = [f for f in res.fits if f['camera'] == camera and f['class'] == cls and f['mode'] == mode and
          f['subset'] == 'all']
    put_text(img, f'{camera} · {cls} · {MODE_LABEL.get(mode, mode)}', (12, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
             pal['ink'], 1, cv2.LINE_AA)
    if not fs:
        put_text(img, 'No trials for this camera and class.', (60, H // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                 pal['muted'], 1, cv2.LINE_AA)
        return bytes(cv2.imencode('.png', img)[1])
    xs = [b['x'] for f in fs for b in f['range']['bins']]
    if not xs:
        put_text(img, 'Too few trials to plot.', (60, H // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, pal['muted'], 1,
                 cv2.LINE_AA)
        return bytes(cv2.imencode('.png', img)[1])
    geo = next((f.get('geometric_m') for f in fs if f.get('geometric_m')), None)
    xlo = max(min(xs + ([geo] if geo else [])) / 1.3, 0.5)
    xhi = max(xs + ([geo] if geo else [])) * 1.3
    X, Y = _axes(img, 56, 40, W - 200, H - 46, xlo, xhi, pal, 'range from the camera (m, log)', 'P(detect)')
    for k, f in enumerate(fs):
        col = SERIES[k % len(SERIES)]
        rr = np.geomspace(xlo, xhi, 120)
        p = curve(f['range'], rr)
        if np.isfinite(p).any():
            pts = np.array([[X(r), Y(q)] for r, q in zip(rr, p) if np.isfinite(q)], np.int32)
            cv2.polylines(img, [pts], False, col, 2, cv2.LINE_AA)
        for b in f['range']['bins']:
            cx, cy = X(b['x']), Y(b['p'])
            if b['lo'] is not None:
                cv2.line(img, (cx, Y(b['lo'])), (cx, Y(b['hi'])), col, 1, cv2.LINE_AA)
            cv2.circle(img, (cx, cy), 3 + min(int(b['n'] ** 0.5), 5), col, -1, cv2.LINE_AA)
        r50 = f['range']['x'][0]
        if r50 and xlo <= r50 <= xhi:
            cv2.drawMarker(img, (X(r50), Y(0.5)), col, cv2.MARKER_DIAMOND, 12, 2)
        put_text(img, f'{FM.model_label(f["model"])}', (W - 190, 50 + 34 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1,
                 cv2.LINE_AA)
        put_text(img, f'R50 {range_text(f, 0).split("  ")[0]}', (W - 190, 66 + 34 * k), cv2.FONT_HERSHEY_SIMPLEX,
                 0.36, pal['ink'], 1, cv2.LINE_AA)
    if geo and xlo <= geo <= xhi:
        gx = X(geo)
        for yy in range(40, H - 46, 10):
            cv2.line(img, (gx, yy), (gx, yy + 5), pal['ink'], 1)
        put_text(img, f'geometric {geo:.0f} m', (gx + 4, H - 52), cv2.FONT_HERSHEY_SIMPLEX, 0.38, pal['ink'], 1,
                 cv2.LINE_AA)
    return bytes(cv2.imencode('.png', img)[1])


def depth_png(res: FoundationResult, dark=False, size=(760, 340)) -> bytes:
    """AbsRel against ground-truth depth per camera, the threshold dashed."""
    import cv2
    from .pngtext import put_text
    pal = _DARK if dark else _LIGHT
    W, H = size
    img = np.full((H, W, 3), pal['bg'], np.uint8)
    d = res.depth or {}
    cams = d.get('cameras') or {}
    put_text(img, f'Depth error · {d.get("name") or "no depth model"} ({d.get("scaled", "")})', (12, 20),
             cv2.FONT_HERSHEY_SIMPLEX, 0.5, pal['ink'], 1, cv2.LINE_AA)
    if not cams:
        return bytes(cv2.imencode('.png', img)[1])
    edges = [0.0] + list(d['edges'])
    centres = [(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)] + [edges[-1] * 1.25]
    top = max([b['abs_rel'] for c in cams.values() for b in c['bins'] if b['abs_rel'] is not None] + [0.2])
    top = min(max(top * 1.1, d['abs_rel_max'] * 2), 2.0)
    x0, y0, x1, y1 = 56, 40, W - 150, H - 46

    def X(v):
        return int(round(x0 + v / (centres[-1] * 1.05) * (x1 - x0)))

    def Y(v):
        return int(round(y1 - min(v, top) / top * (y1 - y0)))
    cv2.rectangle(img, (x0, y0), (x1, y1), pal['muted'], 1)
    for v in np.linspace(0, top, 5):
        cv2.line(img, (x0, Y(v)), (x1, Y(v)), pal['hair'], 1)
        put_text(img, f'{v * 100:.0f}%', (x0 - 44, Y(v) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, pal['muted'], 1,
                 cv2.LINE_AA)
    for e in edges[1:]:
        put_text(img, f'{e:g}', (X(e) - 8, y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, pal['muted'], 1, cv2.LINE_AA)
    yt = Y(d['abs_rel_max'])
    for xx in range(x0, x1, 10):
        cv2.line(img, (xx, yt), (xx + 5, yt), pal['ink'], 1)
    put_text(img, 'ground-truth depth (m)', ((x0 + x1) // 2 - 60, y1 + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
             pal['ink'], 1, cv2.LINE_AA)
    for k, (tag, c) in enumerate(sorted(cams.items())):
        col = SERIES[k % len(SERIES)]
        pts = [(X(cx), Y(b['abs_rel'])) for cx, b in zip(centres, c['bins']) if b['abs_rel'] is not None and
               b['n_px'] >= res.settings.depth_min_px]
        if len(pts) > 1:
            cv2.polylines(img, [np.array(pts, np.int32)], False, col, 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(img, p, 4, col, -1, cv2.LINE_AA)
        u = c['usable_m']
        put_text(img, f'{tag}: usable {"> " + str(int(edges[-1])) if u == math.inf else f"{u:g}"} m',
                 (W - 140, 50 + 20 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
    return bytes(cv2.imencode('.png', img)[1])


def files(res: FoundationResult, project=None) -> list:
    out = [('foundation_range.json', json.dumps(to_doc(res, project), indent=1, default=str).encode()),
           ('foundation_range.csv', to_csv(res).encode()), ('foundation_trials.csv', trials_csv(res).encode())]
    if res.depth:
        out.append(('foundation_depth.csv', depth_csv(res).encode()))
        out.append(('foundation_depth.png', depth_png(res)))
    for cam in res.cameras():
        for cls in res.classes():
            for mode in res.settings.modes():
                if any(f['camera'] == cam and f['class'] == cls and f['mode'] == mode for f in res.fits):
                    out.append((f'foundation_pdetect_{_slug(cam)}_{_slug(cls)}_{mode}.png',
                                pdetect_png(res, cam, cls, mode)))
    for i, e in enumerate(res.examples):
        out.append((f'examples/{i:02d}_{_slug(e["model"])}_{e["mode"]}_{_slug(e["camera"])}.jpg', e['jpg']))
    for r in res.records:
        out.append((f'records/perception_record-{r["id"][:16]}.json', json.dumps(r, indent=1).encode()))
    return out


def export(res: FoundationResult, project, out_dir) -> list[str]:
    paths = []
    for name, data in files(res, project):
        p = os.path.join(out_dir, name)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, 'wb') as fh:
            fh.write(data)
        paths.append(p)
    return paths
