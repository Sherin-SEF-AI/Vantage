"""The learned rig-performance model: from a rig's description alone (its cameras' poses and lenses, its lidars and
radars, the vehicle's body), predict how well a detector finds each class in each region around the car (core.regions:
sector × range), with an uncertainty that is checked, not assumed; and choose which candidate rigs are worth a real
evaluation (active learning).

The training data
-----------------
One perception record (schema vantage.perception_record/1, the round's shared format) is one rig evaluated on one scene
set by one detector: per class and region the number of ground-truth objects (n_gt), true positives, false positives,
misses, recall, precision, AP and a BEV centre error. Records come from core.rig_perception (perception in the loop),
core.redrive and core.robust_perception, through add_record(); they are kept in a JSON-lines store in the user's data
folder (detection.data_dir()/perf_model/records.jsonl, or $VANTAGE_PERF_DIR), or, when a project opts in
(analysis.perf_model.store_in_project), next to the project file. export_records(anonymise=True) drops the vehicle's
name, the project's and the variant's names and every file path, so teams can pool records; the provenance's hashes
stay.

Until core.rig_perception exists, region_eval() is this module's own INTERIM evaluator (marked as such in every record
it writes: detector.backend 'lifted', detector.evaluator 'perf_model.region_eval (interim)'): the 2D detector of
core.detection_eval on a CARLA recording with ground truth, matched per camera image as detection_eval does (COCO's
rules, its ignore regions), then counted per object for a rig = any subset of the recording's cameras, as
detection_eval's rig-level figure counts it (an object at one time slot is detected when any camera of the rig finds
it at the operating point). An object's region is where its 3D box centre is (the ground truth's vehicle frame). The
objects counted are those some camera of the whole recording (the camera library) sees well enough to count (not an
ignore region): a rig without a camera towards a region misses them. A false detection is placed by its box centre's
ray and the range image inside its box. AP is per region over the rig's objects (each at its best score over the rig's
cameras) and its false detections; the BEV centre error is not measured (a 2D detector gives no 3D centre): null.

Features (describe_rig)
-----------------------
Physics only, from the geometry (no image): for every class (its typical height and width) and region, at 12 points
of the region (4 bearings × 3 ranges, at the object's mid-height), each camera's view through its real lens model
(core.lens; a record's unknown module from its stated field and projection): seen (in the image, inside the image
circle, not hidden by the body: the engine's occlusion cube maps), pixels per metre there (the lens's density at that
direction and range), so pixels on the object (height × px/m), θ / θ_max (how near the image edge), depression of the
view, mounting height; then per region: log2 pixels on the object for the best camera (median and 25th percentile over
the points) and the second best, the share of points seen, cameras seeing them, the body's share of the views, the
stereo baseline where two cameras see a point, lidar points on the object per frame (core.sensor_coverage's count),
the share in a radar's field, and log2 range.

The model
---------
Per class and metric, a Gaussian process on the empirical logit of recall, y = log((tp + ½) / (fn + ½)) (Haldane;
Anscombe 1956), whose sampling variance 1/(tp + ½) + 1/(fn + ½) is the heteroscedastic noise of each row (a region
with few objects counts less), times a dispersion φ ≥ 1 that the data choose (frames of one drive are not independent
objects: the same pedestrian is counted at every time slot) plus a homoscedastic σn². The mean is a monotone prior:
a logistic in log2 pixels on the object (slope ≥ 0: recall rises with pixels) fitted first by binomial maximum
likelihood; the GP models the residual with an ARD Matérn-5/2 kernel on standardised features (Rasmussen and Williams
2006, eq. 4.17 and ch. 5). Hyperparameters (length scales, σf, σn, φ) by the log marginal likelihood with its analytic
gradient (R&W eq. 5.9), L-BFGS-B within bounds, from several starts. Above SPARSE_N rows the hyperparameters are fitted
on a random subset and the posterior is FITC on k-means++ inducing points (Snelson and Ghahramani 2006). A prediction is
the latent logit's mean and variance per class × region, turned into recall by Gauss–Hermite quadrature; an interval
is the logit's interval through the logistic (exact for quantiles).

Calibration is reported, not assumed: leave-one-scene-out (scene sets: same town, scenario, seed, weather, duration
and traffic) and leave-one-rig-out cross-validation, refitting everything per fold; on the held-out rows the coverage
of the 68 % and 95 % predictive intervals of the observed empirical logit (latent variance + the row's noise), the
RMSE of recall and the mean negative log predictive density, against two baselines with the same noise model: the
logistic in pixels alone and a constant. Below MIN_RECORDS records (or MIN_ROWS rows for a class) a model is "not
trained (N records, need M)" and predicts nothing.

Active learning
---------------
propose(candidates, k): the objective is a weighted mean of predicted recall over the classes and regions that matter;
its mean and variance per candidate from the joint posterior of that candidate's regions (delta method through the
logistic); candidates are ranked by expected improvement over the best evaluated rig (Jones et al. 1998), or by the
objective's predicted std ('std': where the model knows least), or by mean + κ·std ('ucb'); a batch of k by the
kriging believer (Ginsbourger et al. 2010): each pick's predicted mean joins the training data as an observation before
the next pick. campaign() evaluates the picks for real (core.rig_perception when present, else the interim CARLA
evaluator), adds their records, refits and repeats, recording the learning curve: the model's prospective error and
interval coverage on each round's new rigs before they were added.
"""
from __future__ import annotations

import copy
import datetime as _dt
import hashlib
import json
import math
import os
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from . import regions as RG

SCHEMA = 'vantage.perf_model/1'
RECORD_SCHEMA = 'vantage.perception_record/1'
FEATURES_VERSION = 1
FRAME_V = 'vehicle (ISO 8855, rear axle)'
CLASSES = ('pedestrian', 'cyclist', 'motorcyclist', 'car', 'truck', 'bus', 'vehicle')
SOURCES = ('carla', 'redrive', 'synthetic-test')
METRICS = ('recall', 'precision', 'ap')
# typical object sizes (height, width, m) and radar cross-sections (m²); the perception range's and the sensor
# coverage's typical classes where they have the class (core.perception / core.sensor_coverage DEFAULT_CLASSES)
CLASS_SIZE = {'pedestrian': (1.7, 0.5), 'cyclist': (1.7, 0.6), 'motorcyclist': (1.5, 0.8), 'car': (1.5, 1.8),
              'truck': (3.0, 2.5), 'bus': (3.2, 2.6), 'vehicle': (1.5, 1.8)}
CLASS_RCS = {'pedestrian': 0.5, 'cyclist': 1.0, 'motorcyclist': 3.0, 'car': 10.0, 'truck': 50.0, 'bus': 50.0,
             'vehicle': 10.0}
FEATURE_NAMES = ('log2_px_obj', 'log2_px_obj_p25', 'log2_px_obj_2nd', 'seen_share', 'n_cams', 'body_share',
                 'edge', 'depression_deg', 'mount_h_m', 'baseline_m', 'log2_lidar_pts', 'radar_share', 'log2_range_m')
FEATURE_TEXT = {
    'log2_px_obj': 'pixels on the object (best camera)', 'log2_px_obj_p25': 'pixels on the object across the region',
    'log2_px_obj_2nd': 'pixels on the object (second camera)', 'seen_share': 'share of the region seen',
    'n_cams': 'cameras seeing it', 'body_share': 'views blocked by the body', 'edge': 'nearness to the image edge',
    'depression_deg': 'view depression', 'mount_h_m': 'mounting height', 'baseline_m': 'stereo baseline',
    'log2_lidar_pts': 'lidar points on the object', 'radar_share': 'radar coverage', 'log2_range_m': 'range'}
PX_FLOOR = 0.5                         # log2 of pixels on an unseen object: log2(0.5) = −1
MIN_RECORDS = 8
MIN_ROWS = 25
SPARSE_N = 3000
SUBSET_N = 600
CV_THREADS = 4                         # cross-validation folds fitted at once (numpy releases the GIL)
INDUCING_M = 400
MAX_STRING = 2000                      # a record's strings are names and hashes, never image data
INTERIM = 'perf_model.region_eval (interim)'
ASSUMPTIONS = (
    'Records are what they say: a detector on a scene set (CARLA renders, re-shot drives). The model learns how '
    'detection varies with the rig\'s geometry on the scene sets it was trained on; a new town, weather, detector or '
    'vehicle is outside what it has seen and its intervals then widen only as far as the features move. Features '
    'are computed from the rig\'s geometry with typical object sizes per class, not from images. Recall is modelled '
    'on the empirical logit with binomial noise times a dispersion the data choose (frames of one drive are not '
    'independent objects). Intervals are the Gaussian process\'s; their coverage is measured by cross-validation and '
    'shown, never assumed.')
LIMITS_TEXT = (
    'The interim evaluator (until perception in the loop records are available) counts a 2D detector per camera '
    'image (YOLOX on CARLA renders) and rigs as subsets of one drive\'s cameras: rigs from one drive share its '
    'objects, so leave-one-rig-out is optimistic; leave-one-scene-out is the honest check. Regions with no objects '
    'teach nothing. AP is per region over the rig\'s objects and false detections; no BEV centre error is measured '
    'by a 2D detector. Records from different vehicles are featurised on their own vehicle\'s body only when the '
    'project\'s vehicle matches the record\'s size; otherwise the body is not an occluder for them.')
LIMITS = {'restarts': (1, 10), 'max_iter': (10, 1000), 'min_records': (2, 10000), 'min_rows': (5, 100000),
          'sparse_n': (200, 50000), 'inducing': (20, 2000), 'kappa': (0.0, 10.0), 'max_range_m': (0.0, 1000.0),
          'seed': (0, 2 ** 31 - 1)}


def _utc():
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def _canon_bytes(o) -> bytes:
    from . import provenance as PV
    return PV.canonical(o)


def _sha(o) -> str:
    return hashlib.sha256(_canon_bytes(o)).hexdigest()


def _f(v, default=None):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    return x if math.isfinite(x) else default


# =========================================================================== records
ID_EXCLUDE = ('id', 'created', 'provenance')


def record_id(doc: dict) -> str:
    """sha256 of the record's canonical JSON (core.provenance.canonical) without its id, its created stamp and its
    provenance block: the same rig, scene, detector and metrics give the same id whenever and wherever they were
    measured, so a re-run or a re-import is recognised as a duplicate (the provenance still travels with the record)."""
    return _sha({k: v for k, v in doc.items() if k not in ID_EXCLUDE})


def rig_hash(rig: dict) -> str:
    return _sha(rig)


def _hfov(intr) -> tuple:
    """(horizontal, vertical) field across the image's middle row and column within the image circle, degrees."""
    u = np.array([0.0, float(intr.width)])
    _, th = intr.unproject(u, np.full(2, intr.cy))
    hf = float(np.degrees(np.minimum(th, intr.max_theta).sum()))
    v = np.array([0.0, float(intr.height)])
    _, tv = intr.unproject(np.full(2, intr.cx), v)
    vf = float(np.degrees(np.minimum(tv, intr.max_theta).sum()))
    return hf, vf


def rig_block(project, variant=None, image_size=None) -> dict:
    """A variant as a record's rig block: the visible cameras (module, lens, the image size recorded: image_size =
    {tag: (w, h)} when the recording's images are not the module's full resolution), the visible lidars and radars."""
    from .lens import resolve
    v = variant if variant is not None else project.variant
    cams = []
    for c in v.cameras:
        if not c.visible:
            continue
        intr = resolve(project.module_for(c), project.lens_for(c))
        w, h = (image_size or {}).get(c.tag, (intr.width, intr.height))
        hf, vf = _hfov(intr)
        cams.append({'tag': c.tag, 'module': c.module_id, 'lens': c.lens_id, 'width': int(w), 'height': int(h),
                     'x': round(float(c.x), 4), 'y': round(float(c.y), 4), 'z': round(float(c.z), 4),
                     'yaw': round(float(c.yaw), 3), 'pitch': round(float(c.pitch), 3), 'roll': round(float(c.roll), 3),
                     'hfov_deg': round(hf, 2), 'vfov_deg': round(vf, 2),
                     'max_theta_deg': round(math.degrees(intr.max_theta), 2), 'projection': intr.projection})
    sens = [{'tag': s.tag, 'kind': s.kind, 'model': s.model_id, 'x': round(float(s.x), 4), 'y': round(float(s.y), 4),
             'z': round(float(s.z), 4), 'yaw': round(float(s.yaw), 3), 'pitch': round(float(s.pitch), 3),
             'roll': round(float(s.roll), 3)}
            for s in v.sensors if s.visible and s.kind in ('lidar', 'radar')]
    return {'vehicle': vehicle_block(project), 'cameras': cams, 'sensors': sens}


def vehicle_block(project) -> dict:
    name, L, W, H = '', None, None, None
    try:
        name = str(project.vehicle.name or project.name or '')
        V = project.vehicle.mesh.vertices
        ext = V.max(0) - V.min(0)
        L, W, H = (round(float(x), 3) for x in ext)
    except Exception:                                   # noqa: BLE001 — no body: sizes unknown
        pass
    return {'name': name, 'length_m': L, 'width_m': W, 'height_m': H}


def make_record(*, source, scene, detector, rig, grid, metrics, provenance=None, created=None, extra=None) -> dict:
    """A complete record (id, created, rig_hash filled in). grid: a RegionGrid or its dict."""
    from .. import __version__
    g = grid.to_dict() if isinstance(grid, RG.RegionGrid) else dict(grid)
    sc = dict(scene)
    if not sc.get('scene_hash'):
        sc['scene_hash'] = scene_hash(sc)
    doc = {'schema': RECORD_SCHEMA, 'id': '', 'created': created or _utc(), 'vantage': __version__,
           'source': source, 'scene': sc, 'detector': dict(detector), 'rig': rig, 'rig_hash': rig_hash(rig),
           'grid': g, 'metrics': metrics, 'provenance': provenance}
    if extra:
        doc.update(extra)
    doc['id'] = record_id(doc)
    return doc


def scene_hash(scene: dict) -> str:
    """The scene set's identity: what defines it (town, scenario, seed, weather, duration, traffic), not how it ran."""
    keep = {k: scene.get(k) for k in ('kind', 'town', 'scenario', 'seed', 'weather', 'duration_s', 'traffic',
                                      'walkers', 'source_id')}
    return _sha(keep)


def _home_strings():
    out = {os.path.expanduser('~')}
    return [h for h in out if h and h not in ('/', '~')]


def _bad_strings(o, path='', out=None, homes=None):
    out = [] if out is None else out
    homes = _home_strings() if homes is None else homes
    if isinstance(o, dict):
        for k, v in o.items():
            _bad_strings(v, f'{path}.{k}' if path else str(k), out, homes)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            _bad_strings(v, f'{path}[{i}]', out, homes)
    elif isinstance(o, str):
        if len(o) > MAX_STRING:
            out.append(f'{path}: a string of {len(o)} characters (records hold no image data)')
        elif o.startswith(('/home/', '/Users/', '/root/')) or (len(o) > 2 and o[1:3] in (':\\', ':/')) or \
                any(h in o for h in homes):
            out.append(f'{path}: a user path ({o[:40]}…): records hold no paths')
        elif o.startswith(('data:image', 'iVBOR', '/9j/')):
            out.append(f'{path}: image data')
    return out


def validate_record(doc) -> list:
    """The reasons doc is not a valid perception record ([] when it is)."""
    e = []
    if not isinstance(doc, dict):
        return ['not a JSON object']
    if doc.get('schema') != RECORD_SCHEMA:
        return [f'schema is not {RECORD_SCHEMA}']
    for k in ('id', 'created', 'source', 'scene', 'detector', 'rig', 'rig_hash', 'grid', 'metrics'):
        if k not in doc:
            e.append(f'{k} is missing')
    if e:
        return e
    if not (isinstance(doc['id'], str) and len(doc['id']) == 64 and all(c in '0123456789abcdef' for c in doc['id'])):
        e.append('id is not a sha256')
    elif doc['id'] != record_id(doc):
        e.append('id does not match the record\'s content')
    try:
        _dt.datetime.fromisoformat(str(doc['created']).replace('Z', '+00:00'))
    except ValueError:
        e.append('created is not an ISO-8601 time')
    if doc['source'] not in SOURCES:
        e.append(f'source {doc["source"]!r} is not one of {", ".join(SOURCES)}')
    sc = doc['scene']
    if not isinstance(sc, dict) or not sc.get('kind'):
        e.append('scene has no kind')
    det = doc['detector']
    if not isinstance(det, dict) or not det.get('id') or not det.get('backend'):
        e.append('detector needs id and backend')
    rig = doc['rig']
    if not isinstance(rig, dict) or not isinstance(rig.get('cameras'), list) or not isinstance(rig.get('sensors', []),
                                                                                              list):
        e.append('rig needs cameras and sensors lists')
    else:
        if not rig['cameras'] and not rig.get('sensors'):
            e.append('rig has no camera or sensor')
        for i, c in enumerate(rig['cameras']):
            if not isinstance(c, dict):
                e.append(f'rig.cameras[{i}] is not an object')
                continue
            for k in ('x', 'y', 'z', 'yaw', 'pitch', 'roll'):
                if _f(c.get(k)) is None:
                    e.append(f'rig.cameras[{i}].{k} is not a number')
            if not (isinstance(c.get('width'), int) and isinstance(c.get('height'), int) and c['width'] > 0
                    and c['height'] > 0):
                e.append(f'rig.cameras[{i}] needs a width and height in pixels')
            if not c.get('module') and not (_f(c.get('hfov_deg')) and c.get('projection')):
                e.append(f'rig.cameras[{i}] needs a module and lens, or its hfov_deg and projection')
        for i, s in enumerate(rig.get('sensors', [])):
            if not isinstance(s, dict) or s.get('kind') not in ('lidar', 'radar', 'gnss', 'imu'):
                e.append(f'rig.sensors[{i}] needs a kind')
                continue
            for k in ('x', 'y', 'z'):
                if _f(s.get(k)) is None:
                    e.append(f'rig.sensors[{i}].{k} is not a number')
        if doc['rig_hash'] != rig_hash(rig):
            e.append('rig_hash does not match the rig block')
    if not isinstance(doc['grid'], dict) or 'sectors' not in doc['grid']:
        e.append('grid is not a RegionGrid')
        return e
    grid = RG.RegionGrid.from_dict(doc['grid'])
    m = doc['metrics']
    if not isinstance(m, dict) or not m:
        e.append('metrics is empty')
        return e
    for cls, regs in m.items():
        if cls not in CLASSES:
            e.append(f'metrics: unknown class {cls!r}')
            continue
        if not isinstance(regs, dict):
            e.append(f'metrics.{cls} is not an object')
            continue
        for rid, r in regs.items():
            where = f'metrics.{cls}.{rid}'
            if grid.parse_id(rid) is None:
                e.append(f'{where}: not a region of the grid')
                continue
            if not isinstance(r, dict):
                e.append(f'{where} is not an object')
                continue
            n = {k: r.get(k) for k in ('n_gt', 'tp', 'fp', 'fn')}
            if not all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in n.values()):
                e.append(f'{where}: n_gt, tp, fp, fn must be counts')
                continue
            if n['tp'] + n['fn'] != n['n_gt']:
                e.append(f'{where}: tp + fn ≠ n_gt')
            rec = r.get('recall')
            if n['n_gt'] > 0:
                if _f(rec) is None or abs(float(rec) - n['tp'] / n['n_gt']) > 1e-3:
                    e.append(f'{where}: recall ≠ tp / n_gt')
            elif rec is not None:
                e.append(f'{where}: recall without objects')
            for k in ('precision', 'ap'):
                v = r.get(k)
                if v is not None and (_f(v) is None or not 0.0 <= float(v) <= 1.0):
                    e.append(f'{where}: {k} is not in 0..1 or null')
            a = r.get('ate_m')
            if a is not None and (_f(a) is None or float(a) < 0):
                e.append(f'{where}: ate_m is not a distance or null')
    e += _bad_strings({k: v for k, v in doc.items() if k != 'provenance'})
    return e


def metrics_cell(n_gt, tp, fp, ap=None, ate_m=None, tp_det=None, extra=None) -> dict:
    """One class × region entry. tp_det: true detections counted for precision (default tp)."""
    n_gt, tp, fp = int(n_gt), int(tp), int(fp)
    td = tp if tp_det is None else int(tp_det)
    out = {'n_gt': n_gt, 'tp': tp, 'fp': fp, 'fn': n_gt - tp, 'recall': (tp / n_gt) if n_gt else None,
           'precision': (td / (td + fp)) if (td + fp) else None,
           'ap': None if ap is None else float(ap), 'ate_m': None if ate_m is None else float(ate_m)}
    if extra:
        out.update(extra)
    return out


# =========================================================================== the store
def data_dir() -> str:
    """$VANTAGE_PERF_DIR, else the user's data folder's perf_model/ (beside detection's models/)."""
    env = os.environ.get('VANTAGE_PERF_DIR')
    if env:
        return os.path.abspath(os.path.expanduser(env))
    from . import detection as DT
    return os.path.join(DT.data_dir(), 'perf_model')


def default_store_path() -> str:
    return os.path.join(data_dir(), 'records.jsonl')


def settings_dict(project) -> dict:
    d = getattr(getattr(project, 'analysis', None), 'perf_model', None)
    return d if isinstance(d, dict) else {}


def project_store_path(project) -> str | None:
    """The project's own store when it opts in (analysis.perf_model.store_in_project): <project file stem>.records
    .jsonl beside the project file, or the project's working folder when it has no file yet."""
    if not settings_dict(project).get('store_in_project'):
        return None
    path = getattr(project, 'path', '') or ''
    if path:
        stem = os.path.splitext(os.path.basename(path))[0]
        return os.path.join(os.path.dirname(os.path.abspath(path)), f'{stem}.perception_records.jsonl')
    from .. import workdir as WD
    return WD.project_dir(project, None, 'perf_model', 'records.jsonl')


class Store:
    """Perception records as JSON lines, one record per line, unique by id. Reads are cached on the file's size and
    modification time; writes append (or rewrite through a temporary file on removal)."""

    _lock = threading.RLock()

    def __init__(self, path=None):
        self.path = os.path.abspath(os.path.expanduser(path or default_store_path()))
        self._stamp = None
        self._docs = []
        self.bad_lines = 0

    def _file_stamp(self):
        try:
            s = os.stat(self.path)
            return (s.st_size, s.st_mtime_ns)
        except OSError:
            return None

    def records(self) -> list:
        with self._lock:
            st = self._file_stamp()
            if st is None:
                self._docs, self._stamp, self.bad_lines = [], None, 0
                return []
            if st == self._stamp:
                return list(self._docs)
            docs, seen, bad = [], set(), 0
            with open(self.path, encoding='utf-8') as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                    except ValueError:
                        bad += 1
                        continue
                    if not isinstance(d, dict) or d.get('id') in seen:
                        bad += not isinstance(d, dict)
                        continue
                    seen.add(d.get('id'))
                    docs.append(d)
            self._docs, self._stamp, self.bad_lines = docs, st, bad
            return list(docs)

    def ids(self) -> set:
        out = set()
        for d in self.records():
            out.add(d.get('id'))
            if d.get('origin_id'):
                out.add(d['origin_id'])
        return out

    def add(self, doc) -> dict:
        """{'added': bool, 'id', 'errors': [...]}: a valid record not already in the store is appended."""
        errs = validate_record(doc)
        rid = doc.get('id') if isinstance(doc, dict) else None
        if errs:
            return {'added': False, 'id': rid, 'errors': errs}
        with self._lock:
            ids = self.ids()
            if rid in ids or (doc.get('origin_id') and doc['origin_id'] in ids):
                return {'added': False, 'id': rid, 'errors': [], 'duplicate': True}
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, 'a', encoding='utf-8') as fh:
                fh.write(json.dumps(doc, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n')
            self._stamp = None
        return {'added': True, 'id': rid, 'errors': []}

    def add_many(self, docs) -> dict:
        n_add = n_dup = 0
        errors = []
        for d in docs:
            r = self.add(d)
            n_add += r['added']
            n_dup += bool(r.get('duplicate'))
            if r['errors']:
                errors.append({'id': r['id'], 'errors': r['errors'][:5]})
        return {'added': n_add, 'duplicates': n_dup, 'rejected': len(errors), 'errors': errors[:20]}

    def remove(self, ids) -> int:
        ids = set(ids)
        with self._lock:
            docs = self.records()
            keep = [d for d in docs if d.get('id') not in ids]
            if len(keep) == len(docs):
                return 0
            tmp = self.path + '.part'
            with open(tmp, 'w', encoding='utf-8') as fh:
                for d in keep:
                    fh.write(json.dumps(d, ensure_ascii=False, separators=(',', ':')) + '\n')
            os.replace(tmp, self.path)
            self._stamp = None
            return len(docs) - len(keep)


_STORES = {}


def store_for(project=None, path=None) -> Store:
    """The store to use: path, else the project's own (when it opts in), else the user's."""
    p = path or (project_store_path(project) if project is not None else None) or default_store_path()
    p = os.path.abspath(os.path.expanduser(p))
    s = _STORES.get(p)
    if s is None:
        s = _STORES[p] = Store(p)
    return s


def _as_store(store) -> Store:
    if isinstance(store, Store):
        return store
    return store_for(path=store) if store else store_for()


def add_record(doc, store=None) -> dict:
    """Validate doc (a perception record) and add it to the store (default: the user's) unless its id is there:
    {'added', 'id', 'errors', 'duplicate'?}. A record without an id or created stamp gets them."""
    if isinstance(doc, dict) and doc.get('schema') == RECORD_SCHEMA:
        doc = dict(doc)
        if not doc.get('created'):
            doc['created'] = _utc()
        if isinstance(doc.get('rig'), dict) and not doc.get('rig_hash'):
            doc['rig_hash'] = rig_hash(doc['rig'])
        if not doc.get('id'):
            doc['id'] = ''
            doc['id'] = record_id(doc)
    return _as_store(store).add(doc)


def _match(doc, flt) -> bool:
    if flt is None:
        return True
    if callable(flt):
        return bool(flt(doc))
    for k, v in flt.items():
        vals = v if isinstance(v, (list, tuple, set)) else (v,)
        if k == 'source':
            ok = doc.get('source') in vals
        elif k == 'rig_hash':
            ok = doc.get('rig_hash') in vals
        elif k == 'scene_hash':
            ok = (doc.get('scene') or {}).get('scene_hash') in vals
        elif k in ('town', 'scenario'):
            ok = (doc.get('scene') or {}).get(k) in vals
        elif k == 'detector':
            ok = (doc.get('detector') or {}).get('id') in vals
        elif k == 'class':
            ok = any(c in (doc.get('metrics') or {}) for c in vals)
        elif k == 'id':
            ok = doc.get('id') in vals
        else:
            ok = doc.get(k) in vals
        if not ok:
            return False
    return True


def records(filter=None, store=None) -> list:   # noqa: A002 — the brief's name
    """The store's records matching filter: a dict (source, rig_hash, scene_hash, town, scenario, detector, class, id:
    a value or a list of values) or a callable(doc) -> bool."""
    return [d for d in _as_store(store).records() if _match(d, filter)]


def anonymise(doc: dict) -> dict:
    """The record without names or paths: the vehicle's name, the project's and the variant's names, the scene's
    free-text names and every file path go; geometry, metrics and the provenance's hashes stay. A new id; the
    original id is kept as origin_id so a re-import is still recognised."""
    d = copy.deepcopy(doc)
    rig = d.get('rig') or {}
    if isinstance(rig.get('vehicle'), dict):
        rig['vehicle'].pop('name', None)
    for c in rig.get('cameras') or []:
        if isinstance(c, dict):
            c.pop('name', None)
    pv = d.get('provenance')
    if isinstance(pv, dict):
        keep = {'schema': pv.get('schema'), 'vantage': pv.get('vantage'), 'study': pv.get('study'),
                'created_utc': pv.get('created_utc'), 'result_sha256': pv.get('result_sha256')}
        pj = pv.get('project')
        if isinstance(pj, dict):
            keep['project'] = {'sha256': pj.get('sha256')}
        vr = pv.get('variant')
        if isinstance(vr, dict):
            keep['variant'] = {'sha256': vr.get('sha256')}
        d['provenance'] = keep
    for k in ('project', 'variant', 'label', 'path', 'name'):
        d.pop(k, None)
        (d.get('scene') or {}).pop(k, None)
    d['rig'] = rig
    d['rig_hash'] = rig_hash(rig)
    d['origin_id'] = doc.get('origin_id') or doc.get('id')
    d['anonymised'] = True
    d['id'] = record_id(d)
    return d


def export_records(path, anonymise=True, store=None, docs=None, filter=None) -> int:   # noqa: A002
    """Write records (default: the store's, filtered) as JSON lines; anonymised by default. Returns how many."""
    docs = records(filter, store) if docs is None else list(docs)
    tmp = str(path) + '.part'
    with open(tmp, 'w', encoding='utf-8') as fh:
        for d in docs:
            out = globals()['anonymise'](d) if anonymise else d
            fh.write(json.dumps(out, ensure_ascii=False, separators=(',', ':')) + '\n')
    os.replace(tmp, path)
    return len(docs)


def import_records(path, store=None) -> dict:
    """Add the records of a JSON-lines (or JSON list) file: {'added', 'duplicates', 'rejected', 'errors'}."""
    docs = []
    with open(path, encoding='utf-8') as fh:
        text = fh.read()
    s = text.lstrip()
    if s.startswith('['):
        try:
            v = json.loads(s)
            docs = [x for x in v if isinstance(x, dict)]
        except ValueError:
            docs = []
    else:
        for line in text.splitlines():
            line = line.strip()
            if line:
                try:
                    docs.append(json.loads(line))
                except ValueError:
                    docs.append({'_bad': line[:40]})
    return _as_store(store).add_many(docs)


def store_summary(docs) -> dict:
    """Counts by source, scene set, rig and detector, for the page and the assistant."""
    by = {'source': {}, 'scene': {}, 'rig': {}, 'detector': {}}
    for d in docs:
        sc = d.get('scene') or {}
        by['source'][d.get('source')] = by['source'].get(d.get('source'), 0) + 1
        sk = ' · '.join(str(x) for x in (sc.get('town'), sc.get('scenario'), f'seed {sc.get("seed")}') if x)
        by['scene'][sk] = by['scene'].get(sk, 0) + 1
        by['rig'][d.get('rig_hash', '')[:12]] = by['rig'].get(d.get('rig_hash', '')[:12], 0) + 1
        dk = (d.get('detector') or {}).get('id', '?') + ' · ' + str((d.get('detector') or {}).get('backend', ''))
        by['detector'][dk] = by['detector'].get(dk, 0) + 1
    return {'records': len(docs), 'scenes': len(by['scene']), 'rigs': len(by['rig']), 'by': by}


# =========================================================================== features
@dataclass
class _Cam:
    tag: str
    C: np.ndarray                       # optical centre, vehicle frame (m)
    R: np.ndarray                       # optical -> vehicle rotation
    intr: object
    cube: object = None


def _scaled(intr, w, h):
    """The intrinsics at another image size of the same sensor (a recording's images at a fraction of the module's)."""
    from .lens import Intrinsics
    w, h = int(w or intr.width), int(h or intr.height)
    if w == intr.width and h == intr.height:
        return intr
    s = intr.width / float(w)
    return Intrinsics(width=w, height=h, pixel_mm=intr.pixel_mm * s, f_mm=intr.f_mm, projection=intr.projection,
                      kb=intr.kb, cx=intr.cx / s, cy=intr.cy / (intr.height / float(h)), max_theta=intr.max_theta,
                      focus_near=intr.focus_near, focus_far=intr.focus_far, aspect=intr.aspect)


PROJ_ALIASES = {'pinhole': 'rectilinear', 'rectilinear': 'rectilinear', 'equidistant': 'equidistant',
                'fisheye': 'equidistant', 'kannala_brandt': 'equidistant', 'equisolid': 'equisolid',
                'stereographic': 'stereographic', 'orthographic': 'orthographic'}


def intrinsics_from_fov(width, height, hfov_deg, projection='rectilinear', max_theta_deg=None, pixel_mm=0.003):
    """A lens from its stated horizontal field and projection (a module Vantage does not know): the focal length that
    puts half the field at the image's side edge; the image circle at max_theta_deg (else the image corner's angle)."""
    from .lens import Intrinsics, r_of_theta, theta_of_r
    proj = PROJ_ALIASES.get(str(projection or '').lower(), 'rectilinear')
    w, h = int(width), int(height)
    half = math.radians(min(max(float(hfov_deg), 1.0), 359.0) / 2.0)
    if proj == 'rectilinear':
        half = min(half, math.radians(89.0))
    f_mm = (w / 2.0) * pixel_mm / float(r_of_theta(half, 1.0, proj))
    if max_theta_deg:
        mt = math.radians(float(max_theta_deg))
    else:
        mt = float(theta_of_r(math.hypot(w, h) / 2.0 * pixel_mm, f_mm, proj))
    mt = min(mt, math.radians(89.0 if proj == 'rectilinear' else 120.0))
    return Intrinsics(width=w, height=h, pixel_mm=pixel_mm, f_mm=f_mm, projection=proj, kb=(0.0, 0.0, 0.0, 0.0),
                      cx=w / 2.0, cy=h / 2.0, max_theta=mt, focus_near=0.0, focus_far=math.inf, aspect=1.0)


def _block_intr(c: dict, project=None):
    from .lens import builtin_lens, builtin_module, resolve
    mods = getattr(project, 'modules', {}) or {}
    lenses = getattr(project, 'lenses', {}) or {}
    m = mods.get(c.get('module')) or (builtin_module(c['module']) if c.get('module') else None)
    lens = lenses.get(c.get('lens')) or (builtin_lens(c['lens']) if c.get('lens') else None)
    if m is not None and lens is not None:
        return _scaled(resolve(m, lens), c.get('width'), c.get('height')), True
    return intrinsics_from_fov(c['width'], c['height'], c.get('hfov_deg') or 90.0, c.get('projection'),
                               c.get('max_theta_deg')), False


def _R_opt(yaw, pitch, roll):
    from .mathutil import BODY_TO_OPTICAL, rot_zyx
    R = rot_zyx(float(yaw), float(pitch), float(roll))
    return (BODY_TO_OPTICAL @ R.T).T


_ENGINE = None
_ENGINE_LOCK = threading.Lock()


def shared_engine():
    global _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is None:
            from .analysis import Engine
            _ENGINE = Engine()
        return _ENGINE


def _same_vehicle(a: dict, b: dict, tol=0.15) -> bool:
    try:
        return abs(float(a['length_m']) - float(b['length_m'])) < tol and \
            abs(float(a['width_m']) - float(b['width_m'])) < tol
    except (TypeError, ValueError, KeyError):
        return False


def _region_points(grid: RG.RegionGrid, n_b=4):
    """Sample points of every region (grid.regions() order): (R, P, 2) ground xy and each region's centre range."""
    w = 2 * math.pi / grid.sectors
    e = [0.0] + [float(x) for x in grid.range_edges]
    last = grid.max_range_m if grid.max_range_m > e[-1] else e[-1] + 20.0
    pts, rng = [], []
    for s, r, _, _ in grid.regions():
        b = s * w - w / 2 + w * (np.arange(n_b) + 0.5) / n_b
        r0, r1 = e[r], (e[r + 1] if r + 1 < len(e) else last)
        fr = np.array((0.35, 0.65, 0.9) if r == 0 else (0.2, 0.5, 0.8))
        rr = r0 + (r1 - r0) * fr
        B, RR = np.meshgrid(b, rr)
        pts.append(np.stack([grid.origin_xy[0] + RR.ravel() * np.cos(B.ravel()),
                             grid.origin_xy[1] + RR.ravel() * np.sin(B.ravel())], 1))
        rng.append(0.5 * (r0 + r1))
    return np.asarray(pts), np.asarray(rng)


def _rig_cams_from_block(rig: dict, project=None, engine=None, occlusion=True):
    cams, known = [], 0
    use_body = False
    vg = None
    if occlusion and project is not None and engine is not None:
        use_body = _same_vehicle(rig.get('vehicle') or {}, vehicle_block(project))
        if use_body:
            from .project import Variant
            vg = engine.vehicle_geom(project, Variant(name='record', cameras=[], sensors=[]))
            use_body = vg is not None
    for c in rig.get('cameras') or []:
        intr, ok = _block_intr(c, project)
        known += ok
        C = np.array([float(c['x']), float(c['y']), float(c['z'])])
        cube = engine.cube(C, vg) if use_body else None
        cams.append(_Cam(str(c.get('tag', '')), C, _R_opt(c['yaw'], c['pitch'], c.get('roll', 0.0)), intr, cube))
    return cams, use_body, known


def _sensor_objs(rig: dict):
    from .project import Sensor
    out = []
    for s in rig.get('sensors') or []:
        if s.get('kind') not in ('lidar', 'radar'):
            continue
        out.append(Sensor(tag=str(s.get('tag', '')), kind=s['kind'], model_id=str(s.get('model', '')),
                          x=float(s['x']), y=float(s['y']), z=float(s['z']), yaw=float(s.get('yaw', 0.0)),
                          pitch=float(s.get('pitch', 0.0)), roll=float(s.get('roll', 0.0))))
    return out


def _sensor_model(project, s):
    if project is not None:
        return project.sensor_model_for(s)
    from .sensors import builtin_sensor_models
    ms = builtin_sensor_models()
    return next((m for m in ms if m.id == s.model_id and m.kind == s.kind), None) or \
        next(m for m in ms if m.kind == s.kind)


_LIDAR_CACHE = {}


def describe_rig(project_or_rig, variant=None, grid=None, engine=None, classes=CLASSES, project=None,
                 occlusion=True) -> dict:
    """Physics-informed features per class × region of a rig, from its geometry alone (see the module doc).
    project_or_rig: a Project (its variant, default the active one) or a record's rig block (dict; project, when
    given, resolves its modules and lenses and, on the same vehicle, its body as an occluder). grid: a RegionGrid (or
    its dict); default the project's (regions.default_grid). Returns {'names', 'regions', 'classes', 'X': {class:
    (regions × features) array}, 'rig_hash', 'grid', 'occlusion', 'cameras', 'seconds', ...}."""
    t0 = time.perf_counter()
    if isinstance(project_or_rig, dict):
        rig = project_or_rig
    else:
        project = project_or_rig
        rig = rig_block(project, variant)
    if isinstance(grid, dict):
        grid = RG.RegionGrid.from_dict(grid)
    if grid is None:
        grid = RG.default_grid(project) if project is not None else RG.RegionGrid()
    eng = engine if engine is not None else (shared_engine() if (project is not None and occlusion) else None)
    cams, body, known = _rig_cams_from_block(rig, project, eng, occlusion)
    sens = _sensor_objs(rig)
    G, rcentre = _region_points(grid)                 # (R, P, 2)
    nR, nP = G.shape[:2]
    Gf = G.reshape(-1, 2)
    lidars, radars = [], []
    if sens:
        from . import sensor_coverage as SC
        vg = None
        if body and eng is not None:
            from .project import Variant
            vg = eng.vehicle_geom(project, Variant(name='record', cameras=[], sensors=[]))
        for s in sens:
            m = _sensor_model(project, s)
            cube = eng.cube(s.pos(), vg) if vg is not None else None
            if s.kind == 'lidar':
                key = (s.model_id, round(s.x, 4), round(s.y, 4), round(s.z, 4), round(s.yaw, 3), round(s.pitch, 3),
                       round(s.roll, 3), id(cube) if cube is not None else 0)
                L = _LIDAR_CACHE.get(key)
                if L is None:
                    L = SC._Lidar(s, m, cube)
                    if len(_LIDAR_CACHE) > 32:
                        _LIDAR_CACHE.clear()
                    _LIDAR_CACHE[key] = L
                lidars.append(L)
            else:
                radars.append(SC._Radar(s, m, cube, None, True))
    X = {}
    nC = len(cams)
    for cls in classes:
        h, wd = CLASS_SIZE.get(cls, (1.5, 1.8))
        P = np.c_[Gf, np.full(len(Gf), h / 2.0)]
        pxm = np.zeros((nC, len(P)))
        inview = np.zeros((nC, len(P)), bool)
        hidden = np.zeros((nC, len(P)), bool)
        edge = np.ones((nC, len(P)))
        dep = np.zeros((nC, len(P)))
        for i, c in enumerate(cams):
            d = P - c.C
            dist = np.linalg.norm(d, axis=1)
            u, v, th, ok = c.intr.project_dirs(d @ c.R)
            vis = ok.copy()
            if c.cube is not None and ok.any():
                depth = c.cube.lookup(d)[0]
                hid = ok & (depth <= dist - (0.02 + 0.01 * dist))
                vis &= ~hid
                hidden[i] = hid
            inview[i] = ok
            pxm[i] = np.where(vis, c.intr.density_px_per_cm(th, dist) * 100.0, 0.0)
            edge[i] = np.clip(th / max(c.intr.max_theta, 1e-6), 0.0, 2.0)
            dep[i] = np.degrees(np.arcsin(np.clip(-d[:, 2] / np.maximum(dist, 1e-6), -1, 1)))
        if nC:
            order = np.argsort(-pxm, axis=0)
            b1 = order[0]
            b2 = order[1] if nC > 1 else order[0]
            cols = np.arange(len(P))
            p1 = pxm[b1, cols]
            p2 = pxm[b2, cols] if nC > 1 else np.zeros(len(P))
            seen = pxm > 0
            nseen = seen.sum(0)
            ninv = inview.sum(0)
            body_share = np.where(ninv > 0, hidden.sum(0) / np.maximum(ninv, 1), 0.0)
            e1 = np.where(p1 > 0, edge[b1, cols], 1.0)
            d1 = np.where(p1 > 0, dep[b1, cols], 0.0)
            Cz = np.array([c.C[2] for c in cams])
            h1 = np.where(p1 > 0, Cz[b1], 0.0)
            Cs = np.array([c.C for c in cams])
            base = np.where((p2 > 0) & (p1 > 0), np.linalg.norm(Cs[b1] - Cs[b2], axis=1), 0.0)
        else:
            p1 = p2 = nseen = body_share = base = np.zeros(len(P))
            e1 = np.ones(len(P))
            d1 = h1 = np.zeros(len(P))
        lp = np.zeros(len(P))
        for L in lidars:
            lp += L.points(Gf, 0.0, h, wd)
        rd = np.zeros(len(P), bool)
        for Rd in radars:
            rd |= Rd.sees(P, CLASS_RCS.get(cls, 10.0))
        lpx1 = np.log2(np.maximum(h * p1, PX_FLOOR)).reshape(nR, nP)
        lpx2 = np.log2(np.maximum(h * p2, PX_FLOOR)).reshape(nR, nP)

        def med(a):
            return np.median(np.asarray(a, float).reshape(nR, nP), axis=1)

        def mean(a):
            return np.asarray(a, float).reshape(nR, nP).mean(axis=1)
        X[cls] = np.stack([
            np.median(lpx1, 1), np.percentile(lpx1, 25, axis=1), np.median(lpx2, 1),
            mean(np.asarray(nseen) > 0), mean(nseen), mean(body_share), med(e1), med(d1), med(h1), med(base),
            med(np.log2(1.0 + lp)), mean(rd), np.log2(np.maximum(rcentre, 1.0))], 1)
    return {'schema': 'vantage.perf_features/1', 'version': FEATURES_VERSION, 'names': list(FEATURE_NAMES),
            'regions': [rid for _, _, rid, _ in grid.regions()], 'classes': list(classes), 'X': X,
            'grid': grid.to_dict(), 'rig_hash': rig_hash(rig), 'cameras': nC, 'lenses_known': int(known),
            'lidars': len(lidars), 'radars': len(radars), 'occlusion': bool(body),
            'seconds': round(time.perf_counter() - t0, 4)}


def features_json(feat: dict) -> dict:
    """Features as plain JSON (lists)."""
    out = {k: v for k, v in feat.items() if k != 'X'}
    out['X'] = {c: np.round(np.asarray(a, float), 5).tolist() for c, a in feat['X'].items()}
    return out


def features_from_json(d: dict) -> dict:
    out = dict(d)
    out['X'] = {c: np.asarray(a, float) for c, a in (d.get('X') or {}).items()}
    return out


class FeatureCache:
    """Features of record rigs, by rig hash, grid and features version (and whether the body occludes), kept in memory
    and optionally in a JSON file beside the store."""

    def __init__(self, path=None):
        self.path = path
        self.mem = {}
        self._loaded = False

    @staticmethod
    def key(doc, occlusion) -> str:
        return '|'.join((doc.get('rig_hash', ''), _sha(doc.get('grid') or {})[:16], str(FEATURES_VERSION),
                         'body' if occlusion else 'free'))

    def _load(self):
        if self._loaded or not self.path:
            self._loaded = True
            return
        self._loaded = True
        try:
            with open(self.path, encoding='utf-8') as fh:
                d = json.load(fh)
            if d.get('version') == FEATURES_VERSION:
                self.mem.update({k: features_from_json(v) for k, v in (d.get('rigs') or {}).items()})
        except (OSError, ValueError, AttributeError):
            pass

    def get(self, doc, project=None, engine=None, occlusion=True):
        self._load()
        k = self.key(doc, occlusion and project is not None)
        f = self.mem.get(k)
        if f is None:
            f = describe_rig(doc['rig'], grid=doc.get('grid'), project=project, engine=engine, occlusion=occlusion)
            self.mem[k] = f
        return f

    def save(self):
        if not self.path:
            return
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = self.path + '.part'
        with open(tmp, 'w', encoding='utf-8') as fh:
            json.dump({'version': FEATURES_VERSION, 'rigs': {k: features_json(v) for k, v in self.mem.items()}}, fh)
        os.replace(tmp, self.path)


# =========================================================================== the Gaussian process
SQ5 = math.sqrt(5.0)
GH_X, GH_W = np.polynomial.hermite_e.hermegauss(24)       # probabilists' Gauss–Hermite: ∫ f(x) φ(x) dx
GH_W = GH_W / GH_W.sum()


def sigmoid(x):
    x = np.asarray(x, float)
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def logistic_moments(mu, var):
    """Mean and variance of sigmoid(f) for f ~ N(mu, var) (Gauss–Hermite, 24 nodes)."""
    mu = np.asarray(mu, float)
    sd = np.sqrt(np.maximum(np.asarray(var, float), 0.0))
    s = sigmoid(mu[..., None] + sd[..., None] * GH_X)
    m = (s * GH_W).sum(-1)
    v = (s * s * GH_W).sum(-1) - m * m
    return m, np.maximum(v, 0.0)


def _sqdist(A, B):
    a = (A * A).sum(1)[:, None]
    b = (B * B).sum(1)[None, :]
    return np.maximum(a + b - 2.0 * A @ B.T, 0.0)


def matern52(X1, X2, ell, sf2):
    """ARD Matérn-5/2: sf2 (1 + √5 r + 5/3 r²) exp(−√5 r), r² = Σ ((x − x') / ℓ)²."""
    r = np.sqrt(_sqdist(X1 / ell, X2 / ell))
    return sf2 * (1.0 + SQ5 * r + 5.0 / 3.0 * r * r) * np.exp(-SQ5 * r)


def _same(g):
    """The same-group indicator matrix of labels g (compared as integer codes)."""
    c = np.unique(np.asarray(g), return_inverse=True)[1].ravel()
    return (c[:, None] == c[None, :]).astype(float)


def _nll_and_grad(theta, X, r, s2, d, S=None, Q=None):
    """Negative log marginal likelihood of residuals r with K = k(X, X) + σs² S + σq² Q + diag(φ s2 + σn²), and its
    gradient in θ = [log ℓ (d), log σf, log σn, log φ (, log σs, log σq)] (R&W eq. 5.9: ½ tr((K⁻¹ − α αᵀ) ∂K)). S and Q
    are the same-scene and same-scene-and-region indicator matrices (the scene's and the scene × region's random
    offsets), or None."""
    from scipy.linalg import cho_factor, cho_solve
    ell = np.exp(theta[:d])
    sf2 = math.exp(2 * theta[d])
    sn2 = math.exp(2 * theta[d + 1])
    phi = math.exp(theta[d + 2])
    n = len(r)
    A = X / ell
    R2 = _sqdist(A, A)
    Rr = np.sqrt(R2)
    E = np.exp(-SQ5 * Rr)
    Kf = sf2 * (1.0 + SQ5 * Rr + 5.0 / 3.0 * R2) * E
    K = Kf + np.diag(phi * s2 + sn2 + 1e-8)
    groups = S is not None
    if groups:
        ss2, sq2 = math.exp(2 * theta[d + 3]), math.exp(2 * theta[d + 4])
        K = K + ss2 * S + sq2 * Q
    try:
        cf = cho_factor(K, lower=True, check_finite=False)
    except np.linalg.LinAlgError:
        return 1e25, np.zeros_like(theta)
    alpha = cho_solve(cf, r, check_finite=False)
    logdet = 2.0 * np.log(np.diag(cf[0])).sum()
    nll = 0.5 * float(r @ alpha) + 0.5 * logdet + 0.5 * n * math.log(2 * math.pi)
    from scipy.linalg import lapack
    Ki, info = lapack.dpotri(cf[0], lower=1)
    if info != 0:
        Ki = cho_solve(cf, np.eye(n), check_finite=False)
    else:
        Ki = np.tril(Ki) + np.tril(Ki, -1).T
    W = Ki - np.outer(alpha, alpha)
    g = np.zeros_like(theta)
    M = W * (sf2 * (5.0 / 3.0) * (1.0 + SQ5 * Rr) * E)         # ∂K/∂log ℓj = M-weights ⊙ (a_i − a_j)²
    rs = M.sum(1)
    MA = M @ A
    # Σ_ij M_ij (a_i − a_j)² = 2 Σ_i a_i² rowsum_i − 2 aᵀ M a  (M symmetric)
    g[:d] = 0.5 * (2.0 * (A * A).T @ rs - 2.0 * (A * MA).sum(0))
    g[d] = 0.5 * float((W * (2.0 * Kf)).sum())
    g[d + 1] = 0.5 * float(np.trace(W) * 2.0 * sn2)
    g[d + 2] = 0.5 * float((np.diag(W) * phi * s2).sum())
    if groups:
        g[d + 3] = 0.5 * float((W * S).sum() * 2.0 * ss2)
        g[d + 4] = 0.5 * float((W * Q).sum() * 2.0 * sq2)
    return nll, g


def _kmeanspp(X, m, rng):
    n = len(X)
    if n <= m:
        return X.copy()
    idx = [int(rng.integers(n))]
    d2 = ((X - X[idx[0]]) ** 2).sum(1)
    for _ in range(m - 1):
        p = d2 / d2.sum() if d2.sum() > 0 else np.full(n, 1.0 / n)
        i = int(rng.choice(n, p=p))
        idx.append(i)
        d2 = np.minimum(d2, ((X - X[i]) ** 2).sum(1))
    return X[np.array(idx)]


def fit_logistic_prior(z, succ, trials):
    """recall = sigmoid(a + b z) by binomial maximum likelihood with b ≥ 0 (the monotone prior: more pixels on the
    object, more recall). Returns (a, b)."""
    from scipy.optimize import minimize
    z, k, n = (np.asarray(v, float) for v in (z, succ, trials))

    def f(p):
        eta = p[0] + p[1] * z
        ll = k * eta - n * np.logaddexp(0.0, eta)
        s = sigmoid(eta)
        r = k - n * s
        return -float(ll.sum()), -np.array([r.sum(), (r * z).sum()])
    p0 = np.array([float(logit((k.sum() + 0.5) / (n.sum() + 1.0))), 0.0])
    res = minimize(f, p0, jac=True, method='L-BFGS-B', bounds=[(-30, 30), (0.0, 10.0)])
    return float(res.x[0]), float(res.x[1])


def empirical_logit(succ, trials):
    k = np.asarray(succ, float)
    n = np.asarray(trials, float)
    y = np.log((k + 0.5) / (n - k + 0.5))
    s2 = 1.0 / (k + 0.5) + 1.0 / (n - k + 0.5)
    return y, s2


@dataclass
class GPSettings:
    restarts: int = 3
    max_iter: int = 150
    sparse_n: int = SPARSE_N
    subset_n: int = SUBSET_N
    inducing: int = INDUCING_M
    seed: int = 0


class ClassModel:
    """One class and metric: the logistic prior in log2 pixels, an ARD Matérn-5/2 GP on the residual logit, and (with
    groups) random offsets per scene set and per scene set × region: rows of one drive share its objects and its
    town's difficulty, so they are not independent evidence, and a new scene set adds that much variance to every
    prediction (a hierarchical GP: the offsets are kernels σs²·[same scene] and σq²·[same scene and region])."""

    def __init__(self, cls, metric='recall', names=FEATURE_NAMES):
        self.cls, self.metric = cls, metric
        self.names = list(names)
        self.px_index = self.names.index('log2_px_obj')
        self.trained = False
        self.n_rows = 0
        self.groups = False
        self.ss2 = self.sq2 = 0.0
        self.off = []                      # feature columns left out (a camera-only detector: lidar and radar)

    # ---------------------------------------------------------------- fitting
    def fit(self, X, succ, trials, st: GPSettings | None = None, y=None, s2=None, theta0=None, scenes=None,
            regions=None, off=()):
        """X raw features (n × d); succ/trials the binomial counts (or y, s2 given directly: an empirical logit and
        its sampling variance); scenes/regions: per row its scene set and region (the random offsets), or None."""
        from scipy.optimize import minimize
        st = st or GPSettings()
        X = np.asarray(X, float)
        n, d = X.shape
        if y is None:
            y, s2 = empirical_logit(succ, trials)
            self.a, self.b = fit_logistic_prior(X[:, self.px_index], succ, trials)
            zz, kk, nn = X[:, self.px_index], np.asarray(succ, float), np.asarray(trials, float)
            pp = sigmoid(self.a + self.b * zz)
            wt = nn * pp * (1 - pp)
        else:
            y, s2 = np.asarray(y, float), np.asarray(s2, float)
            w = 1.0 / s2
            A = np.c_[np.ones(n), X[:, self.px_index]]
            coef = np.linalg.lstsq(A * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)[0]
            self.a, self.b = float(coef[0]), float(max(coef[1], 0.0))
            zz, wt = X[:, self.px_index], w
        # the prior's coefficients are uncertain too: their covariance from the fit's information matrix (with the
        # rows' dispersion, below), carried into every prediction as explicit basis functions (R&W 2006, §2.7)
        H = np.c_[np.ones(n), zz]
        info = (H * wt[:, None]).T @ H + 1e-9 * np.eye(2)
        self.B0 = np.linalg.inv(info)
        self.mu_x = X.mean(0)
        sd = X.std(0)
        self.sd_x = np.where(sd > 1e-9, sd, 1.0)
        self.off = sorted(int(j) for j in off)
        Z = self.std(X)
        r = y - self.prior(X)
        self.groups = scenes is not None
        sc = np.asarray([str(v) for v in scenes]) if self.groups else None
        sq = np.asarray([f'{a}|{b}' for a, b in zip(scenes, regions)]) if self.groups else None
        rng = np.random.default_rng(st.seed)
        sub = np.arange(n)
        if n > st.subset_n:
            sub = np.sort(rng.choice(n, st.subset_n, replace=False))
        Zs, rs, s2s = Z[sub], r[sub], s2[sub]
        args = (Zs, rs, s2s, d) + ((_same(sc[sub]), _same(sq[sub])) if self.groups else ())
        ng = 2 if self.groups else 0
        lo = np.r_[np.full(d, math.log(0.1)), math.log(0.02), math.log(1e-3), 0.0, np.full(ng, math.log(1e-3))]
        hi = np.r_[np.full(d, math.log(50.0)), math.log(5.0), math.log(3.0), math.log(200.0), np.full(ng, math.log(3.0))]
        v0 = max(float(np.var(rs)), 0.05)
        starts = [np.r_[np.full(d, math.log(2.0)), 0.5 * math.log(v0), math.log(0.1), math.log(2.0),
                        np.full(ng, math.log(0.3))]]
        if theta0 is not None and len(theta0) == d + 3 + ng:
            starts.insert(0, np.clip(np.asarray(theta0, float), lo, hi))
        for _ in range(max(st.restarts - 1, 0)):
            starts.append(lo + (hi - lo) * rng.uniform(0.15, 0.85, d + 3 + ng))
        best = None
        for s0 in starts[:max(st.restarts, 1) + (theta0 is not None)]:
            try:
                res = minimize(_nll_and_grad, s0, args=args, jac=True, method='L-BFGS-B',
                               bounds=list(zip(lo, hi)), options={'maxiter': st.max_iter})
            except (ValueError, np.linalg.LinAlgError):
                continue
            if best is None or res.fun < best.fun:
                best = res
        if best is None:
            raise RuntimeError(f'{self.cls}: the marginal likelihood could not be optimised')
        self.theta = best.x
        self.nll = float(best.fun)
        self._unpack()
        self._posterior(Z, r, s2, st, rng, sc, sq)
        self.n_rows = n
        self.trained = True
        return self

    def _unpack(self):
        d = len(self.names)
        self.ell = np.exp(self.theta[:d])
        self.sf2 = math.exp(2 * self.theta[d])
        self.sn2 = math.exp(2 * self.theta[d + 1])
        self.phi = math.exp(self.theta[d + 2])
        self.groups = len(self.theta) == d + 5
        self.ss2 = math.exp(2 * self.theta[d + 3]) if self.groups else 0.0
        self.sq2 = math.exp(2 * self.theta[d + 4]) if self.groups else 0.0

    def _posterior(self, Z, r, s2, st: GPSettings | None = None, rng=None, sc=None, sq=None):
        """The posterior for standardised inputs Z, residuals r, sampling variances s2 and the rows' scene and
        scene × region labels: exact below sparse_n rows, else FITC on k-means++ inducing points (the group offsets
        then only on the diagonal: an approximation)."""
        from scipy.linalg import cho_solve, solve_triangular
        st = st or GPSettings()
        noise = self.phi * s2 + self.sn2
        n = len(r)
        self.Ztrain, self.rtrain, self.s2train = Z, r, s2
        self.sc_train = np.asarray(sc if sc is not None else [''] * n)
        self.sq_train = np.asarray(sq if sq is not None else [''] * n)
        if n <= st.sparse_n:
            K = matern52(Z, Z, self.ell, self.sf2) + np.diag(noise + 1e-8)
            if self.groups:
                K += self.ss2 * _same(self.sc_train) + self.sq2 * _same(self.sq_train)
            L = np.linalg.cholesky(K)
            self.kind = 'exact'
            self.U = Z
            self.L = L
            self._Ki = None
            self.alpha = cho_solve((L, True), r)
            self._basis(lambda M: cho_solve((L, True), M))
            return
        rng = rng or np.random.default_rng(st.seed)
        self._posterior_fitc_at(_kmeanspp(Z, min(st.inducing, n), rng), Z, r, s2)

    def _posterior_fitc_at(self, U, Z, r, s2):
        from scipy.linalg import solve_triangular
        noise = self.phi * s2 + self.sn2
        Kmm = matern52(U, U, self.ell, self.sf2) + 1e-6 * self.sf2 * np.eye(len(U))
        Lm = np.linalg.cholesky(Kmm)
        Kmn = matern52(U, Z, self.ell, self.sf2)
        A = solve_triangular(Lm, Kmn, lower=True)
        lam = np.maximum(self.sf2 - (A * A).sum(0), 0.0) + noise + self.ss2 + self.sq2
        B = np.eye(len(U)) + (A / lam) @ A.T
        LB = np.linalg.cholesky(B)
        c = solve_triangular(LB, (A / lam) @ r, lower=True)
        tmp = solve_triangular(LB.T, c, lower=False)
        self.kind = 'fitc'
        self.U, self.Lm, self.LB = U, Lm, LB
        self.w = solve_triangular(Lm.T, tmp, lower=False)
        # the prior's coefficients: their posterior with the rows taken as independent (Λ, FITC's diagonal)
        H = self._H(Z)
        self.Ainv = np.linalg.inv(np.linalg.inv(self._B()) + (H / lam) @ H.T)
        self.delta = np.zeros(2)
        self.KiH = None

    def _B(self):
        """The prior coefficients' covariance: the fit's, times the dispersion the GP found (≥ 1)."""
        B0 = getattr(self, 'B0', None)
        if B0 is None:
            return np.eye(2) * 1e-12
        return np.asarray(B0, float) * max(getattr(self, 'phi', 1.0), 1.0)

    def _H(self, Z):
        """Basis rows [1, log2 px] (2 × n) for standardised inputs Z."""
        z = Z[:, self.px_index] * self.sd_x[self.px_index] + self.mu_x[self.px_index]
        return np.vstack([np.ones(len(Z)), z])

    def _basis(self, solve):
        """The explicit-basis posterior (R&W eqs. 2.41–2.42) for the exact GP: the coefficients' correction δ, the
        weights w = α − K⁻¹Hᵀδ and A⁻¹ = (B⁻¹ + H K⁻¹ Hᵀ)⁻¹."""
        H = self._H(self.Ztrain)
        KiH = solve(H.T)                                  # n × 2
        self.Ainv = np.linalg.inv(np.linalg.inv(self._B()) + H @ KiH)
        self.delta = self.Ainv @ (H @ self.alpha)
        self.KiH = KiH
        self.w = self.alpha - KiH @ self.delta

    def mean(self, X):
        """The posterior mean of the latent logit at raw features X for a new scene set."""
        Z = self.std(X)
        return self.prior(X) + self._H(Z).T @ self.delta + matern52(Z, self.U, self.ell, self.sf2) @ self.w

    def prior(self, X):
        return self.a + self.b * np.asarray(X, float)[:, self.px_index]

    def std(self, X):
        Z = (np.asarray(X, float) - self.mu_x) / self.sd_x
        if self.off:
            Z[:, self.off] = 0.0
        return Z

    # ---------------------------------------------------------------- prediction
    def predict(self, X, full_cov=False, scenes=None, regions=None, offsets=True):
        """Latent logit mean and variance (or covariance) at raw features X. scenes/regions: the rows' scene set and
        region when they belong to a scene set of the training data (its offsets are then known); by default a new
        scene set, whose offsets add σs² (shared by the rows) and σq² (per region) to the variance."""
        from scipy.linalg import solve_triangular
        X = np.asarray(X, float)
        Z = self.std(X)
        m = len(Z)
        Ks = matern52(Z, self.U, self.ell, self.sf2)
        known = scenes is not None and self.groups and self.kind == 'exact'
        if known:
            sc = np.asarray([str(v) for v in scenes])
            sq = np.asarray([f'{a}|{b}' for a, b in zip(scenes, regions)])
            Ks = Ks + self.ss2 * (sc[:, None] == self.sc_train[None, :]) + \
                self.sq2 * (sq[:, None] == self.sq_train[None, :])
        Hs = self._H(Z)
        mu = self.prior(X) + Hs.T @ self.delta + Ks @ self.w
        if self.groups and (offsets or known):
            if known:
                Gss = self.ss2 * (sc[:, None] == sc[None, :]) + self.sq2 * (sq[:, None] == sq[None, :])
            else:                               # one new scene set: a shared offset, and one per region
                Gss = self.ss2 * np.ones((m, m)) + self.sq2 * np.eye(m)
        else:
            Gss = np.zeros((m, m))
        if self.kind == 'exact':
            KsKi = Ks @ self._kinv()                      # a matrix product (BLAS): milliseconds for a rig
            R = Hs - self.KiH.T @ Ks.T                    # 2 × m
            if full_cov:
                return mu, matern52(Z, Z, self.ell, self.sf2) + Gss - KsKi @ Ks.T + R.T @ self.Ainv @ R
            return mu, np.maximum(self.sf2 + np.diag(Gss) - (KsKi * Ks).sum(1) +
                                  ((self.Ainv @ R) * R).sum(0), 1e-12)
        V1 = solve_triangular(self.Lm, Ks.T, lower=True)
        V2 = solve_triangular(self.LB, V1, lower=True)
        if full_cov:
            return mu, matern52(Z, Z, self.ell, self.sf2) + Gss - V1.T @ V1 + V2.T @ V2 + Hs.T @ self.Ainv @ Hs
        return mu, np.maximum(self.sf2 + np.diag(Gss) - (V1 * V1).sum(0) + (V2 * V2).sum(0) +
                              ((self.Ainv @ Hs) * Hs).sum(0), 1e-12)

    def _kinv(self):
        """K⁻¹ of the exact posterior, from its Cholesky factor, computed once."""
        Ki = getattr(self, '_Ki', None)
        if Ki is None or Ki.shape[0] != self.L.shape[0]:
            from scipy.linalg import cho_solve, lapack
            Ki, info = lapack.dpotri(self.L, lower=1)
            Ki = np.tril(Ki) + np.tril(Ki, -1).T if info == 0 else cho_solve((self.L, True), np.eye(len(self.L)))
            self._Ki = Ki
        return Ki

    def noise_var(self, s2):
        return self.phi * np.asarray(s2, float) + self.sn2

    def grad(self, X):
        """∂ mean / ∂ raw feature at X (n × d), for a new scene set: the prior's slope plus the kernel's analytic
        gradient."""
        X = np.asarray(X, float)
        Z = self.std(X)
        D = (Z[:, None, :] - self.U[None, :, :]) / self.ell                # (n, m, d)
        r = np.sqrt((D * D).sum(-1))
        dk_dr_over_r = -self.sf2 * (5.0 / 3.0) * (1.0 + SQ5 * r) * np.exp(-SQ5 * r)   # (dk/dr) / r
        G = np.einsum('nm,nmd,m->nd', dk_dr_over_r, D / self.ell, self.w)
        G = G / self.sd_x
        if self.off:
            G[:, self.off] = 0.0
        G[:, self.px_index] += self.b + self.delta[1]
        return G

    def condition(self, X, y_latent):
        """A copy with pseudo-observations (the kriging believer's): rows X observed at latent values y_latent with
        only the homoscedastic noise σn² (no sampling noise), in a scene set of their own. Hyperparameters
        unchanged; the exact posterior's Cholesky factor is extended by a block (O(n² m)), not refactored."""
        from scipy.linalg import cho_solve, solve_triangular
        m = copy.copy(self)
        Zn = self.std(X)
        k = len(Zn)
        Z = np.vstack([self.Ztrain, Zn])
        r = np.r_[self.rtrain, np.asarray(y_latent, float) - self.prior(X)]
        s2 = np.r_[self.s2train, np.zeros(k)]
        tag = f'believer-{len(r)}'
        sc = np.r_[self.sc_train, np.full(k, tag)]
        sq = np.r_[self.sq_train, np.array([f'{tag}|{i}' for i in range(k)])]
        if self.kind != 'exact':
            m._posterior(Z, r, s2, GPSettings(sparse_n=SPARSE_N), None, sc, sq)
            return m
        K12 = matern52(self.Ztrain, Zn, self.ell, self.sf2)          # no shared group with the training rows
        K22 = matern52(Zn, Zn, self.ell, self.sf2) + np.eye(k) * (self.sn2 + 1e-8)
        if self.groups:
            K22 += self.ss2 + self.sq2 * np.eye(k)
        B = solve_triangular(self.L, K12, lower=True)
        L22 = np.linalg.cholesky(K22 - B.T @ B + 1e-10 * np.eye(k))
        n0 = len(self.rtrain)
        L = np.zeros((n0 + k, n0 + k))
        L[:n0, :n0] = self.L
        L[n0:, :n0] = B.T
        L[n0:, n0:] = L22
        m.Ztrain, m.rtrain, m.s2train, m.sc_train, m.sq_train = Z, r, s2, sc, sq
        m.U, m.L = Z, L
        m._Ki = None
        m.alpha = cho_solve((L, True), r)
        m._basis(lambda M: cho_solve((L, True), M))
        return m

    # ---------------------------------------------------------------- persistence
    def to_dict(self) -> dict:
        d = {'cls': self.cls, 'metric': self.metric, 'names': self.names, 'trained': self.trained,
             'n_rows': self.n_rows}
        if not self.trained:
            return d
        d.update({'a': self.a, 'b': self.b, 'theta': self.theta.tolist(), 'nll': self.nll,
                  'mu_x': self.mu_x.tolist(), 'sd_x': self.sd_x.tolist(), 'kind': self.kind,
                  'Ztrain': self.Ztrain.tolist(), 'rtrain': self.rtrain.tolist(), 's2train': self.s2train.tolist(),
                  'scenes': self.sc_train.tolist(), 'scene_regions': self.sq_train.tolist(), 'off': self.off,
                  'prior_cov': np.asarray(self.B0).tolist()})
        if self.kind == 'fitc':
            d['U'] = self.U.tolist()
        return d

    @staticmethod
    def from_dict(d) -> 'ClassModel':
        m = ClassModel(d['cls'], d.get('metric', 'recall'), d.get('names', FEATURE_NAMES))
        m.n_rows = int(d.get('n_rows', 0))
        if not d.get('trained'):
            return m
        m.a, m.b = float(d['a']), float(d['b'])
        m.theta = np.asarray(d['theta'], float)
        m.nll = float(d.get('nll', 0.0))
        m.mu_x, m.sd_x = np.asarray(d['mu_x'], float), np.asarray(d['sd_x'], float)
        m.off = [int(j) for j in d.get('off') or []]
        m.B0 = np.asarray(d.get('prior_cov') or np.eye(2) * 1e-12, float)
        m._unpack()
        Z, r, s2 = (np.asarray(d[k], float) for k in ('Ztrain', 'rtrain', 's2train'))
        sc, sq = d.get('scenes'), d.get('scene_regions')
        st = GPSettings(sparse_n=SPARSE_N if d.get('kind') == 'fitc' else max(len(r), SPARSE_N))
        if d.get('kind') == 'fitc':
            st.inducing = len(d.get('U') or []) or INDUCING_M
        m._posterior(Z, r, s2, st, np.random.default_rng(0), sc, sq)
        if d.get('kind') == 'fitc' and d.get('U'):        # the saved inducing points, exactly
            m._posterior_fitc_at(np.asarray(d['U'], float), Z, r, s2)
        m.trained = True
        return m


# =========================================================================== rows from records
def _rows_of(doc, cls, metric='recall'):
    """(region ids, successes, trials, y, s2) of one record's class (rows with objects / detections only)."""
    regs = ((doc.get('metrics') or {}).get(cls) or {})
    rid, k, n, ys, vs = [], [], [], [], []
    for r_id, c in regs.items():
        if metric == 'recall':
            nn, kk = int(c.get('n_gt') or 0), int(c.get('tp') or 0)
        elif metric == 'precision':
            td = c.get('tp_det', c.get('tp'))
            nn, kk = int((td or 0) + (c.get('fp') or 0)), int(td or 0)
        else:                                            # ap: logit of AP with a binomial-like variance on n_gt
            ap, nn = c.get('ap'), int(c.get('n_gt') or 0)
            if ap is None or nn < 1:
                continue
            p = min(max(float(ap), 0.5 / nn), 1 - 0.5 / nn) if nn > 1 else 0.5
            rid.append(r_id)
            k.append(float(ap) * nn)
            n.append(nn)
            ys.append(float(logit(p)))
            vs.append(1.0 / (nn * p * (1 - p)))
            continue
        if nn < 1:
            continue
        rid.append(r_id)
        k.append(kk)
        n.append(nn)
    if metric != 'ap':
        y, s2 = empirical_logit(k, n) if rid else (np.zeros(0), np.zeros(0))
        return rid, np.asarray(k, float), np.asarray(n, float), np.asarray(y, float), np.asarray(s2, float)
    return rid, np.asarray(k, float), np.asarray(n, float), np.asarray(ys, float), np.asarray(vs, float)


def training_rows(docs, feats_of, cls, metric='recall'):
    """Stacked rows of one class over records: X, successes, trials, y, s2, and per row the record index, scene key
    and rig hash."""
    Xs, K, Nn, Y, V, rec, scn, rig, reg = [], [], [], [], [], [], [], [], []
    for i, d in enumerate(docs):
        rid, k, n, y, s2 = _rows_of(d, cls, metric)
        if not rid:
            continue
        f = feats_of(d)
        if cls not in f['X']:
            continue
        idx = {r: j for j, r in enumerate(f['regions'])}
        keep = [j for j, r in enumerate(rid) if r in idx]
        if not keep:
            continue
        Xs.append(f['X'][cls][[idx[rid[j]] for j in keep]])
        K.append(k[keep])
        Nn.append(n[keep])
        Y.append(y[keep])
        V.append(s2[keep])
        rec += [i] * len(keep)
        scn += [scene_key(d)] * len(keep)
        rig += [d.get('rig_hash', '')] * len(keep)
        reg += [rid[j] for j in keep]
    if not Xs:
        return None
    return {'X': np.vstack(Xs), 'k': np.concatenate(K), 'n': np.concatenate(Nn), 'y': np.concatenate(Y),
            's2': np.concatenate(V), 'rec': np.asarray(rec), 'scene': np.asarray(scn), 'rig': np.asarray(rig),
            'region': np.asarray(reg)}


def detector_key(doc) -> str:
    """The detector family a record measures: id · backend · evaluator (records of different families measure
    different things, e.g. 2D detections counted per rig against a lifted 3D detector matched in bird's-eye view, and
    are never pooled in one model)."""
    d = doc.get('detector') or {}
    return ' · '.join(str(x) for x in (d.get('id', '?'), d.get('backend', '?'), d.get('evaluator') or 'measured',
                                       d.get('settings_hash') or 'default'))


def scene_key(doc) -> str:
    sc = doc.get('scene') or {}
    return sc.get('scene_hash') or scene_hash(sc)


# =========================================================================== the model
Z68, Z95 = 0.9944578832097535, 1.959963984540054
CURVE_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.68, 0.7, 0.8, 0.9, 0.95, 0.99)


@dataclass
class PerfModelSettings:
    classes: list = field(default_factory=lambda: list(CLASSES))
    metrics: list = field(default_factory=lambda: ['recall'])
    restarts: int = 3
    max_iter: int = 150
    min_records: int = MIN_RECORDS
    min_rows: int = MIN_ROWS
    sparse_n: int = SPARSE_N
    inducing: int = INDUCING_M
    cv: bool = True
    cv_max_folds: int = 10
    objective: dict = field(default_factory=lambda: {'pedestrian': 1.0, 'vehicle': 1.0})   # class weights
    max_range_m: float = 60.0           # the objective's regions: those starting nearer than this
    kappa: float = 1.0                  # the optimiser's term: mean − κ·std
    mode: str = 'ei'                    # ei | std | ucb
    k: int = 4
    reference: str = ''                 # the variant attributions compare against (uid or name; '' = the first)
    detector: str = ''                  # the detector family trained on (detector_key; '' = the most records)
    store_in_project: bool = False
    occlusion: bool = True
    scene_effects: bool = True          # random offsets per scene set and scene set × region (hierarchical GP)
    seed: int = 0

    def to_dict(self):
        return {k: (list(v) if isinstance(v, (list, tuple)) else (dict(v) if isinstance(v, dict) else v))
                for k, v in self.__dict__.items()}

    @staticmethod
    def from_dict(d) -> 'PerfModelSettings':
        s = PerfModelSettings()
        d = d if isinstance(d, dict) else {}
        c = d.get('classes')
        if isinstance(c, (list, tuple)):
            cc = [x for x in c if x in CLASSES]
            if cc:
                s.classes = cc
        m = d.get('metrics')
        if isinstance(m, (list, tuple)):
            mm = [x for x in m if x in METRICS]
            if mm:
                s.metrics = mm
        for k, (lo, hi) in (('restarts', LIMITS['restarts']), ('max_iter', LIMITS['max_iter']),
                            ('min_records', LIMITS['min_records']), ('min_rows', LIMITS['min_rows']),
                            ('sparse_n', LIMITS['sparse_n']), ('inducing', LIMITS['inducing']),
                            ('cv_max_folds', (2, 100)), ('k', (1, 50)), ('seed', LIMITS['seed'])):
            v = _f(d.get(k))
            if v is not None and not isinstance(d.get(k), bool):
                setattr(s, k, int(round(min(max(v, lo), hi))))
        for k in ('kappa', 'max_range_m'):
            v = _f(d.get(k))
            if v is not None and not isinstance(d.get(k), bool):
                lo, hi = LIMITS[k]
                setattr(s, k, float(min(max(v, lo), hi)))
        for k in ('cv', 'store_in_project', 'occlusion', 'scene_effects'):
            if isinstance(d.get(k), bool):
                setattr(s, k, d[k])
        if d.get('mode') in ('ei', 'std', 'ucb'):
            s.mode = d['mode']
        if isinstance(d.get('reference'), str):
            s.reference = d['reference'][:80]
        if isinstance(d.get('detector'), str):
            s.detector = d['detector'][:200]
        ob = d.get('objective')
        if isinstance(ob, dict):
            o = {k: float(v) for k, v in ob.items() if k in CLASSES and _f(v) is not None and float(v) > 0}
            if o:
                s.objective = o
        return s

    def gp(self, restarts=None, max_iter=None) -> GPSettings:
        return GPSettings(restarts=restarts or self.restarts, max_iter=max_iter or self.max_iter,
                          sparse_n=self.sparse_n, inducing=self.inducing, seed=self.seed)


def settings_of(project) -> PerfModelSettings:
    return PerfModelSettings.from_dict(settings_dict(project))


def _ml_noise(res, s2):
    """(φ, τ²) of a baseline by maximum likelihood: res ~ N(0, φ s2 + τ²)."""
    from scipy.optimize import minimize

    def f(p):
        v = math.exp(p[0]) * s2 + math.exp(p[1])
        return 0.5 * float((np.log(v) + res * res / v).sum())
    r = minimize(f, np.array([math.log(2.0), math.log(max(np.var(res), 1e-3) / 2)]), method='L-BFGS-B',
                 bounds=[(0.0, math.log(200.0)), (math.log(1e-6), math.log(25.0))])
    return math.exp(r.x[0]), math.exp(r.x[1])


def _scores(y, mu, var, p_obs, p_pred, n) -> dict:
    sd = np.sqrt(var)
    a = np.abs(y - mu) / sd
    from scipy.stats import norm
    curve = [[lv, float(np.mean(a <= norm.ppf(0.5 + lv / 2)))] for lv in CURVE_LEVELS]
    w = n / n.sum()
    return {'n': int(len(y)), 'coverage68': float(np.mean(a <= Z68)), 'coverage95': float(np.mean(a <= Z95)),
            'rmse': float(np.sqrt(np.mean((p_obs - p_pred) ** 2))),
            'rmse_weighted': float(np.sqrt((w * (p_obs - p_pred) ** 2).sum())),
            'nlpd': float(np.mean(0.5 * np.log(2 * math.pi * var) + 0.5 * (y - mu) ** 2 / var)), 'curve': curve}


def cross_validate(rows, cls, metric, by, st: PerfModelSettings, progress=None, cancel=None, off=()) -> dict:
    """Leave-one-group-out over scene sets (by='scene') or rigs (by='rig', at most st.cv_max_folds folds of whole
    rigs): everything refitted per fold (prior, hyperparameters, posterior), scored on the held-out rows against the
    logistic-in-pixels and constant baselines with the same noise model."""
    groups = rows['scene'] if by == 'scene' else rows['rig']
    ug = sorted(set(groups.tolist()))
    if len(ug) < 2:
        return {'ok': False, 'why': f'needs two {"scene sets" if by == "scene" else "rigs"} (has {len(ug)})'}
    rng = np.random.default_rng(st.seed + (1 if by == 'rig' else 0))
    nf = min(len(ug), st.cv_max_folds)
    perm = rng.permutation(len(ug))
    fold_of = {g: int(perm[i] % nf) for i, g in enumerate(ug)}
    fold = np.array([fold_of[g] for g in groups.tolist()])
    X, k, n, y, s2 = rows['X'], rows['k'], rows['n'], rows['y'], rows['s2']
    pz = X[:, list(FEATURE_NAMES).index('log2_px_obj')]
    out = {m: {'y': [], 'mu': [], 'var': [], 'p': []} for m in ('gp', 'logistic', 'constant')}
    obs_p, obs_n, folds_done = [], [], 0
    def grp(sel, fit=False):
        g = {'scenes': rows['scene'][sel], 'regions': rows['region'][sel]} if st.scene_effects else {}
        if fit:
            g['off'] = off
        return g

    def one(f):
        te, tr = fold == f, fold != f
        if tr.sum() < st.min_rows or te.sum() == 0 or (cancel and cancel()):
            return None
        cm = ClassModel(cls, metric)
        gst = st.gp(restarts=1, max_iter=min(st.max_iter, 80))
        try:
            if metric == 'ap':
                cm.fit(X[tr], None, None, gst, y=y[tr], s2=s2[tr], **grp(tr, True))
            else:
                cm.fit(X[tr], k[tr], n[tr], gst, **grp(tr, True))
        except RuntimeError:
            return None
        mu, var = cm.predict(X[te], **grp(te))
        r = {'gp': (y[te], mu, var + cm.noise_var(s2[te]), logistic_moments(mu, var)[0])}
        # baselines: the prior alone, and a constant, each with its own (φ, τ²)
        a, b = (cm.a, cm.b) if metric == 'ap' else fit_logistic_prior(pz[tr], k[tr], n[tr])
        ml = a + b * pz
        ph, t2 = _ml_noise(y[tr] - ml[tr], s2[tr])
        r['logistic'] = (y[te], ml[te], ph * s2[te] + t2, logistic_moments(ml[te], np.full(te.sum(), t2))[0])
        wc = 1.0 / s2[tr]
        c0 = float((wc * y[tr]).sum() / wc.sum())
        ph, t2 = _ml_noise(y[tr] - c0, s2[tr])
        r['constant'] = (y[te], np.full(te.sum(), c0), ph * s2[te] + t2,
                         logistic_moments(np.full(te.sum(), c0), np.full(te.sum(), t2))[0])
        r['obs'] = ((k[te] / n[te]) if metric != 'ap' else k[te] / np.maximum(n[te], 1), n[te])
        return r
    import concurrent.futures as cf
    with cf.ThreadPoolExecutor(max(1, min(CV_THREADS, nf))) as pool:
        results = list(pool.map(one, range(nf)))
    if cancel and cancel():
        return {'ok': False, 'why': 'cancelled'}
    for r in results:
        if r is None:
            continue
        for meth in ('gp', 'logistic', 'constant'):
            for key, v in zip(('y', 'mu', 'var', 'p'), r[meth]):
                out[meth][key].append(v)
        obs_p.append(r['obs'][0])
        obs_n.append(r['obs'][1])
        folds_done += 1
    if progress:
        progress(1.0)
    if not folds_done:
        return {'ok': False, 'why': 'no fold had enough training rows'}
    po, nn = np.concatenate(obs_p), np.concatenate(obs_n)
    res = {'ok': True, 'by': by, 'folds': folds_done, 'groups': len(ug)}
    for m, v in out.items():
        res[m] = _scores(np.concatenate(v['y']), np.concatenate(v['mu']), np.concatenate(v['var']), po,
                         np.concatenate(v['p']), nn)
    return res


class Model:
    """Per class (and metric) a ClassModel, trained on perception records. predict(features) -> per class × region
    recall mean, std and intervals, and the rig-level summary."""

    def __init__(self, settings: PerfModelSettings | None = None):
        self.st = settings or PerfModelSettings()
        self.models = {}                 # (cls, metric) -> ClassModel
        self.status = {}                 # cls -> text
        self.n_records = 0
        self.n_scenes = 0
        self.n_rigs = 0
        self.cv = {}                     # (cls, metric) -> {'scene': ..., 'rig': ...}
        self.train_rigs = {}             # rig hash -> features (for the incumbent and the store's rigs)
        self.grid = None
        self.detector = ''               # the detector family trained on (detector_key)
        self.families = {}               # every family in the records offered -> records
        self.fitted_utc = ''
        self.seconds = 0.0
        self.notes = []

    @property
    def trained(self) -> bool:
        return any(m.trained for m in self.models.values())

    def reason(self) -> str:
        need = self.st.min_records
        if self.n_records < need:
            return f'not trained ({self.n_records} records, need {need})'
        if not self.trained:
            return f'not trained (no class has {self.st.min_rows} rows with objects)'
        return 'trained'

    def camera_only(self) -> bool:
        """The detector family sees camera images only (the lifted backend, the interim 2D evaluator): its lidar and
        radar features are left out, so a rig's lidar or radar can never move a camera detector's prediction."""
        parts = self.detector.split(' · ')
        return len(parts) >= 2 and (parts[1] == 'lifted' or INTERIM in self.detector)

    def off_columns(self) -> list:
        return [FEATURE_NAMES.index('log2_lidar_pts'), FEATURE_NAMES.index('radar_share')] if self.camera_only() \
            else []

    def extrapolation(self, feats, metric='recall') -> dict:
        """Per class, the share of the rig's regions whose pixels on the object lie beyond the training rows' range
        (the model then leans on its monotone prior)."""
        out = {}
        for (c, m), cm in self.models.items():
            if m != metric or not cm.trained or c not in feats['X']:
                continue
            z = cm.Ztrain[:, cm.px_index] * cm.sd_x[cm.px_index] + cm.mu_x[cm.px_index]
            x = feats['X'][c][:, cm.px_index]
            out[c] = float(np.mean((x > z.max() + 0.25) | (x < z.min() - 0.25)))
        return out

    def classes(self, metric='recall') -> list:
        return [c for (c, m), cm in self.models.items() if m == metric and cm.trained]

    # ---------------------------------------------------------------- fitting
    def fit(self, docs, feats_of=None, project=None, engine=None, progress=None, cancel=None) -> 'Model':
        t0 = time.perf_counter()
        st = self.st
        docs = [d for d in docs if not validate_record(d)] if docs else []
        self.notes = []
        fam = {}
        for d in docs:
            fam[detector_key(d)] = fam.get(detector_key(d), 0) + 1
        self.families = fam
        self.detector = st.detector if st.detector in fam else (max(fam, key=fam.get) if fam else '')
        docs = [d for d in docs if detector_key(d) == self.detector]
        if len(fam) > 1:
            self.notes.append(f'Trained on {self.detector} ({fam.get(self.detector, 0)} records); other detectors\' '
                              'records are kept but not pooled: ' +
                              ', '.join(f'{k} ({n})' for k, n in fam.items() if k != self.detector))
        if feats_of is None:
            cache = FeatureCache()
            eng = engine or (shared_engine() if project is not None and st.occlusion else None)

            def feats_of(d):
                return cache.get(d, project, eng, st.occlusion)
        self.n_records = len(docs)
        self.n_scenes = len({scene_key(d) for d in docs})
        self.n_rigs = len({d.get('rig_hash') for d in docs})
        self.models, self.status, self.cv, self.train_rigs = {}, {}, {}, {}
        self.__dict__.pop("_incumbent", None)
        self.fitted_utc = _utc()
        if self.n_records < st.min_records:
            for c in st.classes:
                self.status[c] = f'not trained ({self.n_records} records, need {st.min_records})'
            self.seconds = time.perf_counter() - t0
            return self
        for d in docs:
            if d.get('rig_hash') not in self.train_rigs:
                self.train_rigs[d['rig_hash']] = feats_of(d)
            if self.grid is None:
                self.grid = d.get('grid')
        jobs = [(c, m) for m in st.metrics for c in st.classes]
        for j, (c, m) in enumerate(jobs):
            if cancel and cancel():
                return self
            rows = training_rows(docs, feats_of, c, m)
            n_rows = 0 if rows is None else len(rows['y'])
            recs = 0 if rows is None else len(set(rows['rec'].tolist()))
            key = c if m == 'recall' else f'{c}:{m}'
            if rows is None or n_rows < st.min_rows or recs < st.min_records:
                self.status[key] = (f'not trained ({recs} records with {c} objects, need {st.min_records})'
                                    if recs < st.min_records else
                                    f'not trained ({n_rows} rows with {c} objects, need {st.min_rows})')
                continue
            cm = ClassModel(c, m)
            grp = {'scenes': rows['scene'], 'regions': rows['region']} if st.scene_effects else {}
            grp['off'] = self.off_columns()
            try:
                if m == 'ap':
                    cm.fit(rows['X'], None, None, st.gp(), y=rows['y'], s2=rows['s2'], **grp)
                else:
                    cm.fit(rows['X'], rows['k'], rows['n'], st.gp(), **grp)
            except RuntimeError as e:
                self.status[key] = f'not trained ({e})'
                continue
            self.models[(c, m)] = cm
            self.status[key] = f'trained on {n_rows} rows of {recs} records'
            if st.cv:
                self.cv[(c, m)] = {by: cross_validate(rows, c, m, by, st, cancel=cancel, off=grp['off'])
                                   for by in ('scene', 'rig')}
            if progress:
                progress((j + 1) / len(jobs), f'{c} ({m})')
        self.seconds = time.perf_counter() - t0
        return self

    # ---------------------------------------------------------------- prediction
    def region_weights(self, regions, grid=None) -> np.ndarray:
        g = RG.RegionGrid.from_dict(grid or self.grid or {})
        e = [0.0] + [float(x) for x in g.range_edges]
        w = []
        for rid in regions:
            sr = g.parse_id(rid)
            w.append(1.0 if sr is not None and e[sr[1]] < self.st.max_range_m else 0.0)
        w = np.asarray(w, float)
        return w if w.sum() > 0 else np.ones(len(regions))

    def predict(self, feats, metric='recall') -> dict:
        """{'trained', 'reason', 'regions', 'classes': {cls: {'mean', 'std', 'lo68', 'hi68', 'lo95', 'hi95', 'mu',
        'var'}}, 'rig': {cls: {'mean', 'std'}}} — nothing when the model is not trained."""
        out = {'trained': self.trained, 'reason': self.reason(), 'regions': list(feats['regions']), 'classes': {},
               'rig': {}, 'metric': metric}
        if not self.trained:
            return out
        w = self.region_weights(feats['regions'], feats.get('grid'))
        for (c, m), cm in self.models.items():
            if m != metric or not cm.trained or c not in feats['X']:
                continue
            mu, cov = cm.predict(feats['X'][c], full_cov=True)
            var = np.maximum(np.diag(cov), 1e-12)
            pm, pv = logistic_moments(mu, var)
            sd = np.sqrt(var)
            out['classes'][c] = {'mean': pm, 'std': np.sqrt(pv), 'lo68': sigmoid(mu - Z68 * sd),
                                 'hi68': sigmoid(mu + Z68 * sd), 'lo95': sigmoid(mu - Z95 * sd),
                                 'hi95': sigmoid(mu + Z95 * sd), 'mu': mu, 'var': var}
            g = w * sigmoid(mu) * (1 - sigmoid(mu))
            out['rig'][c] = {'mean': float((w * pm).sum() / w.sum()),
                             'std': float(math.sqrt(max(g @ cov @ g, 0.0)) / w.sum())}
        return out

    def objective_mean(self, feats, objective=None) -> float:
        """The objective's posterior mean only (no covariance: cheap), for the incumbent."""
        ob = objective or self.st.objective
        w = self.region_weights(feats['regions'], feats.get('grid'))
        mean = wsum = 0.0
        for c, wc in ob.items():
            cm = self.models.get((c, 'recall'))
            if cm is None or not cm.trained or c not in feats['X']:
                continue
            X = feats['X'][c]
            mu = cm.mean(X)
            var = np.full(len(mu), cm.sf2 * 0.0 + cm.ss2 + cm.sq2)        # the new scene set's offsets only
            mean += wc * float((w * logistic_moments(mu, var)[0]).sum() / w.sum())
            wsum += wc
        return mean / wsum if wsum > 0 else math.nan

    def objective(self, feats, objective=None, regions=None, models=None, epistemic=False) -> tuple:
        """(mean, std) of the objective: Σ_class w_c · mean over the regions that count of predicted recall, / Σ w_c;
        std from each class's joint posterior over the rig's regions (delta method through the logistic). epistemic:
        without the new scene set's random offsets (what evaluating rigs can reduce; what active learning ranks)."""
        ob = objective or self.st.objective
        models = models or self.models
        w = self.region_weights(feats['regions'], feats.get('grid')) if regions is None else \
            np.array([float(regions.get(r, 0.0)) for r in feats['regions']])
        if w.sum() <= 0:
            w = np.ones(len(feats['regions']))
        mean = var = wsum = 0.0
        for c, wc in ob.items():
            cm = models.get((c, 'recall'))
            if cm is None or not cm.trained or c not in feats['X']:
                continue
            mu, cov = cm.predict(feats['X'][c], full_cov=True, offsets=not epistemic)
            pm, _ = logistic_moments(mu, np.maximum(np.diag(cov), 1e-12))
            g = w * sigmoid(mu) * (1 - sigmoid(mu)) / w.sum()
            mean += wc * float((w * pm).sum() / w.sum())
            var += wc * wc * float(max(g @ cov @ g, 0.0))
            wsum += wc
        if wsum <= 0:
            return math.nan, math.nan
        return mean / wsum, math.sqrt(var) / wsum

    def objective_term(self, feats, kappa=None, objective=None) -> float:
        """The optimiser's term: predicted objective mean − κ·std (higher is better); nan when not trained."""
        m, s = self.objective(feats, objective)
        return m - (self.st.kappa if kappa is None else kappa) * s

    def explain(self, feats, ref_feats, objective=None, top=6) -> list:
        """What moved the prediction from the reference rig to this one: per feature, its share of the change in the
        objective (integrated gradients of the GP's mean over three points on the straight path, × the feature's
        change, through the logistic, weighted as the objective). [{'feature', 'text', 'delta', 'where'}]."""
        ob = objective or self.st.objective
        w = self.region_weights(feats['regions'], feats.get('grid'))
        grid = RG.RegionGrid.from_dict(feats.get('grid') or {})
        contrib = np.zeros(len(FEATURE_NAMES))
        where = np.zeros((len(FEATURE_NAMES), len(feats['regions'])))
        wsum = sum(ob.values())
        for c, wc in ob.items():
            cm = self.models.get((c, 'recall'))
            if cm is None or not cm.trained or c not in feats['X'] or c not in ref_feats['X']:
                continue
            X1, X0 = feats['X'][c], ref_feats['X'][c]
            G = np.mean([cm.grad(X0 + t * (X1 - X0)) for t in (1 / 6, 0.5, 5 / 6)], axis=0)
            mu_mid = cm.predict(0.5 * (X0 + X1))[0]
            s = sigmoid(mu_mid) * (1 - sigmoid(mu_mid))
            part = G * (X1 - X0) * (s * w / w.sum() * wc / wsum)[:, None]       # (regions, features) in recall
            contrib += part.sum(0)
            where += part.T
        out = []
        for j in np.argsort(-np.abs(contrib))[:top]:
            if abs(contrib[j]) < 1e-4:
                continue
            r = int(np.argmax(np.abs(where[j])))
            sr = grid.parse_id(feats['regions'][r])
            name = grid.region_name(*sr) if sr else feats['regions'][r]
            verb = 'raised' if contrib[j] > 0 else 'lowered'
            out.append({'feature': FEATURE_NAMES[j], 'delta': float(contrib[j]), 'where': feats['regions'][r],
                        'text': f'{FEATURE_TEXT[FEATURE_NAMES[j]].capitalize()} {verb} the predicted objective by '
                                f'{abs(contrib[j]):.3f} (most in {name}).'})
        return out

    # ---------------------------------------------------------------- persistence
    def to_dict(self) -> dict:
        return {'schema': SCHEMA, 'settings': self.st.to_dict(), 'n_records': self.n_records,
                'detector': self.detector, 'families': self.families,
                'n_scenes': self.n_scenes, 'n_rigs': self.n_rigs, 'status': self.status, 'grid': self.grid,
                'fitted_utc': self.fitted_utc, 'seconds': round(self.seconds, 2), 'notes': self.notes,
                'models': [cm.to_dict() for cm in self.models.values()],
                'cv': [{'class': c, 'metric': m, **v} for (c, m), v in self.cv.items()],
                'train_rigs': {h: features_json(f) for h, f in self.train_rigs.items()}}

    @staticmethod
    def from_dict(d) -> 'Model':
        m = Model(PerfModelSettings.from_dict(d.get('settings')))
        m.n_records, m.n_scenes, m.n_rigs = (int(d.get(k) or 0) for k in ('n_records', 'n_scenes', 'n_rigs'))
        m.status = dict(d.get('status') or {})
        m.grid = d.get('grid')
        m.detector = str(d.get('detector') or '')
        m.families = dict(d.get('families') or {})
        m.fitted_utc = d.get('fitted_utc', '')
        m.seconds = float(d.get('seconds') or 0.0)
        m.notes = list(d.get('notes') or [])
        for md in d.get('models') or []:
            cm = ClassModel.from_dict(md)
            m.models[(cm.cls, cm.metric)] = cm
        for cv in d.get('cv') or []:
            m.cv[(cv['class'], cv['metric'])] = {k: v for k, v in cv.items() if k in ('scene', 'rig')}
        m.train_rigs = {h: features_from_json(f) for h, f in (d.get('train_rigs') or {}).items()}
        return m


def default_model_path() -> str:
    return os.path.join(data_dir(), 'model.json')


def save_model(model: Model, path=None) -> str:
    path = path or default_model_path()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + '.part'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(model.to_dict(), fh, separators=(',', ':'))
    os.replace(tmp, path)
    return path


def load_model(path=None) -> Model | None:
    """The saved model (default: the user's data folder's model.json), or None when there is none. Check .trained:
    a model saved before it had enough records says why in .reason()."""
    path = path or default_model_path()
    try:
        with open(path, encoding='utf-8') as fh:
            d = json.load(fh)
    except (OSError, ValueError):
        return None
    if not isinstance(d, dict) or d.get('schema') != SCHEMA:
        return None
    try:
        return Model.from_dict(d)
    except (KeyError, ValueError, TypeError, np.linalg.LinAlgError):
        return None


def train(docs=None, store=None, settings=None, project=None, engine=None, progress=None, cancel=None,
          save=False, path=None) -> Model:
    """Fit a Model on records (default: the store's); save=True writes it where load_model() finds it."""
    docs = records(None, store) if docs is None else docs
    st = settings if isinstance(settings, PerfModelSettings) else PerfModelSettings.from_dict(settings or {})
    m = Model(st).fit(docs, project=project, engine=engine, progress=progress, cancel=cancel)
    if save:
        save_model(m, path)
    return m


# =========================================================================== the interim evaluator
def _rec_cam_block(rc: dict, cam_meta: dict | None = None) -> dict:
    """A recording camera (sim_rig.json's camera entry) as a record camera: the recorded image's size, the pose, the
    lens's field; module and lens from cam_meta[tag] = (module, lens) when known."""
    from .lens import Intrinsics
    pose = rc.get('pose_vehicle') or {}
    ln = rc.get('lens') or {}
    w, h = (ln.get('image_size') or [0, 0])[:2]
    sw = (ln.get('sensor_size') or [w, h])[0] or w
    s = float(w) / float(sw) if sw else 1.0
    proj = ln.get('projection') or 'kannala_brandt'
    kb = tuple((list(ln.get('kb') or []) + [0.0] * 4)[:4])
    pm = 1e-3
    intr = Intrinsics(width=int(w), height=int(h), pixel_mm=pm, f_mm=float(ln.get('f_px') or 1.0) * s * pm,
                      projection=proj if proj in PROJ_ALIASES or proj == 'kannala_brandt' else 'equidistant',
                      kb=kb, cx=float(ln.get('cx') or sw / 2) * s, cy=float(ln.get('cy') or 0) * s,
                      max_theta=math.radians(float(ln.get('max_theta_deg') or 90.0)), focus_near=0.0,
                      focus_far=math.inf, aspect=float(ln.get('aspect') or 1.0))
    hf, vf = _hfov(intr)
    mod, lens = (cam_meta or {}).get(rc.get('tag'), ('', ''))
    return {'tag': str(rc.get('tag')), 'module': mod, 'lens': lens, 'width': int(w), 'height': int(h),
            'x': float(pose.get('x', 0.0)), 'y': float(pose.get('y', 0.0)), 'z': float(pose.get('z', 0.0)),
            'yaw': float(pose.get('yaw', 0.0)), 'pitch': float(pose.get('pitch', 0.0)),
            'roll': float(pose.get('roll', 0.0)), 'hfov_deg': round(hf, 2), 'vfov_deg': round(vf, 2),
            'max_theta_deg': round(float(ln.get('max_theta_deg') or 90.0), 2), 'projection': proj}


def recording_scene(rec) -> dict:
    """A recording's scene block (detection_eval.Recording)."""
    run = rec.run or {}
    sc = run.get('scene') or {}
    scen = run.get('scenario')
    params = scen.get('params') if isinstance(scen, dict) else {}
    out = {'kind': 'carla', 'town': rec.town, 'scenario': rec.scenario or 'free_drive', 'seed': rec.seed,
           'weather': rec.weather, 'duration_s': rec.duration_s, 'traffic': sc.get('traffic', run.get('traffic')),
           'walkers': sc.get('walkers'), 'scenario_params': params or {},
           'frames': int(sum(len(c.frames) for c in rec.cameras.values())),
           'vehicle_blueprint': (run.get('vehicle') or {}).get('blueprint')}
    out['scene_hash'] = scene_hash(out)
    return out


@dataclass
class RegionEvalResult:
    path: str
    cameras: list
    objects: dict                       # (cls, key, slot) -> {'xy', 'sr', 'det': {cam: score}, 'seen': set}
    fps: list                           # [(cls, cam, slot, score, sector, range bin)]
    tps: list                           # [(cls, cam, slot, score, sector, range bin)] true detections (precision)
    grid: RG.RegionGrid
    classes: list
    scene: dict
    detector: dict
    settings: dict
    notes: list
    seconds: float = 0.0


def region_eval(rec_path, settings=None, detector=None, grid=None, classes=CLASSES, write_cache=False,
                progress=None, cancel=None) -> RegionEvalResult | None:
    """THE INTERIM EVALUATOR (until core.rig_perception's records): per camera image of a CARLA recording with ground
    truth, the 2D detector matched to the ground truth as core.detection_eval does; kept per object and time slot for
    rig_metrics() to count any subset of the cameras as a rig. Detections cached in the recording are reused (and
    new ones saved there only with write_cache). None when cancelled."""
    from . import detection as DT
    from . import detection_eval as DE
    from . import groundtruth as GT
    t0 = time.perf_counter()
    st = settings if isinstance(settings, DE.DetectionSettings) else DE.DetectionSettings.from_dict(settings or {})
    rec = DE.open_recording(rec_path)
    grid = grid if isinstance(grid, RG.RegionGrid) else (RG.RegionGrid.from_dict(grid) if grid else RG.RegionGrid())
    ev_classes = [c for c in DT.EVAL_CLASSES if c in classes]
    if detector is None:
        det = DT.load(st.detector())
        det_fn, det_info = det.detect, det.describe()
    else:
        det_fn = detector
        det_info = dict(getattr(detector, 'describe', lambda: {})() or {'model': 'custom'})
    det_key = st.detector().key(det_info.get('sha256', det_info.get('model', '')))
    cams = [c for _, c in sorted(rec.cameras.items())]
    periods = []
    for c in cams:
        ts = np.array([int(f.get('timestamp') or 0) for f in c.frames], float)
        if len(ts) > 1:
            periods.append(float(np.median(np.diff(ts))))
    slot_us = max(float(np.median(periods)), 1.0) if periods else 1.0
    t_min = min((int(f.get('timestamp') or 0) for c in cams for f in c.frames), default=0)
    objects, fps, tps, notes = {}, [], [], []
    n_img = sum(len(c.frames) for c in cams) or 1
    done = 0
    for cam in cams:
        cache = DE._Cache(rec.path, cam.channel, 'fisheye', det_key, True)
        seen_slot = set()
        for fr in cam.frames:
            if cancel and cancel():
                return None
            done += 1
            t_us = int(fr.get('timestamp') or 0)
            slot = int(round((t_us - t_min) / slot_us))
            if slot in seen_slot:                        # one frame per camera and slot (the first)
                continue
            seen_slot.add(slot)
            rel = fr['image']
            path = GT.path_of(rec.gt, rel)
            dets = cache.get(rel, path)
            if dets is None:
                import cv2
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                if img is None:
                    notes.append(f'{cam.tag}: cannot read {os.path.basename(rel)}')
                    continue
                dets = det_fn(img)
                dets = {k: dets[k] for k in ('boxes', 'scores', 'coco', 'cls', 'ms') if k in dets}
                cache.put(rel, path, dets)
            try:
                rng = GT.load_range(rec.gt, fr) if fr.get('range') else None
            except OSError:
                rng = None
            g = DE.gt_objects(fr, st)
            ev = DE.evaluate_image(g, dets, rng, st, ev_classes)
            for e, i, s in ev['gt']:
                cname = DT.EVAL_CLASSES[e]
                k = (cname, g['key'][i], slot)
                o = objects.get(k)
                if o is None:
                    xy = g['xy'][i]
                    if not np.all(np.isfinite(xy)):
                        continue
                    sec, rb = grid.assign(xy[0], xy[1])
                    o = objects[k] = {'xy': (float(xy[0]), float(xy[1])), 'sr': (int(sec), int(rb)), 'det': {},
                                      'seen': set()}
                if g['ignore'][i] == DE.IGN_NONE:
                    o['seen'].add(cam.tag)
                if s > -np.inf:
                    o['det'][cam.tag] = max(o['det'].get(cam.tag, -np.inf), float(s))
            if ev['det']:
                box = np.asarray(dets['boxes'], float).reshape(-1, 4)
                rows = [r for r in ev['det'] if r[3] >= 0]
                if rows and cam.R is not None and cam.K is not None:
                    ii = np.array([r[1] for r in rows])
                    uc = 0.5 * (box[ii, 0] + box[ii, 2])
                    vc = 0.5 * (box[ii, 1] + box[ii, 3])
                    ray, ok = DE.kb_unproject(cam, uc, vc)
                    P = cam.t + (ray @ cam.R.T) * np.array([r[4] for r in rows])[:, None]
                    sec, rb = grid.assign(P[:, 0], P[:, 1])
                    for j, (e, i, s, status, rr, _) in enumerate(rows):
                        if not (ok[j] and math.isfinite(rr) and rb[j] >= 0):
                            continue
                        row = (DT.EVAL_CLASSES[e], cam.tag, slot, float(s), int(sec[j]), int(rb[j]))
                        (tps if status == 1 else fps).append(row)
            if progress and done % 10 == 0:
                progress(min(done / n_img, 0.999), f'{cam.tag} · frame {done} of {n_img}')
        if write_cache:
            cache.save()
    scene = recording_scene(rec)
    dinfo = {'id': str(det_info.get('model') or st.model), 'backend': 'lifted',
             'version': str(det_info.get('sha256', ''))[:16] or str(det_info.get('name', '')),
             'settings_hash': _sha({k: v for k, v in st.to_dict().items() if k not in ('recordings', 'cameras')})[:16],
             'evaluator': INTERIM, 'score': st.score, 'iou': st.iou}
    rig = SR_read(rec.path)
    return RegionEvalResult(rec.path, [{'tag': c.tag, 'block': rig.get(c.tag)} for c in cams], objects, fps, tps,
                            grid, ev_classes, scene, dinfo, {'score': st.score, 'keep_score': st.keep_score}, notes,
                            time.perf_counter() - t0)


def SR_read(path) -> dict:   # noqa: N802
    """tag -> the recording's sim_rig.json camera entry."""
    try:
        with open(os.path.join(path, 'rig', 'sim_rig.json'), encoding='utf-8') as fh:
            d = json.load(fh)
    except (OSError, ValueError):
        return {}
    return {c.get('tag'): c for c in d.get('cameras') or [] if isinstance(c, dict)}


def rig_metrics(res: RegionEvalResult, tags, score=None) -> dict:
    """metrics {class: {region id: cell}} of the rig made of the recording's cameras `tags`: every object some camera
    of the recording sees (not an ignore region) counts; the rig detects it when one of its cameras does at the
    operating point; false and true detections of its cameras per region; AP per region over the rig's objects (best
    score over its cameras) and false detections."""
    from . import detection_eval as DE
    tags = set(tags)
    op = res.settings['score'] if score is None else score
    keep = res.settings.get('keep_score', 0.05)
    cells = {}
    for (cls, key, slot), o in res.objects.items():
        if not o['seen'] or o['sr'][1] < 0:
            continue
        best = max((o['det'].get(t, -np.inf) for t in tags), default=-np.inf)
        c = cells.setdefault((cls, o['sr']), {'n': 0, 'tp': 0, 'scores': [], 'keys': set(), 'fp': [], 'td': 0})
        c['n'] += 1
        c['tp'] += best >= op
        c['keys'].add(key)
        if best >= keep:
            c['scores'].append(best)
    for cls, cam, slot, s, sec, rb in res.fps:
        if cam in tags:
            c = cells.setdefault((cls, (sec, rb)), {'n': 0, 'tp': 0, 'scores': [], 'keys': set(), 'fp': [], 'td': 0})
            c['fp'].append(s)
    for cls, cam, slot, s, sec, rb in res.tps:
        if cam in tags and s >= op:
            c = cells.setdefault((cls, (sec, rb)), {'n': 0, 'tp': 0, 'scores': [], 'keys': set(), 'fp': [], 'td': 0})
            c['td'] += 1
    out = {}
    for (cls, (sec, rb)), c in cells.items():
        fp_op = sum(1 for s in c['fp'] if s >= op)
        ap = None
        if c['n'] > 0:
            sc = np.r_[np.asarray(c['scores'], float), np.asarray(c['fp'], float)]
            tp = np.r_[np.ones(len(c['scores']), bool), np.zeros(len(c['fp']), bool)]
            ap = DE.ap_coco(sc, tp, c['n']) if len(sc) else 0.0
        if c['n'] == 0 and fp_op == 0 and c['td'] == 0:
            continue
        cell = metrics_cell(c['n'], c['tp'], fp_op, ap=ap, tp_det=c['td'],
                            extra={'n_tracks': len(c['keys']), 'tp_det': c['td']})
        out.setdefault(cls, {})[res.grid.region_id(sec, rb)] = cell
    return out


def library_records(res: RegionEvalResult, rigs: dict, project=None, cam_meta=None, source='carla',
                    provenance=None) -> list:
    """One record per rig {name: [camera tags]} of a region_eval result (rigs made of the recording's cameras)."""
    blocks = {c['tag']: _rec_cam_block(c['block'], cam_meta) for c in res.cameras if c.get('block')}
    veh = vehicle_block(project) if project is not None else {'name': '', 'length_m': None, 'width_m': None,
                                                              'height_m': None}
    out = []
    for name, tags in rigs.items():
        tags = [t for t in tags if t in blocks]
        if not tags:
            continue
        rig = {'vehicle': veh, 'cameras': [blocks[t] for t in tags], 'sensors': []}
        m = rig_metrics(res, tags)
        if not m:
            continue
        doc = make_record(source=source, scene=res.scene, detector=res.detector, rig=rig, grid=res.grid,
                          metrics=m, provenance=provenance, extra={'rig_label': str(name)[:80]})
        out.append(doc)
    return out


# =========================================================================== active learning
@dataclass
class Candidate:
    """A rig that could be evaluated for real: a name, its features, and how to evaluate it (a variant of the
    project, or a record's rig block)."""
    name: str
    features: dict | None = None
    variant: object = None
    rig: dict | None = None
    origin: str = ''                    # 'optimiser' | 'designer' | 'variant' | 'reference' | ...


def candidates_from_variants(project, variants=None, engine=None, grid=None, origin='variant') -> list:
    """Candidates for the project's variants (or the given ones), featurised on the project's grid."""
    grid = grid or RG.default_grid(project)
    out = []
    for v in (variants if variants is not None else project.variants):
        if not any(c.visible for c in v.cameras):
            continue
        out.append(Candidate(v.name, describe_rig(project, v, grid, engine), v, None, origin))
    return out


def _features(c):
    if isinstance(c, Candidate):
        return c.features
    if isinstance(c, dict) and 'X' in c:
        return c
    if isinstance(c, (tuple, list)) and len(c) == 2:
        return c[1]
    raise TypeError('a candidate is a Candidate, a features dict or a (name, features) pair')


def _name(c, i):
    if isinstance(c, Candidate):
        return c.name
    if isinstance(c, (tuple, list)) and len(c) == 2:
        return str(c[0])
    return f'candidate {i + 1}'


def expected_improvement(mean, std, best):
    from scipy.stats import norm
    std = np.maximum(np.asarray(std, float), 1e-12)
    z = (np.asarray(mean, float) - best) / std
    return (np.asarray(mean, float) - best) * norm.cdf(z) + std * norm.pdf(z)


def incumbent(model: Model, objective=None) -> float:
    """The best objective among the rigs the model was trained on (its posterior mean there)."""
    key = json.dumps(objective or model.st.objective, sort_keys=True)
    cache = model.__dict__.setdefault('_incumbent', {})
    if key not in cache:
        vals = [model.objective_mean(f, objective) for f in model.train_rigs.values()]
        vals = [v for v in vals if math.isfinite(v)]
        cache[key] = max(vals) if vals else -math.inf
    return cache[key]


def propose(candidates, k=None, objective=None, model=None, mode=None, kappa=None) -> list:
    """The k candidates to evaluate for real, in pick order: [{'index', 'name', 'mean', 'std', 'ei', 'score',
    'mode', 'why'}]. mode 'ei' (expected improvement over the best evaluated rig), 'std' (the objective's largest
    predicted std) or 'ucb' (mean + κ·std), all on the model's own (epistemic) uncertainty: a new scene set's random
    offsets are the same for every candidate and no evaluation removes them; a batch by the kriging believer (each pick's predicted recall joins the
    training data before the next pick). An untrained model proposes nothing (RuntimeError says why)."""
    model = model or load_model()
    if model is None or not model.trained:
        raise RuntimeError(model.reason() if model is not None else 'not trained (no model saved)')
    st = model.st
    k = int(k or st.k)
    mode = mode or st.mode
    kappa = st.kappa if kappa is None else float(kappa)
    ob = objective or st.objective
    feats = [_features(c) for c in candidates]
    best = incumbent(model, ob)
    models = dict(model.models)
    left = list(range(len(feats)))
    picks = []
    while left and len(picks) < k:
        stats = [model.objective(feats[i], ob, models=models, epistemic=True) for i in left]
        mean = np.array([s[0] for s in stats])
        std = np.array([s[1] for s in stats])
        ei = expected_improvement(mean, std, best if math.isfinite(best) else float(np.nanmax(mean)))
        score = {'ei': ei, 'std': std, 'ucb': mean + kappa * std}[mode]
        score = np.where(np.isfinite(score), score, -np.inf)
        j = int(np.argmax(score))
        i = left[j]
        why = {'ei': f'expected improvement {ei[j]:.4f} over the best evaluated rig ({best:.3f})',
               'std': f'the objective is least certain here (std {std[j]:.3f})',
               'ucb': f'mean + {kappa:g}·std = {score[j]:.3f}'}[mode]
        picks.append({'index': i, 'name': _name(candidates[i], i), 'mean': float(mean[j]), 'std': float(std[j]),
                      'ei': float(ei[j]), 'score': float(score[j]), 'mode': mode, 'why': why})
        left.pop(j)
        # the kriging believer: this pick's predicted latent mean becomes an observation
        f = feats[i]
        keep = model.region_weights(f['regions'], f.get('grid')) > 0      # the regions the objective counts
        models = {key: (cm.condition(f['X'][key[0]][keep], cm.predict(f['X'][key[0]][keep], offsets=False)[0])
                        if key[1] == 'recall' and key[0] in ob and key[0] in f['X'] else cm)
                  for key, cm in models.items()}
        if mode == 'ei':
            best = max(best, float(mean[j])) if math.isfinite(best) else float(mean[j])
    return picks


def suggest_simulations(project, k=4, model=None, variants=None, mode=None) -> dict:
    """The assistant's suggest_simulations(k): which of the project's variants to evaluate for real next."""
    model = model or load_model()
    if model is None or not model.trained:
        return {'ok': False, 'text': model.reason() if model is not None else 'not trained (no model saved)',
                'picks': []}
    cands = candidates_from_variants(project, variants)
    picks = propose(cands, k, model=model, mode=mode)
    return {'ok': True, 'picks': picks, 'text': '; '.join(f'{p["name"]}: {p["why"]}' for p in picks)}


def predict_detection(project, variant=None, model=None, engine=None) -> dict:
    """The assistant's predict_detection(variant): predicted recall per class and region with std, the rig-level
    summary, and the reason when there is no trained model."""
    model = model or load_model()
    if model is None:
        return {'trained': False, 'reason': 'not trained (no model saved)'}
    v = variant if variant is not None else project.variant
    f = describe_rig(project, v, RG.RegionGrid.from_dict(model.grid) if model.grid else RG.default_grid(project),
                     engine)
    p = model.predict(f)
    out = {'trained': p['trained'], 'reason': p['reason'], 'variant': v.name, 'rig': p['rig'], 'classes': {}}
    for c, d in p['classes'].items():
        out['classes'][c] = {rid: {'recall': round(float(m), 4), 'std': round(float(s), 4)}
                             for rid, m, s in zip(p['regions'], d['mean'], d['std'])}
    if p['trained']:
        m, s = model.objective(f)
        out['objective'] = {'mean': round(m, 4), 'std': round(s, 4), 'classes': model.st.objective,
                            'max_range_m': model.st.max_range_m}
    return out


def rig_perception_available() -> bool:
    try:
        from . import rig_perception as RP                     # noqa: F401 — bev's perception in the loop
    except ImportError:
        return False
    return hasattr(RP, 'run') and hasattr(RP, 'to_records')


def record_drive(project, variant, out_dir, sim_settings=None, progress=None, cancel=None) -> str:
    """Drive a variant in CARLA with ground truth into out_dir (core.detection_eval.evaluate_variant's bridge run; a
    drive already there of the same rig is reused) without detecting anything yet."""
    from . import detection as DT
    from . import detection_eval as DE

    def empty(img):
        return DT.empty()
    empty.describe = lambda: {'model': 'none'}
    dst = DE.DetectionSettings.from_dict({'frame_step': 1000, 'cache': False, 'bootstrap': 0})
    DE.evaluate_variant(project, variant, sim_settings=sim_settings, det_settings=dst, out_dir=out_dir,
                        progress=progress, cancel=cancel, detector=empty)
    return out_dir


def evaluate_recording(project, path, rig_name, variant=None, detector=None, progress=None, cancel=None,
                       store=None) -> list:
    """Records of one recording as one rig: core.rig_perception (perception in the loop: lifted 3D detection matched
    in bird's-eye view) when it is there, else this module's interim evaluator."""
    from . import provenance as PV
    if rig_perception_available():
        from . import rig_perception as RP
        rst = RP.RigPerceptionSettings.from_dict({'recordings': [{'path': path, 'rig': rig_name}]})
        res = RP.run(project, settings=rst, progress=progress, cancel=cancel, detector=detector)
        return list(RP.to_records(res, project, add=False)) if res is not None else []
    res = region_eval(path, grid=RG.default_grid(project), detector=detector, write_cache=True, progress=progress,
                      cancel=cancel)
    if res is None:
        return []
    meta = {c.tag: (c.module_id, c.lens_id) for c in variant.cameras} if variant is not None else {}
    prov = PV.block(project, None, variant=variant, study=RECORD_SCHEMA)
    return library_records(res, {rig_name: [c['tag'] for c in res.cameras]}, project, meta, provenance=prov)


def default_evaluator(project, sim_settings=None, work_dir=None, detector=None):
    """evaluator(candidate, progress, cancel) -> [records]: the candidate's variant driven in CARLA (record_drive),
    then evaluated by core.rig_perception when it is there, else by the interim evaluator."""
    def ev(cand, progress=None, cancel=None):
        if cand.variant is None:
            raise RuntimeError(f'{cand.name}: only a variant of the project can be driven in CARLA')
        from .. import workdir as WD
        out = os.path.join(work_dir or WD.project_dir(project, None, 'perf_model', 'drives'), WD.slug(cand.name))
        record_drive(project, cand.variant, out, sim_settings,
                     (lambda f, t='': progress(0.7 * f, t)) if progress else None, cancel)
        if cancel and cancel():
            return []
        return evaluate_recording(project, out, cand.name, cand.variant, detector,
                                  (lambda f, t='': progress(0.7 + 0.3 * f, t)) if progress else None, cancel)
    return ev


def campaign(project, candidates, rounds=2, k=2, evaluator=None, store=None, settings=None, objective=None,
             model=None, progress=None, cancel=None, save=True) -> dict:
    """Active learning for real: each round fit the model on the store, pick k candidates (propose), evaluate them
    (evaluator(candidate, progress, cancel) -> records; default_evaluator by default), add their records, and record
    the learning curve: before the new records join, the model's error on them (prospective RMSE of recall, 68 % /
    95 % coverage), the records and rigs so far. Candidates already evaluated are not proposed again."""
    st = settings if isinstance(settings, PerfModelSettings) else (
        PerfModelSettings.from_dict(settings) if settings else settings_of(project))
    store = _as_store(store) if store is not None else store_for(project)
    evaluator = evaluator or default_evaluator(project)
    cands = list(candidates)
    done_idx = set()
    curve, rounds_out = [], []
    stop = cancel or (lambda: False)
    for rnd in range(int(rounds)):
        if stop():
            break
        docs = store.records()
        m = model if (rnd == 0 and model is not None) else Model(st).fit(docs, project=project)
        point = {'round': rnd, 'records': len(docs), 'rigs': len({d.get('rig_hash') for d in docs}),
                 'trained': m.trained, 'reason': m.reason()}
        left = [i for i in range(len(cands)) if i not in done_idx]
        if not left:
            curve.append(point)
            break
        if m.trained:
            picks = propose([cands[i] for i in left], k, objective, m)
            chosen = [left[p['index']] for p in picks]
        else:                                           # no model yet: spread the first evaluations by geometry
            chosen = _spread([cands[i] for i in left], k)
            chosen = [left[j] for j in chosen]
            picks = [{'index': j, 'name': cands[j].name, 'why': 'no model yet: spread over the candidates'}
                     for j in chosen]
        new = []
        for n_i, i in enumerate(chosen):
            if stop():
                break
            sub = None
            if progress:
                def sub(f, t='', _n=n_i):
                    progress((rnd + (_n + f) / max(len(chosen), 1)) / rounds, f'round {rnd + 1}: {cands[i].name}: {t}')
            recs = evaluator(cands[i], sub, cancel) or []
            done_idx.add(i)
            new += recs
        if m.trained and new:
            point['prospective'] = prospective(m, new, project)
        r = store.add_many(new)
        point['added'] = r['added']
        curve.append(point)
        rounds_out.append({'round': rnd, 'picks': picks, 'records': [d.get('id') for d in new]})
    final = Model(st).fit(store.records(), project=project)
    if save and final.trained:
        save_model(final)
    curve.append({'round': len(rounds_out), 'records': final.n_records, 'rigs': final.n_rigs,
                  'trained': final.trained, 'reason': final.reason(), 'final': True})
    return {'schema': SCHEMA, 'rounds': rounds_out, 'curve': curve, 'model': final}


def _spread(cands, k):
    """k candidates far apart in feature space (greedy farthest point), for a start without a model."""
    if not cands:
        return []
    V = np.array([np.concatenate([np.asarray(_features(c)['X'][cl], float).ravel()
                                  for cl in sorted(_features(c)['X'])]) for c in cands])
    V = (V - V.mean(0)) / np.where(V.std(0) > 1e-9, V.std(0), 1.0)
    pick = [int(np.argmax(np.abs(V).sum(1)))]
    while len(pick) < min(k, len(cands)):
        d = np.min([((V - V[p]) ** 2).sum(1) for p in pick], axis=0)
        d[pick] = -1
        pick.append(int(np.argmax(d)))
    return pick


def prospective(model: Model, docs, project=None, metric='recall') -> dict:
    """The model's error on records it has not seen: RMSE of recall, 68 % / 95 % coverage of the observed empirical
    logit, rows."""
    cache = FeatureCache()
    eng = shared_engine() if project is not None and model.st.occlusion else None
    ys, mus, vs, po, pp = [], [], [], [], []
    for (c, m), cm in model.models.items():
        if m != metric or not cm.trained:
            continue
        rows = training_rows(docs, lambda d: cache.get(d, project, eng, model.st.occlusion), c, m)
        if rows is None:
            continue
        mu, var = cm.predict(rows['X'], scenes=rows['scene'], regions=rows['region'])
        ys.append(rows['y'])
        mus.append(mu)
        vs.append(var + cm.noise_var(rows['s2']))
        po.append(rows['k'] / rows['n'])
        pp.append(logistic_moments(mu, var)[0])
    if not ys:
        return {'n': 0}
    y, mu, v, po, pp = (np.concatenate(a) for a in (ys, mus, vs, po, pp))
    a = np.abs(y - mu) / np.sqrt(v)
    return {'n': int(len(y)), 'rmse': float(np.sqrt(np.mean((po - pp) ** 2))),
            'coverage68': float(np.mean(a <= Z68)), 'coverage95': float(np.mean(a <= Z95))}


# =========================================================================== the study (module contract)
@dataclass
class PerfModelResult:
    settings: PerfModelSettings
    model: Model
    store_path: str
    store: dict                         # store_summary
    variant: str = ''
    features: dict | None = None        # the variant's
    prediction: dict | None = None
    reference: str = ''
    ref_features: dict | None = None
    attributions: list = field(default_factory=list)
    objective: tuple = (math.nan, math.nan)
    proposals: list = field(default_factory=list)
    seconds: float = 0.0
    notes: list = field(default_factory=list)


def reference_variant(project, st: PerfModelSettings, variant=None):
    vs = [v for v in project.variants if any(c.visible for c in v.cameras)]
    if not vs:
        return None
    ref = next((v for v in vs if st.reference and (v.uid == st.reference or v.name == st.reference)), None)
    if ref is None:
        ref = vs[0]
    return None if variant is not None and ref is variant else ref


def run(project, engine=None, settings=None, variant=None, progress=None, cancel=None, store=None,
        model=None, rank=True) -> PerfModelResult | None:
    """Train the model on the store's records (or use model), cross-validate it, predict the variant (default: the
    active one), explain it against the reference variant and rank the project's variants for a real evaluation.
    None when cancelled."""
    t0 = time.perf_counter()
    st = settings if isinstance(settings, PerfModelSettings) else (
        PerfModelSettings.from_dict(settings) if isinstance(settings, dict) else settings_of(project))
    S = _as_store(store) if store is not None else store_for(project)
    docs = S.records()

    def prog(f, t=''):
        if progress is not None:
            progress(min(max(float(f), 0.0), 1.0), t)
    prog(0.02, 'Records')
    if model is None:
        model = Model(st).fit(docs, project=project, engine=engine,
                              progress=lambda f, t='': prog(0.05 + 0.8 * f, f'Training · {t}'), cancel=cancel)
        if cancel and cancel():
            return None
    v = variant if variant is not None else project.variant
    grid = RG.RegionGrid.from_dict(model.grid) if model.grid else RG.default_grid(project)
    res = PerfModelResult(st, model, S.path, store_summary(docs), v.name if v is not None else '')
    if v is not None and any(c.visible for c in v.cameras):
        prog(0.88, 'The variant')
        res.features = describe_rig(project, v, grid, engine)
        res.prediction = model.predict(res.features)
        if model.trained:
            ex = model.extrapolation(res.features)
            far = {c: v for c, v in ex.items() if v > 0.25}
            if far:
                res.notes.append('Beyond the training data: ' + ', '.join(
                    f'{c} in {v * 100:.0f} % of the regions' for c, v in far.items()) + ' have more or fewer pixels '
                    'on the object than any rig the model was trained on (higher-resolution images than the '
                    'recordings\'); there the prediction follows the monotone prior in pixels.')
            res.objective = model.objective(res.features)
            ref = reference_variant(project, st, v)
            if ref is not None:
                res.reference = ref.name
                res.ref_features = describe_rig(project, ref, grid, engine)
                res.attributions = model.explain(res.features, res.ref_features)
            try:
                if not rank:
                    raise ValueError('')
                prog(0.94, 'Ranking the variants')
                res.proposals = propose(candidates_from_variants(project, None, engine, grid), st.k, model=model)
            except (RuntimeError, ValueError) as e:
                if str(e):
                    res.notes.append(str(e))
    else:
        res.notes.append('The variant has no visible camera: nothing to predict.')
    res.seconds = time.perf_counter() - t0
    prog(1.0, 'Done')
    return res


def _r(v, nd=4):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return round(x, nd) if math.isfinite(x) else None


def summary(res: PerfModelResult) -> dict:
    m = res.model
    out = {'schema': SCHEMA, 'trained': m.trained, 'reason': m.reason(), 'records': m.n_records,
           'detector': m.detector,
           'scenes': m.n_scenes, 'rigs': m.n_rigs, 'status': m.status, 'variant': res.variant,
           'objective': {'mean': _r(res.objective[0]), 'std': _r(res.objective[1]), 'classes': res.settings.objective,
                         'max_range_m': res.settings.max_range_m},
           'rig': {c: {'mean': _r(d['mean']), 'std': _r(d['std'])} for c, d in
                   ((res.prediction or {}).get('rig') or {}).items()},
           'calibration': cv_table(m), 'proposals': [{k: p[k] for k in ('name', 'mean', 'std', 'ei', 'why')}
                                                     for p in res.proposals],
           'attributions': [a['text'] for a in res.attributions], 'reference': res.reference,
           'seconds': round(res.seconds, 2)}
    return out


def cv_table(model: Model) -> list:
    """Rows: class, metric, CV kind, method, n, coverage68, coverage95, rmse, nlpd, folds."""
    rows = []
    for (c, met), cvs in model.cv.items():
        for by, cv in cvs.items():
            if not cv.get('ok'):
                rows.append({'class': c, 'metric': met, 'cv': by, 'method': '', 'n': 0, 'note': cv.get('why', '')})
                continue
            for meth in ('gp', 'logistic', 'constant'):
                s = cv[meth]
                rows.append({'class': c, 'metric': met, 'cv': by, 'method': meth, 'n': s['n'],
                             'coverage68': _r(s['coverage68'], 3), 'coverage95': _r(s['coverage95'], 3),
                             'rmse': _r(s['rmse'], 4), 'rmse_weighted': _r(s['rmse_weighted'], 4),
                             'nlpd': _r(s['nlpd'], 3), 'folds': cv['folds']})
    return rows


def to_doc(res: PerfModelResult, project=None) -> dict:
    m = res.model
    pred = res.prediction or {}
    grid = RG.RegionGrid.from_dict(m.grid) if m.grid else (RG.default_grid(project) if project else RG.RegionGrid())
    regions = []
    for i, rid in enumerate(pred.get('regions') or []):
        sr = grid.parse_id(rid)
        row = {'region': rid, 'name': grid.region_name(*sr) if sr else rid}
        for c, d in (pred.get('classes') or {}).items():
            row[c] = {'recall': _r(d['mean'][i]), 'std': _r(d['std'][i]), 'interval68': [_r(d['lo68'][i]),
                                                                                         _r(d['hi68'][i])],
                      'interval95': [_r(d['lo95'][i]), _r(d['hi95'][i])]}
        regions.append(row)
    doc = {'schema': SCHEMA, 'project': getattr(project, 'name', None), 'variant': res.variant,
           'frame': FRAME_V, 'units': {'range': 'm', 'recall': '0..1'}, 'grid': grid.to_dict(),
           'summary': summary(res), 'settings': res.settings.to_dict(), 'store': res.store,
           'model': {'status': m.status, 'fitted_utc': m.fitted_utc, 'features': list(FEATURE_NAMES),
                     'hyperparameters': {f'{c}:{met}': {'prior': {'a': _r(cm.a), 'b': _r(cm.b)},
                                                        'length_scales': dict(zip(FEATURE_NAMES,
                                                                                  (_r(x, 3) for x in cm.ell))),
                                                        'sigma_f': _r(math.sqrt(cm.sf2)),
                                                        'sigma_n': _r(math.sqrt(cm.sn2)), 'dispersion': _r(cm.phi),
                                                        'sigma_scene': _r(math.sqrt(cm.ss2)),
                                                        'sigma_scene_region': _r(math.sqrt(cm.sq2)),
                                                        'rows': cm.n_rows, 'posterior': cm.kind}
                                         for (c, met), cm in m.models.items() if cm.trained}},
           'calibration': cv_table(m),
           'calibration_curves': [{'class': c, 'metric': met, 'cv': by,
                                   **{meth: cv[meth]['curve'] for meth in ('gp', 'logistic', 'constant')}}
                                  for (c, met), cvs in m.cv.items() for by, cv in cvs.items() if cv.get('ok')],
           'regions': regions, 'attributions': res.attributions, 'reference': res.reference,
           'proposals': res.proposals, 'assumptions': ASSUMPTIONS, 'limits': LIMITS_TEXT, 'notes': res.notes}
    return doc


def to_csv(res: PerfModelResult) -> str:
    import csv
    import io
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(['class', 'metric', 'cv', 'method', 'n', 'coverage68', 'coverage95', 'rmse', 'rmse_weighted', 'nlpd',
                'folds', 'note'])
    for r in cv_table(res.model):
        w.writerow([r.get(k, '') for k in ('class', 'metric', 'cv', 'method', 'n', 'coverage68', 'coverage95', 'rmse',
                                           'rmse_weighted', 'nlpd', 'folds', 'note')])
    return buf.getvalue()


def predictions_csv(res: PerfModelResult) -> str:
    import csv
    import io
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(['class', 'region', 'recall', 'std', 'lo68', 'hi68', 'lo95', 'hi95'])
    pred = res.prediction or {}
    for c, d in (pred.get('classes') or {}).items():
        for i, rid in enumerate(pred['regions']):
            w.writerow([c, rid] + [f'{float(d[k][i]):.4f}' for k in ('mean', 'std', 'lo68', 'hi68', 'lo95', 'hi95')])
    return buf.getvalue()


def files(res: PerfModelResult, project=None) -> list:
    out = [('perf_model.json', json.dumps(to_doc(res, project), indent=1, default=float).encode('utf-8')),
           ('perf_model_calibration.csv', to_csv(res).encode('utf-8')),
           ('perf_model_predictions.csv', predictions_csv(res).encode('utf-8'))]
    try:
        out.append(('perf_model_calibration.png', calibration_png(res.model)))
    except Exception:                                   # noqa: BLE001 — the picture is optional (OpenCV)
        pass
    return out


def export(res: PerfModelResult, project, out_dir) -> list:
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for name, data in files(res, project):
        p = os.path.join(out_dir, name)
        with open(p, 'wb') as fh:
            fh.write(data)
        paths.append(p)
    return paths


def calibration_png(model: Model, size=(900, 420)) -> bytes:
    """Reliability plot of the leave-one-scene-out (left) and leave-rigs-out (right) intervals: empirical against
    nominal coverage, the GP and both baselines, for every trained class pooled by row."""
    import cv2
    W, H = size
    img = np.full((H, W, 3), 255, np.uint8)
    cols = {'gp': (214, 120, 42), 'logistic': (60, 160, 60), 'constant': (150, 150, 150)}
    for p, by in enumerate(('scene', 'rig')):
        x0, y0, w, h = 50 + p * (W // 2), 40, W // 2 - 80, H - 90
        cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (200, 200, 200), 1)
        cv2.line(img, (x0, y0 + h), (x0 + w, y0), (210, 210, 210), 1)
        title = 'leave one scene set out' if by == 'scene' else 'leave rigs out'
        cv2.putText(img, title, (x0, y0 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1, cv2.LINE_AA)
        for t in (0.0, 0.5, 1.0):
            cv2.putText(img, f'{t:g}', (x0 - 28, int(y0 + h - t * h) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (90, 90, 90), 1, cv2.LINE_AA)
            cv2.putText(img, f'{t:g}', (int(x0 + t * w) - 8, y0 + h + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (90, 90, 90), 1, cv2.LINE_AA)
        for meth, col in cols.items():
            acc = {}
            for (c, met), cvs in model.cv.items():
                cv = cvs.get(by) or {}
                if met != 'recall' or not cv.get('ok'):
                    continue
                for lv, e in cv[meth]['curve']:
                    a = acc.setdefault(lv, [0.0, 0])
                    a[0] += e * cv[meth]['n']
                    a[1] += cv[meth]['n']
            pts = [(int(x0 + lv * w), int(y0 + h - (s / n) * h)) for lv, (s, n) in sorted(acc.items()) if n]
            for a, b in zip(pts, pts[1:]):
                cv2.line(img, a, b, col, 2, cv2.LINE_AA)
            for q in pts:
                cv2.circle(img, q, 3, col, -1, cv2.LINE_AA)
    for i, (meth, col) in enumerate(cols.items()):
        x = 60 + i * 170
        cv2.line(img, (x, H - 22), (x + 24, H - 22), col, 2)
        label = {'gp': 'Gaussian process', 'logistic': 'logistic in pixels', 'constant': 'constant'}[meth]
        cv2.putText(img, label, (x + 30, H - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.putText(img, 'nominal coverage ->   empirical coverage ^', (W - 330, H - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (90, 90, 90), 1, cv2.LINE_AA)
    ok, buf = cv2.imencode('.png', img)
    return buf.tobytes()
