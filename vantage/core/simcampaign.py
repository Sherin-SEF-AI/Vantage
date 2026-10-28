"""Simulation campaigns: many CARLA drives of a rig, each varied in one known way (a fault of known size on a known
sensor, a weather, a rig variant, a scenario, a seed), every drive checked by Vantage's own checks, and the answers
compared with what was injected: how small a fault each check finds, how often it cries wolf, how well it measures
what it finds, and how detection and the sensors' data change with the weather.

The campaign
------------
A base (the project, its CARLA settings `analysis.sim`, the campaign's overrides) and a design:
* fault sets: a kind (rotation, translation, clock_offset, drop, freeze, range_bias: the bridge's, see
  vantage.sim.carla_bridge.FAULT_KINDS) × sensors × axes × magnitudes. Units here: rotation degrees, translation and
  range bias centimetres, clock offset milliseconds, drop per cent of the stream's frames (that exact number of frames,
  chosen by the seed, is never written), freeze seconds (a window in the middle of the drive);
* weathers (CARLA presets, or Vantage conditions 'vantage:<id>' mapped by core.simrig), scenarios, rig variants;
* seeds: each a repeat of the whole design (the traffic, the spawn point and the drop choices follow it);
* clean drives: per seed and setting, a drive with no fault (the false-alarm rate, the baselines).
With `multiplex` on (the default) the faults of different sensors ride in the same drive: each check judges each
sensor on its own (the drive-log check estimates every camera against the lidar, its sync and stream checks are per
stream), so a drive carries one condition per sensor and a design needs as many drives per seed as its longest sensor
list. Each sensor's conditions are shuffled per seed by the seed (so two sensors never share a schedule) and, with
signs 'alternate', the sign of a rotation or translation flips from one seed to the next. Effects of one sensor's fault
on another's check (cross-talk: a trigger line's reference camera late makes the others look early) are measured: the
unfaulted sensors of faulty drives are reported apart from those of clean drives.

Every drive is a run of the bridge (vantage.sim.carla_bridge) in CARLA's Python on a server already running (the page's,
or one the campaign starts and stops through vantage.sim.server.Server: capped, on the NVIDIA GPU), with the faults,
seed, scenario and ground truth on its command line over a rig file built once per variant and weather. A drive is kept
under <out>/drives/<id>/ with its key (the rig file's hash and its command line): a campaign run again (or resumed after
a cancel) reuses every drive whose key matches and whose run ended well, and every evaluation whose settings match.
Only the ground truth an evaluator needs is asked for: detection needs the semantic, instance and range images; rig
health needs CARLA's own car as each camera sees it, from one short parked reference drive per variant with the
instance images (the ego's pixels: ground-truth class 'ego'), at the planned mounts.

Evaluators (each drive)
-----------------------
* drivelog: core.drivelog.run on the drive's nuScenes folder with the variant's plan: each camera's rotation against the
  lidar (verdict, estimate, standard error), the sync check (each camera's offset from the sync plan), the stream
  checks (rates, drops, frozen images: identical content in a row), every finding with the sensors it names.
* rig_health: core.rig_health's statistics run offline over the recorded frames at their stamps (rate, drops, frozen
  frames, brightness, sharpness), each camera's capture-time skew against the rig's own capture schedule
  (rig_health.capture_offsets with schedule_offsets: the stagger a lidar-synchronised plan gives its cameras is taken
  out, so only what the plan does not account for counts), and the body-edge drift indicator against CARLA's own car
  (the reference drive's ego pixels, the live rig's method) on sampled frames.
* asbuilt: core.asbuilt.run. What it can and cannot judge on a CARLA drive: its outline comparison sets the camera's
  view of CARLA's car (an Audi A2 for the sample) against the project's body: that measures the two bodies' difference,
  not the mount, so its verdict is no test of the mount; the image circle is judged (the bridge makes it through
  Vantage's own lens model); its lidar estimate is the drive-log check's.
* detection: core.detection_eval.run on the drive (YOLOX against the ground truth): recall per class and range bin,
  the detection range.
* image_stats: per camera inside the lens circle, on sampled frames: brightness (mean grey), RMS contrast (standard
  deviation / mean), noise (Immerkær's fast estimate, 1996, away from edges), clipped shares, sharpness.
* returns: lidar points per sweep and their intensity, radar detections per scan.

Analyses
--------
* Fault detectability, per check, sensor, fault kind and axis: the detection probability at each magnitude with its
  Wilson score interval (Wilson 1927; Brown, Cai & DasGupta 2001), the false-alarm rate on clean drives, the smallest
  magnitude detected with probability ≥ target (default 0.9): empirically (the smallest tested magnitude from which every
  larger one reaches the target) and from a logistic fit in log10 magnitude with the false-alarm rate as its floor
  (maximum likelihood; a stratified bootstrap gives its interval); estimation accuracy (estimate − injected: bias, RMS),
  absolute and against the clean drive of the same seed; clock offsets against the sync check, drops against the stream
  findings, freezes against rig health and the drive-log check's frozen-image finding (drivelog.streams). A check
  that cannot judge on a CARLA drive what its verdict is about (NOT_APPLICABLE: the as-built check's mount verdict,
  CARLA's car being another body) keeps its answers, marked with the reason, apart from the false alarms; it gives no
  threshold.
* Weather sweep: per weather, detection recall per class and range bin (counts pooled over seeds), the range at which
  recall falls to 50 % (interpolated between range bins), its ratio to the clear drive's, against Vantage's own weather
  model (core.perception with core.weather: each class's far range under the condition over its far range in clear,
  median over every direction); image statistics; lidar and radar returns (CARLA's lidar and radar do not see the
  weather: their counts are expected to stay).
* Rig comparison: core.detection_eval's comparison of the variants' drives (paired when they share scenario and seed).

Frames: the vehicle frame (ISO 8855, rear axle); rotations intrinsic Z-Y-X yaw, pitch, roll in degrees, pitch positive
down, as the bridge applies them in the sensor's own frame; estimates are compared in the same Euler-angle increments the
drive-log check reports (the bridge's pose_delta_deg).
"""
from __future__ import annotations

import base64
import csv
import hashlib
import html
import io
import json
import math
import os
import shutil
import signal
import subprocess
import time
from dataclasses import asdict, dataclass, field
from statistics import NormalDist

import numpy as np

SCHEMA = 'vantage.sim_campaign/1'
DRIVE_SCHEMA = 'vantage.sim_campaign_drive/1'
EVAL_SCHEMA = 'vantage.sim_campaign_eval/1'
FRAME = ('vehicle (ISO 8855, rear axle): x forward, y left, z up, metres; rotations intrinsic Z-Y-X yaw, pitch, roll in '
         'degrees, pitch positive down')

FAULT_KINDS = ('rotation', 'translation', 'clock_offset', 'drop', 'freeze', 'range_bias')
FAULT_LABEL = {'rotation': 'Rotation', 'translation': 'Translation', 'clock_offset': 'Clock offset', 'drop': 'Drop',
               'freeze': 'Freeze', 'range_bias': 'Range bias'}
FAULT_UNIT = {'rotation': '°', 'translation': 'cm', 'clock_offset': 'ms', 'drop': '%', 'freeze': 's',
              'range_bias': 'cm'}
FAULT_SENSORS = {'rotation': ('camera', 'lidar', 'radar', 'imu', 'gnss'),
                 'translation': ('camera', 'lidar', 'radar', 'imu', 'gnss'),
                 'clock_offset': ('camera', 'lidar', 'radar', 'imu', 'gnss'),
                 'drop': ('camera', 'lidar', 'radar', 'imu', 'gnss'), 'freeze': ('camera', 'lidar', 'radar', 'imu', 'gnss'),
                 'range_bias': ('lidar', 'radar')}
AXES = {'rotation': ('yaw', 'pitch', 'roll'), 'translation': ('x', 'y', 'z')}
MAG_LIMITS = {'rotation': (0.001, 30.0), 'translation': (0.01, 200.0), 'clock_offset': (0.01, 1000.0),
              'drop': (0.1, 90.0), 'freeze': (0.1, 60.0), 'range_bias': (0.1, 500.0)}
SIGNS = ('alternate', 'positive', 'negative', 'random')

EVALUATORS = ('drivelog', 'rig_health', 'asbuilt', 'detection', 'image_stats', 'returns')
EVALUATOR_LABEL = {'drivelog': 'Drive-log check', 'rig_health': 'Rig health (offline)', 'asbuilt': 'As-built check',
                   'detection': 'Detection quality', 'image_stats': 'Image statistics', 'returns': 'Lidar and radar returns'}
ASBUILT_ON = ('none', 'clean', 'all')
SERVERS = ('running', 'own')

# check id: (label, sensor kinds it judges, unit of its statistic)
CHECKS = {
    'drivelog.extrinsics': ('Drive-log check · camera rotation', ('camera',), '°'),
    'drivelog.sync': ('Drive-log check · sync', ('camera',), 'ms'),
    'drivelog.streams': ('Drive-log check · streams', ('camera', 'lidar', 'radar', 'imu', 'gnss'), '%'),
    'drivelog.any': ('Drive-log check · any finding', ('camera', 'lidar', 'radar', 'imu', 'gnss'), ''),
    'rig_health.drift': ('Rig health · body-edge drift', ('camera',), 'px'),
    'rig_health.freeze': ('Rig health · frozen stream', ('camera',), 'frames'),
    'rig_health.drops': ('Rig health · dropped frames', ('camera',), '%'),
    'rig_health.sync': ('Rig health · capture-time skew', ('camera',), 'ms'),
    'rig_health.verdict': ('Rig health · verdict', ('camera',), ''),
    'asbuilt.verdict': ('As-built check · verdict', ('camera',), ''),
}
# checks that cannot judge, on a CARLA drive (every drive of a campaign is one), what their verdict is about: their
# answers are kept and shown, marked with the reason, but never counted as false alarms, detections or thresholds
NOT_APPLICABLE = {
    'asbuilt.verdict': 'Not applicable on CARLA: the as-built check judges a camera\'s mount by its view of the car '
                       'body against the project\'s body, and in a CARLA drive the body in the images is CARLA\'s car '
                       '(the CARLA vehicle nearest in size), not the project\'s. Its verdict measures the two bodies\' '
                       'difference, not the mount, so a flag on a clean drive is no false alarm and a flag on a turned '
                       'camera no detection. (It still judges the image circle; its lidar estimate is the drive-log '
                       'check\'s.)',
}


def applicable(check: str) -> bool:
    """Whether a check can judge what it reports on a CARLA drive (NOT_APPLICABLE says why not)."""
    return check not in NOT_APPLICABLE


RELEVANT = {'rotation': ('drivelog.extrinsics', 'rig_health.drift', 'asbuilt.verdict'),
            'translation': ('drivelog.extrinsics', 'rig_health.drift', 'asbuilt.verdict'),
            'clock_offset': ('drivelog.sync', 'rig_health.sync'),
            'drop': ('drivelog.streams', 'rig_health.drops'),
            'freeze': ('rig_health.freeze', 'drivelog.streams'),
            'range_bias': ('drivelog.streams', 'drivelog.any')}
NOT_FAULT_SECTIONS = ('data', 'mapping')    # drive-log findings about the plan's budget or topic names, not the sensor

REFERENCE_S = 1.0               # the parked reference drive (CARLA's car per camera)
DRIFT_FRAMES = 12               # frames per camera the drift indicator is measured on
STAT_FRAMES = 16                # frames per camera for the image statistics
BOOT = 400                      # bootstrap resamples of a threshold fit
# (wall seconds of a bridge run per simulated second, its start, warm-up and tidy-up included; seconds per drive
# beyond the bridge's own run), measured on the development machine (RTX 5080, the sample's data rig: 6 fisheye
# cameras at half size, OS1-128, 5 radars): no ground truth at 320 Hz steps 7.4 (at 80 Hz 4.7); the 1 s parked
# reference drive with instance images 16; full ground truth at 80 Hz 8–17 (12 s drives); used until this campaign has
# measured its own
COST_DEFAULT = {'none': (7.4, 1.0), 'instance': (16.0, 1.0), 'full': (9.5, 1.0)}
EVAL_COST_S = {'drivelog': 13.0, 'rig_health': 8.0, 'asbuilt': 60.0, 'detection': 0.2, 'image_stats': 3.0,
               'returns': 2.0}        # detection: per image
ASSUMPTIONS = (
    'Every drive is CARLA 0.9.16 rendering its own town, traffic, weather and its own car (the nearest CARLA vehicle '
    'by size, not the project\'s body); the faults change what CARLA records (the actual mount, the stamps, the '
    'delivery) while every calibration output states the plan, as on a real faulty rig.',
    'Detection probabilities are shares of drives, each a Bernoulli trial: drives of one seed share the scene, so '
    'conditions are compared within a seed\'s set; Wilson score intervals assume independent trials.',
    'The smallest detectable fault from the logistic fit assumes the detection probability rises monotonically and '
    'smoothly with the log of the magnitude; the empirical figure assumes nothing but rests on the tested magnitudes.',
    'Rig health runs offline on the recorded frames at their stamps (simulated time): rates, drops and freezes are as '
    'the bridge wrote them; network or encoder delays of a live stream do not exist here.',
    'The drift indicator compares a frame with CARLA\'s own car at the planned mount (the reference drive): with the '
    'project\'s body it would measure the difference between the two bodies.',
    'The weather comparison sets CARLA\'s rendered weather (core.simrig maps a Vantage condition\'s fog to CARLA\'s '
    'by a measured calibration, a visibility of about 7000 m / fog_density for road users, and its light and rain by '
    'documented heuristics) against Vantage\'s physical model of the same named condition: the comparison tests the '
    'two models\' agreement, neither is the truth. Vantage\'s 5 % contrast threshold is a human observer\'s, so its '
    'far ranges in fog are an upper bound for a detector; CARLA\'s rain is water on the lens (Vantage\'s lens-drop '
    'share), not a loss with range.',
    'CARLA\'s lidar and radar ignore the weather (fog and rain are rendered effects for the cameras only); its images '
    'carry no sensor noise (the noise estimate measures texture and JPEG).',
)
REFERENCES = (
    'Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference. JASA 22, 209–212.',
    'Brown, L. D., Cai, T. T., DasGupta, A. (2001). Interval estimation for a binomial proportion. Statistical Science '
    '16(2), 101–133.',
    'Immerkær, J. (1996). Fast noise variance estimation. Computer Vision and Image Understanding 64(2), 300–302.',
    'Efron, B., Tibshirani, R. (1993). An Introduction to the Bootstrap. Chapman & Hall (percentile intervals).',
)

LIMITS = {'target': (0.5, 0.999), 'confidence': (0.5, 0.999), 'max_drives': (1, 2000), 'workers': (1, 1)}
INTS = ('max_drives',)


# =========================================================================== settings
def _num(v, lo, hi):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x) or isinstance(v, bool):
        return None
    return min(max(x, lo), hi)


def clean_fault_set(d) -> dict | None:
    """A fault set kept to what is valid: {'kind', 'tags', 'axes', 'magnitudes', 'signs'} (magnitudes positive, in
    the kind's unit, sorted and unique); None when nothing is left."""
    if not isinstance(d, dict) or d.get('kind') not in FAULT_KINDS:
        return None
    kind = d['kind']
    tags = [str(t).strip()[:32] for t in (d.get('tags') or []) if isinstance(t, (str, int)) and str(t).strip()]
    tags = list(dict.fromkeys(tags))[:32]
    lo, hi = MAG_LIMITS[kind]
    mags = sorted({round(x, 6) for x in (_num(m, lo, hi) for m in (d.get('magnitudes') or []) if m is not None)
                   if x is not None})[:32]
    if not tags or not mags:
        return None
    out = {'kind': kind, 'tags': tags, 'magnitudes': mags}
    if kind in AXES:
        ax = [a for a in AXES[kind] if a in (d.get('axes') or AXES[kind])]
        out['axes'] = ax or list(AXES[kind])
        out['signs'] = d.get('signs') if d.get('signs') in SIGNS else 'alternate'
    return out


@dataclass
class CampaignSettings:
    name: str = 'Fault detectability'
    template: str = 'fault_detectability'
    variants: list = field(default_factory=list)       # variant uids; [] = the active variant
    seeds: list = field(default_factory=lambda: [1, 2, 3])
    weathers: list = field(default_factory=list)       # CARLA presets or 'vantage:<id>'; [] = the base's weather
    scenarios: list = field(default_factory=list)      # [{'name', 'params'}]; [] = the base's scenario
    fault_sets: list = field(default_factory=list)     # [clean_fault_set]
    clean: bool = True                                 # a drive with no fault per seed and setting
    multiplex: bool = True                             # one condition per sensor per drive, sensors in parallel
    base: dict = field(default_factory=dict)           # core.simrig.SimSettings over the project's analysis.sim
    evaluators: list = field(default_factory=lambda: ['drivelog', 'rig_health', 'image_stats', 'returns'])
    asbuilt_on: str = 'clean'                          # the as-built check runs on: none | clean | all drives
    drivelog: dict = field(default_factory=dict)       # core.drivelog.DriveLogSettings over its defaults
    health: dict = field(default_factory=dict)         # core.rig_health.HealthSettings over its defaults
    detection: dict = field(default_factory=dict)      # core.detection_eval.DetectionSettings over the project's
    target: float = 0.9                                # detection probability the threshold is quoted at
    confidence: float = 0.95
    server: str = 'running'                            # running (one already on base port) | own (started, stopped)
    out_dir: str = ''                                  # '' = <working folder>/<project>/campaigns/<name>
    max_drives: int = 400

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d) -> 'CampaignSettings':
        """Tolerant of junk: unknown keys ignored, numbers clamped, lists cleaned."""
        s = CampaignSettings()
        d = d if isinstance(d, dict) else {}
        if isinstance(d.get('name'), str) and d['name'].strip():
            s.name = d['name'].strip()[:80]
        if d.get('template') in tuple(TEMPLATES) + ('custom',):
            s.template = d['template']
        if isinstance(d.get('variants'), list):
            s.variants = [str(u)[:64] for u in d['variants'] if isinstance(u, str) and u][:16]
        if isinstance(d.get('seeds'), list):
            seeds = []
            for x in d['seeds']:
                v = _num(x, 0, 999999)
                if v is not None and int(v) not in seeds:
                    seeds.append(int(v))
            s.seeds = seeds[:50] or [1]
        if isinstance(d.get('weathers'), list):
            from . import simrig as SR
            s.weathers = list(dict.fromkeys(
                w for w in d['weathers'] if isinstance(w, str) and (w in SR.CARLA_WEATHERS or
                                                                     (w.startswith('vantage:') and len(w) < 64))))[:16]
        if isinstance(d.get('scenarios'), list):
            from ..sim import carla_bridge as CB
            sc = []
            for x in d['scenarios'][:16]:
                if isinstance(x, str):
                    x = {'name': x}
                if isinstance(x, dict) and x.get('name') in CB.SCENARIOS:
                    c = CB.clean_scenario(x['name'], x.get('params'))[0]
                    c['params'] = {k: v for k, v in c['params'].items() if k in (x.get('params') or {})}
                    sc.append(c)
            s.scenarios = sc
        if isinstance(d.get('fault_sets'), list):
            s.fault_sets = [f for f in (clean_fault_set(x) for x in d['fault_sets'][:16]) if f is not None]
        for k in ('clean', 'multiplex'):
            if isinstance(d.get(k), (bool, int)) and not isinstance(d.get(k), float):
                setattr(s, k, bool(d[k]))
        for k in ('base', 'drivelog', 'health', 'detection'):
            if isinstance(d.get(k), dict):
                setattr(s, k, json.loads(json.dumps(d[k], default=str)))
        if isinstance(d.get('evaluators'), list):
            s.evaluators = [e for e in EVALUATORS if e in d['evaluators']]
        if d.get('asbuilt_on') in ASBUILT_ON:
            s.asbuilt_on = d['asbuilt_on']
        if d.get('server') in SERVERS:
            s.server = d['server']
        if isinstance(d.get('out_dir'), str):
            s.out_dir = d['out_dir'][:4096]
        for k, (lo, hi) in LIMITS.items():
            if k in d and hasattr(s, k):
                v = _num(d[k], lo, hi)
                if v is not None:
                    setattr(s, k, int(round(v)) if k in INTS else v)
        return s


def settings_of(project) -> CampaignSettings:
    d = getattr(project.analysis, 'campaign', None)
    return CampaignSettings.from_dict(d) if d else template('fault_detectability', project)


# =========================================================================== templates
def _cams(var):
    return [c for c in var.cameras if c.visible]


def _pick_cameras(var) -> dict:
    """Which cameras a fault template uses: rotations on the camera looking most forward and the first side camera,
    the clock offset on the one looking most backward, drops and a freeze on the next unused ones; one left clean as a
    control when there are enough."""
    cams = _cams(var)
    if not cams:
        return {}

    def ang(c, a):
        return abs((float(c.yaw) - a + 180.0) % 360.0 - 180.0)
    front = min(cams, key=lambda c: ang(c, 0.0))
    side = next((c for c in cams if c is not front and 30.0 <= ang(c, 0.0) <= 150.0), None)
    used = [front] + ([side] if side else [])
    rest = [c for c in cams if c not in used]
    back = min(rest, key=lambda c: ang(c, 180.0)) if rest else None
    rest = [c for c in rest if c is not back]
    out = {'rotation': [c.tag for c in used], 'clock_offset': [back.tag] if back else [],
           'drop': [rest[0].tag] if rest else [], 'freeze': [rest[-1].tag] if len(rest) >= 3 else
           ([rest[1].tag] if len(rest) == 2 else [])}
    out['control'] = [c.tag for c in rest[1:-1]] if len(rest) >= 3 else []
    return out


TEMPLATES = {
    'fault_detectability': ('Fault detectability', 'Rotations of two cameras about each axis over 0.1–1°, a clock '
                            'offset, dropped frames and a freeze on others, clean drives, three seeds: how small a '
                            'fault each check finds, its false alarms and how well it measures the fault.'),
    'weather_sweep': ('Weather sweep', 'The same scene in six weathers with ground truth: detection recall and range '
                      'per class against Vantage\'s weather model, image statistics, lidar and radar returns.'),
    'rig_ab': ('Rig A vs B', 'Two rig variants driven through the same scenario and seed: detection quality compared '
               'with paired intervals, and the drive-log check of each.'),
}
WEATHER_SWEEP = ('vantage:clear', 'vantage:overcast', 'vantage:light_rain', 'vantage:heavy_rain', 'vantage:fog',
                 'vantage:night')


def template(name, project=None, variant=None) -> CampaignSettings:
    """A campaign from a template, fitted to the project's variant (its camera tags)."""
    s = CampaignSettings(name=TEMPLATES.get(name, ('Campaign',))[0], template=name if name in TEMPLATES else 'custom')
    var = variant or (project.variant if project is not None else None)
    if name == 'fault_detectability':
        pick = _pick_cameras(var) if var is not None else {}
        fs = []
        if pick.get('rotation'):
            fs.append({'kind': 'rotation', 'tags': pick['rotation'], 'axes': ['yaw', 'pitch', 'roll'],
                       'magnitudes': [0.1, 0.2, 0.3, 0.5, 1.0], 'signs': 'alternate'})
        if pick.get('clock_offset'):
            fs.append({'kind': 'clock_offset', 'tags': pick['clock_offset'], 'magnitudes': [1.0, 2.0, 5.0, 10.0]})
        if pick.get('drop'):
            fs.append({'kind': 'drop', 'tags': pick['drop'], 'magnitudes': [1.0, 5.0, 10.0]})
        if pick.get('freeze'):
            fs.append({'kind': 'freeze', 'tags': pick['freeze'], 'magnitudes': [1.0]})
        s.fault_sets = [f for f in (clean_fault_set(x) for x in fs) if f]
        s.seeds = [1, 2, 3]
        # 320 Hz steps: the sample's sync plan rounds to within 1.05 ms (at 80 Hz up to 5.2 ms, which the sync check
        # itself would flag), and 1024 lidar columns per rotation stay whole per step
        s.base = {'duration_s': 10.0, 'traffic': 30, 'tick_hz': 320.0, 'max_tick_hz': 400.0, 'weather': 'ClearNoon',
                  'scenario': {'name': 'free_drive', 'params': {}}, 'rosbag': False}
        s.evaluators = ['drivelog', 'rig_health', 'image_stats', 'returns', 'asbuilt']
        s.asbuilt_on = 'clean'
    elif name == 'weather_sweep':
        s.weathers = list(WEATHER_SWEEP)
        s.seeds = [1]
        s.base = {'duration_s': 12.0, 'traffic': 30, 'scenario': {'name': 'pedestrian_crossing', 'params': {}},
                  'rosbag': False}
        s.evaluators = ['detection', 'image_stats', 'returns']
        s.asbuilt_on = 'none'
    elif name == 'rig_ab':
        vs = [v for v in (project.variants if project is not None else []) if _cams(v)]
        if var is not None and var in vs:
            vs.remove(var)
            vs.insert(0, var)
        s.variants = [v.uid for v in vs[:2]]
        s.seeds = [1]
        s.base = {'duration_s': 12.0, 'traffic': 20, 'scenario': {'name': 'pedestrian_crossing', 'params': {}},
                  'rosbag': False}
        s.evaluators = ['detection', 'drivelog', 'image_stats', 'returns']
        s.asbuilt_on = 'none'
    return CampaignSettings.from_dict(s.to_dict())


# =========================================================================== the design
@dataclass
class Plan:
    drives: list                       # drive specs (dicts), references first
    rigs: dict                         # rig id -> {'variant_uid', 'variant', 'weather'}
    notes: list
    problems: list


def _variants(project, st: CampaignSettings) -> list:
    out = []
    for uid in st.variants or [project.variant.uid]:
        v = project.variant_by_uid(uid)
        if v is not None and v not in out:
            out.append(v)
    return out or [project.variant]


def sensor_kinds(var) -> dict:
    """{tag: kind} of the variant's visible sensors (cameras too)."""
    out = {c.tag: 'camera' for c in var.cameras if c.visible}
    out.update({s.tag: s.kind for s in var.sensors if s.visible})
    return out


def _cond_label(c) -> str:
    unit = FAULT_UNIT[c['kind']]
    sign = '−' if c.get('sign', 1) < 0 else ('+' if c['kind'] in AXES else '')
    ax = f' {c["axis"]}' if c.get('axis') else ''
    return f'{c["tag"]} {c["kind"].replace("_", " ")}{ax} {sign}{c["magnitude"]:g} {unit}'


def _bridge_fault(c: dict, frames_planned: int, rng, duration_s: float) -> dict:
    """The bridge's fault dict for a condition (units converted; a drop share turned into that many frames chosen by
    the seed; a freeze window centred in the drive)."""
    k, m, sg = c['kind'], float(c['magnitude']), float(c.get('sign', 1))
    if k == 'rotation':
        v = [0.0, 0.0, 0.0]
        v[AXES[k].index(c['axis'])] = sg * m
        return {'tag': c['tag'], 'kind': 'rotation', 'deg': v}
    if k == 'translation':
        v = [0.0, 0.0, 0.0]
        v[AXES[k].index(c['axis'])] = sg * m / 100.0
        return {'tag': c['tag'], 'kind': 'translation', 'm': v}
    if k == 'clock_offset':
        return {'tag': c['tag'], 'kind': 'clock_offset', 'ms': m}
    if k == 'range_bias':
        return {'tag': c['tag'], 'kind': 'range_bias', 'm': m / 100.0}
    if k == 'drop':
        n = max(1, int(round(m / 100.0 * frames_planned))) if frames_planned else 0
        n = min(n, max(frames_planned - 2, 0))
        # never the first or last two frames (a drop at an end does not show as a gap in the stamps)
        pool = np.arange(2, max(frames_planned - 2, 2))
        fr = sorted(int(x) for x in rng.choice(pool, size=min(n, len(pool)), replace=False)) if len(pool) else []
        return {'tag': c['tag'], 'kind': 'drop', 'frames': fr}
    if k == 'freeze':
        a = max(0.0, duration_s / 2.0 - m / 2.0)
        return {'tag': c['tag'], 'kind': 'freeze', 'from_s': round(a, 4), 'to_s': round(min(duration_s, a + m), 4)}
    raise ValueError(k)


def _sub_seed(*parts) -> int:
    return int.from_bytes(hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).digest()[:8], 'little')


def _planned_frames(project, var, tag, sim, duration_s) -> int:
    """Frames a sensor writes in a drive: a camera at the reference lidar's rate (camera_rate 'lidar') or its module's,
    any other sensor at its model's rate (as core.simrig plans them)."""
    cam = next((c for c in var.cameras if c.tag == tag), None)
    rate = None
    if cam is not None:
        if sim.camera_rate == 'lidar':
            lid = next((s for s in var.sensors if s.visible and s.kind == 'lidar'), None)
            if lid is not None:
                rate = float(project.sensor_model_for(lid).rate_hz or 0) or None
        if rate is None:
            rate = float(project.module_for(cam).fps or 0) or 30.0
    else:
        s = next((x for x in var.sensors if x.tag == tag), None)
        rate = float(project.sensor_model_for(s).rate_hz or 0) if s is not None else 0.0
        rate = rate or 10.0
    return int(math.floor(duration_s * rate + 1e-6))


def base_sim(project, st: CampaignSettings):
    """The campaign's CARLA settings: the project's analysis.sim with the campaign's base over it; faults and an
    actors scene never come from the project (each drive states its own)."""
    from . import simrig as SR
    d = dict(SR.settings_of(project).to_dict())
    d.update(st.base or {})
    d['faults'] = []
    d['actors'] = []
    return SR.SimSettings.from_dict(d)


def plan(project, st: CampaignSettings | None = None) -> Plan:
    """The drives of the campaign, in the order they run: per variant a reference drive when rig health needs CARLA's
    own car, then seed by seed (so a campaign stopped half-way has whole seeds), every setting with its clean drive and
    its fault drives."""
    st = st or settings_of(project)
    sim = base_sim(project, st)
    notes, problems = [], []
    drives, rigs = [], {}
    weathers = st.weathers or [sim.weather]
    scens = st.scenarios or [dict(sim.scenario)]
    gt_main = 'semantic,instance,range' if 'detection' in st.evaluators else 'none'
    variants = _variants(project, st)
    dur = float(sim.duration_s)
    n = 0
    for var in variants:
        kinds = sensor_kinds(var)
        if 'rig_health' in st.evaluators and _cams(var):
            rid = _rig_id(var, weathers[0])
            rigs.setdefault(rid, {'variant_uid': var.uid, 'variant': var.name, 'weather': weathers[0]})
            drives.append({'id': f'ref-{_slug(var.name)}', 'role': 'reference', 'rig': rid, 'variant_uid': var.uid,
                           'variant': var.name, 'weather': weathers[0], 'scenario': {'name': 'parked', 'params': {}},
                           'seed': int(st.seeds[0]) if st.seeds else 1, 'repeat': 0, 'faults': [], 'conditions': [],
                           'clean': True, 'gt': 'instance', 'duration_s': REFERENCE_S})
    for r, seed in enumerate(st.seeds):
        for var in variants:
            kinds = sensor_kinds(var)
            lanes = {}                       # tag -> [condition]
            for fs in st.fault_sets:
                for tag in fs['tags']:
                    k = kinds.get(tag)
                    if k is None:
                        msg = f'{FAULT_LABEL[fs["kind"]]} on {tag}: {var.name} has no visible sensor {tag}.'
                        if msg not in problems:
                            problems.append(msg)
                        continue
                    if k not in FAULT_SENSORS[fs['kind']]:
                        msg = f'{FAULT_LABEL[fs["kind"]]} does not apply to {tag} (a {k}).'
                        if msg not in problems:
                            problems.append(msg)
                        continue
                    for ax in fs.get('axes') or [None]:
                        for m in fs['magnitudes']:
                            sign = 1
                            if fs['kind'] in AXES:
                                sg = fs.get('signs', 'alternate')
                                sign = (-1 if r % 2 else 1) if sg == 'alternate' else -1 if sg == 'negative' else \
                                    (1 if np.random.default_rng(_sub_seed(seed, tag, ax, m)).random() < 0.5 else -1) \
                                    if sg == 'random' else 1
                            c = {'tag': tag, 'sensor_kind': k, 'kind': fs['kind'], 'axis': ax, 'magnitude': float(m),
                                 'sign': sign, 'unit': FAULT_UNIT[fs['kind']]}
                            c['label'] = _cond_label(c)
                            lanes.setdefault(tag, []).append(c)
            for L in lanes.values():          # the seed shuffles each sensor's schedule (sensors never in step)
                order = np.random.default_rng(_sub_seed(seed, L[0]['tag'], 'lane')).permutation(len(L))
                L[:] = [L[i] for i in order]
            for w in weathers:
                rid = _rig_id(var, w)
                rigs.setdefault(rid, {'variant_uid': var.uid, 'variant': var.name, 'weather': w})
                for sc in scens:
                    common = {'rig': rid, 'variant_uid': var.uid, 'variant': var.name, 'weather': w, 'scenario': sc,
                              'seed': int(seed), 'repeat': r, 'gt': gt_main, 'duration_s': dur}
                    groups = []
                    if st.clean or not lanes:
                        groups.append([])
                    if lanes:
                        if st.multiplex:
                            for j in range(max(len(L) for L in lanes.values())):
                                groups.append([L[j % len(L)] for L in lanes.values()])
                        else:
                            for L in lanes.values():
                                groups.extend([c] for c in L)
                    for conds in groups:
                        n += 1
                        rng = np.random.default_rng(_sub_seed(seed, n, [c['label'] for c in conds]))
                        faults = [_bridge_fault(c, _planned_frames(project, var, c['tag'], sim, dur), rng, dur)
                                  for c in conds]
                        drives.append(dict(common, id=f'd{n:04d}', role='drive', faults=faults,
                                           conditions=conds, clean=not conds))
    if len(drives) > st.max_drives:
        problems.append(f'The design has {len(drives)} drives; only the first {st.max_drives} are kept '
                        '(raise the limit to run them all).')
        drives = drives[:st.max_drives]
    if st.multiplex and st.fault_sets:
        notes.append('Faults of different sensors share drives (multiplexed): each check judges each sensor on its own; '
                     'cross-talk is reported.')
    if 'rig_health' in st.evaluators:
        notes.append('Rig health\'s sync is judged against the rig\'s schedule (rig_health.capture_offsets): the stagger '
                     'the sync plan gives the cameras is no skew.')
    if 'drivelog' in st.evaluators and any(fs.get('kind') == 'freeze' for fs in st.fault_sets):
        notes.append('The drive-log check\'s freeze finding (identical images in a row) feeds drivelog.streams.')
    return Plan(drives, rigs, notes, problems)


def _slug(s) -> str:
    from .. import workdir as WD
    return WD.slug(s)


def _rig_id(var, weather) -> str:
    return f'{_slug(var.name)}--{_slug(weather)}'


# =========================================================================== cost
def drive_class(d) -> str:
    g = d.get('gt') or 'none'
    return 'none' if g == 'none' else 'instance' if g == 'instance' else 'full'


def estimate(project, st: CampaignSettings | None = None, pl: Plan | None = None, out_dir=None) -> dict:
    """The cost before running: drives × the measured wall seconds per drive (this campaign's own drives when it has
    any, else the development machine's figures), what is already done, and the evaluations."""
    st = st or settings_of(project)
    pl = pl or plan(project, st)
    out_dir = out_dir or campaign_dir(project, st)
    sim = base_sim(project, st)
    rates = measured_rates(out_dir)
    total, todo, done = 0.0, 0.0, 0
    for d in pl.drives:
        c = drive_class(d)
        per_s, over = rates.get(c, COST_DEFAULT[c])
        w = over + per_s * float(d['duration_s'])      # the bridge's wall time per simulated second holds its start
        total += w
        if _drive_done(out_dir, d) is not None:
            done += 1
        else:
            todo += w
    n_img = 0
    for d in pl.drives:
        if d['role'] == 'drive' and 'detection' in st.evaluators:
            var = project.variant_by_uid(d['variant_uid']) or project.variant
            n_img += sum(_planned_frames(project, var, c.tag, sim, float(d['duration_s'])) for c in _cams(var))
    ev = 0.0
    nd = sum(d['role'] == 'drive' for d in pl.drives)
    for e in st.evaluators:
        if e == 'detection':
            ev += EVAL_COST_S[e] * n_img
        elif e == 'asbuilt':
            ev += EVAL_COST_S[e] * sum(1 for d in pl.drives if _asbuilt_wanted(st, d))
        else:
            ev += EVAL_COST_S[e] * nd
    return {'drives': len(pl.drives), 'done': done, 'drive_s': round(total, 1), 'remaining_s': round(todo, 1),
            'evaluate_s': round(ev, 1), 'measured': {k: [round(v[0], 2), round(v[1], 1)] for k, v in rates.items()},
            'basis': ('this campaign\'s drives' if rates else 'the development machine (RTX 5080, the sample\'s data '
                                                              'rig); measured figures replace it after the first drive'),
            'text': _cost_text(len(pl.drives), done, total, todo, ev)}


def _cost_text(n, done, total, todo, ev) -> str:
    def hm(s):
        s = int(round(s))
        return f'{s // 3600} h {s % 3600 // 60:02d} min' if s >= 3600 else f'{s // 60} min {s % 60:02d} s'
    left = f'{n - done} to drive ({hm(todo)})' if done else f'{hm(total)} of CARLA'
    return f'{n} drives{f", {done} done" if done else ""}: {left}, then about {hm(ev)} of checks'


def measured_rates(out_dir) -> dict:
    """{ground-truth class: (wall s per simulated s, overhead s per drive)} from the drives done in out_dir."""
    rows = {}
    root = os.path.join(out_dir or '', 'drives')
    if not os.path.isdir(root):
        return {}
    for name in sorted(os.listdir(root)):
        doc = _read_json(os.path.join(root, name, 'campaign_drive.json'))
        if not isinstance(doc, dict) or doc.get('status') != 'ok':
            continue
        perf = doc.get('performance') or {}
        sim_s, wall = perf.get('simulated_s'), perf.get('wall_s')
        tot = doc.get('wall_s')
        if not sim_s or not wall or not tot:
            continue
        rows.setdefault(drive_class(doc.get('drive') or {}), []).append((wall / sim_s, max(0.0, tot - wall)))
    return {k: (float(np.median([r[0] for r in v])), float(np.median([r[1] for r in v]))) for k, v in rows.items()}


# =========================================================================== files and keys
def campaign_dir(project, st: CampaignSettings, base: str | None = None) -> str:
    """The campaign's folder: its own out_dir, else <working folder>/<project>/campaigns/<name> (base: the working
    folder the user chose; None: the default one)."""
    if st.out_dir:
        return os.path.abspath(os.path.expanduser(st.out_dir))
    from .. import workdir as WD
    return WD.project_dir(project, base, 'campaigns', WD.slug(st.name))


def _read_json(path):
    try:
        with open(path, encoding='utf-8') as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _write_json(path, doc):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.part'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(doc, fh, indent=1, default=_jsonable)
    os.replace(tmp, path)


def _jsonable(o):
    if isinstance(o, (np.floating,)):
        v = float(o)
        return v if math.isfinite(v) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=_jsonable).encode()).hexdigest()


# rig-file keys that do not change what CARLA records: identifiers (the same rig in a project re-made, with new uids,
# is the same), when it was written, prose (assumptions, notes) and the render and memory budget the plan reports
RIG_HASH_IGNORED = ('uid', 'variant_uid', 'created', 'assumptions', 'notes', 'bridge_memory_gb', 'render_mpx_per_frame')


def rig_hash(rig_dir) -> str:
    """The rig file's content (everything but RIG_HASH_IGNORED) and its remap tables' arrays (names, types, shapes
    and values: the .npz file's own bytes carry the zip entries' times)."""
    def strip(o):
        if isinstance(o, dict):
            return {k: strip(v) for k, v in o.items() if k not in RIG_HASH_IGNORED}
        if isinstance(o, list):
            return [strip(v) for v in o]
        return o
    doc = strip(_read_json(os.path.join(rig_dir, 'sim_rig.json')) or {})
    npz = os.path.join(rig_dir, 'sim_remap.npz')
    arrays = None
    if os.path.isfile(npz):
        h = hashlib.sha256()
        with np.load(npz) as z:
            for k in sorted(z.files):
                a = np.ascontiguousarray(z[k])
                h.update(f'{k}|{a.dtype.str}|{a.shape}|'.encode())
                h.update(a.tobytes())
        arrays = h.hexdigest()
    return _sha({'rig': doc, 'remap': arrays})


def drive_args(d: dict, sim, faults_path: str) -> list:
    """The bridge's command line for a drive (beside the rig file and the output folder)."""
    a = ['--seed', str(int(d['seed'])), '--scenario', d['scenario']['name'],
         '--scenario-params', json.dumps(d['scenario'].get('params') or {}, sort_keys=True),
         '--faults', faults_path if d['faults'] else 'none', '--ground-truth', d.get('gt') or 'none',
         '--actors', 'none', '--duration', f'{float(d["duration_s"]):g}']
    if d['role'] == 'reference':
        a += ['--traffic', '0']
    return a


def drive_key(d: dict, rhash: str) -> str:
    keep = {k: d[k] for k in ('role', 'seed', 'scenario', 'faults', 'gt', 'duration_s')}
    return _sha({'rig': rhash, 'drive': keep})


def _drive_done(out_dir, d) -> dict | None:
    """The drive's record when it ran to the end with the key it has now (the rig unchanged), else None."""
    root = os.path.join(out_dir, 'drives', d['id'])
    doc = _read_json(os.path.join(root, 'campaign_drive.json'))
    if not isinstance(doc, dict) or doc.get('status') != 'ok':
        return None
    rig_dir = os.path.join(out_dir, 'rigs', d['rig'])
    rh = _read_json(os.path.join(rig_dir, 'rig_key.json')) or {}
    if not rh.get('hash') or doc.get('key') != drive_key(d, rh['hash']):
        return None
    run = _read_json(os.path.join(root, 'run.json'))
    if not isinstance(run, dict) or run.get('status') != 'ok':
        return None
    return doc


# =========================================================================== running
@dataclass
class DriveState:
    id: str
    status: str                        # ok | reused | failed | cancelled | pending
    wall_s: float = 0.0
    error: str = ''


@dataclass
class CampaignResult:
    settings: CampaignSettings
    out_dir: str
    plan: Plan
    states: dict                       # drive id -> DriveState
    evaluations: dict                  # drive id -> {evaluator: result dict}
    observations: list
    detectability: dict
    weather: dict | None
    rigs: dict | None
    cost: dict
    notes: list
    seconds: float = 0.0
    complete: bool = True
    variant: str = ''

    def drive(self, did):
        return next((d for d in self.plan.drives if d['id'] == did), None)


class Cancelled(Exception):
    pass


def _bridge_subprocess(python, timeout_s):
    """A runner: the bridge in CARLA's Python as a capped subprocess, its progress read from stdout; SIGINT on a
    cancel (the bridge tidies CARLA up and removes what it wrote)."""
    from ..sim import server as SV

    def runner(rig_json, out_dir, args, progress, cancel):
        cmd, env = SV.bridge_command(python, rig_json, out_dir, list(args) + ['--timeout', f'{timeout_s:g}'])
        with open(os.path.join(out_dir, 'bridge.log'), 'w', encoding='utf-8') as log:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=log, text=True, env=env, cwd=out_dir)
            asked = False
            try:
                for line in proc.stdout:
                    pr = SV.parse_progress(line)
                    if pr is not None:
                        if progress is not None:
                            progress(pr[0], pr[1])
                    else:
                        log.write(line)
                    if cancel() and not asked:
                        proc.send_signal(signal.SIGINT)
                        asked = True
                proc.wait()
            finally:
                if proc.poll() is None:
                    proc.send_signal(signal.SIGINT)
                    try:
                        proc.wait(60)
                    except subprocess.TimeoutExpired:
                        proc.kill()
        return proc.returncode
    return runner


def carla_python() -> str:
    """CARLA's client Python: the CARLA page's choice (QSettings sim/python), else next to a CARLA folder found."""
    py = os.environ.get('VANTAGE_CARLA_PYTHON', '')
    if py and os.path.isfile(py):
        return py
    try:
        from PyQt6.QtCore import QSettings
        saved = str(QSettings('Vantage', 'Vantage').value('sim/python', '') or '')
    except Exception:                          # noqa: BLE001 — no Qt or no settings: look for it
        saved = ''
    if saved and os.path.isfile(saved):
        return saved
    from ..sim import server as SV
    return next((p for p in (SV.client_python(r) for r in SV.candidate_roots()) if p), '')


def _carla_root() -> str:
    try:
        from PyQt6.QtCore import QSettings
        saved = str(QSettings('Vantage', 'Vantage').value('sim/carla_root', '') or '')
    except Exception:                          # noqa: BLE001
        saved = ''
    from ..sim import server as SV
    if saved and SV.is_carla_root(saved):
        return saved
    return next(iter(SV.candidate_roots()), '')


def _prepare_rig(project, st, sim, rid, info, out_dir, progress=None) -> str:
    """Build (or keep) the rig file of a variant and weather; returns its hash."""
    from . import simrig as SR
    rig_dir = os.path.join(out_dir, 'rigs', rid)
    var = project.variant_by_uid(info['variant_uid']) or project.variant
    d = sim.to_dict()
    d['weather'] = info['weather']
    s2 = SR.SimSettings.from_dict(d)
    rig = SR.build(project, s2, variant=var)
    if rig is None:
        raise RuntimeError(f'{var.name}: nothing to simulate.')
    new = rig_dir + '.new'
    shutil.rmtree(new, ignore_errors=True)
    SR.export(rig, project, new)
    h = rig_hash(new)
    old = _read_json(os.path.join(rig_dir, 'rig_key.json')) or {}
    if old.get('hash') and os.path.isfile(os.path.join(rig_dir, 'sim_rig.json')) and \
            (old['hash'] == h or rig_hash(rig_dir) == h):
        # the same rig: its file and its key stay (the drives are keyed on it). A key made by an older rig_hash, or
        # a file whose prose or budget figures a newer Vantage words differently, still holds the same recording
        shutil.rmtree(new, ignore_errors=True)
        return old['hash']
    shutil.rmtree(rig_dir, ignore_errors=True)
    os.replace(new, rig_dir)
    _write_json(os.path.join(rig_dir, 'rig_key.json'), {'hash': h, 'variant': var.name, 'weather': info['weather']})
    return h


def run_drives(project, st: CampaignSettings, pl: Plan, out_dir, progress=None, cancel=None, runner=None,
               python=None, only=None) -> dict:
    """Drive every planned drive not done yet (only: a set of drive ids). Returns {id: DriveState}. Raises
    RuntimeError when no server answers (and none is to be started) or CARLA's Python is missing; Cancelled on a
    cancel (the drive under way is removed by the bridge; the campaign resumes from there)."""
    from ..sim import server as SV
    stop = cancel or (lambda: False)
    sim = base_sim(project, st)
    states = {}
    todo = [d for d in pl.drives if (only is None or d['id'] in only)]
    pending = [d for d in todo if _drive_done(out_dir, d) is None]
    rig_hashes = {}
    for rid in sorted({d['rig'] for d in todo}):
        rig_hashes[rid] = _prepare_rig(project, st, sim, rid, pl.rigs[rid], out_dir)
    pending = [d for d in todo if _drive_done(out_dir, d) is None]
    for d in todo:
        if d not in pending:
            states[d['id']] = DriveState(d['id'], 'reused', float((_drive_done(out_dir, d) or {}).get('wall_s', 0.0)))
    if not pending:
        return states
    server = None
    own_runner = runner is None
    if runner is None:
        py = python or carla_python()
        if not py or not os.path.isfile(py):
            raise RuntimeError('No Python for CARLA\'s client: choose it on Simulate ▸ CARLA.')
        if not SV.port_open('127.0.0.1', sim.port):
            if st.server != 'own':
                raise RuntimeError(f'No CARLA server answers on port {sim.port}: start one on Simulate ▸ CARLA, or let '
                                   'the campaign start its own.')
            root = _carla_root()
            if not root:
                raise RuntimeError('No CARLA folder found: choose it on Simulate ▸ CARLA.')
            server = SV.Server(root, sim.port, sim.memory_gb, sim.quality, False,
                               os.path.join(out_dir, 'server.log'))
            os.makedirs(out_dir, exist_ok=True)
            server.start()
            t0 = time.monotonic()
            while not server.ready():
                if stop():
                    server.stop()
                    raise Cancelled()
                if not server.running or time.monotonic() - t0 > 240:
                    server.stop()
                    raise RuntimeError('The CARLA server did not come up (see server.log).')
                time.sleep(1.0)
            time.sleep(5.0)                      # the world loads after the port opens
        runner = _bridge_subprocess(py, sim.timeout_s)
    fails = 0
    try:
        for i, d in enumerate(pending):
            if stop():
                raise Cancelled()
            root = os.path.join(out_dir, 'drives', d['id'])
            shutil.rmtree(root, ignore_errors=True)
            os.makedirs(root, exist_ok=True)
            fpath = os.path.join(out_dir, 'faults', d['id'] + '.json')
            if d['faults']:
                _write_json(fpath, d['faults'])
            rig_json = os.path.join(out_dir, 'rigs', d['rig'], 'sim_rig.json')
            args = ['--port', str(sim.port), '--tm-port', str(sim.tm_port)] + drive_args(d, sim, fpath)

            def sub(f, text='', i=i, d=d):
                if progress is not None:
                    progress((i + min(max(f, 0.0), 1.0)) / len(pending),
                             f'Drive {i + 1} of {len(pending)} ({d["id"]}): {text}')
            t0 = time.monotonic()
            code = runner(rig_json, root, args, sub, stop)
            wall = time.monotonic() - t0
            run = _read_json(os.path.join(root, 'run.json'))
            if stop():
                states[d['id']] = DriveState(d['id'], 'cancelled', wall)
                raise Cancelled()
            if not isinstance(run, dict) or run.get('status') != 'ok':
                why = (run or {}).get('error') or (f'the bridge exited with {code}; see bridge.log' if code is not None
                                                    else 'it wrote no report')
                states[d['id']] = DriveState(d['id'], 'failed', wall, str(why).splitlines()[0][:300])
                fails += 1
                if fails >= 2 and own_runner and not SV.port_open('127.0.0.1', sim.port):
                    raise RuntimeError(f'The CARLA server stopped answering on port {sim.port}.')
                continue
            fails = 0
            _write_json(os.path.join(root, 'campaign_drive.json'),
                        {'schema': DRIVE_SCHEMA, 'status': 'ok', 'key': drive_key(d, rig_hashes[d['rig']]),
                         'drive': d, 'wall_s': round(wall, 2), 'performance': run.get('performance') or {},
                         'faults_injected': run.get('faults_injected') or []})
            states[d['id']] = DriveState(d['id'], 'ok', wall)
    finally:
        if server is not None:
            server.stop()
    return states


# =========================================================================== reading a drive
def nus_streams(drive_dir) -> dict:
    """{channel: {'modality', 'frames': [(timestamp µs, path)]}} of a drive's nuScenes folder, every sample and sweep,
    in time order."""
    V = os.path.join(drive_dir, 'nuscenes', 'v1.0-vantage')
    sd = _read_json(os.path.join(V, 'sample_data.json')) or []
    cs = {c['token']: c for c in (_read_json(os.path.join(V, 'calibrated_sensor.json')) or [])}
    sn = {s['token']: s for s in (_read_json(os.path.join(V, 'sensor.json')) or [])}
    out = {}
    for r in sd:
        c = cs.get(r.get('calibrated_sensor_token'))
        s = sn.get(c['sensor_token']) if c else None
        if s is None:
            continue
        e = out.setdefault(s['channel'], {'modality': s.get('modality'), 'frames': []})
        e['frames'].append((int(r['timestamp']), os.path.join(drive_dir, 'nuscenes', r['filename'])))
    for e in out.values():
        e['frames'].sort()
    return out


def _channel_of(tag, streams) -> str | None:
    for ch in (f'CAM_{tag}', f'LIDAR_{tag}', f'RADAR_{tag}', tag):
        if ch in streams:
            return ch
    return None


# =========================================================================== evaluators
def _eval_drivelog(project, var, drive_dir, st: CampaignSettings, engine, cancel) -> dict:
    from . import drivelog as DL
    dls = DL.DriveLogSettings.from_dict(dict({'frames_per_camera': 20}, **(st.drivelog or {})))
    t0 = time.perf_counter()
    res = DL.run(project, os.path.join(drive_dir, 'nuscenes'), dls, cancel=cancel, engine=engine, variant=var)
    if res is None:
        raise Cancelled()
    cams = {}
    for r in res.extrinsics:
        cams[r['tag']] = {'verdict': r.get('verdict'), 'confidence': r.get('confidence'),
                          'delta_deg': r.get('delta_deg'), 'stderr_deg': r.get('stderr_deg'),
                          'observable': r.get('observable'), 'correction_deg': r.get('correction_deg'),
                          'frames': r.get('frames'), 'agree': r.get('agree'), 'reason': r.get('reason')}
    sync = {r['tag']: {'status': r.get('status'), 'own_ms': r.get('own_ms'), 'offset_ms': r.get('offset_ms'),
                       'skew_ms': r.get('skew_ms'), 'spread_ms': r.get('spread_ms')}
            for r in res.sync['cameras'] if r.get('topic')}
    streams = {e['tag']: {'status': e['status'], 'kind': e['kind'], 'rate_hz': e.get('rate_hz'),
                          'expected_hz': e.get('expected_hz'), 'drops': e.get('drops', 0),
                          'drop_share': e.get('drop_share', 0.0), 'messages': e.get('messages'),
                          'problems': list(e.get('problems') or [])} for e in res.streams}
    return {'cameras': cams, 'sync': sync, 'streams': streams,
            'findings': [{'severity': f['severity'], 'section': f['section'], 'tags': f['tags'],
                          'message': f['message']} for f in res.findings],
            'lidar_phase_error_deg': (res.sync.get('lidar') or {}).get('phase_error_deg'),
            'tolerance_deg': dls.tolerance_deg, 'sync_tol_ms': dls.sync_tol_ms,
            'seconds': round(time.perf_counter() - t0, 2)}


def _lens_valid(project, cam, size) -> np.ndarray:
    """(h, w) bool: the output pixels inside the lens's image circle (the module's lens model, scaled)."""
    from .lens import resolve
    intr = resolve(project.module_for(cam), project.lens_for(cam))
    w, h = int(size[0]), int(size[1])
    sx, sy = intr.width / w, intr.height / h
    U, V = np.meshgrid((np.arange(w) + 0.5) * sx, (np.arange(h) + 0.5) * sy)
    _, th = intr.unproject(U.ravel(), V.ravel())
    return (th <= intr.max_theta).reshape(h, w)


def body_masks(project, var, ref_dir, cache=True) -> dict:
    """{camera tag: PNG bytes} of CARLA's own car as each camera sees it at its planned mount, from the reference
    drive's ground truth: a pixel is the car's when it is class 'ego' in at least half the frames; inside the lens
    circle 128, the car 255, outside 0 (vantage.core.carla_live.body_view's encoding)."""
    import cv2
    from . import groundtruth as GT
    out_dir = os.path.join(ref_dir, 'campaign_body')
    if cache and os.path.isdir(out_dir):
        got = {}
        for f in os.listdir(out_dir):
            if f.endswith('.png'):
                with open(os.path.join(out_dir, f), 'rb') as fh:
                    got[f[:-4]] = fh.read()
        if got:
            return got
    doc = GT.read(ref_dir)
    if doc is None:
        return {}
    res = {}
    for tag, c in doc.get('cameras', {}).items():
        cam = next((x for x in var.cameras if x.tag == tag), None)
        if cam is None or not c.get('frames'):
            continue
        acc, n = None, 0
        for fr in c['frames']:
            if not fr.get('semantic'):
                continue
            sem = GT.load_semantic(doc, fr)
            if sem is None:
                continue
            m = (sem == GT.EGO_CLASS).astype(np.uint16)
            acc = m if acc is None else acc + m
            n += 1
        if acc is None:
            continue
        ego = acc * 2 >= n
        valid = _lens_valid(project, cam, (ego.shape[1], ego.shape[0]))
        img = np.where(valid, np.where(ego, 255, 128), 0).astype(np.uint8)
        ok, buf = cv2.imencode('.png', img)
        if ok:
            res[tag] = buf.tobytes()
    if res:
        os.makedirs(out_dir, exist_ok=True)
        for tag, b in res.items():
            with open(os.path.join(out_dir, f'{tag}.png'), 'wb') as fh:
                fh.write(b)
    return res


def _camera_rate(rig_doc, tag) -> float | None:
    for c in (rig_doc or {}).get('cameras', []):
        if c['tag'] == tag:
            return float(c.get('capture', {}).get('recorded_hz') or c.get('capture', {}).get('rate_hz') or 0) or None
    return None


def _eval_rig_health(project, var, drive_dir, rig_doc, bodies, st: CampaignSettings, cancel) -> dict:
    import cv2
    from . import rig_health as RH
    from . import carla_live as CL
    streams = nus_streams(drive_dir)
    run = _read_json(os.path.join(drive_dir, 'run.json')) or {}
    dur = float((run.get('performance') or {}).get('simulated_s') or 0.0)
    hs = RH.HealthSettings.from_dict(dict(st.health or {}, window_s=max(dur, 1.0) + 60.0))
    cams = {}
    first_t = None
    times = {}
    for cam in _cams(var):
        if cancel is not None and cancel():
            raise Cancelled()
        ch = _channel_of(cam.tag, streams)
        if ch is None:
            cams[cam.tag] = {'frames': 0, 'verdict': 'fault', 'reasons': [['fault', 'no frames']]}
            continue
        frames = streams[ch]['frames']
        rate = _camera_rate(rig_doc, cam.tag) or 10.0
        mod = project.module_for(cam)
        sens = (int(mod.width_px or 0), int(mod.height_px or 0))
        ss = RH.StreamStats(rate, None, hs)
        max_run = 0
        t = np.array([f[0] for f in frames], np.int64)
        times[cam.tag] = t / 1e6
        if first_t is None and len(t):
            first_t = t
        for ts, path in frames:
            g = cv2.imread(path, cv2.IMREAD_REDUCED_GRAYSCALE_2)
            if g is None:
                continue
            ss.add(g, ts / 1e6)
            max_run = max(max_run, ss.frozen_run)
        last = (frames[-1][0] / 1e6) if frames else 0.0
        s = ss.summary(now=last)
        s['frozen_frames'] = max_run
        drift, n_d = None, 0
        ev = None
        if bodies.get(cam.tag) is not None and sens[0] > 0:
            try:
                ev = CL.body_view(bodies[cam.tag], sens, key=('carla', cam.tag))
            except ValueError:
                ev = None
        if ev is not None and len(ev.edge) >= 20 and frames:
            vals = []
            for k in np.unique(np.linspace(0, len(frames) - 1, min(DRIFT_FRAMES, len(frames))).astype(int)):
                img = cv2.imread(frames[k][1], cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                a = RH.alignment(img, ev)
                if a is not None:
                    vals.append(a)
            n_d = len(vals)
            drift = float(np.median(vals)) if vals else None
        verdict, reasons = RH.verdict(s, hs, started=None, now=last, drift_px=drift)
        cams[cam.tag] = {'frames': int(s['frames']), 'fps': s['fps'], 'nominal_fps': s['nominal_fps'],
                         'dropped': int(s['dropped']), 'dropped_share': s['dropped_share'],
                         'max_frozen': int(max_run), 'mean': s.get('mean'), 'sharpness': s.get('sharpness'),
                         'clip_bright': s.get('clip_bright'), 'clip_dark': s.get('clip_dark'),
                         'drift_px': drift, 'drift_frames': n_d, 'body_edge_px': int(len(ev.edge)) if ev else 0,
                         'verdict': verdict, 'reasons': [list(r) for r in reasons]}
    planned = RH.schedule_offsets(rig_doc)            # the rig's own capture schedule: the plan's stagger
    offs, ref_tag = RH.capture_offsets(times, planned)
    for tag, c in cams.items():
        o = offs.get(tag)
        c['offset_ms'] = None if o is None else o['offset_ms']
        c['planned_offset_ms'] = None if o is None else o['planned_ms']
        c['skew_ms'] = None if o is None else o['skew_ms']
        c['sync_flag'] = o is not None and abs(o['skew_ms']) > hs.sync_warn_ms
    return {'cameras': cams, 'reference_camera': ref_tag, 'settings': hs.to_dict(),
            'body': 'CARLA\'s own car from the reference drive' if bodies else 'none (no reference drive)'}


def _eval_asbuilt(project, var, drive_dir, engine, cancel) -> dict:
    from . import asbuilt as AB
    t0 = time.perf_counter()
    ab = AB.AsBuiltSettings.from_dict({'frames': 12, 'min_motion_m': 0.5})
    try:
        res = AB.run(project, engine, ab, variant=var, path=os.path.join(drive_dir, 'nuscenes'), cancel=cancel)
    except ValueError as e:
        return {'error': str(e), 'cameras': {}}
    if res is None:
        raise Cancelled()
    s = AB.summary(res)
    return {'verdict': s['verdict'], 'cameras': {r['camera']: r for r in s['per_camera']},
            'findings': s['findings'][:20], 'seconds': round(time.perf_counter() - t0, 2)}


def _eval_detection(project, drive_dir, st: CampaignSettings, cancel, detector=None) -> dict:
    from . import detection_eval as DE
    base = DE.settings_of(project).to_dict()
    base.update(st.detection or {})
    base['recordings'] = [drive_dir]
    ds = DE.DetectionSettings.from_dict(base)
    res = DE.run(None, settings=ds, cancel=cancel, detector=detector)
    if res is None:
        raise Cancelled()
    m0 = res.modes()[0]
    rows = [{'camera': r['camera'], 'class': r['class'], 'range_bin': r['range_bin'], 'n_gt': int(r['n_gt']),
             'detected': int(r.get('detected') or 0), 'recall': r.get('recall'), 'recall_ci': r.get('recall_ci'),
             'precision': r.get('precision'), 'ap50': r.get('ap50'), 'n_det': r.get('n_det'), 'tp': r.get('tp')}
            for r in res.rows if r['rec'] == 0 and r['mode'] == m0]
    ranges = [{'camera': r['camera'], 'class': r['class'], 'text': r.get('text'),
               'range_m': r.get('range_m')} for r in res.ranges if r['rec'] == 0 and r['mode'] == m0]
    from . import detection as DT
    return {'rows': rows, 'ranges': ranges, 'range_edges': list(ds.range_edges), 'detector': res.detector,
            'score': ds.score, 'classes': list(ds.classes), 'class_label': {c: DT.CLASS_LABEL.get(c, c)
                                                                           for c in ds.classes},
            'images': sum(v.get('images', 0) for v in res.timing.values() if isinstance(v, dict)),
            'seconds': round(res.seconds, 2)}


def noise_sigma(g: np.ndarray, valid=None) -> float | None:
    """Immerkær's (1996) fast noise estimate of a grey image: σ = √(π/2) · mean|I ∗ N| / 6, N the difference of two
    Laplacians (it cancels any plane), over the valid pixels away from edges (the strongest tenth of the gradient left
    out, as texture would otherwise count as noise)."""
    import cv2
    g = g.astype(np.float32)
    N = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    r = np.abs(cv2.filter2D(g, -1, N, borderType=cv2.BORDER_REPLICATE))
    G = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    m = np.ones(g.shape, bool) if valid is None else cv2.erode(valid.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    m[[0, -1], :] = False
    m[:, [0, -1]] = False
    if m.sum() < 100:
        return None
    m &= G <= np.percentile(G[m], 90)
    if m.sum() < 100:
        return None
    return float(math.sqrt(math.pi / 2.0) * r[m].mean() / 6.0)


def image_stats(img, valid=None) -> dict:
    """Brightness (mean grey 0–255), RMS contrast (std / mean), noise σ (grey levels), clipped shares and sharpness
    (variance of the Laplacian) of one image inside valid."""
    import cv2
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    m = np.ones(g.shape, bool) if valid is None else valid
    v = g[m].astype(np.float64)
    if not len(v):
        return {}
    mean = float(v.mean())
    lap = cv2.Laplacian(g, cv2.CV_64F)
    return {'mean': mean, 'contrast': float(v.std() / mean) if mean > 0 else None,
            'noise': noise_sigma(g, m), 'clip_bright': float((v >= 253).mean()), 'clip_dark': float((v <= 2).mean()),
            'sharpness': float(lap[m].var())}


def _eval_image_stats(project, var, drive_dir, cancel) -> dict:
    import cv2
    streams = nus_streams(drive_dir)
    out = {}
    for cam in _cams(var):
        if cancel is not None and cancel():
            raise Cancelled()
        ch = _channel_of(cam.tag, streams)
        if ch is None:
            continue
        frames = streams[ch]['frames']
        if not frames:
            continue
        valid = None
        rows = []
        for k in np.unique(np.linspace(0, len(frames) - 1, min(STAT_FRAMES, len(frames))).astype(int)):
            img = cv2.imread(frames[k][1], cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            if valid is None or valid.shape != img.shape:
                valid = _lens_valid(project, cam, (img.shape[1], img.shape[0]))
            rows.append(image_stats(img, valid))
        if not rows:
            continue
        agg = {}
        for k in rows[0]:
            vals = [r[k] for r in rows if r.get(k) is not None]
            agg[k] = float(np.mean(vals)) if vals else None
        agg['frames'] = len(rows)
        out[cam.tag] = agg
    return {'cameras': out}


def _pcd_points(path) -> int | None:
    try:
        with open(path, 'rb') as fh:
            head = fh.read(2048)
    except OSError:
        return None
    for line in head.split(b'\n'):
        if line.startswith(b'POINTS'):
            try:
                return int(line.split()[1])
            except (IndexError, ValueError):
                return None
    return None


def _eval_returns(project, var, drive_dir, cancel) -> dict:
    streams = nus_streams(drive_dir)
    out = {}
    for s in var.sensors:
        if not s.visible or s.kind not in ('lidar', 'radar'):
            continue
        ch = _channel_of(s.tag, streams)
        if ch is None:
            continue
        counts, inten, far = [], [], []
        frames = streams[ch]['frames']
        for k, (_, path) in enumerate(frames):
            if cancel is not None and cancel():
                raise Cancelled()
            if s.kind == 'lidar':
                try:
                    P = np.fromfile(path, np.float32)
                except OSError:
                    continue
                if P.size % 5:
                    continue
                P = P.reshape(-1, 5)
                counts.append(len(P))
                if len(P):
                    inten.append(float(P[:, 3].mean()))
                    far.append(float(np.percentile(np.linalg.norm(P[:, :3], axis=1), 95)))
            else:
                n = _pcd_points(path)
                if n is not None:
                    counts.append(n)
        if counts:
            out[s.tag] = {'kind': s.kind, 'sweeps': len(counts), 'points_mean': float(np.mean(counts)),
                          'points_sd': float(np.std(counts)),
                          'intensity_mean': float(np.mean(inten)) if inten else None,
                          'range_p95_m': float(np.mean(far)) if far else None}
    return {'sensors': out}


def _asbuilt_wanted(st, d) -> bool:
    if 'asbuilt' not in st.evaluators or d['role'] != 'drive':
        return False
    return st.asbuilt_on == 'all' or (st.asbuilt_on == 'clean' and d['clean'])


def eval_key(st: CampaignSettings, name, drive_key_, ref_key=None) -> str:
    """The cache key of one evaluation: the evaluator's settings, the drive's key, the app's version and the checks'
    method revisions (a check whose method changed makes its cached evaluations again, before the version moves)."""
    from . import drivelog as DL
    from . import rig_health as RH
    cfg = {'drivelog': [st.drivelog, DL.METHOD_REVISION],
           'rig_health': [st.health, ref_key, DRIFT_FRAMES, RH.METHOD_REVISION], 'detection': st.detection,
           'asbuilt': 12, 'image_stats': STAT_FRAMES, 'returns': 1}.get(name)
    from .. import __version__
    return _sha({'e': name, 'cfg': cfg, 'drive': drive_key_, 'v': __version__})


def evaluate(project, st: CampaignSettings, pl: Plan, out_dir, engine=None, progress=None, cancel=None,
             detector=None) -> dict:
    """Run the evaluators on every drive done (cached per drive and evaluator). {drive id: {evaluator: result}}."""
    stop = cancel or (lambda: False)
    if engine is None and ({'drivelog', 'asbuilt'} & set(st.evaluators)):
        from .analysis import Engine
        engine = Engine()
    done = [(d, _drive_done(out_dir, d)) for d in pl.drives]
    done = [(d, rec) for d, rec in done if rec is not None]
    refs = {d['variant_uid']: (d, rec) for d, rec in done if d['role'] == 'reference'}
    out = {}
    jobs = []
    for d, rec in done:
        if d['role'] != 'drive':
            continue
        for e in st.evaluators:
            if e == 'asbuilt' and not _asbuilt_wanted(st, d):
                continue
            jobs.append((d, rec, e))
    bodies_by_var = {}
    for j, (d, rec, e) in enumerate(jobs):
        if stop():
            raise Cancelled()
        root = os.path.join(out_dir, 'drives', d['id'])
        ref = refs.get(d['variant_uid'])
        key = eval_key(st, e, rec['key'], ref[1]['key'] if (e == 'rig_health' and ref) else None)
        path = os.path.join(root, 'campaign_eval', f'{e}.json')
        cached = _read_json(path)
        if isinstance(cached, dict) and cached.get('key') == key and 'result' in cached:
            out.setdefault(d['id'], {})[e] = cached['result']
            continue
        if progress is not None:
            progress(j / max(len(jobs), 1), f'{EVALUATOR_LABEL[e]} · {d["id"]} ({j + 1} of {len(jobs)})')
        var = project.variant_by_uid(d['variant_uid']) or project.variant
        rig_doc = _read_json(os.path.join(out_dir, 'rigs', d['rig'], 'sim_rig.json'))
        t0 = time.perf_counter()
        try:
            if e == 'drivelog':
                r = _eval_drivelog(project, var, root, st, engine, stop)
            elif e == 'rig_health':
                if d['variant_uid'] not in bodies_by_var:
                    bodies_by_var[d['variant_uid']] = body_masks(
                        project, var, os.path.join(out_dir, 'drives', ref[0]['id'])) if ref else {}
                r = _eval_rig_health(project, var, root, rig_doc, bodies_by_var[d['variant_uid']], st, stop)
            elif e == 'asbuilt':
                r = _eval_asbuilt(project, var, root, engine, stop)
            elif e == 'detection':
                r = _eval_detection(project, root, st, stop, detector)
            elif e == 'image_stats':
                r = _eval_image_stats(project, var, root, stop)
            else:
                r = _eval_returns(project, var, root, stop)
        except Cancelled:
            raise
        except (FileNotFoundError, ValueError) as err:     # e.g. the detector's model missing, no ground truth
            r = {'error': f'{type(err).__name__}: {err}'}
        r['eval_seconds'] = round(time.perf_counter() - t0, 2)
        _write_json(path, {'schema': EVAL_SCHEMA, 'key': key, 'evaluator': e, 'drive': d['id'], 'result': r})
        out.setdefault(d['id'], {})[e] = r
    if progress is not None:
        progress(1.0, 'Checks done')
    return out


# =========================================================================== statistics
def z_of(conf: float) -> float:
    return NormalDist().inv_cdf(0.5 + conf / 2.0)


def wilson(k: int, n: int, conf: float = 0.95) -> tuple | None:
    """Wilson's score interval for k successes in n trials (Wilson 1927): (lo, hi); None for n = 0."""
    if n <= 0:
        return None
    z = z_of(conf)
    p = k / n
    den = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (0.0 if k == 0 else max(0.0, c - h)), (1.0 if k == n else min(1.0, c + h))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def logistic_fit(mags, k, n, floor: float = 0.0, ridge: float = 1e-3) -> dict:
    """Maximum likelihood fit of p(m) = f + (1 − f)·σ(a + b·log10 m) to k detections in n trials at each magnitude m
    (f the false-alarm floor, fixed); a tiny ridge keeps a perfectly separated data set finite. {'a', 'b', 'ok',
    'separated'}."""
    from scipy.optimize import minimize
    x = np.log10(np.asarray(mags, float))
    k = np.asarray(k, float)
    n = np.asarray(n, float)
    f = min(max(float(floor), 0.0), 0.999)
    if not len(x) or n.sum() <= 0:
        return {'a': None, 'b': None, 'ok': False, 'separated': False}
    xc = float(np.average(x, weights=np.maximum(n, 1e-9)))
    xs = max(float(np.sqrt(np.average((x - xc) ** 2, weights=np.maximum(n, 1e-9)))), 0.05)

    def nll(th):
        p = f + (1 - f) * _sigmoid(th[0] + th[1] * (x - xc) / xs)
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return -float(np.sum(k * np.log(p) + (n - k) * np.log(1 - p))) + ridge * float(th @ th) / 2

    best = None
    for b0 in (1.0, 4.0, 12.0):
        r = minimize(nll, np.array([0.0, b0]), method='BFGS')
        if best is None or r.fun < best.fun:
            best = r
    a, b = float(best.x[0]), float(best.x[1])
    phat = k / np.maximum(n, 1)
    order = np.argsort(x)
    det = phat[order] > f + 1e-9
    separated = bool(np.all(np.diff(phat[order]) >= 0) and np.all(np.isin(phat[order], (0.0, 1.0))) and
                     det.any() and not det.all())
    # back to log10 m: a' = a − b·xc/xs, b' = b/xs
    return {'a': a - b * xc / xs, 'b': b / xs, 'ok': bool(best.success) or best.fun < 1e9, 'separated': separated}


def fit_threshold(fit: dict, target: float, floor: float = 0.0) -> float | None:
    """The magnitude at which the fitted curve reaches target; None when it never does (no rise, or the floor and
    the target leave no room)."""
    if not fit or fit.get('b') is None or fit['b'] <= 0 or floor >= target:
        return None
    q = (target - floor) / (1.0 - floor)
    if not 0.0 < q < 1.0:
        return None
    x = (math.log(q / (1 - q)) - fit['a']) / fit['b']
    if not math.isfinite(x) or x > 12.0:
        return None
    return float(10 ** max(x, -12.0))


def empirical_threshold(mags, k, n, target: float) -> float | None:
    """The smallest tested magnitude from which every tested magnitude at or above it is detected at least target of
    the time; None when the largest is not."""
    rows = sorted((float(m), kk / nn) for m, kk, nn in zip(mags, k, n) if nn > 0)
    best = None
    for m, p in reversed(rows):
        if p + 1e-12 >= target:
            best = m
        else:
            break
    return best


def threshold(mags, k, n, target=0.9, floor=0.0, conf=0.95, boot=BOOT, seed=0) -> dict:
    """The smallest magnitude detected target of the time: empirical, and from the logistic fit with a stratified
    bootstrap percentile interval (each magnitude's trials resampled; a resample that never reaches the target counts
    as beyond the largest magnitude tested)."""
    mags = [float(m) for m in mags]
    k = [int(v) for v in k]
    n = [int(v) for v in n]
    emp = empirical_threshold(mags, k, n, target)
    fit = logistic_fit(mags, k, n, floor)
    m_fit = fit_threshold(fit, target, floor)
    lo_hi = None
    if boot and sum(n) and len(mags) >= 2:
        rng = np.random.default_rng(seed)
        top = max(mags)
        vals = []
        for _ in range(int(boot)):
            kb = [int(rng.binomial(nn, kk / nn)) if nn else 0 for kk, nn in zip(k, n)]
            fb = logistic_fit(mags, kb, n, floor)
            mb = fit_threshold(fb, target, floor)
            vals.append(math.inf if mb is None or mb > 10 * top else mb)
        vals = np.array(vals)
        a = (1 - conf) / 2
        big = 1e300                                   # 'never reached' sorts last without inf − inf in the quantile
        q = np.quantile(np.where(np.isfinite(vals), vals, big), [a, 1 - a])
        lo, hi = (float(x) if x < big / 2 else math.inf for x in q)
        # every size found all of the time or none of it: each resample is the data again, so the percentile interval
        # is a point and says nothing; none is given
        if not (math.isfinite(lo) and math.isfinite(hi) and abs(hi - lo) <= 1e-9 * max(abs(hi), 1.0)):
            lo_hi = [lo if math.isfinite(lo) else None, hi if math.isfinite(hi) else None]
    beyond = m_fit is not None and m_fit > max(mags)
    below = m_fit is not None and m_fit < min(mags)
    return {'empirical': emp, 'fit': m_fit, 'fit_ci': lo_hi, 'a': fit.get('a'), 'b': fit.get('b'),
            'separated': fit.get('separated', False), 'extrapolated': bool(beyond or below), 'target': target,
            'floor': floor}


# =========================================================================== observations
def _faults_injected(out_dir, did) -> list:
    rec = _read_json(os.path.join(out_dir, 'drives', did, 'campaign_drive.json')) or {}
    return rec.get('faults_injected') or []


def _truth(cond, rows) -> dict:
    """What the bridge applied for a condition: the pose increment (deg) or position change (m) in the plan's terms,
    the offset (ms), the dropped share, the frozen frames."""
    row = next((r for r in rows if r.get('tag') == cond['tag'] and r.get('kind') == cond['kind']), None)
    if row is None:
        return {}
    k = cond['kind']
    if k == 'rotation':
        return {'pose_delta_deg': row.get('pose_delta_deg'), 'angle_deg': row.get('angle_deg')}
    if k == 'translation':
        return {'position_delta_m': row.get('position_delta_m')}
    if k == 'clock_offset':
        return {'ms': row.get('ms')}
    if k == 'drop':
        n = row.get('planned_frames') or 0
        fr = row.get('dropped_frames') or row.get('frames') or []
        return {'dropped_frames': len(fr), 'planned_frames': n, 'share': len(fr) / n if n else None}
    if k == 'freeze':
        return {'frozen_frames': len(row.get('frozen_frames') or [])}
    if k == 'range_bias':
        return {'m': row.get('m')}
    return {}


def _outcomes(ev: dict, tag: str, kind: str, rounding_ms: dict) -> dict:
    """{check: (decided, flagged, statistic, estimate)} for one sensor of one drive."""
    out = {}
    dl = ev.get('drivelog')
    if isinstance(dl, dict) and 'cameras' in dl:
        if kind == 'camera':
            c = dl['cameras'].get(tag)
            if c is not None:
                obs = c.get('observable') or [False] * 3
                corr = c.get('correction_deg') or [0.0, 0.0, 0.0]
                stat = max((abs(v) for v, o in zip(c.get('delta_deg') or [], obs) if o), default=None)
                out['drivelog.extrinsics'] = (c['verdict'] in ('ok', 'rotated'), c['verdict'] == 'rotated', stat,
                                              {'delta_deg': c.get('delta_deg'), 'correction_deg': corr,
                                               'stderr_deg': c.get('stderr_deg'), 'observable': obs,
                                               'confidence': c.get('confidence')})
            s = dl['sync'].get(tag)
            if s is not None:
                own = s.get('own_ms')
                out['drivelog.sync'] = (own is not None, s.get('status') in ('warning', 'fault'),
                                        None if own is None else abs(own),
                                        {'own_ms': own, 'own_minus_rounding_ms':
                                            None if own is None else own - rounding_ms.get(tag, 0.0)})
        s = dl['streams'].get(tag)
        if s is not None:
            out['drivelog.streams'] = (True, s['status'] != 'ok', 100.0 * (s.get('drop_share') or 0.0),
                                       {'drops': s.get('drops'), 'drop_share': s.get('drop_share'),
                                        'rate_hz': s.get('rate_hz')})
        bad = [f for f in dl['findings'] if f['severity'] in ('warning', 'fault') and tag in (f.get('tags') or [])
               and f.get('section') not in NOT_FAULT_SECTIONS]
        out['drivelog.any'] = (True, bool(bad), float(len(bad)), {'sections': sorted({f['section'] for f in bad})})
    rh = ev.get('rig_health')
    if isinstance(rh, dict) and 'cameras' in rh and kind == 'camera':
        c = rh['cameras'].get(tag)
        if c is not None and c.get('frames'):
            hs = rh.get('settings') or {}
            d = c.get('drift_px')
            out['rig_health.drift'] = (d is not None, d is not None and d > hs.get('drift_warn_px', 12.0), d,
                                       {'drift_px': d})
            out['rig_health.freeze'] = (True, c['max_frozen'] >= hs.get('freeze_frames', 10), float(c['max_frozen']),
                                        {'max_frozen': c['max_frozen']})
            out['rig_health.drops'] = (True, c['dropped_share'] > hs.get('drops_warn', 0.01),
                                       100.0 * c['dropped_share'], {'dropped': c['dropped'],
                                                                    'dropped_share': c['dropped_share']})
            off = c.get('skew_ms', c.get('offset_ms'))
            out['rig_health.sync'] = (off is not None, bool(c.get('sync_flag')), None if off is None else abs(off),
                                      {'offset_ms': c.get('offset_ms'), 'skew_ms': c.get('skew_ms')})
            out['rig_health.verdict'] = (True, c['verdict'] in ('warning', 'fault'), None,
                                         {'verdict': c['verdict'], 'reasons': c.get('reasons')})
    ab = ev.get('asbuilt')
    if isinstance(ab, dict) and ab.get('cameras') and kind == 'camera':
        c = ab['cameras'].get(tag)
        if c is not None:
            v = c.get('verdict')
            out['asbuilt.verdict'] = (v not in ('not_checkable', None), v in ('does_not_hold', 'holds_with_corrections'),
                                      None, {'verdict': v, 'iou': c.get('iou'), 'outline_mean_px':
                                             c.get('outline_mean_px'), 'lidar_ypr_deg': c.get('lidar_ypr_deg'),
                                             'correction_ypr_deg': c.get('correction_ypr_deg')})
    return out


def observations(project, st, pl: Plan, evals: dict, out_dir) -> list:
    """One row per drive, sensor and check: the condition on that sensor (None when clean), whether the drive is
    wholly clean, the check's decision and statistic, its estimate, and the truth the bridge reports."""
    rows = []
    for d in pl.drives:
        if d['role'] != 'drive' or d['id'] not in evals:
            continue
        var = project.variant_by_uid(d['variant_uid']) or project.variant
        kinds = sensor_kinds(var)
        rig_doc = _read_json(os.path.join(out_dir, 'rigs', d['rig'], 'sim_rig.json')) or {}
        rounding = {c['tag']: float(c.get('capture', {}).get('offset_error_ms') or 0.0)
                    for c in rig_doc.get('cameras', [])}
        inj = _faults_injected(out_dir, d['id'])
        conds = {c['tag']: c for c in d['conditions']}
        for tag, kind in kinds.items():
            oc = _outcomes(evals[d['id']], tag, kind, rounding)
            cond = conds.get(tag)
            truth = _truth(cond, inj) if cond else {}
            for check, (decided, flagged, stat, est) in oc.items():
                rows.append({'drive': d['id'], 'seed': d['seed'], 'repeat': d['repeat'], 'variant': d['variant'],
                             'weather': d['weather'], 'scenario': d['scenario']['name'], 'tag': tag,
                             'sensor_kind': kind, 'check': check, 'clean_drive': d['clean'],
                             'kind': cond['kind'] if cond else None, 'axis': cond.get('axis') if cond else None,
                             'magnitude': cond['magnitude'] if cond else 0.0, 'sign': cond.get('sign', 1) if cond
                             else 0, 'label': cond['label'] if cond else 'clean', 'decided': bool(decided),
                             'flagged': bool(flagged), 'stat': stat, 'estimate': est, 'truth': truth,
                             'applicable': applicable(check)})
    return rows


# =========================================================================== analysis: fault detectability
def _rate(rows, conf):
    n = len(rows)
    k = sum(r['flagged'] for r in rows)
    st_ = [r['stat'] for r in rows if r['stat'] is not None]
    return {'n': n, 'k': k, 'p': k / n if n else None, 'ci': wilson(k, n, conf),
            'undecided': sum(not r['decided'] for r in rows),
            'stat_mean': float(np.mean(st_)) if st_ else None}


def detectability(obs: list, st: CampaignSettings) -> dict:
    """Detection curves, false alarms, thresholds and estimation accuracy (see the module doc)."""
    conf, target = st.confidence, st.target
    clean = [r for r in obs if r['clean_drive']]
    fa = {}
    for r in clean:
        fa.setdefault((r['check'], r['tag']), []).append(r)
    false_alarms = [dict(check=c, tag=t, **_rate(v, conf)) for (c, t), v in sorted(fa.items())]
    for f in false_alarms:
        rows = fa[(f['check'], f['tag'])]
        f['seeds_flagged'] = sorted({r['seed'] for r in rows if r['flagged']})
        # a check that cannot judge on CARLA: its flags are listed, marked, never counted as false alarms
        f['applicable'] = applicable(f['check'])
        f['not_applicable'] = NOT_APPLICABLE.get(f['check'])
    # unfaulted sensors in faulty drives: cross-talk
    xt = {}
    for r in obs:
        if not r['clean_drive'] and r['kind'] is None:
            xt.setdefault((r['check'], r['tag']), []).append(r)
    crosstalk = []
    for (c, t), v in sorted(xt.items()):
        rr = _rate(v, conf)
        base = next((f for f in false_alarms if f['check'] == c and f['tag'] == t), None)
        crosstalk.append(dict(check=c, tag=t, clean_rate=base['p'] if base else None, applicable=applicable(c),
                              not_applicable=NOT_APPLICABLE.get(c), **rr))
    # curves
    curves, thresholds = [], []
    groups = {}
    for r in obs:
        if r['kind'] is None:
            continue
        groups.setdefault((r['check'], r['tag'], r['kind'], r['axis'] or ''), []).append(r)
        if r['axis']:
            groups.setdefault((r['check'], r['tag'], r['kind'], 'any'), []).append(r)
            # the sign matters when the check has a bias of its own (the drive-log check's common pitch on CARLA)
            groups.setdefault((r['check'], r['tag'], r['kind'], 'any +' if r['sign'] > 0 else 'any −'), []).append(r)
        # and per seed (each seed is its own route through the town: the scene matters to scene-based checks)
        groups.setdefault((r['check'], r['tag'], r['kind'], f'seed {r["seed"]}'), []).append(r)
    for (check, tag, kind, axis), rows in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2],
                                                                                 kv[0][3] != 'any', kv[0][3])):
        base = next((f for f in false_alarms if f['check'] == check and f['tag'] == tag), None)
        floor = base['p'] if base and base['p'] is not None else 0.0
        mags = sorted({r['magnitude'] for r in rows})
        pts = []
        for m in mags:
            sel = [r for r in rows if r['magnitude'] == m]
            rr = _rate(sel, conf)
            st_ = [r['stat'] for r in sel if r['stat'] is not None]
            rr['stat_mean'] = float(np.mean(st_)) if st_ else None
            pts.append(dict(magnitude=m, stat_sd=float(np.std(st_)) if len(st_) > 1 else None, **rr))
        relevant = check in RELEVANT.get(kind, ())
        signed = axis.startswith('any ') or axis.startswith('seed ')
        curves.append({'check': check, 'tag': tag, 'kind': kind, 'axis': axis, 'unit': FAULT_UNIT[kind],
                       'relevant': relevant, 'applicable': applicable(check), 'false_alarm': floor,
                       'false_alarm_n': base['n'] if base else 0, 'points': pts})
        if relevant and applicable(check) and len(pts) >= 1 and not signed:
            th = threshold([p['magnitude'] for p in pts], [p['k'] for p in pts], [p['n'] for p in pts],
                           target, floor, conf, BOOT if len(pts) >= 2 else 0, seed=_sub_seed(check, tag, kind, axis)
                           % (2 ** 31))
            # the check's own statistic on the clean drives (its baseline: a plan rounding, an estimator's bias)
            thresholds.append(dict(check=check, tag=tag, kind=kind, axis=axis, unit=FAULT_UNIT[kind],
                                   tested=[p['magnitude'] for p in pts],
                                   clean_stat=base.get('stat_mean') if base else None,
                                   stat_unit=CHECKS[check][2], **th))
    accuracy = _accuracy(obs, conf)
    return {'false_alarms': false_alarms, 'crosstalk': crosstalk, 'curves': curves, 'thresholds': thresholds,
            'accuracy': accuracy, 'target': target, 'confidence': conf}


def _err_stats(errs) -> dict:
    e = np.asarray([x for x in errs if x is not None and math.isfinite(x)], float)
    if not len(e):
        return {'n': 0, 'bias': None, 'rms': None, 'sd': None, 'max_abs': None}
    return {'n': int(len(e)), 'bias': float(e.mean()), 'rms': float(np.sqrt((e ** 2).mean())),
            'sd': float(e.std(ddof=1)) if len(e) > 1 else None, 'max_abs': float(np.abs(e).max())}


def _accuracy(obs, conf) -> list:
    """Estimate − injected per check, sensor, kind and axis: absolute, and paired with the clean drive of the same
    seed (the estimator's response, its scene-dependent bias taken out)."""
    clean = {(r['seed'], r['variant'], r['weather'], r['scenario'], r['tag'], r['check']): r
             for r in obs if r['clean_drive']}
    groups = {}
    for r in obs:
        if r['kind'] is None:
            continue
        groups.setdefault((r['check'], r['tag'], r['kind'], r['axis'] or ''), []).append(r)
    out = []
    for (check, tag, kind, axis), rows in sorted(groups.items()):
        abs_e, pair_e, per_mag = [], [], {}
        for r in rows:
            c0 = clean.get((r['seed'], r['variant'], r['weather'], r['scenario'], r['tag'], r['check']))
            e = p = None
            if check == 'drivelog.extrinsics' and kind == 'rotation':
                tr = (r['truth'] or {}).get('pose_delta_deg')
                est = (r['estimate'] or {}).get('delta_deg')
                if tr and est:
                    i = AXES['rotation'].index(axis)
                    e = est[i] - tr[i]
                    e0 = ((c0 or {}).get('estimate') or {}).get('delta_deg')
                    p = (est[i] - e0[i]) - tr[i] if e0 else None
            elif check == 'drivelog.sync' and kind == 'clock_offset':
                tr = (r['truth'] or {}).get('ms')
                est = (r['estimate'] or {}).get('own_minus_rounding_ms')
                if tr is not None and est is not None:
                    e = est - tr
                    e0 = ((c0 or {}).get('estimate') or {}).get('own_ms')
                    own = (r['estimate'] or {}).get('own_ms')
                    p = (own - e0) - tr if (e0 is not None and own is not None) else None
            elif check in ('drivelog.streams', 'rig_health.drops') and kind == 'drop':
                tr = (r['truth'] or {}).get('share')
                est = (r['estimate'] or {}).get('drop_share' if check == 'drivelog.streams' else 'dropped_share')
                if tr is not None and est is not None:
                    e = 100.0 * (est - tr)
            elif check == 'rig_health.freeze' and kind == 'freeze':
                tr = (r['truth'] or {}).get('frozen_frames')
                est = (r['estimate'] or {}).get('max_frozen')
                if tr is not None and est is not None:
                    e = float(est - tr) if tr else None     # repeats of the held frame, as rig health counts them
            if e is None:
                continue
            abs_e.append(e)
            pair_e.append(p)
            per_mag.setdefault(r['magnitude'], []).append(e)
        if not abs_e:
            continue
        unit = {'drivelog.extrinsics': '°', 'drivelog.sync': 'ms', 'rig_health.freeze': 'frames'}.get(check, 'pt')
        out.append({'check': check, 'tag': tag, 'kind': kind, 'axis': axis, 'unit': unit,
                    'absolute': _err_stats(abs_e), 'paired': _err_stats(pair_e),
                    'per_magnitude': [dict(magnitude=m, **_err_stats(v)) for m, v in sorted(per_mag.items())]})
    return out


# =========================================================================== analysis: weather
def _r50(bins, edges) -> float | None:
    """The range (m) at which recall falls through 50 %: linear between the bins' centres (the last bin's centre
    taken at its lower edge × 1.25); None when it never falls below or never reaches 50 %."""
    centres = []
    for i in range(len(edges) + 1):
        lo = 0.0 if i == 0 else edges[i - 1]
        hi = edges[i] if i < len(edges) else edges[-1] * 1.5
        centres.append((lo + hi) / 2.0)
    pts = [(c, b) for c, b in zip(centres, bins) if b is not None]
    if not pts or pts[0][1] < 0.5:
        return None
    for (c0, r0), (c1, r1) in zip(pts, pts[1:]):
        if r0 >= 0.5 > r1:
            return c0 + (r0 - 0.5) / (r0 - r1) * (c1 - c0)
    return None


def vantage_weather(project, var, conditions) -> dict:
    """Vantage's own prediction per condition (core.perception with core.weather): per class the far range under it
    over the far range in clear, median over every direction and straight ahead; per camera too."""
    from . import perception as PC
    from . import weather as WX
    from .analysis import Engine
    ids = [c.split(':', 1)[1] for c in conditions if c.startswith('vantage:') and c.split(':', 1)[1] in WX.IDS]
    if not ids:
        return {}
    import copy
    proj = copy.copy(project)                       # the variant made active on a copy: the caller's stays as it is
    proj.active_uid = var.uid
    res = PC.analyse(proj, Engine(), None, weather='clear', compare=[i for i in ids if i != 'clear'])
    if res is None:
        return {}
    far0 = res.far                                  # (K, A) clear
    out = {}
    names = [c.name for c in res.classes]
    for i in ids:
        if i == 'clear':
            out[i] = {'label': WX.PRESET[i].label(), 'classes': {n: {'factor_median': 1.0, 'factor_ahead': 1.0,
                                                                     'far_median_m': float(np.median(far0[k])),
                                                                     'far_ahead_m': float(far0[k, 0])}
                                                                 for k, n in enumerate(names)}}
            continue
        row = next((c for c in res.weather['compare'] if c['id'] == i), None)
        if row is None:
            continue
        far = row['far']
        cls = {}
        for k, n in enumerate(names):
            ok = far0[k] > 0
            fac = far[k][ok] / far0[k][ok] if ok.any() else np.array([])
            cls[n] = {'factor_median': float(np.median(fac)) if len(fac) else None,
                      'factor_ahead': float(far[k, 0] / far0[k, 0]) if far0[k, 0] > 0 else None,
                      'far_median_m': float(np.median(far[k])), 'far_ahead_m': float(far[k, 0])}
        out[i] = {'label': row['label'], 'classes': cls, 'per_camera': row['per_camera']}
    return out


VANTAGE_CLASS = {'car': 'Car', 'vehicle': 'Car', 'truck': 'Car', 'bus': 'Car', 'pedestrian': 'Pedestrian',
                 'cyclist': 'Cyclist', 'motorcyclist': 'Cyclist'}


def weather_analysis(project, st, pl: Plan, evals: dict, out_dir) -> dict | None:
    """Per weather: detection recall per class and range bin (counts pooled over seeds), R50 and its ratio to clear,
    Vantage's predicted range factor, image statistics, lidar and radar returns."""
    weathers = list(dict.fromkeys(d['weather'] for d in pl.drives if d['role'] == 'drive'))
    if len(weathers) < 2:
        return None
    rows = []
    edges = None
    for w in weathers:
        ds = [d for d in pl.drives if d['role'] == 'drive' and d['weather'] == w and d['id'] in evals and d['clean']]
        if not ds:
            continue
        cnt, rng_txt = {}, {}
        img, ret = {}, {}
        for d in ds:
            ev = evals[d['id']]
            det = ev.get('detection')
            if isinstance(det, dict) and 'rows' in det:
                edges = det.get('range_edges') or edges
                for r in det['rows']:
                    if r['camera'] != 'all':
                        continue
                    c = cnt.setdefault((r['class'], r['range_bin']), [0, 0])
                    c[0] += r['detected']
                    c[1] += r['n_gt']
                for r in det['ranges']:
                    if r['camera'] == 'all':
                        rng_txt.setdefault(r['class'], []).append(r.get('text'))
            for tag, s in ((ev.get('image_stats') or {}).get('cameras') or {}).items():
                img.setdefault(tag, []).append(s)
            for tag, s in ((ev.get('returns') or {}).get('sensors') or {}).items():
                ret.setdefault(tag, []).append(s)
        classes = sorted({c for c, _ in cnt})
        per_class = {}
        for c in classes:
            tot = cnt.get((c, 'all'), [0, 0])
            bins = []
            labels = []
            from .detection_eval import bin_labels
            for lab in bin_labels(edges or [10.0, 20.0, 40.0]):
                k_, n_ = cnt.get((c, lab), [0, 0])
                labels.append(lab)
                bins.append({'bin': lab, 'n': n_, 'k': k_, 'recall': k_ / n_ if n_ else None,
                             'ci': wilson(k_, n_, st.confidence)})
            min_gt = 5
            r50 = _r50([b['recall'] if b['n'] >= min_gt else None for b in bins], edges or [10.0, 20.0, 40.0])
            per_class[c] = {'n': tot[1], 'k': tot[0], 'recall': tot[0] / tot[1] if tot[1] else None,
                            'ci': wilson(tot[0], tot[1], st.confidence), 'bins': bins, 'r50_m': r50,
                            'detection_range': rng_txt.get(c, [None])[0]}
        imgs = {t: {k: float(np.mean([s[k] for s in v if s.get(k) is not None]))
                    for k in ('mean', 'contrast', 'noise', 'sharpness', 'clip_bright', 'clip_dark')
                    if any(s.get(k) is not None for s in v)} for t, v in img.items()}
        rets = {t: {'kind': v[0]['kind'], 'points_mean': float(np.mean([s['points_mean'] for s in v])),
                    'intensity_mean': (float(np.mean([s['intensity_mean'] for s in v if s.get('intensity_mean')
                                                      is not None])) if any(s.get('intensity_mean') is not None
                                                                            for s in v) else None)}
                for t, v in ret.items()}
        rows.append({'weather': w, 'drives': [d['id'] for d in ds], 'classes': per_class, 'images': imgs,
                     'returns': rets})
    if not rows:
        return None
    ref = next((r for r in rows if r['weather'] in ('vantage:clear', 'ClearNoon')), rows[0])
    for r in rows:
        for c, v in r['classes'].items():
            r0 = (ref['classes'].get(c) or {}).get('r50_m')
            v['r50_factor'] = (v['r50_m'] / r0) if (v.get('r50_m') and r0) else None
            rec0 = (ref['classes'].get(c) or {}).get('recall')
            v['recall_ratio'] = (v['recall'] / rec0) if (v.get('recall') is not None and rec0) else None
        for t, s in r['returns'].items():
            p0 = (ref['returns'].get(t) or {}).get('points_mean')
            s['points_ratio'] = s['points_mean'] / p0 if p0 else None
        for t, s in r['images'].items():
            m0 = (ref['images'].get(t) or {}).get('mean')
            s['brightness_ratio'] = s['mean'] / m0 if (m0 and s.get('mean') is not None) else None
    var = _variants(project, st)[0]
    try:
        pred = vantage_weather(project, var, [r['weather'] for r in rows])
    except Exception as e:                       # noqa: BLE001 — the comparison is optional; say why it is missing
        pred = {'error': f'{type(e).__name__}: {e}'}
    for r in rows:
        wid = r['weather'].split(':', 1)[1] if r['weather'].startswith('vantage:') else None
        p = pred.get(wid) if (wid and isinstance(pred, dict)) else None
        r['vantage'] = p
        for c, v in r['classes'].items():
            vc = VANTAGE_CLASS.get(c)
            pc = ((p or {}).get('classes') or {}).get(vc) if vc else None
            v['vantage_factor'] = pc.get('factor_median') if pc else None
            v['vantage_factor_ahead'] = pc.get('factor_ahead') if pc else None
    notes = ['CARLA renders fog, rain and night for the cameras only: its lidar and radar returns are expected not to '
             'change with the weather (the same seed drives the same traffic).',
             'R50 is the range where recall falls through 50 %, interpolated between the range bins\' centres (bins '
             'with fewer than 5 objects left out); the CARLA factor is R50 under the weather over R50 in clear. '
             'Vantage\'s factor is its perception range\'s far range under the condition over clear, median over '
             'every direction.']
    return {'reference': ref['weather'], 'rows': rows, 'range_edges': edges, 'notes': notes,
            'prediction_error': pred.get('error') if isinstance(pred, dict) else None}


# =========================================================================== analysis: rigs
def rig_analysis(project, st, pl: Plan, evals: dict, out_dir, cancel=None, detector=None) -> dict | None:
    """Detection quality of the variants compared (core.detection_eval over each variant's clean drives, the first
    variant the reference; paired when they share scenario and seed), and their drive-log summaries."""
    variants = list(dict.fromkeys(d['variant_uid'] for d in pl.drives if d['role'] == 'drive'))
    if len(variants) < 2:
        return None
    per = []
    for uid in variants:
        ds = [d for d in pl.drives if d['role'] == 'drive' and d['variant_uid'] == uid and d['clean'] and
              d['id'] in evals]
        per.append((uid, ds))
    out = {'variants': [], 'comparison': None}
    for uid, ds in per:
        v = project.variant_by_uid(uid)
        row = {'variant': v.name if v else uid, 'drives': [d['id'] for d in ds]}
        dl = [evals[d['id']].get('drivelog') for d in ds if isinstance(evals[d['id']].get('drivelog'), dict)]
        if dl:
            row['drivelog'] = {tag: [c['verdict'] for c in (x.get('cameras') or {}).values()]
                               for x in dl[:1] for tag in ['verdicts']}
            row['rotated'] = sum(c.get('verdict') == 'rotated' for x in dl for c in (x.get('cameras') or {}).values())
        out['variants'].append(row)
    if 'detection' in st.evaluators and all(ds for _, ds in per):
        from . import detection_eval as DE
        base = DE.settings_of(project).to_dict()
        base.update(st.detection or {})
        recs = []
        for k in range(min(len(ds) for _, ds in per)):
            for uid, ds in per:
                recs.append(os.path.join(out_dir, 'drives', ds[k]['id']))
        base['recordings'] = recs[:len(per)] if len(recs) >= len(per) else recs
        try:
            res = DE.run(None, settings=DE.DetectionSettings.from_dict(base), cancel=cancel, detector=detector)
        except (FileNotFoundError, ValueError) as e:
            out['error'] = f'{type(e).__name__}: {e}'
            res = None
        if res is not None:
            out['comparison'] = DE.summary(res)
            out['comparison_rows'] = [{k: v for k, v in c.items() if not isinstance(v, np.ndarray)}
                                      for c in res.comparisons]
    return out


# =========================================================================== the whole run
def run(project, engine=None, settings: CampaignSettings | None = None, variant=None, progress=None, cancel=None,
        runner=None, python=None, phases=('drive', 'evaluate', 'analyse'), detector=None) -> CampaignResult | None:
    """Run the campaign: drive what is not driven yet, evaluate what is not evaluated, analyse everything there is.
    phases: any of 'drive', 'evaluate', 'analyse' (a server's time can be kept to the drives). None when cancelled
    (everything finished so far is kept: run again to resume). variant: the variant when the settings name none."""
    t0 = time.perf_counter()
    st = settings or settings_of(project)
    if variant is not None and not st.variants:
        st = CampaignSettings.from_dict(dict(st.to_dict(), variants=[variant.uid]))
    pl = plan(project, st)
    out_dir = campaign_dir(project, st)
    os.makedirs(out_dir, exist_ok=True)
    _write_json(os.path.join(out_dir, 'campaign_plan.json'), {'schema': SCHEMA, 'settings': st.to_dict(),
                                                              'drives': pl.drives, 'notes': pl.notes,
                                                              'problems': pl.problems})
    stop = cancel or (lambda: False)
    w = {'drive': 0.7 if 'drive' in phases else 0.0, 'evaluate': 0.25 if 'evaluate' in phases else 0.0}
    tot = max(sum(w.values()) + 0.05, 1e-9)

    def sub(lo, span):
        return lambda f, text='': progress((lo + span * f) / tot, text) if progress is not None else None
    states = {}
    try:
        if 'drive' in phases:
            states = run_drives(project, st, pl, out_dir, sub(0.0, w['drive']), stop, runner, python)
        for d in pl.drives:
            if d['id'] not in states:
                rec = _drive_done(out_dir, d)
                states[d['id']] = DriveState(d['id'], 'reused' if rec else 'pending',
                                             float((rec or {}).get('wall_s', 0.0)))
        evals = {}
        if 'evaluate' in phases:
            evals = evaluate(project, st, pl, out_dir, engine, sub(w['drive'], w['evaluate']), stop, detector)
        elif 'analyse' in phases:
            evals = cached_evaluations(project, st, pl, out_dir)
    except Cancelled:
        return None
    if stop():
        return None
    obs = observations(project, st, pl, evals, out_dir) if 'analyse' in phases else []
    det = detectability(obs, st) if obs else {'false_alarms': [], 'crosstalk': [], 'curves': [], 'thresholds': [],
                                              'accuracy': [], 'target': st.target, 'confidence': st.confidence}
    wx = weather_analysis(project, st, pl, evals, out_dir) if 'analyse' in phases else None
    rg = rig_analysis(project, st, pl, evals, out_dir, stop, detector) if 'analyse' in phases else None
    notes = list(pl.notes) + list(pl.problems)
    failed = [s for s in states.values() if s.status == 'failed']
    if failed:
        notes.append(f'{len(failed)} drive{"s" * (len(failed) != 1)} failed: ' +
                     '; '.join(f'{s.id}: {s.error}' for s in failed[:5]))
    pending = [s for s in states.values() if s.status == 'pending']
    if pending:
        notes.append(f'{len(pending)} drive{"s" * (len(pending) != 1)} not driven yet: the figures rest on the others.')
    if progress is not None:
        progress(1.0, 'Done')
    var = _variants(project, st)[0]
    res = CampaignResult(st, out_dir, pl, states, evals, obs, det, wx, rg, estimate(project, st, pl, out_dir), notes,
                         time.perf_counter() - t0, not failed and not pending, var.name)
    return res


def cached_evaluations(project, st, pl: Plan, out_dir) -> dict:
    """The evaluations already on disk whose keys match (nothing is computed)."""
    out = {}
    refs = {d['variant_uid']: _drive_done(out_dir, d) for d in pl.drives if d['role'] == 'reference'}
    for d in pl.drives:
        if d['role'] != 'drive':
            continue
        rec = _drive_done(out_dir, d)
        if rec is None:
            continue
        for e in st.evaluators:
            ref = refs.get(d['variant_uid'])
            key = eval_key(st, e, rec['key'], ref['key'] if (e == 'rig_health' and ref) else None)
            c = _read_json(os.path.join(out_dir, 'drives', d['id'], 'campaign_eval', f'{e}.json'))
            if isinstance(c, dict) and c.get('key') == key:
                out.setdefault(d['id'], {})[e] = c['result']
    return out


# =========================================================================== output
def _r(v, d=4):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return round(f, d) + 0.0 if math.isfinite(f) else None


def headline(res: CampaignResult) -> list:
    """The campaign's findings in plain words, most important first."""
    out = []
    det = res.detectability
    for t in det['thresholds']:
        if t['axis'] not in ('', 'any'):
            continue
        what = f'{t["kind"].replace("_", " ")}'
        lab = CHECKS[t['check']][0]
        u = t['unit']
        if t['empirical'] is not None:
            out.append(f'{lab} finds a {what} of {t["tag"]} from {t["empirical"]:g} {u} at least '
                       f'{t["target"] * 100:.0f} % of the time' + (
                           f' (fit: {t["fit"]:.3g} {u})' if t['fit'] is not None and not t['extrapolated'] else '') + '.')
        else:
            out.append(f'{lab} does not reach {t["target"] * 100:.0f} % for a {what} of {t["tag"]} up to '
                       f'{max(t["tested"]):g} {u}.')
    for f in det['false_alarms']:
        if f['check'] in ('drivelog.extrinsics', 'drivelog.sync', 'rig_health.drift') and f['k']:
            out.append(f'{CHECKS[f["check"]][0]} flags {f["tag"]} on {f["k"]} of {f["n"]} clean drives.')
    na = {}
    for f in det['false_alarms']:
        if not f.get('applicable', True) and f['k']:
            na.setdefault(f['check'], []).append(f)
    for c, fs in na.items():
        flags = ', '.join(f'{f["tag"]} {f["k"]} of {f["n"]}' for f in fs)
        out.append(f'{CHECKS[c][0]}: not applicable on CARLA (another car body); its flags on clean drives '
                   f'({flags}) are listed apart, not counted as false alarms.')
    if res.weather:
        for r in res.weather['rows']:
            c = r['classes'].get('car') or r['classes'].get('vehicle')
            if c and c.get('recall') is not None:
                out.append(f'{r["weather"]}: car recall {c["recall"] * 100:.0f} %' +
                           (f', R50 {c["r50_m"]:.0f} m' if c.get('r50_m') else ''))
    return out


def summary(res: CampaignResult) -> dict:
    """Compact figures (the assistant, the batch log): no paths, no arrays."""
    det = res.detectability
    return {
        'campaign': res.settings.name, 'template': res.settings.template, 'variant': res.variant,
        'drives': len(res.plan.drives), 'complete': res.complete,
        'driven': sum(s.status in ('ok', 'reused') for s in res.states.values()),
        'failed': sum(s.status == 'failed' for s in res.states.values()),
        'thresholds': [{'check': t['check'], 'sensor': t['tag'], 'fault': t['kind'], 'axis': t['axis'] or None,
                        'unit': t['unit'], 'empirical': _r(t['empirical']), 'fit': _r(t['fit']),
                        'fit_ci': [_r(x) for x in t['fit_ci']] if t['fit_ci'] else None,
                        'extrapolated': t['extrapolated']} for t in det['thresholds']
                       if t['axis'] in ('', 'any')],
        'false_alarms': [{'check': f['check'], 'sensor': f['tag'], 'flagged': f['k'], 'clean_drives': f['n']}
                         for f in det['false_alarms'] if f['k'] and f.get('applicable', True)],
        # flags of checks that cannot judge on CARLA: listed apart, never false alarms (NOT_APPLICABLE says why)
        'not_applicable_on_carla': [{'check': f['check'], 'sensor': f['tag'], 'flagged': f['k'],
                                     'clean_drives': f['n']} for f in det['false_alarms']
                                    if f['k'] and not f.get('applicable', True)],
        'not_applicable_why': {c: NOT_APPLICABLE[c] for c in sorted({f['check'] for f in det['false_alarms']
                                                                      if not f.get('applicable', True)})},
        'accuracy': [{'check': a['check'], 'sensor': a['tag'], 'fault': a['kind'], 'axis': a['axis'] or None,
                      'unit': a['unit'], 'bias': _r(a['absolute']['bias']), 'rms': _r(a['absolute']['rms']),
                      'paired_rms': _r(a['paired']['rms']), 'n': a['absolute']['n']} for a in det['accuracy']],
        'weather': ([{'weather': r['weather'], 'classes': {c: {'recall': _r(v['recall'], 3), 'r50_m': _r(v['r50_m'], 1),
                                                               'r50_factor': _r(v.get('r50_factor'), 3),
                                                               'vantage_factor': _r(v.get('vantage_factor'), 3)}
                                                           for c, v in r['classes'].items()}}
                     for r in res.weather['rows']] if res.weather else None),
        'headline': headline(res)[:12], 'cost': res.cost.get('text'), 'notes': res.notes[:8],
        'assumptions': ASSUMPTIONS}


def to_doc(res: CampaignResult, project=None) -> dict:
    st = res.settings
    drives = []
    for d in res.plan.drives:
        s = res.states.get(d['id'])
        drives.append({'id': d['id'], 'role': d['role'], 'variant': d['variant'], 'weather': d['weather'],
                       'scenario': d['scenario'], 'seed': d['seed'], 'clean': d['clean'], 'ground_truth': d['gt'],
                       'conditions': [c['label'] for c in d['conditions']], 'faults': d['faults'],
                       'status': s.status if s else 'pending', 'wall_s': _r(s.wall_s, 2) if s else None,
                       'error': (s.error or None) if s else None,
                       'evaluations': sorted((res.evaluations.get(d['id']) or {}).keys())})
    return {
        'schema': SCHEMA, 'project': getattr(project, 'name', ''), 'variant': res.variant,
        'frame': FRAME,
        'units': ('rotations in degrees, translations and range biases in centimetres, clock offsets in ms, drops in per '
                  'cent of a stream\'s frames, freezes in seconds; detection probabilities and recalls as shares 0–1 '
                  'with Wilson score intervals; ranges in metres; drift in sensor pixels'),
        'settings': st.to_dict(), 'campaign_folder': os.path.basename(res.out_dir), 'complete': res.complete,
        'assumptions': list(ASSUMPTIONS), 'references': list(REFERENCES), 'checks': {k: v[0] for k, v in CHECKS.items()},
        'relevant_checks': {k: list(v) for k, v in RELEVANT.items()},
        'not_applicable_on_carla': dict(NOT_APPLICABLE),
        'drives': drives, 'cost': res.cost, 'headline': headline(res),
        'detectability': res.detectability, 'weather': res.weather, 'rigs': res.rigs,
        'observations': res.observations, 'notes': res.notes, 'seconds': _r(res.seconds, 2)}


def _csv(header, rows) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(header)
    for r in rows:
        w.writerow(['' if v is None else (f'{v:.6g}' if isinstance(v, float) else v) for v in r])
    return buf.getvalue().encode('utf-8')


def to_csv(res: CampaignResult) -> dict:
    """{file name: bytes} of the tables."""
    det = res.detectability
    out = {}
    out['campaign_observations.csv'] = _csv(
        ('drive', 'seed', 'variant', 'weather', 'scenario', 'sensor', 'check', 'clean_drive', 'fault', 'axis',
         'magnitude', 'sign', 'decided', 'flagged', 'statistic', 'estimate', 'truth', 'applicable_on_carla'),
        [(r['drive'], r['seed'], r['variant'], r['weather'], r['scenario'], r['tag'], r['check'], r['clean_drive'],
          r['kind'] or '', r['axis'] or '', r['magnitude'], r['sign'], r['decided'], r['flagged'], r['stat'],
          json.dumps(r['estimate'], default=_jsonable), json.dumps(r['truth'], default=_jsonable),
          r.get('applicable', applicable(r['check']))) for r in res.observations])
    out['campaign_detection_curves.csv'] = _csv(
        ('check', 'sensor', 'fault', 'axis', 'unit', 'magnitude', 'trials', 'detected', 'probability', 'ci_lo',
         'ci_hi', 'undecided', 'statistic_mean', 'statistic_sd', 'false_alarm_rate', 'relevant',
         'applicable_on_carla'),
        [(c['check'], c['tag'], c['kind'], c['axis'], c['unit'], p['magnitude'], p['n'], p['k'], p['p'],
          p['ci'][0] if p['ci'] else None, p['ci'][1] if p['ci'] else None, p['undecided'], p['stat_mean'],
          p['stat_sd'], c['false_alarm'], c['relevant'], c.get('applicable', True))
         for c in det['curves'] for p in c['points']])
    out['campaign_thresholds.csv'] = _csv(
        ('check', 'sensor', 'fault', 'axis', 'unit', 'target', 'false_alarm_floor', 'empirical', 'fit', 'fit_ci_lo',
         'fit_ci_hi', 'extrapolated', 'separated', 'tested', 'clean_statistic', 'statistic_unit'),
        [(t['check'], t['tag'], t['kind'], t['axis'], t['unit'], t['target'], t['floor'], t['empirical'], t['fit'],
          (t['fit_ci'] or [None, None])[0], (t['fit_ci'] or [None, None])[1], t['extrapolated'], t['separated'],
          ' '.join(f'{m:g}' for m in t['tested']), t.get('clean_stat'), t.get('stat_unit'))
         for t in det['thresholds']])
    out['campaign_false_alarms.csv'] = _csv(
        ('check', 'sensor', 'set', 'drives', 'flagged', 'rate', 'ci_lo', 'ci_hi', 'undecided', 'applicable_on_carla',
         'note'),
        [(f['check'], f['tag'], 'clean drives', f['n'], f['k'], f['p'], (f['ci'] or [None, None])[0],
          (f['ci'] or [None, None])[1], f['undecided'], f.get('applicable', True), f.get('not_applicable') or '')
         for f in det['false_alarms']] +
        [(f['check'], f['tag'], 'unfaulted in faulty drives', f['n'], f['k'], f['p'], (f['ci'] or [None, None])[0],
          (f['ci'] or [None, None])[1], f['undecided'], f.get('applicable', True), f.get('not_applicable') or '')
         for f in det['crosstalk']])
    out['campaign_accuracy.csv'] = _csv(
        ('check', 'sensor', 'fault', 'axis', 'unit', 'n', 'bias', 'rms', 'sd', 'max_abs', 'paired_n', 'paired_bias',
         'paired_rms'),
        [(a['check'], a['tag'], a['kind'], a['axis'], a['unit'], a['absolute']['n'], a['absolute']['bias'],
          a['absolute']['rms'], a['absolute']['sd'], a['absolute']['max_abs'], a['paired']['n'], a['paired']['bias'],
          a['paired']['rms']) for a in det['accuracy']])
    out['campaign_drives.csv'] = _csv(
        ('drive', 'role', 'variant', 'weather', 'scenario', 'seed', 'clean', 'ground_truth', 'conditions', 'status',
         'wall_s'),
        [(d['id'], d['role'], d['variant'], d['weather'], d['scenario']['name'], d['seed'], d['clean'], d['gt'],
          '; '.join(c['label'] for c in d['conditions']), (res.states.get(d['id']) or DriveState('', 'pending')).status,
          (res.states.get(d['id']) or DriveState('', 'pending')).wall_s) for d in res.plan.drives])
    if res.weather:
        rows = []
        for r in res.weather['rows']:
            for c, v in r['classes'].items():
                for b in v['bins'] + [{'bin': 'all', 'n': v['n'], 'k': v['k'], 'recall': v['recall'], 'ci': v['ci']}]:
                    rows.append((r['weather'], c, b['bin'], b['n'], b['k'], b['recall'], (b['ci'] or [None, None])[0],
                                 (b['ci'] or [None, None])[1], v.get('r50_m') if b['bin'] == 'all' else None,
                                 v.get('r50_factor') if b['bin'] == 'all' else None,
                                 v.get('vantage_factor') if b['bin'] == 'all' else None))
        out['campaign_weather.csv'] = _csv(('weather', 'class', 'range_bin', 'objects', 'detected', 'recall', 'ci_lo',
                                            'ci_hi', 'r50_m', 'r50_factor_vs_clear', 'vantage_range_factor'), rows)
        out['campaign_weather_sensors.csv'] = _csv(
            ('weather', 'sensor', 'what', 'value', 'ratio_to_reference'),
            [(r['weather'], t, k, s.get(k), s.get('brightness_ratio') if k == 'mean' else None)
             for r in res.weather['rows'] for t, s in r['images'].items() for k in ('mean', 'contrast', 'noise',
                                                                                    'sharpness')] +
            [(r['weather'], t, 'points_mean', s['points_mean'], s.get('points_ratio'))
             for r in res.weather['rows'] for t, s in r['returns'].items()])
    return out


# --------------------------------------------------------------------------- figures (PNG, OpenCV: no Qt needed)
_BG, _INK, _INK2, _MUTED, _HAIR = (255, 255, 255), (30, 30, 30), (70, 70, 70), (130, 130, 130), (225, 225, 225)
_SERIES = ((180, 110, 30), (40, 120, 220), (60, 160, 60), (40, 40, 200), (150, 80, 160), (20, 160, 200),
           (120, 120, 120))


def _text(img, t, x, y, col=_INK, s=0.45, right=False, centre=False):
    import cv2
    from .pngtext import ascii_text
    t = ascii_text(t)
    (tw, _), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, s, 1)
    x = x - tw if right else (x - tw / 2 if centre else x)
    cv2.putText(img, t, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, s, col, 1, cv2.LINE_AA)


def curves_png(res: CampaignResult, relevant_only=True) -> bytes | None:
    """Detection probability against magnitude, one panel per check and sensor (axes pooled), the false-alarm rate
    as a dashed line, the fit, the target; PNG."""
    import cv2
    det = res.detectability
    cs = [c for c in det['curves'] if c['axis'] in ('', 'any') and (c['relevant'] or not relevant_only)]
    if not cs:
        return None
    cols = 3
    rows_n = math.ceil(len(cs) / cols)
    pw, ph = 380, 260
    W, H = pw * cols, ph * rows_n + 40
    img = np.full((H, W, 3), _BG, np.uint8)
    _text(img, f'Detection probability against fault size · {res.settings.name} · Wilson {det["confidence"] * 100:.0f} % '
               'intervals, dashed: false alarms on clean drives', 12, 24, s=0.5)
    for i, c in enumerate(cs):
        ox, oy = (i % cols) * pw, 40 + (i // cols) * ph
        x0, y0, x1, y1 = ox + 50, oy + 30, ox + pw - 16, oy + ph - 40
        _text(img, f'{c["tag"]} {c["kind"].replace("_", " ")} - {CHECKS[c["check"]][0]}', x0 - 40, oy + 16, _INK, 0.4)
        for t in (0, 0.25, 0.5, 0.75, 1.0):
            y = y1 - t * (y1 - y0)
            cv2.line(img, (x0, int(y)), (x1, int(y)), _HAIR if t else _INK2, 1)
            _text(img, f'{t * 100:.0f}%', x0 - 6, y + 4, _MUTED, 0.33, right=True)
        mags = [p['magnitude'] for p in c['points']]
        lo, hi = math.log10(min(mags)), math.log10(max(mags))
        if hi - lo < 1e-9:
            lo, hi = lo - 0.5, hi + 0.5
        pad = 0.08 * (hi - lo)
        lo, hi = lo - pad, hi + pad

        def X(m):
            return x0 + (math.log10(m) - lo) / (hi - lo) * (x1 - x0)

        def Y(p):
            return y1 - p * (y1 - y0)
        for m in mags:
            _text(img, f'{m:g}', X(m), y1 + 16, _MUTED, 0.33, centre=True)
        _text(img, c['unit'], x1, y1 + 30, _MUTED, 0.33, right=True)
        ty = Y(det['target'])
        for xx in range(int(x0), int(x1), 8):
            cv2.line(img, (xx, int(ty)), (min(xx + 3, int(x1)), int(ty)), (60, 60, 200), 1)
        fy = Y(c['false_alarm'] or 0.0)
        for xx in range(int(x0), int(x1), 10):
            cv2.line(img, (xx, int(fy)), (min(xx + 5, int(x1)), int(fy)), _INK2, 1)
        th = next((t for t in det['thresholds'] if t['check'] == c['check'] and t['tag'] == c['tag'] and
                   t['kind'] == c['kind'] and t['axis'] == c['axis']), None)
        if th and th.get('b') is not None and th['b'] > 0:
            pts = []
            for u in np.linspace(lo, hi, 80):
                p = th['floor'] + (1 - th['floor']) * float(_sigmoid(th['a'] + th['b'] * u))
                pts.append((int(x0 + (u - lo) / (hi - lo) * (x1 - x0)), int(Y(p))))
            for a, b in zip(pts, pts[1:]):
                cv2.line(img, a, b, (170, 170, 170), 1, cv2.LINE_AA)
        col = _SERIES[0]
        prev = None
        for p in c['points']:
            if p['p'] is None:
                continue
            x = int(X(p['magnitude']))
            if p['ci']:
                cv2.line(img, (x, int(Y(p['ci'][0]))), (x, int(Y(p['ci'][1]))), col, 1, cv2.LINE_AA)
            pt = (x, int(Y(p['p'])))
            if prev is not None:
                cv2.line(img, prev, pt, col, 2, cv2.LINE_AA)
            cv2.circle(img, pt, 3, col, -1, cv2.LINE_AA)
            _text(img, f'{p["k"]}/{p["n"]}', x + 4, pt[1] - 6, _MUTED, 0.3)
            prev = pt
        if th and th.get('empirical') is not None:
            x = int(X(th['empirical']))
            cv2.line(img, (x, int(y0)), (x, int(y1)), (60, 60, 200), 1)
    ok, buf = cv2.imencode('.png', img)
    return buf.tobytes() if ok else None


def weather_png(res: CampaignResult) -> bytes | None:
    """Recall per class against range bin, one line per weather; and CARLA's R50 factor against Vantage's predicted
    range factor; PNG."""
    import cv2
    wx = res.weather
    if not wx:
        return None
    classes = [c for c in ('car', 'pedestrian', 'truck', 'cyclist', 'bus', 'traffic_light')
               if any(c in r['classes'] and r['classes'][c]['n'] for r in wx['rows'])]
    if not classes:
        classes = sorted({c for r in wx['rows'] for c in r['classes']})[:3]
    panels = classes[:3] + ['factor']
    pw, ph = 380, 270
    W, H = pw * len(panels), ph + 70
    img = np.full((H, W, 3), _BG, np.uint8)
    _text(img, f'Weather sweep · {res.settings.name} · recall per range bin, and range factors (CARLA R50 vs Vantage)',
          12, 22, s=0.5)
    for i, r in enumerate(wx['rows']):
        col = _SERIES[i % len(_SERIES)]
        x = 12 + i * 180
        cv2.line(img, (x, 44), (x + 18, 44), col, 2, cv2.LINE_AA)
        _text(img, r['weather'].replace('vantage:', '')[:22], x + 24, 48, _INK2, 0.4)
    for k, c in enumerate(panels):
        ox, oy = k * pw, 64
        x0, y0, x1, y1 = ox + 50, oy + 24, ox + pw - 16, oy + ph - 40
        for t in (0, 0.25, 0.5, 0.75, 1.0):
            y = y1 - t * (y1 - y0)
            cv2.line(img, (x0, int(y)), (x1, int(y)), _HAIR if t else _INK2, 1)
            _text(img, f'{t * 100:.0f}%' if c != 'factor' else f'{t * 1.25:.2f}', x0 - 6, y + 4, _MUTED, 0.33,
                  right=True)
        if c == 'factor':
            ref0 = wx['rows'][0]['classes']
            fc = next((k for k in ('pedestrian', 'car', 'vehicle', 'traffic_light') if (ref0.get(k) or {}).get('r50_m')),
                      'car')
            _text(img, f'Range factor vs clear, {fc.replace("_", " ")}: CARLA R50 (dot), Vantage (ring)', x0 - 40,
                  oy + 14, _INK, 0.4)
            n = len(wx['rows'])
            for i, r in enumerate(wx['rows']):
                col = _SERIES[i % len(_SERIES)]
                xx = int(x0 + (i + 0.5) / n * (x1 - x0))
                v = r['classes'].get(fc) or {}
                for val, filled in ((v.get('r50_factor'), True), (v.get('vantage_factor'), False)):
                    if val is not None:
                        cv2.circle(img, (xx, int(y1 - min(val, 1.25) / 1.25 * (y1 - y0))), 4, col,
                                   -1 if filled else 1, cv2.LINE_AA)
            continue
        _text(img, c.replace('_', ' '), x0 - 40, oy + 14, _INK, 0.42)
        labels = [b['bin'] for b in next(r['classes'][c]['bins'] for r in wx['rows'] if c in r['classes'])]
        n = len(labels)
        for j, lab in enumerate(labels):
            _text(img, lab, x0 + (j + 0.5) / n * (x1 - x0), y1 + 16, _MUTED, 0.33, centre=True)
        for i, r in enumerate(wx['rows']):
            col = _SERIES[i % len(_SERIES)]
            v = r['classes'].get(c)
            if not v:
                continue
            pts = []
            for j, b in enumerate(v['bins']):
                if b['recall'] is None:
                    continue
                pts.append((int(x0 + (j + 0.5) / n * (x1 - x0) + (i - len(wx['rows']) / 2) * 3),
                            int(y1 - b['recall'] * (y1 - y0))))
            for a, b in zip(pts, pts[1:]):
                cv2.line(img, a, b, col, 2, cv2.LINE_AA)
            for p in pts:
                cv2.circle(img, p, 3, col, -1, cv2.LINE_AA)
    ok, buf = cv2.imencode('.png', img)
    return buf.tobytes() if ok else None


def _fmt(v, nd=2, unit=''):
    if v is None:
        return '—'
    return f'{v:.{nd}f}{unit}'


def report_html(res: CampaignResult, project=None) -> str:
    """A self-contained HTML report: the design, the headline, the tables, the figures (embedded PNG)."""
    st = res.settings
    det = res.detectability
    e = html.escape
    parts = [f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{e(st.name)}</title>'
             '<style>body{font-family:system-ui,sans-serif;max-width:1100px;margin:2em auto;padding:0 1em;color:#222}'
             'table{border-collapse:collapse;margin:0.6em 0 1.4em;font-size:13px}th,td{border-bottom:1px solid #ddd;'
             'padding:3px 8px;text-align:right}th:first-child,td:first-child{text-align:left}h2{margin-top:1.6em}'
             '.muted{color:#777}img{max-width:100%}</style></head><body>',
             f'<h1>{e(st.name)}</h1><p class="muted">{e(getattr(project, "name", ""))} · {e(res.variant)} · '
             f'{len(res.plan.drives)} drives · {e(res.cost.get("text", ""))}</p>']
    hl = headline(res)
    if hl:
        parts.append('<h2>Findings</h2><ul>' + ''.join(f'<li>{e(h)}</li>' for h in hl) + '</ul>')
    if det['thresholds']:
        parts.append('<h2>Smallest fault detected</h2><table><tr><th>Check</th><th>Sensor</th><th>Fault</th>'
                     '<th>Axis</th><th>False alarms</th><th>Clean statistic</th><th>Empirical</th><th>Fit</th>'
                     '<th>Interval</th></tr>')
        for t in det['thresholds']:
            ci = t['fit_ci']
            parts.append(f'<tr><td>{e(CHECKS[t["check"]][0])}</td><td>{e(t["tag"])}</td><td>{e(t["kind"])}</td>'
                         f'<td>{e(t["axis"] or "")}</td><td>{_fmt(t["floor"] * 100, 0, " %")}</td>'
                         f'<td>{_fmt(t.get("clean_stat"), 2, " " + (t.get("stat_unit") or ""))}</td>'
                         f'<td>{_fmt(t["empirical"], 3, " " + t["unit"])}</td>'
                         f'<td>{_fmt(t["fit"], 3, " " + t["unit"])}{" (extrapolated)" if t["extrapolated"] else ""}</td>'
                         f'<td>{"—" if not ci else f"{_fmt(ci[0], 3)} – {_fmt(ci[1], 3)}"}</td></tr>')
        parts.append('</table>')
    png = curves_png(res)
    if png:
        parts.append('<h2>Detection curves</h2><img alt="detection curves" src="data:image/png;base64,' +
                     base64.b64encode(png).decode() + '">')
    fas = [f for f in det['false_alarms'] if f.get('applicable', True)]
    nas = [f for f in det['false_alarms'] if not f.get('applicable', True)]
    for title, rows in (('False alarms on clean drives', fas), ('Not applicable on CARLA: listed apart, not false '
                                                                'alarms', nas)):
        if not rows:
            continue
        parts.append(f'<h2>{e(title)}</h2><table><tr><th>Check</th><th>Sensor</th><th>Flagged</th>'
                     '<th>Drives</th><th>Rate</th><th>Interval</th></tr>')
        for f in rows:
            parts.append(f'<tr><td>{e(CHECKS[f["check"]][0])}</td><td>{e(f["tag"])}</td><td>{f["k"]}</td>'
                         f'<td>{f["n"]}</td><td>{_fmt((f["p"] or 0) * 100, 0, " %")}</td>'
                         f'<td>{_fmt(f["ci"][0] * 100, 0)}–{_fmt(f["ci"][1] * 100, 0)} %</td></tr>' if f['ci'] else '')
        parts.append('</table>')
        if rows is nas:
            parts.append('<ul>' + ''.join(f'<li>{e(CHECKS[c][0])}: {e(NOT_APPLICABLE[c])}</li>'
                                          for c in sorted({f['check'] for f in nas})) + '</ul>')
    if det['accuracy']:
        parts.append('<h2>Estimation accuracy (estimate − injected)</h2><table><tr><th>Check</th><th>Sensor</th>'
                     '<th>Fault</th><th>Axis</th><th>n</th><th>Bias</th><th>RMS</th><th>Paired RMS</th></tr>')
        for a in det['accuracy']:
            parts.append(f'<tr><td>{e(CHECKS[a["check"]][0])}</td><td>{e(a["tag"])}</td><td>{e(a["kind"])}</td>'
                         f'<td>{e(a["axis"] or "")}</td><td>{a["absolute"]["n"]}</td>'
                         f'<td>{_fmt(a["absolute"]["bias"], 3, " " + a["unit"])}</td>'
                         f'<td>{_fmt(a["absolute"]["rms"], 3, " " + a["unit"])}</td>'
                         f'<td>{_fmt(a["paired"]["rms"], 3, " " + a["unit"])}</td></tr>')
        parts.append('</table>')
    if res.weather:
        parts.append('<h2>Weather sweep</h2><table><tr><th>Weather</th><th>Class</th><th>Objects</th><th>Recall</th>'
                     '<th>R50</th><th>CARLA factor</th><th>Vantage factor</th></tr>')
        for r in res.weather['rows']:
            for c, v in r['classes'].items():
                parts.append(f'<tr><td>{e(r["weather"])}</td><td>{e(c)}</td><td>{v["n"]}</td>'
                             f'<td>{_fmt((v["recall"] or 0) * 100, 0, " %") if v["recall"] is not None else "—"}</td>'
                             f'<td>{_fmt(v.get("r50_m"), 1, " m")}</td><td>{_fmt(v.get("r50_factor"), 2)}</td>'
                             f'<td>{_fmt(v.get("vantage_factor"), 2)}</td></tr>')
        parts.append('</table>')
        wp = weather_png(res)
        if wp:
            parts.append('<img alt="weather" src="data:image/png;base64,' + base64.b64encode(wp).decode() + '">')
        parts.append('<ul>' + ''.join(f'<li>{e(n)}</li>' for n in res.weather['notes']) + '</ul>')
    if res.rigs and res.rigs.get('comparison'):
        parts.append('<h2>Rig comparison</h2><table><tr><th>Recording</th><th>Class</th><th>Δ recall</th>'
                     '<th>Interval</th><th>Paired</th></tr>')
        for c in res.rigs['comparison']['comparisons']:
            ci = c.get('recall_difference_ci')
            parts.append(f'<tr><td>{e(c["recording"])}</td><td>{e(c["class"])}</td>'
                         f'<td>{_fmt((c["recall_difference"] or 0) * 100, 1, " pt")}</td>'
                         f'<td>{"—" if not ci else f"{ci[0] * 100:+.1f} – {ci[1] * 100:+.1f}"}</td>'
                         f'<td>{"yes" if c["paired"] else "no"}</td></tr>')
        parts.append('</table>')
    parts.append('<h2>Assumptions</h2><ul>' + ''.join(f'<li>{e(a)}</li>' for a in ASSUMPTIONS) + '</ul>')
    parts.append('<h2>References</h2><ul>' + ''.join(f'<li>{e(a)}</li>' for a in REFERENCES) + '</ul>')
    if res.notes:
        parts.append('<h2>Notes</h2><ul>' + ''.join(f'<li>{e(n)}</li>' for n in res.notes) + '</ul>')
    parts.append('</body></html>')
    return '\n'.join(parts)


def files(res: CampaignResult, project=None) -> list:
    """[(name, bytes)]: campaign.json, the CSV tables, the figures and report.html."""
    out = [('campaign.json', json.dumps(to_doc(res, project), indent=1, default=_jsonable).encode('utf-8'))]
    out += sorted(to_csv(res).items())
    png = curves_png(res)
    if png:
        out.append(('campaign_detection_curves.png', png))
    wp = weather_png(res)
    if wp:
        out.append(('campaign_weather.png', wp))
    out.append(('campaign_report.html', report_html(res, project).encode('utf-8')))
    return out


def export(res: CampaignResult, project, out_dir) -> list:
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for name, data in files(res, project):
        p = os.path.join(out_dir, name)
        with open(p, 'wb') as fh:
            fh.write(data)
        paths.append(p)
    return paths
