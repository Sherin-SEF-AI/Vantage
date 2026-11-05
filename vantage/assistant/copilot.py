"""The Copilot: plans for runs, approved by the user before anything runs.

The user says what they want ("a 10 s drive in Town10 at dusk in rain, 30 cars, ground truth on, half-size images");
the model answers through tools only, and its one writing tool, propose_plan, hands the app a Plan: one or more steps,
each a kind of run with the full settings of the page that runs it. Every value says where it came from (the user's
request, the page's default, chosen by the Copilot, forced by a limit) and why; every step is checked with its
settings class's own from_dict and the app's validation (the render budget, the memory free, the port, the detector
model), carries the documented limits that apply as warnings, and an estimate from the app's own estimators.

Nothing runs until the user approves. Approving writes an approval record into the project (analysis.copilot): the
plan, the request, the provider and model, the resolved settings and their SHA-256, the time and the status. The app
then runs each step through the page's own Run code path; a page's Run reached from a plan calls gate() first, which
refuses unless the project holds an approved, intact record whose step settings are exactly what the page is about to
run. An approved plan is immutable: an edit makes a new version that needs approval again. Re-running a plan from its
record uses the stored settings and calls no model.

No Qt here: the panel (ui/assistant.py) shows the plans and runs them; batch --run-plan runs an approved one."""
from __future__ import annotations

import copy
import datetime as _dt
import hashlib
import json
import math
import os
import re
import uuid
from dataclasses import dataclass, field

PLAN_SCHEMA = 'vantage.copilot_plan/1'
RECORD_SCHEMA = 'vantage.copilot_record/1'
RUN_SCHEMA = 'vantage.copilot_run/1'
SLOT = 'copilot'                     # project.analysis.copilot: {'plans': [record, ...]}
MAX_RECORDS = 60
EXIT_REFUSED = 10                    # batch --run-plan: no approved, intact plan with that id, or the gate refused

# round 14's studies (round 15): each runs its core module headless (run_study_kind), in the background in the app
STUDY_KINDS = ('rig_perception', 'foundation_range', 'robust_perception', 'redrive', 'design', 'perf_model_train',
               'perf_model_campaign', 'risk_coverage', 'fleet_twin')
KINDS = ('carla_run', 'campaign', 'detection', 'critical_carla', 'carla_live', 'experiment', 'sample_dataset',
         'study', 'export') + STUDY_KINDS
KIND_LABEL = {'carla_run': 'CARLA drive', 'campaign': 'Simulation campaign', 'detection': 'Detection quality',
              'critical_carla': 'Critical targets in CARLA', 'carla_live': 'CARLA live rig',
              'experiment': 'Designed experiment', 'sample_dataset': 'Sample dataset', 'study': 'Study',
              'export': 'Export', 'rig_perception': 'Rig perception', 'foundation_range': 'Foundation-model range',
              'robust_perception': 'Robust perception', 'redrive': 'Re-drive a recorded drive',
              'design': 'Design agent', 'perf_model_train': 'Train the learned detection model',
              'perf_model_campaign': 'Learned-model campaign', 'risk_coverage': 'Risk-weighted coverage',
              'fleet_twin': 'Fleet twin'}
KIND_PAGE = {'carla_run': 'sim', 'campaign': 'campaign', 'detection': 'detection', 'critical_carla': 'critical_carla',
             'carla_live': 'sim', 'experiment': 'experiments', 'sample_dataset': 'dataset', 'study': '',
             'export': '', 'rig_perception': 'rig_perception', 'foundation_range': 'foundation',
             'robust_perception': 'robust_perception', 'redrive': 'splat', 'design': 'designer',
             'perf_model_train': 'perf_model', 'perf_model_campaign': 'perf_model', 'risk_coverage': 'risk_coverage',
             'fleet_twin': 'fleet_twin'}
# the project.analysis slot a kind's settings live in (what "Open on its page" and Run set); None: kept in the plan.
# A study kind's settings are kept in the plan too (they hold plan-only options besides the page's settings): the
# runner merges the page's part into STUDY_SLOT[kind] (apply_study_settings) before it runs.
KIND_SLOT = {'carla_run': 'sim', 'campaign': 'campaign', 'detection': 'detection', 'critical_carla': 'critical_carla',
             'carla_live': None, 'experiment': 'experiments', 'sample_dataset': 'dataset', 'study': None,
             'export': None, **{k: None for k in STUDY_KINDS}}
SOURCES = {'request': 'from your request', 'default': 'default', 'copilot': 'chosen by the Copilot',
           'limit': 'forced by a limit'}
STATUSES = ('draft', 'approved', 'running', 'done', 'failed', 'cancelled', 'rejected', 'superseded')
RESERVED_PORTS = (2000, 2010)        # kept for a CARLA server of the user's own (allowed when the project already uses it)
LIVE_SCALE = 0.25                    # the live rig's default share of each sensor's size (ui/workspaces/sim_live.py)
STUDIES = ('blind_volume', 'stitched_objects', 'load_cases', 'mounting_tolerance', 'perception_range',
           'sensor_coverage', 'weather')
IMAGE_FORMATS = ('viewport_png', 'bev_png', 'camera_views_png')     # rendered by the app window: not in a plan

ASSUMPTIONS = (
    'A plan states what the app will run: each value is what the page would use, after its own validation; '
    '"default" is the value set on the page now (Vantage\'s default when the page was never changed). Estimates come '
    'from the app\'s own estimators (the rig\'s render megapixels, the campaign cost model with this machine\'s '
    'measured drives where there are any, the sample dataset\'s size estimate) and are estimates. The checks before '
    'approval (render budget, memory free, port, detector model) are repeated when the plan runs.')

# documented limits the model may explain and the plans quote (each from the module that owns it)
LIMITS_DOC = {
    'render_budget': 'The bridge refuses a drive whose cameras render more than {budget:g} Mpx per frame with their '
                     'ground truth (a CARLA 0.9.16 server capped at 12 GB crashed at 306 Mpx: six 1920 × 1536 '
                     'fisheyes with ground truth); a smaller image scale or less ground truth fits.',
    'weather_sensors': 'CARLA\'s fog and rain are rendered effects of the RGB cameras only: lidar and radar returns '
                       'ignore them; CARLA\'s rain is water on the lens (contrast lost at every range), not '
                       'extinction with range; CARLA has no snow.',
    'asbuilt_carla': 'The as-built check cannot judge mounts on CARLA: CARLA drives its own car, not the project\'s '
                     'body, so the ego mask and the body outline do not match the plan.',
    'ports': 'Ports 2000–2010 (and the traffic manager\'s 8100) are kept for a CARLA server of your own; a plan uses '
             'them only when the project is already set to them.',
    'memory': 'A CARLA drive needs the bridge\'s memory (the rig\'s estimate) and, when the page starts the server, '
              'the server\'s cap besides; the check reads MemAvailable.',
    'detector': 'Detection quality runs an off-the-shelf detector (YOLOX, COCO classes) downloaded on request on '
                'Simulate ▸ Detection quality; a plan cannot download it.',
    'faults': 'A fault changes what CARLA records (the actual mount, the stamps, the delivery); the calibration '
              'outputs still state the plan, as on a real faulty rig.',
    'carla_vehicle': 'CARLA renders its own world and its own car (the nearest CARLA vehicle by size), not the '
                     'project\'s body.',
    'live': 'The live rig renders each camera in real time at a share of its size and serves MJPEG streams to the Live '
            'rig page; below real time CARLA slows the streams down.',
    'sample_dataset': 'The sample dataset is synthetic geometry for testing pipelines, not for training.',
    'experiment_carla': 'Experiment responses computed in CARLA drive the simulator once per design point.',
    'scene_set': 'A perception scene set drives every scene once per seed and per variant in CARLA (about 6.5 min of '
                 'wall time per 12 s six-camera drive with ground truth on the development machine: the standard set '
                 'for two rigs took 56 min); drives already there are reused.',
    'perception_detector': 'Rig perception lifts an off-the-shelf 2D detector (YOLOX, COCO classes) to 3D boxes on '
                           'CARLA renders: compare rigs on the same scenes, not absolute performance. A plan cannot '
                           'download the detector.',
    'foundation_models': 'Foundation models (OWLv2, Grounding DINO, Depth Anything V2) need PyTorch and transformers '
                         'and are downloaded on request on Simulate ▸ Foundation models (a plan cannot download them); '
                         'they run on the GPU when there is one. Trained on photographs, measured on renders.',
    'redrive': 'A re-drive rebuilds the recorded drive as 3D Gaussians (PyTorch; a GPU makes it minutes, the CPU much '
               'longer): moving people are rebuilt as rigid bodies, so detection of pedestrians on a re-shot drive is '
               'not valid.',
    'design_options': 'The design agent never changes the project: an option becomes a variant only when you create '
                      'it on Analysis ▸ Design agent (one undo step, written to the design log).',
    'perf_model': 'The learned model is a Gaussian process on recall from the rig\'s geometry, trained on perception '
                  'records; a new town, weather or detector is outside what it has seen. Its store and saved model '
                  'are in your data folder (or the project\'s, when it opts in).',
    'fleet_watch': 'Watching the fleet\'s folders once reads every drive that arrived and has been quiet for the '
                   'watch\'s quiet time; each drive is read once.',
}


_PLAN_WORDS = re.compile(
    r'(^\s*/plan\b)|\b(drive|drives|driving|run|runs|record|recording|campaign|sweep|detection|carla|live rig|'
    r'experiment|dataset|generate|export|simulate|simulation|critical targets?|scenario|seeds?|faults?)\b',
    re.IGNORECASE)


def is_plan_request(text: str) -> bool:
    """Does the message ask for something to be run or written (a Copilot turn: thinking effort high)? Also every
    request the deterministic planner maps to a study kind ("how far can F see a cyclist")."""
    return bool(_PLAN_WORDS.search(text or '')) or bool(match_kind(text))


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).strftime('%Y-%m-%dT%H:%M:%SZ')


def canonical(o) -> bytes:
    return json.dumps(o, sort_keys=True, separators=(',', ':'), ensure_ascii=False, default=str).encode('utf-8')


def sha256(o) -> str:
    return hashlib.sha256(canonical(o)).hexdigest()


# =========================================================================== flat keys
def flatten(d: dict, prefix='') -> dict:
    """{'scenario.name': 'cut_in', 'ground_truth.semantic': True, 'faults': [...]}: nested dicts as dotted keys, lists
    and other values as they are."""
    out = {}
    for k, v in (d or {}).items():
        key = f'{prefix}{k}'
        if isinstance(v, dict) and v:
            out.update(flatten(v, key + '.'))
        else:
            out[key] = v
    return out


def set_path(d: dict, key: str, value):
    parts = key.split('.')
    cur = d
    for p in parts[:-1]:
        if not isinstance(cur.get(p), dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def get_path(d: dict, key: str, default=None):
    cur = d
    for p in key.split('.'):
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def same(a, b) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) \
            and not isinstance(b, bool):
        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-9)
    return canonical(a) == canonical(b)


def parse_value(text):
    """A value the model wrote as JSON text ("Town10HD" quoted, 30, true, [..]); a bare word is taken as a string."""
    if not isinstance(text, str):
        return text
    t = text.strip()
    try:
        return json.loads(t)
    except ValueError:
        return t


# =========================================================================== the kinds
def _tags(project) -> dict:
    v = project.variant
    out = {c.tag: 'camera' for c in v.cameras if c.visible}
    out.update({s.tag: s.kind for s in v.sensors if s.visible})
    return out


def _working(project, working_dir=''):
    from .. import workdir
    return working_dir or workdir.default()


def defaults(kind: str, project) -> dict:
    """The settings a step of this kind starts from: what the page holds now (its default when never changed)."""
    if kind == 'carla_run':
        from ..core import simrig as SR
        return SR.settings_of(project).to_dict()
    if kind == 'campaign':
        from ..core import simcampaign as SC
        return SC.settings_of(project).to_dict()
    if kind == 'detection':
        from ..core import detection_eval as DE
        return DE.settings_of(project).to_dict()
    if kind == 'critical_carla':
        from ..core import critical_carla as CC
        return CC.settings_of(project).to_dict()
    if kind == 'carla_live':
        return {'cameras': [c.tag for c in project.variant.cameras if c.visible], 'scale': LIVE_SCALE,
                'duration_s': 0.0, 'faults': []}
    if kind == 'experiment':
        from ..core import experiments as EX
        return EX.settings_of(project).to_dict()
    if kind == 'sample_dataset':
        from ..core import dataset as DSET
        return DSET.DatasetSettings.from_dict(project.analysis.dataset or {}).to_dict()
    if kind == 'study':
        return {'study': 'blind_volume', 'height': 2.0, 'object_height': 0.9, 'tolerance_deg': None,
                'tolerance_mm': None, 'condition': ''}
    if kind == 'export':
        return {'formats': [], 'folder': '', 'frame': 'rear_axle', 'units': 'm', 'angles': 'deg'}
    if kind in STUDY_KINDS:
        return study_defaults(kind, project)
    raise KeyError(kind)


def _fix_experiment(d: dict, project) -> dict:
    """Experiment factors may name their camera or sensor by tag: the settings store uids."""
    v = project.variant
    out = copy.deepcopy(d)
    for f in out.get('factors') or []:
        if not isinstance(f, dict):
            continue
        t = str(f.get('target') or '')
        if f.get('kind') == 'camera' and v.cam(t) is None:
            c = v.by_tag(t) if hasattr(v, 'by_tag') else None
            if c is not None:
                f['target'] = c.uid
        elif f.get('kind') == 'sensor' and v.sensor(t) is None:
            s = next((x for x in v.sensors if x.tag == t), None)
            if s is not None:
                f['target'] = s.uid
    return out


def _resolve_recording(name, project, working_dir='') -> str:
    """A recording named as the catalogue shows it (relative to the working folder) or by its full path."""
    p = str(name or '').strip()
    if not p:
        return ''
    cands = [p] if os.path.isabs(os.path.expanduser(p)) else []
    cands.append(os.path.join(_working(project, working_dir), p))
    if getattr(project, 'path', ''):
        cands.append(os.path.join(os.path.dirname(os.path.abspath(project.path)), p))
    for c in cands:
        c = os.path.abspath(os.path.expanduser(c))
        if os.path.isdir(c):
            return c
    return os.path.abspath(os.path.expanduser(cands[0]))         # (not there: recording_info says so)


def clean(kind: str, d: dict, project, working_dir='') -> tuple[dict, list]:
    """(settings as the page's settings class keeps them, problems): the class's own from_dict (unknown keys dropped,
    numbers clamped, words checked), and for the parts it does not check, the app's own validators."""
    bad = []
    d = d if isinstance(d, dict) else {}
    if kind == 'carla_run':
        from ..core import simrig as SR
        from ..sim import carla_bridge as CB
        st = SR.SimSettings.from_dict(d)
        _, why = CB.clean_faults(d.get('faults'), _tags(project))
        bad += why
        sc = d.get('scenario') if isinstance(d.get('scenario'), dict) else {}
        if sc:
            _, why = CB.clean_scenario(sc.get('name'), sc.get('params'))
            bad += why
        _, why = CB.clean_actors(d.get('actors'))
        bad += why
        return st.to_dict(), bad
    if kind == 'campaign':
        from ..core import simcampaign as SC
        return SC.CampaignSettings.from_dict(d).to_dict(), bad
    if kind == 'detection':
        from ..core import detection_eval as DE
        dd = dict(d)
        if isinstance(dd.get('recordings'), list):
            dd['recordings'] = [_resolve_recording(r, project, working_dir) for r in dd['recordings'] if str(r).strip()]
        return DE.DetectionSettings.from_dict(dd).to_dict(), bad
    if kind == 'critical_carla':
        from ..core import critical_carla as CC
        return CC.CriticalCarlaSettings.from_dict(d).to_dict(), bad
    if kind == 'carla_live':
        return _clean_live(d, project)
    if kind == 'experiment':
        from ..core import experiments as EX
        return EX.ExperimentSettings.from_dict(_fix_experiment(d, project)).to_dict(), bad
    if kind == 'sample_dataset':
        from ..core import dataset as DSET
        return DSET.DatasetSettings.from_dict(d).to_dict(), bad
    if kind == 'study':
        return _clean_study(d)
    if kind == 'export':
        return _clean_export(d)
    if kind in STUDY_KINDS:
        return clean_study(kind, d, project, working_dir)
    raise KeyError(kind)


def _num(v, lo, hi):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if isinstance(v, bool) or not math.isfinite(x):
        return None
    return min(max(x, lo), hi)


def _clean_live(d, project):
    from ..core import simrig as SR
    from ..sim import live as LV
    bad = []
    vis = [c.tag for c in project.variant.cameras if c.visible]
    cams = d.get('cameras')
    if isinstance(cams, str):
        cams = [t.strip() for t in cams.split(',') if t.strip()]
    if not isinstance(cams, list) or not cams:
        cams = list(vis)
    unknown = [str(t) for t in cams if t not in vis]
    if unknown:
        bad.append(f'No visible camera {", ".join(unknown)} (cameras: {", ".join(vis) or "none"}).')
    cams = [t for t in vis if t in cams]
    lo = SR.LIMITS['image_scale'][0]
    scale = _num(d.get('scale', LIVE_SCALE), lo, 1.0)
    dur = _num(d.get('duration_s', 0.0), 0.0, 3600.0)
    faults = []
    for i, f in enumerate(d.get('faults') or []):
        if not isinstance(f, dict):
            bad.append(f'Live fault {i + 1}: not an object.')
            continue
        at = _num(f.get('at_s'), 0.0, 3600.0)
        if at is None:
            bad.append(f'Live fault {i + 1}: at_s (seconds after the streams start) is missing.')
            continue
        raw = f.get('fault') if isinstance(f.get('fault'), dict) else {k: v for k, v in f.items() if k != 'at_s'}
        try:
            faults.append({'at_s': round(at, 3), 'fault': LV.parse_fault(raw, cams)})
        except ValueError as e:
            bad.append(f'Live fault {i + 1}: {e}')
    faults.sort(key=lambda x: x['at_s'])
    out = {'cameras': cams, 'scale': round(scale if scale is not None else LIVE_SCALE, 2),   # the page's whole percent
           'duration_s': round(dur or 0.0, 3), 'faults': faults}
    if out['duration_s'] and any(f['at_s'] >= out['duration_s'] for f in faults):
        bad.append('A live fault is scheduled after the live rig stops (duration_s).')
    return out, bad


def _clean_study(d):
    from ..core.weather import IDS as WEATHER_IDS
    bad = []
    st = d.get('study')
    if st not in STUDIES:
        bad.append(f'study must be one of {", ".join(STUDIES)}.')
        st = 'blind_volume'
    out = {'study': st, 'height': _num(d.get('height', 2.0), 0.3, 4.0) or 2.0,
           'object_height': _num(d.get('object_height', 0.9), 0.1, 2.5) or 0.9,
           'tolerance_deg': _num(d.get('tolerance_deg'), 0.0, 30.0), 'tolerance_mm': _num(d.get('tolerance_mm'), 0.0,
                                                                                          500.0),
           'condition': d.get('condition') if d.get('condition') in WEATHER_IDS else ''}
    return out, bad


def _clean_export(d):
    from ..core import exporters as ex
    bad = []
    ids = []
    for f in d.get('formats') or []:
        f = str(f).strip()
        try:
            ex.format_by_id(f)
        except (KeyError, ValueError):
            bad.append(f'No export format {f!r}.')
            continue
        if f in IMAGE_FORMATS:
            bad.append(f'{f} is rendered by the app window: export it from the Export dialog.')
            continue
        if f not in ids:
            ids.append(f)
    folder = str(d.get('folder') or '').strip().replace('\\', '/')
    if folder.startswith('/') or folder.startswith('~') or '..' in folder.split('/'):
        bad.append('The export folder is relative to the working folder (no .. and no absolute path).')
        folder = ''
    out = {'formats': ids, 'folder': folder[:200],
           'frame': d.get('frame') if d.get('frame') in ('rear_axle', 'front_axle', 'front_bumper', 'ground_centre',
                                                          'custom') else 'rear_axle',
           'units': d.get('units') if d.get('units') in ('m', 'mm') else 'm',
           'angles': d.get('angles') if d.get('angles') in ('deg', 'rad') else 'deg'}
    if not ids:
        bad.append('Name at least one export format (get_catalogue topic export_formats).')
    return out, bad


def allowed_keys(kind: str, settings: dict) -> set:
    """The keys a step's values may set: every key of the kind's settings, and the chosen scenario's parameters."""
    keys = set(flatten(settings)) | set(settings)
    for k, v in list(settings.items()):
        if isinstance(v, dict):
            keys.add(k)
    if kind == 'carla_run':
        from ..sim import carla_bridge as CB
        for name, spec in CB.SCENARIOS.items():
            keys |= {f'scenario.params.{p}' for p in spec[2]}
        keys |= {'scenario', 'scenario.name', 'scenario.params', 'ground_truth'}
    if kind == 'campaign':
        keys.add('template')
    return keys


# =========================================================================== values in words
# The plan card, the change line and the model's view say a value as the page would: a factor as "F pitch 25–45°
# (5 levels)", a response as "Blind ground in the ROI (m²), less is better", a fault as "F turned pitch +1°", never as
# the JSON the settings hold (the card's tooltip keeps the JSON).
TEXT_MAX = 160


def _g(x) -> str:
    """A number as the pages print it (no trailing zeros, a real minus sign)."""
    if isinstance(x, bool):
        return 'on' if x else 'off'
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    t = f'{x:g}' if abs(x) >= 1e-3 or x == 0 else f'{x:.3g}'
    return t.replace('-', '−')


def _cut(t: str, n=TEXT_MAX) -> str:
    return t if len(t) <= n else t[:n - 1].rstrip(' ,·;') + '…'


def _human(key: str) -> str:
    return str(key).split('.')[-1].replace('_', ' ')


def generic_text(v, depth=0) -> str:
    """Any structured value in plain words: a list as its items, an object as 'key value' pairs."""
    if v is None or v == '':
        return '—'
    if isinstance(v, bool):
        return 'on' if v else 'off'
    if isinstance(v, (int, float)):
        return _g(v)
    if isinstance(v, str):
        return v
    if isinstance(v, (list, tuple)):
        if not v:
            return 'none'
        if all(not isinstance(x, (list, dict)) for x in v):
            return ', '.join(generic_text(x) for x in v)
        return ' · '.join(generic_text(x, depth + 1) for x in v)
    if isinstance(v, dict):
        if not v:
            return 'the page\'s defaults' if depth == 0 else 'none'
        parts = []
        for k, x in v.items():
            t = generic_text(x, depth + 1)
            parts.append(f'{_human(k)} {"(" + t + ")" if isinstance(x, (list, dict)) and x else t}')
        return ', '.join(parts)
    return str(v)


def weather_text(w) -> str:
    """'HardRainSunset', 'Vantage fog', 'the project's condition'."""
    w = str(w or '')
    if w == 'project':
        return 'the project\'s condition'
    if w.startswith('vantage:'):
        from ..core import weather as WX
        cid = w.split(':', 1)[1]
        name = next((c.name for c in WX.PRESETS if c.id == cid), cid.replace('_', ' '))
        return f'Vantage {name[:1].lower()}{name[1:]}'
    return w or '—'


def _tag_of(project, uid):
    if project is None:
        return uid
    v = project.variant
    c = v.cam(uid) if hasattr(v, 'cam') else None
    if c is not None:
        return c.tag
    s = v.sensor(uid) if hasattr(v, 'sensor') else None
    return s.tag if s is not None else uid


def factor_text(d, project=None, raw=False) -> str:
    """'F pitch 25–45° (5 levels)', 'F lens h05r2, h03r2', 'All cameras z offset ±0.05 m', 'Weather fog, clear'.
    raw: the numbers as asked (before the page's limits), to say what a limit changed."""
    from ..core import experiments as EX
    if isinstance(d, dict) and project is not None:
        d = _fix_experiment({'factors': [d]}, project)['factors'][0]
    f = EX.Factor.from_dict(d)
    if f is None:
        return generic_text(d, 1)
    if raw and not f.categorical:
        for k in ('lo', 'hi', 'levels', 'mean', 'sd'):
            x = EX._num(d.get(k))
            if x is not None:
                setattr(f, k, int(x) if k == 'levels' else x)
    name = EX.factor_label(f, project) if project is not None else (f.name or f'{f.target or f.kind} {f.attr}')
    if f.categorical:
        return f'{name} {", ".join(f.choices) or "(no levels)"}'
    u = f.unit
    sp = '' if u in ('°', '') else ' '
    if f.mode == 'offset' and math.isclose(f.lo, -f.hi):
        rng = f'±{_g(f.hi)}{sp}{u}'
    else:
        rng = f'{_g(f.lo)}–{_g(f.hi)}{sp}{u}'
    extra = []
    if f.levels:
        extra.append(f'{f.levels} levels')
    if f.dist != 'uniform':
        extra.append(f'{EX.DIST_LABEL[f.dist].lower()} {_g(f.mean)} ± {_g(f.sd)}{sp}{u}')
    return f'{name} {rng}' + (f' ({", ".join(extra)})' if extra else '')


def response_text(d, project=None) -> str:
    """'Blind ground in the ROI (m²), less is better', with its parameters and target when it has them."""
    from ..core import experiments as EX
    r = EX.clean_response(d) if isinstance(d, dict) else None
    if r is None:
        return generic_text(d, 1)
    rr = EX.resolve_response(r, project)
    t = rr.label + (f' ({rr.unit})' if rr.unit else '')
    if rr.better in ('<=', '<'):
        t += ', less is better'
    elif rr.better in ('>=', '>'):
        t += ', more is better'
    if rr.target is not None and rr.op:
        t += f', target {EX.RQ.OP_TEXT.get(rr.op, rr.op)} {_g(rr.target)}' + (f' {rr.unit}' if rr.unit else '')
    return t


_AXES = {'deg': ('yaw', 'pitch', 'roll'), 'm': ('x', 'y', 'z')}


def carla_fault_text(f) -> str:
    """A CARLA drive's fault: 'F turned pitch +1°', 'L1 range bias +5 cm', 'R drops 5 %', 'F frozen 2–3 s'."""
    if not isinstance(f, dict) or 'kind' not in f:
        return generic_text(f, 1)
    tag, k = f.get('tag', '?'), f['kind']
    try:
        if k == 'rotation':
            bits = [f'{n} {float(v):+g}°' for n, v in zip(_AXES['deg'], f['deg']) if v]
            return f'{tag} turned ' + (', '.join(bits) or '0°')
        if k == 'translation':
            bits = [f'{n} {100 * float(v):+g} cm' for n, v in zip(_AXES['m'], f['m']) if v]
            return f'{tag} moved ' + (', '.join(bits) or '0 cm')
        if k == 'clock_offset':
            return f'{tag} stamped {float(f["ms"]):+g} ms'
        if k == 'range_bias':
            return f'{tag} range bias {100 * float(f["m"]):+g} cm'
        if k == 'drop':
            if 'frames' in f:
                fr = list(f['frames'])
                return f'{tag} drops frames ' + ', '.join(str(x) for x in fr[:8]) + (' …' if len(fr) > 8 else '')
            return f'{tag} drops {100 * float(f["rate"]):g} %'
        if k == 'freeze':
            return f'{tag} frozen {_g(f["from_s"])}–{_g(f["to_s"])} s'
    except (KeyError, TypeError, ValueError):
        pass
    return generic_text(f, 1)


def live_fault_text(item) -> str:
    """A live-rig fault on its schedule: 'at 2 s: F turned pitch +1° for 3 s'."""
    if not isinstance(item, dict) or not isinstance(item.get('fault'), dict):
        return generic_text(item, 1)
    from ..sim import live as LV
    f = item['fault']
    try:
        what = LV.describe_fault(f)
    except (KeyError, TypeError, ValueError):
        what = generic_text(f, 1)
    return f'at {_g(item.get("at_s", 0))} s: {f.get("tag", "?")} {what}'


def fault_set_text(fs) -> str:
    """A campaign's fault set: 'rotation of F, ML by 0.1, 0.2, 0.5° (yaw, pitch, roll; signs alternate)'."""
    if not isinstance(fs, dict) or 'kind' not in fs:
        return generic_text(fs, 1)
    from ..core import simcampaign as SC
    k = fs['kind']
    unit = SC.FAULT_UNIT.get(k, '')
    mags = ', '.join(_g(m) for m in fs.get('magnitudes') or [])
    t = f'{SC.FAULT_LABEL.get(k, k).lower()} of {", ".join(fs.get("tags") or []) or "every sensor"}'
    if mags:
        t += f' by {mags}{"" if unit in ("°", "%") else " "}{unit}'
    extra = []
    if fs.get('axes'):
        extra.append(', '.join(fs['axes']))
    if fs.get('signs'):
        extra.append(f'signs {fs["signs"]}')
    return t + (f' ({"; ".join(extra)})' if extra else '')


def actor_text(a) -> str:
    """An actors scene's actor: 'child at 12, −2 m', 'car at 20, 3.5 m facing 90°'."""
    if not isinstance(a, dict) or 'kind' not in a:
        return generic_text(a, 1)
    p = a.get('pose') or {}
    t = f'{a["kind"].replace("blueprint:", "")} at {_g(p.get("x", 0))}, {_g(p.get("y", 0))} m'
    if p.get('z'):
        t += f' (z {_g(p["z"])} m)'
    if p.get('yaw'):
        t += f' facing {_g(p["yaw"])}°'
    return t


def _scenario_label(name) -> str:
    from ..sim import carla_bridge as CB
    return CB.SCENARIOS.get(name, (str(name),))[0]


def _item_texts(kind, key, value, project=None, raw=False) -> list | None:
    """The items of a list setting in words (None: not a list this module knows)."""
    base = key.split('.')[-1]
    if not isinstance(value, list):
        return None
    if kind == 'experiment' and key == 'factors':
        return [factor_text(f, project, raw) for f in value]
    if kind == 'experiment' and key == 'responses':
        return [response_text(r, project) for r in value]
    if key == 'faults' and kind == 'carla_live':
        return [live_fault_text(f) for f in value]
    if base == 'faults':
        return [carla_fault_text(f) for f in value]
    if base == 'actors':
        return [actor_text(a) for a in value]
    if kind == 'campaign' and key == 'fault_sets':
        return [fault_set_text(f) for f in value]
    if kind == 'campaign' and key == 'scenarios':
        return [_scenario_label(s.get('name')) + (f' ({generic_text(s["params"], 1)})' if s.get('params') else '')
                if isinstance(s, dict) else _scenario_label(s) for s in value]
    if kind == 'campaign' and key == 'weathers':
        return [weather_text(w) for w in value]
    if kind == 'campaign' and key == 'evaluators':
        from ..core import simcampaign as SC
        return [SC.EVALUATOR_LABEL.get(e, e) for e in value]
    if kind == 'campaign' and key == 'variants' and project is not None:
        return [next((v.name for v in project.variants if v.uid == u), u) for u in value]
    if kind in ('detection', 'foundation_range', 'robust_perception') and key == 'recordings':
        return [os.path.basename(str(r).rstrip('/\\')) or str(r) for r in value]
    if kind in STUDY_KINDS:
        t = study_item_texts(kind, key, value, project)
        if t is not None:
            return t
    if kind == 'export' and key == 'formats':
        from ..core import exporters as ex
        out = []
        for f in value:
            try:
                out.append(ex.format_by_id(f).label)
            except (KeyError, ValueError):
                out.append(str(f))
        return out
    return [generic_text(x, 1) for x in value]


def value_text(kind, key, value, project=None, unit='', scale=1.0) -> str:
    """A setting's value in words, as the plan card shows it (unit and scale: the page's own, for a number)."""
    base = key.split('.')[-1]
    if isinstance(value, bool):
        return 'on' if value else 'off'
    if isinstance(value, (int, float)):
        if kind == 'critical_carla' and key == 'spawn_points':
            return _g(value)
        return f'{_g(value * scale)} {unit}'.strip()
    if isinstance(value, str):
        if base == 'weather' or (kind == 'campaign' and key == 'base.weather'):
            return weather_text(value)
        if kind == 'experiment' and key == 'variant':
            if not value:
                return 'the active variant'
            v = next((v for v in project.variants if v.uid == value), None) if project is not None else None
            return v.name if v is not None else value
        if kind == 'experiment' and key == 'design':
            from ..core import experiments as EX
            return EX.DESIGN_LABEL.get(value, value)
        if base == 'name' and key.startswith(('scenario.', 'base.scenario.')):
            return _scenario_label(value)
        if kind in STUDY_KINDS:
            t = study_value_text(kind, key, value, project)
            if t is not None:
                return t
        return value if value else '—'
    if value is None:
        return '—'
    if isinstance(value, list):
        if not value and kind in STUDY_KINDS and (kind, key) in STUDY_EMPTY_TEXT:
            return STUDY_EMPTY_TEXT[(kind, key)]
        if not value:
            return {'cameras': 'every camera', 'variants': 'the active variant',
                    'weathers': 'the base drive\'s weather', 'scenarios': 'the base drive\'s scenario',
                    'faults': 'none', 'actors': 'none', 'recordings': 'none'}.get(base, 'none')
        if kind == 'critical_carla' and key == 'spawn_points':
            return 'chosen by the seed' if value == [-1] else 'points ' + ', '.join(_g(x) for x in value)
        if kind == 'detection' and key == 'range_edges':
            e = [_g(x) for x in value]
            return ', '.join(f'{a}–{b}' for a, b in zip(['0'] + e, e)) + f', {e[-1]}+ m'
        if kind == 'detection' and key in ('height_edges',):
            return ', '.join(_g(x) for x in value) + ' px'
        if kind == 'detection' and key == 'view_yaws':
            return ', '.join(f'{_g(x)}°' for x in value)
        items = _item_texts(kind, key, value, project)
        return _cut(' · '.join(items) if any(' ' in t for t in items) else ', '.join(items))
    if isinstance(value, dict):
        if kind == 'critical_carla' and key == 'actors':
            names = {}
            if project is not None:
                from ..core import critical as CR
                names = {t['uid']: t['name'] for t in CR.settings_of(project).targets}
            return _cut(', '.join(f'{names.get(u, u)}: {c}' for u, c in value.items()) or 'each target\'s own kind')
        return _cut(generic_text(value))
    return _cut(generic_text(value))


def change_text(kind, key, before, after, project=None, label='', unit='', scale=1.0) -> str:
    """One changed setting in words: 'Traffic 30 → 20 vehicles', 'Weather HardRainSunset → Vantage fog'; a list
    setting as what was added or removed ('Factors: added F lens h05r2, h03r2')."""
    label = label or _human(key).capitalize()
    if isinstance(before, list) or isinstance(after, list):
        a = before if isinstance(before, list) else []
        b = after if isinstance(after, list) else []
        ta, tb = _item_texts(kind, key, a, project) or [], _item_texts(kind, key, b, project) or []
        if kind == 'carla_live' and key == 'cameras' or key.split('.')[-1] in ('cameras', 'classes', 'seeds',
                                                                               'evaluators', 'formats', 'weathers'):
            added = [t for t in tb if t not in ta]
            removed = [t for t in ta if t not in tb]
        elif len(a) == len(b):
            pairs = [(x, y) for x, y in zip(ta, tb) if x != y]
            if pairs and len(pairs) <= 2:
                return f'{label}: ' + '; '.join(f'{x} → {y}' for x, y in pairs)
            added, removed = [t for t in tb if t not in ta], [t for t in ta if t not in tb]
        else:
            added, removed = [t for t in tb if t not in ta], [t for t in ta if t not in tb]
        bits = []
        if added:
            bits.append('added ' + ' · '.join(added))
        if removed:
            bits.append('removed ' + ' · '.join(removed))
        return _cut(f'{label}: ' + ('; '.join(bits) if bits else 'reordered'), 2 * TEXT_MAX)
    num = isinstance(before, (int, float)) and isinstance(after, (int, float)) and not isinstance(before, bool) \
        and not isinstance(after, bool)
    if num:
        return f'{label} {_g(before * scale)} → {_g(after * scale)}' + (f' {unit}' if unit else '')
    tb_ = value_text(kind, key, before, project, unit, scale) if before is not None else 'not set'
    ta_ = value_text(kind, key, after, project, unit, scale) if after is not None else 'not set'
    return _cut(f'{label} {tb_} → {ta_}', 2 * TEXT_MAX)


# =========================================================================== settings a step does not use
# The experiment page shows a design's own fields only (ui/workspaces/experiments_ws.DESIGN_FIELDS; a test keeps the
# two in step): a value for another design's field is not used by this one.
DESIGN_USES = {'samples': ('lhs', 'montecarlo'), 'sobol_n': ('sobol',), 'morris_r': ('morris',),
               'levels': ('factorial', 'oat'), 'morris_levels': ('morris',),
               'seed': ('lhs', 'sobol', 'morris', 'montecarlo')}
STUDY_USES = {'height': ('blind_volume',), 'object_height': ('stitched_objects',),
              'tolerance_deg': ('mounting_tolerance',), 'tolerance_mm': ('mounting_tolerance',),
              'condition': ('weather',)}
CAMPAIGN_EVALUATOR_OF = {'drivelog': 'drivelog', 'health': 'rig_health', 'detection': 'detection'}
DETECTION_VIEW_KEYS = ('view_fov_deg', 'view_pitch_deg', 'view_yaws', 'view_width', 'view_aspect')


def _and(names) -> str:
    names = list(names)
    return names[0] if len(names) == 1 else ', '.join(names[:-1]) + ' and ' + names[-1]


def unused(kind, settings, project=None) -> dict:
    """{flat key or key prefix: (label, why)} for the settings this step's design, scenario or study does not use. They
    are shown greyed as 'not used', are never 'forced by a limit' and are not decisions of the plan."""
    s = settings if isinstance(settings, dict) else {}
    out = {}
    if kind == 'experiment':
        from ..core import experiments as EX
        d = s.get('design', 'lhs')
        dl = EX.DESIGN_LABEL.get(d, d)
        for k, designs in DESIGN_USES.items():
            if d not in designs:
                out[k] = ('not used by this design', f'{dl} does not use it ({_and(EX.DESIGN_LABEL[x] for x in designs)}'
                                                     f' {"does" if len(designs) == 1 else "do"})')
        fs = [f for f in (EX.Factor.from_dict(x) for x in s.get('factors') or []) if f is not None]
        if d in DESIGN_USES['levels'] and fs and all(f.categorical or f.levels for f in fs):
            out['levels'] = ('not used by this design', 'every factor sets its own levels')
    elif kind == 'carla_run':
        sc = (s.get('scenario') or {}).get('name') or 'free_drive'
        if s.get('actors'):
            out['traffic'] = ('not used with an actors scene', 'an actors scene has no other traffic')
            out['walkers'] = ('not used with an actors scene', 'an actors scene has no walkers')
        elif sc == 'parked':
            out['traffic'] = ('not used by this scenario', 'the ego stands still with no traffic')
            out['walkers'] = ('not used by this scenario', 'the ego stands still with no walkers')
        elif sc != 'free_drive':
            out['traffic'] = ('not used by this scenario', f'a {_scenario_label(sc).lower()} has its own Other '
                                                           'traffic (scenario.params.traffic)')
        if float(s.get('tick_hz') or 0) > 0:
            for k in ('min_tick_hz', 'max_tick_hz'):
                out[k] = ('not used with a fixed step rate', f'the step rate is set to {_g(s["tick_hz"])} Hz')
        if project is not None and not any(x.kind == 'lidar' and x.visible for x in project.variant.sensors):
            out['ground_truth.lidar_labels'] = ('not used by this rig', 'the rig has no lidar')
    elif kind == 'campaign':
        ev = s.get('evaluators') or []
        for k, e in CAMPAIGN_EVALUATOR_OF.items():
            if e not in ev:
                from ..core import simcampaign as SC
                out[k] = ('not used by this campaign', f'the {SC.EVALUATOR_LABEL[e]} is not among its checks')
    elif kind == 'detection':
        if s.get('views', 'fisheye') == 'fisheye':
            for k in DETECTION_VIEW_KEYS:
                out[k] = ('not used with fisheye images', 'no pinhole views are cut from the images')
    elif kind == 'study':
        st = s.get('study', '')
        for k, studies in STUDY_USES.items():
            if st not in studies:
                out[k] = ('not used by this study', f'{st.replace("_", " ")} does not take it')
    elif kind in STUDY_KINDS:
        out.update(study_unused(kind, s, project))
    return out


def unused_of(key: str, un: dict):
    """(label, why) when the flat key is not used (itself or under a key prefix), else None."""
    if key in un:
        return un[key]
    for k, v in un.items():
        if key.startswith(k + '.'):
            return v
    return None


# =========================================================================== a value the page changed
def _empty(v) -> bool:
    return v is None or v == '' or (isinstance(v, (list, dict, tuple)) and not v)


def _same_scalar(a, g) -> bool:
    if isinstance(a, str) and isinstance(g, bool):          # the settings classes read 'true', 'on', 'yes' as on
        return (a.strip().lower() in ('1', 'true', 'yes', 'on')) == g
    if isinstance(a, bool) or isinstance(g, bool):
        return bool(a) == bool(g) if isinstance(a, (bool, int)) and isinstance(g, (bool, int)) else False
    if isinstance(a, (int, float)) and isinstance(g, (int, float)):
        return math.isclose(float(a), float(g), rel_tol=1e-9, abs_tol=1e-9)
    if isinstance(a, str) and isinstance(g, (int, float)):
        try:
            return math.isclose(float(a), float(g), rel_tol=1e-9, abs_tol=1e-9)
        except ValueError:
            return False
    return a == g or str(a) == str(g)


def _compare(a, g, path, out, ignore_of=None):
    """Walk what was asked (a) against what the page keeps (g): out['changed'] gets [(path, asked, kept)] where a value
    the request gave is not kept (clamped, replaced, items dropped), out['ignored'] the keys the page does not know,
    out['filled'] the ones the page added. Empty values (None, '', [], {}) are all the same: nothing given."""
    if _empty(a):
        if not _empty(g):
            out['filled'].append(path)
        return
    if isinstance(a, dict):
        if not isinstance(g, dict):
            out['changed'].append((path, a, g))
            return
        skip = ignore_of(a, g) if ignore_of else ()
        for k, v in a.items():
            if k in skip:
                continue
            if k not in g:
                out['ignored'].append(f'{path}.{k}' if path else k)
                continue
            _compare(v, g[k], f'{path}.{k}' if path else k, out)
        out['filled'] += [f'{path}.{k}' if path else k for k in g if k not in a]
        return
    if isinstance(a, (list, tuple)):
        if not isinstance(g, list) or len(a) != len(g):
            out['changed'].append((path, a, g))
            return
        for i, (x, y) in enumerate(zip(a, g)):
            _compare(x, y, f'{path}[{i + 1}]', out, ignore_of)
        return
    if not _same_scalar(a, g):
        out['changed'].append((path, a, g))


def _factor_ignore(a, g):
    """What the experiment page derives in a factor (its id kept unique, a categorical factor's range and levels from
    its choices, the mirror of a camera only): asking for it changes nothing."""
    from ..core import experiments as EX
    skip = {'id'}
    if EX.is_categorical(g.get('kind'), g.get('attr')):
        skip |= {'lo', 'hi', 'levels', 'dist', 'mean', 'sd', 'mode'}
    if g.get('kind') != 'camera':
        skip.add('mirror')
    return skip


def _response_ignore(a, g):
    return {'op'} if g.get('target') is None else set()


def meaning(kind, key, asked, kept, project=None) -> dict:
    """Did the page keep what was asked? {'changed': [(path, asked, kept)], 'ignored': [keys], 'filled': [keys]}:
    'changed' empty means the page kept the request's meaning (it may have filled in its defaults or written it its own
    way, e.g. a camera tag as its uid)."""
    out = {'changed': [], 'ignored': [], 'filled': []}
    a, ign = copy.deepcopy(asked), None
    if kind == 'experiment' and key == 'factors' and isinstance(a, list) and project is not None:
        a = _fix_experiment({'factors': a}, project)['factors']
        ign = _factor_ignore
    elif kind == 'experiment' and key == 'responses':
        ign = _response_ignore
    elif kind == 'carla_live' and key == 'cameras':
        if isinstance(a, str):
            a = [t.strip() for t in a.split(',') if t.strip()]
        if isinstance(a, list) and isinstance(kept, list) and set(map(str, a)) == set(map(str, kept)):
            return out                       # the page keeps the rig's order
    elif kind == 'carla_live' and key == 'faults' and isinstance(a, list):
        a = [dict(at_s=f.get('at_s'), fault=f['fault'] if isinstance(f.get('fault'), dict) else
                  {k: v for k, v in f.items() if k != 'at_s'}) if isinstance(f, dict) else f for f in a]
        try:
            a.sort(key=lambda f: float(f['at_s']))
        except (TypeError, ValueError, KeyError):
            pass
    _compare(a, kept, '', out, ign)
    return out


def _changed_text(kind, key, changes, asked=None, kept=None, project=None) -> str:
    """What the page changed in a structured value, in words: 'F pitch 25–95° → F pitch 25–90°' (an item of a
    list, as asked and as kept), 'kept 1 of 2' (items it could not use), 'speed kmh 200 → 90' (a part of an object)."""
    bits, items = [], []
    for path, a, g in changes:
        m = re.match(r'\[(\d+)\]', path)
        if m and isinstance(asked, list) and isinstance(kept, list):
            i = int(m.group(1)) - 1
            if i not in items and i < len(asked) and i < len(kept):
                items.append(i)
                ta = (_item_texts(kind, key, [asked[i]], project, raw=True) or [generic_text(asked[i], 1)])[0]
                tg = (_item_texts(kind, key, [kept[i]], project) or [generic_text(kept[i], 1)])[0]
                bits.append(f'{ta} → {tg}' if ta != tg else f'{ta} ({path.split("]", 1)[1].lstrip(".") or "item"} '
                            f'{generic_text(a, 1)} → {generic_text(g, 1)})')
        elif not path:
            if isinstance(a, list) and isinstance(g, list):
                bits.append(f'kept {len(g)} of {len(a)} (the page drops what it cannot use)')
            else:
                bits.append(f'{value_text(kind, key, a, project)} → {value_text(kind, key, g, project)}')
        else:
            bits.append(f'{_human(path.replace("[", ".").replace("]", ""))} {generic_text(a, 1)} → '
                        f'{generic_text(g, 1)}')
    more = f' and {len(bits) - 3} more' if len(bits) > 3 else ''
    return '; '.join(bits[:3]) + more


# =========================================================================== steps and plans
@dataclass
class Step:
    kind: str
    label: str = ''
    settings: dict = field(default_factory=dict)
    sources: dict = field(default_factory=dict)      # flat key -> {'source', 'reason'}
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    estimate: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict:
        return {'kind': self.kind, 'label': self.label, 'settings': copy.deepcopy(self.settings),
                'sources': copy.deepcopy(self.sources), 'errors': list(self.errors), 'warnings': list(self.warnings),
                'notes': list(self.notes), 'estimate': copy.deepcopy(self.estimate),
                'settings_sha256': step_sha256(self.kind, self.settings)}

    @staticmethod
    def from_dict(d) -> 'Step':
        d = d if isinstance(d, dict) else {}
        return Step(str(d.get('kind') or ''), str(d.get('label') or ''), copy.deepcopy(d.get('settings') or {}),
                    copy.deepcopy(d.get('sources') or {}), list(d.get('errors') or []), list(d.get('warnings') or []),
                    list(d.get('notes') or []), copy.deepcopy(d.get('estimate') or {}))


def step_sha256(kind, settings) -> str:
    return sha256({'kind': kind, 'settings': settings})


def plan_sha256(steps) -> str:
    return sha256([{'kind': s.kind if isinstance(s, Step) else s['kind'],
                    'settings': s.settings if isinstance(s, Step) else s['settings']} for s in steps])


@dataclass
class Plan:
    id: str
    version: int = 1
    title: str = ''
    rationale: str = ''
    request: list = field(default_factory=list)     # the user's words that led to it, in order
    steps: list = field(default_factory=list)       # Step
    status: str = 'draft'
    provider: str = ''
    model: str = ''
    created_utc: str = ''
    parent: int | None = None                       # the version it revises
    diff: list = field(default_factory=list)        # against the parent: [{'step', 'key', 'before', 'after'} | ...]

    @property
    def ok(self) -> bool:
        return bool(self.steps) and all(s.ok for s in self.steps)

    @property
    def errors(self) -> list:
        return [f'Step {i + 1}: {e}' for i, s in enumerate(self.steps) for e in s.errors] + \
            ([] if self.steps else ['The plan has no steps.'])

    def settings_sha256(self) -> str:
        return plan_sha256(self.steps)

    def to_dict(self) -> dict:
        return {'schema': PLAN_SCHEMA, 'id': self.id, 'version': self.version, 'title': self.title,
                'rationale': self.rationale, 'request': list(self.request), 'status': self.status,
                'provider': self.provider, 'model': self.model, 'created_utc': self.created_utc,
                'parent': self.parent, 'diff': copy.deepcopy(self.diff),
                'steps': [s.to_dict() for s in self.steps], 'settings_sha256': self.settings_sha256()}

    @staticmethod
    def from_dict(d) -> 'Plan':
        d = d if isinstance(d, dict) else {}
        return Plan(str(d.get('id') or new_id()), int(d.get('version') or 1), str(d.get('title') or ''),
                    str(d.get('rationale') or ''), list(d.get('request') or []),
                    [Step.from_dict(s) for s in d.get('steps') or []], str(d.get('status') or 'draft'),
                    str(d.get('provider') or ''), str(d.get('model') or ''), str(d.get('created_utc') or ''),
                    d.get('parent'), copy.deepcopy(d.get('diff') or []))


def new_id() -> str:
    return 'P' + uuid.uuid4().hex[:7].upper()


def _limit_reason(kind, key, asked, got, project=None, changes=None) -> str:
    if changes and (isinstance(asked, (list, dict)) or isinstance(got, (list, dict))):
        return _cut('the page changed ' + _changed_text(kind, key, changes, asked, got, project), 240)
    lim = None
    if key.startswith('scenario.params.'):
        from ..sim import carla_bridge as CB
        spec = next((sp[2].get(key.split('.')[-1]) for sp in CB.SCENARIOS.values()
                     if key.split('.')[-1] in sp[2] and sp[2][key.split('.')[-1]][0] == 'num'), None)
        if spec is not None:
            return f'asked {_g(asked)}; the scenario allows {_g(spec[2])}–{_g(spec[3])}, so {_g(got)}'
    try:
        if kind == 'carla_run':
            from ..core import simrig as SR
            lim = SR.LIMITS.get(key)
        elif kind == 'sample_dataset':
            from ..core import dataset as DSET
            lim = DSET.LIMITS.get(key)
        elif kind == 'critical_carla':
            from ..core import critical_carla as CC
            lim = CC.LIMITS.get(key)
        elif kind == 'detection':
            from ..core import detection_eval as DE
            lim = DE.LIMITS.get(key)
        elif kind == 'experiment':
            from ..core import experiments as EX
            lim = EX.LIMITS.get(key)
        elif kind == 'campaign':
            from ..core import simcampaign as SC
            lim = SC.LIMITS.get(key)
        elif kind == 'carla_live':
            from ..core import simrig as SR
            lim = {'scale': (SR.LIMITS['image_scale'][0], 1.0), 'duration_s': (0.0, 3600.0)}.get(key)
        elif kind in STUDY_KINDS:
            lim = study_limits(kind).get(key)
    except Exception:                       # noqa: BLE001 — a reason is a nicety
        lim = None

    def say(v):
        return _g(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else json.dumps(v, default=str)[:80]
    a, g = say(asked), say(got)
    if lim is not None:
        return f'asked {a}; the page allows {_g(lim[0])}–{_g(lim[1])}, so {g}'
    return f'asked {a}; the page keeps {g}'


def resolve_step(project, kind, values=(), label='', base=None, base_sources=None, working_dir='',
                 engine=None, check=True) -> Step:
    """A step from the model's values [{'key', 'value', 'source', 'reason'}] over base (the previous version's
    settings, else the page's): each value parsed, set and cleaned by the page's settings class; a value the class
    changes is marked forced by a limit; a key the kind does not have is an error."""
    if kind not in KINDS:
        return Step(kind, label or kind, {}, {}, [f'Unknown step kind {kind!r} (one of {", ".join(KINDS)}).'])
    start = copy.deepcopy(base) if base is not None else defaults(kind, project)
    sources = copy.deepcopy(base_sources) if base_sources is not None else {}
    errors = []
    raw = copy.deepcopy(start)
    asked = {}
    expanded = set()                    # keys inside an object the request gave whole
    keys = allowed_keys(kind, start)
    vals = list(values or [])
    # a campaign template first: its design is the base the other values change
    for v in list(vals):
        if kind == 'campaign' and isinstance(v, dict) and v.get('key') == 'template':
            from ..core import simcampaign as SC
            name = parse_value(v.get('value'))
            if name in SC.TEMPLATES:
                raw = SC.template(name, project).to_dict()
                for k in flatten(raw):
                    sources[k] = {'source': v.get('source') if v.get('source') in ('request', 'copilot') else
                                  'copilot', 'reason': f'from the {SC.TEMPLATES[name][0]} template'}
                asked['template'] = name
            else:
                errors.append(f'No campaign template {name!r} (one of {", ".join(SC.TEMPLATES)}).')
            vals.remove(v)
    for v in vals:
        if not isinstance(v, dict):
            errors.append('A value is not an object {key, value, source, reason}.')
            continue
        key = str(v.get('key') or '').strip()
        if not key:
            errors.append('A value has no key.')
            continue
        if key not in keys:
            errors.append(f'{KIND_LABEL[kind]} has no setting {key!r} (get_plan_settings lists them).')
            continue
        val = parse_value(v.get('value'))
        if kind == 'carla_run' and key == 'weather' and isinstance(val, str):
            from ..core.weather import IDS as WEATHER_IDS
            if val in WEATHER_IDS:
                val = f'vantage:{val}'                  # a Vantage condition named bare
        if kind == 'carla_run' and key == 'scenario' and isinstance(val, str):
            val = {'name': val, 'params': {}}
        if kind == 'carla_run' and key == 'scenario.name' and val != get_path(raw, 'scenario.name'):
            set_path(raw, 'scenario.params', {})        # another scenario: its own parameters
        set_path(raw, key, val)
        src = v.get('source') if v.get('source') in ('request', 'copilot') else 'copilot'
        why = str(v.get('reason') or '').strip()[:240]
        if isinstance(val, dict) and val and key not in ('actors',):
            # an object (the scenario, the ground truth …): each of its values is the request's, checked on its own
            for fk, fv in flatten(val, key + '.').items():
                asked[fk] = fv
                sources[fk] = {'source': src, 'reason': why}
                expanded.add(fk)
        asked[key] = val
        sources[key] = {'source': src, 'reason': why}
    if kind == 'carla_run':
        _scenario_traffic(raw, asked, sources)
    settings, problems = clean(kind, raw, project, working_dir)
    errors += problems
    flat = flatten(settings)
    not_used = unused(kind, settings, project)
    for key, val in asked.items():
        if key == 'template':
            continue
        if kind == 'detection' and key == 'recordings' or (kind, key) in NAMED_KEYS:
            continue                    # named as the catalogue shows them, kept as full paths (variants as uids)
        got = get_path(settings, key, flat.get(key))
        if key.startswith('scenario.params.') and got is None:
            from ..sim import carla_bridge as CB
            got = CB.clean_scenario(settings['scenario']['name'], settings['scenario'].get('params'))[0][
                'params'].get(key.split('.')[-1])
        if isinstance(val, dict) and val and key in sources and any(k.startswith(key + '.') for k in asked):
            continue                    # its values are checked one by one
        if got is None and key in expanded:
            continue                    # a part of the object the page does not keep (it ignores unknown keys)
        if got is None and not isinstance(val, (dict, list)) and val is not None:
            errors.append(f'{key}: {json.dumps(val, default=str)[:60]} is not accepted by the page.')
            continue
        if unused_of(key, not_used):
            continue                    # a setting this design does not use: never "forced", whatever it holds
        m = meaning(kind, key, val, got, project)
        if m['changed']:
            sources[key] = {'source': 'limit', 'reason': _limit_reason(kind, key, val, got, project, m['changed'])}
        elif m['ignored'] and key in sources:
            names = ', '.join(dict.fromkeys(k.split('.')[-1].split('[')[0] for k in m['ignored']))
            r = sources[key].get('reason', '')
            sources[key]['reason'] = _cut((r + ' · ' if r else '') + f'the page does not use {names}', 240)
    for k in flatten(settings):
        sources.setdefault(k, {'source': 'default', 'reason': 'as set on the page' if getattr(
            project.analysis, KIND_SLOT.get(kind) or '_', None) else 'Vantage\'s default'})
    for k in list(sources):
        if k not in flatten(settings) and k not in ('faults', 'actors', 'factors', 'responses', 'recordings',
                                                     'fault_sets', 'formats', 'cameras', 'seeds', 'weathers',
                                                     'scenarios', 'evaluators') and \
                not k.startswith('scenario.params.') and k != 'template':
            sources.pop(k)
    if kind in STUDY_KINDS:
        study_sources(kind, settings, asked, sources, project)
    st = Step(kind, (label or KIND_LABEL[kind]).strip()[:120], settings, sources, errors)
    if check:
        check_step(project, st, working_dir=working_dir, engine=engine)
    return st


def _scenario_traffic(raw, asked, sources):
    """A CARLA drive's traffic asked for with a scenario: a scenario has its own Other traffic (the page's Traffic is a
    free drive's, and the bridge uses the scenario's), so the request's number goes there unless that was asked too."""
    from ..sim import carla_bridge as CB
    if 'traffic' not in asked or 'scenario.params.traffic' in asked or raw.get('actors'):
        return
    name = get_path(raw, 'scenario.name') or 'free_drive'
    if 'traffic' not in CB.SCENARIOS.get(name, ('', '', {}))[2]:
        return
    val = asked['traffic']
    set_path(raw, 'scenario.params.traffic', val)
    asked['scenario.params.traffic'] = val
    src = sources.get('traffic', {}).get('source', 'copilot')
    sources['scenario.params.traffic'] = {'source': src, 'reason': f'the traffic asked for: a '
                                          f'{_scenario_label(name).lower()} has its own Other traffic'}


def build(project, raw_steps, title='', rationale='', prev: Plan | None = None, request='', provider='', model='',
          working_dir='', engine=None, revise=False) -> Plan:
    """A plan from propose_plan's steps. revise (a follow-up): each step starts from the previous version's step at
    the same place when it is of the same kind, so the values only say what changes; the new version keeps the id."""
    steps = []
    for i, rs in enumerate(raw_steps or []):
        rs = rs if isinstance(rs, dict) else {}
        kind = str(rs.get('kind') or '')
        base = base_src = None
        if revise and prev is not None and i < len(prev.steps) and prev.steps[i].kind == kind:
            base, base_src = prev.steps[i].settings, prev.steps[i].sources
        st = resolve_step(project, kind, rs.get('values') or [], rs.get('label') or
                          (prev.steps[i].label if base is not None else ''), base, base_src, working_dir, engine)
        steps.append(st)
    if prev is not None and (revise or prev.status == 'draft'):
        pid, ver = prev.id, prev.version + 1
        req = list(prev.request) + ([request] if request and request not in prev.request else [])
    else:
        pid, ver, req = new_id(), 1, ([request] if request else [])
    p = Plan(pid, ver, (title or (prev.title if prev is not None and revise else '') or 'Plan').strip()[:120],
             (rationale or (prev.rationale if prev is not None and revise else '')).strip()[:600], req, steps,
             'draft', provider, model, utc_now(), prev.version if prev is not None and pid == prev.id else None)
    if prev is not None and pid == prev.id:
        p.diff = diff(prev, p)
    return p


def diff(a: Plan, b: Plan) -> list:
    """What changed from version a to version b: [{'step': i, 'key', 'before', 'after'}], steps added or removed as
    {'step': i, 'added' | 'removed': kind}."""
    out = []
    for i in range(max(len(a.steps), len(b.steps))):
        sa = a.steps[i] if i < len(a.steps) else None
        sb = b.steps[i] if i < len(b.steps) else None
        if sa is None:
            out.append({'step': i, 'added': sb.kind, 'label': sb.label})
            continue
        if sb is None:
            out.append({'step': i, 'removed': sa.kind, 'label': sa.label})
            continue
        if sa.kind != sb.kind:
            out += [{'step': i, 'removed': sa.kind, 'label': sa.label}, {'step': i, 'added': sb.kind,
                                                                          'label': sb.label}]
            continue
        fa, fb = flatten(sa.settings), flatten(sb.settings)
        for k in sorted(set(fa) | set(fb)):
            if k not in fa or k not in fb or not same(fa[k], fb[k]):
                out.append({'step': i, 'key': k, 'before': fa.get(k), 'after': fb.get(k)})
    return out


def edit_value(project, plan: Plan, step: int, key: str, value, working_dir='', engine=None) -> Plan:
    """The user changed one value on the plan card: a new version (an approved plan is never changed), the value
    from the user (source request), cleaned and checked by the page's own rules."""
    if not 0 <= step < len(plan.steps):
        raise IndexError(step)
    s = plan.steps[step]
    vals = [{'key': key, 'value': json.dumps(value), 'source': 'request', 'reason': 'you changed it on the plan'}]
    ns = resolve_step(project, s.kind, vals, s.label, s.settings, s.sources, working_dir, engine)
    steps = [Step.from_dict(x.to_dict()) for x in plan.steps]
    steps[step] = ns
    for i, x in enumerate(steps):
        if i != step:
            check_step(project, x, working_dir=working_dir, engine=engine)
    p = Plan(plan.id, plan.version + 1, plan.title, plan.rationale, list(plan.request), steps, 'draft', plan.provider,
             plan.model, utc_now(), plan.version)
    p.diff = diff(plan, p)
    return p


# =========================================================================== checks and estimates
def mem_available_gb() -> float | None:
    """MemAvailable from /proc/meminfo (Linux), GB; None elsewhere."""
    try:
        with open('/proc/meminfo', encoding='ascii') as fh:
            for ln in fh:
                if ln.startswith('MemAvailable:'):
                    return int(ln.split()[1]) / 1e6 * 1.024
    except (OSError, ValueError, IndexError):
        return None
    return None


def _port_open(port) -> bool:
    from ..sim import server as SV
    try:
        return bool(SV.port_open('127.0.0.1', int(port)))
    except Exception:                       # noqa: BLE001
        return False


def _check_ports(project, port, tm_port, errors, warnings, notes, server_cap_gb=None):
    from ..core import simrig as SR
    own = SR.settings_of(project)
    lo, hi = RESERVED_PORTS
    if lo <= int(port) <= hi and int(port) != int(own.port):
        errors.append(f'Port {port} is kept for a CARLA server of your own (2000–2010): choose another, e.g. 2400.')
    if int(tm_port) == 8100 and int(own.tm_port) != 8100:
        errors.append('Traffic manager port 8100 is kept for a CARLA server of your own: choose another.')
    if _port_open(port):
        notes.append(f'A CARLA server answers on port {port}: the run uses it.')
        return False
    cap = f', capped at {server_cap_gb:g} GB' if server_cap_gb else ''
    notes.append(f'No server answers on port {port}: the CARLA page starts its own there (headless{cap}).')
    return True


def _check_memory(need_gb, server_gb, will_start, errors, warnings):
    free = mem_available_gb()
    if free is None:
        return
    total = need_gb + (server_gb if will_start else 0.0)
    if free < need_gb:
        errors.append(f'MemAvailable is {free:.1f} GB; the bridge alone needs about {need_gb:.1f} GB: close other '
                      'programs or lower the image scale.')
    elif free < total:
        warnings.append(f'MemAvailable is {free:.1f} GB; the run needs about {total:.1f} GB with the server '
                        f'({server_gb:g} GB cap) the page starts.')


def _weather_warnings(weather: str, warnings):
    w = str(weather or '')
    low = w.lower()
    if 'rain' in low or 'fog' in low or 'storm' in low:
        warnings.append('CARLA\'s lidar and radar ignore rain and fog: they show in the camera images only (rain as '
                        'water on the lens).')
    if 'snow' in low:
        warnings.append('CARLA has no snow: a snow condition gives its light and fog only.')


def _carla_rig(project, settings):
    from ..core import simrig as SR
    st = SR.SimSettings.from_dict(settings)
    return SR.build(project, st, tables=False), st


PROBE_SCALE = 0.25                  # a rig built this small predicts the render cost of a bigger image scale


def _render_mpx(project, settings, scale):
    from ..sim import carla_bridge as CB
    rig, _ = _carla_rig(project, dict(settings, image_scale=scale))
    return CB.render_mpx(rig.doc, None), rig


def _fit_scale(project, settings, budget, probe=None):
    """The largest image scale (5 % steps) whose cameras render within the budget (checked by building the rig at it),
    or None. probe: (scale, Mpx) of a rig built small; the cost grows with the square of the scale."""
    from ..core import simrig as SR
    lo = SR.LIMITS['image_scale'][0]
    s0, m0 = probe if probe else (PROBE_SCALE, _render_mpx(project, settings, PROBE_SCALE)[0])
    sc = min(float(settings.get('image_scale', 0.5)), math.floor(s0 * math.sqrt(budget / max(m0, 1e-6)) * 20) / 20)
    while sc >= lo - 1e-9:
        mpx, rig = _render_mpx(project, settings, round(sc, 2))
        if mpx <= budget:
            return round(sc, 2), (mpx, rig)
        sc -= 0.05
    return None, None


def _measured_run_rate(project, working_dir='', gt=True):
    """(wall seconds per simulated second, runs) from the CARLA drives in the project's runs folder (run.json's
    performance), the ones with ground truth or without as asked; (None, 0) when there are none."""
    from .. import workdir
    root = workdir.project_dir(project, _working(project, working_dir), 'carla')
    rates = []
    try:
        names = sorted(os.listdir(root))[-40:]
    except OSError:
        return None, 0
    for n in names:
        try:
            with open(os.path.join(root, n, 'run.json'), encoding='utf-8') as fh:
                doc = json.load(fh)
        except (OSError, ValueError):
            continue
        perf = doc.get('performance') or {}
        has_gt = os.path.isfile(os.path.join(root, n, 'ground_truth', 'index.json'))
        if doc.get('status') == 'ok' and perf.get('simulated_s') and perf.get('wall_s') and has_gt == gt:
            rates.append(float(perf['wall_s']) / float(perf['simulated_s']))
    if not rates:
        return None, 0
    rates.sort()
    return rates[len(rates) // 2], len(rates)


def check_step(project, st: Step, working_dir='', engine=None, budget=None):
    """Validate a step as the app will run it (errors stop approval), add the documented limits that apply
    (warnings) and its estimate."""
    k, s = st.kind, st.settings
    errors, warnings, notes = [], [], []
    est = {}
    try:
        if k == 'carla_run':
            _check_carla_run(project, st, errors, warnings, notes, est, working_dir, budget)
        elif k == 'campaign':
            _check_campaign(project, s, errors, warnings, notes, est)
        elif k == 'detection':
            _check_detection(project, s, errors, warnings, notes, est, working_dir)
        elif k == 'critical_carla':
            _check_critical(project, s, errors, warnings, notes, est)
        elif k == 'carla_live':
            _check_live(project, s, errors, warnings, notes, est)
        elif k == 'experiment':
            _check_experiment(project, s, errors, warnings, notes, est, engine)
        elif k == 'sample_dataset':
            _check_dataset(project, s, errors, warnings, notes, est)
        elif k == 'study':
            est.update(text='computes in a few seconds on a 10 cm grid (writes nothing)')
        elif k == 'export':
            from ..core import exporters as ex
            est.update(formats=len(s.get('formats') or []), text=f'{len(s.get("formats") or [])} formats into '
                       + (s.get('folder') or 'the working folder\'s exports'))
        elif k in STUDY_KINDS:
            check_study(project, st, errors, warnings, notes, est, working_dir)
    except ValueError as e:
        errors.append(str(e))
    except Exception as e:                  # noqa: BLE001 — a check that fails says so; it never passes silently
        errors.append(f'The check failed: {type(e).__name__}: {e}')
    st.errors = list(dict.fromkeys(st.errors + errors))
    st.warnings = list(dict.fromkeys(warnings))
    st.notes = list(dict.fromkeys(notes))
    st.estimate = est
    return st


def _hm(s) -> str:
    s = int(round(float(s)))
    return f'{s // 3600} h {s % 3600 // 60:02d} min' if s >= 3600 else f'{s // 60} min {s % 60:02d} s'


def _check_carla_run(project, st, errors, warnings, notes, est, working_dir, budget):
    from ..core import dataset as DSET
    from ..core import simcampaign as SC
    from ..core import simrig as SR
    from ..sim import carla_bridge as CB
    s = st.settings
    budget = CB.RENDER_BUDGET_MPX if budget is None else budget
    scale = float(s.get('image_scale', 0.5))
    probe = None
    if scale > 2 * PROBE_SCALE:             # a big rig takes seconds to build: predict its cost from a small one first
        m0, _ = _render_mpx(project, s, PROBE_SCALE)
        probe = (PROBE_SCALE, m0)
        predicted = m0 * (scale / PROBE_SCALE) ** 2
    if probe is not None and predicted > 1.05 * budget:
        mpx, rig, sim = predicted, None, SR.SimSettings.from_dict(s)
    else:
        rig, sim = _carla_rig(project, s)
        mpx = CB.render_mpx(rig.doc, None)
    if mpx > budget:
        src = st.sources.get('image_scale', {}).get('source')
        fit, fitted = _fit_scale(project, s, budget, probe)
        if src != 'request' and fit is not None:
            was = s['image_scale']
            s['image_scale'] = fit
            st.sources['image_scale'] = {'source': 'limit', 'reason': f'image scale {fit:g}: {was:g} with ground '
                                                                      f'truth renders {mpx:.0f} Mpx a frame, over '
                                                                      f'the render budget of {budget:g}'}
            warnings.append(f'Image scale {fit:g}: at {was:g} the cameras with their ground truth exceed the render '
                            f'budget (about {mpx:.0f} of {budget:g} Mpx a frame).')
            mpx, rig = fitted
            sim = SR.SimSettings.from_dict(s)
        else:
            errors.append(f'The cameras would render about {mpx:.0f} Mpx a frame with their ground truth, over the '
                          f'render budget of {budget:g}' + (f': an image scale of {fit:g} fits.' if fit else
                                                            ': lower the image scale or the ground truth.'))
            if rig is None:
                return
    elif mpx > SR.SERVER_MPX_WARN:
        warnings.append(f'{mpx:.0f} Mpx a frame: above {SR.SERVER_MPX_WARN:g} the server may run out of memory.')
    will_start = _check_ports(project, sim.port, sim.tm_port, errors, warnings, notes, sim.memory_gb)
    need = float(rig.doc['simulation'].get('bridge_memory_gb') or 0.0)
    _check_memory(need, float(sim.memory_gb), will_start, errors, warnings)
    w = sim.weather
    if w == 'project':
        cond = (project.analysis.weather or {}).get('condition', '')
        w = f'vantage:{cond}' if cond else ''
    _weather_warnings(w, warnings)
    warnings.append(LIMITS_DOC['carla_vehicle'])
    if sim.faults:
        warnings.append('The drive is deliberately faulty: ' + LIMITS_DOC['faults'])
    gt_on = any(sim.ground_truth.values())
    # the estimate: frames from the rig's cameras, CARLA's time from measured drives, the disk from the sample
    # dataset's estimator for the same rig, duration, rates and image scale
    cams = rig.doc.get('cameras') or []
    dur = float(sim.duration_s)
    frames = 0
    for c in cams:
        hz = float((c.get('capture') or {}).get('recorded_hz') or 0.0)
        frames += int(round(dur * hz)) if hz else 0
    rate, n_runs = _measured_run_rate(project, working_dir, gt_on)
    cls = 'full' if gt_on and sim.ground_truth.get('range') else 'instance' if gt_on else 'none'
    per_s, over = (rate, SC.COST_DEFAULT[cls][1]) if rate else SC.COST_DEFAULT[cls]
    wall = over + per_s * (dur + float(sim.warmup_s))
    de = DSET.estimate(project, DSET.DatasetSettings.from_dict({'duration_s': min(dur, DSET.LIMITS['duration_s'][1]),
                                                               'image_scale': sim.image_scale,
                                                               'camera_rate': sim.camera_rate,
                                                               'keyframe_hz': sim.keyframe_hz,
                                                               'nuscenes': sim.nuscenes, 'rosbag': sim.rosbag}))
    scale_dur = dur / min(dur, DSET.LIMITS['duration_s'][1]) if dur else 1.0
    disk = float(de.get('bytes') or 0.0) * scale_dur
    gt_bytes = 0.0
    if gt_on:
        px = sum(int(c['image']['width']) * int(c['image']['height']) for c in cams)
        per_px = (1 if sim.ground_truth.get('semantic') else 0) + 2 * bool(sim.ground_truth.get('instance')) + \
            2 * bool(sim.ground_truth.get('range'))
        gt_bytes = px * per_px * (frames / max(len(cams), 1)) if cams else 0.0
    est.update(render_mpx=round(mpx, 1), frames=frames, labelled_frames=frames if gt_on else 0,
               carla_s=round(wall, 1), disk_mb=round((disk + gt_bytes) / 1e6, 1),
               basis=(f'CARLA time from {n_runs} drive' + ('s' if n_runs != 1 else '') + ' measured on this machine'
                      if rate else 'CARLA time from the development machine\'s figures (RTX 5080, measured drives '
                                   'replace them)') + '; disk from the sample dataset\'s estimator for the same rig '
                                                      '(ground truth as uncompressed PNG, an upper bound)',
               text=f'{dur:g} s drive · about {_hm(wall)} of CARLA · {frames} camera frames'
                    + (' with ground truth' if gt_on else '') + f' · about {(disk + gt_bytes) / 1e6:,.0f} MB · '
                    f'{mpx:.0f} Mpx a frame')
    for n in (rig.doc.get('notes') or [])[:4]:
        notes.append(str(n))


def _check_campaign(project, s, errors, warnings, notes, est):
    from ..core import detection as DT
    from ..core import detection_eval as DE
    from ..core import simcampaign as SC
    from ..core import simrig as SR
    st = SC.CampaignSettings.from_dict(s)
    pl = SC.plan(project, st)
    for m in pl.problems:
        (warnings if 'only the first' in m else errors).append(m)
    if not any(d['role'] == 'drive' for d in pl.drives):
        errors.append('The campaign has nothing to drive as designed.')
    if 'detection' in st.evaluators:
        model = DE.DetectionSettings.from_dict(dict(DE.settings_of(project).to_dict(), **(st.detection or {}))).model
        ds = DT.status(model)
        if not ds['verified']:
            errors.append('The detection check needs the detector model, which is not downloaded: ' +
                          LIMITS_DOC['detector'])
    if 'asbuilt' in st.evaluators:
        warnings.append(LIMITS_DOC['asbuilt_carla'])
    base = SR.SimSettings.from_dict(dict(SR.settings_of(project).to_dict(), **(st.base or {})))
    _check_ports(project, base.port, base.tm_port, errors, warnings, notes, base.memory_gb)
    for w in list(st.weathers) or [base.weather]:
        _weather_warnings(w, warnings)
    notes.extend(pl.notes[:4])
    e = SC.estimate(project, st, pl)
    est.update(drives=e['drives'], done=e['done'], carla_s=e['drive_s'], remaining_s=e['remaining_s'],
               evaluate_s=e['evaluate_s'], basis=e['basis'], text=e['text'])


def _check_detection(project, s, errors, warnings, notes, est, working_dir):
    from .. import workdir
    from ..core import detection as DT
    from ..core import detection_eval as DE
    st = DE.DetectionSettings.from_dict(s)
    if not st.recordings:
        errors.append('Name the recordings to evaluate (get_catalogue topic recordings).')
    frames = 0
    for r in st.recordings:
        i = DE.recording_info(r)
        if not i['ok']:
            errors.append(f'{workdir.display(r, _working(project, working_dir))}: {i["text"]}')
        else:
            frames += int(i.get('frames') or 0)
    ds = DT.status(st.model)
    if not ds['verified']:
        errors.append('The detector model is not downloaded or does not verify: ' + LIMITS_DOC['detector'])
    from ..core import simcampaign as SC
    secs = SC.EVAL_COST_S['detection'] * frames / max(int(st.frame_step or 1), 1)
    est.update(images=frames, seconds=round(secs, 1), basis='0.2 s per image (the campaign cost model)',
               text=f'{len(st.recordings)} recording' + ('s' if len(st.recordings) != 1 else '') +
                    f', {frames} camera frames, about {_hm(secs)} of detection')


def _check_critical(project, s, errors, warnings, notes, est):
    from ..core import critical as CR
    from ..core import critical_carla as CC
    from ..core import simcampaign as SC
    from ..core import simrig as SR
    st = CC.CriticalCarlaSettings.from_dict(s)
    targets = CR.settings_of(project).targets
    on = [t for t in targets if t.get('enabled')]
    if not on:
        errors.append('There are no critical targets to stand in CARLA (Analysis ▸ Critical targets).')
    elif len(on) < len(targets):
        notes.append(f'{len(on)} of {len(targets)} critical targets are enabled: the others are not driven.')
    pl = CC.plan(project, st)
    if on and not pl.drives:
        errors.append('None of the targets can be stood in CARLA.')
    notes.extend(pl.notes[:4])
    sim = SR.settings_of(project)
    _check_ports(project, sim.port, sim.tm_port, errors, warnings, notes, sim.memory_gb)
    warnings.append(LIMITS_DOC['carla_vehicle'])
    per_s, over = SC.COST_DEFAULT['full']
    wall = len(pl.drives) * (over + per_s * (float(st.duration_s) + float(sim.warmup_s)))
    est.update(targets=len(on), drives=len(pl.drives), carla_s=round(wall, 1),
               text=f'{len(on)} targets in {len(pl.drives)} drives, about {_hm(wall)} of CARLA')


def _check_live(project, s, errors, warnings, notes, est):
    from ..core import simrig as SR
    sim = SR.settings_of(project)
    if not s.get('cameras'):
        errors.append('Choose at least one camera to stream.')
    _check_ports(project, sim.port, sim.tm_port, errors, warnings, notes, sim.memory_gb)
    warnings.append(LIMITS_DOC['live'])
    n = len(s.get('faults') or [])
    est.update(cameras=len(s.get('cameras') or []), faults=n,
               text=f'{len(s.get("cameras") or [])} cameras at {s.get("scale", LIVE_SCALE) * 100:g} %'
                    + (f' for {s["duration_s"]:g} s' if s.get('duration_s') else ' until you stop it')
                    + (f', {n} fault' + ('s' if n != 1 else '') + ' on time' if n else ''))


def _check_experiment(project, s, errors, warnings, notes, est, engine):
    from ..core import experiments as EX
    st = EX.ExperimentSettings.from_dict(s)
    errors.extend(EX.errors(project, st))
    n = EX.point_count(EX.factors_of(st), st)
    if any(r.study in EX.CARLA_STUDY_S for r in EX.responses_of(st, project)):
        warnings.append(LIMITS_DOC['experiment_carla'])
    est.update(points=n, text=f'{n:,} design points ({EX.DESIGN_LABEL.get(st.design, st.design)})')


def _check_dataset(project, s, errors, warnings, notes, est):
    from ..core import dataset as DSET
    st = DSET.DatasetSettings.from_dict(s)
    e = DSET.estimate(project, st)
    if not e.get('ok'):
        errors.append(e.get('reason') or 'Nothing to record in this variant.')
    warnings.append(LIMITS_DOC['sample_dataset'])
    est.update(frames=e.get('frames'), samples=e.get('samples'), disk_mb=round(float(e.get('bytes') or 0) / 1e6, 1),
               seconds=e.get('seconds'), text=f'{st.duration_s:g} s at {st.speed_kmh:g} km/h, about '
                                              f'{float(e.get("bytes") or 0) / 1e6:,.0f} MB in about '
                                              f'{_hm(e.get("seconds") or 0)}')


def revalidate(project, plan: Plan, working_dir='', engine=None) -> Plan:
    """The plan checked again now (before approval and before it runs): its settings are cleaned again; a step whose
    settings the page would no longer keep as they are is an error."""
    out = Plan.from_dict(plan.to_dict())
    for s in out.steps:
        cleaned, problems = clean(s.kind, s.settings, project, working_dir)
        s.errors = list(problems)
        if not same(cleaned, s.settings):
            s.errors.append('The settings no longer clean to themselves (the project or the app changed): propose '
                            'the plan again.')
        before = dict(s.settings)
        check_step(project, s, working_dir=working_dir, engine=engine)
        if not same(before, s.settings):
            s.errors.append('A check changed a value (e.g. the render budget): review the plan again.')
            s.settings = before
    return out


# =========================================================================== records and the gate
def records(project) -> list:
    d = getattr(project.analysis, SLOT, None) or {}
    return d.get('plans') if isinstance(d.get('plans'), list) else []


def find(project, plan_id, version=None) -> dict | None:
    rs = [r for r in records(project) if isinstance(r, dict) and r.get('id') == plan_id and
          (version is None or int(r.get('version') or 0) == int(version))]
    return rs[-1] if rs else None


def verify(rec: dict) -> bool:
    """The record still holds what was approved: its steps hash to the stored settings hash."""
    try:
        return rec.get('schema') == RECORD_SCHEMA and rec.get('status') in ('approved', 'running', 'done', 'failed',
                                                                            'cancelled') and \
            bool(rec.get('approved_utc')) and plan_sha256(rec['steps']) == rec.get('settings_sha256') and \
            all(step_sha256(s['kind'], s['settings']) == s.get('settings_sha256') for s in rec['steps'])
    except (KeyError, TypeError):
        return False


def record(plan: Plan, provider='', model='') -> dict:
    """The approval record of a plan (to store in project.analysis.copilot['plans'])."""
    d = plan.to_dict()
    return {'schema': RECORD_SCHEMA, 'id': plan.id, 'version': plan.version, 'title': plan.title,
            'rationale': plan.rationale, 'request': list(plan.request), 'provider': provider or plan.provider,
            'model': model or plan.model, 'steps': [{k: s[k] for k in ('kind', 'label', 'settings', 'sources',
                                                                       'warnings', 'notes', 'estimate',
                                                                       'settings_sha256')} for s in d['steps']],
            'settings_sha256': d['settings_sha256'], 'approved_utc': utc_now(), 'status': 'approved', 'runs': []}


def approve(project, plan: Plan, provider='', model='', working_dir='', engine=None) -> tuple[dict | None, list]:
    """(the record, []) when the plan can be approved: it validates again now; else (None, the errors). The caller
    stores the record (add_record) in the same edit that applies the first step's settings."""
    if plan.status not in ('draft',):
        return None, [f'This plan is {plan.status}: an approved plan cannot be approved again (edit it for a new '
                      'version).']
    again = revalidate(project, plan, working_dir, engine)
    if not again.ok:
        return None, again.errors
    return record(again, provider, model), []


def add_record(project, rec: dict):
    """Store an approval record (newest last, at most MAX_RECORDS kept; earlier versions of the same plan that were
    never run are marked superseded)."""
    d = dict(getattr(project.analysis, SLOT, None) or {})
    rs = [r for r in d.get('plans') or [] if isinstance(r, dict)]
    rs.append(copy.deepcopy(rec))
    d['plans'] = rs[-MAX_RECORDS:]
    setattr(project.analysis, SLOT, d)
    return d['plans'][-1]


def ticket(rec: dict, step: int) -> dict:
    """What a page's Run receives from a plan: the plan, its version and the step."""
    return {'plan': rec['id'], 'version': int(rec['version']), 'step': int(step)}


def gate(project, tk, kind: str, settings: dict) -> tuple[bool, str]:
    """May a page run this, reached from a plan? Only when the project holds an approved, intact record of that plan
    version whose step is of this kind with exactly these settings. (ok, why not)."""
    if not isinstance(tk, dict) or not tk.get('plan'):
        return False, 'No approved plan: a plan runs only after you approve it.'
    rec = find(project, tk.get('plan'), tk.get('version'))
    if rec is None:
        return False, f'No approval record of plan {tk.get("plan")} v{tk.get("version")} in this project.'
    if not verify(rec):
        return False, f'The approval record of plan {rec.get("id")} v{rec.get("version")} does not verify (it was ' \
                      'changed after approval or never approved).'
    i = int(tk.get('step', -1))
    if not 0 <= i < len(rec['steps']):
        return False, f'Plan {rec["id"]} has no step {i + 1}.'
    s = rec['steps'][i]
    if s['kind'] != kind:
        return False, f'Step {i + 1} of plan {rec["id"]} is a {KIND_LABEL.get(s["kind"], s["kind"])}, not a ' \
                      f'{KIND_LABEL.get(kind, kind)}.'
    want = s['settings']
    have = settings
    if kind == 'carla_live':                # the page holds the cameras and the scale; the runner times the faults
        want = {k: want.get(k) for k in ('cameras', 'scale')}
        have = {k: settings.get(k) for k in ('cameras', 'scale')}
    if not same(want, have):
        fa, fb = flatten(want), flatten(have)
        keys = [k for k in sorted(set(fa) | set(fb)) if not same(fa.get(k), fb.get(k))]
        return False, (f'The page\'s settings differ from the approved plan ({", ".join(keys[:4])}): approve the '
                       'changed plan first.')
    return True, ''


def start_run(rec: dict, via='app') -> dict:
    run = {'started_utc': utc_now(), 'finished_utc': None, 'via': via, 'status': 'running',
           'steps': [{'status': 'pending', 'message': '', 'output': '', 'summary': None} for _ in rec['steps']]}
    rec.setdefault('runs', []).append(run)
    rec['status'] = 'running'
    return run


def finish_run(rec: dict, status: str):
    run = (rec.get('runs') or [None])[-1]
    if run is not None:
        run['status'] = status
        run['finished_utc'] = utc_now()
    rec['status'] = status


def run_folder(project, rec: dict, working_dir='') -> str:
    from .. import workdir
    return workdir.project_dir(project, _working(project, working_dir), 'copilot', f'{rec["id"]}-v{rec["version"]}')


def write_run_file(project, rec: dict, folder: str, step: int | None = None, summary=None) -> str:
    """copilot_plan.json in folder: the record (and the step's summary), stamped with its provenance, which names
    the plan (core.provenance's 'plan' field)."""
    from ..core import provenance as PV
    os.makedirs(folder, exist_ok=True)
    doc = {'schema': RUN_SCHEMA, 'plan': {'id': rec['id'], 'version': rec['version'], 'title': rec['title'],
                                          'settings_sha256': rec['settings_sha256'], 'approved_utc':
                                              rec['approved_utc'], 'provider': rec['provider'], 'model':
                                              rec['model']},
           'request': list(rec.get('request') or []), 'steps': copy.deepcopy(rec['steps']),
           'step': step, 'summary': PV.scrub(summary) if summary is not None else None,
           'runs': copy.deepcopy(rec.get('runs') or [])}
    path = os.path.join(folder, 'copilot_plan.json')
    doc = PV.stamp(doc, project, study='copilot_plan', plan=ticket(rec, step if step is not None else -1))
    tmp = path + '.part'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(doc, fh, indent=1, default=str)
    os.replace(tmp, path)
    return path


def stamp_outputs(project, rec, step, paths, folder):
    """Stamp the JSON a step wrote (run.json, dataset.json …) with a provenance block naming the plan."""
    from ..core import provenance as PV
    for p in paths:
        if p and os.path.isfile(p):
            PV.stamp_path(p, project, output={'name': os.path.relpath(p, folder).replace(os.sep, '/'),
                                              'in_folder': True}, plan=ticket(rec, step))


# =========================================================================== for the model
def for_model(plan: Plan, working_dir='') -> dict:
    """The plan as the model is told it: no machine paths (recordings shown as the catalogue does)."""
    from ..core import provenance as PV
    from .. import workdir
    steps = []
    for i, s in enumerate(plan.steps):
        st = copy.deepcopy(s.settings)
        if s.kind == 'detection':
            st['recordings'] = [workdir.display(r, working_dir or None) for r in st.get('recordings') or []]
        if s.kind in STUDY_KINDS:
            st = study_for_model(s.kind, st, working_dir)
        if s.kind == 'campaign' and st.get('out_dir'):
            st['out_dir'] = workdir.display(st['out_dir'], working_dir or None)
        un = unused(s.kind, s.settings, None)
        chosen = {k: v for k, v in s.sources.items() if v.get('source') != 'default' and not unused_of(k, un)}
        steps.append({'step': i + 1, 'kind': s.kind, 'label': s.label, 'ok': s.ok, 'errors': s.errors,
                      'warnings': s.warnings, 'notes': s.notes, 'estimate': s.estimate,
                      'values_set': {k: {'value': get_path(st, k, flatten(st).get(k)), **v} for k, v in
                                     chosen.items()},
                      'not_used': {k: f'{lab}: {why}' for k, (lab, why) in un.items()
                                   if k in flatten(s.settings) or any(x.startswith(k + '.') for x in
                                                                     flatten(s.settings))}})
    return PV.scrub({'plan': plan.id, 'version': plan.version, 'status': plan.status, 'title': plan.title,
                     'ok': plan.ok, 'steps': steps, 'changed_from_previous_version': plan.diff[:60]})


# =========================================================================== round 14's studies as plan kinds
# Each study kind's settings are the page's own settings (its core module's settings class, cleaned by its from_dict)
# plus a few plan-only options (STUDY_EXTRAS: where the recordings come from, whether to hand the perception records
# to the learned model, ...). The plan keeps all of them; when the step runs, the page's part is merged into the
# page's slot (apply_study_settings, one undo step in the app) and run_study_kind runs the module headless.
STUDY_SLOT = {'rig_perception': 'rig_perception', 'foundation_range': 'foundation',
              'robust_perception': 'robust_perception', 'redrive': 'redrive', 'design': 'designer',
              'perf_model_train': 'perf_model', 'perf_model_campaign': 'perf_model', 'risk_coverage': 'risk_coverage',
              'fleet_twin': 'fleet_twin'}
STUDY_MODULE = {'rig_perception': ('rig_perception', 'RigPerceptionSettings'),
                'foundation_range': ('foundation_range', 'FoundationSettings'),
                'robust_perception': ('robust_perception', 'RobustSettings'),
                'redrive': ('redrive', 'RedriveSettings'), 'design': ('rig_designer', 'DesignerSettings'),
                'perf_model_train': ('perf_model', 'PerfModelSettings'),
                'perf_model_campaign': ('perf_model', 'PerfModelSettings'),
                'risk_coverage': ('risk_coverage', 'RiskSettings'), 'fleet_twin': ('fleet_twin', 'FleetTwinSettings')}
STUDY_EXTRAS = {
    'rig_perception': {'source': 'recordings', 'records': False, 'focus': []},
    'foundation_range': {}, 'robust_perception': {},
    'redrive': {'drive': '', 'reshoot': False},
    'design': {'requirements_from': 'page'},
    'perf_model_train': {'save': True},
    'perf_model_campaign': {'variants': [], 'rounds': 1, 'duration_s': 0.0},
    'risk_coverage': {},
    'fleet_twin': {'action': 'run'},
}
STUDY_EXTRA_CHOICES = {('rig_perception', 'source'): ('recordings', 'carla'),
                       ('design', 'requirements_from'): ('page', 'brief'),
                       ('fleet_twin', 'action'): ('run', 'watch_once')}
STUDY_EXTRA_LIMITS = {'perf_model_campaign': {'rounds': (1, 5), 'duration_s': (0.0, 60.0)}}
# the designer's history and documents are the page's, never a plan's
STUDY_EXCLUDE = {'design': ('documents', 'not_checkable', 'rejected', 'extraction', 'accepted', 'approvals')}
# set by name (a recording as the catalogue shows it, a variant by its name): kept as paths and uids
NAMED_KEYS = {('rig_perception', 'recordings'), ('rig_perception', 'scene_set.variants'),
              ('foundation_range', 'recordings'), ('robust_perception', 'recordings'), ('redrive', 'drive'),
              ('redrive', 'paired_drive'), ('redrive', 'variant'), ('perf_model_campaign', 'variants')}
STUDY_EMPTY_TEXT = {('perf_model_campaign', 'variants'): 'every variant with a camera',
                    ('rig_perception', 'scene_set.variants'): 'the active variant',
                    ('rig_perception', 'focus'): 'every sector', ('design', 'requirements'): 'none',
                    ('design', 'constraints'): 'none', ('foundation_range', 'cameras'): 'every camera'}
# what each kind is for, and requests the deterministic planner maps to it (get_catalogue plan_kinds)
STUDY_ABOUT = {
    'rig_perception': ('3D detection per class and region around the car, rig against rig on the same scenes, with '
                       'intervals; on recordings with ground truth, or a perception scene set driven in CARLA '
                       '(source carla)', 'compare pedestrian detection of rig A and B on the left'),
    'foundation_range': ('how far each detector finds each class with each camera (R50, R90) and how far monocular '
                         'depth is usable; on recordings, or a range sweep in CARLA (source sweep)',
                         'how far can F see a cyclist'),
    'robust_perception': ('the mounting, thermal, vibration and calibration tolerances carried to detection: rig '
                          'recall with its band over builds, what drives the spread', 'what if the mounts are 2 mm off'),
    'redrive': ('a recorded drive re-rendered through a proposed rig: how much of the data survives per camera '
                '(reuse, partial, re-collect); optionally the re-shot drive', 'can we reuse last month\'s drive with '
                                                                               'the new rig'),
    'design': ('the design agent: 3–7 complete rigs from the accepted requirements (or the brief, read by the parser),'
               ' never applied: you create an option on its page', 'design a rig for this brief'),
    'perf_model_train': ('train the learned detection model on the perception records and save it',
                         'train the detection model'),
    'perf_model_campaign': ('active learning: the model proposes k variants, each driven in CARLA and evaluated by rig '
                            'perception, the model retrained after each round', 'which rigs should we simulate next'),
    'risk_coverage': ('coverage weighted by where crashes start (NHTSA pre-crash scenarios, Euro NCAP)',
                      'risk-weighted coverage with the NCAP weights'),
    'fleet_twin': ('the fleet twin: drift per vehicle and sensor, knocks, the recalibration schedule; or watch the '
                   'vehicles\' folders once (action watch_once)', 'which car needs recalibrating'),
}
# measured costs the estimates use (each from the round-14 report that measured it, on the development machine)
COST_SCENE_SET_DRIVE_S = 390.0       # bev: about 6.5 min of wall time per 12 s six-camera drive with ground truth
COST_SWEEP_DRIVE_S = 63.0            # foundation: 60 sweep drives in 63 min (Town01, one still frame)
COST_FOUNDATION_EVAL_S = 0.39        # foundation: 555 s for 1,440 image × model evaluations on the RTX 5080
COST_REDRIVE_S = 350.0               # redrive, 12 s drive on the GPU: scene 188 s, evidence 59, calibration 70, report 33
COST_RESHOOT_S, COST_RESHOOT_MB = 376.0, 747.0      # redrive: the re-shot export of the 12 s drive
COST_DESIGN_S_PER_RIG = 1.1          # designer: 2–3 min for 160 rigs scored on the data rig
COST_DESIGN_SETUP_S = 35.0           # copilot15: 65 s for 12 rigs with a 30 s budget on the 6-cam sample (loaded machine)
COST_RISK_S = 2.0                    # robust: about 2 s per variant
COST_FLEET_S = 5.0                   # fleettwin: 4.2–4.6 s for fleets up to 20 vehicles × 12 sensors × 24 records
COST_TRAIN_S_PER_RECORD = 0.2        # copilot15: 1.8–4.4 s for 24 records with cross-validation on this machine
# the most a study's export takes on the sample (measured by tests/test_copilot_r15.py; the estimate says "at most")
DISK_MB = {'rig_perception': 4.0, 'foundation_range': 20.0, 'robust_perception': 4.0, 'redrive': 30.0, 'design': 2.0,
           'perf_model_train': 2.0, 'perf_model_campaign': 2.0, 'risk_coverage': 3.0, 'fleet_twin': 2.0}
LIDAR_HZ = 10.0                      # the bridge's instants: the reference lidar's sweeps


def _study_mod(kind):
    import importlib
    return importlib.import_module(f'..core.{STUDY_MODULE[kind][0]}', __package__)


def _study_cls(kind):
    mod = _study_mod(kind)
    return mod, getattr(mod, STUDY_MODULE[kind][1])


def study_part(kind, settings) -> dict:
    """The page's part of a study step's settings (without the plan-only options)."""
    ex = STUDY_EXTRAS[kind]
    return {k: copy.deepcopy(v) for k, v in (settings or {}).items() if k not in ex}


def study_defaults(kind, project) -> dict:
    mod = _study_mod(kind)
    d = mod.settings_of(project).to_dict()
    for k in STUDY_EXCLUDE.get(kind, ()):
        d.pop(k, None)
    d.update(copy.deepcopy(STUDY_EXTRAS[kind]))
    return clean_study(kind, d, project)[0]


def study_limits(kind) -> dict:
    out = {}
    try:
        out.update(getattr(_study_mod(kind), 'LIMITS', {}) or {})
        if kind == 'rig_perception':
            from ..core import perception_scenes as PS
            out.update({f'scene_set.{k}': v for k, v in PS.LIMITS.items()})
    except Exception:                       # noqa: BLE001 — a range is a nicety
        pass
    out.update(STUDY_EXTRA_LIMITS.get(kind, {}))
    return out


def find_variant(project, name):
    """A variant by uid, name (any case) or 1-based index; None when there is none."""
    n = str(name or '').strip()
    if not n:
        return None
    for i, v in enumerate(project.variants):
        if n in (v.uid, v.name, str(i + 1)) or n.lower() == v.name.lower():
            return v
    return None


def _variant_uids(project, names, bad, key) -> list:
    out = []
    for n in names if isinstance(names, (list, tuple)) else [names]:
        v = find_variant(project, n)
        if v is None:
            bad.append(f'{key}: no variant {str(n)!r} (variants: {", ".join(x.name for x in project.variants)}).')
        elif v.uid not in out:
            out.append(v.uid)
    return out


def _bool(v, default):
    if isinstance(v, bool):
        return v
    if isinstance(v, str) and v.strip().lower() in ('1', 'true', 'yes', 'on', '0', 'false', 'no', 'off'):
        return v.strip().lower() in ('1', 'true', 'yes', 'on')
    if isinstance(v, (int, float)) and v in (0, 1):
        return bool(v)
    return default


def _clean_extras(kind, d, project, working_dir, bad) -> dict:
    out = {}
    for k, default in STUDY_EXTRAS[kind].items():
        v = d.get(k, default)
        ch = STUDY_EXTRA_CHOICES.get((kind, k))
        if ch is not None:
            if v not in ch:
                bad.append(f'{k} must be one of {", ".join(ch)}.')
                v = default
        elif isinstance(default, bool):
            v = _bool(v, default)
        elif k == 'focus':
            from ..core import regions as RG
            names = RG.default_grid(project).sector_names() if project is not None else list(RG.SECTOR_NAMES_8)
            vv = [v] if isinstance(v, str) else (v if isinstance(v, (list, tuple)) else [])
            got = []
            for x in vv:
                t = str(x).strip().lower().replace(' ', '-')
                if t in names:
                    if t not in got:
                        got.append(t)
                elif t:
                    bad.append(f'focus: no sector {str(x)!r} (sectors: {", ".join(names)}).')
            v = [n for n in names if n in got]
        elif k == 'variants':
            v = _variant_uids(project, v or [], bad, 'variants') if project is not None else []
        elif k == 'drive':
            v = _resolve_recording(v, project, working_dir) if str(v or '').strip() else ''
        elif k in ('rounds', 'duration_s'):
            lo, hi = STUDY_EXTRA_LIMITS[kind][k]
            x = _num(v, lo, hi)
            x = default if x is None else x
            v = int(round(x)) if k == 'rounds' else (0.0 if x < 2.0 else round(float(x), 3))
        out[k] = v
    return out


def _read_brief(raw, project, bad):
    """requirements_from brief: the brief read by Vantage's deterministic parser (designer_llm.deterministic), the
    rows typed on the page (origin user) kept, as the page's Read does."""
    from ..core import docs_ingest as DI
    from . import designer_llm as DL
    brief = raw.get('brief') if isinstance(raw.get('brief'), str) else ''
    if not brief.strip():
        bad.append('requirements_from brief: the brief is empty (set brief to the text to read).')
        return
    ex = DL.deterministic([DI.from_text(brief)], project)
    keep_r = [r for r in raw.get('requirements') or [] if isinstance(r, dict) and r.get('origin') == 'user']
    keep_c = [c for c in raw.get('constraints') or [] if isinstance(c, dict) and c.get('origin') == 'user']
    raw['requirements'] = keep_r + list(ex.requirements)
    raw['constraints'] = keep_c + list(ex.constraints)
    if not ex.requirements and not ex.constraints:
        bad.append('The parser read no requirement and no constraint from the brief: say what the rig must see, how '
                   'far, and its limits (cost, power, cameras), or accept requirements on Analysis ▸ Design agent.')


def clean_study(kind, d, project, working_dir='') -> tuple[dict, list]:
    """A study step's settings as its page's settings class keeps them, plus the plan-only options, and problems."""
    bad = []
    d = d if isinstance(d, dict) else {}
    _mod, cls = _study_cls(kind)
    extras = STUDY_EXTRAS[kind]
    raw = {k: copy.deepcopy(v) for k, v in d.items() if k not in extras and k not in STUDY_EXCLUDE.get(kind, ())}
    ex = _clean_extras(kind, {k: d[k] for k in extras if k in d}, project, working_dir, bad)
    if kind == 'rig_perception':
        from ..core import perception_scenes as PS
        if isinstance(raw.get('recordings'), (list, tuple)):
            recs = []
            for r in raw['recordings']:
                r = {'path': r} if isinstance(r, str) else r
                if isinstance(r, dict) and str(r.get('path') or '').strip():
                    recs.append({'path': _resolve_recording(r['path'], project, working_dir),
                                 'rig': str(r.get('rig') or '').strip()})
            raw['recordings'] = recs
        ss = dict(raw.get('scene_set')) if isinstance(raw.get('scene_set'), dict) else {}
        if 'variants' in ss and project is not None:
            ss['variants'] = _variant_uids(project, ss['variants'] or [], bad, 'scene_set.variants')
        raw['scene_set'] = PS.SceneSetSettings.from_dict(ss).to_dict()
    elif kind in ('foundation_range', 'robust_perception'):
        if isinstance(raw.get('recordings'), (list, tuple)):
            raw['recordings'] = [_resolve_recording(r, project, working_dir) for r in raw['recordings']
                                 if isinstance(r, str) and r.strip()]
    elif kind == 'redrive':
        if isinstance(raw.get('paired_drive'), str) and raw['paired_drive'].strip():
            raw['paired_drive'] = _resolve_recording(raw['paired_drive'], project, working_dir)
        if isinstance(raw.get('variant'), str) and raw['variant'].strip() and project is not None:
            uids = _variant_uids(project, [raw['variant']], bad, 'variant')
            raw['variant'] = uids[0] if uids else ''
    elif kind == 'design' and ex.get('requirements_from') == 'brief':
        _read_brief(raw, project, bad)
    out = cls.from_dict(raw).to_dict()
    for k in STUDY_EXCLUDE.get(kind, ()):
        out.pop(k, None)
    if kind == 'rig_perception':
        from ..core import perception_scenes as PS
        out['scene_set'] = PS.SceneSetSettings.from_dict(out.get('scene_set')).to_dict()
    out.update(ex)
    return out, bad


def _base(p) -> str:
    return os.path.basename(str(p).rstrip('/\\')) or str(p)


def _req_text(r) -> str:
    it = (r or {}).get('item') or {}
    if it.get('text'):
        return str(it['text'])
    return f'{it.get("metric", "?")} {it.get("op", "")} {_g(it.get("target"))}'.strip()


def study_item_texts(kind, key, value, project=None) -> list | None:
    if kind == 'rig_perception' and key == 'recordings':
        return [(_base(r.get('path', '')) + (f' as {r["rig"]}' if r.get('rig') else '')) if isinstance(r, dict)
                else _base(r) for r in value]
    if key in ('variants', 'scene_set.variants') and project is not None:
        return [next((v.name for v in project.variants if v.uid == u), str(u)) for u in value]
    if kind == 'design' and key == 'requirements':
        return [_req_text(r) for r in value]
    if kind == 'design' and key == 'constraints':
        return [str(c.get('text') or f'{c.get("kind")} {c.get("op")} {_g(c.get("value"))} {c.get("unit", "")}')
                for c in value if isinstance(c, dict)]
    return None


def study_value_text(kind, key, value, project=None) -> str | None:
    if key in ('drive', 'paired_drive', 'scene_set.out_dir') and value:
        return _base(value)
    if kind == 'redrive' and key == 'variant':
        if not value:
            return 'the log replay\'s rig B (else the active variant)'
        v = find_variant(project, value) if project is not None else None
        return v.name if v is not None else value
    return None


VIEW_KEYS = ('view_fov_deg', 'view_pitch_deg', 'view_yaws', 'view_width', 'view_aspect')
SWEEP_KEYS = ('sweep_cameras', 'sweep_classes', 'ranges_m', 'offsets_deg', 'road_lateral_m', 'seeds', 'town',
              'weather', 'image_scale', 'duration_s', 'sweep_frames')


def study_unused(kind, s, project=None) -> dict:
    out = {}
    if kind == 'rig_perception':
        if s.get('source', 'recordings') == 'recordings':
            out['scene_set'] = ('not used without a CARLA drive', 'the study runs on the recordings named')
        else:
            out['recordings'] = ('replaced by the scene set\'s drives', 'source carla drives the scene set first')
        if s.get('views', 'fisheye') == 'fisheye':
            for k in VIEW_KEYS:
                out[k] = ('not used with fisheye images', 'no pinhole views are cut from the images')
    elif kind == 'foundation_range':
        if s.get('source', 'recordings') == 'recordings':
            for k in SWEEP_KEYS:
                out[k] = ('not used without a sweep', 'the study runs on the recordings named')
        else:
            out['recordings'] = ('not used by a sweep', 'the sweep drives its own scenes in CARLA')
        if s.get('views') == 'fisheye':
            for k in VIEW_KEYS:
                out[k] = ('not used with fisheye images', 'no pinhole views are cut from the images')
    elif kind == 'redrive':
        if s.get('schedule', 'recorded') == 'recorded':
            out['rate_hz'] = ('not used with the recorded schedule', 'the drive\'s own frame times are used')
        if 'paired' not in (s.get('sources') or []):
            out['paired_drive'] = ('not used without the paired source', 'sources does not include paired')
        if 'loo' not in (s.get('sources') or []):
            out['loo_camera'] = ('not used without the leave-one-out source', 'sources does not include loo')
    elif kind == 'fleet_twin' and s.get('action', 'run') == 'run':
        out['watch_quiet_s'] = ('not used without a watch', 'action run fits the twin on the records there are')
    return out


def study_sources(kind, settings, asked, sources, project=None):
    """What the plan's sources say beyond the values asked: the requirements a brief gave."""
    if kind != 'design':
        return
    n_r, n_c = len(settings.get('requirements') or []), len(settings.get('constraints') or [])
    if settings.get('requirements_from') == 'brief':
        why = f'read from the brief by Vantage\'s parser: {n_r} requirement' + ('s' if n_r != 1 else '') + f', {n_c} constraint' + ('s' if n_c != 1 else '') + ' (approving accepts them)'
        src = sources.get('brief', {}).get('source', 'request')
        for k in ('requirements', 'constraints'):
            sources[k] = {'source': src if src in ('request', 'copilot') else 'request', 'reason': why}


def study_for_model(kind, st, working_dir=''):
    from .. import workdir
    wd = working_dir or None
    st = copy.deepcopy(st)
    if kind == 'rig_perception':
        st['recordings'] = [dict(r, path=workdir.display(r['path'], wd)) for r in st.get('recordings') or []]
        if (st.get('scene_set') or {}).get('out_dir'):
            st['scene_set']['out_dir'] = workdir.display(st['scene_set']['out_dir'], wd)
    if kind in ('foundation_range', 'robust_perception'):
        st['recordings'] = [workdir.display(r, wd) for r in st.get('recordings') or []]
    for k in ('drive', 'paired_drive'):
        if st.get(k):
            st[k] = workdir.display(st[k], wd)
    return st


def apply_study_settings(project, kind, settings):
    """Merge the page's part of a study step into the page's slot (the drive of a re-drive into the log replay's), as
    the page itself would keep it. The caller makes it one undo step."""
    part = study_part(kind, settings)
    if kind == 'fleet_twin':
        from ..core import fleet_twin as FT
        FT.set_settings(project, **part)
        return
    slot = STUDY_SLOT[kind]
    cur = getattr(project.analysis, slot, None)
    new = dict(cur) if isinstance(cur, dict) else {}
    new.update(part)
    setattr(project.analysis, slot, new)
    if kind == 'redrive' and (settings or {}).get('drive'):
        project.analysis.splat = dict(project.analysis.splat or {}, drive=settings['drive'])


# --------------------------------------------------------------------------- checks and estimates
def _carla_root_found() -> bool:
    try:
        from ..core import simcampaign as SC
        if hasattr(SC, '_carla_root'):
            return bool(SC._carla_root())
        from ..sim import server as SV
        return bool(SV.candidate_roots())
    except Exception:                       # noqa: BLE001
        return False


def _carla_ready(project, errors, warnings, notes, port=None, tm_port=None) -> bool:
    """The CARLA a study needs: a server on the port (or a CARLA folder to start one from) and CARLA's client Python;
    errors say what is missing. True when the app starts the server."""
    from ..core import simcampaign as SC
    from ..core import simrig as SR
    sim = SR.settings_of(project)
    port, tm = int(port or sim.port), int(tm_port or sim.tm_port)
    will_start = _check_ports(project, port, tm, errors, warnings, notes, sim.memory_gb)
    if not SC.carla_python():
        errors.append('This needs CARLA, and no CARLA client Python is set: choose the CARLA folder on Simulate ▸ '
                      'CARLA (or run it on recordings you already have).')
    elif will_start and not _carla_root_found():
        errors.append(f'This needs CARLA: no server answers on port {port} and no CARLA folder is set to start one '
                      '(Simulate ▸ CARLA).')
    warnings.append(LIMITS_DOC['carla_vehicle'])
    return will_start


def _detector_ok(model, errors, why=''):
    from ..core import detection as DT
    s = DT.status(model)
    if not s['verified']:
        errors.append((why or 'The 2D detector model') + ' is not downloaded or does not verify: ' +
                      LIMITS_DOC['detector'])
    return s['verified']


def _instants(path, rig='', mode='all', step=1):
    """(instants, cameras) of a recording with ground truth, or raise ValueError saying why it cannot be read."""
    from ..core import rig_perception as RP
    rec = RP.PerceptionRecording(path, rig)
    return len(rec.instants(mode, max(int(step or 1), 1))), len(rec.cams), rec


def _needs(est, carla=False, gpu='', torch=False, models=()):
    est['needs'] = {'carla': bool(carla), 'gpu': gpu or 'no', 'torch': bool(torch), 'models': list(models)}


def _rec_errors(project, paths, errors, working_dir, mode='all', step=1, rigs=None, reader='bev'):
    """(camera images, instants, rig labels) of the recordings; errors say which cannot be read. reader 'bev': the
    rig perception reader (nuScenes annotations); 'frames': detection_eval's (the camera frames' labels), as robust
    perception and the foundation-model study read them."""
    from .. import workdir
    images, frames, labels = 0, 0, []
    for i, p in enumerate(paths):
        if reader == 'frames':
            from ..core import detection_eval as DE
            info = DE.recording_info(p)
            if not info['ok']:
                errors.append(f'{workdir.display(p, _working(project, working_dir))}: {info["text"]}')
                continue
            images += int(info.get('frames') or 0) // max(int(step or 1), 1)
            frames += int(info.get('frames') or 0)
            labels.append(str(info.get('variant') or ''))
            continue
        try:
            n, cams, rec = _instants(p, (rigs or {}).get(p, ''), mode, step)
        except (ValueError, OSError, KeyError) as e:
            errors.append(f'{workdir.display(p, _working(project, working_dir))}: {e}')
            continue
        images += n * cams
        frames += n
        labels.append(rec.rig)
    return images, frames, labels


def check_study(project, st, errors, warnings, notes, est, working_dir=''):
    {'rig_perception': _chk_rig_perception, 'foundation_range': _chk_foundation, 'robust_perception': _chk_robust,
     'redrive': _chk_redrive, 'design': _chk_design, 'perf_model_train': _chk_train,
     'perf_model_campaign': _chk_campaign, 'risk_coverage': _chk_risk,
     'fleet_twin': _chk_fleet}[st.kind](project, st.settings, errors, warnings, notes, est, working_dir)


def _chk_rig_perception(project, s, errors, warnings, notes, est, working_dir):
    from ..core import perception_scenes as PS
    from ..core import rig_perception as RP
    from ..core import simcampaign as SC
    rs = RP.RigPerceptionSettings.from_dict(study_part('rig_perception', s))
    mode = 'keyframes' if rs.instants == 'keyframes' else 'all'
    carla, carla_s, images = False, 0.0, 0
    if s.get('source') == 'carla':
        carla = True
        ss = PS.SceneSetSettings.from_dict(rs.scene_set)
        e = PS.estimate(project, ss)
        for m in e.get('problems') or []:
            (warnings if 'only the first' in m else errors).append(m)
        if not e.get('drives'):
            errors.append('The scene set has nothing to drive for these variants.')
        _carla_ready(project, errors, warnings, notes, ss.port or None, ss.tm_port or None)
        warnings.append(LIMITS_DOC['scene_set'])
        todo = max(int(e.get('drives') or 0) - int(e.get('done') or 0), 0)
        dur = float(ss.duration_s or PS.scene_set(ss.set_id)['duration_s'])
        # simcampaign's cost model (this machine's measured drives where there are any), else bev's measured drive
        carla_s = float(e.get('remaining_s') or 0.0) or todo * COST_SCENE_SET_DRIVE_S * dur / 12.0
        vs = [v for v in project.variants if v.uid in (ss.variants or [project.variant.uid])]
        cams = sum(sum(1 for c in v.cameras if c.visible) for v in vs) / max(len(vs), 1)
        images = int(int(e.get('drives') or 0) * dur * LIDAR_HZ * cams / max(rs.frame_step, 1))
        notes.append(f'Scene set {ss.set_id}: {e.get("text", "")}')
        rigs = [v.name for v in vs]
    else:
        if not rs.recordings:
            errors.append('Name the recordings with ground truth (get_catalogue topic recordings), or set source '
                          'carla to drive a perception scene set.')
        images, _f, rigs = _rec_errors(project, [r['path'] for r in rs.recordings], errors, working_dir, mode,
                                       rs.frame_step, {r['path']: r.get('rig') or '' for r in rs.recordings})
    if len(set(rigs)) < 2:
        notes.append('One rig: its figures with intervals, no rig-against-rig comparison.')
    models = []
    if rs.backend == 'lifted':
        models.append(rs.model)
        _detector_ok(rs.model, errors)
    else:
        notes.append(f'{rs.backend[9:]}: the external detector set up on this computer (checked when it runs).')
    warnings.append(LIMITS_DOC['perception_detector'])
    if s.get('records'):
        notes.append('Its perception records are added to the learned model\'s store.')
    if s.get('focus'):
        notes.append('The answer looks at the ' + _and(s['focus']) + ' sectors; the study measures every region.')
    secs = images * SC.EVAL_COST_S['detection']
    _needs(est, carla=carla, models=models)
    est.update(images=images, seconds=round(secs, 1), carla_s=round(carla_s, 1), disk_mb=DISK_MB['rig_perception'],
               basis='detection 0.2 s per image (the campaign cost model; seconds when the 2D detections are cached)'
                     + ('; the scene set\'s drives by the campaign cost model (measured drives on this machine where '
                        'there are any; about 6.5 min per 12 s six-camera drive measured on the standard set)'
                        if carla else ''),
               text=(f'about {_hm(carla_s)} of CARLA, then ' if carla else '') +
                    f'{images:,} camera images, about {_hm(secs)} of detection · at most '
                    f'{DISK_MB["rig_perception"]:g} MB')


def _chk_foundation(project, s, errors, warnings, notes, est, working_dir):
    from ..core import detection as DT
    from ..core import foundation_models as FM
    from ..core import foundation_range as FR
    fs = FR.FoundationSettings.from_dict(study_part('foundation_range', s))
    if not fs.detectors and not fs.depth_model:
        errors.append('Choose at least one detector or a depth model.')
    models = list(fs.detectors) + ([fs.depth_model] if fs.depth_model else [])
    torch_need = any(m in FM.SPECS for m in models)
    if torch_need:
        why = FM.torch_missing()
        if why:
            errors.append(f'{why} (the foundation models need it).')
    for mid in models:
        if mid in DT.MODELS:
            _detector_ok(mid, errors, DT.status(mid).get('name') or mid)
            continue
        if mid in FM.SPECS and not FM.allowed(mid, fs.opt_in):
            errors.append(f'{FM.SPECS[mid].name}: {FM.SPECS[mid].licence_note}')
        st_ = FM.status(mid, verify=False)
        if not st_.get('present'):
            errors.append(f'{FM.model_label(mid) if hasattr(FM, "model_label") else mid} is not downloaded: '
                          + LIMITS_DOC['foundation_models'])
    carla, carla_s, images = False, 0.0, 0
    if fs.source == 'sweep':
        carla = True
        pl = FR.sweep_plan(project, fs)
        n = len(pl.get('drives') or [])
        if not n:
            errors.append('Nothing to sweep: choose sweep cameras of the variant and classes CARLA can stand in.')
        _carla_ready(project, errors, warnings, notes)
        carla_s = n * COST_SWEEP_DRIVE_S
        cams = [c for c in project.variant.cameras if c.visible and c.tag in fs.sweep_cameras]
        images = n * len(cams) * int(fs.sweep_frames)
        notes.append(f'{len(pl.get("placements") or [])} placements in {n} drives (Town {fs.town}, {fs.weather}).')
    else:
        if not fs.recordings:
            errors.append('Name the recordings with ground truth, or set source sweep to stand the classes at known '
                          'ranges in CARLA.')
        imgs, _f, _l = _rec_errors(project, fs.recordings, errors, working_dir, 'all', fs.frame_step,
                                   reader='frames')
        images = imgs
    kinds = (1 if fs.views in ('fisheye', 'both') else 0) + (len(fs.view_yaws) if fs.views in ('pinhole', 'both')
                                                               else 0)
    evals = images * max(kinds, 1) * max(len(models), 1)
    secs = evals * COST_FOUNDATION_EVAL_S
    warnings.append(LIMITS_DOC['foundation_models'])
    _needs(est, carla=carla, gpu='recommended' if torch_need else 'no', torch=torch_need, models=models)
    est.update(images=images, evaluations=evals, seconds=round(secs, 1), carla_s=round(carla_s, 1),
               disk_mb=DISK_MB['foundation_range'],
               basis='0.39 s per image and model on the RTX 5080 (the round-14 sweep study)'
                     + ('; sweep drives about 63 s each (measured: 60 drives in 63 min)' if carla else ''),
               text=(f'about {_hm(carla_s)} of CARLA, then ' if carla else '') +
                    f'{evals:,} image × model evaluations, about {_hm(secs)} on a GPU · at most '
                    f'{DISK_MB["foundation_range"]:g} MB')


def _chk_robust(project, s, errors, warnings, notes, est, working_dir):
    from ..core import detection_eval as DE
    from ..core import robust_perception as RB
    from ..core import simcampaign as SC
    rb = RB.RobustSettings.from_dict(study_part('robust_perception', s))
    if not rb.recordings:
        errors.append('Name the recordings with ground truth (CARLA runs of this rig: get_catalogue topic '
                      'recordings); a CARLA drive with ground truth makes one.')
    images, _f, _l = _rec_errors(project, rb.recordings, errors, working_dir, 'all', rb.frame_step, reader='frames')
    models = []
    if rb.detect:
        model = DE.settings_of(project).model
        models.append(model)
        _detector_ok(model, errors)
    act = rb.active()
    if not act:
        errors.append('Every tolerance source is off: switch on at least one (mounting, thermal, vibration, '
                      'calibration).')
    builds = 1 + rb.samples + rb.oat_samples * len(act)
    secs = builds * images * SC.EVAL_COST_S['detection'] if rb.detect else 0.0
    notes.append(f'Sources: {", ".join(act) or "none"}; {builds} builds with the detector'
                 + (f', Sobol n = {rb.sobol_n} for the BEV figures' if rb.sobol_n else ''))
    warnings.append('One CARLA drive per recording: the spread is the build variation, not scene sampling.')
    _needs(est, models=models)
    est.update(images=images, builds=builds, seconds=round(secs, 1), disk_mb=DISK_MB['robust_perception'],
               basis='detection 0.2 s per image per build (the campaign cost model)',
               text=f'{builds} builds × {images:,} images, about {_hm(secs)} of detection · at most '
                    f'{DISK_MB["robust_perception"]:g} MB')


def _chk_redrive(project, s, errors, warnings, notes, est, working_dir):
    from .. import workdir
    from ..core import redrive as RDV
    from ..core import splat_reproject as SP
    why = SP.torch_missing()
    if why:
        errors.append(why)
    drive = s.get('drive') or SP.settings_of(project).drive
    if not drive:
        errors.append('Choose the recorded drive (drive: a CARLA run, a nuScenes folder or a ROS 2 bag), or open it '
                      'on Simulate ▸ Log replay.')
    elif not os.path.exists(drive):
        errors.append(f'{workdir.display(drive, _working(project, working_dir))}: not found.')
    st = RDV.RedriveSettings.from_dict(study_part('redrive', s))
    v = RDV.proposed_variant(project, st, SP.settings_of(project))
    if v is None or not any(c.visible for c in v.cameras):
        errors.append('The proposed rig has no visible camera.')
    else:
        notes.append(f'Proposed rig: {v.name}.')
    secs = COST_REDRIVE_S + (COST_RESHOOT_S if s.get('reshoot') else 0.0)
    disk = DISK_MB['redrive'] + (COST_RESHOOT_MB if s.get('reshoot') else 0.0)
    warnings.append(LIMITS_DOC['redrive'])
    _needs(est, gpu='recommended', torch=True)
    est.update(seconds=secs, disk_mb=disk, basis='a 12 s CARLA drive on the RTX 5080 (scene 188 s, evidence 59 s, '
                                                 'calibration 70 s, report 33 s; the re-shot export 376 s and 747 MB): '
                                                 'it grows with the drive\'s length',
               text=f'about {_hm(secs)} on a GPU for a 12 s drive' + (', with the re-shot drive' if s.get('reshoot')
                                                                      else '') + f' · about {disk:,.0f} MB')


def _chk_design(project, s, errors, warnings, notes, est, working_dir):
    from ..core import rig_designer as DSG
    ds = DSG.DesignerSettings.from_dict(study_part('design', s))
    if not ds.items():
        errors.append('No accepted requirement: read them from a brief (requirements_from brief, with the brief) or '
                      'accept them on Analysis ▸ Design agent.')
    if not ds.prices:
        warnings.append('No prices set: every option\'s cost is incomplete (Analysis ▸ Design agent ▸ Prices).')
    warnings.append(LIMITS_DOC['design_options'])
    secs = COST_DESIGN_SETUP_S + min(float(ds.budget_s), ds.evaluations * COST_DESIGN_S_PER_RIG)
    _needs(est)
    est.update(seconds=round(secs, 1), rigs=ds.evaluations, options=ds.options, disk_mb=DISK_MB['design'],
               basis='about 1.1 s per rig scored (2–3 min for 160 on the data rig), the search stopped by the budget; '
                     'about 35 s of set-up and final checks on top (measured)',
               text=f'{len(ds.items())} requirements · {ds.evaluations} rigs scored for {ds.options} options, about '
                    f'{_hm(secs)} · nothing changes until you create an option')


def _store_count(project, errors):
    from ..core import perf_model as PM
    try:
        return len(PM.store_for(project).records())
    except Exception as e:                  # noqa: BLE001 — an unreadable store is said
        errors.append(f'The perception-record store cannot be read: {e}')
        return 0


def _chk_train(project, s, errors, warnings, notes, est, working_dir):
    from ..core import perf_model as PM
    pm = PM.PerfModelSettings.from_dict(study_part('perf_model_train', s))
    n = _store_count(project, errors)
    if n < pm.min_records:
        errors.append(f'Training needs at least {pm.min_records} perception records; the store has {n} (Rig '
                      'perception, Foundation models and Robust perception write them).')
    warnings.append(LIMITS_DOC['perf_model'])
    if s.get('save'):
        notes.append('A trained model is saved where the app loads it (it replaces the saved one).')
    secs = n * COST_TRAIN_S_PER_RECORD
    _needs(est)
    est.update(records=n, seconds=round(secs, 1), disk_mb=DISK_MB['perf_model_train'],
               basis='about 0.2 s a record with cross-validation (24 records in 1.8–4.4 s, measured on the '
                     'development machine); it grows faster than the records for large stores',
               text=f'{n} records, about {_hm(secs)} of training')


def _campaign_candidates(project, s):
    uids = s.get('variants') or []
    return [v for v in project.variants if (not uids or v.uid in uids) and any(c.visible for c in v.cameras)]


def _chk_campaign(project, s, errors, warnings, notes, est, working_dir):
    from ..core import perf_model as PM
    from ..core import rig_perception as RP
    from ..core import simcampaign as SC
    from ..core import simrig as SR
    pm = PM.PerfModelSettings.from_dict(study_part('perf_model_campaign', s))
    cands = _campaign_candidates(project, s)
    if not cands:
        errors.append('No variant with a camera to evaluate.')
    rounds = int(s.get('rounds') or 1)
    n_eval = min(len(cands), pm.k * rounds)
    model = RP.RigPerceptionSettings().model
    _detector_ok(model, errors)
    _carla_ready(project, errors, warnings, notes)
    n = _store_count(project, errors)
    if n < pm.min_records:
        notes.append(f'The store has {n} records (fewer than {pm.min_records}): the first round spreads its picks over '
                     'the rigs by geometry, the model trains once there are enough.')
    sim = SR.settings_of(project)
    dur = float(s.get('duration_s') or sim.duration_s)
    per_s, over = SC.COST_DEFAULT['full']
    wall = over + per_s * (dur + float(sim.warmup_s))
    cams = sum(sum(1 for c in v.cameras if c.visible) for v in cands) / max(len(cands), 1)
    images = int(n_eval * dur * LIDAR_HZ * cams)
    ev = images * SC.EVAL_COST_S['detection']
    carla_s = n_eval * wall
    notes.append(f'{n_eval} of {len(cands)} variants evaluated: {rounds} round' + ('s' if rounds != 1 else '') +
                 f' of {pm.k}, the model retrained after each and saved.')
    warnings.append(LIMITS_DOC['perf_model'])
    _needs(est, carla=True, models=[model])
    est.update(rigs=n_eval, seconds=round(ev, 1), carla_s=round(carla_s, 1), disk_mb=DISK_MB['perf_model_campaign'],
               basis='the campaign cost model per drive (full ground truth) and 0.2 s per image of detection',
               text=f'{n_eval} rigs driven {dur:g} s each, about {_hm(carla_s)} of CARLA and {_hm(ev)} of detection')


def _chk_risk(project, s, errors, warnings, notes, est, working_dir):
    if not any(c.visible for c in project.variant.cameras):
        errors.append('The variant has no visible camera.')
    _needs(est)
    est.update(seconds=COST_RISK_S, disk_mb=DISK_MB['risk_coverage'], basis='about 2 s per variant (round 14)',
               text=f'about {COST_RISK_S:g} s · cameras only, clear weather')


def _chk_fleet(project, s, errors, warnings, notes, est, working_dir):
    from ..core import fleet as FL
    from ..core import fleet_twin as FT
    if not FL.vehicles(project):
        errors.append('The project has no fleet: add the vehicles and their records on Calibrate ▸ Fleet.')
    if s.get('action') == 'watch_once':
        if not FT.watches(project):
            errors.append('No watch folder is set: set one per vehicle on Calibrate ▸ Fleet twin ▸ Watch.')
        warnings.append(LIMITS_DOC['fleet_watch'])
        notes.append('The drives read are added to the fleet\'s observations (one undo step), then the twin is fitted.')
    _needs(est)
    est.update(seconds=COST_FLEET_S, disk_mb=DISK_MB['fleet_twin'],
               basis='4–5 s for fleets up to 20 vehicles × 12 sensors × 24 records (round 14); a watched drive adds '
                     'its online calibration (about 33 s a drive)',
               text=f'about {COST_FLEET_S:g} s' + (' plus about 33 s per new drive' if s.get('action') == 'watch_once'
                                                   else ''))


# --------------------------------------------------------------------------- running a study step (headless)
# The exit codes are batch's for the same study (README "batch mode"): 0 ran, 3 and 5 findings (the plan goes on),
# 2 nothing to do, 4 an input that cannot be read (or PyTorch missing), 7 CARLA, 8 a model missing, 9 a drive or the
# external detector failed, 10 refused by the gate, 130 cancelled.
STUDY_EXIT = {0: 'ran', 2: 'nothing to evaluate', 3: 'ran, with a finding', 4: 'an input cannot be read', 5:
              'ran: no option meets every must', 7: 'CARLA is not there', 8: 'a model is missing',
              9: 'a drive or the external detector failed', 10: 'refused: no approved plan', 130: 'cancelled'}
STUDY_FINDINGS = (3, 5)


@dataclass
class StudyRun:
    rc: int
    message: str = ''
    summary: dict | None = None
    files: list = field(default_factory=list)       # what the step wrote (absolute paths)
    result: object = None                           # the module's result (the page shows it)
    out_dir: str = ''
    fleet_slot: dict | None = None                  # watch_once: the fleet twin's slot after the drives were read

    @property
    def ok(self) -> bool:
        return self.rc == 0 or self.rc in STUDY_FINDINGS


def _sub(progress, a, b):
    if progress is None:
        return None

    def f(x, text=''):
        try:
            progress(a + (b - a) * min(max(float(x), 0.0), 1.0), text)
        except TypeError:
            progress(a + (b - a) * min(max(float(x), 0.0), 1.0))
    return f


def run_study_kind(project, kind, settings, out_dir, engine=None, ticket=None, progress=None, cancel=None,
                   hooks=None, log=None) -> StudyRun:
    """Run one approved study step headless: refused (exit 10) unless the project holds an approved, intact record
    of the ticket's plan whose step is this kind with exactly these settings (gate); then the step's page settings
    are merged into the project (apply_study_settings: pass a snapshot from a thread), the module runs, its export is
    written into out_dir and its summary() returned. hooks: stand-ins for the tests and the app ('detector',
    'backend', 'detectors', 'depth', 'runner', 'python', 'evaluator', 'coverage')."""
    if kind not in STUDY_KINDS:
        raise KeyError(kind)
    ok, why = gate(project, ticket, kind, settings)
    if not ok:
        return StudyRun(EXIT_REFUSED, why)
    hooks = hooks or {}
    stop = cancel or (lambda: False)
    say = log or (lambda m: None)
    apply_study_settings(project, kind, settings)
    os.makedirs(out_dir, exist_ok=True)
    try:
        r = _STUDY_RUN[kind](project, settings, out_dir, engine, progress, stop, hooks, say)
    except KeyboardInterrupt:
        return StudyRun(130, 'interrupted', out_dir=out_dir)
    r.out_dir = out_dir
    r.files = _stamped(project, r.files, out_dir, ticket)
    if r.summary is not None:
        from ..core import provenance as PV
        r.summary = PV.scrub(_jsonable(r.summary))
    if r.rc == 0 and stop():
        r.rc, r.message = 130, 'cancelled'
    return r


def _stamped(project, paths, out_dir, ticket=None):
    """Stamp every JSON a step wrote with a provenance block naming the plan and the step."""
    from ..core import provenance as PV
    try:
        return PV.stamp_paths(list(paths), project, out_dir, **({'plan': ticket} if ticket else {}))
    except Exception:                       # noqa: BLE001 — the files are there; their stamp is a nicety
        return list(paths)


def sector_named(text, sector) -> bool:
    """Does a finding name a region of this sector ('at left · 0–10 m'; 'rear-left' is not 'left')?"""
    return bool(re.search(rf'(?<![\w-]){re.escape(sector)} ·', str(text or '')))


def _focus(res, sectors, classes) -> dict:
    """Per rig and class, the recall and n in each region of the focus sectors, and the findings that name them."""
    g = res.grid
    names = g.sector_names()
    out = {'sectors': list(sectors), 'regions': {}, 'findings': []}
    for rig in [r['name'] for r in res.rigs]:
        for c in classes:
            for s_ in [names.index(x) for x in sectors if x in names]:
                for r_ in range(g.n_ranges):
                    cell = res.cell(rig, c, g.region_id(s_, r_)) or {}
                    if not cell.get('n_gt'):
                        continue
                    out['regions'].setdefault(rig, {}).setdefault(c, {})[g.region_name(s_, r_)] = {
                        'n_gt': cell.get('n_gt'), 'recall': None if cell.get('recall') is None else
                        round(float(cell['recall']), 4), 'recall_ci': cell.get('recall_ci')}
    out['findings'] = [f['text'] for f in res.findings if f.get('significant') and
                       any(sector_named(f.get('text', ''), x) for x in sectors)][:12]
    return out


def _run_rig_perception(p, s, out, eng, progress, stop, hooks, say):
    from ..core import detection as DT
    from ..core import perception_scenes as PS
    from ..core import rig_perception as RP
    from ..core import simcampaign as SC
    st = RP.RigPerceptionSettings.from_dict(study_part('rig_perception', s))
    lo = 0.0
    if s.get('source') == 'carla':
        ss = PS.SceneSetSettings.from_dict(st.scene_set)
        say(f'scene set {ss.set_id}: driving')
        try:
            r = PS.run(p, ss, None, progress=_sub(progress, 0.0, 0.6), cancel=stop, runner=hooks.get('runner'),
                       python=hooks.get('python'))
        except SC.Cancelled:
            return StudyRun(130, 'cancelled while driving the scene set')
        except RuntimeError as e:
            return StudyRun(7, str(e))
        bad = {k: v for k, v in r['states'].items() if v not in ('ok', 'reused')}
        if bad:
            return StudyRun(9, f'{len(bad)} of {len(r["states"])} scene-set drives failed: ' +
                            '; '.join(f'{k}: {(r.get("errors") or {}).get(k, v)}' for k, v in list(bad.items())[:3]))
        st.recordings = list(r['recordings'])
        lo = 0.6
    if not st.recordings:
        return StudyRun(4, 'no recording with ground truth')
    for rr in st.recordings:
        i = RP.recording_info(rr['path'])
        if not i['ok']:
            return StudyRun(4, f'{_base(rr["path"])}: {i["text"]}')
    if st.backend == 'lifted' and hooks.get('detector') is None and hooks.get('backend') is None and \
            not DT.status(st.model)['verified']:
        return StudyRun(8, DT.status(st.model)['text'])
    try:
        res = RP.run(p, eng, st, progress=_sub(progress, lo, 0.97), cancel=stop, detector=hooks.get('detector'),
                     backend=hooks.get('backend'))
    except FileNotFoundError as e:
        return StudyRun(8, str(e))
    except RuntimeError as e:
        return StudyRun(9, str(e))
    except ValueError as e:
        return StudyRun(4, str(e))
    if res is None:
        return StudyRun(130, 'cancelled')
    files = list(RP.export(res, p, out))
    summ = RP.summary(res)
    if s.get('focus'):
        summ['focus'] = _focus(res, s['focus'], res.classes)
    if s.get('records'):
        summ['records_added'] = len(RP.to_records(res, p, add=True))
    return StudyRun(0, f'{len(res.rigs)} rig(s), {res.timing.get("frames", 0)} instants, '
                       f'{len(summ["findings"])} significant finding(s)', summ, files, res)


def _run_foundation(p, s, out, eng, progress, stop, hooks, say):
    from ..core import foundation_models as FM
    from ..core import foundation_range as FR
    st = FR.FoundationSettings.from_dict(study_part('foundation_range', s))
    if st.source == 'recordings' and not st.recordings:
        return StudyRun(2, 'no recording to evaluate')
    models = list(st.detectors) + ([st.depth_model] if st.depth_model else [])
    if hooks.get('detectors') is None and any(m in FM.SPECS for m in models):
        why = FM.torch_missing()
        if why:
            return StudyRun(4, why)
    kw = {}
    if st.source == 'sweep':
        from ..core import simrig as SR
        kw = dict(sim_settings=SR.settings_of(p), out_dir=os.path.join(out, 'sweep'), python=hooks.get('python'),
                  runner=hooks.get('runner'))
    try:
        res = FR.run(p, eng, st, progress=_sub(progress, 0.0, 0.97), cancel=stop, detectors=hooks.get('detectors'),
                     depth=hooks.get('depth'), **kw)
    except FileNotFoundError as e:
        return StudyRun(8, str(e))
    except RuntimeError as e:
        return StudyRun(7 if st.source == 'sweep' else 4, str(e))
    except ValueError as e:
        return StudyRun(2, str(e))
    if res is None:
        return StudyRun(130, 'cancelled')
    files = list(FR.export(res, p, out))
    summ = FR.summary(res)
    return StudyRun(0, f'{len(res.fits)} range fits', summ, files, res)


def _run_robust(p, s, out, eng, progress, stop, hooks, say):
    from ..core import robust_perception as RB
    st = RB.RobustSettings.from_dict(study_part('robust_perception', s))
    if not st.recordings:
        return StudyRun(2, 'no recording to evaluate')
    try:
        res = RB.run(p, eng, st, progress=_sub(progress, 0.0, 0.97), cancel=stop, detector=hooks.get('detector'))
    except FileNotFoundError as e:
        return StudyRun(8, str(e))
    except (ValueError, OSError) as e:
        return StudyRun(2, str(e))
    if res is None:
        return StudyRun(130, 'cancelled')
    files = list(RB.export(res, p, out))
    summ = RB.summary(res)
    heads = [h for r in summ.get('recordings') or [] for h in (r.get('headline') or [])]
    return StudyRun(0, '; '.join(heads[:2]) or f'{len(res.recordings)} recording(s)', summ, files, res)


def _run_redrive(p, s, out, eng, progress, stop, hooks, say):
    from ..core import redrive as RDV
    from ..core import splat_reproject as SP
    why = SP.torch_missing()
    if why:
        return StudyRun(4, why)
    st = RDV.settings_of(p)
    try:
        res = RDV.run(p, eng, st, progress=_sub(progress, 0.0, 0.8 if s.get('reshoot') else 0.97), cancel=stop)
    except (ValueError, OSError, RuntimeError) as e:
        return StudyRun(4, str(e))
    if res is None:
        return StudyRun(130, 'cancelled')
    files = list(RDV.export(res, p, out))
    summ = RDV.summary(res)
    if s.get('reshoot'):
        try:
            doc = RDV.export_dataset(p, os.path.join(out, 'reshot'), st, engine=eng,
                                     prepared=getattr(res, '_prepared', None), progress=_sub(progress, 0.8, 0.98),
                                     cancel=stop)
        except (ValueError, OSError, RuntimeError) as e:
            return StudyRun(4, f'the re-shot drive was not written: {e}', summ, files, res)
        if doc is None:
            return StudyRun(130, 'cancelled while writing the re-shot drive', summ, files, res)
        summ['reshot'] = True
    verdict = summ.get('rig_verdict')
    msg = f'{summ.get("variant")}: {verdict}, {100 * (summ.get("rig_within_share") or 0):.0f} % within ' \
          f'{_g(summ.get("budget"))} grey levels'
    return StudyRun(3 if verdict == 're-collect' else 0, msg, summ, files, res)


def _run_design(p, s, out, eng, progress, stop, hooks, say):
    from ..core import rig_designer as DSG
    st = DSG.DesignerSettings.from_dict(dict(DSG.settings_of(p).to_dict(), **study_part('design', s)))
    if not st.items():
        return StudyRun(2, 'no accepted requirement')
    try:
        res = DSG.run(p, eng, st, progress=_sub(progress, 0.0, 0.97), cancel=stop)
    except ValueError as e:
        return StudyRun(2, str(e))
    if res is None:
        return StudyRun(130, 'cancelled')
    files = list(DSG.export(res, p, out))
    summ = DSG.summary(res)
    passing = [o for o in summ.get('options') or [] if o.get('verdict') == 'pass']
    msg = f'{len(summ.get("options") or [])} options, {len(passing)} meet every must (nothing was created)'
    return StudyRun(0 if passing else 5, msg, summ, files, res)


def _run_train(p, s, out, eng, progress, stop, hooks, say):
    from ..core import perf_model as PM
    st = PM.PerfModelSettings.from_dict(study_part('perf_model_train', s))
    try:
        res = PM.run(p, eng, st, progress=_sub(progress, 0.0, 0.97), cancel=stop)
    except (OSError, ValueError) as e:
        return StudyRun(4, str(e))
    if res is None:
        return StudyRun(130, 'cancelled')
    files = list(PM.export(res, p, out))
    summ = PM.summary(res)
    if not res.model.trained:
        return StudyRun(3, res.model.reason(), summ, files, res)
    if s.get('save'):
        summ['saved'] = os.path.basename(PM.save_model(res.model))
    return StudyRun(0, f'trained on {res.model.n_records} records', summ, files, res)


def _jsonable(o):
    return json.loads(json.dumps(o, default=lambda x: float(x) if hasattr(x, '__float__') else str(x)))


def _run_campaign(p, s, out, eng, progress, stop, hooks, say):
    from ..core import perf_model as PM
    from ..core import simrig as SR
    st = PM.PerfModelSettings.from_dict(study_part('perf_model_campaign', s))
    vs = _campaign_candidates(p, s)
    if not vs:
        return StudyRun(2, 'no variant with a camera to evaluate')
    cands = PM.candidates_from_variants(p, vs, eng)
    ev = hooks.get('evaluator')
    if ev is None:
        sim = SR.settings_of(p)
        if s.get('duration_s'):
            sim = SR.SimSettings.from_dict(dict(sim.to_dict(), duration_s=float(s['duration_s'])))
        ev = PM.default_evaluator(p, sim, os.path.join(out, 'drives'), hooks.get('detector'))
    try:
        doc = PM.campaign(p, cands, rounds=int(s.get('rounds') or 1), k=st.k, evaluator=ev, settings=st,
                          progress=_sub(progress, 0.0, 0.97), cancel=stop, save=True)
    except FileNotFoundError as e:
        return StudyRun(8, str(e))
    except RuntimeError as e:
        return StudyRun(7, str(e))
    except (OSError, ValueError) as e:
        return StudyRun(4, str(e))
    m = doc['model']
    summ = _jsonable({'schema': 'vantage.perf_model_campaign/1', 'rounds': [
        {'round': r['round'], 'picks': [{k: x.get(k) for k in ('name', 'mean', 'std', 'ei', 'why') if k in x}
                                        for x in r['picks']], 'records': len(r['records'])} for r in doc['rounds']],
        'curve': doc['curve'], 'model': {'trained': m.trained, 'records': m.n_records, 'rigs': m.n_rigs,
                                         'reason': m.reason()}})
    path = os.path.join(out, 'perf_model_campaign.json')
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(summ, fh, indent=1)
    files = [path]
    picked = [x['name'] for r in summ['rounds'] for x in r['picks']]
    if stop():
        return StudyRun(130, 'cancelled', summ, files, doc)
    return StudyRun(0 if m.trained else 3, f'evaluated {", ".join(picked) or "nothing"}; the model '
                                           + (f'trained on {m.n_records} records' if m.trained else m.reason()),
                    summ, files, doc)


def _run_risk(p, s, out, eng, progress, stop, hooks, say):
    from ..core import risk_coverage as RC
    st = RC.RiskSettings.from_dict(study_part('risk_coverage', s))
    res = RC.run(p, eng, st, progress=(lambda f: progress(f)) if progress else None, cancel=stop,
                 coverage=hooks.get('coverage'))
    if res is None:
        return StudyRun(130, 'cancelled')
    files = list(RC.export(res, p, out))
    summ = RC.summary(res)
    summ['requirement_proposals'] = RC.requirement_proposal(res)
    return StudyRun(0, '; '.join(str(d.get('text', c)) for c, d in list(res.classes.items())[:2]), summ, files, res)


def _run_fleet(p, s, out, eng, progress, stop, hooks, say):
    from ..core import fleet_twin as FT
    got = []
    if s.get('action') == 'watch_once':
        got = FT.watch_once(p, FT.WatchState(), eng, cancel=stop, log=say)
    try:
        res = FT.run(p, eng, FT.settings_of(p), None, _sub(progress, 0.5 if got else 0.0, 0.97), stop)
    except (ValueError, FT.TwinError) as e:
        return StudyRun(4, str(e))
    if res is None:
        return StudyRun(130 if stop() else 4, 'cancelled' if stop() else 'the fleet twin could not run')
    files = list(FT.export(res, p, out))
    summ = FT.summary(res)
    summ.pop('assumptions', None)
    if s.get('action') == 'watch_once':
        summ['drives_read'] = len(got)
    msg = (f'{len(got)} new drive' + ('s' if len(got) != 1 else '') + ' read; ' if s.get('action') == 'watch_once'
           else '') + f'{len(res.findings)} finding(s)'
    return StudyRun(FT.exit_code(res), msg, summ, files, res,
                    fleet_slot=copy.deepcopy(getattr(p.analysis, 'fleet_twin', None)) if got else None)


_STUDY_RUN = {'rig_perception': _run_rig_perception, 'foundation_range': _run_foundation,
              'robust_perception': _run_robust, 'redrive': _run_redrive, 'design': _run_design,
              'perf_model_train': _run_train, 'perf_model_campaign': _run_campaign, 'risk_coverage': _run_risk,
              'fleet_twin': _run_fleet}


# =========================================================================== the deterministic planner (no model)
# Natural requests for the study kinds, read with patterns on this computer: what the Copilot plans when no model is
# set up (the panel), and the draft get_catalogue('plan_kinds') hands the model for the request it is answering. Each
# value it sets says it came from the request (with the words it read) or was chosen, and why.
_KIND_PATTERNS = (
    ('perf_model_campaign', r'\b(which|what)\s+(rigs?|variants?|designs?)\s+(should|to|do|would)\b.*\b(simulat\w*|'
                            r'evaluat\w*|drive|test|try)\b|\bsimulate\s+next\b|\bactive[- ]learning\b|'
                            r'\blearned[- ]model\s+campaign\b'),
    ('perf_model_train', r'\b(train|retrain|re-train|fit|refit)\b.{0,40}\b(model|learned|predictor|gaussian)\b'),
    ('design', r'\bdesign\s+(me\s+)?(a\s+|the\s+|new\s+)*(rigs?|sensor\s+rigs?|layouts?)\b|\bdesign\s+agent\b|'
               r'\brigs?\s+for\s+(this|the|my|our)\s+brief\b'),
    ('redrive', r'\b(re-?use|re-?drive|re-?shoot|re-?render)\w*\b.{0,60}\b(drives?|logs?|data|recordings?|rigs?)\b|'
                r'\b(drives?|logs?|data|recordings?)\b.{0,40}\b(with|through|on)\s+the\s+new\s+rig\b'),
    ('robust_perception', r'\b(mounts?|mounting|brackets?)\b.{0,40}\b(off|out|wrong|error|shifted|tolerance|slip\w*)\b|'
                          r'\brobust\s+perception\b|\btolerances?\b.{0,40}\b(detection|recall|perception)\b'),
    ('foundation_range', r'\bhow\s+far\b.{0,40}\b(see|sees|detect\w*|find\w*|spot\w*|recogni[sz]\w*)\b|'
                         r'\bfoundation[- ]models?\b|\bdetection\s+range\b|\bmonocular\s+depth\b'),
    ('rig_perception', r'\b(compare|comparing|versus|vs\.?|against|better)\b.{0,80}\b(detect\w*|recall|perception|'
                       r'mAP)\b|\b(detect\w*|recall|perception)\b.{0,60}\brigs?\b.{0,20}\b(and|vs\.?|versus|or)\b|'
                       r'\brig\s+perception\b'),
    ('risk_coverage', r'\brisk[- ]weighted\b|\brisk\s+coverage\b|\b(pre-?crash|crash\w*|ncap)\b.{0,40}\bcover\w*\b|'
                      r'\bcover\w*\b.{0,40}\b(pre-?crash|crash\w*|ncap)\b'),
    ('fleet_twin', r'\bfleet\s+twin\b|\b(drift\w*|recalibrat\w*|knock\w*)\b.{0,40}\b(fleet|vehicles?|cars?)\b|'
                   r'\b(fleet|vehicles?|cars?)\b.{0,40}\b(drift\w*|recalibrat\w*|knock\w*)\b|'
                   r'\bwatch\s+(the\s+)?(fleet|vehicles?|folders?)\b'),
)
_KIND_RX = [(k, re.compile(rx, re.IGNORECASE)) for k, rx in _KIND_PATTERNS]
CLASS_WORDS = (('motorcyclist', r'motorcyclists?|motorbikes?|motorcycles?'),
               ('cyclist', r'cyclists?|bicycles?|bikes?|bicyclists?'),
               ('pedestrian', r'pedestrians?|people|persons?|walkers?|children|child|kids?'),
               ('truck', r'trucks?|lorr(?:y|ies)'), ('bus', r'bus(?:es)?'), ('car', r'cars?'),
               ('vehicle', r'vehicles?'))
SECTOR_WORDS = (('front-left', r'front[- ]left'), ('front-right', r'front[- ]right'), ('rear-left', r'rear[- ]left'),
                ('rear-right', r'rear[- ]right'), ('front', r'front|ahead|forward'), ('rear', r'rear|behind|back'),
                ('left', r'left'), ('right', r'right'))


def match_kind(text) -> str:
    """The study kind a request asks for ('' when none): the deterministic planner's first reading."""
    t = str(text or '')
    for k, rx in _KIND_RX:
        if rx.search(t):
            return k
    return ''


def _words(text, table) -> list:
    out, t = [], ' ' + str(text or '').lower() + ' '
    for name, rx in table:
        m = re.search(rf'\b({rx})\b', t)
        if m:
            out.append(name)
            t = t[:m.start()] + ' ' * (m.end() - m.start()) + t[m.end():]   # 'front-left' is not also 'left'
    return out


def _named_variants(project, text) -> list:
    """Variants the request names: by their full name, or as 'rig A and B' (a variant named A or ending in ' A')."""
    t = str(text or '')
    low = t.lower()
    out = []
    for v in sorted(project.variants, key=lambda v: -len(v.name)):
        if len(v.name) > 2 and v.name.lower() in low and v not in out:
            out.append(v)
    toks = []
    for m in re.finditer(r'\b(?:rigs?|variants?)\s+([A-Za-z0-9]{1,12}(?:\s*(?:,|&|and|vs\.?|versus|or)\s*'
                         r'(?:rig\s+)?[A-Za-z0-9]{1,12})*)', t):
        toks += [x for x in re.split(r'\s*(?:,|&|\band\b|\bvs\.?|\bversus\b|\bor\b|\brig\b)\s*', m.group(1)) if x]
    for tok in toks:
        tl = tok.lower()
        v = next((v for v in project.variants if v.name.lower() == tl or v.name.lower().endswith(' ' + tl)
                  or v.name.lower() == f'rig {tl}'), None)
        if v is not None and v not in out:
            out.append(v)
    order = {v.uid: i for i, v in enumerate(project.variants)}
    pos = {v.uid: (low.find(v.name.lower()) if v.name.lower() in low else 10 ** 6 + order[v.uid]) for v in out}
    for tok in toks:
        for v in out:
            if v.name.lower() == tok.lower() or v.name.lower().endswith(' ' + tok.lower()):
                pos[v.uid] = min(pos[v.uid], low.find(tok.lower()))
    return sorted(out, key=lambda v: pos[v.uid])


def _camera_tags(project, text) -> list:
    tags = {c.tag for c in project.variant.cameras if c.visible}
    out = []
    for m in re.finditer(r'\b([A-Z][A-Z0-9]{0,3})\b', str(text or '')):
        if m.group(1) in tags and m.group(1) not in out:
            prev = str(text)[max(0, m.start() - 8):m.start()].lower()
            if re.search(r'\b(rig|rigs|variant|and)\s*$', prev) and m.group(1) not in ('F', 'B'):
                continue
            out.append(m.group(1))
    return out


def _V(key, value, source='request', reason=''):
    return {'key': key, 'value': json.dumps(value), 'source': source, 'reason': reason}


def _recordings_of(project, variants, working_dir='') -> dict:
    """{variant uid: (path, name as the catalogue shows it)}: each variant's newest CARLA drive with ground truth (the
    rig perception page's recordings first)."""
    from ..core import rig_perception as RP
    from . import copilot_tools as CT
    out = {}
    names = {v.name: v.uid for v in variants}
    for r in RP.settings_of(project).recordings:
        u = names.get(r.get('rig') or '')
        if u and u not in out and os.path.isdir(r['path']):
            out[u] = (r['path'], _base(r['path']))
    for r in CT.find_runs(project, working_dir, gt_only=True):
        u = names.get(r.get('variant') or '')
        if u and u not in out and r.get('status') == 'ok':
            out[u] = (r['path'], r['name'])
    return out


def offline_steps(project, text, working_dir='') -> list:
    """The steps the deterministic planner reads from a request ([] when it does not know it): propose_plan's input,
    [{'kind', 'label', 'values'}]."""
    kind = match_kind(text)
    if not kind:
        return []
    t = str(text or '')
    fn = globals()[f'_offline_{kind}']
    st = fn(project, t, working_dir)
    return [st] if st else []


def _offline_rig_perception(p, t, wd):
    vs = _named_variants(p, t) or [p.variant]
    cls = _words(t, CLASS_WORDS)
    sec = _words(t, SECTOR_WORDS)
    vals = []
    if cls:
        from ..core import bev_backends as BB
        want = [c for c in BB.BEV_CLASSES if c in cls or (c in ('car', 'truck', 'bus') and 'vehicle' in cls)]
        if want:
            vals.append(_V('classes', want, 'request', f'you asked about {_and(cls)}'))
    if sec:
        vals.append(_V('focus', sec, 'request', f'you asked about the {_and(sec)}'))
    have = _recordings_of(p, vs, wd)
    if all(v.uid in have for v in vs):
        vals.append(_V('source', 'recordings', 'copilot', 'every rig has a drive with ground truth already'))
        vals.append(_V('recordings', [{'path': have[v.uid][1], 'rig': v.name} for v in vs], 'copilot',
                       'the newest drive with ground truth of ' + _and(v.name for v in vs)))
    else:
        missing = [v.name for v in vs if v.uid not in have]
        vals.append(_V('source', 'carla', 'copilot', f'no drive with ground truth of {_and(missing)} yet: a perception '
                                                     'scene set drives the same scenes for every rig (paired)'))
        vals.append(_V('scene_set.variants', [v.name for v in vs], 'request' if _named_variants(p, t) else 'copilot',
                       'the rigs you named' if _named_variants(p, t) else 'the active variant'))
        if re.search(r'\bquick\b', t, re.I):
            vals.append(_V('scene_set.set_id', 'quick', 'request', 'you asked for a quick look'))
    if len(vs) < 2:
        label = f'{vs[0].name} detection'
    else:
        label = ' vs '.join(v.name for v in vs[:3]) + ' detection'
    return {'kind': 'rig_perception', 'label': label[:80], 'values': vals}


def _offline_foundation_range(p, t, wd):
    from ..core import foundation_range as FR
    from . import copilot_tools as CT
    cams = _camera_tags(p, t)
    cls = [c for c in _words(t, CLASS_WORDS) if c in [x['name'] for x in FR.DEFAULT_CLASSES]]
    if 'vehicle' in _words(t, CLASS_WORDS) and 'car' not in cls:
        cls.append('car')
    vals = []
    runs = [r for r in CT.find_runs(p, wd, gt_only=True) if r.get('status') == 'ok']
    sweep = bool(cls) or not runs
    if sweep:
        vals.append(_V('source', 'sweep', 'copilot', 'a range sweep stands the class at known ranges in CARLA, so '
                                                     'R50 and R90 are measured where they are'))
        if cams:
            vals.append(_V('sweep_cameras', cams, 'request', f'you named {_and(cams)}'))
        if cls:
            vals.append(_V('sweep_classes', cls, 'request', f'you asked about {_and(cls)}'))
    else:
        vals.append(_V('source', 'recordings', 'copilot', 'the newest drive with ground truth'))
        vals.append(_V('recordings', [runs[0]['name']], 'copilot', 'the newest drive with ground truth'))
        if cams:
            vals.append(_V('cameras', cams, 'request', f'you named {_and(cams)}'))
    if not re.search(r'\bdepth\b', t, re.I):
        vals.append(_V('depth_model', '', 'copilot', 'you asked about detection range, not depth'))
    what = (_and(cls) if cls else 'objects')
    return {'kind': 'foundation_range', 'label': f'How far {_and(cams) if cams else "the cameras"} see {what}'[:80],
            'values': vals}


def _offline_robust_perception(p, t, wd):
    from . import copilot_tools as CT
    vals = []
    mm = re.search(r'(\d+(?:\.\d+)?)\s*(mm|millimet\w*)\b', t, re.I)
    cm = re.search(r'(\d+(?:\.\d+)?)\s*(cm|centimet\w*)\b', t, re.I)
    dg = re.search(r'(\d+(?:\.\d+)?)\s*(°|deg\w*)', t, re.I)
    if mm or cm or dg:
        vals.append(_V('mount_from', 'custom', 'request', 'you gave the mounting error'))
        x_mm = float(mm.group(1)) if mm else (10.0 * float(cm.group(1)) if cm else 0.0)
        vals.append(_V('mount_mm', x_mm, 'request' if (mm or cm) else 'copilot',
                       f'"{(mm or cm).group(0)}"' if (mm or cm) else 'you gave only an angle'))
        vals.append(_V('mount_deg', float(dg.group(1)) if dg else 0.0, 'request' if dg else 'copilot',
                       f'"{dg.group(0)}"' if dg else 'you gave only an offset: no rotation'))
        for k in ('thermal', 'vibration', 'calibration'):
            vals.append(_V(k, False, 'copilot', 'only the mounting varies, so the result says what that error does'))
    runs = [r for r in CT.find_runs(p, wd, gt_only=True) if r.get('status') == 'ok']
    mine = [r for r in runs if r.get('variant') == p.variant.name] or runs
    if mine:
        vals.append(_V('recordings', [mine[0]['name']], 'copilot', 'the newest drive with ground truth'
                       + (f' of {p.variant.name}' if mine[0].get('variant') == p.variant.name else '')))
    return {'kind': 'robust_perception', 'label': 'Mounting error carried to detection', 'values': vals}


def _offline_redrive(p, t, wd):
    from ..core import splat_reproject as SP
    from . import copilot_tools as CT
    vals = []
    vs = _named_variants(p, t)
    if not SP.settings_of(p).drive:
        runs = [r for r in CT.find_runs(p, wd) if r.get('status') == 'ok']
        if runs:
            vals.append(_V('drive', runs[0]['name'], 'copilot', 'the newest recorded drive in the project\'s runs '
                                                               'folder (the log replay has none chosen)'))
    if vs:
        vals.append(_V('variant', vs[0].name, 'request', f'you named {vs[0].name}'))
    elif re.search(r'\bnew\s+rig\b', t, re.I) and len(p.variants) > 1:
        vals.append(_V('variant', p.variants[-1].name, 'copilot', 'the newest variant, as "the new rig"'))
    if re.search(r'\b(re-?shoot|re-?shot|export|dataset)\b', t, re.I):
        vals.append(_V('reshoot', True, 'request', 'you asked for the re-shot drive'))
    return {'kind': 'redrive', 'label': 'Can the drive be reused', 'values': vals}


def _offline_design(p, t, wd):
    from ..core import rig_designer as DSG
    vals = []
    m = re.search(r'(?:brief|requirements?)\s*[:—–-]\s*(.+)$', t, re.I | re.S)
    brief = m.group(1).strip() if m and len(m.group(1).strip()) > 8 else ''
    src = 'request'
    if not brief:
        brief = DSG.settings_of(p).brief or str((getattr(p.analysis, 'assistant', None) or {}).get('brief', ''))
        src = 'copilot'
    have = bool(DSG.settings_of(p).items())
    if brief and (not have or src == 'request'):
        vals.append(_V('brief', brief, src, 'the brief you gave' if src == 'request' else
                       'the project\'s brief (Assistant ▸ Brief, or the design agent\'s)'))
        vals.append(_V('requirements_from', 'brief', 'copilot', 'the requirements are read from the brief by '
                                                                'Vantage\'s parser'))
    n = re.search(r'\b([3-7])\s+(options|rigs|designs)\b', t, re.I)
    if n:
        vals.append(_V('options', int(n.group(1)), 'request', f'"{n.group(0)}"'))
    return {'kind': 'design', 'label': 'Design rigs for the brief', 'values': vals}


def _offline_perf_model_train(p, t, wd):
    return {'kind': 'perf_model_train', 'label': 'Train the detection model',
            'values': [_V('save', True, 'copilot', 'a trained model is what the app and the optimiser load')]}


def _offline_perf_model_campaign(p, t, wd):
    vals = []
    n = re.search(r'\b(\d{1,2})\s+(rigs?|variants?|designs?|candidates?)\b', t, re.I)
    if n:
        vals.append(_V('k', int(n.group(1)), 'request', f'"{n.group(0)}"'))
    else:
        vals.append(_V('k', 2, 'copilot', 'two rigs a round: each is a CARLA drive and a rig perception run'))
    vs = _named_variants(p, t)
    if len(vs) > 1:
        vals.append(_V('variants', [v.name for v in vs], 'request', 'the variants you named'))
    return {'kind': 'perf_model_campaign', 'label': 'Which rigs to simulate next', 'values': vals}


def _offline_risk_coverage(p, t, wd):
    vals = []
    for w, rx in (('ncap', r'\bncap\b'), ('fatal', r'\bfatal\w*\b'), ('cost', r'\bcosts?\b'),
                  ('fyl', r'\bfunctional\s+years?\b|\bfyl\b')):
        if re.search(rx, t, re.I):
            vals.append(_V('weight', w, 'request', f'you asked for the {w} weighting'))
            break
    return {'kind': 'risk_coverage', 'label': 'Risk-weighted coverage', 'values': vals}


def _offline_fleet_twin(p, t, wd):
    watch = bool(re.search(r'\bwatch\w*\b|\bnew\s+drives?\b|\barriv\w*\b', t, re.I))
    vals = [_V('action', 'watch_once' if watch else 'run', 'request' if watch else 'copilot',
               'you asked to read the new drives' if watch else 'fit the twin on the records there are')]
    return {'kind': 'fleet_twin', 'label': 'Fleet twin', 'values': vals}


def offline_plan(project, text, working_dir='', engine=None, prev=None) -> 'Plan | None':
    """A plan from the deterministic planner (provider 'vantage', no model), or None when it does not know the
    request. It is checked and estimated like any plan and needs the same approval."""
    steps = offline_steps(project, text, working_dir)
    if not steps:
        return None
    kind = steps[0]['kind']
    return build(project, steps, KIND_LABEL[kind], f'Read on this computer from your request ({STUDY_ABOUT.get(kind, ("",))[0]}).'
                 [:600], prev=None, request=str(text or ''), provider='vantage', model='deterministic planner',
                 working_dir=working_dir, engine=engine)
