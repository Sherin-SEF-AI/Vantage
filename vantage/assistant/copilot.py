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

KINDS = ('carla_run', 'campaign', 'detection', 'critical_carla', 'carla_live', 'experiment', 'sample_dataset',
         'study', 'export')
KIND_LABEL = {'carla_run': 'CARLA drive', 'campaign': 'Simulation campaign', 'detection': 'Detection quality',
              'critical_carla': 'Critical targets in CARLA', 'carla_live': 'CARLA live rig',
              'experiment': 'Designed experiment', 'sample_dataset': 'Sample dataset', 'study': 'Study',
              'export': 'Export'}
KIND_PAGE = {'carla_run': 'sim', 'campaign': 'campaign', 'detection': 'detection', 'critical_carla': 'critical_carla',
             'carla_live': 'sim', 'experiment': 'experiments', 'sample_dataset': 'dataset', 'study': '',
             'export': ''}
# the project.analysis slot a kind's settings live in (what "Open on its page" and Run set); None: kept in the plan
KIND_SLOT = {'carla_run': 'sim', 'campaign': 'campaign', 'detection': 'detection', 'critical_carla': 'critical_carla',
             'carla_live': None, 'experiment': 'experiments', 'sample_dataset': 'dataset', 'study': None,
             'export': None}
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
}


_PLAN_WORDS = re.compile(
    r'(^\s*/plan\b)|\b(drive|drives|driving|run|runs|record|recording|campaign|sweep|detection|carla|live rig|'
    r'experiment|dataset|generate|export|simulate|simulation|critical targets?|scenario|seeds?|faults?)\b',
    re.IGNORECASE)


def is_plan_request(text: str) -> bool:
    """Does the message ask for something to be run or written (a Copilot turn: thinking effort high)?"""
    return bool(_PLAN_WORDS.search(text or ''))


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
    if kind == 'detection' and key == 'recordings':
        return [os.path.basename(str(r).rstrip('/\\')) or str(r) for r in value]
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
        return value if value else '—'
    if value is None:
        return '—'
    if isinstance(value, list):
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
        if kind == 'detection' and key == 'recordings':
            continue                    # named as the catalogue shows them, kept as full paths
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
