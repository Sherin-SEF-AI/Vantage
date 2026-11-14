"""The Copilot's tools (registered in tools.TOOLS): read-only catalogues, the settings of each kind of step, the
documented limits, estimate and validate (which show nothing to the user), and propose_plan, the one tool that writes:
it creates or revises the plan on the user's screen. None of them starts, stops or deletes anything.

What they return names no folder of this machine: recordings and runs are named relative to the working folder."""
from __future__ import annotations

import json
import os

from . import copilot as CP

TOPICS = ('towns', 'weathers', 'scenarios', 'faults', 'actors', 'rig', 'detector', 'recordings', 'past_runs',
          'export_formats', 'studies', 'plan_kinds')


def _wd(tc):
    from .. import workdir
    return tc.working_dir or workdir.default()


def find_runs(project, working_dir='', gt_only=False, limit=40) -> list:
    """CARLA drives in the project's runs folder, newest first: {'path', 'name', 'town', 'weather', 'duration_s',
    'scenario', 'faults', 'ground_truth', 'status', 'when'}."""
    from .. import workdir
    root = workdir.project_dir(project, working_dir or workdir.default(), 'carla')
    out = []
    try:
        names = os.listdir(root)
    except OSError:
        return out
    for n in names:
        d = os.path.join(root, n)
        f = os.path.join(d, 'run.json')
        if not os.path.isfile(f):
            continue
        try:
            with open(f, encoding='utf-8') as fh:
                doc = json.load(fh)
        except (OSError, ValueError):
            continue
        gt = os.path.isfile(os.path.join(d, 'ground_truth', 'index.json'))
        if gt_only and not gt:
            continue
        sc = doc.get('scenario') or {}
        out.append({'path': d, 'name': workdir.display(d, working_dir or None), 'status': doc.get('status'),
                    'variant': doc.get('variant') or '',
                    'town': doc.get('town'), 'weather': (doc.get('weather') or {}).get('label') if
                    isinstance(doc.get('weather'), dict) else doc.get('weather'),
                    'duration_s': (doc.get('performance') or {}).get('simulated_s'),
                    'scenario': sc.get('name') if isinstance(sc, dict) else sc,
                    'faults': len(doc.get('faults_injected') or []), 'ground_truth': gt,
                    'when': os.path.getmtime(f)})
    out.sort(key=lambda r: -r['when'])
    return out[:limit]


def get_catalogue(tc, topic='', **_):
    from ..core import simrig as SR
    from ..sim import carla_bridge as CB
    p = tc.project
    if topic not in TOPICS:
        return {'error': f'topic must be one of {", ".join(TOPICS)}'}
    if topic == 'towns':
        return {'towns': list(SR.TOWNS), 'carla': SR.CARLA_VERSION,
                'note': 'the 0.9.16 base package; AdditionalMaps adds Town06, 07, 11, 12, 13 and 15 when the server '
                        'has them installed'}
    if topic == 'weathers':
        from ..core import weather as W
        conds = [{'id': c, 'value': f'vantage:{c}'} for c in W.IDS]
        return {'carla_presets': list(SR.CARLA_WEATHERS), 'vantage_conditions': conds, 'project': 'project',
                'note': 'weather takes a CARLA preset, "vantage:<id>" (a Vantage condition: its fog becomes CARLA fog by '
                        'a calibration measured on CARLA 0.9.16, its light and rain by documented heuristics) or '
                        '"project" (the condition selected in the project). Dusk: the *Sunset presets. ' +
                        CP.LIMITS_DOC['weather_sensors']}
    if topic == 'scenarios':
        out = {}
        for name, (label, about, params) in CB.SCENARIOS.items():
            ps = {}
            for k, spec in params.items():
                if spec[0] == 'num':
                    ps[k] = {'default': spec[1], 'range': [spec[2], spec[3]], 'unit': spec[4], 'about': spec[5]}
                else:
                    ps[k] = {'default': spec[1], 'options': list(spec[2]), 'about': spec[3]}
            out[name] = {'label': label, 'about': about, 'params': ps}
        return {'scenarios': out, 'keys': 'scenario.name and scenario.params.<param> in a carla_run step'}
    if topic == 'faults':
        from ..core import simcampaign as SC
        from ..sim import live as LV
        return {'carla_run': {k: {'sensors': list(v[0]), 'about': v[1]} for k, v in CB.FAULT_KINDS.items()},
                'carla_run_limits': CB.FAULT_LIMITS,
                'campaign_fault_sets': {'kinds': list(SC.FAULT_KINDS), 'units': SC.FAULT_UNIT,
                                        'magnitudes_range': SC.MAG_LIMITS, 'axes': SC.AXES, 'signs': list(SC.SIGNS),
                                        'shape': '{kind, tags, axes, magnitudes (in the unit), signs}'},
                'live': {'kinds': list(LV.KINDS), 'max_deg': LV.MAX_DEG, 'max_m': LV.MAX_M, 'max_ms': LV.MAX_MS,
                         'shape': '{at_s, fault: {tag, kind, deg | m | rate | ms, s (optional duration)}}'},
                'note': CP.LIMITS_DOC['faults']}
    if topic == 'actors':
        return {'kinds': list(CB.ACTOR_KINDS) + ['blueprint:<id>'], 'max': CB.MAX_ACTORS,
                'shape': '{id, kind, pose: {x, y, z, yaw}} in the ego\'s vehicle frame (ISO 8855, rear axle on the '
                         'ground; metres, degrees); walkers stand still; no traffic',
                'blueprints': {k: list(v) for k, v in CB.ACTOR_BLUEPRINTS.items()}}
    if topic == 'rig':
        v = p.variant
        return {'variant': v.name, 'cameras': [c.tag for c in v.cameras if c.visible],
                'sensors': [{'tag': s.tag, 'kind': s.kind} for s in v.sensors if s.visible],
                'variants': [x.name for x in p.variants]}
    if topic == 'detector':
        from ..core import detection as DT
        from ..core import detection_eval as DE
        st = DE.settings_of(p)
        s = DT.status(st.model)
        return {'model': s['id'], 'name': s['name'], 'downloaded': s['present'], 'verified': s['verified'],
                'licence': s['licence'], 'classes': list(st.classes),
                'note': CP.LIMITS_DOC['detector']}
    if topic == 'recordings':
        rows = find_runs(p, _wd(tc), gt_only=True)
        return {'recordings': [{k: r[k] for k in ('name', 'town', 'weather', 'duration_s', 'scenario', 'faults',
                                                  'status')} for r in rows if r['status'] == 'ok'],
                'note': 'newest first; CARLA drives with ground truth (detection quality needs it); name them in a '
                        'detection step\'s recordings as shown'}
    if topic == 'past_runs':
        rows = find_runs(p, _wd(tc))
        plans = [{'plan': r.get('id'), 'version': r.get('version'), 'title': r.get('title'), 'status': r.get('status'),
                  'approved_utc': r.get('approved_utc'), 'steps': [s['kind'] for s in r.get('steps') or []]}
                 for r in CP.records(p)[-12:]]
        return {'carla_drives': [{k: r[k] for k in ('name', 'status', 'town', 'weather', 'duration_s', 'scenario',
                                                    'faults', 'ground_truth')} for r in rows],
                'approved_plans': plans}
    if topic == 'export_formats':
        from ..core import exporters as ex
        return {'formats': [{'id': f.id, 'group': f.group, 'label': f.label} for f in ex.FORMATS
                            if f.id not in CP.IMAGE_FORMATS],
                'note': 'images rendered by the window (viewport_png, bev_png, camera_views_png) are exported from the '
                        'Export dialog, not from a plan'}
    if topic == 'studies':
        from . import tools as T
        return {'studies': list(CP.STUDIES), 'about': T.BY_NAME['run_study'].description}
    out = {'kinds': [{'kind': k, 'label': CP.KIND_LABEL[k], 'page': CP.KIND_PAGE[k] or None,
                      'settings_saved_on_the_page': CP.KIND_SLOT[k] is not None or k in CP.STUDY_KINDS,
                      **({'what': CP.STUDY_ABOUT[k][0], 'for_example': CP.STUDY_ABOUT[k][1],
                          'plan_only_options': list(CP.STUDY_EXTRAS[k])} if k in CP.STUDY_KINDS else {})}
                     for k in CP.KINDS],
           'note': 'get_plan_settings(kind) lists a kind\'s settings with their current values and ranges. The study '
                   'kinds (rig_perception … fleet_twin) run round 14\'s studies after approval, in the background; '
                   'their results reach the page and the read tools (get_rig_perception, get_foundation_range, '
                   'get_robust_perception, get_redrive, get_perf_model, design_rig, get_risk_coverage, '
                   'get_fleet_twin), which you then cite.'}
    if tc.request:
        try:
            draft = CP.offline_steps(p, tc.request, _wd(tc))
        except Exception:                   # noqa: BLE001 — a draft is a hint, never in the way
            draft = []
        if draft:
            out['draft_for_this_request'] = {
                'steps': PV_scrub(draft), 'note': 'what Vantage\'s own planner reads from the request: a starting '
                                                  'point for propose_plan (check the values against the request)'}
    return out


def PV_scrub(o):
    from ..core import provenance as PV
    return PV.scrub(o)


def _choices(kind, key):
    from ..core import simrig as SR
    from ..sim import carla_bridge as CB
    if kind == 'carla_run':
        return {'town': list(SR.TOWNS), 'camera_rate': list(SR.CAMERA_RATES), 'channel_names': list(SR.CHANNEL_NAMES),
                'exposure': list(SR.EXPOSURES), 'quality': list(SR.QUALITIES), 'scenario.name': list(CB.SCENARIOS),
                'weather': ['project', 'vantage:<condition>'] + list(SR.CARLA_WEATHERS),
                'vehicle': ['auto'] + list(SR.CARLA_VEHICLES)}.get(key)
    if kind == 'sample_dataset':
        from ..core import dataset as DSET
        return {'scene': list(DSET.SCENES), 'camera_rate': ['lidar', 'module'],
                'channel_names': ['tags', 'nuscenes']}.get(key)
    if kind == 'experiment':
        from ..core import experiments as EX
        return {'design': list(EX.DESIGNS)}.get(key)
    if kind == 'campaign':
        from ..core import simcampaign as SC
        return {'template': list(SC.TEMPLATES) + ['custom'], 'evaluators': list(SC.EVALUATORS),
                'asbuilt_on': list(SC.ASBUILT_ON), 'server': list(SC.SERVERS)}.get(key)
    if kind == 'detection':
        return {'views': ['fisheye', 'pinhole', 'both']}.get(key)
    if kind == 'study':
        from ..core.weather import IDS
        return {'study': list(CP.STUDIES), 'condition': [''] + list(IDS)}.get(key)
    if kind == 'export':
        from ..core.project import FRAME_PRESETS
        return {'units': ['m', 'mm'], 'angles': ['deg', 'rad'], 'frame': list(FRAME_PRESETS)}.get(key)
    if kind in CP.STUDY_KINDS:
        return _study_choices(kind, key)
    return None


def _study_choices(kind, key):
    ch = CP.STUDY_EXTRA_CHOICES.get((kind, key))
    if ch:
        return list(ch)
    try:
        if kind == 'rig_perception':
            from ..core import perception_scenes as PS
            from ..core import rig_perception as RP
            from ..core import regions as RG
            return {'views': list(RP.VIEW_MODES), 'instants': list(RP.INSTANTS), 'cache': list(RP.CACHE_MODES),
                    'scene_set.set_id': list(PS.SCENE_SETS), 'focus': list(RG.SECTOR_NAMES_8),
                    'classes': list(RP.BB.BEV_CLASSES), 'backend': ['lifted', 'external:<name>']}.get(key)
        if kind == 'foundation_range':
            from ..core import foundation_models as FM
            from ..core import foundation_range as FR
            return {'source': list(FR.SOURCES), 'views': ['fisheye', 'pinhole', 'both'],
                    'detectors': ['yolox_s'] + list(FM.DETECTORS), 'depth_model': [''] + list(FM.DEPTH_MODELS),
                    'sweep_classes': [c['name'] for c in FR.DEFAULT_CLASSES], 'device': ['auto', 'cuda', 'cpu']}.get(key)
        if kind == 'robust_perception':
            from ..core import robust_perception as RB
            return list(RB.CHOICES[key]) if key in getattr(RB, 'CHOICES', {}) else None
        if kind == 'redrive':
            from ..core import redrive as RDV
            return {'schedule': ['recorded', 'rate'], 'rolling_shutter': ['module', 'off'],
                    'ground_truth': ['auto', 'always', 'never'], 'sources': list(RDV.SOURCES)}.get(key)
        if kind == 'design':
            from ..core import rig_designer as DSG
            return {'axes': list(DSG.AXES)}.get(key)
        if kind in ('perf_model_train', 'perf_model_campaign'):
            from ..core import perf_model as PM
            return {'mode': ['ei', 'std', 'ucb'], 'classes': list(PM.CLASSES), 'metrics': list(PM.METRICS)}.get(key)
        if kind == 'risk_coverage':
            return {'weight': ['frequency', 'cost', 'fatal', 'fyl', 'ncap'], 'metric': ['resolved', 'seen', 'redundant'],
                    'dataset': ['swanson2019', 'najm2007']}.get(key)
        if kind == 'fleet_twin':
            return {'limit_source': ['auto', 'critical', 'analysis', 'fleet', 'user'], 'pool': ['sensor', 'kind'],
                    'online_obs': ['segments', 'windows']}.get(key)
    except Exception:                       # noqa: BLE001 — choices are a nicety
        return None
    return None


def _limits(kind):
    try:
        if kind == 'carla_run':
            from ..core import simrig as SR
            return SR.LIMITS
        if kind == 'sample_dataset':
            from ..core import dataset as DSET
            return DSET.LIMITS
        if kind == 'critical_carla':
            from ..core import critical_carla as CC
            return CC.LIMITS
        if kind == 'detection':
            from ..core import detection_eval as DE
            return DE.LIMITS
        if kind == 'experiment':
            from ..core import experiments as EX
            return EX.LIMITS
        if kind == 'campaign':
            from ..core import simcampaign as SC
            return SC.LIMITS
        if kind in CP.STUDY_KINDS:
            return CP.study_limits(kind)
    except Exception:                       # noqa: BLE001
        return {}
    return {}


def get_plan_settings(tc, kind='', **_):
    if kind not in CP.KINDS:
        return {'error': f'kind must be one of {", ".join(CP.KINDS)}'}
    from ..core import provenance as PV
    d = CP.defaults(kind, tc.project)
    if kind == 'detection':
        from .. import workdir
        d['recordings'] = [workdir.display(r, tc.working_dir or None) for r in d.get('recordings') or []]
    if kind in CP.STUDY_KINDS:
        d = CP.study_for_model(kind, d, tc.working_dir)
    lim = _limits(kind)
    rows = []
    for k, v in CP.flatten(d).items():
        row = {'key': k, 'value': v}
        if k in lim:
            row['range'] = list(lim[k])
        ch = _choices(kind, k)
        if ch:
            row['choices'] = ch
        uses = {'experiment': CP.DESIGN_USES, 'study': CP.STUDY_USES}.get(kind, {}).get(k)
        if uses:
            row['used_only_by'] = list(uses)      # another design's (or study's) field: not used, leave it out
        rows.append(row)
    out = {'kind': kind, 'label': CP.KIND_LABEL[kind], 'settings': rows,
           'source_of_these_values': 'the page\'s current settings (Vantage\'s defaults where never changed)'}
    if kind == 'carla_run':
        from ..sim import carla_bridge as CB
        name = d['scenario']['name']
        out['scenario_params'] = {k: (spec[1] if spec[0] == 'num' else spec[1]) for k, spec in
                                  CB.SCENARIOS[name][2].items()}
        out['note'] = ('scenario.params.<p> sets a parameter of the chosen scenario (get_catalogue scenarios lists '
                       'them); faults and actors are lists (get_catalogue faults / actors); traffic is a free '
                       'drive\'s: a scenario has its own scenario.params.traffic (a traffic asked for with a scenario '
                       'goes there), an actors scene or parked none; min_tick_hz and max_tick_hz only apply when '
                       'tick_hz is 0; traffic_source: carla (the traffic manager) or sumo (Eclipse SUMO in lockstep, '
                       'needs SUMO in CARLA\'s Python); sumo: {net: auto|generate|a .net.xml or .sumocfg path, '
                       'demand: count|period, period_s, classes, tls: carla|sumo}')
    if kind == 'campaign':
        out['note'] = ('template ("fault_detectability", "weather_sweep", "rig_ab") starts from that design fitted to '
                       'the variant; fault_sets are lists of {kind, tags, axes, magnitudes, signs}')
    if kind in CP.STUDY_KINDS:
        out['what'] = CP.STUDY_ABOUT[kind][0]
        out['plan_only_options'] = {k: v for k, v in CP.STUDY_EXTRAS[kind].items()}
        out['note'] = STUDY_NOTES.get(kind, '') + (' ' if STUDY_NOTES.get(kind) else '') + (
            'Nothing runs until the user approves; the result goes to the page and to the read tools.')
    if kind == 'experiment':
        out['note'] = ('factors: [{kind: camera|sensor|cameras|roi|load_case|weather, target: a tag, attr: x y z yaw '
                       'pitch roll | lens | model | distance | case | condition, mode: value|offset, lo, hi, levels, '
                       'choices}]; responses: [{metric, params, op, target}] (get_requirement_metrics lists metrics)')
    return PV.scrub(out)


STUDY_NOTES = {
    'rig_perception': 'recordings: [{path (as get_catalogue recordings names it), rig (a name; default the '
                      'recording\'s variant)}], the first rig the reference; source carla drives the perception scene '
                      'set (scene_set.set_id standard | quick, scene_set.variants by name, scene_set.seeds) in CARLA '
                      'first; focus: the sectors the user asked about (the study measures every region); records: hand '
                      'the perception records to the learned model\'s store.',
    'foundation_range': 'source recordings (recordings: drives with ground truth) or sweep (sweep_cameras, '
                        'sweep_classes, ranges_m, seeds: stood at known ranges in CARLA); detectors and depth_model '
                        'must be downloaded on Simulate ▸ Foundation models; depth_model "" for none.',
    'robust_perception': 'recordings: CARLA drives with ground truth of this rig; mount_from custom with mount_deg '
                         '/ mount_mm for a given error; thermal, vibration, calibration switch the other sources.',
    'redrive': 'drive: the recorded drive (default the Log replay page\'s); variant: the proposed rig by name; reshoot: '
               'also write the re-shot drive (hundreds of MB).',
    'design': 'requirements_from brief reads the requirements and constraints from brief with Vantage\'s parser (the '
              'card lists them; approving accepts them); page uses the ones accepted on Analysis ▸ Design agent. '
              'Never creates a variant.',
    'perf_model_train': 'save: keep the trained model where the app loads it.',
    'perf_model_campaign': 'k rigs a round for rounds rounds, from variants (by name; [] every variant with a camera); '
                           'each driven duration_s in CARLA (0: the CARLA page\'s) and evaluated by rig perception.',
    'risk_coverage': 'weight frequency | cost | fatal | fyl | ncap; metric resolved | seen | redundant.',
    'fleet_twin': 'action run (fit the twin) or watch_once (read the vehicles\' watch folders once, then fit).',
}


def widen_schemas(T):
    """Offer every plan kind in the Copilot tools' schemas (tools.py lists the round-7 kinds; the study kinds were
    added in round 15): the kind enum of a step and of get_plan_settings, the topics of explain_limit, and the
    kinds propose_plan's description names. Idempotent."""
    kinds = list(CP.KINDS)
    item = getattr(T, '_STEP_ITEM', None)
    if isinstance(item, dict):
        item['properties']['kind']['enum'] = kinds
    for t in getattr(T, 'TOOLS', []):
        if t.name == 'get_plan_settings' and isinstance(t.params.get('kind'), dict):
            t.params['kind'] = dict(t.params['kind'], enum=kinds)
        elif t.name == 'explain_limit' and isinstance(t.params.get('topic'), dict):
            t.params['topic'] = dict(t.params['topic'], enum=list(CP.LIMITS_DOC))
        elif t.name in ('propose_plan', 'estimate_plan', 'validate_plan'):
            steps = t.params.get('steps')
            if isinstance(steps, dict) and isinstance(steps.get('items'), dict):
                steps['items']['properties']['kind']['enum'] = kinds
            if t.name == 'propose_plan' and 'rig_perception' not in t.description:
                t.description = t.description.replace(
                    'sample_dataset, study or export', 'sample_dataset, study, export, or one of round 14\'s studies '
                    '(rig_perception, foundation_range, robust_perception, redrive, design, perf_model_train, '
                    'perf_model_campaign, risk_coverage, fleet_twin: get_catalogue plan_kinds says what each is for)')


def explain_limit(tc, topic='', **_):
    from ..sim import carla_bridge as CB
    if topic not in CP.LIMITS_DOC:
        return {'error': f'topic must be one of {", ".join(CP.LIMITS_DOC)}'}
    return {'topic': topic, 'limit': CP.LIMITS_DOC[topic].format(budget=CB.RENDER_BUDGET_MPX)}


def _build(tc, steps, title, rationale, revise):
    if not isinstance(steps, list) or not steps:
        raise ValueError('steps must be a non-empty list of {kind, label, values}')
    if revise and tc.plan is None:
        raise ValueError('there is no plan on screen to revise: propose one with revise=false')
    return CP.build(tc.project, steps, title, rationale, prev=tc.plan if revise else None, request=tc.request,
                    provider=tc.provider, model=tc.model, working_dir=tc.working_dir, engine=tc.engine,
                    revise=bool(revise))


def estimate_plan(tc, steps=None, title='', rationale='', revise=False, **_):
    try:
        plan = _build(tc, steps, title, rationale, revise)
    except ValueError as e:
        return {'error': str(e)}
    out = CP.for_model(plan, tc.working_dir)
    return {'estimate': [{'step': s['step'], 'kind': s['kind'], **s['estimate']} for s in out['steps']],
            'note': 'an estimate only: nothing was shown to the user (propose_plan does)'}


def validate_plan(tc, steps=None, title='', rationale='', revise=False, **_):
    try:
        plan = _build(tc, steps, title, rationale, revise)
    except ValueError as e:
        return {'error': str(e)}
    out = CP.for_model(plan, tc.working_dir)
    out['note'] = 'checked only: nothing was shown to the user (propose_plan does)'
    return out


def propose_plan(tc, steps=None, title='', rationale='', revise=False, **_):
    try:
        plan = _build(tc, steps, title, rationale, revise)
    except ValueError as e:
        return {'error': str(e)}
    tc.plan = plan
    if tc.on_copilot is not None:
        tc.on_copilot(plan)
    out = CP.for_model(plan, tc.working_dir)
    out['shown_to_the_user'] = True
    out['next'] = ('invalid: it cannot be approved. Fix the errors with propose_plan(revise=true), or tell the user '
                   'plainly why they cannot be fixed' if not plan.ok else
                   'valid: the user approves, edits or rejects it on the plan card; nothing has run')
    return out
