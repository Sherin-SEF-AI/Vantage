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
    return {'kinds': [{'kind': k, 'label': CP.KIND_LABEL[k], 'page': CP.KIND_PAGE[k] or None,
                       'settings_saved_on_the_page': CP.KIND_SLOT[k] is not None} for k in CP.KINDS],
            'note': 'get_plan_settings(kind) lists a kind\'s settings with their current values and ranges'}


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
        return {'units': ['m', 'mm'], 'angles': ['deg', 'rad'],
                'frame': ['rear_axle', 'front_axle', 'front_bumper', 'ground_centre', 'custom']}.get(key)
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
    if kind == 'experiment':
        out['note'] = ('factors: [{kind: camera|sensor|cameras|roi|load_case|weather, target: a tag, attr: x y z yaw '
                       'pitch roll | lens | model | distance | case | condition, mode: value|offset, lo, hi, levels, '
                       'choices}]; responses: [{metric, params, op, target}] (get_requirement_metrics lists metrics)')
    return PV.scrub(out)


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
