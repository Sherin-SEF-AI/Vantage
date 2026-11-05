"""The assistant's tools for round 14's studies: perception in the loop, re-driven logs, the design agent, the learned
rig model, foundation models, tolerances carried to detection, risk-weighted coverage and the fleet twin.

What each one does:
- Reading a page's last result (get_rig_perception, get_redrive, get_foundation_range, get_robust_perception): those
  studies run detectors, reconstructions or CARLA for minutes to hours, so the assistant never starts one; it reads
  the page's last result (ToolContext.studies: the page's summary(), no paths) or says what is set up and how long a
  run takes, and the user runs it from the page.
- Cheap work on the snapshot (get_risk_coverage about a second, get_fleet_twin and get_fleet_tolerances seconds and
  cached, predict_detection and suggest_simulations a second or two with the saved model, get_perf_model,
  plan_perception_scene_set, read_requirements with the brief parser): run here, nothing written.
- design_rig runs the design agent on a snapshot within a time budget, as optimise_rig runs the optimiser. Its options
  are never applied here: the user creates one on Analysis ▸ Design agent (one undo step, logged).

Every result is plain JSON in the units the module states; none carries a file path."""
from __future__ import annotations

import math

BUDGET_MAX = 90.0                       # seconds: design_rig, as optimise_rig


def _T():
    from . import tools
    return tools


def _r(v, n=3):
    return _T()._r(v, n)


def _variant(tc, name):
    """The project's variant by name, uid or 1-based index ('' = the active one); ValueError naming them."""
    p = tc.project
    if not name:
        return p.variant
    for i, v in enumerate(p.variants):
        if str(name) in (v.name, v.uid, str(i + 1)) or str(name).lower() == v.name.lower():
            return v
    raise ValueError(f'no variant {name!r}: ' + ', '.join(v.name for v in p.variants))


def _page(tc, name):
    return (getattr(tc, 'studies', None) or {}).get(name)


def _scrub(doc):
    from ..core import provenance as PV
    return PV.scrub(doc)


# =========================================================================== the pages' heavy studies
def get_rig_perception(tc, cls=None, rig=None, **_):
    """The Rig perception page's last result (per rig and class: recall, mAP, ATE with intervals; the significant
    findings), else what is set up."""
    s = _page(tc, 'rig_perception')
    if s:
        out = dict(s)
        if cls:
            out['per_rig'] = {r: {c: v for c, v in d.items() if c == cls} for r, d in out['per_rig'].items()}
        if rig:
            out['per_rig'] = {r: d for r, d in out['per_rig'].items() if r == rig}
        return {'run': True, **_scrub(out),
                'meaning': '3D detection of each class in each region around the car (8 sectors × range bins), per '
                           'rig on the same CARLA scenes; recall, precision and the centre error at a 2 m match, '
                           'nuScenes mean AP; intervals from a block bootstrap over time, paired when two rigs drove '
                           'the same scene; only differences whose interval excludes zero are findings. Monocular '
                           'lifted YOLOX boxes on CARLA renders: compare rigs, not absolute performance.'}
    from ..core import detection as DT
    from ..core import rig_perception as RP
    st = RP.settings_of(tc.project)
    recs = []
    for r in st.recordings:
        i = RP.recording_info(r['path'])
        recs.append({'rig': r.get('rig') or '', 'ok': i['ok'], 'text': i['text'] if i['ok'] else 'not readable'})
    s = DT.status(st.model)
    return {'run': False, 'backend': st.backend, 'detector': {'id': s['id'], 'verified': s['verified']},
            'recordings': recs, 'classes': list(st.classes),
            'note': 'not run yet: Simulate ▸ Rig perception ▸ Run (seconds per drive with the 2D detections cached, '
                    'minutes without; the assistant cannot run it). Drives come from CARLA (plan_perception_scene_set '
                    'says what a scene set costs).'}


def plan_perception_scene_set(tc, set_id='standard', variants=None, seeds=None, **_):
    """A perception scene set's scenes and its cost in CARLA (drives, minutes, drives already there)."""
    from ..core import perception_scenes as PS
    from ..core import rig_perception as RP
    if set_id not in PS.SCENE_SETS:
        return {'error': f'no scene set {set_id!r}: {", ".join(PS.SCENE_SETS)}'}
    try:
        uids = [_variant(tc, n).uid for n in (variants or [])]
    except ValueError as e:
        return {'error': str(e)}
    sd = [int(x) for x in (seeds or [1]) if isinstance(x, (int, float)) and not isinstance(x, bool)] or [1]
    st = PS.SceneSetSettings.from_dict(dict(RP.settings_of(tc.project).scene_set, set_id=set_id, variants=uids,
                                            seeds=sd))
    est = PS.estimate(tc.project, st)
    return {'set': set_id, 'scenes': PS.describe(st), 'variants': [v.name for v in tc.project.variants
                                                                   if v.uid in (uids or [tc.project.variant.uid])],
            'seeds': st.seeds, 'drives': est.get('drives'), 'done': est.get('done'), 'cost': est.get('text'),
            'problems': est.get('problems') or [],
            'note': 'the same scenes and seeds for every variant, so the rigs compare paired; driving it needs CARLA: '
                    'Simulate ▸ Rig perception ▸ Drive (propose_plan cannot run it yet; the user starts it)'}


def get_redrive(tc, **_):
    """The Log replay page's Re-drive tab: its last data-reuse report, else what is set up."""
    s = _page(tc, 'redrive')
    if s:
        return {'run': True, **_scrub(s),
                'meaning': 'the recorded drive re-rendered through the proposed rig; each re-rendered pixel\'s '
                           'expected error (grey levels) is calibrated on frames the reconstruction never saw: the '
                           'share within the error budget per camera with a verdict (reuse, partial, re-collect), and '
                           'what the old rig saw that the new one no longer does. Re-rendered images: moving people '
                           'are rebuilt as rigid bodies, so detection of pedestrians on a re-shot drive is not valid.'}
    from ..core import redrive as RDV
    from ..core import splat_reproject as SP
    sst, st = SP.settings_of(tc.project), RDV.settings_of(tc.project)
    v = tc.project.variant_by_uid(st.variant) if st.variant else None
    return {'run': False, 'drive_chosen': bool(sst.drive), 'proposed_rig': v.name if v is not None else None,
            'error_budget_grey_levels': st.error_budget, 'torch': SP.torch_missing() is None,
            'note': 'not run yet: Simulate ▸ Log replay ▸ Re-drive ▸ Calibrate & report (minutes on a GPU; the '
                    'assistant cannot run it)'}


def get_foundation_range(tc, camera=None, cls=None, **_):
    """The Foundation models page's last result (R50 / R90 per camera, class and model beside the geometric range;
    usable depth per camera), else the models' state."""
    s = _page(tc, 'foundation')
    if s:
        out = dict(s)
        out['ranges'] = [r for r in out.get('ranges') or [] if (not camera or r['camera'] == camera)
                         and (not cls or r['class'] == cls)]
        return {'run': True, **_scrub(out),
                'meaning': 'P(detect) against range by logistic regression with cluster-bootstrap intervals on CARLA '
                           'frames: R50 and R90 (m) per camera, class, model and image kind (fisheye or rectified '
                           'views), beside Vantage\'s geometric pixels-on-target range (a ratio below 1: the geometry '
                           'promises more than the detector delivers); usable depth: the farthest bin whose AbsRel is '
                           'under the threshold. Models trained on photographs, measured on renders.'}
    from ..core import foundation_models as FM
    from ..core import foundation_range as FR
    st = FR.settings_of(tc.project)
    models = []
    for m in FM.catalogue(st.opt_in, verify=False):
        models.append({k: m.get(k) for k in ('id', 'name', 'task', 'licence', 'commercial', 'allowed')} |
                      {'downloaded': bool((m.get('status') or {}).get('present'))})
    return {'run': False, 'source': st.source, 'recordings': len(st.recordings), 'detectors': list(st.detectors),
            'depth_model': st.depth_model, 'models': models, 'torch': FM.torch_missing() is None,
            'note': 'not run yet: Simulate ▸ Foundation models ▸ Run (minutes on a GPU; downloads and runs are the '
                    'user\'s)'}


def get_robust_perception(tc, **_):
    """The Robust perception page's last result (per recording: rig recall nominal, mean and 5–95 % band over builds,
    the ghost rate, what drives the spread), else what is set up."""
    s = _page(tc, 'robust_perception')
    if s:
        return {'run': True, **_scrub(s),
                'meaning': 'every camera\'s true pose drawn per build from the mounting tolerance, the thermal study\'s '
                           'drift, the road vibration and the calibration residual; the recorded images turned exactly '
                           'through the lens, the detector run on each build: rig recall, BEV error, out-of-view and '
                           'ghost rate per class and region with the band over builds; Sobol indices name the source '
                           'and camera that drive the spread. One CARLA drive: build variation, not scene sampling.'}
    from ..core import robust_perception as RB
    st = RB.settings_of(tc.project)
    return {'run': False, 'recordings': len(st.recordings), 'mode': st.mode, 'sources': st.active(),
            'mounting_from': st.mount_from, 'builds': st.samples, 'sobol_n': st.sobol_n,
            'note': 'not run yet: Simulate ▸ Robust perception ▸ Run (tens of minutes per recording with the detector '
                    'on; the assistant cannot run it)'}


# =========================================================================== cheap studies
def get_risk_coverage(tc, weight=None, metric=None, cls=None, variant=None, **_):
    """Risk-weighted coverage on the snapshot (about a second): per class the share of the pre-crash exposure the
    cameras cover beside the share of the ground, the three largest gaps with their scenarios, and requirement
    proposals."""
    from ..core import risk_coverage as RC
    try:
        v = _variant(tc, variant)
    except ValueError as e:
        return {'error': str(e)}
    d = RC.settings_of(tc.project).to_dict()
    if weight:
        d['weight'] = weight
    if metric:
        d['metric'] = metric
    res = RC.run(tc.project, tc.engine, RC.RiskSettings.from_dict(d), variant=v)
    out = RC.summary(res)
    for c, dd in res.classes.items():
        if c in out['classes']:
            out['classes'][c]['gaps'] = [{'region': g.get('name') or g.get('region'),
                                          'uncovered_share_of_exposure': _r(g.get('uncovered_share'), 4),
                                          'scenarios': [f'{t["template"]}: {t["what"]}'
                                                        for t in (g.get('top_uncovered') or [])[:3]]}
                                         for g in (dd.get('gaps') or [])[:3]]
    if cls:
        out['classes'] = {c: x for c, x in out['classes'].items() if c == cls}
    out['requirement_proposals'] = RC.requirement_proposal(res)
    out['meaning'] = ('where the other road user is in the 0.5–4 s before a crash (NHTSA pre-crash scenario '
                      'statistics weighted by frequency, cost, fatal crashes or functional years lost; or the Euro '
                      'NCAP 2026 crash-avoidance scenarios equally), tested with the perception range\'s pixel rule; '
                      'cameras only, clear weather, constant speeds. requirement_proposals are what this rig reaches '
                      'minus 5 %: propose_requirements can add one (metric risk_coverage.weighted_pct).')
    return out


def _twin(tc, horizon_days=None):
    from ..core import fleet_twin as FT
    st = FT.settings_of(tc.project)
    if horizon_days is not None:
        st = FT.FleetTwinSettings.from_dict(dict(st.to_dict(), horizon_days=horizon_days))
    return FT.run(tc.project, tc.engine, st)


def get_fleet_twin(tc, vehicle=None, sensor=None, horizon_days=None, **_):
    """The fleet twin (cached; seconds): urgency counts, the recalibration schedule, knocks, outlier vehicles and the
    findings; with a vehicle and a sensor, that sensor's axes: now ± σ, drift rate ± σ per 30 days, steps."""
    from ..core import fleet_twin as FT
    res = _twin(tc, horizon_days)
    if res is None:
        return {'error': 'the fleet twin could not run'}
    out = FT.summary(res)
    out.pop('assumptions', None)
    if vehicle or sensor:
        doc = FT.to_doc(res, tc.project)
        cells = [c for c in doc['cells'] if (not vehicle or vehicle in (c['vehicle'], c['vehicle_id']))
                 and (not sensor or c['sensor'] == sensor)]
        out['cells'] = [{**{k: c[k] for k in ('vehicle', 'sensor', 'kind', 'urgency', 'p_now', 'p_horizon',
                                              'recalibrate_by', 'median', 'p90', 'binding_axis', 'limit_source')},
                         'knocks': c['knocks'],
                         'axes': {ax: {'unit': a['unit'], 'now': a['now'], 'rate_now_per_30d': a['rate_now_per_30d'],
                                       'limits': a['limits'], 'steps': a['steps'],
                                       'observations': len(a['observations'])}
                                  for ax, a in c['axes'].items()}} for c in cells[:6]]
        if not cells:
            out['cells'] = []
            out['note'] = 'no such vehicle and sensor in the twin (vehicle: a fleet name or id; sensor: a tag)'
    out['meaning'] = ('each sensor axis\'s drift as a smooth trend fitted across the fleet (Kalman filter and smoother, '
                      'maximum likelihood); knocks by a likelihood-ratio test with a stated false-alarm rate; the date '
                      'each sensor is likely beyond its limit (first passage: critical targets\' inverse tolerances, '
                      'else the mounting tolerance). Dates and limits are only as good as the records\' dates.')
    return out


def get_fleet_tolerances(tc, horizon_days=180, **_):
    """The fleet's measured drift as tolerance inputs: per sensor and axis the installation spread, the drift after
    horizon_days and the total (normal; ± = 2 σ per camera), and the ready core.tolerance settings."""
    from ..core import fleet_twin as FT
    h = float(horizon_days if horizon_days is not None else 180)
    if not math.isfinite(h) or h <= 0:
        return {'error': 'horizon_days: a positive number of days'}
    res = _twin(tc)
    if res is None:
        return {'error': 'the fleet twin could not run'}
    doc = FT.tolerance_inputs(res, min(h, 3650.0), project=tc.project)
    doc.pop('provenance', None)
    doc['note'] = ('the page\'s "Use as mounting tolerance" sets the project\'s tolerance studies to the cameras\' '
                   'largest ± 2 σ (one undo step); Simulate ▸ Robust perception can draw the builds from it '
                   '(mounting from the fleet twin)')
    return _scrub(doc)


# =========================================================================== the learned model
def get_perf_model(tc, **_):
    """The learned detection model: whether a model is saved and trained, its records, scenes and rigs, its
    cross-validated calibration, and the store's records."""
    from ..core import perf_model as PM
    s = _page(tc, 'perf_model')
    m = PM.load_model()
    try:
        store = PM.store_summary(PM.store_for(tc.project).records())
    except Exception:                                  # noqa: BLE001 — an unreadable store has no records
        store = {}
    out = {'saved': m is not None, 'trained': bool(m is not None and m.trained),
           'reason': m.reason() if m is not None else 'not trained (no model saved)',
           'store': store}
    if m is not None:
        out.update(records=m.n_records, scenes=m.n_scenes, rigs=m.n_rigs, detector=m.detector,
                   fitted_utc=m.fitted_utc, calibration=PM.cv_table(m), status=m.status)
    if s:
        out['page'] = _scrub(s)
    out['meaning'] = ('a Gaussian process per class on the empirical logit of recall, from the rig\'s geometry alone '
                      '(pixels on the object, cameras seeing a region, lidar points, radar coverage …), trained on '
                      'perception records; its intervals\' coverage is measured by leave-one-scene-out and '
                      'leave-one-rig-out cross-validation (calibration). Few scene sets: a new town, weather or '
                      'detector is outside what it has seen.')
    return out


def predict_detection(tc, variant=None, cls=None, **_):
    """The saved model's predicted recall per class and region with its std, the rig-level figure and the objective."""
    from ..core import perf_model as PM
    try:
        v = _variant(tc, variant)
    except ValueError as e:
        return {'error': str(e)}
    d = PM.predict_detection(tc.project, v, engine=tc.engine)
    if not d.get('trained'):
        return {'trained': False, 'reason': d.get('reason'),
                'note': 'train it on Analysis ▸ Learned detection model (it needs perception records from Rig '
                        'perception, Foundation models or Robust perception)'}
    classes = d.get('classes') or {}
    if cls:
        if cls not in classes:
            return {'error': f'no class {cls!r}: {", ".join(classes)}'}
        d['classes'] = {cls: classes[cls]}
    else:                                              # every class: the six weakest regions each (the rest is long)
        d['classes'] = {c: dict(sorted(r.items(), key=lambda kv: kv[1]['recall'])[:6]) for c, r in classes.items()}
        d['note'] = 'per class the six regions with the lowest predicted recall; name cls for every region'
    d['meaning'] = ('predicted recall (0–1) of the detector the model was trained on, per region id S<sector>R<range '
                    'bin> of the 8 sectors × 0–10 … 80+ m grid, with its std (the model\'s, checked by '
                    'cross-validation); objective: the weighted mean over the classes that matter')
    return d


def suggest_simulations(tc, k=4, **_):
    """Which of the project's variants to evaluate for real next: the learned model's expected improvement."""
    from ..core import perf_model as PM
    try:
        k = max(1, min(int(k or 4), 12))
    except (TypeError, ValueError):
        return {'error': 'k: a whole number'}
    try:
        d = PM.suggest_simulations(tc.project, k)
    except (RuntimeError, ValueError) as e:
        return {'ok': False, 'text': str(e), 'picks': []}
    d['note'] = ('evaluating a pick for real means driving it in CARLA and running Rig perception: the user does it '
                 '(plan_perception_scene_set gives the cost)')
    return d


# =========================================================================== the design agent
def read_requirements(tc, text=None, use_documents=False, **_):
    """Requirements and constraints read by the brief parser from text (default: the design agent's brief) and, with
    use_documents, the design agent's documents: each row with its metric, target and the sentence it quotes."""
    from ..core import docs_ingest as DI
    from ..core import rig_designer as DSG
    from . import designer_llm as DL
    import os
    st = DSG.settings_of(tc.project)
    body = text if isinstance(text, str) and text.strip() else st.brief
    docs = [DI.from_text(body)] if body and body.strip() else []
    gone = []
    if use_documents:
        for d in st.documents:
            if d.get('path') and os.path.isfile(d['path']):
                docs.append(DI.read(d['path'], d.get('tag') or 'D'))
            else:
                gone.append(d.get('name') or '?')
    if not docs:
        return {'error': 'nothing to read: give text, or type the brief on Analysis ▸ Design agent'}
    ex = DL.deterministic(docs, tc.project)
    rows = [{'metric': r['item'].get('metric'), 'params': r['item'].get('params') or {}, 'op': r['item'].get('op'),
             'target': r['item'].get('target'), 'priority': r['item'].get('priority', 'must'),
             'quote': (r.get('source') or {}).get('quote'), 'where': (r.get('source') or {}).get('id')}
            for r in ex.requirements]
    return {'method': ex.method, 'requirements': rows,
            'constraints': [{k: c.get(k) for k in ('kind', 'op', 'value', 'unit', 'priority', 'text')} |
                            {'quote': (c.get('source') or {}).get('quote')} for c in ex.constraints],
            'not_checkable': [{'quote': (n.get('source') or {}).get('quote'), 'why': n.get('why')}
                              for n in ex.not_checkable][:20],
            'missing_documents': gone,
            'note': 'a proposal only: propose_requirements adds rows to the scorecard; the design agent\'s own rows are '
                    'set on Analysis ▸ Design agent ▸ Read (the user accepts them)'}


def design_rig(tc, evaluations=None, options=None, seed=None, budget_s=None, **_):
    """The design agent on a snapshot: 3–7 complete rigs trading cost, power, requirement margin and detection, each
    re-checked and critiqued. Nothing changes."""
    from ..core import rig_designer as DSG
    st = DSG.settings_of(tc.project)
    if not st.items():
        last = _page(tc, 'designer')
        if last:
            return {'run': False, 'last': _scrub(last), 'note': 'no accepted requirement in the project now'}
        return {'error': 'no accepted requirement: read them on Analysis ▸ Design agent (read_requirements shows '
                         'what the brief gives)'}
    d = st.to_dict()
    d['evaluations'] = int(evaluations) if isinstance(evaluations, (int, float)) and evaluations > 0 else min(
        st.evaluations, 80)
    if isinstance(options, (int, float)) and options > 0:
        d['options'] = int(options)
    if isinstance(seed, (int, float)):
        d['seed'] = int(seed)
    b = float(budget_s) if isinstance(budget_s, (int, float)) and budget_s > 0 else 45.0
    d['budget_s'] = max(10.0, min(b, BUDGET_MAX))
    tc.status('Designing rigs from the requirements')
    try:
        res = DSG.run(tc.project, tc.engine, DSG.DesignerSettings.from_dict(d), cancel=tc.cancelled)
    except ValueError as e:
        return {'error': str(e)}
    if res is None:
        return {'error': 'cancelled'}
    out = DSG.summary(res)
    out['note'] = ('nothing changed: offer the option that fits and let the user create it on Analysis ▸ Design agent '
                   '(a new variant in one undo step, written to the design log; a large change set asks for an '
                   'approval). Costs are the user\'s prices; without them a cost is incomplete. Detection is the '
                   'learned model\'s prediction when one is trained, else the geometric resolved share.')
    return out


def _tools():
    from .tools import Tool
    S = {'type': 'string'}
    return [
        Tool('get_rig_perception', 'Read the rig perception study',
             'The Rig perception page\'s last result: 3D detection per class and region around the car for each rig '
             'on the same CARLA scenes (recall, mAP, centre error with intervals; the significant rig-against-rig '
             'findings), else the recordings and detector set up. Use it for "how well will the perception work", '
             '"which rig detects pedestrians better at the left". Cheap: it reads the page, never runs it.',
             {'cls': {**S, 'description': 'one class (pedestrian, cyclist, motorcyclist, car, truck, bus, vehicle)'},
              'rig': {**S, 'description': 'one rig by its name on the page'}}, get_rig_perception),
        Tool('plan_perception_scene_set', 'Cost a perception scene set',
             'A perception scene set (standard: 6 scenes; quick: 3) for the given variants and seeds: its scenes, the '
             'drives, the minutes of CARLA and the drives already done. Use it before suggesting a new evaluation in '
             'CARLA. Nothing runs.',
             {'set_id': {**S, 'enum': ['standard', 'quick']},
              'variants': {'type': 'array', 'items': S, 'description': 'variant names (default the active one)'},
              'seeds': {'type': 'array', 'items': {'type': 'integer'}, 'description': 'default [1]'}},
             plan_perception_scene_set),
        Tool('get_redrive', 'Read the re-drive report',
             'The Log replay page\'s Re-drive tab: how much of a recorded drive a proposed rig can reuse (per camera '
             'the share within the error budget and a verdict, the reasons, what the old rig saw that the new one '
             'does not), else what is set up. Use it for "do we lose our data if we change the rig". Cheap: it reads '
             'the page.', {}, get_redrive),
        Tool('get_foundation_range', 'Read the foundation-model ranges',
             'The Foundation models page\'s last result: how far each detector (OWLv2, Grounding DINO, YOLOX) finds '
             'each class with each camera (R50, R90 with intervals, beside the geometric range) and how far '
             'monocular depth is usable, else the models\' state. Use it when the geometric perception range seems '
             'optimistic. Cheap: it reads the page.',
             {'camera': {**S, 'description': 'a camera tag'}, 'cls': {**S, 'description': 'a class name'}},
             get_foundation_range),
        Tool('get_robust_perception', 'Read the robust perception study',
             'The Robust perception page\'s last result: the tolerances (mounting, thermal, vibration, calibration) '
             'carried to detection: rig recall with its band over builds, BEV error, ghosts, and which source and '
             'camera drive the spread. Use it for "does the mounting tolerance hurt detection". Cheap: it reads the '
             'page.', {}, get_robust_perception),
        Tool('get_risk_coverage', 'Compute risk-weighted coverage',
             'Coverage weighted by where crashes really start (NHTSA pre-crash scenarios, Euro NCAP tests): per class '
             'the share of the pre-crash exposure the cameras cover beside the share of the ground, the largest gaps '
             'with their scenarios, and requirement proposals. About a second. Use it when ground coverage looks '
             'fine but the question is safety-relevant coverage.',
             {'weight': {**S, 'enum': ['frequency', 'cost', 'fatal', 'fyl', 'ncap']},
              'metric': {**S, 'enum': ['resolved', 'seen', 'redundant']},
              'cls': {**S, 'enum': ['pedestrian', 'cyclist', 'vehicle', 'motorcyclist']},
              'variant': {**S, 'description': 'variant name (default the active one)'}}, get_risk_coverage, 'compute'),
        Tool('get_fleet_twin', 'Read the fleet twin',
             'Every vehicle\'s sensors over time: urgency counts, the recalibration schedule (date each sensor is '
             'likely beyond its limit), knocks, outlier vehicles and findings; with vehicle and sensor, that '
             'sensor\'s axes (now ± σ, drift rate, steps). Seconds, cached. Use it for drift, recalibration and '
             '"which car needs attention".',
             {'vehicle': {**S, 'description': 'a fleet vehicle name or id'}, 'sensor': {**S, 'description': 'a tag'},
              'horizon_days': {'type': 'number', 'description': 'forecast horizon (default the project\'s, 365)'}},
             get_fleet_twin),
        Tool('get_fleet_tolerances', 'Read the measured fleet tolerances',
             'The fleet\'s measured drift as tolerance inputs: per sensor and axis the installation spread and the '
             'drift after horizon_days (normal, ± = 2 σ), and ready tolerance-study settings. Use it to replace an '
             'assumed mounting tolerance with what the fleet shows.',
             {'horizon_days': {'type': 'number', 'description': 'default 180'}}, get_fleet_tolerances),
        Tool('get_perf_model', 'Read the learned detection model',
             'The learned rig-performance model: saved and trained or why not, its records, scenes and rigs, its '
             'cross-validated calibration (interval coverage, RMSE against baselines) and the record store. Use it '
             'before quoting predict_detection.', {}, get_perf_model),
        Tool('predict_detection', 'Predict detection per region',
             'The learned model\'s predicted recall per class and region for a variant, with its std, the rig-level '
             'figure and the objective; without a trained model, why not. A second or two. Use it to compare rigs '
             'without simulating them; check get_perf_model\'s calibration first.',
             {'variant': {**S, 'description': 'variant name (default the active one)'},
              'cls': {**S, 'description': 'one class, every region (default: each class\'s six weakest regions)'}},
             predict_detection, 'compute'),
        Tool('suggest_simulations', 'Suggest rigs to simulate',
             'Which of the project\'s variants are worth evaluating for real next (the learned model\'s expected '
             'improvement over the best evaluated rig), each with why. Use it when deciding what to drive in CARLA.',
             {'k': {'type': 'integer', 'description': 'how many (default 4)'}}, suggest_simulations, 'compute'),
        Tool('read_requirements', 'Read requirements from a brief',
             'Requirements and constraints the brief parser reads from text (default: the design agent\'s brief), '
             'and with use_documents from the design agent\'s documents too (their sentences are then quoted): each '
             'row with its metric, target and the sentence it comes from; what cannot be checked is listed. A '
             'proposal only.',
             {'text': {**S, 'description': 'the brief to read (default the design agent\'s)'},
              'use_documents': {'type': 'boolean', 'description': 'also read the design agent\'s documents'}},
             read_requirements),
        Tool('design_rig', 'Design rigs from the requirements',
             'Run the design agent on the accepted requirements: a search over lenses, modules, camera count, '
             'lidars, radars and aim returning 3–7 complete rigs that trade cost (the user\'s prices), power, the '
             'worst requirement margin and detection, each re-checked, explained and critiqued. Takes up to budget_s '
             'seconds. Use it for "design me a rig that …" once requirements are accepted; for moving the current '
             'cameras use optimise_rig. Nothing changes: the user creates an option on the page.',
             {'evaluations': {'type': 'integer', 'description': 'rigs scored (default 80 here)'},
              'options': {'type': 'integer', 'description': '3-7'},
              'seed': {'type': 'integer'},
              'budget_s': {'type': 'number', 'description': 'seconds, 10-90 (default 45)'}},
             design_rig, 'compute'),
    ]


def register(T=None):
    """Add TOOLS_R14 to tools.TOOLS and tools.BY_NAME once (idempotent)."""
    T = T or _T()
    have = {t.name for t in T.TOOLS}
    for t in TOOLS_R14:
        if t.name not in have:
            T.TOOLS.append(t)
        T.BY_NAME[t.name] = T.BY_NAME.get(t.name, t)


TOOLS_R14 = _tools()

# tools.py imports this module at its end (TOOLS += TOOLS_R14); imported first instead, register when tools is done
import sys as _sys                                                     # noqa: E402
_tm = _sys.modules.get(__package__ + '.tools')
if (_tm is not None and not getattr(getattr(_tm, '__spec__', None), '_initializing', False)
        and hasattr(_tm, 'BY_NAME') and 'get_fleet_twin' not in _tm.BY_NAME):
    register(_tm)
del _sys, _tm
