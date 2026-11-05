"""Perception scene sets: CARLA scenes chosen to put objects of every class in every region around the car, recorded
with the same seeds for every rig variant compared, for the rig-perception study (core.rig_perception).

A scene set is a list of the bridge's scenarios (vantage.sim.carla_bridge.SCENARIOS) with their parameters, each
saying what it puts where, driven in one town for a fixed duration with background traffic (vehicles on the traffic
manager: oncoming, followed and passing cars, and CARLA's parked cars where the town has them) and background
pedestrians on CARLA's walker AI (on the pavements left and right, at every distance). Every scenario is driven once
per seed and per variant with the same seed, so the variants see the same scene (CARLA's traffic and the spawn point
follow the seed): the study then compares them paired. The drives run through core.simcampaign's machinery (plan,
run_drives: the rig file built once per variant, each drive kept under its key and reused when run again, the
bridge in CARLA's own Python); only the ground truth the study needs (camera instance labels: who sees what) is
asked for, with the semantic and range images detection_eval also reads.

What a scene set yields per region is counted from the recordings' own annotations at every evaluated instant
(yields()), before any detector runs, so the user sees where the statistics will be thin. Counts are object-instants
(an object seen at 10 instants counts 10), and neighbouring instants are not independent (the study's block bootstrap
accounts for that); a region with a few hundred object-instants from one object is still one object: yields() also
counts distinct objects.

Limits: CARLA's traffic manager decides where the background traffic and walkers go, so a region the town's roads do
not reach (far to the side in a narrow street) stays empty whatever the set; a crossing scenario puts one walker or
cyclist where its parameters say, the rest is the seed's. Town01–03 have no parked cars of their own (Town10HD has).
"""
from __future__ import annotations

import copy
import json
import os
from dataclasses import asdict, dataclass, field

from . import regions as RG

SCHEMA = 'vantage.perception_scene_set/1'

# (scenario name, params, what it puts where)
SCENE_SETS = {
    'standard': {
        'label': 'Every class, every region (6 scenes)',
        'about': 'Pedestrians crossing near and far from both sides, a cyclist crossing, a car cutting in from behind '
                 '(overtaking on the left), a lead vehicle followed and braking, and a free drive with traffic and '
                 'walkers all round.',
        'town': 'Town03', 'duration_s': 12.0, 'traffic': 30, 'walkers': 40,
        'scenes': [
            ('pedestrian_crossing', {'side': 'right', 'distance_m': 15.0, 'walker': 'adult', 'traffic': 20.0},
             'an adult crossing from the right 15 m ahead; traffic oncoming and around'),
            ('pedestrian_crossing', {'side': 'left', 'distance_m': 40.0, 'walker': 'child', 'traffic': 20.0},
             'a child crossing from the left 40 m ahead'),
            ('cyclist_crossing', {'side': 'left', 'distance_m': 25.0, 'traffic': 20.0},
             'a cyclist crossing from the left 25 m ahead'),
            ('cut_in', {'side': 'left', 'cutter_offset_m': -15.0, 'cut_at_s': 3.0, 'traffic': 20.0},
             'a car overtaking from behind on the left and cutting in'),
            ('lead_vehicle_brake', {'gap_m': 25.0, 'brake_at_s': 6.0, 'traffic': 20.0},
             'a vehicle followed at 25 m, braking'),
            ('free_drive', {}, 'traffic and walkers all round (the seed\'s)'),
        ]},
    'quick': {
        'label': 'Quick (3 short scenes)',
        'about': 'A pedestrian crossing from the right, a cyclist from the left, and a free drive with traffic and '
                 'walkers: a first look, thin far out.',
        'town': 'Town03', 'duration_s': 8.0, 'traffic': 25, 'walkers': 30,
        'scenes': [
            ('pedestrian_crossing', {'side': 'right', 'distance_m': 20.0, 'traffic': 15.0},
             'an adult crossing from the right 20 m ahead'),
            ('cyclist_crossing', {'side': 'left', 'distance_m': 25.0, 'traffic': 15.0},
             'a cyclist crossing from the left 25 m ahead'),
            ('free_drive', {}, 'traffic and walkers all round'),
        ]},
    'campaign': {
        'label': 'Model training campaign (2 scenes per town, weather and seed)',
        'about': 'A free drive in dense traffic with walkers on both pavements, then a pedestrian crossing from the '
                 'right with the same background: many objects in many regions for little CARLA time. For training '
                 'the learned detection model (core.perf_model) over many towns, weathers and seeds, driven with '
                 'camera banks (many cameras on one drive, rigs evaluated as subsets).',
        'town': 'Town03', 'duration_s': 15.0, 'traffic': 40, 'walkers': 60,
        'scenes': [
            ('free_drive', {}, 'traffic and walkers all round (the seed\'s)'),
            ('pedestrian_crossing', {'side': 'right', 'distance_m': 20.0, 'walker': 'adult', 'traffic': 30.0},
             'an adult crossing from the right 20 m ahead; traffic and walkers around'),
        ]},
}
TOWNS = ('Town01', 'Town02', 'Town03', 'Town04', 'Town05', 'Town10HD')   # CARLA 0.9.16's packaged towns
DEFAULT_SET = 'standard'
LIMITS = {'duration_s': (2.0, 120.0), 'traffic': (0, 150), 'walkers': (0, 150), 'port': (1024, 65000),
          'tm_port': (1024, 65000)}


@dataclass
class SceneSetSettings:
    set_id: str = DEFAULT_SET
    variants: list = field(default_factory=list)       # variant uids; [] = the active one
    seeds: list = field(default_factory=lambda: [1])
    town: str = ''                                     # '' = the set's
    duration_s: float = 0.0                            # 0 = the set's
    traffic: int = -1                                  # −1 = the set's (vehicles on the autopilot)
    walkers: int = -1                                  # −1 = the set's (pedestrians on the walker AI)
    weather: str = ''                                  # '' = the project's CARLA weather
    out_dir: str = ''                                  # '' = <working folder>/<project>/campaigns/<set name>
    port: int = 0                                      # 0 = the project's CARLA port
    tm_port: int = 0

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d) -> 'SceneSetSettings':
        s = SceneSetSettings()
        d = d if isinstance(d, dict) else {}
        if d.get('set_id') in SCENE_SETS:
            s.set_id = d['set_id']
        if isinstance(d.get('variants'), list):
            s.variants = [str(u)[:64] for u in d['variants'] if isinstance(u, str) and u][:8]
        if isinstance(d.get('seeds'), list):
            seeds = []
            for x in d['seeds'][:20]:
                try:
                    v = int(x)
                except (TypeError, ValueError):
                    continue
                if 0 <= v <= 999999 and v not in seeds:
                    seeds.append(v)
            s.seeds = seeds or [1]
        for k in ('town', 'weather', 'out_dir'):
            if isinstance(d.get(k), str):
                setattr(s, k, d[k].strip()[:4096 if k == 'out_dir' else 64])
        for k, (lo, hi) in LIMITS.items():
            if k in d and not isinstance(d[k], bool):
                try:
                    x = float(d[k])
                except (TypeError, ValueError):
                    continue
                if k in ('traffic', 'walkers') and x < 0:
                    setattr(s, k, -1)
                elif k in ('duration_s', 'port', 'tm_port') and x == 0:
                    setattr(s, k, 0 if k != 'duration_s' else 0.0)
                else:
                    x = min(max(x, lo), hi)
                    setattr(s, k, x if k == 'duration_s' else int(round(x)))
        return s


def scene_set(set_id) -> dict:
    return copy.deepcopy(SCENE_SETS.get(set_id) or SCENE_SETS[DEFAULT_SET])


def describe(st: SceneSetSettings) -> list:
    """[{'scenario', 'params', 'puts'}] of the set."""
    return [{'scenario': n, 'params': dict(p), 'puts': t} for n, p, t in scene_set(st.set_id)['scenes']]


def campaign_settings(project, st: SceneSetSettings):
    """The set as a core.simcampaign campaign: every scene × seed × variant, clean drives, ground truth for detection,
    no fault."""
    from . import simcampaign as SC
    ss = scene_set(st.set_id)
    base = {'town': st.town or ss['town'], 'duration_s': float(st.duration_s or ss['duration_s']),
            'traffic': int(ss['traffic'] if st.traffic < 0 else st.traffic),
            'walkers': int(ss['walkers'] if st.walkers < 0 else st.walkers)}
    if st.weather:
        base['weather'] = st.weather
    if st.port:
        base['port'] = int(st.port)
    if st.tm_port:
        base['tm_port'] = int(st.tm_port)
    scen = []
    for name, params, _t in ss['scenes']:
        p = dict(params)
        if name != 'free_drive' and st.traffic >= 0:
            p['traffic'] = float(st.traffic)
        scen.append({'name': name, 'params': p})
    d = {'name': f'Perception scene set {st.set_id}', 'template': 'custom', 'variants': list(st.variants),
         'seeds': list(st.seeds), 'scenarios': scen, 'fault_sets': [], 'clean': True, 'multiplex': False,
         'base': base, 'evaluators': ['detection'], 'asbuilt_on': 'none', 'server': 'running',
         'out_dir': st.out_dir, 'max_drives': 400}
    return SC.CampaignSettings.from_dict(d)


def plan(project, st: SceneSetSettings):
    from . import simcampaign as SC
    return SC.plan(project, campaign_settings(project, st))


def out_dir_of(project, st: SceneSetSettings, base=None) -> str:
    from . import simcampaign as SC
    return SC.campaign_dir(project, campaign_settings(project, st), base)


def estimate(project, st: SceneSetSettings, out_dir=None) -> dict:
    """simcampaign's cost estimate of the set (drives × the measured wall time per drive) and the drives already
    there."""
    from . import simcampaign as SC
    cs = campaign_settings(project, st)
    pl = SC.plan(project, cs)
    e = SC.estimate(project, cs, pl, out_dir or SC.campaign_dir(project, cs))
    e['problems'] = list(pl.problems)
    return e


def run(project, st: SceneSetSettings, out_dir=None, progress=None, cancel=None, runner=None, python=None) -> dict:
    """Drive every scene of the set for every variant (drives already done with the same key are reused):
    {'out_dir', 'states': {drive id: status}, 'recordings': [{'path', 'rig'}]} (core.simcampaign.Cancelled on a
    cancel; RuntimeError when no CARLA server answers)."""
    from . import simcampaign as SC
    cs = campaign_settings(project, st)
    pl = SC.plan(project, cs)
    od = out_dir or SC.campaign_dir(project, cs)
    os.makedirs(od, exist_ok=True)
    states = SC.run_drives(project, cs, pl, od, progress=progress, cancel=cancel, runner=runner, python=python)
    return {'out_dir': od, 'states': {k: v.status for k, v in states.items()},
            'errors': {k: v.error for k, v in states.items() if v.error},
            'recordings': recordings(project, st, od, pl)}


def recordings(project, st: SceneSetSettings, out_dir=None, pl=None) -> list:
    """The set's finished drives as the study's recordings: [{'path', 'rig' (the variant's name)}], variants in the
    set's order, scenes and seeds alike for each."""
    from . import simcampaign as SC
    cs = campaign_settings(project, st)
    pl = pl or SC.plan(project, cs)
    od = out_dir or SC.campaign_dir(project, cs)
    out = []
    for d in pl.drives:
        if d['role'] != 'drive':
            continue
        if SC._drive_done(od, d) is None:
            continue
        out.append({'path': os.path.join(od, 'drives', d['id']), 'rig': d['variant']})
    return out


def yields(recs, grid: RG.RegionGrid | None = None, settings=None, progress=None) -> dict:
    """Ground-truth objects the recordings yield per class and region at the evaluated instants (no detector):
    {'object_instants': {class: {region id: n}}, 'objects': {class: {region id: distinct objects}},
     'per_recording': [...], 'regions_empty': [...], 'evaluated_range_m'}. recs: [{'path', 'rig'}] or paths."""
    from . import rig_perception as RP
    import numpy as np
    st = settings or RP.RigPerceptionSettings()
    grid = grid or RG.RegionGrid()
    inst, objs, per = {}, {}, []
    rng = None
    for k, r in enumerate(recs):
        path = r['path'] if isinstance(r, dict) else r
        pr = RP.PerceptionRecording(path)
        off = float(np.hypot(*grid.origin_xy))
        er = (min(st.max_range_m, pr.gt_range_m) if st.max_range_m > 0 else pr.gt_range_m) - off
        rng = er if rng is None else min(rng, er)
        mine = {}
        for _tok, t, _key in pr.instants(st.instants, st.frame_step):
            gt = pr.gt_at(t, pr.ego_at(t))
            if not len(gt['token']):
                continue
            xy = gt['xy']
            s, rb = grid.assign(xy[:, 0], xy[:, 1])
            ok = (grid.range_m(xy[:, 0], xy[:, 1]) <= er) & ~gt['ignore']
            for i in np.flatnonzero(ok):
                cls = gt['cls'][i]
                rid = grid.region_id(int(s[i]), int(rb[i]))
                for c in (cls, 'vehicle') if cls in RP.GROUP['vehicle'] else (cls,):
                    inst.setdefault(c, {})[rid] = inst.setdefault(c, {}).get(rid, 0) + 1
                    mine.setdefault(c, {})[rid] = mine.setdefault(c, {}).get(rid, 0) + 1
                    objs.setdefault(c, {}).setdefault(rid, set()).add((k, gt['token'][i]))
        per.append({'recording': os.path.basename(path), 'rig': (r.get('rig') if isinstance(r, dict) else '') or
                    pr.rig, 'object_instants': mine})
        if progress is not None:
            progress((k + 1) / max(len(recs), 1), os.path.basename(path))
    distinct = {c: {rid: len(v) for rid, v in d.items()} for c, d in objs.items()}
    empty = {c: [rid for _s, _r, rid, _n in grid.regions() if rid not in inst.get(c, {})] for c in RP.EVAL_CLASSES}
    return {'schema': SCHEMA, 'object_instants': inst, 'objects': distinct, 'per_recording': per,
            'regions_empty': empty, 'evaluated_range_m': rng, 'grid': grid.to_dict()}


def to_doc(st: SceneSetSettings, project=None, yields_doc=None, est=None) -> dict:
    return {'schema': SCHEMA, 'project': getattr(project, 'name', None), 'settings': st.to_dict(),
            'scenes': describe(st), 'set': {k: v for k, v in scene_set(st.set_id).items() if k != 'scenes'},
            'estimate': est, 'yields': yields_doc}


def write(path, doc):
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(doc, fh, indent=1, default=str)
