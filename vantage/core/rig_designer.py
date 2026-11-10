"""The design agent: requirements in, a Pareto set of complete rigs out, each explained and critiqued, nothing applied
until the user accepts one.

Inputs: the requirements the user accepted (core.requirements items, each citing the sentence of the brief or a
document it was read from: assistant/designer_llm.py), the constraints no study measures (a budget, sensor counts, a
mounting tolerance) and the prices the user entered. Must requirements and constraints are constraints of the search;
the objectives are

- cost: the sum of the prices of every camera module, lens and sensor of the rig, from the prices entered on the page
  (Vantage's library has no prices). A rig with an item that has no price has no cost: the cost objective is then
  "incomplete" for it, stated, never assumed, and a budget cannot be checked on it;
- power: the rig's steady load from the power budget (core.power: every device and the fixed loads it is set up with);
- requirement margin: the worst must requirement's margin relative to its target (margin / max(|target|, a unit
  floor)), positive when every must requirement passes;
- expected detection: when perf_model.load_model() returns a trained model, its predicted detection per class and
  region (core.regions grid) with its uncertainty, area-weighted; else the geometric resolved share: the share of the
  region grid's ground (to the open bin's edge + 20 m) where some camera resolves the requirement classes at the
  perception range's minimum pixel height, said to be "geometric only".

Search, deterministic for a seed (budgeted in evaluations; no time limit by default, so the options do not depend
on how busy the machine is; a time limit set by hand only cuts it short, and says so):

1. seeds: the active variant, the project's other variants with cameras, and the reference rigs whose layout is
   published (core.reference_rigs) scaled to this vehicle's footprint and snapped onto its body;
2. an outer evolutionary loop over the sensor SET: lens and module swaps from the library (the project's own custom
   hardware included), cameras (or mirrored pairs) added beside an existing mount or removed, lidar and radar models
   swapped, added or removed; locked cameras and sensors are the user's and never change. Every candidate is scored on
   the screening metrics: the cheap studies plus the perception range and the ground coverage (measured fast enough
   for every candidate); the other study metrics only on the front;
3. the inner placement search: the front's rigs through core.optimiser (yaw and pitch of each camera within a few
   degrees, blind ground under the stated tolerance), each placed rig scored again;
4. the final set: the front evaluated in full (every requirement, every study, the Issues list), constrained Pareto
   dominance again, 3–7 options spread along the front.

For each option: the rig as a new variant (not applied), every requirement's pass/fail and margin, cost and power,
predicted or geometric detection, the robust check (every must requirement in random builds within the stated
mounting tolerance), and why each sensor is there: its marginal contribution measured by removing it and scoring the
rig again (the requirements that fail without it, the margins it carries, the ground only it sees, the detection share
it adds) and the requirement it serves. The self-critique is computed: the Issues list's findings, blind ground near
the car, seams across critical targets, calibration observability gaps (core.observability), the worst single camera
off by the tolerance (core.tolerance), lidars and radars inside a camera's image, the power and cost headroom; ranked
by severity.

Accepting an option adds its variant (one undo step with the design-log entries: the brief, the requirements with
their sources, the options considered, the one accepted and why the others were not).
"""
from __future__ import annotations

import copy
import csv
import datetime as _dt
import hashlib
import io
import json
import math
import os
import time
from dataclasses import dataclass, field

import numpy as np

from . import requirements as RQ

SCHEMA = 'vantage.rig_designer/1'
OBJECTIVES = ('cost', 'power', 'margin', 'detection')
OBJECTIVE_LABEL = {'cost': 'Cost', 'power': 'Power (W)', 'margin': 'Requirement margin', 'detection': 'Detection',
                   'cameras': 'Cameras', 'sensors': 'Sensors'}
AXES = ('cost', 'power', 'margin', 'detection', 'cameras', 'sensors')
SCREEN_STUDIES = ('perception', 'coverage')        # study metrics fast enough to score every candidate (≈0.2 s each)
LIMITS = {'options': (3, 7), 'budget_s': (0.0, 1800.0), 'evaluations': (10, 5000), 'seed': (0, 2 ** 31 - 1)}
MAX_CAMERAS = 8                                    # core.project.MAX_CAMERAS: the colour slots and GMSL ports
MAX_LIDARS, MAX_RADARS = 3, 8
SIDE_OFFSET_M = 0.06                               # a camera added beside an existing mount
ROBUST_SAMPLES = 8
PRICE_KINDS = ('module', 'lens', 'sensor')
UNIT_FLOOR = {'m': 1.0, 'm²': 0.01, 'm³': 0.01, '%': 1.0, 'W': 1.0, 'ms': 0.1, 'px': 0.1, 'px/cm': 0.1, '°': 0.1,
              'mm': 1.0, 'h': 0.1, 'min': 1.0, 's': 0.1, 'Hz': 1.0}
SEVERITIES = ('critical', 'serious', 'warning', 'note')
ASSUMPTIONS = (
    'Every figure is a Vantage study\'s own, as on its page (the requirements scorecard reads them): their modelling '
    'limits apply unchanged. Cost is the sum of the prices entered on the page (no price: the cost is incomplete, '
    'never guessed). Power is the power budget\'s steady load with its fixed loads (compute, recorder, switch, time '
    'server). Detection without a trained perception model is geometric only: the share of the ground around the car '
    'where a camera resolves the class at its minimum pixel height, not a detector\'s recall.')
LIMITS_TEXT = (
    'Candidates are screened on the cheap studies, the perception range and the ground coverage; the other study '
    'metrics are evaluated on the front only, so a rig they would fail can leave the search early only when the '
    'screening metrics fail it too. The placement search turns cameras within a few degrees (yaw ±8°, pitch ±5°) for '
    'the least blind ground under the tolerance; it does not slide mounts. Added cameras sit beside an existing mount '
    '(6 cm off) or on a reference rig\'s scaled position; a new mount is a point on the body, not a bracket design. '
    'Marginal contributions are measured on the screening metrics. The robust check moves every camera within the '
    'stated tolerance in random builds, keeping each camera\'s occlusion map of its planned position (as the '
    'tolerance study does). At most 8 cameras.')


class _Stop(Exception):
    pass


# =========================================================================== settings
def _num(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _bool(v, default):
    return v if isinstance(v, bool) else default


@dataclass
class DesignerSettings:
    brief: str = ''
    documents: list = field(default_factory=list)     # [{'name', 'path'}]
    requirements: list = field(default_factory=list)  # [{'item', 'source', 'accepted', 'origin', 'metric_label'}]
    constraints: list = field(default_factory=list)   # [{'kind', 'op', 'value', 'unit', 'priority', 'text', 'source',
    #                                                    'accepted', 'origin'}]
    not_checkable: list = field(default_factory=list)
    rejected: list = field(default_factory=list)
    extraction: dict = field(default_factory=dict)    # method, model, usage, documents (designer_llm.Extraction)
    prices: dict = field(default_factory=dict)        # 'module:<id>' | 'lens:<id>' | 'sensor:<id>' -> price
    currency: str = 'USD'
    options: int = 5
    evaluations: int = 160
    budget_s: float = 0.0               # seconds; 0: no time limit (the evaluations alone decide: reproducible)
    seed: int = 7
    axes: list = field(default_factory=lambda: ['cost', 'margin'])
    vary_lenses: bool = True
    vary_count: bool = True
    vary_lidar: bool = True
    vary_radar: bool = True
    placement: bool = True
    reference_seeds: bool = True
    accepted: list = field(default_factory=list)      # [{'option', 'variant', 'variant_uid', 'date', 'why'}]
    approvals: list = field(default_factory=list)     # [{'option', 'sha256', 'changes', 'approved_utc'}]

    def to_dict(self):
        return {k: copy.deepcopy(getattr(self, k)) for k in self.__dataclass_fields__}

    @staticmethod
    def from_dict(d) -> 'DesignerSettings':
        s = DesignerSettings()
        d = d if isinstance(d, dict) else {}
        if isinstance(d.get('brief'), str):
            s.brief = d['brief'][:20000]
        for k in ('documents', 'not_checkable', 'rejected', 'accepted', 'approvals'):
            if isinstance(d.get(k), list):
                setattr(s, k, [x for x in d[k] if isinstance(x, dict)][:500])
        taken = []
        for r in d.get('requirements') or [] if isinstance(d.get('requirements'), list) else []:
            if not isinstance(r, dict):
                continue
            it = RQ.clean_item(r.get('item'), taken=taken)
            if it is None:
                continue
            if it['id'] in taken:
                it['id'] = RQ._new_id(taken)
            taken.append(it['id'])
            s.requirements.append({'item': it, 'source': r.get('source') if isinstance(r.get('source'), dict) else {},
                                   'accepted': _bool(r.get('accepted'), True), 'origin': str(r.get('origin') or 'user'),
                                   'metric_label': str(r.get('metric_label') or '')})
        for c in d.get('constraints') or [] if isinstance(d.get('constraints'), list) else []:
            if not isinstance(c, dict) or c.get('kind') not in RQ.CONSTRAINT_KINDS or _num(c.get('value')) is None:
                continue
            s.constraints.append({'kind': c['kind'], 'op': c.get('op') if c.get('op') in RQ.OPS else '<=',
                                  'value': float(c['value']), 'unit': str(c.get('unit') or '')[:8],
                                  'priority': c.get('priority') if c.get('priority') in RQ.PRIORITIES else 'must',
                                  'text': str(c.get('text') or '')[:300],
                                  'source': c.get('source') if isinstance(c.get('source'), dict) else {},
                                  'accepted': _bool(c.get('accepted'), True), 'origin': str(c.get('origin') or 'user')})
        if isinstance(d.get('extraction'), dict):
            s.extraction = dict(d['extraction'])
        prices = d.get('prices') if isinstance(d.get('prices'), dict) else {}
        for k, v in prices.items():
            v = _num(v)
            if isinstance(k, str) and k.split(':', 1)[0] in PRICE_KINDS and v is not None and v >= 0:
                s.prices[k[:80]] = float(v)
        if isinstance(d.get('currency'), str) and d['currency'].strip():
            s.currency = d['currency'].strip().upper()[:4]
        for k in ('options', 'evaluations', 'seed'):
            v = _num(d.get(k))
            if v is not None:
                lo, hi = LIMITS[k]
                setattr(s, k, int(round(min(max(v, lo), hi))))
        v = _num(d.get('budget_s'))
        if v is not None:
            s.budget_s = 0.0 if v <= 0 else float(min(max(v, 10.0), LIMITS['budget_s'][1]))   # under 10 s: 10
        ax = d.get('axes')
        if isinstance(ax, list) and len(ax) == 2 and all(a in AXES for a in ax) and ax[0] != ax[1]:
            s.axes = list(ax)
        for k in ('vary_lenses', 'vary_count', 'vary_lidar', 'vary_radar', 'placement', 'reference_seeds'):
            setattr(s, k, _bool(d.get(k), getattr(s, k)))
        return s

    # ---- what the search works with
    def items(self) -> list:
        return [dict(r['item'], params=dict(r['item'].get('params') or {})) for r in self.requirements
                if r.get('accepted', True)]

    def active_constraints(self) -> list:
        return [c for c in self.constraints if c.get('accepted', True)]


def settings_of(project) -> DesignerSettings:
    return DesignerSettings.from_dict(getattr(project.analysis, 'designer', None))


def constraint_set(st: DesignerSettings) -> dict:
    """The tightest accepted constraint of each kind: {kind: constraint row}."""
    out = {}
    for c in st.active_constraints():
        k = c['kind']
        if k not in out:
            out[k] = c
        elif k.startswith('max_') or k == 'budget':
            out[k] = min(out[k], c, key=lambda x: x['value'])
        elif k == 'min_cameras':
            out[k] = max(out[k], c, key=lambda x: x['value'])
        else:                                       # a tolerance: the largest one stated
            out[k] = max(out[k], c, key=lambda x: x['value'])
    return out


# =========================================================================== rigs
@dataclass
class Rig:
    """A complete rig as the search changes it: camera and sensor records (Camera.to_dict / Sensor.to_dict without
    uids), mirrored pairs by tag, what made it."""
    cams: list
    sensors: list
    origin: str = ''
    history: list = field(default_factory=list)

    def key(self) -> str:
        def r(v):
            return round(float(v), 3)
        cams = sorted((c['tag'], c['module_id'], c['lens_id'], r(c['x']), r(c['y']), r(c['z']), r(c['yaw']),
                       r(c['pitch']), r(c['roll'])) for c in self.cams)
        sens = sorted((s['kind'], s['model_id'], r(s['x']), r(s['y']), r(s['z']), r(s['yaw'])) for s in self.sensors)
        return hashlib.sha1(json.dumps([cams, sens]).encode()).hexdigest()[:16]

    def copy(self) -> 'Rig':
        return Rig(copy.deepcopy(self.cams), copy.deepcopy(self.sensors), self.origin, list(self.history))

    def tags(self) -> set:
        return {c['tag'] for c in self.cams} | {s['tag'] for s in self.sensors}

    def cam(self, tag):
        return next((c for c in self.cams if c['tag'] == tag), None)

    def counts(self) -> dict:
        out = {'cameras': len(self.cams), 'lidars': 0, 'radars': 0, 'gnss': 0, 'imu': 0}
        for s in self.sensors:
            k = {'lidar': 'lidars', 'radar': 'radars'}.get(s['kind'], s['kind'])
            out[k] = out.get(k, 0) + 1
        out['sensors'] = len(self.cams) + len(self.sensors)
        return out

    def to_variant(self, name='Design option', note=''):
        from .project import Camera, Sensor, Variant, new_uid
        v = Variant(name=name, note=note)
        uid = {}
        for i, c in enumerate(self.cams):
            d = {k: val for k, val in c.items() if k not in ('link', 'origin')}
            cam = Camera.from_dict(d)
            cam.uid = new_uid()
            cam.calib = {}
            cam.slot = i % MAX_CAMERAS
            uid[c['tag']] = cam.uid
            v.cameras.append(cam)
        for c, cam in zip(self.cams, v.cameras):
            cam.link_uid = uid.get(c.get('link') or '', '')
        for s in self.sensors:
            d = {k: val for k, val in s.items() if k != 'origin'}
            sen = Sensor.from_dict(d)
            sen.uid = new_uid()
            v.sensors.append(sen)
        return v

    @staticmethod
    def from_variant(v, origin='') -> 'Rig':
        tag_of = {c.uid: c.tag for c in v.cameras}
        cams = []
        for c in v.cameras:
            if not c.visible:
                continue
            d = c.to_dict()
            d.pop('uid', None)
            d.pop('link_uid', None)
            d.pop('calib', None)
            d['link'] = tag_of.get(c.link_uid, '') if c.link_uid else ''
            cams.append(d)
        vis = {d['tag'] for d in cams}
        for d in cams:
            if d['link'] not in vis:
                d['link'] = ''
        sens = []
        for s in v.sensors:
            if not s.visible:
                continue
            d = s.to_dict()
            d.pop('uid', None)
            sens.append(d)
        return Rig(cams, sens, origin)


def _with(project, v):
    """The project with variant v added and active: a shallow copy, the live project untouched."""
    q = copy.copy(project)
    q.variants = list(project.variants) + [v]
    q.active_uid = v.uid
    return q


# =========================================================================== cost, power, detection
def price_key(kind: str, iid: str) -> str:
    return f'{kind}:{iid}'


def bill(project, rig: Rig, prices: dict) -> dict:
    """{'cost' (None when an item has no price), 'lines': [{'key', 'what', 'qty', 'price'}], 'missing': [keys]}."""
    lines = {}

    def add(kind, iid, what):
        k = price_key(kind, iid)
        ln = lines.setdefault(k, {'key': k, 'what': what, 'qty': 0, 'price': prices.get(k)})
        ln['qty'] += 1
    for c in rig.cams:
        m = project.modules.get(c['module_id'])
        ln = project.lenses.get(c['lens_id'])
        add('module', c['module_id'], f'{m.vendor} {m.name}'.strip() if m else c['module_id'])
        add('lens', c['lens_id'], (ln.name + (f' ({ln.label})' if ln.label else '')) if ln else c['lens_id'])
    for s in rig.sensors:
        m = project.sensor_models.get(s['model_id'])
        add('sensor', s['model_id'], f'{m.vendor} {m.name}'.strip() if m else s['model_id'])
    out = list(lines.values())
    missing = [ln['key'] for ln in out if ln['price'] is None]
    cost = None if missing else float(sum(ln['qty'] * ln['price'] for ln in out))
    return {'cost': cost, 'lines': out, 'missing': missing}


def _power(project, v) -> tuple[float, list]:
    from . import power as PW
    res = PW.run(project, variant=v)
    return float(res.totals['steady_w']), [r['tag'] for r in res.rows if r['source'] == 'missing']


def perf_model_status():
    """(model or None, text): the learned perception model when perf_model is part of this build and trained."""
    try:
        from . import perf_model as PM
    except ImportError:
        return None, 'geometric only: the learned perception model (core.perf_model) is not part of this build'
    try:
        m = PM.load_model()
    except Exception as e:                            # a damaged store is no reason to stop the design agent
        return None, f'geometric only: the learned perception model could not be loaded ({type(e).__name__}: {e})'
    if m is None:
        return None, 'geometric only: no trained perception model yet (Analysis ▸ Learned rig model)'
    return m, 'learned: the perception model\'s predicted detection per class and region, with its uncertainty'


def _classes_of(items, project) -> list:
    """The perception classes the requirements name (every class when none names one)."""
    from .perception import PerceptionSettings
    names = [c['name'] for c in PerceptionSettings.from_dict(project.analysis.perception).classes]
    want = []
    for it in items:
        c = (it.get('params') or {}).get('cls')
        if c and c in names and c not in want:
            want.append(c)
    return want or names


def geometric_detection(pr, grid, classes) -> tuple[float, dict]:
    """(share, {cls: {region id: share}}): the area-weighted share of each region's ground where some camera
    resolves the class (the perception study's count > 0), to the open bin's edge + 20 m."""
    if pr is None:
        return 0.0, {}
    names = [c.name for c in pr.classes]
    r = np.asarray(pr.ranges, float)
    az = np.radians(np.asarray(pr.az, float))
    edges = np.r_[r[0], np.sqrt(r[:-1] * r[1:]), r[-1]]
    dr = np.diff(edges)
    w = (r * dr)[None, :] * np.ones((len(az), 1)) * (2 * math.pi / max(len(az), 1))
    x = pr.centre[0] + np.cos(az)[:, None] * r[None, :]
    y = pr.centre[1] + np.sin(az)[:, None] * r[None, :]
    s, b = grid.assign(x.ravel(), y.ravel())
    cap = float(grid.range_edges[-1]) + 20.0 if grid.max_range_m <= 0 else grid.max_range_m
    rr = grid.range_m(x.ravel(), y.ravel())
    ok = (b >= 0) & (rr <= cap)
    per, tot_num, tot_den = {}, 0.0, 0.0
    for cname in classes:
        if cname not in names:
            continue
        k = names.index(cname)
        seen = (pr.count[k] > 0).ravel().astype(float)
        ww = w.ravel()
        reg = {}
        for si in range(grid.sectors):
            for bi in range(grid.n_ranges):
                m = ok & (s == si) & (b == bi)
                den = float(ww[m].sum())
                if den > 0:
                    reg[grid.region_id(si, bi)] = round(float((ww[m] * seen[m]).sum() / den), 4)
        per[cname] = reg
        tot_num += float((ww[ok] * seen[ok]).sum())
        tot_den += float(ww[ok].sum())
    return (tot_num / tot_den if tot_den > 0 else 0.0), per


def learned_detection(model, project, v, grid, classes):
    """(mean, std, {cls: {rid: (mean, std)}}) from the perception model, area-weighted over the regions; None when the
    model cannot describe or predict this rig."""
    from . import perf_model as PM
    q = _with(project, v)
    try:
        feats = PM.describe_rig(q, v, grid)
        mean, std = model.predict(feats)
    except Exception:
        return None
    if not isinstance(mean, dict):
        return None
    num = den = var = 0.0
    per = {}
    for c in classes:
        mc = mean.get(c) or mean.get(c.lower()) or {}
        sc = (std or {}).get(c) or (std or {}).get(c.lower()) or {}
        per[c] = {}
        for si, bi, rid, _ in grid.regions():
            if rid not in mc:
                continue
            a = grid.area_m2(si, bi)
            m_, s_ = float(mc[rid]), float(sc.get(rid, 0.0) or 0.0)
            per[c][rid] = (round(m_, 4), round(s_, 4))
            num += a * m_
            var += (a * s_) ** 2
            den += a
    if den <= 0:
        return None
    return num / den, math.sqrt(var) / den, per


# =========================================================================== evaluating a rig
@dataclass
class Eval:
    rig: Rig
    level: str                          # 'screen' | 'full'
    items: list                         # [{'id', 'text', 'metric', 'priority', 'status', 'value', 'margin', 'rel',
    #                                       'unit', 'target', 'op', 'reason'}]
    margin: float                       # worst relative must margin (inf without an evaluated must requirement)
    must_fail: list                     # ids of must requirements that fail or cannot be evaluated
    should_pass: int
    cost: float | None
    cost_missing: list
    power: float
    power_missing: list
    detection: float
    detection_std: float | None
    detection_kind: str                 # 'learned' | 'geometric'
    counts: dict
    constraints: list                   # [{'kind', 'text', 'op', 'value', 'actual', 'status', 'margin'}]
    violation: float
    feasible: bool
    regions: dict = field(default_factory=dict)        # {cls: {rid: share or (mean, std)}}
    res: object = field(default=None, repr=False)      # the RequirementsResult (study results inside)
    variant: object = field(default=None, repr=False)
    robust: dict | None = None

    def objectives(self) -> tuple:
        """Smaller is better: cost (None: unknown), power, −margin, −detection."""
        m = self.margin if math.isfinite(self.margin) else 10.0
        return (self.cost, self.power, -m, -self.detection)


def _rel(margin, item) -> float:
    m = RQ.METRICS.get(item['metric'])
    floor = UNIT_FLOOR.get(m.unit if m else '', 1.0)
    return float(margin) / max(abs(float(item.get('target') or 0.0)), floor)


def _check_constraints(cs: dict, ev_cost, counts, robust=None) -> tuple[list, float, bool]:
    """[(row)], violation, all must constraints met (unknown ones count as met in the search)."""
    rows, viol, ok = [], 0.0, True
    for kind, c in cs.items():
        actual, status, margin = None, 'not_evaluated', None
        if kind == 'budget':
            actual = ev_cost
            if actual is not None:
                passed, margin = RQ.compare(actual, c['op'], c['value'])
                status = 'pass' if passed else 'fail'
                if not passed:
                    viol += abs(margin) / max(c['value'], 1.0)
        elif kind.startswith('max_') or kind == 'min_cameras':
            what = {'max_cameras': 'cameras', 'min_cameras': 'cameras', 'max_lidars': 'lidars',
                    'max_radars': 'radars', 'max_sensors': 'sensors'}[kind]
            actual = float(counts.get(what, 0))
            passed, margin = RQ.compare(actual, c['op'], c['value'])
            status = 'pass' if passed else 'fail'
            if not passed:
                viol += abs(margin)
        elif kind in ('tolerance_mm', 'tolerance_deg') and robust is not None:
            status = 'pass' if robust.get('ok') else 'fail'
            margin = robust.get('worst_rel')
            actual = c['value']
            if not robust.get('ok'):
                viol += abs(robust.get('worst_rel') or 0.1)
        if status == 'fail' and c.get('priority', 'must') == 'must':
            ok = False
        rows.append({'kind': kind, 'text': c.get('text') or RQ.CONSTRAINT_LABEL[kind], 'op': c['op'],
                     'value': c['value'], 'unit': c.get('unit', ''), 'priority': c.get('priority', 'must'),
                     'actual': actual, 'status': status, 'margin': margin})
    return rows, viol, ok


class Designer:
    """One design run: the evaluator (with its caches), the search and the final options."""

    def __init__(self, project, engine=None, settings: DesignerSettings | None = None, progress=None, cancel=None):
        from .analysis import Engine
        from .regions import default_grid
        self.p = project
        self.st = settings or settings_of(project)
        self.eng = engine or Engine()
        self.progress = progress
        self.cancel = cancel
        self.items = [i for i in (RQ.clean_item(x, project) for x in self.st.items()) if i is not None]
        self.cs = constraint_set(self.st)
        self.grid = default_grid(project)
        self.classes = _classes_of(self.items, project)
        self.model, self.model_text = perf_model_status()
        self.det_kind = 'learned' if self.model is not None else 'geometric'
        self.rng = np.random.default_rng(int(self.st.seed))
        self.archive = {}                 # rig key -> Eval (screen)
        self.front = []                   # the archive's non-dominated rigs
        self.evals = 0
        self.notes = []
        self.t0 = time.perf_counter()
        lim = float(self.st.budget_s)
        self.t_end = self.t0 + lim if lim > 0 else math.inf          # no time limit by default
        self.stopped = 'finished'
        lens_pairs = []
        for ln in project.lenses.values():
            mid = ln.module_id
            if mid and mid in project.modules:
                lens_pairs.append((mid, ln.id))
        self.lens_pairs = sorted(set(lens_pairs))
        self.lidars = sorted(m.id for m in project.sensor_models.values() if m.kind == 'lidar')
        self.radars = sorted(m.id for m in project.sensor_models.values() if m.kind == 'radar')
        screen = [i for i in self.items if self._screened(i)]
        self.screen_items = screen
        self.full_items = list(self.items)

    # ---------------------------------------------------------------- helpers
    def stop(self) -> bool:
        return self.cancel is not None and bool(self.cancel())

    def _prog(self, f, text=''):
        if self.progress is not None:
            try:
                self.progress(min(max(f, 0.0), 1.0), text)
            except TypeError:
                self.progress(min(max(f, 0.0), 1.0))

    @staticmethod
    def _screened(it) -> bool:
        m = RQ.METRICS.get(it.get('metric'))
        if m is None:
            return False
        s = RQ.STUDIES[m.study]
        return s.cost == 'cheap' or s.id in SCREEN_STUDIES

    def _settings(self, items):
        extra = []
        have = {RQ.METRICS[i['metric']].study for i in items if i['metric'] in RQ.METRICS}
        if 'perception' not in have:                  # the perception study feeds the detection share
            extra.append({'id': '_pr', 'metric': 'perception.short', 'params': {}, 'op': '<=', 'target': 1e9,
                          'priority': 'should'})
        if 'coverage' not in have:                    # the coverage feeds the ground each camera sees
            extra.append({'id': '_cov', 'metric': 'coverage.blind_m2', 'params': {}, 'op': '<=', 'target': 1e9,
                          'priority': 'should'})
        return RQ.RequirementsSettings.from_dict({'items': list(items) + extra})

    # ---------------------------------------------------------------- evaluation
    def evaluate(self, rig: Rig, level='screen', engine=None) -> Eval | None:
        """The rig scored: requirements (screening ones, or all with the Issues list at level 'full'), cost, power,
        detection and constraints. None when cancelled."""
        v = rig.to_variant('Design candidate')
        q = _with(self.p, v)
        items = self.screen_items if level == 'screen' else self.full_items
        st = self._settings(items)
        eng = engine or self.eng
        res = RQ.run(q, eng, st, variant=v.uid, issues_list=(level == 'full'), cancel=self.cancel)
        if res is None:
            raise _Stop()
        self.evals += 1
        rows, must_fail, rels, should_pass = [], [], [], 0
        by_id = {r.item['id']: r for r in res.items}
        for it in self.full_items:
            r = by_id.get(it['id'])
            m = RQ.METRICS.get(it['metric'])
            if r is None:
                rows.append({'id': it['id'], 'text': it['text'], 'metric': it['metric'], 'priority': it['priority'],
                             'status': 'not_run', 'value': None, 'margin': None, 'rel': None,
                             'unit': m.unit if m else '', 'target': it['target'], 'op': it['op'],
                             'reason': 'evaluated on the front only (a study metric)'})
                continue
            rel = _rel(r.margin, it) if r.margin is not None else None
            rows.append({'id': it['id'], 'text': it['text'], 'metric': it['metric'], 'priority': it['priority'],
                         'status': r.status, 'value': r.value, 'margin': r.margin, 'rel': rel,
                         'unit': m.unit if m else '', 'target': it['target'], 'op': it['op'], 'reason': r.reason})
            if it['priority'] == 'must':
                if r.status == 'pass':
                    rels.append(rel)
                elif r.status == 'fail':
                    rels.append(rel)
                    must_fail.append(it['id'])
                elif r.status == 'not_evaluated':
                    must_fail.append(it['id'])
            elif r.status == 'pass':
                should_pass += 1
        margin = min(rels) if rels else math.inf
        b = bill(self.p, rig, self.st.prices)
        power, pmiss = _power(q, v)
        pr = res.cache.get('perception')
        det, det_std, regions = 0.0, None, {}
        if self.model is not None:
            got = learned_detection(self.model, self.p, v, self.grid, self.classes)
            if got is not None:
                det, det_std, regions = got
        if self.model is None or not regions:
            det, regions = geometric_detection(pr, self.grid, self.classes)
        counts = rig.counts()
        crow, viol, c_ok = _check_constraints(self.cs, b['cost'], counts)
        for it in self.full_items:
            if it['id'] in must_fail:
                rr = next(r for r in rows if r['id'] == it['id'])
                viol += abs(rr['rel']) if rr['rel'] is not None and rr['rel'] < 0 else 1.0
        return Eval(rig, level, rows, margin, must_fail, should_pass, b['cost'], b['missing'], power, pmiss, det,
                    det_std, 'learned' if (self.model is not None and det_std is not None) else 'geometric', counts,
                    crow, viol, not must_fail and c_ok, regions, res, v)

    def screen(self, rig: Rig) -> Eval | None:
        k = rig.key()
        if k in self.archive:
            return self.archive[k]
        ev = self.evaluate(rig, 'screen')
        ev.res = None                     # the study results (images, grids) are not kept for every rig scored
        self.archive[k] = ev
        if not any(dominates(f, ev) for f in self.front):         # the front kept as the archive grows
            self.front = [f for f in self.front if not dominates(ev, f)] + [ev]
        return ev

    # ---------------------------------------------------------------- seeds
    def seeds(self) -> list:
        out = []
        v0 = self.p.variant
        cur = Rig.from_variant(v0, f'the current variant ({v0.name})')
        self.locked = {c.tag for c in v0.cameras if c.locked} | {s.tag for s in v0.sensors if s.locked}
        self.locked_rows = {c['tag']: copy.deepcopy(c) for c in cur.cams if c['tag'] in self.locked}
        self.locked_rows.update({s['tag']: copy.deepcopy(s) for s in cur.sensors if s['tag'] in self.locked})
        out.append(cur)
        for v in self.p.variants:
            if v.uid != v0.uid and any(c.visible for c in v.cameras):
                out.append(self._keep_locked(Rig.from_variant(v, f'variant {v.name}')))
        if self.st.reference_seeds:
            for rig, note in reference_seeds(self.p):
                if rig is not None:
                    out.append(self._keep_locked(rig))
                if note:
                    self.notes.append(note)
        return out

    def _keep_locked(self, rig: Rig) -> Rig:
        """A seed with the user's locked cameras and sensors put back exactly as they are."""
        if not self.locked:
            return rig
        rig.cams = [c for c in rig.cams if c['tag'] not in self.locked]
        rig.sensors = [s for s in rig.sensors if s['tag'] not in self.locked]
        for tag, row in self.locked_rows.items():
            (rig.cams if 'lens_id' in row else rig.sensors).append(copy.deepcopy(row))
        return rig

    # ---------------------------------------------------------------- mutations
    def _free_cams(self, rig):
        return [c for c in rig.cams if c['tag'] not in self.locked]

    def _units(self, rig):
        """Cameras as units: a mirrored pair together, alone otherwise (free ones only)."""
        seen, out = set(), []
        for c in self._free_cams(rig):
            if c['tag'] in seen:
                continue
            p = rig.cam(c['link']) if c.get('link') else None
            if p is not None and p['tag'] not in self.locked:
                out.append([c, p])
                seen |= {c['tag'], p['tag']}
            else:
                out.append([c])
                seen.add(c['tag'])
        return out

    def _limit(self, kind, default):
        c = self.cs.get(kind)
        return int(c['value']) if c is not None else default

    def mutate(self, rig: Rig) -> Rig | None:
        st, rng = self.st, self.rng
        ops = []
        if st.vary_lenses and self.lens_pairs:
            ops += ['lens'] * 4
        if st.vary_count:
            ops += ['add', 'remove']
        if st.vary_lidar and self.lidars:
            ops += ['lidar']
        if st.vary_radar and self.radars:
            ops += ['radar', 'radar_remove']
        ops += ['aim']
        parent = self.archive.get(rig.key())
        if parent is not None and any(r['status'] == 'fail' and r['metric'] == 'perception.range'
                                      for r in parent.items):
            ops += ['reach'] * 4                       # aim the change at a perception range that falls short
        self._parent = parent
        child = rig.copy()
        n_ops = 1 + int(rng.random() < 0.35)
        did = []
        for _ in range(n_ops):
            op = ops[int(rng.integers(len(ops)))]
            text = getattr(self, f'_op_{op}')(child)
            if text:
                did.append(text)
        if not did:
            return None
        child.origin = rig.origin
        child.history = rig.history + did
        return child

    def _fov(self, mid, lid, axis=0) -> float:
        """The lens's horizontal (axis 0) or vertical (1) field on the module, degrees (0: unknown)."""
        k = (mid, lid)
        if not hasattr(self, '_fovs'):
            self._fovs = {}
        if k not in self._fovs:
            from .lens import model_fov
            try:
                f = model_fov(self.p.modules[mid], self.p.lenses[lid], circle=True)
                self._fovs[k] = (float(f[0]), float(f[1]))
            except Exception:
                self._fovs[k] = (0.0, 0.0)
        return self._fovs[k][axis]

    def _horizon(self, c):
        """A camera that has to see far looks at the horizon: its pitch lowered (if needed) until the top of its
        image is 4° above the horizontal."""
        v = self._fov(c['module_id'], c['lens_id'], 1)
        if v > 0:
            c['pitch'] = round(min(c['pitch'], max(0.0, v / 2.0 - 4.0)), 2)

    def _op_reach(self, rig):
        """A perception range short of its target: the camera facing that sector gets a narrower lens, or a narrower
        camera is added beside it (the farther a lens resolves, the narrower it is)."""
        from .perception import SECTORS
        par = getattr(self, '_parent', None)
        short = [r for r in (par.items if par else []) if r['status'] == 'fail' and r['metric'] == 'perception.range']
        if not short:
            return ''
        r = short[int(self.rng.integers(len(short)))]
        it = next((i for i in self.full_items if i['id'] == r['id']), None)
        sec = (it.get('params') or {}).get('sector') if it else ''
        centre = dict(SECTORS).get(sec)
        if centre is None:                            # every sector: the camera of a random direction
            centre = float(self.rng.uniform(-180, 180))
        units = self._units(rig)
        if not units:
            return ''

        def off(c):
            return abs((c['yaw'] - centre + 180.0) % 360.0 - 180.0)
        u = min(units, key=lambda u: min(off(c) for c in u))
        c0 = u[0]
        fov0 = self._fov(c0['module_id'], c0['lens_id'])
        narrower = [p for p in self.lens_pairs if 0 < self._fov(*p) < fov0 - 5.0]
        if not narrower:
            return ''
        same = [p for p in narrower if p[0] == c0['module_id']]
        pool = same if same and self.rng.random() < 0.6 else narrower
        mid, lid = pool[int(self.rng.integers(len(pool)))]
        n_max = min(MAX_CAMERAS, self._limit('max_cameras', MAX_CAMERAS))
        if self.st.vary_count and len(rig.cams) + len(u) <= n_max and self.rng.random() < 0.5:
            n0 = len(rig.cams)
            text = self._add_beside(rig, u, mid, lid)
            for c in rig.cams[n0:]:
                self._horizon(c)
            if len(rig.cams) - n0 == 2:
                rig.cams[-1]['pitch'] = rig.cams[-2]['pitch']
            return text and f'{text}, aimed at the horizon, for {r["text"]}'
        for c in u:
            c['module_id'], c['lens_id'] = mid, lid
            self._horizon(c)
        return f'{"/".join(c["tag"] for c in u)} → {self.p.lenses[lid].name} (aimed at the horizon) for {r["text"]}'

    def _op_lens(self, rig):
        units = self._units(rig)
        if not units:
            return ''
        u = units[int(self.rng.integers(len(units)))]
        mod = u[0]['module_id']
        same = [p for p in self.lens_pairs if p[0] == mod and p[1] != u[0]['lens_id']]
        pool = same if same and self.rng.random() < 0.6 else [p for p in self.lens_pairs if p[1] != u[0]['lens_id']]
        if not pool:
            return ''
        mid, lid = pool[int(self.rng.integers(len(pool)))]
        for c in u:
            c['module_id'], c['lens_id'] = mid, lid
        return f'{"/".join(c["tag"] for c in u)} → {self.p.modules[mid].name} with {self.p.lenses[lid].name}'

    def _op_add(self, rig):
        n_max = min(MAX_CAMERAS, self._limit('max_cameras', MAX_CAMERAS),
                    self._limit('max_sensors', 99) - len(rig.sensors))
        units = self._units(rig) or [[c] for c in rig.cams]
        if not units:
            return ''
        u = units[int(self.rng.integers(len(units)))]
        if len(rig.cams) + len(u) > n_max:
            return ''
        pool = [p for p in self.lens_pairs if p[1] != u[0]['lens_id']]
        if not pool:
            return ''
        mid, lid = pool[int(self.rng.integers(len(pool)))]
        return self._add_beside(rig, u, mid, lid)

    def _add_beside(self, rig, u, mid, lid):
        taken = rig.tags()
        new = []
        for c in u:
            d = copy.deepcopy(c)
            base, k = c['tag'].rstrip('0123456789') or c['tag'], 2
            while f'{base}{k}' in taken:
                k += 1
            d['tag'] = f'{base}{k}'
            taken.add(d['tag'])
            yaw = math.radians(d['yaw'])
            side = 1.0 if d['y'] >= 0 else -1.0           # beside it, away from the centre line for a side camera
            d['x'] = round(d['x'] - math.sin(yaw) * SIDE_OFFSET_M * side, 4)
            d['y'] = round(d['y'] + math.cos(yaw) * SIDE_OFFSET_M * side, 4)
            d['module_id'], d['lens_id'] = mid, lid
            d['port'] = 0
            d['locked'] = False
            new.append(d)
        if len(new) == 2:
            new[0]['link'], new[1]['link'] = new[1]['tag'], new[0]['tag']
            new[1]['y'] = -new[0]['y']
            new[1]['x'] = new[0]['x']
        else:
            new[0]['link'] = ''
        rig.cams += new
        return f'add {"/".join(d["tag"] for d in new)} ({self.p.lenses[lid].name}) beside {"/".join(c["tag"] for c in u)}'

    def _op_remove(self, rig):
        units = self._units(rig)
        lo = max(1, self._limit('min_cameras', 1))
        units = [u for u in units if len(rig.cams) - len(u) >= lo]
        if not units:
            return ''
        u = units[int(self.rng.integers(len(units)))]
        gone = {c['tag'] for c in u}
        rig.cams = [c for c in rig.cams if c['tag'] not in gone]
        for c in rig.cams:
            if c.get('link') in gone:
                c['link'] = ''
        return f'remove {"/".join(sorted(gone))}'

    def _op_aim(self, rig):
        units = self._units(rig)
        if not units:
            return ''
        u = units[int(self.rng.integers(len(units)))]
        dy = float(np.round(self.rng.normal(0, 4.0), 1))
        dp = float(np.round(self.rng.normal(0, 3.0), 1))
        u[0]['yaw'] = round(u[0]['yaw'] + dy, 2)
        u[0]['pitch'] = round(float(np.clip(u[0]['pitch'] + dp, -30.0, 85.0)), 2)
        if len(u) == 2:
            u[1]['yaw'], u[1]['pitch'] = -u[0]['yaw'], u[0]['pitch']
        return f'aim {"/".join(c["tag"] for c in u)} yaw {dy:+.1f}° pitch {dp:+.1f}°'

    def _op_lidar(self, rig):
        free = [s for s in rig.sensors if s['kind'] == 'lidar' and s['tag'] not in self.locked]
        n_l = sum(1 for s in rig.sensors if s['kind'] == 'lidar')
        r = self.rng.random()
        if free and r < 0.3:
            s = free[int(self.rng.integers(len(free)))]
            rig.sensors.remove(s)
            return f'remove lidar {s["tag"]}'
        if free and r < 0.8:
            s = free[int(self.rng.integers(len(free)))]
            pool = [m for m in self.lidars if m != s['model_id']]
            if not pool:
                return ''
            mid = pool[int(self.rng.integers(len(pool)))]
            self._place_sensor(s, mid)
            return f'lidar {s["tag"]} → {self.p.sensor_models[mid].name}'
        if n_l >= min(MAX_LIDARS, self._limit('max_lidars', MAX_LIDARS)) or \
                rig.counts()['sensors'] >= self._limit('max_sensors', 99):
            return ''
        mid = self.lidars[int(self.rng.integers(len(self.lidars)))]
        from .sensors import next_tag
        s = {'tag': next_tag('lidar', rig.tags()), 'name': 'Lidar', 'kind': 'lidar', 'model_id': mid, 'x': 0.0,
             'y': 0.0, 'z': 0.0, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0, 'phase_deg': 0.0, 'port': '', 'visible': True,
             'locked': False}
        self._place_sensor(s, mid, fresh=True, others=rig.sensors)
        rig.sensors.append(s)
        return f'add lidar {s["tag"]} ({self.p.sensor_models[mid].name}) on the roof'

    def _place_sensor(self, s, mid, fresh=False, others=()):
        """A sensor's model changed (its optical centre kept on the same base) or a new one at its default pose."""
        from .sensors import default_pose
        new = self.p.sensor_models[mid]
        if fresh:
            from .project import Sensor
            pose = default_pose(new.kind, new, self.p.vehicle.mesh, getattr(self.p, 'rig', ()),
                                [Sensor.from_dict(o) for o in others])
            s.update(pose)
        else:
            old = self.p.sensor_models.get(s['model_id'])
            if old is not None and new.kind == 'lidar':
                s['z'] = round(s['z'] - old.optical_z_mm / 1000.0 + new.optical_z_mm / 1000.0, 4)
        s['model_id'] = mid

    def _op_radar(self, rig):
        free = [s for s in rig.sensors if s['kind'] == 'radar' and s['tag'] not in self.locked]
        if not free:
            n_r = 0
            if n_r >= self._limit('max_radars', MAX_RADARS) or rig.counts()['sensors'] >= self._limit('max_sensors', 99):
                return ''
            mid = self.radars[int(self.rng.integers(len(self.radars)))]
            from .sensors import next_tag
            s = {'tag': next_tag('radar', rig.tags()), 'name': 'Radar', 'kind': 'radar', 'model_id': mid, 'x': 0.0,
                 'y': 0.0, 'z': 0.0, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0, 'phase_deg': 0.0, 'port': '',
                 'visible': True, 'locked': False}
            self._place_sensor(s, mid, fresh=True, others=rig.sensors)
            rig.sensors.append(s)
            return f'add radar {s["tag"]} ({self.p.sensor_models[mid].name}) in the front bumper'
        model = free[int(self.rng.integers(len(free)))]['model_id']
        group = [s for s in free if s['model_id'] == model]          # same model: changed together
        pool = [m for m in self.radars if m != model]
        if not pool:
            return ''
        mid = pool[int(self.rng.integers(len(pool)))]
        for s in group:
            s['model_id'] = mid
        return f'radars {"/".join(s["tag"] for s in group)} → {self.p.sensor_models[mid].name}'

    def _op_radar_remove(self, rig):
        free = [s for s in rig.sensors if s['kind'] == 'radar' and s['tag'] not in self.locked]
        if not free:
            return ''
        s = free[int(self.rng.integers(len(free)))]
        mirror = next((o for o in free if o is not s and abs(o['y'] + s['y']) < 0.05 and abs(o['x'] - s['x']) < 0.05
                       and abs(s['y']) > 0.05), None)
        gone = [s] + ([mirror] if mirror is not None and self.rng.random() < 0.7 else [])
        rig.sensors = [o for o in rig.sensors if all(o is not g for g in gone)]
        return f'remove radar{"s" * (len(gone) > 1)} {"/".join(g["tag"] for g in gone)}'

    # ---------------------------------------------------------------- the inner placement search
    def place(self, ev: Eval, generations=6, population=10) -> Rig | None:
        """The rig's cameras aimed by core.optimiser (yaw ±8°, pitch ±5°, no slide) for the least blind ground under
        the stated tolerance; None when nothing changed or the optimiser cannot run."""
        from .optimiser import FreeVar, Optimiser, OptSettings
        v = ev.rig.to_variant('Design candidate')
        q = _with(self.p, v)
        tags = {c.uid: c.tag for c in v.cameras}
        fvars, seen = [], set()
        for c in v.cameras:
            if c.uid in seen:
                continue
            if c.tag in self.locked:
                fvars.append(FreeVar(c.uid, c.tag, axis='', move=0.0, yaw=(c.yaw, c.yaw), pitch=(c.pitch, c.pitch),
                                     keep=True))
                seen.add(c.uid)
                continue
            fvars.append(FreeVar(c.uid, c.tag, axis='', move=0.0, yaw=(c.yaw - 8.0, c.yaw + 8.0),
                                 pitch=(max(-30.0, c.pitch - 5.0), min(85.0, c.pitch + 5.0))))
            seen.add(c.uid)
            if c.link_uid and c.link_uid not in seen:
                p_ = v.cam(c.link_uid)
                if p_ is not None and p_.tag not in self.locked:
                    fvars.append(FreeVar(p_.uid, p_.tag, linked_to=c.uid))
                    seen.add(p_.uid)
        tol = self.cs.get('tolerance_mm')
        tdeg = self.cs.get('tolerance_deg')
        n = len(v.cameras)
        ost = OptSettings(objective='blind', under_tolerance=bool(tol or tdeg), tol_deg=float(tdeg['value']) if tdeg
                          else 0.0, tol_mm=float(tol['value']) if tol else 0.0, tol_samples=4 if (tol or tdeg) else 0,
                          zones='any', height=(0.0, 10.0), cable_max=1e3, count=(n, n), generations=generations,
                          population=population, cell=max(0.10, self.p.analysis.cell_m), seed=int(self.st.seed),
                          vars=fvars, tol_seed=int(self.st.seed) + 1)
        try:
            opt = Optimiser(self.eng, q, ost, v)
            if opt.grid is None:
                return None
            opt.run(cancel=lambda: self.stop() or time.perf_counter() > self.t_end)
            best = min(opt.front, key=opt.objectives) if opt.front else None
        except Exception as e:                       # a body the optimiser cannot slide on: keep the rig as it is
            self.notes.append(f'placement search skipped on one rig: {type(e).__name__}: {e}')
            return None
        if best is None:
            return None
        new = ev.rig.copy()
        moved = []
        for uid, g in best.genes.items():
            c = new.cam(tags[uid])
            if c is None or g[3] <= 0.5:
                continue
            if abs(g[1] - c['yaw']) > 0.05 or abs(g[2] - c['pitch']) > 0.05:
                moved.append(c['tag'])
                c['yaw'], c['pitch'] = round(float(g[1]), 2), round(float(g[2]), 2)
        if not moved:
            return None
        new.history = ev.rig.history + [f'placement: {", ".join(moved)} re-aimed for the least blind ground']
        return new

    # ---------------------------------------------------------------- the final checks
    def robust(self, ev: Eval) -> dict | None:
        """Every screening must requirement in ROBUST_SAMPLES random builds, each camera off by up to the stated
        tolerance (uniform, every axis), each keeping its planned position's occlusion map."""
        tol = self.cs.get('tolerance_mm')
        tdeg = self.cs.get('tolerance_deg')
        if tol is None and tdeg is None:
            return None
        mm = float(tol['value']) if tol else 0.0
        deg = float(tdeg['value']) if tdeg else 0.0
        rng = np.random.default_rng(int(self.st.seed) + 11)
        eng = self.eng.share()
        nominal = [np.array([c['x'], c['y'], c['z']], float) for c in ev.rig.cams]
        limit = max(0.05, 3 * mm / 1000.0)
        base_cube = type(eng).cube

        def cube(pos, vg=None):                       # the planned position's occlusion map for a small offset
            pos = np.asarray(pos, float)
            best = min(nominal, key=lambda q: float(np.linalg.norm(q - pos)), default=None)
            if best is not None and float(np.linalg.norm(best - pos)) <= limit:
                pos = best
            return base_cube(eng, pos, vg)
        eng.cube = cube
        musts = [i for i in self.screen_items if i['priority'] == 'must']
        worst = {i['id']: None for i in musts}
        fails = []
        for k in range(ROBUST_SAMPLES):
            if self.stop():
                raise _Stop()
            rig = ev.rig.copy()
            for c in rig.cams:
                d = rng.uniform(-1, 1, 6)
                c['x'] = c['x'] + d[0] * mm / 1000.0
                c['y'] = c['y'] + d[1] * mm / 1000.0
                c['z'] = c['z'] + d[2] * mm / 1000.0
                c['yaw'] += d[3] * deg
                c['pitch'] += d[4] * deg
                c['roll'] += d[5] * deg
            v = rig.to_variant('Robust check')
            q = _with(self.p, v)
            res = RQ.run(q, eng, RQ.RequirementsSettings.from_dict({'items': musts}), variant=v.uid,
                         issues_list=False, cancel=self.cancel)
            if res is None:
                raise _Stop()
            for r in res.items:
                rel = _rel(r.margin, r.item) if r.margin is not None else None
                w = worst.get(r.item['id'])
                if rel is not None and (w is None or rel < w):
                    worst[r.item['id']] = rel
                if r.status != 'pass':
                    fails.append((k, r.item['id']))
        rels = [w for w in worst.values() if w is not None]
        return {'samples': ROBUST_SAMPLES, 'tolerance_mm': mm, 'tolerance_deg': deg, 'ok': not fails,
                'failing': sorted({i for _, i in fails}), 'failed_builds': len({k for k, _ in fails}),
                'worst_rel': min(rels) if rels else None,
                'worst': {k: (round(v, 4) if v is not None else None) for k, v in worst.items()},
                'method': f'{ROBUST_SAMPLES} random builds, every camera off by up to ±{mm:g} mm on each axis'
                          + (f' and ±{deg:g}° about each axis' if deg else '') + ', uniform; the must requirements '
                          'of the screening set re-checked in each'}

    def contributions(self, ev: Eval) -> list:
        """Why each sensor: the rig scored again without it (screening metrics)."""
        out = []
        cov = ev.res.cache.get('coverage') if ev.res is not None else None
        per = getattr(cov, 'per_cam', {}) if cov is not None else {}
        uid_tag = {c.uid: c.tag for c in ev.variant.cameras} if ev.variant is not None else {}
        unique = {uid_tag.get(u): d['unique_m2'] for u, d in per.items()}
        base = {r['id']: r for r in ev.items}
        units = [('camera', c['tag']) for c in ev.rig.cams] + [(s['kind'], s['tag']) for s in ev.rig.sensors]
        for kind, tag in units:
            if self.stop():
                raise _Stop()
            rig = ev.rig.copy()
            rig.cams = [c for c in rig.cams if c['tag'] != tag]
            for c in rig.cams:
                if c.get('link') == tag:
                    c['link'] = ''
            rig.sensors = [s for s in rig.sensors if s['tag'] != tag]
            e2 = self.evaluate(rig, 'screen')
            fails, drops = [], []
            for r in e2.items:
                b = base.get(r['id'])
                if b is None or b['status'] == 'not_run' or r['status'] == 'not_run':
                    continue
                if b['status'] == 'pass' and r['status'] != 'pass':
                    fails.append(r['id'])
                if b['rel'] is not None and r['rel'] is not None and r['rel'] < b['rel'] - 1e-9:
                    drops.append((r['id'], b['rel'] - r['rel'], b['value'], r['value']))
                elif b['rel'] is not None and r['rel'] is None:
                    drops.append((r['id'], math.inf, b['value'], None))
            drops.sort(key=lambda d: -d[1])
            serves = fails[0] if fails else (drops[0][0] if drops else '')
            row = next((c for c in ev.rig.cams if c['tag'] == tag), None) or \
                next((s for s in ev.rig.sensors if s['tag'] == tag), None)
            what = self._what(kind, row)
            text_of = {i['id']: i['text'] for i in self.full_items}
            if fails:
                why = (f'without it {", ".join(text_of.get(f, f) for f in fails[:3])} '
                       f'{"fails" if len(fails) == 1 else "fail"}')
            elif drops and math.isfinite(drops[0][1]):
                d = drops[0]
                why = f'it carries {text_of.get(d[0], d[0])}: {_fmt(d[2])} → {_fmt(d[3])} without it'
            elif kind == 'camera' and unique.get(tag, 0.0) > 0.0005:
                why = f'only it sees {unique[tag]:.3f} m² of the ground around the car'
            else:
                why = 'no requirement needs it: a candidate for removal'
            out.append({'tag': tag, 'kind': kind, 'what': what, 'pose': _pose_of(row), 'serves': serves,
                        'serves_text': text_of.get(serves, ''), 'fails_without': fails,
                        'margins': [{'id': d[0], 'with': d[2], 'without': d[3]} for d in drops[:4]],
                        'unique_ground_m2': round(float(unique.get(tag, 0.0)), 4) if kind == 'camera' else None,
                        'blind_without_m2': (round(float(e2.res.cache['coverage'].metrics['blind_m2']), 4)
                                             if e2.res is not None and e2.res.cache.get('coverage') is not None
                                             else None),
                        'detection_lost': round(ev.detection - e2.detection, 4),
                        'power_w': round(ev.power - e2.power, 2),
                        'cost': (round(ev.cost - e2.cost, 2) if ev.cost is not None and e2.cost is not None else None),
                        'why': why})
        return out

    def _what(self, kind, row):
        if row is None:
            return ''
        if kind == 'camera':
            m = self.p.modules.get(row['module_id'])
            ln = self.p.lenses.get(row['lens_id'])
            return f'{m.name if m else row["module_id"]} · {ln.name if ln else row["lens_id"]}' + (
                f' ({ln.label})' if ln is not None and ln.label else '')
        m = self.p.sensor_models.get(row['model_id'])
        return f'{m.vendor} {m.name}'.strip() if m else row['model_id']

    def critique(self, ev: Eval, contrib) -> list:
        """The option's own weak points, computed (see the module doc), worst first."""
        out = []

        def add(sev, kind, msg, tags=()):
            out.append({'severity': sev, 'kind': kind, 'message': msg, 'tags': list(tags)})
        text_of = {i['id']: i['text'] for i in self.full_items}
        for r in ev.items:
            if r['priority'] == 'must' and r['status'] == 'fail':
                add('critical', 'requirement', f'Fails "{r["text"]}": {_fmt(r["value"])} {r["unit"]} against '
                                               f'{RQ.OP_TEXT.get(r["op"], "")} {_fmt(r["target"])} {r["unit"]}.')
            elif r['priority'] == 'must' and r['status'] == 'not_evaluated':
                add('critical', 'requirement', f'"{r["text"]}" cannot be evaluated on this rig: {r["reason"]}')
            elif r['priority'] == 'should' and r['status'] == 'fail':
                add('warning', 'requirement', f'Misses the should requirement "{r["text"]}" ({_fmt(r["value"])} '
                                              f'{r["unit"]}).')
        for c in ev.constraints:
            if c['kind'].startswith('tolerance'):
                continue                              # the robust check's own finding below
            if c['status'] == 'fail':
                add('critical' if c['priority'] == 'must' else 'warning', 'constraint',
                    f'{c["text"]}: {_fmt(c["actual"])} against {RQ.OP_TEXT.get(c["op"], "")} {_fmt(c["value"])}.')
            elif c['status'] == 'not_evaluated' and c['kind'] == 'budget':
                add('serious', 'cost', 'The budget cannot be checked: '
                                       f'{len(ev.cost_missing)} item{"s" * (len(ev.cost_missing) != 1)} '
                                       f'without a price ({", ".join(ev.cost_missing[:4])}).')
        iss = getattr(ev.res, 'issues', None) or []
        blind_issue = any('blind' in i.message for i in iss if i.severity in ('critical', 'serious'))
        for i in [i for i in iss if i.severity in ('critical', 'serious')][:5]:
            add(i.severity, 'issues', f'Issues list: {i.message}', [t for t in (i.objects or '').split(', ') if t])
        cov = ev.res.cache.get('coverage') if ev.res is not None else None
        if cov is not None and cov.metrics['blind_m2'] > 0.0005 and not blind_issue:
            blind = cov.grid.roi & (cov.count == 0)
            jj, ii = np.nonzero(blind)
            where = ''
            if len(ii):
                from .analysis import describe_location
                where = ' mostly ' + describe_location(float(cov.grid.xs[ii].mean()), float(cov.grid.ys[jj].mean()),
                                                       _with(self.p, ev.variant))
            add('serious' if cov.metrics['blind_m2'] > 0.05 else 'warning', 'blind',
                f'{cov.metrics["blind_m2"]:.3f} m² of the ground near the car (the ROI) is seen by no camera,{where}.')
        out += self._seam_findings(ev, cov)
        out += self._observability(ev)
        out += self._tolerance_single(ev)
        out += self._in_view(ev)
        rob = ev.robust
        if rob is not None and not rob['ok']:
            add('critical' if any(c['kind'].startswith('tolerance') and c['priority'] == 'must'
                                  for c in ev.constraints) else 'serious', 'tolerance',
                f'Does not survive the stated mounting tolerance: {rob["failed_builds"]} of {rob["samples"]} random '
                f'builds fail {", ".join(text_of.get(i, i) for i in rob["failing"][:3])}.')
        b = self.cs.get('budget')
        if b is not None and ev.cost is not None and b['value'] > 0:
            head = (b['value'] - ev.cost) / b['value']
            if 0 <= head < 0.1:
                add('warning', 'cost', f'Cost headroom only {head * 100:.0f} % of the budget '
                                       f'({_money(ev.cost, self.st.currency)} of {_money(b["value"], self.st.currency)}).')
        pw = [it for it in self.full_items if it['metric'] == 'power.steady_w']
        for it in pw:
            head = (float(it['target']) - ev.power) / max(float(it['target']), 1.0)
            if 0 <= head < 0.1:
                add('warning', 'power', f'Power headroom only {head * 100:.0f} % ({ev.power:.0f} W of '
                                        f'{float(it["target"]):g} W).')
        if ev.power_missing:
            add('warning', 'power', f'{", ".join(ev.power_missing[:6])}: no power figure in the library, counted as '
                                    '0 W (enter the datasheet figure, or a typical camera power in the power budget).')
        idle = [c for c in contrib if not c['fails_without'] and not c['margins'] and
                not (c.get('unique_ground_m2') or 0) > 0.0005]
        if idle:
            add('note', 'redundant', f'{", ".join(c["tag"] for c in idle)}: no requirement needs '
                                     f'{"it" if len(idle) == 1 else "them"} (removing '
                                     f'{"it" if len(idle) == 1 else "each"} changes no screened requirement).',
                [c['tag'] for c in idle])
        rank = {s: i for i, s in enumerate(SEVERITIES)}
        out.sort(key=lambda f: rank.get(f['severity'], 9))
        return out

    def _seam_findings(self, ev, cov) -> list:
        from .critical import settings_of as crit_settings
        tg = [t for t in crit_settings(self.p).targets if t.get('enabled', True)]
        if cov is None or not tg or not getattr(cov, 'seams', None):
            return []
        out = []
        tag_of = {c.uid: c.tag for c in ev.variant.cameras}
        cx, cy = cov.centre
        for t in tg:
            az = math.degrees(math.atan2(t['y'] - cy, t['x'] - cx))
            dist = math.hypot(t['x'] - cx, t['y'] - cy)
            half = math.degrees(math.atan2(max(t.get('width', 0.5), 0.3) / 2.0, max(dist, 0.5)))
            for s in cov.seams:
                d = abs((s.angle - az + 180.0) % 360.0 - 180.0)
                if d <= half + 2.0:
                    out.append({'severity': 'warning', 'kind': 'seam',
                                'message': f'The stitching seam between {tag_of.get(s.a, s.a)} and '
                                           f'{tag_of.get(s.b, s.b)} runs across the critical target {t["name"]} '
                                           f'({d:.1f}° from its centre).',
                                'tags': [tag_of.get(s.a, s.a), tag_of.get(s.b, s.b)]})
        return out

    def _observability(self, ev) -> list:
        if not getattr(self.p, 'targets', None):
            return []
        try:
            from . import observability as OB
            res = OB.run(_with(self.p, ev.variant), self.eng, cancel=self.cancel)
        except Exception as e:
            return [{'severity': 'note', 'kind': 'calibration', 'tags': [],
                     'message': f'Calibration observability not checked: {type(e).__name__}: {e}'}]
        if res is None:
            return []
        out = []
        doc = OB.summary(res)
        for c in doc.get('cameras') or []:
            if c.get('verdict') in ('warning', 'critical', 'serious') and c.get('ratio') is not None:
                out.append({'severity': 'warning' if c['verdict'] == 'warning' else 'serious', 'kind': 'calibration',
                            'message': f'{c["tag"]}: the planned calibration boards pin its {c.get("worst_axis")} to '
                                       f'{c["ratio"]:.1f}× the calibration requirement ({c.get("weakest", "")}).',
                            'tags': [c['tag']]})
        for c in doc.get('lidar_pairs') or []:
            if c.get('verdict') in ('warning', 'critical', 'serious') and c.get('ratio') is not None:
                out.append({'severity': 'warning', 'kind': 'calibration',
                            'message': f'{c["pair"]}: the lidar–camera boards pin the lidar to {c["ratio"]:.1f}× its '
                                       'tolerance.', 'tags': c['pair'].split('–')})
        missing = [c['tag'] for c in doc.get('cameras') or [] if c.get('ratio') is None]
        if missing:
            out.append({'severity': 'serious', 'kind': 'calibration', 'tags': missing,
                        'message': f'{", ".join(missing)}: no planned calibration board in view: the camera cannot be '
                                   'calibrated with the current board plan.'})
        return out[:6]

    def _tolerance_single(self, ev) -> list:
        from . import tolerance as TL
        tol = self.cs.get('tolerance_mm')
        tdeg = self.cs.get('tolerance_deg')
        deg = float(tdeg['value']) if tdeg else (0.0 if tol else float(self.p.analysis.tol_deg))
        m = (float(tol['value']) / 1000.0) if tol else (0.0 if tdeg else float(self.p.analysis.tol_mm) / 1000.0)
        q = _with(self.p, ev.variant)
        try:
            worst, uid, _pert = TL.worst_single_camera(self.eng, q, deg, m, cell=max(0.05, self.p.analysis.cell_m))
        except Exception as e:
            return [{'severity': 'note', 'kind': 'tolerance', 'tags': [],
                     'message': f'Single-camera tolerance not checked: {type(e).__name__}: {e}'}]
        cov = ev.res.cache.get('coverage') if ev.res is not None else None
        base = float(cov.metrics['blind_m2']) if cov is not None else 0.0
        extra = float(worst) - base
        tag = next((c.tag for c in ev.variant.cameras if c.uid == uid), uid)
        src = 'the stated tolerance' if (tol or tdeg) else 'the analysis tolerance (none stated in the brief)'
        if extra > 0.01:
            return [{'severity': 'warning', 'kind': 'tolerance', 'tags': [tag],
                     'message': f'{tag} off by {src} (±{deg:g}°, ±{m * 1000:g} mm) leaves {extra:.3f} m² more blind '
                                'ground: the most tolerance-sensitive camera.'}]
        return []

    def _in_view(self, ev) -> list:
        """Lidars and radars inside a camera's image: the share of the image their housing box covers."""
        from .analysis import camera_frames
        from .lens import resolve
        out = []
        q = _with(self.p, ev.variant)
        for c in ev.variant.cameras:
            intr = resolve(q.module_for(c), q.lens_for(c))
            _, W2O = camera_frames(c)
            for s in ev.variant.sensors:
                if s.kind not in ('lidar', 'radar'):
                    continue
                try:
                    P = _box_corners(s, q.sensor_model_for(s))
                except Exception:
                    continue
                d = (P - c.pos()) @ W2O.T
                if np.linalg.norm(P - c.pos(), axis=1).min() < 0.02:
                    continue
                u, v, _, ok = intr.project_dirs(d)
                if ok.sum() < 2:
                    continue
                uu = np.clip(u, 0, intr.width)
                vv = np.clip(v, 0, intr.height)
                share = float((uu.max() - uu.min()) * (vv.max() - vv.min()) / (intr.width * intr.height))
                if share >= 0.005:
                    out.append({'severity': 'warning' if share >= 0.02 else 'note', 'kind': 'in_view',
                                'message': f'{s.tag} ({s.kind}) is inside {c.tag}\'s image: about {share * 100:.1f} % '
                                           'of it (its housing box).', 'tags': [c.tag, s.tag]})
        return out

    # ---------------------------------------------------------------- the search
    def run(self):
        st = self.st
        if not self.items:
            return None, 'No accepted requirements: add or accept some first.'
        self._prog(0.0, 'seeds')
        seeds = self.seeds()
        cur = None
        try:
            for k, rig in enumerate(seeds):
                ev = self.screen(rig)
                if k == 0:
                    cur = ev
                if self.stop():
                    raise _Stop()
                self._prog(0.05 * (k + 1) / len(seeds), f'seeds: {rig.origin}')
            n_target = int(st.evaluations)
            n_search = max(len(seeds) + 1, int(n_target * 0.75))
            tries = 0
            gen = 0
            while self.evals < n_search and tries < n_search * 6:
                if self.stop():
                    raise _Stop()
                if st.budget_s > 0 and time.perf_counter() > self.t0 + 0.6 * st.budget_s:
                    self.stopped = 'time'
                    break
                tries += 1
                front = sorted(self.front, key=lambda e: e.rig.key())
                parent = front[int(self.rng.integers(len(front)))] if front else cur
                child = self.mutate(parent.rig)
                if child is None or child.key() in self.archive:
                    continue
                if not child.cams and not child.sensors:
                    continue
                self.screen(child)
                gen += 1
                self._prog(0.05 + 0.55 * self.evals / max(n_search, 1),
                           f'searching: {self.evals} rigs scored, front {len(front)}')
            if st.placement:
                front = _spread(list(self.front), max(2, st.options), self._axes_key())
                for k, ev in enumerate(front):
                    if self.stop():
                        raise _Stop()
                    if self.evals >= n_target or (st.budget_s > 0 and
                                                  time.perf_counter() > self.t0 + 0.75 * st.budget_s):
                        break
                    rig = self.place(ev)
                    if rig is not None and rig.key() not in self.archive:
                        self.screen(rig)
                    self._prog(0.6 + 0.1 * (k + 1) / max(len(front), 1), 'placement search')
            return self._final(cur), ''
        except _Stop:
            self.stopped = 'cancelled'
            return None, 'Cancelled.'

    def _axes_key(self):
        return lambda e: e.objectives()

    def _final(self, cur):
        st = self.st
        pool = [e for e in self.archive.values() if e.rig.key() != cur.rig.key()]
        fronts = sort_fronts(pool)
        cand = []
        for f in fronts:
            cand += f
            if len(cand) >= st.options + 3:
                break
        cand = _spread(cand, st.options + 3, self._axes_key())
        self._prog(0.72, 'evaluating the front in full')
        cur_full = self.evaluate(cur.rig, 'full')
        cur_full.robust = self.robust(cur_full)
        if cur_full.robust is not None:
            cur_full.constraints, cur_full.violation, ok = self._recheck(cur_full)
            cur_full.feasible = ok and not cur_full.must_fail
        full = []
        for k, ev in enumerate(cand):
            if self.stop():
                raise _Stop()
            if full and time.perf_counter() > self.t_end:
                self.stopped = 'time'
                self.notes.append(f'{len(cand) - k} front rig{"s" * (len(cand) - k != 1)} not evaluated in full: '
                                  'the time limit ran out (the options now depend on the machine\'s speed; set the '
                                  'time limit to 0 for none)')
                break
            fe = self.evaluate(ev.rig, 'full')
            fe.robust = self.robust(fe) if fe.feasible else None      # an infeasible rig needs no robust check
            if fe.robust is not None:
                fe.constraints, fe.violation, ok = self._recheck(fe)
                fe.feasible = ok and not fe.must_fail
            full.append(fe)
            self._prog(0.72 + 0.12 * (k + 1) / max(len(cand), 1), f'full evaluation {k + 1} of {len(cand)}')
        feasible = [e for e in full if e.feasible]
        front = pareto(feasible) if feasible else pareto(full)
        rejected = []
        for e in full:
            if e in front:
                continue
            if not e.feasible and feasible:
                why = 'misses ' + ', '.join(e.must_fail[:3] + [c['kind'] for c in e.constraints
                                                               if c['status'] == 'fail'][:2])
            else:
                dom = next((f for f in front if dominates(f, e)), None)
                why = 'dominated' + (' by another option' if dom is not None else '')
            rejected.append({'rig': e.rig.key(), 'history': e.rig.history[-4:], 'origin': e.rig.origin, 'why': why})
        uniq, seen = [], set()
        for e in sorted(front, key=lambda e: e.rig.key()):     # two rigs with the same figures: one option
            k = tuple(None if x is None else round(x, d) for x, d in zip(e.objectives(), (0, 0, 2, 2)))   # as shown
            if k not in seen:
                seen.add(k)
                uniq.append(e)
            else:
                rejected.append({'rig': e.rig.key(), 'history': e.rig.history[-4:], 'origin': e.rig.origin,
                                 'why': 'the same cost, power, margin and detection as an option shown'})
        front_all, front = front, uniq
        chosen = _spread(front, st.options, self._axes_key())
        if len(chosen) < 3 and len(full) > len(chosen):
            # fewer than three non-dominated rigs: fill with the best of the others, each saying it is dominated
            rest = sorted((e for e in full if e not in chosen and e in uniq or e not in front_all),
                          key=lambda e: (not e.feasible, e.violation, -_m(e.margin)))
            rest = [e for e in rest if e not in chosen]
            for e in rest[:3 - len(chosen)]:
                chosen.append(e)
                rejected = [r for r in rejected if r['rig'] != e.rig.key()]
        chosen.sort(key=lambda e: (not e.feasible, e.cost if e.cost is not None else math.inf, -_m(e.margin)))
        options = []
        for i, e in enumerate(chosen):
            if self.stop():
                raise _Stop()
            label = chr(ord('A') + i)
            self._prog(0.84 + 0.15 * i / max(len(chosen), 1), f'option {label}: why each sensor, self-critique')
            contrib = self.contributions(e)
            crit = self.critique(e, contrib)
            options.append(_option(label, e, contrib, crit, cur_full, self, e in front))
        self._prog(1.0, '')
        cur_doc = _option('current', cur_full, self.contributions(cur_full), [], cur_full, self, False)
        cur_doc['critique'] = self.critique(cur_full, cur_doc['sensors'])
        return DesignerResult(
            settings=copy.deepcopy(st), options=options, current=cur_doc, rejected=rejected,
            search={'rigs_scored': self.evals, 'archive': len(self.archive), 'seeds': [r for r in
                                                                                       {e.rig.origin for e in
                                                                                        self.archive.values()}],
                    'seconds': round(time.perf_counter() - self.t0, 1), 'stopped': self.stopped,
                    'seed': int(st.seed), 'evaluations': int(st.evaluations), 'budget_s': float(st.budget_s),
                    'screening_metrics': sorted({i['metric'] for i in self.screen_items}),
                    'front_only_metrics': sorted({i['metric'] for i in self.full_items} -
                                                 {i['metric'] for i in self.screen_items})},
            detection_kind=self.det_kind, detection_text=self.model_text, notes=list(dict.fromkeys(self.notes)),
            project=getattr(self.p, 'name', ''), variant=self.p.variant.name, variant_uid=self.p.variant.uid,
            created=_dt.datetime.now().astimezone().replace(microsecond=0).isoformat(),
            inputs_key=inputs_key(self.p, st), cost_complete=all(o['cost'] is not None for o in options),
            grid=self.grid.to_dict())

    def _recheck(self, ev):
        crow, viol, ok = _check_constraints(self.cs, ev.cost, ev.counts, ev.robust)
        for it in self.full_items:
            if it['id'] in ev.must_fail:
                rr = next(r for r in ev.items if r['id'] == it['id'])
                viol += abs(rr['rel']) if rr['rel'] is not None and rr['rel'] < 0 else 1.0
        return crow, viol, ok


def _m(x):
    return x if math.isfinite(x) else 10.0


def _fmt(v, dp=None):
    if v is None:
        return '—'
    if isinstance(v, str):
        return v
    if not math.isfinite(v):
        return '∞'
    a = abs(v)
    dp = dp if dp is not None else (0 if a >= 100 else 1 if a >= 10 else 2 if a >= 1 else 3 if a >= 0.01 else 4)
    return f'{v:,.{dp}f}'


def _money(v, cur='USD'):
    if v is None:
        return 'incomplete'
    sym = {'USD': '$', 'EUR': '€', 'GBP': '£'}.get(cur, '')
    return f'{sym}{v:,.0f}' + ('' if sym else f' {cur}')


def _pose_of(row):
    if row is None:
        return {}
    return {k: round(float(row[k]), 3) for k in ('x', 'y', 'z', 'yaw', 'pitch', 'roll') if k in row}


def _box_corners(s, model) -> np.ndarray:
    """The 8 corners of the sensor's housing box in the vehicle frame (core.sensors.body_box)."""
    from .sensors import body_box
    c, half, R, _round = body_box(s, model)
    q = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float) * half
    return q @ R.T + c


# =========================================================================== Pareto
def dominates(a: Eval, b: Eval) -> bool:
    """Constrained dominance: a feasible rig beats an infeasible one; two infeasible ones by their violation; two
    feasible ones by Pareto dominance on cost (only when both are priced), power, margin and detection."""
    if a.feasible != b.feasible:
        return a.feasible
    if not a.feasible:
        return a.violation < b.violation - 1e-9
    oa, ob = a.objectives(), b.objectives()
    if (oa[0] is None) != (ob[0] is None):
        return False
    pairs = [(x, y) for x, y in zip(oa, ob) if x is not None and y is not None]
    return all(x <= y + 1e-9 for x, y in pairs) and any(x < y - 1e-9 for x, y in pairs)


def pareto(evals: list) -> list:
    return [e for e in evals if not any(dominates(o, e) for o in evals if o is not e)]


def sort_fronts(evals: list) -> list:
    rest, out = list(evals), []
    while rest:
        f = pareto(rest)
        if not f:
            break
        out.append(f)
        rest = [e for e in rest if e not in f]
    return out


def _spread(evals: list, k: int, key) -> list:
    """At most k of the rigs spread along the front: each objective's best first, then by crowding distance
    (deterministic: ties by the rig key)."""
    evals = sorted(evals, key=lambda e: e.rig.key())
    if len(evals) <= k:
        return evals
    objs = np.array([[np.nan if x is None else x for x in key(e)] for e in evals], float)
    chosen = []
    for j in range(objs.shape[1]):
        col = objs[:, j]
        if np.isfinite(col).any():
            i = int(np.nanargmin(np.where(np.isfinite(col), col, np.nan)))
            if i not in chosen:
                chosen.append(i)
        if len(chosen) >= k:
            break
    crowd = np.zeros(len(evals))
    for j in range(objs.shape[1]):
        col = objs[:, j]
        ok = np.isfinite(col)
        if ok.sum() < 2:
            continue
        idx = np.argsort(np.where(ok, col, np.inf), kind='stable')
        span = np.nanmax(col[ok]) - np.nanmin(col[ok]) or 1.0
        for a, i in enumerate(idx[:int(ok.sum())]):
            if a == 0 or a == ok.sum() - 1:
                crowd[i] += 1e6
            else:
                crowd[i] += (col[idx[a + 1]] - col[idx[a - 1]]) / span
    for i in np.argsort(-crowd, kind='stable'):
        if len(chosen) >= k:
            break
        if int(i) not in chosen:
            chosen.append(int(i))
    return [evals[i] for i in chosen[:k]]


# =========================================================================== reference rigs on this vehicle
def reference_seeds(project) -> list:
    """[(Rig, note)]: every reference rig with a published layout, scaled from its vehicle's footprint to this one's
    (x from the rear of the body by the length ratio, y by the width ratio, z by the height ratio) and snapped onto
    the body (cameras and radars along their viewing direction, lidars onto the roof); cameras the closest library
    module and lens (reference_rigs.closest_camera)."""
    from . import reference_rigs as RR
    from .samples import snap_point
    from .vehicles import by_id as vehicles, body_spec
    from .sensors import surface_top
    mesh = project.vehicle.mesh
    if mesh is None or not mesh.n_tris:
        return []
    lo, hi = mesh.bbox
    L, W, H, xr = float(hi[0] - lo[0]), float(hi[1] - lo[1]), float(hi[2]), float(lo[0])
    out = []
    for rig in RR.library():
        if rig.layout == 'none' or not rig.cameras:
            continue
        veh = vehicles().get(rig.library_vehicle_id or '')
        if veh is None:
            continue
        try:
            sp = body_spec(veh)
            Lr, Wr, Hr, xrr = float(sp.length), float(sp.width), float(sp.height), -float(sp.rear_overhang)
        except Exception:
            continue
        if any(isinstance(s.position_m, str) for s in rig.sensors):
            continue                                   # named (estimated) mounts: only on their own template body

        def scale(p):
            u = (p[0] - xrr) / Lr
            return (xr + u * L, p[1] / (Wr / 2.0) * (W / 2.0), p[2] / Hr * H)
        cams, sens = [], []
        try:
            for i, c in enumerate(rig.cameras):
                mid, lid, _h, _why = RR.closest_camera(project, c)
                x, y, z = scale(c.position_m)
                yaw = float(c.yaw or 0.0)
                d = np.array([math.cos(math.radians(yaw)), math.sin(math.radians(yaw)), 0.0])
                pt, _n = snap_point(mesh, (x, y, z), d, 0.012)
                cams.append({'tag': f'C{i + 1}', 'name': c.name, 'location': c.role, 'slot': i % MAX_CAMERAS,
                             'x': round(float(pt[0]), 4), 'y': round(float(pt[1]), 4), 'z': round(float(pt[2]), 4),
                             'yaw': yaw, 'pitch': float(c.pitch or 0.0), 'roll': float(c.roll or 0.0),
                             'module_id': mid, 'lens_id': lid, 'port': i + 1 if i < 8 else 0, 'sync': 'A',
                             'cable_m': 3.0, 'standoff_mm': 12.0, 'bracket': 'Flat plate · 25 mm',
                             'snap_normal': False, 'locked': False, 'visible': True, 'route_extra_m': 0.0,
                             'route_note': '', 'link': ''})
            if len(cams) > MAX_CAMERAS:
                continue
            k_l = k_r = 0
            for s in rig.lidars + rig.radars:
                model = s.library_model or ('ouster_os1_128' if s.kind == 'lidar' else 'conti_ars408_21')
                if model not in project.sensor_models:
                    continue
                x, y, z = scale(s.position_m)
                if s.kind == 'lidar':
                    k_l += 1
                    tag = f'L{k_l}'
                    m = project.sensor_models[model]
                    z = surface_top(x, y, mesh, getattr(project, 'rig', ())) + m.optical_z_mm / 1000.0
                else:
                    k_r += 1
                    tag = f'RD{k_r}'
                    yaw = float(s.yaw or 0.0)
                    d = np.array([math.cos(math.radians(yaw)), math.sin(math.radians(yaw)), 0.0])
                    pt, _n = snap_point(mesh, (x, y, z), d, 0.02)
                    x, y, z = float(pt[0]), float(pt[1]), float(pt[2])
                sens.append({'tag': tag, 'name': s.name, 'kind': s.kind, 'model_id': model, 'x': round(x, 4),
                             'y': round(y, 4), 'z': round(z, 4), 'yaw': float(s.yaw or 0.0),
                             'pitch': float(s.pitch or 0.0), 'roll': float(s.roll or 0.0), 'phase_deg': 0.0,
                             'port': '', 'visible': True, 'locked': False})
        except Exception as e:                         # a body the rays miss: skip this reference
            out.append((None, f'reference rig {rig.name} not used: {type(e).__name__}: {e}'))
            continue
        for a in cams:                                 # mirror images become linked pairs
            if a['link']:
                continue
            b = next((b for b in cams if b is not a and not b['link'] and b['lens_id'] == a['lens_id']
                      and abs(b['y'] + a['y']) < 0.05 and abs(b['x'] - a['x']) < 0.05 and abs(b['yaw'] + a['yaw']) < 1
                      and abs(a['y']) > 0.05), None)
            if b is not None:
                a['link'], b['link'] = b['tag'], a['tag']
        out.append((Rig(cams, sens, f'reference rig {rig.company} {rig.name}, scaled to this vehicle'),
                    f'seeded with the {rig.company} {rig.name} layout scaled from the {veh.name} to this vehicle'))
    return out


# =========================================================================== results
@dataclass
class DesignerResult:
    settings: DesignerSettings
    options: list               # [option dict] (see _option)
    current: dict               # the current variant, evaluated the same way
    rejected: list
    search: dict
    detection_kind: str
    detection_text: str
    notes: list
    project: str
    variant: str
    variant_uid: str
    created: str
    inputs_key: str
    cost_complete: bool
    grid: dict

    def option(self, label):
        return next((o for o in self.options if o['label'] == label), None)


def _changes(rig: Rig, cur: Rig, project) -> list:
    """What the option changes against the current variant, in words."""
    out = []
    ca = {c['tag']: c for c in cur.cams}
    cb = {c['tag']: c for c in rig.cams}
    for t in sorted(set(ca) - set(cb)):
        out.append(f'remove camera {t}')
    for t in sorted(set(cb) - set(ca)):
        c = cb[t]
        ln = project.lenses.get(c['lens_id'])
        out.append(f'add camera {t} ({project.modules[c["module_id"]].name if c["module_id"] in project.modules else c["module_id"]}'
                   f' · {ln.name if ln else c["lens_id"]}) at {c["x"]:.2f}, {c["y"]:.2f}, {c["z"]:.2f} m, yaw '
                   f'{c["yaw"]:.0f}°, pitch {c["pitch"]:.0f}°')
    for t in sorted(set(ca) & set(cb)):
        a, b = ca[t], cb[t]
        bits = []
        if (a['module_id'], a['lens_id']) != (b['module_id'], b['lens_id']):
            ln = project.lenses.get(b['lens_id'])
            bits.append(f'lens → {ln.name if ln else b["lens_id"]}' + (f' ({ln.label})' if ln and ln.label else ''))
        if any(abs(a[k] - b[k]) > 5e-4 for k in ('x', 'y', 'z')):
            bits.append(f'moved to {b["x"]:.3f}, {b["y"]:.3f}, {b["z"]:.3f} m')
        if abs(a['yaw'] - b['yaw']) > 0.05 or abs(a['pitch'] - b['pitch']) > 0.05:
            bits.append(f'aimed yaw {b["yaw"]:.1f}°, pitch {b["pitch"]:.1f}°')
        if bits:
            out.append(f'{t}: ' + ', '.join(bits))
    sa = {s['tag']: s for s in cur.sensors}
    sb = {s['tag']: s for s in rig.sensors}
    for t in sorted(set(sa) - set(sb)):
        out.append(f'remove {sa[t]["kind"]} {t}')
    for t in sorted(set(sb) - set(sa)):
        m = project.sensor_models.get(sb[t]['model_id'])
        out.append(f'add {sb[t]["kind"]} {t} ({m.name if m else sb[t]["model_id"]})')
    for t in sorted(set(sa) & set(sb)):
        if sa[t]['model_id'] != sb[t]['model_id']:
            m = project.sensor_models.get(sb[t]['model_id'])
            out.append(f'{t} → {m.name if m else sb[t]["model_id"]}')
        elif any(abs(sa[t][k] - sb[t][k]) > 5e-4 for k in ('x', 'y', 'z', 'yaw', 'pitch')):
            out.append(f'{t} moved')
    return out


def _option(label, e: Eval, contrib, crit, cur: Eval, d: Designer, on_front: bool) -> dict:
    b = bill(d.p, e.rig, d.st.prices)
    verdict = 'pass' if e.feasible and not any(c['status'] == 'not_evaluated' and c['priority'] == 'must'
                                               for c in e.constraints) else ('fail' if not e.feasible else 'incomplete')
    return {'label': label, 'origin': e.rig.origin, 'history': list(e.rig.history),
            'changes': _changes(e.rig, cur.rig, d.p), 'rig': {'cams': copy.deepcopy(e.rig.cams),
                                                               'sensors': copy.deepcopy(e.rig.sensors)},
            'rig_key': e.rig.key(), 'feasible': bool(e.feasible), 'on_front': bool(on_front), 'verdict': verdict,
            'items': copy.deepcopy(e.items), 'constraints': copy.deepcopy(e.constraints),
            'margin': e.margin if math.isfinite(e.margin) else None, 'must_fail': list(e.must_fail),
            'should_pass': e.should_pass, 'cost': e.cost, 'cost_missing': list(e.cost_missing),
            'bill': b['lines'], 'power_w': round(e.power, 2), 'power_missing': list(e.power_missing),
            'detection': round(e.detection, 4), 'detection_std': (round(e.detection_std, 4)
                                                                  if e.detection_std is not None else None),
            'detection_kind': e.detection_kind, 'regions': copy.deepcopy(e.regions), 'counts': dict(e.counts),
            'robust': copy.deepcopy(e.robust), 'sensors': contrib, 'critique': crit,
            'issues': {s: sum(1 for i in (getattr(e.res, 'issues', None) or []) if i.severity == s)
                       for s in ('critical', 'serious', 'warning')}}


def option_variant(project, opt: dict, name=None):
    """The option's rig as a new Variant (new uids), named and noted with where it came from."""
    rig = Rig(copy.deepcopy(opt['rig']['cams']), copy.deepcopy(opt['rig']['sensors']), opt.get('origin', ''))
    base = project.variant.name if project.variants else 'Design'
    name = name or f'{base} · design {opt["label"]}'
    note = (f'Design agent option {opt["label"]} ({opt.get("origin", "")}). Changes: '
            + ('; '.join(opt.get('changes') or ['none'])) + '.')
    return rig.to_variant(name, note[:2000])


def inputs_key(project, st: DesignerSettings) -> str:
    """The design's inputs hashed: the requirements, constraints, prices and search settings, and what the studies
    read (requirements.inputs_key of the active variant)."""
    doc = {'items': st.items(), 'constraints': st.active_constraints(), 'prices': st.prices,
           'search': [st.options, st.evaluations, st.seed, st.vary_lenses, st.vary_count, st.vary_lidar, st.vary_radar,
                      st.placement, st.reference_seeds], 'studies': RQ.inputs_key(project, project.active_uid)}
    return hashlib.sha1(json.dumps(doc, sort_keys=True, default=str).encode()).hexdigest()[:16]


def run(project, engine=None, settings: DesignerSettings | None = None, variant=None, progress=None, cancel=None):
    """The design run on the project's active variant (variant: another one as the starting point). Returns the
    DesignerResult, or None when cancelled; raises ValueError with the reason when it cannot run."""
    p = project
    if variant is not None:
        uid = variant.uid if hasattr(variant, 'uid') else variant
        if p.variant_by_uid(uid) is not None:
            p = copy.copy(project)
            p.active_uid = uid
    d = Designer(p, engine, settings, progress, cancel)
    res, why = d.run()
    if res is None and why != 'Cancelled.':
        raise ValueError(why)
    return res


# =========================================================================== design log, accepting an option
def _cut(t, n=380):
    t = ' '.join(str(t).split())
    return t if len(t) <= n else t[:n - 1] + '…'


def option_line(o, cur='USD') -> str:
    m = o.get('margin')
    return (f'{o["label"]}: {_money(o["cost"], cur)}, {o["power_w"]:.0f} W, margin '
            + ('—' if m is None else f'{m:+.2f}') + f', detection {o["detection"] * 100:.0f} %'
            + ('' if o['detection_kind'] == 'learned' else ' (geometric)') + f', {o["counts"]["cameras"]} cameras, '
            f'{o["counts"]["lidars"]} lidars, {o["counts"]["radars"]} radars' + ('' if o['feasible'] else ' · misses '
                                                                                   'a must requirement'))


def _why_not(o, acc, cur='USD') -> str:
    bits = []
    if not o['feasible']:
        bits.append('it misses ' + ', '.join(o['must_fail'][:3] or [c['kind'] for c in o['constraints']
                                                                    if c['status'] == 'fail'][:2]))
    if o['cost'] is not None and acc['cost'] is not None and o['cost'] > acc['cost'] + 0.5:
        bits.append(f'it costs {_money(o["cost"] - acc["cost"], cur)} more')
    if o['power_w'] > acc['power_w'] + 0.5:
        bits.append(f'it draws {o["power_w"] - acc["power_w"]:.0f} W more')
    if o.get('margin') is not None and acc.get('margin') is not None and o['margin'] < acc['margin'] - 1e-3:
        bits.append(f'less margin ({o["margin"]:+.2f} against {acc["margin"]:+.2f})')
    if o['detection'] < acc['detection'] - 1e-3:
        bits.append(f'less detection ({o["detection"] * 100:.0f} % against {acc["detection"] * 100:.0f} %)')
    return '; '.join(bits) or 'the user preferred the other option'


def log_entries(project, res: DesignerResult, label: str, why: str = '', variant_name='') -> list:
    """The design-log entries of an accepted option (assistant/memory.py's form): the brief, the requirements with
    their sources, the options considered, the one accepted and why the others were not."""
    from ..assistant import memory as M
    st, cur = res.settings, res.settings.currency
    out = []
    docs = ', '.join(d.get('name', '') for d in st.documents if isinstance(d, dict))
    out.append(M.new_entry('note', _cut(f'Design agent brief: {st.brief.strip() or "(no brief text)"}'
                                        + (f' · documents: {docs}' if docs else '')), source='assistant'))
    for r in [r for r in st.requirements if r.get('accepted', True)][:24]:
        src = r.get('source') or {}
        loc = f'{src.get("doc", "")} {src.get("location", "")}'.strip()
        out.append(M.new_entry('constraint', _cut(f'{r["item"]["priority"].capitalize()}: {r["item"]["text"]} '
                                                  f'({RQ.describe(r["item"])})' + (f' · source {src.get("id", "")} '
                                                                                  f'{loc}: "{src.get("quote", "")}"'
                                                                                  if src else '')),
                               source='assistant'))
    for c in st.active_constraints()[:12]:
        src = c.get('source') or {}
        out.append(M.new_entry('constraint', _cut(f'{c["priority"].capitalize()}: {c["text"]} '
                                                  f'({RQ.CONSTRAINT_LABEL[c["kind"]]} {RQ.OP_TEXT.get(c["op"], "")} '
                                                  f'{c["value"]:g} {c.get("unit", "")})'.replace(' )', ')')
                                                  + (f' · source {src.get("id", "")}: "{src.get("quote", "")}"'
                                                     if src else '')), source='assistant'))
    out.append(M.new_entry('note', _cut(f'Design agent: {len(res.options)} options considered ('
                                        f'{res.search["rigs_scored"]} rigs scored, seed {res.search["seed"]}, '
                                        f'detection {res.detection_kind}): '
                                        + ' | '.join(option_line(o, cur) for o in res.options)), source='assistant'))
    acc = res.option(label)
    if acc is not None:
        out.append(M.new_entry('decision', _cut(f'Accepted design option {label}'
                                                + (f' as variant {variant_name}' if variant_name else '') + ': '
                                                + (why.strip() or option_line(acc, cur)) + ' · changes: '
                                                + '; '.join(acc['changes'][:6] or ['none'])), source='assistant'))
        for o in res.options:
            if o['label'] != label:
                out.append(M.new_entry('rejected', _cut(f'Design option {o["label"]} not chosen: '
                                                        f'{_why_not(o, acc, cur)} ({option_line(o, cur)})'),
                                       source='assistant'))
    return [e for e in out if e is not None]


APPROVE_OVER = 6                 # an option changing more than this many things needs the explicit approval step


def needs_approval(opt: dict) -> bool:
    return len(opt.get('changes') or []) > APPROVE_OVER or any(
        ch.startswith(('add lidar', 'remove lidar')) for ch in opt.get('changes') or [])


def option_sha256(opt: dict) -> str:
    from ..assistant import copilot as CP
    return CP.sha256({'rig': opt['rig'], 'changes': opt.get('changes') or [], 'label': opt['label']})


def approve(project, opt: dict) -> dict:
    """The approval record of a large change set (the Copilot's form: what, its SHA-256, when), written into the
    designer's settings; accept() refuses a large option without an intact record."""
    from ..assistant import copilot as CP
    st = settings_of(project)
    rec = {'option': opt['label'], 'sha256': option_sha256(opt), 'changes': list(opt.get('changes') or [])[:60],
           'approved_utc': CP.utc_now()}
    st.approvals = (st.approvals + [rec])[-40:]
    project.analysis.designer = st.to_dict()
    return rec


def accept(project, res: DesignerResult, label: str, why: str = '', name=None):
    """Accept option `label`: its variant added and made active, the design log written, the acceptance recorded
    (run inside store.edit(['variants', 'analysis'], ...) for one undo step). Returns the new Variant. Raises
    ValueError for an unknown option, or a large one without its approval record."""
    from ..assistant import memory as M
    opt = res.option(label)
    if opt is None:
        raise ValueError(f'No option {label!r}.')
    st = settings_of(project)
    if needs_approval(opt):
        sha = option_sha256(opt)
        if not any(a.get('sha256') == sha for a in st.approvals):
            raise ValueError(f'Option {label} changes {len(opt.get("changes") or [])} things: approve the change set '
                             'first.')
    v = option_variant(project, opt, name)
    taken = {x.name for x in project.variants}
    base, k = v.name, 2
    while v.name in taken:
        v.name, k = f'{base} ({k})', k + 1
    project.variants.append(v)
    project.active_uid = v.uid
    for e in log_entries(project, res, label, why, v.name):
        M.add(project, e)
    st = settings_of(project)
    st.accepted = (st.accepted + [{'option': label, 'variant': v.name, 'variant_uid': v.uid,
                                   'date': time.strftime('%Y-%m-%d'), 'why': _cut(why, 300),
                                   'rig_key': opt['rig_key']}])[-40:]
    project.analysis.designer = st.to_dict()
    return v


# =========================================================================== exports
def summary(res: DesignerResult) -> dict:
    cur = res.settings.currency
    return {'options': [{'label': o['label'], 'verdict': o['verdict'], 'cost': o['cost'], 'currency': cur,
                         'power_w': o['power_w'], 'margin': None if o['margin'] is None else round(o['margin'], 4),
                         'detection': o['detection'], 'detection_kind': o['detection_kind'],
                         'counts': o['counts'], 'changes': o['changes'][:8],
                         'critique': [f'{c["severity"]}: {c["message"]}' for c in o['critique'][:4]]}
                        for o in res.options],
            'current': {'verdict': res.current['verdict'], 'cost': res.current['cost'],
                        'power_w': res.current['power_w'], 'margin': res.current['margin']},
            'search': {k: res.search[k] for k in ('rigs_scored', 'seconds', 'stopped', 'seed')},
            'detection': res.detection_text, 'cost_complete': res.cost_complete}


def _clean(o):
    if isinstance(o, float):
        return None if not math.isfinite(o) else round(o, 6)
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.generic):
        return _clean(o.item())
    return o


def to_doc(res: DesignerResult, project=None) -> dict:
    st = res.settings
    reqs = [{'item': r['item'], 'source': {k: v for k, v in (r.get('source') or {}).items() if k != 'path'},
             'accepted': r.get('accepted', True), 'origin': r.get('origin', '')} for r in st.requirements]
    return _clean({
        'schema': SCHEMA, 'project': res.project, 'variant': res.variant, 'created': res.created,
        'units': {'cost': st.currency, 'power': 'W', 'positions': 'm', 'angles': 'degrees',
                  'margin': 'relative to the target (margin / max(|target|, unit floor))',
                  'detection': 'share 0–1, area-weighted over the region grid'},
        'frame': 'vehicle (ISO 8855, rear axle)', 'brief': st.brief,
        'documents': [{'name': d.get('name')} for d in st.documents if isinstance(d, dict)],
        'extraction': {k: v for k, v in st.extraction.items() if k in ('method', 'model', 'usage', 'documents')},
        'requirements': reqs, 'constraints': st.constraints, 'not_checkable': st.not_checkable,
        'rejected_rows': st.rejected, 'prices': st.prices, 'currency': st.currency,
        'search': res.search, 'detection': {'kind': res.detection_kind, 'text': res.detection_text},
        'grid': res.grid, 'current': res.current, 'options': res.options, 'not_chosen': res.rejected,
        'notes': res.notes, 'assumptions': ASSUMPTIONS, 'limits': LIMITS_TEXT, 'inputs_key': res.inputs_key})


CSV_COLUMNS = ('option', 'verdict', 'on_front', 'cost', 'currency', 'cost_missing', 'power_w', 'margin', 'detection',
               'detection_kind', 'cameras', 'lidars', 'radars', 'must_fail', 'robust_ok', 'changes')


def to_csv(res: DesignerResult) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(CSV_COLUMNS)
    for o in [res.current] + res.options:
        w.writerow([o['label'], o['verdict'], o['on_front'], '' if o['cost'] is None else f'{o["cost"]:.2f}',
                    res.settings.currency, ' '.join(o['cost_missing']), f'{o["power_w"]:.2f}',
                    '' if o['margin'] is None else f'{o["margin"]:.4f}', f'{o["detection"]:.4f}',
                    o['detection_kind'], o['counts']['cameras'], o['counts']['lidars'], o['counts']['radars'],
                    ' '.join(o['must_fail']), '' if o['robust'] is None else o['robust']['ok'],
                    '; '.join(o['changes'])])
    return buf.getvalue()


def sensors_csv(res: DesignerResult) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(('option', 'tag', 'kind', 'what', 'x', 'y', 'z', 'yaw', 'pitch', 'serves', 'fails_without',
                'unique_ground_m2', 'detection_lost', 'power_w', 'cost', 'why'))
    for o in res.options:
        for s in o['sensors']:
            p = s['pose']
            w.writerow([o['label'], s['tag'], s['kind'], s['what'], p.get('x'), p.get('y'), p.get('z'), p.get('yaw'),
                        p.get('pitch'), s['serves'], ' '.join(s['fails_without']),
                        '' if s['unique_ground_m2'] is None else s['unique_ground_m2'], s['detection_lost'],
                        s['power_w'], '' if s['cost'] is None else s['cost'], s['why']])
    return buf.getvalue()


def png(res: DesignerResult, axes=None, size=(900, 560)) -> bytes | None:
    """The options on the two chosen axes (the current variant hollow), light theme, for the export."""
    import cv2
    ax = list(axes or res.settings.axes)
    pts = [(o, axis_value(o, ax[0]), axis_value(o, ax[1])) for o in [res.current] + res.options]
    pts = [p for p in pts if p[1] is not None and p[2] is not None]
    W, H = size
    img = np.full((H, W, 3), 255, np.uint8)
    if not pts:
        cv2.putText(img, f'{OBJECTIVE_LABEL[ax[0]]} incomplete: enter the prices', (40, H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 60), 1, cv2.LINE_AA)
        return cv2.imencode('.png', img)[1].tobytes()
    xs = np.array([p[1] for p in pts], float)
    ys = np.array([p[2] for p in pts], float)
    pad = 70
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    x0, x1 = (x0 - 1, x1 + 1) if x1 - x0 < 1e-9 else (x0 - 0.08 * (x1 - x0), x1 + 0.08 * (x1 - x0))
    y0, y1 = (y0 - 1, y1 + 1) if y1 - y0 < 1e-9 else (y0 - 0.08 * (y1 - y0), y1 + 0.08 * (y1 - y0))

    def P(x, y):
        return (int(pad + (x - x0) / (x1 - x0) * (W - 2 * pad)), int(H - pad - (y - y0) / (y1 - y0) * (H - 2 * pad)))
    cv2.line(img, (pad, H - pad), (W - pad, H - pad), (120, 120, 120), 1)
    cv2.line(img, (pad, H - pad), (pad, pad), (120, 120, 120), 1)
    cv2.putText(img, OBJECTIVE_LABEL[ax[0]], (W // 2 - 40, H - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1,
                cv2.LINE_AA)
    cv2.putText(img, OBJECTIVE_LABEL[ax[1]], (8, pad - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1,
                cv2.LINE_AA)
    for v, (x, y) in ((x0, P(x0, y0)), (x1, P(x1, y0))):
        cv2.putText(img, _fmt(v), (x - 10, H - pad + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (90, 90, 90), 1, cv2.LINE_AA)
    for v in (y0, y1):
        cv2.putText(img, _fmt(v), (6, P(x0, v)[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (90, 90, 90), 1, cv2.LINE_AA)
    for o, x, y in pts:
        c = P(x, y)
        col = (214, 120, 42) if o['feasible'] else (59, 59, 208)
        if o['label'] == 'current':
            cv2.circle(img, c, 8, (90, 90, 90), 2, cv2.LINE_AA)
        else:
            cv2.circle(img, c, 7, col, -1, cv2.LINE_AA)
        cv2.putText(img, o['label'], (c[0] + 10, c[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1,
                    cv2.LINE_AA)
    return cv2.imencode('.png', img)[1].tobytes()


def axis_value(o: dict, axis: str):
    if axis == 'cost':
        return o['cost']
    if axis == 'power':
        return o['power_w']
    if axis == 'margin':
        return o['margin']
    if axis == 'detection':
        return o['detection']
    if axis == 'cameras':
        return float(o['counts']['cameras'])
    if axis == 'sensors':
        return float(o['counts']['sensors'])
    return None


def file_names(project) -> list:
    stem = _stem(project)
    return [f'{stem}-design-options.json', f'{stem}-design-options.csv', f'{stem}-design-sensors.csv',
            f'{stem}-design-options.png']


def _stem(project):
    import re
    return re.sub(r'[^A-Za-z0-9_-]+', '-', getattr(project, 'name', '') or 'project').strip('-') or 'project'


def files(res: DesignerResult, project=None) -> list:
    names = file_names(project)
    out = [(names[0], json.dumps(to_doc(res, project), indent=1, ensure_ascii=False).encode('utf-8')),
           (names[1], to_csv(res).encode('utf-8')), (names[2], sensors_csv(res).encode('utf-8'))]
    pic = png(res)
    if pic:
        out.append((names[3], pic))
    return out


def export(res: DesignerResult, project, out_dir) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for name, data in files(res, project):
        path = os.path.join(out_dir, name)
        with open(path, 'wb') as f:
            f.write(data)
        paths.append(path)
    return paths
