"""Requirements scorecard: the programme's written requirements as measurable pass/fail checks, evaluated against the
app's own studies in one place, with one verdict, a design-review pack (PDF) and a CI gate.

A requirement is {'id', 'text' (as written), 'metric', 'params', 'op', 'target', 'priority' ('must' | 'should'),
'note'}. Its metric comes from the registry below: an id, a label, a unit, the study that computes it, whether that
study is cheap (no geometry: worked out again on every edit) or a study (run once, in the background, shared by every
requirement that needs it), the parameters it takes (a sector, an object class, a sensor tag …, their choices taken
from the project) and a function that reads the value off the study's own result. A metric that cannot be evaluated
(no lidar to ask, no vehicle body, a sensor tag that is not in the variant) says why instead of guessing.

The margin is signed so that positive is good: value − target for ≥ and >, target − value for ≤ and <, and
−|value − target| for =. The verdict passes when every must requirement passes; a must requirement that fails makes
it fail, one that cannot be evaluated leaves it not evaluated (and the CI gate fails on both). Should requirements are
counted, never decisive.

Every figure is the study's own: the scorecard computes nothing of its own beyond reading, comparing and reporting.
The studies' modelling limits apply unchanged (each is stated on its page, in its JSON and in the review pack).
"""
from __future__ import annotations

import copy
import csv
import datetime as _dt
import hashlib
import html
import importlib
import io
import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

SCHEMA = 'vantage.requirements/1'
OPS = ('>=', '<=', '==', '<', '>')
OP_TEXT = {'>=': '≥', '<=': '≤', '==': '=', '<': '<', '>': '>'}
OP_WORDS = {'>=': 'at least', '<=': 'at most', '==': 'exactly', '<': 'less than', '>': 'more than'}
PRIORITIES = ('must', 'should')
STATUSES = ('pass', 'fail', 'not_evaluated')
STATUS_TEXT = {'pass': 'Pass', 'fail': 'Fail', 'not_evaluated': 'Not evaluated', 'not_run': 'Not run'}
VERDICT_TEXT = {'pass': 'Pass', 'fail': 'Fail', 'incomplete': 'Not evaluated', 'empty': 'No requirements'}
EXIT_FAIL = 5                       # check(): a must requirement fails or cannot be evaluated
LIMITS = {'mc_samples': (50, 5000), 'mc_pos_mm': (0.0, 200.0)}
ASSUMPTIONS = ('The scorecard reads every figure off the studies\' own results and compares it with the target; it '
               'models nothing itself. Each study\'s assumptions and limits apply unchanged (listed per study in the '
               'review pack). A requirement is only as good as its metric: "resolved" means what the perception range '
               'means by it (an object\'s height in pixels against the class minimum), "detected" what the lidar and '
               'radar coverage means, and so on.')
LIMITS_TEXT = ('Study metrics are evaluated when the scorecard is run (or the CI check), not on every edit; after an '
               'edit their last values are marked stale. The Issues list only re-checks requirements whose metric is '
               'cheap. The brief parser recognises simple phrasings with a number and a unit; it proposes, the user '
               'accepts.')
EVERY = ''                          # the parameter value meaning every sector / every class / the worst sensor


class NotEvaluated(Exception):
    """A metric that cannot be evaluated; the message says why in words for the user."""


class _Cancelled(Exception):
    pass


def _mod(name):
    """A sibling module of vantage.core, or None when it is not part of this build."""
    try:
        return importlib.import_module(f'{__package__}.{name}')
    except ImportError:
        return None


# =========================================================================== parameters
@dataclass(frozen=True)
class Param:
    key: str
    label: str
    choices: Callable                    # (project, variant) -> [(value, label)]
    default: str = EVERY

    def options(self, project, variant=None) -> list:
        try:
            return list(self.choices(project, variant or project.variant))
        except Exception:                 # a hand-edited project may hold anything: no choices, not a crash
            return [(self.default, self.default or 'every')]


def _sectors(p, v):
    from .perception import SECTORS
    return [(EVERY, 'every sector')] + [(n, n.replace('-', ' ')) for n, _ in SECTORS]


def _pr_classes(p, v):
    from .perception import PerceptionSettings
    return [(EVERY, 'every class')] + [(c['name'], c['name']) for c in
                                       PerceptionSettings.from_dict(p.analysis.perception).classes]


def _sc_classes(p, v):
    from .sensor_coverage import CoverageSettings
    return [(EVERY, 'every object')] + [(c['name'], c['name']) for c in
                                        CoverageSettings.from_dict(p.analysis.sensor_coverage).classes]


def _tags(kind, every):
    def f(p, v):
        if kind == 'camera':
            tags = [c.tag for c in v.cameras if c.visible]
        else:
            tags = [s.tag for s in v.sensors if s.visible and s.kind == kind]
        return [(EVERY, every)] + [(t, t) for t in tags]
    return f


def _fixed(*pairs):
    return lambda p, v: list(pairs)


def _vol_objects(p, v):
    from .volume import DEFAULT_OBJECTS
    return [(n, n) for n, _ in DEFAULT_OBJECTS]


P_SECTOR = Param('sector', 'Sector', _sectors)
P_CLASS = Param('cls', 'Object class', _pr_classes)
P_SC_CLASS = Param('cls', 'Object', _sc_classes)
P_MODALITY = Param('modality', 'Sensor type', _fixed(('camera', 'cameras'), ('lidar', 'lidars'), ('radar', 'radars')),
                   'lidar')
P_LIDAR = Param('lidar', 'Lidar', _tags('lidar', 'worst lidar'))
P_LIDAR_ALL = Param('lidar', 'Lidar', _tags('lidar', 'all lidars together'))
P_CAMERA = Param('camera', 'Camera', _tags('camera', 'worst camera'))
P_SEVERITY = Param('severity', 'Severity', _fixed(('critical', 'critical'), ('serious', 'critical or serious'),
                                                  ('warning', 'any (warnings too)')), 'critical')
P_LINES = Param('lines', 'Trigger lines', _fixed(('planned', 'as planned'), ('suggested', 'as suggested')),
                'planned')
P_VOL_OBJECT = Param('object', 'Object', _vol_objects, 'Child 0.9 m')
P_STAT = Param('stat', 'Statistic', _fixed(('p95', '95th percentile'), ('p99', '99th percentile'),
                                           ('max', 'worst sample'), ('mean', 'mean')), 'p95')


# =========================================================================== studies
@dataclass(frozen=True)
class Study:
    id: str
    label: str
    page: str                 # the main window's page that shows it ('' = none)
    menu: str                 # where the user finds it, in words
    cost: str                 # 'cheap' | 'study'
    run: Callable             # (ev) -> result; raises NotEvaluated
    figures: Callable         # (result, ev) -> [(label, text)]
    needs: tuple = ()
    module: str = ''          # the vantage.core module it needs ('' = always there)
    soft: tuple = ()          # studies it uses when they can run (planned before it, failures tolerated)
    png: Callable | None = None
    assumptions: Callable | None = None     # (result) -> text
    weight: float = 1.0       # share of the progress bar
    caption: str = ''         # under its PNG in the review pack


@dataclass(frozen=True)
class Metric:
    id: str
    label: str
    unit: str
    study: str
    better: str               # the natural operator: '>=' (more is better) or '<=' (less is better)
    fn: Callable              # (ev, params) -> (value, evidence dict); raises NotEvaluated
    params: tuple = ()
    decimals: int = 1
    help: str = ''


STUDIES: dict = {}
METRICS: dict = {}


def _study(**kw):
    s = Study(**kw)
    STUDIES[s.id] = s
    return s


def _metric(id, label, unit, study, better, decimals=1, params=(), help=''):
    def deco(fn):
        METRICS[id] = Metric(id, label, unit, study, better, fn, tuple(params), decimals, help)
        return fn
    return deco


def available(study_id) -> bool:
    s = STUDIES.get(study_id)
    return s is not None and (not s.module or _mod(s.module) is not None)


def metric_available(metric_id) -> bool:
    m = METRICS.get(metric_id)
    return m is not None and available(m.study)


# =========================================================================== evaluation context
class _Eval:
    """One evaluation: the project with the chosen variant active, the engine, and every study's result (each run at
    most once, shared by all requirements that need it)."""

    def __init__(self, project, engine=None, settings=None, progress=None, cancel=None, plan=()):
        self.p = project
        self._engine = engine
        self.st = settings or RequirementsSettings()
        self.progress = progress
        self.cancel = cancel
        self.cache, self.errors, self.seconds = {}, {}, {}
        self.plan = list(plan)            # studies this run will compute, in order (for progress)
        self.done_w = 0.0
        self.total_w = sum(STUDIES[s].weight for s in self.plan) or 1.0

    @property
    def v(self):
        return self.p.variant

    @property
    def engine(self):
        if self._engine is None:
            from .analysis import Engine
            self._engine = Engine()
        return self._engine

    def stop(self) -> bool:
        return self.cancel is not None and bool(self.cancel())

    def _prog(self, frac_in_study, sid):
        if self.progress is None:
            return
        w = STUDIES[sid].weight if sid in STUDIES else 0.0
        f = (self.done_w + w * min(max(float(frac_in_study), 0.0), 1.0)) / self.total_w
        try:
            self.progress(min(f, 1.0), STUDIES[sid].label if sid in STUDIES else '')
        except TypeError:
            self.progress(min(f, 1.0))

    def sub_progress(self, sid):
        return lambda f, *_: self._prog(f, sid)

    def study(self, sid):
        """The study's result (run now if needed); NotEvaluated when it cannot run here."""
        if sid in self.cache:
            return self.cache[sid]
        if sid in self.errors:
            raise NotEvaluated(self.errors[sid])
        s = STUDIES.get(sid)
        if s is None:
            raise NotEvaluated(f'Unknown study {sid!r}.')
        if s.module and _mod(s.module) is None:
            self.errors[sid] = f'The {s.label.lower()} study is not part of this build.'
            raise NotEvaluated(self.errors[sid])
        for d in s.needs:
            try:
                self.study(d)
            except NotEvaluated as e:
                self.errors[sid] = str(e)
                raise
        if self.stop():
            raise _Cancelled()
        t0 = time.perf_counter()
        self._prog(0.0, sid)
        try:
            res = s.run(self)
        except NotEvaluated as e:
            self.errors[sid] = str(e)
            raise
        except _Cancelled:
            raise
        except Exception as e:            # a study that fails takes its metrics with it, not the scorecard
            self.errors[sid] = f'The {s.label.lower()} study failed: {type(e).__name__}: {e}'
            raise NotEvaluated(self.errors[sid])
        finally:
            self.seconds[sid] = time.perf_counter() - t0
        if res is None and self.stop():
            raise _Cancelled()
        self.cache[sid] = res
        if sid in self.plan:
            self.done_w += s.weight
        self._prog(0.0, sid)
        return res

    def optional(self, sid):
        try:
            return self.study(sid)
        except NotEvaluated:
            return None


def _fig(label, value, unit='', dp=1):
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return (label, '—')
    if isinstance(value, str):
        return (label, value)
    return (label, f'{value:,.{dp}f}' + (f' {unit}' if unit else ''))


def _ev(figures=(), detail='', **kw):
    return dict(figures=[list(f) for f in figures], detail=detail, **kw)


def _finite(v) -> bool:
    try:
        return v is not None and math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


# =========================================================================== ground coverage and the Issues list
def _need_vehicle(ev):
    if ev.engine.vehicle_geom(ev.p) is None:
        raise NotEvaluated('No vehicle body in the project: import or generate one first.')


def _run_coverage(ev):
    _need_vehicle(ev)
    res = ev.engine.analyse(ev.p, progress=ev.sub_progress('coverage'))
    if res is None:
        raise NotEvaluated('The coverage analysis needs a vehicle body and an ROI.')
    return res


def _coverage_figures(res, ev):
    m = res.metrics
    return [_fig('ROI', m['roi_area'], 'm²', 2), _fig('Covered', m['covered_pct'], '%', 2),
            _fig('Blind ground', m['blind_m2'], 'm²', 3), _fig('Minimum density', m['min_density'], 'px/cm', 2),
            _fig(f'At the minimum density ({ev.p.analysis.min_density:g} px/cm)', m['min_ok_pct'], '%', 2),
            _fig(f'At the target density ({ev.p.analysis.target_density:g} px/cm)', m['target_ok_pct'], '%', 2),
            _fig('Seen by two or more cameras', m['multi_pct'], '%', 2), _fig('In focus', m['in_focus_pct'], '%', 2),
            _fig('Ground cell', m['cell'] * 100, 'cm', 0), _fig('Cameras', str(len(res.cams)))]


def _coverage_png(res, ev=None, px=600) -> bytes | None:
    """The ground coverage as a plan, forward up and left to the left: the best camera's density on the ROI (the
    sequential ramp from the minimum density to 4× it), blind ROI cells red, the body grey, outside the ROI pale."""
    import cv2
    from .sensor_coverage import SEQ
    g = res.grid
    roi, body = g.roi, g.body
    ramp = np.array([[int(h[i:i + 2], 16) for i in (1, 3, 5)] for h in SEQ], float)
    lo = float(ev.p.analysis.min_density) if ev is not None else 3.0
    hi = max(lo * 4.0, lo + 1.0)
    t = np.clip((res.best_dens - lo) / (hi - lo), 0.0, 1.0) * (len(ramp) - 1)
    i0 = np.floor(t).astype(int)
    i1 = np.minimum(i0 + 1, len(ramp) - 1)
    f = (t - i0)[..., None]
    rgb = ramp[i0] * (1 - f) + ramp[i1] * f
    img = np.empty(roi.shape + (3,), float)
    img[:] = (244, 243, 239)
    img[roi] = rgb[roi]
    img[roi & (res.count == 0)] = (208, 59, 59)
    img[roi & (res.count > 0) & (res.best_dens < lo)] = (250, 178, 25)
    img[body] = (120, 119, 114)
    img = img[::-1, ::-1].transpose(1, 0, 2)              # rows: x forward up; columns: y left to the left
    k = max(1, int(px / max(img.shape[:2])))
    img = np.repeat(np.repeat(img, k, 0), k, 1)
    ok, buf = cv2.imencode('.png', np.ascontiguousarray(img[..., ::-1]).astype(np.uint8))
    return buf.tobytes() if ok else None


_study(id='coverage', label='Ground coverage', page='viewports', menu='the Coverage panel under the 3D view',
       cost='study', run=_run_coverage, figures=_coverage_figures, png=_coverage_png, weight=2.0,
       caption='The ROI from above, forward up: the best camera\'s density from the minimum (light blue) to 4× it '
               '(dark blue), amber below the minimum, red where no camera sees the ground, grey the body.',
       assumptions=lambda r: 'Ground points in the ROI, seen by a camera when inside its image and lens circle and not '
                             'behind the body (the occlusion cube); density is the best camera\'s pixels per cm.')


def _cov_metric(key, label, unit, better, dp, help_):
    @_metric(f'coverage.{key}', label, unit, 'coverage', better, dp, help=help_)
    def f(ev, prm, key=key):
        res = ev.study('coverage')
        m = res.metrics
        if key == 'min_density' and m['covered_pct'] <= 0:
            raise NotEvaluated('No camera sees the ROI ground.')
        detail = ''
        if key == 'blind_m2' and m['blind_m2'] > 0:
            blind = res.grid.roi & (res.count == 0)
            jj, ii = np.nonzero(blind)
            if len(ii):
                from .analysis import describe_location
                detail = (f'Blind ground mostly '
                          f'{describe_location(float(res.grid.xs[ii].mean()), float(res.grid.ys[jj].mean()), ev.p)}.')
        return float(m[key]), _ev(_coverage_figures(res, ev)[:7], detail)
    return f


for _k, _l, _u, _b, _d, _h in (
        ('blind_m2', 'Blind ground in the ROI', 'm²', '<=', 3, 'Ground in the ROI no camera sees'),
        ('min_density', 'Minimum ground density', 'px/cm', '>=', 2, 'The lowest best-camera density on the seen ROI'),
        ('covered_pct', 'ROI ground covered', '%', '>=', 2, 'Share of the ROI some camera sees'),
        ('multi_pct', 'ROI seen by two or more cameras', '%', '>=', 1, 'Share of the ROI with overlap'),
        ('min_ok_pct', 'ROI at the minimum density', '%', '>=', 1,
         'Share of the ROI at or above the analysis\' minimum density'),
        ('target_ok_pct', 'ROI at the target density', '%', '>=', 1,
         'Share of the ROI at or above the analysis\' target density')):
    _cov_metric(_k, _l, _u, _b, _d, _h)


def _run_issues(ev):
    from . import rules
    res = ev.optional('coverage')
    vg = ev.engine.vehicle_geom(ev.p)
    mounts = {}
    for c in ev.v.cameras:
        try:
            mounts[c.uid] = rules.mount_info(ev.p, c, vg)
        except Exception:
            pass
    out = [i for i in rules.collect_issues(ev.p, res, vg, mounts) if i.action != 'requirement']
    if res is not None and not any(i.action == 'stitched' for i in out):
        sres = ev.optional('stitched')
        from . import stitched_objects as SO
        iss = SO.issue(sres) if sres is not None else None
        if iss is not None:
            out.append(iss)
            out.sort(key=lambda i: rules.SEVERITY_ORDER.get(i.severity, 9))
    return out


def _issues_figures(iss, ev):
    n = {s: sum(i.severity == s for i in iss) for s in ('critical', 'serious', 'warning')}
    return [(s.capitalize(), str(k)) for s, k in n.items()]


_study(id='issues', label='Issues list', page='viewports', menu='the Issues panel under the 3D view', cost='study',
       run=_run_issues, figures=_issues_figures, soft=('coverage', 'stitched'), weight=0.2)


@_metric('issues.count', 'Issues in the Issues list', '', 'issues', '<=', 0, (P_SEVERITY,),
         'Issues of the chosen severity (or worse) in the Issues list')
def _m_issues(ev, prm):
    iss = ev.study('issues')
    sev = prm.get('severity') or 'critical'
    keep = {'critical': ('critical',), 'serious': ('critical', 'serious'),
            'warning': ('critical', 'serious', 'warning')}.get(sev, ('critical',))
    hits = [i for i in iss if i.severity in keep]
    return float(len(hits)), _ev(_issues_figures(iss, ev),
                                 '; '.join(f'{i.severity}: {i.message}' for i in hits[:4])
                                 + (f'; and {len(hits) - 4} more' if len(hits) > 4 else ''))


# =========================================================================== perception range
def _run_perception(ev):
    from . import perception as PR
    if not any(c.visible for c in ev.v.cameras):
        raise NotEvaluated('No visible camera in the variant.')
    pr = PR.analyse(ev.p, ev.engine, None, PR.PerceptionSettings.from_dict(ev.p.analysis.perception),
                    progress=ev.sub_progress('perception'), cancel=ev.cancel)
    if pr is None:
        raise _Cancelled()
    return pr


def _perception_figures(pr, ev):
    out = [('Sector checks passed', f'{pr.metrics["passed"]} of {pr.metrics["checks"]}')]
    for c in pr.classes:
        out.append((f'{c.name} (min px {c.min_px:g})',
                    ' · '.join(f'{s["name"]} {s["classes"][c.name]["min_m"]:.0f}' for s in pr.sectors) + ' m'))
    out.append(('Required', ' · '.join(f'{s["name"]} {s["required_m"]:g}' for s in pr.sectors) + ' m'))
    return out


def _perception_png(pr, ev=None):
    from . import perception as PR
    return PR.png(pr, size=700)


_study(id='perception', label='Perception range', page='perception', menu='Analysis ▸ Perception range…',
       cost='study', run=_run_perception, figures=_perception_figures, png=_perception_png, weight=1.5,
       assumptions=lambda r: 'An object is resolved when it is in the image and the lens circle, its centre is not '
                             'hidden by the body, and its height in the image is at least the class\' minimum pixels. '
                             'Ranges from the vehicle centre on the ground; minimum pixel heights are typical, set '
                             'your detector\'s.')


def _pick(names, want, what):
    if want in (EVERY, None):
        return list(names)
    for n in names:
        if n.lower() == str(want).lower():
            return [n]
    raise NotEvaluated(f'No {what} {want!r} in the study\'s settings.')


@_metric('perception.range', 'Perception range', 'm', 'perception', '>=', 1, (P_CLASS, P_SECTOR),
         'The shortest range at which the cameras resolve the class in the sector (every direction of it)')
def _m_pr_range(ev, prm):
    pr = ev.study('perception')
    cls = _pick([c.name for c in pr.classes], prm.get('cls'), 'object class')
    secs = _pick([s['name'] for s in pr.sectors], prm.get('sector'), 'sector')
    rows = [(s['classes'][c]['min_m'], c, s['name'], s['classes'][c]['worst_azimuth_deg'])
            for s in pr.sectors if s['name'] in secs for c in cls]
    v, c, sname, az = min(rows, key=lambda r: r[0])
    figs = [(s['name'], ', '.join(f'{c_} {s["classes"][c_]["min_m"]:.1f} m' for c_ in cls)) for s in pr.sectors
            if s['name'] in secs]
    return float(v), _ev(figs, f'Shortest: {c} at {v:.1f} m in the {sname} sector'
                               + (f' (towards {az:.0f}°)' if az is not None else '') + '.')


@_metric('perception.short', 'Perception checks short of the required range', '', 'perception', '<=', 0, (),
         'Sector × class checks where the cameras fall short of the perception range\'s required range')
def _m_pr_short(ev, prm):
    pr = ev.study('perception')
    short = [f'{c} {s["name"]} ({s["classes"][c]["min_m"]:.0f} of {s["required_m"]:g} m)'
             for s in pr.sectors for c, d in s['classes'].items() if not d['pass']]
    return float(len(short)), _ev(_perception_figures(pr, ev)[:1], ('Short: ' + ', '.join(short) + '.') if short
                                  else 'Every class reaches the required range in every sector.')


# =========================================================================== lidar and radar coverage
def _run_sensor_coverage(ev):
    from . import sensor_coverage as SC
    if not any(s.visible and s.kind in ('lidar', 'radar') for s in ev.v.sensors):
        raise NotEvaluated('No lidar or radar in the variant.')
    res = SC.run(ev.p, ev.engine, None, None, progress=ev.sub_progress('sensor_coverage'), cancel=ev.cancel)
    if res is None:
        raise _Cancelled()
    return res


def _sc_figures(res, ev):
    from . import sensor_coverage as SC
    m = res.metrics
    out = [('Sector checks', f'{m["redundant"]} redundant · {m["single"]} single · {m["gap"]} gaps of {m["checks"]}')]
    for L in res.lidars:
        st = SC._stats(L.blind)
        out.append((f'Lidar {L.tag} ({L.spec.name})',
                    f'{100 * L.blocked:.1f} % of points on the vehicle · nearest ring '
                    + (f'{L.ring_m:.1f} m' if math.isfinite(L.ring_m) else '—')
                    + (f' · ground blind {st["median_m"]:.1f} m (median from the body)'
                       if st['median_m'] is not None else '')))
    for R_ in res.radars:
        out.append((f'Radar {R_.tag}', R_.spec.name))
    return out


def _sc_png(res, ev=None):
    from . import sensor_coverage as SC
    return SC.png(res, size=700)


_study(id='sensor_coverage', label='Lidar and radar coverage', page='coverage',
       menu='Analysis ▸ Lidar and radar coverage…', cost='study', run=_run_sensor_coverage, figures=_sc_figures,
       png=_sc_png, weight=2.5,
       assumptions=lambda r: 'Lidar points on an object are counted from the beam geometry (min points per frame '
                             'set per object); a radar detects an object whose centre is in a zone at the range its '
                             'cross-section allows, nothing of the vehicle in between. Library sensor figures are '
                             'nominal.')


def _sc_classes_of(res, want):
    return _pick([c.name for c in res.classes], want, 'object')


@_metric('sensor_coverage.gaps', 'Sectors no sensor type covers', '', 'sensor_coverage', '<=', 0, (P_SC_CLASS,),
         'Sector × object checks where no sensor type (camera, lidar, radar) reaches the required range')
def _m_sc_gaps(ev, prm):
    res = ev.study('sensor_coverage')
    cls = _sc_classes_of(res, prm.get('cls'))
    gaps = [f'{c} {s["name"]}' for s in res.sectors for c in cls if s['classes'][c]['verdict'] == 'gap']
    return float(len(gaps)), _ev(_sc_figures(res, ev)[:1], ('Gaps: ' + ', '.join(gaps) + '.') if gaps else
                                 'Some sensor type reaches the required range in every sector.')


@_metric('sensor_coverage.redundant_pct', 'Sector checks covered by two sensor types', '%', 'sensor_coverage', '>=',
         0, (P_SC_CLASS,), 'Share of sector × object checks where two or more sensor types reach the required range')
def _m_sc_redundant(ev, prm):
    res = ev.study('sensor_coverage')
    cls = _sc_classes_of(res, prm.get('cls'))
    v = [s['classes'][c]['verdict'] for s in res.sectors for c in cls]
    n = sum(x == 'redundant' for x in v)
    return 100.0 * n / max(len(v), 1), _ev(_sc_figures(res, ev)[:1], f'{n} of {len(v)} checks redundant.')


@_metric('sensor_coverage.range', 'Detection range', 'm', 'sensor_coverage', '>=', 1,
         (P_MODALITY, P_SC_CLASS, P_SECTOR),
         'The shortest range at which a sensor type detects the object in the sector (every direction of it)')
def _m_sc_range(ev, prm):
    res = ev.study('sensor_coverage')
    mod = prm.get('modality') or 'lidar'
    if mod not in ('camera', 'lidar', 'radar'):
        raise NotEvaluated(f'Unknown sensor type {mod!r}.')
    cls = _sc_classes_of(res, prm.get('cls'))
    secs = _pick([s['name'] for s in res.sectors], prm.get('sector'), 'sector')
    arr = {'camera': res.camera, 'lidar': res.lidar, 'radar': res.radar}[mod]
    ks = [k for k, c in enumerate(res.classes) if c.name in cls]
    if not np.isfinite(arr[ks]).any():
        why = {'camera': 'no camera resolves it (or the perception range has no class of that name)',
               'lidar': 'no lidar in the variant', 'radar': 'no radar in the variant, or radars are not relied on '
                                                            'for it (RCS 0)'}[mod]
        raise NotEvaluated(f'No {mod} range for {", ".join(cls)}: {why}.')
    rows = []
    for s in res.sectors:
        if s['name'] in secs:
            for c in cls:
                x = s['classes'][c][f'{mod}_m']
                rows.append((0.0 if x is None else float(x), c, s['name'], x is None))
    v, c, sname, hole = min(rows, key=lambda r: r[0])
    figs = [(s['name'], ', '.join(f'{c_} ' + ('—' if s['classes'][c_][f'{mod}_m'] is None else
                                              f'{s["classes"][c_][f"{mod}_m"]:.1f} m') for c_ in cls))
            for s in res.sectors if s['name'] in secs]
    return v, _ev(figs, (f'The {mod}s do not reach {c} in every direction of the {sname} sector.' if hole else
                         f'Shortest: {c} at {v:.1f} m in the {sname} sector.'))


def _lidars_of(res, want):
    if not res.lidars:
        raise NotEvaluated('No lidar in the variant.')
    if want in (EVERY, None):
        return list(res.lidars)
    hit = [L for L in res.lidars if L.tag.lower() == str(want).lower()]
    if not hit:
        raise NotEvaluated(f'No visible lidar {want!r} in the variant.')
    return hit


@_metric('sensor_coverage.lidar_blocked_pct', 'Lidar points landing on the vehicle', '%', 'sensor_coverage', '<=', 1,
         (P_LIDAR,), 'Share of a lidar\'s points per frame that hit the own vehicle or rig')
def _m_sc_blocked(ev, prm):
    res = ev.study('sensor_coverage')
    Ls = _lidars_of(res, prm.get('lidar'))
    L = max(Ls, key=lambda L_: L_.blocked)
    figs = [(L_.tag, f'{100 * L_.blocked:.1f} %') for L_ in Ls]
    sec = ', '.join(f'{n} {100 * x:.0f} %' for n, x in L.blocked_sectors.items() if x)
    return 100.0 * L.blocked, _ev(figs, f'{L.tag}: {100 * L.blocked:.1f} %' + (f' (by heading: {sec})' if sec else ''))


@_metric('sensor_coverage.ground_blind_m', 'Lidar ground blind zone', 'm', 'sensor_coverage', '<=', 2, (P_LIDAR_ALL,),
         'Median over the directions: how far from the body the nearest ground point the lidar(s) reach lies')
def _m_sc_blind(ev, prm):
    from . import sensor_coverage as SC
    res = ev.study('sensor_coverage')
    want = prm.get('lidar')
    if want in (EVERY, None):
        if not res.lidars:
            raise NotEvaluated('No lidar in the variant.')
        st, who = SC._stats(res.blind), 'all lidars together'
    else:
        L = _lidars_of(res, want)[0]
        st, who = SC._stats(L.blind), L.tag
    if st['median_m'] is None:
        raise NotEvaluated(f'{who}: the lidar beams reach the ground in no direction.')
    return float(st['median_m']), _ev([_fig('Median', st['median_m'], 'm', 2), _fig('Best', st['min_m'], 'm', 2),
                                       _fig('Worst', st['max_m'], 'm', 2),
                                       _fig('Directions reaching the ground', 100 * st['directions_reached'], '%', 0)],
                                      f'{who}: median {st["median_m"]:.2f} m from the body outline.')


@_metric('sensor_coverage.overlap_pct', 'Camera image covered by lidar points at 20 m', '%', 'sensor_coverage', '>=',
         0, (P_CAMERA,), 'Share of a camera\'s image whose scene point 20 m out some lidar covers')
def _m_sc_overlap(ev, prm):
    res = ev.study('sensor_coverage')
    if not res.lidars:
        raise NotEvaluated('No lidar in the variant.')
    rows = res.overlap
    want = prm.get('camera')
    if want not in (EVERY, None):
        rows = [o for o in rows if o['tag'].lower() == str(want).lower()]
        if not rows:
            raise NotEvaluated(f'No visible camera {want!r} in the variant.')
    if not rows:
        raise NotEvaluated('No visible camera in the variant.')
    o = min(rows, key=lambda o_: o_['share']['20'])
    return 100.0 * o['share']['20'], _ev([(x['tag'], f'{100 * x["share"]["20"]:.0f} %') for x in rows],
                                         f'{o["tag"]}: {100 * o["share"]["20"]:.0f} % of its image'
                                         + (f' (best lidar {o["best"]})' if o['best'] else '') + '.')


# =========================================================================== sync and timing
def _run_timing(ev):
    from . import timing as TM
    return TM.plan(ev.p)


def _timing_figures(res, ev):
    m, st = res.metrics, res.settings
    out = []
    if res.lidar is not None:
        out += [('Reference lidar', f'{res.lidar.tag} · {res.lidar.rate_hz:g} Hz'),
                _fig('Worst misalignment, as planned', m['worst_ms'], 'ms', 2),
                _fig('Worst misalignment, suggested lines', res.suggestion['worst_ms'], 'ms', 2),
                _fig('Vehicle travel during it', m['worst_travel_cm'], 'cm', 1),
                ('Trigger lines', f'{m["lines_used"]} used of {st.trigger_lines}')]
    else:
        out.append(('Sweep plan', res.reason))
    out += [_fig(f'Worst motion blur at {st.speed_kmh:g} km/h', m['worst_blur_px'], 'px', 2),
            _fig('Longest exposure within the blur limit', m['max_exposure_ms'], 'ms', 2)]
    return out


def _timing_png(res, ev=None):
    from . import timing as TM
    return TM.png(res, size=700)


_study(id='timing', label='Sync and timing', page='timing', menu='Analysis ▸ Sync and timing…', cost='cheap',
       run=_run_timing, figures=_timing_figures, png=_timing_png, weight=0.1,
       assumptions=lambda r: 'A spinning lidar phase-locked to the sync; a camera is aligned when its middle row\'s '
                             'exposure is centred on the sweep crossing its view 20 m out. Blur for a static object '
                             'while driving straight ahead.')


@_metric('timing.misalignment_ms', 'Worst camera–lidar misalignment', 'ms', 'timing', '<=', 2, (P_LINES,),
         'Worst time between a triggered camera\'s exposure centre and the lidar sweep crossing its view')
def _m_tm_mis(ev, prm):
    res = ev.study('timing')
    if res.lidar is None:
        raise NotEvaluated(res.reason)
    sug = prm.get('lines') == 'suggested'
    v = res.suggestion['worst_ms'] if sug else res.metrics['worst_ms']
    if v is None:
        raise NotEvaluated('No camera is on a trigger line (every camera free-runs), so none is aligned.')
    worst = res.metrics.get('worst_tag')
    return float(v), _ev(_timing_figures(res, ev)[:5],
                         (f'With the suggested trigger lines ({res.suggestion["lines"]} lines).' if sug else
                          f'As planned; the worst camera is {worst}.' if worst else 'As planned.'))


@_metric('timing.blur_px', 'Worst motion blur', 'px', 'timing', '<=', 2, (),
         'The most an image point moves during the exposure at the planned speed (worst camera)')
def _m_tm_blur(ev, prm):
    res = ev.study('timing')
    if not res.cams:
        raise NotEvaluated('No visible camera in the variant.')
    m, st = res.metrics, res.settings
    return float(m['worst_blur_px']), _ev(_timing_figures(res, ev)[-2:],
                                          f'{m["worst_blur_tag"]} at {st.speed_kmh:g} km/h, {st.exposure_ms:g} ms '
                                          f'exposure, an object {st.object_m:g} m away.')


# =========================================================================== recording budget
def _run_recording(ev):
    from . import recording as RB
    return RB.budget(ev.p)


def _recording_figures(res, ev):
    t, st = res.totals, res.settings
    return [_fig('Data rate', t['mb_s'], 'MB/s', 1), _fig('Per hour', t['gb_h'], 'GB/h', 0),
            _fig(f'Hours per {st.storage_tb:g} TB disk set', t['hours_per_set'], 'h', 2),
            _fig('Sustained write needed', t['write_needed_mb_s'], f'of {st.write_mb_s:,.0f} MB/s', 0),
            _fig('Uplink load', t['uplink_load'] * 100, '%', 1),
            ('Cameras recorded as', st.camera_format)]


_study(id='recording', label='Recording budget', page='recording', menu='Analysis ▸ Recording budget…', cost='cheap',
       run=_run_recording, figures=_recording_figures, weight=0.1,
       assumptions=lambda r: _mod('recording').ASSUMPTIONS)


@_metric('recording.hours_per_set', 'Driving per disk set', 'h', 'recording', '>=', 2, (),
         'Hours of driving until one disk set is full')
def _m_rb_hours(ev, prm):
    res = ev.study('recording')
    t = res.totals
    if not math.isfinite(t['hours_per_set']):
        raise NotEvaluated('Nothing is recorded: no visible camera, lidar or radar.')
    return float(t['hours_per_set']), _ev(_recording_figures(res, ev)[:3],
                                          f'{t["gb_h"]:,.0f} GB/h into {res.settings.storage_tb:g} TB.')


@_metric('recording.write_load_pct', 'Disk write load (with headroom)', '%', 'recording', '<=', 0, (),
         'The data rate × the write headroom against the disks\' sustained write')
def _m_rb_write(ev, prm):
    res = ev.study('recording')
    t = res.totals
    return 100.0 * t['write_load'], _ev(_recording_figures(res, ev)[:4],
                                        f'{t["write_needed_mb_s"]:,.0f} of {t["write_mb_s"]:,.0f} MB/s.')


@_metric('recording.uplink_load_pct', 'Uplink load', '%', 'recording', '<=', 0, (),
         'The lidars\' and radars\' share of the switch-to-recorder uplink')
def _m_rb_uplink(ev, prm):
    res = ev.study('recording')
    t = res.totals
    if not t['networked']:
        raise NotEvaluated('No lidar or radar is recorded over the uplink.')
    return 100.0 * t['uplink_load'], _ev(_recording_figures(res, ev)[4:5],
                                         f'{t["uplink_mbps"]:,.1f} of {t["uplink_capacity_mbps"]:,.0f} Mbit/s.')


@_metric('recording.gb_per_hour', 'Data per hour of driving', 'GB/h', 'recording', '<=', 0, (),
         'Everything recorded per hour (sensors, GNSS/IMU/CAN and the container overhead)')
def _m_rb_gbh(ev, prm):
    res = ev.study('recording')
    return float(res.totals['gb_h']), _ev(_recording_figures(res, ev)[:2], '')


@_metric('recording.link_load_pct', 'Busiest sensor link', '%', 'recording', '<=', 0, (),
         'The busiest camera GMSL link or lidar/radar link against its rate')
def _m_rb_link(ev, prm):
    res = ev.study('recording')
    rows = [r for r in res.rows if r['load'] is not None]
    if not rows:
        raise NotEvaluated('No sensor link with a known rate.')
    r = max(rows, key=lambda r_: r_['load'])
    return 100.0 * r['load'], _ev([(x['tag'], f'{100 * x["load"]:.0f} % of {x["link"]}') for x in rows],
                                  f'{r["tag"]}: {r["link_mbps"]:,.1f} of {r["link_capacity_mbps"]:,.0f} Mbit/s.')


# =========================================================================== blind volume
def _run_volume(ev):
    from . import volume as VOL
    res = ev.study('coverage')
    st = VOL.VolumeSettings.from_dict(ev.p.analysis.volume)
    vr = VOL.analyse_volume(ev.p, ev.engine, res, st.height, st.dz, st.cell or None,
                            progress=ev.sub_progress('volume'), cancel=ev.cancel)
    if vr is None:
        if ev.stop():
            raise _Cancelled()
        raise NotEvaluated('The blind volume needs a vehicle body.')
    return vr


def _volume_figures(vr, ev):
    m = vr.metrics
    return [_fig(f'Blind volume up to {vr.height:g} m', m['blind_m3'], 'm³', 2), _fig('Share', m['blind_pct'], '%', 1),
            _fig('Covered to (≤ 1 % blind at every level)', m['covered_to_m'], 'm', 2),
            _fig('Never seen at any height', m['never_seen_m2'], 'm²', 3)] + \
        [_fig(f'{n}: undetected', d['undetected_m2'], 'm²', 3) for n, d in m['objects'].items()]


_study(id='volume', label='Blind volume', page='volume', menu='Analysis ▸ Blind volume…', cost='study',
       run=_run_volume, figures=_volume_figures, needs=('coverage',), weight=1.0,
       assumptions=lambda r: 'The ROI columns sampled level by level with the ground analysis\' visibility test; an '
                             'object is detected when any camera sees any part of its column.')


@_metric('volume.blind_m3', 'Blind volume', 'm³', 'volume', '<=', 2, (),
         'Space inside the ROI up to the study height that no camera sees')
def _m_vol(ev, prm):
    vr = ev.study('volume')
    return float(vr.metrics['blind_m3']), _ev(_volume_figures(vr, ev)[:3], '')


@_metric('volume.undetected_m2', 'Ground where an object stays undetected', 'm²', 'volume', '<=', 3, (P_VOL_OBJECT,),
         'Area of the ROI where an object of that height standing there is seen by no camera')
def _m_vol_obj(ev, prm):
    vr = ev.study('volume')
    want = prm.get('object') or P_VOL_OBJECT.default
    d = next((v for n, v in vr.metrics['objects'].items() if n.lower() == str(want).lower()), None)
    if d is None:
        raise NotEvaluated(f'No object {want!r} in the blind volume study.')
    return float(d['undetected_m2']), _ev(_volume_figures(vr, ev)[4:],
                                          f'{want}: reach {d["reach_m"]:.2f} m from the body at worst.')


@_metric('volume.covered_to_m', 'Height covered all round', 'm', 'volume', '>=', 2, (),
         'The height up to which at most 1 % of the ROI is blind at every level')
def _m_vol_h(ev, prm):
    vr = ev.study('volume')
    return float(vr.metrics['covered_to_m']), _ev(_volume_figures(vr, ev)[:3], '')


# =========================================================================== objects in the stitched view
def _run_stitched(ev):
    from . import stitched_objects as SO
    res = ev.study('coverage')
    s = SO.analyse(ev.p, ev.engine, res, progress=ev.sub_progress('stitched'), cancel=_Event(ev.stop))
    if s is None:
        raise _Cancelled()
    return s


class _Event:
    """A cancel callable as the threading.Event the stitched-view study polls."""

    def __init__(self, fn):
        self.is_set = fn


def _stitched_figures(s, ev):
    m = s.metrics
    return [('Object', f'{m["object"]} ({m["diameter_m"]:g} m wide)'),
            _fig('Vanishes at a seam', m['vanish_m2'], 'm²', 3), _fig('Cut off the view', m['cut_m2'], 'm²', 3),
            _fig('Seen by no camera', m['blind_m2'], 'm²', 3),
            _fig('Mean share of its height shown', 100 * m['shown_mean'], '%', 0)]


def _stitched_png(s, ev=None):
    from . import stitched_objects as SO
    return SO.png(s)


_study(id='stitched', label='Objects in the stitched view', page='viewports',
       menu='the Objects heatmap of the 3D view', cost='study', run=_run_stitched, figures=_stitched_figures,
       png=_stitched_png, needs=('coverage',), weight=0.6,
       assumptions=lambda r: 'A pole standing on every ROI cell, drawn as each camera\'s ground projection; the '
                             'stitched view shows a camera\'s smear only where its seams hand the output to it.')


@_metric('stitched.vanish_m2', 'Ground where an object vanishes from the stitched view', 'm²', 'stitched', '<=', 3,
         (), 'Area where a camera sees the object but the seams hand its smear to cameras that do not')
def _m_so_vanish(ev, prm):
    s = ev.study('stitched')
    m = s.metrics
    w = m['worst_seam']
    seam = s.seams[w] if 0 <= w < len(s.seams) else None
    return float(m['vanish_m2']), _ev(_stitched_figures(s, ev),
                                      f'Mostly at the {seam["tags"][0]} | {seam["tags"][1]} seam.' if seam else '')


@_metric('stitched.shown_pct', 'Share of an object the stitched view shows', '%', 'stitched', '>=', 0, (),
         'Mean share of the object\'s height the stitched view shows, where a camera sees it')
def _m_so_shown(ev, prm):
    s = ev.study('stitched')
    return 100.0 * float(s.metrics['shown_mean']), _ev(_stitched_figures(s, ev), '')


# =========================================================================== mounting tolerance
def _tol_cell(ev):
    return max(ev.p.analysis.cell_m, 0.05)


def _run_tol_single(ev):
    from .tolerance import worst_single_camera
    _need_vehicle(ev)
    if not any(c.visible for c in ev.v.cameras):
        raise NotEvaluated('No visible camera in the variant.')
    a = ev.p.analysis
    return worst_single_camera(ev.engine, ev.p, a.tol_deg, a.tol_mm / 1000.0, cell=_tol_cell(ev))


def _tol_single_figures(w, ev):
    a = ev.p.analysis
    tag = next((c.tag for c in ev.v.cameras if c.uid == w[1]), w[1] or '—')
    return [('Tolerance', f'±{a.tol_deg:g}° yaw/pitch or {a.tol_mm:g} mm along the body, one camera at a time'),
            _fig('Worst blind ground', w[0], 'm²', 3), ('Camera', tag), ('Error', _describe_pert(w[2]))]


def _describe_pert(d):
    if not d:
        return 'none makes any ground blind'
    dy, dp = d[0], d[1]
    if any(abs(x) > 0 for x in d[3:]):
        return f'slid {math.hypot(*d[3:5]) * 1000:.0f} mm along the body'
    return ', '.join(f'{n} {x:+g}°' for n, x in (('yaw', dy), ('pitch', dp)) if x)


_study(id='tolerance_single', label='Mounting tolerance (one camera off)', page='tolerance',
       menu='Analysis ▸ Tolerance study…', cost='study', run=_run_tol_single, figures=_tol_single_figures,
       weight=1.5,
       assumptions=lambda r: 'One camera at a time turned by the analysis tolerance in yaw or pitch, or slid along '
                             'the body; position errors keep the nominal occlusion map.')


@_metric('tolerance.single_blind_m2', 'Blind ground with one camera off by the tolerance', 'm²', 'tolerance_single',
         '<=', 3, (), 'The worst blind ground when any one camera is off by the analysis tolerance')
def _m_tol_single(ev, prm):
    w = ev.study('tolerance_single')
    figs = _tol_single_figures(w, ev)
    return float(w[0]), _ev(figs, f'Worst: {figs[2][1]}, {figs[3][1]}.')


def _run_tol_mc(ev):
    from .tolerance import TolSettings, run_monte_carlo
    _need_vehicle(ev)
    if not any(c.visible for c in ev.v.cameras):
        raise NotEvaluated('No visible camera in the variant.')
    a = ev.p.analysis
    st = TolSettings(yaw=a.tol_deg, pitch=a.tol_deg, roll=a.tol_deg, pos_mm=ev.st.mc_pos_mm,
                     samples=int(ev.st.mc_samples), cell=_tol_cell(ev))
    try:
        r = run_monte_carlo(ev.engine, ev.p, st, progress=ev.sub_progress('tolerance_mc'), cancel=ev.cancel, threads=2)
    except RuntimeError:
        if ev.stop():
            raise _Cancelled()
        raise
    if r.cancelled:
        raise _Cancelled()
    return r


def _tol_mc_stat(r, stat):
    b = np.asarray(r.blind, float)
    return {'p95': float(np.percentile(b, 95)), 'p99': float(np.percentile(b, 99)), 'max': float(b.max()),
            'mean': float(b.mean())}.get(stat, float(np.percentile(b, 95)))


def _tol_mc_figures(r, ev):
    st = r.settings
    return [('Mounting errors', f'±{st.yaw:g}° yaw/pitch/roll, ±{st.pos_mm:g} mm, uniform, every camera at once'),
            ('Samples', f'{len(r.blind)} (seed {st.seed})'),
            _fig('Nominal blind ground', r.nominal_blind, 'm²', 3),
            _fig('Median', r.pct(50), 'm²', 3), _fig('95th percentile', r.pct(95), 'm²', 3),
            _fig('Worst sample', float(np.max(r.blind)), 'm²', 3),
            _fig('Samples with no blind ground', r.zero_pct, '%', 1)]


_study(id='tolerance_mc', label='Mounting tolerance (Monte Carlo)', page='tolerance',
       menu='Analysis ▸ Tolerance study…', cost='study', run=_run_tol_mc, figures=_tol_mc_figures, weight=3.0,
       assumptions=lambda r: 'Every camera off at once by a uniform random error within the tolerance; position '
                             'errors keep the nominal occlusion map (accurate for small offsets).')


@_metric('tolerance.mc_blind_m2', 'Blind ground under random mounting errors', 'm²', 'tolerance_mc', '<=', 3,
         (P_STAT,), 'Blind ground over random mounting errors of every camera within the tolerance (Monte Carlo)')
def _m_tol_mc(ev, prm):
    r = ev.study('tolerance_mc')
    stat = prm.get('stat') or 'p95'
    return _tol_mc_stat(r, stat), _ev(_tol_mc_figures(r, ev), '')


# =========================================================================== power budget
def _run_power(ev):
    return _mod('power').run(ev.p)


def _power_figures(res, ev):
    t, st = res.totals, res.settings
    return [_fig('Rig load, steady', t['steady_w'], 'W', 1), _fig('Rig load, peak', t['peak_w'], 'W', 1),
            _fig(f'DC-DC load ({st.dcdc_w:g} W rating)', t['dcdc_load'] * 100, '%', 0),
            _fig('From the vehicle', t['dcdc_input_w'], 'W', 1),
            _fig('Alternator margin', t['alternator_margin_w'], 'W', 0),
            _fig(f'Engine-off runtime ({st.battery_wh:g} Wh battery)', t['runtime_min'], 'min', 0),
            ('Worst circuit', t['worst_circuit'] or '—'), _fig('Enclosure heat', t['heat_w'], 'W', 0)]


_study(id='power', label='Power budget', page='power', menu='Analysis ▸ Power budget…', cost='cheap',
       run=_run_power, figures=_power_figures, module='power', weight=0.1,
       assumptions=lambda r: _mod('power').ASSUMPTIONS)


def _power_metric(id, label, unit, better, dp, help_):
    return _metric(f'power.{id}', label, unit, 'power', better, dp, help=help_)


@_power_metric('runtime_min', 'Engine-off runtime', 'min', '>=', 0,
               'How long the auxiliary battery runs the rig with the engine off')
def _m_pw_runtime(ev, prm):
    res = ev.study('power')
    t = res.totals
    if not math.isfinite(t['runtime_min']):
        raise NotEvaluated('The rig draws nothing: no device with a power figure.')
    return float(t['runtime_min']), _ev(_power_figures(res, ev)[3:6],
                                        f'{res.settings.battery_wh:g} Wh × {res.settings.battery_dod * 100:g} % '
                                        f'usable at {t["dcdc_input_w"]:,.0f} W.')


@_power_metric('dcdc_load_pct', 'DC-DC converter load', '%', '<=', 0,
               'The steady output (loads and cable losses) against the converter\'s rating')
def _m_pw_dcdc(ev, prm):
    res = ev.study('power')
    t = res.totals
    return 100.0 * t['dcdc_load'], _ev(_power_figures(res, ev)[:4],
                                       f'{t["dcdc_output_w"]:,.0f} of {res.settings.dcdc_w:g} W.')


@_power_metric('alternator_margin_w', 'Alternator margin while driving', 'W', '>=', 0,
               'The alternator\'s spare power minus what the rig takes from the vehicle (negative: the battery drains)')
def _m_pw_alt(ev, prm):
    res = ev.study('power')
    t = res.totals
    return float(t['alternator_margin_w']), _ev(_power_figures(res, ev)[3:5],
                                                f'{t["alternator_spare_w"]:,.0f} W spare, {t["dcdc_input_w"]:,.0f} W '
                                                f'taken.')


@_power_metric('steady_w', 'Rig load, steady', 'W', '<=', 1, 'What every device and load draws, steady')
def _m_pw_steady(ev, prm):
    res = ev.study('power')
    return float(res.totals['steady_w']), _ev(_power_figures(res, ev)[:2], '')


@_power_metric('circuit_load_pct', 'Busiest fused circuit', '%', '<=', 0,
               'The steady current of the busiest circuit against its fuse')
def _m_pw_circuit(ev, prm):
    res = ev.study('power')
    cs = [c for c in res.circuits if c['load'] is not None]
    if not cs:
        raise NotEvaluated('No fused circuit carries a load.')
    c = max(cs, key=lambda c_: c_['load'])
    return 100.0 * c['load'], _ev([(x['label'], f'{x["steady_a"]:.2f} A on {x["fuse_a"]:g} A '
                                                f'({100 * x["load"]:.0f} %)') for x in cs],
                                  f'{c["label"]}: {c["steady_a"]:.2f} A steady, {c["peak_a"]:.2f} A peak on a '
                                  f'{c["fuse_a"]:g} A fuse.')


@_power_metric('cable_drop_pct', 'Worst power-cable drop', '%', '<=', 1,
               'The largest steady voltage drop on a lidar\'s, radar\'s or IMU\'s power cable')
def _m_pw_drop(ev, prm):
    res = ev.study('power')
    rows = [r for r in res.rows if r.get('drop_pct') is not None]
    if not rows:
        raise NotEvaluated('No lidar, radar or IMU power cable to check.')
    r = max(rows, key=lambda r_: r_['drop_pct'])
    return float(r['drop_pct']), _ev([(x['tag'], f'{x["drop_pct"]:.2f} % over {x["cable_m"]:.1f} m of '
                                                 f'{x["awg"]:g} AWG') for x in rows],
                                     f'{r["tag"]}: {r["drop_pct"]:.2f} % at {res.settings.bus_v:g} V.')


@_power_metric('heat_w', 'Heat in the enclosure', 'W', '<=', 0, 'The steady power of the devices inside the enclosure')
def _m_pw_heat(ev, prm):
    res = ev.study('power')
    t = res.totals
    return float(t['heat_w']), _ev([('Inside', ', '.join(t['heat_tags']) or 'nothing')], '')


def _plan_power(project, v):
    pw = _mod('power')
    if pw is None:
        return []
    st = pw.PowerSettings.from_dict(project.analysis.power)
    out = [{'metric': 'power.dcdc_load_pct', 'params': {}, 'op': '<=', 'target': 100.0, 'priority': 'must',
            'text': f'The {st.dcdc_w:g} W DC-DC converter carries the rig', 'note': 'The power budget\'s converter '
                                                                                   'rating.'},
           {'metric': 'power.alternator_margin_w', 'params': {}, 'op': '>=', 'target': 0.0, 'priority': 'must',
            'text': 'The alternator carries the rig while driving', 'note': 'The power budget\'s alternator spare '
                                                                           'current.'}]
    if st.runtime_target_min > 0:
        out.append({'metric': 'power.runtime_min', 'params': {}, 'op': '>=', 'target': st.runtime_target_min,
                    'priority': 'should', 'text': f'{st.runtime_target_min:g} min engine-off on the auxiliary battery',
                    'note': 'The power budget\'s engine-off runtime target.'})
    out.append({'metric': 'power.cable_drop_pct', 'params': {}, 'op': '<=', 'target': st.max_drop_pct,
                'priority': 'should', 'text': f'Power-cable drops at most {st.max_drop_pct:g} %',
                'note': 'The power budget\'s cable drop limit.'})
    return out


# =========================================================================== GNSS and IMU
def _pose_tags(p, v):
    tags = [c.tag for c in v.cameras if c.visible] + [s.tag for s in v.sensors if s.visible and s.kind != 'imu']
    return [(EVERY, 'worst sensor')] + [(t, t) for t in tags]


P_ANTENNA = Param('antenna', 'GNSS antenna', _tags('gnss', 'worst antenna'))
P_POSE_TAG = Param('sensor', 'Sensor', _pose_tags)


def _run_gnss(ev):
    gn = _mod('gnss')
    if not any(s.visible and s.kind in ('gnss', 'imu') for s in ev.v.sensors):
        raise NotEvaluated('No GNSS antenna or IMU in the variant.')
    res = gn.run(ev.p, ev.engine, progress=ev.sub_progress('gnss'), cancel=ev.cancel)
    if res is None:
        raise _Cancelled()
    return res


def _gnss_figures(res, ev):
    out = [(f'{a.tag} sky above {a.mask_deg:g}°', f'{100 * a.visible:.1f} % open · {a.expected_sats:.0f} satellites '
                                                  f'expected') for a in res.antennas]
    b = res.baseline
    if b is not None:
        out += [('Heading baseline', f'{b["primary"]} → {b["secondary"]}: {b["horizontal_m"]:.3f} m horizontal'),
                _fig('Heading error (1σ)', b['heading_sigma_deg'], '°', 3)]
    if res.imu is not None:
        out.append(('Reference IMU', f'{res.imu["tag"]} ({res.imu["model"]})'))
    rows = (res.budget or {}).get('rows', [])
    if rows:
        w = max(rows, key=lambda r: r['total_mm'])
        out.append(('Worst sensor position error (1σ)', f'{w["tag"]}: {w["total_mm"]:.0f} mm'))
    return out


def _gnss_png(res, ev=None):
    return _mod('gnss').png(res) if res.antennas else None


_study(id='gnss', label='GNSS and IMU', page='gnss', menu='Analysis ▸ GNSS and IMU…', cost='study', run=_run_gnss,
       figures=_gnss_figures, png=_gnss_png, module='gnss', weight=0.5,
       assumptions=lambda r: _mod('gnss').ASSUMPTIONS)


def _antennas_of(res, want):
    if not res.antennas:
        raise NotEvaluated('No visible GNSS antenna in the variant.')
    if want in (EVERY, None):
        return list(res.antennas)
    hit = [a for a in res.antennas if a.tag.lower() == str(want).lower()]
    if not hit:
        raise NotEvaluated(f'No visible GNSS antenna {want!r} in the variant.')
    return hit


@_metric('gnss.sky_pct', 'Open sky above the mask', '%', 'gnss', '>=', 1, (P_ANTENNA,),
         'Share of the sky above a GNSS antenna\'s elevation mask that the vehicle, the rig and the sensors leave open')
def _m_gn_sky(ev, prm):
    res = ev.study('gnss')
    ants = _antennas_of(res, prm.get('antenna'))
    a = min(ants, key=lambda a_: a_.visible)
    blk = ', '.join(f'{b["name"]} {100 * b["share"]:.1f} %' for b in a.blockers if b.get('share', 0) > 0)
    return 100.0 * a.visible, _ev([(x.tag, f'{100 * x.visible:.1f} % above {x.mask_deg:g}°') for x in ants],
                                  f'{a.tag}: ' + (f'blocked by {blk}.' if blk else 'nothing blocks it above the mask.'))


@_metric('gnss.expected_sats', 'Expected satellites in view', 'satellites', 'gnss', '>=', 0, (P_ANTENNA,),
         'Satellites above the mask in open sky × the open share (an even sky assumed: an indicator)')
def _m_gn_sats(ev, prm):
    res = ev.study('gnss')
    ants = _antennas_of(res, prm.get('antenna'))
    a = min(ants, key=lambda a_: a_.expected_sats)
    return float(a.expected_sats), _ev([(x.tag, f'{x.expected_sats:.1f} of {x.n_sats:g}') for x in ants],
                                       f'{a.tag}: {a.expected_sats:.1f} of {a.n_sats:g} above the mask.')


def _baseline(res):
    if res.baseline is None:
        raise NotEvaluated('No dual-antenna heading: it needs an INS with two antenna inputs and two visible GNSS '
                           'antennas.')
    return res.baseline


@_metric('gnss.heading_sigma_deg', 'Dual-antenna heading error (1σ)', '°', 'gnss', '<=', 3, (),
         'The INS\'s heading error from the antenna baseline (its datasheet figure scaled by the baseline)')
def _m_gn_heading(ev, prm):
    res = ev.study('gnss')
    b = _baseline(res)
    return float(b['heading_sigma_deg']), _ev(_gnss_figures(res, ev),
                                              f'{b["rated_sigma_deg"]:g}° rated at {b["rated_baseline_m"]:g} m, '
                                              f'{b["horizontal_m"]:.3f} m here.')


@_metric('gnss.baseline_m', 'Heading baseline (horizontal)', 'm', 'gnss', '>=', 2, (),
         'The horizontal length between the primary and the secondary GNSS antenna')
def _m_gn_base(ev, prm):
    res = ev.study('gnss')
    b = _baseline(res)
    return float(b['horizontal_m']), _ev(_gnss_figures(res, ev),
                                         f'{b["primary"]} → {b["secondary"]}: {b["length_m"]:.3f} m, '
                                         f'{b["vertical_deg"]:.1f}° vertical.')


@_metric('gnss.pose_sigma_mm', 'Sensor position error from the INS (1σ)', 'mm', 'gnss', '<=', 0, (P_POSE_TAG,),
         'How well a sensor\'s world position is known: INS position ⊕ attitude error on the lever arm ⊕ lever-arm '
         'measurement')
def _m_gn_pose(ev, prm):
    res = ev.study('gnss')
    rows = (res.budget or {}).get('rows', [])
    if res.imu is None or not rows:
        raise NotEvaluated('No IMU or INS in the variant: no pose error budget.')
    want = prm.get('sensor')
    if want not in (EVERY, None):
        rows = [r for r in rows if r['tag'].lower() == str(want).lower()]
        if not rows:
            raise NotEvaluated(f'No visible sensor {want!r} in the error budget.')
    r = max(rows, key=lambda r_: r_['total_mm'])
    return float(r['total_mm']), _ev([(x['tag'], f'{x["total_mm"]:.0f} mm ({x["lever_m"]:.2f} m lever arm)')
                                      for x in rows[:12]],
                                     f'{r["tag"]}: {r["total_mm"]:.0f} mm, {r["far_mm"]:.0f} mm at '
                                     f'{res.settings.label_range_m:g} m.')


def _plan_gnss(project, v):
    gn = _mod('gnss')
    if gn is None:
        return []
    out = []
    if any(s.visible and s.kind == 'gnss' for s in v.sensors):
        out.append({'metric': 'gnss.sky_pct', 'params': {'antenna': EVERY}, 'op': '>=', 'target': 90.0,
                    'priority': 'should', 'text': '≥ 90 % sky above the mask for each GNSS antenna',
                    'note': 'The GNSS study flags more than 10 % of the sky blocked.'})
    ins = [s for s in v.sensors if s.visible and s.kind == 'imu' and project.sensor_model_for(s).antennas >= 2]
    if ins and sum(s.visible and s.kind == 'gnss' for s in v.sensors) >= 2:
        out.append({'metric': 'gnss.baseline_m', 'params': {}, 'op': '>=', 'target': 1.0, 'priority': 'should',
                    'text': 'Dual-antenna heading baseline at least 1 m',
                    'note': 'The GNSS study flags a baseline under 1 m.'})
    return out


# =========================================================================== sun glare and dirt
def _dirt_tags(p, v):
    tags = [c.tag for c in v.cameras if c.visible] + [s.tag for s in v.sensors if s.visible and s.kind == 'lidar']
    return [(EVERY, 'worst camera or lidar')] + [(t, t) for t in tags]


P_DIRT_TAG = Param('sensor', 'Camera or lidar', _dirt_tags)


def _run_exposure(ev):
    xp = _mod('exposure')
    if not any(c.visible for c in ev.v.cameras) and not any(s.visible for s in ev.v.sensors):
        raise NotEvaluated('No visible camera, lidar or radar in the variant.')
    res = xp.run(ev.p, ev.engine, progress=ev.sub_progress('exposure'), cancel=ev.cancel)
    if res is None:
        raise _Cancelled()
    return res


def _exposure_figures(res, ev):
    st = res.settings
    out = [('Campaign', f'{st.lat:.2f}°, {st.lon:.2f}° · {st.start} to {st.end} · {st.hour_from:g}–{st.hour_to:g} h')]
    for it in res.items:
        if it.sun is not None:
            out.append((f'{it.tag} sun', f'{100 * it.sun["share"]:.1f} % in the image' if it.kind == 'camera' else
                        f'{100 * it.sun["share"]:.1f} % in the field'))
    for it in res.items:
        if it.kind != 'radar' and it.dirt:
            out.append((f'{it.tag} dirt', f'{it.dirt["score"]:.2f} ({it.dirt["class"]})'))
    return out


def _exposure_png(res, ev=None):
    return _mod('exposure').png(res, width=1100)


_study(id='exposure', label='Sun glare and dirt', page='exposure', menu='Analysis ▸ Sun glare and dirt…',
       cost='study', run=_run_exposure, figures=_exposure_figures, png=_exposure_png, module='exposure', weight=0.8,
       assumptions=lambda r: _mod('exposure').ASSUMPTIONS)


def _exposure_items(res, kind, want, what):
    items = [it for it in res.items if it.kind == kind and it.sun is not None]
    if not items:
        raise NotEvaluated(f'No visible {what} in the variant.')
    if want not in (EVERY, None):
        items = [it for it in items if it.tag.lower() == str(want).lower()]
        if not items:
            raise NotEvaluated(f'No visible {what} {want!r} in the variant.')
    return items


def _sun_metric(key, label, help_):
    @_metric(f'exposure.{key}_pct', label, '%', 'exposure', '<=', 1, (P_CAMERA,), help_)
    def f(ev, prm, key=key):
        res = ev.study('exposure')
        items = _exposure_items(res, 'camera', prm.get('camera'), 'camera')
        field_ = 'share' if key == 'sun' else key
        it = max(items, key=lambda i: i.sun[field_])
        s_ = it.sun
        when = ', '.join(x for x in (_mod('exposure').months_text(s_['worst_months']),
                                     _mod('exposure').hours_text(s_['worst_hours'])) if x and x != '—')
        return 100.0 * s_[field_], _ev([(i.tag, f'{100 * i.sun[field_]:.1f} %') for i in items],
                                       f'{it.tag}: {100 * s_[field_]:.1f} % of the driving time'
                                       + (f', mostly {when}' if when and key == 'sun' else '') + '.')
    return f


_sun_metric('sun', 'Sun in the camera image', 'Share of the driving time with the sun inside the image, not hidden by '
                                              'the body (clear sky)')
_sun_metric('near', 'Sun just outside the image (flare risk)', 'Share of the driving time with the sun within the '
                                                               'flare margin outside the image')
_sun_metric('low', 'Low sun ahead in the image', 'Share of the driving time with the sun in the image below the '
                                                 'low-sun elevation')


@_metric('exposure.lidar_sun_pct', 'Sun in the lidar\'s field', '%', 'exposure', '<=', 1, (P_LIDAR,),
         'Share of the driving time with the sun inside a lidar\'s field (solar background: an indicator)')
def _m_xp_lidar(ev, prm):
    res = ev.study('exposure')
    items = _exposure_items(res, 'lidar', prm.get('lidar'), 'lidar')
    it = max(items, key=lambda i: i.sun['share'])
    return 100.0 * it.sun['share'], _ev([(i.tag, f'{100 * i.sun["share"]:.1f} %') for i in items], '')


@_metric('exposure.dirt_score', 'Dirt and spray exposure', '', 'exposure', '<=', 2, (P_DIRT_TAG,),
         'A 0–1 heuristic score of a lens or lidar window (wheel spray, rear wake, frontal impact, attitude, height)')
def _m_xp_dirt(ev, prm):
    res = ev.study('exposure')
    items = [it for it in res.items if it.kind in ('camera', 'lidar') and it.dirt]
    if not items:
        raise NotEvaluated('No visible camera or lidar in the variant.')
    want = prm.get('sensor')
    if want not in (EVERY, None):
        items = [it for it in items if it.tag.lower() == str(want).lower()]
        if not items:
            raise NotEvaluated(f'No visible camera or lidar {want!r} in the variant.')
    it = max(items, key=lambda i: i.dirt['score'])
    d = it.dirt
    return float(d['score']), _ev([(i.tag, f'{i.dirt["score"]:.2f} ({i.dirt["class"]})') for i in items],
                                  f'{it.tag}: {d["score"]:.2f}, {d["class"]}' + (f'; {d["advice"]}' if d.get('advice')
                                                                                else '') + '. A heuristic.')


# =========================================================================== fleet
def _run_fleet(ev):
    fl = _mod('fleet')
    res = fl.run(ev.p, variant=ev.v, cancel=ev.cancel)
    if res is None:
        raise _Cancelled()
    if not any(v['records'] for v in res.vehicles):
        raise NotEvaluated('No vehicle with a measured record in the fleet (Calibrate ▸ Fleet…).')
    return res


def _fleet_figures(res, ev):
    cells = list(res.cells.values())
    st = res.settings
    n = {k: sum(c['verdict'] == k for c in cells) for k in ('ok', 'adjust', 're-mount')}
    worst = max(cells, key=lambda c: c['ratio'], default=None)
    name = {v['id']: v['name'] for v in res.vehicles}
    out = [('Fleet', f'{sum(1 for v in res.vehicles if v["records"])} vehicles with records, '
                     f'{sum(v["records"] for v in res.vehicles)} records'),
           ('Tolerance', f'±{st.pos_tol_mm:g} mm, {st.ang_tol_deg:g}°'),
           ('Latest poses', f'{n["ok"]} within tolerance · {n["adjust"]} to adjust · {n["re-mount"]} to re-mount'),
           ('Drifting', str(len(res.flagged('drift')))), ('Jumps', str(len(res.flagged('jump')))),
           ('Outliers', str(len(res.flagged('outlier'))))]
    if worst is not None:
        out.append(('Worst', f'{name.get(worst["vehicle"], worst["vehicle"])} {worst["tag"]}: '
                             f'{100 * worst["ratio"]:.0f} % of the tolerance ({worst["date"]})'))
    return out


def _fleet_png(res, ev=None):
    return _mod('fleet').png(res, width=1100) if res.cells else None


_study(id='fleet', label='Fleet', page='fleet', menu='Calibrate ▸ Fleet…', cost='study', run=_run_fleet,
       figures=_fleet_figures, png=_fleet_png, module='fleet', weight=0.2,
       assumptions=lambda r: _mod('fleet').ASSUMPTIONS)


def _fleet_cells(res, verdicts):
    name = {v['id']: v['name'] for v in res.vehicles}
    hits = [c for c in res.cells.values() if c['verdict'] in verdicts]
    return hits, [(f'{name.get(c["vehicle"], c["vehicle"])} {c["tag"]}',
                   f'{c["verdict"]}: {100 * c["ratio"]:.0f} % of the tolerance') for c in
                  sorted(hits, key=lambda c: -c['ratio'])[:12]]


@_metric('fleet.out_of_tolerance', 'Fleet poses out of tolerance', '', 'fleet', '<=', 0, (),
         'Vehicle × sensor poses whose latest measurement is beyond the fleet tolerance (to adjust or re-mount)')
def _m_fl_out(ev, prm):
    res = ev.study('fleet')
    hits, figs = _fleet_cells(res, ('adjust', 're-mount'))
    return float(len(hits)), _ev(figs or _fleet_figures(res, ev)[:3], '')


@_metric('fleet.remount', 'Fleet poses to re-mount', '', 'fleet', '<=', 0, (),
         'Vehicle × sensor poses beyond three times the tolerance in their latest measurement')
def _m_fl_remount(ev, prm):
    res = ev.study('fleet')
    hits, figs = _fleet_cells(res, ('re-mount',))
    return float(len(hits)), _ev(figs or _fleet_figures(res, ev)[:3], '')


@_metric('fleet.worst_pct', 'Worst fleet pose against the tolerance', '%', 'fleet', '<=', 0, (),
         'The largest offset of any vehicle\'s latest pose as a share of the tolerance (100 % = at the tolerance)')
def _m_fl_worst(ev, prm):
    res = ev.study('fleet')
    if not res.cells:
        raise NotEvaluated('No measured pose of a planned sensor in the fleet.')
    c = max(res.cells.values(), key=lambda c_: c_['ratio'])
    return 100.0 * float(c['ratio']), _ev(_fleet_figures(res, ev), '')


@_metric('fleet.drifting', 'Drifting sensors in the fleet', '', 'fleet', '<=', 0, (),
         'Vehicle × sensor poses whose dated records trend out of tolerance within the drift window')
def _m_fl_drift(ev, prm):
    res = ev.study('fleet')
    name = {v['id']: v['name'] for v in res.vehicles}
    hits = res.flagged('drift')
    return float(len(hits)), _ev([(f'{name.get(v, v)} {t}', 'drifting') for v, t in hits[:12]]
                                 or _fleet_figures(res, ev)[:4], '')


@_metric('fleet.outliers', 'Outlier vehicles in the fleet', '', 'fleet', '<=', 0, (),
         'Vehicle × sensor poses far from the fleet median (robust z-score)')
def _m_fl_outliers(ev, prm):
    res = ev.study('fleet')
    out = [(o['name'], f'{f["tag"]} {o["component"]}: z {o["z"]:.1f}') for f in res.fleet for o in f['outliers']]
    return float(len(out)), _ev(out[:12] or _fleet_figures(res, ev)[:3], '')


def _plan_fleet(project, v):
    fl = _mod('fleet')
    if fl is None or not any(x.get('records') for x in fl.vehicles(project)):
        return []
    st = fl.settings_of(project)
    return [{'metric': 'fleet.out_of_tolerance', 'params': {}, 'op': '<=', 'target': 0.0, 'priority': 'should',
             'text': f'Every vehicle\'s sensors within ±{st.pos_tol_mm:g} mm and {st.ang_tol_deg:g}° of the plan',
             'note': 'The fleet\'s tolerance (its latest measurement of each vehicle).'},
            {'metric': 'fleet.remount', 'params': {}, 'op': '<=', 'target': 0.0, 'priority': 'must',
             'text': 'No sensor in the fleet to re-mount', 'note': 'Beyond three times the fleet tolerance.'}]


# =========================================================================== mount brackets
NOT_BUILT = ('standoff', 'floating', 'inside', 'face')


def _run_brackets(ev):
    if not any(c.visible for c in ev.v.cameras):
        raise NotEvaluated('No visible camera in the variant.')
    _need_vehicle(ev)
    return _mod('brackets').plan(ev.p, ev.engine, variant=ev.v)


def _brackets_figures(res, ev):
    return [(b.tag, f'{b.status}: {b.message}' if b.message else b.status) for b in res.brackets]


_study(id='brackets', label='Mount brackets (plan)', page='brackets', menu='Analysis ▸ Mount brackets…',
       cost='study', run=_run_brackets, figures=_brackets_figures, module='brackets', weight=0.5,
       assumptions=lambda r: _mod('brackets').ASSUMPTIONS)


def _run_brackets_full(ev):
    plan = ev.study('brackets')
    ok = [b.uid for b in plan.brackets if b.status not in NOT_BUILT and b.status != 'skipped']
    if not ok:
        raise NotEvaluated('No camera can take a printed bracket as planned (see the bracket plan: a standoff is '
                           'needed, or no surface).')
    res = _mod('brackets').run(ev.p, ev.engine, cams=ok, variant=ev.v, progress=ev.sub_progress('brackets_full'),
                               cancel=ev.cancel)
    if res is None:
        raise _Cancelled()
    return res


def _brackets_full_figures(res, ev):
    out = []
    for b in res.brackets:
        if b.generated:
            out.append((b.tag, f'{b.kind} on {b.surface or "the surface"} · image blocked '
                               f'{b.params.get("fov_blocked_pct", float("nan")):.1f} % · lever arm '
                               f'{b.params.get("lever_mm", float("nan")):.0f} mm · {b.worst()}'))
        else:
            out.append((b.tag, f'{b.status}: {b.message}'))
    return out


def _brackets_png(res, ev=None):
    return _mod('brackets').sheet_png(res, getattr(ev, 'p', None)) if any(b.generated for b in res.brackets) \
        else None


_study(id='brackets_full', label='Mount brackets (generated)', page='brackets', menu='Analysis ▸ Mount brackets…',
       cost='study', run=_run_brackets_full, figures=_brackets_full_figures, png=_brackets_png, module='brackets',
       needs=('brackets',), weight=6.0, assumptions=lambda r: _mod('brackets').ASSUMPTIONS)


@_metric('brackets.not_buildable', 'Cameras a printed bracket cannot hold as planned', '', 'brackets', '<=', 0, (),
         'Visible cameras whose bracket cannot be generated at the planned pose: a standoff is needed, no surface '
         'within reach, inside the surface, or no housing face towards it')
def _m_br_not(ev, prm):
    res = ev.study('brackets')
    bad = [b for b in res.brackets if b.status in NOT_BUILT]
    return float(len(bad)), _ev(_brackets_figures(res, ev),
                                '; '.join(f'{b.tag} {b.status}' + (f' ({b.required_mm:.0f} mm needed)'
                                                                   if b.status == 'standoff' else '') for b in bad))


def _built(res, want):
    bs = [b for b in res.brackets if b.generated]
    if want not in (EVERY, None):
        bs = [b for b in bs if b.tag.lower() == str(want).lower()]
        if not bs:
            raise NotEvaluated(f'No bracket was generated for camera {want!r} (see the bracket plan).')
    if not bs:
        raise NotEvaluated('No bracket was generated.')
    return bs


@_metric('brackets.fov_blocked_pct', 'Image blocked by the camera\'s bracket', '%', 'brackets_full', '<=', 2,
         (P_CAMERA,), 'Share of the camera\'s image (within the lens circle) the generated bracket shows in')
def _m_br_fov(ev, prm):
    res = ev.study('brackets_full')
    bs = [b for b in _built(res, prm.get('camera')) if _finite(b.params.get('fov_blocked_pct'))]
    if not bs:
        raise NotEvaluated('The lens model does not resolve: the image was not checked for the bracket.')
    b = max(bs, key=lambda b_: b_.params['fov_blocked_pct'])
    return float(b.params['fov_blocked_pct']), _ev(_brackets_full_figures(res, ev),
                                                   f'{b.tag}: {b.params["fov_blocked_pct"]:.2f} % of its image.')


@_metric('brackets.overhang_mm', 'Bracket lever arm', 'mm', 'brackets_full', '<=', 0, (P_CAMERA,),
         'Distance from a bracket\'s base to the camera housing\'s centre (a long arm shakes)')
def _m_br_lever(ev, prm):
    res = ev.study('brackets_full')
    bs = _built(res, prm.get('camera'))
    b = max(bs, key=lambda b_: b_.params.get('lever_mm', 0.0))
    return float(b.params.get('lever_mm', 0.0)), _ev(_brackets_full_figures(res, ev), '')


# =========================================================================== drive-log check
def _run_drivelog(ev):
    dl = _mod('drivelog')
    st = dl.DriveLogSettings.from_dict(ev.p.analysis.drivelog)
    if not st.path:
        raise NotEvaluated('No drive log checked yet: open a ROS 2 bag or a nuScenes folder on Analysis ▸ Drive-log '
                           'check… (the scorecard checks the log opened there).')
    if not os.path.exists(st.path):
        raise NotEvaluated(f'The drive log {os.path.basename(st.path)!r} is not there any more.')
    res = dl.run(ev.p, None, st, progress=ev.sub_progress('drivelog'), cancel=ev.cancel, engine=ev.engine,
                 variant=ev.v)
    if res is None:
        raise _Cancelled()
    return res


def _drivelog_figures(res, ev):
    m = res.metrics
    out = [('Log', f'{os.path.basename(res.path)} ({res.source}, {res.span["duration_s"]:.1f} s)'),
           ('Streams', f'{m["streams_ok"]} of {m["streams"]} OK' + (f'; missing {", ".join(m["missing"])}'
                                                                    if m['missing'] else '')),
           ('Dropped frames', f'{m["drops"]}' + (f' (worst {m["worst_drop"]}: {100 * m["worst_drop_share"]:.1f} %)'
                                                 if m['worst_drop'] else ''))]
    if m['worst_sync_ms'] is not None:
        out.append(('Worst camera trigger offset', f'{m["worst_sync_tag"]}: {m["worst_sync_ms"]:.2f} ms'))
    if m['lidar_phase_error_deg'] is not None:
        out.append(('Lidar phase error', f'{m["lidar_phase_error_deg"]:.1f}°'))
    if m['worst_rotation_deg'] is not None:
        out.append(('Worst camera rotation', f'{m["worst_rotation_tag"]}: {m["worst_rotation_deg"]:.2f}°'))
    if m['gb_per_hour'] is not None:
        out.append(('Data per hour', f'{m["gb_per_hour"]:.1f} GB/h (plan '
                                     + (f'{m["planned_gb_per_hour"]:.1f})' if m['planned_gb_per_hour'] is not None
                                        else '—)')))
    return out


def _drivelog_png(res, ev=None):
    return _mod('drivelog').timeline_png(res, getattr(ev, 'v', None), width=1100)


_study(id='drivelog', label='Drive-log check', page='drivelog', menu='Analysis ▸ Drive-log check…', cost='study',
       run=_run_drivelog, figures=_drivelog_figures, png=_drivelog_png, module='drivelog', weight=4.0,
       assumptions=lambda r: _mod('drivelog').ASSUMPTIONS)


def _dl_metric(key, label, unit, dp, help_, fn):
    @_metric(f'drivelog.{key}', label, unit, 'drivelog', '<=', dp, (), help_)
    def f(ev, prm):
        res = ev.study('drivelog')
        v, detail = fn(res)
        return float(v), _ev(_drivelog_figures(res, ev), detail)
    return f


def _dl_need(v, why):
    if v is None:
        raise NotEvaluated(why)
    return v


_dl_metric('faulty_streams', 'Faulty streams in the drive log', '', 0,
           'Mapped sensors whose stream is missing or faulty (rate, drops, stamps, size) in the recorded log',
           lambda r: (r.metrics['streams_faulty'], ', '.join(e['tag'] for e in r.streams if e['status'] != 'ok')))
_dl_metric('missing_streams', 'Sensors with no stream in the drive log', '', 0,
           'Planned sensors the recorded log has no topic for',
           lambda r: (len(r.metrics['missing']), ', '.join(r.metrics['missing'])))
_dl_metric('drop_pct', 'Worst share of dropped frames', '%', 2,
           'The largest share of frames a stream drops in the recorded log',
           lambda r: (100.0 * r.metrics['worst_drop_share'], r.metrics['worst_drop'] or ''))
_dl_metric('sync_ms', 'Worst measured camera trigger offset', 'ms', 2,
           'The largest offset of a camera\'s exposure against the lidar sweep measured in the log, beyond the plan',
           lambda r: (_dl_need(r.metrics['worst_sync_ms'], 'No camera\'s sync could be measured in the log (it '
                                                           'needs a spinning lidar\'s sweeps).'),
                      r.metrics['worst_sync_tag']))
_dl_metric('rotation_deg', 'Worst camera rotation found in the log', '°', 3,
           'The largest rotation of a camera against its plan found from lidar–image edge alignment (high or medium '
           'confidence)',
           lambda r: (_dl_need(r.metrics['worst_rotation_deg'], 'No camera rotation could be estimated with '
                                                                'confidence from the log.'),
                      r.metrics['worst_rotation_tag']))
_dl_metric('lidar_phase_deg', 'Lidar phase-lock error in the log', '°', 1,
           'How far the lidar\'s measured phase lock is from the planned phase',
           lambda r: (_dl_need(None if r.metrics['lidar_phase_error_deg'] is None
                               else abs(r.metrics['lidar_phase_error_deg']),
                               'No lidar phase lock could be measured in the log.'), ''))


# =========================================================================== lidar–camera calibration
def _lc_settings(ev):
    lc = _mod('lidar_calib')
    return lc, lc.LidarCalibSettings.from_dict(ev.p.analysis.lidar_calib)


def _run_lidar_plan(ev):
    lc, st = _lc_settings(ev)
    try:
        cam, lid, share = lc.pick_pair(ev.p, ev.v, st, ev.engine)
        poses, notes = lc.plan_poses(ev.p, cam, lid, st, ev.engine)
    except ValueError as e:
        raise NotEvaluated(str(e))
    return {'camera': cam.tag, 'lidar': lid.tag, 'share': share, 'poses': len(poses), 'notes': list(notes),
            'sigma': lc.predicted_sigma(ev.p, cam, lid, poses, st), 'tol': (st.pos_tol_mm, st.ang_tol_deg)}


def _lidar_plan_figures(r, ev):
    sg = r['sigma']
    out = [('Pair', f'camera {r["camera"]} and lidar {r["lidar"]} ({100 * r["share"]:.0f} % of the image in the '
                    f'lidar\'s field)'), ('Board poses planned', str(r['poses']))]
    if sg:
        out += [('Predicted 1σ, position', ' · '.join(f'{k[0]} {sg[k]:.1f}' for k in ('x_mm', 'y_mm', 'z_mm')) + ' mm'),
                ('Predicted 1σ, angles', ' · '.join(f'{k.split("_")[0]} {sg[k]:.3f}'
                                                    for k in ('roll_deg', 'pitch_deg', 'yaw_deg')) + '°'),
                ('Tolerance', f'{r["tol"][0]:g} mm, {r["tol"][1]:g}°')]
    return out


_study(id='lidar_plan', label='Lidar–camera calibration (plan)', page='lidar_calib',
       menu='Calibrate ▸ Lidar–camera calibration…', cost='study', run=_run_lidar_plan, figures=_lidar_plan_figures,
       module='lidar_calib', weight=0.6, assumptions=lambda r: _mod('lidar_calib').ASSUMPTIONS)


def _planned_sigma(ev):
    r = ev.study('lidar_plan')
    if not r['sigma']:
        raise NotEvaluated('No board pose fits the pair\'s common field.')
    return r


@_metric('lidar_calib.planned_sigma_mm', 'Lidar–camera calibration from the board planes alone (position, 1σ)', 'mm',
         'lidar_plan', '<=', 1, (),
         'The worst axis of the lidar position the planned board poses determine from their planes alone (a '
         'prediction; the edges and the checker do much better, so it is a check that the plan constrains the lidar '
         'at all, not the calibration\'s accuracy)')
def _m_lc_psig(ev, prm):
    r = _planned_sigma(ev)
    v = max(r['sigma'][k] for k in ('x_mm', 'y_mm', 'z_mm'))
    return float(v), _ev(_lidar_plan_figures(r, ev), '; '.join(r['notes']))


@_metric('lidar_calib.planned_sigma_deg', 'Lidar–camera calibration precision as planned (angle, 1σ)', '°',
         'lidar_plan', '<=', 3, (), 'The worst angle of the lidar the planned board poses determine (a prediction)')
def _m_lc_psig_deg(ev, prm):
    r = _planned_sigma(ev)
    v = max(r['sigma'][k] for k in ('roll_deg', 'pitch_deg', 'yaw_deg'))
    return float(v), _ev(_lidar_plan_figures(r, ev), '')


def _run_lidar_calib(ev):
    lc, st = _lc_settings(ev)
    if st.source != 'folder' or not st.folder:
        raise NotEvaluated('No captured board data: the scorecard takes a lidar–camera calibration only from real '
                           'captures (Calibrate ▸ Lidar–camera calibration, source: a folder); a synthetic run only '
                           'recovers the error it was given.')
    if not os.path.isdir(st.folder):
        raise NotEvaluated(f'The capture folder {os.path.basename(st.folder)!r} is not there any more.')
    try:
        res = lc.run(ev.p, ev.engine, st, variant=ev.v, progress=ev.sub_progress('lidar_calib'), cancel=ev.cancel)
    except ValueError as e:
        raise NotEvaluated(str(e))
    if res is None:
        raise _Cancelled()
    return res


def _lidar_calib_figures(res, ev):
    out = [('Pair', f'camera {res.camera_tag} and lidar {res.lidar_tag}'), ('Captures', res.source),
           ('Poses used', f'{len(res.used)} of {len(res.poses)}'), ('Verdict', f'{res.verdict}: {res.message}')]
    if res.T is not None:
        d, sg = res.delta, res.sigma
        out += [('Lidar against the plan', f'{d["position_mm"]:.1f} mm (largest axis), {d["rotation_deg"]:.3f}°'),
                ('1σ', f'{sg["position_mm"]:.1f} mm, {sg["angle_deg"]:.3f}°')]
    return out


_study(id='lidar_calib', label='Lidar–camera calibration (measured)', page='lidar_calib',
       menu='Calibrate ▸ Lidar–camera calibration…', cost='study', run=_run_lidar_calib,
       figures=_lidar_calib_figures, module='lidar_calib', weight=4.0,
       assumptions=lambda r: _mod('lidar_calib').ASSUMPTIONS)


def _solved(ev):
    res = ev.study('lidar_calib')
    if res.T is None or res.verdict in ('failed', 'unobservable'):
        raise NotEvaluated(f'The calibration did not determine the lidar: {res.message}')
    return res


@_metric('lidar_calib.delta_mm', 'Measured lidar position against the plan', 'mm', 'lidar_calib', '<=', 1, (),
         'How far the lidar sits from its planned position (largest axis, vehicle frame), measured from the captures')
def _m_lc_dmm(ev, prm):
    res = _solved(ev)
    return float(res.delta['position_mm']), _ev(_lidar_calib_figures(res, ev), res.message)


@_metric('lidar_calib.delta_deg', 'Measured lidar rotation against the plan', '°', 'lidar_calib', '<=', 3, (),
         'The rotation between the lidar\'s planned and measured orientation')
def _m_lc_ddeg(ev, prm):
    res = _solved(ev)
    return float(res.delta['rotation_deg']), _ev(_lidar_calib_figures(res, ev), res.message)


@_metric('lidar_calib.sigma_mm', 'Measured calibration precision (position, 1σ)', 'mm', 'lidar_calib', '<=', 1, (),
         'The 1σ of the measured lidar position (worst axis)')
def _m_lc_smm(ev, prm):
    res = _solved(ev)
    return float(res.sigma['position_mm']), _ev(_lidar_calib_figures(res, ev), '')


# =========================================================================== critical targets, calibration observability
def _run_critical(ev):
    cr = _mod('critical')
    st = cr.settings_of(ev.p)
    if not any(t['enabled'] for t in st.targets):
        raise NotEvaluated('No critical target: add them on Analysis ▸ Critical targets.')
    if not any(c.visible for c in ev.v.cameras):
        raise NotEvaluated('No visible camera in the variant.')
    res = cr.run(ev.p, ev.engine, st, envelopes=False, progress=ev.sub_progress('critical'), cancel=ev.cancel)
    if res is None:
        raise _Cancelled()
    return res


def _critical_figures(res, ev):
    return [(r.name, ('resolved by ' + ', '.join(r.met_by)) if r.met else 'not resolved') for r in res.targets]


_study(id='critical', label='Critical targets', page='critical', menu='Analysis ▸ Critical targets…', cost='study',
       run=_run_critical, figures=_critical_figures, module='critical', weight=0.5,
       assumptions=lambda r: _mod('critical').ASSUMPTIONS)


@_metric('critical.unresolved', 'Critical targets not resolved', '', 'critical', '<=', 0, (),
         'Critical targets no camera (with the lidars, where a target needs points) resolves')
def _m_critical(ev, prm):
    res = ev.study('critical')
    miss = [r.name for r in res.targets if not r.met]
    return float(len(miss)), _ev(_critical_figures(res, ev),
                                 ('Not resolved: ' + ', '.join(miss) + '.') if miss else 'Every target resolved.')


def _run_observability(ev):
    ob = _mod('observability')
    if not ev.p.targets:
        raise NotEvaluated('No calibration targets planned: plan them in Calibrate ▸ Targets.')
    if not any(c.visible for c in ev.v.cameras):
        raise NotEvaluated('No visible camera in the variant.')
    d = ob.settings_of(ev.p).to_dict()
    d.update(suggest=False, lidar_pairs=False, fusion=False)          # the cameras' poses only: quicker
    res = ob.run(ev.p, ev.engine, ob.ObservabilitySettings.from_dict(d), progress=ev.sub_progress('observability'),
                 cancel=ev.cancel)
    if res is None:
        raise _Cancelled()
    return res


def _obs_cams(ev):
    cams = ev.study('observability').cameras
    blind = [c['tag'] for c in cams if c['sigma'] is None]
    if not cams or blind:
        raise NotEvaluated(f'{", ".join(blind)} see no calibration board whole: their poses are not determined.'
                           if blind else 'No visible camera in the variant.')
    return cams


_study(id='observability', label='Calibration observability', page='observability',
       menu='Calibrate ▸ Calibration observability…', cost='study', run=_run_observability,
       figures=lambda r, ev: [(c['tag'], f'{c["ratio"]:.2f}× the requirement ({c["worst_axis"]})') for c in r.cameras
                              if c['sigma'] is not None],
       module='observability', weight=0.5, assumptions=lambda r: _mod('observability').ASSUMPTIONS)


@_metric('observability.worst_ratio', 'Loosest camera calibration against the requirement', '×', 'observability',
         '<=', 2, (), 'The worst camera\'s worst axis (1σ) in units of the requirement')
def _m_obs_ratio(ev, prm):
    c = max(_obs_cams(ev), key=lambda c_: c_['ratio'])
    return float(c['ratio']), _ev([], f'{c["tag"]}: {c["worst_axis"]}')


@_metric('observability.worst_deg', 'Loosest camera angle after calibration (1σ)', '°', 'observability', '<=', 3, (),
         'The largest angle 1σ of any camera the planned boards give')
def _m_obs_deg(ev, prm):
    v, tag = max((max(c['sigma'][k] for k in ('yaw_deg', 'pitch_deg', 'roll_deg')), c['tag']) for c in _obs_cams(ev))
    return float(v), _ev([], tag)


@_metric('observability.worst_mm', 'Loosest camera position after calibration (1σ)', 'mm', 'observability', '<=', 1,
         (), 'The largest position 1σ of any camera the planned boards give')
def _m_obs_mm(ev, prm):
    v, tag = max((max(c['sigma'][k] for k in ('x_mm', 'y_mm', 'z_mm')), c['tag']) for c in _obs_cams(ev))
    return float(v), _ev([], tag)


# =========================================================================== items and settings
def _new_id(taken=()) -> str:
    taken = set(taken)
    n = 1
    while f'R{n}' in taken:
        n += 1
    return f'R{n}'


def _num(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def describe(item, project=None) -> str:
    """A requirement in words from its metric, parameters, operator and target (used when no text is given)."""
    m = METRICS.get(item.get('metric'))
    t = _num(item.get('target'))
    if m is None:
        return f'{item.get("metric")} {OP_TEXT.get(item.get("op"), "?")} {item.get("target")}'
    bits = []
    for p_ in m.params:
        val = (item.get('params') or {}).get(p_.key, p_.default)
        if project is not None:
            lab = dict(p_.options(project)).get(val)
        else:
            lab = None
        lab = lab or (val if val else {'sector': 'every sector', 'cls': 'every class'}.get(p_.key, 'worst'))
        bits.append(lab)
    tgt = '?' if t is None else f'{t:g}'
    return f'{m.label}' + (f' ({", ".join(bits)})' if bits else '') + f' {OP_TEXT.get(item.get("op"), "≥")} {tgt} ' \
                                                                        f'{m.unit}'.rstrip()


def clean_item(d, project=None, taken=()) -> dict | None:
    """A requirement as stored, tolerant of junk: a known operator and priority, a finite target (None when not a
    number), the metric's parameters only (missing ones at their defaults), an id. None when d is not a mapping."""
    if not isinstance(d, dict):
        return None
    mid = str(d.get('metric') or '')[:80]
    m = METRICS.get(mid)
    params = {}
    raw = d.get('params') if isinstance(d.get('params'), dict) else {}
    if m is not None:
        for p_ in m.params:
            v = raw.get(p_.key, p_.default)
            params[p_.key] = '' if v is None else str(v)[:60]
    else:
        params = {str(k)[:40]: str(v)[:60] for k, v in raw.items()}
    op = d.get('op') if d.get('op') in OPS else (m.better if m is not None else '>=')
    out = {'id': str(d.get('id') or '')[:20] or _new_id(taken), 'text': str(d.get('text') or '')[:300].strip(),
           'metric': mid, 'params': params, 'op': op, 'target': _num(d.get('target')),
           'priority': d.get('priority') if d.get('priority') in PRIORITIES else 'must',
           'note': str(d.get('note') or '')[:500]}
    if not out['text'] and m is not None:
        out['text'] = describe(out, project)
    return out


def validate_item(item, project=None) -> tuple[bool, str]:
    """(ok, message): whether a requirement can be evaluated as written: a known metric of this build, parameters
    among the project's choices, an operator, a finite target and a priority."""
    if not isinstance(item, dict):
        return False, 'A requirement is a mapping with metric, op and target.'
    mid = item.get('metric')
    m = METRICS.get(mid)
    if m is None:
        return False, f'Unknown metric {mid!r}. Known: {", ".join(sorted(METRICS))}.'
    if not available(m.study):
        return False, f'The {STUDIES[m.study].label.lower()} study is not part of this build.'
    if item.get('op') not in OPS:
        return False, f'Operator {item.get("op")!r} is not one of {", ".join(OPS)}.'
    if _num(item.get('target')) is None:
        return False, f'Target {item.get("target")!r} is not a finite number.'
    if item.get('priority', 'must') not in PRIORITIES:
        return False, f'Priority {item.get("priority")!r} is not must or should.'
    prm = item.get('params') or {}
    if not isinstance(prm, dict):
        return False, 'params must be a mapping.'
    unknown = [k for k in prm if k not in {p_.key for p_ in m.params}]
    if unknown:
        return False, f'{m.label} takes no parameter {", ".join(map(repr, unknown))}.'
    if project is not None:
        v = project.variant_by_uid(settings_of(project).variant) or project.variant
        for p_ in m.params:
            val = prm.get(p_.key, p_.default)
            opts = [str(k) for k, _ in p_.options(project, v)]
            if str(val) not in opts and str(val).lower() not in [o.lower() for o in opts]:
                return False, (f'{p_.label} {val!r} is not one of the project\'s: '
                               f'{", ".join(repr(o) for o in opts)}.')
    return True, 'OK'


@dataclass
class RequirementsSettings:
    items: list = field(default_factory=list)
    variant: str = ''                   # uid of the variant to evaluate ('' = the active one)
    mc_samples: int = 300               # Monte Carlo tolerance samples
    mc_pos_mm: float = 10.0             # Monte Carlo position error, ± mm (angles: the analysis tolerance)
    title: str = 'Design review'

    def to_dict(self):
        return {'items': [dict(i) for i in self.items],
                'settings': {'variant': self.variant, 'mc_samples': self.mc_samples, 'mc_pos_mm': self.mc_pos_mm,
                             'title': self.title}}

    @staticmethod
    def from_dict(d) -> 'RequirementsSettings':
        s = RequirementsSettings()
        d = d if isinstance(d, dict) else {}
        taken = []
        for raw in d.get('items') or []:
            it = clean_item(raw, taken=taken)
            if it is None:
                continue
            if it['id'] in taken:
                it['id'] = _new_id(taken)
            taken.append(it['id'])
            s.items.append(it)
        st = d.get('settings') if isinstance(d.get('settings'), dict) else {}
        if isinstance(st.get('variant'), str):
            s.variant = st['variant'][:40]
        if isinstance(st.get('title'), str) and st['title'].strip():
            s.title = st['title'].strip()[:120]
        for k, (lo, hi) in LIMITS.items():
            v = _num(st.get(k))
            if v is not None:
                setattr(s, k, type(getattr(s, k))(round(min(max(v, lo), hi)) if k == 'mc_samples'
                                                  else min(max(v, lo), hi)))
        return s


def settings_of(project) -> RequirementsSettings:
    return RequirementsSettings.from_dict(getattr(project.analysis, 'requirements', None))


# =========================================================================== results
@dataclass
class ItemResult:
    item: dict
    status: str                 # pass | fail | not_evaluated | not_run (a study metric not run yet)
    value: float | None
    margin: float | None
    reason: str = ''            # why it was not evaluated
    evidence: dict = field(default_factory=dict)
    stale: bool = False         # from an earlier run on different inputs

    @property
    def metric(self) -> Metric | None:
        return METRICS.get(self.item.get('metric'))


@dataclass
class RequirementsResult:
    settings: RequirementsSettings
    items: list                 # [ItemResult] in the order of the settings
    studies: dict               # study id -> {'label', 'seconds', 'error', 'ran'}
    project: str
    variant: str
    variant_uid: str
    inputs_key: str
    created: str
    issues: list | None = None  # the Issues list of the evaluated variant (when the issues study ran)
    cheap_only: bool = False
    cache: dict = field(default_factory=dict, repr=False)      # study id -> result object (not exported)
    errors: dict = field(default_factory=dict, repr=False)
    revision: int = 0
    evaluated: object = field(default=None, repr=False, compare=False)    # the project as the studies saw it

    def counts(self) -> dict:
        out = {}
        for pr in PRIORITIES:
            rs = [r for r in self.items if r.item['priority'] == pr]
            out[pr] = {'total': len(rs), 'pass': sum(r.status == 'pass' for r in rs),
                       'fail': sum(r.status == 'fail' for r in rs),
                       'not_evaluated': sum(r.status in ('not_evaluated', 'not_run') for r in rs)}
        return out

    @property
    def verdict(self) -> str:
        return verdict_of(self.items)

    def item(self, iid) -> ItemResult | None:
        return next((r for r in self.items if r.item['id'] == iid), None)


def verdict_of(items) -> str:
    musts = [r for r in items if r.item['priority'] == 'must']
    if not items:
        return 'empty'
    if any(r.status == 'fail' for r in musts):
        return 'fail'
    if any(r.status != 'pass' for r in musts):
        return 'incomplete'
    return 'pass'


def compare(value, op, target) -> tuple[bool, float]:
    """(passes, margin): the margin is signed so positive is good."""
    eps = 1e-9 * max(1.0, abs(target))
    if op in ('>=', '>'):
        margin = value - target
        ok = margin >= -eps if op == '>=' else margin > eps
    elif op in ('<=', '<'):
        margin = target - value
        ok = margin >= -eps if op == '<=' else margin > eps
    else:
        margin = -abs(value - target)
        ok = margin >= -eps
    return bool(ok), float(margin)


def _evaluate_item(ev, it, run_studies=True) -> ItemResult:
    m = METRICS.get(it['metric'])
    if m is None:
        return ItemResult(it, 'not_evaluated', None, None, f'Unknown metric {it["metric"]!r}.')
    s = STUDIES[m.study]
    base = {'study': s.id, 'study_label': s.label, 'page': s.page, 'menu': s.menu, 'cost': s.cost,
            'variant': ev.v.name, 'metric_label': m.label, 'unit': m.unit}
    ok, msg = validate_item(it)
    if not ok:
        return ItemResult(it, 'not_evaluated', None, None, msg, dict(base))
    if not run_studies and m.study not in ev.cache and m.study not in ev.errors:
        return ItemResult(it, 'not_run', None, None, f'Run the scorecard: this needs the {s.label.lower()} study.',
                          dict(base))
    try:
        value, evd = m.fn(ev, it['params'])
    except NotEvaluated as e:
        return ItemResult(it, 'not_evaluated', None, None, str(e), dict(base))
    if not _finite(value):
        return ItemResult(it, 'not_evaluated', None, None, f'The {s.label.lower()} study gives no finite value.',
                          dict(base, **evd))
    passed, margin = compare(float(value), it['op'], float(it['target']))
    return ItemResult(it, 'pass' if passed else 'fail', float(value), margin, '', dict(base, **evd))


# =========================================================================== running
def inputs_key(project, variant_uid='') -> str:
    """Everything the studies read, hashed: a result made from other inputs is stale. The requirements themselves and
    the assistant's brief are left out (editing a requirement does not make the studies stale)."""
    a = project.analysis.to_dict()
    for k in ('requirements', 'assistant', 'live', 'heatmap'):
        a.pop(k, None)
    veh = project.vehicle
    m = veh.mesh
    doc = {'variants': [v.to_dict() for v in project.variants], 'active': variant_uid or project.active_uid,
           'analysis': a, 'roi': project.roi.to_dict(), 'rig': [r.to_dict() for r in project.rig],
           'ecu': project.ecu.to_dict(), 'library': project.section('library'),
           'targets': [t.to_dict() for t in project.targets], 'obstacles': [o.to_dict() for o in project.obstacles],
           'fleet': project.fleet, 'frame': [project.frame, list(project.custom_origin), project.units],
           'mesh': [id(m), getattr(m, 'n_tris', 0), veh.occluder_target,
                    [(z.name, z.kind) for z in getattr(m, 'zones', [])] if m is not None else []],
           'wheels': veh.wheel_centres, 'streams': project.streams}
    text = json.dumps(doc, sort_keys=True, default=str)
    return hashlib.sha1(text.encode('utf-8')).hexdigest()[:16]


def _project_for(project, uid):
    """The project with the variant uid active (a shallow copy when it is not the active one), and whether the uid
    was found."""
    if not uid or uid == project.active_uid:
        return project, True
    if project.variant_by_uid(uid) is None:
        return project, False
    q = copy.copy(project)
    q.active_uid = uid
    return q, True


def _plan_studies(items, cache=()):
    """The studies the items need (dependencies first), cheap ones first."""
    need = []

    def add(sid):
        s = STUDIES.get(sid)
        if s is None or sid in need:
            return
        for d in s.needs + s.soft:
            add(d)
        need.append(sid)
    for it in items:
        m = METRICS.get(it.get('metric'))
        if m is not None and available(m.study):
            add(m.study)
    order = sorted(need, key=lambda s: (STUDIES[s].cost != 'cheap', need.index(s)))
    return [s for s in order if s not in cache]


def run(project, engine=None, settings: RequirementsSettings | None = None, variant=None, progress=None, cancel=None,
        reuse: RequirementsResult | None = None, cheap_only=False, issues_list=True, revision=0):
    """Evaluate every requirement against a variant (settings.variant, else the active one; variant overrides both):
    cheap metrics at once, each study a metric needs run once and shared. reuse: an earlier result whose study
    results are used again when the inputs are the same (only the requirements changed). cheap_only: study metrics
    are 'not_run' unless reuse holds their study. issues_list: also compute the Issues list (for the review pack).
    Returns None when cancelled."""
    st = settings or settings_of(project)
    uid = (variant.uid if hasattr(variant, 'uid') else variant) or st.variant
    p, found = _project_for(project, uid)
    key = inputs_key(p, p.active_uid) if (reuse is not None or not cheap_only) else ''
    items = [dict(i) for i in st.items]
    cache, errors = {}, {}
    if reuse is not None and reuse.inputs_key == key and reuse.variant_uid == p.active_uid:
        cache, errors = dict(reuse.cache), dict(reuse.errors)
    plan = [] if cheap_only else _plan_studies(items + ([{'metric': 'issues.count'}] if issues_list else []), cache)
    if cheap_only:
        plan = [s for s in _plan_studies(items, cache) if STUDIES[s].cost == 'cheap']
    ev = _Eval(p, engine, st, progress, cancel, plan)
    ev.cache.update(cache)
    ev.errors.update(errors)
    if cache or errors:                     # reused: keep the time each study took when it ran
        ev.seconds.update({sid: s_['seconds'] for sid, s_ in reuse.studies.items() if sid in cache or sid in errors})
    try:
        for sid in plan:
            if ev.stop():
                return None
            try:
                ev.study(sid)
            except NotEvaluated:
                pass
        out = [_evaluate_item(ev, it, run_studies=not cheap_only) for it in items]
    except _Cancelled:
        return None
    if ev.stop():
        return None
    studies = {}
    for sid in list(ev.cache) + [s for s in ev.errors if s not in ev.cache]:
        s = STUDIES.get(sid)
        if s is None:
            continue
        studies[sid] = {'label': s.label, 'seconds': round(ev.seconds.get(sid, 0.0), 3),
                        'error': ev.errors.get(sid, ''), 'ran': sid in ev.cache, 'cost': s.cost}
    if not found:
        for r in out:
            r.evidence.setdefault('note', 'The chosen variant is gone: the active one was evaluated.')
    if progress is not None:
        try:
            progress(1.0, '')
        except TypeError:
            progress(1.0)
    return RequirementsResult(st, out, studies, getattr(project, 'name', ''), p.variant.name, p.variant.uid, key,
                              _dt.datetime.now().astimezone().replace(microsecond=0).isoformat(),
                              ev.cache.get('issues'), cheap_only, dict(ev.cache), dict(ev.errors), revision, p)


def reevaluate(project, prev: RequirementsResult | None, settings: RequirementsSettings | None = None):
    """The requirements evaluated again at once after an edit: cheap metrics computed, study metrics from prev's
    study results when the inputs are unchanged, else carried over from prev and marked stale (or 'not_run')."""
    st = settings or settings_of(project)
    res = run(project, None, st, cheap_only=True, reuse=prev, issues_list=False)
    if prev is None or res is None:
        return res
    same = prev.inputs_key == res.inputs_key and prev.variant_uid == res.variant_uid
    old = {r.item['id']: r for r in prev.items}
    for i, r in enumerate(res.items):
        o = old.get(r.item['id'])
        if r.status != 'not_run' or o is None or o.status == 'not_run' or not _same_check(o.item, r.item):
            continue
        n = ItemResult(r.item, o.status, o.value, o.margin, o.reason, dict(o.evidence), stale=not same)
        if o.value is not None:
            if validate_item(r.item)[0]:
                ok, n.margin = compare(o.value, r.item['op'], float(r.item['target']))
                n.status = 'pass' if ok else 'fail'
            else:
                n.status, n.margin, n.reason = 'not_evaluated', None, validate_item(r.item)[1]
        res.items[i] = n
    res.cheap_only = not same or any(r.status == 'not_run' for r in res.items)
    return res


def _same_check(a, b) -> bool:
    return a.get('metric') == b.get('metric') and (a.get('params') or {}) == (b.get('params') or {})


# =========================================================================== the plan's own settings and templates
def from_plan(project, variant=None) -> list:
    """Requirements implied by what the plan already sets: the perception range's required range per sector, the
    analysis' minimum density, the recording shift, the timing's blur limit, no critical issue …; each with a note
    saying where its target comes from. Only for what the variant has (no lidar: no lidar requirement)."""
    from .perception import PerceptionSettings, SECTORS
    from .recording import RecordingSettings
    from .timing import TimingSettings
    v = variant or project.variant
    a = project.analysis
    out = []

    def add(metric, params, op, target, priority, text, note):
        if metric_available(metric):
            out.append({'metric': metric, 'params': params, 'op': op, 'target': float(target), 'priority': priority,
                        'text': text, 'note': note})
    add('issues.count', {'severity': 'critical'}, '<=', 0, 'must', 'No critical issue in the Issues list',
        'The Issues list\'s critical findings (blind ROI ground, a floating camera, an overloaded link …).')
    if v.cameras:
        add('coverage.blind_m2', {}, '<=', 0.0005, 'must', 'No blind ground in the ROI (at most 5 cm²)',
            'The Issues list flags blind ROI ground above 5 cm².')
        add('coverage.min_density', {}, '>=', a.min_density, 'should',
            f'Ground resolved at {a.min_density:g} px/cm or better everywhere in the ROI',
            'The analysis\' minimum density (Analysis ▸ Thresholds).')
        pst = PerceptionSettings.from_dict(a.perception)
        for name, _ in SECTORS:
            req = float(pst.required.get(name, 0.0))
            if req > 0:
                add('perception.range', {'cls': EVERY, 'sector': name}, '>=', req, 'must',
                    f'Every object class resolved to {req:g} m in the {name.replace("-", " ")} sector',
                    'The perception range\'s required range for the sector.')
    rst = RecordingSettings.from_dict(a.recording)
    if v.cameras or v.sensors:
        add('recording.hours_per_set', {}, '>=', rst.shift_hours, 'must',
            f'{rst.shift_hours:g} h of driving per disk set', 'The recording budget\'s shift length.')
        add('recording.write_load_pct', {}, '<=', 100.0, 'must',
            f'The disks sustain the data rate with {rst.write_headroom:g}× headroom',
            'The recording budget\'s write headroom; 100 % is the disks\' sustained write.')
    if any(s.visible and s.kind in ('lidar', 'radar') for s in v.sensors):
        add('recording.uplink_load_pct', {}, '<=', rst.link_warn * 100.0, 'should',
            f'Uplink to the recorder below {rst.link_warn * 100:g} %', 'The recording budget\'s link margin.')
        add('sensor_coverage.gaps', {'cls': EVERY}, '<=', 0, 'should',
            'Some sensor type reaches the required range in every sector, for every object',
            'The perception range\'s required ranges, judged for cameras, lidars and radars together.')
    tst = TimingSettings.from_dict(a.timing)
    if v.cameras:
        add('timing.blur_px', {}, '<=', tst.blur_limit_px, 'should',
            f'Motion blur at most {tst.blur_limit_px:g} px at {tst.speed_kmh:g} km/h',
            'The sync and timing plan\'s blur limit.')
    if any(s.visible and s.kind == 'lidar' for s in v.sensors) and v.cameras:
        add('timing.misalignment_ms', {'lines': 'planned'}, '<=', 1.0, 'should',
            'Camera–lidar misalignment at most 1 ms',
            'A typical tolerance: the sync plan has no setting for it, change the target to yours.')
    out += _plan_extras(project, v)
    return out


def _plan_extras(project, v) -> list:
    """Requirements from the settings of the studies added later (power, GNSS and IMU, …), where they exist."""
    out = []
    for fn in _EXTRA_PLAN:
        try:
            out += fn(project, v)
        except Exception:
            pass
    return [o for o in out if metric_available(o['metric'])]


def _plan_lidar_calib(project, v):
    lc = _mod('lidar_calib')
    if lc is None or not v.cameras or not any(s.visible and s.kind == 'lidar' for s in v.sensors):
        return []
    st = lc.LidarCalibSettings.from_dict(project.analysis.lidar_calib)
    return [{'metric': 'lidar_calib.planned_sigma_mm', 'params': {}, 'op': '<=', 'target': 2 * st.pos_tol_mm,
             'priority': 'should', 'text': f'The planned calibration boards\' planes alone pin the lidar to '
                                           f'{2 * st.pos_tol_mm:g} mm (1σ)',
             'note': 'From the board planes alone, a fisheye camera 4–8 m from the boards gives about 15–20 mm; the '
                     'edges and the checker bring the calibration to a few millimetres. A plan above this leaves the '
                     'lidar poorly constrained.'}]


def _plan_drivelog(project, v):
    dl = _mod('drivelog')
    if dl is None:
        return []
    st = dl.DriveLogSettings.from_dict(project.analysis.drivelog)
    if not st.path:
        return []
    return [{'metric': 'drivelog.faulty_streams', 'params': {}, 'op': '<=', 'target': 0.0, 'priority': 'must',
             'text': 'Every stream of the recorded drive healthy', 'note': 'The drive-log check of the open log.'},
            {'metric': 'drivelog.sync_ms', 'params': {}, 'op': '<=', 'target': st.sync_tol_ms, 'priority': 'should',
             'text': f'Cameras within {st.sync_tol_ms:g} ms of the lidar sweep in the recorded drive',
             'note': 'The drive-log check\'s sync tolerance.'},
            {'metric': 'drivelog.rotation_deg', 'params': {}, 'op': '<=', 'target': st.tolerance_deg,
             'priority': 'should', 'text': f'No camera rotated more than {st.tolerance_deg:g}° in the recorded drive',
             'note': 'The drive-log check\'s rotation tolerance.'}]


_EXTRA_PLAN: list = [_plan_power, _plan_gnss, _plan_fleet, _plan_lidar_calib, _plan_drivelog]
TEMPLATES = [
    {'text': 'Pedestrians resolved to 40 m in every sector', 'metric': 'perception.range',
     'params': {'cls': 'Pedestrian', 'sector': EVERY}, 'op': '>=', 'target': 40.0, 'priority': 'must'},
    {'text': 'Cars resolved to 80 m ahead', 'metric': 'perception.range',
     'params': {'cls': 'Car', 'sector': 'front'}, 'op': '>=', 'target': 80.0, 'priority': 'must'},
    {'text': 'Traffic signs resolved to 30 m ahead', 'metric': 'perception.range',
     'params': {'cls': 'Traffic sign', 'sector': 'front'}, 'op': '>=', 'target': 30.0, 'priority': 'should'},
    {'text': 'No blind ground in the ROI', 'metric': 'coverage.blind_m2', 'params': {}, 'op': '<=',
     'target': 0.0005, 'priority': 'must'},
    {'text': 'Ground resolved at 3 px/cm or better', 'metric': 'coverage.min_density', 'params': {}, 'op': '>=',
     'target': 3.0, 'priority': 'should'},
    {'text': 'Half of the ROI seen by two cameras (for stitching and stereo)', 'metric': 'coverage.multi_pct',
     'params': {}, 'op': '>=', 'target': 50.0, 'priority': 'should'},
    {'text': 'No critical issue', 'metric': 'issues.count', 'params': {'severity': 'critical'}, 'op': '<=',
     'target': 0.0, 'priority': 'must'},
    {'text': 'No critical or serious issue', 'metric': 'issues.count', 'params': {'severity': 'serious'}, 'op': '<=',
     'target': 0.0, 'priority': 'should'},
    {'text': 'Camera–lidar misalignment at most 1 ms', 'metric': 'timing.misalignment_ms',
     'params': {'lines': 'planned'}, 'op': '<=', 'target': 1.0, 'priority': 'must'},
    {'text': 'Motion blur at most 1 px at the planned speed', 'metric': 'timing.blur_px', 'params': {}, 'op': '<=',
     'target': 1.0, 'priority': 'should'},
    {'text': '8 h of driving per disk set', 'metric': 'recording.hours_per_set', 'params': {}, 'op': '>=',
     'target': 8.0, 'priority': 'must'},
    {'text': 'Disk write load at most 80 % (headroom included)', 'metric': 'recording.write_load_pct', 'params': {},
     'op': '<=', 'target': 80.0, 'priority': 'should'},
    {'text': 'Uplink load at most 70 %', 'metric': 'recording.uplink_load_pct', 'params': {}, 'op': '<=',
     'target': 70.0, 'priority': 'should'},
    {'text': 'Every sensor link below 80 %', 'metric': 'recording.link_load_pct', 'params': {}, 'op': '<=',
     'target': 80.0, 'priority': 'should'},
    {'text': 'No sector without a sensor reaching the required range', 'metric': 'sensor_coverage.gaps',
     'params': {'cls': EVERY}, 'op': '<=', 'target': 0.0, 'priority': 'must'},
    {'text': 'Cars detected by lidar to 100 m ahead', 'metric': 'sensor_coverage.range',
     'params': {'modality': 'lidar', 'cls': 'Car', 'sector': 'front'}, 'op': '>=', 'target': 100.0,
     'priority': 'should'},
    {'text': 'At most 10 % of the lidar points on the own vehicle', 'metric': 'sensor_coverage.lidar_blocked_pct',
     'params': {'lidar': EVERY}, 'op': '<=', 'target': 10.0, 'priority': 'should'},
    {'text': 'Lidar ground blind zone within 3 m of the body', 'metric': 'sensor_coverage.ground_blind_m',
     'params': {'lidar': EVERY}, 'op': '<=', 'target': 3.0, 'priority': 'should'},
    {'text': 'Lidar points cover 50 % of every camera image at 20 m', 'metric': 'sensor_coverage.overlap_pct',
     'params': {'camera': EVERY}, 'op': '>=', 'target': 50.0, 'priority': 'should'},
    {'text': 'A 0.9 m child is seen everywhere around the vehicle (at most 0.1 m² unseen)',
     'metric': 'volume.undetected_m2', 'params': {'object': 'Child 0.9 m'}, 'op': '<=', 'target': 0.1,
     'priority': 'must'},
    {'text': 'Cameras see everything up to 0.5 m all round', 'metric': 'volume.covered_to_m', 'params': {},
     'op': '>=', 'target': 0.5, 'priority': 'should'},
    {'text': 'A child never vanishes from the stitched view (at most 0.05 m²)', 'metric': 'stitched.vanish_m2',
     'params': {}, 'op': '<=', 'target': 0.05, 'priority': 'should'},
    {'text': 'Robust to one camera off by the mounting tolerance (blind ground at most 0.05 m²)',
     'metric': 'tolerance.single_blind_m2', 'params': {}, 'op': '<=', 'target': 0.05, 'priority': 'should'},
    {'text': 'Blind ground at most 0.05 m² in 95 % of builds (random mounting errors)',
     'metric': 'tolerance.mc_blind_m2', 'params': {'stat': 'p95'}, 'op': '<=', 'target': 0.05,
     'priority': 'should'},
]


def templates(project=None) -> list:
    """The typical data-rig requirements of this build (metrics whose study exists), each valid for the project when
    given (a class or tag the project does not have falls back to every)."""
    out = []
    for t in TEMPLATES + _EXTRA_TEMPLATES:
        if not metric_available(t['metric']):
            continue
        it = dict(t, params=dict(t['params']))
        if project is not None:
            m = METRICS[it['metric']]
            for p_ in m.params:
                opts = [str(k).lower() for k, _ in p_.options(project)]
                if str(it['params'].get(p_.key, p_.default)).lower() not in opts:
                    it['params'][p_.key] = p_.default
        out.append(it)
    return out


_EXTRA_TEMPLATES: list = [
    {'text': '30 min engine-off on the auxiliary battery', 'metric': 'power.runtime_min', 'params': {}, 'op': '>=',
     'target': 30.0, 'priority': 'must'},
    {'text': 'DC-DC converter at most 80 % loaded', 'metric': 'power.dcdc_load_pct', 'params': {}, 'op': '<=',
     'target': 80.0, 'priority': 'should'},
    {'text': 'The alternator carries the rig while driving', 'metric': 'power.alternator_margin_w', 'params': {},
     'op': '>=', 'target': 0.0, 'priority': 'must'},
    {'text': 'Every fused circuit at most 80 % loaded', 'metric': 'power.circuit_load_pct', 'params': {}, 'op': '<=',
     'target': 80.0, 'priority': 'should'},
    {'text': 'Power-cable drops at most 5 %', 'metric': 'power.cable_drop_pct', 'params': {}, 'op': '<=',
     'target': 5.0, 'priority': 'should'},
    {'text': '≥ 90 % sky for each GNSS antenna', 'metric': 'gnss.sky_pct', 'params': {'antenna': EVERY}, 'op': '>=',
     'target': 90.0, 'priority': 'must'},
    {'text': 'Heading error at most 0.2° (1σ)', 'metric': 'gnss.heading_sigma_deg', 'params': {}, 'op': '<=',
     'target': 0.2, 'priority': 'should'},
    {'text': 'Dual-antenna baseline at least 1 m', 'metric': 'gnss.baseline_m', 'params': {}, 'op': '>=',
     'target': 1.0, 'priority': 'should'},
    {'text': 'Every sensor\'s position known to 5 cm (1σ)', 'metric': 'gnss.pose_sigma_mm',
     'params': {'sensor': EVERY}, 'op': '<=', 'target': 50.0, 'priority': 'should'},
    {'text': 'Sun in any camera\'s image at most 5 % of the driving time', 'metric': 'exposure.sun_pct',
     'params': {'camera': EVERY}, 'op': '<=', 'target': 5.0, 'priority': 'should'},
    {'text': 'No lens or lidar window highly exposed to dirt (score below 0.5)', 'metric': 'exposure.dirt_score',
     'params': {'sensor': EVERY}, 'op': '<', 'target': 0.5, 'priority': 'should'},
    {'text': 'Every vehicle of the fleet within the installation tolerance', 'metric': 'fleet.out_of_tolerance',
     'params': {}, 'op': '<=', 'target': 0.0, 'priority': 'should'},
    {'text': 'No sensor in the fleet to re-mount', 'metric': 'fleet.remount', 'params': {}, 'op': '<=',
     'target': 0.0, 'priority': 'must'},
    {'text': 'No sensor drifting out of tolerance in the fleet', 'metric': 'fleet.drifting', 'params': {},
     'op': '<=', 'target': 0.0, 'priority': 'should'},
    {'text': 'Every camera can be held by a printed bracket at its planned pose', 'metric': 'brackets.not_buildable',
     'params': {}, 'op': '<=', 'target': 0.0, 'priority': 'should'},
    {'text': 'No bracket shows in a camera image', 'metric': 'brackets.fov_blocked_pct', 'params': {'camera': EVERY},
     'op': '<=', 'target': 0.0, 'priority': 'should'},
    {'text': 'Every stream of the test drive healthy', 'metric': 'drivelog.faulty_streams', 'params': {}, 'op': '<=',
     'target': 0.0, 'priority': 'must'},
    {'text': 'At most 0.1 % of frames dropped in the test drive', 'metric': 'drivelog.drop_pct', 'params': {},
     'op': '<=', 'target': 0.1, 'priority': 'should'},
    {'text': 'Cameras within 1 ms of the lidar sweep in the test drive', 'metric': 'drivelog.sync_ms', 'params': {},
     'op': '<=', 'target': 1.0, 'priority': 'should'},
    {'text': 'No camera rotated more than 0.3° from the plan in the test drive', 'metric': 'drivelog.rotation_deg',
     'params': {}, 'op': '<=', 'target': 0.3, 'priority': 'should'},
    {'text': 'The planned board poses\' planes alone pin the lidar to 20 mm (1σ)',
     'metric': 'lidar_calib.planned_sigma_mm', 'params': {}, 'op': '<=', 'target': 20.0, 'priority': 'should'},
    {'text': 'The lidar within 10 mm of its planned position (measured)', 'metric': 'lidar_calib.delta_mm',
     'params': {}, 'op': '<=', 'target': 10.0, 'priority': 'must'},
    {'text': 'The lidar within 0.3° of its planned orientation (measured)', 'metric': 'lidar_calib.delta_deg',
     'params': {}, 'op': '<=', 'target': 0.3, 'priority': 'must'},
]


# =========================================================================== the brief
_NUM = r'(\d+(?:[.,]\d+)?)'
_UNITS = [
    ('m3', r'(?:m³|m3|cubic\s+met(?:re|er)s?)'),
    ('m2', r'(?:m²|m2|sq\.?\s*m|square\s+met(?:re|er)s?)'),
    ('pxcm', r'(?:px\s*/\s*cm|pixels?\s+per\s+(?:cm|centimet(?:re|er)))'),
    ('kmh', r'(?:km\s*/\s*h|kph|kmh)\b'),
    ('km', r'(?:km|kilomet(?:re|er)s?)\b'),
    ('ms', r'(?:ms|msec|millisec(?:ond)?s?)\b'),
    ('min', r'(?:min|mins|minutes?)\b'),
    ('h', r'(?:h|hrs?|hours?)\b'),
    ('s', r'(?:s|secs?|seconds?)\b'),
    ('pct', r'(?:%|per\s?cent)'),
    ('deg', r'(?:°|deg|degrees?)\b'),
    ('px', r'(?:px|pixels?)\b'),
    ('cm', r'(?:cm|centimet(?:re|er)s?)\b'),
    ('mm', r'(?:mm|millimet(?:re|er)s?)\b'),
    ('m', r'(?:m|met(?:re|er)s?)\b'),
    ('w', r'(?:w|watts?)\b'),
]
_QTY = re.compile(_NUM + r'\s*(' + '|'.join(f'(?P<{k}>{p})' for k, p in _UNITS) + ')', re.I)
_SHOULD = re.compile(r'\b(should|ideally|preferably|desirable|nice to have|would like|aim for|target(?:ing)?)\b', re.I)
_ALIASES = {'pedestrian': ('pedestrians?', 'people', 'persons?', 'walkers?'),
            'car': ('cars?', 'vehicles?'),
            'cyclist': ('cyclists?', 'bikes?', 'bicycles?', 'riders?'),
            'traffic sign': ('traffic\\s+signs?', 'road\\s+signs?', 'signs?')}
_SECTOR_WORDS = [
    ('front-left', r'front[\s-]+left'), ('front-right', r'front[\s-]+right'), ('rear-left', r'rear[\s-]+left'),
    ('rear-right', r'rear[\s-]+right'), ('front', r'\b(?:front|ahead|forwards?)\b'),
    ('rear', r'\b(?:rear|behind|backwards?|back)\b'), ('left', r'\bleft\b'), ('right', r'\bright\b'),
    ('sides', r'\b(?:sides?|lateral(?:ly)?)\b')]


def _quantities(text):
    """[(value in the unit's base, unit key, start, end)]: m, m2, m3, ms (from s, ms), min (from min, h for runtimes
    the caller converts), h, pct, pxcm, deg, px, mm, w."""
    out = []
    for mt in _QTY.finditer(text):
        v = float(mt.group(1).replace(',', '.'))
        unit = next(k for k, _ in _UNITS if mt.group(k))
        if unit == 'km':
            v, unit = round(v * 1000.0, 6), 'm'
        elif unit == 'cm':
            v, unit = round(v / 100.0, 9), 'm'
        elif unit == 's':
            v, unit = round(v * 1000.0, 6), 'ms'
        out.append((v, unit, mt.start(), mt.end()))
    return out


def _classes_in(text, names):
    found = []
    low = text.lower()
    for n in names:
        pats = _ALIASES.get(n.lower(), ()) + (re.escape(n.lower()) + r'(?:s|es)?',)
        hit = None
        for pt in pats:
            mt = re.search(r'\b' + pt + r'\b', low)
            if mt and (hit is None or mt.start() < hit):
                hit = mt.start()
        if hit is not None:
            found.append((hit, n))
    # 'traffic signs' also matches 'signs': keep each class once, in the order they are written
    return [n for _, n in sorted(found)]


def _sectors_in(text):
    low = text.lower()
    if re.search(r'\b(every|each|all)\s+(sector|direction)s?\b|all\s+(?:a)?round|360|around\s+the\s+(?:car|vehicle)',
                 low):
        return [EVERY]
    out, used = [], low
    for name, pt in _SECTOR_WORDS:
        if re.search(pt, used):
            out += ['left', 'right'] if name == 'sides' else [name]
            used = re.sub(pt, ' ', used)
    return out or [EVERY]


def _op_from(text, natural):
    low = text.lower()
    strict_lt = re.search(r'\b(less than|under|below|fewer than)\b|<(?!=)', low)
    strict_gt = re.search(r'\b(more than|over|above|greater than)\b|>(?!=)', low)
    if natural == '<=' and strict_lt and not re.search(r'≤|<=|at most|no more than|within|max', low):
        return '<'
    if natural == '>=' and strict_gt and not re.search(r'≥|>=|at least|min', low):
        return '>'
    return natural


def parse_brief(text, project=None) -> list:
    """Requirements proposed from free text (the project brief): simple phrasings with a number and a unit —
    "pedestrians to 40 m", "8 hours per disk", "≤ 1 ms sync", "30 min engine off", "≥ 90 % sky for each GNSS
    antenna", "3 px/cm", "no blind ground", "no critical issues" … Each proposal: {'item', 'source' (the phrase),
    'metric_label'}; nothing is added until the user accepts. Only metrics of this build are proposed."""
    if not isinstance(text, str) or not text.strip():
        return []
    from .perception import PerceptionSettings, DEFAULT_CLASSES
    pnames = [c['name'] for c in (PerceptionSettings.from_dict(project.analysis.perception).classes
                                  if project is not None else DEFAULT_CLASSES)]
    out = []
    parts = []
    for sent in re.split(r'[;\n•]+|\.(?!\d)', text):
        parts += [f.strip() for f in re.split(r',|\band\b|\bwith\b', sent) if f.strip()]
    carry = ''
    for frag in parts:
        phrase = f'{carry} {frag}'.strip() if carry else frag
        got = _parse_phrase(phrase, pnames, project)
        if got:
            out += got
            carry = ''
        elif not _quantities(frag) and len(phrase) < 160:
            carry = phrase                 # 'pedestrians, cyclists to 30 m': the class list runs into the number
        else:
            carry = ''
    seen, uniq = set(), []
    for o in out:
        it = o['item']
        k = (it['metric'], json.dumps(it['params'], sort_keys=True), it['op'], it['target'])
        if k not in seen:
            seen.add(k)
            uniq.append(o)
    return uniq


def _parse_phrase(phrase, pnames, project) -> list:
    low = phrase.lower()
    qs = _quantities(phrase)
    prio = 'should' if _SHOULD.search(phrase) else 'must'
    src = phrase.strip()
    text = src[0].upper() + src[1:] if src else src
    props = []

    def prop(metric, params, target, op=None):
        if not metric_available(metric):
            return
        m = METRICS[metric]
        it = clean_item({'metric': metric, 'params': params, 'op': op or _op_from(phrase, m.better),
                         'target': target, 'priority': prio, 'text': text,
                         'note': 'Proposed from the project brief.'}, project)
        if it is not None and it['target'] is not None:
            props.append({'item': it, 'source': src, 'metric_label': m.label})

    def q(*units):
        return next(((v, u) for v, u, _, _ in qs if u in units), None)
    # ---- ranges: an object class and a length
    lng = q('m')
    cls = _classes_in(phrase, pnames)
    if lng and cls and not re.search(r'\bblind\b|\bwithin\b.*\bbody\b', low):
        mod = 'lidar' if re.search(r'\blidars?\b', low) else 'radar' if re.search(r'\bradars?\b', low) else None
        for c in cls:
            for sec in _sectors_in(phrase):
                if mod:
                    prop('sensor_coverage.range', {'modality': mod, 'cls': c, 'sector': sec}, lng[0])
                else:
                    prop('perception.range', {'cls': c, 'sector': sec}, lng[0])
        return props
    # ---- recording: hours per disk set
    hrs = q('h')
    if hrs and re.search(r'\b(disks?|drives?|ssds?|storage|recorder|disk\s*sets?|media)\b', low):
        prop('recording.hours_per_set', {}, hrs[0])
        return props
    # ---- engine-off runtime (power budget)
    if re.search(r'engine[\s-]*off|parked|on (?:the )?battery|battery runtime|without the engine', low):
        t = q('min') or ((q('h')[0] * 60.0, 'min') if q('h') else None)
        if t:
            prop('power.runtime_min', {}, t[0])
            return props
    # ---- sync
    ms = q('ms')
    if ms and re.search(r'sync|misalign|trigger|lidar|timestamp|aligned|alignment|skew', low):
        prop('timing.misalignment_ms', {'lines': 'planned'}, ms[0])
        return props
    # ---- percentages
    pct = q('pct')
    if pct:
        if re.search(r'\bsky\b', low):
            prop('gnss.sky_pct', {'antenna': EVERY}, pct[0])
        elif re.search(r'\buplink\b', low):
            prop('recording.uplink_load_pct', {}, pct[0])
        elif re.search(r'\b(write|disk)\b', low):
            prop('recording.write_load_pct', {}, pct[0])
        elif re.search(r'\blidar\b', low) and re.search(r'\b(vehicle|body|roof|blocked|own|self)\b', low):
            prop('sensor_coverage.lidar_blocked_pct', {'lidar': EVERY}, pct[0])
        elif re.search(r'\b(two|2|several|multiple)\s+cameras\b|overlap|redundan', low):
            prop('coverage.multi_pct', {}, pct[0])
        elif re.search(r'\b(cover(?:ed|age)?|seen)\b', low) and re.search(r'\b(ground|roi|area)\b', low):
            prop('coverage.covered_pct', {}, pct[0])
        elif re.search(r'\bsun\b|glare', low):
            prop('exposure.sun_pct', {'camera': EVERY}, pct[0])
        elif re.search(r'dc[\s-]*dc|converter', low):
            prop('power.dcdc_load_pct', {}, pct[0])
        return props
    # ---- densities, blur, heading, areas, volumes
    d = q('pxcm')
    if d:
        prop('coverage.min_density', {}, d[0])
        return props
    px = q('px')
    if px and re.search(r'\bblur', low):
        prop('timing.blur_px', {}, px[0])
        return props
    deg = q('deg')
    if deg and re.search(r'\bheading\b', low):
        prop('gnss.heading_sigma_deg', {}, deg[0])
        return props
    m3 = q('m3')
    if m3 and re.search(r'\bblind\b', low):
        prop('volume.blind_m3', {}, m3[0])
        return props
    m2 = q('m2')
    if m2 and re.search(r'\bblind\b', low):
        prop('coverage.blind_m2', {}, m2[0])
        return props
    if lng and re.search(r'\blidar\b', low) and re.search(r'\bblind\b', low):
        prop('sensor_coverage.ground_blind_m', {'lidar': EVERY}, lng[0])
        return props
    # ---- no number
    if not qs:
        if re.search(r'\bno\s+blind\s+(ground|spots?|areas?|zones?)\b', low):
            prop('coverage.blind_m2', {}, 0.0005)
        elif re.search(r'\bno\s+critical\s+(issues?|findings?|problems?)\b', low):
            prop('issues.count', {'severity': 'critical'}, 0.0)
        elif re.search(r'\bno\s+(serious|major)\s+(issues?|findings?|problems?)\b', low):
            prop('issues.count', {'severity': 'serious'}, 0.0)
    return props


# =========================================================================== the Issues list
def issues(project, vg=None) -> list:
    """A must requirement whose metric is cheap (no geometry) and fails: a serious issue. Study metrics are not run
    from here (they are checked when the scorecard runs)."""
    from .rules import Issue
    st = settings_of(project)
    if not st.items:
        return []
    cheap = [i for i in st.items if i['priority'] == 'must' and i['metric'] in METRICS
             and STUDIES[METRICS[i['metric']].study].cost == 'cheap']
    if not cheap:
        return []
    st2 = copy.copy(st)
    st2.items = cheap
    res = run(project, None, st2, cheap_only=True, issues_list=False)
    out = []
    for r in res.items if res is not None else []:
        if r.status != 'fail':
            continue
        m = r.metric
        out.append(Issue('serious', f'Requirement not met: {r.item["text"]} ({fmt_value(r.value, m)} against '
                                    f'{OP_TEXT[r.item["op"]]} {fmt_value(r.item["target"], m)}).',
                         f'Requirements · {r.item["id"]}', [], 'requirement'))
    return out


# =========================================================================== output
def fmt_value(v, m: Metric | None, signed=False):
    if v is None or not _finite(v):
        return '—'
    dp = m.decimals if m is not None else 2
    s = f'{v:+,.{dp}f}' if signed else f'{v:,.{dp}f}'
    if abs(v) < 10 ** -dp and v != 0:                    # a small target (0.0005 m²): its digits, no trailing zeros
        s = (f'{v:+.{dp + 3}f}' if signed else f'{v:.{dp + 3}f}').rstrip('0')
    return s + (f' {m.unit}' if m is not None and m.unit else '')


def summary(res: RequirementsResult) -> dict:
    """Compact figures (the assistant, the batch log)."""
    c = res.counts()
    return {'verdict': res.verdict, 'variant': res.variant,
            'musts_passed': f'{c["must"]["pass"]}/{c["must"]["total"]}',
            'shoulds_passed': f'{c["should"]["pass"]}/{c["should"]["total"]}',
            'not_evaluated': sum(r.status in ('not_evaluated', 'not_run') for r in res.items),
            'items': [{'id': r.item['id'], 'text': r.item['text'], 'priority': r.item['priority'],
                       'metric': r.item['metric'], 'target': f'{OP_TEXT[r.item["op"]]} '
                                                             f'{fmt_value(r.item["target"], r.metric)}',
                       'value': fmt_value(r.value, r.metric), 'status': r.status,
                       **({'reason': r.reason} if r.reason else {}), **({'stale': True} if r.stale else {})}
                      for r in res.items]}


def _round(v, dp=6):
    return None if v is None or not _finite(v) else round(float(v), dp)


def to_doc(res: RequirementsResult, project=None) -> dict:
    """The JSON export: every requirement with its value, margin, status and evidence; the verdict; the studies."""
    c = res.counts()
    return {
        'schema': SCHEMA,
        'project': res.project or getattr(project, 'name', ''), 'variant': res.variant,
        'created': res.created, 'frame': 'vehicle (ISO 8855, rear axle)',
        'verdict': res.verdict, 'verdict_text': VERDICT_TEXT[res.verdict],
        'counts': c, 'exit_code': EXIT_FAIL if res.verdict in ('fail', 'incomplete') else 0,
        'margin': 'signed so positive is good: value − target for ≥ and >, target − value for ≤ and <, −|value − '
                  'target| for =',
        'inputs_key': res.inputs_key,
        'requirements': [{
            'id': r.item['id'], 'text': r.item['text'], 'priority': r.item['priority'], 'metric': r.item['metric'],
            'metric_label': r.metric.label if r.metric else r.item['metric'],
            'params': dict(r.item['params']), 'op': r.item['op'], 'target': _round(r.item['target']),
            'unit': r.metric.unit if r.metric else '', 'value': _round(r.value), 'margin': _round(r.margin),
            'status': r.status, 'reason': r.reason, 'stale': r.stale, 'note': r.item.get('note', ''),
            'evidence': {k: v for k, v in r.evidence.items() if k not in ('metric_label', 'unit')}}
            for r in res.items],
        'studies': [{'id': sid, **s, 'figures': _figures_of(res, sid, project)} for sid, s in res.studies.items()],
        'issues': [{'severity': i.severity, 'message': i.message, 'objects': i.objects} for i in (res.issues or [])],
        'assumptions': ASSUMPTIONS, 'limits': LIMITS_TEXT,
        'study_assumptions': {sid: _assumptions_of(res, sid) for sid in res.studies if _assumptions_of(res, sid)},
    }


def _figures_of(res, sid, project=None) -> list:
    s = STUDIES.get(sid)
    obj = res.cache.get(sid)
    if s is None or obj is None:
        return []
    ev = _ReadCtx(res, project)
    try:
        return [list(f) for f in s.figures(obj, ev)]
    except Exception:
        return []


def _assumptions_of(res, sid) -> str:
    s = STUDIES.get(sid)
    if s is None or s.assumptions is None or sid not in res.cache:
        return ''
    try:
        return str(s.assumptions(res.cache[sid]))
    except Exception:
        return ''


class _ReadCtx:
    """What a study's figures function reads (the project as evaluated, the settings) after a run."""

    def __init__(self, res, project=None):
        self.st = res.settings
        self._res = res
        self._project = project

    @property
    def p(self):
        if self._res.evaluated is not None:
            return self._res.evaluated
        if self._project is None:
            raise AttributeError('no project')
        return _project_for(self._project, self._res.variant_uid)[0]

    @property
    def v(self):
        return self.p.variant


CSV_COLUMNS = ('id', 'priority', 'requirement', 'metric', 'params', 'op', 'target', 'unit', 'value', 'margin', 'status',
               'reason', 'study', 'variant')


def to_csv(res: RequirementsResult) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator='\n')
    w.writerow(CSV_COLUMNS)
    for r in res.items:
        m = r.metric
        w.writerow([r.item['id'], r.item['priority'], r.item['text'], r.item['metric'],
                    ';'.join(f'{k}={v}' for k, v in r.item['params'].items()), r.item['op'],
                    '' if r.item['target'] is None else f'{r.item["target"]:g}', m.unit if m else '',
                    '' if r.value is None else f'{r.value:.6g}', '' if r.margin is None else f'{r.margin:.6g}',
                    r.status + (' (stale)' if r.stale else ''), r.reason, r.evidence.get('study_label', ''),
                    res.variant])
    return buf.getvalue()


# =========================================================================== the design-review pack (PDF)
_CSS = """
body { font-family: '%(sans)s'; font-size: 9pt; color: #1b1b1a; }
h1 { font-size: 22pt; font-weight: 600; margin-top: 0px; margin-bottom: 4px; }
h2 { font-size: 13pt; font-weight: 600; margin-top: 0px; margin-bottom: 6px; }
h3 { font-size: 10pt; font-weight: 600; margin-top: 12px; margin-bottom: 4px; }
p { margin-top: 0px; margin-bottom: 4px; }
.kicker { color: #898781; font-size: 8pt; }
.sub { color: #52514e; font-size: 11pt; }
.note { color: #52514e; }
.muted { color: #898781; }
table.t { border-collapse: collapse; }
table.t th { background-color: #f3f2ef; color: #52514e; font-weight: 600; font-size: 7.5pt; text-align: left;
             border-bottom: 1px solid #c3c2b7; }
table.t td { font-size: 8pt; border-bottom: 1px solid #e6e5df; }
td.n { font-family: '%(mono)s'; font-size: 7.5pt; text-align: right; }
td.ml { color: #898781; }
"""
STATUS_STYLE = {'pass': ('PASS', '#0ca30c', '#ffffff'), 'fail': ('FAIL', '#d03b3b', '#ffffff'),
                'not_evaluated': ('NOT EVALUATED', '#898781', '#ffffff'), 'not_run': ('NOT RUN', '#898781', '#ffffff')}
VERDICT_STYLE = {'pass': '#0ca30c', 'fail': '#d03b3b', 'incomplete': '#c98500', 'empty': '#898781'}


def _pack_html(res: RequirementsResult, project, img_src) -> tuple[str, list]:
    """The pack as QTextDocument HTML and its sections [(title, kind)] in order (each from a new page)."""
    from .exporters import SEVERITY_STYLE, _families
    sans, mono = _families()
    E = lambda s: html.escape(str(s), quote=True)        # noqa: E731
    c = res.counts()
    out, sections = [], []
    w = out.append
    brk = '<p style="page-break-before: always"></p>'

    def pill(status):
        lab, bg, fg = STATUS_STYLE.get(status, (status.upper(), '#898781', '#ffffff'))
        return (f'<span style="background-color:{bg}; color:{fg}; font-size:7pt; font-weight:600">'
                f'&nbsp;{lab}&nbsp;</span>')

    def table(heads):
        row = ''.join(f'<th align="right" nowrap>{E(h)}</th>' if n else f'<th align="left">{E(h)}</th>'
                      for h, n in heads)
        return f'<table class="t" width="100%" cellspacing="0" cellpadding="4"><thead><tr>{row}</tr></thead>\n'

    def num(v):
        return f'<td class="n" align="right" nowrap>{E(v)}</td>'

    def figures_table(figs):
        if not figs:
            return
        w('<table class="t" width="100%" cellspacing="0" cellpadding="3">')
        for lab, val in figs:
            w(f'<tr><td class="ml" width="42%">{E(lab)}</td><td>{E(val)}</td></tr>')
        w('</table>\n')

    var = project.variant_by_uid(res.variant_uid) if project is not None else None
    veh = getattr(getattr(project, 'vehicle', None), 'name', '') or ''
    title = res.settings.title or 'Design review'
    # ---- title page
    sections.append((title, 'title'))
    w(f'<html><head><style>{_CSS % {"sans": sans, "mono": mono}}</style></head><body>\n')
    w(f'<p class="kicker">VANTAGE · {E(title.upper())} PACK</p>\n<h1>{E(res.project or "Untitled project")}</h1>\n')
    w(f'<p class="sub">{E(res.variant)}' + (f' · {E(veh)}' if veh else '') + '</p>\n')
    if var is not None and var.note:
        w(f'<p class="note">{E(var.note)}</p>\n')
    col = VERDICT_STYLE[res.verdict]
    w('<p>&nbsp;</p><table cellspacing="0" cellpadding="10" width="100%"><tr>'
      f'<td style="background-color:{col}; color:#ffffff" width="34%">'
      f'<span style="font-size:8pt">VERDICT</span><br><span style="font-size:24pt; font-weight:600">'
      f'{E(VERDICT_TEXT[res.verdict].upper())}</span></td>'
      f'<td style="background-color:#f3f2ef" width="33%"><span class="muted" style="font-size:8pt">MUST '
      f'REQUIREMENTS PASSED</span><br><span style="font-size:20pt; font-weight:600">{c["must"]["pass"]} / '
      f'{c["must"]["total"]}</span></td>'
      f'<td style="background-color:#f3f2ef" width="33%"><span class="muted" style="font-size:8pt">SHOULD '
      f'REQUIREMENTS PASSED</span><br><span style="font-size:20pt; font-weight:600">{c["should"]["pass"]} / '
      f'{c["should"]["total"]}</span></td></tr></table>\n')
    fails = [r for r in res.items if r.status == 'fail' and r.item['priority'] == 'must']
    nev = [r for r in res.items if r.status in ('not_evaluated', 'not_run') and r.item['priority'] == 'must']
    w('<p>&nbsp;</p>')
    if fails:
        w('<p><b>Must requirements that fail:</b> ' + '; '.join(f'{E(r.item["id"])} {E(r.item["text"])}'
                                                                 for r in fails) + '.</p>\n')
    if nev:
        w('<p><b>Must requirements not evaluated:</b> ' + '; '.join(f'{E(r.item["id"])} {E(r.item["text"])}'
                                                                    for r in nev) + '.</p>\n')
    meta = [('Project', res.project), ('Variant', res.variant), ('Vehicle', veh or '—'),
            ('Date', res.created.replace('T', ' ')), ('Requirements', str(len(res.items))),
            ('Studies', ', '.join(s['label'] for s in res.studies.values() if s['ran']) or '—'),
            ('Inputs', res.inputs_key), ('Made with', _app_version())]
    w('<p>&nbsp;</p><table cellspacing="0" cellpadding="3" width="100%">')
    for k, val in meta:
        w(f'<tr><td class="ml" width="18%">{E(k)}</td><td>{E(val)}</td></tr>')
    w('</table>\n')
    stale = [r for r in res.items if r.stale]
    if stale:
        w(f'<p class="note">{len(stale)} requirement(s) show values from an earlier run on a different design; '
          'run the scorecard again before signing off.</p>\n')
    # ---- the scorecard
    w(brk)
    sections.append(('Scorecard', 'scorecard'))
    w('<h2>Scorecard</h2>\n<p class="note">Margin: how far the value is from the target, positive when the '
      'requirement is met with room to spare.</p>\n')
    w(table([('#', False), ('Requirement', False), ('Priority', False), ('Target', True), ('Value', True),
             ('Margin', True), ('Status', False)]))
    for r in res.items:
        m = r.metric
        val = fmt_value(r.value, m) + (' *' if r.stale else '')
        w(f'<tr><td nowrap>{E(r.item["id"])}</td><td>{E(r.item["text"])}<br><span class="muted">'
          f'{E(m.label if m else r.item["metric"])}' + (f' · {E(_params_text(r.item, project))}'
                                                         if r.item['params'] else '') + '</span>'
          + (f'<br><span class="muted">{E(r.reason)}</span>' if r.reason else '') + '</td>'
          f'<td nowrap>{E(r.item["priority"])}</td>' + num(f'{OP_TEXT[r.item["op"]]} '
                                                          f'{fmt_value(r.item["target"], m)}')
          + num(val) + num(fmt_value(r.margin, m, signed=True)) + f'<td nowrap>{pill(r.status)}</td></tr>\n')
    w('</table>\n')
    if stale:
        w('<p class="muted">* from an earlier run on a different design (stale).</p>\n')
    # ---- one section per study used
    for sid, s in res.studies.items():
        st = STUDIES.get(sid)
        if st is None or sid == 'issues':
            continue
        w(brk)
        sections.append((st.label, 'study'))
        w(f'<h2>{E(st.label)}</h2>\n<p class="muted">{E(st.menu)} · {E(res.variant)}'
          + (f' · {s["seconds"]:.1f} s' if s.get('seconds') else '') + '</p>\n')
        users = [r for r in res.items if r.evidence.get('study') == sid]
        if users:
            w('<p>Used by ' + ', '.join(f'{E(r.item["id"])} {pill(r.status)}' for r in users) + '</p>\n')
        if s.get('error'):
            w(f'<p class="note">Not evaluated: {E(s["error"])}</p>\n')
            continue
        figures_table(_figures_of(res, sid, project))
        for r in users:
            det = r.evidence.get('detail')
            if det:
                w(f'<p><b>{E(r.item["id"])}</b> {E(det)}</p>\n')
        if st.png is not None and sid in res.cache:
            try:
                data = st.png(res.cache[sid], _ReadCtx(res, project))
            except Exception:
                data = None
            if data:
                name, iw, ih = img_src(sid, data)
                if iw:
                    width = min(560, iw)
                    hgt = int(round(width * ih / iw))
                    if hgt > 500:                  # with the figures above it the plot stays on the study's page
                        width, hgt = int(round(width * 500 / hgt)), 500
                    w(f'<p>&nbsp;</p><p align="center"><img src="{E(name)}" width="{width}" height="{hgt}"></p>\n')
                    if st.caption:
                        w(f'<p class="muted" align="center">{E(st.caption)}</p>\n')
        txt = _assumptions_of(res, sid)
        if txt:
            w(f'<p class="muted">{E(txt)}</p>\n')
    # ---- the Issues list
    w(brk)
    sections.append(('Issues', 'issues'))
    w('<h2>Issues</h2>\n')
    if res.issues is None:
        w('<p class="muted">The Issues list was not computed in this run.</p>\n')
    elif not res.issues:
        w('<p>The Issues list is empty.</p>\n')
    else:
        w(table([('Severity', False), ('Issue', False), ('Where', False)]))
        for i in res.issues:
            lab, bg, fg = SEVERITY_STYLE.get(i.severity, (i.severity.upper(), '#898781', '#ffffff'))
            w(f'<tr><td nowrap><span style="background-color:{bg}; color:{fg}; font-size:7pt; font-weight:600">'
              f'&nbsp;{lab}&nbsp;</span></td><td>{E(i.message)}</td><td class="muted">{E(i.objects)}</td></tr>\n')
        w('</table>\n')
    # ---- assumptions and limits
    w('<h3>Assumptions and limits</h3>\n')
    w(f'<p>{E(ASSUMPTIONS)}</p>\n<p>{E(LIMITS_TEXT)}</p>\n')
    for sid in res.studies:
        txt = _assumptions_of(res, sid)
        if txt and sid in STUDIES:
            w(f'<p><b>{E(STUDIES[sid].label)}.</b> {E(txt)}</p>\n')
    w('</body></html>\n')
    return ''.join(out), sections


def _params_text(item, project=None) -> str:
    m = METRICS.get(item.get('metric'))
    if m is None:
        return ', '.join(f'{k} {v}' for k, v in (item.get('params') or {}).items())
    bits = []
    for p_ in m.params:
        val = item['params'].get(p_.key, p_.default)
        lab = dict(p_.options(project)).get(val) if project is not None else None
        bits.append(f'{p_.label.lower()} {lab or val or "every"}')
    return ', '.join(bits)


def _app_version() -> str:
    try:
        from .. import __version__
        return f'Vantage {__version__}'
    except Exception:
        return 'Vantage'


def review_pack(res: RequirementsResult, project=None) -> tuple[bytes, int, list]:
    """The design-review pack as PDF bytes: (data, pages, sections). Needs a running QGuiApplication (fonts and
    painting), as the app's other PDFs. The requirement texts and the verdict are also written into the PDF's XMP
    metadata (searchable without a PDF text extractor)."""
    from .exporters import HAIR, MUTED, _PdfOut, _require_app, _Sheet, _families
    _require_app()
    from PyQt6.QtCore import QRectF, QSizeF, QUrl
    from PyQt6.QtGui import QFont, QImage, QTextDocument
    sans, _ = _families()
    doc = QTextDocument()
    dev = QImage(8, 8, QImage.Format.Format_RGB32)
    dev.setDotsPerMeterX(3780)
    dev.setDotsPerMeterY(3780)
    doc.documentLayout().setPaintDevice(dev)
    f = QFont(sans)
    f.setPointSizeF(9)
    doc.setDefaultFont(f)
    n_img = [0]

    def src(key, data):
        img = QImage.fromData(data)
        if img.isNull():
            return '', 0, 0
        name = f'vimg://{re.sub(r"[^A-Za-z0-9_]", "_", key)}-{n_img[0]}'
        n_img[0] += 1
        doc.addResource(QTextDocument.ResourceType.ImageResource.value, QUrl(name), img)
        return name, img.width(), img.height()
    text, sections = _pack_html(res, project, src)
    doc.setHtml(text)
    W, H = 794.0, 1123.0
    doc.setPageSize(QSizeF(W, H))
    fmt = doc.rootFrame().frameFormat()
    fmt.setLeftMargin(57)
    fmt.setRightMargin(57)
    fmt.setTopMargin(57)
    fmt.setBottomMargin(68)
    doc.rootFrame().setFrameFormat(fmt)
    n = max(doc.pageCount(), 1)
    title = res.settings.title or 'Design review'
    out = _pdf_out(_PdfOut, f'{title} · {res.project} · {res.variant}', _xmp(res, title))
    p = out.painter
    scale = out.writer.width() / W
    sh = _Sheet(out)
    for i in range(n):
        if i:
            out.new_page()
        p.save()
        p.scale(scale, scale)
        p.translate(0, -i * H)
        doc.drawContents(p, QRectF(0, i * H, W, H))
        p.restore()
        fy = 297.0 - 12.0
        sh.line(15.0, fy - 1.5, 195.0, fy - 1.5, HAIR, 0.2)
        sh.text(15.0, fy - 1.0, 140, 4, f'Vantage · {res.project} · {res.variant} · {title}', sh.font(6.5), MUTED,
                elide=True)
        sh.text(155.0, fy - 1.0, 40, 4, f'Page {i + 1} of {n}', sh.font(6.5), MUTED, align='r')
    return out.finish(), out.pages, sections


def _pdf_out(base, title, xmp):
    """The exporters' A4 PDF writer (_PdfOut) with XMP metadata, which Qt only takes before painting begins."""
    from PyQt6.QtCore import QBuffer, QIODevice, QMarginsF
    from PyQt6.QtGui import QPageLayout, QPageSize, QPainter, QPdfWriter
    out = base.__new__(base)
    out.buf = QBuffer()
    out.buf.open(QIODevice.OpenModeFlag.WriteOnly)
    out.writer = QPdfWriter(out.buf)
    out.res = 300
    out.writer.setResolution(out.res)
    out.writer.setTitle(title)
    out.writer.setCreator('Vantage')
    out.writer.setDocumentXmpMetadata(xmp)
    out.writer.setPageLayout(QPageLayout(QPageSize(QPageSize.PageSizeId.A4), QPageLayout.Orientation.Portrait,
                                         QMarginsF(0, 0, 0, 0)))
    out.painter = QPainter(out.writer)
    out.painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    out.painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
    out.pages = 1
    return out


def _xmp(res: RequirementsResult, title) -> bytes:
    E = lambda s: html.escape(str(s), quote=False)        # noqa: E731
    lines = [f'{r.item["id"]} [{r.item["priority"]}] {r.item["text"]}: {STATUS_TEXT.get(r.status, r.status)}'
             for r in res.items]
    desc = f'Verdict: {VERDICT_TEXT[res.verdict]}. ' + ' | '.join(lines)
    return ('<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
            '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
            '<rdf:Description rdf:about="" xmlns:dc="http://purl.org/dc/elements/1.1/" '
            'xmlns:xmp="http://ns.adobe.com/xap/1.0/" xmlns:pdf="http://ns.adobe.com/pdf/1.3/">'
            f'<dc:title><rdf:Alt><rdf:li xml:lang="x-default">{E(title)} · {E(res.project)} · {E(res.variant)}'
            '</rdf:li></rdf:Alt></dc:title>'
            f'<dc:description><rdf:Alt><rdf:li xml:lang="x-default">{E(desc)}</rdf:li></rdf:Alt></dc:description>'
            f'<xmp:CreatorTool>{E(_app_version())}</xmp:CreatorTool><pdf:Producer>{E(_app_version())}</pdf:Producer>'
            f'<xmp:CreateDate>{E(res.created)}</xmp:CreateDate>'
            '</rdf:Description></rdf:RDF></x:xmpmeta>\n<?xpacket end="w"?>').encode('utf-8')


def files(res: RequirementsResult, project=None) -> list:
    """[(name, bytes)]: requirements.json, requirements.csv and review_pack.pdf."""
    pdf, _, _ = review_pack(res, project)
    return [('requirements.json', (json.dumps(to_doc(res, project), indent=1, ensure_ascii=False) + '\n')
             .encode('utf-8')),
            ('requirements.csv', to_csv(res).encode('utf-8')),
            ('review_pack.pdf', pdf)]


def export(res: RequirementsResult, project, out_dir) -> list[str]:
    """Write requirements.json, requirements.csv and review_pack.pdf into out_dir; returns their paths."""
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for name, data in files(res, project):
        path = os.path.join(out_dir, name)
        with open(path, 'wb') as fh:
            fh.write(data)
        paths.append(path)
    return paths


# =========================================================================== catalogue (the assistant) and the CI gate
def catalogue(project=None) -> dict:
    """The metrics of this build with their parameters (choices from the project when given), units and studies,
    JSON-able: what an item may name."""
    v = (project.variant_by_uid(settings_of(project).variant) or project.variant) if project is not None else None
    out = []
    for m in METRICS.values():
        if not available(m.study):
            continue
        s = STUDIES[m.study]
        out.append({'id': m.id, 'label': m.label, 'unit': m.unit, 'study': s.id, 'study_label': s.label,
                    'cost': s.cost, 'better': m.better, 'help': m.help,
                    'params': [{'key': p_.key, 'label': p_.label, 'default': p_.default,
                                'choices': ([{'value': k, 'label': lab} for k, lab in p_.options(project, v)]
                                            if project is not None else None)} for p_ in m.params]})
    return {'metrics': out, 'operators': list(OPS), 'priorities': list(PRIORITIES),
            'item': {'id': 'R1', 'text': 'the requirement as written', 'metric': 'metric id', 'params': {},
                     'op': '>=', 'target': 0.0, 'priority': 'must', 'note': ''}}


def check(project, engine=None, out_path=None, progress=None, variant=None) -> tuple[dict, int]:
    """The CI gate: evaluate the project's requirements (settings' variant, else the active one) and return (doc,
    exit code): EXIT_FAIL (5) when a must requirement fails or cannot be evaluated, else 0 (also when there are no
    requirements; the doc says so). out_path: a .json file (or '-' for none) gets the JSON; any other path is a
    folder that gets requirements.json, requirements.csv and review_pack.pdf. variant (a Variant or its uid)
    overrides the scorecard's own choice (batch --variant)."""
    res = run(project, engine, variant=variant, progress=progress)
    doc = to_doc(res, project)
    code = EXIT_FAIL if res.verdict in ('fail', 'incomplete') else 0
    if out_path and out_path != '-':
        if out_path.lower().endswith('.json'):
            os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
            with open(out_path, 'w', encoding='utf-8') as fh:
                fh.write(json.dumps(doc, indent=1, ensure_ascii=False) + '\n')
        else:
            export(res, project, out_path)
    return doc, code
