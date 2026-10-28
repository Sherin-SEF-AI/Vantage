"""Designed experiments, sensitivity and uncertainty: many designs of the rig evaluated automatically, what drives
each figure, and how uncertain the figures are.

Factors
-------
A factor is one design input the experiment varies: a camera's or a sensor's pose (x, y, z in metres in the vehicle
frame, ISO 8855 at the rear axle; yaw, pitch, roll in degrees, pitch positive down), a camera's lens or a sensor's
model (categorical: a list of levels), every visible camera moved together (an offset: a mount height offset, a rig
pitched by a degree), the ROI distance, a load case (the typed drops of Analysis ▸ Load cases, or their examples) and a
weather condition (Analysis ▸ Weather's presets). A numeric factor is either a value (the camera's pitch is set to it)
or an offset from the design (added to it); it has a range (low, high), a number of levels for the grid designs and a
distribution for the sampled ones: uniform over the range, normal (mean, sd) or normal truncated to the range. A camera
factor can carry its mirror partner along (the linked camera of a mirror pair gets the mirrored value: y, yaw and roll
change sign, x, z, pitch and the lens are the same), so a symmetric rig stays symmetric.

Responses
---------
Any metric of the requirements scorecard's registry (core.requirements.METRICS: ground coverage, blind area,
densities, perception ranges, lidar and radar coverage, timing, recording, power, GNSS, exposure, blind volume,
stitched view, mounting tolerance, brackets …) with its parameters, and four of this module's own for studies the
registry does not cover yet: critical targets not resolved (core.critical) and the calibration observability's
loosest camera (core.observability). Metrics read off recordings and measurements (drive-log check, measured
lidar–camera calibration, fleet) are not functions of the design and are not offered. A metric that cannot be
evaluated at a design point (no camera sees the ROI any more) gives no value there, with the reason; the analyses use
the points that have one.

Designs
-------
* Full factorial: every combination of the factors' levels (numeric levels evenly spread over the range).
* One at a time: each factor through its levels, the others at the design's values.
* Latin hypercube (McKay, Beckman and Conover 1979): n points, each factor's range cut into n strata of equal
  probability with one point in each, the strata paired at random (seeded).
* Sobol/Saltelli: the A, B and AB_i matrices of Saltelli (2002) and Saltelli et al. (2010) from a scrambled Sobol
  sequence (scipy.stats.qmc, seeded): N(k + 2) points for k factors and N base samples.
* Morris screening (Morris 1991): r trajectories on a p-level grid of the unit cube, each k + 1 points moving one
  factor at a time by Δ = p / (2(p − 1)) in a random order and direction.
* Monte Carlo: n independent draws from the factors' distributions (uncertainty propagation).
Factorial, one at a time and Morris use the factor's range; Latin hypercube, Sobol and Monte Carlo draw from its
distribution (uniform over the range unless set otherwise). A categorical factor takes its levels with equal
probability.

Analysis
--------
* Sobol indices (Sobol 1993): first order S_i = V[E(Y|X_i)] / V(Y) by Saltelli et al. (2010) (their eq. (b):
  mean(f(B)·(f(AB_i) − f(A))) / V), total S_Ti = E[V(Y|X_~i)] / V(Y) by Jansen (1999) (mean((f(A) − f(AB_i))²) / 2V),
  V the variance of f(A) and f(B) together; confidence intervals by bootstrap (Efron and Tibshirani 1993): the N rows
  resampled with replacement, percentile intervals. Validated on the Ishigami function's analytic indices (tests).
* Morris: per factor the mean μ, the mean absolute μ* (Campolongo, Cariboni and Saltelli 2007) and the standard
  deviation σ of the elementary effects, each effect in the response's unit per the factor's whole range (low to high,
  or the first to the last level); a bootstrap interval of μ* over the trajectories.
* Main effects: the mean response per level of a factor (or per sixth of its sampled values) with a t interval;
  interactions: the mean per level of two factors; response surfaces: those means on a two-factor grid.
* Pareto front: the points no other point beats on every response (each in its better direction: more coverage, less
  blind area …); the best point: the front's best on the first response (among the points meeting every response's
  target when targets are set).
* Uncertainty (Monte Carlo): each response's mean, standard deviation and 5th, 50th and 95th percentiles with
  bootstrap intervals, the share of points meeting the response's target with a Wilson interval, and convergence: the
  running mean and its interval against the number of points.

Reproducible: the design is a function of the settings (seed included) and every point's inputs are stored with its
outputs. Each design point is cached under a key of the evaluated inputs (the variant, the analysis settings, the
library, the vehicle's mesh contents, the code version and the point's factor values), so a cancelled run resumes where
it stopped, and a batch run keeps a journal (experiment_points.jsonl) to resume from.
"""
from __future__ import annotations

import copy
import csv
import datetime as _dt
import hashlib
import io
import itertools
import json
import math
import os
import threading
import time
import warnings
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, FIRST_COMPLETED, wait
from dataclasses import dataclass, field

import numpy as np

from . import requirements as RQ

SCHEMA = 'vantage.experiments/1'
SPEC_SCHEMA = 'vantage.experiment_spec/1'
JOURNAL_SCHEMA = 'vantage.experiment_points/1'
FRAME = 'vehicle (ISO 8855, rear axle)'
DESIGNS = ('factorial', 'oat', 'lhs', 'sobol', 'morris', 'montecarlo')
DESIGN_LABEL = {'factorial': 'Full factorial', 'oat': 'One at a time', 'lhs': 'Latin hypercube',
                'sobol': 'Sobol (Saltelli)', 'morris': 'Morris screening', 'montecarlo': 'Monte Carlo'}
DESIGN_USES_DIST = ('lhs', 'sobol', 'montecarlo')
KINDS = ('camera', 'sensor', 'cameras', 'roi', 'load_case', 'weather', 'param')
PROJECT_KINDS = KINDS[:-1]          # 'param': a plain input of run_function (the methods' checks), not of a project
POSE = ('x', 'y', 'z', 'yaw', 'pitch', 'roll')
ANGLES = ('yaw', 'pitch', 'roll')
MODES = ('value', 'offset')
DISTS = ('uniform', 'normal', 'truncnormal')
DIST_LABEL = {'uniform': 'Uniform', 'normal': 'Normal', 'truncnormal': 'Truncated normal'}
EXCLUDED_STUDIES = ('drivelog', 'lidar_calib', 'fleet')     # read recordings and measurements, not the design
MAX_POINTS = 100000
MAX_FACTORS = 24
MAX_RESPONSES = 12
MAX_CHOICES = 24
HEAVY_S = 0.5                       # a study slower than this per point is flagged before a run
PARALLEL_GAIN = 0.25                # each worker thread after the first adds this much speed (measured on the sample:
                                    # the studies hold the interpreter lock for part of a point; 4 threads gave 1.4×
                                    # on the ground coverage, 2.5× on the perception range)
LIMITS = {'samples': (2, MAX_POINTS), 'sobol_n': (2, 16384), 'morris_r': (2, 1000), 'levels': (2, 21),
          'morris_levels': (2, 20), 'seed': (0, 2 ** 31 - 1), 'bootstrap': (0, 5000), 'confidence': (0.5, 0.999),
          'workers': (0, 16)}
# factor ranges: absolute values and offsets (metres, degrees)
VALUE_LIMITS = {'x': (-50.0, 50.0), 'y': (-50.0, 50.0), 'z': (-5.0, 20.0), 'yaw': (-360.0, 360.0),
                'pitch': (-90.0, 90.0), 'roll': (-180.0, 180.0), 'distance': (0.1, 20.0)}
OFFSET_LIMITS = {'x': (-5.0, 5.0), 'y': (-5.0, 5.0), 'z': (-5.0, 5.0), 'yaw': (-90.0, 90.0), 'pitch': (-45.0, 45.0),
                 'roll': (-45.0, 45.0)}
ASSUMPTIONS = ('Every response is the study\'s own figure for the design point, computed as the study computes it on '
               'its page (the studies\' assumptions and limits apply unchanged). A load case moves the body in the '
               'ground coverage analysis only (as Analysis ▸ Load cases does); the other studies see the body as '
               'designed. A lens factor drops the camera\'s unit calibration (it belongs to the old lens). Sobol '
               'indices assume independent factors; bootstrap intervals are percentile intervals over resampled design '
               'rows (conservative for the quasi-random Sobol sequence); main-effect intervals are t intervals.')
LIMITS_TEXT = ('Sobol indices need the number of base samples to settle: check that the intervals are narrow (a few '
               'hundredths) before reading small differences. Morris screening ranks factors; it does not measure '
               'variance shares. Points where a response cannot be evaluated are left out of that response\'s '
               'analysis (their count is reported). A position moved by more than a few centimetres changes what the '
               'body hides from the camera: the studies render a new occlusion map for every new position, so '
               'position factors cost more per point than angles.')
REFERENCES = (
    'I. M. Sobol\', Sensitivity estimates for nonlinear mathematical models, Math. Modelling Comput. Exp. 1 (1993).',
    'A. Saltelli, Making best use of model evaluations to compute sensitivity indices, Comput. Phys. Commun. 145 '
    '(2002).',
    'A. Saltelli, P. Annoni, I. Azzini, F. Campolongo, M. Ratto, S. Tarantola, Variance based sensitivity analysis of '
    'model output, Comput. Phys. Commun. 181 (2010).',
    'M. J. W. Jansen, Analysis of variance designs for model output, Comput. Phys. Commun. 117 (1999).',
    'M. D. Morris, Factorial sampling plans for preliminary computational experiments, Technometrics 33 (1991).',
    'F. Campolongo, J. Cariboni, A. Saltelli, An effective screening design for sensitivity analysis of large models, '
    'Environ. Model. Softw. 22 (2007).',
    'M. D. McKay, R. J. Beckman, W. J. Conover, A comparison of three methods for selecting values of input variables '
    'in the analysis of output from a computer code, Technometrics 21 (1979).',
    'T. Ishigami, T. Homma, An importance quantification technique in uncertainty analysis for computer models, ISUMA '
    '1990.',
    'B. Efron, R. J. Tibshirani, An Introduction to the Bootstrap, Chapman & Hall (1993).',
    'E. B. Wilson, Probable inference, the law of succession, and statistical inference, JASA 22 (1927).')


class _Cancel(Exception):
    pass


def _num(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _clamp(v, lim):
    return min(max(v, lim[0]), lim[1])


def _flag(v, default=False) -> bool:
    if isinstance(v, (bool, int)) and not isinstance(v, float):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in ('1', 'true', 'yes', 'on')
    return default


def _version() -> str:
    try:
        from .. import __version__
        return __version__
    except Exception:
        return ''


# =========================================================================== factors
def is_categorical(kind, attr) -> bool:
    return attr in ('lens', 'model', 'level') or kind in ('load_case', 'weather')


def attr_unit(kind, attr) -> str:
    if is_categorical(kind, attr) or kind == 'param':
        return ''
    return '°' if attr in ANGLES else 'm'


@dataclass
class Factor:
    id: str = 'F1'
    kind: str = 'camera'               # camera | sensor | cameras (every visible camera) | roi | load_case | weather
    target: str = ''                   # a camera's or sensor's uid ('' for the others)
    attr: str = 'pitch'                # x y z yaw pitch roll | lens | model | distance | case | condition
    mode: str = 'value'                # value (set) | offset (added to the design's value)
    lo: float = 0.0
    hi: float = 1.0
    levels: int = 0                    # grid designs; 0 = the experiment's default
    choices: list = field(default_factory=list)      # categorical levels
    dist: str = 'uniform'              # uniform over [lo, hi] | normal(mean, sd) | truncnormal (to [lo, hi])
    mean: float = 0.0
    sd: float = 1.0
    mirror: bool = False               # a camera's mirror partner follows (y, yaw, roll mirrored)
    name: str = ''                     # the user's name for it ('' = from the target)

    @property
    def categorical(self) -> bool:
        return is_categorical(self.kind, self.attr)

    @property
    def unit(self) -> str:
        return attr_unit(self.kind, self.attr)

    def signature(self) -> list:
        """What the factor sets (not its range): two factors with the same signature set the same thing."""
        return [self.kind, self.target, self.attr, '' if self.categorical else self.mode,
                bool(self.mirror and self.kind == 'camera')]

    def to_dict(self):
        return {'id': self.id, 'kind': self.kind, 'target': self.target, 'attr': self.attr, 'mode': self.mode,
                'lo': self.lo, 'hi': self.hi, 'levels': self.levels, 'choices': list(self.choices),
                'dist': self.dist, 'mean': self.mean, 'sd': self.sd, 'mirror': self.mirror, 'name': self.name}

    @staticmethod
    def from_dict(d) -> 'Factor | None':
        """Tolerant of junk: unknown kinds and attributes give None; numbers clamped; lo ≤ hi."""
        if not isinstance(d, dict):
            return None
        f = Factor()
        f.id = str(d.get('id') or 'F1')[:12]
        f.kind = d.get('kind') if d.get('kind') in KINDS else None
        if f.kind is None:
            return None
        f.target = str(d.get('target') or '')[:40]
        attr = str(d.get('attr') or '')
        ok = {'camera': POSE + ('lens',), 'sensor': POSE + ('model',), 'cameras': POSE, 'roi': ('distance',),
              'load_case': ('case',), 'weather': ('condition',), 'param': ('value', 'level')}[f.kind]
        if attr not in ok:
            return None
        f.attr = attr
        f.mode = d.get('mode') if d.get('mode') in MODES else ('offset' if f.kind == 'cameras' else 'value')
        if f.kind == 'cameras':
            f.mode = 'offset'
        if f.kind in ('roi', 'param'):
            f.mode = 'value'
        f.name = str(d.get('name') or '')[:40].strip()
        f.mirror = _flag(d.get('mirror')) and f.kind == 'camera'
        if f.categorical:
            seen = []
            for c in d.get('choices') or []:
                c = str(c)[:60]
                if c and c not in seen:
                    seen.append(c)
            f.choices = seen[:MAX_CHOICES]
            f.mode, f.lo, f.hi, f.dist = 'value', 0.0, float(max(len(f.choices) - 1, 0)), 'uniform'
            f.mean, f.sd, f.levels = 0.0, 1.0, len(f.choices)
            return f
        lim = (VALUE_LIMITS if f.mode == 'value' else OFFSET_LIMITS).get(f.attr, (-1e9, 1e9))
        lo, hi = _num(d.get('lo')), _num(d.get('hi'))
        lo = _clamp(lo if lo is not None else 0.0, lim)
        hi = _clamp(hi if hi is not None else lo, lim)
        f.lo, f.hi = (lo, hi) if lo <= hi else (hi, lo)
        lv = _num(d.get('levels'))
        f.levels = 0 if lv is None or lv < 2 else int(_clamp(round(lv), LIMITS['levels']))
        f.dist = d.get('dist') if d.get('dist') in DISTS else 'uniform'
        m, s = _num(d.get('mean')), _num(d.get('sd'))
        f.mean = _clamp(m, lim) if m is not None else (f.lo + f.hi) / 2.0
        span = max(lim[1] - lim[0], 1e-9)
        f.sd = _clamp(s, (1e-9, span)) if s is not None and s > 0 else max((f.hi - f.lo) / 4.0, 1e-9)
        return f

    def n_levels(self, default: int) -> int:
        if self.categorical:
            return len(self.choices)
        return self.levels or default

    def value(self, u, use_dist: bool, nominal):
        """The factor's value at unit coordinate u (NaN: the design's own value, nominal)."""
        if u is None or (isinstance(u, float) and math.isnan(u)):
            return nominal
        u = float(u)
        if self.categorical:
            if not self.choices:
                return nominal
            k = len(self.choices)
            return self.choices[min(int(u * k + 1e-9), k - 1)]
        if not use_dist or self.dist == 'uniform':
            return float(self.lo + u * (self.hi - self.lo))
        uu = min(max(u, 1e-12), 1.0 - 1e-12)
        if self.dist == 'normal':
            from scipy.special import ndtri
            return float(self.mean + self.sd * ndtri(uu))
        from scipy.stats import truncnorm
        if self.hi - self.lo <= 0:
            return float(self.lo)
        a, b = (self.lo - self.mean) / self.sd, (self.hi - self.mean) / self.sd
        return float(min(max(truncnorm.ppf(uu, a, b, loc=self.mean, scale=self.sd), self.lo), self.hi))

    def values(self, u, use_dist: bool, nominal) -> list:
        """value() for a column of unit coordinates at once (NaN: nominal)."""
        u = np.asarray(u, float)
        nan = ~np.isfinite(u)
        uu = np.where(nan, 0.5, u)
        if self.categorical:
            if not self.choices:
                return [nominal] * len(u)
            k = len(self.choices)
            idx = np.minimum((uu * k + 1e-9).astype(int), k - 1)
            return [nominal if m else self.choices[i] for i, m in zip(idx, nan)]
        if not use_dist or self.dist == 'uniform':
            x = self.lo + uu * (self.hi - self.lo)
        else:
            uc = np.clip(uu, 1e-12, 1.0 - 1e-12)
            if self.dist == 'normal':
                from scipy.special import ndtri
                x = self.mean + self.sd * ndtri(uc)
            elif self.hi - self.lo <= 0:
                x = np.full(len(u), float(self.lo))
            else:
                from scipy.stats import truncnorm
                a, b = (self.lo - self.mean) / self.sd, (self.hi - self.mean) / self.sd
                x = np.clip(truncnorm.ppf(uc, a, b, loc=self.mean, scale=self.sd), self.lo, self.hi)
        return [nominal if m else float(v) for v, m in zip(x, nan)]


def _cam_of(v, uid):
    return v.cam(uid) if v is not None else None


def factor_label(f: Factor, project=None, variant=None) -> str:
    """'F pitch', 'ML+MR yaw offset', 'All cameras z offset', 'L1 model', 'ROI distance', 'Load case', 'Weather'."""
    if f.name:
        return f.name
    v = variant if variant is not None else (project.variant if project is not None else None)
    attr = {'lens': 'lens', 'model': 'model'}.get(f.attr, f.attr)
    off = ' offset' if f.mode == 'offset' and not f.categorical else ''
    if f.kind == 'camera':
        c = _cam_of(v, f.target)
        tag = c.tag if c is not None else '?'
        if f.mirror and c is not None and c.link_uid:
            p_ = v.cam(c.link_uid)
            if p_ is not None:
                tag = f'{tag}+{p_.tag}'
        return f'{tag} {attr}{off}'
    if f.kind == 'sensor':
        s = v.sensor(f.target) if v is not None else None
        return f'{s.tag if s is not None else "?"} {attr}{off}'
    if f.kind == 'cameras':
        return f'All cameras {attr}{off}'
    return {'roi': 'ROI distance', 'load_case': 'Load case', 'weather': 'Weather'}.get(f.kind, f.id)


def nominal_of(f: Factor, project, variant):
    """The design's own value of the factor: what the base design has (an offset's is 0)."""
    if f.kind in ('camera', 'sensor'):
        obj = variant.cam(f.target) if f.kind == 'camera' else variant.sensor(f.target)
        if obj is None:
            return None
        if f.attr == 'lens':
            return obj.lens_id
        if f.attr == 'model':
            return obj.model_id
        return 0.0 if f.mode == 'offset' else float(getattr(obj, f.attr))
    if f.kind == 'cameras':
        return 0.0
    if f.kind == 'roi':
        return float(project.roi.distance)
    if f.kind == 'load_case':
        return ''                         # no load: the body as designed
    if f.kind == 'param':
        return f.choices[0] if f.categorical and f.choices else (f.lo + f.hi) / 2.0
    from . import weather as WE
    return WE.settings_of(project).condition


def choice_options(project, variant, kind, target, attr) -> list:
    """[(value, label)] a categorical factor may take in this project."""
    if attr == 'lens':
        c = variant.cam(target)
        mid = c.module_id if c is not None else ''
        own = [(k, f'{l.name} ({l.label})' if l.label else l.name) for k, l in project.lenses.items()
               if l.module_id in ('', mid)]
        rest = [(k, f'{l.name} (another module\'s)') for k, l in project.lenses.items() if l.module_id not in ('', mid)]
        return own + rest
    if attr == 'model':
        s = variant.sensor(target)
        kind_ = s.kind if s is not None else 'lidar'
        return [(k, m.name) for k, m in project.sensor_models.items() if m.kind == kind_]
    if kind == 'load_case':
        from .loadcases import cases_of
        return [(str(c[0]), f'{c[0]} (front {float(c[1]):g} mm, rear {float(c[2]):g} mm)') for c in cases_of(project)]
    if kind == 'weather':
        from . import weather as WE
        st = WE.settings_of(project)
        return [(c.id, c.name) for c in st.conditions()]
    return []


def targets(project, variant=None) -> list:
    """Every design input a factor can vary, for the page's picker: [{'kind', 'target', 'attr', 'label', 'group'}]."""
    v = variant or project.variant
    out = []
    for c in v.cameras:
        for a in POSE + ('lens',):
            out.append({'kind': 'camera', 'target': c.uid, 'attr': a, 'group': f'Camera {c.tag}',
                        'label': f'{c.tag} · {a}' + ('' if c.visible else ' (hidden)')})
    for s in v.sensors:
        kind = s.kind.upper() if s.kind in ('imu', 'gnss') else s.kind.capitalize()
        for a in POSE + ('model',):
            out.append({'kind': 'sensor', 'target': s.uid, 'attr': a, 'group': f'{kind} {s.tag}',
                        'label': f'{s.tag} · {a}' + ('' if s.visible else ' (hidden)')})
    for a in POSE:
        out.append({'kind': 'cameras', 'target': '', 'attr': a, 'group': 'Every camera',
                    'label': f'All cameras · {a} offset'})
    out.append({'kind': 'roi', 'target': '', 'attr': 'distance', 'group': 'Analysis', 'label': 'ROI distance'})
    out.append({'kind': 'load_case', 'target': '', 'attr': 'case', 'group': 'Analysis', 'label': 'Load case'})
    out.append({'kind': 'weather', 'target': '', 'attr': 'condition', 'group': 'Analysis', 'label': 'Weather'})
    return out


def new_factor(project, variant, kind, target, attr, taken=()) -> dict:
    """A factor with typical starting figures: an angle ±10° around the design, a position ±50 mm as an offset, a
    lens among the camera's module's lenses, the ROI distance ±1 m, every load case, a few weather conditions."""
    v = variant or project.variant
    fid = new_id(taken)
    d = {'id': fid, 'kind': kind, 'target': target, 'attr': attr, 'levels': 0}
    if is_categorical(kind, attr):
        opts = [k for k, _ in choice_options(project, v, kind, target, attr)]
        f0 = Factor.from_dict(dict(d, choices=[]))
        nom = nominal_of(f0, project, v) if f0 is not None else None
        if attr in ('lens', 'model'):
            obj = v.cam(target) if attr == 'lens' else v.sensor(target)
            if attr == 'lens' and obj is not None:
                mine = [k for k, l in project.lenses.items() if l.module_id in ('', obj.module_id)]
                opts = mine or opts
            ch = ([nom] if nom in opts else []) + [o for o in opts if o != nom]
            d['choices'] = ch[:4]
        elif kind == 'weather':
            want = [nom, 'clear', 'fog', 'night', 'heavy_rain']
            d['choices'] = [w for i, w in enumerate(want) if w in opts and w not in want[:i]]
        else:
            d['choices'] = opts[:8]
        if kind == 'camera':
            c = v.cam(target)
            d['mirror'] = bool(c is not None and c.link_uid and v.cam(c.link_uid) is not None)
        return d
    if kind == 'roi':
        dist = float(project.roi.distance)
        return dict(d, mode='value', lo=max(0.5, dist - 1.0), hi=dist + 1.0, dist='uniform', mean=dist, sd=0.25)
    if kind == 'cameras':
        span = 2.0 if attr in ANGLES else 0.05
        return dict(d, mode='offset', lo=-span, hi=span, dist='normal', mean=0.0, sd=span / 2.0)
    obj = v.cam(target) if kind == 'camera' else v.sensor(target)
    if kind == 'camera':
        d['mirror'] = bool(obj is not None and obj.link_uid and v.cam(obj.link_uid) is not None)
    if attr in ANGLES:
        nom = float(getattr(obj, attr)) if obj is not None else 0.0
        lo, hi = nom - 10.0, nom + 10.0
        if attr == 'pitch':
            lo, hi = max(lo, -90.0), min(hi, 90.0)
        return dict(d, mode='value', lo=lo, hi=hi, dist='normal', mean=nom, sd=1.0)
    return dict(d, mode='offset', lo=-0.05, hi=0.05, dist='normal', mean=0.0, sd=0.01)


def new_id(taken=(), prefix='F') -> str:
    taken = set(taken)
    n = 1
    while f'{prefix}{n}' in taken:
        n += 1
    return f'{prefix}{n}'


# =========================================================================== responses
def _run_critical(ev):
    from . import critical as CR
    st = CR.settings_of(ev.p)
    if not any(t['enabled'] for t in st.targets):
        raise RQ.NotEvaluated('No critical target: add them on Analysis ▸ Critical targets.')
    if not any(c.visible for c in ev.v.cameras):
        raise RQ.NotEvaluated('No visible camera in the variant.')
    res = CR.run(ev.p, ev.engine, st, envelopes=False, cancel=ev.cancel)
    if res is None:
        raise RQ._Cancelled()
    return res


def _run_observability(ev):
    from . import observability as OB
    if not ev.p.targets:
        raise RQ.NotEvaluated('No calibration targets planned: plan them in Calibrate ▸ Targets.')
    if not any(c.visible for c in ev.v.cameras):
        raise RQ.NotEvaluated('No visible camera in the variant.')
    d = OB.settings_of(ev.p).to_dict()
    d.update(suggest=False, lidar_pairs=False, fusion=False)       # the cameras' poses only: quicker
    res = OB.run(ev.p, ev.engine, OB.ObservabilitySettings.from_dict(d), cancel=ev.cancel)
    if res is None:
        raise RQ._Cancelled()
    return res


EXTRA_STUDIES = {
    'critical': RQ.Study(id='critical', label='Critical targets', page='critical', menu='Analysis ▸ Critical targets…',
                         cost='study', run=_run_critical, figures=lambda r, ev: [], module='critical', weight=0.5),
    'observability': RQ.Study(id='observability', label='Calibration observability', page='observability',
                              menu='Calibrate ▸ Calibration observability…', cost='study', run=_run_observability,
                              figures=lambda r, ev: [], module='observability', weight=0.5),
}


def _m_crit(ev, prm):
    res = ev.study('critical')
    miss = [r.name for r in res.targets if not r.met]
    return float(len(miss)), {'detail': ('Not resolved: ' + ', '.join(miss)) if miss else 'Every target resolved.'}


def _obs_cams(ev):
    res = ev.study('observability')
    cams = [c for c in res.cameras]
    if not cams:
        raise RQ.NotEvaluated('No visible camera in the variant.')
    blind = [c['tag'] for c in cams if c['sigma'] is None]
    if blind:
        raise RQ.NotEvaluated(f'{", ".join(blind)} see no calibration board whole: their poses are not determined.')
    return cams


def _m_obs_ratio(ev, prm):
    cams = _obs_cams(ev)
    c = max(cams, key=lambda c_: c_['ratio'])
    return float(c['ratio']), {'detail': f'{c["tag"]}: {c["worst_axis"]}'}


def _m_obs_deg(ev, prm):
    cams = _obs_cams(ev)
    v, tag = max((max(c['sigma'][k] for k in ('yaw_deg', 'pitch_deg', 'roll_deg')), c['tag']) for c in cams)
    return float(v), {'detail': tag}


def _m_obs_mm(ev, prm):
    cams = _obs_cams(ev)
    v, tag = max((max(c['sigma'][k] for k in ('x_mm', 'y_mm', 'z_mm')), c['tag']) for c in cams)
    return float(v), {'detail': tag}


EXTRA_METRICS = {
    'critical.unresolved': RQ.Metric('critical.unresolved', 'Critical targets not resolved', '', 'critical', '<=',
                                     _m_crit, (), 0, 'Critical targets no camera (with the lidars) resolves'),
    'observability.worst_ratio': RQ.Metric('observability.worst_ratio', 'Loosest camera calibration against the '
                                           'requirement', '×', 'observability', '<=', _m_obs_ratio, (), 2,
                                           'The worst camera\'s worst axis (1σ) in units of the requirement'),
    'observability.worst_deg': RQ.Metric('observability.worst_deg', 'Loosest camera angle after calibration (1σ)', '°',
                                         'observability', '<=', _m_obs_deg, (), 3,
                                         'The largest angle 1σ of any camera the planned boards give'),
    'observability.worst_mm': RQ.Metric('observability.worst_mm', 'Loosest camera position after calibration (1σ)',
                                        'mm', 'observability', '<=', _m_obs_mm, (), 1,
                                        'The largest position 1σ of any camera the planned boards give'),
}


# ------------------------------------------------------------------ a CARLA-backed response: detection quality
# The design point's variant driven in CARLA with ground truth (core.simrig and the bridge in CARLA's Python, on the
# server already running on Simulate ▸ CARLA's port) and evaluated by the detector (core.detection_eval). One drive at a
# time (one server, one world): the study holds a lock, so parallel workers queue. A point already driven with the same
# rig file is reused, so a cancelled experiment resumes without driving again. Experiments only: the requirements
# scorecard never drives CARLA.
_CARLA_LOCK = threading.Lock()
# typical seconds per design point of a study that drives CARLA, measured on the development machine (RTX 5080, the
# sample's data rig at half size, a 12 s drive with ground truth: 99–107 s; YOLOX-S on its 720 fisheye images with
# OpenCV 4.13 on the CPU: 2–3 min); estimate() states it rather than drive CARLA to time it
CARLA_STUDY_S = {'detection': 260.0}


def _run_detection(ev):
    from . import detection as DT
    from . import detection_eval as DE
    from . import simrig as SR
    from .. import workdir as WD
    st = DE.settings_of(ev.p)
    s = DT.status(st.model)
    if not s['verified']:
        raise RQ.NotEvaluated(s['text'] + ' Download it on Simulate ▸ Detection quality.')
    if not any(c.visible for c in ev.v.cameras):
        raise RQ.NotEvaluated('No visible camera in the variant.')
    sim = SR.settings_of(ev.p)
    key = hashlib.sha1(json.dumps([ev.v.to_dict(), sim.to_dict()], sort_keys=True, default=str).encode()).hexdigest()
    out = WD.project_dir(ev.p, None, 'experiments', 'detection', key[:16])
    with _CARLA_LOCK:
        try:
            res = DE.evaluate_variant(ev.p, ev.v, sim, st, out, progress=ev.sub_progress('detection'),
                                      cancel=ev.cancel)
        except (RuntimeError, ValueError, FileNotFoundError) as e:
            raise RQ.NotEvaluated(str(e)) from None
    if res is None:
        raise RQ._Cancelled()
    return res


def _det_classes(p, v):
    from . import detection as DT
    return [(c, DT.CLASS_LABEL[c]) for c in ('vehicle',) + tuple(x for x in DT.EVAL_CLASSES if x != 'vehicle')]


def _det_ranges(p, v):
    from . import detection_eval as DE
    return [(RQ.EVERY, 'every range')] + [(lab, lab) for lab in DE.bin_labels(DE.settings_of(p).range_edges)]


def _det_metric(rig: bool):
    def fn(ev, prm):
        from . import detection as DT
        res = ev.study('detection')
        cls = prm.get('cls') or 'vehicle'
        rb = prm.get('range') or 'all'
        if rig:
            r = next((x for x in res.rig if x['rec'] == 0 and x['class'] == cls and x['distance_bin'] == rb), None)
            n, k = (r or {}).get('n', 0), (r or {}).get('detected', 0)
        else:
            r = res.row(0, res.modes()[0], prm.get('camera') or 'all', cls, rb)
            n, k = (r or {}).get('n_gt', 0), (r or {}).get('detected', 0)
        if not r or r.get('recall') is None:
            raise RQ.NotEvaluated(f'No {DT.CLASS_LABEL.get(cls, cls).lower()} in the drive'
                                  f'{" within " + rb if rb != "all" else ""}: nothing to detect.')
        ci = r.get('recall_ci')
        return 100.0 * r['recall'], {'detail': f'{k} of {n} objects' + (
            f'; {res.settings.confidence * 100:g} % interval {ci[0] * 100:.0f}–{ci[1] * 100:.0f} %' if ci else '')}
    return fn


EXTRA_STUDIES['detection'] = RQ.Study(
    id='detection', label='Detection quality (CARLA)', page='detection', menu='Simulate ▸ Detection quality…',
    cost='study', run=_run_detection, figures=lambda r, ev: [], module='detection_eval', weight=5.0)
EXTRA_METRICS['detection.recall'] = RQ.Metric(
    'detection.recall', 'Detection recall (CARLA)', '%', 'detection', '>=', _det_metric(False),
    (RQ.Param('cls', 'Class', _det_classes, 'vehicle'), RQ.Param('range', 'Range', _det_ranges),
     RQ.Param('camera', 'Camera', RQ._tags('camera', 'every camera'))), 1,
    'Objects the detector finds at its operating point, per camera image, in a CARLA drive of the design (each point '
    'drives CARLA and runs the detector: about 4 minutes on an RTX 5080 with the sample\'s six cameras)')
EXTRA_METRICS['detection.rig_recall'] = RQ.Metric(
    'detection.rig_recall', 'Rig detection recall (CARLA)', '%', 'detection', '>=', _det_metric(True),
    (RQ.Param('cls', 'Class', _det_classes, 'vehicle'), RQ.Param('range', 'Distance', _det_ranges)), 1,
    'Objects any camera of the rig detects, each counted once per instant, by distance from the vehicle (a CARLA drive '
    'and the detector per point)')


def metric(mid) -> RQ.Metric | None:
    return RQ.METRICS.get(mid) or EXTRA_METRICS.get(mid)


def study_of(sid) -> RQ.Study | None:
    return RQ.STUDIES.get(sid) or EXTRA_STUDIES.get(sid)


def response_available(mid) -> bool:
    m = metric(mid)
    if m is None or m.study in EXCLUDED_STUDIES:
        return False
    s = study_of(m.study)
    return s is not None and (not s.module or RQ._mod(s.module) is not None)


def response_catalogue(project=None, variant=None) -> list:
    """The metrics an experiment can use, grouped by study: [{'id', 'label', 'unit', 'better', 'study',
    'study_label', 'cost', 'params': [{'key', 'label', 'default', 'choices'}]}]."""
    out = []
    v = variant or (project.variant if project is not None else None)
    for mid in list(RQ.METRICS) + [k for k in EXTRA_METRICS if k not in RQ.METRICS]:
        if not response_available(mid):
            continue
        m = metric(mid)
        s = study_of(m.study)
        out.append({'id': m.id, 'label': m.label, 'unit': m.unit, 'better': m.better, 'study': s.id,
                    'study_label': s.label, 'cost': s.cost, 'help': m.help,
                    'params': [{'key': p_.key, 'label': p_.label, 'default': p_.default,
                                'choices': [{'value': k, 'label': lab} for k, lab in p_.options(project, v)]
                                if project is not None else None} for p_ in m.params]})
    return out


def clean_response(d) -> dict | None:
    """A response as stored: a metric of the registry (or this module's), its parameters (missing at their
    defaults), an optional target with its operator."""
    if not isinstance(d, dict):
        return None
    mid = str(d.get('metric') or '')[:80]
    m = metric(mid)
    if m is None:
        return None
    raw = d.get('params') if isinstance(d.get('params'), dict) else {}
    params = {p_.key: ('' if raw.get(p_.key, p_.default) is None else str(raw.get(p_.key, p_.default))[:60])
              for p_ in m.params}
    t = _num(d.get('target'))
    op = d.get('op') if d.get('op') in RQ.OPS else m.better
    return {'metric': mid, 'params': params, 'op': op if t is not None else '', 'target': t}


def response_key(r: dict) -> str:
    prm = r.get('params') or {}
    return r['metric'] + ('?' + '&'.join(f'{k}={prm[k]}' for k in sorted(prm)) if prm else '')


@dataclass
class Response:
    key: str
    metric: str
    params: dict
    label: str
    unit: str
    better: str
    decimals: int
    study: str
    cost: str
    op: str = ''
    target: float | None = None

    @property
    def maximise(self) -> bool:
        return self.better in ('>=', '>')

    def meets(self, y) -> np.ndarray:
        y = np.asarray(y, float)
        if self.target is None or not self.op:
            return np.isfinite(y)
        eps = 1e-9 * max(1.0, abs(self.target))
        with np.errstate(invalid='ignore'):
            return {'>=': y >= self.target - eps, '>': y > self.target + eps, '<=': y <= self.target + eps,
                    '<': y < self.target - eps, '==': np.abs(y - self.target) <= eps}[self.op] & np.isfinite(y)

    def to_dict(self):
        return {'key': self.key, 'metric': self.metric, 'params': dict(self.params), 'label': self.label,
                'unit': self.unit, 'better': self.better, 'study': self.study, 'cost': self.cost,
                'op': self.op, 'target': self.target}


def resolve_response(r: dict, project=None, variant=None) -> Response:
    m = metric(r['metric'])
    s = study_of(m.study)
    bits = []
    for p_ in m.params:
        val = r['params'].get(p_.key, p_.default)
        lab = dict(p_.options(project, variant)).get(val) if project is not None else None
        bits.append(str(lab or val or {'sector': 'every sector', 'cls': 'every class'}.get(p_.key, 'worst')))
    label = m.label + (f' ({", ".join(bits)})' if bits else '')
    return Response(response_key(r), m.id, dict(r['params']), label, m.unit, m.better, m.decimals, s.id, s.cost,
                    r.get('op') or '', r.get('target'))


# =========================================================================== settings (the experiment's spec)
@dataclass
class ExperimentSettings:
    name: str = 'Experiment'
    variant: str = ''                   # uid of the variant the experiment varies ('' = the active one)
    factors: list = field(default_factory=list)      # Factor dicts
    responses: list = field(default_factory=list)    # {'metric', 'params', 'op', 'target'}
    design: str = 'lhs'
    samples: int = 64                   # Latin hypercube and Monte Carlo points
    sobol_n: int = 128                  # Sobol base samples N (N(k + 2) points)
    morris_r: int = 10                  # Morris trajectories r (r(k + 1) points)
    levels: int = 3                     # numeric levels of the factorial and one-at-a-time designs
    morris_levels: int = 4              # Morris grid levels p
    seed: int = 1
    bootstrap: int = 500                # resamples for the intervals (0: none)
    confidence: float = 0.95
    workers: int = 0                    # parallel evaluations (0: as many as help, up to 4)

    def to_dict(self):
        return {'name': self.name, 'variant': self.variant, 'factors': [dict(f) for f in self.factors],
                'responses': [dict(r, params=dict(r['params'])) for r in self.responses], 'design': self.design,
                'samples': self.samples, 'sobol_n': self.sobol_n, 'morris_r': self.morris_r, 'levels': self.levels,
                'morris_levels': self.morris_levels, 'seed': self.seed, 'bootstrap': self.bootstrap,
                'confidence': self.confidence, 'workers': self.workers}

    @staticmethod
    def from_dict(d) -> 'ExperimentSettings':
        """Tolerant of junk: unknown keys ignored, numbers clamped to LIMITS, factors and responses cleaned (unusable
        ones dropped, ids made unique)."""
        s = ExperimentSettings()
        d = d if isinstance(d, dict) else {}
        if isinstance(d.get('name'), str) and d['name'].strip():
            s.name = d['name'].strip()[:80]
        if isinstance(d.get('variant'), str):
            s.variant = d['variant'][:40]
        if d.get('design') in DESIGNS:
            s.design = d['design']
        taken = []
        for raw in (d.get('factors') or [])[:MAX_FACTORS]:
            f = Factor.from_dict(raw)
            if f is None:
                continue
            if f.id in taken or not f.id:
                f.id = new_id(taken)
            taken.append(f.id)
            s.factors.append(f.to_dict())
        seen = set()
        for raw in (d.get('responses') or []):
            r = clean_response(raw)
            if r is None or response_key(r) in seen or len(s.responses) >= MAX_RESPONSES:
                continue
            seen.add(response_key(r))
            s.responses.append(r)
        for k, lim in LIMITS.items():
            v = _num(d.get(k))
            if v is not None:
                v = _clamp(v, lim)
                setattr(s, k, v if k == 'confidence' else int(round(v)))
        return s


def settings_of(project) -> ExperimentSettings:
    return ExperimentSettings.from_dict(getattr(project.analysis, 'experiments', None))


def factors_of(st: ExperimentSettings) -> list:
    return [f for f in (Factor.from_dict(d) for d in st.factors) if f is not None]


def responses_of(st: ExperimentSettings, project=None, variant=None) -> list:
    return [resolve_response(r, project, variant) for r in st.responses if metric(r['metric']) is not None]


# =========================================================================== designs
@dataclass
class Design:
    method: str
    U: np.ndarray                       # (n, k) unit coordinates; NaN: the factor at the design's own value
    use_dist: bool
    info: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.U.shape[0])

    def block(self, i) -> str:
        """Where a point sits in the design: 'A', 'B', 'AB2' (Sobol), 'T3.1' (Morris trajectory 3, step 1), 'F2'
        (one at a time: factor 2 varies), '' otherwise."""
        m = self.method
        if m == 'sobol':
            N = self.info['N']
            b = i // N
            return 'A' if b == 0 else 'B' if b == 1 else f'AB{b - 1}'
        if m == 'morris':
            k1 = self.info['k'] + 1
            return f'T{i // k1 + 1}.{i % k1}'
        if m == 'oat':
            return self.info['factor_of_row'][i]
        return ''


def point_count(factors, st: ExperimentSettings) -> int:
    """How many points the design evaluates (without building it)."""
    k = len(factors)
    if k == 0:
        return 0
    m = st.design
    if m == 'factorial':
        return int(np.prod([max(f.n_levels(st.levels), 1) for f in factors], dtype=float))
    if m == 'oat':
        return sum(max(f.n_levels(st.levels), 1) for f in factors)
    if m == 'sobol':
        return st.sobol_n * (k + 2)
    if m == 'morris':
        return st.morris_r * (k + 1)
    return st.samples


def design_text(factors, st: ExperimentSettings) -> str:
    """'1 536 points = 256 × (4 + 2)' — how the count comes about."""
    k, n, m = len(factors), point_count(factors, st), st.design
    if not k:
        return 'No factor yet'
    if m == 'factorial':
        return f'{n:,} points = ' + ' × '.join(str(f.n_levels(st.levels)) for f in factors)
    if m == 'oat':
        return f'{n:,} points = ' + ' + '.join(str(f.n_levels(st.levels)) for f in factors)
    if m == 'sobol':
        return f'{n:,} points = {st.sobol_n} × ({k} + 2)'
    if m == 'morris':
        return f'{n:,} points = {st.morris_r} × ({k} + 1)'
    return f'{n:,} points'


def make_design(factors, st: ExperimentSettings) -> Design:
    """The design's points in unit coordinates. Raises ValueError when there is no factor or too many points."""
    k = len(factors)
    if k == 0:
        raise ValueError('The experiment has no factor.')
    n = point_count(factors, st)
    if n > MAX_POINTS:
        raise ValueError(f'The design has {n:,} points, more than {MAX_POINTS:,}: fewer levels or samples.')
    m = st.design
    rng = np.random.default_rng(st.seed)
    if m == 'factorial':
        axes = [np.linspace(0.0, 1.0, max(f.n_levels(st.levels), 1)) if f.n_levels(st.levels) > 1 else np.zeros(1)
                for f in factors]
        U = np.array(list(itertools.product(*axes)), float).reshape(-1, k)
        return Design(m, U, False, {'levels': [len(a) for a in axes]})
    if m == 'oat':
        rows, who = [], []
        for j, f in enumerate(factors):
            L = max(f.n_levels(st.levels), 1)
            for u in (np.linspace(0.0, 1.0, L) if L > 1 else [0.0]):
                r = np.full(k, np.nan)
                r[j] = u
                rows.append(r)
                who.append(f.id)
        return Design(m, np.array(rows, float).reshape(-1, k), False, {'factor_of_row': who})
    if m == 'lhs':
        U = np.empty((n, k))
        for j in range(k):
            U[:, j] = (rng.permutation(n) + rng.random(n)) / n
        return Design(m, U, True, {})
    if m == 'montecarlo':
        return Design(m, rng.random((n, k)), True, {})
    if m == 'sobol':
        from scipy.stats import qmc
        N = st.sobol_n
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')               # a base count not a power of two: less balanced, still valid
            S = qmc.Sobol(2 * k, scramble=True, seed=st.seed).random(N)
        A, B = S[:, :k], S[:, k:]
        blocks = [A, B]
        for i in range(k):
            ABi = A.copy()
            ABi[:, i] = B[:, i]
            blocks.append(ABi)
        return Design(m, np.vstack(blocks), True, {'N': N, 'k': k})
    if m == 'morris':
        p = st.morris_levels
        delta = p / (2.0 * (p - 1))
        grid = np.arange(p) / (p - 1)
        starts = grid[grid <= 1.0 - delta + 1e-12]
        Bm = np.tril(np.ones((k + 1, k)), -1)
        J = np.ones((k + 1, k))
        rows = []
        for _ in range(st.morris_r):
            x = rng.choice(starts, size=k)
            D = rng.choice([-1.0, 1.0], size=k)
            perm = rng.permutation(k)
            Bs = (J * x[None, :] + (delta / 2.0) * ((2.0 * Bm - J) * D[None, :] + J))[:, perm]
            rows.append(Bs)
        return Design(m, np.vstack(rows), False, {'r': st.morris_r, 'p': p, 'delta': delta, 'k': k})
    raise ValueError(f'Unknown design {m!r}.')


def design_values(design: Design, factors, nominal) -> list:
    """The design's points as factor values: (n) lists of k values (a number or a level; the design's own value
    where the design leaves the factor alone)."""
    cols = [f.values(design.U[:, j], design.use_dist, nominal[j]) for j, f in enumerate(factors)]
    return [list(row) for row in zip(*cols)] if cols else [[] for _ in range(design.n)]


# =========================================================================== applying a point to the project
def _wheelbase(project) -> float:
    wc = project.vehicle.wheel_centres
    return abs(wc['FL'][0] - wc['RL'][0]) if 'FL' in wc and 'RL' in wc else 2.6


_MIRROR_SIGN = {'y': -1.0, 'yaw': -1.0, 'roll': -1.0}


def apply(project, variant_uid, factors, values):
    """The project with the variant (variant_uid) active and set to the point's values (None: the design's own).
    Returns (project copy, body pose or None, notes). The input project is not changed."""
    from .project import Variant
    v0 = project.variant_by_uid(variant_uid) or project.variant
    q = copy.copy(project)
    v = Variant.from_dict(v0.to_dict())
    q.variants = [v if x.uid == v0.uid else x for x in project.variants]
    q.active_uid = v.uid
    pose, notes = None, []
    for f, val in zip(factors, values):
        if val is None:
            continue
        if f.kind in ('camera', 'sensor'):
            obj = v.cam(f.target) if f.kind == 'camera' else v.sensor(f.target)
            if obj is None:
                continue
            objs = [(obj, 1.0)]
            if f.kind == 'camera' and f.mirror and obj.link_uid:
                pt = v.cam(obj.link_uid)
                if pt is not None:
                    objs.append((pt, -1.0))
            for o, side in objs:
                if f.attr == 'lens':
                    if o.lens_id != val:
                        o.lens_id = str(val)
                        o.calib = {}                      # a unit calibration belongs to the old lens
                elif f.attr == 'model':
                    o.model_id = str(val)
                else:
                    sgn = _MIRROR_SIGN.get(f.attr, 1.0) if side < 0 else 1.0
                    if f.mode == 'offset':
                        setattr(o, f.attr, float(getattr(o, f.attr)) + sgn * float(val))
                    else:
                        setattr(o, f.attr, sgn * float(val))
        elif f.kind == 'cameras':
            for c in v.cameras:
                if c.visible:
                    setattr(c, f.attr, float(getattr(c, f.attr)) + float(val))
        elif f.kind == 'roi':
            q.roi = copy.copy(project.roi)
            q.roi.distance = float(val)
        elif f.kind == 'weather':
            q.analysis = copy.copy(q.analysis)
            q.analysis.weather = dict(q.analysis.weather or {}, condition=str(val))
        elif f.kind == 'load_case' and val:
            from .loadcases import body_pose, cases_of
            case = next((c for c in cases_of(project) if str(c[0]) == str(val)), None)
            if case is None:
                notes.append(f'No load case {val!r}: the body as designed.')
                continue
            fr, rr = float(case[1]), float(case[2])
            if fr or rr:
                R, t, _ = body_pose(fr, rr, _wheelbase(project))
                pose = (R, t)
    return q, pose, notes


# =========================================================================== evaluating a point
class _PosedEngine:
    """The analysis engine with the body moved by a load case: the ground coverage analysis sees the body pose (as
    Analysis ▸ Load cases evaluates it); everything else is the engine's own."""

    def __init__(self, eng, pose):
        self._eng, self._pose = eng, pose

    def __getattr__(self, name):
        return getattr(self._eng, name)

    def analyse(self, project, *a, **kw):
        if kw.get('body_pose') is None:
            kw['body_pose'] = self._pose
        return self._eng.analyse(project, *a, **kw)


class _Eval(RQ._Eval):
    """The scorecard's evaluation context (each study run once and shared by the responses that need it) with this
    module's own studies and the load case's body pose."""

    def __init__(self, project, engine, rq_settings, pose=None, cancel=None):
        super().__init__(project, engine, rq_settings, None, cancel)
        self._posed = None if pose is None else _PosedEngine(engine, pose)

    @property
    def engine(self):
        if self._posed is not None:
            return self._posed
        return RQ._Eval.engine.fget(self)

    def study(self, sid):
        if sid not in EXTRA_STUDIES or sid in RQ.STUDIES:
            return super().study(sid)
        if sid in self.cache:
            return self.cache[sid]
        if sid in self.errors:
            raise RQ.NotEvaluated(self.errors[sid])
        s = EXTRA_STUDIES[sid]
        if s.module and RQ._mod(s.module) is None:
            self.errors[sid] = f'The {s.label.lower()} study is not part of this build.'
            raise RQ.NotEvaluated(self.errors[sid])
        if self.stop():
            raise RQ._Cancelled()
        t0 = time.perf_counter()
        try:
            res = s.run(self)
        except RQ.NotEvaluated as e:
            self.errors[sid] = str(e)
            raise
        except RQ._Cancelled:
            raise
        except Exception as e:                 # a study that fails takes its responses with it, not the experiment
            self.errors[sid] = f'The {s.label.lower()} study failed: {type(e).__name__}: {e}'
            raise RQ.NotEvaluated(self.errors[sid])
        finally:
            self.seconds[sid] = time.perf_counter() - t0
        self.cache[sid] = res
        return res


def evaluate(project, engine, responses, pose=None, rq_settings=None, cancel=None):
    """Every response at one design (the project as apply() made it): (values (m,) with NaN where not evaluated,
    reasons [str], seconds per study)."""
    ev = _Eval(project, engine, rq_settings or RQ.settings_of(project), pose, cancel)
    vals = np.full(len(responses), np.nan)
    why = [''] * len(responses)
    for j, r in enumerate(responses):
        m = metric(r.metric)
        try:
            v, _ = m.fn(ev, r.params)
        except RQ.NotEvaluated as e:
            why[j] = str(e)
            continue
        except RQ._Cancelled:
            raise _Cancel()
        except Exception as e:                 # a metric's own failure: no value at this point, with the reason
            why[j] = f'{type(e).__name__}: {e}'
            continue
        v = _num(v)
        if v is None:
            why[j] = f'The {study_of(m.study).label.lower()} study gives no finite value.'
        else:
            vals[j] = v
    if cancel is not None and cancel():
        raise _Cancel()
    return vals, why, dict(ev.seconds)


# =========================================================================== keys, cache and journal
_MESH_KEYS: OrderedDict = OrderedDict()
_MESH_LOCK = threading.Lock()


def _mesh_key(m) -> str:
    """A hash of the vehicle mesh's contents (stable across sessions, unlike its id)."""
    if m is None:
        return ''
    with _MESH_LOCK:
        hit = _MESH_KEYS.get(id(m))
        if hit is not None and hit[0] is m:
            return hit[1]
    h = hashlib.sha1()
    for a in (m.vertices, m.faces, m.face_part, m.face_zone):
        h.update(np.ascontiguousarray(a).tobytes())
    h.update(json.dumps([[p.name, p.role, getattr(p, 'material', '')] for p in m.parts], default=str).encode())
    h.update(json.dumps([z.to_dict() for z in getattr(m, 'zones', [])], default=str).encode())
    key = h.hexdigest()[:16]
    with _MESH_LOCK:
        _MESH_KEYS[id(m)] = (m, key)
        while len(_MESH_KEYS) > 4:
            _MESH_KEYS.popitem(last=False)
    return key


def base_key(project, variant_uid='') -> str:
    """Everything the responses read, hashed: the evaluated variant, the analysis settings (the experiment's own and
    the scorecard's left out), the ROI, the rig, the library, targets, obstacles, the vehicle (mesh contents, wheels),
    the frame and the code version. A design point's key adds its factor values. (The CARLA settings count: the
    detection responses drive CARLA with them.)"""
    a = project.analysis.to_dict()
    for k in ('requirements', 'assistant', 'live', 'heatmap', 'experiments', 'automation'):
        a.pop(k, None)
    v = project.variant_by_uid(variant_uid) or project.variant
    veh = project.vehicle
    doc = {'variant': v.to_dict(), 'analysis': a, 'roi': project.roi.to_dict(),
           'rig': [r.to_dict() for r in project.rig], 'ecu': project.ecu.to_dict(),
           'library': project.section('library'), 'targets': [t.to_dict() for t in project.targets],
           'target_settings': project.target_settings.to_dict(),
           'obstacles': [o.to_dict() for o in project.obstacles],
           'frame': [project.frame, list(project.custom_origin), project.units],
           'mesh': [_mesh_key(veh.mesh), veh.occluder_target], 'wheels': veh.wheel_centres,
           'version': _version()}
    doc['variant'].pop('created', None)
    doc['variant'].pop('name', None)
    doc['variant'].pop('note', None)
    return hashlib.sha1(json.dumps(doc, sort_keys=True, default=str).encode('utf-8')).hexdigest()[:20]


def _canon(v):
    if v is None or isinstance(v, str):
        return v
    return float(v)


def point_key(bkey: str, factors, values) -> str:
    doc = [bkey, [[f.signature(), _canon(x)] for f, x in zip(factors, values)]]
    return hashlib.sha1(json.dumps(doc, sort_keys=True).encode('utf-8')).hexdigest()[:24]


class PointCache:
    """Responses per design point, by the point's key: {key: {response key: [value or None, reason]}}. Thread-safe.
    With a journal file every new point is appended to it as a JSON line (and read back by load())."""

    def __init__(self, journal=None):
        self.data: dict = {}
        self.lock = threading.Lock()
        self.journal = journal
        self._fh = None

    def __len__(self):
        return len(self.data)

    def get(self, key, rkeys):
        with self.lock:
            d = self.data.get(key)
        if d is None or any(k not in d for k in rkeys):
            return None
        return (np.array([np.nan if d[k][0] is None else d[k][0] for k in rkeys], float), [d[k][1] for k in rkeys])

    def put(self, key, rkeys, vals, why):
        entry = {k: [None if not np.isfinite(v) else float(v), w] for k, v, w in zip(rkeys, vals, why)}
        with self.lock:
            self.data.setdefault(key, {}).update(entry)
            if self._fh is not None:
                self._fh.write(json.dumps({'key': key, 'r': entry}) + '\n')
                self._fh.flush()

    def open_journal(self, bkey):
        """Read the journal (when it is for the same inputs) and keep appending to it; a journal for other inputs is
        started afresh."""
        if not self.journal:
            return 0
        n = 0
        ok = False
        if os.path.exists(self.journal):
            try:
                with open(self.journal, encoding='utf-8') as fh:
                    head = json.loads(fh.readline() or '{}')
                    ok = head.get('schema') == JOURNAL_SCHEMA and head.get('base_key') == bkey
                    if ok:
                        for line in fh:
                            try:
                                d = json.loads(line)
                            except ValueError:
                                continue                  # a line cut short when the run was stopped
                            if isinstance(d, dict) and isinstance(d.get('r'), dict):
                                self.data.setdefault(str(d.get('key')), {}).update(d['r'])
                                n += 1
            except (OSError, ValueError):
                ok = False
        os.makedirs(os.path.dirname(os.path.abspath(self.journal)), exist_ok=True)
        if ok:
            with open(self.journal, 'rb') as fh:
                fh.seek(0, os.SEEK_END)
                cut = fh.tell() > 0 and (fh.seek(-1, os.SEEK_END) or fh.read(1) != b'\n')
            self._fh = open(self.journal, 'a', encoding='utf-8')
            if cut:                                   # a line cut short by a kill: end it, keep the next one whole
                self._fh.write('\n')
        else:
            self._fh = open(self.journal, 'w', encoding='utf-8')
            self._fh.write(json.dumps({'schema': JOURNAL_SCHEMA, 'base_key': bkey, 'version': _version()}) + '\n')
            self._fh.flush()
        return n

    def close(self):
        if self._fh is not None:
            self._fh.close()
            self._fh = None


_SESSION: OrderedDict = OrderedDict()
_SESSION_LOCK = threading.Lock()


def session_cache(bkey) -> PointCache:
    """This session's cache for the inputs bkey (a run cancelled and started again resumes from it)."""
    with _SESSION_LOCK:
        c = _SESSION.pop(bkey, None)
        if c is None:                     # (an empty cache is falsy: it has a length)
            c = PointCache()
        _SESSION[bkey] = c
        while len(_SESSION) > 4:
            _SESSION.popitem(last=False)
        return c


def clear_session_cache():
    with _SESSION_LOCK:
        _SESSION.clear()


# =========================================================================== validation
def _project_for(project, uid):
    """The project with the variant uid active (a shallow copy when it is another) and whether uid was found."""
    if not uid or uid == project.active_uid:
        return project, True
    if project.variant_by_uid(uid) is None:
        return project, False
    q = copy.copy(project)
    q.active_uid = uid
    return q, True


def validate(project, st: ExperimentSettings, variant=None) -> list:
    """[(severity, message)]: 'error' stops a run, 'warning' is told. Checks every factor's target exists, its levels
    are the project's, two factors do not set the same thing, every response is a known, usable metric with valid
    parameters, and the design's size."""
    out = []
    uid = (variant.uid if hasattr(variant, 'uid') else variant) or st.variant
    p, found = _project_for(project, uid)
    v = p.variant
    if not found:
        out.append(('warning', 'The experiment\'s variant is gone: the active one is used.'))
    factors = factors_of(st)
    if not factors:
        out.append(('error', 'Add a factor: a design input to vary.'))
    if not st.responses:
        out.append(('error', 'Add a response: a figure to compute at each design point.'))
    sigs = {}
    for f in factors:
        lab = f'{f.id} ({factor_label(f, p, v)})'
        if f.kind not in PROJECT_KINDS:
            out.append(('error', f'{f.id}: a function\'s parameter, not a design input of the project.'))
            continue
        if f.kind == 'camera' and v.cam(f.target) is None:
            out.append(('error', f'{f.id}: its camera is not in the variant {v.name!r}.'))
            continue
        if f.kind == 'sensor' and v.sensor(f.target) is None:
            out.append(('error', f'{f.id}: its sensor is not in the variant {v.name!r}.'))
            continue
        if f.kind == 'camera' and not v.cam(f.target).visible:
            out.append(('warning', f'{lab}: the camera is hidden, the studies leave it out.'))
        if f.kind == 'sensor' and not v.sensor(f.target).visible:
            out.append(('warning', f'{lab}: the sensor is hidden, the studies leave it out.'))
        if f.mirror and f.kind == 'camera' and not (v.cam(f.target).link_uid and v.cam(v.cam(f.target).link_uid)):
            out.append(('warning', f'{lab}: the camera has no mirror partner; it moves alone.'))
        if f.categorical:
            opts = {k for k, _ in choice_options(p, v, f.kind, f.target, f.attr)}
            bad = [c for c in f.choices if c not in opts]
            if bad:
                out.append(('error', f'{lab}: {", ".join(map(repr, bad))} is not one of the project\'s levels.'))
            if len(f.choices) < 2:
                out.append(('error', f'{lab}: it needs at least two levels.'))
        else:
            if f.hi - f.lo <= 0 and not (st.design in DESIGN_USES_DIST and f.dist == 'normal'):
                out.append(('warning', f'{lab}: its range is empty; it stays at {f.lo:g} {f.unit}.'))
            if f.levels == 1:
                out.append(('warning', f'{lab}: one level.'))
        key = json.dumps(f.signature())
        if key in sigs:
            out.append(('error', f'{f.id} and {sigs[key]} set the same thing.'))
        sigs[key] = f.id
    resp = []
    for r in st.responses:
        m = metric(r['metric'])
        if m is None:
            out.append(('error', f'Unknown metric {r["metric"]!r}.'))
            continue
        if m.study in EXCLUDED_STUDIES:
            out.append(('error', f'{m.label}: read off recordings, not a function of the design.'))
            continue
        if not response_available(m.id):
            out.append(('error', f'{m.label}: its study is not part of this build.'))
            continue
        for p_ in m.params:
            val = r['params'].get(p_.key, p_.default)
            opts = [str(k).lower() for k, _ in p_.options(p, v)]
            if str(val).lower() not in opts:
                out.append(('error', f'{m.label}: {p_.label.lower()} {val!r} is not one of the project\'s.'))
        resp.append(m)
    if any(f.kind == 'load_case' for f in factors):
        blind = sorted({m.label for m in resp if m.study not in ('coverage', 'issues')})
        if blind:
            out.append(('warning', 'The load case moves the body in the ground coverage analysis only: '
                                   + '; '.join(blind) + ' evaluate the body as designed.'))
    if factors:
        n = point_count(factors, st)
        if n > MAX_POINTS:
            out.append(('error', f'The design has {n:,} points, more than {MAX_POINTS:,}.'))
        if st.design == 'sobol' and st.sobol_n & (st.sobol_n - 1):
            out.append(('warning', f'{st.sobol_n} base samples: a power of two (128, 256, 512 …) keeps the Sobol '
                                   'sequence balanced.'))
    return out


def errors(project, st, variant=None) -> list:
    return [m for s, m in validate(project, st, variant) if s == 'error']


# =========================================================================== results
@dataclass
class ExperimentResult:
    settings: ExperimentSettings
    factors: list                       # [Factor]
    labels: list                        # factor labels
    nominal: list                       # the design's own value of each factor
    responses: list                     # [Response]
    design: Design
    values: list                        # (n) lists of k factor values (float or level)
    Y: np.ndarray                       # (n, m), NaN where a response was not evaluated
    reasons: dict                       # (i, j) -> why point i has no value of response j
    base: dict                          # the design itself: {'y': (m,), 'reasons': [m]}
    project: str = ''
    variant: str = ''
    variant_uid: str = ''
    base_key: str = ''
    created: str = ''
    version: str = ''
    seconds: float = 0.0
    cached: int = 0                     # points taken from the cache
    workers: int = 1
    study_seconds: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    analysis: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.values)

    def X(self) -> np.ndarray:
        """(n, k) numbers: a numeric factor's value, a categorical one's level index (−1: not among its levels)."""
        out = np.empty((self.n, len(self.factors)))
        for j, f in enumerate(self.factors):
            col = [r[j] for r in self.values]
            if f.categorical:
                idx = {c: i for i, c in enumerate(f.choices)}
                out[:, j] = [idx.get(c, -1) for c in col]
            else:
                out[:, j] = np.array([np.nan if c is None else c for c in col], float)
        return out

    def response(self, key) -> int:
        return next((j for j, r in enumerate(self.responses) if r.key == key), -1)

    def point(self, i) -> dict:
        return {'index': i, 'block': self.design.block(i),
                'factors': {f.id: v for f, v in zip(self.factors, self.values[i])},
                'responses': {r.key: (None if not np.isfinite(self.Y[i, j]) else float(self.Y[i, j]))
                              for j, r in enumerate(self.responses)}}


def _workers(st: ExperimentSettings, n) -> int:
    if st.workers:
        return max(1, min(st.workers, n))
    return max(1, min(4, (os.cpu_count() or 2) // 2, n))


def run(project, engine=None, settings: ExperimentSettings | None = None, variant=None, progress=None, cancel=None,
        cache=None, journal=None, workers=None) -> ExperimentResult | None:
    """Evaluate the experiment: the design's points on the variant (settings.variant, else the active one; variant
    overrides both), each response at each point, then the analyses. cache: a PointCache (None: this session's for the
    inputs; False: none); journal: a file every point is appended to and a later run resumes from. Raises ValueError
    when the experiment cannot run (see validate); returns None when cancelled (the points done so far stay cached)."""
    from .analysis import Engine
    t0 = time.perf_counter()
    st = settings or settings_of(project)
    uid = (variant.uid if hasattr(variant, 'uid') else variant) or st.variant
    p, _ = _project_for(project, uid)
    v = p.variant
    bad = errors(p, st, v)
    if bad:
        raise ValueError(' '.join(bad))
    warns = [m for s, m in validate(p, st, v) if s == 'warning']
    factors = factors_of(st)
    responses = responses_of(st, p, v)
    design = make_design(factors, st)
    nominal = [nominal_of(f, p, v) for f in factors]
    bkey = base_key(p, v.uid)
    if cache is False:
        cache = PointCache()
    elif cache is None:
        cache = session_cache(bkey)
    if journal:
        cache.journal = journal
        cache.open_journal(bkey)
    rkeys = [r.key for r in responses]
    eng = engine or Engine()
    rq_st = RQ.settings_of(p)
    stop = (lambda: bool(cancel())) if cancel is not None else (lambda: False)
    n = design.n
    values = design_values(design, factors, nominal)
    W = workers if workers else _workers(st, n)
    Y = np.full((n, len(responses)), np.nan)
    reasons = {}
    study_s: dict = {}
    hits = [0]
    lock = threading.Lock()
    tl = threading.local()

    def engine_here():
        e = getattr(tl, 'eng', None)
        if e is None:
            e = tl.eng = eng if W == 1 else eng.share()
        return e

    def one(vals):
        key = point_key(bkey, factors, [nominal[j] if x is None else x for j, x in enumerate(vals)])
        got = cache.get(key, rkeys)
        if got is not None:
            return got[0], got[1], True
        q, pose, _ = apply(p, v.uid, factors, vals)
        y, why, secs = evaluate(q, engine_here(), responses, pose, rq_st, stop)
        cache.put(key, rkeys, y, why)
        with lock:
            for k_, s_ in secs.items():
                study_s[k_] = study_s.get(k_, 0.0) + s_
        return y, why, False

    last = [0.0]

    def prog(done):
        now = time.perf_counter()
        if progress is None or (now - last[0] < 0.2 and done < n):
            return
        last[0] = now
        el = now - t0
        left = el / max(done, 1) * (n - done)
        txt = f'{done:,} of {n:,} points' + (f' · {hits[0]:,} from the cache' if hits[0] else '') + \
            (f' · {_dur(left)} left' if done and done < n else '')
        try:
            progress(done / max(n, 1), txt)
        except TypeError:
            progress(done / max(n, 1))

    try:
        base_y, base_why, _ = one([None] * len(factors))
        if W <= 1:
            for i in range(n):
                if stop():
                    return None
                y, why, hit = one(values[i])
                Y[i] = y
                hits[0] += hit
                for j, w in enumerate(why):
                    if w:
                        reasons[(i, j)] = w
                prog(i + 1)
        else:
            done = 0
            with ThreadPoolExecutor(W) as ex:
                pending = {}
                it = iter(range(n))
                for i in itertools.islice(it, 2 * W):
                    pending[ex.submit(one, values[i])] = i
                while pending:
                    fin, _ = wait(pending, timeout=0.25, return_when=FIRST_COMPLETED)
                    if stop():
                        for fu in pending:
                            fu.cancel()
                        return None
                    for fu in fin:
                        i = pending.pop(fu)
                        y, why, hit = fu.result()
                        Y[i] = y
                        hits[0] += hit
                        for j, w in enumerate(why):
                            if w:
                                reasons[(i, j)] = w
                        done += 1
                        nxt = next(it, None)
                        if nxt is not None:
                            pending[ex.submit(one, values[nxt])] = nxt
                    if fin:
                        prog(done)
    except _Cancel:
        return None
    finally:
        cache.close()
    res = ExperimentResult(st, factors, [factor_label(f, p, v) for f in factors], nominal, responses, design, values,
                           Y, reasons, {'y': base_y, 'reasons': base_why}, getattr(project, 'name', ''), v.name,
                           v.uid, bkey, _dt.datetime.now().astimezone().replace(microsecond=0).isoformat(),
                           _version(), 0.0, hits[0], W, {k: round(s, 3) for k, s in study_s.items()}, warns)
    res.analysis = analyse(res)
    res.seconds = time.perf_counter() - t0
    if progress is not None:
        try:
            progress(1.0, f'{n:,} points')
        except TypeError:
            progress(1.0)
    return res


def _dur(s) -> str:
    if s < 90:
        return f'{s:.0f} s'
    if s < 5400:
        return f'{s / 60:.0f} min'
    return f'{s / 3600:.1f} h'


def run_function(fn, factors, settings: ExperimentSettings | None = None, responses=None, nominal=None,
                 progress=None, cancel=None) -> ExperimentResult | None:
    """The same designs and analyses for any function of the factors (the methods' own checks): fn({factor id:
    value}) -> {response name: value} (or a number for one response). factors: Factor objects or dicts; responses:
    [{'key', 'label', 'unit', 'better'}] (default one response 'y'); nominal: the factors' own values (default the
    middle of each range)."""
    st = settings or ExperimentSettings()
    fs = [f if isinstance(f, Factor) else Factor.from_dict(f) for f in factors]
    if any(f is None for f in fs):
        raise ValueError('A factor is not usable.')
    rs = [Response(r['key'], r['key'], {}, r.get('label', r['key']), r.get('unit', ''), r.get('better', '<='),
                   int(r.get('decimals', 3)), 'function', 'cheap', r.get('op', ''), r.get('target'))
          for r in (responses or [{'key': 'y', 'label': 'y'}])]
    design = make_design(fs, st)
    nom = list(nominal) if nominal is not None else \
        [f.choices[0] if f.categorical else (f.lo + f.hi) / 2.0 for f in fs]
    t0 = time.perf_counter()

    def call(vals):
        out = fn({f.id: x for f, x in zip(fs, vals)})
        if not isinstance(out, dict):
            out = {rs[0].key: out}
        return np.array([np.nan if _num(out.get(r.key)) is None else float(out[r.key]) for r in rs])
    values = design_values(design, fs, nom)
    Y = np.full((design.n, len(rs)), np.nan)
    for i, vals in enumerate(values):
        if cancel is not None and cancel():
            return None
        Y[i] = call(vals)
        if progress is not None and i % 256 == 0:
            progress(i / max(design.n, 1))
    res = ExperimentResult(st, fs, [f.name or f.id for f in fs], nom, rs, design, values, Y, {},
                           {'y': call(nom), 'reasons': [''] * len(rs)}, 'function', '', '', '',
                           _dt.datetime.now().astimezone().replace(microsecond=0).isoformat(), _version())
    res.analysis = analyse(res)
    res.seconds = time.perf_counter() - t0
    return res


# =========================================================================== analysis: the estimators
def _pct_ci(samples, conf):
    a = (1.0 - conf) / 2.0
    s = np.asarray(samples, float)
    s = s[np.isfinite(s)]
    if not len(s):
        return [None, None]
    return [float(np.quantile(s, a)), float(np.quantile(s, 1.0 - a))]


def sobol_indices(fA, fB, fAB, bootstrap=500, confidence=0.95, seed=0) -> dict:
    """First-order (Saltelli et al. 2010) and total (Jansen 1999) Sobol indices from f(A), f(B) (N,) and f(AB_i)
    (k, N), with percentile bootstrap intervals over the N rows. Rows with a missing value anywhere are left out."""
    fA, fB, fAB = np.asarray(fA, float), np.asarray(fB, float), np.atleast_2d(np.asarray(fAB, float))
    k = fAB.shape[0]
    ok = np.isfinite(fA) & np.isfinite(fB) & np.isfinite(fAB).all(axis=0)
    fA, fB, fAB = fA[ok], fB[ok], fAB[:, ok]
    N = len(fA)
    nan = [None] * k
    out = {'N': int(N), 'dropped': int((~ok).sum()), 'S1': nan, 'ST': nan, 'S1_ci': [[None, None]] * k,
           'ST_ci': [[None, None]] * k, 'V': None, 'mean': None}
    if N < 2:
        out['note'] = 'Too few points with every value to estimate the indices.'
        return out

    def est(a, b, ab):
        V = np.var(np.concatenate([a, b], axis=-1), axis=-1, ddof=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            s1 = np.mean(b[..., None, :] * (ab - a[..., None, :]), axis=-1) / V[..., None]
            st = 0.5 * np.mean((a[..., None, :] - ab) ** 2, axis=-1) / V[..., None]
        return s1, st, V
    s1, st, V = est(fA, fB, fAB)
    out['V'] = float(V)
    out['mean'] = float(np.mean(np.concatenate([fA, fB])))
    if not V > 0:
        out['note'] = 'The response does not vary over the design: no index.'
        return out
    out['S1'], out['ST'] = [float(x) for x in s1], [float(x) for x in st]
    B = int(bootstrap)
    if B > 0:
        rng = np.random.default_rng(seed)
        S1b, STb = [], []
        chunk = max(1, int(4_000_000 // max(N * (k + 2), 1)))
        done = 0
        while done < B:
            c = min(chunk, B - done)
            idx = rng.integers(0, N, size=(c, N))
            a, b, ab = fA[idx], fB[idx], np.moveaxis(fAB[:, idx], 0, 1)        # (c, N), (c, N), (c, k, N)
            s1b, stb, _ = est(a, b, ab)
            S1b.append(s1b)
            STb.append(stb)
            done += c
        S1b, STb = np.concatenate(S1b), np.concatenate(STb)
        out['S1_ci'] = [_pct_ci(S1b[:, i], confidence) for i in range(k)]
        out['ST_ci'] = [_pct_ci(STb[:, i], confidence) for i in range(k)]
    return out


def morris_indices(U, y, k, delta=None, bootstrap=500, confidence=0.95, seed=0) -> dict:
    """μ, μ* and σ of the elementary effects of each factor from Morris trajectories (rows in blocks of k + 1, each
    step moving one factor), in the response's unit per the factor's whole range; a bootstrap interval of μ* over
    the trajectories."""
    U, y = np.asarray(U, float), np.asarray(y, float)
    r = len(y) // (k + 1)
    EE = np.full((r, k), np.nan)
    for t in range(r):
        rows = slice(t * (k + 1), (t + 1) * (k + 1))
        Ut, yt = U[rows], y[rows]
        for l in range(k):
            d = Ut[l + 1] - Ut[l]
            j = int(np.argmax(np.abs(d)))
            if abs(d[j]) > 0 and np.isfinite(yt[l]) and np.isfinite(yt[l + 1]):
                EE[t, j] = (yt[l + 1] - yt[l]) / d[j]
    out = {'r': int(r), 'mu': [], 'mu_star': [], 'sigma': [], 'mu_star_ci': [], 'effects': []}
    rng = np.random.default_rng(seed)
    for j in range(k):
        e = EE[:, j][np.isfinite(EE[:, j])]
        out['effects'].append(int(len(e)))
        if not len(e):
            out['mu'].append(None)
            out['mu_star'].append(None)
            out['sigma'].append(None)
            out['mu_star_ci'].append([None, None])
            continue
        out['mu'].append(float(e.mean()))
        out['mu_star'].append(float(np.abs(e).mean()))
        out['sigma'].append(float(e.std(ddof=1)) if len(e) > 1 else 0.0)
        if bootstrap > 0 and len(e) > 1:
            b = np.abs(e)[rng.integers(0, len(e), size=(int(bootstrap), len(e)))].mean(axis=1)
            out['mu_star_ci'].append(_pct_ci(b, confidence))
        else:
            out['mu_star_ci'].append([None, None])
    return out


def _t_half(sd, n, conf):
    if n < 2 or not np.isfinite(sd):
        return None
    from scipy.stats import t as _t
    return float(_t.ppf(0.5 + conf / 2.0, n - 1) * sd / math.sqrt(n))


def _groups(x, categorical, bins):
    """Group labels for a factor column: its distinct values when few (≤ 12, or categorical), else quantile bins.
    Returns (codes (n,) int, -1 for missing; centres list; edges or None)."""
    x = np.asarray(x, float)
    ok = np.isfinite(x) & (x >= 0 if categorical else True)
    vals = np.unique(x[ok])
    if categorical or len(vals) <= 12:
        codes = np.full(len(x), -1)
        codes[ok] = np.searchsorted(vals, x[ok])
        return codes, [float(v) for v in vals], None
    edges = np.unique(np.quantile(x[ok], np.linspace(0, 1, bins + 1)))
    codes = np.full(len(x), -1)
    codes[ok] = np.clip(np.searchsorted(edges, x[ok], side='right') - 1, 0, len(edges) - 2)
    centres = [float(x[ok][codes[ok] == b].mean()) if (codes[ok] == b).any() else float('nan')
               for b in range(len(edges) - 1)]
    return codes, centres, [float(e) for e in edges]


def main_effects(X, y, factors, confidence=0.95, bins=6, rows=None) -> list:
    """Per factor: the mean response per level (or per sixth of its sampled values) with a t interval and the count.
    rows: per factor the rows to use (one at a time: the factor's own sweep), default all."""
    out = []
    y = np.asarray(y, float)
    for j, f in enumerate(factors):
        x = X[:, j]
        yj = y
        if rows is not None:
            x, yj = x[rows[j]], y[rows[j]]
        codes, centres, edges = _groups(x, f.categorical, bins)
        rows_ = []
        for b, c in enumerate(centres):
            sel = (codes == b) & np.isfinite(yj)
            n = int(sel.sum())
            m = float(yj[sel].mean()) if n else None
            sd = float(yj[sel].std(ddof=1)) if n > 1 else None
            h = _t_half(sd, n, confidence) if sd is not None else None
            lev = f.choices[int(c)] if f.categorical and 0 <= int(c) < len(f.choices) else None
            rows_.append({'x': c, 'level': lev, 'n': n, 'mean': m, 'sd': sd,
                          'lo': None if h is None else m - h, 'hi': None if h is None else m + h})
        means = [r['mean'] for r in rows_ if r['mean'] is not None]
        out.append({'factor': f.id, 'binned': edges is not None, 'edges': edges, 'rows': rows_,
                    'range': (max(means) - min(means)) if means else None})
    return out


def binned_grid(xa, xb, y, cat_a=False, cat_b=False, bins=8) -> dict:
    """The mean response on a two-factor grid (levels, or quantile bins of the sampled values): {'xa', 'xb',
    'mean' (nb, na) with None for an empty cell, 'n'}."""
    ca, xa_c, _ = _groups(xa, cat_a, bins)
    cb, xb_c, _ = _groups(xb, cat_b, bins)
    y = np.asarray(y, float)
    M = np.full((len(xb_c), len(xa_c)), np.nan)
    N = np.zeros((len(xb_c), len(xa_c)), int)
    ok = (ca >= 0) & (cb >= 0) & np.isfinite(y)
    np.add.at(N, (cb[ok], ca[ok]), 1)
    S = np.zeros_like(M)
    np.add.at(S, (cb[ok], ca[ok]), y[ok])
    with np.errstate(invalid='ignore', divide='ignore'):
        M = np.where(N > 0, S / np.maximum(N, 1), np.nan)
    return {'xa': xa_c, 'xb': xb_c, 'mean': [[None if not np.isfinite(v) else float(v) for v in row] for row in M],
            'n': N.tolist()}


def pareto_front(Y, maximise) -> np.ndarray:
    """Indices of the rows of Y (n, m) no other row beats on every column (maximise[j]: more is better); rows with a
    missing value are left out. The rows are taken in lexicographic order, so a row can only be beaten by one before
    it, and checked against the front found so far (whatever beats a row, something on the front beats too)."""
    Y = np.asarray(Y, float)
    if Y.ndim != 2 or not len(Y):
        return np.zeros(0, int)
    ok = np.isfinite(Y).all(axis=1)
    idx = np.nonzero(ok)[0]
    if not len(idx):
        return idx
    C = Y[idx] * np.where(np.asarray(maximise, bool), -1.0, 1.0)[None, :]     # smaller is better in C
    order = np.lexsort(C.T[::-1])
    front = []
    F = np.zeros((0, C.shape[1]))
    for i in order:
        c = C[i]
        if len(F) and ((F <= c).all(axis=1) & (F < c).any(axis=1)).any():
            continue
        front.append(i)
        F = C[front]
    return np.sort(idx[np.array(front, int)])


def wilson(k, n, conf=0.95):
    """Wilson's interval for a share k / n."""
    if n <= 0:
        return [None, None]
    from scipy.stats import norm
    z = float(norm.ppf(0.5 + conf / 2.0))
    ph = k / n
    den = 1.0 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return [max(0.0, c - h), min(1.0, c + h)]


def uncertainty(y, bootstrap=500, confidence=0.95, seed=0, meets=None) -> dict:
    """The response's distribution over the Monte Carlo points: mean, standard deviation, percentiles with bootstrap
    intervals, the share meeting its target (Wilson interval) and convergence of the mean."""
    y = np.asarray(y, float)
    v = y[np.isfinite(y)]
    n = len(v)
    out = {'n': int(n), 'missing': int(len(y) - n)}
    if not n:
        return out
    qs = {'p5': 0.05, 'p50': 0.5, 'p95': 0.95}
    out['mean'] = float(v.mean())
    out['sd'] = float(v.std(ddof=1)) if n > 1 else 0.0
    for k_, q in qs.items():
        out[k_] = float(np.quantile(v, q))
    out['min'], out['max'] = float(v.min()), float(v.max())
    if bootstrap > 0 and n > 1:
        rng = np.random.default_rng(seed)
        bs = {k_: [] for k_ in ('mean', 'sd', *qs)}
        chunk = max(1, int(2_000_000 // n))
        done = 0
        while done < bootstrap:
            c = min(chunk, bootstrap - done)
            s = v[rng.integers(0, n, size=(c, n))]
            bs['mean'].append(s.mean(axis=1))
            bs['sd'].append(s.std(axis=1, ddof=1))
            for k_, q in qs.items():
                bs[k_].append(np.quantile(s, q, axis=1))
            done += c
        for k_, b in bs.items():
            out[f'{k_}_ci'] = _pct_ci(np.concatenate(b), confidence)
    if meets is not None:
        mt = np.asarray(meets, bool)[np.isfinite(y)]
        k_ = int(mt.sum())
        out['meets'] = k_ / n
        out['meets_ci'] = wilson(k_, n, confidence)
    # convergence: the running mean and its t interval at geometric steps
    steps = sorted({int(round(s)) for s in np.geomspace(min(5, n), n, num=min(24, n))} | {n})
    conv = []
    cs, cs2 = np.cumsum(v), np.cumsum(v * v)
    for s in steps:
        if s < 2:
            continue
        m = cs[s - 1] / s
        var = max((cs2[s - 1] - s * m * m) / (s - 1), 0.0)
        h = _t_half(math.sqrt(var), s, confidence)
        conv.append({'n': int(s), 'mean': float(m), 'half': h, 'p95': float(np.quantile(v[:s], 0.95))})
    out['convergence'] = conv
    return out


# =========================================================================== analysis of a result
def _fin(v):
    return None if v is None or not np.isfinite(v) else float(v)


def analyse(res: ExperimentResult) -> dict:
    """Every analysis the design supports, per response: statistics, main effects, Sobol indices (Sobol design),
    Morris μ*/σ (Morris), sweeps (one at a time), the uncertainty (Monte Carlo); the Pareto front and the best point
    over all responses."""
    st, d = res.settings, res.design
    X = res.X()
    k = len(res.factors)
    out = {'design': d.method, 'points': res.n, 'responses': {}}
    for j, r in enumerate(res.responses):
        y = res.Y[:, j]
        ok = np.isfinite(y)
        a = {'n': int(ok.sum()), 'missing': int((~ok).sum()),
             'base': _fin(res.base['y'][j]) if len(res.base['y']) else None}
        if ok.any():
            a.update(min=float(y[ok].min()), max=float(y[ok].max()), mean=float(y[ok].mean()),
                     sd=float(y[ok].std(ddof=1)) if ok.sum() > 1 else 0.0)
        whys = {}
        for (i, jj), w in res.reasons.items():
            if jj == j:
                whys[w] = whys.get(w, 0) + 1
        a['not_evaluated'] = [{'reason': w, 'points': c} for w, c in sorted(whys.items(), key=lambda t: -t[1])][:6]
        sweep_rows = None
        if d.method == 'oat':
            sweep_rows = [np.array([i for i in range(res.n) if d.info['factor_of_row'][i] == f.id], int)
                          for f in res.factors]
        a['main_effects'] = main_effects(X, y, res.factors, st.confidence, rows=sweep_rows)
        if d.method == 'sobol':
            N = d.info['N']
            a['sobol'] = sobol_indices(y[:N], y[N:2 * N], y[2 * N:].reshape(k, N), st.bootstrap, st.confidence,
                                       st.seed)
        elif d.method == 'morris':
            a['morris'] = morris_indices(d.U, y, k, d.info['delta'], st.bootstrap, st.confidence, st.seed)
        elif d.method == 'montecarlo':
            a['uncertainty'] = uncertainty(y, st.bootstrap, st.confidence, st.seed,
                                           r.meets(y) if r.target is not None else None)
        if d.method == 'oat':
            sweeps = []
            for jf, f in enumerate(res.factors):
                rows = [i for i in range(res.n) if d.info['factor_of_row'][i] == f.id]
                sweeps.append({'factor': f.id, 'x': [res.values[i][jf] for i in rows],
                               'y': [_fin(y[i]) for i in rows]})
                yy = [y[i] for i in rows if np.isfinite(y[i])]
                sweeps[-1]['swing'] = float(max(yy) - min(yy)) if yy else None
            a['sweeps'] = sweeps
        if r.target is not None:
            mt = r.meets(y)
            a['meets'] = int(mt.sum())
        out['responses'][r.key] = a
    # Pareto front and the best point
    mx = [r.maximise for r in res.responses]
    front = pareto_front(res.Y, mx) if res.responses else np.zeros(0, int)
    out['pareto'] = [int(i) for i in front]
    feas = np.ones(res.n, bool)
    for j, r in enumerate(res.responses):
        if r.target is not None:
            feas &= r.meets(res.Y[:, j])
    out['feasible'] = int(feas.sum()) if any(r.target is not None for r in res.responses) else None
    out['best'] = _best(res, front, feas)
    out['best_per_response'] = {}
    for j, r in enumerate(res.responses):
        y = res.Y[:, j]
        if np.isfinite(y).any():
            i = int(np.nanargmax(y) if r.maximise else np.nanargmin(y))
            out['best_per_response'][r.key] = {'index': i, 'value': float(y[i])}
    return out


def _best(res, front, feas):
    """The front's best point on the first response (then the next ones), among those meeting every target when
    targets are set and some point meets them all."""
    if not res.responses or not len(front):
        return None
    cand = [int(i) for i in front if feas[i]] or [int(i) for i in front]
    sgn = np.array([-1.0 if r.maximise else 1.0 for r in res.responses])
    keyf = lambda i: tuple(sgn * res.Y[i])              # noqa: E731
    i = min(cand, key=keyf)
    return {'index': i, 'meets_targets': bool(feas[i]),
            'values': {r.key: _fin(res.Y[i, j]) for j, r in enumerate(res.responses)},
            'factors': {f.id: res.values[i][jf] for jf, f in enumerate(res.factors)}}


def ranking(res: ExperimentResult, key) -> list:
    """The factors ranked for a response by what the design measures: total Sobol index, Morris μ*, the sweep's
    swing (one at a time) or the main effect's range: [(factor id, score, label of the score)]."""
    a = res.analysis['responses'].get(key)
    if a is None:
        return []
    out = []
    for j, f in enumerate(res.factors):
        if 'sobol' in a and a['sobol']['ST'][j] is not None:
            out.append((f.id, a['sobol']['ST'][j], 'total Sobol index'))
        elif 'morris' in a and a['morris']['mu_star'][j] is not None:
            out.append((f.id, a['morris']['mu_star'][j], 'μ*'))
        elif 'sweeps' in a and a['sweeps'][j]['swing'] is not None:
            out.append((f.id, a['sweeps'][j]['swing'], 'swing'))
        elif a['main_effects'][j]['range'] is not None:
            out.append((f.id, a['main_effects'][j]['range'], 'main-effect range'))
    return sorted(out, key=lambda t: -abs(t[1]))


# =========================================================================== a point as a variant
def describe_point(res: ExperimentResult, i) -> str:
    bits = []
    for j, f in enumerate(res.factors):
        v = res.values[i][j]
        bits.append(f'{res.labels[j]} {_fmt_val(f, v)}')
    return ', '.join(bits)


def _fmt_val(f: Factor, v) -> str:
    if v is None:
        return 'as designed'
    if f.categorical:
        return str(v) if v else 'none'
    if f.unit == '°':
        return f'{v:+.2f}°' if f.mode == 'offset' else f'{v:.2f}°'
    if f.mode == 'offset':
        return f'{v * 1000:+.1f} mm'
    return f'{v:.3f} m'


def to_variant(res: ExperimentResult, i, project, name=None):
    """A new variant with design point i's cameras and sensors (new uids, links kept) and its note; the factors a
    variant cannot hold (ROI, weather, load case) are named in the note. Returns (Variant, note)."""
    from .project import new_uid
    base = project.variant_by_uid(res.variant_uid)
    if base is None:
        raise ValueError('The experiment\'s variant is not in the project any more.')
    q, _, _ = apply(project, res.variant_uid, res.factors, res.values[i])
    v = copy.deepcopy(q.variant)
    v.uid = new_uid()
    v.created = time.time()
    uid_map = {}
    for c in v.cameras:
        uid_map[c.uid] = new_uid()
        c.uid = uid_map[c.uid]
    for c in v.cameras:
        c.link_uid = uid_map.get(c.link_uid, '')
    for s in v.sensors:
        s.uid = new_uid()
    v.seams = {}
    outside = [f'{res.labels[j]} {_fmt_val(f, res.values[i][j])}' for j, f in enumerate(res.factors)
               if f.kind in ('roi', 'weather', 'load_case') and res.values[i][j] not in (None, '')]
    figs = ', '.join(f'{r.label} {_fmt_resp(r, res.Y[i, j])}' for j, r in enumerate(res.responses))
    note = (f'{res.settings.name}, point {i + 1} ({DESIGN_LABEL[res.design.method].lower()}): '
            f'{describe_point(res, i)}. {figs}.')
    if outside:
        note += f' Not held by a variant: {"; ".join(outside)} (set them in the project).'
    v.name = name or f'{res.settings.name} · point {i + 1}'
    v.note = note
    return v, note


def _fmt_resp(r: Response, v) -> str:
    if v is None or not np.isfinite(v):
        return '—'
    return f'{v:,.{max(r.decimals, 0)}f}' + (f' {r.unit}' if r.unit else '')


# =========================================================================== cost
def estimate(project, engine=None, settings: ExperimentSettings | None = None, variant=None, probe=2,
             cancel=None) -> dict:
    """What a run will cost, measured: the design itself and the first `probe` design points evaluated (they go into
    this session's cache, so they are not wasted), timed per study. {'points', 'cached', 'per_point_s', 'total_s',
    'workers', 'studies': [{'id', 'label', 'seconds'}], 'heavy': [labels]}."""
    from .analysis import Engine
    st = settings or settings_of(project)
    uid = (variant.uid if hasattr(variant, 'uid') else variant) or st.variant
    p, _ = _project_for(project, uid)
    v = p.variant
    bad = errors(p, st, v)
    if bad:
        raise ValueError(' '.join(bad))
    factors = factors_of(st)
    responses = responses_of(st, p, v)
    design = make_design(factors, st)
    nominal = [nominal_of(f, p, v) for f in factors]
    bkey = base_key(p, v.uid)
    cache = session_cache(bkey)
    rkeys = [r.key for r in responses]
    eng = engine or Engine()
    rq_st = RQ.settings_of(p)
    values = design_values(design, factors, nominal)
    keys = [point_key(bkey, factors, vals) for vals in values]
    cached = sum(cache.get(kk, rkeys) is not None for kk in keys)
    times, per_study = [], {}
    # a study that drives CARLA is not timed by running it (a drive and the detector per point): its measured figure
    carla = sorted({r.study for r in responses if r.study in CARLA_STUDY_S})
    probed = [r for r in responses if r.study not in CARLA_STUDY_S]
    todo = ([None] + [i for i in range(design.n) if cache.get(keys[i], rkeys) is None][:probe]) if probed else []
    for i in todo:
        if cancel is not None and cancel():
            break
        vals = [None] * len(factors) if i is None else values[i]
        q, pose, _ = apply(p, v.uid, factors, vals)
        t0 = time.perf_counter()
        try:
            y, why, secs = evaluate(q, eng, probed, pose, rq_st, cancel)
        except _Cancel:
            break
        times.append(time.perf_counter() - t0)
        if i is not None and not carla:
            cache.put(keys[i], rkeys, y, why)
        for k_, s_ in secs.items():
            per_study.setdefault(k_, []).append(s_)
    # the first evaluation also warms the engine (cube maps): the later ones are what a point costs
    per = float(np.median(times[1:])) if len(times) > 1 else (times[0] if times else 0.0)
    per += sum(CARLA_STUDY_S[s_] for s_ in carla)
    W = _workers(st, design.n)
    left = design.n - cached
    speed = 1.0 if carla else 1.0 + PARALLEL_GAIN * (W - 1)     # one CARLA server: its points queue
    studies = [{'id': k_, 'label': study_of(k_).label if study_of(k_) else k_,
                'seconds': float(np.median(s_[1:] if len(s_) > 1 else s_))} for k_, s_ in per_study.items()]
    studies += [{'id': s_, 'label': study_of(s_).label, 'seconds': CARLA_STUDY_S[s_], 'typical': True} for s_ in carla]
    return {'points': design.n, 'cached': int(cached), 'per_point_s': per, 'total_s': per * left / speed,
            'workers': W, 'studies': sorted(studies, key=lambda s: -s['seconds']),
            'heavy': [s['label'] for s in studies if s['seconds'] > HEAVY_S]}


# =========================================================================== output
def summary(res: ExperimentResult) -> dict:
    """Compact figures (the assistant, the batch log): the design, the counts, per response its range, the factors
    ranked by what the design measures, and the best point."""
    a = res.analysis
    out = {'name': res.settings.name, 'variant': res.variant, 'design': res.design.method,
           'design_label': DESIGN_LABEL[res.design.method], 'points': res.n, 'cached': res.cached,
           'seconds': round(res.seconds, 2), 'seed': res.settings.seed,
           'factors': [{'id': f.id, 'label': lab, 'unit': f.unit,
                        'range': f.choices if f.categorical else [f.lo, f.hi]} for f, lab in zip(res.factors,
                                                                                                 res.labels)],
           'responses': []}
    for j, r in enumerate(res.responses):
        ar = a['responses'][r.key]
        rk = ranking(res, r.key)
        lab = {f.id: l_ for f, l_ in zip(res.factors, res.labels)}
        out['responses'].append({'key': r.key, 'label': r.label, 'unit': r.unit, 'evaluated': ar['n'],
                                 'min': _r(ar.get('min')), 'max': _r(ar.get('max')), 'base': _r(ar.get('base')),
                                 'drivers': [{'factor': lab[fid], 'score': _r(s, 4), 'by': by} for fid, s, by in rk],
                                 **({'uncertainty': {k: _r(ar['uncertainty'].get(k)) for k in
                                                     ('mean', 'sd', 'p5', 'p50', 'p95', 'meets')}}
                                    if 'uncertainty' in ar else {})})
    b = a.get('best')
    if b is not None:
        out['best'] = {'point': b['index'] + 1, 'design': describe_point(res, b['index']),
                       'values': {r.label: _r(b['values'][r.key]) for r in res.responses},
                       'meets_targets': b['meets_targets']}
    out['pareto_points'] = len(a.get('pareto', []))
    if res.warnings:
        out['warnings'] = list(res.warnings)
    return out


def _r(v, nd=6):
    return None if v is None or not isinstance(v, (int, float)) or not math.isfinite(v) else round(float(v), nd)


def _clean(o):
    """JSON-able: numpy scalars to Python, NaN and infinity to None."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def spec_doc(st: ExperimentSettings, project=None, variant=None) -> dict:
    """The experiment as a re-runnable spec: its settings with the variant named, the factors' targets named (tags)
    as well as by uid, for `vantage project.vantage --experiment spec.json OUT`."""
    v = variant or (project.variant_by_uid(st.variant) or project.variant if project is not None else None)
    fs = []
    for d in st.factors:
        f = Factor.from_dict(d)
        e = dict(d)
        if f is not None and v is not None:
            e['label'] = factor_label(f, project, v)
            obj = v.cam(f.target) if f.kind == 'camera' else v.sensor(f.target) if f.kind == 'sensor' else None
            if obj is not None:
                e['tag'] = obj.tag
        fs.append(e)
    return {'schema': SPEC_SCHEMA, 'project': getattr(project, 'name', ''),
            'variant': {'uid': v.uid, 'name': v.name} if v is not None else None,
            'settings': dict(st.to_dict(), factors=fs, variant=v.uid if v is not None else st.variant),
            'vantage': _version(), 'frame': FRAME}


def load_spec(doc, project=None) -> ExperimentSettings:
    """Settings from a spec (a dict, JSON text or a file path). Factors name their camera or sensor by uid; when the
    uid is not in the project's variant, its tag is used (a spec from a copy of the project). Raises ValueError for
    something that is not a spec."""
    if isinstance(doc, str):
        if os.path.exists(doc):
            with open(doc, encoding='utf-8') as fh:
                doc = json.load(fh)
        else:
            doc = json.loads(doc)
    if not isinstance(doc, dict):
        raise ValueError('Not an experiment spec.')
    if doc.get('schema') not in (None, SPEC_SCHEMA) and 'settings' not in doc:
        raise ValueError(f'Not an experiment spec ({doc.get("schema")!r}).')
    raw = doc.get('settings') if isinstance(doc.get('settings'), dict) else doc
    raw = copy.deepcopy(raw)
    if project is not None:
        vname = (doc.get('variant') or {}).get('name') if isinstance(doc.get('variant'), dict) else None
        v = project.variant_by_uid(raw.get('variant', '')) or \
            next((x for x in project.variants if vname and x.name == vname), None) or project.variant
        raw['variant'] = v.uid
        for f in raw.get('factors') or []:
            if not isinstance(f, dict) or f.get('kind') not in ('camera', 'sensor'):
                continue
            obj = v.cam(f.get('target')) if f['kind'] == 'camera' else v.sensor(f.get('target'))
            if obj is None and f.get('tag'):
                obj = v.by_tag(f['tag']) if f['kind'] == 'camera' else \
                    next((s for s in v.sensors if s.tag == f['tag']), None)
                if obj is not None:
                    f['target'] = obj.uid
    return ExperimentSettings.from_dict(raw)


def to_doc(res: ExperimentResult, project=None) -> dict:
    """The analysis as JSON: the spec, the factors and responses resolved, every analysis, the best point and the
    Pareto front (the points themselves are in the CSV)."""
    a = res.analysis
    doc = {
        'schema': SCHEMA, 'project': res.project or getattr(project, 'name', ''), 'variant': res.variant,
        'variant_uid': res.variant_uid, 'created': res.created, 'vantage': res.version, 'frame': FRAME,
        'units': 'positions m, angles degrees (pitch positive down), responses in their metrics\' units',
        'name': res.settings.name, 'design': res.design.method, 'design_label': DESIGN_LABEL[res.design.method],
        'design_info': {k: v for k, v in res.design.info.items() if k != 'factor_of_row'},
        'seed': res.settings.seed, 'bootstrap': res.settings.bootstrap, 'confidence': res.settings.confidence,
        'inputs_key': res.base_key, 'points': res.n, 'cached_points': res.cached, 'seconds': round(res.seconds, 3),
        'workers': res.workers, 'study_seconds': res.study_seconds,
        'settings': res.settings.to_dict(),
        'factors': [dict(f.to_dict(), label=lab, unit=f.unit, design_value=nom,
                         uses='distribution' if res.design.use_dist else 'range')
                    for f, lab, nom in zip(res.factors, res.labels, res.nominal)],
        'responses': [dict(r.to_dict(), design_value=_r(res.base['y'][j]),
                           design_reason=res.base['reasons'][j] or None) for j, r in enumerate(res.responses)],
        'analysis': a,
        'best': None if a.get('best') is None else dict(a['best'], point=a['best']['index'] + 1,
                                                       description=describe_point(res, a['best']['index'])),
        'warnings': list(res.warnings), 'assumptions': ASSUMPTIONS, 'limits': LIMITS_TEXT,
        'references': list(REFERENCES), 'points_file': 'experiment_points.csv'}
    return _clean(doc)


def to_csv(res: ExperimentResult) -> str:
    """Every point, one row each: its index and place in the design, every factor's value, every response's value
    (blank where not evaluated) and why a response has no value."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator='\n')
    head = ['point', 'block']
    head += [f'{f.id} {lab}' + (f' ({f.unit})' if f.unit else '') for f, lab in zip(res.factors, res.labels)]
    head += [f'{r.label}' + (f' ({r.unit})' if r.unit else '') for r in res.responses]
    head += ['not evaluated']
    w.writerow(head)
    w.writerow(['design', 'as designed'] + ['' if v is None else (v if isinstance(v, str) else f'{v:.10g}')
                                            for v in res.nominal] +
               ['' if not np.isfinite(y) else f'{y:.10g}' for y in res.base['y']] +
               ['; '.join(f'{r.key}: {t}' for r, t in zip(res.responses, res.base['reasons']) if t)])
    for i in range(res.n):
        row = [i + 1, res.design.block(i)]
        row += ['' if v is None else (v if isinstance(v, str) else f'{v:.10g}') for v in res.values[i]]
        row += ['' if not np.isfinite(y) else f'{y:.10g}' for y in res.Y[i]]
        row.append('; '.join(f'{r.key}: {res.reasons[(i, j)]}' for j, r in enumerate(res.responses)
                             if (i, j) in res.reasons))
        w.writerow(row)
    return buf.getvalue()


# --------------------------------------------------------------------------- plots (PNG, drawn with OpenCV)
_BG, _INK, _INK2, _MUTED, _HAIR = (248, 248, 246), (27, 27, 26), (82, 81, 78), (137, 135, 129), (225, 224, 220)
_BLUE, _BLUE_L, _RED, _GREEN, _AMBER = (229, 135, 57), (239, 182, 134), (59, 59, 208), (12, 163, 12), (25, 178, 250)


def _bgr(h):
    return int(h[5:7], 16), int(h[3:5], 16), int(h[1:3], 16)


class _Plot:
    """A light plot canvas with text, axes and marks (OpenCV; text spelled in ASCII)."""

    def __init__(self, w, h):
        import cv2
        self.cv2 = cv2
        self.img = np.full((h, w, 3), _BG, np.uint8)

    def text(self, t, x, y, col=_INK, s=0.42, th=1, right=False, centre=False):
        from .pngtext import ascii_text
        t = ascii_text(t).replace('( deg)', '(deg)')
        (tw, _), _ = self.cv2.getTextSize(t, self.cv2.FONT_HERSHEY_SIMPLEX, s, th)
        if right:
            x -= tw
        elif centre:
            x -= tw / 2
        self.cv2.putText(self.img, t, (int(x), int(y)), self.cv2.FONT_HERSHEY_SIMPLEX, s, col, th, self.cv2.LINE_AA)

    def line(self, x0, y0, x1, y1, col=_HAIR, th=1):
        self.cv2.line(self.img, (int(round(x0)), int(round(y0))), (int(round(x1)), int(round(y1))), col, th,
                      self.cv2.LINE_AA)

    def rect(self, x0, y0, x1, y1, col, fill=True):
        self.cv2.rectangle(self.img, (int(round(x0)), int(round(y0))), (int(round(x1)), int(round(y1))), col,
                           -1 if fill else 1)

    def dot(self, x, y, col, r=3, fill=True):
        self.cv2.circle(self.img, (int(round(x)), int(round(y))), r, col, -1 if fill else 1, self.cv2.LINE_AA)

    def axes(self, x0, y0, x1, y1, xlim, ylim, xlabel='', ylabel='', nx=4, ny=4):
        """Gridlines and tick labels for the box (x0, y0)–(x1, y1) on xlim × ylim; returns the mapping."""
        X = lambda v: x0 + (v - xlim[0]) / ((xlim[1] - xlim[0]) or 1.0) * (x1 - x0)      # noqa: E731
        Y = lambda v: y1 - (v - ylim[0]) / ((ylim[1] - ylim[0]) or 1.0) * (y1 - y0)      # noqa: E731
        for t in (_ticks(*ylim, ny) if ny else []):
            self.line(x0, Y(t), x1, Y(t))
            self.text(_tick(t), x0 - 6, Y(t) + 4, _MUTED, 0.36, right=True)
        for t in (_ticks(*xlim, nx) if nx else []):
            if not ny:
                self.line(X(t), y0, X(t), y1)
            self.line(X(t), y1, X(t), y1 + 4, _MUTED)
            self.text(_tick(t), X(t), y1 + 17, _MUTED, 0.36, centre=True)
        self.line(x0, y1, x1, y1, _INK2)
        if xlabel:
            self.text(xlabel, x1, y1 + 32, _INK2, 0.38, right=True)
        if ylabel:
            self.text(ylabel, x0, y0 - 8, _INK2, 0.38)
        return X, Y

    def png(self) -> bytes:
        ok, buf = self.cv2.imencode('.png', self.img)
        return buf.tobytes()


def _ticks(lo, hi, n=4):
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
        return [lo]
    raw = (hi - lo) / max(n, 1)
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if raw <= m * mag + 1e-12)
    t0 = math.ceil(lo / step - 1e-9) * step
    out = []
    t = t0
    while t <= hi + 1e-9 * step:
        out.append(round(t, 10))
        t += step
    return out


def _tick(v):
    return f'{v:.4g}'


def _lim(vals, pad=0.06, zero=False):
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], float)
    if not len(v):
        return (0.0, 1.0)
    lo, hi = float(v.min()), float(v.max())
    if zero:
        lo, hi = min(lo, 0.0), max(hi, 0.0)
    if hi - lo < 1e-12:
        lo, hi = lo - (abs(lo) * 0.1 or 0.5), hi + (abs(hi) * 0.1 or 0.5)
    d = (hi - lo) * pad
    return (lo - d, hi + d)


def sensitivity_png(res: ExperimentResult, width=1100) -> bytes | None:
    """Sobol indices (first order filled, total outlined, with their intervals), Morris μ* against σ, or the
    one-at-a-time sweeps' swings: one panel per response."""
    m = res.design.method
    if m not in ('sobol', 'morris', 'oat') or not res.responses:
        return None
    k = len(res.factors)
    rs = res.responses[:4]
    ph = (60 + 26 * k) if m != 'morris' else 300
    P = _Plot(width, 50 + len(rs) * (ph + 40))
    title = {'sobol': 'Sobol indices: first order (filled) and total (outline), 95 % bootstrap intervals',
             'morris': 'Morris screening: mu* (mean |elementary effect|) against sigma',
             'oat': 'One at a time: the swing of each response over each factor\'s levels'}[m]
    P.text(f'{res.settings.name} - {res.variant}', 20, 24, _INK, 0.55)
    P.text(title, 20, 42, _INK2, 0.4)
    y = 60
    for j, r in enumerate(rs):
        a = res.analysis['responses'][r.key]
        P.text(r.label + (f' ({r.unit})' if r.unit else ''), 20, y + 16, _INK, 0.45)
        x0, x1 = 220, width - 40
        if m == 'sobol':
            s = a['sobol']
            vals = [v for v in (s['S1'] + s['ST'] + [c_ for c in s['ST_ci'] + s['S1_ci'] for c_ in c])
                    if v is not None]
            X, Y = P.axes(x0, y + 30, x1, y + 30 + 26 * k, (min(0.0, min(vals, default=0.0)),
                                                            max(1.0, max(vals, default=1.0))), (0, 1), 'index', ny=0)
            for jf in range(k):
                yy = y + 30 + 26 * jf
                P.text(res.labels[jf], x0 - 10, yy + 17, _INK2, 0.38, right=True)
                for val, ci, col, fill, off in ((s['S1'][jf], s['S1_ci'][jf], _BLUE, True, 4),
                                                (s['ST'][jf], s['ST_ci'][jf], _INK2, False, 14)):
                    if val is None:
                        continue
                    P.rect(X(0), yy + off, X(val), yy + off + 8, col, fill)
                    if ci[0] is not None:
                        P.line(X(ci[0]), yy + off + 4, X(ci[1]), yy + off + 4, _INK, 1)
                        P.line(X(ci[0]), yy + off + 1, X(ci[0]), yy + off + 7, _INK, 1)
                        P.line(X(ci[1]), yy + off + 1, X(ci[1]), yy + off + 7, _INK, 1)
            if s.get('note'):
                P.text(s['note'], x0, y + 30 + 26 * k + 34, _RED, 0.38)
        elif m == 'oat':
            sw = [w['swing'] for w in a['sweeps']]
            X, Y = P.axes(x0, y + 30, x1, y + 30 + 26 * k, (0.0, max([v for v in sw if v is not None], default=1.0)
                                                            * 1.08 or 1.0), (0, 1), f'swing, {r.unit}', ny=0)
            for jf in range(k):
                yy = y + 30 + 26 * jf
                P.text(res.labels[jf], x0 - 10, yy + 17, _INK2, 0.38, right=True)
                if sw[jf] is not None:
                    P.rect(X(0), yy + 6, X(sw[jf]), yy + 18, _BLUE)
        else:
            mo = a['morris']
            xs = [v for v in mo['mu_star'] if v is not None]
            ys = [v for v in mo['sigma'] if v is not None]
            X, Y = P.axes(x0, y + 30, x1, y + ph - 20, _lim(xs + [0.0], zero=True), _lim(ys + [0.0], zero=True),
                          f'mu*, {r.unit} per range', f'sigma, {r.unit}')
            for jf in range(k):
                if mo['mu_star'][jf] is None:
                    continue
                px, py = X(mo['mu_star'][jf]), Y(mo['sigma'][jf])
                P.dot(px, py, _BLUE, 4)
                P.text(res.labels[jf], px + 7, py - 5, _INK2, 0.36)
        y += ph + 40
    return P.png()


def effects_png(res: ExperimentResult, key=None, width=1100) -> bytes | None:
    """The main effects of every factor on a response (its first by default): the mean per level with its interval,
    on one shared scale."""
    if not res.responses:
        return None
    j = res.response(key) if key else 0
    r = res.responses[j]
    me = res.analysis['responses'][r.key]['main_effects']
    k = len(res.factors)
    cols = min(k, 3)
    rows = int(math.ceil(k / cols))
    pw, ph = (width - 40) // cols, 230
    P = _Plot(width, 70 + rows * ph)
    P.text(f'Main effects on {r.label}' + (f' ({r.unit})' if r.unit else ''), 20, 26, _INK, 0.55)
    P.text('The mean response per level of each factor (bins of its sampled values for a sampled design), with its '
           f'{res.settings.confidence * 100:g} % t interval', 20, 46, _INK2, 0.38)
    allv = [v for e in me for row in e['rows'] for v in (row['mean'], row['lo'], row['hi']) if v is not None]
    ylim = _lim(allv)
    for jf, (f, e) in enumerate(zip(res.factors, me)):
        cx, cy = 20 + (jf % cols) * pw, 70 + (jf // cols) * ph
        rows_ = [rw for rw in e['rows'] if rw['mean'] is not None]
        P.text(res.labels[jf], cx + 60, cy + 12, _INK, 0.42)
        if not rows_:
            continue
        if f.categorical:
            xs = list(range(len(rows_)))
            xlim = (-0.5, len(rows_) - 0.5)
        else:
            xs = [rw['x'] for rw in rows_]
            xlim = _lim(xs, 0.04)
        X, Y = P.axes(cx + 60, cy + 24, cx + pw - 20, cy + ph - 44, xlim, ylim, '' if f.categorical else f.unit,
                      nx=0 if f.categorical else 3, ny=4)
        pts = []
        for x, rw in zip(xs, rows_):
            px = X(x)
            if rw['lo'] is not None:
                P.line(px, Y(rw['lo']), px, Y(rw['hi']), _BLUE_L, 3)
            pts.append((px, Y(rw['mean'])))
            if f.categorical:
                P.text(str(rw['level'])[:14], px, cy + ph - 28, _MUTED, 0.34, centre=True)
        for (ax_, ay), (bx, by) in zip(pts, pts[1:]):
            P.line(ax_, ay, bx, by, _BLUE, 2)
        for px, py in pts:
            P.dot(px, py, _BLUE, 3)
    return P.png()


def _surface_pair(res):
    num = [j for j, f in enumerate(res.factors)]
    return (num[0], num[1]) if len(num) >= 2 else None


def surface_png(res: ExperimentResult, key=None, pair=None, width=900) -> bytes | None:
    """The response's mean on a grid of two factors (their levels, or eighths of their sampled values): a heatmap
    with its colour scale; empty cells pale."""
    if not res.responses or len(res.factors) < 2:
        return None
    j = res.response(key) if key else 0
    r = res.responses[j]
    a, b = pair or _surface_pair(res)
    X = res.X()
    g = binned_grid(X[:, a], X[:, b], res.Y[:, j], res.factors[a].categorical, res.factors[b].categorical)
    M = np.array([[np.nan if v is None else v for v in row] for row in g['mean']], float)
    if not np.isfinite(M).any():
        return None
    from .sensor_coverage import SEQ
    ramp = np.array([_bgr(h) for h in SEQ], float)
    lo, hi = float(np.nanmin(M)), float(np.nanmax(M))
    if r.maximise:
        tval = lambda v: (v - lo) / ((hi - lo) or 1.0)          # noqa: E731
    else:
        tval = lambda v: (hi - v) / ((hi - lo) or 1.0)          # noqa: E731
    P = _Plot(width, 660)
    P.text(f'{r.label} over {res.labels[a]} and {res.labels[b]}', 20, 26, _INK, 0.5)
    P.text('Mean per cell; darker is better (' + ('more' if r.maximise else 'less') + f' {r.unit or "is better"})',
           20, 46, _INK2, 0.38)
    x0, y0, x1, y1 = 110, 90, width - 150, 580
    nb, na = M.shape
    cw, ch = (x1 - x0) / na, (y1 - y0) / nb
    for ib in range(nb):
        for ia in range(na):
            v = M[ib, ia]
            if not np.isfinite(v):
                col = (240, 239, 236)
            else:
                t = min(max(tval(v), 0.0), 1.0) * (len(ramp) - 1)
                i0 = int(math.floor(t))
                i1 = min(i0 + 1, len(ramp) - 1)
                col = tuple(int(c) for c in ramp[i0] * (1 - (t - i0)) + ramp[i1] * (t - i0))
            P.rect(x0 + ia * cw + 1, y1 - (ib + 1) * ch + 1, x0 + (ia + 1) * cw - 1, y1 - ib * ch - 1, col)
            if np.isfinite(v) and na * nb <= 100:
                P.text(f'{v:.3g}', x0 + (ia + 0.5) * cw, y1 - (ib + 0.5) * ch + 4,
                       _INK if tval(v) < 0.6 else (250, 250, 250), 0.34, centre=True)
    fa, fb = res.factors[a], res.factors[b]
    for ia, xv in enumerate(g['xa']):
        lab = fa.choices[int(xv)] if fa.categorical and 0 <= int(xv) < len(fa.choices) else f'{xv:.4g}'
        P.text(str(lab)[:12], x0 + (ia + 0.5) * cw, y1 + 18, _MUTED, 0.34, centre=True)
    for ib, yv in enumerate(g['xb']):
        lab = fb.choices[int(yv)] if fb.categorical and 0 <= int(yv) < len(fb.choices) else f'{yv:.4g}'
        P.text(str(lab)[:12], x0 - 8, y1 - (ib + 0.5) * ch + 4, _MUTED, 0.34, right=True)
    P.text(res.labels[a] + (f' ({fa.unit})' if fa.unit else ''), x1, y1 + 40, _INK2, 0.4, right=True)
    P.text(res.labels[b] + (f' ({fb.unit})' if fb.unit else ''), x0 - 100, y0 - 8, _INK2, 0.4)
    # colour scale
    for s in range(100):
        t = s / 99.0 * (len(ramp) - 1)
        i0 = int(math.floor(t))
        i1 = min(i0 + 1, len(ramp) - 1)
        col = tuple(int(c) for c in ramp[i0] * (1 - (t - i0)) + ramp[i1] * (t - i0))
        yy = y1 - (y1 - y0) * s / 100.0
        P.rect(x1 + 30, yy - (y1 - y0) / 100.0, x1 + 46, yy, col)
    best, worst = (hi, lo) if r.maximise else (lo, hi)
    P.text(f'{best:.4g}', x1 + 52, y0 + 10, _INK2, 0.36)
    P.text(f'{worst:.4g}', x1 + 52, y1, _INK2, 0.36)
    return P.png()


def pareto_png(res: ExperimentResult, keys=None, width=900) -> bytes | None:
    """Every point on two responses (the first two by default), the Pareto front over all responses dark, the best
    point ringed and the design as built marked."""
    if len(res.responses) < 2:
        return None
    ja, jb = (res.response(keys[0]), res.response(keys[1])) if keys else (0, 1)
    ra, rb = res.responses[ja], res.responses[jb]
    ok = np.isfinite(res.Y[:, ja]) & np.isfinite(res.Y[:, jb])
    if not ok.any():
        return None
    P = _Plot(width, 640)
    P.text(f'{ra.label} against {rb.label}', 20, 26, _INK, 0.5)
    P.text('Grey: every point; blue: the Pareto front over all responses; ringed: the best point; square: the design',
           20, 46, _INK2, 0.38)
    xs, ys = res.Y[ok, ja], res.Y[ok, jb]
    bx = [res.base['y'][ja]] if np.isfinite(res.base['y'][ja]) else []
    by = [res.base['y'][jb]] if np.isfinite(res.base['y'][jb]) else []
    X, Y = P.axes(90, 70, width - 40, 570, _lim(list(xs) + bx), _lim(list(ys) + by),
                  f'{ra.label} ({ra.unit})' if ra.unit else ra.label, f'{rb.label} ({rb.unit})' if rb.unit else
                  rb.label)
    for x, y_ in zip(xs, ys):
        P.dot(X(x), Y(y_), (190, 189, 184), 2)
    front = [i for i in res.analysis['pareto']]
    for i in front:
        P.dot(X(res.Y[i, ja]), Y(res.Y[i, jb]), _BLUE, 4)
    if bx and by:
        P.rect(X(bx[0]) - 5, Y(by[0]) - 5, X(bx[0]) + 5, Y(by[0]) + 5, _INK, False)
    b = res.analysis.get('best')
    if b is not None and np.isfinite(res.Y[b['index'], ja]) and np.isfinite(res.Y[b['index'], jb]):
        P.dot(X(res.Y[b['index'], ja]), Y(res.Y[b['index'], jb]), _INK, 8, False)
        P.text(f'best: point {b["index"] + 1}', X(res.Y[b['index'], ja]) + 10, Y(res.Y[b['index'], jb]) - 8, _INK,
               0.38)
    return P.png()


def uncertainty_png(res: ExperimentResult, width=1100) -> bytes | None:
    """Per response (up to three): the distribution of the Monte Carlo points with the 5th, 50th and 95th percentiles,
    and the convergence of the mean with its interval against the number of points."""
    if res.design.method != 'montecarlo' or not res.responses:
        return None
    rs = res.responses[:3]
    P = _Plot(width, 60 + 280 * len(rs))
    P.text(f'Uncertainty: {res.settings.name} - {res.n} Monte Carlo points', 20, 26, _INK, 0.55)
    y0 = 50
    for j, r in enumerate(rs):
        u = res.analysis['responses'][r.key].get('uncertainty', {})
        y = res.Y[:, res.response(r.key)]
        v = y[np.isfinite(y)]
        P.text(r.label + (f' ({r.unit})' if r.unit else ''), 20, y0 + 20, _INK, 0.45)
        if len(v):
            lo, hi = _lim(v, 0.02)
            cnt, edges = np.histogram(v, bins=24, range=(lo, hi))
            X, Y = P.axes(70, y0 + 40, width // 2 - 30, y0 + 230, (lo, hi), (0, max(int(cnt.max()), 1) * 1.1),
                          r.unit, 'points')
            for c_, a_, b_ in zip(cnt, edges[:-1], edges[1:]):
                if c_:
                    P.rect(X(a_) + 1, Y(c_), X(b_) - 1, Y(0), _BLUE_L)
            for q, lab in (('p5', 'p5'), ('p50', 'p50'), ('p95', 'p95')):
                if u.get(q) is not None:
                    P.line(X(u[q]), y0 + 40, X(u[q]), y0 + 230, _INK, 1)
                    P.text(f'{lab} {u[q]:.4g}', X(u[q]) + 3, y0 + 52 + 12 * ('p5', 'p50', 'p95').index(q), _INK, 0.34)
            conv = u.get('convergence') or []
            if conv:
                ns = [c['n'] for c in conv]
                lows = [c['mean'] - (c['half'] or 0) for c in conv]
                highs = [c['mean'] + (c['half'] or 0) for c in conv]
                X2, Y2 = P.axes(width // 2 + 60, y0 + 40, width - 40, y0 + 230, (0, max(ns)), _lim(lows + highs),
                                'points', 'running mean and interval')
                for a_, b_ in zip(range(len(conv)), range(1, len(conv))):
                    P.line(X2(ns[a_]), Y2(lows[a_]), X2(ns[b_]), Y2(lows[b_]), _BLUE_L, 1)
                    P.line(X2(ns[a_]), Y2(highs[a_]), X2(ns[b_]), Y2(highs[b_]), _BLUE_L, 1)
                    P.line(X2(ns[a_]), Y2(conv[a_]['mean']), X2(ns[b_]), Y2(conv[b_]['mean']), _BLUE, 2)
        y0 += 280
    return P.png()


def files(res: ExperimentResult, project=None) -> list:
    """[(name, bytes)]: experiment_spec.json, experiment_points.csv, experiment_analysis.json and the plots that
    apply (sensitivity, effects, surface, pareto, uncertainty PNGs)."""
    v = project.variant_by_uid(res.variant_uid) if project is not None else None
    spec = spec_doc(res.settings, project, v) if project is not None else \
        {'schema': SPEC_SCHEMA, 'settings': res.settings.to_dict(), 'vantage': res.version}
    out = [('experiment_spec.json', (json.dumps(_clean(spec), indent=1, ensure_ascii=False) + '\n').encode('utf-8')),
           ('experiment_points.csv', to_csv(res).encode('utf-8')),
           ('experiment_analysis.json', (json.dumps(to_doc(res, project), indent=1, ensure_ascii=False) + '\n')
            .encode('utf-8'))]
    for name, fn in (('experiment_sensitivity.png', sensitivity_png), ('experiment_effects.png', effects_png),
                     ('experiment_surface.png', surface_png), ('experiment_pareto.png', pareto_png),
                     ('experiment_uncertainty.png', uncertainty_png)):
        try:
            data = fn(res)
        except Exception:                      # a plot that cannot be drawn is left out, never the export
            data = None
        if data:
            out.append((name, data))
    return out


def export(res: ExperimentResult, project, out_dir) -> list[str]:
    """Write files(res) into out_dir; returns their paths."""
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for name, data in files(res, project):
        path = os.path.join(out_dir, name)
        with open(path, 'wb') as fh:
            fh.write(data)
        paths.append(path)
    return paths


def batch(project, spec, out_dir, engine=None, variant=None, progress=None, workers=None) -> tuple[dict, int]:
    """`vantage project.vantage --experiment spec.json OUT`: run the spec on the project (resuming from OUT's journal
    of an interrupted run on the same inputs), write files() into OUT and return (summary, exit code): 0 when it ran,
    4 when the spec cannot be read or does not fit the project."""
    try:
        st = load_spec(spec, project)
    except (OSError, ValueError) as e:
        return {'error': f'The experiment spec cannot be read: {e}'}, 4
    bad = errors(project, st, variant)
    if bad:
        return {'error': ' '.join(bad)}, 4
    os.makedirs(out_dir, exist_ok=True)
    res = run(project, engine, st, variant=variant, progress=progress, cache=PointCache(),
              journal=os.path.join(out_dir, 'experiment_points.jsonl'), workers=workers)
    if res is None:
        return {'error': 'Cancelled.'}, 130
    paths = export(res, project, out_dir)
    return dict(summary(res), files=[os.path.basename(p) for p in paths]), 0
