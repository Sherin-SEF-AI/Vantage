"""The validation suite's machinery: the check registry, the run, and the result.

A check compares one quantity a Vantage model computes with an independent reference (a closed form written out again
in the check, another library's implementation of the same model, a published table or worked example, a measurement,
a brute-force computation or a Monte Carlo simulation) and reports one error figure against a tolerance: the check
passes when the error is finite and at most the tolerance. A check that raises fails, with the exception named; it
never stops the suite.

Checks are registered with @check (see the checks_* modules) and grouped by the model they validate (MODELS). Each is
reproducible: its random numbers come from a generator seeded with the suite's seed and the check's id, and every
input it uses is either built in the check or the project it was given (said in its detail).
"""
from __future__ import annotations

import math
import os
import platform
import sys
import tempfile
import time
import traceback
import zlib
from dataclasses import dataclass, field

import numpy as np

SCHEMA = 'vantage.validation/1'
DEFAULT_SEED = 20261027

# the models the suite validates, in report order: (id, title, modules, methods-page anchor)
MODELS = (
    ('lens', 'Lens projection', 'vantage.core.lens', 'lens'),
    ('coverage', 'Ground coverage and pixel density', 'vantage.core.analysis', 'coverage'),
    ('occlusion', 'Occlusion by the body', 'vantage.core.occlusion', 'occlusion'),
    ('perception', 'Perception range', 'vantage.core.perception', 'perception'),
    ('critical', 'Critical targets and mounting envelopes', 'vantage.core.critical', 'critical'),
    ('lidar', 'Lidar beams, rings and points', 'vantage.core.sensors, vantage.core.sensor_coverage', 'lidar'),
    ('radar', 'Radar range and the bumper fascia', 'vantage.core.sensor_coverage, vantage.core.materials', 'radar'),
    ('glass', 'Cameras behind glass', 'vantage.core.materials', 'glass'),
    ('weather', 'Weather and night', 'vantage.core.weather', 'weather'),
    ('sun', 'Sun position', 'vantage.core.exposure', 'sun'),
    ('suspension', 'Suspension, loads and load transfer', 'vantage.core.suspension', 'suspension'),
    ('terrain', 'Road geometry and the horizon', 'vantage.core.terrain, vantage.core.loadcases', 'terrain'),
    ('timing', 'Sync, triggers and motion blur', 'vantage.core.timing', 'timing'),
    ('gnss', 'GNSS sky view and the pose error budget', 'vantage.core.gnss', 'gnss'),
    ('calib', 'Unit lens calibration', 'vantage.core.calib', 'calib'),
    ('joint', 'Joint extrinsic solve', 'vantage.core.joint', 'joint'),
    ('observability', 'Calibration observability', 'vantage.core.observability', 'observability'),
    ('lidar_calib', 'Lidar–camera calibration', 'vantage.core.lidar_calib', 'lidar-calib'),
    ('formats', 'Dataset calibration files', 'vantage.core.exporters', 'formats'),
    ('experiments', 'Designed experiments and sensitivity', 'vantage.core.experiments', 'experiments'),
)
MODEL_IDS = tuple(m[0] for m in MODELS)
LIMITS = {'seed': (0, 2 ** 31 - 1)}


@dataclass
class ValidationSettings:
    models: tuple = ()             # model ids to run; () = every model
    quick: bool = False            # fewer Monte Carlo trials and samples: a smoke run, tolerances unchanged
    seed: int = DEFAULT_SEED

    def to_dict(self):
        return {'models': list(self.models), 'quick': self.quick, 'seed': self.seed}

    @staticmethod
    def from_dict(d) -> 'ValidationSettings':
        """Tolerant of junk: unknown model ids are dropped, the seed clamped."""
        s = ValidationSettings()
        d = d if isinstance(d, dict) else {}
        ms = d.get('models')
        if isinstance(ms, str):
            ms = [m.strip() for m in ms.split(',')]
        if isinstance(ms, (list, tuple)):
            s.models = tuple(m for m in MODEL_IDS if m in {str(x) for x in ms})
        if isinstance(d.get('quick'), bool):
            s.quick = d['quick']
        try:
            seed = int(d.get('seed', s.seed))
            lo, hi = LIMITS['seed']
            s.seed = min(max(seed, lo), hi)
        except (TypeError, ValueError, OverflowError):
            pass
        return s


@dataclass
class Spec:
    """A registered check."""
    id: str
    model: str
    title: str
    quantity: str             # what is compared, and the error figure reported
    reference: str            # the independent reference it is compared with
    tolerance: float
    unit: str                 # the error figure's unit
    refs: tuple               # keys into references.REFERENCES
    fn: object
    cost: str = 'cheap'       # 'cheap' (< 1 s) | 'study' (seconds)


@dataclass
class Outcome:
    """What a check function returns: the error figure and what backs it."""
    error: float
    detail: dict = field(default_factory=dict)
    figure: dict | None = None
    ok: bool = True           # a further condition the check requires besides error <= tolerance
    tolerance: float | None = None   # overrides the spec's (a band that depends on the sample size)
    note: str = ''


@dataclass
class Check:
    id: str
    model: str
    title: str
    quantity: str
    reference: str
    refs: tuple
    unit: str
    tolerance: float
    error: float
    passed: bool
    detail: dict
    figure: dict | None
    note: str
    seconds: float
    exception: str = ''


@dataclass
class ValidationResult:
    checks: list
    settings: ValidationSettings
    project: str
    variant: str
    started_utc: str
    seconds: float
    environment: dict
    cancelled: bool = False

    @property
    def passed(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    @property
    def failed(self) -> list:
        return [c for c in self.checks if not c.passed]

    @property
    def ok(self) -> bool:
        return bool(self.checks) and not self.failed and not self.cancelled

    def check(self, cid: str) -> Check | None:
        return next((c for c in self.checks if c.id == cid), None)

    def by_model(self) -> list:
        """[(model id, title, modules, anchor, [Check])] in MODELS order, models without checks left out."""
        out = []
        for mid, title, mods, anchor in MODELS:
            cs = [c for c in self.checks if c.model == mid]
            if cs:
                out.append((mid, title, mods, anchor, cs))
        return out


CHECKS: list = []


def check(cid, model, title, quantity, reference, tolerance, unit, refs=(), cost='cheap'):
    """Register a check function fn(ctx) -> Outcome."""
    assert model in MODEL_IDS, model

    def deco(fn):
        assert not any(s.id == cid for s in CHECKS), cid
        CHECKS.append(Spec(cid, model, title, quantity, reference, float(tolerance), unit, tuple(refs), fn, cost))
        return fn
    return deco


class Ctx:
    """What a check gets: the project the project-dependent checks look at, an engine, the seed and a scratch folder."""

    def __init__(self, project, engine, variant, settings: ValidationSettings):
        self.settings = settings
        self.quick = settings.quick
        self.given = project is not None
        # the checks never change the project they were given: they look at a copy
        self.project = project.snapshot() if project is not None else self.sample()
        if variant is not None and self.project.variant_by_uid(variant.uid) is not None:
            self.project.active_uid = variant.uid
        if engine is None:
            from ..core.analysis import Engine
            engine = Engine()
        self.engine = engine
        self._tmp = None
        self.cache = {}

    def sample(self):
        """The built-in sample project (a fresh copy per call)."""
        from ..core import samples
        return samples.tigor_sample_project()

    def rng(self, cid: str, k: int = 0) -> np.random.Generator:
        """The generator of check cid: seeded with the suite's seed and the check's id, so a check's draws do not
        depend on which other checks ran."""
        return np.random.default_rng([int(self.settings.seed), zlib.crc32(cid.encode('utf-8')), int(k)])

    def tmp(self) -> str:
        if self._tmp is None:
            self._tmp = tempfile.TemporaryDirectory(prefix='vantage-validation-')
        return self._tmp.name

    def close(self):
        if self._tmp is not None:
            self._tmp.cleanup()
            self._tmp = None


def environment() -> dict:
    """The versions the run used (no paths, nothing of the machine but its platform)."""
    from .. import __version__
    import scipy
    out = {'vantage': __version__, 'python': platform.python_version(), 'numpy': np.__version__,
           'scipy': scipy.__version__, 'platform': f'{platform.system()} {platform.machine()}'}
    try:
        import cv2
        out['opencv'] = cv2.__version__
    except Exception:                       # noqa: BLE001 — reported as missing, checks that need it fail
        out['opencv'] = None
    try:
        from PyQt6.QtCore import QT_VERSION_STR
        out['qt'] = QT_VERSION_STR
    except Exception:                       # noqa: BLE001
        out['qt'] = None
    return out


def available_models() -> tuple:
    """The model ids whose modules this installation has (the experiments are optional)."""
    import importlib.util
    return tuple(m for m in MODEL_IDS
                 if m != 'experiments' or importlib.util.find_spec('vantage.core.experiments') is not None)


def specs(settings: ValidationSettings | None = None) -> list:
    """The registered checks the settings select, in MODELS order (models whose module is not installed left out)."""
    from . import checks_optics, checks_ranging, checks_environment, checks_vehicle, checks_rig, \
        checks_calibration, checks_formats, checks_experiments  # noqa: F401  (registration)
    st = settings or ValidationSettings()
    use = (set(st.models) if st.models else set(MODEL_IDS)) & set(available_models())
    order = {m: i for i, m in enumerate(MODEL_IDS)}
    return sorted((s for s in CHECKS if s.model in use), key=lambda s: order[s.model])


def _finite(x) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def run_one(spec: Spec, ctx: Ctx) -> Check:
    t0 = time.perf_counter()
    try:
        out = spec.fn(ctx)
        tol = spec.tolerance if out.tolerance is None else float(out.tolerance)
        err = float(out.error)
        passed = _finite(err) and err <= tol and bool(out.ok)
        return Check(spec.id, spec.model, spec.title, spec.quantity, spec.reference, spec.refs, spec.unit, tol, err,
                     passed, out.detail, out.figure, out.note, time.perf_counter() - t0)
    except Exception as e:                  # noqa: BLE001 — a broken model is a failed check, not a crashed suite
        tb = traceback.format_exception(type(e), e, e.__traceback__)
        where = next((ln.strip() for ln in reversed(tb) if 'File "' in ln and 'vantage' in ln), '')
        return Check(spec.id, spec.model, spec.title, spec.quantity, spec.reference, spec.refs, spec.unit,
                     spec.tolerance, float('nan'), False, {}, None, '', time.perf_counter() - t0,
                     exception=f'{type(e).__name__}: {e}'[:400] + (f' ({where[:200]})' if where else ''))


def run(project=None, engine=None, settings=None, variant=None, progress=None, cancel=None) -> ValidationResult:
    """Run the selected checks. project: the project whose vehicle, rig and exports the project-dependent checks look
    at (None: the built-in sample); the other checks build their own references. Returns the result (cancelled=True
    with the checks done so far when cancel() turns true)."""
    import datetime as _dt
    st = settings if isinstance(settings, ValidationSettings) else ValidationSettings.from_dict(settings)
    t0 = time.perf_counter()
    started = _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')
    ctx = Ctx(project, engine, variant, st)
    sel = specs(st)
    out, cancelled = [], False
    try:
        for k, spec in enumerate(sel):
            if cancel is not None and cancel():
                cancelled = True
                break
            out.append(run_one(spec, ctx))
            if progress is not None:
                progress((k + 1) / max(len(sel), 1))
    finally:
        ctx.close()
    p = ctx.project
    return ValidationResult(out, st, getattr(p, 'name', '') if ctx.given else 'sample (built in)',
                            getattr(p.variant, 'name', '') if p is not None else '', started,
                            time.perf_counter() - t0, environment(), cancelled)


# ----------------------------------------------------------------------------- shared helpers for the checks
def series(label, x, y, kind='line'):
    """A figure series, rounded for the JSON (8 significant digits)."""
    return {'label': label, 'kind': kind, 'x': [float(f'{v:.8g}') for v in np.asarray(x, float).ravel()],
            'y': [float(f'{v:.8g}') if _finite(v) else None for v in np.asarray(y, float).ravel()]}


def figure(title, xlabel, ylabel, data, *, logy=False, hline=None, hline_label='', diag=False, kind='xy',
           categories=None):
    """A plot for the report: kind 'xy' (lines and points) or 'bars' (one value per category). hline: a reference
    line (the tolerance); diag: the y = x line (predicted against measured)."""
    return {'kind': kind, 'title': title, 'xlabel': xlabel, 'ylabel': ylabel, 'series': data, 'logy': logy,
            'hline': hline, 'hline_label': hline_label, 'diag': diag, 'categories': categories}


def rel_err(a, b) -> np.ndarray:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.abs(a - b) / np.maximum(np.abs(b), 1e-300)


def qt_app():
    """A QApplication when none runs (the cube maps and some exports paint with Qt); offscreen in a terminal."""
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        if not os.environ.get('DISPLAY') and not os.environ.get('WAYLAND_DISPLAY'):
            os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        app = QApplication(sys.argv[:1])
    return app
