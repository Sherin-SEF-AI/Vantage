"""Fleet twin: every vehicle's sensors followed over time, their drift modelled with uncertainty, the date each will
leave its tolerance forecast, knocks flagged at once, and the measured fleet drift handed back to the design's tolerances.

Observations. Everything the fleet already produces, per vehicle and sensor, as offsets from the plan in the vehicle
frame (Δx Δy Δz in mm, Δyaw Δpitch Δroll in degrees, the fleet page's convention), each with a measurement σ:
- the fleet's records (core.fleet: installation checks, extrinsics, nuScenes and Kalibr tables, variants, typed poses):
  σ is a stated default per kind (SOURCE_SIGMA, "typical", not a datasheet figure), an installation check's scaled by
  its reprojection RMS against INSTALL_RMS_PX; a record whose quality carries sigma_mm / sigma_deg uses those;
- the online self-calibration of recorded drives (core.online_calib): each segment between knocks (default) or each
  window of the smoothed series, with the σ the solve states; rotations relative to the lidar (the drive's reference),
  positions only where the solve fitted them;
- rig-health drift events (core.rig_health: the body edge N px off the plan in a live image): an indicator, not a pose;
  kept as events on the timeline and as alerts, converted to an equivalent rotation (px over the lens's pixels per
  degree at the centre) for the message only, never fed to the filter.

Model, per vehicle × sensor × axis (axes independent): the smooth local linear trend of Harvey (1989, §2.3.2): a level
L and a drift rate R, L' = L + R·dt + η (η ~ N(0, q_l·dt)), R' = R + ζ (ζ ~ N(0, q_r·dt), the rate's random walk;
the integrated noise enters the level with q_r·dt³/3), observed as y = L + ε with ε ~ N(0, (s_k·σ)²), s_k a scale per
source kind. The level starts diffuse (its first observation fixes it: as-built offsets differ from car to car); the
rate starts from a fleet prior N(μ, τ²) per sensor position and axis, so a vehicle with two records borrows strength
from the fleet. Kalman filter and Rauch-Tung-Striebel smoother (Durbin and Koopman 2012, §4.3 and §4.4), vectorised
over every series. Hyperparameters by maximum likelihood across the fleet: q_r, q_l and the source scales per sensor
kind and axis (a pattern search on their logarithms, the scales with a log-normal prior of median 1 and factor e at
one σ so a source with few records keeps its stated σ); μ and τ per sensor position and axis by the exact normal-normal
marginal likelihood of each vehicle's initial rate (its likelihood in the rate from a filter run with the rate
diffuse, an augmented constant state), the "borrowing" of empirical Bayes; a position with fewer than MIN_FLEET
vehicles takes the prior pooled over its kind.

Steps (a knock). The generalised likelihood ratio test of Willsky and Jones (1976) for a jump in the level of unknown
size at an unknown observation among the last `glr_window`: the jump's signature on the innovations is propagated
through the filter's gains, the statistic (Σ ρ ν / F)² / Σ ρ² / F is χ²(1) under no jump for each onset, and the
alarm threshold is the χ²(1) quantile at 1 − α / window (Bonferroni over the onsets), so the false-alarm probability
per axis and observation is at most α (false_alarm). An alarm whose estimated size is at least min_step of the limit
is a step: the level's variance is released at its onset (an intervention), its size and σ come from the smoother,
the test runs on. Detection delay is counted in observations from the first one after the knock.

Forecast and exceedance. From the last observation the state's forecast is Gaussian. The probability that an axis has
left its limits by a date (first passage, not the marginal at that date) is computed on a grid of level × rate: rows
of rate move the level by a whole number of cells per time step (no numerical diffusion), the rate's random walk and
the level noise are exact discrete Gaussian blurs, mass that leaves the limits is absorbed. A sensor's probability
combines its axes as independent. From it: the median and 90 % dates, and the date the probability reaches `risk`:
the recalibration date. Limits per axis, in order (limit_source auto): the critical targets' inverse tolerances
(core.critical: how far each camera's mount may move each way before a target is lost) when targets exist, else the
project's mounting tolerance (analysis.tol_deg / tol_mm), else the user's figures.

Fleet. Per sensor position and axis: the fitted rate prior (mean and spread across vehicles), every vehicle's rate
now with its σ, vehicles whose rate stands out from the others' (robust z), knocks per sensor-year, and the export of
the measured distributions as tolerance inputs: for a new vehicle, the installation offset spread, the drift after
`tol_horizon_days` (rate prior × time plus the random walks) and both together, with a core.tolerance TolSettings
built from them (normal, ± = 2 σ).

Live. A watch folder per vehicle: every new drive (a ROS 2 bag folder or a nuScenes folder) that has been quiet for
watch_quiet_s is calibrated by core.online_calib in the background and its segments added as observations, once per
drive (keyed on its path and size); rig-health snapshots add drift events.

Frames: the vehicle frame (ISO 8855, rear axle); angles intrinsic Z-Y-X, pitch positive down; dates ISO 8601, time in
days.
"""
from __future__ import annotations

import csv
import datetime as _dt
import hashlib
import io
import json
import math
import os
import threading
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from . import fleet as FL
from . import rules as R
from .mathutil import wrap180

SCHEMA = 'vantage.fleet_twin/1'
TOLERANCE_SCHEMA = 'vantage.fleet_twin.tolerances/1'
SCHEDULE_SCHEMA = 'vantage.fleet_twin.schedule/1'
METHOD_REVISION = 1
AXES = FL.COMPONENTS                         # x y z yaw pitch roll
UNITS = FL.UNITS                             # mm mm mm ° ° °
ANGULAR = np.array([False, False, False, True, True, True])
SOURCE_KINDS = FL.RECORD_KINDS + ('online_calib',)
SOURCE_LABEL = dict(FL.KIND_LABEL, online_calib='Online self-calibration')
# stated measurement σ per source kind (mm, deg): typical figures, not a datasheet's; the fit scales them per kind
SOURCE_SIGMA = {'install_check': (2.0, 0.05), 'extrinsics': (1.0, 0.03), 'nuscenes': (1.0, 0.03),
                'kalibr': (2.0, 0.05), 'variant': (5.0, 0.2), 'manual': (5.0, 0.2), 'online_calib': (5.0, 0.05)}
SOURCE_SIGMA_TEXT = (
    'Stated σ per source (typical, not a datasheet figure): installation check 2 mm / 0.05° (scaled by its RMS over '
    '0.5 px), extrinsics and nuScenes tables 1 mm / 0.03°, Kalibr 2 mm / 0.05°, variants and typed poses '
    '5 mm / 0.2°; online self-calibration: the σ of its solve. A record whose quality has sigma_mm / sigma_deg uses '
    'those. The fit then scales each source\'s σ by what the fleet\'s own scatter says.')
INSTALL_RMS_PX = 0.5
MIN_FLEET = 3                 # vehicles a sensor position needs before its own rate prior is fitted
LIMIT_SOURCES = ('auto', 'critical', 'analysis', 'fleet', 'user')
LIMIT_LABEL = {'auto': 'Automatic', 'critical': 'Critical targets', 'analysis': 'Mounting tolerance',
               'fleet': 'Fleet tolerance', 'user': 'My limits'}
POOLS = ('sensor', 'kind')
ONLINE_MODES = ('segments', 'windows')
URGENCY = ('beyond', 'due', 'soon', 'later', 'ok', 'unknown')
URGENCY_SEVERITY = {'beyond': 'critical', 'due': 'serious', 'soon': 'warning', 'later': 'good', 'ok': 'good',
                    'unknown': None}
URGENCY_LABEL = {'beyond': 'Beyond its limit', 'due': 'Recalibrate now', 'soon': 'Recalibrate soon',
                 'later': 'Within the horizon', 'ok': 'Holds', 'unknown': 'Not enough data'}
FRAME_TEXT = FL.FRAME_TEXT
ROTATION_TEXT = FL.ROTATION_TEXT
SOON_DAYS = 30.0              # a recalibration date within this many days is "soon"; past it "due"
DAY_S = 86400.0
LIMITS = {'horizon_days': (30.0, 3650.0), 'false_alarm': (1e-7, 0.05), 'glr_window': (1, 50),
          'min_step': (0.0, 10.0), 'risk': (0.001, 0.9), 'user_mm': (0.1, 1000.0), 'user_deg': (0.01, 45.0),
          'tol_horizon_days': (1.0, 3650.0), 'outlier_z': (1.0, 50.0), 'watch_quiet_s': (1.0, 3600.0)}
INTS = ('glr_window',)
ASSUMPTIONS = (
    'Each axis is modelled on its own (the cross-axis covariance a solve states is dropped). The drift is a smooth '
    'trend: a level that moves with a rate, the rate itself wandering slowly (a random walk), the level with a little '
    'noise of its own; a mount that settles in a few days and then stops is followed only as fast as the fitted '
    'random walk allows. Knocks are sudden level steps; one smaller than min_step of the limit is left to the trend. '
    'The false-alarm rate holds when the fitted model is right (Gaussian innovations); outliers in the records raise '
    'it. Online self-calibration measures each camera relative to the lidar: a lidar that moves shows as every '
    'camera stepping together. Rig-health drift (pixels of the body edge) is an indicator, shown but not used as a '
    'measurement. The exceedance probability takes the axes as independent and the limits per axis; a combined '
    'rotation limit would be tighter. Dates of synthetic or imported records are only as good as their source. '
    + SOURCE_SIGMA_TEXT)


class TwinError(ValueError):
    """A file or drive that cannot be read as observations; the message says why."""


# =========================================================================== settings
@dataclass
class FleetTwinSettings:
    variant: str = ''                  # the plan variant's uid ('' = the fleet's plan: core.fleet's, else active)
    horizon_days: float = 365.0        # forecast this far after the as-of date
    false_alarm: float = 1e-4          # a knock alarm on a step-free axis at most this often per observation
    glr_window: int = 6                # a knock is looked for among the last this many observations
    min_step: float = 0.1              # a step smaller than this share of the limit is left to the trend
    risk: float = 0.1                  # recalibrate when the probability of being beyond the limit reaches this
    limit_source: str = 'auto'         # auto | critical | analysis | fleet | user
    user_mm: float = 10.0              # my limits (limit_source user, and the fallback when nothing else has one)
    user_deg: float = 1.0
    pool: str = 'sensor'               # rate prior per sensor position ('sensor') or per kind ('kind')
    online_obs: str = 'segments'       # online self-calibration imported per segment or per window
    tol_horizon_days: float = 180.0    # the tolerance inputs: drift after this many days
    outlier_z: float = 3.0             # a vehicle's rate this many σ off the others' is named
    as_of: str = ''                    # the date "now" ('' = today)
    watch_quiet_s: float = 10.0        # a new drive in a watch folder waits this long unchanged before it is read

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d: dict | None) -> 'FleetTwinSettings':
        s = FleetTwinSettings()
        d = d if isinstance(d, dict) else {}
        if isinstance(d.get('variant'), str):
            s.variant = d['variant'][:64]
        if d.get('limit_source') in LIMIT_SOURCES:
            s.limit_source = d['limit_source']
        if d.get('pool') in POOLS:
            s.pool = d['pool']
        if d.get('online_obs') in ONLINE_MODES:
            s.online_obs = d['online_obs']
        if isinstance(d.get('as_of'), str) and (not d['as_of'].strip() or FL.parse_date(d['as_of']) is not None):
            s.as_of = d['as_of'].strip()
        for k, (lo, hi) in LIMITS.items():
            if k not in d or isinstance(d[k], bool):
                continue
            try:
                v = float(d[k])
            except (TypeError, ValueError):
                continue
            if math.isfinite(v):
                v = min(max(v, lo), hi)
                setattr(s, k, int(round(v)) if k in INTS else v)
        return s


def slot(project) -> dict:
    """project.analysis.fleet_twin (a plain dict; older files: empty)."""
    a = getattr(project, 'analysis', None)
    d = getattr(a, 'fleet_twin', None) if a is not None else None
    return d if isinstance(d, dict) else {}


def ensure_slot(project) -> dict:
    """The slot, created when missing (for edits)."""
    d = getattr(project.analysis, 'fleet_twin', None)
    if not isinstance(d, dict):
        d = project.analysis.fleet_twin = {}
    return d


def settings_of(project) -> FleetTwinSettings:
    return FleetTwinSettings.from_dict(slot(project).get('settings'))


def set_settings(project, **kw) -> FleetTwinSettings:
    """Change settings in place (clamped); the page runs it inside ctx.store.edit('analysis', ...)."""
    d = ensure_slot(project)
    st = FleetTwinSettings.from_dict(dict(settings_of(project).to_dict(), **kw))
    d['settings'] = st.to_dict()
    return st


def as_of_days(st: FleetTwinSettings) -> float:
    return FL.date_days(st.as_of) if st.as_of else FL.date_days(FL.today())


def date_text(days: float | None) -> str:
    """'YYYY-MM-DD' of a day number (fleet.date_days); '' for None."""
    if days is None or not math.isfinite(days):
        return ''
    try:
        return _dt.date.fromordinal(int(math.floor(days))).isoformat()
    except (ValueError, OverflowError):
        return ''


# =========================================================================== stored observations and events
def _fnum(v):
    return FL._fnum(v)


def clean_observation(d) -> dict | None:
    """A stored observation: {'id', 'vehicle' (fleet vehicle id), 'date', 'tag', 'kind' (a source kind), 'source',
    'plan' ({axis: the plan's value the measurement was made against}, m and deg), 'values' ({axis: [Δ, σ]}, mm and
    deg, against that plan), 'drive' (key), 'note'}. None when it holds no usable value."""
    if not isinstance(d, dict):
        return None
    vals = {}
    raw = d.get('values') if isinstance(d.get('values'), dict) else {}
    for ax, v in raw.items():
        if ax not in AXES or not isinstance(v, (list, tuple)) or len(v) != 2:
            continue
        a, s = _fnum(v[0]), _fnum(v[1])
        if a is not None and s is not None and s > 0:
            vals[ax] = [a, s]
    if not vals or not str(d.get('tag') or '').strip() or FL.parse_date(d.get('date')) is None:
        return None
    plan = {}
    if isinstance(d.get('plan'), dict):
        for ax, v in d['plan'].items():
            if ax in AXES and _fnum(v) is not None:
                plan[ax] = _fnum(v)
    kind = d.get('kind') if d.get('kind') in SOURCE_KINDS else 'online_calib'
    return {'id': str(d.get('id') or FL.new_id()), 'vehicle': str(d.get('vehicle') or ''),
            'date': FL.normalise_date(d.get('date')), 'tag': str(d['tag']).strip(), 'kind': kind,
            'source': str(d.get('source') or '')[:400], 'plan': plan, 'values': vals,
            'drive': str(d.get('drive') or '')[:600], 'note': str(d.get('note') or '')[:400]}


def observations(project) -> list:
    """The stored observations (online self-calibration imports), cleaned."""
    out = []
    for x in slot(project).get('observations') or []:
        o = clean_observation(x)
        if o is not None:
            out.append(o)
    return out


def clean_event(d) -> dict | None:
    if not isinstance(d, dict) or FL.parse_date(d.get('date')) is None or not str(d.get('tag') or '').strip():
        return None
    px = _fnum(d.get('drift_px'))
    if px is None:
        return None
    deg = _fnum(d.get('deg'))
    return {'id': str(d.get('id') or FL.new_id()), 'vehicle': str(d.get('vehicle') or ''),
            'date': FL.normalise_date(d.get('date')), 'tag': str(d['tag']).strip(), 'drift_px': px,
            'deg': deg, 'threshold_px': _fnum(d.get('threshold_px')), 'source': str(d.get('source') or '')[:400]}


def events(project) -> list:
    return [e for e in (clean_event(x) for x in slot(project).get('events') or []) if e is not None]


def add_observations(project, obs) -> int:
    """Append observations (cleaned; duplicates of a stored one, same vehicle, tag, date, drive and values, dropped)
    in place. Returns how many were added."""
    d = ensure_slot(project)
    have = d.setdefault('observations', [])
    if not isinstance(have, list):
        have = d['observations'] = []
    seen = {_obs_key(o) for o in (clean_observation(x) for x in have) if o is not None}
    n = 0
    for x in obs:
        o = clean_observation(x)
        if o is None or _obs_key(o) in seen:
            continue
        seen.add(_obs_key(o))
        have.append(o)
        n += 1
    return n


def _obs_key(o) -> tuple:
    return (o['vehicle'], o['tag'], o['date'], o['drive'],
            tuple(sorted((a, round(v[0], 9)) for a, v in o['values'].items())))


def remove_observations(project, vehicle: str = '', drive: str = '', ids=()) -> int:
    """Remove stored observations of a vehicle (all, or of one drive), or by id. Returns how many."""
    d = ensure_slot(project)
    have = d.get('observations') if isinstance(d.get('observations'), list) else []
    ids = set(ids)

    def gone(x):
        if not isinstance(x, dict):
            return True
        if ids:
            return x.get('id') in ids
        return x.get('vehicle') == vehicle and (not drive or x.get('drive') == drive)
    keep = [x for x in have if not gone(x)]
    d['observations'] = keep
    return len(have) - len(keep)


def add_events(project, evs) -> int:
    d = ensure_slot(project)
    have = d.setdefault('events', [])
    if not isinstance(have, list):
        have = d['events'] = []
    n = 0
    for x in evs:
        e = clean_event(x)
        if e is not None:
            have.append(e)
            n += 1
    return n


# =========================================================================== observations from the fleet's records
def _record_sigma(rec, tag) -> tuple:
    """(σ mm, σ deg) of one pose of a fleet record, and how it was found."""
    q = rec['quality'].get(tag) or next((v for k, v in rec['quality'].items() if k.lower() == tag.lower()), {})
    mm, deg = SOURCE_SIGMA.get(rec['kind'], SOURCE_SIGMA['manual'])
    how = f'stated default for {SOURCE_LABEL.get(rec["kind"], rec["kind"])}'
    smm, sdeg = _fnum(q.get('sigma_mm')), _fnum(q.get('sigma_deg'))
    if smm is not None and smm > 0 and sdeg is not None and sdeg > 0:
        return smm, sdeg, 'the record\'s own σ'
    rms = _fnum(q.get('rms_px'))
    if rec['kind'] == 'install_check' and rms is not None and rms > 0:
        f = min(max(rms / INSTALL_RMS_PX, 0.5), 5.0)
        return mm * f, deg * f, f'installation check, scaled by its RMS {rms:.2f} px'
    return mm, deg, how


def fleet_observations(project, plan, variant) -> dict:
    """{(vehicle id, plan tag): [obs]} from project.fleet's records (each pose all six axes), obs: {'t', 'd' (6),
    'sigma' (6, NaN where not measured), 'kind', 'source', 'date', 'ref'}."""
    vs = FL.vehicles(project)
    plan_tags = [p['tag'] for p in plan]
    by_tag = {p['tag']: p for p in plan}
    hidden = FL._hidden_tags(variant)
    out, cache = {}, {}
    for veh in vs:
        per, _, _ = FL._series(veh['records'], plan_tags, hidden, cache)
        for tag, series in per.items():
            ps = by_tag[tag]
            for days, rec, rt in series:
                if days is None:
                    continue
                d = FL.delta(ps['pose'], rec['poses'][rt])
                smm, sdeg, how = _record_sigma(rec, rt)
                sig = np.array([smm] * 3 + [sdeg] * 3, float)
                out.setdefault((veh['id'], tag), []).append(
                    {'t': days, 'd': d, 'sigma': sig, 'kind': rec['kind'], 'date': rec['date'],
                     'source': rec['source'] or SOURCE_LABEL.get(rec['kind'], ''), 'ref': f'record {rec["id"]}',
                     'sigma_how': how})
    return out


def stored_observations(project, plan) -> dict:
    """{(vehicle id, plan tag): [obs]} from the stored observations, against the twin's plan (the observation's own
    plan converted: measured = its plan + Δ)."""
    by_tag = {p['tag'].lower(): p for p in plan}
    out = {}
    for o in observations(project):
        ps = by_tag.get(o['tag'].lower())
        if ps is None:
            continue
        d = np.full(6, np.nan)
        sig = np.full(6, np.nan)
        for i, ax in enumerate(AXES):
            v = o['values'].get(ax)
            if v is None:
                continue
            base = o['plan'].get(ax, ps['pose'][ax])
            if i < 3:
                d[i] = v[0] + (base - ps['pose'][ax]) * 1000.0
            else:
                d[i] = wrap180(base + v[0] - ps['pose'][ax])
            sig[i] = v[1]
        out.setdefault((o['vehicle'], ps['tag']), []).append(
            {'t': FL.date_days(o['date']), 'd': d, 'sigma': sig, 'kind': o['kind'], 'date': o['date'],
             'source': o['source'], 'ref': f'observation {o["id"]}', 'sigma_how': 'the solve\'s σ'})
    return out


# =========================================================================== observations from online self-calibration
def _extrinsics_doc(res_or_doc) -> dict:
    """The online self-calibration's time-indexed extrinsics (vantage.online_calib.extrinsics/1) from a result, its
    JSON export, or that block itself."""
    if hasattr(res_or_doc, 'segments') and hasattr(res_or_doc, 'sensors'):
        from . import online_calib as OC
        return OC.extrinsics_doc(res_or_doc)
    if isinstance(res_or_doc, dict):
        if str(res_or_doc.get('schema', '')).startswith('vantage.online_calib.extrinsics'):
            return res_or_doc
        if isinstance(res_or_doc.get('extrinsics'), dict):
            return res_or_doc['extrinsics']
    raise TwinError('Not an online self-calibration result: use its JSON export (vantage.online_calib/1).')


def _drive_epoch_date(start_ns) -> str:
    """The drive's start as a date when its clock is a plausible UTC epoch (2000 … 2100), else ''."""
    v = _fnum(start_ns)
    if v is None:
        return ''
    s = v / 1e9
    if 946684800 <= s <= 4102444800:
        return _dt.datetime.fromtimestamp(s, _dt.timezone.utc).replace(tzinfo=None).isoformat(timespec='seconds')
    return ''


def online_observations(res_or_doc, vehicle: str, date: str = '', mode: str = 'segments', drive: str = '',
                        source: str = '') -> list:
    """Observations of one vehicle from an online self-calibration (a result or its JSON): per camera (and the lidar
    when its own state was estimated) each segment's correction at rest with its σ (mode 'segments'), or every window
    of the smoothed series (mode 'windows': consecutive windows are correlated through the smoother, so their σ
    understate the scatter; the fit's scale for the source absorbs some of it). The date: `date` (the drive's start),
    else the drive's own clock when it is a UTC epoch; the segment's middle is added to it. TwinError when there is
    nothing to import."""
    ex = _extrinsics_doc(res_or_doc)
    d0 = date or _drive_epoch_date(ex.get('start_ns'))
    t0 = FL.date_days(d0) if d0 else None
    if t0 is None:
        raise TwinError('The drive has no date: give the date it was recorded.')
    drive = drive or str(ex.get('drive') or '')
    src = source or f'online self-calibration of {ex.get("drive") or "a drive"}'
    out = []
    for tag, s in (ex.get('sensors') or {}).items():
        plan = {}
        py = s.get('plan_yaw_pitch_roll_deg') or []
        pp = s.get('plan_position_m') or []
        for i, ax in enumerate(('yaw', 'pitch', 'roll')):
            if i < len(py) and _fnum(py[i]) is not None:
                plan[ax] = float(py[i])
        for i, ax in enumerate(('x', 'y', 'z')):
            if i < len(pp) and _fnum(pp[i]) is not None:
                plan[ax] = float(pp[i])
        rows = []
        if mode == 'windows' and isinstance(s.get('interpolated'), dict):
            ip = s['interpolated']
            axes = s.get('axes') or []
            for t_s, est, sg in zip(ip.get('t_s') or [], ip.get('correction') or [], ip.get('sigma') or []):
                rows.append((t_s, dict(zip(axes, est)), dict(zip(axes, sg))))
        else:
            for sg in s.get('segments') or []:
                t_mid = 0.5 * ((_fnum(sg.get('t0_s')) or 0.0) + (_fnum(sg.get('t1_s')) or 0.0))
                rows.append((t_mid, sg.get('correction') or {}, sg.get('sigma') or {}))
        for i, (t_s, corr, sig) in enumerate(rows):
            vals = {}
            for ax, v in corr.items():
                a, e = _fnum(v), _fnum(sig.get(ax))
                if a is None or e is None or e <= 0:
                    continue
                if ax in ('x', 'y', 'z'):
                    vals[ax] = [a * 1000.0, e * 1000.0]          # m -> mm
                elif ax in AXES:
                    vals[ax] = [a, e]
            if not vals:
                continue
            day = t0 + (_fnum(t_s) or 0.0) / DAY_S
            out.append({'id': FL.new_id(), 'vehicle': vehicle, 'date': _iso_days(day), 'tag': tag,
                        'kind': 'online_calib', 'source': src, 'plan': plan, 'values': vals, 'drive': drive,
                        'note': f'{"window" if mode == "windows" else "segment"} {i + 1} of {len(rows)}; relative to '
                                f'{s.get("relative_to") or ex.get("reference") or "the lidar"}'})
    if not out:
        raise TwinError('The online self-calibration has no observable correction to import.')
    return out


def _iso_days(days: float) -> str:
    """ISO date and time (seconds) of a day number."""
    base = _dt.datetime.fromordinal(int(math.floor(days)))
    return (base + _dt.timedelta(seconds=round((days - math.floor(days)) * DAY_S))).isoformat(timespec='seconds') \
        if days != math.floor(days) else base.date().isoformat()


def read_online_file(path: str) -> dict:
    """An online self-calibration JSON export (or its extrinsics block) from a file."""
    try:
        with open(path, encoding='utf-8') as fh:
            doc = json.load(fh)
    except (OSError, json.JSONDecodeError) as e:
        raise TwinError(f'Cannot read {os.path.basename(path)}: {e}')
    _extrinsics_doc(doc)
    return doc


# =========================================================================== rig-health drift events
def rig_health_events(project, snapshot: dict, vehicle: str, date: str = '', variant=None) -> list:
    """Drift events from a rig-health snapshot (core.rig_health.snapshot / run_for): every camera whose body edge is
    beyond the drift threshold, with its pixel offset and an equivalent rotation (pixels over the lens's pixels per
    degree at the image centre: an indicator, the edge may sit off the centre)."""
    from .lens import resolve
    from .rig_health import HealthSettings
    var = variant or project.variant
    st = HealthSettings.from_dict(snapshot.get('settings') if isinstance(snapshot, dict) else None)
    cams = {c.tag: c for c in var.cameras}
    day = date or (snapshot.get('time') if isinstance(snapshot, dict) else '') or FL.today()
    out = []
    for tag, c in ((snapshot.get('cameras') or {}) if isinstance(snapshot, dict) else {}).items():
        px = _fnum(c.get('drift_px')) if isinstance(c, dict) else None
        if px is None or px <= st.drift_warn_px:
            continue
        deg = None
        cam = cams.get(tag)
        if cam is not None:
            try:
                intr = resolve(project.module_for(cam), project.lens_for(cam))
                deg = px / (intr.f_px * math.pi / 180.0)
            except Exception:                            # noqa: BLE001  a camera without a resolvable lens
                deg = None
        out.append({'id': FL.new_id(), 'vehicle': vehicle, 'date': FL.normalise_date(day), 'tag': tag,
                    'drift_px': px, 'deg': deg, 'threshold_px': st.drift_warn_px,
                    'source': 'rig health (live)'})
    return out


# =========================================================================== limits
def plan_variant(project, st: FleetTwinSettings | None = None):
    st = st or settings_of(project)
    return (project.variant_by_uid(st.variant) if st.variant else None) or FL.plan_variant(project)


def _critical_result(project, variant, engine=None, compute=False, progress=None, cancel=None):
    """core.critical's result with envelopes for the variant: cached, or (compute) run now; None without targets."""
    from . import critical as CR
    cs = CR.settings_of(project)
    if not any(t.get('enabled') for t in cs.targets):
        return None
    vg = None
    if engine is not None:
        try:
            vg = engine.vehicle_geom(project, variant)
        except Exception:                                # noqa: BLE001  the engine cannot place this variant
            vg = None
    res = CR.cached(project, variant, vg)
    if res is None and compute:
        res = CR.run(project, engine, variant=variant, progress=progress, cancel=cancel, envelopes=True)
    return res if res is not None and res.envelopes else None


def limits(project, plan, st: FleetTwinSettings, crit=None) -> dict:
    """Per plan tag: {'lo', 'hi' (6 magnitudes: Δ may go from −lo to +hi; mm and deg), 'source' (6 texts),
    'target' (6 names or '')}. limit_source auto (or critical): a camera's critical-target envelope on each side a
    target sets (how far it may move before that target is lost), else the mounting tolerance (analysis.tol_mm /
    tol_deg); analysis: the mounting tolerance only; fleet: core.fleet's install tolerance; user: the settings'
    own."""
    a = project.analysis
    analysis = (float(getattr(a, 'tol_mm', 100.0)), float(getattr(a, 'tol_deg', 5.0)))
    fs = FL.settings_of(project)
    base = {'analysis': (analysis, 'mounting tolerance'), 'fleet': ((fs.pos_tol_mm, fs.ang_tol_deg), 'fleet tolerance'),
            'user': ((st.user_mm, st.user_deg), 'my limits')}
    src = st.limit_source
    fallback = base['user'] if src == 'user' else base['fleet'] if src == 'fleet' else base['analysis']
    out = {}
    for ps in plan:
        (mm, deg), text = fallback
        lo = np.array([mm] * 3 + [deg] * 3, float)
        hi = lo.copy()
        why = [text] * 6
        tgt = [''] * 6
        if src in ('auto', 'critical') and crit is not None and ps['kind'] == 'camera':
            env = crit.camera(ps['uid'])
            if env is not None:
                for i, ax in enumerate(AXES):
                    for side, arr in ((-1, lo), (1, hi)):
                        e = (env.lo if side < 0 else env.hi).get(ax)
                        if e is None or not e[1]:            # no target sets this side
                            continue
                        v = float(e[0]) * (1000.0 if i < 3 else 1.0)
                        if math.isfinite(v) and v > 0:
                            arr[i] = v
                            why[i] = 'critical targets'
                            tgt[i] = e[1]
        out[ps['tag']] = {'lo': lo, 'hi': hi, 'source': why, 'target': tgt}
    return out


# =========================================================================== the model
UNIT = np.array([10.0, 10.0, 10.0, 1.0, 1.0, 1.0])   # internal scale: series are fitted in units of 10 mm / 1°
V_RATE0 = 1.0          # diffuse rate prior (unit/day)²: a rate of one unit a day is beyond anything real
BIG = 1.0e6            # variance (unit²) released into the level at a step
LOG_BOUNDS = {'qr': (math.log(1e-16), math.log(1e-4)), 'ql': (math.log(1e-12), math.log(1e-1)),
              'scale': (math.log(0.2), math.log(50.0))}
THETA0 = (math.log(1e-10), math.log(1e-6), 0.0)
STEP0 = (2.5, 2.5, 0.7, 0.1)
SCALE_PRIOR_SD = 1.0   # log-normal prior on a source's σ scale (natural log)
OFFSET_PRIOR_SD = 1.0  # normal prior on a source's offset (internal units: 10 mm, 1°)
HUBER_K = 3.0          # the first fit's innovations beyond this many σ count linearly (knocks not yet found)
# the source an axis's level is measured in when a sensor has several: the first present of these (the others get an
# offset each: online self-calibration is relative to the lidar, an installation check to the floor boards)
REFERENCE_ORDER = ('install_check', 'extrinsics', 'nuscenes', 'kalibr', 'variant', 'manual', 'online_calib')
NK = len(SOURCE_KINDS)
TAU_GRID = np.logspace(-9.0, -0.5, 69)          # rate-prior spreads tried (unit/day)


class _Pack:
    """Series padded to one length, values in internal units: t (days), y, v (stated variance), src (index into
    SOURCE_KINDS), m (valid), step (a level step released before this observation), n (lengths)."""

    def __init__(self, rows):
        S = len(rows)
        n = max((len(r['t']) for r in rows), default=1)
        self.S, self.n = S, max(n, 1)
        self.t = np.zeros((S, self.n))
        self.y = np.zeros((S, self.n))
        self.v = np.ones((S, self.n))
        self.src = np.zeros((S, self.n), int)
        self.m = np.zeros((S, self.n), bool)
        self.step = np.zeros((S, self.n), bool)
        self.len = np.zeros(S, int)
        for i, r in enumerate(rows):
            k = len(r['t'])
            self.len[i] = k
            self.t[i, :k], self.y[i, :k], self.v[i, :k] = r['t'], r['y'], r['v']
            self.src[i, :k] = r['src']
            self.m[i, :k] = True
            if k:
                self.t[i, k:] = r['t'][-1]


def _loglik3(pk: _Pack, qr, ql, s2, off=None, huber=None):
    """Filter with the initial rate diffuse and kept as a constant state (L, R, C = R at the first observation):
    the log-likelihood from the second observation on (the level's first observation is diffuse; a released step's
    observation is left out the same way) and each series' likelihood in its initial rate, as (estimate, variance;
    inf when the series says nothing about it)."""
    S, n = pk.S, pk.n
    Y = pk.y if off is None else pk.y - off
    L = Y[:, 0].copy()
    Rr = np.zeros(S)
    C = np.zeros(S)
    pLL = pk.v[:, 0] * s2[:, 0]
    pLR = np.zeros(S)
    pLC = np.zeros(S)
    pRR = np.full(S, V_RATE0)
    pRC = np.full(S, V_RATE0)
    pCC = np.full(S, V_RATE0)
    ll = np.zeros(S)
    for k in range(1, n):
        m = pk.m[:, k]
        if not m.any():
            break
        dt = np.where(m, pk.t[:, k] - pk.t[:, k - 1], 0.0)
        dt2 = dt * dt
        aLL = pLL + 2 * dt * pLR + dt2 * pRR + qr * dt2 * dt / 3.0 + ql * dt
        aLR = pLR + dt * pRR + qr * dt2 / 2.0
        aLC = pLC + dt * pRC
        aRR = pRR + qr * dt
        st = pk.step[:, k] & m
        aLL = aLL + np.where(st, BIG, 0.0)
        xL = L + dt * Rr
        v = Y[:, k] - xL
        F = aLL + pk.v[:, k] * s2[:, k]
        kL, kR, kC = aLL / F, aLR / F, aLC / F
        L = np.where(m, xL + kL * v, L)
        Rr = np.where(m, Rr + kR * v, Rr)
        C = np.where(m, C + kC * v, C)
        pLL = np.where(m, aLL - kL * aLL, pLL)
        pLR, pLC = np.where(m, aLR - kL * aLR, pLR), np.where(m, aLC - kL * aLC, pLC)
        pRR, pRC = np.where(m, aRR - kR * aLR, pRR), np.where(m, pRC - kR * aLC, pRC)
        pCC = np.where(m, pCC - kC * aLC, pCC)
        z2 = v * v / F
        if huber is not None:                            # robust: a gross innovation counts linearly
            az = np.sqrt(z2)
            z2 = np.where(az <= huber, z2, 2.0 * huber * az - huber * huber)
        ll += np.where(m & ~st, -0.5 * (np.log(2 * np.pi * F) + z2), 0.0)
    lam = 1.0 / np.maximum(pCC, 1e-300) - 1.0 / V_RATE0
    info = lam > 1e-6 / V_RATE0
    vhat = np.where(info, 1.0 / np.where(info, lam, 1.0), np.inf)
    rhat = np.where(info, C / np.maximum(pCC, 1e-300) * np.where(info, vhat, 0.0), 0.0)
    return ll, rhat, vhat


def _lognorm(x, mu, var):
    return -0.5 * (np.log(2 * np.pi * var) + (x - mu) ** 2 / var)


def _prior_fit(rhat, vhat, gid, G):
    """Per prior group: (μ, τ, log-likelihood) maximising Σ log N(r̂; μ, τ² + v̂) + log τ over its informative series
    (normal-normal empirical Bayes; the log τ term is the Gamma(2, 0) prior of Chung et al. 2013, Psychometrika
    78(4) 685-709, which keeps τ off zero, where plain maximum likelihood often lands with a few vehicles and would
    make every vehicle drift alike), τ on a grid then refined; groups with fewer than MIN_FLEET informative series
    keep the diffuse prior (0, ∞)."""
    ok = np.isfinite(vhat)
    mu = np.zeros(G)
    tau = np.full(G, np.inf)
    best = np.full(G, -np.inf)
    if not ok.any():
        return mu, tau, np.zeros(G)
    r, v, g = rhat[ok], vhat[ok], gid[ok]

    def at(tv):                                         # tv: (G,) τ per group -> μ, ll per group
        w = 1.0 / (tv[g] ** 2 + v)
        sw = np.bincount(g, w, G)
        m_ = np.bincount(g, w * r, G) / np.where(sw > 0, sw, 1.0)
        ll_ = np.bincount(g, _lognorm(r, m_[g], 1.0 / w), G) + np.log(np.maximum(tv, 1e-300))
        return m_, ll_
    grid = TAU_GRID
    vals = np.empty((len(grid), G))
    mus = np.empty((len(grid), G))
    for i, tv in enumerate(grid):
        mus[i], vals[i] = at(np.full(G, tv))
    k = np.argmax(vals, axis=0)
    lo = grid[np.maximum(k - 1, 0)]
    hi = grid[np.minimum(k + 1, len(grid) - 1)]
    best_t = grid[k]
    for _ in range(2):                                  # two finer grids between the best's neighbours
        fine = np.linspace(0.0, 1.0, 17)
        cand = lo[None, :] + fine[:, None] * (hi - lo)[None, :]
        cv = np.empty_like(cand)
        cm = np.empty_like(cand)
        for i in range(len(fine)):
            cm[i], cv[i] = at(cand[i])
        j = np.argmax(cv, axis=0)
        span = (hi - lo) / 16.0
        best_t = cand[j, np.arange(G)]
        lo, hi = np.maximum(best_t - span, TAU_GRID[0]), best_t + span
    tau = best_t
    mu, best = at(tau)
    n_info = np.bincount(g, minlength=G)
    pooled = n_info >= MIN_FLEET                        # a group of one keeps the diffuse prior: nothing to borrow
    tau = np.where(pooled, tau, np.inf)
    mu = np.where(pooled, mu, 0.0)
    best = np.where(pooled, best, 0.0)
    return mu, tau, best


class _Fit:
    """Hyperparameters: per noise group (sensor kind × axis) q_r, q_l and a scale per source kind (internal units);
    per prior group (sensor position or kind × axis) the rate prior μ, τ."""

    def __init__(self, G_noise, G_prior):
        self.theta = np.tile(np.array([THETA0[0], THETA0[1]] + [THETA0[2]] * NK + [0.0] * NK), (G_noise, 1))
        self.mu = np.zeros(G_prior)
        self.tau = np.full(G_prior, np.inf)
        self.loglik = np.zeros(G_noise)
        self.evals = 0
        self.iterations = 0


def _unpack(theta, pk: _Pack, ng):
    """Per series q_r, q_l; per observation its σ scale² and its source's offset."""
    th = theta[ng]
    qr, ql = np.exp(th[:, 0]), np.exp(th[:, 1])
    s2 = np.exp(2.0 * np.take_along_axis(th[:, 2:2 + NK], pk.src, axis=1))
    off = np.take_along_axis(th[:, 2 + NK:], pk.src, axis=1)
    return qr, ql, s2, off


def _series_params(fit: _Fit, pk: _Pack, ng):
    return _unpack(fit.theta, pk, ng)


def _objective(theta, pk: _Pack, ng, pg, Gn, Gp, prior_map, used, freeb=None, huber=None):
    """Per noise group: the marginal log-likelihood of its series (initial rates under their fitted prior) plus the
    scales' log-normal and the offsets' normal priors."""
    qr, ql, s2, off = _unpack(theta, pk, ng)
    ll, rhat, vhat = _loglik3(pk, qr, ql, s2, off, huber)
    mu, tau, _ = _prior_fit(rhat, vhat, prior_map, int(prior_map.max()) + 1 if len(prior_map) else 1)
    tg = tau[prior_map]
    inf = ~np.isfinite(tg) | ~np.isfinite(vhat)
    corr = np.where(inf, 0.0, _lognorm(rhat, mu[prior_map], np.where(inf, 1.0, tg ** 2 + vhat))
                    - _lognorm(rhat, 0.0, V_RATE0 + np.where(np.isfinite(vhat), vhat, 0.0)))
    tot = np.bincount(ng, ll + corr, Gn)
    tot -= 0.5 * np.sum((theta[:, 2:2 + NK] * used) ** 2, axis=1) / SCALE_PRIOR_SD ** 2
    tot -= 0.5 * np.sum(theta[:, 2 + NK:] ** 2, axis=1) / OFFSET_PRIOR_SD ** 2
    return tot, mu, tau


def _prior_groups(pg_sensor, pg_kind, rhat, vhat):
    """Per series the prior group it uses: its sensor position's when that has MIN_FLEET informative vehicles, else
    its kind's (same axis) when that has, else its own (no borrowing)."""
    ok = np.isfinite(vhat)
    ns = np.bincount(pg_sensor[ok], minlength=int(pg_sensor.max()) + 1 if len(pg_sensor) else 1)
    nk = np.bincount(pg_kind[ok], minlength=int(pg_kind.max()) + 1 if len(pg_kind) else 1)
    off = int(pg_sensor.max()) + 1 if len(pg_sensor) else 0
    out = np.where(ns[pg_sensor] >= MIN_FLEET, pg_sensor, np.where(nk[pg_kind] >= MIN_FLEET, off + pg_kind, -1))
    lone = out < 0
    base = off + (int(pg_kind.max()) + 1 if len(pg_kind) else 0)
    out[lone] = base + np.arange(int(lone.sum()))
    return out


def fit_hyper(pk: _Pack, ng, pg_sensor, pg_kind, Gn, max_iter=40, tol=0.03, huber=None) -> tuple:
    """Maximum likelihood across the fleet (see the module docstring): a pattern search over the noise groups' log
    parameters, all groups at once; the rate priors profiled inside. Returns (_Fit, prior group per series)."""
    _, rhat0, vhat0 = _loglik3(pk, np.exp(THETA0[0]), np.exp(THETA0[1]), np.ones_like(pk.y))
    pmap = _prior_groups(pg_sensor, pg_kind, rhat0, vhat0)
    Gp = int(pmap.max()) + 1 if len(pmap) else 1
    fit = _Fit(Gn, Gp)
    present = np.zeros((Gn, NK), bool)
    for i in range(pk.S):
        present[ng[i], np.unique(pk.src[i, :pk.len[i]])] = True
    used = present.astype(float)
    # an offset for every source of a group but its reference (the first of REFERENCE_ORDER it has)
    freeb = present.copy()
    for g in range(Gn):
        ref = next((SOURCE_KINDS.index(k) for k in REFERENCE_ORDER if present[g, SOURCE_KINDS.index(k)]), None)
        if ref is not None:
            freeb[g, ref] = False
    move = np.c_[np.ones((Gn, 2)), present, freeb].astype(float)
    dims = [0, 1] + [2 + k for k in range(NK) if present[:, k].any()] + \
        [2 + NK + k for k in range(NK) if freeb[:, k].any()]
    lob = np.array([LOG_BOUNDS['qr'][0], LOG_BOUNDS['ql'][0]] + [LOG_BOUNDS['scale'][0]] * NK + [-5.0] * NK)
    hib = np.array([LOG_BOUNDS['qr'][1], LOG_BOUNDS['ql'][1]] + [LOG_BOUNDS['scale'][1]] * NK + [5.0] * NK)
    step = np.tile(np.array([STEP0[0], STEP0[1]] + [STEP0[2]] * NK + [STEP0[3]] * NK), (Gn, 1))
    theta = fit.theta
    theta[:, 2 + NK:] *= freeb
    f, mu, tau = _objective(theta, pk, ng, None, Gn, Gp, pmap, used, huber=huber)
    fit.evals = 1
    for it in range(max_iter):
        fit.iterations = it + 1
        for d in dims:
            moved = np.zeros(Gn, bool)
            for sgn in (1.0, -1.0):
                th = theta.copy()
                th[:, d] = np.clip(th[:, d] + sgn * step[:, d] * move[:, d], lob[d], hib[d])
                fj, _, _ = _objective(th, pk, ng, None, Gn, Gp, pmap, used, huber=huber)
                fit.evals += 1
                better = fj > f + 1e-9
                theta[better] = th[better]
                f[better] = fj[better]
                moved |= better
            step[~moved, d] *= 0.5
        if np.all((step[:, dims] < tol) | (move[:, dims] == 0)):
            break
    fit.theta = theta
    fit.loglik, fit.mu, fit.tau = _objective(theta, pk, ng, None, Gn, Gp, pmap, used, huber=huber)
    fit.free_offset = freeb
    return fit, pmap


def _kf2(pk: _Pack, mu, var, qr, ql, s2, glr=None, keep=False, off=None):
    """The filter with the rate's prior N(mu, var) at the first observation (2 states: level, rate). glr: (threshold,
    window, after) for the step test (alarms at observations after index `after`); keep: store what the smoother
    needs. Returns a dict."""
    S, n = pk.S, pk.n
    Y = pk.y if off is None else pk.y - off
    L = Y[:, 0].copy()
    Rr = mu.astype(float).copy()
    pLL = pk.v[:, 0] * s2[:, 0]
    pLR = np.zeros(S)
    pRR = var.astype(float).copy()
    ll = np.zeros(S)
    out = {}
    if keep:
        XF = np.zeros((S, n, 2))
        PF = np.zeros((S, n, 3))
        XP = np.zeros((S, n, 2))
        PP = np.zeros((S, n, 3))
        XF[:, 0, 0], XF[:, 0, 1] = L, Rr
        PF[:, 0] = np.c_[pLL, pLR, pRR]
        XP[:, 0], PP[:, 0] = XF[:, 0], PF[:, 0]
        Z = np.zeros((S, n))
    if glr is not None:
        h, W, after = glr
        after = np.broadcast_to(np.asarray(after), (S,))
        D0 = np.zeros((S, W))
        D1 = np.zeros((S, W))
        NUM = np.zeros((S, W))
        DEN = np.zeros((S, W))
        ON = np.full((S, W), -1)
        ak = np.full(S, -1)
        aj = np.full(S, -1)
        ab = np.zeros(S)
        asb = np.zeros(S)
        al = np.zeros(S)
        LMAX = np.zeros((S, n))
    for k in range(1, n):
        m = pk.m[:, k]
        if not m.any():
            break
        dt = np.where(m, pk.t[:, k] - pk.t[:, k - 1], 0.0)
        dt2 = dt * dt
        aLL = pLL + 2 * dt * pLR + dt2 * pRR + qr * dt2 * dt / 3.0 + ql * dt
        aLR = pLR + dt * pRR + qr * dt2 / 2.0
        aRR = pRR + qr * dt
        st = pk.step[:, k] & m
        aLL = aLL + np.where(st, BIG, 0.0)
        xL = L + dt * Rr
        v = Y[:, k] - xL
        F = aLL + pk.v[:, k] * s2[:, k]
        kL, kR = aLL / F, aLR / F
        if keep:
            XP[:, k, 0] = np.where(m, xL, XF[:, k - 1, 0])
            XP[:, k, 1] = np.where(m, Rr, XF[:, k - 1, 1])
            PP[:, k] = np.where(m[:, None], np.c_[aLL, aLR, aRR], PF[:, k - 1])
            Z[:, k] = np.where(m, v / np.sqrt(F), 0.0)
        L = np.where(m, xL + kL * v, L)
        Rr = np.where(m, Rr + kR * v, Rr)
        pLL = np.where(m, aLL - kL * aLL, pLL)
        pLR = np.where(m, aLR - kL * aLR, pLR)
        pRR = np.where(m, aRR - kR * aLR, pRR)
        ll += np.where(m & ~st, -0.5 * (np.log(2 * np.pi * F) + v * v / F), 0.0)
        if keep:
            XF[:, k, 0], XF[:, k, 1] = L, Rr
            PF[:, k] = np.c_[pLL, pLR, pRR]
        if glr is not None:
            mm = m[:, None]
            # the signatures of the onsets in the window move with the state; a new onset at k; a released step
            # resets the window (the step is modelled from there on)
            D0 = np.where(mm, D0 + dt[:, None] * D1, D0)
            roll = lambda a, fill: np.concatenate([np.full((S, 1), fill, a.dtype), a[:, :-1]], axis=1)
            D0n, D1n, NUMn, DENn, ONn = roll(D0, 1.0), roll(D1, 0.0), roll(NUM, 0.0), roll(DEN, 0.0), roll(ON, k)
            D0, D1 = np.where(mm, D0n, D0), np.where(mm, D1n, D1)
            NUM, DEN, ON = np.where(mm, NUMn, NUM), np.where(mm, DENn, DEN), np.where(mm, ONn, ON)
            reset = st[:, None]
            ON = np.where(reset, -1, ON)
            act = (ON >= 0) & mm
            rho = np.where(act, D0, 0.0)
            NUM = NUM + np.where(act, rho * v[:, None] / F[:, None], 0.0)
            DEN = DEN + np.where(act, rho * rho / F[:, None], 0.0)
            D0 = np.where(act, D0 - kL[:, None] * rho, D0)
            D1 = np.where(act, D1 - kR[:, None] * rho, D1)
            with np.errstate(divide='ignore', invalid='ignore'):
                lstat = np.where(act & (DEN > 0), NUM * NUM / np.where(DEN > 0, DEN, 1.0), 0.0)
            best = np.argmax(lstat, axis=1)
            lmax = lstat[np.arange(S), best]
            LMAX[:, k] = lmax
            hit = (lmax > h) & (ak < 0) & (k > after) & m
            if hit.any():
                ak[hit] = k
                aj[hit] = ON[hit, best[hit]]
                dd = DEN[hit, best[hit]]
                ab[hit] = NUM[hit, best[hit]] / dd
                asb[hit] = 1.0 / np.sqrt(dd)
                al[hit] = lmax[hit]
    out['ll'] = ll
    out['x'] = np.c_[L, Rr]
    out['P'] = np.c_[pLL, pLR, pRR]
    if keep:
        out.update(XF=XF, PF=PF, XP=XP, PP=PP, Z=Z)
    if glr is not None:
        out.update(alarm_k=ak, alarm_j=aj, alarm_b=ab, alarm_sb=asb, alarm_l=al, lmax=LMAX)
    return out


def _smooth(pk: _Pack, kf: dict) -> dict:
    """Rauch-Tung-Striebel smoother over a kept filter run: smoothed means, covariances (LL, LR, RR) and the smoother
    gains (for the cross-covariance of consecutive states)."""
    XF, PF, XP, PP = kf['XF'], kf['PF'], kf['XP'], kf['PP']
    S, n = pk.S, pk.n
    XS = XF.copy()
    PS = PF.copy()
    Cg = np.zeros((S, n, 4))
    for k in range(n - 2, -1, -1):
        m = pk.m[:, k + 1]
        if not m.any():
            continue
        dt = pk.t[:, k + 1] - pk.t[:, k]
        f00, f01, f11 = PF[:, k, 0], PF[:, k, 1], PF[:, k, 2]
        a00, a01, a10, a11 = f00 + dt * f01, f01, f01 + dt * f11, f11        # PF Φᵀ
        p00, p01, p11 = PP[:, k + 1, 0] + 1e-15, PP[:, k + 1, 1], PP[:, k + 1, 2] + 1e-15
        det = p00 * p11 - p01 * p01
        det = np.where(np.abs(det) > 1e-300, det, 1e-300)
        i00, i01, i11 = p11 / det, -p01 / det, p00 / det
        c00, c01 = a00 * i00 + a01 * i01, a00 * i01 + a01 * i11
        c10, c11 = a10 * i00 + a11 * i01, a10 * i01 + a11 * i11
        d0 = XS[:, k + 1, 0] - XP[:, k + 1, 0]
        d1 = XS[:, k + 1, 1] - XP[:, k + 1, 1]
        e00 = PS[:, k + 1, 0] - PP[:, k + 1, 0]
        e01 = PS[:, k + 1, 1] - PP[:, k + 1, 1]
        e11 = PS[:, k + 1, 2] - PP[:, k + 1, 2]
        # C E Cᵀ
        t00, t01 = c00 * e00 + c01 * e01, c00 * e01 + c01 * e11
        t10, t11 = c10 * e00 + c11 * e01, c10 * e01 + c11 * e11
        n00 = f00 + t00 * c00 + t01 * c01
        n01 = f01 + t00 * c10 + t01 * c11
        n11 = f11 + t10 * c10 + t11 * c11
        XS[:, k, 0] = np.where(m, XF[:, k, 0] + c00 * d0 + c01 * d1, XS[:, k, 0])
        XS[:, k, 1] = np.where(m, XF[:, k, 1] + c10 * d0 + c11 * d1, XS[:, k, 1])
        PS[:, k] = np.where(m[:, None], np.c_[n00, n01, np.maximum(n11, 0.0)], PS[:, k])
        Cg[:, k] = np.where(m[:, None], np.c_[c00, c01, c10, c11], 0.0)
    return {'XS': XS, 'PS': PS, 'C': Cg}


def step_sizes(pk: _Pack, kf: dict, sm: dict, i: int, j: int) -> tuple:
    """Size and σ of the level step released before observation j of series i, from the smoother: the level at j
    minus the level carried on from j − 1."""
    XS, PS, Cg = sm['XS'], sm['PS'], sm['C']
    dt = pk.t[i, j] - pk.t[i, j - 1]
    b = XS[i, j, 0] - XS[i, j - 1, 0] - dt * XS[i, j - 1, 1]
    P1 = np.array([[PS[i, j - 1, 0], PS[i, j - 1, 1]], [PS[i, j - 1, 1], PS[i, j - 1, 2]]])
    P2 = np.array([[PS[i, j, 0], PS[i, j, 1]], [PS[i, j, 1], PS[i, j, 2]]])
    C = Cg[i, j - 1].reshape(2, 2)
    X12 = C @ P2                                         # cov(x_{j-1}, x_j | all)
    g1 = np.array([-1.0, -dt])
    g2 = np.array([1.0, 0.0])
    var = g1 @ P1 @ g1 + g2 @ P2 @ g2 + 2.0 * g1 @ X12 @ g2
    return float(b), float(math.sqrt(max(var, 0.0)))


def threshold(alpha: float, window: int) -> float:
    """The GLR alarm threshold: χ²(1) quantile at 1 − α / window (Bonferroni over the window's onsets)."""
    from scipy.stats import chi2
    return float(chi2.isf(alpha / max(int(window), 1), 1))


def detect_steps(pk: _Pack, mu, var, qr, ql, s2, min_size, alpha, window, max_passes=60, off=None) -> tuple:
    """Run the step test pass by pass: the first alarm of every series after the last one handled; an alarm whose
    size is at least min_size (internal units, per series) becomes a released step at its onset (the test goes on
    from there), the others are passed over. Returns (step flags (S, n), alarms [(series, onset j, alarm k, size,
    σ, statistic, accepted)])."""
    h = threshold(alpha, window)
    after = np.zeros(pk.S, int)
    pk.step[:] = False
    alarms = []
    for _ in range(max_passes):
        kf = _kf2(pk, mu, var, qr, ql, s2, glr=(h, window, after), off=off)
        ak, aj = kf['alarm_k'], kf['alarm_j']
        new = np.flatnonzero(ak >= 0)
        if not len(new):
            break
        for i in new:
            ok = bool(abs(kf['alarm_b'][i]) >= min_size[i] and aj[i] >= 1)
            alarms.append((int(i), int(aj[i]), int(ak[i]), float(kf['alarm_b'][i]), float(kf['alarm_sb'][i]),
                           float(kf['alarm_l'][i]), ok))
            if ok:
                pk.step[i, aj[i]] = True
                after[i] = aj[i]
            else:
                after[i] = ak[i]
    return pk.step.copy(), alarms


# =========================================================================== first passage
def _blur3(G, sig, axis, absorb):
    """Discrete Gaussian blur of variance sig² (cells) along axis 0 or 1 of a 2-D array: a 3-point kernel below 0.5
    cell (exact variance), else scipy's Gaussian; absorb: mass blurred past the ends is lost (a barrier), else
    reflected."""
    if sig <= 0:
        return G
    if sig < 0.5:
        p = 0.5 * sig * sig
        out = (1 - 2 * p) * G
        if axis == 1:
            out[:, 1:] += p * G[:, :-1]
            out[:, :-1] += p * G[:, 1:]
            if not absorb:
                out[:, 0] += p * G[:, 0]
                out[:, -1] += p * G[:, -1]
        else:
            out[1:] += p * G[:-1]
            out[:-1] += p * G[1:]
            if not absorb:
                out[0] += p * G[0]
                out[-1] += p * G[-1]
        return out
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(G, sig, axis=axis, mode='constant' if absorb else 'reflect', truncate=5.0)


def _shift_rows(G, inc):
    """Each row of G moved by its own whole number of cells (zero fill: what leaves is gone)."""
    out = np.zeros_like(G)
    nx = G.shape[1]
    for v in np.unique(inc):
        rows = inc == v
        if v == 0:
            out[rows] = G[rows]
        elif 0 < v < nx:
            out[rows, v:] = G[rows, :-v]
        elif -nx < v < 0:
            out[rows, :v] = G[rows, -v:]
    return out


BGK_BETA = 0.5826        # Broadie, Glasserman and Kou (1997): ζ(1/2)/√(2π), the discrete-monitoring correction


def first_passage(mean, cov, q_r, q_l, lo, hi, horizon, n_out=180, nx=160, nr=33, max_steps=3000):
    """P(the level has left (−lo, hi) by time t) for t on [0, horizon] (n_out + 1 times), for the smooth trend model
    from a Gaussian state (mean (level, rate), cov (LL, LR, RR)). Grid of level × rate: each rate row moves the level
    by its own whole-cell increments (Bresenham: the row's exact displacement rounded, never smeared), the rate's
    random walk and the level noise are discrete Gaussian blurs of the exact variance, mass beyond the limits is
    absorbed at every step; the limits are moved in by 0.5826 × the level noise's σ per step (the continuity
    correction of Broadie, Glasserman and Kou 1997, Mathematical Finance 7(4) 325-349, for crossings between the
    steps of the rough part; the rate's part is smooth); the level noise the rate's walk adds within one step
    (q_r·dt³/3) is left out. Returns (times, P)."""
    mL, mR = float(mean[0]), float(mean[1])
    PLL, PLR, PRR = (float(c) for c in cov)
    times = np.linspace(0.0, horizon, n_out + 1)
    A = lo + hi
    if not (A > 0 and horizon > 0):
        return times, np.ones_like(times)
    sR0 = math.sqrt(max(PRR, 0.0))
    sRH = math.sqrt(max(PRR + q_r * horizon, 0.0))
    dx = A / nx
    single = sRH * horizon < 0.5 * dx
    rmax = abs(mR) + (0.0 if single else 5.0 * sRH)
    dt = horizon / 160.0
    if rmax > 0:
        dt = min(dt, A / (20.0 * rmax))
    dt = max(dt, horizon / max_steps)
    nt = int(math.ceil(horizon / dt - 1e-9))
    dt = horizon / nt
    shrink = BGK_BETA * math.sqrt(max(q_l * dt, 0.0))
    if shrink > 0:                                       # the continuity correction (never past the middle)
        shrink = min(shrink, 0.45 * A)
        lo, hi = lo - shrink, hi - shrink
        A = lo + hi
        dx = A / nx
    # rows of rate and the initial mass
    edges = -lo + dx * np.arange(nx + 1)
    from scipy.special import ndtr
    if single:
        rates = np.array([mR])
        w = np.array([1.0])
        cond_m = np.array([mL])
        cond_s = math.sqrt(max(PLL, 0.0))
        dr = 1.0
    else:
        rates = mR + np.linspace(-5.0 * sRH, 5.0 * sRH, nr)
        dr = rates[1] - rates[0]
        if sR0 > 0.05 * dr:
            cdf = ndtr((np.r_[-np.inf, 0.5 * (rates[1:] + rates[:-1]), np.inf] - mR) / sR0)
            w = np.diff(cdf)
            c = PLR / PRR
            cond_m = mL + c * (rates - mR)
            cond_s = math.sqrt(max(PLL - PLR * c, 0.0))
        else:
            w = np.zeros(nr)
            w[nr // 2] = 1.0
            cond_m = np.full(nr, mL)
            cond_s = math.sqrt(max(PLL, 0.0))
    if cond_s > 1e-12 * A:
        Gm = np.diff(ndtr((edges[None, :] - np.atleast_1d(cond_m)[:, None]) / cond_s), axis=1)
    else:
        Gm = np.zeros((len(rates), nx))
        idx = np.floor((np.atleast_1d(cond_m) + lo) / dx).astype(int)
        inside = (idx >= 0) & (idx < nx)
        Gm[np.flatnonzero(inside), idx[inside]] = 1.0
    G = Gm * w[:, None]
    surv = np.empty(nt + 1)
    surv[0] = G.sum()
    s_rows = rates * dt / dx                               # cells per step
    sig_r = math.sqrt(max(q_r * dt, 0.0)) / dr if not single else 0.0
    sig_l = math.sqrt(max(q_l * dt, 0.0)) / dx
    prev = np.zeros(len(rates), int)
    for k in range(1, nt + 1):
        cur = np.floor(k * s_rows + 0.5).astype(int)
        inc = cur - prev
        prev = cur
        if np.any(inc):
            G = _shift_rows(G, inc)
        if sig_r > 0:
            G = _blur3(G, sig_r, 0, absorb=False)
        if sig_l > 0:
            G = _blur3(G, sig_l, 1, absorb=True)
        surv[k] = G.sum()
        if surv[k] < 1e-9:
            surv[k + 1:] = 0.0
            break
    tk = np.linspace(0.0, horizon, nt + 1)
    return times, np.clip(1.0 - np.interp(times, tk, surv), 0.0, 1.0)


def marginal_exceed(mean, cov, q_r, q_l, lo, hi, t):
    """P(the level is outside (−lo, hi) at time t) (the forecast's marginal), vectorised over t."""
    from scipy.special import ndtr
    t = np.asarray(t, float)
    m = mean[0] + mean[1] * t
    v = cov[0] + 2 * t * cov[1] + t * t * cov[2] + q_r * t ** 3 / 3.0 + q_l * t
    s = np.sqrt(np.maximum(v, 1e-300))
    return ndtr((-lo - m) / s) + ndtr((m - hi) / s)


def forecast(mean, cov, q_r, q_l, t):
    """Mean and σ of the level at times t after the state."""
    t = np.asarray(t, float)
    m = mean[0] + mean[1] * t
    v = cov[0] + 2 * t * cov[1] + t * t * cov[2] + q_r * t ** 3 / 3.0 + q_l * t
    return m, np.sqrt(np.maximum(v, 0.0))


# =========================================================================== the analysis
@dataclass
class TwinResult:
    settings: FleetTwinSettings
    plan: list                    # fleet.plan_sensors()
    variant: str
    variant_uid: str
    as_of: float                  # day number of "now"
    vehicles: list                # [{'id', 'name', 'vin', 'observations', 'first', 'last', 'worst', 'urgency'}]
    cells: dict                   # (vehicle id, tag) -> cell (see _cell)
    fleet: list                   # per (tag, axis): rate prior, vehicles' rates, outliers
    noise: list                   # per (kind, axis): fitted q_r, q_l, source scales
    findings: list                # [{'severity', 'kind', 'message', 'vehicle', 'tag', 'uids'}], worst first
    schedule: list                # recalibration rows, soonest first
    limits: dict                  # tag -> limits()
    knocks: list                  # every knock found: {'vehicle', 'tag', 'axes', 'size', 'sigma', 'date', ...}
    events: list                  # rig-health drift events
    metrics: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    def cell(self, vid, tag):
        return self.cells.get((vid, tag))

    def vehicle(self, vid):
        return next((v for v in self.vehicles if v['id'] == vid), None)

    def worst(self) -> str:
        order = {s: i for i, s in enumerate(FL.SEVERITIES)}
        return min((f['severity'] for f in self.findings), key=lambda s: order.get(s, 9), default='good')


def _sev_order(s):
    return {'critical': 0, 'serious': 1, 'warning': 2, 'good': 3}.get(s, 4)


def _interp_date(g, P, level, A):
    """The first day (as_of + g) at which P reaches level (linear between grid points); None when it does not."""
    hit = np.flatnonzero(P >= level)
    if not len(hit):
        return None
    k = int(hit[0])
    if k == 0:
        return A + float(g[0])
    f = (level - P[k - 1]) / max(P[k] - P[k - 1], 1e-12)
    return A + float(g[k - 1] + f * (g[k] - g[k - 1]))


def _key_of(project, st: FleetTwinSettings) -> str:
    """Everything the analysis reads, digested (the cache and the Issues list)."""
    v = plan_variant(project, st)
    from . import critical as CR
    try:
        crit = CR.settings_of(project).to_dict()
    except Exception:                                   # noqa: BLE001
        crit = None
    sl = {k: val for k, val in slot(project).items() if k not in ('watch',)}
    doc = {'fleet': getattr(project, 'fleet', None), 'twin': sl, 'st': st.to_dict(), 'as_of': as_of_days(st),
           'plan': FL.plan_sensors(project, v), 'tol': [project.analysis.tol_mm, project.analysis.tol_deg],
           'fleet_st': FL.settings_of(project).to_dict(), 'crit': crit, 'rev': METHOD_REVISION, 'var': v.uid}
    return hashlib.sha256(json.dumps(doc, sort_keys=True, default=str).encode()).hexdigest()


_lock = threading.Lock()
_CACHE: dict = {}


def cached(project, settings: FleetTwinSettings | None = None):
    """The last result for exactly these inputs, if any."""
    st = settings or settings_of(project)
    with _lock:
        return _CACHE.get(_key_of(project, st))


def run(project, engine=None, settings: FleetTwinSettings | None = None, variant=None, progress=None, cancel=None):
    """The fleet twin (see the module docstring). None when cancelled."""
    t_start = time.perf_counter()
    st = settings or settings_of(project)
    key = _key_of(project, st) if variant is None else None
    v = variant or plan_variant(project, st)
    plan = FL.plan_sensors(project, v)
    by_tag = {p['tag']: p for p in plan}

    def prog(f, text=''):
        if progress is not None:
            progress(min(max(f, 0.0), 1.0), text)
    notes = []
    crit = None
    if st.limit_source in ('auto', 'critical'):
        prog(0.02, 'critical targets')
        try:
            crit = _critical_result(project, v, engine, compute=True, cancel=cancel)
        except Exception as e:                           # noqa: BLE001  the study cannot run on this design
            notes.append(f'The critical targets could not be evaluated ({type(e).__name__}: {e}): the mounting '
                         'tolerance is used instead.')
        if cancel is not None and cancel():
            return None
        if crit is None and st.limit_source == 'critical':
            notes.append('No critical target with a mounting envelope: the mounting tolerance is used instead.')
    lim = limits(project, plan, st, crit)
    vs = FL.vehicles(project)
    names = {x['id']: x['name'] for x in vs}
    obs = fleet_observations(project, plan, v)
    for k_, rows in stored_observations(project, plan).items():
        if k_[0] not in names:
            continue
        obs.setdefault(k_, []).extend(rows)
    orphans = sorted({o['vehicle'] for o in observations(project)} - set(names))
    if orphans:
        notes.append(f'{len(orphans)} imported vehicle(s) are no longer in the fleet; their observations are left out.')
    A = as_of_days(st)
    H = st.horizon_days
    kinds = sorted({p['kind'] for p in plan})
    src_ix = {k: i for i, k in enumerate(SOURCE_KINDS)}
    rows, meta = [], []
    for (vid, tag), ol in obs.items():
        ol.sort(key=lambda o: o['t'])
        ps = by_tag[tag]
        for i in range(6):
            sel = [o for o in ol if np.isfinite(o['sigma'][i]) and np.isfinite(o['d'][i])]
            if not sel:
                continue
            rows.append({'t': np.array([o['t'] for o in sel]),
                         'y': np.array([o['d'][i] for o in sel]) / UNIT[i],
                         'v': (np.array([o['sigma'][i] for o in sel]) / UNIT[i]) ** 2,
                         'src': np.array([src_ix[o['kind']] for o in sel])})
            meta.append({'vehicle': vid, 'tag': tag, 'axis': i, 'kind': ps['kind'], 'obs': sel})
    prog(0.1, 'fitting the fleet')
    cells, fleet_rows, noise_rows, knocks = {}, [], [], []
    fit = None
    if rows:
        pk = _Pack(rows)
        ng_key = [(m['kind'], m['axis']) for m in meta]
        nk = sorted(set(ng_key))
        ng = np.array([nk.index(x) for x in ng_key])
        sk = sorted({(m['tag'], m['axis']) for m in meta})
        kk = sorted({(m['kind'], m['axis']) for m in meta})
        pgk = np.array([kk.index((m['kind'], m['axis'])) for m in meta])
        pgs = pgk.copy() if st.pool == 'kind' else np.array([sk.index((m['tag'], m['axis'])) for m in meta])
        lo_u = np.array([lim[m['tag']]['lo'][m['axis']] / UNIT[m['axis']] for m in meta])
        hi_u = np.array([lim[m['tag']]['hi'][m['axis']] / UNIT[m['axis']] for m in meta])
        min_size = st.min_step * np.minimum(lo_u, hi_u)
        alarms = []
        for rnd in range(2):     # a robust fit, the steps, a fit with them released, the steps again
            fit, pmap = fit_hyper(pk, ng, pgs, pgk, len(nk), huber=HUBER_K if rnd == 0 else None)
            if cancel is not None and cancel():
                return None
            qr, ql, s2, off = _series_params(fit, pk, ng)
            tau = fit.tau[pmap]
            mu = np.where(np.isfinite(tau), fit.mu[pmap], 0.0)
            var = np.where(np.isfinite(tau), tau ** 2, V_RATE0)
            _, alarms = detect_steps(pk, mu, var, qr, ql, s2, min_size, st.false_alarm, st.glr_window, off=off)
            prog(0.3 + 0.25 * rnd, 'knocks')
        kf = _kf2(pk, mu, var, qr, ql, s2, glr=(threshold(st.false_alarm, st.glr_window), st.glr_window,
                                                np.full(pk.S, pk.n)), keep=True, off=off)
        sm = _smooth(pk, kf)
        prog(0.65, 'forecasts')
        series = []
        for i, m in enumerate(meta):
            if cancel is not None and cancel():
                return None
            series.append(_series_doc(i, m, pk, kf, sm, alarms, qr, ql, s2, mu, var, tau, lo_u, hi_u, A, H, off))
            if i % 50 == 0:
                prog(0.65 + 0.25 * i / max(len(meta), 1), 'forecasts')
        # cells: one per vehicle and sensor
        per = {}
        for m, sd in zip(meta, series):
            per.setdefault((m['vehicle'], m['tag']), []).append(sd)
        g = np.linspace(0.0, H, 241)
        for (vid, tag), sds in per.items():
            cells[(vid, tag)] = _cell(vid, names.get(vid, vid), by_tag[tag], sds, g, A, st, lim[tag])
            knocks += cells[(vid, tag)]['knocks']
        fleet_rows = _fleet_stats(meta, series, fit, pmap, sk, kk, st, lim, names, pgs)
        noise_rows = [{'kind': k_, 'axis': AXES[a_], 'unit': UNITS[a_],
                       'q_rate': float(math.exp(fit.theta[g_, 0])) * UNIT[a_] ** 2,
                       'q_level': float(math.exp(fit.theta[g_, 1])) * UNIT[a_] ** 2,
                       'rate_walk_per_30d_sqrt30d': float(math.sqrt(math.exp(fit.theta[g_, 0]) * 30.0) * 30.0 *
                                                          UNIT[a_]),
                       'scales': {SOURCE_KINDS[j]: float(math.exp(fit.theta[g_, 2 + j]))
                                  for j in range(NK) if pk.m[ng == g_][pk.src[ng == g_] == j].any()},
                       'offsets': {SOURCE_KINDS[j]: float(fit.theta[g_, 2 + NK + j] * UNIT[a_])
                                   for j in range(NK) if fit.free_offset[g_, j]},
                       'reference': next((k for k in REFERENCE_ORDER if pk.m[ng == g_][
                           pk.src[ng == g_] == SOURCE_KINDS.index(k)].any()), ''),
                       'loglik': float(fit.loglik[g_]), 'series': int(np.sum(ng == g_))}
                      for g_, (k_, a_) in enumerate(nk)]
    prog(0.92, 'findings')
    rows_v = []
    for veh in vs:
        mine = [c for (vid, _), c in cells.items() if vid == veh['id']]
        n_obs = sum(c['observations'] for c in mine)
        urg = min((c['urgency'] for c in mine), key=URGENCY.index, default='unknown')
        dates = [c['first'] for c in mine if c['first'] is not None] + [c['last'] for c in mine if c['last']]
        rows_v.append({'id': veh['id'], 'name': veh['name'], 'vin': veh['vin'], 'observations': n_obs,
                       'sensors': len(mine), 'first': min(dates) if dates else None,
                       'last': max(dates) if dates else None, 'urgency': urg,
                       'next': min((c['date_risk'] for c in mine if c['date_risk'] is not None), default=None)})
    evs = [e for e in events(project) if e['vehicle'] in names]
    for e in evs:
        e['name'] = names[e['vehicle']]
    res = TwinResult(st, plan, v.name, v.uid, A, rows_v, cells, fleet_rows, noise_rows, [], [], lim,
                     sorted(knocks, key=lambda k_: (k_['days'], k_['vehicle'], k_['tag'])), evs, {}, notes)
    res.findings = _findings(res, project)
    res.schedule = _schedule(res)
    res.metrics = {'seconds': time.perf_counter() - t_start, 'series': len(meta),
                   'observations': int(sum(len(m['obs']) for m in meta)),
                   'vehicles': len(vs), 'cells': len(cells), 'knocks': len(knocks),
                   'fit_evaluations': fit.evals if fit else 0, 'fit_iterations': fit.iterations if fit else 0,
                   'limit_source': st.limit_source, 'critical': crit is not None,
                   'glr_threshold': threshold(st.false_alarm, st.glr_window)}
    if key is not None:
        with _lock:
            _CACHE[key] = res
            while len(_CACHE) > 4:
                _CACHE.pop(next(iter(_CACHE)))
    prog(1.0, 'done')
    return res


def _series_doc(i, m, pk, kf, sm, alarms, qr, ql, s2, mu, var, tau, lo_u, hi_u, A, H, off) -> dict:
    """One vehicle × sensor × axis: its observations, the smoothed level and rate with σ, its steps, its forecast
    and the probability of having left its limits (display units: mm or deg, days)."""
    ax = m['axis']
    u = UNIT[ax]
    n = int(pk.len[i])
    t = pk.t[i, :n]
    XS, PS = sm['XS'][i, :n], sm['PS'][i, :n]
    steps = []
    for (si, j, k, b, sb, lstat, ok) in alarms:
        if si != i or not ok:
            continue
        size, sig = step_sizes(pk, kf, sm, i, j)
        steps.append({'j': j, 'k': k, 'delay': k - j + 1, 'days': float(t[j]), 'after_days': float(t[j - 1]),
                      'date': date_text(t[j]), 'after': date_text(t[j - 1]), 'size': size * u, 'sigma': sig * u,
                      'glr_size': b * u, 'glr_sigma': sb * u, 'statistic': lstat,
                      'alarm_date': date_text(t[k])})
    passed = [{'j': j, 'k': k, 'size': b * u, 'statistic': lstat} for (si, j, k, b, sb, lstat, ok) in alarms
              if si == i and not ok]
    xN = kf['x'][i]
    PN = kf['P'][i]
    t_last = float(t[-1])
    span = max(A - t_last, 0.0) + H
    lo, hi = float(lo_u[i]), float(hi_u[i])
    # the probability of being beyond: the grid where it can matter, else the marginal (below 1e-5 at 1.5 σ)
    tt = np.linspace(0.0, span, 61)
    mm_, ss_ = forecast(xN, PN, qr[i], ql[i], tt)
    from scipy.special import ndtr
    loose = float(np.max(ndtr((-lo - mm_) / (1.5 * ss_ + 1e-300)) + ndtr((mm_ - hi) / (1.5 * ss_ + 1e-300))))
    forecastable = bool(np.isfinite(tau[i]) or len(np.unique(np.round(t, 6))) >= 2)
    if not forecastable:                                 # one date and no fleet to borrow a rate from: the level only
        times = np.array([0.0, span])
        P = marginal_exceed(xN, PN, 0.0, 0.0, lo, hi, np.zeros(2))
        P[:] = P[0]
        method = 'no rate yet: the level now only'
    elif loose > 1e-5:
        times, P = first_passage(xN, PN, float(qr[i]), float(ql[i]), lo, hi, span)
        method = 'grid'
    else:
        times = np.linspace(0.0, span, 181)
        P = marginal_exceed(xN, PN, qr[i], ql[i], lo, hi, times)
        method = 'marginal (far inside)'
    ft = np.linspace(0.0, span, 61)
    fm, fs = forecast(xN, PN, qr[i], ql[i], ft)
    if not forecastable:
        fm, fs = np.full_like(ft, xN[0]), np.full_like(ft, math.sqrt(max(PN[0], 0.0)))
    sig_eff = np.sqrt(pk.v[i, :n] * s2[i, :n]) * u
    return {'vehicle': m['vehicle'], 'tag': m['tag'], 'axis': AXES[ax], 'unit': UNITS[ax], 'n': n,
            't': t.tolist(), 'y': ((pk.y[i, :n] - off[i, :n]) * u).tolist(), 'sigma': sig_eff.tolist(),
            'y_raw': (pk.y[i, :n] * u).tolist(), 'offset': (off[i, :n] * u).tolist(),
            'sigma_stated': (np.sqrt(pk.v[i, :n]) * u).tolist(),
            'kinds': [SOURCE_KINDS[j] for j in pk.src[i, :n]], 'refs': [o['ref'] for o in m['obs']],
            'sources': [o['source'] for o in m['obs']], 'dates': [o['date'] for o in m['obs']],
            'level': (XS[:, 0] * u).tolist(), 'level_sd': (np.sqrt(np.maximum(PS[:, 0], 0)) * u).tolist(),
            'rate': (XS[:, 1] * u).tolist(), 'rate_sd': (np.sqrt(np.maximum(PS[:, 2], 0)) * u).tolist(),
            'steps': steps, 'passed': passed, 'glr': kf['lmax'][i, :n].tolist(),
            'now': [float(xN[0] * u), float(math.sqrt(max(PN[0], 0)) * u)],
            'rate_now': [float(xN[1] * u), float(math.sqrt(max(PN[2], 0)) * u)],
            'prior': {'mu': float(mu[i] * u), 'sd': float(math.sqrt(var[i]) * u) if np.isfinite(tau[i]) else None},
            'q_rate': float(qr[i] * u * u), 'q_level': float(ql[i] * u * u),
            't_last': t_last, 'lo': lo * u, 'hi': hi * u,
            'forecastable': forecastable,
            'forecast': {'t': (t_last + ft).tolist(), 'mean': (fm * u).tolist(), 'sd': (fs * u).tolist()},
            'p_exceed': {'t': (t_last + times).tolist(), 'p': P.tolist(), 'method': method}}


def _cell(vid, name, ps, sds, g, A, st: FleetTwinSettings, lim) -> dict:
    """One vehicle and sensor: its axes combined (independent) into the probability of being beyond a limit by each
    day after as_of; the dates; the urgency; its knocks (steps of its axes at the same observation date grouped)."""
    surv = np.ones_like(g)
    worst_ax, worst_p = None, -1.0
    for sd in sds:
        tt = np.asarray(sd['p_exceed']['t'])
        pp = np.asarray(sd['p_exceed']['p'])
        pa = np.interp(A + g, tt, pp, left=pp[0], right=pp[-1])
        surv *= 1.0 - pa
        if pa[-1] > worst_p:
            worst_ax, worst_p = sd['axis'], float(pa[-1])
    P = 1.0 - surv
    d_risk = _interp_date(g, P, st.risk, A)
    d50 = _interp_date(g, P, 0.5, A)
    d90 = _interp_date(g, P, 0.9, A)
    p_now = float(P[0])
    if p_now >= 0.5:
        urg = 'beyond'
    elif not any(sd['forecastable'] for sd in sds) and (d_risk is None or d_risk - A > 1e-9):
        urg = 'unknown'
    elif d_risk is not None and d_risk - A <= 1e-9:
        urg = 'due'
    elif d_risk is not None and d_risk - A <= SOON_DAYS:
        urg = 'soon'
    elif d_risk is not None:
        urg = 'later'
    else:
        urg = 'ok'
    knocks = {}
    for sd in sds:
        for s in sd['steps']:
            k = knocks.setdefault(s['date'], {'vehicle': vid, 'name': name, 'tag': ps['tag'], 'uid': ps['uid'],
                                              'date': s['date'], 'after': s['after'], 'days': s['days'],
                                              'alarm_date': s['alarm_date'], 'delay': s['delay'], 'axes': [],
                                              'size': {}, 'sigma': {}, 'beyond': False})
            k['axes'].append(sd['axis'])
            k['size'][sd['axis']] = s['size']
            k['sigma'][sd['axis']] = s['sigma']
            k['delay'] = min(k['delay'], s['delay'])
            k['beyond'] = bool(k['beyond'] or abs(s['size']) > min(sd['lo'], sd['hi']))
    obs_n = max(sd['n'] for sd in sds)
    dates = [d for sd in sds for d in sd['t']]
    return {'vehicle': vid, 'name': name, 'tag': ps['tag'], 'kind': ps['kind'], 'uid': ps['uid'],
            'axes': {sd['axis']: sd for sd in sds}, 'observations': obs_n,
            'first': min(dates) if dates else None, 'last': max(dates) if dates else None,
            'p_curve': {'days': g.tolist(), 'p': P.tolist()}, 'p_now': p_now, 'p_horizon': float(P[-1]),
            'date_risk': d_risk, 'date_50': d50, 'date_90': d90, 'urgency': urg, 'binding_axis': worst_ax,
            'limit_source': {sd['axis']: lim['source'][AXES.index(sd['axis'])] for sd in sds},
            'limit_target': {sd['axis']: lim['target'][AXES.index(sd['axis'])] for sd in sds},
            'knocks': sorted(knocks.values(), key=lambda k: k['days'])}


def _robust_z(x, sd, floor):
    """Per entry: (x − median of the others) / sqrt(robust σ of the others² + its own σ²), σ never under floor."""
    out = np.zeros(len(x))
    for i in range(len(x)):
        o = np.delete(x, i)
        med = float(np.median(o))
        rs = max(1.4826 * float(np.median(np.abs(o - med))), floor)
        out[i] = (x[i] - med) / math.sqrt(rs * rs + sd[i] ** 2)
    return out


def _fleet_stats(meta, series, fit, pmap, sk, kk, st, lim, names, pgs) -> list:
    out = []
    groups = {}
    for i, (m, sd) in enumerate(zip(meta, series)):
        groups.setdefault((m['tag'], m['axis']), []).append((i, m, sd))
    for (tag, ax), items in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        u = UNIT[ax]
        i0 = items[0][0]
        pg = int(pmap[i0])
        pooled = 'sensor' if st.pool == 'sensor' and pg == sk.index((tag, ax)) and np.isfinite(fit.tau[pg]) else \
            ('kind' if np.isfinite(fit.tau[pg]) else 'none')
        rates = np.array([sd['rate_now'][0] for _, _, sd in items]) * 30.0
        rsd = np.array([sd['rate_now'][1] for _, _, sd in items]) * 30.0
        levels = np.array([sd['level'][0] for _, _, sd in items])
        L = lim[tag]
        floor = 0.02 * min(L['lo'][ax], L['hi'][ax])
        outliers = []
        if len(items) >= MIN_FLEET:
            z = _robust_z(rates, rsd, floor)
            for (i, m, sd), zz, r in zip(items, z, rates):
                if abs(zz) > st.outlier_z:
                    outliers.append({'vehicle': m['vehicle'], 'name': names.get(m['vehicle'], ''), 'z': float(zz),
                                     'rate_per_30d': float(r)})
        span_y = sum(max(sd['t'][-1] - sd['t'][0], 0.0) for _, _, sd in items) / 365.25
        n_steps = sum(len(sd['steps']) for _, _, sd in items)
        out.append({'tag': tag, 'axis': AXES[ax], 'unit': UNITS[ax], 'kind': items[0][1]['kind'],
                    'vehicles': len(items), 'pooled': pooled,
                    'prior_rate_per_30d': float(fit.mu[pg] * u * 30.0) if np.isfinite(fit.tau[pg]) else None,
                    'prior_rate_sd_per_30d': float(fit.tau[pg] * u * 30.0) if np.isfinite(fit.tau[pg]) else None,
                    'rates_per_30d': rates.tolist(), 'rates_sd_per_30d': rsd.tolist(),
                    'vehicle_ids': [m['vehicle'] for _, m, _ in items],
                    'rate_mean_per_30d': float(rates.mean()), 'rate_spread_per_30d': float(rates.std(ddof=1))
                    if len(rates) > 1 else None,
                    'install_mean': float(levels.mean()), 'install_sd': float(levels.std(ddof=1))
                    if len(levels) > 1 else None,
                    'q_rate': items[0][2]['q_rate'], 'q_level': items[0][2]['q_level'],
                    'knocks': n_steps, 'observed_years': span_y,
                    'knocks_per_year': n_steps / span_y if span_y > 0 else None,
                    'knock_rms': float(math.sqrt(np.mean([s['size'] ** 2 for _, _, sd in items for s in sd['steps']])))
                    if n_steps else None,
                    'outliers': sorted(outliers, key=lambda o: -abs(o['z']))})
    return out


def _fmt(v, unit, d=None):
    if unit == 'mm':
        return f'{v:+.{1 if d is None else d}f} mm'
    return f'{v:+.{2 if d is None else d}f}°'


def _findings(res: TwinResult, project) -> list:
    out = []
    active = res.variant_uid == project.variant.uid

    def add(sev, kind, msg, vid='', tag='', uid=''):
        out.append({'severity': sev, 'kind': kind, 'message': msg, 'vehicle': vid, 'tag': tag,
                    'uids': [uid] if uid and active else []})
    if not res.plan:
        add('warning', 'plan', f'The plan, {res.variant}, has no visible camera or sensor to follow.')
    if not res.cells:
        add('warning', 'data', 'No dated measurement of any planned sensor yet: import records on the Fleet page or '
                               'online self-calibrations of recorded drives here.')
    for c in res.cells.values():
        name, tag = c['name'], c['tag']
        ax = c['binding_axis']
        sd = c['axes'].get(ax) if ax else None
        lim_txt = ''
        if sd is not None:
            src = c['limit_source'].get(ax, '')
            tgt = c['limit_target'].get(ax, '')
            lim_txt = f' (limit {sd["lo"]:.3g} … {sd["hi"]:.3g} {sd["unit"]} in {ax}, {src}' + \
                (f': {tgt}' if tgt else '') + ')'
        if c['urgency'] == 'beyond':
            add('critical', 'beyond',
                f'{name}: {tag} is probably beyond its limit now (P = {c["p_now"]:.2f}){lim_txt}.',
                c['vehicle'], tag, c['uid'])
        elif c['urgency'] in ('due', 'soon'):
            rate = sd['rate_now'][0] * 30.0 if sd else 0.0
            add('serious' if c['urgency'] == 'due' else 'warning', c['urgency'],
                f'{name}: {tag} reaches a {res.settings.risk:.0%} chance of being beyond its limit by '
                f'{date_text(c["date_risk"])}' + (f' (median {date_text(c["date_50"])})' if c['date_50'] else '') +
                (f'; {ax} moves {_fmt(rate, sd["unit"], 3)} per 30 days' if sd else '') + f'{lim_txt}: recalibrate.',
                c['vehicle'], tag, c['uid'])
        for k in c['knocks']:
            if k['days'] < res.as_of - 365.0:
                continue
            parts = ', '.join(f'{a} {_fmt(k["size"][a], "mm" if a in "xyz" else "°")} ± '
                              f'{k["sigma"][a]:.2g}' for a in k['axes'])
            add('serious' if k['beyond'] else 'warning', 'knock',
                f'{name}: {tag} knocked between {k["after"]} and {k["date"]}: {parts} (found {k["delay"]} '
                f'observation{"s" * (k["delay"] != 1)} after it).', c['vehicle'], tag, c['uid'])
    for f in res.fleet:
        for o in f['outliers']:
            add('warning', 'outlier',
                f'{o["name"]}: {f["tag"]} {f["axis"]} drifts {_fmt(o["rate_per_30d"], f["unit"], 3)} per 30 days, '
                f'{abs(o["z"]):.1f} σ off the rest of the fleet.',
                o['vehicle'], f['tag'], next((p['uid'] for p in res.plan if p['tag'] == f['tag']), ''))
    for e in res.events:
        if FL.date_days(e['date']) < res.as_of - 90.0:
            continue
        deg = f' (≈{e["deg"]:.2f}° at the image centre)' if e.get('deg') is not None else ''
        add('warning', 'rig_health',
            f'{e.get("name", "")}: rig health saw {e["tag"]}\'s body edge {e["drift_px"]:.0f} px off the plan{deg} on '
            f'{e["date"][:10]}: check it with a calibration.',
            e['vehicle'], e['tag'], next((p['uid'] for p in res.plan if p['tag'] == e['tag']), ''))
    for n in res.notes:
        add('warning', 'note', n)
    if not any(f['severity'] in ('critical', 'serious', 'warning') for f in out) and res.cells:
        add('good', 'ok', f'Every followed sensor holds within its limits for {res.settings.horizon_days:.0f} days '
                          f'at a {res.settings.risk:.0%} risk.')
    out.sort(key=lambda f: _sev_order(f['severity']))
    return out


def _schedule(res: TwinResult) -> list:
    """The recalibration schedule: every vehicle and sensor whose risk date falls within the horizon (or that is
    beyond now), soonest first."""
    rows = []
    for c in res.cells.values():
        if c['date_risk'] is None and c['urgency'] != 'beyond':
            continue
        when = c['date_risk'] if c['date_risk'] is not None else res.as_of
        rows.append({'vehicle': c['name'], 'vehicle_id': c['vehicle'], 'sensor': c['tag'], 'uid': c['uid'],
                     'recalibrate_by': date_text(max(when, res.as_of)), 'days': max(when - res.as_of, 0.0),
                     'urgency': c['urgency'], 'p_now': c['p_now'], 'median': date_text(c['date_50']),
                     'p90': date_text(c['date_90']), 'axis': c['binding_axis'] or '',
                     'limit': c['limit_source'].get(c['binding_axis'], '') if c['binding_axis'] else '',
                     'knocked': bool(c['knocks'])})
    rows.sort(key=lambda r: (r['days'], -r['p_now'], r['vehicle'], r['sensor']))
    return rows


# =========================================================================== tolerance inputs (the hand-off)
def _p95_abs(mean, sd) -> float:
    """95th percentile of |N(mean, sd²)| (bisection on the folded normal's CDF)."""
    from scipy.special import ndtr
    if sd <= 0:
        return abs(mean)
    lo, hi = 0.0, abs(mean) + 6.0 * sd
    for _ in range(60):
        x = 0.5 * (lo + hi)
        c = ndtr((x - mean) / sd) - ndtr((-x - mean) / sd)
        lo, hi = (x, hi) if c < 0.95 else (lo, x)
    return 0.5 * (lo + hi)


def tolerance_inputs(res: TwinResult, horizon_days: float | None = None, project=None) -> dict:
    """The measured fleet's pose spread, as inputs to the design's tolerance studies (core.tolerance,
    robust_perception): per planned sensor and axis, for a NEW vehicle of this fleet, its installation offset
    (the vehicles' first smoothed levels: mean, σ), its drift after horizon_days (the fitted rate prior × time, the
    rate's and the level's random walks, and the knocks as a compound Poisson process: rate per year × mean square
    size), and both together (normal: mean, σ, the 95th percentile of the magnitude). tol_settings: core.tolerance
    TolSettings fields from the cameras' totals (normal, ± = 2 σ; the largest over the cameras, since TolSettings is
    one figure for every camera); per_camera: the same per camera."""
    Hd = float(horizon_days if horizon_days is not None else res.settings.tol_horizon_days)
    sensors = {}
    for f in res.fleet:
        tag, ax = f['tag'], f['axis']
        i = AXES.index(ax)
        u = UNIT[i]
        if f['prior_rate_per_30d'] is not None:
            r_mean, r_sd, how = f['prior_rate_per_30d'] / 30.0, f['prior_rate_sd_per_30d'] / 30.0, 'fitted rate prior'
        else:
            r_mean = f['rate_mean_per_30d'] / 30.0
            r_sd = (f['rate_spread_per_30d'] or 0.0) / 30.0
            how = 'the vehicles\' own rates (too few to fit a prior)'
        lam = f['knocks_per_year'] or 0.0
        k2 = (f['knock_rms'] or 0.0) ** 2
        d_mean = r_mean * Hd
        d_var = (r_sd * Hd) ** 2 + f['q_rate'] * Hd ** 3 / 3.0 + f['q_level'] * Hd + lam * Hd / 365.25 * k2
        i_mean, i_sd = f['install_mean'], f['install_sd'] or 0.0
        t_mean, t_sd = i_mean + d_mean, math.sqrt(i_sd ** 2 + d_var)
        s = sensors.setdefault(tag, {'kind': f['kind'], 'uid': next((p['uid'] for p in res.plan if p['tag'] == tag),
                                                                    ''), 'axes': {}})
        s['vehicles'] = max(s.get('vehicles', 0), f['vehicles'])
        s['axes'][ax] = {
            'unit': f['unit'], 'vehicles': f['vehicles'],
            'install': {'mean': _r(i_mean, 5), 'sd': _r(i_sd, 5)},
            'drift': {'mean': _r(d_mean, 5), 'sd': _r(math.sqrt(d_var), 5), 'rate_per_30d': _r(r_mean * 30, 6),
                      'rate_sd_per_30d': _r(r_sd * 30, 6), 'rate_from': how,
                      'q_rate': _r(f['q_rate'], 12), 'q_level': _r(f['q_level'], 12),
                      'knocks_per_year': _r(lam, 4), 'knock_rms': _r(f['knock_rms'], 5)},
            'total': {'mean': _r(t_mean, 5), 'sd': _r(t_sd, 5), 'p95_abs': _r(_p95_abs(t_mean, t_sd), 5),
                      'distribution': 'normal'}}
    per_cam, worst = {}, {'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0, 'pos_mm': 0.0}
    for tag, s in sensors.items():
        if s['kind'] != 'camera':
            continue
        row = {}
        for ax in ('yaw', 'pitch', 'roll'):
            a = s['axes'].get(ax)
            if a is not None:
                row[ax] = _r(2.0 * a['total']['sd'], 5)
                worst[ax] = max(worst[ax], 2.0 * a['total']['sd'])
        pos = [2.0 * s['axes'][ax]['total']['sd'] for ax in ('x', 'y', 'z') if ax in s['axes']]
        if pos:
            row['pos_mm'] = _r(max(pos), 4)
            worst['pos_mm'] = max(worst['pos_mm'], max(pos))
        per_cam[tag] = row
    return {'schema': TOLERANCE_SCHEMA, 'project': getattr(project, 'name', ''), 'variant': res.variant,
            'variant_uid': res.variant_uid, 'as_of': date_text(res.as_of), 'horizon_days': Hd,
            'frame': FRAME_TEXT, 'rotation': ROTATION_TEXT, 'units': {'position': 'mm', 'angle': 'deg'},
            'meaning': f'the pose of a new vehicle\'s sensor against the plan {Hd:.0f} days after it was built, as the '
                       'measured fleet says: installation offset + drift (normal approximations)',
            'sensors': sensors,
            'tol_settings': {'yaw': _r(worst['yaw'], 5), 'pitch': _r(worst['pitch'], 5), 'roll': _r(worst['roll'], 5),
                             'pos_mm': _r(worst['pos_mm'], 4), 'distribution': 'normal',
                             'note': 'core.tolerance.TolSettings fields: ± = 2 σ of the total, the largest over the '
                                     'cameras'},
            'per_camera': per_cam,
            'method': 'core.fleet_twin: smooth local linear trend per vehicle, sensor and axis; rate prior by '
                      'empirical Bayes across the fleet; see the fleet twin JSON for every fitted figure',
            'assumptions': ASSUMPTIONS}


def tol_settings(res: TwinResult, horizon_days: float | None = None, base=None):
    """A core.tolerance.TolSettings from the measured fleet (normal, ± = 2 σ of the total after horizon_days; the
    other fields from base)."""
    from .tolerance import TolSettings
    doc = tolerance_inputs(res, horizon_days)['tol_settings']
    st = TolSettings(**(asdict(base) if base is not None else {}))
    for k in ('yaw', 'pitch', 'roll', 'pos_mm'):
        if doc[k]:
            setattr(st, k, float(doc[k]))
    st.distribution = 'normal'
    return st


def cached_tolerance_inputs(project, horizon_days: float | None = None) -> dict | None:
    """tolerance_inputs() of the cached result for the project as it is (None when the twin has not run on it)."""
    res = cached(project)
    return tolerance_inputs(res, horizon_days, project) if res is not None else None


# =========================================================================== the Issues list
def issues(project, vg=None) -> list:
    """From the cached result for exactly this project (cheap; none before the twin has run): sensors beyond their
    limit (critical), due for recalibration (serious) or soon (warning), recent knocks, fleet outliers."""
    if not slot(project) and not (isinstance(getattr(project, 'fleet', None), dict) and project.fleet.get('vehicles')):
        return []
    res = cached(project)
    if res is None:
        return []
    out = []
    for f in res.findings:
        if f['severity'] not in ('critical', 'serious', 'warning') or f['kind'] == 'note':
            continue
        name = res.vehicle(f['vehicle'])['name'] if f['vehicle'] and res.vehicle(f['vehicle']) else ''
        out.append(R.Issue(f['severity'], f'Fleet twin: {f["message"]}',
                           f'{name} · {f["tag"]}' if name and f['tag'] else (name or f['tag'] or 'fleet'),
                           list(f['uids'])))
    return out


# =========================================================================== output
def _r(v, d=4):
    if v is None:
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return round(v, d) + 0.0 if math.isfinite(v) else None


def _rl(vals, d=4):
    return None if vals is None else [_r(x, d) for x in vals]


def exit_code(res: TwinResult | None) -> int:
    """Batch: 0 every sensor holds, 3 a sensor is due (or beyond its limit, or knocked beyond it), 4 no result."""
    if res is None:
        return 4
    due = any(c['urgency'] in ('beyond', 'due') for c in res.cells.values()) or \
        any(k['beyond'] for k in res.knocks)
    return 3 if due else 0


def summary(res: TwinResult) -> dict:
    """Compact figures (the assistant, the batch log)."""
    urg = {u: sum(c['urgency'] == u for c in res.cells.values()) for u in URGENCY}
    return {'plan': res.variant, 'as_of': date_text(res.as_of), 'vehicles': len(res.vehicles),
            'followed': len(res.cells), 'observations': res.metrics.get('observations', 0),
            'horizon_days': res.settings.horizon_days, 'risk': res.settings.risk,
            'limit_source': res.settings.limit_source, 'urgency': urg,
            'schedule': [{'vehicle': r['vehicle'], 'sensor': r['sensor'], 'recalibrate_by': r['recalibrate_by'],
                          'p_now': _r(r['p_now'], 3), 'median': r['median'], 'axis': r['axis']}
                         for r in res.schedule[:12]],
            'knocks': [{'vehicle': k['name'], 'sensor': k['tag'], 'between': [k['after'], k['date']],
                        'size': {a: _r(v, 3) for a, v in k['size'].items()},
                        'delay_observations': k['delay']} for k in res.knocks[-12:]],
            'outliers': [{'vehicle': o['name'], 'sensor': f['tag'], 'axis': f['axis'],
                          'rate_per_30d': _r(o['rate_per_30d'], 4), 'z': _r(o['z'], 1)}
                         for f in res.fleet for o in f['outliers']],
            'findings': [{'severity': f['severity'], 'message': f['message']} for f in res.findings
                         if f['severity'] != 'good'][:20],
            'exit_code': exit_code(res), 'assumptions': ASSUMPTIONS}


def _series_out(sd) -> dict:
    return {'axis': sd['axis'], 'unit': sd['unit'], 'observations': [
        {'date': d, 'value': _r(y, 5), 'measured': _r(y0, 5), 'source_offset': _r(o, 5), 'sigma': _r(s, 5),
         'sigma_stated': _r(s0, 5), 'source_kind': k, 'source': src, 'ref': ref, 'level': _r(L, 5),
         'level_sd': _r(Ls, 5), 'rate_per_30d': _r(r * 30, 6), 'rate_sd_per_30d': _r(rs * 30, 6), 'glr': _r(g, 3)}
        for d, y, y0, o, s, s0, k, src, ref, L, Ls, r, rs, g in zip(
            sd['dates'], sd['y'], sd['y_raw'], sd['offset'], sd['sigma'], sd['sigma_stated'], sd['kinds'],
            sd['sources'], sd['refs'], sd['level'], sd['level_sd'], sd['rate'], sd['rate_sd'], sd['glr'])],
        'now': {'value': _r(sd['now'][0], 5), 'sd': _r(sd['now'][1], 5), 'date': date_text(sd['t_last'])},
        'rate_now_per_30d': {'value': _r(sd['rate_now'][0] * 30, 6), 'sd': _r(sd['rate_now'][1] * 30, 6)},
        'rate_prior_per_30d': {'mean': _r(sd['prior']['mu'] * 30, 6),
                               'sd': _r(sd['prior']['sd'] * 30, 6) if sd['prior']['sd'] is not None else None},
        'q_rate': _r(sd['q_rate'], 12), 'q_level': _r(sd['q_level'], 12),
        'limits': {'lo': _r(-sd['lo'], 5), 'hi': _r(sd['hi'], 5)},
        'steps': [{'between': [s['after'], s['date']], 'size': _r(s['size'], 5), 'sigma': _r(s['sigma'], 5),
                   'found_at': s['alarm_date'], 'delay_observations': s['delay'], 'statistic': _r(s['statistic'], 2)}
                  for s in sd['steps']],
        'passed_over': [{'size': _r(s['size'], 5), 'statistic': _r(s['statistic'], 2)} for s in sd['passed']],
        'forecast': {'dates': [date_text(t) for t in sd['forecast']['t'][::6]],
                     'mean': _rl(sd['forecast']['mean'][::6], 5), 'sd': _rl(sd['forecast']['sd'][::6], 5)},
        'p_exceed': {'dates': [date_text(t) for t in sd['p_exceed']['t'][::6]],
                     'p': _rl(sd['p_exceed']['p'][::6], 5), 'method': sd['p_exceed']['method']}}


def to_doc(res: TwinResult, project=None) -> dict:
    """The JSON export: settings, the fitted model, every vehicle × sensor with its axes' series, steps, forecasts and
    exceedance probabilities, the fleet statistics, the schedule, the findings, the tolerance inputs."""
    cells = []
    for (vid, tag), c in sorted(res.cells.items(), key=lambda kv: (kv[1]['name'], kv[0][1])):
        cells.append({'vehicle': c['name'], 'vehicle_id': vid, 'sensor': tag, 'kind': c['kind'],
                      'urgency': c['urgency'], 'p_now': _r(c['p_now'], 5), 'p_horizon': _r(c['p_horizon'], 5),
                      'recalibrate_by': date_text(c['date_risk']), 'median': date_text(c['date_50']),
                      'p90': date_text(c['date_90']), 'binding_axis': c['binding_axis'],
                      'limit_source': c['limit_source'], 'limit_target': c['limit_target'],
                      'p_curve': {'days_after_as_of': _rl(c['p_curve']['days'][::4], 1),
                                  'p': _rl(c['p_curve']['p'][::4], 5)},
                      'knocks': [{'between': [k['after'], k['date']], 'axes': k['axes'],
                                  'size': {a: _r(v, 5) for a, v in k['size'].items()},
                                  'sigma': {a: _r(v, 5) for a, v in k['sigma'].items()},
                                  'delay_observations': k['delay'], 'beyond_limit': k['beyond']}
                                 for k in c['knocks']],
                      'axes': {ax: _series_out(sd) for ax, sd in c['axes'].items()}})
    doc = {'schema': SCHEMA, 'method_revision': METHOD_REVISION, 'project': getattr(project, 'name', ''),
           'variant': res.variant, 'variant_uid': res.variant_uid, 'as_of': date_text(res.as_of),
           'frame': FRAME_TEXT, 'rotation': ROTATION_TEXT,
           'units': {'position': 'mm (offset from the plan)', 'angle': 'deg (offset from the plan)',
                     'rates': 'per 30 days', 'q_rate': 'unit² / day³', 'q_level': 'unit² / day', 'dates': 'ISO 8601'},
           'settings': res.settings.to_dict(), 'assumptions': ASSUMPTIONS,
           'method': {'model': 'smooth local linear trend (Harvey 1989): level, rate with a random walk; diffuse '
                               'level; rate prior N(μ, τ²) per sensor position and axis (empirical Bayes)',
                      'filter': 'Kalman filter and Rauch-Tung-Striebel smoother (Durbin and Koopman 2012)',
                      'steps': 'generalised likelihood ratio of Willsky and Jones (1976), Bonferroni threshold over '
                               'the window', 'glr_threshold': _r(res.metrics.get('glr_threshold'), 4),
                      'false_alarm_per_axis_observation': res.settings.false_alarm,
                      'exceedance': 'first passage on a level × rate grid (absorbing limits)',
                      'source_offsets': 'a constant offset per source kind (per sensor kind and axis, fitted with '
                                        'the noise) against the reference source (the first of '
                                        + ', '.join(REFERENCE_ORDER) + ' a group has): values are shown in the '
                                        'reference\'s terms (value = measured − source_offset)',
                      'source_sigma': SOURCE_SIGMA_TEXT},
           'noise': [{**n, 'q_rate': _r(n['q_rate'], 12), 'q_level': _r(n['q_level'], 12),
                      'scales': {k: _r(v, 4) for k, v in n['scales'].items()},
                      'offsets': {k: _r(v, 5) for k, v in n['offsets'].items()}, 'loglik': _r(n['loglik'], 3)}
                     for n in res.noise],
           'fleet': [{**{k: v for k, v in f.items() if k not in ('rates_per_30d', 'rates_sd_per_30d', 'vehicle_ids')},
                      'vehicles_rates_per_30d': {vid: [_r(a, 6), _r(b, 6)] for vid, a, b in
                                                 zip(f['vehicle_ids'], f['rates_per_30d'], f['rates_sd_per_30d'])}}
                     for f in res.fleet],
           'cells': cells,
           'schedule': [{k: (_r(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in res.schedule],
           'knocks': [{'vehicle': k['name'], 'sensor': k['tag'], 'between': [k['after'], k['date']],
                       'size': {a: _r(v, 5) for a, v in k['size'].items()}, 'delay_observations': k['delay']}
                      for k in res.knocks],
           'rig_health_events': [{k: e[k] for k in ('date', 'tag', 'drift_px', 'deg', 'source')} | {'vehicle': e.get(
               'name', '')} for e in res.events],
           'findings': [{'severity': f['severity'], 'kind': f['kind'], 'message': f['message']} for f in res.findings],
           'tolerance_inputs': tolerance_inputs(res, project=project),
           'metrics': {k: (_r(v, 4) if isinstance(v, float) else v) for k, v in res.metrics.items()},
           'notes': res.notes}
    return doc


CSV_COLUMNS = ('vehicle', 'sensor', 'kind', 'urgency', 'p_now', 'recalibrate_by', 'median', 'p90', 'binding_axis',
               'limit', 'observations', 'last', 'knocks') + tuple(f'{a}_now' for a in AXES) + \
    tuple(f'{a}_sd' for a in AXES) + tuple(f'{a}_rate_30d' for a in AXES)


def to_csv(res: TwinResult) -> str:
    """The fleet table: one row per vehicle and sensor."""
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(CSV_COLUMNS)
    for (vid, tag), c in sorted(res.cells.items(), key=lambda kv: (kv[1]['name'], kv[0][1])):
        now = [c['axes'][a]['now'][0] if a in c['axes'] else '' for a in AXES]
        sd = [c['axes'][a]['now'][1] if a in c['axes'] else '' for a in AXES]
        rate = [c['axes'][a]['rate_now'][0] * 30 if a in c['axes'] else '' for a in AXES]
        f = lambda v, d=4: '' if v == '' or v is None else f'{v:.{d}f}'          # noqa: E731
        w.writerow([c['name'], tag, c['kind'], c['urgency'], f(c['p_now'], 5), date_text(c['date_risk']),
                    date_text(c['date_50']), date_text(c['date_90']), c['binding_axis'] or '',
                    c['limit_source'].get(c['binding_axis'], '') if c['binding_axis'] else '', c['observations'],
                    date_text(c['last']), len(c['knocks'])] + [f(x) for x in now] + [f(x) for x in sd] +
                   [f(x, 6) for x in rate])
    return buf.getvalue()


SCHEDULE_COLUMNS = ('recalibrate_by', 'days', 'vehicle', 'sensor', 'urgency', 'p_now', 'median', 'p90', 'axis',
                    'limit', 'knocked')


def schedule_csv(res: TwinResult) -> str:
    """The recalibration schedule: which vehicle and sensor next, by date."""
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(SCHEDULE_COLUMNS)
    for r in res.schedule:
        w.writerow([r['recalibrate_by'], f'{r["days"]:.0f}', r['vehicle'], r['sensor'], r['urgency'],
                    f'{r["p_now"]:.4f}', r['median'], r['p90'], r['axis'], r['limit'], int(r['knocked'])])
    return buf.getvalue()


def chart_axis(c) -> str:
    """The axis a cell's chart shows: its largest knock's, else the one nearest its limit."""
    if c['knocks']:
        k = max(c['knocks'], key=lambda k: max(abs(v) for v in k['size'].values()))
        return max(k['size'], key=lambda a: abs(k['size'][a]) / max(min(c['axes'][a]['lo'], c['axes'][a]['hi']),
                                                                     1e-12))
    return c['binding_axis'] or next(iter(c['axes']))


URG_HEX = {'beyond': '#d03b3b', 'due': '#ec835a', 'soon': '#fab219', 'later': '#8fbf3a', 'ok': '#0ca30c',
           'unknown': '#6e7f96'}


def _bgr(hex_):
    return int(hex_[5:7], 16), int(hex_[3:5], 16), int(hex_[1:3], 16)


def png(res: TwinResult, width=1200, picks=None) -> bytes:
    """The fleet table (vehicles × sensors coloured by urgency, the recalibration date in each) and, for up to four
    cells (the given ones, else the most urgent), their binding axis over time: observations with σ, the smoothed
    level with its 2 σ band, the steps, the forecast cone and the limits. ASCII text only (OpenCV's fonts)."""
    import cv2
    from .pngtext import put_text
    bg, ink, ink2, muted, grid = (30, 29, 28), (236, 235, 230), (183, 194, 195), (129, 135, 137), (62, 61, 58)
    font, aa = cv2.FONT_HERSHEY_SIMPLEX, cv2.LINE_AA
    tags = [p['tag'] for p in res.plan]
    lab_w = 170
    cw = int(min(84, max(40, (width - lab_w - 24) / max(len(tags), 1))))
    ch = 26
    mat_h = 46 + ch * max(len(res.vehicles), 1) + 34
    order = sorted(res.cells, key=lambda k: (URGENCY.index(res.cells[k]['urgency']), not res.cells[k]['knocks'],
                                             res.cells[k]['date_risk'] or 1e18))
    pick = [k for k in list(picks or []) if k in res.cells] + [k for k in order if k not in (picks or [])]
    pick = pick[:4]
    chart_h = 200
    Hh = 70 + mat_h + (30 + chart_h * len(pick) if pick else 0) + 20
    img = np.full((Hh, width, 3), bg, np.uint8)

    def text(t, x, y, col, s=0.45):
        put_text(img, t, (int(x), int(y)), font, s, col, 1, aa)
    text(f'Fleet twin of {res.variant} as of {date_text(res.as_of)}', 16, 28, ink, 0.65)
    urg = {u: sum(c['urgency'] == u for c in res.cells.values()) for u in URGENCY}
    text(f'{len(res.vehicles)} vehicles, {len(res.cells)} followed sensors; {urg["beyond"]} beyond, {urg["due"]} due, '
         f'{urg["soon"]} soon, {urg["later"]} within {res.settings.horizon_days:.0f} days at '
         f'{res.settings.risk:.0%} risk; {len(res.knocks)} knocks', 16, 52, ink2, 0.45)
    y0 = 70
    for j, t in enumerate(tags):
        text(t[:7], lab_w + j * cw + 6, y0 + 19, ink, 0.42)
    y = y0 + 30
    for v in res.vehicles:
        text(v['name'][:22], 16, y + 18, ink, 0.45)
        for j, t in enumerate(tags):
            x = lab_w + j * cw
            c = res.cells.get((v['id'], t))
            box = (x + 2, y + 2), (x + cw - 2, y + ch - 2)
            if c is None:
                cv2.rectangle(img, *box, grid, 1)
                text('-', x + cw / 2 - 4, y + 18, muted, 0.4)
                continue
            col = _bgr(URG_HEX[c['urgency']])
            a = 0.85 if c['urgency'] in ('beyond', 'due', 'soon') else 0.45
            cv2.rectangle(img, *box, tuple(int(a * p + (1 - a) * q) for p, q in zip(col, bg)), -1)
            lab = 'now' if c['urgency'] in ('beyond', 'due') else (date_text(c['date_risk'])[2:7]
                                                                   if c['date_risk'] else 'ok')
            text(lab, x + 6, y + 17, (20, 20, 20) if a > 0.6 else ink, 0.38)
            if c['knocks']:
                cv2.circle(img, (x + cw - 7, y + 7), 3, ink, -1, aa)
        y += ch
    text(f'fill: urgency (red beyond, orange due, amber within {SOON_DAYS:.0f} days, green later or holds); text: '
         f'recalibrate-by month; dot: a knock', 16, y + 22, muted, 0.4)
    y += 44
    for key in pick:
        c = res.cells[key]
        ax = chart_axis(c)
        sd = c['axes'][ax]
        text(f'{c["name"]} - {c["tag"]} {ax} ({sd["unit"]}): {URGENCY_LABEL[c["urgency"]]}'
             + (f', recalibrate by {date_text(c["date_risk"])}' if c['date_risk'] else ''), 16, y + 14, ink, 0.5)
        X0, Y0, Wp, Hp = 70, y + 26, width - 100, chart_h - 64
        t = np.asarray(sd['t'])
        ft = np.asarray(sd['forecast']['t'])
        t0, t1 = float(t[0]), float(max(ft[-1], t[-1] + 1))
        vals = np.r_[sd['y'], np.asarray(sd['forecast']['mean']) + 2 * np.asarray(sd['forecast']['sd']),
                     np.asarray(sd['forecast']['mean']) - 2 * np.asarray(sd['forecast']['sd']), -sd['lo'], sd['hi']]
        vlo, vhi = float(np.min(vals)), float(np.max(vals))
        pad = 0.08 * max(vhi - vlo, 1e-6)
        vlo, vhi = vlo - pad, vhi + pad
        px = lambda q: X0 + (q - t0) / max(t1 - t0, 1e-9) * Wp                  # noqa: E731
        py = lambda q: Y0 + Hp - (q - vlo) / (vhi - vlo) * Hp                   # noqa: E731
        cv2.rectangle(img, (X0, Y0), (X0 + Wp, Y0 + Hp), grid, 1)
        for lim_v in (-sd['lo'], sd['hi']):
            cv2.line(img, (X0, int(py(lim_v))), (X0 + Wp, int(py(lim_v))), _bgr('#d03b3b'), 1, aa)
        lvl, lsd = np.asarray(sd['level']), np.asarray(sd['level_sd'])
        band = np.r_[[[px(a), py(b)] for a, b in zip(t, lvl + 2 * lsd)], [[px(a), py(b)] for a, b in
                                                                            zip(t[::-1], (lvl - 2 * lsd)[::-1])]]
        cv2.fillPoly(img, [band.astype(np.int32)], (70, 64, 52))
        fm, fs = np.asarray(sd['forecast']['mean']), np.asarray(sd['forecast']['sd'])
        cone = np.r_[[[px(a), py(b)] for a, b in zip(ft, fm + 2 * fs)], [[px(a), py(b)] for a, b in
                                                                          zip(ft[::-1], (fm - 2 * fs)[::-1])]]
        cv2.fillPoly(img, [cone.astype(np.int32)], (58, 50, 70))
        cv2.polylines(img, [np.array([[px(a), py(b)] for a, b in zip(t, lvl)], np.int32)], False,
                      _bgr('#539bf5'), 2, aa)
        cv2.polylines(img, [np.array([[px(a), py(b)] for a, b in zip(ft, fm)], np.int32)], False,
                      _bgr('#9085e9'), 1, aa)
        for a, b, s_ in zip(t, sd['y'], sd['sigma']):
            cv2.line(img, (int(px(a)), int(py(b - s_))), (int(px(a)), int(py(b + s_))), ink2, 1, aa)
            cv2.circle(img, (int(px(a)), int(py(b))), 3, ink, -1, aa)
        for s_ in sd['steps']:
            xs = int(px(s_['days']))
            cv2.line(img, (xs, Y0), (xs, Y0 + Hp), _bgr('#ec835a'), 1, aa)
            text(f'knock {s_["size"]:+.2f}', xs + 4, Y0 + 14, _bgr('#ec835a'), 0.38)
        xa = int(px(res.as_of))
        if X0 <= xa <= X0 + Wp:
            cv2.line(img, (xa, Y0), (xa, Y0 + Hp), muted, 1, aa)
        text(f'{vhi:.3g}', 16, Y0 + 10, muted, 0.35)
        text(f'{vlo:.3g}', 16, Y0 + Hp, muted, 0.35)
        text(date_text(t0), X0, Y0 + Hp + 16, muted, 0.35)
        text(date_text(t1), X0 + Wp - 80, Y0 + Hp + 16, muted, 0.35)
        text('dots: observations (1 sigma); blue: smoothed level (2 sigma band); violet: forecast (2 sigma cone); '
             'red: limits; orange: knocks; grey: as of', X0, Y0 + Hp + 32, muted, 0.36)
        y += chart_h
    ok, buf = cv2.imencode('.png', img)
    return buf.tobytes()


def files(res: TwinResult, project=None, picks=None) -> list:
    """[(name, bytes)]: fleet_twin.json, fleet_twin.csv (the table), fleet_twin_schedule.csv, fleet_twin.png and
    fleet_twin_tolerances.json (the hand-off to the tolerance studies)."""
    doc = to_doc(res, project)
    tol = doc['tolerance_inputs']
    if project is not None:
        from . import provenance as PV
        doc, tol = PV.stamp(doc, project), PV.stamp(tol, project)
    return [('fleet_twin.json', (json.dumps(doc, indent=1, ensure_ascii=False) + '\n').encode('utf-8')),
            ('fleet_twin.csv', to_csv(res).encode('utf-8')),
            ('fleet_twin_schedule.csv', schedule_csv(res).encode('utf-8')),
            ('fleet_twin.png', png(res, picks=picks)),
            ('fleet_twin_tolerances.json', (json.dumps(tol, indent=1, ensure_ascii=False) + '\n').encode('utf-8'))]


def export(res: TwinResult, project, out_dir, picks=None) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for name, data in files(res, project, picks):
        path = os.path.join(out_dir, name)
        with open(path, 'wb') as fh:
            fh.write(data)
        paths.append(path)
    return paths


# =========================================================================== live: watch folders
def watches(project) -> dict:
    """{vehicle id: {'folder', 'enabled'}} (cleaned)."""
    out = {}
    w = slot(project).get('watch')
    for vid, d in (w.items() if isinstance(w, dict) else []):
        if isinstance(d, dict) and isinstance(d.get('folder'), str) and d['folder'].strip():
            out[str(vid)] = {'folder': d['folder'].strip(), 'enabled': bool(d.get('enabled', True))}
    return out


def set_watch(project, vehicle: str, folder: str | None, enabled: bool = True) -> None:
    """Set (or with folder None / '' remove) a vehicle's watch folder, in place."""
    d = ensure_slot(project)
    w = d.get('watch') if isinstance(d.get('watch'), dict) else {}
    if folder:
        w[vehicle] = {'folder': os.path.abspath(os.path.expanduser(folder)), 'enabled': bool(enabled)}
    else:
        w.pop(vehicle, None)
    d['watch'] = w


def seen(project) -> dict:
    """{vehicle id: {drive key: {'drive', 'date', 'status', 'observations', 'message'}}}: drives already read."""
    s = slot(project).get('seen')
    return {str(k): dict(v) for k, v in s.items() if isinstance(v, dict)} if isinstance(s, dict) else {}


def mark_seen(project, vehicle: str, key: str, info: dict) -> None:
    d = ensure_slot(project)
    s = d.get('seen') if isinstance(d.get('seen'), dict) else {}
    s.setdefault(vehicle, {})[key] = {k: info.get(k) for k in ('drive', 'date', 'status', 'observations', 'message')}
    d['seen'] = s


def drive_paths(folder: str) -> list:
    """The drives directly in a folder: nuScenes folders (or a folder holding one, as a simulator writes them),
    ROS 2 bag folders (metadata.yaml or .mcap / .db3 files) and single .mcap / .db3 files; sorted by name."""
    from . import drivelog as DL
    out = []
    try:
        names = sorted(os.listdir(folder))
    except OSError:
        return out
    for n in names:
        if n.startswith('.'):
            continue
        q = os.path.join(folder, n)
        if os.path.isfile(q):
            if n.lower().endswith(('.mcap', '.db3')):
                out.append(q)
            continue
        if not os.path.isdir(q):
            continue
        if DL.is_nuscenes(q):
            out.append(q)
            continue
        sub = os.path.join(q, 'nuscenes')
        if os.path.isdir(sub) and DL.is_nuscenes(sub):
            out.append(sub)
            continue
        try:
            inner = os.listdir(q)
        except OSError:
            continue
        if 'metadata.yaml' in inner or any(x.lower().endswith(('.mcap', '.db3')) for x in inner):
            out.append(q)
    return out


def drive_key(path: str) -> str:
    """A drive's identity: its path and the size and time of every file the logs are read from (core.drivelog's
    log_key), digested."""
    from . import drivelog as DL
    return hashlib.sha256(repr(DL.log_key(path)).encode()).hexdigest()[:20]


def drive_date(path: str, ex: dict | None = None) -> str:
    """When a drive was recorded: its own clock when that is a UTC epoch, else the newest file's modification time."""
    if ex is not None:
        d = _drive_epoch_date(ex.get('start_ns'))
        if d:
            return d
    newest = 0.0
    for root, _, names in os.walk(path) if os.path.isdir(path) else [(os.path.dirname(path), [], [os.path.basename(
            path)])]:
        for n in names:
            try:
                newest = max(newest, os.path.getmtime(os.path.join(root, n)))
            except OSError:
                pass
    return _dt.datetime.fromtimestamp(newest or time.time()).isoformat(timespec='seconds')


class WatchState:
    """Signatures of the drives seen arriving, and when they last changed (in memory: the page or the batch keep
    one). poll() returns the drives ready to read: not read before, unchanged for `quiet` seconds."""

    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.pending = {}            # (vehicle, path) -> (signature, since)
        self.busy = set()            # (vehicle, key) being read

    def poll(self, project, quiet: float | None = None) -> list:
        """[(vehicle id, path, key)] ready now."""
        st = settings_of(project)
        quiet = st.watch_quiet_s if quiet is None else quiet
        done = seen(project)
        now = self.clock()
        ready = []
        live = set()
        for vid, w in watches(project).items():
            if not w['enabled']:
                continue
            for path in drive_paths(w['folder']):
                key = drive_key(path)
                live.add((vid, path))
                if key in done.get(vid, {}) or (vid, key) in self.busy:
                    continue
                sig, since = self.pending.get((vid, path), (None, now))
                if sig != key:
                    self.pending[(vid, path)] = (key, now)
                    continue
                if now - since >= quiet:
                    ready.append((vid, path, key))
        for k in list(self.pending):
            if k not in live:
                del self.pending[k]
        return ready


def read_drive(project, vehicle: str, path: str, engine=None, progress=None, cancel=None, settings=None,
               date: str = '') -> dict:
    """Calibrate one drive (core.online_calib with the project's settings, against the twin's plan) and return
    {'observations', 'drive', 'date', 'status', 'message', 'seconds'}: the observations are not added (the caller
    adds them, as one undo step in the app). The date: `date` when given, else the drive's own clock when it is a
    calendar time (a simulator's may not be the day it was driven), else its newest file's time. None when
    cancelled."""
    from . import online_calib as OC
    st = settings or settings_of(project)
    v = plan_variant(project, st)
    oc = OC.OnlineCalibSettings.from_dict(dict(OC.settings_of(project).to_dict(), path=path, variant_uid=v.uid))
    t0 = time.perf_counter()
    try:
        res = OC.run(project, engine, oc, v, progress, cancel)
    except (ValueError, FileNotFoundError, OSError) as e:
        return {'observations': [], 'drive': os.path.basename(path.rstrip(os.sep)), 'date': '', 'status': 'error',
                'message': str(e), 'seconds': time.perf_counter() - t0}
    if res is None:
        return None
    ex = OC.extrinsics_doc(res)
    date = date or _drive_epoch_date(ex.get('start_ns')) or drive_date(path)
    name = os.path.basename(path.rstrip(os.sep))
    if name == 'nuscenes':
        name = os.path.basename(os.path.dirname(path.rstrip(os.sep)))
    try:
        obs = online_observations(ex, vehicle, date, st.online_obs, drive=drive_key(path),
                                  source=f'online self-calibration of {name}')
        status, msg = 'ok', f'{len(obs)} observation{"s" * (len(obs) != 1)}'
    except TwinError as e:
        obs, status, msg = [], 'empty', str(e)
    knocks = [e['message'] for e in res.events if e['kind'] == 'knock']
    if knocks:
        msg += '; ' + '; '.join(knocks)
    return {'observations': obs, 'drive': name, 'date': date, 'status': status, 'message': msg,
            'seconds': time.perf_counter() - t0}


def watch_once(project, state: WatchState, engine=None, progress=None, cancel=None, quiet=None, log=None) -> list:
    """One round of the watch, headless (batch): every ready drive read, its observations added and the drive marked
    seen, in place. Returns the reports [(vehicle id, report)]."""
    out = []
    for vid, path, key in state.poll(project, quiet):
        if cancel is not None and cancel():
            break
        rep = read_drive(project, vid, path, engine, progress, cancel)
        if rep is None:
            break
        add_observations(project, rep['observations'])
        mark_seen(project, vid, key, dict(rep, observations=len(rep['observations'])))
        if log:
            log(f'{vid}: {rep["drive"]}: {rep["status"]}: {rep["message"]}')
        out.append((vid, rep))
    return out
