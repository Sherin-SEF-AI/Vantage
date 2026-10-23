"""Fleet: the same rig built on many vehicles and calibrated again over months.

A data-collection programme fits one rig design (a variant of the project: the plan) to several vehicles and measures
each vehicle's sensors again and again: the installation check after fitting, a calibration session's extrinsics, a
dataset's calibration tables. project.fleet keeps every vehicle and every measurement (a record: the measured poses of
some sensors on one date, in the vehicle frame), and this module compares them:

- Against the plan, per vehicle and sensor, from the latest record that has it: Δx Δy Δz in mm and Δyaw Δpitch Δroll
  in degrees (wrapped), the rotation between planned and measured orientation, and a verdict like the installation
  check's: ok (worst axis within the position tolerance and rotation within the angle tolerance), adjust (within three
  times) or re-mount. A sensor of the plan no record of the vehicle has is "not measured".
- Over time, per vehicle and sensor across its dated records: the change from the first to the latest, a linear trend
  per 30 days (least squares on the dates; a component counts only when its slope differs from zero at the 5 % level
  of a t-test, so it needs three dated records), the days until the trend takes the sensor out of tolerance, and a flag
  when that happens within the drift window or when the latest step is a jump (it departs from the earlier trend by
  more than jump_frac of the tolerance).
- Across the fleet, per sensor: the median Δ of the vehicles' latest records and a robust z-score per vehicle
  (median absolute deviation, scaled to a standard deviation; never below a quarter of the tolerance, so a vehicle is
  named only when it stands out by most of a tolerance, not when the others agree to a fraction of a millimetre). A
  vehicle beyond outlier_z is named ("V3: F is yawed 2.1° left of the fleet median"); one whose sensor drifts or jumped
  is named for that instead. When the fleet median itself is beyond the tolerance and most
  vehicles sit within tolerance of it, the rig is built that way: the plan should follow (median_variant).

Imports (each returns one record; tags are matched to the plan case-insensitively, unknown ones kept and reported):
the installation check's report (vantage.install_check/1, measured poses in the rear-axle frame), Vantage's
extrinsics JSON/YAML/CSV (converted back from their export frame, units and angle units), nuScenes
calibrated_sensor.json + sensor.json (camera rotations are optical frames), a Kalibr camchain (camera-to-camera only:
placed on the plan by the best rigid fit), a variant of the project, or poses typed in.

No geometry, so it is quick: run() takes about 15 ms for 5 vehicles × 4 records of 14 sensors and 75 ms for
20 × 24 (the page analyses bigger fleets on a worker thread); issues() reads only the newest records.
"""
from __future__ import annotations

import copy
import csv
import datetime as _dt
import functools
import io
import json
import math
import os
import re
from dataclasses import asdict, dataclass

import numpy as np

from . import rules as R
from .mathutil import BODY_TO_OPTICAL, euler_zyx, rot_zyx, wrap180

SCHEMA = 'vantage.fleet/1'
POSE_KEYS = ('x', 'y', 'z', 'yaw', 'pitch', 'roll')
COMPONENTS = ('x', 'y', 'z', 'yaw', 'pitch', 'roll')
UNITS = ('mm', 'mm', 'mm', '°', '°', '°')
RECORD_KINDS = ('install_check', 'extrinsics', 'nuscenes', 'kalibr', 'variant', 'manual')
KIND_LABEL = {'install_check': 'Installation check', 'extrinsics': 'Extrinsics', 'nuscenes': 'nuScenes',
              'kalibr': 'Kalibr', 'variant': 'Variant', 'manual': 'Manual'}
VERDICTS = ('ok', 'adjust', 're-mount', 'missing')
VERDICT_LABEL = {'ok': 'OK', 'adjust': 'Adjust', 're-mount': 'Re-mount', 'missing': 'Not measured'}
VERDICT_SEVERITY = {'ok': 'good', 'adjust': 'warning', 're-mount': 'serious', 'missing': None}
SEVERITIES = ('critical', 'serious', 'warning', 'good')
FRAME_TEXT = 'vehicle (ISO 8855, rear axle): x forward, y left, z up, origin at the rear-axle centre on the ground'
ROTATION_TEXT = 'intrinsic Z-Y-X yaw, pitch, roll in degrees; pitch positive down'
REMOUNT = 3.0                  # beyond this many tolerances: re-mount (as the installation check)
MIN_FLEET = 3                  # vehicles a sensor needs before outliers are named
SIGMA_FLOOR = 0.25             # a robust spread is never taken below this share of the tolerance
SIGNIFICANCE = 0.05            # a trend component counts when its slope is non-zero at this level (two-sided t-test)
MIN_RATE = 0.1                 # ... and the sensor moves at least this share of the tolerance per drift window
MAX_RECORD_BYTES = 200_000_000
ASSUMPTIONS = (
    'Deltas are the measured pose minus the plan in the vehicle frame (rear axle); the angle verdict uses the rotation '
    'between planned and measured orientation, as the installation check does. Trends are straight lines through the '
    'dated records (least squares): a real mount may settle, creep or move in steps, so a trend is a warning, not a '
    'forecast. Records are only as good as their source: an installation check measures the cameras against the floor '
    'boards; a Kalibr camchain holds camera-to-camera transforms only, so it is placed on the plan by the best rigid '
    'fit and cannot show the whole rig shifted or turned; a nuScenes table without a Vantage README is taken to have '
    'its ego origin at the rear axle on the ground.')
LIMITS = {'pos_tol_mm': (0.1, 1000.0), 'ang_tol_deg': (0.01, 45.0), 'drift_window_days': (1.0, 3650.0),
          'outlier_z': (1.0, 50.0), 'jump_frac': (0.05, 10.0)}


class FleetImportError(ValueError):
    """A file that cannot be read as a record; the message says why and what would work."""


# =========================================================================== settings
@dataclass
class FleetSettings:
    pos_tol_mm: float = 10.0          # per axis, as the installation check
    ang_tol_deg: float = 1.0          # rotation between plan and measurement
    drift_window_days: float = 30.0   # a trend that leaves the tolerance within this many days is flagged
    outlier_z: float = 3.5            # robust z-score beyond which a vehicle differs from the fleet
    jump_frac: float = 0.5            # a latest step this share of the tolerance off the earlier trend is a jump
    variant: str = ''                 # uid of the plan variant ('' = the active one)

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d: dict | None) -> 'FleetSettings':
        s = FleetSettings()
        d = d if isinstance(d, dict) else {}
        if isinstance(d.get('variant'), str):
            s.variant = d['variant']
        for k, (lo, hi) in LIMITS.items():
            if k in d:
                try:
                    v = float(d[k])
                except (TypeError, ValueError):
                    continue
                if math.isfinite(v):
                    setattr(s, k, min(max(v, lo), hi))
        return s


def settings_of(project) -> FleetSettings:
    f = getattr(project, 'fleet', None)
    return FleetSettings.from_dict(f.get('settings') if isinstance(f, dict) else None)


# =========================================================================== the data
def new_id() -> str:
    from .project import new_uid
    return new_uid()


def _fnum(v):
    """A finite float or None."""
    if type(v) is float:                                 # the common case, fast
        return v if math.isfinite(v) else None
    if isinstance(v, bool):
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def clean_pose(d) -> dict | None:
    """{x, y, z, yaw, pitch, roll} as finite floats (metres, degrees); None when any is missing or not a number."""
    if not isinstance(d, dict):
        return None
    try:
        out = {k: d[k] for k in POSE_KEYS}
    except KeyError:
        return None
    if all(type(v) is float for v in out.values()):     # the common case: a sum of finite floats is finite
        return out if math.isfinite(sum(out.values())) else None
    out = {k: _fnum(v) for k, v in out.items()}
    return None if None in out.values() else out


def _date_text(v) -> str:
    return v.strip() if isinstance(v, str) else ''


def clean_record(d, fallback_id='') -> dict | None:
    """A record with every field of the right type; poses that are not six finite numbers are dropped."""
    if not isinstance(d, dict):
        return None
    poses = {}
    raw = d.get('poses') if isinstance(d.get('poses'), dict) else {}
    for tag, pose in raw.items():
        q = clean_pose(pose)
        if q is not None and str(tag).strip():
            poses[str(tag).strip()] = q
    quality = {}
    if isinstance(d.get('quality'), dict):
        for tag, q in d['quality'].items():
            if isinstance(q, dict):
                quality[str(tag)] = {str(k): v for k, v in q.items()
                                     if isinstance(v, (str, int, float, bool))
                                     and not (isinstance(v, float) and not math.isfinite(v))}
    kind = d.get('kind') if d.get('kind') in RECORD_KINDS else 'manual'
    return {'id': str(d.get('id') or fallback_id or new_id()), 'date': _date_text(d.get('date')),
            'source': str(d.get('source') or ''), 'kind': kind, 'poses': poses, 'quality': quality,
            'note': str(d.get('note') or '')}


def _unique(i, taken) -> str:
    while i in taken:
        i = f'{i}+'
    taken.add(i)
    return i


def clean_vehicle(d, fallback_id='') -> dict | None:
    """A vehicle with every field of the right type; a record or vehicle without an id gets one derived from its place
    (stable: the same data gives the same ids)."""
    if not isinstance(d, dict):
        return None
    vid = str(d.get('id') or fallback_id or new_id())
    recs, taken = [], set()
    for j, x in enumerate(d.get('records') if isinstance(d.get('records'), list) else []):
        r = clean_record(x, f'{vid}.r{j + 1}')
        if r is not None:
            r['id'] = _unique(r['id'], taken)
            recs.append(r)
    return {'id': vid, 'name': str(d.get('name') or 'Vehicle').strip() or 'Vehicle', 'vin': str(d.get('vin') or ''),
            'note': str(d.get('note') or ''), 'records': recs}


def vehicles(project_or_fleet) -> list:
    """The fleet's vehicles, cleaned (a copy: tolerant of anything a hand-edited project holds)."""
    f = project_or_fleet if isinstance(project_or_fleet, dict) else getattr(project_or_fleet, 'fleet', None)
    vs = f.get('vehicles') if isinstance(f, dict) else None
    out, taken = [], set()
    for i, x in enumerate(vs if isinstance(vs, list) else []):
        v = clean_vehicle(x, f'v{i + 1}')
        if v is not None:
            v['id'] = _unique(v['id'], taken)
            out.append(v)
    return out


def _ensure(fleet: dict) -> list:
    """The fleet dict's vehicle list, cleaned in place (the editing helpers below work on it). Already clean entries
    keep their identity, so a dict looked up before stays the one in the list."""
    clean = vehicles(fleet)
    vs = fleet.get('vehicles')
    if not isinstance(vs, list):
        vs = fleet['vehicles'] = []
    if len(vs) != len(clean) or any(a != b for a, b in zip(vs, clean)):
        vs[:] = clean
    return vs


def vehicle(fleet: dict, vid: str) -> dict | None:
    return next((v for v in _ensure(fleet) if v['id'] == vid), None)


def record(fleet: dict, vid: str, rid: str) -> dict | None:
    v = vehicle(fleet, vid)
    return next((r for r in v['records'] if r['id'] == rid), None) if v else None


def next_name(fleet: dict) -> str:
    taken = {v['name'].upper() for v in _ensure(fleet)}
    i = 1
    while f'V{i}' in taken:
        i += 1
    return f'V{i}'


# ---- edits: each changes the fleet dict in place (the page runs them inside ctx.store.edit('fleet', ...))
def add_vehicle(fleet: dict, name: str = '', vin: str = '', note: str = '') -> dict:
    name = name.strip() or next_name(fleet)
    vs = _ensure(fleet)
    v = {'id': new_id(), 'name': name, 'vin': vin.strip(), 'note': note, 'records': []}
    vs.append(v)
    return v


def set_vehicle(fleet: dict, vid: str, **kw) -> bool:
    v = vehicle(fleet, vid)
    if v is None:
        return False
    fallback = next_name(fleet) if 'name' in kw and not str(kw['name']).strip() else ''
    for k in ('name', 'vin', 'note'):
        if k in kw:
            v[k] = str(kw[k]).strip() if k != 'note' else str(kw[k])
    if not v['name']:
        v['name'] = fallback or v['name'] or 'Vehicle'
    return True


def remove_vehicle(fleet: dict, vid: str) -> bool:
    vs = _ensure(fleet)
    n = len(vs)
    fleet['vehicles'] = [v for v in vs if v['id'] != vid]
    return len(fleet['vehicles']) < n


def add_record(fleet: dict, vid: str, rec: dict) -> dict | None:
    v = vehicle(fleet, vid)
    r = clean_record(rec)
    if v is None or r is None:
        return None
    if any(x['id'] == r['id'] for x in v['records']):
        r['id'] = new_id()
    v['records'].append(r)
    return r


def remove_record(fleet: dict, vid: str, rid: str) -> bool:
    v = vehicle(fleet, vid)
    if v is None:
        return False
    n = len(v['records'])
    v['records'] = [r for r in v['records'] if r['id'] != rid]
    return len(v['records']) < n


def set_record(fleet: dict, vid: str, rid: str, **kw) -> bool:
    """Change a record's date (checked: a date the analysis can place), source or note."""
    r = record(fleet, vid, rid)
    if r is None:
        return False
    if 'date' in kw:
        r['date'] = normalise_date(kw['date'])
    for k in ('source', 'note'):
        if k in kw:
            r[k] = str(kw[k])
    return True


def set_pose(fleet: dict, vid: str, rid: str, tag: str, pose: dict | None) -> bool:
    """Set (or with None remove) one sensor's pose in a record; the pose must be six finite numbers."""
    r = record(fleet, vid, rid)
    if r is None:
        return False
    if pose is None:
        hit = next((t for t in r['poses'] if t.lower() == tag.lower()), None)
        if hit is None:
            return False
        del r['poses'][hit]
        return True
    q = clean_pose(pose)
    if q is None:
        raise ValueError('A pose needs x, y, z in metres and yaw, pitch, roll in degrees, all numbers.')
    hit = next((t for t in r['poses'] if t.lower() == tag.lower()), tag.strip())
    r['poses'][hit] = q
    return True


# =========================================================================== dates
_DATE_FORMATS = ('%Y-%m-%d', '%Y/%m/%d', '%d.%m.%Y', '%Y%m%d')


def parse_date(text) -> _dt.datetime | None:
    """A record's date: ISO 'YYYY-MM-DD' or an ISO date and time (an offset is converted to UTC), also YYYY/MM/DD and
    DD.MM.YYYY. None when it is not a date."""
    if isinstance(text, (_dt.datetime, _dt.date)):
        text = text.isoformat()
    if not isinstance(text, str) or not text.strip():
        return None
    s = text.strip()
    try:
        d = _dt.datetime.fromisoformat(s.replace('Z', '+00:00') if s.endswith('Z') else s)
    except ValueError:
        d = None
        for f in _DATE_FORMATS:
            try:
                d = _dt.datetime.strptime(s, f)
                break
            except ValueError:
                continue
        if d is None:
            return None
    if d.tzinfo is not None:
        d = d.astimezone(_dt.timezone.utc).replace(tzinfo=None)
    return d


def date_days(text) -> float | None:
    """Days since 0001-01-01 (with the time of day as a fraction); None for no date."""
    d = parse_date(text)
    if d is None:
        return None
    return d.toordinal() + (d.hour * 3600 + d.minute * 60 + d.second + d.microsecond / 1e6) / 86400.0


def normalise_date(text) -> str:
    """'YYYY-MM-DD', or 'YYYY-MM-DDTHH:MM:SS' when a time of day is given; ValueError when it is not a date."""
    d = parse_date(text)
    if d is None:
        raise ValueError(f'"{text}" is not a date: use YYYY-MM-DD (or an ISO date and time).')
    if d.hour or d.minute or d.second:
        return d.replace(microsecond=0).isoformat()
    return d.date().isoformat()


def today() -> str:
    return _dt.date.today().isoformat()


def _file_date(path) -> str:
    try:
        return _dt.date.fromtimestamp(os.path.getmtime(path)).isoformat()
    except OSError:
        return today()


# =========================================================================== the plan
def plan_variant(project, st: FleetSettings | None = None):
    st = st or settings_of(project)
    return (project.variant_by_uid(st.variant) if st.variant else None) or project.variant


def plan_sensors(project, variant) -> list:
    """Every visible camera and sensor of the plan: [{'tag', 'uid', 'kind' ('camera' or the sensor's kind), 'slot',
    'pose'}] in the variant's order (cameras first)."""
    out = []
    for c in variant.cameras:
        if c.visible:
            out.append({'tag': c.tag, 'uid': c.uid, 'kind': 'camera', 'slot': c.slot,
                        'pose': {k: float(getattr(c, k)) for k in POSE_KEYS}})
    count = {}
    for s in variant.sensors:
        if s.visible:
            out.append({'tag': s.tag, 'uid': s.uid, 'kind': s.kind, 'slot': count.get(s.kind, 0),
                        'pose': {k: float(getattr(s, k)) for k in POSE_KEYS}})
            count[s.kind] = count.get(s.kind, 0) + 1
    return out


def match_tags(tags, plan_tags, hidden_tags=()) -> dict:
    """Record tags against the plan's, ignoring case. {'map': {record tag: plan tag}, 'unknown': [...], 'hidden':
    [...] (a hidden camera or sensor of the plan), 'duplicates': [...] (a second tag for the same plan sensor)}."""
    low = {}
    for t in plan_tags:
        low.setdefault(t.lower(), t)
    hid = {t.lower() for t in hidden_tags}
    out = {'map': {}, 'unknown': [], 'hidden': [], 'duplicates': []}
    taken = set()
    tags = list(tags)
    for t in sorted(tags, key=lambda t: (t not in plan_tags, tags.index(t))):     # an exact match wins
        p = low.get(t.lower())
        if p is None:
            (out['hidden'] if t.lower() in hid else out['unknown']).append(t)
        elif p in taken:
            out['duplicates'].append(t)
        else:
            out['map'][t] = p
            taken.add(p)
    return out


def _hidden_tags(variant) -> list:
    return [c.tag for c in variant.cameras if not c.visible] + [s.tag for s in variant.sensors if not s.visible]


def import_report(rec: dict, project, st: FleetSettings | None = None) -> dict:
    """How a record's tags match the plan: {'poses', 'matched', 'unknown', 'hidden', 'duplicates', 'text'}."""
    v = plan_variant(project, st)
    plan = plan_sensors(project, v)
    m = match_tags(rec.get('poses', {}).keys(), [p['tag'] for p in plan], _hidden_tags(v))
    n = len(rec.get('poses', {}))
    text = f'{n} pose{"" if n == 1 else "s"} from {rec.get("source") or KIND_LABEL.get(rec.get("kind"), "?")}' + \
        (f' ({rec["date"]})' if rec.get('date') else '')
    if m['unknown']:
        text += f'; not in the plan: {", ".join(m["unknown"])}'
    if m['hidden']:
        text += f'; hidden in the plan: {", ".join(m["hidden"])}'
    if m['duplicates']:
        text += f'; the same sensor twice: {", ".join(m["duplicates"])}'
    return {'poses': n, 'matched': sorted(m['map'].values()), 'unknown': m['unknown'], 'hidden': m['hidden'],
            'duplicates': m['duplicates'], 'text': text}


# =========================================================================== geometry
def _rots(ypr) -> np.ndarray:
    """rot_zyx for an (N, 3) array of yaw, pitch, roll in degrees: (N, 3, 3)."""
    a = np.radians(np.atleast_2d(np.asarray(ypr, float)))
    cy, sy = np.cos(a[:, 0]), np.sin(a[:, 0])
    cp, sp = np.cos(a[:, 1]), np.sin(a[:, 1])
    cr, sr = np.cos(a[:, 2]), np.sin(a[:, 2])
    Rm = np.empty((len(a), 3, 3))
    Rm[:, 0, 0], Rm[:, 0, 1], Rm[:, 0, 2] = cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr
    Rm[:, 1, 0], Rm[:, 1, 1], Rm[:, 1, 2] = sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr
    Rm[:, 2, 0], Rm[:, 2, 1], Rm[:, 2, 2] = -sp, cp * sr, cp * cr
    return Rm


def _angle_between(Ra, Rb) -> np.ndarray:
    """Rotation angle (degrees) from Ra to Rb, broadcast over leading axes; exact at small angles (atan2 of the sine
    and cosine parts, not acos of the trace)."""
    M = np.swapaxes(Ra, -1, -2) @ Rb
    s = 0.5 * np.sqrt((M[..., 2, 1] - M[..., 1, 2]) ** 2 + (M[..., 0, 2] - M[..., 2, 0]) ** 2 +
                      (M[..., 1, 0] - M[..., 0, 1]) ** 2)
    c = 0.5 * (M[..., 0, 0] + M[..., 1, 1] + M[..., 2, 2] - 1.0)
    return np.degrees(np.arctan2(s, c))


def delta(plan: dict, meas: dict) -> np.ndarray:
    """Measured minus planned: [dx, dy, dz] in mm, [dyaw, dpitch, droll] in degrees wrapped to ±180."""
    d = np.array([(meas[k] - plan[k]) * 1000.0 for k in ('x', 'y', 'z')] +
                 [wrap180(meas[k] - plan[k]) for k in ('yaw', 'pitch', 'roll')])
    return d


def _deltas(plan: dict, poses) -> np.ndarray:
    """delta() of many measured poses at once: (N, 6)."""
    P = np.array([[q[k] for k in POSE_KEYS] for q in poses], float) - np.array([plan[k] for k in POSE_KEYS])
    P[:, :3] *= 1000.0
    P[:, 3:] = (P[:, 3:] + 180.0) % 360.0 - 180.0
    P[:, 3:][P[:, 3:] == -180.0] = 180.0
    return P


def _ypr(pose) -> np.ndarray:
    return np.array([pose['yaw'], pose['pitch'], pose['roll']], float)


@functools.lru_cache(maxsize=4096)
def _plan_rot(yaw, pitch, roll) -> np.ndarray:
    return _rots((yaw, pitch, roll))[0]


def rotation_deg(plan: dict, d) -> np.ndarray:
    """Rotation (degrees) between the planned orientation and plan + d (deltas, shape (..., 6))."""
    d = np.asarray(d, float)
    flat = d.reshape(-1, 6)
    ypr = _ypr(plan)
    Rm = _rots(ypr[None, :] + flat[:, 3:])
    return _angle_between(_plan_rot(*ypr.tolist())[None], Rm).reshape(d.shape[:-1])


def ratio_of(d, rot_deg, st: FleetSettings):
    """Worst share of the tolerance: the largest axis offset over the position tolerance or the rotation over the
    angle tolerance (broadcast)."""
    d = np.asarray(d, float)
    pos = np.max(np.abs(d[..., :3]), axis=-1) / st.pos_tol_mm
    return np.maximum(pos, np.asarray(rot_deg, float) / st.ang_tol_deg)


def _between(plan, a, b, st) -> float:
    """Share of the tolerance between two measurements a and b (deltas against the plan)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    Rab = _rots(_ypr(plan)[None, :] + np.array([a[3:], b[3:]]))
    ang = float(_angle_between(Rab[0], Rab[1]))
    return float(max(np.max(np.abs(b[:3] - a[:3])) / st.pos_tol_mm, ang / st.ang_tol_deg))


def verdict_of(ratio) -> str:
    if ratio is None or not math.isfinite(ratio):
        return 'missing'
    return 'ok' if ratio <= 1.0 + 1e-9 else 'adjust' if ratio <= REMOUNT + 1e-9 else 're-mount'


# =========================================================================== the analysis
@dataclass
class FleetResult:
    settings: FleetSettings
    plan: list                  # plan_sensors()
    vehicles: list              # per vehicle: {'id', 'name', 'vin', 'note', 'records', 'dated', 'latest', 'worst',
    #                             'verdict', 'counts', 'unknown', 'hidden', 'missing', 'flags'}
    cells: dict                 # (vehicle id, plan tag) -> cell (see _cell)
    fleet: list                 # per plan sensor: median, spread, outliers, systematic (see _consistency)
    findings: list              # [{'severity', 'kind', 'message', 'vehicle', 'tags', 'uids'}], worst first
    variant: str = ''
    variant_uid: str = ''

    def cell(self, vid, tag):
        return self.cells.get((vid, tag))

    def vehicle(self, vid):
        return next((v for v in self.vehicles if v['id'] == vid), None)

    def sensor(self, tag):
        return next((s for s in self.fleet if s['tag'] == tag), None)

    def worst(self) -> str:
        return self.findings[0]['severity'] if self.findings else 'good'

    def flagged(self, what) -> list:
        """[(vehicle id, tag)] of the cells with a 'drift', 'jump' or 'outlier' flag, or a verdict."""
        out = []
        for (vid, tag), c in self.cells.items():
            if what in ('drift', 'jump'):
                hit = bool(c['drift'] and c['drift']['trend' if what == 'drift' else 'jump'])
            elif what == 'outlier':
                hit = c['outlier'] is not None
            else:
                hit = c['verdict'] == what
            if hit:
                out.append((vid, tag))
        return out


def _series(vrec, plan_tags, hidden, cache=None):
    """Per plan tag the vehicle's measurements in date order: [(days or None, record, record tag)], and the tags the
    plan does not know. cache: a dict reused across vehicles (records mostly share their tag sets)."""
    order = []
    for i, r in enumerate(vrec):
        order.append((date_days(r['date']), i, r))
    order.sort(key=lambda t: (-math.inf if t[0] is None else t[0], t[1]))
    per = {}
    unknown, hid = [], []
    cache = {} if cache is None else cache
    for days, _, r in order:
        key = tuple(r['poses'])
        m = cache.get(key)
        if m is None:
            m = cache[key] = match_tags(key, plan_tags, hidden)
        for rt, pt in m['map'].items():
            per.setdefault(pt, []).append((days, r, rt))
        unknown += [(t, r) for t in m['unknown'] + m['duplicates']]
        hid += [(t, r) for t in m['hidden']]
    return per, unknown, hid


# two-sided 5 % critical values of Student's t for 1 … 30 degrees of freedom, then 40, 60, 120 and the normal limit
_T95 = (12.7062, 4.3027, 3.1824, 2.7764, 2.5706, 2.4469, 2.3646, 2.3060, 2.2622, 2.2281, 2.2010, 2.1788, 2.1604,
        2.1448, 2.1314, 2.1199, 2.1098, 2.1009, 2.0930, 2.0860, 2.0796, 2.0739, 2.0687, 2.0639, 2.0595, 2.0555,
        2.0518, 2.0484, 2.0452, 2.0423)
_T95_TAIL = ((30, 2.0423), (40, 2.0211), (60, 2.0003), (120, 1.9799), (math.inf, 1.9600))


def _t_crit(dof: int) -> float:
    """Two-sided critical value of Student's t at the 5 % level (SIGNIFICANCE) for dof degrees of freedom: the table
    up to 30, then interpolated in 1/dof (exact to 0.001 against the distribution)."""
    dof = max(int(dof), 1)
    if dof <= 30:
        return _T95[dof - 1]
    for (d0, t0), (d1, t1) in zip(_T95_TAIL, _T95_TAIL[1:]):
        if dof <= d1:
            f = (1 / d0 - 1 / dof) / (1 / d0 - (0.0 if d1 == math.inf else 1 / d1))
            return t0 + f * (t1 - t0)
    return 1.96


def _significant(slope, se, n) -> np.ndarray:
    """Which slopes differ from zero (a perfect fit, se = 0, counts when the slope is not zero)."""
    return (np.abs(slope) > _t_crit(n - 2) * se) & (np.abs(slope) > 1e-12)


def _trend(t, D):
    """Least-squares slope per day of each column of D over t, its standard error (None below three points) and the
    mean point."""
    tm = t.mean()
    x = t - tm
    sxx = float(np.sum(x * x))
    if sxx <= 0:
        return np.zeros(D.shape[1]), None, tm
    slope = (x[:, None] * (D - D.mean(axis=0))).sum(axis=0) / sxx
    if len(t) < 3:
        return slope, None, tm
    res = D - (D.mean(axis=0) + slope * x[:, None])
    se = np.sqrt(np.sum(res * res, axis=0) / (len(t) - 2) / sxx)
    return slope, se, tm


def _crossing(plan, d0, slope, st, horizon):
    """Days from the latest measurement d0 until d0 + slope·t leaves the tolerance (0 when it already has), None when
    it does not within the horizon."""
    r0 = float(ratio_of(d0, rotation_deg(plan, d0), st))
    if r0 > 1.0:
        return 0.0
    if not np.any(slope):
        return None
    lo, hi = 0.0, float(horizon)
    for _ in range(3):                    # three grids of 512 steps, each inside the last one's crossing step
        ts = np.linspace(lo, hi, 513)
        D = d0[None, :] + ts[:, None] * slope[None, :]
        r = ratio_of(D, rotation_deg(plan, D), st)
        k = np.nonzero(r > 1.0)[0]
        if not len(k):
            return None
        if k[0] == 0:
            return float(ts[0])
        lo, hi = float(ts[k[0] - 1]), float(ts[k[0]])
    return hi


def _drift(plan, series, st: FleetSettings):
    """Drift of one sensor on one vehicle across its dated records (None with fewer than two)."""
    dated = [(t, r, rt) for t, r, rt in series if t is not None]
    if len(dated) < 2:
        return None
    t = np.array([x[0] for x in dated])
    D = _deltas(plan, [x[1]['poses'][x[2]] for x in dated])
    rot = rotation_deg(plan, D)
    rat = ratio_of(D, rot, st)
    W = st.drift_window_days
    slope, se, _ = _trend(t, D)
    distinct = len(np.unique(np.round(t, 6)))
    if se is None or distinct < 3:
        sig = np.zeros(6, bool)
    else:
        sig = _significant(slope, se, len(t))
    used = np.where(sig, slope, 0.0)
    rate = float(ratio_of(used * W, rotation_deg(plan, used * W), st))      # share of the tolerance per window
    d_last = D[-1]
    days = _crossing(plan, d_last, used, st, max(5.0 * W, 730.0)) if sig.any() else None
    r_last = float(rat[-1])
    proj = d_last + used * W
    r_proj = float(ratio_of(proj, rotation_deg(plan, proj), st))
    trend_flag = bool(sig.any() and rate >= MIN_RATE and r_proj > 1.0 and r_proj > r_last)
    # the latest step against the trend of the records before it
    step = D[-1] - D[-2]
    prev_t, prev_D = t[:-1], D[:-1]
    expected = np.zeros(6)
    if len(np.unique(np.round(prev_t, 6))) >= 2:
        ps, pse, _ = _trend(prev_t, prev_D)
        if pse is not None:                              # only components that are a real trend
            ps = np.where(_significant(ps, pse, len(prev_t)), ps, 0.0)
        else:
            ps = np.zeros(6)                             # two earlier points: not enough to call it a trend
        expected = ps * (t[-1] - t[-2])
    jump_ratio = _between(plan, D[-2] + expected, D[-1], st)
    step_ratio = _between(plan, D[-2], D[-1], st)
    return {'n': len(dated), 'span_days': float(t[-1] - t[0]), 'change': (D[-1] - D[0]).tolist(),
            'trend_per_30d': (slope * 30.0).tolist(), 'significant': sig.tolist(), 'rate_ratio': rate,
            'days_to_tol': days, 'projected_ratio': r_proj, 'trend': trend_flag,
            'step': step.tolist(), 'step_ratio': step_ratio, 'jump_ratio': jump_ratio,
            'jump': bool(jump_ratio >= st.jump_frac),
            'series': [{'date': r['date'], 'days': float(tt), 'd': dd.tolist(), 'rot_deg': float(ro),
                        'ratio': float(ra), 'record': r['id'], 'source': r['source'], 'kind': r['kind']}
                       for (tt, r, _), dd, ro, ra in zip(dated, D, rot, rat)]}


def _cell(vid, ps, series, st):
    """The latest measurement of one plan sensor on one vehicle against the plan, with its drift."""
    days, r, rt = series[-1]
    pose = r['poses'][rt]
    d = delta(ps['pose'], pose)
    rot = float(rotation_deg(ps['pose'], d))
    ratio = float(ratio_of(d, rot, st))
    q = r['quality'].get(rt) or next((v for k, v in r['quality'].items() if k.lower() == rt.lower()), {})
    return {'vehicle': vid, 'tag': ps['tag'], 'kind': ps['kind'], 'uid': ps['uid'], 'verdict': verdict_of(ratio),
            'ratio': ratio, 'd': d.tolist(), 'dpos_mm': float(np.linalg.norm(d[:3])), 'rot_deg': rot,
            'date': r['date'], 'record': r['id'], 'source': r['source'], 'record_kind': r['kind'],
            'record_tag': rt, 'quality': dict(q), 'pose': dict(pose), 'records': len(series),
            'drift': _drift(ps['pose'], series, st), 'outlier': None}


def _robust(X):
    """Median and robust standard deviation (1.4826 × median absolute deviation) of each column."""
    med = np.median(X, axis=0)
    return med, 1.4826 * np.median(np.abs(X - med), axis=0)


def _consistency(ps, cells, names, st: FleetSettings) -> dict:
    """One plan sensor across the fleet: the median Δ of the vehicles' latest records, its spread, the outliers (robust
    z-score per component) and whether the fleet as a whole is built off the plan."""
    out = {'tag': ps['tag'], 'kind': ps['kind'], 'uid': ps['uid'], 'n': len(cells), 'median': None, 'sigma': None,
           'median_ratio': None, 'beyond': sum(c['ratio'] > 1.0 for c in cells), 'near_median': 0,
           'systematic': False, 'outliers': [], 'z': {}}
    if not cells:
        return out
    X = np.array([c['d'] for c in cells])
    med, sig = _robust(X)
    floor = np.array([st.pos_tol_mm] * 3 + [st.ang_tol_deg] * 3) * SIGMA_FLOOR
    spread = np.maximum(sig, floor)
    out['median'], out['sigma'] = med.tolist(), sig.tolist()
    mr = float(ratio_of(med, rotation_deg(ps['pose'], med), st))
    out['median_ratio'] = mr
    near = sum(c['ratio'] > 1.0 and _between(ps['pose'], med, c['d'], st) <= 1.0 for c in cells)
    out['near_median'] = near
    out['systematic'] = bool(len(cells) >= 2 and mr > 1.0 and near > len(cells) / 2)
    if len(cells) >= MIN_FLEET:
        Z = (X - med) / spread
        for i, (c, z) in enumerate(zip(cells, Z)):
            k = int(np.argmax(np.abs(z)))
            out['z'][c['vehicle']] = float(abs(z[k]))
            if abs(z[k]) > st.outlier_z:
                o = {'vehicle': c['vehicle'], 'name': names.get(c['vehicle'], ''), 'component': COMPONENTS[k],
                     'from_median': float(X[i, k] - med[k]), 'z': float(z[k]),
                     'cause': ('drift' if c['drift'] and c['drift']['trend'] else
                               'jump' if c['drift'] and c['drift']['jump'] else None)}
                c['outlier'] = o
                out['outliers'].append(o)
    out['outliers'].sort(key=lambda o: -abs(o['z']))
    return out


_WORDS = {'x': ('forward of', 'behind'), 'y': ('left of', 'right of'), 'z': ('above', 'below'),
          'yaw': ('yawed {v} left of', 'yawed {v} right of'), 'pitch': ('pitched {v} down from', 'pitched {v} up from'),
          'roll': ('rolled {v} to the right of', 'rolled {v} to the left of')}


def describe(component: str, value: float, what: str) -> str:
    """'sits 12 mm forward of the fleet median', 'is yawed 2.1° left of the plan' (yaw positive left, pitch positive
    down, roll positive to the right: right side down)."""
    pos, neg = _WORDS[component]
    w = pos if value >= 0 else neg
    if component in ('x', 'y', 'z'):
        return f'sits {abs(value):.0f} mm {w} {what}'
    return 'is ' + w.format(v=f'{abs(value):.1f}°') + f' {what}'


def _worst_part(c, st) -> str:
    """'4.2°' (the rotation) or '14 mm in z': whichever part of a cell's ratio dominates."""
    d = c['d']
    k = int(np.argmax(np.abs(d[:3])))
    pr = abs(d[k]) / st.pos_tol_mm
    if c['rot_deg'] / st.ang_tol_deg >= pr:
        return f'{c["rot_deg"]:.1f}°'
    return f'{abs(d[k]):.0f} mm in {COMPONENTS[k]}'


def run(project, engine=None, settings: FleetSettings | None = None, variant=None, progress=None, cancel=None):
    """The fleet against the plan (settings.variant, default the active variant; `variant` overrides both). None
    when cancelled."""
    st = settings or settings_of(project)
    v = variant or plan_variant(project, st)
    plan = plan_sensors(project, v)
    plan_tags = [p['tag'] for p in plan]
    hidden = _hidden_tags(v)
    vs = vehicles(project)
    names = {x['id']: x['name'] for x in vs}
    cells, rows = {}, []
    measured_any = set()
    cache = {}
    for i, veh in enumerate(vs):
        if cancel is not None and cancel():
            return None
        per, unknown, hid = _series(veh['records'], plan_tags, hidden, cache)
        for ps in plan:
            s = per.get(ps['tag'])
            if s:
                cells[(veh['id'], ps['tag'])] = _cell(veh['id'], ps, s, st)
                measured_any.add(ps['tag'])
        dated = [d for d in (date_days(r['date']) for r in veh['records']) if d is not None]
        latest = max(veh['records'], key=lambda r: date_days(r['date']) or -math.inf)['date'] if veh['records'] else ''
        rows.append({'id': veh['id'], 'name': veh['name'], 'vin': veh['vin'], 'note': veh['note'],
                     'records': len(veh['records']), 'dated': len(dated), 'latest': latest,
                     'unknown': _uniq_tags(unknown), 'hidden': _uniq_tags(hid), 'missing': [], 'worst': None,
                     'verdict': 'missing', 'counts': {k: 0 for k in VERDICTS}, 'flags': {}})
        if progress:
            progress((i + 1) / max(len(vs), 1) * 0.8, f'Vehicle {veh["name"]}')
    fleet = []
    for ps in plan:
        cs = [cells[(r['id'], ps['tag'])] for r in rows if (r['id'], ps['tag']) in cells]
        fleet.append(_consistency(ps, cs, names, st))
    for r in rows:
        mine = [cells[(r['id'], ps['tag'])] for ps in plan if (r['id'], ps['tag']) in cells]
        r['missing'] = [ps['tag'] for ps in plan if (r['id'], ps['tag']) not in cells]
        for c in mine:
            r['counts'][c['verdict']] += 1
        r['counts']['missing'] = len(r['missing'])
        if mine:
            w = max(mine, key=lambda c: c['ratio'])
            r['worst'] = w['ratio']
            r['verdict'] = w['verdict']
        r['flags'] = {'drift': [c['tag'] for c in mine if c['drift'] and c['drift']['trend']],
                      'jump': [c['tag'] for c in mine if c['drift'] and c['drift']['jump']],
                      'outlier': [c['tag'] for c in mine if c['outlier'] is not None]}
    res = FleetResult(st, plan, rows, cells, fleet, [], v.name, v.uid)
    res.findings = _findings(res, measured_any)
    if progress:
        progress(1.0, 'Done')
    return res


def _uniq_tags(pairs) -> list:
    """[(tag, record)] -> [{'tag', 'records': [(date, source)]}] in first-seen order."""
    out = {}
    for t, r in pairs:
        out.setdefault(t, []).append((r['date'], r['source']))
    return [{'tag': t, 'records': rs} for t, rs in out.items()]


def _tags_text(tags, n=8) -> str:
    tags = list(tags)
    return ', '.join(tags[:n]) + (f' and {len(tags) - n} more' if len(tags) > n else '')


def _finding(out, sev, kind, message, vid='', tags=(), uids=()):
    out.append({'severity': sev, 'kind': kind, 'message': message, 'vehicle': vid, 'tags': list(tags),
                'uids': list(uids)})


def _findings(res: FleetResult, measured_any) -> list:
    st, out = res.settings, []
    tol = f'{st.pos_tol_mm:g} mm / {st.ang_tol_deg:g}°'
    if not res.plan:
        _finding(out, 'warning', 'plan', f'The plan, {res.variant}, has no visible camera or sensor to compare with.')
    if not res.vehicles:
        _finding(out, 'warning', 'empty', 'No vehicles yet: add the vehicles built to this design and import their '
                                          'installation checks or calibrations.')
    by_v = {r['id']: r for r in res.vehicles}
    plan_uid = {p['tag']: p['uid'] for p in res.plan}
    # ---- against the plan
    for r in res.vehicles:
        if not r['records']:
            _finding(out, 'warning', 'no_records', f'{r["name"]} has no records yet.', r['id'])
            continue
        cs = [res.cells[(r['id'], p['tag'])] for p in res.plan if (r['id'], p['tag']) in res.cells]
        for verdict, sev, what in (('re-mount', 'serious', 're-mount'), ('adjust', 'warning', 'adjust')):
            bad = sorted((c for c in cs if c['verdict'] == verdict), key=lambda c: -c['ratio'])
            if bad:
                parts = ', '.join(f'{c["tag"]} ({_worst_part(c, st)} off, {c["date"] or "undated"})' for c in bad[:6])
                more = f' and {len(bad) - 6} more' if len(bad) > 6 else ''
                _finding(out, sev, verdict.replace('-', ''),
                         f'{r["name"]}: {what} {parts}{more}: beyond {"three times " if verdict == "re-mount" else ""}'
                         f'the tolerance ({tol}) in the latest record.',
                         r['id'], [c['tag'] for c in bad], [c['uid'] for c in bad])
        miss = [t for t in r['missing'] if t in measured_any]
        if miss:
            _finding(out, 'warning', 'missing',
                     f'{r["name"]}: {_tags_text(miss)} not measured in any of its records (measured on other '
                     f'vehicles).', r['id'], miss, [plan_uid[t] for t in miss])
        if r['unknown']:
            u = r['unknown']
            seen = '; '.join(f'{x["tag"]} ({x["records"][-1][1] or "?"}, {x["records"][-1][0] or "undated"})'
                             for x in u[:6]) + ('' if len(u) <= 6 else f' and {len(u) - 6} more')
            _finding(out, 'warning', 'unknown',
                     f'{r["name"]}: {len(u)} tag{"" if len(u) == 1 else "s"} in its records match nothing in the '
                     f'plan: {seen}. Rename the sensor in the record or the plan, or check the plan variant.', r['id'],
                     [x['tag'] for x in u])
        if r['hidden']:
            h = [x['tag'] for x in r['hidden']]
            _finding(out, 'warning', 'hidden', f'{r["name"]}: {_tags_text(h)} {"is" if len(h) == 1 else "are"} hidden '
                                               'in the plan, so not compared.', r['id'], h)
    never = [p['tag'] for p in res.plan if p['tag'] not in measured_any]
    if res.vehicles and never and any(r['records'] for r in res.vehicles):
        _finding(out, 'warning', 'never', f'{_tags_text(never)} {"is" if len(never) == 1 else "are"} not measured on '
                                          'any vehicle.', '', never, [plan_uid[t] for t in never])
    # ---- over time
    for (vid, tag), c in res.cells.items():
        d = c['drift']
        if not d:
            continue
        name = by_v[vid]['name']
        if d['trend']:
            comp = [k for k in range(6) if d['significant'][k]]
            k = max(comp, key=lambda k: abs(d['trend_per_30d'][k]) / (st.pos_tol_mm if k < 3 else st.ang_tol_deg))
            rate = d['trend_per_30d'][k]
            unit = ' mm' if k < 3 else '°'
            dec = 1 if k < 3 else 2
            when = ('already out of tolerance and still moving' if d['days_to_tol'] == 0 else
                    f'out of tolerance in about {d["days_to_tol"]:.0f} days at this rate')
            _finding(out, 'warning', 'drift',
                     f'{name}: {tag} drifts {rate:+.{dec}f}{unit} in {COMPONENTS[k]} per 30 days over '
                     f'{d["n"]} records ({d["series"][0]["date"]} to {d["series"][-1]["date"]}): {when}. Check its '
                     'bracket and fasteners.', vid, [tag], [c['uid']])
        if d['jump']:
            s = np.array(d['step'])
            k = int(np.argmax(np.abs(s) / np.array([st.pos_tol_mm] * 3 + [st.ang_tol_deg] * 3)))
            unit = ' mm' if k < 3 else '°'
            dec = 0 if k < 3 else 2
            prev, last = d['series'][-2], d['series'][-1]
            _finding(out, 'warning', 'jump',
                     f'{name}: {tag} moved {s[k]:+.{dec}f}{unit} in {COMPONENTS[k]} between {prev["date"]} and '
                     f'{last["date"]} ({last["source"] or KIND_LABEL.get(last["kind"], "")}), '
                     f'{d["jump_ratio"]:.1f}× the tolerance off its earlier course: a knock or a loose mount?',
                     vid, [tag], [c['uid']])
    # ---- across the fleet
    for f in res.fleet:
        if f['systematic']:
            med = np.array(f['median'])
            k = int(np.argmax(np.abs(med) / np.array([st.pos_tol_mm] * 3 + [st.ang_tol_deg] * 3)))
            _finding(out, 'warning', 'systematic',
                     f'{f["tag"]} {describe(COMPONENTS[k], med[k], "the plan")} on {f["near_median"]} of {f["n"]} '
                     f'vehicles (fleet median): the rig is built that way. Make a variant from the fleet median so the '
                     f'plan matches it.', '', [f['tag']], [f['uid']])
        for o in f['outliers']:
            if o['cause']:
                continue                                  # its drift or jump finding names it
            _finding(out, 'warning', 'outlier',
                     f'{o["name"]}: {f["tag"]} {describe(o["component"], o["from_median"], "the fleet median")} '
                     f'(robust z {abs(o["z"]):.1f}): this vehicle is built differently.', o['vehicle'], [f['tag']],
                     [f['uid']])
    if res.cells and not any(f['severity'] != 'good' for f in out):
        _finding(out, 'good', 'good', f'Every measured sensor on {len(res.vehicles)} vehicle'
                                      f'{"" if len(res.vehicles) == 1 else "s"} is within {tol} of the plan, nothing '
                                      'drifts and no vehicle stands out.')
    elif res.cells:
        ok = sum(c['verdict'] == 'ok' for c in res.cells.values())
        _finding(out, 'good', 'within', f'{ok} of {len(res.cells)} measured sensors are within {tol} of the plan in '
                                        'their latest record.')
    order = {s: i for i, s in enumerate(SEVERITIES)}
    out.sort(key=lambda f: order.get(f['severity'], 9))
    return out


# =========================================================================== the plan from the fleet
def median_poses(res: FleetResult, min_vehicles: int = 2) -> dict:
    """Plan tag -> the fleet median as-built pose (plan + median Δ) for sensors measured on min_vehicles or more."""
    plan = {p['tag']: p for p in res.plan}
    out = {}
    for f in res.fleet:
        if f['n'] < min_vehicles or f['median'] is None:
            continue
        p, m = plan[f['tag']]['pose'], f['median']
        out[f['tag']] = {'x': p['x'] + m[0] / 1000.0, 'y': p['y'] + m[1] / 1000.0, 'z': p['z'] + m[2] / 1000.0,
                         'yaw': wrap180(p['yaw'] + m[3]), 'pitch': p['pitch'] + m[4], 'roll': wrap180(p['roll'] + m[5])}
    return out


def median_variant(project, res: FleetResult, name: str = '', min_vehicles: int = 2, activate: bool = True):
    """Add a copy of the plan variant with every camera and sensor measured on min_vehicles or more vehicles moved to
    the fleet's median as-built pose (the others as planned); mirrored camera pairs are unlinked. Returns the new
    variant (None when no sensor qualifies). The page runs it inside ctx.store.edit('variants', ...)."""
    from .project import Variant, new_uid
    src = project.variant_by_uid(res.variant_uid)
    poses = median_poses(res, min_vehicles)
    if src is None or not poses:
        return None
    v = Variant.from_dict(src.to_dict())
    v.uid = new_uid()
    v.name = name or f'{src.name} · fleet median'
    v.note = (f'The fleet median as-built poses of {_tags_text(sorted(poses), 20)} from {len(res.vehicles)} '
              f'vehicles' + (f'; {src.note}' if src.note else ''))
    moved = set()
    for obj in list(v.cameras) + list(v.sensors):
        q = poses.get(obj.tag) if obj.visible else None
        if q is not None:
            for k in POSE_KEYS:
                setattr(obj, k, round(q[k], 4 if k in ('x', 'y', 'z') else 3))
            moved.add(obj.uid)
    for c in v.cameras:
        if c.uid in moved or c.link_uid in moved:
            c.link_uid = ''                    # as-built poses are not mirror images of each other
    olds = [c.uid for c in v.cameras] + [s.uid for s in v.sensors]
    new = {u: new_uid() for u in olds}
    for c in v.cameras:
        c.uid = new[c.uid]
        c.link_uid = new.get(c.link_uid, '') if c.link_uid else ''
    for s in v.sensors:
        s.uid = new[s.uid]
    v.seams = {}
    project.variants.append(v)
    if activate:
        project.active_uid = v.uid
    return v


# =========================================================================== imports
def _load_text(path) -> str:
    try:
        if os.path.getsize(path) > MAX_RECORD_BYTES:
            raise FleetImportError(f'{os.path.basename(path)} is larger than 200 MB: not a calibration file.')
        with open(path, 'r', encoding='utf-8-sig') as fh:
            return fh.read()
    except UnicodeDecodeError:
        raise FleetImportError(f'{os.path.basename(path)} is not a text file.')
    except OSError as e:
        raise FleetImportError(f'Cannot read {path}: {e.strerror or e}.')


def _record(kind, date, source, poses, quality=None, note=''):
    return {'id': new_id(), 'date': date, 'source': source, 'kind': kind, 'poses': poses, 'quality': quality or {},
            'note': note}


def _pose_from(pos_m, ypr_deg) -> dict:
    p = {'x': float(pos_m[0]), 'y': float(pos_m[1]), 'z': float(pos_m[2]), 'yaw': float(wrap180(ypr_deg[0])),
         'pitch': float(ypr_deg[1]), 'roll': float(wrap180(ypr_deg[2]))}
    if clean_pose(p) is None:
        raise FleetImportError('A pose in the file is not a number.')
    return p


def _pose_from_matrix(Rm, pos_m) -> dict:
    Rm = np.asarray(Rm, float)
    if Rm.shape != (3, 3) or not np.all(np.isfinite(Rm)):
        raise FleetImportError('A rotation in the file is not a 3×3 matrix of numbers.')
    U, _, Vt = np.linalg.svd(Rm)                           # the nearest rotation (files round their matrices)
    Rm = U @ np.diag([1.0, 1.0, np.linalg.det(U @ Vt)]) @ Vt
    return _pose_from(pos_m, euler_zyx(Rm))


def _frame_origin(project, frame: str) -> np.ndarray:
    """Rear-axle coordinates of an export frame's origin, from the project (the vehicle's bumpers and axles)."""
    from .project import FRAME_PRESETS
    if frame in ('', 'rear_axle', None):
        return np.zeros(3)
    if frame not in FRAME_PRESETS:
        raise FleetImportError(f'Unknown frame "{frame}" in the file.')
    if project is None:
        raise FleetImportError(f'The file is in the {frame} frame: open the project of this vehicle to place it.')
    q = copy.copy(project)
    q.frame = frame
    return np.asarray(q.origin_offset(), float)


def read_install_check(doc: dict, source='', fallback_date='') -> dict:
    """A record from an installation check report (vantage.install_check/1): every camera with a measured pose
    (vehicle frame, rear axle, metres and degrees); its RMS, status and corner count as quality; the cameras without
    a pose keep their status and message."""
    if not isinstance(doc, dict) or not isinstance(doc.get('cameras'), list):
        raise FleetImportError('Not an installation check report: no camera list.')
    poses, quality, failed = {}, {}, []
    for row in doc['cameras']:
        if not isinstance(row, dict) or not str(row.get('tag', '')).strip():
            continue
        tag = str(row['tag']).strip()
        q = {k: row[k] for k in ('status', 'message', 'rms_px', 'corners', 'rotation_error_deg')
             if isinstance(row.get(k), (str, int, float)) and not isinstance(row.get(k), bool)}
        if isinstance(row.get('boards'), list):
            q['boards'] = len(row['boards'])
        m = clean_pose(row.get('measured'))
        if m is not None:
            poses[tag] = m
        else:
            failed.append(tag)
        if q:
            quality[tag] = q
    if not poses:
        raise FleetImportError('The installation check report has no measured pose (every camera failed or had no '
                               'image).')
    date = ''
    try:
        date = normalise_date(doc.get('time'))
    except ValueError:
        date = fallback_date or today()
    note = f'Installation check of {doc.get("variant") or "?"}'
    if isinstance(doc.get('tolerance'), dict):
        t = doc['tolerance']
        note += f' (tolerance {t.get("position_mm", "?")} mm / {t.get("angle_deg", "?")}°)'
    if failed:
        note += f'; no pose for {", ".join(failed)}'
    return _record('install_check', date, source, poses, quality, note)


def read_extrinsics(doc: dict, project=None, source='', fallback_date='') -> dict:
    """A record from Vantage's extrinsics export (JSON or YAML, vantage.extrinsics/1): positions converted back from
    the export frame and units (the file's origin_from_rear_axle), angles from degrees or radians."""
    if not isinstance(doc, dict):
        raise FleetImportError('Not an extrinsics export.')
    fr = doc.get('frame') if isinstance(doc.get('frame'), dict) else {}
    units = fr.get('units', 'm')
    if units not in ('m', 'mm'):
        raise FleetImportError(f'Unknown length unit "{units}" in the extrinsics (m or mm).')
    scale = 1000.0 if units == 'mm' else 1.0
    angles = fr.get('angles', 'deg')
    if angles not in ('deg', 'rad'):
        raise FleetImportError(f'Unknown angle unit "{angles}" in the extrinsics (deg or rad).')
    o = fr.get('origin_from_rear_axle')
    if isinstance(o, list) and len(o) == 3 and all(_fnum(x) is not None for x in o):
        origin = np.array([float(x) for x in o]) / scale
    else:
        origin = _frame_origin(project, fr.get('id', 'rear_axle'))
    A = (lambda v: math.degrees(float(v))) if angles == 'rad' else (lambda v: float(v))
    poses = {}
    for key, Tkey, optical in (('cameras', 'T_world_optical', True), ('sensors', 'T_world_sensor', False)):
        for row in doc.get(key) or []:
            if not isinstance(row, dict) or not str(row.get('tag', '')).strip():
                continue
            tag = str(row['tag']).strip()
            pos = row.get('position')
            try:
                if isinstance(pos, list) and len(pos) == 3 and all(_fnum(row.get(k)) is not None
                                                                   for k in ('yaw', 'pitch', 'roll')):
                    p = np.array([float(x) for x in pos]) / scale + origin
                    poses[tag] = _pose_from(p, [A(row['yaw']), A(row['pitch']), A(row['roll'])])
                elif isinstance(row.get(Tkey), list):
                    T = np.array(row[Tkey], float)
                    Rm = T[:3, :3] @ BODY_TO_OPTICAL if optical else T[:3, :3]
                    poses[tag] = _pose_from_matrix(Rm, T[:3, 3] / scale + origin)
            except (TypeError, ValueError, IndexError):
                raise FleetImportError(f'The pose of {tag} in the extrinsics is not a number.')
    if not poses:
        raise FleetImportError('The extrinsics hold no camera or sensor pose.')
    try:
        date = normalise_date(doc.get('exported'))
    except ValueError:
        date = fallback_date or today()
    note = (f'Extrinsics of {doc.get("variant") or "?"} in {units}, {angles}, frame {fr.get("id", "rear_axle")}'
            f' (origin {", ".join(f"{v:.3f}" for v in origin)} m from the rear axle): converted to the rear-axle '
            'frame in metres and degrees')
    return _record('extrinsics', date, source, poses, {}, note)


def read_extrinsics_csv(text: str, project=None, source='', fallback_date='') -> dict:
    """A record from the extrinsics CSV: columns x_<unit> … roll_<angles> and the frame id per row. The CSV has no
    origin numbers, so a frame other than the rear axle is placed with this project's vehicle."""
    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        raise FleetImportError('The CSV is empty.')
    head = [h.strip() for h in rows[0]]
    cols = {h: i for i, h in enumerate(head)}
    unit = next((u for u in ('mm', 'm') if f'x_{u}' in cols), None)
    ang = next((a for a in ('deg', 'rad') if f'yaw_{a}' in cols), None)
    if 'tag' not in cols or unit is None or ang is None:
        raise FleetImportError('Not an extrinsics CSV: it needs tag, x_m (or x_mm) … yaw_deg (or yaw_rad) columns.')
    scale = 1000.0 if unit == 'mm' else 1.0
    A = (lambda v: math.degrees(v)) if ang == 'rad' else (lambda v: v)
    poses, frames = {}, set()
    for row in rows[1:]:
        if not row or not row[cols['tag']].strip():
            continue
        tag = row[cols['tag']].strip()
        try:
            vals = [float(row[cols[f'{k}_{unit}']]) for k in ('x', 'y', 'z')]
            angs = [A(float(row[cols[f'{k}_{ang}']])) for k in ('yaw', 'pitch', 'roll')]
        except (ValueError, IndexError):
            raise FleetImportError(f'The pose of {tag} in the CSV is not a number.')
        frame = row[cols['frame']].strip() if 'frame' in cols and len(row) > cols['frame'] else 'rear_axle'
        frames.add(frame)
        poses[tag] = _pose_from(np.array(vals) / scale + _frame_origin(project, frame), angs)
    if not poses:
        raise FleetImportError('The CSV holds no pose.')
    note = f'Extrinsics CSV in {unit}, {ang}, frame {", ".join(sorted(frames))}'
    if frames - {'rear_axle', ''}:
        note += ' (frame origin from this project\'s vehicle: the CSV does not carry it)'
    return _record('extrinsics', fallback_date or today(), source, poses, {}, note + '; date: the file\'s')


_FRAME_LABEL = {'rear axle': 'rear_axle', 'front axle': 'front_axle', 'front bumper': 'front_bumper',
                'vehicle centre': 'centre', 'custom origin': 'custom'}


def _readme_frame(text: str, project) -> tuple:
    """(origin, frame words, exported stamp) from a Vantage nuScenes README."""
    m = re.search(r'Ego frame here: ISO 8855 · ([a-z ]+?) \(', text)
    stamp = re.search(r'exported (\S+)', text)
    stamp = stamp.group(1) if stamp else ''
    if not m:
        return np.zeros(3), 'rear axle (assumed)', stamp
    words = m.group(1).strip()
    frame = _FRAME_LABEL.get(words)
    if frame == 'custom':
        c = re.search(r'custom point \(([-−0-9.]+), ([-−0-9.]+), ([-−0-9.]+)\) m', text)
        if not c:
            raise FleetImportError('The nuScenes README names a custom frame without its origin.')
        return np.array([float(c.group(i).replace('−', '-')) for i in (1, 2, 3)]), words, stamp
    if frame is None:
        raise FleetImportError(f'Unknown ego frame "{words}" in the nuScenes README.')
    return _frame_origin(project, frame), words, stamp


def _quat_matrix(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if not math.isfinite(n) or n < 1e-9:
        raise FleetImportError('A rotation quaternion in the nuScenes table is not a rotation.')
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def read_nuscenes(path: str, project=None, origin=None, fallback_date='') -> dict:
    """A record from nuScenes calibrated_sensor.json and sensor.json (path: either file or their folder). Camera
    rotations are the optical frame in the ego frame (converted to the body frame); lidar and radar frames are
    x forward, y left, z up. The ego origin comes from a Vantage README.md beside the tables, else `origin`
    (rear-axle coordinates of the file's ego origin, metres), else the rear axle on the ground."""
    folder = path if os.path.isdir(path) else os.path.dirname(path)
    if os.path.isdir(path) and not os.path.isfile(os.path.join(folder, 'calibrated_sensor.json')) and \
            os.path.isfile(os.path.join(folder, 'nuscenes', 'calibrated_sensor.json')):
        folder = os.path.join(folder, 'nuscenes')
    cal_p, sen_p = os.path.join(folder, 'calibrated_sensor.json'), os.path.join(folder, 'sensor.json')
    for p_ in (cal_p, sen_p):
        if not os.path.isfile(p_):
            raise FleetImportError(f'nuScenes needs calibrated_sensor.json and sensor.json side by side: '
                                   f'{os.path.basename(p_)} is missing in {folder}.')
    try:
        cal, sen = json.loads(_load_text(cal_p)), json.loads(_load_text(sen_p))
    except json.JSONDecodeError as e:
        raise FleetImportError(f'A nuScenes table is not valid JSON: {e}.')
    if not isinstance(cal, list) or not isinstance(sen, list):
        raise FleetImportError('The nuScenes tables must be JSON lists.')
    sensors = {s.get('token'): s for s in sen if isinstance(s, dict)}
    readme = os.path.join(folder, 'README.md')
    stamp, frame_words = '', 'rear axle (assumed)'
    if origin is not None:
        o = np.asarray(origin, float)
        frame_words = 'as given'
    elif os.path.isfile(readme):
        o, frame_words, stamp = _readme_frame(_load_text(readme), project)
    else:
        o = np.zeros(3)
    poses, seen, notes = {}, {}, []
    for row in cal:
        if not isinstance(row, dict):
            continue
        s = sensors.get(row.get('sensor_token'))
        if s is None:
            notes.append('a row whose sensor is not in sensor.json')
            continue
        ch, mod = str(s.get('channel', '')), str(s.get('modality', '')).lower()
        extra = row.get('vantage_distortion') or row.get('vantage_sensor') or {}
        tag = str(extra.get('camera') or extra.get('tag') or (ch.split('_', 1)[1] if '_' in ch else ch)).strip()
        if not tag:
            continue
        if tag in seen:
            seen[tag] += 1
            continue
        try:
            Rm = _quat_matrix(row['rotation'])
            t = np.array([float(v) for v in row['translation']], float)
        except (KeyError, TypeError, ValueError):
            raise FleetImportError(f'The pose of {ch} in calibrated_sensor.json is not a number.')
        if t.shape != (3,):
            raise FleetImportError(f'The translation of {ch} is not three numbers.')
        poses[tag] = _pose_from_matrix(Rm @ BODY_TO_OPTICAL if mod == 'camera' else Rm, t + o)
        seen[tag] = 1
    if not poses:
        raise FleetImportError('calibrated_sensor.json holds no sensor of sensor.json.')
    try:
        date = normalise_date(stamp)
    except ValueError:
        date = fallback_date or _file_date(cal_p)
    twice = [t for t, n in seen.items() if n > 1]
    note = f'nuScenes calibration, ego frame {frame_words} (origin {", ".join(f"{v:.3f}" for v in o)} m from the ' \
           f'rear axle on the ground)'
    if twice:
        note += f'; {", ".join(twice)} calibrated more than once: the first row is used'
    if notes:
        note += f'; {len(notes)} {notes[0]}{"s" if len(notes) > 1 else ""} skipped'
    return _record('nuscenes', date, os.path.join(os.path.basename(folder), 'calibrated_sensor.json'), poses, {}, note)


def _rigid_fit(A_pos, A_rot, B_pos, B_rot, w=1.0):
    """The rotation G and translation g that best map the poses A onto B: minimises Σ|G a + g − b|² +
    w Σ|G Ra − Rb|²_F (closed form, an SVD), so even one or two cameras fix it."""
    a0, b0 = A_pos.mean(axis=0), B_pos.mean(axis=0)
    M = (B_pos - b0).T @ (A_pos - a0) + w * sum(Rb @ Ra.T for Ra, Rb in zip(A_rot, B_rot))
    U, _, Vt = np.linalg.svd(M)
    G = U @ np.diag([1.0, 1.0, np.linalg.det(U @ Vt)]) @ Vt
    return G, b0 - G @ a0


def read_kalibr(doc: dict, project, header: str = '', source='', fallback_date='', variant=None) -> dict:
    """A record from a Kalibr camchain (cam0, cam1 … with T_cn_cnm1 from the previous camera). It holds only
    camera-to-camera transforms, so the chain is placed in the vehicle frame by the rigid transform that best fits
    the plan's cameras (positions and axes): a whole-rig shift or turn is invisible, a camera moving against the
    others is not. Cameras are named by Vantage's header line (cam0 = F, …) or by a plan tag in the rostopic."""
    if not isinstance(doc, dict) or 'cam0' not in doc:
        raise FleetImportError('Not a Kalibr camchain: no cam0.')
    if project is None:
        raise FleetImportError('A Kalibr camchain is placed on the plan: open the project first.')
    v = variant or plan_variant(project)
    plan = {p['tag'].lower(): p for p in plan_sensors(project, v) if p['kind'] == 'camera'}
    named = dict(re.findall(r'cam(\d+) = ([^,.\s]+)', header))
    keys = sorted((k for k in doc if re.fullmatch(r'cam\d+', k)), key=lambda k: int(k[3:]))
    T = np.eye(4)                                        # cam_n coordinates -> cam0
    chain, tags = {}, {}
    for i, k in enumerate(keys):
        d = doc[k] if isinstance(doc[k], dict) else {}
        if i and k != f'cam{i}':
            raise FleetImportError(f'The camchain skips from cam{i - 1} to {k}.')
        if i:
            try:
                rel = np.array(d['T_cn_cnm1'], float)             # previous camera -> this one
            except (KeyError, TypeError, ValueError):
                raise FleetImportError(f'{k} has no T_cn_cnm1.')
            if rel.shape != (4, 4) or not np.all(np.isfinite(rel)):
                raise FleetImportError(f'{k}: T_cn_cnm1 is not a 4×4 matrix.')
            T = T @ np.linalg.inv(rel)
        tag = named.get(k[3:])
        if not tag:
            parts = [x for x in str(d.get('rostopic', '')).split('/') if x]
            tag = next((x for x in parts if x.lower() in plan), parts[0] if parts else k)
        tags[k] = tag
        chain[k] = T.copy()
    known = [k for k in keys if tags[k].lower() in plan]
    if not known:
        raise FleetImportError('No camera of the camchain matches a camera of the plan (by the header line or the '
                               'rostopic), so it cannot be placed on the vehicle.')
    A_pos = np.array([chain[k][:3, 3] for k in known])
    A_rot = [chain[k][:3, :3] for k in known]
    B_pos = np.array([[plan[tags[k].lower()]['pose'][a] for a in 'xyz'] for k in known])
    B_rot = [rot_zyx(*_ypr(plan[tags[k].lower()]['pose'])) @ BODY_TO_OPTICAL.T for k in known]
    G, g = _rigid_fit(A_pos, A_rot, B_pos, B_rot)
    poses = {}
    for k in keys:
        Rv = G @ chain[k][:3, :3]                        # optical -> vehicle
        poses[tags[k]] = _pose_from_matrix(Rv @ BODY_TO_OPTICAL, G @ chain[k][:3, 3] + g)
    stamp = re.search(r'exported (\S+)', header)
    try:
        date = normalise_date(stamp.group(1) if stamp else '')
    except ValueError:
        date = fallback_date or today()
    note = (f'Kalibr camchain, {len(keys)} cameras placed on the plan {v.name} by the best rigid fit of '
            f'{len(known)}: camera-to-camera only, a whole-rig shift or turn does not show')
    return _record('kalibr', date, source, poses, {}, note)


def record_from_variant(project, variant, date: str = '') -> dict:
    """A record of a variant's cameras and sensors (an as-built variant, a variant typed from a survey), dated the day
    it was created unless given."""
    poses = {}
    for o in list(variant.cameras) + list(variant.sensors):
        q = clean_pose({k: getattr(o, k) for k in POSE_KEYS})
        if q is not None and o.tag not in poses:
            poses[o.tag] = q
    if not date:
        try:
            date = _dt.date.fromtimestamp(float(variant.created)).isoformat()
        except (TypeError, ValueError, OverflowError, OSError):
            date = today()
    return _record('variant', normalise_date(date), f'variant: {variant.name}', poses, {},
                   f'Poses of the variant {variant.name}')


def manual_record(date: str = '', note: str = '') -> dict:
    return _record('manual', normalise_date(date or today()), 'typed in', {}, {}, note)


def read_file(path: str, project=None, origin=None) -> dict:
    """Read one record from a file or a nuScenes folder, recognised by its content: an installation check report,
    an extrinsics export (JSON, YAML or CSV), nuScenes tables or a Kalibr camchain. FleetImportError says why not."""
    if not path or not os.path.exists(path):
        raise FleetImportError(f'{path or "No file"} does not exist.')
    name = os.path.basename(path.rstrip(os.sep))
    if os.path.isdir(path):
        return read_nuscenes(path, project, origin)
    fdate = _file_date(path)
    text = _load_text(path)
    ext = os.path.splitext(path)[1].lower()
    doc, err = None, ''
    if ext == '.csv':
        return read_extrinsics_csv(text, project, name, fdate)
    if ext not in ('.yaml', '.yml'):
        try:
            doc = json.loads(text)
        except json.JSONDecodeError as e:
            err = f'not valid JSON ({e})'
    if doc is None and ext != '.json':
        try:
            import yaml
            doc = yaml.safe_load(text)
        except Exception as e:                          # yaml.YAMLError and friends
            err = err or f'not valid YAML ({str(e).splitlines()[0]})'
    if doc is None:
        raise FleetImportError(f'{name} is {err or "empty"}.')
    schema = str(doc.get('schema', '')) if isinstance(doc, dict) else ''
    if schema.startswith('vantage.install_check'):
        return read_install_check(doc, name, fdate)
    if schema.startswith('vantage.extrinsics'):
        return read_extrinsics(doc, project, name, fdate)
    if isinstance(doc, list) and doc and all(isinstance(r, dict) for r in doc):
        if 'sensor_token' in doc[0] or ('channel' in doc[0] and 'modality' in doc[0]):
            return read_nuscenes(path, project, origin, fdate)
    if isinstance(doc, dict) and 'cam0' in doc:
        return read_kalibr(doc, project, text[:4000], name, fdate)
    if isinstance(doc, dict) and isinstance(doc.get('cameras'), list) and \
            any(isinstance(r, dict) and 'measured' in r for r in doc['cameras']):
        return read_install_check(doc, name, fdate)
    if schema:
        raise FleetImportError(f'{name} is a {schema} file, which holds no measured poses.')
    raise FleetImportError(f'{name} is not a file Vantage reads poses from: use an installation check report, an '
                           'extrinsics export (JSON, YAML or CSV), nuScenes calibrated_sensor.json with sensor.json, '
                           'or a Kalibr camchain.')


def import_files(project, vehicle_name: str, paths, origin=None) -> dict:
    """Read records from files into the vehicle of that name (ignoring case; added when there is none) of
    project.fleet, in place (batch mode: no undo). Returns a report: the vehicle, every record imported with how its
    tags matched the plan, and the files that could not be read."""
    if not isinstance(project.fleet, dict):
        project.fleet = {}
    vs = _ensure(project.fleet)
    name = str(vehicle_name or '').strip()
    v = next((x for x in vs if x['name'].lower() == name.lower()), None) if name else None
    if v is None:
        v = add_vehicle(project.fleet, name)
    done, errors = [], []
    for path in paths:
        try:
            r = add_record(project.fleet, v['id'], read_file(path, project, origin))
        except FleetImportError as e:
            errors.append({'file': os.path.basename(str(path)), 'error': str(e)})
            continue
        rep = import_report(r, project)
        done.append({'file': os.path.basename(str(path).rstrip(os.sep)), 'kind': r['kind'], 'date': r['date'],
                     'poses': rep['poses'], 'matched': rep['matched'], 'unknown': rep['unknown'],
                     'hidden': rep['hidden'], 'note': r['note']})
    return {'schema': 'vantage.fleet_import/1', 'vehicle': v['name'], 'imported': done, 'errors': errors}


# =========================================================================== the Issues list
def _latest(vrec, plan_tags, hidden, cache) -> dict:
    """Plan tag -> (record, record tag, cleaned pose) of the latest record that has a valid pose for it, reading the
    raw records newest first and cleaning only what it reaches (the Issues list's cheap path; the same choice as
    _series makes on cleaned records)."""
    order = sorted(((date_days(r.get('date')), i, r) for i, r in enumerate(vrec) if isinstance(r, dict)),
                   key=lambda x: (-math.inf if x[0] is None else x[0], x[1]), reverse=True)
    need, out = set(plan_tags), {}
    for _, _, r in order:
        raw = r.get('poses') if isinstance(r.get('poses'), dict) else {}
        poses = {}
        for tag, q in raw.items():
            q = clean_pose(q)
            if q is not None and str(tag).strip():
                poses[str(tag).strip()] = q
        key = tuple(poses)
        m = cache.get(key)
        if m is None:
            m = cache[key] = match_tags(key, plan_tags, hidden)
        for rt, pt in m['map'].items():
            if pt in need:
                out[pt] = (r, rt, poses[rt])
                need.discard(pt)
        if not need:
            break
    return out


def issues(project) -> list:
    """Sensors out of tolerance in their vehicle's latest record (cheap: no drift or fleet statistics, and only the
    newest records are read): adjust is a warning, re-mount serious. Uids are the plan's (clicking selects the sensor
    when the plan is the active variant)."""
    f = getattr(project, 'fleet', None)
    if not isinstance(f, dict) or not isinstance(f.get('vehicles'), list) or not f['vehicles']:
        return []
    st = settings_of(project)
    v = plan_variant(project, st)
    plan = plan_sensors(project, v)
    if not plan:
        return []
    tags = [p['tag'] for p in plan]
    hidden = _hidden_tags(v)
    active = v is project.variant
    out, cache = [], {}
    for veh in f['vehicles']:
        if not isinstance(veh, dict) or not isinstance(veh.get('records'), list):
            continue
        name = str(veh.get('name') or 'Vehicle').strip() or 'Vehicle'
        latest = _latest(veh['records'], tags, hidden, cache)
        for ps in plan:
            hit = latest.get(ps['tag'])
            if hit is None:
                continue
            r, _, pose = hit
            d = delta(ps['pose'], pose)
            rot = float(rotation_deg(ps['pose'], d))
            verdict = verdict_of(float(ratio_of(d, rot, st)))
            if verdict in ('adjust', 're-mount'):
                c = {'d': d.tolist(), 'rot_deg': rot}
                kind = r.get('kind') if r.get('kind') in RECORD_KINDS else 'manual'
                out.append(R.Issue('serious' if verdict == 're-mount' else 'warning',
                                   f'Fleet: {ps["tag"]} on {name} is {_worst_part(c, st)} off the plan in its latest '
                                   f'record ({_date_text(r.get("date")) or "undated"}, '
                                   f'{str(r.get("source") or "") or KIND_LABEL[kind]}): {verdict}.',
                                   f'{name} · {ps["tag"]}', [ps['uid']] if active else []))
    return out


# =========================================================================== output
def _r(v, d=3):
    if v is None:
        return None
    v = float(v)
    return round(v, d) + 0.0 if math.isfinite(v) else None


def _rl(vals, d=3):
    return None if vals is None else [_r(x, d) for x in vals]


def summary(res: FleetResult) -> dict:
    """Compact figures (the assistant, the batch log)."""
    cells = list(res.cells.values())
    worst = sorted(cells, key=lambda c: -c['ratio'])[:8]
    name = {v['id']: v['name'] for v in res.vehicles}
    return {'plan': res.variant, 'vehicles': len(res.vehicles), 'records': sum(v['records'] for v in res.vehicles),
            'sensors': len(res.plan),
            'tolerance': {'position_mm': res.settings.pos_tol_mm, 'angle_deg': res.settings.ang_tol_deg},
            'verdicts': {k: sum(c['verdict'] == k for c in cells) for k in VERDICTS[:3]},
            'not_measured': sum(len(v['missing']) for v in res.vehicles if v['records']),
            'worst': [{'vehicle': name[c['vehicle']], 'sensor': c['tag'], 'verdict': c['verdict'],
                       'share_of_tolerance': _r(c['ratio'], 2), 'offset_mm': _rl(c['d'][:3], 1),
                       'angles_deg': _rl(c['d'][3:], 2), 'date': c['date']} for c in worst if c['ratio'] > 1.0],
            'drifting': [{'vehicle': name[v], 'sensor': t,
                          'trend_per_30d': _rl(res.cells[(v, t)]['drift']['trend_per_30d'], 3),
                          'days_to_tolerance': _r(res.cells[(v, t)]['drift']['days_to_tol'], 0)}
                         for v, t in res.flagged('drift')],
            'jumps': [{'vehicle': name[v], 'sensor': t, 'step': _rl(res.cells[(v, t)]['drift']['step'], 2)}
                      for v, t in res.flagged('jump')],
            'outliers': [{'vehicle': o['name'], 'sensor': f['tag'], 'component': o['component'],
                          'from_median': _r(o['from_median'], 2), 'z': _r(o['z'], 1), 'cause': o['cause']}
                         for f in res.fleet for o in f['outliers']],
            'systematic': [{'sensor': f['tag'], 'median_mm': _rl(f['median'][:3], 1),
                            'median_deg': _rl(f['median'][3:], 2), 'vehicles': f'{f["near_median"]} of {f["n"]}'}
                           for f in res.fleet if f['systematic']],
            'unknown_tags': {v['name']: [u['tag'] for u in v['unknown']] for v in res.vehicles if v['unknown']},
            'findings': [{'severity': f['severity'], 'message': f['message']} for f in res.findings
                         if f['severity'] != 'good'],
            'assumptions': ASSUMPTIONS}


def _cell_doc(c) -> dict:
    d = c['drift']
    out = {'sensor': c['tag'], 'kind': c['kind'], 'verdict': c['verdict'], 'share_of_tolerance': _r(c['ratio'], 4),
           'date': c['date'], 'record': c['record'], 'source': c['source'], 'record_kind': c['record_kind'],
           'records': c['records'], 'offset_mm': _rl(c['d'][:3], 2), 'offset_total_mm': _r(c['dpos_mm'], 2),
           'angles_deg': _rl(c['d'][3:], 4), 'rotation_deg': _r(c['rot_deg'], 4),
           'measured': {k: _r(v, 5) for k, v in c['pose'].items()}, 'quality': c['quality']}
    if d:
        out['drift'] = {'records': d['n'], 'span_days': _r(d['span_days'], 2), 'change': _rl(d['change'], 4),
                        'trend_per_30_days': _rl(d['trend_per_30d'], 4), 'significant': d['significant'],
                        'share_of_tolerance_per_window': _r(d['rate_ratio'], 4),
                        'days_to_tolerance': _r(d['days_to_tol'], 1), 'drifting': d['trend'],
                        'latest_step': _rl(d['step'], 4), 'jump_share_of_tolerance': _r(d['jump_ratio'], 4),
                        'jump': d['jump']}
    if c['outlier']:
        o = c['outlier']
        out['outlier'] = {'component': o['component'], 'from_fleet_median': _r(o['from_median'], 4),
                          'robust_z': _r(o['z'], 2), 'cause': o['cause']}
    return out


def to_doc(res: FleetResult, project=None) -> dict:
    """The JSON export: the plan, every vehicle with its records (the poses as measured) and its latest deltas, the
    fleet statistics per sensor and the findings."""
    vs = {v['id']: v for v in vehicles(project)} if project is not None else {}
    vdocs = []
    for r in res.vehicles:
        recs = vs.get(r['id'], {}).get('records', [])
        vdocs.append({
            'id': r['id'], 'name': r['name'], 'vin': r['vin'], 'note': r['note'], 'verdict': r['verdict'],
            'worst_share_of_tolerance': _r(r['worst'], 4), 'not_measured': r['missing'] if r['records'] else [],
            'unknown_tags': [u['tag'] for u in r['unknown']], 'hidden_tags': [u['tag'] for u in r['hidden']],
            'records': [{'id': x['id'], 'date': x['date'], 'kind': x['kind'], 'source': x['source'], 'note': x['note'],
                         'poses': {t: {k: _r(v, 6) for k, v in q.items()} for t, q in x['poses'].items()},
                         'quality': x['quality']} for x in recs],
            'sensors': [_cell_doc(res.cells[(r['id'], p['tag'])]) for p in res.plan
                        if (r['id'], p['tag']) in res.cells],
        })
    return {
        'schema': SCHEMA, 'project': getattr(project, 'name', ''), 'variant': res.variant,
        'frame': FRAME_TEXT, 'rotation': ROTATION_TEXT,
        'units': {'poses': 'm and deg', 'offsets': 'mm', 'angles': 'deg', 'trends': 'per 30 days',
                  'dates': 'ISO 8601'},
        'settings': {k: v for k, v in res.settings.to_dict().items() if k != 'variant'},
        'verdicts': {'ok': 'worst axis within the position tolerance and rotation within the angle tolerance',
                     'adjust': f'within {REMOUNT:g} times the tolerance', 're-mount': f'beyond {REMOUNT:g} times'},
        'assumptions': ASSUMPTIONS,
        'plan': [{'sensor': p['tag'], 'kind': p['kind'], 'pose': {k: _r(v, 6) for k, v in p['pose'].items()}}
                 for p in res.plan],
        'vehicles': vdocs,
        'fleet': [{'sensor': f['tag'], 'kind': f['kind'], 'vehicles': f['n'], 'median_offset_mm': _rl(
            f['median'][:3], 3) if f['median'] else None, 'median_angles_deg': _rl(f['median'][3:], 4)
            if f['median'] else None, 'robust_sigma': _rl(f['sigma'], 4), 'median_share_of_tolerance':
            _r(f['median_ratio'], 4), 'beyond_tolerance': f['beyond'], 'systematic': f['systematic'],
            'outliers': [{'vehicle': o['name'], 'component': o['component'],
                          'from_fleet_median': _r(o['from_median'], 4), 'robust_z': _r(o['z'], 2), 'cause': o['cause']}
                         for o in f['outliers']]} for f in res.fleet],
        'findings': [{'severity': f['severity'], 'kind': f['kind'], 'message': f['message'], 'sensors': f['tags']}
                     for f in res.findings],
    }


CSV_COLUMNS = ('vehicle', 'vin', 'sensor', 'kind', 'verdict', 'share_of_tolerance', 'date', 'source', 'records',
               'dx_mm', 'dy_mm', 'dz_mm', 'offset_total_mm', 'dyaw_deg', 'dpitch_deg', 'droll_deg', 'rotation_deg',
               'trend_x_mm_30d', 'trend_y_mm_30d', 'trend_z_mm_30d', 'trend_yaw_deg_30d', 'trend_pitch_deg_30d',
               'trend_roll_deg_30d', 'days_to_tolerance', 'drifting', 'jump', 'fleet_z', 'outlier')


def to_csv(res: FleetResult) -> str:
    """The matrix: one row per vehicle and plan sensor (not measured ones too)."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator='\n')
    w.writerow(CSV_COLUMNS)
    z = {f['tag']: f['z'] for f in res.fleet}
    for v in res.vehicles:
        for p in res.plan:
            c = res.cells.get((v['id'], p['tag']))
            if c is None:
                w.writerow([v['name'], v['vin'], p['tag'], p['kind'], 'missing'] + [''] * (len(CSV_COLUMNS) - 5))
                continue
            d = c['drift']
            tr = d['trend_per_30d'] if d else [None] * 6
            row = [v['name'], v['vin'], p['tag'], p['kind'], c['verdict'], _r(c['ratio'], 3), c['date'], c['source'],
                   c['records']] + _rl(c['d'][:3], 2) + [_r(c['dpos_mm'], 2)] + _rl(c['d'][3:], 4) + \
                [_r(c['rot_deg'], 4)] + [_r(x, 4) for x in tr] + \
                [_r(d['days_to_tol'], 1) if d else None, int(bool(d and d['trend'])), int(bool(d and d['jump'])),
                 _r(z[p['tag']].get(v['id']), 2), int(c['outlier'] is not None)]
            w.writerow(['' if x is None else x for x in row])
    return buf.getvalue()


# ---- the PNG (OpenCV, no Qt): the matrix and the drift of the flagged cells
STATUS_HEX = {'good': '#0ca30c', 'warning': '#fab219', 'serious': '#ec835a', 'critical': '#d03b3b'}
AXIS_HEX = ('#e5534b', '#57ab5a', '#539bf5')            # x / roll red, y / pitch green, z / yaw blue
CAM_HEX = ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#008300', '#9085e9', '#e66767']
SENSOR_HEX = {'lidar': '#0aa5b8', 'radar': '#a0743c', 'gnss': '#7f9a1c', 'imu': '#6e7f96'}


def ratio_fill(ratio, verdict) -> tuple:
    """(status hex, opacity) of a matrix cell: the status colour, stronger the nearer the tolerance."""
    if verdict == 'missing' or ratio is None:
        return None, 0.0
    if verdict == 'ok':
        return STATUS_HEX['good'], 0.18 + 0.42 * min(max(ratio, 0.0), 1.0)
    if verdict == 'adjust':
        return STATUS_HEX['warning'], 0.55 + 0.3 * min((ratio - 1.0) / (REMOUNT - 1.0), 1.0)
    return STATUS_HEX['serious'], 0.9


def cell_text(c, st) -> str:
    """The dominant part of a cell's offset, short: '2.1°' or '14 mm'."""
    d = c['d']
    pr = max(abs(x) for x in d[:3]) / st.pos_tol_mm
    if c['rot_deg'] / st.ang_tol_deg >= pr:
        return f'{c["rot_deg"]:.1f}°' if c['rot_deg'] < 10 else f'{c["rot_deg"]:.0f}°'
    return f'{max(abs(x) for x in d[:3]):.0f} mm'


def _bgr(hex_):
    return int(hex_[5:7], 16), int(hex_[3:5], 16), int(hex_[1:3], 16)


def _mix(hex_, a, bg=(30, 29, 28)):
    c = _bgr(hex_)
    return tuple(int(round(a * x + (1 - a) * y)) for x, y in zip(c, bg))


def png(res: FleetResult, width=1200, cells=None) -> bytes:
    """The matrix (vehicles × plan sensors, coloured by the share of the tolerance) and, for up to four cells (the
    given ones, else the drifting, jumping and worst), their offsets over time with the tolerance band. ASCII text
    only (OpenCV's fonts)."""
    import cv2
    bg, ink, ink2, muted, grid = (30, 29, 28), (236, 235, 230), (183, 194, 195), (129, 135, 137), (62, 61, 58)
    font, aa = cv2.FONT_HERSHEY_SIMPLEX, cv2.LINE_AA
    st = res.settings
    tags = [p['tag'] for p in res.plan]
    lab_w = 170
    cw = int(min(72, max(34, (width - lab_w - 24) / max(len(tags), 1))))
    ch = 26
    mat_h = 46 + ch * max(len(res.vehicles), 1) + 34
    pick = []
    for k in list(cells or []) + res.flagged('drift') + res.flagged('jump') + \
            sorted(res.cells, key=lambda k: -res.cells[k]['ratio']):
        if k in res.cells and res.cells[k]['drift'] and k not in pick:
            pick.append(k)
    pick = pick[:4]
    chart_h = 190
    H = 70 + mat_h + (30 + chart_h * len(pick) if pick else 0) + 20
    img = np.full((H, width, 3), bg, np.uint8)

    def text(t, x, y, col, s=0.45, th=1):
        t = t.replace('°', ' deg').replace('Δ', 'd').replace('·', '-').replace('×', 'x').encode('ascii', 'replace') \
            .decode('ascii')
        cv2.putText(img, t, (int(x), int(y)), font, s, col, th, aa)
    nv = len(res.vehicles)
    counts = {k: sum(c['verdict'] == k for c in res.cells.values()) for k in VERDICTS[:3]}
    text(f'Fleet against {res.variant}', 16, 28, ink, 0.65)
    n_j, n_o = len(res.flagged('jump')), len(res.flagged('outlier'))
    text(f'{nv} vehicle{"" if nv == 1 else "s"}, {len(tags)} sensors; tolerance {st.pos_tol_mm:g} mm / '
         f'{st.ang_tol_deg:g} deg; {counts["ok"]} ok, {counts["adjust"]} adjust, {counts["re-mount"]} re-mount; '
         f'{len(res.flagged("drift"))} drifting, {n_j} jump{"" if n_j == 1 else "s"}, '
         f'{n_o} fleet outlier{"" if n_o == 1 else "s"}', 16, 52, ink2, 0.45)
    y0 = 70
    for j, p in enumerate(res.plan):
        x = lab_w + j * cw
        col = _bgr(CAM_HEX[p['slot'] % 8]) if p['kind'] == 'camera' else _bgr(SENSOR_HEX.get(p['kind'], '#888888'))
        cv2.rectangle(img, (x + 4, y0 + 10), (x + 12, y0 + 18), col, -1)
        text(p['tag'][:6], x + 16, y0 + 19, ink, 0.42)
    y = y0 + 30
    if not res.vehicles:
        text('No vehicles.', 16, y + 18, muted)
    for v in res.vehicles:
        text(v['name'][:22], 16, y + 18, ink, 0.45)
        for j, p in enumerate(res.plan):
            x = lab_w + j * cw
            c = res.cells.get((v['id'], p['tag']))
            box = (x + 2, y + 2), (x + cw - 2, y + ch - 2)
            if c is None:
                cv2.rectangle(img, *box, grid, 1)
                text('-', x + cw / 2 - 4, y + 18, muted, 0.4)
                continue
            hx, a = ratio_fill(c['ratio'], c['verdict'])
            cv2.rectangle(img, *box, _mix(hx, a), -1)
            t = cell_text(c, st)
            (tw, _), _ = cv2.getTextSize(t.replace('°', ' deg'), font, 0.38, 1)
            text(t, x + (cw - tw) / 2, y + 17, (20, 20, 20) if a > 0.6 else ink, 0.38)
            d = c['drift']
            if (d and (d['trend'] or d['jump'])) or c['outlier']:
                cv2.circle(img, (x + cw - 7, y + 7), 3, ink, -1, aa)
        y += ch
    text('fill: share of the tolerance (green within, amber adjust, orange re-mount); dot: drifting, jump or fleet '
         'outlier; -: not measured', 16, y + 22, muted, 0.4)
    y += 34
    if pick:
        y += 10
        names = {v['id']: v['name'] for v in res.vehicles}
        for vid, tag in pick:
            c = res.cells[(vid, tag)]
            d = c['drift']
            flag = 'drifting' if d['trend'] else 'jump' if d['jump'] else ''
            text(f'{names[vid]} - {tag}: offset from the plan over time' + (f' ({flag})' if flag else ''), 16, y + 14,
                 ink, 0.5)
            half = (width - 48) // 2
            for panel in range(2):
                X0, Y0, Wp, Hp = 16 + panel * (half + 16) + 40, y + 26, half - 48, chart_h - 60
                ks = (0, 1, 2) if panel == 0 else (3, 4, 5)
                tol = st.pos_tol_mm if panel == 0 else st.ang_tol_deg
                ts = np.array([s['days'] for s in d['series']])
                V = np.array([[s['d'][k] for k in ks] for s in d['series']])
                t1 = ts[-1] + (st.drift_window_days if d['trend'] else 0.0)
                tspan = max(t1 - ts[0], 1.0)
                lim = max(tol * 1.5, float(np.abs(V).max()) * 1.15)
                if d['trend']:
                    proj = V[-1] + np.array([d['trend_per_30d'][k] / 30.0 * st.drift_window_days *
                                             d['significant'][k] for k in ks])
                    lim = max(lim, float(np.abs(proj).max()) * 1.1)
                px = lambda t: X0 + (t - ts[0]) / tspan * Wp                    # noqa: E731
                py = lambda v: Y0 + Hp / 2 - v / lim * Hp / 2                   # noqa: E731
                cv2.rectangle(img, (int(X0), int(py(tol))), (int(X0 + Wp), int(py(-tol))), (44, 52, 44), -1)
                cv2.line(img, (int(X0), int(py(0))), (int(X0 + Wp), int(py(0))), grid, 1, aa)
                cv2.rectangle(img, (int(X0), int(Y0)), (int(X0 + Wp), int(Y0 + Hp)), grid, 1)
                text(f'+{lim:.1f}' if panel else f'+{lim:.0f}', X0 - 40, Y0 + 10, muted, 0.35)
                text(f'-{lim:.1f}' if panel else f'-{lim:.0f}', X0 - 40, Y0 + Hp, muted, 0.35)
                lx = X0
                for word, col in ((('position, mm:', muted), ('x', AXIS_HEX[0]), ('y', AXIS_HEX[1]), ('z', AXIS_HEX[2]))
                                  if panel == 0 else (('angle, deg:', muted), ('yaw', AXIS_HEX[2]),
                                                      ('pitch', AXIS_HEX[1]), ('roll', AXIS_HEX[0]))) + \
                        ((f'band +-{tol:g}', muted),):
                    c_ = col if isinstance(col, tuple) else _bgr(col)
                    text(word, lx, Y0 + Hp + 16, c_, 0.38)
                    lx += cv2.getTextSize(word, font, 0.38, 1)[0][0] + 8
                text(d['series'][0]['date'][:10], X0, Y0 + Hp + 32, muted, 0.35)
                text(d['series'][-1]['date'][:10], px(ts[-1]) - 60, Y0 + Hp + 32, muted, 0.35)
                for i, k in enumerate(ks):
                    col = _bgr(AXIS_HEX[i if panel == 0 else 2 - i])
                    pts = np.array([[px(t), py(v)] for t, v in zip(ts, V[:, i])], np.int32)
                    cv2.polylines(img, [pts], False, col, 2, aa)
                    for q in pts:
                        cv2.circle(img, tuple(int(a) for a in q), 3, col, -1, aa)
                    if d['trend'] and d['significant'][k]:
                        e = V[-1, i] + d['trend_per_30d'][k] / 30.0 * st.drift_window_days
                        cv2.line(img, (int(px(ts[-1])), int(py(V[-1, i]))), (int(px(t1)), int(py(e))), col, 1, aa)
            y += chart_h
    ok, buf = cv2.imencode('.png', img)
    return buf.tobytes()


def files(res: FleetResult, project=None, cells=None) -> list:
    """[(name, bytes)]: fleet.json, fleet.csv (the matrix) and fleet.png (the matrix and drift charts)."""
    return [('fleet.json', (json.dumps(to_doc(res, project), indent=1, ensure_ascii=False) + '\n').encode('utf-8')),
            ('fleet.csv', to_csv(res).encode('utf-8')),
            ('fleet.png', png(res, cells=cells))]


def export(res: FleetResult, project, out_dir, cells=None) -> list[str]:
    """Write fleet.json, fleet.csv and fleet.png into out_dir; returns their paths."""
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for name, data in files(res, project, cells):
        path = os.path.join(out_dir, name)
        with open(path, 'wb') as fh:
            fh.write(data)
        paths.append(path)
    return paths
