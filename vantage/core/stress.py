"""Calibration perturbation and out-of-distribution stress tests: seeded, reproducible schedules of how every camera
and sensor of a rig leaves its calibration over a drive, what each shift does to the ground and to the image, and
the shifted calibrations themselves, so an end-to-end planner, a VLA policy or a BEV detector can be evaluated on
them.

Over a drive timeline (rate_hz, duration_s) each sensor's pose is the plan's moved by:

- **thermal drift**: the ambient temperature rises by thermal_dT over thermal_warm_tau_s (a first-order warm-up) and
  each mount follows it with its own first-order lag (thermal_tau_s); the mount turns by thermal_deg_per_C and moves
  by thermal_mm_per_C per degree, about a direction drawn per sensor. Bounded by dT × the coefficient.
- **body motion**: pitch and roll of the body under the drive's braking, accelerating and cornering, from the
  project's own suspension model (core.terrain.motion_cases: quasi-static load transfer), as gradients in degrees per
  g fitted over its motion cases. It turns every sensor together about the rear-axle centre on the ground: in a
  dataset this is the ego pose's pitch and roll against the ground, which a BEV or occupancy model that assumes a
  flat ground under a level body sees as a calibration error.
- **knocks**: steps at random times on single cameras (a car wash, a branch, a door slam): their count Poisson at
  knock_rate_per_min, their size uniform in knock_deg about a random axis, kept from then on.
- **vibration**: band-limited noise (vib_band_hz, below the timeline's Nyquist rate: what a 10–30 Hz timeline can
  carry, the body's shake, not the engine's or the road's higher bands) of RMS vib_rms_deg and vib_rms_mm per axis.

The severity levels ('none', 'mild', 'moderate', 'severe') are stress-test presets chosen to span from a well-kept
rig to a neglected one. They are illustrative test levels, not measurements of any bracket, camera or vehicle; every
figure can be overridden in the settings.

Effects, per camera and frame: the ground-projection error at effect_ranges metres ahead along the camera's aim
(the pixel that saw a ground point now sees another one; core.research_maps.drift_shift_cm generalised to the whole
rotation and translation), the pixel shift of a point far along the optical axis, and the BEV cell displacement
(the error at each range over bev_cell_m).

Frames: ISO 8855 vehicle frame, rear-axle centre on the ground; sensor rotations are rot_zyx(yaw, pitch, roll) with
pitch positive down; mount perturbations are rotation vectors about the sensor's own axes (x forward, y left, z up).
"""
from __future__ import annotations

import copy
import csv
import io
import json
import math
import os
from dataclasses import dataclass, field, asdict

import numpy as np

from .mathutil import rot_zyx, euler_zyx, wrap180, BODY_TO_OPTICAL

SCHEMA = 'vantage.stress/1'

# illustrative stress-test levels (see the module docstring): not measurements of any product
SEVERITY = {
    'none': dict(thermal_dT=0.0, thermal_deg_per_C=0.0, thermal_mm_per_C=0.0, knock_rate_per_min=0.0,
                 knock_deg=(0.0, 0.0), vib_rms_deg=0.0, vib_rms_mm=0.0, brake_g=0.0, accel_g=0.0, corner_g=0.0),
    'mild': dict(thermal_dT=10.0, thermal_deg_per_C=0.002, thermal_mm_per_C=0.005, knock_rate_per_min=0.0,
                 knock_deg=(0.1, 0.3), vib_rms_deg=0.01, vib_rms_mm=0.05, brake_g=0.2, accel_g=0.15, corner_g=0.2),
    'moderate': dict(thermal_dT=25.0, thermal_deg_per_C=0.004, thermal_mm_per_C=0.01, knock_rate_per_min=0.5,
                     knock_deg=(0.3, 1.0), vib_rms_deg=0.03, vib_rms_mm=0.15, brake_g=0.4, accel_g=0.25,
                     corner_g=0.35),
    'severe': dict(thermal_dT=40.0, thermal_deg_per_C=0.008, thermal_mm_per_C=0.02, knock_rate_per_min=2.0,
                   knock_deg=(1.0, 3.0), vib_rms_deg=0.08, vib_rms_mm=0.4, brake_g=0.8, accel_g=0.3, corner_g=0.5),
}
SEVERITY_TEXT = {
    'none': 'no perturbation: the plan itself',
    'mild': 'a well-kept rig: a 10 °C warm-up, light vibration, everyday braking and cornering, no knocks',
    'moderate': 'a working rig: a 25 °C warm-up, a knock every two minutes or so (0.3–1°), firm braking',
    'severe': 'a neglected rig: a 40 °C warm-up, two knocks a minute (1–3°), emergency braking, hard cornering',
}
DEFAULT_FORMATS = ('nuscenes_calib', 'kalibr_camchain', 'kitti_calib')


@dataclass
class StressSettings:
    severity: str = 'moderate'
    seed: int = 7
    duration_s: float = 60.0
    rate_hz: float = 10.0
    # overrides of the severity's figures (None: the severity's)
    thermal_dT: float | None = None              # °C ambient rise
    thermal_deg_per_C: float | None = None       # mount rotation per °C
    thermal_mm_per_C: float | None = None        # mount translation per °C
    thermal_warm_tau_s: float = 600.0            # ambient warm-up time constant
    thermal_tau_s: float = 120.0                 # each mount's lag behind the ambient
    knock_rate_per_min: float | None = None
    knock_deg: tuple | None = None               # (smallest, largest) knock, degrees
    knock_cameras_only: bool = True
    vib_rms_deg: float | None = None
    vib_rms_mm: float | None = None
    vib_band_hz: tuple = (0.5, 4.0)
    brake_g: float | None = None
    accel_g: float | None = None
    corner_g: float | None = None
    sensors: list = field(default_factory=list)  # tags to perturb ([]: every visible camera and sensor)
    effect_ranges: tuple = (10.0, 20.0, 40.0)
    bev_cell_m: float = 0.5
    sets: int = 5                                # perturbed calibration sets to export (the worst moment included)

    def level(self, k):
        v = getattr(self, k)
        return SEVERITY[self.severity][k] if v is None else v

    def to_dict(self):
        d = asdict(self)
        d['knock_deg'] = list(self.level('knock_deg'))
        d['vib_band_hz'] = list(self.vib_band_hz)
        d['effect_ranges'] = list(self.effect_ranges)
        d['resolved'] = {k: (list(self.level(k)) if k == 'knock_deg' else self.level(k)) for k in SEVERITY['none']}
        return d

    @staticmethod
    def from_dict(d) -> 'StressSettings':
        s = StressSettings()
        for k, v in (d or {}).items():
            if not hasattr(s, k) or k == 'resolved':
                continue
            try:
                if k == 'severity':
                    v = str(v).lower()
                    if v not in SEVERITY:
                        continue
                elif k in ('sensors',):
                    v = [str(x) for x in v]
                elif k in ('knock_deg', 'vib_band_hz', 'effect_ranges'):
                    v = tuple(float(x) for x in v)
                elif k in ('seed', 'sets'):
                    v = int(v)
                elif k == 'knock_cameras_only':
                    v = bool(v)
                elif v is not None:
                    v = float(v)
                    if not math.isfinite(v):
                        continue
            except (TypeError, ValueError):
                continue
            setattr(s, k, v)
        s.rate_hz = min(max(s.rate_hz, 1.0), 100.0)
        s.duration_s = min(max(s.duration_s, 1.0 / s.rate_hz), 3600.0)
        s.sets = min(max(s.sets, 0), 50)
        return s


# ---------------------------------------------------------------- small rotations
def rotvec(v) -> np.ndarray:
    """Rotation matrix of a rotation vector (radians), Rodrigues."""
    v = np.asarray(v, float)
    a = float(np.linalg.norm(v))
    if a < 1e-15:
        return np.eye(3)
    k = v / a
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K


def _unit(rng, n=3):
    u = rng.normal(size=n)
    return u / max(np.linalg.norm(u), 1e-12)


def _band_noise(rng, n, rate, band, rms):
    """(n,) noise band-limited to band (Hz) by masking its spectrum, scaled to rms (0 rms or an empty band: zeros)."""
    if rms <= 0 or n < 4:
        return np.zeros(n)
    w = rng.normal(size=n)
    F = np.fft.rfft(w)
    f = np.fft.rfftfreq(n, 1.0 / rate)
    lo, hi = band[0], min(band[1], 0.45 * rate)
    F[(f < lo) | (f > hi)] = 0.0
    x = np.fft.irfft(F, n)
    s = float(np.sqrt(np.mean(x ** 2)))
    return x / s * rms if s > 1e-12 else np.zeros(n)


# ---------------------------------------------------------------- the drive and the body
def drive_profile(t, st: StressSettings):
    """(long_g, lat_g) over the timeline: a repeating 30 s cycle (accelerate 5 s, cruise, brake 3 s, a left and a
    right corner of 5 s each, cruise) at the severity's accelerations, with 0.5 s ramps."""
    c = np.mod(t, 30.0)
    ax = np.where(c < 5, st.level('accel_g'), 0.0) + np.where((c >= 12) & (c < 15), -st.level('brake_g'), 0.0)
    ay = np.where((c >= 15) & (c < 20), st.level('corner_g'), 0.0) + np.where((c >= 20) & (c < 25),
                                                                             -st.level('corner_g'), 0.0)
    k = max(1, int(round(0.5 * st.rate_hz)))
    ker = np.ones(k) / k
    return np.convolve(ax, ker, 'same'), np.convolve(ay, ker, 'same')


def body_gradients(project) -> dict:
    """The body's pitch (degrees per g of longitudinal acceleration, + accelerating; nose down positive) and roll
    (degrees per g lateral, + turning left) from the project's suspension model, fitted over core.terrain's motion
    cases relative to the load the rig is calibrated at. Zero, with a note, when the model cannot run."""
    try:
        from . import terrain as TR
        mc = TR.motion_cases(project)
    except Exception as ex:                       # no vehicle, or a suspension model that does not solve
        return {'pitch_per_g': 0.0, 'roll_per_g': 0.0, 'note': f'body motion left out: {ex}'}
    lg = [(c['long_g'], c['rel']['pitch_deg']) for c in mc['cases'] if abs(c['lat_g']) < 1e-9]
    la = [(c['lat_g'], c['rel']['roll_deg']) for c in mc['cases'] if abs(c['long_g']) < 1e-9]

    def slope(pts):
        x = np.array([p[0] for p in pts], float)
        y = np.array([p[1] for p in pts], float)
        return float(np.dot(x, y) / np.dot(x, x)) if len(x) and np.dot(x, x) > 0 else 0.0
    return {'pitch_per_g': slope(lg), 'roll_per_g': slope(la),
            'note': 'quasi-static load transfer of the project\'s suspension model (core.suspension), '
                    + ('typical values' if mc['info'].get('typical_suspension') else 'the project\'s values')}


# ---------------------------------------------------------------- the schedule
@dataclass
class Schedule:
    settings: StressSettings
    t: np.ndarray                    # (n,) seconds
    long_g: np.ndarray
    lat_g: np.ndarray
    body: dict                       # pitch_deg, roll_deg (n,), gradients
    sensors: list                    # per sensor: dict (see schedule())
    knocks: list                     # [{t_s, tag, deg, axis}]

    def index(self, t_s) -> int:
        return int(np.clip(np.round(float(t_s) * self.settings.rate_hz), 0, len(self.t) - 1))

    def sensor(self, tag):
        return next((s for s in self.sensors if s['tag'] == tag), None)


def _items(project, st: StressSettings):
    """[(kind, object)] of the visible cameras and sensors of the active variant (or the named tags)."""
    v = project.variant
    out = [('camera', c) for c in v.cameras if c.visible] + [('sensor', s) for s in v.sensors if s.visible]
    if st.sensors:
        out = [(k, o) for k, o in out if o.tag in st.sensors]
    return out


def _R(o):
    return rot_zyx(o.yaw, o.pitch, o.roll)


def schedule(project, settings: StressSettings | dict | None = None) -> Schedule:
    """The perturbation schedule of the project's active variant: per frame the body's pitch and roll and per sensor
    its mount rotation (deg, about its own axes) and translation (mm, its own axes) from thermal drift, knocks and
    vibration, the resulting vehicle-frame pose and, for cameras, the effects (see the module docstring)."""
    st = settings if isinstance(settings, StressSettings) else StressSettings.from_dict(settings)
    rng = np.random.default_rng([int(st.seed), 4021])
    n = max(int(round(st.duration_s * st.rate_hz)), 1)
    t = np.arange(n) / st.rate_hz
    ax, ay = drive_profile(t, st)
    grad = body_gradients(project) if (st.level('brake_g') or st.level('accel_g') or st.level('corner_g')) else \
        {'pitch_per_g': 0.0, 'roll_per_g': 0.0, 'note': 'no driving accelerations at this severity'}
    pitch_b = grad['pitch_per_g'] * ax
    roll_b = grad['roll_per_g'] * ay
    from . import loadcases as LC
    items = _items(project, st)
    # thermal: ambient warm-up, each mount lagging behind it
    dT = st.level('thermal_dT')
    amb = dT * (1.0 - np.exp(-t / max(st.thermal_warm_tau_s, 1e-6)))
    a = 1.0 - math.exp(-1.0 / (st.rate_hz * max(st.thermal_tau_s, 1e-6)))
    mount_T = np.zeros(n)
    for i in range(1, n):
        mount_T[i] = mount_T[i - 1] + (amb[i] - mount_T[i - 1]) * a
    # knocks
    lo, hi = st.level('knock_deg')
    cams = [o.tag for k, o in items if k == 'camera' or not st.knock_cameras_only]
    n_knock = int(rng.poisson(st.level('knock_rate_per_min') * st.duration_s / 60.0)) if cams else 0
    knocks = []
    for _ in range(n_knock):
        knocks.append({'t_s': float(rng.uniform(0, st.duration_s)), 'tag': str(rng.choice(cams)),
                       'deg': float(rng.uniform(lo, hi)), 'axis': [float(x) for x in _unit(rng)]})
    knocks.sort(key=lambda k: k['t_s'])
    out = []
    for kind, o in items:
        u_rot, u_tr = _unit(rng), _unit(rng)
        th_rot = np.outer(mount_T * st.level('thermal_deg_per_C'), u_rot)          # (n, 3) deg
        th_tr = np.outer(mount_T * st.level('thermal_mm_per_C'), u_tr)             # (n, 3) mm
        kn = np.zeros((n, 3))
        for k in knocks:
            if k['tag'] == o.tag:
                kn[t >= k['t_s']] += np.asarray(k['axis']) * k['deg']
        vib = np.stack([_band_noise(rng, n, st.rate_hz, st.vib_band_hz, st.level('vib_rms_deg')) for _ in range(3)],
                       axis=1)
        vtr = np.stack([_band_noise(rng, n, st.rate_hz, st.vib_band_hz, st.level('vib_rms_mm')) for _ in range(3)],
                       axis=1)
        rot = th_rot + kn + vib
        tr = th_tr + vtr
        R0, p0 = _R(o), np.asarray(o.pos(), float)
        Rs, P = np.zeros((n, 3, 3)), np.zeros((n, 3))
        d_ypr = np.zeros((n, 3))
        y0, pi0, r0 = euler_zyx(R0)
        for i in range(n):
            Rb, tb = LC.pose(0.0, pitch_b[i], roll_b[i])
            Rm = R0 @ rotvec(np.radians(rot[i]))
            Rs[i] = Rb @ Rm
            P[i] = Rb @ (p0 + R0 @ (tr[i] / 1000.0)) + tb
            y, pp, rr = euler_zyx(Rs[i])
            d_ypr[i] = (wrap180(y - y0), pp - pi0, wrap180(rr - r0))
        e = {'tag': o.tag, 'uid': o.uid, 'kind': kind if kind == 'camera' else o.kind,
             'R0': R0, 'p0': p0, 'R': Rs, 'p': P,
             'mount_rot_deg': rot, 'mount_tr_mm': tr,
             'parts': {'thermal_rot_deg': th_rot, 'thermal_tr_mm': th_tr, 'knock_rot_deg': kn, 'vib_rot_deg': vib,
                       'vib_tr_mm': vtr},
             'delta_ypr_deg': d_ypr, 'delta_xyz_mm': (P - p0) * 1000.0}
        if kind == 'camera':
            e['effects'] = camera_effects(project, o, Rs, P, st)
        out.append(e)
    body = {'pitch_deg': pitch_b, 'roll_deg': roll_b, **grad}
    return Schedule(st, t, ax, ay, body, out, knocks)


# ---------------------------------------------------------------- effects
def ground_error_m(C0, R0, C, R, d_m):
    """Ground-projection error (m) at d_m metres ahead of a camera along its aim (horizontally): the ground point P the
    planned camera (centre C0, body->vehicle rotation R0) sees there, and where the same pixel's ray meets the ground
    from the perturbed pose (C, R); inf where the turned ray no longer reaches the ground. C, R may be (n, 3), (n,3,3)."""
    C0 = np.asarray(C0, float)
    az = math.atan2(R0[1, 0], R0[0, 0])
    P = np.array([C0[0] + d_m * math.cos(az), C0[1] + d_m * math.sin(az), 0.0])
    b = R0.T @ (P - C0)                                        # the ray in the camera's body frame (the pixel)
    C = np.atleast_2d(np.asarray(C, float))
    R = np.asarray(R, float).reshape(-1, 3, 3)
    r = R @ b                                                  # (n, 3)
    with np.errstate(divide='ignore', invalid='ignore'):
        s = np.where(r[:, 2] < -1e-12, -C[:, 2] / r[:, 2], np.nan)
    Q = C + r * s[:, None]
    err = np.linalg.norm(Q - P, axis=1)
    return np.where(np.isfinite(err), err, np.inf)


def camera_effects(project, cam, R, C, st: StressSettings) -> dict:
    """Per frame: ground_err_cm {range: (n,)}, far_px (n,) (the pixel shift of a point far along the optical axis),
    bev_cells {range: (n,)}."""
    from .lens import resolve
    R0, C0 = _R(cam), np.asarray(cam.pos(), float)
    ge = {f'{d:g}': ground_error_m(C0, R0, C, R, d) * 100.0 for d in st.effect_ranges}
    intr = resolve(project.module_for(cam), project.lens_for(cam))
    W2O0 = BODY_TO_OPTICAL @ R0.T
    d_w = W2O0.T @ np.array([0.0, 0.0, 1.0])                    # the optical axis, vehicle frame
    opt = np.einsum('ij,njk,k->ni', BODY_TO_OPTICAL, np.transpose(R, (0, 2, 1)), d_w)   # in each perturbed optical frame
    u, v, _, ok = intr.project_dirs(opt)
    far = np.hypot(np.asarray(u) - intr.cx, np.asarray(v) - intr.cy)
    far = np.where(np.asarray(ok), far, np.inf)
    bev = {k: v_ / 100.0 / st.bev_cell_m for k, v_ in ge.items()}
    return {'ground_err_cm': ge, 'far_px': far, 'bev_cells': bev}


# ---------------------------------------------------------------- applying and exporting
def apply(project, sched: Schedule, t_s):
    """A copy of the project (mesh shared) with every perturbed camera and sensor of the active variant moved to its
    pose at time t_s (the nearest frame)."""
    i = sched.index(t_s)
    q = project.snapshot()
    v = q.variant
    by_uid = {c.uid: c for c in v.cameras}
    by_uid.update({s.uid: s for s in v.sensors})
    for e in sched.sensors:
        o = by_uid.get(e['uid'])
        if o is None:
            continue
        y, p, r = euler_zyx(e['R'][i])
        o.yaw, o.pitch, o.roll = float(y), float(p), float(r)
        o.x, o.y, o.z = (float(x) for x in e['p'][i])
        if getattr(o, 'locked', False):
            pass                                                 # a locked camera is the user's; the copy may move
    return q


def _finite_max(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return float(a.max()) if a.size else math.inf


def summary(sched: Schedule) -> dict:
    """Per sensor the worst and 95th-percentile figures, the knocks, the body's range."""
    rows = []
    for e in sched.sensors:
        row = {'tag': e['tag'], 'kind': e['kind'],
               'max_rot_deg': float(np.max(np.linalg.norm(e['mount_rot_deg'], axis=1))),
               'max_shift_mm': float(np.max(np.linalg.norm(e['delta_xyz_mm'], axis=1)))}
        if 'effects' in e:
            ef = e['effects']
            for k, a in ef['ground_err_cm'].items():
                fin = a[np.isfinite(a)]
                row[f'ground_err_cm_{k}m_p95'] = float(np.percentile(fin, 95)) if fin.size else math.inf
                row[f'ground_err_cm_{k}m_max'] = _finite_max(a)
            row['far_px_max'] = _finite_max(ef['far_px'])
        rows.append(row)
    return {'severity': sched.settings.severity, 'frames': len(sched.t), 'knocks': sched.knocks,
            'body_pitch_deg': [float(sched.body['pitch_deg'].min()), float(sched.body['pitch_deg'].max())],
            'body_roll_deg': [float(sched.body['roll_deg'].min()), float(sched.body['roll_deg'].max())],
            'sensors': rows}


def worst_frame(sched: Schedule) -> int:
    """The frame with the largest camera ground error at the middle effect range (else the largest rotation)."""
    best, bi = -1.0, 0
    mid = f'{sorted(sched.settings.effect_ranges)[len(sched.settings.effect_ranges) // 2]:g}'
    for e in sched.sensors:
        a = e.get('effects', {}).get('ground_err_cm', {}).get(mid)
        if a is None:
            a = np.linalg.norm(e['mount_rot_deg'], axis=1)
        a = np.where(np.isfinite(a), a, -1.0)
        k = int(np.argmax(a))
        if a[k] > best:
            best, bi = float(a[k]), k
    return bi


def _clean(x):
    """JSON-safe: non-finite numbers (a ray that no longer reaches the ground) become null."""
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def _num(x, nd=6):
    x = float(x)
    return round(x, nd) if math.isfinite(x) else None


def to_doc(sched: Schedule, project) -> dict:
    st = sched.settings
    sensors = []
    for e in sched.sensors:
        d = {'tag': e['tag'], 'uid': e['uid'], 'kind': e['kind'],
             'plan': {'xyz_m': [_num(x) for x in e['p0']], 'ypr_deg': [_num(x) for x in euler_zyx(e['R0'])]},
             'delta_ypr_deg': np.round(e['delta_ypr_deg'], 6).tolist(),
             'delta_xyz_mm': np.round(e['delta_xyz_mm'], 4).tolist(),
             'mount_rot_deg': np.round(e['mount_rot_deg'], 6).tolist(),
             'parts': {k: np.round(v, 6).tolist() for k, v in e['parts'].items()}}
        if 'effects' in e:
            ef = e['effects']
            d['effects'] = {'ground_err_cm': {k: [_num(x, 3) for x in v] for k, v in ef['ground_err_cm'].items()},
                            'far_px': [_num(x, 3) for x in ef['far_px']],
                            'bev_cells': {k: [_num(x, 3) for x in v] for k, v in ef['bev_cells'].items()}}
        sensors.append(d)
    return {'schema': SCHEMA, 'project': project.name, 'variant': project.variant.name,
            'frame': 'vehicle (ISO 8855, rear-axle centre on the ground); mount rotations about each sensor\'s own '
                     'axes (x forward, y left, z up)',
            'settings': st.to_dict(), 'severity_text': SEVERITY_TEXT[st.severity],
            'note': 'Severity levels are illustrative stress-test presets, not measurements of any product.',
            't_s': np.round(sched.t, 6).tolist(), 'long_g': np.round(sched.long_g, 5).tolist(),
            'lat_g': np.round(sched.lat_g, 5).tolist(),
            'body': {'pitch_deg': np.round(sched.body['pitch_deg'], 6).tolist(),
                     'roll_deg': np.round(sched.body['roll_deg'], 6).tolist(),
                     'pitch_per_g': sched.body['pitch_per_g'], 'roll_per_g': sched.body['roll_per_g'],
                     'note': sched.body['note']},
            'knocks': sched.knocks, 'sensors': sensors, 'summary': summary(sched)}


def to_csv(sched: Schedule) -> str:
    rng = [f'{d:g}' for d in sched.settings.effect_ranges]
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator='\n')
    w.writerow(['t_s', 'tag', 'kind', 'd_yaw_deg', 'd_pitch_deg', 'd_roll_deg', 'dx_mm', 'dy_mm', 'dz_mm',
                'body_pitch_deg', 'body_roll_deg'] + [f'ground_err_cm_{r}m' for r in rng] + ['far_px'] +
               [f'bev_cells_{r}m' for r in rng])
    for i, t in enumerate(sched.t):
        for e in sched.sensors:
            ef = e.get('effects')
            row = [f'{t:.3f}', e['tag'], e['kind'], *(f'{x:.6f}' for x in e['delta_ypr_deg'][i]),
                   *(f'{x:.4f}' for x in e['delta_xyz_mm'][i]), f'{sched.body["pitch_deg"][i]:.6f}',
                   f'{sched.body["roll_deg"][i]:.6f}']
            if ef:
                row += [f'{ef["ground_err_cm"][r][i]:.3f}' for r in rng] + [f'{ef["far_px"][i]:.3f}'] + \
                       [f'{ef["bev_cells"][r][i]:.3f}' for r in rng]
            else:
                row += [''] * (2 * len(rng) + 1)
            w.writerow(row)
    return buf.getvalue()


def report_md(sched: Schedule, project, sets=()) -> str:
    st, sm = sched.settings, summary(sched)
    mid = f'{sorted(st.effect_ranges)[len(st.effect_ranges) // 2]:g}'
    L = [f'# Calibration stress test · {project.name} / {project.variant.name}', '',
         f'Severity **{st.severity}**: {SEVERITY_TEXT[st.severity]}. Seed {st.seed}, {st.duration_s:g} s at '
         f'{st.rate_hz:g} Hz ({len(sched.t)} frames).', '',
         'The severity levels are illustrative stress-test presets, not measurements of any product. Body motion: '
         f'{sched.body["pitch_per_g"]:.3f}° pitch per g, {sched.body["roll_per_g"]:.3f}° roll per g '
         f'({sched.body["note"]}); the body pitches {sm["body_pitch_deg"][0]:+.3f}…{sm["body_pitch_deg"][1]:+.3f}° '
         f'and rolls {sm["body_roll_deg"][0]:+.3f}…{sm["body_roll_deg"][1]:+.3f}° over the drive.', '',
         f'## Knocks ({len(sched.knocks)})', '']
    L += [f'- {k["t_s"]:.1f} s · {k["tag"]} · {k["deg"]:.2f}°' for k in sched.knocks] or ['- none']
    L += ['', f'## Per sensor', '',
          f'| Sensor | Kind | Max mount rotation ° | Max shift mm | Ground error at {mid} m, p95 / max cm | '
          'Far-point shift max px |', '|---|---|---|---|---|---|']
    for r in sm['sensors']:
        ge = (f'{r[f"ground_err_cm_{mid}m_p95"]:.1f} / {r[f"ground_err_cm_{mid}m_max"]:.1f}'
              if f'ground_err_cm_{mid}m_p95' in r else '–')
        fp = f'{r["far_px_max"]:.1f}' if 'far_px_max' in r else '–'
        L.append(f'| {r["tag"]} | {r["kind"]} | {r["max_rot_deg"]:.3f} | {r["max_shift_mm"]:.2f} | {ge} | {fp} |')
    if sets:
        L += ['', '## Perturbed calibration sets', '']
        L += [f'- `{s["dir"]}`: t = {s["t_s"]:.1f} s{" (the worst moment)" if s.get("worst") else ""}' for s in sets]
    L += ['', 'Ground error: the pixel that saw a ground point at that range ahead now sees another one, this far away '
              '(inf: the ray no longer reaches the ground). Far-point shift: a point far along the optical axis, in '
              'pixels. BEV cells: the ground error over the BEV cell size in stress_schedule.json.', '']
    return '\n'.join(L)


def export_stress(project, engine=None, out_dir='.', settings=None, formats=DEFAULT_FORMATS) -> list[str]:
    """Write stress_schedule.json, stress_schedule.csv, stress_report.md and settings.sets perturbed calibration
    sets (each in the given export formats, through core.exporters) under out_dir; returns the paths written."""
    from . import exporters as EX
    sched = schedule(project, settings)
    st = sched.settings
    os.makedirs(out_dir, exist_ok=True)
    written = []
    sets = []
    if st.sets > 0 and formats:
        n = len(sched.t)
        idx = sorted(set([int(round(i)) for i in np.linspace(n - 1, 0, st.sets)][:max(st.sets - 1, 1)]
                         + [worst_frame(sched)]))
        wi = worst_frame(sched)
        for i in idx:
            t = float(sched.t[i])
            rel = f'calib/t{t:07.2f}s'.replace('.', '_')
            q = apply(project, sched, t)
            paths = EX.export(list(formats), q, None, EX.ExportOptions(out_dir=os.path.join(out_dir, rel)))
            written += paths
            sets.append({'dir': rel, 't_s': t, 'frame': i, 'worst': i == wi, 'formats': list(formats)})
    doc = _clean(to_doc(sched, project))
    doc['calibration_sets'] = sets
    for name, data in (('stress_schedule.json', json.dumps(doc, indent=1, allow_nan=False)),
                       ('stress_schedule.csv', to_csv(sched)), ('stress_report.md', report_md(sched, project, sets))):
        path = os.path.join(out_dir, name)
        with open(path, 'w', encoding='utf-8') as fh:
            fh.write(data)
        written.append(os.path.abspath(path))
    return written
