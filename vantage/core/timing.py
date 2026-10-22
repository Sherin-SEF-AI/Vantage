"""Sync and timing plan: when each camera should fire so that it sees what the spinning lidar sees, which cameras can
share a hardware trigger line, and how far the image moves during the exposure and the rolling-shutter readout.

The sweep. A spinning lidar is phase-locked to the sync (PTP): at t ms after each sync pulse its head faces the
sensor-frame azimuth phase − 360·f·t/1000 (clockwise seen from its own +z) or phase + 360·f·t/1000 (counter-
clockwise), f its rate in Hz and phase Sensor.phase_deg; azimuths as in core.sensors (counter-clockwise from the
sensor's +x). It measures a point when it faces the point's azimuth in its own frame, whatever the point's elevation.
A camera's view is swept when the lidar faces a point 20 m along the camera's optical axis, seen from the lidar's own
position (so parallax counts for a camera beside it); the sweep enters and leaves the image at the azimuths of the
image's horizontal edges (the middle row through the real lens model, inside the lens circle).

Triggers. A rolling-shutter camera reads its rows top to bottom over readout_ms after the trigger, each exposed for
exposure_ms before it is read, so the middle row's exposure is centred at trigger + readout/2 + exposure/2: the ideal
trigger puts that instant on the sweep crossing. Cameras with the same sync letter share a trigger line and fire
together; the line's trigger is the one that keeps the worst misalignment of its cameras smallest. A camera running at
n times the lidar rate (n an integer) fires n times per rotation at the same phases every rotation: its frame nearest
the sweep is the one matched to it, so its misalignment counts modulo its frame period. At a rate that is not an
integer multiple (or fraction) of the lidar's the frames slip against the sweep and no trigger keeps them aligned.

Motion. A static object object_m from the camera while the vehicle drives straight ahead at speed_kmh moves across
the image at v·sin β / d radians per second (β its bearing from the direction of travel); its image speed in pixels
is that motion projected through the camera's real lens model (exact, fisheye sampling included). Blur is the image
speed times the exposure, the rolling-shutter skew between the first and the last row the image speed times the
readout. Turning (yaw rate), pitching and the object's own motion are not modelled.

Times are in milliseconds after the sync pulse, modulo the reference lidar's period.
"""
from __future__ import annotations

import math
import time
import weakref
from dataclasses import asdict, dataclass

import numpy as np

from . import sensors as SN
from .analysis import camera_frames
from .lens import model_fov, resolve

SCHEMA = 'vantage.sync_plan/1'
AXIS_M = 20.0                    # a camera's view is located by a point this far along its optical axis
LETTERS = 'ABCDEFGH'             # trigger line letters, in the order the suggestion hands them out
TOL_MS = 1e-3                    # a grouping must improve the worst misalignment by more than this to use a line more
RATIO_TOL = 1e-4                 # a frame-rate ratio within this of an integer is an integer
LIMITS = {'speed_kmh': (0.0, 300.0), 'exposure_ms': (0.001, 100.0), 'readout_ms': (0.0, 100.0),
          'object_m': (0.2, 1000.0), 'blur_limit_px': (0.01, 100.0), 'trigger_lines': (1, len(LETTERS))}


@dataclass
class TimingSettings:
    speed_kmh: float = 50.0          # driving speed for the blur and for the travel during a misalignment
    exposure_ms: float = 2.0         # exposure time of each row
    readout_ms: float = 20.0         # rolling-shutter readout, first row to last (enter your sensor's; 0 = global)
    object_m: float = 5.0            # distance of the static object the blur is measured on
    blur_limit_px: float = 1.0       # the most motion blur the dataset accepts
    trigger_lines: int = 4           # hardware trigger outputs available
    lidar_uid: str = ''              # the reference lidar; '' = the first visible spinning lidar

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d: dict | None) -> 'TimingSettings':
        s = TimingSettings()
        d = d or {}
        for k, (lo, hi) in LIMITS.items():
            if k in d:
                try:
                    v = float(d[k])
                except (TypeError, ValueError):
                    continue
                if math.isfinite(v):
                    setattr(s, k, type(getattr(s, k))(round(min(max(v, lo), hi)) if k == 'trigger_lines'
                                                      else min(max(v, lo), hi)))
        if isinstance(d.get('lidar_uid'), str):
            s.lidar_uid = d['lidar_uid'][:40]
        return s

    @property
    def speed_mps(self) -> float:
        return self.speed_kmh / 3.6


@dataclass
class Sweep:
    """A spinning lidar's head over time: it faces sensor-frame azimuth phase + sense·360·rate·t."""
    uid: str
    tag: str
    name: str                        # the model's name
    model_id: str
    rate_hz: float
    sense: int                       # +1 counter-clockwise, -1 clockwise, seen from the lidar's own +z
    phase_deg: float
    pos: np.ndarray                  # optical centre, vehicle frame
    R: np.ndarray                    # sensor -> vehicle rotation

    @staticmethod
    def of(s, model) -> 'Sweep':
        ph = float(s.phase_deg or 0.0)
        return Sweep(s.uid, s.tag, model.name, model.id, float(model.rate_hz), 1 if model.rotation == 'ccw' else -1,
                     ph % 360.0 if math.isfinite(ph) else 0.0, s.pos(), SN.rotation(s))

    @property
    def period_ms(self) -> float:
        return 1000.0 / self.rate_hz

    @property
    def rotation(self) -> str:
        return 'ccw' if self.sense > 0 else 'cw'

    def time_of(self, az_deg, phase=None):
        """ms after the sync pulse at which the head faces sensor-frame azimuth az (in [0, period))."""
        ph = self.phase_deg if phase is None else phase
        return (self.sense * (np.asarray(az_deg, float) - ph)) % 360.0 / (0.36 * self.rate_hz)

    def facing(self, t_ms):
        """Sensor-frame azimuth the head faces t ms after the sync pulse, degrees."""
        return self.phase_deg + self.sense * 0.36 * self.rate_hz * np.asarray(t_ms, float)

    def direction(self, t_ms):
        """The vehicle-frame plan direction the head faces t ms after the sync pulse, degrees counter-clockwise from
        +x (forward)."""
        a = np.radians(self.facing(t_ms))
        d = np.stack([np.cos(a), np.sin(a), np.zeros_like(a)], -1) @ self.R.T
        return np.degrees(np.arctan2(d[..., 1], d[..., 0]))

    def sensor_az(self, dir_deg):
        """Sensor-frame azimuth of a vehicle-frame plan direction (degrees counter-clockwise from +x)."""
        a = np.radians(np.asarray(dir_deg, float))
        q = np.stack([np.cos(a), np.sin(a), np.zeros_like(a)], -1) @ self.R
        return np.degrees(np.arctan2(q[..., 1], q[..., 0]))

    @property
    def up(self) -> int:
        """+1 when the lidar's +z points up in the vehicle (its sense seen from above is its own), -1 upside down."""
        return 1 if self.R[2, 2] >= 0 else -1


@dataclass
class TimingResult:
    settings: TimingSettings
    lidar: Sweep | None              # the reference lidar, None without a spinning lidar
    reason: str                      # why there is no sweep plan ('' when there is one)
    cams: list                       # per visible camera: a dict (keys from plan, _sweep_of, _motion and _assign)
    groups: list                     # per trigger letter in use: a dict
    lidars: list                     # the other spinning lidars: phase advice
    suggestion: dict                 # suggested trigger letters: {'letters': {uid: letter}, 'worst_ms', 'lines'}
    notes: list                      # what needs attention, as sentences
    metrics: dict
    outline: np.ndarray | None = None    # the body seen from above (N, 2), vehicle frame, for the dial
    variant_uid: str = ''

    def cam(self, uid):
        return next((c for c in self.cams if c['uid'] == uid), None)

    @property
    def period_ms(self) -> float | None:
        return self.lidar.period_ms if self.lidar is not None else None


# =========================================================================== helpers
def _wrap(x, c):
    """x folded into [-c/2, c/2)."""
    return (np.asarray(x, float) + c / 2) % c - c / 2


def _best_trigger(ideal, cycle, T):
    """(trigger ms in [0, T), worst |misalignment| ms) of a common trigger for cameras whose ideal triggers are ideal
    (ms), each counted modulo its frame cycle. Exact: the minimum of the worst of these triangle waves lies on one of
    their zeros or half way between a zero of one and a zero of another."""
    ideal = np.asarray(ideal, float)
    cycle = np.asarray(cycle, float)
    a = np.concatenate([t + np.arange(-1, int(math.ceil(T / c)) + 2) * c for t, c in zip(ideal, cycle)])
    if len(a) <= 600:
        cand = np.concatenate([a, ((a[:, None] + a[None, :]) / 2).ravel()]) % T
    else:                            # an absurd frame-rate ratio: a fine grid instead (T / 36 000 resolution)
        cand = np.linspace(0.0, T, 36000, endpoint=False)
    err = np.abs(_wrap(cand[:, None] - ideal[None, :], cycle[None, :]))
    worst = err.max(1)
    i = int(np.argmin(np.round(worst, 9) + 1e-6 * err.mean(1)))
    return float(cand[i]), float(worst[i])


def _letter(c) -> str:
    """The camera's trigger line; '' when it runs free (older projects wrote '-')."""
    s = str(c.sync or '').strip().upper()
    return '' if s in ('-', 'FREE') else s


_OUTLINES = weakref.WeakKeyDictionary()


def _hull(P):
    """Convex hull of plan points (N, 2), counter-clockwise (monotone chain)."""
    P = sorted(set(map(tuple, np.round(P, 6))))
    if len(P) < 3:
        return np.array(P, float)

    def half(pts):
        h = []
        for q in pts:
            while len(h) >= 2 and ((h[-1][0] - h[-2][0]) * (q[1] - h[-2][1]) -
                                   (h[-1][1] - h[-2][1]) * (q[0] - h[-2][0])) <= 0:
                h.pop()
            h.append(q)
        return h[:-1]
    return np.array(half(P) + half(P[::-1]), float)


def plan_outline(mesh, bins: int = 144):
    """The body seen from above as a convex polygon (N, 2) in the vehicle frame (the hull of the farthest occluding
    vertex from the middle of the body in each of `bins` directions). None without a vehicle."""
    if mesh is None or not mesh.n_tris:
        return None
    if mesh in _OUTLINES:
        return _OUTLINES[mesh]
    faces = mesh.occluder_faces()
    V = mesh.tri[faces].reshape(-1, 3)[:, :2] if faces.any() else mesh.vertices[:, :2]
    c = (V.min(0) + V.max(0)) / 2
    d = V - c
    r = np.hypot(d[:, 0], d[:, 1])
    k = ((np.arctan2(d[:, 1], d[:, 0]) + math.pi) / (2 * math.pi) * bins).astype(int) % bins
    far = np.zeros(bins)
    np.maximum.at(far, k, r)
    if (far == 0).any() and (far > 0).any():              # an empty direction takes its neighbours' reach
        i = np.arange(bins)
        ok = far > 0
        far = np.interp(i, np.r_[i[ok] - bins, i[ok], i[ok] + bins], np.r_[far[ok], far[ok], far[ok]])
    t = -math.pi + (np.arange(bins) + 0.5) * 2 * math.pi / bins
    out = _hull(c + far[:, None] * np.c_[np.cos(t), np.sin(t)])
    _OUTLINES[mesh] = out
    return out


# =========================================================================== per camera
def _field_line(cam, intr):
    """Vehicle-frame unit directions (K, 3) across the image's horizontal axis through the principal point, inside the
    lens circle, and the index of the principal point. A camera rolled past 45° uses the image's middle column, which
    is then the more horizontal one."""
    _, W2O = camera_frames(cam)
    row = abs(W2O[0, 2]) <= abs(W2O[1, 2])                # image x (optical x) more level than image y
    n = 48
    if row:
        u = np.r_[np.linspace(0.0, intr.cx, n + 1), np.linspace(intr.cx, intr.width, n + 1)[1:]]
        v = np.full_like(u, intr.cy)
    else:
        v = np.r_[np.linspace(0.0, intr.cy, n + 1), np.linspace(intr.cy, intr.height, n + 1)[1:]]
        u = np.full_like(v, intr.cx)
    d, th = intr.unproject(u, v)
    keep = th <= intr.max_theta + 1e-9
    keep[n] = True
    return (d @ W2O)[keep], int(np.count_nonzero(keep[:n]))


def _sweep_of(cam, intr, lid: Sweep, s):
    """Where the lidar sees the camera's view: the azimuth of its axis point and of its horizontal field edges in the
    lidar's frame, and when the sweep enters the image, crosses the axis point and leaves it."""
    D, i0 = _field_line(cam, intr)
    P = cam.pos() + AXIS_M * D
    _, az, _ = SN.angles(s, P)
    off = np.degrees(np.unwrap(np.radians(az)))
    off = off - off[i0]
    centre = float(az[i0])
    left, right = float(off.max()), float(off.min())
    span = left - right
    if span > 360.0:                                      # parallax beside the lidar: never more than all round
        left, right, span = 180.0, -180.0, 360.0
    cross = float(lid.time_of(centre))
    enter = float(lid.time_of(centre + (left if lid.sense < 0 else right)))
    return {'azimuth_deg': (centre + 180.0) % 360.0 - 180.0, 'left_deg': (centre + left + 180.0) % 360.0 - 180.0,
            'right_deg': (centre + right + 180.0) % 360.0 - 180.0, 'field_deg': span, 'enter_ms': enter,
            'sweep_ms': span / 360.0 * lid.period_ms, 'cross_ms': cross}


def _motion(cam, intr, st: TimingSettings, grid=(41, 33)):
    """Image speed (px/s) of a static object st.object_m from the camera while the vehicle drives forward, over the
    image inside the lens circle and at the principal point, and what it means for blur and rolling-shutter skew."""
    _, W2O = camera_frames(cam)
    gu, gv = grid
    U, V = np.meshgrid(np.linspace(0.5, intr.width - 0.5, gu), np.linspace(0.5, intr.height - 0.5, gv))
    u = np.r_[intr.cx, U.ravel()]
    v = np.r_[intr.cy, V.ravel()]
    d, th = intr.unproject(u, v)
    keep = th <= intr.max_theta + 1e-9
    keep[0] = True
    P = st.object_m * (d[keep] @ W2O)                     # the object, relative to the camera, vehicle frame
    v_ = st.speed_mps
    h = 1e-3                                              # s: central difference of the projected motion
    step = np.array([v_ * h / 2, 0.0, 0.0])
    ua, va, _, _ = intr.project_dirs((P - step) @ W2O.T)  # the vehicle moves +x: the object moves -x relative to it
    ub, vb, _, _ = intr.project_dirs((P + step) @ W2O.T)
    rate = np.hypot(ua - ub, va - vb) / h
    centre, worst = float(rate[0]), float(rate.max())
    exp_s, ro_s = st.exposure_ms / 1000.0, st.readout_ms / 1000.0
    return {'rate_centre_px_s': centre, 'rate_worst_px_s': worst,
            'blur_centre_px': centre * exp_s, 'blur_worst_px': worst * exp_s,
            'max_exposure_ms': st.blur_limit_px / worst * 1000.0 if worst > 1e-9 else None,
            'skew_centre_px': centre * ro_s, 'skew_px': worst * ro_s}


def _rate_fit(fps, rate_hz):
    """(ratio, cycle as a fraction of the lidar period or None when the frames drift, alignment word)."""
    ratio = fps / rate_hz if rate_hz > 0 else 0.0
    n = round(ratio)
    if n >= 1 and abs(ratio - n) <= RATIO_TOL * max(1.0, ratio):
        return ratio, 1.0 / n, 'integer'
    if 0 < ratio < 1:
        m = round(1.0 / ratio)
        if abs(1.0 / ratio - m) <= RATIO_TOL * m:
            return ratio, 1.0, 'fraction'
    return ratio, None, 'drifts'


# =========================================================================== groups
def _assign(cams, letters: dict, T: float, speed_mps: float):
    """The cameras grouped by trigger letter: per group the common trigger and its worst misalignment; per camera the
    trigger pulse it is matched to, its signed misalignment (positive: the exposure is centred after the sweep) and
    the vehicle's travel during it."""
    by = {}
    for c in cams:
        L = letters.get(c['uid'], '')
        if L and c['triggered'] and c.get('ideal_ms') is not None:
            by.setdefault(L, []).append(c)
    groups, per = [], {}
    for L in sorted(by):
        m = by[L]
        ideal = np.array([c['ideal_ms'] for c in m])
        cyc = np.array([c['cycle_ms'] for c in m])
        tau, worst = _best_trigger(ideal, cyc, T)
        for c, t0, cy in zip(m, ideal, cyc):
            d = float(_wrap(tau - t0, cy))
            per[c['uid']] = {'trigger_ms': float((t0 + d) % T), 'misalign_ms': d,
                             'travel_cm': abs(d) / 1000.0 * speed_mps * 100.0}
        period = float(cyc.min())
        fps = sorted({c['fps'] for c in m})
        groups.append({'letter': L, 'uids': [c['uid'] for c in m], 'tags': [c['tag'] for c in m],
                       'trigger_ms': tau % period, 'period_ms': period, 'worst_ms': worst,
                       'travel_cm': worst / 1000.0 * speed_mps * 100.0, 'fps': fps, 'mixed_rates': len(fps) > 1})
    worst = max((g['worst_ms'] for g in groups), default=None)
    return groups, per, worst


def evaluate_groups(res: TimingResult, letters: dict):
    """(groups, worst misalignment ms) if the cameras used these trigger letters (uid -> letter; cameras left out keep
    their own)."""
    if res.lidar is None:
        return [], None
    use = {c['uid']: c['sync'] for c in res.cams}
    use.update({u: str(L).strip().upper() for u, L in letters.items()})
    groups, _, worst = _assign(res.cams, use, res.lidar.period_ms, res.settings.speed_mps)
    return groups, worst


def suggest_groups(res: TimingResult, lines: int | None = None) -> dict:
    """Trigger letters (uid -> 'A', 'B' …) for the triggered cameras that keep the worst misalignment smallest with
    at most `lines` trigger lines (the settings' by default), using no more lines than that needs. The groups are runs
    of cameras next to each other in the circular order of their ideal triggers (an exact search over every such
    partition); letters follow the groups' trigger times."""
    if res.lidar is None:
        return {}
    cams = [c for c in res.cams if c['triggered'] and c.get('ideal_ms') is not None]
    n = len(cams)
    if not n:
        return {}
    T = res.lidar.period_ms
    L = max(1, min(int(lines or res.settings.trigger_lines), n, len(LETTERS)))
    ideal = np.array([c['ideal_ms'] for c in cams])
    cyc = np.array([c['cycle_ms'] for c in cams])
    key = ideal % cyc[0] if np.allclose(cyc, cyc[0]) else ideal % T
    order = [int(i) for i in np.argsort(key, kind='stable')]
    memo = {}

    def cost(a, length):                                  # worst misalignment of the run of `length` from position a
        if (a, length) not in memo:
            idx = [order[(a + t) % n] for t in range(length)]
            memo[a, length] = _best_trigger(ideal[idx], cyc[idx], T)[1]
        return memo[a, length]
    best = None
    for s in range(n if L > 1 else 1):                    # where the first run starts in the circular order
        dp = [[None] * (n + 1) for _ in range(L + 1)]     # dp[k][j]: (worst, sum, i) covering j cameras with k runs
        dp[0][0] = (0.0, 0.0, -1)
        for k in range(1, L + 1):
            for j in range(k, n + 1):
                for i in range(k - 1, j):
                    if dp[k - 1][i] is None:
                        continue
                    c = cost((s + i) % n, j - i)
                    cand = (max(dp[k - 1][i][0], c), dp[k - 1][i][1] + c, i)
                    if dp[k][j] is None or (round(cand[0] / TOL_MS), cand[1]) < (round(dp[k][j][0] / TOL_MS),
                                                                                   dp[k][j][1]):
                        dp[k][j] = cand
        for k in range(1, L + 1):
            if dp[k][n] is not None:
                rank = (round(dp[k][n][0] / TOL_MS), k, round(dp[k][n][1], 6))
                if best is None or rank < best[0]:
                    best = (rank, s, k, dp)
    _, s, k, dp = best
    runs, j = [], n
    while k > 0:
        i = dp[k][j][2]
        runs.append([order[(s + t) % n] for t in range(i, j)])
        j, k = i, k - 1
    timed = []
    for idx in runs:
        tau, _ = _best_trigger(ideal[idx], cyc[idx], T)
        timed.append((tau % float(cyc[idx].min()), idx))
    timed.sort(key=lambda r: r[0])
    return {cams[i]['uid']: LETTERS[g] for g, (_, idx) in enumerate(timed) for i in idx}


# =========================================================================== other lidars
def _lidar_advice(ref: Sweep, s, model) -> dict:
    """The phase that makes another spinning lidar face the same vehicle-frame direction as the reference at the same
    time (when the reference faces forward, for a lidar turning the other way or at another rate), and how far apart
    in time the two sweep the same direction now and with that phase."""
    o = Sweep.of(s, model)
    t_f = float(ref.time_of(ref.sensor_az(0.0)))          # the reference faces forward
    phase = float((o.sensor_az(0.0) - o.sense * 0.36 * o.rate_hz * t_f) % 360.0)
    same_sense = ref.sense * ref.up == o.sense * o.up
    same_rate = abs(o.rate_hz - ref.rate_hz) <= 1e-9 * ref.rate_hz
    dirs = np.arange(0.0, 360.0, 1.0)
    t_r = ref.time_of(ref.sensor_az(dirs))
    T = ref.period_ms

    def apart(ph):
        dt = _wrap(o.time_of(o.sensor_az(dirs), ph) - t_r, T)
        return float(np.abs(dt).max()), float(dt[0])
    now, lag = apart(o.phase_deg)
    after, _ = apart(phase)
    note = ''
    if not same_rate:
        note = (f'{o.tag} turns at {o.rate_hz:g} Hz against {ref.tag}\'s {ref.rate_hz:g} Hz: the two sweeps drift '
                f'apart whatever the phase')
    elif not same_sense:
        note = (f'{o.tag} turns the other way round from {ref.tag} (seen from above): the sweeps meet ahead and behind '
                f'and are half a turn apart at the sides')
    return {'uid': s.uid, 'tag': s.tag, 'name': model.name, 'rate_hz': o.rate_hz, 'rotation': o.rotation,
            'phase_deg': o.phase_deg, 'suggested_phase_deg': phase, 'misalign_ms': now, 'after_ms': after,
            'lag_ms': lag, 'same_sense': same_sense, 'same_rate': same_rate, 'note': note}


def suggested_phases(res: TimingResult) -> dict:
    """uid -> phase_deg for the other spinning lidars."""
    return {d['uid']: round(d['suggested_phase_deg'], 2) % 360.0 for d in res.lidars}


# =========================================================================== the plan
def plan(project, variant=None, settings: TimingSettings | None = None) -> TimingResult:
    """The sync and timing plan of a variant (the active one by default; a Variant or its uid)."""
    t0 = time.perf_counter()
    st = settings or TimingSettings.from_dict(getattr(project.analysis, 'timing', None))
    var = project.variant_by_uid(variant) if isinstance(variant, str) else variant
    var = var or project.variant
    notes = []
    spinning = [s for s in var.sensors if s.visible and s.kind == 'lidar' and project.sensor_model_for(s).spinning
                and project.sensor_model_for(s).rate_hz > 0]
    ref = next((s for s in spinning if s.uid == st.lidar_uid), None)
    if ref is None and spinning:
        ref = spinning[0]
        if st.lidar_uid:
            notes.append(f'The chosen reference lidar is not a visible spinning lidar any more; {ref.tag} is used.')
    lid = Sweep.of(ref, project.sensor_model_for(ref)) if ref is not None else None
    reason = ''
    if lid is None:
        others = [s for s in var.sensors if s.visible and s.kind == 'lidar']
        reason = ('The lidars of this variant do not spin (a fixed-field or non-repetitive scanner has no sweep to '
                  'follow): the cameras need no phase plan for them.' if others else
                  'No spinning lidar in this variant: add one for the sweep plan. The motion figures need none.')
    T = lid.period_ms if lid is not None else None
    cams = []
    for c in (c for c in var.cameras if c.visible):
        mod, lens = project.module_for(c), project.lens_for(c)
        intr = resolve(mod, lens)
        hfov = min(model_fov(mod, lens)[0], 2 * math.degrees(intr.max_theta))
        e = {'uid': c.uid, 'tag': c.tag, 'name': c.name, 'slot': c.slot, 'pos': tuple(float(v) for v in c.pos()),
             'sync': _letter(c), 'triggered': bool(_letter(c)) and bool(mod.hw_sync), 'hw_sync': bool(mod.hw_sync),
             'fps': float(mod.fps), 'hfov_deg': hfov, 'frame_ms': 1000.0 / mod.fps if mod.fps > 0 else None}
        e.update(_motion(c, intr, st))
        if lid is not None:
            ratio, frac, how = _rate_fit(e['fps'], lid.rate_hz)
            e.update(_sweep_of(c, intr, lid, ref))
            e['ideal_ms'] = float((e['cross_ms'] - st.readout_ms / 2 - st.exposure_ms / 2) % T)
            slip = abs(e['fps'] - round(ratio) * lid.rate_hz)        # Hz: the frames' beat against the rotation
            e.update({'ratio': ratio, 'alignment': how, 'cycle_ms': T * (frac or 1.0),
                      'beat_s': 1.0 / slip if how == 'drifts' and round(ratio) >= 1 and slip > 1e-9 else None})
        cams.append(e)
    # ---- trigger groups
    groups, worst, suggestion = [], None, {'letters': {}, 'worst_ms': None, 'lines': 0}
    if lid is not None:
        groups, per, worst = _assign(cams, {c['uid']: c['sync'] for c in cams}, T, st.speed_mps)
        for e in cams:
            e.update(per.get(e['uid'], {'trigger_ms': None, 'misalign_ms': None, 'travel_cm': None}))
    res = TimingResult(st, lid, reason, cams, groups, [], suggestion, notes, {},
                       plan_outline(project.vehicle.mesh), var.uid)
    if lid is not None:
        letters = suggest_groups(res)
        _, after = evaluate_groups(res, letters)
        res.suggestion = {'letters': letters, 'worst_ms': after, 'lines': len(set(letters.values()))}
        res.lidars = [_lidar_advice(lid, s, project.sensor_model_for(s)) for s in spinning if s is not ref]
    # ---- what needs attention
    for e in cams:
        if lid is not None and e['alignment'] == 'drifts':
            notes.append(f'{e["tag"]}: {e["fps"]:g} fps is {e["ratio"]:.3f}× the lidar\'s {lid.rate_hz:g} Hz, not an '
                         f'integer multiple: its frames slip against the sweep'
                         + (f' (back in step every {e["beat_s"]:.1f} s)' if e['beat_s'] else '') + '.')
        if not e['hw_sync']:
            notes.append(f'{e["tag"]}: the module has no hardware trigger input; it free-runs and cannot be '
                         f'phase-locked to the sweep.')
        elif lid is not None and not e['sync']:
            notes.append(f'{e["tag"]}: on no trigger line (empty sync group); it free-runs.')
        if e['frame_ms'] and st.readout_ms + st.exposure_ms > e['frame_ms'] + 1e-9:
            notes.append(f'{e["tag"]}: readout and exposure ({st.readout_ms + st.exposure_ms:g} ms) take longer than '
                         f'a frame at {e["fps"]:g} fps ({e["frame_ms"]:.1f} ms).')
        if e['blur_worst_px'] > st.blur_limit_px:
            notes.append(f'{e["tag"]}: up to {e["blur_worst_px"]:.1f} px of motion blur at {st.speed_kmh:g} km/h on an '
                         f'object {st.object_m:g} m away (limit {st.blur_limit_px:g} px): expose at most '
                         f'{e["max_exposure_ms"]:.2f} ms.')
    for g in groups:
        if g['mixed_rates']:
            notes.append(f'Trigger line {g["letter"]} drives cameras at different frame rates '
                         f'({", ".join(f"{f:g}" for f in g["fps"])} fps); one line triggers one rate.')
    if len(groups) > st.trigger_lines:
        notes.append(f'{len(groups)} trigger groups in use but {st.trigger_lines} trigger lines available.')
    for d in res.lidars:
        if d['note']:
            notes.append(d['note'] + '.')
    worst_blur = max(cams, key=lambda e: e['blur_worst_px'], default=None)
    exps = [e['max_exposure_ms'] for e in cams if e['max_exposure_ms'] is not None]
    res.metrics = {
        'cameras': len(cams), 'triggered': sum(e['triggered'] for e in cams),
        'worst_ms': worst, 'worst_travel_cm': worst / 1000.0 * st.speed_mps * 100.0 if worst is not None else None,
        'worst_tag': (max((e for e in cams if e.get('misalign_ms') is not None),
                          key=lambda e: abs(e['misalign_ms']), default={}).get('tag')),
        'lines_used': len(groups), 'period_ms': T,
        'rotation_travel_m': st.speed_mps * T / 1000.0 if T else None,
        'worst_blur_px': worst_blur['blur_worst_px'] if worst_blur else 0.0,
        'worst_blur_tag': worst_blur['tag'] if worst_blur else None,
        'max_exposure_ms': min(exps) if exps else None,
        'drifting': [e['tag'] for e in cams if lid is not None and e['alignment'] == 'drifts'],
        'seconds': time.perf_counter() - t0}
    return res


# =========================================================================== output
def _r(v, dp=3):
    return None if v is None or not math.isfinite(float(v)) else round(float(v), dp) + 0.0


def summary(res: TimingResult) -> dict:
    """Compact figures (the assistant, the batch report header)."""
    st, lid, m = res.settings, res.lidar, res.metrics
    tags = {c['uid']: c['tag'] for c in res.cams}
    out = {
        'lidar': None if lid is None else {'tag': lid.tag, 'model': lid.name, 'rate_hz': lid.rate_hz,
                                           'rotation': lid.rotation, 'phase_deg': round(lid.phase_deg, 2),
                                           'period_ms': _r(lid.period_ms)},
        'speed_kmh': st.speed_kmh, 'exposure_ms': st.exposure_ms, 'readout_ms': st.readout_ms,
        'worst_misalignment_ms': _r(m['worst_ms'], 2), 'worst_travel_cm': _r(m['worst_travel_cm'], 1),
        'trigger_lines_used': m['lines_used'], 'trigger_lines_available': st.trigger_lines,
        'travel_per_rotation_m': _r(m['rotation_travel_m'], 2),
        'cameras': [{'tag': c['tag'], 'group': c['sync'] or None, 'fps_ratio': _r(c.get('ratio'), 3),
                     'crossing_ms': _r(c.get('cross_ms'), 2), 'trigger_ms': _r(c.get('trigger_ms'), 2),
                     'misalignment_ms': _r(c.get('misalign_ms'), 2), 'travel_cm': _r(c.get('travel_cm'), 1),
                     'blur_worst_px': _r(c['blur_worst_px'], 2), 'blur_centre_px': _r(c['blur_centre_px'], 2),
                     'max_exposure_ms': _r(c['max_exposure_ms'], 2), 'skew_px': _r(c['skew_px'], 1)}
                    for c in res.cams],
        'notes': list(res.notes),
    }
    if res.reason:
        out['no_sweep_plan'] = res.reason
    if lid is not None:
        out['suggested_groups'] = {tags[u]: L for u, L in res.suggestion['letters'].items()}
        out['suggested_worst_misalignment_ms'] = _r(res.suggestion['worst_ms'], 2)
        out['other_lidars'] = [{'tag': d['tag'], 'phase_deg': round(d['phase_deg'], 2),
                                'suggested_phase_deg': round(d['suggested_phase_deg'], 2),
                                'misalignment_ms': _r(d['misalign_ms'], 2), 'after_ms': _r(d['after_ms'], 2)}
                               for d in res.lidars]
    return out


def to_doc(res: TimingResult, project=None, L=None, A=None, origin=(0.0, 0.0, 0.0)) -> dict:
    """The JSON export. L converts lengths (metres) to the export units, A angles (degrees); origin is the export
    frame's origin in the vehicle frame. Times stay in ms, image figures in px, the speed in km/h."""
    L = L or (lambda v: round(float(v), 4))
    A = A or (lambda v: round(float(v), 4))
    o = np.zeros(3)
    o[:len(origin)] = np.asarray(origin, float)[:3]
    ln = lambda v: None if v is None or not math.isfinite(float(v)) else L(v)
    an = lambda v: None if v is None else A(v)
    st, lid = res.settings, res.lidar
    var = project.variant_by_uid(res.variant_uid) if project is not None else None
    doc = {
        'schema': SCHEMA,
        'project': getattr(project, 'name', ''), 'variant': var.name if var is not None else '',
        'conventions': {
            'time': 'ms after the sync pulse (the top of each lidar period), modulo the reference lidar\'s period',
            'azimuth': 'in the reference lidar\'s own frame, counter-clockwise from its +x; a camera\'s view is the '
                       f'point {AXIS_M:g} m along its optical axis, its field the image\'s horizontal edges',
            'rolling_shutter': 'rows are read top to bottom over readout_ms; the middle row\'s exposure is centred '
                               'at trigger + readout/2 + exposure/2',
            'misalignment': 'trigger pulse minus ideal trigger, modulo the camera\'s frame period when its frame '
                            'rate is an integer multiple of the lidar\'s (positive: the exposure comes after the '
                            'sweep)',
            'motion': 'a static object at object distance from the camera while the vehicle drives straight ahead, '
                      'projected through the real lens model, over the image inside the lens circle'},
        'settings': {'speed_kmh': st.speed_kmh, 'exposure_ms': st.exposure_ms, 'readout_ms': st.readout_ms,
                     'object': L(st.object_m), 'blur_limit_px': st.blur_limit_px,
                     'trigger_lines': st.trigger_lines},
        'lidar': None if lid is None else {
            'tag': lid.tag, 'model': lid.name, 'rate_hz': lid.rate_hz, 'period_ms': _r(lid.period_ms, 6),
            'rotation': lid.rotation, 'phase': A(lid.phase_deg),
            'position': [L(lid.pos[i] - o[i]) for i in range(3)],
            'travel_per_rotation': ln(res.metrics['rotation_travel_m'])},
        'no_sweep_plan': res.reason or None,
        'cameras': [{
            'tag': c['tag'], 'name': c['name'], 'group': c['sync'] or None, 'triggered': c['triggered'],
            'fps': c['fps'], 'fps_ratio': _r(c.get('ratio'), 6), 'alignment': c.get('alignment'),
            'position': [L(c['pos'][i] - o[i]) for i in range(3)],
            'azimuth': an(c.get('azimuth_deg')), 'field': an(c.get('field_deg')),
            'field_left': an(c.get('left_deg')), 'field_right': an(c.get('right_deg')),
            'sweep_enters_ms': _r(c.get('enter_ms')), 'sweep_ms': _r(c.get('sweep_ms')),
            'crossing_ms': _r(c.get('cross_ms')), 'ideal_trigger_ms': _r(c.get('ideal_ms')),
            'trigger_ms': _r(c.get('trigger_ms')), 'misalignment_ms': _r(c.get('misalign_ms')),
            'travel': ln(c['travel_cm'] / 100.0 if c.get('travel_cm') is not None else None),
            'motion': {'blur_centre_px': _r(c['blur_centre_px']), 'blur_worst_px': _r(c['blur_worst_px']),
                       'max_exposure_ms': _r(c['max_exposure_ms']), 'skew_centre_px': _r(c['skew_centre_px']),
                       'skew_px': _r(c['skew_px'])}}
            for c in res.cams],
        'groups': [{'letter': g['letter'], 'cameras': g['tags'], 'trigger_ms': _r(g['trigger_ms']),
                    'period_ms': _r(g['period_ms']), 'worst_misalignment_ms': _r(g['worst_ms']),
                    'travel': ln(g['travel_cm'] / 100.0)} for g in res.groups],
        'suggested_groups': None if lid is None else {
            'groups': {res.cam(u)['tag']: L_ for u, L_ in res.suggestion['letters'].items()},
            'lines': res.suggestion['lines'], 'worst_misalignment_ms': _r(res.suggestion['worst_ms'])},
        'other_lidars': [{'tag': d['tag'], 'model': d['name'], 'rate_hz': d['rate_hz'], 'rotation': d['rotation'],
                          'phase': A(d['phase_deg']), 'suggested_phase': A(d['suggested_phase_deg']),
                          'misalignment_ms': _r(d['misalign_ms']), 'after_ms': _r(d['after_ms']),
                          'note': d['note'] or None} for d in res.lidars],
        'metrics': {'worst_misalignment_ms': _r(res.metrics['worst_ms']),
                    'worst_travel': ln(res.metrics['worst_travel_cm'] / 100.0
                                       if res.metrics['worst_travel_cm'] is not None else None),
                    'trigger_lines_used': res.metrics['lines_used'],
                    'worst_blur_px': _r(res.metrics['worst_blur_px']),
                    'max_exposure_ms': _r(res.metrics['max_exposure_ms'])},
        'notes': list(res.notes),
    }
    return doc


def export(res: TimingResult, project, out_dir: str) -> list[str]:
    """sync_plan.json (in the project's frame and units, angles in degrees) and sync_plan.png into out_dir."""
    import datetime as _dt
    import json
    import os
    os.makedirs(out_dir, exist_ok=True)
    mm = getattr(project, 'units', 'm') == 'mm'
    L = (lambda v: round(float(v) * 1000.0, 1) + 0.0) if mm else (lambda v: round(float(v), 4) + 0.0)
    doc = to_doc(res, project, L=L, A=lambda v: round(float(v), 4) + 0.0, origin=project.origin_offset())
    head = {'project': project.name, 'variant': doc.pop('variant'),
            'created': _dt.datetime.now().astimezone().replace(microsecond=0).isoformat(),
            'frame': project.frame, 'units': 'mm' if mm else 'm', 'angles': 'deg'}
    doc = {'schema': doc.pop('schema'), **head, **{k: v for k, v in doc.items() if k != 'project'}}
    paths = [os.path.join(out_dir, 'sync_plan.json'), os.path.join(out_dir, 'sync_plan.png')]
    with open(paths[0], 'w', encoding='utf-8') as fh:
        fh.write(json.dumps(doc, indent=1, ensure_ascii=False) + '\n')
    with open(paths[1], 'wb') as fh:
        fh.write(png(res))
    return paths


# =========================================================================== the dial
# camera colours as the app's dark theme (ui.theme.CAMERA['dark']), here for the PNG without Qt
CAM_COLORS = ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#008300', '#9085e9', '#e66767']


def _bgr(hex_):
    return int(hex_[5:7], 16), int(hex_[3:5], 16), int(hex_[1:3], 16)


def tick_step(period_ms) -> float:
    """ms between the dial's ticks: about 20 to 40 round the dial."""
    for s in (0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500):
        if period_ms / s <= 40:
            return float(s)
    return 1000.0


def dial_layout(res: TimingResult, R: float) -> dict:
    """Radii of the dial for an outer radius R: the rim (ticks outside it, the trigger letters in a ring inside it),
    one track per camera inside that, and the hub with the vehicle outline (its scale in px per metre about the
    lidar)."""
    n = max(len(res.cams), 1)
    rim = R - 14
    hub = R * 0.42
    th = min(22.0, (rim - 18 - hub) / n)
    tracks = [rim - 18 - (i + 0.5) * th for i in range(len(res.cams))]
    inner = (tracks[-1] - th / 2 - 6) if tracks else hub
    scale = None
    if res.lidar is not None:
        pts = [np.asarray(res.lidar.pos[:2])]
        if res.outline is not None:
            pts.append(res.outline)
        pts += [np.array([c['pos'][:2]]) for c in res.cams]
        P = np.vstack(pts) - res.lidar.pos[:2]
        far = float(np.hypot(P[:, 0], P[:, 1]).max()) if len(P) else 1.0
        scale = (inner - 8) / max(far, 0.5)
    return {'rim': rim, 'tracks': tracks, 'track_w': th, 'inner': inner, 'scale': scale}


def png(res: TimingResult, size=900) -> bytes:
    """The dial as PNG: one lidar rotation round the circle (forward up, left to the left, the direction the reference
    lidar faces at each time), ms ticks from the sync pulse, the sweep direction; one track per camera with its field
    as the sweep crosses it, the crossing (a bar), the matched frame's exposure from the first row to the last (a thin
    arc) and its middle row's exposure centre (a dot); trigger marks with their line letters on the rim; the vehicle
    and the cameras in the middle."""
    import cv2
    head = 64
    img = np.full((size + head, size, 3), (30, 29, 28), np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    aa = cv2.LINE_AA
    ink, ink2, muted, grid = (236, 235, 230), (183, 194, 195), (129, 135, 137), (62, 61, 58)

    def text(t, x, y, col, s=0.45, th=1):
        cv2.putText(img, t, (int(x), int(y)), font, s, col, th, aa)
    lid, st, m = res.lidar, res.settings, res.metrics
    if lid is None:
        text('Sync and timing plan', 16, 26, ink, 0.6)
        text(res.reason.split(':')[0][:110], 16, 50, muted, 0.45)
        y = head + 30
        text(f'Motion at {st.speed_kmh:g} km/h, object {st.object_m:g} m away, exposure {st.exposure_ms:g} ms, '
             f'readout {st.readout_ms:g} ms', 16, y, ink2, 0.5)
        y += 34
        for c in res.cams:
            col = _bgr(CAM_COLORS[c['slot'] % 8])
            cv2.rectangle(img, (16, y - 11), (28, y + 1), col, -1)
            mx = c['max_exposure_ms']
            text(f'{c["tag"]:<4} blur {c["blur_worst_px"]:.2f} px worst, {c["blur_centre_px"]:.2f} px centre;  '
                 f'skew {c["skew_px"]:.1f} px;  max exposure ' + (f'{mx:.2f} ms' if mx is not None else '-'),
                 36, y, ink, 0.5)
            y += 28
        ok, buf = cv2.imencode('.png', img)
        return buf.tobytes()
    cx, cy = size / 2, head + size / 2
    R = size / 2 - 40
    lay = dial_layout(res, R)
    T = lid.period_ms

    def at(t, r):
        a = math.radians(float(lid.direction(t)))
        return cx - math.sin(a) * r, cy - math.cos(a) * r

    def arc(t0, t1, r):
        n = max(int(abs(t1 - t0) / T * 360) + 2, 2)
        return np.round(np.array([at(t, r) for t in np.linspace(t0, t1, n)]) * 16).astype(np.int32)

    def pt(x, y):
        return int(round(x * 16)), int(round(y * 16))
    # rim: the rotation, ms ticks, the sync pulse and the sweep direction
    cv2.circle(img, pt(cx, cy), int(lay['rim'] * 16), grid, 1, aa, 4)
    step = tick_step(T)
    k = 0
    while k * step < T - 1e-9:
        t = k * step
        major = k % 2 == 0
        x0, y0 = at(t, lay['rim'])
        x1, y1 = at(t, lay['rim'] + (9 if major else 5))
        cv2.line(img, pt(x0, y0), pt(x1, y1), ink if k == 0 else muted, 2 if k == 0 else 1, aa, 4)
        if major:
            lbl = f'{t:g}' + (' ms' if k == 0 else '')
            (w, h), _ = cv2.getTextSize(lbl, font, 0.4, 1)
            x, y = at(t, lay['rim'] + 22)
            text(lbl, x - w / 2, y + h / 2, ink if k == 0 else muted, 0.4)
        k += 1
    a0 = arc(0.04 * T, 0.12 * T, R + 20)
    cv2.polylines(img, [a0], False, ink2, 1, aa, 4)
    cv2.arrowedLine(img, tuple(int(v) for v in a0[-4]), tuple(int(v) for v in a0[-1]), ink2, 1, aa, 4, tipLength=0.6)
    # tracks: field, crossing, exposure
    tw = lay['track_w']
    for c, r in zip(res.cams, lay['tracks']):
        col = _bgr(CAM_COLORS[c['slot'] % 8])
        dim = tuple(int(0.35 * a + 0.65 * b) for a, b in zip(col, (30, 29, 28)))
        cv2.polylines(img, [arc(c['enter_ms'], c['enter_ms'] + c['sweep_ms'], r)], False, dim,
                      max(int(tw * 0.7), 2), aa, 4)
        x0, y0 = at(c['cross_ms'], r - tw * 0.45)
        x1, y1 = at(c['cross_ms'], r + tw * 0.45)
        cv2.line(img, pt(x0, y0), pt(x1, y1), col, 3, aa, 4)
        if c.get('trigger_ms') is not None:                 # the matched frame: centred misalign_ms off the crossing
            t0 = c['cross_ms'] + c['misalign_ms'] - st.readout_ms / 2 - st.exposure_ms / 2
            cv2.polylines(img, [arc(t0, t0 + st.readout_ms + st.exposure_ms, r)], False, ink, 1, aa, 4)
            x, y = at(t0 + st.readout_ms / 2 + st.exposure_ms / 2, r)
            cv2.circle(img, pt(x, y), int(4.5 * 16), ink, -1, aa, 4)
            cv2.circle(img, pt(x, y), int(4.5 * 16), col, 1, aa, 4)
        (w, h), _ = cv2.getTextSize(c['tag'], font, 0.42, 1)
        x, y = at(c['enter_ms'] - (w / 2 + 8) / (2 * math.pi * r) * T, r)      # just before the sweep enters
        text(c['tag'], x - w / 2, y + h / 2, col, 0.42)
    # trigger marks on the rim, one per line pulse
    for g in res.groups:
        t = g['trigger_ms']
        while t < T - 1e-9:
            x, y = at(t, lay['rim'] - 1)
            a = math.radians(float(lid.direction(t)))
            ux, uy = -math.sin(a), -math.cos(a)
            tri = np.array([[x, y], [x + ux * 9 - uy * 5, y + uy * 9 + ux * 5],
                            [x + ux * 9 + uy * 5, y + uy * 9 - ux * 5]])
            cv2.fillPoly(img, [np.round(tri * 16).astype(np.int32)], ink, aa, 4)
            (w, h), _ = cv2.getTextSize(g['letter'], font, 0.4, 1)
            lx, ly = at(t, lay['rim'] - 9)
            text(g['letter'], lx - w / 2, ly + h / 2, ink, 0.4)
            t += g['period_ms']
    # hub: the vehicle and the cameras about the lidar
    k_ = lay['scale']
    hp = lambda x, y: (cx - (y - lid.pos[1]) * k_, cy - (x - lid.pos[0]) * k_)
    if res.outline is not None:
        poly = np.round(np.array([hp(x, y) for x, y in res.outline]) * 16).astype(np.int32)
        cv2.fillPoly(img, [poly], (72, 70, 66), aa, 4)
        cv2.polylines(img, [poly], True, (150, 148, 142), 1, aa, 4)
    for c in res.cams:
        x, y = hp(*c['pos'][:2])
        cv2.circle(img, pt(x, y), 4 * 16, _bgr(CAM_COLORS[c['slot'] % 8]), -1, aa, 4)
    x, y = hp(*lid.pos[:2])
    cv2.circle(img, pt(x, y), 5 * 16, ink, 1, aa, 4)
    cv2.circle(img, pt(x, y), 2 * 16, ink, -1, aa, 4)
    # header
    text(f'Sync and timing plan  {lid.tag} {lid.name}, {lid.rate_hz:g} Hz {lid.rotation}, '
         f'phase {lid.phase_deg:.1f} deg', 16, 26, ink, 0.6)
    w = m['worst_ms']
    line = (f'worst misalignment {w:.1f} ms ({m["worst_travel_cm"]:.0f} cm at {st.speed_kmh:g} km/h)' if w is not None
            else 'no camera on a trigger line')
    line += f'   {m["lines_used"]} of {st.trigger_lines} trigger lines   {m["rotation_travel_m"]:.2f} m per rotation'
    text(line, 16, 50, ink2, 0.45)
    text('bar: sweep crosses the view   dot: middle row exposed   thin: first to last row', 16, size + head - 12,
         muted, 0.4)
    ok, buf = cv2.imencode('.png', img)
    return buf.tobytes()
