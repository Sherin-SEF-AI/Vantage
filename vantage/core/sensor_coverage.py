"""Lidar and radar coverage: where the rig's lidars and radars reach, what of them lands on the own vehicle, and how far
out they detect the objects a driving dataset needs, next to the cameras.

Lidar: every channel's beams are traced at a fixed azimuth step (each traced beam stands for step / h_res real points)
against the same occluder as the cameras: the engine's occlusion cube from the lidar's optical centre, so the body and
every rig part count. Hits within 5 cm of the optical centre are its own mount and ignored. A beam that meets the
vehicle before the ground and before its range lands on the own vehicle (the blocked share); the others that reach the
ground make the ground rings, the density map (points per m² per frame, summed over the lidars) and the ground blind
zone (per direction, how far from the body the nearest ground point any lidar reaches lies).

Objects stand on the ground as in the perception range (the same classes and sizes): azimuth about the vehicle centre,
counter-clockwise from +x, range from that centre on the ground. A lidar's points on an object are counted, not
traced: the channels whose elevation in the lidar's own frame (so a tilted lidar's real beam elevations at that
azimuth) lies between the object's bottom and top as seen from the lidar, whose beam the vehicle does not block before
the object, at a distance within the lidar's range on a 10 % target and beyond its nearest range; each such channel
puts the object's angular width / h_res points on it. The lidar range of a class is the farthest range at which the
points summed over all lidars reach the class's min_points.

Radar: the object's centre lies in a radar zone at the zone's range scaled by (rcs / 10 m²)^¼, with nothing of the
vehicle in between (hits within 0.15 m of the radar are the bumper fascia it sits behind). A class with rcs 0 is left
to the other sensors.

The camera range of a class is the perception range of the class of the same name; the required range per sector is
the perception range's, so all three are judged against one requirement: a sector is redundant when two or more
sensor types reach it, single with one, a gap with none.
"""
from __future__ import annotations

import json
import math
import os
import threading
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from . import perception as PR
from . import sensors as S
from .analysis import camera_frames
from .lens import resolve

SCHEMA = 'vantage.sensor_coverage/1'

# Typical detector needs: set your own. min_points: lidar points a detector needs on the object in one frame; rcs: its
# radar cross-section, m² (0: radars are not relied on for it). Sizes as the perception range's classes.
DEFAULT_CLASSES = [
    {'name': 'Car', 'height': 1.5, 'width': 1.8, 'base': 0.0, 'min_points': 20.0, 'rcs': 10.0},
    {'name': 'Pedestrian', 'height': 1.7, 'width': 0.5, 'base': 0.0, 'min_points': 8.0, 'rcs': 0.5},
    {'name': 'Cyclist', 'height': 1.7, 'width': 0.6, 'base': 0.0, 'min_points': 10.0, 'rcs': 1.0},
    {'name': 'Traffic sign', 'height': 0.6, 'width': 0.6, 'base': 2.0, 'min_points': 5.0, 'rcs': 0.0},
]
MODALITIES = ('camera', 'lidar', 'radar')
VERDICTS = ('redundant', 'single', 'gap')
AZ_STEP = 0.25                       # azimuth step of the traced lidar beams, degrees
OWN_MOUNT_M = 0.05                   # lidar: occluder hits this close to the optical centre are its own mount
FASCIA_M = 0.15                      # radar: occluder hits this close are the bumper fascia it sits behind
OVERLAP_AT = (10.0, 20.0, 50.0)      # scene distances for the camera-lidar overlap
RINGS_AT = 20.0                      # lidar rings across a camera's centre column at this distance
CAR_AT = (25.0, 50.0, 100.0)         # points on a car ahead of each lidar at these distances
IMAGE_GRID = (48, 36)                # camera image samples for the overlap (columns, rows)
COLUMN_SAMPLES = 600
BLOCKED_BINS = 72                    # blocked share per 5° of beam heading
DENS_LO, DENS_HI = 1.0, 1000.0       # density colour scale, points per m² per frame (log)
# one colour per sensor type, here and in the UI (lidar and radar as the sensors are drawn in the 3D and plan views)
MODALITY_COLORS = {'camera': '#9085e9', 'lidar': '#0aa5b8', 'radar': '#a0743c'}
SEQ = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6', '#256abf',
       '#1c5cab', '#184f95', '#104281', '#0d366b']       # the app's sequential ramp (ui.theme.SEQ)


@dataclass
class CoverageClass:
    name: str
    height: float
    width: float
    base: float = 0.0
    min_points: float = 10.0
    rcs: float = 1.0


def clean_class(c) -> dict | None:
    """An object class as stored ({name, height, width, base, min_points, rcs}), or None when it is not a usable one.
    min_points and rcs may be missing (a perception class): 10 points and 1 m² then."""
    try:
        o = {'name': str(c.get('name') or 'Object')[:40], 'height': float(c['height']), 'width': float(c['width']),
             'base': float(c.get('base', 0.0) or 0.0), 'min_points': float(c.get('min_points', 10.0)),
             'rcs': float(c.get('rcs', 1.0))}
    except (KeyError, TypeError, ValueError, AttributeError):
        return None
    if all(math.isfinite(v) for k, v in o.items() if k != 'name') and o['height'] > 0 and o['width'] > 0 \
            and o['base'] >= 0 and o['min_points'] > 0 and o['rcs'] >= 0:
        return o
    return None


@dataclass
class CoverageSettings:
    classes: list = field(default_factory=lambda: [dict(c) for c in DEFAULT_CLASSES])
    max_range: float = 250.0
    azimuths: int = 360
    samples: int = 120               # ranges per azimuth, log-spaced from 1 m to max_range
    grid_extent: float = 40.0        # side of the square ground density map, centred on the vehicle, m
    grid_cell: float = 0.25

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d: dict | None) -> 'CoverageSettings':
        s = CoverageSettings()
        d = d if isinstance(d, dict) else {}
        cls = [o for o in map(clean_class, d.get('classes') or []) if o is not None]
        seen = set()
        for o in cls:                    # names match the perception classes and key the verdicts: keep them unique
            base, n = o['name'], 2
            while o['name'] in seen:
                o['name'] = f'{base} {n}'
                n += 1
            seen.add(o['name'])
        if cls:
            s.classes = cls
        for k, lo, hi in (('max_range', 10.0, 1000.0), ('azimuths', 36, 1440), ('samples', 20, 400),
                          ('grid_extent', 10.0, 200.0), ('grid_cell', 0.1, 2.0)):
            if k in d:
                try:
                    v = float(d[k])
                except (TypeError, ValueError):
                    continue
                if math.isfinite(v):
                    setattr(s, k, type(getattr(s, k))(min(max(v, lo), hi)))
        if s.grid_extent / s.grid_cell > 800:                # at most 800 × 800 cells
            s.grid_cell = s.grid_extent / 800
        return s

    def object_classes(self) -> list:
        return [CoverageClass(**c) for c in self.classes]


@dataclass
class LidarCoverage:
    sensor: object                  # core.project.Sensor (of the analysed snapshot)
    spec: object                    # core.sensors.SensorModel
    blocked: float                  # share of each frame's points that land on the own vehicle or rig
    blocked_channel: np.ndarray     # (C,) per channel, lowest beam first
    blocked_sectors: dict           # perception sector -> share, by the beam's heading in plan
    blocked_az: np.ndarray          # (BLOCKED_BINS,) share per 5° of heading from +x, nan where it does not scan
    ring_m: float                   # nearest ground ring, horizontal distance from the lidar (inf: none)
    near: np.ndarray                # (A,) nearest ground point it reaches per azimuth, from the centre (inf: none)
    blind: np.ndarray               # (A,) the same from the body outline
    rings: list                     # [(channel, (N, 2))] its unblocked ground rings in plan
    far: np.ndarray                 # (K, A) range per class from this lidar alone
    car_points: dict                # distance ahead of the lidar (m) -> points on a car per frame
    points_per_frame: float

    @property
    def tag(self):
        return self.sensor.tag


@dataclass
class RadarCoverage:
    sensor: object
    spec: object
    far: np.ndarray                 # (K, A) range per class from this radar alone, nan where it ignores the class

    @property
    def tag(self):
        return self.sensor.tag


@dataclass
class CoverageResult:
    settings: CoverageSettings
    classes: list                   # [CoverageClass]
    required: dict                  # perception sector -> required range, m
    lidars: list                    # [LidarCoverage]
    radars: list                    # [RadarCoverage]
    cams: list                      # [(uid, tag, slot)]
    centre: tuple                   # (x, y) of the vehicle centre
    body: tuple | None              # ((x0, y0), (x1, y1)) footprint bounding box, None without a vehicle
    outline: np.ndarray             # (A,) body outline's distance from the centre per azimuth (0 without a vehicle)
    az: np.ndarray                  # (A,) degrees
    ranges: np.ndarray              # (R,) m, log-spaced
    camera: np.ndarray              # (K, A) camera range (the perception range), nan: no camera or no such class
    lidar: np.ndarray               # (K, A) lidar range, nan: no lidar
    radar: np.ndarray               # (K, A) radar range, nan: no radar or rcs 0
    points: np.ndarray              # (K, A, R) float32 lidar points per frame on the object, summed over the lidars
    near: np.ndarray                # (A,) nearest ground point any lidar reaches, from the centre (inf: none)
    blind: np.ndarray               # (A,) the same from the body outline
    density: np.ndarray             # (ny, nx) float32 lidar ground points per m² per frame
    grid: tuple                     # (x0, y0, cell) of the density map
    sectors: list                   # per sector: {'name', 'centre_deg', 'required_m', 'classes': {name: {...}}}
    overlap: list                   # per camera: lidar coverage of its image
    metrics: dict
    variant_uid: str = ''

    def az_index(self, az_deg) -> int:
        """Index of the azimuth sample nearest to a direction."""
        return int(round((az_deg % 360.0) / 360.0 * len(self.az))) % len(self.az)

    def at(self, k, az_deg) -> dict:
        """{'camera', 'lidar', 'radar'} range of class k in a direction (None where the sensor type has none)."""
        a = self.az_index(az_deg)
        return {m: (None if not math.isfinite(float(v[k, a])) else float(v[k, a]))
                for m, v in (('camera', self.camera), ('lidar', self.lidar), ('radar', self.radar))}

    def points_at(self, k, az_deg, r) -> float:
        """Lidar points per frame on class k at range r in a direction (nearest sample)."""
        a = self.az_index(az_deg)
        i = int(np.argmin(np.abs(np.log(self.ranges) - math.log(max(r, 1e-3)))))
        return float(self.points[k, a, i])


# =========================================================================== geometry
def _sector_of(az_deg):
    return PR._sector_of(az_deg)          # the perception range's sectors, so both judge the same directions


def _bins(az_deg, n):
    """Index of the nearest of n azimuth samples (k · 360 / n) for each azimuth."""
    return np.rint((np.asarray(az_deg, float) % 360.0) * n / 360.0).astype(int) % n


_OUTLINES = {}                       # (geometry version, n, centre) -> outline: a body's outline is drawn once
_outline_lock = threading.Lock()


def _outline(vg, centre, n) -> np.ndarray:
    """Distance from the centre to the body's outline in plan per azimuth sample, from the body's plan mask."""
    if vg is None:
        return np.zeros(n)
    key = (vg.version, n, round(centre[0], 4), round(centre[1], 4))
    with _outline_lock:
        if key in _OUTLINES:
            return _OUTLINES[key].copy()
    from .occlusion import _paint_lock          # QPainter rasterising on worker threads: one at a time
    cell = 0.04
    lo, hi = vg.bbox
    x0, y0 = float(lo[0]) - 0.1, float(lo[1]) - 0.1
    nx, ny = int(math.ceil((hi[0] - lo[0] + 0.2) / cell)), int(math.ceil((hi[1] - lo[1] + 0.2) / cell))
    with _paint_lock:
        mask = vg.body.plan_mask(x0, y0, cell, nx, ny)
    jj, ii = np.nonzero(mask)
    out = np.zeros(n)
    if not len(ii):
        return out
    for ox in (0.125, 0.375, 0.625, 0.875):  # 16 points per cell: the outline's samples stay denser than the bins
        for oy in (0.125, 0.375, 0.625, 0.875):
            dx = x0 + (ii + ox) * cell - centre[0]
            dy = y0 + (jj + oy) * cell - centre[1]
            np.maximum.at(out, _bins(np.degrees(np.arctan2(dy, dx)), n), np.hypot(dx, dy) + cell * 0.125)
    for _ in range(4):                       # a thin part seen end-on can still leave a bin empty
        gap = out == 0
        if not gap.any():
            break
        out[gap] = np.maximum(np.roll(out, 1), np.roll(out, -1))[gap]
    out = np.maximum(out, np.minimum(np.roll(out, 1), np.roll(out, -1)))      # no single-bin notches (aliasing)
    with _outline_lock:
        while len(_OUTLINES) >= 4:
            _OUTLINES.pop(next(iter(_OUTLINES)))
        _OUTLINES[key] = out
    return out.copy()


class _Lidar:
    """One lidar's traced beams (channels × azimuths) and what the vehicle does to them."""

    def __init__(self, s, m, cube):
        self.s, self.m = s, m
        self.pos = s.pos().astype(float)
        self.R = S.rotation(s)
        self.E = m.elevations()
        self.azs = S.scan_azimuths(m, AZ_STEP)
        self.closed = m.az_half() >= 180.0
        self.step = float(self.azs[1] - self.azs[0]) if len(self.azs) > 1 else 2 * m.az_half()
        self.weight = self.step / max(m.h_res, 1e-6)          # real points per traced beam
        D = S.directions(self.E, self.azs) @ self.R.T          # (C, N, 3) world
        self.D = D
        C, N = D.shape[:2]
        self.cube = cube
        if cube is not None:
            depth = cube.lookup(D.reshape(-1, 3))[0].reshape(C, N).astype(float)
            depth[depth <= OWN_MOUNT_M] = np.inf
        else:
            depth = np.full((C, N), np.inf)
        self.depth = depth
        self.hit = np.isfinite(depth)
        with np.errstate(divide='ignore', invalid='ignore'):
            tg = np.where(D[..., 2] < -1e-9, -self.pos[2] / D[..., 2], np.inf)
        self.tg = np.where(tg > 0, tg, np.inf)
        self.blocked = self.hit & (depth < np.minimum(self.tg, m.range_m))
        # horizontal reach (lidar frame) of each beam before the vehicle; an object beyond every hit of its column is
        # behind all of them, so the cumulative hit count per column answers most placements at once
        cosE = np.cos(np.radians(self.E))[:, None]
        self.vh = np.where(self.hit, depth * cosE, np.inf)
        self.vmax = np.where(self.hit, self.vh, 0.0).max(0)                       # (N,)
        self.cum = np.concatenate([np.zeros((N, 1)), np.cumsum(self.hit.T, axis=1)], axis=1)   # (N, C + 1)

    def col(self, alpha):
        j = np.rint((alpha - self.azs[0]) / self.step).astype(int)
        return j % len(self.azs) if self.closed else np.clip(j, 0, len(self.azs) - 1)

    def points(self, G, base, height, width) -> np.ndarray:
        """Points per frame on objects standing at ground points G (n, 2): base, height, width per object (or one)."""
        m = self.m
        G = np.asarray(G, float).reshape(-1, 2)
        n = len(G)
        base = np.broadcast_to(np.asarray(base, float), n)
        top = base + np.broadcast_to(np.asarray(height, float), n)
        width = np.broadcast_to(np.asarray(width, float), n)
        qb = (np.c_[G, base] - self.pos) @ self.R
        qt = (np.c_[G, top] - self.pos) @ self.R
        qc = 0.5 * (qb + qt)
        eb = np.degrees(np.arctan2(qb[:, 2], np.hypot(qb[:, 0], qb[:, 1])))
        et = np.degrees(np.arctan2(qt[:, 2], np.hypot(qt[:, 0], qt[:, 1])))
        c0 = np.searchsorted(self.E, np.minimum(eb, et), 'left')
        c1 = np.searchsorted(self.E, np.maximum(eb, et), 'right')
        alpha = np.degrees(np.arctan2(qc[:, 1], qc[:, 0]))
        rho = np.hypot(qc[:, 0], qc[:, 1])
        dist = np.linalg.norm(qc, axis=1)
        ok = (c1 > c0) & (dist >= m.range_min_m) & (dist <= m.range_m)
        if not self.closed:
            ok &= np.abs(alpha) <= m.az_half() + 1e-9
        j = self.col(alpha)
        reach = rho - (0.02 + 0.01 * rho)
        cnt = (c1 - c0).astype(float)
        beyond = reach > self.vmax[j]
        cnt = np.where(beyond, cnt - (self.cum[j, c1] - self.cum[j, c0]), cnt)
        idx = np.flatnonzero(ok & ~beyond)
        if len(idx):                          # near the vehicle: only the hits before the object block it
            jj, r_, a0, a1 = j[idx], reach[idx], c0[idx], c1[idx]
            nb = np.zeros(len(idx))
            for c in range(int(a0.min()), int(a1.max())):
                nb += (a0 <= c) & (c < a1) & (self.vh[c, jj] < r_)
            cnt[idx] -= nb
        gh = np.maximum(np.hypot(G[:, 0] - self.pos[0], G[:, 1] - self.pos[1]), 1e-6)
        wdeg = np.degrees(2.0 * np.arctan2(width / 2.0, gh))
        return np.where(ok, np.maximum(cnt, 0.0) * wdeg / max(m.h_res, 1e-6), 0.0)

    def sees(self, P) -> np.ndarray:
        """Vehicle-frame points (n, 3) in the lidar's field and range with nothing of the vehicle in between."""
        P = np.asarray(P, float).reshape(-1, 3)
        ok = S.in_field(self.s, self.m, P)
        if self.cube is not None and ok.any():
            d = P[ok] - self.pos
            dist = np.linalg.norm(d, axis=1)
            depth = self.cube.lookup(d)[0]
            ok[ok] = (depth > dist - (0.02 + 0.01 * dist)) | (depth <= OWN_MOUNT_M)
        return ok

    def ground(self):
        """(where each beam meets the ground (C, N, 3), mask (C, N) of the beams that reach it unblocked and within
        their range)."""
        m = self.m
        ok = ~self.blocked & np.isfinite(self.tg) & (self.tg <= m.range_m) & (self.tg >= m.range_min_m)
        P = self.pos + self.D * np.where(ok, self.tg, 0.0)[..., None]
        return P, ok

    def ring_m(self) -> float:
        """Nearest ground ring: the horizontal distance of the nearest ground point any beam meets within its range
        (the mount's geometry; what the vehicle blocks is in the blind zone)."""
        m = self.m
        t = np.where((self.tg <= m.range_m) & (self.tg >= m.range_min_m), self.tg, np.inf)
        h = t * np.hypot(self.D[..., 0], self.D[..., 1])
        return float(np.min(h)) if h.size else math.inf


class _Radar:
    def __init__(self, s, m, cube):
        self.s, self.m, self.cube = s, m, cube
        self.pos = s.pos().astype(float)

    def sees(self, P, rcs) -> np.ndarray:
        P = np.asarray(P, float).reshape(-1, 3)
        ok = S.in_field(self.s, self.m, P, rcs)
        if self.cube is not None and ok.any():
            d = P[ok] - self.pos
            dist = np.linalg.norm(d, axis=1)
            depth = self.cube.lookup(d)[0]
            ok[ok] = (depth > dist - (0.02 + 0.01 * dist)) | (depth <= FASCIA_M)
        return ok


def object_points(project, engine, sensor, cls, ground_xy) -> np.ndarray:
    """Points per frame a lidar puts on objects of class cls (CoverageClass) standing at ground points (n, 2), with
    the vehicle's blocking."""
    vg = engine.vehicle_geom(project)
    L = _Lidar(sensor, project.sensor_model_for(sensor), engine.cube(sensor.pos(), vg) if vg is not None else None)
    return L.points(ground_xy, cls.base, cls.height, cls.width)


def _far_range(ok, ranges, test=None, iters=14):
    """Per row of ok (A, R): the farthest range that passes, the last passing sample refined by bisection towards the
    next (failing) one with test(rows, r) -> bool."""
    A, R = ok.shape
    has = ok.any(1)
    i = R - 1 - np.argmax(ok[:, ::-1], axis=1)
    far = np.where(has, ranges[i], 0.0)
    rows = np.flatnonzero(has & (i < R - 1))
    if test is not None and len(rows):
        lo, hi = ranges[i[rows]].astype(float), ranges[i[rows] + 1].astype(float)
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            t = test(rows, mid)
            lo, hi = np.where(t, mid, lo), np.where(t, hi, mid)
        far[rows] = lo
    return far


def _runs(mask, closed):
    """Index arrays of the unbroken runs of True in a 1-D mask (a closed mask wraps around)."""
    n = len(mask)
    if mask.all():
        return [np.r_[np.arange(n), 0]] if closed else [np.arange(n)]
    out, start = [], None
    for i in range(n):
        if mask[i] and start is None:
            start = i
        elif not mask[i] and start is not None:
            out.append(np.arange(start, i))
            start = None
    if start is not None:
        out.append(np.arange(start, n))
    if closed and len(out) > 1 and mask[0] and mask[-1]:
        out[0] = np.r_[out.pop(), out[0]]
    return [r for r in out if len(r) > 1]


# =========================================================================== the study
def _project_for(project, variant):
    """The project with variant active (a shallow copy when it is not the active one)."""
    if variant is None or variant is project.variant:
        return project
    import copy
    q = copy.copy(project)
    q.variants = list(project.variants) + ([] if any(v is variant for v in project.variants) else [variant])
    q.active_uid = variant.uid
    return q


def run(project, engine, variant=None, settings: CoverageSettings | None = None, progress=None, cancel=None):
    """The lidar and radar coverage of a variant (the active one by default). Returns None when cancelled."""
    t0 = time.perf_counter()
    p = _project_for(project, variant)
    v = p.variant
    st = settings or CoverageSettings.from_dict(getattr(p.analysis, 'sensor_coverage', None))
    classes = st.object_classes()
    pst = PR.PerceptionSettings.from_dict(getattr(p.analysis, 'perception', None))
    required = dict(pst.required)
    K, A, Rn = len(classes), int(st.azimuths), int(st.samples)

    def prog(f):
        if progress is not None:
            progress(min(max(float(f), 0.0), 1.0))

    def stop():
        return cancel is not None and cancel()

    vg = engine.vehicle_geom(p)
    if vg is not None:
        lo, hi = vg.bbox
        centre = (float(lo[0] + hi[0]) / 2, float(lo[1] + hi[1]) / 2)
        body = ((float(lo[0]), float(lo[1])), (float(hi[0]), float(hi[1])))
    else:
        centre, body = (0.0, 0.0), None
    az = np.arange(A) * 360.0 / A
    ranges = np.geomspace(1.0, float(st.max_range), Rn)
    ca, sa = np.cos(np.radians(az)), np.sin(np.radians(az))
    G = np.stack([centre[0] + ca[:, None] * ranges[None, :], centre[1] + sa[:, None] * ranges[None, :]],
                 -1).reshape(-1, 2)

    def inside(Q):
        if body is None:
            return np.zeros(len(Q), bool)
        (x0, y0), (x1, y1) = body
        return (Q[:, 0] > x0 - 0.05) & (Q[:, 0] < x1 + 0.05) & (Q[:, 1] > y0 - 0.05) & (Q[:, 1] < y1 + 0.05)

    def ground_at(rows, r):
        return np.c_[centre[0] + ca[rows] * r, centre[1] + sa[rows] * r]
    ins = inside(G)
    outline = _outline(vg, centre, A)
    cams = [c for c in v.cameras if c.visible]
    sens = [s for s in v.sensors if s.visible]
    lids = [_Lidar(s, p.sensor_model_for(s), engine.cube(s.pos(), vg) if vg is not None else None)
            for s in sens if s.kind == 'lidar']
    rads = [_Radar(s, p.sensor_model_for(s), engine.cube(s.pos(), vg) if vg is not None else None)
            for s in sens if s.kind == 'radar']
    prog(0.05)
    if stop():
        return None

    # ---- cameras: the perception range, for the classes of the same name, on the same azimuths and out as far
    camera = np.full((K, A), np.nan)
    if cams:
        pst_c = PR.PerceptionSettings.from_dict(pst.to_dict())
        pst_c.azimuths, pst_c.max_range = A, float(st.max_range)
        pr = PR.analyse(p, engine, None, pst_c, progress=lambda f: prog(0.05 + 0.4 * f), cancel=cancel)
        if pr is None:
            return None
        names = [c.name for c in pr.classes]
        for k, cl in enumerate(classes):
            if cl.name in names:
                camera[k] = pr.far[names.index(cl.name)]
    prog(0.45)

    # ---- lidars: blocked beams, ground rings and density, the ground blind zone
    x0g, y0g = centre[0] - st.grid_extent / 2, centre[1] - st.grid_extent / 2
    nxg = nyg = int(math.ceil(st.grid_extent / st.grid_cell))
    density = np.zeros(nyg * nxg)
    near_all = np.full(A, np.inf)
    lid_out = []
    sec_names = [n for n, _ in PR.SECTORS]
    for li, L in enumerate(lids):
        if stop():
            return None
        m = L.m
        heading = np.degrees(np.arctan2(L.D[..., 1], L.D[..., 0]))
        sec = _sector_of(heading)
        bsec = {name: (float(L.blocked[sec == i].mean()) if (sec == i).any() else None)
                for i, name in enumerate(sec_names)}
        hb = np.floor(((heading % 360.0) + 180.0 / BLOCKED_BINS) / (360.0 / BLOCKED_BINS)).astype(int) % BLOCKED_BINS
        tot = np.bincount(hb.ravel(), minlength=BLOCKED_BINS)
        blk = np.bincount(hb.ravel(), weights=L.blocked.ravel().astype(float), minlength=BLOCKED_BINS)
        with np.errstate(invalid='ignore', divide='ignore'):
            baz = np.where(tot > 0, blk / np.maximum(tot, 1), np.nan)
        P, okg = L.ground()
        # density: every traced beam on the ground stands for step / h_res points
        Q = P[okg]
        ix = np.floor((Q[:, 0] - x0g) / st.grid_cell).astype(int)
        iy = np.floor((Q[:, 1] - y0g) / st.grid_cell).astype(int)
        g = (ix >= 0) & (ix < nxg) & (iy >= 0) & (iy < nyg)
        density += np.bincount(iy[g] * nxg + ix[g], minlength=nyg * nxg) * L.weight
        # the nearest ground point per direction about the centre (with the midpoints between neighbouring beams of
        # a channel, so a ring seen end-on from the centre leaves no direction empty)
        nb = np.roll(okg, -1, axis=1)
        mid_ok = okg & nb
        if not L.closed:
            mid_ok[:, -1] = False
        Mp = 0.5 * (P + np.roll(P, -1, axis=1))[mid_ok]
        Q2 = np.vstack([Q, Mp])
        near = np.full(A, np.inf)
        if len(Q2):
            dx, dy = Q2[:, 0] - centre[0], Q2[:, 1] - centre[1]
            np.minimum.at(near, _bins(np.degrees(np.arctan2(dy, dx)), A), np.hypot(dx, dy))
        near_all = np.minimum(near_all, near)
        rings = []
        sub = max(1, int(round(1.0 / L.step)))              # about 1° between the drawn points
        idx = np.arange(0, len(L.azs), sub)
        for c in range(len(L.E)):
            mk = okg[c, idx]
            if not mk.any():
                continue
            for run_ in _runs(mk, L.closed and len(L.azs) % sub == 0):
                rings.append((c, P[c, idx[run_], :2].copy()))
        car = next((cl for cl in classes if cl.name.lower() == 'car'), CoverageClass(**DEFAULT_CLASSES[0]))
        hd = math.radians(S.heading(L.s))
        car_pts = {}
        for d in CAR_AT:
            Gc = np.array([[L.pos[0] + d * math.cos(hd), L.pos[1] + d * math.sin(hd)]])
            car_pts[d] = float(L.points(Gc, car.base, car.height, car.width)[0])
        lid_out.append(LidarCoverage(
            L.s, m, float(L.blocked.mean()), L.blocked.mean(1), bsec, baz, L.ring_m(), near,
            np.where(np.isfinite(near), np.maximum(near - outline, 0.0), np.inf), rings, np.zeros((K, A)), car_pts,
            float(m.n_channels() * m.columns())))
        prog(0.45 + 0.15 * (li + 1) / max(len(lids), 1))
    density = (density / st.grid_cell ** 2).reshape(nyg, nxg).astype(np.float32)

    # ---- objects for the lidars: points on the object, summed over the lidars
    points = np.zeros((K, A, Rn), np.float32)
    lidar = np.full((K, A), np.nan)
    for k, cl in enumerate(classes):
        if stop():
            return None
        if not lids:
            break
        each = []
        for L in lids:
            each.append(np.where(ins, 0.0, L.points(G, cl.base, cl.height, cl.width)).reshape(A, Rn))
        tot = np.sum(each, axis=0)
        points[k] = tot

        def test(rows, r, Ls=lids, cl=cl):
            Gr = ground_at(rows, r)
            n = np.zeros(len(rows))
            for L in Ls:
                n += L.points(Gr, cl.base, cl.height, cl.width)
            return (n >= cl.min_points) & ~inside(Gr)
        lidar[k] = _far_range(tot >= cl.min_points, ranges, test)
        for li, L in enumerate(lids):
            lid_out[li].far[k] = _far_range(each[li] >= cl.min_points, ranges,
                                            lambda rows, r, L=L, cl=cl: test(rows, r, [L], cl))
        prog(0.6 + 0.2 * (k + 1) / K)

    # ---- objects for the radars: the centre in a zone, nothing of the vehicle in between
    radar = np.full((K, A), np.nan)
    rad_out = [RadarCoverage(R_.s, R_.m, np.full((K, A), np.nan)) for R_ in rads]
    for k, cl in enumerate(classes):
        if stop():
            return None
        if not rads or cl.rcs <= 0:
            continue
        zc = cl.base + cl.height / 2
        P = np.c_[G, np.full(len(G), zc)]
        each = [(R_.sees(P, cl.rcs) & ~ins).reshape(A, Rn) for R_ in rads]

        def test(rows, r, Rs=rads, cl=cl, zc=zc):
            Gr = ground_at(rows, r)
            Pr = np.c_[Gr, np.full(len(Gr), zc)]
            ok = np.zeros(len(rows), bool)
            for R_ in Rs:
                ok |= R_.sees(Pr, cl.rcs)
            return ok & ~inside(Gr)
        radar[k] = _far_range(np.any(each, axis=0), ranges, test)
        for qi, R_ in enumerate(rads):
            rad_out[qi].far[k] = _far_range(each[qi], ranges, lambda rows, r, R_=R_, cl=cl: test(rows, r, [R_], cl))
    prog(0.88)

    # ---- sectors: shortest range per sensor type against the one requirement
    sec = _sector_of(az)
    sectors = []
    counts = dict.fromkeys(VERDICTS, 0)
    for si, (name, cdeg) in enumerate(PR.SECTORS):
        msk = sec == si
        req = float(required.get(name, 0.0))
        out = {}
        for k, cl in enumerate(classes):
            vals = {}
            for mod, arr in (('camera', camera), ('lidar', lidar), ('radar', radar)):
                vv = arr[k, msk]
                vals[mod] = float(vv.min()) if vv.size and np.isfinite(vv).all() else None
            meets = {mod: (x is not None and x >= req) for mod, x in vals.items()}
            n = sum(meets.values())
            verdict = 'redundant' if n >= 2 else 'single' if n == 1 else 'gap'
            counts[verdict] += 1
            out[cl.name] = {'camera_m': vals['camera'], 'lidar_m': vals['lidar'], 'radar_m': vals['radar'],
                            'meets': meets, 'modalities': n, 'verdict': verdict}
        sectors.append({'name': name, 'centre_deg': cdeg, 'required_m': req, 'classes': out})
    if stop():
        return None

    # ---- cameras and lidars: can the lidar points be projected into each camera?
    overlap = [_overlap(p, engine, vg, c, lids) for c in cams] if lids else \
        [{'uid': c.uid, 'tag': c.tag, 'slot': c.slot, 'share': {f'{d:g}': 0.0 for d in OVERLAP_AT}, 'per_lidar': {},
          'best': None, 'best_share': 0.0, 'rings': 0} for c in cams]
    prog(1.0)
    metrics = {'seconds': time.perf_counter() - t0, 'lidars': len(lids), 'radars': len(rads), 'cameras': len(cams),
               'checks': len(sectors) * K, **counts,
               'lidar_points_per_s': float(sum(L.m.pps() for L in lids))}
    return CoverageResult(st, classes, required, lid_out, rad_out, [(c.uid, c.tag, c.slot) for c in cams], centre, body,
                          outline, az, ranges, camera, lidar, radar, points, near_all,
                          np.where(np.isfinite(near_all), np.maximum(near_all - outline, 0.0), np.inf), density,
                          (x0g, y0g, float(st.grid_cell)), sectors, overlap, metrics, v.uid)


def _overlap(p, engine, vg, cam, lids) -> dict:
    """How much of a camera's image the lidars cover: scene points at OVERLAP_AT along rays sampled over the image
    (within the lens circle and not on the body; on the ground where a ray meets it sooner), in some lidar's field and
    range and not hidden from it; and the best lidar's rings across the image's centre column at RINGS_AT."""
    intr = resolve(p.module_for(cam), p.lens_for(cam))
    _, W2O = camera_frames(cam)
    pos = cam.pos().astype(float)
    cube = engine.cube(pos, vg) if vg is not None else None

    def scene(u, v, D):
        d_opt, th = intr.unproject(u, v)
        dw = d_opt @ W2O                                    # optical -> vehicle frame
        ok = th <= intr.max_theta
        with np.errstate(divide='ignore', invalid='ignore'):
            tg = np.where(dw[:, 2] < -1e-9, -pos[2] / dw[:, 2], np.inf)
        t = np.minimum(D, np.where(tg > 0, tg, np.inf))
        if cube is not None:
            depth = cube.lookup(dw)[0]
            ok &= depth > t - (0.02 + 0.01 * t)
        return pos + dw * t[:, None], ok
    nu, nv = IMAGE_GRID
    U, V = np.meshgrid((np.arange(nu) + 0.5) * intr.width / nu, (np.arange(nv) + 0.5) * intr.height / nv)
    share, per = {}, np.zeros((len(lids), len(OVERLAP_AT)))
    for di, D in enumerate(OVERLAP_AT):
        P, ok = scene(U.ravel(), V.ravel(), D)
        n = max(int(ok.sum()), 1)
        anyl = np.zeros(len(P), bool)
        for li, L in enumerate(lids):
            cov = ok & L.sees(P)
            per[li, di] = cov.sum() / n
            anyl |= cov
        share[f'{D:g}'] = float(anyl.sum() / n) if ok.any() else 0.0
    i20 = int(np.argmin(np.abs(np.array(OVERLAP_AT) - RINGS_AT)))
    best = int(np.argmax(per[:, i20])) if len(lids) and per[:, i20].max() > 0 else -1
    rings = 0
    if best >= 0:
        L = lids[best]
        v = np.linspace(0.0, intr.height, COLUMN_SAMPLES)
        P, ok = scene(np.full(len(v), intr.cx), v, RINGS_AT)
        ok &= L.sees(P)
        _, _, el = S.angles(L.s, P)
        c0, c1 = ok[:-1], ok[1:]
        pair = c0 | c1
        if pair.any():
            # between two covered samples, every beam whose elevation lies between theirs crosses the column; where
            # only one is covered, the span reaches towards the other as far as the lidar's field (its edge beam)
            e0, e1 = el[:-1], el[1:]
            a = np.where(c0, e0, np.clip(e0, L.E[0], L.E[-1]))
            b = np.where(c1, e1, np.clip(e1, L.E[0], L.E[-1]))
            lo, hi = np.minimum(a, b)[pair], np.maximum(a, b)[pair]
            mark = np.zeros(len(L.E) + 1)
            np.add.at(mark, np.searchsorted(L.E, lo, 'left'), 1)
            np.add.at(mark, np.searchsorted(L.E, hi, 'right'), -1)
            rings = int((np.cumsum(mark)[:-1] > 0).sum())
    return {'uid': cam.uid, 'tag': cam.tag, 'slot': cam.slot, 'share': share,
            'per_lidar': {L.s.tag: float(per[li, i20]) for li, L in enumerate(lids)},
            'best': lids[best].s.tag if best >= 0 else None, 'best_share': float(per[best, i20]) if best >= 0 else 0.0,
            'rings': rings}


# =========================================================================== output
def _r(v, nd=1):
    return None if v is None or not math.isfinite(float(v)) else round(float(v), nd)


def _stats(a):
    f = np.asarray(a, float)
    f = f[np.isfinite(f)]
    if not f.size:
        return {'min_m': None, 'median_m': None, 'max_m': None, 'directions_reached': 0.0}
    return {'min_m': round(float(f.min()), 2), 'median_m': round(float(np.median(f)), 2),
            'max_m': round(float(f.max()), 2), 'directions_reached': round(f.size / len(a), 3)}


def summary(res: CoverageResult) -> dict:
    """Compact figures (the assistant, the batch report header)."""
    m = res.metrics
    return {
        'sectors': {cl.name: {s['name']: {'required_m': s['required_m'],
                                          **{f'{mod}_m': _r(s['classes'][cl.name][f'{mod}_m']) for mod in MODALITIES},
                                          'verdict': s['classes'][cl.name]['verdict']} for s in res.sectors}
                    for cl in res.classes},
        'verdicts': {v: m[v] for v in VERDICTS}, 'checks': m['checks'],
        'lidars': [{'lidar': L.tag, 'model': L.spec.name, 'blocked_pct': round(100 * L.blocked, 1),
                    'nearest_ring_m': _r(L.ring_m, 2), 'ground_blind_from_body': _stats(L.blind),
                    'points_on_car_ahead': {f'{d:g} m': round(v, 1) for d, v in L.car_points.items()}}
                   for L in res.lidars],
        'radars': [{'radar': R_.tag, 'model': R_.spec.name,
                    'farthest_m': {cl.name: _r(np.nanmax(R_.far[k]) if np.isfinite(R_.far[k]).any() else None)
                                   for k, cl in enumerate(res.classes) if cl.rcs > 0}} for R_ in res.radars],
        'ground_blind_from_body': _stats(res.blind),
        'camera_lidar_overlap': [{'camera': o['tag'], 'lidar_share_pct': {k: round(100 * v, 1)
                                                                          for k, v in o['share'].items()},
                                  'best_lidar': o['best'], 'rings_at_20m': o['rings']} for o in res.overlap],
    }


def to_doc(res: CoverageResult, project=None, L=None, A=None, origin=(0.0, 0.0)) -> dict:
    """The JSON export. L converts lengths to the export units, A angles; origin is the export frame's origin in the
    vehicle frame ((x, y) or (x, y, z)). Densities stay in points per m² per frame."""
    L = L or (lambda v: round(float(v), 4))
    A = A or (lambda v: round(float(v), 4))
    o = [float(x) for x in origin] + [0.0] * (3 - len(origin))
    rng = lambda v: None if v is None or not math.isfinite(float(v)) else L(v)
    pos = lambda s: [L(s.x - o[0]), L(s.y - o[1]), L(s.z - o[2])]
    x0, y0, cell = res.grid
    v = None if project is None else (project.variant_by_uid(res.variant_uid) or project.variant)
    return {
        'schema': SCHEMA,
        'project': getattr(project, 'name', ''), 'variant': v.name if v is not None else '',
        'centre': [L(res.centre[0] - o[0]), L(res.centre[1] - o[1])],
        'directions': 'azimuth about the vehicle centre, counter-clockwise from +x (forward); range from that centre '
                      'on the ground',
        'detected_when': {
            'camera': 'the perception range of the class of the same name (pixel height through the real lens)',
            'lidar': 'the points summed over all lidars on the object reach min_points: per channel whose elevation '
                     'lies between the object\'s bottom and top as seen from the lidar, not blocked by the vehicle and '
                     'within its range on a 10 % target, the object\'s angular width / h_res points',
            'radar': 'the object\'s centre lies in a radar zone at the zone\'s range scaled by (rcs / 10 m²)^0.25, '
                     'with nothing of the vehicle in between; rcs 0 leaves the class to the other sensors'},
        'classes': [{'name': c.name, 'height': L(c.height), 'width': L(c.width), 'above_ground': L(c.base),
                     'min_points': c.min_points, 'rcs_m2': c.rcs} for c in res.classes],
        'required_ranges': {k: L(v) for k, v in res.required.items()},
        'azimuth': [A(a) for a in res.az],
        'per_class': {c.name: {mod: [rng(v) for v in arr[k]] for mod, arr in
                               (('camera', res.camera), ('lidar', res.lidar), ('radar', res.radar))}
                      for k, c in enumerate(res.classes)},
        'sectors': [{'name': s['name'], 'centre': A(s['centre_deg']), 'required': L(s['required_m']),
                     'classes': {n: {'camera': rng(v['camera_m']), 'lidar': rng(v['lidar_m']),
                                     'radar': rng(v['radar_m']), 'meets': v['meets'], 'modalities': v['modalities'],
                                     'verdict': v['verdict']} for n, v in s['classes'].items()}}
                    for s in res.sectors],
        'lidars': [{'tag': li.tag, 'model': li.spec.name, 'model_id': li.spec.id, 'position': pos(li.sensor),
                    'yaw': A(li.sensor.yaw), 'pitch': A(li.sensor.pitch), 'roll': A(li.sensor.roll),
                    'points_per_frame': round(li.points_per_frame), 'blocked_share': round(li.blocked, 4),
                    'blocked_per_channel': [round(float(b), 4) for b in li.blocked_channel],
                    'channel_elevation': [A(e) for e in li.spec.elevations()],
                    'blocked_per_sector': {k: (None if v is None else round(v, 4))
                                           for k, v in li.blocked_sectors.items()},
                    'nearest_ring': rng(li.ring_m),
                    'ground_blind_from_body': [rng(v) for v in li.blind],
                    'per_class': {c.name: [rng(v) for v in li.far[k]] for k, c in enumerate(res.classes)},
                    'points_on_car_ahead': [{'distance': L(d), 'points': round(v, 2)}
                                            for d, v in li.car_points.items()]}
                   for li in res.lidars],
        'radars': [{'tag': r.tag, 'model': r.spec.name, 'model_id': r.spec.id, 'position': pos(r.sensor),
                    'yaw': A(r.sensor.yaw), 'pitch': A(r.sensor.pitch), 'roll': A(r.sensor.roll),
                    'zones': [{'range_on_car': L(z[0]), 'azimuth_half': A(z[1]), 'elevation_half': A(z[2])}
                              for z in r.spec.zones],
                    'per_class': {c.name: [rng(v) for v in r.far[k]] for k, c in enumerate(res.classes) if c.rcs > 0}}
                   for r in res.radars],
        'ground_blind_from_body': [rng(v) for v in res.blind],
        'camera_lidar': {'distances': [L(d) for d in OVERLAP_AT], 'rings_at': L(RINGS_AT),
                         'cameras': [{'camera': ov['tag'],
                                      'lidar_share': {k: round(v, 4) for k, v in ov['share'].items()},
                                      'per_lidar': {k: round(v, 4) for k, v in ov['per_lidar'].items()},
                                      'best_lidar': ov['best'], 'rings_in_centre_column': ov['rings']}
                                     for ov in res.overlap]},
        'ground_density': {'unit': 'lidar points per m² per frame', 'origin': [L(x0 - o[0]), L(y0 - o[1])],
                           'cell': L(cell), 'rows': 'along +y, each along +x',
                           'values': [[round(float(v), 1) if v else 0 for v in row] for row in res.density]},
    }


# =========================================================================== the plan image
PNG_RINGS = PR.PNG_RINGS


def _hex_rgb(h):
    return int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)


def heat_rgba(res: CoverageResult, n: int, rmax: float) -> np.ndarray:
    """The ground density as an (n, n, 4) RGBA image of the plan on the square-root range scale: the vehicle centre in
    the middle, rmax at n / 2 pixels, forward up, left to the left; transparent where no lidar point lands."""
    k = (n / 2) / math.sqrt(max(rmax, 1.0))
    yy, xx = np.mgrid[0:n, 0:n] + 0.5
    dx, dy = xx - n / 2, yy - n / 2
    s = np.hypot(dx, dy)
    r = (s / k) ** 2
    a = np.arctan2(-dx, -dy)                         # forward up, left to the left
    X = res.centre[0] + r * np.cos(a)
    Y = res.centre[1] + r * np.sin(a)
    x0, y0, cell = res.grid
    ny, nx = res.density.shape
    ix, iy = np.floor((X - x0) / cell).astype(int), np.floor((Y - y0) / cell).astype(int)
    ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    v = np.zeros((n, n), np.float32)
    v[ok] = res.density[iy[ok], ix[ok]]
    lut = np.array([_hex_rgb(c) for c in SEQ], np.uint8)
    t = (np.log10(np.maximum(v, 1e-9)) - math.log10(DENS_LO)) / (math.log10(DENS_HI) - math.log10(DENS_LO))
    idx = np.clip(np.rint(t * (len(SEQ) - 1)), 0, len(SEQ) - 1).astype(int)
    out = np.zeros((n, n, 4), np.uint8)
    out[..., :3] = lut[idx]
    out[..., 3] = np.where((v > 0) & (s <= n / 2), 215, 0)
    return out


def png(res: CoverageResult, size=900, k_sel=0, layer='density') -> bytes:
    """The plan as PNG: forward up, left to the left, square-root range scale with rings; for class k_sel the camera,
    lidar and radar range envelopes, the required range per sector coloured by its verdict (green redundant, amber
    single, red gap), the radar zones for its cross-section, and the lidar ground density (layer 'density') or the
    ground rings (layer 'rings'); the body in the middle."""
    import cv2
    head = 56
    img = np.full((size + head, size, 3), (30, 29, 28), np.uint8)
    cx, cy = size / 2, head + size / 2
    rmax = float(res.ranges[-1])
    scale = (size / 2 - 34) / math.sqrt(rmax)
    k_sel = min(max(int(k_sel or 0), 0), len(res.classes) - 1)
    cl = res.classes[k_sel]
    font, aa = cv2.FONT_HERSHEY_SIMPLEX, cv2.LINE_AA
    bgr = lambda h: _hex_rgb(h)[::-1]

    def at(r, a):
        s, t = math.sqrt(max(r, 0.0)) * scale, math.radians(a)
        return cx - math.sin(t) * s, cy - math.cos(t) * s

    def world(x, y):
        dx, dy = x - res.centre[0], y - res.centre[1]
        return at(math.hypot(dx, dy), math.degrees(math.atan2(dy, dx)))

    def poly(pts):
        return np.round(np.array(pts, float).reshape(-1, 2) * 16).astype(np.int32)       # 4 fractional bits

    def text(t, x, y, col, s=0.45, th=1):
        cv2.putText(img, t, (int(x), int(y)), font, s, col, th, aa)
    if layer == 'density':
        n = int(2 * (size / 2 - 34))
        h = heat_rgba(res, n, rmax).astype(float)
        y0, x0 = int(round(cy - n / 2)), int(round(cx - n / 2))
        roi = img[y0:y0 + n, x0:x0 + n].astype(float)
        al = h[..., 3:4] / 255.0 * 0.8
        img[y0:y0 + n, x0:x0 + n] = (roi * (1 - al) + h[..., 2::-1] * al).astype(np.uint8)
    for ring in PNG_RINGS:
        if ring <= rmax * 1.001:
            rr = int(round(math.sqrt(ring) * scale * 16))
            cv2.circle(img, (int(cx * 16), int(cy * 16)), rr, (62, 61, 58), 1, aa, 4)
            text(f'{ring} m', cx + 4, cy - rr / 16 + 14, (140, 140, 135), 0.4)
    for name, a in PR.SECTORS:
        x, y = at(rmax, a - 22.5)
        cv2.line(img, (int(cx), int(cy)), (int(x), int(y)), (52, 51, 49), 1, aa)
    if layer == 'rings':
        col = bgr(MODALITY_COLORS['lidar'])
        for li in res.lidars:
            for _, P in li.rings:
                pts = [world(x, y) for x, y in P if math.hypot(x - res.centre[0], y - res.centre[1]) <= rmax]
                if len(pts) > 1:
                    cv2.polylines(img, [poly(pts)], False, tuple(int(c * 0.7) for c in col), 1, aa, 4)
    if cl.rcs > 0:
        over = img.copy()
        for r_ in res.radars:
            for w in S.radar_wedges(r_.sensor, r_.spec, cl.rcs):
                cv2.fillPoly(over, [poly([world(x, y) for x, y in w])], bgr(MODALITY_COLORS['radar']), aa, 4)
        cv2.addWeighted(over, 0.12, img, 0.88, 0, img)
    tone = {'redundant': (12, 163, 12), 'single': (25, 178, 250), 'gap': (59, 59, 208)}     # good, warning, critical
    for s in res.sectors:
        pts = [at(s['required_m'], s['centre_deg'] - 22.5 + t * 45.0 / 24) for t in range(25)]
        for j in range(0, 24, 2):                                         # dashed
            cv2.polylines(img, [poly(pts[j:j + 2])], False, tone[s['classes'][cl.name]['verdict']], 2, aa, 4)
    if res.lidars:                                                        # the nearest ground any lidar reaches
        ok = np.isfinite(res.near)
        pts = [at(r, a) if g else None for r, a, g in zip(res.near, res.az, ok)]
        for j in range(0, len(pts) - 1, 2):                               # dotted
            if pts[j] is not None and pts[j + 1] is not None:
                cv2.polylines(img, [poly(pts[j:j + 2])], False, bgr(MODALITY_COLORS['lidar']), 1, aa, 4)
    for mod, arr in (('radar', res.radar), ('lidar', res.lidar), ('camera', res.camera)):
        if np.isfinite(arr[k_sel]).all():
            cv2.polylines(img, [poly([at(r, a) for r, a in zip(arr[k_sel], res.az)])], True,
                          bgr(MODALITY_COLORS[mod]), 2, aa, 4)
    if res.body is not None:
        edge = poly([at(r, a) for r, a in zip(res.outline, res.az)])
        cv2.fillPoly(img, [edge], (72, 70, 66), aa, 4)
        cv2.polylines(img, [edge], True, (150, 148, 142), 1, aa, 4)
    for li in res.lidars:
        x, y = world(li.sensor.x, li.sensor.y)
        cv2.circle(img, (int(x), int(y)), 4, bgr(MODALITY_COLORS['lidar']), -1, aa)
        text(li.tag, x + 6, y - 4, (220, 219, 214), 0.4)
    for r_ in res.radars:
        x, y = world(r_.sensor.x, r_.sensor.y)
        cv2.rectangle(img, (int(x) - 3, int(y) - 3), (int(x) + 3, int(y) + 3), bgr(MODALITY_COLORS['radar']), -1)
        text(r_.tag, x + 6, y + 12, (220, 219, 214), 0.4)
    for name, a in PR.SECTORS:
        x, y = at(rmax * 0.96, a)
        (w, h), _ = cv2.getTextSize(name, font, 0.45, 1)
        text(name, min(max(x - w / 2, 4), size - w - 4), min(max(y + h / 2, head + h + 4), size + head - 6),
             (175, 174, 168))
    m = res.metrics
    text(f'Lidar and radar coverage  {cl.name}:  {m["redundant"]} redundant, {m["single"]} single, {m["gap"]} gaps '
         f'of {m["checks"]} sector checks', 16, 24, (236, 235, 230), 0.55)
    x = 16
    for mod in MODALITIES:
        v = res.at(k_sel, 0.0)[mod]
        label = f'{mod} ' + (f'{v:.0f} m ahead' if v is not None else '-')
        cv2.rectangle(img, (x, 38), (x + 10, 48), bgr(MODALITY_COLORS[mod]), -1)
        text(label, x + 15, 48, (220, 219, 214), 0.45)
        x += cv2.getTextSize(label, font, 0.45, 1)[0][0] + 32
    text('dashed: required range (green 2+ sensor types, amber 1, red none); dotted: nearest lidar ground',
         size - 650, size + head - 10, (140, 140, 135), 0.4)
    ok, buf = cv2.imencode('.png', img)
    return buf.tobytes()


def export(res: CoverageResult, project, out_dir, units=None, angles='deg', k_sel=0) -> list[str]:
    """sensor_coverage.json (in the project's frame, in m or mm) and sensor_coverage.png into out_dir."""
    units = units or getattr(project, 'units', 'm')
    units = units if units in ('m', 'mm') else 'm'
    sc, nd = (1000.0, 0) if units == 'mm' else (1.0, 3)

    def L(v):
        r = round(float(v) * sc, nd) + 0.0
        return int(r) if nd == 0 else r

    def A(v):
        return round(math.radians(float(v)), 6) + 0.0 if angles == 'rad' else round(float(v), 4) + 0.0
    origin = np.asarray(project.origin_offset(), float) if hasattr(project, 'origin_offset') else np.zeros(3)
    doc = to_doc(res, project, L, A, origin)
    head = {'project': doc.pop('project'), 'variant': doc.pop('variant'),
            'frame': getattr(project, 'frame', 'rear_axle'), 'units': units, 'angles': angles,
            'created': time.strftime('%Y-%m-%dT%H:%M:%S')}
    doc = {'schema': doc.pop('schema'), **head, **doc}
    os.makedirs(out_dir, exist_ok=True)
    paths = [os.path.join(out_dir, 'sensor_coverage.json'), os.path.join(out_dir, 'sensor_coverage.png')]
    with open(paths[0], 'w', encoding='utf-8') as fh:
        fh.write(json.dumps(doc, indent=1, ensure_ascii=False) + '\n')
    with open(paths[1], 'wb') as fh:
        fh.write(png(res, k_sel=k_sel))
    return paths
