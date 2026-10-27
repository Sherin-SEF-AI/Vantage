"""Road geometry and body motion: what the cameras see on a real road instead of flat ground, and how braking,
accelerating and cornering move them.

The road (analysis.terrain 'features')
- A height field z = h(x, y) in the vehicle frame (ISO 8855: x forward, y left, z up, origin at the rear-axle centre on
  flat ground, metres): the sum of road primitives, each a profile across one direction.
  ramp: a grade (%) starting a distance ahead of the front bumper (or behind the rear one);
  crest: a vertical curve of a radius from level into a grade (negative: the road falls away beyond it; positive: a
         sag); garage: a ramp built with a transition at half its grade; speed bump: a cosine hump of a height and
         length; kerb: a step of a height, beside the car (left / right of the body's side) or across the road (ahead /
         behind); crowned road: the cross fall (%) from a crown line falling to both sides.
  A distance may be negative: the primitive starts under the car.
- The vehicle stands on the road: each tyre (a circle of the wheel radius with a flat tread) rests on the height field,
  and the suspension (core.suspension, the kerb state) takes the twist four wheels cannot all follow (a kerb under one
  wheel), so the body's heave, pitch and roll come from the same static solve as the load cases.
- A ground point is seen when it is in the camera's image, the body (and rig, housings) does not hide it (the cube
  maps, as on flat ground), it faces the camera, and the camera's ray reaches it without passing under the road on the
  way (a ray march through the height field). Engine.analyse does this for the coverage analysis whenever the project
  has a road (and it is switched on); without one the analysis is the flat-ground one, unchanged.

This study (run) reports what the road changes against flat ground: blind area and density, how far the road ahead and
behind stays in view (a crest or a garage ramp hides what lies beyond), the distance at which a speed bump or a kerb is
first seen as the car approaches it and where it drops out of view near the bumper, and whether the body or a sensor
would touch the road driving over a primitive.

Body motion (analysis.terrain 'motion'): brake dive, acceleration squat and cornering roll from the suspension model
(core.suspension.transfer: wheel rates and anti-roll bars, a centre-of-gravity height, anti-dive / anti-lift /
anti-squat percentages and roll centres, typical defaults), relative to the load the cameras are calibrated at: pitch,
roll and heave per case, every camera's horizon shift (rows and degrees) and, as in the load cases, the blind area and
the stitched ground's shift with the body moved (on flat ground).
"""
from __future__ import annotations

import csv
import io
import json
import math
import os
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from . import loadcases as LC
from . import suspension as SU
from .mathutil import euler_zyx

SCHEMA = 'vantage.terrain/1'
KINDS = ('ramp', 'crest', 'garage', 'bump', 'kerb', 'crown')
KIND_LABEL = {'ramp': 'Ramp', 'crest': 'Crest', 'garage': 'Garage ramp', 'bump': 'Speed bump', 'kerb': 'Kerb',
              'crown': 'Crowned road'}
WHERE = ('ahead', 'behind')
SIDES = ('left', 'right', 'ahead', 'behind')
SEVERITIES = ('critical', 'serious', 'warning', 'note', 'good')
ISSUE_SEVERITIES = ('critical', 'serious', 'warning')
MAX_FEATURES = 12
FACE_M = 0.01           # a kerb's face rises over 1 cm: near vertical, and still a height field
STEP_M = 0.02           # ray-march step along the part of a ray low enough for the road to reach it
MAX_SAMPLES = 256       # samples per ray at most (a long ray over a smooth crest is sampled more coarsely)
TREAD_M = 0.19          # a tyre's flat tread width (typical 185-205 mm tyres)
GAP_M = 0.5             # the road counts as in view across a shorter stretch out of view (a bump's shadow)

# per kind: (key, label, unit, default, lo, hi, decimals, step, scale (stored × scale = shown), tip)
PARAMS = {
    'ramp': (('grade', 'Grade', '%', 8.0, -40.0, 40.0, 1, 0.5, 1.0,
              'Rise of the road per 100 m beyond the start (negative: it falls away)'),
             ('start', 'Starts', 'm', 3.0, -8.0, 200.0, 2, 0.1, 1.0,
              'From the bumper to where the grade begins (negative: under the car)')),
    'crest': (('radius', 'Radius', 'm', 300.0, 5.0, 20000.0, 0, 10.0, 1.0,
               'Radius of the vertical curve: the smaller, the sooner the road beyond drops out of view'),
              ('grade', 'Grade beyond', '%', -8.0, -40.0, 40.0, 1, 0.5, 1.0,
               'The grade the curve leads into, against the car\'s own (negative: a crest, the road falls away; '
               'positive: a sag)'),
              ('start', 'Starts', 'm', 5.0, -8.0, 200.0, 2, 0.1, 1.0,
               'From the bumper to where the road starts to curve')),
    'garage': (('grade', 'Grade', '%', -15.0, -40.0, 40.0, 1, 0.5, 1.0,
                'The ramp\'s grade (negative: down into the garage)'),
               ('transition', 'Transition', 'm', 2.4, 0.0, 20.0, 2, 0.1, 1.0,
                'Length of the transition at half the grade between the level floor and the ramp'),
               ('start', 'Starts', 'm', 1.0, -8.0, 200.0, 2, 0.1, 1.0,
                'From the bumper to where the transition begins (negative: under the car)')),
    'bump': (('height', 'Height', 'mm', 0.08, 0.005, 0.5, 0, 5.0, 1000.0, 'Height of the hump'),
             ('length', 'Length', 'm', 0.6, 0.05, 10.0, 2, 0.05, 1.0, 'Length of the hump along the road'),
             ('start', 'Distance', 'm', 2.0, -8.0, 200.0, 2, 0.1, 1.0,
              'From the bumper to the hump\'s near edge (negative: under the car)')),
    'kerb': (('height', 'Height', 'mm', 0.12, 0.01, 0.5, 0, 5.0, 1000.0, 'Height of the kerb\'s face'),
             ('offset', 'Distance', 'm', 0.5, -1.0, 50.0, 2, 0.05, 1.0,
              'From the body\'s side (or the bumper, for a kerb across the road) to the kerb\'s face')),
    'crown': (('crown', 'Cross fall', '%', 2.5, 0.0, 15.0, 1, 0.5, 1.0,
               'The road falls this much per 100 m to both sides of the crown line'),
              ('offset', 'Crown line', 'm', 0.0, -10.0, 10.0, 2, 0.1, 1.0,
               'Where the crown line runs: metres left (+) or right (-) of the car\'s centre line')),
}
CHOICE = {'ramp': ('where', WHERE, 'ahead'), 'crest': ('where', WHERE, 'ahead'), 'garage': ('where', WHERE, 'ahead'),
          'bump': ('where', WHERE, 'ahead'), 'kerb': ('side', SIDES, 'right')}
TYPICAL = {
    'ramp': 'Example: an 8 % grade 3 m ahead. A plain change of grade, no transition.',
    'crest': 'Example: an urban crest, a vertical curve of a few hundred metres into a falling grade.',
    'garage': 'Example: a 15 % ramp down with a 2.4 m transition at half the grade, a common layout for ramps '
              'steeper than about 10 %.',
    'bump': 'Typical car-park speed bump: 50-100 mm high, 0.3-0.9 m long; a road hump is 75-100 mm over 3.7 m.',
    'kerb': 'Typical kerb upstand: 100-150 mm.',
    'crown': 'Typical cross fall of an asphalt road: 2-2.5 %.',
}
# braking, accelerating, cornering: (name, longitudinal g (+ accelerating), lateral g (+ turning left)), typical
MOTIONS = (('At rest', 0.0, 0.0), ('Hard braking', -0.8, 0.0), ('Normal braking', -0.3, 0.0),
           ('Full acceleration', 0.3, 0.0), ('Cornering left', 0.0, 0.5), ('Cornering right', 0.0, -0.5),
           ('Braking in a bend', -0.4, 0.4))
MOTION_NOTE = ('Typical accelerations: an emergency stop with ABS on dry asphalt reaches about 0.8-1.0 g, everyday '
               'braking stays under about 0.3 g, a compact car accelerates at about 0.3 g (0-100 km/h in 10 s), '
               'brisk cornering 0.5 g (everyday driving stays under about 0.3-0.4 g).')
LIMITS = {'long_g': (-1.5, 1.0), 'lat_g': (-1.5, 1.5), 'range_m': (10.0, 200.0), 'min_px': (0.5, 50.0)}
ASSUMPTIONS = ('The road is a height field of the primitives added together; the car stands on it on four tyres (a '
               'circle of the wheel radius with a flat tread), its suspension taking any twist (the static model of '
               'the load cases, at kerb). A point is seen when it is in the image, the body, rig and housings do not '
               'hide it, it faces the camera and the ray reaches it above the road (a ray march, 2 cm steps near the '
               'road). The coverage analysis uses the road; the load cases and body motion stay on flat ground. No '
               'road texture, markings or wetness; a primitive is seen when its height spans the minimum pixels in '
               'one image (contrast is not modelled). Body clearance uses the body\'s underside (wheels excluded, '
               'no rig parts) and the camera and sensor positions; the tyres do not deform around an edge. Typical '
               'figures are typical, not this vehicle\'s or road\'s.')


# =========================================================================== settings
def _num(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def default_feature(kind: str) -> dict:
    d = {'kind': kind}
    for key, _, _, dflt, *_ in PARAMS[kind]:
        d[key] = dflt
    if kind in CHOICE:
        k, _, dflt = CHOICE[kind]
        d[k] = dflt
    return d


def clean_feature(d) -> dict | None:
    """A road primitive as stored: its kind, every parameter (missing ones typical, numbers clamped) and its side;
    None when it is not one."""
    if not isinstance(d, dict):
        return None
    kind = str(d.get('kind', '')).strip().lower()
    if kind not in PARAMS:
        return None
    out = default_feature(kind)
    for key, _, _, _, lo, hi, *_ in PARAMS[kind]:
        v = _num(d.get(key))
        if v is not None:
            out[key] = min(max(v, lo), hi)
    if kind in CHOICE:
        k, opts, _ = CHOICE[kind]
        v = str(d.get(k, '')).strip().lower()
        if v in opts:
            out[k] = v
    name = d.get('name')
    if isinstance(name, str) and name.strip():
        out['name'] = name.strip()[:60]
    return out


@dataclass
class TerrainSettings:
    enabled: bool = True                         # use the road in the coverage analysis (off: flat ground)
    features: list = field(default_factory=list)  # clean_feature dicts
    long_g: float = -0.5                         # the slider case: + accelerating, - braking
    lat_g: float = 0.0                           # + turning left (the body rolls right side down)
    load_case: str = ''                          # suspension case carried while driving ('' = the calibration load)
    dynamics: SU.Dynamics = field(default_factory=SU.Dynamics)
    range_m: float = 60.0                        # how far ahead and behind the road is checked
    min_px: float = 3.0                          # a bump or kerb counts as seen when its height spans this many px

    @property
    def active(self) -> bool:
        return self.enabled and bool(self.features)

    def to_dict(self):
        return {'enabled': self.enabled, 'features': [dict(f) for f in self.features],
                'motion': dict(self.dynamics.to_dict(), long_g=self.long_g, lat_g=self.lat_g,
                               load_case=self.load_case),
                'range_m': self.range_m, 'min_px': self.min_px}

    @staticmethod
    def from_dict(d) -> 'TerrainSettings':
        """Tolerant: junk is skipped, numbers clamped to LIMITS (and each primitive's own limits)."""
        s = TerrainSettings()
        d = d if isinstance(d, dict) else {}
        v = d.get('enabled')
        if isinstance(v, (bool, int)):
            s.enabled = bool(v)
        elif isinstance(v, str):
            s.enabled = v.strip().lower() in ('1', 'true', 'yes', 'on')
        if isinstance(d.get('features'), list):
            s.features = [f for f in (clean_feature(x) for x in d['features'][:MAX_FEATURES * 4])
                          if f is not None][:MAX_FEATURES]
        m = d.get('motion') if isinstance(d.get('motion'), dict) else {}
        s.dynamics = SU.Dynamics.from_dict(m)
        for k in ('long_g', 'lat_g'):
            v = _num(m.get(k))
            if v is not None:
                lo, hi = LIMITS[k]
                setattr(s, k, min(max(v, lo), hi))
        if isinstance(m.get('load_case'), str):
            s.load_case = m['load_case'].strip()
        for k in ('range_m', 'min_px'):
            v = _num(d.get(k))
            if v is not None:
                lo, hi = LIMITS[k]
                setattr(s, k, min(max(v, lo), hi))
        return s


def settings_of(project) -> TerrainSettings:
    return TerrainSettings.from_dict(getattr(project.analysis, 'terrain', None))


def describe(f: dict) -> str:
    """A primitive in a few words: 'Speed bump 80 mm × 0.6 m, 2.0 m ahead'."""
    k = f['kind']
    if k == 'ramp':
        return f'Ramp {f["grade"]:+g} %, {f["start"]:g} m {f["where"]}'
    if k == 'crest':
        word = 'Crest' if f['grade'] < 0 else 'Sag'
        return f'{word} R {f["radius"]:g} m into {f["grade"]:+g} %, {f["start"]:g} m {f["where"]}'
    if k == 'garage':
        return f'Garage ramp {f["grade"]:+g} % ({f["transition"]:g} m transition), {f["start"]:g} m {f["where"]}'
    if k == 'bump':
        return f'Speed bump {f["height"] * 1000:.0f} mm × {f["length"]:g} m, {f["start"]:g} m {f["where"]}'
    if k == 'kerb':
        where = {'left': 'left of the car', 'right': 'right of the car', 'ahead': 'ahead', 'behind': 'behind'}
        return f'Kerb {f["height"] * 1000:.0f} mm, {f["offset"]:g} m {where[f["side"]]}'
    side = 'left' if f['offset'] > 0 else 'right'
    return f'Crowned road {f["crown"]:g} %' + (f', crown {abs(f["offset"]):g} m {side}' if f['offset'] else '')


# =========================================================================== the height field
def references(project) -> dict:
    """Where distances are measured from: the bumpers (x) and the body's sides (y), m."""
    g = SU.vehicle_geometry(project)
    dims = g['dims'] or {}
    xs = g['x']
    xf = dims.get('front_x', (xs['FL'] + xs['FR']) / 2 + 0.85)
    xr = dims.get('rear_x', (xs['RL'] + xs['RR']) / 2 - 0.8)
    w = dims.get('Width', 1.75)
    return {'front_x': float(xf), 'rear_x': float(xr), 'left_y': float(w) / 2, 'right_y': -float(w) / 2,
            'width': float(w)}


class Feature:
    """One primitive placed on the vehicle: a profile h(s) of the distance s across its own direction."""

    def __init__(self, d: dict, ref: dict):
        self.d = d
        self.kind = k = d['kind']
        self.abs = k == 'crown'
        if k == 'crown':
            self.axis, self.sign, self.origin = 1, 1.0, float(d['offset'])
        elif k == 'kerb' and d['side'] in ('left', 'right'):
            self.axis = 1
            if d['side'] == 'left':
                self.sign, self.origin = 1.0, ref['left_y'] + d['offset']
            else:
                self.sign, self.origin = -1.0, ref['right_y'] - d['offset']
        else:
            where = d['side'] if k == 'kerb' else d['where']
            dist = d['offset'] if k == 'kerb' else d['start']
            self.axis = 0
            if where == 'ahead':
                self.sign, self.origin = 1.0, ref['front_x'] + dist
            else:
                self.sign, self.origin = -1.0, ref['rear_x'] - dist

    @property
    def along(self) -> bool:
        """Across the road ahead or behind (a function of x): the car drives over it."""
        return self.axis == 0

    def q(self, x, y):
        """The signed distance across the primitive's direction (linear in x and y)."""
        return self.sign * (np.asarray(x if self.axis == 0 else y, float) - self.origin)

    def s(self, x, y):
        v = self.q(x, y)
        return np.abs(v) if self.abs else v

    def highest(self, qa, qb):
        """The highest the profile gets between the signed distances qa and qb (arrays): every profile is monotone or
        has one peak (a bump's crest, a crown's line), so the ends and the peak bound it exactly."""
        lo, hi = np.minimum(qa, qb), np.maximum(qa, qb)
        f = (lambda q: self.profile(np.abs(q))[0]) if self.abs else (lambda q: self.profile(q)[0])
        top = np.maximum(f(lo), f(hi))
        peak = 0.0 if self.abs else (self.d['length'] / 2 if self.kind == 'bump' else None)
        if peak is not None:
            inside = (lo <= peak) & (peak <= hi)
            top = np.where(inside, np.maximum(top, f(np.full_like(lo, peak))), top)
        return top

    def ds(self, x, y):
        """(ds/dx, ds/dy)."""
        c = np.asarray(x if self.axis == 0 else y, float)
        g = self.sign * (np.sign(c - self.origin) if self.abs else np.ones_like(c))
        z = np.zeros_like(c)
        return (g, z) if self.axis == 0 else (z, g)

    def profile(self, s):
        """(h, dh/ds) at s (m)."""
        d, k = self.d, self.kind
        s = np.asarray(s, float)
        zero = np.zeros_like(s)
        if k == 'ramp':
            g = d['grade'] / 100.0
            on = s > 0
            return np.where(on, g * s, 0.0), np.where(on, g, 0.0)
        if k == 'crest':
            g, R = d['grade'] / 100.0, d['radius']
            if g == 0:
                return zero, zero
            sg, Lc = math.copysign(1.0, g), R * abs(g)
            cur = (s > 0) & (s < Lc)
            past = s >= Lc
            h = np.where(cur, sg * s * s / (2 * R), np.where(past, sg * Lc * Lc / (2 * R) + g * (s - Lc), 0.0))
            return h, np.where(cur, sg * s / R, np.where(past, g, 0.0))
        if k == 'garage':
            g, Lt = d['grade'] / 100.0, d['transition']
            tr = (s > 0) & (s < Lt)
            past = s >= Lt
            h = np.where(tr, g / 2 * s, np.where(past, g / 2 * Lt + g * (s - Lt), 0.0))
            return h, np.where(tr, g / 2, np.where(past, g, 0.0))
        if k == 'bump':
            H, L = d['height'], d['length']
            on = (s > 0) & (s < L)
            return (np.where(on, H * np.sin(math.pi * s / L) ** 2, 0.0),
                    np.where(on, H * math.pi / L * np.sin(2 * math.pi * s / L), 0.0))
        if k == 'kerb':
            H = d['height']
            return H * np.clip(s / FACE_M, 0.0, 1.0), np.where((s > 0) & (s < FACE_M), H / FACE_M, 0.0)
        c = d['crown'] / 100.0                                           # crown
        return -c * s, np.full_like(s, -c)

    def breaks(self) -> list:
        """Distances s where the profile changes form (for sampling and meshing)."""
        d, k = self.d, self.kind
        if k == 'crest':
            return [0.0, d['radius'] * abs(d['grade']) / 100.0]
        if k == 'garage':
            return [0.0, d['transition']]
        if k == 'bump':
            return [0.0, d['length'] / 2, d['length']]
        if k == 'kerb':
            return [0.0, FACE_M]
        return [0.0]

    def top_s(self) -> float:
        """Where a bump's or kerb's top is (its crest; a kerb's edge, 3 cm onto it)."""
        return self.d['length'] / 2 if self.kind == 'bump' else FACE_M + 0.03

    def s_range(self, x0, y0, x1, y1):
        c0, c1 = (x0, x1) if self.axis == 0 else (y0, y1)
        a, b = sorted((self.sign * (c0 - self.origin), self.sign * (c1 - self.origin)))
        if not self.abs:
            return a, b
        if a <= 0 <= b:
            return 0.0, max(-a, b)
        return min(abs(a), abs(b)), max(abs(a), abs(b))

    def extremes(self, x0, y0, x1, y1):
        """(min, max) of h over the box."""
        a, b = self.s_range(x0, y0, x1, y1)
        s = np.concatenate([np.linspace(a, b, 513), [v for v in self.breaks() if a <= v <= b]])
        h, _ = self.profile(s)
        return float(h.min()), float(h.max())

    def label(self) -> str:
        return self.d.get('name') or describe(self.d)


class Terrain:
    """The road under the vehicle: the height field of the primitives and the vehicle's stance on it."""

    def __init__(self, features: list, ref: dict, key: str = ''):
        self.spec = [dict(f) for f in features]
        self.ref = ref
        self.features = [Feature(f, ref) for f in self.spec]
        self.key = key
        self.R = self.t = None          # the body's pose on the road (None: as on flat ground)
        self.stance = None              # dict, see stance()

    # ---- the field
    def height(self, x, y) -> np.ndarray:
        x, y = np.asarray(x, float), np.asarray(y, float)
        h = np.zeros(np.broadcast(x, y).shape)
        for f in self.features:
            h += f.profile(f.s(x, y))[0]
        return h

    def gradient(self, x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        gx = np.zeros(np.broadcast(x, y).shape)
        gy = np.zeros_like(gx)
        for f in self.features:
            _, dh = f.profile(f.s(x, y))
            sx, sy = f.ds(x, y)
            gx += dh * sx
            gy += dh * sy
        return gx, gy

    def bounds(self, x0, y0, x1, y1):
        """(lowest, highest) the road can be over the box (bounds: each primitive's extremes added up)."""
        lo = hi = 0.0
        for f in self.features:
            a, b = f.extremes(x0, y0, x1, y1)
            lo, hi = lo + a, hi + b
        return lo, hi

    def pose(self, body_pose=None):
        """The body's pose in the world on this road, a load case's (body_pose, relative to standing on flat ground)
        applied first; None when it stands as on flat ground and nothing moves it."""
        if self.R is None:
            return body_pose
        if body_pose is None:
            return self.R, self.t
        Rb, tb = (np.asarray(v, float) for v in body_pose)
        return self.R @ Rb, self.R @ tb + self.t

    # ---- visibility through the road
    def clear(self, C, P, on_surface=True) -> np.ndarray:
        """Bool (K,): the road neither hides the world points P (K,3) from the world point C (a camera) nor turns its
        back on them. on_surface: the points lie on the road (bool or (K,) array; a point above it, on a pole, has no
        facing test)."""
        P = np.asarray(P, float).reshape(-1, 3)
        C = np.asarray(C, float)
        K = len(P)
        out = np.ones(K, bool)
        if K == 0 or not self.features:
            return out
        v = C[None, :] - P
        gx, gy = self.gradient(P[:, 0], P[:, 1])
        facing = -gx * v[:, 0] - gy * v[:, 1] + v[:, 2] > 0
        on = np.broadcast_to(np.asarray(on_surface, bool), (K,))
        out &= facing | ~on
        top = np.full(K, 1e-3)                  # the highest the road gets under each ray (exact per primitive)
        for f in self.features:
            top += f.highest(np.full(K, float(f.q(C[0], C[1]))), f.q(P[:, 0], P[:, 1]))
        d = -v
        L = np.linalg.norm(d, axis=1)
        dz = d[:, 2]
        with np.errstate(divide='ignore', invalid='ignore'):
            s0 = np.where(dz < -1e-12, (C[2] - top) / -dz, 0.0)        # below `top` from here on
        s0 = np.clip(np.nan_to_num(s0, nan=0.0), 0.0, 1.0)
        s1 = 1.0 - np.clip(0.01 / np.maximum(L, 1e-6), 0.0, 0.5)        # stop 1 cm short of the point itself
        todo = out & (s1 > s0) & (L > 0.02)
        n = np.clip(np.ceil((s1 - s0) * L / STEP_M), 2, MAX_SAMPLES).astype(int)
        lo = 0
        for hi in (4, 8, 16, 32, 64, 128, MAX_SAMPLES):
            idx = np.nonzero(todo & (n > lo) & (n <= hi))[0]
            lo = hi
            chunk = max(1, 400_000 // hi)
            for a in range(0, len(idx), chunk):
                i = idx[a:a + chunk]
                t = s0[i, None] + (s1 - s0)[i, None] * ((np.arange(hi) + 0.5) / hi)[None, :]
                X = C[0] + t * d[i, 0:1]
                Y = C[1] + t * d[i, 1:2]
                Z = C[2] + t * d[i, 2:3]
                H = self.height(X, Y)
                tol = 0.002 + 0.001 * (1.0 - t) * L[i, None]
                out[i] &= ~(Z < H - tol).any(axis=1)
        return out

    def describe(self) -> str:
        return '; '.join(f.label() for f in self.features) or 'flat ground'


# =========================================================================== the vehicle on the road
def _contact_height(ter: Terrain, x, y, r):
    """Height the tyre at (x, y) rests at (m): the highest of the road under its tread, less the circle's rise."""
    dx = np.linspace(-r, r, 61)
    lift = r - np.sqrt(np.maximum(r * r - dx * dx, 0.0))
    dy = np.array([-TREAD_M / 2, 0.0, TREAD_M / 2])
    H = ter.height(x + dx[:, None], y + dy[None, :])
    return float((H - lift[:, None]).max())


def _pose_of(sv, xy):
    """The body's pose from a solve's corner drops, exact for large angles: the drop field w + b x + c y (mm, b and c
    in mm/m) is the plane the body's ground points end up on, so sin(pitch) = b and sin(roll)·cos(pitch) = -c (the
    load cases' small-angle pose takes tan instead: the same below a degree, 0.1° off on a 15 % ramp)."""
    A = np.array([[1.0, xy[c][0], xy[c][1]] for c in SU.CORNERS])
    heave, b, c = np.linalg.lstsq(A, np.array([sv.corners[k].drop_mm for k in SU.CORNERS]), rcond=None)[0]
    pitch = math.asin(min(max(b / 1000.0, -1.0), 1.0))
    roll = -math.asin(min(max(c / 1000.0 / math.cos(pitch), -1.0), 1.0))
    return LC.pose(heave, math.degrees(pitch), math.degrees(roll)), (float(heave), math.degrees(pitch),
                                                                     math.degrees(roll))


def stance(ter: Terrain, project, su=None, xy=None, r=None) -> dict:
    """The car standing on the road at kerb: each tyre's contact height and the body's heave, pitch and roll from the
    suspension (a road that is one plane under the tyres moves the body with it; a twist is taken by the springs).
    Sets ter.R, ter.t (None when it stands as on flat ground)."""
    su = su or SU.settings_for(project)
    xy = xy or SU.corner_positions(project, su)
    r = r or SU.wheel_radius(project)
    R, t = np.eye(3), np.zeros(3)
    g = {c: 0.0 for c in SU.CORNERS}
    contact = dict(g)
    sv = None
    for _ in range(16):                                 # the contacts shift as the body tilts: a few rounds settle
        prev = dict(g)
        for c in SU.CORNERS:
            w = R @ np.array([xy[c][0], xy[c][1], r]) + t       # the wheel centre in the world
            contact[c] = _contact_height(ter, w[0], w[1], r) * 1000.0
            # the tyre rests on the road: its centre one radius above the contact, so the body's ground point under
            # it (the solve's corner) sits r·(1 - cos tilt) higher than the contact
            g[c] = contact[c] + r * (1.0 - R[2, 2]) * 1000.0
        if all(abs(v) < 1e-9 for v in g.values()):
            sv = None
            break
        if sv is not None and max(abs(g[c] - prev[c]) for c in SU.CORNERS) < 1e-7:
            break                                       # settled: the last pose stands on these contacts
        sv = SU.solve(su, xy, [], ground=g)
        (R, t), att = _pose_of(sv, xy)
    if sv is None:
        ter.R = ter.t = None
        ter.stance = {'contact_mm': dict(g), 'heave_mm': 0.0, 'pitch_deg': 0.0, 'roll_deg': 0.0, 'lifted': [],
                      'bump_stops': [], 'flat': True}
        return ter.stance
    ter.R, ter.t = R, t
    ter.stance = {'contact_mm': {c: float(v) for c, v in contact.items()}, 'heave_mm': att[0],
                  'pitch_deg': att[1], 'roll_deg': att[2],
                  'lifted': [c for c in SU.CORNERS if sv.corners[c].lifted], 'bump_stops': sv.bump, 'flat': False,
                  'travel_mm': {c: sv.corners[c].travel_mm for c in SU.CORNERS}}
    return ter.stance


def build(project, settings: TerrainSettings | None = None, features=None) -> Terrain | None:
    """The road of the project (its settings or the features given) with the vehicle standing on it; None when there
    is none (flat ground)."""
    st = settings or settings_of(project)
    feats = st.features if features is None else [f for f in (clean_feature(x) for x in features) if f]
    if not feats:
        return None
    ter = Terrain(feats, references(project), _key(project, feats))
    stance(ter, project)
    return ter


def _key(project, feats) -> str:
    veh = project.vehicle
    m = veh.mesh
    return json.dumps([feats, veh.wheel_centres, id(m), getattr(m, 'n_tris', 0),
                       getattr(project.analysis, 'suspension', None)], sort_keys=True, default=str)


_CACHE: dict = {}
_LOCK = threading.Lock()


def active(project, variant=None) -> Terrain | None:
    """The road the coverage analysis uses: None (flat ground) unless the project has primitives and the road is on.
    Cached on the inputs (the stance is a few small solves)."""
    st = settings_of(project)
    if not st.active:
        return None
    key = _key(project, st.features)
    with _LOCK:
        hit = _CACHE.get(key)
    if hit is not None:
        return hit
    ter = build(project, st)
    with _LOCK:
        if len(_CACHE) > 16:
            _CACHE.pop(next(iter(_CACHE)))
        _CACHE[key] = ter
    return ter


# =========================================================================== clearance
_UNDER: dict = {}


def underside(mesh, cell=0.04, top=0.6) -> np.ndarray:
    """The body's underside as points (N,3): the lowest surface in each cell-sized plan square below `top` metres,
    from triangles sampled densely enough (wheels and ignored parts left out). Cached per mesh."""
    if mesh is None or getattr(mesh, 'n_tris', 0) == 0:
        return np.zeros((0, 3))
    key = (id(mesh), mesh.n_tris, cell, top)
    with _LOCK:
        hit = _UNDER.get(key)
    if hit is not None:
        return hit
    from .mesh import ROLE_GLASS, ROLE_MOUNTABLE
    keep = mesh.part_mask({ROLE_MOUNTABLE, ROLE_GLASS})
    tri = mesh.tri[keep]
    tri = tri[tri[:, :, 2].min(axis=1) < top]
    pts = [tri.reshape(-1, 3)]
    if len(tri):
        edge = np.max(np.linalg.norm(tri - np.roll(tri, 1, axis=1), axis=2), axis=1)
        n = np.clip(np.ceil(edge / cell).astype(int), 1, 64)
        for k in np.unique(n):
            if k < 2:
                continue
            sel = tri[n == k]
            i, j = np.meshgrid(np.arange(k + 1), np.arange(k + 1))
            m = (i + j) <= k
            a, b = i[m] / k, j[m] / k
            P = sel[:, 0, None, :] * (1 - a - b)[None, :, None] + sel[:, 1, None, :] * a[None, :, None] + \
                sel[:, 2, None, :] * b[None, :, None]
            pts.append(P.reshape(-1, 3))
    P = np.concatenate(pts)
    P = P[P[:, 2] < top]
    if len(P):
        ij = np.floor(P[:, :2] / cell).astype(np.int64)
        code = (ij[:, 0] - ij[:, 0].min()) * (int(ij[:, 1].max() - ij[:, 1].min()) + 1) + (ij[:, 1] - ij[:, 1].min())
        order = np.lexsort((P[:, 2], code))
        first = np.ones(len(order), bool)
        first[1:] = code[order][1:] != code[order][:-1]
        P = P[order[first]]
    with _LOCK:
        if len(_UNDER) > 8:
            _UNDER.pop(next(iter(_UNDER)))
        _UNDER[key] = P
    return P


def _where_on_body(x, ref, xy):
    xf = (xy['FL'][0] + xy['FR'][0]) / 2
    xr = (xy['RL'][0] + xy['RR'][0]) / 2
    if x > xf + 0.25:
        return 'front overhang'
    if x < xr - 0.25:
        return 'rear overhang'
    return 'underbody between the axles'


def _devices(project, variant=None):
    """(uid, tag, kind, position) of every visible camera and sensor with a usable position."""
    v = variant or project.variant
    out = [(c.uid, c.tag, 'camera', c.pos()) for c in v.cameras if c.visible]
    out += [(s.uid, s.tag, s.kind, s.pos()) for s in v.sensors if s.visible]
    return [d for d in out if np.all(np.isfinite(np.asarray(d[3], float)))]


def clearance(ter: Terrain, project, variant=None, under=None, devices=True) -> dict:
    """At the car's stance: the smallest gap between the body's underside and the road (m, negative = it would cut
    into the road) and where, and (devices) each camera's and sensor's height above the road under it."""
    xy = SU.corner_positions(project, SU.settings_for(project))
    P = underside(project.vehicle.mesh) if under is None else under
    R, t = (ter.R, ter.t) if ter.R is not None else (np.eye(3), np.zeros(3))
    out = {'body_m': None, 'body_at': None, 'body_where': '', 'devices': []}
    if len(P):
        W = P @ R.T + t
        gap = W[:, 2] - ter.height(W[:, 0], W[:, 1])
        k = int(np.argmin(gap))
        out.update(body_m=float(gap[k]), body_at=[float(v) for v in P[k]],
                   body_where=_where_on_body(P[k][0], ter.ref, xy))
    for uid, tag, kind, p in (_devices(project, variant) if devices else []):
        w = R @ np.asarray(p, float) + t
        out['devices'].append({'uid': uid, 'tag': tag, 'kind': kind,
                               'clear_m': float(w[2] - ter.height(w[0], w[1]))})
    return out


_DRIVE: dict = {}


def _drive_steps(ter: Terrain, project, i: int, step=0.1) -> dict:
    """The car standing on the road at each step of driving over primitive i, and its body's clearance on the way
    (cached per road: a camera moved does not repeat this)."""
    ck = (ter.key, i, step) if ter.key else None
    with _LOCK:
        hit = _DRIVE.get(ck) if ck is not None else None
    if hit is not None:
        return hit
    f = ter.features[i]
    key = 'offset' if f.kind == 'kerb' else 'start'
    ref = ter.ref
    length = ref['front_x'] - ref['rear_x']
    extra = f.d['length'] if f.kind == 'bump' else 0.5
    under = underside(project.vehicle.mesh)
    su = SU.settings_for(project)
    xy = SU.corner_positions(project, su)
    r = SU.wheel_radius(project)
    out = {'steps': [], 'body_m': math.inf, 'body_where': '', 'body_at_m': None, 'lifted_at_m': None}
    for d in np.arange(0.5, -(length + extra + 0.5) - 1e-9, -step):
        g = dict(f.d)
        g[key] = float(d)
        t = Terrain(ter.spec[:i] + [g] + ter.spec[i + 1:], ref)
        sv = stance(t, project, su, xy, r)
        out['steps'].append((float(d), t))
        if sv['lifted'] and out['lifted_at_m'] is None:
            out['lifted_at_m'] = float(d)
        cl = clearance(t, project, None, under, devices=False)
        if cl['body_m'] is not None and cl['body_m'] < out['body_m']:
            out.update(body_m=cl['body_m'], body_where=cl['body_where'], body_at_m=float(d))
    if not math.isfinite(out['body_m']):
        out['body_m'] = None
    if ck is not None:
        with _LOCK:
            if len(_DRIVE) > 32:
                _DRIVE.pop(next(iter(_DRIVE)))
            _DRIVE[ck] = out
    return out


def drive_over(ter: Terrain, project, i: int, variant=None, step=0.1) -> dict:
    """Drive over primitive i (one across the road): it moves from just beyond the bumper to past the far end of the
    car while the car stands on the road at each step; the smallest body and device clearance met on the way and where
    ('body_at_m', 'device_at_m': the primitive's distance from the bumper then; negative = under the car)."""
    base = _drive_steps(ter, project, i, step)
    out = {k: v for k, v in base.items() if k != 'steps'}
    out.update(device_m=None, device='', device_at_m=None)
    devs = _devices(project, variant)
    if devs:
        P = np.array([p for *_, p in devs], float)
        for d, t in base['steps']:
            R, tt = (t.R, t.t) if t.R is not None else (np.eye(3), np.zeros(3))
            W = P @ R.T + tt
            gap = W[:, 2] - t.height(W[:, 0], W[:, 1])
            k = int(np.argmin(gap))
            if out['device_m'] is None or gap[k] < out['device_m']:
                out.update(device_m=float(gap[k]), device=devs[k][1], device_at_m=d)
    return out


# =========================================================================== seeing points on the road
class Views:
    """A variant's visible cameras as the coverage analysis sees them (lens, occlusion cube map): world points on
    the road in, per camera visibility (image, body, the road), density and image positions out."""

    def __init__(self, project, engine, variant=None):
        from .lens import resolve
        v = variant or project.variant
        vg = engine.vehicle_geom(project, v)
        self.cams = []
        for c in v.cameras:
            if not c.visible:
                continue
            self.cams.append((c, resolve(project.module_for(c), project.lens_for(c)), engine.cube(c.pos(), vg)))

    def see(self, Pw, ter: Terrain | None, on_surface=True, visibility=True):
        """{uid: (vis, dens px/cm, u, v)} for world points Pw (K,3), the body standing on the road ter (None: flat
        ground). visibility=False: only where they land in the images (u, v), with the body's pose."""
        from .analysis import camera_frames, eval_points
        Pw = np.asarray(Pw, float).reshape(-1, 3)
        pose = ter.pose() if ter is not None else None
        Pb = Pw if pose is None else (Pw - pose[1]) @ pose[0]
        out = {}
        for c, intr, cube in self.cams:
            _, W2O = camera_frames(c)
            u, vv, _, _ = intr.project_dirs((Pb - c.pos()) @ W2O.T)
            if not visibility:
                out[c.uid] = (None, None, u, vv)
                continue
            vis, dens, _, _ = eval_points(Pb, c, intr, cube)
            if ter is not None:
                C = c.pos() if pose is None else pose[0] @ c.pos() + pose[1]
                vis = vis & ter.clear(C, Pw, on_surface)
                dens = np.where(vis, dens, 0.0)
            out[c.uid] = (vis, dens, u, vv)
        return out


def _runs(d, mask, min_len=0.0):
    """Intervals [(from, to)] of the sample distances d where mask holds (consecutive samples), at least min_len."""
    out = []
    start = None
    for k in range(len(d)):
        if mask[k] and start is None:
            start = k
        if start is not None and (not mask[k] or k == len(d) - 1):
            end = k if mask[k] else k - 1
            if d[end] - d[start] >= min_len:
                out.append((float(d[start]), float(d[end])))
            start = None
    return out


def profile(ter: Terrain | None, views: Views, ref: dict, direction='ahead', range_m=60.0, y=0.0) -> dict:
    """Along the road's centre line ahead of the front bumper (or behind the rear one): the road's height and where
    some camera sees it, on this road and on flat ground; the stretches the road itself hides."""
    d = np.unique(np.round(np.r_[np.arange(0.0, 10.0, 0.1), np.arange(10.0, range_m + 1e-9, 0.25)], 4))
    x = ref['front_x'] + d if direction == 'ahead' else ref['rear_x'] - d
    Y = np.full_like(x, y)
    h = ter.height(x, Y) if ter is not None else np.zeros_like(x)
    Pw = np.stack([x, Y, h], 1)
    flat = views.see(np.stack([x, Y, np.zeros_like(x)], 1), None)
    road = views.see(Pw, ter) if ter is not None else flat
    seen = np.zeros(len(d), bool)
    seen_flat = np.zeros(len(d), bool)
    dens = np.zeros(len(d))
    per = {}
    for c, _, _ in views.cams:
        vr, dr, _, _ = road[c.uid]
        vf = flat[c.uid][0]
        seen |= vr
        seen_flat |= vf
        dens = np.maximum(dens, dr)
        per[c.tag] = {'seen_to_m': float(d[vr].max()) if vr.any() else None,
                      'near_m': float(d[vr].min()) if vr.any() else None}
    hidden = seen_flat & ~seen
    return {'direction': direction, 'd': d, 'h': h, 'seen': seen, 'seen_flat': seen_flat, 'density': dens,
            'near_m': float(d[seen].min()) if seen.any() else None,
            'near_flat_m': float(d[seen_flat].min()) if seen_flat.any() else None,
            'seen_to_m': _seen_to(d, seen), 'seen_to_flat_m': _seen_to(d, seen_flat),
            'hidden': _runs(d, hidden, 0.2), 'per_camera': per, 'range_m': float(range_m)}


def _seen_to(d, seen, gap=GAP_M):
    """How far the road stays in view from its nearest seen point: up to the first stretch out of view longer than
    about gap (a bump's short shadow does not end it; those stretches are listed as hidden)."""
    if not seen.any():
        return None
    ds = d[seen]
    jumps = np.nonzero(np.diff(ds) > gap)[0]
    return float(ds[jumps[0]]) if len(jumps) else float(ds[-1])


def sweep(ter: Terrain, views: Views, i: int, project, min_px=3.0, range_m=60.0) -> dict:
    """A speed bump or kerb moved towards the car: at each distance, which camera sees its top (in the image, not
    hidden by the body or the road) with its height spanning at least min_px pixels. 'first_seen_m': the farthest
    distance it is seen at (approaching, where it comes into view), 'lost_m': the nearest (closer, it is out of view),
    'gaps': distances in between it is not seen at."""
    f = ter.features[i]
    key = 'offset' if f.kind == 'kerb' else 'start'
    across = f.along
    ref = ter.ref
    if across:
        ds = np.unique(np.round(np.r_[np.arange(0.0, 5.0, 0.25), np.arange(5.0, 20.0, 1.0),
                                      np.arange(20.0, range_m + 1e-9, 5.0)], 4))
        lat = np.linspace(-ref['width'] / 2, ref['width'] / 2, 9)
    else:
        ds = np.unique(np.round(np.r_[np.arange(0.0, 3.0, 0.1), np.arange(3.0, 8.0 + 1e-9, 0.5)], 4))
        lat = np.linspace(ref['rear_x'], ref['front_x'], 9)
    su = SU.settings_for(project)
    xy = SU.corner_positions(project, su)
    r = SU.wheel_radius(project)
    rows = []
    for dist in ds:
        g = dict(f.d)
        g[key] = float(dist)
        t = Terrain(ter.spec[:i] + [g] + ter.spec[i + 1:], ref)
        stance(t, project, su, xy, r)
        ft = t.features[i]
        st = ft.top_s()
        c = ft.origin + ft.sign * st
        x, y = (np.full_like(lat, c), lat) if across else (lat, np.full_like(lat, c))
        h = t.height(x, y)
        top = np.stack([x, y, h], 1)
        base = np.stack([x, y, h - ft.profile(np.full_like(lat, st))[0]], 1)
        a = views.see(top, t)
        b = views.see(base, t, visibility=False)
        best = (None, 0.0, 0.0)
        for cam, _, _ in views.cams:
            vis, _, u, v = a[cam.uid]
            _, _, ub, vb = b[cam.uid]
            px = np.hypot(u - ub, v - vb)
            ok = vis & np.isfinite(px) & (px >= min_px)
            share = float(ok.mean())
            if share >= 0.5 and (share, float(np.median(px[ok]))) > best[1:]:
                best = (cam.tag, share, float(np.median(px[ok])))
        rows.append((float(dist), best[0], best[2]))
    seen = np.array([r_[1] is not None for r_ in rows])
    dd = np.array([r_[0] for r_ in rows])
    out = {'index': i, 'kind': f.kind, 'label': f.label(), 'distances': rows, 'first_seen_m': None, 'by': None,
           'lost_m': None, 'lost_by': None, 'gaps': [], 'across': across, 'placed_m': float(f.d[key])}
    if seen.any():
        k_far = int(np.nonzero(seen)[0].max())
        k_near = int(np.nonzero(seen)[0].min())
        out.update(first_seen_m=rows[k_far][0], by=rows[k_far][1], lost_m=rows[k_near][0], lost_by=rows[k_near][1],
                   gaps=_runs(dd, ~seen & (dd > rows[k_near][0]) & (dd < rows[k_far][0])))
    return out


# =========================================================================== body motion
def motion_cases(project, settings: TerrainSettings | None = None, variant=None) -> dict:
    """The body's attitude in every motion case (MOTIONS and the slider case) with the driving load on board, relative
    to the load the cameras are calibrated at, and every camera's horizon move: {'cases': [...], 'info': {...}}.
    Cheap (no ground analysis)."""
    p = SU._project_for(project, variant)
    st = settings or settings_of(p)
    su = SU.settings_for(p)
    xy = SU.corner_positions(p, su)
    comps = SU.rig_components(p, su)
    cal = su.case(su.calibrated_at)
    cal_stat = SU.solve(su, xy, SU.case_loads(p, su, cal, comps)) if cal is not None else SU.solve(su, xy, [])
    drive = su.case(st.load_case) or cal
    loads = SU.case_loads(p, su, drive, comps) if drive is not None else []
    cg = SU.cg_height_estimate(p)
    r = SU.wheel_radius(p)
    R0, t0 = LC.pose(cal_stat.heave_mm, cal_stat.pitch_deg, cal_stat.roll_deg)
    ref = (cal_stat.heave_mm, cal_stat.pitch_deg, cal_stat.roll_deg) if cal is not None else None
    cases = list(MOTIONS) + [('Sliders', st.long_g, st.lat_g)]
    out = []
    tr0 = None
    for name, ax, ay in cases:
        sv, tr = SU.dynamic_solve(su, xy, loads, ax, ay, st.dynamics, cg, r)
        tr0 = tr0 or tr
        R, t = LC.pose(sv.heave_mm, sv.pitch_deg, sv.roll_deg)
        dR, _ = LC.relative(R, t, R0, t0)
        _, rp, rr = euler_zyx(dR)
        hz = LC.horizon_moves(p, R, R0)
        out.append({'name': name, 'long_g': float(ax), 'lat_g': float(ay), 'statics': sv,
                    'heave_mm': sv.heave_mm, 'pitch_deg': sv.pitch_deg, 'roll_deg': sv.roll_deg,
                    'rel': {'heave_mm': sv.heave_mm - cal_stat.heave_mm, 'pitch_deg': rp, 'roll_deg': rr},
                    'horizon': hz, 'devices': LC.device_moves(p, R, t, R0, t0),
                    'corner': LC.CornerCase(name, sv.heave_mm, sv.pitch_deg, sv.roll_deg,
                                            {k: sv.corners[k].drop_mm for k in SU.CORNERS}, ref,
                                            cal.name if cal is not None else 'kerb', sv.bump),
                    'bump_stops': sv.bump, 'lifted': [k for k in SU.CORNERS if sv.corners[k].lifted],
                    'image': None})
    dyn = st.dynamics
    info = {'calibrated_at': cal.name if cal is not None else 'kerb (empty, as modelled)',
            'driving_load': drive.name if drive is not None else 'kerb (empty, as modelled)',
            'mass_kg': tr0['mass_kg'], 'cog_m': list(tr0['cog']), 'cg_height_m': tr0['cog'][2],
            'cg_estimated': dyn.cg_height <= 0, 'kerb_cg_height_m': dyn.cg_height if dyn.cg_height > 0 else cg,
            'roll_axis_m': tr0['roll_axis_z'], 'wheel_radius_m': r, 'typical_suspension': su.is_typical,
            'dynamics': dyn.to_dict()}
    return {'cases': out, 'info': info}


# =========================================================================== the study
@dataclass
class TerrainResult:
    settings: TerrainSettings
    features: list                  # clean dicts, in the settings' order
    road: bool                      # a road was studied (primitives present and switched on)
    stance: dict | None
    coverage: dict | None           # flat vs this road
    per_camera: list
    profiles: dict                  # 'ahead' / 'behind' -> profile()
    sweeps: list
    clearance: dict | None
    drive_over: list                # per primitive across the road
    motion: dict                    # motion_cases() with the ground figures
    findings: list
    variant: str = ''
    variant_uid: str = ''
    seconds: float = 0.0
    flat_result: object = None      # AnalysisResult on flat ground (for the page; not exported)
    road_result: object = None      # AnalysisResult on this road

    def worst(self) -> str:
        return self.findings[0]['severity'] if self.findings else 'good'


def _coverage(flat, road) -> tuple[dict, list]:
    m0, m1 = flat.metrics, road.metrics
    a = flat.grid.cell ** 2
    keys = ('blind_m2', 'covered_pct', 'multi_pct', 'min_density', 'in_focus_pct', 'min_ok_pct', 'target_ok_pct')
    roi = road.grid.roi
    cov0, cov1 = roi & (flat.count > 0), roi & (road.count > 0)
    out = {'flat': {k: m0[k] for k in keys}, 'road': {k: m1[k] for k in keys},
           'mean_density_flat': float(flat.best_dens[cov0].mean()) if cov0.any() else 0.0,
           'mean_density_road': float(road.best_dens[cov1].mean()) if cov1.any() else 0.0,
           'lost_m2': float((cov0 & ~cov1).sum() * a), 'gained_m2': float((cov1 & ~cov0).sum() * a),
           'roi_m2': m1['roi_area']}
    per = []
    for ce0 in flat.cams:
        k = road.cam_index(ce0.uid)
        if k < 0:
            continue
        ce1 = road.cams[k]
        s0, s1 = roi & ce0.vis, roi & ce1.vis
        per.append({'uid': ce0.uid, 'tag': ce0.tag, 'slot': ce0.slot, 'seen_flat_m2': float(s0.sum() * a),
                    'seen_road_m2': float(s1.sum() * a), 'hidden_m2': float((s0 & ~s1).sum() * a),
                    'mean_density_flat': float(ce0.dens[s0].mean()) if s0.any() else 0.0,
                    'mean_density_road': float(ce1.dens[s1].mean()) if s1.any() else 0.0})
    return out, per


def run(project, engine=None, settings: TerrainSettings | None = None, variant=None, progress=None,
        cancel=None, cell=None) -> TerrainResult | None:
    """The whole study: the road (coverage against flat ground, the road ahead and behind, the bumps' and kerbs'
    first-seen distances, clearance) and the body motion cases with their ground figures. None when cancelled."""
    t0 = time.perf_counter()
    p = SU._project_for(project, variant)
    st = settings or settings_of(p)
    v = p.variant
    prog = progress or (lambda f: None)
    stop = cancel or (lambda: False)
    if engine is None:
        from .analysis import Engine
        engine = Engine()
    has_car = p.vehicle.mesh is not None and getattr(p.vehicle.mesh, 'n_tris', 0) > 0
    cams = [c for c in v.cameras if c.visible]
    cell = cell or max(float(p.analysis.cell_m), 0.05)
    ter = build(p, st) if (st.active and has_car) else None
    res = TerrainResult(st, [dict(f) for f in st.features], ter is not None, None, None, [], {}, [], None, [],
                        {'cases': [], 'info': {}}, [], v.name, v.uid)
    # ---- the road
    if ter is not None:
        res.stance = dict(ter.stance)
        res.clearance = clearance(ter, p, v)
        prog(0.03)
        if cams:
            flat = engine.analyse(p, variant=v, cell=cell, terrain=False)
            if stop():
                return None
            road = engine.analyse(p, variant=v, cell=cell, terrain=ter)
            if stop():
                return None
            if flat is not None and road is not None:
                res.coverage, res.per_camera = _coverage(flat, road)
                res.flat_result, res.road_result = flat, road
            prog(0.25)
            views = Views(p, engine, v)
            for k, direction in enumerate(('ahead', 'behind')):
                res.profiles[direction] = profile(ter, views, ter.ref, direction, st.range_m)
                if stop():
                    return None
            prog(0.35)
            idx = [i for i, f in enumerate(ter.features) if f.kind in ('bump', 'kerb')]
            for n, i in enumerate(idx):
                res.sweeps.append(sweep(ter, views, i, p, st.min_px, st.range_m))
                if stop():
                    return None
                prog(0.35 + 0.15 * (n + 1) / max(len(idx), 1))
        for i, f in enumerate(ter.features):
            if f.along:
                res.drive_over.append(dict(drive_over(ter, p, i, v), index=i, label=f.label()))
                if stop():
                    return None
        prog(0.55)
    # ---- body motion
    res.motion = motion_cases(p, st)
    if has_car and cams:
        lr = LC.evaluate(engine, p, [c['corner'] for c in res.motion['cases']], cell=cell, cancel=cancel,
                         progress=lambda f: prog(0.55 + 0.45 * f))
        if lr is None:
            return None
        for c, r in zip(res.motion['cases'], lr):
            c['image'] = r
    res.findings = _findings(res, p)
    res.seconds = time.perf_counter() - t0
    prog(1.0)
    return res


# =========================================================================== findings and issues
def _finding(out, sev, kind, message, uids=()):
    out.append({'severity': sev, 'kind': kind, 'message': message, 'uids': list(uids)})


def _m(v, nd=1):
    return '—' if v is None else f'{v:.{nd}f} m'


def _findings(res: TerrainResult, project) -> list:
    out = []
    st = res.settings
    if not st.features:
        _finding(out, 'note', 'flat', 'Flat ground: add a ramp, crest, speed bump, kerb, garage ramp or crowned road '
                                      'to see what the road changes.')
    elif not st.enabled:
        _finding(out, 'note', 'off', 'The road is switched off: the coverage analysis uses flat ground.')
    if res.road:
        s = res.stance or {}
        if s.get('lifted'):
            _finding(out, 'warning', 'lift', f'Standing on this road the {", ".join(s["lifted"])} wheel would lift '
                                              '(no load left on it); the body attitude assumes four wheels on the '
                                              'ground.')
        cl = res.clearance or {}
        if cl.get('body_m') is not None and cl['body_m'] < 0:
            _finding(out, 'serious', 'touch', f'The body would cut {-cl["body_m"] * 100:.0f} cm into the road where it '
                                              f'stands ({cl["body_where"]}).')
        for dv in cl.get('devices', []):
            if dv['clear_m'] < 0.05:
                _finding(out, 'serious' if dv['clear_m'] < 0 else 'warning', 'device',
                         f'{dv["tag"]} is {dv["clear_m"] * 100:.0f} cm above the road where the car stands.',
                         [dv['uid']])
        for dr in res.drive_over:
            if dr['body_m'] is not None and dr['body_m'] < 0:
                _finding(out, 'serious', 'touch', f'Driving over "{dr["label"]}" the body would hit the road: it '
                                                  f'would reach {-dr["body_m"] * 100:.0f} cm into it at the '
                                                  f'{dr["body_where"]} (with the primitive {dr["body_at_m"]:.1f} m '
                                                  'from the bumper).')
            elif dr['body_m'] is not None and dr['body_m'] < 0.03:
                _finding(out, 'warning', 'touch', f'Driving over "{dr["label"]}" the body clears the road by only '
                                                  f'{dr["body_m"] * 100:.1f} cm ({dr["body_where"]}).')
            if dr['device_m'] is not None and dr['device_m'] < 0.05:
                dev = next((d for d in _devices(project) if d[1] == dr['device']), None)
                _finding(out, 'serious' if dr['device_m'] < 0 else 'warning', 'device',
                         f'Driving over "{dr["label"]}" {dr["device"]} comes within {dr["device_m"] * 100:.0f} cm of '
                         'the road.', [dev[0]] if dev else [])
        cov = res.coverage
        if cov is not None:
            d = cov['road']['blind_m2'] - cov['flat']['blind_m2']
            if d > 0.05:
                _finding(out, 'warning', 'blind', f'On this road the blind ground grows by {d:.2f} m² (to '
                                                  f'{cov["road"]["blind_m2"]:.2f} m²): {cov["lost_m2"]:.2f} m² that '
                                                  'flat ground shows is hidden or out of view.')
            else:
                _finding(out, 'good', 'blind', f'Blind ground on this road {cov["road"]["blind_m2"]:.2f} m² '
                                               f'(flat: {cov["flat"]["blind_m2"]:.2f} m²); mean density '
                                               f'{cov["mean_density_road"]:.1f} px/cm (flat '
                                               f'{cov["mean_density_flat"]:.1f}).')
        for direction, pr in res.profiles.items():
            for a, b in pr['hidden']:
                end = f'beyond {a:.1f} m (to the end of the check, {pr["range_m"]:g} m)' \
                    if b >= pr['range_m'] - 0.3 else f'from {a:.1f} to {b:.1f} m'
                _finding(out, 'note', 'hidden', f'The road itself hides the centre line {direction} {end}: no '
                                                f'camera sees it there; on flat ground they do.')
        for sw in res.sweeps:
            if sw['first_seen_m'] is None:
                _finding(out, 'serious', 'unseen', f'"{sw["label"]}": no camera resolves it (its height never spans '
                                                   f'{st.min_px:g} px with its top in view) at any distance checked.')
            else:
                far = sw['distances'][-1][0]
                first = (f'from {sw["first_seen_m"]:.1f} m' if sw['first_seen_m'] < far - 1e-6 else
                         f'from at least {far:.0f} m')
                gap = ''
                if sw['gaps']:
                    gap = '; hidden at ' + ', '.join(f'{a:.1f} m' if a == b else f'{a:.1f}-{b:.1f} m'
                                                     for a, b in sw['gaps'])
                near = (f'down to {sw["lost_m"]:.1f} m ({sw["lost_by"]}); closer it is out of view' if sw['lost_m'] > 0
                        else f'all the way to the {"bumper" if sw["across"] else "body side"} ({sw["lost_by"]})')
                _finding(out, 'note', 'seen', f'"{sw["label"]}" is seen {first} ({sw["by"]}) {near}{gap}.')
    mot = res.motion
    for c in mot.get('cases', []):
        if c['lifted']:
            _finding(out, 'warning', 'lift', f'{c["name"]}: the {", ".join(c["lifted"])} wheel would lift; the '
                                             'attitude assumes four wheels on the ground.')
    bad = [c for c in mot.get('cases', []) if c['image'] is not None and c['image'].status != 'good']
    if bad:
        w = max(bad, key=lambda c: c['image'].ground_shift_cm)
        g = max(bad, key=lambda c: c['image'].seam_ghost_cm)
        _finding(out, 'warning', 'ghost', f'Against the calibration at "{mot["info"]["calibrated_at"]}" the stitched '
                                          f'ground shifts up to {w["image"].ground_shift_cm:.1f} cm ({w["name"]}) and '
                                          f'the seams ghost up to {g["image"].seam_ghost_cm:.1f} cm ({g["name"]}); '
                                          f'{len(bad)} of {len(mot["cases"])} cases exceed the load cases\' limits (a '
                                          '2 cm seam ghost or 0.01 m² more blind ground). Record the body attitude '
                                          '(IMU) with the data to correct it.')
    fwd = [(c, h) for c in mot.get('cases', []) for h in c['horizon'] if h['facing'] == 'forward']
    if fwd:
        c, h = max(fwd, key=lambda ch: abs(ch[1]['dv_px']))
        if abs(h['dv_px']) >= 0.5:
            _finding(out, 'note', 'horizon', f'{c["name"]} moves the horizon of {h["tag"]} '
                                             f'{abs(h["dv_px"]):.0f} px {"up" if h["dv_px"] < 0 else "down"} '
                                             f'({abs(h["d_deg"]):.2f}°) against the calibration.', [h['uid']])
    order = {s: i for i, s in enumerate(SEVERITIES)}
    out.sort(key=lambda f: order.get(f['severity'], 9))
    return out


_ISSUES: dict = {}


def issues(project, vg=None) -> list:
    """The road's cheap problems for the Issues list (no camera analysis; cached on the inputs): standing on the road
    a wheel lifts, the body cuts into the road or a camera or sensor sits under 5 cm above it, and driving over a
    primitive across the road the body or a device touches it. Only while the project has a road switched on."""
    from . import rules
    st = settings_of(project)
    if not st.active or project.vehicle.mesh is None:
        return []
    v = project.variant
    key = _key(project, st.features) + json.dumps(
        [[c.uid, c.visible, c.x, c.y, c.z] for c in v.cameras] + [[s.uid, s.visible, s.x, s.y, s.z]
                                                                   for s in v.sensors], default=str)
    with _LOCK:
        hit = _ISSUES.get(key)
    if hit is not None:
        return list(hit)
    ter = active(project)
    res = TerrainResult(st, st.features, True, dict(ter.stance), None, [], {}, [], clearance(ter, project), [],
                        {'cases': [], 'info': {}}, [], v.name, v.uid)
    for i, f in enumerate(ter.features):
        if f.along:
            res.drive_over.append(dict(drive_over(ter, project, i), index=i, label=f.label()))
    out = []
    for f in _findings(res, project):
        if f['severity'] in ISSUE_SEVERITIES:
            out.append(rules.Issue(f['severity'], f['message'], f'{v.name} · road', list(f['uids']), 'terrain'))
    with _LOCK:
        if len(_ISSUES) > 16:
            _ISSUES.pop(next(iter(_ISSUES)))
        _ISSUES[key] = out
    return list(out)


# =========================================================================== output
def _r(v, d=3):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return round(v, d) if math.isfinite(v) else None


def _horizon_doc(h):
    return {'tag': h['tag'], 'facing': h['facing'], 'dv_px': _r(h['dv_px'], 1), 'd_deg': _r(h['d_deg'], 3),
            'd_tilt_deg': _r(h['d_tilt_deg'], 3), 'in_view': h['in_view']}


def _case_doc(c):
    im = c['image']
    return {'name': c['name'], 'long_g': c['long_g'], 'lat_g': c['lat_g'],
            'heave_mm': _r(c['heave_mm'], 2), 'pitch_deg': _r(c['pitch_deg'], 4), 'roll_deg': _r(c['roll_deg'], 4),
            'vs_calibration': {k: _r(v, 4) for k, v in c['rel'].items()}, 'bump_stops': c['bump_stops'],
            'wheels_lifting': c['lifted'], 'horizon': [_horizon_doc(h) for h in c['horizon']],
            'ground': ({'blind_m2': _r(im.blind_m2), 'blind_change_m2': _r(im.delta_blind),
                        'ground_shift_cm': _r(im.ground_shift_cm, 2), 'seam_ghost_cm': _r(im.seam_ghost_cm, 2),
                        'status': im.status} if im is not None else None)}


def _profile_doc(pr):
    return {'near_m': pr['near_m'], 'near_flat_m': pr['near_flat_m'], 'seen_to_m': pr['seen_to_m'],
            'seen_to_flat_m': pr['seen_to_flat_m'], 'hidden_by_road_m': [list(h) for h in pr['hidden']],
            'per_camera': pr['per_camera'], 'range_m': pr['range_m']}


def _sweep_doc(sw):
    return {'feature': sw['label'], 'first_seen_m': sw['first_seen_m'], 'first_seen_by': sw['by'],
            'lost_m': sw['lost_m'], 'lost_by': sw['lost_by'], 'gaps_m': [list(g) for g in sw['gaps']],
            'checked': [{'distance_m': d, 'camera': cam, 'height_px': _r(px, 1)} for d, cam, px in sw['distances']]}


def summary(res: TerrainResult) -> dict:
    """Compact figures (the assistant, the batch log)."""
    cov = res.coverage
    out = {'road': [describe(f) for f in res.features] if res.road else 'flat ground',
           'enabled': res.settings.enabled,
           'stance': ({'pitch_deg': _r(res.stance['pitch_deg'], 3), 'roll_deg': _r(res.stance['roll_deg'], 3),
                       'heave_mm': _r(res.stance['heave_mm'], 1)} if res.stance else None),
           'blind_m2': ({'flat': _r(cov['flat']['blind_m2']), 'road': _r(cov['road']['blind_m2'])} if cov else None),
           'road_seen_to_m': {k: {'road': pr['seen_to_m'], 'flat': pr['seen_to_flat_m']}
                              for k, pr in res.profiles.items()},
           'first_seen': [{'feature': s['label'], 'first_seen_m': s['first_seen_m'], 'by': s['by'],
                           'lost_m': s['lost_m']} for s in res.sweeps],
           'body_clearance_m': _r(res.clearance['body_m']) if res.clearance and res.clearance['body_m'] is not None
           else None,
           'motion': [{'name': c['name'], 'pitch_deg': _r(c['rel']['pitch_deg'], 3),
                       'roll_deg': _r(c['rel']['roll_deg'], 3), 'heave_mm': _r(c['rel']['heave_mm'], 1),
                       'horizon_px': {h['tag']: _r(h['dv_px'], 1) for h in c['horizon'] if h['facing'] == 'forward'},
                       'ground_shift_cm': _r(c['image'].ground_shift_cm, 1) if c['image'] is not None else None}
                      for c in res.motion.get('cases', [])],
           'findings': [{'severity': f['severity'], 'message': f['message']} for f in res.findings
                        if f['severity'] in ISSUE_SEVERITIES],
           'assumptions': ASSUMPTIONS}
    return out


def to_doc(res: TerrainResult, project=None) -> dict:
    """The JSON export: the road, the stance, the coverage against flat ground, the road ahead and behind, the bumps'
    and kerbs' first-seen distances, clearance and the body motion cases."""
    info = res.motion.get('info', {})
    return {
        'schema': SCHEMA, 'project': getattr(project, 'name', ''), 'variant': res.variant,
        'frame': 'vehicle (ISO 8855, rear axle)',
        'units': {'distance': 'm (from the bumper or the body side)', 'height': 'm', 'angle': 'deg', 'drop': 'mm',
                  'area': 'm²', 'density': 'px/cm', 'image': 'px (rows, positive = down the image)',
                  'acceleration': 'g (longitudinal + accelerating, lateral + turning left)'},
        'signs': {'pitch': 'positive = nose down', 'roll': 'positive = right side lower',
                  'horizon_d_deg': 'positive = the horizon rises in the view'},
        'assumptions': ASSUMPTIONS, 'motion_assumptions': SU.DYN_ASSUMPTIONS, 'motion_note': MOTION_NOTE,
        'settings': res.settings.to_dict(),
        'road': {'features': [dict(f, description=describe(f)) for f in res.features], 'used': res.road,
                 'stance': ({k: (v if not isinstance(v, float) else _r(v, 4)) for k, v in res.stance.items()}
                            if res.stance else None),
                 'coverage': res.coverage, 'per_camera': res.per_camera,
                 'profiles': {k: _profile_doc(pr) for k, pr in res.profiles.items()},
                 'first_seen': [_sweep_doc(s) for s in res.sweeps],
                 'clearance': res.clearance, 'drive_over': res.drive_over},
        'motion': {'info': info, 'cases': [_case_doc(c) for c in res.motion.get('cases', [])]},
        'findings': [{'severity': f['severity'], 'kind': f['kind'], 'message': f['message']} for f in res.findings],
    }


MOTION_COLUMNS = ('case', 'long_g', 'lat_g', 'heave_mm', 'pitch_deg', 'roll_deg', 'rel_heave_mm', 'rel_pitch_deg',
                  'rel_roll_deg', 'camera', 'facing', 'horizon_dv_px', 'horizon_d_deg', 'horizon_d_tilt_deg',
                  'blind_change_m2', 'ground_shift_cm', 'seam_ghost_cm', 'status')


def motion_csv(res: TerrainResult) -> str:
    """One row per motion case and camera (the horizon), the body figures repeated."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator='\n')
    w.writerow(MOTION_COLUMNS)
    for c in res.motion.get('cases', []):
        im = c['image']
        head = [c['name'], c['long_g'], c['lat_g'], _r(c['heave_mm'], 2), _r(c['pitch_deg'], 4),
                _r(c['roll_deg'], 4), _r(c['rel']['heave_mm'], 2), _r(c['rel']['pitch_deg'], 4),
                _r(c['rel']['roll_deg'], 4)]
        tail = [_r(im.delta_blind), _r(im.ground_shift_cm, 2), _r(im.seam_ghost_cm, 2), im.status] if im else \
            ['', '', '', '']
        for h in c['horizon'] or [None]:
            if h is None:
                w.writerow(head + ['', '', '', '', ''] + tail)
            else:
                w.writerow(head + [h['tag'], h['facing'], _r(h['dv_px'], 1), _r(h['d_deg'], 3),
                                   _r(h['d_tilt_deg'], 3)] + tail)
    return buf.getvalue()


def profile_csv(res: TerrainResult) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator='\n')
    w.writerow(['direction', 'distance_m', 'height_m', 'seen', 'seen_flat', 'best_density_px_cm'])
    for direction, pr in res.profiles.items():
        for d, h, s, sf, de in zip(pr['d'], pr['h'], pr['seen'], pr['seen_flat'], pr['density']):
            w.writerow([direction, _r(d, 2), _r(h, 4), int(bool(s)), int(bool(sf)), _r(de, 3)])
    return buf.getvalue()


def png(res: TerrainResult, project=None, size=(1200, 520), reach=30.0) -> bytes:
    """The road's side view: its height along the centre line behind and ahead of the car (vertical scale
    exaggerated), the car's outline, where some camera sees the road (green), where only flat ground would be seen
    (red: hidden by the road) and the primitives' places."""
    import cv2
    W, H = size
    img = np.full((H, W, 3), 250, np.uint8)
    ahead, behind = res.profiles.get('ahead'), res.profiles.get('behind')
    ref = None
    if project is not None:
        ref = references(project)
    xf = ref['front_x'] if ref else 3.2
    xr = ref['rear_x'] if ref else -0.8
    x_lo, x_hi = xr - reach, xf + reach
    hs = [0.0]
    for pr in (ahead, behind):
        if pr is not None:
            sel = pr['d'] <= reach
            hs += list(pr['h'][sel])
    zl, zh = min(min(hs), 0.0) - 0.2, max(max(hs), 0.0) + 1.8
    pad = 40
    sx = (W - 2 * pad) / (x_hi - x_lo)
    ex = max(1.0, min(10.0, 0.6 * (H - 2 * pad) / max(zh - zl, 0.1) / sx))      # vertical exaggeration
    sz = sx * ex

    def P(x, z):
        return int(round(pad + (x - x_lo) * sx)), int(round(H - pad - (z - zl) * sz))
    cv2.line(img, P(x_lo, 0), P(x_hi, 0), (215, 215, 215), 1, cv2.LINE_AA)
    for pr, sgn, x0 in ((ahead, 1, xf), (behind, -1, xr)):
        if pr is None:
            continue
        sel = pr['d'] <= reach
        xs = x0 + sgn * pr['d'][sel]
        zs = pr['h'][sel]
        seen, flat = pr['seen'][sel], pr['seen_flat'][sel]
        for k in range(len(xs) - 1):
            col = (60, 160, 60) if seen[k] else ((50, 60, 210) if flat[k] else (150, 150, 150))
            cv2.line(img, P(xs[k], zs[k]), P(xs[k + 1], zs[k + 1]), col, 3, cv2.LINE_AA)
    # the car: its bumpers, the wheels' contact and its cameras (height true to scale × exaggeration)
    st = res.stance or {}
    pitch = math.radians(st.get('pitch_deg', 0.0) or 0.0)
    heave = (st.get('heave_mm', 0.0) or 0.0) / 1000.0
    hgt = 1.5
    if project is not None:
        dims = SU.vehicle_geometry(project)['dims']
        hgt = dims.get('Height', 1.5)

    def body(x, z):
        return x * math.cos(pitch) + z * math.sin(pitch), -x * math.sin(pitch) + z * math.cos(pitch) - heave
    outline = [body(xr, 0.15), body(xf, 0.15), body(xf, hgt * 0.55), body(xr + 0.6 * (xf - xr), hgt),
               body(xr + 0.25 * (xf - xr), hgt), body(xr, hgt * 0.6)]
    pts = np.array([P(x, z) for x, z in outline], np.int32)
    cv2.polylines(img, [pts], True, (90, 90, 90), 2, cv2.LINE_AA)
    if project is not None:
        for c in project.variant.cameras:
            if c.visible:
                x, z = body(c.x, c.z)
                cv2.circle(img, P(x, z), 4, (30, 30, 30), -1, cv2.LINE_AA)
    for f in res.features:
        if f['kind'] in ('bump', 'kerb', 'ramp', 'crest', 'garage') and f.get('where', f.get('side')) in WHERE:
            ft = Feature(f, ref or {'front_x': xf, 'rear_x': xr, 'left_y': 0.9, 'right_y': -0.9, 'width': 1.8})
            x = ft.origin
            if x_lo <= x <= x_hi:
                cv2.line(img, P(x, zl + 0.05), P(x, zl + 0.25), (40, 40, 40), 1, cv2.LINE_AA)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, f'Road along the centre line, {reach:g} m behind and ahead (height x{ex:.1f}): green seen, red '
                     'hidden by the road, grey out of view', (pad, 24), font, 0.5, (40, 40, 40), 1, cv2.LINE_AA)
    ok, buf = cv2.imencode('.png', img)
    return buf.tobytes() if ok else b''


def files(res: TerrainResult, project=None) -> list:
    """[(name, bytes)]: terrain.json, terrain_motion.csv, and with a road terrain_profile.csv and
    terrain_profile.png."""
    out = [('terrain.json', (json.dumps(to_doc(res, project), indent=1, default=_json_default) + '\n').encode()),
           ('terrain_motion.csv', motion_csv(res).encode('utf-8'))]
    if res.profiles:
        out.append(('terrain_profile.csv', profile_csv(res).encode('utf-8')))
        out.append(('terrain_profile.png', png(res, project)))
    return out


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


def export(res: TerrainResult, project, out_dir) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for name, data in files(res, project):
        path = os.path.join(out_dir, name)
        with open(path, 'wb') as fh:
            fh.write(data)
        paths.append(path)
    return paths
