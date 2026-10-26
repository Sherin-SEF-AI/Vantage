"""Suspension and loads: how far each corner of the body drops under a load case, and so how far every camera and
sensor of the rig moves from where it was calibrated.

The model (static loads only)
- Each corner (FL, FR, RL, RR) is a suspension spring in series with the tyre. The spring is the wheel rate (spring
  rate × motion ratio², the motion ratio being spring travel per wheel travel), plus an optional bump stop that adds
  force once the wheel travel passes its gap (a rate, or a progressive force-travel curve). A measured wheel-load vs
  wheel-travel curve (a K&C rig's bounce test, or CAD kinematics) replaces the spring and the bump stop at that corner;
  the kerb corner load places the car on it. An anti-roll bar per axle adds force against a left/right travel
  difference: its rate is the bar's contribution to each wheel's rate in pure roll (F_left = k_bar (s_left - s_right)
  / 2), so a single wheel in bump sees half of it.
- The body is rigid on its four corners. Four supports make the problem statically indeterminate; it is solved the
  standard way for a rigid body on springs: compatibility (the body's drops at the four corners lie on one plane:
  heave, pitch and roll) plus force balance and moment balance about x and y of the added weights, with each corner's
  force-deflection law. That is seven equations (three of balance, four of the springs in series) in seven unknowns
  (heave, pitch, roll and the four suspension travels), solved by Newton's method with a line search: one step for
  linear springs, a few with bump stops and curves.
- Load items are masses at points in the vehicle frame (x forward, y left, z up, origin at the rear-axle centre on the
  ground, metres). Static vertical loads only depend on x and y; z is kept for the centre of gravity.
- The vehicle model (and the camera positions on it) is the kerb state: empty, no rig, at the design ride heights.
  Results are given from kerb and, for the cameras, relative to the calibration load (the case the cameras are
  calibrated at): that difference is what the calibration has to absorb.

Braking, accelerating and cornering (transfer, dynamic_solve; used by core.terrain): the quasi-static load transfer of
a steady acceleration, with anti-dive, anti-lift and anti-squat percentages and roll-centre heights as inputs, solved
with the same corners (the linkage's share bypasses the springs). The same solve takes a road that is not flat under
the tyres (ground: each tyre's road raised or lowered).

Out of scope: dampers and transients, every other kinematic effect of braking, accelerating or cornering, wheel motion
along a kinematic arc (the wheel travels vertically), chassis torsion, bushing compliance,
damper and bush friction (a real car settles a few millimetres either side of the friction-free position, depending on
how it was loaded), tyre pressure changes, fuel level. A wheel that would lift is reported, not solved.

Signs: drops and travels positive down (bump); pitch positive nose down; roll positive right side down (right-hand
about +x, ISO 8855). Units: N/mm, mm, kg, N; positions in metres.

File format (JSON or YAML; schema vantage.suspension_data/1). Bare numbers are in the default units, which a 'units'
block may change ({rate: lbf/in, length: in, mass: lb, force: lbf, position: mm}); any value may also be a string with
its own unit ('25 N/mm', '140 lbf/in', '25 kN/m', '380 kg', '3.7 kN', '70 mm', '1.49 m').
  schema: vantage.suspension_data/1
  source: imported            # typical | entered | imported
  note: K&C test 2026-03, front struts and rear twist beam
  vehicle: {kerb_mass: 1231 kg, front_pct: 60, left_pct: 50, track_front: 1.49 m, track_rear: 1.49 m,
            gross_mass: 1650 kg, driver_side: right}            # track 0 = from the vehicle model
  axles:                                                         # applies to both corners of the axle
    front: {spring_rate: 23 N/mm, motion_ratio: 1.0, tyre_rate: 200 N/mm, arb_rate: 12 N/mm, bump_gap: 70 mm,
            bump_curve: [[0, 0], [10, 800], [20, 3000]]}       # [compression mm beyond contact, force N]
    rear: {wheel_rate: 19 N/mm, axle_load: 492 kg}               # axle_load: split left/right by left_pct
  corners:                                                       # per corner, over the axle values
    FL: {kerb_load: 372 kg, ride_height: 356 mm,                 # wheel centre to wheel-arch lip at kerb
         curve: [[3000, -20], [3650, 0], [4500, 25]]}            # [wheel load N, wheel travel mm, bump +]
  loads:
    occupant_mass: 75 kg
    items: [{id: driver, name: Driver, kind: occupant, mass: 75, x: 1.05, y: -0.39, z: 0.52}, ...]
    rig_masses: {L1: 0.9}                                        # kg per camera, sensor or rig part (uid or tag)
    cases: [{id: cal, name: Rig only, items: [rig, computer]}, {name: Crew, items: {rig: null, driver: 82}}]
    calibrated_at: cal                                           # a case id or name; '' = kerb
  tolerance: {mm: 10, deg: 0.2}                                  # camera move vs calibration that is flagged
CSV: a "parameter,value,unit" sheet (parameter = <FL|FR|RL|RR|front|rear>.<field>, or a vehicle field: kerb_mass,
front_pct, left_pct, track_front, track_rear, gross_mass, driver_side, arb_front, arb_rear, occupant_mass) or a
ride-height table: load and travel columns with their units in the header (load_N or 'load (kN)', travel_mm, or
ride_height_mm, which falls as the load rises) and an optional corner (FL…) or axle (front/rear) column.
"""
from __future__ import annotations

import copy
import csv
import io
import json
import math
import os
import re
import threading
from dataclasses import dataclass, field

import numpy as np

from .mathutil import euler_zyx

SCHEMA = 'vantage.suspension/1'
FILE_SCHEMA = 'vantage.suspension_data/1'
CORNERS = ('FL', 'FR', 'RL', 'RR')
CORNER_LABEL = {'FL': 'Front left', 'FR': 'Front right', 'RL': 'Rear left', 'RR': 'Rear right'}
AXLE = {'FL': 'front', 'FR': 'front', 'RL': 'rear', 'RR': 'rear'}
AXLE_CORNERS = {'front': ('FL', 'FR'), 'rear': ('RL', 'RR')}
PARTNER = {'FL': 'FR', 'FR': 'FL', 'RL': 'RR', 'RR': 'RL'}
G = 9.80665
LBF = 4.4482216152605        # N
LB = 0.45359237              # kg
ITEM_KINDS = ('occupant', 'luggage', 'roof', 'equipment', 'rig', 'other')
KIND_LABEL = {'occupant': 'Occupant', 'luggage': 'Luggage', 'roof': 'Roof load', 'equipment': 'Equipment',
              'rig': 'Rig (from the design)', 'other': 'Other'}
SOURCES = ('typical', 'entered', 'imported')
SEVERITIES = ('critical', 'serious', 'warning', 'note', 'good')
ISSUE_SEVERITIES = ('critical', 'serious', 'warning')
MAX_ITEMS = 60
MAX_CASES = 40
MAX_CURVE = 400
UNITS_DOC = {'rate': 'N/mm', 'length': 'mm', 'mass': 'kg', 'force': 'N', 'position': 'm'}

ASSUMPTIONS = ('Static loads on a rigid body on four corner springs (each the suspension in series with the tyre, with '
               'optional bump stops, measured wheel-load curves and anti-roll bars), solved for force and moment '
               'balance with the corners\' drops on one plane. No anti-dive or anti-squat (no braking, accelerating '
               'or cornering), no kinematic wheel path, chassis torsion, bushing compliance or friction: a real car '
               'settles within a few millimetres of this. The vehicle model is the kerb state. Rigid body: the rig\'s '
               'cameras and sensors do not move relative to each other, only relative to the road. Rig masses are '
               'estimates unless entered; typical suspension figures are typical, not this vehicle\'s.')
TYPICAL_NOTE = 'Typical values for a compact sedan: replace them with your vehicle\'s suspension data.'

# ---- typical figures (used until the vehicle's own data is entered; every one is labelled typical in the UI)
# Ride frequencies: passenger cars are sprung for roughly 1.0-1.5 Hz with the rear 10-20 % above the front (Olley's
# flat-ride rule; Gillespie, Fundamentals of Vehicle Dynamics, SAE 1992, ch. 5; Milliken & Milliken, Race Car Vehicle
# Dynamics, SAE 1995, ch. 16). The ride rate is the wheel rate in series with the tyre, the wheel rate the spring rate
# times the motion ratio squared (Milliken). Unsprung mass about 35 kg per corner on a compact car (typical).
TYP_RIDE_HZ = {'front': 1.25, 'rear': 1.5}
TYP_UNSPRUNG_KG = 35.0
# Tyre vertical rate: about 180-250 N/mm for 14-16 inch passenger tyres at road pressure (typical).
TYP_TYRE = 200.0
# Anti-roll bars as their contribution to the wheel rate in roll (a rear twist beam acts as one): typical of compact
# front-driven cars, whose roll stiffness is biased to the front.
TYP_ARB = {'front': 12.0, 'rear': 8.0}
# Wheel travel from kerb to bump-stop contact, and a generic progressive polyurethane jounce bumper at the wheel (the
# shape only: compression beyond contact mm -> force N).
TYP_BUMP_GAP = {'front': 70.0, 'rear': 60.0}
TYP_BUMP_CURVE = [[0.0, 0.0], [5.0, 300.0], [10.0, 800.0], [15.0, 1600.0], [20.0, 3000.0], [25.0, 5500.0],
                  [30.0, 9000.0]]
# Kerb mass when nothing is known: about 120 kg per m³ of the body's bounding box (compact cars, SUVs and vans fall
# within about ±20 % of it); a front-engined, front-driven compact car carries about 60 % on its front axle.
TYP_KG_PER_M3 = 120.0
TYP_FRONT_PCT = 60.0
TYP_KERB_KG = 1250.0
TYP_WHEELBASE = 2.6
TYP_TRACK = 1.5
# Occupant: 75 kg, the driver mass of EU Regulation 1230/2012 (68 kg plus 7 kg of luggage, as in ISO 2416).
OCCUPANT_KG = 75.0
# Rig estimates: a sensor's housing volume (from its library size) at a typical density, plus its mount; a camera's
# module weight plus a 15 g bracket (as core.brackets); rig parts as 3 mm aluminium shells; the ECU 2.5 kg.
TYP_DENSITY = {'lidar': 1.2, 'radar': 1.2, 'gnss': 0.4, 'imu': 1.2}           # g per cm³ of housing
TYP_MOUNT_KG = {'lidar': 0.4, 'radar': 0.1, 'gnss': 0.1, 'imu': 0.1}
CAMERA_BRACKET_KG = 0.015
CAMERA_FALLBACK_KG = 0.08
ECU_KG = 2.5
RIG_WALL_M = 0.003
ALU_KG_M3 = 2700.0

LIMITS = {'spring_rate': (0.5, 2000.0), 'motion_ratio': (0.1, 3.0), 'tyre_rate': (10.0, 5000.0),
          'kerb_load': (5.0, 5000.0), 'ride_height': (0.0, 2000.0), 'bump_gap': (0.0, 500.0),
          'bump_rate': (0.0, 1e6), 'arb_rate': (0.0, 2000.0), 'track': (0.0, 4.0), 'gross_mass': (0.0, 100000.0),
          'occupant_mass': (1.0, 300.0), 'item_mass': (0.0, 10000.0), 'position': (-30.0, 30.0),
          'tol_mm': (0.1, 1000.0), 'tol_deg': (0.001, 10.0), 'pct': (1.0, 99.0), 'kerb_mass': (50.0, 100000.0),
          'axle_load': (10.0, 10000.0), 'wheel_rate': (0.5, 2000.0), 'rig_mass': (0.0, 2000.0)}
CORNER_FIELDS = ('spring_rate', 'motion_ratio', 'wheel_rate', 'tyre_rate', 'kerb_load', 'ride_height', 'bump_gap',
                 'bump_rate', 'bump_curve', 'curve')
FIELD_KIND = {'spring_rate': 'rate', 'wheel_rate': 'rate', 'motion_ratio': 'ratio', 'tyre_rate': 'rate',
              'kerb_load': 'mass', 'axle_load': 'mass', 'ride_height': 'length', 'bump_gap': 'length',
              'bump_rate': 'rate', 'arb_rate': 'rate', 'track_front': 'position', 'track_rear': 'position',
              'gross_mass': 'mass', 'kerb_mass': 'mass', 'front_pct': 'percent', 'left_pct': 'percent',
              'occupant_mass': 'mass', 'arb_front': 'rate', 'arb_rear': 'rate'}
VEHICLE_PARAMS = ('kerb_mass', 'front_pct', 'left_pct', 'track_front', 'track_rear', 'gross_mass', 'driver_side',
                  'arb_front', 'arb_rear', 'occupant_mass')


# =========================================================================== units
_U = {
    'rate': ({'n/mm': 1.0, 'kn/m': 1.0, 'n/m': 1e-3, 'kn/mm': 1e3, 'n/cm': 0.1, 'lbf/in': LBF / 25.4,
              'lb/in': LBF / 25.4, 'lbs/in': LBF / 25.4, 'kgf/mm': G, 'kg/mm': G, 'kgf/cm': G / 10, 'kg/cm': G / 10},
             'a spring rate', 'N/mm, kN/m, N/m, lbf/in or kgf/mm'),
    'force': ({'n': 1.0, 'kn': 1000.0, 'dan': 10.0, 'lbf': LBF, 'lb': LBF, 'lbs': LBF, 'kgf': G, 'kg': G},
              'a load', 'N, kN, daN, kgf (kg) or lbf'),
    'mass': ({'kg': 1.0, 'g': 1e-3, 't': 1000.0, 'lb': LB, 'lbs': LB, 'lbm': LB, 'kgf': 1.0, 'n': 1.0 / G,
              'kn': 1000.0 / G, 'dan': 10.0 / G, 'lbf': LB},
             'a mass', 'kg, g, t, lb, or a weight in N, kN, lbf'),
    'length': ({'mm': 1.0, 'cm': 10.0, 'm': 1000.0, 'in': 25.4, 'inch': 25.4, 'inches': 25.4, '"': 25.4,
                'ft': 304.8}, 'a length', 'mm, cm, m or in'),
    'position': ({'m': 1.0, 'mm': 1e-3, 'cm': 1e-2, 'in': 0.0254, 'ft': 0.3048}, 'a position', 'm, mm, cm or in'),
    'ratio': ({'': 1.0}, 'a ratio', 'no unit'),
    'percent': ({'%': 1.0, 'pct': 1.0, 'percent': 1.0}, 'a percentage', '%'),
    'angle': ({'°': 1.0, 'deg': 1.0, 'degrees': 1.0, 'rad': 180.0 / math.pi}, 'an angle', '° or rad'),
}
_NUM = re.compile(r'^\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*(.*?)\s*$')


def _norm_unit(u: str) -> str:
    return (u or '').strip().lower().replace(' ', '').replace('·', '').replace('newtons', 'n').replace('newton', 'n')


def unit_factor(unit: str, kind: str, where: str = '') -> float:
    """Factor from `unit` to the base unit of `kind` (N/mm, N, kg, mm, m, ratio, %). Raises ValueError."""
    table, what, valid = _U[kind]
    u = _norm_unit(unit)
    if u in table:
        return table[u]
    if not u and kind in ('ratio', 'percent', 'angle'):
        return 1.0
    raise ValueError(f'{where + ": " if where else ""}unknown unit "{unit}" for {what} (use {valid})')


def quantity(v, kind: str, unit: str | None = None, where: str = '') -> float:
    """A number in the base unit of `kind`: v is a number in `unit` (default: the base unit) or a string with its own
    unit ('25 N/mm', '140 lbf/in', '3.7 kN', '70 mm'). Raises ValueError with a clear message."""
    pre = f'{where}: ' if where else ''
    if isinstance(v, bool):
        raise ValueError(f'{pre}expected a number, got {v!r}')
    if isinstance(v, (int, float)):
        x = float(v) * (unit_factor(unit, kind, where) if unit else 1.0)
    else:
        s = str(v).strip().replace('−', '-')
        if s.count(',') == 1 and '.' not in s:
            s = s.replace(',', '.')                       # a decimal comma
        m = _NUM.match(s)
        if not m:
            raise ValueError(f'{pre}"{v}" is not a number')
        x = float(m.group(1))
        u = m.group(2)
        x *= unit_factor(u, kind, where) if u else (unit_factor(unit, kind, where) if unit else 1.0)
    if not math.isfinite(x):
        raise ValueError(f'{pre}"{v}" is not a finite number')
    return x


def _clamp(v, lim):
    lo, hi = LIMITS[lim]
    return min(max(v, lo), hi)


class _Reader:
    """Reads values for from_dict: tolerant (junk is skipped) or strict (collects errors for an import)."""

    def __init__(self, units=None, strict=False):
        self.units = {}
        self.strict = strict
        self.errors: list[str] = []
        self.warnings: list[str] = []
        if isinstance(units, dict):
            for k, u in units.items():
                if k in _U:
                    try:
                        unit_factor(u, k, f'units.{k}')
                        self.units[k] = str(u)
                    except ValueError as e:
                        self.errors.append(str(e))

    def num(self, v, kind, where, lim=None):
        try:
            x = quantity(v, kind, self.units.get(kind), where)
        except ValueError as e:
            self.errors.append(str(e))
            return None
        if lim is not None:
            c = _clamp(x, lim)
            if c != x:
                lo, hi = LIMITS[lim]
                self.warnings.append(f'{where}: {x:g} is outside {lo:g} to {hi:g}; {c:g} is used')
                x = c
        return x

    def text(self, v, where, choices=None, default=None):
        if v is None:
            return default
        s = str(v).strip()
        if choices is not None and s.lower() not in choices:
            self.errors.append(f'{where}: "{v}" is not one of {", ".join(choices)}')
            return default
        return s.lower() if choices is not None else s


# =========================================================================== settings
@dataclass
class Corner:
    spring_rate: float = 25.0           # N/mm at the spring
    motion_ratio: float = 1.0           # spring travel per wheel travel
    tyre_rate: float = TYP_TYRE         # N/mm, vertical
    kerb_load: float = 300.0            # kg on this wheel at kerb (sprung and unsprung)
    ride_height: float = 0.0            # mm, wheel centre to wheel-arch lip at kerb (0 = not given)
    bump_gap: float = 70.0              # mm of wheel travel from kerb to bump-stop contact
    bump_rate: float = 0.0              # N/mm at the wheel once in contact (used without a bump curve)
    bump_curve: list = field(default_factory=list)   # [[compression mm, force N], ...] at the wheel
    curve: list = field(default_factory=list)        # [[wheel load N, wheel travel mm], ...] measured, bump +

    @property
    def wheel_rate(self) -> float:
        return self.spring_rate * self.motion_ratio ** 2

    @property
    def has_bump(self) -> bool:
        return bool(self.bump_curve) or self.bump_rate > 0

    def to_dict(self):
        return {'spring_rate': self.spring_rate, 'motion_ratio': self.motion_ratio, 'tyre_rate': self.tyre_rate,
                'kerb_load': self.kerb_load, 'ride_height': self.ride_height, 'bump_gap': self.bump_gap,
                'bump_rate': self.bump_rate, 'bump_curve': [list(p) for p in self.bump_curve],
                'curve': [list(p) for p in self.curve]}


@dataclass
class LoadItem:
    id: str
    name: str
    kind: str = 'other'                 # ITEM_KINDS; 'rig' takes its mass and position from the design
    mass: float = 0.0                   # kg
    x: float = 0.0                      # m, vehicle frame
    y: float = 0.0
    z: float = 0.5

    def to_dict(self):
        return {'id': self.id, 'name': self.name, 'kind': self.kind, 'mass': self.mass, 'x': self.x, 'y': self.y,
                'z': self.z}


@dataclass
class LoadCase:
    id: str
    name: str
    items: dict = field(default_factory=dict)        # item id -> mass kg in this case, or None for the item's own

    def to_dict(self):
        return {'id': self.id, 'name': self.name, 'items': dict(self.items)}


def _typical_corners(kerb_kg=TYP_KERB_KG, front_pct=TYP_FRONT_PCT, left_pct=50.0) -> dict:
    """Typical corners for a car of this kerb mass: rates from the typical ride frequencies."""
    out = {}
    for c in CORNERS:
        ax = AXLE[c]
        share = (front_pct if ax == 'front' else 100.0 - front_pct) / 100.0
        side = (left_pct if c in ('FL', 'RL') else 100.0 - left_pct) / 100.0
        kerb = kerb_kg * share * side
        sprung = max(kerb - TYP_UNSPRUNG_KG, 0.3 * kerb)
        ride = (2 * math.pi * TYP_RIDE_HZ[ax]) ** 2 * sprung / 1000.0              # N/mm
        wheel = ride * TYP_TYRE / max(TYP_TYRE - ride, 1.0)
        out[c] = Corner(spring_rate=round(wheel * 2) / 2, motion_ratio=1.0, tyre_rate=TYP_TYRE,
                        kerb_load=round(kerb, 1), bump_gap=TYP_BUMP_GAP[ax], bump_rate=0.0,
                        bump_curve=[list(p) for p in TYP_BUMP_CURVE])
    return out


@dataclass
class SuspensionSettings:
    corners: dict = field(default_factory=_typical_corners)
    arb_front: float = TYP_ARB['front']    # N/mm, the bar's contribution to each wheel's rate in roll
    arb_rear: float = TYP_ARB['rear']
    track_front: float = 0.0               # m between the tyre centres, 0 = from the vehicle model
    track_rear: float = 0.0
    gross_mass: float = 0.0                # kg, gross vehicle mass to check the cases against (0 = not checked)
    driver_side: str = 'left'              # left | right: which front seat the driver is in
    occupant_mass: float = OCCUPANT_KG
    items: list = field(default_factory=list)       # LoadItem
    rig_masses: dict = field(default_factory=dict)  # camera / sensor / rig part uid (or tag) -> kg
    cases: list = field(default_factory=list)       # LoadCase
    calibrated_at: str = ''                         # id of the case the cameras are calibrated at, '' = kerb
    tol_mm: float = 10.0                            # a camera move (vs calibration) above this is flagged
    tol_deg: float = 0.2
    source: str = 'typical'                         # where the suspension figures come from: typical | entered | imported
    note: str = ''

    # ------------------------------------------------------------------ derived
    @property
    def kerb_mass(self) -> float:
        return sum(self.corners[c].kerb_load for c in CORNERS)

    @property
    def front_pct(self) -> float:
        m = self.kerb_mass
        return 100.0 * (self.corners['FL'].kerb_load + self.corners['FR'].kerb_load) / m if m > 0 else 50.0

    @property
    def left_pct(self) -> float:
        m = self.kerb_mass
        return 100.0 * (self.corners['FL'].kerb_load + self.corners['RL'].kerb_load) / m if m > 0 else 50.0

    def set_kerb(self, mass=None, front_pct=None, left_pct=None):
        """Spread a kerb mass over the corners by the front-axle and left-side shares (keeping what is not given)."""
        m = self.kerb_mass if mass is None else float(mass)
        f = (self.front_pct if front_pct is None else float(front_pct)) / 100.0
        lft = (self.left_pct if left_pct is None else float(left_pct)) / 100.0
        for c in CORNERS:
            share = (f if AXLE[c] == 'front' else 1 - f) * (lft if c in ('FL', 'RL') else 1 - lft)
            self.corners[c].kerb_load = _clamp(m * share, 'kerb_load')

    def arb(self, axle) -> float:
        return self.arb_front if axle == 'front' else self.arb_rear

    def item(self, iid):
        return next((i for i in self.items if i.id == iid), None)

    def case(self, cid):
        return next((c for c in self.cases if c.id == cid), None)

    @property
    def is_typical(self) -> bool:
        return self.source == 'typical'

    def new_id(self, stem, kind='item'):
        """A free id for a new load item (kind 'item') or case ('case') from a name."""
        taken = {i.id for i in self.items} if kind == 'item' else {c.id for c in self.cases}
        stem = re.sub(r'[^a-z0-9]+', '_', str(stem).lower()).strip('_') or 'item'
        if stem not in taken:
            return stem
        n = 2
        while f'{stem}_{n}' in taken:
            n += 1
        return f'{stem}_{n}'

    # ------------------------------------------------------------------ dict / file
    def to_dict(self) -> dict:
        return {'units': dict(UNITS_DOC), 'source': self.source, 'note': self.note,
                'vehicle': {'kerb_mass': round(self.kerb_mass, 3), 'front_pct': round(self.front_pct, 3),
                            'left_pct': round(self.left_pct, 3), 'track_front': self.track_front,
                            'track_rear': self.track_rear, 'gross_mass': self.gross_mass,
                            'driver_side': self.driver_side},
                'axles': {'front': {'arb_rate': self.arb_front}, 'rear': {'arb_rate': self.arb_rear}},
                'corners': {c: self.corners[c].to_dict() for c in CORNERS},
                'loads': {'occupant_mass': self.occupant_mass, 'items': [i.to_dict() for i in self.items],
                          'rig_masses': dict(self.rig_masses), 'cases': [c.to_dict() for c in self.cases],
                          'calibrated_at': self.calibrated_at},
                'tolerance': {'mm': self.tol_mm, 'deg': self.tol_deg}}

    @staticmethod
    def from_dict(d, base: 'SuspensionSettings | None' = None, strict=False) -> 'SuspensionSettings':
        """Settings from a dict (the project's analysis.suspension, or a file): missing parts come from base (default:
        generic typical values), junk is skipped and numbers are clamped to LIMITS. strict: raise ValueError listing
        every unreadable value (an import); the clamping notes are left in .warnings on the result."""
        s = copy.deepcopy(base) if base is not None else SuspensionSettings()
        d = d if isinstance(d, dict) else {}
        rd = _Reader(d.get('units'), strict)
        src = str(d.get('source') or '').strip().lower()
        if src in SOURCES:
            s.source = src
        if 'note' in d and d['note'] is not None:
            s.note = str(d['note'])[:500]
        veh = d.get('vehicle') if isinstance(d.get('vehicle'), dict) else {}
        for k in ('track_front', 'track_rear'):
            if k in veh:
                v = rd.num(veh[k], 'position', f'vehicle.{k}', 'track')
                if v is not None:
                    setattr(s, k, 0.0 if v < 0.3 else v)
        if 'gross_mass' in veh:
            v = rd.num(veh['gross_mass'], 'mass', 'vehicle.gross_mass', 'gross_mass')
            if v is not None:
                s.gross_mass = v
        s.driver_side = rd.text(veh.get('driver_side'), 'vehicle.driver_side', ('left', 'right'), s.driver_side)
        kerb = {}
        for k, lim in (('kerb_mass', 'kerb_mass'), ('front_pct', 'pct'), ('left_pct', 'pct')):
            if k in veh:
                v = rd.num(veh[k], 'mass' if k == 'kerb_mass' else 'percent', f'vehicle.{k}', lim)
                if v is not None:
                    kerb[k] = v
        if kerb:
            s.set_kerb(kerb.get('kerb_mass'), kerb.get('front_pct'), kerb.get('left_pct'))
        lp = kerb.get('left_pct', s.left_pct)
        axles = d.get('axles') if isinstance(d.get('axles'), dict) else {}
        for ax in ('front', 'rear'):
            a = axles.get(ax)
            if not isinstance(a, dict):
                continue
            if 'arb_rate' in a:
                v = rd.num(a['arb_rate'], 'rate', f'axles.{ax}.arb_rate', 'arb_rate')
                if v is not None:
                    setattr(s, f'arb_{ax}', v)
            if 'axle_load' in a:
                v = rd.num(a['axle_load'], 'mass', f'axles.{ax}.axle_load', 'axle_load')
                if v is not None:
                    left, right = AXLE_CORNERS[ax]
                    s.corners[left].kerb_load = _clamp(v * lp / 100.0, 'kerb_load')
                    s.corners[right].kerb_load = _clamp(v * (100.0 - lp) / 100.0, 'kerb_load')
            for c in AXLE_CORNERS[ax]:
                _read_corner(rd, s.corners[c], a, f'axles.{ax}')
        corners = d.get('corners') if isinstance(d.get('corners'), dict) else {}
        for key, cd in corners.items():
            c = corner_key(key)
            if c is None:
                rd.errors.append(f'corners: "{key}" is not a corner (FL, FR, RL, RR)')
                continue
            if isinstance(cd, dict):
                _read_corner(rd, s.corners[c], cd, f'corners.{c}')
        for k in ('arb_front', 'arb_rear'):               # flat aliases (the CSV sheet's names)
            if k in veh:
                v = rd.num(veh[k], 'rate', f'vehicle.{k}', 'arb_rate')
                if v is not None:
                    setattr(s, k, v)
        loads = d.get('loads') if isinstance(d.get('loads'), dict) else {}
        if 'occupant_mass' in loads:
            v = rd.num(loads['occupant_mass'], 'mass', 'loads.occupant_mass', 'occupant_mass')
            if v is not None:
                s.occupant_mass = v
        if isinstance(loads.get('items'), list):
            s.items = _read_items(rd, loads['items'])
        if isinstance(loads.get('rig_masses'), dict):
            s.rig_masses = {}
            for k, v in loads['rig_masses'].items():
                m = rd.num(v, 'mass', f'loads.rig_masses.{k}', 'rig_mass')
                if m is not None:
                    s.rig_masses[str(k)] = m
        if isinstance(loads.get('cases'), list):
            s.cases = _read_cases(rd, loads['cases'], s)
        else:                                             # keep the base cases, without items that no longer exist
            known = {i.id for i in s.items}
            for c in s.cases:
                c.items = {k: v for k, v in c.items.items() if k in known}
        if 'calibrated_at' in loads:
            ref = loads['calibrated_at']
            ref = '' if ref is None else str(ref).strip()
            hit = next((c.id for c in s.cases if c.id == ref), None) or \
                next((c.id for c in s.cases if c.name.strip().lower() == ref.lower()), None)
            if ref and ref.lower() != 'kerb' and hit is None:
                rd.warnings.append(f'loads.calibrated_at: no case "{ref}"; the cameras are taken as calibrated at '
                                   'kerb')
            s.calibrated_at = hit or ''
        if s.calibrated_at and s.case(s.calibrated_at) is None:
            s.calibrated_at = ''
        tol = d.get('tolerance') if isinstance(d.get('tolerance'), dict) else {}
        if 'mm' in tol:
            v = rd.num(tol['mm'], 'length', 'tolerance.mm', 'tol_mm')
            if v is not None:
                s.tol_mm = v
        if 'deg' in tol:
            v = rd.num(tol['deg'], 'angle', 'tolerance.deg', 'tol_deg')
            if v is not None:
                s.tol_deg = v
        if strict and rd.errors:
            raise ValueError('\n'.join(rd.errors))
        s.warnings = rd.warnings
        return s


def corner_key(key) -> str | None:
    """'FL', 'fl', 'LF', 'front left', 'Front-Left' -> 'FL'."""
    k = re.sub(r'[\s_\-]+', ' ', str(key).strip().lower())
    names = {'fl': 'FL', 'lf': 'FL', 'front left': 'FL', 'left front': 'FL', 'fr': 'FR', 'rf': 'FR',
             'front right': 'FR', 'right front': 'FR', 'rl': 'RL', 'lr': 'RL', 'rear left': 'RL', 'left rear': 'RL',
             'rr': 'RR', 'rear right': 'RR', 'right rear': 'RR'}
    return names.get(k)


def _read_points(rd, v, where, kinds, sort_on):
    """[[a, b], ...] (or {a: b}) as floats in base units, sorted on column sort_on; None on a bad table."""
    if isinstance(v, dict):
        v = [[a, b] for a, b in v.items()]
    if not isinstance(v, (list, tuple)):
        rd.errors.append(f'{where}: expected a list of [value, value] pairs')
        return None
    pts = []
    for i, p in enumerate(v):
        if isinstance(p, dict):
            p = list(p.values())
        if not isinstance(p, (list, tuple)) or len(p) < 2:
            rd.errors.append(f'{where}[{i}]: expected a pair, got {p!r}')
            return None
        a = rd.num(p[0], kinds[0], f'{where}[{i}]')
        b = rd.num(p[1], kinds[1], f'{where}[{i}]')
        if a is None or b is None:
            return None
        pts.append([a, b])
    if len(pts) > MAX_CURVE:
        rd.errors.append(f'{where}: {len(pts)} points, at most {MAX_CURVE}')
        return None
    pts.sort(key=lambda p: p[sort_on])
    return pts


def check_curve(pts, where='curve') -> str | None:
    """Why a wheel-load (N) vs travel (mm) curve cannot be used, or None: at least two points, travel and load both
    rising."""
    if len(pts) < 2:
        return f'{where}: at least two points are needed'
    for (l0, t0), (l1, t1) in zip(pts, pts[1:]):
        if not t1 > t0:
            return f'{where}: two points at the same travel ({t0:g} mm)'
        if not l1 > l0:
            return (f'{where}: the wheel load must rise with the travel ({l1:g} N at {t1:g} mm after {l0:g} N at '
                    f'{t0:g} mm)')
    return None


def check_bump_curve(pts, where='bump_curve') -> str | None:
    if len(pts) < 2:
        return f'{where}: at least two points are needed'
    for (u0, f0), (u1, f1) in zip(pts, pts[1:]):
        if not u1 > u0:
            return f'{where}: two points at the same compression ({u0:g} mm)'
        if f1 < f0:
            return f'{where}: the force must not fall as the bump stop compresses ({f1:g} N at {u1:g} mm)'
    if pts[0][0] < 0:
        return f'{where}: compression starts at contact (0 mm), not {pts[0][0]:g} mm'
    return None


def _read_corner(rd, c: Corner, d: dict, where):
    mr = d.get('motion_ratio')
    if mr is not None:
        v = rd.num(mr, 'ratio', f'{where}.motion_ratio', 'motion_ratio')
        if v is not None:
            c.motion_ratio = v
    for k in ('spring_rate', 'tyre_rate', 'kerb_load', 'ride_height', 'bump_gap', 'bump_rate'):
        if k in d and d[k] is not None:
            v = rd.num(d[k], FIELD_KIND[k], f'{where}.{k}', k)
            if v is not None:
                setattr(c, k, v)
    if d.get('wheel_rate') is not None:
        v = rd.num(d['wheel_rate'], 'rate', f'{where}.wheel_rate', 'wheel_rate')
        if v is not None:
            c.spring_rate = _clamp(v / c.motion_ratio ** 2, 'spring_rate')
    if 'bump_curve' in d:
        pts = _read_points(rd, d['bump_curve'] or [], f'{where}.bump_curve', ('length', 'force'), 0)
        if pts is not None:
            why = check_bump_curve(pts, f'{where}.bump_curve') if pts else None
            if why:
                rd.errors.append(why)
            else:
                c.bump_curve = pts
    if 'curve' in d:
        pts = _read_points(rd, d['curve'] or [], f'{where}.curve', ('force', 'length'), 1)
        if pts is not None:
            why = check_curve(pts, f'{where}.curve') if pts else None
            if why:
                rd.errors.append(why)
            else:
                c.curve = pts


def _read_items(rd, lst) -> list:
    out = []
    taken = set()
    for i, it in enumerate(lst[:MAX_ITEMS]):
        if not isinstance(it, dict):
            rd.errors.append(f'loads.items[{i}]: expected a mapping')
            continue
        name = str(it.get('name') or it.get('id') or f'Item {i + 1}')[:80]
        iid = re.sub(r'[^a-z0-9_]+', '_', str(it.get('id') or name).lower()).strip('_') or f'item_{i + 1}'
        while iid in taken:
            iid += '_2'
        kind = rd.text(it.get('kind'), f'loads.items[{i}].kind', ITEM_KINDS, 'other')
        item = LoadItem(iid, name, kind)
        if it.get('mass') is not None:
            v = rd.num(it['mass'], 'mass', f'loads.items[{i}].mass', 'item_mass')
            if v is not None:
                item.mass = v
        for k in ('x', 'y', 'z'):
            if it.get(k) is not None:
                v = rd.num(it[k], 'position', f'loads.items[{i}].{k}', 'position')
                if v is not None:
                    setattr(item, k, v)
        taken.add(iid)
        out.append(item)
    if len(lst) > MAX_ITEMS:
        rd.warnings.append(f'loads.items: {len(lst)} items, the first {MAX_ITEMS} are used')
    return out


def _read_cases(rd, lst, s) -> list:
    known = {i.id for i in s.items}
    out = []
    taken = set()
    for i, cd in enumerate(lst[:MAX_CASES]):
        if not isinstance(cd, dict):
            rd.errors.append(f'loads.cases[{i}]: expected a mapping')
            continue
        name = str(cd.get('name') or cd.get('id') or f'Case {i + 1}')[:80]
        cid = re.sub(r'[^a-z0-9_]+', '_', str(cd.get('id') or name).lower()).strip('_') or f'case_{i + 1}'
        while cid in taken:
            cid += '_2'
        raw = cd.get('items') or []
        if isinstance(raw, dict):
            pairs = list(raw.items())
        else:
            pairs = []
            for e in raw:
                if isinstance(e, dict):
                    pairs.append((e.get('id'), e.get('mass')))
                else:
                    pairs.append((e, None))
        items = {}
        for iid, m in pairs:
            iid = str(iid)
            if iid not in known:
                rd.warnings.append(f'loads.cases[{i}] ({name}): no load item "{iid}"; left out')
                continue
            if m is None or (isinstance(m, str) and not m.strip()):
                items[iid] = None
            else:
                v = rd.num(m, 'mass', f'loads.cases[{i}].items.{iid}', 'item_mass')
                items[iid] = v
        taken.add(cid)
        out.append(LoadCase(cid, name, items))
    return out


# =========================================================================== geometry from the vehicle
_GEOM_CACHE: dict = {}
_LOCK = threading.Lock()                # the caches are shared by the UI thread and background studies


def vehicle_geometry(project) -> dict:
    """Wheelbase, corner x positions and the tyre centre lines measured on the vehicle model: {'x': {corner: x},
    'track': {'front', 'rear'} (m between tyre centres, None when unknown), 'ride': {corner: mm, wheel centre to the
    wheel-arch lip, None when the model has no arch above the wheel}, 'dims': measured dimensions}. Cached per mesh
    and wheel centres."""
    veh = project.vehicle
    wc = veh.wheel_centres or {}
    mesh = veh.mesh
    key = (id(mesh), getattr(mesh, 'n_tris', 0), json.dumps(wc, sort_keys=True, default=float))
    with _LOCK:
        hit = _GEOM_CACHE.get(key)
    if hit is not None:
        return hit
    xs = {}
    for c in CORNERS:
        if c in wc:
            xs[c] = float(wc[c][0])
    if not xs:
        xs = {'FL': TYP_WHEELBASE, 'FR': TYP_WHEELBASE, 'RL': 0.0, 'RR': 0.0}
    for c in CORNERS:                                    # a missing corner mirrors its partner
        if c not in xs:
            xs[c] = xs.get(PARTNER[c], TYP_WHEELBASE if AXLE[c] == 'front' else 0.0)
    ys, ride = {}, {}
    if mesh is not None and getattr(mesh, 'n_tris', 0):
        from .mesh import ROLE_WHEELS, ROLE_MOUNTABLE, ROLE_GLASS
        wheel = mesh.part_mask({ROLE_WHEELS})
        body = mesh.part_mask({ROLE_MOUNTABLE, ROLE_GLASS})
        cen = mesh.centroids
        for c in CORNERS:
            if c not in wc:
                continue
            x, y, z = (float(v) for v in wc[c][:3])
            sel = wheel & (np.abs(cen[:, 0] - x) < 0.6) & (np.sign(cen[:, 1]) == (1.0 if y >= 0 else -1.0))
            if sel.sum() >= 4:
                v = mesh.tri[sel].reshape(-1, 3)
                yt = float(v[:, 1].min() + v[:, 1].max()) / 2
            else:
                yt = y
            ys[c] = yt
            if body.any():
                t, f = mesh.raycast(np.array([[x, yt, z]]), np.array([[0.0, 0.0, 1.0]]), faces_mask=body,
                                    max_t=1.0)
                ride[c] = round(float(t[0]) * 1000.0, 1) if f[0] >= 0 else None
    elif wc:
        ys = {c: float(wc[c][1]) for c in CORNERS if c in wc}
    track = {}
    for ax, (l, r) in AXLE_CORNERS.items():
        if l in ys and r in ys:
            track[ax] = abs(ys[l] - ys[r])
        elif l in ys or r in ys:
            track[ax] = 2 * abs(ys.get(l, ys.get(r)))
        else:
            track[ax] = None
    dims = veh.measured() if mesh is not None else {}
    out = {'x': xs, 'track': track, 'ride': {c: ride.get(c) for c in CORNERS}, 'dims': dims,
           'measured': bool(wc)}
    with _LOCK:
        if len(_GEOM_CACHE) > 16:
            _GEOM_CACHE.pop(next(iter(_GEOM_CACHE)))
        _GEOM_CACHE[key] = out
    return out


def corner_positions(project, st: SuspensionSettings) -> dict:
    """{corner: (x, y)} of the tyre contact centres, m: x from the wheel centres, y from the tracks (the settings'
    when set, else measured on the model, else the wheel centres, else typical)."""
    g = vehicle_geometry(project)
    out = {}
    for ax, (l, r) in AXLE_CORNERS.items():
        tr = getattr(st, f'track_{ax}') or g['track'][ax] or TYP_TRACK
        out[l] = (g['x'][l], tr / 2)
        out[r] = (g['x'][r], -tr / 2)
    return out


def estimate_kerb(project) -> float:
    """Kerb mass estimated from the size of the vehicle model (typical density of a car's bounding box)."""
    dims = vehicle_geometry(project)['dims']
    if all(k in dims for k in ('Length', 'Width', 'Height')):
        return round(min(max(dims['Length'] * dims['Width'] * dims['Height'] * TYP_KG_PER_M3, 400.0), 6000.0), -1)
    return TYP_KERB_KG


def seat_positions(project, driver_side='left') -> dict:
    """Estimated occupant centres of mass (x, y, z, m) per seat, from the model's size and wheelbase: front seats at
    43 % of the wheelbase ahead of the rear axle, rear seats at 8 %, front occupants 23.5 % of the width either side of
    the centre line, rear outboard 22 %, at a third of the height. Typical of compact cars; edit them for yours."""
    g = vehicle_geometry(project)
    dims = g['dims']
    xr = (g['x']['RL'] + g['x']['RR']) / 2
    wb = max((g['x']['FL'] + g['x']['FR']) / 2 - xr, 0.5)
    W = dims.get('Width', 1.75)
    H = dims.get('Height', 1.5)
    rear_x = dims.get('rear_x', xr - 0.8)
    zs = round(0.34 * H, 3)
    sd = 1.0 if driver_side == 'left' else -1.0
    fy, ry = round(0.235 * W, 3), round(0.22 * W, 3)
    xf, xb = round(xr + 0.43 * wb, 3), round(xr + 0.08 * wb, 3)
    boot_x = round(xr + 0.5 * (rear_x - xr), 3)
    return {'driver': (xf, sd * fy, zs), 'passenger': (xf, -sd * fy, zs), 'rear_left': (xb, ry, zs),
            'rear_centre': (xb, 0.0, zs), 'rear_right': (xb, -ry, zs),
            'luggage': (boot_x, 0.0, round(0.5 * H, 3)), 'roof': (round(xr + 0.45 * wb, 3), 0.0, round(H + 0.1, 3)),
            'computer': (round(xr + 0.45 * (rear_x - xr), 3), 0.25, round(0.4 * H, 3)),
            'battery': (round(xr + 0.45 * (rear_x - xr), 3), -0.25, round(0.4 * H, 3))}


PRESETS = (  # id, name, kind, mass (None = the occupant mass)
    ('driver', 'Driver', 'occupant', None), ('passenger', 'Front passenger', 'occupant', None),
    ('rear_left', 'Rear left', 'occupant', None), ('rear_centre', 'Rear centre', 'occupant', None),
    ('rear_right', 'Rear right', 'occupant', None), ('luggage', 'Boot luggage', 'luggage', 40.0),
    ('roof', 'Roof load', 'roof', 30.0), ('computer', 'Recording computer and disks', 'equipment', 8.0),
    ('battery', 'Auxiliary battery and DC-DC', 'equipment', 15.0))
PRESET_NAME = {p[0]: p[1] for p in PRESETS}


def preset_item(project, st: SuspensionSettings, pid: str) -> LoadItem:
    """A new load item from a preset (a seat, the boot, the roof, the recording equipment), placed on the vehicle."""
    pr = next(p for p in PRESETS if p[0] == pid)
    x, y, z = seat_positions(project, st.driver_side)[pid]
    return LoadItem(st.new_id(pid), pr[1], pr[2], st.occupant_mass if pr[3] is None else pr[3], x, y, z)


def typical(project=None, driver_side='left') -> SuspensionSettings:
    """Typical settings for the project's vehicle (a compact sedan's figures scaled to the model's estimated kerb
    mass), with the rig, the seats, the boot and the roof as load items and five cases calibrated with the rig only."""
    kerb = estimate_kerb(project) if project is not None else TYP_KERB_KG
    s = SuspensionSettings(corners=_typical_corners(kerb), driver_side='right' if driver_side == 'right' else 'left')
    if project is not None:
        ride = vehicle_geometry(project)['ride']
        for c in CORNERS:
            if ride.get(c):
                s.corners[c].ride_height = ride[c]
    s.items = [LoadItem('rig', 'Rig: cameras, sensors, rig parts, ECU', 'rig')]
    if project is not None:
        s.items += [preset_item(project, s, pid) for pid, *_ in PRESETS]
    else:
        s.items += [LoadItem(pid, name, kind, OCCUPANT_KG if m is None else m) for pid, name, kind, m in PRESETS]
    rig = {'rig': None, 'computer': None, 'battery': None}
    s.cases = [LoadCase('cal', 'Rig only, no occupants', dict(rig)),
               LoadCase('driver', 'Driver', dict(rig, driver=None)),
               LoadCase('crew', 'Driver + operator', dict(rig, driver=None, passenger=None)),
               LoadCase('four', '4 occupants', dict(rig, driver=None, passenger=None, rear_left=None, rear_right=None)),
               LoadCase('full', 'Full: 5 occupants + boot', dict(rig, driver=None, passenger=None, rear_left=None,
                                                                rear_centre=None, rear_right=None, luggage=None))]
    s.calibrated_at = 'cal'
    s.source = 'typical'
    return s


def settings_for(project) -> SuspensionSettings:
    """The project's suspension settings: analysis.suspension over the typical settings for its vehicle."""
    d = getattr(project.analysis, 'suspension', None) or {}
    veh = d.get('vehicle') if isinstance(d, dict) and isinstance(d.get('vehicle'), dict) else {}
    side = str(veh.get('driver_side', 'left')).strip().lower()
    return SuspensionSettings.from_dict(d, base=typical(project, side))


# =========================================================================== the rig's masses
def _box_shell(l, w, h):
    solid = l * w * h
    shell = 2 * (l * w + l * h + w * h) * RIG_WALL_M
    return min(solid, shell) * ALU_KG_M3


def rig_part_mass(part) -> float:
    """A rig part as a 3 mm aluminium shell (capped at the solid): a typical estimate."""
    if part.kind == 'box':
        return _box_shell(max(part.length, 1e-3), max(part.width, 1e-3), max(part.height, 1e-3))
    r = max(part.radius, 1e-3)
    L = max(part.height if part.kind == 'cylinder' else part.length, 1e-3)
    solid = math.pi * r * r * L
    shell = (2 * math.pi * r * L + 2 * math.pi * r * r) * RIG_WALL_M
    return min(solid, shell) * ALU_KG_M3


def sensor_mass(model) -> float:
    """A sensor's typical mass: its housing volume (library size) at a typical density for its kind, plus a mount."""
    try:
        dx, dy, h = (max(float(v), 1.0) / 10.0 for v in model.size_mm)            # cm
    except (TypeError, ValueError):
        dx = dy = h = 8.0
    round_ = model.kind == 'gnss' or getattr(model, 'scan', '') in ('spinning', 'nonrepetitive')
    vol = math.pi * dx * dy / 4 * h if round_ else dx * dy * h                       # cm³
    kind = model.kind if model.kind in TYP_DENSITY else 'radar'
    return max(vol * TYP_DENSITY[kind] / 1000.0, 0.05) + TYP_MOUNT_KG[kind]


def rig_components(project, st: SuspensionSettings | None = None, variant=None) -> list[dict]:
    """Every visible camera, sensor and rig part of the variant and the ECU, with its mass (kg) and where it is:
    [{'uid', 'tag', 'name', 'kind', 'mass', 'estimate' (typical or module figure), 'source', 'x', 'y', 'z'}]. A mass
    in st.rig_masses (by uid, else tag) overrides the estimate."""
    st = st or settings_for(project)
    v = variant or project.variant
    over = st.rig_masses or {}
    out = []

    def add(uid, tag, name, kind, est, src, pos):
        try:
            pos = [float(v) for v in pos]
        except (TypeError, ValueError):
            return
        if not all(math.isfinite(v) for v in pos):
            return                                       # a broken position cannot carry a load
        m = over.get(uid, over.get(tag))
        out.append({'uid': uid, 'tag': tag, 'name': name, 'kind': kind, 'estimate': est,
                    'mass': float(m) if m is not None else est, 'source': 'entered' if m is not None else src,
                    'x': float(pos[0]), 'y': float(pos[1]), 'z': float(pos[2])})
    for c in v.cameras:
        if not c.visible:
            continue
        mod = project.module_for(c)
        try:
            w = float(mod.weight_g) / 1000.0
        except (TypeError, ValueError):
            w = float('nan')
        if math.isfinite(w) and w > 0:
            add(c.uid, c.tag, f'{c.name} ({mod.name})', 'camera', w + CAMERA_BRACKET_KG,
                'module weight + typical bracket', c.pos())
        else:
            add(c.uid, c.tag, f'{c.name} ({mod.name})', 'camera', CAMERA_FALLBACK_KG, 'typical', c.pos())
    for s in v.sensors:
        if not s.visible:
            continue
        m = project.sensor_model_for(s)
        add(s.uid, s.tag, f'{m.vendor} {m.name}'.strip(), s.kind, round(sensor_mass(m), 3),
            'typical (housing size)', s.pos())
    for p in getattr(project, 'rig', []) or []:
        if not p.visible:
            continue
        add(p.uid, p.name, p.name, 'rig part', round(rig_part_mass(p), 3), 'typical (3 mm aluminium)',
            (p.x, p.y, p.z))
    e = project.ecu
    add('ecu', 'ECU', e.name or 'ECU', 'ecu', ECU_KG, 'typical', (e.x, e.y, e.z))
    return out


def case_loads(project, st: SuspensionSettings, case: LoadCase, comps=None) -> list[tuple]:
    """[(name, kg, x, y, z)] of a case: its items with their masses in this case; the rig item as every component."""
    out = []
    for iid, m in case.items.items():
        it = st.item(iid)
        if it is None:
            continue
        if it.kind == 'rig':
            comps = comps if comps is not None else rig_components(project, st)
            tot = sum(c['mass'] for c in comps)
            k = (float(m) / tot) if (m is not None and tot > 0) else 1.0    # a case's rig mass scales every component
            out += [(f'{it.name}: {c["tag"]}', c['mass'] * k, c['x'], c['y'], c['z']) for c in comps]
        else:
            mass = it.mass if m is None else float(m)
            out.append((it.name, mass, it.x, it.y, it.z))
    return out


def item_mass(project, st, it: LoadItem, comps=None):
    """(mass kg, (x, y, z)) of an item; the rig item's from its components."""
    if it.kind != 'rig':
        return it.mass, (it.x, it.y, it.z)
    comps = comps if comps is not None else rig_components(project, st)
    m = sum(c['mass'] for c in comps)
    if m <= 0:
        return 0.0, (0.0, 0.0, 0.0)
    return m, tuple(sum(c['mass'] * c[k] for c in comps) / m for k in ('x', 'y', 'z'))


# =========================================================================== the static solve
def _interp(x, xp, fp):
    """Linear interpolation with linear extrapolation from the end segments; (value, slope)."""
    xp, fp = np.asarray(xp, float), np.asarray(fp, float)
    i = int(np.clip(np.searchsorted(xp, x) - 1, 0, len(xp) - 2))
    k = (fp[i + 1] - fp[i]) / (xp[i + 1] - xp[i])
    return float(fp[i] + k * (x - xp[i])), float(k)


class _CornerLaw:
    """The suspension force a corner adds at a wheel travel s from kerb (N, positive = bump) and its slope."""

    def __init__(self, c: Corner):
        self.c = c
        self.kerb_n = c.kerb_load * G
        if c.curve:
            pts = np.asarray(c.curve, float)
            self.L, self.T = pts[:, 0], pts[:, 1]
            self.s0 = _interp(self.kerb_n, self.L, self.T)[0]       # travel on the curve at the kerb load
            self.range = (float(self.T[0]) - self.s0, float(self.T[-1]) - self.s0)
        if c.bump_curve:
            b = np.asarray(c.bump_curve, float)
            self.bu, self.bf = b[:, 0], b[:, 1] - b[0, 1]

    def force(self, s):
        c = self.c
        if c.curve:
            f, k = _interp(self.s0 + s, self.T, self.L)
            return f - self.kerb_n, k
        f, k = c.wheel_rate * s, c.wheel_rate
        u = s - c.bump_gap
        if c.has_bump and u > 0:
            if c.bump_curve:
                bf, bk = _interp(u, self.bu, self.bf)
                f, k = f + max(bf, 0.0), k + (bk if bf > 0 else 0.0)
            else:
                f, k = f + c.bump_rate * u, k + c.bump_rate
        return f, k


@dataclass
class CornerState:
    added_n: float          # tyre force the load adds, N
    load_kg: float          # wheel load with the kerb load, kg
    travel_mm: float        # suspension travel from kerb, bump +
    tyre_mm: float          # tyre deflection the load adds
    drop_mm: float          # the body's drop at the corner: travel + tyre
    ride_mm: float | None   # loaded ride height (wheel centre to arch), None when the design one is not given
    bump_mm: float          # bump-stop compression (0 = clear)
    on_bump: bool
    lifted: bool            # the wheel would carry no load: the solve assumes four wheels on the ground
    beyond_curve: bool      # the travel is outside the measured curve (extrapolated)


@dataclass
class Statics:
    added_kg: float
    total_kg: float
    cog: tuple              # centre of gravity of the added load (x, y, z), m
    corners: dict           # corner -> CornerState
    heave_mm: float         # drop at the rear-axle centre
    pitch_deg: float        # positive = nose down
    roll_deg: float         # positive = right side lower
    iterations: int
    converged: bool

    @property
    def bump(self) -> list:
        return [c for c in CORNERS if self.corners[c].on_bump]


def solve(st: SuspensionSettings, xy: dict, loads, ground=None, moment=None, link=None) -> Statics:
    """Static solve of the added loads [(name, kg, x, y, z)] on the four corners at xy {corner: (x, y)}: Newton's method
    on the seven equations (vertical force and the moments about x and y, and each corner's spring in series with its
    tyre); unknowns heave w, slopes b (mm/m along x) and c (mm/m along y), and the four suspension travels.

    Optional, for a road that is not flat and for the quasi-static load transfer of braking, accelerating and
    cornering (see transfer()):
    - ground: {corner: mm} the road under each tyre raised (+) or lowered from flat; the tyre deflects by the body's
      drop plus the rise. A road that is one plane under the four tyres moves the body with it and loads no spring.
    - moment: (Mx, My) N·m added to the loads' Σ weight × x and Σ weight × y (an inertial load transfer moves load
      between the tyres without adding any).
    - link: {corner: N} the part of each tyre's added force that bypasses the spring (taken by the suspension linkage:
      anti-dive, anti-squat, the roll centres; and the unsprung masses' own load transfer).
    Without them the solve is exactly the static one above."""
    loads = [(n, float(m), float(x), float(y), float(z)) for n, m, x, y, z in loads
             if math.isfinite(float(m)) and float(m) != 0.0]
    W = np.array([m * G for _, m, *_ in loads])
    Wt = float(W.sum())
    Mx = float(sum(w * l[2] for w, l in zip(W, loads)))
    My = float(sum(w * l[3] for w, l in zip(W, loads)))
    X = np.array([xy[c][0] for c in CORNERS])
    Y = np.array([xy[c][1] for c in CORNERS])
    kt = np.array([st.corners[c].tyre_rate for c in CORNERS])
    laws = [_CornerLaw(st.corners[c]) for c in CORNERS]
    arb = np.array([st.arb(AXLE[c]) for c in CORNERS])
    pi = [CORNERS.index(PARTNER[c]) for c in CORNERS]
    g = None if ground is None else np.array([float(ground.get(c, 0.0)) for c in CORNERS])
    lk = None if link is None else np.array([float(link.get(c, 0.0)) for c in CORNERS])
    extra = 0.0
    if moment is not None:
        Mx += float(moment[0])
        My += float(moment[1])
        extra += abs(float(moment[0])) + abs(float(moment[1]))
    if g is not None:
        extra += float(np.abs(kt * g).sum())
    if lk is not None:
        extra += float(np.abs(lk).sum())

    def residual(u):
        w, b, cc = u[:3]
        s = u[3:]
        D = w + b * X + cc * Y
        Ft = kt * ((D if g is None else D + g) - s)
        r = np.empty(7)
        r[0] = Ft.sum() - Wt
        r[1] = (Ft * X).sum() - Mx
        r[2] = (Ft * Y).sum() - My
        J = np.zeros((7, 7))
        J[0, :3] = [kt.sum(), (kt * X).sum(), (kt * Y).sum()]
        J[1, :3] = [(kt * X).sum(), (kt * X * X).sum(), (kt * X * Y).sum()]
        J[2, :3] = [(kt * Y).sum(), (kt * X * Y).sum(), (kt * Y * Y).sum()]
        J[0, 3:], J[1, 3:], J[2, 3:] = -kt, -kt * X, -kt * Y
        for i in range(4):
            f, k = laws[i].force(s[i])
            j = pi[i]
            a = arb[i] * (s[i] - s[j]) / 2.0
            r[3 + i] = Ft[i] - f - a
            if lk is not None:
                r[3 + i] -= lk[i]
            J[3 + i, :3] = [kt[i], kt[i] * X[i], kt[i] * Y[i]]
            J[3 + i, 3 + i] = -kt[i] - k - arb[i] / 2.0
            J[3 + i, 3 + j] = arb[i] / 2.0
        return r, J

    u = np.zeros(7)
    tol = 1e-7 * (abs(Wt) + 1.0) + 1e-9 * extra
    r, J = residual(u)
    nr = float(np.linalg.norm(r))
    it = 0
    converged = nr < tol
    while not converged and it < 100:
        it += 1
        try:
            du = np.linalg.solve(J, -r)
        except np.linalg.LinAlgError:
            du = np.linalg.lstsq(J, -r, rcond=None)[0]
        step = 1.0
        while True:
            r2, J2 = residual(u + step * du)
            n2 = float(np.linalg.norm(r2))
            if n2 < (1 - 1e-4 * step) * nr or step < 1e-6:
                break
            step /= 2
        u = u + step * du
        r, J, nr = r2, J2, n2
        converged = nr < tol
    w, b, cc = (float(v) for v in u[:3])
    s = u[3:]
    D = w + b * X + cc * Y
    Dg = D if g is None else D + g              # the tyre's deflection is the body's drop plus the road's rise
    Ft = kt * (Dg - s)
    corners = {}
    for i, c in enumerate(CORNERS):
        co = st.corners[c]
        law = laws[i]
        si = float(s[i])
        bump = max(si - co.bump_gap, 0.0) if co.has_bump else 0.0
        beyond = bool(co.curve) and not (law.range[0] - 1e-9 <= si <= law.range[1] + 1e-9)
        load = float(co.kerb_load + Ft[i] / G)
        corners[c] = CornerState(float(Ft[i]), load, si, float(Dg[i]) - si, float(D[i]),
                                 (co.ride_height - si) if co.ride_height > 0 else None, bump,
                                 bool(bump > 1e-9), bool(load <= 0.0), bool(beyond))
    added = sum(m for _, m, *_ in loads)
    cog = tuple(sum(ld[1] * ld[k] for ld in loads) / added for k in (2, 3, 4)) if added else (0.0, 0.0, 0.0)
    return Statics(added, st.kerb_mass + added, cog, corners, w, math.degrees(math.atan(b / 1000.0)),
                   -math.degrees(math.atan(cc / 1000.0)), it, converged)


# =========================================================================== braking, accelerating, cornering
# The quasi-static load transfer of a steady acceleration (no dampers, no transients): the inertial force of the
# vehicle at its centre of gravity moves load between the tyres (the pitch moment M·a_x·h, the roll moment M·a_y·h). Of
# that load, the unsprung masses' share and the share the suspension linkage takes (anti-dive and anti-lift under
# braking, anti-squat or anti-lift at the driven axle under acceleration, the roll centres in a turn) reaches the tyres
# without deflecting the springs; the rest deflects the springs (and the anti-roll bars), and all of it the tyres.
# Typical figures, every one labelled typical in the UI:
# - centre of gravity: passenger cars carry it at roughly 0.35-0.40 of their overall height (0.5-0.6 m on a car
#   1.4-1.6 m tall); 0.37 is used unless the height is entered.
# - anti-dive 20 % front, anti-lift 10 % rear under braking, 15 % at the driven axle under acceleration: passenger cars
#   mostly stay well under 50 % (more makes the brakes feel wooden; Milliken & Milliken, Race Car Vehicle Dynamics, SAE
#   1995, ch. 17; Gillespie, Fundamentals of Vehicle Dynamics, SAE 1992, ch. 9).
# - roll centres 0.08 m front (MacPherson strut), 0.20 m rear (twist beam): typical of compact front-driven cars.
# - unsprung mass 35 kg per corner (TYP_UNSPRUNG_KG) at the wheel centre's height.
TYP_CG_FRACTION = 0.37
TYP_CG_M = 0.55
TYP_WHEEL_R = 0.31
TYP_ANTI = {'dive': 20.0, 'lift': 10.0, 'squat': 15.0}
TYP_RC = {'front': 0.08, 'rear': 0.20}
DRIVES = ('front', 'rear', 'awd')
DRIVE_LABEL = {'front': 'Front-wheel drive', 'rear': 'Rear-wheel drive', 'awd': 'All-wheel drive (50/50)'}
DYN_LIMITS = {'cg_height': (0.0, 3.0), 'anti_dive': (0.0, 100.0), 'anti_lift': (0.0, 100.0),
              'anti_squat': (0.0, 100.0), 'rc_front': (-0.5, 1.5), 'rc_rear': (-0.5, 1.5), 'unsprung_kg': (0.0, 300.0)}
DYN_ASSUMPTIONS = ('Quasi-static load transfer of a steady acceleration on the static suspension model: the pitch '
                   'moment M·a_x·h and the roll moment M·a_y·h of the whole vehicle at its centre of gravity move load '
                   'between the tyres; the unsprung masses\' share and the linkage\'s share (anti-dive, anti-lift, '
                   'anti-squat as percentages of the axle\'s load transfer; the roll centres in a turn) bypass the '
                   'springs. No dampers or transients (the first overshoot of a hard stop is larger), no tyre lateral '
                   'compliance, no aerodynamic loads, no body flex; the rig moves rigidly with the body.')


@dataclass
class Dynamics:
    """Inputs of the load transfer under braking, accelerating and cornering (typical defaults)."""
    cg_height: float = 0.0                  # m, the empty (kerb) vehicle's centre of gravity above the road; 0 = estimate
    anti_dive: float = TYP_ANTI['dive']     # % of the front axle's braking load transfer its linkage takes
    anti_lift: float = TYP_ANTI['lift']     # % of the rear axle's braking load transfer its linkage takes
    anti_squat: float = TYP_ANTI['squat']   # % of the driven axle's load transfer under acceleration its linkage takes
    drive: str = 'front'                    # front | rear | awd (50/50)
    rc_front: float = TYP_RC['front']       # m, roll-centre height
    rc_rear: float = TYP_RC['rear']
    unsprung_kg: float = TYP_UNSPRUNG_KG    # per corner

    def to_dict(self):
        return {'cg_height': self.cg_height, 'anti_dive': self.anti_dive, 'anti_lift': self.anti_lift,
                'anti_squat': self.anti_squat, 'drive': self.drive, 'rc_front': self.rc_front,
                'rc_rear': self.rc_rear, 'unsprung_kg': self.unsprung_kg}

    @staticmethod
    def from_dict(d) -> 'Dynamics':
        out = Dynamics()
        d = d if isinstance(d, dict) else {}
        if str(d.get('drive', '')).strip().lower() in DRIVES:
            out.drive = str(d['drive']).strip().lower()
        for k, (lo, hi) in DYN_LIMITS.items():
            if k in d:
                try:
                    v = float(d[k])
                except (TypeError, ValueError):
                    continue
                if math.isfinite(v):
                    setattr(out, k, min(max(v, lo), hi))
        return out


def cg_height_estimate(project) -> float:
    """The empty vehicle's centre of gravity above the road, m: TYP_CG_FRACTION of the model's height (typical)."""
    dims = vehicle_geometry(project)['dims']
    h = dims.get('Height')
    return round(TYP_CG_FRACTION * float(h), 3) if h and math.isfinite(float(h)) and h > 0.3 else TYP_CG_M


def wheel_radius(project) -> float:
    """The tyres' loaded radius, m: the wheel centres' height on the model, else typical."""
    wc = project.vehicle.wheel_centres or {}
    zs = [float(v[2]) for v in wc.values() if len(v) > 2 and math.isfinite(float(v[2])) and float(v[2]) > 0.1]
    return float(np.mean(zs)) if zs else TYP_WHEEL_R


def transfer(st: SuspensionSettings, xy: dict, loads, ax_g: float, ay_g: float, dyn: Dynamics,
             cg_height: float | None = None, wheel_r: float = TYP_WHEEL_R) -> dict:
    """The quasi-static load transfer of a longitudinal (ax_g, g, + accelerating, - braking) and lateral (ay_g, g, +
    turning left, so the body rolls right side down) acceleration with the loads [(name, kg, x, y, z)] on board:
    {'moment': (Mx, My) for solve, 'link': {corner: N} for solve, 'mass_kg', 'cog': (x, y, z) of the loaded vehicle,
    'sprung_kg', 'sprung_cg_z', 'front_share' (of the sprung mass), 'axle_n' (load the front axle gains), 'anti':
    (front, rear) shares the linkage takes, 'roll_axis_z' (under the sprung centre of gravity)}. cg_height: the empty
    vehicle's (dyn.cg_height when set, else this, else typical)."""
    h_k = dyn.cg_height if dyn.cg_height > 0 else (cg_height if cg_height and cg_height > 0 else TYP_CG_M)
    X = {c: float(xy[c][0]) for c in CORNERS}
    Y = {c: float(xy[c][1]) for c in CORNERS}
    Mk = st.kerb_mass
    xk = sum(st.corners[c].kerb_load * X[c] for c in CORNERS) / Mk if Mk > 0 else 0.0
    yk = sum(st.corners[c].kerb_load * Y[c] for c in CORNERS) / Mk if Mk > 0 else 0.0
    ld = [(float(m), float(x), float(y), float(z)) for _, m, x, y, z in loads
          if math.isfinite(float(m)) and float(m) != 0.0]
    M = Mk + sum(m for m, *_ in ld)
    xf, xr = (X['FL'] + X['FR']) / 2, (X['RL'] + X['RR']) / 2
    L = max(xf - xr, 0.5)
    Tf, Tr = max(Y['FL'] - Y['FR'], 0.3), max(Y['RL'] - Y['RR'], 0.3)
    if M <= 0:
        return {'moment': (0.0, 0.0), 'link': {c: 0.0 for c in CORNERS}, 'mass_kg': 0.0, 'cog': (0.0, 0.0, 0.0),
                'sprung_kg': 0.0, 'sprung_cg_z': 0.0, 'front_share': 0.5, 'axle_n': 0.0, 'anti': (0.0, 0.0),
                'roll_axis_z': 0.0}
    xc = (Mk * xk + sum(m * x for m, x, _, _ in ld)) / M
    yc = (Mk * yk + sum(m * y for m, _, y, _ in ld)) / M
    zc = (Mk * h_k + sum(m * z for m, _, _, z in ld)) / M
    mu = min(max(dyn.unsprung_kg, 0.0), 0.2 * M)          # per corner; the sprung mass keeps at least 20 %
    Mu = 4.0 * mu
    ms = M - Mu
    hu = max(float(wheel_r), 0.0)
    hs = (M * zc - Mu * hu) / ms
    xs = (M * xc - mu * sum(X.values())) / ms
    fs = min(max((xs - xr) / L, 0.0), 1.0)
    ax, ay = float(ax_g) * G, float(ay_g) * G             # m/s²
    Mx, My = -M * ax * zc, -M * ay * zc                    # N·m: Σ tyre force × x (and × y) the transfer adds
    dF = Mx / L                                            # the front axle gains, the rear loses
    dFu = -Mu * ax * hu / L                                # of which the unsprung masses' share
    dFs = dF - dFu
    if ax_g < 0:
        af, ar = dyn.anti_dive / 100.0, dyn.anti_lift / 100.0
    elif ax_g > 0:
        share = {'front': 1.0, 'rear': 0.0, 'awd': 0.5}.get(dyn.drive, 1.0)    # of the tractive force at the front
        af, ar = dyn.anti_squat / 100.0 * share, dyn.anti_squat / 100.0 * (1.0 - share)
    else:
        af = ar = 0.0
    link_f = dFu + af * dFs
    link_r = -dFu - ar * dFs
    lat_f = (2.0 * mu * hu + ms * fs * dyn.rc_front) * ay / Tf      # the right tyre gains, the left loses
    lat_r = (2.0 * mu * hu + ms * (1.0 - fs) * dyn.rc_rear) * ay / Tr
    link = {'FL': link_f / 2 - lat_f, 'FR': link_f / 2 + lat_f, 'RL': link_r / 2 - lat_r, 'RR': link_r / 2 + lat_r}
    return {'moment': (Mx, My), 'link': link, 'mass_kg': M, 'cog': (xc, yc, zc), 'sprung_kg': ms,
            'sprung_cg_z': hs, 'front_share': fs, 'axle_n': dF, 'anti': (af, ar),
            'roll_axis_z': fs * dyn.rc_front + (1.0 - fs) * dyn.rc_rear}


def dynamic_solve(st: SuspensionSettings, xy: dict, loads, ax_g: float, ay_g: float, dyn: Dynamics,
                  cg_height: float | None = None, wheel_r: float = TYP_WHEEL_R) -> tuple[Statics, dict]:
    """The body's attitude with the loads on board under a steady acceleration (ax_g, ay_g in g; see transfer):
    (Statics from kerb, the transfer's figures)."""
    tr = transfer(st, xy, loads, ax_g, ay_g, dyn, cg_height, wheel_r)
    if not ax_g and not ay_g:
        return solve(st, xy, loads), tr
    return solve(st, xy, loads, moment=tr['moment'], link=tr['link']), tr


# =========================================================================== cases
@dataclass
class CaseResult:
    id: str
    name: str
    loads: list               # [(name, kg, x, y, z)]
    statics: Statics
    rel: dict                 # heave_mm, pitch_deg, roll_deg relative to the calibration load
    devices: list             # loadcases.device_moves relative to the calibration load
    worst_mm: float           # the largest device move relative to the calibration load
    turn_deg: float           # the body's rotation relative to the calibration load
    over_gross: float         # kg over the gross vehicle mass (0 = within or not checked)
    image: object = None      # loadcases.LoadResult after evaluate

    @property
    def worst_device(self):
        return max(self.devices, key=lambda d: d['move_mm']) if self.devices else None


@dataclass
class SuspensionResult:
    settings: SuspensionSettings
    cases: list               # CaseResult, in the settings' order
    calibration: str          # the name of the load the cameras are calibrated at
    calibration_statics: Statics
    xy: dict
    components: list
    findings: list
    variant: str = ''
    variant_uid: str = ''
    images: bool = False      # the ground figures were evaluated
    image_note: str = ''

    def case(self, cid):
        return next((c for c in self.cases if c.id == cid), None)

    def worst(self) -> str:
        return self.findings[0]['severity'] if self.findings else 'good'


def _project_for(project, variant):
    if variant is None or variant is project.variant:
        return project
    q = copy.copy(project)
    q.variants = list(project.variants) + ([] if any(v is variant for v in project.variants) else [variant])
    q.active_uid = variant.uid
    return q


def solve_cases(project, st: SuspensionSettings | None = None, variant=None) -> SuspensionResult:
    """The static solve of every case, the calibration load and each case's device moves relative to it (cheap: no
    ground analysis)."""
    from . import loadcases as LC
    p = _project_for(project, variant)
    st = st or settings_for(p)
    xy = corner_positions(p, st)
    comps = rig_components(p, st)
    cal = st.case(st.calibrated_at)
    zero = solve(st, xy, [])
    cal_stat = solve(st, xy, case_loads(p, st, cal, comps)) if cal is not None else zero
    R0, t0 = LC.pose(cal_stat.heave_mm, cal_stat.pitch_deg, cal_stat.roll_deg)
    out = []
    for case in st.cases:
        loads = case_loads(p, st, case, comps)
        sv = cal_stat if (cal is not None and case.id == cal.id) else solve(st, xy, loads)
        R, t = LC.pose(sv.heave_mm, sv.pitch_deg, sv.roll_deg)
        dev = LC.device_moves(p, R, t, R0, t0)
        dR, _ = LC.relative(R, t, R0, t0)
        _, rp, rr = euler_zyx(dR)
        rel = {'heave_mm': sv.heave_mm - cal_stat.heave_mm, 'pitch_deg': rp, 'roll_deg': rr}
        over = max(sv.total_kg - st.gross_mass, 0.0) if st.gross_mass > 0 else 0.0
        out.append(CaseResult(case.id, case.name, loads, sv, rel, dev,
                              max((d['move_mm'] for d in dev), default=0.0), LC.turn_deg(dR), over))
    v = p.variant
    res = SuspensionResult(st, out, cal.name if cal is not None else 'kerb (empty, as modelled)', cal_stat, xy, comps,
                           [], v.name, v.uid)
    res.findings = _findings(res)
    return res


def corner_cases(res: SuspensionResult) -> list:
    """loadcases.CornerCase for every case of a solve, referenced to its calibration load."""
    from . import loadcases as LC
    cs = res.calibration_statics
    ref = (cs.heave_mm, cs.pitch_deg, cs.roll_deg) if res.settings.calibrated_at else None
    return [LC.CornerCase(c.name, c.statics.heave_mm, c.statics.pitch_deg, c.statics.roll_deg,
                          {k: c.statics.corners[k].drop_mm for k in CORNERS}, ref, res.calibration, c.statics.bump)
            for c in res.cases]


def run(project, engine=None, settings: SuspensionSettings | None = None, variant=None, progress=None, cancel=None,
        cell=None, images=True) -> SuspensionResult | None:
    """Every case: the static solve, the device moves and (images, with a vehicle model and cameras) the ground
    analysis with the body moved (blind area, ground shift, seam ghost). Returns None when cancelled."""
    from . import loadcases as LC
    p = _project_for(project, variant)
    res = solve_cases(p, settings)
    if progress is not None:
        progress(0.05)
    if images:
        cams = [c for c in p.variant.cameras if c.visible]
        if p.vehicle.mesh is None or not cams or not res.cases:
            res.image_note = 'No ground figures: they need a vehicle model, cameras and at least one case.'
        else:
            if engine is None:
                from .analysis import Engine
                engine = Engine()
            cell = cell or max(float(p.analysis.cell_m), 0.05)
            lr = LC.evaluate(engine, p, corner_cases(res), cell=cell, cancel=cancel,
                             progress=(lambda f: progress(0.05 + 0.95 * f)) if progress is not None else None)
            if lr is None:
                return None
            if lr:
                for c, r in zip(res.cases, lr):
                    c.image = r
                res.images = True
            else:
                res.image_note = 'No ground figures: the region of interest is empty.'
            res.findings = _findings(res)
    elif cancel is not None and cancel():
        return None
    if progress is not None:
        progress(1.0)
    return res


# =========================================================================== findings and issues
def _finding(out, sev, kind, message, case=None, uids=()):
    out.append({'severity': sev, 'kind': kind, 'message': message, 'case': case, 'uids': list(uids)})


def _corner_list(cs):
    return ', '.join(cs)


def _findings(res: SuspensionResult) -> list:
    st = res.settings
    out = []
    if st.is_typical:
        _finding(out, 'note', 'typical', TYPICAL_NOTE)
    for c in res.cases:
        sv = c.statics
        if not sv.converged:
            _finding(out, 'warning', 'solve', f'{c.name}: the static solve did not converge; check the curves and '
                                              'rates.', c.id)
        lifted = [k for k in CORNERS if sv.corners[k].lifted]
        if lifted:
            _finding(out, 'serious', 'lift', f'{c.name}: the {_corner_list(lifted)} wheel would lift (no load left on '
                                             'it); the figures assume four wheels on the ground.', c.id)
        if c.over_gross > 0:
            _finding(out, 'serious', 'gross', f'{c.name}: {sv.total_kg:,.0f} kg is {c.over_gross:,.0f} kg over the '
                                              f'{st.gross_mass:,.0f} kg gross vehicle mass.', c.id)
    bumped = [c for c in res.cases if c.statics.bump]
    if bumped:
        parts = []
        for c in bumped:
            sv = c.statics
            k = max(sv.bump, key=lambda x: sv.corners[x].bump_mm)
            parts.append(f'{c.name} ({_corner_list(sv.bump)}, {sv.corners[k].bump_mm:.1f} mm into the {k} stop)')
        _finding(out, 'warning', 'bump', 'Bump stops touched at static load: ' + '; '.join(parts) + '. The body sits '
                 'on the stops, so the cameras move abruptly over bumps.', bumped[-1].id)
    beyond = [c for c in res.cases if any(c.statics.corners[k].beyond_curve for k in CORNERS)]
    if beyond:
        _finding(out, 'warning', 'curve', 'Outside the measured curves (extrapolated from their ends): ' + '; '.join(
            f'{c.name} ({_corner_list([k for k in CORNERS if c.statics.corners[k].beyond_curve])})' for c in beyond)
            + '.', beyond[-1].id)
    over = [c for c in res.cases if c.worst_mm > st.tol_mm or c.turn_deg > st.tol_deg]
    if over:
        w = max(over, key=lambda c: (c.worst_mm / st.tol_mm, c.turn_deg / st.tol_deg))
        d = w.worst_device
        _finding(out, 'warning', 'calibration',
                 f'{len(over)} of {len(res.cases)} cases move the rig beyond the {st.tol_mm:g} mm / {st.tol_deg:g}° '
                 f'calibration tolerance from "{res.calibration}": worst {w.name}, {d["tag"] if d else "—"} '
                 f'{w.worst_mm:.0f} mm and the body {w.turn_deg:.2f}° (pitch {w.rel["pitch_deg"]:+.2f}°, roll '
                 f'{w.rel["roll_deg"]:+.2f}°). Calibrate at the load you record with, or log the load per drive.',
                 w.id, [x['uid'] for x in (w.devices if d else [])
                        if x['move_mm'] > st.tol_mm])
    if res.images:
        bad = [c for c in res.cases if c.image is not None and c.image.status != 'good']
        if bad:
            w = max(bad, key=lambda c: c.image.ground_shift_cm)
            _finding(out, 'serious' if any(c.image.status == 'serious' for c in bad) else 'warning', 'ghost',
                     f'Against the calibration at "{res.calibration}" the seams ghost by ' +
                     ', '.join(f'{c.image.seam_ghost_cm:.1f} cm ({c.name})' for c in bad) +
                     f'; the stitched ground shifts up to {w.image.ground_shift_cm:.1f} cm ({w.name}).',
                     max(bad, key=lambda c: c.image.seam_ghost_cm).id)
    if not any(f['severity'] in ISSUE_SEVERITIES for f in out) and res.cases:
        w = max(res.cases, key=lambda c: c.worst_mm)
        _finding(out, 'good', 'calibration', f'Every case stays within the {st.tol_mm:g} mm / {st.tol_deg:g}° '
                                             f'tolerance of the calibration load (worst {w.name}: {w.worst_mm:.0f} mm, '
                                             f'{w.turn_deg:.2f}°).')
    order = {s: i for i, s in enumerate(SEVERITIES)}
    out.sort(key=lambda f: order.get(f['severity'], 9))
    return out


_ISSUE_CACHE: dict = {}


def issues(project, vg=None) -> list:
    """The static solve's problems for the Issues list (no ground analysis; a few ms, cached on the inputs): a wheel
    that would lift, a case over the gross mass, bump stops touched at static load, a measured curve extrapolated, the
    rig moving beyond the calibration tolerance. Only once the project has suspension settings (the dialog or an
    import made them); with typical suspension figures the messages say so."""
    from . import rules
    if not getattr(project.analysis, 'suspension', None):
        return []
    v = project.variant
    try:
        key = json.dumps([project.analysis.suspension, project.vehicle.wheel_centres, id(project.vehicle.mesh),
                          [(c.uid, c.visible, c.x, c.y, c.z, c.yaw, c.pitch, c.roll, c.module_id,
                            project.module_for(c).weight_g) for c in v.cameras],
                          [(s.uid, s.visible, s.x, s.y, s.z, s.yaw, s.pitch, s.roll, s.model_id,
                            project.sensor_model_for(s).size_mm) for s in v.sensors],
                          [p.to_dict() for p in project.rig], [project.ecu.x, project.ecu.y, project.ecu.z]],
                         sort_keys=True, default=str)
    except (TypeError, ValueError):
        key = None
    with _LOCK:
        hit = _ISSUE_CACHE.get(key) if key is not None else None
    if hit is not None:
        return list(hit)
    res = solve_cases(project)
    tail = ' (typical suspension values)' if res.settings.is_typical else ''
    out = []
    for f in res.findings:
        if f['severity'] not in ISSUE_SEVERITIES:
            continue
        tags = [d['tag'] for c in res.cases if c.id == f['case'] for d in c.devices if d['uid'] in f['uids']]
        out.append(rules.Issue(f['severity'], f['message'] + tail, ', '.join(tags) or f'{res.variant} · load cases',
                               list(f['uids'])))
    if key is not None:
        with _LOCK:
            if len(_ISSUE_CACHE) > 32:
                _ISSUE_CACHE.pop(next(iter(_ISSUE_CACHE)))
            _ISSUE_CACHE[key] = out
    return list(out)


# =========================================================================== import / export of the suspension file
def file_doc(st: SuspensionSettings) -> dict:
    return dict({'schema': FILE_SCHEMA}, **st.to_dict())


def dumps(st: SuspensionSettings, fmt='json') -> str:
    """The suspension file: json or yaml (everything), csv (the parameter sheet: the suspension, not the loads)."""
    if fmt == 'json':
        return json.dumps(file_doc(st), indent=1) + '\n'
    if fmt in ('yaml', 'yml'):
        import yaml
        return yaml.safe_dump(file_doc(st), sort_keys=False, default_flow_style=None, width=120, allow_unicode=True)
    if fmt == 'csv':
        return params_csv(st)
    raise ValueError(f'unknown format {fmt}')


def params_csv(st: SuspensionSettings) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator='\n')
    w.writerow(['parameter', 'value', 'unit'])
    w.writerow(['source', st.source, ''])
    w.writerow(['kerb_mass', f'{st.kerb_mass:g}', 'kg'])
    w.writerow(['front_pct', f'{st.front_pct:g}', '%'])
    w.writerow(['left_pct', f'{st.left_pct:g}', '%'])
    w.writerow(['track_front', f'{st.track_front:g}', 'm'])
    w.writerow(['track_rear', f'{st.track_rear:g}', 'm'])
    w.writerow(['gross_mass', f'{st.gross_mass:g}', 'kg'])
    w.writerow(['driver_side', st.driver_side, ''])
    w.writerow(['occupant_mass', f'{st.occupant_mass:g}', 'kg'])
    w.writerow(['arb_front', f'{st.arb_front:g}', 'N/mm'])
    w.writerow(['arb_rear', f'{st.arb_rear:g}', 'N/mm'])
    for c in CORNERS:
        co = st.corners[c]
        for k, u in (('spring_rate', 'N/mm'), ('motion_ratio', ''), ('tyre_rate', 'N/mm'), ('kerb_load', 'kg'),
                     ('ride_height', 'mm'), ('bump_gap', 'mm'), ('bump_rate', 'N/mm')):
            w.writerow([f'{c}.{k}', f'{getattr(co, k):g}', u])
        if co.bump_curve:
            w.writerow([f'{c}.bump_curve', ' '.join(f'{a:g}:{b:g}' for a, b in co.bump_curve), 'mm:N'])
    return buf.getvalue()


def curves_csv(st: SuspensionSettings) -> str:
    """The measured curves as a ride-height table: corner, load_N, travel_mm."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator='\n')
    w.writerow(['corner', 'load_N', 'travel_mm'])
    for c in CORNERS:
        for load, travel in st.corners[c].curve:
            w.writerow([c, f'{load:g}', f'{travel:g}'])
    return buf.getvalue()


def save_file(st: SuspensionSettings, path: str) -> list[str]:
    """Write the suspension file by its extension (.json, .yaml/.yml, .csv: the sheet, plus <name>_curves.csv when a
    corner has a measured curve); returns the paths written."""
    ext = os.path.splitext(path)[1].lower().lstrip('.') or 'json'
    if ext not in ('json', 'yaml', 'yml', 'csv'):
        raise ValueError(f'"{os.path.basename(path)}": save as .json, .yaml or .csv')
    with open(path, 'w', encoding='utf-8', newline='') as fh:
        fh.write(dumps(st, ext))
    paths = [path]
    if ext == 'csv' and any(st.corners[c].curve for c in CORNERS):
        cp = os.path.splitext(path)[0] + '_curves.csv'
        with open(cp, 'w', encoding='utf-8', newline='') as fh:
            fh.write(curves_csv(st))
        paths.append(cp)
    return paths


_HEAD = re.compile(r'^\s*([a-z_ ]*?[a-z])\s*(?:[\[(]\s*([^\])]*)\s*[\])])?\s*$')
_LOAD_WORDS = ('load', 'force', 'weight', 'wheel_load', 'wheel load', 'vertical load', 'fz')
_TRAVEL_WORDS = ('travel', 'wheel_travel', 'wheel travel', 'deflection', 'displacement', 'compression', 'jounce', 'bump')
_RIDE_WORDS = ('ride_height', 'ride height', 'height', 'ride')


def _header(h: str):
    """'load_N' / 'Load (kN)' / 'travel [mm]' / 'ride height mm' -> (name, unit or '')."""
    s = h.strip().lower()
    m = _HEAD.match(s)
    if m and m.group(2):
        return m.group(1).strip(), m.group(2).strip()
    parts = re.split(r'[_\s]+', s)
    if len(parts) > 1:
        tail = _norm_unit(parts[-1])
        if any(tail in _U[k][0] for k in ('force', 'mass', 'length')):
            return ' '.join(parts[:-1]), parts[-1]
    return s.replace('_', ' '), ''


def _rows(text):
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith('#')]
    if not lines:
        raise ValueError('The file is empty.')
    try:
        dialect = csv.Sniffer().sniff(lines[0], delimiters=',;\t')
        delim = dialect.delimiter
    except csv.Error:
        delim = ','
    return [r for r in csv.reader(lines, delimiter=delim)]


def parse_curve_table(text: str, corner=None) -> tuple[dict, list]:
    """A ride-height table: {corner: [[load N, travel mm], ...]} and warnings. Columns: a load (load_N, 'load (kN)',
    load_kg: a mass is read as its weight), a travel (travel_mm, bump +) or a ride height (ride_height_mm: it falls as
    the load rises), and an optional corner or axle column; without one, corner names where the table goes (FL,
    'front', 'rear', 'all'). Raises ValueError with what is wrong."""
    rows = _rows(text)
    head = [_header(h) for h in rows[0]]
    names = [n for n, _ in head]
    warnings = []
    li = next((i for i, n in enumerate(names) if n in _LOAD_WORDS or n.startswith('load') or 'load' in n.split()),
              None)
    ti = next((i for i, n in enumerate(names) if n in _TRAVEL_WORDS or n.startswith('travel')), None)
    ri = next((i for i, n in enumerate(names) if n in _RIDE_WORDS), None) if ti is None else None
    ci = next((i for i, n in enumerate(names) if n in ('corner', 'wheel', 'position', 'axle')), None)
    if li is None or (ti is None and ri is None):
        raise ValueError('Not a ride-height table: the header needs a load column (load_N, "load (kN)", load_kg) and a '
                         f'travel column (travel_mm) or a ride height (ride_height_mm); it has {", ".join(rows[0])}.')
    lu = head[li][1]
    lkind = 'force'
    if lu and _norm_unit(lu) in _U['mass'][0] and _norm_unit(lu) not in _U['force'][0]:
        lkind = 'mass'
    if lu and _norm_unit(lu) in ('kg', 't', 'g', 'lbm'):
        lkind = 'mass'
    if not lu:
        warnings.append(f'"{rows[0][li]}" has no unit: read as N.')
        lu = 'N'
    di = ti if ti is not None else ri
    du = head[di][1]
    if not du:
        warnings.append(f'"{rows[0][di]}" has no unit: read as mm.')
        du = 'mm'
    lf = unit_factor(lu, lkind, rows[0][li]) * (G if lkind == 'mass' else 1.0)
    df = unit_factor(du, 'length', rows[0][di])
    target = None
    if ci is None:
        if corner is None:
            raise ValueError('The table has no corner column: choose the corner (or axle) it is for.')
        target = _targets(corner)
        if target is None:
            raise ValueError(f'"{corner}" is not a corner or an axle.')
    out: dict = {}
    for n, r in enumerate(rows[1:], start=2):
        if not any(x.strip() for x in r):
            continue
        try:
            load = quantity(r[li], 'force' if lkind == 'force' else 'mass', None, f'row {n}') * lf
            d = quantity(r[di], 'length', None, f'row {n}') * df
        except (ValueError, IndexError) as e:
            raise ValueError(f'row {n}: {e if isinstance(e, ValueError) else "missing a value"}') from None
        cs = target
        if ci is not None:
            key = r[ci].strip() if ci < len(r) else ''
            cs = _targets(key)
            if cs is None:
                raise ValueError(f'row {n}: "{key}" is not a corner (FL, FR, RL, RR) or an axle (front, rear)')
        for c in cs:
            out.setdefault(c, []).append([load, d if ti is not None else -d])
    if not out:
        raise ValueError('The table has no rows.')
    for c, pts in out.items():
        pts.sort(key=lambda p: p[1])
        why = check_curve(pts, f'{c}')
        if why:
            raise ValueError(why)
        if len(pts) > MAX_CURVE:
            raise ValueError(f'{c}: {len(pts)} points, at most {MAX_CURVE}')
    return out, warnings


def _targets(key):
    k = str(key).strip().lower()
    if k in ('front', 'f', 'front axle'):
        return list(AXLE_CORNERS['front'])
    if k in ('rear', 'r', 'rear axle'):
        return list(AXLE_CORNERS['rear'])
    if k in ('all', 'every', '*'):
        return list(CORNERS)
    c = corner_key(k)
    return [c] if c else None


def parse_params(text: str, base: SuspensionSettings) -> tuple[SuspensionSettings, list]:
    """A "parameter,value,unit" sheet over base: (settings, warnings). Raises ValueError listing every bad row."""
    rows = _rows(text)
    if rows and rows[0] and rows[0][0].strip().lower() in ('parameter', 'param', 'name', 'key'):
        rows = rows[1:]
    d = {'vehicle': {}, 'axles': {'front': {}, 'rear': {}}, 'corners': {c: {} for c in CORNERS}, 'loads': {}}
    errors = []
    valid = sorted(set(CORNER_FIELDS) - {'curve'} | {'arb_rate', 'axle_load'})
    for n, r in enumerate(rows, start=2):
        if not r or not r[0].strip():
            continue
        key = r[0].strip()
        val = r[1].strip() if len(r) > 1 else ''
        unit = r[2].strip() if len(r) > 2 else ''
        k = key.lower()
        if k in ('source', 'note'):
            d[k] = val
            continue
        if k == 'driver_side':
            d['vehicle']['driver_side'] = val
            continue
        if '.' not in k:
            if k not in VEHICLE_PARAMS:
                errors.append(f'row {n}: unknown parameter "{key}" (vehicle parameters: {", ".join(VEHICLE_PARAMS)}; '
                              'corner or axle ones as FL.spring_rate, front.tyre_rate)')
                continue
            tgt = d['loads'] if k == 'occupant_mass' else d['vehicle']
            tgt[k] = _with_unit(val, unit)
            continue
        scope, fld = k.split('.', 1)
        if fld not in valid and fld != 'bump_curve':
            errors.append(f'row {n}: unknown field "{fld}" in "{key}" (fields: {", ".join(valid)})')
            continue
        if fld == 'bump_curve':
            try:
                pts = [[float(a), float(b)] for a, b in (p.split(':') for p in val.split())]
            except ValueError:
                errors.append(f'row {n}: "{val}" is not a bump curve (compression:force pairs, e.g. 0:0 10:800)')
                continue
            v = pts
        else:
            v = _with_unit(val, unit)
        if scope in ('front', 'rear'):
            d['axles'][scope][fld] = v
        elif corner_key(scope):
            if fld in ('arb_rate', 'axle_load'):
                errors.append(f'row {n}: {fld} belongs to an axle (front.{fld} or rear.{fld})')
                continue
            d['corners'][corner_key(scope)][fld] = v
        else:
            errors.append(f'row {n}: "{scope}" is not a corner (FL, FR, RL, RR) or an axle (front, rear)')
    if errors:
        raise ValueError('\n'.join(errors))
    st = SuspensionSettings.from_dict(d, base=base, strict=True)
    return st, st.warnings


def _with_unit(val, unit):
    return f'{val} {unit}'.strip() if unit and unit not in ('%', '-') else val


def load_file(path: str, base: SuspensionSettings, corner=None) -> tuple[SuspensionSettings, str, list]:
    """Read a suspension file over base: JSON / YAML (the file schema), a CSV parameter sheet or a CSV ride-height
    table (curves for the corners it names, or `corner`). Returns (settings, what was read, warnings); the settings'
    source becomes 'imported'. Raises ValueError with a clear message."""
    name = os.path.basename(path)
    try:
        with open(path, 'r', encoding='utf-8-sig') as fh:
            text = fh.read()
    except (OSError, UnicodeDecodeError) as e:
        raise ValueError(f'{name}: cannot read it ({e})') from None
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in ('.json', '.yaml', '.yml'):
            if ext == '.json':
                try:
                    doc = json.loads(text)
                except json.JSONDecodeError as e:
                    raise ValueError(f'not valid JSON (line {e.lineno}: {e.msg})') from None
            else:
                import yaml
                try:
                    doc = yaml.safe_load(text)
                except yaml.YAMLError as e:
                    raise ValueError(f'not valid YAML ({e})') from None
            if not isinstance(doc, dict):
                raise ValueError('expected a mapping at the top level')
            sch = str(doc.get('schema', '') or '')
            if sch and not sch.startswith('vantage.suspension_data/'):
                raise ValueError(f'schema "{sch}" is not a suspension file ({FILE_SCHEMA})')
            if not any(k in doc for k in ('corners', 'axles', 'vehicle', 'loads')):
                raise ValueError('no corners, axles, vehicle or loads in it')
            st = SuspensionSettings.from_dict(doc, base=base, strict=True)
            what = 'suspension file'
            warnings = st.warnings
        elif ext in ('.csv', '.txt', '.tsv'):
            rows = _rows(text)
            first = rows[0][0].strip().lower() if rows and rows[0] else ''
            if first in ('parameter', 'param', 'name', 'key'):
                st, warnings = parse_params(text, base)
                what = 'parameter sheet'
            else:
                curves, warnings = parse_curve_table(text, corner)
                st = copy.deepcopy(base)
                for c, pts in curves.items():
                    st.corners[c].curve = pts
                what = f'ride-height curve{"s" if len(curves) > 1 else ""} for {", ".join(sorted(curves, key=CORNERS.index))}'
        else:
            raise ValueError('use a .json, .yaml or .csv file')
    except ValueError as e:
        raise ValueError(f'{name}: {e}') from None
    st.source = 'imported'
    st.note = st.note or f'from {name}'
    return st, what, list(warnings)


# =========================================================================== output
def _r(v, d=3):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return round(v, d) if math.isfinite(v) else None


def _corner_doc(cs: CornerState):
    return {'load_kg': _r(cs.load_kg, 2), 'added_n': _r(cs.added_n, 1), 'travel_mm': _r(cs.travel_mm, 2),
            'tyre_mm': _r(cs.tyre_mm, 2), 'drop_mm': _r(cs.drop_mm, 2), 'ride_height_mm': _r(cs.ride_mm, 1),
            'bump_stop_mm': _r(cs.bump_mm, 2), 'on_bump_stop': cs.on_bump, 'wheel_lifts': cs.lifted,
            'beyond_curve': cs.beyond_curve}


def _device_doc(d):
    return {'tag': d['tag'], 'kind': d['kind'], 'dx_mm': _r(d['dx_mm'], 2), 'dy_mm': _r(d['dy_mm'], 2),
            'dz_mm': _r(d['dz_mm'], 2), 'move_mm': _r(d['move_mm'], 2), 'd_yaw_deg': _r(d['d_yaw_deg'], 4),
            'd_pitch_deg': _r(d['d_pitch_deg'], 4), 'd_roll_deg': _r(d['d_roll_deg'], 4)}


def summary(res: SuspensionResult) -> dict:
    """Compact figures (the assistant, the batch log)."""
    st = res.settings
    return {'suspension': 'typical values' if st.is_typical else st.source, 'calibrated_at': res.calibration,
            'kerb_kg': _r(st.kerb_mass, 1), 'tolerance': {'mm': st.tol_mm, 'deg': st.tol_deg},
            'cases': [{'name': c.name, 'added_kg': _r(c.statics.added_kg, 1),
                       'drops_mm': {k: _r(c.statics.corners[k].drop_mm, 1) for k in CORNERS},
                       'pitch_deg': _r(c.statics.pitch_deg, 3), 'roll_deg': _r(c.statics.roll_deg, 3),
                       'vs_calibration': {'pitch_deg': _r(c.rel['pitch_deg'], 3), 'roll_deg': _r(c.rel['roll_deg'], 3),
                                          'heave_mm': _r(c.rel['heave_mm'], 1), 'worst_move_mm': _r(c.worst_mm, 1),
                                          'worst_device': c.worst_device['tag'] if c.worst_device else None},
                       'bump_stops': c.statics.bump,
                       'seam_ghost_cm': _r(c.image.seam_ghost_cm, 1) if c.image is not None else None,
                       'ground_shift_cm': _r(c.image.ground_shift_cm, 1) if c.image is not None else None}
                      for c in res.cases],
            'findings': [{'severity': f['severity'], 'message': f['message']} for f in res.findings
                         if f['severity'] in ISSUE_SEVERITIES],
            'assumptions': ASSUMPTIONS}


def to_doc(res: SuspensionResult, project=None) -> dict:
    """The JSON export: the inputs, every case's corner loads and drops, attitude, device moves relative to the
    calibration load and (after an evaluation) the ground figures."""
    st = res.settings
    return {
        'schema': SCHEMA, 'project': getattr(project, 'name', ''), 'variant': res.variant,
        'frame': 'vehicle (ISO 8855, rear axle)',
        'units': {'drop': 'mm', 'travel': 'mm', 'load': 'kg (wheel load), N (added)', 'angle': 'deg',
                  'position': 'm', 'move': 'mm'},
        'signs': {'drop': 'positive = lower', 'travel': 'positive = bump', 'pitch': 'positive = nose down',
                  'roll': 'positive = right side lower (right-hand about +x)'},
        'assumptions': ASSUMPTIONS, 'typical_suspension': st.is_typical,
        'note': TYPICAL_NOTE if st.is_typical else st.note,
        'suspension': st.to_dict(),
        'corner_positions_m': {c: [_r(res.xy[c][0], 4), _r(res.xy[c][1], 4)] for c in CORNERS},
        'rig': [{'tag': c['tag'], 'name': c['name'], 'kind': c['kind'], 'mass_kg': _r(c['mass'], 3),
                 'source': c['source'], 'x': _r(c['x']), 'y': _r(c['y']), 'z': _r(c['z'])} for c in res.components],
        'calibrated_at': res.calibration,
        'cases': [{
            'name': c.name, 'id': c.id,
            'items': [{'name': n, 'mass_kg': _r(m, 3), 'x': _r(x), 'y': _r(y), 'z': _r(z)} for n, m, x, y, z in c.loads],
            'added_kg': _r(c.statics.added_kg, 2), 'total_kg': _r(c.statics.total_kg, 1),
            'added_cog_m': [_r(v) for v in c.statics.cog], 'over_gross_kg': _r(c.over_gross, 1),
            'corners': {k: _corner_doc(c.statics.corners[k]) for k in CORNERS},
            'heave_mm': _r(c.statics.heave_mm, 2), 'pitch_deg': _r(c.statics.pitch_deg, 4),
            'roll_deg': _r(c.statics.roll_deg, 4), 'bump_stops': c.statics.bump,
            'converged': c.statics.converged,
            'vs_calibration': {'heave_mm': _r(c.rel['heave_mm'], 2), 'pitch_deg': _r(c.rel['pitch_deg'], 4),
                               'roll_deg': _r(c.rel['roll_deg'], 4), 'turn_deg': _r(c.turn_deg, 4),
                               'worst_move_mm': _r(c.worst_mm, 2),
                               'devices': [_device_doc(d) for d in c.devices]},
            'ground': ({'blind_m2': _r(c.image.blind_m2), 'blind_change_m2': _r(c.image.delta_blind),
                        'min_density_px_cm': _r(c.image.min_density, 2),
                        'ground_shift_cm': _r(c.image.ground_shift_cm, 2),
                        'seam_ghost_cm': _r(c.image.seam_ghost_cm, 2), 'status': c.image.status}
                       if c.image is not None else None)} for c in res.cases],
        'ground_note': '' if res.images else (res.image_note or 'Not evaluated (statics only).'),
        'findings': [{'severity': f['severity'], 'kind': f['kind'], 'message': f['message']} for f in res.findings],
    }


CSV_COLUMNS = ('case', 'added_kg', 'total_kg', 'FL_drop_mm', 'FR_drop_mm', 'RL_drop_mm', 'RR_drop_mm',
               'FL_load_kg', 'FR_load_kg', 'RL_load_kg', 'RR_load_kg', 'heave_mm', 'pitch_deg', 'roll_deg',
               'rel_pitch_deg', 'rel_roll_deg', 'rel_heave_mm', 'worst_move_mm', 'worst_device', 'bump_stops',
               'blind_change_m2', 'ground_shift_cm', 'seam_ghost_cm', 'status')


def to_csv(res: SuspensionResult) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator='\n')
    w.writerow(CSV_COLUMNS)
    blank = lambda v: '' if v is None else v          # noqa: E731
    for c in res.cases:
        s = c.statics
        im = c.image
        w.writerow([c.name, _r(s.added_kg, 2), _r(s.total_kg, 1)] +
                   [_r(s.corners[k].drop_mm, 2) for k in CORNERS] + [_r(s.corners[k].load_kg, 1) for k in CORNERS] +
                   [_r(s.heave_mm, 2), _r(s.pitch_deg, 4), _r(s.roll_deg, 4), _r(c.rel['pitch_deg'], 4),
                    _r(c.rel['roll_deg'], 4), _r(c.rel['heave_mm'], 2), _r(c.worst_mm, 2),
                    c.worst_device['tag'] if c.worst_device else '', ' '.join(s.bump)] +
                   [blank(_r(im.delta_blind)) if im else '', blank(_r(im.ground_shift_cm, 2)) if im else '',
                    blank(_r(im.seam_ghost_cm, 2)) if im else '', im.status if im else ''])
    return buf.getvalue()


def devices_csv(res: SuspensionResult) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator='\n')
    w.writerow(['case', 'tag', 'kind', 'dx_mm', 'dy_mm', 'dz_mm', 'move_mm', 'd_yaw_deg', 'd_pitch_deg',
                'd_roll_deg'])
    for c in res.cases:
        for d in c.devices:
            dd = _device_doc(d)
            w.writerow([c.name] + [dd[k] for k in ('tag', 'kind', 'dx_mm', 'dy_mm', 'dz_mm', 'move_mm', 'd_yaw_deg',
                                                   'd_pitch_deg', 'd_roll_deg')])
    return buf.getvalue()


def files(res: SuspensionResult, project=None) -> list:
    """[(name, bytes)]: suspension_load_cases.json and .csv, suspension_devices.csv (every camera's and sensor's move
    per case) and suspension_data.json (the inputs, to import again)."""
    return [('suspension_load_cases.json', (json.dumps(to_doc(res, project), indent=1) + '\n').encode('utf-8')),
            ('suspension_load_cases.csv', to_csv(res).encode('utf-8')),
            ('suspension_devices.csv', devices_csv(res).encode('utf-8')),
            ('suspension_data.json', dumps(res.settings, 'json').encode('utf-8'))]


def export(res: SuspensionResult, project, out_dir) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for name, data in files(res, project):
        path = os.path.join(out_dir, name)
        with open(path, 'wb') as fh:
            fh.write(data)
        paths.append(path)
    return paths
