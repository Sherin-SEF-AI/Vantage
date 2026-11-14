"""Aerodynamics of the roof rig: drag of every member, plate, foot and sensor, the EV range it costs, vortex shedding
against the structure's modes (lock-in) and the wind load on each sensor. Not CFD: published drag coefficients of
simple bluff bodies times the rig's own frontal areas, with the corrections stated below.

Frame and units. The vehicle frame (ISO 8855: x forward, y left, z up, origin at the rear-axle centre on the ground).
Speeds in km/h for people, m/s inside; forces N; energy Wh/km; range km.

Air: density ρ = p/(R T) (dry air, R = 287.05 J/kg K), kinematic viscosity from Sutherland's law (μ0 = 1.716e-5 Pa s
at 273.15 K, S = 110.4 K: White, Fluid Mechanics, eq. 1.36). The relative wind U is the vehicle's speed from ahead
plus a crosswind from the left (the load case); the roof's boundary layer slows parts close to the roof: the 1/7-power
profile u/U = (z/δ)^(1/7) with the turbulent flat-plate thickness δ = 0.37 x Re_x^(−1/5) from the roof's front edge
(White eq. 7.42), times roof_speed_factor (1 = the free stream; the flow over a car's roof runs faster than the free
stream near its front: typical 1.1-1.3, set it if known).

Drag of an element (its force along the local relative wind):
- Slender members and cylinders (cross-flow principle, Hoerner 1965 ch. 3): only the velocity normal to the axis
  makes pressure drag, ½ρU_n² Cd₂D k(L/D) · (projected width × length); the axial component makes skin friction on
  the perimeter (Cf below) and, on a free upstream end, end drag Cd 0.8 × the end area (a blunt face, Hoerner ch. 3).
  Circular sections: Cd₂D 1.2 below Re 2·10⁵ and 0.3 above 5·10⁵ (the drag crisis, log-interpolated between: White,
  Fluid Mechanics, Table 7.2). Square and rectangular sections: Cd₂D from the streamwise-to-frontal depth ratio,
  2.0 for a square face-on (White Table 7.2: 2.1 sharp-edged), down to 1.0 for a long afterbody (approximate reading
  of the rectangular-cylinder data of Delany & Sorensen 1953, NACA TN 3038, reproduced by Hoerner ch. 3). Finite
  length: k(L/D) = 0.63, 0.68, 0.74, 0.82, 0.98, 1 at L/D = 1, 2, 5, 10, 40, ∞ (White Table 7.2, after
  Wieselsberger); an end standing on a wall (a mast on the rack, a lidar on its plate) doubles the effective L/D
  (the image of the end in the wall, Hoerner ch. 3).
- Boxes (radar housings, enclosures, mast heads, foot blocks): Cd 1.18 for a square plate face-on, 1.07 for a cube
  (White Table 7.3), interpolated on the streamwise depth / frontal size and held at 1.07 beyond, times the projected
  area of the box on the plane across the wind.
- Plates (mount plates lying flat): their edge as a long rectangle (Cd from the depth ratio) plus turbulent skin
  friction on both faces, Cf = 0.074 Re_L^(−1/5) (White eq. 7.45).
- Shielding: an element in the wake of another (their frontal outlines overlapping across the wind, it downstream)
  keeps a share k(s/D) of its drag on the overlap, s the streamwise spacing and D the upstream element's width:
  0 below 2 D, rising to 0.35 at 4 D, 0.7 at 10 D and 1 at 40 D (approximate reading of tandem-cylinder data,
  Zdravkovich 1977, J. Fluids Eng. 99:618). Interference at the joints and with the roof is not modelled.

Range: the road load F = Crr·m·g + ½ρ·CdA·V² (Crr typical 0.010, Gillespie 1992 ch. 4), divided by the drive train's
efficiency (battery to wheel, typical 0.85), plus the auxiliaries' power over the speed: Wh/km. The rig adds its drag
and its mass. The base Cd·A is the vehicle family's typical Cd (Hucho 1998, ch. 4: compact cars 0.30-0.35, SUVs
0.35-0.40, vans 0.35-0.45) times a frontal area of 0.84 × width × height (typical of cars), unless set; a measured
consumption at the speed replaces the road-load model. Range loss = 1 − base / (base + added) per km.

Vortex shedding: f = St·U/D with St 0.20 for a circular section (300 < Re < 2·10⁵; Blevins 1990, Flow-Induced
Vibration, ch. 3) and 0.13 for a square section (Blevins ch. 3; Okajima 1982). A mode of the structure (core.fea when
a result is at hand, else core.thermo_vib's beam modes) in which the element moves across the wind locks in where
f_shedding = f_mode: at V = f·D/St, counted only for modes that move the element along its shedding force (across its axis and
the flow). Within the driving speeds that is a warning, with the Scruton number
Sc = 4π·m·ζ/(ρD²) (m mass per metre): the higher it is, the smaller the motion (Blevins ch. 3).

Wind load: the forces of the load case (the highway speed with the crosswind) on each device and rig part go to
core.thermo_vib.wind_response (the beam chain) and, when asked, to core.fea's static solve.

Wind noise: not estimated as a level. The tone of each element is its shedding frequency (stated per element), but
the level inside the car depends on the roof's transmission loss and on the slots and edges (open T-slots whistle:
fit slot covers), which no defensible simple model gives.

Out of scope: CFD, the vehicle's own flow field beyond the boundary layer and the speed factor, interference drag at
joints, lift, unsteady gust loads, rain and snow, galloping of rectangular sections, the drag of cables and ties.
"""
from __future__ import annotations

import csv
import io
import json
import math
import os
from dataclasses import dataclass, field

import numpy as np

SCHEMA = 'vantage.aero/1'
FRAME = 'vehicle (ISO 8855, rear axle): x forward, y left, z up'
R_AIR = 287.05
G = 9.80665
FAMILY_CD = {'sedan': 0.30, 'hatch': 0.32, 'suv': 0.36, 'van': 0.38, 'truck': 0.60}
CD_SOURCE = 'typical by body family (Hucho, Aerodynamics of Road Vehicles, 4th ed. 1998, ch. 4)'
AREA_FACTOR = 0.84
CYL_SUB, CYL_SUPER, RE_CRISIS = 1.2, 0.3, (2e5, 5e5)
K_AR = ((1.0, 0.63), (2.0, 0.68), (5.0, 0.74), (10.0, 0.82), (40.0, 0.98), (1e9, 1.0))
RECT_2D = ((0.0, 2.0), (0.5, 2.3), (1.0, 2.0), (2.0, 1.6), (3.0, 1.3), (5.0, 1.0), (1e9, 1.0))
BOX_3D = ((0.0, 1.18), (1.0, 1.07), (1e9, 1.07))
TANDEM = ((0.0, 0.0), (2.0, 0.0), (4.0, 0.35), (10.0, 0.7), (40.0, 1.0), (1e9, 1.0))
END_CD = 0.8
ST_CIRCLE, ST_SQUARE = 0.20, 0.13
SC_SERIOUS, SC_NOTE = 10.0, 50.0    # Scruton number: below 10 large lock-in motion is possible; above about 50 it is
                                    # small (typical thresholds, Blevins 1990 ch. 3, the amplitude against mass-damping)
SOURCES = {
    'cd_circle': 'White, Fluid Mechanics (7th ed.), Table 7.2: circular cylinder 1.2 laminar (subcritical), 0.3 '
                 'turbulent (supercritical)',
    'k_ar': 'White Table 7.2 (finite circular cylinder at Re 10⁵, after Wieselsberger): L/D 1, 2, 5, 10, 40, ∞ → '
            '0.63, 0.68, 0.74, 0.82, 0.98, 1.00',
    'cd_rect': 'approximate reading of 2D rectangular cylinders (Delany & Sorensen 1953, NACA TN 3038; Hoerner, '
               'Fluid-Dynamic Drag 1965, ch. 3); square face-on 2.0 (White Table 7.2: 2.1)',
    'cd_box': 'White Table 7.3: square plate 1.18, cube 1.07 (3D, face-on)',
    'tandem': 'approximate reading of tandem circular cylinders (Zdravkovich 1977, J. Fluids Eng. 99:618)',
    'strouhal': 'Blevins, Flow-Induced Vibration (2nd ed. 1990), ch. 3: circular 0.20, square 0.13 (Okajima 1982)',
    'cf': 'White eq. 7.45: turbulent flat plate Cf = 0.074 Re_L^(−1/5)',
    'bl': 'White eq. 7.42: turbulent boundary layer δ/x = 0.37 Re_x^(−1/5), 1/7-power profile',
    'crr': 'typical passenger-car tyre on asphalt 0.010 (Gillespie, Fundamentals of Vehicle Dynamics 1992, ch. 4)',
    'cd_base': CD_SOURCE, 'area': f'frontal area ≈ {AREA_FACTOR} × width × height (typical of cars)',
    'sutherland': 'White eq. 1.36: Sutherland, μ0 = 1.716e-5 Pa s at 273.15 K, S = 110.4 K',
}
LIMITS = {'speed1_kmh': (10.0, 250.0), 'speed2_kmh': (10.0, 250.0), 'v_max_kmh': (10.0, 250.0),
          'load_kmh': (10.0, 250.0), 'crosswind_ms': (0.0, 40.0), 'air_c': (-40.0, 50.0),
          'pressure_kpa': (50.0, 110.0), 'cd_base': (0.0, 1.5), 'frontal_m2': (0.0, 15.0), 'mass_kg': (0.0, 40000.0),
          'battery_kwh': (1.0, 1000.0), 'consumption_wh_km': (0.0, 2000.0), 'crr': (0.003, 0.03),
          'drive_eff': (0.3, 1.0), 'aux_w': (0.0, 10000.0), 'roof_speed_factor': (0.5, 2.0),
          'lockin_band': (0.02, 0.5), 'zeta': (0.001, 0.3)}
ASSUMPTIONS = ('Not CFD: published bluff-body drag coefficients (cited) times the rig\'s frontal areas, cross-flow '
               'principle for slender parts, finite-length and wall corrections, a 1/7-power boundary layer over the '
               'roof, approximate tandem shielding; no interference at joints. Range from the road-load equation '
               'with typical Cd, rolling resistance, drive efficiency and auxiliaries unless set. Shedding at '
               'constant Strouhal numbers; lock-in where a mode equals the shedding frequency.')


def _num(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


@dataclass
class AeroSettings:
    speed1_kmh: float = 100.0
    speed2_kmh: float = 130.0
    v_max_kmh: float = 130.0        # the highest driving speed (the lock-in check runs up to it)
    load_kmh: float = 130.0         # the wind load case: this speed …
    crosswind_ms: float = 15.0      # … with this crosswind from the left
    air_c: float = 20.0
    pressure_kpa: float = 101.325
    cd_base: float = 0.0            # 0 = the vehicle family's typical Cd
    frontal_m2: float = 0.0         # 0 = 0.84 × width × height
    mass_kg: float = 0.0            # 0 = the kerb mass (core.suspension) + a 75 kg driver
    battery_kwh: float = 40.0       # usable energy: set the vehicle's
    consumption_wh_km: float = 0.0  # measured at speed2 (0 = the road-load model)
    crr: float = 0.010
    drive_eff: float = 0.85
    aux_w: float = 400.0            # auxiliaries (HVAC, 12 V) while driving, typical
    roof_speed_factor: float = 1.0
    lockin_band: float = 0.2        # a mode within ±this of the shedding frequency is flagged (lock-in range)
    zeta: float = 0.02              # structural damping for the Scruton number
    use_frame: bool = True          # the synthesised frame (core.framing) rather than the rig parts as drawn

    def to_dict(self):
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    @staticmethod
    def from_dict(d) -> 'AeroSettings':
        s = AeroSettings()
        d = d if isinstance(d, dict) else {}
        for k, (lo, hi) in LIMITS.items():
            v = _num(d.get(k))
            if v is not None:
                setattr(s, k, min(max(v, lo), hi))
        if isinstance(d.get('use_frame'), bool):
            s.use_frame = d['use_frame']
        return s


def settings_of(project) -> AeroSettings:
    return AeroSettings.from_dict(getattr(project.analysis, 'aero', None))


# =========================================================================== air and coefficients
def air(t_c=20.0, p_kpa=101.325) -> tuple[float, float]:
    """(density kg/m³, kinematic viscosity m²/s) of dry air."""
    T = t_c + 273.15
    rho = p_kpa * 1000.0 / (R_AIR * T)
    mu = 1.716e-5 * (T / 273.15) ** 1.5 * (273.15 + 110.4) / (T + 110.4)
    return rho, mu / rho


def _interp(table, x, log=False):
    xs = np.array([a for a, _ in table], float)
    ys = np.array([b for _, b in table], float)
    return float(np.interp(x, xs, ys))


def k_aspect(L_over_D) -> float:
    """White Table 7.2's finite-cylinder factor, interpolated in log(L/D)."""
    x = max(float(L_over_D), 1.0)
    xs = np.log([a for a, _ in K_AR])
    ys = [b for _, b in K_AR]
    return float(np.interp(math.log(x), xs, ys))


def cd_circle(re) -> float:
    lo, hi = RE_CRISIS
    if re <= lo:
        return CYL_SUB
    if re >= hi:
        return CYL_SUPER
    f = math.log(re / lo) / math.log(hi / lo)
    return CYL_SUB + f * (CYL_SUPER - CYL_SUB)


def cd_rect(depth_ratio) -> float:
    return _interp(RECT_2D, max(depth_ratio, 0.0))


def cd_box(depth_ratio) -> float:
    return _interp(BOX_3D, max(depth_ratio, 0.0))


def tandem(s_over_d) -> float:
    return _interp(TANDEM, max(s_over_d, 0.0))


def cf_turbulent(re_l) -> float:
    return 0.074 * max(re_l, 1e4) ** -0.2


def bl_factor(z_above, x_run, U, nu) -> float:
    """u/U at z_above over the roof, x_run behind its front edge (1/7 power, turbulent)."""
    if z_above <= 0:
        return 0.0
    rex = max(U * max(x_run, 0.05) / nu, 1e4)
    delta = 0.37 * max(x_run, 0.05) * rex ** -0.2
    return float(min(1.0, (z_above / delta) ** (1.0 / 7.0)))


# =========================================================================== the rig's elements
@dataclass
class Element:
    id: str
    name: str
    kind: str                       # prism | cylinder | box | plate
    centre: np.ndarray
    key: str = ''                   # FE block or body key (member id, plate id, device uid)
    tv_key: str = ''                # core.thermo_vib key (device uid or rig part uid)
    axis: np.ndarray | None = None  # prism, cylinder
    ex: np.ndarray | None = None    # prism: its narrow side
    b: float = 0.0                  # prism: narrow, deep side; cylinder: D in b
    d: float = 0.0
    L: float = 0.0
    R: np.ndarray | None = None     # box: axes (columns)
    half: np.ndarray | None = None
    walls: int = 0                  # ends standing on a wall (0, 1, 2)
    free_ends: tuple = (False, False)
    kg_m: float = 0.0
    bottom: float = 0.0

    def width_across(self, u):
        """Frontal width of a slender element across the unit direction u (⊥ its axis)."""
        if self.kind == 'cylinder':
            return self.b
        ey = np.cross(self.axis, self.ex)
        return self.b * abs(float(u @ ey)) + self.d * abs(float(u @ self.ex))

    def depth_along(self, u):
        if self.kind == 'cylinder':
            return self.b
        ey = np.cross(self.axis, self.ex)
        return self.b * abs(float(u @ self.ex)) + self.d * abs(float(u @ ey))

    def outline(self, u):
        """Frontal outline seen along u: (centre (2,), half extents (2,)) in the plane across u (e1, e2)."""
        e1, e2 = _across(u)
        pts = self.corners()
        q = np.c_[(pts - self.centre) @ e1, (pts - self.centre) @ e2]
        lo, hi = q.min(axis=0), q.max(axis=0)
        c = np.array([self.centre @ e1, self.centre @ e2]) + (lo + hi) / 2
        return c, (hi - lo) / 2

    def corners(self):
        if self.kind in ('prism', 'cylinder'):
            ey = np.cross(self.axis, self.ex)
            hb, hd = self.b / 2, (self.d if self.kind == 'prism' else self.b) / 2
            out = []
            for s in (-0.5, 0.5):
                for a in (-hb, hb):
                    for c in (-hd, hd):
                        out.append(self.centre + s * self.L * self.axis + a * self.ex + c * ey)
            return np.array(out)
        sg = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], float)
        return self.centre + (sg * self.half) @ self.R.T


def _across(u):
    u = np.asarray(u, float)
    e1 = np.cross([0.0, 0.0, 1.0], u)
    if np.linalg.norm(e1) < 1e-9:
        e1 = np.array([0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(u, e1)


def elements(project, st: AeroSettings, frame=None, variant=None) -> tuple[list, list]:
    """(the rig's elements above the roof, notes): the synthesised frame (members, plates, feet) with the sensors on
    it, or the rig parts as drawn (st.use_frame False, or no frame)."""
    from . import framing as FR
    from . import thermo_vib as TV
    from .suspension import rig_part_mass
    v = variant or project.variant
    notes = []
    rf = TV.roof_box(project)
    out = []
    fr = frame
    if st.use_frame and fr is None:
        fr = FR.synthesise(project, variant=v)
    if st.use_frame and fr is not None and fr.members:
        pts = [np.asarray(j['point']) for j in fr.joints]
        for m in fr.members:
            o, ex, ey, ez = m.frame()
            pr = m.prof
            ends = []
            for P in (m.p0, m.p1):
                ends.append(not any(np.linalg.norm(q - P) < max(pr.b, pr.d) for q in pts) and
                            not any(np.linalg.norm(q - P) < max(pr.b, pr.d) for q in
                                    [P2.centre for P2 in fr.plates if m.id in P2.on]))
            walls = 1 if m.role in ('post', 'standoff') else 0
            if m.role in ('post', 'standoff') and not ends[1]:
                walls = 2
            out.append(Element(m.id, m.name, 'prism', (m.p0 + m.p1) / 2, m.id, m.from_uid, ez, ex, pr.b, pr.d,
                               m.length, walls=walls, free_ends=tuple(ends), kg_m=pr.kg_m,
                               bottom=float(min(m.p0[2], m.p1[2]) - pr.d / 2)))
        for P in fr.plates:
            c, ex, ey, ez = P.frame()
            out.append(Element(P.id, P.name, 'plate', c + ez * P.t / 2, P.id, '', R=np.stack([ex, ey, ez], axis=1),
                               half=np.array([P.lx, P.ly, P.t]) / 2, bottom=float(c[2])))
        for F in fr.feet:
            a = math.radians(F.yaw)
            R = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1.0]])
            out.append(Element(F.id, F.name, 'box', F.centre + [0, 0, F.height / 2], '', '', R=R,
                               half=np.array([F.lx, F.ly, F.height]) / 2, bottom=float(F.centre[2])))
        on_frame = {mt['uid'] for mt in fr.mounts if mt['status'] in ('plate', 'top plate')}
        for d in FR._devices(project, v, rf, fr.settings):
            out.append(_device_element(d, d['uid'] in on_frame))
        for mt in fr.mounts:
            p = next((q for q in project.rig if q.uid == mt['uid']), None)
            if p is not None and mt['kind'] == 'payload':
                R = _rz(p.yaw)
                out.append(Element(p.uid, p.name, 'box', np.array([p.x, p.y, p.z]), p.uid, p.uid, R=R,
                                   half=np.array([p.length, p.width, p.height]) / 2, bottom=float(p.z - p.height / 2)))
    else:
        if st.use_frame:
            notes.append('No frame to take (Analysis ▸ Frame is empty): the rig parts as drawn.')
        for p in getattr(project, 'rig', []) or []:
            if not p.visible:
                continue
            c = np.array([p.x, p.y, p.z])
            if p.kind == 'bar':
                ax = _rz(p.yaw)[:, 1]
                out.append(Element(p.uid, p.name, 'cylinder', c, '', p.uid, ax, _rz(p.yaw)[:, 0], 2 * p.radius,
                                   2 * p.radius, p.length, free_ends=(True, True),
                                   kg_m=rig_part_mass(p) / max(p.length, 1e-3), bottom=float(p.z - p.radius)))
            elif p.kind == 'cylinder':
                out.append(Element(p.uid, p.name, 'cylinder', c, '', p.uid, np.array([0, 0, 1.0]),
                                   np.array([1.0, 0, 0]), 2 * p.radius, 2 * p.radius, p.height, walls=1,
                                   kg_m=rig_part_mass(p) / max(p.height, 1e-3), bottom=float(p.z - p.height / 2)))
            else:
                kind = 'plate' if p.height <= FR.PLATFORM_MAX_H else 'box'
                out.append(Element(p.uid, p.name, kind, c, '', p.uid, R=_rz(p.yaw),
                                   half=np.array([p.length, p.width, p.height]) / 2,
                                   bottom=float(p.z - p.height / 2)))
        for d in FR._devices(project, v, rf, FR.FramingSettings()):
            out.append(_device_element(d, False))
    return out, notes


def _rz(yaw):
    a = math.radians(yaw)
    return np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])


def _device_element(d, on_frame) -> Element:
    if d['round']:
        ax = d['R'][:, 2]
        return Element(d['uid'], d['tag'], 'cylinder', np.asarray(d['centre']), d['uid'], d['uid'], ax,
                       d['R'][:, 0], 2 * float(d['half'][0]), 2 * float(d['half'][0]), 2 * float(d['half'][2]),
                       walls=1, kg_m=d['mass'] / max(2 * float(d['half'][2]), 1e-3),
                       bottom=float(d['base'][2]))
    return Element(d['uid'], d['tag'], 'box', np.asarray(d['centre']), d['uid'], d['uid'], R=d['R'],
                   half=np.asarray(d['half'], float), bottom=float(d['base'][2]))


# =========================================================================== forces
def element_force(el: Element, U: np.ndarray, rho, nu, speed_factor=1.0) -> dict:
    """The element's aerodynamic force (vector N) in the relative wind U (m/s, the air's velocity relative to the
    car) with its local speed factor: {'F', 'cd', 'area', 're', 'parts'}."""
    U = np.asarray(U, float) * speed_factor
    Um = float(np.linalg.norm(U))
    if Um < 1e-9:
        return {'F': np.zeros(3), 'cd': 0.0, 'area': 0.0, 're': 0.0, 'parts': {}}
    u = U / Um
    q = 0.5 * rho
    F = np.zeros(3)
    parts = {}
    if el.kind in ('prism', 'cylinder'):
        a = el.axis
        Ua = float(U @ a)
        Un = U - Ua * a
        un = float(np.linalg.norm(Un))
        cd = area = re = 0.0
        if un > 1e-9:
            n = Un / un
            D = el.width_across(n)
            re = un * D / nu
            ld = el.L / max(D, 1e-6) * (2.0 if el.walls >= 1 else 1.0)
            k = 1.0 if el.walls >= 2 else k_aspect(ld)
            cd2 = cd_circle(re) if el.kind == 'cylinder' else cd_rect(el.depth_along(n) / max(D, 1e-6))
            cd = cd2 * k
            area = D * el.L
            Fn = q * un * un * cd * area * n
            F += Fn
            parts['pressure_n'] = float(np.linalg.norm(Fn))
        if abs(Ua) > 1e-9:
            perim = (math.pi * el.b) if el.kind == 'cylinder' else 2 * (el.b + el.d)
            Ff = q * Ua * Ua * cf_turbulent(abs(Ua) * el.L / nu) * perim * el.L * math.copysign(1.0, Ua) * a
            end_area = (math.pi * el.b ** 2 / 4) if el.kind == 'cylinder' else el.b * el.d
            up = 0 if Ua > 0 else 1                 # the end facing into the wind
            Fe = (q * Ua * Ua * END_CD * end_area * math.copysign(1.0, Ua) * a) if el.free_ends[up] else 0.0
            F += Ff + Fe
            parts['friction_n'] = float(np.linalg.norm(Ff))
            parts['end_n'] = float(np.linalg.norm(Fe)) if not isinstance(Fe, float) else 0.0
        return {'F': F, 'cd': cd, 'area': area, 're': re, 'parts': parts}
    R, h = el.R, el.half
    proj = 4 * (h[1] * h[2] * abs(R[:, 0] @ u) + h[0] * h[2] * abs(R[:, 1] @ u) + h[0] * h[1] * abs(R[:, 2] @ u))
    depth = 2 * float(np.abs(R.T @ u) @ h)
    front = math.sqrt(max(proj, 1e-12))
    if el.kind == 'plate':
        # the edge across the wind: a long rectangle; skin friction on both faces
        k = int(np.argmin(h))
        thick = 2 * h[k]
        edge = proj
        cd = cd_rect(depth / max(thick, 1e-6))
        Fp = q * Um * Um * cd * edge * u
        L = depth
        plan = 4 * float(np.prod(np.delete(h, k)))
        Ff = q * Um * Um * cf_turbulent(Um * L / nu) * 2 * plan * u
        F = Fp + Ff
        parts = {'pressure_n': float(np.linalg.norm(Fp)), 'friction_n': float(np.linalg.norm(Ff))}
        return {'F': F, 'cd': cd, 'area': edge, 're': Um * L / nu, 'parts': parts}
    cd = cd_box(depth / max(front, 1e-6))
    F = q * Um * Um * cd * proj * u
    return {'F': F, 'cd': cd, 'area': proj, 're': Um * front / nu, 'parts': {'pressure_n': float(np.linalg.norm(F))}}


def forces(els, U, rho, nu, rf, st: AeroSettings) -> list[dict]:
    """Every element's force in the relative wind U (vector m/s), with the roof's boundary layer and shielding:
    [{'id', 'name', 'F' (3,), 'drag_n' (against the motion, −Fx… along −x), 'cd', 'area', 're', 'speed_factor',
    'shield'}]."""
    Um = float(np.linalg.norm(U))
    u = U / max(Um, 1e-12)
    out = []
    for el in els:
        x_run = rf['x_front'] - float(el.centre[0]) + 0.3
        z_above = float(el.centre[2]) - rf['top']
        sf = bl_factor(max(z_above, 0.005), x_run, Um, nu) * st.roof_speed_factor
        r = element_force(el, U, rho, nu, sf)
        out.append({'id': el.id, 'name': el.name, 'kind': el.kind, 'F': r['F'], 'cd': r['cd'], 'area': r['area'],
                    're': r['re'], 'speed_factor': sf, 'shield': 1.0, 'parts': r['parts']})
    # shielding: the strongest upstream shadow on each element
    outl = [el.outline(u) for el in els]
    for j, ej in enumerate(els):
        best = 1.0
        cj, hj = outl[j]
        aj = 4 * hj[0] * hj[1]
        if aj <= 0:
            continue
        for i, ei in enumerate(els):
            if i == j:
                continue
            s = float((ej.centre - ei.centre) @ u)
            if s <= 0:
                continue
            ci, hi = outl[i]
            ov = np.maximum(0.0, np.minimum(cj + hj, ci + hi) - np.maximum(cj - hj, ci - hi))
            fo = float(ov[0] * ov[1]) / aj
            if fo <= 0:
                continue
            Di = max(2 * min(hi[0], hi[1]), 1e-3)
            f = 1.0 - min(fo, 1.0) * (1.0 - tandem(s / Di))
            best = min(best, f)
        out[j]['shield'] = best
        out[j]['F'] = out[j]['F'] * best
    for o in out:
        o['drag_n'] = float(-o['F'][0])
    return out


# =========================================================================== the vehicle
def vehicle_defaults(project, st: AeroSettings) -> dict:
    """{'family', 'cd', 'cd_source', 'area', 'area_source', 'mass', 'mass_source', 'width', 'height'}."""
    from . import suspension as SU
    from . import vehicles as VH
    spec = project.vehicle.spec or {}
    mesh = project.vehicle.mesh
    W = float(spec.get('Width') or 0.0)
    H = float(spec.get('Height') or 0.0)
    if (W <= 0 or H <= 0) and mesh is not None and getattr(mesh, 'n_tris', 0):
        lo, hi = mesh.bbox
        W = W or float(hi[1] - lo[1])
        H = H or float(hi[2] - lo[2])
    W, H = W or 1.8, H or 1.5
    fam = ''
    name = (project.vehicle.spec_name or project.vehicle.name or '').lower()
    for m in VH.library():
        if m.id.lower() in name or m.title.lower() in name or (name and name in m.title.lower()):
            fam = m.family
            break
    if not fam:
        fam = 'van' if H > 1.9 else ('suv' if H > 1.62 else 'sedan')
    cd = st.cd_base if st.cd_base > 0 else FAMILY_CD.get(fam, 0.32)
    area = st.frontal_m2 if st.frontal_m2 > 0 else AREA_FACTOR * W * H
    try:
        kerb = float(SU.settings_for(project).kerb_mass)
        msrc = 'core.suspension kerb mass + a 75 kg driver'
    except Exception:                               # an unreadable suspension block: the typical kerb
        kerb, msrc = SU.TYP_KERB_KG, 'typical kerb + a 75 kg driver'
    mass = st.mass_kg if st.mass_kg > 0 else kerb + SU.OCCUPANT_KG
    return {'family': fam, 'cd': cd, 'cd_source': 'set' if st.cd_base > 0 else SOURCES['cd_base'],
            'area': area, 'area_source': 'set' if st.frontal_m2 > 0 else SOURCES['area'], 'mass': mass,
            'mass_source': 'set' if st.mass_kg > 0 else msrc, 'width': W, 'height': H}


def road_load(V, rho, cda, mass, crr) -> float:
    return crr * mass * G + 0.5 * rho * cda * V * V


def consumption(V, rho, cda, mass, st: AeroSettings, extra_n=0.0) -> float:
    """Wh/km at V m/s with extra_n of added resistance."""
    return ((road_load(V, rho, cda, mass, st.crr) + extra_n) / st.drive_eff + st.aux_w / max(V, 0.1)) / 3.6


# =========================================================================== the study
@dataclass
class AeroResult:
    settings: AeroSettings
    variant: str
    variant_uid: str
    air: dict
    vehicle: dict
    elements: list            # per element: its geometry summary and drag at each report speed
    speeds: list              # per report speed: drag, CdA, power, consumption, range loss
    curve: dict               # {'kmh': [...], 'drag_n', 'loss_pct', 'added_w'}
    load_case: dict           # {'kmh', 'crosswind_ms', 'yaw_deg', 'forces' per element, 'devices' (beam chain)}
    shedding: list            # per element and mode: lock-in speeds
    tones: list               # per element: aeolian tone at each report speed
    loads: dict               # {FE key: (Fx, Fy, Fz)} of the load case, for core.fea.static_wind
    tv_loads: dict            # {thermo_vib key: (Fx, Fy, Fz)}
    modes_source: str
    findings: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    rig_kg: float = 0.0


def _modes_for(project, els, st, fea_result, variant):
    """[(element id, mode label, f Hz, share of the mode's motion across the wind)] from the FE result, else from
    core.thermo_vib's beam modes."""
    out = []
    if fea_result is not None and fea_result.modes:
        X = fea_result.shapes.get('X')
        word_dir = {'fore-aft': np.array([1.0, 0, 0]), 'side to side': np.array([0, 1.0, 0]),
                    'up and down': np.array([0, 0, 1.0])}
        for m_i, m in enumerate(fea_result.modes):
            for el in els:
                lift = _lift_dir(el)
                if lift is None:
                    continue
                d = m['devices'].get(el.name)
                share = None
                if d is not None:                     # its motion along the shedding force (mode scaled to 1 mm)
                    share = abs(float(np.asarray(d['move_mm'], float) @ lift))
                elif el.name in m.get('parts', {}):
                    share = abs(float(np.asarray(m['parts'][el.name], float) @ lift))
                elif m['part'] == el.name and m['motion'] in word_dir:
                    share = abs(float(word_dir[m['motion']] @ lift))
                if share is not None and share > 0.3:
                    out.append((el.id, f'FE mode {m["mode"]} ({m["part"]}, {m["motion"]})', m['f_hz'], share))
        return out, 'core.fea (the frame\'s FE modes)'
    from . import thermo_vib as TV
    tst = TV.settings_of(project)
    p = project
    v = variant or p.variant
    rf = TV.roof_box(p)
    tree = TV.rig_tree(p, rf)
    devs = TV.devices(p, v)
    nodes = [TV.carrier(tree, d['P'], rf) for d in devs]
    pm = TV.part_modes(tree, TV._carried(tree, devs, nodes), tst)
    by_uid = {}
    for i, md in pm.items():
        by_uid[tree[i]['uid']] = md
    for el in els:
        md = by_uid.get(el.tv_key)
        if md is not None:
            out.append((el.id, f'{md.name} beam mode', md.f1, 1.0))
    for d, (node, Q) in zip(devs, nodes):
        md = TV._bracket_mode(d, node, Q, rf, tst, None)
        el = next((e for e in els if e.tv_key == d['uid']), None)
        if el is not None:
            out.append((el.id, f'{md.name} beam mode', md.f1, 1.0))
            for i in (TV.chain_of(tree, node) if isinstance(node, int) else []):
                if i in pm:
                    out.append((el.id, f'{pm[i].name} beam mode (carrying {d["tag"]})', pm[i].f1, 1.0))
    return out, 'core.thermo_vib (one-mode beam estimates of the rig parts as drawn)'


def _lift_dir(el: Element, flow=(1.0, 0.0, 0.0)):
    """The direction of an element's vortex-shedding force: across both its axis and the flow (None for a box, a
    plate or an element along the flow)."""
    if el.kind not in ('prism', 'cylinder'):
        return None
    l_ = np.cross(el.axis, np.asarray(flow, float))
    n = float(np.linalg.norm(l_))
    return l_ / n if n > 0.2 else None


def run(project, engine=None, settings: AeroSettings | dict | None = None, variant=None, progress=None, cancel=None,
        frame=None, fea_result=None) -> AeroResult | None:
    """Drag and range at the report speeds, the curve over speed, the wind load case, shedding against the modes
    (core.fea's when given or cached, else the beam modes)."""
    from . import framing as FR
    from . import thermo_vib as TV
    st = settings if isinstance(settings, AeroSettings) else (
        AeroSettings.from_dict(settings) if isinstance(settings, dict) else settings_of(project))
    v = variant or project.variant
    prog = progress or (lambda f: None)
    stop = cancel or (lambda: False)
    rho, nu = air(st.air_c, st.pressure_kpa)
    rf = TV.roof_box(project)
    if st.use_frame and frame is None:
        frame = FR.synthesise(project, variant=v)
    els, notes = elements(project, st, frame, v)
    veh = vehicle_defaults(project, st)
    cda = veh['cd'] * veh['area']
    from .suspension import rig_part_mass
    rig_kg = (frame.mass.get('frame_kg', 0.0) if (st.use_frame and frame is not None and frame.members) else
              sum(rig_part_mass(p) for p in getattr(project, 'rig', []) or [] if p.visible))
    if stop():
        return None
    # ---------------------------------------------------------------- the report speeds
    per_el = {el.id: {'id': el.id, 'name': el.name, 'kind': el.kind, 'key': el.key, 'tv_key': el.tv_key,
                      'centre_m': [float(x) for x in el.centre]} for el in els}
    speeds = []
    for kmh in (st.speed1_kmh, st.speed2_kmh):
        V = kmh / 3.6
        fs = forces(els, np.array([-V, 0.0, 0.0]), rho, nu, rf, st)
        D = sum(f['drag_n'] for f in fs)
        for f in fs:
            per_el[f['id']].setdefault('at', {})[f'{kmh:g}'] = {
                'drag_n': f['drag_n'], 'cd': f['cd'], 'area_m2': f['area'], 're': f['re'],
                'speed_factor': f['speed_factor'], 'shield': f['shield']}
        speeds.append(_range_row(kmh, D, rho, cda, veh['mass'], rig_kg, st))
    prog(0.3)
    # ---------------------------------------------------------------- the curve
    kmhs = np.arange(10.0, max(st.v_max_kmh, st.speed2_kmh) + 0.1, 10.0)
    curve = {'kmh': [], 'drag_n': [], 'loss_pct': [], 'added_w': []}
    for kmh in kmhs:
        V = kmh / 3.6
        D = sum(f['drag_n'] for f in forces(els, np.array([-V, 0.0, 0.0]), rho, nu, rf, st))
        row = _range_row(kmh, D, rho, cda, veh['mass'], rig_kg, st, measured=False)
        curve['kmh'].append(float(kmh))
        curve['drag_n'].append(D)
        curve['loss_pct'].append(row['range_loss_pct'])
        curve['added_w'].append(row['added_power_w'])
    if stop():
        return None
    prog(0.5)
    # ---------------------------------------------------------------- the wind load case
    V = st.load_kmh / 3.6
    U = np.array([-V, -st.crosswind_ms, 0.0])
    fs = forces(els, U, rho, nu, rf, st)
    loads, tv_loads = {}, {}
    lc = []
    for el, f in zip(els, fs):
        lc.append({'id': el.id, 'name': el.name, 'force_n': [float(x) for x in f['F']],
                   'force_abs_n': float(np.linalg.norm(f['F']))})
        if el.key:
            loads[el.key] = tuple(float(x) for x in f['F'])
        if el.tv_key:
            prev = np.asarray(tv_loads.get(el.tv_key, (0.0, 0.0, 0.0)))
            tv_loads[el.tv_key] = tuple(float(x) for x in prev + f['F'])
    try:
        beam = TV.wind_response(project, tv_loads, variant=v)['devices']
    except Exception as e:                           # the beam chain is a companion figure
        beam = []
        notes.append(f'core.thermo_vib.wind_response failed: {type(e).__name__}: {e}')
    load_case = {'kmh': st.load_kmh, 'crosswind_ms': st.crosswind_ms,
                 'yaw_deg': math.degrees(math.atan2(st.crosswind_ms, V)), 'airspeed_ms': float(np.linalg.norm(U)),
                 'forces': lc, 'total_n': [float(x) for x in sum((f['F'] for f in fs), np.zeros(3))],
                 'devices': [d for d in beam if d['uid'] in {el.tv_key for el in els} or d['force_n'] > 0]}
    prog(0.7)
    # ---------------------------------------------------------------- shedding and lock-in
    if fea_result is None:
        try:
            from . import fea as FE
            fea_result = FE.cached(project, v)
        except Exception:                            # no FE result: the beam modes
            fea_result = None
    modes, msrc = _modes_for(project, els, st, fea_result, v)
    by = {el.id: el for el in els}
    sf_at = {f['id']: f['speed_factor'] for f in forces(els, np.array([-st.speed2_kmh / 3.6, 0, 0]), rho, nu, rf,
                                                         st)}
    shed = []
    for eid, label, fn, share in modes:
        el = by[eid]
        if el.kind not in ('prism', 'cylinder') or fn <= 0:
            continue
        n = np.array([1.0, 0, 0]) - (el.axis @ [1.0, 0, 0]) * el.axis
        if np.linalg.norm(n) < 0.2:
            continue                                  # along the flow: no cross-flow shedding
        n /= np.linalg.norm(n)
        D = el.width_across(n)
        St_ = ST_CIRCLE if el.kind == 'cylinder' else ST_SQUARE
        k = max(sf_at.get(eid, 1.0), 1e-3)
        v_lock = fn * D / (St_ * k)
        m = el.kg_m if el.kg_m > 0 else 0.0
        sc = 4 * math.pi * m * st.zeta / (rho * D * D) if m > 0 else None
        shed.append({'element': el.name, 'id': eid, 'mode': label, 'f_hz': fn, 'D_m': D, 'strouhal': St_,
                     'lock_kmh': v_lock * 3.6, 'band_kmh': [v_lock * 3.6 * (1 - st.lockin_band),
                                                           v_lock * 3.6 * (1 + st.lockin_band)],
                     'in_range': v_lock * 3.6 * (1 - st.lockin_band) <= st.v_max_kmh, 'scruton': sc,
                     'share': share})
    tones = []
    for el in els:
        if el.kind not in ('prism', 'cylinder'):
            continue
        n = np.array([1.0, 0, 0]) - (el.axis @ [1.0, 0, 0]) * el.axis
        if np.linalg.norm(n) < 0.2:
            continue
        n /= np.linalg.norm(n)
        D = el.width_across(n)
        St_ = ST_CIRCLE if el.kind == 'cylinder' else ST_SQUARE
        tones.append({'element': el.name, 'D_m': D, 'hz': {f'{k:g}': St_ * (k / 3.6) * sf_at.get(el.id, 1.0) / D
                                                           for k in (st.speed1_kmh, st.speed2_kmh)}})
    res = AeroResult(st, v.name, v.uid, {'rho_kg_m3': rho, 'nu_m2_s': nu, 't_c': st.air_c, 'p_kpa': st.pressure_kpa},
                     dict(veh, cda_m2=cda, rig_kg=rig_kg), list(per_el.values()), speeds, curve, load_case,
                     sorted(shed, key=lambda s: s['lock_kmh']), tones, loads, tv_loads, msrc, [], notes, rig_kg)
    res.findings = _findings(res)
    prog(1.0)
    return res


def _range_row(kmh, D, rho, cda, mass, rig_kg, st: AeroSettings, measured=True) -> dict:
    V = kmh / 3.6
    extra = D + st.crr * rig_kg * G
    if measured and st.consumption_wh_km > 0 and abs(kmh - st.speed2_kmh) < 1e-6:
        base = st.consumption_wh_km
        src = 'measured (set)'
    else:
        base = consumption(V, rho, cda, mass, st)
        src = 'road-load model'
    added = extra / st.drive_eff / 3.6
    loss = 100.0 * (1.0 - base / (base + added))
    rng = st.battery_kwh * 1000.0 / base
    return {'kmh': kmh, 'drag_n': D, 'rig_cda_m2': D / (0.5 * rho * V * V) if V > 0 else 0.0,
            'mass_n': st.crr * rig_kg * G, 'added_power_w': extra * V / st.drive_eff, 'base_wh_km': base,
            'added_wh_km': added, 'range_loss_pct': loss, 'range_km': rng, 'range_with_km': st.battery_kwh * 1000.0 /
            (base + added), 'base_source': src}


def _finding(sev, kind, message, uids=()):
    return {'severity': sev, 'kind': kind, 'message': message, 'uids': list(uids)}


def _findings(res: AeroResult) -> list:
    out = []
    if not res.elements:
        out.append(_finding('note', 'empty', 'Nothing above the roof: no rig drag.'))
        return out
    hi = res.speeds[-1]
    out.append(_finding('good', 'summary', f'The rig adds {hi["drag_n"]:.0f} N of drag at {hi["kmh"]:g} km/h '
                        f'(CdA {hi["rig_cda_m2"]:.3f} m² on the car\'s {res.vehicle["cda_m2"]:.2f} m²): '
                        f'{hi["range_loss_pct"]:.1f} % less range ({hi["range_km"]:.0f} → '
                        f'{hi["range_with_km"]:.0f} km on {res.settings.battery_kwh:g} kWh).'))
    if hi['range_loss_pct'] > 10:
        out.append(_finding('warning', 'range', f'{hi["range_loss_pct"]:.0f} % range loss at {hi["kmh"]:g} km/h: '
                            'fairings, aero bars or a lower mast cut it.'))
    seen = set()
    for s in res.shedding:
        if not s['in_range'] or (s['element'], s['mode']) in seen:
            continue
        seen.add((s['element'], s['mode']))
        sc = s['scruton']
        sev = ('serious' if sc < SC_SERIOUS else 'warning' if sc < SC_NOTE else 'note') if sc is not None \
            else 'warning'
        out.append(_finding(sev, 'lock_in', f'{s["element"]} sheds vortices at {s["mode"]} ({s["f_hz"]:.1f} Hz) at '
                            f'{s["lock_kmh"]:.0f} km/h (lock-in {s["band_kmh"][0]:.0f}-{s["band_kmh"][1]:.0f} km/h)'
                            + (f', Scruton number {sc:.0f}' if sc is not None else '') +
                            (': its mass and damping keep the motion small.' if sev == 'note' else
                             ': a strake, a different section or a stiffer member moves it out of the driving '
                             'speeds.')))
    for d in res.load_case.get('devices', []):
        if d['turn_abs_deg'] > 0.1:
            out.append(_finding('warning', 'wind_turn', f'{d["tag"]} turns {d["turn_abs_deg"]:.2f}° in the '
                                f'{res.load_case["kmh"]:g} km/h + {res.load_case["crosswind_ms"]:g} m/s crosswind '
                                'case (beam chain): check its calibration tolerance.', [d['uid']]))
    return out


# =========================================================================== outputs
def inputs_key(project, variant=None) -> str:
    from . import framing as FR
    a = project.analysis
    return FR.inputs_key(project, variant) + json.dumps({'aero': a.aero, 'suspension': a.suspension,
                                                        'thermo_vib': a.thermo_vib, 'fea': a.fea},
                                                       sort_keys=True, default=str)


_ISSUES: dict = {}


def issues(project, vg=None) -> list:
    """Lock-in within the driving speeds and a large range loss (beam modes; cached on the inputs)."""
    from . import rules
    if not any(p.visible for p in getattr(project, 'rig', []) or []):
        return []
    key = inputs_key(project)
    if key in _ISSUES:
        return list(_ISSUES[key])
    res = run(project, fea_result=None)
    out = [rules.Issue(f['severity'], f['message'], f'{res.variant} · aerodynamics', list(f['uids']), 'aero')
           for f in res.findings if f['severity'] in ('critical', 'serious', 'warning')]
    if len(_ISSUES) > 8:
        _ISSUES.pop(next(iter(_ISSUES)))
    _ISSUES[key] = out
    return list(out)


def _r(v, d=4):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return round(v, d) if math.isfinite(v) else None


def _clean(o, d=5):
    if isinstance(o, dict):
        return {k: _clean(v, d) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v, d) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist(), d)
    if isinstance(o, (float, np.floating)):
        return _r(o, d)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def summary(res: AeroResult) -> dict:
    return {'variant': res.variant, 'rig_kg': _r(res.rig_kg, 2),
            'speeds': [{'kmh': s['kmh'], 'drag_n': _r(s['drag_n'], 1), 'rig_cda_m2': _r(s['rig_cda_m2'], 4),
                        'range_loss_pct': _r(s['range_loss_pct'], 2), 'range_km': _r(s['range_km'], 1),
                        'range_with_km': _r(s['range_with_km'], 1)} for s in res.speeds],
            'lock_in': [{'element': s['element'], 'mode': s['mode'], 'lock_kmh': _r(s['lock_kmh'], 1)}
                        for s in res.shedding if s['in_range']],
            'modes_from': res.modes_source,
            'findings': [f['message'] for f in res.findings if f['severity'] in ('critical', 'serious', 'warning')]}


def to_doc(res: AeroResult, project=None) -> dict:
    return {'schema': SCHEMA, 'project': getattr(project, 'name', ''), 'variant': res.variant, 'frame': FRAME,
            'units': {'speed': 'km/h (m/s where stated)', 'force': 'N', 'energy': 'Wh/km', 'range': 'km',
                      'area': 'm²', 'frequency': 'Hz', 'move': 'mm', 'turn': 'deg'},
            'not_cfd': 'Published bluff-body coefficients times the rig\'s areas: not CFD.',
            'assumptions': ASSUMPTIONS, 'model': __doc__.split('Out of scope:')[0].strip(),
            'limits': 'Out of scope: ' + __doc__.split('Out of scope:')[1].strip(), 'sources': SOURCES,
            'settings': res.settings.to_dict(), 'air': _clean(res.air, 7), 'vehicle': _clean(res.vehicle),
            'elements': _clean(res.elements), 'speeds': _clean(res.speeds), 'curve': _clean(res.curve),
            'load_case': _clean(res.load_case), 'shedding': _clean(res.shedding), 'tones': _clean(res.tones),
            'modes_from': res.modes_source,
            'findings': [{'severity': f['severity'], 'kind': f['kind'], 'message': f['message']} for f in res.findings],
            'notes': list(res.notes)}


def drag_csv(res: AeroResult) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator='\n')
    ks = [f'{k:g}' for k in (res.settings.speed1_kmh, res.settings.speed2_kmh)]
    w.writerow(['element', 'kind'] + [f'{q}_{k}' for k in ks for q in ('drag_n', 'cd', 'area_m2', 'shield')])
    for e in res.elements:
        row = [e['name'], e['kind']]
        for k in ks:
            a = e.get('at', {}).get(k, {})
            row += [_r(a.get('drag_n'), 3), _r(a.get('cd'), 3), _r(a.get('area_m2'), 5), _r(a.get('shield'), 3)]
        w.writerow(row)
    return buf.getvalue()


def shedding_csv(res: AeroResult) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator='\n')
    w.writerow(['element', 'mode', 'f_hz', 'D_m', 'strouhal', 'lock_kmh', 'band_lo_kmh', 'band_hi_kmh', 'in_range',
                'scruton'])
    for s in res.shedding:
        w.writerow([s['element'], s['mode'], _r(s['f_hz'], 3), _r(s['D_m'], 4), s['strouhal'], _r(s['lock_kmh'], 1),
                    _r(s['band_kmh'][0], 1), _r(s['band_kmh'][1], 1), s['in_range'], _r(s['scruton'], 2)])
    return buf.getvalue()


def png(res: AeroResult, size=(900, 420)) -> bytes:
    """Range loss and rig drag over speed."""
    import cv2
    W, H = size
    img = np.full((H, W, 3), 250, np.uint8)
    k, loss, drag = res.curve['kmh'], res.curve['loss_pct'], res.curve['drag_n']
    if len(k) >= 2:
        x0, x1, y0, y1 = 70, W - 70, 30, H - 50
        lm = max(max(loss), 1e-6) * 1.1
        dm = max(max(drag), 1e-6) * 1.1
        cv2.rectangle(img, (x0, y0), (x1, y1), (200, 200, 200), 1)

        def pt(kv, val, top):
            return (int(x0 + (kv - k[0]) / (k[-1] - k[0]) * (x1 - x0)), int(y1 - val / top * (y1 - y0)))
        for a, b in zip(range(len(k) - 1), range(1, len(k))):
            cv2.line(img, pt(k[a], loss[a], lm), pt(k[b], loss[b], lm), (40, 90, 200), 2, cv2.LINE_AA)
            cv2.line(img, pt(k[a], drag[a], dm), pt(k[b], drag[b], dm), (60, 160, 60), 2, cv2.LINE_AA)
        cv2.putText(img, f'range loss % (max {max(loss):.1f})', (x0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 90, 200),
                    1, cv2.LINE_AA)
        cv2.putText(img, f'rig drag N (max {max(drag):.0f})', (x0 + 300, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (60, 160, 60), 1, cv2.LINE_AA)
        for kv in k[::2]:
            cv2.putText(img, f'{kv:g}', (pt(kv, 0, 1)[0] - 10, y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (90, 90, 90),
                        1, cv2.LINE_AA)
        cv2.putText(img, 'km/h', (x1 - 30, y1 + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (90, 90, 90), 1, cv2.LINE_AA)
    ok, buf = cv2.imencode('.png', img)
    return buf.tobytes() if ok else b''


def files(res: AeroResult, project=None) -> list:
    return [('aero.json', (json.dumps(to_doc(res, project), indent=1) + '\n').encode()),
            ('aero_drag.csv', drag_csv(res).encode()), ('aero_shedding.csv', shedding_csv(res).encode()),
            ('aero.png', png(res))]


def export(res: AeroResult, project, out_dir) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for name, data in files(res, project):
        path = os.path.join(out_dir, name)
        with open(path, 'wb') as fh:
            fh.write(data)
        paths.append(path)
    return paths


# =========================================================================== validation
# Consumer Reports, "How Rooftop Carriers Affect Fuel Economy" (28 May 2020): steady 65 mph on a highway, both
# directions. 2019 Nissan Altima: about 48 mpg bare, −11 % with the roof rack only, −19 % with rack and carrier.
# 2019 Toyota RAV4: 41 mpg bare, −2 % with the empty rack, −13 % with rack and carrier. Neither the rack nor the box
# is specified.
CR_2020 = {'altima': {'rack_pct': 11.0, 'box_pct': 19.0}, 'rav4': {'rack_pct': 2.0, 'box_pct': 13.0},
           'mph': 65.0, 'source': 'Consumer Reports, How Rooftop Carriers Affect Fuel Economy, 2020-05-28'}


def generic_rack_loss(kmh=104.6, cd=0.26, W=1.850, H=1.445, mass=1530.0, bar_w=0.06, bar_h=0.025, bar_len=1.2,
                      height=0.07, st: AeroSettings | None = None) -> dict:
    """Fuel (energy) loss of two crossbars on a midsize sedan at 65 mph from this module's coefficients, with
    st.aux_w = 0 and constant drive efficiency (an engine at steady speed: % fuel ≈ % road-load energy). The car's
    figures are a 2019 Altima's class (Cd 0.26 as Nissan states, 1.85 × 1.445 m); the bars a typical rectangular
    60 × 25 mm steel bar on 70 mm feet (the test's bars are not stated)."""
    st = st or AeroSettings(aux_w=0.0)
    rho, nu = air(20.0)
    rf = {'x_front': 1.0, 'top': 1.445}
    els = []
    for i, x in enumerate((0.85, 0.05)):
        els.append(Element(f'bar{i}', f'bar {i}', 'prism', np.array([x, 0.0, rf['top'] + height]), '', '',
                           np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, -1.0]), bar_h, bar_w, bar_len,
                           free_ends=(True, True)))
    V = kmh / 3.6
    D = sum(f['drag_n'] for f in forces(els, np.array([-V, 0.0, 0.0]), rho, nu, rf, st))
    cda = cd * AREA_FACTOR * W * H
    base = road_load(V, rho, cda, mass, st.crr)
    return {'drag_n': D, 'base_n': base, 'loss_pct': 100.0 * (1 - base / (base + D)), 'rig_cda_m2': D / (0.5 * rho * V * V)}


def validation_checks() -> list[dict]:
    def cylinder():
        rho, nu = air(20.0)
        el = Element('c', 'c', 'cylinder', np.zeros(3), axis=np.array([0, 0, 1.0]), ex=np.array([1.0, 0, 0]), b=0.05,
                     d=0.05, L=1e6, walls=2)
        V = 1e5 * nu / 0.05
        F = element_force(el, np.array([-V, 0, 0]), rho, nu)['F']
        cd = float(np.linalg.norm(F)) / (0.5 * rho * V * V * 0.05 * 1e6)
        # independent: Wieselsberger's measurements as in Schlichting, Boundary-Layer Theory, Fig. 1.12: Cd ≈ 1.2 at
        # Re 10⁴-10⁵; Strouhal 0.21 (Roshko 1954, NACA TN 3169, Re 10³-10⁵)
        e_cd = abs(cd / 1.2 - 1) * 100
        e_st = abs(ST_CIRCLE / 0.21 - 1) * 100
        return {'error': max(e_cd, e_st), 'detail': {'cd_model': cd, 'cd_ref': 1.2, 'st_model': ST_CIRCLE,
                                                      'st_ref': 0.21}}

    def rack():
        r = generic_rack_loss()
        lo, hi = CR_2020['rav4']['rack_pct'], CR_2020['altima']['rack_pct']
        inside = lo <= r['loss_pct'] <= hi
        err = 0.0 if inside else min(abs(r['loss_pct'] - lo), abs(r['loss_pct'] - hi))
        return {'error': err, 'detail': dict(r, measured_band_pct=[lo, hi], source=CR_2020['source'])}
    return [
        {'id': 'aero.cylinder', 'model': 'aero', 'title': 'Long circular cylinder: drag and Strouhal number',
         'quantity': 'worst of |Cd/1.2 − 1| at Re 10⁵ and |St/0.21 − 1|, %', 'reference':
         'Wieselsberger\'s measurements (Schlichting, Boundary-Layer Theory, Fig. 1.12) and Roshko 1954 (NACA TN '
         '3169)', 'tolerance': 10.0, 'unit': '%', 'refs': (), 'fn': cylinder, 'cost': 'cheap',
         'why': 'the model takes its constants from White and Blevins: this checks them against the measurements '
                'those books summarise (a second source, not an independent experiment)'},
        {'id': 'aero.roof_rack_cr2020', 'model': 'aero', 'title': 'Bare roof rack against measured fuel economy',
         'quantity': 'percentage points outside the measured band of the two cars (0 inside)',
         'reference': CR_2020['source'] + ': rack only −2 % (RAV4) to −11 % (Altima) at 65 mph', 'tolerance': 2.0,
         'unit': 'pp', 'refs': (), 'fn': rack, 'cost': 'cheap',
         'why': 'measured on real cars; the racks are not stated, so the band of the two is the reference'},
    ]
