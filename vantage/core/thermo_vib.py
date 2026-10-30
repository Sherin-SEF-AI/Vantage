"""Thermal and vibration drift: how midday sun warps the rig and how the road shakes it, and what that does to every
camera's and sensor's calibration.

Frame and units. The vehicle frame (ISO 8855: x forward, y left, z up, origin at the rear-axle centre on the ground,
metres). Moves are given in millimetres in that frame; angle changes as the change of each device's own yaw, pitch
and roll (degrees, intrinsic Z-Y-X, pitch positive down, as the project states them). Temperatures in °C, frequencies
in Hz, accelerations in m/s² (and g), PSDs one-sided.

Solar thermal warping (hour by hour, relative to the calibration)
- The sun comes from core.exposure (the NOAA solar position): its location, its campaign's dates (the day with the
  highest noon sun in the date range, or a set date) and its daily driving window, one sample per hour. Clear-sky
  irradiance is the Meinel model (Meinel & Meinel 1976): direct normal DNI = S·0.7^(AM^0.678) with S the solar
  constant 1361 W/m² (Kopp & Lean 2011) over the sun's distance², the air mass AM of Kasten & Young (1989), diffuse
  horizontal 0.1·DNI (the usual companion of the Meinel model), ground albedo 0.2 (typical asphalt and grass).
- Ambient air follows a cosine day: the set peak at the peak hour (15:00), the set daily range lower 12 h away. The
  cameras are calibrated in a garage at the calibration temperature (15 °C) with every part at that temperature.
- Each rig part (core.rig: roof-rack crossbar, mast, enclosure, platform, mast head) is a thin-walled aluminium
  shell (3 mm, as core.suspension's rig masses) split into two lumped halves: the top and bottom halves of a bar or a
  box, the sun-lit and shaded halves of an upright mast. Each half's steady energy balance: absorbed sun (direct on
  every facet facing the sun, isotropic sky diffuse and ground-reflected light by the facet's view of sky and ground)
  = convection (h from the airflow speed: a flat plate's average Nusselt number for boxes and the roof, the
  Churchill-Bernstein correlation for round tubes in crossflow; natural convection as the floor, mixed by the cube
  rule) + radiation (to the sky at the Swinbank temperature 0.0552·T^1.5 and to the ground at air temperature, or to
  the roof below a part sitting on it) + conduction to the other half through the walls. The roof is a single steel
  plate, insulated below (the headliner). The body's structure stays at air temperature (it is shaded and heavy).
  A thin aluminium wall reaches steady state in minutes (each half's time constant is reported), so the steady
  balance hour by hour is the first-order answer.
- Deformations, composed along the chain body → roof → part → part → device:
  uniform expansion α·ΔT of each part about its base (and of the body about the vehicle origin, of the roof skin
  about its centre); thermal bowing, curvature κ = α·ΔT_gradient / depth (depth: between the two halves' centroids):
  a mast bends away from the sun as a cantilever (tip slope κ·a, deflection κ·a²/2), a crossbar bows up between its
  feet as a simply supported beam (w = κ/2·(s²/4 − η²)), a box domes as a plate on its corners; the rack-on-roof
  mismatch: an aluminium crossbar on feet standing on the steel roof. Boundary condition (feet setting): 'sliding'
  (the default: the feet pin the bar against the roof but one slides along it, so the bar expands freely about its
  centre and the mismatch (α_al·ΔT_bar − α_steel·ΔT_roof)·span shows as foot slip), or 'clamped' (both feet hold the
  bar's length to the roof's: the mismatch becomes an axial force E·A·strain, which amplifies the thermal bow by
  1/(1 − N/N_cr) with N_cr = π²EI/span² the pinned Euler load).
- Every device's move (Δx Δy Δz mm) and angle change (Δψ Δθ Δφ) relative to the calibration, hour by hour; for pairs
  that share a calibration (each camera with each lidar, stereo pairs as core.timing finds them) the relative angle
  in degrees and in the camera's image in pixels through its exact lens model (the worst over the image, plus the
  parallax of the relative move at a reference range at the image centre), against the calibration tolerance (the
  installation check's 1° / 10 mm, the observability study's requirement, or set ones).

Road vibration (random, stationary)
- ISO 8608 road classes A-H: Gd(n) = Gd(n0)·(n/n0)^-2 with n0 = 0.1 cycles/m and Gd(n0) the class's geometric mean
  (16·4^k × 10⁻⁶ m³), over the standard's spatial band 0.011-2.83 cycles/m; at a speed V the temporal PSD is
  G(f) = Gd(f/V)/V.
- A half-car (heave and pitch of the sprung body, one unsprung mass per axle, the tyres as springs) built from the
  project's suspension (core.suspension: wheel rates and tyre rates per axle, the kerb loads, unsprung mass per corner
  from the terrain study's inputs, typical 35 kg) plus the rig's mass, pitch inertia from a dynamic index of 1
  (k² = a·b, typical of cars, Gillespie 1992 ch. 5), damping ζ = 0.25 of each axle's critical (typical 0.2-0.4). The
  rear wheel sees the front wheel's road delayed by wheelbase/V (both tracks the same road: no roll). A half-car
  rather than a quarter-car because the mounts sit away from the centre of gravity: pitch adds vertical motion at the
  bumpers and fore-aft motion at roof height, and the wheelbase delay cancels and doubles some frequencies.
- Each mount flexes as uncoupled single modes on the body's motion: the device's own bracket (a cantilever of its
  standoff: a typical printed bracket for a camera, 20 × 6 mm section of the bracket generator's material, or a
  typical 40 × 5 mm aluminium plate for a lidar, radar or antenna; a generated bracket's beam stiffness from
  core.brackets replaces it), a mast (a cantilever of its height carrying everything above it) and a crossbar (a
  simply supported beam between its feet carrying its share of everything on it). First mode
  f1 = (1/2π)·√(k/m) with k = 3EI/L³ and m = m_tip + 0.24·m_beam for a cantilever (Rayleigh), k = 48EI/s³ and
  m = m_carried + 0.49·m_beam for a crossbar. Damping ζ = 0.02 of critical for bolted aluminium, 0.05 for a printed
  bracket (typical structural values, Harris' Shock and Vibration Handbook). Base excitation: the relative
  deflection per unit base acceleration is −1/(ωn² − ω² + 2iζωnω); the tip turns 1.5/L rad per metre of deflection.
- A mount resonates when its first mode lies in the body band (1-2 Hz), the wheel-hop band (10-15 Hz) or within 15 %
  of the peak of the body's acceleration at the mount at the set speed (and, on cobblestones, of the sett passing
  frequency V/pitch).
- Motion blur from the angular rate across the optical axis: ω_rms × exposure (core.timing's) × the lens's local
  px/rad at the image centre. Pairs: the relative flex of two devices' mounts (the body's own motion cancels).

Events: cobblestones as a rough ISO class at a lower speed (RMS figures, and the sett frequency); an expansion joint
(a step) and a speed bump (a half-sine) as transients: the half-car and every mount mode integrated together
(exact zero-order-hold discretisation, 0.5 ms), peak deflection, peak angle and the ring-down.

Out of scope: shading of one part by another or by the body, wind and rain, the sensors' own self-heating, the rig's
joints and clamp slip under load, the roof panel's doming, roll excitation (the left and right tracks differ),
coupled modes of a rig (each mode on the body's motion alone), the tyre's enveloping of short bumps (the step is
taken sharp: conservative), nonlinear dampers and bump stops. Typical figures are typical, not this rig's.
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

from . import suspension as SU
from .mathutil import BODY_TO_OPTICAL, euler_zyx, rot_zyx, wrap180

SCHEMA = 'vantage.thermo_vib/1'
FRAME = 'vehicle (ISO 8855, rear axle): x forward, y left, z up'
SEVERITIES = ('critical', 'serious', 'warning', 'note', 'good')
ISSUE_SEVERITIES = ('critical', 'serious', 'warning')
ISSUE_KINDS = ('resonance', 'thermal_drift')

# ---------------------------------------------------------------- materials (typical, with sources)
MATERIALS = {
    'al6061': {'label': 'Aluminium 6061-T6', 'E': 68.9e9, 'alpha': 23.6e-6, 'rho': 2700.0, 'c': 896.0, 'k': 167.0,
               'source': 'typical: ASM Aerospace Specification Metals, Aluminum 6061-T6 data sheet'},
    'steel': {'label': 'Low-carbon steel sheet', 'E': 205e9, 'alpha': 11.7e-6, 'rho': 7854.0, 'c': 434.0, 'k': 60.5,
              'source': 'typical: Incropera & DeWitt, Fundamentals of Heat and Mass Transfer, Table A.1 (plain '
                        'carbon steel); E of automotive sheet (EN 10130 DC04)'},
    'stainless': {'label': 'Stainless steel 304', 'E': 193e9, 'alpha': 17.3e-6, 'rho': 7900.0, 'c': 477.0,
                  'k': 14.9, 'source': 'typical: Incropera & DeWitt Table A.1 (AISI 304); ASM data sheet'},
    'cfrp': {'label': 'Carbon-fibre tube', 'E': 70e9, 'alpha': 2.0e-6, 'rho': 1550.0, 'c': 900.0, 'k': 5.0,
             'source': 'typical roll-wrapped carbon tube, quasi-isotropic lay-up (axial α −1 to +3 µm/m·K)'},
    'asa': {'label': 'ASA (printed)', 'E': 1.8e9, 'alpha': 95e-6, 'rho': 1070.0, 'c': 1300.0, 'k': 0.17,
            'source': 'typical printed ASA (core.brackets\' modulus; CTE of ASA data sheets, 90-100 µm/m·K)'},
}
RIG_MATERIALS = ('al6061', 'stainless', 'cfrp')
# solar absorptance and thermal emittance of a part's surface (typical)
FINISHES = {
    'black_anodised': {'label': 'Black anodised', 'a': 0.88, 'e': 0.88,
                       'source': 'Gilmore, Spacecraft Thermal Control Handbook (2002), App. A (black anodize)'},
    'clear_anodised': {'label': 'Clear anodised', 'a': 0.35, 'e': 0.80,
                       'source': 'Gilmore (2002), App. A (clear anodize, 0.27-0.35 / 0.76-0.84)'},
    'bare': {'label': 'Bare aluminium', 'a': 0.15, 'e': 0.05,
             'source': 'Incropera & DeWitt Table A.12 (aluminium foil 0.15 / 0.05)'},
    'white_paint': {'label': 'White paint', 'a': 0.26, 'e': 0.90,
                    'source': 'Incropera & DeWitt Table A.12 (white acrylic paint)'},
}
# roof paint colours: solar absorptance (1 − solar reflectance) and emittance of a car's clear-coated paint (typical)
ROOF_PAINTS = {
    'white': {'label': 'White', 'a': 0.25, 'e': 0.90},
    'silver': {'label': 'Silver', 'a': 0.45, 'e': 0.90},
    'colour': {'label': 'Mid colour', 'a': 0.70, 'e': 0.90},
    'black': {'label': 'Black', 'a': 0.95, 'e': 0.90},
}
PAINT_SOURCE = ('typical: Levinson et al., Potential benefits of solar reflective car shells, Applied Energy 88 '
                '(2011) 4343 (solar reflectance of white, silver, coloured and black car paints)')

WALL_M = SU.RIG_WALL_M          # rig parts as 3 mm shells, as core.suspension's mass estimate
ROOF_SKIN_M = 0.0007            # typical roof skin, 0.7 mm steel
SIGMA = 5.670374e-8             # Stefan-Boltzmann, W/m²K⁴
SOLAR_CONSTANT = 1361.0         # W/m² (Kopp & Lean 2011)
FEET_INSET_M = 0.06             # a rack's feet sit this far in from the bar's ends (core.rig's preset: 0.12 m spare)
CARRY_M = 0.25                  # a device this close to a rig part rides on it
SUPPORT_GAP_M = 0.03            # a part resting this close on another (or the roof) is carried by it
DOME_MARGIN = 0.0

# ---------------------------------------------------------------- road and vibration
ISO_CLASSES = 'ABCDEFGH'
ISO_N0 = 0.1                    # cycles/m
ISO_BAND = (0.011, 2.83)        # cycles/m, the spatial band of ISO 8608's classification
ISO_LABEL = {'A': 'A · very good', 'B': 'B · good', 'C': 'C · average', 'D': 'D · poor', 'E': 'E · very poor',
             'F': 'F', 'G': 'G', 'H': 'H'}
BODY_BAND = (1.0, 2.0)          # Hz: body bounce and pitch of passenger cars (typical)
HOP_BAND = (10.0, 15.0)         # Hz: wheel hop (typical)
NEAR_PEAK = 0.15                # a mode within this fraction of an excitation peak resonates with it
DYNAMIC_INDEX = 1.0             # k_pitch² / (a·b): typical of passenger cars 0.8-1.2 (Gillespie 1992, ch. 5)
F_LO, F_HI, PER_DECADE = 0.1, 300.0, 400
DT_S = 0.0005                   # event integration step
TAIL_S = 3.0                    # integrated after the last wheel leaves the input
RING_FRACTION = 0.1             # ring-down: until the response stays below this share of its peak
# typical brackets when no generated one exists: (material, section width m, thickness m, ζ)
TYP_BRACKET = {'camera': ('asa', 0.020, 0.006, 0.05), 'other': ('al6061', 0.040, 0.005, None)}
TYP_SENSOR_LEVER_M = 0.03       # a body-mounted lidar, radar or antenna: its bracket's typical lever
MIN_LEVER_M = 0.01

LIMITS = {'t_cal_c': (-30.0, 45.0), 't_max_c': (-20.0, 60.0), 'daily_range_c': (0.0, 30.0),
          'peak_hour': (10.0, 18.0), 'air_kmh': (0.0, 200.0), 'albedo': (0.0, 0.9), 'tol_deg': (0.001, 10.0),
          'tol_mm': (0.01, 100.0), 'px_limit': (0.05, 20.0), 'range_m': (1.0, 200.0), 'speed_kmh': (5.0, 200.0),
          'zeta_body': (0.05, 1.0), 'zeta_mount': (0.002, 0.3), 'zeta_printed': (0.002, 0.3),
          'bump_h_mm': (5.0, 300.0), 'bump_len_m': (0.1, 10.0), 'bump_kmh': (2.0, 120.0), 'joint_mm': (1.0, 100.0),
          'cobble_kmh': (5.0, 120.0), 'sett_m': (0.05, 0.5)}
CHOICES = {'material': RIG_MATERIALS, 'finish': tuple(FINISHES), 'roof_paint': tuple(ROOF_PAINTS),
           'feet': ('sliding', 'clamped'), 'tol_source': ('install', 'observability', 'custom'),
           'road_class': tuple(ISO_CLASSES), 'cobble_class': tuple(ISO_CLASSES)}
TOL_LABEL = {'install': 'installation check', 'observability': 'observability requirement', 'custom': 'set here'}

ASSUMPTIONS = ('Clear sky (Meinel model), the sun from core.exposure\'s NOAA position; ambient air a cosine day; rig '
               'parts thin-walled 3 mm shells split into two lumped halves in steady energy balance (absorbed sun = '
               'convection + radiation + conduction between the halves), no shading between parts or by the body; '
               'the roof a single steel plate insulated below; the body\'s structure at air temperature; small '
               'displacements composed along body → roof → part → device. Vibration: ISO 8608 road over its '
               '0.011-2.83 cycles/m band, a linear half-car (no roll), each mount\'s first mode uncoupled on the '
               'body\'s motion. Typical materials, finishes, damping and bracket sections, not this rig\'s.')


# =========================================================================== settings
def _num(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


@dataclass
class ThermoVibSettings:
    # thermal
    date: str = ''                  # ISO date; '' = the day with the highest noon sun in the exposure campaign
    heading_deg: float | None = None    # vehicle heading clockwise from north; None = the worst of 12 headings
    t_cal_c: float = 15.0           # the garage the cameras are calibrated in
    t_max_c: float = 45.0           # the day's peak air temperature
    daily_range_c: float = 12.0     # peak minus the night's low
    peak_hour: float = 15.0         # local time of the peak
    air_kmh: float = 30.0           # airflow over the rig (driving speed; 0 = parked)
    material: str = 'al6061'        # rig parts
    finish: str = 'black_anodised'
    roof_paint: str = 'silver'
    albedo: float = 0.2
    feet: str = 'sliding'           # the rack's feet on the roof: sliding | clamped
    # tolerance
    tol_source: str = 'install'     # install (1° / 10 mm) | observability (its requirement) | custom
    tol_deg: float = 1.0
    tol_mm: float = 10.0
    px_limit: float = 1.0           # a pair's relative drift in the camera's image
    range_m: float = 10.0           # the parallax of a pair's relative move is taken at this range
    # vibration
    road_class: str = 'C'
    speed_kmh: float = 50.0
    zeta_body: float = 0.25
    zeta_mount: float = 0.02
    zeta_printed: float = 0.05
    # events
    bump_h_mm: float = 80.0
    bump_len_m: float = 0.6
    bump_kmh: float = 20.0
    joint_mm: float = 20.0
    cobble_class: str = 'D'
    cobble_kmh: float = 30.0
    sett_m: float = 0.12

    def to_dict(self):
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    @staticmethod
    def from_dict(d) -> 'ThermoVibSettings':
        """Tolerant: unknown keys dropped, numbers clamped to LIMITS, bad values fall back to the defaults."""
        s = ThermoVibSettings()
        d = d if isinstance(d, dict) else {}
        for k, (lo, hi) in LIMITS.items():
            v = _num(d.get(k))
            if v is not None:
                setattr(s, k, min(max(v, lo), hi))
        for k, opts in CHOICES.items():
            v = d.get(k)
            if isinstance(v, str) and v.strip() in opts:
                setattr(s, k, v.strip())
            elif isinstance(v, str) and v.strip().upper() in opts:
                setattr(s, k, v.strip().upper())
        v = d.get('date')
        if isinstance(v, str):
            import datetime as _dt
            try:
                s.date = _dt.date.fromisoformat(v.strip()[:10]).isoformat() if v.strip() else ''
            except ValueError:
                s.date = ''
        h = d.get('heading_deg')
        h = _num(h) if h is not None else None
        s.heading_deg = None if h is None else round(h % 360.0, 3)
        return s


def settings_of(project) -> ThermoVibSettings:
    return ThermoVibSettings.from_dict(getattr(project.analysis, 'thermo_vib', None))


def tolerance(project, st: ThermoVibSettings) -> dict:
    """{'deg', 'mm', 'px', 'source', 'label'}: the calibration tolerance the drifts are checked against."""
    if st.tol_source == 'observability':
        from .observability import ObservabilitySettings
        ob = ObservabilitySettings.from_dict(getattr(project.analysis, 'observability', None))
        deg, mm = float(ob.need_deg), float(ob.need_mm)
    elif st.tol_source == 'custom':
        deg, mm = st.tol_deg, st.tol_mm
    else:
        from .install import InstallSettings
        ins = InstallSettings()
        deg, mm = float(ins.ang_tol_deg), float(ins.pos_tol_mm)
    return {'deg': deg, 'mm': mm, 'px': st.px_limit, 'source': st.tol_source,
            'label': f'{deg:g}° / {mm:g} mm ({TOL_LABEL[st.tol_source]}), {st.px_limit:g} px for a pair'}


# =========================================================================== closed forms
def cantilever_f1(E, I, L, m_tip, m_beam=0.0) -> float:
    """First bending mode (Hz) of a cantilever of modulus E (Pa), second moment I (m⁴) and length L (m) carrying a
    tip mass m_tip (kg), its own mass m_beam by Rayleigh's 0.24 (33/140): (1/2π)·√(3EI / ((m_tip + 0.24·m_beam)·L³))."""
    m = m_tip + 0.24 * m_beam
    return math.sqrt(3.0 * E * I / (max(m, 1e-9) * L ** 3)) / (2 * math.pi)


def tube_I(r, t=WALL_M) -> float:
    """Second moment (m⁴) of a round tube of outer radius r and wall t."""
    ri = max(r - t, 0.0)
    return math.pi / 4.0 * (r ** 4 - ri ** 4)


def tube_area(r, t=WALL_M) -> float:
    ri = max(r - t, 0.0)
    return math.pi * (r * r - ri * ri)


def thermal_curvature(alpha, dT_gradient, depth) -> float:
    """κ = α·ΔT / depth (1/m) of a beam whose two faces depth apart differ by ΔT (linear profile)."""
    return alpha * dT_gradient / max(depth, 1e-9)


def cantilever_bow(kappa, a) -> tuple[float, float]:
    """(deflection m, slope rad) at a from the root of a cantilever bent at uniform curvature κ: κa²/2, κa."""
    return kappa * a * a / 2.0, kappa * a


def ss_bow(kappa, span, eta) -> tuple[float, float]:
    """(deflection m, slope rad) at η from the middle of a simply supported span bent at uniform curvature κ (convex
    upwards for κ > 0): w = κ/2·(span²/4 − η²), dw/dη = −κη."""
    return kappa / 2.0 * (span * span / 4.0 - eta * eta), -kappa * eta


def iso8608_gd(cls: str) -> float:
    """Gd(n0) (m³) of an ISO 8608 class: the class's geometric mean 16·4^k × 10⁻⁶ m³ (A = 16, B = 64, … H = 262144)."""
    k = ISO_CLASSES.index(str(cls).upper())
    return 16e-6 * 4.0 ** k


def road_psd_spatial(n, cls: str, band=ISO_BAND) -> np.ndarray:
    """Gd(n) = Gd(n0)·(n/n0)^-2 (m³) inside the band (cycles/m), 0 outside."""
    n = np.asarray(n, float)
    g = iso8608_gd(cls) * (np.maximum(n, 1e-12) / ISO_N0) ** -2.0
    return np.where((n >= band[0]) & (n <= band[1]), g, 0.0)


def road_psd_temporal(f, cls: str, speed_mps: float, band=ISO_BAND) -> np.ndarray:
    """G(f) = Gd(f/V)/V (m²/Hz), the road's height PSD under a wheel at speed V."""
    v = max(float(speed_mps), 1e-6)
    return road_psd_spatial(np.asarray(f, float) / v, cls, band) / v


def integrate(f, psd) -> float:
    """∫ psd df by the trapezoid rule over the (sorted) grid f."""
    f = np.asarray(f, float)
    p = np.asarray(psd, float)
    return float(np.sum(0.5 * (p[1:] + p[:-1]) * np.diff(f)))


def road_rms(cls: str, n1=ISO_BAND[0], n2=ISO_BAND[1], points=4000) -> float:
    """RMS road height (m) of a class over [n1, n2] cycles/m, integrated numerically on a log grid."""
    n = np.geomspace(n1, n2, points)
    return math.sqrt(integrate(n, road_psd_spatial(n, cls, (n1, n2))))


def transmissibility(r, zeta) -> np.ndarray:
    """Textbook base-excitation transmissibility of a damped single degree of freedom at frequency ratio r:
    √((1 + (2ζr)²) / ((1 − r²)² + (2ζr)²))."""
    r = np.asarray(r, float)
    return np.sqrt((1 + (2 * zeta * r) ** 2) / ((1 - r * r) ** 2 + (2 * zeta * r) ** 2))


def sdof_relative(f, fn, zeta) -> np.ndarray:
    """Relative deflection (m) of a single mode per unit base acceleration (m/s²): −1/(ωn² − ω² + 2iζωnω)."""
    w = 2 * math.pi * np.asarray(f, float)
    wn = 2 * math.pi * fn
    return -1.0 / (wn * wn - w * w + 2j * zeta * wn * w)


def quarter_car(f, ms, mu, ks, cs, kt) -> np.ndarray:
    """Sprung-mass displacement per unit road displacement of a quarter-car (sprung ms, unsprung mu, suspension ks,
    cs, tyre kt; SI units), by solving its two equations of motion at each frequency."""
    w = 2 * math.pi * np.asarray(f, float)
    out = np.empty(len(w), complex)
    for i, om in enumerate(w):
        A = np.array([[-ms * om * om + 1j * om * cs + ks, -(1j * om * cs + ks)],
                      [-(1j * om * cs + ks), -mu * om * om + 1j * om * cs + ks + kt]])
        out[i] = np.linalg.solve(A, np.array([0.0, kt]))[0]
    return out


def resonance(f1, peaks=()) -> list:
    """Why a first mode f1 (Hz) resonates: 'body' (in 1-2 Hz), 'wheel hop' (10-15 Hz) and the names of excitation
    peaks [(name, Hz)] it lies within NEAR_PEAK of."""
    out = []
    if BODY_BAND[0] <= f1 <= BODY_BAND[1]:
        out.append('body')
    if HOP_BAND[0] <= f1 <= HOP_BAND[1]:
        out.append('wheel hop')
    for name, fp in peaks:
        if fp and fp > 0 and abs(f1 - fp) <= NEAR_PEAK * fp and name not in out:
            out.append(name)
    return out


# =========================================================================== sun, air and heat transfer
def air_mass(el_deg) -> np.ndarray:
    """Relative optical air mass (Kasten & Young 1989) at a solar elevation (degrees); inf below the horizon."""
    el = np.asarray(el_deg, float)
    z = 90.0 - el
    with np.errstate(invalid='ignore', divide='ignore'):
        am = 1.0 / (np.cos(np.radians(z)) + 0.50572 * np.maximum(96.07995 - z, 1e-6) ** -1.6364)
    return np.where(el > 0, am, np.inf)


def clear_sky(el_deg, dist_au=1.0) -> tuple[float, float, float]:
    """(DNI, DHI, GHI) W/m² of the Meinel clear sky at a solar elevation (degrees) and the sun's distance (AU)."""
    if not el_deg > 0:
        return 0.0, 0.0, 0.0
    am = float(air_mass(el_deg))
    dni = SOLAR_CONSTANT / dist_au ** 2 * 0.7 ** (am ** 0.678)
    dhi = 0.1 * dni
    return dni, dhi, dni * math.sin(math.radians(el_deg)) + dhi


def air_props(T_k) -> tuple[float, float, float]:
    """(kinematic viscosity m²/s, conductivity W/mK, Prandtl) of air at T (K): Incropera Table A.4, linear between 300
    and 350 K."""
    t = min(max((T_k - 300.0) / 50.0, -0.6), 1.6)
    return 15.89e-6 + t * (20.92e-6 - 15.89e-6), 26.3e-3 + t * (30.0e-3 - 26.3e-3), 0.707 - t * 0.007


def h_flat_plate(v, L, T_film) -> float:
    """Average forced-convection coefficient (W/m²K) of a flat plate of length L along a flow v (m/s): laminar
    0.664·Re^½·Pr^⅓, mixed (0.037·Re^0.8 − 871)·Pr^⅓ beyond Re 5·10⁵ (Incropera eq. 7.30 and 7.38)."""
    if v <= 0 or L <= 0:
        return 0.0
    nu, k, pr = air_props(T_film)
    re = v * L / nu
    nu_l = 0.664 * re ** 0.5 * pr ** (1 / 3) if re <= 5e5 else (0.037 * re ** 0.8 - 871.0) * pr ** (1 / 3)
    return nu_l * k / L


def h_cylinder(v, D, T_film) -> float:
    """Average coefficient (W/m²K) of a cylinder of diameter D in crossflow v: Churchill-Bernstein (Incropera eq.
    7.54)."""
    if v <= 0 or D <= 0:
        return 0.0
    nu, k, pr = air_props(T_film)
    re = v * D / nu
    nud = 0.3 + 0.62 * re ** 0.5 * pr ** (1 / 3) / (1 + (0.4 / pr) ** (2 / 3)) ** 0.25 * \
        (1 + (re / 282000.0) ** 0.625) ** 0.8
    return nud * k / D


def h_natural(dT, L) -> float:
    """Simplified laminar natural convection in air, 1.32·(ΔT/L)^¼ W/m²K (Holman, Heat Transfer, Table 7-2), at
    least 1."""
    return max(1.32 * (abs(dT) / max(L, 1e-3)) ** 0.25, 1.0)


def sky_temperature(T_air_k) -> float:
    """Clear-sky radiant temperature (K), Swinbank (1963): 0.0552·T^1.5."""
    return 0.0552 * T_air_k ** 1.5


def ambient(hour, st: ThermoVibSettings) -> float:
    """Air temperature (°C) at a local hour: the peak at peak_hour, the daily range lower 12 h away (cosine)."""
    return st.t_max_c - st.daily_range_c / 2.0 * (1.0 - math.cos(2 * math.pi * (hour - st.peak_hour) / 24.0))


# =========================================================================== the rig as a chain of parts
def _rot(yaw):
    c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _unit(v, fallback=(1.0, 0.0, 0.0)):
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else np.asarray(fallback, float)


def _half(p):
    """Half extents (local x, y, z) of a part's solid."""
    if p.kind == 'box':
        return np.array([p.length, p.width, p.height]) / 2
    if p.kind == 'cylinder':
        return np.array([p.radius, p.radius, p.height / 2])
    return np.array([p.radius, p.length / 2, p.radius])


def _top(p) -> float:
    return p.z + _half(p)[2]


def _bottom(p) -> float:
    return p.z - _half(p)[2]


def part_distance(p, P) -> tuple[float, np.ndarray]:
    """(distance m from P to the part's solid, the nearest point of the solid), vehicle frame."""
    R = _rot(p.yaw)
    c = np.array([p.x, p.y, p.z])
    u = R.T @ (np.asarray(P, float) - c)
    h = _half(p)
    if p.kind == 'box':
        q = np.clip(u, -h, h)
    elif p.kind == 'cylinder':
        rho = math.hypot(u[0], u[1])
        s = min(1.0, p.radius / rho) if rho > 1e-12 else 1.0
        q = np.array([u[0] * s, u[1] * s, min(max(u[2], -h[2]), h[2])])
    else:
        rho = math.hypot(u[0], u[2])
        s = min(1.0, p.radius / rho) if rho > 1e-12 else 1.0
        q = np.array([u[0] * s, min(max(u[1], -h[1]), h[1]), u[2] * s])
    return float(np.linalg.norm(u - q)), c + R @ q


def _xy_box(p):
    from .rig import footprint
    fp = footprint(p)
    return fp.min(axis=0), fp.max(axis=0)


def roof_box(project) -> dict:
    """The roof (core.rig.roof): {'x_front', 'x_rear', 'half_width', 'top', 'centre', 'length'}."""
    from .rig import roof
    xf, xr, hw, top = roof(project.vehicle.mesh)
    return {'x_front': xf, 'x_rear': xr, 'half_width': hw, 'top': top, 'centre': np.array([(xf + xr) / 2, 0.0, top]),
            'length': max(xf - xr, 0.3)}


def _on_roof(P, rf, margin=0.2) -> bool:
    return (rf['x_rear'] - margin <= P[0] <= rf['x_front'] + margin and abs(P[1]) <= rf['half_width'] + margin
            and P[2] >= rf['top'] - 0.10)


def rig_tree(project, rf=None) -> list[dict]:
    """The visible rig parts with what carries each: [{'part', 'uid', 'name', 'kind', 'supports': [index] or
    ['roof'] or ['body'], 'base': base point (the bottom centre; a bar's centre)}]."""
    rf = rf or roof_box(project)
    parts = [p for p in getattr(project, 'rig', []) or [] if p.visible]
    out = []
    for i, p in enumerate(parts):
        lo, hi = _xy_box(p)
        sup = []
        for j, q in enumerate(parts):
            if j == i or q.z >= p.z:
                continue
            if abs(_bottom(p) - _top(q)) > SUPPORT_GAP_M:
                continue
            ql, qh = _xy_box(q)
            if np.all(lo <= qh + 0.01) and np.all(ql <= hi + 0.01):
                sup.append(j)
        if not sup:
            c = np.array([p.x, p.y, _bottom(p)])
            sup = ['roof'] if _on_roof(c, rf) else ['body']
        base = np.array([p.x, p.y, p.z]) if p.kind == 'bar' else np.array([p.x, p.y, _bottom(p)])
        out.append({'part': p, 'uid': p.uid, 'name': p.name, 'kind': p.kind, 'supports': sup, 'base': base})
    return out


def carrier(tree, P, rf) -> tuple[object, np.ndarray]:
    """What a device at P rides on: (index into the tree, or 'roof' / 'body'; the contact point)."""
    best, bq, bd = None, None, CARRY_M
    for i, t in enumerate(tree):
        d, q = part_distance(t['part'], P)
        if d <= bd + 1e-12:
            best, bq, bd = i, q, d
    if best is not None:
        return best, bq
    P = np.asarray(P, float)
    if _on_roof(P, rf):
        return 'roof', np.array([P[0], P[1], rf['top']])
    return 'body', P.copy()


def chain_of(tree, node) -> list:
    """The parts from the roof (or body) up to node (an index), first supports only."""
    out = []
    while isinstance(node, (int, np.integer)):
        out.append(int(node))
        node = tree[node]['supports'][0]
    return out[::-1]


# =========================================================================== thermal model of a part
@dataclass
class ThermalPart:
    """A part as facets in two lumped halves: normals (P, 3) and centres (P, 3) in the vehicle frame, areas (P,),
    node (P,) 0 = the half facing g (top, or the sun's side of a mast) and 1; conductance between the halves (W/K)."""
    name: str
    normals: np.ndarray
    centres: np.ndarray
    areas: np.ndarray
    node: np.ndarray
    G: float
    char: tuple             # ('plate', L along the flow) or ('cyl', D)
    alpha_s: float
    eps: float
    rho_c_t: float          # heat capacity per area (J/m²K)
    below_roof: bool        # the downward facets see the roof (else the ground)
    g: np.ndarray           # gradient axis

    def depth(self) -> float:
        """Distance between the two halves' area centroids along g (m)."""
        if not (self.node == 1).any():
            return 0.0
        c = self.centres @ self.g
        a = self.areas
        m0, m1 = self.node == 0, self.node == 1
        return float(np.sum(a[m0] * c[m0]) / a[m0].sum() - np.sum(a[m1] * c[m1]) / a[m1].sum())


def _ring(n=24):
    t = (np.arange(n) + 0.5) * 2 * math.pi / n
    return np.cos(t), np.sin(t)


def thermal_part(p, mat, fin, sun_h=None, near_roof=False) -> ThermalPart:
    """Facets and halves of a rig part. sun_h: the sun's horizontal direction (vehicle frame), the gradient axis of an
    upright mast; bars and boxes split top and bottom."""
    R = _rot(p.yaw)
    c = np.array([p.x, p.y, p.z])
    k, t = mat['k'], WALL_M
    rct = mat['rho'] * mat['c'] * t
    if p.kind == 'box':
        L, W, H = max(p.length, 1e-3), max(p.width, 1e-3), max(p.height, 1e-3)
        N, C, A, nd = [], [], [], []
        N += [[0, 0, 1], [0, 0, -1]]
        C += [[0, 0, H / 2], [0, 0, -H / 2]]
        A += [L * W, L * W]
        nd += [0, 1]
        for n, off, side in (([1, 0, 0], L / 2, W), ([-1, 0, 0], L / 2, W), ([0, 1, 0], W / 2, L),
                             ([0, -1, 0], W / 2, L)):
            for zz, node in ((H / 4, 0), (-H / 4, 1)):
                N.append(n)
                C.append([n[0] * off, n[1] * off, zz])
                A.append(side * H / 2)
                nd.append(node)
        N = np.array(N, float) @ R.T
        C = np.array(C, float) @ R.T + c
        G = k * t * 2 * (L + W) / H
        flow = abs(L * math.cos(math.radians(p.yaw))) + abs(W * math.sin(math.radians(p.yaw)))
        return ThermalPart(p.name, N, C, np.array(A), np.array(nd), G, ('plate', flow), fin['a'], fin['e'], rct,
                           near_roof, np.array([0.0, 0.0, 1.0]))
    r = max(p.radius, 1e-3)
    ca, sa = _ring()
    if p.kind == 'cylinder':
        Lax = max(p.height, 1e-3)
        g = _unit(sun_h if sun_h is not None else [1.0, 0.0, 0.0])
        N = np.c_[ca, sa, np.zeros_like(ca)]
        C = N * r + c
        A = np.full(len(ca), 2 * math.pi * r * Lax / len(ca))
        nd = np.where(N @ g > 0, 0, 1)
        top = math.pi * r * r / 2                     # the cap, half to each half
        N = np.vstack([N, [[0, 0, 1], [0, 0, 1]]])
        C = np.vstack([C, [c + [0, 0, Lax / 2]] * 2])
        A = np.r_[A, top, top]
        nd = np.r_[nd, 0, 1]
    else:
        Lax = max(p.length, 1e-3)
        g = np.array([0.0, 0.0, 1.0])
        loc = np.c_[ca, np.zeros_like(ca), sa]           # around the bar's local y axis
        N = loc @ R.T
        C = N * r + c
        A = np.full(len(ca), 2 * math.pi * r * Lax / len(ca))
        nd = np.where(N[:, 2] > 0, 0, 1)
    G = math.pi * k * t * Lax / (2 * r)                # a cosine profile round the wall (see the docstring)
    return ThermalPart(p.name, np.asarray(N, float), np.asarray(C, float), np.asarray(A, float), np.asarray(nd),
                       G, ('cyl', 2 * r), fin['a'], fin['e'], rct, near_roof, g)


def roof_part(rf, paint) -> ThermalPart:
    """The roof skin as one upward steel plate (insulated below)."""
    A = rf['length'] * 2 * rf['half_width']
    return ThermalPart('Roof', np.array([[0.0, 0.0, 1.0]]), np.array([rf['centre']]), np.array([A]),
                       np.array([0]), 0.0, ('plate', rf['length']), paint['a'], paint['e'],
                       MATERIALS['steel']['rho'] * MATERIALS['steel']['c'] * ROOF_SKIN_M, False,
                       np.array([0.0, 0.0, 1.0]))


def absorbed(tp: ThermalPart, sun, dni, dhi, ghi, albedo) -> np.ndarray:
    """Absorbed sun per facet (W): direct on facets facing the sun, isotropic sky diffuse and ground-reflected light
    by each facet's view of the sky (1 + n_z)/2 and of the ground (1 − n_z)/2."""
    nz = tp.normals[:, 2]
    direct = dni * np.maximum(tp.normals @ sun, 0.0) if dni > 0 else np.zeros(len(nz))
    return tp.alpha_s * tp.areas * (direct + dhi * (1 + nz) / 2 + albedo * ghi * (1 - nz) / 2)


def solve_part(tp: ThermalPart, q_abs, T_air_c, v_mps, T_below_c=None, iters=60) -> dict:
    """Steady temperatures (°C) of the part's halves: {'T': [T0, T1], 'mean', 'dT' (T0 − T1), 'h', 'tau_s'}."""
    Ta = T_air_c + 273.15
    Tsky = sky_temperature(Ta)
    Tb = (T_below_c if T_below_c is not None else T_air_c) + 273.15
    nn = 2 if (tp.node == 1).any() else 1
    masks = [tp.node == i for i in range(nn)]
    Ai = np.array([tp.areas[m].sum() for m in masks])
    Qi = np.array([q_abs[m].sum() for m in masks])
    Fs = (1 + tp.normals[:, 2]) / 2
    kind, Lc = tp.char
    T = np.full(nn, Ta + 10.0)

    def coeff(Tm):
        film = (Tm + Ta) / 2
        hf = h_cylinder(v_mps, Lc, film) if kind == 'cyl' else h_flat_plate(v_mps, Lc, film)
        hn = h_natural(Tm - Ta, Lc)
        return (hf ** 3 + hn ** 3) ** (1 / 3)

    def resid(T):
        h = coeff(float(np.mean(T)))
        out = np.empty(nn)
        for i, m in enumerate(masks):
            rad = tp.eps * SIGMA * np.sum(tp.areas[m] * (Fs[m] * (T[i] ** 4 - Tsky ** 4) +
                                                          (1 - Fs[m]) * (T[i] ** 4 - Tb ** 4)))
            cond = tp.G * (T[i] - T[1 - i]) if nn == 2 else 0.0
            out[i] = Qi[i] - h * Ai[i] * (T[i] - Ta) - rad - cond
        return out, h
    h = coeff(Ta)
    for _ in range(iters):
        r, h = resid(T)
        J = np.empty((nn, nn))
        for j in range(nn):
            d = np.zeros(nn)
            d[j] = 1e-3
            J[:, j] = (resid(T + d)[0] - r) / 1e-3
        step = np.linalg.solve(J, -r)
        step = np.clip(step, -50.0, 50.0)
        T = T + step
        if np.max(np.abs(step)) < 1e-7:
            break
    Tc = T - 273.15
    lin = h + 4 * tp.eps * SIGMA * float(np.mean(T)) ** 3
    tau = tp.rho_c_t / max(lin, 1e-6)                  # per unit area: ρ c t / (h + 4εσT³)
    mean = float(np.sum(Ai * Tc) / Ai.sum())
    return {'T': [float(x) for x in Tc], 'mean': mean, 'dT': float(Tc[0] - Tc[1]) if nn == 2 else 0.0,
            'h': float(h), 'tau_s': float(tau)}


# =========================================================================== deformation fields
def _part_field(t, mat, temp, cal_c, feet_amp, Q, sup_alpha_dT=None, span=None, kappa=0.0, g=None):
    """Motion (u m, ω rad, both vehicle frame) at the clamped point Qc of one part relative to its base, for a part
    at temperature temp (solve_part) against the calibration temperature, and Qc."""
    p = t['part']
    alpha = mat['alpha']
    dTm = temp['mean'] - cal_c
    _, Qc = part_distance(p, Q)
    B = t['base']
    R = _rot(p.yaw)
    if p.kind == 'cylinder':
        a = min(max(Qc[2] - B[2], 0.0), p.height)
        u = alpha * dTm * (Qc - B)
        gh = _unit([g[0], g[1], 0.0]) if g is not None else np.array([1.0, 0.0, 0.0])
        d, s = cantilever_bow(kappa, a)
        u = u - d * gh
        w = s * np.cross([0.0, 0.0, 1.0], -gh)
        return u, w, Qc
    if p.kind == 'bar':
        yb = R[:, 1]
        eta = float((Qc - B) @ yb)
        u = alpha * dTm * (Qc - B)
        if sup_alpha_dT is not None and t.get('clamped'):
            u = u - (alpha * dTm - sup_alpha_dT) * eta * yb    # the feet hold the bar's length to the support's
        w, s = ss_bow(kappa * feet_amp, span, eta)
        u = u + w * np.array([0.0, 0.0, 1.0])
        om = s * np.cross(yb, [0.0, 0.0, 1.0])
        return u, om, Qc
    # box: uniform expansion about its base, a dome on its corners
    hx, hy = max(p.length, 1e-3) / 2, max(p.width, 1e-3) / 2
    loc = R.T @ (Qc - B)
    dx, dy = min(max(loc[0], -hx), hx), min(max(loc[1], -hy), hy)
    u = alpha * dTm * (Qc - B)
    wz = kappa / 2 * (hx * hx - dx * dx + hy * hy - dy * dy) * (1.0 if loc[2] > 1e-6 else 0.0)
    om = R @ np.array([-kappa * dy, kappa * dx, 0.0]) * (1.0 if loc[2] > 1e-6 else 0.0)
    return u + wz * np.array([0.0, 0.0, 1.0]), om, Qc


class Warp:
    """One thermal state of the rig (an hour at a heading): every part's temperature and curvature, and the motion of
    any point riding on the body, the roof or a part relative to the calibration."""

    def __init__(self, tree, rf, st: ThermoVibSettings, T_air, T_roof, temps: dict, cal_c):
        self.tree, self.rf, self.st = tree, rf, st
        self.T_air, self.T_roof, self.temps, self.cal = T_air, T_roof, temps, cal_c
        self.mat = MATERIALS[st.material]
        self.steel = MATERIALS['steel']
        self.info = {}
        for i, t in enumerate(tree):
            tp = temps[t['uid']]
            kappa = thermal_curvature(self.mat['alpha'], tp['dT'], tp['depth']) if tp['depth'] > 0 else 0.0
            d = {'kappa': kappa, 'amp': 1.0, 'span': None, 'sup_alpha_dT': None, 'N': 0.0, 'N_cr': None,
                 'mismatch_mm': 0.0, 'stress_mpa': 0.0}
            p = t['part']
            if p.kind == 'bar':
                span = max(p.length - 2 * FEET_INSET_M, 0.6 * p.length)
                d['span'] = span
                sup = t['supports'][0]
                if sup == 'roof':
                    sa = self.steel['alpha'] * (T_roof - cal_c)
                elif sup == 'body':
                    sa = self.steel['alpha'] * (T_air - cal_c)
                else:
                    sa = self.mat['alpha'] * (temps[tree[sup]['uid']]['mean'] - cal_c)
                d['sup_alpha_dT'] = sa
                strain = self.mat['alpha'] * (tp['mean'] - cal_c) - sa
                d['mismatch_mm'] = strain * span * 1000.0
                I, A = tube_I(p.radius), tube_area(p.radius)
                d['N_cr'] = math.pi ** 2 * self.mat['E'] * I / span ** 2
                if st.feet == 'clamped':
                    t['clamped'] = True
                    N = self.mat['E'] * A * strain                 # compressive positive
                    d['N'], d['stress_mpa'] = N, -N / A / 1e6
                    ratio = N / d['N_cr']
                    d['amp'] = 1.0 / (1.0 - ratio) if ratio < 0.95 else 20.0
                else:
                    t['clamped'] = False
            self.info[t['uid']] = d

    def body(self, Q):
        return self.steel['alpha'] * (self.T_air - self.cal) * np.asarray(Q, float), np.zeros(3)

    def roof(self, Q):
        u, w = self.body(Q)
        d = np.asarray(Q, float) - self.rf['centre']
        return u + self.steel['alpha'] * (self.T_roof - self.T_air) * np.array([d[0], d[1], 0.0]), w

    def motion(self, node, Q) -> tuple[np.ndarray, np.ndarray]:
        """(U m, Ω rad) of a point Q riding on node ('body', 'roof' or a tree index), vehicle frame."""
        Q = np.asarray(Q, float)
        if node == 'body':
            return self.body(Q)
        if node == 'roof':
            return self.roof(Q)
        t = self.tree[node]
        B = t['base']
        Us, Ws = [], []
        for s in t['supports']:
            u, w = self.motion(s, B)
            Us.append(u)
            Ws.append(w)
        Ub, Wb = np.mean(Us, axis=0), np.mean(Ws, axis=0)
        d = self.info[t['uid']]
        tp = self.temps[t['uid']]
        u, w, Qc = _part_field(t, self.mat, tp, self.cal, d['amp'], Q, d['sup_alpha_dT'], d['span'], d['kappa'],
                               tp.get('g'))
        U = Ub + np.cross(Wb, Qc - B) + u
        W = Wb + w
        return U + np.cross(W, Q - Qc), W


def small_rotation(w) -> np.ndarray:
    """The rotation matrix of a rotation vector w (rad), Rodrigues."""
    th = float(np.linalg.norm(w))
    if th < 1e-15:
        return np.eye(3)
    k = np.asarray(w, float) / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def angle_change(R, w) -> tuple[float, float, float]:
    """(Δyaw, Δpitch, Δroll) degrees of a device of rotation R (device → vehicle) turned by the rotation vector w."""
    a = euler_zyx(R)
    b = euler_zyx(small_rotation(w) @ R)
    return wrap180(b[0] - a[0]), b[1] - a[1], wrap180(b[2] - a[2])


# =========================================================================== lens helpers
def px_per_rad(intr) -> float:
    """The lens's local scale at the image centre, px per radian (mean of the two image axes)."""
    d = 1e-3
    u, v, _, _ = intr.project_dirs(np.array([[math.sin(d), 0.0, math.cos(d)], [0.0, math.sin(d), math.cos(d)]]))
    return float((abs(u[0] - intr.cx) + abs(v[1] - intr.cy)) / 2 / d)


_GRIDS: dict = {}


def _grid_dirs(intr, n=(25, 19)):
    key = (intr.width, intr.height, intr.f_mm, intr.projection, tuple(intr.kb), intr.cx, intr.cy, intr.max_theta,
           intr.aspect, n)
    hit = _GRIDS.get(key)
    if hit is None:
        U, V = np.meshgrid(np.linspace(0.5, intr.width - 0.5, n[0]), np.linspace(0.5, intr.height - 0.5, n[1]))
        u = np.r_[intr.cx, U.ravel()]
        v = np.r_[intr.cy, V.ravel()]
        d, th = intr.unproject(u, v)
        keep = th <= intr.max_theta + 1e-9
        keep[0] = True
        hit = (d[keep], u[keep], v[keep])
        if len(_GRIDS) > 32:
            _GRIDS.pop(next(iter(_GRIDS)))
        _GRIDS[key] = hit
    return hit


def px_shift(intr, W2O, w) -> tuple[float, float]:
    """(at the centre, worst over the image) image shift in px of the camera's view turned by the rotation vector w
    (vehicle frame, rad), through the exact lens model."""
    if float(np.linalg.norm(w)) < 1e-15:
        return 0.0, 0.0
    d, u, v = _grid_dirs(intr)
    Ro = small_rotation(W2O @ np.asarray(w, float))      # the rotation in the optical frame
    u2, v2, _, _ = intr.project_dirs(d @ Ro.T)
    s = np.hypot(u2 - u, v2 - v)
    return float(s[0]), float(np.max(s))


# =========================================================================== devices
def devices(project, variant=None) -> list[dict]:
    """Every visible camera and sensor of the variant: {'uid', 'tag', 'name', 'kind', 'slot', 'P', 'R', 'mass',
    'intr' (cameras), 'cam' (the camera or sensor object)}."""
    from .sensors import rotation as sensor_rotation
    v = variant or project.variant
    comps = {c['uid']: c for c in SU.rig_components(project, SU.settings_for(project), v)}
    out = []
    for c in v.cameras:
        if not c.visible or not all(math.isfinite(x) for x in (c.x, c.y, c.z, c.yaw, c.pitch, c.roll)):
            continue
        from .lens import resolve
        try:
            intr = resolve(project.module_for(c), project.lens_for(c))
        except Exception:                               # a broken lens: no pixel figures for it
            intr = None
        m = comps.get(c.uid, {}).get('mass', SU.CAMERA_FALLBACK_KG)
        out.append({'uid': c.uid, 'tag': c.tag, 'name': c.name, 'kind': 'camera', 'slot': c.slot, 'P': c.pos(),
                    'R': rot_zyx(c.yaw, c.pitch, c.roll), 'mass': float(m), 'intr': intr, 'obj': c})
    for s in v.sensors:
        if not s.visible or not all(math.isfinite(x) for x in (s.x, s.y, s.z, s.yaw, s.pitch, s.roll)):
            continue
        m = comps.get(s.uid, {}).get('mass', 1.0)
        out.append({'uid': s.uid, 'tag': s.tag, 'name': s.name, 'kind': s.kind, 'slot': None, 'P': s.pos(),
                    'R': sensor_rotation(s), 'mass': float(m), 'intr': None, 'obj': s})
    return out


def pairs_of(project, devs, variant=None) -> list[dict]:
    """Pairs that share a calibration: every camera with every lidar, and the stereo pairs (core.timing)."""
    from .timing import stereo_pairs
    by = {d['uid']: d for d in devs}
    out = []
    for a in devs:
        if a['kind'] != 'camera':
            continue
        for b in devs:
            if b['kind'] == 'lidar':
                out.append({'a': a['uid'], 'b': b['uid'], 'kind': 'camera-lidar', 'label': f'{a["tag"]}–{b["tag"]}'})
    for ca, cb, _, base in stereo_pairs(project, variant):
        if ca.uid in by and cb.uid in by:
            out.append({'a': ca.uid, 'b': cb.uid, 'kind': 'stereo', 'label': f'{ca.tag}–{cb.tag}',
                        'baseline_m': float(base)})
    return out


# =========================================================================== the sun over the day
def hot_day(ex) -> 'object':
    """The date in the exposure campaign with the highest noon sun (core.exposure's NOAA position)."""
    import datetime as _dt
    from .exposure import _jd0, sun_position
    d0, d1 = ex.dates()
    n = (d1 - d0).days + 1
    days = [d0 + _dt.timedelta(days=i) for i in range(n)]
    jd = np.array([_jd0(d) for d in days]) + (12.0 - ex.lon / 15.0) / 24.0          # local solar noon, UTC
    _, el = sun_position(jd, ex.lat, ex.lon)
    return days[int(np.argmax(el))]


def day_hours(project, st: ThermoVibSettings) -> dict:
    """The hour samples of the day: {'date', 'local_h', 'az', 'el', 'dist_au', 'lat', 'lon', 'note'} from
    core.exposure's settings (location, campaign dates, driving window), one per hour (sample centres)."""
    import datetime as _dt
    from .exposure import ExposureSettings, campaign, solar_ephemeris, sun_position
    ex = ExposureSettings.from_dict(getattr(project.analysis, 'exposure', None))
    day = _dt.date.fromisoformat(st.date) if st.date else hot_day(ex)
    d = ex.to_dict()
    d.update(start=day.isoformat(), end=day.isoformat(), minutes=60.0, days=1.0)
    one = ExposureSettings.from_dict(d)
    cp = campaign(one)
    az, el = sun_position(cp.jd, one.lat, one.lon)
    dist = solar_ephemeris(cp.jd)['distance_au']
    return {'date': day.isoformat(), 'local_h': np.asarray(cp.local_h, float), 'az': np.asarray(az, float),
            'el': np.asarray(el, float), 'dist_au': np.asarray(dist, float), 'lat': one.lat, 'lon': one.lon,
            'window': (one.hour_from, one.hour_to), 'note': cp.note, 'auto_date': not st.date}


# =========================================================================== the thermal study
def _sun_vehicle(az, el, heading):
    from .exposure import vehicle_dirs
    return vehicle_dirs(np.array([az]), np.array([el]), np.array([heading]))[0, 0]


def thermal_state(project, tree, rf, st: ThermoVibSettings, hour, az, el, dist, heading) -> tuple[Warp, dict]:
    """The rig's thermal state at one hour and heading: (Warp, {'t_air', 't_roof', 'sun', 'dni', 'dhi', 'ghi',
    'parts': {uid: temps}})."""
    T_air = ambient(hour, st)
    dni, dhi, ghi = clear_sky(el, dist)
    sun = _sun_vehicle(az, el, heading) if el > 0 else np.array([0.0, 0.0, -1.0])
    v = st.air_kmh / 3.6
    paint = ROOF_PAINTS[st.roof_paint]
    rp = roof_part(rf, paint)
    roof_t = solve_part(rp, absorbed(rp, sun, dni, dhi, ghi, st.albedo), T_air, v)
    T_roof = roof_t['mean']
    mat, fin = MATERIALS[st.material], FINISHES[st.finish]
    temps = {}
    sun_h = np.array([sun[0], sun[1], 0.0]) if el > 0 and math.hypot(sun[0], sun[1]) > 1e-6 else None
    for t in tree:
        p = t['part']
        near = _bottom(p) - rf['top'] < 0.25 and t['supports'] != ['body']
        tp = thermal_part(p, mat, fin, sun_h, near)
        r = solve_part(tp, absorbed(tp, sun, dni, dhi, ghi, st.albedo), T_air, v, T_roof if near else None)
        r['depth'] = tp.depth()
        r['g'] = tp.g
        temps[t['uid']] = r
    w = Warp(tree, rf, st, T_air, T_roof, temps, st.t_cal_c)
    return w, {'t_air': T_air, 't_roof': T_roof, 'roof_tau_s': roof_t['tau_s'], 'sun': sun, 'dni': dni, 'dhi': dhi,
               'ghi': ghi, 'parts': temps}


def device_drift(w: Warp, dev, node) -> dict:
    """A device's move (mm) and angle changes (deg) relative to the calibration."""
    U, W = w.motion(node, dev['P'])
    dyaw, dpitch, droll = angle_change(dev['R'], W)
    return {'dx_mm': float(U[0] * 1000), 'dy_mm': float(U[1] * 1000), 'dz_mm': float(U[2] * 1000),
            'move_mm': float(np.linalg.norm(U) * 1000), 'd_yaw_deg': float(dyaw), 'd_pitch_deg': float(dpitch),
            'd_roll_deg': float(droll), 'turn_deg': float(math.degrees(np.linalg.norm(W))), '_U': U, '_W': W}


def pair_drift(pa, a, b, st: ThermoVibSettings, da, db) -> dict:
    """The relative drift of b against camera a: angle (deg), and in a's image the rotation's shift (centre, worst
    px) plus the parallax of the relative move at st.range_m (centre)."""
    w = db['_W'] - da['_W']
    t = db['_U'] - da['_U']
    out = {'deg': float(math.degrees(np.linalg.norm(w))), 'move_mm': float(np.linalg.norm(t) * 1000)}
    intr = a['intr']
    if intr is not None:
        W2O = BODY_TO_OPTICAL @ a['R'].T
        c, worst = px_shift(intr, W2O, w)
        to = W2O @ t
        par = math.hypot(to[0], to[1]) / st.range_m * px_per_rad(intr)
        out.update(px_centre=c, px_rot=worst, px_parallax=par, px=worst + par)
    else:
        out.update(px_centre=None, px_rot=None, px_parallax=None, px=None)
    return out


# =========================================================================== the half-car
@dataclass
class HalfCar:
    m: float                # sprung mass, kg
    I: float                # pitch inertia, kg m²
    x_cg: float
    z_cg: float
    x_f: float
    x_r: float
    k_f: float              # N/m per axle (both wheels)
    k_r: float
    c_f: float
    c_r: float
    kt_f: float
    kt_r: float
    mu_f: float
    mu_r: float
    source: str = 'typical'

    @property
    def a(self):
        return self.x_f - self.x_cg

    @property
    def b(self):
        return self.x_cg - self.x_r

    @property
    def wheelbase(self):
        return self.x_f - self.x_r

    def matrices(self):
        """(M, C, K, Kt) over q = [heave z (up), pitch θ (nose down), front unsprung, rear unsprung]."""
        a, b = self.a, self.b
        M = np.diag([self.m, self.I, self.mu_f, self.mu_r])

        def mat(kf, kr):
            return np.array([[kf + kr, -a * kf + b * kr, -kf, -kr],
                             [-a * kf + b * kr, a * a * kf + b * b * kr, a * kf, -b * kr],
                             [-kf, a * kf, kf, 0.0],
                             [-kr, -b * kr, 0.0, kr]])
        K = mat(self.k_f, self.k_r)
        K[2, 2] += self.kt_f
        K[3, 3] += self.kt_r
        C = mat(self.c_f, self.c_r)
        Kt = np.array([[0.0, 0.0], [0.0, 0.0], [self.kt_f, 0.0], [0.0, self.kt_r]])
        return M, C, K, Kt

    def response(self, f, v_mps) -> np.ndarray:
        """q per unit road height under the front wheel (the rear's delayed by wheelbase/V): complex (F, 4)."""
        M, C, K, Kt = self.matrices()
        w = 2 * math.pi * np.asarray(f, float)
        A = -w[:, None, None] ** 2 * M + 1j * w[:, None, None] * C + K
        tau = self.wheelbase / max(v_mps, 1e-6)
        U = np.stack([np.ones_like(w, dtype=complex), np.exp(-1j * w * tau)], axis=1)
        rhs = U @ Kt.T
        return np.linalg.solve(A, rhs[..., None])[..., 0]

    def modes(self) -> dict:
        """Undamped natural frequencies (Hz): the body's two (bounce-like, pitch-like) and the two wheel hops."""
        M, _, K, _ = self.matrices()
        ev, vec = np.linalg.eig(np.linalg.solve(M, K))
        f = np.sqrt(np.maximum(ev.real, 0)) / (2 * math.pi)
        order = np.argsort(f)
        f, vec = f[order], vec[:, order].real
        body = [float(x) for x in f[:2]]
        bounce = 0 if abs(vec[0, 0]) * 1.0 >= abs(vec[1, 0]) * max(self.a, self.b) else 1
        return {'bounce_hz': body[bounce], 'pitch_hz': body[1 - bounce], 'hop_hz': [float(x) for x in f[2:]]}

    def to_dict(self):
        return {'sprung_kg': self.m, 'pitch_inertia_kgm2': self.I, 'cg_x_m': self.x_cg, 'cg_z_m': self.z_cg,
                'front_axle_x_m': self.x_f, 'rear_axle_x_m': self.x_r, 'wheel_rate_n_mm': [self.k_f / 1000,
                                                                                       self.k_r / 1000],
                'damping_ns_m': [self.c_f, self.c_r], 'tyre_rate_n_mm': [self.kt_f / 1000, self.kt_r / 1000],
                'unsprung_kg': [self.mu_f, self.mu_r], 'suspension_source': self.source}


def half_car(project, st: ThermoVibSettings, variant=None) -> HalfCar:
    """The half-car of the project's suspension (wheel and tyre rates, kerb loads, unsprung masses) plus the rig."""
    su = SU.settings_for(project)
    g = SU.vehicle_geometry(project)
    x_f = (g['x']['FL'] + g['x']['FR']) / 2
    x_r = (g['x']['RL'] + g['x']['RR']) / 2
    if x_f - x_r < 0.5:
        x_f = x_r + SU.TYP_WHEELBASE
    dyn = SU.Dynamics.from_dict((getattr(project.analysis, 'terrain', None) or {}).get('motion')
                                if isinstance(getattr(project.analysis, 'terrain', None), dict) else None)
    mu = dyn.unsprung_kg
    c = su.corners
    kerb_f = c['FL'].kerb_load + c['FR'].kerb_load
    kerb_r = c['RL'].kerb_load + c['RR'].kerb_load
    ms_f, ms_r = max(kerb_f - 2 * mu, 0.3 * kerb_f), max(kerb_r - 2 * mu, 0.3 * kerb_r)
    comps = SU.rig_components(project, su, variant)
    m = ms_f + ms_r + sum(x['mass'] for x in comps)
    x_cg = (ms_f * x_f + ms_r * x_r + sum(x['mass'] * x['x'] for x in comps)) / m
    x_cg = min(max(x_cg, x_r + 0.1 * (x_f - x_r)), x_f - 0.1 * (x_f - x_r))
    z_cg = dyn.cg_height if dyn.cg_height > 0 else SU.cg_height_estimate(project)
    a, b = x_f - x_cg, x_cg - x_r
    I = m * DYNAMIC_INDEX * a * b
    k_f = (c['FL'].wheel_rate + c['FR'].wheel_rate) * 1000.0
    k_r = (c['RL'].wheel_rate + c['RR'].wheel_rate) * 1000.0
    m_f, m_r = m * b / (a + b), m * a / (a + b)
    c_f = 2 * st.zeta_body * math.sqrt(k_f * m_f)
    c_r = 2 * st.zeta_body * math.sqrt(k_r * m_r)
    return HalfCar(m, I, x_cg, z_cg, x_f, x_r, k_f, k_r, c_f, c_r, (c['FL'].tyre_rate + c['FR'].tyre_rate) * 1000.0,
                   (c['RL'].tyre_rate + c['RR'].tyre_rate) * 1000.0, 2 * mu, 2 * mu, su.source)


def base_accel(hc: HalfCar, X, P) -> np.ndarray:
    """Acceleration (F, 3) at body point P per unit road height, from the half-car response X (F, 4) at the grid's
    angular frequencies applied by the caller: here X holds accelerations (−ω² q)."""
    P = np.asarray(P, float)
    ax = (P[2] - hc.z_cg) * X[:, 1]
    az = X[:, 0] - (P[0] - hc.x_cg) * X[:, 1]
    return np.stack([ax, np.zeros_like(ax), az], axis=1)


def _base_jac(hc: HalfCar, P) -> np.ndarray:
    """(3, 4): body-point acceleration from q̈."""
    P = np.asarray(P, float)
    return np.array([[0.0, P[2] - hc.z_cg, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [1.0, -(P[0] - hc.x_cg), 0.0, 0.0]])


# =========================================================================== mount modes
@dataclass
class Mode:
    """One mount's first bending mode on the body: a cantilever (bracket, mast) or a crossbar between its feet."""
    key: str
    name: str
    kind: str               # bracket | mast | bar
    base: np.ndarray        # where the body (or the roof) drives it
    tip: np.ndarray         # where its deflection is taken (the tip, or the crossbar's middle)
    dirs: list              # [(e: deflection direction, axis: rotation axis per metre of deflection)]
    k: float
    m_eff: float
    zeta: float
    L: float
    E: float
    I: float
    m_tip: float
    m_beam: float
    source: str
    span: float = 0.0
    axis: np.ndarray | None = None      # a crossbar's own axis

    @property
    def f1(self) -> float:
        return math.sqrt(self.k / max(self.m_eff, 1e-9)) / (2 * math.pi)

    def shape(self, P) -> tuple[float, float, np.ndarray]:
        """(deflection share at P, rotation rad per m of the mode's deflection, the point the rotation acts from) for
        a device at P riding on this mode."""
        if self.kind == 'bar':
            s = self.span
            eta = float((np.asarray(P, float) - self.base) @ self.axis)
            x = s / 2 - abs(eta)                           # from the nearer foot
            phi = x * (3 * s * s - 4 * x * x) / s ** 3
            dphi = -math.copysign(1.0, eta if eta != 0 else 1.0) * (3 * s * s - 12 * x * x) / s ** 3
            q = self.base + eta * self.axis
            return phi, dphi, q
        return 1.0, 1.5 / self.L, self.tip


def _cantilever_dirs(a):
    """Two bending directions square to a beam axis a, each with its rotation axis (a × e)."""
    a = _unit(a, (0.0, 0.0, 1.0))
    e1 = np.array([0.0, 0.0, 1.0]) - a[2] * a
    if np.linalg.norm(e1) < 1e-6:
        e1 = np.array([1.0, 0.0, 0.0]) - a[0] * a
    e1 = _unit(e1)
    e2 = np.cross(a, e1)
    return [(e1, np.cross(a, e1)), (e2, np.cross(a, e2))]


def _bracket_mode(dev, node, Q, rf, st, brackets, printed=None) -> Mode:
    """The device's own bracket as a cantilever from the surface or part it rides on. printed: {'E' MPa, 'rho'
    g/cm³, 'label'} of the bracket generator's material for a camera's printed bracket."""
    P = dev['P']
    gen = None
    if brackets is not None and dev['kind'] == 'camera':
        b = next((b for b in getattr(brackets, 'brackets', []) or [] if b.uid == dev['uid']), None)
        if b is not None and isinstance(b.params, dict) and b.params.get('stiffness'):
            gen = b
    if dev['kind'] == 'camera':
        a = dev['R'][:, 0]
        L = max(float(dev['obj'].standoff_mm) / 1000.0, MIN_LEVER_M)
        mat_id, b_w, b_t, _ = TYP_BRACKET['camera']
        zeta = st.zeta_printed
    else:
        if isinstance(node, int):
            d = P - Q
            a = _unit(d, (0.0, 0.0, 1.0))
            L = max(float(np.linalg.norm(d)), MIN_LEVER_M)
        elif node == 'roof':
            a = np.array([0.0, 0.0, 1.0])
            L = max(P[2] - rf['top'], MIN_LEVER_M)
        else:
            a = dev['R'][:, 0]
            L = TYP_SENSOR_LEVER_M
        mat_id, b_w, b_t, _ = TYP_BRACKET['other']
        zeta = st.zeta_mount
    mat = dict(MATERIALS[mat_id])
    if mat_id == 'asa' and printed is not None:            # the bracket generator's material (core.brackets)
        mat.update(E=printed['E'] * 1e6, rho=printed['rho'] * 1000.0, label=printed['label'])
    E, I = mat['E'], b_w * b_t ** 3 / 12.0
    m_beam = mat['rho'] * b_w * b_t * L
    src = (f'typical {"printed " if mat_id == "asa" else ""}{mat["label"]} bracket, {b_w * 1000:.0f} × '
           f'{b_t * 1000:.0f} mm section, {L * 1000:.0f} mm lever')
    if gen is not None:
        sf = gen.params['stiffness']
        E = float(sf.get('E_mpa', 1800.0)) * 1e6
        I = float(sf.get('section_I_mm4', 1.0)) * 1e-12
        L = max(float(gen.params.get('lever_mm', L * 1000.0)) / 1000.0, MIN_LEVER_M)
        m_beam = float((gen.mesh or {}).get('mass_g', 15.0)) / 1000.0
        src = f'generated bracket: beam estimate, {L * 1000:.0f} mm lever'
    base = P - a * L
    k = 3 * E * I / L ** 3
    return Mode(f'bracket:{dev["uid"]}', f'{dev["tag"]} bracket', 'bracket', base, P.copy(), _cantilever_dirs(a), k,
                dev['mass'] + 0.24 * m_beam, zeta, L, E, I, dev['mass'], m_beam, src)


def _carried(tree, devs, nodes) -> dict:
    """Mass (kg) each part carries: the devices riding on it or on parts above it, and those parts, each shared over
    the parts that carry it."""
    from .suspension import rig_part_mass
    n = len(tree)
    own = np.zeros(n)
    for i, t in enumerate(tree):
        own[i] = rig_part_mass(t['part'])
    load = np.zeros(n)
    for d, (node, _) in zip(devs, nodes):
        if isinstance(node, int):
            load[node] += d['mass']
    order = sorted(range(n), key=lambda i: -tree[i]['part'].z)       # top down
    for i in order:
        sups = [s for s in tree[i]['supports'] if isinstance(s, int)]
        for s in sups:
            load[s] += (load[i] + own[i]) / len(sups)
    return {tree[i]['uid']: (float(load[i]), float(own[i])) for i in range(n)}


def part_modes(tree, carried, st: ThermoVibSettings) -> dict:
    """The flexible parts' first modes: a mast as a cantilever from its base, a crossbar between its feet. Boxes are
    taken as rigid."""
    mat = MATERIALS[st.material]
    scale = mat['rho'] / SU.ALU_KG_M3
    out = {}
    for i, t in enumerate(tree):
        p = t['part']
        load, own = carried[t['uid']]
        own *= scale
        if p.kind == 'cylinder':
            L = max(p.height, 0.02)
            I = tube_I(p.radius)
            base = t['base']
            tip = base + np.array([0.0, 0.0, L])
            k = 3 * mat['E'] * I / L ** 3
            out[i] = Mode(f'part:{p.uid}', p.name, 'mast', base, tip, _cantilever_dirs([0.0, 0.0, 1.0]), k,
                          load + 0.24 * own, st.zeta_mount, L, mat['E'], I, load, own,
                          f'{mat["label"]} tube Ø{2 * p.radius * 1000:.0f} × {WALL_M * 1000:.0f} mm, '
                          f'{L * 1000:.0f} mm tall')
        elif p.kind == 'bar':
            span = max(p.length - 2 * FEET_INSET_M, 0.6 * p.length)
            I = tube_I(p.radius)
            yb = _rot(p.yaw)[:, 1]
            k = 48 * mat['E'] * I / span ** 3
            dirs = [(np.array([0.0, 0.0, 1.0]), np.cross(yb, [0.0, 0.0, 1.0]))]
            eh = _unit(np.cross(yb, [0.0, 0.0, 1.0]))
            dirs.append((eh, np.cross(yb, eh)))
            out[i] = Mode(f'part:{p.uid}', p.name, 'bar', t['base'].copy(), t['base'].copy(), dirs, k,
                          load + 0.49 * own, st.zeta_mount, span, mat['E'], I, load, own,
                          f'{mat["label"]} tube Ø{2 * p.radius * 1000:.0f} × {WALL_M * 1000:.0f} mm, '
                          f'{span * 1000:.0f} mm between its feet', span=span, axis=yb)
    return out


def freq_grid(extra=()) -> np.ndarray:
    """Log grid F_LO-F_HI Hz, PER_DECADE points a decade, densified ±6 % around the given frequencies."""
    n = int(round(math.log10(F_HI / F_LO) * PER_DECADE)) + 1
    f = [np.geomspace(F_LO, F_HI, n)]
    for fx in extra:
        if F_LO < fx < F_HI:
            f.append(fx * (1 + np.linspace(-0.06, 0.06, 61)))
    return np.unique(np.concatenate(f))


# =========================================================================== the study
@dataclass
class ThermoVibResult:
    settings: ThermoVibSettings
    variant: str
    variant_uid: str
    tol: dict
    day: dict               # date, location, hours (local), sun
    heading_deg: float
    heading_auto: bool
    headings: list          # [{'heading', 'worst_ratio', 'worst_deg', 'worst_mm', 'worst_px'}]
    hours: list             # per hour: air, roof, sun, irradiance, parts' temperatures
    parts: list             # the rig parts: support, thermal summary, mismatch, mode
    devices: list           # per device: carrier, thermal series and worst, vibration, events
    pairs: list
    vehicle: dict
    psd: dict               # 'f', 'road', 'cobble', per device 'accel' (F,), 'flex' (F,), 'trans' (F,)
    events: dict
    findings: list
    notes: list = field(default_factory=list)
    seconds: float = 0.0

    def worst(self) -> str:
        return self.findings[0]['severity'] if self.findings else 'good'

    def device(self, uid):
        return next((d for d in self.devices if d['uid'] == uid), None)


def _series_worst(series, tol) -> dict:
    best = {'ratio': 0.0, 'deg': 0.0, 'mm': 0.0, 'hour': None}
    for s in series:
        deg = max(abs(s['d_yaw_deg']), abs(s['d_pitch_deg']), abs(s['d_roll_deg']), s['turn_deg'])
        mm = max(abs(s['dx_mm']), abs(s['dy_mm']), abs(s['dz_mm']))
        r = max(deg / tol['deg'], mm / tol['mm'])
        if r >= best['ratio']:
            best = {'ratio': r, 'deg': deg, 'mm': mm, 'hour': s['hour']}
    return best


def _thermal_day(project, tree, rf, st, day, heading, devs, nodes, pairs, by, tol):
    hours, dev_series, pair_series = [], {d['uid']: [] for d in devs}, {i: [] for i in range(len(pairs))}
    for k, h in enumerate(day['local_h']):
        w, info = thermal_state(project, tree, rf, st, float(h), float(day['az'][k]), float(day['el'][k]),
                                float(day['dist_au'][k]), heading)
        dd = {}
        for d, (node, _) in zip(devs, nodes):
            x = device_drift(w, d, node)
            dd[d['uid']] = x
            dev_series[d['uid']].append(dict({kk: v for kk, v in x.items() if not kk.startswith('_')}, hour=float(h)))
        for i, pr in enumerate(pairs):
            a, b = by[pr['a']], by[pr['b']]
            pair_series[i].append(dict(pair_drift(project, a, b, st, dd[a['uid']], dd[b['uid']]), hour=float(h)))
        parts = {}
        for t in tree:
            tp = info['parts'][t['uid']]
            wi = w.info[t['uid']]
            parts[t['uid']] = {'T': tp['T'], 'mean_c': tp['mean'], 'gradient_c': tp['dT'], 'depth_m': tp['depth'],
                               'kappa_per_m': wi['kappa'], 'tau_s': tp['tau_s'], 'h_w_m2k': tp['h'],
                               'mismatch_mm': wi['mismatch_mm'], 'axial_n': wi['N'], 'euler_n': wi['N_cr'],
                               'stress_mpa': wi['stress_mpa'], 'bow_amplification': wi['amp']}
        hours.append({'hour': float(h), 't_air_c': info['t_air'], 't_roof_c': info['t_roof'],
                      'roof_tau_s': info['roof_tau_s'], 'sun_az_deg': float(day['az'][k]),
                      'sun_el_deg': float(day['el'][k]), 'dni': info['dni'], 'dhi': info['dhi'], 'ghi': info['ghi'],
                      'parts': parts})
    worst_dev = {u: _series_worst(s, tol) for u, s in dev_series.items()}
    worst_px = 0.0
    for i, s in pair_series.items():
        for e in s:
            if e['px'] is not None:
                worst_px = max(worst_px, e['px'] / tol['px'])
    ratio = max([w['ratio'] for w in worst_dev.values()] + [worst_px])
    return hours, dev_series, pair_series, worst_dev, ratio


def _rms(H, G, f) -> float:
    """RMS of a response with transfer H ((F,) or (F, k) per unit road height) to the road PSD G."""
    H = np.asarray(H)
    p = np.abs(H) ** 2
    if p.ndim == 2:
        p = p.sum(axis=1)
    return math.sqrt(max(integrate(f, p * G), 0.0))


def _vib_transfer(hc, pm, modes, dev, node, tree, X_acc, w, f):
    """Per unit road height at each frequency: the device's flex (relative to the body) linear (F, 3) m and angular
    (F, 3) rad, its body motion linear and angular, the base acceleration at its mount (F, 3) and the modes it
    rides on."""
    P = dev['P']
    chain = [pm[i] for i in (chain_of(tree, node) if isinstance(node, int) else []) if i in pm]
    chain.append(modes[f'bracket:{dev["uid"]}'])
    lin = np.zeros((len(f), 3), complex)
    ang = np.zeros((len(f), 3), complex)
    for md in chain:
        aB = base_accel(hc, X_acc, md.base)
        rel = sdof_relative(f, md.f1, md.zeta)
        phi, dphi, q = md.shape(P)
        for e, ax in md.dirs:
            u = rel * (aB @ e)                              # (F,)
            ang_k = (u * dphi)[:, None] * ax[None, :]
            lin += (u * phi)[:, None] * e[None, :] + np.cross(ang_k, (P - q)[None, :])
            ang += ang_k
    X = -X_acc / np.where(w > 0, w * w, 1.0)[:, None]       # displacement from acceleration
    body_ang = np.zeros((len(f), 3), complex)
    body_ang[:, 1] = X[:, 1]
    return lin, ang, body_ang, base_accel(hc, X_acc, chain[-1].base), chain


def _events(hc, all_modes, dev_links, st):
    """The bump (half-sine) and the expansion joint (step) through the half-car and every mount mode: per device the
    peak base acceleration (g), peak flex (mm, deg), peak angle with the body's pitch (deg), the body's ring-down."""
    from scipy.linalg import expm
    M, C, K, Kt = hc.matrices()
    Mi = np.linalg.inv(M)
    keys = list(all_modes)
    nd = sum(len(all_modes[k].dirs) for k in keys)
    n = 8 + 2 * nd
    A = np.zeros((n, n))
    B = np.zeros((n, 2))
    A[:4, 4:8] = np.eye(4)
    A[4:8, :4] = -Mi @ K
    A[4:8, 4:8] = -Mi @ C
    B[4:8] = Mi @ Kt
    slot = {}
    j = 8
    for k in keys:
        md = all_modes[k]
        J = _base_jac(hc, md.base)
        wn = 2 * math.pi * md.f1
        for di, (e, _) in enumerate(md.dirs):
            slot[(k, di)] = j
            row = e @ J                                       # base accel along e from q̈
            A[j, j + 1] = 1.0
            A[j + 1, j] = -wn * wn
            A[j + 1, j + 1] = -2 * md.zeta * wn
            A[j + 1, :4] += -(row @ (-Mi @ K))
            A[j + 1, 4:8] += -(row @ (-Mi @ C))
            B[j + 1] = -(row @ (Mi @ Kt))
            j += 2
    aug = np.zeros((n + 2, n + 2))
    aug[:n, :n] = A * DT_S
    aug[:n, n:] = B * DT_S
    E = expm(aug)
    Ad, Bd = E[:n, :n], E[:n, n:]
    L = hc.wheelbase
    out = {}
    for name, v_kmh, prof, length, height in (
            ('bump', st.bump_kmh, 'half-sine', st.bump_len_m, st.bump_h_mm / 1000.0),
            ('joint', st.speed_kmh, 'step', 0.0, st.joint_mm / 1000.0)):
        v = v_kmh / 3.6
        lead = 0.1
        t_end = lead + (L + length) / v + TAIL_S
        steps = int(math.ceil(t_end / DT_S))
        t = np.arange(steps) * DT_S
        sf = v * (t - lead)
        sr = sf - L

        def z(s):
            if prof == 'step':
                return np.where(s >= 0, height, 0.0)
            return np.where((s >= 0) & (s <= length), height * np.sin(np.pi * np.clip(s, 0, length) / length), 0.0)
        U = np.stack([z(sf), z(sr)], axis=1)
        X = np.zeros((steps, n))
        x = np.zeros(n)
        for i in range(steps):
            X[i] = x
            x = Ad @ x + Bd @ U[i]
        qdd = (-(Mi @ K) @ X[:, :4].T - (Mi @ C) @ X[:, 4:8].T + (Mi @ Kt) @ U.T).T     # (steps, 4)
        t_in = lead + (L + length) / v
        devs = {}
        for uid, (P, links) in dev_links.items():
            lin = np.zeros((steps, 3))
            ang = np.zeros((steps, 3))
            for k in links:
                md = all_modes[k]
                phi, dphi, q = md.shape(P)
                for di, (e, ax) in enumerate(md.dirs):
                    u = X[:, slot[(k, di)]]
                    a_k = (u * dphi)[:, None] * ax[None, :]
                    lin += (u * phi)[:, None] * e[None, :] + np.cross(a_k, (P - q)[None, :])
                    ang += a_k
            Jm = _base_jac(hc, all_modes[links[-1]].base)
            acc = qdd @ Jm.T
            amag = np.linalg.norm(acc, axis=1)
            pk = float(amag.max())
            after = np.nonzero((t > t_in) & (amag > RING_FRACTION * pk))[0]
            ring = float(t[after[-1]] - t_in) if len(after) else 0.0
            tot = ang.copy()
            tot[:, 1] += X[:, 1]
            md0 = min((all_modes[k] for k in links), key=lambda m: m.f1)
            devs[uid] = {'peak_accel_g': pk / 9.80665, 'peak_flex_mm': float(np.max(np.linalg.norm(lin, axis=1)) * 1000),
                         'peak_flex_deg': float(math.degrees(np.max(np.linalg.norm(ang, axis=1)))),
                         'peak_angle_deg': float(math.degrees(np.max(np.linalg.norm(tot, axis=1)))),
                         'body_ring_down_s': ring,
                         'mount_ring_down_s': math.log(1 / RING_FRACTION) / (md0.zeta * 2 * math.pi * md0.f1)}
        out[name] = {'profile': prof, 'speed_kmh': v_kmh, 'height_mm': height * 1000.0,
                     'length_m': length if prof != 'step' else None,
                     'peak_heave_mm': float(np.max(np.abs(X[:, 0])) * 1000),
                     'peak_pitch_deg': float(math.degrees(np.max(np.abs(X[:, 1])))),
                     'devices': devs}
    return out


def run(project, engine=None, settings: ThermoVibSettings | None = None, variant=None, progress=None,
        cancel=None, brackets=None, events=True) -> ThermoVibResult | None:
    """The whole study: the thermal drift hour by hour (at the set heading, or the worst of 12), the road vibration of
    every mount, the events, the pairs and the findings. brackets: a core.brackets result whose generated brackets'
    beam stiffness replaces the typical brackets. None when cancelled. engine is not needed (kept for the common
    signature)."""
    t0 = time.perf_counter()
    p = SU._project_for(project, variant)
    v = p.variant
    st = settings if isinstance(settings, ThermoVibSettings) else (
        ThermoVibSettings.from_dict(settings) if isinstance(settings, dict) else settings_of(p))
    prog = progress or (lambda f: None)
    stop = cancel or (lambda: False)
    tol = tolerance(p, st)
    rf = roof_box(p)
    tree = rig_tree(p, rf)
    devs = devices(p, v)
    nodes = [carrier(tree, d['P'], rf) for d in devs]
    by = {d['uid']: d for d in devs}
    pairs = pairs_of(p, devs, v)
    notes = []
    # ---------------------------------------------------------------- thermal
    day = day_hours(p, st)
    if day['note']:
        notes.append(day['note'])
    heads = [st.heading_deg] if st.heading_deg is not None else [float(h) for h in range(0, 360, 30)]
    env = []
    best = None
    for i, hd in enumerate(heads):
        r = _thermal_day(p, tree, rf, st, day, hd, devs, nodes, pairs, by, tol)
        env.append({'heading': hd, 'worst_ratio': r[4],
                    'worst_deg': max([w['deg'] for w in r[3].values()], default=0.0),
                    'worst_mm': max([w['mm'] for w in r[3].values()], default=0.0)})
        if best is None or r[4] > best[1][4] + 1e-12:
            best = (hd, r)
        if stop():
            return None
        prog(0.4 * (i + 1) / len(heads))
    heading, (hours, dev_series, pair_series, worst_dev, _) = best
    # ---------------------------------------------------------------- vibration
    hc = half_car(p, st, v)
    modes_hc = hc.modes()
    carried = _carried(tree, devs, nodes)
    pm = part_modes(tree, carried, st)
    from .brackets import MATERIALS as BM, BracketSettings
    bm = BracketSettings.from_dict(getattr(p.analysis, 'brackets', None)).material
    printed = dict(BM[bm], label=f'{bm}')
    modes = {md.key: md for md in pm.values()}
    for d, (node, Q) in zip(devs, nodes):
        modes[f'bracket:{d["uid"]}'] = _bracket_mode(d, node, Q, rf, st, brackets, printed)
    V = st.speed_kmh / 3.6
    Vc = st.cobble_kmh / 3.6
    sett_f = Vc / st.sett_m
    extra = [m.f1 for m in modes.values()] + [modes_hc['bounce_hz'], modes_hc['pitch_hz']] + modes_hc['hop_hz']
    f = freq_grid(extra)
    w = 2 * math.pi * f
    G = road_psd_temporal(f, st.road_class, V)
    Gc = road_psd_temporal(f, st.cobble_class, Vc)
    Xa = -(w * w)[:, None] * hc.response(f, V)
    Xc = -(w * w)[:, None] * hc.response(f, Vc)
    prog(0.5)
    if stop():
        return None
    from .timing import TimingSettings
    tm = TimingSettings.from_dict(getattr(p.analysis, 'timing', None))
    exp_s = tm.exposure_ms / 1000.0
    psd = {'f': f, 'road': G, 'cobble': Gc, 'devices': {}}
    dev_out = []
    links = {}
    for d, (node, Q) in zip(devs, nodes):
        out = {}
        for tag, XX, GG in (('road', Xa, G), ('cobble', Xc, Gc)):
            lin, ang, body_ang, aB, chain = _vib_transfer(hc, pm, modes, d, node, tree, XX, w, f)
            rate = 1j * w[:, None] * (ang + body_ang)
            r = {'accel_rms': _rms(aB, GG, f), 'flex_mm_rms': _rms(lin, GG, f) * 1000,
                 'flex_deg_rms': math.degrees(_rms(ang, GG, f)),
                 'angle_deg_rms': math.degrees(_rms(ang + body_ang, GG, f)),
                 'rate_deg_s_rms': math.degrees(_rms(rate, GG, f))}
            if d['intr'] is not None:
                W2O = BODY_TO_OPTICAL @ d['R'].T
                ro = rate @ W2O.T
                wperp = _rms(ro[:, :2], GG, f)
                r['blur_px'] = wperp * exp_s * px_per_rad(d['intr'])
            else:
                r['blur_px'] = None
            pa = (np.abs(aB) ** 2).sum(axis=1) * GG
            r['peak_hz'] = float(f[int(np.argmax(pa))]) if pa.max() > 0 else None
            out[tag] = r
            if tag == 'road':
                psd['devices'][d['uid']] = {'accel': pa, 'flex': (np.abs(ang) ** 2).sum(axis=1) * GG,
                                            'h_body': np.sqrt((np.abs(aB) ** 2).sum(axis=1)) / (w * w),
                                            '_lin': lin, '_ang': ang}
        low = min(chain, key=lambda m: m.f1)
        peaks = [('excitation peak', out['road']['peak_hz']), ('cobblestone setts', sett_f)]
        flags = []
        for md in chain:
            why = resonance(md.f1, peaks)
            if why:
                flags.append({'mode': md.name, 'kind': md.kind, 'f1_hz': md.f1, 'why': why})
        psd['devices'][d['uid']]['trans'] = transmissibility(f / low.f1, low.zeta)
        psd['devices'][d['uid']]['mode_hz'] = low.f1
        links[d['uid']] = (d['P'], [md.key for md in chain])
        dev_out.append({'uid': d['uid'], 'tag': d['tag'], 'name': d['name'], 'kind': d['kind'], 'slot': d['slot'],
                        'mass_kg': d['mass'],
                        'on': (tree[node]['name'] if isinstance(node, int) else node),
                        'on_uid': (tree[node]['uid'] if isinstance(node, int) else node),
                        'chain': [tree[i]['name'] for i in chain_of(tree, node)] if isinstance(node, int) else [],
                        'thermal': dev_series[d['uid']], 'thermal_worst': worst_dev[d['uid']],
                        'modes': [{'name': md.name, 'kind': md.kind, 'f1_hz': md.f1, 'k_n_m': md.k,
                                   'm_eff_kg': md.m_eff, 'zeta': md.zeta, 'source': md.source} for md in chain],
                        'f1_hz': low.f1, 'resonance': flags, 'vib': out['road'], 'cobble': out['cobble'],
                        'px_per_rad': px_per_rad(d['intr']) if d['intr'] is not None else None})
    prog(0.65)
    if stop():
        return None
    # ---------------------------------------------------------------- pairs
    pair_out = []
    for i, pr in enumerate(pairs):
        a, b = by[pr['a']], by[pr['b']]
        s = pair_series[i]
        wk = max(s, key=lambda e: (e['px'] if e['px'] is not None else e['deg']), default=None)
        da, db = psd['devices'][a['uid']], psd['devices'][b['uid']]
        rel_ang = db['_ang'] - da['_ang']
        rel_lin = db['_lin'] - da['_lin']
        vib = {'deg_rms': math.degrees(_rms(rel_ang, G, f)), 'mm_rms': _rms(rel_lin, G, f) * 1000}
        if a['intr'] is not None:
            W2O = BODY_TO_OPTICAL @ a['R'].T
            vib['px_rms'] = _rms((rel_ang @ W2O.T)[:, :2], G, f) * px_per_rad(a['intr'])
        else:
            vib['px_rms'] = None
        base = b['P'] - a['P']
        if np.linalg.norm(base) > 1e-6:
            e = _unit(base)
            vib['baseline_mm_rms'] = _rms((rel_lin @ e)[:, None], G, f) * 1000
        else:
            vib['baseline_mm_rms'] = 0.0
        pair_out.append(dict(pr, tags=(a['tag'], b['tag']), thermal=s, thermal_worst=wk, vib=vib))
    # ---------------------------------------------------------------- events
    ev = {}
    if events and devs:
        ev = _events(hc, modes, links, st)
        for d in dev_out:
            d['events'] = {k: e['devices'].get(d['uid']) for k, e in ev.items()}
    else:
        for d in dev_out:
            d['events'] = {}
    ev['cobble'] = {'class': st.cobble_class, 'speed_kmh': st.cobble_kmh, 'sett_m': st.sett_m, 'sett_hz': sett_f}
    if stop():
        return None
    prog(0.95)
    for k in list(psd['devices']):
        psd['devices'][k].pop('_lin', None)
        psd['devices'][k].pop('_ang', None)
    # ---------------------------------------------------------------- parts
    parts_out = []
    peak_i = int(np.argmax([h['t_air_c'] for h in hours])) if hours else None
    for i, t in enumerate(tree):
        p_ = t['part']
        th = [h['parts'][t['uid']] for h in hours]
        worst_g = max(th, key=lambda e: abs(e['gradient_c']), default=None)
        parts_out.append({'uid': t['uid'], 'name': t['name'], 'kind': p_.kind,
                          'on': [tree[s]['name'] if isinstance(s, int) else s for s in t['supports']],
                          'peak': th[peak_i] if peak_i is not None else None,
                          'max_mean_c': max((e['mean_c'] for e in th), default=None),
                          'max_gradient_c': worst_g['gradient_c'] if worst_g else None,
                          'max_mismatch_mm': max((abs(e['mismatch_mm']) for e in th), default=0.0),
                          'tau_s': max((e['tau_s'] for e in th), default=None),
                          'mode': ({'kind': pm[i].kind, 'f1_hz': pm[i].f1, 'k_n_m': pm[i].k, 'm_eff_kg': pm[i].m_eff,
                                    'source': pm[i].source} if i in pm else None)})
    res = ThermoVibResult(st, v.name, v.uid, tol, {k: (v_.tolist() if isinstance(v_, np.ndarray) else v_)
                                                   for k, v_ in day.items()},
                          float(heading), st.heading_deg is None, env, hours, parts_out, dev_out, pair_out,
                          dict(hc.to_dict(), modes=modes_hc, wheelbase_m=hc.wheelbase,
                               speed_kmh=st.speed_kmh, road_class=st.road_class,
                               road_rms_mm=road_rms(st.road_class) * 1000), psd, ev, [], notes)
    res.findings = _findings(res, p)
    res.seconds = time.perf_counter() - t0
    prog(1.0)
    return res


# =========================================================================== findings and issues
def _finding(out, sev, kind, message, uids=()):
    out.append({'severity': sev, 'kind': kind, 'message': message, 'uids': list(uids)})


def _hh(h) -> str:
    if h is None:
        return '—'
    m = int(round((h % 24) * 60))
    return f'{m // 60:02d}:{m % 60:02d}'


def _findings(res: ThermoVibResult, project) -> list:
    out = []
    st, tol = res.settings, res.tol
    if not res.devices:
        _finding(out, 'note', 'empty', 'No visible cameras or sensors in this variant: nothing to drift or shake.')
        return out
    # resonance
    for d in res.devices:
        for fl in d['resonance']:
            words = []
            for w in fl['why']:
                if w == 'body':
                    words.append('in the body band (1-2 Hz)')
                elif w == 'wheel hop':
                    words.append('in the wheel-hop band (10-15 Hz)')
                elif w == 'excitation peak':
                    words.append(f'at the peak of the body\'s shaking at the mount ({d["vib"]["peak_hz"]:.1f} Hz on a '
                                 f'class {st.road_class} road at {st.speed_kmh:g} km/h)')
                else:
                    words.append(f'at the sett frequency on cobblestones ({res.events["cobble"]["sett_hz"]:.0f} Hz at '
                                 f'{st.cobble_kmh:g} km/h)')
            sev = 'warning' if fl['why'] == ['cobblestone setts'] else 'serious'
            _finding(out, sev, 'resonance',
                     f'{d["tag"]}: its {fl["mode"]} resonates at {fl["f1_hz"]:.1f} Hz, ' + ' and '.join(words) +
                     '. Stiffen the mount (a shorter lever, a thicker section, a strut) or move the mode away.',
                     [d['uid']])
    # thermal drift of each device
    over = [d for d in res.devices if d['thermal_worst']['ratio'] > 1.0]
    for d in over:
        w = d['thermal_worst']
        _finding(out, 'serious', 'thermal_drift',
                 f'{d["tag"]} on {d["on"]}: up to {w["deg"]:.2f}° and {w["mm"]:.1f} mm of thermal drift at '
                 f'{_hh(w["hour"])} ({res.day["date"]}, heading {res.heading_deg:.0f}°), beyond the {tol["deg"]:g}° / '
                 f'{tol["mm"]:g} mm tolerance ({TOL_LABEL[tol["source"]]}): calibrate warm, shade the rig or use a '
                 'stiffer, lighter-coloured mount.', [d['uid']])
    # pairs
    for pr in res.pairs:
        wk = pr['thermal_worst']
        if wk is None or wk['px'] is None:
            continue
        if wk['px'] > tol['px']:
            sev = 'serious' if wk['px'] > 3 * tol['px'] else 'warning'
            _finding(out, sev, 'pair_drift',
                     f'{pr["label"]} ({pr["kind"]}): the sun turns them {wk["deg"]:.3f}° apart by {_hh(wk["hour"])}: '
                     f'{wk["px"]:.1f} px in {pr["tags"][0]}\'s image (limit {tol["px"]:g} px). Their shared '
                     'calibration drifts over the day.', [pr['a'], pr['b']])
        vp = pr['vib'].get('px_rms')
        if vp is not None and vp > tol['px']:
            _finding(out, 'warning', 'vibration_pair',
                     f'{pr["label"]}: their mounts flex {pr["vib"]["deg_rms"]:.3f}° RMS apart on a class '
                     f'{st.road_class} road at {st.speed_kmh:g} km/h ({vp:.1f} px RMS in {pr["tags"][0]}\'s image).',
                     [pr['a'], pr['b']])
    # blur
    from .timing import TimingSettings
    tm = TimingSettings.from_dict(getattr(project.analysis, 'timing', None))
    for d in res.devices:
        b = d['vib'].get('blur_px')
        if b is not None and 3 * b > tm.blur_limit_px:
            _finding(out, 'warning', 'vibration_blur',
                     f'{d["tag"]}: the mount\'s shaking smears {b:.2f} px RMS ({3 * b:.2f} px at 3σ) in a '
                     f'{tm.exposure_ms:g} ms exposure on a class {st.road_class} road at {st.speed_kmh:g} km/h '
                     f'(limit {tm.blur_limit_px:g} px).', [d['uid']])
    # bump
    for d in res.devices:
        e = (d.get('events') or {}).get('bump')
        if e and e['peak_flex_deg'] > tol['deg']:
            _finding(out, 'warning', 'bump',
                     f'{d["tag"]}: a {st.bump_h_mm:g} mm speed bump at {st.bump_kmh:g} km/h flexes its mount '
                     f'{e["peak_flex_deg"]:.2f}° (it rings down in {e["mount_ring_down_s"]:.2f} s).', [d['uid']])
    # rack feet
    for pt in res.parts:
        if pt['kind'] != 'bar' or not pt['peak']:
            continue
        pk = max((h['parts'][pt['uid']] for h in res.hours), key=lambda e: abs(e['axial_n']), default=None)
        if st.feet == 'clamped' and pk and pk['euler_n'] and pk['axial_n'] > 0.8 * pk['euler_n']:
            _finding(out, 'serious', 'rack_buckling',
                     f'{pt["name"]}: clamped feet hold {pk["axial_n"] / 1000:.1f} kN of thermal compression, '
                     f'{pk["axial_n"] / pk["euler_n"] * 100:.0f} % of its buckling load: it will bow out. Let one '
                     'foot slide.', [pt['uid']])
        elif pt['max_mismatch_mm'] > 0.05:
            what = (f'one foot slides {pt["max_mismatch_mm"]:.2f} mm' if st.feet == 'sliding' else
                    f'the clamped feet take it as {abs(pk["stress_mpa"]):.0f} MPa in the bar')
            _finding(out, 'note', 'rack_mismatch',
                     f'{pt["name"]}: the aluminium bar and the steel roof differ by {pt["max_mismatch_mm"]:.2f} mm '
                     f'over its span at the day\'s peak: {what}.', [pt['uid']])
    slow = [pt for pt in res.parts if pt['tau_s'] and pt['tau_s'] > 900]
    if slow:
        _finding(out, 'note', 'slow', 'Slow to warm (time constant over 15 min): ' +
                 ', '.join(f'{pt["name"]} {pt["tau_s"] / 60:.0f} min' for pt in slow) +
                 '. The hour-by-hour figures are steady states; these parts lag them.', [pt['uid'] for pt in slow])
    if not any(f['severity'] in ISSUE_SEVERITIES for f in out):
        wd = max(res.devices, key=lambda d: d['thermal_worst']['ratio'])
        lo = min(res.devices, key=lambda d: d['f1_hz'])
        _finding(out, 'good', 'ok',
                 f'Thermal drift stays within the {tol["deg"]:g}° / {tol["mm"]:g} mm tolerance (worst {wd["tag"]}: '
                 f'{wd["thermal_worst"]["deg"]:.3f}°, {wd["thermal_worst"]["mm"]:.2f} mm at '
                 f'{_hh(wd["thermal_worst"]["hour"])}), and no mount resonates in band (lowest first mode '
                 f'{lo["f1_hz"]:.0f} Hz, {lo["tag"]}).')
    if not project.rig:
        _finding(out, 'note', 'no_rig', 'No rig parts: every device is taken as fixed to the body or the roof on its '
                                        'bracket. Add the roof rack or mast it rides on (Vehicle ▸ Add rig part) to '
                                        'see their warping and flex.')
    su = SU.settings_for(project)
    if su.is_typical:
        _finding(out, 'note', 'typical', 'The half-car uses typical suspension figures: enter your vehicle\'s in the '
                                         'suspension tab for its own body and wheel-hop frequencies.')
    order = {s: i for i, s in enumerate(SEVERITIES)}
    out.sort(key=lambda f: order.get(f['severity'], 9))
    return out


_ISSUES: dict = {}
_LOCK = threading.Lock()


def inputs_key(project, variant=None) -> str:
    """Everything the study reads, as text: a result made from other inputs is stale."""
    v = variant or project.variant
    a = project.analysis
    m = project.vehicle.mesh
    return json.dumps({'mesh': [id(m), getattr(m, 'n_tris', 0)], 'wheels': project.vehicle.wheel_centres,
                       'variant': v.to_dict(), 'rig': [r.to_dict() for r in project.rig],
                       'thermo_vib': a.thermo_vib, 'suspension': a.suspension, 'terrain': a.terrain,
                       'exposure': a.exposure, 'timing': a.timing, 'observability': a.observability,
                       'library': project.section('library')}, sort_keys=True, default=str)


def issues(project, vg=None) -> list:
    """Resonance in band and thermal drift beyond the calibration tolerance, for the Issues list (no events; cached on
    the inputs)."""
    from . import rules
    v = project.variant
    if not any(c.visible for c in v.cameras) and not any(s.visible for s in v.sensors):
        return []
    key = inputs_key(project)
    with _LOCK:
        hit = _ISSUES.get(key)
    if hit is not None:
        return list(hit)
    res = run(project, events=False)
    out = [rules.Issue(f['severity'], f['message'], f'{v.name} · thermal and vibration', list(f['uids']),
                       'thermo_vib') for f in res.findings
           if f['severity'] in ISSUE_SEVERITIES and f['kind'] in ISSUE_KINDS]
    with _LOCK:
        if len(_ISSUES) > 8:
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


def _clean(o, d=4):
    if isinstance(o, dict):
        return {k: _clean(v, d) for k, v in o.items() if not str(k).startswith('_')}
    if isinstance(o, (list, tuple)):
        return [_clean(v, d) for v in o]
    if isinstance(o, np.ndarray):
        return [_clean(v, d) for v in o.tolist()]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return _r(o, d)
    return o


def _third_octaves(f, y) -> tuple[list, list]:
    """A PSD averaged into one-third-octave bands (centre Hz, mean PSD) for the export."""
    f = np.asarray(f, float)
    y = np.asarray(y, float)
    centres = 1000.0 * 2.0 ** (np.arange(-30, 9) / 3.0)
    cs, vs = [], []
    for c in centres:
        lo, hi = c / 2 ** (1 / 6), c * 2 ** (1 / 6)
        m = (f >= lo) & (f < hi)
        if m.sum() >= 2 and F_LO <= c <= F_HI:
            cs.append(float(c))
            vs.append(integrate(f[m], y[m]) / (f[m][-1] - f[m][0]))
    return cs, vs


def summary(res: ThermoVibResult) -> dict:
    """Compact figures (the batch log, the assistant)."""
    wd = max(res.devices, key=lambda d: d['thermal_worst']['ratio'], default=None)
    lo = min(res.devices, key=lambda d: d['f1_hz'], default=None)
    wp = max((p for p in res.pairs if p['thermal_worst'] and p['thermal_worst']['px'] is not None),
             key=lambda p: p['thermal_worst']['px'], default=None)
    return {'date': res.day['date'], 'heading_deg': res.heading_deg, 'tolerance': res.tol['label'],
            'worst_drift': ({'device': wd['tag'], 'deg': _r(wd['thermal_worst']['deg'], 4),
                             'mm': _r(wd['thermal_worst']['mm'], 3), 'hour': _hh(wd['thermal_worst']['hour'])}
                            if wd else None),
            'worst_pair': ({'pair': wp['label'], 'px': _r(wp['thermal_worst']['px'], 3),
                            'deg': _r(wp['thermal_worst']['deg'], 4)} if wp else None),
            'lowest_mode': {'device': lo['tag'], 'f1_hz': _r(lo['f1_hz'], 2)} if lo else None,
            'resonances': [{'device': d['tag'], 'mode': r['mode'], 'f1_hz': _r(r['f1_hz'], 2), 'why': r['why']}
                           for d in res.devices for r in d['resonance']],
            'findings': [{'severity': f['severity'], 'message': f['message']} for f in res.findings
                         if f['severity'] in ISSUE_SEVERITIES]}


def to_doc(res: ThermoVibResult, project=None) -> dict:
    """The JSON export: frame, units, the settings with every default, the sun's day, every device's drift hour by
    hour and its vibration, the pairs, the parts, the half-car, the PSDs in one-third octaves and the findings."""
    st = res.settings
    f = res.psd['f']
    road_c, road_v = _third_octaves(f, res.psd['road'])
    psd_dev = {}
    for d in res.devices:
        pd = res.psd['devices'].get(d['uid'])
        if pd is not None:
            psd_dev[d['tag']] = {'accel_m2_s4_hz': _third_octaves(f, pd['accel'])[1],
                                 'flex_rad2_hz': _third_octaves(f, pd['flex'])[1]}
    return {
        'schema': SCHEMA, 'project': getattr(project, 'name', ''), 'variant': res.variant, 'frame': FRAME,
        'units': {'move': 'mm (vehicle frame)', 'angle': 'deg (each device\'s own yaw, pitch, roll change)',
                  'temperature': '°C', 'frequency': 'Hz', 'acceleration': 'm/s² RMS', 'image': 'px',
                  'psd': 'one-sided; road m²/Hz, acceleration (m/s²)²/Hz, angle rad²/Hz, one-third-octave means',
                  'time': 'local hours'},
        'signs': {'pitch': 'positive = nose (or the device) down', 'reference': 'every drift is relative to the '
                  'calibration in the garage at the calibration temperature'},
        'assumptions': ASSUMPTIONS, 'model': __doc__.split('Out of scope:')[0].strip(),
        'limits': 'Out of scope: ' + __doc__.split('Out of scope:')[1].strip(),
        'typical_sources': {'materials': {k: dict(m) for k, m in MATERIALS.items()},
                            'finishes': {k: dict(m) for k, m in FINISHES.items()},
                            'roof_paints': {'values': {k: dict(m) for k, m in ROOF_PAINTS.items()},
                                            'source': PAINT_SOURCE},
                            'iso8608': 'ISO 8608:2016, Gd(n0) the class\'s geometric mean, w = 2, n0 = 0.1 cycles/m',
                            'damping': 'body ζ typical 0.2-0.4 (Gillespie 1992); structure ζ 0.01-0.05 (Harris)'},
        'settings': st.to_dict(), 'tolerance': res.tol,
        'day': {'date': res.day['date'], 'auto_date': res.day['auto_date'], 'lat': res.day['lat'],
                'lon': res.day['lon'], 'window_h': list(res.day['window']), 'heading_deg': res.heading_deg,
                'heading_auto': res.heading_auto},
        'headings': _clean(res.headings),
        'hours': _clean(res.hours),
        'parts': _clean(res.parts),
        'devices': [_clean({k: v for k, v in d.items() if k not in ('thermal',)}) |
                    {'thermal': _clean(d['thermal'])} for d in res.devices],
        'pairs': _clean([{k: v for k, v in pr.items()} for pr in res.pairs]),
        'vehicle': _clean(res.vehicle),
        'events': _clean(res.events),
        'psd': {'third_octave_hz': _clean(road_c), 'road_m2_hz': _clean(road_v, 12), 'devices': _clean(psd_dev, 12),
                'speed_kmh': st.speed_kmh, 'road_class': st.road_class},
        'findings': [{'severity': x['severity'], 'kind': x['kind'], 'message': x['message']} for x in res.findings],
        'notes': list(res.notes), 'seconds': _r(res.seconds, 3),
    }


THERMAL_COLUMNS = ('hour', 'device', 'kind', 'on', 'dx_mm', 'dy_mm', 'dz_mm', 'move_mm', 'd_yaw_deg', 'd_pitch_deg',
                   'd_roll_deg', 'turn_deg', 't_air_c', 't_roof_c')


def thermal_csv(res: ThermoVibResult) -> str:
    """One row per hour and device: its move and angle changes from the calibration."""
    buf = io.StringIO()
    wr = csv.writer(buf, lineterminator='\n')
    wr.writerow(THERMAL_COLUMNS)
    for k, h in enumerate(res.hours):
        for d in res.devices:
            s = d['thermal'][k]
            wr.writerow([_hh(h['hour']), d['tag'], d['kind'], d['on']] +
                        [_r(s[c], 4) for c in THERMAL_COLUMNS[4:12]] + [_r(h['t_air_c'], 2), _r(h['t_roof_c'], 2)])
    return buf.getvalue()


def pairs_csv(res: ThermoVibResult) -> str:
    buf = io.StringIO()
    wr = csv.writer(buf, lineterminator='\n')
    wr.writerow(['pair', 'kind', 'hour', 'deg', 'move_mm', 'px', 'px_rotation', 'px_parallax'])
    for pr in res.pairs:
        for s in pr['thermal']:
            wr.writerow([pr['label'], pr['kind'], _hh(s['hour']), _r(s['deg'], 5), _r(s['move_mm'], 4),
                         _r(s['px'], 3), _r(s['px_rot'], 3), _r(s['px_parallax'], 3)])
    return buf.getvalue()


def vibration_csv(res: ThermoVibResult) -> str:
    buf = io.StringIO()
    wr = csv.writer(buf, lineterminator='\n')
    wr.writerow(['device', 'kind', 'on', 'f1_hz', 'resonance', 'accel_rms_m_s2', 'flex_mm_rms', 'flex_deg_rms',
                 'angle_deg_rms', 'blur_px_rms', 'cobble_flex_deg_rms', 'bump_peak_flex_deg', 'joint_peak_flex_deg',
                 'mount_ring_down_s'])
    for d in res.devices:
        ev = d.get('events') or {}
        wr.writerow([d['tag'], d['kind'], d['on'], _r(d['f1_hz'], 2),
                     '; '.join(f'{r["mode"]}: {", ".join(r["why"])}' for r in d['resonance']),
                     _r(d['vib']['accel_rms'], 4), _r(d['vib']['flex_mm_rms'], 5), _r(d['vib']['flex_deg_rms'], 5),
                     _r(d['vib']['angle_deg_rms'], 5), _r(d['vib']['blur_px'], 4), _r(d['cobble']['flex_deg_rms'], 5),
                     _r(ev['bump']['peak_flex_deg'], 5) if ev.get('bump') else '',
                     _r(ev['joint']['peak_flex_deg'], 5) if ev.get('joint') else '',
                     _r(ev['bump']['mount_ring_down_s'], 3) if ev.get('bump') else ''])
    return buf.getvalue()


def psd_csv(res: ThermoVibResult) -> str:
    """The full-resolution PSDs: road height and each device's base acceleration."""
    buf = io.StringIO()
    wr = csv.writer(buf, lineterminator='\n')
    tags = [d['tag'] for d in res.devices]
    wr.writerow(['f_hz', 'road_m2_hz'] + [f'{t}_accel_m2_s4_hz' for t in tags])
    acc = [res.psd['devices'][d['uid']]['accel'] for d in res.devices]
    for i, fr in enumerate(res.psd['f']):
        wr.writerow([f'{fr:.5g}', f'{res.psd["road"][i]:.5g}'] + [f'{a[i]:.5g}' for a in acc])
    return buf.getvalue()


def png(res: ThermoVibResult, size=(1300, 560)) -> bytes:
    """Two panels: each device's thermal turn (deg) over the day, and the road and mount acceleration PSDs."""
    import cv2
    W, H = size
    img = np.full((H, W, 3), 250, np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    pal = [(214, 120, 42), (38, 89, 217), (112, 158, 25), (0, 133, 201), (129, 81, 213), (0, 131, 0), (233, 133, 144),
           (103, 103, 230)]
    # left: drift
    x0, y0, x1, y1 = 60, 50, W // 2 - 30, H - 50
    hrs = [h['hour'] for h in res.hours]
    vals = [[s['turn_deg'] for s in d['thermal']] for d in res.devices]
    top = max([max(v) for v in vals if v] + [res.tol['deg'] * 0.05, 1e-6])
    cv2.rectangle(img, (x0, y0), (x1, y1), (200, 200, 200), 1)
    if len(hrs) > 1:
        def P(h, v):
            return (int(x0 + (h - hrs[0]) / (hrs[-1] - hrs[0]) * (x1 - x0)), int(y1 - v / top * (y1 - y0)))
        for k, v in enumerate(vals):
            col = pal[k % len(pal)]
            for i in range(len(v) - 1):
                cv2.line(img, P(hrs[i], v[i]), P(hrs[i + 1], v[i + 1]), col, 2, cv2.LINE_AA)
            cv2.putText(img, res.devices[k]['tag'], (P(hrs[-1], v[-1])[0] + 4, P(hrs[-1], v[-1])[1]), font, 0.4, col,
                        1, cv2.LINE_AA)
        for h in (hrs[0], hrs[-1]):
            cv2.putText(img, _hh(h), (P(h, 0)[0] - 18, y1 + 18), font, 0.45, (90, 90, 90), 1, cv2.LINE_AA)
    cv2.putText(img, f'Thermal turn from calibration, deg (top {top:.3g}) - {res.day["date"]}, heading '
                     f'{res.heading_deg:.0f}', (x0, 30), font, 0.5, (40, 40, 40), 1, cv2.LINE_AA)
    # right: PSDs (log-log)
    x0, x1 = W // 2 + 40, W - 30
    f = res.psd['f']
    series = [('road', res.psd['road'], (60, 60, 60))]
    for k, d in enumerate(res.devices):
        series.append((d['tag'], res.psd['devices'][d['uid']]['accel'], pal[k % len(pal)]))
    pos = np.concatenate([s[1][s[1] > 0] for s in series if (s[1] > 0).any()]) if series else np.array([1.0])
    lo_y, hi_y = math.log10(max(pos.min(), pos.max() * 1e-9)), math.log10(pos.max())
    cv2.rectangle(img, (x0, y0), (x1, y1), (200, 200, 200), 1)

    def Q(fr, val):
        return (int(x0 + (math.log10(fr) - math.log10(F_LO)) / (math.log10(F_HI) - math.log10(F_LO)) * (x1 - x0)),
                int(y1 - (math.log10(val) - lo_y) / max(hi_y - lo_y, 1e-9) * (y1 - y0)))
    for lo, hi in (BODY_BAND, HOP_BAND):
        cv2.rectangle(img, (Q(lo, 10 ** hi_y)[0], y0), (Q(hi, 10 ** hi_y)[0], y1), (235, 235, 235), -1)
    for name, y, col in series:
        idx = np.nonzero(y > 0)[0][::3]
        for a, b in zip(idx[:-1], idx[1:]):
            if b - a <= 3:
                cv2.line(img, Q(f[a], y[a]), Q(f[b], y[b]), col, 1, cv2.LINE_AA)
    for fr in (1, 10, 100):
        cv2.putText(img, f'{fr} Hz', (Q(fr, 10 ** lo_y)[0] - 14, y1 + 18), font, 0.45, (90, 90, 90), 1, cv2.LINE_AA)
    cv2.putText(img, f'Road (grey, m2/Hz) and mount acceleration PSD ((m/s2)2/Hz), class {res.settings.road_class} at '
                     f'{res.settings.speed_kmh:g} km/h', (x0 - 20, 30), font, 0.5, (40, 40, 40), 1, cv2.LINE_AA)
    ok, buf = cv2.imencode('.png', img)
    return buf.tobytes() if ok else b''


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


def files(res: ThermoVibResult, project=None) -> list:
    """[(name, bytes)]: thermo_vib.json, the hour-by-hour drift and pair CSVs, the vibration table, the PSDs and a
    PNG of the drift and the PSDs."""
    return [('thermo_vib.json', (json.dumps(to_doc(res, project), indent=1, default=_json_default) + '\n').encode()),
            ('thermo_vib_thermal.csv', thermal_csv(res).encode('utf-8')),
            ('thermo_vib_pairs.csv', pairs_csv(res).encode('utf-8')),
            ('thermo_vib_vibration.csv', vibration_csv(res).encode('utf-8')),
            ('thermo_vib_psd.csv', psd_csv(res).encode('utf-8')),
            ('thermo_vib.png', png(res))]


def export(res: ThermoVibResult, project, out_dir) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for name, data in files(res, project):
        path = os.path.join(out_dir, name)
        with open(path, 'wb') as fh:
            fh.write(data)
        paths.append(path)
    return paths
