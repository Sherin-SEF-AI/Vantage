"""Active cleaning and de-icing: washer nozzles, air nozzles and lens heaters placed on the rig's cameras and lidars;
whether their spray covers each window, how long a clean takes, how often the dirt the exposure study predicts makes
one necessary, how long the washer tank lasts over a shift, whether a heater clears ice at speed in the cold, and what
it all adds to the power budget and the harness.

Devices and their figures
-------------------------
A device is placed on one camera or lidar (its target) and is one of: a washer nozzle (liquid), an air nozzle
(compressed air) or a heater. Its figures come from a product of the library (every figure with its source: the
maker's public datasheet, the maker's published measurement fitted here, or a typical value marked as assumed) or are
given by the user (marked so). Library (read 2026-11-11):
* ARaymond telescope washer nozzles (Automotive Cleaning System brochure, p. 15): 900 ml/min at 2.5 bar (35 mm stroke,
  1–6 bar) and 750 ml/min at 2 bar (10 mm stroke, 2–8 bar). Spray angle: adjustable, not stated (assumed 40°).
* HELLA washer pumps (Washer pumps and water tanks 2026, pp. 2–3): 8TW 004 223 / 005 206, 12 V or 24 V, 1 l/min at 2
  bar (or at 1.5 bar); the headlamp-cleaning pump 007540141 / 010720227, 7.2 l/min at 2.5 bar. Their power is not
  stated (pump_w is assumed, 40 W, until you enter yours). Tank: 1.5 l (p. 7).
* Solero SCV32 3/2-way sensor-cleaning valve (Sensor Cleaning brochure 2026, p. 9): 12 V nominal, 8.9 Ω (1.35 A,
  16.2 W while open), 1.5 l/min at Δp 0.5 bar. SPV22 2/2 valve (p. 13): 210 Nl/min of air at 4 bar, at most 0.5 A.
* Lens heaters: e-con Systems, "What makes automatic de-icing and defogging cameras crucial" (26 March 2026), tests 2–5
  on a GMSL camera: 2 W at 8.13 V (0.254 A) and 5.5 W at 12.4 V (0.44 A); from −30 °C, 5.5 W brought the lens to
  20 °C in 5 min, 4.5 W in 8 min, 3.5 W and 2 W never. LG Innotek's heating camera module (New Equipment Digest, 18
  December 2024): at most 4 W; a frozen lens at −18 °C restored in 4 minutes.

Spray geometry (exact)
---------------------
The window: a camera's lens front, a disc at the optical centre square to the optical axis, of the drawn lens glass's
diameter (0.54 × the module's smaller side, as the 3D view draws it) unless the device gives window_mm; a spinning
lidar's cylindrical window band (its housing radius, between where its lowest and highest beams leave it); a
fixed-field lidar's front face. Each is sampled at equal-area points with their normals. A nozzle sits beside the
window (side top / bottom / left / right in the window's own frame, as the image is the right way up; for a round
lidar band an azimuth), standoff_mm beyond its edge and ahead_mm in front of it, aimed at the window's centre (moved
aim_mm towards the far side). A point is covered when the ray from the nozzle to it lies within the spray cone (half
the spray angle about the aim), the nozzle is in front of the surface there, and it lies within reach_mm. The covered
share of the window is that exactly (to the sampling); the share of the spray that lands on the window is the solid
angle the covered points subtend from the nozzle (Σ dA·cos θ / d²) over the cone's 2π(1 − cos α).

Clearing, duty cycle and fluid (a stated simple model)
------------------------------------------------------
A dirt kind has a dose (ml of fluid per cm² of window to clear it) and an obscuration rate (share of the window per
hour at core.exposure's dirt and spray score 1, linear in the score). Both are ASSUMED figures (no published
accumulation rates or cleaning doses for automotive lenses were found): set them from a test drive or a bench test.
A clean takes dose × covered area / (flow × the share that lands); the window needs one each time its obscuration
reaches max_obscured: events per hour = rate × score / max_obscured. The nozzle's flow at the pump's pressure follows
the orifice law (Q ∝ √p) and no more than the pump gives (one nozzle sprays at a time, through its own valve). Fluid
per hour = Σ events × flow × max(event_s, the clean's time); the tank's runtime is its volume over that.

Air nozzles (a round free jet)
------------------------------
A sonic orifice of orifice_mm at supply_bar: mass flow C_d·A·p0·√(γ/RT0)·(2/(γ+1))^((γ+1)/(2(γ−1))) (C_d 0.8,
typical); its momentum flux J = ṁ·c* + (p* − p_a)·A. Downstream the jet is a self-similar round jet: centreline
velocity u_c = B·√(4J/(π ρ_a)) / x, Gaussian profile u = u_c·exp(−ln 2·(r/(S x))²), B = 5.8 and spreading rate
S = 0.094 (S. B. Pope, Turbulent Flows, Cambridge 2000, §5.1, Table 5.1, after Hussein, Capp & George 1994). A window
point is cleared of drops where the jet's speed there is at least shed_ms (assumed 10 m/s). The compressor's power:
isothermal compression of the mass flow to the supply pressure over an overall efficiency (assumed 0.5).

Heaters (lumped heat balance)
-----------------------------
The heated lens assembly is one lumped body: heat capacity C and still-air conductance G0 (to the surroundings, by
natural convection and radiation). For the e-con heater both are fitted to the maker's two measured warm-ups (5.5 W:
−30 → 20 °C in 300 s; 4.5 W: in 480 s) by T(t) = T_a + (P/G)(1 − e^(−tG/C)): G0 = 0.0767 W/K, C = 19.3 J/K, which
predicts the other two outcomes the maker reports (3.5 W and 2 W never reach 20 °C: their steady rises 45.6 and
26.1 K are below 50 K). On the road the exposed front (exposed_mm²) adds forced convection, h from the flat-plate
correlations at the film temperature (F. P. Incropera et al., Fundamentals of Heat and Mass Transfer: laminar
Nu = 0.664 Re^½ Pr^⅓, turbulent from the leading edge Nu = 0.037 Re^⅘ Pr^⅓, mixed (0.037 Re^⅘ − 871) Pr^⅓;
air properties from Sutherland's laws), over the length along the flow, at the vehicle's speed plus a headwind:
G = G0 + h·A_exposed (the default regime is turbulent from the leading edge: the flow around a vehicle's body is
disturbed, and it is the conservative case for sizing a heater). Steady lens temperature T_a + P/G; ice of thickness
δ on the window: time to reach 0 °C (the lumped warm-up) plus melting, ρ_ice·δ·A·(c_ice·(0 − T_a) + L_f) over the
power left at 0 °C, P − G·(0 − T_a) (ice: 917 kg/m³, 2.05 kJ/kg·K, 334 kJ/kg, typical). Never when that is ≤ 0.

Power and harness. Average and peak power of the pump, valves, compressor and heaters go to the power budget as extra
loads (power_loads, in core.power's 'other' rows form); washer tubing and heater, valve and air wiring runs to the
harness (harness_runs: the Manhattan route × the ECU's routing slack + its service loop, the routed length of the
target's own cable when routing is on).

Limits: the spray is a cone (no droplet ballistics, wind or the vehicle's airflow deflecting it); doses and rates
are assumptions; the heater model is one lumped body with the forced convection on its exposed front; the air jet
is a free jet (no wall effects); the washer fluid's freezing point and the tubing's own heating are not modelled.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import math
import os
import threading
from collections import OrderedDict
from dataclasses import asdict, dataclass, field

import numpy as np

SCHEMA = 'vantage.cleaning/1'
FRAME = 'vehicle (ISO 8855, rear axle): x forward, y left, z up'
KINDS = ('washer', 'air', 'heater')
KIND_LABEL = {'washer': 'Washer nozzle', 'air': 'Air nozzle', 'heater': 'Lens heater'}
SIDES = ('top', 'bottom', 'left', 'right')
REGIMES = ('turbulent', 'mixed', 'laminar')
SEVERITIES = ('critical', 'serious', 'warning', 'good')
READ = '2026-11-11'
ASSUMPTIONS = ('Spray cones are geometric (no droplet ballistics, wind or airflow); dirt obscuration rates and '
               'cleaning doses are assumed figures (set yours from a test); the air jet is a free round jet (Pope); '
               'the heater is one lumped body fitted to e-con Systems\' published warm-ups with flat-plate forced '
               'convection on its exposed front; ice and air properties are typical; pump power is assumed where the '
               'maker states none.')

_AR = 'ARaymond, Automotive Cleaning System brochure, p. 15 (telescope nozzle)'
_HE = 'HELLA, Washer pumps and water tanks 2026 (BI)'
_SO = 'Solero Technologies, Sensor Cleaning brochure (2026)'
_EC = 'e-con Systems, "What makes Automatic De-icing and Defogging Cameras Crucial in Dynamic Environments", 26 Mar 2026'
_LG = 'LG Innotek, high-performance heating camera module, New Equipment Digest, 18 Dec 2024'
# e-con's warm-ups fitted by the lumped model (see the docstring and fit_lumped)
ECON_TESTS = ((5.5, 300.0), (4.5, 480.0))      # (W, s from −30 °C to 20 °C)
ECON_NOT_REACHED = (3.5, 2.0)
ECON_G0, ECON_C = 0.0767, 19.3

# products: figures and the source of each ('datasheet' | 'measured' | 'fitted' | 'assumed')
PRODUCTS = {
    'araymond_tele35': {'kind': 'washer', 'maker': 'ARaymond', 'name': 'Telescope nozzle, 35 mm stroke',
                        'flow_ml_min': 900.0, 'ref_bar': 2.5, 'spray_deg': 40.0,
                        'sources': {'flow_ml_min': 'datasheet', 'ref_bar': 'datasheet', 'spray_deg': 'assumed'},
                        'cite': _AR + ': 900 ml/min at 2.5 bar, 1–6 bar, adjustable spray angle (not stated).'},
    'araymond_tele10': {'kind': 'washer', 'maker': 'ARaymond', 'name': 'Telescope nozzle, 10 mm stroke',
                        'flow_ml_min': 750.0, 'ref_bar': 2.0, 'spray_deg': 40.0,
                        'sources': {'flow_ml_min': 'datasheet', 'ref_bar': 'datasheet', 'spray_deg': 'assumed'},
                        'cite': _AR + ': 750 ml/min at 2 bar, 2–8 bar, integrated non-return valve.'},
    'custom_washer': {'kind': 'washer', 'maker': '', 'name': 'Washer nozzle (your figures)', 'flow_ml_min': 500.0,
                      'ref_bar': 2.0, 'spray_deg': 40.0,
                      'sources': {'flow_ml_min': 'user', 'ref_bar': 'user', 'spray_deg': 'user'},
                      'cite': 'Given by the user.'},
    'air_spv22': {'kind': 'air', 'maker': 'Solero', 'name': 'Air nozzle on an SPV22 valve', 'orifice_mm': 1.0,
                  'supply_bar': 4.0, 'valve_w': 6.0,
                  'sources': {'orifice_mm': 'assumed', 'supply_bar': 'datasheet', 'valve_w': 'datasheet'},
                  'cite': _SO + ', SPV22 p. 13: 210 Nl/min of air at 4 bar, at most 0.5 A (12 V × 0.5 A); the nozzle '
                                'orifice is assumed.'},
    'custom_air': {'kind': 'air', 'maker': '', 'name': 'Air nozzle (your figures)', 'orifice_mm': 1.0,
                   'supply_bar': 3.0, 'valve_w': 6.0,
                   'sources': {'orifice_mm': 'user', 'supply_bar': 'user', 'valve_w': 'user'}, 'cite': 'Given by the '
                                                                                                       'user.'},
    'econ_5w5': {'kind': 'heater', 'maker': 'e-con Systems', 'name': 'Lens heater, 5.5 W', 'power_w': 5.5,
                 'g0_w_k': ECON_G0, 'c_j_k': ECON_C,
                 'sources': {'power_w': 'measured', 'g0_w_k': 'fitted', 'c_j_k': 'fitted'},
                 'cite': _EC + ': 5.5 W at 12.4 V, 0.44 A; −30 → 20 °C in 5 min (G0 and C fitted here).'},
    'econ_4w5': {'kind': 'heater', 'maker': 'e-con Systems', 'name': 'Lens heater, 4.5 W', 'power_w': 4.5,
                 'g0_w_k': ECON_G0, 'c_j_k': ECON_C,
                 'sources': {'power_w': 'measured', 'g0_w_k': 'fitted', 'c_j_k': 'fitted'},
                 'cite': _EC + ': 4.5 W; −30 → 20 °C in 8 min.'},
    'lg_innotek_4w': {'kind': 'heater', 'maker': 'LG Innotek', 'name': 'Heating camera module, 4 W', 'power_w': 4.0,
                      'g0_w_k': ECON_G0, 'c_j_k': ECON_C,
                      'sources': {'power_w': 'datasheet', 'g0_w_k': 'assumed', 'c_j_k': 'assumed'},
                      'cite': _LG + ': at most 4 W; frozen lens at −18 °C restored in 4 min. G0 and C assumed as '
                                    'e-con\'s fit.'},
    'custom_heater': {'kind': 'heater', 'maker': '', 'name': 'Heater (your figures)', 'power_w': 5.0,
                      'g0_w_k': ECON_G0, 'c_j_k': ECON_C,
                      'sources': {'power_w': 'user', 'g0_w_k': 'user', 'c_j_k': 'user'}, 'cite': 'Given by the user.'},
}
PRODUCT_IDS = tuple(PRODUCTS)
PUMPS = {
    'hella_1l': {'name': 'HELLA washer pump 8TW 004 223 / 005 206', 'flow_ml_min': 1000.0, 'bar': 2.0,
                 'cite': _HE + ', pp. 2–3: 12 V or 24 V, 2 bar 1 l/min or 1.5 bar 1 l/min. Power not stated.'},
    'hella_hcs': {'name': 'HELLA headlamp-cleaning pump 007540141', 'flow_ml_min': 7200.0, 'bar': 2.5,
                  'cite': _HE + ', p. 5: 12 V, 2.5 bar, 7.2 l/min. Power not stated.'},
    'custom': {'name': 'Pump (your figures)', 'flow_ml_min': 1000.0, 'bar': 2.0, 'cite': 'Given by the user.'},
}
VALVE = {'name': 'Solero SCV32 3/2-way valve', 'w': 12.0 ** 2 / 8.9,
         'cite': _SO + ', SCV32 p. 9: 12 V nominal, 8.9 Ω, 1.5 l/min at Δp 0.5 bar.'}
TANK_CITE = _HE + ', p. 7: 1.5 l washer tank.'

# dirt kinds: (label, obscuration per hour at exposure score 1, dose ml/cm²) — ASSUMED figures
DIRT = {'dust': ('Dry road dust', 0.02, 0.10), 'spray': ('Wet-road spray film', 0.30, 0.15),
        'mud': ('Mud and slush', 0.60, 0.40), 'salt': ('Dried road salt', 0.10, 0.30),
        'insects': ('Insects', 0.05, 0.60), 'rain': ('Rain drops', 0.80, 0.05)}
DIRT_IDS = tuple(DIRT)

# physics constants
R_AIR = 287.05                  # J/(kg·K)
GAMMA = 1.4
P_ATM = 101325.0
CP_AIR = 1006.0                 # J/(kg·K)
SIGMA = 5.670374419e-8
RHO_ICE, C_ICE, L_FUSION = 917.0, 2050.0, 334000.0      # typical
RHO_NORMAL = 1.2922             # kg/m³ at DIN 1343 normal conditions (0 °C, 1.01325 bar)
JET_B, JET_S = 5.8, 0.094       # Pope (2000) Table 5.1
CD_ORIFICE = 0.8                # typical

LIMITS = {'pump_w': (0.0, 500.0), 'tank_l': (0.1, 50.0), 'shift_h': (0.5, 24.0), 'ambient_c': (-40.0, 40.0),
          'speed_kmh': (0.0, 250.0), 'wind_ms': (0.0, 40.0), 'ice_mm': (0.0, 20.0), 'event_s': (0.1, 30.0),
          'max_obscured': (0.005, 0.5), 'heater_on_c': (-40.0, 40.0), 'shed_ms': (1.0, 100.0),
          'compressor_eff': (0.05, 1.0), 'air_event_s': (0.1, 60.0), 'min_cover': (0.0, 1.0)}
DEVICE_LIMITS = {'standoff_mm': (0.0, 500.0), 'ahead_mm': (-50.0, 500.0), 'aim_mm': (-200.0, 200.0),
                 'azimuth_deg': (-360.0, 360.0), 'window_mm': (0.0, 300.0), 'reach_mm': (5.0, 2000.0),
                 'flow_ml_min': (1.0, 20000.0), 'ref_bar': (0.1, 20.0), 'spray_deg': (1.0, 170.0),
                 'orifice_mm': (0.1, 10.0), 'supply_bar': (0.2, 20.0), 'valve_w': (0.0, 100.0),
                 'power_w': (0.0, 200.0), 'g0_w_k': (0.001, 10.0), 'c_j_k': (0.1, 10000.0),
                 'exposed_mm2': (0.0, 1e5), 'length_mm': (0.0, 1000.0)}
OVERRIDES = ('flow_ml_min', 'ref_bar', 'spray_deg', 'orifice_mm', 'supply_bar', 'valve_w', 'power_w', 'g0_w_k',
             'c_j_k')
MAX_DEVICES = 64
LIDAR_STANDOFF_MM = 80.0         # propose_layout: nozzles around a round lidar stand this far off
N_DISC = 1600                   # window sample points


def _num(v):
    if isinstance(v, bool):
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _clamp(v, lim):
    return min(max(v, lim[0]), lim[1])


# =========================================================================== settings
def clean_device(d, i=0) -> dict | None:
    """A device as stored: {'id', 'kind', 'product', 'target' (camera or sensor uid), 'side', 'standoff_mm',
    'ahead_mm', 'aim_mm', 'azimuth_deg', 'window_mm', 'reach_mm', 'exposed_mm2', 'length_mm', 'enabled', 'params':
    {figure: value given by the user}}; None when it is not usable."""
    if not isinstance(d, dict):
        return None
    prod = d.get('product') if d.get('product') in PRODUCTS else None
    kind = d.get('kind') if d.get('kind') in KINDS else (PRODUCTS[prod]['kind'] if prod else None)
    if kind is None:
        return None
    if prod is None or PRODUCTS[prod]['kind'] != kind:
        prod = f'custom_{kind}'
    target = d.get('target')
    if not isinstance(target, str) or not target:
        return None
    did = str(d.get('id') or f'C{i + 1}')[:12]
    out = {'id': did, 'kind': kind, 'product': prod, 'target': target[:40],
           'side': d.get('side') if d.get('side') in SIDES else 'top',
           'standoff_mm': 15.0, 'ahead_mm': 10.0, 'aim_mm': 0.0, 'azimuth_deg': 0.0, 'window_mm': 0.0,
           'reach_mm': 400.0, 'exposed_mm2': 0.0, 'length_mm': 0.0,
           'enabled': d.get('enabled') is not False, 'params': {}}
    if kind == 'air':
        out['standoff_mm'], out['ahead_mm'] = 10.0, 25.0
    for k in ('standoff_mm', 'ahead_mm', 'aim_mm', 'azimuth_deg', 'window_mm', 'reach_mm', 'exposed_mm2',
              'length_mm'):
        v = _num(d.get(k))
        if v is not None:
            out[k] = _clamp(v, DEVICE_LIMITS[k])
    prm = d.get('params')
    if isinstance(prm, dict):
        for k in OVERRIDES:
            v = _num(prm.get(k))
            if v is not None and k in PRODUCTS[prod]:
                out['params'][k] = _clamp(v, DEVICE_LIMITS[k])
    return out


@dataclass
class CleaningSettings:
    devices: list = field(default_factory=list)
    pump: str = 'hella_1l'
    pump_w: float = 40.0            # the pump's power while running (assumed: HELLA states none)
    pump_flow_ml_min: float = 0.0   # 'custom' pump: 0 = the library's
    tank_l: float = 1.5             # HELLA's 1.5 l tank
    reservoir: list = field(default_factory=list)   # [x, y, z] of the tank and pump; [] = behind the front bumper
    shift_h: float = 8.0
    dirt: str = 'spray'             # the dirt kind for the duty cycle and the tank
    ambient_c: float = -10.0        # for the heaters
    speed_kmh: float = 80.0
    wind_ms: float = 0.0            # headwind added to the speed
    ice_mm: float = 1.0
    regime: str = 'turbulent'
    event_s: float = 1.0            # shortest spray per cleaning event
    max_obscured: float = 0.05      # a window is cleaned when this share of it is obscured
    heater_on_c: float = 5.0        # heaters run at and below this ambient temperature
    shed_ms: float = 10.0           # air speed that sheds drops from a window (assumed)
    compressor_eff: float = 0.5     # overall (assumed)
    air_event_s: float = 2.0        # an air blast per cleaning event
    min_cover: float = 0.9          # a window covered less than this is a finding
    rates: dict = field(default_factory=dict)       # {dirt id: obscuration per hour at score 1} overriding DIRT
    doses: dict = field(default_factory=dict)       # {dirt id: ml/cm²}

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d) -> 'CleaningSettings':
        """Tolerant of junk: unknown keys ignored, numbers clamped, devices cleaned (ids made unique)."""
        s = CleaningSettings()
        d = d if isinstance(d, dict) else {}
        devs, ids = [], set()
        for i, raw in enumerate((d.get('devices') if isinstance(d.get('devices'), list) else [])[:MAX_DEVICES]):
            dv = clean_device(raw, i)
            if dv is None:
                continue
            if dv['id'] in ids:
                n = 1
                while f'C{n}' in ids:
                    n += 1
                dv['id'] = f'C{n}'
            ids.add(dv['id'])
            devs.append(dv)
        s.devices = devs
        if d.get('pump') in PUMPS:
            s.pump = d['pump']
        if d.get('dirt') in DIRT:
            s.dirt = d['dirt']
        if d.get('regime') in REGIMES:
            s.regime = d['regime']
        res = d.get('reservoir')
        if isinstance(res, (list, tuple)) and len(res) == 3 and all(_num(x) is not None for x in res):
            s.reservoir = [float(x) for x in res]
        for k, lim in LIMITS.items():
            if k in d:
                v = _num(d[k])
                if v is not None:
                    setattr(s, k, _clamp(v, lim))
        v = _num(d.get('pump_flow_ml_min'))
        if v is not None:
            s.pump_flow_ml_min = _clamp(v, (0.0, 50000.0))
        for key, lim in (('rates', (0.0, 100.0)), ('doses', (0.0, 10.0))):
            src = d.get(key)
            if isinstance(src, dict):
                setattr(s, key, {k: _clamp(_num(v), lim) for k, v in src.items() if k in DIRT and _num(v) is not None})
        return s

    def rate(self, dirt) -> float:
        return self.rates.get(dirt, DIRT[dirt][1])

    def dose(self, dirt) -> float:
        return self.doses.get(dirt, DIRT[dirt][2])

    def pump_flow(self) -> float:
        return self.pump_flow_ml_min if (self.pump == 'custom' and self.pump_flow_ml_min > 0) else \
            PUMPS[self.pump]['flow_ml_min']


def settings_of(project) -> CleaningSettings:
    return CleaningSettings.from_dict(getattr(getattr(project, 'analysis', None), 'cleaning', None))


def figure(dev: dict, key: str):
    """(value, source) of a device's figure: the user's override, else its product's with its source."""
    if key in dev.get('params', {}):
        return dev['params'][key], 'user'
    p = PRODUCTS[dev['product']]
    return p.get(key), p['sources'].get(key, 'datasheet')


# =========================================================================== windows
@dataclass
class Window:
    kind: str                   # 'disc' | 'rect' | 'band'
    centre: np.ndarray          # (3,) vehicle frame
    R: np.ndarray               # (3, 3) columns: u (right as the image is seen), v (up), n (out of the window)
    size: tuple                 # disc: (radius,); rect: (half width, half height); band: (radius, z0, z1)
    points: np.ndarray          # (N, 3)
    normals: np.ndarray         # (N, 3)
    dA: np.ndarray              # (N,) area per point, m²
    uv: np.ndarray              # (N, 2) in the window frame (band: azimuth rad, z), for describing where
    area: float

    def frame_point(self, a, b, c=0.0):
        return self.centre + self.R @ np.array([a, b, c], float)


def _disc_points(n=N_DISC):
    """Equal-area points over the unit disc (Vogel's sunflower spiral) and their area each."""
    k = np.arange(n) + 0.5
    r = np.sqrt(k / n)
    t = k * math.pi * (3.0 - math.sqrt(5.0))
    return np.c_[r * np.cos(t), r * np.sin(t)], math.pi / n


def lens_diameter_m(module) -> float:
    """The lens glass's diameter as the 3D view draws it (ui.overlays3d.camera_model: 0.27 × the smaller side)."""
    w, h, _ = (float(x) / 1000.0 for x in module.body_mm)
    return 0.54 * min(w, h)


def camera_window(project, cam, window_mm=0.0) -> Window:
    from .mathutil import rot_zyx
    Rb = rot_zyx(cam.yaw, cam.pitch, cam.roll)
    R = np.c_[-Rb[:, 1], Rb[:, 2], Rb[:, 0]]           # u = right, v = up, n = the optical axis
    rad = (window_mm / 2000.0) if window_mm > 0 else lens_diameter_m(project.module_for(cam)) / 2
    q, a = _disc_points()
    loc = np.c_[q * rad, np.zeros(len(q))]
    P = cam.pos() + loc @ R.T
    return Window('disc', cam.pos().astype(float), R, (rad,), P, np.repeat(R[:, 2][None], len(P), 0),
                  np.full(len(P), a * rad * rad), q * rad, math.pi * rad * rad)


def lidar_window(project, s, window_mm=0.0) -> Window:
    """A spinning lidar's window band (cylinder of the housing's radius between its lowest and highest beams at that
    radius, a few mm more), or a fixed-field lidar's front face."""
    from . import sensors as S
    m = project.sensor_model_for(s)
    R = S.rotation(s)
    dx, dy, h = (max(float(v), 1.0) / 1000.0 for v in m.size_mm)
    oz = m.optical_z_mm / 1000.0
    lo, hi = m.v_fov()
    c0 = np.array([s.x, s.y, s.z], float)
    if m.az_half() >= 180.0:
        r = dx / 2
        z0 = max(-(r * math.tan(math.radians(min(max(-lo, 0.0), 80.0))) + 0.003), -oz + 0.12 * h)
        z1 = min(r * math.tan(math.radians(min(max(hi, 0.0), 80.0))) + 0.003, h - oz - 0.14 * h)
        if z1 - z0 < 0.15 * h:
            z0, z1 = max(-oz + 0.1 * h, -0.2 * h), min(h - oz - 0.1 * h, 0.2 * h)
        na, nz = 160, 10
        az = (np.arange(na) + 0.5) / na * 2 * math.pi
        zz = z0 + (np.arange(nz) + 0.5) / nz * (z1 - z0)
        A, Z = np.meshgrid(az, zz)
        A, Z = A.ravel(), Z.ravel()
        nl = np.c_[np.cos(A), np.sin(A), np.zeros(len(A))]
        loc = np.c_[r * nl[:, :2], Z]
        P = c0 + loc @ R.T
        Nn = nl @ R.T
        dA = np.full(len(P), 2 * math.pi * r * (z1 - z0) / len(P))
        Rw = np.c_[-R[:, 1], R[:, 2], R[:, 0]]
        return Window('band', c0, Rw, (r, z0, z1), P, Nn, dA, np.c_[A, Z], 2 * math.pi * r * (z1 - z0))
    hw = dy / 2 * 0.8
    z0 = max(-oz + 0.1 * h, -(dx / 2) * math.tan(math.radians(min(max(-lo, 0.0), 80.0))) - 0.003)
    z1 = min(h - oz - 0.1 * h, (dx / 2) * math.tan(math.radians(min(max(hi, 0.0), 80.0))) + 0.003)
    hh = max((z1 - z0) / 2, 0.005)
    zc = (z0 + z1) / 2
    Rw = np.c_[-R[:, 1], R[:, 2], R[:, 0]]
    centre = c0 + R @ np.array([dx / 2, 0.0, zc])
    nu, nv = 48, 24
    U, V = np.meshgrid((np.arange(nu) + 0.5) / nu * 2 - 1, (np.arange(nv) + 0.5) / nv * 2 - 1)
    uv = np.c_[U.ravel() * hw, V.ravel() * hh]
    P = centre + np.c_[uv, np.zeros(len(uv))] @ Rw.T
    return Window('rect', centre, Rw, (hw, hh), P, np.repeat(Rw[:, 2][None], len(P), 0),
                  np.full(len(P), 4 * hw * hh / len(P)), uv, 4 * hw * hh)


def window_of(project, target_uid, window_mm=0.0):
    """(Window, object, kind 'camera' | 'lidar') of a target uid in the active variant, or None."""
    v = project.variant
    cam = next((c for c in v.cameras if c.uid == target_uid), None)
    if cam is not None:
        return camera_window(project, cam, window_mm), cam, 'camera'
    s = next((x for x in v.sensors if x.uid == target_uid), None)
    if s is not None and project.sensor_model_for(s).is_lidar:
        return lidar_window(project, s, window_mm), s, 'lidar'
    return None


def nozzle_pose(win: Window, dev: dict) -> tuple[np.ndarray, np.ndarray]:
    """(nozzle position, unit aim) in the vehicle frame for a device beside a window."""
    so, ah, aim = dev['standoff_mm'] / 1000.0, dev['ahead_mm'] / 1000.0, dev['aim_mm'] / 1000.0
    if win.kind == 'band':
        r, z0, z1 = win.size
        a = math.radians(dev['azimuth_deg'])
        Rl = np.c_[win.R[:, 2], -win.R[:, 0], win.R[:, 1]]      # back to the lidar frame's axes
        radial = Rl @ np.array([math.cos(a), math.sin(a), 0.0])
        up = Rl[:, 2]
        top = dev['side'] != 'bottom'
        zn = (z1 + ah) if top else (z0 - ah)
        pos = win.centre + radial * (r + so) + up * zn
        zt = (z0 + z1) / 2 + (-aim if top else aim)
        target = win.centre + radial * r + up * zt
    else:
        ext = win.size[0] if win.kind == 'disc' else (win.size[1] if dev['side'] in ('top', 'bottom')
                                                      else win.size[0])
        d = {'top': (0.0, 1.0), 'bottom': (0.0, -1.0), 'left': (-1.0, 0.0), 'right': (1.0, 0.0)}[dev['side']]
        pos = win.frame_point(d[0] * (ext + so), d[1] * (ext + so), ah)
        target = win.frame_point(-d[0] * aim, -d[1] * aim, 0.0)
    ax = target - pos
    n = np.linalg.norm(ax)
    return pos, (ax / n if n > 1e-12 else win.R[:, 2] * -1.0)


def cone_cover(apex, axis, half_deg, P, N=None, reach=math.inf) -> np.ndarray:
    """Points P (N, 3) inside the cone (apex, unit axis, half angle) within reach of the apex, and (normals N given)
    on the side of their surface facing the apex."""
    D = P - apex
    d = np.linalg.norm(D, axis=1)
    cosang = (D @ axis) / np.maximum(d, 1e-15)
    ok = (cosang >= math.cos(math.radians(half_deg))) & (d <= reach)
    if N is not None:
        ok &= np.einsum('ij,ij->i', N, -D) > 0
    return ok


def capture_share(apex, axis, half_deg, win: Window, covered) -> float:
    """Share of a uniform cone's solid angle that lands on the covered window points: Σ dA·cos θ / d² over
    2π(1 − cos α)."""
    if not covered.any():
        return 0.0
    D = win.points[covered] - apex
    d = np.linalg.norm(D, axis=1)
    cos_t = np.abs(np.einsum('ij,ij->i', win.normals[covered], D)) / np.maximum(d, 1e-15)
    omega = float((win.dA[covered] * cos_t / np.maximum(d, 1e-15) ** 2).sum())
    return min(omega / (2 * math.pi * (1 - math.cos(math.radians(half_deg)))), 1.0)


def where_missed(win: Window, covered) -> tuple[float, str]:
    """(share of the window not covered, where it is: 'top', 'bottom', 'left' or 'right' (the side its area-weighted
    centre lies towards), 'edge' (evenly round the rim) or 'all over'; a band: 'front', 'left', 'rear', 'right' by
    azimuth, 'between the nozzles' or 'all round')."""
    miss = ~covered
    share = float(win.dA[miss].sum() / max(win.dA.sum(), 1e-15))
    if share <= 0:
        return 0.0, ''
    if win.kind == 'band':
        a = (np.degrees(win.uv[miss, 0]) + 45.0) % 360.0
        names = ('front', 'left', 'rear', 'right')
        w = np.bincount((a // 90).astype(int) % 4, win.dA[miss], 4)
        if share > 0.6:
            return share, 'all round'
        if w.max() < 0.4 * w.sum():
            return share, 'between the nozzles'
        return share, names[int(np.argmax(w))]
    ext = max(float(np.abs(win.uv).max()), 1e-9)
    if share > 0.85:
        return share, 'all over'
    w = win.dA[miss]
    cu, cv = float((win.uv[miss, 0] * w).sum() / w.sum()), float((win.uv[miss, 1] * w).sum() / w.sum())
    if math.hypot(cu, cv) < 0.12 * ext:              # missed evenly round the rim
        return share, 'edge'
    if abs(cv) >= abs(cu):
        return share, 'top' if cv > 0 else 'bottom'
    return share, 'right' if cu > 0 else 'left'


# =========================================================================== physics
def air_props(T_k) -> dict:
    """Dry air at 1 atm: density (ideal gas), viscosity (Sutherland: 1.716e-5 Pa·s at 273.15 K, S = 110.4 K),
    conductivity (Sutherland form: 0.0241 W/m·K at 273.15 K, S = 194 K), Prandtl number with c_p 1006 J/kg·K."""
    T = float(T_k)
    mu = 1.716e-5 * (T / 273.15) ** 1.5 * (273.15 + 110.4) / (T + 110.4)
    k = 0.0241 * (T / 273.15) ** 1.5 * (273.15 + 194.0) / (T + 194.0)
    rho = P_ATM / (R_AIR * T)
    return {'rho': rho, 'mu': mu, 'nu': mu / rho, 'k': k, 'pr': mu * CP_AIR / k}


def flat_plate_h(U, L, k, nu, pr, regime='turbulent') -> float:
    """Average heat transfer coefficient (W/m²K) of a flat plate of length L in a parallel flow U (Incropera:
    laminar 0.664 Re^½ Pr^⅓; turbulent from the leading edge 0.037 Re^⅘ Pr^⅓; mixed with transition at Re 5e5)."""
    if U <= 0 or L <= 0:
        return 0.0
    Re = U * L / nu
    if regime == 'laminar' or (regime == 'mixed' and Re <= 5e5):
        nu_ = 0.664 * Re ** 0.5 * pr ** (1 / 3)
    elif regime == 'mixed':
        nu_ = (0.037 * Re ** 0.8 - 871.0) * pr ** (1 / 3)
    else:
        nu_ = 0.037 * Re ** 0.8 * pr ** (1 / 3)
    return nu_ * k / L


def fit_lumped(tests=ECON_TESTS, dT=50.0) -> tuple[float, float]:
    """(G W/K, C J/K) of a lumped body that warms by dT in each test's time at its power: T rise
    (P/G)(1 − exp(−tG/C)), from two tests (bisection on G)."""
    (p1, t1), (p2, t2) = tests

    def c_of(G, P, t):
        x = 1.0 - dT * G / P
        return -t * G / math.log(x)
    lo, hi = 1e-6, min(p1, p2) / dT * (1 - 1e-9)

    def f(G):
        return c_of(G, p2, t2) - c_of(G, p1, t1)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(lo) * f(mid) <= 0:
            hi = mid
        else:
            lo = mid
    G = 0.5 * (lo + hi)
    return G, c_of(G, p1, t1)


def heater_state(P, g0, C, exposed_m2, length_m, st: CleaningSettings, window_m2) -> dict:
    """A heater at ambient, speed and wind: conductance, steady lens temperature, time to clear ice (s, None when it
    cannot)."""
    Ta = st.ambient_c
    U = st.speed_kmh / 3.6 + st.wind_ms
    Tf = 273.15 + (Ta + max(Ta + P / max(g0, 1e-9), Ta)) / 2           # film: between ambient and the lens
    ap = air_props(min(max(Tf, 200.0), 400.0))
    h = flat_plate_h(U, length_m, ap['k'], ap['nu'], ap['pr'], st.regime)
    G = g0 + h * exposed_m2
    t_ss = Ta + P / G
    out = {'h_w_m2k': h, 'g_w_k': G, 'steady_c': t_ss, 'air_ms': U, 'clears': False, 'warm_s': None,
           'melt_s': None, 'clear_s': None}
    if Ta >= 0:
        out.update(clears=True, warm_s=0.0, melt_s=0.0, clear_s=0.0)
        return out
    need = G * (0.0 - Ta)
    if P <= need:
        return out
    warm = -(C / G) * math.log(1.0 - need / P)
    q_ice = RHO_ICE * st.ice_mm / 1000.0 * window_m2 * (C_ICE * (0.0 - Ta) + L_FUSION)
    melt = q_ice / (P - need)
    out.update(clears=True, warm_s=warm, melt_s=melt, clear_s=warm + melt)
    return out


def air_jet(orifice_mm, supply_bar, T0=288.15) -> dict:
    """A sonic orifice at supply_bar (gauge): mass flow, normal litres per minute, momentum flux and the jet's
    velocity scale √(4J/(πρ_a)) (m²/s: u_c = B × it / x)."""
    p0 = supply_bar * 1e5 + P_ATM
    A = math.pi * (orifice_mm / 1000.0) ** 2 / 4
    g = GAMMA
    crit = (2 / (g + 1)) ** (g / (g - 1))
    if P_ATM / p0 <= crit:                              # choked
        mdot = CD_ORIFICE * A * p0 * math.sqrt(g / (R_AIR * T0)) * (2 / (g + 1)) ** ((g + 1) / (2 * (g - 1)))
        Tstar = T0 * 2 / (g + 1)
        u = math.sqrt(g * R_AIR * Tstar)
        pe = p0 * crit
    else:                                              # subsonic: isentropic to ambient
        pe = P_ATM
        Te = T0 * (pe / p0) ** ((g - 1) / g)
        u = math.sqrt(2 * CP_AIR * (T0 - Te))
        mdot = CD_ORIFICE * A * (pe / (R_AIR * Te)) * u
    J = mdot * u + (pe - P_ATM) * A * CD_ORIFICE
    rho_a = P_ATM / (R_AIR * T0)
    return {'mdot_kg_s': mdot, 'nl_min': mdot / RHO_NORMAL * 60000.0, 'exit_ms': u, 'momentum_n': J,
            'scale_m2_s': math.sqrt(4 * J / (math.pi * rho_a)), 'p0_pa': p0}


def jet_speed(jet, apex, axis, P) -> np.ndarray:
    """Speed (m/s) of the round jet at points P: u_c(x) = B·scale/x, Gaussian across with half-width S·x."""
    D = P - apex
    x = D @ axis
    r = np.linalg.norm(D - x[:, None] * axis[None, :], axis=1)
    x = np.maximum(x, 1e-4)
    uc = JET_B * jet['scale_m2_s'] / x
    return np.where(D @ axis > 0, np.minimum(uc, jet['exit_ms']) * np.exp(-math.log(2) * (r / (JET_S * x)) ** 2),
                    0.0)


def compressor_w(jet, eff) -> float:
    """Isothermal compression power of the jet's mass flow to its supply pressure over an overall efficiency."""
    return jet['mdot_kg_s'] * R_AIR * 288.15 * math.log(jet['p0_pa'] / P_ATM) / max(eff, 1e-6)


# =========================================================================== the study
@dataclass
class CleaningResult:
    settings: CleaningSettings
    devices: list               # per device: dict (see run)
    targets: list               # per cleaned window: {'uid', 'tag', 'kind', 'cover', 'dirt_score', ...}
    totals: dict
    harness: list
    power_rows: list
    findings: list
    variant: str = ''
    variant_uid: str = ''
    key: str = ''
    windows: dict = field(default_factory=dict, repr=False)   # uid -> (Window, washed mask, air mask): not exported

    def device(self, did):
        return next((d for d in self.devices if d['id'] == did), None)

    def target(self, uid):
        return next((t for t in self.targets if t['uid'] == uid), None)

    def worst(self) -> str:
        return self.findings[0]['severity'] if self.findings else 'good'


def _project_for(project, variant):
    if variant is None or variant is project.variant:
        return project
    q = copy.copy(project)
    q.variants = list(project.variants) + ([] if any(v is variant for v in project.variants) else [variant])
    q.active_uid = variant.uid
    return q


def dirt_scores(project, engine=None) -> dict:
    """{camera or lidar uid: core.exposure's dirt and spray score} of the visible ones (0 when exposure fails)."""
    from . import exposure as EXP
    try:
        ex = EXP.run(project, engine)
    except Exception:                                            # noqa: BLE001
        return {}
    return {it.uid: float(it.dirt.get('score', 0.0)) for it in (ex.items if ex else []) if it.kind in
            ('camera', 'lidar')}


def _reservoir(project, st):
    if st.reservoir:
        return np.array(st.reservoir, float)
    mesh = project.vehicle.mesh
    if mesh is not None:
        lo, hi = mesh.vertices.min(0), mesh.vertices.max(0)
        return np.array([float(hi[0]) - 0.35, 0.45 * float(hi[1]), 0.45])
    return np.array([3.0, 0.4, 0.45])


def run(project, engine=None, settings: CleaningSettings | None = None, variant=None, progress=None, cancel=None,
        scores=None) -> CleaningResult | None:
    """The cleaning layout of a variant (the active one by default): per device its coverage, clearing, flow, duty
    and power; per target the union of its washers and air nozzles; the tank, power and harness totals. Cheap (no
    geometry beyond the windows). scores: {uid: dirt score} (None: from core.exposure)."""
    p = _project_for(project, variant)
    v = p.variant
    st = settings or settings_of(p)
    if scores is None:
        scores = dirt_scores(p, engine)
    pump = PUMPS[st.pump]
    pump_bar, pump_q = pump['bar'], st.pump_flow()
    devs, per_target = [], {}
    dirt_rate, dose = st.rate(st.dirt), st.dose(st.dirt)
    for i, dv in enumerate(st.devices):
        if cancel is not None and cancel():
            return None
        if progress is not None:
            progress(i / max(len(st.devices), 1))
        w = window_of(p, dv['target'], dv['window_mm'])
        row = {'id': dv['id'], 'kind': dv['kind'], 'product': dv['product'],
               'product_name': PRODUCTS[dv['product']]['name'], 'maker': PRODUCTS[dv['product']]['maker'],
               'cite': PRODUCTS[dv['product']]['cite'], 'enabled': dv['enabled'], 'target': dv['target'],
               'tag': None, 'figures': {}, 'problem': ''}
        for k in OVERRIDES:
            val, src = figure(dv, k)
            if val is not None:
                row['figures'][k] = {'value': val, 'source': src}
        if w is None:
            row['problem'] = 'Its camera or lidar is not in the variant (or is not a lidar).'
            devs.append(row)
            continue
        win, obj, tkind = w
        row['tag'] = obj.tag
        row['target_kind'] = tkind
        row['window'] = {'kind': win.kind, 'area_cm2': win.area * 1e4, 'size_mm': [x * 1000 for x in win.size]}
        tgt = per_target.setdefault(obj.uid, {'uid': obj.uid, 'tag': obj.tag, 'kind': tkind, 'win': win,
                                              'wash': np.zeros(len(win.points), bool),
                                              'air': np.zeros(len(win.points), bool), 'heaters': [],
                                              'dirt_score': scores.get(obj.uid, 0.0), 'washers': [], 'airs': []})
        if not dv['enabled'] or not getattr(obj, 'visible', True):
            row['problem'] = 'Switched off.' if not dv['enabled'] else 'Its target is hidden.'
            devs.append(row)
            continue
        if dv['kind'] in ('washer', 'air'):
            pos, axis = nozzle_pose(win, dv)
            row['nozzle'] = pos.tolist()
            row['aim'] = axis.tolist()
        if dv['kind'] == 'washer':
            q_ref, _ = figure(dv, 'flow_ml_min')
            p_ref, _ = figure(dv, 'ref_bar')
            spray, _ = figure(dv, 'spray_deg')
            q = min(q_ref * math.sqrt(pump_bar / max(p_ref, 1e-9)), pump_q)
            cov = cone_cover(pos, axis, spray / 2, win.points, win.normals, dv['reach_mm'] / 1000.0)
            cap = capture_share(pos, axis, spray / 2, win, cov)
            share, where = where_missed(win, cov)
            covered_cm2 = float(win.dA[cov].sum() * 1e4)
            t_clear = dose * covered_cm2 / (q / 60.0 * cap) if cap > 0 else math.inf
            tgt['wash'] |= cov
            tgt['washers'].append(dv['id'])
            row.update(cover=1 - share, missed=share, missed_where=where, capture=cap, flow_ml_min=q,
                       clear_s=t_clear, half_deg=spray / 2, reach_m=dv['reach_mm'] / 1000.0)
        elif dv['kind'] == 'air':
            ori, _ = figure(dv, 'orifice_mm')
            sup, _ = figure(dv, 'supply_bar')
            jet = air_jet(ori, sup)
            u = jet_speed(jet, pos, axis, win.points)
            facing = np.einsum('ij,ij->i', win.normals, pos[None] - win.points) > 0
            cov = (u >= st.shed_ms) & facing
            share, where = where_missed(win, cov)
            tgt['air'] |= cov
            tgt['airs'].append(dv['id'])
            half = math.degrees(math.atan(JET_S * math.sqrt(max(math.log(2) ** -1 * math.log(
                max(JET_B * jet['scale_m2_s'] / max(float(np.linalg.norm(win.centre - pos)), 1e-3) / st.shed_ms, 1.0)),
                0.0))))
            row.update(cover=1 - share, missed=share, missed_where=where, jet=jet, speed_at_centre_ms=float(
                jet_speed(jet, pos, axis, win.centre[None])[0]), compressor_w=compressor_w(jet, st.compressor_eff),
                half_deg=max(half, 1.0), reach_m=float(np.linalg.norm(win.centre - pos)) * 1.5)
        else:
            P, _ = figure(dv, 'power_w')
            g0, _ = figure(dv, 'g0_w_k')
            C, _ = figure(dv, 'c_j_k')
            if tkind == 'camera':
                mw, mh, ml = (float(x) / 1000.0 for x in p.module_for(obj).body_mm)
                exposed = dv['exposed_mm2'] / 1e6 if dv['exposed_mm2'] > 0 else mw * mh
                length = dv['length_mm'] / 1000.0 if dv['length_mm'] > 0 else min(mw, mh)
            else:
                exposed = dv['exposed_mm2'] / 1e6 if dv['exposed_mm2'] > 0 else win.area
                length = dv['length_mm'] / 1000.0 if dv['length_mm'] > 0 else max(win.size[0] * 2, 0.02)
            hs = heater_state(P, g0, C, exposed, length, st, win.area)
            tgt['heaters'].append(dv['id'])
            row.update(power_w=P, exposed_m2=exposed, length_m=length, **hs)
        devs.append(row)
    targets, windows = [], {}
    for tgt in per_target.values():
        windows[tgt['uid']] = (tgt['win'], tgt['wash'], tgt['air'])
        win = tgt['win']
        wa = float(win.dA[tgt['wash']].sum() / win.dA.sum()) if tgt['washers'] else None
        aa = float(win.dA[tgt['air']].sum() / win.dA.sum()) if tgt['airs'] else None
        share, where = where_missed(win, tgt['wash']) if tgt['washers'] else (1.0, '')
        events_h = dirt_rate * tgt['dirt_score'] / st.max_obscured if tgt['washers'] else 0.0
        clear_s = max((d['clear_s'] for d in devs if d['id'] in tgt['washers'] and d.get('clear_s') is not None),
                      default=0.0)
        targets.append({'uid': tgt['uid'], 'tag': tgt['tag'], 'kind': tgt['kind'], 'window': win.kind,
                        'area_cm2': win.area * 1e4, 'washers': tgt['washers'], 'airs': tgt['airs'],
                        'heaters': tgt['heaters'], 'wash_cover': wa, 'air_cover': aa, 'missed': share,
                        'missed_where': where, 'dirt_score': tgt['dirt_score'], 'events_per_h': events_h,
                        'interval_min': 60.0 / events_h if events_h > 0 else None, 'clear_s': clear_s})
    totals, prow = _totals(p, st, devs, targets)
    res = CleaningResult(st, devs, targets, totals, harness_runs(p, st, devs), prow, [], v.name, v.uid,
                         windows=windows)
    res.findings = _findings(p, res, scores)
    res.key = cache_key(project, st, v)
    _cache_put(res.key, res)
    if progress is not None:
        progress(1.0)
    return res


def _totals(p, st: CleaningSettings, devs, targets) -> tuple[dict, list]:
    """Fluid, tank runtime, average and peak power; and the power budget's extra rows."""
    by_id = {d['id']: d for d in devs}
    fluid_h, pump_s_h, valve_s_h, air_s_h, comp_w_avg, comp_peak = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    for t in targets:
        n = t['events_per_h']
        for did in t['washers']:
            d = by_id[did]
            if d.get('flow_ml_min') is None:
                continue
            dur = max(st.event_s, d['clear_s']) if math.isfinite(d['clear_s']) else st.event_s
            fluid_h += n * d['flow_ml_min'] / 60.0 * dur
            pump_s_h += n * dur
            valve_s_h += n * dur
        for did in t['airs']:
            d = by_id[did]
            air_s_h += n * st.air_event_s
            comp_w_avg += n * st.air_event_s / 3600.0 * d['compressor_w']
            comp_peak = max(comp_peak, d['compressor_w'])
    heaters = [d for d in devs if d['kind'] == 'heater' and d.get('power_w') is not None and not d['problem']]
    heat_on = st.ambient_c <= st.heater_on_c
    heat_w = sum(d['power_w'] for d in heaters) if heat_on else 0.0
    duty = min(pump_s_h / 3600.0, 1.0)
    pump_avg = st.pump_w * duty
    valve_avg = VALVE['w'] * min(valve_s_h / 3600.0, 1.0)
    air_valves = [by_id[i] for t in targets for i in t['airs']]
    air_valve_avg = sum(figure_val(d, 'valve_w') for d in air_valves) * min(air_s_h / 3600.0, 1.0) / \
        max(len(air_valves), 1) if air_valves else 0.0
    tank_ml = st.tank_l * 1000.0
    runtime_h = tank_ml / fluid_h if fluid_h > 0 else math.inf
    rows = []
    if any(d['kind'] == 'washer' and not d['problem'] for d in devs):
        rows.append(['Cleaning: washer pump', pump_avg, st.pump_w, 'other', False])
        rows.append(['Cleaning: washer valves', valve_avg, VALVE['w'], 'other', False])
    if air_valves:
        rows.append(['Cleaning: air compressor and valves', comp_w_avg + air_valve_avg,
                     comp_peak + max(figure_val(d, 'valve_w') for d in air_valves), 'other', False])
    if heaters:
        rows.append(['Cleaning: lens heaters', heat_w, sum(d['power_w'] for d in heaters), 'other', False])
    tot = {'fluid_ml_h': fluid_h, 'fluid_ml_shift': fluid_h * st.shift_h, 'tank_ml': tank_ml, 'runtime_h': runtime_h,
           'shift_h': st.shift_h, 'tank_ok': runtime_h >= st.shift_h, 'pump_duty': duty,
           'pump_w': st.pump_w, 'pump_source': 'assumed' if st.pump != 'custom' else 'user',
           'heaters_w': heat_w, 'heaters_on': heat_on, 'average_w': sum(r[1] for r in rows),
           'peak_w': sum(r[2] for r in rows), 'devices': len(devs),
           'washers': sum(d['kind'] == 'washer' for d in devs), 'airs': sum(d['kind'] == 'air' for d in devs),
           'heaters': len(heaters), 'dirt': st.dirt, 'dirt_label': DIRT[st.dirt][0],
           'rate_per_h': st.rate(st.dirt), 'dose_ml_cm2': st.dose(st.dirt),
           'pump': PUMPS[st.pump]['name'], 'pump_cite': PUMPS[st.pump]['cite'], 'valve_cite': VALVE['cite'],
           'tank_cite': TANK_CITE}
    return tot, rows


def figure_val(dev_row, key) -> float:
    f = dev_row.get('figures', {}).get(key)
    return float(f['value']) if f else 0.0


def clearing_table(res: CleaningResult) -> list:
    """Per washer and dirt kind: the time one clean takes and the cleaning events per hour (the duty)."""
    st = res.settings
    out = []
    by_t = {t['uid']: t for t in res.targets}
    for d in res.devices:
        if d['kind'] != 'washer' or d.get('capture') is None:
            continue
        t = by_t.get(d['target'], {})
        cov_cm2 = d['window']['area_cm2'] * d['cover']
        for dk in DIRT_IDS:
            ts = st.dose(dk) * cov_cm2 / (d['flow_ml_min'] / 60.0 * d['capture']) if d['capture'] > 0 else math.inf
            ev = st.rate(dk) * t.get('dirt_score', 0.0) / st.max_obscured
            out.append({'device': d['id'], 'tag': d['tag'], 'dirt': dk, 'label': DIRT[dk][0], 'clear_s': ts,
                        'events_per_h': ev, 'duty': min(ev * max(ts, st.event_s) / 3600.0, 1.0) if
                        math.isfinite(ts) else None})
    return out


def harness_runs(p, st: CleaningSettings, devs) -> list:
    """Tubing from the tank and pump to each washer and air nozzle, and wiring from the power distribution beside the
    compute box to each heater and valve: (the Manhattan route × the ECU's slack + its service loop; a heater's run
    follows its target's routed cable when routing is on)."""
    from . import harness as HN
    e = p.ecu
    rf = float(e.route_factor or 1.0)
    loop = float(e.service_loop_m or 0.0)
    ecu = np.array([e.x, e.y, e.z], float)
    res_pos = _reservoir(p, st)
    out = []
    objs = {o.uid: o for o in list(p.variant.cameras) + list(p.variant.sensors)}
    for d in devs:
        if d['problem']:
            continue
        o = objs.get(d['target'])
        if o is None:
            continue
        end = np.array(d.get('nozzle') or o.pos(), float)
        if d['kind'] in ('washer', 'air'):
            L = HN.manhattan(res_pos, end) * rf + loop
            out.append({'device': d['id'], 'what': 'tubing' if d['kind'] == 'washer' else 'air line',
                        'from': 'tank and pump' if d['kind'] == 'washer' else 'compressor', 'to': d['tag'],
                        'length_m': L, 'route': 'Manhattan estimate', 'path': [res_pos.tolist(), end.tolist()]})
        routed = None
        try:
            routed = HN.routed_length(p, o)
        except Exception:                                        # noqa: BLE001
            routed = None
        L = routed if routed is not None else HN.manhattan(ecu, o.pos()) * rf + loop
        if d['kind'] == 'heater':
            out.append({'device': d['id'], 'what': 'heater wiring', 'from': 'power distribution', 'to': d['tag'],
                        'length_m': float(L), 'route': 'routed with its cable' if routed is not None else
                        'Manhattan estimate', 'path': [ecu.tolist(), o.pos().tolist()]})
        else:
            out.append({'device': d['id'], 'what': 'valve wiring', 'from': 'power distribution', 'to': d['tag'],
                        'length_m': HN.manhattan(ecu, res_pos) * rf + loop, 'route': 'Manhattan estimate',
                        'path': [ecu.tolist(), res_pos.tolist()]})
    return out


def power_loads(project, res: CleaningResult | None = None) -> list:
    """The cleaning's extra loads for the power budget in core.power's 'other' rows form: [name, steady W (the
    average over the duty), peak W, circuit, in the enclosure]."""
    res = res or cached(project)
    if res is None:
        res = run(project)
    return [list(r) for r in res.power_rows]


def power_budget(project, res: CleaningResult | None = None):
    """core.power's budget with the cleaning's loads added as other loads (the project is not changed)."""
    from . import power as PW
    st = PW.PowerSettings.from_dict(getattr(project.analysis, 'power', None))
    d = st.to_dict()
    d['other'] = [r for r in d['other'] if not str(r[0]).startswith('Cleaning:')] + power_loads(project, res)
    return PW.run(project, None, PW.PowerSettings.from_dict(d))


def washer_cover(project) -> dict:
    """{camera or lidar uid: the share of its window its washers cover}: for core.sotif's lens-dirt condition."""
    res = cached(project)
    if res is None:
        if not settings_of(project).devices:
            return {}
        res = run(project, scores={})
    return {t['uid']: t['wash_cover'] for t in res.targets if t['wash_cover'] is not None}


def propose_layout(project, scores=None, min_score=0.25) -> list:
    """A starting layout: a washer (ARaymond 10 mm telescope nozzle) above every camera whose dirt score is at least
    min_score, four around every spinning lidar that does, one above a fixed lidar; an e-con 5.5 W heater on every
    camera."""
    scores = dirt_scores(project) if scores is None else scores
    v = project.variant
    out, n = [], 0

    def add(**kw):
        nonlocal n
        n += 1
        out.append(clean_device(dict(id=f'C{n}', **kw), n - 1))
    for c in v.cameras:
        if not c.visible:
            continue
        if scores.get(c.uid, 0.0) >= min_score:
            add(kind='washer', product='araymond_tele10', target=c.uid, side='top')
        add(kind='heater', product='econ_5w5', target=c.uid)
    for s in v.sensors:
        if not s.visible or not project.sensor_model_for(s).is_lidar or scores.get(s.uid, 0.0) < min_score:
            continue
        if project.sensor_model_for(s).az_half() >= 180.0:
            for az in (0.0, 90.0, 180.0, 270.0):
                add(kind='washer', product='araymond_tele10', target=s.uid, side='top', azimuth_deg=az,
                    standoff_mm=LIDAR_STANDOFF_MM, ahead_mm=15.0)
        else:
            add(kind='washer', product='araymond_tele10', target=s.uid, side='top')
    return out


def _findings(p, res: CleaningResult, scores) -> list:
    out = []
    st = res.settings

    def f(sev, kind, msg, tags=(), uids=()):
        out.append({'severity': sev, 'kind': kind, 'message': msg, 'tags': list(tags), 'uids': list(uids)})
    for d in res.devices:
        if d['problem'] and d['problem'] != 'Switched off.':
            f('warning', 'device', f'{d["id"]} ({KIND_LABEL[d["kind"]].lower()}): {d["problem"]}', [d['tag'] or ''])
            continue
        if d['problem']:
            continue
        if d['kind'] == 'washer' and not math.isfinite(d['clear_s']):
            f('serious', 'cover', f'{d["id"]} on {d["tag"]}: no spray lands on the window.', [d['tag']], [d['target']])
        if d['kind'] == 'heater' and st.ambient_c < 0 and not d['clears']:
            f('serious', 'heater', f'{d["tag"]}\'s heater ({d["id"]}, {d["power_w"]:g} W) holds the lens at '
                                   f'{d["steady_c"]:.1f} °C at {st.speed_kmh:g} km/h and {st.ambient_c:g} °C: it '
                                   'cannot clear ice.', [d['tag']], [d['target']])
    for tg in res.targets:                       # the union of a window's nozzles against it
        for key, ids, what in (('wash_cover', tg['washers'], 'washer cone'), ('air_cover', tg['airs'], 'air jet')):
            cov = tg[key]
            if cov is None or cov >= st.min_cover:
                continue
            where = tg['missed_where'] if key == 'wash_cover' else ''
            loc = {'top': 'its top', 'bottom': 'its bottom', 'left': 'its left side', 'right': 'its right side',
                   'edge': 'its edge', 'all over': 'nearly all of it', 'all round': 'all round',
                   'front': 'its front', 'rear': 'its rear', 'centre': 'its centre',
                   'between the nozzles': 'between the nozzles'}.get(where, where)
            what_ = what + ('s' if len(ids) > 1 else '')
            f('warning', 'cover', f'{tg["tag"]}\'s {what_} ({", ".join(ids)}) miss{"" if len(ids) > 1 else "es"} '
                                  f'{100 * (1 - cov):.0f} % of the {"lens" if tg["kind"] == "camera" else "window"}'
                                  + (f', mostly {loc}.' if loc else '.'), [tg['tag']], [tg['uid']])
    t = res.totals
    if t['washers'] and not t['tank_ok']:
        f('serious', 'tank', f'The {t["tank_ml"] / 1000:g} l tank lasts {t["runtime_h"]:.1f} h with '
                             f'{t["dirt_label"].lower()}: less than the {st.shift_h:g} h shift '
                             f'({t["fluid_ml_shift"] / 1000:.1f} l needed).')
    wiped = {}
    try:
        from . import weather as W
        wiped = W.wiped_shares(p, None)
    except Exception:                                            # noqa: BLE001
        pass
    cleaned = {t_['uid'] for t_ in res.targets if t_['washers'] or t_['airs']}
    for o in list(p.variant.cameras) + [s for s in p.variant.sensors if p.sensor_model_for(s).is_lidar]:
        if not getattr(o, 'visible', True):
            continue
        sc = scores.get(o.uid, 0.0)
        if sc >= 0.5 and o.uid not in cleaned:
            if o.uid in wiped:
                continue
            f('warning', 'uncleaned', f'{o.tag}: dirt and spray score {sc:.2f} (high) and no washer or air nozzle.',
              [o.tag], [o.uid])
    for o in p.variant.cameras:
        if o.uid in wiped and o.uid in cleaned:
            f('warning', 'glass', f'{o.tag} sits behind the windscreen: the vehicle\'s wipers and washers clear '
                                  f'{100 * (wiped[o.uid] or 0):.0f} % of its view; its own nozzle sprays the inside.',
              [o.tag], [o.uid])
    order = {s: i for i, s in enumerate(SEVERITIES)}
    out.sort(key=lambda x: order.get(x['severity'], 9))
    return out


# =========================================================================== cache and issues
_CACHE: OrderedDict = OrderedDict()
_LOCK = threading.Lock()


def cache_key(project, settings: CleaningSettings | None = None, variant=None) -> str:
    st = settings or settings_of(project)
    v = variant or project.variant
    doc = {'schema': SCHEMA, 'st': st.to_dict(), 'v': v.to_dict(),
           'mods': [project.module_for(c).to_dict() for c in v.cameras],
           'sens': [project.sensor_model_for(s).to_dict() for s in v.sensors], 'ecu': project.ecu.__dict__,
           'exposure': getattr(project.analysis, 'exposure', None), 'harness': getattr(project.analysis, 'harness',
                                                                                         None)}
    return hashlib.sha256(json.dumps(doc, sort_keys=True, default=str).encode()).hexdigest()


def _cache_put(key, res):
    with _LOCK:
        _CACHE.pop(key, None)
        _CACHE[key] = res
        while len(_CACHE) > 8:
            _CACHE.popitem(last=False)


def cached(project, settings=None, variant=None) -> CleaningResult | None:
    try:
        key = cache_key(project, settings, variant)
    except Exception:                                            # noqa: BLE001
        return None
    with _LOCK:
        return _CACHE.get(key)


def clear_cache():
    with _LOCK:
        _CACHE.clear()


def issues(project, vg=None) -> list:
    """From the cached result only (nothing until the page or the batch ran it): serious findings (a heater that
    cannot clear ice, a tank that does not last the shift, a nozzle that misses the window)."""
    from .rules import Issue
    res = cached(project)
    if res is None:
        return []
    return [Issue(f['severity'], 'Cleaning: ' + f['message'], ', '.join(t for t in f['tags'] if t) or 'Cleaning',
                  list(f['uids']), 'cleaning') for f in res.findings if f['severity'] in ('critical', 'serious')]


# =========================================================================== output
def _r(v, d=3):
    if v is None:
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return round(v, d) if math.isfinite(v) else None


def _clean(o, d=4):
    if isinstance(o, float):
        return _r(o, d)
    if isinstance(o, dict):
        return {k: _clean(v, d) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(x, d) for x in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist(), d)
    return o


def summary(res: CleaningResult) -> dict:
    t = res.totals
    return {'devices': len(res.devices), 'tank_runtime_h': _r(t['runtime_h'], 2), 'shift_h': t['shift_h'],
            'fluid_l_shift': _r(t['fluid_ml_shift'] / 1000.0, 3), 'average_w': _r(t['average_w'], 2),
            'peak_w': _r(t['peak_w'], 2), 'heaters_w': _r(t['heaters_w'], 2),
            'targets': [{'tag': x['tag'], 'wash_cover': _r(x['wash_cover'], 3), 'air_cover': _r(x['air_cover'], 3),
                         'dirt_score': _r(x['dirt_score'], 3), 'events_per_h': _r(x['events_per_h'], 2)}
                        for x in res.targets],
            'heaters': [{'id': d['id'], 'tag': d['tag'], 'steady_c': _r(d.get('steady_c'), 2),
                         'clear_min': _r((d.get('clear_s') or math.inf) / 60.0, 2)} for d in res.devices
                        if d['kind'] == 'heater' and not d['problem']],
            'findings': [{'severity': f['severity'], 'message': f['message']} for f in res.findings],
            'assumptions': ASSUMPTIONS}


def to_doc(res: CleaningResult, project=None) -> dict:
    devs = []
    for d in res.devices:
        x = {k: v for k, v in d.items() if k != 'jet'}
        if 'jet' in d:
            x['jet'] = {k: _r(v, 4) for k, v in d['jet'].items()}
        devs.append(_clean(x))
    return {'schema': SCHEMA, 'project': getattr(project, 'name', ''), 'variant': res.variant, 'frame': FRAME,
            'units': 'metres (positions), mm (device offsets), ml, l, W, °C, s, km/h; shares 0–1',
            'assumptions': ASSUMPTIONS, 'settings': res.settings.to_dict(),
            'dirt_kinds': {k: {'label': v[0], 'rate_per_h_at_score_1': res.settings.rate(k),
                               'dose_ml_cm2': res.settings.dose(k), 'source': 'assumed'} for k, v in DIRT.items()},
            'devices': devs, 'targets': _clean(res.targets), 'clearing': _clean(clearing_table(res)),
            'totals': _clean(res.totals), 'power_loads': _clean(res.power_rows), 'harness': _clean(res.harness),
            'findings': res.findings,
            'models': {'jet': f'Pope (2000) §5.1: B = {JET_B}, S = {JET_S}', 'heater': 'lumped body, e-con fit '
                                                                                      f'G0 {ECON_G0} W/K, C {ECON_C} '
                                                                                      'J/K',
                       'convection': 'flat plate (Incropera), Sutherland air properties'}}


CSV_COLUMNS = ('id', 'kind', 'target', 'product', 'cover', 'missed_where', 'flow_ml_min', 'capture', 'clear_s',
               'power_w', 'steady_c', 'ice_clear_s', 'problem')


def to_csv(res: CleaningResult) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(CSV_COLUMNS)
    for d in res.devices:
        g = lambda k: '' if d.get(k) is None or (isinstance(d.get(k), float) and not math.isfinite(d[k])) \
            else (f'{d[k]:.4g}' if isinstance(d[k], float) else d[k])           # noqa: E731
        w.writerow([d['id'], d['kind'], d['tag'] or d['target'], d['product_name'], g('cover'), g('missed_where'),
                    g('flow_ml_min'), g('capture'), g('clear_s'), g('power_w'), g('steady_c'), g('clear_s')
                    if d['kind'] == 'heater' else '', d['problem']])
    return buf.getvalue()


def png(res: CleaningResult, project, size=900) -> bytes:
    """Plan view: the body outline, each cleaned window and its nozzles with their cones, coloured by cover."""
    import cv2
    img = np.full((size, size, 3), 250, np.uint8)
    pts = [np.array(d['nozzle'])[:2] for d in res.devices if d.get('nozzle')]
    objs = {o.uid: o for o in list(project.variant.cameras) + list(project.variant.sensors)}
    pts += [objs[t['uid']].pos()[:2] for t in res.targets if t['uid'] in objs]
    mesh = project.vehicle.mesh
    if mesh is not None:
        lo, hi = mesh.vertices.min(0)[:2], mesh.vertices.max(0)[:2]
        pts += [lo, hi]
    if not pts:
        ok, buf = cv2.imencode('.png', img)
        return buf.tobytes()
    P = np.array(pts)
    c = (P.min(0) + P.max(0)) / 2
    span = max(float((P.max(0) - P.min(0)).max()) * 1.2, 1.0)
    k = size / span

    def px(xy):          # x forward up, y left left
        return int(size / 2 - (xy[1] - c[1]) * k), int(size / 2 - (xy[0] - c[0]) * k)
    if mesh is not None:
        cv2.rectangle(img, px((hi[0], hi[1])), px((lo[0], lo[1])), (180, 180, 180), 2)
    for d in res.devices:
        if not d.get('nozzle'):
            continue
        n = np.array(d['nozzle'])
        a = np.array(d['aim'])
        good = d['cover'] >= res.settings.min_cover
        col = (12, 163, 12) if good else (59, 59, 208)
        for s in (-1, 1):
            ang = math.radians(d['half_deg']) * s
            ca, sa = math.cos(ang), math.sin(ang)
            dxy = np.array([a[0] * ca - a[1] * sa, a[0] * sa + a[1] * ca])
            cv2.line(img, px(n[:2]), px(n[:2] + dxy * max(d['reach_m'] * 0.3, 0.05)), col, 1, cv2.LINE_AA)
        cv2.circle(img, px(n[:2]), 4, col, -1, cv2.LINE_AA)
    for t in res.targets:
        o = objs.get(t['uid'])
        if o is None:
            continue
        cv2.circle(img, px(o.pos()[:2]), 6, (40, 40, 40), 1, cv2.LINE_AA)
        cv2.putText(img, t['tag'], (px(o.pos()[:2])[0] + 8, px(o.pos()[:2])[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (40, 40, 40), 1, cv2.LINE_AA)
    ok, buf = cv2.imencode('.png', img)
    return buf.tobytes()


def files(res: CleaningResult, project=None) -> list:
    doc = to_doc(res, project)
    if project is not None:
        from . import provenance as PV
        doc = PV.stamp(doc, project)
    out = [('cleaning.json', (json.dumps(doc, indent=1, ensure_ascii=False) + '\n').encode('utf-8')),
           ('cleaning.csv', to_csv(res).encode('utf-8'))]
    if project is not None:
        out.append(('cleaning.png', png(res, project)))
    return out


def export(res: CleaningResult, project, out_dir) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for name, data in files(res, project):
        path = os.path.join(out_dir, name)
        with open(path, 'wb') as fh:
            fh.write(data)
        paths.append(path)
    return paths


# =========================================================================== validation
# METU ME 212 (F. Arınç, Spring 2013), forced convection over a flat plate, worked examples (Incropera's method):
# Example 1: air at 40 °C, 1 m/s along a 0.5 m plate at 100 °C, properties at 350 K: k 0.03003, ν 20.76e-6, Pr 0.697:
# h̄ = 5.48 W/m²K (laminar). Example 5: a truck roof, 105 km/h (29.2 m/s), L = 10 m, turbulent over the whole length,
# air at 300 K: k 0.0263, ν 15.89e-6, Pr 0.707: h = 56.2 W/m²K.
CONVECTION_EXAMPLES = (
    {'name': 'Example 1 (laminar)', 'U': 1.0, 'L': 0.5, 'k': 0.03003, 'nu': 20.76e-6, 'pr': 0.697, 'regime': 'laminar',
     'h': 5.48, 'T_film_k': 343.15},
    {'name': 'Example 5 (truck roof, turbulent)', 'U': 29.2, 'L': 10.0, 'k': 0.0263, 'nu': 15.89e-6, 'pr': 0.707,
     'regime': 'turbulent', 'h': 56.2, 'T_film_k': 300.0},
)


def ellipse_section(h, alpha_deg, beta_deg) -> tuple[float, float, float]:
    """A cone of half angle α whose apex is h above a plane, its axis tilted β from the plane's normal: the section
    is an ellipse (α + β < 90°): (semi-major a, semi-minor b, offset of its centre from the axis' foot along the
    tilt). a = h·sinα·cosα / (cos²β − sin²α), b = h·sinα / √(cos²β − sin²α)."""
    a_, b_ = math.radians(alpha_deg), math.radians(beta_deg)
    den = math.cos(b_) ** 2 - math.sin(a_) ** 2
    A = h * math.sin(a_) * math.cos(a_) / den
    B = h * math.sin(a_) / math.sqrt(den)
    x1, x2 = h * math.tan(b_ - a_), h * math.tan(b_ + a_)
    return A, B, (x1 + x2) / 2 - h * math.tan(b_)


def _circle_overlap(R, r, d) -> float:
    """Area of the intersection of two circles of radii R and r, centres d apart (closed form)."""
    if d >= R + r:
        return 0.0
    if d <= abs(R - r):
        return math.pi * min(R, r) ** 2
    a1 = r * r * math.acos((d * d + r * r - R * R) / (2 * d * r))
    a2 = R * R * math.acos((d * d + R * R - r * r) / (2 * d * R))
    a3 = 0.5 * math.sqrt((-d + r + R) * (d + r - R) * (d - r + R) * (d + r + R))
    return a1 + a2 - a3


def _plane_window(radius, n=40000) -> Window:
    q, a = _disc_points(n)
    P = np.c_[q * radius, np.zeros(len(q))]
    return Window('disc', np.zeros(3), np.eye(3), (radius,), P, np.tile([0.0, 0.0, 1.0], (len(P), 1)),
                  np.full(len(P), a * radius ** 2), q * radius, math.pi * radius ** 2)


def validation_checks() -> list:
    """Spec-compatible dicts for vantage.validation (model 'cleaning'): fn(ctx) returns a suite Outcome."""
    def _cone(ctx):
        from ..validation.suite import Outcome
        n = 20000 if ctx.quick else 80000
        win = _plane_window(0.02, n)
        rows, worst = {}, 0.0
        # axis square to the window, offset: a circle of radius h·tanα overlapping the disc
        for h, al, off in ((0.02, 20.0, 0.0), (0.03, 15.0, 0.012), (0.01, 40.0, 0.018)):
            apex = np.array([off, 0.0, h])
            cov = cone_cover(apex, np.array([0.0, 0.0, -1.0]), al, win.points, win.normals)
            got = float(win.dA[cov].sum())
            ref = _circle_overlap(0.02, h * math.tan(math.radians(al)), off)
            rows[f'square h={h} α={al} off={off}'] = {'model_mm2': round(got * 1e6, 4), 'closed_mm2': round(ref * 1e6, 4)}
            worst = max(worst, abs(got - ref) / (math.pi * 0.02 ** 2))
        # tilted axis: an ellipse wholly inside the disc
        for h, al, be in ((0.006, 15.0, 30.0), (0.004, 20.0, 45.0)):
            A, B, cx = ellipse_section(h, al, be)
            b_ = math.radians(be)
            apex = np.array([-h * math.tan(b_) - cx, 0.0, h])         # the ellipse centred on the disc
            axis = np.array([math.sin(b_), 0.0, -math.cos(b_)])
            cov = cone_cover(apex, axis, al, win.points, win.normals)
            got = float(win.dA[cov].sum())
            ref = math.pi * A * B
            rows[f'tilted h={h} α={al} β={be}'] = {'model_mm2': round(got * 1e6, 4), 'closed_mm2': round(ref * 1e6, 4),
                                                   'a_mm': round(A * 1e3, 3), 'b_mm': round(B * 1e3, 3)}
            worst = max(worst, abs(got - ref) / (math.pi * 0.02 ** 2))
        return Outcome(worst, {'rows': rows, 'note': 'error relative to the window\'s area'})

    def _conv(ctx):
        from ..validation.suite import Outcome
        rows, worst = {}, 0.0
        for ex in CONVECTION_EXAMPLES:
            h = flat_plate_h(ex['U'], ex['L'], ex['k'], ex['nu'], ex['pr'], ex['regime'])
            ap = air_props(ex['T_film_k'])
            h2 = flat_plate_h(ex['U'], ex['L'], ap['k'], ap['nu'], ap['pr'], ex['regime'])
            rows[ex['name']] = {'given_properties': round(h, 3), 'own_properties': round(h2, 3), 'published': ex['h']}
            worst = max(worst, abs(h - ex['h']) / ex['h'])
            rows[ex['name']]['own_properties_rel_err'] = round(abs(h2 - ex['h']) / ex['h'], 4)
        own = max(r['own_properties_rel_err'] for r in rows.values())
        return Outcome(worst, {'rows': rows}, ok=own <= 0.03,
                       note='ok also requires Sutherland\'s air properties within 3 % of the published h')

    def _heater(ctx):
        from ..validation.suite import Outcome
        G, C = fit_lumped()
        wrong = 0
        pred = {}
        for P_ in ECON_NOT_REACHED:
            reach = P_ / G > 50.0
            pred[f'{P_:g} W'] = {'steady_rise_k': round(P_ / G, 2), 'reaches_20c': reach}
            wrong += int(reach)
        for P_, t in ECON_TESTS:
            pred[f'{P_:g} W'] = {'time_s': round(-(C / G) * math.log(1 - 50 * G / P_), 1), 'measured_s': t}
        lg = -(C / G) * math.log(1 - 38.0 * G / 4.0)
        return Outcome(float(wrong), {'G_w_k': round(G, 5), 'C_j_k': round(C, 3), 'predictions': pred,
                                      'lg_innotek_4w_minus18_to_20c_s': round(lg, 1),
                                      'lg_innotek_measured_s': 240.0},
                       ok=abs(G - ECON_G0) < 5e-4 and abs(C - ECON_C) < 0.1)

    return [
        {'id': 'cleaning.spray_cone', 'model': 'cleaning', 'title': 'Spray cone on a lens window against closed forms',
         'quantity': 'largest |sampled covered area − closed form| over 3 square-on cones (circle–circle overlap) and '
                     '2 tilted cones (the ellipse π·a·b of the conic section), relative to the window\'s area',
         'reference': 'circle–circle intersection area and the cone\'s elliptical section, written out in the check',
         'tolerance': 0.005, 'unit': 'share of the window', 'refs': (), 'fn': _cone, 'cost': 'cheap'},
        {'id': 'cleaning.convection', 'model': 'cleaning',
         'title': 'Flat-plate convection against two published worked examples',
         'quantity': 'largest relative difference of cleaning.flat_plate_h from the examples\' h̄ with their own air '
                     'properties (and within 3 % with Sutherland\'s properties)',
         'reference': 'METU ME 212 (F. Arınç), forced convection, Examples 1 and 5 (Incropera\'s correlations)',
         'tolerance': 0.01, 'unit': 'relative', 'refs': ('incropera',), 'fn': _conv, 'cost': 'cheap'},
        {'id': 'cleaning.heater_econ', 'model': 'cleaning',
         'title': 'Lens heater warm-up against e-con Systems\' measurements',
         'quantity': 'outcomes mispredicted: the lumped model fitted to the 5.5 W and 4.5 W warm-ups (−30 → 20 °C in '
                     '300 and 480 s) must predict that 3.5 W and 2 W never reach 20 °C',
         'reference': _EC + ', tests 4–5', 'tolerance': 0.0, 'unit': 'outcomes', 'refs': ('econ_heater',),
         'fn': _heater, 'cost': 'cheap'},
    ]


REFERENCES = {
    'incropera': 'F. P. Incropera, D. P. DeWitt, T. L. Bergman, A. S. Lavine, Fundamentals of Heat and Mass Transfer, '
                 'Wiley: flat-plate correlations (laminar, turbulent, mixed).',
    'pope2000': 'S. B. Pope, Turbulent Flows, Cambridge University Press, 2000, §5.1 and Table 5.1 (the round jet).',
    'econ_heater': _EC + '.',
}


def validation_references() -> dict:
    """{key: citation} for vantage.validation.references (the keys the checks above cite)."""
    return dict(REFERENCES)
