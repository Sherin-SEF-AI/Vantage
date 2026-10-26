"""Surface materials of the vehicle and what they do to the rig's radars and the cameras behind glass.

Materials. Every face of the body and the rig has a material class (CLASSES): painted steel, aluminium, chrome,
carbon fibre (all conductive: a 77 GHz wave does not pass), a painted plastic bumper fascia, unpainted plastic trim,
a painted PC/ABS part, a PC radar cover, a PC lamp lens, a laminated windscreen, tempered side glass, tyre rubber
(absorbs). A mesh part takes the class set on it (mesh.Part.surface, and Part.surface_mm for its thickness), else one
from its role and name (a glass part, 'bumper', 'chrome', 'tyre' …; rig parts are aluminium). A plain painted body part
(the procedural bodies have one) is split by position when MaterialSettings.auto_regions is on: its front and rear
ends below the bumper line are the painted bumper fascia, and when the model has no forward-facing glass, the band
between the side windows and the bonnet that faces forward and up is the windscreen (the rear window likewise).

Radar through the fascia. A planar multilayer model: the fascia's substrate with the vehicle's paint (primer,
basecoat, clear coat) on the outside, each layer a complex permittivity eps'(1 - j tan d) and a thickness, solved with
the characteristic-matrix method for TE and TM at the ray's incidence angle (Macleod, Thin-Film Optical Filters, ch. 2;
Born & Wolf, Principles of Optics, §1.6), the two polarisations' powers averaged. One-way power transmission T, two-way
T²; the radar equation makes range ∝ (two-way transmission)^(1/4). Rays are traced over each radar's field
(its zones' azimuth and elevation half-angles) from its position; the first surface a ray leaves the vehicle through
within NEAR_M is what the radar looks through (a radar outside the body looks through nothing). A conductive or
absorbing first surface blocks the ray; a second surface within NEAR_M is the vehicle in the way. Each zone's range is
derated by the mean two-way transmission over its unblocked rays. The fascia's reflection back into the radar and the
cavity between the radar's front (taken as a metal plane, reflection -1: the worst case) and the fascia give the
multipath risk: the build-up 1/|1 + r e^(-j 2 k g cos θ)|² against the gap g, highest at a resonant spacing. Metal
within NEAR_METAL_M in or just beside the field adds ghost risk.

Cameras behind glass. A camera inside the body whose view leaves through a glass face within GLASS_NEAR_M looks
through it. Per image ray: the glass as plane-parallel layers (indices, thicknesses) at the ray's incidence shifts the
ray sideways without turning it (Σ d (sin θ − cos θ tan θ_j)); fitting one translation to all rays gives the camera's
apparent centre, and what is left is the error for near objects. Transmission and reflection are the incoherent slab
formulas with Fresnel at the two air–glass faces and the glass's absorption (from its visible transmission at normal
incidence). The view reflected at the glass's inner face looks down at the dashboard when it points downwards: veiling
glare, as the share of the dashboard's luminance the glass reflects into the image.

Wipers: core.wipers (the sweep on the windscreen and the cameras' footprints on it).
"""
from __future__ import annotations

import csv
import io
import json
import math
import os
import threading
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass, field

import numpy as np

from . import sensors as S
from .mathutil import BODY_TO_OPTICAL, rot_zyx
from .mesh import ROLE_GLASS, ROLE_IGNORED, ROLE_WHEELS, is_mirror

SCHEMA = 'vantage.materials/1'
FRAME = 'vehicle (ISO 8855, rear axle): x forward, y left, z up'
C0 = 299_792_458.0
EPS0 = 8.8541878128e-12
FASCIA_M = 0.15                 # as core.sensor_coverage.FASCIA_M: a surface this close is what a radar looks through
NEAR_M = 0.60                   # the first surface within this is what a radar or camera looks through
NEAR_METAL_M = 0.30             # metal this close to a radar, in or beside its field: ghost and multipath risk
METAL_MARGIN_DEG = 10.0         # 'beside the field': this far outside its edge
GLASS_NEAR_M = 0.40             # a camera whose view leaves the body through glass this close sits behind it
GLASS_RAY_M = 1.0               # its image rays are followed this far to the glass (a raked windscreen is far below)
MAX_INCIDENCE_DEG = 89.0
IMAGE_GRID = (32, 24)           # camera rays sampled over the image (columns, rows)
SEVERITIES = ('critical', 'serious', 'warning', 'good')
RISK_WORDS = ('low', 'moderate', 'high')

# =========================================================================== the library
# Dielectrics at 77 GHz: (name, eps', tan δ, source). tan δ of glass follows ITU-R P.2040's conductivity law.
DIELECTRICS = {
    'pp_talc': ('PP/EPDM bumper compound (talc-filled)', 2.75, 0.005,
                'Typical: Emilsson (Chalmers MSc thesis 2017, Table 5) measured PP-blend bumper substrates at '
                '2.23–3.25 − j0.001…0.031 (talc raises ε′); Ameerudeen & Nourjoo (Chalmers MSc thesis 2024, '
                'Table 4.2) PP at 2.44–2.74, tan δ 3e-4…8e-3.'),
    'pp_black': ('PP, black (carbon-filled trim)', 2.6, 0.01,
                 'Typical: PP 2.44–2.74 (Ameerudeen & Nourjoo 2024); carbon black raises the loss (Emilsson 2017).'),
    'pc': ('Polycarbonate', 2.82, 0.006, 'Typical: PC 2.80–2.82, tan δ 5e-3…7e-3 (Ameerudeen & Nourjoo 2024).'),
    'pc_abs': ('PC/ABS', 2.78, 0.008, 'Typical: PC/ABS 2.73–2.81, tan δ 6e-3…9e-3 (Ameerudeen & Nourjoo 2024).'),
    'primer': ('Primer', 2.8, 0.011, 'Emilsson 2017, Table 6: primers 2.78–2.81 − j0.024…0.036; 4–5 µm.'),
    'clear': ('Clear coat', 3.5, 0.02, 'ε′ 3.5 as Emilsson 2017 models top coats (after Schnabel et al.); '
                                       'tan δ 0.02 assumed; 40 µm (Emilsson, Table 12).'),
    'base_solid': ('Solid basecoat', 4.5, 0.15,
                   'Between the two solid basecoats Emilsson 2017 measured (Table 8: BC6 3.22 − j0.007, BC2 '
                   '5.99 − j1.61); 20 µm assumed (10–30 µm measured).'),
    'base_metallic': ('Metallic basecoat (aluminium flakes)', 43.0, 1e-4,
                      'Emilsson 2017, Table 8: factory basecoats with about 6 % aluminium flakes 37.2–48.9 − '
                      'j0.002…0.005; 15 µm (Table 11).'),
    'base_pearl': ('Pearl basecoat (mica)', 8.5, 0.13,
                   'Emilsson 2017, Table 8: mica effect basecoat BC5 7.0–10.0 − j0.9…1.3; 15 µm.'),
    'base_repair': ('Repair basecoat (aftermarket metallic)', 115.0, 1e-4,
                    'Emilsson 2017, §4.4.3: aftermarket repainted metallic basecoats fitted at ε′ 90–140; 15 µm.'),
    'glass': ('Soda-lime glass', 6.31, None,
              'ITU-R P.2040-3, Table 3: glass ε′ 6.31, σ = 0.0036 f^1.3394 S/m (f in GHz), tan δ ≈ 0.045 at '
              '77 GHz.'),
    'pvb': ('PVB interlayer', 2.8, 0.05, 'Assumed typical for a polymer interlayer; 0.76 mm.'),
}
GLASS_SIGMA = (0.0036, 1.3394)            # ITU-R P.2040-3: σ = c f^d S/m, f in GHz

# Paint finishes: the layers over a painted part's substrate, inner to outer (dielectric id, thickness mm).
FINISHES = {
    'none': (),
    'solid': (('primer', 0.005), ('base_solid', 0.020), ('clear', 0.040)),
    'metallic': (('primer', 0.005), ('base_metallic', 0.015), ('clear', 0.040)),
    'pearl': (('primer', 0.005), ('base_pearl', 0.015), ('clear', 0.040)),
    'repaired': (('primer', 0.005), ('base_metallic', 0.015), ('clear', 0.040), ('base_repair', 0.015),
                 ('clear', 0.040)),
}
FINISH_LABEL = {'none': 'Unpainted', 'solid': 'Solid', 'metallic': 'Metallic', 'pearl': 'Pearl',
                'repaired': 'Repaired metallic'}


@dataclass(frozen=True)
class MaterialClass:
    """One surface material. kind: 'metal' (conductive: reflects 77 GHz), 'absorber', 'dielectric' or 'glass'.
    layers: the substrate as ((dielectric id, thickness mm), ...) inner to outer; painted parts take the vehicle's
    paint over it. optical: glass as ((refractive index, thickness mm), ...) and t_vis its visible transmission at
    normal incidence."""
    id: str
    name: str
    kind: str
    layers: tuple = ()
    painted: bool = False
    optical: tuple = ()
    t_vis: float = 0.0
    color: str = '#999999'
    note: str = ''

    @property
    def thickness_mm(self) -> float:
        return float(sum(d for _, d in self.layers)) if self.layers else float(sum(d for _, d in self.optical))

    @property
    def opaque(self) -> bool:
        return self.kind in ('metal', 'absorber')


CLASSES = OrderedDict((c.id, c) for c in (
    MaterialClass('steel', 'Painted steel panel', 'metal', painted=True, color='#8a8f96',
                  note='Conductive: a 77 GHz wave does not pass; the paint on it does not change that.'),
    MaterialClass('aluminium', 'Aluminium', 'metal', color='#a9b0b8',
                  note='Conductive (rig parts are anodised aluminium).'),
    MaterialClass('chrome', 'Chrome-plated trim or grille', 'metal', color='#d6dbe0',
                  note='Electroplated chrome is a metal layer: opaque to radar. Radar emblems use a thin indium '
                       'layer instead (set them as a radar cover).'),
    MaterialClass('cfrp', 'Carbon fibre', 'metal', color='#3a3d42',
                  note='Carbon fibres conduct: treated as opaque to radar.'),
    MaterialClass('fascia', 'Bumper fascia (PP, painted)', 'dielectric', (('pp_talc', 3.0),), True, color='#c9a46a',
                  note='Typical bumper fascia: talc-filled PP/EPDM, 3.0 mm, painted in the vehicle\'s paint.'),
    MaterialClass('trim', 'Plastic trim (black PP, unpainted)', 'dielectric', (('pp_black', 2.5),), color='#57534d',
                  note='Unpainted textured trim, 2.5 mm.'),
    MaterialClass('pc_abs', 'PC/ABS part (painted)', 'dielectric', (('pc_abs', 2.5),), True, color='#b7a483',
                  note='Mirror caps and similar painted mouldings, 2.5 mm.'),
    MaterialClass('radome', 'Radar cover (PC, unpainted)', 'dielectric', (('pc', 2.0),), color='#d9c08f',
                  note='A dedicated radar cover or emblem, 2.0 mm, no paint.'),
    MaterialClass('lamp', 'Lamp lens (PC)', 'dielectric', (('pc', 3.0),), optical=((1.585, 3.0),), t_vis=0.89,
                  color='#cfd8dc', note='Polycarbonate lens, 3.0 mm; n 1.585, about 89 % visible transmission.'),
    MaterialClass('windscreen', 'Laminated windscreen', 'glass', (('glass', 2.1), ('pvb', 0.76), ('glass', 2.1)),
                  optical=((1.52, 2.1), (1.48, 0.76), (1.52, 2.1)), t_vis=0.88, color='#6f8fa6',
                  note='2.1 mm glass, 0.76 mm PVB, 2.1 mm glass; n 1.52 and 1.48; 88 % visible transmission at '
                       'normal incidence (clear; a green tint is nearer 80 %; UN ECE R43 asks at least 70 %).'),
    MaterialClass('side_glass', 'Tempered glass (side and rear windows)', 'glass', (('glass', 3.5),),
                  optical=((1.52, 3.5),), t_vis=0.85, color='#7f9cb0',
                  note='3.5 mm tempered glass, n 1.52, 85 % visible transmission (a light tint is common).'),
    MaterialClass('tyre', 'Tyre rubber', 'absorber', color='#2a2a2a',
                  note='Carbon-black filled rubber absorbs 77 GHz: treated as opaque.'),
))
CLASS_IDS = tuple(CLASSES)
GLASS_IDS = ('windscreen', 'side_glass')

ASSUMPTIONS = (
    'Material figures are typical values from the literature named in each note, not measurements of this '
    'vehicle: measure your fascia (a radome tester) and set its class and thickness. Surfaces are treated as planar '
    'layers at each ray\'s incidence (no curvature, no diffraction at edges); a radar\'s antenna pattern is uniform '
    'over its zones; its front face reflects like a metal plane in the cavity estimate. Glass is plane-parallel '
    'where a ray crosses it (a curved or wedged (head-up display) windscreen adds a prism deviation not modelled). '
    'Automatic regions (bumper ends, the windscreen band) are geometric guesses for a body modelled as one part.')
LIMITS_TEXT = (
    'The two-way loss and range derating hold for the far field; the multipath risk is an indicator (reflection '
    'level, cavity resonance, metal nearby), not a simulation of ghost targets. Thresholds are typical design '
    'targets (3 dB and 6 dB two-way, -10 dB reflection); use your radar supplier\'s.')


# =========================================================================== settings
LIMITS = {'radar_ghz': (60.0, 100.0), 'loss_warn_db': (0.1, 30.0), 'loss_serious_db': (0.2, 40.0),
          'refl_warn_db': (-40.0, -1.0), 'min_gap_mm': (0.0, 50.0), 'fascia_depth_m': (0.05, 1.5),
          'fascia_top_m': (0.0, 3.0), 'dash_reflectance': (0.0, 1.0), 'road_reflectance': (0.01, 1.0),
          'glare_warn': (0.0, 1.0), 'transmission_warn': (0.0, 1.0), 'calib_m': (0.3, 50.0)}


def _flag(v, default):
    if isinstance(v, (bool, int)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in ('1', 'true', 'yes', 'on')
    return default


@dataclass
class MaterialSettings:
    paint: str = 'solid'               # on the painted parts: none | solid | metallic | pearl | repaired
    radar_ghz: float = 76.5            # the radars' carrier (76–77 GHz long range; 77–81 GHz short range)
    loss_warn_db: float = 3.0          # two-way loss through what a radar looks through: a warning from this
    loss_serious_db: float = 6.0
    refl_warn_db: float = -10.0        # reflection back into the radar above this: multipath risk
    min_gap_mm: float = 4.0            # radar to fascia closer than this (about a wavelength): its near field
    auto_regions: bool = True          # split a plain painted body part into bumper fascia and windscreen regions
    fascia_depth_m: float = 0.35       # the bumper fascia reaches this far in from the body's front and rear ends
    fascia_top_m: float = 0.0          # and up to this height (0: 55 % of the body's height, at most 0.9 m)
    dash_reflectance: float = 0.05     # a dark dashboard (veiling glare)
    road_reflectance: float = 0.10     # the sunlit road the camera looks at
    glare_warn: float = 0.12           # glass reflecting more than this share of the dashboard's luminance: a warning
    transmission_warn: float = 0.60    # glass letting less than this through along a camera's axis: a warning
    calib_m: float = 3.0               # a calibration target's distance through the glass (the intrinsics absorb this)
    glare_shield: list = field(default_factory=list)   # uids of cameras with a black glare shield under them
    wipers: dict = field(default_factory=dict)          # core.wipers.WiperSettings

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d: dict | None) -> 'MaterialSettings':
        s = MaterialSettings()
        d = d if isinstance(d, dict) else {}
        if d.get('paint') in FINISHES:
            s.paint = d['paint']
        for k, (lo, hi) in LIMITS.items():
            if k in d:
                try:
                    v = float(d[k])
                except (TypeError, ValueError):
                    continue
                if math.isfinite(v):
                    setattr(s, k, min(max(v, lo), hi))
        s.loss_serious_db = max(s.loss_serious_db, s.loss_warn_db)
        s.auto_regions = _flag(d.get('auto_regions', True), True)
        g = d.get('glare_shield')
        if isinstance(g, (list, tuple)):
            s.glare_shield = sorted({str(u)[:40] for u in g if isinstance(u, str) and u})[:256]
        w = d.get('wipers')
        s.wipers = dict(w) if isinstance(w, dict) else {}
        return s


def settings_of(project) -> MaterialSettings:
    return MaterialSettings.from_dict(getattr(project.analysis, 'materials', None))


# =========================================================================== physics: radar through layers
def wavelength_m(f_ghz: float) -> float:
    return C0 / (float(f_ghz) * 1e9)


def tan_delta(diel: str, f_ghz: float) -> float:
    name, eps, tan, _ = DIELECTRICS[diel]
    if tan is None:                                        # glass: from ITU-R P.2040's conductivity law
        c, d = GLASS_SIGMA
        sigma = c * float(f_ghz) ** d
        return float(sigma / (2 * math.pi * f_ghz * 1e9 * EPS0 * eps))
    return float(tan)


def stack_rt(layers, f_ghz: float, theta_deg):
    """Complex amplitude reflection and transmission of a planar stack between air on both sides, for incidence
    angles theta_deg (array, degrees from the normal). layers: [(eps', tan δ, thickness m), ...] in the order the wave
    meets them. Returns (r_te, t_te, r_tm, t_tm), each shaped like theta_deg. Characteristic matrix with tilted
    admittances (Macleod ch. 2): power reflectance |r|², transmittance |t|² (air on both sides)."""
    th = np.radians(np.clip(np.asarray(theta_deg, float), 0.0, MAX_INCIDENCE_DEG))
    s0, c0 = np.sin(th), np.cos(th)
    k0 = 2 * math.pi / wavelength_m(f_ghz)
    out = []
    for te in (True, False):
        eta0 = c0 if te else 1.0 / c0
        m11 = np.ones(th.shape, complex)
        m12 = np.zeros(th.shape, complex)
        m21 = np.zeros(th.shape, complex)
        m22 = np.ones(th.shape, complex)
        for eps, tan, d in layers:
            N = np.sqrt(complex(eps * (1 - 1j * tan)))
            ct = np.sqrt(1 - (s0 / N) ** 2 + 0j)
            eta = N * ct if te else N / ct
            dl = k0 * N * d * ct
            c, s = np.cos(dl), np.sin(dl)
            a12, a21 = 1j * s / eta, 1j * eta * s
            m11, m12, m21, m22 = m11 * c + m12 * a21, m11 * a12 + m12 * c, m21 * c + m22 * a21, m21 * a12 + m22 * c
        B = m11 + m12 * eta0
        Cc = m21 + m22 * eta0
        den = eta0 * B + Cc
        out += [(eta0 * B - Cc) / den, 2 * eta0 / den]
    return tuple(out)


def layers_of(cls: MaterialClass, paint: str, f_ghz: float, thickness_mm: float = 0.0) -> list:
    """The radar layers of a dielectric or glass class, inner to outer: [(eps', tan δ, thickness m)]. thickness_mm > 0
    scales the substrate to that total; painted classes take the paint finish on the outside."""
    if not cls.layers:
        return []
    base = sum(d for _, d in cls.layers)
    k = thickness_mm / base if thickness_mm > 0 and base > 0 else 1.0
    out = [(DIELECTRICS[i][1], tan_delta(i, f_ghz), d * k / 1000.0) for i, d in cls.layers]
    if cls.painted:
        out += [(DIELECTRICS[i][1], tan_delta(i, f_ghz), d / 1000.0) for i, d in FINISHES.get(paint, ())]
    return out


def transmission(layers, f_ghz, theta_deg) -> tuple:
    """(one-way power transmission, power reflection seen from the inner side, complex r_te, complex r_tm), the two
    polarisations' powers averaged, for incidence angles theta_deg."""
    if not layers:
        th = np.asarray(theta_deg, float)
        one = np.ones(th.shape)
        return one, np.zeros(th.shape), np.zeros(th.shape, complex), np.zeros(th.shape, complex)
    r_te, t_te, r_tm, t_tm = stack_rt(layers, f_ghz, theta_deg)
    T = 0.5 * (np.abs(t_te) ** 2 + np.abs(t_tm) ** 2)
    R = 0.5 * (np.abs(r_te) ** 2 + np.abs(r_tm) ** 2)
    return T, R, r_te, r_tm


def db(x, floor=1e-12):
    return 10.0 * np.log10(np.maximum(np.asarray(x, float), floor))


def range_factor(two_way_transmission) -> float:
    """The radar equation: detection range ∝ (received power)^(1/4), so a two-way power transmission T² scales the
    range by (T²)^(1/4)."""
    return float(max(float(two_way_transmission), 0.0) ** 0.25)


def cavity_buildup(r_te, r_tm, gap_m, theta_deg, f_ghz, r_radar=-1.0):
    """Energy build-up 1/|1 − r_radar r e^(−j 2 k g cos θ)|² between the radar's front (reflection r_radar) and the
    fascia (reflection r, seen from the radar), the polarisations averaged; above 1 near a resonant spacing."""
    k0 = 2 * math.pi / wavelength_m(f_ghz)
    ph = np.exp(-1j * 2 * k0 * np.asarray(gap_m, float) * np.cos(np.radians(np.asarray(theta_deg, float))))
    f_te = 1.0 / np.abs(1 - r_radar * r_te * ph) ** 2
    f_tm = 1.0 / np.abs(1 - r_radar * r_tm * ph) ** 2
    return 0.5 * (f_te + f_tm)


def half_waves(cls: MaterialClass, f_ghz, theta_deg, thickness_mm=0.0) -> float:
    """The substrate's thickness in half wavelengths inside it at this incidence (its reflection is least at whole
    numbers when the paint is thin): d √(ε′ − sin²θ) / (λ0 / 2)."""
    if not cls.layers:
        return float('nan')
    base = sum(d for _, d in cls.layers)
    k = thickness_mm / base if thickness_mm > 0 and base > 0 else 1.0
    s2 = math.sin(math.radians(float(theta_deg))) ** 2
    opt = sum(d * k / 1000.0 * math.sqrt(max(DIELECTRICS[i][1] - s2, 1e-9)) for i, d in cls.layers)
    return float(opt / (wavelength_m(f_ghz) / 2))


def best_thickness(cls: MaterialClass, paint, f_ghz, theta_deg, current_mm) -> dict | None:
    """The substrate thickness nearest the current one at which the fascia (with its paint) reflects least at this
    incidence: {'thickness_mm', 'reflection_db', 'loss_db'} (two-way loss), None for a class without layers."""
    if not cls.layers:
        return None
    cur = current_mm if current_mm > 0 else cls.thickness_mm
    ts = np.linspace(max(0.3, 0.4 * cur), max(2.0 * cur, cur + 4.0), 700)
    base = sum(d for _, d in cls.layers)
    layers = [(DIELECTRICS[i][1], tan_delta(i, f_ghz), ts * (d / base) / 1000.0) for i, d in cls.layers]
    if cls.painted:
        layers += [(DIELECTRICS[i][1], tan_delta(i, f_ghz), d / 1000.0) for i, d in FINISHES.get(paint, ())]
    T, R, _, _ = transmission(layers, f_ghz, np.full(len(ts), float(theta_deg)))    # every thickness at once
    mins = [i for i in range(1, len(ts) - 1) if R[i] <= R[i - 1] and R[i] <= R[i + 1]]
    if not mins:
        return None
    i = min(mins, key=lambda j: abs(ts[j] - cur))
    return {'thickness_mm': float(ts[i]), 'reflection_db': float(db(R[i])), 'loss_db': float(-2 * db(T[i]))}


def class_figures(cls: MaterialClass, paint: str, f_ghz: float, thickness_mm: float = 0.0) -> dict:
    """A class's radar figures at normal incidence with the vehicle's paint (on painted classes): two-way loss and
    reflection in dB; opaque classes block (inf loss, 0 dB reflection for metal)."""
    if cls.kind == 'metal':
        return {'two_way_loss_db': float('inf'), 'reflection_db': 0.0}
    if cls.opaque or not cls.layers:
        return {'two_way_loss_db': float('inf'), 'reflection_db': None}
    T, R, _, _ = transmission(layers_of(cls, paint, f_ghz, thickness_mm), f_ghz, np.array([0.0]))
    return {'two_way_loss_db': float(-2 * db(T[0])), 'reflection_db': float(db(R[0]))}


# =========================================================================== physics: light through glass
def fresnel(n, theta_deg):
    """Power reflectance (R_s, R_p) of an air → n interface at incidence theta_deg."""
    th = np.radians(np.clip(np.asarray(theta_deg, float), 0.0, MAX_INCIDENCE_DEG))
    c, s = np.cos(th), np.sin(th)
    root = np.sqrt(np.maximum(n * n - s * s, 0.0))
    rs = ((c - root) / (c + root)) ** 2
    rp = ((n * n * c - root) / (n * n * c + root)) ** 2
    return rs, rp


def _internal_t0(n, t_vis):
    """Internal transmittance A0 at normal incidence that makes an incoherent slab of index n pass t_vis."""
    R = float(((n - 1) / (n + 1)) ** 2)
    if t_vis <= 0:
        return 0.0
    a = R * R * t_vis
    b = (1 - R) ** 2
    A = (-b + math.sqrt(b * b + 4 * a * t_vis)) / (2 * a) if a > 0 else t_vis / b
    return float(min(max(A, 0.0), 1.0))


def glass_optics(cls: MaterialClass, theta_deg, thickness_mm=0.0) -> dict:
    """Light through a glass class at incidence theta_deg (array): 'T' visible transmission, 'R' reflectance of the
    whole slab (both faces), 'shift_m' the ray's sideways shift (perpendicular to it), 'slide_m' the same shift
    measured along the surface. Incoherent slab, Fresnel at the outer faces (inner faces between the layers are
    index-matched enough to neglect), s and p averaged."""
    th = np.radians(np.clip(np.asarray(theta_deg, float), 0.0, MAX_INCIDENCE_DEG))
    base = sum(d for _, d in cls.optical) or 1.0
    k = thickness_mm / base if thickness_mm > 0 else 1.0
    s, c = np.sin(th), np.cos(th)
    slide = np.zeros(th.shape)
    path = np.zeros(th.shape)
    for n, d in cls.optical:
        dm = d * k / 1000.0
        sj = s / n
        cj = np.sqrt(1 - sj * sj)
        slide += dm * (np.tan(th) - sj / cj)
        path += dm / cj
    n_out = cls.optical[0][0] if cls.optical else 1.5
    A0 = _internal_t0(n_out, cls.t_vis)
    A = A0 ** (path / (base * k / 1000.0)) if A0 > 0 else np.zeros(th.shape)
    T, R = np.zeros(th.shape), np.zeros(th.shape)
    for Rf in fresnel(n_out, np.degrees(th)):
        den = 1 - Rf * Rf * A * A
        T += 0.5 * (1 - Rf) ** 2 * A / den
        R += 0.5 * (Rf + Rf * (1 - Rf) ** 2 * A * A / den)
    return {'T': T, 'R': R, 'slide_m': slide, 'shift_m': slide * c}


# =========================================================================== which material where
def _part_rule(part) -> tuple[str | None, str]:
    """(class id or None for 'by position', why) from a part's role, name and imported material."""
    sid = getattr(part, 'surface', '') or ''
    if sid in CLASSES:
        return sid, 'set'
    if part.role == ROLE_IGNORED:
        return '', 'ignored'
    if part.role == ROLE_WHEELS:
        return 'tyre', 'role'
    if part.material == 'rig':
        return 'aluminium', 'rig'
    text = f'{part.name} {part.material}'.lower()
    for keys, cid in ((('windscreen', 'windshield', 'front glass'), 'windscreen'),
                      (('chrome',), 'chrome'), (('carbon', 'cfrp'), 'cfrp'),
                      (('radome', 'emblem', 'badge', 'radar cover'), 'radome'),
                      (('bumper', 'fascia'), 'fascia'),
                      (('grille', 'grill', 'trim', 'cladding', 'moulding', 'molding', 'skirt', 'spoiler'), 'trim'),
                      (('tyre', 'tire', 'rubber'), 'tyre'), (('lamp', 'light', 'lens'), 'lamp'),
                      (('alumin',), 'aluminium'), (('steel', 'metal'), 'steel')):
        if any(k in text for k in keys):
            return cid, 'name'
    if part.role == ROLE_GLASS or any(k in text for k in ('glass', 'window')):
        return 'glass', 'glass'                          # windscreen or side glass by orientation
    if is_mirror(part.name):
        return 'pc_abs', 'name'
    return None, 'body'


@dataclass
class FaceMaterials:
    """Per face of a mesh: its class (index into CLASS_IDS, -1 none) and thickness override (mm, 0 = typical), the
    outward-oriented normals, and per part how its class was chosen."""
    cls: np.ndarray
    thick: np.ndarray
    normals: np.ndarray
    parts: list                 # [{'index', 'name', 'role', 'material', 'class', 'why', 'faces', 'area_m2', ...}]
    regions: dict               # automatic regions: {'fascia': n faces, 'windscreen': n, 'rear_window': n, ...}

    def of(self, f) -> MaterialClass | None:
        i = int(self.cls[f]) if 0 <= f < len(self.cls) else -1
        return CLASSES[CLASS_IDS[i]] if i >= 0 else None


WHY_TEXT = {'set': 'set by you', 'role': 'from its role', 'rig': 'rig part', 'name': 'from its name',
            'glass': 'glass, by orientation', 'body': 'painted body', 'ignored': 'ignored (not in the geometry)'}


def _outward(mesh) -> np.ndarray:
    """Face normals, each part's turned outward when most of its area faces the body's centre (an imported part wound
    inside out); a part is wound one way, so a concave spot or a flat plate keeps its faces as they are."""
    n = mesh.normals
    lo, hi = mesh.bbox
    ctr = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2])
    d = mesh.centroids - ctr
    vote = np.einsum('ij,ij->i', n, d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)) * mesh.areas
    fp = mesh.face_part
    k = int(fp.max()) + 1 if len(fp) else 0
    flip = np.bincount(fp, weights=vote, minlength=k) < -1e-9
    return n * np.where(flip[fp], -1.0, 1.0)[:, None]


_FM_CACHE = OrderedDict()
_LOCK = threading.RLock()


def face_materials(mesh, parts, st: MaterialSettings) -> FaceMaterials:
    """The material of every face of mesh, whose face_part indexes `parts` (the vehicle's current parts, then the
    rig's). Cached on the mesh, the parts and the region settings."""
    key = (id(mesh), mesh.n_tris, tuple((p.name, p.material, p.role, getattr(p, 'surface', ''),
                                         float(getattr(p, 'surface_mm', 0.0) or 0.0)) for p in parts),
           st.auto_regions, st.fascia_depth_m, st.fascia_top_m)
    with _LOCK:
        hit = _FM_CACHE.get(key)
        if hit is not None and hit[0] is mesh:
            _FM_CACHE.move_to_end(key)
            return hit[1]
    fm = _classify(mesh, parts, st)
    with _LOCK:
        _FM_CACHE[key] = (mesh, fm)
        while len(_FM_CACHE) > 6:
            _FM_CACHE.popitem(last=False)
    return fm


def _classify(mesh, parts, st: MaterialSettings) -> FaceMaterials:
    nf = mesh.n_tris
    idx = {c: i for i, c in enumerate(CLASS_IDS)}
    cls = np.full(nf, -1, np.int16)
    thick = np.zeros(nf, np.float32)
    if nf == 0:
        return FaceMaterials(cls, thick, np.zeros((0, 3)), [], {})
    fp = np.clip(mesh.face_part, 0, max(len(parts) - 1, 0))
    n = _outward(mesh)
    c = mesh.centroids
    a = mesh.areas
    rules = [_part_rule(p) for p in parts] or [(None, 'body')]
    body_like = np.zeros(nf, bool)
    glass_faces = np.zeros(nf, bool)
    for i, (cid, why) in enumerate(rules):
        m = fp == i
        if not m.any():
            continue
        if cid == 'glass':
            glass_faces |= m
        elif cid is None:
            body_like |= m
            cls[m] = idx['steel']
        elif cid:
            cls[m] = idx[cid]
        th = float(getattr(parts[i], 'surface_mm', 0.0) or 0.0) if i < len(parts) else 0.0
        if th > 0:
            thick[m] = th
    # glass by orientation: facing forward (and not sideways) is the windscreen
    front = (n[:, 0] > 0.3) & (np.abs(n[:, 1]) < 0.6)
    cls[glass_faces & front] = idx['windscreen']
    cls[glass_faces & ~front] = idx['side_glass']
    regions = {}
    solid = np.array([r[0] != '' and r[1] not in ('role', 'rig') for r in rules] or [True], bool)[fp]
    if solid.any():
        lo = c[solid].min(axis=0)
        hi = c[solid].max(axis=0)
        H = float(mesh.tri[solid][:, :, 2].max())
    else:
        lo, hi, H = c.min(axis=0), c.max(axis=0), float(c[:, 2].max())
    if st.auto_regions and body_like.any():
        top = st.fascia_top_m if st.fascia_top_m > 0 else min(0.55 * H, 0.9)
        dep = st.fascia_depth_m
        fas = body_like & (c[:, 2] <= top) & ((c[:, 0] >= hi[0] - dep) | (c[:, 0] <= lo[0] + dep))
        cls[fas] = idx['fascia']
        regions['fascia'] = int(fas.sum())
        regions['fascia_top_m'] = float(top)
        side = glass_faces & ~front
        has_front = (glass_faces & front & (n[:, 2] > -0.2)).any()
        has_rear = (glass_faces & (n[:, 0] < -0.3) & (np.abs(n[:, 1]) < 0.6)).any()
        belt = float(c[side, 2].min()) if side.any() else 0.6 * H
        for name, sign, has in (('windscreen', 1.0, has_front), ('rear_window', -1.0, has_rear)):
            if has:
                continue
            band = body_like & ~fas & (sign * n[:, 0] >= 0.40) & (n[:, 2] >= 0.30) & (np.abs(n[:, 1]) < 0.5) & \
                (c[:, 2] > belt) & (c[:, 2] < H - 0.03)
            if side.any():
                xg = float(c[side, 0].max()) if sign > 0 else float(c[side, 0].min())
                band &= (sign * (c[:, 0] - xg) <= 0.10) & (sign * (xg - c[:, 0]) <= 1.2)
            else:
                band &= (c[:, 2] >= 0.6 * H) & (sign * (c[:, 0] - (lo[0] + hi[0]) / 2) > 0)
            if band.sum() >= 4 and a[band].sum() > 0.05:
                cls[band] = idx['windscreen' if sign > 0 else 'side_glass']
                regions[name] = int(band.sum())
    infos = []
    for i, p in enumerate(parts):
        m = fp == i
        cid, why = rules[i]
        ids = sorted({CLASS_IDS[k] for k in np.unique(cls[m]) if k >= 0})
        infos.append({'index': i, 'name': p.name, 'role': p.role, 'material': p.material,
                      'class': cid if cid not in (None, 'glass') else (ids[0] if len(ids) == 1 else ''),
                      'classes': ids, 'why': why, 'faces': int(m.sum()), 'area_m2': float(a[m].sum()),
                      'thickness_mm': float(getattr(p, 'surface_mm', 0.0) or 0.0)})
    return FaceMaterials(cls, thick, n, infos, regions)


# =========================================================================== geometry helpers
def _geometry(project, vg=None):
    """(mesh, parts): the body with the rig merged (the engine's when given) and the parts its face_part indexes,
    the vehicle's current Part objects first (the engine's copy may hold older ones after an undo)."""
    from .rig import merge, rig_mesh
    veh = project.vehicle.mesh
    if veh is None or veh.n_tris == 0:
        return None, []
    rig = [r for r in getattr(project, 'rig', ()) if r.visible]
    mesh = getattr(vg, 'mesh', None)
    rm = None
    if mesh is None or mesh.n_tris < veh.n_tris or len(mesh.parts) < len(veh.parts):
        rm = rig_mesh(rig) if rig else None
        mesh = merge(veh, rm)
    parts = list(veh.parts) + list(mesh.parts[len(veh.parts):])
    return mesh, parts


def _near_faces(mesh, pos, radius) -> np.ndarray:
    """Indices of faces some point of which may lie within radius of pos."""
    if mesh.n_tris == 0:
        return np.zeros(0, np.int64)
    c = mesh.centroids
    r = np.linalg.norm(mesh.tri - c[:, None, :], axis=2).max(axis=1)
    return np.flatnonzero(np.linalg.norm(c - pos, axis=1) - r <= radius)


def _hits(O, D, tri, tmax):
    """Ray-triangle distances (K, M), inf where a ray misses (Möller–Trumbore), for rays from one origin O."""
    K, M = len(D), len(tri)
    if K == 0 or M == 0:
        return np.full((K, M), np.inf)
    v0, e1, e2 = tri[:, 0], tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    out = np.full((K, M), np.inf)
    step = max(1, 2_000_000 // max(M, 1))
    tv = O[None, :] - v0                                   # (M, 3)
    q = np.cross(tv, e1)                                   # (M, 3)
    tq = np.einsum('mj,mj->m', e2, q)
    for s in range(0, K, step):
        d = D[s:s + step]
        p = np.cross(d[:, None, :], e2[None])             # (k, M, 3)
        det = np.einsum('kmj,mj->km', p, e1)
        with np.errstate(divide='ignore', invalid='ignore'):
            inv = 1.0 / np.where(np.abs(det) < 1e-14, np.nan, det)
            u = np.einsum('mj,kmj->km', tv, p) * inv
            v = (d @ q.T) * inv
            t = tq[None, :] * inv
        ok = (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9) & (t > 1e-6) & (t <= tmax)
        out[s:s + step] = np.where(ok, t, np.inf)
    return out


def _first_two(T):
    """Per row of a hit matrix: (index of the first hit or -1, its distance, the second hit's distance)."""
    if T.shape[1] == 0:
        n = T.shape[0]
        return np.full(n, -1), np.full(n, np.inf), np.full(n, np.inf)
    j = np.argmin(T, axis=1)
    t1 = T[np.arange(len(T)), j]
    T2 = T.copy()
    T2[np.arange(len(T)), j] = np.inf
    t2 = T2.min(axis=1)
    return np.where(np.isfinite(t1), j, -1), t1, t2


def _inside(mesh, p) -> bool:
    from .rules import _inside as inside
    try:
        return bool(inside(mesh, p))
    except Exception:
        return False


def _finite(o, attrs=('x', 'y', 'z', 'yaw', 'pitch', 'roll')) -> bool:
    try:
        return all(math.isfinite(float(getattr(o, a))) for a in attrs)
    except (TypeError, ValueError):
        return False


def _where(az, el) -> str:
    h = 'left' if az > 10 else 'right' if az < -10 else ''
    v = 'upper' if el > 4 else 'lower' if el < -4 else ''
    return ' '.join(x for x in (v, h) if x) or 'centre'


# =========================================================================== radars
_RADARS = OrderedDict()


def radar_report(project, s, mesh, fm: FaceMaterials, st: MaterialSettings) -> dict | None:
    """What one radar looks through, the loss and range derating per zone, and the multipath risk (see the module
    docstring). None for a radar without zones. Cached on the geometry, the materials, the settings and the radar, so
    moving a camera does not trace the radars again."""
    m = project.sensor_model_for(s)
    try:
        key = (id(mesh), id(fm), json.dumps([st.to_dict(), s.to_dict(), m.to_dict()], sort_keys=True, default=str))
    except (TypeError, ValueError):
        key = None
    with _LOCK:
        hit = _RADARS.get(key) if key is not None else None
        if hit is not None and hit[0] is mesh and hit[1] is fm:
            _RADARS.move_to_end(key)
            return hit[2]
    rep = _radar_report(project, s, m, mesh, fm, st)
    if key is not None:
        with _LOCK:
            _RADARS[key] = (mesh, fm, rep)
            while len(_RADARS) > 48:
                _RADARS.popitem(last=False)
    return rep


def _radar_report(project, s, m, mesh, fm: FaceMaterials, st: MaterialSettings) -> dict | None:
    zones = [z for z in (m.zones or []) if z[0] > 0]
    if not zones or not _finite(s):
        return None
    f = st.radar_ghz
    lam = wavelength_m(f)
    pos = s.pos().astype(float)
    R = S.rotation(s)
    A = max(z[1] for z in zones)
    E = max(z[2] for z in zones)
    na = int(min(max(2 * round(A / 2.5) + 1, 9), 73))
    ne = int(min(max(2 * round(E / 2.0) + 1, 5), 17))
    az = np.linspace(-A, A, na)
    el = np.linspace(-E, E, ne)
    D = (S.directions(el, az) @ R.T).reshape(-1, 3)
    AZ, EL = np.meshgrid(az, el)
    AZ, EL = AZ.ravel(), EL.ravel()
    near = _near_faces(mesh, pos, NEAR_M)
    near = near[fm.cls[near] >= 0]                       # ignored parts are not there
    inside = _inside(mesh, pos)
    T_hit = _hits(pos, D, mesh.tri[near], NEAR_M)
    j, t1, t2 = _first_two(T_hit)
    face = np.where(j >= 0, near[np.maximum(j, 0)], -1) if len(near) else np.full(len(D), -1)
    K = len(D)
    T1 = np.ones(K)                                   # one-way power transmission per ray
    refl = np.zeros(K)
    theta = np.full(K, np.nan)
    blocked = np.zeros(K, bool)
    why_blocked = np.full(K, '', object)
    r_te = np.zeros(K, complex)
    r_tm = np.zeros(K, complex)
    cls_of = np.where(face >= 0, fm.cls[np.maximum(face, 0)], -1)
    hit = face >= 0
    if not inside:
        # outside the body: the first surface along a ray is the vehicle in the way, not a cover
        blocked[hit] = True
        why_blocked[hit] = 'vehicle'
    else:
        # leaves the body, then meets it again (a second triangle at the same distance is the same edge); or meets
        # nothing near: inside a closed body the vehicle is in the way further out
        again = (hit & (t2 <= NEAR_M) & (t2 > t1 + 2e-3)) | ~hit
        blocked |= again
        why_blocked[again] = 'vehicle'
        nrm = fm.normals[np.maximum(face, 0)]
        cosv = np.abs(np.einsum('ij,ij->i', D, nrm))
        theta = np.where(hit, np.degrees(np.arccos(np.clip(cosv, 0, 1))), np.nan)
        for k in np.unique(cls_of[hit]):
            sel = hit & (cls_of == k)
            if k < 0:
                blocked[sel] = True
                why_blocked[sel] = 'vehicle'
                continue
            mc = CLASSES[CLASS_IDS[k]]
            if mc.opaque:
                blocked[sel] = True
                why_blocked[sel] = mc.id
                continue
            for th_mm in np.unique(fm.thick[face[sel]]):
                ss = sel & (fm.thick[np.maximum(face, 0)] == th_mm)
                Tk, Rk, rte, rtm = transmission(layers_of(mc, st.paint, f, float(th_mm)), f, theta[ss])
                T1[ss], refl[ss], r_te[ss], r_tm[ss] = Tk, Rk, rte, rtm
    T2 = np.where(blocked, 0.0, T1 ** 2)
    # ---- boresight (the middle ray; az and el grids are odd and symmetric)
    b = (ne // 2) * na + na // 2
    b_face = int(face[b])
    b_cls = fm.of(b_face) if (b_face >= 0 and inside) else None
    gap = float(t1[b]) if b_face >= 0 else float('nan')
    # ---- the dominant surface over the field (by rays)
    through = hit & inside
    dom = None
    if through.any():
        vals, cnt = np.unique(cls_of[through], return_counts=True)
        top = int(vals[np.argmax(cnt)])
        dom = CLASSES[CLASS_IDS[top]] if top >= 0 else None
    look = b_cls or dom
    th_mm = float(fm.thick[b_face]) if b_face >= 0 else 0.0
    if look is not None and b_face >= 0 and fm.of(b_face) is not look:
        th_mm = 0.0
    if not inside:
        mount = 'exposed'
    elif look is None:
        mount = 'enclosed'                                 # inside, nothing within NEAR_M along its field
    elif look.opaque:
        mount = 'blocked'
    else:
        mount = 'behind' if (b_face >= 0 and gap <= FASCIA_M) else 'deep'
    zrep = []
    for rng, ah, eh in zones:
        zm = (np.abs(AZ) <= ah + 1e-9) & (np.abs(EL) <= eh + 1e-9)
        ok = zm & ~blocked
        mean2 = float(T2[ok].mean()) if ok.any() else 0.0
        k = range_factor(mean2) if ok.any() else 0.0
        zrep.append({'range_m': float(rng), 'az_half_deg': float(ah), 'el_half_deg': float(eh),
                     'rays': int(zm.sum()), 'blocked_share': float(blocked[zm].mean()) if zm.any() else 0.0,
                     'loss_db': max(0.0, float(-db(mean2))) if ok.any() else float('inf'),
                     'factor': k, 'derated_m': float(rng * k)})
    rep = {'uid': s.uid, 'tag': s.tag, 'name': s.name, 'model': f'{m.vendor} {m.name}'.strip(), 'mount': mount,
           'inside': bool(inside), 'class': look.id if look else '', 'class_name': look.name if look else '',
           'paint': st.paint if (look is not None and look.painted and not look.opaque) else 'none',
           'thickness_mm': (th_mm or (look.thickness_mm if look else 0.0)) if look and not look.opaque else 0.0,
           'gap_mm': gap * 1000 if math.isfinite(gap) else None,
           'incidence_deg': float(theta[b]) if math.isfinite(theta[b]) else None,
           'boresight_loss_db': max(0.0, float(-db(T2[b]))) if not blocked[b] else float('inf'),
           'boresight_blocked': bool(blocked[b]), 'reflection_db': None, 'zones': zrep,
           'blocked_share': float(blocked.mean()), 'blocked_by': {},
           'map': {'az': az, 'el': el, 'loss_db': np.where(blocked, np.inf, -db(np.maximum(T2, 1e-12))).reshape(ne, na),
                   'hit': (hit & inside).reshape(ne, na)},
           'half_waves': None, 'gap_half_waves': None, 'cavity': None, 'best_thickness': None, 'metal_near': None,
           'risk': 'low', 'reasons': []}
    for w in np.unique(why_blocked[blocked]):
        rep['blocked_by'][str(w)] = float((why_blocked == w).mean())
    lam_mm = lam * 1000
    if look is not None and not look.opaque and b_face >= 0 and not blocked[b]:
        th_b = float(theta[b])
        rep['reflection_db'] = float(db(refl[b]))
        rep['half_waves'] = half_waves(look, f, th_b, th_mm)
        rep['gap_half_waves'] = gap / (lam / 2)
        g = np.linspace(max(gap - lam / 2, 0.0), gap + lam / 2, 241)
        Fg = cavity_buildup(r_te[b], r_tm[b], g, th_b, f)
        F0 = float(cavity_buildup(r_te[b], r_tm[b], gap, th_b, f))
        lo_, hi_ = float(Fg.min()), float(Fg.max())
        res_idx = (F0 - lo_) / (hi_ - lo_) if hi_ - lo_ > 1e-9 else 0.0
        mins = [i for i in range(1, len(g) - 1) if Fg[i] <= Fg[i - 1] and Fg[i] <= Fg[i + 1]] or \
            [int(np.argmin(Fg))]
        anti = float(g[min(mins, key=lambda i: abs(g[i] - gap))])      # the anti-resonance nearest the gap
        rep['cavity'] = {'buildup_db': float(db(F0)), 'min_db': float(db(lo_)), 'max_db': float(db(hi_)),
                         'resonance': float(res_idx), 'best_gap_mm': anti * 1000, 'move_mm': (anti - gap) * 1000,
                         'ripple_db': float(db(hi_) - db(lo_)), 'wavelength_mm': lam_mm}
        rep['best_thickness'] = best_thickness(look, st.paint, f, th_b, th_mm or look.thickness_mm)
    rep['metal_near'] = _metal_near(mesh, fm, s, pos, R, A, E)
    _risk(rep, st)
    return rep


def _metal_near(mesh, fm, s, pos, R, A, E):
    """The nearest conductive surface within NEAR_METAL_M of the radar, ahead of it, inside its field or within
    METAL_MARGIN_DEG of its edge: {'class', 'name', 'distance_mm', 'outside_deg' (0 inside), 'where'}; None."""
    near = _near_faces(mesh, pos, NEAR_METAL_M)
    if not len(near):
        return None
    metal = np.array([k >= 0 and CLASSES[CLASS_IDS[k]].kind == 'metal' for k in fm.cls[near]], bool)
    near = near[metal]
    if not len(near):
        return None
    from .mesh import closest_on_triangles
    q = closest_on_triangles(pos, mesh.tri[near])
    d = q - pos
    dist = np.linalg.norm(d, axis=1)
    loc = d @ R                                            # sensor frame
    ahead = (loc[:, 0] > 0.005) & (dist <= NEAR_METAL_M) & (dist > 1e-4)
    if not ahead.any():
        return None
    azs = np.degrees(np.arctan2(loc[:, 1], loc[:, 0]))
    els = np.degrees(np.arcsin(np.clip(loc[:, 2] / np.maximum(dist, 1e-12), -1, 1)))
    out_deg = np.maximum(np.maximum(np.abs(azs) - A, np.abs(els) - E), 0.0)
    keep = ahead & (out_deg <= METAL_MARGIN_DEG)
    if not keep.any():
        return None
    k = np.flatnonzero(keep)[np.argmin(dist[keep] + 0.002 * out_deg[keep])]
    mc = fm.of(int(near[k]))
    return {'class': mc.id, 'name': mc.name, 'distance_mm': float(dist[k] * 1000), 'outside_deg': float(out_deg[k]),
            'where': _where(float(azs[k]), float(els[k]))}


def _risk(rep, st: MaterialSettings):
    """Multipath and ghost risk from the reflection level, the cavity, the gap and metal nearby, with reasons."""
    score, why = 0, []
    if rep['blocked_share'] >= 0.05 and any(k in ('steel', 'aluminium', 'chrome', 'cfrp')
                                            for k in rep['blocked_by']):
        score = 2
        why.append(f'metal in the beam ({rep["blocked_share"] * 100:.0f} % of the field)')
    mn = rep['metal_near']
    if mn is not None:
        if mn['outside_deg'] <= 0:
            score = max(score, 2)
            why.append(f'{_lc(mn["name"])} {mn["distance_mm"]:.0f} mm away inside the field ({mn["where"]})')
        else:
            score = max(score, 1)
            why.append(f'{_lc(mn["name"])} {mn["distance_mm"]:.0f} mm away, {mn["outside_deg"]:.0f}° beside the '
                       f'field ({mn["where"]})')
    r = rep['reflection_db']
    cav = rep['cavity']
    if r is not None and r > st.refl_warn_db:
        if cav is not None and cav['resonance'] > 0.75:
            score = max(score, 2)
            why.append(f'the fascia reflects {r:.1f} dB back and the gap is near a resonance '
                       f'({cav["buildup_db"]:+.1f} dB build-up)')
        else:
            score = max(score, 1)
            why.append(f'the fascia reflects {r:.1f} dB back (target below {st.refl_warn_db:g} dB)')
    g = rep['gap_mm']
    if rep['mount'] == 'behind' and g is not None and g < st.min_gap_mm:
        score = max(score, 1)
        why.append(f'the fascia is {g:.1f} mm from the radar, inside about one wavelength (its near field)')
    rep['risk'] = RISK_WORDS[score]
    rep['reasons'] = why


# =========================================================================== cameras behind glass
def glass_report(project, cam, mesh, fm: FaceMaterials, st: MaterialSettings) -> dict | None:
    """How a camera behind glass sees through it (module docstring); None when it is not behind glass."""
    from .lens import resolve
    if not cam.visible or not _finite(cam):
        return None
    pos = cam.pos().astype(float)
    R = rot_zyx(cam.yaw, cam.pitch, cam.roll)
    near = _near_faces(mesh, pos, GLASS_NEAR_M)
    near = near[fm.cls[near] >= 0]
    if not len(near):
        return None
    fwd = R[:, 0]
    tb = _hits(pos, fwd[None], mesh.tri[near], GLASS_NEAR_M)[0]
    if not np.isfinite(tb).any():
        return None
    bf = int(near[int(np.argmin(tb))])
    bcls = fm.of(bf)
    if bcls is None or bcls.kind != 'glass' or not bcls.optical:
        return None
    if not _inside(mesh, pos) and float(fwd @ fm.normals[bf]) <= 0:
        return None                                      # outside, looking at the glass from the street
    intr = resolve(project.module_for(cam), project.lens_for(cam))
    gw, gh = IMAGE_GRID
    us = (np.arange(gw) + 0.5) / gw * intr.width
    vs = (np.arange(gh) + 0.5) / gh * intr.height
    U, V = np.meshgrid(us, vs)
    U, V = np.r_[U.ravel(), intr.cx], np.r_[V.ravel(), intr.cy]       # the last sample: the optical axis
    d_opt, th = intr.unproject(U, V)
    valid = np.isfinite(th) & (th <= intr.max_theta) & np.isfinite(d_opt).all(axis=1)
    W2O = BODY_TO_OPTICAL @ R.T
    Dw = d_opt @ W2O                                       # optical -> vehicle
    Dw /= np.maximum(np.linalg.norm(Dw, axis=1, keepdims=True), 1e-12)
    far = _near_faces(mesh, pos, GLASS_RAY_M)
    far = far[fm.cls[far] >= 0]
    Th = _hits(pos, Dw, mesh.tri[far], GLASS_RAY_M)
    j, t1, _ = _first_two(Th)
    face = np.where(j >= 0, far[np.maximum(j, 0)], -1)
    kls = np.where(face >= 0, fm.cls[np.maximum(face, 0)], -1)
    gl = valid & (face >= 0) & np.isin(kls, [CLASS_IDS.index(g) for g in GLASS_IDS])
    blocked = valid & (face >= 0) & ~gl
    if not gl.any():
        return None
    nrm = fm.normals[np.maximum(face, 0)]
    cosv = np.einsum('ij,ij->i', Dw, nrm)
    theta = np.degrees(np.arccos(np.clip(np.abs(cosv), 0, 1)))
    T = np.full(len(Dw), np.nan)
    Rf = np.full(len(Dw), np.nan)
    shift = np.zeros((len(Dw), 3))
    for k in np.unique(kls[gl]):
        mc = CLASSES[CLASS_IDS[k]]
        for th_mm in np.unique(fm.thick[face[gl & (kls == k)]]):
            sel = gl & (kls == k) & (fm.thick[np.maximum(face, 0)] == th_mm)
            o = glass_optics(mc, theta[sel], float(th_mm))
            T[sel], Rf[sel] = o['T'], o['R']
            # the exit ray is parallel, slid back along the surface (towards the normal) by slide_m
            nn = nrm[sel] * np.sign(cosv[sel])[:, None]
            et = Dw[sel] - np.einsum('ij,ij->i', Dw[sel], nn)[:, None] * nn
            et /= np.maximum(np.linalg.norm(et, axis=1, keepdims=True), 1e-12)
            perp = et - np.einsum('ij,ij->i', et, Dw[sel])[:, None] * Dw[sel]
            shift[sel] = -o['slide_m'][:, None] * perp
    idx = np.flatnonzero(gl)
    Ds, Ss = Dw[idx], shift[idx]
    # one translation for all rays: (Σ P_i) T = Σ P_i s_i with P_i = I − d dᵀ
    P = np.eye(3)[None] - Ds[:, :, None] * Ds[:, None, :]
    A_ = P.sum(axis=0)
    bvec = np.einsum('kij,kj->i', P, Ss)
    try:
        Tc = np.linalg.lstsq(A_, bvec, rcond=None)[0]
    except np.linalg.LinAlgError:
        Tc = np.zeros(3)

    def px_err(D_m, centre):
        X = pos + Ss + D_m * Ds                            # the scene point each pixel really sees
        o = (X - pos - centre) @ W2O.T
        u, v, _, ok = intr.project_dirs(o)
        e = np.hypot(u - U[idx], v - V[idx])
        return e[np.isfinite(e)]
    errs = {}
    for D_m in (1.0, 3.0, 10.0):
        raw, fit = px_err(D_m, np.zeros(3)), px_err(D_m, Tc)
        errs[f'{D_m:g}'] = {'naive_rms_px': float(np.sqrt(np.mean(raw ** 2))) if len(raw) else 0.0,
                            'naive_max_px': float(raw.max()) if len(raw) else 0.0,
                            'fitted_rms_px': float(np.sqrt(np.mean(fit ** 2))) if len(fit) else 0.0,
                            'fitted_max_px': float(fit.max()) if len(fit) else 0.0}
    # what a calibration through the glass at calib_m absorbs into the intrinsics: p'' ≈ (1 + k)(p − c) + c + Δ
    X = pos + Ss + st.calib_m * Ds
    o = (X - pos) @ W2O.T
    u2, v2, _, _ = intr.project_dirs(o)
    ok = np.isfinite(u2) & np.isfinite(v2)
    intr_eq = None
    if ok.sum() >= 3:
        du, dv = U[idx][ok] - intr.cx, V[idx][ok] - intr.cy
        Am = np.zeros((2 * ok.sum(), 3))
        Am[0::2, 0], Am[1::2, 0] = du, dv
        Am[0::2, 1], Am[1::2, 2] = 1.0, 1.0
        rhs = np.empty(2 * ok.sum())
        rhs[0::2], rhs[1::2] = u2[ok] - U[idx][ok], v2[ok] - V[idx][ok]
        kk, ddu, ddv = np.linalg.lstsq(Am, rhs, rcond=None)[0]
        intr_eq = {'distance_m': st.calib_m, 'focal_ppm': float(kk * 1e6),
                   'principal_shift_px': [float(ddu), float(ddv)]}
    # the reflection at the inner face points down at the dashboard?
    nn = nrm * np.sign(cosv)[:, None]
    refl_dir = Dw - 2 * np.einsum('ij,ij->i', Dw, nn)[:, None] * nn
    dash = gl & (refl_dir[:, 2] < -0.05)
    b = len(U) - 1
    b_ok = bool(gl[b])
    T_vals, R_vals = T[gl], Rf[gl]
    r_dash = float(Rf[dash].mean()) if dash.any() else 0.0
    ratio = r_dash * st.dash_reflectance / max(st.road_reflectance, 1e-6)
    th_b = float(theta[b]) if b_ok else float(np.nanmedian(theta[gl]))
    t_b = float(T[b]) if b_ok else float(np.nanmedian(T_vals))
    nv = max(int(valid.sum()), 1)
    ray = np.linalg.norm(Ss, axis=1) * 1000
    rep = {'uid': cam.uid, 'tag': cam.tag, 'slot': cam.slot, 'class': bcls.id, 'class_name': bcls.name,
           'thickness_mm': float(fm.thick[bf]) or bcls.thickness_mm,
           'glass_share': float(gl.sum() / nv), 'blocked_share': float(blocked.sum() / nv),
           'cabin_share': float((valid & (face < 0)).sum() / nv),
           'gap_mm': float(tb.min() * 1000),
           'incidence_deg': th_b, 'incidence_range_deg': [float(theta[gl].min()), float(theta[gl].max())],
           'transmission': t_b,
           'transmission_min': float(np.nanmin(T_vals)), 'transmission_mean': float(np.nanmean(T_vals)),
           'stops_lost': float(math.log2(1.0 / max(t_b, 1e-6))),
           'reflectance': float(Rf[b]) if b_ok else float(np.nanmedian(R_vals)),
           'dash_share': float(dash.sum() / max(gl.sum(), 1)), 'dash_reflectance': r_dash,
           'veiling_ratio': ratio, 'contrast_kept': float(1.0 / (1.0 + ratio)),
           'apparent_shift_mm': [float(x * 1000) for x in Tc],
           'apparent_shift_norm_mm': float(np.linalg.norm(Tc) * 1000),
           'ray_shift_mm': [float(ray.min()), float(ray.max())],
           'errors': errs, 'intrinsics_equivalent': intr_eq,
           'shield': cam.uid in st.glare_shield}
    return rep


# =========================================================================== the study
@dataclass
class MaterialsResult:
    settings: MaterialSettings
    parts: list                 # per mesh part: its class and why (FaceMaterials.parts)
    regions: dict
    radars: list                # radar_report dicts
    cameras: list               # glass_report dicts
    wipers: object              # core.wipers.WiperResult or None
    findings: list              # [{'severity', 'kind', 'message', 'tags', 'uids'}], worst first
    metrics: dict
    variant: str = ''
    variant_uid: str = ''
    frequency_ghz: float = 76.5

    def worst(self) -> str:
        return self.findings[0]['severity'] if self.findings else 'good'

    def radar(self, tag):
        return next((r for r in self.radars if r['tag'] == tag), None)

    def radar_by_uid(self, uid):
        return next((r for r in self.radars if r['uid'] == uid), None)

    def camera(self, tag):
        return next((c for c in self.cameras if c['tag'] == tag), None)


def _project_for(project, variant):
    if variant is None or variant is project.variant:
        return project
    import copy
    q = copy.copy(project)
    q.variants = list(project.variants) + ([] if any(v is variant for v in project.variants) else [variant])
    q.active_uid = variant.uid
    return q


def run(project, engine=None, settings: MaterialSettings | None = None, variant=None, progress=None,
        cancel=None) -> MaterialsResult | None:
    """Materials, radars through the fascia, cameras behind glass and the wipers of a variant (the active one by
    default). engine: its vehicle geometry is reused (None: the body and rig are merged here). Returns None when
    cancelled."""
    from . import wipers as W
    t0 = time.perf_counter()
    p = _project_for(project, variant)
    v = p.variant
    st = settings or settings_of(p)

    def prog(x):
        if progress is not None:
            progress(min(max(float(x), 0.0), 1.0))

    def stop():
        return cancel is not None and cancel()
    vg = engine.vehicle_geom(p) if engine is not None else None
    mesh, parts = _geometry(p, vg)
    if mesh is None:
        res = MaterialsResult(st, [], {}, [], [], None, [], {'seconds': time.perf_counter() - t0, 'faces': 0},
                              v.name, v.uid, st.radar_ghz)
        res.findings = _findings(res)
        return res
    fm = face_materials(mesh, parts, st)
    prog(0.1)
    radars = []
    sens = [s for s in v.sensors if s.visible and s.kind == 'radar']
    for i, s in enumerate(sens):
        if stop():
            return None
        r = radar_report(p, s, mesh, fm, st)
        if r is not None:
            radars.append(r)
        prog(0.1 + 0.5 * (i + 1) / max(len(sens), 1))
    cams = []
    for c in v.cameras:
        if stop():
            return None
        g = glass_report(p, c, mesh, fm, st)
        if g is not None:
            cams.append(g)
    prog(0.75)
    if stop():
        return None
    wr = W.analyse(p, mesh, fm, W.WiperSettings.from_dict(st.wipers), cams)
    prog(1.0)
    res = MaterialsResult(st, fm.parts, dict(fm.regions), radars, cams, wr, [],
                          {'seconds': time.perf_counter() - t0, 'faces': int(mesh.n_tris), 'radars': len(radars),
                           'glass_cameras': len(cams)}, v.name, v.uid, st.radar_ghz)
    res.findings = _findings(res)
    return res


def _finding(out, sev, kind, message, items=()):
    out.append({'severity': sev, 'kind': kind, 'message': message, 'tags': [i['tag'] for i in items],
                'uids': [i['uid'] for i in items]})


def _lc(name) -> str:
    """A class name inside a sentence: 'Bumper fascia (PP, painted)' → 'bumper fascia (PP, painted)'."""
    return name[:1].lower() + name[1:] if name else name


def _paint_word(paint):
    return {'none': 'unpainted', 'solid': 'solid paint', 'metallic': 'metallic paint', 'pearl': 'pearl paint',
            'repaired': 'repaired metallic paint'}.get(paint, paint)


def _findings(res: MaterialsResult) -> list:
    st = res.settings
    out = []
    for r in res.radars:
        z0 = r['zones'][0]
        if r['mount'] == 'exposed':
            _finding(out, 'good', 'radar', f'Radar {r["tag"]} sits in front of the body: it looks through nothing '
                                           f'but its own radome (exposed to stone chips and dirt).', [r])
        elif r['mount'] == 'blocked':
            sev = 'critical' if r['blocked_share'] >= 0.5 else 'serious'
            _finding(out, sev, 'radar_blocked',
                     f'Radar {r["tag"]} looks into {_lc(r["class_name"])} ({r["blocked_share"] * 100:.0f} % of its '
                     f'field): 77 GHz does not pass it. Put the radar behind a plastic part (bumper fascia, radar '
                     f'cover) or in front of the panel.', [r])
        elif r['mount'] == 'enclosed':
            _finding(out, 'warning', 'radar_deep',
                     f'Radar {r["tag"]} sits inside the body with no surface within {NEAR_M * 100:.0f} cm along its '
                     f'field: the vehicle blocks it further out. Move it up to a bumper fascia or a radar cover.', [r])
        elif not math.isfinite(z0['loss_db']):
            _finding(out, 'warning', 'radar_vehicle', f'Radar {r["tag"]}: its {z0["az_half_deg"]:g}° beam is blocked '
                                                      f'by the vehicle within {NEAR_M * 100:.0f} cm.', [r])
        else:
            loss = z0['loss_db']
            what = (f'{_lc(r["class_name"])}, {r["thickness_mm"]:.1f} mm'
                    + (f', {_paint_word(r["paint"])}' if r['paint'] != 'none' else ''))
            text = (f'Radar {r["tag"]} looks through {what}: {loss:.1f} dB two-way loss over its '
                    f'{z0["az_half_deg"]:g}° beam, range {z0["range_m"]:.0f} → {z0["derated_m"]:.0f} m on a car '
                    f'({(z0["factor"] - 1) * 100:+.0f} %).')
            bt = r['best_thickness']
            if loss >= st.loss_warn_db:
                fix = []
                if r['paint'] in ('metallic', 'repaired', 'pearl'):
                    fix.append('no effect paint over the radar (a solid paint or an unpainted radar cover)')
                if bt is not None and bt['loss_db'] < loss - 0.3:
                    fix.append(f'a {bt["thickness_mm"]:.2f} mm fascia ({bt["loss_db"]:.1f} dB)')
                sev = 'serious' if loss >= st.loss_serious_db else 'warning'
                _finding(out, sev, 'radar_loss', text + (' Use ' + ' or '.join(fix) + '.' if fix else ''), [r])
            else:
                _finding(out, 'good', 'radar_loss', text, [r])
            if r['mount'] == 'deep':
                how = (f'sits {r["gap_mm"]:.0f} mm behind the {_lc(r["class_name"])}' if r['gap_mm'] is not None
                       else f'meets nothing within {NEAR_M * 100:.0f} cm along its boresight')
                _finding(out, 'warning', 'radar_deep',
                         f'Radar {r["tag"]} {how}: deeper than {FASCIA_M * 1000:.0f} mm, the coverage study counts '
                         f'the body as blocking it.', [r])
        vb = r['blocked_by'].get('vehicle', 0.0)
        if r['mount'] not in ('blocked',) and vb >= 0.05:
            _finding(out, 'warning', 'radar_vehicle',
                     f'Radar {r["tag"]}: the vehicle itself is within {NEAR_M * 100:.0f} cm in {vb * 100:.0f} % of '
                     f'its field.', [r])
        if r['risk'] == 'high':
            _finding(out, 'warning', 'radar_multipath',
                     f'Radar {r["tag"]}: high multipath and ghost-target risk: ' + '; '.join(r['reasons']) + '.'
                     + _cavity_hint(r), [r])
        elif r['risk'] == 'moderate':               # on the page, not in the Issues list
            bt = r['best_thickness']
            hint = (f' The fascia reflects least at {bt["thickness_mm"]:.2f} mm ({bt["reflection_db"]:.0f} dB).'
                    if bt is not None and r['reflection_db'] is not None and r['reflection_db'] > st.refl_warn_db
                    and bt['reflection_db'] < st.refl_warn_db else '')
            _finding(out, 'warning', 'radar_note',
                     f'Radar {r["tag"]}: moderate multipath risk: ' + '; '.join(r['reasons']) + '.' + _cavity_hint(r)
                     + hint, [r])
    for c in res.cameras:
        tv = c['transmission']
        txt = (f'Camera {c["tag"]} looks through the {_lc(c["class_name"])} at {c["incidence_deg"]:.0f}° '
               f'incidence: {tv * 100:.0f} % of the light passes ({c["stops_lost"]:.2f} stops darker); its centre '
               f'appears {c["apparent_shift_norm_mm"]:.2f} mm off, {c["errors"]["3"]["fitted_rms_px"]:.2f} px left '
               f'at 3 m.')
        _finding(out, 'warning' if tv < st.transmission_warn else 'good', 'glass_transmission', txt, [c])
        if c['glass_share'] < 0.98:
            _finding(out, 'warning', 'glass_view',
                     f'Camera {c["tag"]}: {100 * (1 - c["glass_share"]):.0f} % of its image does not look through the '
                     f'glass: {100 * c["blocked_share"]:.0f} % sees the body (pillars, roof) and '
                     f'{100 * c["cabin_share"]:.0f} % the cabin (the dashboard; nothing inside is modelled).', [c])
        if c['dash_share'] > 0 and c['dash_reflectance'] > st.glare_warn and not c['shield']:
            _finding(out, 'warning', 'glass_glare',
                     f'Camera {c["tag"]}: the glass reflects {c["dash_reflectance"] * 100:.0f} % of the dashboard '
                     f'into {c["dash_share"] * 100:.0f} % of its view (veiling glare: about '
                     f'{(1 - c["contrast_kept"]) * 100:.0f} % contrast lost on a sunlit road). Fit a black, textured '
                     'glare shield between the camera and the glass.', [c])
    wr = res.wipers
    if wr is not None:
        out += [dict(f) for f in wr.findings]
    order = {s: i for i, s in enumerate(SEVERITIES)}
    out.sort(key=lambda f: order.get(f['severity'], 9))
    return out


def _cavity_hint(r) -> str:
    cav = r.get('cavity')
    if not cav or cav['resonance'] < 0.5 or abs(cav['move_mm']) < 0.05:
        return ''
    return (f' Moving the radar {abs(cav["move_mm"]):.1f} mm {"back" if cav["move_mm"] > 0 else "forward"} '
            f'(a {cav["best_gap_mm"]:.1f} mm gap) puts it between resonances.')


# =========================================================================== the Issues list and cache
_RUNS = OrderedDict()
ISSUE_KINDS = ('radar_blocked', 'radar_loss', 'radar_multipath', 'glass_transmission', 'glass_glare', 'wiper')


def _key(project, vg=None):
    v = project.variant
    m = project.vehicle.mesh
    try:
        return json.dumps([id(m), getattr(m, 'n_tris', 0), id(getattr(vg, 'mesh', None)),
                           [(p.name, p.material, p.role, getattr(p, 'surface', ''), getattr(p, 'surface_mm', 0.0))
                            for p in (m.parts if m is not None else [])],
                           [r.to_dict() for r in project.rig if r.visible], project.analysis.materials,
                           [(s.uid, s.tag, s.kind, s.visible, s.x, s.y, s.z, s.yaw, s.pitch, s.roll, s.model_id,
                             project.sensor_model_for(s).zones) for s in v.sensors if s.kind == 'radar'],
                           [(c.uid, c.tag, c.slot, c.visible, c.x, c.y, c.z, c.yaw, c.pitch, c.roll, c.module_id,
                             c.lens_id, c.calib) for c in v.cameras],
                           v.uid, v.name], sort_keys=True, default=str)
    except (TypeError, ValueError):
        return None


def cached_run(project, vg=None) -> MaterialsResult:
    """run() for the active variant, cached on everything it reads (the Issues list and the 3D view ask often)."""
    key = _key(project, vg)
    with _LOCK:
        hit = _RUNS.get(key) if key is not None else None
        if hit is not None:
            _RUNS.move_to_end(key)
            return hit

    class _E:                                             # the given geometry, without an engine
        def vehicle_geom(self, _p):
            return vg
    res = run(project, _E() if vg is not None else None)
    if key is not None:
        with _LOCK:
            _RUNS[key] = res
            while len(_RUNS) > 8:
                _RUNS.popitem(last=False)
    return res


def issues(project, vg=None) -> list:
    """Radars looking through metal or losing more than the warning level, high multipath risk, cameras behind glass
    that pass too little light or catch the dashboard's reflection, cameras looking through unwiped glass
    (core.wipers). Cached on the inputs (a few tens of ms when something moved)."""
    from .rules import Issue
    if project.vehicle.mesh is None:
        return []
    v = project.variant
    if not any(s.visible and s.kind == 'radar' for s in v.sensors) and not v.cameras:
        return []
    try:
        res = cached_run(project, vg)
    except Exception as e:                    # never take the whole Issues list down; say so instead
        import traceback
        traceback.print_exc()
        return [Issue('warning', f'The materials check could not run ({type(e).__name__}: {e}).',
                      f'{v.name} · materials')]
    out = []
    for f in res.findings:
        if f['severity'] == 'good' or not any(f['kind'].startswith(k) for k in ISSUE_KINDS):
            continue
        where = None
        u = f['uids'][0] if f['uids'] else None
        o = (v.sensor(u) if u else None) or (v.cam(u) if u else None)
        if o is not None:
            where = (o.x, o.y)
        out.append(Issue(f['severity'], f['message'], ', '.join(f['tags']) or f'{res.variant} · materials',
                         list(f['uids']), where=where))
    return out


def clear_cache():
    with _LOCK:
        _RUNS.clear()
        _FM_CACHE.clear()
        _RADARS.clear()


# =========================================================================== the coverage study's hook
def radar_range_factors(project, sensor, vg=None) -> list:
    """Per zone of a radar's model, the factor its range shrinks by through what it looks through (1 when exposed or
    nothing is known); for core.sensor_coverage."""
    m = project.sensor_model_for(sensor)
    n = len(m.zones or [])
    try:
        res = cached_run(project, vg)
    except Exception:
        return [1.0] * n
    r = res.radar_by_uid(sensor.uid) if res is not None else None
    if r is None or len(r['zones']) != n:
        return [1.0] * n
    return [float(z['factor']) if z['loss_db'] != float('inf') else 1.0 for z in r['zones']]


# =========================================================================== output
def _r(v, nd=3):
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return round(f, nd) if math.isfinite(f) else None


def _radar_doc(r) -> dict:
    return {'tag': r['tag'], 'model': r['model'], 'mount': r['mount'], 'looks_through': r['class_name'] or None,
            'class': r['class'] or None, 'paint': r['paint'], 'thickness_mm': _r(r['thickness_mm']),
            'gap_mm': _r(r['gap_mm'], 2), 'incidence_deg': _r(r['incidence_deg'], 2),
            'boresight_two_way_loss_db': _r(r['boresight_loss_db'], 2),
            'reflection_db': _r(r['reflection_db'], 2), 'thickness_half_waves': _r(r['half_waves'], 3),
            'gap_half_waves': _r(r['gap_half_waves'], 3),
            'zones': [{'range_on_car_m': _r(z['range_m'], 1), 'azimuth_half_deg': z['az_half_deg'],
                       'elevation_half_deg': z['el_half_deg'], 'two_way_loss_db': _r(z['loss_db'], 2),
                       'range_factor': _r(z['factor'], 4), 'derated_range_m': _r(z['derated_m'], 1),
                       'blocked_share': _r(z['blocked_share'], 4)} for z in r['zones']],
            'blocked_share': _r(r['blocked_share'], 4), 'blocked_by': {k: _r(v, 4) for k, v in r['blocked_by'].items()},
            'cavity': {k: _r(v, 3) for k, v in r['cavity'].items()} if r['cavity'] else None,
            'best_thickness': {k: _r(v, 3) for k, v in r['best_thickness'].items()} if r['best_thickness'] else None,
            'metal_near': dict(r['metal_near'], distance_mm=_r(r['metal_near']['distance_mm'], 1),
                               outside_deg=_r(r['metal_near']['outside_deg'], 1)) if r['metal_near'] else None,
            'multipath_risk': r['risk'], 'reasons': r['reasons']}


def _camera_doc(c) -> dict:
    return {'tag': c['tag'], 'glass': c['class_name'], 'thickness_mm': _r(c['thickness_mm']),
            'gap_mm': _r(c['gap_mm'], 1), 'incidence_deg': _r(c['incidence_deg'], 2),
            'incidence_range_deg': [_r(x, 2) for x in c['incidence_range_deg']],
            'image_through_glass': _r(c['glass_share'], 4), 'image_on_body': _r(c['blocked_share'], 4),
            'image_in_cabin': _r(c['cabin_share'], 4),
            'transmission': _r(c['transmission'], 4), 'transmission_min': _r(c['transmission_min'], 4),
            'stops_lost': _r(c['stops_lost'], 3), 'reflectance': _r(c['reflectance'], 4),
            'dashboard_reflection': {'image_share': _r(c['dash_share'], 4), 'reflectance': _r(c['dash_reflectance'], 4),
                                     'contrast_kept': _r(c['contrast_kept'], 4), 'glare_shield': c['shield']},
            'apparent_centre_shift_mm': [_r(x, 3) for x in c['apparent_shift_mm']],
            'ray_shift_mm': [_r(x, 3) for x in c['ray_shift_mm']],
            'pixel_error': {f'{k}_m': {kk: _r(vv, 3) for kk, vv in e.items()} for k, e in c['errors'].items()},
            'intrinsics_equivalent': ({'distance_m': c['intrinsics_equivalent']['distance_m'],
                                       'focal_ppm': _r(c['intrinsics_equivalent']['focal_ppm'], 1),
                                       'principal_shift_px': [_r(x, 3) for x in
                                                              c['intrinsics_equivalent']['principal_shift_px']]}
                                      if c['intrinsics_equivalent'] else None)}


def summary(res: MaterialsResult) -> dict:
    """Compact figures (the assistant, the batch log)."""
    out = {'variant': res.variant, 'paint': res.settings.paint, 'frequency_ghz': res.frequency_ghz,
           'radars': [{'tag': r['tag'], 'mount': r['mount'], 'looks_through': r['class_name'] or None,
                       'two_way_loss_db': _r(r['zones'][0]['loss_db'], 2),
                       'range_m': [_r(z['derated_m'], 1) for z in r['zones']],
                       'range_factor': [_r(z['factor'], 3) for z in r['zones']], 'multipath_risk': r['risk']}
                      for r in res.radars],
           'glass_cameras': [{'tag': c['tag'], 'glass': c['class_name'], 'incidence_deg': _r(c['incidence_deg'], 1),
                              'transmission': _r(c['transmission'], 3),
                              'apparent_shift_mm': _r(c['apparent_shift_norm_mm'], 2),
                              'dashboard_reflectance': _r(c['dash_reflectance'], 3)} for c in res.cameras],
           'findings': [{'severity': f['severity'], 'message': f['message']} for f in res.findings
                        if f['severity'] != 'good'],
           'assumptions': ASSUMPTIONS}
    if res.wipers is not None:
        from . import wipers as W
        out['wipers'] = W.summary(res.wipers)
    return out


def to_doc(res: MaterialsResult, project=None) -> dict:
    """The JSON export: the material of every part, every radar's view through the fascia, every camera behind
    glass, the wipers, the library and the findings."""
    from . import wipers as W
    st = res.settings
    doc = {'schema': SCHEMA, 'project': getattr(project, 'name', ''), 'variant': res.variant, 'frame': FRAME,
           'units': 'metres and degrees unless named (mm, dB: two-way power loss positive, reflection negative); '
                    'loss and range at the radar frequency',
           'frequency_ghz': res.frequency_ghz, 'wavelength_mm': _r(wavelength_m(res.frequency_ghz) * 1000, 4),
           'settings': st.to_dict(), 'assumptions': ASSUMPTIONS, 'limits': LIMITS_TEXT,
           'parts': [{'name': p['name'], 'role': p['role'], 'imported_material': p['material'],
                      'class': p['class'] or None, 'classes': p['classes'], 'chosen': WHY_TEXT.get(p['why'], p['why']),
                      'thickness_mm': _r(p['thickness_mm']) or None, 'area_m2': _r(p['area_m2'], 4)}
                     for p in res.parts],
           'automatic_regions': res.regions,
           'radars': [_radar_doc(r) for r in res.radars],
           'cameras_behind_glass': [_camera_doc(c) for c in res.cameras],
           'wipers': W.to_doc(res.wipers) if res.wipers is not None else None,
           'library': {'classes': [{'id': c.id, 'name': c.name, 'kind': c.kind,
                                    'layers': [{'material': DIELECTRICS[i][0], 'thickness_mm': d,
                                                'eps_r': DIELECTRICS[i][1],
                                                'tan_delta': _r(tan_delta(i, res.frequency_ghz), 5)}
                                               for i, d in c.layers],
                                    'painted': c.painted,
                                    'optical': [{'n': n, 'thickness_mm': d} for n, d in c.optical] or None,
                                    'visible_transmission': c.t_vis or None, 'note': c.note,
                                    'at_normal_incidence': {k: _r(v, 3) for k, v in
                                                            class_figures(c, st.paint, res.frequency_ghz).items()}}
                                   for c in CLASSES.values()],
                       'paint': {k: [{'material': DIELECTRICS[i][0], 'thickness_mm': d} for i, d in v]
                                 for k, v in FINISHES.items()},
                       'sources': {k: v[3] for k, v in DIELECTRICS.items()}},
           'findings': [{'severity': f['severity'], 'kind': f['kind'], 'message': f['message'], 'objects': f['tags']}
                        for f in res.findings],
           'seconds': _r(res.metrics.get('seconds'), 3)}
    return doc


RADAR_COLUMNS = ('tag', 'mount', 'looks_through', 'paint', 'thickness_mm', 'gap_mm', 'incidence_deg',
                 'boresight_two_way_loss_db', 'zone', 'range_on_car_m', 'two_way_loss_db', 'range_factor',
                 'derated_range_m', 'reflection_db', 'thickness_half_waves', 'cavity_resonance', 'multipath_risk')
GLASS_COLUMNS = ('tag', 'glass', 'thickness_mm', 'incidence_deg', 'image_through_glass', 'transmission',
                 'stops_lost', 'apparent_shift_x_mm', 'apparent_shift_y_mm', 'apparent_shift_z_mm',
                 'fitted_error_3m_rms_px', 'naive_error_3m_rms_px', 'dashboard_reflectance', 'contrast_kept',
                 'wiped_share')


def radars_csv(res: MaterialsResult) -> str:
    """One row per radar zone."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator='\n')
    w.writerow(RADAR_COLUMNS)
    blank = lambda v: '' if v is None else v          # noqa: E731
    for r in res.radars:
        d = _radar_doc(r)
        for k, z in enumerate(d['zones']):
            w.writerow([blank(x) for x in (
                d['tag'], d['mount'], d['looks_through'], d['paint'], d['thickness_mm'], d['gap_mm'],
                d['incidence_deg'], d['boresight_two_way_loss_db'], k + 1, z['range_on_car_m'], z['two_way_loss_db'],
                z['range_factor'], z['derated_range_m'], d['reflection_db'], d['thickness_half_waves'],
                (d['cavity'] or {}).get('resonance'), d['multipath_risk'])])
    return buf.getvalue()


def glass_csv(res: MaterialsResult) -> str:
    """One row per camera behind glass."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator='\n')
    w.writerow(GLASS_COLUMNS)
    wiped = {c['tag']: c['share_wiped'] for c in (res.wipers.cameras if res.wipers is not None else [])}
    blank = lambda v: '' if v is None else v          # noqa: E731
    for c in res.cameras:
        d = _camera_doc(c)
        sh = d['apparent_centre_shift_mm']
        w.writerow([blank(x) for x in (
            d['tag'], d['glass'], d['thickness_mm'], d['incidence_deg'], d['image_through_glass'], d['transmission'],
            d['stops_lost'], sh[0], sh[1], sh[2], d['pixel_error']['3_m']['fitted_rms_px'],
            d['pixel_error']['3_m']['naive_rms_px'], d['dashboard_reflection']['reflectance'],
            d['dashboard_reflection']['contrast_kept'], _r(wiped.get(c['tag']), 4))])
    return buf.getvalue()


_ASCII = str.maketrans({'·': '-', '–': '-', '—': '-', '°': ' deg', '×': 'x', '≥': '>=', '≤': '<=', '²': '2',
                        '→': '->', 'µ': 'u', 'λ': 'lambda', '±': '+/-'})


def _ascii(text) -> str:
    return str(text).translate(_ASCII).encode('ascii', 'replace').decode('ascii')


LOSS_RAMP = ['#e8f1fb', '#c6dcf4', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']


def loss_rgb(loss_db, vmax=10.0) -> np.ndarray:
    """RGB 0..255 for two-way losses (array): a sequential ramp from 0 dB to vmax; nan/inf (blocked) dark red."""
    x = np.asarray(loss_db, float)
    cols = np.array([[int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)] for h in LOSS_RAMP], float)
    t = np.clip(np.where(np.isfinite(x), x, 0.0) / vmax, 0, 1) * (len(cols) - 1)
    i0 = np.floor(t).astype(int)
    i1 = np.minimum(i0 + 1, len(cols) - 1)
    f = (t - i0)[..., None]
    rgb = cols[i0] * (1 - f) + cols[i1] * f
    return np.where(np.isfinite(x)[..., None], rgb, np.array([176.0, 49.0, 47.0]))


def png(res: MaterialsResult, width=1400) -> bytes:
    """Each radar's two-way loss over its field (azimuth across, elevation up; dark red where blocked) and the
    windscreen with the wiped area and the cameras' footprints."""
    import cv2
    from . import wipers as W
    font, aa = cv2.FONT_HERSHEY_SIMPLEX, cv2.LINE_AA
    bg, ink, ink2, muted = (30, 29, 28), (236, 235, 230), (200, 199, 192), (140, 140, 135)
    pw, ph = 300, 150
    per_row = max(1, (width - 40) // (pw + 30))
    rows = math.ceil(len(res.radars) / per_row) if res.radars else 0
    top = 70
    ws_h = 360 if res.wipers is not None and res.wipers.windscreen is not None else 0
    H = top + rows * (ph + 70) + (ws_h + 50 if ws_h else 0) + 40
    img = np.full((H, width, 3), bg, np.uint8)

    def text(t, x, y, col=ink, s=0.45, th=1):
        cv2.putText(img, _ascii(t), (int(x), int(y)), font, s, col, th, aa)
    text(f'Materials: radars through the fascia, glass and wipers  {res.variant}', 20, 30, ink, 0.65)
    text(f'{FINISH_LABEL.get(res.settings.paint, res.settings.paint)} paint  {res.frequency_ghz:g} GHz  two-way loss '
         f'0 to 10 dB (dark red: blocked)', 20, 54, ink2, 0.45)
    for k, r in enumerate(res.radars):
        x0 = 20 + (k % per_row) * (pw + 30)
        y0 = top + (k // per_row) * (ph + 70)
        mp = r['map']
        rgb = loss_rgb(mp['loss_db'])[::-1, ::-1]          # elevation up, azimuth left to the left
        tile = cv2.resize(rgb.astype(np.uint8), (pw, ph), interpolation=cv2.INTER_NEAREST)[:, :, ::-1]
        img[y0 + 22:y0 + 22 + ph, x0:x0 + pw] = tile
        z0 = r['zones'][0]
        head = f'{r["tag"]}  {r["class_name"] or r["mount"]}'
        text(head, x0, y0 + 16, ink, 0.45)
        sub = (f'{z0["loss_db"]:.1f} dB  {z0["range_m"]:.0f} -> {z0["derated_m"]:.0f} m  risk {r["risk"]}'
               if math.isfinite(z0['loss_db']) else f'blocked  risk {r["risk"]}')
        text(sub, x0, y0 + ph + 40, ink2, 0.4)
        az = mp['az']
        text(f'{az[-1]:+.0f}', x0, y0 + ph + 56, muted, 0.32)
        text(f'{az[0]:+.0f} deg az', x0 + pw - 70, y0 + ph + 56, muted, 0.32)
    if ws_h:
        y0 = top + rows * (ph + 70) + 20
        W.draw_png(res.wipers, img, (20, y0, width - 40, ws_h), text)
    ok, buf = cv2.imencode('.png', img)
    return buf.tobytes() if ok else b''


def files(res: MaterialsResult, project=None) -> list:
    """[(name, bytes)]: materials.json, materials_radars.csv, materials_glass.csv, materials.png."""
    return [('materials.json', (json.dumps(to_doc(res, project), indent=1) + '\n').encode('utf-8')),
            ('materials_radars.csv', radars_csv(res).encode('utf-8')),
            ('materials_glass.csv', glass_csv(res).encode('utf-8')),
            ('materials.png', png(res))]


def export(res: MaterialsResult, project, out_dir) -> list[str]:
    """Write files() into out_dir; returns their paths."""
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for name, data in files(res, project):
        path = os.path.join(out_dir, name)
        with open(path, 'wb') as fh:
            fh.write(data)
        paths.append(path)
    return paths
