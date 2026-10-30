"""ISP auto-exposure metering masks and the photometric seam model: what each camera's auto-exposure (AE) should meter,
and how far the exposures of neighbouring cameras drift apart across the bird's-eye view's stitch seams over a
data-collection campaign and in hard scenes (low sun ahead, a tunnel exit, night under street lights).

Metering masks
--------------
Per camera, through the real lens model (core.lens) at sensor resolution (or a set fraction of it), every pixel ray is
classed: outside the lens' image circle; the vehicle body and the rig parts (the engine's occlusion cube from the
camera's centre, as core.synth draws them: body surface nearer than 5 cm is the lens' own module); the sensor and
camera housings (core.housings, composited into the cube); the sky above the horizon plus a margin (the horizontal
plane of the level vehicle through the lens, as core.rig_health draws the horizon: rays above −margin elevation);
and the rest, the ground, which the AE meters. Exports: the mask as PNG (255 = metered), a downsampled metering grid
(cols × rows weights 0…wmax: the metered share of each cell, times the centre weighting when that metering is
chosen) and a list of weighted rectangles approximating it (cells at least half metered, greedy largest rectangles,
corners on even pixels), for ISPs that take weighted AE regions.

ISP interfaces (checked 2026-10-30 against NVIDIA's Jetson Linux API reference): libargus takes weighted regions
through ``Argus::IAutoControlSettings::setAeRegions(const std::vector<Argus::AcRegion>&)``; an ``AcRegion(left, top,
right, bottom, weight)`` is a ``Rectangle<uint32_t>`` in pixel space with a float weight; the most regions a camera
takes is ``ICameraProperties::getMaxAeRegions()`` and the smallest ``getMinAeRegionSize()``. GStreamer's
``nvarguscamerasrc`` has one region (``aeregion="left top right bottom weight"``). libargus drives raw sensors through
the Jetson ISP; a sensor with its own ISP that streams YUV over V4L2 (Sony's ISX031 in the STURDeCAM31 is one) meters
in that ISP. V4L2 has no standard control for per-cell metering weights (V4L2_CID_EXPOSURE_METERING only selects
average, centre-weighted, spot or matrix): the grid is the vendor-neutral form to hand to a vendor's control.

Photometric model (typical, documented)
---------------------------------------
* The sun: core.exposure's NOAA position for the exposure study's place, dates, driving window and headings (sampled
  coarser here: day_step, minutes, heading_step). Daylight samples only (the sun above the horizon); night is its
  own scenario.
* Direct normal illuminance E_dn = 127.5 klux·exp(−0.21·m) (the IES / Ne'eman & Selkowitz clear-sky model: solar
  illuminance constant and luminous extinction coefficient, typical), m the air mass (Kasten & Young 1989).
* The sky: the CIE standard clear sky (CIE S 011/E:2003 type 12, a = −1, b = −0.32, c = 10, d = −3, e = 0.45):
  L(Z, χ) = L_z·f(χ)φ(Z)/(f(Z_s)φ(0)) with φ(Z) = 1 + a·exp(b/cos Z), f(χ) = 1 + c(exp(dχ) − exp(dπ/2)) + e·cos²χ;
  zenith luminance L_z = (1.376·T − 1.81)·tan γ_s + 0.38 kcd/m² (Karayel et al. 1984; T the Linke turbidity, typical
  3). The diffuse horizontal illuminance is that sky integrated over the hemisphere.
* The ground: Lambertian, albedo ρ_g (asphalt, typical 0.12), L = ρ_g·E/π with E = E_dn·sin γ_s + E_dh in the sun
  and E_dh in the vehicle's own shadow (the body and rig's convex outline cast along the sun onto the ground). The
  body and the housings: Lambertian, albedo ρ_b (0.3, a mid-grey car, assumed) and 0.06, lit by half the sky and half
  the ground's reflection (no direct sun: a vertical-surface simplification).
* Scenarios: low sun ahead (the sun low_sun_el_deg above the horizon straight ahead); a tunnel exit (the vehicle in
  a straight tunnel, its portal exit_m ahead of the bumper: inside, a lit road of interior_cd and walls and ceiling of
  wall_cd, typical CIE 88 interior-zone figures; through the portal, the daylit road and the clear sky with the sun
  tunnel_sun_el_deg up ahead); night under street lights (core.weather's night condition: its ambient illuminance on
  the ground plus both headlamps' pattern on the road; the sky 0.01 cd/m², assumed urban sky glow; luminaires as
  sources not modelled).
* Veiling glare: the sun in or near a camera's field (within flare_margin_deg of the image circle, not hidden by the
  body) adds L_v = k·E_lens/θ² (θ in degrees from the sun, at least 1°; E_lens = E_dn·cos α at the lens): the
  Stiles–Holladay form, whose coefficient is 10 for the eye; k = 1 is assumed for a clean coated lens. Every image
  adds a uniform veil of veil_index × its mean luminance (the ISO 9358 veiling glare index, 2 % assumed). The veil
  lifts the black level and lowers contrast.

Auto-exposure, HDR, stitching (typical / assumed)
-------------------------------------------------
* AE meters the masked (or, for comparison, the whole) frame: the mean of log luminance, or a centre-weighted linear
  mean (a Gaussian of 0.35 × the image's shorter side about the principal point, a heuristic), and maps it to the AE
  target (18 % of the long exposure's saturation). Statistics see the sensor's clipped range (below: the long
  exposure's noise floor; above: the merged HDR saturation). Exposure × gain follows the reflected-light meter
  equation N²/(t·g) = L·S/K (K = 12.5 cd·s/m², ISO 2720; S = 100 at unity gain, assumed: the sensitivity is not on
  the module's sheet), with t from one row (1/(fps × rows × 1.1)) to the frame period (or max_exposure_ms) and the
  gain up to max_gain_db (30 dB assumed). EV = log2(L_eff·100/K), the exposure value at ISO 100 that maps L_eff to
  the target; the seam step is the EV difference.
* HDR: the module's stated dynamic range (e-con states 120 dB for the STURDeCAM31 / ISX031) is modelled as a generic
  multi-exposure merge of hdr_exposures sub-exposures hdr_ratio apart (3 × 16, assumed: Sony's scheme for the ISX031
  is not public on the sheets read), each with the rest of the range. At each transition the merged SNR dips by
  10·log10(ratio) dB (shot-noise limited, typical).
* Stitching: the bird's-eye gain compensation is core.stitch.estimate_gains (Brown & Lowe: sigma_n 10 grey levels,
  sigma_g 0.5, gains clipped to 0.4…2.5, overlaps with grey levels outside 8…247 left out), solved here the same way on
  the predicted grey level of each overlapping pair's shared ground (the analysis' coverage grid: equal luminance for
  both cameras). The ISP output is taken as a power law of gamma output_gamma with mid-grey at the AE target and clipped
  at the long exposure's saturation (no highlight roll-off: a heuristic); a gain g then moves the exposure by
  γ·log2(g) EV. The residual seam step is the exposure step plus the gains' correction.

Limits: a flat open ground with no buildings, trees or other shadows than the vehicle's; the vehicle level; clear
sky; Lambertian surfaces; one luminance for each body and housing pixel; colour and white balance not modelled; AE
converged (no AE dynamics or smoothing); the HDR merge and tone curve are assumptions, not the ISX031's.
"""
from __future__ import annotations

import csv
import io
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from .lens import dr_dtheta, r_of_theta, resolve
from .mathutil import BODY_TO_OPTICAL, rot_zyx

SCHEMA = 'vantage.isp/1'
FRAME = 'vehicle (ISO 8855, rear axle): x forward, y left, z up'
SCENARIOS = ('campaign', 'low_sun', 'tunnel', 'night')
SCENARIO_LABEL = {'campaign': 'Campaign', 'low_sun': 'Low sun ahead', 'tunnel': 'Tunnel exit',
                  'night': 'Night, street lights'}
METERING = ('mean_log', 'centre')
METERING_LABEL = {'mean_log': 'mean of log luminance', 'centre': 'centre-weighted mean'}
# mask classes
METERED, OUTSIDE, BODY, HOUSING, SKY = 0, 1, 2, 3, 4
CLASS_NAMES = ('metered ground', 'outside the image circle', 'body and rig', 'housings', 'sky and horizon margin')
CLASS_KEYS = ('metered', 'outside', 'body', 'housing', 'sky')
NEAR_CLIP = 0.05                     # body surface nearer than this is the lens' own module (as core.synth)
OWN_MOUNT_M = 0.05                   # the sun hidden by something at least this far (core.exposure's rule)
METER_COLS = 60                      # metering rays across the image (rows in proportion)
GROUND_FAR_M = 60.0                  # ground nearer than this can lie in the vehicle's shadow
SHADOW_CELL_M = 0.10
CENTRE_SIGMA = 0.35                  # centre weighting: Gaussian sigma, share of the image's shorter side
RECT_MIN_FRACTION = 0.5              # a cell at least this metered may sit in a rectangle

# photometry (typical figures, see the module docstring)
SOLAR_ILLUMINANCE_LX = 127500.0      # IES solar illuminance constant
EXTINCTION_CLEAR = 0.21              # IES clear-sky luminous extinction coefficient
SUN_SOLID_ANGLE_SR = 6.8e-5          # the sun's disc
CIE_CLEAR = (-1.0, -0.32, 10.0, -3.0, 0.45)   # CIE S 011/E:2003 standard sky type 12 (clear, low turbidity)
METER_K = 12.5                       # ISO 2720 reflected-light meter constant, cd·s/m²
NIGHT_SKY_CD = 0.01                  # urban night sky glow (assumed)
HOUSING_ALBEDO = 0.06                # dark plastic and metal
STILES_HOLLADAY_EYE = 10.0           # cd·m⁻²·deg² per lx, the eye's (for reference)
ROW_BLANKING = 1.1                   # frame time = rows × line time × this (vertical blanking, assumed)
# core.stitch.estimate_gains (read from its code: keep in step)
GAIN_RANGE = (0.4, 2.5)
STITCH_SIGMA_N, STITCH_SIGMA_G = 10.0, 0.5
VALID_LEVELS = (8.0, 247.0)
STITCH_RES_M, STITCH_STEP = 0.02, 2   # overlap pixels as estimate_gains counts them (2 cm grid, every second)
MIN_OVERLAP = 50

ASSUMPTIONS = ('Clear sky (CIE standard clear sky with Karayel\'s zenith luminance; IES direct illuminance), a flat '
               'open Lambertian ground of the set albedo with only the vehicle\'s own shadow, the vehicle level; '
               'body and housings lit by half the sky and half the ground (no direct sun). Veiling glare: the '
               'Stiles–Holladay form with an assumed lens coefficient plus a uniform veiling glare index. AE '
               'converged; the reflected-light meter equation with an assumed sensitivity of ISO 100 at unity gain; '
               'a generic multi-exposure HDR merge (sub-exposure ratio and count assumed, not the ISX031\'s '
               'published scheme); a power-law output with mid-grey at the AE target, clipped at the long exposure\'s '
               'saturation. Scenario figures (tunnel luminances, night sky) are typical. Colour and white balance '
               'are not modelled.')
LIMITS_TEXT = ('Typical models, not measurements: no buildings, trees, other vehicles or their shadows; no AE '
               'dynamics; the predicted EV steps compare cameras with each other and are less reliable in absolute '
               'terms than in difference.')

LIMITS = {'grid_cols': (2, 64), 'grid_rows': (2, 64), 'weight_max': (1, 255), 'max_regions': (1, 64),
          'horizon_margin_deg': (0.0, 20.0), 'mask_scale': (1, 8), 'day_step': (1, 366), 'minutes': (5.0, 180.0),
          'heading_step': (5.0, 180.0), 'ground_albedo': (0.02, 0.6), 'body_albedo': (0.02, 0.95),
          'turbidity': (1.5, 8.0), 'f_number': (0.0, 32.0), 'iso_base': (10.0, 3200.0), 'max_gain_db': (0.0, 60.0),
          'max_exposure_ms': (0.0, 1000.0), 'ae_target': (0.02, 0.8), 'output_gamma': (1.0, 3.0),
          'hdr_exposures': (1, 4), 'hdr_ratio': (2.0, 64.0), 'glare_k': (0.0, 20.0), 'veil_index': (0.0, 0.2),
          'flare_margin_deg': (0.0, 45.0), 'low_sun_el_deg': (1.0, 30.0), 'tunnel_sun_el_deg': (2.0, 80.0),
          'exit_m': (-10.0, 500.0), 'tunnel_width_m': (4.0, 40.0), 'tunnel_height_m': (3.0, 15.0),
          'interior_cd': (0.1, 500.0), 'wall_cd': (0.1, 500.0), 'seam_limit_ev': (0.1, 6.0),
          'body_limit': (0.01, 1.0)}
INTS = ('grid_cols', 'grid_rows', 'weight_max', 'max_regions', 'mask_scale', 'day_step', 'hdr_exposures')


@dataclass
class IspSettings:
    scenario: str = 'campaign'         # the scenario the page shows (every scenario is computed)
    metering: str = 'mean_log'         # mean_log | centre
    grid_cols: int = 16                # metering grid
    grid_rows: int = 16
    weight_max: int = 15               # grid weights 0…weight_max
    max_regions: int = 8               # rectangles (libargus: ICameraProperties::getMaxAeRegions() is the device's)
    horizon_margin_deg: float = 2.0    # rays above −margin elevation are sky
    mask_scale: int = 1                # mask resolution: sensor / this
    day_step: int = 30                 # the campaign, sampled coarser than the exposure study
    minutes: float = 30.0
    heading_step: float = 45.0         # used when the exposure study's headings are uniform
    ground_albedo: float = 0.12        # asphalt (typical)
    body_albedo: float = 0.30          # a mid-grey car (assumed)
    turbidity: float = 3.0             # Linke turbidity of the clear sky (typical)
    f_number: float = 0.0              # 0: the lens's (F2.0 when it gives none)
    iso_base: float = 100.0            # ISO-equivalent sensitivity at unity gain (assumed)
    max_gain_db: float = 30.0          # analogue + digital (assumed)
    max_exposure_ms: float = 0.0       # 0: the frame period
    ae_target: float = 0.18            # metered mean → this share of the long exposure's saturation
    output_gamma: float = 2.2
    hdr_exposures: int = 3             # sub-exposures (assumed)
    hdr_ratio: float = 16.0            # between sub-exposures (assumed)
    glare_k: float = 1.0               # Stiles–Holladay-type coefficient of the lens, cd·m⁻²·deg²/lx (assumed)
    veil_index: float = 0.02           # uniform veiling glare index (assumed)
    flare_margin_deg: float = 10.0     # the sun this close outside the image circle still veils
    low_sun_el_deg: float = 8.0
    tunnel_sun_el_deg: float = 40.0
    exit_m: float = 3.0                # tunnel portal ahead of the front bumper (negative: behind it)
    tunnel_width_m: float = 10.0
    tunnel_height_m: float = 5.5
    interior_cd: float = 5.0           # tunnel road luminance, cd/m² (CIE 88 interior zone, typical)
    wall_cd: float = 3.0               # tunnel walls and ceiling, cd/m² (typical)
    night_condition: str = 'night_lit'   # core.weather's condition for the night scenario
    seam_limit_ev: float = 1.0         # a residual seam step above this is a finding
    body_limit: float = 0.30           # the body above this share of the unmasked metering is a finding

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d) -> 'IspSettings':
        """Tolerant: unknown keys dropped, numbers clamped to LIMITS, bad values left at the defaults."""
        s = IspSettings()
        d = d if isinstance(d, dict) else {}
        for k, (lo, hi) in LIMITS.items():
            if k not in d or isinstance(d[k], bool):
                continue
            try:
                v = float(d[k])
            except (TypeError, ValueError):
                continue
            if not math.isfinite(v):
                continue
            v = min(max(v, lo), hi)
            setattr(s, k, int(round(v)) if k in INTS else v)
        if d.get('scenario') in SCENARIOS:
            s.scenario = d['scenario']
        if d.get('metering') in METERING:
            s.metering = d['metering']
        nc = d.get('night_condition')
        if isinstance(nc, str) and nc.strip():
            s.night_condition = nc.strip()[:40]
        return s


def settings_of(project) -> IspSettings:
    return IspSettings.from_dict(getattr(getattr(project, 'analysis', None), 'isp', None))


# =========================================================================== daylight
def air_mass(el_deg):
    """Relative optical air mass at solar elevation el (degrees), Kasten & Young (1989); inf below the horizon."""
    el = np.asarray(el_deg, float)
    with np.errstate(invalid='ignore', divide='ignore'):
        m = 1.0 / (np.sin(np.radians(el)) + 0.50572 * np.power(np.maximum(el + 6.07995, 1e-9), -1.6364))
    return np.where(el > -0.5, m, np.inf)


def direct_normal_lx(el_deg):
    """Clear-sky direct normal illuminance (lux) of the sun at elevation el (IES model, typical)."""
    el = np.asarray(el_deg, float)
    return np.where(el > 0, SOLAR_ILLUMINANCE_LX * np.exp(-EXTINCTION_CLEAR * air_mass(el)), 0.0)


def zenith_luminance(el_deg, turbidity=3.0):
    """Clear-sky zenith luminance (cd/m²), Karayel et al. (1984): (1.376 T − 1.81) tan γ + 0.38 kcd/m²."""
    el = np.clip(np.asarray(el_deg, float), 0.0, 89.0)
    return np.maximum((1.376 * turbidity - 1.81) * np.tan(np.radians(el)) + 0.38, 0.0) * 1000.0


def cie_relative(zenith_rad, chi_rad, sun_zenith_rad, params=CIE_CLEAR):
    """Relative luminance L/L_z of a sky element at zenith angle Z and angular distance χ from the sun, sun zenith
    angle Z_s (radians), for a CIE S 011 standard sky (a, b, c, d, e): f(χ)φ(Z) / (f(Z_s)φ(0))."""
    a, b, c, d, e = params
    Z = np.asarray(zenith_rad, float)
    chi = np.asarray(chi_rad, float)
    Zs = np.asarray(sun_zenith_rad, float)

    def f(x):
        return 1.0 + c * (np.exp(d * x) - math.exp(d * math.pi / 2)) + e * np.cos(x) ** 2
    with np.errstate(divide='ignore', over='ignore'):
        phi = 1.0 + a * np.exp(b / np.maximum(np.cos(Z), 1e-9))
    phi0 = 1.0 + a * math.exp(b)
    return f(chi) * phi / (f(Zs) * phi0)


_DIFFUSE = {}


def _diffuse_table(params=CIE_CLEAR):
    """∫ L/L_z · cos Z dω over the sky hemisphere for solar elevations 0…90° every 0.5° (sr)."""
    if params in _DIFFUSE:
        return _DIFFUSE[params]
    nz, na = 90, 180
    Z = (np.arange(nz) + 0.5) * (math.pi / 2) / nz
    A = (np.arange(na) + 0.5) * math.pi / na                        # 0…π, mirrored
    dZ, dA = (math.pi / 2) / nz, math.pi / na
    Zg, Ag = np.meshgrid(Z, A, indexing='ij')
    els = np.arange(0.0, 90.0 + 1e-9, 0.5)
    out = np.zeros(len(els))
    for k, e in enumerate(els):
        zs = math.radians(90.0 - e)
        cchi = np.cos(Zg) * math.cos(zs) + np.sin(Zg) * math.sin(zs) * np.cos(Ag)
        rel = cie_relative(Zg, np.arccos(np.clip(cchi, -1, 1)), zs, params)
        out[k] = 2.0 * float((rel * np.cos(Zg) * np.sin(Zg)).sum() * dZ * dA)
    _DIFFUSE[params] = (els, out)
    return els, out


def daylight(el_deg, turbidity=3.0) -> dict:
    """Clear-sky illuminances (lux) and zenith luminance (cd/m²) at solar elevations el: E_dn (direct normal),
    E_dh (diffuse horizontal, the CIE clear sky integrated), E_h (global horizontal)."""
    el = np.asarray(el_deg, float)
    els, tab = _diffuse_table()
    lz = zenith_luminance(el, turbidity)
    edh = np.where(el > 0, lz * np.interp(np.clip(el, 0, 90), els, tab), 0.0)
    edn = direct_normal_lx(el)
    return {'E_dn': edn, 'E_dh': edh, 'E_h': edn * np.sin(np.radians(np.maximum(el, 0.0))) + edh, 'L_z': lz}


def sky_luminance(dirs, sun_dir, el_deg, turbidity=3.0):
    """Clear-sky luminance (cd/m²) of directions dirs (..., P, 3) (z up) for the sun at unit vector(s) sun_dir
    (..., 3) and elevation el (...,); 0 below the horizon."""
    dirs = np.asarray(dirs, float)
    s = np.asarray(sun_dir, float)
    el = np.asarray(el_deg, float)
    Z = np.arccos(np.clip(dirs[..., 2], -1, 1))
    chi = np.arccos(np.clip(np.einsum('...pk,...k->...p', dirs, s), -1, 1))
    zs = np.radians(90.0 - el)[..., None]
    L = zenith_luminance(el, turbidity)[..., None] * cie_relative(Z, chi, zs)
    return np.where(dirs[..., 2] > 0, L, 0.0)


# =========================================================================== auto-exposure
def hdr_levels(st: IspSettings, dr_db: float) -> dict:
    """The merged HDR range in AE-relative signal x (the long exposure saturates at x = 1, the AE target sits at
    ae_target): floor (the long exposure's noise floor), top (the shortest's saturation) and the transitions."""
    n = max(int(st.hdr_exposures), 1)
    span = 20.0 * math.log10(st.hdr_ratio) * (n - 1)
    dr = dr_db if dr_db > 0 else 72.0 + span
    single = max(dr - span, 20.0)
    return {'floor': 10 ** (-single / 20.0), 'top': st.hdr_ratio ** (n - 1), 'dr_db': dr, 'single_db': single,
            'transitions': [st.hdr_ratio ** k for k in range(n - 1)], 'snr_dip_db': 10.0 * math.log10(st.hdr_ratio),
            'dr_given': dr_db > 0}


def ae_solve(L, w, mode='mean_log', lo=1e-5, hi=256.0, target=0.18, iters=40):
    """The luminance L_eff (cd/m², returned as log2, shape (S,)) the AE maps to the target: the weighted statistic of
    the relative signal x = target·L/L_eff, clipped to [lo, hi], equals target. L (S, P) luminance, w (P,) or (S, P)
    weights. mode: 'mean_log' (geometric mean) or 'centre' (linear mean; the centre weighting is in w)."""
    L = np.asarray(L, np.float64)
    w = np.broadcast_to(np.asarray(w, np.float64), L.shape)
    ws = np.maximum(w.sum(1), 1e-300)
    lg = np.log2(np.maximum(L, 1e-12))
    lt, llo, lhi = math.log2(target), math.log2(lo), math.log2(hi)
    on = w > 0
    # without clipping the answer is closed form; bisect only the rows where a weighted ray clips
    if mode == 'mean_log':
        u0 = (w * lg).sum(1) / ws
    else:
        u0 = np.log2(np.maximum((w * L).sum(1) / ws, 1e-300))
    y0 = lt + lg - u0[:, None]
    rows = np.nonzero((((y0 < llo) | (y0 > lhi)) & on).any(1))[0]
    if not len(rows):
        return u0
    lg, w, ws, on = lg[rows], w[rows], ws[rows], on[rows]
    # a bracket that holds the answer: at a every ray clips high (statistic > target), at b every ray clips low
    a = np.where(on, lg, np.inf).min(1) - (lhi - lt) - 1.0
    b = np.where(on, lg, -np.inf).max(1) + (lt - llo) + 1.0
    a = np.where(np.isfinite(a), a, 0.0)
    b = np.where(np.isfinite(b), b, 0.0)
    for _ in range(iters):
        u = 0.5 * (a + b)
        y = np.clip(lt + lg - u[:, None], llo, lhi)
        s = (w * y).sum(1) / ws if mode == 'mean_log' else np.log2(np.maximum((w * np.exp2(y)).sum(1) / ws, 1e-300))
        up = s > lt                                    # still brighter than the target: a higher L_eff
        a = np.where(up, u, a)
        b = np.where(up, b, u)
    out = u0.copy()
    out[rows] = 0.5 * (a + b)
    return out


def ev100(log2_leff):
    """Exposure value at ISO 100 that maps luminance L_eff to the AE target: log2(L_eff·100/K)."""
    return np.asarray(log2_leff, float) + math.log2(100.0 / METER_K)


def exposure_setting(log2_leff, f_number, fps, rows, st: IspSettings) -> dict:
    """Exposure time and gain for the wanted L_eff within the module's limits: t·g = N²K/(L_eff·S), t from one row
    to the frame period (or max_exposure_ms), g from 1 to max_gain_db. Returns arrays: t_s, gain, the achieved
    log2 L_eff and the shortfall in EV (positive: darker than wanted, negative: brighter)."""
    leff = np.exp2(np.asarray(log2_leff, float))
    H = f_number ** 2 * METER_K / (leff * st.iso_base)
    t_max = 1.0 / max(fps, 1e-3)
    if st.max_exposure_ms > 0:
        t_max = min(t_max, st.max_exposure_ms / 1000.0)
    t_min = 1.0 / (max(fps, 1e-3) * max(rows, 1) * ROW_BLANKING)
    t = np.clip(H, t_min, t_max)
    g = np.clip(H / t, 1.0, 10 ** (st.max_gain_db / 20.0))
    Ha = t * g
    la = np.log2(f_number ** 2 * METER_K / (Ha * st.iso_base))
    return {'t_s': t, 'gain': g, 'log2_leff': la, 'short_ev': np.log2(H / Ha)}


# =========================================================================== stitching gains
def balance_gains(levels_a, levels_b, pairs, n, sigma_n=STITCH_SIGMA_N, sigma_g=STITCH_SIGMA_G):
    """core.stitch.estimate_gains' solve, vectorised over samples, for one grey channel: levels_a/levels_b (S, K) the
    grey levels of cameras i and j on pair k's shared ground, pairs [(i, j, N)]. Pairs with a level outside 8…247 are
    left out. Returns gains (S, n), clipped to GAIN_RANGE."""
    Ia = np.asarray(levels_a, float)
    Ib = np.asarray(levels_b, float)
    S = Ia.shape[0]
    A = np.zeros((S, n, n))
    b = np.zeros((S, n))
    ok = (Ia > VALID_LEVELS[0]) & (Ia < VALID_LEVELS[1]) & (Ib > VALID_LEVELS[0]) & (Ib < VALID_LEVELS[1])
    for k, (i, j, N) in enumerate(pairs):
        Nn = N * ok[:, k]
        A[:, i, i] += Nn * (Ia[:, k] ** 2 / sigma_n ** 2 + 1 / sigma_g ** 2)
        A[:, j, j] += Nn * (Ib[:, k] ** 2 / sigma_n ** 2 + 1 / sigma_g ** 2)
        A[:, i, j] -= Nn * Ia[:, k] * Ib[:, k] / sigma_n ** 2
        A[:, j, i] -= Nn * Ia[:, k] * Ib[:, k] / sigma_n ** 2
        b[:, i] += Nn / sigma_g ** 2
        b[:, j] += Nn / sigma_g ** 2
    idx = np.arange(n)
    zero = A[:, idx, idx] == 0
    A[:, idx, idx] = np.where(zero, 1.0, A[:, idx, idx])
    b = np.where(zero, 1.0, b)
    g = np.linalg.solve(A, b[..., None])[..., 0]
    return np.clip(g, *GAIN_RANGE), ok


def grey_level(x, gamma):
    """8-bit output grey level of relative signal x (power law, clipped at the long exposure's saturation)."""
    return 255.0 * np.clip(np.asarray(x, float), 0.0, 1.0) ** (1.0 / gamma)


# =========================================================================== masks
def _cam_frames(cam):
    R = rot_zyx(cam.yaw, cam.pitch, cam.roll)
    return R, BODY_TO_OPTICAL @ R.T


def _finite_pose(o) -> bool:
    return all(math.isfinite(float(getattr(o, k))) for k in ('x', 'y', 'z', 'yaw', 'pitch', 'roll'))


def classify_rays(d_opt, theta, intr, W2O, pos, cube, margin_deg):
    """Mask class of optical-frame rays (K, 3): OUTSIDE the image circle, BODY (body and rig), HOUSING, SKY (above
    −margin elevation, vehicle frame), else METERED. Also returns the vehicle-frame directions and the ground
    distance along each ray (inf when it does not reach the ground)."""
    dw = d_opt @ W2O                                   # optical -> vehicle
    valid = theta <= intr.max_theta
    with np.errstate(divide='ignore', invalid='ignore'):
        tg = np.where(dw[:, 2] < -1e-6, -pos[2] / dw[:, 2], np.inf)
    cls = np.full(len(dw), METERED, np.uint8)
    if cube is not None:
        bd, _, hid = cube.lookup_ids(dw)
        near = valid & np.isfinite(bd) & (bd < tg) & ((bd > NEAR_CLIP) | (hid >= 0))
        cls[near & (hid < 0)] = BODY
        cls[near & (hid >= 0)] = HOUSING
    else:
        near = np.zeros(len(dw), bool)
    sky = valid & ~near & (dw[:, 2] > -math.sin(math.radians(margin_deg)))
    cls[sky] = SKY
    cls[~valid] = OUTSIDE
    return cls, dw, tg


def _pixel_grid(intr, w, h):
    us = (np.arange(w) + 0.5) / w * intr.width
    vs = (np.arange(h) + 0.5) / h * intr.height
    U, V = np.meshgrid(us, vs)
    d, th = intr.unproject(U.ravel(), V.ravel())
    return U.ravel(), V.ravel(), d, th


def camera_mask(project, engine, cam, margin_deg=2.0, w=None, h=None, vg=None) -> np.ndarray:
    """(h, w) uint8 mask classes of a camera's image (sensor resolution by default) through its lens model."""
    intr = resolve(project.module_for(cam), project.lens_for(cam))
    w, h = int(w or intr.width), int(h or intr.height)
    _, W2O = _cam_frames(cam)
    cube = _cube(engine, cam.pos(), vg)
    out = np.empty(w * h, np.uint8)
    rows = max(1, 400_000 // w)                        # in bands: bounded memory at full resolution
    for r0 in range(0, h, rows):
        r1 = min(h, r0 + rows)
        us = (np.arange(w) + 0.5) / w * intr.width
        vs = (np.arange(r0, r1) + 0.5) / h * intr.height
        U, V = np.meshgrid(us, vs)
        d, th = intr.unproject(U.ravel(), V.ravel())
        cls, _, _ = classify_rays(d, th, intr, W2O, cam.pos(), cube, margin_deg)
        out[r0 * w:r1 * w] = cls
    return out.reshape(h, w)


def _cube(engine, pos, vg):
    if engine is None or vg is None:
        return None
    return engine.cube(pos, vg)


def horizon_lines(intr, W2O, step_deg=0.5) -> list:
    """The horizon (the level vehicle's horizontal plane) through the lens: [(N, 2)] sensor-pixel polylines."""
    az = np.radians(np.arange(0.0, 360.0, step_deg))
    dirs = np.c_[np.cos(az), np.sin(az), np.zeros_like(az)]
    u, v, _, ok = intr.project_dirs(dirs @ W2O.T)
    lines, cur = [], []
    for a, b, o in zip(u, v, ok):
        if o:
            cur.append((float(a), float(b)))
        elif cur:
            lines.append(cur)
            cur = []
    if cur:
        if lines and ok[0]:
            lines[0] = cur + lines[0]
        else:
            lines.append(cur)
    return [np.array(ln) for ln in lines if len(ln) > 1]


def centre_weights(U, V, intr):
    """Centre weighting of pixels (heuristic): Gaussian of CENTRE_SIGMA × the shorter image side about the principal
    point."""
    s = CENTRE_SIGMA * min(intr.width, intr.height)
    return np.exp(-0.5 * ((U - intr.cx) ** 2 + (V - intr.cy) ** 2) / (s * s))


def metering_grid(mask, cols, rows, wmax, cweight=None) -> np.ndarray:
    """(rows, cols) int weights 0…wmax: each cell's metered share (mask == METERED), times the centre weighting
    (rows, cols, 0…1, normalised to its maximum) when given."""
    h, w = mask.shape
    ys = np.linspace(0, h, rows + 1).round().astype(int)
    xs = np.linspace(0, w, cols + 1).round().astype(int)
    m = (mask == METERED).astype(np.float64)
    cs = np.pad(m.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    tot = cs[ys[1:]][:, xs[1:]] - cs[ys[:-1]][:, xs[1:]] - cs[ys[1:]][:, xs[:-1]] + cs[ys[:-1]][:, xs[:-1]]
    area = np.outer(np.diff(ys), np.diff(xs)).astype(float)
    frac = tot / np.maximum(area, 1)
    if cweight is not None:
        frac = frac * cweight / max(float(cweight.max()), 1e-12)
    return np.clip(np.round(frac * wmax), 0, wmax).astype(int), tot / np.maximum(area, 1)


def _largest_rect(B):
    """The largest all-True axis-aligned rectangle of a bool matrix: (r0, c0, r1, c1) exclusive, or None."""
    rows, cols = B.shape
    hgt = np.zeros(cols, int)
    best, area = None, 0
    for r in range(rows):
        hgt = np.where(B[r], hgt + 1, 0)
        stack = []
        for c in range(cols + 1):
            hc = hgt[c] if c < cols else 0
            start = c
            while stack and stack[-1][1] >= hc:
                s, hs = stack.pop()
                if hs * (c - s) > area:
                    area, best = hs * (c - s), (r - hs + 1, s, r + 1, c)
                start = s
            stack.append((start, hc))
    return best if area > 0 else None


def regions(grid, frac, size, wmax, max_regions) -> list:
    """Weighted rectangles approximating the metered area: greedy largest rectangles over the cells at least half
    metered. [{'left', 'top', 'right', 'bottom' (sensor px, even), 'weight' (0…1)}], largest first."""
    W, H = size
    rows, cols = grid.shape
    free = frac >= RECT_MIN_FRACTION
    ys = np.linspace(0, H, rows + 1)
    xs = np.linspace(0, W, cols + 1)
    out = []
    for _ in range(max_regions):
        r = _largest_rect(free)
        if r is None:
            break
        r0, c0, r1, c1 = r
        free[r0:r1, c0:c1] = False
        left = int(math.floor(xs[c0] / 2) * 2)
        top = int(math.floor(ys[r0] / 2) * 2)
        right = min(int(math.ceil(xs[c1] / 2) * 2), int(W))
        bottom = min(int(math.ceil(ys[r1] / 2) * 2), int(H))
        wt = float(grid[r0:r1, c0:c1].mean()) / wmax
        out.append({'left': left, 'top': top, 'right': right, 'bottom': bottom, 'weight': round(wt, 4),
                    'cells': [int(c0), int(r0), int(c1), int(r1)]})
    return out


# =========================================================================== per camera geometry
@dataclass
class CamGeom:
    uid: str
    tag: str
    slot: int
    model: str
    intr: object
    W2O: np.ndarray
    pos: np.ndarray
    axis: np.ndarray               # optical axis, vehicle frame
    fps: float
    f_number: float
    dr_db: float
    cube: object
    mask: np.ndarray               # (h, w) classes at mask resolution
    shares: dict                   # pixel share of each class (whole frame)
    dirs: np.ndarray               # (P, 3) metering rays, vehicle frame
    cls: np.ndarray                # (P,)
    tg: np.ndarray                 # (P,) distance to the ground along each ray (inf: none)
    gxy: np.ndarray                # (P, 2) ground point (nan: none)
    cw: np.ndarray                 # (P,) centre weight
    pix_per_ray: float             # sensor pixels each metering ray stands for
    horizon: list
    grid: np.ndarray = None
    frac: np.ndarray = None
    rects: list = field(default_factory=list)
    rect_cover: dict = field(default_factory=dict)

    @property
    def size(self):
        return int(self.intr.width), int(self.intr.height)

    def weights(self, masked: bool, mode: str):
        w = self.cw if mode == 'centre' else np.ones(len(self.cls))
        return w * (self.cls == METERED) if masked else w

    def pixel_sr(self, theta):
        """Solid angle (sr) of one sensor pixel at off-axis angle theta."""
        i = self.intr
        th = np.maximum(np.asarray(theta, float), 1e-4)
        r = np.maximum(r_of_theta(th, i.f_mm, i.projection, i.kb), 1e-9)
        dr = np.maximum(dr_dtheta(th, i.f_mm, i.projection, i.kb), 1e-9)
        return i.pixel_mm ** 2 / max(i.aspect, 1e-6) * np.sin(th) / (r * dr)


def camera_geometry(project, engine, cam, st: IspSettings, vg=None) -> CamGeom:
    """Masks, metering rays, grid and rectangles of one camera."""
    module, lens = project.module_for(cam), project.lens_for(cam)
    intr = resolve(module, lens)
    _, W2O = _cam_frames(cam)
    pos = cam.pos()
    cube = _cube(engine, pos, vg)
    W, H = int(intr.width), int(intr.height)
    sc = max(int(st.mask_scale), 1)
    mask = camera_mask(project, engine, cam, st.horizon_margin_deg, max(W // sc, 1), max(H // sc, 1), vg)
    counts = np.bincount(mask.ravel(), minlength=5)[:5] / mask.size
    shares = {k: float(v) for k, v in zip(CLASS_KEYS, counts)}
    mc = METER_COLS
    mr = max(2, int(round(mc * H / W)))
    U, V, d, th = _pixel_grid(intr, mc, mr)
    cls, dw, tg = classify_rays(d, th, intr, W2O, pos, cube, st.horizon_margin_deg)
    gxy = np.where(np.isfinite(tg)[:, None], pos[None, :2] + dw[:, :2] * np.where(np.isfinite(tg), tg, 0)[:, None],
                   np.nan)
    N = float(lens.f_number) if st.f_number <= 0 and lens.f_number and lens.f_number > 0 else \
        (st.f_number if st.f_number > 0 else 2.0)
    g = CamGeom(cam.uid, cam.tag, int(cam.slot), f'{module.name} · {lens.name}', intr, W2O, pos,
                W2O.T @ np.array([0.0, 0.0, 1.0]), float(module.fps or 30.0), N, float(module.dynamic_range_db or 0.0),
                cube, mask, shares, dw, cls, tg, gxy, centre_weights(U, V, intr), W * H / (mc * mr),
                horizon_lines(intr, W2O))
    gc = (np.arange(st.grid_cols) + 0.5) / st.grid_cols * W
    gr = (np.arange(st.grid_rows) + 0.5) / st.grid_rows * H
    GU, GV = np.meshgrid(gc, gr)
    cwg = centre_weights(GU, GV, intr) if st.metering == 'centre' else None
    g.grid, g.frac = metering_grid(mask, st.grid_cols, st.grid_rows, st.weight_max, cwg)
    g.rects = regions(g.grid, g.frac, (W, H), st.weight_max, st.max_regions)
    g.rect_cover = rect_cover(mask, g.rects, (W, H))
    return g


def rect_cover(mask, rects, size) -> dict:
    """Share of the metered pixels the rectangles hold, and the share of the rectangles' pixels not metered."""
    h, w = mask.shape
    W, H = size
    inside = np.zeros((h, w), bool)
    for r in rects:
        inside[int(r['top'] * h / H):int(math.ceil(r['bottom'] * h / H)),
               int(r['left'] * w / W):int(math.ceil(r['right'] * w / W))] = True
    met = mask == METERED
    n = int(met.sum())
    return {'metered_inside': float((met & inside).sum() / n) if n else 0.0,
            'spill': float((inside & ~met).sum() / max(int(inside.sum()), 1))}


# =========================================================================== environments
class _Env:
    """What every ray of a camera sees in one scene: luminances of the open rays (not body, inside the circle), the
    body and housings, the sun (for veil and disc) and the ground at given points."""
    sun_v = None                    # (S, 3) vehicle frame, or None
    E_dn = None                     # (S,)

    def ground_lum(self, xy, sl):          # (S_chunk, M)
        raise NotImplementedError

    def open_lum(self, g: CamGeom, sl):    # (S_chunk, P)
        raise NotImplementedError

    def body_lum(self, sl, albedo):        # (S_chunk,)
        raise NotImplementedError


class DayEnv(_Env):
    def __init__(self, sun_v, el, st: IspSettings, hull_pts=None):
        self.sun_v = np.asarray(sun_v, float).reshape(-1, 3)
        self.el = np.asarray(el, float).ravel()
        self.st = st
        dl = daylight(self.el, st.turbidity)
        self.E_dn, self.E_dh, self.E_h = dl['E_dn'], dl['E_dh'], dl['E_h']
        self.hull_pts = hull_pts             # (V, 3) the body and rig's vertices for the shadow, or None
        self.S = len(self.el)

    def hulls(self, sl):
        """The vehicle's shadow outline on the ground for each sample of the chunk: [(K, 2) or None (sun on the
        horizon: everything near in shade)], cached for the last chunk."""
        key = (sl.start, sl.stop)
        if getattr(self, '_hk', None) == key:
            return self._hulls
        import cv2
        out = []
        P = self.hull_pts
        for k in range(self.S)[sl]:
            s = self.sun_v[k]
            if s[2] <= 0.02:
                out.append(None)
                continue
            G = P[:, :2] - s[None, :2] * (P[:, 2:3] / s[2])
            out.append(cv2.convexHull(G.astype(np.float32)).reshape(-1, 2).astype(float))
        self._hk, self._hulls = key, out
        return out

    def shadow(self, xy, sl):
        """(S_chunk, M) bool: ground points in the vehicle's shadow."""
        n = len(range(self.S)[sl])
        out = np.zeros((n, len(xy)), bool)
        if self.hull_pts is None or not len(xy):
            return out
        xy = np.asarray(xy, float)
        ok = np.isfinite(xy).all(1) & (np.hypot(xy[:, 0], xy[:, 1]) < GROUND_FAR_M)
        for r, H in enumerate(self.hulls(sl)):
            if H is None:
                out[r] = ok
                continue
            lo, hi = H.min(0), H.max(0)
            box = (xy[:, 0] >= lo[0]) & (xy[:, 0] <= hi[0]) & (xy[:, 1] >= lo[1]) & (xy[:, 1] <= hi[1])
            c = np.nonzero(ok & box)[0]
            if not len(c) or len(H) < 3:
                continue
            E = np.roll(H, -1, 0) - H
            cr = E[None, :, 0] * (xy[c, None, 1] - H[None, :, 1]) - E[None, :, 1] * (xy[c, None, 0] - H[None, :, 0])
            out[r, c] = (cr >= 0).all(1) | (cr <= 0).all(1)
        return out

    def ground_lum(self, xy, sl):
        sh = self.last_shadow = self.shadow(xy, sl)
        E = np.where(sh, self.E_dh[sl][:, None], self.E_h[sl][:, None])
        return self.st.ground_albedo * E / math.pi

    def open_lum(self, g, sl):
        P = len(g.cls)
        S = len(self.el[sl])
        L = np.zeros((S, P))
        gr = np.isfinite(g.tg)
        if gr.any():
            L[:, gr] = self.ground_lum(g.gxy[gr], sl)
        up = ~gr
        if up.any():
            L[:, up] = sky_luminance(np.broadcast_to(g.dirs[up], (S,) + g.dirs[up].shape), self.sun_v[sl],
                                     self.el[sl], self.st.turbidity)
        return L

    def body_lum(self, sl, albedo):
        return albedo * (0.5 * self.E_dh[sl] + 0.5 * self.st.ground_albedo * self.E_h[sl]) / math.pi


class TunnelEnv(_Env):
    """The vehicle in a straight tunnel along +x, centred on y = 0; the portal at x = exit_x; daylight beyond."""

    def __init__(self, st: IspSettings, front_x: float):
        self.st = st
        self.exit_x = front_x + st.exit_m
        el = st.tunnel_sun_el_deg
        self.day = DayEnv([[math.cos(math.radians(el)), 0.0, math.sin(math.radians(el))]], [el], st)
        self.sun_v, self.E_dn = self.day.sun_v, self.day.E_dn
        self.out_ground = st.ground_albedo * float(self.day.E_h[0]) / math.pi

    def classify(self, pos, d):
        """For rays from pos along d (K, 3): 0 interior road, 1 interior wall or ceiling, 2 outside ground, 3 sky."""
        st = self.st
        hw, ht = st.tunnel_width_m / 2, st.tunnel_height_m
        with np.errstate(divide='ignore', invalid='ignore'):
            t_floor = np.where(d[:, 2] < -1e-9, -pos[2] / d[:, 2], np.inf)
            t_ceil = np.where(d[:, 2] > 1e-9, (ht - pos[2]) / d[:, 2], np.inf)
            t_wall = np.where(np.abs(d[:, 1]) > 1e-9, (np.sign(d[:, 1]) * hw - pos[1]) / d[:, 1], np.inf)
            t_exit = np.where(d[:, 0] > 1e-9, (self.exit_x - pos[0]) / d[:, 0], np.inf)
        t_in = np.minimum(np.minimum(t_floor, t_ceil), t_wall)
        out = np.where(t_floor <= np.minimum(t_ceil, t_wall), 0, 1)
        through = t_exit < t_in
        return np.where(through, np.where(d[:, 2] < 0, 2, 3), out)

    def ground_lum(self, xy, sl):
        L = np.where(np.asarray(xy)[:, 0] > self.exit_x, self.out_ground, self.st.interior_cd)
        return L[None, :]

    def open_lum(self, g, sl):
        k = self.classify(g.pos, g.dirs)
        sky = sky_luminance(g.dirs[None], self.sun_v, self.day.el, self.st.turbidity)[0]
        L = np.choose(k, [np.full(len(k), self.st.interior_cd), np.full(len(k), self.st.wall_cd),
                          np.full(len(k), self.out_ground), sky])
        return L[None, :]

    def body_lum(self, sl, albedo):
        return np.array([albedo * self.st.interior_cd / self.st.ground_albedo])

    def sun_reaches(self, pos):
        return self.classify(pos, self.sun_v)[0] == 3


class NightEnv(_Env):
    """Night: core.weather's condition (street-light ambient illuminance taken on the ground too) and both
    headlamps on the road; the sky NIGHT_SKY_CD."""

    def __init__(self, st: IspSettings, cond, wst, lamps):
        self.st, self.cond, self.wst, self.lamps = st, cond, wst, lamps
        self.sun_v = None
        self.E_dn = np.zeros(1)

    def ground_lum(self, xy, sl):
        from . import weather as WX
        xy = np.asarray(xy, float)
        P = np.c_[xy, np.zeros(len(xy))]
        E = WX.illuminance(P, self.cond, self.wst, self.lamps, normals=np.broadcast_to([0.0, 0.0, 1.0], P.shape))
        return (self.st.ground_albedo * np.asarray(E, float) / math.pi)[None, :]

    def open_lum(self, g, sl):
        L = np.full(len(g.cls), NIGHT_SKY_CD)
        gr = np.isfinite(g.tg) & (g.tg < 1e4)
        if gr.any():
            L[gr] = self.ground_lum(g.gxy[gr], sl)[0]
        return L[None, :]

    def body_lum(self, sl, albedo):
        return np.array([albedo * float(self.cond.ambient_lux) / math.pi])


# =========================================================================== one camera in one scene
def frame_luminance(g: CamGeom, env: _Env, st: IspSettings, sl=slice(None), sun_ok=None) -> dict:
    """Luminance (cd/m²) of every metering ray of camera g (S_chunk, P), with the veil added, plus the sun's disc as
    an extra ray (its weight in rays, 0 when not in the image). sun_ok (S,) bool: the sun reaches the lens."""
    L = env.open_lum(g, sl).astype(np.float64)
    S = L.shape[0]
    L = np.array(np.broadcast_to(L, (S, len(g.cls))))
    bl = np.broadcast_to(env.body_lum(sl, st.body_albedo), (S,))
    hl = np.broadcast_to(env.body_lum(sl, HOUSING_ALBEDO), (S,))
    L[:, g.cls == BODY] = bl[:, None]
    L[:, g.cls == HOUSING] = hl[:, None]
    L[:, g.cls == OUTSIDE] = 0.0
    inside = g.cls != OUTSIDE
    disc_w = np.zeros(S)
    disc_L = np.zeros(S)
    glare = np.zeros((S, len(g.cls)))
    glare_e = np.zeros(S)
    sun_angle = np.full(S, np.nan)
    sv = np.zeros((S, 3))
    if env.sun_v is not None:
        sv = env.sun_v[sl] if len(env.sun_v) > 1 else np.broadcast_to(env.sun_v, (S, 3))
        edn = env.E_dn[sl] if len(env.E_dn) > 1 else np.broadcast_to(env.E_dn, (S,))
        ok = np.ones(S, bool) if sun_ok is None else np.broadcast_to(sun_ok, (S,))
        if g.cube is not None:
            depth, _ = g.cube.lookup(sv)
            ok = ok & ~(np.isfinite(depth) & (depth >= OWN_MOUNT_M))
        u, v, th, img = g.intr.project_dirs(sv @ g.W2O.T)
        alpha = np.arccos(np.clip(sv @ g.axis, -1, 1))
        sun_angle = np.degrees(alpha)
        near = ok & (alpha <= g.intr.max_theta + math.radians(st.flare_margin_deg)) & (edn > 0)
        if near.any() and st.glare_k > 0:
            glare_e = np.where(near, st.glare_k * edn * np.maximum(np.cos(alpha), 0.0), 0.0)
            glare[near] = glare_e[near, None] / veil_angle(sv[near], g.dirs) ** 2
        disc = ok & img & (edn > 0)
        if disc.any():
            disc_w[disc] = SUN_SOLID_ANGLE_SR / g.pixel_sr(th[disc]) / g.pix_per_ray
            disc_L[disc] = edn[disc] / SUN_SOLID_ANGLE_SR
    n_in = max(int(inside.sum()), 1)
    mean_in = ((L * inside).sum(1) + disc_w * disc_L) / (n_in + disc_w)
    veil = st.veil_index * mean_in[:, None] + glare
    L = L + np.where(inside, veil, 0.0)
    return {'L': L, 'disc_w': disc_w, 'disc_L': disc_L, 'veil': np.where(inside, veil, 0.0),
            'sun_angle': sun_angle, 'uniform': st.veil_index * mean_in, 'glare_e': glare_e, 'sun': np.array(sv)}


def veil_angle(sun, dirs):
    """Degrees between the sun (S, 3) and directions (P, 3), at least 1° (the glare formula's inner limit)."""
    return np.maximum(np.degrees(np.arccos(np.clip(np.asarray(sun) @ np.asarray(dirs).T, -1, 1))), 1.0)


def veil_at(fl, dirs):
    """The veil (cd/m²) a camera adds in vehicle-frame directions dirs (P, 3): (S, P)."""
    return fl['uniform'][:, None] + fl['glare_e'][:, None] / veil_angle(fl['sun'], dirs) ** 2


def meter(g: CamGeom, fl: dict, st: IspSettings, masked: bool, lv: dict):
    """log2 L_eff (S,) the AE settles on: masked metering, or the whole frame (the sun's disc included)."""
    w = g.weights(masked, st.metering)
    L = fl['L']
    if not masked:
        S = L.shape[0]
        wd = fl['disc_w'] * (float(np.mean(g.cw)) if st.metering == 'centre' else 1.0)   # centre: a mean weight
        L = np.c_[L, fl['disc_L']]
        w = np.c_[np.broadcast_to(w, (S, len(w))), wd]
    return ae_solve(L, w, st.metering, lv['floor'], lv['top'], st.ae_target)


# =========================================================================== result
@dataclass
class IspResult:
    settings: IspSettings
    variant: str
    variant_uid: str
    cams: list                     # [dict] per camera (see run)
    seams: list                    # [dict] per seam
    scenarios: dict                # scenario -> {'cams': {uid: {...}}, 'seams': [{...}], 'label': str, ...}
    campaign: dict                 # sampling, curves, worst cases
    findings: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    geoms: dict = field(default_factory=dict)      # uid -> CamGeom (masks for export and drawing)
    previews: dict = field(default_factory=dict)   # uid -> {'rgb': (h, w, 3) uint8, 'mask': (h, w) uint8}

    def cam(self, key):
        return next((c for c in self.cams if key in (c['uid'], c['tag'])), None)

    def seam(self, key):
        return next((s for s in self.seams if s['key'] == key), None)


def _project_for(project, variant):
    if variant is None or variant is project.variant:
        return project
    import copy
    q = copy.copy(project)
    q.variants = list(project.variants) + ([] if any(v is variant for v in project.variants) else [variant])
    q.active_uid = variant.uid
    return q


def _overlaps(res, uids, max_pts=64):
    """Every pair of cameras that share ground in the analysis grid: [(i, j, N stitch pixels, points (K, 2))]."""
    if res is None:
        return []
    g = res.grid
    X, Y = np.meshgrid(g.xs, g.ys)
    free = g.roi & ~g.body
    vis = {c.uid: c.vis & free for c in res.cams}
    scale = (g.cell / (STITCH_RES_M * STITCH_STEP)) ** 2
    out = []
    for i, a in enumerate(uids):
        for j in range(i + 1, len(uids)):
            b = uids[j]
            if a not in vis or b not in vis:
                continue
            m = vis[a] & vis[b]
            n = int(m.sum())
            if n * scale < MIN_OVERLAP:
                continue
            jj, ii = np.nonzero(m)
            k = np.linspace(0, n - 1, min(max_pts, n)).round().astype(int)
            out.append((i, j, n * scale, np.c_[X[jj[k], ii[k]], Y[jj[k], ii[k]]]))
    return out


def _seam_list(res, uids):
    if res is None:
        return []
    idx = {u: i for i, u in enumerate(uids)}
    return [(idx[s.a], idx[s.b], float(s.angle)) for s in res.seams if s.a in idx and s.b in idx]


def _hull_points(vg):
    if vg is None:
        return None
    m = vg.occ_mesh if getattr(vg, 'occ_mesh', None) is not None else vg.mesh
    V = np.asarray(m.vertices, float)
    V = V[V[:, 2] > 0.02]
    if len(V) > 4000:
        V = V[np.linspace(0, len(V) - 1, 4000).round().astype(int)]
    return V if len(V) >= 3 else None


def _scene(geoms, env, st, levels, sl, pairs, sun_ok_of=None):
    """AE of every camera and the grey levels on every overlap for one environment chunk: dict of arrays."""
    out = {'masked': [], 'unmasked': [], 'veil': [], 'shadow': [], 'sun_angle': [], 'fl': []}
    for g in geoms:
        sok = sun_ok_of(g) if sun_ok_of is not None else None
        fl = frame_luminance(g, env, st, sl, sok)
        lv = levels[g.uid]
        m = meter(g, fl, st, True, lv)
        u = meter(g, fl, st, False, lv)
        out['masked'].append(m)
        out['unmasked'].append(u)
        wm = g.weights(True, st.metering)
        out['veil'].append((fl['veil'] * wm).sum(1) / max(wm.sum(), 1e-12))
        out['sun_angle'].append(fl['sun_angle'])
        out['fl'].append(fl)
        if isinstance(env, DayEnv):
            gr = np.isfinite(g.tg)                       # open_lum just tested these rays
            sh = env.last_shadow
            ww = wm[gr]
            out['shadow'].append((sh * ww).sum(1) / max(ww.sum(), 1e-12))
        else:
            out['shadow'].append(np.zeros(len(m)))
    S = len(out['masked'][0])
    la, lb = [], []
    for i, j, _, pts in pairs:
        gl = np.broadcast_to(env.ground_lum(pts, sl), (S, len(pts)))
        P3 = np.c_[pts, np.zeros(len(pts))]
        for k, side in ((i, la), (j, lb)):
            d = P3 - geoms[k].pos
            d /= np.linalg.norm(d, axis=1, keepdims=True)
            side.append((gl + veil_at(out['fl'][k], d)).mean(1))
    out['pair_La'] = np.stack(la, 1) if la else np.zeros((S, 0))
    out['pair_Lb'] = np.stack(lb, 1) if lb else np.zeros((S, 0))
    for k in ('masked', 'unmasked', 'veil', 'shadow', 'sun_angle'):
        out[k] = np.stack(out[k], 1)
    return out


def _settle(geoms, sc, st):
    """Exposure settings within each module's limits, the achieved L_eff and EV per camera: arrays (S, n)."""
    n = len(geoms)
    t = np.zeros_like(sc['masked'])
    gain = np.ones_like(t)
    la = np.zeros_like(t)
    short = np.zeros_like(t)
    for i, g in enumerate(geoms):
        e = exposure_setting(sc['masked'][:, i], g.f_number, g.fps, g.intr.height, st)
        t[:, i], gain[:, i], la[:, i], short[:, i] = e['t_s'], e['gain'], e['log2_leff'], e['short_ev']
    return {'t_s': t, 'gain': gain, 'log2_leff': la, 'short_ev': short, 'ev': ev100(la), 'n': n}


def _seam_steps(ex, sc, pairs, seams, st):
    """Seam steps before and after the stitcher's gains (EV, +: camera a's image of the shared ground brighter),
    whether a side clipped, and the gains."""
    la = ex['log2_leff']
    if not pairs:
        z = np.zeros((la.shape[0], len(seams)))
        return z, z.copy(), np.zeros_like(z, bool), np.ones((la.shape[0], la.shape[1]))
    I = np.array([p[0] for p in pairs])
    J = np.array([p[1] for p in pairs])
    lxa = math.log2(st.ae_target) + np.log2(np.maximum(sc['pair_La'], 1e-12)) - la[:, I]
    lxb = math.log2(st.ae_target) + np.log2(np.maximum(sc['pair_Lb'], 1e-12)) - la[:, J]
    Va, Vb = grey_level(np.exp2(lxa), st.output_gamma), grey_level(np.exp2(lxb), st.output_gamma)
    gains, ok = balance_gains(Va, Vb, [(int(i), int(j), N) for i, j, N, _ in pairs], la.shape[1])
    # the same ground in a and b (each with its own veil): log2(x_a / x_b)
    pk = {(int(i), int(j)): k for k, (i, j, _, _) in enumerate(pairs)}
    cols = []
    for a, b, _ in seams:
        k = pk.get((min(a, b), max(a, b)))
        if k is None:
            cols.append(la[:, b] - la[:, a])
        else:
            d = lxa[:, k] - lxb[:, k]
            cols.append(d if a < b else -d)
    before = np.stack(cols, 1) if seams else np.zeros((la.shape[0], 0))
    corr = np.stack([st.output_gamma * np.log2(gains[:, a] / gains[:, b]) for a, b, _ in seams], 1) \
        if seams else np.zeros_like(before)
    after = before + corr
    clip = np.zeros_like(before, bool)
    for s, (a, b, _) in enumerate(seams):
        k = pk.get((min(a, b), max(a, b)))
        clip[:, s] = True if k is None else ~ok[:, k]
    return before, after, clip, gains


def _hdr_stats(g, fl, la, lv, st):
    """Shares of the image (inside the circle) clipped, under the noise floor and within ½ EV of an HDR transition,
    and the scene's dynamic range (dB, the sun's disc left out), for one sample."""
    inside = g.cls != OUTSIDE
    L = fl['L'][0][inside]
    x = st.ae_target * L / 2.0 ** la
    lx = np.log2(np.maximum(x, 1e-30))
    near = np.zeros(len(x), bool)
    for tr in lv['transitions']:
        near |= np.abs(lx - math.log2(tr)) <= 0.5
    met = (g.cls == METERED)[inside]
    pos = L[L > 0]
    return {'clipped': float((x > lv['top']).mean()), 'under_floor': float((x < lv['floor']).mean()),
            'near_transition': float(near.mean()), 'metered_near_transition': float(near[met].mean()) if met.any()
            else 0.0, 'scene_db': float(20 * math.log10(pos.max() / pos.min())) if len(pos) > 1 else 0.0,
            'sun_in_image': bool(fl['disc_w'][0] > 0)}


def run(project, engine=None, settings: IspSettings | None = None, variant=None, progress=None, cancel=None,
        previews=False) -> IspResult | None:
    """The metering masks and the photometric seam study of a variant's visible cameras (the active variant by
    default). engine: the analysis engine (body, rig and housings, the seams); None makes one. previews: also render
    the synthetic parking scene through each lens for the page. Returns None when cancelled."""
    from . import exposure as XP
    from .analysis import Engine
    t0 = time.perf_counter()
    p = _project_for(project, variant)
    v = p.variant
    st = settings or settings_of(p)

    def prog(f, text=''):
        if progress is not None:
            progress(min(max(float(f), 0.0), 1.0), text)

    def stop():
        return cancel is not None and cancel()

    eng = engine if engine is not None else Engine()
    vg = eng.vehicle_geom(p, v)
    cams = [c for c in v.cameras if c.visible and _finite_pose(c)]
    skipped = [c.tag for c in v.cameras if c.visible and not _finite_pose(c)]
    prog(0.01, 'masks')
    geoms = []
    for k, c in enumerate(cams):
        if stop():
            return None
        prog(0.02 + 0.33 * k / max(len(cams), 1), f'{c.tag} mask')
        geoms.append(camera_geometry(p, eng, c, st, vg))
    uids = [g.uid for g in geoms]
    if not geoms:
        res = IspResult(st, v.name, v.uid, [], [], {}, {'samples': 0, 'hours': [], 'headings': [], 'slots': 0,
                                                      'days_sampled': 0, 'worst': [], 'note': ''})
        res.metrics = {'seconds': time.perf_counter() - t0, 'cameras': 0, 'seams': 0, 'overlap_pairs': 0,
                       'skipped': skipped, 'gain_range': list(GAIN_RANGE),
                       'gain_range_ev': [st.output_gamma * math.log2(x) for x in GAIN_RANGE]}
        res.findings = _findings(res)
        return res
    ares = None
    if vg is not None and len(geoms) > 1:
        ares = eng.analyse(p, v, z=0.0, cameras=cams, terrain=False)
    pairs = _overlaps(ares, uids)
    seams = _seam_list(ares, uids)
    levels = {g.uid: hdr_levels(st, g.dr_db) for g in geoms}
    hull = _hull_points(vg)
    # ---------------- the campaign: the exposure study's place, dates and headings, sampled coarser
    xst = XP.ExposureSettings.from_dict(getattr(p.analysis, 'exposure', None))
    xd = xst.to_dict()
    xd.update({'minutes': st.minutes, 'days': float(st.day_step)})
    if isinstance(xst.headings, str):
        xd['heading_step'] = st.heading_step
    xst2 = XP.ExposureSettings.from_dict(xd)
    xst2.days = float(st.day_step)                         # the exposure settings clamp the day step to 60
    headings, hw = xst2.heading_table()
    cp = XP.campaign(xst2, len(headings))
    az, el = XP.sun_position(cp.jd, xst2.lat, xst2.lon)
    up = np.nonzero(el > 0)[0]
    K = len(headings)
    sun_v = XP.vehicle_dirs(az[up], el[up], headings).reshape(-1, 3)
    s_el = np.repeat(el[up], K)
    s_w = (cp.w[up][:, None] * hw[None, :]).ravel()
    s_t = np.repeat(up, K)                                  # time-sample index
    s_head = np.tile(headings, len(up))
    n_t = max(1, int(round(xst2.window_h() * 60.0 / xst2.minutes)))
    slot = s_t % n_t
    env = DayEnv(sun_v, s_el, st, hull)
    S = len(s_el)
    acc = {k: [] for k in ('masked', 'unmasked', 'veil', 'shadow', 'sun_angle', 'pair_La', 'pair_Lb')}
    CH = 192
    for c0 in range(0, S, CH):
        if stop():
            return None
        prog(0.36 + 0.5 * c0 / max(S, 1), 'campaign')
        sc = _scene(geoms, env, st, levels, slice(c0, min(S, c0 + CH)), pairs)
        for k in acc:
            acc[k].append(sc[k])
    n = len(geoms)
    if S:
        camp = {k: np.concatenate(v_, 0) for k, v_ in acc.items()}
    else:
        camp = {k: np.zeros((0, n)) for k in ('masked', 'unmasked', 'veil', 'shadow', 'sun_angle')}
        camp['pair_La'] = camp['pair_Lb'] = np.zeros((0, len(pairs)))
    ex = _settle(geoms, camp, st)
    before, after, clip, gains = _seam_steps(ex, camp, pairs, seams, st)
    if stop():
        return None
    # ---------------- the scenarios
    prog(0.88, 'scenarios')
    from . import weather as WX
    front_x = float(vg.bbox[1][0]) if vg is not None else max([g.pos[0] for g in geoms] + [0.0])
    lse = st.low_sun_el_deg
    envs = {'low_sun': DayEnv([[math.cos(math.radians(lse)), 0.0, math.sin(math.radians(lse))]], [lse], st, hull),
            'tunnel': TunnelEnv(st, front_x)}
    wst = WX.settings_of(p)
    cond = wst.condition_for(st.night_condition)
    envs['night'] = NightEnv(st, cond, wst, WX.lamp_positions(vg, wst))
    scen = {}
    for key, e in envs.items():
        sok = (lambda g, e=e: np.array([e.sun_reaches(g.pos)])) if isinstance(e, TunnelEnv) else None
        sc = _scene(geoms, e, st, levels, slice(0, 1), pairs, sok)
        exs = _settle(geoms, sc, st)
        b, a, cl, gn = _seam_steps(exs, sc, pairs, seams, st)
        cams_s = {}
        for i, g in enumerate(geoms):
            cams_s[g.uid] = {
                'ev': float(exs['ev'][0, i]), 'ev_wanted': float(ev100(sc['masked'][0, i])),
                'ev_unmasked': float(ev100(sc['unmasked'][0, i])),
                'unmasked_off_ev': float(sc['unmasked'][0, i] - sc['masked'][0, i]),
                'exposure_ms': float(exs['t_s'][0, i] * 1000.0), 'gain_db': float(20 * math.log10(exs['gain'][0, i])),
                'short_ev': float(exs['short_ev'][0, i]), 'veil_pct': float(100 * sc['veil'][0, i]
                                                                             / 2.0 ** exs['log2_leff'][0, i]),
                'shadow_share': float(sc['shadow'][0, i]), 'stitch_gain': float(gn[0, i]),
                'hdr': _hdr_stats(g, sc['fl'][i], float(exs['log2_leff'][0, i]), levels[g.uid], st)}
        scen[key] = {'label': SCENARIO_LABEL[key], 'cams': cams_s,
                     'seams': [{'before_ev': float(b[0, s]), 'after_ev': float(a[0, s]), 'clipped': bool(cl[0, s])}
                               for s in range(len(seams))]}
        if key == 'night':
            scen[key]['condition'] = cond.name
            scen[key]['ambient_lux'] = float(cond.ambient_lux)
    # ---------------- per camera, per seam, the curves
    out_cams = []
    W_ = s_w / max(s_w.sum(), 1e-12) if S else s_w
    for i, g in enumerate(geoms):
        wu = g.weights(False, st.metering)
        body_w = float(wu[(g.cls == BODY) | (g.cls == HOUSING)].sum() / max(wu.sum(), 1e-12))
        removed = float(1.0 - g.weights(True, st.metering).sum() / max(wu.sum(), 1e-12))
        d = {'uid': g.uid, 'tag': g.tag, 'slot': g.slot, 'model': g.model, 'size': list(g.size),
             'mask_size': [int(g.mask.shape[1]), int(g.mask.shape[0])], 'shares': g.shares,
             'removed_share': removed, 'body_share_unmasked': body_w, 'f_number': g.f_number, 'fps': g.fps,
             'dynamic_range_db': levels[g.uid]['dr_db'], 'dr_given': levels[g.uid]['dr_given'],
             'grid': g.grid.tolist(), 'rects': g.rects, 'rect_cover': g.rect_cover}
        if S:
            off = camp['unmasked'][:, i] - camp['masked'][:, i]           # +: the ground comes out darker
            ev = ex['ev'][:, i]
            vp = 100 * camp['veil'][:, i] / np.exp2(ex['log2_leff'][:, i])
            d['campaign'] = {'ev_min': float(ev.min()), 'ev_median': float(np.median(ev)), 'ev_max': float(ev.max()),
                             'unmasked_off_mean_ev': float((off * W_).sum()),
                             'unmasked_off_max_ev': float(off[np.argmax(np.abs(off))]),
                             'veil_share_5pct': float(W_[vp >= 5.0].sum()), 'veil_max_pct': float(vp.max()),
                             'shadow_mean': float((camp['shadow'][:, i] * W_).sum()),
                             'short_share': float(W_[np.abs(ex['short_ev'][:, i]) > 0.05].sum())}
        out_cams.append(d)
    out_seams = []
    hours = (xst2.hour_from + (np.arange(n_t) + 0.5) * xst2.window_h() / n_t) % 24.0
    for s, (a, b, ang) in enumerate(seams):
        key = f'{geoms[a].tag}|{geoms[b].tag}'
        d = {'key': key, 'a': geoms[a].uid, 'b': geoms[b].uid, 'tag_a': geoms[a].tag, 'tag_b': geoms[b].tag,
             'slot_a': geoms[a].slot, 'slot_b': geoms[b].slot, 'angle_deg': ang}
        if S:
            ab, aa = np.abs(before[:, s]), np.abs(after[:, s])
            k = int(np.argmax(aa))
            curve_b = np.zeros(n_t)
            curve_a = np.zeros(n_t)
            np.maximum.at(curve_b, slot, ab)
            np.maximum.at(curve_a, slot, aa)
            d['campaign'] = {'before_max_ev': float(ab.max()), 'after_max_ev': float(aa.max()),
                             'before_p95_ev': float(_wquant(ab, s_w, 0.95)),
                             'after_p95_ev': float(_wquant(aa, s_w, 0.95)),
                             'after_over_share': float(W_[aa > st.seam_limit_ev].sum()),
                             'clipped_share': float(W_[clip[:, s]].sum()),
                             'worst': _when(k, cp, s_t, s_head, s_el, sun_v, xst2, before[k, s], after[k, s],
                                            camp, a, b, geoms, ex)}
            d['curve_before'] = curve_b.tolist()
            d['curve_after'] = curve_a.tolist()
        out_seams.append(d)
    worst = []
    for d in out_seams:
        if 'campaign' in d:
            w = d['campaign']['worst']
            worst.append({'scenario': 'campaign', 'seam': d['key'], 'before_ev': w['before_ev'],
                          'after_ev': w['after_ev'], 'when': w['text']})
    for key, sd in scen.items():
        for d, x in zip(out_seams, sd['seams']):
            worst.append({'scenario': key, 'seam': d['key'], 'before_ev': x['before_ev'], 'after_ev': x['after_ev'],
                          'when': SCENARIO_LABEL[key], 'clipped': x['clipped']})
    worst.sort(key=lambda r: -abs(r['after_ev']))
    campaign = {'samples': int(S), 'time_samples': int(len(cp.jd)), 'days_sampled': int(cp.n_days),
                'day_step': int(cp.day_step), 'headings': [float(h) for h in headings], 'slots': int(n_t),
                'hours': hours.tolist(), 'driving_hours': float(cp.w.sum()),
                'daylight_hours': float(cp.w[up].sum()), 'place': {'lat': xst2.lat, 'lon': xst2.lon},
                'dates': [xst2.start, xst2.end], 'window': [xst2.hour_from, xst2.hour_to], 'note': cp.note,
                'worst': worst[:8]}
    res = IspResult(st, v.name, v.uid, out_cams, out_seams, scen, campaign, geoms={g.uid: g for g in geoms})
    res.metrics = {'seconds': 0.0, 'cameras': len(geoms), 'seams': len(seams), 'overlap_pairs': len(pairs),
                   'skipped': skipped, 'gain_range': list(GAIN_RANGE),
                   'gain_range_ev': [st.output_gamma * math.log2(x) for x in GAIN_RANGE]}
    if previews:
        prog(0.93, 'previews')
        res.previews = render_previews(p, eng, cams, st, vg)
    res.findings = _findings(res)
    res.metrics['seconds'] = time.perf_counter() - t0
    prog(1.0, 'done')
    return res


def _wquant(x, w, q):
    if not len(x):
        return 0.0
    o = np.argsort(x)
    cw = np.cumsum(w[o])
    return float(x[o][min(int(np.searchsorted(cw, q * cw[-1])), len(x) - 1)])


def _when(k, cp, s_t, s_head, s_el, sun_v, xst, b, a, camp, ia, ib, geoms, ex) -> dict:
    """The worst campaign sample of a seam, described."""
    import datetime as _dt
    jd = float(cp.jd[s_t[k]])
    when = _dt.datetime(1970, 1, 1) + _dt.timedelta(days=jd - 2440587.5)
    lh = float(cp.local_h[s_t[k]])
    sv = sun_v[k]
    rel = math.degrees(math.atan2(sv[1], sv[0]))
    off = ((lh - ((jd - 0.5) % 1.0) * 24.0 + 12.0) % 24.0) - 12.0           # local minus UTC, hours
    date = (when + _dt.timedelta(hours=off)).date().isoformat()
    side = 'ahead' if abs(rel) < 30 else 'behind' if abs(rel) > 150 else ('to the left' if rel > 0 else
                                                                          'to the right')
    ta, tb = geoms[ia].tag, geoms[ib].tag
    va, vb = float(camp['veil'][k, ia] / np.exp2(ex['log2_leff'][k, ia])), \
        float(camp['veil'][k, ib] / np.exp2(ex['log2_leff'][k, ib]))
    sa, sb = float(camp['shadow'][k, ia]), float(camp['shadow'][k, ib])
    cause = []
    if max(va, vb) >= 0.05:
        cause.append(f'veil {100 * va:.0f} % ({ta}) against {100 * vb:.0f} % ({tb}) of mid-grey')
    if abs(sa - sb) >= 0.1:
        cause.append(f'the vehicle\'s shadow over {100 * sa:.0f} % of {ta}\'s metered ground and {100 * sb:.0f} % '
                     f'of {tb}\'s')
    text = (f'{date} {int(lh):02d}:{int(round((lh % 1) * 60)) % 60:02d} local, heading {float(s_head[k]):.0f}°, sun '
            f'{float(s_el[k]):.0f}° up {side}')
    return {'date': date, 'local_hour': lh, 'heading_deg': float(s_head[k]), 'sun_el_deg': float(s_el[k]),
            'sun_rel_az_deg': rel, 'before_ev': float(b), 'after_ev': float(a), 'veil_a': va, 'veil_b': vb,
            'shadow_a': sa, 'shadow_b': sb, 'cause': '; '.join(cause), 'text': text}


# =========================================================================== previews
def render_previews(project, engine, cams, st: IspSettings, vg=None, width=320) -> dict:
    """The synthetic parking scene through each camera's lens (core.synth) with its mask at the same resolution."""
    from . import synth as SY
    world = SY.parking_world(project)
    out = {}
    for c in cams:
        intr = resolve(project.module_for(c), project.lens_for(c))
        w = int(width)
        h = max(1, int(round(w * intr.height / intr.width)))
        rgb = SY.render_world(project, engine, c, world, w, h, noise=0.0)
        out[c.uid] = {'rgb': rgb, 'mask': camera_mask(project, engine, c, st.horizon_margin_deg, w, h, vg)}
    return out


# =========================================================================== findings and issues
def _f(out, sev, kind, msg, uids=()):
    out.append({'severity': sev, 'kind': kind, 'message': msg, 'uids': list(uids)})


def _findings(res: IspResult) -> list:
    st = res.settings
    out = []
    lim = st.seam_limit_ev
    if not res.cams:
        _f(out, 'note', 'no_cameras', 'The variant has no visible cameras.')
        return out
    for c in res.cams:
        if c['body_share_unmasked'] > st.body_limit:
            _f(out, 'warning', 'body_metering',
               f'{c["tag"]}: the body, rig and housings take {100 * c["body_share_unmasked"]:.0f} % of its unmasked '
               f'metering (limit {100 * st.body_limit:.0f} %): give its AE the mask.', [c['uid']])
        cm = c.get('campaign')
        if cm and abs(cm['unmasked_off_max_ev']) >= 1.0:
            _f(out, 'note', 'unmasked_off',
               f'{c["tag"]}: metering the whole frame instead of the mask moves its exposure by '
               f'{cm["unmasked_off_mean_ev"]:+.1f} EV on average and up to {cm["unmasked_off_max_ev"]:+.1f} EV '
               f'(positive: the ground comes out darker).', [c['uid']])
        if cm and cm['veil_share_5pct'] >= 0.05:
            _f(out, 'note', 'veil',
               f'{c["tag"]}: the sun veils its image by 5 % of mid-grey or more for '
               f'{100 * cm["veil_share_5pct"]:.0f} % of daylight driving (up to {cm["veil_max_pct"]:.0f} %): lifted blacks, less contrast.', [c['uid']])
    for s in res.seams:
        cm = s.get('campaign')
        if not cm:
            continue
        a = cm['after_max_ev']
        if a > lim:
            w = cm['worst']
            _f(out, 'critical' if a > 2 * lim else 'serious', 'seam_step',
               f'Seam {s["key"]}: after the bird\'s-eye gain balancing a step of up to {a:.1f} EV remains (limit '
               f'{lim:g} EV; {100 * cm["after_over_share"]:.1f} % of daylight driving over it), worst {w["text"]}'
               + (f': {w["cause"]}.' if w['cause'] else '.'), [s['a'], s['b']])
    for key, sd in res.scenarios.items():
        for s, x in zip(res.seams, sd['seams']):
            if abs(x['after_ev']) > lim:
                _f(out, 'serious' if abs(x['after_ev']) <= 2 * lim else 'critical', 'scenario_seam',
                   f'{sd["label"]}: seam {s["key"]} keeps a {abs(x["after_ev"]):.1f} EV step after balancing '
                   f'({abs(x["before_ev"]):.1f} EV before' + ('; one side clipped, left out of the gain estimate'
                                                             if x['clipped'] else '') + ').', [s['a'], s['b']])
        for c in res.cams:
            x = sd['cams'][c['uid']]
            if x['short_ev'] > 0.5:
                _f(out, 'note', 'ae_limit',
                   f'{sd["label"]}: {c["tag"]} reaches its exposure limits ({x["exposure_ms"]:.1f} ms, '
                   f'{x["gain_db"]:.0f} dB) and stays {x["short_ev"]:.1f} EV darker than its AE target.', [c['uid']])
    tun = res.scenarios.get('tunnel')
    if tun and res.cams:
        evs = [tun['cams'][c['uid']]['ev'] for c in res.cams]
        _f(out, 'note', 'tunnel_spread',
           f'Tunnel exit: the cameras\' exposures span {max(evs) - min(evs):.1f} EV '
           f'({res.cams[int(np.argmax(evs))]["tag"]} darkest).')
    if res.campaign['samples'] and not any(f['kind'] == 'seam_step' for f in out) and res.seams:
        m = max(s['campaign']['after_max_ev'] for s in res.seams if 'campaign' in s)
        _f(out, 'good', 'seams_ok', f'Over the campaign\'s daylight samples every seam stays within {m:.2f} EV after '
                                    f'balancing (limit {lim:g} EV).')
    if not res.seams:
        _f(out, 'note', 'no_seams', 'No stitch seams: fewer than two cameras or no vehicle body.')
    return out


def issues(project, engine=None, variant=None) -> list:
    """Cheap check for the Issues list: cameras whose unmasked metering the body, rig and housings dominate (a
    160-pixel-wide mask). [{severity, kind, message, uids}]."""
    st = settings_of(project)
    p = _project_for(project, variant)
    vg = engine.vehicle_geom(p, p.variant) if engine is not None else None
    out = []
    for c in p.variant.cameras:
        if not (c.visible and _finite_pose(c)) or vg is None:
            continue
        intr = resolve(p.module_for(c), p.lens_for(c))
        w = 160
        h = max(1, round(w * intr.height / intr.width))
        m = camera_mask(p, engine, c, st.horizon_margin_deg, w, h, vg)
        if st.metering == 'centre':
            U, V = np.meshgrid((np.arange(w) + 0.5) / w * intr.width, (np.arange(h) + 0.5) / h * intr.height)
            wt = centre_weights(U, V, intr)
        else:
            wt = np.ones(m.shape)
        share = float(wt[(m == BODY) | (m == HOUSING)].sum() / wt.sum())
        if share > st.body_limit:
            _f(out, 'warning', 'isp_body_metering',
               f'{c.tag}: the vehicle fills {100 * share:.0f} % of its auto-exposure metering; use a metering mask '
               f'(Analysis ▸ Exposure metering and HDR seams…).', [c.uid])
    return out


# =========================================================================== export
def _r(v, d=4):
    if v is None:
        return None
    v = float(v)
    return round(v, d) if math.isfinite(v) else None


def _clean(o, d=4):
    if isinstance(o, dict):
        return {k: _clean(v, d) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v, d) for v in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return _r(o, d)
    return o


def grids_doc(res: IspResult) -> dict:
    st = res.settings
    return {'schema': SCHEMA + '#grid', 'cols': st.grid_cols, 'rows': st.grid_rows, 'weight_max': st.weight_max,
            'order': 'row-major, row 0 at the top of the image, column 0 at the left',
            'weighting': 'metered share of each cell' + (' × centre weighting' if st.metering == 'centre' else ''),
            'note': 'V4L2 has no standard control for per-cell metering weights (V4L2_CID_EXPOSURE_METERING selects '
                    'average, centre-weighted, spot or matrix only): hand this grid to the ISP vendor\'s control.',
            'cameras': {c['tag']: {'size': c['size'], 'weights': c['grid']} for c in res.cams}}


def regions_doc(res: IspResult) -> dict:
    return {'schema': SCHEMA + '#regions', 'units': 'sensor pixels of the full-resolution mode; left/top inclusive, '
                                                    'right/bottom exclusive; corners on even pixels',
            'libargus': 'Argus::IAutoControlSettings::setAeRegions(std::vector<Argus::AcRegion>) with '
                        'Argus::AcRegion(left, top, right, bottom, weight); at most '
                        'ICameraProperties::getMaxAeRegions() regions, none smaller than getMinAeRegionSize(); scale '
                        'the rectangles for a binned or cropped sensor mode',
            'nvarguscamerasrc': 'one region: aeregion="left top right bottom weight" (the largest rectangle)',
            'cameras': {c['tag']: {'size': c['size'], 'regions': [{k: r[k] for k in ('left', 'top', 'right', 'bottom',
                                                                                     'weight')} for r in c['rects']],
                                   'metered_inside': _r(c['rect_cover']['metered_inside']),
                                   'spill': _r(c['rect_cover']['spill'])} for c in res.cams}}


def to_doc(res: IspResult, project=None) -> dict:
    st = res.settings
    lv = {c['tag']: hdr_levels(st, c['dynamic_range_db'] if c['dr_given'] else 0.0) for c in res.cams}
    return _clean({
        'schema': SCHEMA, 'project': getattr(project, 'name', ''), 'variant': res.variant, 'frame': FRAME,
        'units': {'luminance': 'cd/m²', 'illuminance': 'lx', 'exposure': 'EV at ISO 100 (log2 of L_eff·100/12.5)',
                  'step': 'EV (+: camera a\'s image of the shared ground brighter than b\'s)', 'time': 'ms',
                  'gain': 'dB', 'angle': 'deg', 'share': '0–1', 'image': 'sensor px (OpenCV, pixel centres on '
                                                                         'integers)'},
        'settings': st.to_dict(), 'assumptions': ASSUMPTIONS, 'limits': LIMITS_TEXT,
        'models': {'sky': 'CIE S 011/E:2003 standard sky type 12 (a −1, b −0.32, c 10, d −3, e 0.45), zenith luminance '
                          'Karayel et al. 1984', 'direct': 'IES: 127.5 klux · exp(−0.21 · air mass), Kasten & Young '
                                                            'air mass', 'meter': 'ISO 2720, K = 12.5',
                   'glare': 'Stiles–Holladay form L = k·E/θ² (k assumed for the lens) + uniform veiling glare index',
                   'stitch': 'core.stitch.estimate_gains (Brown & Lowe), gains 0.4–2.5',
                   'hdr': {t: {k: v for k, v in x.items()} for t, x in lv.items()}},
        'cameras': [{k: v for k, v in c.items() if k not in ('grid',)} for c in res.cams],
        'seams': res.seams, 'scenarios': res.scenarios, 'campaign': res.campaign,
        'findings': res.findings, 'metrics': res.metrics})


def seams_csv(res: IspResult) -> str:
    """Per seam and time-of-day slot: the worst step over days and headings, before and after balancing (EV)."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator='\n')
    w.writerow(['seam', 'local_hour', 'worst_before_ev', 'worst_after_ev'])
    hrs = res.campaign.get('hours', [])
    for s in res.seams:
        for h, b, a in zip(hrs, s.get('curve_before', []), s.get('curve_after', [])):
            w.writerow([s['key'], f'{h:.3f}', f'{b:.4f}', f'{a:.4f}'])
    return buf.getvalue()


def file_tag(tag: str) -> str:
    return ''.join(ch if ch.isalnum() or ch in '-_' else '_' for ch in str(tag)) or 'cam'


def files(res: IspResult, project=None) -> list:
    """[(name, bytes)]: isp.json, isp_grids.json, isp_regions.json, isp_seams.csv and isp_mask_<TAG>.png per camera
    (255 = metered, at the mask resolution)."""
    import cv2
    out = [('isp.json', (json.dumps(to_doc(res, project), indent=1, ensure_ascii=False) + '\n').encode('utf-8')),
           ('isp_grids.json', (json.dumps(grids_doc(res), indent=1) + '\n').encode('utf-8')),
           ('isp_regions.json', (json.dumps(regions_doc(res), indent=1) + '\n').encode('utf-8')),
           ('isp_seams.csv', seams_csv(res).encode('utf-8'))]
    for c in res.cams:
        g = res.geoms.get(c['uid'])
        if g is None:
            continue
        ok, buf = cv2.imencode('.png', np.where(g.mask == METERED, 255, 0).astype(np.uint8))
        if ok:
            out.append((f'isp_mask_{file_tag(c["tag"])}.png', buf.tobytes()))
    return out


def export(res: IspResult, project, out_dir) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for name, data in files(res, project):
        path = os.path.join(out_dir, name)
        with open(path, 'wb') as fh:
            fh.write(data)
        paths.append(path)
    return paths
