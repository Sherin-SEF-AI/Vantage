"""Thermal (LWIR, 8–14 µm) cameras and the night photon budget: how far a thermal camera sees a pedestrian at night,
in fog, in rain and on a hot summer road, next to how far the rig's visible cameras see the same pedestrian in the
light of UN R112 headlamps.

Library
-------
library('modules') / library('lenses') put real LWIR cores and cameras into the built-in library (core.lens's
MODALITY_LIBRARIES hook), every figure from the maker's public datasheet (THERMAL_CHECKED, each entry names its
document): Teledyne FLIR Boson+ 640 and 320 (Product Datasheet 102-2013-45, Release 117, 16 Sep 2026: Table 1, Table 11
lens optics, Table 12 mechanics, Table 13 NEDT), the FLIR ADK automotive development kit (datasheet
18-2872-OEM-COR-ADK, 01/07/19), Lynred's PICO640S detector (datasheet ref. 06/2020/01) and the Teledyne FLIR Tura
automotive camera (product sheet 01/14/2026). Module.modality = 'lwir'; Module.modality_params holds the thermal
figures (keys in PARAM_KEYS). A figure a sheet does not state is absent, never guessed; the study then shows it as
needed and says which assumption it used.

Radiative transfer (8–14 µm, or the module's stated band)
---------------------------------------------------------
* Emission: Planck's law (CODATA 2018 constants) integrated over the band by Gauss–Legendre quadrature; checked
  against the closed-form series of the Planck integral (Widger & Woodall 1976). A surface of emissivity ε at
  temperature T in surroundings of radiance L_env sends ε·B(T) + (1 − ε)·L_env. Emissivities: human skin 0.98
  (Steketee 1973), clothing 0.95 (ISO 9920), asphalt 0.93 and window glass 0.92 (Incropera & DeWitt, Table A.11
  ranges), paint 0.90 and tyre rubber 0.95 (the same table; typical), far background (vegetation, buildings) 0.95.
* Temperatures: clothing surface from the ISO 7730:2005 heat balance (eq. 2–5: metabolic rate, clothing insulation,
  air and mean radiant temperature, air speed), exposed skin at Fanger's comfort skin temperature 35.7 − 0.0275·M,
  the clear night sky at Swinbank's (1963) T_sky = 0.0552·T_air^1.5 (an overcast, foggy or rainy sky at the air
  temperature), the road and the air per condition (typical figures, editable). A car's paint +2 K and tyres +10 K over
  the air after driving: assumed.
* Background: the pedestrian's silhouette below the camera's height is seen against the road behind it, above it
  against the far scene (the share follows from the camera height): that is what inverts the contrast on a hot road.
* Atmosphere: water vapour from the relative humidity and temperature with the empirical LWIR transmission FLIR fits
  for its radiometric cameras (Minkina & Dudzik 2009, §2.4; the constants as Thermimage's raw2temp prints them, from a
  FLIR SC660); fog from its meteorological optical range (σ_vis = 3.0/MOR, core.weather) scaled per wavelength by the
  Mie extinction of a Shettle & Fenn (1979) droplet distribution (radiation fog: modified gamma α = 6, b = 3 µm⁻¹;
  advection fog: α = 3, b = 0.3 µm⁻¹) with the refractive index of water (Hale & Querry 1973), computed with Bohren &
  Huffman's BHMIE recursion; rain and snow as ITU-R P.1814 (drops much larger than the wavelength: the visible
  attenuation applies). Path radiance adds equally to target and background: it leaves the contrast, which falls by
  the transmission.
* Optics and detector: irradiance at the focal plane E = π·τ·L/(4N²) (lens transmission τ, f-number N); the
  datasheet NETD (stated at f/1 clear aperture for the Boson+, PICO640S and Tura) becomes a noise-equivalent radiance
  difference NEΔL = NETD·dL/dT at the sheet's scene temperature, scaled by N²/τ as Teledyne FLIR's datasheet states
  for its lens configurations (Boson+ Table 13). Contrast-to-noise ratio CNR = |ΔL_apparent|·fill/NEΔL with the
  pixel fill of a target narrower or shorter than one pixel. A window in front of the lens multiplies the signal
  (germanium, ZnSe, chalcogenide: Fresnel loss of the uncoated slab from the refractive index, or a typical AR-coated
  figure); window and windscreen glass is opaque in the LWIR (soda-lime silica absorbs beyond about 5 µm: Rubin 1985),
  so a thermal camera behind it sees nothing.
* Range: detection, recognition and identification per class from the pixels across the target's critical dimension
  √(height × width) (DRI convention 1.5 / 6 / 12 pixels, as Lynred's and Teledyne FLIR's range figures use; or
  Johnson's 1958 50 % criteria 1.0 / 4.0 / 6.4 cycles = 2 / 8 / 12.8 pixels), together with CNR ≥ snr_min (Rose's
  criterion, 5 by default). The pixels come from core.perception.object_pixels through the camera's real lens model.

Visible cameras at night: the UN R112 photon budget
---------------------------------------------------
* Headlamps: the pedestrian is lit by the vehicle's two headlamps (core.weather's lamp positions) and the ambient
  light. The intensity toward each point is core.weather's typical class B pattern (it meets R112's test points:
  validation weather.headlamps_r112) or, with beam_model 'r112_min', that pattern scaled down until its weakest test
  point sits exactly at the regulation's minimum: the dimmest lamp R112 approves. R112_PASSING / R112_DRIVING are the
  regulation's class B tables (UN R112, 01 series, Supplement 4, §6.2.4 and §6.3.3, as published in OJ L 250,
  22.8.2014, pp. 79–82).
* Reflected light: clothing reflectance from its Munsell value by the ASTM D1535 polynomial (relative to the
  perfect diffuser: dark N2.5 ≈ 4.5 %, medium N5 ≈ 19.3 %, light N8 ≈ 57.6 %); luminance ρ·E/π (Lambertian) at the share of the pedestrian's height that
  is lit (core.weather's lit_share), through the extinction on the way back.
* Electrons: π·τ·L/(4N²)·t·Φ·A·QE as core.weather.camera_floor (exposure ≤ 1/fps and ≤ the night limit, pixel area,
  Φ = 1e16 photons per lux·s·m²); noise √(signal + background + dark current·t + read noise²). QE, read noise and dark
  current come from the module's modality_params when its datasheet states them, else core.weather's typical
  figures, listed as needed. SNR ≥ snr_min, the target's contrast after the fog's extinction above the detector's
  threshold, and the same pixel criteria give the visible camera's range.

Limits
------
Typical models, not measurements of a scene: the emissivities, temperatures and clothing figures are typical; the
target is a uniform silhouette (no hot face against cold clothing as separate blobs, no MTF of the optics, no fixed-
pattern noise, no scene clutter); the humidity fit is FLIR's empirical one (fitted over short paths; clipped to
[0, 1] and to HUMIDITY_MAX_M); the fog droplet distributions are two model fogs; the road's emissivity is its
hemispherical figure (at grazing angles it reflects more sky); a headlamp's veiling light in fog, oncoming glare and
retroreflectors are not modelled. Detection ranges are what the camera resolves: an upper bound on what a trained
detector finds.
"""
from __future__ import annotations

import csv
import functools
import io
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field, replace

import numpy as np

SCHEMA = 'vantage.thermal/1'
FRAME = 'vehicle (ISO 8855, rear axle): x forward, y left, z up, metres; ranges along each camera\'s heading'
THERMAL_CHECKED = '2026-11-11'         # when the datasheets below were read

# ---------------------------------------------------------------- physical constants (CODATA 2018, exact SI values)
H_PLANCK = 6.62607015e-34              # J·s
C_LIGHT = 299792458.0                  # m/s
K_BOLTZ = 1.380649e-23                 # J/K
SIGMA_SB = 5.670374419e-8              # W/(m²·K⁴), derived
C1 = 2.0 * H_PLANCK * C_LIGHT ** 2     # W·m² (per steradian: radiance form)
C2 = H_PLANCK * C_LIGHT / K_BOLTZ      # m·K
T0 = 273.15

LWIR_BAND = (8.0, 14.0)                # µm
GL_NODES = 48                          # Gauss–Legendre nodes for a band integral (exact to ~1e-12 for 8–14 µm)
SPECTRAL_NODES = 16                    # nodes for the study's spectral sums (fog varies smoothly in the band)

LEVELS = ('detection', 'recognition', 'identification')
LEVEL_SHORT = {'detection': 'D', 'recognition': 'R', 'identification': 'I'}
# pixels across the critical dimension √(h·w) for each level (50 % probability)
CRITERIA = {
    'dri': {'detection': 1.5, 'recognition': 6.0, 'identification': 12.0},
    'johnson': {'detection': 2.0, 'recognition': 8.0, 'identification': 12.8},
}
CRITERIA_LABEL = {'dri': 'DRI pixels 1.5 / 6 / 12', 'johnson': 'Johnson 1958: 1 / 4 / 6.4 cycles'}
CRITERIA_NOTE = {
    'dri': 'Pixels across the critical dimension √(h·w): detection 1.5, recognition 6, identification 12 — the '
           'convention behind Lynred\'s and Teledyne FLIR\'s published range figures (Lynred PICO640S datasheet: '
           '"Range for Johnson\'s criteria", a 1.80 m × 0.50 m human).',
    'johnson': 'J. Johnson, "Analysis of image forming systems", Image Intensifier Symposium, Fort Belvoir, 1958: '
               '50 % probability at 1.0 ± 0.25 (detection), 4.0 ± 0.8 (recognition) and 6.4 ± 1.5 (identification) '
               'line pairs across the critical dimension (2 pixels per line pair).',
}
ROSE_K = 5.0                           # A. Rose, J. Opt. Soc. Am. 38, 196 (1948): SNR ≥ 5 to detect with certainty

# ---------------------------------------------------------------- surfaces: emissivity (8–14 µm) and the source
SURFACES = {
    'skin': (0.98, 'J. Steketee, "Spectral emissivity of skin and pericardium", Phys. Med. Biol. 18(5), 686–694, '
                   '1973: 0.98 ± 0.01 from 3 to 14 µm, independent of pigmentation'),
    'clothing': (0.95, 'ISO 9920:2007 (estimation of the thermal insulation and water vapour resistance of a '
                       'clothing ensemble): 0.95 for ordinary textiles (typical)'),
    'asphalt': (0.93, 'F. P. Incropera, D. P. DeWitt, Fundamentals of Heat and Mass Transfer, Table A.11: asphalt '
                      'pavement 0.85–0.93 (the upper end: fresh, dark asphalt)'),
    'paint': (0.90, 'Incropera & DeWitt, Table A.11: paints 0.90–0.98 (typical car paint taken at 0.90)'),
    'rubber': (0.95, 'Incropera & DeWitt, Table A.11 and makers\' emissivity tables: rubber 0.93–0.95 (typical)'),
    'glass': (0.92, 'Incropera & DeWitt, Table A.11: window glass 0.90–0.95'),
    'background': (0.95, 'far scene of vegetation, walls and parked cars: 0.95 (typical)'),
}
# thermal signature per class: (surface, share of the frontal silhouette, temperature source) and the size (m)
CLASSES = {
    'pedestrian': {'height': 1.8, 'width': 0.5,
                   'parts': (('skin', 'skin_share', 'skin'), ('clothing', 'rest', 'clothing'))},
    'cyclist': {'height': 1.7, 'width': 0.6,
                'parts': (('skin', 0.07, 'skin'), ('clothing', 0.63, 'clothing'), ('paint', 0.30, 'air'))},
    'car': {'height': 1.5, 'width': 1.8,
            'parts': (('paint', 0.65, 'air+2'), ('rubber', 0.15, 'air+10'), ('glass', 0.20, 'air'))},
}
CLASS_LABEL = {'pedestrian': 'Pedestrian', 'cyclist': 'Cyclist', 'car': 'Car'}
CLASS_NOTE = ('Sizes: a 1.8 × 0.5 m person (Lynred\'s and FLIR\'s range figures), a 1.7 × 0.6 m cyclist and a '
              '1.5 × 1.8 m car seen head-on (core.perception\'s classes). A cyclist is 70 % person and 30 % bicycle '
              'at the air temperature; a car 65 % paint (+2 K), 15 % tyres (+10 K) and 20 % glass: assumed shares and '
              'warm-ups after driving.')

# Munsell value → luminous reflectance Y (ASTM D1535-14, the Newhall–Nickerson–Judd polynomial), for the clothing
CLOTHING = {'dark': 2.5, 'medium': 5.0, 'light': 8.0}
CLOTHING_LABEL = {'dark': 'Dark (N2.5)', 'medium': 'Medium (N5)', 'light': 'Light (N8)'}

# ---------------------------------------------------------------- windows in front of a thermal lens
# (label, transmission 8–14 µm, source). Uncoated: an incoherent lossless slab, T = 2n/(n² + 1).
def _slab(n):
    return 2.0 * n / (n * n + 1.0)


WINDOWS = {
    'none': ('No window', 1.0, 'nothing between the lens and the scene'),
    'ge_ar': ('Germanium, AR coated', 0.92, 'typical figure of a germanium window AR coated for 8–12 µm (both faces '
                                            'about 1 % reflection plus absorption)'),
    'ge': ('Germanium, uncoated', _slab(4.003), 'Fresnel loss of an uncoated slab, n = 4.003 at 10 µm (H. H. Li, J. '
                                                 'Phys. Chem. Ref. Data 9, 561, 1980); absorption neglected'),
    'znse_ar': ('ZnSe, AR coated', 0.95, 'typical figure of a ZnSe window AR coated for 8–12 µm'),
    'znse': ('ZnSe, uncoated', _slab(2.403), 'Fresnel loss of an uncoated slab, n = 2.403 at 10.6 µm (typical '
                                              'catalogue figure); absorption neglected'),
    'chalcogenide_ar': ('Chalcogenide glass, AR coated', 0.92, 'typical figure of an AR-coated chalcogenide (GASIR, '
                                                               'IG6) window'),
    'glass': ('Window or windscreen glass', 0.0, 'soda-lime silica glass absorbs strongly beyond about 5 µm (M. '
                                                 'Rubin, "Optical properties of soda lime silica glasses", Solar '
                                                 'Energy Materials 12, 275–288, 1985): opaque in 8–14 µm'),
}

# ---------------------------------------------------------------- water: complex refractive index (Hale & Querry)
# G. M. Hale, M. R. Querry, "Optical constants of water in the 200-nm to 200-µm wavelength region", Appl. Opt. 12,
# 555–563 (1973), as tabulated by refractiveindex.info (read 2026-11-11): λ µm, n, k
WATER_NK = (
    (0.55, 1.333, 1.96e-9), (7.5, 1.304, 0.0326), (7.8, 1.297, 0.0335), (8.0, 1.291, 0.0343), (8.2, 1.286, 0.0351),
    (8.4, 1.281, 0.0361), (8.6, 1.275, 0.0372), (8.8, 1.269, 0.0385), (9.0, 1.262, 0.0399), (9.2, 1.255, 0.0415),
    (9.4, 1.247, 0.0433), (9.6, 1.239, 0.0454), (9.8, 1.229, 0.0479), (10.0, 1.218, 0.0508), (10.5, 1.185, 0.0662),
    (11.0, 1.153, 0.0968), (11.5, 1.126, 0.142), (12.0, 1.111, 0.199), (12.5, 1.123, 0.259), (13.0, 1.146, 0.305),
    (13.5, 1.177, 0.343), (14.0, 1.210, 0.370), (14.5, 1.241, 0.388))
# Shettle & Fenn (1979), AFGL-TR-79-0214: modified gamma n(r) = a·r^α·exp(−b·r^γ), r in µm, n per cm³ per µm
FOGS = {'radiation': {'alpha': 6.0, 'b': 3.0, 'gamma': 1.0, 'a': 607.5, 'label': 'Radiation fog (small drops)'},
        'advection': {'alpha': 3.0, 'b': 0.3, 'gamma': 1.0, 'a': 0.027, 'label': 'Advection fog (large drops)'}}

# FLIR's empirical atmospheric transmission (Minkina & Dudzik, Infrared Thermography: Errors and Uncertainties, Wiley
# 2009, §2.4; constants from a FLIR SC660 as G. Tattersall's Thermimage raw2temp gives them)
FLIR_ATM = {'X': 1.9, 'a1': 0.006569, 'a2': 0.01262, 'b1': -0.002276, 'b2': -0.00667}
HUMIDITY_MAX_M = 2000.0                 # the fit is used out to this path length (it is a short-range fit)

# ---------------------------------------------------------------- UN R112, class B (cd per headlamp)
# UN Regulation No. 112, 01 series, Supplement 4 (OJ L 250, 22.8.2014), §6.2.4: passing beam test points (RH
# traffic); h degrees, positive LEFT, v degrees, positive UP (core.weather.headlamp_cd's convention)
R112_PASSING = {
    'B50L': {'h': 3.43, 'v': 0.57, 'max': 350.0},
    'BR': {'h': -2.5, 'v': 1.0, 'max': 1750.0},
    '75R': {'h': -1.15, 'v': -0.57, 'min': 10100.0, 'range_m': 75.0},
    '75L': {'h': 3.43, 'v': -0.57, 'max': 10600.0, 'range_m': 75.0},
    '50L': {'h': 3.43, 'v': -0.86, 'max': 13200.0, 'range_m': 50.0},
    '50R': {'h': -1.72, 'v': -0.86, 'min': 10100.0, 'range_m': 50.0},
    '50V': {'h': 0.0, 'v': -0.86, 'min': 5100.0, 'range_m': 50.0},
    '25L': {'h': 9.0, 'v': -1.72, 'min': 1700.0, 'range_m': 25.0},
    '25R': {'h': -9.0, 'v': -1.72, 'min': 1700.0, 'range_m': 25.0},
}
R112_ZONES = {'III': {'max': 625.0, 'text': 'any point in zone III (above the cut-off, 8L–8R up to 4U)'},
              'IV': {'min': 2500.0, 'h': (-5.15, 5.15), 'v': (-1.72, -0.86),
                     'text': 'any point in zone IV (0.86D to 1.72D, 5.15L to 5.15R)'},
              'I': {'max_rel': 2.0, 'text': 'any point in zone I (1.72D to 4D, 9L to 9R): below 2 × 50R or 50L'}}
# §6.3.3: driving beam
R112_DRIVING = {
    'Imax': {'min': 40500.0, 'max': 215000.0},
    'H-5L': {'h': 5.0, 'v': 0.0, 'min': 5100.0},
    'H-2.5L': {'h': 2.5, 'v': 0.0, 'min': 20300.0},
    'H-2.5R': {'h': -2.5, 'v': 0.0, 'min': 20300.0},
    'H-5R': {'h': -5.0, 'v': 0.0, 'min': 5100.0},
}
R112_HV_SHARE = 0.8                    # §6.3.3.1: HV lies within the 80 % isolux of I_max
R112_SOURCE = ('UN Regulation No. 112 (headlamps emitting an asymmetrical passing beam or a driving beam), 01 series '
               'of amendments, Supplement 4: §6.2.4 passing beam and §6.3.3 driving beam, class B; as published in '
               'the Official Journal of the EU, L 250, 22.8.2014, pp. 79–82 (read 2026-11-11)')
BEAM_MODELS = ('typical', 'r112_min')
BEAM_MODEL_LABEL = {'typical': 'Typical lamp', 'r112_min': 'R112 minimum'}
BEAMS = ('low', 'high')

ASSUMPTIONS = (
    'Typical models, not measurements of a scene: emissivities and temperatures are typical figures from the cited '
    'literature (ISO 7730 clothing heat balance, Swinbank\'s night sky); a car\'s +2 K paint and +10 K tyres are '
    'assumed. The target is a uniform silhouette against the road below the camera\'s height and the far scene '
    'above it; no optics MTF, fixed-pattern noise or clutter. Humidity: FLIR\'s empirical LWIR transmission fit; fog: '
    'Mie extinction of Shettle & Fenn model fogs scaled to the visibility; rain and snow: ITU-R P.1814. Visible '
    'cameras: core.weather\'s typical QE, read noise and dark current where the datasheet states none; headlamps: '
    'core.weather\'s R112-shaped pattern (or scaled to R112\'s minimum); clothing reflectance from its Munsell value. '
    'Not modelled: veiling light from the own headlamps in fog, oncoming glare, retroreflectors, wet clothing, the '
    'road\'s lower emissivity at grazing angles. A range is what the camera resolves: an upper bound for a trained '
    'detector.')
LIMITS_TEXT = ('Ranges are along each camera\'s heading, to a pedestrian standing lateral_m to its left (negative: '
               'to its right, the kerb side of right-hand traffic, R112\'s 50R and 75R line).')

PARAM_KEYS = ('band_um', 'detector', 'netd_mk', 'netd_f_number', 'netd_scene_k', 'netd_fps', 'netd_note',
              'frame_rates_hz', 'export_fps_hz', 'shutterless', 'heated_window', 'time_constant_ms', 'source',
              'qe', 'read_noise_e', 'dark_current_e_s')


# =========================================================================== Planck
def planck(lam_um, T_K):
    """Spectral radiance B_λ(T), W/(m²·sr·µm), at wavelengths lam_um (µm) and temperatures T_K (K); broadcasts."""
    lam = np.asarray(lam_um, float) * 1e-6
    T = np.asarray(T_K, float)
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        x = C2 / (lam * T)
        out = C1 / lam ** 5 / np.expm1(x) * 1e-6
    return np.where(np.isfinite(out), out, 0.0)


def dplanck_dT(lam_um, T_K):
    """∂B_λ/∂T, W/(m²·sr·µm·K)."""
    lam = np.asarray(lam_um, float) * 1e-6
    T = np.asarray(T_K, float)
    x = C2 / (lam * T)
    with np.errstate(over='ignore', invalid='ignore'):
        e = np.exp(np.minimum(x, 700.0))
        out = C1 / lam ** 5 * x * e / (T * (e - 1.0) ** 2) * 1e-6
    return np.where(np.isfinite(out), out, 0.0)


@functools.lru_cache(maxsize=64)
def _gl(lo, hi, n):
    x, w = np.polynomial.legendre.leggauss(int(n))
    return 0.5 * (hi - lo) * x + 0.5 * (hi + lo), 0.5 * (hi - lo) * w


def band_nodes(band=LWIR_BAND, n=GL_NODES):
    """Gauss–Legendre nodes (µm) and weights over the band."""
    return _gl(float(band[0]), float(band[1]), int(n))


def band_radiance(T_K, band=LWIR_BAND, n=GL_NODES):
    """In-band radiance ∫B_λ dλ, W/(m²·sr), of a blackbody at T_K (scalar or array)."""
    lam, w = band_nodes(band, n)
    T = np.asarray(T_K, float)
    return np.tensordot(planck(lam, T[..., None]), w, axes=([-1], [0]))


def band_dl_dt(T_K, band=LWIR_BAND, n=GL_NODES):
    """In-band ∂L/∂T, W/(m²·sr·K)."""
    lam, w = band_nodes(band, n)
    T = np.asarray(T_K, float)
    return np.tensordot(dplanck_dT(lam, T[..., None]), w, axes=([-1], [0]))


def band_radiance_series(T_K, band=LWIR_BAND, terms=40) -> float:
    """The band radiance from the closed-form series of the Planck integral (W. K. Widger, M. P. Woodall, "Integration
    of the Planck blackbody radiation function", Bull. Am. Meteorol. Soc. 57(10), 1217–1219, 1976):
    ∫_x^∞ t³/(e^t − 1) dt = Σ_n e^(−nx)(x³/n + 3x²/n² + 6x/n³ + 6/n⁴), x = c2/(λT). The validation reference."""
    T = float(T_K)

    def tail(lam_um):
        x = C2 / (lam_um * 1e-6 * T)
        n = np.arange(1, int(terms) + 1, dtype=float)
        return float(np.sum(np.exp(-n * x) * (x ** 3 / n + 3 * x ** 2 / n ** 2 + 6 * x / n ** 3 + 6 / n ** 4)))
    k = C1 * (T / C2) ** 4                    # ∫B_λ dλ = c1 (T/c2)⁴ ∫ t³/(e^t − 1) dt (λ from ∞ to 0 ↔ t 0 to ∞)
    return k * (tail(float(band[1])) - tail(float(band[0])))


def apparent_temperature(L, band=LWIR_BAND) -> float:
    """The blackbody temperature (K) whose band radiance is L (Newton's method)."""
    L = float(L)
    T = 300.0
    for _ in range(40):
        f = float(band_radiance(T, band)) - L
        d = float(band_dl_dt(T, band))
        step = f / max(d, 1e-12)
        T = min(max(T - step, 50.0), 2000.0)
        if abs(step) < 1e-9:
            break
    return T


def surface_radiance(eps, T_K, L_env, band=LWIR_BAND, lam=None):
    """ε·B(T) + (1 − ε)·L_env: spectral at nodes lam when given, else in-band."""
    if lam is None:
        return eps * band_radiance(T_K, band) + (1.0 - eps) * L_env
    return eps * planck(lam, T_K) + (1.0 - eps) * L_env


# =========================================================================== temperatures
def sky_temperature_k(t_air_c, sky='clear') -> float:
    """Night sky temperature: Swinbank (Q. J. R. Meteorol. Soc. 89, 339–348, 1963) T_sky = 0.0552·T_air^1.5 for a
    clear sky; an overcast, foggy or rainy sky at the air temperature."""
    Ta = float(t_air_c) + T0
    return 0.0552 * Ta ** 1.5 if sky == 'clear' else Ta


def clothing_temperature_c(t_air_c, t_rad_c, met_w_m2=140.0, clo=1.0, air_speed=1.0, work_w_m2=0.0) -> float:
    """Clothing surface temperature, °C: the ISO 7730:2005 heat balance (eq. 2–5)
    t_cl = 35.7 − 0.028(M − W) − I_cl{3.96e-8 f_cl[(t_cl + 273)⁴ − (t_r + 273)⁴] + f_cl h_c (t_cl − t_a)},
    h_c = max(2.38|t_cl − t_a|^0.25, 12.1√v_ar), f_cl from I_cl (m²·K/W, 1 clo = 0.155)."""
    from scipy.optimize import brentq
    Icl = 0.155 * max(float(clo), 0.0)
    fcl = 1.0 + 1.290 * Icl if Icl <= 0.078 else 1.05 + 0.645 * Icl
    M, W, ta, tr = float(met_w_m2), float(work_w_m2), float(t_air_c), float(t_rad_c)
    hcf = 12.1 * math.sqrt(max(float(air_speed), 0.0))

    def f(tcl):
        hc = max(2.38 * abs(tcl - ta) ** 0.25, hcf)
        return (35.7 - 0.028 * (M - W) - Icl * (3.96e-8 * fcl * ((tcl + 273.0) ** 4 - (tr + 273.0) ** 4)
                                                 + fcl * hc * (tcl - ta)) - tcl)
    if Icl <= 0:
        return 35.7 - 0.028 * (M - W)
    return float(brentq(f, -80.0, 80.0, xtol=1e-10))


def skin_temperature_c(met_w_m2=140.0, work_w_m2=0.0) -> float:
    """Fanger's comfort mean skin temperature 35.7 − 0.0275(M − W) (P. O. Fanger, Thermal Comfort, 1970; ISO 7730's
    basis). Exposed skin in the cold runs cooler: this is the warm end."""
    return 35.7 - 0.0275 * (float(met_w_m2) - float(work_w_m2))


def munsell_reflectance(V) -> float:
    """Luminous reflectance (0–1) of a neutral surface of Munsell value V (ASTM D1535: Y = 1.1914V − 0.22533V² +
    0.23352V³ − 0.020484V⁴ + 0.00081939V⁵, in %)."""
    V = float(V)
    return (1.1914 * V - 0.22533 * V ** 2 + 0.23352 * V ** 3 - 0.020484 * V ** 4 + 0.00081939 * V ** 5) / 100.0


# =========================================================================== atmosphere
def water_vapour(rh_pct, t_c) -> float:
    """FLIR's water-vapour term H2O = RH·exp(1.5587 + 0.06939T − 0.00027816T² + 0.00000068455T³) (≈ the absolute
    humidity, g/m³)."""
    T = float(t_c)
    return max(float(rh_pct), 0.0) / 100.0 * math.exp(1.5587 + 0.06939 * T - 0.00027816 * T * T
                                                       + 0.00000068455 * T ** 3)


def humidity_transmission(d_m, rh_pct, t_c):
    """LWIR transmission of humid air over d_m (array): FLIR's two-term fit (Minkina & Dudzik 2009), applied to two
    halves of the path as Thermimage does, clipped to [0, 1] and evaluated no further than HUMIDITY_MAX_M (beyond, the
    transmission per metre at that length continues)."""
    d = np.maximum(np.asarray(d_m, float), 0.0)
    h = water_vapour(rh_pct, t_c)
    c = FLIR_ATM
    sq = math.sqrt(h)

    def tau(x):
        s = np.sqrt(x / 2.0)
        t1 = c['X'] * np.exp(-s * (c['a1'] + c['b1'] * sq)) + (1 - c['X']) * np.exp(-s * (c['a2'] + c['b2'] * sq))
        return np.clip(t1, 1e-9, 1.0) ** 2
    out = tau(np.minimum(d, HUMIDITY_MAX_M))
    far = d > HUMIDITY_MAX_M
    if np.any(far):
        k = -math.log(float(tau(np.array(HUMIDITY_MAX_M)))) / HUMIDITY_MAX_M
        out = np.where(far, out * np.exp(-k * (d - HUMIDITY_MAX_M)), out)
    return out


def water_index(lam_um):
    """Complex refractive index of liquid water at lam_um (linear interpolation of Hale & Querry's table)."""
    t = np.array(WATER_NK)
    lam = np.asarray(lam_um, float)
    return np.interp(lam, t[:, 0], t[:, 1]) + 1j * np.interp(lam, t[:, 0], t[:, 2])


def mie_q(m, x) -> tuple:
    """(Q_ext, Q_sca, Q_back) of a homogeneous sphere of relative refractive index m (n + ik) and size parameter x:
    C. F. Bohren, D. R. Huffman, Absorption and Scattering of Light by Small Particles (Wiley 1983), Appendix A,
    BHMIE (the logarithmic derivative by downward recurrence, the Riccati–Bessel functions upward)."""
    m = complex(m)
    x = float(x)
    if x < 1e-6:
        return 0.0, 0.0, 0.0
    y = m * x
    nstop = int(x + 4.0 * x ** (1.0 / 3.0) + 2.0)
    nmx = int(max(nstop, abs(y))) + 15
    D = np.zeros(nmx + 1, complex)
    for n in range(nmx, 0, -1):
        D[n - 1] = n / y - 1.0 / (D[n] + n / y)
    psi0, psi1 = math.cos(x), math.sin(x)
    chi0, chi1 = -math.sin(x), math.cos(x)
    xi1 = complex(psi1, -chi1)
    qsca = qext = 0.0
    qb = 0j
    for n in range(1, nstop + 1):
        psi = (2 * n - 1) * psi1 / x - psi0
        chi = (2 * n - 1) * chi1 / x - chi0
        xi = complex(psi, -chi)
        da, db = D[n] / m + n / x, m * D[n] + n / x
        an = (da * psi - psi1) / (da * xi - xi1)
        bn = (db * psi - psi1) / (db * xi - xi1)
        qsca += (2 * n + 1) * (abs(an) ** 2 + abs(bn) ** 2)
        qext += (2 * n + 1) * (an.real + bn.real)
        qb += (2 * n + 1) * (-1) ** n * (an - bn)
        psi0, psi1 = psi1, psi
        chi0, chi1 = chi1, chi
        xi1 = complex(psi1, -chi1)
    return 2.0 * qext / x ** 2, 2.0 * qsca / x ** 2, abs(qb) ** 2 / x ** 2


FOG_RADII = np.geomspace(0.05, 80.0, 140)          # µm


@functools.lru_cache(maxsize=16)
def _fog_extinction_per_cm3(kind, lam_key) -> tuple:
    """Mie extinction coefficient (1/km per unit of the distribution's a) of a Shettle & Fenn fog at each λ."""
    f = FOGS[kind]
    r = FOG_RADII
    n = r ** f['alpha'] * np.exp(-f['b'] * r ** f['gamma'])            # shape (a = 1), per cm³ per µm
    w = np.gradient(r)
    out = []
    for lam in lam_key:
        mm = complex(water_index(lam))
        q = np.array([mie_q(mm, 2 * math.pi * ri / lam)[0] for ri in r])
        # σ = ∫ Q π r² n dr: r in µm, n per cm³·µm → µm²/cm³ = 1e-12 m² / 1e-6 m³ = 1e-6 /m = 1e-3 /km
        out.append(float(np.sum(q * math.pi * r ** 2 * n * w)) * 1e-3)
    return tuple(out)


def fog_ratio(kind, lam_um) -> np.ndarray:
    """σ_ext(λ) / σ_ext(0.55 µm) of the model fog (radiation or advection)."""
    lam = np.atleast_1d(np.asarray(lam_um, float))
    key = tuple(round(float(v), 6) for v in lam) + (0.55,)
    s = np.array(_fog_extinction_per_cm3(kind if kind in FOGS else 'radiation', key))
    return s[:-1] / s[-1]


def fog_visibility_m(kind) -> float:
    """The MOR (5 % contrast, core.weather's 3.0/σ) of the Shettle & Fenn fog with its published a: a check of the
    Mie sum against the visibility those model fogs are known for (≈ 130 m advection, ≈ 450 m radiation, quoted at
    2 %: 3.912/σ)."""
    from . import weather as W
    s = _fog_extinction_per_cm3(kind, (0.55,))[0] * FOGS[kind]['a']       # 1/km
    return W.MOR_LN / (s / 1000.0)


def rain_extinction(rate_mm_h) -> float:
    """1/m: ITU-R P.1814 (Carbonneau) rain attenuation 1.076·R^0.67 dB/km, independent of the wavelength (drops much
    larger than it), as core.weather."""
    from . import weather as W
    return float(W.rain_optical_db_km(rate_mm_h)) / W.DB_PER_NEPER / 1000.0


# =========================================================================== conditions
@dataclass(frozen=True)
class ThermalCondition:
    id: str
    name: str
    t_air_c: float = 10.0
    rh_pct: float = 70.0
    t_road_c: float = 12.0
    sky: str = 'clear'                 # clear | overcast (fog and rain: overcast)
    visibility_m: float = 0.0          # fog MOR, 0 = none
    fog: str = 'radiation'             # radiation | advection
    rain_mm_h: float = 0.0
    snow_mm_h: float = 0.0
    clo: float = 1.0                   # clothing insulation of the pedestrian
    ambient_lux: float = 0.1           # the visible cameras' ambient light (night)

    def label(self) -> str:
        bits = [f'{self.t_air_c:g} °C air', f'road {self.t_road_c:g} °C']
        if self.visibility_m > 0:
            bits.append(f'fog {self.visibility_m:g} m')
        if self.rain_mm_h > 0:
            bits.append(f'rain {self.rain_mm_h:g} mm/h')
        if self.snow_mm_h > 0:
            bits.append(f'snow {self.snow_mm_h:g} mm/h')
        return f'{self.name} · ' + ', '.join(bits)

    def to_doc(self) -> dict:
        d = asdict(self)
        d['label'] = self.label()
        d['t_sky_c'] = round(sky_temperature_k(self.t_air_c, self.sky) - T0, 2)
        return d

    def weather(self, beam='low'):
        """The core.weather Condition the visible cameras see: night, the headlamps' beam, fog, rain, snow."""
        from . import weather as W
        return W.Condition(f'thermal_{self.id}', self.name, W.NIGHT, float(self.ambient_lux), beam,
                           float(self.visibility_m), float(self.rain_mm_h), float(self.snow_mm_h), 'wet'
                           if self.t_air_c > 0.5 else 'dry')


CONDITIONS = (
    ThermalCondition('clear_night', 'Clear night', 10.0, 70.0, 12.0, 'clear', clo=1.0),
    ThermalCondition('fog', 'Fog at night', 8.0, 100.0, 9.0, 'overcast', visibility_m=150.0, fog='radiation',
                     clo=1.0),
    ThermalCondition('rain', 'Heavy rain at night', 10.0, 95.0, 10.0, 'overcast', rain_mm_h=25.0, clo=1.0),
    ThermalCondition('hot_road', 'Hot summer night', 30.0, 50.0, 40.0, 'clear', clo=0.5),
)
COND = {c.id: c for c in CONDITIONS}
COND_IDS = tuple(COND)
COND_NOTES = {'clear_night': 'A dry, clear night at 10 °C: the asphalt a little warmer than the air.',
              'fog': 'Dense fog (150 m, as core.weather\'s fog) of small radiation-fog drops at 8 °C.',
              'rain': 'A 25 mm/h downpour (core.weather\'s heavy rain) at 10 °C, the road at the air temperature.',
              'hot_road': 'A summer night at 30 °C after a hot day: the asphalt still at 40 °C, warmer than people.'}
COND_LIMITS = {'t_air_c': (-30.0, 45.0), 'rh_pct': (0.0, 100.0), 't_road_c': (-30.0, 70.0),
               'visibility_m': (0.0, 50000.0), 'rain_mm_h': (0.0, 150.0), 'snow_mm_h': (0.0, 30.0),
               'clo': (0.0, 3.0), 'ambient_lux': (0.0, 100.0)}
COND_EDITABLE = tuple(COND_LIMITS) + ('sky', 'fog')


def _num(v):
    if isinstance(v, bool):
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def clean_condition(base: ThermalCondition, d) -> ThermalCondition:
    """base with the edits in d (tolerant: junk dropped, numbers clamped; a fog thinner than 20 m is 20 m)."""
    if not isinstance(d, dict):
        return base
    kw = {}
    for k, (lo, hi) in COND_LIMITS.items():
        v = _num(d.get(k)) if k in d else None
        if v is None:
            continue
        kw[k] = 0.0 if k == 'visibility_m' and v <= 0 else min(max(v, 20.0 if k == 'visibility_m' else lo), hi)
    if d.get('sky') in ('clear', 'overcast'):
        kw['sky'] = d['sky']
    if d.get('fog') in FOGS:
        kw['fog'] = d['fog']
    return replace(base, **kw)


# =========================================================================== settings
LIMITS = {'snr_min': (1.0, 50.0), 'lateral_m': (-10.0, 10.0), 'max_range_m': (20.0, 2000.0), 'samples': (40, 800),
          'skin_share': (0.0, 0.5), 'met_w_m2': (46.0, 232.0), 'air_speed': (0.0, 10.0),
          'clothing_value': (1.0, 9.5), 'bg_reflectance': (0.01, 0.9), 'dark_current_e_s': (0.0, 1e6),
          'assumed_f_number': (0.6, 8.0), 'assumed_transmission': (0.1, 1.0)}
INTS = ('samples',)


@dataclass
class ThermalSettings:
    conditions: list = field(default_factory=lambda: list(COND_IDS))
    params: dict = field(default_factory=dict)       # {condition id: {field: value}}: edits of the presets
    classes: list = field(default_factory=lambda: list(CLASSES))
    criteria: str = 'dri'              # dri | johnson
    snr_min: float = ROSE_K
    lateral_m: float = -1.5            # where the pedestrian stands: to the camera axis's left (negative: right)
    max_range_m: float = 1000.0
    samples: int = 360                 # ranges, log-spaced from 1 m
    skin_share: float = 0.10           # share of the pedestrian's silhouette that is exposed skin
    met_w_m2: float = 140.0            # walking at about 3 km/h (ISO 7730 Annex B)
    air_speed: float = 1.0             # air speed relative to the pedestrian, m/s
    clothing: str = 'medium'           # dark | medium | light (Munsell value of the clothing, visible cameras)
    clothing_value: float = 0.0        # 0 = the preset's; else a Munsell value
    beam_model: str = 'typical'        # typical | r112_min
    bg_reflectance: float = 0.10       # the scene behind the pedestrian (visible cameras), typical
    dark_current_e_s: float = 25.0     # visible sensor dark current at operating temperature, typical
    assumed_f_number: float = 1.0      # a thermal lens whose f-number is not stated
    assumed_transmission: float = 0.90  # a thermal lens whose transmission is not stated
    windows: dict = field(default_factory=dict)       # {camera uid: WINDOWS id} in front of a thermal camera

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d: dict | None) -> 'ThermalSettings':
        """Tolerant of junk: unknown keys ignored, numbers clamped to LIMITS, ids checked."""
        s = ThermalSettings()
        d = d if isinstance(d, dict) else {}
        c = d.get('conditions')
        if isinstance(c, (list, tuple)):
            keep = {x for x in c if x in COND}
            s.conditions = [x for x in COND_IDS if x in keep]
        k = d.get('classes')
        if isinstance(k, (list, tuple)):
            keep = {x for x in k if x in CLASSES}
            s.classes = [x for x in CLASSES if x in keep] or list(CLASSES)
        params = d.get('params')
        if isinstance(params, dict):
            out = {}
            for cid, p in params.items():
                if cid in COND and isinstance(p, dict):
                    cc = clean_condition(COND[cid], p)
                    diff = {f: getattr(cc, f) for f in COND_EDITABLE if getattr(cc, f) != getattr(COND[cid], f)}
                    if diff:
                        out[cid] = diff
            s.params = out
        if d.get('criteria') in CRITERIA:
            s.criteria = d['criteria']
        if d.get('clothing') in CLOTHING:
            s.clothing = d['clothing']
        if d.get('beam_model') in BEAM_MODELS:
            s.beam_model = d['beam_model']
        w = d.get('windows')
        if isinstance(w, dict):
            s.windows = {str(u)[:40]: v for u, v in list(w.items())[:64] if v in WINDOWS and v != 'none'}
        for key, (lo, hi) in LIMITS.items():
            if key in d:
                v = _num(d[key])
                if v is None:
                    continue
                if key == 'clothing_value' and v <= 0:
                    s.clothing_value = 0.0
                    continue
                v = min(max(v, lo), hi)
                setattr(s, key, int(round(v)) if key in INTS else v)
        return s

    def condition(self, cid) -> ThermalCondition:
        return clean_condition(COND.get(cid, CONDITIONS[0]), self.params.get(cid, {}))

    def selected_conditions(self) -> list:
        return [self.condition(c) for c in self.conditions]

    def reflectance(self) -> float:
        return munsell_reflectance(self.clothing_value or CLOTHING[self.clothing])


def settings_of(project) -> ThermalSettings:
    return ThermalSettings.from_dict(getattr(getattr(project, 'analysis', None), 'thermal', None))


# =========================================================================== library
_FLIR_BOSONPLUS = ('Teledyne FLIR Boson+ Product Datasheet, Doc. 102-2013-45, Release 117 (16 Sep 2026), '
                   'oem.flir.com/bosonplus')
_FLIR_BOSON_SLOW = ('the original Boson datasheet (102-2013-40, Release 340, March 2021) documents "Slow" export '
                    'variants fixed below 9 Hz (8.6 Hz) "to comply with US export regulations"')
_FLIR_ADK = 'FLIR ADK datasheet 18-2872-OEM-COR-ADK (01/07/19), www.flir.com/adk'
_LYNRED_PICO = ('Lynred PICO640S datasheet (ref. 06/2020/01), '
                'www.lynred.com/sites/default/files/2021-10/PICO640S%20datasheet.pdf')
_FLIR_TURA = 'Teledyne FLIR Tura product sheet (01/14/2026), oem.flir.com/TURA'

# Boson+ Table 11: config → (H, V, D °, F/#, focal mm, transmission, distortion %, MTF at Nyquist, shutterless)
_BOSONPLUS_640 = {
    'AS95': (95.0, 77.0, 130.5, 1.10, 4.9, 0.90, 56, 0.34, True),
    'AS50': (49.9, 39.3, 65.9, 1.01, 9.2, 0.90, 18, 0.42, True),
    'A032': (32.0, 25.6, 40.8, 1.00, 14.0, 0.92, 3, 0.42, False),
    'AS32': (32.0, 25.7, 40.9, 1.01, 13.6, 0.90, 3, 0.42, True),
    'A024': (24.3, 19.5, 31.3, 1.04, 18.0, 0.90, 2, 0.43, False),
    'AS24': (24.3, 19.2, 31.3, 1.09, 18.0, 0.90, 2, 0.42, True),
    'A018': (18.0, 14.4, 23.1, 1.00, 24.0, 0.89, 2, 0.41, False),
    'HS14': (13.5, 10.8, 17.2, 1.00, 32.0, 0.87, 2, 0.42, True),
    'A012': (12.2, 9.8, 15.7, 1.00, 36.0, 0.90, 1, 0.40, False),
    'A008': (8.0, 6.4, 10.3, 1.01, 55.0, 0.89, 2, 0.40, False),
}
_BOSONPLUS_320 = {
    'H092': (91.9, 74.0, 124.8, 1.00, 2.3, 0.87, 43, 0.42, False),
    'A050': (50.0, 40.0, 65.9, 1.00, 4.3, 0.93, 12, 0.42, False),
    'AS50': (50.0, 40.0, 65.0, 1.10, 4.5, 0.93, 14, 0.31, False),
    'A034': (34.1, 27.3, 43.6, 1.01, 6.3, 0.94, 1, 0.43, False),
    'A024': (24.1, 19.2, 30.6, 1.02, 9.1, 0.82, 1, 0.42, False),
    'A016': (16.0, 12.8, 20.4, 1.01, 13.8, 0.84, 3, 0.46, False),
    'A012': (12.2, 9.7, 15.7, 1.04, 18.0, 0.90, 1, 0.43, False),
    'A004': (4.0, 3.2, 5.2, 1.01, 55.0, 0.86, 1, 0.40, False),
}
# Boson+ Table 12: camera length (mm) and weight (g) with the lens (22xxx shuttered, 23xxx shutterless where given)
_BOSONPLUS_MECH = {
    ('640', 'AS95'): (28, 20), ('640', 'AS50'): (40, 31), ('640', 'A032'): (35, 39), ('640', 'AS32'): (33, 23),
    ('640', 'A024'): (38, 41), ('640', 'AS24'): (37, 27), ('640', 'A018'): (45, 52), ('640', 'HS14'): (52, 53),
    ('640', 'A012'): (64, 132), ('640', 'A008'): (75, 193),
    ('320', 'H092'): (33, 24), ('320', 'A050'): (27, 18), ('320', 'AS50'): (27, 16), ('320', 'A034'): (27, 17),
    ('320', 'A024'): (27, 16), ('320', 'A016'): (28, 24), ('320', 'A012'): (38, 38), ('320', 'A004'): (75, 192),
}
LENS_SPECS: dict = {}          # built-in thermal lens id → {'transmission', 'distortion_pct', 'mtf_nyquist', ...}


def _tmod(mid, vendor, name, res, pitch, fps, *, url, note, params, interface='', size=(0.0, 0.0, 0.0), weight=0.0,
          power=0.0, temp=(), ip='', sync=False, cables=(3.0,)):
    from .lens import Module
    p = {'band_um': [8.0, 14.0]}
    p.update(params)
    return Module(id=mid, name=name, vendor=vendor, sensor=p.get('detector', 'Uncooled microbolometer'),
                  width_px=int(res[0]), height_px=int(res[1]), pixel_um=float(pitch), optical_format='',
                  interface=interface, fps=float(fps), hw_sync=bool(sync), size_mm=tuple(float(v) for v in size),
                  weight_g=float(weight), cables_m=tuple(float(c) for c in cables), power_w=float(power),
                  builtin=True, shutter='', dynamic_range_db=0.0, ip_rating=ip,
                  temp_c=tuple(float(t) for t in temp), chroma='monochrome', url=url,
                  note=f'Figures from {note} Source: {url} (read {THERMAL_CHECKED}).', modality='lwir',
                  modality_params=p)


def _stated_base(focal, sheet, hs):
    from .lens import FIT_BASES, theta_of_r

    def err(b):
        return sum((2 * math.degrees(float(theta_of_r(h, focal, b))) - v) ** 2 for v, h in zip(sheet, hs) if v)
    wide = max(sheet) / 2 >= 80
    return min((b for b in FIT_BASES if not (wide and b == 'rectilinear')), key=err)


def _tlens(lid, name, m, *, sheet, focal=0.0, fno=0.0, focus=0.0, basis, spec=None, url=''):
    """A thermal lens: with a stated focal length and FOV the distortion terms are fitted to the FOV on that focal
    length (core.lens's method for other makers' lenses); with a FOV only the focal length is derived from it."""
    from .lens import Lens, best_base, fit_sheet, half_sizes, model_fov, monotonic, resolve, theta_of_r
    hs = half_sizes(m)
    sheet = tuple(float(v or 0.0) for v in sheet)
    b0, f0, _ = best_base(sheet, hs)
    tries = ([(round(float(focal), 3), _stated_base(float(focal), sheet, hs))] if focal else []) + \
        [(round(f0, 3), b0)]
    for f, b in tries:
        fit = tuple(v if v else min(2 * math.degrees(float(theta_of_r(h, f, b))), 190.0) for v, h in zip(sheet, hs))
        kb = fit_sheet(f, fit, hs, base=b)
        probe = Lens(id=lid, module_id=m.id, focal_mm=f, projection='kannala_brandt', kb=kb, sheet_fov=sheet)
        err = max((abs(g - w) for g, w in zip(model_fov(m, probe, circle=True), sheet) if w), default=0.0)
        if err <= 0.5 and monotonic(f, kb, resolve(m, probe).max_theta):
            break
    derived = not focal or f != round(float(focal), 3)
    shown = ' × '.join(f'{v:g}° {a}' for v, a in zip(sheet, 'HVD') if v)
    gaps = ' and '.join(a for v, a in zip(sheet, 'HVD') if not v)
    how = (f'FOV {shown}' + (f', focal length {focal:g} mm' if focal else '') + (f', F/{fno:g}' if fno else '')
           + ' as stated; '
           + (f'no lens curve that keeps rising reproduces that FOV at the stated {focal:g} mm (a nominal figure), so '
              if focal and derived else '')
           + (f'the focal length ({f:.3g} mm) is derived from the FOV with {"an" if b[0] in "aeiou" else "a"} {b} '
              'base projection; ' if derived else '')
           + 'the distortion terms are fitted to reproduce the FOV'
           + (f' ({gaps} not stated: there the {b} projection is followed).' if gaps else '.'))
    label = 'fisheye' if sheet[0] >= 180 else 'wide' if sheet[0] >= 100 else 'standard' if sheet[0] >= 70 else 'narrow'
    ln = Lens(id=lid, name=name, label=label, module_id=m.id, focal_mm=f, f_number=float(fno), focus_near_m=focus,
              projection='kannala_brandt', kb=kb, sheet_fov=sheet, builtin=True,
              note=f'{how} {basis}. Source: {url} (read {THERMAL_CHECKED}).')
    if ln.f_number <= 0:
        ln.note += ' Aperture not stated.'
    LENS_SPECS[lid] = dict(spec or {})
    return ln


def _modules() -> list:
    out = []
    for fmt, res in (('640', (640, 512)), ('320', (320, 256))):
        out.append(_tmod(
            f'flir_bosonplus_{fmt}', 'Teledyne FLIR', f'Boson+ {fmt}', res, 12.0, 60.0, interface='CMOS / MIPI / USB3',
            url='https://oem.flir.com/bosonplus', size=(21.0, 21.0, 0.0), power=1.55, temp=(-40, 80), sync=True,
            note=f'{_FLIR_BOSONPLUS}: Table 1. Uncooled VOx microbolometer core, {res[0]} × {res[1]} at 12 µm, '
                 '8–14 µm, up to 60 Hz (selectable 60, 30, 20, 15, 12, 10, 8.6, 4.3 Hz). NEDT ≤ 20 mK (industrial '
                 'grade; professional ≤ 30 mK) at f/1.0 lensless, high gain, 20 °C, imaging a 30 °C background, '
                 'averager off (Table 13; scaled by (f/#)²/τ with a lens). Power 0.6–1.55 W by configuration and '
                 'temperature (1.55 W used). The 21 × 21 × 11 mm lensless core; length and weight with each lens in '
                 'the lens note (Table 12). External sync. ' + ('Shuttered (all 320 configurations have a shutter).'
                                                                if fmt == '320' else
                                                                'Shuttered and shutterless ("S") configurations.')
                 + f' Export: {_FLIR_BOSON_SLOW}; Release 117 lists 8.6 Hz among the selectable rates.',
            params={'detector': 'Uncooled VOx microbolometer', 'netd_mk': 20.0, 'netd_f_number': 1.0,
                    'netd_scene_k': 303.15, 'netd_fps': 60.0,
                    'netd_note': 'Industrial grade ≤ 20 mK, professional ≤ 30 mK (Table 13)',
                    'frame_rates_hz': [60.0, 30.0, 20.0, 15.0, 12.0, 10.0, 8.6, 4.3], 'export_fps_hz': 8.6,
                    'shutterless': fmt == '640', 'time_constant_ms': 8.0, 'source': _FLIR_BOSONPLUS}))
    out.append(_tmod(
        'flir_adk', 'Teledyne FLIR', 'ADK (thermal automotive development kit)', (640, 512), 12.0, 60.0,
        interface='USB 2.0 / GMSL / Ethernet / FPD-Link', url='https://www.flir.com/adk', size=(35.0, 40.0, 47.0),
        weight=100.0, power=1.0, temp=(-40, 85), ip='IP67',
        note=f'{_FLIR_ADK}. A Boson (uncooled VOx, 640 × 512, 12 µm) in an IP67 enclosure with a heated window, '
             '8–14 µm, thermal sensitivity < 50 mK (the camera\'s figure: no f-number stated), full frame 30 & 60 Hz, '
             '9 Hz optional (export). 1 W without the heater, about 4 W average and 12 W maximum with it. 35 × 40 × '
             '47 mm (W × H × D), about 100 g. "Detection and classification range of up to 200 meters (adult '
             'pedestrian)" (not tied to a lens).',
        params={'detector': 'Uncooled VOx microbolometer (Boson)', 'netd_mk': 50.0,
                'netd_note': 'Thermal sensitivity < 50 mK as stated for the camera (no f-number or scene given)',
                'frame_rates_hz': [60.0, 30.0], 'export_fps_hz': 9.0, 'heated_window': True, 'source': _FLIR_ADK}))
    out.append(_tmod(
        'lynred_pico640s', 'Lynred', 'PICO640S (detector)', (640, 480), 17.0, 120.0, interface='',
        url='https://www.lynred.com/products/pico640s', power=0.13, temp=(-40, 85),
        note=f'{_LYNRED_PICO}. An uncooled infrared detector (no lens, no camera electronics): 640 × 480 at 17 µm, '
             'thermal sensitivity < 30 mK (f/1, 300 K, 30 Hz), frame rate up to 120 Hz, thermal time constant 12 ms, '
             'TECless and shutterless compatible, under 130 mW, −40 to +85 °C, external frame synchronisation. '
             'Detector material and spectral band are not stated on the sheet (8–14 µm used, the LWIR band).',
        params={'detector': 'Uncooled microbolometer (material not stated)', 'netd_mk': 30.0, 'netd_f_number': 1.0,
                'netd_scene_k': 300.0, 'netd_fps': 30.0, 'netd_note': '< 30 mK (f/1, 300 K, 30 Hz)',
                'frame_rates_hz': [120.0], 'shutterless': True, 'time_constant_ms': 12.0, 'source': _LYNRED_PICO},
        sync=True))
    for hfov, size, weight in ((24, (22.0, 34.0, 62.0), 46.5), (42, (20.0, 34.0, 53.0), 23.2),
                               (70, (20.0, 34.0, 52.0), 25.6)):
        out.append(_tmod(
            f'flir_tura_{hfov}', 'Teledyne FLIR', f'Tura {hfov}° (ASIL B automotive)', (640, 512), 12.0, 60.0,
            interface='GMSL2', url='https://oem.flir.com/tura', size=(size[1], size[0], size[2]), weight=weight,
            power=1.5, temp=(-40, 85), ip='IP6K9K', sync=True,
            note=f'{_FLIR_TURA}. ISO 26262 ASIL B LWIR camera, 640 × 512 at 12 µm, 8–14 µm, NEdT 40 mK after filters '
                 'normalised to f/1 clear aperture at 60 Hz, selectable 60 / 30 / 20 / 15 / 12 / 10 / 8.6 Hz, '
                 'shutterless, heated window (2 W maximum), GMSL2 (MAX96717) or FPD-Link (DS90UB935-Q1) over mini '
                 f'FAKRA. About 1.5 W at 25 °C (2.3 W at 85 °C). {hfov}° HFOV: {size[2]:g} × {size[1]:g} × '
                 f'{size[0]:g} mm (L × W × H), {weight:g} g. IP6K9K. Lens focal length and f-number not stated.',
            params={'detector': 'Uncooled microbolometer (AEC-Q qualified)', 'netd_mk': 40.0, 'netd_f_number': 1.0,
                    'netd_fps': 60.0, 'netd_note': '40 mK after filters, normalised f/1 clear aperture at 60 Hz '
                                                   '(scene temperature not stated: 300 K used)',
                    'frame_rates_hz': [60.0, 30.0, 20.0, 15.0, 12.0, 10.0, 8.6], 'shutterless': True,
                    'heated_window': True, 'source': _FLIR_TURA}))
    return out


def _lenses(M) -> list:
    out = []
    for fmt, table in (('640', _BOSONPLUS_640), ('320', _BOSONPLUS_320)):
        m = M[f'flir_bosonplus_{fmt}']
        for cfg, (h, v, dg, fno, f, tau, dist, mtf, sless) in table.items():
            length, weight = _BOSONPLUS_MECH[(fmt, cfg)]
            out.append(_tlens(
                f'bosonplus{fmt}_{cfg.lower()}', f'{cfg} ({h:g}°)', m, sheet=(h, v, dg), focal=f, fno=fno,
                basis=f'{_FLIR_BOSONPLUS}, Table 11 (config 22{fmt} {cfg}): average transmission {tau * 100:.0f} %, '
                      f'distortion < {dist} %, MTF {mtf * 100:.0f} % at Nyquist on axis{", shutterless" if sless else ""}'
                      f'; Table 12: camera {length} mm long, {weight} g with this lens. Lens material not stated',
                url='https://oem.flir.com/bosonplus',
                spec={'transmission': tau, 'distortion_pct': float(dist), 'mtf_nyquist': mtf,
                      'length_mm': float(length), 'weight_g': float(weight), 'shutterless': sless}))
    for hfov in (75, 50, 32, 24):
        out.append(_tlens(f'adk_{hfov}', f'{hfov}° HFOV', M['flir_adk'], sheet=(float(hfov), 0.0, 0.0),
                          basis=f'{_FLIR_ADK}: the HFOV options 75°, 50°, 32° and 24°. Focal length, f-number and '
                                'transmission not stated', url='https://www.flir.com/adk'))
    pico = M['lynred_pico640s']
    for f, h, v, rng in ((19.0, 32.0, 21.0, 180.0), (50.0, 12.4, 9.3, 455.0), (75.0, 8.3, 6.2, 700.0)):
        out.append(_tlens(
            f'pico640s_{int(f)}mm', f'{f:g} mm ({h:g}°)', pico, sheet=(h, 0.0 if f == 19.0 else v, 0.0), focal=f,
            basis=f'{_LYNRED_PICO}: the {f:g} mm lens of its range figure ({h:g}° × {v:g}° as printed'
                  + ('; 21° V cannot be a 19 mm lens\'s on a 480 × 17 µm sensor (24.2°): V left out' if f == 19.0
                     else '') + f'), recognition of a 1.80 × 0.50 m human at {rng:g} m "for Johnson\'s criteria, '
                  'target ΔT = 2 K, perfect atmospheric and optics transmissions, theoretical square pixel". '
                  'F-number and transmission not stated (the sheet assumes perfect optics)',
            url='https://www.lynred.com/sites/default/files/2021-10/PICO640S%20datasheet.pdf',
            spec={'recognition_m': rng}))
    for hfov in (24, 42, 70):
        out.append(_tlens(f'tura_{hfov}', f'{hfov}° HFOV', M[f'flir_tura_{hfov}'], sheet=(float(hfov), 0.0, 0.0),
                          basis=f'{_FLIR_TURA}: HFOV {hfov}°. Focal length, f-number and transmission not stated',
                          url='https://oem.flir.com/tura'))
    return out


@functools.lru_cache(maxsize=1)
def _library():
    mods = _modules()
    return tuple(mods), tuple(_lenses({m.id: m for m in mods}))


def library(what='modules') -> list:
    """The thermal products for core.lens's built-in library: what = 'modules' or 'lenses' (fresh copies)."""
    import copy
    mods, lenses = _library()
    if what == 'modules':
        return [copy.deepcopy(m) for m in mods]
    if what == 'lenses':
        return [copy.deepcopy(ln) for ln in lenses]
    return []


def is_thermal(module) -> bool:
    return getattr(module, 'modality', 'rgb') == 'lwir'


# =========================================================================== the camera's thermal figures
def camera_figures(module, lens, st: ThermalSettings, window='none') -> dict:
    """What the radiometry needs of a thermal camera: the band, NETD and how it scales, f-number and transmission,
    the noise-equivalent radiance difference at the focal plane's reference; the figures not stated (needed) and the
    assumptions used for them."""
    p = dict(getattr(module, 'modality_params', None) or {})
    band = p.get('band_um') if isinstance(p.get('band_um'), (list, tuple)) and len(p.get('band_um')) == 2 else None
    band = (float(band[0]), float(band[1])) if band and 0 < float(band[0]) < float(band[1]) else LWIR_BAND
    needed, assumed = [], []
    netd = _num(p.get('netd_mk'))
    if netd is None or netd <= 0:
        needed.append('NETD')
        netd = 50.0
        assumed.append('NETD 50 mK (typical uncooled)')
    n_ref = _num(p.get('netd_f_number'))
    t_ref = _num(p.get('netd_scene_k'))
    if t_ref is None or t_ref <= 0:
        t_ref = 300.0
        if n_ref:
            assumed.append('NETD scene temperature 300 K')
    N = float(getattr(lens, 'f_number', 0.0) or 0.0)
    spec = LENS_SPECS.get(getattr(lens, 'id', ''), {})
    tau = _num(spec.get('transmission'))
    if N <= 0:
        needed.append('lens f-number')
        N = st.assumed_f_number
        assumed.append(f'F/{N:g}')
    if tau is None:
        needed.append('lens transmission')
        tau = st.assumed_transmission
        assumed.append(f'lens transmission {tau * 100:.0f} %')
    dldt = float(band_dl_dt(t_ref, band))
    if n_ref and n_ref > 0:
        # NETD normalised to f/N_ref, clear aperture: the camera's NEΔL (object space) scales by (N/N_ref)²/τ
        ne_dl = netd * 1e-3 * dldt * (N / n_ref) ** 2 / max(tau, 1e-6)
        how = f'{netd:g} mK at f/{n_ref:g} × (F{N:.2f}/f{n_ref:g})² / τ {tau:.2f}'
    else:
        ne_dl = netd * 1e-3 * dldt
        how = f'{netd:g} mK as stated for the camera with its lens'
    tw = WINDOWS.get(window, WINDOWS['none'])[1]
    return {'band_um': list(band), 'netd_mk': netd, 'netd_f_number': n_ref, 'netd_scene_k': t_ref,
            'f_number': N, 'transmission': tau, 'window': window, 'window_transmission': tw,
            'dl_dt': dldt, 'ne_dl': ne_dl, 'netd_eff_mk': ne_dl / dldt * 1e3, 'how': how,
            'needed': needed, 'assumed': assumed}


def detector_irradiance(L, f_number, transmission=1.0):
    """Focal-plane irradiance W/m² from object radiance L W/(m²·sr): E = π·τ·L/(4N²) (an aplanatic lens)."""
    return math.pi * transmission * np.asarray(L, float) / (4.0 * float(f_number) ** 2)


# =========================================================================== scene radiances
def scene(cond: ThermalCondition, st: ThermalSettings) -> dict:
    """The temperatures of the condition (°C): air, road, sky, mean radiant, clothing, skin."""
    t_sky = sky_temperature_k(cond.t_air_c, cond.sky) - T0
    t_rad = 0.5 * (t_sky + cond.t_road_c)        # a standing person: the sky above, the road below
    t_cl = clothing_temperature_c(cond.t_air_c, t_rad, st.met_w_m2, cond.clo, st.air_speed)
    t_sk = skin_temperature_c(st.met_w_m2)
    return {'air': cond.t_air_c, 'road': cond.t_road_c, 'sky': t_sky, 'radiant': t_rad, 'clothing': t_cl,
            'skin': t_sk}


def _part_temp(src, temps):
    if src in temps:
        return temps[src]
    if src.startswith('air'):
        return temps['air'] + float(src[3:] or 0.0)
    raise KeyError(src)


def class_radiance(cls, temps, st: ThermalSettings, lam) -> np.ndarray:
    """Spectral radiance (at nodes lam) of the class's silhouette: its parts' ε·B(T) + (1 − ε)·L_env, area-weighted;
    L_env the mean radiant surroundings."""
    spec = CLASSES[cls]
    L_env = planck(lam, temps['radiant'] + T0)
    out = np.zeros_like(lam, dtype=float)
    shares = []
    for surf, share, src in spec['parts']:
        s = st.skin_share if share == 'skin_share' else (1.0 - st.skin_share if share == 'rest' else float(share))
        shares.append((surf, s, src))
    tot = sum(s for _, s, _ in shares) or 1.0
    for surf, s, src in shares:
        eps = SURFACES[surf][0]
        out += s / tot * surface_radiance(eps, _part_temp(src, temps) + T0, L_env, lam=lam)
    return out


def background_radiance(road_share, temps, lam) -> np.ndarray:
    """Spectral radiance of what lies behind the target: road_share of it the road (reflecting the sky), the rest the
    far scene at the air temperature."""
    L_sky = planck(lam, temps['sky'] + T0)
    road = surface_radiance(SURFACES['asphalt'][0], temps['road'] + T0, L_sky, lam=lam)
    far = surface_radiance(SURFACES['background'][0], temps['air'] + T0, planck(lam, temps['radiant'] + T0), lam=lam)
    return road_share * road + (1.0 - road_share) * far


def road_share(cam_z, cls) -> float:
    """Share of the class's silhouette seen against the road: the part below the camera's height (a line of sight
    past the target that slopes down meets the road behind it)."""
    h = CLASSES[cls]['height']
    return float(min(max(float(cam_z) / h, 0.0), 1.0))


def path_transmission(cond: ThermalCondition, d, lam):
    """(N, L) spectral transmission over distances d at nodes lam: humid air × fog (Mie-scaled) × rain/snow."""
    from . import weather as W
    d = np.asarray(d, float)
    tau = humidity_transmission(d, cond.rh_pct, cond.t_air_c)[:, None] * np.ones((1, len(lam)))
    sig = rain_extinction(cond.rain_mm_h)
    if cond.snow_mm_h > 0:
        sig += float(W.snow_optical_db_km(cond.snow_mm_h, 10000.0, 'wet' if cond.t_air_c > 0.5 else 'dry')) \
            / W.DB_PER_NEPER / 1000.0
    tau = tau * np.exp(-sig * d)[:, None]
    if cond.visibility_m > 0:
        s_vis = W.MOR_LN / cond.visibility_m
        tau = tau * np.exp(-np.outer(d, s_vis * fog_ratio(cond.fog, lam)))
    return tau


def apparent_contrast(cls, cond, st, cam_z, d, band=LWIR_BAND, temps=None) -> dict:
    """The target's radiance difference against its background at the camera over distances d: ΔL (W/(m²·sr), signed),
    the apparent ΔT (K, at the background's temperature) and the band transmission."""
    lam, w = band_nodes(band, SPECTRAL_NODES)
    temps = temps or scene(cond, st)
    rs = road_share(cam_z, cls)
    dL = class_radiance(cls, temps, st, lam) - background_radiance(rs, temps, lam)
    tau = path_transmission(cond, d, lam)
    dL_app = (tau * dL[None, :]) @ w
    L_bg = background_radiance(rs, temps, lam) @ w
    T_bg = apparent_temperature(L_bg, band)
    dldt = float(band_dl_dt(T_bg, band))
    tau_band = (tau * planck(lam, T_bg)[None, :]) @ w / max(float(band_radiance(T_bg, band)), 1e-12)
    return {'dL': dL_app, 'dT': dL_app / dldt, 'dT0': float(dL @ w) / dldt, 'tau': tau_band, 'road_share': rs,
            'T_bg_c': T_bg - T0}


# =========================================================================== headlamps (UN R112)
def r112_table(lamp_height_m=0.75) -> list:
    """The regulation's class B test points with the illuminance each puts at the range it is named for: E = I/d² on
    a surface facing the lamp (per lamp, and both lamps), the minimum (or maximum) intensity from the table."""
    rows = []
    for beam, table in (('low', R112_PASSING), ('high', R112_DRIVING)):
        for name, t in table.items():
            lim = t.get('min', t.get('max'))
            kind = 'min' if 'min' in t else 'max'
            rng = t.get('range_m') or (None if beam == 'low' else 100.0 if name == 'Imax' else None)
            e = lim / rng ** 2 if rng else None
            rows.append({'beam': beam, 'point': name, 'h_deg': t.get('h'), 'v_deg': t.get('v'), 'limit': kind,
                         'cd': lim, 'cd_max': t.get('max') if kind == 'min' and 'max' in t else None,
                         'range_m': rng, 'lux_per_lamp': e, 'lux_two_lamps': 2 * e if e else None,
                         'road_m': (lamp_height_m / math.tan(math.radians(-t['v'])) if t.get('v', 0) < 0 else None)})
    return rows


@functools.lru_cache(maxsize=32)
def r112_scale(beam='low', traffic='right', low_cd=18000.0, high_cd=60000.0) -> float:
    """The factor that scales core.weather's typical pattern down (or up) until its weakest R112 minimum sits exactly
    at the regulation's figure: the dimmest lamp R112 approves of that shape."""
    from . import weather as W

    def I(h, v):
        return float(W.headlamp_cd(h, v, beam, traffic, low_cd, high_cd))
    s = traffic == 'left'
    ratios = []
    if beam == 'low':
        for t in R112_PASSING.values():
            if 'min' in t:
                ratios.append(t['min'] / max(I(-t['h'] if s else t['h'], t['v']), 1e-9))
        z = R112_ZONES['IV']
        zmin = min(I(h, v) for h in np.linspace(*z['h'], 9) for v in np.linspace(*z['v'], 4))
        ratios.append(z['min'] / max(zmin, 1e-9))
    else:
        imax = max(I(h, v) for h in np.linspace(-2, 2, 17) for v in np.linspace(-1.5, 1.5, 13))
        ratios.append(R112_DRIVING['Imax']['min'] / imax)
        for name in ('H-5L', 'H-2.5L', 'H-2.5R', 'H-5R'):
            t = R112_DRIVING[name]
            ratios.append(t['min'] / max(I(-t['h'] if s else t['h'], t['v']), 1e-9))
    return float(max(ratios))


def lamp_cd(h_deg, v_deg, beam, model, wst):
    """One headlamp's intensity (cd) toward (h left +, v up +): core.weather's typical pattern, or that pattern scaled
    to R112's minimum (r112_scale)."""
    from . import weather as W
    I = W.headlamp_cd(h_deg, v_deg, beam, wst.traffic, wst.low_beam_cd, wst.high_beam_cd)
    if model == 'r112_min':
        I = I * r112_scale(beam, wst.traffic, float(wst.low_beam_cd), float(wst.high_beam_cd))
    return I


def lamp_illuminance(P, lamps, beam, model, wst, sigma=0.0):
    """Illuminance (lux) at points P (..., 3) on a surface facing each lamp from the two headlamps, through the
    extinction sigma (1/m)."""
    P = np.asarray(P, float)
    E = np.zeros(P.shape[:-1])
    for L in np.asarray(lamps, float):
        d = P - L
        r = np.maximum(np.linalg.norm(d, axis=-1), 1e-3)
        h = np.degrees(np.arctan2(d[..., 1], d[..., 0]))
        v = np.degrees(np.arcsin(np.clip(d[..., 2] / r, -1, 1)))
        E = E + lamp_cd(h, v, beam, model, wst) / (r * r) * np.exp(-sigma * r)
    return E


def lit_illuminance(G, height, lamps, beam, model, wst, ambient, sigma=0.0):
    """The illuminance that at least lit_share of a target's height (standing at ground points G (N, 2)) receives:
    core.weather's light test (HEIGHT_SAMPLES points up it, the k-th brightest)."""
    from . import weather as W
    G = np.asarray(G, float).reshape(-1, 2)
    nS = W.HEIGHT_SAMPLES
    z = height * (np.arange(nS) + 0.5) / nS
    P = np.concatenate([np.broadcast_to(G[:, None, :], (len(G), nS, 2)),
                        np.broadcast_to(z[None, :, None], (len(G), nS, 1))], axis=-1)
    E = lamp_illuminance(P, lamps, beam, model, wst, sigma) + float(ambient)
    k = max(1, int(math.ceil(wst.lit_share * nS - 1e-9)))
    return np.sort(E, axis=1)[:, nS - k]


# =========================================================================== the study
@dataclass
class ThermalResult:
    settings: ThermalSettings
    conditions: list                    # [ThermalCondition]
    cams: list                          # per visible camera: dict (tag, uid, slot, modality, figures, glass …)
    ranges: np.ndarray                  # (R,) m, along each camera's heading
    rows: list                          # range table rows (see run)
    curves: dict                        # {(tag, condition id, beam): {'snr': (R,), 'dT': (R,) or 'lux': (R,)}}
    scenes: dict                        # {condition id: temperatures and the pedestrian's contrast at 0 m}
    illuminance: dict                   # {(beam, model): (R,) lux on the pedestrian on the kerb line}
    r112: list
    findings: list
    metrics: dict
    variant: str = ''
    variant_uid: str = ''

    def row(self, tag, cond, cls='pedestrian', level='detection', beam='') -> dict | None:
        return next((r for r in self.rows if r['tag'] == tag and r['condition'] == cond and r['class'] == cls
                     and r['level'] == level and r['beam'] == beam), None)

    def cam(self, tag) -> dict | None:
        return next((c for c in self.cams if c['tag'] == tag), None)

    def worst(self) -> str:
        return self.findings[0]['severity'] if self.findings else 'good'

    def first_to_see(self, cond, cls='pedestrian', level='detection') -> dict:
        """The forward-facing thermal and visible camera that sees the class farthest under the condition (visible:
        low beam): {'thermal': row, 'visible': row, 'winner': 'thermal' | 'visible' | 'tie' | ''}."""
        fwd = {c['tag'] for c in self.cams if c['forward']}

        def best(mod, beam):
            rs = [r for r in self.rows if r['condition'] == cond and r['class'] == cls and r['level'] == level
                  and r['tag'] in fwd and (r['modality'] == 'lwir') == (mod == 'lwir') and r['beam'] == beam]
            return max(rs, key=lambda r: r['range_m'], default=None)
        th, vi = best('lwir', ''), best('vis', 'low')
        if th is None or vi is None:
            win = 'thermal' if th is not None and th['range_m'] > 0 else 'visible' if vi is not None else ''
        else:
            win = 'tie' if abs(th['range_m'] - vi['range_m']) < 0.5 else (
                'thermal' if th['range_m'] > vi['range_m'] else 'visible')
        return {'thermal': th, 'visible': vi, 'winner': win}

    def to_dict(self):
        return to_doc(self)


def _margin_range(ranges, margins):
    """The far end of the first run of samples where every margin (ln ratios, ≥ 0 = met) holds, interpolated in range
    to the crossing of the margin that fails; and which margin it was. (0, name) when the target is never resolved."""
    names = list(margins)
    M = np.stack([np.asarray(margins[n], float) for n in names])            # (K, R)
    M = np.where(np.isfinite(M), M, -1e9)
    ok = (M >= 0).all(0)
    if not ok.any():
        k = int(np.argmin(M[:, 0])) if M.shape[1] else 0
        return 0.0, names[k]
    i0 = int(np.argmax(ok))
    bad = np.flatnonzero(~ok[i0:])
    if not len(bad):
        return float(ranges[-1]), 'range limit'
    j = i0 + int(bad[0])
    i = j - 1
    k = int(np.argmin(M[:, j]))
    mi, mj = M[k, i], M[k, j]
    t = mi / (mi - mj) if mi - mj > 1e-12 else 0.0
    t = min(max(t, 0.0), 1.0)
    return float(ranges[i] + t * (ranges[j] - ranges[i])), names[k]


def _forward(cam) -> bool:
    return abs(((float(cam.yaw) + 180.0) % 360.0) - 180.0) <= 30.0


def _glass_reports(project, engine, cams) -> dict:
    """{camera uid: core.materials glass report} for the cameras that look through the body's glass."""
    if project.vehicle.mesh is None or not cams:
        return {}
    try:
        from . import materials as MAT
        vg = engine.vehicle_geom(project) if engine is not None else None
        res = MAT.cached_run(project, vg)
    except Exception:                       # the materials study is optional here: no glass information
        return {}
    return {c['uid']: c for c in res.cameras}


def _visible_figures(module, lens, wst, st) -> dict:
    """A visible camera's photon budget figures: datasheet where stated (modality_params), else typical (needed)."""
    from . import weather as W
    p = dict(getattr(module, 'modality_params', None) or {})
    needed = []

    def fig(key, default, label):
        v = _num(p.get(key))
        if v is None or v < 0:
            needed.append(label)
            return float(default)
        return v
    qe = fig('qe', wst.qe, 'QE')
    rn = fig('read_noise_e', wst.read_noise_e, 'read noise')
    dc = fig('dark_current_e_s', st.dark_current_e_s, 'dark current')
    fl = W.camera_floor(module, lens, replace(wst, qe=qe, read_noise_e=rn))
    if not fl['f_number_given']:
        needed.append('lens f-number')
    t = fl['exposure_ms'] / 1000.0
    return {'qe': qe, 'read_noise_e': rn, 'dark_current_e_s': dc, 'exposure_ms': fl['exposure_ms'],
            'f_number': fl['f_number'], 'electrons_per_cd_m2': fl['electrons_per_cd_m2'],
            'dark_e': dc * t, 'needed': needed}


def run(project, engine=None, settings: ThermalSettings | None = None, variant=None, progress=None,
        cancel=None) -> ThermalResult | None:
    """The thermal study of a variant (the active one by default): per visible camera, condition and class the
    detection, recognition and identification range; thermal cameras from the LWIR radiometry, visible cameras from
    the headlamp photon budget (low and high beam). Returns None when cancelled."""
    from . import weather as W
    from .lens import resolve
    from .perception import ObjectClass, object_pixels
    from .materials import _project_for
    t0 = time.perf_counter()
    p = _project_for(project, variant)
    v = p.variant
    st = settings or settings_of(p)
    wst = W.settings_of(p)
    conds = st.selected_conditions()
    classes = list(st.classes)
    crit = CRITERIA[st.criteria]
    ranges = np.geomspace(1.0, float(st.max_range_m), int(st.samples))
    vg = engine.vehicle_geom(p) if engine is not None else None
    lamps = W.lamp_positions(vg, wst)
    # event cameras are core.event_camera's (no frame exposure to budget): left out
    cams = [c for c in v.cameras if c.visible and getattr(p.module_for(c), 'modality', 'rgb') != 'event']
    glass = _glass_reports(p, engine, cams)
    rho = st.reflectance()
    rows, curves, cam_docs = [], {}, []
    scenes = {}
    for c in conds:
        temps = scene(c, st)
        scenes[c.id] = {'temperatures_c': {k: round(float(x), 2) for k, x in temps.items()},
                        'pedestrian_dT0_k': {}}
    steps = max(1, len(cams) * max(1, len(conds)))
    done = 0
    for cam in cams:
        if cancel is not None and cancel():
            return None
        module, lens = p.module_for(cam), p.lens_for(cam)
        intr = resolve(module, lens)
        lwir = is_thermal(module)
        g = glass.get(cam.uid)
        win = st.windows.get(cam.uid, 'none')
        doc = {'uid': cam.uid, 'tag': cam.tag, 'slot': cam.slot, 'modality': getattr(module, 'modality', 'rgb'),
               'module': module.id, 'module_name': f'{module.vendor} {module.name}'.strip(), 'lens': lens.id,
               'lens_name': lens.name, 'width_px': module.width_px, 'height_px': module.height_px,
               'pixel_um': module.pixel_um, 'z': float(cam.z), 'yaw': float(cam.yaw), 'pitch': float(cam.pitch),
               'forward': _forward(cam), 'glass': (None if g is None else
                                                   {'class': g['class'], 'class_name': g['class_name'],
                                                    'transmission_visible': g['transmission']}),
               'window': win if lwir else 'none'}
        if lwir:
            if g is not None and win == 'none':
                win = 'glass'                 # looking through the body's glass with no IR window: opaque
            fig = camera_figures(module, lens, st, win)
            doc['figures'] = fig
            doc['needed'] = fig['needed']
            doc['window'] = win
        else:
            fig = _visible_figures(module, lens, wst, st)
            tvis = float(g['transmission']) if g is not None else 1.0
            fig['glass_transmission'] = tvis
            doc['figures'] = fig
            doc['needed'] = fig['needed']
        cam_docs.append(doc)
        # geometry: the target on the line along the camera's heading, lateral_m to its left
        a = math.radians(float(cam.yaw))
        u = np.array([math.cos(a), math.sin(a)])
        n = np.array([-math.sin(a), math.cos(a)])
        G = np.asarray([float(cam.x), float(cam.y)]) + ranges[:, None] * u[None, :] + st.lateral_m * n[None, :]
        # the occlusion cube counts the body's glass as body: a camera behind glass sees out through it here (what
        # the glass does to its light, or the LWIR, is the glass report's and the window's)
        cube = engine.cube(cam.pos()) if (engine is not None and vg is not None and g is None) else None
        geo = {}
        for k in classes:
            spec = CLASSES[k]
            hpx, wpx, seen = object_pixels(cam, intr, G, ObjectClass(k, spec['height'], spec['width'], 1.0), cube)
            crit_px = np.sqrt(np.maximum(hpx, 0.0) * np.maximum(wpx, 0.0))
            crit_px = np.where(seen & (wpx > 0), crit_px, 0.0)
            fill = np.clip(wpx, 0.0, 1.0) * np.clip(hpx, 0.0, 1.0)
            mid = np.c_[G, np.full(len(G), spec['height'] / 2)]
            d_cam = np.linalg.norm(mid - cam.pos(), axis=1)
            geo[k] = (crit_px, fill, seen, d_cam)
        for c in conds:
            if cancel is not None and cancel():
                return None
            temps = scene(c, st)
            beams = [''] if lwir else list(BEAMS)
            for beam in beams:
                sig = 0.0 if lwir else W.visible_extinction(c.weather(beam))
                for k in classes:
                    crit_px, fill, seen, d_cam = geo[k]
                    with np.errstate(divide='ignore'):
                        px_m = {lvl: np.log(np.maximum(crit_px, 1e-12) / crit[lvl]) for lvl in LEVELS}
                    if lwir:
                        ac = apparent_contrast(k, c, st, cam.z, d_cam, tuple(fig['band_um']), temps)
                        cnr = np.abs(ac['dL']) * fill * fig['window_transmission'] / fig['ne_dl']
                        snr_m = np.log(np.maximum(cnr, 1e-12) / st.snr_min)
                        margins_extra = {'thermal contrast': snr_m}
                        if k == 'pedestrian':
                            curves[(cam.tag, c.id, beam)] = {'snr': cnr, 'dT': ac['dT'], 'tau': ac['tau']}
                            scenes[c.id]['pedestrian_dT0_k'][cam.tag] = round(ac['dT0'], 3)
                            scenes[c.id].setdefault('road_share', {})[cam.tag] = round(ac['road_share'], 3)
                    else:
                        E = lit_illuminance(G, CLASSES[k]['height'], lamps, beam, st.beam_model, wst, c.ambient_lux,
                                            sig)
                        refl = rho if k != 'car' else W.object_look('car', wst)[1]
                        L = refl * E / math.pi * np.exp(-sig * d_cam)
                        L_bg = st.bg_reflectance * c.ambient_lux / math.pi * np.exp(-sig * d_cam)
                        ke = fig['electrons_per_cd_m2'] * fig['glass_transmission']
                        S, Sb = ke * L, ke * L_bg
                        noise = np.sqrt(fill * S + (1 - fill) * Sb + fig['dark_e'] + fig['read_noise_e'] ** 2)
                        snr = fill * np.maximum(S - Sb, 0.0) / np.maximum(noise, 1e-12)
                        c0 = W.object_look(k, wst)[0]
                        con_m = math.log(max(c0, 1e-9) / wst.contrast_threshold) - sig * d_cam
                        margins_extra = {'light': np.log(np.maximum(snr, 1e-12) / st.snr_min)}
                        if sig > 0:
                            margins_extra['fog contrast'] = con_m
                        if k == 'pedestrian':
                            curves[(cam.tag, c.id, beam)] = {'snr': snr, 'lux': E}
                    for lvl in LEVELS:
                        margins = {'view': np.where(seen, 1.0, -1.0), 'pixels': px_m[lvl], **margins_extra}
                        rng, lim = _margin_range(ranges, margins)
                        if lwir and fig['window_transmission'] <= 0:
                            lim = 'glass'
                        rows.append({'tag': cam.tag, 'uid': cam.uid, 'modality': 'lwir' if lwir else doc['modality'],
                                     'condition': c.id, 'class': k, 'level': lvl, 'beam': beam,
                                     'range_m': rng, 'limited_by': lim})
            done += 1
            if progress is not None:
                progress(done / steps)
    # the illuminance R112 puts on a pedestrian on the kerb line ahead (the lamps' axis), per beam and lamp model
    illum = {}
    if lamps is not None:
        mid = np.asarray(lamps, float).mean(0)
        Gk = np.c_[mid[0] + ranges, np.full(len(ranges), mid[1] + st.lateral_m)]
        for beam in BEAMS:
            for model in BEAM_MODELS:
                illum[(beam, model)] = lit_illuminance(Gk, CLASSES['pedestrian']['height'], lamps, beam, model,
                                                       wst, 0.0)
    res = ThermalResult(st, conds, cam_docs, ranges, rows, curves, scenes, illum, r112_table(), [],
                        {'seconds': time.perf_counter() - t0, 'cameras': len(cams),
                         'thermal_cameras': sum(1 for d in cam_docs if d['modality'] == 'lwir'),
                         'conditions': len(conds), 'clothing_reflectance': rho,
                         'r112_scale': {b: r112_scale(b, wst.traffic, float(wst.low_beam_cd),
                                                      float(wst.high_beam_cd)) for b in BEAMS}},
                        v.name, v.uid)
    res.findings = _findings(res)
    if progress is not None:
        progress(1.0)
    return res


# =========================================================================== findings and issues
SEVERITIES = ('critical', 'serious', 'warning', 'good')


def _finding(out, sev, kind, message, cams=()):
    out.append({'severity': sev, 'kind': kind, 'message': message, 'tags': [c['tag'] for c in cams],
                'uids': [c['uid'] for c in cams]})


def _findings(res: ThermalResult) -> list:
    out = []
    thermal = [c for c in res.cams if c['modality'] == 'lwir']
    for c in thermal:
        if c['window'] == 'glass':
            _finding(out, 'critical', 'glass',
                     f'{c["tag"]} is a thermal camera behind {c["glass"]["class_name"].lower() if c["glass"] else "glass"}'
                     ': glass is opaque from 8 to 14 µm (soda-lime silica absorbs beyond about 5 µm), so it sees '
                     'nothing. Mount it outside the glass, or behind a germanium or ZnSe window (set the window on '
                     'the Thermal page).', [c])
        if c['needed']:
            _finding(out, 'warning', 'needed',
                     f'{c["tag"]} ({c["module_name"]}): {", ".join(c["needed"])} not stated on the datasheet; the study '
                     f'assumed {", ".join(c["figures"]["assumed"])}.', [c])
    if not thermal:
        _finding(out, 'warning', 'no_thermal',
                 'No thermal camera in this variant: the study shows the visible cameras at night only. Add one (a '
                 'Teledyne FLIR Boson+ 640 on the front bumper, for example) to compare.')
    for cond in res.conditions:
        f = res.first_to_see(cond.id)
        th, vi = f['thermal'], f['visible']
        if th is not None and vi is not None:
            sev = 'good' if f['winner'] in ('thermal', 'tie') else 'warning'
            _finding(out, sev, 'first',
                     f'{cond.name}: {th["tag"]} (thermal) detects a pedestrian at {th["range_m"]:.0f} m '
                     f'({th["limited_by"]}), {vi["tag"]} (visible, low beam) at {vi["range_m"]:.0f} m '
                     f'({vi["limited_by"]}): ' + ('the thermal camera sees it first.' if f['winner'] == 'thermal' else
                                                  'the visible camera sees it first.' if f['winner'] == 'visible'
                                                  else 'both at the same range.'))
        dT = res.scenes[cond.id]['pedestrian_dT0_k']
        neg = [t for t, v in dT.items() if v < 0]
        if neg:
            _finding(out, 'warning', 'inversion',
                     f'{cond.name}: the pedestrian is colder than what lies behind it for {", ".join(neg)} '
                     f'(apparent ΔT {min(dT[t] for t in neg):+.2f} K): the contrast inverts on the warm road, and '
                     'a detector trained on warm people may miss them.',
                     [c for c in thermal if c['tag'] in neg])
    order = {s: i for i, s in enumerate(SEVERITIES)}
    out.sort(key=lambda f: order.get(f['severity'], 9))
    return out


def issues(project, vg=None) -> list:
    """Thermal cameras looking through the body's glass with no IR window set (critical): cheap (the materials study
    is cached on its inputs)."""
    from .rules import Issue
    v = project.variant
    th = [c for c in v.cameras if c.visible and is_thermal(project.module_for(c))]
    if not th or project.vehicle.mesh is None:
        return []
    st = settings_of(project)
    try:
        from . import materials as MAT
        res = MAT.cached_run(project, vg)
    except Exception:
        return []
    out = []
    for c in th:
        g = res.camera(c.tag)
        if g is not None and st.windows.get(c.uid, 'none') == 'none':
            out.append(Issue('critical', f'{c.tag} is a thermal camera behind {g["class_name"].lower()}: glass is '
                                         'opaque to LWIR (8–14 µm). Mount it outside the glass or behind a germanium '
                                         'or ZnSe window.', c.tag, [c.uid], where=(c.x, c.y)))
    return out


# =========================================================================== perception and weather hooks
def perception_margin(project, cam, cls_name, cond, d_cam, wpx=None, hpx=None, settings=None):
    """For core.perception under a night condition: ln of how far a thermal camera's CNR stays above snr_min for an
    object class (perception's names: 'Pedestrian', 'Cyclist', 'Car'; others are taken as a pedestrian) at distances
    d_cam, in place of core.weather's visible light test (≥ 0: resolved). cond: a core.weather Condition (its fog,
    rain and snow; the air at 10 °C, 70 % RH, a clear sky unless fog or rain make it overcast)."""
    st = settings or settings_of(project)
    k = str(cls_name).strip().lower()
    k = k if k in CLASSES else 'pedestrian'
    wet = getattr(cond, 'rain_mm_h', 0.0) > 0 or getattr(cond, 'visibility_m', 0.0) > 0
    tc = ThermalCondition('perception', getattr(cond, 'name', 'condition'), 10.0, 95.0 if wet else 70.0,
                          10.0 if wet else 12.0, 'overcast' if wet else 'clear',
                          float(getattr(cond, 'visibility_m', 0.0)), 'radiation', float(getattr(cond, 'rain_mm_h', 0.0)),
                          float(getattr(cond, 'snow_mm_h', 0.0)))
    module, lens = project.module_for(cam), project.lens_for(cam)
    fig = camera_figures(module, lens, st, st.windows.get(cam.uid, 'none'))
    ac = apparent_contrast(k, tc, st, cam.z, np.asarray(d_cam, float), tuple(fig['band_um']))
    fill = 1.0
    if wpx is not None and hpx is not None:
        fill = np.clip(np.asarray(wpx, float), 0, 1) * np.clip(np.asarray(hpx, float), 0, 1)
    cnr = np.abs(ac['dL']) * fill * fig['window_transmission'] / fig['ne_dl']
    return np.log(np.maximum(cnr, 1e-12) / st.snr_min)


def add_thermal_camera(project, module_id='flir_bosonplus_640', lens_id='bosonplus640_as32', engine=None, tag='T'):
    """Add a thermal camera on the front bumper of the active variant, looking straight ahead and level (a pedestrian
    AEB camera's pose), offset to the left of the centre line (a front camera usually sits on it). Returns it."""
    from .project import Camera, MAX_CAMERAS, new_uid
    v = project.variant
    if len(v.cameras) >= MAX_CAMERAS:
        raise ValueError(f'up to {MAX_CAMERAS} cameras per variant')
    vg = engine.vehicle_geom(project) if engine is not None else None
    if vg is not None:
        lo, hi = vg.bbox
        x, yc = float(hi[0]) + 0.01, 0.5 * float(lo[1] + hi[1])
    else:
        x, yc = 3.6, 0.0
    tags = {c.tag for c in v.cameras}
    t, i = tag, 2
    while t in tags:
        t, i = f'{tag}{i}', i + 1
    if module_id not in project.modules or lens_id not in project.lenses:
        from .lens import merge_builtins
        project.modules, project.lenses = merge_builtins(project.modules, project.lenses)
    cam = Camera(uid=new_uid(), tag=t, name='Thermal front', location='Front bumper', slot=v.free_slot(),
                 x=round(x, 3), y=round(yc + 0.25, 3), z=0.5, yaw=0.0, pitch=0.0, module_id=module_id,
                 lens_id=lens_id, port=0)
    v.cameras.append(cam)
    return cam


# =========================================================================== output
def _r(v, d=2):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return round(v, d) if math.isfinite(v) else None


def summary(res: ThermalResult) -> dict:
    """Compact figures (the assistant, the batch log): per condition the first camera to detect a pedestrian, each
    camera's pedestrian detection range, the findings."""
    out = {'conditions': {}, 'criteria': res.settings.criteria, 'beam_model': res.settings.beam_model,
           'thermal_cameras': [c['tag'] for c in res.cams if c['modality'] == 'lwir']}
    for c in res.conditions:
        f = res.first_to_see(c.id)
        out['conditions'][c.id] = {
            'label': c.label(), 'winner': f['winner'],
            'thermal': None if f['thermal'] is None else {'camera': f['thermal']['tag'],
                                                          'range_m': _r(f['thermal']['range_m'], 1),
                                                          'limited_by': f['thermal']['limited_by']},
            'visible_low_beam': None if f['visible'] is None else {'camera': f['visible']['tag'],
                                                                   'range_m': _r(f['visible']['range_m'], 1),
                                                                   'limited_by': f['visible']['limited_by']},
            'pedestrian_dT_k': res.scenes[c.id]['pedestrian_dT0_k']}
    out['pedestrian_detection_m'] = {
        f'{r["tag"]}{"/" + r["beam"] if r["beam"] else ""}': {} for r in res.rows}
    for r in res.rows:
        if r['class'] == 'pedestrian' and r['level'] == 'detection':
            out['pedestrian_detection_m'][f'{r["tag"]}{"/" + r["beam"] if r["beam"] else ""}'][r['condition']] = \
                _r(r['range_m'], 1)
    out['findings'] = [{'severity': f['severity'], 'message': f['message']} for f in res.findings
                       if f['severity'] != 'good']
    return out


def to_doc(res: ThermalResult, project=None) -> dict:
    """The JSON export: the settings, conditions and scene temperatures, every camera's figures, the range table, the
    pedestrian's CNR/SNR against range, the R112 table and illuminance, the findings."""
    st = res.settings
    return {
        'schema': SCHEMA, 'project': getattr(project, 'name', ''), 'variant': res.variant, 'frame': FRAME,
        'units': {'range': 'm', 'temperature': '°C (apparent ΔT in K)', 'radiance': 'W/(m²·sr), in band',
                  'illuminance': 'lx', 'netd': 'mK'},
        'settings': st.to_dict(), 'criteria': {'id': st.criteria, 'pixels': CRITERIA[st.criteria],
                                               'note': CRITERIA_NOTE[st.criteria]},
        'assumptions': ASSUMPTIONS, 'limits': LIMITS_TEXT, 'classes': {k: {'height_m': CLASSES[k]['height'],
                                                                           'width_m': CLASSES[k]['width']}
                                                                       for k in st.classes},
        'class_note': CLASS_NOTE,
        'conditions': [dict(c.to_doc(), scene=res.scenes.get(c.id)) for c in res.conditions],
        'cameras': [{k: v for k, v in c.items()} for c in res.cams],
        'ranges': [{**r, 'range_m': _r(r['range_m'], 2)} for r in res.rows],
        'pedestrian_curves': {f'{t}|{cid}|{b}': {k: [_r(x, 4) for x in np.asarray(v)[::4]] for k, v in d.items()}
                              for (t, cid, b), d in res.curves.items()},
        'curve_ranges_m': [_r(x, 3) for x in res.ranges[::4]],
        'r112': {'source': R112_SOURCE, 'points': res.r112, 'scale_to_minimum': res.metrics.get('r112_scale'),
                 'illuminance_on_pedestrian_lx': {f'{b}|{m}': [_r(x, 4) for x in np.asarray(v)[::4]]
                                                  for (b, m), v in res.illuminance.items()}},
        'findings': [{'severity': f['severity'], 'kind': f['kind'], 'message': f['message'], 'cameras': f['tags']}
                     for f in res.findings],
        'metrics': {k: v for k, v in res.metrics.items()},
    }


CSV_COLUMNS = ('camera', 'modality', 'condition', 'class', 'level', 'beam', 'range_m', 'limited_by')


def to_csv(res: ThermalResult) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator='\n')
    w.writerow(CSV_COLUMNS)
    for r in res.rows:
        w.writerow([r['tag'], r['modality'], r['condition'], r['class'], r['level'], r['beam'],
                    _r(r['range_m'], 2), r['limited_by']])
    return buf.getvalue()


def curves_csv(res: ThermalResult) -> str:
    """The pedestrian's CNR (thermal) or SNR (visible) against range, one column per camera, condition and beam."""
    keys = sorted(res.curves)
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator='\n')
    w.writerow(['range_m'] + [f'{t} {cid}{" " + b if b else ""} snr' for t, cid, b in keys])
    for i, d in enumerate(res.ranges):
        w.writerow([_r(d, 3)] + [_r(res.curves[k]['snr'][i], 4) for k in keys])
    return buf.getvalue()


PNG_COLORS = ['#3987e5', '#ec835a', '#9085e9', '#fab219', '#2fb5c7', '#d55181', '#6cbf5c', '#b07aa1']


def _bgr(hex_):
    return int(hex_[5:7], 16), int(hex_[3:5], 16), int(hex_[1:3], 16)


def png_ranges(res: ThermalResult, cls='pedestrian', level='detection', width=1000) -> bytes:
    """Bars: the range per camera (thermal; visible low and high beam) per condition, for one class and level."""
    import cv2
    from .pngtext import put_text
    font, aa = cv2.FONT_HERSHEY_SIMPLEX, cv2.LINE_AA
    keys = []
    for c in res.cams:
        for b in ([''] if c['modality'] == 'lwir' else ['low', 'high']):
            keys.append((c['tag'], b, c['modality']))
    conds = res.conditions
    bar_h, gap = 14, 4
    group_h = len(conds) * (bar_h + 2) + gap * 3
    head, left = 64, 150
    H = head + len(keys) * group_h + 40
    img = np.full((H, width, 3), (30, 29, 28), np.uint8)
    vals = [r['range_m'] for r in res.rows if r['class'] == cls and r['level'] == level]
    vmax = max(vals + [10.0])
    step = next(s for s in (5, 10, 20, 25, 50, 100, 200, 250, 500, 1000) if vmax / s <= 8)
    vtop = math.ceil(vmax / step) * step
    plot_w = width - left - 30
    put_text(img, f'{CLASS_LABEL.get(cls, cls)} {level} range at night ({res.variant})', (16, 24), font, 0.6,
             (236, 235, 230), 1, aa)
    x = 16
    for i, c in enumerate(conds):
        col = _bgr(PNG_COLORS[i % len(PNG_COLORS)])
        cv2.rectangle(img, (x, 38), (x + 10, 48), col, -1)
        put_text(img, c.name, (x + 15, 48), font, 0.45, (200, 199, 194), 1, aa)
        x += cv2.getTextSize(c.name, font, 0.45, 1)[0][0] + 34
    for k in range(0, int(vtop / step) + 1):
        xx = int(left + plot_w * k * step / vtop)
        cv2.line(img, (xx, head), (xx, H - 34), (62, 61, 58), 1)
        put_text(img, f'{k * step:g}', (xx - 8, H - 18), font, 0.4, (140, 140, 135), 1, aa)
    put_text(img, 'range, m', (16, H - 18), font, 0.4, (140, 140, 135), 1, aa)
    y = head
    for tag, beam, mod in keys:
        lab = f'{tag} ' + ('thermal' if mod == 'lwir' else f'visible {beam}')
        put_text(img, lab, (16, y + group_h // 2 + 4), font, 0.45, (220, 219, 214), 1, aa)
        yy = y + gap
        for i, c in enumerate(conds):
            r = res.row(tag, c.id, cls, level, beam)
            val = r['range_m'] if r else 0.0
            w = int(plot_w * val / vtop)
            col = _bgr(PNG_COLORS[i % len(PNG_COLORS)])
            if w > 0:
                cv2.rectangle(img, (left, yy), (left + w, yy + bar_h), col, -1)
            put_text(img, f'{val:.0f}', (left + w + 4, yy + bar_h - 3), font, 0.38, (190, 189, 184), 1, aa)
            yy += bar_h + 2
        y += group_h
    ok, buf = cv2.imencode('.png', img)
    return buf.tobytes()


def png_cnr(res: ThermalResult, cond=None, width=1000, height=560) -> bytes:
    """The pedestrian's contrast-to-noise ratio (thermal) and signal-to-noise ratio (visible, low beam) against range,
    log-log, for one condition (the first by default), with the snr_min line."""
    import cv2
    from .pngtext import put_text
    font, aa = cv2.FONT_HERSHEY_SIMPLEX, cv2.LINE_AA
    cond = cond or (res.conditions[0].id if res.conditions else '')
    img = np.full((height, width, 3), (30, 29, 28), np.uint8)
    L, T, R, B = 70, 50, width - 20, height - 50
    xs = np.log10(res.ranges)
    x0, x1 = float(xs[0]), float(xs[-1])
    y0, y1 = -1.0, 4.0

    def X(v):
        return int(L + (v - x0) / (x1 - x0) * (R - L))

    def Y(v):
        return int(B - (min(max(v, y0), y1) - y0) / (y1 - y0) * (B - T))
    for e in range(int(y0), int(y1) + 1):
        cv2.line(img, (L, Y(e)), (R, Y(e)), (62, 61, 58), 1)
        put_text(img, f'{10 ** e:g}', (8, Y(e) + 4), font, 0.4, (140, 140, 135), 1, aa)
    for e in range(0, int(x1) + 1):
        for m in (1, 2, 5):
            v = math.log10(m * 10 ** e)
            if x0 <= v <= x1:
                cv2.line(img, (X(v), T), (X(v), B), (52, 51, 49) if m != 1 else (70, 69, 66), 1)
                put_text(img, f'{m * 10 ** e:g}', (X(v) - 8, B + 18), font, 0.4, (140, 140, 135), 1, aa)
    yl = Y(math.log10(res.settings.snr_min))
    cv2.line(img, (L, yl), (R, yl), (59, 59, 208), 1, aa)
    name = next((c.name for c in res.conditions if c.id == cond), cond)
    put_text(img, f'Pedestrian contrast-to-noise (thermal) and SNR (visible, low beam): {name}', (16, 24), font, 0.55,
             (236, 235, 230), 1, aa)
    put_text(img, 'range, m', (R - 70, height - 10), font, 0.42, (160, 160, 155), 1, aa)
    i = 0
    for (tag, cid, beam), d in sorted(res.curves.items()):
        if cid != cond or beam == 'high':
            continue
        col = _bgr(PNG_COLORS[i % len(PNG_COLORS)])
        y = np.log10(np.maximum(np.asarray(d['snr'], float), 10 ** (y0 - 1)))
        pts = np.array([[X(a), Y(b)] for a, b in zip(xs, y)], np.int32)
        cv2.polylines(img, [pts], False, col, 2, aa)
        lab = f'{tag} {"thermal" if beam == "" else "visible"}'
        put_text(img, lab, (L + 10 + 150 * (i % 6), T + 16 + 16 * (i // 6)), font, 0.42, col, 1, aa)
        i += 1
    ok, buf = cv2.imencode('.png', img)
    return buf.tobytes()


def files(res: ThermalResult, project=None) -> list:
    """[(name, bytes)]: thermal.json, thermal_ranges.csv, thermal_curves.csv, thermal_ranges.png, thermal_cnr.png."""
    out = [('thermal.json', (json.dumps(to_doc(res, project), indent=1, default=str) + '\n').encode('utf-8')),
           ('thermal_ranges.csv', to_csv(res).encode('utf-8')),
           ('thermal_curves.csv', curves_csv(res).encode('utf-8'))]
    try:
        out += [('thermal_ranges.png', png_ranges(res)), ('thermal_cnr.png', png_cnr(res))]
    except ImportError:                     # no OpenCV: the data files only
        pass
    return out


def export(res: ThermalResult, project, out_dir) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for name, data in files(res, project):
        path = os.path.join(out_dir, name)
        with open(path, 'wb') as fh:
            fh.write(data)
        paths.append(path)
    return paths


# =========================================================================== validation
VALIDATION_MODEL = ('thermal', 'Thermal cameras (LWIR) and the night photon budget', 'vantage.core.thermal_ir',
                    'thermal')
VALIDATION_REFERENCES = {
    'widger1976': 'W. K. Widger, M. P. Woodall, "Integration of the Planck blackbody radiation function", Bull. Am. '
                  'Meteorol. Soc. 57(10), 1217–1219, 1976 (the series of the Planck integral).',
    'codata2018': 'E. Tiesinga et al., "CODATA recommended values of the fundamental physical constants: 2018", Rev. '
                  'Mod. Phys. 93, 025010, 2021 (h, c, k exact; σ derived).',
    'lynred_pico640s': 'Lynred, PICO640S datasheet (ref. 06/2020/01): recognition distances of a 1.80 m × 0.50 m '
                       'human with 19, 50 and 75 mm lenses for Johnson\'s criteria (read 2026-11-11).',
    'unr112': 'UN Regulation No. 112, Uniform provisions concerning the approval of motor vehicle headlamps emitting '
              'an asymmetrical passing beam or a driving beam or both (class B photometric test points).',
    'shettle1979': 'E. P. Shettle, R. W. Fenn, "Models for the aerosols of the lower atmosphere and the effects of '
                   'humidity variations on their optical properties", AFGL-TR-79-0214, 1979 (the modified gamma fog '
                   'drop distributions).',
    'hale1973': 'G. M. Hale, M. R. Querry, "Optical constants of water in the 200-nm to 200-µm wavelength region", '
                'Appl. Opt. 12, 555–563, 1973.',
    'bohren1983': 'C. F. Bohren, D. R. Huffman, Absorption and Scattering of Light by Small Particles, Wiley, 1983 '
                  '(Appendix A, BHMIE, and its worked example).',
}


def validation_checks() -> list:
    """Checks for vantage/validation (Spec-compatible dicts, fn(ctx) -> suite.Outcome). Model 'thermal'
    (VALIDATION_MODEL) and the references in VALIDATION_REFERENCES are for the lead to register."""
    from ..validation.suite import Outcome, figure, series

    def planck_band(ctx):
        errs = {}
        for T in (233.15, 263.15, 283.15, 303.15, 313.15, 373.15):
            a, b = float(band_radiance(T)), band_radiance_series(T)
            errs[f'{T - T0:g} °C'] = abs(a - b) / b
        # the series over the whole spectrum against σT⁴/π (Stefan–Boltzmann, σ from h, c, k), and the quadrature
        # over 3–100 µm (98 % of it at 300 K) against the series
        T = 300.0
        sb = SIGMA_SB * T ** 4 / math.pi
        errs['series 0–∞ vs σT⁴/π (300 K)'] = abs(band_radiance_series(T, (1e-3, 1e12), 5000) - sb) / sb
        q = float(band_radiance(T, (3.0, 100.0), 400))
        errs['quadrature 3–100 µm (300 K)'] = abs(q - band_radiance_series(T, (3.0, 100.0))) / q
        Ts = np.linspace(230, 380, 31)
        fig = figure('In-band (8–14 µm) radiance: quadrature against the series', 'T, K', 'W/(m²·sr)',
                     [series('Gauss–Legendre', Ts, band_radiance(Ts)),
                      series('Widger & Woodall series', Ts, [band_radiance_series(t) for t in Ts], 'points')])
        return Outcome(max(errs.values()), {'relative_error': errs}, fig)

    def johnson_pico(ctx):
        # the sheet: recognition of a 1.80 × 0.50 m human; 19 mm → 180 m, 50 mm → 455 m, 75 mm → 700 m; 17 µm pitch
        stated = {19.0: 180.0, 50.0: 455.0, 75.0: 700.0}
        dc = math.sqrt(1.8 * 0.5)
        n = CRITERIA['dri']['recognition']
        errs, mine = {}, {}
        for f, rng in stated.items():
            r = dc * (f * 1e-3) / (n * 17e-6)              # range at which √(h·w) spans n pixels
            mine[f'{f:g} mm'] = r
            errs[f'{f:g} mm'] = abs(r - rng) / rng
        # and through the module's own path: the camera's pixels on target at that range via the lens model
        from .lens import resolve
        mods = {m.id: m for m in library('modules')}
        lns = {ln.id: ln for ln in library('lenses')}
        intr = resolve(mods['lynred_pico640s'], lns['pico640s_50mm'])
        ifov = 1.0 / intr.f_px
        r_model = dc / (n * ifov)
        errs['50 mm through the lens model'] = abs(r_model - stated[50.0]) / stated[50.0]
        fig = figure('Recognition range of a 1.8 × 0.5 m human (6 px across √(h·w))', 'lens', 'range, m',
                     [series('this model', [19, 50, 75], list(mine.values()), 'points'),
                      series('Lynred datasheet', [19, 50, 75], list(stated.values()), 'points')])
        return Outcome(max(errs.values()), {'relative_error': errs, 'model_m': mine, 'stated_m': stated}, fig)

    def r112_points(ctx):
        from ..validation.checks_environment import R112_MAX, R112_MIN, R112_HIGH_MIN
        err = 0.0
        bad = []
        for name, (h, v, lo) in R112_MIN.items():
            t = R112_PASSING[name]
            if (t['h'], t['v'], t.get('min')) != (h, v, lo):
                bad.append(name)
        for name, (h, v, hi) in R112_MAX.items():
            t = R112_PASSING[name]
            if (t['h'], t['v'], t.get('max')) != (h, v, hi):
                bad.append(name)
        for name, (h, v, lo) in R112_HIGH_MIN.items():
            key = {'2.5° L': 'H-2.5L', '2.5° R': 'H-2.5R', '5° L': 'H-5L', '5° R': 'H-5R'}[name]
            t = R112_DRIVING[key]
            if (t['h'], t['v'], t['min']) != (h, v, lo):
                bad.append(key)
        # E = I/d² at the named distance, against the table figure written out again
        for row in r112_table():
            if row['lux_per_lamp'] is not None:
                ref = row['cd'] / row['range_m'] ** 2
                err = max(err, abs(row['lux_per_lamp'] - ref) / ref)
        # the R112-minimum lamp meets every minimum with the binding one exactly at it
        from . import weather as W
        wst = W.WeatherSettings()
        lows = []
        for name, t in R112_PASSING.items():
            if 'min' in t:
                lows.append(float(lamp_cd(t['h'], t['v'], 'low', 'r112_min', wst)) / t['min'])
        z = R112_ZONES['IV']
        lows.append(min(float(lamp_cd(h, v, 'low', 'r112_min', wst)) for h in np.linspace(*z['h'], 9)
                        for v in np.linspace(*z['v'], 4)) / z['min'])
        err = max(err, abs(min(lows) - 1.0))
        return Outcome(err, {'transcription_mismatch': bad, 'r112_min_lowest_ratio': min(lows)}, ok=not bad)

    def mie_bhmie(ctx):
        qe, qs, qb = mie_q(1.55, 2 * math.pi * 0.525 / 0.6328)      # x = 5.213: radius 0.525 µm at 0.6328 µm
        ref = (3.10543, 3.10543, 2.92534)
        errs = {'Q_ext': abs(qe - ref[0]) / ref[0], 'Q_sca': abs(qs - ref[1]) / ref[1],
                'Q_back': abs(qb - ref[2]) / ref[2]}
        big = mie_q(1.33 + 1e-9j, 400.0)[0]
        errs['Q_ext → 2 (x = 400)'] = abs(big - 2.0) / 2.0 - 0.02 if abs(big - 2.0) / 2.0 > 0.02 else 0.0
        return Outcome(max(errs.values()), {'relative_error': errs, 'q': [qe, qs, qb], 'reference': ref})

    def fog_visibility(ctx):
        # Shettle & Fenn's model fogs are quoted at visibilities of about 130 m (heavy advection) and 450 m (moderate
        # radiation) by the 2 % contrast definition (3.912/σ): the Mie sum with their a must give them
        stated = {'advection': 130.0, 'radiation': 450.0}
        from . import weather as W
        mine = {k: fog_visibility_m(k) * 3.912 / W.MOR_LN for k in stated}
        errs = {k: abs(mine[k] - stated[k]) / stated[k] for k in stated}
        return Outcome(max(errs.values()), {'relative_error': errs, 'visibility_2pct_m': mine, 'stated_m': stated,
                                            'lwir_to_visible_10um': {k: float(fog_ratio(k, [10.0])[0]) for k in FOGS}})

    common = {'model': 'thermal', 'cost': 'cheap'}
    return [
        dict(common, id='thermal.planck_band', title='Planck band radiance (8–14 µm)',
             quantity='relative error of the Gauss–Legendre band integral against the Widger–Woodall series at −40 to '
                      '100 °C, and of the full-spectrum integral against σT⁴/π',
             reference='the closed-form series of the Planck integral (Widger & Woodall 1976) and the '
                       'Stefan–Boltzmann law (CODATA 2018 constants)', tolerance=1e-6, unit='relative',
             refs=('widger1976', 'codata2018'), fn=planck_band),
        dict(common, id='thermal.johnson_pico640s', title='Johnson recognition range against a datasheet',
             quantity='relative error of the recognition range (6 pixels across √(1.8 × 0.5 m)) against Lynred\'s '
                      'stated 180 / 455 / 700 m with 19 / 50 / 75 mm lenses on the 17 µm PICO640S',
             reference='Lynred PICO640S datasheet, "Recognition distances for human measuring 1.80 m x 0.50 m"',
             tolerance=0.03, unit='relative', refs=('lynred_pico640s',), fn=johnson_pico),
        dict(common, id='thermal.r112_points', title='UN R112 test points and illuminance',
             quantity='largest relative error of E = I/d² at the points\' named distances, and of the R112-minimum '
                      'lamp at its binding test point; transcription of the class B table against the validation '
                      'suite\'s own copy',
             reference='UN R112 01 series §6.2.4 and §6.3.3 (class B), written out independently in '
                       'validation.checks_environment', tolerance=1e-6, unit='relative', refs=('unr112',),
             fn=r112_points),
        dict(common, id='thermal.mie_bhmie', title='Mie extinction (fog in the LWIR)',
             quantity='relative error of Q_ext, Q_sca and Q_back for m = 1.55, x = 5.213, and of Q_ext → 2 for a '
                      'large drop', reference='Bohren & Huffman (1983), BHMIE\'s worked example: Q_ext = Q_sca = '
                                              '3.10543, Q_back = 2.92534; the extinction paradox', tolerance=1e-4,
             unit='relative', refs=('bohren1983',), fn=mie_bhmie),
        dict(common, id='thermal.fog_visibility', title='Model fogs reproduce their visibility',
             quantity='relative error of the visibility (2 % contrast, 3.912/σ at 0.55 µm) that the Mie sum gives the '
                      'Shettle & Fenn advection and radiation fogs with their published number densities',
             reference='Shettle & Fenn (1979): heavy advection fog ≈ 130 m, moderate radiation fog ≈ 450 m (as quoted '
                       'with the model parameters); the visibilities are rounded, hence 5 %', tolerance=0.05,
             unit='relative', refs=('shettle1979', 'bohren1983'), fn=fog_visibility),
    ]
