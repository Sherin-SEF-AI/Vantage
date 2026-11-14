"""4D imaging radar: a MIMO virtual array's beam, the bumper cavity's ghosts and the road's two-ray multipath.

The zone model (core.sensors) gives a radar a field and a range per zone. An imaging radar measures azimuth AND
elevation by digital beamforming over a virtual array, and how well it separates targets, how strongly a strong one
masks a weak one beside it, and where it sees targets that are not there follow from that array. This module models
it per radar of the rig (SensorModel.ranging == 'mimo', whose `params` this module owns; a zone radar is compared too).

Virtual array. A MIMO radar with Tx antennas at t_i and Rx antennas at r_j measures every Tx–Rx pair: in the far
field the pair behaves as one element at t_i + r_j (the convolution of the two layouts), so 12 Tx × 16 Rx give 192
virtual channels (Richards, Fundamentals of Radar Signal Processing, 2nd ed. 2014, §9.5; TI TIDUEN5A §3.1).
Positions are in the antenna plane of the radar's own frame (y left, z up; the radar looks along +x), in half
wavelengths at the carrier. Redundant pairs (the same position twice) are used once.

Beam pattern. The array factor AF(u, v) = Σ w_n exp(jπ (y_n u + z_n v)) over the unique virtual positions (y_n, z_n
in half wavelengths) with direction cosines u = sin(az) cos(el), v = sin(el), steered to the boresight and normalised
to 0 dB there. Beamforming is row–column, as the usual processing chain does it (TI's cascade chain: an azimuth FFT
over the virtual array's longest row, the elevation from the column with the most rows): the pattern is the row's
azimuth factor times the column's elevation factor (for a filled rectangle the same as beamforming over all of it).
The taper w is applied along the row and along the column, x = (p − centre) / (span + pitch): uniform (highest
sidelobe −13.26 dB, −3 dB beamwidth 0.886 λ/(N d)); Hann (−31.5 dB, 1.44 bins: Harris, "On the use of windows
for harmonic analysis with the discrete Fourier transform", Proc. IEEE 66(1), 1978, Table 1); Taylor with a design
sidelobe level and n̄ (Doerry, "Catalog of window taper functions for sidelobe control", Sandia SAND2017-
4042, 2017, §3.6; the same formula as scipy.signal.windows.taylor). Angular resolution here is the −3 dB beamwidth of
the tapered pattern (at the boresight and steered to the field's edge, where it widens about as 1/cos az); the peak
sidelobe level is the highest lobe outside the main lobe's first nulls; grating lobes are other maxima within 3 dB
of the main lobe, checked in the visible region for every steering inside the field (the angle ambiguity).

Element pattern. Each antenna element's one-way power pattern is a Gaussian main beam, −3 (az/a3)² − 3 (el/e3)² dB
(floor −40 dB), with a3/e3 the half-angles at −3 dB where the maker states them (TI SPRUJ22C: ±30° and ±3°), else
derived (ARS 548: the elevation from its two range zones) or assumed (the stated field's edge taken as −6 dB).

Detection range per class. The radar equation: received power ∝ G_t G_r σ / R⁴, so at a fixed threshold R ∝ σ^¼
(Skolnik, Introduction to Radar Systems, 3rd ed. 2001, §2.1). A product's free-space range R0 on its reference cross-
section σ0 is its datasheet's (taken at the element's boresight), or, where the datasheet gives the chip's transmit
power, noise figure and the antenna gain instead, the link budget SNR = P_t G² λ² σ N_rx T / ((4π)³ R⁴ k T0 F L)
(coherent integration over the active frame time T and the receivers; the threshold, losses and T are settings,
stated as assumptions). At a range d the margin over the threshold is 40 log(R0 (σ/σ0)^¼ / R) + the two-way element
gain toward the target + 40 log F − the two-way loss through the fascia (core.materials, at the radar's own field).

Ground multipath (two-ray). The road reflects: the field at the target is F = |1 + Γ ρ (R_d/R_r) g e^(−jk(R_r −
R_d))| times the direct one, R_d and R_r the direct and the reflected path, Γ the road's Fresnel reflection at the
grazing angle for the radar's polarisation (core.materials.road_reflection, permittivities from Kurz et al. 2021),
ρ = exp(−2 (k σ_h sin ψ)²) the roughness loss (Ament 1953, as Beckmann & Spizzichino 1963 give it), g the element's
amplitude toward the image over that toward the target. A monostatic radar has the factor both ways (F⁴ in power), so a
point target fades where the two paths cancel and gains up to 12 dB where they add; a real target's scatterers spread
over its height (h ± the class's span, evenly; their powers add, F⁴ averaged over the height in closed form) fill the
nulls in part (set the span to 0 for the point target, the worst case). The detection range is the end of the
first stretch detected without a gap that is at least 2 m and a quarter of its end long (a sidelobe catching a
target at arm's length does not count); the gaps after its start are the fades. The image path also makes a mirror
target below the road (an elevation ghost) at the image's elevation, two reflections down; it is a separate ghost only
when the elevation offset exceeds the elevation resolution.

Bumper cavity ghosts. A radar behind a fascia (core.materials finds the class, thickness, gap and the incidence) sees
part of every echo reflected by the fascia back onto its own front (a metal plane by default, the worst case) and out
again. Two reflections (the fascia's normal n, the radar face's normal x) turn a direction d into H_n H_x d (H the
mirror about each plane): the cavity images every target, rotated by twice the fascia's tilt to the radar face, at an
extra range of about the gap × cos(incidence) per round trip, weaker by |r_fascia r_face| per bounce (the slab model:
the characteristic matrix of core.materials, Macleod 2010 ch. 2) and by the element pattern toward the image. An image
offset by less than the angular and the range resolution merges with the target (an amplitude ripple and an angle
bias, not a ghost); one offset further inside the field is a ghost target. Mounting advice follows from sweeping the
gap (the cavity's anti-resonance), the tilt between fascia and radar (ghost out of the field against the extra loss),
the fascia thickness (core.materials.best_thickness) and the radar's pitch (the first fade of the weakest class).

The library (library(): TI AWR2944EVM, the TI MMWCAS cascade in its 4- and 2-device modes, Continental ARS 548 RDI as an
array, Arbe Phoenix) holds the figures their makers state, each marked stated / derived / assumed in params['figures'];
a virtual layout that is not public is an ideal filled aperture of the stated resolution (labelled 'assumed').
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

SCHEMA = 'vantage.imaging_radar/1'
FRAME = 'vehicle (ISO 8855, rear axle): x forward, y left, z up; radar frame: x along its boresight, y left, z up'
C0 = 299_792_458.0
K_BOLTZ = 1.380649e-23
T0_K = 290.0
CHECKED = '2026-11-11'                 # when the datasheets and papers below were read
WINDOWS = ('uniform', 'hann', 'taylor')
WINDOW_LABEL = {'uniform': 'Uniform', 'hann': 'Hann', 'taylor': 'Taylor'}
POLARISATIONS = ('auto', 'vertical', 'horizontal')
ELEMENT_FLOOR_DB = -40.0
GRATING_DB = -3.0                      # a secondary maximum within this of the main lobe is a grating lobe
MERGE_RIPPLE_DB = 1.0                  # a merged image that moves the amplitude by more than this is reported
UNIFORM_K = 0.886                      # −3 dB beamwidth of a uniform aperture: 0.886 λ / (N d) radians
SEVERITIES = ('critical', 'serious', 'warning', 'good')

REFERENCES = {
    'richards2014': 'M. A. Richards, Fundamentals of Radar Signal Processing, 2nd ed., McGraw-Hill, 2014 (§9.5 the '
                    'MIMO virtual array; §7 the threshold SNR).',
    'balanis2016': 'C. A. Balanis, Antenna Theory: Analysis and Design, 4th ed., Wiley, 2016, §6.3 (the uniform linear '
                   'array: half-power beamwidth 0.886 λ / (N d) at broadside, first sidelobe −13.26 dB).',
    'harris1978': 'F. J. Harris, "On the use of windows for harmonic analysis with the discrete Fourier transform", '
                  'Proc. IEEE 66(1), 51–83, 1978, Table 1 (Hann: highest sidelobe −31.5 dB, 3 dB bandwidth 1.44 bins; '
                  'rectangle −13 dB, 0.89 bins).',
    'doerry2017': 'A. W. Doerry, "Catalog of window taper functions for sidelobe control", Sandia report '
                  'SAND2017-4042, '
                  '2017, §3.6 (the Taylor taper).',
    'ament1953': 'W. S. Ament, "Toward a theory of reflection by a rough surface", Proc. IRE 41(1), 142–146, 1953; '
                 'P. Beckmann and A. Spizzichino, The Scattering of Electromagnetic Waves from Rough Surfaces, '
                 'Pergamon, 1963 (the specular roughness factor exp(−2 (k σ_h sin ψ)²)).',
    'kurz2021': 'V. Kurz, H. Stuelzebach, F. Pfeiffer, C. van Driesten and E. Biebl, "Road surface characteristics for '
                'the automotive 77 GHz band", Adv. Radio Sci. 19, 165–172, 2021, Table 2.',
    'deep2020': 'Y. Deep, P. Held, S. S. Ram et al., "Polarimetric radar cross-sections of pedestrians at automotive '
                'radar frequencies", IET Radar, Sonar & Navigation 14(4), 2020 (arXiv:1910.13706, §IV-A: co-polar '
                'RCS of a walking adult at 77 GHz −10 to +5 dBsm).',
    'ti_tiduen5': 'Texas Instruments, TIDEP-01012 Imaging Radar Using Cascaded mmWave Sensor Reference Design, '
                  'TIDUEN5A, June 2019, revised March 2020 (Table 1; §2.3.1; §3.1, Figures 6–8; §4.2.2.1).',
    'ti_swru553': 'Texas Instruments, MMWCAS-RF-EVM user\'s guide SWRU553A (antenna offsets in wavelengths at '
                  '78.5 GHz) and the mmWave Studio 4-chip cascade MATLAB example (TX/RX position indices).',
    'ti_spruj22': 'Texas Instruments, AWR2944/AWR2944P Evaluation Module user\'s guide SPRUJ22C, November 2021, '
                  'revised September 2024, §2.5 (Figures 2-17 to 2-20).',
    'ti_swrs273': 'Texas Instruments, AWR2943/AWR2944 datasheet SWRS273E, November 2021, revised February 2026, '
                  '§6.8 RF specifications and the power-consumption table.',
    'arbe_phoenix': 'Arbe Robotics, Phoenix product page, https://arberobotics.com/product/ (read 2026-11-11); '
                    'Frost & Sullivan, Arbe award write-up, 2021 (4D point cloud over Ethernet).',
    'ti_spruis6': 'Texas Instruments, MMWCAS-DSP-EVM user\'s guide SPRUIS6 (NVMe SSD capture, 1 Gbit/s Ethernet).',
    'ars548': 'AUMOVIO Engineering Solutions (Continental), ARS 548 RDI datasheet FS_ARS-548-RDI_EN (2025), and its '
              'figures as third-party listings reproduce them.',
    'skolnik2001': 'M. I. Skolnik, Introduction to Radar Systems, 3rd ed., McGraw-Hill, 2001 (the radar equation).',
    'macleod2010': 'H. A. Macleod, Thin-Film Optical Filters, 4th ed., CRC Press, 2010, ch. 2.',
    'bornwolf1999': 'M. Born and E. Wolf, Principles of Optics, 7th ed., Cambridge University Press, 1999, §1.5–1.6.',
}

ASSUMPTIONS = (
    'Array: the virtual array is the sum of every Tx and Rx position in the far field (no mutual coupling, no '
    'calibration error, no TDM motion phase); a layout the maker does not publish is an ideal filled aperture of the '
    'stated resolution (best-case sidelobes, labelled assumed). The element pattern is a Gaussian main beam from the '
    'stated or assumed half-angles. Range: the datasheet range is a free-space figure at the element\'s boresight on '
    'its reference cross-section; a link-budget range uses the chip\'s stated power and noise figure with the '
    'threshold, losses and integration time of the settings. Cross-sections are typical values (cited), not your '
    'targets\'. Ground: a flat road at z = 0, a smooth dielectric half-space with the cited permittivity and an rms '
    'roughness; the target a point scatterer at the class\'s height. Cavity: a planar fascia and a flat radar front '
    'whose reflection is a setting (0 dB: metal, the worst case); one image per bounce, the Tx- and Rx-side images '
    'taken as one (together they may be up to 6 dB stronger).')
LIMITS_TEXT = (
    'Not a full-wave or ray-traced simulation: no diffraction at the fascia\'s edges or curvature, no multipath off '
    'other vehicles, guard rails or the own body beyond the fascia, no Doppler or range sidelobes, no interference '
    'between radars, no weather (core.weather has rain and fog). The two-ray model holds for a flat road; '
    'speed bumps, crests and a wet film change Γ. CARLA\'s radar sensor is a ray cast: it models none of the above '
    '(see the report on the CARLA check).')

# Typical radar cross-sections at 77 GHz and the height of the scattering centre (assumed): the car's is the
# reference datasheets quote ranges on (TI TIDUEN5A Table 1: "an RCS of approximately 10 m²"); the pedestrian's lies in
# the −10 to +5 dBsm Deep et al. measured (TIDUEN5A's "human RCS objects at 150 m" against 350 m on 10 m² in the same
# mode implies 10 (150/350)⁴ = 0.34 m², close); no peer-reviewed mean for a cyclist was found: 1 m² is assumed.
DEFAULT_CLASSES = [
    {'name': 'Car', 'rcs_m2': 10.0, 'height_m': 0.6, 'span_m': 0.4,
     'source': 'TI TIDUEN5A Table 1 (car ≈ 10 m², the datasheets\' reference); heights assumed'},
    {'name': 'Pedestrian', 'rcs_m2': 0.5, 'height_m': 1.0, 'span_m': 0.6,
     'source': 'Deep et al. 2020 §IV-A: −10 to +5 dBsm at 77 GHz (−3 dBsm used); heights assumed (legs to head)'},
    {'name': 'Cyclist', 'rcs_m2': 1.0, 'height_m': 0.9, 'span_m': 0.5,
     'source': 'assumed: no peer-reviewed mean found; between the pedestrian and the car; heights assumed'},
]


# =========================================================================== the library
_TI_CAS = 'https://www.ti.com/lit/pdf/tiduen5'
_TI_2944 = 'https://www.ti.com/lit/pdf/spruj22'
_TI_2944_DS = 'https://www.ti.com/lit/ds/symlink/awr2944.pdf'
_ARBE = 'https://arberobotics.com/product/'
_ARS548 = 'https://engineering-solutions.aumovio.com/wp-content/uploads/2025/10/FS_ARS-548-RDI_EN.pdf'

# TI 4-device cascade (MMWCAS-RF-EVM): TX1..TX12 and RX1..RX16 as (azimuth, elevation) in half wavelengths at 78.5
# GHz, the indices of TI's cascade example; they make the guide's 86-element azimuth array (§3.1, Fig. 7), its
# minimum-redundancy elevation (rows 0, 1, 4, 6: "equivalent to 7 antennas") and, for TX [1 2 3 10 11 12] with RX
# [4 3 2 1 16 15 14 13], the 2-device mode's 23 azimuth elements (Fig. 8) — the tests check all three.
CAS_TX = ((11, 6), (10, 4), (9, 1), (32, 0), (28, 0), (24, 0), (20, 0), (16, 0), (12, 0), (8, 0), (4, 0), (0, 0))
CAS_RX = tuple((y, 0) for y in (11, 12, 13, 14, 50, 51, 52, 53, 46, 47, 48, 49, 0, 1, 2, 3))
CAS2_TX_IDS = (1, 2, 3, 10, 11, 12)
CAS2_RX_IDS = (4, 3, 2, 1, 16, 15, 14, 13)
# TI AWR2944EVM (SPRUJ22C Fig. 2-17/2-18): four Rx λ/2 apart; TX1, TX3, TX4 at 0, 2λ, 4λ; TX2 λ beside TX1 and 0.8λ up
EVM2944_TX = ((0, 0), (2, 1.6), (4, 0), (8, 0))
EVM2944_RX = ((0, 0), (1, 0), (2, 0), (3, 0))

# a library model of the zone kind with a stated array: the analysis uses the array model's params for it
ZONE_TO_ARRAY = {'conti_ars548_rdi': 'conti_ars548_rdi_mimo'}

# link-budget defaults the library's ranges are computed with (the study's settings can change them)
LINK_DEFAULTS = {'snr_db': 15.0, 'loss_db': 6.0}


def wavelength_m(f_ghz: float) -> float:
    return C0 / (float(f_ghz) * 1e9)


def link_range(params: dict, rcs_m2: float, snr_db: float, loss_db: float, integration_ms: float | None = None):
    """Free-space detection range (m) from the radar equation with coherent integration over the active frame time
    and every receiver: SNR = P_t G² λ² σ N_rx T / ((4π)³ R⁴ k T0 F L). None when a figure is missing."""
    try:
        pt = 10 ** ((float(params['tx_power_dbm']) - 30) / 10)
        g = 10 ** (float(params['element_gain_dbi']) / 10)
        nf = 10 ** (float(params['noise_figure_db']) / 10)
        n_rx = int(params['n_rx'])
        t = float(integration_ms if integration_ms is not None else params['integration_ms']) / 1000.0
        lam = wavelength_m(float(params['carrier_ghz']))
    except (KeyError, TypeError, ValueError):
        return None
    snr = 10 ** (snr_db / 10)
    L = 10 ** (loss_db / 10)
    r4 = pt * g * g * lam ** 2 * float(rcs_m2) * n_rx * t / ((4 * math.pi) ** 3 * K_BOLTZ * T0_K * nf * L * snr)
    return float(r4 ** 0.25) if r4 > 0 else None


def _model(mid, name, vendor, zones, params, note, url, **kw):
    if not kw.get('data_mbps'):
        note += ' The output data rate is not stated (it depends on the chirp set and the point-cloud mode).'
    note = f'{note} (Added after the sensor library\'s {S.CHECKED} check.)'
    return S.SensorModel(mid, name, 'radar', vendor, 'fixed', zones=[list(map(float, z)) for z in zones],
                         note=note, url=url, builtin=True, ranging='mimo', params=params, **kw)


def library() -> list:
    """The imaging radars with public figures, as SensorModels (kind 'radar', ranging 'mimo'). Each model's zones are
    its stated field and range on a car, so every other study (coverage, overlays, recording) uses it as a radar."""
    out = []
    # ---------------------------------------------------------------- TI AWR2944EVM
    p = {'f_lo_ghz': 76.0, 'f_hi_ghz': 81.0, 'carrier_ghz': 78.5, 'bandwidth_ghz': 4.5, 'n_tx': 4, 'n_rx': 4,
         'tx': [list(t) for t in EVM2944_TX], 'rx': [list(r) for r in EVM2944_RX], 'layout': 'public',
         'element_az_3db_deg': 30.0, 'element_el_3db_deg': 3.0, 'element_gain_dbi': 13.0,
         'tx_power_dbm': 13.5, 'noise_figure_db': 12.0, 'integration_ms': 12.8, 'rcs_ref_m2': 10.0,
         'range_from': 'link',
         'figures': {'band': 'stated (SWRS273E §6.8: 76–81 GHz)',
                     'bandwidth_ghz': 'stated maximum (SWRS273E: VCO2 continuous range 4.5 GHz); set your chirp\'s',
                     'layout': 'stated (SPRUJ22C Fig. 2-17/2-18)',
                     'element': 'stated (SPRUJ22C §2.5: 13 dBi; ±30° and ±3° at −3 dB, ±45° and ±5° at −6 dB)',
                     'tx_power_dbm': 'stated (SWRS273E §6.8)', 'noise_figure_db': 'stated (SWRS273E §6.8)',
                     'integration_ms': 'derived: 256 chirps × 50 µs, the datasheet\'s power-measurement chirp set',
                     'range': 'derived: link budget (15 dB threshold, 6 dB losses assumed)',
                     'carrier_ghz': 'assumed: band centre'}}
    r0 = link_range(p, 10.0, LINK_DEFAULTS['snr_db'], LINK_DEFAULTS['loss_db'])
    p['range_ref_m'] = round(r0, 1)
    out.append(_model(
        'ti_awr2944_evm', 'AWR2944EVM', 'Texas Instruments', [[r0, 45.0, 5.0]], p,
        f'Single-chip 4D radar evaluation module: 4 Tx × 4 Rx = 16 virtual channels (a row of 12 and an offset row of '
        f'4 at 0.8 λ for elevation), 76–81 GHz, up to 4.5 GHz sweep. Patch arrays 13 dBi, ±30° × ±3° at −3 dB, '
        f'±45° × ±5° at −6 dB (the field here). 13.5 dBm per Tx, 12 dB noise figure. The range on a car '
        f'({r0:.0f} m) is computed from the link budget with a 12.8 ms integration, 15 dB threshold and 6 dB losses '
        f'(assumed): no range is stated. Board size and power: not stated (a 20 × 100 × 80 mm placeholder; the chip '
        f'draws 1.44–1.98 W). Frame rate as the datasheet\'s power-measurement set (25.6 ms: 39 Hz). Read {CHECKED}: '
        f'SPRUJ22C §2.5, SWRS273E §6.8.',
        _TI_2944, range_min_m=0.3, rate_hz=39.0, interface='100BASE-T1 (MATEnet) or RJ45', time_sync='none',
        size_mm=(20.0, 100.0, 80.0), optical_z_mm=40.0, power_w=0.0))
    # ---------------------------------------------------------------- TI MMWCAS cascade, 4 devices
    p = {'f_lo_ghz': 76.0, 'f_hi_ghz': 81.0, 'carrier_ghz': 78.5, 'bandwidth_ghz': round(C0 / (2 * 0.6) / 1e9, 4),
         'n_tx': 12, 'n_rx': 16, 'tx': [list(t) for t in CAS_TX], 'rx': [list(r) for r in CAS_RX],
         'layout': 'public', 'range_res_m': 0.6, 'az_res_deg': 1.4, 'el_res_deg': 18.0, 'v_max_mps': 133 / 3.6,
         'v_res_mps': 0.53 / 3.6, 'range_ref_m': 150.0, 'rcs_ref_m2': 10.0, 'range_from': 'datasheet',
         'element_az_3db_deg': round(70.0 / math.sqrt(2), 2), 'element_el_3db_deg': 12.7,
         'figures': {'band': 'stated (TIDUEN5A §2.2)', 'layout': 'stated (TIDUEN5A §3.1 Fig. 6–7; SWRU553A)',
                     'range_res_m': 'stated example (TIDUEN5A Table 1, MIMO)',
                     'bandwidth_ghz': 'derived from the stated range resolution',
                     'az_res_deg': 'stated (Table 1)', 'el_res_deg': 'stated (Table 1)',
                     'range': 'stated: 150 m on about 10 m² in MIMO (Table 1; 350 m in Tx beamforming)',
                     'velocity': 'stated (Table 1: 133 km/h, 0.53 km/h)',
                     'element': 'assumed: ±70° stated field as the −6 dB edge; elevation of a 4-element series-fed '
                                'column (§2.3.1) at λ/2: ±12.7°'}}
    out.append(_model(
        'ti_mmwcas_4chip', 'MMWCAS-RF-EVM cascade (4 devices, MIMO)', 'Texas Instruments', [[150.0, 70.0, 12.7]], p,
        'Four cascaded AWR2243 (76–81 GHz): 12 Tx × 16 Rx = 192 virtual channels, 86 unique in azimuth at λ/2 '
        '(1.4° stated) and a minimum-redundancy elevation array of four rows (18° stated). MIMO mode: 150 m on '
        'about 10 m², 60 cm range resolution (an example chirp), ±133 km/h; ±70° field. Board 5 V, 8 A maximum. '
        'Elevation field and the board size are not stated (±12.7° and a 20 × 160 × 120 mm placeholder). Read '
        f'{CHECKED}: TIDUEN5A Table 1, §2.3.1, §3.1. Raw ADC data leave each device over a 4-lane CSI-2 (2.4 Gbit/s) '
        f'to the MMWCAS-DSP-EVM, which records to its NVMe SSD and offloads over 1 Gbit/s Ethernet (SPRUIS6); the '
        f'interface here is that Ethernet.', _TI_CAS, range_min_m=0.5, rate_hz=10.0,
        interface='1000BASE-T (MMWCAS-DSP-EVM)', time_sync='none', size_mm=(20.0, 160.0, 120.0), optical_z_mm=60.0,
        power_w=0.0))
    # ---------------------------------------------------------------- TI MMWCAS cascade, 2 devices
    tx2 = [list(CAS_TX[i - 1]) for i in CAS2_TX_IDS]
    rx2 = [list(CAS_RX[i - 1]) for i in CAS2_RX_IDS]
    r2 = 150.0 * (8 / 16) ** 0.25
    p = {'f_lo_ghz': 76.0, 'f_hi_ghz': 81.0, 'carrier_ghz': 78.5, 'bandwidth_ghz': round(C0 / (2 * 0.6) / 1e9, 4),
         'n_tx': 6, 'n_rx': 8, 'tx': tx2, 'rx': rx2, 'layout': 'public', 'range_res_m': 0.6,
         'range_ref_m': round(r2, 1), 'rcs_ref_m2': 10.0, 'range_from': 'datasheet',
         'element_az_3db_deg': round(70.0 / math.sqrt(2), 2), 'element_el_3db_deg': 12.7,
         'figures': {'layout': 'stated (TIDUEN5A §3.1 Fig. 8: TX [1 2 3 10 11 12], RX [4 3 2 1 16 15 14 13])',
                     'range': 'derived: the 4-device range × (8/16)^¼ (half the receivers, the same frame time)',
                     'element': 'assumed as the 4-device mode'}}
    out.append(_model(
        'ti_mmwcas_2chip', 'MMWCAS-RF-EVM cascade (2 devices, MIMO)', 'Texas Instruments', [[r2, 70.0, 12.7]], p,
        'The cascade board in its 2-device mode: 6 Tx × 8 Rx = 48 channels, 23 unique in azimuth, the 4-device '
        f'elevation rows. Range derived from the 4-device figure ({r2:.0f} m). Read {CHECKED}: TIDUEN5A §3.1.',
        _TI_CAS, range_min_m=0.5, rate_hz=10.0, interface='1000BASE-T (MMWCAS-DSP-EVM)', time_sync='none',
        size_mm=(20.0, 160.0, 120.0), optical_z_mm=60.0, power_w=0.0))
    # ---------------------------------------------------------------- Continental ARS 548 RDI as an array
    el3 = 14.0 * math.sqrt(3.0 / (-20 * math.log10(100.0 / 301.0)))
    p = {'f_lo_ghz': 76.0, 'f_hi_ghz': 77.0, 'carrier_ghz': 76.5, 'bandwidth_ghz': round(C0 / (2 * 0.22) / 1e9, 4),
         'range_res_m': 0.22, 'az_res_deg': 1.2, 'az_res_edge_deg': 1.68, 'az_edge_deg': 45.0, 'el_res_deg': 2.3,
         'n_virtual': 192, 'layout': 'assumed', 'range_ref_m': 301.0, 'rcs_ref_m2': 10.0, 'range_from': 'datasheet',
         'element_az_3db_deg': round(60.0 / math.sqrt(2), 2), 'element_el_3db_deg': round(el3, 2),
         'figures': {'band': 'stated (77 GHz long-range band)', 'range_res_m': 'stated (0.22 m)',
                     'az_res_deg': 'stated beam width (3 dB) 1.2° at 0…±15°, 1.68° at ±45°; elevation 2.3° — as '
                                   'third-party listings reproduce the datasheet (its PDF could not be read here)',
                     'n_virtual': '192 virtual antennas: a third-party listing, not the datasheet',
                     'layout': 'assumed: an ideal filled aperture of the stated beam widths (not public)',
                     'range': 'stated: 301 m (taken on the 10 m² car, as the zone model)',
                     'element': 'elevation derived from the two zones (±4° to 301 m, ±14° to 100 m: R ∝ gain^¼); '
                                'azimuth assumed: the ±60° internal field as the −6 dB edge'}}
    out.append(_model(
        'conti_ars548_rdi_mimo', 'ARS 548 RDI (array)', 'Continental', [[301.0, 50.0, 4.0], [100.0, 50.0, 14.0]], p,
        'The ARS 548 RDI\'s zone model with its stated beam: 3 dB beam width 1.2° in azimuth at the boresight (1.68° '
        'at ±45°), 2.3° in elevation, 0.22 m range resolution, 76–77 GHz. Its virtual layout is not public (192 '
        'virtual channels by third-party listings): the array here is an ideal filled aperture of those beam widths '
        '(assumed), so its sidelobes are a best case. Elevation element pattern derived from the two zones. '
        f'Read {CHECKED}.', _ARS548, range_min_m=0.2, rate_hz=20.0, interface='100BASE-T1', time_sync='none',
        size_mm=(39.0, 137.0, 90.0), optical_z_mm=45.0, power_w=18.0))
    # ---------------------------------------------------------------- Arbe Phoenix
    p = {'f_lo_ghz': 76.0, 'f_hi_ghz': 81.0, 'carrier_ghz': 78.5, 'bandwidth_ghz': round(C0 / (2 * 0.2) / 1e9, 4),
         'range_res_m': 0.2, 'n_tx': 48, 'n_rx': 48, 'n_virtual': 2304, 'az_res_deg': 0.7, 'el_res_deg': 0.9,
         'layout': 'assumed', 'range_ref_m': 300.0, 'rcs_ref_m2': 10.0, 'range_from': 'datasheet',
         'element_az_3db_deg': round(60.0 / math.sqrt(2), 2), 'element_el_3db_deg': round(15.0 / math.sqrt(2), 2),
         'figures': {'band': 'stated (76–81 GHz)', 'range_res_m': 'stated best (20 cm – 80 cm)',
                     'bandwidth_ghz': 'derived from the best range resolution',
                     'channels': 'stated (2,304 = 48 Tx × 48 Rx)', 'resolution': 'stated (0.7° az, 0.9° el)',
                     'layout': 'assumed: an ideal filled aperture of the stated resolution (not public; a filled '
                               'λ/2 aperture of that size needs about 16 000 elements, so the real array is sparse '
                               'and its sidelobes are not known)',
                     'range': 'stated 300 m; the cross-section is not stated (taken as the 10 m² car)',
                     'element': 'assumed: the stated 120° × 30° field\'s edge as −6 dB'}}
    out.append(_model(
        'arbe_phoenix', 'Phoenix', 'Arbe', [[300.0, 60.0, 15.0]], p,
        'Imaging radar on Arbe\'s chipset: 48 Tx × 48 Rx = 2,304 virtual channels, 0.7° azimuth and 0.9° elevation '
        'resolution, 120° × 30° field, 300 m, 20–80 cm range resolution, 20 frames/s, 76–81 GHz (product page). '
        'The layout is not public: an ideal filled aperture of the stated resolution is assumed. A 4D point cloud '
        'over Ethernet (Frost & Sullivan 2021 award write-up; the link speed is not stated). Size and power not '
        'stated (a 40 × 150 × 100 mm placeholder). Read ' + CHECKED + '.', _ARBE,
        range_min_m=0.5, rate_hz=20.0, interface='Ethernet (speed not stated)', time_sync='none',
        size_mm=(40.0, 150.0, 100.0), optical_z_mm=50.0, power_w=0.0))
    return out


PRODUCT_IDS = ('ti_awr2944_evm', 'ti_mmwcas_4chip', 'ti_mmwcas_2chip', 'conti_ars548_rdi_mimo', 'arbe_phoenix')


# =========================================================================== the virtual array
@dataclass
class ArraySpec:
    """A radar's antennas: Tx and Rx positions (n, 2) as (azimuth y, elevation z) in half wavelengths at f_ghz."""
    tx: np.ndarray
    rx: np.ndarray
    f_ghz: float
    layout: str                 # 'public' | 'assumed'
    n_tx: int
    n_rx: int
    n_virtual_stated: int = 0
    note: str = ''
    _virtual: np.ndarray | None = None

    @property
    def virtual(self) -> np.ndarray:
        """Every Tx–Rx pair's virtual element (n_tx × n_rx, 2), half wavelengths."""
        if self._virtual is not None:
            return self._virtual
        return virtual_array(self.tx, self.rx)

    def unique(self):
        """(unique virtual positions (N, 2), how many pairs share each)."""
        return unique_positions(self.virtual)


def virtual_array(tx, rx) -> np.ndarray:
    """The MIMO virtual array: every t_i + r_j (the convolution of the two layouts), (n_tx · n_rx, 2)."""
    tx = np.asarray(tx, float).reshape(-1, 2)
    rx = np.asarray(rx, float).reshape(-1, 2)
    return (tx[:, None, :] + rx[None, :, :]).reshape(-1, 2)


def unique_positions(V, tol=1e-6):
    V = np.asarray(V, float).reshape(-1, 2)
    if not len(V):
        return np.zeros((0, 2)), np.zeros(0, int)
    key = np.round(V / tol).astype(np.int64)
    _, idx, cnt = np.unique(key, axis=0, return_index=True, return_counts=True)
    return V[idx], cnt


def elements_for(res_deg: float) -> int:
    """Elements at λ/2 a uniform aperture needs for a −3 dB beamwidth of res_deg at its boresight."""
    if not (res_deg and res_deg > 0):
        return 1
    return max(1, int(math.ceil(UNIFORM_K * 2 / math.radians(res_deg) - 1e-9)))


def array_spec(model) -> ArraySpec | None:
    """The model's array from its params: the stated Tx/Rx layout, else an ideal filled aperture of the stated
    azimuth (and elevation) resolution (layout 'assumed'); None when neither is given."""
    p = model.params if isinstance(getattr(model, 'params', None), dict) else {}
    f = _num(p.get('carrier_ghz'), 77.0)
    tx, rx = p.get('tx'), p.get('rx')
    try:
        if tx and rx:
            T = np.asarray(tx, float).reshape(-1, 2)
            R = np.asarray(rx, float).reshape(-1, 2)
            if np.all(np.isfinite(T)) and np.all(np.isfinite(R)):
                return ArraySpec(T, R, f, 'public', len(T), len(R), int(_num(p.get('n_virtual'), 0)),
                                 'the maker\'s layout')
    except (TypeError, ValueError):
        pass
    az = _num(p.get('az_res_deg'), 0.0)
    if az <= 0:
        return None
    el = _num(p.get('el_res_deg'), 0.0)
    na, ne = elements_for(az), elements_for(el) if el > 0 else 1
    ys, zs = np.meshgrid(np.arange(na, dtype=float), np.arange(ne, dtype=float), indexing='ij')
    V = np.c_[ys.ravel(), zs.ravel()]
    nt, nr = int(_num(p.get('n_tx'), 0)), int(_num(p.get('n_rx'), 0))
    return ArraySpec(np.zeros((1, 2)), V, f, 'assumed', nt, nr, int(_num(p.get('n_virtual'), 0)),
                     f'an ideal filled aperture of {na} × {ne} elements at λ/2 for the stated {az:g}°'
                     + (f' × {el:g}°' if el > 0 else '') + ' (the maker\'s layout is not public)', V)


def _num(v, default):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    return x if math.isfinite(x) else default


# =========================================================================== tapers
def taylor_coefficients(sll_db: float, nbar: int) -> np.ndarray:
    """F_m, m = 1 … n̄−1, of the Taylor taper w(x) = 1 + 2 Σ F_m cos(2π m x) (Doerry 2017 §3.6)."""
    nbar = max(int(nbar), 1)
    A = math.acosh(10 ** (abs(sll_db) / 20)) / math.pi
    s2 = nbar ** 2 / (A ** 2 + (nbar - 0.5) ** 2)
    ms = np.arange(1, nbar)
    F = np.zeros(len(ms))
    for k, m in enumerate(ms):
        num = np.prod([1 - m ** 2 / s2 / (A ** 2 + (i - 0.5) ** 2) for i in range(1, nbar)])
        den = np.prod([1 - m ** 2 / i ** 2 for i in range(1, nbar) if i != m])
        F[k] = ((-1) ** (m + 1)) / 2 * num / den
    return F


def taper(x, kind: str, sll_db: float = 30.0, nbar: int = 4) -> np.ndarray:
    """A taper at normalised aperture positions x (−½ … ½), peak 1: 'uniform', 'hann', 'taylor'."""
    x = np.asarray(x, float)
    if kind == 'hann':
        return 0.5 * (1 + np.cos(2 * math.pi * x))
    if kind == 'taylor':
        F = taylor_coefficients(sll_db, nbar)
        w = np.ones_like(x)
        for m, f in enumerate(F, 1):
            w = w + 2 * f * np.cos(2 * math.pi * m * x)
        return w / max(float(np.max(np.abs(w))), 1e-12) if w.size else w
    return np.ones_like(x)


def _norm_pos(p):
    """Positions along one axis mapped to (p − centre) / (span + pitch), pitch the smallest spacing (an even grid of N
    maps as the discrete window of N points)."""
    p = np.asarray(p, float)
    u = np.unique(np.round(p, 6))
    if len(u) < 2:
        return np.zeros_like(p)
    pitch = float(np.min(np.diff(u)))
    span = float(u[-1] - u[0])
    return (p - (u[-1] + u[0]) / 2) / (span + pitch)


def weights(U, kind: str, sll_db: float = 30.0, nbar: int = 4) -> np.ndarray:
    """A separable taper over the unique virtual positions U (N, 2): azimuth taper × elevation taper."""
    U = np.asarray(U, float).reshape(-1, 2)
    return taper(_norm_pos(U[:, 0]), kind, sll_db, nbar) * taper(_norm_pos(U[:, 1]), kind, sll_db, nbar)


# =========================================================================== the array factor
def _grid(U, w):
    """(azimuth positions (Ny,), elevation positions (Nz,), weights (Ny, Nz)) of the virtual array on its own grid."""
    ys, iy = np.unique(np.round(U[:, 0], 6), return_inverse=True)
    zs, iz = np.unique(np.round(U[:, 1], 6), return_inverse=True)
    W = np.zeros((len(ys), len(zs)), complex)
    np.add.at(W, (iy, iz), w)
    return ys, zs, W


def af_cut(U, w, u=None, v=None):
    """Normalised complex array factor along a cut: u (direction cosines) at v = 0, or v at u = 0."""
    ys, zs, W = _grid(U, w)
    tot = W.sum()
    if u is not None:
        a = W.sum(axis=1)
        return np.exp(1j * math.pi * np.outer(np.asarray(u, float), ys)) @ a / tot
    b = W.sum(axis=0)
    return np.exp(1j * math.pi * np.outer(np.asarray(v, float), zs)) @ b / tot


def af_uv(U, w, u, v):
    """Normalised complex array factor on the product grid u (Nu,) × v (Nv,): (Nv, Nu)."""
    ys, zs, W = _grid(U, w)
    Eu = np.exp(1j * math.pi * np.outer(ys, np.asarray(u, float)))        # (Ny, Nu)
    Ev = np.exp(1j * math.pi * np.outer(zs, np.asarray(v, float)))        # (Nz, Nv)
    return (Ev.T @ W.T @ Eu) / W.sum()


def af_azel(U, w, az_deg, el_deg, chunk=2_000_000):
    """Normalised complex array factor over azimuth (Na,) × elevation (Ne,) in degrees: (Ne, Na)."""
    ys, zs, W = _grid(U, w)
    az, el = np.radians(np.asarray(az_deg, float)), np.radians(np.asarray(el_deg, float))
    t = W @ np.exp(1j * math.pi * np.outer(zs, np.sin(el)))               # (Ny, Ne)
    out = np.empty((len(el), len(az)), complex)
    step = max(1, chunk // max(len(az) * len(ys), 1))
    for j0 in range(0, len(el), step):
        j1 = min(j0 + step, len(el))
        U_ = np.sin(az)[None, :] * np.cos(el[j0:j1])[:, None]            # (k, Na)
        E = np.exp(1j * math.pi * U_[..., None] * ys)                     # (k, Na, Ny)
        out[j0:j1] = np.einsum('kay,yk->ka', E, t[:, j0:j1])
    return out / W.sum()


def db_pow(a, floor=-120.0):
    return np.maximum(20 * np.log10(np.maximum(np.abs(a), 1e-12)), floor)


def lobe(theta, p_db, centre=0.0) -> dict:
    """The main lobe of a cut (angles theta, power p_db) nearest centre: its −3 dB width, first nulls and the highest
    sidelobe outside them. Widths nan when the pattern never falls 3 dB within the cut."""
    th = np.asarray(theta, float)
    p = np.asarray(p_db, float)
    i = int(np.argmin(np.abs(th - centre)))
    while 0 < i < len(p) - 1 and (p[i - 1] > p[i] or p[i + 1] > p[i]):
        i = i - 1 if p[i - 1] > p[i + 1] else i + 1
    pk = p[i]

    def cross(step):
        j = i
        while 0 <= j + step < len(p) and p[j + step] > pk - 3.0:
            j += step
        k = j + step
        if not 0 <= k < len(p):
            return float('nan')
        a, b = p[j] - (pk - 3.0), p[k] - (pk - 3.0)
        return float(th[j] + (th[k] - th[j]) * a / (a - b)) if a != b else float(th[j])

    def null(step):
        j = i
        while 0 <= j + step < len(p) and p[j + step] <= p[j]:
            j += step
        return j
    lo, hi = cross(-1), cross(+1)
    nl, nr = null(-1), null(+1)
    side = np.r_[p[:nl], p[nr + 1:]]
    psl = float(side.max() - pk) if side.size else float('-inf')
    return {'peak_deg': float(th[i]), 'hpbw_deg': hi - lo, 'lo_deg': lo, 'hi_deg': hi,
            'null_lo_deg': float(th[nl]), 'null_hi_deg': float(th[nr]), 'psl_db': psl}


@dataclass
class Beamformer:
    """Row–column beamforming as the usual processing chain does it (TI's cascade chain: an azimuth FFT over the
    virtual array's longest row, the elevation from the column with the most rows at the azimuth peak): the row's
    azimuth positions y with their taper, the column's elevation positions z with theirs, half wavelengths. A filled
    2D aperture gives the same pattern as beamforming over all of it."""
    y: np.ndarray
    wy: np.ndarray
    z: np.ndarray
    wz: np.ndarray
    row_z: float = 0.0          # the elevation of the row used
    col_y: float = 0.0          # the azimuth position of the column used

    def az(self, u):
        """Normalised complex azimuth factor at direction cosines u."""
        return np.exp(1j * math.pi * np.outer(np.asarray(u, float), self.y)) @ self.wy / self.wy.sum()

    def el(self, v):
        return np.exp(1j * math.pi * np.outer(np.asarray(v, float), self.z)) @ self.wz / self.wz.sum()

    def uv(self, u, v):
        """(Nv, Nu) normalised complex pattern on the product grid."""
        return np.outer(self.el(v), self.az(u))

    def azel(self, az_deg, el_deg):
        """(Ne, Na) normalised complex pattern over azimuth × elevation in degrees (u = sin az cos el, v = sin el)."""
        az, el = np.radians(np.asarray(az_deg, float)), np.radians(np.asarray(el_deg, float))
        ev = self.el(np.sin(el))
        u = np.sin(az)[None, :] * np.cos(el)[:, None]
        ys = self.y
        out = np.empty(u.shape, complex)
        step = max(1, 2_000_000 // max(u.shape[1] * len(ys), 1))
        for j0 in range(0, u.shape[0], step):
            E = np.exp(1j * math.pi * u[j0:j0 + step, :, None] * ys)
            out[j0:j0 + step] = (E @ self.wy) / self.wy.sum()
        return out * ev[:, None]


def beamformer(U, kind='uniform', sll_db=30.0, nbar=4) -> Beamformer:
    """The row and the column of the unique virtual positions U (N, 2) the beamformer uses, tapered."""
    U = np.asarray(U, float).reshape(-1, 2)
    zr = np.round(U[:, 1], 6)
    yr = np.round(U[:, 0], 6)
    zs, zc = np.unique(zr, return_counts=True)
    row_z = float(zs[np.argmax(zc)])                                     # the longest row (the first such)
    y = np.sort(U[zr == row_z, 0])
    ys, yc = np.unique(yr, return_counts=True)
    best = ys[yc == yc.max()]
    mid = 0.5 * (y.min() + y.max()) if len(y) else 0.0
    col_y = float(best[np.argmin(np.abs(best - mid))])                  # the fullest column nearest the row's middle
    z = np.sort(U[yr == col_y, 1])
    return Beamformer(y, taper(_norm_pos(y), kind, sll_db, nbar), z, taper(_norm_pos(z), kind, sll_db, nbar),
                      row_z, col_y)


def steered_hpbw(bf: Beamformer, az0_deg: float, step=None) -> float:
    """The −3 dB beamwidth (degrees, in azimuth) of the beam steered to az0 at zero elevation."""
    span = max(float(np.ptp(bf.y)) if len(bf.y) else 0.0, 1.0)
    est = math.degrees(UNIFORM_K * 2 / span) / max(math.cos(math.radians(az0_deg)), 0.05)
    half = min(max(4 * est, 2.0), 89.0)
    step = step or max(est / 60, 0.002)
    th = np.arange(max(az0_deg - half, -90.0), min(az0_deg + half, 90.0) + step / 2, step)
    a = bf.az(np.sin(np.radians(th)) - math.sin(math.radians(az0_deg)))
    return lobe(th, db_pow(a), az0_deg)['hpbw_deg']


def grating_lobes(bf: Beamformer, fov_az: float, fov_el: float, n=401) -> list:
    """Maxima of the pattern within GRATING_DB of the main lobe, away from it, that enter the visible region for some
    steering in the field (±fov_az × ±fov_el): [{'du', 'dv', 'level_db', 'az_deg', 'el_deg' (where it shows when the
    beam points at the boresight, None when invisible there), 'visible_when'}]."""
    g = np.linspace(-2.0, 2.0, n)
    A = db_pow(bf.uv(g, g))
    out = []
    bu = 2.0 / max(float(np.ptp(bf.y)) + 1.0, 1.0) if len(bf.y) > 1 else 2.0
    bv = 2.0 / max(float(np.ptp(bf.z)) + 1.0, 1.0) if len(bf.z) > 1 else 2.0
    core = (np.abs(g)[None, :] <= bu) & (np.abs(g)[:, None] <= bv)        # the main lobe's own neighbourhood
    loc = np.ones_like(A, bool)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di or dj:
                loc &= A >= np.roll(np.roll(A, di, 0), dj, 1)
    cand = loc & ~core & (A >= GRATING_DB)
    cand[[0, -1], :] = False
    cand[:, [0, -1]] = False
    steer = [(math.sin(math.radians(a)) * math.cos(math.radians(e)), math.sin(math.radians(e)))
             for a in (-fov_az, 0.0, fov_az) for e in (-fov_el, 0.0, fov_el)]
    for j, i in zip(*np.nonzero(cand)):
        du, dv = float(g[i]), float(g[j])
        vis = [(u0, v0) for u0, v0 in steer if (u0 + du) ** 2 + (v0 + dv) ** 2 < 1 - 1e-9]
        if not vis:
            continue
        at = None
        if du * du + dv * dv < 1:
            el = math.degrees(math.asin(dv))
            at = (math.degrees(math.asin(min(max(du / max(math.cos(math.radians(el)), 1e-9), -1), 1))), el)
        out.append({'du': du, 'dv': dv, 'level_db': float(A[j, i]), 'az_deg': at[0] if at else None,
                    'el_deg': at[1] if at else None, 'visible_when': len(vis)})
    out.sort(key=lambda d: -d['level_db'])
    return out[:12]


def element_db(az_deg, el_deg, az3: float, el3: float):
    """One-way power pattern of an element, dB: a Gaussian main beam −3 (az/az3)² − 3 (el/el3)², floor −40 dB."""
    a = np.asarray(az_deg, float) / max(az3, 1e-3)
    e = np.asarray(el_deg, float) / max(el3, 1e-3)
    return np.maximum(-3.0 * (a * a + e * e), ELEMENT_FLOOR_DB)


# =========================================================================== two-ray ground multipath
def roughness_factor(grazing_deg, sigma_h_m: float, f_ghz: float):
    """Ament's specular roughness factor ρ = exp(−2 (k σ_h sin ψ)²)."""
    k = 2 * math.pi / wavelength_m(f_ghz)
    return np.exp(-2 * (k * sigma_h_m * np.sin(np.radians(np.asarray(grazing_deg, float)))) ** 2)


def mean_f4(a, phi, width):
    """The mean of |1 + a e^(−jφ)|⁴ over a phase that runs linearly through φ ± width/2 (a slice of a target's height):
    (1 + a²)² + 4a(1 + a²) E[cos φ] + 4a² E[cos² φ], E[cos φ] = cos φ sinc(w/2), E[cos² φ] = ½ + ½ cos 2φ sinc(w)."""
    a, phi, w = np.asarray(a, float), np.asarray(phi, float), np.asarray(width, float)
    c1 = np.cos(phi) * np.sinc(w / (2 * math.pi))            # np.sinc(t) = sin(πt) / (πt)
    c2 = 0.5 + 0.5 * np.cos(2 * phi) * np.sinc(w / math.pi)
    return (1 + a * a) ** 2 + 4 * a * (1 + a * a) * c1 + 4 * a * a * c2


def two_ray(d, h_r: float, h_t: float, f_ghz: float, gamma, spread=True, ratio=1.0):
    """One-way field factor F = |1 + Γ s ratio e^(−jk(R_r − R_d))| at ground distances d (m) between a radar h_r and a
    target h_t above a flat ground, s = R_d / R_r (spread=True) or 1; gamma: complex reflection, scalar or per d.
    Returns (F, R_d, R_r, grazing deg)."""
    d = np.asarray(d, float)
    Rd = np.hypot(d, h_t - h_r)
    Rr = np.hypot(d, h_t + h_r)
    k = 2 * math.pi / wavelength_m(f_ghz)
    s = Rd / Rr if spread else 1.0
    F = np.abs(1 + np.asarray(gamma) * s * np.asarray(ratio) * np.exp(-1j * k * (Rr - Rd)))
    return F, Rd, Rr, np.degrees(np.arctan2(h_t + h_r, d))


# =========================================================================== settings
LIMITS = {'taylor_sll_db': (13.0, 60.0), 'taylor_nbar': (2, 12), 'face_reflection_db': (-30.0, 0.0),
          'ghost_floor_db': (-60.0, -3.0), 'roughness_mm': (0.0, 10.0), 'max_range_m': (20.0, 1000.0),
          'link_snr_db': (0.0, 30.0), 'link_loss_db': (0.0, 30.0), 'integration_ms': (0.1, 1000.0),
          'map_step_deg': (0.1, 2.0), 'tilt_max_deg': (5.0, 60.0)}
CLASS_LIMITS = {'rcs_m2': (1e-4, 1e5), 'height_m': (0.0, 5.0), 'span_m': (0.0, 2.0)}
SPAN_SAMPLES = 7                       # height slices a target's echo is averaged over (incoherently)


@dataclass
class ImagingRadarSettings:
    window: str = 'taylor'              # the angle FFT's taper: uniform | hann | taylor
    taylor_sll_db: float = 30.0         # Taylor design sidelobe level (dB below the peak)
    taylor_nbar: int = 4
    road: str = 'asphalt_concrete'      # core.materials.ROAD_SURFACES key, or 'none' (no ground reflection)
    roughness_mm: float = 1.0           # rms height of the road's texture (assumed typical)
    polarisation: str = 'auto'          # the product's when stated, else vertical; or vertical | horizontal
    face_reflection_db: float = 0.0     # the radar front's reflection in the cavity (0 dB: metal, the worst case)
    ghost_floor_db: float = -30.0       # images weaker than this (relative to the target) are not listed
    max_range_m: float = 400.0          # detection is searched out to here
    link_snr_db: float = 15.0           # link-budget threshold SNR (Pd 0.9, Pfa 1e-6 needs about 13–15 dB)
    link_loss_db: float = 6.0           # link-budget losses (taper, implementation, radome not included)
    integration_ms: float = 0.0         # link-budget active frame time (0: the product's)
    map_step_deg: float = 0.5           # the beam map's step (finer when the beam needs it)
    tilt_max_deg: float = 30.0          # the mounting advice sweeps the fascia tilt up to this
    swap_model: str = ''                # evaluate every radar as this library model (the project is not changed)
    classes: list = field(default_factory=lambda: [dict(c) for c in DEFAULT_CLASSES])
    radars: list = field(default_factory=list)    # sensor uids to study ([] every visible radar)

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d) -> 'ImagingRadarSettings':
        s = ImagingRadarSettings()
        d = d if isinstance(d, dict) else {}
        if d.get('window') in WINDOWS:
            s.window = d['window']
        from .materials import ROAD_SURFACES
        if d.get('road') in ROAD_SURFACES or d.get('road') == 'none':
            s.road = d['road']
        if d.get('polarisation') in POLARISATIONS:
            s.polarisation = d['polarisation']
        for k, (lo, hi) in LIMITS.items():
            if k in d:
                try:
                    v = float(d[k])
                except (TypeError, ValueError):
                    continue
                if math.isfinite(v):
                    v = min(max(v, lo), hi)
                    setattr(s, k, int(round(v)) if k == 'taylor_nbar' else v)
        im = d.get('integration_ms')
        try:
            if float(im) == 0.0:
                s.integration_ms = 0.0
        except (TypeError, ValueError):
            pass
        sm = d.get('swap_model')
        s.swap_model = str(sm)[:80] if isinstance(sm, str) else ''
        cl = d.get('classes')
        if isinstance(cl, list):
            out = []
            for c in cl[:16]:
                if not isinstance(c, dict):
                    continue
                name = str(c.get('name', '')).strip()[:40]
                try:
                    rcs, h = float(c.get('rcs_m2')), float(c.get('height_m', 1.0))
                    sp = float(c.get('span_m', 0.0))
                except (TypeError, ValueError):
                    continue
                if not name or not (math.isfinite(rcs) and math.isfinite(h)) or name in {o['name'] for o in out}:
                    continue
                sp = sp if math.isfinite(sp) else 0.0
                lo, hi = CLASS_LIMITS['rcs_m2']
                hlo, hhi = CLASS_LIMITS['height_m']
                h = min(max(h, hlo), hhi)
                out.append({'name': name, 'rcs_m2': min(max(rcs, lo), hi), 'height_m': h,
                            'span_m': min(max(sp, 0.0), CLASS_LIMITS['span_m'][1], h),
                            'source': str(c.get('source', 'yours'))[:200]})
            if out:
                s.classes = out
        r = d.get('radars')
        if isinstance(r, (list, tuple)):
            s.radars = sorted({str(u)[:40] for u in r if isinstance(u, str) and u})[:64]
        return s


def settings_of(project) -> ImagingRadarSettings:
    return ImagingRadarSettings.from_dict(getattr(project.analysis, 'imaging_radar', None))


# =========================================================================== one radar
@dataclass
class RadarStudy:
    """What the analysis needs of one radar: its pose, the model analysed and where its figures come from."""
    sensor: object
    model: object               # the SensorModel analysed (the array's, the swap's or the zone model's)
    zone_model: object          # the radar's own model in the project
    array: ArraySpec | None
    params: dict
    swapped: bool
    az3: float
    el3: float
    f_ghz: float
    r0_m: float                 # free-space range on rcs0 at the element's boresight
    rcs0: float
    range_from: str             # 'datasheet' | 'link' | 'zone'


def _study_for(project, s, st: ImagingRadarSettings) -> RadarStudy:
    own = project.sensor_model_for(s)
    swapped = False
    m = own
    if st.swap_model:
        sm = project.sensor_models.get(st.swap_model)
        if sm is not None and sm.kind == 'radar':
            m, swapped = sm, sm.id != own.id
    if m.ranging != 'mimo' and m.id in ZONE_TO_ARRAY:
        am = project.sensor_models.get(ZONE_TO_ARRAY[m.id]) or next(
            (x for x in library() if x.id == ZONE_TO_ARRAY[m.id]), None)
        if am is not None:
            m = am
    p = dict(m.params) if m.ranging == 'mimo' and isinstance(m.params, dict) else {}
    arr = array_spec(m) if p else None
    zones = [z for z in (m.zones or []) if z[0] > 0]
    az_half = max((z[1] for z in zones), default=45.0)
    el_half = max((z[2] for z in zones), default=10.0)
    az3 = _num(p.get('element_az_3db_deg'), az_half / math.sqrt(2))
    el3 = _num(p.get('element_el_3db_deg'), el_half / math.sqrt(2))
    f = _num(p.get('carrier_ghz'), 76.5)
    rcs0 = _num(p.get('rcs_ref_m2'), S.REF_RCS)
    src = p.get('range_from', 'datasheet') if p else 'zone'
    r0 = None
    if src == 'link':
        r0 = link_range(p, rcs0, st.link_snr_db, st.link_loss_db, st.integration_ms or None)
    if r0 is None:
        r0 = _num(p.get('range_ref_m'), 0.0) or max((z[0] for z in zones), default=100.0)
        if src == 'link':
            src = 'datasheet'
    return RadarStudy(s, m, own, arr, p, swapped, float(az3), float(el3), float(f), float(r0), float(rcs0), src)


def _swapped_sensor(s, model_id):
    import copy
    q = copy.copy(s)
    q.model_id = model_id
    return q


def _polarisation(rs: RadarStudy, st: ImagingRadarSettings) -> str:
    if st.polarisation in ('vertical', 'horizontal'):
        return st.polarisation
    p = rs.params.get('polarisation')
    return p if p in ('vertical', 'horizontal') else 'vertical'


def _beam(rs: RadarStudy, st: ImagingRadarSettings, maps=True) -> dict | None:
    """Resolution, sidelobes, grating lobes, the cuts and (maps) the beam map of the radar's array."""
    arr = rs.array
    if arr is None:
        return None
    U, cnt = arr.unique()
    bf = beamformer(U, st.window, st.taylor_sll_db, st.taylor_nbar)
    bf1 = beamformer(U, 'uniform')
    na, ne = len(bf.y), len(bf.z)
    span_y = float(np.ptp(bf.y)) if na else 0.0
    span_z = float(np.ptp(bf.z)) if ne else 0.0
    est = math.degrees(UNIFORM_K * 2 / max(span_y + 1, 1.0))
    step = min(max(est / 25, 0.005), 0.25)
    th = np.arange(-90.0, 90.0 + step / 2, step)
    sv = np.sin(np.radians(th))
    cuts = {'deg': th}
    out = {}
    for name in ('az', 'el'):
        a_w = db_pow(getattr(bf, name)(sv))
        a_u = db_pow(getattr(bf1, name)(sv))
        L, Lu = lobe(th, a_w), lobe(th, a_u)
        out[name] = {'hpbw_deg': L['hpbw_deg'], 'hpbw_uniform_deg': Lu['hpbw_deg'], 'psl_db': L['psl_db'],
                     'psl_uniform_db': Lu['psl_db'], 'null_deg': (L['null_lo_deg'], L['null_hi_deg'])}
        cuts[name] = a_w
        cuts[name + '_uniform'] = a_u
    zones = [z for z in (rs.model.zones or []) if z[0] > 0]
    fov_az = max((z[1] for z in zones), default=45.0)
    fov_el = max((z[2] for z in zones), default=10.0)
    edge = _num(rs.params.get('az_edge_deg'), min(fov_az, 60.0))
    hp_edge = steered_hpbw(bf, edge) if na > 1 else float('nan')
    gl = grating_lobes(bf, fov_az, fov_el) if len(U) > 1 else []
    lam = wavelength_m(rs.f_ghz)
    B = _num(rs.params.get('bandwidth_ghz'), 0.0)
    rr = _num(rs.params.get('range_res_m'), 0.0) or (C0 / (2 * B * 1e9) if B > 0 else float('nan'))
    res = {'n_tx': arr.n_tx, 'n_rx': arr.n_rx,
           'n_virtual': int(len(arr.virtual)) if arr.layout == 'public' else int(arr.n_virtual_stated),
           'n_unique': int(len(U)),
           'n_virtual_stated': arr.n_virtual_stated, 'n_az': na, 'n_el': ne, 'redundant': int((cnt - 1).sum()),
           'row_el': bf.row_z, 'column_az': bf.col_y,
           'aperture_az_lambda': (span_y + 1) / 2 if na > 1 else 0.0,
           'aperture_el_lambda': (span_z + 1) / 2 if ne > 1 else 0.0,
           'layout': arr.layout, 'layout_note': arr.note, 'wavelength_mm': lam * 1000, 'carrier_ghz': rs.f_ghz,
           'window': st.window, 'az_res_deg': out['az']['hpbw_deg'], 'el_res_deg': out['el']['hpbw_deg'] if ne > 1
           else float('nan'), 'az_res_uniform_deg': out['az']['hpbw_uniform_deg'],
           'el_res_uniform_deg': out['el']['hpbw_uniform_deg'] if ne > 1 else float('nan'),
           'az_res_edge_deg': hp_edge, 'az_edge_deg': edge, 'psl_az_db': out['az']['psl_db'],
           'psl_el_db': out['el']['psl_db'] if ne > 1 else float('-inf'),
           'psl_az_uniform_db': out['az']['psl_uniform_db'], 'grating': gl, 'range_res_m': rr,
           'stated': {k: rs.params[k] for k in ('az_res_deg', 'el_res_deg', 'az_res_edge_deg', 'range_res_m',
                                                 'v_max_mps', 'v_res_mps', 'n_virtual', 'n_tx', 'n_rx')
                      if k in rs.params},
           'fov_az_deg': fov_az, 'fov_el_deg': fov_el, 'cuts': cuts,
           'tx': np.asarray(arr.tx, float), 'rx': np.asarray(arr.rx, float), 'virtual': U, 'counts': cnt}
    if maps:
        ms = min(st.map_step_deg, max(min(res['az_res_deg'] if math.isfinite(res['az_res_deg']) else 2.0,
                                          res['el_res_deg'] if math.isfinite(res['el_res_deg']) else 2.0) / 3, 0.2))
        az = np.arange(-90.0, 90.0 + ms / 2, ms)
        el = np.arange(-60.0, 60.0 + ms / 2, ms)
        A = db_pow(bf.azel(az, el)).astype(np.float32)
        E = element_db(az[None, :], el[:, None], rs.az3, rs.el3).astype(np.float32)
        res['map'] = {'az': az, 'el': el, 'af_db': A, 'db': np.maximum(A + E, -80.0).astype(np.float32)}
    return res


def _fascia_normal(project, s, mesh, fm, near_m):
    """The face normal (radar frame, unit) of the first surface the radar's boresight leaves the body through."""
    from . import materials as M
    pos = s.pos().astype(float)
    R = S.rotation(s)
    near = M._near_faces(mesh, pos, near_m)
    if not len(near):
        return None
    T = M._hits(pos, R[:, 0][None, :], mesh.tri[near], near_m)
    if not np.isfinite(T).any():
        return None
    j = int(np.argmin(T[0]))
    n = fm.normals[near[j]]
    n = R.T @ (n / max(np.linalg.norm(n), 1e-12))
    return n if n[0] >= 0 else -n            # pointing away from the radar


def _householder(n):
    n = np.asarray(n, float)
    return np.eye(3) - 2 * np.outer(n, n)


def _dir(az, el):
    a, e = np.radians(az), np.radians(el)
    return np.stack([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e) * np.ones_like(a)], -1)


def _angles(D):
    D = np.asarray(D, float)
    return (np.degrees(np.arctan2(D[..., 1], D[..., 0])),
            np.degrees(np.arcsin(np.clip(D[..., 2] / np.maximum(np.linalg.norm(D, axis=-1), 1e-12), -1, 1))))


def _cavity(rs: RadarStudy, rep: dict | None, normal, beam, st: ImagingRadarSettings, mst) -> dict | None:
    """The fascia's images of targets (cavity ghosts), the merged ripple and the tilt and gap advice."""
    from . import materials as M
    if rep is None or rep.get('mount') not in ('behind', 'deep') or not rep.get('class'):
        return None
    cls = M.CLASSES.get(rep['class'])
    if cls is None or cls.opaque:
        return None
    f = rs.f_ghz
    th_mm = float(rep.get('thickness_mm') or cls.thickness_mm)
    inc = float(rep.get('incidence_deg') or 0.0)
    gap = float(rep.get('gap_mm') or 0.0) / 1000.0
    layers = M.layers_of(cls, rep.get('paint', mst.paint) if cls.painted else 'none', f, th_mm)
    T1, R1, r_te, r_tm = M.transmission(layers, f, np.array([inc]))
    r = float(math.sqrt(R1[0]))
    face = 10 ** (st.face_reflection_db / 20)
    a1 = r * face                                         # one bounce's amplitude relative to the direct echo
    n = np.asarray(normal if normal is not None else [math.cos(math.radians(inc)), 0, math.sin(math.radians(inc))],
                   float)
    n = n / np.linalg.norm(n)
    Hm = _householder(n) @ _householder([1.0, 0, 0])
    tilt_plane = 'vertical' if abs(n[2]) >= abs(n[1]) else 'horizontal'
    zones = [z for z in (rs.model.zones or []) if z[0] > 0]
    fov_az = max((z[1] for z in zones), default=45.0)
    fov_el = max((z[2] for z in zones), default=10.0)
    az_res = beam['az_res_deg'] if beam else float('nan')
    el_res = beam['el_res_deg'] if beam else float('nan')
    rres = beam['range_res_m'] if beam else float('nan')
    ghosts = []
    for order in (1, 2):
        lvl0 = 20 * math.log10(max(a1 ** order, 1e-12))
        for t_az in (-fov_az, -fov_az / 2, 0.0, fov_az / 2, fov_az):
            d = _dir(np.array(t_az), np.array(0.0))
            g = d
            for _ in range(order):
                g = Hm @ g
            gaz, gel = (float(v) for v in _angles(g))
            ce = float(np.dot(d, g))
            off = math.degrees(math.acos(min(max(ce, -1.0), 1.0)))
            lvl = lvl0 + float(element_db(gaz, gel, rs.az3, rs.el3) - element_db(t_az, 0.0, rs.az3, rs.el3))
            in_field = abs(gaz) <= fov_az + 1e-9 and abs(gel) <= fov_el + 1e-9 and g[0] > 0
            dr = order * gap * math.cos(math.radians(inc))
            sep_az, sep_el = abs(gaz - t_az), abs(gel)
            res_ang = (math.isfinite(az_res) and sep_az > az_res) or (math.isfinite(el_res) and sep_el > el_res)
            res_rng = math.isfinite(rres) and dr > rres
            ghosts.append({'kind': 'cavity', 'order': order, 'target_az_deg': float(t_az), 'target_el_deg': 0.0,
                           'ghost_az_deg': gaz, 'ghost_el_deg': gel, 'angle_offset_deg': off,
                           'range_offset_m': dr, 'level_db': lvl, 'in_field': bool(in_field),
                           'resolved': 'angle' if res_ang else 'range' if res_rng else '',
                           'listed': bool(lvl >= st.ghost_floor_db)})
    # merged images move the amplitude and the angle
    rip = (20 * math.log10(1 + a1), 20 * math.log10(max(1 - a1, 1e-6)))
    off0 = next(g['angle_offset_deg'] for g in ghosts if g['order'] == 1 and g['target_az_deg'] == 0.0)
    bias = a1 / (1 + a1) * off0 if off0 < (az_res if math.isfinite(az_res) else 180) else 0.0
    # gap: the cavity's anti-resonance nearest the gap, at this carrier
    lam = wavelength_m(f)
    gap_adv = None
    if gap > 0:
        gg = np.linspace(max(gap - lam / 2, 0.0), gap + lam / 2, 241)
        Fg = M.cavity_buildup(r_te[0], r_tm[0], gg, inc, f, r_radar=-face)
        F0 = float(M.cavity_buildup(r_te[0], r_tm[0], gap, inc, f, r_radar=-face))
        mins = [i for i in range(1, len(gg) - 1) if Fg[i] <= Fg[i - 1] and Fg[i] <= Fg[i + 1]] or [int(np.argmin(Fg))]
        k = min(mins, key=lambda i: abs(gg[i] - gap))
        gap_adv = {'gap_mm': gap * 1000, 'buildup_db': float(10 * np.log10(F0)), 'best_gap_mm': float(gg[k] * 1000),
                   'best_buildup_db': float(10 * np.log10(Fg[k])), 'move_mm': float((gg[k] - gap) * 1000),
                   'ripple_db': float(10 * np.log10(Fg.max() / Fg.min()))}
    # tilt: the boresight target's first image sits 2α off; sweep α
    tilts = np.arange(0.0, st.tilt_max_deg + 0.5, 1.0)
    Tt, Rt, _, _ = M.transmission(layers, f, tilts)
    half = fov_el if tilt_plane == 'vertical' else fov_az
    res_pl = el_res if tilt_plane == 'vertical' else az_res
    rows = []
    for a, t, rr_ in zip(tilts, Tt, Rt):
        lv = 20 * math.log10(max(math.sqrt(rr_) * face, 1e-12))
        ga = 2 * a
        e_off = float(element_db(0.0, ga, rs.az3, rs.el3) if tilt_plane == 'vertical'
                      else element_db(ga, 0.0, rs.az3, rs.el3))
        inside = ga <= half
        resolved = math.isfinite(res_pl) and ga > res_pl
        rows.append({'tilt_deg': float(a), 'loss_db': float(-2 * 10 * math.log10(max(t, 1e-12))),
                     'reflection_db': float(10 * math.log10(max(rr_, 1e-12))), 'ghost_level_db': lv + e_off,
                     'ghost_in_field': bool(inside and resolved), 'merged': bool(not resolved)})
    ok = [x for x in rows if x['loss_db'] <= mst.loss_warn_db]
    merge_ok = 20 * math.log10(10 ** (MERGE_RIPPLE_DB / 20) - 1)        # a merged image this weak moves < 1 dB

    def clean(x):
        if x['merged']:
            return x['ghost_level_db'] <= merge_ok
        return not x['ghost_in_field'] or x['ghost_level_db'] < st.ghost_floor_db
    pool = ok or rows
    good = [x for x in pool if clean(x)]
    best = min(good, key=lambda x: abs(x['tilt_deg'] - inc)) if good else min(pool, key=lambda x: x['ghost_level_db'])
    best_th = M.best_thickness(cls, rep.get('paint', mst.paint) if cls.painted else 'none', f, inc, th_mm)
    return {'class': cls.id, 'class_name': cls.name, 'thickness_mm': th_mm, 'gap_mm': gap * 1000,
            'incidence_deg': inc, 'reflection_db': float(10 * math.log10(max(R1[0], 1e-12))),
            'loss_db': float(-2 * 10 * math.log10(max(T1[0], 1e-12))), 'face_reflection_db': st.face_reflection_db,
            'image_db': float(20 * math.log10(max(a1, 1e-12))), 'normal': [float(x) for x in n],
            'tilt_plane': tilt_plane, 'ghosts': ghosts, 'ripple_db': rip, 'angle_bias_deg': float(bias),
            'gap_advice': gap_adv, 'tilt_sweep': rows, 'best_tilt': best, 'best_thickness': best_th}


def _ground(rs: RadarStudy, st: ImagingRadarSettings, s, beam, fascia_loss, pitch=None, n=12000) -> dict:
    """Detection range along the radar's heading per class over a flat road, the fades and the mirror ghosts."""
    from .materials import ROAD_SURFACES, road_reflection
    pol = _polarisation(rs, st)
    h_r = float(s.z)
    if pitch is not None:
        import copy
        s = copy.copy(s)
        s.pitch = pitch
    R = S.rotation(s)
    fwd = R[:, 0]
    hx = np.array([fwd[0], fwd[1], 0.0])
    hn = np.linalg.norm(hx)
    hx = hx / hn if hn > 1e-9 else np.array([1.0, 0.0, 0.0])
    d = np.geomspace(0.5, st.max_range_m, n)
    lam = wavelength_m(rs.f_ghz)
    k = 2 * math.pi / lam
    pos = s.pos().astype(float)
    el_res = beam['el_res_deg'] if beam else float('nan')
    out = {'surface': st.road, 'surface_name': ROAD_SURFACES[st.road][0] if st.road in ROAD_SURFACES else 'none',
           'eps_r': ROAD_SURFACES[st.road][1] if st.road in ROAD_SURFACES else None, 'polarisation': pol,
           'radar_height_m': h_r, 'pitch_deg': float(s.pitch), 'classes': {}}

    def path(h_t):
        """Direct and image geometry, gains and the road's reflection for a scatterer h_t up."""
        P = pos[None, :] * np.array([1, 1, 0]) + d[:, None] * hx[None, :]
        q_d = (P + np.array([0, 0, h_t]) - pos) @ R
        q_i = (P - np.array([0, 0, h_t]) - pos) @ R
        Rd, Rr = np.linalg.norm(q_d, axis=1), np.linalg.norm(q_i, axis=1)
        az_d, el_d = _angles(q_d)
        az_i, el_i = _angles(q_i)
        E_d = element_db(az_d, el_d, rs.az3, rs.el3)
        E_i = element_db(az_i, el_i, rs.az3, rs.el3)
        psi = np.degrees(np.arctan2(h_t + h_r, d))
        if st.road in ROAD_SURFACES and h_r > 0:
            gam = road_reflection(st.road, psi, pol) * roughness_factor(psi, st.roughness_mm / 1000, rs.f_ghz)
        else:
            gam = np.zeros(len(d), complex)
        a = np.abs(gam) * (Rd / Rr) * 10 ** ((E_i - E_d) / 20)
        phi = k * (Rr - Rd) - np.angle(gam)
        dphi = k * ((h_t + h_r) / Rr - (h_t - h_r) / Rd)          # how fast the phase turns with the height
        F = np.abs(1 + a * np.exp(-1j * phi))
        return dict(Rd=Rd, Rr=Rr, az_d=az_d, el_d=el_d, az_i=az_i, el_i=el_i, E_d=E_d, E_i=E_i, gam=gam, F=F,
                    a=a, phi=phi, dphi=dphi)
    for c in st.classes:
        h_t = float(c['height_m'])
        span = float(c.get('span_m', 0.0))
        # the scatterers spread evenly over h_t ± span: SPAN_SAMPLES slices, each slice's F⁴ averaged over its
        # height in closed form (the phase turns linearly across a slice: E[cos φ] = cos φ0 sinc(φ' δ / 2) …), so
        # evenly spaced slices cannot line up on a null together
        n_sl = SPAN_SAMPLES if span > 0 else 1
        dl = 2 * span / n_sl
        hs = h_t - span + dl * (np.arange(n_sl) + 0.5) if span > 0 else np.array([h_t])
        hs = np.maximum(hs, 0.02)
        r0c = rs.r0_m * (c['rcs_m2'] / rs.rcs0) ** 0.25
        mid = path(h_t)
        loss = fascia_loss(mid['az_d'])
        p_two = np.zeros(len(d))
        p_free = np.zeros(len(d))
        for h in hs:
            g = mid if h == h_t else path(h)
            base = (r0c / g['Rd']) ** 4 * 10 ** (2 * g['E_d'] / 10)
            p_two += base * np.maximum(mean_f4(g['a'], g['phi'], g['dphi'] * dl), 1e-36)
            p_free += base
        margin = 10 * np.log10(p_two / len(hs)) - loss
        free = 10 * np.log10(p_free / len(hs)) - loss
        det = margin >= 0
        runs = _runs(det)
        if runs:
            # the first stretch detected without a gap that is long enough to be the target's own (not a sidelobe
            # catching it at arm's length): at least 2 m and a quarter of its end; else the longest
            main = next((ab for ab in runs if d[ab[1]] - d[ab[0]] >= max(2.0, 0.25 * d[ab[1]])),
                        max(runs, key=lambda ab: d[ab[1]] - d[ab[0]]))
            near, cont = float(d[main[0]]), float(d[main[1]])
            last = float(d[runs[-1][1]])
            fades = [[float(d[a[1]]), float(d[b[0]])] for a, b in zip(runs, runs[1:]) if b[0] > main[0]]
            merged = []
            for a, b in fades:
                if merged and a - merged[-1][1] < 0.5:
                    merged[-1][1] = b
                else:
                    merged.append([a, b])
            fades = merged
            closer = float(d[runs[0][0]]) if runs[0][0] < main[0] else None
        else:
            near = cont = last = float('nan')
            fades, closer = [], None
        fdet = free >= 0
        free_far = float(d[np.flatnonzero(fdet)[-1]]) if fdet.any() else float('nan')
        # the mirror target below the road at a few ranges (the class's middle height)
        ghosts = []
        Rd, Rr, gam, E_i, E_d = mid['Rd'], mid['Rr'], mid['gam'], mid['E_i'], mid['E_d']
        for r_at in (10.0, 25.0, 50.0, 100.0):
            if not (np.isfinite(last) and r_at <= last):
                continue
            i = int(np.argmin(np.abs(d - r_at)))
            g_lvl = float(40 * math.log10(max(abs(gam[i]) * Rd[i] / Rr[i], 1e-12)) + 2 * (E_i[i] - E_d[i]))
            off = abs(float(mid['el_i'][i] - mid['el_d'][i]))
            ghosts.append({'kind': 'ground', 'class': c['name'], 'range_m': float(d[i]),
                           'target_el_deg': float(mid['el_d'][i]), 'ghost_el_deg': float(mid['el_i'][i]),
                           'ghost_az_deg': float(mid['az_i'][i]), 'target_az_deg': float(mid['az_d'][i]),
                           'angle_offset_deg': off, 'range_offset_m': float(Rr[i] - Rd[i]), 'level_db': g_lvl,
                           'cross_level_db': float(20 * math.log10(max(abs(gam[i]) * Rd[i] / Rr[i], 1e-12))
                                                   + (E_i[i] - E_d[i])),
                           'resolved': 'angle' if (math.isfinite(el_res) and off > el_res) else '',
                           'in_field': True, 'listed': bool(g_lvl >= st.ghost_floor_db)})
        out['classes'][c['name']] = {
            'rcs_m2': c['rcs_m2'], 'height_m': h_t, 'span_m': span, 'free_range_m': float(r0c), 'free_m': free_far,
            'near_m': near, 'continuous_m': cont, 'max_m': last, 'fades': fades, 'closer_m': closer,
            'first_null_m': float(2 * h_r * h_t / lam) if h_r > 0 else float('nan'),
            'ghosts': ghosts, 'curve': {'d': d, 'margin_db': margin.astype(np.float32),
                                        'free_db': free.astype(np.float32)}}
    return out


def _runs(mask) -> list:
    """[(first, last)] index pairs of the runs of True in a boolean array."""
    m = np.asarray(mask, bool)
    if not m.any():
        return []
    e = np.diff(np.r_[0, m.astype(np.int8), 0])
    return list(zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1) - 1))


def _vs_azimuth(rs: RadarStudy, st, fascia_loss) -> dict:
    """Free-space detection range per class against azimuth at zero elevation (the 4D field in plan)."""
    zones = [z for z in (rs.model.zones or []) if z[0] > 0]
    fov = max((z[1] for z in zones), default=60.0)
    az = np.linspace(-min(fov + 20, 90.0), min(fov + 20, 90.0), 181)
    E = element_db(az, 0.0, rs.az3, rs.el3)
    loss = fascia_loss(az)
    out = {'az_deg': az}
    for c in st.classes:
        r0c = rs.r0_m * (c['rcs_m2'] / rs.rcs0) ** 0.25
        out[c['name']] = r0c * 10 ** ((2 * E - loss) / 40)
    return out


def _zone_side(project, s, own, st, rep, materials_factors) -> dict:
    """What the zone model (the radar's own SensorModel) says per class: range on the boresight and the field."""
    zones = [z for z in (own.zones or []) if z[0] > 0]
    out = {'model': f'{own.vendor} {own.name}'.strip(), 'ranging': own.ranging or 'zone',
           'az_half_deg': max((z[1] for z in zones), default=0.0),
           'el_half_deg': max((z[2] for z in zones), default=0.0), 'classes': {}}
    for c in st.classes:
        k = S.radar_scale(c['rcs_m2'])
        rng = 0.0
        for i, (zr, _ah, _eh) in enumerate(zones):
            fac = materials_factors[i] if materials_factors and i < len(materials_factors) else 1.0
            rng = max(rng, zr * k * fac)
        out['classes'][c['name']] = rng
    return out


def evaluate(project, s, st: ImagingRadarSettings | None = None, geom=None, maps=True, advise=True) -> dict:
    """One radar's study (see the module docstring). geom: (mesh, fm, material settings) or None to build them."""
    from . import materials as M
    st = st or settings_of(project)
    rs = _study_for(project, s, st)
    mst = M.settings_of(project)
    if geom is None:
        mesh, parts = M._geometry(project, None)
        fm = M.face_materials(mesh, parts, mst) if mesh is not None else None
    else:
        mesh, fm, mst = geom
    s_eval = _swapped_sensor(s, rs.model.id) if rs.model.id != s.model_id else s
    rep = None
    normal = None
    if mesh is not None and fm is not None:
        try:
            rep = M.radar_report(project, s_eval, mesh, fm, mst)
            if rep is not None and rep.get('mount') in ('behind', 'deep'):
                normal = _fascia_normal(project, s_eval, mesh, fm, M.NEAR_M)
        except Exception:                                  # noqa: BLE001 — the study works without the materials
            rep = None
    if rep is not None and rep.get('map') is not None:
        mp = rep['map']
        row = int(np.argmin(np.abs(np.asarray(mp['el'], float))))
        az_m = np.asarray(mp['az'], float)
        ls = np.asarray(mp['loss_db'], float)[row]
        ls = np.where(np.isfinite(ls), ls, 200.0)

        def fascia_loss(az):
            return np.interp(np.asarray(az, float), az_m, ls)
    else:
        def fascia_loss(az):
            return np.zeros(np.shape(az))
    beam = _beam(rs, st, maps=maps)
    ground = _ground(rs, st, s_eval, beam, fascia_loss)
    cav = _cavity(rs, rep, normal, beam, st, mst)
    vs_az = _vs_azimuth(rs, st, fascia_loss)
    factors = None
    if rep is not None and not rs.swapped:
        factors = [float(z['factor']) if z['loss_db'] != float('inf') else 1.0 for z in rep['zones']]
    zone = _zone_side(project, s, rs.zone_model, st, rep, factors)
    # pitch advice: the weakest class's continuous range against the radar's pitch
    weakest = min(st.classes, key=lambda c: c['rcs_m2'])['name'] if st.classes else None
    pitch_rows = []
    if advise and weakest and st.road != 'none':
        one = ImagingRadarSettings.from_dict(dict(st.to_dict(), classes=[c for c in st.classes
                                                                         if c['name'] == weakest]))
        for dp in np.arange(-4.0, 4.01, 1.0):
            g = _ground(rs, one, s_eval, beam, fascia_loss, pitch=float(s.pitch + dp), n=4000)['classes'][weakest]
            pitch_rows.append({'pitch_deg': float(s.pitch + dp), 'continuous_m': g['continuous_m'],
                               'near_m': g['near_m'], 'fades': len(g['fades'])})
    return {'uid': s.uid, 'tag': s.tag, 'name': s.name, 'model_id': rs.model.id,
            'model': f'{rs.model.vendor} {rs.model.name}'.strip(), 'own_model_id': rs.zone_model.id,
            'own_model': f'{rs.zone_model.vendor} {rs.zone_model.name}'.strip(), 'ranging': rs.model.ranging or 'zone',
            'swapped': rs.swapped, 'params': rs.params, 'figures': dict(rs.params.get('figures', {})),
            'element_az_3db_deg': rs.az3, 'element_el_3db_deg': rs.el3, 'carrier_ghz': rs.f_ghz,
            'range_ref_m': rs.r0_m, 'rcs_ref_m2': rs.rcs0, 'range_from': rs.range_from,
            'pose': {k: float(getattr(s, k)) for k in ('x', 'y', 'z', 'yaw', 'pitch', 'roll')},
            'mount': rep.get('mount') if rep else 'none', 'beam': beam, 'ground': ground, 'cavity': cav,
            'vs_az': vs_az, 'zone': zone, 'pitch_sweep': pitch_rows, 'weakest': weakest}


# =========================================================================== the study
@dataclass
class ImagingRadarResult:
    settings: ImagingRadarSettings
    radars: list
    findings: list
    metrics: dict
    variant: str = ''
    variant_uid: str = ''
    swap_note: str = ''

    def worst(self) -> str:
        return self.findings[0]['severity'] if self.findings else 'good'

    def radar(self, tag):
        return next((r for r in self.radars if r['tag'] == tag), None)


def _project_for(project, variant):
    if variant is None or variant is project.variant:
        return project
    import copy
    q = copy.copy(project)
    q.variants = list(project.variants) + ([] if any(v is variant for v in project.variants) else [variant])
    q.active_uid = variant.uid
    return q


def _finite_pose(s) -> bool:
    try:
        return all(math.isfinite(float(getattr(s, k))) for k in ('x', 'y', 'z', 'yaw', 'pitch', 'roll'))
    except (TypeError, ValueError):
        return False


def radars_of(project, st: ImagingRadarSettings) -> list:
    """The visible radars of the active variant the study looks at (the selection, else all) with a finite pose."""
    sel = set(st.radars)
    return [s for s in project.variant.sensors if s.kind == 'radar' and s.visible and (not sel or s.uid in sel)
            and _finite_pose(s)]


def run(project, engine=None, settings: ImagingRadarSettings | None = None, variant=None, progress=None,
        cancel=None, maps=True, advise=True) -> ImagingRadarResult | None:
    """Every visible radar of a variant (the active one by default) as a 4D imaging radar. Returns None when
    cancelled. engine: its vehicle geometry is reused for the fascia (None: built here). maps: the beam maps (for the
    page and the PNG); advise: the pitch sweep of the mounting advice."""
    from . import materials as M
    t0 = time.perf_counter()
    p = _project_for(project, variant)
    v = p.variant
    st = settings or settings_of(p)

    def prog(x):
        if progress is not None:
            progress(min(max(float(x), 0.0), 1.0))
    vg = engine.vehicle_geom(p) if engine is not None else None
    mst = M.settings_of(p)
    mesh, parts = M._geometry(p, vg)
    fm = M.face_materials(mesh, parts, mst) if mesh is not None else None
    prog(0.05)
    sens = radars_of(p, st)
    out = []
    for i, s in enumerate(sens):
        if cancel is not None and cancel():
            return None
        out.append(evaluate(p, s, st, (mesh, fm, mst), maps=maps, advise=advise))
        prog(0.05 + 0.95 * (i + 1) / max(len(sens), 1))
    note = ''
    if st.swap_model:
        sm = p.sensor_models.get(st.swap_model)
        note = (f'Every radar evaluated as {sm.vendor} {sm.name} (the project is unchanged).' if sm is not None and
                sm.kind == 'radar' else f'The swap model "{st.swap_model}" is not a radar in the library: ignored.')
    res = ImagingRadarResult(st, out, [], {'seconds': time.perf_counter() - t0, 'radars': len(out)}, v.name, v.uid,
                             note)
    res.findings = _findings(res)
    return res


# =========================================================================== findings
def _f(out, sev, kind, msg, r=None):
    out.append({'severity': sev, 'kind': kind, 'message': msg, 'tags': [r['tag']] if r else [],
                'uids': [r['uid']] if r else []})


def _rng(v):
    return f'{v:.0f} m' if v is not None and math.isfinite(v) else '—'


def _findings(res: ImagingRadarResult) -> list:
    out = []
    st = res.settings
    if res.swap_note:
        _f(out, 'good' if 'ignored' not in res.swap_note else 'warning', 'swap', res.swap_note)
    if not res.radars:
        _f(out, 'warning', 'none', 'No visible radar to study: add one (Rig ▸ Sensors) or show the hidden ones.')
    for r in res.radars:
        b = r['beam']
        t = r['tag']
        if b is None:
            _f(out, 'warning', 'no_array', f'{t} ({r["model"]}) is a zone model with no stated array: angular '
               f'resolution and sidelobes are not known. Swap in an imaging radar to compare.', r)
        else:
            vis = [g for g in b['grating'] if g['level_db'] >= GRATING_DB]
            if vis:
                g = vis[0]
                _f(out, 'serious', 'grating', f'{t}: a grating lobe at {g["level_db"]:.1f} dB enters the visible '
                   f'region when the beam steers within the field: targets there show up twice (angle ambiguity).', r)
            psl = b['psl_az_db']
            if math.isfinite(psl) and psl > -20.0:
                _f(out, 'warning', 'sidelobes', f'{t}: the highest azimuth sidelobe is {psl:.1f} dB with the '
                   f'{WINDOW_LABEL[st.window].lower()} taper: a car\'s sidelobes can hide a pedestrian beside it '
                   f'(10 m² against 0.5 m² is 13 dB).', r)
            el = b['el_res_deg']
            if not math.isfinite(el) or el > 10.0:
                _f(out, 'warning', 'elevation', f'{t}: elevation resolution '
                   + (f'{el:.1f}°' if math.isfinite(el) else 'none (one row)') + ': it cannot tell a bridge or a '
                   'sign gantry from a car under it.', r)
            st_az = b['stated'].get('az_res_deg')
            if st_az and math.isfinite(b['az_res_uniform_deg']):
                _f(out, 'good', 'resolution', f'{t}: {b["az_res_deg"]:.2f}° × '
                   + (f'{el:.2f}°' if math.isfinite(el) else '—') + f' resolution with the {st.window} taper '
                   f'({b["az_res_uniform_deg"]:.2f}° uniform; {st_az:g}° stated).', r)
        c = r['cavity']
        if c is not None:
            vis = [g for g in c['ghosts'] if g['in_field'] and g['listed'] and g['resolved']]
            if vis:
                g = max(vis, key=lambda g: g['level_db'])
                sev = 'serious' if g['level_db'] > -10 else 'warning'
                _f(out, sev, 'cavity_ghost', f'{t}: the fascia images a target at {g["target_az_deg"]:+.0f}° '
                   f'as a ghost at {g["ghost_az_deg"]:+.0f}° az {g["ghost_el_deg"]:+.0f}° el, {g["level_db"]:.1f} dB, '
                   f'inside the field ({c["class_name"].lower()}, {c["incidence_deg"]:.0f}° to the radar).', r)
            elif c['image_db'] > st.ghost_floor_db and c['ripple_db'][0] > MERGE_RIPPLE_DB:
                _f(out, 'warning', 'cavity_merge', f'{t}: the fascia\'s image ({c["image_db"]:.1f} dB) merges with '
                   f'every target: amplitude ±{c["ripple_db"][0]:.1f}/{c["ripple_db"][1]:.1f} dB, angle bias up to '
                   f'{c["angle_bias_deg"]:.2f}°.', r)
            ga = c['gap_advice']
            if ga is not None and abs(ga['move_mm']) > 0.2 and ga['buildup_db'] - ga['best_buildup_db'] > 1.0:
                _f(out, 'warning', 'cavity_gap', f'{t}: move the radar {ga["move_mm"]:+.1f} mm to a '
                   f'{ga["best_gap_mm"]:.1f} mm gap: the cavity build-up falls from {ga["buildup_db"]:+.1f} to '
                   f'{ga["best_buildup_db"]:+.1f} dB.', r)
        g = r['ground']
        if r['weakest'] and r['weakest'] in g['classes']:
            w = g['classes'][r['weakest']]
            fd = [f for f in w['fades'] if f[1] - f[0] >= 1.0]
            if fd:
                a, bb = fd[0]
                _f(out, 'warning', 'fade', f'{t}: {r["weakest"].lower()}s fade at {a:.0f}–{bb:.0f} m where the road\'s '
                   f'reflection cancels the direct path ({len(fd)} fade{"s" if len(fd) > 1 else ""} before '
                   f'{_rng(w["max_m"])}).', r)
            if math.isfinite(w['near_m']) and w['near_m'] > 3.0:
                _f(out, 'warning', 'near_blind', f'{t}: {r["weakest"].lower()}s nearer than {w["near_m"]:.1f} m are '
                   f'below the elevation beam (±{r["element_el_3db_deg"]:.1f}° at −3 dB, the radar '
                   f'{g["radar_height_m"]:.2f} m up).', r)
        zc, gc = r['zone']['classes'], g['classes']
        for name in zc:
            if name in gc and zc[name] > 0 and math.isfinite(gc[name]['continuous_m']):
                diff = gc[name]['continuous_m'] / zc[name] - 1
                if diff < -0.25:
                    _f(out, 'warning', 'zone_gap', f'{t}: the zone model gives {name.lower()}s {zc[name]:.0f} m; the '
                       f'array, the ground and the fascia give {gc[name]["continuous_m"]:.0f} m without a gap.', r)
                    break
    order = {k: i for i, k in enumerate(SEVERITIES)}
    out.sort(key=lambda f: order[f['severity']])
    return out


# =========================================================================== summary, export
def _r(v, nd=3):
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return v
    return round(x, nd) if math.isfinite(x) else None


def summary(res: ImagingRadarResult) -> dict:
    rows = []
    for r in res.radars:
        b = r['beam'] or {}
        g = r['ground']['classes']
        c = r['cavity']
        rows.append({'tag': r['tag'], 'model': r['model'], 'swapped': r['swapped'], 'mount': r['mount'],
                     'az_res_deg': _r(b.get('az_res_deg'), 2), 'el_res_deg': _r(b.get('el_res_deg'), 2),
                     'psl_az_db': _r(b.get('psl_az_db'), 1), 'grating_lobes': len(b.get('grating', [])),
                     'range_res_m': _r(b.get('range_res_m'), 3),
                     'continuous_m': {k: _r(v['continuous_m'], 1) for k, v in g.items()},
                     'fades': {k: len(v['fades']) for k, v in g.items()},
                     'cavity_image_db': _r(c['image_db'], 1) if c else None,
                     'cavity_ghosts_in_field': sum(1 for x in c['ghosts'] if x['in_field'] and x['listed']
                                                   and x['resolved']) if c else 0})
    return {'schema': SCHEMA, 'variant': res.variant, 'radars': rows, 'worst': res.worst(),
            'findings': [f['message'] for f in res.findings if f['severity'] != 'good'][:12],
            'window': res.settings.window, 'swap_model': res.settings.swap_model,
            'seconds': _r(res.metrics.get('seconds'), 2)}


def _ghost_doc(g):
    return {k: (_r(v, 3) if isinstance(v, float) else v) for k, v in g.items()}


def _radar_doc(r) -> dict:
    b = r['beam']
    bd = None
    if b is not None:
        cuts = b['cuts']
        keep = np.linspace(0, len(cuts['deg']) - 1, min(len(cuts['deg']), 721)).astype(int)
        bd = {k: (_r(b[k], 4) if isinstance(b[k], float) else b[k]) for k in
              ('n_tx', 'n_rx', 'n_virtual', 'n_unique', 'n_virtual_stated', 'n_az', 'n_el', 'redundant',
               'aperture_az_lambda', 'aperture_el_lambda', 'layout', 'layout_note', 'wavelength_mm', 'carrier_ghz',
               'window', 'az_res_deg', 'el_res_deg', 'az_res_uniform_deg', 'el_res_uniform_deg', 'az_res_edge_deg',
               'az_edge_deg', 'psl_az_db', 'psl_el_db', 'psl_az_uniform_db', 'range_res_m', 'fov_az_deg',
               'fov_el_deg')}
        bd['stated'] = b['stated']
        bd['grating_lobes'] = [_ghost_doc(g) for g in b['grating']]
        bd['tx_half_wavelengths'] = np.round(b['tx'], 4).tolist()
        bd['rx_half_wavelengths'] = np.round(b['rx'], 4).tolist() if b['layout'] == 'public' else \
            'an ideal filled aperture (the virtual array itself): ' + b['layout_note']
        bd['virtual_unique_half_wavelengths'] = np.round(b['virtual'], 4).tolist() if len(b['virtual']) <= 4096 \
            else f'{len(b["virtual"])} positions (a filled grid of {b["n_az"]} × {b["n_el"]})'
        bd['cuts'] = {'deg': np.round(cuts['deg'][keep], 4).tolist(),
                      **{k: np.round(cuts[k][keep], 2).tolist() for k in ('az', 'az_uniform', 'el', 'el_uniform')}}
    g = r['ground']
    gd = {k: g[k] for k in ('surface', 'surface_name', 'eps_r', 'polarisation', 'radar_height_m', 'pitch_deg')}
    gd['classes'] = {n: {k: (_r(v, 2) if isinstance(v, float) else v) for k, v in c.items() if k != 'curve'} |
                     {'ghosts': [_ghost_doc(x) for x in c['ghosts']]} for n, c in g['classes'].items()}
    c = r['cavity']
    cd = None
    if c is not None:
        cd = {k: (_r(v, 3) if isinstance(v, float) else v) for k, v in c.items()
              if k not in ('ghosts', 'tilt_sweep')}
        cd['ghosts'] = [_ghost_doc(x) for x in c['ghosts']]
        cd['tilt_sweep'] = [_ghost_doc(x) for x in c['tilt_sweep']]
    va = r['vs_az']
    return {'tag': r['tag'], 'uid': r['uid'], 'name': r['name'], 'model_id': r['model_id'], 'model': r['model'],
            'own_model': r['own_model'], 'own_model_id': r['own_model_id'], 'swapped': r['swapped'],
            'ranging': r['ranging'], 'pose': r['pose'], 'mount': r['mount'], 'carrier_ghz': r['carrier_ghz'],
            'element_az_3db_deg': _r(r['element_az_3db_deg']), 'element_el_3db_deg': _r(r['element_el_3db_deg']),
            'range_ref_m': _r(r['range_ref_m'], 1), 'rcs_ref_m2': r['rcs_ref_m2'], 'range_from': r['range_from'],
            'figures': r['figures'], 'beam': bd, 'ground': gd, 'cavity': cd,
            'vs_azimuth_m': {'az_deg': np.round(va['az_deg'], 2).tolist(),
                             **{k: np.round(v, 1).tolist() for k, v in va.items() if k != 'az_deg'}},
            'zone_model': {k: (v if k != 'classes' else {n: _r(x, 1) for n, x in v.items()})
                           for k, v in r['zone'].items()},
            'pitch_sweep': [_ghost_doc(x) for x in r['pitch_sweep']], 'weakest_class': r['weakest']}


def to_doc(res: ImagingRadarResult, project=None) -> dict:
    return {'schema': SCHEMA, 'project': getattr(project, 'name', ''), 'variant': res.variant, 'frame': FRAME,
            'units': 'metres, degrees, dB (power) relative to the main beam or the direct echo; array positions in '
                     'half wavelengths at the carrier (y left, z up in the radar frame)',
            'settings': res.settings.to_dict(), 'swap': res.swap_note, 'assumptions': ASSUMPTIONS,
            'limits': LIMITS_TEXT, 'references': REFERENCES, 'radars': [_radar_doc(r) for r in res.radars],
            'findings': [{'severity': f['severity'], 'kind': f['kind'], 'message': f['message'], 'sensors': f['tags']}
                         for f in res.findings], 'seconds': _r(res.metrics.get('seconds'), 3)}


CSV_COLUMNS = ('tag', 'model', 'swapped', 'mount', 'carrier_ghz', 'n_virtual', 'n_unique', 'layout', 'window',
               'az_res_deg', 'el_res_deg', 'az_res_edge_deg', 'psl_az_db', 'psl_el_db', 'grating_lobes',
               'range_res_m', 'class', 'rcs_m2', 'zone_model_m', 'free_m', 'near_m', 'continuous_m', 'max_m', 'fades',
               'cavity_class', 'gap_mm', 'incidence_deg', 'cavity_image_db', 'cavity_ghosts_in_field')


def to_csv(res: ImagingRadarResult) -> str:
    """One row per radar and class."""
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(CSV_COLUMNS)
    for r in res.radars:
        b = r['beam'] or {}
        c = r['cavity']
        for name, g in r['ground']['classes'].items():
            w.writerow([r['tag'], r['model'], int(r['swapped']), r['mount'], r['carrier_ghz'], b.get('n_virtual', ''),
                        b.get('n_unique', ''), b.get('layout', 'zone'), r['ground'] and res.settings.window,
                        _r(b.get('az_res_deg'), 3), _r(b.get('el_res_deg'), 3), _r(b.get('az_res_edge_deg'), 3),
                        _r(b.get('psl_az_db'), 2), _r(b.get('psl_el_db'), 2), len(b.get('grating', [])),
                        _r(b.get('range_res_m'), 3), name, g['rcs_m2'], _r(r['zone']['classes'].get(name), 1),
                        _r(g['free_m'], 1), _r(g['near_m'], 2), _r(g['continuous_m'], 1), _r(g['max_m'], 1),
                        ' '.join(f'{a:.1f}-{bb:.1f}' for a, bb in g['fades']),
                        c['class'] if c else '', _r(c['gap_mm'], 2) if c else '',
                        _r(c['incidence_deg'], 1) if c else '',
                        _r(c['image_db'], 2) if c else '',
                        sum(1 for x in c['ghosts'] if x['in_field'] and x['listed'] and x['resolved']) if c else ''])
    return buf.getvalue()


GHOST_COLUMNS = ('tag', 'kind', 'order_or_class', 'target_az_deg', 'target_el_deg', 'range_m', 'ghost_az_deg',
                 'ghost_el_deg', 'angle_offset_deg', 'range_offset_m', 'level_db', 'in_field', 'resolved')


def ghosts_csv(res: ImagingRadarResult) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(GHOST_COLUMNS)
    for r in res.radars:
        for g in all_ghosts(r):
            w.writerow([r['tag'], g['kind'], g.get('order', g.get('class', '')), _r(g['target_az_deg'], 2),
                        _r(g['target_el_deg'], 2), _r(g.get('range_m'), 1), _r(g['ghost_az_deg'], 2),
                        _r(g['ghost_el_deg'], 2), _r(g['angle_offset_deg'], 3), _r(g['range_offset_m'], 4),
                        _r(g['level_db'], 2), int(g['in_field']), g['resolved']])
    return buf.getvalue()


def all_ghosts(r) -> list:
    """The listed ghosts of a radar (cavity and ground), strongest first."""
    out = [g for g in (r['cavity']['ghosts'] if r['cavity'] else []) if g['listed']]
    for c in r['ground']['classes'].values():
        out += [g for g in c['ghosts'] if g['listed']]
    return sorted(out, key=lambda g: -g['level_db'])


def advice(r) -> list:
    """Mounting advice for one radar as plain sentences."""
    out = []
    c = r['cavity']
    if c is not None:
        ga = c['gap_advice']
        if ga is not None and abs(ga['move_mm']) > 0.05:
            out.append(f'Gap: {ga["gap_mm"]:.1f} mm to the fascia; the cavity\'s anti-resonance is at '
                       f'{ga["best_gap_mm"]:.1f} mm (move {ga["move_mm"]:+.1f} mm): build-up {ga["buildup_db"]:+.1f} → '
                       f'{ga["best_buildup_db"]:+.1f} dB (ripple over half a wavelength {ga["ripple_db"]:.1f} dB).')
        bt = c['best_tilt']
        plane = 'up or down (pitch)' if c['tilt_plane'] == 'vertical' else 'sideways (yaw)'
        if abs(bt['tilt_deg'] - c['incidence_deg']) >= 1.0:
            out.append(f'Tilt: the fascia meets the radar at {c["incidence_deg"]:.0f}°; at {bt["tilt_deg"]:.0f}° '
                       f'(tilt {plane} by {bt["tilt_deg"] - c["incidence_deg"]:+.0f}° against the fascia, then aim '
                       f'the radar again) the first image is {bt["ghost_level_db"]:.1f} dB and '
                       + ('merges with the target' if bt['merged'] else 'inside the field' if bt['ghost_in_field']
                          else 'outside the field') + f', for {bt["loss_db"]:.1f} dB two-way loss.')
        else:
            out.append(f'Tilt: {c["incidence_deg"]:.0f}° to the fascia is already the best within the loss limit.')
        th = c['best_thickness']
        if th is not None and abs(th['thickness_mm'] - c['thickness_mm']) > 0.05:
            out.append(f'Fascia: {c["thickness_mm"]:.2f} mm; {th["thickness_mm"]:.2f} mm reflects least at this '
                       f'incidence ({th["reflection_db"]:.1f} dB, two-way loss {th["loss_db"]:.1f} dB).')
    ps = [x for x in r['pitch_sweep'] if math.isfinite(x['continuous_m'])]
    if ps and r['weakest']:
        now = next((x for x in ps if abs(x['pitch_deg'] - r['pose']['pitch']) < 1e-6), None)
        best = max(ps, key=lambda x: (x['continuous_m'], -abs(x['pitch_deg'] - r['pose']['pitch'])))
        if now is not None and best['continuous_m'] > 1.05 * now['continuous_m']:
            out.append(f'Pitch: at {best["pitch_deg"]:+.0f}° (now {now["pitch_deg"]:+.0f}°, positive down) '
                       f'{r["weakest"].lower()}s are detected without a gap to {best["continuous_m"]:.0f} m '
                       f'(now {now["continuous_m"]:.0f} m).')
        elif now is not None:
            out.append(f'Pitch: {now["pitch_deg"]:+.0f}° keeps {r["weakest"].lower()}s without a gap to '
                       f'{now["continuous_m"]:.0f} m; ±4° does not do better.')
    g = r['ground']
    if r['weakest'] in g['classes']:
        w = g['classes'][r['weakest']]
        if math.isfinite(w['first_null_m']):
            out.append(f'Height: {g["radar_height_m"]:.2f} m up, the farthest two-ray null for '
                       f'{r["weakest"].lower()}s is at about {w["first_null_m"]:.0f} m (2 h_r h_t / λ); mounting lower '
                       f'brings it nearer, higher pushes it out.')
    return out


# =========================================================================== the 3D lobe
def lobe_mesh(sensor, model, settings: ImagingRadarSettings | None = None, scale_m: float = 2.0,
              floor_db: float = -40.0, n_az: int = 81, n_el: int = 61):
    """The radar's one-way beam (array factor steered at the boresight × element pattern) as a closed surface in the
    vehicle frame: (triangles (K, 3, 3), dB at each triangle (K,)); the radius is scale_m × (dB − floor) / −floor.
    Sampled denser near the boresight (az = 90 t², el = 80 t² in sign), so a narrow beam keeps its shape. None for a
    model without an array."""
    st = settings or ImagingRadarSettings()
    arr = array_spec(model)
    if arr is None:
        return None
    p = model.params or {}
    zones = [z for z in (model.zones or []) if z[0] > 0]
    az3 = _num(p.get('element_az_3db_deg'), max((z[1] for z in zones), default=45.0) / math.sqrt(2))
    el3 = _num(p.get('element_el_3db_deg'), max((z[2] for z in zones), default=10.0) / math.sqrt(2))
    U, _ = arr.unique()
    bf = beamformer(U, st.window, st.taylor_sll_db, st.taylor_nbar)
    ta = np.linspace(-1, 1, n_az)
    te = np.linspace(-1, 1, n_el)
    az = 90.0 * np.sign(ta) * ta ** 2
    el = 80.0 * np.sign(te) * te ** 2
    A = db_pow(bf.azel(az, el)) + element_db(az[None, :], el[:, None], az3, el3)
    rad = scale_m * np.clip((A - floor_db) / -floor_db, 0.0, 1.0)
    D = _dir(az[None, :], el[:, None])                             # (ne, na, 3), radar frame
    P = D * rad[..., None]
    Rm = S.rotation(sensor)
    W = P @ Rm.T + sensor.pos()
    a, b, c, d = W[:-1, :-1], W[:-1, 1:], W[1:, 1:], W[1:, :-1]
    tri = np.concatenate([np.stack([a, b, c], -2).reshape(-1, 3, 3), np.stack([a, c, d], -2).reshape(-1, 3, 3)])
    v = 0.25 * (A[:-1, :-1] + A[:-1, 1:] + A[1:, 1:] + A[1:, :-1])
    val = np.concatenate([v.ravel(), v.ravel()])
    keep = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) > 1e-12
    return tri[keep], val[keep]


# =========================================================================== PNG
def _ramp(x):
    """dB (−40 … 0) to an RGB sequential ramp (dark to light)."""
    cols = np.array([[13, 54, 107], [37, 106, 191], [57, 135, 229], [109, 167, 236], [158, 197, 244],
                     [198, 220, 244], [240, 244, 250]], float)
    t = np.clip((np.asarray(x, float) + 40.0) / 40.0, 0, 1) * (len(cols) - 1)
    i0 = np.floor(t).astype(int)
    i1 = np.minimum(i0 + 1, len(cols) - 1)
    f = (t - i0)[..., None]
    return cols[i0] * (1 - f) + cols[i1] * f


def png(res: ImagingRadarResult, width=1500) -> bytes:
    """Per radar: the azimuth and elevation cuts (taper and uniform), the beam map with the field, and the detection
    margin against range per class over the road (fades where it dips below 0 dB)."""
    import cv2
    from .pngtext import ascii_text
    font, aa = cv2.FONT_HERSHEY_SIMPLEX, cv2.LINE_AA
    bg, ink, ink2, muted, hair = (30, 29, 28), (236, 235, 230), (200, 199, 192), (140, 140, 135), (70, 70, 66)
    orange = (57, 135, 229)                               # BGR
    cls_col = [(229, 135, 57), (90, 180, 240), (120, 200, 120), (220, 120, 200)]
    row_h = 260
    H = 70 + max(len(res.radars), 1) * row_h + 20
    img = np.full((H, width, 3), bg, np.uint8)

    def text(t, x, y, col=ink, s=0.45):
        cv2.putText(img, ascii_text(t), (int(x), int(y)), font, s, col, 1, aa)
    text(f'Imaging radar: array, ground and fascia  {res.variant}', 20, 30, ink, 0.65)
    text(f'{WINDOW_LABEL[res.settings.window]} taper; road {res.settings.road}; dB power. {res.swap_note}', 20, 54,
         ink2, 0.42)

    def plot(x0, y0, w, h, xs, series, xlim, ylim, xlabel, logx=False):
        cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), hair, 1)
        tx = (lambda v: np.log10(v)) if logx else (lambda v: v)
        a, b = tx(xlim[0]), tx(xlim[1])
        for ys, col in series:
            m = np.isfinite(ys) & (xs >= xlim[0]) & (xs <= xlim[1])
            if m.sum() < 2:
                continue
            px = x0 + (tx(xs[m]) - a) / (b - a) * w
            py = y0 + h - (np.clip(ys[m], ylim[0], ylim[1]) - ylim[0]) / (ylim[1] - ylim[0]) * h
            pts = np.c_[px, py].astype(np.int32)
            cv2.polylines(img, [pts], False, col, 1, aa)
        text(xlabel, x0, y0 + h + 14, muted, 0.33)
        text(f'{xlim[0]:.3g}', x0, y0 + h + 26, muted, 0.3)
        text(f'{xlim[1]:.3g}', x0 + w - 30, y0 + h + 26, muted, 0.3)
        text(f'{ylim[1]:g}', x0 - 28, y0 + 8, muted, 0.3)
        text(f'{ylim[0]:g}', x0 - 28, y0 + h, muted, 0.3)
    for k, r in enumerate(res.radars):
        y0 = 70 + k * row_h
        b = r['beam']
        head = f'{r["tag"]}  {r["model"]}' + ('  (swapped)' if r['swapped'] else '') + f'  mount {r["mount"]}'
        text(head, 20, y0 + 14, ink, 0.48)
        if b is not None:
            ch = (f'{b["n_virtual"]} channels ({b["n_unique"]} unique)' if b['layout'] == 'public' else
                  f'ideal {b["n_az"]} x {b["n_el"]} aperture (assumed; {b["n_virtual_stated"] or "?"} channels stated)')
            sub = (f'{ch}; res {b["az_res_deg"]:.2f} x '
                   + (f'{b["el_res_deg"]:.2f}' if math.isfinite(b['el_res_deg']) else '-') +
                   f' deg; PSL {b["psl_az_db"]:.1f} dB; range res {b["range_res_m"]:.2f} m')
            text(sub, 20, y0 + 32, ink2, 0.38)
            cu = b['cuts']
            half = min(max(8 * b['az_res_deg'], 10.0), 90.0)
            plot(50, y0 + 45, 300, 160, cu['deg'], [(cu['az_uniform'], muted), (cu['az'], orange)],
                 (-half, half), (-50, 0), 'azimuth cut, deg (orange: taper; grey: uniform)')
            eh = 90.0 if not math.isfinite(b['el_res_deg']) else min(max(8 * b['el_res_deg'], 10.0), 90.0)
            plot(400, y0 + 45, 300, 160, cu['deg'], [(cu['el_uniform'], muted), (cu['el'], orange)],
                 (-eh, eh), (-50, 0), 'elevation cut, deg')
            mp = b.get('map')
            if mp is not None:
                ha = min(b['fov_az_deg'] + 15.0, 90.0)
                he = min(b['fov_el_deg'] + 15.0, 60.0)
                ia = (mp['az'] >= -ha) & (mp['az'] <= ha)
                ie = (mp['el'] >= -he) & (mp['el'] <= he)
                azs, els = mp['az'][ia], mp['el'][ie]
                tile = _ramp(mp['db'][np.ix_(ie, ia)][::-1, ::-1]).astype(np.uint8)
                tile = cv2.resize(tile, (300, 200), interpolation=cv2.INTER_AREA)[:, :, ::-1]
                img[y0 + 30:y0 + 230, 740:1040] = tile
                fx = lambda a: int(740 + (azs[-1] - a) / (azs[-1] - azs[0]) * 300)       # noqa: E731
                fy = lambda e: int(y0 + 30 + (els[-1] - e) / (els[-1] - els[0]) * 200)   # noqa: E731
                cv2.rectangle(img, (fx(b['fov_az_deg']), fy(b['fov_el_deg'])), (fx(-b['fov_az_deg']),
                              fy(-b['fov_el_deg'])), ink, 1)
                text(f'beam x element, -40..0 dB, +-{ha:.0f} az x +-{he:.0f} el (box: field)', 740, y0 + 246, muted,
                     0.33)
        else:
            text('zone model: no stated array', 20, y0 + 32, ink2, 0.38)
        gc = r['ground']['classes']
        series = []
        for i, (name, g) in enumerate(gc.items()):
            series.append((g['curve']['margin_db'].astype(float), cls_col[i % len(cls_col)]))
        if gc:
            d = next(iter(gc.values()))['curve']['d']
            plot(1100, y0 + 45, 360, 160, d, series, (1.0, res.settings.max_range_m), (-20, 80),
                 'detection margin dB vs range m (log); ' + ', '.join(gc), logx=True)
            zy = y0 + 45 + 160 - (0 + 20) / 100 * 160
            cv2.line(img, (1100, int(zy)), (1460, int(zy)), ink2, 1)
    ok, buf = cv2.imencode('.png', img)
    return buf.tobytes() if ok else b''


def files(res: ImagingRadarResult, project=None) -> list:
    """[(name, bytes)]: imaging_radar.json, imaging_radar.csv, imaging_radar_ghosts.csv, imaging_radar.png."""
    return [('imaging_radar.json', (json.dumps(to_doc(res, project), indent=1, default=_json_default) + '\n')
             .encode('utf-8')),
            ('imaging_radar.csv', to_csv(res).encode('utf-8')),
            ('imaging_radar_ghosts.csv', ghosts_csv(res).encode('utf-8')),
            ('imaging_radar.png', png(res))]


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


def export(res: ImagingRadarResult, project, out_dir) -> list:
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for name, data in files(res, project):
        path = os.path.join(out_dir, name)
        with open(path, 'wb') as fh:
            fh.write(data)
        paths.append(path)
    return paths


# =========================================================================== issues (cached)
_CACHE = OrderedDict()
_LOCK = threading.RLock()


def cache_key(project) -> str | None:
    """Everything a cheap run reads: the radars, their models, the settings, the vehicle's geometry and the
    materials settings."""
    try:
        v = project.variant
        sens = [s.to_dict() for s in v.sensors if s.kind == 'radar' and s.visible]
        if not sens:
            return f'none:{v.uid}'                # a variant without radars: still its own (the page names it)
        st = settings_of(project)
        mids = {s['model_id'] for s in sens} | ({st.swap_model} if st.swap_model else set())
        mods = {m: project.sensor_models[m].to_dict() for m in sorted(mids) if m in project.sensor_models}
        mesh = project.vehicle.mesh
        geo = (id(mesh), getattr(mesh, 'n_tris', 0)) if mesh is not None else None
        rig = [r.to_dict() if hasattr(r, 'to_dict') else str(r) for r in getattr(project, 'rig', ())]
        return json.dumps([sens, mods, st.to_dict(), getattr(project.analysis, 'materials', None), str(geo), rig],
                          sort_keys=True, default=str)
    except Exception:                                      # noqa: BLE001
        return None


def cached_run(project) -> ImagingRadarResult | None:
    key = cache_key(project)
    with _LOCK:
        if key is not None and key in _CACHE:
            _CACHE.move_to_end(key)
            return _CACHE[key]
    res = run(project, maps=False, advise=False)
    if key is not None:
        with _LOCK:
            _CACHE[key] = res
            while len(_CACHE) > 6:
                _CACHE.popitem(last=False)
    return res


ISSUE_KINDS = ('grating', 'cavity_ghost', 'fade', 'near_blind')


def issues(project, vg=None) -> list:
    """Grating lobes, cavity ghosts inside the field, fades and the near blind zone of the weakest class, for the
    radars with an array model (ranging 'mimo', or a zone model with a known array). Cached on the inputs."""
    from .rules import Issue
    v = project.variant
    sens = [s for s in v.sensors if s.kind == 'radar' and s.visible]
    if not sens:
        return []
    arr = {s.uid for s in sens if (lambda m: m.ranging == 'mimo' or m.id in ZONE_TO_ARRAY)(project.sensor_model_for(s))}
    if not arr:
        return []
    try:
        res = cached_run(project)
    except Exception as e:                                 # noqa: BLE001
        return [Issue('warning', f'The imaging radar check could not run ({type(e).__name__}: {e}).',
                      f'{v.name} · imaging radar')]
    out = []
    for f in res.findings if res is not None else []:
        if f['severity'] == 'good' or f['kind'] not in ISSUE_KINDS or not set(f['uids']) & arr:
            continue
        o = v.sensor(f['uids'][0]) if f['uids'] else None
        out.append(Issue(f['severity'], f['message'], ', '.join(f['tags']), list(f['uids']),
                         action='imaging_radar', where=(o.x, o.y) if o is not None else None))
    return out


def clear_cache():
    with _LOCK:
        _CACHE.clear()


# =========================================================================== validation (for vantage.validation)
def _outcome(error, detail=None, ok=True, figure=None, note=''):
    from ..validation.suite import Outcome
    return Outcome(float(error), detail or {}, figure, ok, None, note)


def _check_ula(ctx):
    """A uniform linear array's −3 dB beamwidth against 0.886 λ/(N d); the uniform, Hann and Taylor sidelobes."""
    worst, rows = 0.0, []
    for N in (8, 16, 32, 64, 128):
        U = np.c_[np.arange(N, dtype=float), np.zeros(N)]
        th = np.linspace(-1, 1, 8001) * 2 * math.degrees(UNIFORM_K * 2 / N)
        p = db_pow(af_cut(U, np.ones(N), u=np.sin(np.radians(th))))
        hp = lobe(th, p)['hpbw_deg']
        ref = math.degrees(UNIFORM_K * 2 / N)                 # 0.886 λ / (N λ/2)
        e = abs(hp - ref) / ref
        rows.append({'N': N, 'hpbw_deg': round(hp, 5), 'closed_form_deg': round(ref, 5), 'rel_error': e})
        worst = max(worst, e)
    N = 128
    U = np.c_[np.arange(N, dtype=float), np.zeros(N)]
    th = np.linspace(-1, 1, 12001) * 12 * math.degrees(UNIFORM_K * 2 / N)
    sv = np.sin(np.radians(th))
    side = {}
    for kind, ref, bw in (('uniform', -13.26, 0.886), ('hann', -31.47, 1.44), ('taylor', -30.0, None)):
        w = weights(U, kind, 30.0, 4)
        L = lobe(th, db_pow(af_cut(U, w, u=sv)))
        side[kind] = {'psl_db': round(L['psl_db'], 3), 'reference_db': ref,
                      'hpbw_bins': round(math.radians(L['hpbw_deg']) * N / 2, 4), 'reference_bins': bw}
    side_ok = (abs(side['uniform']['psl_db'] + 13.26) < 0.05 and abs(side['hann']['psl_db'] + 31.47) < 0.3
               and abs(side['taylor']['psl_db'] + 30.0) < 0.5 and abs(side['hann']['hpbw_bins'] - 1.44) < 0.01)
    return _outcome(worst, {'beamwidth': rows, 'tapers_128': side}, ok=side_ok)


def _check_taylor(ctx):
    """The Taylor taper against scipy.signal.windows.taylor (an independent implementation)."""
    from scipy.signal.windows import taylor as sp_taylor
    worst = 0.0
    for M_, nbar, sll in ((16, 4, 30), (64, 5, 35), (86, 4, 30), (128, 8, 45)):
        x = (np.arange(M_) - M_ / 2 + 0.5) / M_
        a = taper(x, 'taylor', sll, nbar)
        b = sp_taylor(M_, nbar=nbar, sll=sll, norm=True)
        worst = max(worst, float(np.max(np.abs(a / a.max() - b / b.max()))))
    return _outcome(worst, {'cases': 4})


def _check_slab(ctx):
    """The fascia slab model (materials.stack_rt) against the Fresnel equations: a thick lossy slab reflects as one
    interface; a lossless slab as the Airy sum of its Fresnel coefficients."""
    from . import materials as M
    f = 77.0
    lam = wavelength_m(f)
    worst, rows = 0.0, []
    for eps, tan, theta in ((2.75, 0.2, 0.0), (8.6, 0.2, 40.0), (4.46, 0.3, 70.0)):
        r_te, _, r_tm, _ = M.stack_rt([(eps, tan, 0.3)], f, np.array([theta]))
        f_te, f_tm = M.fresnel_halfspace(eps * (1 - 1j * tan), theta)
        e = max(abs(abs(r_te[0]) ** 2 - abs(f_te) ** 2), abs(abs(r_tm[0]) ** 2 - abs(f_tm) ** 2))
        rows.append({'case': 'thick lossy', 'eps': eps, 'tan_delta': tan, 'theta': theta, 'difference': e})
        worst = max(worst, e)
    for eps, d, theta in ((2.75, 3.0e-3, 0.0), (2.75, 3.0e-3, 30.0), (3.2, 1.7e-3, 55.0)):
        n = math.sqrt(eps)
        st_ = math.sin(math.radians(theta))
        ct = math.cos(math.radians(theta))
        ct2 = math.sqrt(1 - (st_ / n) ** 2)
        delta = 2 * math.pi / lam * n * d * ct2
        f_te, f_tm = M.fresnel_halfspace(eps, theta)
        r_te, _, r_tm, _ = M.stack_rt([(eps, 0.0, d)], f, np.array([theta]))
        for r12, rm in ((complex(f_te), r_te[0]), (complex(f_tm), r_tm[0])):
            airy = r12 * (1 - np.exp(-2j * delta)) / (1 - r12 ** 2 * np.exp(-2j * delta))
            e = abs(abs(airy) ** 2 - abs(rm) ** 2)
            rows.append({'case': 'lossless Airy', 'eps': eps, 'mm': d * 1000, 'theta': theta, 'difference': e})
            worst = max(worst, e)
        del ct
    return _outcome(worst, {'cases': rows})


def _check_two_ray(ctx):
    """two_ray against the closed form 2|sin(k (R_r − R_d) / 2)| for Γ = −1 (exact paths, no spreading), its
    asymptote 2|sin(2π h_r h_t / (λ d))| far out, and the nulls at d_n = 2 h_r h_t / (n λ), n = 1, 2, 3, 5."""
    f, hr, ht = 77.0, 0.5, 1.0
    lam = wavelength_m(f)
    d = np.geomspace(2.0, 2000.0, 4001)
    F, Rd, Rr, _ = two_ray(d, hr, ht, f, -1.0, spread=False)
    ex = 2 * np.abs(np.sin(math.pi / lam * (np.hypot(d, ht + hr) - np.hypot(d, ht - hr))))
    e_exact = float(np.max(np.abs(F - ex)))
    far = d >= 500.0
    asym = 2 * np.abs(np.sin(2 * math.pi * hr * ht / (lam * d[far])))
    e_asym = float(np.max(np.abs(F[far] - asym)))
    nulls = []
    for n in (1, 2, 3, 5):
        dn = 2 * hr * ht / (n * lam)
        Fn, *_ = two_ray(np.array([dn]), hr, ht, f, -1.0, spread=False)
        nulls.append({'n': n, 'd_m': round(dn, 3), 'F': float(Fn[0])})
    return _outcome(e_exact, {'asymptote_max_error': e_asym, 'nulls': nulls},
                    ok=e_asym < 2e-3 and all(x['F'] < 0.01 for x in nulls))


def _check_virtual_array(ctx):
    """The TI cascade's virtual array against the design guide: 86 unique azimuth elements (4 devices), 23 (2
    devices), the elevation co-array 0…6 (7 antennas); error = the number of mismatches."""
    m4 = virtual_array(CAS_TX, CAS_RX)
    az4 = np.unique(m4[m4[:, 1] == 0, 0])
    m2 = virtual_array([CAS_TX[i - 1] for i in CAS2_TX_IDS], [CAS_RX[i - 1] for i in CAS2_RX_IDS])
    az2 = np.unique(m2[m2[:, 1] == 0, 0])
    rows = np.unique(m4[:, 1])
    co = np.unique(np.abs(rows[:, None] - rows[None, :]))
    mism = int(len(az4) != 86) + int(len(az2) != 23) + int(len(co) != 7) + int(len(m4) != 192)
    U, _ = unique_positions(m4)
    hp = lobe(np.linspace(-10, 10, 40001), db_pow(af_cut(U, np.ones(len(U)), u=np.sin(np.radians(
        np.linspace(-10, 10, 40001))))))['hpbw_deg']
    return _outcome(mism, {'az_elements_4dev': int(len(az4)), 'az_elements_2dev': int(len(az2)),
                           'elevation_coarray': co.tolist(), 'channels': int(len(m4)),
                           'uniform_hpbw_deg': round(hp, 3), 'rayleigh_2_over_N_deg': round(math.degrees(2 / 86), 3),
                           'stated_deg': 1.4})


def validation_checks() -> list:
    """Spec-compatible dicts for vantage.validation (the lead registers them; model id 'imaging_radar')."""
    return [
        {'id': 'imaging_radar.ula_beamwidth', 'model': 'imaging_radar',
         'title': 'Uniform linear array: beamwidth and taper sidelobes',
         'quantity': 'largest relative difference of the −3 dB beamwidth of uniform arrays of 8–128 elements at λ/2 '
                     'from 0.886 λ / (N d); the 128-element uniform, Hann and Taylor (30 dB, n̄ 4) highest sidelobes '
                     'must be within 0.05, 0.3 and 0.5 dB of −13.26, −31.47 and −30 dB and the Hann beamwidth within '
                     '0.01 bins of 1.44',
         'reference': 'the closed form for a uniform aperture (Balanis §6.3) and Harris\'s window table',
         'tolerance': 0.01, 'unit': 'relative', 'refs': ('balanis2016', 'harris1978', 'doerry2017'),
         'fn': _check_ula, 'cost': 'cheap'},
        {'id': 'imaging_radar.taylor', 'model': 'imaging_radar', 'title': 'The Taylor taper',
         'quantity': 'largest difference of imaging_radar.taper(\'taylor\') from scipy.signal.windows.taylor for '
                     '16–128 points, n̄ 4–8, 30–45 dB',
         'reference': 'SciPy\'s independent implementation of the Taylor window', 'tolerance': 1e-9,
         'unit': 'amplitude', 'refs': ('doerry2017',), 'fn': _check_taylor, 'cost': 'cheap'},
        {'id': 'imaging_radar.slab_fresnel', 'model': 'imaging_radar',
         'title': 'The fascia slab against the Fresnel equations',
         'quantity': 'largest difference of the power reflectance of materials.stack_rt from the Fresnel interface '
                     '(a 0.3 m lossy slab, whose back face returns nothing) and from the Airy sum of the Fresnel '
                     'coefficients (lossless 1.7–3 mm slabs), TE and TM, 0–70°',
         'reference': 'Fresnel\'s equations and the Airy summation written out in the check (Born & Wolf §1.5–1.6)',
         'tolerance': 1e-9, 'unit': 'power ratio', 'refs': ('bornwolf1999', 'macleod2010'), 'fn': _check_slab,
         'cost': 'cheap'},
        {'id': 'imaging_radar.two_ray', 'model': 'imaging_radar', 'title': 'Two-ray ground multipath',
         'quantity': 'largest difference of imaging_radar.two_ray (Γ = −1, no spreading) from 2|sin(k(R_r − R_d)/2)| '
                     'for a radar 0.5 m and a target 1 m up, 2–2000 m; beyond 500 m within 2e-3 of the asymptote '
                     '2|sin(2π h_r h_t / (λ d))|; at the nulls d_n = 2 h_r h_t / (n λ), n = 1, 2, 3, 5, below 0.01',
         'reference': 'the two-ray closed form (flat earth) written in the check', 'tolerance': 1e-9,
         'unit': 'field ratio', 'refs': ('skolnik2001',), 'fn': _check_two_ray, 'cost': 'cheap'},
        {'id': 'imaging_radar.ti_cascade', 'model': 'imaging_radar',
         'title': 'The TI cascade\'s virtual array against its design guide',
         'quantity': 'mismatches of the virtual array (12 Tx × 16 Rx = 192 channels, 86 unique azimuth elements, 23 '
                     'in the 2-device mode, an elevation co-array of 7)',
         'reference': 'TI TIDUEN5A §3.1, Figures 7 and 8', 'tolerance': 0.0, 'unit': 'count',
         'refs': ('ti_tiduen5', 'ti_swru553'), 'fn': _check_virtual_array, 'cost': 'cheap'},
    ]


def validation_references() -> dict:
    """{key: citation} the checks and the docs cite (merged into vantage.validation.references)."""
    return dict(REFERENCES)


VALIDATION_MODEL = ('imaging_radar', '4D imaging radar: array, ghosts and ground', 'vantage.core.imaging_radar',
                    'imaging-radar')
