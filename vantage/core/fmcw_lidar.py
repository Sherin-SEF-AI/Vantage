"""Coherent FMCW ("4D") lidar: the radial velocity of every return, the ego motion and the moving points it gives from
one sweep, and what sunlight and other lidars do to it against a pulsed time-of-flight lidar.

The library
-----------
library() adds the FMCW lidars whose makers publish figures (SensorModel kind 'lidar', ranging 'fmcw'): Aeva Aeries II
and Atlas, Voyant Carbon 30 and Carbon 60. The FMCW figures live in SensorModel.params (PARAM_KEYS); a figure the
maker does not state is absent, and the model fields every lidar must have (a scan grid, a frame rate, a size) that a
maker does not state are named in params['assumed'] and in the note. SiLC (Eyeonic) and Aurora (FirstLight) publish no
datasheet figures, so they are not in the library.

Doppler per point
-----------------
An FMCW lidar measures, with the range, the rate of change of the range along each ray: the radial velocity
    v_r = d · (v_p − v_s)
with d the unit ray from the sensor to the point, v_p the point's velocity over the ground and v_s the sensor's (both
in the vehicle's axes). v_r > 0 means the range grows (the point recedes), v_r < 0 that it closes. A rigid object moving
at v with yaw rate ω moves its point p at v + ω ẑ × (p − c). The sensor moves at v_s = v_ego + ω_ego ẑ × r_s. The
measurement adds Gaussian noise of the datasheet's velocity precision (1 σ), and a return beyond the stated velocity
range carries no velocity (NaN).

  * Ego motion from one sweep: every static point obeys v_r = −d · v_s, so v_s is the least-squares solution over the
    static points; RANSAC on three-point solutions keeps the moving points out (fit_sensor_velocity). With the vehicle
    origin at the rear axle and no side slip there, v_s = (v_x − ω y_s, ω x_s, 0) gives the forward speed and the yaw
    rate from one lidar placed off the rear axle (ego_from_sensor_velocity).
  * Moving points from one sweep: the residual r = v_r + d · v̂_s is the point's own velocity along the ray; a point is
    moving when |r| exceeds the threshold. A mover whose motion is across the ray (a car crossing far ahead at the edge
    of the field, say) has r ≈ 0: Doppler alone cannot see it move, and the study counts those points apart
    ('across the ray').

Scenes: a scripted scene (SCENES: urban and highway traffic around the car, roadside walls, the project's obstacles,
the body blocking the beams when an engine is given) or a CARLA recording with ground truth (load_recording): the
velocities are CARLA's own (the bridge writes every actor's get_velocity() to /ground_truth/boxes and the ego's twist to
/odom in its ROS 2 bag); without a bag they come from the boxes differenced over frames (track_velocity). Points take
their object from the nuScenes-panoptic labels when the drive has them, else from the boxes they fall in. CARLA renders
no Doppler lidar: the 4D cloud of a recording is CARLA's points with the radial velocity computed from CARLA's own
velocities plus the datasheet noise, so the ego-velocity error and the moving-point recall measure the estimators on
real geometry and traffic, not a sensor's own velocity errors.

Sunlight and cross-talk
-----------------------
  * Solar spectral irradiance: ASTM G173-03 (extraterrestrial, direct normal at air mass 1.5, global on the 37° tilt),
    the rows around 905 nm and 1550 nm as pvlib ships them (G173_ROWS), averaged over the receiver's filter. Direct
    normal at another sun elevation by Beer–Lambert from the AM1.5 transmittance with Kasten & Young's (1989) air mass;
    diffuse scaled with the sine of the sun's elevation (a first-order approximation).
  * Time of flight (905 nm, direct detection): a surface of reflectivity ρ in a pixel of solid angle Ω behind an
    aperture A and a filter Δλ puts N_b = η ρ E_λ Δλ A Ω λ / (π h c) background photoelectrons per second on the
    detector (Lambertian radiance ρE/π); the sun's disc inside a pixel puts η E_dn,λ Δλ A min(1, Ω/Ω_sun) λ/(hc). The
    noise per range bin of τ is σ² = (N_b + DCR)·τ for a photon-counting SPAD receiver (most current 905 nm lidars;
    DCR its dark counts), σ² = F·N_b·τ + σ_amp² for a linear-mode APD (excess noise F, plus the amplifier); Gaussian
    approximation of the counts.
    The range on a 10 % target is that of a threshold following each beam's own noise (constant false-alarm rate),
    R = R_ref (σ_ref/σ)^½, the datasheet range R_ref holding at its stated sunlight (100 klx, on a target facing the
    sun, where the note says so) or else in the dark. False returns come from a threshold that tracks the ambient light
    over a window of its channel's sweep (st.tof_adapt_deg of azimuth, not shot by shot): a beam brighter than its
    window's median (a sunlit edge, the sun's disc) fires, P = 1 − (1 − Q(z σ_med/σ))^M over the M bins of a shot, z
    set for st.p_false_ref per point.
  * FMCW (1550 nm, coherent): the beat with the local oscillator only picks up light in the oscillator's spatial and
    temporal mode, so a thermal source adds n̄ = L_λ λ⁵ / (2 h c²) photons per mode to the oscillator's one photon of
    shot noise (Siegman 1966, the antenna theorem). The SNR falls by 1 + η n̄: a sunlit 10 % target gives n̄ ≈ 10⁻⁶ (no
    range loss); the sun's disc itself n̄ ≈ 0.3 (the few beams on the disc lose ≈ 6 % of their range). False alarms of
    a coherent detection threshold scale as P_fa^(1/(1+η n̄)) (the exponential tail of the beat's noise power).
  * Cross-talk: another time-of-flight lidar in mutual view puts a share of interference points into a frame: 1.5 % in
    the worst case measured between two Velodyne HDL-64E (Diehm et al. 2018, as cited by Robinson et al. 2025), 1.04 %
    between SICK LMS-531 (Popko et al. 2020). The study takes the cited worst case per interferer in mutual view (both
    within the other's field and range): an upper bound, not a prediction for a given pair. An FMCW receiver rejects
    pulsed light (incoherent with its oscillator, and at another wavelength) and another FMCW lidar's light unless its
    chirp overlaps in both time and optical frequency (Behroozpour et al. 2017): no interference points.

All receiver figures (aperture, filter, pixel field, efficiencies, amplifier noise, bin) are typical values, settings
the user can replace with their unit's; in direct sun the time-of-flight range ratio depends only on the irradiance
ratio (background-limited), at night on the dark counts or the amplifier noise. Not modelled: saturation and stray
light (a beam near the sun's disc is treated as on it or off it), multipath, atmospheric attenuation of the signal, the
lidar's own range noise, interference between FMCW lidars whose chirps do overlap, and the rolling time within a sweep
of a scripted scene.
"""
from __future__ import annotations

import csv
import io
import json
import math
import os
import struct
from dataclasses import asdict, dataclass, field

import numpy as np

from .sensors import SensorModel

SCHEMA = 'vantage.fmcw/1'
READ = '2026-11-11'                     # when the FMCW makers' pages were read for the library below
LIB_CHECKED = '2026-11-04'              # core.sensors.CHECKED: the rest of the sensor library's reading
FRAME = 'vehicle (ISO 8855, rear axle)'
SIGN = 'v_r > 0: the range grows (the point recedes); v_r < 0: it closes'

# physical constants (CODATA 2018 exact values; IAU 2015 Resolution B3 nominal solar values)
H = 6.62607015e-34
C = 299792458.0
KB = 1.380649e-23
R_SUN_M = 6.957e8
T_SUN_K = 5772.0
AU_M = 1.495978707e11
SUN_HALF_RAD = math.asin(R_SUN_M / AU_M)              # 4.65 mrad: the disc's angular radius
OMEGA_SUN = 2 * math.pi * (1 - math.cos(SUN_HALF_RAD))  # 6.8e-5 sr
LUX_PER_WM2 = 110.0                     # luminous efficacy of daylight, lm/W (typically 105–120): klx shown for context

REFERENCES = {
    'astm_g173': 'ASTM G173-03, Standard Tables for Reference Solar Spectral Irradiances: Direct Normal and '
                 'Hemispherical on 37° Tilted Surface (SMARTS v2.9.2); the rows used here as pvlib-python ships them, '
                 'pvlib/data/ASTMG173.csv (https://github.com/pvlib/pvlib-python, read 2026-11-11).',
    'kasten1989': 'F. Kasten and A. T. Young, "Revised optical air mass tables and approximation formula", Applied '
                  'Optics 28(22), 4735–4738, 1989.',
    'siegman1966': 'A. E. Siegman, "The antenna properties of optical heterodyne receivers", Applied Optics 5(10), '
                   '1588–1594, 1966.',
    'behroozpour2017': 'B. Behroozpour, P. A. M. Sandborn, M. C. Wu and B. E. Boser, "Lidar system architectures and '
                       'circuits", IEEE Communications Magazine 55(10), 135–142, 2017.',
    'diehm2018': 'A. L. Diehm, M. Hammer, M. Hebel and M. Arens, "Mitigation of crosstalk effects in multi-LiDAR '
                 'configurations", Proc. SPIE 10796, Electro-Optical Remote Sensing XII, 2018 (over 1000 interference '
                 'points, 1.5 %, between two Velodyne HDL-64E; figure as cited by Robinson et al. 2025, §I).',
    'popko2020': 'G. B. Popko, T. K. Gaylord and C. R. Valenta, "Interference measurements between single-beam, '
                 'mechanical scanning, time-of-flight lidars", Optical Engineering 59(5), 053106, 2020 (1.04 % of '
                 'points in the worst case; as cited by Robinson et al. 2025, §I).',
    'robinson2025': 'J. Robinson, M. Lovric, R. Woodman, D. Croft and V. Donzella, "Real-time mitigation of LiDAR '
                    'mutual interference", University of Warwick, WRAP 190972 (CC BY 4.0), 2025, §I and §II.',
    'iau2015': 'IAU 2015 Resolution B3 on recommended nominal conversion constants for selected solar and planetary '
               'properties (R☉ = 6.957 × 10⁸ m, T_eff☉ = 5772 K).',
}

# ASTM G173-03 rows (wavelength nm: extraterrestrial, global tilt 37°, direct + circumsolar; W m⁻² nm⁻¹) from pvlib's
# ASTMG173.csv: 885–925 nm and 1530–1570 nm, every row (1 nm steps)
_G173_TXT = """
885 .95469 .94423 .87913|886 .92558 .90752 .84515|887 .923 .91062 .84799|888 .94066 .92228 .85899
889 .94367 .93455 .87041
890 .94236 .92393 .86078|891 .93928 .92584 .86255|892 .93259 .90881 .84688|893 .918 .87327 .81412|894 .924 .8513 .79413
895 .926 .81357 .75956|896 .93425 .76253 .71265|897 .92686 .66566 .6231|898 .92378 .7178 .67137|899 .91396 .54871 .51461
900 .91378 .7426 .69429|901 .89834 .59933 .56162|902 .8772 .66791 .625|903 .92233 .68889 .64483|904 .921 .84457 .78862
905 .918 .81709 .76337|906 .9078 .77558 .72502|907 .9145 .63854 .59833|908 .90143 .65217 .61091|909 .88552 .70431 .65912
910 .89496 .62467 .58553|911 .8973 .66808 .6258|912 .89016 .68893 .64508|913 .9002 .62834 .58906|914 .8982 .62649 .5874
915 .8883 .67836 .6355|916 .8947 .57646 .54099|917 .8886 .73017 .6835|918 .8881 .59271 .55612|919 .8921 .73877 .69161
920 .8854 .74414 .69657|921 .8703 .78049 .73004|922 .8628 .70026 .65584|923 .83335 .74504 .69698|924 .86957 .7215 .67571
925 .87642 .7111 .66621
1530 .26731 .25522 .24789|1531 .28272 .26973 .262|1532 .28263 .27839 .27039|1533 .28041 .27714 .26918
1534 .27543 .26892 .26122|1535 .27754 .26686 .25924|1536 .28058 .27464 .26679|1537 .28096 .27336 .26558
1538 .27338 .27202 .26427|1539 .27808 .27295 .26519|1540 .26857 .26491 .25737|1541 .27297 .26904 .26141
1542 .27536 .26927 .26165|1543 .27499 .27208 .26439|1544 .27716 .2721 .26443|1545 .27768 .27705 .26922
1546 .27722 .27481 .26705|1547 .27354 .27309 .26537|1548 .26698 .26675 .2592|1549 .27267 .27342 .26568
1550 .26961 .2699 .26226|1551 .27075 .27058 .26293|1552 .274 .27182 .26415|1553 .2709 .27132 .26366
1554 .26495 .26474 .25727|1555 .26712 .26759 .26005|1556 .26292 .2631 .25569|1557 .27013 .27062 .26301
1558 .26805 .26848 .26093|1559 .2678 .26808 .26054|1560 .26546 .26568 .25821|1561 .26968 .27002 .26242
1562 .26833 .26756 .26003|1563 .26688 .26667 .25917|1564 .26307 .26264 .25525|1565 .26682 .26728 .25975
1566 .26262 .26245 .25506|1567 .26352 .26308 .25566|1568 .26042 .25722 .24997|1569 .26275 .25452 .24736
1570 .26068 .24175 .23497
"""
G173_ROWS = np.array([[float(v) for v in row.split()] for line in _G173_TXT.strip().splitlines()
                      for row in line.split('|')])
G173_ZENITH_DEG = 48.19                 # G173's sun: air mass 1.5
G173_TILT_INCIDENCE_DEG = 48.19 - 37.0  # the sun's incidence on G173's 37° tilted surface

# ---------------------------------------------------------------------- the FMCW figures a SensorModel's params hold
PARAM_KEYS = {
    'wavelength_nm': 'laser wavelength, nm',
    'velocity_max_mps': 'largest radial velocity measured, m/s (either sign)',
    'velocity_precision_mps': 'radial-velocity precision, m/s (1 σ)',
    'range_precision_m': 'range precision, m (1 σ)',
    'points_per_s_raw': 'raw point rate the maker states, points/s',
    'range_vehicle_m': 'detection range on vehicles, m',
    'range_pedestrian_m': 'detection range on pedestrians, m',
    'lines_per_frame': 'scan lines per frame (the maker\'s figure)',
    'velocity_text': 'the maker\'s words on velocity when it gives no figure',
    'assumed': 'model fields the maker does not state, filled in as the note says',
}
DEFAULT_WAVELENGTH_NM = 1550.0          # the telecom C band, where FMCW lidars run (when a maker does not state it)
DEFAULT_VEL_SIGMA = 0.03                # m/s, used only when the maker gives no figure (Voyant states 2.8 cm/s)


def _note(url, text, extra='') -> str:
    return f'{text} Source: {url}{extra} (read {READ}, added after the library\'s {LIB_CHECKED} check).'


def library() -> list[SensorModel]:
    """The FMCW lidars with public figures (new instances on every call: a project edits its own copies)."""
    aeva2 = 'https://www.aeva.com/aeries-ii/'
    aeva_pr = ('https://www.businesswire.com/news/home/20220201005588/en (Aeva press release, 1 February 2022: '
               '4 million raw points/s, 120° × 30° maximum field, up to 500 m, up to 1000 lines per frame)')
    atlas = 'https://www.aeva.com/atlas/'
    voy = 'https://voyantphotonics.com/products/'
    voy_r = 'https://www.mybotshop.de/Voyant-Carbon-Lidar_1'
    return [
        SensorModel('aeva_aeries_ii', 'Aeries II', 'lidar', 'Aeva', 'solid', 1000, -15.0, 15.0,
                    h_fov=120.0, h_res=0.3, rate_hz=10.0, range_m=350.0, range_max_m=500.0, range_min_m=0.5,
                    points_per_s=4_000_000, bytes_per_point=16.0, interface='1000BASE-T1', time_sync='PTP',
                    size_mm=(120.0, 160.0, 90.0), optical_z_mm=45.0, power_w=0.0, url=aeva2, builtin=True,
                    ranging='fmcw',
                    params={'points_per_s_raw': 4_000_000.0, 'range_vehicle_m': 500.0, 'range_pedestrian_m': 350.0,
                            'lines_per_frame': 1000.0,
                            'velocity_text': 'per-point velocity "with centimetre per second precision" (no figure)',
                            'assumed': ['rate_hz', 'h_res', 'range_m', 'size_mm', 'optical_z_mm', 'interface',
                                        'time_sync', 'bytes_per_point', 'wavelength_nm', 'velocity_precision_mps']},
                    note=_note(aeva2, 'FMCW 4D lidar (silicon photonics): Aeva states a 120° × 30° maximum field, up '
                                      'to 1000 lines per frame, 4 million raw points/s, detection to 500 m on '
                                      'vehicles and about 350 m on pedestrians, and per-point velocity with '
                                      'centimetre-per-second precision. It gives no range on a 10 % target (its '
                                      'pedestrian figure, 350 m, is used: optimistic for a dark target), no frame '
                                      'rate (10 Hz assumed, so 0.3° between points from 4 M points/s over 1000 lines), '
                                      'no wavelength (1550 nm assumed), no velocity figure (3 cm/s used), no size, '
                                      'power, interface or time sync (a 120 × 160 × 90 mm box, 1000BASE-T1 and PTP '
                                      'are placeholders). Bytes per point (x, y, z, v) assumed 16.',
                               f'; {aeva_pr}')),
        SensorModel('aeva_atlas', 'Atlas', 'lidar', 'Aeva', 'solid', 1000, -15.0, 15.0,
                    h_fov=120.0, h_res=0.3, rate_hz=10.0, range_m=250.0, range_min_m=0.5,
                    points_per_s=0.0, bytes_per_point=16.0, interface='1000BASE-T1', time_sync='PTP',
                    size_mm=(80.0, 110.0, 60.0), optical_z_mm=30.0, power_w=0.0, url=atlas, builtin=True,
                    ranging='fmcw',
                    params={'velocity_text': 'per-point instantaneous velocity (no figure)',
                            'assumed': ['channels', 'h_res', 'rate_hz', 'size_mm', 'optical_z_mm', 'interface',
                                        'time_sync', 'bytes_per_point', 'wavelength_nm', 'velocity_precision_mps']},
                    note=_note(atlas, 'Automotive-grade FMCW 4D lidar: Aeva states up to 250 m on a 10 % target, a '
                                      '120° (h) × 30° (v) field, ISO 26262 ASIL-B, −40 to 85 °C, 70 % smaller than '
                                      'Aeries II. It states no point rate, resolution or frame rate (Aeries II\'s '
                                      'grid, 1000 lines at 0.3° and 10 Hz, is assumed: point-rate figures for this '
                                      'model are not Aeva\'s), no wavelength (1550 nm assumed), no velocity figure '
                                      '(3 cm/s used), no size, power, interface or time sync (placeholders).')),
        SensorModel('voyant_carbon30', 'Carbon 30', 'lidar', 'Voyant Photonics', 'solid', 128, -15.0, 15.0,
                    h_fov=120.0, h_res=120.0 / 763, rate_hz=10.0, range_m=70.0, range_max_m=150.0, range_min_m=0.1,
                    points_per_s=977_000, bytes_per_point=16.0, interface='1000BASE-T', time_sync='PTP',
                    size_mm=(60.0, 110.0, 60.0), optical_z_mm=30.0, power_w=35.0, url=voy, builtin=True,
                    ranging='fmcw',
                    params={'velocity_max_mps': 63.0, 'velocity_precision_mps': 0.028, 'range_precision_m': 0.025,
                            'points_per_s_raw': 977_000.0, 'lines_per_frame': 128.0,
                            'assumed': ['size_mm orientation', 'interface rate', 'bytes_per_point', 'wavelength_nm']},
                    note=_note(voy, 'FMCW lidar on a chip: Voyant states a 30° (v) × 120° (h) field, 128 lines, up '
                                    'to 977,000 points/s, 150 m maximum range and radial velocities up to 63 m/s. A '
                                    'reseller\'s sheet (mybotshop.de) adds 70 m on a 10 % target, 2.8 cm/s velocity '
                                    'and 2.5 cm range precision, 10 frames/s, under 35 W (used as 35 W), under 300 g, '
                                    '110 × 60 × 60 mm, M12 X-code with PoE++ (802.3bt) and gPTP (802.1AS); its field '
                                    'is 90° wide where Voyant\'s page says 120°. The link rate is not stated '
                                    '(1000BASE-T assumed), nor the wavelength (1550 nm assumed). 0.157° between points '
                                    'follows from 977,000 points/s over 128 lines at 10 Hz.', f'; {voy_r}')),
        SensorModel('voyant_carbon60', 'Carbon 60', 'lidar', 'Voyant Photonics', 'solid', 128, -30.0, 30.0,
                    h_fov=60.0, h_res=60.0 / 763, rate_hz=10.0, range_m=30.0, range_max_m=75.0, range_min_m=0.1,
                    points_per_s=977_000, bytes_per_point=16.0, interface='1000BASE-T', time_sync='PTP',
                    size_mm=(60.0, 110.0, 60.0), optical_z_mm=30.0, power_w=35.0, url=voy_r, builtin=True,
                    ranging='fmcw',
                    params={'velocity_max_mps': 63.0, 'velocity_precision_mps': 0.028, 'range_precision_m': 0.025,
                            'points_per_s_raw': 977_000.0, 'lines_per_frame': 128.0,
                            'assumed': ['size_mm orientation', 'interface rate', 'bytes_per_point', 'wavelength_nm']},
                    note=_note(voy_r, 'The wide variant of Carbon (from a reseller\'s sheet; Voyant\'s own page shows '
                                      'Carbon 30): 60° × 60° field, 128 lines (0.47°), 30 m on a 10 % target, 75 m '
                                      'maximum, 2.8 cm/s velocity precision, 63 m/s maximum radial velocity (Voyant), '
                                      'up to 977,000 points/s, under 35 W, 110 × 60 × 60 mm, PoE++ and gPTP. Link '
                                      'rate and wavelength not stated (1000BASE-T and 1550 nm assumed).',
                               f'; {voy}')),
    ]


def is_fmcw(model) -> bool:
    return model is not None and getattr(model, 'kind', '') == 'lidar' and getattr(model, 'ranging', '') == 'fmcw'


@dataclass
class FmcwSpec:
    """The figures the Doppler model uses for one lidar, each with where it came from."""
    wavelength_nm: float
    velocity_sigma: float               # m/s, 1 σ
    velocity_max: float                 # m/s; inf = not stated (nothing is dropped)
    range_sigma: float                  # m, 1 σ (0 = not stated: none added)
    sources: dict                       # {figure: 'datasheet' | 'assumed' | 'setting'}


def spec_of(model, override_sigma: float = 0.0) -> FmcwSpec:
    p = dict(getattr(model, 'params', None) or {})
    src = {}

    def num(k):
        try:
            v = float(p.get(k))
        except (TypeError, ValueError):
            return None
        return v if math.isfinite(v) and v > 0 else None
    wl = num('wavelength_nm')
    src['wavelength_nm'] = 'datasheet' if wl else 'assumed'
    sig = num('velocity_precision_mps')
    src['velocity_sigma'] = 'datasheet' if sig else 'assumed'
    if override_sigma > 0:
        sig, src['velocity_sigma'] = override_sigma, 'setting'
    vmax = num('velocity_max_mps')
    src['velocity_max'] = 'datasheet' if vmax else 'not stated'
    rs = num('range_precision_m')
    src['range_sigma'] = 'datasheet' if rs else 'not stated'
    return FmcwSpec(wl or DEFAULT_WAVELENGTH_NM, sig or DEFAULT_VEL_SIGMA, vmax or math.inf, rs or 0.0, src)


# ======================================================================= Doppler
def rigid_velocity(P, centre, v, yaw_rate) -> np.ndarray:
    """(n, 3) velocities of points P of a rigid body moving at v (3,) with yaw rate (rad/s) about ẑ through centre."""
    P = np.asarray(P, float)
    r = P - np.asarray(centre, float)
    w = float(yaw_rate)
    return np.asarray(v, float)[None, :] + np.stack([-w * r[:, 1], w * r[:, 0], np.zeros(len(r))], 1)


def sensor_velocity(v_ego, yaw_rate, r_s) -> np.ndarray:
    """The sensor's velocity (3,) on a vehicle whose origin moves at v_ego (3,) turning at yaw_rate (rad/s)."""
    r = np.asarray(r_s, float)
    w = float(yaw_rate)
    return np.asarray(v_ego, float) + np.array([-w * r[1], w * r[0], 0.0])


def radial_velocity(P, sensor_pos, v_sensor, v_points=None) -> tuple[np.ndarray, np.ndarray]:
    """(v_r (n,), d (n, 3)): each point's radial velocity seen from the sensor (SIGN) and its unit ray. v_points (n, 3)
    the points' own velocities (None: a static scene)."""
    P = np.asarray(P, float)
    d = P - np.asarray(sensor_pos, float)[None, :]
    n = np.linalg.norm(d, axis=1)
    d = d / np.maximum(n, 1e-9)[:, None]
    rel = -np.asarray(v_sensor, float)[None, :] if v_points is None else \
        np.asarray(v_points, float) - np.asarray(v_sensor, float)[None, :]
    return np.einsum('ij,ij->i', d, rel), d


def measure(v_true, sigma, v_max, rng) -> np.ndarray:
    """The measured radial velocity: Gaussian noise of sigma, NaN beyond ±v_max (outside the stated range)."""
    v = np.asarray(v_true, float)
    out = v + (rng.normal(0.0, sigma, v.shape) if sigma > 0 else 0.0)
    if math.isfinite(v_max):
        out = np.where(np.abs(v) > v_max, np.nan, out)
    return out


def fit_sensor_velocity(D, vr, sigma=DEFAULT_VEL_SIGMA, seed=0, iters=150, thresh=None, sample=6000) -> dict | None:
    """The sensor's velocity from one sweep: v_r = −d · v_s on the static points, RANSAC over three-point solutions
    (on up to `sample` points) then least squares on the inliers, twice. Returns {'v' (3,), 'inliers' (n,) bool,
    'rms', 'cov' (3, 3), 'n', 'n_inliers'} or None (fewer than 3 usable rays, or rays in one plane)."""
    D = np.asarray(D, float)
    vr = np.asarray(vr, float)
    ok = np.isfinite(vr) & np.all(np.isfinite(D), axis=1)
    idx = np.flatnonzero(ok)
    if len(idx) < 3:
        return None
    thr = float(thresh) if thresh else max(4.0 * float(sigma), 0.05)
    rng = np.random.default_rng(seed)
    sub = idx if len(idx) <= sample else rng.choice(idx, sample, replace=False)
    Ds, vs = D[sub], vr[sub]
    best, best_n = None, -1
    for _ in range(int(iters)):
        k = rng.choice(len(sub), 3, replace=False)
        A = -Ds[k]
        if abs(np.linalg.det(A)) < 1e-3:
            continue
        v = np.linalg.solve(A, vs[k])
        n_in = int(np.count_nonzero(np.abs(vs + Ds @ v) < thr))
        if n_in > best_n:
            best, best_n = v, n_in
    if best is None:
        best = np.linalg.lstsq(-Ds, vs, rcond=None)[0]
    v = best
    for _ in range(3):
        res = np.abs(vr + D @ v)
        inl = ok & (res < thr)
        if np.count_nonzero(inl) < 3:
            break
        A = -D[inl]
        if np.linalg.matrix_rank(A) < 3:
            return None
        v = np.linalg.lstsq(A, vr[inl], rcond=None)[0]
    res = vr + D @ v
    inl = ok & (np.abs(res) < thr)
    A = -D[inl]
    rms = float(np.sqrt(np.mean(res[inl] ** 2))) if inl.any() else float('nan')
    try:
        cov = np.linalg.inv(A.T @ A) * max(rms, 1e-12) ** 2
    except np.linalg.LinAlgError:
        cov = np.full((3, 3), np.nan)
    return {'v': v, 'inliers': inl, 'rms': rms, 'cov': cov, 'n': int(ok.sum()), 'n_inliers': int(inl.sum())}


def ego_from_sensor_velocity(v_s, r_s, min_lever=0.3) -> dict:
    """Forward speed and yaw rate of the vehicle from one sensor's velocity: no side slip at the rear axle (the origin),
    so v_s = (v_x − ω y_s, ω x_s, v_z). The yaw rate needs the sensor off the rear axle (|x_s| ≥ min_lever m); else
    it is None and v_x is v_s,x."""
    v_s = np.asarray(v_s, float)
    x_s, y_s = float(r_s[0]), float(r_s[1])
    if abs(x_s) < min_lever:
        return {'vx': float(v_s[0]), 'yaw_rate': None, 'vz': float(v_s[2])}
    w = float(v_s[1]) / x_s
    return {'vx': float(v_s[0] + w * y_s), 'yaw_rate': w, 'vz': float(v_s[2])}


def compensated(D, vr, v_s) -> np.ndarray:
    """The radial velocity with the sensor's own motion removed: the point's own velocity along the ray."""
    return np.asarray(vr, float) + np.asarray(D, float) @ np.asarray(v_s, float)


# ======================================================================= scenes
CLASSES = ('car', 'truck', 'bus', 'cyclist', 'motorcyclist', 'pedestrian')
SIZES = {'car': (4.5, 1.85, 1.5), 'truck': (9.0, 2.5, 3.4), 'bus': (12.0, 2.55, 3.2), 'cyclist': (1.8, 0.6, 1.7),
         'motorcyclist': (2.1, 0.8, 1.6), 'pedestrian': (0.5, 0.6, 1.75)}
# a scene: [(name, class, x, y, heading deg, speed km/h over the ground, yaw rate deg/s)] in the vehicle frame at t = 0
SCENES = {
    'urban': {'label': 'Urban', 'ego_kmh': 50.0, 'walls_m': 9.0, 'actors': [
        ('lead car', 'car', 22.0, 0.0, 0.0, 40.0, 0.0),
        ('oncoming car', 'car', 38.0, 3.5, 180.0, 50.0, 0.0),
        ('oncoming bus', 'bus', 70.0, 3.6, 180.0, 40.0, 0.0),
        ('crossing cyclist', 'cyclist', 16.0, -5.0, 90.0, 18.0, 0.0),
        ('crossing pedestrian', 'pedestrian', 11.0, 4.5, -90.0, 5.0, 0.0),
        ('parked car', 'car', 14.0, -3.6, 0.0, 0.0, 0.0),
        ('overtaking motorcycle', 'motorcyclist', -9.0, 2.6, 0.0, 65.0, 0.0),
        ('following car', 'car', -14.0, 0.0, 0.0, 50.0, 0.0),
        ('turning car', 'car', 28.0, -9.0, 60.0, 20.0, 15.0),
        ('far crossing car', 'car', 90.0, -20.0, 90.0, 50.0, 0.0)]},
    'highway': {'label': 'Highway', 'ego_kmh': 120.0, 'walls_m': 12.0, 'actors': [
        ('lead car', 'car', 45.0, 0.0, 0.0, 110.0, 0.0),
        ('truck in the next lane', 'truck', 25.0, -3.7, 0.0, 85.0, 0.0),
        ('overtaking car', 'car', -12.0, 3.7, 0.0, 140.0, 0.0),
        ('oncoming car', 'car', 80.0, 7.5, 180.0, 130.0, 0.0),
        ('oncoming truck', 'truck', 140.0, 8.0, 180.0, 90.0, 0.0),
        ('following car', 'car', -30.0, 0.0, 0.0, 120.0, 0.0),
        ('car merging', 'car', 60.0, -5.5, 6.0, 100.0, 2.0)]},
}
HIT_NONE, HIT_GROUND, HIT_WALL, HIT_OBSTACLE = -1, 0, 1, 2       # an actor i is 10 + i


@dataclass
class Frame:
    """One sweep of one lidar as the Doppler model needs it (vehicle frame)."""
    P: np.ndarray                       # (n, 3) float32 points
    inst: np.ndarray                    # (n,) int: 0 static, else the object's id (objects)
    sensor_pos: np.ndarray              # (3,)
    v_ego: np.ndarray                   # (3,) the vehicle origin's velocity, vehicle axes
    yaw_rate: float                     # rad/s
    objects: dict                       # {id: {'name', 'cls', 'c' (3,), 'v' (3,), 'w' rad/s, 'size'?, 'yaw'?}}
    t: float = 0.0                      # s from the drive's start
    label: str = ''
    unknown: np.ndarray | None = None   # (n,) bool: points whose motion is unknown (left out of the scores)

    @property
    def v_sensor(self) -> np.ndarray:
        return sensor_velocity(self.v_ego, self.yaw_rate, self.sensor_pos)


def scene_actors(scene: str) -> list:
    sc = SCENES.get(scene) or SCENES['urban']
    out = []
    for i, (name, cls, x, y, hd, kmh, wdps) in enumerate(sc['actors']):
        L, W, Hh = SIZES[cls]
        a = math.radians(hd)
        v = kmh / 3.6
        out.append({'id': i + 1, 'name': name, 'cls': cls, 'c': np.array([x, y, Hh / 2]), 'size': (L, W, Hh),
                    'yaw': hd, 'v': np.array([v * math.cos(a), v * math.sin(a), 0.0]),
                    'w': math.radians(wdps)})
    return out


def scan_dirs(sensor, model, max_rays: int) -> tuple[np.ndarray, np.ndarray, int, int]:
    """(D (n, 3) vehicle-frame unit rays, elevation (n,) deg, channels used, columns used): the scan, thinned to at
    most about max_rays rays with about the same angular step across and along the channels."""
    from . import sensors as S
    E = model.elevations()
    C = len(E)
    half = model.az_half()
    width = 360.0 if half >= 180.0 else 2 * half
    J = model.columns()
    if C * J > max_rays:                                  # thin both ways to about the same angular step
        lo, hi = model.v_fov()
        a = math.sqrt(width * max(hi - lo, 1.0) / max_rays)
        Ck = int(min(C, max(2, round(max(hi - lo, 1.0) / a))))
        J = int(max(1, min(J, max_rays // max(Ck, 1))))
        if Ck < C:
            E = E[np.unique(np.linspace(0, C - 1, Ck).round().astype(int))]
            C = len(E)
    az = S.scan_azimuths(model, width / J)
    D = (S.directions(E, az) @ S.rotation(sensor).T).reshape(-1, 3)
    el = np.repeat(E, len(az))
    return D, el, C, len(az)


def cast_scene(project, sensor, model, actors, walls_m=0.0, engine=None, vg=None, max_rays=40000):
    """Ray-cast one sweep of a lidar against the ground, roadside walls at y = ±walls_m (6 m high, 0 = none), the
    project's obstacles, the actors' boxes and (with an engine and a vehicle) the body. Returns (P (k, 3), hit (k,)
    int, D (k, 3), rays cast, rays blocked by the body)."""
    from . import drivescene as DS
    from . import groundtruth as GT
    pos = np.asarray(sensor.pos(), float)
    D, _, _, _ = scan_dirs(sensor, model, max_rays)
    n = len(D)
    with np.errstate(divide='ignore', invalid='ignore'):
        t = np.where((D[:, 2] < -1e-9) & (pos[2] > 0), -pos[2] / D[:, 2], np.inf)
    hit = np.where(np.isfinite(t), HIT_GROUND, HIT_NONE).astype(np.int32)

    def nearer(tt, kind):
        tt = np.asarray(tt, float)
        closer = tt < t
        t[closer] = tt[closer]
        hit[closer] = kind
    if walls_m > 0:
        for sgn in (1.0, -1.0):
            tin, _ = GT.ray_box(pos, D, (20.0, sgn * (walls_m + 0.5), 3.0), (400.0, 1.0, 6.0), np.eye(3))
            nearer(tin, HIT_WALL)
    for ob in getattr(project, 'obstacles', ()) or ():
        if ob.height > 0 and ob.radius > 0:
            nearer(DS._ray_cyl(pos, D, ob.x, ob.y, ob.radius, 0.0, ob.height), HIT_OBSTACLE)
    for a in actors:
        tin, _ = GT.ray_box(pos, D, a['c'], a['size'], GT.box_matrix(a['yaw']))
        nearer(np.where(tin > 1e-6, tin, np.inf), 10 + int(a['id']))
    blocked = 0
    if engine is not None:
        try:
            if vg is None:
                vg = engine.vehicle_geom(project)
            if vg is not None:
                dv = engine.cube(pos, vg.static).lookup(D)[0].astype(float)
                body = (dv > 0.05) & (dv < t)
                blocked = int(body.sum())
                t[body] = np.inf
                hit[body] = HIT_NONE
        except Exception:                                        # noqa: BLE001 — no body: the scene alone
            pass
    lim = float(model.range_m)
    ok = (hit != HIT_NONE) & (t >= model.range_min_m) & (t <= lim)
    return pos + D[ok] * t[ok, None], hit[ok], D[ok], n, blocked


def scene_frame(project, sensor, model, settings, engine=None, vg=None) -> Frame:
    """The scripted scene of the settings seen by one lidar."""
    actors = scene_actors(settings.scene)
    walls = (SCENES.get(settings.scene) or SCENES['urban'])['walls_m'] if settings.walls else 0.0
    P, hit, _, n, blocked = cast_scene(project, sensor, model, actors, walls, engine, vg, settings.max_rays)
    inst = np.where(hit >= 10, hit - 10, 0).astype(np.int32)      # hit 10 + id, actor ids start at 1
    objs = {a['id']: {'name': a['name'], 'cls': a['cls'], 'c': a['c'], 'v': a['v'], 'w': a['w'],
                      'size': a['size'], 'yaw': a['yaw']} for a in actors}
    fr = Frame(P.astype(np.float32), inst, np.asarray(sensor.pos(), float),
               np.array([settings.ego_speed_kmh / 3.6, 0.0, 0.0]), math.radians(settings.yaw_rate_dps), objs,
               label=f'{(SCENES.get(settings.scene) or SCENES["urban"])["label"]} scene')
    fr.rays, fr.blocked = n, blocked
    return fr


def point_velocities(fr: Frame) -> np.ndarray:
    """(n, 3) every point's velocity over the ground (0 for static points and points of unknown objects)."""
    V = np.zeros((len(fr.P), 3))
    for oid, o in fr.objects.items():
        m = fr.inst == oid
        if m.any():
            V[m] = rigid_velocity(fr.P[m], o['c'], o['v'], o.get('w', 0.0))
    return V


# ======================================================================= the 4D cloud and its scores
@dataclass
class Cloud:
    """A 4D cloud: the points, their measured radial velocity and what the study made of them."""
    P: np.ndarray                       # (n, 3) float32
    v_r: np.ndarray                     # (n,) float32 measured (NaN: outside the velocity range)
    v_true: np.ndarray                  # (n,) float32 true radial velocity
    v_comp: np.ndarray                  # (n,) float32 measured, the sensor's estimated motion removed
    moving: np.ndarray                  # (n,) bool, from Doppler alone
    moving_gt: np.ndarray               # (n,) bool, the ground truth (speed over the ground > moving_mps)
    inst: np.ndarray                    # (n,) int


def evaluate_frame(fr: Frame, spec: FmcwSpec, moving_mps=0.3, seed=0) -> tuple[Cloud, dict]:
    """The 4D cloud of a frame and its scores: the sensor's velocity fitted from the sweep against the truth, the
    vehicle's speed and yaw rate from it, moving points found by Doppler against the ground truth, per object."""
    rng = np.random.default_rng(seed)
    V = point_velocities(fr)
    vs = fr.v_sensor
    vt, D = radial_velocity(fr.P, fr.sensor_pos, vs, V)
    vm = measure(vt, spec.velocity_sigma, spec.velocity_max, rng)
    fit = fit_sensor_velocity(D, vm, spec.velocity_sigma, seed=seed)
    unknown = fr.unknown if fr.unknown is not None else np.zeros(len(vt), bool)
    speed = np.linalg.norm(V[:, :2], axis=1)
    gt = (speed > moving_mps) & ~unknown
    thr = max(float(moving_mps), 3.0 * spec.velocity_sigma)
    if fit is not None:
        vc = compensated(D, vm, fit['v'])
        mov = np.isfinite(vc) & (np.abs(vc) > thr)
    else:
        vc = np.full(len(vm), np.nan)
        mov = np.zeros(len(vm), bool)
    along = np.abs(np.einsum('ij,ij->i', D, V))                  # the point's own velocity along the ray
    across = gt & (along <= thr)
    finite = np.isfinite(vm)
    scored = ~unknown
    tp = int(np.count_nonzero(mov & gt))
    fp = int(np.count_nonzero(mov & ~gt & scored))
    fn = int(np.count_nonzero(~mov & gt))
    m = {'points': int(len(vt)), 'static_points': int(np.count_nonzero((fr.inst == 0) & scored)),
         'moving_points': int(gt.sum()), 'unknown_points': int(unknown.sum()),
         'moving_found': tp, 'false_moving': fp, 'missed_moving': fn,
         'recall': tp / max(int(gt.sum()), 1) if gt.any() else None,
         'precision': tp / max(tp + fp, 1) if (tp + fp) else None,
         'recall_along_ray': int((mov & gt & ~across).sum()) / int((gt & ~across).sum()) if (gt & ~across).any()
         else None,
         'moving_found_along': int((mov & gt & ~across).sum()), 'across_ray_points': int(across.sum()),
         'outside_velocity_range': int((~finite).sum()),
         'v_r_min': float(np.nanmin(vm)) if finite.any() else None,
         'v_r_max': float(np.nanmax(vm)) if finite.any() else None,
         'threshold_mps': thr, 'velocity_sigma_mps': spec.velocity_sigma,
         'velocity_max_mps': spec.velocity_max if math.isfinite(spec.velocity_max) else None,
         'v_sensor_true': [float(v) for v in vs]}
    if fit is not None:
        e = fit['v'] - vs
        eg = ego_from_sensor_velocity(fit['v'], fr.sensor_pos)
        m.update(v_sensor_est=[float(v) for v in fit['v']], ego_error_mps=float(np.linalg.norm(e)),
                 ego_error_xyz=[float(v) for v in e], fit_rms_mps=fit['rms'], fit_inliers=fit['n_inliers'],
                 ego_sigma_mps=float(np.sqrt(max(np.trace(fit['cov']), 0.0))) if np.all(np.isfinite(fit['cov']))
                 else None,
                 speed_true=float(fr.v_ego[0]), speed_est=eg['vx'], speed_error_mps=eg['vx'] - float(fr.v_ego[0]),
                 yaw_rate_true_dps=math.degrees(fr.yaw_rate),
                 yaw_rate_est_dps=None if eg['yaw_rate'] is None else math.degrees(eg['yaw_rate']))
    else:
        m.update(v_sensor_est=None, ego_error_mps=None, speed_error_mps=None)
    per = []
    for oid, o in sorted(fr.objects.items()):
        k = fr.inst == oid
        if not k.any():
            continue
        sp = float(np.linalg.norm(np.asarray(o['v'], float)[:2]))
        per.append({'id': int(oid), 'name': o.get('name', str(oid)), 'class': o.get('cls', ''),
                    'points': int(k.sum()), 'speed_mps': sp, 'moving': sp > moving_mps,
                    'found': int((mov & k).sum()), 'share_found': float((mov & k).sum() / k.sum()),
                    'mean_v_r': float(np.nanmean(vm[k])) if np.isfinite(vm[k]).any() else None,
                    'mean_own_v_r': float(np.nanmean(vc[k])) if np.isfinite(vc[k]).any() else None,
                    'across_ray': float((across & k).sum() / k.sum())})
    m['objects'] = per
    m['moving_objects'] = sum(1 for o in per if o['moving'])
    m['moving_objects_found'] = sum(1 for o in per if o['moving'] and o['found'] >= 3)
    cl = Cloud(fr.P.astype(np.float32), vm.astype(np.float32), vt.astype(np.float32), vc.astype(np.float32),
               mov, gt, fr.inst.astype(np.int32))
    return cl, m


# ======================================================================= recordings (CARLA drives with ground truth)
def _quat(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _nus_dir(folder) -> str | None:
    root = os.path.join(folder, 'nuscenes')
    if not os.path.isdir(root):
        return None
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name)
        if name.startswith('v1.0') and os.path.isfile(os.path.join(p, 'sample_data.json')):
            return p
    return None


def _bag_dir(folder) -> str | None:
    root = os.path.join(folder, 'rosbag')
    if not os.path.isdir(root):
        return None
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name)
        if os.path.isdir(p) and any(f.endswith(('.mcap', '.db3')) for f in os.listdir(p)):
            return p
    return None


def recording_lidars(folder) -> list:
    """The lidar tags of a recording (its nuScenes LIDAR_<tag> channels)."""
    nd = _nus_dir(folder)
    if nd is None:
        return []
    with open(os.path.join(nd, 'sensor.json'), encoding='utf-8') as fh:
        return sorted(s['channel'][6:] for s in json.load(fh) if s.get('modality') == 'lidar'
                      and s['channel'].startswith('LIDAR_'))


def track_velocity(ts, C, yaws=None, t=None, window=0.25) -> tuple[np.ndarray, float]:
    """An object's velocity (3,) and yaw rate (rad/s) at time t from its track (times ts (k,) s, centres C (k, 3), yaw
    radians (k,)): a least-squares line through the samples within ±window s of t (the two nearest when fewer than two).
    The fallback when a drive records no velocities."""
    ts = np.asarray(ts, float)
    C = np.asarray(C, float)
    if len(ts) < 2:
        return np.zeros(3), 0.0
    t = float(ts[len(ts) // 2] if t is None else t)
    k = np.abs(ts - t) <= window
    if k.sum() < 2:
        k = np.zeros(len(ts), bool)
        k[np.argsort(np.abs(ts - t))[:2]] = True
    tt = ts[k] - t
    A = np.c_[np.ones(k.sum()), tt]
    v = np.linalg.lstsq(A, C[k], rcond=None)[0][1]
    w = 0.0
    if yaws is not None:
        y = np.unwrap(np.asarray(yaws, float)[k])
        w = float(np.linalg.lstsq(A, y, rcond=None)[0][1])
    return v, w


class Recording:
    """A CARLA drive written by Vantage's bridge, read for the Doppler model: the sweeps of one lidar, the ego's motion,
    every actor's box and velocity (CARLA's own from the bag, else differenced boxes)."""

    def __init__(self, folder: str, tag: str | None = None):
        self.folder = os.path.abspath(folder)
        nd = _nus_dir(self.folder)
        if nd is None:
            raise ValueError(f'{folder}: no nuScenes tables (nuscenes/v1.0-*/sample_data.json)')
        self.version = os.path.basename(nd)

        def load(name):
            with open(os.path.join(nd, name), encoding='utf-8') as fh:
                return json.load(fh)
        sensors = {s['token']: s['channel'] for s in load('sensor.json')}
        lidars = sorted(c[6:] for c in sensors.values() if c.startswith('LIDAR_'))
        if not lidars:
            raise ValueError(f'{folder}: the recording has no lidar')
        self.tag = tag if tag in lidars else lidars[0]
        cs = {c['token']: c for c in load('calibrated_sensor.json')}
        poses = {e['token']: e for e in load('ego_pose.json')}
        chan = f'LIDAR_{self.tag}'
        rows = [r for r in load('sample_data.json') if sensors.get(cs[r['calibrated_sensor_token']]['sensor_token'])
                == chan]
        rows.sort(key=lambda r: r['timestamp'])
        self.sweeps = rows
        self.poses = poses
        self.cs = cs
        self.t0 = int(rows[0]['timestamp']) if rows else 0
        c = cs[rows[0]['calibrated_sensor_token']]
        self.R_mount = _quat(c['rotation'])
        self.t_mount = np.asarray(c['translation'], float)
        self.panoptic = {}
        gt_idx = os.path.join(self.folder, 'ground_truth', 'index.json')
        if os.path.isfile(gt_idx):
            with open(gt_idx, encoding='utf-8') as fh:
                doc = json.load(fh)
            for sw in ((doc.get('lidars') or {}).get(self.tag) or {}).get('sweeps', []):
                if sw.get('panoptic'):
                    self.panoptic[sw['sample_data_token']] = sw['panoptic']
        self.boxes, self.odom, self.cloud_t = [], [], {}
        self.velocity_source = 'none'
        bag = _bag_dir(self.folder)
        if bag is not None:
            self._read_bag(bag)
        self.ego_source = 'bag /odom twist' if self.odom else 'ego poses differenced'

    def _read_bag(self, path):
        from . import rosbag as RB
        with RB.BagReader(path) as bag:
            want = [t for t in ('/ground_truth/boxes', '/odom') if t in bag.topics]
            for topic, t_ns, msg in bag.messages(want, skip_corrupt=True):
                if not isinstance(msg, dict):
                    continue
                if topic == '/odom':
                    tw = msg['twist']['twist']
                    self.odom.append((t_ns, np.array([tw['linear'][k] for k in 'xyz'], float),
                                      np.array([tw['angular'][k] for k in 'xyz'], float)))
                else:
                    try:
                        d = json.loads(msg['data'])
                    except (ValueError, KeyError, TypeError):
                        continue
                    self.boxes.append((int(d.get('stamp_ns', t_ns)), d.get('boxes') or []))
        self.boxes.sort(key=lambda b: b[0])
        self.odom.sort(key=lambda o: o[0])
        have_v = any('velocity' in b for _, bs in self.boxes for b in bs)
        self.velocity_source = 'CARLA (bag /ground_truth/boxes)' if have_v else \
            ('boxes differenced' if self.boxes else 'none')

    # ---- the ego
    def _pose(self, row):
        e = self.poses[row['ego_pose_token']]
        return _quat(e['rotation']), np.asarray(e['translation'], float)

    def ego_motion(self, i) -> tuple[np.ndarray, float]:
        """(v (3,) vehicle axes, yaw rate rad/s) at sweep i: the bag's /odom twist nearest in time, else the ego poses
        differenced (central)."""
        t_ns = int(self.sweeps[i]['timestamp']) * 1000
        if self.odom:
            ts = np.array([o[0] for o in self.odom])
            k = int(np.argmin(np.abs(ts - t_ns)))
            return self.odom[k][1].copy(), float(self.odom[k][2][2])
        a, b = max(i - 1, 0), min(i + 1, len(self.sweeps) - 1)
        if a == b:
            return np.zeros(3), 0.0
        Ra, ta = self._pose(self.sweeps[a])
        Rb, tb = self._pose(self.sweeps[b])
        dt = (int(self.sweeps[b]['timestamp']) - int(self.sweeps[a]['timestamp'])) * 1e-6
        R, _ = self._pose(self.sweeps[i])
        v = R.T @ (tb - ta) / dt
        dyaw = math.atan2(Rb[1, 0], Rb[0, 0]) - math.atan2(Ra[1, 0], Ra[0, 0])
        return v, math.atan2(math.sin(dyaw), math.cos(dyaw)) / dt

    # ---- the objects
    def objects_at(self, i) -> dict:
        """{id: {'name', 'cls', 'c', 'v', 'w', 'size', 'yaw'}} of the boxes nearest sweep i, in its vehicle frame."""
        if not self.boxes:
            return {}
        t_ns = int(self.sweeps[i]['timestamp']) * 1000
        ts = np.array([b[0] for b in self.boxes])
        k = int(np.argmin(np.abs(ts - t_ns)))
        R, T = self._pose(self.sweeps[i])
        out = {}
        for b in self.boxes[k][1]:
            oid = int(b['id'])
            c = R.T @ (np.asarray(b['centre'], float) - T)
            yaw_w = float(b.get('yaw', 0.0))
            if 'velocity' in b:
                v_w = np.asarray(b['velocity'], float)
            else:
                v_w = self._diff_velocity(oid, k)
            w = self._yaw_rate(oid, k)
            yaw_v = yaw_w - math.atan2(R[1, 0], R[0, 0])
            out[oid] = {'name': b.get('name', ''), 'cls': _class_of(b.get('category', '')), 'c': c,
                        'v': R.T @ v_w, 'v_world': v_w, 'w': w, 'size': tuple(float(s) for s in b['size_lwh']),
                        'yaw': math.degrees(yaw_v), 'category': b.get('category', '')}
        return out

    def _track(self, oid, k, span=3):
        ts, C, Y = [], [], []
        for j in range(max(k - span, 0), min(k + span + 1, len(self.boxes))):
            for b in self.boxes[j][1]:
                if int(b['id']) == oid:
                    ts.append(self.boxes[j][0] * 1e-9)
                    C.append(b['centre'])
                    Y.append(float(b.get('yaw', 0.0)))
        return ts, C, Y

    def _diff_velocity(self, oid, k) -> np.ndarray:
        ts, C, Y = self._track(oid, k)
        return track_velocity(ts, C, Y, self.boxes[k][0] * 1e-9)[0] if len(ts) >= 2 else np.zeros(3)

    def _yaw_rate(self, oid, k) -> float:
        ts, C, Y = self._track(oid, k, 2)
        return track_velocity(ts, C, Y, self.boxes[k][0] * 1e-9)[1] if len(ts) >= 2 else 0.0

    def velocity_check(self, max_frames=0) -> dict:
        """Differenced boxes against CARLA's own velocities (where the bag has both): how well the fallback recovers
        velocity. {'n', 'median_mps', 'p95_mps', 'max_mps'} over moving actors."""
        if not self.boxes or self.velocity_source.startswith('boxes'):
            return {'n': 0}
        errs = []
        rng = range(len(self.boxes)) if not max_frames else range(min(max_frames, len(self.boxes)))
        for k in rng:
            for b in self.boxes[k][1]:
                v = np.asarray(b.get('velocity', (0, 0, 0)), float)
                if np.linalg.norm(v[:2]) < 0.3:
                    continue
                e = self._diff_velocity(int(b['id']), k) - v
                errs.append(float(np.linalg.norm(e[:2])))
        if not errs:
            return {'n': 0}
        e = np.asarray(errs)
        return {'n': int(len(e)), 'median_mps': float(np.median(e)), 'p95_mps': float(np.percentile(e, 95)),
                'max_mps': float(e.max())}

    # ---- a sweep
    def frame(self, i, moving_mps=0.3) -> Frame:
        row = self.sweeps[i]
        a = np.fromfile(os.path.join(self.folder, 'nuscenes', row['filename']), np.float32)
        if a.size % 5:
            raise ValueError(f'{row["filename"]}: not a nuScenes .pcd.bin (x, y, z, intensity, ring)')
        a = a.reshape(-1, 5)
        c = self.cs[row['calibrated_sensor_token']]
        Rm, tm = _quat(c['rotation']), np.asarray(c['translation'], float)
        P = a[:, :3].astype(float) @ Rm.T + tm
        objs = self.objects_at(i)
        inst = np.zeros(len(P), np.int32)
        unknown = np.zeros(len(P), bool)
        pan = self.panoptic.get(row['token'])
        if pan is not None:
            lab = np.load(os.path.join(self.folder, pan))['data'].astype(np.int64)
            if len(lab) == len(P):
                ii = lab % 1000
                inst = np.where((ii > 0) & (ii < 500), ii, 0).astype(np.int32)    # 500+: the map's parked cars
                unknown = (inst > 0) & ~np.isin(inst, list(objs))
                inst[unknown] = 0
        else:
            from . import groundtruth as GT
            for oid, o in objs.items():
                vrel = float(np.linalg.norm(np.asarray(o['v']) - sensor_velocity(*self.ego_motion(i), tm)))
                marg = 0.1 + 0.1 * vrel                             # a sweep spans 0.1 s: a mover smears that far
                k = GT.in_box(P, o['c'], o['size'], GT.box_matrix(o['yaw']), marg) & (P[:, 2] > 0.15)
                inst[k & (inst == 0)] = oid
        v, w = self.ego_motion(i)
        fr = Frame(P.astype(np.float32), inst, tm.copy(), v, w, objs,
                   t=(int(row['timestamp']) - self.t0) * 1e-6, label=f'sweep {i}', unknown=unknown)
        fr.rays, fr.blocked = len(P), 0
        return fr

    def describe(self) -> dict:
        return {'folder': os.path.basename(self.folder), 'lidar': self.tag, 'sweeps': len(self.sweeps),
                'velocities': self.velocity_source, 'ego': self.ego_source,
                'labels': 'nuScenes-panoptic' if self.panoptic else 'points in boxes'}


def _class_of(category: str) -> str:
    c = str(category)
    if c.startswith('human.'):
        return 'pedestrian'
    for pre, cls in (('vehicle.bicycle', 'cyclist'), ('vehicle.motorcycle', 'motorcyclist'), ('vehicle.bus', 'bus'),
                     ('vehicle.truck', 'truck'), ('vehicle.', 'car')):
        if c.startswith(pre):
            return cls
    return 'other'


def radar_doppler_check(folder, max_sweeps=40) -> dict:
    """CARLA's own radar Doppler against this module's radial velocity from the recorded truth: for every radar
    detection of the drive's radars (nuScenes .pcd, vx/vy the radial velocity along the ray, the bridge's convention =
    SIGN), the predicted v_r = d · (v_p − v_s) with v_s from the bag's ego twist and v_p from the actor boxes the
    detection falls in (static elsewhere). Returns {'n', 'median_abs_mps', 'p95_abs_mps', 'sign_agreement', 'n_moving',
    'moving_median_abs_mps'} or {'n': 0}."""
    from . import groundtruth as GT
    rec = Recording(folder)
    nd = _nus_dir(rec.folder)
    with open(os.path.join(nd, 'sample_data.json'), encoding='utf-8') as fh:
        rows = json.load(fh)
    with open(os.path.join(nd, 'sensor.json'), encoding='utf-8') as fh:
        chans = {s['token']: s['channel'] for s in json.load(fh)}
    radar = [r for r in rows
             if chans.get(rec.cs[r['calibrated_sensor_token']]['sensor_token'], '').startswith('RADAR_')]
    radar.sort(key=lambda r: r['timestamp'])
    if not radar or not rec.odom:
        return {'n': 0}
    step = max(len(radar) // max(int(max_sweeps), 1), 1)
    errs, errs_mv, agree = [], [], []
    bts = np.array([b[0] for b in rec.boxes]) if rec.boxes else None
    for r in radar[::step]:
        pts = read_pcd(os.path.join(rec.folder, 'nuscenes', r['filename']))
        if pts is None or not len(pts['x']):
            continue
        c = rec.cs[r['calibrated_sensor_token']]
        Rm, tm = _quat(c['rotation']), np.asarray(c['translation'], float)
        Ps = np.c_[pts['x'], pts['y'], pts['z']].astype(float)
        u = Ps / np.maximum(np.linalg.norm(Ps, axis=1), 1e-9)[:, None]
        h2 = u[:, 0] ** 2 + u[:, 1] ** 2
        meas = np.where(h2 > 1e-9, (pts['vx'] * u[:, 0] + pts['vy'] * u[:, 1]) / np.maximum(h2, 1e-9), np.nan)
        P = Ps @ Rm.T + tm
        t_ns = int(r['timestamp']) * 1000
        ots = np.array([o[0] for o in rec.odom])
        k = int(np.argmin(np.abs(ots - t_ns)))
        v_ego, w_ego = rec.odom[k][1], float(rec.odom[k][2][2])
        vs = sensor_velocity(v_ego, w_ego, tm)
        V = np.zeros((len(P), 3))
        mv = np.zeros(len(P), bool)
        if bts is not None:
            kb = int(np.argmin(np.abs(bts - t_ns)))
            # boxes in this radar sample's vehicle frame (its own ego pose)
            R, T = rec._pose(r)
            for b in rec.boxes[kb][1]:
                cc = R.T @ (np.asarray(b['centre'], float) - T)
                yaw = float(b.get('yaw', 0.0)) - math.atan2(R[1, 0], R[0, 0])
                inb = GT.in_box(P, cc, b['size_lwh'], GT.box_matrix(math.degrees(yaw)), 0.3)
                if inb.any():
                    v = R.T @ np.asarray(b.get('velocity', (0, 0, 0)), float)
                    V[inb] = v
                    mv[inb] = np.linalg.norm(v[:2]) > 0.3
        pred, _ = radial_velocity(P, tm, vs, V)
        ok = np.isfinite(meas)
        e = np.abs(pred[ok] - meas[ok])
        errs += e.tolist()
        errs_mv += np.abs(pred[ok & mv] - meas[ok & mv]).tolist()
        big = ok & (np.abs(pred) > 0.5)
        agree += (np.sign(pred[big]) == np.sign(meas[big])).tolist()
    if not errs:
        return {'n': 0}
    e = np.asarray(errs)
    out = {'n': int(len(e)), 'median_abs_mps': float(np.median(e)), 'p95_abs_mps': float(np.percentile(e, 95)),
           'sign_agreement': float(np.mean(agree)) if agree else None, 'n_moving': len(errs_mv),
           'moving_median_abs_mps': float(np.median(errs_mv)) if errs_mv else None}
    return out


def read_pcd(path) -> dict | None:
    """A binary PCD v0.7 file (nuScenes radar, or this module's export) as {field: array}; None when unreadable."""
    try:
        with open(path, 'rb') as fh:
            data = fh.read()
    except OSError:
        return None
    hdr, pos = {}, 0
    while pos < len(data):
        end = data.index(b'\n', pos)
        line = data[pos:end].decode('ascii', 'replace').strip()
        pos = end + 1
        if not line or line.startswith('#'):
            continue
        k, _, v = line.partition(' ')
        hdr[k.upper()] = v.split()
        if k.upper() == 'DATA':
            break
    if hdr.get('DATA', [''])[0] != 'binary':
        return None
    names, sizes, types = hdr['FIELDS'], [int(s) for s in hdr['SIZE']], hdr['TYPE']
    counts = [int(c) for c in hdr.get('COUNT', ['1'] * len(names))]
    n = int(hdr['POINTS'][0])
    kind = {'F': 'f', 'I': 'i', 'U': 'u'}
    dt = np.dtype([(nm, f'<{kind[t]}{s}', (c,) if c > 1 else ()) for nm, s, t, c in zip(names, sizes, types, counts)])
    arr = np.frombuffer(data, dt, count=n, offset=pos)
    return {nm: arr[nm].astype(float) if arr[nm].dtype.kind == 'f' else arr[nm] for nm in names}


# ======================================================================= sunlight
CONDITIONS = (
    # id, label, sun elevation deg (None: no sun), azimuth deg from the vehicle's heading (0 ahead), sky
    ('night', 'Night', None, 0.0, 'none'),
    ('overcast', 'Overcast day', None, 0.0, 'overcast'),
    ('noon_ahead', 'Noon sun ahead', 60.0, 0.0, 'clear'),
    ('low_ahead', 'Low sun ahead', 5.0, 0.0, 'clear'),
    ('low_behind', 'Low sun behind', 5.0, 180.0, 'clear'),
    ('noon_behind', 'Noon sun behind', 60.0, 180.0, 'clear'),
)
COND_LABEL = {c[0]: c[1] for c in CONDITIONS}
OVERCAST_SHARE = 0.2                    # an overcast sky's irradiance against a clear noon's (typically 10–30 %)


def band_irradiance(lo_nm: float, hi_nm: float) -> dict:
    """G173's mean spectral irradiances over [lo, hi] nm, W m⁻² nm⁻¹: {'etr', 'global', 'direct', 'diffuse_tilt'}
    (diffuse on G173's tilted surface = global − direct · cos(incidence))."""
    w = G173_ROWS[:, 0]
    k = (w >= lo_nm - 1e-9) & (w <= hi_nm + 1e-9)
    if not k.any():
        k = np.zeros(len(w), bool)
        k[np.argmin(np.abs(w - (lo_nm + hi_nm) / 2))] = True
    etr, glo, dirn = (float(np.mean(G173_ROWS[k, j])) for j in (1, 2, 3))
    diff = max(glo - dirn * math.cos(math.radians(G173_TILT_INCIDENCE_DEG)), 0.0)
    return {'etr': etr, 'global': glo, 'direct': dirn, 'diffuse_tilt': diff}


def air_mass(elev_deg: float) -> float:
    """Kasten & Young (1989): relative optical air mass at a sun elevation (degrees above the horizon)."""
    h = max(float(elev_deg), 0.0)
    return 1.0 / (math.sin(math.radians(h)) + 0.50572 * (h + 6.07995) ** -1.6364)


def sky(cond, band: dict) -> dict:
    """{'dn', 'diff', 's' (3,) or None, 'elev', 'klx'}: the direct normal and the diffuse horizontal spectral
    irradiance (W m⁻² nm⁻¹) in a condition, and the sun's direction in the vehicle frame."""
    cid, _, elev, az, kind = cond
    if kind == 'none':
        return {'dn': 0.0, 'diff': 0.0, 's': None, 'elev': None, 'klx': 0.0}
    m15 = air_mass(90.0 - G173_ZENITH_DEG)
    tau = (band['direct'] / band['etr']) ** (1.0 / m15) if band['etr'] > 0 else 0.0
    noon = math.sin(math.radians(90.0 - G173_ZENITH_DEG))
    if kind == 'overcast':
        e60 = 60.0
        dn60 = band['etr'] * tau ** air_mass(e60)
        diff = OVERCAST_SHARE * (dn60 * math.sin(math.radians(e60)) + band['diffuse_tilt'] * math.sin(
            math.radians(e60)) / noon)
        return {'dn': 0.0, 'diff': diff, 's': None, 'elev': None, 'klx': None}
    dn = band['etr'] * tau ** air_mass(elev)
    diff = band['diffuse_tilt'] * max(math.sin(math.radians(elev)), 0.0) / noon
    a, h = math.radians(az), math.radians(elev)
    s = np.array([math.cos(h) * math.cos(a), math.cos(h) * math.sin(a), math.sin(h)])
    return {'dn': dn, 'diff': diff, 's': s, 'elev': elev, 'klx': None}


def illuminance_klx(cond) -> float:
    """Rough global horizontal illuminance of a condition, klx (broadband: 1361 W/m² at the top of the atmosphere,
    0.7 transmittance per air mass, diffuse 10 % of the direct; × LUX_PER_WM2): context only, never used in the
    model."""
    cid, _, elev, _, kind = cond
    if kind == 'none':
        return 0.0
    if kind == 'overcast':
        return OVERCAST_SHARE * illuminance_klx((cid, '', 60.0, 0.0, 'clear'))
    dn = 1361.0 * 0.7 ** air_mass(elev)
    return (dn * math.sin(math.radians(elev)) * 1.1) * LUX_PER_WM2 / 1000.0


def occupancy(L_lambda_W_m2_sr_m, wavelength_m) -> float:
    """Photons per spatial mode (per polarisation) of an unpolarised radiance L_λ (W m⁻² sr⁻¹ m⁻¹): n̄ = L_λ λ⁵/(2hc²)
    (Planck's B_λ = 2hc²/λ⁵ · n̄)."""
    return float(L_lambda_W_m2_sr_m) * wavelength_m ** 5 / (2 * H * C * C)


def planck_occupancy(wavelength_m, T) -> float:
    """1/(exp(hc/λkT) − 1): a blackbody's photons per mode."""
    return 1.0 / math.expm1(H * C / (wavelength_m * KB * T))


def planck_irradiance_toa(wavelength_m) -> float:
    """Spectral irradiance of a 5772 K blackbody sun at 1 au, W m⁻² nm⁻¹: Ω_sun · B_λ(T)."""
    lam = wavelength_m
    B = 2 * H * C * C / lam ** 5 / math.expm1(H * C / (lam * KB * T_SUN_K))
    return OMEGA_SUN * B * 1e-9


def surface_irradiance(n, sk) -> np.ndarray:
    """Spectral irradiance (W m⁻² nm⁻¹) on surfaces of normals n (k, 3): direct · max(0, n · s) + diffuse from an
    isotropic sky over the visible half (1 + n_z)/2."""
    n = np.atleast_2d(np.asarray(n, float))
    E = sk['diff'] * (1.0 + n[:, 2]) / 2.0
    if sk['s'] is not None:
        E = E + sk['dn'] * np.maximum(n @ sk['s'], 0.0)
    return E


def q_tail(z) -> np.ndarray:
    """The Gaussian upper tail Q(z)."""
    from scipy.special import erfc
    return 0.5 * erfc(np.asarray(z, float) / math.sqrt(2.0))


def q_inverse(p: float) -> float:
    from scipy.special import erfcinv
    return float(math.sqrt(2.0) * erfcinv(2.0 * min(max(p, 1e-300), 0.5)))


def tof_background(L_scene, st, sun_dn=0.0) -> np.ndarray:
    """Background photoelectrons per second in one ToF pixel: a scene of spectral radiance L (W m⁻² sr⁻¹ nm⁻¹,
    per beam) through the filter, aperture and pixel field, plus the sun's disc (direct normal sun_dn, W m⁻² nm⁻¹, per
    beam) when it is in the pixel."""
    lam = st.tof_wavelength_nm * 1e-9
    A = math.pi * (st.tof_aperture_mm * 1e-3 / 2) ** 2
    omega = (st.tof_ifov_mrad * 1e-3) ** 2
    eta = st.tof_optics * (st.tof_pde if st.tof_receiver == 'spad' else st.tof_qe)
    ph = lam / (H * C)
    P = (np.asarray(L_scene, float) * omega + np.asarray(sun_dn, float) * min(1.0, omega / OMEGA_SUN)) \
        * st.tof_filter_nm * A
    return eta * P * ph


def tof_noise_var(N_e_per_s, st) -> np.ndarray:
    """Noise variance per range bin, photoelectrons²: a SPAD (photon counting) (N_b + dark counts) τ; a linear-mode APD
    F N_b τ + σ_amp² (shot noise with its excess noise, plus the amplifier). Gaussian approximation of the counts."""
    tau = st.tof_bin_ns * 1e-9
    N = np.asarray(N_e_per_s, float)
    if st.tof_receiver == 'spad':
        return (N + st.tof_dcr) * tau
    return st.tof_excess_noise * N * tau + st.tof_noise_e ** 2


def sun_in_beam(D, s, half_rad) -> np.ndarray:
    """Beams (unit D (k, 3)) whose pixel (half field half_rad) overlaps the sun's disc in direction s."""
    if s is None:
        return np.zeros(len(D), bool)
    return D @ s >= math.cos(SUN_HALF_RAD + half_rad)


def blinded_points_per_frame(sensor, model, s, half_rad) -> int:
    """Points of a full frame whose pixel overlaps the sun's disc: per channel within the disc's reach in elevation,
    the columns within its azimuth window (exact on the datasheet grid, not on the thinned scan)."""
    if s is None:
        return 0
    from . import sensors as S
    q = (s @ S.rotation(sensor))                      # the sun in the sensor frame
    el_s = math.degrees(math.asin(max(min(q[2], 1.0), -1.0)))
    az_s = math.degrees(math.atan2(q[1], q[0]))
    reach = math.degrees(SUN_HALF_RAD + half_rad)
    if model.az_half() < 180.0 and abs(az_s) > model.az_half() + reach:
        return 0
    n = 0
    for e in model.elevations():
        de = abs(e - el_s)
        if de > reach:
            continue
        w = 2 * math.sqrt(max(reach * reach - de * de, 0.0)) / max(math.cos(math.radians(e)), 1e-3)
        n += max(int(math.ceil(w / max(model.h_res, 1e-3))), 1)
    return n


def _local_median(v, el, az, window_deg, wrap) -> np.ndarray:
    """Each ray's median of v over its own channel (rays of one beam elevation) within ±window_deg/2 of its azimuth
    (wrapping round for a spinning lidar): the noise a threshold tracking the ambient light sees."""
    from scipy.ndimage import median_filter
    v = np.asarray(v, float)
    key = np.round(np.asarray(el, float), 4)
    _, inv = np.unique(key, return_inverse=True)
    out = np.empty_like(v)
    for k in range(inv.max() + 1 if len(inv) else 0):
        idx = np.flatnonzero(inv == k)
        order = idx[np.argsort(az[idx])]
        step = float(np.median(np.diff(np.sort(az[idx])))) if len(idx) > 1 else 1.0
        n = int(max(1, round(window_deg / max(step, 1e-6))))
        n += 1 - n % 2
        out[order] = median_filter(v[order], size=min(n, len(order)), mode='wrap' if wrap else 'nearest')
    return out


def sun_study(project, sensor, model, st, engine=None, vg=None) -> list:
    """Per condition, a time-of-flight receiver and an FMCW receiver with this lidar's field, pose and datasheet range:
    the 10 % target's range (median over the beams near the horizon), the beams the sun blinds, the false returns per
    second. The lidar's own kind is flagged ('own')."""
    from . import sensors as S
    fm = is_fmcw(model)
    D, el, _, _ = scan_dirs(sensor, model, min(st.max_rays, 20000))
    pos = np.asarray(sensor.pos(), float)
    with np.errstate(divide='ignore', invalid='ignore'):
        tg = np.where(D[:, 2] < -1e-9, -pos[2] / D[:, 2], np.inf)
    ground = tg <= model.range_m
    horiz = np.hypot(D[:, 0], D[:, 1])
    nf = -np.c_[D[:, 0], D[:, 1], np.zeros(len(D))] / np.maximum(horiz, 1e-9)[:, None]    # facing the sensor
    nf[horiz < 1e-6] = (0.0, 0.0, -1.0)
    up = np.tile([0.0, 0.0, 1.0], (len(D), 1))
    near_h = np.abs(D[:, 2]) <= math.sin(math.radians(2.0))
    if not near_h.any():
        near_h = np.abs(D[:, 2]) <= np.abs(D[:, 2]).min() + 1e-6
    half = st.tof_ifov_mrad * 1e-3 / 2
    pps = model.pps()
    frame_pts = pps / max(model.rate_hz, 1e-6)
    b_tof = band_irradiance(st.tof_wavelength_nm - st.tof_filter_nm / 2, st.tof_wavelength_nm + st.tof_filter_nm / 2)
    spec = spec_of(model) if fm else spec_of(SensorModel('x', ranging='fmcw'))
    lam_f = spec.wavelength_nm
    b_fm = band_irradiance(lam_f - 2.0, lam_f + 2.0)
    bins = max(int(round(2 * max(model.range_max_m, model.range_m) / C / (st.tof_bin_ns * 1e-9))), 1)
    ref_kind = st.tof_reference
    if ref_kind == 'auto':
        ref_kind = 'sun' if (not fm and '100 klx' in (model.note or '')) else 'dark'
    ref_cond = next(c for c in CONDITIONS if c[0] == ('noon_ahead' if ref_kind == 'sun' else 'night'))

    def tof_sigma2(cond):
        sk = sky(cond, b_tof)
        Et = surface_irradiance(nf, sk)                                # 10 % target facing the sensor
        Lt = st.target_reflectivity * Et / math.pi
        Eb = np.where(ground, surface_irradiance(up, sk)[0] * st.ground_reflectivity,
                      Et * st.scene_reflectivity)
        Lb = Eb / math.pi
        on_sun = sun_in_beam(D, sk['s'], half)
        sun = np.where(on_sun, sk['dn'], 0.0)
        return tof_noise_var(tof_background(Lt, st, sun), st), tof_noise_var(tof_background(Lb, st, sun), st), \
            on_sun, sk
    # the reference: the datasheet range holds on a 10 % target in the dark, or (in sunlight) on one facing the noon
    # sun (the worst a sunlit target gets)
    _, _, _, sk_ref = tof_sigma2(ref_cond)
    if ref_kind == 'sun':
        E_ref = sk_ref['dn'] + sk_ref['diff'] * 0.5
        s_ref_t = float(tof_noise_var(tof_background(st.target_reflectivity * E_ref / math.pi, st), st))
    else:
        s_ref_t = float(tof_noise_var(0.0, st))
    az_v = np.degrees(np.arctan2(D[:, 1], D[:, 0]))
    ahead = near_h & (np.abs(az_v) <= 30.0)
    r_cap = model.range_max_m if model.range_max_m > model.range_m else math.inf    # never past the rated maximum
    p_bin = 1.0 - (1.0 - st.p_false_ref) ** (1.0 / bins)
    z = q_inverse(p_bin)
    eta_h = st.fmcw_eta
    pfa_bin_f = 1.0 - (1.0 - st.p_false_ref) ** (1.0 / bins)
    rows = []
    for cond in CONDITIONS:
        if cond[0] not in st.conditions:
            continue
        s2t, s2b, on_sun, sk = tof_sigma2(cond)
        r_tof_all = np.minimum(model.range_m * (s_ref_t / s2t) ** 0.25, r_cap)   # at a constant false-alarm rate
        r_tof = r_tof_all[near_h]
        s_med = _local_median(s2b, el, az_v, st.tof_adapt_deg, model.az_half() >= 180.0)
        fixed = q_tail(z * np.sqrt(s_med / s2b))                          # per bin: beams brighter than that fire
        p_shot = 1.0 - (1.0 - fixed) ** bins
        blinded = blinded_points_per_frame(sensor, model, sk['s'], half)
        # FMCW: photons per mode from the target and from the sun's disc
        skf = sky(cond, b_fm)
        lam = lam_f * 1e-9
        Lt_f = st.target_reflectivity * surface_irradiance(nf, skf) / math.pi * 1e9     # per metre
        n_t = Lt_f * lam ** 5 / (2 * H * C * C)
        n_sun = occupancy(skf['dn'] / OMEGA_SUN * 1e9, lam) if skf['s'] is not None else 0.0
        on_sun_f = sun_in_beam(D, skf['s'], half)
        nbar = n_t + np.where(on_sun_f, n_sun, 0.0)
        r_fm_all = model.range_m * (1.0 + eta_h * nbar) ** -0.5
        r_fm = r_fm_all[near_h]
        pf = pfa_bin_f ** (1.0 / (1.0 + eta_h * nbar))
        pshot_f = 1.0 - (1.0 - pf) ** bins
        sun_share = min(blinded / max(frame_pts, 1.0), 1.0)
        # the scene's beams from the thinned scan, the sun's disc counted on the datasheet grid (blinded points)
        off = ~(on_sun | on_sun_f)
        tof_false = float(np.mean(p_shot[off])) * pps * (1.0 - sun_share) if off.any() else 0.0
        fm_false = float(np.mean(pshot_f[off])) * pps * (1.0 - sun_share) if off.any() else 0.0
        if blinded:
            p_sun = float(1.0 - (1.0 - q_tail(z * math.sqrt(float(np.median(s2b)) / float(
                tof_noise_var(tof_background(0.0, st, sk['dn']), st))))) ** bins)
            tof_false += p_sun * blinded * model.rate_hz
            pf_s = pfa_bin_f ** (1.0 / (1.0 + eta_h * n_sun))
            fm_false += (1.0 - (1.0 - pf_s) ** bins) * blinded * model.rate_hz
        rows.append({
            'condition': cond[0], 'label': cond[1], 'sun_elevation_deg': cond[2], 'sun_azimuth_deg': cond[3],
            'klx_approx': round(illuminance_klx(cond), 1),
            'irradiance_905_dn': sk['dn'], 'irradiance_905_diffuse': sk['diff'],
            'sun_points_per_frame': blinded, 'sun_share': sun_share,
            'tof': {'range_m': float(np.median(r_tof)), 'range_min_m': float(np.min(r_tof)),
                    'range_ahead_m': float(np.median(r_tof_all[ahead])) if ahead.any() else None,
                    'range_change': float(np.median(r_tof)) / model.range_m - 1.0,
                    'false_per_s': tof_false, 'false_share': tof_false / max(pps, 1.0),
                    'blinded_range_m': float(min(model.range_m * (s_ref_t / float(
                        tof_noise_var(tof_background(0.0, st, sk['dn']), st))) ** 0.25, r_cap))
                    if sk['s'] is not None else None},
            'fmcw': {'range_m': float(np.median(r_fm)), 'range_min_m': float(np.min(r_fm)),
                     'range_ahead_m': float(np.median(r_fm_all[ahead])) if ahead.any() else None,
                     'range_change': float(np.median(r_fm)) / model.range_m - 1.0,
                     'false_per_s': fm_false, 'false_share': fm_false / max(pps, 1.0),
                     'photons_per_mode_target': float(np.median(n_t[near_h])),
                     'photons_per_mode_sun': n_sun,
                     'blinded_range_m': float(model.range_m * (1.0 + eta_h * n_sun) ** -0.5)
                     if skf['s'] is not None else None},
            'own': 'fmcw' if fm else 'tof'})
    ref = {'condition': ref_cond[0], 'why': ('the note states 100 klx sunlight' if st.tof_reference == 'auto'
                                            and ref_kind == 'sun' else f'setting: {ref_kind}')}
    for r in rows:
        r['tof_reference'] = ref
    return rows


def crosstalk(sensor, model, st) -> dict:
    """Interference points per frame from st.crosstalk_cars other test cars with one spinning ToF lidar each (a roof
    lidar 1.9 m up, 360°, 120 m), placed st.crosstalk_distance_m apart ahead in the next lane (alternately left and
    behind). A ToF victim takes st.crosstalk_rate of its points per frame per interferer in mutual view (the cited
    worst case); an FMCW victim none."""
    from . import sensors as S
    frame_pts = model.pps() / max(model.rate_hz, 1e-6)
    pos = np.asarray(sensor.pos(), float)
    seen = []
    for k in range(int(st.crosstalk_cars)):
        side = 1.0 if k % 2 == 0 else -1.0
        ahead = (k // 2 + 1) * st.crosstalk_distance_m * (1.0 if k % 4 < 2 else -1.0)
        P = np.array([[ahead + 1.5, side * st.crosstalk_lateral_m, 1.9]])
        in_victim = bool(S.in_field(sensor, model, P)[0])
        rng = float(np.linalg.norm(P[0] - pos))
        in_other = rng <= 120.0                                         # a spinning lidar sees all round
        seen.append({'x': float(P[0, 0]), 'y': float(P[0, 1]), 'range_m': rng, 'mutual_view': in_victim and in_other})
    n_mut = sum(1 for s in seen if s['mutual_view'])
    tof_pts = min(st.crosstalk_rate * n_mut, 1.0) * frame_pts
    return {'interferers': seen, 'in_mutual_view': n_mut, 'rate_per_interferer': st.crosstalk_rate,
            'tof_points_per_frame': tof_pts, 'tof_points_per_s': tof_pts * model.rate_hz,
            'fmcw_points_per_frame': 0.0, 'fmcw_points_per_s': 0.0, 'own': 'fmcw' if is_fmcw(model) else 'tof',
            'basis': 'upper bound: the worst case measured between two HDL-64E (1.5 %, Diehm et al. 2018) per '
                     'interferer in mutual view; FMCW: none (coherent detection, Behroozpour et al. 2017)'}


# ======================================================================= settings
SOURCES = ('scene', 'recording')
REFS = ('auto', 'dark', 'sun')
RECEIVERS = ('spad', 'apd')
LIMITS = {'ego_speed_kmh': (0.0, 250.0), 'yaw_rate_dps': (-60.0, 60.0), 'max_rays': (2000.0, 400000.0),
          'max_sweeps': (1.0, 2000.0), 'moving_mps': (0.05, 5.0), 'velocity_sigma_mps': (0.0, 2.0),
          'seed': (0.0, 2 ** 31 - 1), 'tof_wavelength_nm': (700.0, 1700.0), 'tof_filter_nm': (0.5, 200.0),
          'tof_aperture_mm': (1.0, 200.0), 'tof_ifov_mrad': (0.05, 20.0), 'tof_optics': (0.01, 1.0),
          'tof_qe': (0.001, 1.0), 'tof_pde': (0.001, 1.0), 'tof_dcr': (0.0, 1e8), 'tof_excess_noise': (1.0, 50.0),
          'tof_noise_e': (0.0, 1000.0),
          'tof_bin_ns': (0.1, 100.0), 'tof_adapt_deg': (0.1, 360.0), 'target_reflectivity': (0.01, 1.0),
          'scene_reflectivity': (0.0, 1.0),
          'ground_reflectivity': (0.0, 1.0), 'p_false_ref': (1e-12, 0.1), 'fmcw_eta': (0.01, 1.0),
          'crosstalk_cars': (0.0, 20.0), 'crosstalk_distance_m': (2.0, 200.0), 'crosstalk_lateral_m': (0.0, 30.0),
          'crosstalk_rate': (0.0, 1.0)}
INTS = ('max_rays', 'max_sweeps', 'seed', 'crosstalk_cars')
ASSUMPTIONS = (
    'CARLA renders no Doppler lidar: the radial velocity of every point is computed from the scene\'s (or CARLA\'s '
    'own) velocities plus the datasheet\'s velocity noise, so the ego-velocity error and the moving-point recall '
    'score the estimators, not a sensor\'s own errors. The time-of-flight receiver (aperture, filter, pixel field, '
    'efficiencies, dark counts or amplifier noise, bin) is a typical 905 nm SPAD (or APD) receiver, not a product\'s: '
    'in direct sun its range '
    'ratio depends only on the irradiance ratio. Cross-talk uses the worst case measured between two lidars per '
    'interferer in mutual view: an upper bound. Where an FMCW maker states no velocity precision, 3 cm/s is used; no '
    'wavelength, 1550 nm.')


@dataclass
class FmcwSettings:
    source: str = 'scene'               # scene | recording
    scene: str = 'urban'                # SCENES
    ego_speed_kmh: float = 50.0
    yaw_rate_dps: float = 0.0
    walls: bool = True                  # roadside walls (buildings, barriers): static points beside the road
    recording: str = ''                 # a CARLA drive folder (ground_truth/, nuscenes/, rosbag/)
    recording_tag: str = ''             # its lidar ('' = the first)
    max_sweeps: int = 40
    counterpart: str = 'aeva_aeries_ii'  # the FMCW model a time-of-flight lidar is compared with, at its pose
    moving_mps: float = 0.3             # a point moves when its speed over the ground exceeds this (CARLA's own 0.3)
    velocity_sigma_mps: float = 0.0     # 0 = the datasheet's (or 3 cm/s where none is stated)
    max_rays: int = 40000
    seed: int = 1
    conditions: list = field(default_factory=lambda: [c[0] for c in CONDITIONS])
    tof_reference: str = 'auto'         # auto (100 klx where the note says so, else dark) | dark | sun
    tof_wavelength_nm: float = 905.0
    tof_filter_nm: float = 20.0         # typical: un-stabilised 905 nm diodes drift ~0.27 nm/°C
    tof_aperture_mm: float = 20.0
    tof_ifov_mrad: float = 1.5
    tof_optics: float = 0.7
    tof_receiver: str = 'spad'          # spad (photon counting: most current 905 nm lidars) | apd (linear mode)
    tof_pde: float = 0.2                # a SPAD's photon detection efficiency at 905 nm
    tof_dcr: float = 20000.0            # a SPAD pixel's dark counts per second
    tof_qe: float = 0.7                 # a silicon APD at 905 nm
    tof_excess_noise: float = 3.5       # the APD's excess noise factor F
    tof_noise_e: float = 10.0           # the APD amplifier's noise per bin, photoelectrons rms (input-referred)
    tof_bin_ns: float = 5.0             # the pulse's width
    tof_adapt_deg: float = 10.0         # the azimuth window over which the threshold tracks the ambient light
    target_reflectivity: float = 0.10
    scene_reflectivity: float = 0.3     # walls and vehicles seen around the target
    ground_reflectivity: float = 0.15   # asphalt in the near infrared
    p_false_ref: float = 1e-5           # false returns per point at the reference condition (both receivers)
    fmcw_eta: float = 0.5               # heterodyne efficiency × quantum efficiency
    crosstalk_cars: int = 1
    crosstalk_distance_m: float = 10.0
    crosstalk_lateral_m: float = 3.5
    crosstalk_rate: float = 0.015       # Diehm et al. 2018: 1.5 % (two HDL-64E, worst case)

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d) -> 'FmcwSettings':
        s = FmcwSettings()
        d = d if isinstance(d, dict) else {}
        if d.get('source') in SOURCES:
            s.source = d['source']
        if d.get('scene') in SCENES:
            s.scene = d['scene']
        if d.get('tof_reference') in REFS:
            s.tof_reference = d['tof_reference']
        if d.get('tof_receiver') in RECEIVERS:
            s.tof_receiver = d['tof_receiver']
        for k in ('recording', 'recording_tag', 'counterpart'):
            if isinstance(d.get(k), str):
                setattr(s, k, d[k].strip())
        v = d.get('walls')
        if isinstance(v, (bool, int)):
            s.walls = bool(v)
        elif isinstance(v, str):
            s.walls = v.strip().lower() in ('1', 'true', 'yes', 'on')
        cs = d.get('conditions')
        if isinstance(cs, str):
            cs = [c.strip() for c in cs.split(',')]
        if isinstance(cs, (list, tuple)):
            s.conditions = [c[0] for c in CONDITIONS if c[0] in {str(x) for x in cs}]
        for k, (lo, hi) in LIMITS.items():
            if k in d:
                try:
                    v = float(d[k])
                except (TypeError, ValueError):
                    continue
                if math.isfinite(v):
                    v = min(max(v, lo), hi)
                    setattr(s, k, int(round(v)) if k in INTS else v)
        return s


def settings_of(project) -> FmcwSettings:
    return FmcwSettings.from_dict(getattr(getattr(project, 'analysis', None), 'fmcw', None) or {})


def cache_key(project, settings=None, variant=None) -> str:
    """sha256 of everything the result depends on: the settings, the visible lidars (pose, model figures), the
    obstacles, the body and the rig."""
    import hashlib
    from .experiments import _mesh_key
    from .rig import rig_key
    st = settings or settings_of(project)
    v = variant or project.variant
    lid = []
    for s in v.sensors:
        if s.kind == 'lidar' and getattr(s, 'visible', True):
            m = project.sensor_model_for(s)
            lid.append([s.uid, s.tag, s.x, s.y, s.z, s.yaw, s.pitch, s.roll, m.to_dict()])
    cp = (getattr(project, 'sensor_models', {}) or {}).get(st.counterpart)
    doc = {'schema': SCHEMA, 'settings': st.to_dict(), 'variant': getattr(v, 'uid', ''), 'lidars': lid,
           'counterpart': cp.to_dict() if cp is not None else st.counterpart,
           'obstacles': [[o.x, o.y, o.radius, o.height] for o in getattr(project, 'obstacles', ()) or ()],
           'mesh': _mesh_key(getattr(project.vehicle, 'mesh', None)),
           'rig': rig_key([r for r in getattr(project, 'rig', ()) if r.visible])}
    return hashlib.sha256(json.dumps(doc, sort_keys=True, default=str).encode('utf-8')).hexdigest()


# ======================================================================= the study
@dataclass
class FmcwResult:
    settings: FmcwSettings
    lidars: list                        # per visible lidar: dict (see _lidar_row)
    clouds: dict                        # {tag: Cloud} of the shown sweep
    source: dict                        # what the scene or the recording was
    variant: str = ''
    seconds: float = 0.0
    sweeps: dict = field(default_factory=dict)   # {tag: [per-sweep metrics]} (a recording)
    checks: dict = field(default_factory=dict)   # recording cross-checks (radar Doppler, differenced velocities)

    def lidar(self, tag):
        return next((r for r in self.lidars if r['tag'] == tag), None)


def _counterpart(project, st):
    m = (getattr(project, 'sensor_models', {}) or {}).get(st.counterpart)
    if m is None or not is_fmcw(m):
        m = next((x for x in library() if x.id == st.counterpart), None) or library()[0]
    return m


def _summ_sweeps(ms: list) -> dict:
    def arr(k):
        return np.array([m[k] for m in ms if m.get(k) is not None], float)
    out = {'sweeps': len(ms)}
    e = arr('ego_error_mps')
    if len(e):
        out.update(ego_error_mean_mps=float(e.mean()), ego_error_p95_mps=float(np.percentile(e, 95)),
                   ego_error_max_mps=float(e.max()))
    se = np.abs(arr('speed_error_mps'))
    if len(se):
        out['speed_error_mean_mps'] = float(se.mean())
    gt = sum(m['moving_points'] for m in ms)
    tp = sum(m['moving_found'] for m in ms)
    fp = sum(m['false_moving'] for m in ms)
    acr = sum(m['across_ray_points'] for m in ms)
    out.update(moving_points=gt, moving_found=tp, false_moving=fp, across_ray_points=acr,
               recall=tp / gt if gt else None, precision=tp / (tp + fp) if tp + fp else None,
               recall_along_ray=sum(m['moving_found_along'] for m in ms) / (gt - acr) if gt - acr > 0 else None,
               points=sum(m['points'] for m in ms),
               outside_velocity_range=sum(m['outside_velocity_range'] for m in ms))
    return out


def run(project, engine=None, settings=None, variant=None, progress=None, cancel=None) -> FmcwResult | None:
    """Every visible lidar of the variant: an FMCW lidar's 4D cloud and scores, a time-of-flight lidar's scores as its
    counterpart FMCW model (settings.counterpart) would give them at the same pose, and both receivers under the sun
    and cross-talk. Source: the scripted scene, or a recording's sweeps (its lidar, carried as each FMCW lidar)."""
    import time
    t0 = time.perf_counter()
    st = settings if isinstance(settings, FmcwSettings) else FmcwSettings.from_dict(settings)
    var = variant or project.variant
    lidars = [s for s in var.sensors if s.kind == 'lidar' and getattr(s, 'visible', True)]
    rows, clouds, sweeps, checks = [], {}, {}, {}
    vg = None
    if engine is not None and st.source == 'scene':
        try:
            vg = engine.vehicle_geom(project)
        except Exception:                                        # noqa: BLE001
            vg = None
    rec = None
    if st.source == 'recording':
        if not st.recording or not os.path.isdir(st.recording):
            raise ValueError('No recording: choose a CARLA drive folder (with nuscenes/ and ground_truth/)')
        rec = Recording(st.recording, st.recording_tag or None)
    total = max(len(lidars), 1)
    for i, s in enumerate(lidars):
        if cancel is not None and cancel():
            return None
        m = project.sensor_model_for(s)
        fm = is_fmcw(m)
        dm = m if fm else _counterpart(project, st)
        spec = spec_of(dm, st.velocity_sigma_mps)
        row = {'tag': s.tag, 'uid': s.uid, 'model': m.id, 'name': f'{m.vendor} {m.name}'.strip(), 'fmcw': fm,
               'doppler_model': dm.id, 'doppler_name': f'{dm.vendor} {dm.name}'.strip(),
               'points_per_s': m.pps(), 'doppler_points_per_s': dm.pps(), 'spec': asdict(spec),
               'velocity_max_mps': spec.velocity_max if math.isfinite(spec.velocity_max) else None}
        if rec is None:
            fr = scene_frame(project, s, dm, st, engine, vg)
            cl, met = evaluate_frame(fr, spec, st.moving_mps, st.seed + i)
            row['metrics'] = met
            row['summary'] = _summ_sweeps([met])
            row['rays'], row['blocked'] = fr.rays, fr.blocked
        else:
            n = len(rec.sweeps)
            idx = np.unique(np.linspace(0, n - 1, min(st.max_sweeps, n)).round().astype(int))
            ms, cl = [], None
            for j, k in enumerate(idx):
                if cancel is not None and cancel():
                    return None
                fr = rec.frame(int(k), st.moving_mps)
                c, met = evaluate_frame(fr, spec, st.moving_mps, st.seed + int(k))
                met['t_s'] = fr.t
                met['sweep'] = int(k)
                ms.append(met)
                if cl is None or met['moving_points'] > (row.get('metrics') or {}).get('moving_points', -1):
                    cl, row['metrics'] = c, met            # show the sweep with the most moving points
                if progress is not None:
                    progress((i + (j + 1) / len(idx)) / total * 0.9)
            sweeps[s.tag] = ms
            row['summary'] = _summ_sweeps(ms)
            row['recording_lidar'] = rec.tag
        clouds[s.tag] = cl
        row['sun'] = sun_study(project, s, m, st, engine, vg)
        row['crosstalk'] = crosstalk(s, m, st)
        rows.append(row)
        if progress is not None and rec is None:
            progress((i + 1) / total * 0.95)
    if rec is not None:
        src = rec.describe()
        try:
            checks['differenced_velocity'] = rec.velocity_check()
        except Exception as ex:                                  # noqa: BLE001
            checks['differenced_velocity'] = {'n': 0, 'error': str(ex)}
        try:
            checks['radar_doppler'] = radar_doppler_check(rec.folder, 30)
        except Exception as ex:                                  # noqa: BLE001
            checks['radar_doppler'] = {'n': 0, 'error': str(ex)}
    else:
        sc = SCENES.get(st.scene) or SCENES['urban']
        src = {'scene': st.scene, 'label': sc['label'], 'actors': len(sc['actors']),
               'ego_speed_kmh': st.ego_speed_kmh, 'yaw_rate_dps': st.yaw_rate_dps,
               'walls_m': sc['walls_m'] if st.walls else None}
    if progress is not None:
        progress(1.0)
    return FmcwResult(st, rows, clouds, src, getattr(var, 'name', ''), time.perf_counter() - t0, sweeps, checks)


# ======================================================================= colours, overlays
def velocity_rgba(v, scale=None, alpha=0.95) -> np.ndarray:
    """(n, 4) float32 colours of radial velocities: closing blue, receding red, still grey; |v| ≥ scale saturates.
    NaN (outside the velocity range) dark grey."""
    v = np.asarray(v, float)
    fin = np.isfinite(v)
    if scale is None:
        scale = max(float(np.percentile(np.abs(v[fin]), 95)) if fin.any() else 1.0, 1.0)
    t = np.clip(np.where(fin, v, 0.0) / scale, -1.0, 1.0)
    grey = np.array([0.62, 0.62, 0.62])
    red = np.array([0.86, 0.20, 0.16])
    blue = np.array([0.16, 0.42, 0.90])
    a = np.abs(t)[:, None]
    rgb = np.where((t >= 0)[:, None], grey * (1 - a) + red * a, grey * (1 - a) + blue * a)
    rgb[~fin] = (0.25, 0.25, 0.25)
    return np.c_[rgb, np.full(len(v), alpha)].astype(np.float32)


def sweep_velocity_rgba(project, sensor, points, settings=None) -> np.ndarray | None:
    """For the 3D view's lidar points (View ▸ Overlays ▸ Lidar points): colours of an FMCW lidar's sweep of the static
    scene by radial velocity at the study's ego speed (None for a time-of-flight lidar: keep its range colours)."""
    m = project.sensor_model_for(sensor)
    if not is_fmcw(m):
        return None
    st = settings or settings_of(project)
    v_s = sensor_velocity(np.array([st.ego_speed_kmh / 3.6, 0.0, 0.0]), math.radians(st.yaw_rate_dps),
                          sensor.pos())
    vr, _ = radial_velocity(np.asarray(points, float), sensor.pos(), v_s)
    return velocity_rgba(vr, scale=max(st.ego_speed_kmh / 3.6, 1.0))


# ======================================================================= the Issues list
def issues(project) -> list:
    """Cheap (no geometry): an FMCW lidar whose stated velocity range is below the closing speed of the configured
    scene (the ego's speed plus the fastest oncoming actor's): those returns carry no velocity."""
    from . import rules as R
    st = settings_of(project)
    sc = SCENES.get(st.scene) or SCENES['urban']
    oncoming = max((kmh for _, _, _, _, hd, kmh, _ in sc['actors'] if abs(abs(hd) - 180.0) < 30.0), default=0.0)
    closing = (st.ego_speed_kmh + oncoming) / 3.6
    out = []
    for s in project.variant.sensors:
        if s.kind != 'lidar' or not getattr(s, 'visible', True):
            continue
        m = project.sensor_model_for(s)
        if not is_fmcw(m):
            continue
        sp = spec_of(m)
        if math.isfinite(sp.velocity_max) and sp.velocity_max < closing:
            out.append(R.Issue('warning', f'{s.tag} ({m.name}) measures radial velocities up to '
                                          f'{sp.velocity_max:g} m/s; oncoming traffic in the {sc["label"].lower()} '
                                          f'scene closes at {closing:.0f} m/s: those returns carry no velocity.',
                               s.tag, [s.uid], 'fmcw'))
    return out


# ======================================================================= output
def _r(v, d=4):
    return None if v is None or not isinstance(v, (int, float)) or not math.isfinite(float(v)) else round(float(v), d)


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return _r(float(o))
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    return o


def summary(res: FmcwResult) -> dict:
    """Compact figures for the assistant and the batch log."""
    out = {'source': res.source, 'lidars': []}
    for r in res.lidars:
        sm = r['summary']
        sun = {x['condition']: {'tof_range_m': _r(x['tof']['range_m'], 1), 'fmcw_range_m': _r(x['fmcw']['range_m'], 1),
                                'tof_false_per_s': _r(x['tof']['false_per_s'], 1),
                                'fmcw_false_per_s': _r(x['fmcw']['false_per_s'], 3)} for x in r['sun']}
        out['lidars'].append({
            'tag': r['tag'], 'model': r['name'], 'fmcw': r['fmcw'],
            'evaluated_as': None if r['fmcw'] else r['doppler_name'],
            'points_per_s': r['points_per_s'], 'velocity_max_mps': r['velocity_max_mps'],
            'velocity_sigma_mps': r['spec']['velocity_sigma'],
            'ego_error_mean_mps': _r(sm.get('ego_error_mean_mps')), 'moving_recall': _r(sm.get('recall'), 3),
            'moving_precision': _r(sm.get('precision'), 3), 'recall_along_ray': _r(sm.get('recall_along_ray'), 3),
            'sun': sun, 'crosstalk_tof_points_per_s': _r(r['crosstalk']['tof_points_per_s'], 0)})
    if res.checks:
        out['checks'] = _clean(res.checks)
    out['assumptions'] = ASSUMPTIONS
    return out


def to_doc(res: FmcwResult, project=None) -> dict:
    return {
        'schema': SCHEMA, 'project': getattr(project, 'name', ''), 'variant': res.variant,
        'frame': FRAME, 'units': 'metres, m/s, degrees; spectral irradiance W m⁻² nm⁻¹', 'sign': SIGN,
        'settings': res.settings.to_dict(), 'source': _clean(res.source), 'assumptions': ASSUMPTIONS,
        'references': REFERENCES, 'seconds': _r(res.seconds, 2),
        'lidars': [_clean({k: v for k, v in r.items() if k != 'uid'}) for r in res.lidars],
        'sweeps': _clean({t: [{k: v for k, v in m.items() if k != 'objects'} for m in ms]
                          for t, ms in res.sweeps.items()}),
        'checks': _clean(res.checks)}


CSV_COLUMNS = ('tag', 'model', 'fmcw', 'evaluated_as', 'points_per_s', 'velocity_max_mps', 'velocity_sigma_mps',
               'sweeps', 'points', 'ego_error_mean_mps', 'ego_error_p95_mps', 'speed_error_mean_mps', 'moving_points',
               'moving_recall', 'moving_precision', 'recall_along_ray', 'across_ray_points', 'outside_velocity_range')
SUN_COLUMNS = ('tag', 'condition', 'klx_approx', 'tof_range_m', 'tof_range_change', 'tof_false_per_s',
               'fmcw_range_m', 'fmcw_range_change', 'fmcw_false_per_s', 'sun_points_per_frame',
               'fmcw_photons_per_mode_sun', 'tof_crosstalk_points_per_s', 'fmcw_crosstalk_points_per_s')


def to_csv(res: FmcwResult) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator='\n')
    w.writerow(CSV_COLUMNS)
    for r in res.lidars:
        sm = r['summary']
        w.writerow([r['tag'], r['name'], r['fmcw'], '' if r['fmcw'] else r['doppler_name'], round(r['points_per_s']),
                    _blank(r['velocity_max_mps']), _blank(_r(r['spec']['velocity_sigma']))] +
                   [_blank(_r(sm.get(k))) for k in ('sweeps', 'points', 'ego_error_mean_mps', 'ego_error_p95_mps',
                                                    'speed_error_mean_mps', 'moving_points', 'recall', 'precision',
                                                    'recall_along_ray', 'across_ray_points',
                                                    'outside_velocity_range')])
    return buf.getvalue()


def sun_csv(res: FmcwResult) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator='\n')
    w.writerow(SUN_COLUMNS)
    for r in res.lidars:
        ct = r['crosstalk']
        for x in r['sun']:
            w.writerow([r['tag'], x['condition'], x['klx_approx'], _r(x['tof']['range_m'], 2),
                        _r(x['tof']['range_change']), _r(x['tof']['false_per_s'], 2), _r(x['fmcw']['range_m'], 2),
                        _r(x['fmcw']['range_change']), _r(x['fmcw']['false_per_s'], 4), x['sun_points_per_frame'],
                        _r(x['fmcw']['photons_per_mode_sun']), _r(ct['tof_points_per_s'], 1),
                        _r(ct['fmcw_points_per_s'], 1)])
    return buf.getvalue()


def _blank(v):
    return '' if v is None else v


# ---- the 4D cloud as files
CLOUD_FIELDS = (('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('velocity', 'f4'), ('velocity_comp', 'f4'),
                ('moving', 'u1'), ('instance', 'u2'))
ROS_FIELDS = 'x y z (float32, offsets 0/4/8), intensity (float32, 12), velocity (float32, 16: radial m/s, ' + SIGN + \
    ', NaN outside the velocity range); point_step 20'


def cloud_array(cl: Cloud) -> np.ndarray:
    """The cloud as a structured array (CLOUD_FIELDS), vehicle frame."""
    a = np.zeros(len(cl.P), np.dtype(list(CLOUD_FIELDS)))
    a['x'], a['y'], a['z'] = cl.P[:, 0], cl.P[:, 1], cl.P[:, 2]
    a['velocity'], a['velocity_comp'] = cl.v_r, cl.v_comp
    a['moving'] = cl.moving.astype(np.uint8)
    a['instance'] = np.clip(cl.inst, 0, 65535).astype(np.uint16)
    return a


def ros_points(cl: Cloud, intensity=None) -> np.ndarray:
    """The proposed sensor_msgs/PointCloud2 layout of a 4D cloud (ROS_FIELDS) as a structured array for
    core.rosbag.pointcloud2."""
    dt = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('intensity', '<f4'), ('velocity', '<f4')])
    a = np.zeros(len(cl.P), dt)
    a['x'], a['y'], a['z'] = cl.P[:, 0], cl.P[:, 1], cl.P[:, 2]
    a['intensity'] = 0.0 if intensity is None else np.asarray(intensity, np.float32)
    a['velocity'] = cl.v_r
    return a


def pcd_bytes(cl: Cloud) -> bytes:
    """Binary PCD v0.7 with x y z velocity velocity_comp moving instance."""
    a = cloud_array(cl)
    t = {'f': 'F', 'u': 'U', 'i': 'I'}
    hdr = ('# .PCD v0.7 - Point Cloud Data file format (Vantage FMCW lidar: ' + SIGN + ')\nVERSION 0.7\n'
           f'FIELDS {" ".join(n for n, _ in CLOUD_FIELDS)}\n'
           f'SIZE {" ".join(str(np.dtype(f).itemsize) for _, f in CLOUD_FIELDS)}\n'
           f'TYPE {" ".join(t[np.dtype(f).kind] for _, f in CLOUD_FIELDS)}\n'
           f'COUNT {" ".join("1" for _ in CLOUD_FIELDS)}\nWIDTH {len(a)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n'
           f'POINTS {len(a)}\nDATA binary\n')
    return hdr.encode('ascii') + a.astype(a.dtype.newbyteorder('<')).tobytes()


def ply_bytes(cl: Cloud) -> bytes:
    """Binary little-endian PLY with vertex properties x y z velocity velocity_comp moving instance."""
    a = cloud_array(cl)
    ply = {'f4': 'float', 'u1': 'uchar', 'u2': 'ushort'}
    hdr = ('ply\nformat binary_little_endian 1.0\ncomment Vantage FMCW lidar 4D cloud, vehicle frame (ISO 8855), '
           f'velocity: radial m/s, {SIGN}\nelement vertex {len(a)}\n' +
           ''.join(f'property {ply[f]} {n}\n' for n, f in CLOUD_FIELDS) + 'end_header\n')
    return hdr.encode('ascii') + a.astype(a.dtype.newbyteorder('<')).tobytes()


def read_ply(data: bytes) -> np.ndarray:
    """This module's PLY back as a structured array (for round trips)."""
    end = data.index(b'end_header\n') + len(b'end_header\n')
    hdr = data[:end].decode('ascii').splitlines()
    n = int(next(h for h in hdr if h.startswith('element vertex')).split()[-1])
    back = {'float': '<f4', 'uchar': 'u1', 'ushort': '<u2'}
    dt = np.dtype([(h.split()[2], back[h.split()[1]]) for h in hdr if h.startswith('property')])
    return np.frombuffer(data, dt, count=n, offset=end)


def files(res: FmcwResult, project=None) -> list:
    """[(name, bytes)]: fmcw.json, fmcw_lidars.csv, fmcw_sun.csv and per lidar its 4D cloud as fmcw_<tag>.pcd and
    fmcw_<tag>.ply."""
    out = [('fmcw.json', (json.dumps(to_doc(res, project), indent=1, allow_nan=False) + '\n').encode('utf-8')),
           ('fmcw_lidars.csv', to_csv(res).encode('utf-8')), ('fmcw_sun.csv', sun_csv(res).encode('utf-8'))]
    for tag, cl in res.clouds.items():
        if cl is None:
            continue
        safe = ''.join(ch if ch.isalnum() or ch in '-_' else '_' for ch in tag)
        out += [(f'fmcw_{safe}.pcd', pcd_bytes(cl)), (f'fmcw_{safe}.ply', ply_bytes(cl))]
    return out


def export(res: FmcwResult, project, out_dir) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for name, data in files(res, project):
        path = os.path.join(out_dir, name)
        with open(path, 'wb') as fh:
            fh.write(data)
        paths.append(path)
    return paths


def write_bag(res: FmcwResult, out_dir, frame_prefix='') -> str | None:
    """A ROS 2 bag (MCAP) with each lidar's 4D cloud on /<tag>/points4d in the proposed layout (ROS_FIELDS)."""
    from . import rosbag as RB
    if not any(cl is not None for cl in res.clouds.values()):
        return None
    path = os.path.join(out_dir, 'fmcw_4d_bag')
    w = RB.BagWriter(path, overwrite=True)
    try:
        for tag, cl in res.clouds.items():
            if cl is None:
                continue
            topic = f'/{"".join(ch if ch.isalnum() else "_" for ch in tag.lower())}/points4d'
            w.add_topic(topic, 'sensor_msgs/msg/PointCloud2')
            w.write(topic, 0, RB.pointcloud2(ros_points(cl), 0, f'{frame_prefix}base_link'))
    finally:
        w.close()
    return path


# ======================================================================= validation (for vantage/validation)
def _check_ego_fit(ctx=None):
    """Synthetic static points around a sensor moving at a known velocity, with the datasheet noise and 20 % movers:
    the fitted velocity against the truth, in units of its own predicted 1 σ."""
    rng = np.random.default_rng(7)
    n = 4000
    az = rng.uniform(-math.pi, math.pi, n)
    el = rng.uniform(-0.4, 0.2, n)
    D = np.c_[np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)]
    v_true = np.array([13.9, 0.42, -0.05])
    sig = 0.03
    vr = -D @ v_true + rng.normal(0, sig, n)
    mov = rng.random(n) < 0.2
    vr[mov] += rng.uniform(-15, 15, mov.sum())
    fit = fit_sensor_velocity(D, vr, sig, seed=3)
    err = fit['v'] - v_true
    sd = np.sqrt(np.diag(fit['cov']))
    return float(np.max(np.abs(err) / sd)), {'v_true': v_true.tolist(), 'v_fit': fit['v'].tolist(),
                                              'sigma_fit': sd.tolist(), 'inliers': fit['n_inliers']}


def _check_box_doppler(ctx=None):
    """A box 20 m ahead closing at 10 m/s on a sensor moving at 15 m/s, turning at 10°/s: the radial velocity of its
    centre-facing point against the closed form d · (v + ω × r − v_s)."""
    c = np.array([20.0, 2.0, 0.8])
    v = np.array([-10.0, 0.5, 0.0])
    w = math.radians(10.0)
    sp = np.array([1.5, 0.0, 1.9])
    vs = sensor_velocity([15.0, 0.0, 0.0], w * 0.5, sp)
    P = np.array([[c[0] - 2.25, c[1], c[2]], [c[0] - 2.25, c[1] + 0.9, c[2]]])
    vp = rigid_velocity(P, c, v, w)
    got, _ = radial_velocity(P, sp, vs, vp)
    want = []
    for p in P:
        d = (p - sp) / np.linalg.norm(p - sp)
        r = p - c
        vpt = v + np.array([-w * r[1], w * r[0], 0.0])
        vsx = np.array([15.0 - (w * 0.5) * sp[1], (w * 0.5) * sp[0], 0.0])
        want.append(float(d @ (vpt - vsx)))
    return float(np.max(np.abs(np.asarray(got) - np.asarray(want)))), {'got': got.tolist(), 'want': want}


def _check_solar_planck(ctx=None):
    """G173's extraterrestrial irradiance around 905 nm and 1550 nm against a 5772 K blackbody sun at 1 au (relative
    difference, largest of the two bands), and the photons per mode of the sun's disc at 1550 nm (from G173 direct ×
    the air mass) against Planck's occupation at the sun's 1550 nm brightness temperature."""
    out = {}
    worst = 0.0
    for lo, hi in ((895, 915), (1540, 1560)):
        b = band_irradiance(lo, hi)
        bb = planck_irradiance_toa((lo + hi) / 2 * 1e-9)
        e = abs(b['etr'] / bb - 1.0)
        out[f'{lo}-{hi}'] = {'g173_etr': b['etr'], 'planck': bb, 'rel': e}
        worst = max(worst, e)
    b = band_irradiance(1540, 1560)
    n_g173 = occupancy(b['etr'] / OMEGA_SUN * 1e9, 1550e-9)
    n_bb = planck_occupancy(1550e-9, T_SUN_K)
    out['occupancy_1550'] = {'g173': n_g173, 'planck_5772K': n_bb, 'rel': abs(n_g173 / n_bb - 1),
                             'brightness_temperature_K': H * C / (1550e-9 * KB * math.log1p(1.0 / n_g173))}
    worst = max(worst, abs(n_g173 / n_bb - 1))
    return worst, out


def _check_tof_background(ctx=None):
    """The receiver's background on a 30 % wall facing the sun at noon: tof_background against N = η ρ E Δλ A Ω λ/(πhc)
    written out here, and the 4th-root range law in the background-limited limit."""
    st = FmcwSettings()
    b = band_irradiance(895, 915)
    cond = next(c for c in CONDITIONS if c[0] == 'noon_ahead')
    sk = sky(cond, b)
    n = -sk['s'] * np.array([1, 1, 0])
    n = n / np.linalg.norm(n)
    E = float(surface_irradiance(-n[None] * -1, sk)[0])
    got = float(tof_background(np.array([0.3 * E / math.pi]), st)[0])
    A = math.pi * 0.01 ** 2
    want = 0.7 * 0.2 * 0.3 * E * 20.0 * A * (1.5e-3) ** 2 * 905e-9 / (math.pi * H * C)     # optics × SPAD PDE
    return abs(got / want - 1.0), {'got_e_per_s': got, 'closed_form_e_per_s': want, 'E_W_m2_nm': E}


def validation_checks() -> list:
    """Spec-compatible check descriptions for vantage/validation (the lead registers them under a model 'fmcw'): each
    'fn' returns (error, detail); wrap in suite.Outcome(error, detail)."""
    return [
        {'id': 'fmcw.ego_fit', 'model': 'fmcw', 'title': 'Ego velocity from one sweep of static points',
         'quantity': 'largest error of the fitted sensor velocity per axis in units of its own 1 σ, 4000 rays, 3 cm/s '
                     'noise, 20 % movers', 'reference': 'the known velocity the synthetic points were made with',
         'tolerance': 4.0, 'unit': 'σ', 'refs': ('rep103',), 'fn': _check_ego_fit, 'cost': 'cheap'},
        {'id': 'fmcw.box_doppler', 'model': 'fmcw', 'title': 'Doppler of a known moving box',
         'quantity': 'largest difference of the radial velocity of two points of a closing, yawing box seen from a '
                     'turning sensor', 'reference': 'd · (v + ω × r − v_s) written out in the check',
         'tolerance': 1e-9, 'unit': 'm/s', 'refs': ('iso8855',), 'fn': _check_box_doppler, 'cost': 'cheap'},
        {'id': 'fmcw.solar_planck', 'model': 'fmcw', 'title': 'Solar irradiance and photons per mode against Planck',
         'quantity': 'largest relative difference of G173\'s extraterrestrial band irradiance (905, 1550 nm) and of '
                     'the sun\'s photons per mode at 1550 nm from a 5772 K blackbody',
         'reference': 'Planck\'s law for the IAU 2015 nominal sun at 1 au. The sun is not a blackbody: near 1.6 µm '
                      '(the H⁻ opacity minimum) the photosphere is seen deeper and hotter, a brightness temperature '
                      'some 500 K above T_eff (G173 gives 6320 K at 1550 nm), so 25 % there; 905 nm agrees to 0.2 %',
         'tolerance': 0.25, 'unit': 'relative', 'refs': ('astm_g173', 'iau2015', 'siegman1966'),
         'fn': _check_solar_planck, 'cost': 'cheap'},
        {'id': 'fmcw.tof_background', 'model': 'fmcw', 'title': 'Time-of-flight solar background',
         'quantity': 'relative difference of the background photoelectron rate on a sunlit 30 % wall',
         'reference': 'the Lambertian closed form N = η ρ E Δλ A Ω λ / (π h c)', 'tolerance': 1e-9,
         'unit': 'relative', 'refs': ('astm_g173', 'kasten1989'), 'fn': _check_tof_background, 'cost': 'cheap'},
    ]
