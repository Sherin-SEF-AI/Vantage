"""Neuromorphic event cameras (Module.modality 'event'): the makers' products, a per-pixel event model, the event
formats and the studies an event camera on a data rig needs: its event rate and bandwidth over a drive, its motion
blur against the RGB cameras' smear, and a tunnel exit, where its dynamic range keeps contrast that an RGB camera's
exposure loses. CARLA's DVS camera (sensor.camera.dvs) is recorded by the bridge beside a reference RGB view of the
same pose, and compare_dvs() checks this model against it.

The library (library('modules'), library('lenses'); core.lens adds them to the built-in library)
-----------------------------------------------------------------------------------------------
Figures as the makers state them on their public briefs (read 2026-11-11; each module's note names its sources; a
figure not stated is absent from modality_params, never guessed):
* Prophesee EVK4 HD with the Sony IMX636: 1280 × 720, 4.86 µm, 1/2.5"; nominal contrast threshold 25 %; latency
  < 100 µs at 1000 lux and < 1000 µs at 5 lux; dynamic range > 86 dB (5 lux–100 klux) and > 120 dB (80 mlux–100
  klux; low-light cut-off 0.08 lux, not guaranteed); EVT 2.1 and EVT 3.0 output (Sony/Prophesee IMX636-AAMR-C
  product brief 2024 v3.0, p. 1). The kit: 30 × 30 × 36 mm, 40 g, typically 0.5 W over USB 3.0, camera bandwidth
  1.6 Gbit/s, a C-mount f/2.0 lens of 47.7° diagonal (Prophesee EVK4 HD brief 2026, p. 2; it also gives a single
  latency of 220 µs without its light level). The sensor's readout: 1066 Mev/s, 1 µs timestamps, NCT 15.7 % contrast,
  low-light cut-off 40 mlx (Prophesee/Sony press release of ISSCC 2020 paper 5.10, Finateu et al., "Key
  Specifications", p. 4, read 2026-11-11: the sensor the IMX636 productises).
* Prophesee EVK5 HD with the Sony IMX646: 1280 × 720, 4.86 µm; latency 800 µs at 1000 lux (EVK5 HD brief 2023,
  p. 2) and < 9000 µs at 3 lux; dynamic range > 110 dB (30 mlux–100 klux) (Prophesee's IMX646 page); 35 × 35 × 35.27
  mm, 95 g, typically 0.5 W, 1.6 Gbit/s; lenses of 47.7° and 158.9° diagonal. The contrast threshold is not stated.
* Prophesee EVK3 with the GenX320: 320 × 320, 6.3 µm, 1/5"; dynamic range > 140 dB, low-light cut-off 0.05 lux,
  1 µs timestamps; 10 Mev/s in MIPI streaming (the sensor's power-mode table, 22.8 mW); an M12 lens of 1.8 mm f/2.8,
  58° H and V, 76° D (GenX320 DICE brief 2025, pp. 1–2; EVK3 GenX320 COB brief 2026, p. 1). Contrast threshold and
  latency not stated.

The event model (EventSim)
--------------------------
A pixel's photoreceptor follows log intensity L = ln(I); it holds a reference L_ref and emits an ON event each time
L − L_ref reaches +θ_on and an OFF event at −θ_off, moving L_ref by the threshold each time. Between two frames L is
interpolated linearly in time and every crossing gets its own time (ESIM: H. Rebecq, D. Gehrig, D. Scaramuzza,
"ESIM: an Open Event Camera Simulator", CoRL 2018; the algorithm of CARLA's DVS camera, DVSCamera.cpp). On top, as
v2e (Y. Hu, S.-C. Liu, T. Delbruck, "v2e: From Video Frames to Realistic DVS Events", CVPRW 2021, arXiv:2006.07722;
MIT licence; v2ecore/emulator.py and emulator_utils.py read 2026-11-11):
* threshold mismatch: θ per pixel ~ N(θ, σ), at least 0.01, drawn once (fixed pattern). No maker states σ: the
  default 0.03 ln units is v2e's default (sigma_thres); DVS128 measured 2.1 % contrast (Lichtsteiner, Posch,
  Delbruck, IEEE JSSC 43(2), 2008);
* refractory period: an event within it of the pixel's last is not emitted (the reference still moves, as ESIM);
  makers state only that the dead time is adjustable: 0 unless set;
* latency and bandwidth: a first-order low-pass on L whose time constant is the stated latency at the pixel's
  sensor-plane illuminance (log-log interpolation between the stated points): a photoreceptor's bandwidth grows with
  its photocurrent (v2e scales its cut-off with intensity the same way). Below the low-light cut-off a pixel has no
  signal; above the top of the stated range it saturates;
* leak events: the reference drifts down by leak_hz·θ per second, so a still pixel emits ON events at about leak_hz
  (v2e subtract_leak_current; 0.1 Hz v2e's 'noisy' preset);
* shot noise: Poisson ON and OFF events at shot_hz/2 each, scaled by (f − 1)·i + 1 with i the pixel's intensity on
  0…1 and f = 0.25, so dark pixels are noisiest (v2e generate_shot_noise; 5 Hz is v2e's 'noisy' preset). Shot events
  do not move the reference and ignore the refractory period, as v2e;
* readout: timestamps on a 1 µs grid; the sensor's (or its link's) maximum event rate caps the output: 'erc' drops
  events evenly within each window above the cap (Prophesee's event-rate controller), 'fifo' serves events at the
  maximum rate from a buffer, delays them, and drops what overflows it.
Sensor-plane illuminance from scene luminance: E = π·L·T / (4·N²) (lens transmission T, f-number N; cos⁴ fall-off
not modelled).

Event formats (Prophesee's documented layouts, docs.prophesee.ai, read 2026-11-11): EVT 2.0, one 32-bit word per
event (type 4, time 6 LSB, x 11, y 11) plus a TIME_HIGH word whenever t >> 6 changes; EVT 2.1, one 64-bit word per
(µs, row, polarity, 32-column group) with a 32-bit validity mask; EVT 3.0, 16-bit words (ADDR_Y, ADDR_X, VECT_BASE_X,
VECT_12, VECT_8, TIME_LOW, TIME_HIGH), runs of neighbouring events vectorised. encode_evt2/decode_evt2 and
encode_evt3/decode_evt3 write and read them; a file's header follows Metavision's RAW files ('% key value' lines,
'% end').

Limits: the model is a model of a DVS pixel, not of the IMX636's circuit; its noise and mismatch figures are v2e's
typical ones (no maker states them); the readout timing (arbiter order, row-level timestamps) is not modelled.
CARLA's DVS camera works on the rendered 8-bit (display-referred) image, not on linear light (compare_dvs measures
the difference).

Measured on CARLA 0.9.16 (round 17, Town03, 320 × 180, θ = ln 1.25): CARLA stamps the events on the simulation clock;
at exposure compensation 0 its DVS and this model in CARLA's algorithm on a reference RGB view at the same pose agree
to +0.23 % in count on a drive (per-pixel r = 0.9995; event times within ±138 µs, interquartile), +0.6 % standing and
+0.31 % at 50 km/h; with both at −2.79 EV the DVS made 0.52× the model's events (same pattern, r = 0.994): the DVS
camera probably does not apply exposure_compensation as the RGB camera does, so the bridge records event cameras at
the rig's exposure_ev (0 by default). The model in linear light makes several times the DVS's events (8.3× on the
drive): a physical sensor sees log changes of linear light, larger than a gamma-encoded image's.
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

from . import lens as L
# the event model itself lives beside the CARLA bridge, which runs it in CARLA's own Python (vantage.sim.dvs_model)
from ..sim.dvs_model import (  # noqa: F401  (re-exported: this module's names as before)
    CARLA_DVS_DEFAULTS, CARLA_DVS_DTYPE, CARLA_GREY, COMPARE_SCHEMA, EVENTS_SCHEMA, EVT2_CD_OFF, EVT2_CD_ON,
    EVT2_TIME_HIGH, MIN_THRESHOLD, NEVER, REC709, SHOT_INTEN_FACTOR, Events, EventSim, SimConfig, carla_log_grey,
    compare_dvs, encode_evt2, linear_log_luminance, raw_header, rate_series)

SCHEMA = 'vantage.event_camera/1'
FRAME = 'vehicle (ISO 8855, rear axle): x forward, y left, z up'
MODALITY = 'event'
CHECKED = '2026-11-11'
FORMATS = ('EVT3.0', 'EVT2.1', 'EVT2.0')
FORMAT_LABEL = {'EVT3.0': 'EVT 3.0', 'EVT2.1': 'EVT 2.1', 'EVT2.0': 'EVT 2.0'}
BANDWIDTH_MODES = ('erc', 'fifo')
EVENT_DTYPE = np.dtype([('x', '<u2'), ('y', '<u2'), ('t', '<i8'), ('p', 'i1')])

ASSUMPTIONS = (
    'The event model is ESIM with v2e\'s threshold mismatch, leak and shot noise (typical figures: no maker states '
    'them); a pixel\'s time constant is its stated latency at its illuminance; the readout\'s arbiter and row-level '
    'timestamps are not modelled. The event rate over a drive is estimated from Vantage\'s synthetic parking scene '
    'rendered at a fraction of the sensor\'s resolution and scaled by the pixel count (texture finer than the rendered '
    'pixels is missing: a lower bound). Scene luminance: the synthetic scene\'s reflectances times scene_white_cd. The '
    'tunnel exit uses the exposure metering study\'s tunnel (luminances, sun, portal) with a stopped car ahead; an RGB '
    'camera\'s auto exposure follows its metered log-mean with a first-order lag (ae_tau_s, assumed) and its HDR range '
    'is the exposure metering study\'s model of the module\'s stated dynamic range.')


# =========================================================================== the library
_IMX636_BRIEF = 'https://prophesee.ai/wp-content/uploads/2024/05/IMX636-Product-Brief-2024-v3.0.pdf'
_EVK4_BRIEF = 'https://www.prophesee.ai/wp-content/uploads/2026/03/EVK4-HD-Prophesee-Evaluation-Kit-Brief-2026.pdf'
_ISSCC_PR = ('https://www.prophesee.ai/wp-content/uploads/2020/02/EMBARGO-FEB-19-7am-CET-PROPHESEE-SONY-Event-based-'
             'Vision-Sensor-.pdf')
_EVK5_BRIEF = 'https://prophesee.ai/wp-content/uploads/2023/09/EVK5-HD-Prophesee-Evaluation-Kit-Brief-2023-EN.pdf'
_IMX646_PAGE = 'https://prophesee.ai/event-based-sensor-imx646-sony-prophesee'
_GENX320_DICE = 'https://prophesee.ai/wp-content/uploads/2025/01/GENX320-Product-Brief-2025-DICE-OK.pdf'
_EVK3_320 = 'https://www.prophesee.ai/wp-content/uploads/2026/03/GENX320-Product-Brief-2026-EVK3-COB-OK.pdf'

# modality_params keys (absent = not stated by the maker)
PARAM_KEYS = {
    'contrast_threshold_pct': 'nominal contrast threshold, % change of illuminance (θ = ln(1 + pct/100))',
    'contrast_sensitivity_nct_pct': 'measured nominal contrast threshold (NCT) of the sensor, % contrast',
    'latency_us': 'pixel latency upper bounds [[lux, µs], …] at the sensor-plane illuminance',
    'dynamic_range_db': 'dynamic range the maker guarantees, dB', 'dynamic_range_lux': '[low, high] lux of it',
    'dynamic_range_max_db': 'the widest dynamic range stated (not guaranteed), dB',
    'dynamic_range_max_lux': '[low, high] lux of it', 'low_light_cutoff_lux': 'the low-light cut-off, lux',
    'pixel_pitch_um': 'pixel pitch, µm', 'resolution': '[width, height] pixels',
    'max_event_rate_mevs': 'the sensor\'s maximum event rate, million events per second',
    'link_gbps': 'the camera\'s maximum output bandwidth, Gbit/s',
    'refractory_us': 'refractory (dead) time, µs', 'event_formats': 'output formats',
    'timestamp_us': 'timestamp resolution, µs', 'sources': 'where each figure comes from'}


def _module(mid, name, sensor, res, pixel_um, fmt, *, url, note, size, weight, power, interface, params, dr_db,
            sync=True, temp=()):
    return L.Module(id=mid, name=name, vendor='Prophesee', sensor=sensor, width_px=res[0], height_px=res[1],
                    pixel_um=pixel_um, optical_format=fmt, interface=interface, fps=0.0, hw_sync=sync,
                    size_mm=tuple(float(v) for v in size), weight_g=float(weight), cables_m=(3.0,),
                    power_w=float(power), builtin=True, shutter='', dynamic_range_db=float(dr_db), ip_rating='',
                    temp_c=tuple(temp), chroma='monochrome', url=url,
                    note=(note + f' An event camera records no frames (frame rate 0): its data rate is its event rate '
                                 f'(Analysis ▸ Event cameras). Source: {url} (read {CHECKED}). USB 3.0 cable: 3 m '
                                 'typical (the kit\'s length is not stated).'),
                    modality=MODALITY, modality_params=params)


def _modules() -> list:
    imx636 = {
        'contrast_threshold_pct': 25.0, 'contrast_sensitivity_nct_pct': 15.7,
        'latency_us': [[1000.0, 100.0], [5.0, 1000.0]],
        'dynamic_range_db': 86.0, 'dynamic_range_lux': [5.0, 100000.0],
        'dynamic_range_max_db': 120.0, 'dynamic_range_max_lux': [0.08, 100000.0], 'low_light_cutoff_lux': 0.08,
        'pixel_pitch_um': 4.86, 'resolution': [1280, 720], 'max_event_rate_mevs': 1066.0, 'link_gbps': 1.6,
        'event_formats': ['EVT2.1', 'EVT3.0'], 'timestamp_us': 1.0,
        'sources': {
            'contrast_threshold_pct, latency_us, dynamic_range_*, low_light_cutoff_lux, pixel_pitch_um, resolution, '
            'event_formats': f'IMX636-AAMR-C product brief 2024 v3.0, p. 1 ({_IMX636_BRIEF})',
            'link_gbps': f'EVK4 HD brief 2026, p. 2 ({_EVK4_BRIEF}); it also states 220 µs latency without a light '
                         'level',
            'max_event_rate_mevs, timestamp_us, contrast_sensitivity_nct_pct':
                f'Prophesee/Sony press release of ISSCC 2020 paper 5.10 (Finateu et al.), Key Specifications, p. 4 '
                f'({_ISSCC_PR})'}}
    imx646 = {
        'latency_us': [[1000.0, 800.0], [3.0, 9000.0]], 'dynamic_range_db': 110.0,
        'dynamic_range_lux': [0.03, 100000.0], 'pixel_pitch_um': 4.86, 'resolution': [1280, 720],
        'link_gbps': 1.6,
        'sources': {'latency_us, dynamic_range_*': f'Prophesee\'s IMX646 page ({_IMX646_PAGE}); EVK5 HD brief 2023, '
                                                   f'p. 2 ({_EVK5_BRIEF}) gives 800 µs at 1000 lux and > 110 dB',
                    'link_gbps, pixel_pitch_um, resolution': f'EVK5 HD brief 2023, p. 2 ({_EVK5_BRIEF})'}}
    genx320 = {
        'dynamic_range_db': 140.0, 'low_light_cutoff_lux': 0.05, 'pixel_pitch_um': 6.3, 'resolution': [320, 320],
        'max_event_rate_mevs': 10.0, 'timestamp_us': 1.0,
        'sources': {'all': f'GenX320 DICE brief 2025, pp. 1–2 ({_GENX320_DICE}): > 140 dB, 0.05 lux cut-off, 1 µs '
                           'row-level timestamps, 10 Mev/s in MIPI streaming mode (22.8 mW); EVK3 GenX320 COB '
                           f'brief 2026 ({_EVK3_320})'}}
    return [
        _module('prophesee_evk4_hd', 'EVK4 HD', 'Sony IMX636', (1280, 720), 4.86, '1/2.5"', url=_EVK4_BRIEF,
                interface='USB 3.0', size=(30.0, 30.0, 36.0), weight=40.0, power=0.5, dr_db=86.0, params=imx636,
                note=('Prophesee\'s evaluation kit with the Sony IMX636 event sensor: figures from the IMX636-AAMR-C '
                      'product brief (2024 v3.0), the EVK4 HD brief (2026; 30 × 30 × 36 mm, 40 g without lens, '
                      'typ. 0.5 W over USB, trigger in and sync in/out) and the ISSCC 2020 key specifications (readout '
                      '1066 Mev/s). Dynamic range > 86 dB guaranteed from 5 lux, > 120 dB down to the 0.08 lux cut-off.')),
        _module('prophesee_evk5_hd', 'EVK5 HD', 'Sony IMX646', (1280, 720), 4.86, '1/2.5"', url=_EVK5_BRIEF,
                interface='USB 3.0', size=(35.0, 35.0, 35.27), weight=95.0, power=0.5, dr_db=110.0, params=imx646,
                note=('Prophesee\'s evaluation kit with the Sony IMX646: the EVK5 HD brief (2023; 95 ± 2 g without '
                      'lens, typ. 0.5 W) and Prophesee\'s IMX646 page (latency < 800 µs at 1000 lux, < 9000 µs at 3 '
                      'lux; > 110 dB from 30 mlux). The contrast threshold and the maximum event rate are not '
                      'stated.')),
        _module('prophesee_evk3_genx320', 'EVK3 GenX320', 'Prophesee GenX320', (320, 320), 6.3, '',
                url=_EVK3_320, interface='USB 3.0', size=(108.0, 76.0, 45.0), weight=0.0, power=0.0, dr_db=140.0,
                params=genx320, sync=True,
                note=('Prophesee\'s GenX320 (320 × 320, 6.3 µm, > 140 dB, 0.05 lux cut-off) on its EVK3 board '
                      '(108 × 76 × 45 mm, PCBs only, USB powered; synchronisation on SMA): the GenX320 DICE brief '
                      '(2025) and the EVK3 COB brief (2026). The sensor streams up to 10 Mev/s over MIPI (22.8 mW); '
                      'the kit\'s power, the contrast threshold and the latency are not stated. Prophesee calls it a '
                      '1/5" sensor; its 320 × 6.3 µm array is 2.85 mm across, 16 % under that type\'s nominal 3.4 mm, so '
                      'the optical format is left blank here (the library checks format against array).')),
    ]


def library(what='modules') -> list:
    """The event cameras for core.lens's built-in library: what = 'modules' or 'lenses'."""
    mods = _modules()
    if what == 'modules':
        return mods
    if what != 'lenses':
        return []
    M = {m.id: m for m in mods}
    return [
        L._vlens('prophesee_evk4_lens', 'C-mount f/2.0 (47.7° D)', M['prophesee_evk4_hd'], sheet=(0.0, 0.0, 47.7),
                 fno=2.0, label='narrow', url=_EVK4_BRIEF,
                 basis='The kit\'s lens (Soyo Security 1/2.5" f/2.0 C-mount, EVK4 HD brief 2026): only its diagonal '
                       'field is stated.'),
        L._vlens('prophesee_evk5_lens_48', 'CS-mount (47.7° D)', M['prophesee_evk5_hd'], sheet=(0.0, 0.0, 47.7),
                 label='narrow', url=_EVK5_BRIEF, basis='The EVK5 HD brief lists a 47.7° diagonal lens (aperture '
                                                        'not stated).'),
        L._vlens('prophesee_evk5_lens_159', 'CS-mount wide (158.9° D)', M['prophesee_evk5_hd'],
                 sheet=(0.0, 0.0, 158.9), label='wide', url=_EVK5_BRIEF,
                 basis='The EVK5 HD brief lists a 158.9° diagonal lens (aperture not stated).'),
        L._vlens('prophesee_evk3_genx320_lens', 'SND2636A1SD-8M M12 (1.8 mm)', M['prophesee_evk3_genx320'],
                 sheet=(58.0, 58.0, 76.0), focal=1.8, fno=2.8, url=_EVK3_320,
                 basis='The EVK3 GenX320 COB brief (2026): 1.8 mm, f/2.8 fixed iris, 58° H and V, 76° D, no IR-cut '
                       'filter.'),
    ]


def is_event(module) -> bool:
    return getattr(module, 'modality', 'rgb') == MODALITY


def _num(v, default=None):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


# =========================================================================== settings
LIMITS = {'speed_kmh': (0.0, 250.0), 'object_m': (0.2, 500.0), 'duration_s': (0.02, 2.0),
          'render_scale': (0.05, 1.0), 'max_step_px': (0.1, 4.0), 'max_frames': (4, 2000),
          'scene_white_cd': (1.0, 1e6), 'f_number': (0.0, 32.0), 'lens_transmission': (0.05, 1.0),
          'threshold_pct': (1.0, 200.0), 'sigma_ln': (0.0, 0.5), 'refractory_us': (0.0, 1e5),
          'latency_us': (1.0, 1e6), 'shot_noise_hz': (0.0, 1e4), 'leak_hz': (0.0, 1e3),
          'fifo_events': (16, 10 ** 9), 'erc_window_us': (1.0, 1e6), 'edge_contrast': (0.01, 0.99),
          'exposure_ms': (0.0, 100.0), 'tunnel_pre_s': (0.1, 10.0), 'tunnel_post_s': (0.1, 10.0),
          'tunnel_hz': (20.0, 5000.0), 'tunnel_scale': (0.05, 1.0), 'supersample': (1, 4), 'ae_tau_s': (0.0, 10.0),
          'target_m': (2.0, 300.0), 'target_albedo': (0.01, 0.95), 'min_contrast': (0.001, 1.0),
          'min_target_events': (1, 10 ** 6), 'window_ms': (0.1, 1000.0), 'lamp_cd': (0.0, 1e5),
          'activity_hz': (0.0, 1e5), 'link_warn': (0.1, 1.0), 'virtual_pitch_deg': (-60.0, 90.0),
          'virtual_offset_m': (-2.0, 2.0), 'seed': (0, 999999), 'evt3_bits': (0.0, 256.0)}
INTS = ('max_frames', 'fifo_events', 'min_target_events', 'seed', 'supersample')
BOOLS = ('photoreceptor',)
STRS = {'bandwidth_mode': BANDWIDTH_MODES, 'event_format': FORMATS}


@dataclass
class EventSettings:
    speed_kmh: float = 50.0            # driving speed of the rate and blur studies
    object_m: float = 5.0              # the blur study's static object distance (as the sync plan's)
    duration_s: float = 0.25           # the rendered drive of the rate study
    render_scale: float = 0.25         # its rendering: this fraction of the sensor's width and height
    max_step_px: float = 0.5           # frames close enough that nothing moves more than this between them
    max_frames: int = 400
    scene_white_cd: float = 10000.0    # luminance of a white surface in the rendered scene (daylight, assumed)
    f_number: float = 0.0              # 0: the lens's (f/2.0 when it states none)
    lens_transmission: float = 0.9     # typical
    threshold_pct: float = 25.0        # used when the module states no contrast threshold (the IMX636's nominal)
    sigma_ln: float = 0.03             # threshold mismatch, ln units (v2e default; no maker states it)
    refractory_us: float = 0.0         # used when the module states none (makers: adjustable dead time)
    latency_us: float = 500.0          # used when the module states no latency (assumed)
    shot_noise_hz: float = 5.0         # v2e 'noisy'
    leak_hz: float = 0.1               # v2e 'noisy'
    photoreceptor: bool = True         # the latency low-pass on (off: ideal, instantaneous pixels)
    bandwidth_mode: str = 'erc'        # erc | fifo
    fifo_events: int = 65536           # the fifo's buffer, events
    erc_window_us: float = 200.0       # the event-rate controller's window
    event_format: str = 'EVT3.0'       # the format the data rate is quoted in
    edge_contrast: float = 0.5         # the blur study's edge: dark side 1 − this of the bright side
    exposure_ms: float = 0.0           # the RGB cameras' exposure for the smear; 0: the sync plan's
    tunnel_pre_s: float = 1.0          # the tunnel exit: from this long before the camera passes the portal
    tunnel_post_s: float = 1.5         # … to this long after it
    tunnel_hz: float = 250.0           # the event camera's frames through the tunnel exit
    tunnel_scale: float = 0.2          # their resolution: this fraction of the sensor's
    supersample: int = 2               # sub-pixels per side averaged into each rendered pixel (smooth edges)
    ae_tau_s: float = 0.25             # the RGB auto exposure's time constant (assumed)
    target_m: float = 25.0             # a stopped car this far past the portal, in the ego's lane
    target_albedo: float = 0.05        # a dark car
    min_contrast: float = 0.1          # Weber contrast the RGB camera must keep to see the car
    min_target_events: int = 20        # events on the car within one window for the event camera to see it
    window_ms: float = 10.0
    lamp_cd: float = 1500.0            # the tunnel's ceiling luminaires (assumed)
    activity_hz: float = 0.0           # events per pixel per second for the recording budget; 0: TYPICAL_ACTIVITY_HZ
    evt3_bits: float = 0.0             # EVT 3.0 bits per event for the budget; 0: EVT3_TYPICAL_BITS
    link_warn: float = 0.8
    rgb_tag: str = ''                  # the RGB camera compared; '' = the nearest visible RGB camera
    virtual_module: str = 'prophesee_evk4_hd'   # a variant without event cameras: this one, beside the RGB camera
    virtual_lens: str = 'prophesee_evk4_lens'
    virtual_pitch_deg: float = 5.0
    virtual_offset_m: float = 0.10     # beside it: this far to its left
    seed: int = 0

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d) -> 'EventSettings':
        """Tolerant: unknown keys dropped, numbers clamped to LIMITS, bad values left at the defaults."""
        s = EventSettings()
        d = d if isinstance(d, dict) else {}
        for k, (lo, hi) in LIMITS.items():
            if k not in d or isinstance(d[k], bool):
                continue
            v = _num(d[k])
            if v is None:
                continue
            v = min(max(v, lo), hi)
            setattr(s, k, int(round(v)) if k in INTS else float(v))
        for k in BOOLS:
            v = d.get(k)
            if isinstance(v, bool):
                setattr(s, k, v)
            elif isinstance(v, str) and v.strip().lower() in ('0', 'false', 'no', 'off', '1', 'true', 'yes', 'on'):
                setattr(s, k, v.strip().lower() in ('1', 'true', 'yes', 'on'))
        for k, allowed in STRS.items():
            if d.get(k) in allowed:
                setattr(s, k, d[k])
        for k in ('rgb_tag', 'virtual_module', 'virtual_lens'):
            if isinstance(d.get(k), str) and len(d[k]) <= 80:
                setattr(s, k, d[k].strip())
        return s


def settings_of(project) -> EventSettings:
    return EventSettings.from_dict(getattr(getattr(project, 'analysis', None), 'event_camera', None))


# the recording budget's default activity (events per pixel per second): an EVK4 HD beside the sample's F at 50 km/h
# through Vantage's rendered scene (the event study at default settings, round 17: 31.5). CARLA's DVS camera measured
# 91 (inside Town03's tunnel) to 169 (outside) at 50 km/h, inflated by its rendering's frame-to-frame flicker, and 0.39
# on a drive standing most of the time at lights (see the round-17 report): replace it with your own drives' figure
TYPICAL_ACTIVITY_HZ = 31.5
TYPICAL_ACTIVITY_NOTE = ('typical: an EVK4 HD at 50 km/h in Vantage\'s rendered scene (31.5 events per pixel per second); '
                         'set your own in the event study\'s settings')
# EVT 3.0 bits per event measured on CARLA's DVS stream at 50 km/h (19.6 Mev/s on 320 × 180 pixels: 30.6; 23.7–35.0 on
# Vantage's rendered drive): the budget's default
EVT3_TYPICAL_BITS = 30.6


# =========================================================================== resolved parameters
@dataclass
class EventParams:
    """A module's event figures resolved for the model: stated where the maker states them, else the settings'
    (the *_source fields say which)."""
    module_id: str
    name: str
    width: int
    height: int
    pixel_um: float
    threshold: float                   # ln units, both polarities
    threshold_pct: float
    threshold_source: str
    sigma: float
    refractory_s: float
    refractory_source: str
    latency_pts: list                  # [(lux, s)], ascending lux
    latency_source: str
    dr_db: float
    dr_lux: tuple                      # (low, high): the range the model keeps a signal in
    cutoff_lux: float
    max_rate_evs: float | None         # the sensor's readout cap
    link_bps: float | None             # the camera's output bandwidth
    formats: list
    timestamp_s: float = 1e-6

    def latency_at(self, lux):
        """The latency (s) at sensor-plane illuminance lux: log-log interpolation between the stated points (one
        point: inversely proportional to illuminance), beyond them extrapolated on the same slope."""
        E = np.maximum(np.asarray(lux, float), 1e-9)
        pts = sorted(self.latency_pts)
        if len(pts) == 1:
            e0, l0 = pts[0]
            return l0 * (e0 / E) if self.latency_source != 'assumed' else np.full(E.shape, l0)
        (e0, l0), (e1, l1) = pts[0], pts[-1]
        k = math.log(l1 / l0) / math.log(e1 / e0)
        return l0 * (E / e0) ** k

    @property
    def max_output_evs(self) -> float | None:
        return self.max_rate_evs

    def to_dict(self):
        d = asdict(self)
        d['latency_pts'] = [[float(a), float(b)] for a, b in self.latency_pts]
        d['dr_lux'] = list(self.dr_lux)
        return d


def params_of(module, settings: EventSettings | None = None) -> EventParams:
    st = settings or EventSettings()
    mp = dict(getattr(module, 'modality_params', None) or {})
    pct = _num(mp.get('contrast_threshold_pct'))
    th_src = 'stated' if pct is not None and pct > 0 else 'assumed'
    pct = pct if th_src == 'stated' else st.threshold_pct
    lat = []
    for pt in mp.get('latency_us') or []:
        if isinstance(pt, (list, tuple)) and len(pt) == 2:
            e, us = _num(pt[0]), _num(pt[1])
            if e and us and e > 0 and us > 0:
                lat.append((e, us * 1e-6))
    lat_src = 'stated' if lat else 'assumed'
    if not lat:
        lat = [(1000.0, st.latency_us * 1e-6)]
    ref = _num(mp.get('refractory_us'))
    ref_src = 'stated' if ref is not None else 'settings'
    ref = ref if ref is not None else st.refractory_us
    lo_hi = mp.get('dynamic_range_max_lux') or mp.get('dynamic_range_lux')
    cutoff = _num(mp.get('low_light_cutoff_lux'))
    dr = _num(mp.get('dynamic_range_max_db')) or _num(mp.get('dynamic_range_db')) or \
        _num(getattr(module, 'dynamic_range_db', 0.0)) or 120.0
    if isinstance(lo_hi, (list, tuple)) and len(lo_hi) == 2 and _num(lo_hi[0]) and _num(lo_hi[1]):
        rng = (float(lo_hi[0]), float(lo_hi[1]))
    else:
        top = 100000.0
        low = cutoff if cutoff else top / 10 ** (dr / 20.0)
        rng = (low, top)
    cutoff = cutoff if cutoff else rng[0]
    mev = _num(mp.get('max_event_rate_mevs'))
    gbps = _num(mp.get('link_gbps'))
    res = mp.get('resolution') if isinstance(mp.get('resolution'), (list, tuple)) else None
    w, h = (int(res[0]), int(res[1])) if res and len(res) == 2 else (int(module.width_px), int(module.height_px))
    fm = [f for f in (mp.get('event_formats') or []) if isinstance(f, str)]
    ts = _num(mp.get('timestamp_us'))
    return EventParams(module.id, f'{module.vendor} {module.name}'.strip(), w, h,
                       float(_num(mp.get('pixel_pitch_um')) or module.pixel_um), math.log1p(pct / 100.0), pct,
                       th_src, st.sigma_ln, ref * 1e-6, ref_src, sorted(lat), lat_src, float(dr), rng, float(cutoff),
                       mev * 1e6 if mev else None, gbps * 1e9 if gbps else None, fm,
                       (ts or 1.0) * 1e-6)


def sensor_lux(lum_cd, f_number=2.0, transmission=0.9):
    """Sensor-plane illuminance (lux) of scene luminance (cd/m²) through a lens: E = π L T / (4 N²)."""
    N = f_number if f_number and f_number > 0 else 2.0
    return math.pi * np.asarray(lum_cd, float) * transmission / (4.0 * N * N)


def simulate(frames, times, cfg: SimConfig, lux=None, seed=0) -> tuple[Events, dict]:
    """Events from a sequence of log-intensity frames (n, h, w) at times (s); lux: per-frame sensor illuminance
    (n, h, w) or None. Returns the time-sorted events and the simulator's counts."""
    frames = np.asarray(frames)
    sim = EventSim(frames.shape[1:], cfg, seed)
    out = []
    for i, t in enumerate(times):
        out.append(sim.step(frames[i], t, None if lux is None else lux[i]))
    return Events.concat(out, sim.w, sim.h).sorted(), dict(sim.counts)


# =========================================================================== the readout cap
def apply_bandwidth(ev: Events, max_rate_evs: float | None, mode='erc', window_us=200.0, fifo_events=65536) -> tuple:
    """(kept events, stats). erc: within each window, events beyond max_rate × window are dropped evenly; fifo: the
    readout serves max_rate events per second in time order from a buffer of fifo_events, delaying them (the delay
    is reported, the timestamps are the pixels'), and drops what arrives at a full buffer."""
    n = len(ev)
    stats = {'mode': mode, 'max_rate_evs': max_rate_evs, 'in': n, 'kept': n, 'dropped': 0, 'dropped_share': 0.0,
             'max_delay_us': 0.0, 'mean_delay_us': 0.0}
    if not n or not max_rate_evs or max_rate_evs <= 0:
        return ev, stats
    ev = ev.sorted()
    if mode == 'erc':
        cap = max_rate_evs * window_us * 1e-6
        win = (ev.t - ev.t[0]) // max(int(round(window_us)), 1)
        starts = np.r_[0, np.flatnonzero(np.diff(win)) + 1]
        counts = np.diff(np.r_[starts, n])
        keep = np.ones(n, bool)
        for s0, c in zip(starts[counts > cap], counts[counts > cap]):
            m = int(math.floor(cap))
            k = np.zeros(c, bool)
            if m > 0:
                k[np.round(np.linspace(0, c - 1, m)).astype(int)] = True
            keep[s0:s0 + c] = k
        out = ev.take(keep)
    else:
        a = ev.t.astype(np.float64) * 1e-6
        svc = 1.0 / max_rate_evs
        i = np.arange(n)
        dep = (i + 1) * svc + np.maximum.accumulate(a - i * svc)         # Lindley, no drops
        backlog = (dep - a) / svc
        keep = np.ones(n, bool)
        if backlog.max() > fifo_events:                                   # exact, sequentially, when it overflows
            from collections import deque
            last = -1e30
            q = deque()                                                   # departure times of the queued events
            dep = np.zeros(n)
            for j in range(n):
                while q and q[0] <= a[j]:
                    q.popleft()
                if len(q) >= fifo_events:
                    keep[j] = False
                    continue
                last = max(last, a[j]) + svc
                dep[j] = last
                q.append(last)
        delay = (dep - a)[keep] * 1e6
        out = ev.take(keep)
        if len(delay):
            stats['max_delay_us'] = float(delay.max())
            stats['mean_delay_us'] = float(delay.mean())
    stats['kept'] = len(out)
    stats['dropped'] = n - len(out)
    stats['dropped_share'] = (n - len(out)) / n
    return out, stats


# =========================================================================== event formats
EVT3_ADDR_Y, EVT3_ADDR_X, EVT3_VECT_BASE_X, EVT3_VECT_12, EVT3_VECT_8, EVT3_TIME_LOW, EVT3_TIME_HIGH = \
    0x0, 0x2, 0x3, 0x4, 0x5, 0x6, 0x8


def split_header(data: bytes) -> tuple[dict, bytes]:
    """(header {key: value}, payload) of a RAW file."""
    head = {}
    pos = 0
    while data.startswith(b'%', pos):
        end = data.find(b'\n', pos)
        if end < 0:
            break
        line = data[pos + 1:end].decode('ascii', 'replace').strip()
        pos = end + 1
        if line == 'end':
            break
        k, _, v = line.partition(' ')
        head[k] = v
    return head, data[pos:]


def decode_evt2(data: bytes, width=0, height=0) -> Events:
    w = np.frombuffer(data, '<u4').astype(np.uint64)
    if not len(w):
        return Events.empty(width, height)
    typ = (w >> np.uint64(28)).astype(np.int64)
    is_th = typ == EVT2_TIME_HIGH
    hi = np.maximum.accumulate(np.where(is_th, np.arange(len(w)), -1))
    base = np.where(hi >= 0, (w[np.maximum(hi, 0)] & np.uint64(0x0FFFFFFF)).astype(np.int64), 0)
    cd = (typ == EVT2_CD_ON) | (typ == EVT2_CD_OFF)
    t = (base[cd] << 6) | ((w[cd] >> np.uint64(22)) & np.uint64(63)).astype(np.int64)
    return Events(((w[cd] >> np.uint64(11)) & np.uint64(0x7FF)).astype(np.uint16),
                  (w[cd] & np.uint64(0x7FF)).astype(np.uint16), t.astype(np.int64),
                  np.where(typ[cd] == EVT2_CD_ON, 1, -1).astype(np.int8), width, height)


def evt21_words(ev: Events) -> int:
    """64-bit words of EVT 2.1: one per (µs, row, polarity, 32-column group) plus a TIME_HIGH per change of t >> 6."""
    if not len(ev):
        return 0
    key = np.unique(np.stack([ev.t, ev.y.astype(np.int64), ev.p.astype(np.int64), ev.x.astype(np.int64) // 32], 1),
                    axis=0)
    hi = np.unique(ev.t >> 6)
    return int(len(key) + len(hi))


def encode_evt3(ev: Events) -> bytes:
    """EVT 3.0 (16-bit words): TIME_HIGH (t >> 12 & 0xFFF) and TIME_LOW (t & 0xFFF) when they change, ADDR_Y when
    the row changes, then per run of neighbouring same-polarity events in a row at one µs: a single event as ADDR_X
    (pol << 11 | x), two or more as VECT_BASE_X (pol << 11 | x0) followed by VECT_12 / VECT_8 validity words."""
    ev = ev.sorted()
    n = len(ev)
    if not n:
        return b''
    o = np.lexsort((ev.x, ev.p, ev.y, ev.t))
    t, y, x, p = ev.t[o].astype(np.int64), ev.y[o].astype(np.int64), ev.x[o].astype(np.int64), ev.p[o]
    pol = (p > 0).astype(np.int64)
    th, tl = (t >> 12) & 0xFFF, t & 0xFFF
    first = np.r_[True, np.zeros(n - 1, bool)]
    new_th = first | np.r_[False, th[1:] != th[:-1]] | np.r_[False, (t[1:] >> 24) != (t[:-1] >> 24)]
    new_t = first | np.r_[False, t[1:] != t[:-1]]
    new_y = first | np.r_[False, y[1:] != y[:-1]]
    run_start = new_t | new_y | np.r_[True, pol[1:] != pol[:-1]] | np.r_[True, x[1:] != x[:-1] + 1]
    run_id = np.cumsum(run_start) - 1
    starts = np.flatnonzero(run_start)
    run_len = np.diff(np.r_[starts, n])
    pos_in = np.arange(n) - starts[run_id]
    single = run_len[run_id] == 1
    # words per event: [TIME_HIGH][TIME_LOW][ADDR_Y][ADDR_X | VECT_BASE_X + VECT | VECT]
    slots = np.zeros((n, 5), np.int64)
    use = np.zeros((n, 5), bool)
    slots[:, 0], use[:, 0] = (EVT3_TIME_HIGH << 12) | th, new_th
    slots[:, 1], use[:, 1] = (EVT3_TIME_LOW << 12) | tl, new_t
    slots[:, 2], use[:, 2] = (EVT3_ADDR_Y << 12) | y, new_y
    slots[:, 3], use[:, 3] = np.where(single, (EVT3_ADDR_X << 12) | (pol << 11) | x,
                                      (EVT3_VECT_BASE_X << 12) | (pol << 11) | x), single | (pos_in == 0)
    chunk = (pos_in % 12 == 0) & ~single
    left = run_len[run_id] - pos_in                   # events of the run from this one on
    m = np.minimum(left, 12)
    mask = (np.int64(1) << m) - 1
    vect8 = m <= 8
    slots[:, 4] = np.where(vect8, (EVT3_VECT_8 << 12) | (mask & 0xFF), (EVT3_VECT_12 << 12) | (mask & 0xFFF))
    use[:, 4] = chunk
    words = slots[use]                                 # row-major: each event's words in order
    return words.astype('<u2').tobytes()


def decode_evt3(data: bytes, width=0, height=0) -> Events:
    """EVT 3.0 back to events (a plain loop: for checks and small files)."""
    w = np.frombuffer(data, '<u2')
    xs, ys, ts, ps = [], [], [], []
    th = tl = y = 0
    hi_wrap = 0
    base_x = base_pol = 0
    prev_th = None
    for word in w.tolist():
        typ = word >> 12
        val = word & 0xFFF
        if typ == EVT3_TIME_HIGH:
            if prev_th is not None and val < prev_th:
                hi_wrap += 1
            prev_th = th = val
        elif typ == EVT3_TIME_LOW:
            tl = val
        elif typ == EVT3_ADDR_Y:
            y = val & 0x7FF
        elif typ in (EVT3_ADDR_X, EVT3_VECT_BASE_X):
            pol = (val >> 11) & 1
            xx = val & 0x7FF
            if typ == EVT3_ADDR_X:
                xs.append(xx), ys.append(y), ts.append((hi_wrap << 24) | (th << 12) | tl), ps.append(1 if pol else -1)
            else:
                base_x, base_pol = xx, pol
        elif typ in (EVT3_VECT_12, EVT3_VECT_8):
            nb = 12 if typ == EVT3_VECT_12 else 8
            for b in range(nb):
                if (val >> b) & 1:
                    xs.append(base_x + b), ys.append(y), ts.append((hi_wrap << 24) | (th << 12) | tl)
                    ps.append(1 if base_pol else -1)
            base_x += nb
    return Events(np.array(xs, np.uint16), np.array(ys, np.uint16), np.array(ts, np.int64), np.array(ps, np.int8),
                  width, height)


def format_bits(ev: Events) -> dict:
    """Bits each format needs for these events: {'EVT2.0', 'EVT2.1', 'EVT3.0'} -> bits (measured by encoding)."""
    if not len(ev):
        return {f: 0 for f in FORMATS}
    return {'EVT2.0': len(encode_evt2(ev)) * 8, 'EVT2.1': evt21_words(ev) * 64, 'EVT3.0': len(encode_evt3(ev)) * 8}


# =========================================================================== event images and statistics
def event_frame(ev: Events, width, height, t0_us=None, t1_us=None, base=None) -> np.ndarray:
    """RGB uint8 image of the events in [t0, t1): ON blue, OFF red, both white, on a grey (or the given) background."""
    sel = ev if t0_us is None else ev.between(t0_us, t1_us)
    on = np.zeros((height, width), np.int32)
    off = np.zeros((height, width), np.int32)
    if len(sel):
        np.add.at(on, (sel.y[sel.p > 0].astype(np.intp), sel.x[sel.p > 0].astype(np.intp)), 1)
        np.add.at(off, (sel.y[sel.p < 0].astype(np.intp), sel.x[sel.p < 0].astype(np.intp)), 1)
    if base is None:
        img = np.full((height, width, 3), 32, np.uint8)
    else:
        g = np.asarray(base, float)
        g = g if g.ndim == 3 else np.repeat(g[..., None], 3, 2)
        img = (g * 0.45).astype(np.uint8)
    img[(on > 0) & (off == 0)] = (70, 150, 255)
    img[(off > 0) & (on == 0)] = (255, 80, 60)
    img[(on > 0) & (off > 0)] = (240, 240, 240)
    return img


def time_surface(ev: Events, width, height, t_us, tau_us=20000.0) -> np.ndarray:
    """exp(−(t − t_last) / τ) per pixel (0 where no event came before t)."""
    sel = ev.take(ev.t <= t_us)
    last = np.full((height, width), -np.inf)
    if len(sel):
        np.maximum.at(last, (sel.y.astype(np.intp), sel.x.astype(np.intp)), sel.t.astype(float))
    return np.where(np.isfinite(last), np.exp(-(t_us - last) / tau_us), 0.0)


def _png(rgb) -> bytes:
    import cv2
    ok, buf = cv2.imencode('.png', np.ascontiguousarray(np.asarray(rgb, np.uint8)[..., ::-1]))
    return buf.tobytes() if ok else b''


# =========================================================================== the rig's event cameras
VIRTUAL_UID = 'event-virtual'


def _project_with_library(project):
    """A shallow copy of the project whose library holds the event cameras (a project saved before them lacks them)."""
    import copy
    q = copy.copy(project)
    q.modules = dict(project.modules)
    q.lenses = dict(project.lenses)
    for m in library('modules'):
        q.modules.setdefault(m.id, m)
    for l in library('lenses'):
        q.lenses.setdefault(l.id, l)
    return q


def _variant(project, variant):
    if isinstance(variant, str):
        variant = project.variant_by_uid(variant)
    return variant or project.variant


def event_cameras(project, variant=None) -> list:
    """The visible cameras of the variant whose module is an event sensor."""
    v = _variant(project, variant)
    return [c for c in v.cameras if c.visible and is_event(project.module_for(c))]


def rgb_cameras(project, variant=None) -> list:
    v = _variant(project, variant)
    return [c for c in v.cameras if c.visible and project.module_for(c).modality in ('rgb', 'nir')]


def comparator(project, cam, st: EventSettings, variant=None):
    """The RGB camera an event camera is compared with: settings.rgb_tag, else the nearest visible RGB camera."""
    rgbs = rgb_cameras(project, variant)
    if not rgbs:
        return None
    named = next((c for c in rgbs if c.tag == st.rgb_tag), None)
    if named is not None:
        return named
    return min(rgbs, key=lambda c: float(np.linalg.norm(c.pos() - cam.pos())))


def virtual_camera(project, st: EventSettings, variant=None):
    """A variant without event cameras: settings.virtual_module beside the RGB camera named by settings.rgb_tag (else F,
    else the first visible camera), virtual_offset_m to its left, at its yaw and virtual_pitch_deg."""
    from .project import Camera
    rgbs = rgb_cameras(project, variant)
    base = next((c for c in rgbs if c.tag == st.rgb_tag), None) or next((c for c in rgbs if c.tag == 'F'), None) \
        or (rgbs[0] if rgbs else None)
    mod = st.virtual_module if st.virtual_module in project.modules else 'prophesee_evk4_hd'
    ln = st.virtual_lens if st.virtual_lens in project.lenses else 'prophesee_evk4_lens'
    if base is None:
        return Camera(uid=VIRTUAL_UID, tag='EV', name='Event camera (virtual)', x=3.5, y=0.0, z=1.2, yaw=0.0,
                      pitch=st.virtual_pitch_deg, module_id=mod, lens_id=ln, slot=7)
    a = math.radians(base.yaw)
    off = st.virtual_offset_m
    return Camera(uid=VIRTUAL_UID, tag=f'{base.tag}-EV', name=f'Event camera beside {base.tag} (virtual)',
                  x=base.x - math.sin(a) * off, y=base.y + math.cos(a) * off, z=base.z, yaw=base.yaw,
                  pitch=st.virtual_pitch_deg, roll=0.0, module_id=mod, lens_id=ln, slot=(base.slot + 4) % 8)


def _intr(project, cam):
    return L.resolve(project.module_for(cam), project.lens_for(cam))


def _f_number(project, cam, st: EventSettings) -> float:
    if st.f_number > 0:
        return st.f_number
    n = _num(project.lens_for(cam).f_number)
    return n if n and n > 0 else 2.0


# =========================================================================== study 1: the event rate over a drive
def _rays(intr, w, h, R):
    from .synth import pixel_rays
    from .mathutil import BODY_TO_OPTICAL
    d_opt, th = pixel_rays(intr, w, h)
    return d_opt @ (BODY_TO_OPTICAL @ R.T), th <= intr.max_theta


def render_sequence(project, engine, cam, w, h, times, speed_mps, progress=None, cancel=None):
    """Luminance-proportional frames (n, h, w) of Vantage's synthetic parking scene through the camera while the
    vehicle drives forward at speed_mps: the camera moves through the scene (its body, rig and housings, rendered once,
    move with it). Values are the renderer's reflectance × shading, 0…1."""
    import copy
    from . import synth
    from .mathutil import rot_zyx
    world = synth.parking_world(project)
    q = copy.copy(project)
    q.vehicle = copy.copy(project.vehicle)
    q.vehicle.mesh = None                              # no body: the scene alone, from the moving camera
    R = rot_zyx(cam.yaw, cam.pitch, cam.roll)
    pos = cam.pos()
    def render(pj, pose=None):                         # (the renderer's wall test multiplies inf by 0 for some rays)
        with np.errstate(invalid='ignore', divide='ignore'):
            return synth.render_world(pj, engine, cam, world, w, h, pose=pose, noise=0.0).astype(np.float32)
    with_body = render(project)
    bare = render(q)
    body = np.abs(with_body - bare).sum(2) > 0.5      # the body, rig and housings in this image
    out = np.zeros((len(times), h, w), np.float32)
    for i, t in enumerate(times):
        if cancel is not None and cancel():
            return None
        img = render(q, (R, pos + np.array([speed_mps * t, 0.0, 0.0])))
        img[body] = with_body[body]
        out[i] = (img @ np.array(REC709, np.float32)) / 255.0
        if progress is not None and i % 8 == 0:
            progress(i / max(len(times), 1))
    return out


def _motion(project, cam, st: EventSettings, exposure_ms=None):
    """Image speed of a static object st.object_m away (core.timing's model) at st.speed_kmh, full resolution."""
    from . import timing as T
    tst = T.TimingSettings.from_dict(dict(getattr(project.analysis, 'timing', None) or {},
                                          speed_kmh=st.speed_kmh, object_m=st.object_m))
    if exposure_ms is not None:
        tst.exposure_ms = exposure_ms
    return T._motion(cam, _intr(project, cam), tst), tst


def rate_study(project, engine, cam, ep: EventParams, st: EventSettings, progress=None, cancel=None) -> dict | None:
    """The event rate and bandwidth of an event camera over a short rendered drive (see the module doc)."""
    intr = _intr(project, cam)
    w = max(16, int(round(intr.width * st.render_scale)))
    h = max(16, int(round(intr.height * st.render_scale)))
    mot, _ = _motion(project, cam, st)
    rate_px = mot['rate_worst_px_s'] * w / intr.width
    n = int(min(max(math.ceil(st.duration_s * rate_px / st.max_step_px), 8), st.max_frames))
    times = np.linspace(0.0, st.duration_s, n + 1)
    fr = render_sequence(project, engine, cam, w, h, times, st.speed_kmh / 3.6,
                         None if progress is None else (lambda f: progress(0.8 * f)), cancel)
    if fr is None:
        return None
    lum = np.maximum(fr * st.scene_white_cd, 1e-4)
    N = _f_number(project, cam, st)
    lux = sensor_lux(lum, N, st.lens_transmission)
    cfg = SimConfig.for_params(ep, st)
    ev, counts = simulate(np.log(lum), times, cfg, lux, seed=st.seed)
    scale = intr.width * intr.height / float(w * h)
    dur_us = int(round(st.duration_s * 1e6))
    # the first frame starts the pixels; the rate counts the events after it
    rate_evs = len(ev) / st.duration_s * scale
    t_b, r_b = rate_series(ev, 1000, 0, dur_us)
    r_b = r_b * scale
    bits = format_bits(ev)
    bpe = {f: (bits[f] / len(ev) if len(ev) else {'EVT2.0': 32.0, 'EVT2.1': 64.0}.get(f, EVT3_TYPICAL_BITS))
           for f in FORMATS}
    # the formats share their time and row words among the events of one µs and row: at the rendered resolution
    # there are fewer of them per µs than on the sensor (bpe above: an upper bound); the same events with their
    # times compressed to the sensor's density give a lower bound
    dense = Events(ev.x, ev.y, np.floor(ev.t / scale).astype(np.int64), ev.p, w, h)
    bd = format_bits(dense)
    bpe_dense = {f: (bd[f] / len(ev) if len(ev) else bpe[f]) for f in FORMATS}
    mbps = {f: rate_evs * bpe[f] / 1e6 for f in FORMATS}
    fmt = st.event_format
    caps = [c for c in (ep.max_rate_evs, (ep.link_bps / bpe[fmt]) if ep.link_bps and bpe[fmt] > 0 else None) if c]
    cap = min(caps) if caps else None
    kept, bw = apply_bandwidth(ev, cap / scale if cap else None, st.bandwidth_mode, st.erc_window_us,
                               max(int(st.fifo_events / scale), 16))
    mid = len(times) // 2
    t_mid = int(round(times[mid] * 1e6))
    grey = np.clip(fr[mid] / max(float(fr[mid].max()), 1e-6), 0, 1) ** (1 / 2.2) * 255
    prev = event_frame(ev, w, h, t_mid - 5000, t_mid + 5000, grey)
    signal = counts['signal']
    return {'width': w, 'height': h, 'frames': len(times), 'duration_s': st.duration_s, 'scale': scale,
            'frame_step_ms': 1000.0 * st.duration_s / n, 'image_speed_px_s': mot['rate_worst_px_s'],
            'events_render': len(ev), 'signal_events_render': signal, 'shot_events_render': counts['shot'],
            'refractory_dropped_render': counts['refractory_dropped'],
            'rate_evs': rate_evs, 'peak_1ms_evs': float(r_b.max()) if len(r_b) else 0.0,
            'p99_1ms_evs': float(np.percentile(r_b, 99)) if len(r_b) else 0.0,
            'activity_hz': rate_evs / (intr.width * intr.height),
            'on_share': float((ev.p > 0).mean()) if len(ev) else 0.0,
            'bits_per_event': bpe, 'bits_per_event_dense': bpe_dense, 'mbps': mbps,
            'mbps_dense': {f: rate_evs * bpe_dense[f] / 1e6 for f in FORMATS}, 'format': fmt, 'cap_evs': cap,
            'cap_source': ('the sensor\'s readout' if cap and ep.max_rate_evs and cap == ep.max_rate_evs else
                           'the camera\'s link' if cap else 'not stated'),
            'link_mbps': ep.link_bps / 1e6 if ep.link_bps else None,
            'link_load': (mbps[fmt] / (ep.link_bps / 1e6)) if ep.link_bps else None,
            'bandwidth': bw, 'series_t_ms': (t_b / 1000.0).tolist(), 'series_evs': r_b.tolist(),
            'preview': prev, 'lux_median': float(np.median(lux[mid])), 'f_number': N}


# =========================================================================== study 2: motion blur
def edge_events(ep: EventParams, st: EventSettings, rate_px_s, lum_bright, f_number, cols=64, rows=24, travel=40.0,
                seed=0) -> dict:
    """A straight edge (dark side 1 − edge_contrast of the bright one, one pixel wide in the image) moving across the
    sensor at rate_px_s: per pixel the first event's time against the edge centre's crossing time. Mismatch, latency,
    refractory and the timestamp grid set the spread; noise is left out."""
    r = max(float(rate_px_s), 1.0)
    x0 = (cols - travel) / 2.0 + 0.0317             # (off the frames' 0.1 px grid: no pixel sits exactly on a tie)
    T = travel / r
    n = int(max(min(math.ceil(travel / 0.1), 4000), 20))
    times = np.linspace(0.0, T, n + 1)
    xs = np.arange(cols) + 0.5
    Lb, Ld = lum_bright, lum_bright * (1.0 - st.edge_contrast)
    cfg = SimConfig.for_params(ep, st, noise=False)
    sim = EventSim((rows, cols), cfg, seed)
    out = []
    for t in times:
        e = x0 + r * t                                   # the edge: bright left of it ... dark right? (moves right)
        cover = np.clip(e - (xs - 0.5), 0.0, 1.0)        # the share of the pixel on the edge's left (bright) side
        Lrow = Ld + (Lb - Ld) * cover                     # pixels turn bright as the edge passes: ON events
        Lim = np.repeat(Lrow[None, :], rows, 0)
        out.append(sim.step(np.log(Lim), t, sensor_lux(Lim, f_number, st.lens_transmission)))
    ev = Events.concat(out, cols, rows).sorted()
    on = ev.take(ev.p > 0)
    tc = (xs - x0) / r                                    # when the edge's centre crosses each pixel's centre, s
    errs = []
    seen = np.zeros((rows, cols), bool)
    order = np.argsort(on.t, kind='stable')
    for i in order:
        yy, xx = int(on.y[i]), int(on.x[i])
        if seen[yy, xx] or not (2 <= xx < cols - 2) or not (0.0 < tc[xx] < T):
            continue
        seen[yy, xx] = True
        errs.append(on.t[i] * 1e-6 - tc[xx])
    errs = np.asarray(errs)
    if not len(errs):
        return {'events': len(ev), 'pixels': 0, 'spread_px': None, 'lag_px': None, 'spread_us': None,
                'lag_us': None, 'rate_px_s': r}
    p10, p50, p90 = np.percentile(errs, [10, 50, 90])
    ts = time_surface(ev, cols, rows, int(T * 0.6e6), tau_us=max(2e6 / r, 50.0))
    return {'events': len(ev), 'pixels': int(len(errs)), 'rate_px_s': r, 'spread_us': float((p90 - p10) * 1e6),
            'lag_us': float(p50 * 1e6), 'spread_px': float((p90 - p10) * r), 'lag_px': float(p50 * r),
            'lux': float(sensor_lux(Lb, f_number, st.lens_transmission)), 'time_surface': ts}


def blur_study(project, cam, rgb, ep: EventParams, st: EventSettings) -> dict:
    """The event camera's time-surface edge spread at its worst image speed against the RGB camera's smear."""
    exp_ms = st.exposure_ms if st.exposure_ms > 0 else None
    mot, tst = _motion(project, cam, st, exp_ms)
    N = _f_number(project, cam, st)
    edge = edge_events(ep, st, mot['rate_worst_px_s'], st.scene_white_cd * 0.4, N, seed=st.seed)
    out = {'speed_kmh': st.speed_kmh, 'object_m': st.object_m, 'exposure_ms': tst.exposure_ms,
           'blur_limit_px': tst.blur_limit_px, 'event_rate_px_s': mot['rate_worst_px_s'],
           'event_rate_centre_px_s': mot['rate_centre_px_s'],
           'same_sensor_smear_px': mot['rate_worst_px_s'] * tst.exposure_ms / 1000.0, 'edge': edge}
    if rgb is not None:
        m2, _ = _motion(project, rgb, st, exp_ms)
        out.update({'rgb_tag': rgb.tag, 'rgb_rate_px_s': m2['rate_worst_px_s'], 'rgb_smear_px': m2['blur_worst_px'],
                    'rgb_smear_centre_px': m2['blur_centre_px']})
    return out


# =========================================================================== study 3: the tunnel exit
def _box_hit(P, D, lo, hi):
    """Ray distances to an axis-aligned box (inf for a miss)."""
    with np.errstate(divide='ignore', invalid='ignore'):
        t1 = (lo[None] - P[None]) / D if P.ndim == 1 else (lo - P) / D
        t2 = (hi[None] - P[None]) / D if P.ndim == 1 else (hi - P) / D
    tmin = np.nanmax(np.minimum(t1, t2), axis=1)
    tmax = np.nanmin(np.maximum(t1, t2), axis=1)
    return np.where((tmax >= tmin) & (tmin > 1e-6), tmin, np.inf)


class TunnelScene:
    """The exposure metering study's tunnel (core.isp_photometrics.TunnelEnv: its portal, size, interior and wall
    luminances, the clear sky with the sun ahead) with lane markings (albedo 0.5 against the road's), wall panels, a
    row of ceiling luminaires (lamp_cd, every 10 m) and a stopped car (4.5 × 1.8 × 1.5 m, target_albedo) target_m past
    the portal in the ego's lane, lit by the global horizontal illuminance (a simplification)."""

    def __init__(self, project, st: EventSettings, isp_st=None):
        from . import isp_photometrics as I
        self.I = I
        self.st = st
        self.ist = isp_st or I.settings_of(project)
        meas = project.vehicle.measured() if project.vehicle.mesh is not None else {}
        front_x = float(meas.get('front_x', 3.6))
        self.env = I.TunnelEnv(self.ist, front_x)
        self.exit_x = self.env.exit_x
        el = self.ist.tunnel_sun_el_deg
        self.el = el
        self.sun_v = np.array([math.cos(math.radians(el)), 0.0, math.sin(math.radians(el))])
        dl = I.daylight(np.array([el]), self.ist.turbidity)
        self.E_h = float(dl['E_h'][0])
        self.out_ground = self.ist.ground_albedo * self.E_h / math.pi
        self.target_lum = st.target_albedo * self.E_h / math.pi
        x0 = self.exit_x + st.target_m
        self.box = (np.array([x0, -0.9, 0.0]), np.array([x0 + 4.5, 0.9, 1.5]))

    def luminance(self, pos, D):
        """(luminance cd/m², class 0 road in, 1 wall/ceiling, 2 road out, 3 sky, 4 the car) of rays D from pos."""
        st, ist, env = self.st, self.ist, self.env
        k = env.classify(pos, D)
        n = len(D)
        Lum = np.zeros(n)
        with np.errstate(divide='ignore', invalid='ignore'):
            tf = np.where(D[:, 2] < -1e-9, -pos[2] / D[:, 2], np.inf)
        P = pos + D * np.where(np.isfinite(tf), tf, 0.0)[:, None]
        hw = ist.tunnel_width_m / 2
        ay = np.abs(P[:, 1])
        lane = (np.abs(ay - 1.75) < 0.075) & ((P[:, 0] % 9.0) < 3.0)
        edge = np.abs(ay - (hw - 0.3)) < 0.075
        alb = np.where(lane | edge, 0.5, ist.ground_albedo) / ist.ground_albedo
        Lum = np.where(k == 0, ist.interior_cd * alb, Lum)
        Lum = np.where(k == 2, self.out_ground * alb, Lum)
        w = k == 1
        if w.any():
            Dw = D[w]
            with np.errstate(divide='ignore', invalid='ignore'):
                tc = np.where(Dw[:, 2] > 1e-9, (ist.tunnel_height_m - pos[2]) / Dw[:, 2], np.inf)
                tw = np.where(np.abs(Dw[:, 1]) > 1e-9, (np.sign(Dw[:, 1]) * hw - pos[1]) / Dw[:, 1], np.inf)
            ceil = tc < tw
            Q = pos + Dw * np.where(ceil, tc, tw)[:, None]
            lamp = ceil & ((Q[:, 0] % 10.0) < 1.2) & (np.abs(Q[:, 1]) < 0.3)
            panel = np.where(np.floor(Q[:, 0] / 2.5) % 2 == 0, 1.2, 0.8)
            low = (~ceil) & (Q[:, 2] < 1.0)
            Lw = ist.wall_cd * np.where(ceil, 0.7, panel) * np.where(low, 0.6, 1.0)
            Lum[w] = np.where(lamp, st.lamp_cd, Lw)
        s = k == 3
        if s.any():
            Lum[s] = self.I.sky_luminance(D[s][None], self.sun_v, np.array([self.el]), ist.turbidity)[0]
        out = (k == 2) | (k == 3)
        tb = np.full(n, np.inf)
        if out.any():
            tb[out] = _box_hit(pos, D[out], *self.box)
        car = out & (tb < np.where(k == 2, tf, np.inf))
        Lum = np.where(car, self.target_lum, Lum)
        k = np.where(car, 4, k)
        return np.maximum(Lum, 1e-3), k


def tunnel_study(project, cam, rgb, ep: EventParams, st: EventSettings, progress=None, cancel=None) -> dict | None:
    """The tunnel exit at st.speed_kmh: the event camera's frames at tunnel_hz through the event model, the RGB
    camera's frames at its frame rate through its auto exposure and HDR range; when each sees the stopped car."""
    from .mathutil import rot_zyx
    sc = TunnelScene(project, st)
    I = sc.I
    v = st.speed_kmh / 3.6
    t0, t1 = -st.tunnel_pre_s, st.tunnel_post_s
    dx0 = sc.exit_x - cam.x                         # the event camera passes the portal at t = 0

    ss = int(st.supersample)

    def setup(c, scale):
        intr = _intr(project, c)
        w = max(16, int(round(intr.width * scale)))
        h = max(16, int(round(intr.height * scale)))
        D, ok = _rays(intr, w * ss, h * ss, rot_zyx(c.yaw, c.pitch, c.roll))
        return intr, w, h, D, ok

    def look(pos, D, ok, w, h):
        """Luminance and class per output pixel: the mean of its ss × ss sub-pixels' luminance (edges move smoothly),
        the class of its centre-most sub-pixel; outside the image circle 1e-3 cd/m², class −1."""
        Lum, k = sc.luminance(pos, D)
        Lum = np.where(ok, Lum, 1e-3).reshape(h, ss, w, ss).mean((1, 3)).reshape(-1)
        kk = np.where(ok, k, -1).reshape(h, ss, w, ss)[:, ss // 2, :, ss // 2].reshape(-1)
        return Lum, kk

    # ---- the event camera
    intr, w, h, D, ok = setup(cam, st.tunnel_scale)
    times = np.arange(t0, t1 + 1e-9, 1.0 / st.tunnel_hz)
    N = _f_number(project, cam, st)
    cfg = SimConfig.for_params(ep, st)
    sim = EventSim((h, w), cfg, st.seed)
    evs, shots, tmask, lux_in, inrange = [], [], np.zeros((len(times), h * w), bool), [], []
    portal_share = []
    mid_img = None
    for i, t in enumerate(times):
        if cancel is not None and cancel():
            return None
        pos = cam.pos() + np.array([dx0 + v * t, 0.0, 0.0])
        Lum, k = look(pos, D, ok, w, h)
        E = sensor_lux(Lum, N, st.lens_transmission)
        sg, sh = sim.step_split(np.log(Lum).reshape(h, w), t, E.reshape(h, w))
        evs.append(sg)
        shots.append(sh)
        tmask[i] = k == 4
        po = (k == 2) | (k == 3) | (k == 4)
        portal_share.append(float(po.mean()))
        inr = (E >= ep.cutoff_lux) & (E <= ep.dr_lux[1])
        inrange.append(float(inr[po].mean()) if po.any() else None)
        lux_in.append(float(np.median(E[(k == 0) | (k == 1)])) if ((k == 0) | (k == 1)).any() else None)
        if mid_img is None and t >= 0.1:
            mid_img = (i, Lum.reshape(h, w))
        if progress is not None and i % 10 == 0:
            progress(0.6 * i / len(times))
    ev = Events.concat(evs, w, h).sorted()                # the signal (the car is seen by these)
    noise = Events.concat(shots, w, h).sorted()
    lux_ok = [x for x in lux_in if x is not None]
    t_s = ev.t * 1e-6 + t0
    fi = np.clip(np.searchsorted(times, t_s, 'left'), 0, len(times) - 1)
    on_target = tmask[fi, ev.y.astype(np.intp) * w + ev.x.astype(np.intp)] if len(ev) else np.zeros(0, bool)
    # at the rendered resolution an edge crosses a pixel 1/lin times less often than on the sensor: the car is judged
    # over window_ms / lin, its events scaled to the sensor (× area) and to one window_ms (× lin)
    lin = w / float(intr.width)
    win = st.window_ms / 1000.0 / lin
    wedges = np.arange(t0, t1 + win, win)
    area = intr.width * intr.height / float(w * h)        # events scale with the pixel count (see rate_study)
    tgt_counts, _ = np.histogram(t_s[on_target], wedges)
    all_counts, _ = np.histogram(np.r_[t_s, noise.t * 1e-6 + t0], wedges)
    tgt_counts = tgt_counts * area * lin
    all_counts = all_counts * area * lin
    in_view_ev = np.array([tmask[(times >= a) & (times < a + win)].sum(1).max(initial=0) >= 4
                           for a in wedges[:-1]])
    seen_ev = tgt_counts >= st.min_target_events
    tgt_lux = float(sensor_lux(sc.target_lum, N, st.lens_transmission))
    ev_doc = {'tag': cam.tag, 'width': w, 'height': h, 'frames': len(times), 'hz': st.tunnel_hz, 'events': len(ev),
              'noise_events': len(noise), 'target_counts': 'signal events only (shot noise left out)',
              'count_scale': area * lin, 'effective_window_ms': 1000.0 * win,
              'counts_note': 'signal events on the car per window_ms on the sensor: counted over window_ms / s at the '
                             'rendered scale s (an edge crosses a rendered pixel 1/s times less often), scaled by the '
                             'pixel count and s',
              'window_ms': st.window_ms, 'window_t_s': wedges[:-1].tolist(), 'target_events': tgt_counts.tolist(),
              'events_per_window': all_counts.tolist(), 'target_in_view': in_view_ev.tolist(),
              'target_seen': seen_ev.tolist(), 'portal_in_range_share': inrange,
              'interior_lux_median': float(np.median(lux_ok)) if lux_ok else None,
              'target_lux': tgt_lux, 'latency_target_us': float(ep.latency_at(tgt_lux)) * 1e6,
              'latency_interior_us': float(ep.latency_at(max(float(np.median(lux_ok)) if lux_ok else 1.0, 1e-6))) * 1e6,
              'blind_s': float(((in_view_ev) & ~seen_ev).sum() * win),
              'first_seen_s': float(wedges[:-1][seen_ev][0]) if seen_ev.any() else None,
              'first_in_view_s': float(wedges[:-1][in_view_ev][0]) if in_view_ev.any() else None}
    prev_ev = None
    if mid_img is not None:
        i, Lm = mid_img
        g = np.clip(np.log(Lm / Lm.min()) / max(np.log(Lm.max() / Lm.min()), 1e-6), 0, 1) * 255
        a = int(round((times[i] - t0) * 1e6))
        prev_ev = event_frame(Events.concat([ev, noise], w, h), w, h, a - int(win * 5e5), a + int(win * 5e5), g)
        ev_doc['preview_t_s'] = float(times[i])
    out = {'scene': {'exit_x': sc.exit_x, 'interior_cd': sc.ist.interior_cd, 'wall_cd': sc.ist.wall_cd,
                     'outside_road_cd': sc.out_ground, 'target_cd': sc.target_lum, 'E_h_lux': sc.E_h,
                     'sun_el_deg': sc.el, 'target_m': st.target_m, 'lamp_cd': st.lamp_cd,
                     'speed_kmh': st.speed_kmh, 'scene_db': 20 * math.log10(max(sc.out_ground, sc.target_lum) /
                                                                              max(sc.ist.interior_cd, 1e-3))},
           'event': ev_doc, 'preview_event': prev_ev}
    # ---- the RGB camera
    if rgb is not None:
        mod = project.module_for(rgb)
        intr2, w2, h2, D2, ok2 = setup(rgb, min(st.tunnel_scale, 640.0 / max(_intr(project, rgb).width, 1)))
        fps = float(mod.fps) if mod.fps and mod.fps > 0 else 30.0
        times2 = np.arange(t0, t1 + 1e-9, 1.0 / fps)
        lv = I.hdr_levels(sc.ist, float(mod.dynamic_range_db or 0.0))
        target = sc.ist.ae_target
        log_eff = None
        rows = []
        prev_rgb = None
        for i, t in enumerate(times2):
            if cancel is not None and cancel():
                return None
            pos = rgb.pos() + np.array([dx0 + v * t, 0.0, 0.0])
            Lum, k = look(pos, D2, ok2, w2, h2)
            sel = k >= 0
            m = float(np.mean(np.log(Lum[sel])))
            if log_eff is None:
                log_eff = m                                       # converged in the tunnel
            else:
                a = 1.0 - math.exp(-(1.0 / fps) / st.ae_tau_s) if st.ae_tau_s > 0 else 1.0
                log_eff += a * (m - log_eff)
            x = target * Lum / math.exp(log_eff)
            xc = np.clip(x, lv['floor'], lv['top'])
            po = sel & ((k == 2) | (k == 3) | (k == 4))
            tg = sel & (k == 4)
            g8 = I.grey_level(x, sc.ist.output_gamma)          # the ISP's 8-bit output (clipped at x = 1)
            row = {'t_s': float(t), 'clipped_portal_share': float((x[po] > lv['top']).mean()) if po.any() else 0.0,
                   'output_clipped_portal_share': float((x[po] >= 1.0).mean()) if po.any() else 0.0,
                   'target_px': int(tg.sum()), 'contrast': None, 'output_contrast': None, 'physical_contrast': None,
                   'seen': False, 'seen_output': False, 'log2_l_eff': float(log_eff / math.log(2))}
            if tg.sum() >= 4:
                # the car against the outside road around it: road pixels inside its box widened by half each side
                vv, uu = np.divmod(np.flatnonzero(tg), w2)
                du, dv = max((uu.max() - uu.min()) // 2, 2), max((vv.max() - vv.min()) // 2, 2)
                V, U = np.divmod(np.arange(len(k)), w2)
                near = sel & (k == 2) & (U >= uu.min() - du) & (U <= uu.max() + du) & (V >= vv.min() - dv) & \
                    (V <= vv.max() + dv)
                if near.sum() >= 4:
                    xb, xt = float(np.median(xc[near])), float(np.median(xc[tg]))
                    gb, gt = float(np.median(g8[near])), float(np.median(g8[tg]))
                    row['contrast'] = abs(xb - xt) / xb
                    row['output_contrast'] = abs(gb - gt) / max(gb, 1.0)
                    row['physical_contrast'] = abs(float(np.median(Lum[near])) - float(np.median(Lum[tg]))) / \
                        float(np.median(Lum[near]))
                    row['seen'] = row['contrast'] >= st.min_contrast
                    row['seen_output'] = row['output_contrast'] >= st.min_contrast
            rows.append(row)
            if prev_rgb is None and t >= 0.1:
                prev_rgb = np.where(sel, g8, 0.0).reshape(h2, w2)        # what the ISP outputs
            if progress is not None and i % 5 == 0:
                progress(0.6 + 0.4 * i / len(times2))
        in_view = np.array([r['target_px'] >= 4 for r in rows])
        seen = np.array([r['seen'] for r in rows])
        seen_o = np.array([r['seen_output'] for r in rows])
        out['rgb'] = {'tag': rgb.tag, 'width': w2, 'height': h2, 'fps': fps, 'dr_db': lv['dr_db'],
                      'dr_given': lv['dr_given'], 'hdr_top': lv['top'], 'hdr_floor': lv['floor'],
                      'ae_tau_s': st.ae_tau_s, 'rows': rows,
                      'blind_s': float((in_view & ~seen).sum() / fps),
                      'first_seen_s': float(times2[seen][0]) if seen.any() else None,
                      'output_blind_s': float((in_view & ~seen_o).sum() / fps),
                      'output_first_seen_s': float(times2[seen_o][0]) if seen_o.any() else None,
                      'output_gamma': sc.ist.output_gamma,
                      'first_in_view_s': float(times2[in_view][0]) if in_view.any() else None,
                      'max_clipped_portal_share': max((r['clipped_portal_share'] for r in rows), default=0.0),
                      'max_output_clipped_portal_share': max((r['output_clipped_portal_share'] for r in rows),
                                                             default=0.0)}
        out['preview_rgb'] = None if prev_rgb is None else np.repeat(prev_rgb[..., None], 3, 2).astype(np.uint8)
    if progress is not None:
        progress(1.0)
    return out


# =========================================================================== the study
@dataclass
class EventResult:
    settings: EventSettings
    variant: str
    variant_uid: str
    cams: list                          # per event camera: dict (see run)
    findings: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    def cam(self, uid):
        return next((c for c in self.cams if c['uid'] == uid), None)


def run(project, engine=None, settings: EventSettings | None = None, variant=None, progress=None, cancel=None) \
        -> EventResult | None:
    """The event cameras of a variant (the active one by default; a virtual one beside the RGB camera when it has
    none): event rate and bandwidth, motion blur, the tunnel exit. Returns None when cancelled."""
    from .analysis import Engine
    t0 = time.perf_counter()
    v = _variant(project, variant)
    p = _project_with_library(project)
    if v is not project.variant:
        p.variants = list(project.variants) + ([] if any(x is v for x in project.variants) else [v])
        p.active_uid = v.uid
    st = settings or settings_of(project)
    eng = engine if engine is not None else Engine()

    def prog(f, text=''):
        if progress is not None:
            progress(min(max(float(f), 0.0), 1.0), text)

    def stop():
        return cancel is not None and cancel()
    cams = [(c, False) for c in event_cameras(p, v)]
    if not cams:
        cams = [(virtual_camera(p, st, v), True)]
    out = []
    n = len(cams)
    for i, (c, virtual) in enumerate(cams):
        base = i / n
        mod = p.module_for(c)
        ep = params_of(mod, st)
        rgb = comparator(p, c, st, v)
        prog(base, f'{c.tag}: the event rate over a drive')
        rs = rate_study(p, eng, c, ep, st, (lambda f, b=base: prog(b + 0.45 * f / n, f'{c.tag}: rendering')), stop)
        if rs is None:
            return None
        prog(base + 0.45 / n, f'{c.tag}: motion blur')
        bl = blur_study(p, c, rgb, ep, st)
        if stop():
            return None
        prog(base + 0.5 / n, f'{c.tag}: tunnel exit')
        tn = tunnel_study(p, c, rgb, ep, st, (lambda f, b=base: prog(b + (0.5 + 0.5 * f) / n, f'{c.tag}: tunnel')),
                          stop)
        if tn is None:
            return None
        out.append({'uid': c.uid, 'tag': c.tag, 'name': c.name, 'virtual': virtual, 'slot': int(c.slot),
                    'module': f'{mod.vendor} {mod.name}'.strip(), 'module_id': mod.id, 'lens_id': c.lens_id,
                    'pose': {k: float(getattr(c, k)) for k in ('x', 'y', 'z', 'yaw', 'pitch', 'roll')},
                    'params': ep.to_dict(), 'rgb_tag': rgb.tag if rgb is not None else None,
                    'rate': rs, 'blur': bl, 'tunnel': tn})
    res = EventResult(st, v.name, v.uid, out)
    res.metrics = {'seconds': time.perf_counter() - t0, 'cameras': len(out),
                   'virtual': any(c['virtual'] for c in out)}
    res.findings = _findings(res)
    prog(1.0, 'Done')
    return res


def _f(out, sev, kind, msg, uids=()):
    out.append({'severity': sev, 'kind': kind, 'message': msg, 'uids': list(uids)})


def _findings(res: EventResult) -> list:
    out = []
    st = res.settings
    for c in res.cams:
        tag, r, b, tn, pr = c['tag'], c['rate'], c['blur'], c['tunnel'], c['params']
        u = [] if c['virtual'] else [c['uid']]
        if c['virtual']:
            _f(out, 'note', 'virtual', f'The variant has no event camera: the study shows a virtual {c["module"]} '
                                       f'beside {c["rgb_tag"] or "the rig"} ({tag}). Add one from the library to plan it.')
        fmt = r['format']
        if r['link_load'] is not None and r['link_load'] > 1.0:
            _f(out, 'serious', 'link', f'{tag}: {r["mbps"][fmt]:.0f} Mbit/s of {FORMAT_LABEL[fmt]} at '
                                       f'{st.speed_kmh:g} km/h is more than its {r["link_mbps"]:.0f} Mbit/s link.', u)
        elif r['link_load'] is not None and r['link_load'] > st.link_warn:
            _f(out, 'warning', 'link', f'{tag}: {100 * r["link_load"]:.0f} % of its link at {st.speed_kmh:g} km/h.', u)
        if r['bandwidth']['dropped_share'] > 0.001:
            _f(out, 'warning', 'rate', f'{tag}: the readout ({r["cap_source"]}, {r["cap_evs"] / 1e6:.0f} Mev/s) drops '
                                       f'{100 * r["bandwidth"]["dropped_share"]:.1f} % of the events in the busiest '
                                       f'{st.erc_window_us:g} µs windows.', u)
        e = b['edge']
        if e.get('spread_px') is not None and b.get('rgb_smear_px') is not None:
            sev = 'good' if e['spread_px'] < b['rgb_smear_px'] else 'warning'
            _f(out, sev, 'blur', f'{tag}: a moving edge is placed within {e["spread_px"]:.2f} px (10–90 % of its '
                                 f'events; {e["spread_us"]:.0f} µs) where {b["rgb_tag"]}\'s {b["exposure_ms"]:g} ms '
                                 f'exposure smears {b["rgb_smear_px"]:.1f} px at {st.speed_kmh:g} km/h; the events '
                                 f'lag the edge by {e["lag_px"]:.2f} px ({e["lag_us"]:.0f} µs).', u)
        ev, rg = tn['event'], tn.get('rgb')
        ev_text = (f'{tag} sees it in every {st.window_ms:g} ms window while it is in view' if ev['blind_s'] == 0 else
                   f'{tag} misses it for {1000 * ev["blind_s"]:.0f} ms (fewer than {st.min_target_events} events '
                   f'in a {st.window_ms:g} ms window)')
        if rg is not None and rg['first_in_view_s'] is not None:
            lost = []
            if rg['output_blind_s'] > 0:
                lost.append(f'its ISP output (8-bit, gamma {rg["output_gamma"]:g}, clipped at the long exposure) '
                            f'for {1000 * rg["output_blind_s"]:.0f} ms')
            if rg['blind_s'] > 0:
                lost.append(f'its merged {rg["dr_db"]:.0f} dB HDR signal for {1000 * rg["blind_s"]:.0f} ms')
            if lost:
                _f(out, 'warning', 'tunnel', f'Tunnel exit at {st.speed_kmh:g} km/h: {rg["tag"]} cannot tell the '
                                             f'stopped car from the road in ' + ' and in '.join(lost)
                                             + f' while it is in view (contrast under {100 * st.min_contrast:g} %); '
                                             + ev_text + '.', u)
            else:
                _f(out, 'good', 'tunnel', f'Tunnel exit at {st.speed_kmh:g} km/h: {rg["tag"]} keeps the stopped '
                                          f'car\'s contrast throughout; ' + ev_text + '.', u)
        for k, src in (('threshold', pr['threshold_source']), ('latency', pr['latency_source'])):
            if src == 'assumed':
                _f(out, 'note', 'assumed', f'{tag}: {c["module"]} states no {k}: the settings\' figure is used.', u)
    order = {'critical': 0, 'serious': 1, 'warning': 2, 'note': 3, 'good': 4}
    out.sort(key=lambda f: order.get(f['severity'], 5))
    return out


# =========================================================================== the recording budget and issues
def typical_activity(st: EventSettings | None = None) -> tuple[float, str]:
    st = st or EventSettings()
    if st.activity_hz > 0:
        return st.activity_hz, 'the event study\'s setting'
    return TYPICAL_ACTIVITY_HZ, TYPICAL_ACTIVITY_NOTE


def recording_row(project, cam, settings: EventSettings | None = None, rate_evs=None) -> dict:
    """An event camera's data rate for the recording budget (cheap): its event rate (given, else the typical activity ×
    its pixels) × the bits per event of the chosen format, and its link."""
    st = settings or settings_of(project)
    m = project.module_for(cam)
    ep = params_of(m, st)
    if rate_evs is None:
        act, src = typical_activity(st)
        rate_evs = act * ep.width * ep.height
    else:
        src = 'given'
    fmt = st.event_format
    bpe = 32.0 if fmt == 'EVT2.0' else (st.evt3_bits or EVT3_TYPICAL_BITS) if fmt == 'EVT3.0' else 64.0
    mbps = rate_evs * bpe / 1e6
    if fmt == 'EVT2.0' and rate_evs > 0:
        mbps += 32.0 / 64e-6 / 1e6                   # a TIME_HIGH word every 64 µs
    cap = ep.link_bps / 1e6 if ep.link_bps else None
    return {'tag': cam.tag, 'uid': cam.uid, 'events_per_s': float(rate_evs), 'format': fmt,
            'bits_per_event': bpe, 'mbps': mbps, 'source': src, 'link_mbps': cap,
            'load': mbps / cap if cap else None, 'max_rate_evs': ep.max_rate_evs,
            'text': f'{FORMAT_LABEL[fmt]} · {rate_evs / 1e6:.2f} Mev/s · {bpe:g} bit/event'}


def issues(project, vg=None) -> list:
    """Cheap: an event camera whose typical data rate fills its link, or whose typical event rate is above its sensor's
    readout."""
    from . import rules as R
    st = settings_of(project)
    act, _ = typical_activity(st)
    if act <= 0:
        return []
    out = []
    for c in event_cameras(project):
        r = recording_row(project, c, st)
        if r['load'] is not None and r['load'] > st.link_warn:
            out.append(R.Issue('serious' if r['load'] > 1 else 'warning',
                               f'{c.tag}: a typical drive\'s events ({r["mbps"]:.0f} Mbit/s of {r["format"]}) fill '
                               f'{100 * r["load"]:.0f} % of its {r["link_mbps"]:.0f} Mbit/s link.', c.tag, [c.uid]))
        if r['max_rate_evs'] and r['events_per_s'] > r['max_rate_evs']:
            out.append(R.Issue('warning', f'{c.tag}: a typical drive\'s {r["events_per_s"] / 1e6:.0f} Mev/s is more '
                                          f'than its sensor reads out ({r["max_rate_evs"] / 1e6:.0f} Mev/s).',
                               c.tag, [c.uid]))
    return out


# =========================================================================== CARLA: the rig file and the comparison
def sim_event_cameras(project, variant=None, image_scale=1.0, reference_rgb=True, reference_frames=40,
                      settings: EventSettings | None = None, include_virtual=False) -> list:
    """The variant's event cameras as the CARLA bridge records them (sim_rig.json event_cameras): one pinhole view
    at the lens model's horizontal field (CARLA's DVS camera is a pinhole: the lens's distortion is not applied to the
    events), the DVS's thresholds from the module (θ = ln(1 + contrast threshold)), and, with reference_rgb, an RGB
    view at the same pose, size and step whose frames feed this module's model in the bridge."""
    from . import exporters as EX
    from ..sim import frames as FR
    from .mathutil import rot_zyx
    st = settings or settings_of(project)
    p = _project_with_library(project)
    v = _variant(p, variant)
    cams = event_cameras(p, v)
    if not cams and include_virtual:
        cams = [virtual_camera(p, st, v)]
    tags = EX.safe_tags(v) if any(c in v.cameras for c in cams) else {}
    out = []
    for c in cams:
        m = p.module_for(c)
        ep = params_of(m, st)
        fov = min(L.model_fov(m, p.lens_for(c), circle=True)[0], 170.0)
        w = max(16, int(round(ep.width * image_scale)))
        h = max(16, int(round(ep.height * image_scale)))
        f_px = w / (2.0 * math.tan(math.radians(fov) / 2.0))
        key = tags.get(c.uid) or ''.join(ch if ch.isalnum() else '_' for ch in c.tag)
        R = rot_zyx(c.yaw, c.pitch, c.roll)
        pp, yy, rr = FR.carla_rotation(R)
        loc = FR.carla_location(c.pos())
        th = ep.threshold
        out.append({
            'uid': c.uid, 'tag': c.tag, 'safe_tag': key, 'name': c.name, 'module': f'{m.vendor} {m.name}'.strip(),
            'module_id': m.id, 'channel': f'EVENTS_{key.upper()}', 'topic': f'/{key.lower()}/events',
            'frame_id': f'{key}_optical_frame',
            'pose_vehicle': {k: float(getattr(c, k)) for k in ('x', 'y', 'z', 'yaw', 'pitch', 'roll')},
            'carla': {'location': [round(float(x), 6) for x in loc],
                      'rotation': {'pitch': round(pp, 6), 'yaw': round(yy, 6), 'roll': round(rr, 6)}},
            'width': w, 'height': h, 'fov_deg': float(fov), 'f_px': float(f_px),
            'K': [[f_px, 0.0, (w - 1) / 2.0], [0.0, f_px, (h - 1) / 2.0], [0.0, 0.0, 1.0]],
            'sensor_size': [ep.width, ep.height], 'image_scale': float(image_scale),
            'dvs': {'positive_threshold': round(th, 6), 'negative_threshold': round(th, 6),
                    'sigma_positive_threshold': 0.0, 'sigma_negative_threshold': 0.0,
                    'refractory_period_ns': int(round(ep.refractory_s * 1e9)), 'use_log': True, 'log_eps': 0.001},
            'reference_rgb': bool(reference_rgb), 'reference_frames': int(reference_frames),
            'model': {'threshold': th, 'threshold_pct': ep.threshold_pct, 'threshold_source': ep.threshold_source,
                      'refractory_s': ep.refractory_s},
            'lens_note': 'a pinhole at the lens model\'s horizontal field: the events are not distorted as the real '
                         'lens would',
            'virtual': c.uid == VIRTUAL_UID})
    return out


def add_to_sim_rig(doc: dict, project, variant=None, settings: EventSettings | None = None, include_virtual=False,
                   **kw) -> dict:
    """Adds the variant's event cameras to a sim_rig.json document (core.simrig.build's): event_cameras, their static
    transforms and a note; their own image scale is the rig's output.image_scale unless given."""
    from .mathutil import BODY_TO_OPTICAL, rot_zyx
    kw.setdefault('image_scale', float(doc.get('output', {}).get('image_scale', 1.0)))
    evc = sim_event_cameras(project, variant, settings=settings, include_virtual=include_virtual, **kw)
    doc['event_cameras'] = evc
    uids = {e['uid'] for e in evc}
    gone = [c for c in doc.get('cameras', []) if c.get('uid') in uids]
    if gone:                                           # a rig built before event cameras were known: they are not RGB
        doc['cameras'] = [c for c in doc['cameras'] if c.get('uid') not in uids]
        tags = {c['safe_tag'] for c in gone}
        chans = {c.get('channel') for c in gone}
        nus = doc.get('nuscenes') or {}
        toks = {r['token'] for r in nus.get('sensor', []) if r.get('channel') in chans}
        if nus:
            nus['sensor'] = [r for r in nus.get('sensor', []) if r.get('channel') not in chans]
            nus['calibrated_sensor'] = [r for r in nus.get('calibrated_sensor', []) if r.get('sensor_token') not in toks]
        doc['tf_static'] = [t for t in doc.get('tf_static', []) if t.get('child') not in
                            {f'{k}_link' for k in tags} | {f'{k}_optical_frame' for k in tags}]
    for e in evc:
        T = np.eye(4)
        pv = e['pose_vehicle']
        T[:3, :3] = rot_zyx(pv['yaw'], pv['pitch'], pv['roll'])
        T[:3, 3] = [pv['x'], pv['y'], pv['z']]
        To = np.eye(4)
        To[:3, :3] = BODY_TO_OPTICAL.T
        doc.setdefault('tf_static', []).append({'parent': 'base_link', 'child': f'{e["safe_tag"]}_link',
                                                'matrix': [[round(float(x), 10) for x in r] for r in T]})
        doc['tf_static'].append({'parent': f'{e["safe_tag"]}_link', 'child': f'{e["safe_tag"]}_optical_frame',
                                 'matrix': [[round(float(x), 10) for x in r] for r in To]})
    if evc:
        doc.setdefault('notes', []).append(
            f'{len(evc)} event camera(s) recorded with CARLA\'s DVS camera (sensor.camera.dvs) into events/'
            + (' with a reference RGB view each' if any(e['reference_rgb'] for e in evc) else '') + '.')
    return doc


def load_recording(out_dir) -> dict | None:
    """A CARLA drive's events/index.json (EVENTS_SCHEMA) with each channel's file paths; None without one."""
    path = os.path.join(out_dir, 'events', 'index.json')
    if not os.path.isfile(path):
        return None
    with open(path, encoding='utf-8') as fh:
        doc = json.load(fh)
    if doc.get('schema') != EVENTS_SCHEMA:
        return None
    doc['root'] = os.path.join(out_dir, 'events')
    return doc


def recording_events(doc: dict, tag: str, which='carla') -> Events | None:
    """A recorded channel's events: which = 'carla' (the DVS camera), 'model' (this model on the reference RGB view,
    CARLA's algorithm), 'model_linear' (on the view in linear light)."""
    ch = next((c for c in doc.get('channels', []) if c['tag'] == tag), None)
    if ch is None or not ch.get('files', {}).get(which):
        return None
    return Events.from_npz(os.path.join(doc['root'], ch['files'][which]))


def recording_summary(doc: dict) -> list:
    """Per channel of a recording: the event rate statistics and the comparison the bridge measured."""
    out = []
    for ch in doc.get('channels', []):
        ev = recording_events(doc, ch['tag'])
        row = {'tag': ch['tag'], 'width': ch['width'], 'height': ch['height'], 'duration_s': ch['duration_s'],
               'events': ch.get('events'), 'compare': ch.get('compare'), 'compare_linear': ch.get('compare_linear')}
        if ev is not None and len(ev):
            _, r = rate_series(ev, 1000, 0, int(ch['duration_s'] * 1e6))
            px = ch['width'] * ch['height']
            row.update({'rate_evs': len(ev) / ch['duration_s'], 'peak_1ms_evs': float(r.max()),
                        'p99_1ms_evs': float(np.percentile(r, 99)), 'activity_hz': len(ev) / ch['duration_s'] / px,
                        'on_share': float((ev.p > 0).mean()), 'bits': format_bits(ev)})
        out.append(row)
    return out


# =========================================================================== exports
def _r(v, d=4):
    if v is None:
        return None
    if isinstance(v, (float, np.floating)):
        v = float(v)
        return round(v, d) if math.isfinite(v) else None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, dict):
        return {k: _r(x, d) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_r(x, d) for x in v]
    if isinstance(v, np.ndarray):
        return None
    return v


DROP = ('preview', 'preview_event', 'preview_rgb', 'time_surface')


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items() if k not in DROP}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    return o


def summary(res: EventResult) -> dict:
    rows = []
    for c in res.cams:
        r, b, tn = c['rate'], c['blur'], c['tunnel']
        rg = tn.get('rgb') or {}
        rows.append({'tag': c['tag'], 'virtual': c['virtual'], 'module': c['module'],
                     'rate_mev_s': _r(r['rate_evs'] / 1e6, 3), 'peak_mev_s': _r(r['peak_1ms_evs'] / 1e6, 3),
                     'mbps': _r(r['mbps'][r['format']], 2), 'format': r['format'], 'link_load': _r(r['link_load'], 3),
                     'dropped_share': _r(r['bandwidth']['dropped_share'], 4),
                     'edge_spread_px': _r(b['edge'].get('spread_px'), 3), 'rgb_smear_px': _r(b.get('rgb_smear_px'), 2),
                     'tunnel_event_blind_ms': _r(1000 * tn['event']['blind_s'], 1),
                     'tunnel_rgb_blind_ms': _r(1000 * rg['blind_s'], 1) if rg else None,
                     'tunnel_rgb_output_blind_ms': _r(1000 * rg['output_blind_s'], 1) if rg else None,
                     'rgb_tag': c['rgb_tag']})
    return {'schema': SCHEMA, 'variant': res.variant, 'cameras': rows, 'worst': res.findings[0]['severity']
            if res.findings else 'good', 'findings': len(res.findings), 'seconds': _r(res.metrics.get('seconds'), 2)}


def to_doc(res: EventResult, project=None) -> dict:
    return {'schema': SCHEMA, 'project': getattr(project, 'name', None), 'variant': res.variant,
            'variant_uid': res.variant_uid, 'frame': FRAME,
            'units': {'rate': 'events per second (ev/s)', 'data': 'Mbit/s (10⁶ bit/s)', 'time': 's unless _ms/_us',
                      'illuminance': 'lux at the sensor plane', 'luminance': 'cd/m²', 'image': 'pixels'},
            'settings': res.settings.to_dict(), 'cameras': _r(_clean(res.cams)), 'findings': res.findings,
            'metrics': _r(res.metrics), 'assumptions': ASSUMPTIONS,
            'references': ['Rebecq, Gehrig, Scaramuzza, ESIM, CoRL 2018', 'Hu, Liu, Delbruck, v2e, CVPRW 2021',
                           'Lichtsteiner, Posch, Delbruck, IEEE JSSC 43(2), 2008',
                           'Sony/Prophesee IMX636-AAMR-C product brief 2024 v3.0', 'Prophesee EVK4 HD brief 2026',
                           'Prophesee/Sony ISSCC 2020 press release (Finateu et al., paper 5.10)',
                           'Prophesee EVK5 HD brief 2023; IMX646 page', 'Prophesee GenX320 briefs 2025/2026',
                           'Prophesee EVT 2.0 / 2.1 / 3.0 format documentation']}


CSV_COLUMNS = ('tag', 'module', 'virtual', 'rate_mev_s', 'peak_1ms_mev_s', 'p99_1ms_mev_s', 'activity_hz', 'format',
               'bits_per_event', 'mbps', 'link_mbps', 'link_load', 'cap_mev_s', 'dropped_share', 'edge_spread_px',
               'edge_lag_px', 'rgb_tag', 'rgb_smear_px', 'tunnel_event_first_seen_s', 'tunnel_event_blind_ms',
               'tunnel_rgb_first_seen_s', 'tunnel_rgb_blind_ms', 'tunnel_rgb_output_blind_ms')


def to_csv(res: EventResult) -> str:
    buf = io.StringIO()
    wr = csv.writer(buf, lineterminator='\n')
    wr.writerow(CSV_COLUMNS)
    for c in res.cams:
        r, b, tn = c['rate'], c['blur'], c['tunnel']
        rg = tn.get('rgb') or {}
        f = r['format']
        vals = (c['tag'], c['module'], c['virtual'], r['rate_evs'] / 1e6, r['peak_1ms_evs'] / 1e6,
                r['p99_1ms_evs'] / 1e6, r['activity_hz'], f, r['bits_per_event'][f], r['mbps'][f], r['link_mbps'],
                r['link_load'], (r['cap_evs'] or 0) / 1e6 or None, r['bandwidth']['dropped_share'],
                b['edge'].get('spread_px'), b['edge'].get('lag_px'), b.get('rgb_tag'), b.get('rgb_smear_px'),
                tn['event']['first_seen_s'], 1000 * tn['event']['blind_s'], rg.get('first_seen_s'),
                1000 * rg['blind_s'] if rg else None, 1000 * rg['output_blind_s'] if rg else None)
        wr.writerow(['' if x is None else (f'{x:.6g}' if isinstance(x, float) else x) for x in vals])
    return buf.getvalue()


def tunnel_csv(cam: dict) -> str:
    tn = cam['tunnel']
    buf = io.StringIO()
    wr = csv.writer(buf, lineterminator='\n')
    ev = tn['event']
    wr.writerow(('camera', 't_s', 'events', 'target_events', 'target_in_view', 'target_seen', 'target_seen_output',
                 'clipped_portal_share', 'output_clipped_portal_share', 'contrast', 'output_contrast',
                 'physical_contrast'))

    def g(v):
        return '' if v is None else f'{v:.4f}'
    for t, n, k, iv, s in zip(ev['window_t_s'], ev['events_per_window'], ev['target_events'], ev['target_in_view'],
                              ev['target_seen']):
        wr.writerow((ev['tag'], f'{t:.4f}', f'{n:.0f}', f'{k:.0f}', int(iv), int(s), '', '', '', '', '', ''))
    for r in (tn.get('rgb') or {}).get('rows', []):
        wr.writerow((tn['rgb']['tag'], f'{r["t_s"]:.4f}', '', '', int(r['target_px'] >= 4), int(r['seen']),
                     int(r['seen_output']), g(r['clipped_portal_share']), g(r['output_clipped_portal_share']),
                     g(r['contrast']), g(r['output_contrast']), g(r['physical_contrast'])))
    return buf.getvalue()


def file_tag(tag) -> str:
    return ''.join(ch if ch.isalnum() or ch in '-_' else '_' for ch in str(tag)) or 'cam'


def files(res: EventResult, project=None) -> list:
    out = [('event_cameras.json', (json.dumps(to_doc(res, project), indent=1) + '\n').encode('utf-8')),
           ('event_cameras.csv', to_csv(res).encode('utf-8'))]
    for c in res.cams:
        t = file_tag(c['tag'])
        r = c['rate']
        rows = '\n'.join(f'{a:.3f},{b:.1f}' for a, b in zip(r['series_t_ms'], r['series_evs']))
        out.append((f'event_rate_{t}.csv', ('t_ms,events_per_s\n' + rows + '\n').encode('utf-8')))
        out.append((f'tunnel_{t}.csv', tunnel_csv(c).encode('utf-8')))
        for name, img in ((f'events_{t}.png', r.get('preview')), (f'tunnel_{t}_events.png',
                                                                    c['tunnel'].get('preview_event')),
                          (f'tunnel_{t}_rgb.png', c['tunnel'].get('preview_rgb'))):
            if img is not None:
                out.append((name, _png(img)))
        ts = c['blur']['edge'].get('time_surface')
        if ts is not None:
            g = (np.clip(ts, 0, 1) * 255).astype(np.uint8)
            g = np.repeat(np.repeat(g, 4, 0), 4, 1)
            out.append((f'time_surface_{t}.png', _png(np.repeat(g[..., None], 3, 2))))
    return out


def export(res: EventResult, project, out_dir) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for name, data in files(res, project):
        path = os.path.join(out_dir, name)
        with open(path, 'wb') as fh:
            fh.write(data)
        paths.append(path)
    return paths


# =========================================================================== validation
def carla_reference_loop(it, itdt, ref, last, C, t_prev_ns, t_now_ns, refractory_ns=0, tol=1e-6):
    """A literal per-pixel transcription of CARLA 0.9.16's DVSCamera.cpp Simulation() loop (float32 values, ns
    times): the independent reference the validation compares EventSim(carla=True) with. Arrays are flat; ref and
    last are updated in place; returns [(pixel, t_ns, pol)]."""
    out = []
    f32 = np.float32
    dtn = t_now_ns - t_prev_ns
    for i in range(len(it)):
        a, b, r = f32(it[i]), f32(itdt[i]), f32(ref[i])
        if abs(float(a) - float(b)) > tol:
            pol = 1.0 if b >= a else -1.0
            c = f32(C)
            cur = r
            while True:
                cur = f32(cur + f32(pol) * c)
                if (pol > 0 and cur > a and cur <= b) or (pol < 0 and cur < a and cur >= b):
                    edt = int(float(f32(cur - a)) * dtn / float(f32(b - a)))
                    t = t_prev_ns + edt
                    if last[i] == 0 or t - last[i] >= refractory_ns:
                        out.append((i, t, pol > 0))
                        last[i] = t
                    ref[i] = cur
                else:
                    break
    return out


def validation_checks() -> list:
    """Checks for vantage/validation (Spec-compatible dicts; fn(ctx) -> suite.Outcome)."""
    from ..validation.suite import Outcome

    def ramp(ctx):
        # ln I rises by exactly 1.0 over 1 s in 100 frames, then falls to −0.05 over the next second: θ gives
        # n = floor(1 / θ) ON events at t = k θ s (the reference ends at n θ), then floor((n θ + 0.05) / θ) OFF events
        errs, detail = [], {}
        for th in (0.15, 0.22, 0.3):
            cfg = SimConfig(th, th)
            up = np.linspace(0.0, 1.0, 101)
            seq = np.r_[up, 1.0 - 1.05 * up[1:]]
            tt = np.linspace(0.0, 2.0, len(seq))
            ev, _ = simulate(seq[:, None, None] * np.ones((1, 2, 3)), tt, cfg)
            on, off = ev.take(ev.p > 0), ev.take(ev.p < 0)
            n_on, n_off = len(on) / 6.0, len(off) / 6.0
            exp_n = int(math.floor(1.0 / th))
            want = np.array([k * th for k in range(1, exp_n + 1)])
            got = np.sort(np.unique(on.t)) * 1e-6
            terr = float(np.abs(got - want).max()) * 1e6 if len(got) == len(want) else 1e9
            exp_off = int(math.floor((exp_n * th + 0.05) / th))
            errs.append(abs(n_on - exp_n) + abs(n_off - exp_off) + max(terr - 1.0, 0.0))
            detail[f'theta_{th:g}'] = {'on': n_on, 'off': n_off, 'expected_on': exp_n, 'expected_off': exp_off,
                                       'max_time_error_us': terr}
        return Outcome(error=float(max(errs)), detail=detail)

    def bandwidth(ctx):
        # 200 000 events uniformly over 0.1 s (2 Mev/s) through a 1 Mev/s cap: erc keeps 1 Mev/s in every window
        # (half); fifo with a 1000-event buffer keeps μT + B = 100 000 + 1000 (deterministic queue)
        n = 200000
        t = np.floor(np.linspace(0, 100000, n, endpoint=False)).astype(np.int64)
        ev = Events(np.zeros(n, np.uint16), np.zeros(n, np.uint16), t, np.ones(n, np.int8), 1, 1)
        k1, s1 = apply_bandwidth(ev, 1e6, 'erc', 200.0)
        _, r1 = rate_series(k1, 200, 0, 100000)
        k2, s2 = apply_bandwidth(ev, 1e6, 'fifo', fifo_events=1000)
        err1 = max(0.0, float(r1.max()) / 1e6 - 1.0)
        err2 = abs(len(k2) - 101000) / 101000
        return Outcome(error=max(err1, err2), detail={'erc_kept': len(k1), 'erc_max_rate_mevs': float(r1.max()) / 1e6,
                                                      'fifo_kept': len(k2), 'fifo_expected': 101000,
                                                      'fifo_max_delay_us': s2['max_delay_us']})

    def carla_dvs(ctx):
        # a scripted scene: a bar of random texture sliding across 48 × 32 pixels over 12 frames at 100 Hz, 8-bit
        # BGR; EventSim(carla=True) against the literal DVSCamera.cpp loop
        rng = np.random.default_rng(17)
        tex = rng.integers(20, 235, (32, 96, 3)).astype(np.uint8)
        frames = [tex[:, k * 3:k * 3 + 48] for k in range(12)]
        logs = [carla_log_grey(f).reshape(-1) for f in frames]
        th = 0.223
        sim = EventSim((32, 48), SimConfig.carla_dvs({'positive_threshold': th, 'negative_threshold': th}))
        mine = [sim.step(l.reshape(32, 48), 0.01 * k) for k, l in enumerate(logs)]
        mine = Events.concat(mine, 48, 32)
        ref = logs[0].copy()
        last = np.zeros(len(ref), np.int64)
        theirs = []
        for k in range(1, len(logs)):
            theirs += carla_reference_loop(logs[k - 1], logs[k], ref, last, th, (k - 1) * 10_000_000, k * 10_000_000)
        ref_ev = Events(np.array([i % 48 for i, _, _ in theirs], np.uint16),
                        np.array([i // 48 for i, _, _ in theirs], np.uint16),
                        np.array([t // 1000 for _, t, _ in theirs], np.int64),
                        np.array([1 if p else -1 for _, _, p in theirs], np.int8), 48, 32)
        cmp = compare_dvs(ref_ev, mine, 48, 32)
        err = (1.0 - (cmp['exact_match_share'] or 0.0)) + (cmp['count_error'] or 0.0)
        return Outcome(error=float(err), detail=cmp)

    return [
        {'id': 'event_ramp_count', 'model': 'event camera', 'title': 'Events from a known log-intensity ramp',
         'quantity': 'event count error plus the event times\' error beyond 1 µs', 'reference': 'the analytic '
         'crossings of a linear ramp: floor(ΔlnI / θ) events at t = kθ / (ΔlnI/s)', 'tolerance': 0.0, 'unit': 'events',
         'refs': ('rebecq2018_esim',), 'fn': ramp, 'cost': 'cheap'},
        {'id': 'event_bandwidth_cap', 'model': 'event camera', 'title': 'The readout cap (ERC and FIFO)',
         'quantity': 'the ERC\'s output rate over the cap (share) and the FIFO\'s kept events against μT + B',
         'reference': 'a deterministic queue: a uniform 2 Mev/s burst into a 1 Mev/s readout keeps μT + B events',
         'tolerance': 0.002, 'unit': 'share', 'refs': ('kleinrock1975',), 'fn': bandwidth, 'cost': 'cheap'},
        {'id': 'event_carla_dvs', 'model': 'event camera', 'title': 'Agreement with CARLA\'s DVS camera algorithm',
         'quantity': 'share of events not matched exactly (pixel, polarity, ±1 µs) plus the count error',
         'reference': 'a literal transcription of CARLA 0.9.16 DVSCamera.cpp\'s ESIM loop on a scripted scene '
                      '(the round-17 CARLA drive measured the server itself: see the module doc)',
         'tolerance': 0.0, 'unit': 'share', 'refs': ('carla_dvs', 'rebecq2018_esim'), 'fn': carla_dvs,
         'cost': 'cheap'},
    ]


VALIDATION_REFERENCES = {
    'rebecq2018_esim': 'H. Rebecq, D. Gehrig, D. Scaramuzza, "ESIM: an Open Event Camera Simulator", Conference on '
                       'Robot Learning (CoRL), 2018.',
    'hu2021_v2e': 'Y. Hu, S.-C. Liu, T. Delbruck, "v2e: From Video Frames to Realistic DVS Events", IEEE/CVF CVPR '
                  'Workshops, 2021 (arXiv:2006.07722).',
    'carla_dvs': 'CARLA 0.9.16, sensor.camera.dvs (DVSCamera.cpp, "based on ESIM"), https://carla.readthedocs.io.',
    'kleinrock1975': 'L. Kleinrock, Queueing Systems, Vol. 1: Theory, Wiley, 1975 (Lindley\'s recursion for a '
                     'single-server queue).',
}
