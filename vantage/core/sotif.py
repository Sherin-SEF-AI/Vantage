"""SOTIF insufficiency matrix and the ODD speed bound (ISO 21448, Safety of the intended functionality): which
triggering conditions make the rig's perception functionally insufficient, measured from Vantage's own studies, and
the highest speed at which the rig still detects the critical road users within the stopping distance under each.

This supports a safety case; it is not a certification. ISO 21448 asks the developer to identify triggering conditions
and functional insufficiencies, to evaluate the residual risk and to argue that it is acceptable: the matrix, the ODD
bound and the scenario areas below are evidence for that argument, computed from documented models with their limits
stated. The standard's own text is not reproduced here; its concepts are described in plain words.

Triggering conditions (rows)
----------------------------
Each row is one condition of the operating environment or of the rig itself, and every figure in it comes from a study
the app already has (cited per cell as 'study', 'module' and 'function'):
* weather and light (core.weather through core.perception and core.sensor_coverage): rain in four intensity bins
  (2.5, 7.6, 25 and 50 mm/h: the upper edges of the American Meteorological Society's light and moderate classes, a
  heavy rate and the Met Office's 'violent' threshold), fog in four visibility bins (1000 m, the WMO's fog threshold,
  200, 100 and 50 m), moderate and heavy snowfall (2 and 5 mm/h water), night on an unlit road and under street
  lights. Cameras: the perception range under the condition (contrast, light and pixels: core.perception with
  core.weather's thresholds; visible and NIR cameras only, the weather model being theirs: LWIR and event cameras
  enter through their own rows below); lidars: each lidar's range per class (core.sensor_coverage: points on the object) capped
  by its datasheet range derated by the condition's extinction (core.weather.lidar_range); radars: each radar's range
  per class derated by the condition's attenuation (core.weather.radar_range). Lens drops (core.weather.lens_drop_share)
  and lidar false near returns (core.weather.false_return_share) are field-of-view losses, not range;
* low sun ahead (core.exposure): a camera with the low sun in its image (any share of driving time above sun_min_share)
  is blinded while it is; each such camera is removed in turn and the worst case kept;
* lens dirt (core.exposure's dirt and spray score): a camera or lidar scoring dirt_min_score or more is obscured; each in
  turn is removed and the worst case kept, unless core.cleaning's layout keeps its window clean (a washer covering at
  least cleaning_min_cover of it): then the row says so and keeps the sensor;
* blind volumes near the car (core.volume): where a child 0.9 m tall can stand unseen by every camera; the row
  reports the hidden area in the ego lane ahead of the bumper (start-off) and all round;
* sensor housing occlusion (core.exposure's body mask over each image, which includes the rig parts and the housings
  composited into the occlusion cube; core.sensor_coverage's share of a lidar's points that land on the vehicle);
* tunnel exit (core.isp_photometrics' tunnel case: the step from the tunnel's interior road luminance to the daylit
  road; the camera's HDR headroom above the AE target from its stated dynamic range): a camera is blind while its AE
  closes the part of the step its headroom does not cover, at ae_ev_per_s (assumed); that blind time adds to the
  reaction time;
* thermal (LWIR) and event cameras, when core.thermal_ir and core.event_camera are part of the build and the rig has
  such cameras (Module.modality): night on an unlit road with the LWIR cameras (core.thermal_ir.run's clear-night
  detection range per camera, NETD and DRI pixels, capping that camera's geometric far range from core.perception;
  or thermal_ir.sotif_ranges when a build offers it), and the tunnel exit with the event cameras
  (core.event_camera.tunnel_study's rendered exit: the seconds the event camera cannot see the target; or
  event_camera.sotif_tunnel). Without those modules the row says what it needs and stays not evaluated.

Functional insufficiencies (columns)
------------------------------------
* range below the stopping distance, per critical class (Pedestrian, Cyclist and Car by default): the detection range
  in the ego lane corridor ahead (a lane_half_m wide band each side of the vehicle's centre line, from the front
  bumper; every point of the corridor out to that distance is reached by the sensors of the basis) against the
  stopping distance at the ODD's target speed on the condition's road surface, plus a stop margin;
* the blind volume hides a child (core.volume's 'Child 0.9 m');
* the field of view is lost: a camera blinded, obscured or under drops, a lidar's beams on the vehicle or returning
  from droplets, per sensor with its share.
The basis says which sensors make the detection range: 'two' (default: the second-best sensor type at each azimuth, so
two kinds of sensor agree), 'any' (the best), or one kind alone ('cameras', 'lidars', 'radars').

The ODD bound
-------------
The stopping distance is the distance travelled during the delay (perception plus reaction plus any extra blind
time) at constant speed, then while the deceleration ramps up at a constant jerk, then at full deceleration:
    d(v) = v·t_d + v·t_j − a·t_j²/6 + (v − a·t_j/2)² / (2a)      for v ≥ a·t_j/2, t_j = a / jerk
    d(v) = (2/3)·v·√(2v/jerk) + v·t_d                             below (the vehicle stops during the ramp)
with a = min(the model's deceleration, μ·g) on the condition's road (μ: peak tyre–road adhesion). Stop models:
* 'r157' (default): UN Regulation No. 157 (ALKS), 01 series, Amendment 4 (E/ECE/TRANS/505/Rev.3/Add.156/Amend.4,
  3 March 2023), Annex 3 §3.3, Table 1 (printed p. 41), the competent and careful human driver performance model:
  risk evaluation time 0.4 s, 0.75 s from finished perception to the start of deceleration, deceleration rising to
  0.774 g in 0.6 s (road friction 1.0);
* 'r157_fsm': the same Annex's §3.4 Table 3 (printed p. 60), the fuzzy safety model of the ALKS vehicle: reaction time
  0.75 s, jerk 12.65 m/s³, maximum deceleration 6 m/s²;
* 'aashto': design stopping sight distance of AASHTO's Green Book as WSDOT Design Manual M 22-01.23 (September 2024)
  §1260.03(1) states it: 2.5 s perception-reaction time and a constant 11.2 ft/s² (3.41 m/s²); its Exhibit 1260-1
  table is the validation reference;
* 'custom': the settings' figures.
Road adhesion (peak, the lower end of each range): dry asphalt 0.8, wet asphalt 0.5, hard-packed snow 0.2, ice 0.1:
J. Y. Wong, Theory of Ground Vehicles, 4th ed., Wiley 2008, Table 1.3 (average coefficients of road adhesion).
Ranges are measured from the front bumper; a stop margin (default 2 m: the safety distance d1 when two vehicles reach
a complete stop in R157 Annex 3 Table 3) is subtracted. The maximum safe speed of a condition is the largest v with
d(v) ≤ range − margin (closed form on the main branch, bisection on the ramp branch); the ODD speed limit is the
minimum over the conditions the ODD includes.

Known and unknown scenarios (ISO 21448's four areas, in plain words)
------------------------------------------------------------------
Area 1, known and not hazardous: conditions in the ODD whose every cell is sufficient at the target speed. Area 2,
known and hazardous: conditions with an insufficient cell (the ODD must restrict them: a lower speed, an exclusion or a
fix to the rig). Area 3, unknown and hazardous: what these studies cannot evaluate (their stated 'not modelled' lists,
cells not evaluated): the target of further testing (CARLA campaigns, test drives). Area 4, unknown and not hazardous:
cannot be listed by construction; testing that finds no hazard moves scenarios there. The aim of the SOTIF activities
is to shrink areas 2 and 3.

Limits: every study's own limits apply (each is quoted in the JSON); the range is the far range (objects nearer than
a camera's nearest resolving range are the blind-volume row's business); the detector minimums are the perception
range's typical pixel heights, not a trained detector's measured recall; lens drops and dirt rates are heuristics;
conditions are taken one at a time (no combinations); the sensors of the basis are taken as fused without latency.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import html
import io
import json
import math
import os
import threading
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass, field, replace

import numpy as np

SCHEMA = 'vantage.sotif/1'
FRAME = 'vehicle (ISO 8855, rear axle): x forward, y left, z up'
G = 9.80665
DISCLAIMER = ('This supports a safety case under ISO 21448 (SOTIF); it is not a certification. The figures come from '
              'Vantage\'s documented models with their stated limits; a safety case also needs the function\'s own '
              'validation (test drives, simulation campaigns) and an assessment of the residual risk.')
ASSUMPTIONS = ('Every cell is read off one of Vantage\'s studies (the perception range with the weather model, the '
               'lidar and radar coverage, sun glare and dirt exposure, the blind volume, the ISP tunnel case); their '
               'assumptions and limits apply unchanged (quoted under "studies"). The detection range is the far range '
               'in the ego lane corridor ahead, reached by the sensors of the basis at every point; conditions are one '
               'at a time; lens drops and dirt are heuristics; the AE adaptation rate at a tunnel exit is assumed; '
               'road adhesion is a typical table value.')

# ---- stop models (see the module docstring for the sources)
R157 = ('UN Regulation No. 157, 01 series, Amendment 4 (E/ECE/TRANS/505/Rev.3/Add.156/Amend.4, 3 March 2023), Annex 3')
STOP_MODELS = {
    'r157': {'label': 'R157 careful driver', 'perception_s': 0.4, 'reaction_s': 0.75, 'decel_g': 0.774,
             'jerk_ms3': 0.774 * G / 0.6,
             'source': R157 + ' §3.3 Table 1 (printed p. 41): risk evaluation 0.4 s; 0.75 s from finished perception '
                              'to the start of deceleration; 0.6 s to 0.774 g (road friction 1.0).'},
    'r157_fsm': {'label': 'R157 ALKS (FSM)', 'perception_s': 0.0, 'reaction_s': 0.75, 'decel_g': 6.0 / G,
                 'jerk_ms3': 12.65,
                 'source': R157 + ' §3.4 Table 3 (printed p. 60): reaction time 0.75 s, jerk 12.65 m/s³, maximum '
                                  'deceleration of the ALKS vehicle 6 m/s².'},
    'aashto': {'label': 'AASHTO sight distance', 'perception_s': 0.0, 'reaction_s': 2.5, 'decel_g': 11.2 * 0.3048 / G,
               'jerk_ms3': math.inf,
               'source': 'AASHTO, A Policy on Geometric Design of Highways and Streets, as WSDOT Design Manual '
                         'M 22-01.23 (September 2024) §1260.03(1) states it: 2.5 s perception and reaction, a constant '
                         'deceleration of 11.2 ft/s²; Exhibit 1260-1.'},
    'custom': {'label': 'Custom', 'source': 'The settings\' figures (given by the user).'},
}
STOP_IDS = tuple(STOP_MODELS)
ROAD_MU = {'dry': 0.8, 'wet': 0.5, 'snow': 0.2, 'ice': 0.1}
ROADS = tuple(ROAD_MU)
ROAD_LABEL = {'dry': 'Dry', 'wet': 'Wet', 'snow': 'Snow', 'ice': 'Ice'}
ROAD_SOURCE = ('J. Y. Wong, Theory of Ground Vehicles, 4th ed., Wiley 2008, Table 1.3: peak coefficients of road '
               'adhesion, the lower end of each range (asphalt dry 0.8–0.9, wet 0.5–0.7; snow hard-packed 0.2; ice 0.1).')
MARGIN_SOURCE = R157 + ' §3.4 Table 3: safety distance d1 = 2 m when the two vehicles reach a complete stop.'

BASES = ('two', 'any', 'cameras', 'lidars', 'radars')
BASIS_LABEL = {'two': 'Two sensor types', 'any': 'Any sensor', 'cameras': 'Cameras', 'lidars': 'Lidars',
               'radars': 'Radars'}
DEFAULT_CLASSES = ('Pedestrian', 'Cyclist', 'Car')
INSUFFICIENCIES = ('range', 'child', 'fov')
INSUFFICIENCY_LABEL = {'range': 'Detection range below the stopping distance', 'child': 'Blind volume hides a child',
                       'fov': 'Field of view lost'}
VERDICTS = ('ok', 'insufficient', 'not_evaluated', 'n/a')
CHILD = 'Child 0.9 m'
MAX_D = 400.0                       # the corridor is searched out to this from the bumper, m
CORRIDOR_STEP = 0.25
AREAS = (
    (1, 'Known, not hazardous', 'Conditions in the ODD whose every cell is sufficient at the target speed.'),
    (2, 'Known, hazardous', 'Conditions with an insufficient cell: the ODD restricts them (a lower speed or an '
                            'exclusion) or the rig is changed.'),
    (3, 'Unknown, hazardous', 'What these studies cannot evaluate: their not-modelled effects and cells not '
                              'evaluated. Reduce it by testing: simulation campaigns and drives.'),
    (4, 'Unknown, not hazardous', 'Not listable by construction: testing that finds no hazard moves scenarios here.'),
)
NOT_MODELLED = (
    ('weather', 'veiling light from the own headlamps in fog, oncoming glare, wet-road reflections, spray from other '
                'vehicles, a wet radar fascia, rain clutter'),
    ('perception', 'a trained detector\'s recall (the range is pixels on target against typical minimums)'),
    ('exposure', 'buildings, trees and terrain shading or reflecting the sun; dirt as a measured accumulation'),
    ('combinations', 'two conditions at once (night and rain, fog at a tunnel exit)'),
    ('dynamics', 'sensor latency and fusion delay; the vehicle pitching under braking'),
    ('targets', 'occluded road users (behind parked cars), road users at the corridor\'s edge entering late'),
)

# ---- the triggering conditions: (id, label, kind, params, road, note)
RAIN_BINS = ((2.5, 'light'), (7.6, 'moderate'), (25.0, 'heavy'), (50.0, 'violent'))
FOG_BINS = (1000.0, 200.0, 100.0, 50.0)
SNOW_BINS = ((2.0, 'moderate'), (5.0, 'heavy'))
TRIGGERS = (
    ('clear', 'Clear day (reference)', 'weather', {'preset': 'clear'}, 'dry', 'The reference: no weather model.'),
    ('sun_low', 'Low sun ahead', 'sun', {}, 'dry', 'A camera with the low sun in its image (core.exposure).'),
    *[(f'rain_{r:g}'.replace('.', '_'), f'Rain, {w} ({r:g} mm/h)', 'weather',
       {'preset': 'heavy_rain', 'rain_mm_h': r}, 'wet',
       'AMS Glossary of Meteorology, "Rain": light up to 2.5 mm/h, moderate 2.6–7.6, heavy above 7.6; '
       'the Met Office calls 50 mm/h and more violent.') for r, w in RAIN_BINS],
    *[(f'fog_{int(m)}', f'Fog, {m:g} m visibility', 'weather', {'preset': 'fog', 'visibility_m': m}, 'dry',
       'WMO: fog when the visibility is below 1000 m (meteorological optical range).') for m in FOG_BINS],
    *[(f'snow_{int(s)}', f'Snow, {w} ({s:g} mm/h)', 'weather', {'preset': 'snow', 'snow_mm_h': s}, 'snow',
       'Dry snowfall, water equivalent.') for s, w in SNOW_BINS],
    ('night', 'Night, unlit road', 'weather', {'preset': 'night'}, 'dry', 'Low beam, 0.1 lx of moonlight.'),
    ('night_lit', 'Night, street lights', 'weather', {'preset': 'night_lit'}, 'dry', 'About 10 lx.'),
    ('dirt', 'Lens dirt', 'dirt', {}, 'wet', 'A camera or lidar with a high dirt and spray exposure score.'),
    ('blind', 'Blind volume near the car', 'blind', {}, 'dry', 'A child 0.9 m tall standing close to the car.'),
    ('housing', 'Sensor housing occlusion', 'housing', {}, 'dry', 'The body, rig parts and housings in each view.'),
    ('tunnel', 'Tunnel exit', 'tunnel', {}, 'dry', 'Leaving a lit tunnel into daylight.'),
    ('thermal_night', 'Night, unlit road, with the thermal cameras', 'thermal', {}, 'dry',
     'LWIR cameras\' own range (core.thermal_ir).'),
    ('event_tunnel', 'Tunnel exit, with the event cameras', 'event', {}, 'dry',
     'Event cameras\' dynamic range (core.event_camera).'),
)
TRIGGER = {t[0]: t for t in TRIGGERS}
TRIGGER_IDS = tuple(TRIGGER)
OPTIONAL = ('thermal_night', 'event_tunnel')     # only when the rig has such cameras
SEVERITIES = ('critical', 'serious', 'warning', 'good')

# accepted range of every number (values outside are clamped)
LIMITS = {'target_kmh': (5.0, 250.0), 'margin_m': (0.0, 50.0), 'lane_half_m': (0.2, 6.0),
          'perception_s': (0.0, 5.0), 'reaction_s': (0.0, 5.0), 'decel_g': (0.05, 1.5), 'jerk_ms3': (0.5, 1000.0),
          'sun_min_share': (0.0, 0.5), 'dirt_min_score': (0.0, 1.0), 'fov_max_share': (0.0, 1.0),
          'cleaning_min_cover': (0.0, 1.0), 'ae_ev_per_s': (0.1, 100.0)}


def _num(v):
    if isinstance(v, bool):
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


# =========================================================================== stopping distance
@dataclass(frozen=True)
class Stop:
    """A stop: delay t_d at constant speed, then the deceleration ramps at jerk to decel (m/s²)."""
    delay_s: float
    decel: float
    jerk: float = math.inf

    @property
    def t_j(self) -> float:
        return 0.0 if not math.isfinite(self.jerk) or self.jerk <= 0 else self.decel / self.jerk


def stop_of(st: 'SotifSettings', road: str = 'dry', extra_delay_s: float = 0.0) -> Stop:
    """The stop of the settings' model on a road surface (the deceleration capped at μ·g; the ramp keeps the model's
    jerk, so a lower deceleration is reached sooner) with an extra delay (a tunnel exit's blind time)."""
    m = st.model()
    mu = ROAD_MU.get(road, ROAD_MU['dry'])
    a = min(m['decel_g'], mu) * G
    return Stop(m['perception_s'] + m['reaction_s'] + max(float(extra_delay_s), 0.0), a, m['jerk_ms3'])


def stopping_distance(v, stop: Stop) -> float:
    """Distance (m) to stop from v (m/s) with stop's delay, jerk ramp and deceleration (closed form; module doc)."""
    v = max(float(v), 0.0)
    a, tj, td = stop.decel, stop.t_j, stop.delay_s
    if v <= 0:
        return 0.0
    if tj <= 0:
        return v * td + v * v / (2.0 * a)
    if v >= a * tj / 2.0:
        return v * td + v * tj - a * tj * tj / 6.0 + (v - a * tj / 2.0) ** 2 / (2.0 * a)
    T = math.sqrt(2.0 * v / stop.jerk)
    return v * td + 2.0 / 3.0 * v * T


def max_safe_speed(range_m, stop: Stop) -> float:
    """The largest speed (m/s) whose stopping distance fits in range_m (0 when range ≤ 0): the inverse of
    stopping_distance, closed form on the main branch, bisection where the vehicle stops during the ramp."""
    R = float(range_m)
    if not math.isfinite(R):
        return math.inf
    if R <= 0:
        return 0.0
    a, tj, td = stop.decel, stop.t_j, stop.delay_s
    B = td + tj / 2.0
    v = a * (-B + math.sqrt(B * B + 2.0 * (R + a * tj * tj / 24.0) / a))
    if tj <= 0 or v >= a * tj / 2.0:
        return v
    lo, hi = 0.0, a * tj / 2.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if stopping_distance(mid, stop) <= R:
            lo = mid
        else:
            hi = mid
    return lo


def kmh(v_ms) -> float:
    return float(v_ms) * 3.6


# =========================================================================== settings
@dataclass
class SotifSettings:
    stop_model: str = 'r157'
    target_kmh: float = 50.0           # the ODD's target speed: the cells' verdicts are at this speed
    margin_m: float = 2.0              # subtracted from every range (R157 Annex 3 Table 3's d1)
    classes: list = field(default_factory=lambda: list(DEFAULT_CLASSES))   # the critical classes
    basis: str = 'two'
    lane_half_m: float = 1.75          # the ego lane corridor: this far each side of the centre line
    odd: list = field(default_factory=lambda: [t for t in TRIGGER_IDS])    # the conditions the ODD includes
    roads: dict = field(default_factory=dict)        # {trigger id: road surface} overriding the defaults
    perception_s: float = 0.4          # 'custom' model
    reaction_s: float = 0.75
    decel_g: float = 0.774
    jerk_ms3: float = 12.65
    sun_min_share: float = 0.0         # a camera with the low sun in its image more than this share of the time
    dirt_min_score: float = 0.5        # core.exposure's dirt score from which a lens counts as obscured ('high')
    fov_max_share: float = 0.10        # a sensor losing more of its view than this is a field-of-view insufficiency
    cleaning_min_cover: float = 0.9    # core.cleaning: a washer covering this share of a window keeps it clean
    ae_ev_per_s: float = 4.0           # AE adaptation rate at a tunnel exit (assumed)

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d) -> 'SotifSettings':
        """Tolerant of junk: unknown keys ignored, numbers clamped to LIMITS, ids checked."""
        s = SotifSettings()
        d = d if isinstance(d, dict) else {}
        if d.get('stop_model') in STOP_IDS:
            s.stop_model = d['stop_model']
        if d.get('basis') in BASES:
            s.basis = d['basis']
        cls = d.get('classes')
        if isinstance(cls, (list, tuple)):
            out = []
            for c in cls:
                if isinstance(c, str) and c.strip() and c.strip()[:40] not in out:
                    out.append(c.strip()[:40])
            s.classes = out[:16]
        odd = d.get('odd')
        if isinstance(odd, (list, tuple)):
            keep = {t for t in odd if t in TRIGGER}
            s.odd = [t for t in TRIGGER_IDS if t in keep]
        roads = d.get('roads')
        if isinstance(roads, dict):
            s.roads = {k: v for k, v in roads.items() if k in TRIGGER and v in ROADS and v != TRIGGER[k][4]}
        for k, (lo, hi) in LIMITS.items():
            if k in d:
                v = _num(d[k])
                if v is not None:
                    setattr(s, k, min(max(v, lo), hi))
        return s

    def model(self) -> dict:
        if self.stop_model == 'custom':
            return {'label': STOP_MODELS['custom']['label'], 'perception_s': self.perception_s,
                    'reaction_s': self.reaction_s, 'decel_g': self.decel_g, 'jerk_ms3': self.jerk_ms3,
                    'source': STOP_MODELS['custom']['source']}
        return STOP_MODELS[self.stop_model]

    def road(self, tid) -> str:
        return self.roads.get(tid, TRIGGER[tid][4] if tid in TRIGGER else 'dry')


def settings_of(project) -> SotifSettings:
    return SotifSettings.from_dict(getattr(getattr(project, 'analysis', None), 'sotif', None))


def weather_condition(tid: str, wst=None):
    """The core.weather Condition of a weather trigger (its preset with the bin's figures; the project's edits of the
    preset other than the bin's own figure are kept)."""
    from . import weather as W
    t = TRIGGER[tid]
    prm = dict(t[3])
    base = (wst.condition_for(prm.pop('preset')) if wst is not None else W.PRESET[prm.pop('preset')])
    if base.clear:
        return base
    return replace(base, id=tid, name=t[1].split(' (')[0], **prm)


# =========================================================================== geometry: the corridor ahead
def corridor_range(far_az, az_deg, centre, front_x, half_w, max_d=MAX_D, step=CORRIDOR_STEP, ny=9) -> float:
    """The distance ahead of the front bumper (m) out to which every point of the corridor |y| ≤ half_w is within the
    far range of its azimuth (far_az (A,) about centre, azimuths az_deg (A,) evenly spaced from 0, nearest sample).
    max_d when it never ends."""
    far = np.nan_to_num(np.asarray(far_az, float), nan=0.0, posinf=1e9)
    A = len(far)
    if A == 0:
        return 0.0
    d = np.arange(0.0, max_d + step / 2, step)
    y = np.linspace(-half_w, half_w, ny)
    X = front_x + d[:, None] - centre[0]
    Y = y[None, :] - centre[1]
    r = np.hypot(X, Y)
    a = np.degrees(np.arctan2(Y, X)) % 360.0
    da = 360.0 / A
    a0 = float(az_deg[0]) if len(az_deg) else 0.0
    idx = np.rint((a - a0) / da).astype(int) % A
    ok = (far[idx] >= r).all(axis=1)
    if ok.all():
        return float(max_d)
    first = int(np.argmin(ok))
    return 0.0 if first == 0 else float(d[first - 1])


def combine(cam, lid, rad, basis: str) -> np.ndarray:
    """Per azimuth the range the basis gives from the three sensor types' ranges (A,) each (nan or None: none)."""
    arrs = [np.nan_to_num(np.asarray(x, float), nan=0.0) if x is not None else None for x in (cam, lid, rad)]
    n = next((len(x) for x in arrs if x is not None), 0)
    z = np.zeros(n)
    cam, lid, rad = (x if x is not None else z for x in arrs)
    if basis == 'cameras':
        return cam
    if basis == 'lidars':
        return lid
    if basis == 'radars':
        return rad
    S = np.sort(np.stack([cam, lid, rad]), axis=0)
    return S[-1] if basis == 'any' else S[-2]


# =========================================================================== the study
@dataclass
class SotifResult:
    settings: SotifSettings
    rows: list                  # per trigger: dict (see run)
    odd: dict                   # {'speed_kmh', 'binding', 'target_kmh', 'meets_target', 'included'}
    areas: list                 # [{'area', 'label', 'text', 'items': [...]}]
    trace: list                 # requirements linked to cells
    proposals: list             # requirement proposals from the ODD target
    findings: list
    studies: dict               # study id -> {'module', 'assumptions', 'seconds', 'error'}
    classes: list               # the critical classes used
    geometry: dict              # front_x, centre, lane
    metrics: dict
    variant: str = ''
    variant_uid: str = ''
    key: str = ''

    def row(self, tid) -> dict | None:
        return next((r for r in self.rows if r['id'] == tid), None)

    def worst(self) -> str:
        return self.findings[0]['severity'] if self.findings else 'good'


def _project_for(project, variant):
    if variant is None or variant is project.variant:
        return project
    q = copy.copy(project)
    q.variants = list(project.variants) + ([] if any(v is variant for v in project.variants) else [variant])
    q.active_uid = variant.uid
    return q


def _source(study, module, function, figure) -> dict:
    return {'study': study, 'module': f'vantage.core.{module}', 'function': function, 'figure': figure}


VISIBLE = ('rgb', 'nir')


def _hide(p, cams):
    """A copy of the project with these cameras hidden (the project itself unchanged)."""
    if not cams:
        return p
    q = p.snapshot()
    uids = {c.uid for c in cams}
    for c in q.variant.cameras:
        if c.uid in uids:
            c.visible = False
    return q


def _optional_module(name):
    import importlib
    try:
        return importlib.import_module(f'{__package__}.{name}')
    except ImportError:
        return None


def _modality(module) -> str:
    return getattr(module, 'modality', 'rgb') or 'rgb'


class _Ctx:
    """What every trigger needs, computed once: the perception range with every weather condition, the lidar and
    radar coverage, exposure, the blind volume."""

    def __init__(self, p, engine, st: SotifSettings, prog, cancel):
        self.p, self.engine, self.st, self.prog, self.cancel = p, engine, st, prog, cancel
        self.errors, self.seconds = {}, {}

    def stop(self):
        return self.cancel is not None and self.cancel()


def _timed(ctx, sid, fn):
    t0 = time.perf_counter()
    try:
        return fn()
    except Exception as e:                       # noqa: BLE001 — a study that fails leaves its cells not evaluated
        ctx.errors[sid] = f'{type(e).__name__}: {e}'
        return None
    finally:
        ctx.seconds[sid] = ctx.seconds.get(sid, 0.0) + time.perf_counter() - t0


def run(project, engine=None, settings: SotifSettings | None = None, variant=None, progress=None,
        cancel=None) -> SotifResult | None:
    """The insufficiency matrix and the ODD bound of a variant (the active one by default). Returns None when
    cancelled."""
    from . import perception as PR
    from . import weather as W
    t0 = time.perf_counter()
    p = _project_for(project, variant)
    v = p.variant
    st = settings or settings_of(p)
    if engine is None:
        from .analysis import Engine
        engine = Engine()

    def prog(f, text=''):
        if progress is not None:
            try:
                progress(min(max(float(f), 0.0), 1.0), text)
            except TypeError:
                progress(min(max(float(f), 0.0), 1.0))
    ctx = _Ctx(p, engine, st, prog, cancel)
    wst = W.settings_of(p)
    pst = PR.PerceptionSettings.from_dict(getattr(p.analysis, 'perception', None))
    names = [c['name'] for c in pst.classes]
    classes = [c for c in st.classes if c in names] or [n for n in DEFAULT_CLASSES if n in names] or names[:3]
    ctx.classes = classes
    vg = engine.vehicle_geom(p)
    if vg is not None:
        lo, hi = vg.bbox
        centre = (float(lo[0] + hi[0]) / 2, float(lo[1] + hi[1]) / 2)
        front_x = float(hi[0])
    else:
        centre, front_x = (0.0, 0.0), 0.0
    ctx.centre, ctx.front_x = centre, front_x
    weather_ids = [t[0] for t in TRIGGERS if t[2] == 'weather']
    conds = [weather_condition(tid, wst) for tid in weather_ids]
    prog(0.02, 'perception range')
    # the weather model (contrast, headlamps, the photon budget) is a visible camera's: LWIR and event cameras are
    # left out of the perception range here and enter through their own modules' rows
    other = [c for c in v.cameras if c.visible and _modality(p.module_for(c)) not in VISIBLE]
    p_vis = _hide(p, other)
    cams = [c for c in v.cameras if c.visible and c not in other]
    ctx.pr = ctx.pr_lwir = None
    if cams:
        ctx.pr = _timed(ctx, 'perception', lambda: PR.analyse(
            p_vis, engine, None, pst, progress=lambda f, *_: prog(0.02 + 0.45 * f, 'perception range'),
            cancel=cancel, weather='clear', compare=conds, weather_settings=wst))
        if ctx.pr is None and ctx.stop():
            return None
    else:
        ctx.errors['perception'] = 'No visible camera in the variant (LWIR and event cameras have their own rows).'
    lw = [c for c in other if _modality(p.module_for(c)) == 'lwir']
    if lw:                                  # the LWIR cameras' geometry alone (pixels on target through their lenses)
        p_lw = _hide(p, [c for c in v.cameras if c.visible and c not in lw])
        ctx.pr_lwir = _timed(ctx, 'perception', lambda: PR.analyse(p_lw, engine, None, pst, cancel=cancel,
                                                                    weather='clear'))
    if ctx.stop():
        return None
    ctx.cmp = {row['id']: row for row in ((ctx.pr.weather or {}).get('compare', []) if ctx.pr is not None else [])}
    prog(0.5, 'lidar and radar coverage')
    ctx.sc = None
    if any(s.visible and s.kind in ('lidar', 'radar') for s in v.sensors):
        from . import sensor_coverage as SC
        ctx.sc = _timed(ctx, 'sensor_coverage', lambda: SC.run(p, engine, None, None, cancel=cancel))
    if ctx.stop():
        return None
    prog(0.65, 'sun and dirt exposure')
    from . import exposure as EXP
    ctx.ex = _timed(ctx, 'exposure', lambda: EXP.run(p, engine, None, None, cancel=cancel))
    if ctx.stop():
        return None
    prog(0.75, 'blind volume')
    from . import volume as VOL
    vst = VOL.VolumeSettings.from_dict(getattr(p.analysis, 'volume', None))
    objs = tuple(VOL.DEFAULT_OBJECTS)
    ctx.vr = _timed(ctx, 'volume', lambda: VOL.analyse_volume(p, engine, None, vst.height, vst.dz, vst.cell or None,
                                                              objects=objs, cancel=cancel)) if cams else None
    if ctx.stop():
        return None
    ctx.wst = wst
    ctx.conds = {c.id: c for c in conds}
    ctx.drops = {}
    try:
        scores = W._dirt_scores(p, vg)
        wiped = W.wiped_shares(p, vg)
        lamps = W.lamp_positions(vg, wst)
        for c in conds:
            if c.rain_mm_h > 0 or c.snow_mm_h > 0 or c.visibility_m > 0:
                ctx.drops[c.id] = W.sensor_rows(p, engine, c, wst, scores, lamps, wiped)
    except Exception as e:                                       # noqa: BLE001
        ctx.errors['weather'] = f'{type(e).__name__}: {e}'
    ctx.cleaning = _cleaning_cover(p)
    prog(0.85, 'matrix')
    rows = []
    for t in TRIGGERS:
        if ctx.stop():
            return None
        r = _trigger_row(ctx, t)
        if r is None:
            continue
        rows.append(r)
    res = SotifResult(st, rows, {}, [], [], [], [], {}, classes,
                      {'front_x': front_x, 'centre': list(centre), 'lane_half_m': st.lane_half_m}, {},
                      v.name, v.uid)
    res.odd = _odd(res)
    res.areas = _areas(res)
    res.trace = _trace(p, res)
    res.proposals = requirement_proposal(res)
    res.findings = _findings(res)
    res.studies = _studies(ctx)
    res.metrics = {'seconds': time.perf_counter() - t0, 'rows': len(rows),
                   'cells': sum(len(r['cells']) for r in rows),
                   'insufficient': sum(c['verdict'] == 'insufficient' for r in rows for c in r['cells']),
                   'not_evaluated': sum(c['verdict'] == 'not_evaluated' for r in rows for c in r['cells'])}
    res.key = cache_key(project, st, v)
    _cache_put(res.key, res)
    prog(1.0, 'done')
    return res


# ---------------------------------------------------------------------- per-trigger figures
def _class_index(ctx, name, classes) -> int | None:
    return next((k for k, c in enumerate(classes) if getattr(c, 'name', c) == name), None)


def _lidar_far(ctx, name, alpha=0.0, drop=()) -> np.ndarray | None:
    """The lidars' range per azimuth for a class (perception's azimuths), each capped by its datasheet range derated
    for extinction alpha (1/m); lidars whose tag is in drop left out. None: no lidar."""
    from . import weather as W
    sc = ctx.sc
    if sc is None or not sc.lidars:
        return None
    k = _class_index(ctx, name, sc.classes)
    if k is None:
        return None
    out = None
    for L in sc.lidars:
        if L.sensor.tag in drop:
            continue
        cap = W.lidar_range(float(L.spec.range_m), alpha) if L.spec.range_m > 0 else math.inf
        f = np.minimum(np.nan_to_num(L.far[k], nan=0.0), cap)
        out = f if out is None else np.maximum(out, f)
    return _regrid(ctx, sc.az, out) if out is not None else np.zeros(len(_az(ctx)))


def _radar_far(ctx, name, db_km=0.0) -> np.ndarray | None:
    from . import weather as W
    sc = ctx.sc
    if sc is None or not sc.radars:
        return None
    k = _class_index(ctx, name, sc.classes)
    if k is None:
        return None
    out = None
    for R_ in sc.radars:
        f = np.nan_to_num(R_.far[k], nan=0.0)
        if db_km > 0:
            f = np.where(f > 0, W.radar_range(f, db_km), 0.0)
        out = f if out is None else np.maximum(out, f)
    return _regrid(ctx, sc.az, out)


def _az(ctx):
    if ctx.pr is not None:
        return ctx.pr.az
    if ctx.sc is not None:
        return ctx.sc.az
    return np.arange(360.0)


def _regrid(ctx, az_src, arr):
    """arr over az_src onto the perception's azimuths (nearest sample)."""
    az = _az(ctx)
    if arr is None:
        return None
    if len(az_src) == len(az) and np.allclose(az_src, az):
        return np.asarray(arr, float)
    A = len(az_src)
    idx = np.rint((np.asarray(az) - az_src[0]) / (360.0 / A)).astype(int) % A
    return np.asarray(arr, float)[idx]


def _cam_far(ctx, name, cond_id='clear', drop=()) -> np.ndarray | None:
    """The cameras' range per azimuth for a class under a weather condition (the perception's compare row); with drop
    (camera tags) the clear-weather range without those cameras. None: no camera result."""
    pr = ctx.pr
    if pr is None:
        return None
    k = _class_index(ctx, name, pr.classes)
    if k is None:
        return None
    if drop:
        keep = [i for i, c in enumerate(pr.cams) if c[1] not in drop]
        return pr.cam_far[keep, k].max(0) if keep else np.zeros(len(pr.az))
    row = ctx.cmp.get(cond_id)
    return None if row is None else row['far'][k]


def _range_cell(ctx, tid, name, cam, lid, rad, sources, road, extra_delay=0.0, note='') -> dict:
    st = ctx.st
    az = _az(ctx)
    have = {'camera': cam is not None, 'lidar': lid is not None, 'radar': rad is not None}
    basis_ok = {'cameras': have['camera'], 'lidars': have['lidar'], 'radars': have['radar'],
                'any': any(have.values()), 'two': sum(have.values()) >= 2}[st.basis]
    by_type = {k: corridor_range(x, az, ctx.centre, ctx.front_x, st.lane_half_m) if x is not None else None
               for k, x in (('camera', cam), ('lidar', lid), ('radar', rad))}
    cell = {'insufficiency': 'range', 'class': name, 'sensor': 'rig', 'unit': 'm', 'road': road,
            'by_type_m': {k: _r(v, 2) for k, v in by_type.items()}, 'sources': sources, 'note': note}
    if not basis_ok:
        why = ('fewer than two sensor types in the rig' if st.basis == 'two' else
               f'no {st.basis[:-1]} in the rig or its study did not run')
        cell.update(value=None, threshold=None, verdict='not_evaluated', reason=f'Basis "{BASIS_LABEL[st.basis]}": '
                                                                                 f'{why}.')
        return cell
    far = combine(cam, lid, rad, st.basis)
    R = corridor_range(far, az, ctx.centre, ctx.front_x, st.lane_half_m)
    stop = stop_of(st, road, extra_delay)
    v_t = st.target_kmh / 3.6
    need = stopping_distance(v_t, stop) + st.margin_m
    vmax = max_safe_speed(R - st.margin_m, stop)
    cell.update(value=R, threshold=need, verdict='ok' if R >= need else 'insufficient', max_kmh=kmh(vmax),
                extra_delay_s=extra_delay, reason='')
    return cell


def _fov_cell(sensor, kind, share, source, note='', threshold=None, evaluated=True) -> dict:
    th = threshold
    if not evaluated:
        return {'insufficiency': 'fov', 'class': '', 'sensor': sensor, 'kind': kind, 'unit': 'share', 'value': None,
                'threshold': th, 'verdict': 'not_evaluated', 'sources': [source], 'note': note, 'reason': note}
    return {'insufficiency': 'fov', 'class': '', 'sensor': sensor, 'kind': kind, 'unit': 'share',
            'value': float(share), 'threshold': th, 'verdict': 'insufficient' if share > th else 'ok',
            'sources': [source], 'note': note, 'reason': ''}


def _range_cells(ctx, tid, road, cam_fn, lid_fn, rad_fn, sources, extra_delay=0.0, note='') -> list:
    return [_range_cell(ctx, tid, n, cam_fn(n), lid_fn(n), rad_fn(n), sources, road, extra_delay, note)
            for n in ctx.classes]


def _trigger_row(ctx, t) -> dict | None:
    from . import weather as W
    tid, label, kind, prm, _road, note = t
    st = ctx.st
    road = st.road(tid)
    p, v = ctx.p, ctx.p.variant
    row = {'id': tid, 'label': label, 'kind': kind, 'road': road, 'note': note, 'in_odd': tid in st.odd,
           'cells': [], 'mitigated': [], 'detail': ''}
    src_cam = _source('perception', 'perception', 'analyse', 'far range per azimuth (compare row)')
    src_lid = _source('sensor_coverage', 'sensor_coverage', 'run', 'lidar range per class, capped by '
                                                                  'weather.lidar_range of its datasheet range')
    src_rad = _source('sensor_coverage', 'sensor_coverage', 'run', 'radar range per class, derated by '
                                                                  'weather.radar_range')
    if kind == 'weather':
        c = ctx.conds.get(tid) if tid != 'clear' else W.PRESET['clear']
        alpha = W.optical_extinction(c, ctx.wst.lidar_nm) if not c.clear else 0.0
        db = W.radar_db_km(c, ctx.wst.radar_ghz, ctx.wst.temp_c) if not c.clear else 0.0
        row['condition'] = c.to_doc()
        row['detail'] = (f'Lidar extinction {alpha * 1000:.3g} /km at {ctx.wst.lidar_nm:g} nm; radar attenuation '
                         f'{db:.3g} dB/km at {ctx.wst.radar_ghz:g} GHz.') if not c.clear else 'No weather model.'
        row['cells'] = _range_cells(ctx, tid, road, lambda n: _cam_far(ctx, n, c.id), lambda n: _lidar_far(ctx, n, alpha),
                                    lambda n: _radar_far(ctx, n, db), [src_cam, src_lid, src_rad])
        for r in ctx.drops.get(c.id, []):
            if r['kind'] == 'camera' and r.get('lens_drops', 0) > 0:
                row['cells'].append(_fov_cell(r['tag'], 'camera', r['lens_drops'], _source(
                    'weather', 'weather', 'lens_drop_share', 'share of the image under drops (heuristic)'),
                    'Image under drops (heuristic, core.weather).', st.fov_max_share))
            elif r['kind'] == 'lidar' and r.get('false_returns', 0) > 0:
                row['cells'].append(_fov_cell(r['tag'], 'lidar', r['false_returns'], _source(
                    'weather', 'weather', 'false_return_share', 'beams with a false near return (heuristic)'),
                    'False near returns from droplets or flakes (heuristic, core.weather).', st.fov_max_share))
        return row
    if kind == 'sun':
        ex = ctx.ex
        src = _source('exposure', 'exposure', 'run', 'share of driving time with the low sun in the image')
        if ex is None:
            row['cells'] = [_na_range(ctx, n, road, 'The sun and dirt exposure study did not run.') for n in ctx.classes]
            return row
        blinded = [(it.tag, it.sun['low']) for it in ex.items if it.kind == 'camera' and it.sun is not None
                   and it.sun['low'] > st.sun_min_share]
        row['detail'] = ('Cameras with the low sun in their image: ' + ', '.join(
            f'{tg} {100 * s:.1f} % of driving time' for tg, s in blinded) + '.') if blinded else \
            'No camera has the low sun in its image in this campaign.'
        row['cells'] = _worst_removal(ctx, tid, road, [tg for tg, _ in blinded], [], [src, src_cam, src_lid, src_rad],
                                      'blinded by the low sun')
        for tg, s in blinded:
            row['cells'].append(_fov_cell(tg, 'camera', 1.0, src, f'Blinded while the sun is low in its image '
                                                                 f'({100 * s:.1f} % of driving time).',
                                          st.fov_max_share))
        return row
    if kind == 'dirt':
        ex = ctx.ex
        src = _source('exposure', 'exposure', 'run', 'dirt and spray exposure score (heuristic)')
        if ex is None:
            row['cells'] = [_na_range(ctx, n, road, 'The sun and dirt exposure study did not run.') for n in ctx.classes]
            return row
        dirty = [(it.tag, it.kind, it.dirt['score'], it.uid) for it in ex.items
                 if it.kind in ('camera', 'lidar') and it.dirt and it.dirt.get('score', 0) >= st.dirt_min_score]
        cams_lost, lids_lost = [], []
        for tg, k, s, uid in dirty:
            cov = ctx.cleaning.get(uid)
            if cov is not None and cov >= st.cleaning_min_cover:
                row['mitigated'].append({'sensor': tg, 'score': s, 'cleaning_cover': cov,
                                         'text': f'{tg}: dirt score {s:.2f}, its washer covers {100 * cov:.0f} % of the '
                                                 'window (core.cleaning)'})
                continue
            (cams_lost if k == 'camera' else lids_lost).append(tg)
        row['detail'] = ('Obscured (dirt score ≥ ' + f'{st.dirt_min_score:g}): ' + ', '.join(
            f'{tg} {s:.2f}' for tg, k, s, _ in dirty) + '.') if dirty else \
            f'No camera or lidar scores {st.dirt_min_score:g} or more.'
        row['cells'] = _worst_removal(ctx, tid, road, cams_lost, lids_lost, [src, src_cam, src_lid, src_rad],
                                      'obscured by dirt')
        for tg, k, s, uid in dirty:
            m = next((x for x in row['mitigated'] if x['sensor'] == tg), None)
            row['cells'].append(_fov_cell(tg, k, 0.0 if m else 1.0, src, m['text'] if m else
                                          f'Obscured: dirt score {s:.2f} (heuristic), no washer covers it.',
                                          st.fov_max_share))
        return row
    if kind == 'blind':
        vr = ctx.vr
        src = _source('volume', 'volume', 'analyse_volume', 'object_seen["Child 0.9 m"]: ground where a child is '
                                                            'seen by no camera')
        if vr is None or CHILD not in getattr(vr, 'object_seen', {}):
            row['cells'] = [{'insufficiency': 'child', 'class': 'Child', 'sensor': 'rig', 'unit': 'm', 'value': None,
                             'threshold': 0.0, 'verdict': 'not_evaluated', 'sources': [src],
                             'reason': ctx.errors.get('volume', 'The blind volume needs a vehicle body and a '
                                                                'visible camera.'), 'note': ''}]
            return row
        und = vr.roi & ~vr.object_seen[CHILD]
        X, Y = np.meshgrid(vr.xs, vr.ys)
        ahead = und & (X >= ctx.front_x) & (np.abs(Y - ctx.centre[1]) <= st.lane_half_m)
        reach = float((X[ahead] - ctx.front_x).max() + vr.cell / 2) if ahead.any() else 0.0
        area_ahead = float(ahead.sum() * vr.cell ** 2)
        area_all = float(und.sum() * vr.cell ** 2)
        row['detail'] = (f'A child 0.9 m tall stays unseen on {area_all:.3f} m² around the car; in the lane ahead '
                         f'{area_ahead:.3f} m², up to {reach:.2f} m from the bumper.')
        row['cells'] = [{'insufficiency': 'child', 'class': 'Child', 'sensor': 'rig', 'unit': 'm', 'value': reach,
                         'threshold': 0.0, 'verdict': 'insufficient' if area_ahead > 0 else 'ok', 'sources': [src],
                         'area_ahead_m2': area_ahead, 'area_m2': area_all,
                         'reach_m': float(vr.metrics['objects'][CHILD]['reach_m']), 'reason': '',
                         'note': 'Start-off: the lane ahead of the bumper must be seen clear before moving off.'}]
        row['start_off'] = area_ahead <= 0
        return row
    if kind == 'housing':
        src_ex = _source('exposure', 'exposure', 'run', 'share of the image circle the body, rig and housings cover')
        src_sc = _source('sensor_coverage', 'sensor_coverage', 'run', 'share of the lidar\'s points on the vehicle')
        row['cells'] = _range_cells(ctx, tid, road, lambda n: _cam_far(ctx, n, 'clear'), lambda n: _lidar_far(ctx, n),
                                    lambda n: _radar_far(ctx, n), [src_cam, src_lid, src_rad],
                                    note='The occlusion is in every range already (the clear-day figures).')
        if ctx.ex is not None:
            for it in ctx.ex.items:
                if it.kind == 'camera' and it.image is not None:
                    row['cells'].append(_fov_cell(it.tag, 'camera', _body_share(it.image), src_ex,
                                                  'Image circle covered by the body, rig parts and housings.',
                                                  st.fov_max_share))
        if ctx.sc is not None:
            for L in ctx.sc.lidars:
                row['cells'].append(_fov_cell(L.sensor.tag, 'lidar', float(L.blocked), src_sc,
                                              'Points of each frame that land on the vehicle or the rig.',
                                              st.fov_max_share))
        return row
    if kind == 'tunnel':
        tun = _tunnel(ctx)
        src = _source('isp', 'isp_photometrics', 'daylight, hdr_levels', 'luminance step at the tunnel exit and '
                                                                         'the HDR headroom')
        row['detail'] = tun['text']
        row['tunnel'] = tun
        row['cells'] = _range_cells(ctx, tid, road, lambda n: _cam_far(ctx, n, 'clear'), lambda n: _lidar_far(ctx, n),
                                    lambda n: _radar_far(ctx, n), [src, src_cam, src_lid, src_rad],
                                    extra_delay=tun['blind_s'],
                                    note=f'The cameras are blind for {tun["blind_s"]:.2f} s: added to the delay.')
        for c in tun['cameras']:
            row['cells'].append(_fov_cell(c['tag'], 'camera', 1.0 if c['blind_s'] > 0 else 0.0, src,
                                          f'Saturated for {c["blind_s"]:.2f} s.' if c['blind_s'] > 0 else
                                          'Its HDR headroom covers the step.', st.fov_max_share))
        return row
    if kind == 'thermal':
        lw = [c for c in v.cameras if c.visible and _modality(p.module_for(c)) == 'lwir']
        if not lw:
            return None
        return _thermal_row(ctx, row, lw, road)
    if kind == 'event':
        ev = [c for c in v.cameras if c.visible and _modality(p.module_for(c)) == 'event']
        if not ev:
            return None
        return _event_row(ctx, row, ev, road)
    return None


def _na_range(ctx, name, road, why) -> dict:
    return {'insufficiency': 'range', 'class': name, 'sensor': 'rig', 'unit': 'm', 'road': road, 'value': None,
            'threshold': None, 'verdict': 'not_evaluated', 'reason': why, 'sources': [], 'note': ''}


def _worst_removal(ctx, tid, road, cams, lids, sources, why) -> list:
    """Per class the range cell with each listed camera or lidar removed in turn, the worst kept (the clear day's
    when none is listed); the cell names the sensor that sets it."""
    if not cams and not lids:
        cells = _range_cells(ctx, tid, road, lambda n: _cam_far(ctx, n, 'clear'), lambda n: _lidar_far(ctx, n),
                             lambda n: _radar_far(ctx, n), sources, note='No sensor lost: the clear-day figures.')
        return cells
    out = []
    for n in ctx.classes:
        worst = None
        for tg in cams:
            c = _range_cell(ctx, tid, n, _cam_far(ctx, n, drop=(tg,)), _lidar_far(ctx, n), _radar_far(ctx, n),
                            sources, road, note=f'{tg} {why}.')
            c['lost'] = tg
            if worst is None or (c['value'] if c['value'] is not None else math.inf) < \
                    (worst['value'] if worst['value'] is not None else math.inf):
                worst = c
        for tg in lids:
            c = _range_cell(ctx, tid, n, _cam_far(ctx, n, 'clear'), _lidar_far(ctx, n, drop=(tg,)),
                            _radar_far(ctx, n), sources, road, note=f'{tg} {why}.')
            c['lost'] = tg
            if worst is None or (c['value'] if c['value'] is not None else math.inf) < \
                    (worst['value'] if worst['value'] is not None else math.inf):
                worst = c
        out.append(worst)
    return out


def _body_share(image: dict) -> float:
    """Share of the image circle (inside the sensor) that core.exposure's body mask covers."""
    body = np.asarray(image['body'], bool)
    nr, nc = body.shape
    us = (np.arange(nc) + 0.5) / nc * image['w']
    vs = (np.arange(nr) + 0.5) / nr * image['h']
    U, V = np.meshgrid(us, vs)
    rx, ry = max(float(image['rx']), 1e-9), max(float(image['ry']), 1e-9)
    inside = ((U - image['cx']) / rx) ** 2 + ((V - image['cy']) / ry) ** 2 <= 1.0
    n = int(inside.sum())
    return float((body & inside).sum() / n) if n else 0.0


def _forward_cameras(ctx) -> list:
    """Visible cameras looking within 60° of straight ahead (what sees the road at a tunnel exit)."""
    out = []
    for c in ctx.p.variant.cameras:
        if c.visible and abs((float(c.yaw) + 180.0) % 360.0 - 180.0) <= 60.0:
            out.append(c)
    return out


def _tunnel(ctx) -> dict:
    """The tunnel exit: the luminance step (EV) from the ISP study's tunnel case and per forward camera the time it
    is saturated: (step − headroom) / ae_ev_per_s, headroom = log2(HDR top / AE target) from its dynamic range."""
    from . import isp_photometrics as ISP
    ist = ISP.settings_of(ctx.p)
    day = ISP.daylight(ist.tunnel_sun_el_deg, ist.turbidity)
    L_out = ist.ground_albedo * float(np.asarray(day['E_h']).ravel()[0]) / math.pi
    L_in = ist.interior_cd
    step = math.log2(max(L_out, 1e-9) / max(L_in, 1e-9))
    cams = []
    for c in _forward_cameras(ctx):
        m = ctx.p.module_for(c)
        if _modality(m) not in ('rgb', 'nir'):
            continue
        dr = float(getattr(m, 'dynamic_range_db', 0.0) or 0.0)
        lv = ISP.hdr_levels(ist, dr)
        head = math.log2(lv['top'] / max(ist.ae_target, 1e-9))
        over = max(step - head, 0.0)
        cams.append({'tag': c.tag, 'dynamic_range_db': lv['dr_db'], 'dr_given': lv['dr_given'], 'headroom_ev': head,
                     'blind_s': over / ctx.st.ae_ev_per_s})
    blind = min((c['blind_s'] for c in cams), default=0.0)
    text = (f'Tunnel road {L_in:g} cd/m² to the daylit road {L_out:,.0f} cd/m² (sun {ist.tunnel_sun_el_deg:g}° up, '
            f'albedo {ist.ground_albedo:g}): a step of {step:.1f} EV. ' +
            ('; '.join(f'{c["tag"]}: headroom {c["headroom_ev"]:.1f} EV'
                       + (f', saturated {c["blind_s"]:.2f} s' if c['blind_s'] > 0 else ', covers it')
                       + ('' if c['dr_given'] else ' (dynamic range not stated: typical)') for c in cams)
             if cams else 'No camera looks ahead.') + f' AE adapts at {ctx.st.ae_ev_per_s:g} EV/s (assumed).')
    return {'L_in': L_in, 'L_out': L_out, 'step_ev': step, 'cameras': cams, 'blind_s': blind, 'text': text}


def thermal_far(ctx, lw) -> tuple[dict | None, str]:
    """{class name: (A,) far range per azimuth} of the LWIR cameras on a clear night, and the source in words.
    core.thermal_ir.sotif_ranges(project, engine, condition) when the build has it; else core.thermal_ir.run on its
    'clear_night' condition: each LWIR camera's detection range along its heading (radiometry, NETD, DRI pixels) caps
    that camera's own geometric far range from the perception study (pixels on target through its lens), measured
    from the vehicle centre (the camera's distance from it added). None when neither is available."""
    TI = _optional_module('thermal_ir')
    if TI is None:
        return None, 'core.thermal_ir is not part of this build'
    fn = getattr(TI, 'sotif_ranges', None)
    if fn is not None:
        out = fn(ctx.p, ctx.engine, 'night') or {}
        return {k: np.nan_to_num(np.asarray(v, float), nan=0.0) for k, v in out.items()}, \
            'core.thermal_ir.sotif_ranges'
    pr = ctx.pr_lwir
    if not hasattr(TI, 'run') or pr is None:
        return None, 'core.thermal_ir has no study this build can read'
    tst = TI.ThermalSettings.from_dict(dict(TI.settings_of(ctx.p).to_dict(), conditions=['clear_night']))
    tr = TI.run(ctx.p, ctx.engine, tst, cancel=ctx.cancel)
    if tr is None:
        return None, 'cancelled'
    tags = [c[1] for c in pr.cams]
    out = {}
    for name in ctx.classes:
        k = _class_index(ctx, name, pr.classes)
        if k is None:
            continue
        far = np.zeros(len(pr.az))
        for c in lw:
            if c.tag not in tags:
                continue
            row = tr.row(c.tag, 'clear_night', name.lower(), 'detection')
            if row is None:
                continue
            off = float(np.hypot(c.x - ctx.centre[0], c.y - ctx.centre[1]))
            far = np.maximum(far, np.minimum(pr.cam_far[tags.index(c.tag), k], float(row['range_m']) + off))
        out[name] = _regrid(ctx, pr.az, far)
    return out, ('core.thermal_ir.run (clear night, detection: NETD and DRI pixels) capping each LWIR camera\'s '
                 'geometric far range from core.perception')


def _thermal_row(ctx, row, lw, road) -> dict:
    """Night on an unlit road with the LWIR cameras (thermal_far) besides the visible cameras' night range."""
    src = _source('thermal', 'thermal_ir', 'run / sotif_ranges', 'LWIR far range per azimuth at night')
    tags = ', '.join(c.tag for c in lw)
    try:
        th, how = thermal_far(ctx, lw)
    except Exception as e:                                       # noqa: BLE001
        th, how = None, f'core.thermal_ir failed: {type(e).__name__}: {e}'
    if th is None:
        row['detail'] = f'The rig has LWIR cameras ({tags}); their night range is not available: {how}.'
        row['cells'] = [_na_range(ctx, n, road, row['detail']) for n in ctx.classes]
        return row

    def cam(n):
        vis = _cam_far(ctx, n, 'night')
        t = th.get(n)
        if t is None:
            return vis
        return t if vis is None else np.maximum(vis, t)
    row['detail'] = f'LWIR cameras {tags} with the visible cameras at night; {how}.'
    row['cells'] = _range_cells(ctx, row['id'], road, cam, lambda n: _lidar_far(ctx, n), lambda n: _radar_far(ctx, n),
                                [src, _source('perception', 'perception', 'analyse', 'far range, night')])
    return row


def event_blind(ctx, ev, step_ev) -> tuple[dict | None, dict, str]:
    """({event camera tag: seconds it cannot see the target after the tunnel exit}, {frame camera tag: the same from
    the event study's own AE simulation}, the source). core.event_camera.sotif_tunnel(project, step_ev) when the build
    has it; else core.event_camera.tunnel_study per event camera (its rendered tunnel exit, the event model against
    its RGB comparator's AE and HDR)."""
    EC = _optional_module('event_camera')
    if EC is None:
        return None, {}, 'core.event_camera is not part of this build'
    fn = getattr(EC, 'sotif_tunnel', None)
    if fn is not None:
        return {k: float(v) for k, v in (fn(ctx.p, step_ev) or {}).items()}, {}, 'core.event_camera.sotif_tunnel'
    if not hasattr(EC, 'tunnel_study'):
        return None, {}, 'core.event_camera has no tunnel study this build can read'
    est = EC.settings_of(ctx.p)
    out, rgb = {}, {}
    for c in ev:
        tn = EC.tunnel_study(ctx.p, c, EC.comparator(ctx.p, c, est), EC.params_of(ctx.p.module_for(c), est), est,
                             cancel=ctx.cancel)
        if tn is None:
            return None, {}, 'cancelled'
        out[c.tag] = float(tn['event']['blind_s'])
        if tn.get('rgb'):
            rgb[tn['rgb']['tag']] = float(tn['rgb']['blind_s'])
    return out, rgb, 'core.event_camera.tunnel_study (the rendered tunnel exit, the event model and the RGB AE)'


def _event_row(ctx, row, ev, road) -> dict:
    """A tunnel exit with the event cameras: an event camera that looks ahead and sees the target sooner than the
    frame cameras shortens the blind time added to the delay."""
    tun = _tunnel(ctx)
    tags = ', '.join(c.tag for c in ev)
    src = _source('event', 'event_camera', 'tunnel_study / sotif_tunnel', 'event camera blind time at the tunnel '
                                                                          'exit')
    try:
        blind, rgb, how = event_blind(ctx, ev, tun['step_ev'])
    except Exception as e:                                       # noqa: BLE001
        blind, rgb, how = None, {}, f'core.event_camera failed: {type(e).__name__}: {e}'
    if blind is None:
        row['detail'] = f'The rig has event cameras ({tags}); their tunnel figures are not available: {how}.'
        row['cells'] = [_na_range(ctx, n, road, row['detail']) for n in ctx.classes]
        return row
    fwd = {c.tag for c in _forward_cameras(ctx)}
    ahead = [b for t, b in blind.items() if t in fwd]
    extra = min(min(ahead, default=math.inf), tun['blind_s'])
    row['event'] = {'event_blind_s': blind, 'frame_blind_s_event_study': rgb, 'frame_blind_s': tun['blind_s']}
    row['detail'] = (f'Event cameras {tags}: blind ' + ', '.join(f'{k} {v:.2f} s' for k, v in blind.items())
                     + (f'; their RGB comparators (the event study\'s AE): ' + ', '.join(
                         f'{k} {v:.2f} s' for k, v in rgb.items()) if rgb else '')
                     + f'; the frame cameras here {tun["blind_s"]:.2f} s ({how}).')
    row['cells'] = _range_cells(ctx, row['id'], road, lambda n: _cam_far(ctx, n, 'clear'),
                                lambda n: _lidar_far(ctx, n), lambda n: _radar_far(ctx, n), [src],
                                extra_delay=extra, note=f'Blind time {extra:.2f} s added to the delay.')
    return row


def _cleaning_cover(p) -> dict:
    """{sensor uid: the share of its window the washers cover} from core.cleaning when it is in the build."""
    CL = _optional_module('cleaning')
    fn = getattr(CL, 'washer_cover', None) if CL is not None else None
    if fn is None:
        return {}
    try:
        return dict(fn(p) or {})
    except Exception:                                            # noqa: BLE001
        return {}


# ---------------------------------------------------------------------- ODD, areas, trace, findings
def _binding(row) -> dict | None:
    rc = [c for c in row['cells'] if c['insufficiency'] == 'range' and c['verdict'] != 'not_evaluated']
    if not rc:
        return None
    return min(rc, key=lambda c: c['value'])


def _odd(res: SotifResult) -> dict:
    st = res.settings
    table = []
    for r in res.rows:
        b = _binding(r)
        stop = stop_of(st, r['road'], b.get('extra_delay_s', 0.0) if b else 0.0)
        vmax = b['max_kmh'] if b else None
        r['max_kmh'] = vmax
        r['binding'] = b['class'] if b else None
        r['binding_range_m'] = b['value'] if b else None
        r['decel_ms2'] = stop.decel
        r['delay_s'] = stop.delay_s
        r['verdict'] = ('not_evaluated' if not r['cells'] or all(c['verdict'] == 'not_evaluated' for c in r['cells'])
                        else 'insufficient' if any(c['verdict'] == 'insufficient' for c in r['cells']) else 'ok')
        table.append(r['id'])
    inc = [r for r in res.rows if r['in_odd'] and r.get('max_kmh') is not None]
    lim = min(inc, key=lambda r: r['max_kmh'], default=None)
    return {'speed_kmh': lim['max_kmh'] if lim else None, 'binding': lim['id'] if lim else None,
            'binding_label': lim['label'] if lim else None, 'binding_class': lim['binding'] if lim else None,
            'target_kmh': st.target_kmh,
            'meets_target': (lim['max_kmh'] >= st.target_kmh - 1e-9) if lim else None,
            'included': [r['id'] for r in res.rows if r['in_odd']],
            'excluded': [r['id'] for r in res.rows if not r['in_odd']],
            'model': st.model()['label'], 'model_source': st.model()['source'], 'road_source': ROAD_SOURCE,
            'margin_m': st.margin_m, 'margin_source': MARGIN_SOURCE if st.margin_m == 2.0 else 'given by the user'}


def odd_speed(rows, include) -> float | None:
    """The ODD speed limit (km/h) over the rows whose ids are in include: the minimum of their maximum safe speeds."""
    vals = [r['max_kmh'] for r in rows if r['id'] in include and r.get('max_kmh') is not None]
    return min(vals) if vals else None


def _areas(res: SotifResult) -> list:
    a1, a2, a3 = [], [], []
    for r in res.rows:
        if r['verdict'] == 'not_evaluated' or any(c['verdict'] == 'not_evaluated' for c in r['cells']):
            why = '; '.join(sorted({c.get('reason', '') for c in r['cells'] if c['verdict'] == 'not_evaluated'
                                    and c.get('reason')}))
            a3.append({'id': r['id'], 'label': r['label'], 'text': why or 'Not evaluated.'})
        if r['verdict'] == 'insufficient':
            bad = [c for c in r['cells'] if c['verdict'] == 'insufficient']
            a2.append({'id': r['id'], 'label': r['label'], 'in_odd': r['in_odd'],
                       'text': '; '.join(_cell_text(c) for c in bad[:4])})
        elif r['verdict'] == 'ok' and r['in_odd']:
            a1.append({'id': r['id'], 'label': r['label'], 'text': 'Every cell sufficient at '
                                                                   f'{res.settings.target_kmh:g} km/h.'})
    a3 += [{'id': f'not_modelled.{k}', 'label': k.capitalize(), 'text': f'Not modelled: {t}.'} for k, t in NOT_MODELLED]
    out = []
    for (n, lab, text), items in zip(AREAS, (a1, a2, a3, [])):
        out.append({'area': n, 'label': lab, 'text': text, 'items': items})
    return out


def _cell_text(c) -> str:
    if c['insufficiency'] == 'range':
        if c['value'] is None:
            return f'{c["class"]}: not evaluated'
        return (f'{c["class"]} detected to {c["value"]:.1f} m ahead, needs {c["threshold"]:.1f} m'
                + (f' (without {c["lost"]})' if c.get('lost') else ''))
    if c['insufficiency'] == 'child':
        return f'a child can hide {c["value"]:.2f} m ahead of the bumper' if c['value'] else 'a child stays unseen'
    if c['value'] is None:
        return f'{c["sensor"]}: not evaluated'
    return f'{c["sensor"]} loses {100 * c["value"]:.0f} % of its view'


def _trace(p, res: SotifResult) -> list:
    """Requirements of the scorecard that check what a cell measures: the perception range (class, front sector),
    the lidar and radar range, the child's undetected area, the sun and dirt exposure, and the ODD speed; each with the
    cells it relates to and the matrix's own figure for comparison (the scorecard's own value comes from its run)."""
    from . import requirements as RQ
    try:
        items = RQ.settings_of(p).items
    except Exception:                                            # noqa: BLE001
        return []
    out = []
    clear = res.row('clear')
    for it in items:
        m = it.get('metric', '')
        prm = it.get('params') or {}
        cells = []
        if m in ('perception.range', 'sensor_coverage.range', 'perception.short'):
            cls = prm.get('cls') or ''
            sector = prm.get('sector') or ''
            if sector in ('', 'front', 'front-left', 'front-right'):
                for r in res.rows:
                    for c in r['cells']:
                        if c['insufficiency'] == 'range' and (not cls or cls.lower() == c['class'].lower()):
                            cells.append(f'{r["id"]}/{c["class"]}')
        elif m == 'volume.undetected_m2' and (prm.get('object') or '').lower().startswith('child'):
            cells = ['blind/Child']
        elif m.startswith('exposure.'):
            cells = [f'{tid}/fov' for tid in ('sun_low', 'dirt')]
        elif m.startswith('sotif.'):
            cells = ['odd']
        if not cells:
            continue
        fig = None
        if m in ('perception.range', 'sensor_coverage.range') and clear is not None:
            cls = (prm.get('cls') or '').lower()
            cc = [c for c in clear['cells'] if c['insufficiency'] == 'range' and c['value'] is not None
                  and (not cls or c['class'].lower() == cls)]
            if cc:
                fig = {'label': 'clear-day corridor range ahead (from the bumper)', 'value_m': _r(min(
                    c['value'] for c in cc), 2)}
        out.append({'requirement': it.get('id'), 'text': it.get('text', ''), 'metric': m, 'params': prm,
                    'op': it.get('op'), 'target': it.get('target'), 'priority': it.get('priority'),
                    'cells': cells[:24], 'matrix_figure': fig})
    return out


def requirement_proposal(res: SotifResult) -> list:
    """Requirements that would keep the ODD at its target speed: per critical class the perception range in the front
    sector at least the stopping distance at the target speed on a dry road plus the margin and the bumper's distance
    from the vehicle centre (the scorecard measures from the centre), as scorecard items to accept."""
    st = res.settings
    stop = stop_of(st, 'dry')
    need = stopping_distance(st.target_kmh / 3.6, stop) + st.margin_m
    off = res.geometry['front_x'] - res.geometry['centre'][0]
    out = []
    for n in res.classes:
        t = math.ceil(need + off)
        out.append({'metric': 'perception.range', 'params': {'cls': n, 'sector': 'front'}, 'op': '>=', 'target': t,
                    'priority': 'must',
                    'text': f'{n}s resolved at least {t:g} m ahead (stopping from {st.target_kmh:g} km/h, '
                            f'{st.model()["label"]}, dry road, plus {st.margin_m:g} m and the bumper offset '
                            f'{off:.1f} m)'})
    return out


def _findings(res: SotifResult) -> list:
    out = []
    st = res.settings
    odd = res.odd
    if odd['speed_kmh'] is not None and not odd['meets_target']:
        out.append({'severity': 'serious', 'kind': 'odd', 'trigger': odd['binding'],
                    'message': f'The ODD speed limit is {odd["speed_kmh"]:.0f} km/h, below the target '
                               f'{st.target_kmh:g} km/h: set by {odd["binding_label"].lower()} '
                               f'({odd["binding_class"].lower()}s).'})
    for r in res.rows:
        if not r['in_odd']:
            continue
        for c in r['cells']:
            if c['verdict'] != 'insufficient':
                continue
            if c['insufficiency'] == 'range':
                out.append({'severity': 'serious' if r['id'] == 'clear' else 'warning', 'kind': 'range',
                            'trigger': r['id'], 'message': f'{r["label"]}: {_cell_text(c)} at {st.target_kmh:g} km/h '
                                                           f'(safe up to {c["max_kmh"]:.0f} km/h).'})
            elif c['insufficiency'] == 'child':
                out.append({'severity': 'serious', 'kind': 'child', 'trigger': r['id'],
                            'message': f'{r["label"]}: a child 0.9 m tall can stand unseen up to {c["value"]:.2f} m '
                                       'ahead of the bumper: the path cannot be confirmed clear at start-off.'})
            else:
                note = (c.get('note') or '').rstrip('.')
                out.append({'severity': 'warning', 'kind': 'fov', 'trigger': r['id'],
                            'message': f'{r["label"]}: {_cell_text(c)}' + (f' ({note[0].lower() + note[1:]}).'
                                                                           if note else '.'),
                            'sensor': c['sensor']})
    ne = [r['label'] for r in res.rows if r['verdict'] == 'not_evaluated']
    if ne:
        out.append({'severity': 'warning', 'kind': 'not_evaluated', 'trigger': '',
                    'message': 'Not evaluated: ' + ', '.join(ne) + '.'})
    order = {s: i for i, s in enumerate(SEVERITIES)}
    out.sort(key=lambda f: order.get(f['severity'], 9))
    return out


def _studies(ctx) -> dict:
    from . import exposure as EXP
    from . import perception as PR
    from . import weather as W
    out = {}
    for sid, mod, ass in (('perception', 'perception', 'Objects resolved by pixels on target against typical '
                                                       'minimums; weather thresholds of core.weather.'),
                          ('weather', 'weather', W.ASSUMPTIONS),
                          ('sensor_coverage', 'sensor_coverage', 'Lidar points counted from the beam geometry; '
                                                                 'radar zones scaled by RCS; library figures nominal.'),
                          ('exposure', 'exposure', EXP.ASSUMPTIONS + ' ' + EXP.DIRT_MODEL),
                          ('volume', 'volume', 'ROI columns sampled level by level with the ground analysis\' '
                                               'visibility test.'),
                          ('isp', 'isp_photometrics', 'Typical tunnel luminances (CIE 88 interior zone), a clear '
                                                      'sky; the HDR merge is an assumption.')):
        out[sid] = {'module': f'vantage.core.{mod}', 'assumptions': ass,
                    'seconds': _r(ctx.seconds.get(sid), 3), 'error': ctx.errors.get(sid, '')}
    _ = PR
    return out


# =========================================================================== cache and issues
_CACHE: OrderedDict = OrderedDict()
_LOCK = threading.Lock()
CACHE_KEPT = 4


def cache_key(project, settings: SotifSettings | None = None, variant=None) -> str:
    """Everything the result depends on: the settings, the variant (cameras, modules, lenses, sensors), the vehicle
    mesh, the rig parts, the housings and the settings of the studies it reads."""
    from . import housings as H
    from .experiments import _mesh_key
    from .rig import rig_key
    st = settings or settings_of(project)
    v = variant or project.variant
    an = project.analysis
    doc = {'schema': SCHEMA, 'settings': st.to_dict(), 'variant': v.to_dict(),
           'cams': [H._cam_key(project, c) for c in v.cameras],
           'sensors': [project.sensor_model_for(s).to_dict() for s in v.sensors],
           'mesh': _mesh_key(getattr(project.vehicle, 'mesh', None)),
           'rig': rig_key([r for r in getattr(project, 'rig', ()) if r.visible]),
           'housings': repr(H.key(project, v)),
           'studies': {k: getattr(an, k, None) for k in ('perception', 'weather', 'sensor_coverage', 'exposure',
                                                         'volume', 'isp', 'cleaning', 'requirements')},
           'cell': getattr(an, 'cell_m', None)}
    return hashlib.sha256(json.dumps(doc, sort_keys=True, default=str).encode()).hexdigest()


def _cache_put(key, res):
    with _LOCK:
        _CACHE.pop(key, None)
        _CACHE[key] = res
        while len(_CACHE) > CACHE_KEPT:
            _CACHE.popitem(last=False)


def cached(project, settings: SotifSettings | None = None, variant=None) -> SotifResult | None:
    """The result for exactly this design and these settings when it has been computed, else None (cheap)."""
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
    """For the Issues list, from the cached result only (cheap; nothing until the page or the batch ran it): the ODD
    speed limit below its target, and a child hidden ahead at start-off."""
    from .rules import Issue
    res = cached(project)
    if res is None:
        return []
    out = []
    cams = {c.tag: c.uid for c in project.variant.cameras}
    for f in res.findings:
        if f['kind'] == 'odd':
            out.append(Issue('serious', 'SOTIF: ' + f['message'], 'ODD', [], 'sotif'))
        elif f['kind'] == 'child':
            out.append(Issue('serious', 'SOTIF: ' + f['message'], 'Blind volume', [], 'sotif'))
        elif f['kind'] == 'fov' and f.get('sensor') in cams:
            continue                                   # the studies that own them list those
    return out


# =========================================================================== output
def _r(v, d=3):
    if v is None:
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return round(v, d) if math.isfinite(v) else None


def summary(res: SotifResult) -> dict:
    """Compact figures (the assistant, the batch log): the ODD limit and what binds it, per condition its maximum safe
    speed and verdict, the findings."""
    return {'odd_speed_kmh': _r(res.odd['speed_kmh'], 1), 'target_kmh': res.settings.target_kmh,
            'meets_target': res.odd['meets_target'], 'binding': res.odd['binding'],
            'binding_class': res.odd['binding_class'], 'stop_model': res.odd['model'], 'basis': res.settings.basis,
            'conditions': [{'id': r['id'], 'label': r['label'], 'in_odd': r['in_odd'], 'road': r['road'],
                            'max_kmh': _r(r.get('max_kmh'), 1), 'binding': r.get('binding'),
                            'range_m': _r(r.get('binding_range_m'), 1), 'verdict': r['verdict']} for r in res.rows],
            'areas': {a['label']: len(a['items']) for a in res.areas},
            'findings': [{'severity': f['severity'], 'message': f['message']} for f in res.findings],
            'disclaimer': DISCLAIMER}


def _cell_doc(c) -> dict:
    out = {}
    for k, v in c.items():
        if isinstance(v, float):
            out[k] = _r(v, 4)
        elif isinstance(v, dict) and k == 'by_type_m':
            out[k] = v
        else:
            out[k] = v
    return out


def to_doc(res: SotifResult, project=None) -> dict:
    """sotif.json: the matrix (rows × cells with their sources), the ODD table, the scenario areas, the requirements
    trace and proposals, the studies' assumptions, the disclaimer."""
    st = res.settings
    return {
        'schema': SCHEMA, 'project': getattr(project, 'name', ''), 'variant': res.variant, 'frame': FRAME,
        'units': 'metres, km/h, seconds, m/s²; shares 0–1',
        'disclaimer': DISCLAIMER, 'assumptions': ASSUMPTIONS,
        'settings': st.to_dict(),
        'stop_model': dict(st.model(), id=st.stop_model), 'road_adhesion': ROAD_MU, 'road_source': ROAD_SOURCE,
        'geometry': {'front_bumper_x_m': _r(res.geometry['front_x'], 3),
                     'centre_m': [_r(q, 3) for q in res.geometry['centre']],
                     'lane_half_width_m': res.geometry['lane_half_m'],
                     'range': 'from the front bumper along the ego lane corridor'},
        'classes': res.classes,
        'insufficiencies': INSUFFICIENCY_LABEL,
        'matrix': [{'id': r['id'], 'label': r['label'], 'kind': r['kind'], 'road': r['road'], 'in_odd': r['in_odd'],
                    'note': r['note'], 'detail': r['detail'], 'verdict': r['verdict'],
                    'mitigated': r['mitigated'], 'cells': [_cell_doc(c) for c in r['cells']],
                    **({'tunnel': {k: (_r(v, 3) if isinstance(v, float) else v) for k, v in r['tunnel'].items()}}
                       if 'tunnel' in r else {})} for r in res.rows],
        'odd': {**{k: (_r(v, 2) if isinstance(v, float) else v) for k, v in res.odd.items()},
                'table': [{'id': r['id'], 'label': r['label'], 'in_odd': r['in_odd'], 'road': r['road'],
                           'mu': ROAD_MU[r['road']], 'decel_ms2': _r(r.get('decel_ms2'), 3),
                           'delay_s': _r(r.get('delay_s'), 3), 'binding_class': r.get('binding'),
                           'range_m': _r(r.get('binding_range_m'), 2), 'max_kmh': _r(r.get('max_kmh'), 2),
                           'verdict': r['verdict']} for r in res.rows]},
        'areas': res.areas,
        'requirements': {'trace': res.trace, 'proposals': res.proposals},
        'findings': res.findings,
        'studies': res.studies,
        'not_modelled': dict(NOT_MODELLED),
        'metrics': {k: (_r(v, 3) if isinstance(v, float) else v) for k, v in res.metrics.items()},
    }


CSV_COLUMNS = ('trigger', 'label', 'in_odd', 'road', 'insufficiency', 'class', 'sensor', 'value', 'unit',
               'threshold', 'verdict', 'max_kmh', 'study', 'function', 'note')


def to_csv(res: SotifResult) -> str:
    """One row per matrix cell."""
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(CSV_COLUMNS)
    for r in res.rows:
        for c in r['cells']:
            src = c.get('sources') or [{}]
            w.writerow([r['id'], r['label'], int(r['in_odd']), r['road'], c['insufficiency'], c.get('class', ''),
                        c.get('sensor', ''), '' if c.get('value') is None else f'{c["value"]:.4g}', c.get('unit', ''),
                        '' if c.get('threshold') is None else f'{c["threshold"]:.4g}', c['verdict'],
                        '' if c.get('max_kmh') is None else f'{c["max_kmh"]:.1f}',
                        '; '.join(s.get('study', '') for s in src), '; '.join(s.get('function', '') for s in src),
                        c.get('note') or c.get('reason', '')])
    return buf.getvalue()


def odd_csv(res: SotifResult) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(('trigger', 'label', 'in_odd', 'road', 'mu', 'decel_ms2', 'delay_s', 'binding_class', 'range_m',
                'max_kmh', 'verdict'))
    for r in res.rows:
        w.writerow([r['id'], r['label'], int(r['in_odd']), r['road'], ROAD_MU[r['road']],
                    f'{r.get("decel_ms2", 0):.3f}', f'{r.get("delay_s", 0):.3f}', r.get('binding') or '',
                    '' if r.get('binding_range_m') is None else f'{r["binding_range_m"]:.2f}',
                    '' if r.get('max_kmh') is None else f'{r["max_kmh"]:.1f}', r['verdict']])
    return buf.getvalue()


VERDICT_STYLE = {'ok': ('#0ca30c', '#ffffff', 'OK'), 'insufficient': ('#d03b3b', '#ffffff', 'INSUFFICIENT'),
                 'not_evaluated': ('#c98500', '#ffffff', 'NOT EVALUATED'), 'n/a': ('#898781', '#ffffff', '—')}


def _cell_html(c) -> str:
    bg, fg, _ = VERDICT_STYLE.get(c['verdict'], VERDICT_STYLE['n/a'])
    if c['insufficiency'] == 'range':
        txt = '—' if c['value'] is None else f'{c["value"]:.0f} / {c["threshold"]:.0f} m'
    elif c['insufficiency'] == 'child':
        txt = '—' if c['value'] is None else f'{c["value"]:.2f} m'
    else:
        txt = '—' if c['value'] is None else f'{c["sensor"]} {100 * c["value"]:.0f} %'
    return f'<td style="background:{bg};color:{fg};text-align:center">{html.escape(txt)}</td>'


def _fov_short(cells) -> str:
    return ', '.join(f'{c["sensor"]} {100 * c["value"]:.0f} %' for c in cells[:6]) + (' …' if len(cells) > 6 else '')


def report_html(res: SotifResult, project=None) -> str:
    """The report as one self-contained HTML page (also the PDF's source): disclaimer, ODD, matrix, ODD table, areas,
    requirements, findings, studies and their assumptions."""
    E = lambda s: html.escape(str(s))                      # noqa: E731
    st = res.settings
    odd = res.odd
    title = f'SOTIF insufficiency matrix · {getattr(project, "name", "")} · {res.variant}'
    h = [f'<html><head><meta charset="utf-8"><title>{E(title)}</title><style>'
         'body{font-family:sans-serif;font-size:9pt;color:#1b1b1b} h1{font-size:15pt} h2{font-size:11.5pt;'
         'margin-top:14pt} table{border-collapse:collapse;margin:4pt 0} td,th{border:1px solid #c8c6c0;padding:2pt '
         '4pt;font-size:8pt} th{background:#eceae4;text-align:left} .muted{color:#6b6a66} .box{border:1px solid '
         '#c98500;background:#fff6e0;padding:5pt}</style></head><body>',
         f'<h1>{E(title)}</h1>', f'<p class="box"><b>Not a certification.</b> {E(DISCLAIMER)}</p>']
    sp = '—' if odd['speed_kmh'] is None else f'{odd["speed_kmh"]:.0f} km/h'
    h.append(f'<h2>ODD speed limit: {E(sp)}</h2><p>Target {st.target_kmh:g} km/h · stop model: '
             f'{E(odd["model"])} ({E(odd["model_source"])}) · basis: {E(BASIS_LABEL[st.basis])} · lane corridor '
             f'±{st.lane_half_m:g} m · margin {st.margin_m:g} m. '
             + (f'Set by {E(odd["binding_label"])} ({E(odd["binding_class"])}).' if odd['binding'] else '') + '</p>')
    cls = res.classes
    h.append('<h2>Insufficiency matrix</h2><table><tr><th>Triggering condition</th><th>Road</th>'
             + ''.join(f'<th>{E(c)}: range / needed</th>' for c in cls)
             + '<th>Child hidden ahead</th><th>Field of view lost</th><th>Max safe speed</th></tr>')
    for r in res.rows:
        cells = {c['class']: c for c in r['cells'] if c['insufficiency'] == 'range'}
        child = next((c for c in r['cells'] if c['insufficiency'] == 'child'), None)
        fov = [c for c in r['cells'] if c['insufficiency'] == 'fov' and c['verdict'] == 'insufficient']
        h.append(f'<tr><td>{E(r["label"])}{"" if r["in_odd"] else " <span class=muted>(outside the ODD)</span>"}'
                 f'</td><td>{E(ROAD_LABEL[r["road"]])}</td>'
                 + ''.join(_cell_html(cells[c]) if c in cells else '<td>—</td>' for c in cls)
                 + (_cell_html(child) if child else '<td>—</td>')
                 + (f'<td style="background:#d03b3b;color:#fff">{E(_fov_short(fov))}</td>'
                    if fov else '<td>—</td>')
                 + f'<td>{"—" if r.get("max_kmh") is None else f"{r["max_kmh"]:.0f} km/h"}</td></tr>')
    h.append('</table><p class="muted">Range: the detection range in the ego lane corridor ahead of the bumper '
             'against the stopping distance at the target speed plus the margin. Every cell cites its study in the '
             'JSON (study, module, function).</p>')
    h.append('<h2>ODD table</h2><table><tr><th>Condition</th><th>In ODD</th><th>Road (μ)</th><th>Deceleration</th>'
             '<th>Delay</th><th>Binding class</th><th>Range</th><th>Max safe speed</th></tr>')
    for r in res.rows:
        h.append(f'<tr><td>{E(r["label"])}</td><td>{"yes" if r["in_odd"] else "no"}</td><td>{E(ROAD_LABEL[r["road"]])}'
                 f' ({ROAD_MU[r["road"]]:g})</td><td>{r.get("decel_ms2", 0):.2f} m/s²</td>'
                 f'<td>{r.get("delay_s", 0):.2f} s</td><td>{E(r.get("binding") or "—")}</td>'
                 f'<td>{"—" if r.get("binding_range_m") is None else f"{r["binding_range_m"]:.1f} m"}</td>'
                 f'<td>{"—" if r.get("max_kmh") is None else f"{r["max_kmh"]:.0f} km/h"}</td></tr>')
    h.append(f'</table><p class="muted">{E(ROAD_SOURCE)}</p>')
    h.append('<h2>Known and unknown scenarios</h2>')
    for a in res.areas:
        h.append(f'<p><b>Area {a["area"]}: {E(a["label"])}.</b> {E(a["text"])}</p>')
        if a['items']:
            h.append('<ul>' + ''.join(f'<li>{E(i["label"])}: {E(i["text"])}</li>' for i in a['items']) + '</ul>')
    h.append('<h2>Requirements</h2>')
    if res.trace:
        h.append('<table><tr><th>Requirement</th><th>Metric</th><th>Target</th><th>Cells</th><th>Matrix figure</th>'
                 '</tr>' + ''.join(
                     f'<tr><td>{E(t["requirement"])} {E(t["text"])}</td><td>{E(t["metric"])}</td>'
                     f'<td>{E(t["op"])} {E(t["target"])}</td><td>{E(", ".join(t["cells"][:6]))}</td>'
                     f'<td>{E(t["matrix_figure"]["value_m"]) + " m" if t["matrix_figure"] else "—"}</td></tr>'
                     for t in res.trace) + '</table>')
    else:
        h.append('<p class="muted">No requirement of the scorecard checks a figure of the matrix.</p>')
    h.append('<p>Proposed: ' + '; '.join(E(p_['text']) for p_ in res.proposals) + '.</p>')
    h.append('<h2>Findings</h2><ul>' + ''.join(f'<li><b>{E(f["severity"])}</b>: {E(f["message"])}</li>'
                                               for f in res.findings) + '</ul>')
    h.append('<h2>Studies and their assumptions</h2><ul>' + ''.join(
        f'<li><b>{E(k)}</b> ({E(v["module"])}): {E(v["assumptions"])}'
        + (f' <i>Error: {E(v["error"])}</i>' if v['error'] else '') + '</li>' for k, v in res.studies.items())
        + '</ul>')
    h.append(f'<p class="muted">{E(ASSUMPTIONS)}</p></body></html>')
    return ''.join(h)


def report_pdf(res: SotifResult, project=None) -> bytes:
    """The report as A4 PDF bytes, laid out from report_html. Needs a running QGuiApplication, as the app's other
    PDFs."""
    from .exporters import HAIR, MUTED, _PdfOut, _Sheet, _families, _require_app
    _require_app()
    from PyQt6.QtCore import QRectF, QSizeF
    from PyQt6.QtGui import QFont, QImage, QTextDocument
    from .requirements import _pdf_out
    sans, _ = _families()
    doc = QTextDocument()
    dev = QImage(8, 8, QImage.Format.Format_RGB32)
    dev.setDotsPerMeterX(3780)
    dev.setDotsPerMeterY(3780)
    doc.documentLayout().setPaintDevice(dev)
    f = QFont(sans)
    f.setPointSizeF(8.5)
    doc.setDefaultFont(f)
    doc.setHtml(report_html(res, project))
    W_, H_ = 794.0, 1123.0
    doc.setPageSize(QSizeF(W_, H_))
    fmt = doc.rootFrame().frameFormat()
    for k in ('setLeftMargin', 'setRightMargin', 'setTopMargin'):
        getattr(fmt, k)(50)
    fmt.setBottomMargin(64)
    doc.rootFrame().setFrameFormat(fmt)
    n = max(doc.pageCount(), 1)
    title = f'SOTIF · {getattr(project, "name", "")} · {res.variant}'
    xmp = ('<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>\n<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF '
           'xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"><rdf:Description rdf:about="" '
           'xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title><rdf:Alt><rdf:li xml:lang="x-default">'
           f'{html.escape(title)}</rdf:li></rdf:Alt></dc:title><dc:description><rdf:Alt><rdf:li xml:lang="x-default">'
           f'{html.escape(DISCLAIMER)} ODD speed limit: '
           f'{"—" if res.odd["speed_kmh"] is None else round(res.odd["speed_kmh"], 1)} km/h.</rdf:li></rdf:Alt>'
           '</dc:description></rdf:Description></rdf:RDF></x:xmpmeta>\n<?xpacket end="w"?>').encode('utf-8')
    out = _pdf_out(_PdfOut, title, xmp)
    p = out.painter
    scale = out.writer.width() / W_
    sh = _Sheet(out)
    for i in range(n):
        if i:
            out.new_page()
        p.save()
        p.scale(scale, scale)
        p.translate(0, -i * H_)
        doc.drawContents(p, QRectF(0, i * H_, W_, H_))
        p.restore()
        fy = 297.0 - 12.0
        sh.line(15.0, fy - 1.5, 195.0, fy - 1.5, HAIR, 0.2)
        sh.text(15.0, fy - 1.0, 140, 4, f'Vantage · {title} · not a certification', sh.font(6.5), MUTED, elide=True)
        sh.text(155.0, fy - 1.0, 40, 4, f'Page {i + 1} of {n}', sh.font(6.5), MUTED, align='r')
    return out.finish()


def files(res: SotifResult, project=None, pdf: bool | None = None) -> list:
    """[(name, bytes)]: sotif.json (with provenance when a project is given), sotif_matrix.csv, sotif_odd.csv,
    sotif_report.html and sotif_report.pdf (when a QGuiApplication runs, or pdf=True)."""
    doc = to_doc(res, project)
    if project is not None:
        from . import provenance as PV
        doc = PV.stamp(doc, project)
    out = [('sotif.json', (json.dumps(doc, indent=1, ensure_ascii=False) + '\n').encode('utf-8')),
           ('sotif_matrix.csv', to_csv(res).encode('utf-8')),
           ('sotif_odd.csv', odd_csv(res).encode('utf-8')),
           ('sotif_report.html', report_html(res, project).encode('utf-8'))]
    if pdf is None:
        try:
            from PyQt6.QtGui import QGuiApplication
            pdf = QGuiApplication.instance() is not None
        except ImportError:
            pdf = False
    if pdf:
        out.append(('sotif_report.pdf', report_pdf(res, project)))
    return out


def export(res: SotifResult, project, out_dir) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for name, data in files(res, project):
        path = os.path.join(out_dir, name)
        with open(path, 'wb') as fh:
            fh.write(data)
        paths.append(path)
    return paths


# =========================================================================== validation (for vantage/validation)
# WSDOT Design Manual M 22-01.23 (September 2024), Exhibit 1260-1 (p. 1260-2): design speed (mph) -> design stopping
# sight distance (ft), as printed; AASHTO's 2.5 s and 11.2 ft/s², rounded up to the next 5 ft
WSDOT_1260_1 = {25: 155, 30: 200, 35: 250, 40: 305, 45: 360, 50: 425, 55: 495, 60: 570, 65: 645, 70: 730, 75: 820,
                80: 910}


def _integrate_stop(v0, stop: Stop, dt=1e-5) -> float:
    """The stopping distance by stepping the kinematics (independent of the closed form): constant speed for the
    delay, then deceleration rising at the jerk to its value; midpoint rule."""
    s, v, t = v0 * stop.delay_s, v0, 0.0
    j = stop.jerk if math.isfinite(stop.jerk) else math.inf
    while v > 0:
        a0 = min(stop.decel, j * t) if math.isfinite(j) else stop.decel
        a1 = min(stop.decel, j * (t + dt)) if math.isfinite(j) else stop.decel
        am = 0.5 * (a0 + a1)
        if v - am * dt <= 0:
            s += v * v / (2 * am)
            break
        s += v * dt - 0.5 * am * dt * dt
        v -= am * dt
        t += dt
    return s


def validation_checks() -> list:
    """Spec-compatible dicts for vantage.validation (model 'sotif'): fn(ctx) returns a suite Outcome."""
    def _closed_form(ctx):
        from ..validation.suite import Outcome
        worst, rows = 0.0, {}
        for mid in ('r157', 'r157_fsm', 'aashto'):
            st = SotifSettings(stop_model=mid)
            for road in ROADS:
                stop = stop_of(st, road)
                for kph in (5.0, 10.0, 30.0, 50.0, 80.0, 130.0):
                    v = kph / 3.6
                    a = stopping_distance(v, stop)
                    b = _integrate_stop(v, stop, dt=2e-5 if ctx.quick else 5e-6)
                    worst = max(worst, abs(a - b))
                    rows[f'{mid}/{road}/{kph:g}'] = round(a, 4)
                    vb = max_safe_speed(a, stop)
                    worst = max(worst, abs(stopping_distance(vb, stop) - a))
        return Outcome(worst, {'stopping_distance_m': rows})

    def _wsdot(ctx):
        from ..validation.suite import Outcome
        st = SotifSettings(stop_model='aashto')
        stop = stop_of(st, 'dry')
        worst, rows = 0.0, {}
        for mph, ft in WSDOT_1260_1.items():
            d_ft = stopping_distance(mph * 0.44704, stop) / 0.3048
            design = 5.0 * math.ceil(d_ft / 5.0 - 1e-9)
            rows[f'{mph} mph'] = {'model_ft': round(d_ft, 2), 'rounded_ft': design, 'table_ft': ft}
            worst = max(worst, abs(design - ft))
        return Outcome(worst, {'rows': rows})

    return [
        {'id': 'sotif.stopping_closed_form', 'model': 'sotif',
         'title': 'Stopping distance with a jerk ramp against a time-stepped integration',
         'quantity': 'largest |closed form − integration| of sotif.stopping_distance over the R157 driver, R157 FSM '
                     'and AASHTO models, four road surfaces and 5–130 km/h, and of the inverse max_safe_speed '
                     'round trip, metres',
         'reference': 'the kinematics stepped in time (midpoint rule, 5 µs) in the check, independent of the closed '
                      'form', 'tolerance': 0.005, 'unit': 'm', 'refs': ('unece_r157',), 'fn': _closed_form,
         'cost': 'cheap'},
        {'id': 'sotif.wsdot_ssd', 'model': 'sotif',
         'title': 'Design stopping sight distance against WSDOT Exhibit 1260-1 (AASHTO)',
         'quantity': 'largest difference between the AASHTO stop model\'s distance rounded up to 5 ft and the '
                     'table\'s 12 design values (25–80 mph), feet',
         'reference': 'WSDOT Design Manual M 22-01.23 (September 2024), Exhibit 1260-1, as printed', 'tolerance': 0.0,
         'unit': 'ft', 'refs': ('wsdot_1260', 'aashto_greenbook'), 'fn': _wsdot, 'cost': 'cheap'},
    ]


REFERENCES = {
    'unece_r157': R157 + ': the competent and careful human driver model and the fuzzy safety model.',
    'wsdot_1260': 'Washington State DOT, Design Manual M 22-01.23, September 2024, Chapter 1260 Sight Distance, '
                  '§1260.03 and Exhibit 1260-1.',
    'aashto_greenbook': 'AASHTO, A Policy on Geometric Design of Highways and Streets (the Green Book): stopping sight '
                        'distance, 2.5 s brake reaction time and 11.2 ft/s² (3.4 m/s²) deceleration.',
    'wong2008': 'J. Y. Wong, Theory of Ground Vehicles, 4th ed., Wiley, 2008, Table 1.3.',
    'iso21448': 'ISO 21448:2022, Road vehicles — Safety of the intended functionality.',
}


def validation_references() -> dict:
    """{key: citation} for vantage.validation.references (the keys the checks above cite)."""
    return dict(REFERENCES)
