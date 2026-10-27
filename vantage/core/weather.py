"""Weather: how rain, fog, snow and night change what each sensor of the rig records, as documented typical models.

Conditions
----------
A condition is a set of figures (Condition): daylight or night with the vehicle's headlights (low or high beam), the
ambient illuminance on a vertical surface (lux: sky, moon or street lights), a fog or haze given by its meteorological
optical range (MOR, the WMO visibility: the distance at which a black object's contrast against the horizon falls to
5 %), a rain rate and a snowfall rate (mm/h of water). The presets (clear day, overcast, night on an unlit road and
under street lights, light and heavy rain, fog, snow) carry typical figures; every one but 'clear' can be edited and
saved with the project (analysis.weather). 'Clear' is the reference: no weather model at all, so every study gives
exactly what it gave before weather existed.

Cameras
-------
* Extinction (Koschmieder): an object of inherent contrast C0 against its background is seen with contrast
  C0·exp(−σd) at distance d; it is resolved while that stays above the detector's contrast threshold ε (typical 0.05,
  the threshold behind the MOR). σ at 550 nm: fog or haze 3.0 / MOR (= −ln 0.05 / MOR); rain 1.076·R^0.67 dB/km (ITU-R
  P.1814, Carbonneau's coefficients; drops are much larger than the wavelength, so it holds for light and the lidar
  alike); snow a·S^b dB/km with a = 5.42e-5·λ + 5.50, b = 1.38 (dry) or a = 1.02e-4·λ + 3.79, b = 0.72 (wet), λ in nm,
  S the snow rate (ITU-R P.1814, Table 3, the constants as it prints them). Fog, rain and snow add.
* Light: the object's luminance L = ρ·E/π (Lambertian, reflectance ρ) under the ambient illuminance plus the vehicle's
  two headlamps, E = Σ I(h, v)·exp(−σ d_lamp) / d_lamp², with I the lamp's luminous intensity toward the object. The
  lamp pattern is a typical one shaped after the UN R112 class B test points (cut-off 0.7° below the horizon on the
  oncoming side rising at 15° on the kerb side, a hot spot of 18 000 cd per lamp below it, a wide foreground spread;
  high beam adds a 60 000 cd spot): the test checks it meets R112's minimum and maximum intensities. A camera resolves
  the object while at least lit_share of its height reaches the camera's minimum luminance L_min.
* L_min from the module's figures: the electrons a pixel collects from luminance L are
  π·T·L / (4N²) · t · Φ · A · QE (lens f-number N and transmission T, exposure t ≤ 1/fps and ≤ the night limit, pixel
  area A from the module's pixel pitch, Φ ≈ 1e16 photons per lux·s·m² of white light, an effective quantum
  efficiency QE with the colour filter); L_min is where the signal-to-noise ratio √(S + read noise²) reaches snr_min.
  QE, read noise and snr_min are typical figures, not the sensor's datasheet. The module's stated dynamic range is
  checked against the ~120 dB a night scene with oncoming headlights spans (a figure commonly quoted for automotive
  HDR sensors).
* Lens drops (rain, snow): a heuristic share of the image under drops, ∝ √(rate) × the lens's dirt and spray exposure
  score (core.exposure's model, reused), times the share of the view the wipers do not clear for a camera behind the
  windscreen (core.wipers via core.materials' study). Drops do not shorten the range; they hide part of the image.

Lidar
-----
The lidar equation for an extended Lambertian target: received power ∝ ρ·exp(−2αR)/R², so a range R_c in clear air
becomes R with R·exp(αR) = R_c, i.e. R = W(α R_c)/α (W: Lambert's function). α at the lidar's wavelength: fog from the
MOR with Kim's wavelength exponent q(V) (σ = 3.0/V · (λ/550 nm)^−q), rain and snow as for cameras. Returns beyond the
derated range drop out; their intensity falls by exp(−2αR). False near returns from droplets and flakes are a
heuristic: a share 1 − exp(−α_bs · 6 m) of the beams, α_bs = α_fog + α_rain + 3·α_snow (flakes are large discrete
reflectors), at short range. Night does not shorten a lidar's range (its datasheet range is quoted in sunlight).

Radar
-----
Rain: ITU-R P.838-3's specific attenuation γ = k·R^α dB/km at the radar's frequency (the mean of its horizontal and
vertical coefficients: the polarisation is not known). Fog: ITU-R P.840's liquid-water coefficient K_l(f, T) times the
fog's liquid water content, which Kunkel (1984) relates to its extinction (σ = 144.7·LWC^0.88 km⁻¹). Wet snow is taken
as rain of the same water rate (an upper-end assumption); dry snow is lossless at 77 GHz. Range: the radar equation
with two-way loss, R⁴·exp(2aR) = R_c⁴ → R = W(a R_c / 2) / (a/2), a = γ / 4343 per metre. A wet fascia or radome adds
loss this model does not include; rain clutter is not modelled.

GNSS antennas and IMUs are not affected (modelled).
"""
from __future__ import annotations

import csv
import io
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field, replace

import numpy as np

SCHEMA = 'vantage.weather/1'
FRAME = 'vehicle (ISO 8855, rear axle): x forward, y left, z up'
MOR_LN = -math.log(0.05)             # 2.996: σ = MOR_LN / MOR (WMO's 5 % contrast threshold)
DB_PER_NEPER = 10.0 / math.log(10.0)  # 4.343 dB of power per neper
PHOTONS_PER_LUX_S = 1.0e16           # photons per m² per lux·second, white light 400–700 nm (typical)
HDR_NIGHT_DB = 120.0                 # scene dynamic range of night driving with oncoming headlights (typical figure)
HEIGHT_SAMPLES = 8                   # points up an object's height for the light test
DROP_K = 0.08                        # lens drop heuristic: share = DROP_K·√(rate mm/h)·exposure score
DROP_MAX = 0.6
FALSE_NEAR_M = 6.0                   # lidar false returns: the near zone they appear in
SNOW_CLUTTER = 3.0                   # flakes give more discrete false returns than their extinction suggests
LAMP_INSET_M = (0.15, 0.30)          # headlamps: behind the bumper's front, inboard of the body's side
DAY, NIGHT = 'day', 'night'
LIGHTS = (DAY, NIGHT)
BEAMS = ('low', 'high')
SNOW_TYPES = ('dry', 'wet')
TRAFFIC = ('right', 'left')

ASSUMPTIONS = (
    'Typical models, not measurements: Koschmieder contrast extinction with a 5 % threshold; rain and snow '
    'extinction after ITU-R P.1814; fog wavelength scaling after Kim et al.; radar rain attenuation after ITU-R '
    'P.838-3 '
    'and fog after ITU-R P.840 with Kunkel\'s liquid-water relation; a typical UN R112-shaped headlamp pattern; the '
    'camera\'s minimum luminance from a photon budget with a typical quantum efficiency, read noise and required '
    'signal-to-noise ratio; object contrasts and reflectances are typical. Lens drops and lidar false returns are '
    'heuristics. Not modelled: veiling light from the own headlamps in fog, oncoming glare, wet-road reflections, '
    'spray from other vehicles, a wet radar fascia, rain clutter, street lights as visible sources.')

# Typical inherent contrast against the background (day) and diffuse reflectance (night) per object class. A
# retroreflective sign sends far more back toward the vehicle than a diffuse surface: an effective reflectance of 10
# (≈ 3 cd/(lx·m²) of retroreflection at the camera's observation angle, typical).
OBJECT_LOOKS = {'car': (0.5, 0.3), 'pedestrian': (0.3, 0.2), 'cyclist': (0.3, 0.2), 'traffic sign': (0.6, 10.0)}
GENERIC_LOOK = (0.3, 0.2)

# accepted range of every number (values outside are clamped)
LIMITS = {'contrast_threshold': (0.005, 0.5), 'snr_min': (1.0, 50.0), 'qe': (0.01, 1.0), 'read_noise_e': (0.0, 100.0),
          'max_exposure_ms': (0.1, 100.0), 'transmission': (0.1, 1.0), 'lit_share': (0.05, 1.0),
          'lamp_height_m': (0.2, 2.0), 'low_beam_cd': (1000.0, 100000.0), 'high_beam_cd': (5000.0, 215000.0),
          'lidar_nm': (400.0, 2000.0), 'radar_ghz': (1.0, 300.0), 'temp_c': (-20.0, 40.0)}
COND_LIMITS = {'ambient_lux': (0.0, 150000.0), 'visibility_m': (20.0, 50000.0), 'rain_mm_h': (0.0, 150.0),
               'snow_mm_h': (0.0, 30.0)}
LOOK_LIMITS = {'contrast': (0.01, 1.0), 'reflectance': (0.01, 50.0)}


# =========================================================================== conditions
@dataclass(frozen=True)
class Condition:
    id: str
    name: str
    light: str = DAY                 # day | night (the vehicle's headlamps on)
    ambient_lux: float = 30000.0     # illuminance on a vertical surface from the sky, the moon or street lights
    beam: str = 'low'                # night: low | high
    visibility_m: float = 0.0        # fog or haze: meteorological optical range (MOR); 0 = none
    rain_mm_h: float = 0.0
    snow_mm_h: float = 0.0           # water equivalent
    snow: str = 'dry'                # dry | wet

    @property
    def clear(self) -> bool:
        return self.id == 'clear'

    @property
    def night(self) -> bool:
        return self.light == NIGHT

    def label(self) -> str:
        """The name with its figures: 'Fog · 150 m', 'Heavy rain · 25 mm/h', 'Night, unlit road · low beam, 0.1 lx'."""
        if self.clear:
            return self.name
        bits = []
        if self.visibility_m > 0:
            bits.append(f'{self.visibility_m:g} m')
        if self.rain_mm_h > 0:
            bits.append(f'{self.rain_mm_h:g} mm/h')
        if self.snow_mm_h > 0:
            bits.append(f'{self.snow_mm_h:g} mm/h {self.snow}')
        if self.night:
            bits.append(f'{self.beam} beam, {self.ambient_lux:g} lx')
        return self.name + (' · ' + ', '.join(bits) if bits else '')

    def to_doc(self) -> dict:
        d = asdict(self)
        d['label'] = self.label()
        return d


PRESETS = (
    Condition('clear', 'Clear', DAY, 30000.0),
    Condition('overcast', 'Overcast', DAY, 5000.0, visibility_m=10000.0),
    Condition('night', 'Night, unlit road', NIGHT, 0.1),
    Condition('night_lit', 'Night, street lights', NIGHT, 10.0),
    Condition('light_rain', 'Light rain', DAY, 3000.0, rain_mm_h=2.5),
    Condition('heavy_rain', 'Heavy rain', DAY, 1000.0, rain_mm_h=25.0),
    Condition('fog', 'Fog', DAY, 2000.0, visibility_m=150.0),
    Condition('snow', 'Snow', DAY, 2000.0, snow_mm_h=2.0),
)
PRESET = {c.id: c for c in PRESETS}
IDS = tuple(c.id for c in PRESETS)
EDITABLE = ('light', 'ambient_lux', 'beam', 'visibility_m', 'rain_mm_h', 'snow_mm_h', 'snow')
NOTES = {'clear': 'The reference: no weather model (the studies as they were before weather existed).',
         'overcast': 'Daylight under cloud with a light haze (10 km).',
         'night': 'An unlit road under a moonlit sky (0.1 lx): only the headlamps light what is ahead.',
         'night_lit': 'A street-lit road (about 10 lx on a vertical surface; EN 13201 lighting classes give 2–50 lx).',
         'light_rain': 'Light rain (2.5 mm/h is the upper end of light rain).',
         'heavy_rain': 'A downpour (25 mm/h).',
         'fog': 'Dense fog (a visibility below 200 m closes roads to fast traffic in many countries).',
         'snow': 'Moderate dry snowfall (2 mm/h of water).'}


def _num(v):
    if isinstance(v, bool):
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def clean_condition(base: Condition, d) -> Condition:
    """base with the edits in d (tolerant: junk dropped, numbers clamped; visibility 0 or 20 m – 50 km)."""
    if base.clear or not isinstance(d, dict):
        return base
    kw = {}
    if d.get('light') in LIGHTS:
        kw['light'] = d['light']
    if d.get('beam') in BEAMS:
        kw['beam'] = d['beam']
    if d.get('snow') in SNOW_TYPES:
        kw['snow'] = d['snow']
    for k, (lo, hi) in COND_LIMITS.items():
        v = _num(d.get(k)) if k in d else None
        if v is None:
            continue
        if k == 'visibility_m' and v <= 0:
            kw[k] = 0.0
        else:
            kw[k] = min(max(v, lo), hi)
    return replace(base, **kw)


@dataclass
class WeatherSettings:
    condition: str = 'clear'           # the condition the perception range and the sample dataset use
    params: dict = field(default_factory=dict)      # {condition id: {field: value}}: edits of the presets
    compare: list = field(default_factory=lambda: list(IDS))
    objects: dict = field(default_factory=dict)     # {class name: {'contrast': c, 'reflectance': r}}
    contrast_threshold: float = 0.05
    snr_min: float = 5.0
    qe: float = 0.35                   # effective quantum efficiency with the colour filter (typical)
    read_noise_e: float = 3.0          # electrons (typical for an automotive HDR sensor)
    max_exposure_ms: float = 20.0      # the longest exposure at night (motion blur), also ≤ 1/fps
    transmission: float = 0.85         # lens
    lit_share: float = 0.4             # share of an object's height that must be lit enough at night
    traffic: str = 'right'             # right-hand traffic: the low beam's cut-off rises on the right
    lamp_height_m: float = 0.65
    low_beam_cd: float = 18000.0       # per lamp, the hot spot
    high_beam_cd: float = 60000.0      # per lamp, the high beam's spot (added to the low beam)
    lidar_nm: float = 905.0
    radar_ghz: float = 77.0
    temp_c: float = 10.0               # fog water temperature (ITU-R P.840)

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d: dict | None) -> 'WeatherSettings':
        """Tolerant of junk: unknown keys ignored, numbers clamped to LIMITS, condition ids checked."""
        s = WeatherSettings()
        d = d if isinstance(d, dict) else {}
        if d.get('condition') in IDS:
            s.condition = d['condition']
        params = d.get('params')
        if isinstance(params, dict):
            out = {}
            for cid, p in params.items():
                if cid in PRESET and cid != 'clear' and isinstance(p, dict):
                    c = clean_condition(PRESET[cid], p)
                    diff = {k: getattr(c, k) for k in EDITABLE if getattr(c, k) != getattr(PRESET[cid], k)}
                    if diff:
                        out[cid] = diff
            s.params = out
        cmp_ = d.get('compare')
        if isinstance(cmp_, (list, tuple)):
            keep = {c for c in cmp_ if c in IDS}
            s.compare = [c for c in IDS if c in keep]
        objs = d.get('objects')
        if isinstance(objs, dict):
            out = {}
            for name, o in list(objs.items())[:64]:
                if not isinstance(o, dict):
                    continue
                e = {}
                for k, (lo, hi) in LOOK_LIMITS.items():
                    v = _num(o.get(k))
                    if v is not None:
                        e[k] = min(max(v, lo), hi)
                if e:
                    out[str(name)[:40]] = e
            s.objects = out
        if d.get('traffic') in TRAFFIC:
            s.traffic = d['traffic']
        for k, (lo, hi) in LIMITS.items():
            if k in d:
                v = _num(d[k])
                if v is not None:
                    setattr(s, k, min(max(v, lo), hi))
        return s

    def condition_for(self, cid: str) -> Condition:
        base = PRESET.get(cid, PRESET['clear'])
        return clean_condition(base, self.params.get(cid, {}))

    def conditions(self) -> list:
        return [self.condition_for(c) for c in IDS]

    def selected(self) -> Condition:
        return self.condition_for(self.condition)


def settings_of(project) -> WeatherSettings:
    return WeatherSettings.from_dict(getattr(getattr(project, 'analysis', None), 'weather', None))


def resolve_condition(project, cond=None, st: WeatherSettings | None = None) -> Condition:
    """A Condition from a Condition, a preset id (with the project's edits) or None (the project's selected one)."""
    st = st or settings_of(project)
    if isinstance(cond, Condition):
        return cond
    if isinstance(cond, str):
        return st.condition_for(cond)
    return st.selected()


# =========================================================================== extinction and attenuation
def kim_q(v_km: float) -> float:
    """Kim et al.'s wavelength exponent of fog and haze extinction for a visibility in km."""
    if v_km > 50.0:
        return 1.6
    if v_km > 6.0:
        return 1.3
    if v_km > 1.0:
        return 0.16 * v_km + 0.34
    if v_km > 0.5:
        return v_km - 0.5
    return 0.0


def rain_optical_db_km(rate):
    """ITU-R P.1814 (Carbonneau): optical rain attenuation, dB/km, for a rate in mm/h."""
    return 1.076 * np.power(np.maximum(rate, 0.0), 0.67)


def snow_optical_db_km(rate, nm=550.0, kind='dry'):
    """ITU-R P.1814: optical snow attenuation, dB/km, for a water rate in mm/h at wavelength nm (a·S^b with P.1814's
    printed constants; the 5.4958776 and 3.7855466 some later papers quote differ by 0.1 %)."""
    if kind == 'wet':
        a, b = 1.02e-4 * nm + 3.79, 0.72
    else:
        a, b = 5.42e-5 * nm + 5.50, 1.38
    return a * np.power(np.maximum(rate, 0.0), b)


def optical_extinction(c: Condition, nm: float = 550.0) -> float:
    """Extinction coefficient, 1/m, at wavelength nm: fog or haze (MOR, Kim's q away from 550 nm) + rain + snow."""
    if c.clear:
        return 0.0
    s = 0.0
    if c.visibility_m > 0:
        s += MOR_LN / c.visibility_m * (nm / 550.0) ** (-kim_q(c.visibility_m / 1000.0))
    s += float(rain_optical_db_km(c.rain_mm_h)) / DB_PER_NEPER / 1000.0
    s += float(snow_optical_db_km(c.snow_mm_h, nm, c.snow)) / DB_PER_NEPER / 1000.0
    return s


def visible_extinction(c: Condition) -> float:
    return optical_extinction(c, 550.0)


# ITU-R P.838-3 coefficients (Tables 1–4): (a_j, b_j, c_j), m, c
_P838 = {
    'kH': (((-5.33980, -0.10008, 1.13098), (-0.35351, 1.26970, 0.45400), (-0.23789, 0.86036, 0.15354),
            (-0.94158, 0.64552, 0.16817)), -0.18961, 0.71147),
    'kV': (((-3.80595, 0.56934, 0.81061), (-3.44965, -0.22911, 0.51059), (-0.39902, 0.73042, 0.11899),
            (0.50167, 1.07319, 0.27195)), -0.16398, 0.63297),
    'aH': (((-0.14318, 1.82442, -0.55187), (0.29591, 0.77564, 0.19822), (0.32177, 0.63773, 0.13164),
            (-5.37610, -0.96230, 1.47828), (16.1721, -3.29980, 3.43990)), 0.67849, -1.95537),
    'aV': (((-0.07771, 2.33840, -0.76284), (0.56727, 0.95545, 0.54039), (-0.20238, 1.14520, 0.26809),
            (-48.2991, 0.791669, 0.116226), (48.5833, 0.791459, 0.116479)), -0.053739, 0.83433),
}


def p838(f_ghz: float) -> dict:
    """ITU-R P.838-3 coefficients kH, αH, kV, αV at f GHz (1–1000 GHz)."""
    lf = math.log10(min(max(float(f_ghz), 1.0), 1000.0))

    def g(key):
        terms, m, c = _P838[key]
        return sum(a * math.exp(-((lf - b) / cc) ** 2) for a, b, cc in terms) + m * lf + c
    return {'kH': 10 ** g('kH'), 'aH': g('aH'), 'kV': 10 ** g('kV'), 'aV': g('aV')}


def rain_radar_db_km(rate, f_ghz: float = 77.0):
    """Specific rain attenuation, dB/km, on a horizontal path: ITU-R P.838-3 with the mean of the horizontal and
    vertical coefficients (eq. 4–5 with τ = 45°: the polarisation is not known)."""
    q = p838(f_ghz)
    k = (q['kH'] + q['kV']) / 2
    a = (q['kH'] * q['aH'] + q['kV'] * q['aV']) / (2 * k)
    return k * np.power(np.maximum(rate, 0.0), a)


def liquid_water_coefficient(f_ghz: float, temp_c: float = 10.0) -> float:
    """ITU-R P.840 specific attenuation coefficient of liquid water droplets, (dB/km)/(g/m³) (double-Debye model)."""
    th = 300.0 / (temp_c + 273.15)
    e0 = 77.66 + 103.3 * (th - 1)
    e1 = 0.0671 * e0
    e2 = 3.52
    fp = 20.20 - 146.0 * (th - 1) + 316.0 * (th - 1) ** 2
    fs = 39.8 * fp
    f = float(f_ghz)
    ep = (e0 - e1) / (1 + (f / fp) ** 2) + (e1 - e2) / (1 + (f / fs) ** 2) + e2
    epp = f * (e0 - e1) / (fp * (1 + (f / fp) ** 2)) + f * (e1 - e2) / (fs * (1 + (f / fs) ** 2))
    eta = (2 + ep) / epp
    return 0.819 * f / (epp * (1 + eta ** 2))


def fog_lwc(c: Condition) -> float:
    """Fog liquid water content, g/m³, from its extinction (Kunkel 1984: σ = 144.7·LWC^0.88 km⁻¹)."""
    if c.visibility_m <= 0:
        return 0.0
    sigma_km = MOR_LN / c.visibility_m * 1000.0
    return (sigma_km / 144.7) ** (1.0 / 0.88)


def radar_db_km(c: Condition, f_ghz: float = 77.0, temp_c: float = 10.0) -> float:
    """One-way specific attenuation, dB/km: rain (P.838-3) + fog (P.840 × Kunkel's LWC) + wet snow taken as rain."""
    if c.clear:
        return 0.0
    rate = c.rain_mm_h + (c.snow_mm_h if c.snow == 'wet' else 0.0)
    return float(rain_radar_db_km(rate, f_ghz)) + liquid_water_coefficient(f_ghz, temp_c) * fog_lwc(c)


def _lambertw(x):
    from scipy.special import lambertw
    return np.real(lambertw(np.asarray(x, float)))


def lidar_range(r_clear, alpha):
    """The range an extended target reaches in extinction alpha (1/m) for clear range r_clear: R·exp(αR) = R_c."""
    r = np.asarray(r_clear, float)
    if alpha <= 0:
        return r.copy() if r.ndim else float(r)
    out = _lambertw(alpha * r) / alpha
    return out if r.ndim else float(out)


def radar_range(r_clear, db_km):
    """The radar range for two-way loss at db_km (one way): R⁴·exp(2aR) = R_c⁴, a = db_km / 4343 per metre."""
    r = np.asarray(r_clear, float)
    a = float(db_km) / DB_PER_NEPER / 1000.0
    if a <= 0:
        return r.copy() if r.ndim else float(r)
    out = _lambertw(a * r / 2) / (a / 2)
    return out if r.ndim else float(out)


def false_return_share(c: Condition, nm: float = 905.0) -> float:
    """Heuristic share of lidar beams with a false return from droplets or flakes in the near zone."""
    if c.clear:
        return 0.0
    fog = MOR_LN / c.visibility_m * (nm / 550.0) ** (-kim_q(c.visibility_m / 1000.0)) if c.visibility_m > 0 else 0.0
    rain = float(rain_optical_db_km(c.rain_mm_h)) / DB_PER_NEPER / 1000.0
    snow = float(snow_optical_db_km(c.snow_mm_h, nm, c.snow)) / DB_PER_NEPER / 1000.0
    return 1.0 - math.exp(-(fog + rain + SNOW_CLUTTER * snow) * FALSE_NEAR_M)


def contrast_range(c0: float, eps: float, sigma: float) -> float:
    """The distance at which contrast c0 falls to eps in extinction sigma (inf without extinction, 0 when c0 ≤ eps)."""
    if c0 <= eps:
        return 0.0
    return math.inf if sigma <= 0 else math.log(c0 / eps) / sigma


# =========================================================================== headlamps
def headlamp_cd(h_deg, v_deg, beam='low', traffic='right', low_cd=18000.0, high_cd=60000.0):
    """Luminous intensity (cd) of one headlamp toward (h, v): h degrees to the left of straight ahead, v above the
    horizontal. A typical pattern shaped after UN R112's class B test points (see the module docstring)."""
    h = np.asarray(h_deg, float)
    v = np.asarray(v_deg, float)
    if traffic == 'left':
        h = -h
    # the cut-off: 0.7° down on the oncoming (left) side (R112's 75L at 0.57° down lies on its gradient), rising at
    # 15° on the kerb side up to 1° above the horizon
    cut = np.where(h >= 0, -0.7, np.minimum(-0.7 + math.tan(math.radians(15.0)) * (-h), 1.0))
    d = cut - v                                                  # degrees below the cut-off

    def below(d0, s):
        return np.where(d >= 0, np.exp(-0.5 * (np.maximum(d - d0, 0.0) / s) ** 2), np.exp(np.minimum(d, 0.0) / 0.12))
    hot = low_cd * np.exp(-0.5 * ((h + 1.5) / 4.0) ** 2) * below(0.3, 1.6)
    spread = 0.22 * low_cd * np.exp(-0.5 * (h / 22.0) ** 2) * below(0.5, 10.0)     # the foreground, to ~25° down
    stray = 200.0 * np.exp(-0.5 * (h / 30.0) ** 2) * np.exp(-0.5 * (np.maximum(np.abs(v) - 3.0, 0.0) / 5.0) ** 2)
    out = hot + spread + stray
    if beam == 'high':
        out = out + high_cd * np.exp(-0.5 * (h / 4.5) ** 2) * np.exp(-0.5 * ((v + 0.2) / 2.2) ** 2)
    return np.where(np.abs(h) < 90.0, out, 0.0)


def lamp_positions(vg, st: WeatherSettings) -> np.ndarray:
    """The two headlamps (2, 3) in the vehicle frame: behind the body's front, inboard of its sides, at
    lamp_height_m (a typical car's without a body)."""
    if vg is not None:
        lo, hi = vg.bbox
        x = float(hi[0]) - LAMP_INSET_M[0]
        y = max(0.5 * float(hi[1] - lo[1]) - LAMP_INSET_M[1], 0.2)
        yc = 0.5 * float(hi[1] + lo[1])
    else:
        x, y, yc = 3.2 - LAMP_INSET_M[0], 0.6, 0.0
    return np.array([[x, yc + y, st.lamp_height_m], [x, yc - y, st.lamp_height_m]])


def illuminance(P, c: Condition, st: WeatherSettings, lamps, normals=None):
    """Illuminance (lux) at vehicle-frame points P (..., 3): the ambient plus, at night, both headlamps (inverse square,
    through the extinction). With normals (..., 3) the lamps' light falls on that surface (cosine law); without, on
    a surface facing the lamp."""
    P = np.asarray(P, float)
    E = np.full(P.shape[:-1], float(c.ambient_lux))
    if not c.night:
        return E
    sig = visible_extinction(c)
    for L in np.asarray(lamps, float):
        d = P - L
        r = np.maximum(np.linalg.norm(d, axis=-1), 1e-3)
        h = np.degrees(np.arctan2(d[..., 1], d[..., 0]))
        v = np.degrees(np.arcsin(np.clip(d[..., 2] / r, -1, 1)))
        I = headlamp_cd(h, v, c.beam, st.traffic, st.low_beam_cd, st.high_beam_cd)
        cos = 1.0 if normals is None else np.maximum(-np.sum(np.asarray(normals, float) * d, axis=-1) / r, 0.0)
        E = E + I * cos / (r * r) * np.exp(-sig * r)
    return E


def beam_reach(c: Condition, st: WeatherSettings, lamps, lux=1.0, height=0.5, az_deg=0.0, far=400.0) -> float:
    """Distance (m, from the lamps' midpoint along azimuth az_deg) out to which the lamps alone give at least `lux`
    on a surface facing them at `height` (the usual 1 lx reach of a beam)."""
    lamps = np.asarray(lamps, float)
    m = lamps.mean(0)
    r = np.linspace(1.0, far, 2000)
    a = math.radians(az_deg)
    P = np.c_[m[0] + r * math.cos(a), m[1] + r * math.sin(a), np.full(len(r), height)]
    E = illuminance(P, replace(c, ambient_lux=0.0, light=NIGHT), st, lamps)
    ok = E >= lux
    if not ok.any():
        return 0.0
    i = len(r) - 1 - int(np.argmax(ok[::-1]))
    return float(r[i])


def lit_range(c: Condition, st: WeatherSettings, lamps, l_min, origin_xy, az_deg, height=1.7, reflectance=0.2,
              far=400.0) -> float:
    """How far from origin_xy along az_deg an object (a pedestrian by default) stays lit to a camera's minimum
    luminance l_min: the perception range's light test (lit_share of its height, the path's extinction) along one
    line. inf when the ambient light alone is enough, 0 when it is never lit enough."""
    sig = visible_extinction(c)
    if reflectance * c.ambient_lux / math.pi >= l_min and sig <= 0:
        return math.inf
    r = np.geomspace(0.5, far, 600)
    a = math.radians(az_deg)
    G = np.c_[origin_xy[0] + r * math.cos(a), origin_xy[1] + r * math.sin(a)]
    lum = object_luminance(G, 0.0, height, reflectance, c, st, lamps) * np.exp(-sig * r)
    ok = lum >= l_min
    if not ok.any():
        return 0.0
    if ok[-1]:
        return math.inf if not c.night or reflectance * c.ambient_lux / math.pi >= l_min else float(r[-1])
    return float(r[len(r) - 1 - int(np.argmax(ok[::-1]))])


# =========================================================================== cameras
def camera_floor(module, lens, st: WeatherSettings) -> dict:
    """The camera's minimum luminance (cd/m²) for snr_min from a photon budget of the module's pixel and the lens's
    f-number (typical QE and read noise): electrons per cd/m² and the exposure used."""
    N = float(getattr(lens, 'f_number', 0.0) or 0.0)
    n_given = N > 0
    N = N if n_given else 2.0
    fps = float(getattr(module, 'fps', 30.0) or 30.0)
    t = min(st.max_exposure_ms / 1000.0, 1.0 / max(fps, 1e-3))
    area = (float(getattr(module, 'pixel_um', 3.0) or 3.0) * 1e-6) ** 2
    k_e = math.pi * st.transmission / (4.0 * N * N) * t * PHOTONS_PER_LUX_S * area * st.qe
    snr, sr = st.snr_min, st.read_noise_e
    s_req = (snr * snr + math.sqrt(snr ** 4 + 4.0 * snr * snr * sr * sr)) / 2.0
    dr = float(getattr(module, 'dynamic_range_db', 0.0) or 0.0)
    return {'l_min': s_req / max(k_e, 1e-12), 'electrons_per_cd_m2': k_e, 'electrons_needed': s_req,
            'exposure_ms': t * 1000.0, 'f_number': N, 'f_number_given': n_given,
            'pixel_um': float(getattr(module, 'pixel_um', 3.0) or 3.0), 'dynamic_range_db': dr}


def object_look(name: str, st: WeatherSettings) -> tuple:
    """(inherent contrast, reflectance) of an object class: the project's edit, else the typical figure."""
    c, r = OBJECT_LOOKS.get(str(name).strip().lower(), GENERIC_LOOK)
    o = st.objects.get(name) or {}
    return float(o.get('contrast', c)), float(o.get('reflectance', r))


def object_luminance(G, base, height, reflectance, c: Condition, st: WeatherSettings, lamps):
    """Luminance (cd/m²) of an object standing at ground points G (N, 2) that at least lit_share of its height
    reaches: ρ·E/π at HEIGHT_SAMPLES points up the object, the k-th brightest."""
    G = np.asarray(G, float).reshape(-1, 2)
    if not c.night:
        return np.full(len(G), reflectance * c.ambient_lux / math.pi)
    z = base + height * (np.arange(HEIGHT_SAMPLES) + 0.5) / HEIGHT_SAMPLES
    P = np.concatenate([np.broadcast_to(G[:, None, :], (len(G), HEIGHT_SAMPLES, 2)),
                        np.broadcast_to(z[None, :, None], (len(G), HEIGHT_SAMPLES, 1))], axis=-1)
    E = illuminance(P, c, st, lamps)
    k = max(1, int(math.ceil(st.lit_share * HEIGHT_SAMPLES - 1e-9)))
    Ek = np.sort(E, axis=1)[:, HEIGHT_SAMPLES - k]
    return reflectance * Ek / math.pi


def camera_margin(c: Condition, st: WeatherSettings, d_cam, contrast, lum, l_min):
    """ln of how far the object stays above the camera's thresholds (≥ 0: resolved): the smaller of the contrast
    margin ln(C0/ε) − σd and the light margin ln(L·exp(−σd)/L_min)."""
    sig = visible_extinction(c)
    d = np.asarray(d_cam, float)
    mc = math.log(max(contrast, 1e-9) / st.contrast_threshold) - sig * d
    with np.errstate(divide='ignore'):
        ml = np.log(np.maximum(np.asarray(lum, float), 0.0) / max(l_min, 1e-12)) - sig * d
    return np.minimum(mc, ml)


def lens_drop_share(score: float, c: Condition, wiped: float = 0.0) -> float:
    """Heuristic share of a lens's image under drops (rain) or flakes (snow): DROP_K·√(rate)·exposure score, minus the
    wiped share."""
    rate = c.rain_mm_h + 1.5 * c.snow_mm_h
    if c.clear or rate <= 0:
        return 0.0
    return float(min(DROP_MAX, DROP_K * math.sqrt(rate) * max(score, 0.0)) * (1.0 - min(max(wiped, 0.0), 1.0)))


def wiped_shares(project, vg=None) -> dict:
    """{camera uid: share of its view through glass the wipers clear} for the cameras behind the windscreen, from
    core.materials' cached study (core.wipers); {} when there is none (no body, no windscreen, wipers off)."""
    if getattr(project.vehicle, 'mesh', None) is None:
        return {}
    try:
        from . import materials as M
        res = M.cached_run(project, vg)
    except Exception:
        return {}
    wp = getattr(res, 'wipers', None)
    out = {}
    for c in (getattr(wp, 'cameras', None) or []):
        v = _num(c.get('share_wiped'))
        if c.get('uid') and v is not None:
            out[c['uid']] = min(max(v, 0.0), 1.0)
    return out


# =========================================================================== the dataset's weather
@dataclass
class SceneWeather:
    """What the sample dataset's simulators need of a condition (core.drivescene)."""
    condition: Condition
    settings: WeatherSettings
    sigma_vis: float                   # 1/m at 550 nm
    sigma_lidar: float                 # 1/m at the lidar's wavelength
    radar_db_km: float
    false_share: float
    lamps: np.ndarray                  # (2, 3) vehicle frame
    drops: dict = field(default_factory=dict)       # camera uid -> lens drop share
    tags: dict = field(default_factory=dict)        # camera uid -> tag

    def lidar_rmax(self, r_clear):
        return lidar_range(r_clear, self.sigma_lidar)

    def camera_floor(self, module, lens) -> dict:
        return camera_floor(module, lens, self.settings)


def scene_weather(project, cond=None, engine=None) -> SceneWeather | None:
    """The dataset's weather for a condition (a Condition, an id, or None: the project's selected one); None for
    clear (the simulators then run exactly as without weather)."""
    st = settings_of(project)
    c = resolve_condition(project, cond, st)
    if c.clear:
        return None
    vg = engine.vehicle_geom(project) if engine is not None else None
    drops = {}
    if c.rain_mm_h > 0 or c.snow_mm_h > 0:
        scores = _dirt_scores(project, vg)
        wiped = wiped_shares(project, vg)
        for cam in project.variant.cameras:
            if cam.uid in scores:
                drops[cam.uid] = lens_drop_share(scores[cam.uid], c, wiped.get(cam.uid, 0.0))
    return SceneWeather(c, st, visible_extinction(c), optical_extinction(c, st.lidar_nm),
                        radar_db_km(c, st.radar_ghz, st.temp_c), false_return_share(c, st.lidar_nm),
                        lamp_positions(vg, st), drops, {cam.uid: cam.tag for cam in project.variant.cameras})


def _dirt_scores(project, vg) -> dict:
    """{camera uid: dirt and spray exposure score} for the visible cameras (core.exposure's heuristic)."""
    from . import exposure as EXP
    from .analysis import camera_frames
    mesh = vg.mesh if vg is not None else project.vehicle.mesh
    wheels, _ = EXP.wheel_centres(project, mesh)
    body = EXP._body_info(project.vehicle.mesh if project.vehicle.mesh is not None else mesh)
    rec = set(EXP.ExposureSettings.from_dict(getattr(project.analysis, 'exposure', None)).recessed)
    out = {}
    for cam in project.variant.cameras:
        if not cam.visible or not EXP._finite_pose(cam):
            continue
        _, W2O = camera_frames(cam)
        axis = (W2O.T @ np.array([0.0, 0.0, 1.0]))[None, :]
        d = EXP._dirt_entry('camera', cam.pos()[None, :], [axis, EXP._mount_normal(mesh, cam.pos())[None, :]], axis,
                            wheels, body, cam.uid in rec)
        out[cam.uid] = float(d['score'])
    return out


# =========================================================================== the study
@dataclass
class WeatherResult:
    settings: WeatherSettings
    conditions: list                   # [Condition] compared (the selected one included)
    selected: str
    perception: object                 # core.perception.PerceptionResult of the selected condition, with the comparison
    sensors: dict                      # {condition id: [row]}
    findings: list                     # [{'severity', 'condition', 'message', 'tags'}], worst first
    metrics: dict
    variant: str = ''
    variant_uid: str = ''
    lamps: object = None               # (2, 3) the headlamps in the vehicle frame

    def condition(self, cid) -> Condition | None:
        return next((c for c in self.conditions if c.id == cid), None)

    def compare_row(self, cid) -> dict | None:
        w = getattr(self.perception, 'weather', None) or {}
        return next((r for r in w.get('compare', []) if r['id'] == cid), None)


def _project_for(project, variant):
    if variant is None or variant is project.variant:
        return project
    import copy
    q = copy.copy(project)
    q.variants = list(project.variants) + ([] if any(v is variant for v in project.variants) else [variant])
    q.active_uid = variant.uid
    return q


def _pct(new, old):
    return None if not old or not math.isfinite(old) or not math.isfinite(new) else 100.0 * (new - old) / old


def sensor_rows(project, engine, c: Condition, st: WeatherSettings, scores=None, lamps=None, wiped=None) -> list:
    """Per visible sensor, how this condition changes it (see the module docstring)."""
    v = project.variant
    vg = engine.vehicle_geom(project) if engine is not None else None
    lamps = lamp_positions(vg, st) if lamps is None else lamps
    sig = visible_extinction(c)
    rows = []
    if scores is None:
        scores = _dirt_scores(project, vg)
    if wiped is None:
        wiped = wiped_shares(project, vg)
    for cam in v.cameras:
        if not cam.visible:
            continue
        m, ln = project.module_for(cam), project.lens_for(cam)
        fl = camera_floor(m, ln, st)
        reach = {}
        if c.night:
            # how far out along the camera's heading a pedestrian (ρ 0.2) is lit to this camera's L_min
            reach = {'lit_range_m': lit_range(c, st, lamps, fl['l_min'], cam.pos()[:2], float(cam.yaw)),
                     'lux_needed': fl['l_min'] * math.pi / 0.2}
        wp = wiped.get(cam.uid)
        drops = lens_drop_share(scores.get(cam.uid, 0.0), c, wp or 0.0)
        hdr_short = c.night and 0 < fl['dynamic_range_db'] < HDR_NIGHT_DB
        rows.append({'uid': cam.uid, 'tag': cam.tag, 'kind': 'camera', 'slot': int(cam.slot),
                     'model': f'{m.name} · {ln.name}', 'extinction_per_km': sig * 1000.0,
                     'contrast_range_m': contrast_range(0.3, st.contrast_threshold, sig),
                     'min_luminance_cd_m2': fl['l_min'], 'exposure_ms': fl['exposure_ms'],
                     'f_number': fl['f_number'], 'f_number_given': fl['f_number_given'],
                     'dynamic_range_db': fl['dynamic_range_db'], 'hdr_short': bool(hdr_short),
                     'dirt_score': scores.get(cam.uid, 0.0), 'wiped_share': wp, 'lens_drops': drops, **reach})
    alpha_l = optical_extinction(c, st.lidar_nm)
    db = radar_db_km(c, st.radar_ghz, st.temp_c)
    fr = false_return_share(c, st.lidar_nm)
    for s in v.sensors:
        if not s.visible:
            continue
        m = project.sensor_model_for(s)
        row = {'uid': s.uid, 'tag': s.tag, 'kind': m.kind, 'model': f'{m.vendor} {m.name}'.strip()}
        if m.is_lidar:
            r = lidar_range(float(m.range_m), alpha_l)
            rmax = lidar_range(float(m.range_max_m), alpha_l) if m.range_max_m > 0 else None
            row.update({'wavelength_nm': st.lidar_nm, 'extinction_per_km': alpha_l * 1000.0,
                        'clear_range_m': float(m.range_m), 'range_m': r, 'change_pct': _pct(r, float(m.range_m)),
                        'clear_range_max_m': float(m.range_max_m) if m.range_max_m > 0 else None,
                        'range_max_m': rmax, 'two_way_loss_db_50m': 2 * alpha_l * 50.0 * DB_PER_NEPER,
                        'false_returns': fr})
        elif m.is_radar:
            zones = [{'clear_m': float(z[0]), 'range_m': radar_range(float(z[0]), db), 'az_half_deg': float(z[1])}
                     for z in m.zones]
            far0 = max((z['clear_m'] for z in zones), default=0.0)
            far = max((z['range_m'] for z in zones), default=0.0)
            row.update({'frequency_ghz': st.radar_ghz, 'attenuation_db_km': db, 'zones': zones,
                        'clear_range_m': far0, 'range_m': far, 'change_pct': _pct(far, far0),
                        'two_way_loss_db': 2 * db * far0 / 1000.0})
        else:
            row['note'] = 'not affected (modelled)'
        rows.append(row)
    return rows


def _finding(out, sev, cid, message, tags=()):
    out.append({'severity': sev, 'condition': cid, 'message': message, 'tags': list(tags)})


def _findings(res: WeatherResult) -> list:
    out = []
    pr = res.perception
    clear = res.compare_row('clear')
    names = [k.name for k in pr.classes] if pr is not None else []
    for c in res.conditions:
        if c.clear:
            continue
        row = res.compare_row(c.id)
        if row is not None and clear is not None:
            lost = row['passed'] < clear['passed']
            worst = None
            for n in names:
                a, b = clear['per_class'][n]['ahead_m'], row['per_class'][n]['ahead_m']
                if a > 0 and (worst is None or b / a < worst[2]):
                    worst = (n, b, b / a, a)
            if worst is not None and worst[2] < 0.9:
                msg = (f'{c.label()}: {worst[0].lower()}s are resolved to {worst[1]:.0f} m ahead '
                       f'(clear {worst[3]:.0f} m); {row["passed"]} of {row["checks"]} sector checks pass '
                       f'(clear {clear["passed"]}).')
                _finding(out, 'serious' if lost else 'warning', c.id, msg)
            elif lost:
                _finding(out, 'serious', c.id, f'{c.label()}: {row["passed"]} of {row["checks"]} sector checks pass '
                                               f'(clear {clear["passed"]}).')
        for r in res.sensors.get(c.id, []):
            if r['kind'] == 'camera':
                if r['lens_drops'] >= 0.10:
                    _finding(out, 'warning', c.id, f'{c.label()}: about {r["lens_drops"] * 100:.0f} % of '
                                                   f'{r["tag"]}\'s image under drops (heuristic, exposure score '
                                                   f'{r["dirt_score"]:.2f}): a washer jet, an air knife or a '
                                                   'hydrophobic coating.', [r['tag']])
                if r['hdr_short']:
                    _finding(out, 'warning', c.id, f'{c.label()}: {r["tag"]} states {r["dynamic_range_db"]:g} dB of '
                                                   f'dynamic range; night scenes with oncoming headlights span about '
                                                   f'{HDR_NIGHT_DB:g} dB (typical): expect clipped lights or lost '
                                                   'shadows.', [r['tag']])
            elif r['kind'] in ('lidar', 'radar') and r.get('change_pct') is not None and r['change_pct'] <= -10.0:
                extra = (f'; false near returns on about {r["false_returns"] * 100:.0f} % of beams (heuristic)'
                         if r['kind'] == 'lidar' and r.get('false_returns', 0) >= 0.01 else '')
                _finding(out, 'warning', c.id, f'{c.label()}: {r["tag"]} reaches {r["range_m"]:.0f} m '
                                               f'(clear {r["clear_range_m"]:.0f} m, {r["change_pct"]:.0f} %){extra}.',
                         [r['tag']])
    order = {'critical': 0, 'serious': 1, 'warning': 2, 'good': 3}
    out.sort(key=lambda f: order.get(f['severity'], 9))
    return out


def run(project, engine=None, settings: WeatherSettings | None = None, variant=None, progress=None,
        cancel=None) -> WeatherResult | None:
    """Every compared condition: the perception range (geometry once, each condition's thresholds on top) and how each
    sensor degrades. Returns None when cancelled."""
    from . import perception as PR
    t0 = time.perf_counter()
    p = _project_for(project, variant)
    st = settings or settings_of(p)
    if engine is None:
        from .analysis import Engine
        engine = Engine()
    vg = engine.vehicle_geom(p)
    sel = st.selected()
    conds = [st.condition_for(cid) for cid in IDS if cid in st.compare or cid == sel.id or cid == 'clear']

    def prog(f):
        if progress is not None:
            progress(min(max(float(f), 0.0), 1.0))
    pr = PR.analyse(p, engine, None, None, progress=lambda f, *_: prog(0.8 * f), cancel=cancel, weather=sel,
                    compare=conds, weather_settings=st)
    if pr is None:
        return None
    lamps = lamp_positions(vg, st)
    scores = _dirt_scores(p, vg)
    wiped = wiped_shares(p, vg)
    sensors = {}
    for i, c in enumerate(conds):
        if cancel is not None and cancel():
            return None
        sensors[c.id] = sensor_rows(p, engine, c, st, scores, lamps, wiped)
        prog(0.8 + 0.2 * (i + 1) / len(conds))
    res = WeatherResult(st, conds, sel.id, pr, sensors, [], {}, p.variant.name, p.variant.uid, lamps)
    res.findings = _findings(res)
    res.metrics = {'seconds': time.perf_counter() - t0, 'conditions': len(conds),
                   'cameras': len(pr.cams), 'sensors': sum(1 for s in p.variant.sensors if s.visible)}
    return res


# =========================================================================== output
def _r(v, d=2):
    if v is None:
        return None
    v = float(v)
    return None if not math.isfinite(v) else round(v, d)


def summary(res: WeatherResult) -> dict:
    """Compact figures (the assistant, the batch log): per condition the range ahead per object and the sector checks
    passed, how far each lidar and radar reaches, the findings."""
    out = []
    for c in res.conditions:
        row = res.compare_row(c.id) or {}
        sens = [{'tag': r['tag'], 'kind': r['kind'], 'range_m': _r(r['range_m'], 1),
                 'clear_range_m': _r(r['clear_range_m'], 1)} for r in res.sensors.get(c.id, [])
                if r['kind'] in ('lidar', 'radar')]
        out.append({'condition': c.id, 'label': c.label(),
                    'ahead_m': {n: _r(v['ahead_m'], 1) for n, v in row.get('per_class', {}).items()},
                    'sector_checks_passed': row.get('passed'), 'checks': row.get('checks'), 'lidar_radar': sens})
    return {'selected': res.selected, 'conditions': out,
            'findings': [{'severity': f['severity'], 'message': f['message']} for f in res.findings],
            'assumptions': ASSUMPTIONS}


def _sensor_doc(r):
    skip = {'uid', 'slot'}
    out = {}
    for k, v in r.items():
        if k in skip:
            continue
        if isinstance(v, float):
            out[k] = _r(v, 4)
        elif isinstance(v, list):
            out[k] = [{kk: (_r(vv, 3) if isinstance(vv, float) else vv) for kk, vv in z.items()} for z in v]
        else:
            out[k] = v
    return out


def to_doc(res: WeatherResult, project=None) -> dict:
    """weather.json: the conditions, the perception range per condition (range ahead and per sector per object, the
    sector checks), each sensor's degradation per condition, the findings, the models and their limits."""
    pr = res.perception
    st = res.settings
    return {
        'schema': SCHEMA, 'project': getattr(project, 'name', ''), 'variant': res.variant,
        'frame': FRAME, 'units': 'metres, degrees, lux, cd/m² (luminance), cd (intensity), dB, mm/h',
        'selected': res.selected,
        'settings': st.to_dict(),
        'assumptions': ASSUMPTIONS,
        'conditions': [dict(c.to_doc(), note=NOTES.get(c.id, '')) for c in res.conditions],
        'objects': {k.name: dict(zip(('contrast', 'reflectance'), object_look(k.name, st))) for k in pr.classes},
        'perception': [
            {'condition': row['id'], 'passed': row['passed'], 'checks': row['checks'],
             'per_class': {n: {'ahead_m': _r(v['ahead_m']), 'median_m': _r(v['median_m']),
                               'sectors': {s: _r(m) for s, m in v['sectors'].items()},
                               'passed': v['passed']} for n, v in row['per_class'].items()},
             'per_camera_near_heading_m': {t: {n: _r(m) for n, m in d.items()} for t, d in row['per_camera'].items()}}
            for row in (pr.weather or {}).get('compare', [])],
        'sensors': {cid: [_sensor_doc(r) for r in rows] for cid, rows in res.sensors.items()},
        'findings': res.findings,
        'headlamps': {'positions_m': None if res.lamps is None else [[_r(q, 3) for q in L] for L in res.lamps],
                      'traffic': st.traffic, 'low_beam_cd_per_lamp': st.low_beam_cd,
                      'high_beam_cd_per_lamp': st.high_beam_cd,
                      'reach_1lx_m': None if res.lamps is None else
                      {b: _r(beam_reach(replace(PRESET['night'], beam=b), st, res.lamps, 1.0, 0.5), 1) for b in BEAMS}},
    }


CSV_CONDITIONS = ('condition', 'label', 'object', 'ahead_m', 'median_m', 'sector_checks_passed', 'checks')
CSV_SENSORS = ('condition', 'tag', 'kind', 'model', 'range_m', 'clear_range_m', 'change_pct', 'extinction_per_km',
               'attenuation_db_km', 'false_returns_pct', 'min_luminance_cd_m2', 'lit_range_m', 'lens_drops_pct')


def to_csv(res: WeatherResult) -> tuple[str, str]:
    """(conditions CSV: one row per condition and object; sensors CSV: one row per condition and sensor)."""
    a = io.StringIO()
    w = csv.writer(a, lineterminator='\n')
    w.writerow(CSV_CONDITIONS)
    for c in res.conditions:
        row = res.compare_row(c.id)
        if row is None:
            continue
        for n, v in row['per_class'].items():
            w.writerow([c.id, c.label(), n, _r(v['ahead_m'], 1), _r(v['median_m'], 1), row['passed'], row['checks']])
    b = io.StringIO()
    w = csv.writer(b, lineterminator='\n')
    w.writerow(CSV_SENSORS)
    blank = lambda v: '' if v is None else v          # noqa: E731
    for c in res.conditions:
        for r in res.sensors.get(c.id, []):
            pct = lambda k: _r(r[k] * 100, 1) if r.get(k) is not None else None          # noqa: E731
            w.writerow([c.id, r['tag'], r['kind'], r.get('model', ''), blank(_r(r.get('range_m'), 1)),
                        blank(_r(r.get('clear_range_m'), 1)), blank(_r(r.get('change_pct'), 1)),
                        blank(_r(r.get('extinction_per_km'), 4)), blank(_r(r.get('attenuation_db_km'), 3)),
                        blank(pct('false_returns')), blank(_r(r.get('min_luminance_cd_m2'), 4)),
                        blank(_r(r.get('lit_range_m'), 1)), blank(pct('lens_drops'))])
    return a.getvalue(), b.getvalue()


COND_COLORS = {'clear': '#9b9a95', 'overcast': '#3987e5', 'night': '#9085e9', 'night_lit': '#2fb5c7',
               'light_rain': '#5bb0f0', 'heavy_rain': '#1f5fbf', 'fog': '#fab219', 'snow': '#d55181'}


def png(res: WeatherResult, size=900, k_sel=0) -> bytes:
    """The envelope of one object class (the first by default) per condition around the vehicle, square-root range
    scale, with a legend of the range ahead per condition."""
    import cv2
    from .perception import PNG_RINGS, _bgr
    pr = res.perception
    head = 56
    img = np.full((size + head, size, 3), (30, 29, 28), np.uint8)
    cx, cy = size / 2, head + size / 2
    rmax = float(pr.ranges[-1])
    scale = (size / 2 - 34) / math.sqrt(rmax)
    k_sel = min(max(int(k_sel or 0), 0), len(pr.classes) - 1)
    font, aa = cv2.FONT_HERSHEY_SIMPLEX, cv2.LINE_AA

    def at(r, a):
        s, t = math.sqrt(max(r, 0.0)) * scale, math.radians(a)
        return cx - math.sin(t) * s, cy - math.cos(t) * s

    def poly(pts):
        return np.round(np.array(pts) * 16).astype(np.int32)

    def text(t, x, y, col, s=0.45):
        cv2.putText(img, t.encode('ascii', 'replace').decode(), (int(x), int(y)), font, s, col, 1, aa)
    for ring in PNG_RINGS:
        if ring <= rmax * 1.001:
            rr = int(round(math.sqrt(ring) * scale * 16))
            cv2.circle(img, (int(cx * 16), int(cy * 16)), rr, (62, 61, 58), 1, aa, 4)
            text(f'{ring} m', cx + 4, cy - rr / 16 + 14, (140, 140, 135), 0.4)
    cls = pr.classes[k_sel]
    rows = (pr.weather or {}).get('compare', [])
    for row in rows:
        far = row.get('far')
        if far is None:
            continue
        col = _bgr(COND_COLORS.get(row['id'], '#cccccc'))
        cv2.polylines(img, [poly([at(r, a) for r, a in zip(far[k_sel], pr.az)])], True, col,
                      3 if row['id'] == res.selected else 2, aa, 4)
    if pr.body is not None:
        (x0, y0), (x1, y1) = pr.body
        pts = []
        corners = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
        for (ax, ay), (bx, by) in zip(corners, corners[1:] + corners[:1]):
            for t in np.linspace(0, 1, 12, endpoint=False):
                dx, dy = ax + (bx - ax) * t - pr.centre[0], ay + (by - ay) * t - pr.centre[1]
                pts.append(at(math.hypot(dx, dy), math.degrees(math.atan2(dy, dx))))
        cv2.fillPoly(img, [poly(pts)], (72, 70, 66), aa, 4)
    text(f'{cls.name} by condition: farthest range resolved (sqrt range scale)', 16, 24, (236, 235, 230), 0.6)
    x, y = 16, 46
    for row in rows:
        c = res.condition(row['id'])
        label = f'{c.name} {row["per_class"][cls.name]["ahead_m"]:.0f} m'
        w = cv2.getTextSize(label, font, 0.42, 1)[0][0] + 28
        if x + w > size - 8:
            x, y = 16, y + 16
        cv2.rectangle(img, (x, y - 9), (x + 10, y + 1), _bgr(COND_COLORS.get(row['id'], '#cccccc')), -1)
        text(label, x + 15, y, (220, 219, 214), 0.42)
        x += w
    ok, buf = cv2.imencode('.png', img)
    return buf.tobytes()


def files(res: WeatherResult, project=None) -> list:
    """[(name, bytes)]: weather.json, weather_conditions.csv, weather_sensors.csv, weather.png."""
    a, b = to_csv(res)
    return [('weather.json', (json.dumps(to_doc(res, project), indent=1, default=str) + '\n').encode('utf-8')),
            ('weather_conditions.csv', a.encode('utf-8')), ('weather_sensors.csv', b.encode('utf-8')),
            ('weather.png', png(res))]


def export(res: WeatherResult, project, out_dir) -> list[str]:
    """Write files() into out_dir; returns their paths."""
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for name, data in files(res, project):
        path = os.path.join(out_dir, name)
        with open(path, 'wb') as fh:
            fh.write(data)
        paths.append(path)
    return paths
