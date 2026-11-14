"""Thermal intrinsic calibration curves: how a camera's focal length, principal point and distortion move with its
temperature, fitted per unit from lens calibrations taken at several temperatures, and the intrinsics compensated for
a capture's measured temperature.

Input. Unit lens calibrations (core.calib records: Kannala-Brandt fx fy cx cy k1..k4, OpenCV pixel coordinates at the
record's resolution) each with the camera's temperature while its images were taken ('temp_c'; calibration files
carry it as temp_c / temperature_c / temperature / sensor_temp_c, see calib.TEMP_KEYS). They are kept per camera in
project.analysis.thermal_intrinsics['sweeps'] {camera uid: [record, ...]}; records of another resolution with the
sensor's aspect are rescaled to the module's full resolution first (calib.unit_lens's convention: pixel centres at
integers, scaled about the pixel corner).

Fit. Each parameter p (fx, fy, cx, cy, k1..k4) against ΔT = T − T_ref (T_ref = reference_c, 20 °C by default, the
temperature the glass data below refer to) as a polynomial of degree 1 (linear) or 2 (quadratic) by ordinary least
squares (equal weights: a calibration's own covariance is not in its record; the residual scale is estimated from the
scatter about the curve, so it includes every calibration's error), with the coefficients' covariance s²(XᵀX)⁻¹.
The degree is chosen per parameter by leave-one-out cross-validation: the prediction error of each record from the
curve fitted to the others (PRESS, exact in closed form for least squares: e_i / (1 − h_ii), h the hat matrix'
diagonal). A quadratic is kept only when its leave-one-out RMS is at least QUAD_GAIN (10 %) below the linear's: with 4
to 8 temperatures the cross-validated error is itself uncertain by tens of per cent, so a smaller gain is noise, and
compensation needs the curve that predicts an unseen temperature best, not the one that fits the seen ones closest (an
F-test answers whether the curvature is non-zero, not whether it helps prediction). Leave-one-out alone is swayed by
one odd calibration (leaving it out of a straight line looks like curvature), so the quadratic's curvature must also
stand CURV_T (3) standard errors from zero: on simulated linear sweeps with calibration-sized scatter this keeps a
false quadratic to a few per cent of units (tests/test_thermal_intrinsics.py measures it). Leave-one-out needs at least
degree + 2 records; with 2 records the line through them is used without an uncertainty, with 1 nothing is fitted.
Predictions come with a confidence interval (of the curve) and a prediction interval (of one new calibration at that
temperature), Student t with n − degree − 1 degrees of freedom. Beyond the temperatures measured the curve is an
extrapolation: the result says so, and the issue check judges the operating range only where a curve exists.

Use. compensate(record, T) gives the record at temperature T from a record that carries a fitted curve (its 'thermal'
block: the values are those at reference_c; p(T) = p + a1·ΔT + a2·ΔT²), lens_at(project, cam, T) the lens model;
the capture's temperature comes from its metadata (temperature_of_metadata: the camera's sensor temperature under the
common names) or a ROS 2 bag's sensor_msgs/Temperature topic (temperatures_from_bag), or the user types it.

Exports: thermal_curves.json (vantage.thermal_intrinsics/1), one OpenCV YAML per camera (calib.to_opencv_yaml with the
thermal fields it documents: thermal_reference_c, thermal_model, thermal_params, thermal_coeffs, thermal_sigma,
thermal_range_c) and kalibr/camchain_thermal.yaml (Kalibr's pinhole-equi cameras at the reference temperature plus a
'thermal' block per camera, which Kalibr's reader ignores), and thermal_curves.csv.

SYNTHETIC SWEEPS, NOT MEASUREMENTS. No camera here has been calibrated at several temperatures. simulate_sweep makes
such a set with the synthetic calibration pipeline (calib.simulated_unit's unit, calib.board_views' rendered
checkerboards, calib.calibrate) on a temperature model built from published material data:
- the image scale (the camera constant a calibration measures, the distance from the exit pupil to the sensor) of a
  fixed-focus module grows with the lens holder between the lens and the sensor: Δc/c = α_h·ΔT (the holder's length
  taken as the focal length); a module that stays in focus (athermalised, or refocused) follows the lens's own focal
  length instead: Δf/f = x_f·ΔT, x_f = α_g − (dn_rel/dT)/(n − 1), the thin-lens thermo-optic coefficient (Jamieson
  1981), with dn_rel/dT varying with temperature as the glass datasheet's rows give it;
- the defocus of a fixed-focus module is (α_h − x_f)·f·ΔT (the lens's focal plane moves by x_f·f·ΔT, the sensor by
  α_h·f·ΔT), compared with the depth of focus ±N·c (N the f-number, c two pixels: a heuristic circle of confusion);
- the principal point moves by α_h·e·ΔT in a seeded direction, e the eccentricity of the holder's fixing about the
  optical axis (HEURISTIC: a sensor board screwed off-centre; a typical few mm, not a datasheet figure);
- distortion k1..k4 are held (a uniform change of scale leaves the Kannala-Brandt terms unchanged; the glass's small
  dispersion change is not modelled).
Real temperature sweeps of the user's own units (a climate chamber, or calibrations logged with the sensor's
temperature over a warm-up) are needed before any curve is trusted: the model gives magnitudes, not a unit's curve.
Published measured drift for comparison (MEASURED_REFERENCE): Elias et al. 2020 found the camera constant and
principal point of smartphone cameras moving linearly with temperature, 0.19-0.41 µm per K of battery temperature.
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

from . import calib as CL

SCHEMA = 'vantage.thermal_intrinsics/1'
METHOD_REVISION = 1
PARAMS = CL.THERMAL_PARAMS                 # ('fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'k3', 'k4')
PX_PARAMS = ('fx', 'fy', 'cx', 'cy')
UNIT = {'fx': 'px', 'fy': 'px', 'cx': 'px', 'cy': 'px', 'k1': '', 'k2': '', 'k3': '', 'k4': ''}
QUAD_GAIN = 0.10                           # a quadratic must cut the leave-one-out RMS by this share to be kept
CURV_T = 3.0                               # ... and its curvature stand this many standard errors from zero
DATE_READ = '2026-11-11'

# ------------------------------------------------------------------------------------------------ published data
REFERENCES = {
    'schott_glass': 'SCHOTT AG, Optical Glass Data Sheets (collection, US edition, 2007): N-BK7 page 11 (as of '
                    '09/03/2001), N-LAK9 page 35, N-SF6 page 74: nd, α(−30/+70 °C) and the temperature coefficients '
                    'of the refractive index Δn_rel/ΔT at the e line (546.1 nm), read ' + DATE_READ + '.',
    'jamieson1981': 'T. H. Jamieson, "Thermal effects in optical systems", Optical Engineering 20(2), 156–160, 1981 '
                    '(the thermo-optic coefficient of a thin lens, x_f = α − (dn/dT)/(n − 1)).',
    'apollo_polymers': 'Apollo Optical Systems, "Common Polymers for Optical Components" (property table: nD, dn/dT, '
                       'CTE of acrylic, polycarbonate, COC and COP), https://www.apollooptical.com/'
                       'common-polymers-for-optical-components, read ' + DATE_READ + '.',
    'lu2010': 'S.-H. D. Lu, "Tutorial for Plastic Optical Materials", OPTI 521, College of Optical Sciences, University '
              'of Arizona, 29 Nov 2010, Table 1 (PMMA, BK7, aluminium: CTE 23.6 ppm/°C).',
    'elias2020': 'M. Elias, A. Eltner, F. Liebold, H.-G. Maas, "Assessing the Influence of Temperature Changes on the '
                 'Geometric Stability of Smartphone- and Raspberry Pi Cameras", Sensors 20(3), 643, 2020, '
                 'doi:10.3390/s20030643, Table 2 and sections 4.1-4.3, read ' + DATE_READ + '.',
}

# Lens materials. dn_rel: Δn_rel/ΔT (relative to air) in 1e-6/K at the e line for the datasheet's three temperature
# ranges, at their mid temperatures (−30, +30, +70 °C); alpha: CTE in 1e-6/K; nd.
LENS_MATERIALS = {
    'n-bk7': {'label': 'Glass N-BK7', 'nd': 1.51680, 'alpha': 7.1, 'dn_rel': ((-30.0, 2.9), (30.0, 3.0), (70.0, 3.1)),
              'source': 'SCHOTT data sheet N-BK7 (page 11): nd 1.51680, α(−30/+70 °C) 7.1e-6/K, Δn_rel/ΔT at e '
                        '2.9 / 3.0 / 3.1 e-6/K for −40/−20, +20/+40, +60/+80 °C'},
    'n-lak9': {'label': 'Glass N-LAK9', 'nd': 1.69100, 'alpha': 6.3, 'dn_rel': ((30.0, 3.7),),
               'source': 'SCHOTT data sheet N-LAK9 (page 35): nd 1.69100, α(−30/+70 °C) 6.3e-6/K, Δn_rel/ΔT at e '
                         '3.7e-6/K (+20/+40 °C)'},
    'n-sf6': {'label': 'Glass N-SF6', 'nd': 1.80518, 'alpha': 9.0, 'dn_rel': ((30.0, 1.5),),
              'source': 'SCHOTT data sheet N-SF6 (page 74): nd 1.80518, α(−30/+70 °C) 9.0e-6/K, Δn_rel/ΔT at e '
                        '1.5e-6/K (+20/+40 °C)'},
    'pmma': {'label': 'Plastic PMMA (acrylic)', 'nd': 1.49, 'alpha': 67.4, 'dn_rel': ((30.0, -85.0),),
             'source': 'Apollo Optical Systems polymer table: acrylic nD 1.49, dn/dT −8.5e-5/°C, CTE 6.74e-5/°C '
                       '(Lu 2010 Table 1 agrees: −8.5 and 72 ppm/°C)'},
    'cop': {'label': 'Plastic COP (Zeonex)', 'nd': 1.53, 'alpha': 65.0, 'dn_rel': ((30.0, -80.0),),
            'source': 'Apollo Optical Systems polymer table: COP (Zeonex) nD 1.53, dn/dT −8e-5/°C, CTE 6.0–7.0e-5/°C '
                      '(6.5e-5 used)'},
}
# Lens holder (barrel and mount) materials: CTE in 1e-6/K.
HOUSING_MATERIALS = {
    'aluminium': {'label': 'Aluminium', 'alpha': 23.6,
                  'source': 'Lu 2010 Table 1: aluminium CTE 23.6 ppm/°C'},
    'polycarbonate': {'label': 'Polycarbonate', 'alpha': 68.0,
                      'source': 'Apollo Optical Systems polymer table: polycarbonate CTE 6.6–7.0e-5/°C (6.8e-5 used)'},
    'glass_holder': {'label': 'Glass-matched (N-BK7 CTE)', 'alpha': 7.1,
                     'source': 'a holder matched to the glass: SCHOTT N-BK7 α(−30/+70 °C) 7.1e-6/K'},
}
SCALES = (('housing', 'Fixed focus: follows the holder'), ('lens', 'In focus: follows the lens'))

# Elias et al. 2020, Table 2 (smartphones, last minus first of two series each; Δt the battery temperature change):
# (camera and start, Δt °C, Δc mm, Δc px, Δx0 px, Δy0 px). µm per K is our division (the battery lags the camera).
MEASURED_REFERENCE = (
    ('Nexus 5, cold start, series 1', 22.9, 0.0078, 5.60, 14.62, -6.11),
    ('Nexus 5, cold start, series 2', 25.0, 0.0082, 5.85, 14.32, -5.11),
    ('Galaxy S8, cold start, series 1', 21.3, 0.0055, 3.95, -29.17, 23.46),
    ('Galaxy S8, cold start, series 2', 26.9, 0.0075, 5.37, -32.87, 26.22),
)
MEASURED_PIXEL_UM = 1.40          # both phones (Elias et al. 2020, Table 1)

ASSUMPTIONS = (
    'Curves are fitted per camera unit by least squares with equal weights; the degree (linear or quadratic) is chosen '
    'per parameter by leave-one-out cross-validation (a quadratic must predict at least 10 % better).',
    'The temperature of a record is the camera\'s while its calibration images were taken (the sensor\'s own reading '
    'when the camera reports one); a temperature that lags the optics (an enclosure or a battery reading) shifts and '
    'flattens the curve.',
    'Outside the temperatures measured a curve is extrapolated: its uncertainty grows and the shape is not known.',
    'Simulated sweeps use published material constants (SCHOTT, Apollo Optical Systems, Lu 2010) with the thin-lens '
    'thermo-optic coefficient (Jamieson 1981); the principal-point drift (holder CTE × an eccentricity) is a heuristic. '
    'They are synthetic data for trying the method: real temperature sweeps of the units are needed.',
)


def x_f(material: str, temp_c: float = 30.0) -> float:
    """The thin-lens thermo-optic coefficient (1/f)·df/dT of a lens of this material at temp_c, per K (Jamieson
    1981): α − (dn_rel/dT)/(n − 1), dn_rel/dT interpolated linearly between the datasheet's temperature ranges."""
    m = LENS_MATERIALS[material]
    pts = m['dn_rel']
    dn = float(np.interp(temp_c, [p[0] for p in pts], [p[1] for p in pts]) if len(pts) > 1 else pts[0][1])
    return (m['alpha'] - dn / (m['nd'] - 1.0)) * 1e-6


def scale_change(material: str, housing: str, scale: str, t_c, ref_c: float = 20.0):
    """Relative change of the image scale (camera constant) at t_c against ref_c: α_h·ΔT for a fixed-focus module
    ('housing'), ∫x_f dT for one that stays in focus ('lens')."""
    t = np.asarray(t_c, float)
    if scale == 'lens':
        # x_f varies (slightly) with temperature through dn/dT: integrate it from ref_c
        grid = np.linspace(-100.0, 200.0, 3001)
        xf = np.array([x_f(material, g) for g in grid])
        cum = np.concatenate([[0.0], np.cumsum((xf[1:] + xf[:-1]) / 2 * np.diff(grid))])
        return np.interp(t, grid, cum) - np.interp(ref_c, grid, cum)
    return HOUSING_MATERIALS[housing]['alpha'] * 1e-6 * (t - ref_c)


# ------------------------------------------------------------------------------------------------ settings
LIMITS = {'reference_c': (-60.0, 150.0), 'confidence': (0.5, 0.999), 'warn_px': (0.05, 50.0),
          'operating_lo_c': (-60.0, 150.0), 'operating_hi_c': (-60.0, 150.0),
          'sweep_lo_c': (-60.0, 150.0), 'sweep_hi_c': (-60.0, 150.0), 'sweep_steps': (2, 15), 'views': (6, 40),
          'scale': (0.25, 1.0), 'noise': (0.0, 10.0), 'mount_ecc_mm': (0.0, 20.0), 'seed': (0, 2 ** 31 - 1)}
INTS = ('sweep_steps', 'views', 'seed')
DEGREES = ('auto', 'linear', 'quadratic')


@dataclass
class ThermalSettings:
    reference_c: float = 20.0           # T_ref: the record a curve gives at its reference is the values here
    degree: str = 'auto'                # 'auto' (leave-one-out) | 'linear' | 'quadratic'
    confidence: float = 0.95            # of the intervals
    warn_px: float = 1.0                # a drift over the operating range above this is flagged
    operating_lo_c: float = -40.0       # the range the issue check judges (the module's own range when it states one)
    operating_hi_c: float = 85.0
    # the synthetic sweep
    lens_material: str = 'n-bk7'
    housing_material: str = 'aluminium'
    image_scale: str = 'housing'        # 'housing' (fixed focus) | 'lens' (stays in focus)
    mount_ecc_mm: float = 3.0           # heuristic eccentricity of the holder's fixing (principal-point drift)
    sweep_lo_c: float = -20.0
    sweep_hi_c: float = 80.0
    sweep_steps: int = 6
    views: int = 14
    scale: float = 0.5                  # rendered image size / sensor size
    noise: float = 2.0                  # grey levels of image noise
    seed: int = 0

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d) -> 'ThermalSettings':
        s = ThermalSettings()
        d = d if isinstance(d, dict) else {}
        for k, (lo, hi) in LIMITS.items():
            if k in d:
                try:
                    v = float(d[k])
                except (TypeError, ValueError, OverflowError):
                    continue
                if math.isfinite(v):
                    v = min(max(v, lo), hi)
                    setattr(s, k, int(round(v)) if k in INTS else v)
        if d.get('degree') in DEGREES:
            s.degree = d['degree']
        if d.get('lens_material') in LENS_MATERIALS:
            s.lens_material = d['lens_material']
        if d.get('housing_material') in HOUSING_MATERIALS:
            s.housing_material = d['housing_material']
        if d.get('image_scale') in dict(SCALES):
            s.image_scale = d['image_scale']
        if s.sweep_hi_c <= s.sweep_lo_c:
            s.sweep_hi_c = min(s.sweep_lo_c + 10.0, LIMITS['sweep_hi_c'][1])
            s.sweep_lo_c = s.sweep_hi_c - 10.0
        if s.operating_hi_c <= s.operating_lo_c:
            s.operating_hi_c = min(s.operating_lo_c + 1.0, LIMITS['operating_hi_c'][1])
            s.operating_lo_c = s.operating_hi_c - 1.0
        return s


def settings_of(project) -> ThermalSettings:
    return ThermalSettings.from_dict(getattr(project.analysis, 'thermal_intrinsics', None))


def _record_ok(r) -> bool:
    try:
        return (isinstance(r, dict) and all(k in r for k in CL.KEYS) and float(r['width']) > 0
                and float(r['height']) > 0 and float(r['fx']) > 0 and float(r['fy']) > 0 and len(r['k']) >= 4
                and all(math.isfinite(float(r[k])) for k in ('fx', 'fy', 'cx', 'cy'))
                and all(math.isfinite(float(x)) for x in list(r['k'])[:4]))
    except (TypeError, ValueError):
        return False


def sweeps_of(project) -> dict:
    """{camera uid: [records with a temperature]} stored on the project (junk dropped)."""
    d = getattr(project.analysis, 'thermal_intrinsics', None)
    raw = d.get('sweeps') if isinstance(d, dict) else None
    out = {}
    for uid, recs in (raw or {}).items() if isinstance(raw, dict) else ():
        good = [CL.clean_record(r) for r in (recs or []) if _record_ok(r) and CL._temp(r.get('temp_c')) is not None]
        if good:
            out[str(uid)] = good
    return out


def with_sweep(d: dict | None, uid: str, records) -> dict:
    """The analysis.thermal_intrinsics dict with camera uid's records replaced (None or [] removes them)."""
    out = dict(d) if isinstance(d, dict) else {}
    sw = dict(out.get('sweeps') or {})
    if records:
        sw[uid] = [CL.clean_record(r) for r in records]
    else:
        sw.pop(uid, None)
    out['sweeps'] = sw
    return out


# ------------------------------------------------------------------------------------------------ records
def rescale(rec: dict, width: int, height: int) -> dict:
    """A record converted to width × height (same aspect), pixel centres at integers (calib.unit_lens' rule)."""
    sx, sy = width / float(rec['width']), height / float(rec['height'])
    out = dict(rec, width=int(width), height=int(height), fx=float(rec['fx']) * sx, fy=float(rec['fy']) * sy,
               cx=(float(rec['cx']) + 0.5) * sx - 0.5, cy=(float(rec['cy']) + 0.5) * sy - 0.5)
    th = rec.get('thermal')
    if isinstance(th, dict) and isinstance(th.get('coeffs'), dict):
        th = json.loads(json.dumps(th))
        for p, s in (('fx', sx), ('cx', sx), ('fy', sy), ('cy', sy)):
            if p in th['coeffs']:
                th['coeffs'][p] = [float(v) * s for v in th['coeffs'][p]]
            if p in (th.get('sigma') or {}):
                th['sigma'][p] = float(th['sigma'][p]) * s
        out['thermal'] = th
    return out


def values(rec: dict) -> dict:
    """{parameter: value} of a record (k split into k1..k4)."""
    out = {p: float(rec[p]) for p in PX_PARAMS}
    out.update({f'k{i + 1}': float(v) for i, v in enumerate(list(rec['k'])[:4])})
    return out


def temperature_of_metadata(meta) -> float | None:
    """The camera's temperature (°C) from a capture's metadata: a dict (JSON sidecar, EXIF-like, a driver's frame
    metadata) with one of calib.TEMP_KEYS or the sensor/ISP names drivers use, searched one level deep too. Kelvin
    values (above 200, under a key ending in _k) are converted."""
    if not isinstance(meta, dict):
        return None
    t = CL.temperature_of(meta)
    if t is not None:
        return t
    for k in ('SensorTemperature', 'sensor_temperature_c', 'CameraTemperature', 'ambient_temperature',
              'AmbientTemperature', 'die_temp_c'):
        if k in meta:
            t = CL._temp(meta[k])
            if t is not None:
                return t
    for k, v in meta.items():
        if isinstance(k, str) and k.lower().endswith(('temp_k', 'temperature_k')):
            try:
                kv = float(v)
            except (TypeError, ValueError):
                continue
            if 150.0 <= kv <= 500.0:
                return kv - 273.15
    for v in meta.values():
        if isinstance(v, dict):
            t = temperature_of_metadata(v)
            if t is not None:
                return t
    return None


TEMPERATURE_TYPE = 'sensor_msgs/msg/Temperature'
TEMPERATURE_DEF = 'std_msgs/Header header\nfloat64 temperature\nfloat64 variance\n'


def ensure_temperature_type():
    """Make core.rosbag know sensor_msgs/Temperature (for writing; bags carry their own definitions for reading)."""
    from . import rosbag as RB
    try:
        RB.schema_text(TEMPERATURE_TYPE)
    except Exception:
        RB.register(TEMPERATURE_TYPE, TEMPERATURE_DEF)


def temperatures_from_bag(path: str, topic: str | None = None) -> tuple:
    """(times ns, temperatures °C) of a sensor_msgs/Temperature topic in a ROS 2 bag (the first such topic when topic
    is None, else the named one); ValueError when the bag has none."""
    from . import rosbag as RB
    with RB.BagReader(path) as bag:
        tops = [t for t, ty in bag.topics.items() if ty.replace('/msg/', '/') == 'sensor_msgs/Temperature']
        if topic:
            if topic not in bag.topics:
                raise ValueError(f'the bag has no topic {topic}')
            tops = [topic]
        if not tops:
            raise ValueError('the bag has no sensor_msgs/Temperature topic')
        ts, vs = [], []
        for _, t_ns, msg in bag.messages(tops[:1]):
            if isinstance(msg, dict) and CL._temp(msg.get('temperature')) is not None:
                ts.append(int(t_ns))
                vs.append(float(msg['temperature']))
    return np.array(ts, np.int64), np.array(vs, float)


def temperature_at(times_ns, temps, t_ns) -> float | None:
    """The temperature at t_ns, linear between readings (held at the ends); None without readings."""
    if len(times_ns) == 0:
        return None
    return float(np.interp(float(t_ns), np.asarray(times_ns, float), np.asarray(temps, float)))


# ------------------------------------------------------------------------------------------------ the fit
def _student_t(conf: float, dof: int) -> float:
    if dof <= 0:
        return float('nan')
    try:
        from scipy.stats import t
        return float(t.ppf(0.5 + conf / 2.0, dof))
    except Exception:                    # scipy missing: the normal quantile with a small-sample inflation
        from statistics import NormalDist
        z = NormalDist().inv_cdf(0.5 + conf / 2.0)
        return z * (1.0 + (z * z + 1.0) / (4.0 * dof))


def polyfit(dt, y, degree: int) -> dict:
    """Least-squares polynomial of y on dt: {'degree', 'coef' [c0, c1, c2] (zero-padded), 'cov' (3×3), 'sigma'
    (residual SD, NaN without degrees of freedom), 'dof', 'press_rmse' (leave-one-out RMS, NaN when not defined),
    'loo' (the leave-one-out residuals)}."""
    dt, y = np.asarray(dt, float), np.asarray(y, float)
    n, p = len(dt), degree + 1
    X = np.vander(dt, p, increasing=True)
    sc = np.maximum(np.abs(X).max(0), 1e-12)                  # column scaling, then QR: well conditioned
    Q, R = np.linalg.qr(X / sc)
    Rinv = np.linalg.pinv(R)
    beta = (Rinv @ (Q.T @ y)) / sc
    XtX_inv = (Rinv @ Rinv.T) / np.outer(sc, sc)
    r = y - X @ beta
    dof = n - p
    s2 = float(r @ r / dof) if dof > 0 else float('nan')
    cov = np.full((3, 3), np.nan) if dof <= 0 else np.zeros((3, 3))
    if dof > 0:
        cov[:p, :p] = s2 * XtX_inv
    h = np.sum(Q * Q, axis=1) if n >= p else np.ones(n)
    with np.errstate(divide='ignore', invalid='ignore'):
        loo = np.where(h < 1 - 1e-9, r / (1 - h), np.nan)
    press = float(np.sqrt(np.mean(loo ** 2))) if n >= p + 1 and np.all(np.isfinite(loo)) else float('nan')
    coef = np.zeros(3)
    coef[:p] = beta
    return {'degree': degree, 'coef': coef, 'cov': cov, 'sigma': math.sqrt(s2) if dof > 0 else float('nan'),
            'dof': dof, 'press_rmse': press, 'loo': loo, 'n': n}


def loo_brute(dt, y, degree: int) -> np.ndarray:
    """Leave-one-out residuals by refitting without each point (the independent check of polyfit's closed form)."""
    dt, y = np.asarray(dt, float), np.asarray(y, float)
    out = np.empty(len(dt))
    for i in range(len(dt)):
        m = np.arange(len(dt)) != i
        X = np.vander(dt[m], degree + 1, increasing=True)
        b, *_ = np.linalg.lstsq(X, y[m], rcond=None)
        out[i] = y[i] - float((np.vander(dt[i:i + 1], degree + 1, increasing=True) @ b)[0])
    return out


def choose(dt, y, degree: str = 'auto') -> tuple:
    """(fit, reason, {degree: press_rmse}) for one parameter."""
    n = len(dt)
    fits = {d: polyfit(dt, y, d) for d in (1, 2) if n >= d + 1}
    press = {d: f['press_rmse'] for d, f in fits.items()}
    if n == 0:
        raise ValueError('no records')
    if n == 1:
        return polyfit(dt, y, 0), 'one temperature: no drift can be fitted', press
    if degree == 'quadratic' and n >= 3:
        return fits[2], 'quadratic (chosen in the settings)', press
    if degree == 'linear' or n < 4 or degree == 'quadratic':
        why = ('linear (chosen in the settings)' if degree == 'linear' else
               'linear: a quadratic needs 3 temperatures' if degree == 'quadratic' else
               'linear: two temperatures give the line through them, without an uncertainty' if n == 2 else
               'linear: choosing by cross-validation needs 4 temperatures or more')
        return fits[1], why, press
    p1, p2 = press[1], press[2]
    c2, se2 = fits[2]['coef'][2], math.sqrt(fits[2]['cov'][2, 2]) if np.isfinite(fits[2]['cov'][2, 2]) else math.inf
    t2 = abs(c2) / se2 if se2 > 0 else math.inf
    if math.isfinite(p2) and math.isfinite(p1) and p2 < p1 * (1 - QUAD_GAIN):
        if t2 < CURV_T:
            return fits[1], (f'linear: the quadratic predicts a left-out temperature {100 * (1 - p2 / p1):.0f} % '
                             f'better but its curvature is only {t2:.1f} σ (one odd calibration, not a curve)'), press
        return fits[2], (f'quadratic: predicts a left-out temperature {100 * (1 - p2 / p1):.0f} % better, curvature '
                         f'{t2:.1f} σ'), press
    if math.isfinite(p1) and math.isfinite(p2):
        return fits[1], (f'linear: the quadratic predicts a left-out temperature only {100 * (1 - p2 / p1):.0f} % '
                         f'better (needs {100 * QUAD_GAIN:.0f} %)' if p2 < p1 else
                         'linear: the quadratic predicts a left-out temperature worse'), press
    return fits[1], 'linear', press


@dataclass
class Curve:
    """One camera unit's thermal curves (at the records' common resolution)."""
    uid: str
    tag: str
    serial: str
    width: int
    height: int
    reference_c: float
    temps: list                          # °C, one per record
    params: dict                         # name -> fit dict (+ 'reason', 'press' {1: rmse, 2: rmse})
    confidence: float = 0.95
    k_base: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    truth: dict | None = None            # a simulated sweep's true model (simulate_sweep), for comparison

    @property
    def n(self) -> int:
        return len(self.temps)

    @property
    def range_c(self) -> tuple:
        return (min(self.temps), max(self.temps)) if self.temps else (self.reference_c, self.reference_c)

    def value(self, name, temp_c) -> float:
        f = self.params[name]
        d = np.asarray(temp_c, float) - self.reference_c
        c = f['coef']
        return c[0] + c[1] * d + c[2] * d * d

    def interval(self, name, temp_c) -> dict:
        """{'value', 'se_mean', 'se_pred', 'ci': (lo, hi), 'pi': (lo, hi)} at temp_c (NaN where not estimable)."""
        f = self.params[name]
        d = float(temp_c) - self.reference_c
        x = np.array([1.0, d, d * d])
        v = float(x @ f['coef'])
        se_m = float(math.sqrt(max(0.0, x @ f['cov'] @ x))) if np.all(np.isfinite(f['cov'])) else float('nan')
        se_p = math.hypot(se_m, f['sigma']) if math.isfinite(se_m) else float('nan')
        t = _student_t(self.confidence, f['dof'])
        return {'value': v, 'se_mean': se_m, 'se_pred': se_p, 'ci': (v - t * se_m, v + t * se_m),
                'pi': (v - t * se_p, v + t * se_p)}

    def slope(self, name, temp_c=None) -> float:
        """d(parameter)/dT at temp_c (the reference by default), per K."""
        d = 0.0 if temp_c is None else float(temp_c) - self.reference_c
        c = self.params[name]['coef']
        return float(c[1] + 2 * c[2] * d)

    def drift(self, name, lo_c, hi_c) -> float:
        """Largest change of the parameter over [lo_c, hi_c] against its value at the reference."""
        ts = np.linspace(lo_c, hi_c, 101)
        return float(np.max(np.abs(self.value(name, ts) - self.value(name, self.reference_c))))

    def record_at(self, temp_c: float) -> dict:
        v = {p: float(self.value(p, temp_c)) for p in PARAMS}
        rec = CL.make_record(self.width, self.height, v['fx'], v['fy'], v['cx'], v['cy'],
                             [v[f'k{i}'] for i in range(1, 5)], serial=self.serial, source='thermal curve',
                             date='', temp_c=temp_c)
        return rec

    def thermal_block(self) -> dict:
        lo, hi = self.range_c
        return {'model': 'polynomial', 'reference_c': float(self.reference_c), 'range_c': [float(lo), float(hi)],
                'coeffs': {p: [float(self.params[p]['coef'][1]), float(self.params[p]['coef'][2])] for p in PARAMS},
                'sigma': {p: (float(self.params[p]['sigma']) if math.isfinite(self.params[p]['sigma']) else 0.0)
                          for p in PARAMS},
                'degree': {p: int(self.params[p]['degree']) for p in PARAMS}, 'n': self.n}

    def reference_record(self) -> dict:
        """The record at the reference temperature carrying the curve: what compensate() and the exports use."""
        rec = self.record_at(self.reference_c)
        rec.pop('temp_c', None)
        rec['thermal'] = self.thermal_block()
        return rec


def fit_unit(records, reference_c: float = 20.0, degree: str = 'auto', size=None, confidence: float = 0.95,
             uid: str = '', tag: str = '') -> Curve:
    """Curves of one unit from its records (each with temp_c). size: (width, height) to rescale the records to (the
    module's full resolution); default the first record's."""
    recs = [r for r in records if _record_ok(r) and CL._temp(r.get('temp_c')) is not None]
    if not recs:
        raise ValueError('no calibration with a temperature')
    w, h = size or (int(recs[0]['width']), int(recs[0]['height']))
    notes = []
    aspect = w / h
    keep = []
    for r in recs:
        if abs(float(r['width']) / float(r['height']) - aspect) > 0.005 * aspect:
            notes.append(f'a calibration at {r["temp_c"]:g} °C is {r["width"]} × {r["height"]}, not the sensor\'s '
                         f'aspect: left out')
            continue
        keep.append(rescale(r, w, h) if (int(r['width']), int(r['height'])) != (w, h) else dict(r))
    if not keep:
        raise ValueError('no calibration with the sensor\'s aspect')
    keep.sort(key=lambda r: float(r['temp_c']))
    T = np.array([float(r['temp_c']) for r in keep])
    if len(set(np.round(T, 3))) < len(T):
        notes.append('several calibrations at the same temperature: each counts as its own point (their scatter '
                     'is part of the residual)')
    dt = T - reference_c
    vals = [values(r) for r in keep]
    params = {}
    for p in PARAMS:
        y = np.array([v[p] for v in vals])
        f, why, press = choose(dt, y, degree)
        f = dict(f, reason=why, press=press, y=y)
        params[p] = f
    if len(set(np.round(T, 3))) >= 2 and (T.max() - T.min()) < 10.0:
        notes.append(f'the calibrations span only {T.max() - T.min():.1f} K: the drift per K is poorly determined')
    return Curve(uid, tag, str(keep[0].get('serial', '')), int(w), int(h), float(reference_c), [float(t) for t in T],
                 params, float(confidence), notes=notes)


def compensate(rec_or_curve, temp_c: float) -> dict:
    """The intrinsics at temp_c: from a Curve, or from a record carrying a 'thermal' block (its values are those at
    the block's reference_c). A record without a curve comes back unchanged (with a note in 'thermal_note')."""
    if isinstance(rec_or_curve, Curve):
        out = rec_or_curve.record_at(temp_c)
        out['thermal'] = rec_or_curve.thermal_block()
        return out
    rec = dict(rec_or_curve)
    th = rec.get('thermal')
    if not isinstance(th, dict) or not isinstance(th.get('coeffs'), dict):
        rec['thermal_note'] = 'no thermal curve: the calibration is used as it is'
        return rec
    d = float(temp_c) - float(th.get('reference_c', 20.0))
    v = values(rec)
    for p in PARAMS:
        a = (list(th['coeffs'].get(p, [0.0, 0.0])) + [0.0, 0.0])[:2]
        v[p] += float(a[0]) * d + float(a[1]) * d * d
    out = dict(rec, fx=v['fx'], fy=v['fy'], cx=v['cx'], cy=v['cy'], k=[v[f'k{i}'] for i in range(1, 5)],
               temp_c=float(temp_c))
    lo, hi = (list(th.get('range_c') or [d, d]) + [0, 0])[:2]
    if not (float(lo) - 1e-9 <= float(temp_c) <= float(hi) + 1e-9):
        out['thermal_note'] = (f'{temp_c:g} °C is outside the {lo:g} … {hi:g} °C the curve was fitted over: '
                               'extrapolated')
    return out


def curve_for(project, cam, settings: ThermalSettings | None = None) -> Curve | None:
    """The fitted curve of a camera from its stored sweep (None without one)."""
    st = settings or settings_of(project)
    recs = sweeps_of(project).get(cam.uid)
    if not recs:
        return None
    m = project.module_for(cam)
    return fit_unit(recs, st.reference_c, st.degree, (m.width_px, m.height_px), st.confidence, cam.uid, cam.tag)


def lens_at(project, cam, temp_c: float, settings=None):
    """The camera's lens model at temp_c: its sweep's curve, else its unit calibration's thermal block; the lens in
    use (unchanged) when it has neither."""
    cur = curve_for(project, cam, settings)
    base = project.nominal_lens_for(cam)
    m = project.module_for(cam)
    if cur is not None:
        return CL.unit_lens(base, m, CL.clean_record(compensate(cur, temp_c)))
    if cam.calib and isinstance(cam.calib.get('thermal'), dict):
        return CL.unit_lens(base, m, CL.clean_record(compensate(cam.calib, temp_c)))
    return project.lens_for(cam)


# ------------------------------------------------------------------------------------------------ synthetic sweeps
def true_model(module, lens, base_rec: dict, st: ThermalSettings) -> dict:
    """The temperature model a simulated sweep follows (see the module docstring): per parameter a function of T
    and the figures behind it."""
    rng = np.random.default_rng([int(st.seed), 7193])
    ang = rng.uniform(0, 2 * math.pi)
    a_h = HOUSING_MATERIALS[st.housing_material]['alpha'] * 1e-6
    pix_mm = module.pixel_um * 1e-3 * module.width_px / float(base_rec['width'])
    f_mm = float(base_rec['fx']) * pix_mm
    pp_px_per_k = a_h * st.mount_ecc_mm / pix_mm
    ref = 20.0                             # the glass data's reference temperature
    xf = x_f(st.lens_material, 30.0)
    n_f = float(lens.f_number) if lens.f_number and lens.f_number > 0 else 2.0
    dof_mm = n_f * 2 * pix_mm             # depth of focus, one side: N × a two-pixel circle of confusion
    return {'reference_c': ref, 'alpha_housing': a_h, 'x_f': xf, 'f_mm': f_mm, 'pixel_mm': pix_mm,
            'pp_px_per_k': pp_px_per_k, 'pp_dir': (math.cos(ang), math.sin(ang)), 'f_number': n_f,
            'depth_of_focus_mm': dof_mm, 'defocus_mm_per_k': (a_h - xf) * f_mm if st.image_scale == 'housing' else 0.0,
            'lens': LENS_MATERIALS[st.lens_material]['source'],
            'housing': HOUSING_MATERIALS[st.housing_material]['source'], 'image_scale': st.image_scale,
            'lens_material': st.lens_material, 'housing_material': st.housing_material,
            'mount_ecc_mm': st.mount_ecc_mm}


def true_record(base_rec: dict, model: dict, temp_c: float) -> dict:
    """base_rec (at the model's reference temperature) moved to temp_c by the model."""
    s = float(scale_change(model['lens_material'], model['housing_material'], model['image_scale'], temp_c,
                           model['reference_c']))
    d = float(temp_c) - model['reference_c']
    dx, dy = model['pp_dir']
    k = model['pp_px_per_k'] * d
    return dict(base_rec, fx=float(base_rec['fx']) * (1 + s), fy=float(base_rec['fy']) * (1 + s),
                cx=float(base_rec['cx']) + k * dx, cy=float(base_rec['cy']) + k * dy, temp_c=float(temp_c))


def sweep_temps(st: ThermalSettings) -> list:
    return [float(t) for t in np.linspace(st.sweep_lo_c, st.sweep_hi_c, st.sweep_steps)]


def simulate_sweep(module, lens, st: ThermalSettings | None = None, base_rec: dict | None = None, render=True,
                   progress=None, cancel=None, serial='') -> dict:
    """A synthetic temperature sweep of one unit: {'records' (measured, each with temp_c), 'truth' (the true records
    at the module's resolution), 'model' (true_model)}. render=True runs the real synthetic calibration (rendered
    checkerboards through the true intrinsics, calib.calibrate on them); render=False adds Gaussian noise of the size
    those calibrations show (for fast tests). None when cancelled."""
    st = st or ThermalSettings()
    base = base_rec or CL.simulated_unit(module, lens, seed=st.seed, serial=serial or f'SIM{st.seed:04d}')
    base = rescale(base, module.width_px, module.height_px)
    model = true_model(module, lens, base, st)
    temps = sweep_temps(st)
    truth = [true_record(base, model, t) for t in temps]
    out = []
    rng = np.random.default_rng([int(st.seed), 4401])
    for i, (t, tr) in enumerate(zip(temps, truth)):
        if cancel is not None and cancel():
            return None
        if render:
            ims = CL.board_views(module, lens, CL.clean_record(tr), n=st.views, seed=st.seed * 1000 + i + 1,
                                 scale=st.scale, noise=st.noise)
            r = CL.calibrate(ims, (9, 6), 60.0, module, lens, serial=base.get('serial', ''),
                             source=f'simulated sweep at {t:g} °C')
            if r.record is None:
                if progress:
                    progress((i + 1) / len(temps), f'{t:g} °C: {r.message}')
                continue
            rec = dict(r.record, temp_c=float(t))
        else:
            e = rng.normal(0.0, 1.0, 6)
            rec = dict(tr, fx=tr['fx'] + 0.6 * e[0], fy=tr['fy'] + 0.6 * e[1], cx=tr['cx'] + 0.5 * e[2],
                       cy=tr['cy'] + 0.5 * e[3], k=[k * (1 + 0.002 * rng.normal()) for k in tr['k']],
                       source=f'simulated (noise only) at {t:g} °C', rms_px=0.08, views=st.views)
        rec['date'] = ''
        out.append(CL.clean_record(rec))
        if progress:
            progress((i + 1) / len(temps), f'{t:g} °C calibrated ({i + 1} of {len(temps)})')
    return {'records': out, 'truth': [CL.clean_record(r) for r in truth], 'model': model}


def compare_truth(curve: Curve, sim: dict) -> dict:
    """How far the fitted curve is from a simulated sweep's truth: per pixel parameter the slope (per K) fitted and
    true, its σ and the error in σ, and the worst error of the curve over the swept range (px)."""
    model = sim['model']
    truth = [rescale(r, curve.width, curve.height) for r in sim['truth']]
    T = np.array([r['temp_c'] for r in truth])
    out = {}
    for p in PX_PARAMS:
        y = np.array([float(r[p]) for r in truth])
        true_slope = float(np.polyfit(T - curve.reference_c, y, 1)[0]) if len(T) > 1 else 0.0
        f = curve.params[p]
        se = math.sqrt(f['cov'][1, 1]) if np.isfinite(f['cov'][1, 1]) else float('nan')
        est = curve.slope(p, float(np.mean(T)))
        worst = float(np.max(np.abs(curve.value(p, T) - y)))
        out[p] = {'true_per_k': true_slope, 'fitted_per_k': est, 'se_per_k': se,
                  'error_sigma': abs(est - true_slope) / se if se and math.isfinite(se) else float('nan'),
                  'worst_px': worst, 'true_drift_px': float(np.ptp(y))}
    out['_model'] = {k: model[k] for k in ('alpha_housing', 'x_f', 'f_mm', 'pp_px_per_k', 'defocus_mm_per_k',
                                            'depth_of_focus_mm', 'image_scale')}
    return out


# ------------------------------------------------------------------------------------------------ the study
@dataclass
class ThermalResult:
    settings: ThermalSettings
    curves: list                         # Curve per camera with a sweep
    cameras: list                        # [{'uid', 'tag', 'n', 'state'}] every visible camera of the variant
    findings: list                       # [{'severity', 'message', 'uids'}]
    metrics: dict
    variant_uid: str = ''
    variant: str = ''

    def curve(self, uid) -> Curve | None:
        return next((c for c in self.curves if c.uid == uid), None)


def _operating(st, module) -> tuple:
    rng = tuple(getattr(module, 'temp_c', ()) or ())
    if len(rng) == 2 and all(isinstance(v, (int, float)) and math.isfinite(v) for v in rng) and rng[1] > rng[0]:
        return float(rng[0]), float(rng[1]), 'the module\'s operating range'
    return st.operating_lo_c, st.operating_hi_c, 'the operating range in the settings'


def run(project, engine=None, settings=None, variant=None, progress=None, cancel=None) -> ThermalResult | None:
    """Fit every camera of the variant that has a temperature sweep (cheap: no rendering)."""
    t0 = time.perf_counter()
    st = settings or settings_of(project)
    var = project.variant_by_uid(variant) if isinstance(variant, str) else variant
    var = var or project.variant
    sw = sweeps_of(project)
    curves, cams, findings = [], [], []
    vis = [c for c in var.cameras if c.visible]
    for i, c in enumerate(vis):
        if cancel is not None and cancel():
            return None
        recs = sw.get(c.uid, [])
        e = {'uid': c.uid, 'tag': c.tag, 'n': len(recs), 'state': 'none'}
        if recs:
            try:
                cur = curve_for(project, c, st)
                curves.append(cur)
                e['state'] = 'fitted' if cur.n >= 2 else 'one'
                lo, hi, what = _operating(st, project.module_for(c))
                e['drift_px'] = max(cur.drift(p, lo, hi) for p in PX_PARAMS)
                e['operating'] = (lo, hi)
                for n in cur.notes:
                    findings.append({'severity': 'warning', 'message': f'{c.tag}: {n}.', 'uids': [c.uid]})
                if cur.n >= 2 and e['drift_px'] > st.warn_px:
                    worst = max(PX_PARAMS, key=lambda p: cur.drift(p, lo, hi))
                    findings.append({'severity': 'warning', 'uids': [c.uid], 'message':
                                     f'{c.tag}: {worst} moves up to {cur.drift(worst, lo, hi):.2f} px over {what} '
                                     f'({lo:g} … {hi:g} °C; limit {st.warn_px:g} px): compensate it for the '
                                     f'temperature of each capture'})
                clo, chi = cur.range_c
                if cur.n >= 2 and (lo < clo - 5 or hi > chi + 5):
                    findings.append({'severity': 'note', 'uids': [c.uid], 'message':
                                     f'{c.tag}: calibrated at {clo:g} … {chi:g} °C, operated at {lo:g} … {hi:g} °C: '
                                     f'the curve is extrapolated outside its sweep'})
                if all(r.get('source', '').startswith('simulated') for r in recs):
                    findings.append({'severity': 'note', 'uids': [c.uid], 'message':
                                     f'{c.tag}: the sweep is simulated (a material model, not a measurement): '
                                     f'calibrate the unit at several temperatures before relying on the curve'})
            except ValueError as ex:
                e['state'] = 'error'
                findings.append({'severity': 'warning', 'message': f'{c.tag}: {ex}.', 'uids': [c.uid]})
        cams.append(e)
        if progress:
            progress((i + 1) / max(1, len(vis)), c.tag)
    fitted = [e for e in cams if e.get('drift_px') is not None and e['state'] == 'fitted']
    worst = max(fitted, key=lambda e: e['drift_px'], default=None)
    metrics = {'cameras': len(cams), 'with_sweep': sum(e['n'] > 0 for e in cams),
               'fitted': len(fitted), 'records': sum(e['n'] for e in cams),
               'worst_drift_px': worst['drift_px'] if worst else None, 'worst_tag': worst['tag'] if worst else None,
               'seconds': time.perf_counter() - t0}
    return ThermalResult(st, curves, cams, findings, metrics, var.uid, var.name)


def issues(project, vg=None) -> list:
    """Cheap: one Issue per camera whose fitted drift over its operating range exceeds the limit."""
    if not sweeps_of(project):
        return []
    from .rules import Issue
    res = run(project)
    return [Issue('warning', f['message'], f['message'].split(':')[0], list(f['uids']))
            for f in res.findings if f['severity'] == 'warning']


def _r(v, d=6):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return round(v, d) + 0.0 if math.isfinite(v) else None


def curve_doc(cur: Curve) -> dict:
    lo, hi = cur.range_c
    doc = {'uid': cur.uid, 'tag': cur.tag, 'serial': cur.serial, 'resolution': [cur.width, cur.height],
           'reference_c': cur.reference_c, 'range_c': [lo, hi], 'temperatures_c': cur.temps,
           'confidence': cur.confidence, 'notes': list(cur.notes), 'params': {}}
    for p in PARAMS:
        f = cur.params[p]
        ref = cur.interval(p, cur.reference_c)
        doc['params'][p] = {
            'unit': UNIT[p], 'degree': int(f['degree']), 'reason': f['reason'],
            'value_at_reference': _r(f['coef'][0], 8), 'per_k': _r(f['coef'][1], 9), 'per_k2': _r(f['coef'][2], 11),
            'se': [_r(math.sqrt(f['cov'][i, i]), 9) if np.isfinite(f['cov'][i, i]) else None for i in range(3)],
            'residual_sd': _r(f['sigma'], 9), 'dof': int(f['dof']),
            'loo_rmse': {'linear': _r(f['press'].get(1), 9), 'quadratic': _r(f['press'].get(2), 9)},
            'prediction_sd_at_reference': _r(ref['se_pred'], 9),
            'measured': [_r(v, 8) for v in f['y']]}
    doc['thermal_block'] = cur.thermal_block()
    if cur.truth is not None:
        doc['simulated_truth'] = json.loads(json.dumps(cur.truth, default=float))
    return doc


def summary(res: ThermalResult) -> dict:
    out = {'variant': res.variant, 'reference_c': res.settings.reference_c, 'degree': res.settings.degree,
           'cameras': [], 'findings': [f['message'] for f in res.findings],
           'worst_drift_px': _r(res.metrics['worst_drift_px'], 3), 'worst_tag': res.metrics['worst_tag']}
    for e in res.cameras:
        row = {'tag': e['tag'], 'calibrations': e['n'], 'state': e['state']}
        cur = res.curve(e['uid'])
        if cur is not None:
            row.update({'range_c': list(cur.range_c),
                        'per_k': {p: _r(cur.params[p]['coef'][1], 5) for p in PX_PARAMS},
                        'degree': {p: int(cur.params[p]['degree']) for p in PX_PARAMS},
                        'drift_over_operating_px': _r(e.get('drift_px'), 3)})
        out['cameras'].append(row)
    return out


def to_doc(res: ThermalResult, project=None) -> dict:
    return {'schema': SCHEMA, 'method_revision': METHOD_REVISION,
            'project': getattr(project, 'name', ''), 'variant': res.variant,
            'conventions': {'pixels': 'OpenCV pixel coordinates (pixel centre at integer i) at each curve\'s resolution '
                                      '(the module\'s full resolution)',
                            'model': 'p(T) = value_at_reference + per_k·(T − reference_c) + per_k2·(T − reference_c)²',
                            'distortion': 'cv::fisheye (Kannala-Brandt) k1..k4',
                            'temperature': '°C, the camera\'s temperature while its calibration images were taken',
                            'degree': f'chosen per parameter by leave-one-out cross-validation (quadratic kept when its '
                                      f'leave-one-out RMS is {100 * QUAD_GAIN:.0f} % or more below the linear\'s)'},
            'settings': res.settings.to_dict(), 'assumptions': list(ASSUMPTIONS),
            'curves': [curve_doc(c) for c in res.curves],
            'cameras': [{k: v for k, v in e.items() if k != 'uid'} for e in res.cameras],
            'findings': res.findings, 'metrics': {k: v for k, v in res.metrics.items()},
            'references': REFERENCES}


def to_csv(res: ThermalResult) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(['camera', 'parameter', 'unit', 'degree', 'value_at_reference', 'per_k', 'per_k2', 'se_per_k',
                'residual_sd', 'loo_rmse_linear', 'loo_rmse_quadratic', 'reference_c', 'lo_c', 'hi_c', 'n'])
    for c in res.curves:
        lo, hi = c.range_c
        for p in PARAMS:
            f = c.params[p]
            w.writerow([c.tag, p, UNIT[p], f['degree'], _r(f['coef'][0], 8), _r(f['coef'][1], 9),
                        _r(f['coef'][2], 11), _r(math.sqrt(f['cov'][1, 1]), 9) if np.isfinite(f['cov'][1, 1]) else '',
                        _r(f['sigma'], 9), _r(f['press'].get(1), 9), _r(f['press'].get(2), 9), c.reference_c, lo, hi,
                        c.n])
    return buf.getvalue()


def kalibr_yaml(res: ThermalResult) -> str:
    """Kalibr camchain (pinhole-equi, the reference-temperature intrinsics) with a 'thermal' block per camera:
    reference_c, range_c and per parameter [a1 per K, a2 per K²] and the residual sigma, in the file's pixels.
    Kalibr reads only the keys it knows; the block is for tools that compensate."""
    lines = []
    for i, c in enumerate(res.curves):
        r = c.reference_record()
        th = r['thermal']
        lines += [f'cam{i}:', '  camera_model: pinhole', f'  intrinsics: [{r["fx"]!r}, {r["fy"]!r}, {r["cx"]!r}, '
                  f'{r["cy"]!r}]', '  distortion_model: equidistant',
                  f'  distortion_coeffs: [{", ".join(repr(float(k)) for k in r["k"])}]',
                  f'  resolution: [{c.width}, {c.height}]', f'  rostopic: /{c.tag}/image_raw',
                  '  thermal:', f'    reference_c: {th["reference_c"]!r}',
                  f'    range_c: [{th["range_c"][0]!r}, {th["range_c"][1]!r}]',
                  '    model: "p(T) = p + a1 (T - reference_c) + a2 (T - reference_c)^2"', '    coeffs:']
        lines += [f'      {p}: [{th["coeffs"][p][0]!r}, {th["coeffs"][p][1]!r}]' for p in PARAMS]
        lines += ['    sigma:'] + [f'      {p}: {th["sigma"][p]!r}' for p in PARAMS]
    return '\n'.join(lines) + '\n'


def png(res: ThermalResult, width=900) -> bytes:
    """fx, cx and cy of every fitted camera against temperature (change from the reference, px) with the measured
    points: one panel per parameter."""
    import cv2
    H, pad = 260, 46
    img = np.full((H * 3, width, 3), 255, np.uint8)
    cols = [(229, 135, 57), (38, 89, 217), (112, 158, 25), (0, 133, 201), (129, 81, 213), (0, 131, 0)]
    curves = [c for c in res.curves if c.n >= 2]
    for k, p in enumerate(('fx', 'cx', 'cy')):
        y0 = k * H
        cv2.putText(img, f'{p} - {p}(T_ref)  [px]', (pad, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1,
                    cv2.LINE_AA)
        if not curves:
            continue
        lo = min(c.range_c[0] for c in curves) - 5
        hi = max(c.range_c[1] for c in curves) + 5
        ys = [c.params[p]['y'] - c.value(p, c.reference_c) for c in curves]
        span = max(0.5, max(float(np.max(np.abs(y))) for y in ys) * 1.2)
        X = lambda t: int(pad + (t - lo) / (hi - lo) * (width - 2 * pad))           # noqa: E731
        Y = lambda v: int(y0 + H / 2 + 8 - v / span * (H / 2 - 30))                  # noqa: E731
        cv2.line(img, (pad, Y(0)), (width - pad, Y(0)), (200, 200, 200), 1)
        for t in np.arange(math.ceil(lo / 20) * 20, hi, 20):
            cv2.putText(img, f'{t:g}', (X(t) - 8, y0 + H - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (90, 90, 90), 1,
                        cv2.LINE_AA)
        for j, c in enumerate(curves):
            col = cols[j % len(cols)]
            ts = np.linspace(lo, hi, 80)
            ref = c.value(p, c.reference_c)
            pts = np.array([[X(t), Y(c.value(p, t) - ref)] for t in ts], np.int32)
            cv2.polylines(img, [pts], False, col, 2, cv2.LINE_AA)
            for t, y in zip(c.temps, c.params[p]['y']):
                cv2.circle(img, (X(t), Y(y - ref)), 3, col, -1, cv2.LINE_AA)
            cv2.putText(img, c.tag, (width - pad + 4, Y(c.value(p, hi) - ref)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col,
                        1, cv2.LINE_AA)
    ok, buf = cv2.imencode('.png', img)
    return buf.tobytes()


def files(res: ThermalResult, project=None) -> list:
    out = [('thermal_curves.json', (json.dumps(to_doc(res, project), indent=1, ensure_ascii=False, default=float)
                                    + '\n').encode('utf-8')),
           ('thermal_curves.csv', to_csv(res).encode('utf-8'))]
    if res.curves:
        out.append(('thermal_curves.png', png(res)))
        out.append(('kalibr/camchain_thermal.yaml', kalibr_yaml(res).encode('utf-8')))
        for c in res.curves:
            out.append((f'opencv/{c.tag}_thermal.yaml', CL.to_opencv_yaml(c.reference_record(), c.tag).encode('utf-8')))
    return out


def export(res: ThermalResult, project, out_dir: str) -> list:
    paths = []
    for name, data in files(res, project):
        p = os.path.join(out_dir, name)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, 'wb') as fh:
            fh.write(data)
        paths.append(p)
    return paths


def exit_code(res: ThermalResult | None) -> int:
    """Batch: 0 every fitted camera within the limit, 1 a drift over the limit, 2 nothing to fit (no sweeps)."""
    if res is None or not res.curves:
        return 2
    return 1 if any(f['severity'] == 'warning' and 'moves up to' in f['message'] for f in res.findings) else 0


# ------------------------------------------------------------------------------------------------ validation
def _check_press(ctx):
    from ..validation.suite import Outcome
    rng = ctx.rng('thermal_intrinsics.press_closed_form')
    worst, n_cases = 0.0, 0
    for n in (4, 6, 9):
        for d in (1, 2):
            T = np.sort(rng.uniform(-30, 80, n))
            y = 860 + 0.02 * (T - 20) + 1e-4 * (T - 20) ** 2 + rng.normal(0, 0.4, n)
            fit = polyfit(T - 20, y, d)
            brute = loo_brute(T - 20, y, d)
            worst = max(worst, float(np.max(np.abs(fit['loo'] - brute))))
            n_cases += 1
    return Outcome(worst, {'cases': n_cases, 'largest difference px': worst})


def _check_coverage(ctx):
    from ..validation.suite import Outcome
    rng = ctx.rng('thermal_intrinsics.interval_coverage')
    trials = 400 if ctx.quick else 2000
    T = np.linspace(-20, 80, 6)
    hits, sig = 0, 0.5
    for _ in range(trials):
        y = 860 + 0.02 * (T - 20) + rng.normal(0, sig, len(T))
        cur = Curve('u', 'U', '', 1920, 1536, 20.0, list(T), {}, 0.95)
        f = polyfit(T - 20, y, 1)
        cur.params = {p: dict(f, reason='', press={}, y=y) for p in PARAMS}
        t_new = float(rng.uniform(-20, 80))
        y_new = 860 + 0.02 * (t_new - 20) + rng.normal(0, sig)
        lo, hi = cur.interval('fx', t_new)['pi']
        hits += lo <= y_new <= hi
    cov = hits / trials
    band = 2.6 * math.sqrt(0.95 * 0.05 / trials)
    return Outcome(abs(cov - 0.95), {'trials': trials, 'coverage': cov}, tolerance=max(band, 0.012))


def _check_recovery(ctx):
    from ..validation.suite import Outcome
    p = ctx.sample()
    cam = next(v for v in p.variants if v.sensors).cameras[0]
    m, lens = p.module_for(cam), p.nominal_lens_for(cam)
    st = ThermalSettings(housing_material='polycarbonate', sweep_steps=4 if ctx.quick else 6,
                         views=10 if ctx.quick else 14, scale=0.35 if ctx.quick else 0.5, seed=3)
    sim = simulate_sweep(m, lens, st)
    cur = fit_unit(sim['records'], 20.0, 'linear', (m.width_px, m.height_px))
    cmp_ = compare_truth(cur, sim)
    err = max(cmp_[q]['error_sigma'] for q in PX_PARAMS)
    return Outcome(float(err), {q: {k: _r(v, 5) for k, v in cmp_[q].items()} for q in PX_PARAMS},
                   ok=all(math.isfinite(cmp_[q]['error_sigma']) for q in PX_PARAMS))


def validation_checks() -> list:
    """Spec-compatible dicts for vantage.validation (the lead registers them; model 'calib')."""
    return [
        {'id': 'thermal_intrinsics.press_closed_form', 'model': 'calib',
         'title': 'Leave-one-out errors of a thermal curve in closed form',
         'quantity': 'largest difference between the leave-one-out residuals from the hat matrix (e/(1−h)) and from '
                     'refitting without each point, linear and quadratic curves of 4, 6 and 9 temperatures',
         'reference': 'brute-force refits (numpy lstsq), an independent computation', 'tolerance': 1e-7,
         'unit': 'px', 'refs': ('hastie2009',), 'fn': _check_press, 'cost': 'cheap'},
        {'id': 'thermal_intrinsics.interval_coverage', 'model': 'calib',
         'title': 'Prediction intervals of a thermal curve cover a new calibration as often as stated',
         'quantity': 'difference between the share of new calibrations (at random temperatures, Gaussian scatter) '
                     'inside the 95 % prediction interval of a 6-temperature linear curve and 0.95',
         'reference': 'Monte Carlo with known truth (2000 trials; the band is 2.6 binomial σ)', 'tolerance': 0.02,
         'unit': 'share', 'refs': ('hastie2009',), 'fn': _check_coverage, 'cost': 'cheap'},
        {'id': 'thermal_intrinsics.drift_recovery', 'model': 'calib',
         'title': 'A simulated temperature sweep\'s drift is recovered by calibrating the rendered boards',
         'quantity': 'worst error of the fitted drift per K of fx, fy, cx, cy against the sweep\'s true drift, in '
                     'the fit\'s own standard errors (polycarbonate holder, N-BK7 lens, 6 temperatures −20 … 80 °C, '
                     'each calibrated by calib.calibrate on 14 rendered checkerboard views)',
         'reference': 'the true intrinsics the views were rendered with (the material model: SCHOTT, Apollo '
                      'Optical Systems, Jamieson 1981)', 'tolerance': 3.0, 'unit': 'σ',
         'refs': ('schott_glass', 'jamieson1981', 'apollo_polymers'), 'fn': _check_recovery, 'cost': 'study'},
    ]


VALIDATION_REFERENCES = dict(REFERENCES, hastie2009='T. Hastie, R. Tibshirani, J. Friedman, The Elements of '
                             'Statistical Learning, 2nd ed., Springer, 2009, §7.10 (cross-validation; the '
                             'leave-one-out shortcut for linear smoothers, eq. 7.52).')


def measured_comparison(f_mm: float = 4.0) -> list:
    """The model's camera-constant drift (µm per K) for a lens-to-sensor distance f_mm per holder material next to
    Elias et al. 2020's smartphone measurements (µm per K of battery temperature, our division of their Table 2)."""
    rows = [{'what': f'model: {v["label"]} holder, {f_mm:g} mm lens to sensor', 'um_per_k': v['alpha'] * 1e-3 * f_mm}
            for v in HOUSING_MATERIALS.values()]
    rows += [{'what': f'Elias et al. 2020: {name}', 'um_per_k': dc * 1000.0 / dt,
              'pp_um_per_k': math.hypot(dx, dy) * MEASURED_PIXEL_UM / dt} for name, dt, dc, _, dx, dy in
             MEASURED_REFERENCE]
    return rows
