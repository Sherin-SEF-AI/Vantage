"""Checks of the environment models: weather (extinction, rain and fog attenuation, the lidar and radar range
equations, headlamps, the camera's photon budget) and the sun's position."""
from __future__ import annotations

import math

import numpy as np

from .suite import Outcome, check, figure, rel_err, series

# Recommendation ITU-R P.838-3, Table 5 (as printed): GHz -> (kH, αH, kV, αV)
P838_TABLE5 = {
    '1': ('0.0000259', '0.9691', '0.0000308', '0.8592'), '5': ('0.0002162', '1.6969', '0.0002428', '1.5317'),
    '10': ('0.01217', '1.2571', '0.01129', '1.2156'), '20': ('0.09164', '1.0568', '0.09611', '0.9847'),
    '24': ('0.1425', '1.0101', '0.1404', '0.9561'), '35': ('0.3374', '0.9047', '0.3224', '0.8761'),
    '60': ('0.8606', '0.7656', '0.8515', '0.7486'), '76': ('1.1185', '0.7199', '1.1139', '0.7091'),
    '77': ('1.1320', '0.7177', '1.1276', '0.7073'), '79': ('1.1579', '0.7135', '1.1541', '0.7038'),
    '80': ('1.1704', '0.7115', '1.1668', '0.7021'), '94': ('1.3179', '0.6888', '1.3175', '0.6828'),
    '100': ('1.3671', '0.6815', '1.3680', '0.6765'), '150': ('1.5823', '0.6494', '1.5896', '0.6466'),
    '300': ('1.6286', '0.6296', '1.6286', '0.6262'), '1000': ('1.3795', '0.6396', '1.3822', '0.6365'),
}
# Recommendation ITU-R P.1814, Table 2 (rain, France) and Table 3 (snow; λ in nm): as printed
P1814_RAIN = (1.076, 0.67)
P1814_SNOW = {'wet': (0.000102, 3.79, 0.72), 'dry': (0.0000542, 5.50, 1.38)}


def _half_unit(text: str) -> float:
    """Half a unit in the last printed digit of a table value."""
    d = len(text.split('.')[1]) if '.' in text else 0
    return 0.5 * 10.0 ** -d


# ============================================================================ weather
@check('weather.p838_table', 'weather', 'Rain attenuation coefficients against ITU-R P.838-3',
       'largest difference of weather.p838\'s k_H, α_H, k_V and α_V (the recommendation\'s curve fits, eqs. 2–3 and '
       'Tables 1–4) from its Table 5 at 16 frequencies from 1 to 1000 GHz, in units of half the last printed digit',
       'Recommendation ITU-R P.838-3, Table 5, as printed', 1.0, 'half last digits', refs=('itu_p838',))
def _p838(ctx):
    from ..core import weather as W
    worst, rows = 0.0, {}
    keys = ('kH', 'aH', 'kV', 'aV')
    for f, vals in P838_TABLE5.items():
        q = W.p838(float(f))
        errs = [abs(q[k] - float(v)) / _half_unit(v) for k, v in zip(keys, vals)]
        rows[f + ' GHz'] = {k: round(q[k], 6) for k in keys}
        worst = max(worst, max(errs))
    k = (1.1320 + 1.1276) / 2
    a = (1.1320 * 0.7177 + 1.1276 * 0.7073) / (2 * k)
    gam = float(W.rain_radar_db_km(25.0, 77.0))
    ok = abs(gam - k * 25.0 ** a) / (k * 25.0 ** a) < 1e-3          # eqs. 4–5 at τ = 45° from the table's values
    return Outcome(worst, {'model': rows, 'rain_77ghz_25mm_h_db_km': gam}, ok=ok)


@check('weather.p840_water', 'weather', 'Fog attenuation at radar frequencies (ITU-R P.840)',
       'largest relative difference of weather.liquid_water_coefficient (P.840\'s K_l from its double-Debye '
       'permittivity) from the Rayleigh absorption of small water drops computed from the complex permittivity '
       'directly, (6π/λ)·Im(−(ε−1)/(ε+2)), at 24, 77 and 100 GHz and 0–20 °C',
       'Rayleigh absorption by drops much smaller than the wavelength, from the same Debye permittivity, written out '
       'in the check', 2e-3, 'relative', refs=('itu_p840', 'kunkel1984'))
def _p840(ctx):
    from ..core import weather as W
    worst, rows = 0.0, {}
    for f, T in ((24.0, 0.0), (77.0, 10.0), (100.0, 20.0), (77.0, 0.0)):
        th = 300.0 / (T + 273.15)
        e0 = 77.66 + 103.3 * (th - 1)
        e1, e2 = 0.0671 * e0, 3.52
        fp = 20.20 - 146.0 * (th - 1) + 316.0 * (th - 1) ** 2
        fs = 39.8 * fp
        eps = (e0 - e1) / (1 - 1j * f / fp) + (e1 - e2) / (1 - 1j * f / fs) + e2
        K = (eps - 1) / (eps + 2)
        lam = 0.299792458 / f
        alpha = 6 * math.pi / lam * K.imag / 1e6 * 1000 * W.DB_PER_NEPER            # dB/km per g/m³
        got = W.liquid_water_coefficient(f, T)
        rows[f'{f:g} GHz {T:g} °C'] = {'model': got, 'rayleigh': alpha}
        worst = max(worst, abs(got - alpha) / alpha)
    fog = W.PRESET['fog']
    kunkel = abs(144.7 * W.fog_lwc(fog) ** 0.88 - W.MOR_LN / fog.visibility_m * 1000.0)
    return Outcome(worst, {'cases': rows, 'kunkel_round_trip_per_km': kunkel}, ok=kunkel < 1e-9)


@check('weather.p1814_coefficients', 'weather', 'Optical rain and snow attenuation against ITU-R P.1814',
       'largest relative difference of the optical rain attenuation (1–100 mm/h) and the wet and dry snow attenuation '
       '(0.5–10 mm/h at 550, 905 and 1550 nm) from P.1814\'s Table 2 (France: k 1.076, α 0.67) and Table 3 as '
       'printed',
       'Recommendation ITU-R P.1814, Tables 2 and 3', 5e-3, 'relative', refs=('itu_p1814',))
def _p1814(ctx):
    from ..core import weather as W
    R = np.array([1.0, 5.0, 25.0, 100.0])
    e = {'rain': float(rel_err(W.rain_optical_db_km(R), P1814_RAIN[0] * R ** P1814_RAIN[1]).max())}
    S = np.array([0.5, 2.0, 10.0])
    for kind, (a1, a0, b) in P1814_SNOW.items():
        for nm in (550.0, 905.0, 1550.0):
            got = W.snow_optical_db_km(S, nm, kind)
            e[f'{kind} snow {nm:g} nm'] = float(rel_err(got, (a1 * nm + a0) * S ** b).max())
    return Outcome(max(e.values()), e, note='weather.py uses the snow constants as P.1814 prints them (5.42e-5·λ + '
                                            '5.50, 1.02e-4·λ + 3.79); until 0.2.0 it carried 5.4958776 and '
                                            '1.023e-4·λ + 3.7855466, 0.1 % off them.')


@check('weather.koschmieder', 'weather', 'Contrast extinction (Koschmieder) and the meteorological optical range',
       'largest relative difference of the visible extinction for fog of MOR 50 m–10 km from −ln 0.05 / MOR, of the '
       'distance at which contrast C0 falls to ε from ln(C0/ε)/σ (residual of C0·e^(−σd) = ε), and of the '
       'extinction away from 550 nm from Kim\'s (λ / 550 nm)^−q',
       'Koschmieder\'s law with WMO\'s 5 % contrast threshold; Kim et al.\'s q(V)', 1e-9, 'relative',
       refs=('koschmieder1924', 'wmo8', 'kim2001'))
def _koschmieder(ctx):
    from ..core import weather as W
    e = {}
    V = np.array([50.0, 150.0, 1000.0, 10000.0])
    got = np.array([W.visible_extinction(W.Condition('f', 'f', visibility_m=v)) for v in V])
    e['extinction'] = float(rel_err(got, -math.log(0.05) / V).max())
    res = []
    for c0, eps, sig in ((0.5, 0.05, 0.02), (0.3, 0.02, 0.001), (0.9, 0.1, 0.0035)):
        d = W.contrast_range(c0, eps, sig)
        res.append(abs(c0 * math.exp(-sig * d) - eps) / eps)
    e['contrast range'] = max(res)
    kim = {60: 1.6, 20: 1.3, 3: 0.16 * 3 + 0.34, 0.8: 0.3, 0.2: 0.0}
    e['Kim q'] = max(abs(W.kim_q(v) - q) for v, q in kim.items())
    haze = W.Condition('h', 'h', visibility_m=10000.0)
    e['905 nm in haze'] = abs(W.optical_extinction(haze, 905.0) / (W.visible_extinction(haze) * (905 / 550) ** -1.3)
                              - 1)
    sig = -math.log(0.05) / 150.0
    d = np.linspace(0, 300, 61)
    fig = figure('Contrast of a black object in fog of 150 m MOR', 'distance, m', 'apparent contrast',
                 [series('C0·exp(−σd), σ = −ln 0.05 / 150 m', d, np.exp(-sig * d))], hline=0.05,
                 hline_label='5 % (WMO)')
    return Outcome(max(e.values()), e, fig)


@check('weather.range_equations', 'weather', 'Lidar and radar range in extinction (Lambert W)',
       'largest relative difference of weather.lidar_range (R·e^(αR) = R_c, solved with Lambert\'s W) and '
       'weather.radar_range (R⁴·e^(2aR) = R_c⁴) from a bracketing root finder on the same equations',
       'scipy.optimize.brentq on the defining equations', 1e-9, 'relative', refs=('corless1996', 'skolnik2001'))
def _range_eq(ctx):
    from scipy.optimize import brentq
    from ..core import weather as W
    e = []
    for a, rc in ((0.02, 90.0), (0.002, 200.0), (1e-5, 50.0), (0.05, 120.0)):
        r = W.lidar_range(rc, a)
        ref = brentq(lambda x: x * math.exp(a * x) - rc, 1e-9, rc, xtol=1e-14, rtol=1e-15)
        e.append(abs(r - ref) / ref)
    for db, r0 in ((11.2, 250.0), (3.0, 100.0), (0.4, 160.0)):
        a = db / W.DB_PER_NEPER / 1000
        r = W.radar_range(r0, db)
        ref = brentq(lambda x: x ** 4 * math.exp(2 * a * x) - r0 ** 4, 1e-6, r0, xtol=1e-14, rtol=1e-15)
        e.append(abs(r - ref) / ref)
    rc = np.linspace(5, 200, 40)
    fig = figure('Lidar range in fog (λ 905 nm) against the clear-air range', 'clear-air range, m', 'range, m',
                 [series(f'MOR {v:g} m', rc, W.lidar_range(rc, W.optical_extinction(W.Condition('f', 'f',
                                                                                              visibility_m=v), 905.0)))
                  for v in (500.0, 150.0, 60.0)] + [series('clear', rc, rc)])
    return Outcome(max(e), {'cases': len(e)}, fig)


# UN R112 class B passing beam and driving beam (per headlamp, cd; h degrees left +, v up +, right-hand traffic)
R112_MIN = {'75R': (-1.15, -0.57, 10100), '50R': (-1.72, -0.86, 10100), '50V': (0.0, -0.86, 5100),
            '25L': (9.0, -1.72, 1700), '25R': (-9.0, -1.72, 1700)}
R112_MAX = {'50L': (3.43, -0.86, 13200), '75L': (3.43, -0.57, 10600), 'B50L': (3.43, 0.57, 350),
            'BR': (-2.5, 1.0, 1750)}
R112_HIGH_MIN = {'2.5° L': (2.5, 0.0, 20300), '2.5° R': (-2.5, 0.0, 20300), '5° L': (5.0, 0.0, 5100),
                 '5° R': (-5.0, 0.0, 5100)}


@check('weather.headlamps_r112', 'weather', 'The headlamp pattern meets UN R112',
       'largest shortfall, as a share of the requirement, of the typical low-beam pattern\'s intensity at R112\'s '
       'class B test points (minimums 75R, 50R, 50V, 25L, 25R and zone IV; maximums 50L, 75L, B50L, BR and zone I '
       'below twice the 50R/75R), and of the high beam (I_max 40 500–215 000 cd, ±2.5° and ±5°)',
       'UN Regulation No. 112, class B photometric requirements', 0.0, 'share of the requirement',
       refs=('unr112',))
def _r112(ctx):
    from ..core import weather as W

    def I(h, v, beam='low'):
        return float(W.headlamp_cd(h, v, beam))
    short = {}
    for name, (h, v, lo) in R112_MIN.items():
        short[name] = max(0.0, 1 - I(h, v) / lo)
    for name, (h, v, hi) in R112_MAX.items():
        short[name] = max(0.0, I(h, v) / hi - 1)
    zone4 = min(I(h, v) for h in np.linspace(-5.15, 5.15, 9) for v in (-0.86, -1.3, -1.72))
    short['zone IV ≥ 2500'] = max(0.0, 1 - zone4 / 2500)
    zone1 = max(I(h, v) for h in np.linspace(-9, 9, 13) for v in np.linspace(-4, -1.72, 5))
    cap = 2 * min(I(-1.72, -0.86), I(-1.15, -0.57))
    short['zone I < 2 × 50R'] = max(0.0, zone1 / cap - 1)
    imax = max(I(h, v, 'high') for h in np.linspace(-1, 1, 9) for v in np.linspace(-1, 1, 9))
    short['high beam I_max'] = max(0.0, 1 - imax / 40500, imax / 215000 - 1)
    for name, (h, v, lo) in R112_HIGH_MIN.items():
        short[f'high {name}'] = max(0.0, 1 - I(h, v, 'high') / lo)
    hs = np.linspace(-15, 15, 121)
    fig = figure('Low-beam intensity along the horizontal through 50R (0.86° down)', 'h, ° (left +)',
                 'intensity per lamp, cd', [series('low beam, 0.86° down', hs, [I(h, -0.86) for h in hs]),
                                           series('low beam, 0.57° up', hs, [I(h, 0.57) for h in hs])])
    return Outcome(max(short.values()), {'shortfall': short, 'points_cd': {
        n: round(I(h, v), 1) for n, (h, v, _) in {**R112_MIN, **R112_MAX}.items()}}, fig)


@check('weather.photon_budget', 'weather', 'The camera\'s minimum luminance from its photon budget',
       'relative difference of the electrons per cd/m² weather.camera_floor gives from π·T/(4N²)·t·Φ·A·QE, and of '
       'the signal-to-noise ratio at its L_min from snr_min (S / √(S + read noise²)), STURDeCAM31 with H05R2',
       'the photon budget of a pixel behind a lens of f-number N (Janesick), written out in the check', 1e-9,
       'relative', refs=('janesick2007',))
def _photon(ctx):
    from ..core import weather as W
    from ..core.lens import builtin_lenses, builtin_modules
    st = W.WeatherSettings()
    m, ln = builtin_modules()[0], builtin_lenses()[0]
    fl = W.camera_floor(m, ln, st)
    N = ln.f_number if ln.f_number > 0 else 2.0
    t = min(st.max_exposure_ms / 1000, 1 / m.fps)
    k = math.pi * st.transmission / (4 * N * N) * t * W.PHOTONS_PER_LUX_S * (m.pixel_um * 1e-6) ** 2 * st.qe
    S = fl['l_min'] * k
    e = {'electrons per cd/m²': abs(fl['electrons_per_cd_m2'] - k) / k,
         'SNR at L_min': abs(S / math.sqrt(S + st.read_noise_e ** 2) - st.snr_min) / st.snr_min}
    return Outcome(max(e.values()), dict(e, l_min_cd_m2=fl['l_min']))


@check('weather.illuminance', 'weather', 'Headlamp illuminance: inverse square, cosine and extinction',
       'largest relative difference of weather.illuminance at a point 20 m ahead from I/d², from I·cos(i)/d² on a '
       'surface tilted 60° away, and from E_ambient + I·e^(−σd)/d² in fog',
       'the photometric inverse-square law with Lambert\'s cosine and Beer–Lambert extinction', 1e-6, 'relative',
       refs=('unr112', 'koschmieder1924'))
def _illuminance(ctx):
    from ..core import weather as W
    st = W.WeatherSettings()
    L = np.array([[0.0, 0.0, 0.65]])
    night = W.Condition('n', 'n', 'night', ambient_lux=0.0)
    P = np.array([[20.0, 0.0, 0.65 - 20 * math.tan(math.radians(1.0))]])
    d = float(np.linalg.norm(P - L))
    I = float(W.headlamp_cd(0.0, -1.0))
    e = [abs(float(W.illuminance(P, night, st, L)[0]) - I / d ** 2) / (I / d ** 2)]
    n = np.array([[-math.cos(math.radians(60)), 0.0, math.sin(math.radians(60))]])
    u = (P - L) / d
    cos = float(-(n @ u.T)[0, 0])
    e.append(abs(float(W.illuminance(P, night, st, L, normals=n)[0]) - I / d ** 2 * cos) / (I / d ** 2 * cos))
    fog = W.Condition('n', 'n', 'night', ambient_lux=2.0, visibility_m=100.0)
    want = 2.0 + I / d ** 2 * math.exp(-W.visible_extinction(fog) * d)
    e.append(abs(float(W.illuminance(P, fog, st, L)[0]) - want) / want)
    return Outcome(max(e), {'lux_clear': I / d ** 2, 'lux_fog': want})


# ============================================================================ sun
@check('sun.spa_example', 'sun', 'Sun position against the SPA paper\'s worked example',
       'largest difference of the topocentric zenith angle and azimuth exposure.sun_at gives for Golden, Colorado, '
       '2003-10-17 12:30:30 local time (UTC−7), from Reda & Andreas\' 50.11162° and 194.34024°',
       'I. Reda and A. Andreas, NREL Solar Position Algorithm, worked example', 0.01, 'degrees',
       refs=('reda2004', 'noaa'))
def _spa(ctx):
    from ..core import exposure as X
    az, el = X.sun_at('2003-10-17T19:30:30', 39.742476, -105.1786)
    e = {'zenith': abs(90.0 - el - 50.11162), 'azimuth': abs((az - 194.34024 + 180) % 360 - 180)}
    return Outcome(max(e.values()), dict(e, zenith_deg=90.0 - el, azimuth_deg=az))


@check('sun.meeus_examples', 'sun', 'Solar coordinates against Meeus\'s worked examples',
       'largest difference of the sun\'s declination, right ascension and apparent longitude (degrees), distance '
       '(AU × 100) and equation of time (minutes / 10) at 1992 October 13.0 from Meeus\'s examples 25.a and 28.b; '
       'and of the noon elevation 90° − |φ − δ| (no refraction) at two latitudes',
       'J. Meeus, Astronomical Algorithms, examples 25.a and 28.b; spherical astronomy at solar noon', 2e-3,
       'degrees', refs=('meeus1998', 'noaa'))
def _meeus(ctx):
    from ..core import exposure as X
    eph = X.solar_ephemeris(X.julian_day('1992-10-13T00:00'))
    e = {'declination': abs(float(eph['declination']) + 7.78507),
         'right ascension': abs(float(eph['right_ascension']) - 198.38083),
         'apparent longitude': abs(float(eph['apparent_longitude']) - 199.90895),
         'distance ×100': abs(float(eph['distance_au']) - 0.99766) * 100,
         'equation of time /10': abs(float(eph['equation_of_time_min']) - (13 + 42.6 / 60)) / 10}
    E = 13 + 42.6 / 60
    for lat in (-36.85, 52.0):
        az, el = X.sun_at('1992-10-13T00:00', lat, (720 - E) / 4, refract=False)
        e[f'noon elevation at {lat:g}°'] = abs(el - (90 - abs(lat + 7.78507)))
    return Outcome(max(e.values()), e)
