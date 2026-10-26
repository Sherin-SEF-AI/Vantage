"""Weather (core.weather): the published models against their sources and independent derivations, the headlamp
pattern against UN R112, clear = the studies exactly as before, the perception range under fog and at night against
hand computations, the sample dataset's simulators, the comparison study and its export, and the perception page
driven like a user (offscreen)."""
import json
import math
import os

import numpy as np
import pytest
from scipy.optimize import brentq

from vantage.core import perception as PR, samples, weather as W
from vantage.core.analysis import Engine


@pytest.fixture(scope='module')
def sample():
    p = samples.tigor_sample_project()
    e = Engine()
    return p, e, e.analyse(p)


@pytest.fixture(scope='module')
def rig():
    p = samples.tigor_sample_project()
    p.active_uid = next(v for v in p.variants if v.sensors).uid
    e = Engine()
    e.vehicle_geom(p)
    return p, e


# ----------------------------------------------------------------------------- settings
def test_settings_are_tolerant_and_round_trip():
    st = W.WeatherSettings.from_dict({
        'condition': 'moon', 'params': {'fog': {'visibility_m': 5, 'rain_mm_h': 'x', 'beam': 'dazzle'},
                                        'clear': {'visibility_m': 50}, 'bogus': {'rain_mm_h': 3},
                                        'night': {'ambient_lux': -4, 'beam': 'high', 'light': 'night'},
                                        'snow': 'junk'},
        'compare': ['fog', 'fog', 'nope', 'clear'], 'objects': {'Car': {'contrast': 7, 'reflectance': 'a'}, 'X': 3},
        'qe': 9, 'snr_min': float('nan'), 'traffic': 'up', 'high_beam_cd': 1e9, 'temp_c': True})
    assert st.condition == 'clear' and st.qe == 1.0 and st.snr_min == 5.0 and st.traffic == 'right'
    assert st.high_beam_cd == 215000.0 and st.temp_c == 10.0
    assert st.params == {'fog': {'visibility_m': 20.0}, 'night': {'ambient_lux': 0.0, 'beam': 'high'}}
    assert st.compare == ['clear', 'fog'] and st.objects == {'Car': {'contrast': 1.0}}
    assert st.condition_for('fog').visibility_m == 20.0 and st.condition_for('clear') == W.PRESET['clear']
    assert W.WeatherSettings.from_dict(st.to_dict()) == st
    assert W.WeatherSettings.from_dict(None) == W.WeatherSettings() == W.WeatherSettings.from_dict('junk')
    # visibility 0 means no fog; an edit equal to the preset is not kept
    st = W.WeatherSettings.from_dict({'params': {'fog': {'visibility_m': 0}, 'night': {'ambient_lux': 0.1}}})
    assert st.condition_for('fog').visibility_m == 0.0 and 'night' not in st.params
    assert [c.id for c in st.conditions()] == list(W.IDS) and W.IDS[0] == 'clear'
    assert W.PRESET['fog'].label() == 'Fog · 150 m' and W.PRESET['clear'].label() == 'Clear'


# ----------------------------------------------------------------------------- the models against their sources
def test_p838_reproduces_its_table_5():
    """ITU-R P.838-3 Table 5 (the recommendation's own tabulated coefficients) at four frequencies."""
    table = {10: (0.01217, 1.2571, 0.01129, 1.2156), 24: (0.1425, 1.0101, 0.1404, 0.9561),
             77: (1.1320, 0.7177, 1.1276, 0.7073), 80: (1.1704, 0.7115, 1.1668, 0.7021)}
    for f, (kh, ah, kv, av) in table.items():
        q = W.p838(f)
        assert q['kH'] == pytest.approx(kh, rel=5e-4) and q['kV'] == pytest.approx(kv, rel=5e-4)
        assert q['aH'] == pytest.approx(ah, abs=2e-4) and q['aV'] == pytest.approx(av, abs=2e-4)
    # γ = k R^α with eq. 4–5 at τ = 45°, on a horizontal path, from the table's values
    k = (1.1320 + 1.1276) / 2
    a = (1.1320 * 0.7177 + 1.1276 * 0.7073) / (2 * k)
    assert float(W.rain_radar_db_km(25.0, 77.0)) == pytest.approx(k * 25.0 ** a, rel=1e-3)
    assert 9.0 < float(W.rain_radar_db_km(25.0, 77.0)) < 13.0
    assert float(W.rain_radar_db_km(0.0, 77.0)) == 0.0


def test_p840_coefficient_is_the_rayleigh_absorption_of_its_permittivity():
    """K_l = 0.819 f / (ε''(1 + η²)) is Rayleigh absorption by small drops: (6π/λ)·Im(−(ε−1)/(ε+2)) per unit water
    density, in dB/km per g/m³. Computed here from the complex permittivity directly."""
    for f, T in ((24.0, 0.0), (77.0, 10.0), (100.0, 20.0)):
        th = 300.0 / (T + 273.15)
        e0 = 77.66 + 103.3 * (th - 1)
        e1, e2 = 0.0671 * e0, 3.52
        fp = 20.20 - 146.0 * (th - 1) + 316.0 * (th - 1) ** 2
        fs = 39.8 * fp
        eps = (e0 - e1) / (1 - 1j * f / fp) + (e1 - e2) / (1 - 1j * f / fs) + e2     # Debye, ε' + jε''
        K = (eps - 1) / (eps + 2)
        lam = 0.299792458 / f                                                          # m
        alpha = 6 * math.pi / lam * K.imag / 1e6                                       # Np·m⁻¹ per g/m³
        assert W.liquid_water_coefficient(f, T) == pytest.approx(alpha * 1000 * W.DB_PER_NEPER, rel=2e-3)
    assert W.liquid_water_coefficient(77, 0) > W.liquid_water_coefficient(77, 20) > 0      # colder water absorbs more
    # Kunkel: σ = 144.7 LWC^0.88 km⁻¹ round trip, and dense fog is a fraction of a dB/km at 77 GHz
    fog = W.PRESET['fog']
    assert 144.7 * W.fog_lwc(fog) ** 0.88 == pytest.approx(W.MOR_LN / fog.visibility_m * 1000.0)
    assert 0.05 < W.radar_db_km(fog) < 1.0 and W.radar_db_km(W.PRESET['clear']) == 0.0


def test_optical_extinction_rain_snow_fog_and_kim():
    assert float(W.rain_optical_db_km(25.0)) == pytest.approx(1.076 * 25 ** 0.67)
    assert float(W.snow_optical_db_km(2.0, 905.0)) == pytest.approx((5.42e-5 * 905 + 5.4958776) * 2 ** 1.38)
    assert float(W.snow_optical_db_km(2.0, 905.0, 'wet')) == pytest.approx((1.023e-4 * 905 + 3.7855466) * 2 ** 0.72)
    assert [W.kim_q(v) for v in (60, 20, 3, 0.8, 0.2)] == pytest.approx([1.6, 1.3, 0.16 * 3 + 0.34, 0.3, 0.0])
    fog = W.Condition('f', 'f', visibility_m=150.0)
    assert W.visible_extinction(fog) == pytest.approx(-math.log(0.05) / 150.0)   # Koschmieder with WMO's 5 %
    assert W.optical_extinction(fog, 1550.0) == pytest.approx(W.visible_extinction(fog))   # dense fog: grey
    haze = W.Condition('h', 'h', visibility_m=10000.0)
    assert W.optical_extinction(haze, 905.0) == pytest.approx(W.visible_extinction(haze) * (905 / 550) ** -1.3)
    rain = W.Condition('r', 'r', rain_mm_h=10.0)
    assert W.visible_extinction(rain) == pytest.approx(1.076 * 10 ** 0.67 / W.DB_PER_NEPER / 1000)
    assert W.visible_extinction(W.PRESET['clear']) == 0.0


def test_contrast_range_is_where_koschmieder_reaches_the_threshold():
    for c0, eps, sig in ((0.5, 0.05, 0.02), (0.3, 0.02, 0.001)):
        d = W.contrast_range(c0, eps, sig)
        assert c0 * math.exp(-sig * d) == pytest.approx(eps)
    assert W.contrast_range(0.3, 0.05, 0.0) == math.inf and W.contrast_range(0.04, 0.05, 0.01) == 0.0


def test_lidar_and_radar_range_equations_solved():
    for a, rc in ((0.02, 90.0), (0.002, 200.0), (1e-5, 50.0)):
        r = W.lidar_range(rc, a)
        assert r == pytest.approx(brentq(lambda x: x * math.exp(a * x) - rc, 1e-9, rc), rel=1e-9)
    np.testing.assert_allclose(W.lidar_range(np.array([10.0, 90.0]), 0.0), [10.0, 90.0])
    for db, r0 in ((11.2, 250.0), (3.0, 100.0)):
        a = db / W.DB_PER_NEPER / 1000
        r = W.radar_range(r0, db)
        assert r == pytest.approx(brentq(lambda x: x ** 4 * math.exp(2 * a * x) - r0 ** 4, 1e-6, r0), rel=1e-9)
        # the two-way loss at the derated range equals what the 1/R⁴ law gains back
        assert 2 * db * r / 1000 == pytest.approx(40 * math.log10(r0 / r), rel=1e-6)
    assert W.radar_range(100.0, 0.0) == 100.0


def test_headlamp_pattern_meets_r112_class_b():
    """UN R112 class B test points (cd, per headlamp, right-hand traffic): h positive left, v positive up."""
    I = lambda h, v, beam='low': float(W.headlamp_cd(h, v, beam))          # noqa: E731
    assert I(-1.15, -0.57) >= 10100 and I(-1.72, -0.86) >= 10100          # 75R, 50R
    assert I(0.0, -0.86) >= 5100                                          # 50V
    assert I(3.43, -0.86) <= 13200 and I(3.43, -0.57) <= 10600            # 50L, 75L
    assert I(3.43, 0.57) <= 350 and I(-2.5, 1.0) <= 1750                  # B50L (glare), BR
    assert I(9.0, -1.72) >= 1700 and I(-9.0, -1.72) >= 1700               # 25L, 25R
    zone4 = [I(h, v) for h in np.linspace(-5.15, 5.15, 9) for v in (-0.86, -1.3, -1.72)]
    assert min(zone4) >= 2500
    zone1 = max(I(h, v) for h in np.linspace(-9, 9, 13) for v in np.linspace(-4, -1.72, 5))
    assert zone1 < 2 * min(I(-1.72, -0.86), I(-1.15, -0.57))
    hb = [I(h, v, 'high') for h in np.linspace(-1, 1, 5) for v in np.linspace(-1, 1, 5)]
    assert 40500 <= max(hb) <= 215000                                     # Imax
    assert I(-2.5, 0.0, 'high') >= 20300 and I(2.5, 0.0, 'high') >= 20300 and I(5.0, 0.0, 'high') >= 5100
    # left-hand traffic mirrors the pattern; nothing behind the lamp
    assert float(W.headlamp_cd(1.72, -0.86, traffic='left')) == pytest.approx(I(-1.72, -0.86))
    assert I(120.0, -2.0) == 0.0
    # the beam reaches typical figures: a low beam's 1 lx line some tens of metres, a high beam's a few hundred
    st = W.WeatherSettings()
    lamps = W.lamp_positions(None, st)
    low = W.beam_reach(W.PRESET['night'], st, lamps, 1.0, 0.5)
    high = W.beam_reach(W.Condition('n', 'n', 'night', beam='high'), st, lamps, 1.0, 0.5)
    assert 40 < low < 120 and 200 < high < 500


def test_illuminance_is_inverse_square_with_the_cosine_and_extinction():
    st = W.WeatherSettings()
    L = np.array([[0.0, 0.0, 0.65]])
    c = W.Condition('n', 'n', 'night', ambient_lux=0.0)
    P = np.array([[20.0, 0.0, 0.65 - 20 * math.tan(math.radians(1.0))]])     # 1° down, straight ahead
    d = float(np.linalg.norm(P - L))
    I = float(W.headlamp_cd(0.0, -1.0))
    assert float(W.illuminance(P, c, st, L)[0]) == pytest.approx(I / d ** 2, rel=1e-6)
    n = np.array([[-math.cos(math.radians(60)), 0.0, math.sin(math.radians(60))]])      # tilted away by 60° ...
    u = (P - L) / d
    cos = float(-(n @ u.T)[0, 0])
    assert float(W.illuminance(P, c, st, L, normals=n)[0]) == pytest.approx(I / d ** 2 * cos, rel=1e-6)
    fog = W.Condition('n', 'n', 'night', ambient_lux=2.0, visibility_m=100.0)
    s = W.visible_extinction(fog)
    assert float(W.illuminance(P, fog, st, L)[0]) == pytest.approx(2.0 + I / d ** 2 * math.exp(-s * d), rel=1e-6)
    day = W.Condition('d', 'd', 'day', ambient_lux=5000.0)
    assert float(W.illuminance(P, day, st, L)[0]) == 5000.0             # lamps count at night only


def test_camera_floor_is_the_photon_budget():
    from vantage.core.lens import builtin_lenses, builtin_modules
    st = W.WeatherSettings()
    m, ln = builtin_modules()[0], builtin_lenses()[0]
    fl = W.camera_floor(m, ln, st)
    N = ln.f_number if ln.f_number > 0 else 2.0
    t = min(st.max_exposure_ms / 1000, 1 / m.fps)
    k = math.pi * st.transmission / (4 * N * N) * t * W.PHOTONS_PER_LUX_S * (m.pixel_um * 1e-6) ** 2 * st.qe
    assert fl['electrons_per_cd_m2'] == pytest.approx(k)
    S = fl['l_min'] * k                                                    # the signal at L_min
    assert S / math.sqrt(S + st.read_noise_e ** 2) == pytest.approx(st.snr_min)
    # twice the pixel pitch: a quarter of the luminance (once read noise is negligible); a slower lens: more
    big = W.camera_floor(type(m)(**{**m.to_dict(), 'pixel_um': 2 * m.pixel_um,
                                    'size_mm': tuple(m.size_mm), 'cables_m': tuple(m.cables_m),
                                    'temp_c': tuple(m.temp_c)}), ln, W.WeatherSettings(read_noise_e=0.0))
    ref = W.camera_floor(m, ln, W.WeatherSettings(read_noise_e=0.0))
    assert big['l_min'] == pytest.approx(ref['l_min'] / 4)
    assert 0.01 < fl['l_min'] < 3.0


# ----------------------------------------------------------------------------- the perception range
def test_clear_runs_exactly_the_geometry(sample):
    p, e, res = sample
    plain = PR.analyse(p, e, res)
    explicit = PR.analyse(p, e, res, weather='clear', compare=list(W.IDS))
    for k in ('far', 'near', 'best', 'cam_far', 'count'):
        a, b = getattr(plain, k), getattr(explicit, k)
        assert a.dtype == b.dtype and a.tobytes() == b.tobytes()
    assert json.dumps(plain.sectors) == json.dumps(explicit.sectors)
    # the comparison's clear row is the plain result
    row = next(r for r in explicit.weather['compare'] if r['id'] == 'clear')
    assert row['far'].tobytes() == plain.far.tobytes() and row['passed'] == plain.metrics['passed']
    # an untouched project exports as before: no condition block, the same summary keys
    assert 'condition' not in PR.to_doc(plain, p) and 'condition' not in PR.summary(plain)


def test_far_range_without_weather_failure_is_the_plain_far_range():
    rng = np.random.default_rng(3)
    A, R = 90, 60
    ranges = np.geomspace(1, 200, R)
    hpx = 900.0 / ranges[None, :] * rng.uniform(0.5, 1.5, (A, 1))
    vis = rng.random((A, R)) > 0.1
    ok = vis & (hpx >= 20.0)
    rows = np.arange(A)
    a = PR._far_range(ok, vis, hpx, ranges, 20.0, rows)
    b = PR._far_range_wx(ok, vis, hpx, ranges, 20.0, rows, np.full((A, R), np.inf))
    assert a.tobytes() == b.tobytes()
    # a margin falling linearly in range crosses zero exactly where it says
    m = 3.0 - ranges[None, :] / 10.0 * np.ones((A, 1))                    # 0 at 30 m
    okw = ok & (m >= 0)
    f = PR._far_range_wx(okw, vis, hpx, ranges, 20.0, rows, m)
    j = int(np.searchsorted(ranges, 30.0))                                # the samples either side of 30 m
    good = (a > ranges[j] + 1) & vis[:, j - 1] & vis[:, j]
    assert good.sum() > 10 and np.allclose(f[good], 30.0, atol=1e-9)
    assert (f[~good & okw.any(1)] <= 30.0 + 1e-9).all()


def test_fog_stops_a_camera_at_its_contrast_range(sample):
    """Every object needs 1 px (so the lens resolves cars beyond 250 m ahead in clear air); in fog of MOR V the far
    range ahead is where the distance from camera F to the car's middle reaches ln(C0/ε)/σ (hand geometry)."""
    p, e, res = sample
    q = p.snapshot()
    for c in q.variant.cameras:
        c.visible = c.tag == 'F'
    q.analysis.perception = {'classes': [{'name': 'Car', 'height': 1.5, 'width': 1.8, 'min_px': 1.0, 'base': 0}],
                             'max_range': 250.0}
    clear = PR.analyse(q, e, res)
    assert clear.at(0, 0.0)[0] > 240
    for V in (60.0, 100.0):
        fog = W.Condition('fog', 'Fog', visibility_m=V)
        pr = PR.analyse(q, e, res, weather=fog)
        cam = q.variant.by_tag('F')
        d = math.log(0.5 / 0.05) / (W.MOR_LN / V)                     # the car's contrast 0.5, threshold 0.05
        cx = pr.centre[0]
        dz = 0.75 - cam.z
        r = cam.x - cx + math.sqrt(d * d - dz * dz - cam.y ** 2)      # ground range from the centre, straight ahead
        assert pr.at(0, 0.0)[0] == pytest.approx(r, abs=0.05)
        assert PR.to_doc(pr, q)['condition']['visibility'] == pytest.approx(V)
        assert PR.summary(pr)['condition'] == fog.label()


def test_night_needs_light_and_the_threshold_holds_where_it_stops(sample):
    p, e, res = sample
    st = W.WeatherSettings()
    dark = W.Condition('night', 'Night', 'night', ambient_lux=0.0)
    pr = PR.analyse(p, e, res, weather=dark)
    k = [c.name for c in pr.classes].index('Pedestrian')
    rear = [s for s in pr.sectors if s['name'] == 'rear'][0]
    assert rear['classes']['Pedestrian']['min_m'] == 0.0                 # no lamps behind, no ambient light
    far = pr.at(k, 0.0)[0]
    assert 5.0 < far < PR.analyse(p, e, res).at(k, 0.0)[0]
    # at that range straight ahead the best camera's light margin is (about) zero: hand computation
    tag = pr.at(k, 0.0)[2]
    cam = p.variant.by_tag(tag)
    G = np.array([[pr.centre[0] + far, pr.centre[1]]])
    lamps = W.lamp_positions(e.vehicle_geom(p), st)
    lum = W.object_luminance(G, 0.0, 1.7, 0.2, dark, st, lamps)[0]
    fl = W.camera_floor(p.module_for(cam), p.lens_for(cam), st)
    assert math.log(lum / fl['l_min']) == pytest.approx(0.0, abs=0.08)
    # brighter surroundings and the high beam see further; enough ambient light is daylight for the camera
    lit = PR.analyse(p, e, res, weather=W.Condition('n', 'n', 'night', ambient_lux=1000.0))
    assert lit.far.tobytes() == PR.analyse(p, e, res).far.tobytes()
    high = PR.analyse(p, e, res, weather=W.Condition('n', 'n', 'night', ambient_lux=0.0, beam='high'))
    assert high.at(k, 0.0)[0] > far
    sign = [c.name for c in pr.classes].index('Traffic sign')
    assert pr.at(sign, 0.0)[0] > 0                                       # retroreflective


def test_the_comparison_study(rig, tmp_path):
    import cv2
    p, e = rig
    p.analysis.weather = {'condition': 'fog', 'params': {'heavy_rain': {'rain_mm_h': 40.0}}}
    try:
        res = W.run(p, e)
    finally:
        p.analysis.weather = {}
    assert [c.id for c in res.conditions] == list(W.IDS) and res.selected == 'fog'
    assert PR._condition(res.perception).id == 'fog'
    clear = res.compare_row('clear')
    assert clear['passed'] == PR.analyse(p, e).metrics['passed']
    night = res.compare_row('night')
    assert night['passed'] < clear['passed']
    lidar = next(r for r in res.sensors['fog'] if r['kind'] == 'lidar')
    a = W.optical_extinction(res.condition('fog'), 905.0)
    assert lidar['range_m'] * math.exp(a * lidar['range_m']) == pytest.approx(lidar['clear_range_m'])
    assert 0 < lidar['false_returns'] < 0.5
    radar = next(r for r in res.sensors['heavy_rain'] if r['tag'] == 'RD1')
    db = float(W.rain_radar_db_km(40.0, 77.0))
    assert radar['attenuation_db_km'] == pytest.approx(db) and radar['range_m'] < radar['clear_range_m']
    assert all(r['kind'] != 'radar' or r['range_m'] == r['clear_range_m'] for r in res.sensors['night'])
    drops = {r['tag']: r['lens_drops'] for r in res.sensors['heavy_rain'] if r['kind'] == 'camera'}
    assert max(drops.values()) > 0 and all(r['lens_drops'] == 0 for r in res.sensors['fog'] if r['kind'] == 'camera')
    assert any(f['condition'] == 'night' and f['severity'] == 'serious' for f in res.findings)
    json.dumps(W.summary(res))
    doc = W.to_doc(res, p)
    assert doc['schema'] == W.SCHEMA and doc['selected'] == 'fog' and len(doc['perception']) == len(W.IDS)
    assert doc['headlamps']['reach_1lx_m']['high'] > doc['headlamps']['reach_1lx_m']['low']
    json.dumps(doc)
    paths = W.export(res, p, str(tmp_path))
    assert [os.path.basename(x) for x in paths] == ['weather.json', 'weather_conditions.csv', 'weather_sensors.csv',
                                                     'weather.png']
    lines = (tmp_path / 'weather_conditions.csv').read_text().splitlines()
    assert lines[0].split(',') == list(W.CSV_CONDITIONS) and len(lines) == 1 + len(W.IDS) * 4
    assert cv2.imread(str(tmp_path / 'weather.png')) is not None
    assert W.run(p, e, cancel=lambda: True) is None


# ----------------------------------------------------------------------------- the sample dataset
def test_dataset_settings_keep_clear_out_of_the_tokens():
    from vantage.core import dataset as DSET
    st = DSET.DatasetSettings()
    assert 'weather' not in st.to_dict() and DSET.DatasetSettings.from_dict(st.to_dict()) == st
    st = DSET.DatasetSettings.from_dict({'weather': 'fog'})
    assert st.to_dict()['weather'] == 'fog' and DSET.DatasetSettings.from_dict(st.to_dict()) == st
    assert DSET.DatasetSettings.from_dict({'weather': 'hail'}).weather == ''
    p = samples.tigor_sample_project()
    assert W.scene_weather(p) is None                      # clear: the simulators run without weather
    p.analysis.weather = {'condition': 'snow'}
    assert W.scene_weather(p).condition.id == 'snow' and W.scene_weather(p, 'clear') is None


def _scene():
    from vantage.core import dataset as DSET
    return DSET.scene_for(samples.tigor_sample_project(), DSET.DatasetSettings(duration_s=2.0))


def test_simulators_reduce_to_clear_without_attenuation(rig):
    """A weather with nothing to attenuate gives the clear lidar sweep and radar scan bit for bit."""
    from vantage.core import drivescene as DS
    p, e = rig
    scene = _scene()
    neutral = W.SceneWeather(W.Condition('x', 'x'), W.WeatherSettings(), 0.0, 0.0, 0.0, 0.0,
                             W.lamp_positions(e.vehicle_geom(p), W.WeatherSettings()))
    lid = next(s for s in p.variant.sensors if s.kind == 'lidar')
    ls = DS.LidarSim(p, e, lid)
    a, b = ls.sweep(scene, 0.5, seed=[1, 2]), ls.sweep(scene, 0.5, seed=[1, 2], wx=neutral)
    for k in ('xyz', 'intensity', 'ring', 't', 'obj'):
        assert a[k].tobytes() == b[k].tobytes()
    rad = next(s for s in p.variant.sensors if s.tag == 'RD1')
    rs = DS.RadarSim(p, e, rad)
    a, b = rs.scan(scene, 0.5, seed=4), rs.scan(scene, 0.5, seed=4, wx=neutral)
    for k in a:
        assert a[k].tobytes() == b[k].tobytes()


def test_lidar_in_fog_and_radar_in_rain(rig):
    from vantage.core import drivescene as DS
    p, e = rig
    scene = _scene()
    p.analysis.weather = {'condition': 'fog', 'params': {'fog': {'visibility_m': 60.0}}}
    try:
        wx = W.scene_weather(p, None, e)
    finally:
        p.analysis.weather = {}
    lid = next(s for s in p.variant.sensors if s.kind == 'lidar')
    m = p.sensor_model_for(lid)
    ls = DS.LidarSim(p, e, lid)
    clear, fog = ls.sweep(scene, 0.5, seed=[1, 2]), ls.sweep(scene, 0.5, seed=[1, 2], wx=wx)
    real = fog['obj'] != DS.P_WEATHER
    r = np.linalg.norm(fog['xyz'][real], axis=1)
    rmax = W.lidar_range(m.range_m * math.sqrt(max(DS.REFL.values()) / 0.10), wx.sigma_lidar)
    assert r.max() <= rmax + 0.1 < np.linalg.norm(clear['xyz'], axis=1).max()
    false = (~real).sum()
    assert 0.5 * wx.false_share * ls.C * ls.J < false <= 1.05 * wx.false_share * ls.C * ls.J
    rf = np.linalg.norm(fog['xyz'][~real], axis=1)
    assert rf.min() >= m.range_min_m - 0.1 and rf.max() <= m.range_min_m + 2 * 6.0 + 0.1
    # radar: a strong attenuation keeps every detection inside the radar equation's derated reach
    rad = next(s for s in p.variant.sensors if s.tag == 'RD1')
    rm = p.sensor_model_for(rad)
    rs = DS.RadarSim(p, e, rad)
    from dataclasses import replace
    wet = replace(wx, radar_db_km=60.0)
    det_c, det_w = rs.scan(scene, 0.5, seed=4, clutter=0), rs.scan(scene, 0.5, seed=4, clutter=0, wx=wet)
    reach = W.radar_range(rm.max_range() * DS.SN.radar_scale(max(DS.TARGET_RCS.values())), 60.0)
    rng_w = np.sqrt(det_w['x'] ** 2 + det_w['y'] ** 2 + det_w['z'] ** 2)
    assert len(det_w['x']) < len(det_c['x']) and (rng_w <= reach + 1.0).all()


def test_rendered_conditions_look_like_them(rig):
    from vantage.core import drivescene as DS
    p, e = rig
    scene = _scene()
    cam = p.variant.by_tag('F')
    side = p.variant.by_tag('ML')
    out = {}
    for cid in ('clear', 'night', 'fog', 'heavy_rain', 'snow'):
        p.analysis.weather = {'condition': cid, 'params': {'fog': {'visibility_m': 40.0}}}
        try:
            wx = W.scene_weather(p, None, e)
        finally:
            p.analysis.weather = {}
        for c in (cam, side):
            sim = DS.CameraSim(p, e, c, 320, 256)
            img, ids = sim.render(scene, 1.0, noise=0.0, seed=[1, 2, 3], wx=wx, want_ids=True)
            out[cid, c.tag] = img.astype(float)
            out['ids', c.tag] = ids
    # night: the front camera sees its headlamp pool, the side camera little but the ambient light
    bright = lambda im: (im.mean(-1) > 120).mean()                       # noqa: E731
    assert out['night', 'F'].mean() < out['clear', 'F'].mean() and bright(out['night', 'F']) > 0.05
    med = lambda im: float(np.median(im.mean(-1)))                       # noqa: E731
    assert med(out['night', 'ML']) < 0.3 * med(out['clear', 'ML']) and bright(out['night', 'ML']) < 0.1
    # fog: what the camera sees of the scene's objects drifts toward the airlight (exp(−σd) of its own colour left)
    air = np.array([0.80, 0.81, 0.83]) * 0.92 * 255
    obj = out['ids', 'ML'] >= 0
    gap = lambda im: float(np.abs(im[obj] - air).mean())                  # noqa: E731
    assert obj.mean() > 0.02 and gap(out['fog', 'ML']) < 0.7 * gap(out['clear', 'ML'])
    # rain: a wet, darker road and drops on the lens (which differ from the clear frame); snow: bright flakes
    assert out['heavy_rain', 'F'].mean() < out['clear', 'F'].mean()
    assert (np.abs(out['snow', 'F'] - out['clear', 'F']).max(-1) > 60).mean() > 0.001
    for k, v in out.items():
        assert k[0] == 'ids' or (np.isfinite(v).all() and v.min() >= 0 and v.max() <= 255), k


def test_dataset_generation_names_the_weather(rig, tmp_path):
    from vantage.core import dataset as DSET
    p, e = rig
    p.analysis.weather = {'condition': 'night'}
    try:
        res = DSET.generate(p, e, {'duration_s': 0.5, 'image_scale': 0.08, 'rosbag': False}, str(tmp_path))
    finally:
        p.analysis.weather = {}
    doc = json.loads((tmp_path / 'dataset.json').read_text())
    assert doc['scene']['weather']['id'] == 'night' and 'Weather: ' in doc['assumptions']
    assert res.weather['id'] == 'night' and any('Weather: Night' in n for n in res.notes)
    assert 'weather: Night, unlit road' in (tmp_path / 'README.md').read_text()


# ----------------------------------------------------------------------------- the page, driven like a user
from test_gui import app, win, pump, settle, _wait     # noqa: E402,F401  (fixtures)


def _open(app, win):
    act = win.actions_by_id['perception_range']
    act.trigger()
    pump(app, 0.2)
    pg = win.page('perception')
    assert win.page_name() == 'perception' and pg.isVisible()
    return pg


def _type(app, f, text):
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    QTest.mouseClick(f.lineEdit(), Qt.MouseButton.LeftButton)
    QTest.keyClick(f.lineEdit(), Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
    QTest.keyClicks(f.lineEdit(), text)
    QTest.keyClick(f.lineEdit(), Qt.Key.Key_Return)
    pump(app, 0.1)


def test_page_picks_a_condition_edits_it_compares_and_undoes(app, win, tmp_path):
    pg = _open(app, win)
    ctl = win.ctx.perception
    _wait(app, lambda: ctl.result is not None and not ctl.busy and pg.wres is not None, 60)
    pump(app, 0.2)
    clear_far = ctl.result.far.copy()
    car_ahead = ctl.result.at(0, 0.0)[0]
    t = pg.wcond
    assert t.rowCount() == len(W.IDS) and t.item(0, 0).text() == 'Clear (reference)'
    assert t.item(0, 1).text() == f'{car_ahead:.0f}' and t.horizontalHeaderItem(1).text() == 'Car'
    assert not pg.wx_fields['visibility_m'].isEnabled()
    # pick fog in the combo: saved, undoable, the envelopes follow with the clear one dashed
    pg.cond.setCurrentIndex(pg.cond.findData('fog'))
    pump(app, 0.1)
    assert win.store.project.analysis.weather['condition'] == 'fog'
    assert pg.wx_fields['visibility_m'].isEnabled() and pg.wx_fields['visibility_m'].value() == 150
    _wait(app, lambda: PR._condition(ctl.result).id == 'fog' and not ctl.busy and not ctl.stale, 60)
    assert 'Fog' in pg.status.text() and pg.map.pr.weather['clear_far'] is not None
    # a denser fog, typed into its field: the front range shrinks below the clear one
    _type(app, pg.wx_fields['visibility_m'], '40')
    assert win.store.project.analysis.weather['params']['fog']['visibility_m'] == 40
    _wait(app, lambda: PR._condition(ctl.result).visibility_m == 40 and not ctl.busy and not ctl.stale, 60)
    assert ctl.result.at(0, 0.0)[0] < car_ahead - 5
    _wait(app, lambda: pg.wres is not None and pg.wres.condition('fog').visibility_m == 40, 60)
    pump(app, 0.2)
    from PyQt6.QtCore import Qt
    row = [pg.wcond.item(r, 0).data(Qt.ItemDataRole.UserRole) for r in range(pg.wcond.rowCount())].index('fog')
    assert pg.wcond.item(row, 0).text() == 'Fog' and pg.wcond.item(row, 0).toolTip().startswith('Fog · 40 m')
    pg.wcond.setCurrentCell(row, 0)
    pump(app, 0.1)
    tags = [pg.wsens.item(r, 0).text() for r in range(pg.wsens.rowCount())]
    assert tags == [c.tag for c in win.store.variant.cameras if c.visible]
    # undo the visibility, then the condition: back to the clear envelopes exactly
    win.store.undo.undo()
    pump(app, 0.1)
    assert pg.wx_fields['visibility_m'].value() == 150
    win.store.undo.undo()
    pump(app, 0.1)
    assert win.store.project.analysis.weather == {} and pg.cond.currentData() == 'clear'
    _wait(app, lambda: PR._condition(ctl.result).clear and not ctl.busy and not ctl.stale, 60)
    assert ctl.result.far.tobytes() == clear_far.tobytes()
    # night: the beam choice is offered; export the comparison
    pg.cond.setCurrentIndex(pg.cond.findData('night'))
    pump(app, 0.1)
    assert pg.beam.isEnabled()
    pg.beam.buttons['high'].click()
    pump(app, 0.1)
    assert win.store.project.analysis.weather['params']['night']['beam'] == 'high'
    _wait(app, lambda: pg.wres is not None and pg.wres.selected == 'night' and not pg._wbusy, 60)
    paths = pg.export_weather(str(tmp_path))
    assert [os.path.basename(x) for x in paths][0] == 'weather.json'
    assert json.loads((tmp_path / 'weather.json').read_text())['selected'] == 'night'
    # a new project replaces the comparison (File ▸ New / Open)
    old = pg.wres
    win.store.set_project(samples.tigor_sample_project())
    pump(app, 0.1)
    assert pg.cond.currentData() == 'clear' and pg.wres is not old
    _wait(app, lambda: pg.wres is not None and pg.wres.selected == 'clear' and not pg._wbusy, 60)
    assert pg.wcond.rowCount() == len(W.IDS)
