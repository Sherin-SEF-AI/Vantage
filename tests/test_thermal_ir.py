"""Thermal (LWIR) cameras and the night photon budget (core.thermal_ir, ui.workspaces.thermal_ws): the library from
the makers' datasheets, Planck against its closed-form series, the atmosphere (humidity, Mie fog, rain), NETD and the
contrast-to-noise ratio, Johnson ranges against a datasheet, glass blocking the LWIR, the UN R112 table points, the
visible night photon budget, settings, exports and the page driven like a user."""
import csv
import io
import json
import math

import numpy as np
import pytest

from vantage.core import thermal_ir as TH


# ----------------------------------------------------------------------------- helpers
def _data_rig():
    from vantage.core import samples
    p = samples.tigor_sample_project()
    v = next(v for v in p.variants if v.sensors)
    p.active_uid = v.uid
    return p


def _engine():
    from vantage.core.analysis import Engine
    return Engine()


# ----------------------------------------------------------------------------- the library
def test_library_enters_the_builtin_library_with_datasheet_figures():
    from vantage.core.lens import builtin_lenses, builtin_modules, resolve, model_fov
    mods = {m.id: m for m in builtin_modules()}
    lenses = {ln.id: ln for ln in builtin_lenses()}
    th = {k: m for k, m in mods.items() if m.modality == 'lwir'}
    # Boson+ 640 and 320, the ADK, Lynred's PICO640S and the Tura (three HFOV variants)
    assert {'flir_bosonplus_640', 'flir_bosonplus_320', 'flir_adk', 'lynred_pico640s', 'flir_tura_24',
            'flir_tura_42', 'flir_tura_70'} <= set(th)
    b = th['flir_bosonplus_640']
    assert (b.width_px, b.height_px, b.pixel_um) == (640, 512, 12.0) and b.vendor == 'Teledyne FLIR'
    mp = b.modality_params
    assert mp['netd_mk'] == 20.0 and mp['netd_f_number'] == 1.0 and mp['band_um'] == [8.0, 14.0]
    assert mp['export_fps_hz'] == 8.6 and 'Release 117' in b.note and 'VOx' in mp['detector']
    assert th['flir_adk'].modality_params['netd_mk'] == 50.0 and 'netd_f_number' not in th['flir_adk'].modality_params
    assert th['flir_adk'].ip_rating == 'IP67' and th['flir_adk'].modality_params['export_fps_hz'] == 9.0
    pico = th['lynred_pico640s']
    assert (pico.width_px, pico.height_px, pico.pixel_um) == (640, 480, 17.0)
    assert pico.modality_params['netd_scene_k'] == 300.0 and pico.fps == 120.0
    assert th['flir_tura_42'].modality_params['shutterless'] is True and th['flir_tura_42'].ip_rating == 'IP6K9K'
    # a figure the sheet does not give is absent: the Tura's f-number, the PICO's detector material
    assert lenses['tura_42'].f_number == 0.0 and 'material not stated' in pico.modality_params['detector']
    # every thermal lens sits on its module and reproduces the stated horizontal FOV within 0.5°
    tl = [ln for ln in lenses.values() if ln.module_id in th]
    assert len(tl) == 10 + 8 + 4 + 3 + 3
    for ln in tl:
        fh = model_fov(mods[ln.module_id], ln)[0]
        assert abs(fh - ln.sheet_fov[0]) < 0.5, (ln.id, fh, ln.sheet_fov)
        assert 'read 2026-11-11' in ln.note
    a032 = lenses['bosonplus640_a032']
    assert a032.focal_mm == 14.0 and a032.f_number == 1.0 and TH.LENS_SPECS['bosonplus640_a032']['transmission'] == 0.92
    # the 24° Boson+ lens is rectilinear enough: 18 mm on 7.68 mm gives 24.1° (pinhole)
    assert 2 * math.degrees(math.atan(3.84 / 18.0)) == pytest.approx(lenses['bosonplus640_a024'].sheet_fov[0], abs=0.3)
    assert resolve(b, lenses['bosonplus640_as32']).width == 640
    # the visible library is unchanged: modules default to rgb
    assert mods['sturdecam31'].modality == 'rgb'
    # fresh copies: editing one does not change the library
    m1 = TH.library('modules')[0]
    m1.modality_params['netd_mk'] = 1.0
    assert TH.library('modules')[0].modality_params['netd_mk'] == 20.0 and TH.library('nonsense') == []


def test_module_round_trip_keeps_the_thermal_params():
    from vantage.core.lens import Module
    m = next(x for x in TH.library('modules') if x.id == 'flir_tura_24')
    m2 = Module.from_dict(json.loads(json.dumps(m.to_dict())))
    assert m2.modality == 'lwir' and m2.modality_params == m.modality_params


# ----------------------------------------------------------------------------- Planck
def test_planck_band_against_series_and_stefan_boltzmann():
    for T in (233.15, 273.15, 300.0, 330.0, 400.0):
        assert float(TH.band_radiance(T)) == pytest.approx(TH.band_radiance_series(T), rel=1e-9)
    T = 300.0
    assert TH.band_radiance_series(T, (1e-3, 1e12), 5000) == pytest.approx(TH.SIGMA_SB * T ** 4 / math.pi, rel=1e-9)
    # known value: a 300 K blackbody radiates ~ 46 W/(m²·sr) in 8–14 µm (about 35 % of σT⁴/π)
    assert 0.33 < float(TH.band_radiance(300.0)) / (TH.SIGMA_SB * 300 ** 4 / math.pi) < 0.40
    # Wien: the peak at 2898/T µm
    lam = np.linspace(5, 15, 4001)
    assert lam[np.argmax(TH.planck(lam, 300.0))] == pytest.approx(2897.77 / 300.0, abs=0.01)
    # dL/dT against a central difference
    d = (float(TH.band_radiance(300.01)) - float(TH.band_radiance(299.99))) / 0.02
    assert float(TH.band_dl_dt(300.0)) == pytest.approx(d, rel=1e-6)
    assert TH.apparent_temperature(float(TH.band_radiance(287.3))) == pytest.approx(287.3, abs=1e-6)


# ----------------------------------------------------------------------------- the atmosphere
def test_mie_against_bhmie_example_and_fogs():
    qe, qs, qb = TH.mie_q(1.55, 2 * math.pi * 0.525 / 0.6328)
    assert qe == pytest.approx(3.10543, rel=1e-4) and qs == pytest.approx(3.10543, rel=1e-4)
    assert qb == pytest.approx(2.92534, rel=1e-4)
    assert TH.mie_q(1.33, 400.0)[0] == pytest.approx(2.0, abs=0.05)               # the extinction paradox
    # Shettle & Fenn's fogs: about 130 m and 450 m at 2 % contrast
    from vantage.core import weather as W
    assert TH.fog_visibility_m('advection') * 3.912 / W.MOR_LN == pytest.approx(130, rel=0.05)
    assert TH.fog_visibility_m('radiation') * 3.912 / W.MOR_LN == pytest.approx(450, rel=0.05)
    # small drops pass the LWIR far better than light; large drops hardly better
    assert (TH.fog_ratio('radiation', [8, 10, 12]) < 0.6).all()
    assert (TH.fog_ratio('advection', [8, 10, 12]) > 0.8).all()


def test_humidity_and_rain_transmission():
    d = np.array([0.0, 10, 100, 1000, 5000])
    t = TH.humidity_transmission(d, 70, 10)
    assert t[0] == pytest.approx(1.0) and (np.diff(t) < 0).all() and (t > 0).all() and (t <= 1).all()
    # more water vapour, less transmission
    assert TH.humidity_transmission(500, 90, 30) < TH.humidity_transmission(500, 30, 30) < \
        TH.humidity_transmission(500, 30, 5) + 0.2
    assert TH.water_vapour(100, 20) == pytest.approx(17.3, rel=0.03)               # ≈ saturated g/m³ at 20 °C
    from vantage.core import weather as W
    assert TH.rain_extinction(25.0) == pytest.approx(1.076 * 25 ** 0.67 / W.DB_PER_NEPER / 1000)


def test_contrast_falls_monotonically_with_fog_and_range():
    st = TH.ThermalSettings()
    d = np.array([5.0, 20, 50, 100, 200, 400])
    base = TH.COND['clear_night']
    prev = None
    for mor in (0.0, 1000.0, 300.0, 150.0, 60.0):
        c = TH.clean_condition(base, {'visibility_m': mor, 'sky': 'overcast' if mor else 'clear'})
        dt = TH.apparent_contrast('pedestrian', c, st, 0.5, d)['dT']
        assert (np.diff(np.abs(dt)) < 0).all()                     # farther: less contrast
        if prev is not None:
            assert (np.abs(dt) <= np.abs(prev) + 1e-9).all()        # denser fog: less contrast at every range
        prev = dt if mor else None


def test_cnr_scales_with_netd_and_f_number():
    lens = next(ln for ln in TH.library('lenses') if ln.id == 'bosonplus640_a032')
    m = next(x for x in TH.library('modules') if x.id == 'flir_bosonplus_640')
    st = TH.ThermalSettings()
    f20 = TH.camera_figures(m, lens, st)
    m.modality_params['netd_mk'] = 40.0
    f40 = TH.camera_figures(m, lens, st)
    assert f40['ne_dl'] == pytest.approx(2 * f20['ne_dl'])                          # noise ∝ NETD
    lens.f_number = 2.0
    f_slow = TH.camera_figures(m, lens, st)
    assert f_slow['ne_dl'] == pytest.approx(4 * f40['ne_dl'])                       # ∝ N² (FLIR Table 13)
    # NEΔL = NETD · dL/dT(30 °C scene) · N²/τ
    assert f20['ne_dl'] == pytest.approx(0.020 * float(TH.band_dl_dt(303.15)) * 1.0 / 0.92)
    # the focal-plane irradiance chain gives the same ratio: CNR = ΔE / NEE
    dL = 0.5
    cnr_e = TH.detector_irradiance(dL, 1.0, 0.92) / TH.detector_irradiance(0.020 * float(TH.band_dl_dt(303.15)), 1.0)
    assert cnr_e == pytest.approx(dL / f20['ne_dl'])
    # a camera whose sheet gives no f-number or transmission: assumed and listed as needed
    adk = next(x for x in TH.library('modules') if x.id == 'flir_adk')
    fa = TH.camera_figures(adk, next(ln for ln in TH.library('lenses') if ln.id == 'adk_24'), st)
    assert set(fa['needed']) == {'lens f-number', 'lens transmission'} and fa['netd_eff_mk'] == pytest.approx(50.0)


def test_iso7730_clothing_temperature_and_hot_road_inversion():
    # ISO 7730 balance: in still air at the skin temperature there is nothing to lose; colder air, colder clothes
    t20 = TH.clothing_temperature_c(20, 20, 70, 1.0, 0.1)
    t10 = TH.clothing_temperature_c(10, 10, 70, 1.0, 0.1)
    assert 20 < t20 < 32 and t10 < t20
    # the residual of the ISO 7730 equation at the solution
    Icl, fcl = 0.155, 1.05 + 0.645 * 0.155
    hc = max(2.38 * abs(t20 - 20) ** 0.25, 12.1 * math.sqrt(0.1))
    rhs = 35.7 - 0.028 * 70 - Icl * (3.96e-8 * fcl * ((t20 + 273) ** 4 - (20 + 273) ** 4) + fcl * hc * (t20 - 20))
    assert rhs == pytest.approx(t20, abs=1e-6)
    st = TH.ThermalSettings()
    hot, night = TH.COND['hot_road'], TH.COND['clear_night']
    assert TH.apparent_contrast('pedestrian', night, st, 0.5, [10.0])['dT0'] > 3.0
    low = TH.apparent_contrast('pedestrian', hot, st, 0.5, [10.0])['dT0']
    high = TH.apparent_contrast('pedestrian', hot, st, 1.5, [10.0])['dT0']
    assert high < low < 0                     # inverted, the more so the more of the person is seen against the road
    assert TH.road_share(0.9, 'pedestrian') == pytest.approx(0.5)
    assert TH.munsell_reflectance(5.0) == pytest.approx(0.1927, abs=0.001)    # ASTM D1535: N5, perfect diffuser


# ----------------------------------------------------------------------------- Johnson, R112
def test_johnson_range_against_lynred_datasheet():
    for chk in TH.validation_checks():
        o = chk['fn'](None)
        assert o.error <= chk['tolerance'] and o.ok, (chk['id'], o.error, o.detail)
    assert {c['id'] for c in TH.validation_checks()} == {'thermal.planck_band', 'thermal.johnson_pico640s',
                                                         'thermal.r112_points', 'thermal.mie_bhmie',
                                                         'thermal.fog_visibility'}
    assert all(c['model'] == 'thermal' and set(c['refs']) <= set(TH.VALIDATION_REFERENCES) for c in
               TH.validation_checks())


def test_r112_table_points():
    rows = {(r['beam'], r['point']): r for r in TH.r112_table()}
    assert rows[('low', '50R')]['cd'] == 10100 and rows[('low', '50R')]['limit'] == 'min'
    assert rows[('low', '50R')]['lux_per_lamp'] == pytest.approx(10100 / 50 ** 2)
    assert rows[('low', '75R')]['lux_per_lamp'] == pytest.approx(10100 / 75 ** 2)
    assert rows[('low', '25L')]['lux_per_lamp'] == pytest.approx(1700 / 625) and rows[('low', 'B50L')]['cd'] == 350
    assert rows[('high', 'Imax')]['cd'] == 40500 and rows[('high', 'Imax')]['cd_max'] == 215000
    assert rows[('high', 'Imax')]['lux_per_lamp'] == pytest.approx(4.05)
    # the 50R point is the road 50 m ahead for a lamp at 0.75 m (R112's screen geometry)
    assert rows[('low', '50R')]['road_m'] == pytest.approx(50.0, rel=0.01)
    from vantage.core import weather as W
    wst = W.WeatherSettings()
    # the typical lamp meets the minimums; the R112-minimum lamp meets them with the weakest exactly at it
    for name, t in TH.R112_PASSING.items():
        if 'min' in t:
            assert float(TH.lamp_cd(t['h'], t['v'], 'low', 'typical', wst)) >= t['min']
            assert float(TH.lamp_cd(t['h'], t['v'], 'low', 'r112_min', wst)) >= t['min'] * (1 - 1e-9)
    assert 0 < TH.r112_scale('low') < 1 and 0 < TH.r112_scale('high') < 1
    # illuminance on the pedestrian falls with range; the high beam lights farther
    lamps = W.lamp_positions(None, wst)
    G = np.c_[np.array([20.0, 50, 100]) + 3.05, np.full(3, -1.5)]
    lo = TH.lit_illuminance(G, 1.8, lamps, 'low', 'typical', wst, 0.0)
    hi = TH.lit_illuminance(G, 1.8, lamps, 'high', 'typical', wst, 0.0)
    assert (np.diff(lo) < 0).all() and (hi > lo).all()
    assert (TH.lit_illuminance(G, 1.8, lamps, 'low', 'r112_min', wst, 0.0) < lo).all()


# ----------------------------------------------------------------------------- settings
def test_settings_round_trip_and_junk():
    st = TH.ThermalSettings(conditions=['fog', 'hot_road'], criteria='johnson', snr_min=3.0, beam_model='r112_min',
                            clothing='dark', windows={'abc': 'ge_ar'},
                            params={'fog': {'visibility_m': 80.0, 'fog': 'advection'}})
    d = json.loads(json.dumps(st.to_dict()))
    assert TH.ThermalSettings.from_dict(d) == st
    j = TH.ThermalSettings.from_dict({'conditions': ['nope', 'rain', 3], 'criteria': 'x', 'snr_min': 'abc',
                                      'lateral_m': 1e9, 'samples': 3.7, 'max_range_m': float('nan'),
                                      'windows': {'u': 'glass2', 'v': 'znse'}, 'clothing_value': -2,
                                      'params': {'fog': {'visibility_m': 5, 't_air_c': 1e6, 'sky': 'purple'},
                                                 'nope': {}, 'rain': 'x'}, 'classes': ['nothing']})
    assert j.conditions == ['rain'] and j.criteria == 'dri' and j.snr_min == TH.ROSE_K and j.lateral_m == 10.0
    assert j.samples == 40 and j.max_range_m == 1000.0 and j.windows == {'v': 'znse'} and j.clothing_value == 0.0
    assert j.params == {'fog': {'t_air_c': 45.0, 'visibility_m': 20.0}} and j.classes == list(TH.CLASSES)
    assert TH.ThermalSettings.from_dict(None) == TH.ThermalSettings() == TH.ThermalSettings.from_dict('junk')
    assert j.condition('fog').visibility_m == 20.0 and j.condition('fog').sky == 'overcast'


# ----------------------------------------------------------------------------- the study
@pytest.fixture(scope='module')
def study():
    p = _data_rig()
    e = _engine()
    cam = TH.add_thermal_camera(p, engine=e)
    res = TH.run(p, e)
    return p, e, cam, res


def test_study_thermal_against_visible_at_night(study):
    p, e, cam, res = study
    assert cam.tag == 'T' and cam.module_id == 'flir_bosonplus_640' and cam.z == 0.5
    assert res.cam('T')['modality'] == 'lwir' and res.cam('F')['modality'] == 'rgb'
    assert [c.id for c in res.conditions] == list(TH.COND_IDS)
    for cid in TH.COND_IDS:
        f = res.first_to_see(cid)
        assert f['thermal']['tag'] == 'T' and f['visible']['tag'] == 'F' and f['winner'] == 'thermal'
        # the visible camera in the low beam is light-limited at night, the high beam sees farther
        lo = res.row('F', cid, beam='low')
        hi = res.row('F', cid, beam='high')
        assert 10 < lo['range_m'] < hi['range_m'] and lo['limited_by'] in ('light', 'fog contrast')
    # fog shortens both; recognition and identification are shorter than detection
    assert res.row('F', 'fog', beam='low')['range_m'] < res.row('F', 'clear_night', beam='low')['range_m']
    t = {lvl: res.row('T', 'clear_night', level=lvl)['range_m'] for lvl in TH.LEVELS}
    assert t['identification'] < t['recognition'] < t['detection']
    # the recognition range is pixel-limited on a clear night: 6 px across √(1.8 × 0.5) at the 32° lens's IFOV
    from vantage.core.lens import resolve
    intr = resolve(p.module_for(cam), p.lens_for(cam))
    exp = math.sqrt(0.9) * intr.f_px / 6.0
    r = res.row('T', 'clear_night', level='recognition')
    assert r['limited_by'] == 'pixels' and r['range_m'] == pytest.approx(exp, rel=0.03)
    # the hot road inverts the contrast: the study says so
    assert res.scenes['hot_road']['pedestrian_dT0_k']['T'] < 0 < res.scenes['clear_night']['pedestrian_dT0_k']['T']
    assert any(f['kind'] == 'inversion' for f in res.findings)
    # rear and side cameras get no headlamp light
    assert res.row('R', 'clear_night', beam='low')['range_m'] == 0.0


def test_denser_fog_and_worse_netd_shorten_the_thermal_range():
    p = _data_rig()
    e = _engine()
    cam = TH.add_thermal_camera(p, engine=e, lens_id='bosonplus640_a012')       # narrow: contrast-limited in fog
    out = []
    for mor in (300.0, 150.0, 60.0):
        st = TH.ThermalSettings(conditions=['fog'], params={'fog': {'visibility_m': mor, 'fog': 'advection'}})
        out.append(TH.run(p, e, st).row('T', 'fog')['range_m'])
    assert out[0] > out[1] > out[2] > 0
    p.modules['flir_bosonplus_640'].modality_params['netd_mk'] = 200.0
    st = TH.ThermalSettings(conditions=['fog'], params={'fog': {'visibility_m': 150.0, 'fog': 'advection'}})
    worse = TH.run(p, e, st).row('T', 'fog')
    assert worse['range_m'] < out[1] and worse['limited_by'] == 'thermal contrast'


def test_glass_blocks_lwir_and_an_ir_window_lets_it_through():
    from vantage.core import materials as MAT
    p = _data_rig()
    e = _engine()
    cam = TH.add_thermal_camera(p, engine=e)
    # behind the windscreen, inside the cabin, looking ahead
    vg = e.vehicle_geom(p)
    found = None
    for x in np.arange(0.8, 2.6, 0.1):
        for z in (1.15, 1.25, 1.3, 1.35):
            cam.x, cam.y, cam.z, cam.pitch = float(x), 0.0, float(z), 3.0
            MAT.clear_cache()
            if MAT.cached_run(p, vg).camera(cam.tag) is not None:
                found = (x, z)
                break
        if found:
            break
    assert found is not None, 'no pose behind the windscreen found'
    res = TH.run(p, e)
    assert res.cam('T')['window'] == 'glass'
    r = res.row('T', 'clear_night')
    assert r['range_m'] == 0.0 and r['limited_by'] == 'glass'
    assert any(f['kind'] == 'glass' and f['severity'] == 'critical' for f in res.findings)
    iss = TH.issues(p, vg)
    assert len(iss) == 1 and iss[0].severity == 'critical' and iss[0].uids == [cam.uid] and 'glass' in iss[0].message
    # a germanium window cut into the body: it sees again, at the window's transmission
    p.analysis.thermal = TH.ThermalSettings(windows={cam.uid: 'ge_ar'}).to_dict()
    res2 = TH.run(p, e)
    assert res2.cam('T')['window'] == 'ge_ar' and res2.row('T', 'clear_night')['range_m'] > 100
    assert TH.issues(p, vg) == []
    assert TH.WINDOWS['ge'][1] == pytest.approx(2 * 4.003 / (4.003 ** 2 + 1))      # uncoated Fresnel slab ~ 47 %
    # no thermal camera, no vehicle: nothing to say
    q = _data_rig()
    assert TH.issues(q) == []


def test_perception_range_works_on_a_thermal_camera(study):
    from vantage.core import perception as PR
    p, e, cam, _ = study
    st = PR.PerceptionSettings(azimuths=36, samples=60)
    pr = PR.analyse(p, e, settings=st)
    tags = [t for _, t, _ in pr.cams]
    ci = tags.index('T')
    k = [c.name for c in pr.classes].index('Pedestrian')
    far = pr.cam_far[ci, k, 0]                                # ahead
    # its pixels on target follow the thermal module's 640 px: 24 px tall at f_px × 1.7 / range
    from vantage.core.lens import resolve
    intr = resolve(p.module_for(cam), p.lens_for(cam))
    assert far == pytest.approx(intr.f_px * 1.7 / 24.0, rel=0.08)
    # the hook core.perception can use at night: a CNR margin that falls with range
    from vantage.core import weather as W
    m = TH.perception_margin(p, cam, 'Pedestrian', W.PRESET['fog'], np.array([10.0, 100.0, 400.0]))
    assert m[0] > m[1] > m[2]


def test_summary_doc_csv_png_files(study, tmp_path):
    p, e, cam, res = study
    s = TH.summary(res)
    json.dumps(s)
    assert s['conditions']['fog']['winner'] == 'thermal' and s['thermal_cameras'] == ['T']
    assert s['pedestrian_detection_m']['F/low']['clear_night'] > 0
    doc = TH.to_doc(res, p)
    txt = json.dumps(doc, default=str)
    assert doc['schema'] == TH.SCHEMA and doc['r112']['source'].startswith('UN Regulation No. 112')
    assert len(doc['ranges']) == len(res.rows) and 'NaN' not in txt
    rows = list(csv.DictReader(io.StringIO(TH.to_csv(res))))
    assert len(rows) == len(res.rows) and rows[0]['camera'] == res.rows[0]['tag']
    head = TH.curves_csv(res).splitlines()[0].split(',')
    assert head[0] == 'range_m' and len(head) == 1 + len(res.curves)
    names = [n for n, _ in TH.files(res, p)]
    assert names == ['thermal.json', 'thermal_ranges.csv', 'thermal_curves.csv', 'thermal_ranges.png',
                     'thermal_cnr.png']
    paths = TH.export(res, p, str(tmp_path))
    assert all((tmp_path / n).stat().st_size > 100 for n in names) and len(paths) == 5
    import cv2
    img = cv2.imread(str(tmp_path / 'thermal_ranges.png'))
    assert img is not None and img.shape[1] == 1000


def test_cancel_and_empty_variant():
    p = _data_rig()
    assert TH.run(p, None, cancel=lambda: True) is None
    p.variant.cameras = []
    res = TH.run(p, None)
    assert res.rows == [] and any(f['kind'] == 'no_thermal' for f in res.findings)
    TH.summary(res)
    TH.to_doc(res, p)


# ----------------------------------------------------------------------------- the page, driven like a user
from test_gui import app, win, pump, settle, _wait     # noqa: E402,F401  (fixtures)


def _open(win, name, cls):
    if name not in win._page_widgets:
        pg = cls(win.ctx)
        win.pages.addWidget(pg)
        win._page_widgets[name] = pg
    win.set_page(name)
    return win._page_widgets[name]


def _type_into_field(app, f, text):
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    QTest.mouseClick(f.lineEdit(), Qt.MouseButton.LeftButton)
    QTest.keyClick(f.lineEdit(), Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
    QTest.keyClicks(f.lineEdit(), text)
    QTest.keyClick(f.lineEdit(), Qt.Key.Key_Return)
    pump(app, 0.1)


def test_page_driven_like_a_user(app, win, tmp_path):
    from vantage.ui.workspaces.thermal_ws import ThermalPage
    store = win.store
    v = next(x for x in store.project.variants if x.sensors)
    if store.project.active_uid != v.uid:
        store.edit('variants', lambda q: setattr(q, 'active_uid', v.uid), 'data rig')
        settle(app, win)
    pg = _open(win, 'thermal', ThermalPage)
    pump(app, 0.2)
    _wait(app, lambda: pg.result is not None and not pg.busy)
    assert pg.result.metrics['thermal_cameras'] == 0
    assert any(f['kind'] == 'no_thermal' for f in pg.result.findings) and len(pg.findings) == len(pg.result.findings)
    n = len(store.variant.cameras)
    # add a Boson+ 640: one undo step, the study runs again and the thermal camera wins at night
    pg.b_add.click()
    pump(app, 0.1)
    assert len(store.variant.cameras) == n + 1 and store.variant.cameras[-1].module_id == 'flir_bosonplus_640'
    _wait(app, lambda: pg.result is not None and not pg.busy and pg.result.cam('T') is not None and not pg.stale)
    assert pg.tiles['thermal'].sev == 'good' and pg.tiles['thermal'].note.startswith('T')
    assert pg.table.rowCount() == len(pg.result.cams) * 2 - 1
    assert pg.r112.rowCount() == len(pg.result.r112) and pg.r112.item(0, 1).text() == 'B50L'
    assert any(t == 'T' for t, _ in pg._win_boxes.values())
    # the hot road: pick it, the tile shows the inverted contrast
    pg.cond_pick.buttons['hot_road'].click()
    pump(app, 0.1)
    assert pg.tiles['dT'].value.startswith('-') and pg.tiles['dT'].sev == 'warning'
    # make the hot road's fog dense: typed into the field, saved with the project, undoable
    _type_into_field(app, pg.cond_fields['visibility_m'], '80')
    assert store.project.analysis.thermal['params']['hot_road']['visibility_m'] == 80.0
    _wait(app, lambda: not pg.busy and not pg.stale and pg.result.settings.condition('hot_road').visibility_m == 80)
    # Johnson's criteria with a click; nonsense clamped
    pg.crit.buttons['johnson'].click()
    pump(app, 0.1)
    assert store.project.analysis.thermal['criteria'] == 'johnson'
    _type_into_field(app, pg.fields['snr_min'][0], '0.01')
    assert store.project.analysis.thermal['snr_min'] == 1.0 and pg.fields['snr_min'][0].value() == 1.0
    # a germanium window for T from its combo box
    box = next(b for u, (t, (a, b)) in pg._win_boxes.items() if t == 'T')
    box.setCurrentIndex(box.findData('ge_ar'))
    pump(app, 0.1)
    uid = store.variant.by_tag('T').uid
    assert store.project.analysis.thermal['windows'] == {uid: 'ge_ar'}
    # a condition off
    pg.cond_toggles['rain'].click()
    pump(app, 0.1)
    assert 'rain' not in store.project.analysis.thermal['conditions']
    _wait(app, lambda: not pg.busy and not pg.stale and [c.id for c in pg.result.conditions] ==
          ['clear_night', 'fog', 'hot_road'])
    # export
    paths = pg.export(str(tmp_path))
    assert json.loads((tmp_path / 'thermal.json').read_text())['schema'] == TH.SCHEMA and len(paths) >= 5
    # undo every edit, back to the start: the thermal camera goes again
    for _ in range(6):
        store.undo.undo()
    pump(app, 0.1)
    assert store.project.analysis.thermal == {} and len(store.variant.cameras) == n
    _wait(app, lambda: not pg.busy and not pg.stale and pg.result.cam('T') is None)
    assert pg.crit.value() == 'dri' and pg.cond_toggles['rain'].isChecked()
    # hidden: edits do not run it; shown again they do
    win.set_page('viewports')
    pump(app, 0.2)
    shown = pg.result
    store.edit('analysis', lambda q: setattr(q.analysis, 'thermal', {'snr_min': 3.0}), 'test')
    pump(app, 0.5)
    assert pg.result is shown and pg.stale
    _open(win, 'thermal', ThermalPage)
    _wait(app, lambda: not pg.busy and pg.result is not shown and pg.result.settings.snr_min == 3.0)
