"""Thermal intrinsic calibration curves (core.thermal_intrinsics, the temperature fields of core.calib, the page)."""
import json
import math
import os
import time

import numpy as np
import pytest

from vantage.core import calib as CL
from vantage.core import samples
from vantage.core import thermal_intrinsics as TI


def _cam(p=None):
    p = p or samples.tigor_sample_project()
    var = next(v for v in p.variants if v.sensors)
    p.active_uid = var.uid
    return p, var.cameras[0]


def _records(slopes, temps, noise=0.0, seed=0, curv=None, base=None):
    """Records of a 1920×1536 unit whose fx fy cx cy move by slopes (px/K) and curv (px/K²) from 20 °C."""
    rng = np.random.default_rng(seed)
    base = base or CL.make_record(1920, 1536, 860.0, 860.5, 959.5, 767.5, [0.01, 0.1, -0.2, 0.05])
    curv = curv or {}
    out = []
    for t in temps:
        d = t - 20.0
        r = dict(base)
        for q in TI.PX_PARAMS:
            r[q] = base[q] + slopes.get(q, 0.0) * d + curv.get(q, 0.0) * d * d + rng.normal(0, noise)
        r['temp_c'] = float(t)
        r['source'] = 'test'
        out.append(r)
    return out


# ------------------------------------------------------------------------------------------------ physics
def test_thermo_optic_coefficients_from_the_cited_data():
    # Jamieson 1981 x_f = α − (dn_rel/dT)/(n − 1), SCHOTT N-BK7: α 7.1e-6, dn_rel/dT (e, +20/+40) 3.0e-6, nd 1.51680
    assert TI.x_f('n-bk7', 30.0) == pytest.approx(7.1e-6 - 3.0e-6 / 0.51680, rel=1e-9)
    # the datasheet's rows: 2.9 at −30 °C, 3.1 at +70 °C
    assert TI.x_f('n-bk7', -30.0) == pytest.approx(7.1e-6 - 2.9e-6 / 0.51680, rel=1e-9)
    assert TI.x_f('pmma') == pytest.approx(67.4e-6 + 85e-6 / 0.49, rel=1e-9)        # a plastic lens: ~40× glass
    assert TI.x_f('pmma') > 100 * TI.x_f('n-bk7')
    # fixed focus follows the holder: aluminium 23.6 ppm/K over 50 K
    assert TI.scale_change('n-bk7', 'aluminium', 'housing', 70.0) == pytest.approx(23.6e-6 * 50)
    # in focus follows the lens: integral of x_f, close to x_f(mid)·ΔT
    s = TI.scale_change('n-bk7', 'aluminium', 'lens', 70.0)
    assert s == pytest.approx(TI.x_f('n-bk7', 45.0) * 50, rel=0.02)
    rows = TI.measured_comparison(4.0)
    assert rows[1]['um_per_k'] == pytest.approx(68e-6 * 4000)                         # polycarbonate, 4 mm
    assert rows[3]['um_per_k'] == pytest.approx(7.8 / 22.9, rel=1e-6)                 # Elias 2020 Table 2


# ------------------------------------------------------------------------------------------------ the fit
def test_closed_form_leave_one_out_equals_refits():
    rng = np.random.default_rng(1)
    for n, d in ((4, 1), (6, 1), (6, 2), (9, 2)):
        T = np.sort(rng.uniform(-30, 80, n))
        y = 860 + 0.02 * (T - 20) + rng.normal(0, 0.5, n)
        f = TI.polyfit(T - 20, y, d)
        assert np.allclose(f['loo'], TI.loo_brute(T - 20, y, d), atol=1e-9)


def test_fit_recovers_a_synthetic_drift_within_its_uncertainty():
    slopes = {'fx': 0.06, 'fy': 0.06, 'cx': 0.03, 'cy': -0.02}
    temps = np.linspace(-20, 80, 8)
    misses = 0
    for seed in range(40):
        cur = TI.fit_unit(_records(slopes, temps, noise=0.6, seed=seed), 20.0, 'linear')
        for q, s in slopes.items():
            f = cur.params[q]
            se = math.sqrt(f['cov'][1, 1])
            assert 0 < se < 0.05
            misses += abs(f['coef'][1] - s) > 2 * se
    # 2σ: about 5 % outside; 160 slopes
    assert misses <= 20
    cur = TI.fit_unit(_records(slopes, temps, noise=0.0), 20.0, 'auto')
    for q, s in slopes.items():
        assert cur.slope(q) == pytest.approx(s, abs=1e-9)
        assert cur.params[q]['degree'] == 1
    # the k terms do not move
    assert cur.slope('k1') == pytest.approx(0.0, abs=1e-12)


def test_cross_validation_picks_a_quadratic_only_when_warranted():
    temps = np.linspace(-30, 80, 7)
    # strong curvature, small scatter: quadratic
    cur = TI.fit_unit(_records({'fx': 0.05}, temps, noise=0.2, seed=2, curv={'fx': 2e-3}), 20.0, 'auto')
    assert cur.params['fx']['degree'] == 2 and 'quadratic' in cur.params['fx']['reason']
    assert cur.params['fx']['coef'][2] == pytest.approx(2e-3, rel=0.15)
    # a straight line with scatter: linear, nearly always
    rng = np.random.default_rng(3)
    T = np.linspace(-20, 80, 6)
    false = sum(TI.choose(T - 20, 860 + 0.06 * (T - 20) + rng.normal(0, 1.0, 6), 'auto')[0]['degree'] == 2
                for _ in range(600))
    assert false / 600 < 0.10
    # too few temperatures for the choice, and the forced settings
    f, why, _ = TI.choose(T[:3] - 20, np.array([1.0, 2.0, 2.5]), 'auto')
    assert f['degree'] == 1 and '4 temperatures' in why
    assert TI.choose(T - 20, T * 0.1, 'quadratic')[0]['degree'] == 2
    f, why, press = TI.choose(T[:2] - 20, np.array([1.0, 2.0]), 'auto')
    assert f['degree'] == 1 and math.isnan(f['sigma']) and 'without an uncertainty' in why
    f, why, _ = TI.choose(T[:1] - 20, np.array([1.0]), 'auto')
    assert f['degree'] == 0


def test_intervals_and_extrapolation():
    temps = np.linspace(0, 60, 6)
    cur = TI.fit_unit(_records({'fx': 0.05}, temps, noise=0.5, seed=4), 20.0, 'linear', confidence=0.95)
    mid, far = cur.interval('fx', 30.0), cur.interval('fx', 120.0)
    assert mid['ci'][0] < mid['value'] < mid['ci'][1]
    assert mid['pi'][1] - mid['pi'][0] > mid['ci'][1] - mid['ci'][0]
    assert far['se_mean'] > 2 * mid['se_mean']                     # extrapolation widens the band
    # coverage of the 95 % prediction interval (Monte Carlo, known truth)
    rng = np.random.default_rng(7)
    hits = 0
    for _ in range(600):
        T = np.linspace(-20, 80, 6)
        y = 860 + 0.02 * (T - 20) + rng.normal(0, 0.5, 6)
        c = TI.Curve('u', 'U', '', 1920, 1536, 20.0, list(T), {}, 0.95)
        f = TI.polyfit(T - 20, y, 1)
        c.params = {q: dict(f, reason='', press={}, y=y) for q in TI.PARAMS}
        t = rng.uniform(-20, 80)
        lo, hi = c.interval('fx', t)['pi']
        hits += lo <= 860 + 0.02 * (t - 20) + rng.normal(0, 0.5) <= hi
    assert abs(hits / 600 - 0.95) < 0.03


def test_compensation_round_trip_and_rescaling():
    temps = np.linspace(-20, 80, 6)
    cur = TI.fit_unit(_records({'fx': 0.06, 'cx': 0.03}, temps, noise=0.3, seed=5, curv={'fx': 1e-3}), 20.0,
                      'quadratic')
    ref = cur.reference_record()
    assert 'temp_c' not in ref and ref['thermal']['reference_c'] == 20.0
    for t in (-20.0, 35.0, 80.0):
        a, b = TI.compensate(ref, t), cur.record_at(t)
        for q in TI.PX_PARAMS:
            assert a[q] == pytest.approx(b[q], abs=1e-9)
        assert a['k'] == pytest.approx(b['k'], abs=1e-12)
        assert a['temp_c'] == t and 'thermal_note' not in a
    assert 'extrapolated' in TI.compensate(ref, 110.0)['thermal_note']
    assert 'no thermal curve' in TI.compensate(CL.make_record(10, 8, 5, 5, 4, 3, [0] * 4), 50)['thermal_note']
    assert TI.compensate(cur, 50.0)['fx'] == pytest.approx(cur.record_at(50.0)['fx'])
    # half resolution and back; the curve's px coefficients scale with it
    half = TI.rescale(ref, 960, 768)
    assert half['fx'] == pytest.approx(ref['fx'] / 2)
    assert half['thermal']['coeffs']['fx'][0] == pytest.approx(ref['thermal']['coeffs']['fx'][0] / 2)
    back = TI.rescale(half, 1920, 1536)
    for q in TI.PX_PARAMS:
        assert back[q] == pytest.approx(ref[q])
    assert TI.compensate(half, 60.0)['fx'] * 2 == pytest.approx(TI.compensate(ref, 60.0)['fx'])
    # a sweep at half resolution fits to the module's full resolution
    recs = [TI.rescale(r, 960, 768) for r in _records({'fx': 0.06}, temps)]
    c2 = TI.fit_unit(recs, 20.0, 'linear', (1920, 1536))
    assert c2.slope('fx') == pytest.approx(0.06, abs=1e-9)
    # a record of another aspect is left out with a note
    odd = dict(recs[0], width=1000, height=1000)
    c3 = TI.fit_unit(recs + [odd], 20.0, 'linear', (1920, 1536))
    assert c3.n == len(recs) and any('aspect' in n for n in c3.notes)


def test_settings_tolerate_junk():
    s = TI.ThermalSettings.from_dict({'reference_c': 'x', 'degree': 'cubic', 'warn_px': -5, 'sweep_steps': 99.7,
                                      'views': float('nan'), 'lens_material': 'diamond', 'housing_material': None,
                                      'image_scale': 'lens', 'sweep_lo_c': 50, 'sweep_hi_c': 10, 'seed': 1e20})
    assert s.reference_c == 20.0 and s.degree == 'auto' and s.warn_px == 0.05 and s.sweep_steps == 15
    assert s.views == 14 and s.lens_material == 'n-bk7' and s.housing_material == 'aluminium'
    assert s.image_scale == 'lens' and s.sweep_hi_c > s.sweep_lo_c and s.seed == 2 ** 31 - 1
    assert TI.ThermalSettings.from_dict(None) == TI.ThermalSettings()
    assert TI.ThermalSettings.from_dict(TI.ThermalSettings(views=20).to_dict()).views == 20
    p, cam = _cam()
    p.analysis.thermal_intrinsics = {'sweeps': {cam.uid: [{'fx': 'junk'}, _records({}, [20.0])[0]], 'x': 'y'}}
    assert list(TI.sweeps_of(p)) == [cam.uid] and len(TI.sweeps_of(p)[cam.uid]) == 1
    p.analysis.thermal_intrinsics = {'sweeps': 'nonsense'}
    assert TI.sweeps_of(p) == {}


# ------------------------------------------------------------------------------------------------ calib additions
def test_calibration_files_carry_the_temperature_and_the_curve(tmp_path):
    r0 = CL.make_record(1920, 1536, 860, 860, 960, 768, [0.01, 0.1, -0.2, 0.05])
    assert 'temp_c' not in r0                                   # unchanged when no temperature is given
    r = CL.make_record(1920, 1536, 860, 860, 960, 768, [0.01, 0.1, -0.2, 0.05], temp_c=41.5)
    assert r['temp_c'] == 41.5 and CL.make_record(1, 1, 1, 1, 0, 0, [0] * 4, temp_c='hot').get('temp_c') is None
    # OpenCV YAML: temperature and the thermal block, read back
    cur = TI.fit_unit(_records({'fx': 0.06, 'cy': -0.03}, np.linspace(-20, 80, 6), noise=0.2, seed=8), 20.0,
                      'quadratic')
    rec = dict(cur.reference_record(), temp_c=41.5)
    path = tmp_path / 'F.yaml'
    path.write_text(CL.to_opencv_yaml(rec, 'F'))
    text = path.read_text()
    for key in ('temperature_c: 41.5', 'thermal_reference_c: 20.0', 'thermal_coeffs', 'thermal_sigma',
                'thermal_range_c', 'thermal_params: "fx fy cx cy k1 k2 k3 k4"'):
        assert key in text
    back = CL.read_file(str(path))[0]
    assert back['temp_c'] == 41.5 and back['fx'] == pytest.approx(rec['fx'])
    for q in TI.PARAMS:
        assert back['thermal']['coeffs'][q] == pytest.approx(rec['thermal']['coeffs'][q], rel=1e-9, abs=1e-15)
    assert back['thermal']['range_c'] == pytest.approx([-20.0, 80.0])
    assert TI.compensate(back, 70.0)['fx'] == pytest.approx(cur.record_at(70.0)['fx'], rel=1e-9)
    assert 'thermal' in CL.clean_record(back) and 'temp_c' in CL.clean_record(back)
    # a plain file with no thermal fields still reads as before
    (tmp_path / 'plain.yaml').write_text(CL.to_opencv_yaml(r0, 'R'))
    plain = CL.read_file(str(tmp_path / 'plain.yaml'))[0]
    assert 'temp_c' not in plain and 'thermal' not in plain
    # JSON (generic mapping, Vantage export, Kalibr camchain) with a temperature field
    (tmp_path / 'g.json').write_text(json.dumps({'fx': 860, 'fy': 860, 'cx': 960, 'cy': 768, 'k': [0, 0, 0, 0],
                                                 'width': 1920, 'height': 1536, 'model': 'fisheye',
                                                 'temperature': 63.0}))
    assert CL.read_file(str(tmp_path / 'g.json'))[0]['temp_c'] == 63.0
    p, cam = _cam()
    cam.calib = dict(rec)
    (tmp_path / 'v.json').write_text(CL.to_json(p, [cam]))
    v = CL.read_file(str(tmp_path / 'v.json'))[0]
    assert v['temp_c'] == 41.5 and v['thermal']['coeffs']['fx'] == rec['thermal']['coeffs']['fx']
    p.analysis.thermal_intrinsics = {'sweeps': {cam.uid: _records({'fx': 0.06}, np.linspace(-20, 80, 5))}}
    res = TI.run(p)
    (tmp_path / 'camchain.yaml').write_text(TI.kalibr_yaml(res))
    kb = CL.read_file(str(tmp_path / 'camchain.yaml'))
    assert len(kb) == 1 and kb[0]['thermal']['coeffs']['fx'][0] == pytest.approx(0.06)


def test_temperature_from_metadata_and_a_ros_bag(tmp_path):
    assert TI.temperature_of_metadata({'sensor_temp_c': 47.25}) == 47.25
    assert TI.temperature_of_metadata({'exif': {'CameraTemperature': '31'}}) == 31.0
    assert TI.temperature_of_metadata({'die_temp_k': 313.15}) == pytest.approx(40.0)
    assert TI.temperature_of_metadata({'gain': 3}) is None and TI.temperature_of_metadata(None) is None
    from vantage.core import rosbag as RB
    TI.ensure_temperature_type()
    with RB.BagWriter(str(tmp_path / 'bag')) as bag:
        bag.add_topic('/cam_F/temperature', TI.TEMPERATURE_TYPE)
        for i in range(5):
            t = 1_000_000_000 * (i + 1)
            bag.write('/cam_F/temperature', t, {'header': RB.header(t, 'cam_F'), 'temperature': 30.0 + 2 * i,
                                                'variance': 0.01})
    ts, vs = TI.temperatures_from_bag(str(tmp_path / 'bag'))
    assert list(vs) == [30.0, 32.0, 34.0, 36.0, 38.0]
    assert TI.temperature_at(ts, vs, 2_500_000_000) == pytest.approx(33.0)
    assert TI.temperature_at(ts, vs, 0) == 30.0 and TI.temperature_at([], [], 5) is None
    with pytest.raises(ValueError):
        TI.temperatures_from_bag(str(tmp_path / 'bag'), '/nope')


# ------------------------------------------------------------------------------------------------ the study
def test_run_findings_issues_lens_at_and_exports(tmp_path):
    p, cam = _cam()
    m = p.module_for(cam)
    assert TI.run(p).curves == [] and TI.issues(p) == [] and TI.exit_code(TI.run(p)) == 2
    temps = np.linspace(-20, 80, 6)
    recs = _records({'fx': 0.08, 'cx': 0.02}, temps, noise=0.1, seed=9)
    p.analysis.thermal_intrinsics = {'warn_px': 2.0, 'sweeps': {cam.uid: recs}}
    res = TI.run(p)
    assert len(res.curves) == 1 and res.metrics['fitted'] == 1
    e = next(c for c in res.cameras if c['uid'] == cam.uid)
    assert e['operating'] == m.temp_c                            # the module's own range (−40 … 85)
    assert e['drift_px'] == pytest.approx(0.08 * 65, rel=0.05)   # fx over −40 … 85 from 20 °C: 65 K the far side
    assert any('moves up to' in f['message'] for f in res.findings)
    assert any('extrapolated' in f['message'] for f in res.findings)
    iss = TI.issues(p)
    assert len(iss) == 1 and iss[0].uids == [cam.uid] and iss[0].severity == 'warning'
    assert TI.exit_code(res) == 1
    # the lens at 80 °C has the larger focal length
    l20, l80 = TI.lens_at(p, cam, 20.0), TI.lens_at(p, cam, 80.0)
    from vantage.core.lens import resolve
    assert resolve(m, l80).f_px - resolve(m, l20).f_px == pytest.approx(0.08 * 60, rel=0.05)
    # exports
    paths = TI.export(res, p, str(tmp_path))
    names = {os.path.relpath(x, tmp_path) for x in paths}
    assert {'thermal_curves.json', 'thermal_curves.csv', 'thermal_curves.png', 'kalibr/camchain_thermal.yaml',
            f'opencv/{cam.tag}_thermal.yaml'} <= names
    doc = json.loads((tmp_path / 'thermal_curves.json').read_text())
    assert doc['schema'] == TI.SCHEMA and doc['curves'][0]['params']['fx']['per_k'] == pytest.approx(0.08, abs=0.01)
    assert 'schott_glass' in doc['references']
    back = CL.read_file(str(tmp_path / 'opencv' / f'{cam.tag}_thermal.yaml'))[0]
    assert back['thermal']['coeffs']['fx'][0] == pytest.approx(res.curves[0].slope('fx'))
    s = TI.summary(res)
    json.dumps(s)
    assert s['cameras'][0]['per_k']['fx'] == pytest.approx(0.08, abs=0.01)
    # the camera's own calibration carrying the curve is used when there is no sweep
    cam.calib = res.curves[0].reference_record()
    p.analysis.thermal_intrinsics = {}
    assert resolve(m, TI.lens_at(p, cam, 80.0)).f_px == pytest.approx(resolve(m, l80).f_px, rel=1e-6)


def test_simulated_sweep_through_the_real_calibration_pipeline_recovers_the_drift():
    """Render checkerboards through the true intrinsics at each temperature, calibrate them (calib.calibrate), fit:
    the recovered drift per K agrees with the truth within 3 of its standard errors."""
    p, cam = _cam()
    m, lens = p.module_for(cam), p.nominal_lens_for(cam)
    st = TI.ThermalSettings(lens_material='pmma', image_scale='lens', housing_material='polycarbonate', sweep_steps=4,
                            views=10, scale=0.35, seed=1)
    t0 = time.time()
    sim = TI.simulate_sweep(m, lens, st)
    assert time.time() - t0 < 60
    assert len(sim['records']) == 4 and all('temp_c' in r for r in sim['records'])
    assert sim['records'][0]['width'] == round(m.width_px * 0.35)
    cur = TI.fit_unit(sim['records'], 20.0, 'linear', (m.width_px, m.height_px))
    cmp_ = TI.compare_truth(cur, sim)
    # PMMA in focus: 0.24 % per K of 862 px, about 0.21 px/K
    assert cmp_['fx']['true_per_k'] == pytest.approx(TI.x_f('pmma', 30) * cur.value('fx', 20.0), rel=0.05)
    for q in TI.PX_PARAMS:
        assert cmp_[q]['error_sigma'] < 3.0, (q, cmp_[q])
    assert cmp_['fx']['fitted_per_k'] == pytest.approx(cmp_['fx']['true_per_k'], rel=0.2)
    # the noise-only path for fast use, and cancel
    fast = TI.simulate_sweep(m, lens, TI.ThermalSettings(seed=2), render=False)
    assert len(fast['records']) == 6
    assert TI.simulate_sweep(m, lens, st, cancel=lambda: True) is None


def test_validation_checks_run():
    from vantage.validation.suite import Outcome, ValidationSettings, Ctx
    specs = TI.validation_checks()
    assert {s['id'] for s in specs} == {'thermal_intrinsics.press_closed_form', 'thermal_intrinsics.interval_coverage',
                                        'thermal_intrinsics.drift_recovery'}
    for s in specs:
        assert set(s) >= {'id', 'model', 'title', 'quantity', 'reference', 'tolerance', 'unit', 'refs', 'fn', 'cost'}
        assert all(r in TI.VALIDATION_REFERENCES for r in s['refs'])
    ctx = Ctx.__new__(Ctx)
    ctx.settings, ctx.quick = ValidationSettings(quick=True), True
    for s in specs:
        out = s['fn'](ctx)
        assert isinstance(out, Outcome) and out.ok
        assert out.error <= (out.tolerance if out.tolerance is not None else s['tolerance']), (s['id'], out)


# ------------------------------------------------------------------------------------------------ the page
from test_gui import app, win, pump, settle   # noqa: E402,F401  (fixtures)


def _open(win, name, cls):
    if name not in win._page_widgets:
        pg = cls(win.ctx)
        win.pages.addWidget(pg)
        win._page_widgets[name] = pg
    win.set_page(name)
    return win._page_widgets[name]


def test_page_simulates_fits_compensates_uses_exports_and_undoes(app, win, tmp_path):
    from vantage.ui.workspaces.thermal_intrinsics_ws import ThermalIntrinsicsPage
    pg = _open(win, 'thermal_intrinsics', ThermalIntrinsicsPage)
    pump(app, 0.3)
    store = win.store
    cam = pg._camera()
    assert cam is not None and pg.recs.rowCount() == 0 and 'no calibrations' in pg.headline.text()
    # settings through the fields (undoable)
    pg.fields['sweep_steps'].setValue(3)
    pg.fields['sweep_steps'].committed.emit(3.0)
    pg.fields['views'].committed.emit(8.0)
    pg.housing.setCurrentIndex(pg.housing.findData('polycarbonate'))
    pg.housing.activated.emit(pg.housing.currentIndex())
    assert TI.settings_of(store.project).sweep_steps == 3 and TI.settings_of(store.project).views == 8
    assert TI.settings_of(store.project).housing_material == 'polycarbonate'
    # simulate (a background job), wait for it
    assert pg.simulate()
    t = time.time()
    while pg.busy and time.time() - t < 120:
        pump(app, 0.1)
    pump(app, 0.3)
    assert not pg.busy and pg.recs.rowCount() == 3, pg.status.text()
    assert pg.table.rowCount() == len(TI.PARAMS) and pg.table.item(0, 0).text() == 'fx'
    assert 'simulated' in pg.headline.text()
    assert any('simulated' in f['message'] for f in pg.result.findings) and pg.findings
    # degree, compensation, chart parameter
    pg.degree.buttons['linear'].click()
    pump(app, 0.2)
    assert TI.settings_of(store.project).degree == 'linear'
    assert all(pg.result.curve(cam.uid).params[q]['degree'] == 1 for q in TI.PX_PARAMS)
    pg.comp_t.setValue(75.0)
    pg.comp_t.committed.emit(75.0)
    assert 'fx ' in pg.comp.text() and 'px' in pg.comp.text()
    pg.pick.buttons['cy'].click()
    assert pg.chart.param == 'cy'
    pg.chart.resize(600, 260)
    pg.chart.repaint()
    # use as the unit calibration, then undo
    assert pg.use_as_calibration()
    c = next(x for x in store.project.variant.cameras if x.uid == cam.uid)
    assert c.calib.get('thermal', {}).get('reference_c') == 20.0
    store.undo.undo()
    pump(app, 0.2)
    c = next(x for x in store.project.variant.cameras if x.uid == cam.uid)
    assert not c.calib
    # export
    paths = pg.export(str(tmp_path))
    assert any(x.endswith('thermal_curves.json') for x in paths)
    assert (tmp_path / 'kalibr' / 'camchain_thermal.yaml').exists()
    # remove a record, clear, undo
    pg.recs.selectRow(0)
    pg._remove()
    pump(app, 0.2)
    assert pg.recs.rowCount() == 2
    pg.b_clear.click()
    pump(app, 0.2)
    assert pg.recs.rowCount() == 0
    store.undo.undo()
    pump(app, 0.2)
    assert pg.recs.rowCount() == 2
    # import a file with a temperature and one without
    rec = CL.make_record(1920, 1536, 870, 870, 960, 768, [0.01, 0.1, -0.2, 0.05], temp_c=55.0)
    (tmp_path / 'a.yaml').write_text(CL.to_opencv_yaml(rec, cam.tag))
    (tmp_path / 'b.yaml').write_text(CL.to_opencv_yaml({k: v for k, v in rec.items() if k != 'temp_c'}, cam.tag))
    assert pg.import_files([str(tmp_path / 'a.yaml'), str(tmp_path / 'b.yaml')], temp_c=5.0) == 2
    pump(app, 0.2)
    temps = sorted(r['temp_c'] for r in TI.sweeps_of(store.project)[cam.uid])
    assert 55.0 in temps and 5.0 in temps and len(temps) == 4
    # project replaced while the page is open
    from vantage.core import samples as SM
    win.store.set_project(SM.tigor_sample_project())
    pump(app, 0.3)
    pg.refresh()
    assert pg.recs.rowCount() == 0 and pg.result.curves == []
