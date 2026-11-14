"""core.aero: air, coefficients, cross-flow, shielding, the road-load range, the cylinder and roof-rack validations,
the sample's rig with lock-in against beam and FE modes, the wind load case into core.thermo_vib, the page."""
import json
import math

import numpy as np
import pytest

from vantage.core import aero as AE
from vantage.core import thermo_vib as TV
from test_framing import scene


def test_air_isa():
    rho, nu = AE.air(15.0, 101.325)
    assert rho == pytest.approx(1.225, rel=1e-3)                # ISA sea level
    assert nu == pytest.approx(1.461e-5, rel=5e-3)              # μ 1.789e-5 Pa s at 15 °C (ISA)


def test_coefficient_tables():
    assert AE.k_aspect(1) == pytest.approx(0.63) and AE.k_aspect(40) == pytest.approx(0.98)
    assert AE.k_aspect(1e12) == pytest.approx(1.0)
    assert AE.cd_circle(1e4) == 1.2 and AE.cd_circle(1e6) == 0.3 and 0.3 < AE.cd_circle(3e5) < 1.2
    assert AE.cd_rect(1.0) == pytest.approx(2.0) and AE.cd_box(1.0) == pytest.approx(1.07)
    assert AE.tandem(1.0) == 0 and AE.tandem(100) == 1


def _bar(axis=(0, 1.0, 0), L=1.0, D=0.04, kind='prism', walls=0, ends=(False, False)):
    a = np.asarray(axis, float)
    ex = np.cross(a, [0, 0, 1.0]) if abs(a[2]) < 0.9 else np.array([1.0, 0, 0])
    ex /= np.linalg.norm(ex)
    return AE.Element('b', 'b', kind, np.zeros(3), axis=a, ex=ex, b=D, d=D, L=L, walls=walls, free_ends=ends)


def test_crossflow_principle():
    rho, nu = AE.air()
    el = _bar(L=2.0)
    F0 = AE.element_force(el, np.array([-30.0, 0, 0]), rho, nu)
    q = 0.5 * rho * 900
    k = AE.k_aspect(2.0 / 0.04)
    assert F0['parts']['pressure_n'] == pytest.approx(q * 2.0 * k * 0.04 * 2.0, rel=1e-9)
    # yawed 30°: the normal component's dynamic pressure falls as cos²
    b = math.radians(30)
    F1 = AE.element_force(el, np.array([-30.0 * math.cos(b), -30.0 * math.sin(b), 0]), rho, nu)
    assert F1['parts']['pressure_n'] == pytest.approx(F0['parts']['pressure_n'] * math.cos(b) ** 2, rel=1e-9)
    # a vertical cylinder sees the same drag from any heading
    cyl = _bar((0, 0, 1.0), L=0.5, D=0.06, kind='cylinder', walls=1)
    a = AE.element_force(cyl, np.array([-30.0, 0, 0]), rho, nu)['F']
    c = AE.element_force(cyl, np.array([0, -30.0, 0]), rho, nu)['F']
    assert np.linalg.norm(a) == pytest.approx(np.linalg.norm(c))
    # a wall at one end doubles the effective L/D
    assert a[0] < 0 and AE.element_force(_bar((0, 0, 1.0), 0.5, 0.06, 'cylinder', 0), np.array([-30.0, 0, 0]),
                                         rho, nu)['cd'] < AE.element_force(cyl, np.array([-30.0, 0, 0]), rho, nu)['cd']


def test_drag_scales_with_speed_squared():
    rho, nu = AE.air()
    el = _bar()
    a = AE.element_force(el, np.array([-20.0, 0, 0]), rho, nu)['F'][0]
    b = AE.element_force(el, np.array([-40.0, 0, 0]), rho, nu)['F'][0]
    assert b / a == pytest.approx(4.0, rel=1e-9)


def test_shielding_in_tandem():
    rho, nu = AE.air()
    st = AE.AeroSettings()
    rf = {'x_front': 10.0, 'top': -10.0}
    a = AE.Element('a', 'a', 'cylinder', np.array([0.0, 0, 0]), axis=np.array([0, 0, 1.0]),
                   ex=np.array([1.0, 0, 0]), b=0.1, d=0.1, L=1.0)
    b = AE.Element('b', 'b', 'cylinder', np.array([-0.4, 0, 0]), axis=np.array([0, 0, 1.0]),
                   ex=np.array([1.0, 0, 0]), b=0.1, d=0.1, L=1.0)
    f = AE.forces([a, b], np.array([-30.0, 0, 0]), rho, nu, rf, st)
    assert f[0]['shield'] == 1.0 and f[1]['shield'] == pytest.approx(AE.tandem(4.0))
    assert f[1]['drag_n'] == pytest.approx(f[0]['drag_n'] * AE.tandem(4.0))


def test_boundary_layer():
    rho, nu = AE.air()
    assert AE.bl_factor(10.0, 1.0, 30.0, nu) == 1.0
    small = AE.bl_factor(0.002, 2.0, 30.0, nu)
    assert 0 < small < 0.8


def test_range_formula():
    st = AE.AeroSettings(crr=0.01, drive_eff=0.8, aux_w=0.0, battery_kwh=50)
    rho = 1.2
    V = 30.0
    base = AE.consumption(V, rho, 0.6, 1500, st)
    hand = (0.01 * 1500 * AE.G + 0.5 * 1.2 * 0.6 * 900) / 0.8 / 3.6
    assert base == pytest.approx(hand)
    row = AE._range_row(108.0, 50.0, rho, 0.6, 1500, 0.0, st)
    assert row['range_loss_pct'] == pytest.approx(100 * (1 - hand / (hand + 50 / 0.8 / 3.6)))
    assert row['range_km'] == pytest.approx(50000 / hand)


def test_settings_tolerant():
    s = AE.AeroSettings.from_dict({'speed1_kmh': 'fast', 'crosswind_ms': 99, 'use_frame': 'no', 'crr': -1})
    assert s.speed1_kmh == 100 and s.crosswind_ms == 40 and s.use_frame is True and s.crr == 0.003


def test_validation_checks():
    for c in AE.validation_checks():
        out = c['fn']()
        assert out['error'] <= c['tolerance'], (c['id'], out)
    r = AE.generic_rack_loss()
    assert 2.0 <= r['loss_pct'] <= 11.0


# ------------------------------------------------------------------ the sample
@pytest.fixture(scope='module')
def res():
    p = scene()
    return p, AE.run(p)


def test_sample_rig(res):
    p, r = res
    assert {e['kind'] for e in r.elements} >= {'prism', 'plate', 'box', 'cylinder'}
    s1, s2 = r.speeds
    assert s2['drag_n'] > s1['drag_n'] > 0
    assert s1['rig_cda_m2'] == pytest.approx(s2['rig_cda_m2'], rel=0.05)
    assert 0 < s1['range_loss_pct'] < s2['range_loss_pct'] < 60
    assert all(np.diff(r.curve['drag_n']) > 0)
    # drag is the sum of the elements'
    assert s2['drag_n'] == pytest.approx(sum(e['at']['130']['drag_n'] for e in r.elements))


def test_load_case_into_thermo_vib(res):
    p, r = res
    lc = r.load_case
    assert lc['yaw_deg'] == pytest.approx(math.degrees(math.atan2(15.0, 130 / 3.6)))
    assert lc['total_n'][0] < 0 and lc['total_n'][1] < 0
    l1 = next(d for d in lc['devices'] if d['tag'] == 'L1')
    assert l1['force_n'] > 0 and l1['move_abs_mm'] > 0 and l1['move_mm'][1] < 0


def test_wind_response_linear_and_zero():
    p = scene()
    L1 = next(s for s in p.variant.sensors if s.tag == 'L1')
    z = TV.wind_response(p, {})
    assert all(d['move_abs_mm'] == 0 for d in z['devices'])
    a = TV.wind_response(p, {L1.uid: (0, -10.0, 0)})
    b = TV.wind_response(p, {L1.uid: (0, -20.0, 0)})
    da = next(d for d in a['devices'] if d['uid'] == L1.uid)
    db = next(d for d in b['devices'] if d['uid'] == L1.uid)
    assert db['move_mm'][1] == pytest.approx(2 * da['move_mm'][1]) and da['move_mm'][1] < 0
    # through a mast's own spring: at least F/k of the mast's cantilever
    mast = next(q for q in p.rig if q.kind == 'cylinder')
    k = 3 * TV.MATERIALS['al6061']['E'] * TV.tube_I(mast.radius) / mast.height ** 3
    assert abs(da['move_mm'][1]) >= 10.0 / k * 1000 * 0.999


def test_shedding_beam_and_fe_modes(res):
    p, r = res
    assert 'thermo_vib' in r.modes_source
    assert all(s['lock_kmh'] > 0 for s in r.shedding)
    # St·V/D at the lock-in speed equals the mode
    for s in r.shedding:
        assert s['strouhal'] * s['lock_kmh'] / 3.6 / s['D_m'] == pytest.approx(s['f_hz'], rel=0.3)
    from vantage.core import fea as FE
    fr = FE.run(p, settings=FE.FEASettings(mesh_mm=80, plate_cells_mm=50, modes=4))
    r2 = AE.run(p, fea_result=fr)
    assert 'core.fea' in r2.modes_source and r2.shedding
    assert any(s['mode'].startswith('FE mode') for s in r2.shedding)


def test_outputs(res, tmp_path):
    p, r = res
    AE.export(r, p, str(tmp_path))
    doc = json.load(open(tmp_path / 'aero.json'))
    assert doc['schema'] == AE.SCHEMA and 'not CFD' in doc['not_cfd']
    assert doc['sources']['k_ar']
    assert (tmp_path / 'aero.png').read_bytes()[:4] == b'\x89PNG'
    json.dumps(AE.summary(r))


def test_rig_as_drawn_and_empty():
    p = scene()
    r = AE.run(p, settings=AE.AeroSettings(use_frame=False))
    assert any(e['kind'] == 'cylinder' for e in r.elements)
    from vantage.core import samples
    q = samples.tigor_sample_project()
    e = AE.run(q)
    assert e.elements == [] or all(x['kind'] != 'prism' for x in e.elements)


def test_issues_cached():
    p = scene()
    a = AE.issues(p)
    b = AE.issues(p)
    assert [i.message for i in a] == [i.message for i in b]


# ------------------------------------------------------------------ the page
from test_gui import app, win, pump, settle, _wait   # noqa: E402,F401  (fixtures)


def _open(win, name, cls):
    if name not in win._page_widgets:
        pg = cls(win.ctx)
        win.pages.addWidget(pg)
        win._page_widgets[name] = pg
    win.set_page(name)
    return win._page_widgets[name]


def test_page(app, win, tmp_path):
    from vantage.core.rig import apply_fit, fit_plan
    from vantage.ui.workspaces.aero_ws import AeroPage
    s = win.ctx.store

    def fit(p):
        p.active_uid = next(v for v in p.variants if v.sensors).uid
        for k in ('roof_rack', 'sensor_mast'):
            apply_fit(p, fit_plan(p, k))
    s.edit(['rig', 'variant'], fit, 'fit a rack and a mast')
    pg = _open(win, 'aero_test', AeroPage)
    pg.activate()
    pump(app, 0.3)
    assert pg.result is not None and pg.t_range.rowCount() == 2 and pg.t_drag.rowCount() > 5
    loss0 = pg.result.speeds[-1]['range_loss_pct']
    pg.fields['speed2_kmh'].committed.emit(150.0)
    pump(app, 0.3)
    assert AE.settings_of(s.project).speed2_kmh == 150.0
    assert pg.result.speeds[-1]['range_loss_pct'] > loss0
    s.undo.undo()
    pump(app, 0.3)
    assert AE.settings_of(s.project).speed2_kmh == 130.0
    pg.t_frame.click()
    pump(app, 0.3)
    assert AE.settings_of(s.project).use_frame is False
    paths = pg.export(str(tmp_path))
    assert any(x.endswith('aero.json') for x in paths)
