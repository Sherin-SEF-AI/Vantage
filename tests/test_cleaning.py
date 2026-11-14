"""core.cleaning: the spray-cone geometry against closed forms, the clearing, duty and tank runtime, the heater's heat
balance and its fit to the maker's measurements, the air jet, power and harness hooks, settings, the sample's Data rig
and the page driven like a user."""
import csv
import io
import json
import math

import numpy as np
import pytest

from vantage.core import cleaning as CL


class _Ctx:
    quick = True


# =========================================================================== geometry
def test_validation_checks_pass():
    checks = CL.validation_checks()
    assert {c['id'] for c in checks} == {'cleaning.spray_cone', 'cleaning.convection', 'cleaning.heater_econ'}
    for c in checks:
        out = c['fn'](_Ctx())
        assert math.isfinite(out.error) and out.error <= c['tolerance'] + 1e-12 and out.ok, (c['id'], out.detail)


def test_cone_square_on_a_disc_is_a_circle():
    win = CL._plane_window(0.02, 40000)
    for h, al in ((0.01, 30.0), (0.05, 10.0), (0.02, 60.0)):
        cov = CL.cone_cover(np.array([0.0, 0.0, h]), np.array([0.0, 0.0, -1.0]), al, win.points, win.normals)
        r = h * math.tan(math.radians(al))
        assert win.dA[cov].sum() == pytest.approx(math.pi * min(r, 0.02) ** 2, rel=5e-3)
    # behind the window, or out of reach, nothing
    assert not CL.cone_cover(np.array([0.0, 0.0, -0.01]), np.array([0.0, 0.0, 1.0]), 40, win.points,
                             win.normals).any()
    assert not CL.cone_cover(np.array([0.0, 0.0, 0.05]), np.array([0.0, 0.0, -1.0]), 40, win.points, win.normals,
                             reach=0.04).any()


def test_ellipse_section_by_brute_force():
    h, al, be = 0.01, 12.0, 35.0
    A, B, off = CL.ellipse_section(h, al, be)
    b = math.radians(be)
    axis = np.array([math.sin(b), 0.0, -math.cos(b)])
    # the two ends of the major axis lie on the cone's surface
    foot = np.array([h * math.tan(b), 0.0])
    for s in (-1, 1):
        P = np.array([foot[0] + off + s * A, 0.0, 0.0]) - np.array([0.0, 0.0, h])
        assert math.degrees(math.acos(P @ axis / np.linalg.norm(P))) == pytest.approx(al, abs=1e-6)
    P = np.array([foot[0] + off, B, 0.0]) - np.array([0.0, 0.0, h])
    assert math.degrees(math.acos(P @ axis / np.linalg.norm(P))) == pytest.approx(al, abs=1e-6)


def test_capture_share_of_a_disc_square_on():
    R, h = 0.02, 0.03
    win = CL._plane_window(R, 40000)
    apex, axis = np.array([0.0, 0.0, h]), np.array([0.0, 0.0, -1.0])
    al = math.degrees(math.atan(R / h)) + 5.0                  # the cone covers the whole disc and more
    cov = CL.cone_cover(apex, axis, al, win.points, win.normals)
    assert cov.all()
    omega_disc = 2 * math.pi * (1 - h / math.hypot(h, R))
    want = omega_disc / (2 * math.pi * (1 - math.cos(math.radians(al))))
    assert CL.capture_share(apex, axis, al, win, cov) == pytest.approx(want, rel=5e-3)


def test_where_a_nozzle_misses_and_the_camera_window():
    from vantage.core import samples
    p = samples.tigor_sample_project()
    cam = p.variant.cameras[0]
    win = CL.camera_window(p, cam)
    assert win.kind == 'disc' and win.area == pytest.approx(math.pi * win.size[0] ** 2)
    assert win.R[:, 2] @ np.array([math.cos(math.radians(cam.yaw)), math.sin(math.radians(cam.yaw)), 0]) > 0.5
    assert win.R[:, 1][2] > 0                                   # v is up
    # a narrow cone from just above, aimed at the centre, grazes past the lens' top: it misses the top
    dev = CL.clean_device({'kind': 'washer', 'target': cam.uid, 'side': 'top', 'standoff_mm': 5, 'ahead_mm': 3,
                           'params': {'spray_deg': 30.0}})
    pos, axis = CL.nozzle_pose(win, dev)
    assert (pos - win.centre) @ win.R[:, 1] > win.size[0]       # above the lens
    cov = CL.cone_cover(pos, axis, 15.0, win.points, win.normals)
    share, where = CL.where_missed(win, cov)
    assert 0 < share < 1 and where == 'top'
    # from further ahead, aimed at the near rim: the far side (bottom) is what it misses
    far = dict(dev, ahead_mm=20.0, aim_mm=-0.8 * win.size[0] * 1000)
    pos2, ax2 = CL.nozzle_pose(win, far)
    assert CL.where_missed(win, CL.cone_cover(pos2, ax2, 15.0, win.points, win.normals))[1] == 'bottom'
    dev = dict(dev, side='left')
    pos, _ = CL.nozzle_pose(win, dev)
    assert (pos - win.centre) @ win.R[:, 0] < -win.size[0]      # left as the image is seen


def test_lidar_band_window_and_nozzles_round_it():
    from vantage.core import samples
    p = samples.tigor_sample_project()
    p.active_uid = next(v for v in p.variants if v.sensors).uid
    s = next(x for x in p.variant.sensors if x.kind == 'lidar')
    win = CL.lidar_window(p, s)
    m = p.sensor_model_for(s)
    assert win.kind == 'band' and win.size[0] == pytest.approx(m.size_mm[0] / 2000)
    assert win.size[1] < 0 < win.size[2]
    assert np.allclose(np.linalg.norm(win.normals, axis=1), 1)
    one = CL.clean_device({'kind': 'washer', 'target': s.uid, 'standoff_mm': 80, 'ahead_mm': 15})
    four = [dict(one, azimuth_deg=a) for a in (0, 90, 180, 270)]
    cov1 = np.zeros(len(win.points), bool)
    cov4 = np.zeros(len(win.points), bool)
    for i, d in enumerate(four):
        pos, ax = CL.nozzle_pose(win, d)
        c = CL.cone_cover(pos, ax, 20.0, win.points, win.normals, 0.4)
        cov4 |= c
        if i == 0:
            cov1 = c
            # the first nozzle covers the band's front
            assert np.degrees(np.abs(((win.uv[c, 0] + math.pi) % (2 * math.pi)) - math.pi)).max() < 90
    assert win.dA[cov4].sum() > 0.9 * win.dA.sum() and win.dA[cov4].sum() > 2.5 * win.dA[cov1].sum()


# =========================================================================== physics
def test_air_properties_and_convection():
    a = CL.air_props(300.0)
    assert a['rho'] == pytest.approx(1.177, rel=5e-3) and a['nu'] == pytest.approx(15.89e-6, rel=0.02)
    assert a['k'] == pytest.approx(0.0263, rel=0.02) and a['pr'] == pytest.approx(0.707, rel=0.02)
    # the regimes: laminar below transition equals mixed; turbulent is more; h ∝ U^½ laminar, U^0.8 turbulent
    k, nu, pr = a['k'], a['nu'], a['pr']
    assert CL.flat_plate_h(2.0, 0.05, k, nu, pr, 'mixed') == CL.flat_plate_h(2.0, 0.05, k, nu, pr, 'laminar')
    assert CL.flat_plate_h(30.0, 0.05, k, nu, pr, 'turbulent') > CL.flat_plate_h(30.0, 0.05, k, nu, pr, 'laminar')
    assert CL.flat_plate_h(8.0, 0.05, k, nu, pr, 'laminar') / CL.flat_plate_h(2.0, 0.05, k, nu, pr, 'laminar') == \
        pytest.approx(2.0)
    assert CL.flat_plate_h(32.0, 0.05, k, nu, pr) / CL.flat_plate_h(2.0, 0.05, k, nu, pr) == pytest.approx(16 ** 0.8)
    assert CL.flat_plate_h(0.0, 0.05, k, nu, pr) == 0.0


def test_lumped_fit_reproduces_the_makers_warmups():
    G, C = CL.fit_lumped()
    for P, t in CL.ECON_TESTS:
        assert (P / G) * (1 - math.exp(-t * G / C)) == pytest.approx(50.0, abs=1e-6)
    assert G == pytest.approx(CL.ECON_G0, abs=5e-4) and C == pytest.approx(CL.ECON_C, abs=0.1)
    assert all(P / G < 50.0 for P in CL.ECON_NOT_REACHED)


def test_heater_state():
    st = CL.CleaningSettings(ambient_c=-10.0, speed_kmh=0.0, ice_mm=1.0)
    still = CL.heater_state(5.5, CL.ECON_G0, CL.ECON_C, 6.25e-4, 0.025, st, 1.4e-4)
    assert still['steady_c'] == pytest.approx(-10 + 5.5 / CL.ECON_G0)
    # by hand: warm to 0 °C, then melt the ice with what is left
    G = CL.ECON_G0
    warm = -(CL.ECON_C / G) * math.log(1 - 10 * G / 5.5)
    melt = CL.RHO_ICE * 1e-3 * 1.4e-4 * (CL.C_ICE * 10 + CL.L_FUSION) / (5.5 - 10 * G)
    assert still['clear_s'] == pytest.approx(warm + melt)
    # faster, colder, thicker ice: slower; too little power: never
    temps = [CL.heater_state(5.5, G, CL.ECON_C, 6.25e-4, 0.025,
                             CL.CleaningSettings(ambient_c=-10, speed_kmh=v), 1.4e-4)['steady_c'] for v in (0, 30, 80, 130)]
    assert all(b < a for a, b in zip(temps, temps[1:]))
    assert CL.heater_state(0.5, G, CL.ECON_C, 6.25e-4, 0.025, st, 1.4e-4)['clear_s'] is None
    thick = CL.heater_state(5.5, G, CL.ECON_C, 6.25e-4, 0.025, CL.CleaningSettings(ambient_c=-10, speed_kmh=0,
                                                                                    ice_mm=3.0), 1.4e-4)
    assert thick['clear_s'] > still['clear_s']
    warm_day = CL.heater_state(5.5, G, CL.ECON_C, 6.25e-4, 0.025, CL.CleaningSettings(ambient_c=5.0), 1.4e-4)
    assert warm_day['clear_s'] == 0.0


def test_air_jet():
    j = CL.air_jet(1.0, 4.0)
    p0 = 4e5 + CL.P_ATM
    A = math.pi * 1e-6 / 4
    g = 1.4
    mdot = 0.8 * A * p0 * math.sqrt(g / (287.05 * 288.15)) * (2 / (g + 1)) ** ((g + 1) / (2 * (g - 1)))
    assert j['mdot_kg_s'] == pytest.approx(mdot)
    assert j['exit_ms'] == pytest.approx(math.sqrt(g * 287.05 * 288.15 * 2 / (g + 1)))
    assert j['nl_min'] == pytest.approx(mdot / 1.2922 * 60000)
    assert j['nl_min'] < 210.0                       # within the SPV22 valve's 210 Nl/min at 4 bar
    # centreline decay ∝ 1/x, Gaussian across
    apex, axis = np.zeros(3), np.array([1.0, 0, 0])
    P = np.array([[0.05, 0, 0], [0.10, 0, 0], [0.10, CL.JET_S * 0.10, 0]])
    u = CL.jet_speed(j, apex, axis, P)
    assert u[0] / u[1] == pytest.approx(2.0) and u[2] / u[1] == pytest.approx(0.5)
    assert CL.jet_speed(j, apex, axis, np.array([[-0.1, 0, 0]]))[0] == 0.0
    sub = CL.air_jet(1.0, 0.3)                       # below the critical ratio: subsonic
    assert sub['exit_ms'] < j['exit_ms'] and sub['mdot_kg_s'] < mdot
    assert CL.compressor_w(j, 0.5) == pytest.approx(mdot * 287.05 * 288.15 * math.log(p0 / CL.P_ATM) / 0.5)


# =========================================================================== settings
def test_settings_round_trip_and_junk():
    st = CL.CleaningSettings(devices=[CL.clean_device({'kind': 'washer', 'target': 'abc', 'params': {'flow_ml_min': 300}})],
                             tank_l=4.0, dirt='mud', rates={'mud': 1.0})
    assert CL.CleaningSettings.from_dict(st.to_dict()) == st
    junk = CL.CleaningSettings.from_dict({
        'devices': [{'kind': 'washer', 'target': 'a', 'id': 'X'}, {'kind': 'washer', 'target': 'b', 'id': 'X'},
                    {'kind': 'laser'}, {'kind': 'heater'}, 'zz',
                    {'kind': 'heater', 'target': 'c', 'product': 'araymond_tele10', 'params': {'flow_ml_min': 9,
                                                                                              'power_w': 'x'}}],
        'tank_l': -1, 'dirt': 'lava', 'regime': 7, 'pump': 'nope', 'reservoir': [1, 'a', 2],
        'rates': {'mud': 'x', 'dust': 99999}, 'ambient_c': float('inf')})
    assert [d['id'] for d in junk.devices] == ['X', 'C1', 'C6']
    h = junk.devices[2]
    assert h['kind'] == 'heater' and h['product'] == 'custom_heater' and h['params'] == {}
    assert junk.tank_l == 0.1 and junk.dirt == 'spray' and junk.regime == 'turbulent' and junk.pump == 'hella_1l'
    assert junk.reservoir == [] and junk.rates == {'dust': 100.0} and junk.ambient_c == -10.0
    assert CL.CleaningSettings.from_dict(None) == CL.CleaningSettings()
    v, src = CL.figure(st.devices[0], 'flow_ml_min')
    assert (v, src) == (300.0, 'user')
    assert CL.figure(st.devices[0], 'ref_bar') == (2.0, 'user')        # a custom nozzle: every figure is yours
    lib = CL.clean_device({'kind': 'washer', 'product': 'araymond_tele35', 'target': 'abc'})
    assert CL.figure(lib, 'flow_ml_min') == (900.0, 'datasheet') and CL.figure(lib, 'spray_deg') == (40.0, 'assumed')


# =========================================================================== the sample's Data rig
@pytest.fixture(scope='module')
def rig():
    from vantage.core import samples
    p = samples.tigor_sample_project()
    p.active_uid = next(v for v in p.variants if v.sensors).uid
    scores = CL.dirt_scores(p)
    p.analysis.cleaning = {'devices': CL.propose_layout(p, scores)}
    CL.clear_cache()
    res = CL.run(p, scores=scores)
    return p, scores, res


def test_proposed_layout_on_the_data_rig(rig):
    p, scores, res = rig
    tags = {t['tag']: t for t in res.targets}
    assert {'F', 'R', 'L1'} <= set(tags)                 # moderate dirt or more (sample: F 0.39, R 0.50, L1 0.28)
    assert all(t['heaters'] for t in res.targets if t['kind'] == 'camera')
    assert tags['F']['wash_cover'] == pytest.approx(1.0) and tags['L1']['wash_cover'] > 0.9
    assert len(tags['L1']['washers']) == 4
    t = res.totals
    assert t['runtime_h'] == pytest.approx(t['tank_ml'] / t['fluid_ml_h'])
    assert t['heaters_w'] == pytest.approx(5.5 * 6)
    for d in res.devices:
        if d['kind'] == 'heater':
            assert d['clears'] and d['steady_c'] > 0             # 5.5 W at −10 °C and 80 km/h
    json.dumps(CL.summary(res))


def test_runtime_scales_with_tank_rate_and_dose(rig):
    p, scores, base = rig
    st = CL.settings_of(p)

    def run(**kw):
        d = st.to_dict()
        d.update(kw)
        return CL.run(p, settings=CL.CleaningSettings.from_dict(d), scores=scores).totals
    t0 = base.totals
    assert run(tank_l=3.0)['runtime_h'] == pytest.approx(2 * t0['runtime_h'])
    assert run(rates={'spray': 0.6})['runtime_h'] == pytest.approx(t0['runtime_h'] / 2)
    assert run(max_obscured=0.10)['runtime_h'] == pytest.approx(2 * t0['runtime_h'])
    assert run(dirt='dust')['runtime_h'] > t0['runtime_h']
    short = run(tank_l=0.1)
    assert not short['tank_ok']
    res = CL.run(p, settings=CL.CleaningSettings.from_dict(dict(st.to_dict(), tank_l=0.1)), scores=scores)
    assert any(f['kind'] == 'tank' and f['severity'] == 'serious' for f in res.findings)


def test_flow_follows_the_orifice_law_and_the_pump(rig):
    p, scores, _ = rig
    cam = p.variant.cameras[0]
    devs = [{'kind': 'washer', 'product': 'araymond_tele35', 'target': cam.uid}]
    r = CL.run(p, settings=CL.CleaningSettings.from_dict({'devices': devs}), scores=scores)
    assert r.devices[0]['flow_ml_min'] == pytest.approx(900 * math.sqrt(2.0 / 2.5))
    r = CL.run(p, settings=CL.CleaningSettings.from_dict({'devices': devs, 'pump': 'custom',
                                                          'pump_flow_ml_min': 300}), scores=scores)
    assert r.devices[0]['flow_ml_min'] == pytest.approx(300.0)


def test_findings_name_the_miss(rig):
    p, scores, _ = rig
    qr = next(c for c in p.variant.cameras if c.tag == 'QR')
    devs = [{'id': 'W1', 'kind': 'washer', 'target': qr.uid, 'side': 'top', 'standoff_mm': 5, 'ahead_mm': 3,
             'params': {'spray_deg': 30}},
            {'id': 'H1', 'kind': 'heater', 'product': 'econ_5w5', 'target': qr.uid},
            {'id': 'Z9', 'kind': 'washer', 'target': 'gone'}]
    st = CL.CleaningSettings.from_dict({'devices': devs, 'ambient_c': -25, 'speed_kmh': 160, 'wind_ms': 10})
    r = CL.run(p, settings=st, scores=scores)
    msgs = [f['message'] for f in r.findings]
    assert any(m.startswith("QR's washer cone (W1) misses") and 'mostly its top' in m for m in msgs)
    assert any('cannot clear ice' in m and 'QR' in m for m in msgs)
    assert any('Z9' in m for m in msgs)
    assert r.device('Z9')['problem']
    iss = CL.issues(_with(p, st))
    assert any('cannot clear ice' in i.message for i in iss)


def _with(p, st):
    q = p.snapshot()
    q.analysis.cleaning = st.to_dict()
    CL.run(q, scores=CL.dirt_scores(q))
    return q


def test_air_nozzle_on_a_camera(rig):
    p, scores, _ = rig
    f = next(c for c in p.variant.cameras if c.tag == 'F')
    st = CL.CleaningSettings.from_dict({'devices': [{'kind': 'air', 'product': 'air_spv22', 'target': f.uid}]})
    r = CL.run(p, settings=st, scores=scores)
    d = r.devices[0]
    assert 0.5 < d['cover'] <= 1.0 and d['speed_at_centre_ms'] > st.shed_ms
    win = r.windows[f.uid][0]
    assert d['speed_at_centre_ms'] == pytest.approx(float(CL.jet_speed(d['jet'], np.array(d['nozzle']),
                                                                       np.array(d['aim']), win.centre[None])[0]))
    assert d['compressor_w'] > 0 and any('compressor' in row[0] for row in r.power_rows)


def test_power_and_harness_hooks(rig):
    p, scores, res = rig
    from vantage.core import power as PW
    rows = CL.power_loads(p, res)
    assert all(len(r) == 5 and r[0].startswith('Cleaning:') and r[3] in PW.CIRCUITS for r in rows)
    assert sum(r[1] for r in rows) == pytest.approx(res.totals['average_w'])
    q = p.snapshot()
    q.analysis.cleaning = {}                         # the budget without the cleaning (core.power adds its loads)
    base = PW.run(q)
    assert PW.run(p).totals['steady_w'] == pytest.approx(base.totals['steady_w'] + res.totals['average_w'])
    withc = CL.power_budget(p, res)
    assert withc.totals['steady_w'] == pytest.approx(base.totals['steady_w'] + res.totals['average_w'])
    assert withc.totals['runtime_min'] < base.totals['runtime_min']
    tubes = [h for h in res.harness if h['what'] == 'tubing']
    assert len(tubes) == res.totals['washers'] and all(h['length_m'] > 0 for h in res.harness)
    assert len([h for h in res.harness if h['what'] == 'heater wiring']) == res.totals['heaters']


def test_sotif_reads_the_washer_cover(rig):
    p, scores, res = rig
    cov = CL.washer_cover(p)
    tags = {c.uid: c.tag for c in p.variant.cameras}
    assert cov and {tags.get(u, '') for u in cov} >= {'F', 'R'}
    assert all(0 <= v <= 1 for v in cov.values())


def test_exports(rig):
    p, scores, res = rig
    names = dict(CL.files(res, p))
    assert set(names) == {'cleaning.json', 'cleaning.csv', 'cleaning.png'}
    doc = json.loads(names['cleaning.json'])
    assert doc['schema'] == CL.SCHEMA and doc['provenance']['schema'] == 'vantage.provenance/1'
    assert all(v['source'] == 'assumed' for v in doc['dirt_kinds'].values())
    assert len(doc['devices']) == len(res.devices) and doc['clearing']
    rows = list(csv.reader(io.StringIO(names['cleaning.csv'].decode())))
    assert tuple(rows[0]) == CL.CSV_COLUMNS and len(rows) == len(res.devices) + 1
    assert names['cleaning.png'][:8] == b'\x89PNG\r\n\x1a\n'


def test_no_devices_and_hidden_target():
    from vantage.core import samples
    p = samples.tigor_sample_project()
    r = CL.run(p, scores={})
    assert r.devices == [] and r.targets == [] and r.totals['average_w'] == 0 and r.findings == []
    cam = p.variant.cameras[0]
    cam.visible = False
    r = CL.run(p, settings=CL.CleaningSettings.from_dict({'devices': [{'kind': 'washer', 'target': cam.uid}]}),
               scores={})
    assert r.devices[0]['problem'] == 'Its target is hidden.'


# =========================================================================== the page
from test_gui import app, win, pump, settle, _wait   # noqa: E402,F401 (fixtures)


def _open(win, name, cls):
    if name not in win._page_widgets:
        pg = cls(win.ctx)
        win.pages.addWidget(pg)
        win._page_widgets[name] = pg
    win.set_page(name)
    return win._page_widgets[name]


def test_page(app, win, tmp_path):
    from vantage.ui.workspaces.cleaning_ws import CleaningPage
    uid = next(v for v in win.store.project.variants if v.sensors).uid

    def fn(p):
        p.active_uid = uid
    win.store.edit('variants', fn, 'Data rig')
    settle(app, win)
    pg = _open(win, 'cleaning', CleaningPage)
    pg.activate()
    _wait(app, lambda: pg.result is not None and not pg.busy, timeout=60)
    assert pg.t_dev.rowCount() == 0 and pg.tiles['cover'].value == '—'
    # add a washer on F like a user: kind, target, product, Add
    pg.c_kind.setCurrentIndex(list(CL.KINDS).index('washer'))
    pg.c_target.setCurrentIndex(pg.c_target.findText('F'))
    pg.c_prod.setCurrentIndex(pg.c_prod.findData('araymond_tele10'))
    pg.b_add.click()
    assert len(CL.settings_of(win.store.project).devices) == 1 and pg.selected == 'C1'
    _wait(app, lambda: pg.result is not None and not pg.busy and len(pg.result.devices) == 1, timeout=60)
    assert pg.t_res.rowCount() == 1 and pg.tiles['cover'].value == '100'
    # move it: standoff and aim; the cover drops; undo restores
    f, _ = pg.dev_fields['standoff_mm']
    f.committed.emit(5.0)
    pg._set_device_key('ahead_mm', 3.0)
    sp, _ = pg.fig_fields['spray_deg']
    sp.committed.emit(30.0)
    d0 = CL.settings_of(win.store.project).devices[0]
    assert d0['standoff_mm'] == 5.0 and d0['ahead_mm'] == 3.0 and d0['params'] == {'spray_deg': 30.0}
    _wait(app, lambda: pg.result is not None and not pg.busy and not pg.stale and
          pg.result.devices[0]['figures']['spray_deg']['value'] == 30.0, timeout=60)
    assert pg.result.devices[0]['cover'] < 0.9
    assert any('misses' in f['message'] and 'its top' in f['message'] for f in pg.result.findings) and pg.findings
    for _ in range(3):
        win.store.undo.undo()
    pump(app, 0.2)
    d0 = CL.settings_of(win.store.project).devices[0]
    assert d0['standoff_mm'] == 15.0 and d0['ahead_mm'] == 10.0 and d0['params'] == {}
    # a user figure: the flow, marked as the user's
    ff, _ = pg.fig_fields['flow_ml_min']
    ff.committed.emit(400.0)
    assert CL.settings_of(win.store.project).devices[0]['params'] == {'flow_ml_min': 400.0}
    _wait(app, lambda: pg.result is not None and not pg.busy and pg.result.devices[0]['figures']['flow_ml_min']
          ['source'] == 'user', timeout=60)
    # propose a layout, then look at it in 3D and on the window map
    n = pg.propose()
    assert n == len(CL.settings_of(win.store.project).devices) > 6
    _wait(app, lambda: pg.result is not None and not pg.busy and len(pg.result.devices) == n, timeout=60)
    assert pg.t_res.rowCount() == n and pg.t_harn.rowCount() > 0 and pg.t_clear.rowCount() > 0
    assert len(pg.map.items) == 3
    pg._set_view('3d')
    pump(app, 0.3)
    lines, pts = __import__('vantage.ui.workspaces.cleaning_ws', fromlist=['x']).device_draws(
        win.store.project, pg.result)
    assert lines and pts and len(pts[0].points) > 0
    pg.t_dev.selectRow(0)
    pump(app, 0.1)
    assert pg.selected == CL.settings_of(win.store.project).devices[0]['id']
    pg.b_remove.click()
    assert len(CL.settings_of(win.store.project).devices) == n - 1
    pg.c_pump.setCurrentIndex(list(CL.PUMPS).index('hella_hcs'))
    assert CL.settings_of(win.store.project).pump == 'hella_hcs'
    _wait(app, lambda: pg.result is not None and not pg.busy and not pg.stale, timeout=60)
    paths = pg.export(str(tmp_path))
    assert {x.rsplit('/', 1)[-1] for x in paths} == {'cleaning.json', 'cleaning.csv', 'cleaning.png'}
    pg._set_view('map')
    pg.b_defaults.click()
    assert win.store.project.analysis.cleaning == {}
