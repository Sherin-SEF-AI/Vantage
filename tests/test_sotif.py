"""core.sotif: the stop models against independent references, the ODD bound's monotonicity, the corridor geometry,
the matrix on the sample's Data rig (every cell traced to a study), the requirements trace, exports, the cache and
the page driven like a user."""
import csv
import io
import json
import math

import numpy as np
import pytest

from vantage.core import sotif as SO


class _Ctx:
    quick = True


# =========================================================================== stopping distance
@pytest.mark.parametrize('model', ['r157', 'r157_fsm', 'aashto'])
@pytest.mark.parametrize('road', SO.ROADS)
def test_stopping_distance_matches_a_time_stepped_integration(model, road):
    stop = SO.stop_of(SO.SotifSettings(stop_model=model), road)
    for kph in (3.0, 12.0, 50.0, 100.0):
        v = kph / 3.6
        assert SO.stopping_distance(v, stop) == pytest.approx(SO._integrate_stop(v, stop, dt=2e-5), abs=2e-3)


def test_stopping_distance_closed_forms_by_hand():
    # no jerk: v·t + v²/2a
    s = SO.Stop(2.5, 11.2 * 0.3048)
    v = 50 * 0.44704
    assert SO.stopping_distance(v, s) == pytest.approx(v * 2.5 + v * v / (2 * 11.2 * 0.3048))
    # R157: 1.15 s delay, 0.774 g reached in 0.6 s; at 50 km/h the main branch
    st = SO.SotifSettings()
    stop = SO.stop_of(st, 'dry')
    assert stop.delay_s == pytest.approx(1.15) and stop.t_j == pytest.approx(0.6)
    a, tj, v = 0.774 * SO.G, 0.6, 50 / 3.6
    want = v * 1.15 + v * tj - a * tj ** 2 / 6 + (v - a * tj / 2) ** 2 / (2 * a)
    assert SO.stopping_distance(v, stop) == pytest.approx(want)
    # wet road: μ 0.5 caps the deceleration; the jerk stays the model's, so the ramp is shorter
    w = SO.stop_of(st, 'wet')
    assert w.decel == pytest.approx(0.5 * SO.G) and w.t_j == pytest.approx(0.5 * SO.G / (0.774 * SO.G / 0.6))
    # the FSM's jerk equals R157 Table 1's 0.774 g in 0.6 s to the printed digits
    assert SO.STOP_MODELS['r157_fsm']['jerk_ms3'] == pytest.approx(SO.STOP_MODELS['r157']['jerk_ms3'], abs=0.01)


def test_max_safe_speed_inverts_the_stopping_distance_on_both_branches():
    for model in ('r157', 'r157_fsm', 'aashto', 'custom'):
        for road in SO.ROADS:
            stop = SO.stop_of(SO.SotifSettings(stop_model=model, jerk_ms3=2.0), road)
            for R in (0.05, 0.3, 2.0, 15.0, 80.0, 300.0):
                v = SO.max_safe_speed(R, stop)
                assert SO.stopping_distance(v, stop) == pytest.approx(R, abs=1e-6)
    stop = SO.stop_of(SO.SotifSettings(), 'dry')
    assert SO.max_safe_speed(0.0, stop) == 0.0 and SO.max_safe_speed(-3, stop) == 0.0
    assert SO.max_safe_speed(math.inf, stop) == math.inf


def test_the_odd_bound_is_monotone():
    st = SO.SotifSettings()
    # more range, never a lower speed; a slicker road, never a higher one; more delay, never a higher one
    for road in SO.ROADS:
        vs = [SO.max_safe_speed(R, SO.stop_of(st, road)) for R in np.linspace(0, 300, 61)]
        assert all(b >= a for a, b in zip(vs, vs[1:]))
    for R in (10.0, 60.0, 200.0):
        vs = [SO.max_safe_speed(R, SO.stop_of(st, road)) for road in ('dry', 'wet', 'snow', 'ice')]
        assert all(b <= a for a, b in zip(vs, vs[1:]))
        vs = [SO.max_safe_speed(R, SO.stop_of(st, 'dry', d)) for d in (0.0, 0.5, 1.0, 3.0)]
        assert all(b <= a for a, b in zip(vs, vs[1:]))
    # the ODD limit over more conditions is never higher
    rows = [{'id': f'c{i}', 'max_kmh': float(x)} for i, x in enumerate((90, 40, 70, 55, 120))]
    ids = [r['id'] for r in rows]
    lims = [SO.odd_speed(rows, ids[:k]) for k in range(1, 6)]
    assert lims == [90, 40, 40, 40, 40]
    assert SO.odd_speed(rows, []) is None


def test_validation_checks_pass():
    checks = SO.validation_checks()
    assert {c['id'] for c in checks} == {'sotif.stopping_closed_form', 'sotif.wsdot_ssd'}
    for c in checks:
        out = c['fn'](_Ctx())
        assert math.isfinite(out.error) and out.error <= c['tolerance'] + 1e-12 and out.ok, (c['id'], out.detail)
        assert set(c) >= {'id', 'model', 'title', 'quantity', 'reference', 'tolerance', 'unit', 'refs', 'fn', 'cost'}
    # the WSDOT table rows, independently: 2.5 s and 11.2 ft/s², rounded up to 5 ft
    for mph, ft in SO.WSDOT_1260_1.items():
        v = mph * 1.466667
        assert 5 * math.ceil((v * 2.5 + v * v / (2 * 11.2)) / 5) == ft


# =========================================================================== corridor and basis
def test_corridor_range_against_geometry():
    az = np.arange(360.0)
    centre, front = (1.0, 0.0), 3.0
    # a uniform far range R: the corridor ends where its edge (y = ±w) reaches R from the centre
    R, w = 50.0, 1.75
    d = SO.corridor_range(np.full(360, R), az, centre, front, w, step=0.01)
    want = math.sqrt(R * R - w * w) - (front - centre[0])
    assert d == pytest.approx(want, abs=0.02)
    # a blind wedge to the front-left at 2°..4°: the corridor stops where the wedge meets it (y = w)
    far = np.full(360, 200.0)
    far[2:5] = 0.0
    d = SO.corridor_range(far, az, centre, front, w, step=0.01)
    x_hit = w / math.tan(math.radians(4.5))          # the nearest corridor point seen inside the wedge (2–4°, ±0.5)
    assert front - centre[0] + d < x_hit + 0.05
    assert d > 0
    assert SO.corridor_range(np.zeros(360), az, centre, front, w) == 0.0
    assert SO.corridor_range(np.full(360, 1e6), az, centre, front, w, max_d=120) == 120


def test_combine_basis():
    cam, lid, rad = np.array([10.0, 50.0]), np.array([30.0, 20.0]), np.array([np.nan, 80.0])
    assert list(SO.combine(cam, lid, rad, 'any')) == [30.0, 80.0]
    assert list(SO.combine(cam, lid, rad, 'two')) == [10.0, 50.0]
    assert list(SO.combine(cam, lid, rad, 'cameras')) == [10.0, 50.0]
    assert list(SO.combine(cam, None, rad, 'two')) == [0.0, 50.0]
    assert list(SO.combine(cam, lid, rad, 'radars')) == [0.0, 80.0]


def test_settings_round_trip_and_junk():
    st = SO.SotifSettings(stop_model='aashto', target_kmh=80.0, basis='any', classes=['Car'], odd=['clear', 'fog_50'],
                          roads={'fog_50': 'wet'})
    assert SO.SotifSettings.from_dict(st.to_dict()) == st
    junk = SO.SotifSettings.from_dict({'stop_model': 'warp', 'target_kmh': 'fast', 'margin_m': -5, 'basis': 7,
                                       'classes': [None, '', 'Car', 'Car'], 'odd': ['nope', 'clear'],
                                       'roads': {'clear': 'lava', 'rain_25': 'dry', 'zzz': 'wet'},
                                       'decel_g': float('nan'), 'lane_half_m': 99})
    assert junk.stop_model == 'r157' and junk.target_kmh == 50.0 and junk.margin_m == 0.0 and junk.basis == 'two'
    assert junk.classes == ['Car'] and junk.odd == ['clear'] and junk.roads == {'rain_25': 'dry'}
    assert junk.decel_g == 0.774 and junk.lane_half_m == 6.0
    assert SO.SotifSettings.from_dict(None) == SO.SotifSettings()
    assert SO.SotifSettings.from_dict([1, 2]) == SO.SotifSettings()


# =========================================================================== the sample's Data rig
@pytest.fixture(scope='module')
def rig():
    from vantage.core import samples
    from vantage.core.analysis import Engine
    p = samples.tigor_sample_project()
    p.active_uid = next(v for v in p.variants if v.sensors).uid
    eng = Engine()
    SO.clear_cache()
    res = SO.run(p, eng)
    return p, eng, res


def test_every_cell_traces_to_a_study(rig):
    p, eng, res = rig
    known = {'perception', 'sensor_coverage', 'weather', 'exposure', 'volume', 'isp', 'thermal', 'event'}
    assert len(res.rows) >= 18 and res.metrics['cells'] > 60
    for r in res.rows:
        assert r['cells'], r['id']
        for c in r['cells']:
            assert c['verdict'] in SO.VERDICTS
            if c['verdict'] == 'not_evaluated':
                assert c.get('reason')
                continue
            assert c['sources'] and all(s['study'] in known and s['module'].startswith('vantage.core.')
                                        for s in c['sources']), (r['id'], c)
    for sid, s in res.studies.items():
        assert s['error'] == '', (sid, s['error'])


def test_matrix_figures_follow_the_models(rig):
    p, eng, res = rig
    clear, fog, night = res.row('clear'), res.row('fog_50'), res.row('night')
    cc = {c['class']: c for c in clear['cells'] if c['insufficiency'] == 'range'}
    assert set(cc) == {'Pedestrian', 'Cyclist', 'Car'}
    assert all(c['verdict'] == 'ok' for c in cc.values())
    # the camera's corridor range ahead is the perception's far range ahead less the bumper's offset, roughly
    from vantage.core import perception as PR
    pr = PR.analyse(p, eng)
    k = [c.name for c in pr.classes].index('Pedestrian')
    off = res.geometry['front_x'] - res.geometry['centre'][0]
    assert cc['Pedestrian']['by_type_m']['camera'] <= pr.far[k, 0] - off + 0.3
    assert cc['Pedestrian']['by_type_m']['camera'] > 0.8 * (pr.far[k, 0] - off)
    # thicker fog never helps; night shortens the cameras
    for name in ('Pedestrian', 'Car'):
        ranges = []
        for tid in ('fog_1000', 'fog_200', 'fog_100', 'fog_50'):
            c = next(x for x in res.row(tid)['cells'] if x['insufficiency'] == 'range' and x['class'] == name)
            ranges.append((c['by_type_m']['camera'], c['by_type_m']['lidar']))
        assert all(b[0] <= a[0] + 1e-9 and b[1] <= a[1] + 1e-9 for a, b in zip(ranges, ranges[1:]))
        nc = next(x for x in night['cells'] if x['insufficiency'] == 'range' and x['class'] == name)
        assert nc['by_type_m']['camera'] < cc[name]['by_type_m']['camera']
    fc = {c['class']: c for c in fog['cells'] if c['insufficiency'] == 'range'}
    assert fc['Pedestrian']['verdict'] == 'insufficient'
    # heavier rain: more radar attenuation, never more range; wet road: a longer stop than dry
    rr = [next(x for x in res.row(t)['cells'] if x['insufficiency'] == 'range' and x['class'] == 'Car')
          for t in ('rain_2_5', 'rain_7_6', 'rain_25', 'rain_50')]
    assert all(b['by_type_m']['radar'] <= a['by_type_m']['radar'] for a, b in zip(rr, rr[1:]))
    assert rr[0]['threshold'] > cc['Car']['threshold']
    # the ODD limit is the minimum over the included rows, and the binding row says why
    inc = [r for r in res.rows if r['in_odd'] and r.get('max_kmh') is not None]
    assert res.odd['speed_kmh'] == pytest.approx(min(r['max_kmh'] for r in inc))
    assert res.odd['binding'] == min(inc, key=lambda r: r['max_kmh'])['id']
    # every included row's max speed holds its binding range
    for r in inc:
        stop = SO.stop_of(res.settings, r['road'], next(c.get('extra_delay_s', 0.0) for c in r['cells']
                                                         if c['insufficiency'] == 'range' and c['value'] is not None))
        assert SO.stopping_distance(r['max_kmh'] / 3.6, stop) == pytest.approx(r['binding_range_m'] -
                                                                                res.settings.margin_m, abs=1e-6)


def test_odd_bound_excluding_conditions(rig):
    p, eng, res = rig
    tids = [r['id'] for r in res.rows if r.get('max_kmh') is not None]
    lims = [SO.odd_speed(res.rows, tids[:k]) for k in range(1, len(tids) + 1)]
    assert all(b <= a for a, b in zip(lims, lims[1:]))
    st = SO.SotifSettings(odd=['clear', 'night_lit'])
    p2 = p.snapshot()
    p2.analysis.sotif = st.to_dict()
    res2 = SO.run(p2, eng)
    assert res2.odd['speed_kmh'] >= res.odd['speed_kmh']
    assert set(res2.odd['included']) == {'clear', 'night_lit'}


def test_sun_dirt_housing_blind_and_tunnel_rows(rig):
    p, eng, res = rig
    sun = res.row('sun_low')
    lost = {c['sensor'] for c in sun['cells'] if c['insufficiency'] == 'fov'}
    assert 'F' in lost                                   # the sample campaign puts the low sun in F's image
    rc = next(c for c in sun['cells'] if c['insufficiency'] == 'range' and c['class'] == 'Pedestrian')
    assert rc.get('lost') and rc['value'] <= next(c['value'] for c in res.row('clear')['cells']
                                                  if c['insufficiency'] == 'range' and c['class'] == 'Pedestrian')
    hs = {c['sensor']: c['value'] for c in res.row('housing')['cells'] if c['insufficiency'] == 'fov'}
    assert 'L1' in hs and 0.2 < hs['L1'] < 0.5           # a third of the roof lidar's points land on the roof
    assert all(0 <= v <= 1 for v in hs.values())
    blind = res.row('blind')['cells'][0]
    assert blind['insufficiency'] == 'child' and blind['value'] is not None and blind['area_m2'] >= 0
    tun = res.row('tunnel')['tunnel']
    assert 8.0 < tun['step_ev'] < 11.0 and tun['cameras']
    assert all(c['blind_s'] >= 0 for c in tun['cameras'])


def test_dirt_row_and_cleaning_mitigation(rig):
    p, eng, _ = rig
    from vantage.core import cleaning as CL
    q = p.snapshot()
    q.analysis.sotif = SO.SotifSettings(dirt_min_score=0.3).to_dict()
    r1 = SO.run(q, eng)
    dirt = r1.row('dirt')
    lost = {c['sensor'] for c in dirt['cells'] if c['insufficiency'] == 'fov' and c['value'] == 1.0}
    assert {'F', 'R'} <= lost and not dirt['mitigated']
    rc = next(c for c in dirt['cells'] if c['insufficiency'] == 'range' and c['class'] == 'Pedestrian')
    assert rc['lost'] == 'F' and rc['by_type_m']['camera'] == 0.0     # F is the only camera ahead
    tags = {c.tag: c.uid for c in q.variant.cameras}
    q.analysis.cleaning = {'devices': [{'kind': 'washer', 'product': 'araymond_tele10', 'target': tags['F']},
                                       {'kind': 'washer', 'product': 'araymond_tele10', 'target': tags['R']}]}
    CL.clear_cache()
    r2 = SO.run(q, eng)
    dirt = r2.row('dirt')
    assert {m['sensor'] for m in dirt['mitigated']} == {'F', 'R'}
    rc2 = next(c for c in dirt['cells'] if c['insufficiency'] == 'range' and c['class'] == 'Pedestrian')
    assert rc2['value'] >= rc['value']


def test_requirements_trace_and_proposals(rig):
    p, eng, res = rig
    q = p.snapshot()
    from vantage.core import requirements as RQ
    q.analysis.requirements = {'items': [
        {'id': 'R1', 'metric': 'perception.range', 'params': {'cls': 'Pedestrian', 'sector': 'front'}, 'op': '>=',
         'target': 60, 'priority': 'must', 'text': 'Pedestrians at 60 m ahead'},
        {'id': 'R2', 'metric': 'volume.undetected_m2', 'params': {'object': 'Child 0.9 m'}, 'op': '<=', 'target': 0.1},
        {'id': 'R3', 'metric': 'power.runtime_min', 'params': {}, 'op': '>=', 'target': 30}]}
    r = SO.run(q, eng)
    tr = {t['requirement']: t for t in r.trace}
    assert set(tr) == {'R1', 'R2'}
    assert 'clear/Pedestrian' in tr['R1']['cells'] and tr['R1']['matrix_figure']['value_m'] > 0
    assert tr['R2']['cells'] == ['blind/Child']
    props = r.proposals
    assert {x['params']['cls'] for x in props} == set(r.classes)
    for x in props:
        it = RQ.clean_item(x, q)
        assert it['metric'] == 'perception.range' and it['target'] == x['target']
        need = SO.stopping_distance(r.settings.target_kmh / 3.6, SO.stop_of(r.settings, 'dry')) + r.settings.margin_m
        assert x['target'] >= need


def test_thermal_and_event_rows_use_their_adapters(rig, monkeypatch):
    p, eng, _ = rig
    q = p.snapshot()
    lw = q.variant.cameras[0].tag
    monkeypatch.setattr(SO, '_modality', lambda m, _t={'n': 0}: 'rgb')
    r0 = SO.run(q, eng)
    assert r0.row('thermal_night') is None and r0.row('event_tunnel') is None
    mods = {id(q.module_for(q.variant.cameras[0]))}
    monkeypatch.setattr(SO, '_modality', lambda m: 'lwir' if id(m) in mods else 'rgb')
    real = SO._optional_module
    monkeypatch.setattr(SO, '_optional_module', lambda n: None if n == 'thermal_ir' else real(n))
    r1 = SO.run(q, eng)
    row = r1.row('thermal_night')
    assert row is not None and all(c['verdict'] == 'not_evaluated' for c in row['cells'])
    assert 'not part of this build' in row['detail'] and lw in row['detail']
    # the cameras taken as LWIR (the sample's six share one module) leave the visible cameras' (weather) ranges
    assert all(c['by_type_m']['camera'] in (None, 0.0) for c in r1.row('clear')['cells']
               if c['insufficiency'] == 'range')

    class FakeThermal:
        @staticmethod
        def sotif_ranges(project, engine, cond):
            assert cond == 'night'
            return {'Pedestrian': np.full(360, 90.0), 'Cyclist': np.full(360, 90.0), 'Car': np.full(360, 120.0)}
    monkeypatch.setattr(SO, '_optional_module', lambda n: FakeThermal if n == 'thermal_ir' else real(n))
    r2 = SO.run(q, eng)
    row = r2.row('thermal_night')
    ped = next(c for c in row['cells'] if c['insufficiency'] == 'range' and c['class'] == 'Pedestrian')
    night = next(c for c in r2.row('night')['cells'] if c['insufficiency'] == 'range' and c['class'] == 'Pedestrian')
    assert night['by_type_m']['camera'] is None and ped['by_type_m']['camera'] > 30.0   # only the LWIR ranges
    assert any(s['study'] == 'thermal' for s in ped['sources'])


def test_real_thermal_and_event_modules(rig):
    """With core.thermal_ir and core.event_camera in the build: a FLIR Boson+ 640 on the bumper extends the camera
    range at night beyond the visible cameras'; an event camera beside F gets its own tunnel-exit row."""
    TI = pytest.importorskip('vantage.core.thermal_ir')
    EC = pytest.importorskip('vantage.core.event_camera')
    if not hasattr(TI, 'add_thermal_camera') or not hasattr(EC, 'tunnel_study'):
        pytest.skip('the thermal or event module has no study to read')
    p, eng, res = rig
    q = p.snapshot()
    t = TI.add_thermal_camera(q, engine=eng)
    r = SO.run(q, eng)
    vis = {c['class']: c for c in r.row('night')['cells'] if c['insufficiency'] == 'range'}
    th = {c['class']: c for c in r.row('thermal_night')['cells'] if c['insufficiency'] == 'range'}
    base = {c['class']: c for c in res.row('night')['cells'] if c['insufficiency'] == 'range'}
    for n in ('Pedestrian', 'Car'):
        assert vis[n]['by_type_m']['camera'] == pytest.approx(base[n]['by_type_m']['camera'])   # T not in it
        assert th[n]['by_type_m']['camera'] > vis[n]['by_type_m']['camera'] + 5.0
        assert th[n]['verdict'] in ('ok', 'insufficient') and any(s['study'] == 'thermal' for s in th[n]['sources'])
    assert t.tag in r.row('thermal_night')['detail']


def test_exports_and_provenance(rig):
    p, eng, res = rig
    names = dict(SO.files(res, p, pdf=False))
    assert set(names) == {'sotif.json', 'sotif_matrix.csv', 'sotif_odd.csv', 'sotif_report.html'}
    doc = json.loads(names['sotif.json'])
    assert doc['schema'] == SO.SCHEMA and doc['disclaimer'].startswith('This supports a safety case')
    assert doc['provenance']['schema'] == 'vantage.provenance/1'
    assert doc['odd']['speed_kmh'] == pytest.approx(round(res.odd['speed_kmh'], 2))
    assert len(doc['matrix']) == len(res.rows) and doc['areas'][1]['area'] == 2
    rows = list(csv.reader(io.StringIO(names['sotif_matrix.csv'].decode())))
    assert tuple(rows[0]) == SO.CSV_COLUMNS and len(rows) - 1 == res.metrics['cells']
    html = names['sotif_report.html'].decode()
    assert 'not a certification' in html.lower() and 'Insufficiency matrix' in html
    s = SO.summary(res)
    json.dumps(s)
    assert s['odd_speed_kmh'] == pytest.approx(round(res.odd['speed_kmh'], 1))


def test_cache_and_issues(rig):
    p, eng, res = rig
    hit = SO.cached(p)                    # this result, or a later run of an identical snapshot
    assert hit is not None and hit.key == res.key == SO.cache_key(p)
    iss = SO.issues(p)
    if res.odd['meets_target'] is False:
        assert any('ODD speed limit' in i.message for i in iss)
    q = p.snapshot()
    q.variant.cameras[0].pitch += 3.0
    assert SO.cached(q) is None and SO.issues(q) == []


def test_empty_variant_and_cancel():
    from vantage.core import samples
    from vantage.core.analysis import Engine
    p = samples.tigor_sample_project()
    p.variant.cameras.clear()
    p.variant.sensors.clear()
    res = SO.run(p, Engine())
    assert res is not None and res.odd['speed_kmh'] is None
    assert all(c['verdict'] == 'not_evaluated' for r in res.rows for c in r['cells'] if c['insufficiency'] == 'range')
    assert any(f['kind'] == 'not_evaluated' for f in res.findings)
    q = samples.tigor_sample_project()
    assert SO.run(q, Engine(), cancel=lambda: True) is None


# =========================================================================== the page
from test_gui import app, win, pump, settle, _wait   # noqa: E402,F401 (fixtures)


def _open(win, name, cls):
    if name not in win._page_widgets:
        pg = cls(win.ctx)
        win.pages.addWidget(pg)
        win._page_widgets[name] = pg
    win.set_page(name)
    return win._page_widgets[name]


def _data_rig(app, win):
    uid = next(v for v in win.store.project.variants if v.sensors).uid

    def fn(p):
        p.active_uid = uid
    win.store.edit('variants', fn, 'Data rig')
    settle(app, win)


def test_page(app, win, tmp_path):
    from vantage.ui.workspaces.sotif_ws import SotifPage
    _data_rig(app, win)
    pg = _open(win, 'sotif', SotifPage)
    pg.activate()
    _wait(app, lambda: pg.result is not None and not pg.busy, timeout=180)
    n = len(pg.result.rows)
    assert pg.t_mat.rowCount() == n and pg.t_odd.rowCount() == n
    assert pg.t_mat.columnCount() == 3 + len(pg.result.classes) + 1
    assert 'km/h' in pg.tiles['odd'].unit
    v0 = pg.result.odd['speed_kmh']
    # the target speed, like a user: the field commits, one undo step
    f, _ = pg.fields['target_kmh']
    f.committed.emit(30.0)
    assert win.store.project.analysis.sotif['target_kmh'] == 30.0
    _wait(app, lambda: pg.result is not None and not pg.busy and pg.result.settings.target_kmh == 30.0, timeout=180)
    assert pg.result.odd['speed_kmh'] == pytest.approx(v0)            # the bound does not depend on the target
    win.store.undo.undo()
    pump(app, 0.2)
    assert SO.settings_of(win.store.project).target_kmh == 50.0
    # take the binding condition out of the ODD with its toggle: the limit can only rise
    binding = pg.result.odd['binding']
    i = pg._cond_ids.index(binding)
    pg.t_cond.cellWidget(i, 0).click()
    assert binding not in SO.settings_of(win.store.project).odd
    _wait(app, lambda: pg.result is not None and not pg.busy and binding not in pg.result.odd['included'],
          timeout=180)
    assert pg.result.odd['speed_kmh'] >= v0
    # a road surface from its combo
    j = pg._cond_ids.index('clear')
    pg.t_cond.cellWidget(j, 1).setCurrentIndex(SO.ROADS.index('wet'))
    assert SO.settings_of(win.store.project).roads == {'clear': 'wet'}
    pg.segs['stop_model'].buttons['aashto'].click()
    assert SO.settings_of(win.store.project).stop_model == 'aashto'
    _wait(app, lambda: pg.result is not None and not pg.busy and pg.result.settings.stop_model == 'aashto', timeout=180)
    # the proposals go into the scorecard as one undo step
    from vantage.core import requirements as RQ
    n0 = len(RQ.settings_of(win.store.project).items)
    assert pg.add_proposals() == len(pg.result.proposals)
    assert len(RQ.settings_of(win.store.project).items) == n0 + len(pg.result.proposals)
    assert pg.add_proposals() == 0
    win.store.undo.undo()
    pump(app, 0.2)
    assert len(RQ.settings_of(win.store.project).items) == n0
    _wait(app, lambda: pg.result is not None and not pg.busy and not pg.stale, timeout=180)
    paths = pg.export(str(tmp_path))
    names = {p.rsplit('/', 1)[-1] for p in paths}
    assert {'sotif.json', 'sotif_report.pdf', 'sotif_report.html'} <= names
    assert (tmp_path / 'sotif_report.pdf').read_bytes()[:5] == b'%PDF-'
    pg.b_defaults.click()
    assert win.store.project.analysis.sotif == {}
