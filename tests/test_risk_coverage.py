"""Risk-weighted coverage (core.risk_coverage): the published tables load and sum to their printed totals, each
scenario's kinematics place the target where hand geometry says, the turn paths are the protocol's, a rig covering
everything scores 100 %, every weight is traced to its table row, and the page driven like a user."""
import json
import math

import numpy as np
import pytest

from vantage.core import risk_coverage as RC

BODY = RC.Body(x_front=3.5, x_rear=-0.8, width=1.8)


# =========================================================================== the tables
def test_najm2007_sums_to_published_totals():
    rows = RC.NAJM2007
    assert len(rows) == 37 and sorted(r[0] for r in rows) == list(range(1, 38))
    crashes = sum(r[2] for r in rows)
    assert crashes == 5_942_243                              # Table 13, unrounded (stated: ~5,942,000)
    assert abs(crashes - RC.NAJM2007_TOTALS['crashes']) / RC.NAJM2007_TOTALS['crashes'] < 1e-4
    cost = sum(r[3] for r in rows)
    assert abs(cost - RC.NAJM2007_TOTALS['cost_musd']) / RC.NAJM2007_TOTALS['cost_musd'] < 2e-3   # 'about $120 bn'
    fyl = sum(r[4] for r in rows)
    assert abs(fyl - RC.NAJM2007_TOTALS['fyl']) < 37 * 500 + 1     # each row rounded to 1,000
    by = {r[1]: r for r in rows}
    assert by['Lead Vehicle Stopped'][2] == 974_855 and by['Lead Vehicle Stopped'][0] == 26
    assert by['Pedestrian Crash Without Prior Vehicle Maneuver'][2] == 39_324


def test_swanson2019_and_v2p_sum_to_published_totals():
    rows = RC.SWANSON2019
    assert abs(sum(r[2] for r in rows) - RC.SWANSON2019_TOTALS['crashes']) <= 5        # Table 12 rounding
    assert sum(r[3] for r in rows) == RC.SWANSON2019_TOTALS['fatal']
    assert abs(sum(r[4] for r in rows) - RC.SWANSON2019_TOTALS['cost_musd']) <= 5
    assert abs(sum(r[5] for r in rows) - RC.SWANSON2019_TOTALS['equivalent_lives']) <= 5
    by = {r[1]: r for r in rows}
    assert by['Rear-End/LVS'][2] == 1_050_558
    v = RC.V2P2016
    assert len(v) == 45 and abs(sum(r[2] for r in v) - RC.V2P2016_TOTALS['crashes']) <= 5
    gc = next(r for r in v if r[:2] == ('Going Straight', 'Crossing Road'))
    assert gc[2] == 23_558 and gc[3] == 2029


def test_every_mapping_names_a_real_row_and_template():
    names19 = {r[1] for r in RC.SWANSON2019}
    names07 = {r[1] for r in RC.NAJM2007}
    pairs = {(r[0], r[1]) for r in RC.V2P2016}
    for mp, names in ((RC.MAP_SWANSON2019, names19), (RC.MAP_NAJM2007, names07)):
        for k, items in mp.items():
            assert k in names, k
            assert sum(s for _, s in items) == pytest.approx(1.0)
            assert all(t in RC.TEMPLATES for t, _ in items)
    for k, items in RC.MAP_V2P2016.items():
        assert k in pairs, k
        assert all(RC.TEMPLATES[t]['cls'] == 'pedestrian' for t, _ in items)
    for t in RC.NCAP:
        assert t['src'] in RC.NCAP_SOURCES and t['sec'] and t['page']


# =========================================================================== hand geometry
def test_rear_end_stationary_by_hand():
    t = dict(RC.NCAP_BY_ID['CCRs'], ego=(36.0,), pct=(50,))
    hp = RC.hazard_points(t, BODY, [1.0, 3.0])
    # a 4.5 m car's centre half its length ahead of the bumper, plus 10 m/s × τ
    assert hp['xy'][0] == pytest.approx([3.5 + 2.25 + 10.0, 0.0])
    assert hp['xy'][1] == pytest.approx([3.5 + 2.25 + 30.0, 0.0])
    assert hp['w'].sum() == pytest.approx(1.0)


def test_crossing_pedestrian_by_hand_and_start_point():
    t = dict(RC.NCAP_BY_ID['CPNA'], ego=(36.0,), pct=(25,))
    hp = RC.hazard_points(t, BODY, [1.0, 10.0])
    y_imp = -0.9 + 0.25 * 1.8
    # 5 km/h from the right: 1.389 m per second; standing at its start (5.0 m) before that
    assert hp['xy'][0] == pytest.approx([3.5 + 0.25 + 10.0, y_imp - 5 / 3.6])
    assert hp['xy'][1] == pytest.approx([3.5 + 0.25 + 100.0, y_imp - 5.0])
    t = dict(RC.NCAP_BY_ID['CPFA'], ego=(36.0,))
    hp = RC.hazard_points(t, BODY, [1.0])
    assert hp['xy'][0][1] == pytest.approx(0.0 + 8 / 3.6)             # from the left at 8 km/h


def test_braking_lead_by_hand():
    t = dict(RC.NCAP_BY_ID['CCRb'], ego=(72.0,), pct=(50,))
    v0, a, g0 = 20.0, 4.0, 20.0
    # the gap closes when ½ a t² reaches g0 while the lead still moves (t < v0 / a = 5 s): t = √10 s
    tc = math.sqrt(2 * g0 / a)
    hp = RC.hazard_points(t, BODY, [0.5, tc + 0.5])
    assert hp['xy'][0][0] == pytest.approx(3.5 + 2.25 + g0 - 0.5 * a * (tc - 0.5) ** 2, abs=0.05)
    assert hp['xy'][1][0] == pytest.approx(3.5 + 2.25 + g0, abs=1e-6)  # before the braking: the headway


def test_reversing_and_dooring_place_targets_behind():
    hp = RC.hazard_points(RC.NCAP_BY_ID['CPMRCs'], BODY, [1.0])
    assert (hp['xy'][:, 0] < BODY.x_rear).all()
    hp = RC.hazard_points(RC.NCAP_BY_ID['CBDA'], BODY, [2.0])
    assert (hp['xy'][:, 0] < BODY.x_rear + 0.55 * BODY.length).all() and (hp['xy'][:, 1] > BODY.width / 2).all()
    hp = RC.hazard_points(RC.NCAP_BY_ID['ELK-Ov-C2C'], BODY, [2.0])
    assert (hp['xy'][:, 0] < BODY.x_rear + 0.25 * BODY.length).all() and (hp['xy'][:, 1] > 0).all()


def test_turn_path_is_the_protocols():
    for side, tab in (('left', RC.TURN_FAR), ('right', RC.TURN_NEAR)):
        for v, (a, R, b) in tab.items():
            s, x, y, psi = RC.turn_path(side, v)
            assert abs(math.degrees(psi[-1])) == pytest.approx(2 * a + b, abs=0.05)
            assert 2 * a + b == pytest.approx(90.0, abs=0.02)
            k = np.abs(np.gradient(psi, s))
            assert k.max() == pytest.approx(1 / R, rel=0.02)
            assert np.sign(psi[-1]) == (1 if side == 'left' else -1)


def test_turning_scenarios_by_hand():
    # CCFtap: the oncoming car approaches in its lane 1.75 m left of the centre line: 3.5 m left of the ego's lane
    hp = RC.hazard_points(RC.NCAP_BY_ID['CCFtap'], BODY, [4.0])
    far = hp['xy'][np.argmax(hp['xy'][:, 0])]
    assert far[0] > 30 and far[1] > 0
    # the right hook (CBTAns): the cyclist behind on the right before the ego turns across it
    hp = RC.hazard_points(RC.NCAP_BY_ID['CBTAns'], BODY, [4.0])
    assert hp['xy'][0][0] < 1.0 and hp['xy'][0][1] < 0
    # CPTAfs: the pedestrian ahead-left of the ego before the left turn
    hp = RC.hazard_points(RC.NCAP_BY_ID['CPTAfs'], BODY, [3.0])
    assert (hp['xy'][:, 1] > 0).all() and (hp['xy'][:, 0] > 0).all()
    hp = RC.hazard_points(RC.TEMPLATES['turn-target-LTAP'], BODY, [0.5])
    assert (hp['xy'][:, 0] > BODY.x_front).all()


# =========================================================================== weights and coverage
class _Cover:
    def __init__(self, fn):
        self.fn = fn

    def covered(self, xy, cls, metric='resolved'):
        return self.fn(np.asarray(xy, float))

    def inside_body(self, xy):
        return np.zeros(len(xy), bool)


@pytest.mark.parametrize('weight', RC.WEIGHTS)
def test_everything_covered_is_100_percent(weight):
    st = RC.RiskSettings(weight=weight, ttc_samples=4, map_m=60, cell_m=2.0)
    res = RC.run(None, settings=st, coverage=_Cover(lambda xy: np.ones(len(xy), bool)))
    for c, d in res.classes.items():
        assert d['weighted'] == pytest.approx(1.0) and d['area'] == pytest.approx(1.0)
        assert not d['gaps']
        assert sum(r['hazard_share'] for r in d['regions'].values()) + d['outside_grid'] == pytest.approx(1.0)


def test_half_plane_coverage_and_gap_naming():
    st = RC.RiskSettings(weight='ncap', ttc_samples=6, map_m=60, cell_m=2.0)
    res = RC.run(None, settings=st, coverage=_Cover(lambda xy: xy[:, 1] >= 0))    # only the left half
    d = res.classes['pedestrian']
    assert 0.0 < d['weighted'] < 1.0
    assert d['area'] == pytest.approx(0.5, abs=0.05)
    assert all(('right' in g['name']) or ('front' in g['name']) or ('rear' in g['name']) for g in d['gaps'])
    assert 'right' in d['gaps'][0]['name'] or d['gaps'][0]['covered'] < 1


def test_weights_sum_and_trace_to_rows():
    W = RC.weighting(RC.RiskSettings(weight='frequency'))
    for c, d in W.items():
        assert sum(i['weight'] for i in d['items']) == pytest.approx(1.0)
        for it in d['items']:
            tr = it['trace']
            assert ('table' in tr and tr['row']) or tr.get('section')
    # the stopped-lead rear-end share: its crashes over every mapped vehicle crash
    rows = {r[1]: r[2] for r in RC.SWANSON2019}
    tot = sum(rows[k] * s for k, items in RC.MAP_SWANSON2019.items() for t, s in items
              if RC.TEMPLATES[t]['cls'] == 'vehicle')
    it = next(i for i in W['vehicle']['items'] if i['trace']['row'] == 'Rear-End/LVS')
    assert it['weight'] == pytest.approx(rows['Rear-End/LVS'] / tot)
    assert it['trace']['value'] == 1_050_558 and 'Table 12' in it['trace']['table']
    # rows left out are reported (control loss, road departure ...)
    lo = {x['row'] for x in W['vehicle']['left_out']}
    assert 'Control Loss/No Vehicle Action' in lo
    # NCAP: equal
    W = RC.weighting(RC.RiskSettings(weight='ncap'))
    ws = [i['weight'] for i in W['pedestrian']['items']]
    assert max(ws) == pytest.approx(min(ws))
    # fyl uses the 2004 data
    assert RC.RiskSettings(weight='fyl').effective_dataset() == 'najm2007'


def test_settings_tolerant():
    s = RC.RiskSettings.from_dict({'weight': 'x', 'ttc_min': 5, 'ttc_max': 1, 'ttc_samples': 'a', 'cell_m': 0})
    assert s.weight == 'frequency' and s.ttc_max > s.ttc_min and s.cell_m == RC.LIMITS['cell_m'][0]
    assert RC.RiskSettings.from_dict(None) == RC.RiskSettings()


def test_real_rig_and_exports(tmp_path):
    from vantage.core import samples
    p = samples.tigor_sample_project()
    v = next(v for v in p.variants if v.sensors)
    res = RC.run(p, settings=RC.RiskSettings(ttc_samples=6), variant=v)
    for c in RC.CLASSES:
        d = res.classes[c]
        assert 0 < d['weighted'] <= 1 and 0 < d['area'] <= 1
    paths = RC.export(res, p, str(tmp_path))
    doc = json.load(open(tmp_path / 'risk_coverage.json'))
    assert doc['schema'] == RC.SCHEMA and doc['sources']['tables']['najm2007']['rows'] == 37
    assert any(p_.endswith('risk_coverage_weights.csv') for p_ in paths)
    props = RC.requirement_proposal(res)
    assert props and all(x['value'] <= x['measured'] for x in props)


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
    from vantage.ui.workspaces.risk_ws import RiskPage
    pg = _open(win, 'risk_coverage', RiskPage)
    pg.activate()
    _wait(app, lambda: pg.result is not None and not pg.busy, timeout=120)
    assert pg.t_cls.rowCount() == 4 and pg.t_src.rowCount() > 0
    assert 'exposure' in pg.head.text()
    w0 = pg.result.classes['pedestrian']['weighted']
    pg.segs['weight'].set_value('ncap')
    pg.set_option('weight', 'ncap')
    _wait(app, lambda: pg.result is not None and not pg.busy and pg.result.settings.weight == 'ncap', timeout=120)
    assert win.store.project.analysis.risk_coverage['weight'] == 'ncap'
    for c in RC.CLASSES:
        pg.c_cls.set_value(c)
        pg.refresh()
        pump(app, 0.05)
    win.store.undo.undo()
    pump(app, 0.2)
    assert RC.settings_of(win.store.project).weight == 'frequency'
    _wait(app, lambda: pg.result is not None and not pg.busy and pg.result.settings.weight == 'frequency',
          timeout=120)
    assert pg.result.classes['pedestrian']['weighted'] == pytest.approx(w0)
    paths = pg.export(str(tmp_path))
    assert any(p.endswith('risk_coverage.json') for p in paths)


# =========================================================================== round 15: the cache and the Issues list
@pytest.fixture(scope='module')
def cached_rig():
    """The sample's Data rig with its result computed once and cached."""
    from vantage.core import samples
    from vantage.core.analysis import Engine
    p = samples.tigor_sample_project()
    p.active_uid = next(v for v in p.variants if v.sensors).uid
    eng = Engine()
    RC.clear_cache()
    res = RC.run(p, eng)
    return p, eng, res


def test_cache_hit_and_miss_on_each_input(cached_rig, monkeypatch):
    import time
    from vantage.core.project import RigPart
    p, eng, res = cached_rig
    assert RC.cached(p) is res and RC.cached(p.snapshot()) is res               # a worker's snapshot: the same key
    t = time.perf_counter()
    assert RC.run(p, eng) is res and time.perf_counter() - t < 0.05             # a second run: from the cache
    assert res.variant_uid == p.variant.uid
    cam = p.variant.cameras[0]

    def miss(change, undo):
        q = p.snapshot()
        change(q)
        assert RC.cached(q) is None, change
        undo(q)
        assert RC.cached(q) is res, undo
    c0 = lambda q: q.variant.cameras[0]                                          # noqa: E731
    miss(lambda q: setattr(c0(q), 'yaw', cam.yaw + 1.0), lambda q: setattr(c0(q), 'yaw', cam.yaw))
    miss(lambda q: setattr(c0(q), 'z', cam.z + 0.01), lambda q: setattr(c0(q), 'z', cam.z))
    miss(lambda q: setattr(c0(q), 'visible', False), lambda q: setattr(c0(q), 'visible', True))
    other_lens = next(k for k in p.lenses if k != cam.lens_id)
    miss(lambda q: setattr(c0(q), 'lens_id', other_lens), lambda q: setattr(c0(q), 'lens_id', cam.lens_id))
    other_mod = next(k for k in p.modules if k != cam.module_id)
    miss(lambda q: setattr(c0(q), 'module_id', other_mod), lambda q: setattr(c0(q), 'module_id', cam.module_id))
    miss(lambda q: setattr(c0(q), 'calib', {'fx': 300.0}), lambda q: setattr(c0(q), 'calib', cam.calib))
    mesh = p.vehicle.mesh                                                         # the body
    miss(lambda q: setattr(q.vehicle, 'mesh', mesh.transformed(scale=1.02)), lambda q: setattr(q.vehicle, 'mesh', mesh))
    occ = p.vehicle.occluder_target
    miss(lambda q: setattr(q.vehicle, 'occluder_target', occ // 2), lambda q: setattr(q.vehicle, 'occluder_target', occ))
    miss(lambda q: q.rig.append(RigPart(name='bar', kind='box', x=1.0, y=0.0, z=1.7)), lambda q: q.rig.pop())
    st = RC.settings_of(p)                                                        # the study's settings
    miss(lambda q: setattr(q.analysis, 'risk_coverage', {'weight': 'fatal'}),
         lambda q: setattr(q.analysis, 'risk_coverage', st.to_dict()))
    miss(lambda q: setattr(q.analysis, 'risk_coverage', {'ttc_max': 3.0}),
         lambda q: setattr(q.analysis, 'risk_coverage', {}))
    q = p.snapshot()                                                              # the Issues thresholds: no
    q.analysis.risk_coverage = {'issue_min': {'pedestrian': 99}}
    assert RC.cached(q) is res
    per = getattr(p.analysis, 'perception', None)                                 # the perception's classes
    miss(lambda q: setattr(q.analysis, 'perception', {'classes': [{'name': 'Pedestrian', 'height': 1.2,
                                                                     'width': 0.5, 'min_px': 40.0}]}),
         lambda q: setattr(q.analysis, 'perception', per))
    other = next(v for v in p.variants if v.uid != p.active_uid)                  # another variant
    miss(lambda q: setattr(q, 'active_uid', other.uid), lambda q: setattr(q, 'active_uid', p.active_uid))
    key = RC.cache_key(p)                                                         # the hazard tables
    monkeypatch.setattr(RC, 'NCAP', RC.NCAP[:-1])
    monkeypatch.setattr(RC, '_TABLES_VERSION', [])
    assert RC.cache_key(p) != key and RC.cached(p) is None
    monkeypatch.undo()
    assert RC.cache_key(p) == key and RC.cached(p) is res


def test_a_fake_coverage_is_never_cached(cached_rig):
    p, eng, res = cached_rig

    class All:
        def covered(self, xy, cls, metric='resolved'):
            return np.ones(len(np.asarray(xy).reshape(-1, 2)), bool)

        def inside_body(self, xy):
            return np.zeros(len(np.asarray(xy).reshape(-1, 2)), bool)
    q = p.snapshot()
    q.analysis.risk_coverage = {'ttc_samples': 4}
    r = RC.run(q, settings=RC.settings_of(q), coverage=All())
    assert r.classes['pedestrian']['weighted'] == pytest.approx(1.0) and RC.cached(q) is None
    assert RC.cached(p) is res


def test_issues_from_the_cache_timing_severities_and_selection(cached_rig):
    import time
    from vantage.core import rules
    p, eng, res = cached_rig
    q = p.snapshot()
    q.variant.cameras[0].yaw += 2.0                                               # not cached: nothing, at once
    t = time.perf_counter()
    assert RC.issues(q) == []
    assert time.perf_counter() - t < 0.03
    assert RC.issues(p) == []                                                     # the defaults: the sample passes
    assert RC.thresholds(p)['pedestrian'] == {'pct': 80.0, 'source': 'the default threshold', 'priority': ''}
    ped = 100 * res.classes['pedestrian']['weighted']
    moto = 100 * res.classes['motorcyclist']['weighted']
    q = p.snapshot()
    q.analysis.risk_coverage = {'issue_min': {'pedestrian': ped + 3, 'motorcyclist': moto + 12}}
    times = []
    for _ in range(20):
        t = time.perf_counter()
        iss = RC.issues(q)
        times.append(time.perf_counter() - t)
    assert sorted(times)[10] < 0.03                                               # median well under 30 ms
    by = {i.message.split(':')[0]: i for i in iss}
    assert set(by) == {'Pedestrians', 'Motorcyclists'}
    assert by['Pedestrians'].severity == 'warning' and by['Motorcyclists'].severity == 'serious'   # 3 vs 12 points
    g = res.classes['pedestrian']['gaps'][0]
    assert g['name'] in by['Pedestrians'].message and "the risk page's threshold" in by['Pedestrians'].message
    sr = res.grid.parse_id(g['region'])
    assert by['Pedestrians'].where == pytest.approx(tuple(res.grid.centre_xy(*sr)))
    uids = {c.uid for c in p.variant.cameras}
    for i in iss:
        assert i.uids and set(i.uids) <= uids and i.action == 'risk_coverage'
    gv = res.classes['motorcyclist']['gaps'][0]                                    # cameras looking at the gap
    assert by['Motorcyclists'].uids == [x['uid'] for x in gv['cameras_looking']] and gv['cameras_looking']
    assert 'too small in the image' in by['Motorcyclists'].message
    # a requirements scorecard row takes precedence; a must is serious however close
    q.analysis.requirements = {'items': [{'id': 'R7', 'text': 'peds', 'metric': 'risk_coverage.weighted_pct',
                                          'params': {'cls': 'pedestrian'}, 'op': '>=', 'target': round(ped + 1, 1),
                                          'priority': 'must'}]}
    th = RC.thresholds(q)
    assert th['pedestrian']['source'] == 'requirement R7' and th['pedestrian']['priority'] == 'must'
    i = next(x for x in RC.issues(q) if x.message.startswith('Pedestrians'))
    assert i.severity == 'serious' and 'requirement R7' in i.message
    q.analysis.requirements = {'items': [{'id': 'R8', 'text': 'all', 'metric': 'risk_coverage.weighted_pct',
                                          'params': {}, 'op': '>=', 'target': 50.0, 'priority': 'should'}]}
    assert RC.issues(q) == []                                                     # every class ≥ 50 %: none
    # the Issues list collects them (the rules hook) and merge_issues refreshes only these rows
    q.analysis.requirements = {}
    out = rules.collect_issues(q, eng.analyse(q), eng.vg, {}, None)
    assert sum(x.action == 'risk_coverage' for x in out) == 2
    q.analysis.risk_coverage = {}
    other = [x for x in out if x.action != 'risk_coverage']
    assert RC.merge_issues(out, q) == other


def test_background_run_after_an_analysis(cached_rig):
    """What the analysis worker does (proposed for ui/store.py): wants_prime, prime, then merge_issues."""
    p, eng, res = cached_rig
    q = p.snapshot()
    q.variant.cameras[1].pitch += 3.0
    assert RC.wants_prime(q) and not RC.wants_prime(p)
    assert RC.prime(q, eng, cancel=lambda: True) is None and RC.wants_prime(q)    # a new edit cancels it
    r = RC.prime(q, eng)
    assert r is not None and RC.cached(q) is r and not RC.wants_prime(q)
    q.analysis.risk_coverage = {'issue_min': {'cyclist': 100.0}}
    assert [i.message.split(':')[0] for i in RC.merge_issues([], q)] == ['Cyclists']
    q.variant.cameras.clear()
    assert not RC.wants_prime(q) and RC.prime(q, eng) is None and RC.issues(q) == []


def test_page_thresholds_push_the_issues_list(app, win):
    from vantage.ui.workspaces.risk_ws import RiskPage
    store = win.ctx.store
    pg = _open(win, 'risk_coverage', RiskPage)
    pg.activate()
    _wait(app, lambda: pg.result is not None and not pg.busy and not pg.stale, timeout=120)
    assert RC.cached(store.project) is pg.result
    assert pg.th_fields['pedestrian'].value() == pytest.approx(RC.ISSUE_MIN_DEFAULT['pedestrian'])
    ped = 100 * pg.result.classes['pedestrian']['weighted']
    an = win.ctx.analysis
    assert pg.set_threshold('pedestrian', math.ceil(ped) + 2)
    pump(app, 0.2)
    rows = [i for i in an.issues if i.action == 'risk_coverage']
    assert len(rows) == 1 and rows[0].message.startswith('Pedestrians') and rows[0].severity == 'warning'
    assert RC.settings_of(store.project).issue_min == {'pedestrian': math.ceil(ped) + 2}
    assert pg.result is not None and not pg.stale                                 # a threshold is not an input
    store.undo.undo()
    pump(app, 0.2)
    assert RC.settings_of(store.project).issue_min == {}
    pg.push_issues()
    pump(app, 0.1)
    assert not [i for i in an.issues if i.action == 'risk_coverage']
    pg._sync_fields()
    assert pg.th_fields['pedestrian'].value() == pytest.approx(RC.ISSUE_MIN_DEFAULT['pedestrian'])
