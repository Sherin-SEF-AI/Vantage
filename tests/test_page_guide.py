"""The page guide (vantage/assistant/page_guide.py): the assistant knows the page the user is on and suggests what to
do there. Every page id of the main window has an entry whose prerequisites run on the sample, an empty project and a
project without sensors; page_state is cheap (< 30 ms), JSON-able and has no paths; the recommendations for each
state (a missing prerequisite, never run, stale, with a result, long work as a plan); the ranking in recommend.py
(above the starters, below critical items, additive); and in the window: every page's focus targets exist on the real
widget, switching through every page shows that page's items and its "On <page>: …" line within the budget without an
exception, and the model's context (Groq, a fake server) carries the page."""
import json
import os
import sys
import time
from types import SimpleNamespace

import pytest

from vantage.assistant import page_guide as PG
from vantage.assistant import recommend as RC
from vantage.assistant import tools as T
from vantage.core import samples
from vantage.core.project import Project

BUILT_IN = ('viewports',)


def _page_ids():
    from vantage.ui.mainwindow import PAGE_CLASSES
    return list(BUILT_IN) + list(PAGE_CLASSES)


def _no_sensors():
    p = samples.tigor_sample_project()
    p.active_uid = next(v for v in p.variants if v.sensors).uid
    p.variant.sensors = []
    return p


def _data_rig():
    p = samples.tigor_sample_project()
    p.active_uid = next(v for v in p.variants if v.sensors).uid
    return p


def _home_free(o):
    """No string in o looks like a path of this machine."""
    home = os.path.expanduser('~')
    text = json.dumps(o, default=str)
    return home not in text and '/tmp/' not in text and 'C:\\\\' not in text


class FakeSettings:
    def __init__(self, **kv):
        self.kv = kv

    def value(self, k, default=None):
        return self.kv.get(k, default)


def _win(pages=None, **ctx):
    ctx.setdefault('settings', FakeSettings())
    return SimpleNamespace(_page_widgets=pages or {}, ctx=SimpleNamespace(**ctx), _jobs={})


# ============================================================================================ the guide
def test_every_page_has_a_guide_with_the_windows_label():
    from vantage.ui.assistant import PAGES
    ids = _page_ids()
    assert set(PG.GUIDES) == set(ids)
    for pid in ids:
        g = PG.GUIDES[pid]
        assert g.page == pid and g.label == PAGES[pid], pid
        assert g.purpose.endswith('.') and 40 < len(g.purpose) < 260, pid
        for a in g.after:
            assert a.page in PG.GUIDES, (pid, a.page)
        for b in g.before:
            assert b in PG.GUIDES, (pid, b)
        if g.long:                                       # long work goes through the Copilot's approval
            from vantage.assistant import copilot as CP
            assert not g.plan_kind or g.plan_kind in CP.KINDS, (pid, g.plan_kind)
        assert g.result or g.result_fn or pid in ('photo',) or g.result_fn is None


@pytest.mark.parametrize('make', [samples.tigor_sample_project, Project, _no_sensors, _data_rig])
def test_prerequisite_checks_run_on_any_project_with_and_without_a_window(make, tmp_path):
    p = make()
    for w in (None, _win(), _win(settings=FakeSettings(**{'sim/carla_root': str(tmp_path)}))):
        for g in PG.GUIDES.values():
            for n in g.needs:
                v = n.check(p, w)                        # never raises
                assert v in (True, False, None), (g.page, n.id, v)
            st = PG.page_state(g.page, p, w)
            json.dumps(st)


def test_the_checks_say_what_is_missing_on_each_kind_of_project():
    st = PG.page_state('coverage', _no_sensors())
    assert st['status'] == 'needs' and [m['id'] for m in st['missing']] == ['lidar_or_radar']
    assert PG.page_state('coverage', _data_rig())['status'] != 'needs'
    assert [m['id'] for m in PG.page_state('lidar_calib', _no_sensors())['missing']] == ['lidar']
    assert not PG.page_state('lidar_calib', _data_rig())['missing']
    e = Project()
    e.variant.cameras = []
    assert 'cameras' in [m['id'] for m in PG.page_state('tolerance', e)['missing']]
    assert [m['id'] for m in PG.page_state('gnss', _data_rig())['missing']] == ['gnss_or_imu']
    assert PG.page_state('requirements', samples.tigor_sample_project())['missing'][0]['id'] == 'items'


def test_page_state_is_cheap_jsonable_and_has_no_paths():
    p = _data_rig()
    ids = _page_ids()
    for pid in ids:
        PG.page_state(pid, p, None)                     # imports warmed
    worst = 0.0
    for pid in ids:
        ts = []
        for _ in range(3):
            t0 = time.perf_counter()
            st = PG.page_state(pid, p, None)
            ts.append((time.perf_counter() - t0) * 1000)
        worst = max(worst, min(ts))
        assert min(ts) < PG.BUDGET_MS, (pid, ts)
        text = json.dumps(st)
        assert json.loads(text)['page'] == pid and _home_free(st)
        assert set(st) >= {'page', 'label', 'purpose', 'status', 'missing', 'findings', 'summary', 'settings',
                           'after', 'headline'}
    print(f'\npage_state on the sample: the slowest page {worst:.2f} ms')


def test_compact_leaves_out_paths_secrets_and_cuts_lists():
    doc = {'path': '/home/x/run', 'api_key': 'gsk_123', 'out_dir': 'a', 'n': 3, 'note': 'see /home/jo/a.bag',
           'rows': list(range(10)), 'x': float('nan'), 'deep': {'a': {'b': {'c': {'d': 1}}}}, 'url': 'http://h'}
    c = PG.compact(PG._scrub(doc))
    assert 'path' not in c and 'api_key' not in c and 'out_dir' not in c and 'url' not in c
    assert c['n'] == 3 and c['rows'][:4] == [0, 1, 2, 3] and c['rows'][-1] == '… 6 more' and c['x'] is None
    assert '/home' not in json.dumps(c)


# ============================================================================================ the recommendations
def _recs(pid, st):
    return {r.id.split(':', 2)[2]: r for r in PG.page_recommendations(pid, st, RC.Context(page=pid))}


def test_a_missing_prerequisite_points_at_what_provides_it():
    st = PG.page_state('gnss', _data_rig())
    r = _recs('gnss', st)['need:gnss_or_imu']
    assert r.kind == 'open' and r.target == {'page': 'gnss', 'focus': 'b_add_gnss'}
    assert 'GNSS antenna or an IMU' in r.title and r.about == (('page', 'gnss'),) and r.source == PG.SOURCE
    det = _recs('detection', PG.page_state('detection', _data_rig()))
    assert det['need:recordings'].target == {'page': 'detection'}
    assert 'run' not in det and 'plan' not in det                     # nothing to run until it is there


def test_never_run_points_at_the_pages_own_run_button():
    st = PG.page_state('tolerance', _data_rig())
    assert st['status'] == 'never_run'
    r = _recs('tolerance', st)['run']
    assert r.kind == 'open' and r.target == {'page': 'tolerance', 'focus': 'b_run'} and 'Run study' in r.title
    assert 'Not run' in r.why


def test_stale_says_so_and_runs_again():
    from vantage.core import thermo_vib as TV
    p = _data_rig()
    res = TV.run(p)
    pg = SimpleNamespace(result=res, stale=True, busy=False)
    st = PG.page_state('thermo_vib', p, _win({'thermo_vib': pg}))
    assert st['status'] == 'stale' and st['has_result']
    r = _recs('thermo_vib', st)['run']
    assert 'older than the design' in r.why and r.target['focus'] == 'b_run'
    pg.stale = False
    assert PG.page_state('thermo_vib', p, _win({'thermo_vib': pg}))['status'] == 'done'
    pg.busy = True
    st = PG.page_state('thermo_vib', p, _win({'thermo_vib': pg}))
    assert st['status'] == 'running' and 'run' not in _recs('thermo_vib', st)


def test_a_result_gives_its_findings_with_the_figures_the_next_page_and_the_export():
    from vantage.core import harness as HN
    p = _data_rig()
    res = HN.run(p)
    s = HN.summary(res)
    ctl = SimpleNamespace(result=res, stale=False, busy=False)
    st = PG.page_state('harness', p, _win({'harness': SimpleNamespace()}, harness_ctl=ctl))
    assert st['status'] == 'done' and st['findings']
    first = next(f for f in s['findings'] if f['severity'] == 'serious')
    assert st['findings'][0]['message'] == first['message']           # read off the result, unchanged
    recs = _recs('harness', st)
    ask = next(r for k, r in recs.items() if k.startswith('finding:'))
    assert ask.kind == 'ask' and first['message'] in ask.prompt and first['message'][:40] in ask.why
    assert recs['next:brackets'].target == {'page': 'brackets'} and recs['next:brackets'].kind == 'open'
    assert recs['export'].target == {'page': 'harness', 'focus': 'b_export'}
    assert recs['help'].kind == 'ask' and 'harness' in recs['help'].prompt.lower()
    # the summary is cached per result object: the same dict until a new run
    assert PG.page_state('harness', p, _win({'harness': SimpleNamespace()}, harness_ctl=ctl))['summary'] == \
        st['summary']


def test_rig_perception_names_the_weakest_region_with_its_recall():
    from vantage.core import regions as RG
    p = _data_rig()
    grid = RG.default_grid(p)
    rid = grid.region_id(2, 3)
    name = grid.region_name(2, 3)
    metrics = {'Data rig': {'pedestrian': {'all': {'recall': 0.8, 'n_gt': 400},
                                           rid: {'recall': 0.42, 'n_gt': 60},
                                           grid.region_id(0, 0): {'recall': 0.95, 'n_gt': 80},
                                           grid.region_id(1, 1): {'recall': 0.1, 'n_gt': 3}}}}   # too thin
    res = SimpleNamespace(rigs=[{'name': 'Data rig'}], settings=SimpleNamespace(min_gt=10), classes=['pedestrian'],
                          metrics=metrics, grid=grid)
    f = PG._f_rig_perception({'findings': []}, res, p)
    assert f[0]['title'] == f'Ask why pedestrian recall drops at {name}'
    assert '0.42' in f[0]['message'] and '60 object-instants' in f[0]['message']


def test_long_work_is_a_plan_for_approval_never_a_run(tmp_path):
    from vantage.sim import server as SV
    root = tmp_path / 'carla'
    (root / SV.BINARY).parent.mkdir(parents=True, exist_ok=True)
    (root / SV.BINARY).write_text('#!/bin/sh\n')
    w = _win(settings=FakeSettings(**{'sim/carla_root': str(root)}))
    for pid in ('sim', 'campaign', 'experiments', 'perf_model'):
        st = PG.page_state(pid, _data_rig(), w)
        assert st['status'] == 'never_run', (pid, st['missing'])
        recs = _recs(pid, st)
        assert 'run' not in recs and recs['plan'].kind == 'plan', pid
        assert recs['plan'].prompt.startswith('Plan this for my approval with propose_plan')
        assert f'({PG.GUIDES[pid].plan_kind})' in recs['plan'].prompt
    # without CARLA set up: what is missing, not a plan
    st = PG.page_state('sim', _data_rig(), _win(settings=FakeSettings(**{'sim/carla_root': str(tmp_path / 'no')})))
    assert st['status'] == 'needs' and 'plan' not in _recs('sim', st)


# ============================================================================================ recommend.py
def test_page_rules_are_additive_and_rank_between_critical_items_and_the_starters():
    from vantage.core.analysis import Engine
    p = samples.tigor_sample_project()
    p.analysis.requirements = {'items': [{'id': 'R1', 'text': 'x', 'metric': 'coverage.blind_m2', 'params': {},
                                          'op': '<=', 'target': 1, 'priority': 'must', 'note': ''}]}
    # without the page's state: exactly as before (the starters only)
    assert [r.id for r in RC.recommend(RC.State(p), RC.Context())] == ['start:summary', 'start:child', 'start:review']
    # with it: the page's items join, ranked above the starters
    st = RC.State(p, pages={'tolerance': PG.page_state('tolerance', p)})
    ids = [r.id for r in RC.recommend(st, RC.Context(page='tolerance'))]
    assert ids[:2] == ['page:tolerance:run', 'page:tolerance:help'] and ids[2:] == ['start:summary', 'start:child',
                                                                                  'start:review']
    strip = RC.context_items(RC.recommend(st, RC.Context(page='tolerance')), RC.Context(page='tolerance'))
    assert [r.id for r in strip] == ['page:tolerance:run', 'page:tolerance:help']
    # below every critical item (the sample's blind ground)
    e = Engine()
    res = e.analyse(p)
    st = RC.State(p, res, pages={'tolerance': PG.page_state('tolerance', p)})
    recs = RC.recommend(st, RC.Context(page='tolerance'))
    order = [r.id for r in recs]
    crit = [i for i, r in enumerate(recs) if r.severity == 'critical']
    mine = [i for i, r in enumerate(recs) if r.id.startswith(RC.PAGE_PREFIX)]
    assert crit and mine and max(crit) < min(mine), order
    starters = [i for i, r in enumerate(recs) if r.id.startswith('start:')]
    assert not starters or min(starters) > max(mine)


def test_a_page_item_another_rule_already_gives_is_left_out(tmp_path):
    p = _data_rig()
    carla = {'installed': True, 'port': 2000, 'server_up': None, 'runs': 0, 'last_run': None, 'render': None}
    ps = PG.page_state('sim', p, _win(settings=FakeSettings(**{'sim/carla_root': ''})))
    ps = dict(ps, status='never_run', missing=[])
    recs = RC.recommend(RC.State(p, carla=carla, pages={'sim': ps}), RC.Context(page='sim'))
    ids = [r.id for r in recs]
    assert 'carla:first' in ids and 'page:sim:plan' not in ids and 'page:sim:help' in ids


def test_dismissed_page_items_come_back_when_the_page_state_changes():
    p = _data_rig()
    d = RC.Dismissed()
    st = PG.page_state('tolerance', p)
    r = _recs('tolerance', st)['run']
    d.dismiss(r)
    assert d.hides(r)
    st2 = dict(st, status='stale', has_result=True)
    r2 = _recs('tolerance', st2)['run']
    assert not d.hides(r2)


# ============================================================================================ the model's context
def test_the_tool_and_the_context_note_carry_the_page():
    from vantage.assistant import agent as A
    p = _data_rig()
    pc = PG.page_state('coverage', p)
    pc['recommended'] = [{'title': 'Run Lidar and radar coverage: press Run', 'kind': 'open', 'why': ''}]
    tc = T.ToolContext(p, None, None, page='Lidar and radar coverage', page_context=pc)
    out = json.loads(T.run(tc, 'get_page_context', {})[0])
    assert out['page'] == 'coverage' and out['purpose'] == PG.GUIDES['coverage'].purpose
    note = A.context_note(tc)
    assert '[Open page: Lidar and radar coverage (id coverage):' in note and 'get_page_context' in note
    assert 'Run Lidar and radar coverage' in note and len(note) < 2600
    bare = T.ToolContext(p, None, None, page='x')
    assert 'note' in json.loads(T.run(bare, 'get_page_context', {})[0])
    assert '[Open page' not in A.context_note(bare)


def test_claudes_strict_tool_limits_hold_with_the_new_tool():
    from vantage.assistant import claude as CL
    strict = CL.strict_names()
    assert len(strict) <= CL.STRICT_MAX
    assert 'get_page_context' in {t['name'] for t in CL.tool_defs()}


# ============================================================================================ the window
from test_gui import app, win, pump, settle, _wait  # noqa: E402,F401  (fixtures)


def _fake_carla(tmp_path):
    from vantage.sim import server as SV
    root = tmp_path / 'carla'
    root.mkdir(exist_ok=True)
    (root / SV.BINARY).parent.mkdir(parents=True, exist_ok=True)
    (root / SV.BINARY).write_text('#!/bin/sh\n')
    return str(root)


def test_every_page_in_the_window_its_targets_strip_and_header(app, win, monkeypatch, capsys):
    """Opens every page offscreen: every focus target the guide names is an attribute of the real page widget (a
    button with text), every estimate label exists; switching shows that page's items and its "On <page>:" line
    within the budget, with no exception anywhere."""
    from PyQt6.QtWidgets import QAbstractButton, QLabel
    from vantage.sim import server as SV
    monkeypatch.setattr(SV, 'port_open', lambda *a, **k: False)
    errors = []
    monkeypatch.setattr(sys, 'excepthook', lambda t, v, tb: errors.append(f'{t.__name__}: {v}'))
    win.show_assistant()
    panel = win.assistant_panel
    rc = panel.recommender
    times = {}
    for pid in _page_ids():
        g = PG.GUIDES[pid]
        t0 = time.perf_counter()
        win.set_page(pid)
        want = f'On {g.label}:'
        while not (panel.strip.isVisible() and panel.strip.purpose.text().startswith(want)) and \
                time.perf_counter() - t0 < 5:
            pump(app, 0.01)
        times[pid] = (time.perf_counter() - t0) * 1000
        assert panel.strip.purpose.text().startswith(want), (pid, panel.strip.purpose.text())
        assert rc.context.page == pid and rc.page_state['page'] == pid
        ids = list(panel.strip.list.rows)
        assert ids and any(('page', pid) in {tuple(a) for a in r.about} for r in rc.strip()), (pid, ids)
        assert any(i.startswith(f'page:{pid}:') for i in [r.id for r in rc.recs]), pid
        assert rc.ms['page'] < PG.BUDGET_MS, (pid, rc.ms)
        pg = win.page(pid) if pid != 'viewports' else win.viewarea
        targets = [(pid, g.run), (pid, g.export)] + \
            [(n.fix['page'], n.fix.get('focus')) for n in g.needs if n.fix] + \
            [(pid if a.here else a.page, a.focus) for a in g.after]
        for page, attr in targets:
            if not attr:
                continue
            w = win.page(page) if page != 'viewports' else win.viewarea
            assert isinstance(getattr(w, attr, None), QAbstractButton) and getattr(w, attr).text(), (pid, page, attr)
        if g.estimate:
            assert isinstance(getattr(pg, g.estimate, None), QLabel), (pid, g.estimate)
        json.dumps(rc.page_state)
    win.set_page('viewports')
    pump(app, 0.3)
    assert not errors, errors
    med = sorted(times.values())[len(times) // 2]
    with capsys.disabled():
        print(f'\nstrip shows the page\'s items after a page change: median {med:.0f} ms, slowest '
              f'{max(times, key=times.get)} {max(times.values()):.0f} ms (first open of each page included)')
    # a page opened before: the strip follows within the budget (the debounce for a page change is shorter)
    for pid in ('power', 'recording', 'tolerance'):
        t0 = time.perf_counter()
        win.set_page(pid)
        while not panel.strip.purpose.text().startswith(f'On {PG.GUIDES[pid].label}:') and \
                time.perf_counter() - t0 < 3:
            pump(app, 0.005)
        assert (time.perf_counter() - t0) * 1000 < 400, pid


def test_a_page_items_open_points_at_the_button_and_runs_nothing(app, win):
    win.show_assistant()
    panel = win.assistant_panel
    win.set_page('tolerance')
    _wait(app, lambda: any(r.id == 'page:tolerance:run' for r in panel.recommender.recs), 10)
    rec = next(r for r in panel.recommender.recs if r.id == 'page:tolerance:run')
    n, rev = win.store.undo.index(), win.store.revision
    panel.run_cta(rec)
    pump(app, 0.2)
    pg = win.page('tolerance')
    assert win.page_name() == 'tolerance'
    assert pg.result is None and 'tolerance' not in win._jobs                   # nothing ran
    assert win.store.undo.index() == n and win.store.revision == rev


def test_the_models_context_carries_the_page_with_groq(app, win, monkeypatch):
    from fake_groq import FakeGroq
    from vantage.assistant import secrets as K
    srv = FakeGroq().start()
    monkeypatch.setenv('VANTAGE_ASSISTANT_URL', srv.url)
    try:
        st = win.ctx.settings
        K.save('test-key', 'groq')
        st.setValue('assistant/enabled', True)
        st.setValue('assistant/provider', 'groq')
        st.setValue('assistant/consent_groq', True)
        st.setValue('assistant/model', 'openai/gpt-oss-120b')
        st.setValue('assistant/daily_cap', 0)
        win.ctx.assistant.reload()
        win.show_assistant()
        win.set_page('recording')
        pump(app, 0.3)
        tc = win.ctx.assistant._context()
        assert tc.page_context['page'] == 'recording' and tc.page_context['status'] in ('done', 'never_run')
        assert tc.page_context['recommended'] and _home_free(tc.page_context)
        srv.script += [srv.tool_call('get_page_context', {}, 'p1'), srv.answer('Check the disks first.')]
        win.assistant_panel.ask('What should I do here?')
        t0 = time.time()
        while win.ctx.assistant.busy and time.time() - t0 < 60:
            pump(app, 0.05)
        pump(app, 0.2)
        first = srv.requests[0]['messages'][-1]['content']
        assert '[Open page: Recording budget (id recording):' in first
        assert PG.GUIDES['recording'].purpose in first
        tool = json.loads(srv.requests[1]['messages'][-1]['content'])
        assert tool['page'] == 'recording' and tool['label'] == 'Recording budget'
    finally:
        srv.stop()
