"""The design assistant: the Groq client (against tests/fake_groq.py, never the real service), key storage, the tools
over the analysis, the conversation loop, and the panel driven with real events."""
import json
import os
import stat
import threading
import time

import numpy as np
import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest

from fake_groq import FakeGroq
from vantage.assistant import agent as A
from vantage.assistant import client as C
from vantage.assistant import secrets as K
from vantage.assistant import tools as T
from vantage.core import rules, samples
from vantage.core.analysis import Engine, camera_frames, eval_points
from vantage.core.lens import resolve


@pytest.fixture()
def srv(monkeypatch):
    s = FakeGroq().start()
    monkeypatch.setenv('VANTAGE_ASSISTANT_URL', s.url)
    yield s
    s.stop()


@pytest.fixture(scope='module')
def sample():
    p = samples.tigor_sample_project()
    e = Engine()
    res = e.analyse(p)
    vg = e.vehicle_geom(p)
    mounts = {c.uid: rules.mount_info(p, c, vg) for c in p.variant.cameras}
    return p, e, res, mounts, rules.collect_issues(p, res, vg, mounts)


def _tc(sample, **kw):
    p, e, res, mounts, issues = sample
    return T.ToolContext(p.snapshot(), res, e.share(), issues, mounts, **kw)


# ============================================================================== client
def test_models_are_chat_models_most_capable_first(srv):
    ids = C.Client('test-key').models()
    assert ids[0] == 'openai/gpt-oss-120b'
    assert 'whisper-large-v3' not in ids and not any('guard' in i for i in ids)
    assert set(ids) == {'openai/gpt-oss-120b', 'openai/gpt-oss-20b', 'qwen/qwen3.8-27b', 'llama-3.1-8b-instant'}


@pytest.mark.parametrize('status, kw, kind', [
    (401, {}, 'auth'), (429, {'retry_after': 7}, 'rate'), (400, {'code': 'tool_use_failed'}, 'tool_use_failed'),
    (400, {'message': 'bad'}, 'bad_request'), (500, {}, 'server')])
def test_errors_are_actionable(srv, status, kw, kind):
    srv.script.append(srv.error(status, **kw))
    key = 'wrong' if status == 401 else 'test-key'
    with pytest.raises(C.ApiError) as ei:
        list(C.Client(key).stream({'model': 'openai/gpt-oss-120b', 'messages': []}))
    assert ei.value.kind == kind
    if kind == 'rate':
        assert ei.value.retry_after == 7
    if kind == 'auth':
        assert 'Preferences' in ei.value.message


def test_unknown_model_and_unreachable_server(srv):
    with pytest.raises(C.ApiError) as ei:
        list(C.Client('test-key').stream({'model': 'nope/model', 'messages': []}))
    assert ei.value.kind == 'model'
    with pytest.raises(C.ApiError) as ei:
        C.Client('k', url='http://127.0.0.1:9/openai/v1', timeout=2).models()
    assert ei.value.kind == 'network'


def test_stream_assembles_reasoning_content_and_tool_calls(srv):
    srv.script.append(srv.tool_call('get_camera', {'camera': 'ML'}, reasoning='thinking'))
    out = list(C.Client('test-key').stream({'model': 'openai/gpt-oss-120b', 'messages': []}))
    assert ''.join(d.reasoning for d in out) == 'thinking'
    args = ''.join(tc['function'].get('arguments', '') for d in out for tc in d.tool_calls)
    assert json.loads(args) == {'camera': 'ML'}
    assert any(d.usage for d in out)


def test_reasoning_parameters_per_model_family():
    assert C.reasoning_params('openai/gpt-oss-120b', 'high', True) == {'reasoning_effort': 'high', 'include_reasoning': True}
    assert C.reasoning_params('openai/gpt-oss-20b', 'xhigh', False) == {'reasoning_effort': 'medium',
                                                                        'include_reasoning': False}
    q = C.reasoning_params('qwen/qwen3.8-27b', 'low', True)
    assert q == {'reasoning_format': 'parsed', 'reasoning_effort': 'low'}
    assert C.reasoning_params('qwen/qwen3.8-27b', 'low', False)['reasoning_format'] == 'hidden'
    assert C.reasoning_params('llama-3.3-70b-versatile', 'high', True) == {}
    assert C.reasoning_params('someone/new-model', 'high', True) == {}


# ============================================================================== key storage
def test_key_is_saved_privately_and_the_environment_wins(monkeypatch):
    K.delete()
    assert K.load() == (None, None)
    assert K.save('  gsk_abcdefghijklmnop  ') == 'file'
    path = os.path.join(K.config_dir(), K.FILE)
    assert os.environ['XDG_CONFIG_HOME'] in path                    # the test's own folder, not the user's
    if os.name == 'posix':
        assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    assert K.load() == ('gsk_abcdefghijklmnop', 'file')
    assert K.mask('gsk_abcdefghijklmnop') == 'gsk_…mnop'
    monkeypatch.setenv(K.ENV, 'gsk_fromenv')
    assert K.load() == ('gsk_fromenv', 'environment')
    monkeypatch.delenv(K.ENV)
    K.delete()
    assert K.load() == (None, None) and not os.path.exists(path)


# ============================================================================== tools
def test_overview_and_coverage_match_the_project(sample):
    p, e, res, *_ = sample
    tc = _tc(sample, selection=['QR'])
    ov = T.get_overview(tc)
    assert [c['tag'] for c in ov['cameras']] == [c.tag for c in p.variant.cameras]
    ml = next(c for c in ov['cameras'] if c['tag'] == 'ML')
    assert ml['mirrored_with'] == 'MR' and 90 < ml['hfov_deg'] < 220
    assert ov['selected_cameras'] == ['QR'] and 3.5 < ov['vehicle']['length_m'] < 4.5
    cov = T.get_coverage(tc)
    assert cov['blind_m2'] == pytest.approx(res.metrics['blind_m2'], abs=1e-3)
    assert {x['camera'] for x in cov['per_camera']} == {c.tag for c in p.variant.cameras}


def test_explain_point_agrees_with_the_analysis(sample):
    p, e, res, *_ = sample
    tc = _tc(sample)
    rng = np.random.default_rng(3)
    for x, y in zip(rng.uniform(-3, 5.5, 25), rng.uniform(-3, 3, 25)):
        P = np.array([[x, y, 0.0]])
        want = set()
        for c in p.variant.cameras:
            vis, *_ = eval_points(P, c, resolve(p.module_for(c), p.lens_for(c)), e.cube(c.pos()))
            if vis[0]:
                want.add(c.tag)
        got = T.explain_point(tc, x=x, y=y)
        assert set(got['seen_by']) == want, (x, y, got)
        for row in got['cameras']:
            assert row['seen'] == (row['camera'] in want)
            assert row['seen'] or row['reason'] in ('outside the lens field of view', 'outside the image frame',
                                                    'hidden by the body', 'behind the camera')


def test_blind_regions_cover_the_blind_area(sample):
    hidden = samples.tigor_sample_project()
    hidden.variant.by_tag('QR').visible = False                     # a hole on the right
    e = Engine()
    res = e.analyse(hidden)
    tc = T.ToolContext(hidden.snapshot(), res, e)
    out = T.find_blind_regions(tc, min_area_m2=0.0)
    regions = out['blind_regions']                                  # each rounded to 0.001 m²
    assert sum(r['area_m2'] for r in regions) <= out['blind_total_m2'] + 0.0005 * (len(regions) + 1)
    big = out['blind_regions'][0]
    assert big['area_m2'] > 0.3 and 'right' in big['where']
    assert all(isinstance(v, str) for v in big['per_camera_at_sample'].values())


def test_simulation_matches_running_the_analysis(sample):
    p, *_ = sample
    tc = _tc(sample)
    rep = T.simulate_changes(tc, changes=[{'camera': 'ML', 'dpitch': 6}])
    assert [(c['camera'], c['property']) for c in rep['changes']] == [('ML', 'pitch'), ('MR', 'pitch')]
    q = p.snapshot()
    for tag in ('ML', 'MR'):
        q.variant.by_tag(tag).pitch += 6
    want = Engine().analyse(q).metrics
    for k in ('blind_m2', 'covered_pct', 'min_density'):
        assert rep['effect'][k]['after'] == pytest.approx(want[k], abs=2e-3), k
    assert p.variant.by_tag('ML').pitch != q.variant.by_tag('ML').pitch    # the design itself is untouched


def test_change_rules(sample):
    p, *_ = sample
    tc = _tc(sample)
    tc.project.variant.by_tag('F').locked = True
    assert 'locked' in T.simulate_changes(tc, changes=[{'camera': 'F', 'dz': 0.01}])['error']
    assert 'no camera' in T.simulate_changes(tc, changes=[{'camera': 'ZZ', 'dz': 0.01}])['error']
    assert 'pitch' in T.simulate_changes(tc, changes=[{'camera': 'R', 'pitch': 120}])['error']
    rep = T.simulate_changes(tc, changes=[{'camera': 'ML', 'dz': -0.02, 'mirror_partner': False}])
    assert [c['camera'] for c in rep['changes']] == ['ML']
    rep = T.simulate_changes(tc, changes=[{'camera': 'ML', 'dz': -0.02}, {'camera': 'MR', 'dz': -0.03}])
    assert {(c['camera'], round(c['to'], 3)) for c in rep['changes']} == {('ML', 0.93), ('MR', 0.92)}


def test_proposal_reaches_the_ui_with_targets(sample):
    seen = []
    tc = _tc(sample, on_proposal=seen.append)
    out = T.propose_changes(tc, changes=[{'camera': 'QR', 'dpitch': 3}], title='Tilt QR', rationale='why')
    assert out['proposal'] == 1 and len(seen) == 1
    prop = seen[0]
    qr, ql = tc.project.variant.by_tag('QR'), tc.project.variant.by_tag('QL')
    assert prop.targets == {qr.uid: {'pitch': pytest.approx(qr.pitch + 3)}, ql.uid: {'pitch': pytest.approx(ql.pitch + 3)}}
    assert prop.before[qr.uid]['pitch'] == qr.pitch
    assert {e['key'] for e in prop.effects} >= {'blind_m2', 'covered_pct', 'min_density'}


def test_tools_never_raise(sample):
    tc = _tc(sample)
    for name, args in (('get_camera', {'camera': 'nope'}), ('explain_point', {'x': 'a'}), ('nope', {}),
                       ('simulate_changes', {'changes': 'x'}), ('propose_changes', {})):
        text, ok = T.run(tc, name, args)
        assert not ok and 'error' in json.loads(text), name
    text, ok = T.run(tc, 'find_blind_regions', {'min_area_m2': 'x'})      # a bad optional value falls back
    assert ok and 'blind_regions' in json.loads(text)
    big = T.dump({'rows': list(range(20000))})
    assert len(big) <= T.MAX_CHARS + 20
    assert all(s['function']['name'] in T.BY_NAME for s in T.schemas())


# ============================================================================== conversation
def _run(session, tc, text='How is it?'):
    ev = []
    session.run_turn(text, tc, ev.append, threading.Event())
    return ev


def test_a_turn_with_a_tool_call(srv, sample):
    s = A.Session('test-key', A.Settings('openai/gpt-oss-120b', 'high'))
    ev = _run(s, _tc(sample))
    kinds = [e.kind for e in ev]
    assert kinds[0] == 'turn' and kinds[-1] == 'done' and ev[-1].data['ok']
    assert 'tool_start' in kinds and 'tool_done' in kinds and 'reasoning' in kinds
    assert [m['role'] for m in s.messages] == ['system', 'user', 'assistant', 'tool', 'assistant']
    first = srv.requests[0]
    assert first['reasoning_effort'] == 'high' and first['include_reasoning'] is True
    assert first['tool_choice'] == 'auto' and len(first['tools']) == len(T.offered(vision=False))     # no look on Groq
    tool_msg = srv.requests[1]['messages'][-1]
    assert tool_msg['role'] == 'tool' and tool_msg['tool_call_id'] == 'call_get_coverage'
    assert json.loads(tool_msg['content'])['blind_m2'] is not None
    assert '[Vantage context:' in srv.requests[0]['messages'][1]['content']
    assert s.usage.prompt == 2400


def test_retries_and_fallbacks(srv, sample):
    s = A.Session('test-key', A.Settings('qwen/qwen3.8-27b'))
    srv.script += [srv.error(400, 'Failed to call a function', 'tool_use_failed', failed_generation='{oops'),
                   srv.error(429, 'slow down', retry_after=0.05),
                   srv.error(400, 'reasoning_format is not supported with this model'),
                   srv.answer('Fine.')]
    ev = _run(s, _tc(sample))
    assert ev[-1].data['ok'], [e.data for e in ev if e.kind == 'error']
    assert sum(e.kind == 'status' for e in ev) == 3
    assert 'reasoning_format' in srv.requests[0] and 'reasoning_format' not in srv.requests[-1]


def test_a_failed_turn_leaves_the_conversation_as_it_was(srv, sample):
    s = A.Session('test-key', A.Settings('openai/gpt-oss-120b'))
    _run(s, _tc(sample))
    before = list(s.messages)
    srv.script.append(srv.error(500, 'boom'))
    ev = _run(s, _tc(sample), 'Again?')
    assert ev[-1].data['ok'] is False and any(e.kind == 'error' for e in ev)
    assert s.messages == before


def test_tool_rounds_are_bounded(srv, sample, monkeypatch):
    monkeypatch.setattr(A, 'MAX_ROUNDS', 3)
    s = A.Session('test-key', A.Settings('openai/gpt-oss-120b'))
    srv.script += [srv.tool_call('get_issues', {}, f'c{i}') for i in range(5)]
    ev = _run(s, _tc(sample))
    assert any(e.kind == 'error' and e.data['kind'] == 'rounds' for e in ev)
    assert len(srv.requests) == 3 and len(s.messages) == 1


# ============================================================================== the panel
from test_gui import app, win, pump, _wait  # noqa: E402,F401  (fixtures)


def _ready(win, srv):
    K.save('test-key')
    win.ctx.settings.setValue('assistant/enabled', True)
    win.ctx.settings.setValue('assistant/provider', 'groq')
    win.ctx.settings.setValue('assistant/consent_groq', True)       # asked once per provider (test_copilot covers it)
    win.ctx.settings.setValue('assistant/model', 'openai/gpt-oss-120b')
    win.ctx.assistant.reload()


def _wait_idle(app, win, timeout=30):
    t = time.time()
    while win.ctx.assistant.busy and time.time() - t < timeout:
        pump(app, 0.05)
    pump(app, 0.2)


def test_panel_question_answer_and_proposal(app, win, srv):
    _ready(win, srv)
    win.actions_by_id['assistant'].trigger()
    pump(app, 0.2)
    panel = win.assistant_panel
    assert win.docks['assistant'].isVisible() and panel.composer.edit.hasFocus()
    srv.script += [srv.tool_call('propose_changes', {'changes': [{'camera': 'QR', 'dpitch': 3}], 'title': 'Tilt QR',
                                                     'rationale': 'test'}, 'p1'),
                   srv.answer('Done: **proposal** below.\n\n| a | b |\n|---|--:|\n| 1 | 2 |')]
    QTest.keyClicks(panel.composer.edit, 'Improve the right rear')
    QTest.keyClick(panel.composer.edit, Qt.Key.Key_Return)
    pump(app, 0.1)
    assert panel.composer.edit.toPlainText() == ''
    _wait_idle(app, win)
    from vantage.ui.assistant import AssistantTurn, ProposalCard, ToolRow, UserTurn
    turns = [panel.col.itemAt(i).widget() for i in range(panel.col.count() - 1)]
    assert isinstance(turns[0], UserTurn) and isinstance(turns[1], AssistantTurn)
    ans = turns[1]
    assert len(ans.tools) == 1 and isinstance(next(iter(ans.tools.values())), ToolRow)
    assert 'proposal' in ans.texts[-1] and '<table' in ans.text_w.toHtml().lower()
    card = ans.findChild(ProposalCard)
    assert card is not None and card.b_apply.isVisible()
    st = win.store
    qr = st.project.variant.by_tag('QR')
    p0, n0 = qr.pitch, st.undo.index()
    QTest.mouseClick(card.b_apply, Qt.MouseButton.LeftButton)
    pump(app, 0.2)
    assert st.project.variant.by_tag('QR').pitch == pytest.approx(p0 + 3)
    assert st.project.variant.by_tag('QL').pitch == pytest.approx(st.project.variant.by_tag('QR').pitch)
    assert st.undo.index() == n0 + 1 and 'Assistant' in st.undo.undoText()
    assert card.b_undo.isVisible() and not card.b_apply.isVisible()
    QTest.mouseClick(card.b_undo, Qt.MouseButton.LeftButton)
    pump(app, 0.2)
    assert st.project.variant.by_tag('QR').pitch == pytest.approx(p0) and card.b_apply.isVisible()


def test_stop_continues_from_before_the_question(app, win, srv):
    _ready(win, srv)
    win.show_assistant()
    ctl = win.ctx.assistant
    srv.delay = 0.2
    srv.script.append(srv.answer('x' * 400, chunk=4))
    assert ctl.ask('A long answer, please')
    pump(app, 0.4)
    assert ctl.busy
    QTest.keyClick(win.assistant_panel.composer.edit, Qt.Key.Key_Escape)
    pump(app, 0.2)
    assert not ctl.busy and [m['role'] for m in ctl.session.messages] == ['system']
    srv.delay = 0.0
    srv.script.append(srv.answer('Short.'))
    assert ctl.ask('Short then')
    _wait_idle(app, win)
    assert [m['role'] for m in ctl.session.messages] == ['system', 'user', 'assistant']


def test_preferences_test_the_key_and_save_it(app, win, srv):
    K.delete()
    from vantage.ui.dialogs.misc import PreferencesDialog
    d = PreferencesDialog(win.ctx, win, tab='assistant')
    d.show()
    pg = d.assistant
    assert d.tabs.currentWidget() is pg
    pg.on.setChecked(True)
    QTest.keyClicks(pg.key_edit, 'test-key')
    QTest.mouseClick(pg.test, Qt.MouseButton.LeftButton)
    _wait(app, lambda: 'Connected' in pg.status.text() or 'not' in pg.status.text(), 10)
    assert 'Connected · 4 chat models' in pg.status.text()
    assert pg.model.count() == 4 and pg.model.currentData() == 'openai/gpt-oss-120b'
    pg.model.setCurrentIndex(pg.model.findData('qwen/qwen3.8-27b'))
    d.apply()
    assert K.load() == ('test-key', 'file')
    ctl = win.ctx.assistant
    assert ctl.ready and ctl.model == 'qwen/qwen3.8-27b'
    assert ctl.s.value('assistant/models') and 'test-key' not in str(ctl.s.allKeys()) + str(
        [ctl.s.value(k) for k in ctl.s.allKeys()])                  # the key is never in the settings file
    d2 = PreferencesDialog(win.ctx, win, tab='assistant')
    d2.assistant._link('remove')
    d2.apply()
    assert K.load() == (None, None) and not ctl.ready


def test_wrong_key_shows_what_to_do(app, win, srv):
    K.save('wrong-key')
    win.ctx.settings.setValue('assistant/enabled', True)
    win.ctx.settings.setValue('assistant/provider', 'groq')
    win.ctx.settings.setValue('assistant/consent_groq', True)
    win.ctx.assistant.reload()
    win.ask_assistant('Hello?')
    _wait_idle(app, win)
    from PyQt6.QtWidgets import QLabel
    from vantage.ui.assistant import Notice
    n = win.assistant_panel.findChildren(Notice)
    assert n and any('Preferences' in w.text() for w in n[-1].findChildren(QLabel))
    assert win.ctx.assistant.session.messages[-1]['role'] == 'system'     # nothing half-done kept


def test_issues_ask_and_new_project(app, win, srv):
    _ready(win, srv)
    srv.script.append(srv.answer('It is the wheel arch.'))
    from vantage.ui.bottom.panels import IssuesTab
    tab = win.bottom.findChild(IssuesTab)
    tab.refresh()
    assert tab.t.rowCount() >= 1
    tab._ask(0)
    _wait_idle(app, win)
    sent = srv.requests[-1]['messages'][1]['content']
    assert sent.startswith('Explain this issue and suggest a fix:') and win.ctx.analysis.issues[0].message in sent
    assert win.ctx.assistant.session is not None
    win.store.project_replaced.emit()
    pump(app, 0.1)
    assert win.ctx.assistant.session is None and win.assistant_panel.empty is not None


# ============================================================================== agentic tools
@pytest.fixture(scope='module')
def tilted():
    """The sample with both quarter cameras tilted 12° up: blind patches an optimiser can close."""
    p = samples.tigor_sample_project()
    for tag in ('QR', 'QL'):
        p.variant.by_tag(tag).pitch -= 12
    e = Engine()
    return p, e, e.analyse(p)


def test_compare_variants_and_studies(sample):
    tc = _tc(sample)
    out = T.compare_variants(tc)
    rows = {r['variant']: r for r in out['variants']}
    assert len(rows) == len(sample[0].variants) and rows['6-cam · v3']['active']
    assert rows['4-cam']['blind_m2'] > rows['6-cam · v3']['blind_m2']
    vol = T.run_study(tc, study='blind_volume', height=1.0)
    assert vol['height_m'] == 1.0 and {'Child 0.9 m', 'Pole 0.5 m'} <= set(vol['objects'])
    so = T.run_study(tc, study='stitched_objects', object_height=0.9)
    assert so['vanishes_at_seams_m2'] > 0 and len(so['per_seam']) == len(sample[2].seams)
    lc = T.run_study(tc, study='load_cases')
    assert lc['cases'][0]['status'] in ('good', 'warning', 'serious') and 'examples' in lc['note']
    mt = T.run_study(tc, study='mounting_tolerance', tolerance_deg=5, tolerance_mm=100)
    assert mt['worst_blind_m2'] >= mt['baseline_blind_m2'] - 1e-6
    assert 'error' in T.run_study(tc, study='nope')


def test_studies_run_inside_a_conversation(sample):
    """In the app the session sets tc.cancel to a threading.Event; the studies take a callable."""
    tc = _tc(sample)
    tc.cancel = threading.Event()
    for study in ('blind_volume', 'stitched_objects', 'load_cases', 'mounting_tolerance'):
        out = T.run_study(tc, study=study)
        assert 'error' not in out, (study, out)
    tc.cancel.set()
    assert T.run_study(tc, study='blind_volume') == {'error': 'cancelled'}


def test_optimiser_finds_the_fix_and_says_when_there_is_none(tilted, sample):
    p, e, res = tilted
    tc = T.ToolContext(p.snapshot(), res, e.share())
    out = T.optimise_camera(tc, camera='QR', vary=['pitch'], objective='blind', budget_s=20)
    assert out['worth_proposing'] and out['best_change']['dpitch'] > 0
    eff = out['checked_at_project_resolution']
    assert eff['blind_m2']['after'] < eff['blind_m2']['before'] - 0.01
    assert out['search']['stopped'] in ('converged', 'budget')
    assert p.variant.by_tag('QR').pitch == tc.project.variant.by_tag('QR').pitch      # nothing applied
    calm = T.optimise_camera(_tc(sample), camera='F', vary=['pitch'], objective='blind', budget_s=10)
    assert not calm.get('worth_proposing', False)
    assert 'error' in T.optimise_camera(tc, camera='QR', vary=['spin'])
    assert 'error' in T.optimise_camera(tc, camera='QR', objective='speed')


def test_seam_and_threshold_proposals(sample):
    seen = []
    tc = _tc(sample, on_proposal=seen.append)
    res = sample[2]
    tags = {c.uid: c.tag for c in sample[0].variant.cameras}
    s0 = res.seams[0]
    pair = [tags[s0.a], tags[s0.b]]
    out = T.propose_changes(tc, seams=[{'between': pair[::-1], 'angle_deg': s0.angle + 5}],
                            settings={'min_density': 3.5}, title='Seam and threshold')
    assert out['proposal'] == 1
    prop = seen[0]
    assert prop.seams == {f'{s0.a}|{s0.b}': pytest.approx(s0.angle + 5)} and prop.settings == {'min_density': 3.5}
    assert prop.settings_before['min_density'] == sample[0].analysis.min_density
    assert {r['kind'] for r in prop.rows} == {'seam', 'setting'}
    assert 'no seam' in T.simulate_changes(tc, seams=[{'between': ['F', 'R'], 'angle_deg': 0}])['error']
    assert 'move a seam' in T.simulate_changes(tc, seams=[{'between': pair, 'angle_deg': s0.angle + 90}])['error']
    assert 'must be' in T.simulate_changes(tc, settings={'roi_distance': 50})['error']
    assert 'target density' in T.simulate_changes(tc, settings={'min_density': 9.0})['error']


def test_plan_and_app_actions_go_to_the_ui(sample):
    plans, ui = [], []
    tc = _tc(sample, on_plan=plans.append, on_ui=ui.append)
    T.update_plan(tc, steps=[{'step': 'Read', 'status': 'done'}, {'step': 'Fix', 'status': 'weird'}, {'nope': 1}])
    assert plans == [[{'step': 'Read', 'status': 'done'}, {'step': 'Fix', 'status': 'pending'}]]
    assert 'error' in T.update_plan(tc, steps=[])
    T.show_in_app(tc, action='select_cameras', cameras=['ml', 'nope'])
    T.show_in_app(tc, action='frame_point', x=1, y=-2)
    assert ui == [{'action': 'select_cameras', 'cameras': ['ML']}, {'action': 'frame_point', 'x': 1.0, 'y': -2.0}]
    assert 'error' in T.show_in_app(tc, action='delete_everything')
    assert 'error' in T.show_in_app(tc, action='heatmap', heatmap='rainbow')


def test_agent_mode_plans_and_wraps_up_within_its_budget(srv, sample, monkeypatch):
    monkeypatch.setattr(A, 'AGENT_ROUNDS', 4)
    s = A.Session('test-key', A.Settings('openai/gpt-oss-120b', agent=True))
    srv.script += [srv.tool_call('update_plan', {'steps': [{'step': 'Look', 'status': 'in_progress'}]}, 'p1'),
                   srv.tool_call('get_coverage', {}, 'c1'), srv.tool_call('get_issues', {}, 'c2'),
                   srv.answer('Summary with what I found.')]
    tc = _tc(sample, brief='- 4 px/cm at the bumpers')
    tc.agent = True
    ev = _run(s, tc, 'Close every blind patch')
    assert ev[-1].data['ok'] and any(e.kind == 'plan' for e in ev)
    assert any(e.kind == 'progress' and e.data['agent'] for e in ev)
    first, last = srv.requests[0], srv.requests[-1]
    assert 'AGENT MODE' in first['messages'][0]['content']
    assert 'Project brief from the user' in first['messages'][1]['content'] and '4 px/cm' in first['messages'][1]['content']
    assert last['tool_choice'] == 'none' and 'budget' in last['messages'][-1]['content']
    assert len(srv.requests) == 4


def test_slash_commands():
    from vantage.ui.assistant import expand_slash
    assert expand_slash('/goal no blind ground') == ('no blind ground', True, None)
    assert expand_slash('/goal') == ('', None, 'agent')
    assert expand_slash('/new')[2] == 'new' and expand_slash('/brief')[2] == 'brief'
    prompt, agent, _ = expand_slash('/optimise QR')
    assert 'camera QR' in prompt and 'optimise_camera' in prompt and agent is False
    assert 'compare_variants' in expand_slash('/compare')[0]
    assert 'load_cases' in expand_slash('/study load_cases')[0]
    assert expand_slash('plain question') == ('plain question', None, None)
    assert expand_slash('/unknown thing') == ('/unknown thing', None, None)


# ============================================================================== agentic UI
def test_agent_toggle_slash_popup_and_brief(app, win, srv):
    _ready(win, srv)
    win.show_assistant()
    panel, ctl = win.assistant_panel, win.ctx.assistant
    QTest.mouseClick(panel.composer.agent, Qt.MouseButton.LeftButton)
    pump(app, 0.1)
    from PyQt6.QtWidgets import QLabel
    assert ctl.agent_mode and any('agent a goal' in w.text() for w in panel.empty.findChildren(QLabel))
    QTest.mouseClick(panel.composer.agent, Qt.MouseButton.LeftButton)
    assert not ctl.agent_mode
    QTest.keyClicks(panel.composer.edit, '/co')
    pump(app, 0.1)
    pop = panel.composer.popup
    assert pop.isVisible() and [x[0] for x in pop.items] == ['/compare']
    srv.script.append(srv.answer('6-cam is best.'))
    QTest.keyClick(panel.composer.edit, Qt.Key.Key_Return)
    _wait_idle(app, win)
    assert not pop.isVisible()
    assert srv.requests[-1]['messages'][1]['content'].startswith('Compare all variants')
    n = win.store.undo.index()
    ctl.set_brief('Keep the rear camera in the handle.')
    assert ctl.brief() == 'Keep the rear camera in the handle.' and win.store.undo.index() == n + 1
    assert panel.b_brief.text() == 'Brief ✓'
    win.store.undo.undo()
    assert ctl.brief() == ''


def test_agent_run_in_the_panel_with_a_variant_and_a_measurement(app, win, srv):
    _ready(win, srv)
    win.show_assistant()
    ctl, st = win.ctx.assistant, win.store
    srv.script += [srv.tool_call('update_plan', {'steps': [{'step': 'Try QR', 'status': 'in_progress'}]}, 'p1'),
                   srv.tool_call('show_in_app', {'action': 'frame_point', 'x': -0.8, 'y': -0.7}, 'u1'),
                   srv.tool_call('propose_changes', {'changes': [{'camera': 'QR', 'dpitch': 2}], 'title': 'QR down'}, 'c1'),
                   srv.tool_call('update_plan', {'steps': [{'step': 'Try QR', 'status': 'done'}]}, 'p2'),
                   srv.answer('Proposed.')]
    ctl.set_agent(True)
    win.ask_assistant('Fix the rear')
    pump(app, 0.1)
    assert ctl.agent_running
    _wait_idle(app, win)
    from vantage.ui.assistant import PlanCard, ProposalCard
    panel = win.assistant_panel
    plan = panel.findChildren(PlanCard)[-1]
    assert plan.count.text() == '1 of 1'
    assert any(p.plan.pinned_marker == (-0.8, -0.7) for p in win.viewarea.visible_panes())
    card = panel.findChildren(ProposalCard)[-1]
    assert card.b_variant.isVisible()
    v0, qr0 = st.project.variant, st.project.variant.by_tag('QR').pitch
    nvar = len(st.project.variants)
    QTest.mouseClick(card.b_variant, Qt.MouseButton.LeftButton)
    pump(app, 0.2)
    assert len(st.project.variants) == nvar + 1 and st.project.variant is not v0
    assert st.project.variant.by_tag('QR').pitch == pytest.approx(qr0 + 2)
    assert st.project.variant.by_tag('QL').pitch == pytest.approx(qr0 + 2)      # mirrored in the copy
    assert v0.by_tag('QR').pitch == pytest.approx(qr0)                          # the original is untouched
    _wait(app, lambda: ctl.proposals[1].get('measured') is not None, 30)
    assert card.measured.isVisible() and 'Measured after applying' in card.measured.text()
    QTest.mouseClick(card.b_undo, Qt.MouseButton.LeftButton)
    pump(app, 0.2)
    assert len(st.project.variants) == nvar and st.project.variant.uid == v0.uid


def test_groq_optional_arguments_may_be_null_and_the_null_is_dropped(srv, sample):
    """GPT-OSS writes null for an optional argument it does not set; Groq refuses a call that does not match the
    schema, so the Groq schemas allow null on optional arguments, and the null is dropped before the tool runs."""
    s = A.Session('test-key', A.Settings('openai/gpt-oss-120b'))
    srv.script += [srv.tool_call('get_gnss_imu', {'antenna': None}, 'c1'), srv.answer('Done.')]
    ev = _run(s, _tc(sample))
    assert ev[-1].data['ok']
    done = next(e for e in ev if e.kind == 'tool_done')
    assert done.data['ok'] and 'does not match' not in done.data['result']
    tools = {t['function']['name']: t['function']['parameters'] for t in srv.requests[0]['tools']}
    assert 'null' in tools['get_gnss_imu']['properties']['antenna']['type']
    for name, sch in tools.items():                      # required arguments stay non-null; enums gain None
        for k, p in sch['properties'].items():
            if k in sch.get('required', ()):
                assert p.get('type') != 'null' and not (isinstance(p.get('type'), list) and 'null' in p['type'])
            elif isinstance(p.get('type'), list) and 'enum' in p:
                assert None in p['enum']
    assert T.validate_args('get_gnss_imu', {}) == []
    assert all('null' not in json.dumps(T.input_schema(t)) for t in T.TOOLS)   # Claude's strict schemas unchanged


def test_a_refused_tool_call_is_retried_with_the_reason(srv, sample):
    s = A.Session('test-key', A.Settings('openai/gpt-oss-120b'))
    srv.script += [srv.error(400, 'tool call validation failed: parameters for tool run_study did not match schema',
                             'tool_use_failed', failed_generation='{"tolerance_deg": "x"}'),
                   srv.answer('Fine.')]
    ev = _run(s, _tc(sample))
    assert ev[-1].data['ok']
    retry = srv.requests[1]['messages'][-1]
    assert retry['role'] == 'user' and 'refused your last tool call' in retry['content']
    assert srv.requests[0]['messages'] != srv.requests[1]['messages']        # not the identical request again


def test_br_in_a_table_cell_is_a_line_break():
    from vantage.ui.assistant import md_to_html
    out = md_to_html('| a | b |\n|---|---|\n| one<br>two | x<br/>y |')
    assert '&lt;br' not in out and out.lower().count('<br') >= 2
    assert '&lt;script&gt;' in md_to_html('<script>x</script>')              # everything else stays escaped
