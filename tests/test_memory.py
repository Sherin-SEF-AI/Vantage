"""The design log (assistant/memory.py, kept in project.analysis.assistant_log): it survives saving and loading, old
and damaged files load, every edit is one undo step, the model only proposes entries (written when the user accepts),
discarding a proposal records it as rejected, the model gets the log (capped) with every question and is told when a
change repeats a rejected one, matching recommendations are left out, and the panel's Design log view lists, adds,
edits and deletes. Against tests/fake_groq.py and tests/fake_anthropic.py only."""
import json
import threading
import time
import zipfile

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest

from fake_groq import FakeGroq
from vantage.assistant import agent as A
from vantage.assistant import memory as M
from vantage.assistant import recommend as RC
from vantage.assistant import tools as T
from vantage.core import rules, samples
from vantage.core.analysis import Engine
from vantage.core.project import AnalysisSettings, Project


@pytest.fixture(autouse=True)
def _keep_assistant_settings():
    """The assistant's settings (and keys) as they were before each test: these tests change consent, the automatic
    choice and the keys, which other test modules in the same session must not inherit."""
    from PyQt6.QtCore import QSettings
    from vantage.assistant import providers as _PR
    from vantage.assistant import secrets as _K
    s = QSettings('Vantage', 'Vantage')
    s.beginGroup('assistant')
    before = {k: s.value(k) for k in s.allKeys()}
    s.endGroup()
    keys = {p: _K.load(p) for p in _PR.PROVIDERS}
    yield
    s.beginGroup('assistant')
    s.remove('')
    for k, v in before.items():
        s.setValue(k, v)
    s.endGroup()
    for p, (k, src) in keys.items():
        if k and src != 'environment':
            _K.save(k, p)
        elif not k:
            _K.delete(p)


@pytest.fixture(scope='module')
def sample():
    p = samples.tigor_sample_project()
    e = Engine()
    res = e.analyse(p)
    vg = e.vehicle_geom(p)
    mounts = {c.uid: rules.mount_info(p, c, vg) for c in p.variant.cameras}
    return p, e, res, mounts, rules.collect_issues(p, res, vg, mounts)


def _tc(sample, project=None, **kw):
    p, e, res, mounts, issues = sample
    return T.ToolContext((project or p).snapshot(), res, e.share(), issues, mounts, **kw)


def _uid(p, tag):
    return p.variant.by_tag(tag).uid


# ============================================================================================ the file
def test_the_log_round_trips_and_old_files_load(tmp_path):
    p = samples.tigor_sample_project()
    assert M.entries(p) == [] and p.analysis.assistant_log == []
    a = M.add(p, M.new_entry('constraint', 'Keep F above 0.55 m: the tow bar', [_uid(p, 'F')]))
    b = M.add(p, M.new_entry('rejected', 'rejected: pitch F −4.0° (blind +0.060 m²)', [_uid(p, 'F')],
                             changes=[{'camera': 'F', 'prop': 'pitch', 'delta': -4.0}]))
    c = M.add(p, M.new_entry('note', 'Check the mirror camera brackets', [], 'assistant'))
    path = str(tmp_path / 'log.vantage')
    p.save(path)
    q = Project.load(path)
    assert M.entries(q) == M.entries(p) == [a, b, c]
    assert M.entries(q)[1]['changes'] == [{'camera': 'F', 'prop': 'pitch', 'delta': -4.0}]
    assert M.about_tags(q, M.entries(q)[0]) == ['F'] and M.entries(q)[2]['source'] == 'assistant'
    # an older file (no field) loads with an empty log, and saves again without trouble
    with zipfile.ZipFile(path) as z:
        files = {n: z.read(n) for n in z.namelist()}
    doc = json.loads(files['project.json'])
    del doc['analysis']['assistant_log']
    old = str(tmp_path / 'old.vantage')
    with zipfile.ZipFile(old, 'w') as z:
        for n, d in files.items():
            z.writestr(n, json.dumps(doc) if n == 'project.json' else d)
    q = Project.load(old)
    assert q.analysis.assistant_log == [] and M.entries(q) == []
    q.save(old)
    assert Project.load(old).analysis.assistant_log == []
    # a damaged one: junk is left out, never a crash
    assert AnalysisSettings.from_dict({'assistant_log': 'junk'}).assistant_log == []
    q.analysis.assistant_log = [1, None, {'text': ''}, {'kind': 'bogus', 'text': ' x  y ', 'date': 'yesterday',
                                                       'about': 'F', 'source': 'robot', 'changes': [{'prop': 'z'}]}]
    got = M.entries(q)
    assert len(got) == 1 and got[0]['kind'] == 'note' and got[0]['text'] == 'x y' and got[0]['about'] == []
    assert got[0]['source'] == 'user' and len(got[0]['date']) == 10 and 'changes' not in got[0]
    # snapshots (what the assistant's thread reads) carry their own copy
    s = p.snapshot()
    M.add(s, M.new_entry('note', 'only in the copy'))
    assert len(M.entries(s)) == 4 and len(M.entries(p)) == 3


def test_edit_and_remove_and_the_export():
    p = samples.tigor_sample_project()
    e = M.add(p, M.new_entry('decision', 'Use the H05R2 lens on F', [_uid(p, 'F')]))
    assert M.update(p, e['id'], text='Use the H05R2 lens on F and R', about=[_uid(p, 'F'), _uid(p, 'R')])
    assert M.entries(p)[0]['text'].endswith('F and R') and M.about_tags(p, M.entries(p)[0]) == ['F', 'R']
    assert not M.update(p, 'nope', text='x') and not M.update(p, e['id'], text='  ')
    doc = M.export_doc(p)
    assert doc['schema'] == 'vantage.design_log/1' and doc['entries'][0]['about'] == ['F', 'R']
    md = M.to_markdown(p)
    assert md.startswith('## Design log') and '**Decision**' in md
    assert M.remove(p, e['id']) and not M.remove(p, e['id']) and M.entries(p) == [] and M.to_markdown(p) == ''


# ============================================================================================ rejected changes
def test_a_discarded_proposal_becomes_a_rejected_entry(sample):
    got = []
    tc = _tc(sample, on_proposal=got.append)
    out = T.propose_changes(tc, changes=[{'camera': 'QR', 'dpitch': 3}], title='Tilt QR', rationale='t')
    prop = got[0]
    e = M.rejected_entry(prop)
    assert e['kind'] == 'rejected' and e['text'].startswith('rejected: pitch QR +3.0°')
    assert '"Tilt QR"' in e['text'] and 'QL' not in e['text'].split(' — ')[0]       # the mirrored row is implied
    assert {'camera': 'QR', 'prop': 'pitch', 'delta': 3.0} in e['changes']
    assert {'camera': 'QL', 'prop': 'pitch', 'delta': 3.0} in e['changes']
    assert set(e['about']) == set(prop.targets)
    eff = {x['key']: x for x in prop.effects}
    d = eff['blind_m2']['after'] - eff['blind_m2']['before']
    if abs(d) > 1e-6:
        assert f'blind {d:+.3f} m²' in e['text']
    assert 'design_log' not in out


@pytest.mark.parametrize('delta, hit', [(-4.0, True), (-3.0, True), (-6.0, True), (-1.0, False), (4.0, False)])
def test_a_change_that_repeats_a_rejected_one_is_flagged(sample, delta, hit):
    p = samples.tigor_sample_project()
    M.add(p, M.new_entry('rejected', 'rejected: pitch F −4.0° (blind +0.060 m²)', [_uid(p, 'F')],
                         changes=[{'camera': 'F', 'prop': 'pitch', 'delta': -4.0}]))
    out = T.simulate_changes(_tc(sample, p), changes=[{'camera': 'F', 'dpitch': delta}])
    assert ('design_log' in out) == hit
    if hit:
        assert out['design_log']['repeats_rejected'][0]['text'].startswith('rejected: pitch F')
        assert 'Do not propose it again' in out['design_log']['note']
    # an entry the user wrote in words (no stored change) is understood too
    q = samples.tigor_sample_project()
    M.add(q, M.new_entry('rejected', 'Do not pitch F -4° again: it shows the bumper'))
    assert ('design_log' in T.simulate_changes(_tc(sample, q), changes=[{'camera': 'F', 'dpitch': delta}])) == hit


def test_parse_changes():
    tags = ['F', 'R', 'ML', 'MR', 'QL', 'QR']
    assert M.parse_changes('rejected: pitch F −4.0°, yaw ML +2.5°', tags) == [
        {'camera': 'F', 'prop': 'pitch', 'delta': -4.0}, {'camera': 'ML', 'prop': 'yaw', 'delta': 2.5}]
    assert M.parse_changes('A copy with QR pitched -4° gives 3.3 px/cm', tags) == [
        {'camera': 'QR', 'prop': 'pitch', 'delta': -4.0}]
    assert M.parse_changes('lower ML 20 mm', tags) == [{'camera': 'ML', 'prop': 'z', 'delta': -0.02}]
    assert M.parse_changes('z MR +15 mm', tags) == [{'camera': 'MR', 'prop': 'z', 'delta': 0.015}]
    assert M.parse_changes('Explain the issues', tags) == [] and M.parse_changes('', tags) == []


def test_matching_recommendations_are_left_out(sample):
    p = sample[0].snapshot()                    # the analysed project itself (its uids are the result's)
    res = sample[2]
    tags = {c.uid: c.tag for c in p.variant.cameras}
    tag, d = next((tags[u], d) for u, d in res.per_cam.items() if d['best_m2'] > 0 and
                  d['min_dens'] < p.analysis.target_density)
    h = {'delta': -4.0, 'min_dens': d['min_dens'] + 0.5, 'blind_m2': res.metrics['blind_m2'],
         'min_now': d['min_dens'], 'blind_now': res.metrics['blind_m2']}
    st = RC.State(p, res, hints={f'pitch:{tag}': h})
    cx = RC.Context(selection=(('camera', tag, tag),))
    recs = RC.recommend(st, cx)
    r = next(x for x in recs if x.id == f'density:{tag}')
    assert 'try pitch -4°' in r.title
    kept, gone = M.suppress(p, recs)
    assert kept == recs and gone == []                                   # nothing rejected yet
    M.add(p, M.new_entry('rejected', f'rejected: pitch {tag} −4.0°', [_uid(p, tag)],
                         changes=[{'camera': tag, 'prop': 'pitch', 'delta': -4.0}]))
    kept, gone = M.suppress(p, RC.recommend(st, cx))
    assert [x.id for x in gone] == [f'density:{tag}'] and all(x.id != f'density:{tag}' for x in kept)
    assert len(kept) == len(recs) - 1
    # the opposite direction is a different idea: kept
    h2 = dict(h, delta=2.0)
    kept, gone = M.suppress(p, RC.recommend(RC.State(p, res, hints={f'pitch:{tag}': h2}), cx))
    assert not gone and any(x.id == f'density:{tag}' and '+2°' in x.title for x in kept)


# ============================================================================================ the model's side
def test_the_model_only_proposes_entries(sample):
    got = []
    tc = _tc(sample, on_log=got.append)
    out = json.loads(T.run(tc, 'add_design_log', {'kind': 'constraint', 'text': 'Keep F above 0.55 m',
                                                  'about': ['F']})[0])
    assert out['status'].startswith('shown to the user') and out['proposed']['about'] == ['F']
    assert len(got) == 1 and got[0]['source'] == 'assistant' and got[0]['about'] == [_uid(tc.project, 'F')]
    assert M.entries(tc.project) == [] and M.entries(sample[0]) == []            # nothing written
    bad = json.loads(T.run(tc, 'add_design_log', {'kind': 'note', 'text': 'x', 'about': ['NOPE']})[0])
    assert 'unknown camera' in bad['error'] and len(got) == 1
    bad = json.loads(T.run(tc, 'add_design_log', {'kind': 'note', 'text': ' ', 'about': []})[0])
    assert 'error' in bad
    p = samples.tigor_sample_project()
    M.add(p, M.new_entry('decision', 'F stays on the grille'))
    out = json.loads(T.run(_tc(sample, p), 'get_design_log', {})[0])
    assert out['entries'][0]['text'] == 'F stays on the grille' and out['decision'] == 1 and out['count'] == 1


def test_the_prompt_carries_the_log_capped():
    p = samples.tigor_sample_project()
    assert M.prompt_block(p) == ''
    M.add(p, M.new_entry('note', 'A note'))
    M.add(p, M.new_entry('constraint', 'Keep F above 0.55 m', [_uid(p, 'F')]))
    M.add(p, M.new_entry('rejected', 'rejected: pitch F −4.0°'))
    lines = M.prompt_text(p).splitlines()
    assert lines[0].startswith('- constraint') and '[about F]' in lines[0] and lines[1].startswith('- rejected')
    assert lines[2].startswith('- note')
    for i in range(200):
        M.add(p, M.new_entry('note', f'Note number {i} ' + 'x' * 60))
    text = M.prompt_text(p)
    assert len(text) <= M.PROMPT_CAP + 120 and text.endswith('(get_design_log lists them all)')
    assert text.splitlines()[0].startswith('- constraint') and 'Note number 199' in text   # newest kept first
    assert M.prompt_block(p).startswith('DESIGN LOG\n[The project')


def test_groq_gets_the_log_in_its_system_message(sample, monkeypatch):
    srv = FakeGroq().start()
    monkeypatch.setenv('VANTAGE_ASSISTANT_URL', srv.url)
    try:
        p = samples.tigor_sample_project()
        M.add(p, M.new_entry('constraint', 'Keep F above 0.55 m', [_uid(p, 'F')]))
        s = A.Session('test-key', A.Settings('openai/gpt-oss-120b'))
        srv.script += [srv.answer('Noted.')]
        ev = []
        s.run_turn('Hello', _tc(sample, p), ev.append, threading.Event())
        assert ev[-1].data['ok']
        system = srv.requests[0]['messages'][0]['content']
        assert 'DESIGN LOG' in system and 'Keep F above 0.55 m [about F]' in system and 'THE DESIGN LOG' in system
        assert s.messages[0]['content'] == A.SYSTEM                      # the stored history is unchanged
    finally:
        srv.stop()


def test_claude_gets_the_log_as_a_system_message_only_when_it_changes(sample, monkeypatch):
    pytest.importorskip('anthropic', reason='the anthropic SDK is not installed in this Python')
    from fake_anthropic import FakeAnthropic
    from vantage.assistant import claude as CL
    srv = FakeAnthropic(key='sk-ant-test-key').start()
    monkeypatch.setenv('VANTAGE_CLAUDE_URL', srv.url)
    try:
        p = samples.tigor_sample_project()
        s = CL.ClaudeSession('sk-ant-test-key', A.Settings('claude-opus-5-5'))
        srv.script += [srv.reply(text='a'), srv.reply(text='b'), srv.reply(text='c'), srv.reply(text='d'),
                       srv.reply(text='e')]
        run = lambda q: s.run_turn(q, _tc(sample, p), lambda e: None, threading.Event())  # noqa: E731
        run('One')
        assert [m['role'] for m in srv.requests[0]['messages']] == ['user']           # an empty log: nothing
        M.add(p, M.new_entry('constraint', 'Keep F above 0.55 m', [_uid(p, 'F')]))
        run('Two')
        roles = [m['role'] for m in srv.requests[1]['messages']]
        assert roles == ['user', 'assistant', 'user', 'system']
        assert 'Keep F above 0.55 m' in srv.requests[1]['messages'][-1]['content']
        run('Three')                                                                     # unchanged: not again
        assert [m['role'] for m in srv.requests[2]['messages']].count('system') == 1
        assert srv.requests[2]['messages'][:4] == srv.requests[1]['messages'][:4]       # append-only
        assert srv.requests[2]['system'] == srv.requests[0]['system']                    # the frozen system prompt
        p.analysis.assistant_log = []
        run('Four')
        assert srv.requests[3]['messages'][-1]['content'] == CL.EMPTY_LOG
        run('Five')
        assert [m['role'] for m in srv.requests[4]['messages']].count('system') == 2     # the empty log once
    finally:
        srv.stop()


# ============================================================================================ the panel
from test_gui import app, win, pump, _wait  # noqa: E402,F401  (fixtures)


@pytest.fixture()
def groq(monkeypatch):
    s = FakeGroq().start()
    monkeypatch.setenv('VANTAGE_ASSISTANT_URL', s.url)
    yield s
    s.stop()


def _ready(win):
    from vantage.assistant import secrets as K
    st = win.ctx.settings
    K.save('test-key', 'groq')
    st.setValue('assistant/enabled', True)
    st.setValue('assistant/provider', 'groq')
    st.setValue('assistant/auto', False)
    st.setValue('assistant/consent_groq', True)
    st.setValue('assistant/model', 'openai/gpt-oss-120b')
    st.setValue('assistant/daily_cap', 0)
    win.ctx.assistant.reload()
    win.ctx.assistant.new_chat()


def _idle(app, win, timeout=60):
    t = time.time()
    while win.ctx.assistant.busy and time.time() - t < timeout:
        pump(app, 0.05)
    pump(app, 0.2)


def test_accept_and_reject_in_the_conversation(app, win, groq):
    from vantage.ui.assistant import LogProposalCard, ProposalCard
    _ready(win)
    win.show_assistant()
    ctl, panel, st = win.ctx.assistant, win.assistant_panel, win.store
    groq.script += [groq.tool_call('add_design_log', {'kind': 'constraint', 'text': 'Keep F above 0.55 m',
                                                      'about': ['F']}, 'c1'),
                    groq.tool_call('propose_changes', {'changes': [{'camera': 'QR', 'dpitch': 3}], 'title': 'Tilt QR',
                                                       'rationale': 't'}, 'c2'),
                    groq.answer('Proposed.')]
    panel._send('Keep F above 0.55 m, and improve QR')
    _idle(app, win)
    lc = panel.findChild(LogProposalCard)
    pc = panel.findChild(ProposalCard)
    assert lc is not None and lc.b_ok.isVisible() and pc is not None
    assert ctl.log() == []                                               # proposed only
    n0 = st.undo.index()
    QTest.mouseClick(lc.b_ok, Qt.MouseButton.LeftButton)
    pump(app, 0.1)
    log = ctl.log()
    assert len(log) == 1 and log[0]['kind'] == 'constraint' and log[0]['source'] == 'assistant'
    assert M.about_tags(st.project, log[0]) == ['F'] and not lc.b_ok.isVisible() and 'Added' in lc.state.text()
    assert st.undo.index() == n0 + 1 and panel.b_log.text() == 'Log (1)'
    # discarding the proposal writes the rejected entry (one more undo step)
    QTest.mouseClick(pc.b_discard, Qt.MouseButton.LeftButton)
    pump(app, 0.1)
    log = ctl.log()
    assert len(log) == 2 and log[1]['kind'] == 'rejected' and log[1]['text'].startswith('rejected: pitch QR +3.0°')
    assert st.undo.index() == n0 + 2 and 'Design log' in st.undo.undoText()
    st.undo.undo()
    pump(app, 0.1)
    assert len(ctl.log()) == 1
    st.undo.redo()
    pump(app, 0.1)
    assert len(ctl.log()) == 2
    # the next question carries the log, and a repeated proposal is flagged to the model
    groq.script += [groq.tool_call('simulate_changes', {'changes': [{'camera': 'QR', 'dpitch': 3}]}, 'c3'),
                    groq.answer('That was rejected before.')]
    panel._send('Try QR again')
    _idle(app, win)
    req = groq.requests[-1]
    assert 'rejected: pitch QR +3.0°' in req['messages'][0]['content']
    assert 'Keep F above 0.55 m' in req['messages'][0]['content']
    tool = json.loads(req['messages'][-1]['content'])
    assert tool['design_log']['repeats_rejected'][0]['kind'] == 'rejected'
    # a proposed entry the user discards is never written
    groq.script += [groq.tool_call('add_design_log', {'kind': 'note', 'text': 'Not this', 'about': []}, 'c4'),
                    groq.answer('ok')]
    panel._send('Note something')
    _idle(app, win)
    cards = [c for c in panel.findChildren(LogProposalCard) if c.b_no.isVisible()]
    QTest.mouseClick(cards[-1].b_no, Qt.MouseButton.LeftButton)
    pump(app, 0.1)
    assert len(ctl.log()) == 2 and cards[-1].state.text() == 'Discarded.'


def test_the_design_log_view(app, win):
    from vantage.ui.assistant import DesignLogDialog
    win.show_assistant()
    panel, ctl, st = win.assistant_panel, win.ctx.assistant, win.store
    QTest.mouseClick(panel.b_log, Qt.MouseButton.LeftButton)
    pump(app, 0.1)
    d = panel.log_dialog
    assert isinstance(d, DesignLogDialog) and d.isVisible() and d.empty.isVisible() and d.t.rowCount() == 0
    assert not d.b_save.isEnabled()
    d.kind.setCurrentIndex(M.KINDS.index('constraint'))
    QTest.keyClicks(d.text, 'Keep F above 0.55 m')
    d.about.setText('F, ml')
    QTest.mouseClick(d.b_save, Qt.MouseButton.LeftButton)
    pump(app, 0.1)
    log = ctl.log()
    assert len(log) == 1 and log[0]['kind'] == 'constraint' and log[0]['source'] == 'user'
    assert M.about_tags(st.project, log[0]) == ['F', 'ML'] and d.t.rowCount() == 1 and not d.empty.isVisible()
    assert d.t.item(0, 0).text() == 'Constraint' and d.t.item(0, 2).text() == 'F, ML' and panel.b_log.text() == 'Log (1)'
    # an unknown tag is refused, nothing written
    d.new()
    QTest.keyClicks(d.text, 'Something about XX')
    d.about.setText('XX')
    assert not d.save() and 'XX' in d.note.text() and len(ctl.log()) == 1
    # edit the selected entry
    d.t.selectRow(0)
    pump(app, 0.05)
    assert d.b_save.text() == 'Save' and d.text.text() == 'Keep F above 0.55 m'
    d.text.setText('Keep F above 0.60 m')
    d.kind.setCurrentIndex(M.KINDS.index('decision'))
    QTest.mouseClick(d.b_save, Qt.MouseButton.LeftButton)
    pump(app, 0.1)
    assert ctl.log()[0]['text'] == 'Keep F above 0.60 m' and ctl.log()[0]['kind'] == 'decision'
    assert d.t.item(0, 1).text() == 'Keep F above 0.60 m'
    # delete it; undo brings it back and the view follows
    d.t.selectRow(0)
    pump(app, 0.05)
    QTest.mouseClick(d.b_delete, Qt.MouseButton.LeftButton)
    pump(app, 0.1)
    assert ctl.log() == [] and d.t.rowCount() == 0
    st.undo.undo()
    pump(app, 0.2)
    assert len(ctl.log()) == 1 and d.t.rowCount() == 1 and panel.b_log.text() == 'Log (1)'
    d.close()


def test_the_recommender_leaves_out_what_the_log_rejected(app, win, monkeypatch):
    win.show_assistant()
    rc = win.assistant_panel.recommender
    p = win.store.project
    fake = [RC.Recommendation('density:F', "F's minimum density is 3.1 px/cm (target 5): try pitch -4°", 'why', 'ask',
                              prompt='A copy with F pitched -4° gives 3.4 px/cm.', about=(('camera', 'F'),),
                              evidence={'delta': -4.0}, priority=60, base=60),
            RC.Recommendation('other', 'Something else', 'why', 'ask', prompt='x', priority=50, base=50)]
    monkeypatch.setattr(RC, 'recommend', lambda *a, **k: list(fake))
    rc.refresh()
    assert [r.id for r in rc.recs] == ['density:F', 'other'] and rc.suppressed == []
    win.ctx.assistant.add_log('rejected', 'rejected: pitch F −4.0°', [p.variant.by_tag('F').uid])
    rc.refresh()
    assert [r.id for r in rc.recs] == ['other'] and [r.id for r in rc.suppressed] == ['density:F']
    win.store.undo.undo()
    rc.refresh()
    assert [r.id for r in rc.recs] == ['density:F', 'other']
