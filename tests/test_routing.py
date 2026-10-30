"""Automatic model choice (assistant/routing.py and the panel's controller): the routing table, the fallbacks when a key
or the SDK is missing (said once), the label on each answer, the re-ask chip, one provider's history per conversation
with a hand-over summary when the provider changes, consent and the daily cap per request, usage per provider, and the
Preferences setting. Against tests/fake_groq.py and tests/fake_anthropic.py only (never a real API)."""
import json
import time

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest

from fake_groq import FakeGroq
from vantage.assistant import agent as A
from vantage.assistant import providers as PR
from vantage.assistant import routing as RT
from vantage.assistant import secrets as K

BOTH = {PR.GROQ: 'gsk_test', PR.CLAUDE: 'sk-ant-test'}


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


# ============================================================================================ the table
@pytest.mark.parametrize('text, agent, planning, tier, reason', [
    ('What is the minimum density of F?', False, False, 'fast', 'quick'),
    ('Explain the issues list', False, False, 'fast', 'quick'),
    ('Summarise this design', False, False, 'fast', 'quick'),
    ('Get the blind ground to zero', True, False, 'deep', 'goal'),
    ('Plan a 10 s drive in Town10 in rain', False, True, 'deep', 'plan'),
    ('Optimise the whole rig for coverage', False, False, 'deep', 'rig'),
    ('Can you improve all the cameras together?', False, False, 'deep', 'rig'),
    ('Use optimise_rig on this', False, False, 'deep', 'rig'),
    ('Where is the ground blind?', False, False, 'deep', 'image'),
    ('Which side has the bigger gap?', False, False, 'deep', 'image'),
    ('What does ML see?', False, False, 'deep', 'image'),
    ('Show me the coverage map', False, False, 'deep', 'image'),
    ('What shape is the blind patch at the rear?', False, False, 'deep', 'image'),
    ('Optimise the rig', True, True, 'deep', 'goal'),                  # first match wins: the mode
])
def test_the_routing_table(text, agent, planning, tier, reason):
    assert RT.classify(text, agent, planning) == (tier, reason)


def test_models_fallbacks_and_the_label():
    r = RT.route('Where is it blind?', keys=BOTH)
    assert (r.provider, r.model, r.tier, r.reason, r.fallback) == ('claude', 'claude-opus-5-5', 'deep', 'image', '')
    assert r.label == 'Claude Opus 5.5 · image question'
    r = RT.route('How many cameras?', keys=BOTH)
    assert (r.provider, r.model) == ('groq', 'openai/gpt-oss-120b') and r.label == 'GPT-OSS 120B · quick question'
    assert RT.route('x', agent=True, keys=BOTH).label == 'Claude Opus 5.5 · goal mode'
    # no Groq key: quick questions go to Claude Sonnet
    r = RT.route('How many cameras?', keys={PR.CLAUDE: 'k'})
    assert (r.provider, r.model, r.fallback) == ('claude', 'claude-sonnet-5-5', '')
    # no Claude key, or no SDK: the best available model, and why
    r = RT.route('x', agent=True, keys={PR.GROQ: 'k'})
    assert (r.provider, r.model) == ('groq', 'openai/gpt-oss-120b') and 'no Claude key' in r.fallback
    assert r.label == 'GPT-OSS 120B · goal mode'
    r = RT.route('Where is it blind?', keys=BOTH, claude_sdk=False)
    assert r.provider == 'groq' and 'anthropic SDK is not installed' in r.fallback and 'Pictures need Claude' in r.note
    assert RT.route('x', keys={}) is None and RT.route('x', agent=True, keys={PR.CLAUDE: 'k'}, claude_sdk=False) is None
    d = RT.route('x', agent=True, keys=BOTH).to_dict()
    assert set(d) >= {'provider', 'model', 'tier', 'reason', 'label', 'fallback'} and json.dumps(d)
    # deterministic: the same request and keys always route the same way
    assert all(RT.route('Where is it blind?', keys=BOTH) == RT.route('Where is it blind?', keys=BOTH)
               for _ in range(5))


def test_the_alternative_and_the_default():
    fast = RT.route('How many cameras?', keys=BOTH)
    alt = RT.alternative(fast, BOTH)
    assert (alt.provider, alt.model, alt.reason) == ('claude', 'claude-opus-5-5', 'reask')
    assert alt.label == 'Claude Opus 5.5 · re-asked'
    back = RT.alternative(alt, BOTH)
    assert (back.provider, back.model) == ('groq', 'openai/gpt-oss-120b')
    assert RT.alternative(RT.route('x', agent=True, keys={PR.GROQ: 'k'}), {PR.GROQ: 'k'}) is None   # nothing else
    opus = RT.route('x', agent=True, keys={PR.CLAUDE: 'k'})
    assert RT.alternative(opus, {PR.CLAUDE: 'k'}).model == 'claude-sonnet-5-5'
    assert RT.default_auto(True, BOTH) and not RT.default_auto(False, BOTH)
    assert not RT.default_auto(True, {PR.GROQ: 'k'}) and not RT.default_auto(True, {PR.GROQ: 'k', PR.CLAUDE: None})


def test_the_handover_is_a_short_local_summary():
    md = '\n'.join(f'### You\n\nQuestion {i} about F?\n\n### Assistant\n\nAnswer {i}: ' + 'x' * 600 + '\n'
                   for i in range(12))
    h = RT.handover_text(md, PR.GROQ)
    assert h.startswith(A.HANDOVER_HEAD + ' (with Groq') and h.endswith(']')
    assert 'Question 11 about F?' in h and 'Question 0 ' not in h and 'earlier exchanges left out' in h
    assert len(h) < RT.HANDOVER_CHARS + 400 and '…' in h
    assert A.strip_handover(h + '\n\nThe new question') == 'The new question'
    assert A.strip_handover('A plain question') == 'A plain question'
    assert RT.handover_text('', PR.CLAUDE).endswith('- (nothing yet)\n]')


# ============================================================================================ the panel
from test_gui import app, win, pump, _wait  # noqa: E402,F401  (fixtures)


@pytest.fixture()
def groq(monkeypatch):
    s = FakeGroq(key='gsk_test').start()
    monkeypatch.setenv('VANTAGE_ASSISTANT_URL', s.url)
    yield s
    s.stop()


@pytest.fixture()
def claude(monkeypatch):
    pytest.importorskip('anthropic', reason='the anthropic SDK is not installed in this Python')
    from fake_anthropic import FakeAnthropic
    s = FakeAnthropic(key='sk-ant-test').start()
    monkeypatch.setenv('VANTAGE_CLAUDE_URL', s.url)
    yield s
    s.stop()


def _ready(win, keys, auto=True, consent=(PR.GROQ, PR.CLAUDE)):
    st = win.ctx.settings
    for p in PR.PROVIDERS:
        K.delete(p)
        st.setValue(f'assistant/consent_{p}', p in consent)
    for p, k in keys.items():
        K.save(k, p)
    st.setValue('assistant/enabled', True)
    st.setValue('assistant/provider', 'groq')
    st.setValue('assistant/model', 'openai/gpt-oss-120b')
    st.setValue('assistant/claude_model', 'claude-opus-5-5')
    st.setValue('assistant/auto', auto)
    st.setValue('assistant/daily_cap', 0)
    st.setValue('assistant/day', '')
    win.ctx.assistant.reload()
    win.ctx.assistant.new_chat()


def _idle(app, win, timeout=60):
    t = time.time()
    while win.ctx.assistant.busy and time.time() - t < timeout:
        pump(app, 0.05)
    pump(app, 0.2)


def _turns(panel):
    from vantage.ui.assistant import AssistantTurn
    return [w for w in panel.findChildren(AssistantTurn) if w.isVisible()]


def _send(app, win, text, agent=None):
    win.assistant_panel._send(text, agent)
    pump(app, 0.1)
    _idle(app, win)


def test_without_claude_a_goal_falls_back_and_says_so_once(app, win, groq):
    _ready(win, {PR.GROQ: 'gsk_test'})
    win.show_assistant()
    ctl, panel = win.ctx.assistant, win.assistant_panel
    assert ctl.auto and ctl.ready and panel.composer.model_btn.text().startswith('Automatic')
    groq.script += [groq.answer('Plan: first the coverage.'), groq.answer('Again.')]
    _send(app, win, 'Get the blind ground to zero', agent=True)
    t1 = _turns(panel)[-1]
    assert t1.model_lbl.text() == 'GPT-OSS 120B · goal mode' and ctl.last_route.fallback
    notes = [w.text() for w in t1.findChildren(type(t1.status)) if 'no Claude key' in w.text()]
    assert notes, 'the fallback is said'
    _send(app, win, 'Get the blind ground to zero, once more', agent=True)
    t2 = _turns(panel)[-1]
    assert t2 is not t1 and not [w for w in t2.findChildren(type(t2.status)) if 'no Claude key' in w.text()]
    assert len(groq.requests) == 2 and groq.requests[0]['model'] == 'openai/gpt-oss-120b'
    assert not t1.b_reask.isVisible()                    # nothing else to ask
    assert 'look' not in [t['function']['name'] for t in groq.requests[0]['tools']]


def test_consent_and_the_cap_apply_per_request(app, win, groq):
    from vantage.ui.assistant import ConsentCard
    _ready(win, {PR.GROQ: 'gsk_test'}, consent=())
    win.show_assistant()
    ctl, panel = win.ctx.assistant, win.assistant_panel
    groq.script += [groq.answer('Six cameras.')]
    _send(app, win, 'How many cameras are there?')
    cards = [c for c in panel.findChildren(ConsentCard) if c.isVisible()]
    assert cards and cards[-1].provider == PR.GROQ and not groq.requests            # nothing sent yet
    QTest.mouseClick(cards[-1].b_ok, Qt.MouseButton.LeftButton)
    pump(app, 0.1)
    _idle(app, win)
    assert len(groq.requests) == 1 and ctl.consented(PR.GROQ) and not ctl.consented(PR.CLAUDE)
    assert ctl.tokens_today(PR.GROQ) > 0
    win.ctx.settings.setValue('assistant/daily_cap', 1)
    n = len(groq.requests)
    _send(app, win, 'And how many lidars?')
    assert len(groq.requests) == n and ctl.blocked(PR.GROQ) == 'cap'
    assert not ctl.ask('And how many lidars?')
    win.ctx.settings.setValue('assistant/daily_cap', 0)


def test_preferences_automatic_setting(app, win):
    from vantage.ui.assistant import AssistantPrefs
    st = win.ctx.settings
    for p in PR.PROVIDERS:
        K.delete(p)
    st.remove('assistant/auto')
    st.remove('assistant/provider')
    w = AssistantPrefs(st)
    assert not w.auto.isChecked() and w.fresh                # a new installation without both keys: off
    K.save('gsk_test', PR.GROQ)
    K.save('sk-ant-test', PR.CLAUDE)
    w = AssistantPrefs(st)
    assert w.auto.isChecked()                                # ... with both keys: on
    w.apply()
    assert st.value('assistant/auto', type=bool) is True
    st.remove('assistant/auto')                              # an existing user (a provider was chosen): off
    w = AssistantPrefs(st)
    assert not w.fresh and not w.auto.isChecked()
    w.apply()
    assert st.value('assistant/auto', type=bool) is False
    w = AssistantPrefs(st)
    w.auto.setChecked(True)
    w.apply()
    win.ctx.assistant.reload()
    assert win.ctx.assistant.auto and st.value('assistant/auto', type=bool) is True
    assert 'Claude Opus 5.5' in w.auto_note.text()
    # the privacy line names the pictures
    w._provider_changed(PR.CLAUDE)
    assert 'pictures the assistant draws' in w.sent_note.text()
    w._provider_changed(PR.GROQ)
    assert 'Pictures need Claude' in w.sent_note.text()
    # the composer's menu turns it off by picking a model
    win.assistant_panel.composer._fill_menu()
    acts = win.assistant_panel.composer.model_btn.menu().actions()
    assert acts[0].text().startswith('Automatic') and acts[0].isChecked()
    next(a for a in acts if a.text().startswith('GPT-OSS 120B')).trigger()
    assert not win.ctx.assistant.auto
    for p in PR.PROVIDERS:
        K.delete(p)


def test_auto_routes_each_request_and_hands_over(app, win, groq, claude):
    _ready(win, {PR.GROQ: 'gsk_test', PR.CLAUDE: 'sk-ant-test'})
    win.show_assistant()
    ctl, panel = win.ctx.assistant, win.assistant_panel
    groq.script += [groq.answer('There are **six** cameras.')]
    _send(app, win, 'How many cameras are there?')
    t1 = _turns(panel)[-1]
    assert t1.model_lbl.text() == 'GPT-OSS 120B · quick question' and ctl.session.provider == PR.GROQ
    assert len(groq.requests) == 1 and not claude.requests
    # a spatial question: Claude Opus, a new conversation with a hand-over summary
    claude.script += [claude.reply(tools=[('look', {'view': 'top'})]), claude.reply(text='Behind the rear bumper.')]
    _send(app, win, 'Where is the ground blind?')
    t2 = _turns(panel)[-1]
    assert t2.model_lbl.text() == 'Claude Opus 5.5 · image question' and ctl.session.provider == PR.CLAUDE
    first = claude.requests[0]
    assert first['model'] == 'claude-opus-5-5'
    text = first['messages'][0]['content'][0]['text']
    assert text.startswith('[Earlier in this conversation (with Groq') and 'How many cameras are there?' in text
    assert 'Where is the ground blind?' in text and 'six' in text
    assert all(m['role'] in ('user', 'assistant', 'system') for r in claude.requests for m in r['messages'])
    assert not any('tool_calls' in m for r in claude.requests for m in r['messages'])        # no Groq format
    assert claude.images and t2.pictures                                                     # it looked
    assert 'A new conversation with Claude' in ' '.join(w.text() for w in t2.findChildren(type(t2.status)))
    # back to a quick question: Groq again, with a summary of the Claude part; never Claude's blocks
    groq.script += [groq.answer('F.')]
    _send(app, win, 'Which camera is at the front?')
    g = groq.requests[-1]['messages']
    assert [m['role'] for m in g] == ['system', 'user']
    assert g[1]['content'].startswith('[Earlier in this conversation (with Claude') and 'rear bumper' in g[1]['content']
    assert not any(isinstance(m.get('content'), list) for m in g)
    # usage per provider, both counted against the day
    us = ctl.usages()
    assert set(us) == {PR.GROQ, PR.CLAUDE} and us[PR.CLAUDE].requests == 2 and us[PR.GROQ].requests == 2
    assert 'Groq:' in ctl.usage_line() and 'Claude:' in ctl.usage_line()
    assert ctl.tokens_today(PR.GROQ) > 0 and ctl.tokens_today(PR.CLAUDE) > 0
    assert 'How many cameras' in ctl.transcript() and 'Behind the rear bumper' in ctl.transcript()
    # re-ask the quick question with Claude Opus from its chip
    t3 = _turns(panel)[-1]
    assert t3.b_reask.isVisible() and t3.b_reask.text() == 'Re-ask with Claude Opus 5.5'
    claude.script += [claude.reply(text='The front camera is F.')]
    n = len(claude.requests)
    QTest.mouseClick(t3.b_reask, Qt.MouseButton.LeftButton)
    pump(app, 0.1)
    _idle(app, win)
    t4 = _turns(panel)[-1]
    assert t4 is not t3 and t4.model_lbl.text() == 'Claude Opus 5.5 · re-asked' and len(claude.requests) == n + 1
    assert claude.requests[-1]['messages'][0]['content'][0]['text'].startswith('[Earlier in this conversation')
    assert t4.b_reask.text() == 'Re-ask with GPT-OSS 120B'


def test_a_new_provider_needs_its_own_consent(app, win, groq, claude):
    from vantage.ui.assistant import ConsentCard
    _ready(win, {PR.GROQ: 'gsk_test', PR.CLAUDE: 'sk-ant-test'}, consent=(PR.GROQ,))
    win.show_assistant()
    panel = win.assistant_panel
    _send(app, win, 'What does F see?')
    cards = [c for c in panel.findChildren(ConsentCard) if c.isVisible()]
    assert cards and cards[-1].provider == PR.CLAUDE and 'Send this question to Claude' in \
        cards[-1].findChildren(type(panel.title))[0].text()
    assert not claude.requests and not groq.requests
    claude.script += [claude.reply(text='The front.')]
    QTest.mouseClick(cards[-1].b_ok, Qt.MouseButton.LeftButton)
    pump(app, 0.1)
    _idle(app, win)
    assert len(claude.requests) == 1 and win.ctx.assistant.consented(PR.CLAUDE)


# ============================================================================================ the whole-rig tools
ADD = {'camera': 'SL', 'add': True, 'x': 1.6, 'y': 0.92, 'z': 0.95, 'yaw': 90.0, 'pitch': 50.0, 'roll': 0.0,
       'mirror_partner': True, 'name': 'Side left', 'location': 'left B-pillar'}


def test_the_rig_tools_are_registered_and_the_add_change_is_strict():
    from vantage.assistant import claude as CL
    from vantage.assistant import tools as T
    assert {'optimise_rig', 'add_camera_options'} <= set(T.BY_NAME)
    assert len(T.BY_NAME) == len(T.TOOLS) == len({t.name for t in T.TOOLS})
    names = [s['function']['name'] for s in T.schemas()]
    assert 'optimise_rig' in names and 'add_camera_options' in names
    assert T.validate_args('propose_changes', {'changes': [ADD], 'title': 't', 'rationale': 'r'}) == []
    assert T.validate_args('propose_changes', {'changes': [dict(ADD, add='yes')]})
    assert T.validate_args('propose_changes', {'changes': [dict(ADD, colour='red')]})
    d = next(t for t in CL.tool_defs() if t['name'] == 'propose_changes')
    item = d['input_schema']['properties']['changes']['items']
    assert d['strict'] is True and item['additionalProperties'] is False
    assert {'add', 'module', 'name', 'location'} <= set(item['properties'])


def test_an_add_change_adds_a_camera_to_a_copy():
    from vantage.assistant import tools as T
    from vantage.core import samples
    p = samples.tigor_sample_project()
    n = len(p.variant.cameras)
    q = p.snapshot()
    rows, targets, before, errors = T.apply_changes(q, [ADD])
    assert not errors and len(q.variant.cameras) == n + 2 and len(p.variant.cameras) == n
    added = [c for c in q.variant.cameras if c.uid in targets]
    assert [c.tag for c in added][0] == 'SL' and all('__add__' in targets[c.uid] for c in added)
    sl = added[0]
    assert (sl.x, sl.y, sl.z, sl.yaw, sl.pitch) == (1.6, 0.92, 0.95, 90.0, 50.0)
    assert added[1].y == pytest.approx(-0.92) and added[1].link_uid == sl.uid
    assert {r['kind'] for r in rows} == {'add'} and before == {}
    # never more cameras than the recorder takes
    full = p.snapshot()
    for i in range(8 - n):
        T.apply_changes(full, [dict(ADD, camera=f'X{i}', mirror_partner=False, y=0.9 - 0.1 * i)])
    _, _, _, errors = T.apply_changes(full, [dict(ADD, camera='Y', mirror_partner=False)])
    assert errors and 'up to 8 cameras' in errors[0]


def test_accepting_an_add_proposal_adds_the_camera_and_undo_removes_it(app, win, groq):
    from vantage.assistant import memory as M
    from vantage.ui.assistant import ProposalCard
    _ready(win, {PR.GROQ: 'gsk_test'}, auto=False, consent=(PR.GROQ,))
    win.show_assistant()
    ctl, panel, st = win.ctx.assistant, win.assistant_panel, win.store
    groq.script += [groq.tool_call('propose_changes', {'changes': [ADD], 'title': 'Add side cameras',
                                                       'rationale': 'close the side gap'}, 'p1'),
                    groq.answer('Proposed.')]
    _send(app, win, 'Add a camera on each side')
    card = panel.findChild(ProposalCard)
    assert card is not None and card.b_apply.isVisible() and len(card.prop.targets) == 2
    tool = json.loads(groq.requests[-1]['messages'][-1]['content'])
    assert tool['proposal'] == card.prop.id and tool['changes'][0]['camera'] == 'SL'
    n, n0 = len(st.project.variant.cameras), st.undo.index()
    QTest.mouseClick(card.b_apply, Qt.MouseButton.LeftButton)
    pump(app, 0.2)
    v = st.project.variant
    assert len(v.cameras) == n + 2 and v.by_tag('SL') is not None and v.by_tag('SL').z == pytest.approx(0.95)
    assert st.undo.index() == n0 + 1 and 'Assistant' in st.undo.undoText()
    assert card.b_undo.isVisible()
    QTest.mouseClick(card.b_undo, Qt.MouseButton.LeftButton)     # the card's Undo is Edit ▸ Undo
    pump(app, 0.2)
    assert len(st.project.variant.cameras) == n and st.project.variant.by_tag('SL') is None
    assert card.b_apply.isVisible() and st.undo.index() == n0    # pending again: it can be applied again
    st.undo.redo()
    pump(app, 0.2)
    assert st.project.variant.by_tag('SL') is not None
    st.undo.undo()
    pump(app, 0.2)
    # as a new variant: the copy gets the cameras (their own uids, still mirrored), the original stays as it was
    ctl.proposals[card.prop.id]['state'] = 'pending'
    orig = st.project.active_uid
    assert ctl.apply(card.prop.id, as_variant=True) == ''
    pump(app, 0.2)
    v = st.project.variant
    assert v.uid != orig and len(v.cameras) == n + 2
    sl, sr = v.by_tag('SL'), next(c for c in v.cameras if c.link_uid == v.by_tag('SL').uid)
    assert sl.uid not in card.prop.targets and sr.link_uid == sl.uid
    assert len(st.project.variant_by_uid(orig).cameras) == n
    st.undo.undo()
    pump(app, 0.2)
    # discarding it records the idea in the design log
    ctl.proposals[card.prop.id]['state'] = 'pending'
    ctl.discard(card.prop.id)
    pump(app, 0.1)
    e = ctl.log()[-1]
    assert e['kind'] == 'rejected' and e['text'].startswith('rejected: added SL') and '"Add side cameras"' in e['text']
    assert 'SL' not in [c.tag for c in st.project.variant.cameras] and len(M.entries(st.project)) == 1
