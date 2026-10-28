"""The Claude provider through Anthropic's official SDK, against tests/fake_anthropic.py (a local server streaming the
Messages API's server-sent events; never the real API): the request as the documentation says it must be for
claude-opus-5-5 and claude-sonnet-5-5 (adaptive thinking, an explicit effort, no sampling or budget parameters,
tool_choice auto with strict and eager-streamed tools, prompt caching, the compaction and fallback betas), parallel
tool calls answered in one message, the append-only history with the thinking blocks passed back unchanged, the
stop reasons checked before any tool runs (max_tokens, refusal), the typed errors, the usage line, plans proposed
through Claude, and what is never sent. The SDK is optional: without it these tests skip."""
import json
import os
import threading
import time

import pytest

anthropic = pytest.importorskip('anthropic', reason='the anthropic SDK is not installed in this Python (it is an '
                                                    'optional dependency: pip install "anthropic==2.3.0"; the Claude '
                                                    'tests run in an environment that has it)')

from fake_anthropic import FakeAnthropic                       # noqa: E402
from vantage.assistant import agent as A                       # noqa: E402
from vantage.assistant import claude as CL                     # noqa: E402
from vantage.assistant import copilot as CP                    # noqa: E402
from vantage.assistant import providers as PR                  # noqa: E402
from vantage.assistant import secrets as K                     # noqa: E402
from vantage.assistant import tools as T                       # noqa: E402
from vantage.core import rules, samples                        # noqa: E402
from vantage.core.analysis import Engine                       # noqa: E402

KEY = 'sk-ant-test-key'


@pytest.fixture()
def srv(monkeypatch):
    s = FakeAnthropic(key=KEY).start()
    monkeypatch.setenv('VANTAGE_CLAUDE_URL', s.url)
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


def _session(model='claude-opus-5-5', **kw):
    return CL.ClaudeSession(KEY, A.Settings(model, **kw))


def _run(s, tc, text='How is it?'):
    ev = []
    s.run_turn(text, tc, ev.append, threading.Event())
    return ev


def _errors(ev):
    return [e.data for e in ev if e.kind == 'error']


# ============================================================================================ the request
def test_the_request_is_what_the_documentation_asks_for(srv, sample):
    s = _session()
    srv.script += [srv.reply(text='A plain answer.')]
    ev = _run(s, _tc(sample), 'Why is the ground blind?')
    assert ev[-1].data['ok'], _errors(ev)
    body = srv.requests[0]
    assert body['model'] == 'claude-opus-5-5' and body['max_tokens'] == CL.MAX_TOKENS
    assert body['thinking'] == {'type': 'adaptive', 'display': 'summarized'}
    assert body['output_config'] == {'effort': 'medium'}                    # a plain question
    for never in ('temperature', 'top_p', 'top_k'):                         # 400s on these models
        assert never not in body, never
    assert 'budget_tokens' not in body['thinking'] and set(body['thinking']) <= {'type', 'display'}
    assert body['tool_choice'] == {'type': 'auto'}
    names = [t['name'] for t in body['tools']]
    assert names == sorted(names) and len(names) == len(T.TOOLS)           # a fixed order
    assert all(t['eager_input_streaming'] is True for t in body['tools'])
    strict = [t for t in body['tools'] if t.get('strict')]
    assert {t['name'] for t in body['tools']} - {t['name'] for t in strict} == {'optimise_camera',
                                                                                'propose_requirements'}

    def closed(o):
        if isinstance(o, dict):
            if o.get('type') == 'object':
                assert o.get('additionalProperties') is False and isinstance(o.get('properties'), dict)
            for v in o.values():
                closed(v)
        elif isinstance(o, list):
            for v in o:
                closed(v)
    for t in strict:
        closed(t['input_schema'])
    pp = next(t for t in body['tools'] if t['name'] == 'propose_plan')
    assert pp['strict'] and pp['input_schema']['required'] == ['title', 'rationale', 'revise', 'steps']
    # prompt caching: a frozen system prompt with its breakpoint, automatic caching of the tail
    assert body['system'][0]['cache_control'] == {'type': 'ephemeral'} and body['cache_control'] == {'type': 'ephemeral'}
    assert 'THE COPILOT' in body['system'][0]['text'] and '[Vantage context:' not in body['system'][0]['text']
    assert body['context_management'] == {'edits': [{'type': 'compact_20260112'}]}
    assert body['fallbacks'] == 'default'
    headers = srv.raw[-1][0]
    betas = {k.lower(): v for k, v in headers.items()}['anthropic-beta'].split(',')
    assert set(betas) == {'compact-2026-01-12', 'server-side-fallback-2026-07-01'}
    # a planning turn thinks harder; the system prompt and the tools stay byte for byte the same
    srv.script += [srv.reply(text='Planned.')]
    s.settings.planning = True
    _run(s, _tc(sample), 'a 10 s drive in Town10')
    b2 = srv.requests[1]
    assert b2['output_config'] == {'effort': 'high'}
    assert json.dumps(b2['system']) == json.dumps(body['system']) and json.dumps(b2['tools']) == json.dumps(body['tools'])
    # the fallback is a setting (on by default); off, neither the parameter nor its beta is sent
    s.settings.fallbacks = False
    srv.script += [srv.reply(text='x')]
    _run(s, _tc(sample), 'again')
    assert 'fallbacks' not in srv.requests[2]
    betas = {k.lower(): v for k, v in srv.raw[-1][0].items()}['anthropic-beta'].split(',')
    assert betas == ['compact-2026-01-12']
    # the Sonnet model by its exact id
    s2 = _session('claude-sonnet-5-5', show_reasoning=False)
    srv.script += [srv.reply(text='y')]
    _run(s2, _tc(sample))
    assert srv.requests[3]['model'] == 'claude-sonnet-5-5' and srv.requests[3]['thinking'] == {'type': 'adaptive'}


def test_parallel_tool_calls_one_result_message_and_an_append_only_history(srv, sample):
    s = _session()
    srv.script += [srv.reply(thinking='Two lookups at once.', text='Checking.',
                             tools=[('get_issues', {}, 'toolu_a'), ('explain_point', {'x': 'left', 'y': 1}, 'toolu_b')]),
                   srv.reply(thinking='Now I know.', text='**Done.** The issues are listed.'),
                   srv.reply(tools=[('get_coverage', {}, 'toolu_c')], thinking='Coverage too.'),
                   srv.reply(text='Coverage is complete.')]
    ev = _run(s, _tc(sample), 'Explain the issues')
    assert ev[-1].data['ok'], _errors(ev)
    ev2 = _run(s, _tc(sample), 'And the coverage?')
    assert ev2[-1].data['ok'], _errors(ev2)
    assert [e.data['text'] for e in ev if e.kind == 'reasoning'][0].startswith('Two lookups')
    r1 = srv.requests[1]['messages']
    assert [m['role'] for m in r1] == ['user', 'assistant', 'user']
    results = r1[2]['content']
    assert [b['type'] for b in results] == ['tool_result', 'tool_result']              # one message, both results
    assert results[0]['tool_use_id'] == 'toolu_a' and not results[0].get('is_error')
    assert results[1]['tool_use_id'] == 'toolu_b' and results[1]['is_error'] is True
    assert 'schema' in results[1]['content']                    # validated before running: 'left' is no number
    # the assistant's content went back unchanged: thinking with its signature, text, tool calls
    sent = srv.sent[0]
    back = r1[1]['content']
    assert [b['type'] for b in back] == [b['type'] for b in sent]
    assert back[0] == {'type': 'thinking', 'thinking': sent[0]['thinking'], 'signature': sent[0]['signature']}
    assert back[2]['input'] == {} and back[3]['input'] == {'x': 'left', 'y': 1}
    # append-only: every request's messages are the previous request's, byte for byte, and more
    for a, b in zip(srv.requests, srv.requests[1:]):
        assert len(b['messages']) > len(a['messages'])
        for ma, mb in zip(a['messages'], b['messages']):
            assert json.dumps(ma, sort_keys=True) == json.dumps(mb, sort_keys=True)
    assert all(json.dumps(x['system']) == json.dumps(srv.requests[0]['system']) for x in srv.requests)
    md = s.transcript_markdown()
    assert '### You' in md and 'Explain the issues' in md and '[Vantage context' not in md and 'Coverage is' in md


def test_stop_reasons_are_checked_before_any_tool_runs(srv, sample):
    s = _session()
    seen = []
    srv.script += [srv.reply(text='Ok.')]
    _run(s, _tc(sample))
    before = s.history()
    # a refusal: shown plainly with its stop_details, nothing run, the turn leaves nothing behind
    srv.script += [srv.refusal('cyber', 'Declined by a safety classifier.', text='Partial')]
    ev = _run(s, _tc(sample, on_proposal=seen.append), 'Something')
    err = _errors(ev)[0]
    assert err['kind'] == 'refusal' and 'cyber' in err['message'] and 'Declined by a safety classifier' in \
        err['message'] and 'Nothing was run' in err['message']
    assert s.history() == before
    # max_tokens with a tool_use: the (possibly cut) input is not run
    srv.script += [srv.reply(tools=[('propose_changes', {'changes': [{'camera': 'QR', 'dpitch': 3}],
                                                         'title': 'Tilt'})], stop='max_tokens')]
    ev = _run(s, _tc(sample, on_proposal=seen.append), 'Tilt QR')
    assert _errors(ev)[0]['kind'] == 'truncated' and not seen and s.history() == before
    # an input the SDK parses but the schema refuses never runs either
    srv.script += [srv.reply(tools=[('propose_changes', {'changes': 'all of them', 'title': 'x'})]),
                   srv.reply(text='Sorry, fixed.')]
    ev = _run(s, _tc(sample, on_proposal=seen.append), 'Tilt QR again')
    assert ev[-1].data['ok'] and not seen
    done = [e for e in ev if e.kind == 'tool_done'][0]
    assert not done.data['ok'] and 'schema' in done.data['result']


def test_typed_errors_and_retries(srv, sample):
    s = CL.ClaudeSession('sk-ant-wrong', A.Settings('claude-opus-5-5'))
    ev = _run(s, _tc(sample))
    assert _errors(ev)[0]['kind'] == 'auth' and 'Preferences' in _errors(ev)[0]['message']
    s = _session()
    srv.script += [srv.error(429, 'slow down', retry_after=0.01)] * 3 + [srv.reply(text='Fine.')]
    ev = _run(s, _tc(sample))
    assert ev[-1].data['ok'], _errors(ev)
    assert any(e.kind == 'status' and 'Rate limited by Claude' in e.data['text'] for e in ev)
    srv.script += [srv.error(529)] * 6
    ev = _run(s, _tc(sample))
    assert _errors(ev)[0]['kind'] == 'overloaded'
    srv.script += [srv.error(400, 'messages: roles must alternate')]
    ev = _run(s, _tc(sample))
    assert _errors(ev)[0]['kind'] == 'bad_request' and 'alternate' in _errors(ev)[0]['message']
    s3 = CL.ClaudeSession(KEY, A.Settings('claude-opus-5-5'), url='http://127.0.0.1:9')
    s3.client = s3.client.with_options(max_retries=0, timeout=2.0)
    ev = _run(s3, _tc(sample))
    assert _errors(ev)[0]['kind'] == 'network'
    with pytest.raises(A.ApiError):
        CL.ClaudeSession('', A.Settings('claude-opus-5-5'))         # never an SDK default key


def test_a_fallback_turn_is_echoed_without_the_declined_thinking(srv, sample):
    s = _session()
    srv.script += [srv.reply(thinking='declined half', fallback=('claude-opus-5-5', 'claude-opus-4-8'),
                             text='Answer from the fallback.'), srv.reply(text='next')]
    ev = _run(s, _tc(sample))
    assert ev[-1].data['ok'], _errors(ev)
    _run(s, _tc(sample), 'more')
    back = srv.requests[1]['messages'][1]['content']
    assert [b['type'] for b in back] == ['fallback', 'text']


def test_usage_line_cache_reads_and_cost(srv, sample):
    s = _session()
    srv.script += [srv.reply(text='one'), srv.reply(text='two')]
    _run(s, _tc(sample))
    _run(s, _tc(sample), 'again')
    u = s.usage
    assert u.provider == 'claude' and u.requests == 2 and u.cache_read == 2000 and u.cache_write == 2300
    line = u.line()
    assert 'cache_read_input_tokens' in line and '$' in line
    want = (2 * 1500 * 4.0 + 2 * 120 * 20.0 + 2300 * 5.0 + 2000 * 0.20) / 1e6
    assert u.cost() == pytest.approx(want)


def test_a_plan_proposed_through_claude(srv, sample, tmp_path, monkeypatch):
    from vantage import workdir
    monkeypatch.setattr(workdir, 'default', lambda: str(tmp_path))
    p = samples.tigor_sample_project()
    v = next(v for v in p.variants if v.sensors)
    p.active_uid = v.uid
    s = _session(planning=True)
    steps = [{'kind': 'sample_dataset', 'label': 'Sample', 'values': [
        {'key': 'duration_s', 'value': '20', 'source': 'request', 'reason': '20 s'},
        {'key': 'speed_kmh', 'value': '40', 'source': 'request', 'reason': '40 km/h'}]}]
    srv.script += [srv.reply(thinking='The user wants a dataset.', tools=[('propose_plan', {
        'title': 'Sample dataset', 'rationale': 'as asked', 'revise': False, 'steps': steps})]),
                   srv.reply(text='It is on the card: approve it to write it.')]
    tc = T.ToolContext(p.snapshot(), None, Engine(), working_dir=str(tmp_path), provider='claude',
                       model='claude-opus-5-5')
    ev = _run(s, tc, 'a 20 s sample dataset at 40 km/h')
    assert ev[-1].data['ok'], _errors(ev)
    plan = next(e.data['plan'] for e in ev if e.kind == 'copilot')
    assert plan.ok and plan.provider == 'claude' and plan.steps[0].settings['duration_s'] == 20.0
    assert srv.requests[0]['output_config'] == {'effort': 'high'}
    result = json.loads(srv.requests[1]['messages'][-1]['content'][0]['content'])
    assert result['shown_to_the_user'] and result['ok']


def test_nothing_private_reaches_anthropic(srv, sample, tmp_path, monkeypatch):
    from test_copilot import scan, secrets_and_paths
    from vantage import workdir
    monkeypatch.setattr(workdir, 'default', lambda: str(tmp_path / 'work'))
    p = samples.tigor_sample_project()
    needles = secrets_and_paths(p, tmp_path)
    s = _session()
    srv.script += [srv.reply(tools=[('get_overview', {}), ('get_rig_health', {}), ('get_automation', {}),
                                    ('get_catalogue', {'topic': 'past_runs'})]),
                   srv.reply(text='ok')]
    tc = T.ToolContext(p.snapshot(), None, Engine(), working_dir=str(tmp_path / 'work'))
    ev = _run(s, tc, f'My files are in {tmp_path} and {os.path.expanduser("~")}/Documents')
    assert ev[-1].data['ok'], _errors(ev)
    assert len(srv.raw) == 2
    scan(srv.raw, {k: v for k, v in needles.items() if k != 'groq_key'})
    assert '~/Documents' in srv.raw[0][1].decode()


# ============================================================================================ the panel on Claude
from test_gui import app, win, pump, _wait  # noqa: E402,F401  (fixtures)


def test_the_panel_on_claude_and_switching_back_to_groq(app, win, srv):
    from fake_groq import FakeGroq
    st = win.ctx.settings
    K.save(KEY, 'claude')
    st.setValue('assistant/enabled', True)
    st.setValue('assistant/provider', 'claude')
    st.setValue('assistant/claude_model', 'claude-opus-5-5')
    st.setValue('assistant/consent_claude', True)
    st.setValue('assistant/daily_cap', 0)
    ctl = win.ctx.assistant
    ctl.reload()
    assert ctl.ready and ctl.provider == 'claude' and ctl.model == 'claude-opus-5-5'
    win.show_assistant()
    panel = win.assistant_panel
    srv.script += [srv.reply(thinking='Reading the issues.', tools=[('get_issues', {})]),
                   srv.reply(text='**Two issues.** Both at the rear.')]
    panel.ask('Explain the issues')
    t0 = time.time()
    while ctl.busy and time.time() - t0 < 60:
        pump(app, 0.05)
    pump(app, 0.2)
    assert ctl.session.provider == 'claude' and 'Two issues' in ctl.transcript()
    assert 'Claude' in panel.title.text() and 'cache_read_input_tokens' in panel.foot.text()
    from vantage.ui.assistant import AssistantTurn
    turn = [w for w in panel.findChildren(AssistantTurn)][-1]
    assert 'Claude' in turn.model_lbl.text() and turn.thinking is not None
    sent = panel.show_sent()
    assert 'claude-opus-5-5' in sent.text.toPlainText() and KEY not in sent.text.toPlainText()
    sent.close()
    assert ctl.tokens_today('claude') > 0
    # switching provider starts a new conversation on the other one, with its own key
    g = FakeGroq().start()
    try:
        import os as _os
        _os.environ['VANTAGE_ASSISTANT_URL'] = g.url
        K.save('test-key', 'groq')
        st.setValue('assistant/provider', 'groq')
        st.setValue('assistant/consent_groq', True)
        ctl.reload()
        assert ctl.session is None and ctl.provider == 'groq' and ctl.key == 'test-key'
        g.script.append(g.answer('From Groq.'))
        panel.ask('Hello?')
        t0 = time.time()
        while ctl.busy and time.time() - t0 < 60:
            pump(app, 0.05)
        assert ctl.session.provider == 'groq' and 'From Groq' in ctl.transcript() and g.requests
        assert KEY not in json.dumps(g.requests)
    finally:
        _os.environ.pop('VANTAGE_ASSISTANT_URL', None)
        g.stop()
        K.delete('groq')
        K.delete('claude')
