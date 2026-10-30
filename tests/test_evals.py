"""The assistant benchmark (vantage/assistant/evals.py): the questions' expected numbers computed live on the samples,
the number extraction, the grader's exact partial scores, the dry run through the real loop and tools, short runs
against tests/fake_groq.py and tests/fake_anthropic.py (never a real API), the CLI's consent, key and exit codes,
the project left unchanged by grading, and --compare."""
import copy
import dataclasses
import io
import json
import math
import os

import pytest

from fake_groq import FakeGroq
from vantage.assistant import copilot as CP
from vantage.assistant import evals as E
from vantage.assistant import providers as PR
from vantage.assistant import tools as T

GROQ_KEY = 'test-key'
CLAUDE_KEY = 'sk-ant-test-key'


@pytest.fixture(autouse=True)
def _quiet_machine(monkeypatch):
    """No CARLA server answers and the machine has memory to spare (the Copilot's checks read both)."""
    from vantage.sim import server as SV
    monkeypatch.setattr(SV, 'port_open', lambda *a, **k: False)
    monkeypatch.setattr(CP, 'mem_available_gb', lambda: 30.0)


@pytest.fixture()
def groq(monkeypatch):
    s = FakeGroq(key=GROQ_KEY).start()
    monkeypatch.setenv('VANTAGE_ASSISTANT_URL', s.url)
    yield s
    s.stop()


def _main(argv):
    buf = io.StringIO()
    code = E.main(argv, out=buf)
    return code, buf.getvalue()


def _core():
    return [q for q in E.QUESTIONS if not q.optional]


def _ns(q):
    env = E.Env(q)
    return {'V': env.values(), 'R': {}, 'P': {}, 'env': env}


def _text(step, ns):
    return step.text(ns) if callable(step.text) else step.text.format_map(ns)


# ============================================================================================ the questions
def test_the_question_set_covers_every_category_with_unique_ids():
    ids = [q.id for q in E.QUESTIONS]
    assert len(ids) == len(set(ids)) and 40 <= len(_core()) <= 60
    assert {q.category for q in E.QUESTIONS} == set(E.CATEGORIES)
    for q in E.QUESTIONS:
        assert q.project in E.PROJECTS and all(k in E.SETUPS for k, _ in q.setup)
        assert q.script, q.id                                        # every question has a reference conversation
        assert q.numbers or q.mentions or q.plan or q.proposal or q.must_not, q.id
        for n in q.numbers:
            E.unit_of(n.unit)                                        # every expected unit is one the parser knows


def test_every_expected_number_computes_on_the_samples_and_is_finite():
    for q in E.QUESTIONS:
        env = E.Env(q)
        vals = env.values()
        assert set(vals) == {n.key for n in q.numbers if not n.dynamic}
        assert all(math.isfinite(v) for v in vals.values()), (q.id, vals)
        for m in q.mentions:                                         # the computed mentions too
            rx = m(env) if callable(m) else m
            assert isinstance(rx, str) and rx


def test_expected_values_agree_with_independent_computations():
    """A few expected values against the app's figures by another path (the analysis itself, the sample's data)."""
    from vantage.core import samples
    from vantage.core.analysis import Engine
    p = samples.tigor_sample_project()
    res = Engine().analyse(p)
    env = E.Env(E.BY_ID['coverage.blind'])
    assert env.values()['blind'] == pytest.approx(res.metrics['blind_m2'], abs=1e-9)
    assert env.values()['cov'] == pytest.approx(res.metrics['covered_pct'], abs=1e-9)
    lid = E.Env(E.BY_ID['facts.lidar']).values()
    assert lid == {'ch': 128.0, 'rng': 90.0, 'pps': 1_310_720.0}          # the OS1-128 library entry (1024 × 10 Hz)
    assert E.Env(E.BY_ID['facts.cameras']).values()['n'] == len(p.variant.cameras) == 6
    # QR switched off alone: the analysis' own "if removed" figure equals simulate_changes on that change
    e = E.Env(E.BY_ID['whatif.qr_off'])
    sim = e.tool('simulate_changes', changes=[{'camera': 'QR', 'visible': False, 'mirror_partner': False}])
    assert e.values()['b'] == pytest.approx(sim['effect']['blind_m2']['after'], abs=0.01)


# ============================================================================================ numbers
@pytest.mark.parametrize('text, want', [
    ('0.18 m², 18 cm and 3 px/cm', [(0.18, 'area'), (18.0, 'length'), (3.0, 'density')]),
    ('covers 99.97 % (53%)', [(99.97, 'pct'), (53.0, 'pct')]),
    ('1,310,720 points/s or 1.31 million points per second', [(1310720.0, 'pps'), (1310000.0, 'pps')]),
    ('1 310 720 rays/s', [(1310720.0, 'pps')]),
    ('yaw −98° and -1.5 m; x=-0.8', [(-98.0, 'angle'), (-1.5, 'length'), (-0.8, None)]),
    ('3–5 m, 3 - 5 m, between 2 and 4 m', [(3.0, 'length'), (5.0, 'length')] * 2 + [(2.0, 'length'),
                                                                                   (4.0, 'length')]),
    ('(389, 0, 1073) mm', [(389.0, 'length'), (0.0, 'length'), (1073.0, 'length')]),
    ('OS1-128 has 128 channels; 6-cam', [(128.0, None), (6.0, None)]),
    ('from –40 °C to +85 °C', [(-40.0, 'temp'), (85.0, 'temp')]),
    ('2.03 h; 345 min; 5.75 hours; 20 ms', [(2.03, 'time'), (345.0, 'time'), (5.75, 'time'), (20.0, 'time')]),
    ('960 × 768 px · 25.6 Mpx · 0.5 px/mm · 30.41 m³', [(960.0, 'px'), (768.0, 'px'), (25.6, 'mpx'),
                                                        (0.5, 'density'), (30.41, 'volume')]),
])
def test_number_extraction(text, want):
    got = [(round(q.value, 6), q.family) for q in E.extract(text)]
    assert got == want


def test_unit_conversion_rounding_and_the_subject_nearby():
    f = E.find_number
    assert f('the blind ground is 18 cm²', 0.0018, 'm²', 0.0001)[0] is not None        # cm² -> m²
    assert f('ML reaches 1.42 m', 142, 'cm', 1)[0] is not None                         # m -> cm
    assert f('about 0.5 px/mm', 5.0, 'px/cm', 0.05)[0] is not None                     # px/mm -> px/cm
    assert f('the runtime is 5.75 hours', 345.2, 'min', 1.0)[0] is not None           # h -> min
    assert f('minimum 3.1 px/cm', 3.116, 'px/cm', 0.05)[0] is not None                # shown rounding widens
    assert f('minimum 3 px/cm', 3.116, 'px/cm', 0.05)[0] is None                      # ... at most to 2 × tol
    assert f('minimum 3.12', 3.116, 'px/cm', 0.05)[0] is None                         # a bare number has no unit
    assert f('0.073 m² blind', 0.073, '', 0.001)[0] is None                           # a count is unit-less only
    assert f('6 cameras', 6, '', 0, r'camera')[0] is not None
    # the subject: a number of the right kind next to another subject does not count
    t = 'ML sees 9.36 m²; on the far side MR sees 9.26 m²'
    assert f(t, 9.26, 'm²', 0.005, r'\bMR\b')[0] is not None
    hit, cands = f('QL 9.36 m² in front', 9.36, 'm²', 0.005, r'\bMR\b')
    assert hit is None and cands == []
    # a table: the unit and the subject come from the column header and the row
    tab = '| Variant | Blind (m²) |\n|---|---|\n| 4-cam | 0.073 |\n| 6-cam · v3 | 0.010 |'
    assert f(tab, 0.073, 'm²', 0.001, r'4-cam')[0] is not None
    assert f(tab, 0.010, 'm²', 0.001, r'4-cam')[0] is None
    # a heading just above a bullet names its subject
    assert f('**ML**:\n- seen 9.36 m²', 9.36, 'm²', 0.005, r'\bML\b')[0] is not None


# ============================================================================================ the grader
def _run(q, **kw):
    return E.run_question(q, E.DRY, working_dir=kw.pop('work', ''), **kw)


def _scores(rec):
    return {c['kind'] + ':' + c['label']: c['score'] for c in rec['checks']}


def test_a_reference_answer_scores_one_and_wrong_ones_score_exactly_less(tmp_path, monkeypatch):
    q = E.BY_ID['whatif.mirror_pitch']        # tools_all, tools_none, 1 number, 2 must-nots, side effects: 6 checks
    w = str(tmp_path)
    ok = _run(q, work=w)
    assert ok['status'] == 'ok' and ok['score'] == 1.0 and len(ok['checks']) == 6
    sim = q.script[0][0]
    wrong_number = dataclasses.replace(q, script=[[sim, E.Say('Seen by two or more cameras: 61.0 %.')]])
    assert _run(wrong_number, work=w)['score'] == pytest.approx(5 / 6, abs=1e-4)
    no_tool = dataclasses.replace(q, script=[[E.Say('Seen by two or more: {V[multi]:.1f} %.')]])
    r = _run(no_tool, work=w)
    assert r['score'] == pytest.approx(5 / 6, abs=1e-4) and _scores(r)['tools_all:ran simulate_changes'] == 0
    proposed = dataclasses.replace(q, script=[[sim, E.call('propose_changes', {'changes': [{'camera': 'ML',
                                                                                            'dpitch': 5}]}),
                                               q.script[0][1]]])
    r = _run(proposed, work=w)
    assert r['score'] == pytest.approx(5 / 6, abs=1e-4) and _scores(r)['tools_none:did not call propose_changes'] == 0
    claim = dataclasses.replace(q, script=[[sim, E.Say('I have pitched ML and MR down; two or more now see '
                                                       '{V[multi]:.1f} %.')]])
    r = _run(claim, work=w)
    assert r['score'] == pytest.approx(5 / 6, abs=1e-4)
    assert [c['score'] for c in r['checks'] if c['kind'] == 'must_not'] == [0.0, 1.0]
    # a side effect: a tool that edits the snapshot it was given
    tool = T.BY_NAME['simulate_changes']
    real = tool.fn

    def editing(tc, **kw):
        tc.project.name += ' (edited)'                      # an edit that leaves the figures as they were
        return real(tc, **kw)
    monkeypatch.setattr(tool, 'fn', editing)
    r = _run(q, work=w)
    assert r['score'] == pytest.approx(5 / 6, abs=1e-4) and _scores(r)['side_effects:the project is unchanged'] == 0


def test_plan_and_proposal_checks_give_exact_partial_credit(tmp_path):
    w = str(tmp_path)
    q = E.BY_ID['copilot.export']    # propose_plan, 2 must-nots, plan kind/draft/valid/formats, side effects: 8
    assert _run(q, work=w)['score'] == 1.0
    one = dataclasses.replace(q, script=[[E.call('propose_plan', {
        'title': 'x', 'rationale': 'x', 'revise': False, 'steps': [{'kind': 'export', 'label': 'x', 'values': [
            E._pv('formats', ['extrinsics_json'])]}]}), E.Say('On the card.')]])
    r = _run(one, work=w)
    assert len(r['checks']) == 8 and r['score'] == pytest.approx(7 / 8, abs=1e-4)
    assert _scores(r)['plan:export.formats'] == 0
    nothing = dataclasses.replace(q, script=[[E.Say('I have started the export.')]])
    r = _run(nothing, work=w)            # no plan: kind, draft, valid, formats and the tool fail; a claim of a run
    assert r['score'] == pytest.approx(2 / 8, abs=1e-4)
    g = E.BY_ID['goal.four_cam_blind']   # 9 checks; without a proposal: tools_all ½, the proposal's 3 and its number 0
    steps = [s for s in g.script[0] if not (isinstance(s, E.Call) and s.calls[0][0] == 'propose_changes')]
    r = _run(dataclasses.replace(g, script=[steps[:-1] + [E.Say('I found nothing better.')]]), work=w)
    assert len(r['checks']) == 9 and r['score'] == pytest.approx(4.5 / 9, abs=1e-4)
    ref = _run(g, work=w)
    assert ref['score'] == 1.0 and ref['proposals']
    eff = {e['key']: e for e in ref['proposals'][0]['effects']}
    assert eff['blind_m2']['after'] < eff['blind_m2']['before']
    # the dynamic number is the proposal's own predicted figure, and the same change re-simulated agrees
    env = E.Env(g)
    rows = ref['proposals'][0]['rows']
    ml = {r_['prop']: r_['after'] - r_['before'] for r_ in rows if r_['camera'] == 'ML'}
    again = env.tool('simulate_changes', changes=[{'camera': 'ML', **{'d' + k: v for k, v in ml.items()}}])
    assert again['effect']['blind_m2']['after'] == pytest.approx(eff['blind_m2']['after'], abs=1e-3)


def test_a_follow_up_revises_the_same_plan(tmp_path):
    r = _run(E.BY_ID['copilot.revise'], work=str(tmp_path))
    assert r['score'] == 1.0 and len(r['turns']) == 2
    assert [p['version'] for p in r['plans']] == [1, 2] and len({p['id'] for p in r['plans']}) == 1
    assert r['plans'][-1]['status'] == 'draft'


def test_grading_leaves_the_project_unchanged(tmp_path):
    q = E.BY_ID['studies.stitched']
    env = E.Env(q)
    before = E.project_hash(env.project)
    rec = _run(q, work=str(tmp_path))
    E.grade(q, env, copy.deepcopy(rec) | {'_proposals': [], '_plans': [], '_hash': rec['_hash']})
    env.values()
    assert E.project_hash(env.project) == before and rec['score'] == 1.0
    assert not CP.records(env.project)


def test_a_question_needing_an_absent_tool_is_not_applicable(monkeypatch, tmp_path):
    q = dataclasses.replace(E.BY_ID['coverage.blind'], id='coverage.blind', needs=('no_such_tool',))
    r = _run(q, work=str(tmp_path))
    assert r['status'] == 'n/a' and r['score'] is None and 'no_such_tool' in r['error']
    s = E.summarise([r, _run(E.BY_ID['coverage.density'], work=str(tmp_path))])
    assert s['n_a'] == 1 and s['scored'] == 1 and s['overall'] == 1.0


# ============================================================================================ the dry run
@pytest.fixture(scope='module')
def dry(tmp_path_factory):
    from vantage.sim import server as SV
    mp = pytest.MonkeyPatch()
    mp.setattr(SV, 'port_open', lambda *a, **k: False)
    mp.setattr(CP, 'mem_available_gb', lambda: 30.0)
    out = str(tmp_path_factory.mktemp('dry'))
    code, text = _main(['--dry-run', '--out', out])
    mp.undo()
    with open(os.path.join(out, 'results.json'), encoding='utf-8') as f:
        doc = json.load(f)
    return code, text, out, doc


def test_the_full_dry_run_scores_every_reference_answer_one(dry):
    code, text, out, doc = dry
    assert code == 0 and doc['schema'] == E.SCHEMA and doc['provider'] == E.DRY
    recs = {r['id']: r for r in doc['records']}
    assert set(recs) == {q.id for q in E.QUESTIONS}
    for q in _core():
        r = recs[q.id]
        assert r['status'] == 'ok' and r['score'] == 1.0, (q.id, [c for c in r['checks'] if c['score'] < 1])
    for q in E.QUESTIONS:              # the optional questions run when their tools are offered, else n/a
        r = recs[q.id]
        assert r['status'] == 'n/a' or 0.0 <= r['score'] <= 1.0
    assert doc['summary']['categories'].keys() == set(E.CATEGORIES)
    # dynamic expected numbers (a proposal's predicted effect) were computed and are finite
    goal = recs['goal.four_cam_blind']
    num = next(c for c in goal['checks'] if c['kind'] == 'number')
    assert num['score'] == 1.0 and 'expected' in num['detail']
    # the outputs: JSON, the Markdown report, a transcript per question
    rep = open(os.path.join(out, 'report.md'), encoding='utf-8').read()
    assert '**Overall: 100 %**' in rep and 'Grading limits' in rep and 'dry run' in rep
    for q in _core():
        assert os.path.isfile(os.path.join(out, 'transcripts', q.id + '.md'))
        assert os.path.isfile(os.path.join(out, 'transcripts', q.id + '.json'))
    assert doc['totals']['requests'] == sum(r['requests'] for r in doc['records']) > len(_core())


def test_the_dry_run_measures_the_requests_exactly_as_they_would_be_sent(dry):
    _, _, _, doc = dry
    rec = next(r for r in doc['records'] if r['id'] == 'coverage.blind')
    req = E.first_request(E.BY_ID['coverage.blind'], PR.GROQ, 'reference')
    assert rec['sent_chars'][0] == len(json.dumps(req, ensure_ascii=False, default=str))
    assert rec['usage']['estimated'] and rec['usage']['input_tokens'] == sum(c // 4 for c in rec['sent_chars'])


def test_estimate_ranges_and_prices():
    qs = _core()
    g = E.estimate(qs, PR.GROQ, 'openai/gpt-oss-120b')
    assert g['cost_usd'] is None and g['requests'][0] < g['requests'][1]
    assert g['input_tokens'][0] < g['input_tokens'][1] and 'Groq: no cost' in g['method']
    gp = E.estimate(qs, PR.GROQ, 'openai/gpt-oss-120b', {'input': 0.15, 'output': 0.60})
    lo_in, hi_in = gp['input_tokens']
    assert gp['cost_usd'][0] == pytest.approx((lo_in * 0.15 + gp['output_tokens'][0] * 0.6) / 1e6, abs=0.01)
    for model in PR.CLAUDE_MODELS:
        c = E.estimate(qs, PR.CLAUDE, model)
        assert 0 < c['cost_usd'][0] < c['cost_usd'][1]
    assert E.estimate(qs, PR.CLAUDE, 'claude-opus-5-5')['cost_usd'][1] > \
        E.estimate(qs, PR.CLAUDE, 'claude-sonnet-5-5')['cost_usd'][1]
    two = E.estimate(qs[:3], PR.GROQ, 'openai/gpt-oss-120b', repeat=2)
    one = E.estimate(qs[:3], PR.GROQ, 'openai/gpt-oss-120b')
    assert two['requests'] == [2 * one['requests'][0], 2 * one['requests'][1]]


def test_repeat_reports_the_spread(tmp_path):
    out = str(tmp_path / 'r')
    code, _ = _main(['--dry-run', '--only', 'coverage.blind', '--repeat', '2', '--out', out])
    doc = json.load(open(os.path.join(out, 'results.json'), encoding='utf-8'))
    assert code == 0 and len(doc['records']) == 2
    row = doc['summary']['questions'][0]
    assert row['runs'] == 2 and row['sd'] == 0.0 and row['score'] == 1.0
    assert os.path.isfile(os.path.join(out, 'transcripts', 'coverage.blind-2.md'))


def test_min_score_exit_code(tmp_path, monkeypatch):
    assert _main(['--dry-run', '--only', 'coverage.blind', '--min-score', '1', '--out', str(tmp_path / 'a')])[0] == 0
    q = E.BY_ID['coverage.blind']
    monkeypatch.setattr(q, 'script', [[E.Say('Blind: 0.5 m², covered 12 %.')]])
    code, text = _main(['--dry-run', '--only', 'coverage.blind', '--min-score', '0.9', '--out', str(tmp_path / 'b')])
    assert code == E.EXIT_BELOW and 'Below --min-score' in text


# ============================================================================================ real runs, faked
GROQ_QS = ('coverage.blind', 'whatif.mirror_pitch', 'copilot.export', 'facts.lidar')


def _script_fake(srv, ids, claude=False):
    """The reference conversations as the fake server's scripted replies."""
    n = 0
    for qid in ids:
        q = E.BY_ID[qid]
        ns = _ns(q)
        for turn in q.script:
            for step in turn:
                n += 1
                if isinstance(step, E.Call):
                    calls = [(name, a(ns) if callable(a) else a) for name, a in step.calls]
                    if claude:
                        srv.script.append(srv.reply(tools=calls, thinking='Reading the figures.'))
                    else:
                        assert len(calls) == 1
                        srv.script.append(srv.tool_call(calls[0][0], calls[0][1], f'c{n}'))
                else:
                    text = _text(step, ns)
                    srv.script.append(srv.reply(text=text) if claude else srv.answer(text, reasoning='Done.'))


def test_a_short_run_against_the_fake_groq_server(groq, monkeypatch, tmp_path):
    monkeypatch.setenv('GROQ_API_KEY', GROQ_KEY)
    _script_fake(groq, GROQ_QS)
    out = str(tmp_path / 'groq')
    code, text = _main(['--provider', 'groq', '--model', 'openai/gpt-oss-120b', '--only', ','.join(GROQ_QS),
                        '--out', out, '--yes', '--groq-price-in', '0.15', '--groq-price-out', '0.60'])
    assert code == 0, text
    assert 'What is sent' in text and 'Key: from the GROQ_API_KEY environment variable (not shown)' in text
    assert GROQ_KEY not in text
    doc = json.load(open(os.path.join(out, 'results.json'), encoding='utf-8'))
    assert doc['provider'] == 'groq' and doc['model'] == 'openai/gpt-oss-120b'
    recs = {r['id']: r for r in doc['records']}
    assert all(recs[i]['score'] == 1.0 for i in GROQ_QS), {i: recs[i]['checks'] for i in GROQ_QS}
    n_req = sum(len(t) for i in GROQ_QS for t in E.BY_ID[i].script)
    assert len(groq.requests) == n_req and not groq.script
    r = recs['coverage.blind']                          # the fake's usage: 1200 in, 85 out per request
    assert r['usage']['input_tokens'] == 1200 * r['requests'] and not r['usage']['estimated']
    assert r['cost_usd'] == pytest.approx((1200 * 0.15 + 85 * 0.6) * r['requests'] / 1e6)
    assert len(r['sent_chars']) == r['requests'] and r['seconds'] >= 0      # the fake can answer in < 5 ms
    assert os.path.isfile(os.path.join(out, 'report.md')) and os.path.isfile(os.path.join(out, 'preview',
                                                                                         'coverage.blind.json'))
    # what was sent: the panel's system prompt and tools; never the key in a body, the working folder or this home
    first = groq.requests[0]
    assert first['messages'][0]['content'].startswith('You are the design assistant built into Vantage')
    assert {t['function']['name'] for t in first['tools']} == E.offered_tools(PR.GROQ)
    preview = json.load(open(os.path.join(out, 'preview', 'coverage.blind.json'), encoding='utf-8'))
    assert preview['messages'] == first['messages'] and preview['tools'] == first['tools']
    for _, body in groq.raw:
        s = body.decode('utf-8', 'replace')
        assert GROQ_KEY not in s and str(tmp_path) not in s and os.path.expanduser('~') not in s


def test_yes_is_required_and_nothing_is_sent_without_it(groq, monkeypatch, tmp_path):
    monkeypatch.setenv('GROQ_API_KEY', GROQ_KEY)
    code, text = _main(['--provider', 'groq', '--only', 'coverage.blind', '--out', str(tmp_path)])
    assert code == E.EXIT_USAGE and 'Add --yes' in text and 'Estimate:' in text
    assert groq.raw == [] and groq.requests == []
    assert os.path.isfile(os.path.join(str(tmp_path), 'preview', 'coverage.blind.json'))


def test_no_key_exits_3_and_sends_nothing(groq, monkeypatch, tmp_path):
    from vantage.assistant import secrets as K
    monkeypatch.delenv('GROQ_API_KEY', raising=False)
    K.delete('groq')
    assert K.load('groq') == (None, None)
    code, text = _main(['--provider', 'groq', '--only', 'coverage.blind', '--out', str(tmp_path), '--yes'])
    assert code == E.EXIT_UNAVAILABLE and 'No Groq API key' in text and groq.raw == []


def test_a_refused_key_stops_the_run_with_exit_3(groq, monkeypatch, tmp_path):
    monkeypatch.setenv('GROQ_API_KEY', 'wrong-key')
    code, text = _main(['--provider', 'groq', '--only', 'coverage.blind,coverage.density', '--out', str(tmp_path),
                        '--yes'])
    assert code == E.EXIT_UNAVAILABLE and 'Stopped' in text
    assert len(groq.raw) == 1                                       # one request, then it stopped
    assert os.path.isfile(os.path.join(str(tmp_path), 'results.json'))


def test_a_turn_that_runs_out_of_time_scores_zero(groq, monkeypatch, tmp_path):
    monkeypatch.setenv('GROQ_API_KEY', GROQ_KEY)
    groq.delay = 0.2
    groq.script.append(groq.answer('The blind ground is 0.010 m² and the ROI is 99.97 % covered. ' * 4))
    code, text = _main(['--provider', 'groq', '--only', 'coverage.blind', '--out', str(tmp_path), '--yes',
                        '--timeout', '0.5'])
    doc = json.load(open(os.path.join(str(tmp_path), 'results.json'), encoding='utf-8'))
    r = doc['records'][0]
    assert code == 0 and r['status'] == 'timeout' and r['score'] == 0.0 and doc['summary']['errors'] == 1


def test_list_sends_nothing(groq, monkeypatch):
    monkeypatch.setenv('GROQ_API_KEY', GROQ_KEY)
    code, text = _main(['--list'])
    assert code == 0 and groq.raw == []
    for q in E.QUESTIONS:
        assert q.id in text
    assert 'checks:' in text and 'nothing is sent' in text
    code, text = _main(['--list', '--category', 'copilot'])
    assert code == 0 and 'copilot.drive' in text and 'coverage.blind' not in text


@pytest.mark.parametrize('argv', [['--only', 'no.such'], ['--category', 'nope'], ['--repeat', '0'],
                                  ['--min-score', '7'], ['--bogus'], ['--compare', 'a.json']])
def test_usage_errors_exit_2(argv):
    assert _main(argv)[0] == E.EXIT_USAGE


def test_claude_without_the_sdk_is_unavailable(tmp_path, monkeypatch):
    monkeypatch.setenv('ANTHROPIC_API_KEY', CLAUDE_KEY)
    monkeypatch.setattr(PR, 'claude_available', lambda: False)
    code, text = _main(['--provider', 'claude', '--only', 'coverage.blind', '--out', str(tmp_path), '--yes'])
    assert code == E.EXIT_UNAVAILABLE and 'anthropic' in text


def test_a_short_run_against_the_fake_anthropic_server(monkeypatch, tmp_path):
    pytest.importorskip('anthropic', reason='the Claude run needs the anthropic SDK (the qt67 environment has it)')
    from fake_anthropic import FakeAnthropic
    srv = FakeAnthropic(key=CLAUDE_KEY).start()
    try:
        monkeypatch.setenv('VANTAGE_CLAUDE_URL', srv.url)
        monkeypatch.setenv('ANTHROPIC_API_KEY', CLAUDE_KEY)
        ids = ('coverage.blind', 'copilot.export', 'studies.gnss')
        _script_fake(srv, ids, claude=True)
        out = str(tmp_path / 'claude')
        code, text = _main(['--provider', 'claude', '--model', 'claude-sonnet-5-5', '--only', ','.join(ids),
                            '--out', out, '--yes'])
        assert code == 0, text
        doc = json.load(open(os.path.join(out, 'results.json'), encoding='utf-8'))
        recs = {r['id']: r for r in doc['records']}
        assert all(recs[i]['score'] == 1.0 for i in ids), {i: recs[i]['checks'] for i in ids}
        assert len(srv.requests) == sum(len(t) for i in ids for t in E.BY_ID[i].script)
        body = srv.requests[0]
        assert body['model'] == 'claude-sonnet-5-5' and body['output_config'] == {'effort': 'medium'}
        assert {t['name'] for t in body['tools']} == E.offered_tools(PR.CLAUDE)
        plan_req = next(b for b in srv.requests if 'extrinsics' in json.dumps(b['messages'][0]))
        assert plan_req['output_config'] == {'effort': 'high'}            # a Copilot turn, as in the panel
        r = recs['coverage.blind']
        assert r['usage']['cache_write'] > 0 and r['cost_usd'] and r['cost_usd'] > 0
        assert '$' in text                                               # the estimate priced from providers.py
        for _, raw in srv.raw:
            assert CLAUDE_KEY.encode() not in raw and str(tmp_path).encode() not in raw
    finally:
        srv.stop()


# ============================================================================================ compare
def test_compare_two_runs(dry, tmp_path):
    _, _, out, doc = dry
    b = copy.deepcopy(doc)
    row = next(r for r in b['summary']['questions'] if r['id'] == 'coverage.blind')
    row['score'] = 0.5
    b['summary']['categories']['coverage']['score'] = 0.9
    b['summary']['overall'] = 0.97
    b['model'] = 'other-model'
    pb = str(tmp_path / 'b.json')
    json.dump(b, open(pb, 'w', encoding='utf-8'))
    text, cmp_doc = E.compare(doc, b)
    q = next(x for x in cmp_doc['questions'] if x['id'] == 'coverage.blind')
    assert q['a'] == 1.0 and q['b'] == 0.5 and q['delta'] == -0.5
    assert cmp_doc['categories']['coverage']['delta'] == pytest.approx(-0.1)
    assert '| coverage.blind | 100 % | 50 % | -50 ▼ |' in text and 'other-model' in text
    code, printed = _main(['--compare', os.path.join(out, 'results.json'), pb, '--out', str(tmp_path / 'c')])
    assert code == 0 and printed.strip() == text.strip()
    assert os.path.isfile(str(tmp_path / 'c' / 'compare.md'))
    bad = str(tmp_path / 'bad.json')
    json.dump({'schema': 'something/else'}, open(bad, 'w', encoding='utf-8'))
    assert _main(['--compare', bad, pb])[0] == E.EXIT_USAGE
