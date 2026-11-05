"""Round 15: the Copilot plans and starts round 14's studies (rig perception, foundation-model range, robust
perception, re-drive, the design agent, the learned model's training and its campaign, risk-weighted coverage, the
fleet twin). Natural requests map to each kind both through a (fake) model and through Vantage's own deterministic
planner; every value states its source; the checks refuse what cannot run (no CARLA, no PyTorch, no model); nothing
runs without an approved, intact record (the gate); each kind runs on fixtures through the headless runner and its
result reaches the read tools and, in the app, the page; cancel stops it; a saved plan runs from its record.

Never a real API (tests/fake_groq.py), never a CARLA server (tests/fake_carla through the bridge in-process), never
the user's data folder or model store (VANTAGE_PERF_DIR and the model folders are the test's own)."""
import json
import os
import sys
import threading
import time

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from fake_groq import FakeGroq  # noqa: E402
from vantage.assistant import agent as A  # noqa: E402
from vantage.assistant import copilot as CP  # noqa: E402
from vantage.assistant import copilot_tools as CT  # noqa: E402
from vantage.assistant import r14_tools as R14  # noqa: E402
from vantage.assistant import tools as T  # noqa: E402
from vantage.core import samples  # noqa: E402
from vantage.core.analysis import Engine  # noqa: E402


def V(key, value, source='request', reason='you asked for it'):
    return {'key': key, 'value': json.dumps(value), 'source': source, 'reason': reason}


def _rig():
    p = samples.tigor_sample_project()
    v = next(v for v in p.variants if v.sensors)
    p.active_uid = v.uid
    return p


def _ab(p):
    """Two named rigs: 'Rig A' (the data rig) and 'Rig B' (its copy with the cameras 20° up)."""
    from vantage.core.project import Variant
    a = p.variant
    a.name = 'Rig A'
    b = Variant.from_dict(a.to_dict())
    b.uid, b.name = 'rig-b-uid', 'Rig B'
    for c in b.cameras:
        c.pitch -= 20.0
    p.variants.append(b)
    return p


@pytest.fixture(autouse=True)
def _machine(tmp_path, monkeypatch):
    """No CARLA anywhere (no server, no client Python, no folder), memory to spare, and the test's own model folders
    and record store."""
    from vantage.core import detection as DT
    from vantage.core import foundation_models as FM
    from vantage.core import perf_model as PM
    from vantage.core import simcampaign as SC
    from vantage.sim import server as SV
    monkeypatch.setattr(SV, 'port_open', lambda *a, **k: False)
    monkeypatch.setattr(SC, 'carla_python', lambda: '')
    monkeypatch.setattr(CP, '_carla_root_found', lambda: False)
    monkeypatch.setattr(CP, 'mem_available_gb', lambda: 30.0)
    monkeypatch.setenv('VANTAGE_PERF_DIR', str(tmp_path / 'perf'))
    monkeypatch.setenv(DT.ENV_DIR, str(tmp_path / 'models'))
    monkeypatch.setenv(FM.ENV_DIR, str(tmp_path / 'foundation'))
    PM._STORES.clear()
    yield
    PM._STORES.clear()


@pytest.fixture()
def work(tmp_path, monkeypatch):
    from vantage import workdir
    d = tmp_path / 'work'
    d.mkdir()
    monkeypatch.setattr(workdir, 'default', lambda: str(d))
    return str(d)


@pytest.fixture()
def models_ok(monkeypatch):
    """The 2D detector counts as downloaded (the tests hand the study a fake one)."""
    from vantage.core import detection as DT
    real = DT.status
    monkeypatch.setattr(DT, 'status', lambda m: dict(real(m), present=True, verified=True))


@pytest.fixture()
def carla_ok(monkeypatch):
    """A CARLA server answers (the fake carla stands in for it through the hooks' runner)."""
    from vantage.core import simcampaign as SC
    from vantage.sim import server as SV
    monkeypatch.setattr(SV, 'port_open', lambda *a, **k: True)
    monkeypatch.setattr(SC, 'carla_python', lambda: sys.executable)


def _recs_ab(p, work):
    """Rig A (F and L) and Rig B (F only) recordings with ground truth in the project's runs folder."""
    from test_rig_perception import make_rec
    from vantage import workdir
    root = workdir.project_dir(p, work, 'carla')
    a, b = os.path.join(root, 'rig-a'), os.path.join(root, 'rig-b')
    la = make_rec(a, 'Rig A', ('F', 'L'))
    make_rec(b, 'Rig B', ('F',))
    t = time.time()
    os.utime(os.path.join(a, 'run.json'), (t - 10, t - 10))
    os.utime(os.path.join(b, 'run.json'), (t, t))
    return a, b, la


def _approved(p, steps, work, engine=None):
    plan = CP.build(p, steps, 'Test plan', 'as asked', request='test', working_dir=work, engine=engine)
    assert plan.ok, plan.errors
    rec, errs = CP.approve(p, plan, working_dir=work, engine=engine)
    assert rec is not None, errs
    CP.add_record(p, rec)
    return CP.find(p, rec['id'], rec['version'])


def _run(p, rec, out, hooks=None, eng=None, i=0, cancel=None):
    """As the plan runner (app or batch) does: a run in the record, the step through run_study_kind, its summary kept."""
    run = CP.start_run(rec, 'test')
    s = rec['steps'][i]
    r = CP.run_study_kind(p, s['kind'], s['settings'], str(out), eng, CP.ticket(rec, i), hooks=hooks, cancel=cancel)
    run['steps'][i].update(status='done' if r.ok else 'failed', message=r.message, summary=r.summary)
    CP.finish_run(rec, 'done' if r.ok else 'failed')
    return r


def _tc(p, work, eng=None, **kw):
    return T.ToolContext(p, None, eng or Engine(), working_dir=work, provider='groq', model='m', **kw)


def _plan_of(json_path):
    return json.load(open(json_path))['provenance'].get('plan')


REQUESTS = {   # the brief's requests -> the kind each must map to
    'rig_perception': 'compare pedestrian detection of rig A and B on the left',
    'foundation_range': 'how far can F see a cyclist',
    'robust_perception': 'what if the mounts are 2 mm off',
    'redrive': 'can we reuse last month\'s drive with the new rig',
    'design': 'design a rig for this brief: pedestrians at 80 m forward, 360° up to 30 m, under 300 W, budget under $6k',
    'perf_model_train': 'train the detection model',
    'perf_model_campaign': 'which rigs should we simulate next',
}


# ============================================================================================ the kinds
def test_the_study_kinds_are_plan_kinds_with_their_pages_slots_and_schemas():
    assert set(CP.STUDY_KINDS) <= set(CP.KINDS)
    for k in CP.STUDY_KINDS:
        assert CP.KIND_LABEL[k] and CP.KIND_PAGE[k] and CP.STUDY_SLOT[k] and CP.KIND_SLOT[k] is None
        assert k in CP.STUDY_ABOUT and k in CP.STUDY_EXTRAS and k in CP.DISK_MB
        from vantage.ui.mainwindow import PAGE_CLASSES
        assert CP.KIND_PAGE[k] in PAGE_CLASSES, k
    # the tools' schemas offer them (propose, estimate, validate, get_plan_settings) and explain their limits
    for name in ('propose_plan', 'estimate_plan', 'validate_plan'):
        assert T.BY_NAME[name].params['steps']['items']['properties']['kind']['enum'] == list(CP.KINDS)
    assert T.BY_NAME['get_plan_settings'].params['kind']['enum'] == list(CP.KINDS)
    assert set(T.BY_NAME['explain_limit'].params['topic']['enum']) == set(CP.LIMITS_DOC)
    assert 'rig_perception' in T.BY_NAME['propose_plan'].description
    assert T.validate_args('propose_plan', {'title': 't', 'rationale': 'r', 'revise': False, 'steps': [
        {'kind': 'rig_perception', 'label': 'x', 'values': []}]}) == []
    # widening twice changes nothing
    CT.widen_schemas(T)
    assert T.BY_NAME['propose_plan'].params['steps']['items']['properties']['kind']['enum'] == list(CP.KINDS)


def test_defaults_clean_to_themselves_and_every_setting_has_a_source(work):
    p = _rig()
    for k in CP.STUDY_KINDS:
        d = CP.defaults(k, p)
        again, _ = CP.clean(k, d, p, work)
        assert CP.same(again, d), k
        st = CP.resolve_step(p, k, [], working_dir=work)
        assert set(CP.flatten(st.settings)) <= set(st.sources), k
        assert all(v['source'] == 'default' for v in st.sources.values()), k
        assert st.estimate.get('text') and 'needs' in st.estimate, (k, st.estimate)
        assert set(st.estimate['needs']) == {'carla', 'gpu', 'torch', 'models'}
        json.dumps(CP.for_model(CP.Plan('P1', steps=[st]), work))


def test_values_are_cleaned_clamped_by_the_page_and_named_things_resolved(work, models_ok):
    p = _ab(_rig())
    a, b, _ = _recs_ab(p, work)
    from vantage import workdir
    na, nb = workdir.display(a, work), workdir.display(b, work)
    assert not os.path.isabs(na)
    st = CP.resolve_step(p, 'rig_perception', [
        V('recordings', [{'path': na, 'rig': 'Rig A'}, nb]), V('score', 5),
        V('focus', ['Left', 'front left']), V('classes', ['pedestrian', 'unicorn'])], working_dir=work)
    assert st.ok, st.errors
    s = st.settings
    assert [r['path'] for r in s['recordings']] == [a, b] and s['recordings'][0]['rig'] == 'Rig A'
    assert s['score'] == 0.99 and st.sources['score']['source'] == 'limit' and '0.01–0.99' in st.sources['score'][
        'reason']
    assert s['focus'] == ['front-left', 'left'] and s['classes'] == ['pedestrian']
    assert st.sources['classes']['source'] == 'limit'                       # the page dropped what it cannot use
    assert st.sources['recordings']['source'] == 'request'                  # named as the catalogue does: no limit
    assert st.estimate['images'] > 0 and st.estimate['needs']['models'] == ['yolox_s']
    # errors: an unknown sector, variant or key
    bad = CP.resolve_step(p, 'rig_perception', [V('focus', ['upwards']), V('source', 'carla'),
                                                 V('scene_set.variants', ['Rig Z'])], working_dir=work)
    assert any('no sector' in e for e in bad.errors) and any("no variant 'Rig Z'" in e for e in bad.errors)
    assert any('needs CARLA' in e for e in bad.errors)
    assert CP.resolve_step(p, 'design', [V('colour', 'red')]).errors[0].startswith('Design agent has no setting')
    c = CP.resolve_step(p, 'perf_model_campaign', [V('rounds', 99), V('variants', ['Rig B', 'rig a'])],
                        working_dir=work)
    assert c.settings['rounds'] == 5 and c.sources['rounds']['source'] == 'limit'
    assert c.settings['variants'] == ['rig-b-uid', p.variants[[v.name for v in p.variants].index('Rig A')].uid]
    assert c.sources['variants']['source'] == 'request'
    # in words on the card, never JSON
    from vantage.ui import assistant as UA
    rows = {r[1]: r for r in UA.plan_rows(st, p)}
    assert rows['recordings'][3] == 'rig-a as Rig A · rig-b' and rows['focus'][3] == 'front-left, left'
    assert rows['source'][0] == 'Plan'
    rows = {r[1]: r for r in UA.plan_rows(c, p)}
    assert rows['variants'][3] == 'Rig B · Rig A'


def test_the_design_step_reads_the_brief_and_says_so(work):
    p = _rig()
    from test_rig_designer import BRIEF
    st = CP.resolve_step(p, 'design', [V('brief', BRIEF), V('requirements_from', 'brief', 'copilot', 'the parser'),
                                       V('evaluations', 22)], working_dir=work)
    assert st.ok, st.errors
    assert st.settings['requirements'] and all(r['accepted'] for r in st.settings['requirements'])
    assert st.sources['requirements']['source'] == 'request' and 'parser' in st.sources['requirements']['reason']
    assert 'accepted' not in st.settings and 'approvals' not in st.settings and 'documents' not in st.settings
    again, _ = CP.clean('design', st.settings, p, work)
    assert CP.same(again, st.settings)                      # idempotent: the record verifies when it runs
    from vantage.ui import assistant as UA
    rows = {r[1]: r for r in UA.plan_rows(st, p)}
    assert 'pedestrian' in rows['requirements'][3].lower() or '80' in rows['requirements'][3]
    empty = CP.resolve_step(p, 'design', [V('requirements_from', 'brief')], working_dir=work)
    assert any('brief is empty' in e for e in empty.errors)


# ============================================================================================ natural requests
def test_the_deterministic_planner_maps_each_request_to_its_kind_and_values(work, models_ok):
    p = _ab(_rig())
    a, b, _ = _recs_ab(p, work)
    for kind, text in REQUESTS.items():
        assert CP.match_kind(text) == kind, text
        assert CP.is_plan_request(text), text
    for t in ('Why is the ground blind?', 'Which camera matters most?', 'What does ML see?'):
        assert not CP.match_kind(t) and not CP.is_plan_request(t), t
    got = {}
    for kind, text in REQUESTS.items():
        plan = CP.offline_plan(p, text, work)
        assert plan is not None and plan.provider == 'vantage' and plan.request == [text]
        st = plan.steps[0]
        assert st.kind == kind
        got[kind] = st
        for k, src in st.sources.items():                   # every value says where it came from, and why
            assert src['source'] in CP.SOURCES and (src['source'] == 'default' or src['reason']), (kind, k)
    s = got['rig_perception']
    assert s.ok, s.errors
    assert s.settings['classes'] == ['pedestrian'] and s.settings['focus'] == ['left']
    assert s.settings['source'] == 'recordings' and [(r['path'], r['rig']) for r in s.settings['recordings']] == \
        [(a, 'Rig A'), (b, 'Rig B')]
    assert s.sources['classes']['source'] == 'request' and s.sources['recordings']['source'] == 'copilot'
    f = got['foundation_range'].settings
    assert (f['source'], f['sweep_cameras'], f['sweep_classes'], f['depth_model']) == ('sweep', ['F'], ['cyclist'], '')
    assert got['foundation_range'].sources['sweep_classes']['source'] == 'request'
    assert any('needs CARLA' in e for e in got['foundation_range'].errors)  # no CARLA here: refused, clearly
    r = got['robust_perception'].settings
    assert (r['mount_from'], r['mount_mm'], r['mount_deg']) == ('custom', 2.0, 0.0)
    assert not r['thermal'] and not r['vibration'] and not r['calibration'] and r['recordings'] == [a]   # Rig A's
    assert got['robust_perception'].sources['mount_mm'] == {'source': 'request', 'reason': '"2 mm"'}
    d = got['redrive'].settings
    assert d['drive'] == b and d['variant'] == 'rig-b-uid'
    g = got['design']
    assert g.ok and g.settings['requirements_from'] == 'brief' and g.settings['requirements']
    assert got['perf_model_train'].settings['save'] is True
    assert any('at least 8 perception records' in e for e in got['perf_model_train'].errors)
    c = got['perf_model_campaign']
    assert c.settings['k'] == 2 and c.settings['variants'] == []
    # without recordings of the rigs named, rig perception drives a scene set (and says it needs CARLA)
    q = _ab(_rig())
    s2 = CP.offline_plan(q, REQUESTS['rig_perception'], work).steps[0]
    assert s2.settings['source'] == 'carla' and s2.settings['scene_set']['variants'] == [q.variants[
        [v.name for v in q.variants].index('Rig A')].uid, 'rig-b-uid'] or s2.settings['source'] == 'recordings'
    # nothing ran: no record, no output
    assert CP.records(p) == []


@pytest.fixture()
def groq(monkeypatch):
    s = FakeGroq().start()
    monkeypatch.setenv('VANTAGE_ASSISTANT_URL', s.url)
    yield s
    s.stop()


def test_a_model_plans_each_request_through_the_tools_and_nothing_runs(groq, work, models_ok):
    p = _ab(_rig())
    _recs_ab(p, work)
    e = Engine()
    for kind, text in REQUESTS.items():
        tc = T.ToolContext(p.snapshot(), None, e.share(), working_dir=work, provider='groq', model='m', request=text)
        cat = json.loads(T.run(tc, 'get_catalogue', {'topic': 'plan_kinds'})[0])
        row = next(k for k in cat['kinds'] if k['kind'] == kind)
        assert row['what'] and row['for_example'] and row['settings_saved_on_the_page']
        draft = cat['draft_for_this_request']['steps']
        assert draft[0]['kind'] == kind
        assert json.loads(T.run(tc, 'get_plan_settings', {'kind': kind})[0])['settings']
        s = A.Session('test-key', A.Settings('openai/gpt-oss-120b', planning=True))
        groq.script += [groq.tool_call('get_catalogue', {'topic': 'plan_kinds'}, 'c1'),
                        groq.tool_call('propose_plan', {'title': CP.KIND_LABEL[kind], 'rationale': 'as asked',
                                                        'revise': False, 'steps': draft}, 'c2'),
                        groq.answer('The plan is on the card: approve it to run.')]
        ev = []
        s.run_turn(text, tc, ev.append, threading.Event())
        assert ev[-1].data['ok'], [x.data for x in ev if x.kind == 'error']
        plans = [x.data['plan'] for x in ev if x.kind == 'copilot']
        assert len(plans) == 1 and plans[0].steps[0].kind == kind and plans[0].request == [text]
        assert plans[0].provider == 'groq'
    # the model was offered the study kinds, and never shown a path of this machine
    body = json.dumps(groq.requests[0])
    assert '"rig_perception"' in body and '"perf_model_campaign"' in body
    assert work not in json.dumps(groq.requests)
    assert CP.records(p) == []


# ============================================================================================ refusals
def test_refusals_say_what_is_missing(work, monkeypatch):
    from vantage.core import foundation_models as FM
    from vantage.core import splat_reproject as SP
    p = _rig()
    # the 2D detector, CARLA, PyTorch, the models, the records, the requirements, the fleet, the watch folder
    st = CP.resolve_step(p, 'rig_perception', [V('source', 'carla')], working_dir=work)
    assert any('2D detector model is not downloaded' in e for e in st.errors)
    assert any('needs CARLA' in e and 'client Python' in e for e in st.errors)
    from vantage.core import simcampaign as SC
    monkeypatch.setattr(SC, 'carla_python', lambda: sys.executable)
    st = CP.resolve_step(p, 'rig_perception', [V('source', 'carla')], working_dir=work)
    assert any('no server answers on port' in e and 'no CARLA folder' in e for e in st.errors)
    monkeypatch.setattr(FM, 'torch_missing', lambda: 'PyTorch is not installed.')
    monkeypatch.setattr(SP, 'torch_missing', lambda: 'PyTorch is not installed.')
    st = CP.resolve_step(p, 'foundation_range', [V('detectors', ['owlv2_b16_ens']), V('recordings', ['nothere'])],
                         working_dir=work)
    assert any('PyTorch is not installed' in e for e in st.errors)
    assert any('not downloaded' in e for e in st.errors) and any('nothere' in e for e in st.errors)
    assert st.estimate['needs']['torch'] and st.estimate['needs']['gpu'] == 'recommended'
    st = CP.resolve_step(p, 'redrive', [V('drive', 'nothere')], working_dir=work)
    assert any('PyTorch' in e for e in st.errors) and any('not found' in e for e in st.errors)
    assert any('at least 8' in e for e in CP.resolve_step(p, 'perf_model_train', [], working_dir=work).errors)
    st = CP.resolve_step(p, 'perf_model_campaign', [], working_dir=work)
    assert any('needs CARLA' in e or 'no server answers' in e for e in st.errors)
    assert st.estimate['needs']['carla'] and st.estimate['carla_s'] > 0
    assert any('No accepted requirement' in e for e in CP.resolve_step(p, 'design', [], working_dir=work).errors)
    assert any('no fleet' in e for e in CP.resolve_step(p, 'fleet_twin', [], working_dir=work).errors)
    from test_fleet_twin import fleet_of
    q = fleet_of(_rig(), n=2, records=3)
    assert CP.resolve_step(q, 'fleet_twin', [], working_dir=work).ok
    assert any('No watch folder' in e for e in CP.resolve_step(q, 'fleet_twin', [V('action', 'watch_once')],
                                                                working_dir=work).errors)
    # an invalid plan cannot be approved
    plan = CP.build(p, [{'kind': 'redrive', 'label': 'r', 'values': [V('drive', 'nothere')]}], 'R', '',
                    working_dir=work)
    rec, errs = CP.approve(p, plan, working_dir=work)
    assert rec is None and errs


def test_nothing_runs_without_an_approved_intact_record(work, models_ok, tmp_path, monkeypatch):
    from vantage.core import risk_coverage as RC
    p = _ab(_rig())
    a, b, la = _recs_ab(p, work)
    calls = []
    monkeypatch.setattr(RC, 'run', lambda *a_, **k: calls.append(1))
    steps = [{'kind': 'risk_coverage', 'label': 'risk', 'values': [V('weight', 'ncap')]}]
    plan = CP.build(p, steps, 'R', '', working_dir=work)
    st = plan.steps[0]
    out = tmp_path / 'out'
    for tk in (None, {'plan': plan.id, 'version': 1, 'step': 0}, {'plan': 'PFORGED', 'version': 1, 'step': 0}):
        r = CP.run_study_kind(p, 'risk_coverage', st.settings, str(out), None, tk)
        assert r.rc == CP.EXIT_REFUSED and not r.ok and r.message
    rec = _approved(p, steps, work)
    tk = CP.ticket(rec, 0)
    r = CP.run_study_kind(p, 'risk_coverage', dict(st.settings, weight='fatal'), str(out), None, tk)
    assert r.rc == CP.EXIT_REFUSED and 'differ' in r.message
    r = CP.run_study_kind(p, 'rig_perception', st.settings, str(out), None, tk)
    assert r.rc == CP.EXIT_REFUSED and 'not a' in r.message
    rec['steps'][0]['settings']['weight'] = 'cost'                          # changed after approval
    r = CP.run_study_kind(p, 'risk_coverage', rec['steps'][0]['settings'], str(out), None, tk)
    assert r.rc == CP.EXIT_REFUSED and 'does not verify' in r.message
    assert calls == [] and not os.path.exists(out)
    assert (p.analysis.risk_coverage or {}).get('weight') != 'cost'        # nothing applied either


# ============================================================================================ each kind runs
def test_rig_perception_runs_on_recordings_and_reaches_the_read_tool(work, models_ok, tmp_path):
    from test_rig_perception import FakeDetector
    from vantage.core import perf_model as PM
    p = _ab(_rig())
    a, b, la = _recs_ab(p, work)
    rec = _approved(p, [{'kind': 'rig_perception', 'label': 'A vs B', 'values': [
        V('recordings', [{'path': a, 'rig': 'Rig A'}, {'path': b, 'rig': 'Rig B'}]), V('classes', ['car', 'pedestrian']),
        V('focus', ['front', 'left']), V('records', True), V('bootstrap', 100), V('min_gt', 5), V('min_objects', 1),
        V('cache', 'off')]}], work)
    det = FakeDetector(la)
    r = _run(p, rec, tmp_path / 'out', hooks={'detector': det})
    assert r.rc == 0 and r.ok, r.message
    s = r.summary
    assert s['rigs'] == ['Rig A', 'Rig B'] and det.calls > 0 and s['findings']
    assert s['focus']['sectors'] == ['front', 'left'] and s['focus']['regions']['Rig A']['car']
    assert all(k.startswith(('front ', 'left ')) for k in s['focus']['regions']['Rig A']['car'])
    assert s['records_added'] > 0 and len(PM.store_for(p).records()) == s['records_added']
    doc = os.path.join(tmp_path / 'out', 'rig_perception.json')
    assert _plan_of(doc) == CP.ticket(rec, 0)
    assert work not in json.dumps(s) and p.analysis.rig_perception['score'] == 0.3      # applied to the page's slot
    sz = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(tmp_path / 'out') for f in fs)
    assert sz / 1e6 <= CP.DISK_MB['rig_perception']                       # the estimate's "at most" holds
    # the read tool cites the plan's result (no page in the headless run)
    out = R14.get_rig_perception(_tc(p, work), cls='car', sector='left')
    assert out['run'] and out['from_plan']['plan'] == rec['id'] and set(out['per_rig']['Rig A']) == {'car'}
    assert out['findings'] and all('at left ·' in f for f in out['findings'])
    assert all(k.startswith('left ') for k in out['focus']['regions']['Rig A']['car'])


def test_rig_perception_drives_a_scene_set_in_the_fake_carla(work, models_ok, carla_ok, tmp_path):
    from test_rig_perception import Oracle
    from test_simcampaign import BASE, Runner, _project
    import copy
    from vantage.core import project as PJ
    p = _project(cams=('F', 'ML'))
    p.analysis.sim = dict(BASE)
    w = copy.deepcopy(p.variant)
    w.uid, w.name = PJ.new_uid(), 'Front only'
    w.cameras = [c for c in w.cameras if c.tag == 'F']
    p.variants.append(w)
    rec = _approved(p, [{'kind': 'rig_perception', 'label': 'scene set', 'values': [
        V('source', 'carla'), V('scene_set', {'set_id': 'quick', 'variants': [p.variant.name, 'Front only'],
                                              'seeds': [2], 'duration_s': 1.2, 'traffic': 6, 'walkers': 0}),
        V('bootstrap', 50)]}], work)
    st = rec['steps'][0]
    assert st['estimate']['needs']['carla'] and st['estimate']['carla_s'] > 0
    assert any('scene set' in w_.lower() for w_ in st['warnings'])
    runner = Runner()
    r = _run(p, rec, tmp_path / 'out', hooks={'runner': runner, 'backend': Oracle()})
    assert r.rc == 0, r.message
    assert len(runner.runs) == 6 and r.summary['rigs'] == [p.variant.name, 'Front only']
    assert r.summary['recordings'] == 6


def test_foundation_range_sweeps_in_the_fake_carla(work, models_ok, carla_ok, tmp_path):
    import glob
    from test_foundation import GTDetector, _project, _runner
    from vantage.core import foundation_range as FR
    p = _project()
    rec = _approved(p, [{'kind': 'foundation_range', 'label': 'F pedestrians', 'values': [
        V('source', 'sweep'), V('sweep_cameras', ['F']), V('sweep_classes', ['pedestrian', 'car']),
        V('ranges_m', [7, 16]), V('offsets_deg', [0, 25]), V('road_lateral_m', []), V('seeds', [1]),
        V('image_scale', 0.1), V('bootstrap', 30), V('depth_model', ''), V('views', 'both'), V('view_width', 256),
        V('detectors', ['yolox_s'])]}], work)
    st = FR.FoundationSettings.from_dict(CP.study_part('foundation_range', rec['steps'][0]['settings']))
    out = tmp_path / 'out'

    class Lazy:                     # knows the truth of the drives the sweep has just made
        det = None

        def __call__(self, img):
            if self.det is None:
                dirs = sorted({os.path.dirname(f) for f in glob.glob(str(out / 'sweep' / '**' / 'run.json'),
                                                                       recursive=True)})
                self.det = GTDetector(dirs, st)
            return self.det(img)
    calls = []
    r = _run(p, rec, out, hooks={'runner': _runner(calls), 'detectors': {'yolox_s': Lazy()}})
    assert r.rc == 0, r.message
    assert calls and r.summary and r.result.fits
    assert _plan_of(str(out / 'foundation_range.json')) == CP.ticket(rec, 0)
    got = R14.get_foundation_range(_tc(p, work), camera='F')
    assert got['run'] and got['from_plan']['plan'] == rec['id']


def test_robust_perception_runs_and_says_what_the_mounting_error_costs(work, models_ok, tmp_path):
    from test_robust_perception import colour_detector, make_recording
    p = _rig()
    path = make_recording(str(tmp_path / 'robust_rec'))
    rec = _approved(p, [{'kind': 'robust_perception', 'label': '2 mm', 'values': [
        V('recordings', [path]), V('mount_from', 'custom'), V('mount_mm', 2.0), V('mount_deg', 0.0),
        V('thermal', False), V('vibration', False), V('calibration', False), V('samples', 3), V('sobol_n', 8),
        V('oat_samples', 2), V('frame_step', 1), V('bev_frame_step', 1), V('boot', 20)]}], work)
    assert 'builds' in rec['steps'][0]['estimate']
    r = _run(p, rec, tmp_path / 'out', hooks={'detector': colour_detector})
    assert r.rc == 0, r.message
    assert r.summary['recordings'] and [k for k, x in r.result.sources.items() if x.on] == ['mounting']
    got = R14.get_robust_perception(_tc(p, work))
    assert got['run'] and got['from_plan']['step'] == 1


def test_redrive_reports_how_much_of_the_drive_survives(work, tmp_path):
    pytest.importorskip('torch')
    from vantage.core import dataset as DSET
    from vantage.core.project import Variant
    p = _rig()
    v = p.variant
    b = Variant.from_dict(v.to_dict())
    b.uid, b.name = 'proposed-b', 'Data rig · cameras 20° up'
    for c in b.cameras:
        c.pitch -= 20.0
    p.variants.append(b)
    eng = Engine()
    drive = str(tmp_path / 'drive')
    assert DSET.generate(p, eng, DSET.DatasetSettings.from_dict({'duration_s': 1.0, 'image_scale': 0.15}), drive)
    p.analysis.splat = {'device': 'cpu', 'max_gaussians': 6000, 'voxel_m': 0.1, 'optimise': False,
                        'train_scale': 0.25, 'holdout_every': 4, 'eval_instants': 1, 'render_scale': 0.12,
                        'colour_step': 3, 'val_frames': 6, 'val_sweeps': 1}
    rec = _approved(p, [{'kind': 'redrive', 'label': 'reuse', 'values': [
        V('drive', drive), V('variant', b.name), V('calib_views', 12), V('calib_pixels', 600),
        V('report_instants', 2), V('sources', ['holdout']), V('ground_truth', 'always'), V('conf_scale', 0.5)]}],
        work, eng)
    assert rec['steps'][0]['settings']['variant'] == 'proposed-b'
    r = _run(p, rec, tmp_path / 'out', eng=eng)
    assert r.ok, r.message
    assert r.summary['rig_verdict'] in ('reuse', 'partial', 're-collect') and r.rc == (
        3 if r.summary['rig_verdict'] == 're-collect' else 0)
    assert p.analysis.splat['drive'] == drive
    got = R14.get_redrive(_tc(p, work))
    assert got['run'] and got['from_plan']['plan'] == rec['id']


def test_design_runs_and_never_creates_a_variant(work, tmp_path):
    from test_rig_designer import BRIEF, _design_project
    p = _design_project()
    n = len(p.variants)
    rec = _approved(p, [{'kind': 'design', 'label': 'brief', 'values': [
        V('brief', BRIEF), V('requirements_from', 'brief', 'copilot', 'parse it'), V('options', 3),
        V('evaluations', 12), V('budget_s', 30), V('seed', 5), V('reference_seeds', False), V('vary_lidar', False),
        V('vary_radar', False)]}], work)
    eng = Engine()
    r = _run(p, rec, tmp_path / 'out', eng=eng)
    assert r.rc in (0, 5) and r.ok, r.message
    assert len(p.variants) == n and r.summary['options']                   # offered, never applied
    assert p.analysis.designer['requirements'] and 'nothing was created' in r.message


def test_the_learned_model_trains_and_a_campaign_proposes_evaluates_and_retrains(work, carla_ok, models_ok,
                                                                                  tmp_path):
    import numpy as np
    from test_perf_model import GRID1, synth_record, synth_set
    from vantage.core import perf_model as PM
    p = _rig()
    S = PM.store_for(p)
    docs, _ = synth_set(24, 3)
    S.add_many(docs)
    rec = _approved(p, [{'kind': 'perf_model_train', 'label': 'train', 'values': [
        V('restarts', 1), V('max_iter', 60), V('cv_max_folds', 3)]}], work)
    assert rec['steps'][0]['estimate']['records'] == 24
    t0 = time.perf_counter()
    r = _run(p, rec, tmp_path / 'train')
    secs = time.perf_counter() - t0
    print(f'perf_model_train: {secs:.1f} s for 24 records ({secs / 24:.2f} s a record)')
    assert r.rc == 0, r.message
    assert r.summary and r.summary.get('saved') and PM.load_model() is not None and PM.load_model().trained
    assert secs / 24 <= 4 * CP.COST_TRAIN_S_PER_RECORD + 1.0           # the estimate's constant is of this order
    got = R14.get_perf_model(_tc(p, work))
    assert got['trained'] and got['page']['from_plan']['plan'] == rec['id']
    # the campaign: k a round from the project's variants, evaluated by a fake evaluator (no CARLA), retrained
    seen = []
    rng = np.random.default_rng(1)

    def evaluator(cand, progress=None, cancel=None):
        seen.append(cand.name)
        return [synth_record(900 + len(seen), rng.uniform(2, 7, GRID1.n_regions), rng.uniform(0, 1, GRID1.n_regions),
                             rng, scene='S9')]
    names = [v.name for v in p.variants if any(c.visible for c in v.cameras)][:3]
    rec2 = _approved(p, [{'kind': 'perf_model_campaign', 'label': 'next', 'values': [
        V('variants', names), V('rounds', 2), V('k', 1), V('restarts', 1), V('max_iter', 40), V('cv', False)]}], work)
    r2 = _run(p, rec2, tmp_path / 'campaign', hooks={'evaluator': evaluator})
    assert r2.ok, r2.message
    assert len(seen) == 2 and set(seen) <= set(names) and len(set(seen)) == 2
    assert [len(x['picks']) for x in r2.summary['rounds']] == [1, 1] and r2.summary['curve'][-1].get('final')
    assert len(S.records()) == 26
    got = R14.get_perf_model(_tc(p, work))
    assert got['campaign']['from_plan']['plan'] == rec2['id']


def test_risk_coverage_and_the_fleet_twin_run(work, tmp_path):
    from test_fleet_twin import fleet_of
    p = fleet_of(_rig(), n=3, records=4)
    rec = _approved(p, [{'kind': 'risk_coverage', 'label': 'risk', 'values': [V('weight', 'ncap')]},
                        {'kind': 'fleet_twin', 'label': 'twin', 'values': [V('horizon_days', 200)]}], work)
    r = _run(p, rec, tmp_path / 'risk')
    assert r.rc == 0 and r.summary['classes'] and 'requirement_proposals' in r.summary
    assert p.analysis.risk_coverage['weight'] == 'ncap'
    r2 = _run(p, rec, tmp_path / 'twin', i=1)
    assert r2.rc in (0, 3) and r2.summary['vehicles'] == 3
    from vantage.core import fleet_twin as FT
    assert FT.settings_of(p).horizon_days == 200
    assert _plan_of(next(os.path.join(tmp_path / 'twin', f) for f in os.listdir(tmp_path / 'twin')
                         if f.endswith('.json'))) == CP.ticket(rec, 1)


def test_cancel_stops_a_study_step(work, models_ok, tmp_path):
    from test_rig_perception import FakeDetector
    p = _ab(_rig())
    a, b, la = _recs_ab(p, work)
    rec = _approved(p, [{'kind': 'rig_perception', 'label': 'A', 'values': [
        V('recordings', [{'path': a, 'rig': 'Rig A'}]), V('cache', 'off')]}], work)
    n = []
    r = _run(p, rec, tmp_path / 'out', hooks={'detector': FakeDetector(la)}, cancel=lambda: n.append(1) or len(n) > 3)
    assert r.rc == 130 and not r.ok and not os.path.exists(tmp_path / 'out' / 'rig_perception.json')


# ============================================================================================ --run-plan
def _batch(argv):
    from vantage import batch
    return batch.run(batch.parser().parse_args(argv))


def _saved_plan(tmp_path, work):
    from test_fleet_twin import fleet_of
    p = fleet_of(_rig(), n=3, records=4)
    rec = _approved(p, [{'kind': 'risk_coverage', 'label': 'risk', 'values': [V('weight', 'fatal')]},
                        {'kind': 'fleet_twin', 'label': 'twin', 'values': []}], work)
    path = str(tmp_path / 'rig.vantage')
    p.save(path)
    return path, rec


def test_a_saved_plan_runs_headless_from_its_record(work, tmp_path):
    """What batch --run-plan does for a study step (the lead wires it: reports15/copilot15.md): the record loaded
    from the file, verified, each step gated and run from its stored settings, no model asked."""
    from vantage.core.project import Project
    path, rec0 = _saved_plan(tmp_path, work)
    p = Project.load(path)
    rec = CP.find(p, rec0['id'])
    assert CP.verify(rec)
    run = CP.start_run(rec, 'batch')
    for i, s in enumerate(rec['steps']):
        r = CP.run_study_kind(p, s['kind'], s['settings'], str(tmp_path / f'step-{i + 1}-{s["kind"]}'), None,
                              CP.ticket(rec, i))
        assert r.ok, r.message
        run['steps'][i].update(status='done', summary=r.summary)
    CP.finish_run(rec, 'done')
    out = CP.write_run_file(p, rec, str(tmp_path))
    doc = json.load(open(out))
    assert [s['status'] for s in doc['runs'][-1]['steps']] == ['done', 'done']
    assert _plan_of(str(tmp_path / 'step-1-risk_coverage' / 'risk_coverage.json')) == CP.ticket(rec, 0)
    # a changed record is refused
    q = Project.load(path)
    q.analysis.copilot['plans'][0]['steps'][0]['settings']['weight'] = 'cost'
    rr = CP.find(q, rec0['id'])
    assert CP.run_study_kind(q, 'risk_coverage', rr['steps'][0]['settings'], str(tmp_path / 'x'), None,
                             CP.ticket(rr, 0)).rc == CP.EXIT_REFUSED


def test_batch_run_plan_runs_study_steps(work, tmp_path, capsys):
    from vantage import batch
    if 'run_study_kind' not in open(batch.__file__, encoding='utf-8').read():
        pytest.skip('batch --run-plan does not dispatch the study kinds yet (the proposed batch diff: '
                    'reports15/copilot15.md)')
    path, rec = _saved_plan(tmp_path, work)
    out = tmp_path / 'planrun'
    code = _batch([path, '--run-plan', rec['id'], str(out)])
    assert code in (0, 3), capsys.readouterr().err
    assert _plan_of(str(out / 'step-1-risk_coverage' / 'risk_coverage.json'))['plan'] == rec['id']
    run = json.loads((out / 'copilot_plan.json').read_text())
    assert [s['status'] for s in run['runs'][-1]['steps']] == ['done', 'done'] and run['runs'][-1]['via'] == 'batch'


# ============================================================================================ the app
from test_gui import app, win, pump, settle, _wait  # noqa: E402,F401  (fixtures)


def _cards(panel):
    from vantage.ui.assistant import CopilotCard
    return panel.findChildren(CopilotCard)


def test_offline_plan_card_approve_run_in_background_page_shows_it_and_read_tool_cites_it(app, win, work, models_ok,
                                                                                          tmp_path):
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    from test_rig_perception import FakeDetector
    from vantage import workdir
    old = win.ctx.settings.value(workdir.KEY, '')
    workdir.put(win.ctx.settings, work)
    st = win.ctx.settings
    st.setValue('assistant/enabled', False)                 # no model set up: Vantage's own planner plans it
    ctl = win.ctx.assistant
    ctl.reload()
    try:
        p = win.store.project
        win.store.edit(['variants'], lambda pj: _ab(pj), 'two rigs')
        a, b, la = _recs_ab(win.store.project, work)
        win.show_assistant()
        panel = win.assistant_panel
        assert not ctl.ready and panel.composer.edit.isEnabled()
        QTest.keyClicks(panel.composer.edit, '/plan ' + REQUESTS['rig_perception'])
        assert panel.composer.go.isEnabled()
        QTest.keyClick(panel.composer.edit, Qt.Key.Key_Return)
        pump(app, 0.3)
        card = _cards(panel)[-1]
        plan = card.plan
        assert plan.provider == 'vantage' and plan.steps[0].kind == 'rig_perception', plan.errors
        assert plan.ok, plan.errors
        assert card.b_approve.isVisible() and card.b_approve.isEnabled()
        pg = win.page('rig_perception')
        assert pg.result is None and CP.records(win.store.project) == []       # nothing ran before approval
        det = FakeDetector(la)
        ctl.hooks = {'detector': det}
        QTest.mouseClick(card.b_approve, Qt.MouseButton.LeftButton)
        pump(app, 0.2)
        assert ctl.runner is not None and ctl.runner.running                    # in the background: the UI is free
        _wait(app, lambda: not ctl.runner.running, 300)
        rec = CP.find(win.store.project, plan.id, plan.version)
        assert rec['status'] == 'done', rec['runs'][-1]
        assert rec['provider'] == 'vantage' and rec['runs'][-1]['steps'][0]['summary']['rigs'] == ['Rig A', 'Rig B']
        assert pg.result is not None and [r['name'] for r in pg.result.rigs] == ['Rig A', 'Rig B'] and det.calls
        assert not pg.stale and 'instants' in pg.status_text
        assert 'rig_perception' in ctl._studies()
        got = R14.get_rig_perception(ctl._context(), cls='pedestrian')
        assert got['run'] and 'from_plan' not in got and got['rigs'] == ['Rig A', 'Rig B']    # the page's own
        folder = CP.run_folder(win.store.project, rec, work)
        assert _plan_of(os.path.join(folder, 'step-1-rig_perception', 'rig_perception.json')) == CP.ticket(rec, 0)
        chips = __import__('vantage.ui.assistant', fromlist=['plan_chips']).plan_chips(card.plan, rec)
        assert any('Which rig detects better' in c.title for c in chips)
    finally:
        ctl.hooks = {}
        workdir.put(win.ctx.settings, str(old or ''))


def test_cancel_a_running_study_step_in_the_app(app, win, work, models_ok, tmp_path):
    from test_rig_perception import FakeDetector
    from vantage import workdir
    old = win.ctx.settings.value(workdir.KEY, '')
    workdir.put(win.ctx.settings, work)
    ctl = win.ctx.assistant
    try:
        win.store.edit(['variants'], lambda pj: _ab(pj), 'two rigs')
        a, b, la = _recs_ab(win.store.project, work)
        gate = threading.Event()

        class Slow(FakeDetector):
            def __call__(self, img):
                gate.wait(5)
                return super().__call__(img)
        plan = CP.build(win.store.project, [{'kind': 'rig_perception', 'label': 'A', 'values': [
            V('recordings', [{'path': a, 'rig': 'Rig A'}]), V('cache', 'off')]}], 'A', '', working_dir=work)
        assert plan.ok, plan.errors
        ctl._add_plan(plan)
        ctl.hooks = {'detector': Slow(la)}
        assert ctl.approve_plan(plan) == ''
        pump(app, 0.3)
        assert ctl.runner.running
        ctl.stop_plan()
        gate.set()
        pump(app, 0.5)
        rec = CP.find(win.store.project, plan.id, plan.version)
        assert rec['status'] == 'cancelled' and rec['runs'][-1]['steps'][0]['status'] == 'cancelled'
        assert win.page('rig_perception').result is None
    finally:
        ctl.hooks = {}
        workdir.put(win.ctx.settings, str(old or ''))
