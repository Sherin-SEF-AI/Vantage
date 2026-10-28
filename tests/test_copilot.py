"""The Copilot: plans for runs (every kind, every value with its source and reason, the app's own validation and
estimates), the approval gate (records, verify, the pages' Run refusing a plan without its record), immutability of
an approved plan, re-runs from the record without the model, the batch --run-plan, the plan ID in the outputs'
provenance, what is never sent to a provider (every payload the fake servers received is scanned), and the panel's
plan card driven like a user. Never a real API (tests/fake_groq.py, tests/fake_anthropic.py), never a CARLA server
(tests/fake_carla)."""
import copy
import json
import os
import sys
import threading
import time

import pytest

from fake_groq import FakeGroq
from vantage.assistant import agent as A
from vantage.assistant import copilot as CP
from vantage.assistant import copilot_tools as CT
from vantage.assistant import providers as PR
from vantage.assistant import secrets as K
from vantage.assistant import tools as T
from vantage.core import samples
from vantage.core.analysis import Engine

HERE = os.path.dirname(os.path.abspath(__file__))
FAKE_CARLA = os.path.join(HERE, 'fake_carla')


def _rig():
    p = samples.tigor_sample_project()
    v = next(v for v in p.variants if v.sensors)
    p.active_uid = v.uid
    return p


def V(key, value, source='request', reason='you asked for it'):
    return {'key': key, 'value': json.dumps(value), 'source': source, 'reason': reason}


@pytest.fixture(autouse=True)
def _quiet_machine(monkeypatch):
    """No CARLA server answers anywhere and the machine has memory to spare, unless a test says otherwise."""
    from vantage.sim import server as SV
    monkeypatch.setattr(SV, 'port_open', lambda *a, **k: False)
    monkeypatch.setattr(CP, 'mem_available_gb', lambda: 30.0)


@pytest.fixture()
def work(tmp_path, monkeypatch):
    """A working folder of the test's own (never the user's ~/Documents/Vantage)."""
    from vantage import workdir
    d = tmp_path / 'work'
    d.mkdir()
    monkeypatch.setattr(workdir, 'default', lambda: str(d))
    return str(d)


def _drive_recordings(project, work, n=3):
    """n CARLA drives with ground truth in the project's runs folder (the detection tests' contract recordings)."""
    from test_detection import make_recording, scene_a
    from vantage import workdir
    root = workdir.project_dir(project, work, 'carla')
    out = []
    for i in range(n):
        d = os.path.join(root, f'drive-{i}')
        make_recording(d, scene_a(i % 2 == 1), n_frames=6, variant=f'Rig {i}')
        t = time.time() - 100 + i
        os.utime(os.path.join(d, 'run.json'), (t, t))
        out.append(d)
    return out


def _with_targets(p, *labels):
    from vantage.core import critical as CR
    st = CR.settings_of(p)
    for lab in labels:
        st.targets.append(CR.preset(p, lab, st.targets))
    p.analysis.critical = st.to_dict()
    return p


# ============================================================================================ plans of every kind
def test_a_carla_drive_plan_states_every_value_its_source_and_the_documented_limits(work):
    p = _rig()
    st = CP.resolve_step(p, 'carla_run', [
        V('town', 'Town10HD', reason='you said Town10'), V('weather', 'SoftRainSunset', 'copilot', 'dusk in rain'),
        V('duration_s', 10), V('traffic', 30), V('walkers', 10), V('scenario.name', 'pedestrian_crossing'),
        V('image_scale', 0.5), V('port', 2400, 'copilot', 'a port of the Copilot\'s own range'),
        V('tm_port', 8500, 'copilot', 'its traffic manager')], 'Dusk rain drive', working_dir=work)
    assert st.ok, st.errors
    s = st.settings
    assert (s['town'], s['weather'], s['duration_s'], s['traffic'], s['walkers']) == \
        ('Town10HD', 'SoftRainSunset', 10.0, 30, 10)
    assert s['scenario']['name'] == 'pedestrian_crossing' and all(s['ground_truth'].values())
    # every value of the page's settings has a source; the asked ones say so with their reasons
    from vantage.core import simrig as SR
    assert set(CP.flatten(SR.SimSettings().to_dict())) <= set(st.sources) | {'scenario.params'}
    assert st.sources['town'] == {'source': 'request', 'reason': 'you said Town10'}
    assert st.sources['weather']['source'] == 'copilot' and st.sources['seed']['source'] == 'default'
    # documented limits as warnings; the estimate from the app's own estimators
    assert any('lidar and radar ignore rain' in w for w in st.warnings)
    assert any('own car' in w for w in st.warnings)
    e = st.estimate
    assert e['frames'] > 0 and e['labelled_frames'] == e['frames'] and e['carla_s'] > 10 and e['disk_mb'] > 0
    assert 0 < e['render_mpx'] <= 200 and 'development machine' in e['basis']
    assert any('starts its own' in n for n in st.notes)


def test_values_out_of_range_are_forced_by_a_limit_and_unknown_keys_are_errors(work):
    p = _rig()
    st = CP.resolve_step(p, 'carla_run', [V('traffic', 900), V('image_scale', 0.2), V('duration_s', 2),
                                          V('port', 2400), V('tm_port', 8500)], working_dir=work)
    assert st.ok and st.settings['traffic'] == 300
    assert st.sources['traffic']['source'] == 'limit' and '0–300' in st.sources['traffic']['reason']
    bad = CP.resolve_step(p, 'carla_run', [V('warp_speed', 9)], working_dir=work)
    assert not bad.ok and any('no setting' in e for e in bad.errors)
    bad = CP.resolve_step(p, 'carla_run', [V('scenario.name', 'cut_in'), V('scenario.params.gap_banana', 3)],
                          working_dir=work)
    assert not bad.ok
    nope = CP.resolve_step(p, 'teleport', [])
    assert not nope.ok and 'Unknown step kind' in nope.errors[0]


def test_the_render_budget_lowers_a_chosen_image_scale_and_refuses_a_requested_one(work):
    p = _rig()
    chosen = CP.resolve_step(p, 'carla_run', [V('image_scale', 1.0, 'copilot', 'full size'), V('port', 2400),
                                              V('tm_port', 8500)], working_dir=work)
    assert chosen.ok and chosen.settings['image_scale'] < 1.0
    src = chosen.sources['image_scale']
    assert src['source'] == 'limit' and 'render budget' in src['reason']
    assert any('exceed the render budget' in w for w in chosen.warnings) and chosen.estimate['render_mpx'] <= 200
    asked = CP.resolve_step(p, 'carla_run', [V('image_scale', 1.0), V('port', 2400), V('tm_port', 8500)],
                            working_dir=work)
    assert not asked.ok and any('render budget' in e and 'fits' in e for e in asked.errors)


def test_ports_memory_and_a_server_that_answers(work, monkeypatch):
    p = _rig()
    st = CP.resolve_step(p, 'carla_run', [V('port', 2005), V('image_scale', 0.2)], working_dir=work)
    assert not st.ok and any('2000–2010' in e for e in st.errors)
    p.analysis.sim = {'port': 2005, 'tm_port': 8600}            # the project's own setting: allowed
    assert CP.resolve_step(p, 'carla_run', [V('image_scale', 0.2)], working_dir=work).ok
    monkeypatch.setattr(CP, 'mem_available_gb', lambda: 0.5)
    low = CP.resolve_step(p, 'carla_run', [V('image_scale', 0.2)], working_dir=work)
    assert not low.ok and any('MemAvailable' in e for e in low.errors)
    monkeypatch.setattr(CP, 'mem_available_gb', lambda: 5.0)
    warn = CP.resolve_step(p, 'carla_run', [V('image_scale', 0.2)], working_dir=work)
    assert warn.ok and any('MemAvailable' in w and 'cap' in w for w in warn.warnings)
    from vantage.sim import server as SV
    monkeypatch.setattr(SV, 'port_open', lambda *a, **k: True)
    up = CP.resolve_step(p, 'carla_run', [V('image_scale', 0.2)], working_dir=work)
    assert up.ok and any('answers on port 2005' in n for n in up.notes)
    assert not any('MemAvailable' in w for w in up.warnings)        # no server to start: the bridge alone fits


def test_a_campaign_plan_from_a_template_and_its_limits(work):
    p = _rig()
    st = CP.resolve_step(p, 'campaign', [V('template', 'fault_detectability', 'copilot', 'faults with known answers'),
                                         V('fault_sets', [{'kind': 'rotation', 'tags': ['F', 'ML'],
                                                           'axes': ['yaw', 'pitch', 'roll'],
                                                           'magnitudes': [0.1, 0.2, 0.5, 1.0]}]),
                                         V('seeds', [1, 2, 3]), V('clean', True),
                                         V('evaluators', ['drivelog', 'asbuilt'], 'copilot', 'the checks')],
                         working_dir=work)
    assert st.ok, st.errors
    assert st.settings['fault_sets'][0]['magnitudes'] == [0.1, 0.2, 0.5, 1.0] and st.settings['seeds'] == [1, 2, 3]
    assert st.sources['name']['reason'].startswith('from the Fault detectability')
    assert any('cannot judge mounts on CARLA' in w for w in st.warnings)
    assert st.estimate['drives'] > 3 and 'drives' in st.estimate['text']
    bad = CP.resolve_step(p, 'campaign', [V('fault_sets', [{'kind': 'rotation', 'tags': ['ZZ'],
                                                            'magnitudes': [0.5]}])], working_dir=work)
    assert not bad.ok


def test_detection_needs_recordings_with_ground_truth_and_the_model(work, monkeypatch):
    from vantage.core import detection as DT
    p = _rig()
    recs = _drive_recordings(p, work)
    cat = CT.get_catalogue(T.ToolContext(p, None, None, working_dir=work), topic='recordings')
    names = [r['name'] for r in cat['recordings']]
    assert names[0].endswith('drive-2') and not any(n.startswith('/') for n in names)
    st = CP.resolve_step(p, 'detection', [V('recordings', names[:2], 'copilot', 'the two latest drives')],
                         working_dir=work)
    assert not st.ok and any('detector model' in e for e in st.errors)        # not downloaded in the tests
    real = DT.status
    monkeypatch.setattr(DT, 'status', lambda m: dict(real(m), present=True, verified=True))
    st = CP.resolve_step(p, 'detection', [V('recordings', names[:2], 'copilot', 'the two latest drives')],
                         working_dir=work)
    assert st.ok, st.errors
    assert st.settings['recordings'] == [recs[2], recs[1]] and st.estimate['images'] > 0
    out = CP.for_model(CP.Plan('P1', steps=[st]), work)
    assert work not in json.dumps(out)                                        # the model sees no machine path
    gone = CP.resolve_step(p, 'detection', [V('recordings', ['no/such/drive'])], working_dir=work)
    assert not gone.ok


def test_critical_targets_live_experiment_dataset_study_and_export_plans(work):
    p = _with_targets(_rig(), 'Child 2 m ahead')
    cc = CP.resolve_step(p, 'critical_carla', [V('seed', 2)], working_dir=work)
    assert cc.ok, cc.errors
    assert cc.estimate['targets'] == 1 and cc.estimate['drives'] >= 1
    assert not CP.resolve_step(_rig(), 'critical_carla', [], working_dir=work).ok        # no targets placed
    live = CP.resolve_step(p, 'carla_live', [V('cameras', ['F', 'QR']), V('faults', [
        {'at_s': 10, 'fault': {'tag': 'QR', 'kind': 'freeze'}}])], working_dir=work)
    assert live.ok, live.errors
    assert live.settings['faults'] == [{'at_s': 10.0, 'fault': {'tag': 'QR', 'kind': 'freeze'}}]
    assert not CP.resolve_step(p, 'carla_live', [V('cameras', ['F']), V('faults', [
        {'at_s': 1, 'fault': {'tag': 'QR', 'kind': 'freeze'}}])], working_dir=work).ok   # QR not streamed
    f = p.variant.by_tag('F')
    ex = CP.resolve_step(p, 'experiment', [V('factors', [
        {'kind': 'camera', 'target': 'F', 'attr': 'pitch', 'mode': 'value', 'lo': 25, 'hi': 45, 'levels': 5},
        {'kind': 'camera', 'target': 'F', 'attr': 'lens', 'choices': [p.lens_for(f).id, next(
            k for k in p.lenses if k != p.lens_for(f).id)]}]),
        V('responses', [{'metric': 'coverage.blind_m2', 'params': {}}]), V('design', 'factorial')], working_dir=work)
    assert ex.ok, ex.errors
    assert ex.settings['factors'][0]['target'] == f.uid and ex.estimate['points'] == 10
    ds = CP.resolve_step(p, 'sample_dataset', [V('duration_s', 20), V('speed_kmh', 40)], working_dir=work)
    assert ds.ok and ds.estimate['disk_mb'] > 0 and any('synthetic' in w for w in ds.warnings)
    sy = CP.resolve_step(p, 'study', [V('study', 'blind_volume'), V('height', 1.0)], working_dir=work)
    assert sy.ok and sy.settings['study'] == 'blind_volume'
    exp = CP.resolve_step(p, 'export', [V('formats', ['extrinsics_json', 'viewport_png'])], working_dir=work)
    assert not exp.ok and any('rendered by the app window' in e for e in exp.errors)
    exp = CP.resolve_step(p, 'export', [V('formats', ['extrinsics_json']), V('folder', '../../etc')], working_dir=work)
    assert not exp.ok
    assert CP.resolve_step(p, 'export', [V('formats', ['extrinsics_json'])], working_dir=work).ok


# ============================================================================================ versions and approval
def test_a_follow_up_revises_the_plan_with_a_diff_and_an_invalid_plan_cannot_be_approved(work):
    p = _rig()
    v1 = CP.build(p, [{'kind': 'sample_dataset', 'label': 'Sample', 'values': [V('duration_s', 20),
                                                                               V('speed_kmh', 40)]}],
                  'A sample dataset', 'as asked', request='a 20 s sample dataset at 40 km/h', working_dir=work)
    assert v1.ok and v1.version == 1 and v1.status == 'draft'
    v2 = CP.build(p, [{'kind': 'sample_dataset', 'label': '', 'values': [V('duration_s', 10, reason='halve it')]}],
                  prev=v1, revise=True, request='halve it', working_dir=work)
    assert (v2.id, v2.version, v2.parent) == (v1.id, 2, 1) and v2.title == 'A sample dataset'
    assert v2.diff == [{'step': 0, 'key': 'duration_s', 'before': 20.0, 'after': 10.0}]
    assert v2.steps[0].settings['speed_kmh'] == 40.0                  # what was not changed stays
    assert v2.request == ['a 20 s sample dataset at 40 km/h', 'halve it']
    bad = CP.build(p, [{'kind': 'export', 'label': 'x', 'values': []}], 'Export', '', working_dir=work)
    assert not bad.ok
    rec, errors = CP.approve(p, bad, working_dir=work)
    assert rec is None and errors


def test_approval_records_the_plan_and_the_gate_holds(work):
    p = _rig()
    plan = CP.build(p, [{'kind': 'sample_dataset', 'label': 'Sample', 'values': [V('duration_s', 2)]}], 'Sample', '',
                    request='a short sample', provider='claude', model='claude-opus-5-5', working_dir=work)
    rec, errors = CP.approve(p, plan, working_dir=work)
    assert not errors and CP.verify(rec)
    assert rec['provider'] == 'claude' and rec['model'] == 'claude-opus-5-5' and rec['request'] == ['a short sample']
    assert rec['settings_sha256'] == CP.plan_sha256(rec['steps']) and rec['approved_utc'].endswith('Z')
    CP.add_record(p, rec)
    tk = CP.ticket(rec, 0)
    from vantage.core import dataset as DSET
    p.analysis.dataset = copy.deepcopy(rec['steps'][0]['settings'])
    now = DSET.DatasetSettings.from_dict(p.analysis.dataset).to_dict()
    assert CP.gate(p, tk, 'sample_dataset', now) == (True, '')
    # every way round the gate is refused
    assert not CP.gate(p, None, 'sample_dataset', now)[0]
    assert not CP.gate(p, {'plan': 'PNOPE', 'version': 1, 'step': 0}, 'sample_dataset', now)[0]
    assert 'not a' in CP.gate(p, tk, 'carla_run', now)[1]
    assert 'differ' in CP.gate(p, tk, 'sample_dataset', dict(now, duration_s=3.0))[1]
    assert not CP.gate(p, dict(tk, step=5), 'sample_dataset', now)[0]
    stored = CP.find(p, rec['id'], rec['version'])
    stored['steps'][0]['settings']['duration_s'] = 30.0               # edited after approval
    assert not CP.verify(stored) and 'does not verify' in CP.gate(p, tk, 'sample_dataset', now)[1]
    # the record survives an undo of an edit made before it was written
    stored['steps'][0]['settings']['duration_s'] = 2.0
    before = p.section('analysis')
    before['copilot'] = {}
    p.restore('analysis', before)
    assert CP.find(p, rec['id'], rec['version']) is stored
    # an approved plan is immutable: an edit is a new version that needs approval
    plan.status = 'approved'
    v2 = CP.edit_value(p, plan, 0, 'duration_s', 4.0, work)
    assert (v2.version, v2.status, v2.parent) == (2, 'draft', 1) and plan.steps[0].settings['duration_s'] == 2.0
    assert v2.steps[0].sources['duration_s']['source'] == 'request'
    assert CP.find(p, plan.id, 2) is None and not CP.gate(p, CP.ticket({'id': plan.id, 'version': 2}, 0),
                                                         'sample_dataset', v2.steps[0].settings)[0]
    assert CP.approve(p, plan)[0] is None                              # never approved twice


def test_the_plan_id_is_in_the_outputs_provenance(work, tmp_path):
    from vantage.core import provenance as PV
    p = _rig()
    plan = CP.build(p, [{'kind': 'study', 'label': 'Blind volume', 'values': [V('study', 'blind_volume')]}], 'Study',
                    '', working_dir=work)
    rec, _ = CP.approve(p, plan, working_dir=work)
    CP.add_record(p, rec)
    path = CP.write_run_file(p, rec, str(tmp_path / 'run'), 0, {'blind_m3': 1.5, 'where': str(tmp_path)})
    doc = json.load(open(path))
    pv = doc['provenance']
    assert pv['plan'] == {'plan': rec['id'], 'version': rec['version'], 'step': 0} and PV.verify(doc)['ok']
    assert str(tmp_path) not in json.dumps(doc['summary'])
    plain = PV.block(p, {'x': 1})
    assert plain['plan'] is None
    # the content hash ignores the records (their runs change as they run)
    h = PV.project_digest(p)['sha256']
    p.analysis.copilot = {}
    assert PV.project_digest(p)['sha256'] == h


# ============================================================================================ the tools
def test_the_copilot_tools_read_and_propose_and_none_runs_anything(work):
    p = _rig()
    for gone in ('generate_sample_dataset', 'run_experiment'):
        assert gone not in T.BY_NAME
    writers = [t.name for t in T.TOOLS if t.kind in ('copilot',)]
    assert writers == ['propose_plan']
    seen = []
    tc = T.ToolContext(p, None, Engine(), working_dir=work, request='a 20 s dataset', provider='groq', model='m')
    tc.on_copilot = seen.append
    for topic in CT.TOPICS:
        out = json.loads(T.run(tc, 'get_catalogue', {'topic': topic})[0])
        assert 'error' not in out, topic
    for kind in CP.KINDS:
        out = json.loads(T.run(tc, 'get_plan_settings', {'kind': kind})[0])
        assert out['settings'], kind
    assert 'Mpx' in json.loads(T.run(tc, 'explain_limit', {'topic': 'render_budget'})[0])['limit']
    steps = [{'kind': 'sample_dataset', 'label': 'Sample', 'values': [V('duration_s', 20), V('speed_kmh', 40)]}]
    args = {'title': 'Sample', 'rationale': 'as asked', 'revise': False, 'steps': steps}
    assert T.validate_args('propose_plan', args) == []
    est = json.loads(T.run(tc, 'estimate_plan', args)[0])
    assert est['estimate'][0]['disk_mb'] > 0 and not seen
    val = json.loads(T.run(tc, 'validate_plan', args)[0])
    assert val['ok'] and not seen
    out, ok = T.run(tc, 'propose_plan', args)
    out = json.loads(out)
    assert ok and out['shown_to_the_user'] and len(seen) == 1 and seen[0].request == ['a 20 s dataset']
    assert out['steps'][0]['values_set']['duration_s']['source'] == 'request'
    tc.plan = seen[0]
    out = json.loads(T.run(tc, 'propose_plan', dict(args, revise=True, steps=[
        {'kind': 'sample_dataset', 'label': '', 'values': [V('duration_s', 10)]}]))[0])
    assert out['version'] == 2 and out['changed_from_previous_version'][0]['after'] == 10.0
    # inputs are checked against the schema before a tool runs (a partial or mistyped input never runs)
    assert T.validate_args('propose_plan', {'title': 'x', 'steps': 'no'})
    assert T.validate_args('get_catalogue', {'topic': 'nope'})
    assert T.validate_args('explain_point', {'x': 'a', 'y': 1})
    assert T.validate_args('explain_point', {'x': 1, 'y': 2, 'zz': 3})
    assert T.validate_args('explain_point', {'x': 1, 'y': 2.5}) == []


def test_plan_requests_are_told_apart_from_questions():
    for t in ('a 10 s drive in Town10 at dusk in rain', 'a fault campaign: turn F and ML', 'compare detection on '
              'the two latest drives', 'critical targets in CARLA, all targets, 2 seeds', 'start the live rig',
              'an experiment: F pitch 25-45', 'a 20 s sample dataset at 40 km/h', '/plan anything'):
        assert CP.is_plan_request(t), t
    for t in ('Why is the ground blind?', 'Which camera matters most?'):
        assert not CP.is_plan_request(t), t


# ============================================================================================ batch --run-plan
def _batch(argv):
    from vantage import batch
    return batch.run(batch.parser().parse_args(argv))


# ============================================================================================ values in words
WORDY = {   # kind -> values a model writes in its own way: lists and objects the card must say in words
    'carla_run': [V('town', 'Town10HD'), V('weather', 'fog'), V('scenario', {'name': 'cut_in', 'params': {'speed_kmh': 40}}),
                  V('ground_truth', {'semantic': True, 'range': False}), V('tick_hz', 80),
                  V('actors', [{'kind': 'child', 'pose': {'x': 12, 'y': -2}}]),
                  V('faults', [{'tag': 'F', 'kind': 'rotation', 'deg': [0, 1, 0]},
                               {'tag': 'L1', 'kind': 'range_bias', 'm': 0.05}]),
                  V('image_scale', 0.25), V('port', 2400), V('tm_port', 8500)],
    'carla_live': [V('cameras', 'F, QR'), V('scale', 0.3),
                   V('faults', [{'at_s': 4, 'tag': 'F', 'deg': [0, 1, 0]}, {'at_s': 2, 'tag': 'QR', 'kind': 'freeze'}])],
    'experiment': [V('factors', [{'id': 'F1', 'kind': 'camera', 'target': 'F', 'attr': 'pitch', 'mode': 'value',
                                  'lo': 25, 'hi': 45, 'levels': 5},
                                 {'id': 'F2', 'kind': 'camera', 'target': 'F', 'attr': 'lens',
                                  'choices': ['h05r2', 'h03r2'], 'levels': 2}]),
                   V('responses', [{'metric': 'coverage.blind_m2', 'params': []}]), V('design', 'factorial'),
                   V('samples', 1)],
    'campaign': [V('template', 'fault_detectability'), V('fault_sets', [
        {'kind': 'rotation', 'tags': ['F', 'ML'], 'axes': ['yaw', 'pitch', 'roll'], 'magnitudes': [0.1, 0.5, 1]}]),
        V('seeds', [1, 2, 3]), V('clean', True), V('evaluators', ['drivelog', 'asbuilt'])],
    'critical_carla': [V('seed', 2)],
    'sample_dataset': [V('duration_s', 20), V('nuscenes', 'true')],
    'study': [V('study', 'stitched_objects'), V('object_height', 1.0)],
    'export': [V('formats', ['extrinsics_json', 'nuscenes_calib'])],
}


def _step(p, kind, values, work):
    st = CP.resolve_step(p, kind, values, working_dir=work)
    assert st.ok, (kind, st.errors)
    return st


def test_the_card_says_every_value_of_every_kind_in_words_with_the_json_kept(work, monkeypatch):
    from vantage.core import detection as DT
    from vantage.ui import assistant as UA
    p = _with_targets(_rig(), 'Child 2 m ahead')
    real = DT.status
    monkeypatch.setattr(DT, 'status', lambda m: dict(real(m), present=True, verified=True))
    recs = _drive_recordings(p, work, 2)
    wordy = dict(WORDY, detection=[V('recordings', [r['name'] for r in CT.find_runs(p, work)][:2]),
                                   V('views', 'both')])
    assert set(wordy) == set(CP.KINDS)
    seen = {}
    for kind, values in wordy.items():
        st = _step(p, kind, values, work)
        rows = UA.plan_rows(st, p)
        assert rows and len({r[1] for r in rows}) == len(rows)
        for sec, key, lab, text, src, why in rows:
            raw = CP.get_path(st.settings, key, CP.flatten(st.settings).get(key))
            assert not text.lstrip().startswith(('[', '{')) and '":' not in text, (kind, key, text)   # never JSON
            assert len(text) <= CP.TEXT_MAX, (kind, key)
            if isinstance(raw, (list, dict)) and raw:
                seen[(kind, key)] = text
    assert seen[('experiment', 'factors')] == 'F pitch 25–45° (5 levels) · F lens h05r2, h03r2'
    assert seen[('experiment', 'responses')] == 'Blind ground in the ROI (m²), less is better'
    assert seen[('carla_run', 'faults')] == 'F turned pitch +1° · L1 range bias +5 cm'
    assert seen[('carla_run', 'actors')] == 'child at 12, −2 m'
    assert seen[('carla_live', 'faults')] == 'at 2 s: QR frozen · at 4 s: F turned pitch +1°'
    assert seen[('carla_live', 'cameras')] == 'F, QR'
    assert seen[('campaign', 'fault_sets')].startswith('rotation of F, ML by 0.1, 0.5, 1°')
    assert seen[('campaign', 'evaluators')] == 'Drive-log check · As-built check'
    assert seen[('detection', 'recordings')] == ', '.join(os.path.basename(r) for r in (recs[1], recs[0]))
    assert seen[('detection', 'range_edges')] == '0–10, 10–20, 20–40, 40+ m'
    assert seen[('detection', 'view_yaws')] == '−60°, 0°, 60°'
    assert seen[('critical_carla', 'spawn_points')] == 'chosen by the seed'
    assert seen[('export', 'formats')] == 'Extrinsics · nuScenes sensor calibration'
    rows = {r[1]: r for r in UA.plan_rows(_step(p, 'carla_run', WORDY['carla_run'], work), p)}
    assert rows['weather'][3] == 'Vantage fog' and rows['scenario.name'][3] == 'Cut-in'
    assert rows['scenario.params.speed_kmh'][2:4] == ('Speed', '40 km/h')
    assert rows['ground_truth.range'][2:4] == ('Range', 'off')
    # the generic fallback: any other structured value in words
    assert CP.generic_text({'max_px': [1, 2.5], 'opts': {'fast': True}, 'name': 'x'}) == \
        'max px (1, 2.5), opts (fast on), name x'
    assert CP.generic_text([{'a': 1}, {'a': -2}]) == 'a 1 · a −2' and CP.generic_text({}) == 'the page\'s defaults'
    assert CP.value_text('campaign', 'drivelog', {'speed_kmh': 30}) == 'speed kmh 30'
    assert CP.value_text('campaign', 'weathers', ['vantage:heavy_rain', 'ClearNoon']) == \
        'Vantage heavy rain · ClearNoon'


def test_the_change_line_says_each_changed_setting_in_words(work):
    from vantage.ui import assistant as UA
    p = _rig()
    v1 = CP.build(p, [{'kind': 'carla_run', 'values': [V('weather', 'HardRainSunset'), V('traffic', 30),
                                                        V('duration_s', 10), V('image_scale', 0.25), V('port', 2400),
                                                        V('tm_port', 8500), V('faults', [
                                                            {'tag': 'F', 'kind': 'rotation', 'deg': [0, 1, 0]}])]}],
                  'Drive', working_dir=work)
    v2 = CP.build(p, [{'kind': 'carla_run', 'values': [V('traffic', 20), V('weather', 'fog'), V('faults', [
        {'tag': 'F', 'kind': 'rotation', 'deg': [0, 1, 0]}, {'tag': 'L1', 'kind': 'range_bias', 'm': 0.05}])]}],
                  prev=v1, revise=True, working_dir=work)
    lines = UA.change_lines(v2, p)
    assert lines == ['Faults: added L1 range bias +5 cm', 'Traffic 30 → 20 vehicles',
                     'Weather HardRainSunset → Vantage fog'], lines
    v3 = CP.edit_value(p, v2, 0, 'faults', [], working_dir=work)
    assert UA.change_lines(v3, p) == ['Faults: removed F turned pitch +1° · L1 range bias +5 cm']
    v4 = CP.edit_value(p, v3, 0, 'image_scale', 0.5, working_dir=work)
    assert UA.change_lines(v4, p) == ['Image size 25 → 50 %']
    # an experiment's factor changed in place; a second step added; a scenario's traffic
    e1 = CP.build(p, [{'kind': 'experiment', 'values': WORDY['experiment']}], 'Exp', working_dir=work)
    e2 = CP.build(p, [{'kind': 'experiment', 'values': [V('factors', [
        {'id': 'F1', 'kind': 'camera', 'target': 'F', 'attr': 'pitch', 'lo': 25, 'hi': 40, 'levels': 4},
        {'id': 'F2', 'kind': 'camera', 'target': 'F', 'attr': 'lens', 'choices': ['h05r2', 'h03r2']}]),
        V('design', 'lhs')]}, {'kind': 'study', 'values': [V('study', 'blind_volume')]}],
        prev=e1, revise=True, working_dir=work)
    lines = UA.change_lines(e2, p)
    assert 'Step 1 · Factors: F pitch 25–45° (5 levels) → F pitch 25–40° (4 levels)' in lines, lines
    assert 'Step 1 · Design Full factorial → Latin hypercube' in lines and 'Step 2 added: Study' in lines
    d1 = CP.build(p, [{'kind': 'carla_run', 'values': [V('scenario.name', 'pedestrian_crossing'), V('traffic', 30),
                                                        V('image_scale', 0.25), V('port', 2400), V('tm_port', 8500)]}],
                  'D', working_dir=work)
    d2 = CP.build(p, [{'kind': 'carla_run', 'values': [V('traffic', 20)]}], prev=d1, revise=True, working_dir=work)
    assert UA.change_lines(d2, p) == ['Other traffic 30 → 20 vehicles',
                                      'Traffic 30 → 20 vehicles (not used by this scenario)']


def test_forced_only_when_the_page_changes_the_meaning_and_unused_settings_are_not_decisions(work):
    p = _rig()
    f = p.variant.by_tag('F')
    # the demo's experiment: the page filling in a response's defaults is not a limit; samples on a factorial design
    # is not used (never forced, not a decision); the factor's tag written as its uid is not a change
    ex = _step(p, 'experiment', WORDY['experiment'], work)
    assert ex.settings['factors'][0]['target'] == f.uid
    assert ex.sources['responses']['source'] == 'request' and ex.sources['factors']['source'] == 'request'
    assert ex.sources['samples']['source'] == 'request'
    assert not [k for k, s in ex.sources.items() if s['source'] == 'limit']
    un = CP.unused('experiment', ex.settings, p)
    assert un['samples'][0] == 'not used by this design' and 'Full factorial' in un['samples'][1]
    assert un['levels'] == ('not used by this design', 'every factor sets its own levels')
    assert set(un) == {'samples', 'sobol_n', 'morris_r', 'morris_levels', 'seed', 'levels'}
    m = CP.for_model(CP.Plan('P1', steps=[ex]))['steps'][0]
    assert 'samples' not in m['values_set'] and 'factors' in m['values_set'] and 'samples' in m['not_used']
    # the same request on a design that does use samples: a real limit
    lhs = _step(p, 'experiment', WORDY['experiment'][:2] + [V('design', 'lhs'), V('samples', 1)], work)
    assert lhs.sources['samples'] == {'source': 'limit', 'reason': 'asked 1; the page allows 2–100000, so 2'}
    # every kind's equivalents: numbers as text or integers, a list as text, a list in another order, objects
    # without the parts the page fills in, a categorical factor's derived range, the live rig's own fault form
    for kind, values in WORDY.items():
        st = _step(p if kind != 'critical_carla' else _with_targets(_rig(), 'Child 2 m ahead'), kind, values, work)
        forced = {k: s for k, s in st.sources.items() if s['source'] == 'limit'}
        assert not forced, (kind, forced)
    for kind, values in SCRIPTED.items():
        st = _step(p if kind != 'critical_carla' else _with_targets(_rig(), 'Child 2 m ahead'), kind, values, work)
        assert not [k for k, s in st.sources.items() if s['source'] == 'limit'], kind
    # real changes are forced, said in words
    st = _step(p, 'carla_run', [V('faults', [{'tag': 'F', 'kind': 'rotation', 'deg': [0, 45, 0]}]),
                                V('scenario', {'name': 'cut_in', 'params': {'speed_kmh': 200}}), V('image_scale', 0.25),
                                V('port', 2400), V('tm_port', 8500)], work)
    assert st.sources['faults'] == {'source': 'limit',
                                    'reason': 'the page changed F turned pitch +45° → F turned pitch +30°'}
    assert st.sources['scenario.params.speed_kmh'] == {'source': 'limit',
                                                       'reason': 'asked 200; the scenario allows 5–90, so 90'}
    st = _step(p, 'experiment', [V('factors', [{'kind': 'camera', 'target': 'F', 'attr': 'pitch', 'lo': 25,
                                                'hi': 95}]), V('responses', [{'metric': 'coverage.blind_m2'}]),
                                 V('design', 'lhs')], work)
    assert st.sources['factors'] == {'source': 'limit', 'reason': 'the page changed F pitch 25–95° → F pitch 25–90°'}
    st = _step(p, 'experiment', [V('factors', [{'kind': 'camera', 'target': 'F', 'attr': 'pitch', 'lo': 25, 'hi': 45},
                                               {'kind': 'teleport', 'attr': 'x'}]),
                                 V('responses', [{'metric': 'coverage.blind_m2', 'params': {}, 'colour': 'red'}]),
                                 V('design', 'lhs')], work)
    assert st.sources['factors']['source'] == 'limit' and 'kept 1 of 2' in st.sources['factors']['reason']
    assert st.sources['responses']['source'] == 'request' and 'does not use colour' in st.sources['responses']['reason']
    st = _step(p, 'carla_live', [V('scale', 3.0)], work)
    assert st.sources['scale'] == {'source': 'limit', 'reason': 'asked 3; the page allows 0.05–1, so 1'}
    # a scenario has its own traffic: the request's goes there, the free drive's is not used
    st = _step(p, 'carla_run', [V('scenario.name', 'pedestrian_crossing'), V('traffic', 500), V('image_scale', 0.25),
                                V('port', 2400), V('tm_port', 8500)], work)
    assert st.settings['scenario']['params']['traffic'] == 100.0
    assert st.sources['scenario.params.traffic'] == {'source': 'limit',
                                                     'reason': 'asked 500; the scenario allows 0–100, so 100'}
    assert st.sources['traffic']['source'] == 'request'                  # 300 kept, never "forced": not used
    assert CP.unused('carla_run', st.settings, p)['traffic'][0] == 'not used by this scenario'
    st = _step(p, 'carla_run', [V('scenario.name', 'cut_in'), V('traffic', 5), V('scenario.params.traffic', 7),
                                V('image_scale', 0.25), V('port', 2400), V('tm_port', 8500)], work)
    assert st.settings['scenario']['params']['traffic'] == 7.0          # what was asked for the scenario stays
    st = _step(p, 'carla_run', [V('tick_hz', 80), V('min_tick_hz', 500), V('image_scale', 0.25), V('port', 2400),
                                V('tm_port', 8500)], work)
    assert st.sources['min_tick_hz']['source'] == 'request' and 'min_tick_hz' in CP.unused('carla_run', st.settings)
    # the other kinds' unused settings
    assert set(CP.unused('study', {'study': 'blind_volume'})) == {'object_height', 'tolerance_deg', 'tolerance_mm',
                                                                 'condition'}
    assert set(CP.unused('detection', {'views': 'fisheye'})) == set(CP.DETECTION_VIEW_KEYS)
    assert set(CP.unused('campaign', {'evaluators': ['drivelog']})) == {'health', 'detection'}
    assert CP.unused('sample_dataset', {}) == {} and CP.unused('export', {}) == {}


def test_the_design_fields_the_copilot_knows_are_the_experiment_pages_own():
    from vantage.ui.workspaces.experiments_ws import DESIGN_FIELDS
    from vantage.core import experiments as EX
    page = {k: tuple(d) for k, _lab, _dec, _tip, d in DESIGN_FIELDS if tuple(d) != tuple(EX.DESIGNS)}
    assert page == CP.DESIGN_USES


def test_batch_runs_an_approved_plan_and_refuses_anything_else(work, tmp_path, capsys):
    from vantage.core import provenance as PV
    p = _rig()
    p.analysis.dataset = {'image_scale': 0.1}
    plan = CP.build(p, [
        {'kind': 'sample_dataset', 'label': 'Sample', 'values': [V('duration_s', 0.6), V('speed_kmh', 20),
                                                                  V('image_scale', 0.1)]},
        {'kind': 'study', 'label': 'Load cases', 'values': [V('study', 'load_cases')]},
        {'kind': 'export', 'label': 'Extrinsics', 'values': [V('formats', ['extrinsics_json'])]}],
        'Sample, study, export', '', working_dir=work)
    assert plan.ok, plan.errors
    rec, _ = CP.approve(p, plan, working_dir=work)
    CP.add_record(p, rec)
    path = str(tmp_path / 'rig.vantage')
    p.save(path)
    out = tmp_path / 'planrun'
    code = _batch([path, '--run-plan', rec['id'], str(out)])
    err = capsys.readouterr().err
    assert code == 0, err
    ds = next((out / 'step-1-sample_dataset').glob('**/dataset.json'))
    pv = json.loads(ds.read_text())['provenance']
    assert pv['plan'] == {'plan': rec['id'], 'version': 1, 'step': 0} and pv['interface'] == 'batch'
    study = json.loads((out / 'step-2-study' / 'load_cases.json').read_text())
    assert study['provenance']['plan']['step'] == 1 and study['cases']
    ext = json.loads((out / 'step-3-export' / 'extrinsics.json').read_text())
    assert ext['provenance']['plan']['step'] == 2
    run = json.loads((out / 'copilot_plan.json').read_text())
    assert run['runs'][-1]['via'] == 'batch' and [s['status'] for s in run['runs'][-1]['steps']] == ['done'] * 3
    assert run['provenance']['plan']['plan'] == rec['id']
    # refused: an unknown plan, a record changed after approval
    assert _batch([path, '--run-plan', 'PNOTHERE', str(tmp_path / 'x')]) == CP.EXIT_REFUSED
    from vantage.core.project import Project
    q = Project.load(path)
    q.analysis.copilot['plans'][0]['steps'][0]['settings']['duration_s'] = 9.0
    q.save(path)
    assert _batch([path, '--run-plan', rec['id'], str(tmp_path / 'y')]) == CP.EXIT_REFUSED
    assert 'does not verify' in capsys.readouterr().err


# ============================================================================================ conversations (Groq)
@pytest.fixture()
def groq(monkeypatch):
    s = FakeGroq().start()
    monkeypatch.setenv('VANTAGE_ASSISTANT_URL', s.url)
    yield s
    s.stop()


def _propose(steps, title='Plan', revise=False):
    return {'title': title, 'rationale': 'as you asked', 'revise': revise, 'steps': steps}


SCRIPTED = {   # kind -> the values a model would set for the brief's example requests
    'carla_run': [V('town', 'Town10HD'), V('weather', 'SoftRainSunset', 'copilot', 'dusk in rain'),
                  V('duration_s', 10), V('traffic', 30), V('walkers', 10), V('scenario.name', 'pedestrian_crossing'),
                  V('image_scale', 0.5), V('port', 2400, 'copilot', 'not 2000-2010'), V('tm_port', 8500, 'copilot', '')],
    'campaign': [V('template', 'fault_detectability'), V('fault_sets', [
        {'kind': 'rotation', 'tags': ['F', 'ML'], 'axes': ['yaw', 'pitch', 'roll'], 'magnitudes': [0.1, 0.5, 1.0]}]),
        V('seeds', [1, 2, 3]), V('clean', True)],
    'critical_carla': [V('seed', 2)],
    'carla_live': [V('faults', [{'at_s': 10, 'fault': {'tag': 'QR', 'kind': 'freeze'}}])],
    'experiment': [V('factors', [{'kind': 'camera', 'target': 'F', 'attr': 'pitch', 'mode': 'value', 'lo': 25,
                                  'hi': 45, 'levels': 5}]), V('responses', [{'metric': 'coverage.blind_m2'}]),
                   V('design', 'oat')],
    'sample_dataset': [V('duration_s', 20), V('speed_kmh', 40)],
    'study': [V('study', 'blind_volume')],
    'export': [V('formats', ['extrinsics_json', 'nuscenes_calib'])],
}


def test_scripted_conversations_produce_plans_of_every_kind(groq, work, monkeypatch):
    from vantage.core import detection as DT
    p = _with_targets(_rig(), 'Child 2 m ahead')
    recs = _drive_recordings(p, work, 2)
    real = DT.status
    monkeypatch.setattr(DT, 'status', lambda m: dict(real(m), present=True, verified=True))
    scripted = dict(SCRIPTED, detection=[V('recordings', [r['name'] for r in CT.find_runs(p, work)][:2])])
    e = Engine()
    for kind, values in scripted.items():
        s = A.Session('test-key', A.Settings('openai/gpt-oss-120b', planning=True))
        groq.script += [groq.tool_call('get_catalogue', {'topic': 'plan_kinds'}, 'c1'),
                        groq.tool_call('propose_plan', _propose([{'kind': kind, 'label': kind, 'values': values}]),
                                       'c2'),
                        groq.answer('The plan is on the card: approve it to run.')]
        tc = T.ToolContext(p.snapshot(), e.analyse(p) if kind in ('study', 'export') else None, e.share(),
                           working_dir=work, provider='groq', model='openai/gpt-oss-120b')
        ev = []
        s.run_turn(f'Plan a {kind}', tc, ev.append, threading.Event())
        assert ev[-1].data['ok'], [x.data for x in ev if x.kind == 'error']
        plans = [x.data['plan'] for x in ev if x.kind == 'copilot']
        assert len(plans) == 1 and plans[0].steps[0].kind == kind and plans[0].ok, (kind, plans[0].errors)
        assert plans[0].request == [f'Plan a {kind}'] and plans[0].provider == 'groq'
    assert len(recs) == 2


# ============================================================================================ never sent
def secrets_and_paths(project, tmp_path):
    """What must never leave the machine, set up in the project and the settings: keys, a stream password, the
    automation command, this machine's folders (image and point-cloud bytes are scanned for by their signatures)."""
    from PyQt6.QtCore import QSettings
    from vantage.core import automation as AU
    pw = 'S3cretStreamPw!'
    project.path = str(tmp_path / 'secret-folder' / 'rig.vantage')
    project.streams = {project.variant.cameras[0].uid: f'rtsp://admin:{pw}@10.0.0.5/stream'}
    command = 'rm -rf /very/secret/automation-command'
    AU.CommandStore(QSettings('Vantage', 'Vantage')).put(project.path, command)
    return {'groq_key': 'test-key', 'claude_key': 'sk-ant-test-key', 'stream_pw': pw, 'command': command,
            'home': os.path.expanduser('~'), 'tmp': str(tmp_path)}


def scan(raws, needles: dict):
    """Every request body (and header but the auth one) the fakes received: none holds a needle, an image or a
    point cloud."""
    for headers, body in raws:
        text = body.decode('utf-8', 'replace')
        for name, n in needles.items():
            if name.endswith('_key'):
                assert n not in text, name                       # keys go in the auth header only
            elif name in ('home', 'tmp'):
                assert n not in text, (name, text[text.find(n) - 80:text.find(n) + 80])
            else:
                assert n not in text, name
        for sig in ('iVBORw0KGgo', '/9j/4', '"type": "image"', '"type":"image"', 'data:image', '.pcd'):
            assert sig not in text, sig
        for k, v in headers.items():
            if k.lower() not in ('authorization', 'x-api-key'):
                for name, n in needles.items():
                    assert n not in str(v), (k, name)


def test_nothing_private_reaches_groq(groq, work, tmp_path, monkeypatch):
    p = _rig()
    needles = secrets_and_paths(p, tmp_path)
    s = A.Session('test-key', A.Settings('openai/gpt-oss-120b'))
    groq.script += [groq.tool_call('get_overview', {}, 'c0'), groq.tool_call('get_rig_health', {}, 'c1'),
                    groq.tool_call('get_automation', {}, 'c2'),
                    groq.tool_call('get_catalogue', {'topic': 'past_runs'}, 'c3'),
                    groq.tool_call('propose_plan', _propose([{'kind': 'sample_dataset', 'label': 's', 'values': [
                        V('duration_s', 5)]}]), 'c4'),
                    groq.answer('ok')]
    tc = T.ToolContext(p.snapshot(), None, Engine(), working_dir=work)
    ev = []
    s.run_turn(f'My files are in {tmp_path} and {os.path.expanduser("~")}/Documents', tc, ev.append,
               threading.Event())
    assert ev[-1].data['ok']
    assert len(groq.raw) >= 6
    scan(groq.raw, {k: v for k, v in needles.items() if k != 'claude_key'})


# ============================================================================================ keys per provider
def test_each_provider_has_its_own_key_in_the_keyring_or_a_private_file(monkeypatch):
    for prov in PR.PROVIDERS:
        K.delete(prov)
    assert K.save('gsk_groqgroqgroqgroq', 'groq') == 'file' and K.save('sk-ant-claudeclaude', 'claude') == 'file'
    assert K.load('groq') == ('gsk_groqgroqgroqgroq', 'file') and K.load('claude') == ('sk-ant-claudeclaude', 'file')
    assert os.path.basename(K._path('claude')) != os.path.basename(K._path('groq'))
    assert os.environ['XDG_CONFIG_HOME'] in K._path('claude')              # the test's own folder
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'sk-ant-fromenv')
    assert K.load('claude') == ('sk-ant-fromenv', 'environment') and K.load('groq')[1] == 'file'
    monkeypatch.delenv('ANTHROPIC_API_KEY')
    K.delete('claude')
    assert K.load('claude') == (None, None) and K.load('groq')[0] == 'gsk_groqgroqgroqgroq'
    K.delete('groq')

    class FakeKeyring:                                  # a keyring of the test's own: never the system's
        def __init__(self):
            self.d = {}

        def get_password(self, s, a):
            return self.d.get((s, a))

        def set_password(self, s, a, k):
            self.d[(s, a)] = k

        def delete_password(self, s, a):
            self.d.pop((s, a), None)
    kr = FakeKeyring()
    monkeypatch.setattr(K, '_keyring', lambda: kr)
    assert K.save('gsk_inthering', 'groq') == 'keyring' and K.save('sk-ant-inthering', 'claude') == 'keyring'
    assert kr.d == {('Vantage', 'groq-api-key'): 'gsk_inthering', ('Vantage', 'anthropic-api-key'): 'sk-ant-inthering'}
    assert K.load('claude') == ('sk-ant-inthering', 'keyring')
    K.delete('claude')
    assert ('Vantage', 'anthropic-api-key') not in kr.d and K.load('groq')[0] == 'gsk_inthering'
    with pytest.raises(KeyError):
        K.load('openai')


def test_costs_and_the_usage_line():
    u = PR.Usage(provider=PR.CLAUDE, model='claude-opus-5-5')
    u.add_claude('claude-opus-5-5', 1_000_000, 100_000, 200_000, 2_000_000)
    assert u.cost() == pytest.approx(4.0 + 2.0 + 1.0 + 0.4)
    assert 'cache_read_input_tokens' in u.line() and '$' in u.line()
    s = PR.Usage(provider=PR.CLAUDE, model='claude-sonnet-5-5')
    s.add_claude('claude-sonnet-5-5', 1_000_000, 1_000_000)
    assert s.cost() == pytest.approx(12.0)
    g = PR.Usage(provider=PR.GROQ, model='openai/gpt-oss-120b')
    g.add({'prompt_tokens': 1000, 'completion_tokens': 500})
    assert g.cost() is None and 'not estimated' in g.line()
    assert g.cost({'input': 1.0, 'output': 2.0}) == pytest.approx(0.002)
    assert PR.PRICE_SOURCE[PR.CLAUDE].startswith('Anthropic')


# ============================================================================================ the panel
from test_gui import app, win, pump, settle, _wait  # noqa: E402,F401  (fixtures)


def _ready(win, provider='groq', key='test-key', consent=True):
    st = win.ctx.settings
    K.save(key, provider)
    st.setValue('assistant/enabled', True)
    st.setValue('assistant/provider', provider)
    st.setValue(f'assistant/consent_{provider}', consent)
    st.setValue('assistant/model', 'openai/gpt-oss-120b')
    st.setValue('assistant/claude_model', 'claude-opus-5-5')
    st.setValue('assistant/daily_cap', 0)
    win.ctx.assistant.reload()


def _idle(app, win, timeout=60):
    t = time.time()
    while win.ctx.assistant.busy and time.time() - t < timeout:
        pump(app, 0.05)
    pump(app, 0.2)


def _small_rig(win):
    """The data rig with one camera and a mast lidar: a quick drive against the fake carla (as test_sim's)."""
    v = next(v for v in win.store.project.variants if v.sensors)

    def fn(pj):
        pj.active_uid = v.uid
        vv = pj.variant
        vv.cameras[:] = vv.cameras[:1]
        next(s for s in vv.sensors if s.kind == 'lidar').z = 2.0
    win.store.edit(['variants'], fn, 'test rig')


def _cards(panel):
    from vantage.ui.assistant import CopilotCard
    return panel.findChildren(CopilotCard)


DRIVE = [V('duration_s', 1), V('image_scale', 0.05), V('traffic', 1), V('warmup_s', 0.1),
         V('town', 'Town10HD'), V('weather', 'ClearSunset', 'copilot', 'dusk'), V('port', 2400, 'copilot', 'free'),
         V('tm_port', 8500, 'copilot', 'free')]


def test_plan_card_edit_diff_open_approve_record_and_the_page_run(app, win, groq, tmp_path, monkeypatch):
    from vantage import workdir
    from vantage.sim import server as SV
    from vantage.ui.workspaces.sim_ws import SimPage
    old_work = win.ctx.settings.value(workdir.KEY, '')
    workdir.put(win.ctx.settings, str(tmp_path / 'work'))
    try:
        _ready(win)
        _small_rig(win)
        settle(app, win)
        monkeypatch.setattr(SV, 'port_open', lambda *a, **k: True)           # the fake carla stands in for a server
        pg = win.page('sim')
        assert isinstance(pg, SimPage)
        pg.python, pg.extra_pythonpath = sys.executable, FAKE_CARLA
        win.show_assistant()
        panel, ctl = win.assistant_panel, win.ctx.assistant
        groq.script += [groq.tool_call('propose_plan', _propose([{'kind': 'carla_run', 'label': 'Short dusk drive',
                                                                  'values': DRIVE}], 'A short drive'), 'c1'),
                        groq.answer('The plan is on the card; nothing runs until you approve it.')]
        from PyQt6.QtCore import Qt
        from PyQt6.QtTest import QTest
        QTest.keyClicks(panel.composer.edit, 'a short drive in Town10 at dusk')
        QTest.keyClick(panel.composer.edit, Qt.Key.Key_Return)
        _idle(app, win)
        assert groq.requests[0]['tools'] and any(t['function']['name'] == 'propose_plan'
                                                  for t in groq.requests[0]['tools'])
        card = _cards(panel)[-1]
        plan = card.plan
        assert plan.ok and plan.version == 1 and card.b_approve.isVisible() and card.b_approve.isEnabled()
        assert 'needs your approval' in card.badge.text()
        # nothing has started: no run, no record, no page busy
        assert not pg.busy and pg.result is None and CP.records(win.store.project) == []
        runs = workdir.project_dir(win.store.project, str(tmp_path / 'work'), 'carla')
        assert not os.path.isdir(runs) or not os.listdir(runs)
        # the table: every value with its source; a value edited in place (the page's own number field)
        t = card.tables[0]
        r = t.keys.index('duration_s')
        assert t.item(r, 2).text() == 'your request' and t.item(r, 3).text() == 'you asked for it'
        assert t.item(t.keys.index('seed'), 2).text() == 'default'
        ed = card._edit_cell(t, r)
        from vantage.ui.widgets import ScrubSpinBox
        assert isinstance(ed, ScrubSpinBox)
        ed.set_value(1.5)
        ed.committed.emit(1.5)
        pump(app, 0.2)
        assert card.plan.version == 2 and card.plan.steps[0].settings['duration_s'] == 1.5
        assert card.diff_lbl is not None and 'Changed from v1' in card.diff_lbl.text()
        assert card.change_items == ['Duration 1 → 1.5 s'] and 'duration_s' not in card.diff_lbl.text()
        assert plan.status == 'superseded' and ctl.current_plan is card.plan
        # Open on its page: the page's fields filled as one undo step; nothing runs
        n0 = win.store.undo.index()
        QTest.mouseClick(card.b_open, Qt.MouseButton.LeftButton)
        pump(app, 0.3)
        assert win.page_name() == 'sim' and win.store.undo.index() == n0 + 1
        assert pg.settings.duration_s == 1.5 and pg.settings.town == 'Town10HD' and not pg.busy
        win.store.undo.undo()
        pump(app, 0.2)
        assert pg.settings.duration_s != 1.5
        # a forged ticket is refused by the page itself
        assert not pg.run(plan={'plan': card.plan.id, 'version': 2, 'step': 0})
        assert 'No approval record' in pg.g_status.text() and not pg.busy
        # Approve and run: the record, then the page's own run
        QTest.mouseClick(card.b_approve, Qt.MouseButton.LeftButton)
        pump(app, 0.3)
        rec = CP.find(win.store.project, card.plan.id, 2)
        assert rec is not None and CP.verify(rec) and rec['request'] == ['a short drive in Town10 at dusk']
        assert rec['provider'] == 'groq' and rec['model'] == 'openai/gpt-oss-120b'
        assert card.plan.status == 'approved' and not card.b_approve.isVisible()
        _wait(app, lambda: not ctl.runner.running, 600)
        rec = CP.find(win.store.project, card.plan.id, 2)
        assert rec['status'] == 'done', rec['runs'][-1]
        assert pg.result['status'] == 'ok' and os.path.isfile(os.path.join(pg.out_dir, 'run.json'))
        run_doc = json.load(open(os.path.join(pg.out_dir, 'run.json')))
        assert run_doc['provenance']['plan'] == {'plan': rec['id'], 'version': 2, 'step': 0}
        assert json.load(open(os.path.join(pg.out_dir, 'copilot_plan.json')))['plan']['id'] == rec['id']
        folder = CP.run_folder(win.store.project, rec, str(tmp_path / 'work'))
        rd = json.load(open(os.path.join(folder, 'copilot_plan.json')))
        assert rd['runs'][-1]['steps'][0]['status'] == 'done' and rd['provenance']['plan']['plan'] == rec['id']
        assert 'Approved' in card.state_lbl.text() and '1 done' in card.state_lbl.text()
        # the record survives undoing everything since
        while win.store.undo.canUndo():
            win.store.undo.undo()
        pump(app, 0.2)
        assert CP.find(win.store.project, rec['id'], 2) is not None
        # run again from the Plans list: the stored settings, no model asked
        n_req = len(groq.requests)
        dlg = panel.show_plans()
        assert dlg.t.rowCount() == 1 and rec['id'] in dlg.t.item(0, 0).text()
        assert dlg.run_again() == ''
        _wait(app, lambda: not ctl.runner.running, 600)
        rec = CP.find(win.store.project, rec['id'], 2)
        assert rec['status'] == 'done' and [r['via'] for r in rec['runs']] == ['app', 'rerun']
        assert len(groq.requests) == n_req
        dlg.close()
    finally:
        workdir.put(win.ctx.settings, str(old_work or ''))


def test_plan_card_shows_values_in_words_keeps_the_json_greys_unused_and_edits_a_list(app, win, work):
    from PyQt6.QtGui import QColor
    from PyQt6.QtWidgets import QLineEdit
    from vantage.ui.assistant import CopilotCard
    from vantage.ui.theme import THEME
    ctl = win.ctx.assistant
    plan = CP.build(win.store.project.snapshot(), [{'kind': 'experiment', 'label': 'Front pitch vs lens',
                                                    'values': WORDY['experiment']}], 'Experiment', working_dir=work)
    assert plan.ok, plan.errors
    card = CopilotCard(ctl, plan)
    card.resize(900, 1000)
    card.show()
    pump(app, 0.2)
    try:
        t = card.tables[0]
        st = card.plan.steps[0]
        r = t.keys.index('factors')
        assert t.item(r, 1).text() == 'F pitch 25–45° (5 levels) · F lens h05r2, h03r2'
        tip = t.item(r, 1).toolTip()
        assert tip.startswith('factors: F pitch 25–45° (5 levels) · F lens h05r2, h03r2\n\n')
        assert json.loads(tip.split('\n\n', 1)[1]) == st.settings['factors']
        assert 120 <= t.columnWidth(1) <= 320
        rr = t.keys.index('responses')
        assert t.item(rr, 1).text() == 'Blind ground in the ROI (m²), less is better'
        assert t.item(rr, 2).text() == 'your request'
        assert not any(t.item(i, 2) is not None and t.item(i, 2).text() == 'forced by a limit'
                       for i in range(t.rowCount()))
        # samples on a factorial design: last, in its own section, greyed, "not used"
        rs = t.keys.index('samples')
        first = min(i for i, k in enumerate(t.keys) if k is not None and t.item(i, 2).text() == 'not used')
        assert t.keys[first - 1] is None and t.item(first - 1, 0).text() == 'Not used by this design'
        assert rs >= first and t.item(rs, 2).text() == 'not used' and 'Full factorial' in t.item(rs, 3).text()
        muted = QColor(THEME.tok['muted']).name()
        assert all(t.item(rs, c).foreground().color().name() == muted for c in range(4))
        assert t.item(t.keys.index('design'), 2).foreground().color().name() != muted
        # a list value edited in place (its JSON, the existing editor): a new version, the change in words
        ed = card._edit_cell(t, r)
        assert isinstance(ed, QLineEdit) and json.loads(ed.text()) == st.settings['factors']
        new = copy.deepcopy(st.settings['factors'])
        new[0]['hi'] = 40.0
        ed.setText(json.dumps(new))
        ed.returnPressed.emit()
        pump(app, 0.2)
        assert card.plan.version == 2 and card.plan.steps[0].settings['factors'][0]['hi'] == 40.0
        assert card.change_items == ['Factors: F pitch 25–45° (5 levels) → F pitch 25–40° (5 levels)']
        txt = card.diff_lbl.text()
        assert 'Changed from v1' in txt and '"hi"' not in txt and '{' not in txt
        assert json.loads(card.diff_lbl.toolTip())[0]['key'] == 'factors'
        # a number through the page's own field still works
        from vantage.ui.widgets import ScrubSpinBox
        t = card.tables[0]
        ed = card._edit_cell(t, t.keys.index('bootstrap'))
        assert isinstance(ed, ScrubSpinBox)
        ed.set_value(200)
        ed.committed.emit(200.0)
        pump(app, 0.2)
        assert card.plan.version == 3 and card.change_items == ['Bootstrap 500 → 200']
    finally:
        card.hide()
        card.deleteLater()
        pump(app, 0.1)


def test_consent_reject_and_what_was_sent(app, win, groq, tmp_path):
    _ready(win, consent=False)
    win.show_assistant()
    panel, ctl = win.assistant_panel, win.ctx.assistant
    groq.script += [groq.tool_call('propose_plan', _propose([{'kind': 'sample_dataset', 'label': 'Sample',
                                                              'values': [V('duration_s', 5)]}], 'Sample'), 'c1'),
                    groq.answer('On the card.')]
    panel.ask(f'a 5 s sample dataset; my files are in {os.path.expanduser("~")}/projects')
    pump(app, 0.2)
    from vantage.ui.assistant import ConsentCard
    cards = panel.findChildren(ConsentCard)
    assert cards and cards[-1].isVisible() and 'Groq' in cards[-1].b_ok.text() and not groq.requests
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    QTest.mouseClick(cards[-1].b_ok, Qt.MouseButton.LeftButton)
    _idle(app, win)
    assert ctl.consented('groq') and not ctl.consented('claude') and groq.requests
    card = _cards(panel)[-1]
    QTest.mouseClick(card.b_reject, Qt.MouseButton.LeftButton)
    pump(app, 0.2)
    assert card.plan.status == 'rejected' and not card.b_approve.isVisible() and 'Rejected' in card.state_lbl.text()
    assert CP.records(win.store.project) == []
    dlg = panel.show_sent()
    text = dlg.text.toPlainText()
    assert 'propose_plan' in text and '~/projects' in text and os.path.expanduser('~') + '/' not in text
    assert 'test-key' not in text
    dlg.close()
    # the daily cap: nothing is sent once it is used up
    win.ctx.settings.setValue('assistant/daily_cap', 1)
    n = len(groq.requests)
    panel.ask('another question')
    pump(app, 0.3)
    assert len(groq.requests) == n
    win.ctx.settings.setValue('assistant/daily_cap', 0)


def test_preferences_switch_provider_and_say_when_the_sdk_is_missing(app, win, monkeypatch):
    from vantage.ui.dialogs.misc import PreferencesDialog
    K.delete('groq')
    K.delete('claude')
    monkeypatch.setattr(PR, 'claude_available', lambda: False)
    d = PreferencesDialog(win.ctx, win, tab='assistant')
    pg = d.assistant
    pg.on.setChecked(True)
    pg.provider.set_value('claude')
    pg._provider_changed('claude')
    assert 'pip install' in pg.sdk.text() and not pg.test.isEnabled() and not pg.fallbacks.isHidden()
    assert [pg.model.itemData(i) for i in range(pg.model.count())] == ['claude-opus-5-5', 'claude-sonnet-5-5']
    pg._edited('sk-ant-prefs-key')
    pg._provider_changed('groq')
    pg._edited('gsk-prefs-key')
    pg._provider_changed('claude')
    d.apply()
    assert K.load('claude') == ('sk-ant-prefs-key', 'file') and K.load('groq') == ('gsk-prefs-key', 'file')
    ctl = win.ctx.assistant
    assert ctl.provider == 'claude' and ctl.model == 'claude-opus-5-5' and ctl.key == 'sk-ant-prefs-key'
    assert not ctl.ready and not ctl.available
    panel = win.assistant_panel
    panel._sync()
    assert 'pip install' in panel.foot.text() and 'SDK' in panel.title.text()
    settings_text = str([win.ctx.settings.value(k) for k in win.ctx.settings.allKeys()])
    assert 'sk-ant-prefs-key' not in settings_text and 'gsk-prefs-key' not in settings_text
    win.ctx.settings.setValue('assistant/provider', 'groq')
    ctl.reload()
    assert ctl.provider == 'groq' and ctl.key == 'gsk-prefs-key'
    K.delete('groq')
    K.delete('claude')


def test_every_page_refuses_a_plan_without_its_approval_record(app, win, tmp_path):
    """The gate is in the pages' own Run entry points: reached from a plan (a ticket), each refuses unless the project
    holds that plan's intact approval record with exactly the page's settings."""
    forged = {'plan': 'PFORGED', 'version': 1, 'step': 0}
    pages = {'campaign': lambda pg: pg.run(plan=forged), 'detection': lambda pg: pg.run(plan=forged),
             'critical_carla': lambda pg: pg.run(plan=forged),
             'experiments': lambda pg: pg.run(confirm=False, plan=forged),
             'dataset': lambda pg: pg.generate(str(tmp_path / 'ds'), plan=forged)}
    for name, go in pages.items():
        pg = win.page(name)
        assert go(pg) is False, name
        assert not pg.busy, name
        text = getattr(pg, 'status_text', '') or pg.g_status.text()
        assert 'approval record' in text, (name, text)
    live = win.page('sim').live
    assert live.start(plan=forged) is False and not live.running
    assert 'approval record' in live.status_line.text.text()
    assert not os.path.exists(tmp_path / 'ds')
    # a real record whose settings differ from the page's: refused too
    p = win.store.project
    plan = CP.build(p, [{'kind': 'sample_dataset', 'label': 's', 'values': [V('duration_s', 3), V('image_scale', 0.1)]}],
                    'S', '')
    rec, _ = CP.approve(p, plan)
    win.store.edit('analysis', lambda pj: CP.add_record(pj, rec), 'test record')
    pg = win.page('dataset')
    assert pg.generate(str(tmp_path / 'ds'), plan=CP.ticket(rec, 0)) is False and 'differ' in pg.g_status.text()
