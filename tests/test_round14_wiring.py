"""Round 14 wired into the app: every new page opens from its menu action, every new export format writes exactly the
files it declares (or its NOT_WRITTEN note), every new batch option runs on the sample with its exit code, every new
assistant and MCP tool returns JSON, the Issues list reads the fleet twin, the optimiser takes the learned model's
term, robust perception takes the fleet twin's tolerances and the 3D metric of rig perception, rig perception marks a
re-shot drive's records, and the requirements scorecard has the risk-weighted coverage metric. Nothing here needs the
network, the GPU, CARLA or a downloaded model."""
import json
import os

import numpy as np
import pytest

from vantage.core import exporters as ex
from vantage.core import perf_model as PM
from vantage.core import samples
from vantage.core.analysis import Engine

from test_gui import app, win, pump, settle, _wait   # noqa: F401 (fixtures)

PAGES = ('fleet_twin', 'perf_model', 'designer', 'risk_coverage', 'rig_perception', 'foundation', 'robust_perception')
HEAVY = ('rig_perception', 'redrive', 'foundation_range', 'robust_perception', 'design_options')
CHEAP = ('risk_coverage', 'fleet_twin', 'fleet_twin_schedule', 'fleet_twin_tolerances', 'perf_model',
         'perception_records')
TOOLS = ('get_rig_perception', 'plan_perception_scene_set', 'get_redrive', 'get_foundation_range',
         'get_robust_perception', 'get_risk_coverage', 'get_fleet_twin', 'get_fleet_tolerances', 'get_perf_model',
         'predict_detection', 'suggest_simulations', 'read_requirements', 'design_rig')


@pytest.fixture(autouse=True)
def _perf_dir(tmp_path, monkeypatch):
    """The learned model's store and saved model: never the user's."""
    monkeypatch.setenv('VANTAGE_PERF_DIR', str(tmp_path / 'perf'))
    PM._STORES.clear()


@pytest.fixture(scope='module')
def rig():
    """The sample's data rig (6 cameras, a lidar, 5 radars), analysed once."""
    p = samples.tigor_sample_project()
    p.active_uid = next(v for v in p.variants if v.sensors).uid
    eng = Engine()
    return p, eng, eng.analyse(p)


def _written(paths, out):
    return sorted(os.path.relpath(x, out).replace(os.sep, '/') for x in paths)


def _toy_model(save=True):
    """A model trained on synthetic perception records (test_perf_model's analytic truth), saved where load_model()
    finds it."""
    from test_perf_model import synth_set
    docs, feats = synth_set(24, 6)
    st = PM.PerfModelSettings(classes=['pedestrian'], restarts=1, objective={'pedestrian': 1.0})
    m = PM.Model(st).fit(docs, feats_of=lambda d: feats[d['rig_hash']])
    assert m.trained
    if save:
        PM.save_model(m)
    return m, docs


# ======================================================================= pages from their menus
def test_every_new_page_opens_from_its_menu_action(app, win):
    from vantage.ui.mainwindow import PAGE_CLASSES
    for name in PAGES:
        win.actions_by_id[name].trigger()
        pump(app, 0.2)
        pg = win._page_widgets.get(name)
        assert pg is not None and type(pg).__name__ == PAGE_CLASSES[name][1], name
        assert win.pages.currentWidget() is pg, name
    settle(app, win)
    win.set_page('viewports')
    pump(app, 0.2)


# ======================================================================= exports
def test_every_new_format_is_registered_with_a_group_and_a_default():
    ids = {f.id: f for f in ex.FORMATS}
    for fid in HEAVY + CHEAP:
        f = ids[fid]
        assert f.group in ex.GROUPS and f.label and f.ext_label
        assert f.default is (fid in ('risk_coverage', 'fleet_twin', 'fleet_twin_schedule', 'fleet_twin_tolerances')), fid


def test_new_formats_write_exactly_their_declared_names(rig, tmp_path):
    p, eng, res = rig
    for fid in HEAVY + CHEAP:
        out = str(tmp_path / fid)
        fmt = ex.format_by_id(fid)
        want = fmt.filename(p, p.variant)
        want = sorted([want] if isinstance(want, str) else want)
        got = _written(ex.export([fid], p, res, ex.ExportOptions(out_dir=out, engine=eng)), out)
        assert got == want, (fid, got, want)
        rel, text, size = ex.preview(fid, p, res, ex.ExportOptions(out_dir=out, engine=eng))
        assert isinstance(text, str) and text, fid
        if fid in HEAVY:                                 # nothing set up on the sample: said, and nothing written
            assert want == [] and ('Nothing' in text or 'No ' in text), (fid, text)
    # the notes say why, and the cheap ones carry their provenance
    note = (tmp_path / 'perf_model' / 'perf_model' / 'NOT_WRITTEN.txt').read_text()
    assert 'no saved model' in note
    doc = json.loads((tmp_path / 'risk_coverage' / 'risk_coverage' / 'risk_coverage.json').read_text())
    assert doc['schema'] == 'vantage.risk_coverage/1' and doc['provenance']['study'] == 'risk_coverage'
    tol = json.loads((tmp_path / 'fleet_twin_tolerances' / 'fleet_twin' / 'fleet_twin_tolerances.json').read_text())
    assert tol['schema'] == 'vantage.fleet_twin.tolerances/1' and tol['provenance']['output']['format'] == \
        'fleet_twin_tolerances'


def test_records_and_model_formats_with_data(rig, tmp_path):
    """With records in the store and a saved model: the anonymised records and the model's description, exactly the
    declared names; the records carry no project name or path."""
    p, eng, res = rig
    m, docs = _toy_model()
    S = PM.store_for(p)
    for d in docs[:5]:
        PM.add_record(d, S)
    for fid in ('perception_records', 'perf_model'):
        out = str(tmp_path / fid)
        want = sorted(ex.format_by_id(fid).filename(p, p.variant))
        got = _written(ex.export([fid], p, res, ex.ExportOptions(out_dir=out, engine=eng)), out)
        assert not any(x.endswith('NOT_WRITTEN.txt') for x in got), (fid, got)
        # the model's calibration figure is drawn when the model has cross-validation to draw (the rest always)
        assert set(want) <= set(got) and set(got) - set(want) <= {'perf_model/perf_model_calibration.png'}, (fid, got)
    lines = (tmp_path / 'perception_records' / 'perception_records' / 'perception_records.jsonl').read_text()
    rows = [json.loads(x) for x in lines.splitlines() if x.strip()]
    assert len(rows) == 5 and all(r['schema'] == PM.RECORD_SCHEMA for r in rows)
    assert 'Test car' not in lines and str(tmp_path) not in lines              # anonymised: no vehicle name, no path
    doc = json.loads((tmp_path / 'perf_model' / 'perf_model' / 'perf_model.json').read_text())
    assert doc['schema'] == PM.SCHEMA and doc['provenance']['study'] == 'perf_model'
    rel, text, _ = ex.preview('perception_records', p, res, ex.ExportOptions(engine=eng))
    assert '5 perception records' in text


# ======================================================================= batch
def _batch(*argv, project=None, tmp=None):
    from vantage import batch
    if project is not None:
        path = os.path.join(tmp, 'p.vantage')
        project.save(path)
        argv = (path,) + argv
    else:
        argv = ('--sample',) + argv
    return batch.run(batch.parser().parse_args(list(argv)))


def test_batch_risk_coverage_and_its_threshold(rig, tmp_path):
    p, _, _ = rig
    out = tmp_path / 'risk.json'
    assert _batch('--risk-coverage', str(out), '--risk-min', 'pedestrian=50', project=p, tmp=str(tmp_path)) == 0
    doc = json.loads(out.read_text())
    assert doc['schema'] == 'vantage.risk_coverage/1' and doc['provenance']['output']['option'] == '--risk-coverage'
    w = doc['classes']['pedestrian']['weighted']
    assert 0.5 < w <= 1.0
    assert _batch('--risk-coverage', str(out), '--risk-min', f'pedestrian={100 * w + 1:.1f}', project=p,
                  tmp=str(tmp_path)) == 3
    d = tmp_path / 'risk'
    assert _batch('--risk-coverage', str(d), '--risk-weight', 'ncap', project=p, tmp=str(tmp_path)) == 0
    assert (d / 'risk_coverage_weights.csv').is_file() and (d / 'risk_coverage_pedestrian.png').is_file()
    with pytest.raises(SystemExit):
        _batch('--risk-coverage', '-', '--risk-min', 'dog=10', project=p, tmp=str(tmp_path))


def test_batch_fleet_twin(rig, tmp_path):
    p, _, _ = rig
    out = tmp_path / 'twin'
    assert _batch('--fleet-twin', str(out), '--as-of', '2026-11-01', project=p, tmp=str(tmp_path)) == 0
    names = sorted(os.listdir(out))
    assert names == ['fleet_twin.csv', 'fleet_twin.json', 'fleet_twin.png', 'fleet_twin_schedule.csv',
                     'fleet_twin_tolerances.json']
    doc = json.loads((out / 'fleet_twin.json').read_text())
    assert doc['as_of'] == '2026-11-01' and doc['provenance']['output']['option'] == '--fleet-twin'
    f = tmp_path / 'online.json'
    f.write_text('{}')
    assert _batch('--fleet-twin-online', 'Nobody', str(f), project=p, tmp=str(tmp_path)) == 4   # not in the fleet
    with pytest.raises(SystemExit):
        _batch('--fleet-twin', str(out), '--as-of', 'yesterday', project=p, tmp=str(tmp_path))


def test_batch_learned_model_options(rig, tmp_path, capsys):
    p, _, _ = rig
    assert _batch('--perf-model', str(tmp_path / 'm'), project=p, tmp=str(tmp_path)) == 3        # no records
    assert (tmp_path / 'm' / 'perf_model.json').is_file()
    assert _batch('--perf-records-export', str(tmp_path / 'none.jsonl'), project=p, tmp=str(tmp_path)) == 2
    assert _batch('--suggest-simulations', '2', project=p, tmp=str(tmp_path)) == 3              # no model
    _, docs = _toy_model(save=False)
    src = tmp_path / 'in.jsonl'
    src.write_text(''.join(json.dumps(d) + '\n' for d in docs[:3]) + 'not json\n')
    assert _batch('--perf-records-import', str(src), project=p, tmp=str(tmp_path)) == 0
    assert len(PM.store_for(p).records()) == 3
    assert _batch('--perf-records-export', str(tmp_path / 'out.jsonl'), project=p, tmp=str(tmp_path)) == 0
    assert len((tmp_path / 'out.jsonl').read_text().splitlines()) == 3
    assert _batch('--perf-records-import', str(tmp_path / 'missing.jsonl'), project=p, tmp=str(tmp_path)) == 4
    _toy_model()
    capsys.readouterr()
    assert _batch('--suggest-simulations', '2', project=p, tmp=str(tmp_path)) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc['ok'] and doc['picks'] and doc['schema'] == 'vantage.perf_model.suggestions/1'


def test_batch_studies_say_what_is_missing(rig, tmp_path):
    p, _, _ = rig
    t = str(tmp_path)
    assert _batch('--rig-perception', str(tmp_path / 'rp'), project=p, tmp=t) == 4               # no recording
    assert _batch('--rig-perception', str(tmp_path / 'rp'), '--perception-recording', str(tmp_path / 'nope:A'),
                  project=p, tmp=t) == 4
    assert _batch('--foundation-range', str(tmp_path / 'fr'), project=p, tmp=t) == 2
    assert _batch('--robust-perception', str(tmp_path / 'rb'), project=p, tmp=t) == 2
    assert _batch('--redrive', str(tmp_path / 'no-drive'), str(tmp_path / 'rd'), project=p, tmp=t) == 4
    assert _batch('--design', str(tmp_path / 'ds'), project=p, tmp=t) == 2                       # no requirement
    assert _batch('--perception-scene-set', 'nonsense', str(tmp_path / 'ss'), project=p, tmp=t) == 2
    assert not (tmp_path / 'ss').exists()                                                       # CARLA never tried


def test_batch_robust_perception_on_a_synthetic_drive(rig, tmp_path):
    """No detector (BEV figures only) on test_robust_perception's synthetic drive; the threshold is about the
    detector's recall, so it has nothing to fail."""
    from test_robust_perception import make_recording
    p, _, _ = rig
    rec = make_recording(str(tmp_path / 'rec'))
    out = tmp_path / 'rb'
    assert _batch('--robust-perception', str(out), '--robust-recording', rec, '--robust-no-detect', '--robust-sobol',
                  '8', '--robust-samples', '2', '--robust-min-recall', '0.99', project=p, tmp=str(tmp_path)) == 0
    doc = json.loads((out / 'robust_perception.json').read_text())
    assert doc['schema'] == 'vantage.robust_perception/1' and doc['provenance']['output']['option'] == \
        '--robust-perception'
    assert rec not in json.dumps(doc)


def test_batch_read_requirements_sets_the_design_agents_rows(rig, tmp_path):
    p, _, _ = rig
    brief = tmp_path / 'brief.md'
    brief.write_text('# Robotaxi\n\nThe rig must see pedestrians at 80 m forward. It draws under 300 W.\n')
    out = tmp_path / 'extraction.json'
    saved = tmp_path / 'saved.vantage'
    assert _batch('--read-requirements', str(out), str(brief), '--save', str(saved), project=p, tmp=str(tmp_path)) == 0
    doc = json.loads(out.read_text())
    assert doc['schema'] == 'vantage.designer_extraction/1' and doc['provenance']['output']['option'] == \
        '--read-requirements'
    metrics = {r['item']['metric'] for r in doc['requirements']}
    assert {'perception.range', 'power.steady_w'} <= metrics
    assert all(r['source']['quote'] in brief.read_text() for r in doc['requirements'])
    from vantage.core import rig_designer as DSG
    from vantage.core.project import Project
    q = Project.load(str(saved))
    st = DSG.settings_of(q)
    assert st.items() and st.documents[0]['name'] == 'brief.md'
    assert _batch('--read-requirements', str(out), str(tmp_path / 'missing.pdf'), project=p, tmp=str(tmp_path)) == 4


def test_batch_help_and_mcp_run_batch_rules():
    from vantage import batch
    from vantage.mcp import batchrun as B
    text = batch.parser().format_help()
    for opt in ('--rig-perception', '--perception-scene-set', '--redrive', '--design', '--read-requirements',
                '--perf-model', '--perf-records-import', '--perf-records-export', '--suggest-simulations',
                '--foundation-range', '--robust-perception', '--risk-coverage', '--fleet-twin', '--fleet-twin-online',
                '--fleet-twin-watch', '--as-of'):
        assert opt in text, opt
    assert 'CARLA' in B.check(['--perception-scene-set', 'standard', 'out'])
    assert B.check(['--perf-save']) and B.check(['--fleet-twin-watch'])
    assert B.check(['--risk-coverage', 'risk.json']) == ''


# ======================================================================= assistant and MCP tools
def test_every_new_tool_returns_json_through_mcp(tmp_path, monkeypatch):
    from vantage.assistant import tools as T
    from vantage.mcp import service as SV
    from vantage.mcp.server import Router
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'cfg'))
    assert set(TOOLS) <= set(T.BY_NAME) and set(TOOLS) <= set(SV.BY_NAME)
    r = Router(None, 'file')
    r.call('open_project', {'sample': True})
    args = {'plan_perception_scene_set': {'set_id': 'quick'}, 'get_fleet_tolerances': {'horizon_days': 90},
            'read_requirements': {'text': 'Pedestrians at 60 m ahead, under 250 W.'},
            'design_rig': {'budget_s': 10}}
    for name in TOOLS:
        out = r.call(name, args.get(name, {}))
        doc = json.loads(out['content'][0]['text'])
        assert isinstance(doc, dict), name
        if name in ('design_rig',):                     # no accepted requirement in the sample: it says so
            assert 'requirement' in doc.get('error', '') + json.dumps(doc), doc
            continue
        assert not out['isError'], (name, doc)
    doc = json.loads(r.call('get_risk_coverage', {'cls': 'pedestrian'})['content'][0]['text'])
    assert set(doc['classes']) == {'pedestrian'} and doc['requirement_proposals']
    assert doc['requirement_proposals'][0]['item']['metric'] == 'risk_coverage.weighted_pct'
    doc = json.loads(r.call('read_requirements', {'text': 'Pedestrians at 60 m ahead.'})['content'][0]['text'])
    assert any(x['metric'] == 'perception.range' and x['target'] == 60 for x in doc['requirements'])
    doc = json.loads(r.call('predict_detection', {})['content'][0]['text'])
    assert doc['trained'] is False and 'reason' in doc


def test_prediction_tools_with_a_trained_model(rig):
    from vantage.assistant import r14_tools as R
    from vantage.assistant import tools as T
    p, eng, res = rig
    _toy_model()
    tc = T.ToolContext(p.snapshot(), res, eng.share())
    d = R.predict_detection(tc, cls='pedestrian')
    assert d['trained'] and d['classes']['pedestrian'] and all(0 <= v['recall'] <= 1 for v in
                                                               d['classes']['pedestrian'].values())
    s = R.suggest_simulations(tc, k=2)
    assert s['ok'] and 1 <= len(s['picks']) <= 2
    g = R.get_perf_model(tc)
    assert g['trained'] and g['calibration']
    # a page's last result is what the reading tools return (the window passes them in ToolContext.studies)
    tc.studies = {'rig_perception': {'rigs': ['A'], 'per_rig': {'A': {'car': {'recall': 0.5}}}, 'findings': []}}
    assert R.get_rig_perception(tc)['run'] is True


def test_claude_strict_tools_stay_within_the_limits():
    from vantage.assistant import claude as CL
    from vantage.assistant import tools as T
    strict = CL.strict_names()
    assert len(strict) <= CL.STRICT_MAX
    assert sum(CL.optional_count(T.input_schema(T.BY_NAME[n])) for n in strict) <= CL.STRICT_OPTIONAL_MAX
    assert len(CL.tool_defs()) == len(T.TOOLS)


# ======================================================================= the Issues list and the cross hooks
def test_issues_list_reads_the_fleet_twin(rig):
    from vantage.core import fleet_twin as FT
    from vantage.core import rules
    p, eng, res = rig
    q = p.snapshot()
    calls = []
    real = FT.issues
    try:
        FT.issues = lambda project, vg=None: calls.append(project) or [rules.Issue('serious', 'Fleet twin: test',
                                                                                    'V1 · F', [])]
        out = rules.collect_issues(q, res, eng.vehicle_geom(q))
    finally:
        FT.issues = real
    assert calls and any(i.message == 'Fleet twin: test' for i in out)


def test_optimiser_takes_the_learned_models_term(rig):
    from vantage.core import optimiser as OP
    p, eng, _ = rig
    st = OP.OptSettings(generations=1, population=6, tol_samples=0, under_tolerance=False, cell=0.25,
                        perf_weight=1.0)
    o = OP.Optimiser(eng, p, st)
    assert o.perf is None and 'off' in o.perf_note                   # no model saved
    m, _ = _toy_model()
    o = OP.Optimiser(eng, p, st)
    assert o.perf is not None and 'weighted 1' in o.perf_note
    c = o.evaluate(o.nominal())
    assert c.perf != 0.0 and np.isfinite(c.perf)
    # the term is the model's own: mean − κ·std of the candidate's features
    f = PM.describe_rig(p, p.variant, o._perf_grid, eng)
    assert c.perf == pytest.approx(m.objective_term(f), abs=1e-9)
    base = OP.Optimiser(eng, p, OP.OptSettings(generations=1, population=6, tol_samples=0, under_tolerance=False,
                                               cell=0.25))
    c0 = base.evaluate(base.nominal())
    assert o.objectives(c)[0] == pytest.approx(base.objectives(c0)[0] - c.perf, abs=1e-9)


def test_robust_mounting_from_the_fleet_twin_and_the_3d_metric(tmp_path, monkeypatch):
    from test_robust_perception import colour_detector, make_recording, zero_settings
    from vantage.core import fleet_twin as FT
    from vantage.core import robust_perception as RB
    rec = make_recording(str(tmp_path / 'rec'))
    tol = {'per_camera': {'F': {'yaw': 0.2, 'pitch': 0.4, 'roll': 0.1, 'pos_mm': 2.0}}, 'horizon_days': 180,
           'as_of': '2026-11-01'}
    monkeypatch.setattr(FT, 'cached_tolerance_inputs', lambda project, horizon_days=None: tol)
    p = samples.tigor_sample_project()
    src = RB.resolve_sources(p, zero_settings(rec, mount_from='fleet_twin'), ['F', 'L'])['mounting']
    assert src.kind == 'normal' and 'fleet_twin' in src.origin
    assert src.per_cam['F']['deg_axes'] == [0.2, 0.4, 0.1] and src.per_cam['F']['mm'] == 2.0
    assert src.per_cam['L']['deg'] == 0.0 and any('L: not followed' in n for n in src.notes)
    row = next(r for r in src.table() if r['camera'] == 'F')
    assert row['pitch_deg'] == 0.4
    # the 3D metric through rig_perception: with no tolerance every build is the nominal one
    res = RB.run(None, settings=zero_settings(rec, bev3d=True), detector=colour_detector)
    rr = res.recordings[0]
    ped = rr.bev3d['pedestrian']['all']
    assert ped['n_gt'] > 0 and ped['nominal'] == pytest.approx(1.0)
    assert ped['recall']['p05'] == ped['recall']['p95'] == ped['nominal']
    assert 0 < ped['nominal_ate_m'] < 2.0                   # matched within the 2 m threshold
    s = RB.summary(res)['recordings'][0]
    assert s['pedestrian_bev3d']['nominal_recall'] == pytest.approx(1.0)
    doc = RB.to_doc(res)
    assert doc['recordings'][0]['bev3d'] and 'lift_boxes' in doc['recordings'][0]['bev3d_method']
    assert RB.run(None, settings=zero_settings(rec), detector=colour_detector).recordings[0].bev3d == {}


def test_rig_perception_marks_a_reshot_drives_records(tmp_path):
    from vantage.core import rig_perception as RP
    (tmp_path / 'dataset.json').write_text(json.dumps({'generator': 'redrive'}))
    assert RP.record_source(str(tmp_path), 'carla') == 'redrive'
    (tmp_path / 'dataset.json').write_text(json.dumps({'generator': 'carla_bridge'}))
    assert RP.record_source(str(tmp_path), 'carla') == 'carla'
    assert RP.record_source(str(tmp_path / 'none'), 'nuscenes') == 'redrive'


def test_requirements_scorecard_has_risk_weighted_coverage(rig):
    from vantage.core import requirements as RQ
    from vantage.core import risk_coverage as RC
    p, eng, _ = rig
    assert RQ.metric_available('risk_coverage.weighted_pct')
    q = p.snapshot()
    rc = RC.run(q, eng, variant=q.variant)
    want = 100 * rc.classes['cyclist']['weighted']
    items = [{'metric': 'risk_coverage.weighted_pct', 'params': {'cls': 'cyclist'}, 'op': '>=', 'target': 50.0,
              'priority': 'must'},
             {'metric': 'risk_coverage.weighted_pct', 'params': {}, 'op': '>=', 'target': 99.9, 'priority': 'must'}]
    assert all(RQ.validate_item(it, q)[0] for it in items)
    q.analysis.requirements = {'items': items, 'settings': {}}
    res = RQ.run(q, eng)
    a, b = res.items
    assert a.status == 'pass' and a.value == pytest.approx(want, abs=0.05)
    worst = min(100 * d['weighted'] for d in rc.classes.values() if d['weighted'] is not None)
    assert b.status == 'fail' and b.value == pytest.approx(worst, abs=0.05)
    # the risk page's proposals are scorecard rows
    for prop in RC.requirement_proposal(rc):
        assert RQ.validate_item(prop['item'], q)[0]


# ======================================================================= the documentation
def test_docs_name_every_round14_setting_flag_and_section():
    """docs/project-format.md names every key of the new settings; README and docs.html name every new batch flag;
    every new docs.html section has its id and its contents link; the public pages name no machine path."""
    import re
    from vantage import batch
    from vantage.core import fleet_twin as FT
    from vantage.core import foundation_range as FR
    from vantage.core import redrive as RDV
    from vantage.core import rig_designer as DSG
    from vantage.core import rig_perception as RP
    from vantage.core import risk_coverage as RC
    from vantage.core import robust_perception as RB
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fmt = open(os.path.join(root, 'docs', 'project-format.md'), encoding='utf-8').read()
    named = set(re.findall(r'`([A-Za-z_][\w]*)`', fmt))
    for st in (RP.RigPerceptionSettings(), RDV.RedriveSettings(), DSG.DesignerSettings(), PM.PerfModelSettings(),
               FR.FoundationSettings(), RB.RobustSettings(), RC.RiskSettings(), FT.FleetTwinSettings()):
        assert not [k for k in st.to_dict() if k not in named], type(st).__name__
    readme = open(os.path.join(root, 'README.md'), encoding='utf-8').read()
    docs = open(os.path.join(root, 'site', 'docs.html'), encoding='utf-8').read()
    dests = set(batch.ROUND14_OPTIONS) | {d for mods, _ in batch._GOES_WITH.items() for d in mods
                                         if any(m in batch.ROUND14_OPTIONS for m in _)}
    flags = [a.option_strings[-1] for a in batch.parser()._actions if a.option_strings and a.dest in dests]
    assert len(flags) >= 40
    for f in flags:
        assert f in readme and f in docs, f
    for sec in ('rig-perception', 'redrive', 'foundation', 'robust-perception', 'risk-coverage', 'designer',
                'perf-model', 'fleet-twin'):
        assert f'id="{sec}"' in docs and f'href="#{sec}"' in docs, sec
    home = os.path.expanduser('~')
    for page in ('README.md', 'CHANGELOG.md', 'docs/overview.md', 'docs/user-guide.md', 'docs/project-format.md',
                 'site/docs.html', 'site/guide.html', 'site/index.html'):
        text = open(os.path.join(root, page), encoding='utf-8').read()
        assert home not in text, page
