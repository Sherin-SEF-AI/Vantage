"""Round 17's wiring: one table per surface (ui/pages17, assistant/page_guide17, assistant/r17_tools, mcp/r17,
batch17, core/exporters17, rules.ISSUES17) and the hooks into the power budget, the harness, the 3D view, the
requirements scorecard and perception. The studies themselves are tested in their own files."""
from __future__ import annotations

import json
import os
import re

import pytest

from vantage.core import samples
from vantage.core.analysis import Engine

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope='module')
def rig():
    """The sample's data rig (cameras, a roof lidar, radars) with a cleaning layout proposed."""
    from vantage.core import cleaning as CL
    p = samples.tigor_sample_project()
    p.active_uid = next(v for v in p.variants if v.sensors).uid
    p.analysis.cleaning = {'devices': CL.propose_layout(p)}
    CL.clear_cache()
    return p


def _batch(*argv, project, tmp):
    from vantage import batch
    path = os.path.join(tmp, 'p.vantage')
    project.save(path)
    return batch.run(batch.parser().parse_args([path] + list(argv)))


# ======================================================================= the tables
def test_pages_table_guides_and_assistant_labels():
    import importlib
    from vantage.assistant import page_guide as PG
    from vantage.ui import mainwindow as MW
    from vantage.ui.assistant import PAGES as LABELS
    from vantage.ui.icon_data import ICONS
    from vantage.ui.pages17 import PAGES
    keys = [p['key'] for p in PAGES]
    assert len(keys) == len(set(keys))
    for pg in PAGES:
        assert MW.PAGE_CLASSES[pg['key']] == (pg['module'], pg['cls'])
        mod = importlib.import_module(f'vantage.ui.workspaces.{pg["module"]}')
        assert hasattr(mod, pg['cls']), pg['key']
        assert pg['icon'] in ICONS and pg['menu'] in ('File', 'Analysis', 'Calibrate', 'Simulate', 'Embedded'), pg
        assert pg['tip'] and pg['text'].endswith('…')
        g = PG.GUIDES[pg['key']]
        assert g.label == pg['label'] == LABELS[pg['key']]
        if pg.get('core'):
            core = importlib.import_module(f'vantage.core.{pg["core"]}')
            assert callable(core.summary) and g.module == pg['core'], pg['key']


def test_tools_are_offered_to_the_assistant_and_over_mcp():
    from vantage.assistant import tools as T
    from vantage.assistant.r17_tools import TOOLS_R17
    from vantage.mcp import service as SV
    for t in TOOLS_R17:
        assert T.BY_NAME[t.name] is t and t.name in SV.BY_NAME and SV.BY_NAME[t.name].read_only
    for name in ('add_thermal_camera', 'add_frame_to_rig'):
        assert not SV.BY_NAME[name].read_only
    for name in ('diff_projects', 'lab_versions', 'lab_compare', 'lab_pooled_twin'):
        assert SV.BY_NAME[name].read_only


def test_read_tools_on_the_data_rig(rig):
    from vantage.assistant import tools as T
    eng = Engine()
    tc = T.ToolContext(rig.snapshot(), None, eng.share())
    th = T.BY_NAME['get_thermal_night'].fn(tc, condition='fog', **{'class': 'pedestrian'})
    assert th['asked'] == {'condition': 'fog', 'class': 'pedestrian', 'level': 'detection'} and th['ranges']
    so = T.BY_NAME['get_sotif'].fn(tc, target_kmh=50)
    assert so['target_kmh'] == 50 and so['conditions'] and 'not a certification' in so['disclaimer']
    cl = T.BY_NAME['get_cleaning'].fn(tc, ambient_c=-10)
    assert cl['devices'] > 0 and cl['tank_runtime_h'] > 0
    fr = T.BY_NAME['get_frame'].fn(tc)
    assert 'members' in fr and 'cut_list' in fr
    ae = T.BY_NAME['get_aerodynamics'].fn(tc)
    assert ae['speeds'] and 'not CFD' in ae['meaning']
    ev = T.BY_NAME['get_event_cameras'].fn(tc)
    assert ev['fast'] is True and 'error' not in ev
    fl = T.BY_NAME['get_fmcw_lidar'].fn(tc, scene='highway', ego_speed_kmh=100)
    assert fl['lidars'] and fl['source']
    ir = T.BY_NAME['get_imaging_radar'].fn(tc, radar=next(s.tag for s in rig.variant.sensors if s.kind == 'radar'))
    assert ir['radars'] and ir['radar']['advice'] is not None
    json.dumps(ir, allow_nan=False)                      # plain JSON: no infinite sidelobe levels
    nr = T.BY_NAME['get_neural_replay'].fn(tc)
    assert nr['settings']['projection'] == 'ut' and 'smpl_path' not in nr['settings'] and 'pose_model' in nr
    eg = T.BY_NAME['get_ego_composite'].fn(tc)
    assert eg['result'] is None and 'how' in eg
    ec = T.BY_NAME['get_edge_config'].fn(tc)
    assert ec and 'error' not in ec
    tf = T.BY_NAME['get_trigger_firmware'].fn(tc)
    assert 'lines' in tf and 'targets' in tf
    ti = T.BY_NAME['get_thermal_intrinsics'].fn(tc, camera='F', temperature_c=40)
    assert ti['camera']['tag'] == 'F' and 'note' in ti['camera']                 # no sweep on the sample
    fm = T.BY_NAME['get_frame_modes'].fn(tc)
    assert fm.get('how') or fm.get('modes_hz') is not None
    assert T.BY_NAME['get_sotif'].fn(tc, variant='nope')['error'].startswith('no variant')
    text = json.dumps([th, so, cl, fr, ae, ev, fl, ir, nr, eg, ec, tf, ti, fm], default=str)
    assert os.path.expanduser('~') not in text


def test_mcp_add_thermal_camera_is_one_undo_step(tmp_path):
    from vantage.mcp import service as SV
    from vantage.mcp.session import FileSession
    p = samples.tigor_sample_project()
    path = str(tmp_path / 'p.vantage')
    p.save(path)
    s = FileSession()
    assert 'error' not in s.open(path=path)
    n = len(s.project().variant.cameras)
    out = SV.call(s, 'add_thermal_camera', {})
    assert not out['isError'], out
    doc = out['structuredContent']
    assert doc['camera']['module_id'] == 'flir_bosonplus_640' and len(s.project().variant.cameras) == n + 1
    bad = SV.call(s, 'add_thermal_camera', {'module_id': 'sturdecam31'})['structuredContent']
    assert 'not a thermal' in bad['error'] and len(s.project().variant.cameras) == n + 1
    s.undo()
    assert len(s.project().variant.cameras) == n


def test_mcp_never_accepts_a_licence_for_the_user(tmp_path):
    from vantage.mcp import service as SV
    from vantage.mcp.session import FileSession
    p = samples.tigor_sample_project()
    path = str(tmp_path / 'p.vantage')
    p.save(path)
    s = FileSession()
    s.open(path=path)
    bad = SV.call(s, 'edit_project', {'ops': [{'op': 'set', 'target': 'analysis',
                                                'props': {'neural_replay': {'smpl_accept': True}}}]})
    assert 'only the user sets' in json.dumps(bad) and not s.project().analysis.neural_replay.get('smpl_accept')
    ok = SV.call(s, 'edit_project', {'ops': [{'op': 'set', 'target': 'analysis',
                                               'props': {'neural_replay': {'walkers': 'rigid'}}}]})
    assert not ok['isError'] and s.project().analysis.neural_replay['walkers'] == 'rigid'


def test_mcp_diff_projects(tmp_path):
    from vantage.mcp import service as SV
    from vantage.mcp.session import FileSession
    p = samples.tigor_sample_project()
    a, b = str(tmp_path / 'a.vantage'), str(tmp_path / 'b.vantage.d')
    p.save(a)
    p.variant.cameras[0].x += 0.012
    p.save(b)
    s = FileSession()
    s.open(path=a)
    doc = SV.call(s, 'diff_projects', {'a': a, 'b': b})['structuredContent']
    assert doc['same'] is False and doc['text']
    assert 'error' in SV.call(s, 'diff_projects', {'a': str(tmp_path / 'x.txt')})['structuredContent']


def test_mcp_run_batch_keeps_the_lab_the_users():
    from vantage.mcp import batchrun as B
    for opt in ('--lab-publish', '--lab-sync', '--lab-pull', '--lab-pooled-twin'):
        assert B.check([opt, 'x', 'y']), opt
    for opt in ('--thermal', '--sotif', '--cleaning', '--framing', '--fea', '--aero', '--diff'):
        assert B.check([opt, 'out']) == '', opt


# ======================================================================= batch
def test_batch_options_and_their_modifiers():
    from vantage import batch
    text = batch.parser().format_help()
    for opt in ('--thermal', '--sotif', '--sotif-speed', '--cleaning', '--cleaning-propose', '--framing', '--fea',
                '--fea-mesh', '--aero', '--diff', '--diff-json', '--lab-publish', '--lab-pull', '--lab-sync',
                '--lab-pooled-twin', '--trigger-firmware', '--edge-config', '--edge-dry-run', '--edge-faults',
                '--thermal-intrinsics', '--imaging-radar', '--imaging-radar-as', '--usd-export', '--usd-splats',
                '--ego-composite', '--ego-out', '--ego-check', '--event-cameras', '--event-drive', '--neural-replay',
                '--neural-projection', '--neural-relight'):
        assert opt in text, opt
    args = batch.parser().parse_args(['--sample', '--sotif-speed', '80'])
    with pytest.raises(SystemExit, match='--sotif-speed goes with --sotif'):
        batch._modifiers(args)


def test_batch_studies_write_their_files(rig, tmp_path):
    t = str(tmp_path)
    assert _batch('--thermal', str(tmp_path / 'th'), project=rig, tmp=t) in (0, 2)
    assert {'thermal.json', 'thermal_ranges.csv'} <= set(os.listdir(tmp_path / 'th'))
    code = _batch('--sotif', str(tmp_path / 'so.json'), '--sotif-speed', '30', project=rig, tmp=t)
    doc = json.loads((tmp_path / 'so.json').read_text())
    assert doc['schema'] == 'vantage.sotif/1' and doc['settings']['target_kmh'] == 30
    assert code == (0 if doc['odd']['meets_target'] else 3)
    assert _batch('--cleaning', str(tmp_path / 'cl'), project=rig, tmp=t) in (0, 3)
    assert 'cleaning.json' in os.listdir(tmp_path / 'cl')
    assert _batch('--aero', str(tmp_path / 'ae'), project=rig, tmp=t) in (0, 2, 3)
    assert _batch('--fmcw', str(tmp_path / 'fm'), project=rig, tmp=t) in (0, 3)
    assert 'fmcw.json' in os.listdir(tmp_path / 'fm')
    assert _batch('--fmcw', str(tmp_path / 'fm2'), '--fmcw-recording', str(tmp_path / 'missing'), project=rig,
                  tmp=t) == 2
    assert _batch('--imaging-radar', str(tmp_path / 'ir'), project=rig, tmp=t) in (0, 2)
    assert 'imaging_radar.json' in os.listdir(tmp_path / 'ir')
    assert _batch('--event-drive', str(tmp_path / 'none'), project=rig, tmp=t) == 2
    assert _batch('--neural-replay', str(tmp_path / 'nr'), project=rig, tmp=t) == 4          # no log-replay drive
    assert _batch('--usd-export', str(tmp_path / 'usd'), project=rig, tmp=t) == 0
    assert 'vantage_rig.usda' in os.listdir(tmp_path / 'usd')
    assert _batch('--ego-composite', str(tmp_path / 'no-drive'), project=rig, tmp=t) == 4
    assert _batch('--edge-config', str(tmp_path / 'ed'), project=rig, tmp=t) == 0
    assert {'edge.json', 'vantage-edge.service'} <= set(os.listdir(tmp_path / 'ed'))
    assert _batch('--thermal-intrinsics', str(tmp_path / 'ti'), project=rig, tmp=t) == 2     # no sweeps
    q = samples.tigor_sample_project()                      # no rig parts, no sensor: nothing to frame
    assert _batch('--framing', str(tmp_path / 'fr'), project=q, tmp=t) == 2


def test_batch_diff(tmp_path, capsys):
    p = samples.tigor_sample_project()
    other = str(tmp_path / 'o.vantage')
    p.save(other)
    assert _batch('--diff', other, project=p, tmp=str(tmp_path)) == 0
    p.variant.cameras[0].yaw += 2.0
    p.save(other)
    q = samples.tigor_sample_project()
    out = str(tmp_path / 'd.json')
    assert _batch('--diff', other, '--diff-json', out, project=q, tmp=str(tmp_path)) == 1
    assert json.loads(open(out).read())['same'] is False


# ======================================================================= exports, issues and hooks
def test_round17_export_formats(rig, tmp_path):
    from vantage.core import exporters as ex
    from vantage.core import exporters17 as X17
    ids = [f.id for f in ex.FORMATS if f.id in X17.FORMATS17]
    assert ids == list(X17.FORMATS17) and all(f.group in ex.GROUPS for f in ex.FORMATS if f.id in ids)
    want = ['thermal', 'sotif', 'cleaning', 'aero']
    paths = ex.export(want, rig, None, ex.ExportOptions(out_dir=str(tmp_path)))
    names = {os.path.relpath(x, tmp_path).replace(os.sep, '/') for x in paths}
    for n in ('thermal/thermal.json', 'sotif/sotif.json', 'sotif/sotif_matrix.csv', 'cleaning/cleaning.json',
              'aero/aero.json'):
        assert any(x.endswith(n) for x in names), n
    path0, text, size = ex.preview('sotif', rig, None, ex.ExportOptions(out_dir=str(tmp_path / 'pv')))
    assert 'ODD speed limit' in text and size > 0
    q = samples.tigor_sample_project()
    q.variant.cameras = []
    assert ex.export(['thermal'], q, None, ex.ExportOptions(out_dir=str(tmp_path / 'none'))) == []


def test_issues_hook_survives_a_broken_study(rig, monkeypatch):
    from vantage.core import rules
    monkeypatch.setattr(rules, 'ISSUES17', ('thermal_ir', 'no_such_study'))
    assert isinstance(rules._round17_issues(rig, None), list)


def test_power_and_harness_count_the_cleaning(rig):
    from vantage.core import cleaning as CL
    from vantage.core import harness as HN
    from vantage.core import power as PW
    res = CL.run(rig)
    q = rig.snapshot()
    q.analysis.cleaning = {}
    with_c, without = PW.run(rig), PW.run(q)
    assert [r for r in with_c.rows if r['source'] == 'cleaning']
    assert with_c.totals['steady_w'] == pytest.approx(without.totals['steady_w'] + res.totals['average_w'])
    assert CL.power_budget(rig, res).totals['steady_w'] == pytest.approx(with_c.totals['steady_w'])   # not twice
    h = HN.run(rig)
    assert len(h.cleaning) == len(res.harness) > 0
    assert 'cleaning_runs' in HN.to_doc(h, rig) and 'cleaning:0' in HN.to_csv(h)


def test_scorecard_metrics_and_overlay():
    from vantage.core import requirements as RQ
    from vantage.ui.context import OVERLAYS
    assert {'sotif.odd_speed_kmh', 'sotif.insufficient_cells'} <= set(RQ.METRICS)
    assert RQ.METRICS['sotif.odd_speed_kmh'].better == '>=' and RQ.STUDIES['sotif'].page == 'sotif'
    assert 'cleaning' in [o[0] for o in OVERLAYS]


def test_perception_uses_thermal_radiometry_at_night():
    """A thermal camera in fog at night is not judged by the visible light test (core.perception)."""
    from vantage.core import perception as PR
    from vantage.core import thermal_ir as TH
    p = samples.tigor_sample_project()
    eng = Engine()
    TH.add_thermal_camera(p, engine=eng)
    res = PR.analyse(p, eng, weather='fog')
    tags = [c['tag'] for c in res.weather['cameras']]
    assert 'T' not in tags and tags                 # no visible-light floor for the thermal camera


# ======================================================================= docs and validation
# each round-17 settings slot: (core module, its settings class)
SETTINGS17 = (('thermal_ir', 'ThermalSettings'), ('event_camera', 'EventSettings'), ('fmcw_lidar', 'FmcwSettings'),
              ('imaging_radar', 'ImagingRadarSettings'), ('sotif', 'SotifSettings'),
              ('cleaning', 'CleaningSettings'), ('framing', 'FramingSettings'), ('fea', 'FEASettings'),
              ('aero', 'AeroSettings'), ('trigger_firmware', 'FirmwareSettings'), ('edge_daemon', 'EdgeSettings'),
              ('thermal_intrinsics', 'ThermalSettings'), ('ego_composite', 'EgoSettings'),
              ('usd_export', 'UsdSettings'), ('neural_replay', 'NeuralSettings'), ('relight', 'RelightSettings'))
CHANGELOG17 = ('Thermal', 'Event cameras', 'FMCW lidar', 'Imaging radar', 'Your own vehicle in CARLA', 'OpenUSD',
               'Trigger-box firmware', 'Edge watchdog', 'Thermal intrinsics', 'SOTIF', 'Cleaning', 'Frame modes',
               'Aerodynamics', 'lab server', 'exact lens projection (3DGUT)', 'Articulated pedestrians')


def test_docs_name_every_round17_flag_setting_and_tool():
    import importlib
    from vantage import batch
    from vantage.assistant.r17_tools import TOOLS_R17
    from vantage.mcp import r17 as M17
    fmt = open(os.path.join(ROOT, 'docs', 'project-format.md'), encoding='utf-8').read()
    named = set(re.findall(r'`([A-Za-z_][\w]*)`', fmt))
    for mod, cls in SETTINGS17:
        st = getattr(importlib.import_module(f'vantage.core.{mod}'), cls)()
        assert not [k for k in st.to_dict() if k not in named], (mod, cls)
    cli = open(os.path.join(ROOT, 'docs', 'command-line.md'), encoding='utf-8').read()
    dests = set(batch.ROUND17_OPTIONS) | {d for mods, mains in batch._GOES_WITH.items() for d in mods
                                         if any(m in batch.ROUND17_OPTIONS for m in mains)}
    flags = [a.option_strings[-1] for a in batch.parser()._actions if a.option_strings and a.dest in dests]
    assert len(flags) >= 30
    for f in flags:
        assert f in cli, f
    names = [t.name for t in TOOLS_R17] + [t.name for t in M17.tools(lambda *a, **k: type('T', (), {'name': a[0]}))]
    for n in names:
        assert f'`{n}`' in cli, n
    log = open(os.path.join(ROOT, 'CHANGELOG.md'), encoding='utf-8').read()
    top = log.index('\n## ') + 1                                   # the newest entry (0.10.0 released round 17)
    newest = log[top:log.index('\n## ', top + 3)]
    for word in CHANGELOG17:
        assert word in newest, word


def test_packaging_finds_every_round17_page():
    import sys
    sys.path.insert(0, os.path.join(ROOT, 'packaging'))
    try:
        import release
    finally:
        sys.path.pop(0)
    from vantage.ui.pages17 import PAGES
    mods = release.page_modules(ROOT)
    assert all(f'vantage.ui.workspaces.{p["module"]}' in mods for p in PAGES)
    assert any(dst.replace(os.sep, '/').endswith('firmware/rust/firmware/.cargo')
               for _, dst in release.data_files(ROOT))


def test_validation_registers_the_round17_models():
    from vantage.validation import suite
    ids = {s.id for s in suite.specs()}
    for prefix in ('trigger_fw.', 'edge.', 'thermal_intrinsics.', 'fmcw.', 'thermal.', 'sotif.', 'cleaning.',
                   'fea.', 'aero.', 'imaging_radar.', 'event.', 'neural_replay.', 'relight.', 'ego_composite.',
                   'usd.', 'formats.project_dir'):
        assert any(i.startswith(prefix) for i in ids), prefix
