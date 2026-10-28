"""The rig studies wired into the rest of the app: each page opens from its menu, each export format writes its files,
batch mode runs each flag with its exit code, and the assistant reads each study (the studies themselves are tested
in their own files: test_power, test_gnss, test_exposure, test_brackets, test_fleet, test_dataset, test_drivelog,
test_lidar_calib, test_requirements, test_critical, test_materials, test_weather, test_automation, test_terrain,
test_harness, test_observability, test_experiments, test_validation, test_provenance, test_sim)."""
import json
import os
import sys
import time

import numpy as np
import pytest

from vantage.core import samples, sensors
from vantage.core.project import Sensor

# action id -> (page name, menu)
PAGES = {'power_budget': ('power', 'Analysis'), 'gnss_imu': ('gnss', 'Analysis'), 'sun_dirt': ('exposure', 'Analysis'),
         'mount_brackets': ('brackets', 'Analysis'), 'requirements': ('requirements', 'Analysis'),
         'sample_dataset': ('dataset', 'Export'), 'drive_log': ('drivelog', 'Calibrate'),
         'fleet': ('fleet', 'Calibrate'), 'lidar_calib': ('lidar_calib', 'Calibrate'),
         'asbuilt': ('asbuilt', 'Calibrate'), 'critical_targets': ('critical', 'Analysis'),
         'materials': ('materials', 'Analysis'), 'weather': ('perception', 'Analysis'),
         'automation': ('automation', 'Export'), 'terrain': ('terrain', 'Analysis'),
         'cable_harness': ('harness', 'Analysis'), 'observability': ('observability', 'Calibrate'),
         'experiments': ('experiments', 'Analysis'), 'carla': ('sim', 'Simulate'),
         'carla_live': ('sim', 'Simulate'), 'detection': ('detection', 'Simulate'), 'campaign': ('campaign', 'Simulate'),
         'critical_carla': ('critical_carla', 'Simulate')}
FORMATS = {'power_budget': ['power_budget/power_budget.json', 'power_budget/power_budget.csv',
                            'power_budget/power_circuits.csv'],
           'gnss_imu': ['gnss_imu/gnss_imu.json', 'gnss_imu/ins_lever_arms.txt', 'gnss_imu/gnss_sky.png'],
           'exposure': ['exposure/exposure.json', 'exposure/exposure.png'],
           'fleet': ['fleet/fleet.json', 'fleet/fleet.csv', 'fleet/fleet.png'],
           'brackets': ['brackets/brackets.json', 'brackets/brackets_sheet.png'],
           'requirements': ['requirements/requirements.json', 'requirements/requirements.csv',
                            'requirements/review_pack.pdf'],
           'critical_targets': ['critical_targets/critical_targets.json', 'critical_targets/critical_targets.csv',
                                'critical_targets/mounting_envelopes.csv', 'critical_targets/mounting_envelopes.png',
                                'critical_targets/critical_targets.png'],
           'materials': ['materials/materials.json', 'materials/materials_radars.csv', 'materials/materials_glass.csv',
                         'materials/materials.png'],
           'weather': ['weather/weather.json', 'weather/weather_conditions.csv', 'weather/weather_sensors.csv',
                       'weather/weather.png'],
           'ci_templates': ['ci/github/vantage.yml', 'ci/gitlab/vantage.gitlab-ci.yml', 'ci/README.md'],
           'terrain': ['terrain/terrain.json', 'terrain/terrain_motion.csv'],
           'harness': ['harness/harness.json', 'harness/harness.csv', 'harness/harness.png'],
           'observability': ['observability/observability.json', 'observability/observability.csv',
                             'observability/observability.png', 'observability/fusion_density.png'],
           'sim_rig': ['carla/sim_rig.json', 'carla/sim_remap.npz']}


def _rig():
    """The sample's data rig with two GNSS antennas and a dual-antenna INS where Vehicle ▸ Add puts them."""
    p = samples.tigor_sample_project()
    v = next(x for x in p.variants if x.sensors)
    p.active_uid = v.uid
    for kind in ('gnss', 'gnss', 'imu'):
        m = p.sensor_models[sensors.default_model_id(kind)]
        v.sensors.append(Sensor(kind=kind, model_id=m.id, tag=sensors.next_tag(kind, v.tags()), name=m.name,
                                **sensors.default_pose(kind, m, p.vehicle.mesh, p.rig, v.sensors)))
    return p


def test_new_export_formats_write_their_files(tmp_path):
    from vantage.core import exporters as ex
    p = _rig()
    for fid, want in FORMATS.items():
        f = ex.format_by_id(fid)
        names = ex._names(f, p, p.variant)
        assert set(want) <= set(names), fid
        out = tmp_path / fid
        paths = ex.export([fid], p, None, ex.ExportOptions(out_dir=str(out)))
        got = {os.path.relpath(x, out).replace(os.sep, '/') for x in paths}
        assert set(want) <= got and got <= set(names), fid
        assert all(os.path.getsize(x) > 0 for x in paths)
    doc = json.loads((tmp_path / 'gnss_imu' / 'gnss_imu' / 'gnss_imu.json').read_text())
    assert doc['schema'] == 'vantage.gnss_imu/1' and [a['tag'] for a in doc['antennas']] == ['GN1', 'GN2']
    assert (tmp_path / 'requirements' / 'requirements' / 'review_pack.pdf').read_bytes()[:4] == b'%PDF'
    for fid, schema in (('critical_targets', 'vantage.critical_targets/1'), ('materials', 'vantage.materials/1'),
                        ('weather', 'vantage.weather/1'), ('terrain', 'vantage.terrain/1'),
                        ('harness', 'vantage.harness/1'), ('observability', 'vantage.observability/1')):
        doc = json.loads((tmp_path / fid / fid / f'{fid}.json').read_text())
        assert doc['schema'] == schema and doc['variant'] == p.variant.name, fid
    mdoc = json.loads((tmp_path / 'materials' / 'materials' / 'materials.json').read_text())
    radars = {r['tag'] for r in mdoc['radars']}
    assert radars == {s.tag for s in p.variant.sensors if s.kind == 'radar'}
    import yaml
    wf = yaml.safe_load((tmp_path / 'ci_templates' / 'ci' / 'github' / 'vantage.yml').read_text())
    assert 'jobs' in wf and (tmp_path / 'ci_templates' / 'ci' / 'README.md').stat().st_size > 0


def test_report_of_a_rig_with_sensors_and_an_analysis(tmp_path):
    """The report lists the lidars, radars, GNSS antennas and IMU and still reads the analysis' metrics after them
    (the sensor table once overwrote the metrics, so a rig's report failed to export)."""
    from vantage.core import exporters as ex
    from vantage.core.analysis import Engine
    p = _rig()
    eng = Engine()
    res = eng.analyse(p)
    paths = ex.export(['report_html', 'report_pdf'], p, res, ex.ExportOptions(out_dir=str(tmp_path), engine=eng))
    html = (tmp_path / 'report.html').read_text('utf-8')
    assert 'Lidars, radars, GNSS and IMU' in html and 'IMU1' in html and 'Pixel density' in html
    assert (tmp_path / 'report.pdf').read_bytes()[:4] == b'%PDF' and len(paths) == 2


def _batch(tmp_path, p, **kw):
    from vantage import batch
    from test_advanced import _args
    src = tmp_path / 'p.vantage'
    p.save(str(src))
    a = _args(**kw)
    a.sample, a.path = False, str(src)
    return batch.run(a)


def test_batch_flags_and_exit_codes(tmp_path):
    from vantage import batch
    from test_advanced import _args
    p = _rig()
    for flag in ('power_budget', 'gnss', 'exposure', 'check', 'fleet'):
        out = tmp_path / f'{flag}.json'
        assert _batch(tmp_path, p, **{flag: str(out)}) == 0, flag
        assert json.loads(out.read_text())['schema'].startswith('vantage.'), flag
        assert batch.wanted(_args(**{flag: '-'}))
    # a converter too small for the rig is critical: exit 2
    p.analysis.power = {'dcdc_w': 50}
    assert _batch(tmp_path, p, power_budget=str(tmp_path / 'pw.json')) == 2
    p.analysis.power = {}
    # an antenna in the cabin has no sky: exit 2
    g = next(s for s in p.variant.sensors if s.kind == 'gnss')
    z0, g.z = g.z, 1.0
    assert _batch(tmp_path, p, gnss=str(tmp_path / 'g.json')) == 2
    g.z = z0
    # a must requirement that fails: exit 5, and the review pack in a folder
    from vantage.core import requirements as RQ
    item = RQ.clean_item({'metric': 'timing.misalignment_ms', 'op': '<=', 'target': 0.001, 'priority': 'must',
                          'text': 'Cameras within 1 µs of the lidar'}, p)
    assert item is not None and RQ.validate_item(item, p)[0]
    p.analysis.requirements = {'items': [item]}
    assert _batch(tmp_path, p, check=str(tmp_path / 'review')) == 5
    assert (tmp_path / 'review' / 'review_pack.pdf').read_bytes()[:4] == b'%PDF'
    # a drive log that is not there cannot be read: exit 4
    p.analysis.requirements = {}
    assert _batch(tmp_path, p, drive_log=[str(tmp_path / 'no-such-bag')]) == 4
    # nor an as-built check's recording
    assert _batch(tmp_path, p, as_built_check=[str(tmp_path / 'no-such-recording')]) == 4


def _with_targets(p, *labels):
    from vantage.core import critical as CR
    st = CR.settings_of(p)
    for lab in labels:
        st.targets.append(CR.preset(p, lab, st.targets))
    p.analysis.critical = st.to_dict()
    return p


def test_batch_flags_of_round_six(tmp_path, capsys):
    """--observability, --harness, --terrain, --critical-targets, --materials, --weather (with --condition),
    --ci-templates and --watch: their files and exit codes."""
    from vantage import batch
    from test_advanced import _args
    for flag in ('observability', 'verify', 'harness', 'terrain', 'critical_targets', 'materials', 'weather',
                 'condition', 'watch', 'on_change', 'ci_templates'):
        assert hasattr(_args(), flag), flag
    for flag in ('observability', 'harness', 'terrain', 'critical_targets', 'materials', 'weather', 'ci_templates',
                 'watch'):
        assert batch.wanted(_args(**{flag: 'x'})), flag
    assert not batch.wanted(_args(condition='fog'))          # a modifier, like --variant
    p = _with_targets(_rig(), 'Child 2 m ahead', 'Car 30 m ahead')
    # every target resolved: exit 0; the JSON, or the files in a folder
    out = tmp_path / 'critical.json'
    assert _batch(tmp_path, p, critical_targets=str(out)) == 0
    doc = json.loads(out.read_text())
    assert doc['schema'] == 'vantage.critical_targets/1' and len(doc['targets']) == 2
    assert _batch(tmp_path, p, critical_targets=str(tmp_path / 'env')) == 0
    assert (tmp_path / 'env' / 'mounting_envelopes.csv').is_file()
    # a target no camera can resolve (a sign 400 m ahead): exit 2
    from vantage.core import critical as CR
    st = CR.settings_of(p)
    far = CR.preset(p, 'sign', st.targets)
    far['x'] = 400.0
    st.targets.append(far)
    p.analysis.critical = st.to_dict()
    assert _batch(tmp_path, p, critical_targets=str(out)) == 2
    assert 'NOT RESOLVED' in capsys.readouterr().err
    # calibration observability: every camera sees a board (exit 0; a folder gets the files); with only the board
    # behind the car, the front camera sees none (exit 2)
    q = _rig()
    assert _batch(tmp_path, q, observability=str(tmp_path / 'obs')) == 0
    assert json.loads((tmp_path / 'obs' / 'observability.json').read_text())['schema'] == 'vantage.observability/1'
    boards = list(q.targets)
    q.targets = [t for t in boards if t.name == 'T8']
    out = tmp_path / 'obs.json'
    assert _batch(tmp_path, q, observability=str(out)) == 2
    assert {c['tag']: c['verdict'] for c in json.loads(out.read_text())['cameras']}['F'] == 'critical'
    q.targets = boards
    with pytest.raises(SystemExit):
        batch.run(_args(verify=3))                            # --verify needs --observability
    # the cable harness: every run within its link (exit 0); a compute box 20 m behind the car puts the T1 and coax
    # runs over their limits (exit 2)
    out = tmp_path / 'harness.json'
    assert _batch(tmp_path, q, harness=str(out)) == 0
    doc = json.loads(out.read_text())
    assert doc['schema'] == 'vantage.harness/1' and {'F', 'L1', 'RD1'} <= {r['device'] for r in doc['runs']}
    x0, q.ecu.x = q.ecu.x, -20.0
    assert _batch(tmp_path, q, harness=str(out)) == 2
    q.ecu.x = x0
    # the road: flat ground (body motion only) is fine; a 30 cm hump hits the underbody driving over it (exit 2)
    out = tmp_path / 'road.json'
    assert _batch(tmp_path, q, terrain=str(out)) == 0
    doc = json.loads(out.read_text())
    assert doc['schema'] == 'vantage.terrain/1' and doc['variant'] == q.variant.name
    q.analysis.terrain = {'features': [{'kind': 'bump', 'height': 0.30, 'length': 0.6, 'start': 1.0}]}
    assert _batch(tmp_path, q, terrain=str(out)) == 2
    assert any(f['kind'] == 'touch' for f in json.loads(out.read_text())['findings'])
    q.analysis.terrain = {}
    # materials: the sample's radars behind the fascia are fine (exit 0); a steel body blocks them (exit 2)
    out = tmp_path / 'materials.json'
    assert _batch(tmp_path, q, materials=str(out)) == 0
    doc = json.loads(out.read_text())
    assert doc['schema'] == 'vantage.materials/1' and {r['tag'] for r in doc['radars']} >= {'RD1', 'RD2'}
    q.vehicle.mesh.parts[0].surface = 'steel'
    assert _batch(tmp_path, q, materials=str(out)) == 2
    assert any(r['mount'] == 'blocked' for r in json.loads(out.read_text())['radars'])
    q.vehicle.mesh.parts[0].surface = ''
    # weather: every condition compared, exit 0; --condition picks the one the studies use
    out = tmp_path / 'weather.json'
    assert _batch(tmp_path, q, weather=str(out), condition='fog') == 0
    doc = json.loads(out.read_text())
    assert doc['schema'] == 'vantage.weather/1' and doc['selected'] == 'fog'
    out = tmp_path / 'night.json'
    assert _batch(tmp_path, q, perception_range=str(out), condition='night') == 0
    assert json.loads(out.read_text())['condition']['id'] == 'night'
    # CI templates into a folder
    assert _batch(tmp_path, q, ci_templates=str(tmp_path / 'ci')) == 0
    assert (tmp_path / 'ci' / 'ci' / 'github' / 'vantage.yml').is_file()
    # --watch: one run of the project's automation, the command's own exit code passed through
    src = tmp_path / 'w' / 'rig.vantage'
    src.parent.mkdir()
    q.save(str(src))
    a = _args(watch=str(src), on_change='exit 7', watch_runs=1, formats='extrinsics_json')
    assert batch.run(a) == 7
    assert (src.parent / 'exports' / 'automation' / 'extrinsics.json').is_file()
    with pytest.raises(SystemExit):
        batch.run(_args(on_change='exit 7'))                  # --on-change needs --watch


def test_assistant_reads_the_rig_studies():
    from vantage.assistant import tools as T
    from vantage.core.analysis import Engine
    p = _rig()
    tc = T.ToolContext(p.snapshot(), None, Engine())
    for name in ('get_power_budget', 'get_gnss_imu', 'get_sun_and_dirt', 'get_fleet', 'get_brackets',
                 'get_sample_dataset', 'get_drive_log_check', 'get_asbuilt_check', 'get_requirements',
                 'get_requirement_metrics', 'get_materials', 'get_automation', 'get_road_and_body_motion',
                 'get_cable_harness', 'get_calibration_observability',
                 'plan_lidar_calibration'):
        assert name in T.BY_NAME, name
        out = json.loads(T.run(tc, name, {})[0])
        assert 'error' not in out, (name, out)
    g = json.loads(T.run(tc, 'get_gnss_imu', {'antenna': 'GN2'})[0])
    assert g['antenna_detail']['tag'] == 'GN2'
    assert json.loads(T.run(tc, 'get_drive_log_check', {})[0])['checked'] is False
    assert json.loads(T.run(tc, 'get_asbuilt_check', {})[0])['checked'] is False
    # the assistant can open each page
    desc = T.BY_NAME['show_in_app'].description
    assert all(page in desc for page, _ in PAGES.values())
    from vantage.ui.assistant import PAGES as NAMES
    assert all(page in NAMES for page, _ in PAGES.values())
    # materials: every radar of the rig; automation: off, no command, nothing that could run one
    m = json.loads(T.run(tc, 'get_materials', {})[0])
    assert {r['tag'] for r in m['radars']} == {s.tag for s in p.variant.sensors if s.kind == 'radar'}
    a = json.loads(T.run(tc, 'get_automation', {})[0])
    assert a['trigger'] == 'off' and T.BY_NAME['get_automation'].params == {}
    assert not [t for t in T.TOOLS if 'automation' in t.name and t.name != 'get_automation']
    # critical targets: none placed says where to place them; placed ones are read with their cameras
    assert 'Critical targets' in json.loads(T.run(tc, 'get_critical_targets', {})[0])['error']
    q = _with_targets(p.snapshot(), 'Child 2 m ahead')
    c = json.loads(T.run(T.ToolContext(q, None, Engine()), 'get_critical_targets', {'envelopes': False})[0])
    assert c['targets'] == 1 and c['per_target'][0]['resolved'] and c['per_target'][0]['cameras']
    # the weather study: every condition, the chosen one selected
    w = json.loads(T.run(tc, 'run_study', {'study': 'weather', 'condition': 'heavy_rain'})[0])
    assert w['selected'] == 'heavy_rain' and {r['condition'] for r in w['conditions']} >= {'clear', 'heavy_rain'}
    assert 'error' in json.loads(T.run(tc, 'run_study', {'study': 'weather', 'condition': 'sandstorm'})[0])


def test_rig_study_issues_reach_the_issues_list():
    from vantage.core.analysis import Engine
    from vantage.core.rules import collect_issues
    p = _rig()
    eng = Engine()
    vg = eng.vehicle_geom(p)
    assert not [i for i in collect_issues(p, None, vg) if 'GN' in i.objects]
    p.analysis.power = {'dcdc_w': 50}                    # the converter cannot carry the rig
    iss = collect_issues(p, None, vg)
    assert any(i.severity == 'critical' and 'DC-DC' in i.objects for i in iss)
    p.analysis.power = {}
    # a critical target no camera resolves (serious), and an automation setting the project file may not carry
    from vantage.core import critical as CR
    _with_targets(p, 'Child 2 m ahead')
    st = CR.settings_of(p)
    far = CR.preset(p, 'sign', st.targets)
    far['x'], far['name'] = 400.0, 'Far sign'
    st.targets.append(far)
    p.analysis.critical = st.to_dict()
    p.analysis.automation = {'enabled': True, 'command': 'rm -rf ~'}
    iss = collect_issues(p, None, vg)
    assert any(i.severity == 'serious' and 'Far sign' in i.message for i in iss)
    assert not any('Child' in i.message and 'No camera' in i.message for i in iss)
    assert any(i.message.startswith('Automation:') and 'command' in i.message for i in iss)
    assert not any('rm -rf' in i.message for i in iss)
    # the cable harness: off, the cable check is the Manhattan estimate; on, the routed length everywhere
    from vantage.core import harness as HN, power as PW, rules
    p.analysis.critical, p.analysis.automation = {}, {}
    f = p.variant.by_tag('F')
    est = rules.route_length(p, f)
    assert est == rules.route_length(p, f, routed=False) and est < f.cable_m
    assert not any(i.message.startswith('Routed cable') and f.uid in i.uids for i in collect_issues(p, None, vg))
    p.analysis.harness = {'enabled': True}
    routed = rules.route_length(p, f)
    assert routed == pytest.approx(HN.routed_length(p, f)) and routed > f.cable_m > est
    assert rules.route_length(p, f, routed=False) == est                 # the optimiser's inner loop keeps it
    assert '→' in rules.route_text(p, f)
    l1 = next(s for s in p.variant.sensors if s.tag == 'L1')
    assert PW.route_length(p, l1) == pytest.approx(HN.routed_length(p, l1))
    iss = collect_issues(p, None, vg)
    assert any(i.message.startswith('Routed cable') and f.uid in i.uids for i in iss)  # rules' cable check, routed
    x0, p.ecu.x = p.ecu.x, -20.0                                         # T1 and coax runs over their limits
    assert any(i.severity == 'serious' and 'longer than the' in i.message for i in collect_issues(p, None, vg))
    p.ecu.x, p.analysis.harness = x0, {}


# ----------------------------------------------------------------------------- the window
from test_gui import app, win, pump, settle, _wait     # noqa: E402,F401  (fixtures)


def test_every_rig_page_opens_from_its_menu(app, win):
    menus = {a.text().replace('&', ''): a.menu() for a in win.menuBar().actions()}
    for act_id, (page, menu) in PAGES.items():
        act = win.actions_by_id[act_id]
        assert act in menus[menu].actions(), act_id
        assert act.icon() is not None and not act.icon().isNull(), act_id
        act.trigger()
        pump(app, 0.3)
        assert win.page_name() == page and win.page(page).isVisible(), act_id
        assert page in win.PAGE_DOCKS
    win.set_page('viewports')
    settle(app, win)
    # the command palette knows them too
    assert all(win.actions_by_id[a].text() for a in PAGES)
    # Weather and night opens the perception range on its weather comparison, the condition picker focused
    pg = win.page('perception')
    pg.wtabs.setCurrentIndex(2)
    win.set_page('viewports')
    win.actions_by_id['weather'].trigger()
    pump(app, 0.3)
    assert win.page_name() == 'perception' and pg.wtabs.currentIndex() == 0
    win.set_page('viewports')
    settle(app, win)
    # the new overlays are in View ▸ Overlays
    menus = {a.text().replace('&', ''): a.menu() for a in win.menuBar().actions()}
    ov = next(a.menu() for a in menus['View'].actions() if a.menu() and a.text().replace('&', '') == 'Overlays')
    assert {'Critical targets', 'Wiper sweep'} <= {a.text() for a in ov.actions()}


def test_new_overlays_draw_in_the_main_views(app, win):
    """View ▸ Overlays ▸ Critical targets, Cable harness and Wiper sweep draw through the plan view's and the 3D scene's
    hooks, only while switched on; the critical page's own views do not draw the targets twice."""
    from unittest import mock
    from vantage.ui import harnessview as HV
    from vantage.ui.workspaces import critical_ws as CW, materials_ws as MW
    st = win.store
    st.project.active_uid = next(v for v in st.project.variants if v.sensors).uid
    _with_targets(st.project, 'Child 2 m ahead')
    st.project_replaced.emit()
    settle(app, win)
    win.set_view('top')
    settle(app, win)
    plan = win.viewarea.panes[0].plan
    sc = win.viewarea.panes[0].vp3d.provider
    vp = win.viewarea.panes[0].vp3d
    keys = ('critical', 'harness', 'wipers')
    was = {k: win.ctx.overlays.get(k) for k in keys}
    try:
        with mock.patch.object(CW, 'paint_plan', wraps=CW.paint_plan) as cp, \
                mock.patch.object(HV, 'paint_plan', wraps=HV.paint_plan) as hp:
            for k in keys:
                win.ctx.set_overlay(k, False)
            plan.grab()
            assert cp.call_count == 0 and hp.call_count == 0          # off: the hooks are not even asked
            for k in keys:
                win.ctx.set_overlay(k, True)
            plan.grab()
            assert cp.call_count == 1 and hp.call_count == 1
        hc = HV.controller(win.ctx)                            # the overlay routes the harness in the background
        _wait(app, lambda: hc.fresh() and not hc.busy, 60)
        assert HV.scene_draws(win.ctx)
        with mock.patch.object(CW, 'scene_tris', wraps=CW.scene_tris) as ct, \
                mock.patch.object(CW, 'scene_lines', wraps=CW.scene_lines) as cl, \
                mock.patch.object(HV, 'scene_draws', wraps=HV.scene_draws) as hd, \
                mock.patch.object(MW, 'scene_lines', wraps=MW.scene_lines) as wl:
            from vantage.ui.viewport3d import LineDraw, TriDraw
            tris, lines = sc.tris_extra(vp), sc.lines(vp)
            assert ct.call_count == cl.call_count == hd.call_count == wl.call_count == 1
            # every draw is of its list's kind (a line among the triangles breaks the software renderer)
            assert all(not isinstance(d, LineDraw) for d in tris) and all(isinstance(d, LineDraw) for d in lines)
            assert any(isinstance(d, TriDraw) for d in tris)
            # the critical page's own 3D view draws its targets itself, once (the main-view hook steps aside)
            own = CW.CriticalScene(win.ctx, CW.controller(win.ctx))
            n = ct.call_count
            own.tris_extra(vp)
            assert ct.call_count == n + 1
        errors = []                                            # an exception while painting reaches sys.excepthook
        with mock.patch.object(sys, 'excepthook', lambda *a: errors.append(a)):
            plan.grab()
            vp.update()
            pump(app, 0.3)                                     # the software 3D view paints them
            settle(app, win)
        assert not errors, errors
    finally:
        for k in keys:
            win.ctx.set_overlay(k, bool(was[k]))
        settle(app, win)


def test_automation_runs_when_the_window_saves(app, win, tmp_path):
    """The window owns the automation controller from the start (the page need not be open) and its saves trigger a
    run, also a save with nothing changed."""
    ctl = win.ctx.automation
    assert 'automation' not in win._page_widgets and not ctl.busy
    st = win.store
    st.project.path = str(tmp_path / 'proj' / 'rig.vantage')
    os.makedirs(os.path.dirname(st.project.path))
    st.edit('analysis', lambda q: setattr(q.analysis, 'automation', {'enabled': True, 'formats': ['extrinsics_json']}),
            'automation on')
    assert win.save()
    _wait(app, lambda: len(ctl.runs()) == 1 and not ctl.busy)
    out = tmp_path / 'proj' / 'exports' / 'automation'
    assert (out / 'extrinsics.json').is_file() and (out / 'vantage_summary.json').is_file()
    assert ctl.runs()[0].trigger == 'save' and ctl.runs()[0].status == 'passed'
    assert win.save()                                   # nothing changed: still a save, one more run
    _wait(app, lambda: len(ctl.runs()) == 2 and not ctl.busy)
    assert ctl.runs()[0].n_changed() == 0
    st.edit('analysis', lambda q: setattr(q.analysis, 'automation', {}), 'automation off')
    settle(app, win)


def test_recording_page_lists_an_imu(app, win):
    """A variant with an IMU (and antennas, which record nothing themselves) opens on the recording budget page."""
    st = win.store
    n0 = st.undo.index()
    for a in ('add_gnss', 'add_imu'):
        win.actions_by_id[a].trigger()
    settle(app, win)
    win.actions_by_id['recording_budget'].trigger()
    pump(app, 0.3)
    pg = win.page('recording')
    assert win.page_name() == 'recording' and pg.result.row('IMU1')['kind'] == 'imu'
    assert pg.result.row('GN1') is None
    tags = [pg.table.item(i, 0).text() for i in range(pg.table.rowCount()) if pg.table.item(i, 0)]
    kinds = {pg.table.item(i, 0).text(): pg.table.item(i, 1).text() for i in range(pg.table.rowCount())
             if pg.table.item(i, 0) and pg.table.item(i, 1)}
    assert 'IMU1' in tags and kinds['IMU1'] == 'IMU'
    assert '1 IMU' in pg.tiles['rate'].note
    while st.undo.index() > n0:
        st.undo.undo()
    win.set_page('viewports')
    settle(app, win)


def test_export_dialog_fits_the_screen(app, win):
    """With every format listed the list is taller than a laptop screen: it scrolls, and the dialog (with its Export
    button) stays on the screen it opens on."""
    from vantage.ui.dialogs.export_dialog import ExportDialog
    d = ExportDialog(win.ctx, win)
    d.show()
    pump(app, 0.2)
    avail = d.screen().availableGeometry()          # the test screen is 800 × 800: shorter than the full list
    g = d.frameGeometry()
    assert avail.top() <= g.top() and g.bottom() <= avail.bottom(), (avail, g)
    sc = d.format_scroll
    assert sc.widget().sizeHint().height() > sc.viewport().height()          # the list scrolls instead
    btn = d.go.mapToGlobal(d.go.rect().bottomRight())
    assert d.go.isVisible() and avail.top() <= btn.y() <= avail.bottom()    # the Export button is on the screen
    d.close()
    pump(app, 0.1)


# ----------------------------------------------------------------------------- the working folder
def test_working_folder_paths(tmp_path):
    import datetime as dt
    from PyQt6.QtCore import QSettings
    from vantage import workdir
    st = QSettings(str(tmp_path / 's.ini'), QSettings.Format.IniFormat)
    assert workdir.get(st) == workdir.default() and workdir.default().endswith(os.path.join('Documents', 'Vantage'))
    assert workdir.put(st, str(tmp_path / 'work')) == str(tmp_path / 'work') == workdir.get(st)
    p = _rig()
    when = dt.datetime(2026, 10, 23, 13, 5, 9)
    d1 = workdir.dataset_dir(p, base=str(tmp_path / 'work'), when=when)
    assert d1 == str(tmp_path / 'work' / 'tigor-ev-svm' / 'datasets' / 'data-rig-6-cam-lidar-radars-20261023-130509')
    os.makedirs(d1)
    assert workdir.dataset_dir(p, base=str(tmp_path / 'work'), when=when).endswith('-130509-2')   # never overwritten
    assert workdir.display(d1, str(tmp_path / 'work')) == os.path.join('tigor-ev-svm', 'datasets',
                                                                       os.path.basename(d1))
    assert workdir.display('/elsewhere/x', str(tmp_path / 'work')) == 'x'
    assert workdir.put(st, '') == workdir.default()


def test_assistant_writes_a_sample_dataset_into_the_working_folder(tmp_path):
    from vantage.assistant import tools as T
    from vantage.core.analysis import Engine
    p = _rig()
    p.analysis.dataset = {'image_scale': 0.1}
    shown = []
    tc = T.ToolContext(p.snapshot(), None, Engine(), working_dir=str(tmp_path))
    tc.on_ui = shown.append
    text, ok = T.run(tc, 'generate_sample_dataset', {'duration_s': 0.6, 'speed_kmh': 20})
    out = json.loads(text)
    assert ok and out['written'] and out['samples'] >= 1 and out['settings']['duration_s'] == 0.6
    assert str(tmp_path) not in text and out['folder_in_working_folder'].startswith(os.path.join('tigor-ev-svm',
                                                                                                'datasets'))
    folder = tmp_path / out['folder_in_working_folder']
    assert (folder / 'nuscenes' / 'v1.0-vantage' / 'sample.json').is_file() and (folder / 'rosbag').is_dir()
    assert shown and shown[0]['action'] == 'dataset_written' and shown[0]['result'].out_dir == str(folder)
    # the drive-log check knows a drive Vantage wrote: judged by the rate and size it was written with, it is clean
    from vantage.core import drivelog as DL
    bag = next((folder / 'rosbag').iterdir())
    assert DL.vantage_dataset(str(bag))['schema'] == DL.DATASET_SCHEMA and DL.vantage_dataset(str(tmp_path)) is None
    chk = DL.run(p, str(bag), engine=Engine())
    assert chk.metrics['streams_faulty'] == 0 and chk.metrics['faults'] == 0, [f['message'] for f in chk.findings]
    assert any('sample dataset Vantage wrote' in f['message'] for f in chk.findings)
    # a second run makes a new folder; the first stays
    out2 = json.loads(T.run(tc, 'generate_sample_dataset', {'duration_s': 0.6})[0])
    assert out2['folder_in_working_folder'] != out['folder_in_working_folder'] and folder.is_dir()
    assert T.BY_NAME['generate_sample_dataset'].kind == 'compute'


def test_dataset_page_and_preferences_use_the_working_folder(app, win, tmp_path):
    from vantage import workdir
    from vantage.ui.dialogs.misc import PreferencesDialog
    st = win.ctx.settings
    old = st.value(workdir.KEY, '')
    try:
        d = PreferencesDialog(win.ctx, win)
        d.working.setText(str(tmp_path / 'work'))
        d.apply()
        assert workdir.get(st) == str(tmp_path / 'work')
        win.store.project.analysis.dataset = {'image_scale': 0.1, 'duration_s': 0.6}
        win.actions_by_id['sample_dataset'].trigger()
        pump(app, 0.3)
        pg = win.page('dataset')
        pg.refresh()
        assert 'tigor-ev-svm' in pg.where.text() and pg.b_generate.text() == 'Generate'
        assert pg.generate()                                   # no folder dialog: the working folder
        _wait(app, lambda: not pg.busy, 120)
        assert pg.result is not None and pg.result.out_dir.startswith(str(tmp_path / 'work' / 'tigor-ev-svm'))
        # a result the assistant wrote is shown on the page the same way
        res = pg.result
        pg.result = None
        pg.show_written(res)
        assert pg.result is res and 'Load it' in pg.out.toPlainText()
    finally:
        st.setValue(workdir.KEY, old)
        win.store.project.analysis.dataset = {}
        win.set_page('viewports')
        settle(app, win)


def test_dataset_page_picks_its_own_weather(app, win):
    """The sample dataset follows the perception page's weather unless the page picks a condition of its own (one undo
    step); the perception controller notices a weather edit made while its page is hidden."""
    st = win.store
    win.actions_by_id['sample_dataset'].trigger()
    pump(app, 0.3)
    pg = win.page('dataset')
    assert pg.weather.currentData() == '' and pg.settings.weather == ''
    n0 = st.undo.index()
    pg.weather.setCurrentIndex(pg.weather.findData('fog'))
    pump(app, 0.1)
    assert pg.settings.weather == 'fog' and st.undo.index() == n0 + 1
    st.undo.undo()
    pump(app, 0.1)
    assert pg.settings.weather == '' and pg.weather.currentData() == ''
    from vantage.ui.perceptionview import controller as perception_controller
    ctl = perception_controller(win.ctx)
    win.set_page('perception')
    _wait(app, lambda: ctl.result is not None and not ctl.busy and not ctl.stale, 120)
    win.set_page('viewports')
    assert not ctl.stale
    st.edit('analysis', lambda q: setattr(q.analysis, 'weather', {'condition': 'night'}), 'night')
    assert ctl.stale                     # at once, from the edit itself (not only once the analysis has run again)
    st.undo.undo()
    settle(app, win)


def test_dataset_preview_plays_the_drive(app, win):
    """Play steps the preview through the drive at the chosen speed, loops, and replays played frames from the cache
    without rendering them again; pause, the slider and leaving the page stop it."""
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    st = win.store
    st.project.analysis.dataset = {'duration_s': 1.0}
    try:
        win.actions_by_id['sample_dataset'].trigger()
        pump(app, 0.3)
        pg = win.page('dataset')
        pg.refresh()
        _wait(app, lambda: pg.preview_data is not None and not pg._pbusy, 60)
        pg.slider.setValue(0)
        assert pg.start_play() and pg.playing and pg.b_play.toolTip().startswith('Pause')
        seen = set()
        t0 = time.time()
        while time.time() - t0 < 40 and len(pg._frames) < 6:   # every instant of the 1 s drive rendered once
            pump(app, 0.02)
            seen.add(pg.slider.value())
        assert {0, 2, 4, 6, 8, 10} <= seen and len(pg._frames) == 6      # 0.2 s steps at 1×
        gen = pg._pgen
        pump(app, 1.5)                                          # the second loop comes from the cache
        assert pg.playing and pg._pgen == gen
        pg.stop_play()
        assert not pg.playing and pg.b_play.toolTip().startswith('Play')
        # 2×: 0.4 s steps
        pg.speed.set_value('2')
        pg.slider.setValue(0)
        pg.start_play()
        seen = set()
        t0 = time.time()
        while time.time() - t0 < 20 and len(seen) < 3:
            pump(app, 0.02)
            seen.add(pg.slider.value())
        assert seen <= {0, 4, 8} and {4, 8} & seen
        QTest.keyClick(pg, Qt.Key.Key_Space)                    # Space pauses
        assert not pg.playing
        pg.start_play()
        win.set_page('viewports')                               # leaving the page stops it
        pump(app, 0.2)
        assert not pg.playing
    finally:
        st.project.analysis.dataset = {}
        settle(app, win)


def test_camera_menu_opens_the_parking_guidelines(app, win):
    menus = {a.text().replace('&', ''): a.menu() for a in win.menuBar().actions()}
    act = win.actions_by_id['parking']
    assert act in menus['Camera'].actions()
    was = win.ctx.overlays.get('guidelines')
    try:
        win.ctx.set_overlay('guidelines', False)
        pump(app, 0.1)
        act.trigger()
        pump(app, 0.3)
        bar = win.page('camviews').guide_bar
        assert win.page_name() == 'camviews' and win.ctx.overlays['guidelines']
        assert win.actions_by_id['ov_guidelines'].isChecked() and bar.on.isChecked() and bar.isVisible()
        act.trigger()                                            # again: stays on, nothing toggles off
        pump(app, 0.1)
        assert win.ctx.overlays['guidelines'] and win.page_name() == 'camviews'
    finally:
        win.ctx.set_overlay('guidelines', bool(was))
        win.set_page('viewports')
        settle(app, win)



# ----------------------------------------------------------------------------- round 7: experiments, validation,
# provenance, the CARLA simulator
def _with_experiment(p, design='oat'):
    """A small experiment on the active variant: F's pitch over three levels, the blind ground and the minimum density."""
    from vantage.core import experiments as EXP
    F = p.variant.by_tag('F')
    f = dict(EXP.new_factor(p, p.variant, 'camera', F.uid, 'pitch'), lo=F.pitch - 5.0, hi=F.pitch + 5.0, levels=3,
             mirror=False)
    st = EXP.ExperimentSettings.from_dict({'factors': [f], 'design': design, 'seed': 2, 'bootstrap': 50,
                                           'responses': [{'metric': 'coverage.blind_m2'},
                                                         {'metric': 'coverage.min_density'}]})
    p.analysis.experiments = st.to_dict()
    return st


def test_experiment_export_format_runs_the_project_experiment(tmp_path):
    from vantage.core import exporters as ex
    from vantage.core import experiments as EXP
    p = _rig()
    f = ex.format_by_id('experiment')
    assert f.group == 'Documents' and not f.default and ex._names(f, p, p.variant) == []    # none set up: nothing
    assert ex.export(['experiment'], p, None, ex.ExportOptions(out_dir=str(tmp_path / 'none'))) == []
    assert 'No experiment' in ex.preview('experiment', p, None, ex.ExportOptions(out_dir=str(tmp_path)))[1]
    _with_experiment(p)
    names = ex._names(f, p, p.variant)
    assert names[:3] == ['experiment/experiment_spec.json', 'experiment/experiment_points.csv',
                         'experiment/experiment_analysis.json'] and 'experiment/experiment_sensitivity.png' in names
    assert 'One at a time: 3 design points' in ex.preview('experiment', p, None, ex.ExportOptions(out_dir=str(tmp_path)))[1]
    paths = ex.export(['experiment'], p, None, ex.ExportOptions(out_dir=str(tmp_path)))
    got = {os.path.relpath(x, tmp_path).replace(os.sep, '/') for x in paths}
    assert set(names[:3]) <= got <= set(names)
    doc = json.loads((tmp_path / 'experiment' / 'experiment_analysis.json').read_text())
    assert doc['schema'] == EXP.SCHEMA and doc['provenance']['settings']['experiments']['design'] == 'oat'
    assert doc['provenance']['seeds'] == {'experiments.seed': 2}
    # the spec round-trips into batch mode
    spec = json.loads((tmp_path / 'experiment' / 'experiment_spec.json').read_text())
    assert EXP.load_spec(spec, p).to_dict()['factors'][0]['target'] == p.variant.by_tag('F').uid


def test_batch_flags_of_round_seven(tmp_path, capsys, monkeypatch):
    """--experiment, --validate (with --validate-models, --validate-quick), --carla-rig and --carla-run: their files
    and exit codes, each JSON stamped with its provenance."""
    from vantage import batch
    from vantage.core import experiments as EXP
    from test_advanced import _args
    for flag in ('experiment', 'validate', 'validate_models', 'validate_quick', 'carla_rig', 'carla_run', 'carla_python',
                 'reproduce', 'tolerance'):
        assert hasattr(_args(), flag), flag
    for flag, v in (('experiment', ['s.json', 'o']), ('validate', 'v'), ('carla_rig', 'c'), ('carla_run', 'r'),
                    ('reproduce', ['r.json'])):
        assert batch.wanted(_args(**{flag: v})), flag
    for flag, v in (('validate_models', 'lens'), ('validate_quick', True), ('carla_python', 'py')):
        assert not batch.wanted(_args(**{flag: v})), flag                 # modifiers, like --variant
    with pytest.raises(SystemExit):
        batch.run(_args(validate_quick=True))                  # --validate-quick needs --validate
    with pytest.raises(SystemExit):
        batch.run(_args(carla_python='python3'))              # --carla-python needs --carla-run
    p = _rig()
    # an experiment from its spec: exit 0, its files stamped; a spec naming a camera the project lacks: exit 4
    st = _with_experiment(p.snapshot())
    spec = tmp_path / 'experiment_spec.json'
    spec.write_text(json.dumps(EXP.spec_doc(st, p)))
    out = tmp_path / 'exp'
    assert _batch(tmp_path, p, experiment=[str(spec), str(out)]) == 0
    doc = json.loads((out / 'experiment_analysis.json').read_text())
    assert doc['schema'] == 'vantage.experiments/1' and doc['provenance']['output']['option'] == '--experiment'
    assert (out / 'experiment_points.csv').is_file() and (out / 'experiment_points.jsonl').is_file()
    rep = tmp_path / 'rep.json'                                # and it reproduces, bit for bit
    assert _batch(tmp_path, p, reproduce=[str(out / 'experiment_analysis.json'), str(rep)]) == 0
    r = json.loads(rep.read_text())                            # (the spec found by its hash, not the copy beside
    assert r['verdict'] == 'reproduces' and r['reasons'] == []  # the results: no input differs)
    bad = json.loads(spec.read_text())
    bad['settings']['factors'][0].update(target='nope', tag='NOPE')
    spec.write_text(json.dumps(bad))
    assert _batch(tmp_path, p, experiment=[str(spec), str(tmp_path / 'exp2')]) == 4
    assert 'experiment:' in capsys.readouterr().err
    # the validation suite, no project: the checks of one model, the report written and stamped (no project in it)
    out = tmp_path / 'val'
    a = _args(validate=str(out), validate_models='sun', validate_quick=True)
    a.sample = False
    assert batch.run(a) == 0
    vdoc = json.loads((out / 'validation.json').read_text())
    assert vdoc['summary']['ok'] and {m['id'] for m in vdoc['models']} == {'sun'}
    assert vdoc['provenance']['project'] is None and vdoc['provenance']['output']['option'] == '--validate'
    assert (out / 'validation.html').is_file() and (out / 'validation.csv').is_file()
    a = _args(reproduce=[str(out / 'validation.json'), str(tmp_path / 'vrep.json')])
    a.sample = False
    assert batch.run(a) == 0 and json.loads((tmp_path / 'vrep.json').read_text())['verdict'] == 'reproduces'
    # a model that no longer matches its reference: exit 6
    from vantage.core import weather as W
    monkeypatch.setattr(W, 'rain_optical_db_km', lambda rate: 1.58 * np.power(np.maximum(rate, 0.0), 0.63))
    a = _args(validate=str(tmp_path / 'val2'), validate_models='weather', validate_quick=True)
    a.sample = False
    assert batch.run(a) == 6
    assert 'FAIL  weather.p1814_coefficients' in capsys.readouterr().err
    monkeypatch.undo()
    with pytest.raises(SystemExit):
        batch.run(_args(validate=str(tmp_path / 'v3'), validate_models='nope'))
    # the CARLA rig: exit 0 and its two files; a variant with nothing to simulate: exit 2
    out = tmp_path / 'carla'
    assert _batch(tmp_path, p, carla_rig=str(out)) == 0
    rig = json.loads((out / 'sim_rig.json').read_text())
    assert rig['schema'] == 'vantage.sim_rig/1' and (out / 'sim_remap.npz').stat().st_size > 0
    assert rig['provenance']['settings']['sim']['town'] == 'Town10HD'
    q = p.snapshot()
    for o in list(q.variant.cameras) + list(q.variant.sensors):
        o.visible = False
    assert _batch(tmp_path, q, carla_rig=str(tmp_path / 'carla2')) == 2
    assert _batch(tmp_path, q, carla_run=str(tmp_path / 'run0'), carla_python=sys.executable) == 2
    assert 'Nothing to simulate' in capsys.readouterr().err
    # --carla-run: no server on the project's port, or no client Python: exit 7, nothing run
    import socket
    s_ = socket.socket()
    s_.bind(('127.0.0.1', 0))
    free = s_.getsockname()[1]
    s_.close()
    p.analysis.sim = {'port': free}
    assert _batch(tmp_path, p, carla_run=str(tmp_path / 'run1'), carla_python=sys.executable) == 7
    assert f'no CARLA server answers on port {free}' in capsys.readouterr().err
    assert _batch(tmp_path, p, carla_run=str(tmp_path / 'run2'), carla_python=str(tmp_path / 'no-python')) == 7
    assert 'no Python for CARLA' in capsys.readouterr().err


def test_batch_carla_run_with_the_fake_carla(tmp_path, monkeypatch):
    """--carla-run end to end against tests/fake_carla (a listening socket stands for the server): the rig exported
    into OUT/rig, the bridge run in its own process, the recording and run.json written and stamped; exit 0."""
    import socket
    from vantage.sim import server as SV
    fake = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fake_carla')
    real = SV.bridge_command

    def with_fake(python, rig, out, args=(), **kw):
        cmd, env = real(python, rig, out, args, scope=False)
        env['PYTHONPATH'] = fake + os.pathsep + env['PYTHONPATH']
        return cmd, env
    monkeypatch.setattr(SV, 'bridge_command', with_fake)
    srv = socket.socket()
    srv.bind(('127.0.0.1', 0))
    srv.listen(4)
    try:
        p = _rig()
        p.variant.cameras[:] = p.variant.cameras[:1]
        p.analysis.sim = {'port': srv.getsockname()[1], 'image_scale': 0.05, 'duration_s': 0.5, 'traffic': 1,
                          'warmup_s': 0.1, 'town': 'Town01'}
        out = tmp_path / 'drive'
        assert _batch(tmp_path, p, carla_run=str(out), carla_python=sys.executable) == 0
    finally:
        srv.close()
    run = json.loads((out / 'run.json').read_text())
    assert run['status'] == 'ok' and run['town'] == 'Town01' and run['provenance']['output']['option'] == '--carla-run'
    assert (out / 'rig' / 'sim_rig.json').is_file() and (out / 'nuscenes').is_dir() and (out / 'rosbag').is_dir()
    assert json.loads((out / 'dataset.json').read_text())['generator'] == 'carla'


def test_assistant_reads_experiments_validation_provenance_and_the_carla_rig(tmp_path):
    from vantage.assistant import tools as T
    from vantage.core import exporters as ex
    from vantage.core.analysis import Engine
    p = _rig()
    for name in ('get_experiment', 'run_experiment', 'run_validation', 'list_validation_checks', 'check_result',
                 'get_sim_rig'):
        assert name in T.BY_NAME, name
    tc = T.ToolContext(p.snapshot(), None, Engine(), working_dir=str(tmp_path))
    assert 'Experiments' in json.loads(T.run(tc, 'get_experiment', {})[0])['error']          # no page result
    assert 'Add a factor' in json.loads(T.run(tc, 'run_experiment', {})[0])['error']          # none set up
    q = p.snapshot()
    _with_experiment(q)
    tq = T.ToolContext(q, None, Engine(), working_dir=str(tmp_path))
    assert 'more than 2' in json.loads(T.run(tq, 'run_experiment', {'max_points': 2})[0])['error']
    e = json.loads(T.run(tq, 'run_experiment', {})[0])
    assert e['points'] == 3 and e['design'] == 'oat' and {r['key'] for r in e['responses']} >= {'coverage.blind_m2'}
    page = {**e, 'stale': False}
    assert json.loads(T.run(T.ToolContext(q, None, Engine(), experiment=page), 'get_experiment', {})[0])['points'] == 3
    # the validation suite: one model, quick; the catalogue without running
    v = json.loads(T.run(tc, 'run_validation', {'models': ['sun']})[0])
    assert v['ok'] and v['passed'] == v['checks'] > 0 and v['models'] == {'sun': f'{v["checks"]}/{v["checks"]}'}
    assert 'error' in json.loads(T.run(tc, 'run_validation', {'models': ['nope']})[0])
    cat = json.loads(T.run(tc, 'list_validation_checks', {'models': ['occlusion']})[0])
    assert {c['id'] for c in cat['checks']} >= {'occlusion.cube_vs_rays', 'occlusion.cube_depth'}
    # a result's provenance: intact, then edited; never a file outside the working or project folder
    ex.export(['extrinsics_json'], p, None, ex.ExportOptions(out_dir=str(tmp_path / 'exports')))
    r = json.loads(T.run(tc, 'check_result', {'path': 'exports/extrinsics.json'})[0])
    assert r['provenance'] and r['intact'] is True and r['study'] == 'extrinsics_json' and r['same_project_as_open']
    assert r['differences_from_now'] == [] and r['file'] == os.path.join('exports', 'extrinsics.json')
    f = tmp_path / 'exports' / 'extrinsics.json'
    d = json.loads(f.read_text())
    d['cameras'][0]['tag'] = 'EDITED'                         # a number or a name changed after Vantage wrote it
    f.write_text(json.dumps(d))
    assert json.loads(T.run(tc, 'check_result', {'path': str(f)})[0])['intact'] is False
    outside = tmp_path.parent / 'outside.json'
    outside.write_text('{}')
    assert 'error' in json.loads(T.run(tc, 'check_result', {'path': str(outside)})[0])
    assert T.BY_NAME['check_result'].kind == 'read'
    # the CARLA rig, planned only
    g = json.loads(T.run(tc, 'get_sim_rig', {})[0])
    assert [c['tag'] for c in g['cameras']] == [c.tag for c in p.variant.cameras if c.visible]
    assert {x['tag'] for x in g['lidars']} == {'L1'} and g['tick_hz'] > 0
    assert 'error' in json.loads(T.run(tc, 'get_sim_rig', {'variant': 'nope'})[0])


def test_validation_from_the_help_menu_and_stamped_page_exports(app, win, tmp_path):
    """Help ▸ Validate the models runs in the background and writes its report into the folder given (stamped with
    the provenance); a study page's Export button stamps its JSON too."""
    act = win.actions_by_id['validate_models']
    menus = {a.text().replace('&', ''): a.menu() for a in win.menuBar().actions()}
    assert act in menus['Help'].actions() and not act.icon().isNull()
    assert win.validate_models(folder=str(tmp_path / 'val'), open_report=False, quick=True, models=('sun',))
    _wait(app, lambda: 'validation' not in win._jobs, 120)
    doc = json.loads((tmp_path / 'val' / 'validation.json').read_text())
    assert doc['summary']['ok'] and doc['project'] == win.store.project.name
    assert doc['provenance']['interface'] == 'app' and doc['provenance']['project']['name'] == win.store.project.name
    assert win.validation_report == str(tmp_path / 'val' / 'validation.html')
    # the power budget's Export button
    win.actions_by_id['power_budget'].trigger()
    pump(app, 0.3)
    pg = win.page('power')
    _wait(app, lambda: pg.result is not None)
    paths = pg.export(str(tmp_path / 'power'))
    pdoc = json.loads((tmp_path / 'power' / 'power_budget.json').read_text())
    assert pdoc['provenance']['output']['name'] == 'power_budget.json' and pdoc['provenance']['study'] == 'power_budget'
    assert pdoc['provenance']['settings']['power']                           # its settings, every default filled in
    assert len(paths) == 3
    win.set_page('viewports')
    settle(app, win)


# ============================================================================ round 8: ground truth, faults, scenarios,
# detection quality, campaigns, critical targets in CARLA, the live rig (against tests/fake_carla)
FAKE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fake_carla')
NARROW = 'h03r2'                    # a 37° lens: one pinhole view per frame


def _fake_bridges(monkeypatch):
    """Every bridge and live server the app starts runs against tests/fake_carla in this Python, unscoped."""
    from vantage.core import carla_live as CL
    from vantage.sim import server as SV
    real = SV.bridge_command

    def bridge(python, rig, out, args=(), **kw):
        kw['scope'] = False
        cmd, env = real(python, rig, out, args, **kw)
        env['PYTHONPATH'] = FAKE_DIR + os.pathsep + env['PYTHONPATH']
        return cmd, env
    monkeypatch.setattr(SV, 'bridge_command', bridge)
    real_live = CL.command

    def live(python, rig_path, http_port=0, args=(), memory_gb=4.0, scope=True, extra_path=''):
        return real_live(python, rig_path, http_port, args, memory_gb, False, FAKE_DIR)
    monkeypatch.setattr(CL, 'command', live)


class _Server:
    """A listening socket on a free port standing for the CARLA server (the fake carla needs none)."""

    def __enter__(self):
        import socket
        self.sock = socket.socket()
        self.sock.bind(('127.0.0.1', 0))
        self.sock.listen(16)
        self.port = self.sock.getsockname()[1]
        return self

    def __exit__(self, *exc):
        self.sock.close()


def _free_port():
    import socket
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def _small(p, cams=('F',), lens=None, **sim):
    """The data rig cut to a few cameras and its lidar (on a mast: no self-hits in the fake), small CARLA runs."""
    v = p.variant
    v.cameras[:] = [c for c in v.cameras if c.tag in cams]
    if lens:
        for c in v.cameras:
            c.lens_id = lens
    v.sensors[:] = [s for s in v.sensors if s.kind == 'lidar']
    v.sensors[0].z = 2.0
    p.analysis.sim = dict({'image_scale': 0.08, 'duration_s': 0.6, 'traffic': 4, 'warmup_s': 0.25, 'town': 'Town01'},
                          **sim)
    return p


def _oracle(drive):
    """A stand-in detector that finds every labelled object of a drive exactly (the wiring is under test)."""
    import cv2
    from vantage.core import detection as DT
    from vantage.core import groundtruth as GT

    class Oracle:
        by = None

        def detect(self, img):
            if self.by is None:
                doc = GT.read(drive)
                self.by = {}
                for c in doc['cameras'].values():
                    for f in c['frames']:
                        self.by[cv2.imread(GT.path_of(doc, f['image'])).tobytes()] = f['objects']
            objs = [o for o in self.by.get(img.tobytes(), []) if DT.GT_TO_CLASS.get(o['class'])]
            return {'boxes': np.array([o['bbox'] for o in objs], np.float32).reshape(-1, 4),
                    'scores': np.full(len(objs), 0.9, np.float32),
                    'cls': np.array([DT.CLASSES.index(DT.GT_TO_CLASS[o['class']]) for o in objs], np.int8),
                    'coco': np.zeros(len(objs), np.int16), 'ms': 0.0, 'input': list(img.shape[1::-1])}

        __call__ = detect

        def describe(self):
            return {'model': 'oracle', 'name': 'ground-truth oracle', 'sha256': 'oracle'}
    return Oracle()


@pytest.fixture(scope='module')
def gt_drive(tmp_path_factory):
    """A drive with ground truth recorded by the bridge in-process against the fake carla: the ego parked with a car
    and a pedestrian stood ahead of F (an actors scene), so the ground truth has objects in view."""
    import importlib.util
    from vantage.core import simrig as SR
    from vantage.sim import carla_bridge as CB
    name = 'fake_carla_for_tests'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, os.path.join(FAKE_DIR, 'carla', '__init__.py'))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    fc = sys.modules[name]
    p = _small(_rig(), ('F',), lens=NARROW, traffic=0, duration_s=0.5)
    p.variant.by_tag('F').pitch = 5.0                  # a narrow lens looking ahead: the actors in its view
    p.analysis.sim['actors'] = [{'id': 'car1', 'kind': 'car', 'pose': {'x': 16.0, 'y': 0.0, 'yaw': 180.0}},
                                {'id': 'ped1', 'kind': 'pedestrian', 'pose': {'x': 11.0, 'y': -1.0, 'yaw': 90.0}}]
    root = tmp_path_factory.mktemp('gt')
    rig = SR.build(p)
    SR.export(rig, p, str(root / 'rig'))
    out = str(root / 'drive')
    fc.reset()
    doc, remaps = CB.load_rig(str(root / 'rig' / 'sim_rig.json'))
    a = CB.resolve_args(CB.parser().parse_args([str(root / 'rig' / 'sim_rig.json'), out]), doc)
    CB.Bridge(fc, doc, remaps, out, a, None, None).run()
    assert os.path.isfile(os.path.join(out, 'ground_truth', 'index.json'))
    return {'p': p, 'dir': out}


def test_round8_batch_modifiers_and_wanted():
    from vantage import batch
    from test_advanced import _args
    for main in (('carla_live', ['5']), ('ground_truth_summary', ['d']), ('detection_quality', 'o'),
                 ('campaign', 'o'), ('critical_carla', 'o')):
        assert batch.wanted(_args(**{main[0]: main[1]})), main
    for mod, v in (('carla_faults', 'f.json'), ('carla_scenario', 'cut_in'), ('carla_ground_truth', 'all'),
                   ('live_fault', ['1:{}']), ('live_cameras', 'F'), ('detection_recording', ['d']),
                   ('detection_download', True), ('campaign_template', 'rig_ab'), ('campaign_phase', 'drive'),
                   ('critical_carla_drives', 'd')):
        assert not batch.wanted(_args(**{mod: v})), mod                  # modifiers, like --variant
        with pytest.raises(SystemExit):
            batch.run(_args(**{mod: v}))                                 # … and never without their option
    with pytest.raises(SystemExit):
        batch.run(_args(carla_python='python3', campaign=None))
    a = batch.parser().parse_args(['--sample', '--carla-live', '5', 'out', '--live-fault', '2:{"tag": "F"}',
                                   '--live-fault', '3:{"tag": "R"}', '--campaign', 'c', '--campaign-phase', 'analyse'])
    assert a.carla_live == ['5', 'out'] and len(a.live_fault) == 2 and a.campaign_phase == 'analyse'


def test_batch_carla_options_ground_truth_summary_and_reproduce(tmp_path, capsys, monkeypatch):
    """--carla-rig and --carla-run carry --carla-scenario, --carla-faults, --carla-actors, --carla-ground-truth,
    --carla-walkers and --carla-spawn-point into the rig, each checked first (exit 2); a fake drive with them, its
    ground truth read back by --ground-truth-summary (no project), which reproduces."""
    from vantage import batch
    from test_advanced import _args
    p = _rig()
    faults = tmp_path / 'faults.json'
    faults.write_text(json.dumps([{'tag': 'F', 'kind': 'rotation', 'deg': [0, 0.5, 0]},
                                  {'tag': 'L1', 'kind': 'range_bias', 'm': 0.05}]))
    actors = json.dumps([{'id': 'c1', 'kind': 'child', 'pose': {'x': 2.5, 'y': 0.0}}])
    out = tmp_path / 'rig'
    assert _batch(tmp_path, p, carla_rig=str(out), carla_scenario='pedestrian_crossing',
                  carla_scenario_params='{"distance_m": 25}', carla_faults=str(faults), carla_actors=actors,
                  carla_ground_truth='semantic,range', carla_walkers=3, carla_spawn_point=7) == 0
    rig = json.loads((out / 'sim_rig.json').read_text())
    assert [(f['tag'], f['kind']) for f in rig['faults']] == [('F', 'rotation'), ('L1', 'range_bias')]
    assert rig['faults'][0]['deg'] == [0.0, 0.5, 0.0]
    assert rig['scenario']['name'] == 'pedestrian_crossing' and rig['scenario']['params']['distance_m'] == 25.0
    assert rig['scenario']['walkers'] == 3
    assert rig['ground_truth']['semantic'] and rig['ground_truth']['range'] and not rig['ground_truth']['instance']
    assert [a['id'] for a in rig['actors']['items']] == ['c1'] and rig['actors']['spawn_point'] == 7
    assert rig['provenance']['settings']['sim']['faults'][0]['tag'] == 'F'          # what the rig was built with
    assert (out / 'faults.json').is_file() and (out / 'actors.json').is_file()
    # each option is checked against the rig before anything is written
    for kw, why in (({'carla_faults': '[{"tag": "NOPE", "kind": "drop", "rate": 0.1}]'}, 'no sensor NOPE'),
                    ({'carla_faults': '[{"tag": "RD1", "kind": "range_bias", "m": 0.1}, '
                                      '{"tag": "F", "kind": "range_bias", "m": 0.1}]'}, 'does not apply to F'),
                    ({'carla_scenario': 'cut_in', 'carla_scenario_params': '{"nope": 1}'}, "no parameter 'nope'"),
                    ({'carla_scenario_params': '{not json'}, 'neither a file nor JSON'),
                    ({'carla_actors': '[{"id": "a", "kind": "dragon", "pose": {"x": 1, "y": 0}}]'}, 'unknown kind'),
                    ({'carla_ground_truth': 'semantic,colour'}, 'unknown colour')):
        assert _batch(tmp_path, p, carla_rig=str(tmp_path / 'bad'), **kw) == 2, kw
        assert why in capsys.readouterr().err, kw
    assert not (tmp_path / 'bad').exists()
    # six fisheyes at full size with ground truth: over the bridge's render budget, refused before anything is written
    big = _rig()
    big.analysis.sim = {'image_scale': 1.0}
    assert _batch(tmp_path, big, carla_run=str(tmp_path / 'big'), carla_python=sys.executable,
                  carla_ground_truth='all') == 2
    assert "over the bridge's budget of 200" in capsys.readouterr().err and not (tmp_path / 'big').exists()
    # --carla-run against the fake: the options reach the drive (a fault injected, ground truth recorded)
    _fake_bridges(monkeypatch)
    q = _small(_rig(), ('F',), lens=NARROW)
    drive = tmp_path / 'drive'
    with _Server() as srv:
        q.analysis.sim['port'] = srv.port
        assert _batch(tmp_path, q, carla_run=str(drive), carla_python=sys.executable, carla_ground_truth='all',
                      carla_faults='[{"tag": "F", "kind": "rotation", "deg": [0, 1, 0]}]', carla_walkers=0) == 0
    err = capsys.readouterr().err
    assert 'ground truth:' in err and 'fault F rotation' in err
    run = json.loads((drive / 'run.json').read_text())
    assert run['status'] == 'ok' and [f['tag'] for f in run['faults_injected']] == ['F']
    assert run['faults_injected'][0]['pose_delta_deg'] == pytest.approx([0.0, 1.0, 0.0], abs=1e-6)
    assert (drive / 'ground_truth' / 'index.json').is_file()
    # its ground truth in brief, with no project; and it reproduces from the drive found by its name
    gt = tmp_path / 'gt.json'
    a = _args(ground_truth_summary=[str(drive), str(gt)])
    a.sample = False
    assert batch.run(a) == 0
    doc = json.loads(gt.read_text())
    assert doc['schema'] == 'vantage.ground_truth_summary/1' and doc['cameras']['F'] > 0 and doc['lidars']['L1'] > 0
    assert doc['faults_injected'][0]['kind'] == 'rotation' and 'plan_pose' not in doc['faults_injected'][0]
    assert doc['lidar_labels']['L1']['matched_in_order_share'] > 0.9 and 'car' in doc['classes']
    pv = doc['provenance']
    assert pv['project'] is None and pv['output']['option'] == '--ground-truth-summary'
    assert pv['inputs'][0]['option'] == '--ground-truth-summary' and pv['inputs'][0]['kind'] == 'folder'
    a = _args(reproduce=[str(gt), str(tmp_path / 'rep.json')])
    a.sample = False
    assert batch.run(a) == 0 and json.loads((tmp_path / 'rep.json').read_text())['verdict'] == 'reproduces'
    a = _args(ground_truth_summary=[str(tmp_path)])                   # a folder without ground truth: exit 2
    a.sample = False
    assert batch.run(a) == 2 and 'has no ground truth' in capsys.readouterr().err


def test_batch_detection_quality_and_its_export_format(gt_drive, tmp_path, capsys, monkeypatch):
    """--detection-quality: 2 with nothing to evaluate, 8 without the model (or a failed download); with a stand-in
    detector that finds every labelled object, recall and precision 1 and its files stamped. The export format
    detection_quality writes the same evaluation when the model and the recordings are there."""
    from vantage.core import detection as DT
    from vantage.core import exporters as ex
    p = _rig()
    p.analysis.detection = {'cache': False, 'bootstrap': 50}
    assert _batch(tmp_path, p, detection_quality=str(tmp_path / 'd0')) == 2           # no recording
    assert 'no recording' in capsys.readouterr().err
    assert _batch(tmp_path, p, detection_quality=str(tmp_path / 'd0'), detection_recording=[str(tmp_path)]) == 2
    assert 'no ground truth' in capsys.readouterr().err
    rec = gt_drive['dir']
    assert _batch(tmp_path, p, detection_quality=str(tmp_path / 'd0'), detection_recording=[rec]) == 8
    assert 'not downloaded' in capsys.readouterr().err

    def fail(*a, **k):
        raise OSError('network unreachable')
    monkeypatch.setattr(DT, 'download', fail)
    assert _batch(tmp_path, p, detection_quality=str(tmp_path / 'd0'), detection_recording=[rec],
                  detection_download=True) == 8
    assert 'the download failed: network unreachable' in capsys.readouterr().err
    # the model there (a stand-in: the wiring is under test)
    real = DT.status
    monkeypatch.setattr(DT, 'status', lambda m=DT.DEFAULT_MODEL, folder=None: dict(real(m, folder), verified=True,
                                                                                   present=True))
    monkeypatch.setattr(DT, 'load', lambda settings=None, folder=None: _oracle(rec))
    out = tmp_path / 'det'
    assert _batch(tmp_path, p, detection_quality=str(out), detection_recording=[rec]) == 0
    doc = json.loads((out / 'detection_quality.json').read_text())
    assert doc['schema'] == 'vantage.detection_quality/1' and doc['provenance']['output']['option'] == \
        '--detection-quality'
    assert doc['provenance']['inputs'][0]['option'] == '--detection-recording'
    rows = [r for r in doc['per_camera'] if r['camera'] == 'all' and r['range_bin'] == 'all' and r['n_gt']]
    assert rows and all(r['recall'] == 1.0 and r['precision'] == 1.0 for r in rows), rows
    assert {r['class'] for r in rows} >= {'car', 'pedestrian'}
    assert 'recall 1.00' in capsys.readouterr().err
    for n in ('detection_quality.csv', 'detection_ranges.csv', 'detection_rig.csv', 'detection_quality.png'):
        assert (out / n).stat().st_size > 0, n
    # the export format: nothing without recordings; the five files with them, the study's settings in provenance
    f = ex.format_by_id('detection_quality')
    assert f.group == 'Documents' and not f.default and ex._names(f, p, p.variant) == []
    p.analysis.detection = {'cache': False, 'bootstrap': 50, 'recordings': [rec]}
    names = ex._names(f, p, p.variant)
    assert names == [f'detection_quality/{n}' for n in ('detection_quality.json', 'detection_quality.csv',
                                                         'detection_ranges.csv', 'detection_rig.csv',
                                                         'detection_quality.png')]
    assert 'The detector' in ex.preview('detection_quality', p, None, ex.ExportOptions(out_dir=str(tmp_path)))[1]
    paths = ex.export(['detection_quality'], p, None, ex.ExportOptions(out_dir=str(tmp_path / 'exp')))
    got = {os.path.relpath(x, tmp_path / 'exp').replace(os.sep, '/') for x in paths}
    assert set(names) <= got and any(g.startswith('detection_quality/detections/') for g in got)
    edoc = json.loads((tmp_path / 'exp' / 'detection_quality' / 'detection_quality.json').read_text())
    assert edoc['provenance']['study'] == 'detection_quality'
    assert edoc['provenance']['settings']['detection']['bootstrap'] == 50


def _campaign(out=None, **kw):
    from vantage.core import simcampaign as SC
    d = {'name': 'Batch drops', 'template': 'custom', 'seeds': [1],
         'fault_sets': [{'kind': 'drop', 'tags': ['R'], 'magnitudes': [20.0]}], 'clean': True,
         'evaluators': ['image_stats', 'returns'], 'base': {'image_scale': 0.08, 'duration_s': 0.8, 'traffic': 4,
                                                            'warmup_s': 0.25, 'tick_hz': 40.0}}
    if out:
        d['out_dir'] = out
    d.update(kw)
    return SC.CampaignSettings.from_dict(d).to_dict()


def test_batch_campaign_through_the_fake_bridge(tmp_path, capsys, monkeypatch):
    """--campaign: the project's campaign driven (each drive a bridge process against the fake carla), checked,
    analysed and exported (stamped); run again it reuses every drive; --campaign-phase analyse needs no server; the
    export format sim_campaign writes the analysis from the drives on disk. Exit 2 for a design naming a sensor the
    variant lacks, 7 without a server, 8 when its detection check has no model, 9 when a drive fails; it cannot be
    reproduced (it drives CARLA)."""
    from vantage import batch
    from vantage.core import exporters as ex
    from test_advanced import _args
    _fake_bridges(monkeypatch)
    p = _small(_rig(), ('F', 'R'), lens=NARROW)
    p.analysis.campaign = _campaign()
    out = tmp_path / 'camp'
    with _Server() as srv:
        p.analysis.sim['port'] = srv.port
        assert _batch(tmp_path, p, campaign=str(out), carla_python=sys.executable) == 0
        err = capsys.readouterr().err
        assert '2 of 2 drives done, 0 failed' in err
        doc = json.loads((out / 'campaign.json').read_text())
        assert doc['schema'] == 'vantage.sim_campaign/1' and doc['provenance']['output']['option'] == '--campaign'
        assert doc['provenance']['settings']['campaign']['name'] == 'Batch drops'
        for n in ('campaign_observations.csv', 'campaign_drives.csv', 'campaign_report.html'):
            assert (out / n).stat().st_size > 0, n
        runs = sorted(os.listdir(out / 'drives' / 'drives'))
        assert runs == ['d0001', 'd0002']
        drop = json.loads((out / 'drives' / 'drives' / 'd0002' / 'run.json').read_text())['faults_injected']
        assert drop[0]['tag'] == 'R' and drop[0]['kind'] == 'drop' and drop[0]['dropped_frames']
        mtime = os.path.getmtime(out / 'drives' / 'drives' / 'd0002' / 'run.json')
        assert _batch(tmp_path, p, campaign=str(out), carla_python=sys.executable) == 0     # everything reused
        assert os.path.getmtime(out / 'drives' / 'drives' / 'd0002' / 'run.json') == mtime
    capsys.readouterr()
    assert _batch(tmp_path, p, campaign=str(out), campaign_phase='analyse') == 0             # no server needed
    assert '2 of 2 drives done' in capsys.readouterr().err
    a = _args(reproduce=[str(out / 'campaign.json'), str(tmp_path / 'rep.json')])
    a.path, a.sample = None, True
    assert batch.run(a) == 4                                       # it drives CARLA: not run again
    # the export format: the campaign's drives on disk, analysed when exported (nothing driven)
    f = ex.format_by_id('sim_campaign')
    assert f.group == 'Documents' and not f.default
    q = p.snapshot()
    q.analysis.campaign = _campaign(str(out / 'drives'))
    names = ex._names(f, q, q.variant)
    assert names[0] == 'sim_campaign/campaign.json' and 'sim_campaign/campaign_report.html' in names
    assert '2 of 2 drives done' in ex.preview('sim_campaign', q, None, ex.ExportOptions(out_dir=str(tmp_path)))[1]
    paths = ex.export(['sim_campaign'], q, None, ex.ExportOptions(out_dir=str(tmp_path / 'exp')))
    got = {os.path.relpath(x, tmp_path / 'exp').replace(os.sep, '/') for x in paths}
    assert set(names) <= got
    edoc = json.loads((tmp_path / 'exp' / 'sim_campaign' / 'campaign.json').read_text())
    assert edoc['provenance']['study'] == 'sim_campaign' and len(edoc['drives']) == 2
    q.analysis.campaign = _campaign(str(tmp_path / 'no-drives'))
    assert ex._names(f, q, q.variant) == [] and 'No drive' in ex.preview('sim_campaign', q, None,
                                                                          ex.ExportOptions(out_dir=str(tmp_path)))[1]
    # the exits: a sensor the variant lacks (2), no server (7), the detection check without its model (8), a failed
    # drive (9)
    p.analysis.campaign = _campaign(fault_sets=[{'kind': 'drop', 'tags': ['NOPE'], 'magnitudes': [5.0]}])
    assert _batch(tmp_path, p, campaign=str(tmp_path / 'c2')) == 2
    assert 'no visible sensor NOPE' in capsys.readouterr().err
    p.analysis.campaign = _campaign()
    p.analysis.sim['port'] = _free_port()
    assert _batch(tmp_path, p, campaign=str(tmp_path / 'c3'), carla_python=sys.executable) == 7
    assert 'No CARLA server answers' in capsys.readouterr().err
    p.analysis.campaign = _campaign(evaluators=['detection'])
    assert _batch(tmp_path, p, campaign=str(tmp_path / 'c4'), carla_python=sys.executable) == 8
    p.analysis.campaign = _campaign()
    from vantage.sim import server as SV
    monkeypatch.setattr(SV, 'bridge_command', lambda python, rig, o, args=(), **kw: (
        [sys.executable, '-c', 'import sys; sys.exit(3)'], dict(os.environ)))
    with _Server() as srv:
        p.analysis.sim['port'] = srv.port
        assert _batch(tmp_path, p, campaign=str(tmp_path / 'c5'), carla_python=sys.executable) == 9
    assert '0 of 2 drives done, 2 failed' in capsys.readouterr().err


def test_batch_critical_targets_in_carla(tmp_path, capsys, monkeypatch):
    """--critical-carla: 2 with no target, or no drive in --critical-carla-drives; 7 with no server; one target
    stood in the fake CARLA, compared and exported (stamped); compared again from its drives without CARLA, which
    reproduces."""
    from vantage import batch
    from test_advanced import _args
    p = _small(_rig(), ('F',))
    assert _batch(tmp_path, p, critical_carla=str(tmp_path / 'c0')) == 2
    assert 'no critical target' in capsys.readouterr().err
    p = _with_targets(p, 'Child 2 m ahead')
    p.analysis.critical_carla = {'image_scale': 0.08, 'spawn_points': [3], 'frames': 2, 'duration_s': 1.0}
    (tmp_path / 'empty').mkdir()
    assert _batch(tmp_path, p, critical_carla=str(tmp_path / 'c1'), critical_carla_drives=str(tmp_path / 'empty')) == 2
    p.analysis.sim['port'] = _free_port()
    assert _batch(tmp_path, p, critical_carla=str(tmp_path / 'c2'), carla_python=sys.executable) == 7
    assert 'No CARLA server' in capsys.readouterr().err
    _fake_bridges(monkeypatch)
    out = tmp_path / 'crit'
    with _Server() as srv:
        p.analysis.sim['port'] = srv.port
        code = _batch(tmp_path, p, critical_carla=str(out), carla_python=sys.executable)
    err = capsys.readouterr().err
    doc = json.loads((out / 'critical_carla.json').read_text())
    assert doc['schema'] == 'vantage.critical_carla/1' and doc['provenance']['output']['option'] == '--critical-carla'
    agree = doc['metrics']['agree'] == doc['metrics']['pairs']
    assert code == (0 if agree else 3) and 'camera views compared in 1 drives' in err
    assert (out / 'critical_carla.csv').is_file() and (out / 'critical_carla.png').is_file()
    assert [d for d in os.listdir(out / 'drives') if os.path.isdir(out / 'drives' / d)]
    # the drives compared again without CARLA: the same verdicts, and that result reproduces from its drives (found by
    # their folder's name beside the project)
    import shutil
    shutil.copytree(out / 'drives', tmp_path / 'drives')
    again = tmp_path / 'again'
    assert _batch(tmp_path, p, critical_carla=str(again), critical_carla_drives=str(tmp_path / 'drives')) == code
    d2 = json.loads((again / 'critical_carla.json').read_text())
    assert d2['metrics']['agree'] == doc['metrics']['agree'] and d2['metrics']['pairs'] == doc['metrics']['pairs']
    src = tmp_path / 'p.vantage'
    a = _args(reproduce=[str(again / 'critical_carla.json'), str(tmp_path / 'rep.json')])
    a.path, a.sample = str(src), False
    assert batch.run(a) == 0 and json.loads((tmp_path / 'rep.json').read_text())['verdict'] == 'reproduces'
    a = _args(reproduce=[str(out / 'critical_carla.json'), str(tmp_path / 'rep2.json')])
    a.path, a.sample = str(src), False
    assert batch.run(a) == 4                                        # the run that drove CARLA is not driven again


def test_batch_carla_live_session(tmp_path, capsys, monkeypatch):
    """--carla-live: the cameras live in the fake CARLA, rig health on their streams, a fault injected on time: a
    frozen camera is a fault (exit 3), rig_health.json stamped; bad options 2, no server 7; never reproduced."""
    import dataclasses
    from vantage import batch
    from test_advanced import _args
    _fake_bridges(monkeypatch)
    p = _rig()
    p.modules['sturdecam31'] = dataclasses.replace(p.modules['sturdecam31'], fps=10.0)
    _small(p, ('F', 'R'), lens=NARROW, traffic=0, warmup_s=0.1)
    assert _batch(tmp_path, p, carla_live=['2', str(tmp_path / 'l0')], live_cameras='F,NOPE') == 2
    assert 'no visible camera NOPE' in capsys.readouterr().err
    assert _batch(tmp_path, p, carla_live=['2', str(tmp_path / 'l0')], live_fault=['1:{"tag": "F", "kind": "melt"}'],
                  carla_python=sys.executable) == 2
    assert 'Unknown fault kind' in capsys.readouterr().err
    p.analysis.sim['port'] = _free_port()
    assert _batch(tmp_path, p, carla_live=['2', str(tmp_path / 'l1')], carla_python=sys.executable) == 7
    assert 'No CARLA server' in capsys.readouterr().err
    out = tmp_path / 'live'
    with _Server() as srv:
        p.analysis.sim['port'] = srv.port
        code = _batch(tmp_path, p, carla_live=['3', str(out)], live_scale=0.08, carla_python=sys.executable,
                      live_fault=['1:{"tag": "R", "kind": "freeze"}'])
    err = capsys.readouterr().err
    assert code == 3 and 'fault at' in err and 'R    fault' in err
    h = json.loads((out / 'rig_health.json').read_text())
    assert h['cameras']['R']['status'] == 'fault' and any('frozen' in r for r in h['cameras']['R']['reasons'])
    assert h['cameras']['F']['drift_against'] == 'carla_body'
    assert h['provenance']['output']['option'] == '--carla-live'
    live = json.loads((out / 'live.json').read_text())
    assert live['fault_log'][0]['tag'] == 'R' and (out / 'rig' / 'sim_rig.json').is_file()
    a = _args(reproduce=[str(out / 'rig_health.json'), str(tmp_path / 'rep.json')])
    a.path, a.sample = str(tmp_path / 'p.vantage'), False
    assert batch.run(a) == 4


def test_assistant_reads_round8(gt_drive, tmp_path):
    """The read-only tools of round 8: the pages' results when there are, else what is set up; a drive's ground truth
    inside the working folder only; no URL or machine path ever."""
    import shutil
    from vantage.assistant import tools as T
    from vantage.core.analysis import Engine
    p = _rig()
    for name in ('get_detection_quality', 'get_campaign', 'get_critical_carla', 'get_carla_live', 'get_ground_truth'):
        assert T.BY_NAME[name].kind == 'read', name
    tc = T.ToolContext(p.snapshot(), None, Engine(), working_dir=str(tmp_path))
    d = json.loads(T.run(tc, 'get_detection_quality', {})[0])
    assert d['evaluated'] is False and d['model']['verified'] is False and 'not run yet' in d['note']
    assert str(tmp_path) not in json.dumps(d)
    c = json.loads(T.run(tc, 'get_campaign', {})[0])
    assert c['run'] is False and c['template'] == 'fault_detectability' and c['drives'] == 49 and c['cost']
    assert {f['kind'] for f in c['fault_sets']} == {'rotation', 'clock_offset', 'drop', 'freeze'}
    q = _with_targets(p.snapshot(), 'Child 2 m ahead', 'Kerb by the right front wheel')
    cc = json.loads(T.run(T.ToolContext(q, None, Engine()), 'get_critical_carla', {})[0])
    assert cc['compared'] is False and [t['stood_in'] for t in cc['targets']] == [True, False] and cc['drives'] >= 1
    assert json.loads(T.run(tc, 'get_carla_live', {})[0])['state'] == 'off'
    # the pages' results, as the controller hands them over (their summary(): no paths; a message may have one)
    home = os.path.expanduser('~')
    page = {'campaign': 'X', 'notes': [f'drive d0001 failed: see {home}/w/drives/d0001/bridge.log']}
    tp = T.ToolContext(p.snapshot(), None, Engine(), campaign=page, detection={'recordings': []},
                       critical_carla={'agreement_pct': 100.0},
                       carla_live={'state': 'streaming', 'message': f'see {home}/x/live.log',
                                   'cameras': {'F': {'reasons': ['no frame from http://127.0.0.1:5/F.mjpg']}}})
    out = json.loads(T.run(tp, 'get_campaign', {})[0])
    assert out['run'] and home not in json.dumps(out) and 'bridge.log' in json.dumps(out)
    assert json.loads(T.run(tp, 'get_detection_quality', {})[0])['evaluated'] is True
    assert json.loads(T.run(tp, 'get_critical_carla', {})[0])['agreement_pct'] == 100.0
    lv = json.loads(T.run(tp, 'get_carla_live', {})[0])
    assert 'http' not in json.dumps(lv) and home not in json.dumps(lv) and lv['state'] == 'streaming'
    # a drive's ground truth: inside the working folder only
    shutil.copytree(gt_drive['dir'], tmp_path / 'proj' / 'carla' / 'drive1')
    g = json.loads(T.run(tc, 'get_ground_truth', {'run': 'proj/carla/drive1'})[0])
    assert g['cameras'] == {'F': g['cameras']['F']} and g['objects'] > 0 and g['run'] == os.path.join(
        'proj', 'carla', 'drive1')
    assert {'car', 'pedestrian'} <= set(g['objects_by_class'])
    assert 'error' in json.loads(T.run(tc, 'get_ground_truth', {'run': gt_drive['dir']})[0])     # outside it
    assert 'error' in json.loads(T.run(tc, 'get_ground_truth', {'run': 'proj'})[0])               # no ground truth


def test_experiments_detection_responses_drive_carla_per_point(gt_drive, tmp_path, monkeypatch):
    """detection.recall and detection.rig_recall: in the catalogue with their parameters, CARLA-backed (an
    experiment's estimate states their measured cost instead of driving CARLA); without the model each point says why;
    with it every point is a drive (here the fake bridge in-process) evaluated by a stand-in detector."""
    import importlib.util
    from vantage import workdir
    from vantage.core import detection as DT
    from vantage.core import detection_eval as DE
    from vantage.core import experiments as EX
    from vantage.sim import carla_bridge as CB
    p = gt_drive['p'].snapshot()
    cat = {r['id']: r for r in EX.response_catalogue(p) if r['study'] == 'detection'}
    assert set(cat) == {'detection.recall', 'detection.rig_recall'} and cat['detection.recall']['cost'] == 'study'
    assert [q['key'] for q in cat['detection.recall']['params']] == ['cls', 'range', 'camera']
    assert {c['value'] for c in cat['detection.recall']['params'][0]['choices']} >= {'vehicle', 'pedestrian', 'car'}
    F = p.variant.by_tag('F')
    st = EX.ExperimentSettings.from_dict({
        'design': 'oat', 'levels': 2, 'workers': 2, 'bootstrap': 0,
        'factors': [dict(EX.new_factor(p, p.variant, 'camera', F.uid, 'pitch'), lo=F.pitch, hi=F.pitch + 3.0,
                         levels=2, mirror=False)],
        'responses': [{'metric': 'detection.recall', 'params': {'cls': 'car'}},
                      {'metric': 'detection.rig_recall', 'params': {'cls': 'pedestrian'}}]})
    assert EX.errors(p, st) == []

    def no_carla(*a, **k):
        raise AssertionError('the estimate drove CARLA')
    monkeypatch.setattr(DE, 'evaluate_variant', no_carla)
    est = EX.estimate(p, None, st)
    assert est['per_point_s'] >= EX.CARLA_STUDY_S['detection'] and est['workers'] >= 1
    assert any(s['id'] == 'detection' and s.get('typical') for s in est['studies'])
    assert est['total_s'] >= est['points'] * EX.CARLA_STUDY_S['detection']    # one server: the points queue
    monkeypatch.undo()
    # no model: every point says why, nothing driven
    monkeypatch.setattr(DE, 'evaluate_variant', no_carla)
    res = EX.run(p, None, st)
    assert np.isnan(res.Y).all() and 'not downloaded' in ' '.join(res.reasons.values())
    monkeypatch.undo()
    # the model there: each design point driven (in-process against the fake) and evaluated
    name = 'fake_carla_for_tests'
    fc = sys.modules.get(name)
    if fc is None:
        spec = importlib.util.spec_from_file_location(name, os.path.join(FAKE_DIR, 'carla', '__init__.py'))
        fc = importlib.util.module_from_spec(spec)
        sys.modules[name] = fc
        spec.loader.exec_module(fc)
    drives = []

    def runner(rig_json, out, progress, cancel):
        drives.append(out)
        fc.reset()
        doc, remaps = CB.load_rig(rig_json)
        a = CB.resolve_args(CB.parser().parse_args([rig_json, out]), doc)
        CB.Bridge(fc, doc, remaps, out, a, None, cancel).run()
    real = DE.evaluate_variant

    def via_fake(project, variant, sim, det, out, **kw):
        kw.pop('runner', None)
        return real(project, variant, sim, det, out, runner=runner, detector=_oracle(out), **kw)
    monkeypatch.setattr(DE, 'evaluate_variant', via_fake)
    real_status = DT.status
    monkeypatch.setattr(DT, 'status', lambda m=DT.DEFAULT_MODEL, folder=None: dict(real_status(m, folder),
                                                                                   verified=True, present=True))
    monkeypatch.setattr(workdir, 'default', lambda: str(tmp_path / 'work'))
    p.analysis.detection = {'cache': False, 'bootstrap': 50}
    res = EX.run(p, None, st)
    assert res.n == 2 and len(drives) == 2 and all(d.startswith(str(tmp_path / 'work')) for d in drives)
    assert np.allclose(res.Y, 100.0), (res.Y, res.reasons)          # the stand-in finds every labelled object
    res2 = EX.run(p, None, st, cache=EX.PointCache())               # a re-run (no cached points) reuses both drives
    assert len(drives) == 2 and np.allclose(res2.Y, 100.0)


def test_round8_hooks_carla_guards_and_live_drift(app, win, gt_drive, tmp_path, monkeypatch):
    """The CARLA page embeds the live rig panel (Simulate ▸ CARLA live rig shows it); the critical-targets page opens
    Critical targets in CARLA; after a run with ground truth the CARLA page opens it in Detection quality; one CARLA
    job at a time (a recording, the live rig, a campaign, critical targets) and stopping the server stops the live
    rig; the Live rig page measures the drift of a live CARLA stream against CARLA's own car."""
    import cv2
    from vantage.core import carla_live as CL
    from vantage.ui.workspaces.sim_live import SimLivePanel
    win.actions_by_id['carla_live'].trigger()
    pump(app, 0.3)
    sim = win.page('sim')
    assert win.page_name() == 'sim' and isinstance(sim.live, SimLivePanel) and sim.live.isVisible()
    assert sim.live.host is sim and sim.right.sizes()[2] > 0
    # Analysis ▸ Critical targets ▸ In CARLA…
    win.set_page('critical')
    pump(app, 0.2)
    win.page('critical').b_carla.click()
    pump(app, 0.3)
    assert win.page_name() == 'critical_carla'
    # the CARLA page after a run with ground truth: Detection quality opens it there
    win.set_page('sim')
    sim.python = sys.executable
    sim.result, sim.out_dir = {'status': 'ok'}, gt_drive['dir']
    sim._update_buttons()
    assert sim.b_detect.isEnabled() and sim.b_drivelog.isEnabled()
    n = win.store.undo.count()
    sim.b_detect.click()
    pump(app, 0.3)
    det = win.page('detection')
    assert win.page_name() == 'detection' and det.settings.recordings == [os.path.abspath(gt_drive['dir'])]
    assert win.store.undo.count() == n + 1                                # one undoable edit
    sim.result, sim.out_dir = None, ''
    win.set_page('sim')
    sim._update_buttons()
    assert not sim.b_detect.isEnabled()
    # one CARLA job at a time
    camp, crit = win.page('campaign'), win.page('critical_carla')
    assert win.carla_user() == ''
    sim.live.state = 'streaming'                                           # (as while streaming)
    try:
        assert not sim.run() and 'Live streaming is on' in sim.g_status.text()
        assert not camp.run() and 'live rig is streaming' in camp.status_text
        assert not crit.run() and 'live rig is streaming' in crit.status_text
        stopped = []
        monkeypatch.setattr(sim.live, 'stop', lambda why='stopped': stopped.append(why) or True)
        sim.stop_server()                                                  # the live rig goes first
        assert stopped == ['the server is stopping']
    finally:
        sim.live.state = 'off'
    camp.busy = True
    try:
        assert 'simulation campaign is running' in win.carla_user('sim')
        assert not sim.run() and 'simulation campaign' in sim.g_status.text()
        assert not sim.live.start() and 'simulation campaign' in sim.live.status_line.text.text()
        assert not crit.run()
        assert win.carla_user('campaign') == ''
    finally:
        camp.busy = False
    sim.busy = True
    try:
        assert not camp.run() and 'CARLA recording is running' in camp.status_text
    finally:
        sim.busy = False
    exp = win.page('experiments')                       # an experiment counts only while it drives CARLA
    exp.busy, exp.uses_carla = True, False
    try:
        assert win.carla_user('sim') == ''
        exp.uses_carla = True
        assert 'designed experiment' in win.carla_user('sim')
        assert not sim.run() and 'designed experiment' in sim.g_status.text()
        assert win.carla_user('experiments') == ''
    finally:
        exp.busy, exp.uses_carla = False, False
    # the Live rig page: a camera streamed by the live rig is compared with CARLA's own car (its body mask)
    m = np.full((96, 120), 128, np.uint8)                                 # in the image (the lens circle)
    m[60:, :] = 255                                                       # CARLA's bonnet in the lower part
    ok, png = cv2.imencode('.png', m)
    ev = CL.body_view(png.tobytes(), (1920, 1536), ('carla', 9, 'F'))
    url = 'http://127.0.0.1:9/F.mjpg'
    win.store.edit('streams', lambda pj: pj.streams.__setitem__('F', url), 'live test stream')
    sim.live.state, sim.live._urls, sim.live.carla_views = 'streaming', {'F': url}, {'F': ev}
    info = sim.live.info
    try:
        win.set_page('rig')
        rig = win.page('rig')
        pump(app, 0.2)
        assert rig._live_views() == {'F': ev}
        rig.show_edge.setChecked(True)
        rig.views.pop('F', None)
        _wait(app, lambda: (rig._drift_step(), rig.views.get('F') is ev)[1], 30)
        assert rig.views['F'].horizon == [] and rig.views['F'].contours           # CARLA's outline, no horizon
        # the sync skew of live CARLA streams is judged against the live rig's own schedule, not the plan's stagger
        kept = dict(win.store.project.streams)
        win.store.edit('streams', lambda pj: setattr(pj, 'streams', {'F': url}), 'only the live stream')
        sim.live.info = dict(sim.live.info or {}, schedule={'F': (0.0, 100.0)})
        assert rig._planned() == {'F': (0.0, 100.0)}
        rig._judge_against()
        assert rig.monitor.plan_mode == 'given' and rig.monitor.planned == {'F': (0.0, 100.0)}
        rig.snap = {'cameras': {'F': {'sync_skew_ms': 0.0}, 'R': {'sync_offset_ms': 33.0, 'sync_planned_ms': 33.3,
                                                                  'sync_skew_ms': -0.3}},
                    'rig': {'sync_reference': 'F', 'sync_against': 'plan'}}
        tip = rig._sync_tip(rig.snap['rig'])
        assert 'live rig\'s schedule; reference F' in tip and 'R: offset +33.0 ms, planned +33.3 ms, skew -0.3 ms' in tip
        win.store.edit('streams', lambda pj: setattr(pj, 'streams', dict(kept, F='synthetic')), 'not live')
        assert rig._live_views() == {} and rig._planned() == 'plan'         # the stream is not the live rig's
        rig._judge_against()
        assert rig.monitor.plan_mode == 'plan'
        rig.snap = None
    finally:
        sim.live.state, sim.live._urls, sim.live.carla_views, sim.live.info = 'off', {}, {}, info
    # the assistant reads the pages through its context
    tc = win.ctx.assistant._context()
    assert tc.carla_live['state'] == 'off' and tc.detection is None and tc.campaign is None
    win.set_page('viewports')
    settle(app, win)


def test_format_page_and_docs_name_round8():
    """docs/project-format.md names every key of the new settings; README and docs.html name every new batch flag
    and exit codes 8 and 9."""
    import re
    from vantage import batch
    from vantage.core import critical_carla as CC
    from vantage.core import detection_eval as DE
    from vantage.core import simcampaign as SC
    from vantage.core import simrig as SR
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fmt = open(os.path.join(root, 'docs', 'project-format.md'), encoding='utf-8').read()
    named = set(re.findall(r'`([A-Za-z_][\w]*)`', fmt))
    for st in (DE.DetectionSettings(), SC.CampaignSettings(), CC.CriticalCarlaSettings(), SR.SimSettings()):
        assert not [k for k in st.to_dict() if k not in named], type(st).__name__
    readme = open(os.path.join(root, 'README.md'), encoding='utf-8').read()
    docs = open(os.path.join(root, 'site', 'docs.html'), encoding='utf-8').read()
    flags = [a.option_strings[-1] for a in batch.parser()._actions
             if a.option_strings and a.dest in ('carla_live', 'live_cameras', 'live_scale', 'live_fault',
                                                 'detection_quality', 'detection_recording', 'detection_download',
                                                 'campaign', 'campaign_template', 'campaign_phase', 'critical_carla',
                                                 'critical_carla_drives', 'ground_truth_summary', 'carla_scenario',
                                                 'carla_scenario_params', 'carla_faults', 'carla_actors',
                                                 'carla_ground_truth', 'carla_walkers', 'carla_spawn_point')]
    assert len(flags) == 20
    for f in flags:
        assert f in readme and f in docs, f
    assert '<td><code>8</code></td>' in docs and '<td><code>9</code></td>' in docs
    assert '8 with\n`--detection-quality`' in readme and '9 with `--campaign`' in readme
    for sec in ('carla-gt', 'carla-live', 'detection', 'campaigns', 'critical-carla'):
        assert f'id="{sec}"' in docs and f'href="#{sec}"' in docs, sec
