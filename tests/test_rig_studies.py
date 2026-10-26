"""The rig studies wired into the rest of the app: each page opens from its menu, each export format writes its files,
batch mode runs each flag with its exit code, and the assistant reads each study (the studies themselves are tested
in their own files: test_power, test_gnss, test_exposure, test_brackets, test_fleet, test_dataset, test_drivelog,
test_lidar_calib, test_requirements, test_critical, test_materials, test_weather, test_automation, test_terrain,
test_harness, test_observability)."""
import json
import os
import sys
import time

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
         'cable_harness': ('harness', 'Analysis'), 'observability': ('observability', 'Calibrate')}
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
                             'observability/observability.png', 'observability/fusion_density.png']}


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
