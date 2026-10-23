"""The rig studies wired into the rest of the app: each page opens from its menu, each export format writes its files,
batch mode runs each flag with its exit code, and the assistant reads each study (the studies themselves are tested
in their own files: test_power, test_gnss, test_exposure, test_brackets, test_fleet, test_dataset, test_drivelog,
test_lidar_calib, test_requirements)."""
import json
import os

import pytest

from vantage.core import samples, sensors
from vantage.core.project import Sensor

# action id -> (page name, menu)
PAGES = {'power_budget': ('power', 'Analysis'), 'gnss_imu': ('gnss', 'Analysis'), 'sun_dirt': ('exposure', 'Analysis'),
         'mount_brackets': ('brackets', 'Analysis'), 'requirements': ('requirements', 'Analysis'),
         'sample_dataset': ('dataset', 'Export'), 'drive_log': ('drivelog', 'Calibrate'),
         'fleet': ('fleet', 'Calibrate'), 'lidar_calib': ('lidar_calib', 'Calibrate')}
FORMATS = {'power_budget': ['power_budget/power_budget.json', 'power_budget/power_budget.csv',
                            'power_budget/power_circuits.csv'],
           'gnss_imu': ['gnss_imu/gnss_imu.json', 'gnss_imu/ins_lever_arms.txt', 'gnss_imu/gnss_sky.png'],
           'exposure': ['exposure/exposure.json', 'exposure/exposure.png'],
           'fleet': ['fleet/fleet.json', 'fleet/fleet.csv', 'fleet/fleet.png'],
           'brackets': ['brackets/brackets.json', 'brackets/brackets_sheet.png'],
           'requirements': ['requirements/requirements.json', 'requirements/requirements.csv',
                            'requirements/review_pack.pdf']}


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


def test_assistant_reads_the_rig_studies():
    from vantage.assistant import tools as T
    from vantage.core.analysis import Engine
    p = _rig()
    tc = T.ToolContext(p.snapshot(), None, Engine())
    for name in ('get_power_budget', 'get_gnss_imu', 'get_sun_and_dirt', 'get_fleet', 'get_brackets',
                 'get_sample_dataset', 'get_drive_log_check', 'get_requirements', 'get_requirement_metrics',
                 'plan_lidar_calibration'):
        assert name in T.BY_NAME, name
        out = json.loads(T.run(tc, name, {})[0])
        assert 'error' not in out, (name, out)
    g = json.loads(T.run(tc, 'get_gnss_imu', {'antenna': 'GN2'})[0])
    assert g['antenna_detail']['tag'] == 'GN2'
    assert json.loads(T.run(tc, 'get_drive_log_check', {})[0])['checked'] is False
    # the assistant can open each page
    desc = T.BY_NAME['show_in_app'].description
    assert all(page in desc for page, _ in PAGES.values())


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
