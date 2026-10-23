"""The rig studies wired into the rest of the app: each page opens from its menu, each export format writes its files,
batch mode runs each flag with its exit code, and the assistant reads each study (the studies themselves are tested
in their own files: test_power, test_gnss, test_exposure, test_brackets, test_fleet, test_dataset, test_drivelog,
test_lidar_calib, test_requirements)."""
import json
import os
import time

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
