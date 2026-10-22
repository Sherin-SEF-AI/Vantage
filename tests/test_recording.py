"""Recording budget: data rates of cameras, lidars and radars against known figures, link parsing and loads, the
recorder's disks and uplink, the findings and Issues, the settings, the export and the page (offscreen)."""
import copy
import csv
import json
import math
import time

import pytest

from vantage.core import recording as RB, rules, samples, sensors
from vantage.core.project import Sensor

RAW_MBPS = 1920 * 1536 * 30 * 16 / 1e6          # 1415.6 Mbit/s: one camera, raw YUV 4:2:2, before any overhead
OS1_MBPS = 1_310_720 * 12 * 8 / 1e6             # 125.8 Mbit/s: an OS1-128's packets


@pytest.fixture(scope='module')
def base():
    return samples.tigor_sample_project()


@pytest.fixture()
def p(base):
    return copy.deepcopy(base)


def _add(p, kind, model_id, tag, **kw):
    m = p.sensor_models[model_id]
    s = Sensor(kind=kind, model_id=model_id, tag=tag, **sensors.default_pose(kind, m, p.vehicle.mesh, p.rig))
    for k, v in kw.items():
        setattr(s, k, v)
    p.variant.sensors.append(s)
    return s


def _settings(**kw):
    return RB.RecordingSettings.from_dict(kw)


# ----------------------------------------------------------------------------- data rates
def test_a_raw_camera_records_its_pixels(p):
    res = RB.budget(p)
    cams = [c for c in p.variant.cameras if c.visible]
    assert [r['tag'] for r in res.rows] == [c.tag for c in cams] and len(cams) == 6
    f = res.row('F')
    assert f['kind'] == 'camera' and f['model'] == 'STURDeCAM31' and f['slot'] == p.variant.by_tag('F').slot
    assert f['mbps'] == pytest.approx(RAW_MBPS) and round(f['mbps'], 1) == 1415.6
    assert f['mb_s'] == pytest.approx(RAW_MBPS / 8) and f['gb_h'] == pytest.approx(RAW_MBPS / 8 * 3.6)
    # its GMSL link carries the raw pixels with the link overhead, the existing budget's figure
    assert f['link_mbps'] == pytest.approx(rules.camera_gbps(p, p.variant.by_tag('F')) * 1000)
    assert f['link_capacity_mbps'] == pytest.approx(p.ecu.link_gbps * 1000)
    assert f['load'] == pytest.approx(RAW_MBPS * 1.10 / 6000) and f['link_severity'] == 'good'
    # totals: six cameras, GNSS/IMU/CAN, then the container overhead on all of it
    t = res.totals
    assert t['by_kind_mbps'] == {'camera': pytest.approx(6 * RAW_MBPS), 'lidar': 0, 'radar': 0}
    assert t['mbps'] == pytest.approx((6 * RAW_MBPS + 2.0) * 1.03)
    assert t['mb_s'] == pytest.approx(t['mbps'] / 8) and t['gb_h'] == pytest.approx(t['mb_s'] * 3.6)
    assert t['tb_shift'] == pytest.approx(t['gb_h'] * 8 / 1000)
    # a hidden camera records nothing
    p.variant.by_tag('F').visible = False
    assert RB.budget(p).row('F') is None and RB.budget(p).totals['counts']['camera'] == 5


def test_compressed_cameras_and_the_ecu_pixel_format(p):
    jpeg = RB.budget(p, settings=_settings(camera_format='jpeg')).row('F')
    h265 = RB.budget(p, settings=_settings(camera_format='h265', h265_bpp=0.1)).row('F')
    assert jpeg['mbps'] == pytest.approx(RAW_MBPS / 16 * 2.0) and 'JPEG' in jpeg['format']
    assert h265['mbps'] == pytest.approx(RAW_MBPS / 16 * 0.1)
    assert jpeg['link_mbps'] == pytest.approx(RAW_MBPS * 1.10)        # the link still carries raw pixels
    p.ecu.bpp = 12                                                     # RAW12: raw recording and link both follow
    f = RB.budget(p).row('F')
    assert f['mbps'] == pytest.approx(RAW_MBPS * 12 / 16) and f['link_mbps'] == pytest.approx(RAW_MBPS * 12 / 16 * 1.1)


def test_lidar_packets_points_and_radars(p):
    _add(p, 'lidar', 'ouster_os1_128', 'L1', port='eth1')
    _add(p, 'radar', 'radar_long', 'RD1')
    _add(p, 'radar', 'radar_imaging', 'RD2')
    res = RB.budget(p)
    l1 = res.row('L1')
    assert l1['kind'] == 'lidar' and l1['model'] == 'Ouster OS1-128 (Rev 7)' and l1['port'] == 'eth1'
    assert l1['mbps'] == pytest.approx(OS1_MBPS) and round(l1['mbps'], 1) == 125.8
    assert l1['link'] == '1000BASE-T' and l1['link_capacity_mbps'] == 1000.0
    assert l1['load'] == pytest.approx(OS1_MBPS / 1000) and l1['link_severity'] == 'good'
    rd1, rd2 = res.row('RD1'), res.row('RD2')
    assert rd1['mbps'] == pytest.approx(1.0) and rd1['link_capacity_mbps'] == 100.0      # CAN-FD / 100BASE-T1: fastest
    assert rd2['mbps'] == pytest.approx(30.0) and (rd1['slot'], rd2['slot']) == (0, 1)
    t = res.totals
    assert t['uplink_mbps'] == pytest.approx(OS1_MBPS + 31.0)
    assert t['uplink_load'] == pytest.approx(t['uplink_mbps'] / 1e4)
    assert t['mbps'] == pytest.approx((6 * RAW_MBPS + OS1_MBPS + 31.0 + 2.0) * 1.03)
    # decoded clouds: 32 bytes a point; the link still carries the packets
    pts = RB.budget(p, settings=_settings(lidar_format='points')).row('L1')
    assert pts['mbps'] == pytest.approx(1_310_720 * 32 * 8 / 1e6) and pts['link_mbps'] == pytest.approx(OS1_MBPS)
    # radars left out of the recording: their rows stay, recording nothing, and off the uplink
    res = RB.budget(p, settings=_settings(include_radars=False))
    assert not res.row('RD2')['recorded'] and res.row('RD2')['mbps'] == 0.0 and res.row('RD2')['load'] is not None
    assert res.totals['uplink_mbps'] == pytest.approx(OS1_MBPS) and res.totals['by_kind_mbps']['radar'] == 0.0
    # a hidden sensor records nothing
    p.variant.sensors[0].visible = False
    assert RB.budget(p).row('L1') is None


def test_hours_per_disk_set_is_capacity_over_rate(p):
    st = _settings(storage_tb=30.0, extra_mbps=0.0, container_overhead=0.0)
    t = RB.budget(p, settings=st).totals
    rate = 6 * RAW_MBPS * 1e6 / 8                                     # bytes/s
    assert t['hours_per_set'] == pytest.approx(30e12 / rate / 3600)
    assert t['sets_per_shift'] == math.ceil(t['tb_shift'] / 30.0) == 2
    # nothing visible: nothing recorded, the disk never fills
    for c in p.variant.cameras:
        c.visible = False
    res = RB.budget(p, settings=_settings(extra_mbps=0.0))
    assert res.totals['mbps'] == 0.0 and math.isinf(res.totals['hours_per_set']) and res.findings == []
    assert RB.to_doc(res, p)['totals']['hours_per_disk_set'] is None


def test_link_capacity_from_the_interface():
    cap = RB.link_capacity_mbps
    assert cap('10GBASE-T') == 10000 and cap('1000BASE-T') == cap('1000BASE-T1') == 1000
    assert cap('100BASE-TX') == cap('100BASE-T1') == 100 and cap('2.5GBASE-T') == 2500
    assert cap('CAN-FD') == 5 and cap('CAN FD') == 5 and cap('CAN') == 1
    assert cap('CAN-FD / 100BASE-T1') == 100 and cap('CAN / CAN-FD') == 5 and cap('1000BASE-T1 / 10GBASE-T') == 10000
    assert cap('') is None and cap('USB 3') is None and cap(None) is None
    # every built-in sensor's interface is known
    assert all(cap(m.interface) for m in sensors.builtin_sensor_models())


# ----------------------------------------------------------------------------- findings and the Issues list
def test_a_lidar_over_its_link_is_critical(p):
    p.sensor_models['ouster_os1_128'].interface = '100BASE-TX'
    s = _add(p, 'lidar', 'ouster_os1_128', 'L1')
    res = RB.budget(p)
    l1 = res.row('L1')
    assert l1['load'] == pytest.approx(OS1_MBPS / 100) and l1['link_severity'] == 'critical'
    f = next(f for f in res.findings if f['kind'] == 'link')
    assert f['severity'] == 'critical' and f['tags'] == ['L1'] and '100BASE-TX' in f['message']
    iss = [i for i in RB.issues(p) if i.severity == 'critical']
    assert len(iss) == 1 and iss[0].uids == [s.uid] and iss[0].objects == 'L1' and isinstance(iss[0], rules.Issue)
    # over the margin but under the rate: a warning
    p.sensor_models['ouster_os1_128'].interface = '1000BASE-T'
    p.analysis.recording = {'link_warn': 0.1}
    assert [(i.severity, i.objects) for i in RB.issues(p) if 'L1' in i.objects] == [('warning', 'L1')]
    # an interface Vantage does not know: a note on the page, not an issue
    p.sensor_models['ouster_os1_128'].interface = 'Fibre'
    res = RB.budget(p)
    assert res.row('L1')['load'] is None and any(f['kind'] == 'link_unknown' for f in res.findings)
    assert not any('L1' in i.objects for i in RB.issues(p))


def test_write_speed_shortfall_is_critical(p):
    need = (6 * RAW_MBPS + 2.0) * 1.03 / 8                            # MB/s
    res = RB.budget(p, settings=_settings(write_mb_s=1000.0))           # below even the plain rate
    t = res.totals
    assert t['write_needed_mb_s'] == pytest.approx(need * 1.3) and t['write_load'] == pytest.approx(need * 1.3 / 1000)
    f = res.findings[0]
    assert (f['severity'], f['kind']) == ('critical', 'write') and 'even before' in f['message']
    res = RB.budget(p, settings=_settings(write_mb_s=1200.0))           # enough for the data, not for the headroom
    f = res.findings[0]
    assert (f['severity'], f['kind']) == ('critical', 'write') and 'headroom' in f['message']
    p.analysis.recording = {'write_mb_s': 1200.0}
    assert [i.severity for i in RB.issues(p)][0] == 'critical'
    good = RB.budget(p, settings=_settings(write_mb_s=3000.0))
    assert next(f for f in good.findings if f['kind'] == 'write')['severity'] == 'good'


def test_uplink_over_capacity_is_serious(p):
    _add(p, 'lidar', 'ouster_os1_128', 'L1')
    _add(p, 'lidar', 'hesai_ot128', 'L2')
    p.analysis.recording = {'uplink_gbps': 0.2}
    res = RB.budget(p)
    t = res.totals
    assert t['uplink_capacity_mbps'] == 200.0 and t['uplink_load'] > 1.0
    up = next(f for f in res.findings if f['kind'] == 'uplink')
    assert up['severity'] == 'serious' and set(up['tags']) == {'L1', 'L2'}
    assert ('serious', 'L1, L2') in [(i.severity, i.objects) for i in RB.issues(p)]
    p.analysis.recording = {'uplink_gbps': 0.3}                         # 250 of 300 Mbit/s: over the 80 % margin
    assert next(f for f in RB.budget(p).findings if f['kind'] == 'uplink')['severity'] == 'warning'
    p.analysis.recording = {}
    assert next(f for f in RB.budget(p).findings if f['kind'] == 'uplink')['severity'] == 'good'


def test_the_disk_set_fills_before_the_shift_ends(p):
    res = RB.budget(p)                                                  # six raw cameras: 8 TB lasts about 2 h
    f = next(f for f in res.findings if f['kind'] == 'disk')
    assert f['severity'] == 'warning' and '2.0 h' in f['message'] and '4 sets' in f['message']
    res = RB.budget(p, settings=_settings(storage_tb=40.0))
    assert next(f for f in res.findings if f['kind'] == 'disk')['severity'] == 'good'
    res = RB.budget(p, settings=_settings(camera_format='h265'))
    assert res.totals['hours_per_set'] > 8 and not any(f['severity'] != 'good' for f in res.findings)


def test_the_sample_gains_no_critical_issue(p):
    """Six raw cameras with the default settings: the disks keep up and every link is well under its rate. The disk
    set filling in about two hours of an eight-hour shift is the only finding (a warning)."""
    iss = RB.issues(p)
    assert not any(i.severity in ('critical', 'serious') for i in iss)
    assert [(i.severity, i.message.split(' ')[0]) for i in iss] == [('warning', 'One')]
    assert not any('Gbps' in i.message for i in iss)                    # the GMSL budget's issues say Gbps
    for v in p.variants:                                                # no variant of the sample is over a limit
        assert not any(f['severity'] in ('critical', 'serious') for f in RB.budget(p, v).findings)
    # with a lidar and two radars still nothing critical
    _add(p, 'lidar', 'ouster_os1_128', 'L1')
    _add(p, 'radar', 'radar_long', 'RD1')
    _add(p, 'radar', 'radar_corner', 'RD2')
    assert not any(i.severity in ('critical', 'serious') for i in RB.issues(p))


def test_issues_are_cheap(p):
    _add(p, 'lidar', 'ouster_os1_128', 'L1')
    _add(p, 'radar', 'radar_long', 'RD1')
    RB.issues(p)
    t0 = time.perf_counter()
    for _ in range(200):
        RB.issues(p)
    assert (time.perf_counter() - t0) / 200 < 0.005                    # well under 5 ms (no geometry)


def test_a_variant_other_than_the_active_one(p):
    four = next(v for v in p.variants if v.name == '4-cam')
    res = RB.budget(p, four)
    assert res.variant == '4-cam' and res.variant_uid == four.uid
    assert [r['tag'] for r in res.rows] == [c.tag for c in four.cameras if c.visible] and len(res.rows) == 4


# ----------------------------------------------------------------------------- settings
def test_settings_are_validated():
    d = RB.RecordingSettings().to_dict()
    assert RB.RecordingSettings.from_dict(d) == RB.RecordingSettings()
    assert RB.RecordingSettings.from_dict(None).to_dict() == d
    st = RB.RecordingSettings.from_dict({'camera_format': 'png', 'lidar_format': 'pcap', 'jpeg_bpp': 'lots',
                                         'h265_bpp': float('nan'), 'storage_tb': -5, 'write_mb_s': 0,
                                         'write_headroom': 0.5, 'link_warn': 3, 'container_overhead': 2,
                                         'shift_hours': '10', 'include_radars': 'false', 'nonsense': 1})
    assert st.camera_format == 'raw' and st.lidar_format == 'packets'
    assert st.jpeg_bpp == 2.0 and st.h265_bpp == 0.15                   # not numbers: the defaults
    assert st.storage_tb == RB.LIMITS['storage_tb'][0] and st.write_mb_s == 1.0 and st.write_headroom == 1.0
    assert st.link_warn == 1.0 and st.container_overhead == 1.0 and st.shift_hours == 10.0
    assert st.include_radars is False and 'nonsense' not in st.to_dict()
    st = RB.RecordingSettings.from_dict({'camera_format': 'h265', 'lidar_format': 'points', 'include_radars': 0,
                                         'uplink_gbps': 25})
    assert (st.camera_format, st.lidar_format, st.include_radars, st.uplink_gbps) == ('h265', 'points', False, 25.0)
    # the project's own settings are used when none are given
    p = samples.tigor_sample_project()
    p.analysis.recording = {'camera_format': 'jpeg', 'storage_tb': 16}
    res = RB.budget(p)
    assert res.settings.camera_format == 'jpeg' and res.settings.storage_tb == 16.0


# ----------------------------------------------------------------------------- output
def test_summary_doc_and_export(p, tmp_path):
    _add(p, 'lidar', 'ouster_os1_128', 'L1', port='2')
    _add(p, 'radar', 'radar_long', 'RD1')
    res = RB.budget(p)
    s = RB.summary(res)
    json.dumps(s)
    assert s['data_rate_mbit_s'] == round(res.totals['mbps'], 1) and len(s['per_sensor']) == 8
    assert [f['severity'] for f in s['findings']] == ['warning'] and 'assumptions' not in s['findings'][0]
    paths = RB.export(res, p, str(tmp_path / 'out'))
    assert [x.rsplit('/', 1)[-1] for x in paths] == ['recording_budget.json', 'recording_budget.csv']
    doc = json.loads((tmp_path / 'out' / 'recording_budget.json').read_text())
    assert doc['schema'] == RB.SCHEMA == 'vantage.recording_budget/1'
    assert doc['project'] == p.name and doc['variant'] == p.variant.name
    assert doc['settings']['raw_bits_per_pixel'] == 16 and doc['settings']['camera_format'] == 'raw'
    assert [r['tag'] for r in doc['sensors']] == ['F', 'R', 'ML', 'MR', 'QL', 'QR', 'L1', 'RD1']
    l1 = doc['sensors'][6]
    assert l1['data_mbit_s'] == pytest.approx(OS1_MBPS, abs=1e-3)
    assert l1['link_load_pct'] == pytest.approx(12.58, abs=0.01)
    assert l1['port'] == '2' and l1['link_capacity_mbit_s'] == 1000.0
    assert sum(r['share_pct'] for r in doc['sensors']) < 100.0
    assert doc['totals']['data_mbit_s'] == pytest.approx(res.totals['mbps'], abs=1e-3)
    assert doc['totals']['hours_per_disk_set'] == pytest.approx(res.totals['hours_per_set'], abs=1e-3)
    assert doc['findings'][0]['severity'] == 'warning' and doc['findings'][0]['kind'] == 'disk'
    with open(tmp_path / 'out' / 'recording_budget.csv', newline='') as fh:
        rows = list(csv.DictReader(fh))
    assert [r['tag'] for r in rows] == ['F', 'R', 'ML', 'MR', 'QL', 'QR', 'L1', 'RD1', 'total']
    assert float(rows[0]['data_mbit_s']) == pytest.approx(RAW_MBPS, abs=1e-3) and rows[0]['link'] == 'GMSL2'
    assert float(rows[-1]['data_mbit_s']) == pytest.approx(res.totals['mbps'], abs=1e-3)
    assert float(rows[-1]['link_mbit_s']) == pytest.approx(OS1_MBPS + 1.0, abs=1e-3) and rows[-1]['link'] == 'uplink'
    assert [n for n, _ in RB.files(res, p)] == ['recording_budget.json', 'recording_budget.csv']


# ----------------------------------------------------------------------------- the page, driven like a user
from test_gui import app, win, pump, settle, _wait     # noqa: E402,F401  (fixtures)


def _type_into_field(app, f, text):
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    QTest.mouseClick(f.lineEdit(), Qt.MouseButton.LeftButton)
    QTest.keyClick(f.lineEdit(), Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
    QTest.keyClicks(f.lineEdit(), text)
    QTest.keyClick(f.lineEdit(), Qt.Key.Key_Return)
    pump(app, 0.1)


def test_page_edits_undoes_and_leaves(app, win, tmp_path):
    menus = {a.text().replace('&', ''): a.menu() for a in win.menuBar().actions()}
    act = win.actions_by_id['recording_budget']
    assert act in menus['Analysis'].actions()
    act.trigger()                                                      # Analysis ▸ Recording budget…
    pump(app, 0.2)
    pg = win.page('recording')
    assert win.page_name() == 'recording' and pg.isVisible()
    store = win.store
    res = pg.result
    assert res is not None and res.variant == store.variant.name
    assert pg.table.rowCount() == len(res.rows) + 3 and pg.table.item(0, 0).text() == res.rows[0]['tag']
    assert pg.table.item(0, 4).text() == f'{RAW_MBPS:,.1f}'
    assert pg.tiles['disk'].sev == 'warning' and pg.tiles['write'].sev == 'good'
    assert pg.tiles['disk'].value == f'{res.totals["hours_per_set"]:,.1f}'
    assert len(pg.findings) == len(res.findings) and len(pg.bar.segs) == len(res.rows) + 2
    hours = res.totals['hours_per_set']
    # a bigger disk set, typed into its field: saved with the project, computed again at once, undoable
    f = pg.fields['storage_tb'][0]
    _type_into_field(app, f, '40')
    assert store.project.analysis.recording['storage_tb'] == 40
    assert pg.result.totals['hours_per_set'] == pytest.approx(hours * 5) and pg.tiles['disk'].sev == 'good'
    # cameras recorded as H.265, a click on the format
    pg.cam_fmt.buttons['h265'].click()
    pump(app, 0.1)
    assert store.project.analysis.recording['camera_format'] == 'h265' and pg.result.settings.camera_format == 'h265'
    assert pg.result.row('F')['mbps'] == pytest.approx(RAW_MBPS / 16 * 0.15)
    # nonsense is clamped to what is usable; the field shows what is stored
    _type_into_field(app, pg.fields['write_headroom'][0], '0.2')
    assert store.project.analysis.recording['write_headroom'] == 1.0 and pg.fields['write_headroom'][0].value() == 1.0
    store.undo.undo()
    store.undo.undo()
    pump(app, 0.1)
    assert store.project.analysis.recording['camera_format'] == 'raw' and pg.cam_fmt.value() == 'raw'
    assert pg.result.row('F')['mbps'] == pytest.approx(RAW_MBPS)
    store.undo.undo()
    pump(app, 0.1)
    assert store.project.analysis.recording == {} and f.value() == 8.0          # back to the defaults, as it started
    assert pg.result.totals['hours_per_set'] == pytest.approx(hours)
    # the raw pixel format is the ECU's: one figure for the recording and the GMSL budget
    _type_into_field(app, pg.fields['raw_bpp'][0], '12')
    assert store.project.ecu.bpp == 12 and pg.result.row('F')['mbps'] == pytest.approx(RAW_MBPS * 12 / 16)
    store.undo.undo()
    pump(app, 0.1)
    assert store.project.ecu.bpp == 16
    paths = pg.export(str(tmp_path))
    assert json.loads((tmp_path / 'recording_budget.json').read_text())['schema'] == RB.SCHEMA and len(paths) == 2
    # leave the page: edits are no longer computed until it shows again
    win.set_page('viewports')
    pump(app, 0.2)
    assert not pg.isVisible()
    shown = pg.result
    store.edit('analysis', lambda q: setattr(q.analysis, 'recording', {'storage_tb': 16.0}), 'test')
    pump(app, 0.1)
    assert pg.result is shown and pg._dirty
    act.trigger()
    pump(app, 0.2)
    assert pg.result is not shown and pg.result.settings.storage_tb == 16.0 and f.value() == 16.0
