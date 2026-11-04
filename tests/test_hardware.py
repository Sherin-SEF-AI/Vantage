"""Custom hardware (core/hardware.py): the spec registry lists the figures the app reads, templates and validation,
from_spec / to_spec round trips, custom sensors and cameras in the project library, rig hardware selections that drive
the studies (power, recording, clock tree, link budget) and put the old settings back, save and load; and the dialog
(offscreen)."""
import copy
import json
import math

import pytest

from vantage.core import hardware as HW
from vantage.core import samples
from vantage.core.analysis import Engine


def _rig():
    p = samples.tigor_sample_project()
    p.active_uid = next(v.uid for v in p.variants if v.sensors)
    return p


VALID = {
    'camera_module': {'name': 'Test GMSL cam', 'width_px': 2880, 'height_px': 1860, 'pixel_um': 2.1, 'fps': 30,
                      'shutter': 'rolling', 'power_w': 1.8, 'optical_format': '1/1.7"', 'size_mm': [30, 30, 40]},
    'lens': {'name': 'Test 120° lens', 'module_id': 'sturdecam31', 'projection': 'equisolid',
             'sheet_fov': [120.0, 0.0, 0.0]},
    'lidar': {'name': 'Test 64', 'channels': 64, 'v_min': -25, 'v_max': 15, 'h_fov': 360, 'h_res': 0.2,
              'rate_hz': 10, 'range_m': 120, 'points_per_s': 1_152_000, 'size_mm': [100, 100, 80],
              'optical_z_mm': 40, 'power_w': 15, 'time_sync': 'PTP', 'interface': '1000BASE-T1'},
    'radar': {'name': 'Test radar', 'zones': [[200, 10, 5], [70, 45, 8]], 'rate_hz': 20, 'data_mbps': 3,
              'power_w': 5},
    'gnss': {'name': 'Test antenna', 'mask_deg': 10, 'lna_gain_db': 30},
    'ins': {'name': 'Test INS', 'antennas': 2, 'heading_sigma_deg': 0.05, 'heading_baseline_m': 2,
            'pos_sigma_cm': 1, 'attitude_sigma_deg': 0.02, 'rate_hz': 100, 'power_w': 6,
            'accel_noise_density': 2e-4, 'accel_random_walk': 3e-5, 'gyro_noise_density': 3e-5,
            'gyro_random_walk': 2e-6},
    'imu': {'name': 'Test IMU', 'rate_hz': 400, 'accel_noise_density': 1e-3},
    'compute': {'name': 'Test compute', 'ports': 16, 'link_gbps': 12, 'csi_gbps_per_deser': 20, 'power_w': 90},
    'recorder': {'name': 'Test recorder', 'write_mb_s': 3000, 'storage_tb': 16, 'power_w': 55},
    'switch': {'name': 'Test switch', 'uplink_gbps': 25, 'power_w': 20, 'sync_ms': 31.25, 'jitter_ns': 4},
    'time_server': {'name': 'Test GM', 'power_w': 7, 'ppm': 0.01, 'aging_ppb_day': 0.5, 'pps_ns': 15},
    'converter': {'name': 'Test DC-DC', 'rated_w': 1000, 'efficiency': 0.94, 'output_v': 24},
}


def test_every_kind_has_fields_the_app_reads():
    assert set(HW.kinds()) == set(VALID)
    for k in HW.kinds():
        fs = HW.fields(k)
        keys = [f.key for f in fs]
        assert len(keys) == len(set(keys)) and 'name' in keys, k
        assert all(f.used_by for f in fs if f.key not in ('id',)), k        # every field names who reads it
        t = HW.template(k)
        assert t['kind'] == k and set(t['spec']) == set(keys) and set(t['help']) == set(keys)
        json.dumps(t)
        assert HW.validate(k, t)[0], k                               # a blank sheet misses its required figures
        assert HW.validate(k, VALID[k])[0] == [], (k, HW.validate(k, VALID[k]))
    # the sensor fields are the SensorModel's, the camera's the Module's, the compute's the ECU's
    from vantage.core.lens import Lens, Module
    from vantage.core.project import ECU
    from vantage.core.sensors import SensorModel
    for k in ('lidar', 'radar', 'gnss', 'ins', 'imu'):
        assert {f.key for f in HW.fields(k)} <= set(SensorModel.__dataclass_fields__), k
    assert {f.key for f in HW.fields('camera_module')} <= set(Module.__dataclass_fields__)
    assert {f.key for f in HW.fields('lens')} <= set(Lens.__dataclass_fields__)
    assert {k for k, (sec, _) in HW._APPLY['compute'].items() if sec == 'ecu'} <= {f.key for f in HW.fields('compute')}
    assert {sk for sec, sk in HW._APPLY['compute'].values() if sec == 'ecu'} <= set(ECU.__dataclass_fields__)


def test_validation_catches_inconsistent_specs():
    e, w = HW.validate('lidar', dict(VALID['lidar'], v_min=20))
    assert any('Lowest beam' in x for x in e)
    e, w = HW.validate('lidar', dict(VALID['lidar'], beams=[0.0] * 10))
    assert any('10 given for 64 channels' in x for x in e)
    e, w = HW.validate('lidar', dict(VALID['lidar'], points_per_s=10_000_000))
    assert not e and any('Point rate' in x for x in w)
    e, _ = HW.validate('radar', dict(VALID['radar'], zones=[[200, 10]]))
    assert e
    _, w = HW.validate('radar', dict(VALID['radar'], zones=[[70, 45, 8], [200, 10, 5]]))
    assert any('far zone' in x for x in w)
    _, w = HW.validate('camera_module', dict(VALID['camera_module'], optical_format='1/4"'))
    assert any('diagonal' in x for x in w)
    e, _ = HW.validate('lens', {'name': 'x', 'projection': 'kannala_brandt'})
    assert any('focal length or a stated field' in x for x in e)
    e, _ = HW.validate('lens', {'name': 'x', 'focal_mm': 2, 'projection': 'rectilinear', 'sheet_fov': [170, 0, 0]})
    assert any('rectilinear' in x for x in e)
    e, _ = HW.validate('converter', dict(VALID['converter'], efficiency=1.5))
    assert e
    e, _ = HW.validate('compute', dict(VALID['compute'], ports=2.5))
    assert any('whole number' in x for x in e)
    e, _ = HW.validate('recorder', {'name': 'r', 'write_mb_s': 'fast', 'storage_tb': 4, 'power_w': 1})
    assert any('a number' in x for x in e)


def test_round_trips_and_the_app_objects():
    from vantage.core.lens import Lens, Module, resolve, builtin_modules
    from vantage.core.sensors import SensorModel
    for k in ('camera_module', 'lidar', 'radar', 'gnss', 'ins', 'imu'):
        obj = HW.from_spec(k, VALID[k])
        spec = HW.to_spec(obj)
        again = HW.from_spec(k, spec)
        assert again.to_dict() == obj.to_dict(), k
        assert not obj.builtin
    m = HW.from_spec('camera_module', VALID['camera_module'])
    assert isinstance(m, Module) and m.sensor_w_mm == pytest.approx(2880 * 2.1e-3)
    r = HW.from_spec('radar', VALID['radar'])
    assert isinstance(r, SensorModel) and r.kind == 'radar' and r.range_m == 200 and r.time_sync == 'none'
    ins = HW.from_spec('ins', VALID['ins'])
    assert ins.kind == 'imu' and ins.antennas == 2 and HW.to_spec(ins)['antennas'] == 2
    assert HW.from_spec('imu', VALID['imu']).antennas == 0
    # a lens from its stated field: the model reproduces it on its module
    ln = HW.from_spec('lens', VALID['lens'])
    mod = next(mm for mm in builtin_modules() if mm.id == 'sturdecam31')
    from vantage.core.lens import model_fov
    assert isinstance(ln, Lens) and model_fov(mod, ln)[0] == pytest.approx(120.0, abs=0.5)
    # a Kannala–Brandt lens fitted to its stated fields
    kb = HW.from_spec('lens', {'name': 'kb', 'module_id': 'sturdecam31', 'projection': 'kannala_brandt',
                               'sheet_fov': [190.0, 150.0, 0.0]})
    assert kb.projection == 'kannala_brandt' and any(kb.kb) and 'fitted' in kb.note
    assert model_fov(mod, kb, circle=True)[0] == pytest.approx(190.0, abs=1.5)
    assert resolve(mod, kb).f_px > 0
    with pytest.raises(ValueError):
        HW.from_spec('lidar', {'name': 'x'})
    # the template sheet itself is a valid input once filled
    t = HW.template('recorder')
    t['spec'].update(VALID['recorder'])
    assert HW.from_spec(None, t)['write_mb_s'] == 3000


def test_custom_sensors_and_cameras_join_the_project_library():
    p = _rig()
    lid = HW.add_to_project(p, 'lidar', VALID['lidar'])
    assert lid in p.sensor_models and not p.sensor_models[lid].builtin
    assert HW.add_to_project(p, 'lidar', VALID['lidar']) == f'{lid}_2'              # ids stay unique
    mid = HW.add_to_project(p, 'camera_module', VALID['camera_module'])
    lid2 = HW.add_to_project(p, 'lens', dict(VALID['lens'], module_id=mid))
    assert mid in p.modules and lid2 in p.lenses and p.lenses[lid2].module_id == mid
    # a custom lidar on the rig changes the coverage rings: a 64-channel at −25° reaches the ground nearer
    from vantage.core import sensor_coverage as SC
    e = Engine()
    base = SC.run(p, e).lidars[0].ring_m
    s = next(x for x in p.variant.sensors if x.kind == 'lidar')
    s.model_id = lid
    near = SC.run(p, e).lidars[0].ring_m
    assert near < base                       # its lowest beam at −25° (the OS1-128's at −21.2°) meets the ground nearer
    # a custom camera is used by the link budget and the recording budget
    from vantage.core import recording as RC
    from vantage.core import rules
    c = p.variant.cameras[0]
    before = RC.budget(p).rows[0]['mbps']
    c.module_id, c.lens_id = mid, lid2
    after = RC.budget(p).rows[0]['mbps']
    assert after == pytest.approx(before * (2880 * 1860) / (1920 * 1536), rel=1e-6)
    assert rules.camera_gbps(p, c) == pytest.approx(2880 * 1860 * 30 * 16 * 1.1 / 1e9)
    # every item is listed for the Library panel
    kinds = {k for k, *_ in HW.library_items(p)}
    assert {'lidar', 'camera_module', 'lens'} <= kinds


def test_rig_hardware_drives_the_studies_and_comes_off_cleanly():
    from vantage.core import clock_tree as CT
    from vantage.core import power as PW
    from vantage.core import recording as RC
    p = samples.template_project('van')                          # 8 cameras: one disk set cannot keep up
    e = Engine()
    rec0 = RC.budget(p)
    assert any(f['severity'] == 'critical' for f in rec0.findings)
    pw0 = PW.run(p, e).totals['steady_w']
    snap = (copy.deepcopy(p.analysis.power), copy.deepcopy(p.analysis.recording), p.ecu.to_dict())
    rid = HW.add_to_project(p, 'recorder', VALID['recorder'])
    assert HW.selected(p, 'recorder')['name'] == 'Test recorder'
    rec1 = RC.budget(p)
    assert not any(f['severity'] == 'critical' for f in rec1.findings) and rec1.settings.write_mb_s == 3000
    assert 'recorder' in RC.to_doc(rec1, p)['hardware'] and 'Test recorder' in RC.to_doc(rec1, p)['hardware'][
        'recorder']
    HW.add_to_project(p, 'compute', VALID['compute'])
    assert p.ecu.name == 'Test compute' and p.ecu.ports == 16 and p.ecu.link_gbps == 12
    pw1 = PW.run(p, e)
    assert pw1.totals['steady_w'] == pytest.approx(pw0 + (90 - 60) + (55 - 40), abs=0.6)
    HW.add_to_project(p, 'converter', VALID['converter'])
    pw2 = PW.run(p, e)
    assert pw2.settings.bus_v == 24 and pw2.settings.dcdc_w == 1000 and pw2.settings.dcdc_eff == 0.94
    assert set(PW.to_doc(pw2, p)['hardware']) == {'recorder', 'compute', 'converter'}
    HW.add_to_project(p, 'switch', VALID['switch'])
    HW.add_to_project(p, 'time_server', VALID['time_server'])
    topo = CT.build_topology(p)
    sw = topo.node('switch')
    assert sw.p['sync_ms'] == 31.25 and sw.p['jitter_ns'] == 4 and 'Test switch' in sw.typical['sync_ms'][1]
    assert 'Test switch' in sw.label and topo.node('server').p['ppm'] == 0.01
    assert topo.node('gnss').p['pps_ns'] == 15
    assert RC.budget(p).settings.uplink_gbps == 25
    # clearing every selection puts the old settings back exactly
    for role in HW.RIG_ROLES:
        HW.select_item(p, role, None)
    assert (p.analysis.power, p.analysis.recording, p.ecu.to_dict()) == snap
    assert CT.build_topology(p).node('switch').p['sync_ms'] == CT.DEFAULT_SYNC_MS
    # switching between two recorders restores in between
    r2 = HW.add_to_project(p, 'recorder', dict(VALID['recorder'], name='Slow', write_mb_s=500))
    HW.select_item(p, 'recorder', rid)
    assert p.analysis.recording['write_mb_s'] == 3000 and r2 != rid
    HW.select_item(p, 'recorder', None)
    assert p.analysis.recording == snap[1]


def test_saved_and_loaded(tmp_path):
    from vantage.core.project import Project
    p = _rig()
    HW.add_to_project(p, 'recorder', VALID['recorder'])
    HW.add_to_project(p, 'radar', VALID['radar'])
    path = str(tmp_path / 'hw.vantage')
    p.save(path)
    q = Project.load(path)
    assert HW.selected(q, 'recorder')['write_mb_s'] == 3000 and q.analysis.recording['write_mb_s'] == 3000
    assert any(m.name == 'Test radar' for m in q.sensor_models.values())
    s = q.snapshot()
    assert s.hardware == q.hardware and s.hardware is not q.hardware


# ---------------------------------------------------------------- the dialog
from test_gui import app, win, pump, settle  # noqa: E402,F401  (fixtures)


def test_the_dialog_builds_validates_and_adds(app, win, tmp_path):  # noqa: F811
    from vantage.ui.dialogs.hardware_dialog import HardwareDialog
    p = _rig()
    win.store.set_project(p)
    settle(app, win)
    d = HardwareDialog(win.ctx, win, kind='recorder')
    d.show()
    pump(app, 0.1)
    assert d.kind == 'recorder' and set(d.inputs) == {f.key for f in HW.fields('recorder')}
    assert not d.b_add.isEnabled() and 'required' in d.msg.text()
    d.set_values(VALID['recorder'])
    pump(app, 0.05)
    assert d.b_add.isEnabled(), d.msg.text()
    n0 = win.store.undo.count()
    iid = d.add()
    settle(app, win)
    proj = win.store.project
    assert HW.selected(proj, 'recorder')['id'] == iid and proj.analysis.recording['write_mb_s'] == 3000
    assert win.store.undo.count() == n0 + 1
    win.store.undo.undo()
    settle(app, win)
    assert HW.selected(win.store.project, 'recorder') is None
    # a spec sheet saved and loaded; a kind switch rebuilds the form; a duplicate pre-fills from the library
    path = str(tmp_path / 'lidar.json')
    d.set_kind('lidar')
    d.save_template(path)
    data = json.load(open(path))
    assert data['kind'] == 'lidar' and 'help' in data
    data['spec'].update(VALID['lidar'])
    json.dump(data, open(path, 'w'))
    d.load_sheet(path)
    assert d.values()['channels'] == 64 and d.b_add.isEnabled()
    d.duplicate('ouster_os1_128')
    assert d.values()['channels'] == 128 and d.values()['name'].endswith('(custom)')
    d.set_values(dict(d.values(), v_min=30))
    pump(app, 0.05)
    assert not d.b_add.isEnabled() and 'Lowest beam' in d.msg.text()
    d.close()


def test_the_library_panel_lists_custom_hardware(app, win):  # noqa: F811
    from vantage.ui.outliner import LibraryPanel, select_hardware
    p = _rig()
    rid = HW.add_to_project(p, 'recorder', VALID['recorder'], select=False)
    HW.add_to_project(p, 'radar', VALID['radar'])
    win.store.set_project(p)
    settle(app, win)
    panel = win.findChildren(LibraryPanel)[0]
    panel.kind.setCurrentIndex(panel.kind.findText('Hardware'))
    pump(app, 0.05)
    tree = panel.tree
    names = [tree.topLevelItem(i).text(0) for i in range(tree.topLevelItemCount())]
    assert names[0] == 'New custom hardware…' and 'Recorder' in names and 'Radar' in names
    select_hardware(win.ctx, 'recorder', rid)
    settle(app, win)
    assert win.store.project.analysis.recording['write_mb_s'] == 3000
    win.store.undo.undo()
    settle(app, win)
    assert HW.selected(win.store.project, 'recorder') is None
    panel.kind.setCurrentIndex(0)
