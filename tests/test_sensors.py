"""Lidars, radars, GNSS antennas and IMUs of the rig: the library, the geometry of beams and fields, persistence, the
exports and the sensor UI."""
import json
import math
import os
import time

import numpy as np
import pytest

from vantage.core import sensors as S
from vantage.core.project import Project, Sensor, Variant


def _lidar(model_id='velodyne_vlp16', **kw):
    d = dict(kind='lidar', model_id=model_id, tag='L1', x=0.0, y=0.0, z=2.0)
    d.update(kw)
    return Sensor(**d)


def test_library_figures_are_consistent():
    ms = {m.id: m for m in S.builtin_sensor_models()}
    assert {'ouster_os1_128', 'hesai_ot128', 'hesai_at128', 'velodyne_vlp16', 'livox_mid360'} <= set(ms)
    vlp = ms['velodyne_vlp16']
    assert vlp.n_channels() == 16 and vlp.v_step() == pytest.approx(2.0)
    assert vlp.pps() == 300_000 and vlp.data_bps() == pytest.approx(300_000 * 3.25 * 8)
    os1 = ms['ouster_os1_128']
    assert os1.columns() == 1024 and os1.pps() == 128 * 1024 * 10
    at = ms['hesai_at128']
    assert at.az_half() == pytest.approx(60.0) and at.columns() * at.n_channels() * at.rate_hz == at.pps()
    for m in ms.values():
        assert m.note and m.builtin
        if m.kind == 'radar':
            assert m.zones and m.data_bps() > 0 and m.max_range() > 0
        assert S.SensorModel.from_dict(json.loads(json.dumps(m.to_dict()))) == m


def test_ground_ring_radius_is_height_over_tan_of_the_beam():
    m = S.SensorModel('t', channels=3, v_min=-30.0, v_max=-10.0, h_res=1.0, range_m=200.0, range_min_m=0.1)
    s = _lidar('t', z=2.0)
    rings = dict(S.ground_rings(s, m, 1.0))
    for ch, el in enumerate((-30.0, -20.0, -10.0)):
        r = np.hypot(rings[ch][:, 0], rings[ch][:, 1])
        assert np.allclose(r, 2.0 / math.tan(math.radians(-el)), rtol=1e-9)
        assert np.allclose(rings[ch][0], rings[ch][-1])            # a spinning lidar's ring is closed
    # a ring beyond the lidar's (slant) range is left out; pitching the lidar down brings the front of a ring closer
    m.range_m = 5.0
    assert set(dict(S.ground_rings(s, m, 1.0))) == {0}
    m.range_m = 200.0
    tilted = dict(S.ground_rings(_lidar('t', z=2.0, pitch=5.0), m, 1.0))
    front = tilted[2][np.argmin(np.abs(np.arctan2(tilted[2][:, 1], tilted[2][:, 0])))]
    assert front[0] == pytest.approx(2.0 / math.tan(math.radians(15.0)), rel=1e-3)


def test_angles_and_field_follow_the_pose():
    m = S.builtin_sensor_models()[2]                         # AT128: 120° wide, fixed field
    s = Sensor(kind='lidar', model_id=m.id, x=1.0, y=0.0, z=1.5, yaw=90.0)
    P = np.array([[1.0, 20.0, 1.5], [21.0, 0.0, 1.5], [1.0, 20.0, 1.5 + 20 * math.tan(math.radians(20))]])
    r, az, el = S.angles(s, P)
    assert r[0] == pytest.approx(20.0) and az[0] == pytest.approx(0.0, abs=1e-9) and el[0] == pytest.approx(0.0)
    assert az[1] == pytest.approx(-90.0)
    assert list(S.in_field(s, m, P)) == [True, False, False]   # beside it, and above its 12.9° top beam
    assert S.heading(s) == pytest.approx(90.0)


def test_radar_range_scales_with_the_fourth_root_of_the_cross_section():
    m = next(m for m in S.builtin_sensor_models() if m.id == 'radar_long')
    s = Sensor(kind='radar', model_id=m.id, x=0.0, z=0.5)
    far = 250.0 * (0.5 / 10.0) ** 0.25
    assert S.radar_scale(0.5) == pytest.approx((0.05) ** 0.25)
    assert list(S.in_field(s, m, [[far - 1, 0, 0.5], [far + 1, 0, 0.5]], rcs=0.5)) == [True, False]
    # the wide near beam: 45° off the axis is seen close by, not far out
    a = math.radians(40)
    assert list(S.in_field(s, m, [[60 * math.cos(a), 60 * math.sin(a), 0.5],
                                  [100 * math.cos(a), 100 * math.sin(a), 0.5]])) == [True, False]
    w = S.radar_wedges(s, m)
    assert len(w) == 2 and np.hypot(*w[0][1:].T).max() == pytest.approx(250.0)


def test_beam_angle_files():
    ouster = json.dumps({'beam_intrinsics': {'beam_altitude_angles': [21.0, 0.5, -21.5]}})
    assert S.parse_beam_angles(ouster) == [-21.5, 0.5, 21.0]
    assert S.parse_beam_angles('Channel,Elevation,Azimuth\n1,15.0,1.2\n2,-16.0,0.3\n') == [-16.0, 15.0]
    yml = 'lasers:\n  - {laser_id: 1, vert_correction: 0.0174533}\n  - {laser_id: 0, vert_correction: -0.2617994}\n'
    assert S.parse_beam_angles(yml) == pytest.approx([-15.0, 1.0], abs=1e-4)
    assert S.parse_beam_angles('# beams\n-2\n0\n2\n') == [-2.0, 0.0, 2.0]
    for bad in ('', '{"a": 1}', '1', '100\n200'):
        with pytest.raises(ValueError):
            S.parse_beam_angles(bad)
    m = S.SensorModel('t', beams=[3.0, -1.0, 1.0])
    assert list(m.elevations()) == [-1.0, 1.0, 3.0] and m.n_channels() == 3


def test_tags_poses_and_persistence(tmp_path):
    from vantage.core.samples import tigor_sample_project
    p = tigor_sample_project()
    v = p.variant
    assert S.next_tag('lidar', v.tags()) == 'L1' and S.next_tag('radar', v.tags() | {'RD1'}) == 'RD2'
    lid = p.sensor_models['ouster_os1_128']
    pose = S.default_pose('lidar', lid, p.vehicle.mesh, p.rig)
    top = p.vehicle.mesh.bbox[1][2]
    assert pose['z'] == pytest.approx(top + lid.optical_z_mm / 1000, abs=0.01)
    rad = p.sensor_models['radar_long']
    rp = S.default_pose('radar', rad, p.vehicle.mesh)
    assert rp['x'] > p.vehicle.measured()['front_x'] - 0.1 and rp['z'] == 0.5
    v.sensors += [Sensor(kind='lidar', model_id=lid.id, tag='L1', **pose),
                  Sensor(kind='radar', model_id=rad.id, tag='RD1', **rp)]
    custom = S.SensorModel.from_dict(dict(lid.to_dict(), id='mine', name='Mine', builtin=False, beams=[-5, 0, 5]))
    p.sensor_models['mine'] = custom
    v.sensors[0].model_id = 'mine'
    f = str(tmp_path / 'p.vantage')
    p.save(f)
    q = Project.load(f)
    assert [s.to_dict() for s in q.variant.sensors] == [s.to_dict() for s in v.sensors]
    assert q.sensor_models['mine'].beams == [-5.0, 0.0, 5.0] and 'velodyne_vlp16' in q.sensor_models
    assert q.sensor_model_for(q.variant.sensors[0]).name == 'Mine'
    # a project from before sensors existed still gets the library; an unknown model falls back to the same kind
    d = json.loads(json.dumps(p.to_json()))
    d.pop('sensor_models')
    assert Variant.from_dict({'name': 'old'}).sensors == []
    s = Sensor(kind='radar', model_id='gone')
    assert p.sensor_model_for(s).kind == 'radar'
    # snapshots for worker threads copy the sensors and the library
    snap = p.snapshot()
    snap.variant.sensors[0].x += 1
    snap.sensor_models['mine'].range_m = 1
    assert v.sensors[0].x != snap.variant.sensors[0].x and p.sensor_models['mine'].range_m != 1


def test_footprint_and_housing():
    m = S.builtin_sensor_models()[0]
    s = _lidar(m.id, x=1.0, y=0.5, z=2.0)
    c, half, R, rnd = S.body_box(s, m)
    assert rnd and half[2] == pytest.approx(0.037) and c[2] == pytest.approx(2.0 + 0.037 - 0.03618)
    fp = S.footprint(s, m)
    assert np.allclose(np.hypot(fp[:, 0] - 1.0, fp[:, 1] - 0.5), 0.0435)
    r = next(m for m in S.builtin_sensor_models() if m.kind == 'radar')
    fp = S.footprint(Sensor(kind='radar', model_id=r.id, x=3, yaw=90), r)
    assert np.ptp(fp[:, 0]) == pytest.approx(r.size_mm[1] / 1000) and np.ptp(fp[:, 1]) == pytest.approx(r.size_mm[0] / 1000)


def test_issues_for_misplaced_sensors():
    from vantage.core.analysis import Engine
    from vantage.core.rules import sensor_issues, collect_issues
    from vantage.core.samples import tigor_sample_project
    p = tigor_sample_project()
    vg = Engine().vehicle_geom(p)
    lid = p.sensor_models['ouster_os1_128']
    roof = S.default_pose('lidar', lid, p.vehicle.mesh, p.rig)
    front = S.default_pose('radar', p.sensor_models['radar_long'], p.vehicle.mesh)
    v = p.variant
    v.sensors = [Sensor(kind='lidar', model_id=lid.id, tag='L1', **roof),
                 Sensor(kind='radar', model_id='radar_long', tag='RD1', **front)]
    assert sensor_issues(p, vg) == []
    v.sensors[0].z = 0.9                                  # in the cabin
    v.sensors[1].x -= 0.08                                # just behind the fascia: fine
    iss = sensor_issues(p, vg)
    assert [(i.severity, i.objects) for i in iss] == [('critical', 'L1')]
    v.sensors[1].x = 2.93                                 # deep in the engine bay
    v.sensors.append(Sensor(kind='radar', model_id='radar_corner', tag='F', x=5.0, z=-0.2))
    iss = sensor_issues(p, vg)
    kinds = sorted((i.severity, i.objects) for i in iss)
    assert kinds == [('critical', 'F'), ('critical', 'L1'), ('warning', 'F'), ('warning', 'RD1')]
    assert any(i.objects == 'L1' for i in collect_issues(p, None, vg))


# ------------------------------------------------------------------ exports
@pytest.fixture
def rigged():
    from vantage.core.samples import tigor_sample_project
    p = tigor_sample_project()
    v = p.variant
    v.sensors += [Sensor(kind='lidar', model_id='ouster_os1_128', tag='L1', x=1.2, y=0.1, z=1.9, yaw=10.0, pitch=2.0,
                         phase_deg=45.0, port='eth1'),
                  Sensor(kind='radar', model_id='radar_long', tag='RD1', x=3.3, y=0.0, z=0.5),
                  Sensor(kind='lidar', model_id='hesai_at128', tag='F', x=3.0, y=0.0, z=0.8)]   # a tag like a camera's
    return p


def test_sensor_frames_in_extrinsics_urdf_tf_and_csv(rigged, tmp_path):
    import xml.etree.ElementTree as ET
    from vantage.core import exporters as ex
    paths = ex.export(['extrinsics_json', 'extrinsics_csv', 'ros2'], rigged, None,
                      ex.ExportOptions(out_dir=str(tmp_path), frame='front_bumper', units='mm'))
    assert all(os.path.exists(f) for f in paths)
    doc = json.load(open(tmp_path / 'extrinsics.json'))
    sens = {d['tag']: d for d in doc['sensors']}
    assert set(sens) == {'L1', 'RD1', 'F'} and sens['F']['frame'] == 'F_2_link'          # apart from camera F
    import copy
    q = copy.copy(rigged)
    q.frame = 'front_bumper'
    o = q.origin_offset()
    T = np.array(sens['L1']['T_world_sensor'], float)
    s = rigged.variant.sensors[0]
    P_sensor = np.array([10.0, -2.0, 1.0])                    # a point 10 m ahead of the lidar, in its frame
    want = (S.rotation(s) @ P_sensor + s.pos() - o) * 1000
    assert np.allclose(T[:3, :3] @ (P_sensor * 1000) + T[:3, 3], want, atol=0.01)
    assert sens['L1']['field']['channels'] == 128 and sens['L1']['sync_phase'] == 45.0 and sens['L1']['port'] == 'eth1'
    assert sens['RD1']['field']['zones'][0]['range_car'] == 250000.0
    rows = (tmp_path / 'extrinsics.csv').read_text().splitlines()
    assert len(rows) == 1 + len(rigged.variant.cameras) + 3 and rows[-2].startswith('RD1,')
    root = ET.parse(tmp_path / 'vantage_cameras.urdf').getroot()
    links = {e.get('name') for e in root.iter('link')}
    assert {'L1_link', 'RD1_link', 'F_2_link', 'F_link'} <= links
    j = next(e for e in root.iter('joint') if e.get('name') == 'L1_joint')
    xyz = [float(v) for v in j.find('origin').get('xyz').split()]
    rpy = [float(v) for v in j.find('origin').get('rpy').split()]
    assert np.allclose(xyz, s.pos() - o, atol=1e-3) and np.allclose(rpy, np.radians([0.0, 2.0, 10.0]), atol=1e-5)
    launch = (tmp_path / 'static_tf.launch.py').read_text()
    compile(launch, 'static_tf.launch.py', 'exec')
    assert "'L1_link'" in launch and "'RD1_link'" in launch


def test_sensors_in_nuscenes_and_kitti(rigged, tmp_path):
    from vantage.core import exporters as ex
    from vantage.core.analysis import camera_frames
    from vantage.core.mathutil import BODY_TO_OPTICAL
    ex.export(['nuscenes_calib', 'kitti_calib'], rigged, None, ex.ExportOptions(out_dir=str(tmp_path)))
    sensors = json.load(open(tmp_path / 'nuscenes' / 'sensor.json'))
    cal = json.load(open(tmp_path / 'nuscenes' / 'calibrated_sensor.json'))
    ch = [x['channel'] for x in sensors]
    assert ch[-3:] == ['LIDAR_L1', 'RADAR_RD1', 'LIDAR_F_2'] and [x['modality'] for x in sensors][-2] == 'radar'
    c = cal[-3]
    assert c['camera_intrinsic'] == [] and np.allclose(c['translation'], [1.2, 0.1, 1.9])
    w, x, y, z = c['rotation']
    from scipy.spatial.transform import Rotation
    s = rigged.variant.sensors[0]
    assert np.allclose(Rotation.from_quat([x, y, z, w]).as_matrix(), S.rotation(s), atol=1e-8)
    # KITTI: the first lidar into camera 00, X_cam = R X_lidar + T
    velo = {k: np.array(v.split(), float) for k, v in (l.split(': ', 1) for l in
                                                       (tmp_path / 'kitti' / 'calib_velo_to_cam.txt').read_text()
                                                       .splitlines() if not l.startswith(('calib_time', 'note')))}
    R, T = velo['R'].reshape(3, 3), velo['T']
    cam0 = rigged.variant.cameras[0]
    Rc, W2O = camera_frames(cam0)
    P_l = np.array([5.0, 1.0, -0.5])
    P_w = S.rotation(s) @ P_l + s.pos()
    assert np.allclose(R @ P_l + T, W2O @ (P_w - cam0.pos()), atol=1e-5)
    veh = (tmp_path / 'kitti' / 'calib_cams_to_vehicle.txt').read_text()
    assert 'Tr_L1_to_vehicle:' in veh and 'Tr_RD1_to_vehicle:' in veh
    # without a lidar there is no velodyne file
    rigged.variant.sensors = [q for q in rigged.variant.sensors if q.kind == 'radar']
    names = ex._names(ex.format_by_id('kitti_calib'), rigged, rigged.variant)
    assert 'kitti/calib_velo_to_cam.txt' not in names


# ------------------------------------------------------------------ GNSS antennas and IMUs
def _nav(p):
    """Two GNSS antennas and a dual-antenna INS where Vehicle ▸ Add GNSS antenna / Add IMU put them."""
    v = p.variant
    for kind in ('gnss', 'gnss', 'imu'):
        m = p.sensor_models[S.default_model_id(kind)]
        v.sensors.append(Sensor(kind=kind, model_id=m.id, tag=S.next_tag(kind, v.tags()), name=m.name,
                                **S.default_pose(kind, m, p.vehicle.mesh, p.rig, v.sensors)))
    return v.sensors[-3:]


def test_gnss_and_imu_models_and_placement(tmp_path):
    from vantage.core import rig as rigm
    from vantage.core.samples import tigor_sample_project
    ms = {m.id: m for m in S.builtin_sensor_models()}
    assert {m.kind for m in ms.values()} == set(S.KINDS)
    ins = ms['ins_dual']
    assert ins.kind == 'imu' and ins.antennas == 2 and not ins.has_field and ins.data_bps() == pytest.approx(0.5e6)
    assert ins.heading_sigma(2.0) == pytest.approx(0.1) and ins.heading_sigma(1.0) == pytest.approx(0.2)
    assert math.isnan(ms['imu_mems'].heading_sigma(2.0)) and math.isnan(ins.heading_sigma(0.0))
    ant = ms['gnss_survey']
    assert ant.kind == 'gnss' and ant.data_bps() == 0 and ant.v_fov() == (10.0, 90.0)
    s = Sensor(kind='gnss', model_id='gnss_survey', tag='GN1', z=1.5)
    assert not S.in_field(s, ant, np.array([[0.0, 0.0, 5.0], [10.0, 0.0, 1.5]])).any()   # no field to be in
    assert S.body_box(s, ant)[3]                                                           # drawn as a puck
    assert S.next_tag('gnss', {'GN1', 'F'}) == 'GN2' and S.next_tag('imu', set()) == 'IMU1'
    # a junk file keeps sane values
    j = S.SensorModel.from_dict({'id': 'x', 'kind': 'imu', 'antennas': 7, 'heading_baseline_m': -1})
    assert j.antennas == 2 and j.heading_baseline_m == 2.0
    p = tigor_sample_project()
    g1, g2, imu = _nav(p)
    xf, xr, _, _ = rigm.roof(p.vehicle.mesh)
    assert xr < g1.x < g2.x < xf and g2.x - g1.x > 0.9 and g1.y == g2.y == 0.0
    for g in (g1, g2):            # the base stands on the roof: the phase centre is optical_z above it
        top = S.surface_top(g.x, g.y, p.vehicle.mesh, p.rig)
        assert g.z == pytest.approx(top + ant.optical_z_mm / 1000, abs=2e-3) and top > 1.3
    assert (imu.x, imu.y, imu.z) == (0.0, 0.0, 0.45)
    # on a rack, the antenna stands on the rack, not on the roof under it
    p.rig = rigm.preset('roof_rack', p.vehicle.mesh)
    pose = S.default_pose('gnss', ant, p.vehicle.mesh, p.rig, [])
    plat = next(q for q in p.rig if q.kind == 'box')
    if abs(pose['x'] - plat.x) <= plat.length / 2:
        assert pose['z'] == pytest.approx(plat.z + plat.height / 2 + ant.optical_z_mm / 1000, abs=2e-3)
    path = str(tmp_path / 'nav.vantage')
    p.save(path)
    q = Project.load(path)
    assert [(x.tag, x.kind, x.model_id) for x in q.variant.sensors] == [(x.tag, x.kind, x.model_id)
                                                                        for x in p.variant.sensors]


def test_gnss_and_imu_in_rules_recording_and_exports(tmp_path):
    from vantage.core import exporters as ex
    from vantage.core import recording as RB
    from vantage.core.analysis import Engine
    from vantage.core.rules import sensor_issues
    from vantage.core.samples import tigor_sample_project
    p = tigor_sample_project()
    v = next(x for x in p.variants if x.sensors)
    p.active_uid = v.uid
    g1, g2, imu = _nav(p)
    vg = Engine().vehicle_geom(p)
    assert sensor_issues(p, vg) == []                    # an IMU in the cabin is where it belongs
    g1.z = 1.0                                           # an antenna in the cabin has no sky
    assert [(i.severity, i.objects) for i in sensor_issues(p, vg)] == [('critical', 'GN1')]
    g1.z = 1.55
    # the INS records its stated output; an antenna records nothing itself
    rb = RB.budget(p)
    assert rb.row('IMU1')['mbps'] == pytest.approx(0.5) and rb.row('GN1') is None
    assert rb.totals['counts']['imu'] == 1 and rb.totals['by_kind_mbps']['imu'] == pytest.approx(0.5)
    out = tmp_path / 'x'
    ex.export(['extrinsics_json', 'nuscenes_calib', 'kitti_calib', 'ros2'], p, None, ex.ExportOptions(out_dir=str(out)))
    doc = json.loads((out / 'extrinsics.json').read_text())
    rows = {d['tag']: d for d in doc['sensors']}
    assert rows['GN1']['field']['position'] == 'phase centre' and rows['IMU1']['field']['antenna_inputs'] == 2
    assert 'IMU' in doc['frame']['sensor_frame']
    chans = [r['channel'] for r in json.loads((out / 'nuscenes' / 'sensor.json').read_text())]
    assert not any(c.startswith(('GNSS', 'IMU')) for c in chans) and 'LIDAR_L1' in chans
    lid = next(x for x in v.sensors if x.kind == 'lidar')
    txt = (out / 'kitti' / 'calib_imu_to_velo.txt').read_text().splitlines()
    T = np.array([float(x) for x in txt[2].split()[1:]])
    assert T == pytest.approx([imu.x - lid.x, imu.y - lid.y, imu.z - lid.z], abs=1e-6)   # both frames unrotated
    urdf = (out / 'vantage_cameras.urdf').read_text()
    assert '<link name="GN1_link">' in urdf and '<link name="IMU1_link">' in urdf
    names = ex._names(ex.format_by_id('kitti_calib'), p, v)
    assert 'kitti/calib_imu_to_velo.txt' in names


# ------------------------------------------------------------------ the UI
from test_gui import app, win, pump, settle, _wait     # noqa: E402,F401  (fixtures)


def test_add_edit_pick_drag_and_delete_sensors(app, win, tmp_path):
    from PyQt6.QtCore import QPointF, Qt
    from PyQt6.QtGui import QMouseEvent
    from vantage.ui import sensorview as SV
    st = win.store
    n0 = st.undo.index()
    win.actions_by_id['add_lidar'].trigger()
    win.actions_by_id['add_radar'].trigger()
    settle(app, win)
    lid, rad = st.variant.sensors
    assert (lid.kind, lid.tag, rad.kind, rad.tag) == ('lidar', 'L1', 'radar', 'RD1')
    assert st.sel_obj == ('sensor', rad.uid) and st.undo.index() == n0 + 2
    insp = win.inspector
    assert insp.current == 'sensor'
    page = insp.pages['sensor']
    assert page.zones.isVisible() and page.zones.rowCount() == 2 and not page.b_beams.isVisible()
    # the outliner lists them and selects with the store
    items = win.outliner.findItems('RD1', Qt.MatchFlag.MatchContains | Qt.MatchFlag.MatchRecursive)
    assert items and items[0].isSelected()
    # move the lidar from the inspector; edit a built-in model's figure: an edited copy for this sensor only
    st.select_obj(('sensor', lid.uid))
    settle(app, win)
    z0 = lid.z
    page.pz.set_value(z0 + 0.3 - st.project.origin_offset()[2])
    page.pz.committed.emit(page.pz.value())
    assert st.variant.sensor(lid.uid).z == pytest.approx(z0 + 0.3, abs=1e-4)
    page.fields['range_m'][0].committed.emit(150.0)
    l2 = st.variant.sensor(lid.uid)
    m2 = st.project.sensor_model_for(l2)
    assert m2.id == 'ouster_os1_128_edited' and m2.range_m == 150 and not m2.builtin
    assert st.project.sensor_models['ouster_os1_128'].range_m == 90.0
    st.undo.undo()
    assert st.project.sensor_model_for(st.variant.sensor(lid.uid)).id == 'ouster_os1_128'
    assert 'ouster_os1_128_edited' not in st.project.sensor_models
    # import measured beam angles
    f = tmp_path / 'meta.json'
    f.write_text(json.dumps({'beam_intrinsics': {'beam_altitude_angles': list(np.linspace(20, -20, 64))}}))
    assert page.import_beams(str(f))
    mm = st.project.sensor_model_for(st.variant.sensor(lid.uid))
    assert mm.n_channels() == 64 and mm.v_fov() == pytest.approx((-20.0, 20.0))
    st.undo.undo()
    # the 3D view picks the lidar housing; the scene draws housings and rings
    lines = SV.scene_lines(win.ctx)
    assert any(ld.color.name() == SV.color('lidar').name() for ld in lines)
    assert SV.scene_tris(win.ctx, False).tri.shape[0] > 40
    s = st.variant.sensor(lid.uid)
    c = S.body_box(s, st.project.sensor_model_for(s))[0]
    o = c + np.array([-5.0, 0.0, 2.0])
    assert SV.pick(st.project, o, c - o) == lid.uid
    assert SV.pick(st.project, o, np.array([0.0, 0.0, 1.0])) is None
    # drag the radar in the top view: one undo step
    plan = win.viewarea.panes[1].plan
    plan.fit()
    settle(app, win)
    r = st.variant.sensor(rad.uid)
    x0 = r.x
    q = plan.w2s(r.x, r.y)
    n1 = st.undo.index()

    def ev(t, pos, buttons=Qt.MouseButton.LeftButton):
        return QMouseEvent(t, QPointF(pos), QPointF(pos), Qt.MouseButton.LeftButton, buttons,
                           Qt.KeyboardModifier.NoModifier)
    plan.mousePressEvent(ev(QMouseEvent.Type.MouseButtonPress, q))
    for k in range(1, 6):
        plan.mouseMoveEvent(ev(QMouseEvent.Type.MouseMove, QPointF(q.x() + 4 * k, q.y())))
    plan.mouseReleaseEvent(ev(QMouseEvent.Type.MouseButtonRelease, QPointF(q.x() + 20, q.y()), Qt.MouseButton.NoButton))
    r2 = st.variant.sensor(rad.uid)
    assert r2.x > x0 + 0.05 and st.undo.index() == n1 + 1 and st.sel_obj == ('sensor', rad.uid)
    plan.grab()
    # saved with the project, deleted with Delete, and undo brings it back
    win.delete_selection()
    assert [x.tag for x in st.variant.sensors] == ['L1']
    st.undo.undo()
    assert [x.tag for x in st.variant.sensors] == ['L1', 'RD1']
    while st.undo.index() > n0:
        st.undo.undo()
    assert st.variant.sensors == []
    settle(app, win)


def test_add_gnss_antennas_and_an_imu(app, win):
    from PyQt6.QtCore import Qt
    from vantage.ui import sensorview as SV
    st = win.store
    n0 = st.undo.index()
    for a in ('add_gnss', 'add_gnss', 'add_imu'):
        win.actions_by_id[a].trigger()
    settle(app, win)
    g1, g2, imu = st.variant.sensors
    assert [(x.kind, x.tag) for x in (g1, g2, imu)] == [('gnss', 'GN1'), ('gnss', 'GN2'), ('imu', 'IMU1')]
    assert g1.x < g2.x and st.undo.index() == n0 + 3 and st.sel_obj == ('sensor', imu.uid)
    page = win.inspector.pages['sensor']
    assert win.inspector.current == 'sensor' and page.s_one.header.title.text() == 'IMU IMU1'
    vis = {k for k, (_, row) in page.fields.items() if row.isVisibleTo(page)}
    assert {'antennas', 'heading_sigma_deg', 'power_w'} <= vis and 'channels' not in vis and 'mask_deg' not in vis
    assert 'Heading baseline GN1–GN2' in page.info.text() and not page.zones.isVisibleTo(page)
    # editing an INS figure makes an edited copy, as for a lidar
    page.fields['antennas'][0].committed.emit(1.0)
    assert st.project.sensor_model_for(st.variant.sensor(imu.uid)).antennas == 1
    st.undo.undo()
    st.select_obj(('sensor', g2.uid))
    settle(app, win)
    vis = {k for k, (_, row) in page.fields.items() if row.isVisibleTo(page)}
    assert vis == {'mask_deg', 'power_w'} and 'Secondary antenna' in page.info.text()
    # the baseline and the IMU axes are drawn; the outliner names the group for what it holds
    lines = SV.scene_lines(win.ctx)
    assert any(ld.color.name() == SV.color('gnss').name() for ld in lines)
    assert any(ld.color.name() == SV.color('imu').name() for ld in lines)
    assert win.outliner.findItems('Sensors (3)', Qt.MatchFlag.MatchContains | Qt.MatchFlag.MatchRecursive)
    menus = {a.text().replace('&', ''): a.menu() for a in win.menuBar().actions()}
    subs = [a.text() for a in menus['Vehicle'].actions() if a.menu()]
    assert 'Add GNSS antenna' in subs and 'Add IMU' in subs
    while st.undo.index() > n0:
        st.undo.undo()
    assert st.variant.sensors == []
    settle(app, win)
