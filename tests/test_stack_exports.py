"""Exports into the AV research stack (core/stack_exports.py): every file parses, every transform recomposes to the
pose the variant has in the vehicle frame (recomputed here independently), Autoware's roll/pitch/yaw recompose as URDF
fixed axes, HUGSIM's Euler angles as scipy's intrinsic XYZ, the ego crops keep every own-vehicle pixel out."""
import ast
import json
import math
import os

import numpy as np
import pytest
import yaml

from vantage.core import samples
from vantage.core import stack_exports as SX
from vantage.core.analysis import Engine
from vantage.core.mathutil import rot_zyx, BODY_TO_OPTICAL
from vantage.core.sensors import rotation


@pytest.fixture(scope='module')
def rig():
    p = samples.tigor_sample_project()
    p.active_uid = next(v.uid for v in p.variants if v.sensors)
    return p


@pytest.fixture(scope='module')
def eng():
    return Engine()


@pytest.fixture(scope='module')
def out(rig, eng, tmp_path_factory):
    d = tmp_path_factory.mktemp('stack')
    return str(d), SX.export_stack(rig, eng, str(d))


def _T_cam(c):
    """Optical frame -> vehicle, from the camera's own fields (not core.exporters)."""
    R = rot_zyx(c.yaw, c.pitch, c.roll)
    T = np.eye(4)
    T[:3, :3] = R @ BODY_TO_OPTICAL.T
    T[:3, 3] = c.pos()
    return T


def _T_sensor(s):
    T = np.eye(4)
    T[:3, :3] = rotation(s)
    T[:3, 3] = s.pos()
    return T


def _urdf(roll, pitch, yaw):
    cr, sr, cp, sp, cy, sy = (math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw),
                              math.sin(yaw))
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _lidar(rig):
    return next(s for s in rig.variant.sensors if s.kind == 'lidar' and s.visible)


def test_every_target_writes_files_that_parse(out):
    d, res = out
    assert set(res) == set(SX.TARGETS) and all(isinstance(v, list) and v for v in res.values()), res
    for paths in res.values():
        for p in paths:
            assert os.path.getsize(p) > 0 and os.path.commonpath([d, p]) == d
            if p.endswith(('.yaml', '.yml')):
                yaml.safe_load(open(p))
            elif p.endswith('.json'):
                json.load(open(p))
            elif p.endswith('.py'):
                ast.parse(open(p).read())


def test_autoware_recomposes_to_the_vehicle_frame_poses(rig, out):
    d, _ = out
    cfg = os.path.join(d, 'autoware/vantage_sensor_kit_description/config')
    base = yaml.safe_load(open(f'{cfg}/sensors_calibration.yaml'))['base_link']['sensor_kit_base_link']
    kit = yaml.safe_load(open(f'{cfg}/sensor_kit_calibration.yaml'))['sensor_kit_base_link']

    def T(e):
        M = np.eye(4)
        M[:3, :3] = _urdf(e['roll'], e['pitch'], e['yaw'])
        M[:3, 3] = [e['x'], e['y'], e['z']]
        return M
    T_bk = T(base)
    lid = _lidar(rig)
    m = rig.sensor_model_for(lid)
    # the kit sits at the lidar's base, base_link's axes
    np.testing.assert_allclose(T_bk[:3, 3], _T_sensor(lid)[:3, 3] - rotation(lid) @ [0, 0, m.optical_z_mm / 1000],
                               atol=2e-6)
    cams = sorted(rig.variant.cameras, key=lambda c: (c.slot, c.tag))
    for i, c in enumerate(cams):
        got = T_bk @ T(kit[f'camera{i}/camera_link'])
        # camera_link is the camera body (x along the axis); its optical frame is rpy(-pi/2, 0, -pi/2) below it
        opt = got.copy()
        opt[:3, :3] = got[:3, :3] @ _urdf(-math.pi / 2, 0, -math.pi / 2)
        np.testing.assert_allclose(opt, _T_cam(c), atol=2e-6)
    np.testing.assert_allclose((T_bk @ T(kit[f'{lid.tag.lower()}_base_link']))[:3, :3], rotation(lid), atol=1e-6)
    assert len([k for k in kit if k.endswith('/radar_link')]) == len([s for s in rig.variant.sensors
                                                                       if s.kind == 'radar'])
    assert os.path.exists(os.path.join(d, 'autoware/camera_info/camera0.yaml'))


def test_navsim_and_mmdet3d_transforms(rig, out):
    d, _ = out
    lid = _lidar(rig)
    T_vl = _T_sensor(lid)
    nav = json.load(open(os.path.join(d, 'navsim/vantage_cams.json')))
    by_tag = {c.tag: c for c in rig.variant.cameras}
    # all eight NAVSIM slots are keys (Cameras.from_camera_dict reads all eight); the unfilled ones are empty
    assert list(nav['cams']) == list(SX.NAVSIM_SLOTS)
    filled = {k: v for k, v in nav['cams'].items() if v['data_path'] is not None}
    assert len(filled) == min(8, len(rig.variant.cameras)) == len(nav['vantage']['slots'])
    for slot, cam in filled.items():
        c = by_tag[nav['vantage']['slots'][slot]]
        S = np.eye(4)
        S[:3, :3], S[:3, 3] = np.array(cam['sensor2lidar_rotation']), cam['sensor2lidar_translation']
        np.testing.assert_allclose(T_vl @ S, _T_cam(c), atol=1e-6)
        assert np.allclose(np.array(cam['cam_intrinsic'])[2], [0, 0, 1]) and len(cam['distortion']) == 5
    assert nav['vantage']['slots']['CAM_F0'] == 'F' and nav['vantage']['slots']['CAM_B0'] == 'R'
    w, x, y, z = nav['lidar2ego_rotation']
    from scipy.spatial.transform import Rotation
    np.testing.assert_allclose(Rotation.from_quat([x, y, z, w]).as_matrix(), rotation(lid), atol=1e-8)
    mm = json.load(open(os.path.join(d, 'mmdet3d/vantage_rig_infos.json')))
    np.testing.assert_allclose(np.array(mm['lidar_points']['lidar2ego']), T_vl, atol=1e-8)
    for name, im in mm['images'].items():
        cam2ego, lidar2cam = np.array(im['cam2ego']), np.array(im['lidar2cam'])
        np.testing.assert_allclose(cam2ego @ lidar2cam, T_vl, atol=1e-7)          # lidar -> cam -> ego = lidar -> ego
    assert mm['images']['CAM_FRONT']['vantage_lens']['camera'] == 'F'
    tree = ast.parse(open(os.path.join(d, 'mmdet3d/vantage_nus_multiview.py')).read())
    names = {t.id for n in tree.body if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}
    assert {'dataset_type', 'data_prefix', 'train_pipeline', 'train_dataloader', 'vantage_ego_crops'} <= names


def test_hugsim_euler_angles_are_scipy_intrinsic_xyz(rig, out):
    from scipy.spatial.transform import Rotation
    d, _ = out
    doc = yaml.safe_load(open(os.path.join(d, 'hugsim/vantage_camera.yaml')))
    assert 'CAM_FRONT' in doc['cams'] and doc['cam_align'][0][1] == 'CAM_FRONT'
    f = rig.variant.by_tag('F')
    e = doc['cams']['CAM_FRONT']['extrinsics']
    v2c = np.eye(4)
    v2c[:3, :3] = Rotation.from_euler('XYZ', e['v2c_rot'], degrees=True).as_matrix()   # HUGSIM's rt2pose
    v2c[:3, 3] = e['v2c_trans']
    np.testing.assert_allclose(np.linalg.inv(v2c), _T_cam(f), atol=1e-6)
    # l2c: the lidar in nuScenes LIDAR_TOP axes (x right, y forward)
    l2c = np.eye(4)
    l2c[:3, :3] = Rotation.from_euler('XYZ', e['l2c_rot'], degrees=True).as_matrix()
    l2c[:3, 3] = e['l2c_trans']
    lid = _T_sensor(_lidar(rig))
    nus = lid.copy()
    nus[:3, :3] = lid[:3, :3] @ np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    np.testing.assert_allclose(v2c @ nus, l2c, atol=1e-6)
    it = doc['cams']['CAM_FRONT']['intrinsics']
    m = rig.module_for(f)
    assert (it['W'], it['H']) == (m.width_px, m.height_px) and 0 < it['fovx'] < 180


def test_the_ego_crop_keeps_every_vehicle_pixel_out(rig, eng):
    from vantage.core import exporters as EX
    ctx = SX._ctx(rig)
    seen = 0
    for cm in ctx.models:
        mask, outside = SX.ego_mask(rig, eng, cm)
        cr = SX.ego_crop(mask, outside)
        x0, y0, x1, y1 = cr['rect_xyxy']
        assert x1 > x0 and y1 > y0
        assert not (mask | outside)[y0:y1, x0:x1].any(), cm.tag
        assert not mask[:cr['keep_rows_above']][mask.shape[0] // 2:].any()
        seen += mask.any()
    assert seen >= 3                                       # the sample's cameras see the car in most views
    # the rectangle search against a brute force on a small random grid
    rng = np.random.default_rng(1)
    bad = rng.random((9, 11)) < 0.25
    x0, y0, x1, y1 = SX.largest_free_rect(bad)
    best = 0
    for a in range(9):
        for b in range(a + 1, 10):
            for c in range(11):
                for e in range(c + 1, 12):
                    if not bad[a:b, c:e].any():
                        best = max(best, (b - a) * (e - c))
    assert (x1 - x0) * (y1 - y0) == best and not bad[y0:y1, x0:x1].any()


def test_openpcdet_ranges_and_lerobot_info(rig, out):
    d, _ = out
    cfg = yaml.safe_load(open(os.path.join(d, 'openpcdet/vantage_custom_dataset.yaml')))
    pcr = cfg['POINT_CLOUD_RANGE']
    vx, vy, vz = cfg['DATA_PROCESSOR'][2]['VOXEL_SIZE']
    assert round((pcr[3] - pcr[0]) / vx) % 16 == 0 and round((pcr[4] - pcr[1]) / vy) % 16 == 0
    assert round((pcr[5] - pcr[2]) / vz) == 40
    lid = _lidar(rig)
    assert pcr[2] == pytest.approx(-(lid.z + 0.3), abs=0.01)             # 0.3 m below the ground, lidar frame
    with pytest.raises(ValueError):
        SX.export_openpcdet(samples.tigor_sample_project(), None, d)     # the 6-cam variant has no lidar
    info = json.load(open(os.path.join(d, 'lerobot/meta/info.json')))
    assert info['codebase_version'] == 'v3.0' and info['fps'] > 0
    imgs = {k: v for k, v in info['features'].items() if k.startswith('observation.images.')}
    assert len(imgs) == len(rig.variant.cameras)
    f = imgs['observation.images.f']
    m = rig.module_for(rig.variant.by_tag('F'))
    assert f['shape'] == [m.height_px, m.width_px, 3] and f['names'] == ['height', 'width', 'channel']
    assert {'timestamp', 'frame_index', 'episode_index', 'index', 'task_index'} <= set(info['features'])
    cal = json.load(open(os.path.join(d, 'lerobot/meta/vantage_calibration.json')))
    np.testing.assert_allclose(np.array(cal['cameras']['observation.images.f']['T_vehicle_camera']),
                               _T_cam(rig.variant.by_tag('F')), atol=1e-7)


def test_slot_assignment_by_yaw(rig):
    ctx = SX._ctx(rig)
    s = SX.assign_slots(ctx.models, SX.NUSCENES_SLOTS)
    tags = {k: v.cam.tag for k, v in s.items()}
    assert tags['CAM_FRONT'] == 'F' and tags['CAM_BACK'] == 'R'
    assert tags['CAM_FRONT_LEFT'] in ('ML', 'QL') and tags['CAM_FRONT_RIGHT'] in ('MR', 'QR')
