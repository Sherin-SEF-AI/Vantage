"""Reference rigs (core/reference_rigs.py): every figure sourced with its provenance, counts consistent with the sensors
listed, the nuScenes and KITTI poses as their calibration files give them, variants built with real library modules
and sensors, and the reference projects analysed."""
import math

import numpy as np
import pytest

from vantage.core import reference_rigs as RR
from vantage.core.analysis import Engine

LIB = RR.library()


def test_every_rig_is_sourced_and_every_figure_has_its_provenance():
    assert len(LIB) >= 8 and len({r.id for r in LIB}) == len(LIB)
    companies = {r.company for r in LIB}
    assert {'Waymo', 'Tesla', 'Wayve', 'Mobileye'} <= companies
    for r in LIB:
        assert r.sources and all(u.startswith('https://') and w for w, u in r.sources), r.id
        assert r.layout in ('measured', 'mixed', 'estimated', 'none'), r.id
        for s in r.sensors:
            for k, p in s.prov.items():
                assert p in RR.PROVENANCE, (r.id, s.name, k)
                assert getattr(s, k) not in (None, ''), (r.id, s.name, k)       # a provenance only for a value
            for k in ('hfov_deg', 'range_m', 'position_m', 'yaw'):
                if getattr(s, k) is not None:
                    assert k in s.prov, (r.id, s.name, k)                      # every value says where it is from


def test_counts_agree_with_the_sensors_listed():
    for r in LIB:
        if r.layout == 'none' and not r.sensors:
            continue
        for kind, lst in (('cameras', r.cameras), ('lidars', r.lidars), ('radars', r.radars)):
            n = r.counts.get(kind)
            if n is not None and (lst or r.layout != 'none'):
                assert n == len(lst), (r.id, kind)
    rigs = RR.by_id()
    assert rigs['waymo_6th_gen'].counts == {'cameras': 13, 'lidars': 4, 'radars': 6}
    assert rigs['waymo_5th_gen'].counts['cameras'] == 29 and rigs['waymo_5th_gen'].counts['lidars'] is None
    assert rigs['mobileye_supervision'].counts['cameras'] == 11
    assert rigs['wayve'].counts['cameras'] is None          # not published, never guessed


def test_nuscenes_poses_are_the_calibration_file_s():
    r = RR.by_id()['nuscenes']
    front = next(c for c in r.cameras if c.name == 'CAM_FRONT')
    assert front.position_m == (1.700791, 0.015946, 1.510958)
    assert front.hfov_deg == pytest.approx(2 * math.degrees(math.atan(800 / 1266.417)), abs=0.01)
    assert abs(front.yaw) < 0.5 and abs(front.pitch) < 0.5
    back = next(c for c in r.cameras if c.name == 'CAM_BACK')
    assert abs(abs(back.yaw) - 180) < 0.5 and back.hfov_deg == pytest.approx(89.34, abs=0.01)
    fl = next(c for c in r.cameras if c.name == 'CAM_FRONT_LEFT')
    assert 50 < fl.yaw < 60
    # the quaternion conversion round-trips through Vantage's own camera frame
    from vantage.core.analysis import camera_frames
    from vantage.core.project import Camera
    for ch, t, q, fx, cy, cx in RR.NUSCENES_CAMERAS:
        y, p, ro = RR.camera_ypr_from_quaternion(q)
        R, W2O = camera_frames(Camera(yaw=y, pitch=p, roll=ro))
        np.testing.assert_allclose(W2O.T, RR._q_to_R(q), atol=1e-9)      # optical → world equals the file's
    radar = next(s for s in r.radars if s.name == 'RADAR_FRONT_LEFT')
    assert radar.position_m == (2.422, 0.8, 0.78) and 85 < radar.yaw < 92


def test_kitti_relative_poses_are_the_calibration_s():
    r = RR.by_id()['kitti']
    c = {s.name.split(' (')[0]: np.array(s.position_m) for s in r.cameras}
    assert c['Cam 1'][1] - c['Cam 0'][1] == pytest.approx(-0.537, abs=1e-3)        # T_01: 0.537 m to the right
    assert c['Cam 2'][1] - c['Cam 0'][1] == pytest.approx(0.0596, abs=1e-3)        # T_02: 0.06 m to the left
    assert c['Cam 3'][1] - c['Cam 0'][1] == pytest.approx(-0.4731, abs=1e-3)
    lid = r.lidars[0]
    assert c['Cam 0'][0] - lid.position_m[0] == pytest.approx(0.27, abs=0.01)     # velo_to_cam: 0.27 m behind
    assert all(s.position_m[2] == 1.65 for s in r.cameras) and lid.position_m[2] == 1.73
    # an independent check: the calibration puts the Velodyne 0.076 m above cam0, the paper's heights 0.08 m
    _, velo = RR._kitti_offsets()
    assert velo[2] == pytest.approx(1.73 - 1.65, abs=0.01)
    assert all(s.prov['position_m'] == 'estimated' for s in r.cameras)            # absolute fore-aft: estimated


def test_unpublished_layouts_are_not_turned_into_variants():
    from vantage.core import samples
    p = samples.template_project('lincoln_mkz_2017')
    for rid in ('waymo_5th_gen', 'waymo_6th_gen', 'mobileye_supervision', 'wayve', 'argoverse2',
                'waymo_open_dataset'):
        with pytest.raises(ValueError, match='not published'):
            RR.to_variant(p, RR.by_id()[rid])


@pytest.mark.parametrize('rid', ['nuscenes', 'kitti', 'tesla_hw3'])
def test_a_reference_project_has_real_modules_and_analyses(rid):
    p = RR.reference_project(rid)
    rig = RR.by_id()[rid]
    v = p.variant
    assert v.name.startswith(rig.company) and len(v.cameras) == len(rig.cameras)
    assert p.vehicle.template == rig.library_vehicle_id
    for cam, ref in zip(v.cameras, rig.cameras):
        assert cam.module_id in p.modules and cam.lens_id in p.lenses and p.modules[cam.module_id].builtin
        if ref.library_model:
            assert cam.module_id == ref.library_model
        if ref.prov.get('position_m') in ('measured', 'stated'):
            assert (cam.x, cam.y, cam.z) == pytest.approx(ref.position_m, abs=1e-4)
    for s, ref in zip(v.sensors, rig.lidars + rig.radars):
        from vantage.core.sensors import builtin_sensor_models
        assert s.model_id in {m.id for m in builtin_sensor_models()}
        assert (s.x, s.y, s.z) == pytest.approx(ref.position_m, abs=1e-6)
    assert 'Sources:' in v.note and rig.layout in v.note
    res = Engine().analyse(p)
    assert res is not None
    if rid == 'nuscenes':
        # six level cameras at 1.5 m (39° vertical) see the ground only beyond about 4 m, so not the 2 m ROI; their
        # fields close the circle round the car
        yaws = sorted(c.yaw % 360 for c in v.cameras)
        gaps = np.diff(yaws + [yaws[0] + 360])
        assert sum(c.hfov_deg for c in rig.cameras) > 360 and gaps.max() < 80
    if rid == 'tesla_hw3':                # estimated mounts land on the body's skin, the repeaters on the fenders
        rep = [c for c in v.cameras if 'Rearward' in c.name]
        assert all(abs(c.y) > 0.8 and 1.5 < c.x < 2.8 for c in rep) and res.metrics['covered_pct'] > 50


def test_library_lens_choices_record_the_field_mismatch():
    from vantage.core import samples
    p = samples.template_project('renault_zoe_ph1')
    rig = RR.by_id()['nuscenes']
    for c in rig.cameras:
        mid, lid, h, why = RR.closest_camera(p, c)
        assert mid == 'basler_aca1600_60gc' and 'H ' in why and f'{c.hfov_deg:.1f}' in why
    tesla = RR.by_id()['tesla_hw3']
    side = next(c for c in tesla.cameras if c.hfov_deg is None)
    assert 'not published' in RR.closest_camera(p, side)[3]


def test_summary_table():
    t = RR.summary_table()
    assert len(t) == len(LIB) and all(r['sources'] for r in t)
    nus = next(r for r in t if r['id'] == 'nuscenes')
    assert nus['cameras'] == 6 and nus['layout'] == 'measured' and nus['provenance']['measured'] > 0
