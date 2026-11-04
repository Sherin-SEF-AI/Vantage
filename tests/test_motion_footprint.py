"""Motion footprints (core/motion_footprint.py) against closed forms: a straight drive shifts every sampled point by
v·Δt along x, a pure yaw rate turns it about the rear-axle origin by ω·Δt, a rolling shutter's rows are sampled later
from the top of the image to the bottom, zero motion gives the static footprint exactly, the lidar's sweep time
follows its rotation, and a camera phase-locked to the lidar has a smaller temporal wedge than one forced onto a
trigger line with the camera opposite it."""
import copy
import math

import numpy as np
import pytest

from vantage.core import motion_footprint as MF
from vantage.core import samples
from vantage.core import sensors as SN
from vantage.core import timing as TM
from vantage.core.analysis import Engine
from vantage.core.lens import resolve


@pytest.fixture(scope='module')
def rig():
    p = samples.tigor_sample_project()
    p.active_uid = next(v.uid for v in p.variants if v.sensors)
    e = Engine()
    return p, e, e.analyse(p)


def test_ego_pose_closed_forms():
    psi, x, y = MF.ego_pose([0.0, 0.1, 0.5], 10.0, 0.0)
    assert np.allclose(x, [0.0, 1.0, 5.0]) and np.allclose(y, 0.0) and np.allclose(psi, 0.0)
    w = math.radians(20.0)
    psi, x, y = MF.ego_pose(0.5, 10.0, 20.0)
    assert psi == pytest.approx(w * 0.5)
    assert (x, y) == pytest.approx((10.0 / w * math.sin(w * 0.5), 10.0 / w * (1 - math.cos(w * 0.5))))
    # a very small yaw rate tends to the straight line
    _, x2, y2 = MF.ego_pose(0.5, 10.0, 1e-9)
    assert (float(x2), float(y2)) == pytest.approx((5.0, 0.0), abs=1e-6)


def test_zero_motion_is_the_static_footprint_exactly(rig):
    p, e, r = rig
    w = MF.warp_footprints(p, e, 0.0, 0.0, result=r)
    assert len(w) == len([c for c in p.variant.cameras if c.visible])
    for x in w.values():
        assert np.array_equal(x['segments_warped'], x['segments']) and x['max_shift_m'] == 0.0
        for a, b in zip(x['border_warped'], x['border']):
            assert np.array_equal(a, b)
    for lid in MF.lidar_sweep(p, e, 0.0, 0.0, far=20.0):
        for ring in lid['rings']:
            assert np.array_equal(ring['warped'], ring['static'])


def test_a_straight_drive_shifts_by_speed_times_the_sample_time(rig):
    p, e, r = rig
    st = TM.TimingSettings.from_dict({'readout_ms': 0.0, 'exposure_ms': 2.0})     # a global shutter
    v = 15.0
    w = MF.warp_footprints(p, e, v, 0.0, result=r, settings=st, ref_ms=10.0)
    for x in w.values():
        dt = x['segments_dt_ms'].reshape(-1)
        assert np.allclose(dt, dt[0])                                  # one time for the whole frame
        assert dt[0] == pytest.approx(x['trigger_ms'] + 1.0 - 10.0)     # trigger + exposure / 2, after the reference
        d = (x['segments_warped'] - x['segments']).reshape(-1, 3)
        assert np.allclose(d[:, 0], v * dt / 1000.0) and np.allclose(d[:, 1:], 0.0)
        assert abs(x['trigger_ms'] - 10.0) <= 1000.0 / 30.0 / 2 + 1e-9  # the frame nearest the reference


def test_a_pure_yaw_rate_turns_points_about_the_rear_axle(rig):
    p, e, r = rig
    w = MF.warp_footprints(p, e, 0.0, 10.0, result=r)
    for x in w.values():
        P = x['segments'].reshape(-1, 3)
        Q = x['segments_warped'].reshape(-1, 3)
        a = np.radians(10.0) * x['segments_dt_ms'].reshape(-1) / 1000.0
        assert np.allclose(Q[:, 0], np.cos(a) * P[:, 0] - np.sin(a) * P[:, 1])
        assert np.allclose(Q[:, 1], np.sin(a) * P[:, 0] + np.cos(a) * P[:, 1])
        assert np.allclose(np.hypot(Q[:, 0], Q[:, 1]), np.hypot(P[:, 0], P[:, 1]))   # about the origin


def test_rolling_shutter_rows_are_sampled_top_to_bottom(rig):
    p, e, r = rig
    st = TM.TimingSettings.from_dict({'readout_ms': 20.0, 'exposure_ms': 2.0})
    cam = p.variant.by_tag('F')
    intr = resolve(p.module_for(cam), p.lens_for(cam))
    # ground points straight ahead, near to far: near points are low in the image (late rows), far ones higher
    P = np.c_[np.linspace(cam.x + 0.6, cam.x + 12.0, 40), np.zeros(40), np.zeros(40)]
    dt, rows = MF.sample_times(cam, intr, P, 5.0, st)
    order = np.argsort(rows)
    assert np.all(np.diff(dt[order]) >= -1e-12) and rows[0] > rows[-1]
    assert dt[order][0] >= 5.0 + 1.0 - 1e-9 and dt[order][-1] <= 5.0 + 20.0 + 1.0 + 1e-9
    assert np.allclose(dt, 5.0 + rows / intr.height * 20.0 + 1.0)


def test_lidar_sweep_times_follow_the_rotation(rig):
    p, e, _ = rig
    lids = MF.lidar_sweep(p, e, 10.0, 0.0, far=25.0, az_step=2.0)
    assert len(lids) == 1 and lids[0]['rings']
    L = lids[0]
    s = next(x for x in p.variant.sensors if x.uid == L['uid'])
    sw = TM.Sweep.of(s, p.sensor_model_for(s))
    for ring in L['rings']:
        P = ring['static']
        _, az, _ = SN.angles(s, P)
        want = TM._wrap(sw.time_of(az), sw.period_ms)
        assert np.allclose(ring['dt_ms'], want) and np.all(np.abs(ring['dt_ms']) <= sw.period_ms / 2 + 1e-9)
        assert np.allclose(ring['warped'][:, 0] - P[:, 0], 10.0 * ring['dt_ms'] / 1000.0)


def test_a_phase_locked_camera_has_the_smaller_temporal_wedge(rig):
    p, e, _ = rig

    def wedges(sync):
        q = copy.deepcopy(p)
        for c in q.variant.cameras:
            c.sync = sync.get(c.tag, 'A')
        pl = TM.plan(q)
        return pl, {w['camera_tag']: w for w in MF.temporal_wedges(q, e, plan=pl)}

    pl1, w1 = wedges({'F': 'B'})                    # F alone on its line: its trigger is its ideal
    pl2, w2 = wedges({'F': 'B', 'R': 'B'})          # F sharing a line with the camera opposite it
    f1, f2 = w1['F'], w2['F']
    assert abs(pl1.cam(p.variant.by_tag('F').uid)['misalign_ms']) < 1e-6
    assert abs(pl2.cam(p.variant.by_tag('F').uid)['misalign_ms']) > 5.0
    assert f1['max_dt_ms'] < f2['max_dt_ms'] - 5.0
    # phase-locked: the lag across the field is the sweep's own, Δt = sense·(φ − φ_axis) / (0.36·f), symmetric
    lid = pl1.lidar
    half = f1['field_span_deg'] / 2
    assert f1['max_dt_ms'] == pytest.approx(max(half, f1['field_span_deg'] - half) / (0.36 * lid.rate_hz), rel=0.05)
    assert float(np.abs(f1['dt_ms']).min()) < 0.2
    assert f1['wedges'] and all(len(w) == 2 for w in f1['wedges']) and f1['threshold_ms'] == 10.0
    # a looser threshold leaves less of the field in the wedge; one beyond the worst lag leaves none
    loose = {w['camera_tag']: w for w in MF.temporal_wedges(p, e, threshold_ms=15.0, plan=TM.plan(p))}
    tight = {w['camera_tag']: w for w in MF.temporal_wedges(p, e, threshold_ms=5.0, plan=TM.plan(p))}
    assert all(loose[k]['wedge_deg'] <= tight[k]['wedge_deg'] for k in loose)
    none = MF.temporal_wedges(p, e, threshold_ms=1000.0)
    assert all(w['wedges'] == [] and w['wedge_deg'] == 0.0 for w in none)


def test_without_a_spinning_lidar_there_are_no_sweeps(rig):
    p, e, _ = rig
    q = copy.deepcopy(p)
    for s in q.variant.sensors:
        if s.kind == 'lidar':
            s.visible = False
    assert MF.lidar_sweep(q, e, 10.0, 0.0) == [] and MF.temporal_wedges(q, e) == []
