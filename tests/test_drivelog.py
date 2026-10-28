"""Drive-log check: a synthetic drive of the data rig written as a ROS 2 bag with known faults (a camera dropping 10 %
of its frames, a camera triggered 7 ms late, a camera rendered 0.6° off in yaw while its calibration is the plan, the
lidar phase-locked 20° off, a camera topic missing, a camera at the wrong image size, a radar stamped ahead of the
log), each found with the right magnitude; the maths on its own (CDR headers, point-time units, the sweep fit and phase,
drops, depth edges, motion compensation, sync offsets); settings, summary, export, a nuScenes folder; and the page.

The drive: the sample data rig with its cameras pitched 15° down (looking out like a data-collection rig's, so the
roof lidar shares their view) and a smaller test camera module (480 × 384 px, the same sensor size and lens), an INS
and a GNSS antenna added, standing for 0.5 s and then accelerating at 3 m/s² straight ahead in a garage with pillars
and boxes 7–20 m round it. Images are rendered with core.synth.render_world at the exposure centre of each frame the
check samples (other frames repeat the nearest rendered image); lidar points are traced column by column at their
firing time from the moving lidar (not motion compensated, as a real driver delivers them)."""
import json
import math
import os
import time

import numpy as np
import pytest

from vantage.core import drivelog as DL
from vantage.core import rosbag as RB
from vantage.core import samples, sensors as SN, synth, timing as TM
from vantage.core.analysis import Engine
from vantage.core.lens import Module
from vantage.core.mathutil import rot_zyx
from vantage.core.project import Sensor

BASE = 1_700_000_000_000_000_000           # a sync pulse (multiple of the 100 ms lidar period)
DUR = 1.2                                  # s
T_STILL, ACC = 0.5, 3.0                    # stands still, then accelerates straight ahead
PHASE_TRUE = 20.0                          # the lidar is locked at 20°, the plan says 0°
LATE = {'R': 7.0}                          # ms
YAW_ERR = {'ML': 0.6}                      # deg, images rendered so; the calibration stays the plan
DROP_EVERY = ('MR', 10, 5)                 # MR loses every 10th frame (j % 10 == 5)
WRONG_SIZE = ('QL', 480, 360)
MISSING = 'QR'
FUTURE = ('RD3', 50.0)                     # RD3's stamps run 50 ms ahead of the recorder
GYRO_BIAS = np.radians([0.2, -0.1, 0.05])  # rad/s
N_FRAMES = 6


def x_of(t):
    """Vehicle travel (m) along +x at t s after the start."""
    t = np.asarray(t, float)
    return np.where(t < T_STILL, 0.0, 0.5 * ACC * (t - T_STILL) ** 2)


def v_of(t):
    t = np.asarray(t, float)
    return np.where(t < T_STILL, 0.0, ACC * (t - T_STILL))


def make_project():
    p = samples.tigor_sample_project()
    v = next(v for v in p.variants if v.sensors)
    p.active_uid = v.uid
    m = p.modules['sturdecam31']
    p.modules['test_cam'] = Module.from_dict(dict(m.to_dict(), id='test_cam', name='Test camera', width_px=480,
                                                  height_px=384, pixel_um=m.pixel_um * m.width_px / 480,
                                                  builtin=False))
    for c in v.cameras:
        c.module_id = 'test_cam'
        c.pitch = 15.0
    taken = v.tags()
    for kind, mid in (('imu', 'ins_dual'), ('gnss', 'gnss_survey')):
        tag = SN.next_tag(kind, taken)
        taken.add(tag)
        v.sensors.append(Sensor(kind=kind, model_id=mid, tag=tag, name=tag,
                                **SN.default_pose(kind, p.sensor_models[mid], p.vehicle.mesh, p.rig, v.sensors)))
    return p, v


def garage_world(p):
    w = synth.parking_world(p)
    w.wall_r, w.wall_h = 30.0, 8.0
    w.boxes, w.cylinders = [], []
    rng = np.random.default_rng(3)
    cols = [(0.16, 0.24, 0.42), (0.85, 0.84, 0.80), (0.58, 0.12, 0.10), (0.12, 0.12, 0.13), (0.20, 0.45, 0.25)]
    n = 60
    for k in range(n):
        a = 2 * math.pi * (k + 0.3 * rng.random()) / n
        r = 7.0 + 13.0 * rng.random()
        x, y = w.centre[0] + r * math.cos(a), w.centre[1] + r * math.sin(a)
        c = cols[k % len(cols)]
        if k % 2:
            w.boxes.append((x, y, float(rng.random() * 90), 1.0 + rng.random(), 1.0 + rng.random(),
                            2.5 + rng.random(), c))
        else:
            w.cylinders.append((x, y, 0.3 + 0.2 * rng.random(), 3.0 + rng.random(), c, False))
    return w




def shifted(world, dx):
    """The world seen from the vehicle dx metres further along +x."""
    w = synth.World(centre=(world.centre[0] - dx, world.centre[1]),
                    markings=[(a - dx, b, c - dx, d, e, f) for a, b, c, d, e, f in world.markings],
                    boxes=[(b[0] - dx,) + tuple(b[1:]) for b in world.boxes],
                    cylinders=[(c[0] - dx,) + tuple(c[1:]) for c in world.cylinders],
                    wall_r=world.wall_r, wall_h=world.wall_h, boards=world.boards, sun=world.sun)
    return w


def lidar_sweep(p, eng, s, world, t0_s, phase, rate=10.0, sense=-1):
    """One sweep from t0_s (a sync pulse): each column fired when the head faces its azimuth, traced from where the
    lidar is at that instant; points in the sensor frame then, with t (ns after the sweep start) and ring."""
    m = p.sensor_model_for(s)
    E = m.elevations()
    az = SN.scan_azimuths(m)
    C, N = len(E), len(az)
    Ds = SN.directions(E, az).reshape(-1, 3)
    R = SN.rotation(s)
    D = Ds @ R.T
    tcol = ((sense * (az - phase)) % 360.0) / (360.0 * rate)
    t = np.tile(tcol, C)
    O = np.tile(s.pos(), (len(D), 1))
    O[:, 0] += x_of(t0_s + t)
    best = np.full(len(D), np.inf)
    with np.errstate(divide='ignore', invalid='ignore'):
        best = np.minimum(best, np.where(D[:, 2] < -1e-6, -O[:, 2] / D[:, 2], np.inf))
        ox, oy = O[:, 0] - world.centre[0], O[:, 1] - world.centre[1]
        a = D[:, 0] ** 2 + D[:, 1] ** 2
        b = 2 * (ox * D[:, 0] + oy * D[:, 1])
        c = ox * ox + oy * oy - world.wall_r ** 2
        tw = (-b + np.sqrt(np.maximum(b * b - 4 * a * c, 0))) / (2 * np.maximum(a, 1e-12))
    zw = O[:, 2] + tw * D[:, 2]
    best = np.minimum(best, np.where((tw > 0) & (zw >= 0) & (zw <= world.wall_h) & (a > 1e-9), tw, np.inf))
    for bx in world.boxes:
        best = np.minimum(best, synth._ray_box_many(O, D, bx)[0])
    for x, y, r, hh, _, _ in world.cylinders:
        best = np.minimum(best, synth._ray_cyl_many(O, D, x, y, r, hh))
    bd, _ = eng.cube(s.pos()).lookup(D)
    best = np.where(np.isfinite(bd) & (bd > 0.05) & (bd < best), bd, best)
    ok = np.isfinite(best) & (best >= m.range_min_m) & (best <= m.range_m)
    o = np.argsort(t[ok], kind='stable')
    sel = np.nonzero(ok)[0][o]
    arr = np.zeros(len(sel), [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('intensity', 'f4'), ('t', 'u4'),
                              ('ring', 'u2')])
    P = Ds[sel] * best[sel, None]
    arr['x'], arr['y'], arr['z'] = P[:, 0], P[:, 1], P[:, 2]
    arr['intensity'] = 50.0
    arr['t'] = np.round(t[sel] * 1e9).astype(np.uint32)
    arr['ring'] = np.repeat(np.arange(C), N)[sel]
    return arr


def make_drive(path, p, v, eng, n_frames=N_FRAMES, odom=True):
    """Write the synthetic drive as a bag at path; returns the truth (what was injected)."""
    import cv2
    world = garage_world(p)
    plan = TM.plan(p, v)
    tm = TM.TimingSettings.from_dict(p.analysis.timing)
    conv = (tm.readout_ms + tm.exposure_ms) / 2 / 1000.0          # trigger -> exposure centre, s
    lid = next(s for s in v.sensors if s.kind == 'lidar')
    msgs = []                      # (topic, log_ns, kind, payload)
    # cameras: the plan's trigger times, stamped at the trigger
    frames = {}
    for c in v.cameras:
        if c.tag == MISSING:
            continue
        e = plan.cam(c.uid)
        cyc = e['cycle_ms'] / 1000.0
        j, out = 0, []
        while True:
            ts = e['trigger_ms'] / 1000.0 + LATE.get(c.tag, 0.0) / 1000.0 + j * cyc
            if ts > DUR:
                break
            if not (c.tag == DROP_EVERY[0] and j % DROP_EVERY[1] == DROP_EVERY[2]):
                out.append((j, ts))
            j += 1
        frames[c.tag] = out
        for j, ts in out:
            msgs.append((f'/{c.tag}/image', BASE + int(round((ts + 0.015) * 1e9)), 'image', (c, j, ts)))
    # lidar sweeps, stamped at their start, logged 5 ms after their end
    for k in range(int(DUR * 10)):
        msgs.append(('/L1/points', BASE + (k + 1) * 100_000_000 + 5_000_000, 'lidar', k))
    # radars
    for s in v.sensors:
        if s.kind != 'radar':
            continue
        rate = p.sensor_model_for(s).rate_hz
        for k in range(int(DUR * rate)):
            ts = k / rate                                       # logged 4 ms after the detection time
            msgs.append((f'/{s.tag}/detections', BASE + int(round((ts + 0.004) * 1e9)), 'radar', (s, ts)))
    imu = next(s for s in v.sensors if s.kind == 'imu')
    gn = next(s for s in v.sensors if s.kind == 'gnss')
    for k in range(int(DUR * 100)):
        msgs.append((f'/{imu.tag}/imu', BASE + k * 10_000_000 + 1_000_000, 'imu', k / 100))
    for k in range(int(DUR * 10)):
        msgs.append((f'/{gn.tag}/fix', BASE + k * 100_000_000 + 30_000_000, 'fix', k))
    if odom:
        for k in range(int(DUR * 50)):
            msgs.append(('/odom', BASE + k * 20_000_000 + 2_000_000, 'odom', k / 50))
    msgs.append(('/tf_static', BASE + 1_000_000, 'tf', None))
    t0 = min(m[1] for m in msgs)
    t1 = max(m[1] for m in msgs)
    targets = DL._targets(t0, t1, n_frames)
    exact = {}                     # camera tag -> frame indices sampled by the check (rendered at their own time)
    for tag, out in frames.items():
        logs = [BASE + int(round((ts + 0.015) * 1e9)) for _, ts in out]
        exact[tag] = set()
        for tg in targets:
            i = next((i for i, lg in enumerate(logs) if lg >= tg), None)
            if i is not None:
                exact[tag].add(out[i][0])
    renders = {}

    def render(c, j, ts):
        key = (c.tag, j)
        if key not in renders:
            te = ts + conv
            pose = (rot_zyx(c.yaw + YAW_ERR.get(c.tag, 0.0), c.pitch, c.roll), c.pos())
            w, h = (WRONG_SIZE[1], WRONG_SIZE[2]) if c.tag == WRONG_SIZE[0] else (None, None)
            renders[key] = synth.render_world(p, eng, c, shifted(world, float(x_of(te))), w, h, pose=pose,
                                              seed=j, noise=1.5)
        return renders[key]
    still = {}
    noise_rng = np.random.default_rng(11)
    with RB.BagWriter(str(path), compression='none') as bw:
        types = {'image': None, 'lidar': 'sensor_msgs/msg/PointCloud2', 'radar': 'sensor_msgs/msg/PointCloud2',
                 'imu': 'sensor_msgs/msg/Imu', 'fix': 'sensor_msgs/msg/NavSatFix', 'odom': 'nav_msgs/msg/Odometry',
                 'tf': 'tf2_msgs/msg/TFMessage'}
        enc = {'F': 'jpeg', 'ML': 'png', 'MR': 'bgr8', 'R': 'rgb8', 'QL': 'jpeg'}
        added = set()
        for topic, log, kind, pay in sorted(msgs, key=lambda m: m[1]):
            if topic not in added:
                ty = types[kind] or ('sensor_msgs/msg/Image' if enc[topic.split('/')[1]] in ('rgb8', 'bgr8')
                                     else 'sensor_msgs/msg/CompressedImage')
                bw.add_topic(topic, ty)
                added.add(topic)
            if kind == 'image':
                c, j, ts = pay
                stamp = BASE + int(round(ts * 1e9))
                if j in exact[c.tag] or c.tag not in still:
                    img = render(c, j, ts)
                    still.setdefault(c.tag, img)
                else:                      # the frames between: the still with fresh sensor noise (a real camera's
                    img = still[c.tag]     # frames never repeat; a repeat is a frozen camera)
                    img = np.clip(img.astype(np.int16) + noise_rng.integers(-2, 3, img.shape), 0, 255).astype(np.uint8)
                fmt = enc[c.tag]
                if fmt == 'rgb8':
                    msg = RB.image_msg(img, stamp, f'{c.tag}_optical_frame', 'rgb8')
                elif fmt == 'bgr8':
                    msg = RB.image_msg(np.ascontiguousarray(img[..., ::-1]), stamp, f'{c.tag}_optical_frame', 'bgr8')
                else:
                    ext = '.png' if fmt == 'png' else '.jpg'
                    ok, buf = cv2.imencode(ext, img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95] if fmt == 'jpeg' else [])
                    msg = RB.compressed_image(buf.tobytes(), stamp, f'{c.tag}_optical_frame', fmt)
            elif kind == 'lidar':
                arr = lidar_sweep(p, eng, lid, world, pay * 0.1, PHASE_TRUE)
                msg = RB.pointcloud2(arr, BASE + pay * 100_000_000, 'L1')
            elif kind == 'radar':
                s, ts = pay
                ahead = FUTURE[1] / 1000.0 if s.tag == FUTURE[0] else 0.0
                stamp = BASE + int(round((ts + ahead) * 1e9))
                pts = np.zeros(12, [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('v_r', 'f4'), ('rcs', 'f4')])
                pts['x'] = np.linspace(5, 60, 12)
                msg = RB.pointcloud2(pts, stamp, s.tag)
            elif kind == 'imu':
                tt = pay
                Rv = rot_zyx(imu.yaw, imu.pitch, imu.roll)
                f_v = np.array([float(ACC) if tt >= T_STILL else 0.0, 0.0, 9.81])
                f_i = Rv.T @ f_v + np.random.default_rng(int(tt * 1000)).normal(0, 0.01, 3)
                w_i = GYRO_BIAS + np.random.default_rng(int(tt * 1000) + 7).normal(0, 0.0005, 3)
                msg = {'header': RB.header(log - 1_000_000, imu.tag),
                       'angular_velocity': dict(zip('xyz', map(float, w_i))),
                       'linear_acceleration': dict(zip('xyz', map(float, f_i)))}
            elif kind == 'fix':
                status = -1 if pay == 4 else 0 if pay in (2, 7) else 2
                x = float(x_of(pay * 0.1))
                msg = {'header': RB.header(BASE + pay * 100_000_000, gn.tag),
                       'status': {'status': status, 'service': 1}, 'latitude': 48.0 + 0.0,
                       'longitude': 11.0 + x / (6378137.0 * math.cos(math.radians(48.0))) * 180 / math.pi,
                       'altitude': 500.0, 'position_covariance': [0.0004, 0, 0, 0, 0.0004, 0, 0, 0, 0.0025],
                       'position_covariance_type': 2}
            elif kind == 'odom':
                tt = pay
                msg = {'header': RB.header(BASE + int(round(tt * 1e9)), 'odom'), 'child_frame_id': 'base_link',
                       'pose': {'pose': {'position': {'x': float(x_of(tt)), 'y': 0.0, 'z': 0.0},
                                         'orientation': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0}}},
                       'twist': {'twist': {'linear': {'x': float(v_of(tt)), 'y': 0.0, 'z': 0.0},
                                           'angular': {'x': 0.0, 'y': 0.0, 'z': 0.0}}}}
            else:
                msg = {'transforms': [RB.transform_stamped(np.eye(4), BASE, 'base_link', 'L1')]}
            bw.write(topic, log, msg)
    return {'frames': frames, 'targets': targets, 't0': t0, 't1': t1, 'conv_s': conv}


make_drive.__test__ = False


# ============================================================================ the maths on its own
def test_settings_round_trip_clamp_and_junk():
    st = DL.DriveLogSettings.from_dict({'frames_per_camera': '1000', 'tolerance_deg': -3, 'search_deg': 'x',
                                        'drop_factor': float('nan'), 'camera_stamp': 'nonsense', 'sync_tol_ms': 2.5,
                                        'lidar_stamp': 'end', 'path': 7,
                                        'mappings': {f'/b{i}': {'/F/x': 'F', 3: 'G', '/L': 4} for i in range(30)}})
    assert st.frames_per_camera == 200 and isinstance(st.frames_per_camera, int)
    assert st.tolerance_deg == 0.01 and st.search_deg == 2.0 and st.drop_factor == 1.5
    assert st.camera_stamp == 'mid_exposure' and st.lidar_stamp == 'end' and st.sync_tol_ms == 2.5 and st.path == ''
    assert len(st.mappings) == DL.MAPPINGS_KEPT and st.mappings['/b29'] == {'/F/x': 'F'}
    assert DL.DriveLogSettings.from_dict(st.to_dict()) == st
    assert DL.DriveLogSettings.from_dict(None) == DL.DriveLogSettings()
    json.dumps(st.to_dict())


def test_stamp_conventions_move_to_the_exposure_centre():
    assert DL.stamp_offset_ms('trigger', 20.0, 2.0) == pytest.approx(11.0)
    assert DL.stamp_offset_ms('readout_end', 20.0, 2.0) == pytest.approx(-11.0)
    assert DL.stamp_offset_ms('mid_exposure', 20.0, 2.0) == 0.0


def test_header_and_size_read_straight_from_the_cdr_bytes():
    t = BASE + 123_456_789
    img = RB.encode('sensor_msgs/msg/Image', RB.image_msg(np.zeros((7, 13, 3), np.uint8), t, 'F_optical_frame'))
    head = DL.cdr_header(img)
    assert head[0] == t and head[1] == 'F_optical_frame'
    assert DL.cdr_dims(img, head) == (7, 13)
    for name in ('', 'abc', 'abcd', 'abcde'):                       # every string length mod 4
        arr = np.zeros(5, [('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        pc = RB.encode('sensor_msgs/msg/PointCloud2', RB.pointcloud2(arr.reshape(1, 5), t, name))
        h = DL.cdr_header(pc)
        assert h[1] == name and DL.cdr_dims(pc, h) == (1, 5)
    s = RB.encode('std_msgs/msg/String', {'data': 'hello world'})
    h = DL.cdr_header(s)
    assert h is None or abs(h[0] - t) > 10 * 86400 * 10 ** 9      # no plausible header in a String
    assert DL.cdr_header(b'\x00\x01') is None
    import cv2
    for ext in ('.jpg', '.png'):                                     # sizes read in place, without decoding
        b = cv2.imencode(ext, np.zeros((9, 21, 3), np.uint8))[1].tobytes()
        assert DL.jpeg_size(b) == (21, 9)
        raw = RB.encode('sensor_msgs/msg/CompressedImage', RB.compressed_image(b, t, 'F_optical_frame', ext[1:]))
        assert DL.cdr_compressed_dims(memoryview(raw), DL.cdr_header(memoryview(raw))) == (21, 9)


@pytest.mark.parametrize('values, dtype, scale_to_s, absolute', [
    (np.linspace(0, 99_000_000, 50), np.uint32, 1e-9, False),          # Ouster: ns after the stamp
    (np.linspace(0, 0.099, 50), np.float32, 1.0, False),               # Velodyne: s
    (np.linspace(-99_000, 0, 50), np.float64, 1e-6, False),            # µs before the stamp (end stamped)
    (np.linspace(0, 0.099, 50) + 1_700_000_000.25, np.float64, 1.0, True),   # Hesai: absolute s
    (np.linspace(0, 99e6, 50) + 1.7e18 + 2.5e8, np.float64, 1e-9, True)])    # absolute ns
def test_point_time_field_units_are_detected(values, dtype, scale_to_s, absolute):
    stamp = 1_700_000_000_000_000_000
    v = np.asarray(values, dtype)
    rel = DL._rel_times(v, stamp)
    if absolute:
        want = v.astype(float) * scale_to_s - stamp / 1e9
    else:
        want = v.astype(float) * scale_to_s
    assert np.allclose(rel, want, atol=2e-6)


def test_float32_epoch_times_are_refused():
    assert DL._rel_times(np.full(50, 1.7e9, np.float32) + np.arange(50, dtype=np.float32), BASE) is None


def _ring_cloud(phase, sense, rate=10.0, n_az=720, rings=(-10.0, -5.0, 0.0, 5.0), stamp=BASE, time_field=True,
                wall=20.0, box=None):
    """A spinning lidar's sweep in a round room (wall at `wall` m) with an optional box face: (half-width deg, range)
    straight ahead. Columns in firing order from the sync pulse, column-major (every ring of a column, then the next)."""
    az = -180.0 + np.arange(n_az) * 360.0 / n_az
    t = ((sense * (az - phase)) % 360.0) / (360.0 * rate)
    o = np.argsort(t, kind='stable')
    az, t = az[o], t[o]
    E, A = np.meshgrid(np.asarray(rings), az, indexing='xy')
    T_ = np.broadcast_to(t[:, None], E.shape)
    r = np.full(E.shape, wall)
    if box is not None:
        half, rng = box
        r = np.where(np.abs(A) <= half, rng, r)
    e, a = np.radians(E), np.radians(A)
    names = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('ring', 'u2')] + ([('t', 'u4')] if time_field else [])
    arr = np.zeros(E.size, names)
    arr['x'] = (r * np.cos(e) * np.cos(a)).ravel()
    arr['y'] = (r * np.cos(e) * np.sin(a)).ravel()
    arr['z'] = (r * np.sin(e)).ravel()
    arr['ring'] = np.broadcast_to(np.arange(len(rings))[None, :], E.shape).ravel()
    if time_field:
        arr['t'] = np.round(T_.ravel() * 1e9).astype(np.uint32)
    return arr


@pytest.mark.parametrize('phase, sense', [(20.0, -1), (-35.0, 1), (170.0, -1)])
def test_sweep_fit_gives_the_rate_sense_and_phase(phase, sense):
    sw = DL.process_cloud(_ring_cloud(phase, sense), BASE)
    assert sw['has_time'] and sw['fit'] is not None
    assert sw['fit']['b'] == pytest.approx(sense * 3600.0, rel=1e-4)
    ph, src = DL.sweep_phase(sw, 10.0, sense)
    assert src == 'point times' and abs((ph - phase + 180) % 360 - 180) < 0.05
    # a later sweep (stamped 3 periods on, with the same phase) gives the same phase
    sw3 = DL.process_cloud(_ring_cloud(phase, sense), BASE + 300_000_000)
    assert abs((DL.sweep_phase(sw3, 10.0, sense)[0] - phase + 180) % 360 - 180) < 0.05
    # without point times, from the stamp: the first column is the head's direction at a start stamp
    sw0 = DL.process_cloud(_ring_cloud(phase, sense, time_field=False), BASE)
    ph0, src0 = DL.sweep_phase(sw0, 10.0, sense, 'start')
    assert src0 == 'stamps' and abs((ph0 - phase + 180) % 360 - 180) < 0.6
    # … and the last column at an end stamp (one period later)
    swe = DL.process_cloud(_ring_cloud(phase, sense, time_field=False), BASE + 100_000_000)
    phe, _ = DL.sweep_phase(swe, 10.0, sense, 'end')
    assert abs((phe - phase + 180) % 360 - 180) < 0.6


def test_interval_stats_count_drops_gaps_jitter_and_stamps_going_back():
    per = 100_000_000
    t = list(BASE + np.arange(50) * per)
    del t[30:32]                         # 2 missing
    del t[10]                            # 1 missing
    t.insert(5, t[5])                    # a duplicate stamp
    s = DL.interval_stats(np.array(t, np.int64), 0.1, 1.5)
    assert s['drops'] == 3 and s['drop_events'] == 2
    assert s['longest_gap_ms'] == pytest.approx(300.0)
    assert s['nonmonotonic'] == 1
    assert s['rate_hz'] == pytest.approx(10.0)
    assert s['missing_share'] == pytest.approx(0.3 / 5.0, rel=1e-6)
    assert s['jitter_ms'] == pytest.approx(0.0, abs=1e-6)
    assert len(s['gaps']) == 2


def test_depth_edges_sit_on_the_box_sides_and_not_on_smooth_walls():
    arr = _ring_cloud(0.0, -1, n_az=1440, rings=tuple(np.linspace(-2, 2, 9)), box=(5.0, 10.0))
    sw = DL.process_cloud(arr, BASE)
    E = sw['E'].astype(float)
    assert len(E) > 0
    rng = np.linalg.norm(E, axis=1)
    az = np.degrees(np.arctan2(E[:, 1], E[:, 0]))
    assert np.allclose(rng, 10.0, atol=0.02)                  # the nearer side of the jump
    assert np.all(np.abs(np.abs(az) - 5.0) <= 0.13)           # half way between the last box point and the wall
    assert DL.process_cloud(_ring_cloud(0.0, -1, n_az=1440), BASE)['E'].shape[0] == 0


def test_motion_compensation_moves_points_by_the_vehicle_travel_and_turn():
    t = np.linspace(0.0, 1.0, 101)
    yaw = 0.2 * t                          # rad, 0.2 rad/s
    pos = np.stack([5.0 * t, np.zeros_like(t), np.zeros_like(t)], 1)
    m = DL.Motion('odometry', t, pos, np.stack([yaw, 0 * t, 0 * t], 1), speed=(t, np.full_like(t, 5.0)))
    X = np.array([[20.0, 3.0, 1.0]])                        # a world point
    t_i, t_c = 0.40, 0.45

    def in_vehicle(tt):
        R = rot_zyx(math.degrees(0.2 * tt), 0, 0)
        return (X - [5.0 * tt, 0, 0]) @ R
    got = m.apply(in_vehicle(t_i), np.array([t_i]), t_c)
    assert np.allclose(got, in_vehicle(t_c), atol=2e-3)
    imu = DL.Motion('imu', t, omega=np.tile([0.0, 0.0, 0.2], (len(t), 1)))
    X0 = np.array([[20.0, 0.0, 0.0]])
    got = imu.apply(X0, np.array([t_i]), t_c)                # rotation only: the bearing turns by −0.2 × 0.05 rad
    assert math.degrees(math.atan2(got[0, 1], got[0, 0])) == pytest.approx(-math.degrees(0.01), abs=0.03)


def test_head_azimuth_follows_the_phase_lock():
    lock = {'use_rate_hz': 10.0, 'sense': -1, '_base_ns': BASE, '_t': np.array([0.0, 1.0]),
            '_phi': np.array([20.0, 20.0])}
    t = BASE + np.array([0, 25_000_000, 99_000_000, 1_000_000_000 + 50_000_000], np.int64)
    az = (DL.head_azimuth(lock, t) + 180) % 360 - 180
    assert np.allclose(az, [20.0, 20.0 - 90.0, 20.0 + 3.6, -160.0], atol=1e-6)


def test_topics_map_by_tag_frame_and_type():
    p, v = make_project()
    tops = {'/F/image_raw/compressed': ('sensor_msgs/msg/CompressedImage', '', 30),
            '/cam2/image': ('sensor_msgs/msg/Image', 'ML_optical_frame', 30),
            '/F/camera_info': ('sensor_msgs/msg/CameraInfo', 'F_optical_frame', 30),
            '/L1/points': ('sensor_msgs/msg/PointCloud2', 'L1', 10),
            '/RD2/detections': ('sensor_msgs/msg/PointCloud2', 'RD2', 20),
            '/imu/data': ('sensor_msgs/msg/Imu', 'imu_link', 100),
            '/F/image_raw': ('sensor_msgs/msg/Image', '', 3),           # fewer messages than the compressed one
            '/odom': ('nav_msgs/msg/Odometry', 'odom', 50)}
    m = DL.auto_map(v, tops)
    assert m == {'/F/image_raw/compressed': 'F', '/cam2/image': 'ML', '/L1/points': 'L1', '/RD2/detections': 'RD2',
                 '/imu/data': 'IMU1'}
    info = DL.LogInfo('x', 'rosbag', {t: ty for t, (ty, _, _) in tops.items()}, {t: n for t, (_, _, n) in tops.items()},
                      0, 1)
    m2 = DL.mapping_for(v, info, {'/F/image_raw': 'F', '/L1/points': '', '/RD2/detections': 'F',
                                  '/gone': 'R'})
    assert m2['/F/image_raw'] == 'F' and '/F/image_raw/compressed' not in m2      # one topic per sensor
    assert '/L1/points' not in m2                                                  # mapped to no sensor
    assert m2.get('/RD2/detections') == 'RD2'                 # a camera tag does not fit a cloud: edit ignored
    assert DL.compatible_tags(v, 'sensor_msgs/msg/PointCloud2') == ['L1', 'RD1', 'RD2', 'RD3', 'RD4', 'RD5']


# ============================================================================ the synthetic drive
@pytest.fixture(scope='module')
def drive(tmp_path_factory):
    p, v = make_project()
    eng = Engine()
    path = tmp_path_factory.mktemp('drive') / 'rig_drive'
    truth = make_drive(path, p, v, eng)
    st = DL.DriveLogSettings(camera_stamp='trigger', frames_per_camera=N_FRAMES)
    res = DL.run(p, str(path), st, engine=eng)
    return {'p': p, 'v': v, 'eng': eng, 'path': str(path), 'truth': truth, 'res': res, 'st': st}


def test_every_topic_is_mapped_to_its_sensor(drive):
    res = drive['res']
    assert res.mapping['/F/image'] == 'F' and res.mapping['/L1/points'] == 'L1'
    assert res.mapping['/IMU1/imu'] == 'IMU1' and res.mapping['/GN1/fix'] == 'GN1'
    assert '/odom' not in res.mapping and '/tf_static' not in res.mapping
    assert res.stream('F')['status'] == 'ok' and res.stream('F')['rate_hz'] == pytest.approx(30.0, rel=1e-3)
    assert res.stream('L1')['rate_hz'] == pytest.approx(10.0, rel=1e-3) and res.stream('L1')['status'] == 'ok'
    assert res.stream('RD1')['expected_hz'] == 15.0 and res.stream('RD1')['status'] == 'ok'


def test_the_camera_dropping_frames_is_found_with_its_count(drive):
    res, truth = drive['res'], drive['truth']
    kept = [j for j, _ in truth['frames']['MR']]
    lost = sorted(set(range(min(kept), max(kept) + 1)) - set(kept))
    e = res.stream('MR')
    assert e['drops'] == len(lost) >= 3 and e['drop_events'] == len(lost)
    assert e['longest_gap_ms'] == pytest.approx(2000.0 / 30.0, abs=0.05)
    assert e['drop_share'] == pytest.approx(len(lost) / (len(kept) + len(lost)))
    assert e['status'] == 'fault' and len(e['gaps']) == len(lost)
    assert all(res.stream(t)['drops'] == 0 for t in ('F', 'R', 'ML', 'L1'))
    assert res.metrics['worst_drop'] == 'MR'


def test_the_late_camera_and_the_lidar_phase_are_measured(drive):
    res = drive['res']
    L = res.sync['lidar']
    assert L['source'] == 'point times' and L['sweeps'] == 12
    assert L['phase_deg'] == pytest.approx(PHASE_TRUE, abs=0.2) and L['spread_deg'] < 0.1
    assert L['late_ms'] == pytest.approx(PHASE_TRUE / 3600 * 1000, abs=0.05)
    cams = {r['tag']: r for r in res.sync['cameras']}
    assert cams['R']['own_ms'] == pytest.approx(LATE['R'], abs=0.2)
    assert cams['R']['skew_ms'] == pytest.approx(LATE['R'], abs=0.05) and cams['R']['skew_ref'] == 'F'
    for tag in ('F', 'ML', 'MR', 'QL'):
        assert abs(cams[tag]['own_ms']) < 0.2
        assert cams[tag]['deviation_ms'] == pytest.approx(-PHASE_TRUE / 3600 * 1000, abs=0.2)
    assert cams['QR']['topic'] is None
    text = ' '.join(f['message'] for f in res.findings if f['severity'] == 'fault')
    assert 'R fires 7.0 ms late' in text and 'L1 is phase-locked at 20.0°' in text
    assert res.metrics['worst_sync_tag'] == 'R'


def test_the_rotated_camera_is_recovered_and_the_others_pass(drive):
    res = drive['res']
    ml = res.extrinsic('ML')
    d, obs = ml['delta_deg'], ml['observable']
    assert ml['verdict'] == 'rotated' and ml['confidence'] in ('high', 'medium') and ml['apply']
    assert obs[0] and d[0] == pytest.approx(YAW_ERR['ML'], abs=0.15)
    for k in (1, 2):
        assert abs(d[k]) < 0.15 or not obs[k]
    assert ml['score_corrected'] > ml['score_plan']
    for tag in ('F', 'R', 'MR'):
        r = res.extrinsic(tag)
        assert r['verdict'] in ('ok', 'uncertain'), (tag, r['delta_deg'])
        assert all(abs(x) < 0.15 for x, o in zip(r['delta_deg'], r['observable']) if o), (tag, r['delta_deg'])
        assert not r['apply'] or max(abs(x) for x in r['delta_deg']) < 0.15
    assert res.extrinsic('QL')['verdict'] == 'not checked' and 'aspect' in res.extrinsic('QL')['reason']
    assert res.extrinsic('QR')['reason'] == 'no topic'
    assert ml['compensation'].startswith('odometry')
    assert res.metrics['worst_rotation_tag'] == 'ML'


def test_missing_topic_wrong_size_and_a_clock_ahead(drive):
    res = drive['res']
    qr, ql, rd3 = res.stream('QR'), res.stream('QL'), res.stream('RD3')
    assert qr['topic'] is None and qr['status'] == 'fault'
    assert ql['size'] == (WRONG_SIZE[1], WRONG_SIZE[2]) and ql['status'] == 'fault'
    assert any('480 × 360' in m and 'different aspect' in m for m in ql['problems'])
    assert rd3['latency_ms']['median'] == pytest.approx(4.0 - FUTURE[1], abs=0.01) and rd3['status'] == 'fault'
    assert res.stream('GN1')['status'] == 'ok'
    assert res.metrics['missing'] == ['QR']
    faults = [f for f in res.findings if f['severity'] == 'fault']
    assert {t for f in faults for t in f['tags']} == {'MR', 'QL', 'QR', 'RD3', 'L1', 'R', 'ML'}


def test_data_rates_come_from_the_message_bytes(drive):
    res = drive['res']
    rd = RB.BagReader(drive['path'])
    sizes, first, last = {}, {}, {}
    for topic, t, raw in rd.messages(decode=False):
        sizes[topic] = sizes.get(topic, 0) + len(raw)
        first.setdefault(topic, t)
        last[topic] = t
    rd.close()
    rows = {r['tag']: r for r in res.data['rows']}
    f = rows['F']
    assert f['bytes'] == sizes['/F/image']
    span = (last['/F/image'] - first['/F/image']) / 1e9 + 1 / 30
    assert f['mbps'] == pytest.approx(sizes['/F/image'] * 8 / 1e6 / span, rel=1e-9)
    assert f['bpp'] == pytest.approx(f['bytes_per_msg'] * 8 / (480 * 384))
    assert f['compression'] > 3 and f['compressed']                       # JPEG against 16 bit/px raw
    assert rows['R']['bpp'] == pytest.approx(24.0, rel=1e-3)              # raw rgb8 messages: 24 bit/px + header
    assert res.data['totals']['bytes'] == sum(sizes.values())


def test_imu_at_rest_and_gnss_fix_shares(drive):
    res = drive['res']
    imu = next(r for r in res.imu if r['tag'] == 'IMU1')
    assert imu['gravity'] == pytest.approx(9.81, abs=0.01)
    assert np.allclose(imu['gyro_bias_dps'], np.degrees(GYRO_BIAS), atol=0.01)
    assert imu['stationary_s'] >= 0.4 and imu['tilt_deg'] < 0.2
    gn = next(r for r in res.gnss if r['tag'] == 'GN1')
    assert gn['fixes'] == 12 and gn['no_fix_share'] == pytest.approx(1 / 12)
    assert gn['rtk_share'] == pytest.approx(9 / 12) and gn['h_sigma_m'] == pytest.approx(0.02)
    assert gn['status'] == 'warning'


def test_summary_doc_files_and_export(drive, tmp_path):
    res, p = drive['res'], drive['p']
    sm = DL.summary(res)
    json.dumps(sm)
    assert sm['dropped_frames'] == res.metrics['drops'] and sm['worst_rotation_camera'] == 'ML'
    assert 'jpeg' not in json.dumps(sm) and drive['path'] not in json.dumps(sm)
    doc = DL.to_doc(res, p)
    json.dumps(doc)
    assert doc['schema'] == 'vantage.drive_log_check/1' and doc['frame'].startswith('vehicle')
    assert doc['sync']['reference_lidar']['phase_error_deg'] == pytest.approx(20.0, abs=0.2)
    ml = next(c for c in doc['extrinsics']['cameras'] if c['camera'] == 'ML')
    assert ml['correction_ypr_deg'][0] == pytest.approx(0.6, abs=0.15)
    names = [n for n, _ in DL.files(res, p)]
    assert names[:3] == ['drive_log_check.json', 'drive_log_streams.csv', 'drive_log_timeline.png']
    assert {'drive_log_overlay_ML.png', 'drive_log_overlay_F.png'} <= set(names)
    paths = DL.export(res, p, tmp_path / 'out')
    import cv2
    for q in paths:
        assert os.path.getsize(q) > 0
        if q.endswith('.png'):
            assert cv2.imread(q) is not None
    rows = open(tmp_path / 'out' / 'drive_log_streams.csv').read().splitlines()
    assert rows[0].startswith('sensor,kind,topic') and len(rows) == 1 + len(res.streams)


def test_the_corrected_variant_takes_the_confident_rotation(drive):
    res, p, v = drive['res'], drive['p'], drive['v']
    nv = DL.corrected_variant(p, res)
    assert nv is not None and nv.name == f'{v.name} · drive-log corrected' and nv.uid != v.uid
    ml, ml0 = nv.by_tag('ML'), v.by_tag('ML')
    assert ml.yaw == pytest.approx(ml0.yaw + YAW_ERR['ML'], abs=0.15) and ml.uid != ml0.uid
    assert nv.by_tag('F').yaw == v.by_tag('F').yaw and nv.by_tag('F').pitch == v.by_tag('F').pitch
    assert {s.uid for s in nv.sensors}.isdisjoint({s.uid for s in v.sensors})
    assert 'ML yaw' in nv.note
    assert DL.corrected_variant(p, res, tags=[]) is None


def test_a_mapping_edit_checks_again_from_the_same_scan(drive):
    res, p = drive['res'], drive['p']
    st = DL.DriveLogSettings.from_dict(dict(drive['st'].to_dict(),
                                            mappings={DL.norm_path(drive['path']): {'/QL/image': ''}}))
    res2 = DL.run(p, drive['path'], st, engine=drive['eng'], scan=res.scan)
    assert res2.scan is res.scan and res2.stream('QL')['topic'] is None
    assert any(f['section'] == 'mapping' and '/QL/image' in f['message'] for f in res2.findings)
    assert res2.extrinsic('ML')['delta_deg'] == pytest.approx(res.extrinsic('ML')['delta_deg'], abs=1e-9)


def test_without_an_engine_the_own_vehicle_still_hides_what_it_hides(drive):
    """A run given no engine left the own vehicle out of the rotation check: the image edges on it were kept and so
    were the lidar edges it hides from a camera. On the sample's synthetic urban drive the roof lidar sees building
    edges over the right mirror that MR cannot see past its own mirror; scored against the mirror's edge they pulled
    a correctly mounted MR 0.36°/0.38°/0.47° off, 'rotated' (medium). The run now makes an engine when none is
    given: the same rows as with one (here ML used 5525 edge points without the vehicle, 2859 with it)."""
    res = drive['res']
    res2 = DL.run(drive['p'], drive['path'], drive['st'], scan=res.scan)
    assert [r['tag'] for r in res2.extrinsics] == [r['tag'] for r in res.extrinsics]
    for a, b in zip(res.extrinsics, res2.extrinsics):
        assert (b['verdict'], b.get('points'), b.get('frames')) == (a['verdict'], a.get('points'), a.get('frames')), \
            a['tag']
        if a.get('delta_deg') is not None:
            assert b['delta_deg'] == pytest.approx(a['delta_deg'], abs=1e-9), a['tag']
    assert res2.stream('L1')['blocked_share'] == pytest.approx(res.stream('L1')['blocked_share'])


def test_a_mirror_camera_off_its_plan_is_recovered(drive):
    """MR on the right mirror, its plan yawed 0.6° from the pose its images were rendered at: the own vehicle is
    masked at the plan pose (as on a real rig) and the correction back to the true pose is found. In the garage its
    view holds pillars and box sides, upright edges: yaw is pinned down, pitch and roll are said not to be."""
    p, v = drive['p'], drive['v']
    c = v.by_tag('MR')
    yaw0 = c.yaw
    c.yaw = yaw0 + 0.6
    try:
        r = DL.run(p, drive['path'], drive['st'], engine=drive['eng'], scan=drive['res'].scan).extrinsic('MR')
    finally:
        c.yaw = yaw0
    assert r['verdict'] == 'rotated' and r['confidence'] in ('high', 'medium') and r['apply']
    assert r['observable'][0] and r['delta_deg'][0] == pytest.approx(-0.6, abs=0.15)
    assert r['correction_deg'][0] == r['delta_deg'][0]
    assert all(abs(d) < 0.15 or not o for d, o in zip(r['delta_deg'][1:], r['observable'][1:]))
    assert r['corrected_deg'][0] == pytest.approx(yaw0, abs=0.15)


def test_a_variant_without_sensors_and_cancel(drive):
    from vantage.core.project import Variant
    p = drive['p']
    empty = Variant(name='Empty')
    p.variants.append(empty)
    try:
        r = DL.run(p, drive['path'], drive['st'], variant=empty, scan=drive['res'].scan)
        assert r.streams == [] and r.extrinsics == [] and r.mapping == {}
        assert sum(f['section'] == 'mapping' for f in r.findings) == 11          # 5 cameras, the lidar, 5 radars
    finally:
        p.variants.remove(empty)
    assert DL.run(p, drive['path'], drive['st'], cancel=lambda: True) is None
    calls = []

    def late_cancel():                                     # during the camera checks, after the scan
        calls.append(1)
        return len(calls) > 3
    assert DL.run(p, drive['path'], drive['st'], cancel=late_cancel, scan=drive['res'].scan) is None


def test_a_bag_without_odometry_or_point_times(drive, tmp_path):
    """The stamps-only phase (the first column at the start stamp) and no travel compensation are said so."""
    src = RB.BagReader(drive['path'])
    out = tmp_path / 'bare'
    with RB.BagWriter(str(out), compression='none') as bw:
        for t in src.topics:
            if t not in ('/odom',):
                bw.add_topic(t, src.topics[t])
        for topic, t, msg in src.messages():
            if topic == '/odom':
                continue
            if topic == '/L1/points':
                arr = RB.pointcloud2_array(msg)
                keep = [n for n in arr.dtype.names if n != 't']
                arr2 = np.zeros(arr.shape, [(n, arr.dtype[n]) for n in keep])
                for n in keep:
                    arr2[n] = arr[n]
                msg = RB.pointcloud2(arr2, RB.stamp_ns(msg['header']['stamp']), 'L1')
            bw.write(topic, t, msg)
    src.close()
    res = DL.run(drive['p'], str(out), drive['st'], engine=drive['eng'])
    L = res.sync['lidar']
    assert L['source'].startswith('header stamps') and L['phase_deg'] == pytest.approx(PHASE_TRUE, abs=0.6)
    assert res.extrinsic('ML')['compensation'].startswith('IMU')
    assert any('without travel compensation' in f['message'] for f in res.findings)


# ============================================================================ round 8: freezes, the search, the gates
def test_content_digests_ignore_the_header_and_frozen_runs_are_found():
    rng = np.random.default_rng(5)
    a, b = rng.integers(0, 256, 90_000, np.uint8).tobytes(), rng.integers(0, 256, 90_000, np.uint8).tobytes()
    assert DL.content_digest(b'hdr1' + a, 4) == DL.content_digest(b'hdr2' + a, 4) != DL.content_digest(b'hdr1' + b, 4)
    big = rng.integers(0, 256, 3_000_000, np.uint8).tobytes()        # beyond DIGEST_WHOLE: sampled blocks
    big2 = rng.integers(0, 256, 3_000_000, np.uint8).tobytes()
    assert DL.content_digest(big) == DL.content_digest(bytes(big)) != DL.content_digest(big2)
    # stamp order decides; (held, first repeat, last repeat, repeats) in it; an unknown fingerprint (0) never counts
    assert DL.frozen_runs(np.array([5, 1, 2, 3, 4, 6, 7]), [9, 1, 2, 2, 2, 9, 9]) == [(1, 2, 3, 2), (4, 5, 6, 2)]
    assert DL.frozen_runs(np.arange(4), [0, 0, 0, 0]) == [] and DL.frozen_runs(np.arange(1), [3]) == []


@pytest.fixture(scope='module')
def frozen(drive, tmp_path_factory):
    """The drive with F's driver stalled (its 10th to 14th frames are its 9th image again, at its own rate) and R
    repeating one image once; checked."""
    src = RB.BagReader(drive['path'])
    out = tmp_path_factory.mktemp('frozen') / 'frozen'
    seen, held = {'/F/image': [], '/R/image': []}, {}
    with RB.BagWriter(str(out), compression='none') as bw:
        for t in src.topics:
            bw.add_topic(t, src.topics[t])
        for topic, t, msg in src.messages():
            if topic in seen:
                k = len(seen[topic])
                seen[topic].append(RB.stamp_ns(msg['header']['stamp']))
                if (topic, k) in (('/F/image', 9), ('/R/image', 20)):
                    held[topic] = msg['data']
                elif (topic == '/F/image' and 10 <= k < 15) or (topic == '/R/image' and k == 21):
                    msg = dict(msg, data=held[topic])
            bw.write(topic, t, msg)
    src.close()
    return {'path': str(out), 'seen': seen, 'res': DL.run(drive['p'], str(out), drive['st'], engine=drive['eng'])}


def test_a_frozen_camera_is_found_with_its_time_range(drive, frozen):
    """F's driver stalls: its 10th to 14th frames are its 9th image again, at its own rate (the stamps stay regular,
    so the rate and drop checks see nothing); R repeats one image once. F froze (a fault, with the time range), R gets
    a warning, the other cameras nothing."""
    res, seen = frozen['res'], frozen['seen']
    f, r = res.stream('F'), res.stream('R')
    assert f['frozen']['frames'] == 5 and f['frozen']['longest'] == 5 and f['status'] == 'fault'
    t = (np.array(seen['/F/image']) - res.scan.start_ns) / 1e9
    run = f['frozen']['runs'][0]
    assert (run['held_s'], run['from_s'], run['to_s']) == pytest.approx((t[9], t[10], t[14]), abs=1e-6)
    assert f['drops'] == 0 and f['rate_hz'] == pytest.approx(30.0, rel=1e-3)          # the stamps saw nothing
    assert any(m.startswith('F froze: 5 frames') for m in f['problems'])
    assert r['frozen']['frames'] == 1 and r['status'] == 'warning' and any('repeat' in m for m in r['problems'])
    for tag in ('ML', 'MR'):
        assert not res.stream(tag)['frozen']['frames']
    assert res.metrics['frozen_frames'] == 6 and res.metrics['worst_freeze'] == 'F'
    s = DL.summary(res)
    assert s['frozen_frames'] == 6 and next(x for x in s['streams_table'] if x['sensor'] == 'F')['frozen_frames'] == 5
    d = next(x for x in DL.to_doc(res, drive['p'])['streams'] if x['sensor'] == 'F')
    assert d['frozen']['longest_run'] == 5 and len(d['frozen']['runs']) == 1
    assert drive['res'].metrics['frozen_frames'] == 0              # the drive itself: every frame differs


def test_the_search_finds_the_rotation_when_the_coarse_map_is_flat(drive, monkeypatch):
    """On a textured scene nearly every pixel lies within the coarse width of a Canny edge and the coarse map goes
    flat (on a CARLA drive a camera pitched 1° read −0.08° while only it seeded the search). With the coarse map
    flattened outright, the fine map's own grid still finds ML's 0.6°."""
    orig = DL._edge_maps

    def flat(*a, **k):
        m = orig(*a, **k)
        return None if m is None else (np.full_like(m[0], 255), m[1])
    monkeypatch.setattr(DL, '_edge_maps', flat)
    r = DL.run(drive['p'], drive['path'], drive['st'], engine=drive['eng'], scan=drive['res'].scan).extrinsic('ML')
    assert r['verdict'] == 'rotated' and r['confidence'] in ('high', 'medium')
    assert r['delta_deg'][0] == pytest.approx(YAW_ERR['ML'], abs=0.15)


def test_frames_split_between_two_rotations_are_found():
    best, obs, tol = np.array([0.0, 0.85, 0.0]), np.array([True, True, True]), 0.3
    main = [[0.0, 0.85 + d, 0.0] for d in np.linspace(-0.1, 0.1, 12)]
    other = [[0.0, -0.30, 0.0], [0.02, -0.28, 0.0], [0.0, -0.33, 0.01], [0.0, -0.31, 0.0], [0.01, -0.29, 0.0]]
    sp = DL._split_frames(np.array(main + other), best, obs, tol)
    assert sp['n'] == 5 and sp['share'] == pytest.approx(5 / 17) and sp['centre_deg'][1] == pytest.approx(-0.30, abs=0.01)
    assert sp['distance_deg'] == pytest.approx(1.15, abs=0.02)
    assert DL._split_frames(np.array(main), best, obs, tol) is None
    scattered = main + [[0.0, v, 0.0] for v in (-1.2, -0.6, 0.0, 0.3, 1.5)]            # away, but not together
    assert DL._split_frames(np.array(scattered), best, obs, tol) is None
    near = main + [[0.0, 0.85 - 0.45, 0.0]] * 5                                         # within twice the tolerance
    assert DL._split_frames(np.array(near), best, obs, tol) is None
    pitch_free = np.array([True, False, True])                                          # only pinned-down axes count
    assert DL._split_frames(np.array(main + other), best, pitch_free, tol) is None


def test_kerb_height_edges_are_left_out(drive):
    """Lidar depth edges within GROUND_EDGE_M of the ground plane (kerbs, the feet of the garage's pillars and boxes
    here) never reach the score; the rotations are found as before."""
    rows = [drive['res'].extrinsic(t) for t in ('F', 'ML', 'MR', 'R')]
    assert all(r['ground_edges_dropped'] > 0 for r in rows), [r.get('ground_edges_dropped') for r in rows]
    assert DL.GROUND_EDGE_M == pytest.approx(0.4)


# ============================================================================ nuScenes folders
def _write_nuscenes(root, p, v, eng):
    """A small nuScenes folder by hand: CAM_F at 12 Hz with one sample missing, LIDAR_L1 at 10 Hz stamped at the sweep
    end (x y z intensity ring), ego poses standing still."""
    import cv2
    ver = root / 'v1.0-test'
    for d in (ver, root / 'samples' / 'CAM_F', root / 'sweeps' / 'LIDAR_L1'):
        d.mkdir(parents=True)
    cam = v.by_tag('F')
    lid = next(s for s in v.sensors if s.kind == 'lidar')
    world = garage_world(p)
    img = synth.render_world(p, eng, cam, world, noise=1.0)
    cv2.imwrite(str(root / 'samples' / 'CAM_F' / 'f.jpg'), img[..., ::-1])
    sd, ego = [], []
    for k in range(12):
        if k == 6:
            continue
        ts = BASE // 1000 + int(k * 1e6 / 12)
        sd.append({'token': f'c{k}', 'sample_token': '', 'ego_pose_token': f'ec{k}', 'calibrated_sensor_token': 'cF',
                   'timestamp': ts, 'fileformat': 'jpg', 'is_key_frame': k % 6 == 0, 'height': 384, 'width': 480,
                   'filename': 'samples/CAM_F/f.jpg', 'prev': '', 'next': ''})
        ego.append({'token': f'ec{k}', 'timestamp': ts, 'rotation': [1, 0, 0, 0], 'translation': [0, 0, 0]})
    arr = lidar_sweep(p, eng, lid, world, 0.0, PHASE_TRUE)
    pts = np.stack([arr['x'], arr['y'], arr['z'], arr['intensity'], arr['ring'].astype(np.float32)], 1)
    pts.astype(np.float32).tofile(str(root / 'sweeps' / 'LIDAR_L1' / 'l.pcd.bin'))
    for k in range(10):
        ts = BASE // 1000 + (k + 1) * 100_000                        # the sweep end
        sd.append({'token': f'l{k}', 'sample_token': '', 'ego_pose_token': f'el{k}', 'calibrated_sensor_token': 'cL',
                   'timestamp': ts, 'fileformat': 'pcd', 'is_key_frame': k % 5 == 0, 'height': 0, 'width': 0,
                   'filename': 'sweeps/LIDAR_L1/l.pcd.bin', 'prev': '', 'next': ''})
        ego.append({'token': f'el{k}', 'timestamp': ts, 'rotation': [1, 0, 0, 0], 'translation': [0, 0, 0]})
    tables = {'sensor': [{'token': 'sF', 'channel': 'CAM_F', 'modality': 'camera'},
                         {'token': 'sL', 'channel': 'LIDAR_L1', 'modality': 'lidar'}],
              'calibrated_sensor': [{'token': 'cF', 'sensor_token': 'sF', 'translation': [0, 0, 0],
                                     'rotation': [1, 0, 0, 0], 'camera_intrinsic': []},
                                    {'token': 'cL', 'sensor_token': 'sL', 'translation': [0, 0, 0],
                                     'rotation': [1, 0, 0, 0], 'camera_intrinsic': []}],
              'sample_data': sd, 'ego_pose': ego}
    for name, rows in tables.items():
        (ver / f'{name}.json').write_text(json.dumps(rows))
    return root


def test_a_nuscenes_folder_is_read(tmp_path):
    p, v = make_project()
    eng = Engine()
    root = _write_nuscenes(tmp_path / 'nus', p, v, eng)
    assert DL.is_nuscenes(root) and DL.is_nuscenes(root / 'v1.0-test') and not DL.is_nuscenes(tmp_path)
    info = DL.open_log(root)
    assert info.source == 'nuscenes' and info.counts == {'CAM_F': 11, 'LIDAR_L1': 10}
    res = DL.run(p, str(root), DL.DriveLogSettings(frames_per_camera=3), engine=eng)
    assert res.source == 'nuscenes' and res.mapping == {'CAM_F': 'F', 'LIDAR_L1': 'L1'}
    f = res.stream('F')
    assert f['rate_hz'] == pytest.approx(12.0, rel=1e-3) and f['drops'] == 1 and f['size'] == (480, 384)
    assert any('runs at 12.00 Hz, the plan says 30 Hz' in m for m in f['problems'])
    assert 'latency_ms' not in f                                      # nuScenes has no separate log time
    lid = res.stream('L1')
    assert lid['points'] == pytest.approx(len(np.fromfile(root / 'sweeps' / 'LIDAR_L1' / 'l.pcd.bin',
                                                          np.float32)) / 5)
    L = res.sync['lidar']
    assert L['source'].startswith('header stamps (end') and L['phase_deg'] == pytest.approx(PHASE_TRUE, abs=0.6)
    assert res.extrinsic('F')['frames'] >= 1 and res.extrinsic('F')['compensation'].startswith('odometry')
    assert any('nuScenes' in n for n in res.notes)


def test_a_dataset_written_by_core_dataset(tmp_path):
    """End to end with the synthetic dataset module (nuScenes folder and ROS 2 bag of the same drive)."""
    DS = pytest.importorskip('vantage.core.dataset', reason='core/dataset.py is not there')
    if not hasattr(DS, 'generate') or not hasattr(DS, 'DatasetSettings'):
        pytest.skip('core/dataset.py has no generate() yet')
    p = samples.tigor_sample_project()
    p.active_uid = next(v for v in p.variants if v.sensors).uid
    eng = Engine()
    st = DS.DatasetSettings.from_dict({'duration_s': 1.0, 'image_scale': 0.25, 'keyframe_hz': 2})
    try:
        out = DS.generate(p, eng, st, str(tmp_path / 'ds'))
    except TypeError as e:
        pytest.skip(f'core/dataset.generate has another signature: {e}')
    nus = next((os.path.join(r, d) for r, ds, _ in os.walk(tmp_path / 'ds') for d in ds
                if os.path.isfile(os.path.join(r, d, 'sample_data.json'))), None)
    bag = next((os.path.join(r, d) for r, ds, _ in os.walk(tmp_path / 'ds') for d in ds
                if os.path.isfile(os.path.join(r, d, 'metadata.yaml'))), None)
    if nus is None and bag is None:
        pytest.skip(f'the dataset module wrote neither a nuScenes folder nor a bag ({out})')
    for path in (x for x in (nus, bag) if x):
        res = DL.run(p, path, DL.DriveLogSettings(frames_per_camera=3), engine=eng)
        cams = [e for e in res.streams if e['kind'] == 'camera']
        assert cams and all(e['topic'] for e in cams), path
        assert all(e['drops'] == 0 for e in cams), path
        assert res.stream('L1')['topic'] is not None
        assert res.sync['lidar'] is not None and abs(res.sync['lidar']['phase_error_deg']) < 1.0, path
        for r in res.sync['cameras']:                      # triggered as the plan says, stamped mid-exposure
            assert r.get('own_ms') is not None and abs(r['own_ms']) < 0.5, (path, r['tag'], r.get('own_ms'))
        for r in res.extrinsics:                           # rendered with the plan's extrinsics
            assert r['verdict'] != 'rotated', (path, r['tag'], r['delta_deg'])


# ============================================================================ the page
from test_gui import app, win, pump, settle, _wait   # noqa: E402,F401 (fixtures)


def _open(win, name, cls):
    if name not in win._page_widgets:
        pg = cls(win.ctx)
        win.pages.addWidget(pg)
        win._page_widgets[name] = pg
    win.set_page(name)
    return win._page_widgets[name]


def test_page_opens_runs_shows_exports_applies_and_undoes(app, win, drive, tmp_path):
    from vantage.ui.workspaces.drivelog_ws import DriveLogPage
    p, _ = make_project()
    win.store.set_project(p)
    pump(app, 0.3)
    pg = _open(win, 'drivelog', DriveLogPage)
    pump(app, 0.2)
    assert not pg.b_run.isEnabled() and pg.topics.rowCount() == 0
    assert pg.open_path(drive['path'])
    pump(app, 0.2)
    assert pg.b_run.isEnabled() and pg.topics.rowCount() == 15
    row = next(r for r in range(pg.topics.rowCount()) if pg.topics.item(r, 0).text() == '/F/image')
    assert pg.topics.cellWidget(row, 3).currentData() == 'F'
    # a setting through its field, undoable
    pg.fields['tolerance_deg'].set_value(0.5)
    pg.fields['tolerance_deg'].committed.emit(0.5)
    assert pg.settings.tolerance_deg == 0.5
    win.store.undo.undo()
    pump(app, 0.1)
    assert pg.settings.tolerance_deg == 0.3 and pg.settings.path == DL.norm_path(drive['path'])
    pg._set('frames', frames_per_camera=N_FRAMES, camera_stamp='trigger')
    assert pg.run()
    t0 = time.time()
    while (pg.busy or pg.result is None) and time.time() - t0 < 240:
        pump(app, 0.1)
    res = pg.result
    assert res is not None and not pg.busy and 'Failed' not in pg.status.text()
    assert pg.streams.rowCount() == len(res.streams) and pg.sync.rowCount() == 6 and pg.ext.rowCount() == 6
    assert pg.timeline.rows and pg.tiles['drops'].value == str(res.metrics['drops'])
    assert len(pg.findings) >= len(res.findings)
    # the overlay: ML, plan then corrected
    pg.tabs.setCurrentIndex(2)
    i = pg.cam_pick.findText('ML')
    pg.cam_pick.setCurrentIndex(i)
    pump(app, 0.1)
    assert pg.overlay.img is not None and len(pg.overlay.pts[0]) > 100
    plan_caption = pg.overlay.caption
    pg.mode.buttons['corrected'].click()
    pump(app, 0.1)
    assert pg.overlay.caption != plan_caption and 'yaw +0.' in pg.overlay.caption
    pg.slider.setValue(pg.slider.maximum())
    pump(app, 0.1)
    assert pg.frame_label.text().startswith(f'{pg.slider.maximum() + 1}/')
    # export
    paths = pg.export(str(tmp_path / 'exp'))
    assert any(q.endswith('drive_log_check.json') for q in paths) and all(os.path.exists(q) for q in paths)
    # a mapping edit through the drop-down: QL to no sensor, undoable
    row = next(r for r in range(pg.topics.rowCount()) if pg.topics.item(r, 0).text() == '/QL/image')
    cb = pg.topics.cellWidget(row, 3)
    cb.setCurrentIndex(0)
    cb.activated.emit(0)
    pump(app, 0.1)
    assert pg.settings.overrides(drive['path']) == {'/QL/image': ''} and pg.stale
    win.store.undo.undo()
    pump(app, 0.1)
    assert pg.settings.overrides(drive['path']) == {} and not pg.stale
    # apply the confident corrections as a new variant, one undo step
    n = len(win.store.project.variants)
    assert pg.b_apply.isEnabled() and '(1)' in pg.b_apply.text()
    pg.b_apply.click()
    pump(app, 0.2)
    pj = win.store.project
    assert len(pj.variants) == n + 1 and pj.variant.name.endswith('· drive-log corrected')
    assert pj.variant.by_tag('ML').yaw == pytest.approx(drive['v'].by_tag('ML').yaw + 0.6, abs=0.15)
    win.store.undo.undo()
    pump(app, 0.2)
    assert len(win.store.project.variants) == n
    # a new project drops the result
    win.store.set_project(samples.tigor_sample_project())
    pump(app, 0.2)
    assert pg.result is None and pg.streams.rowCount() == 0


def test_page_shows_a_frozen_camera_and_its_threshold_is_undoable(app, win, frozen):
    """The frozen drive on the page: F's run on the timeline (amber, from the held frame to its last repeat, with its
    times in the row's tip), its fault among the findings, the frozen frames on the drops tile; the freeze threshold
    is a field (undoable), and raised above F's 5 repeats it turns F's fault into a warning."""
    from PyQt6.QtGui import QColor, QImage
    from vantage.ui.theme import STATUS
    from vantage.ui.workspaces.drivelog_ws import DriveLogPage
    p, _ = make_project()
    win.store.set_project(p)
    pump(app, 0.3)
    pg = _open(win, 'drivelog', DriveLogPage)
    pump(app, 0.2)
    assert pg.open_path(frozen['path'])
    pg._set('frames', frames_per_camera=N_FRAMES, camera_stamp='trigger')
    f = pg.fields['freeze_frames']
    assert f.value() == 3
    f.set_value(6)
    f.committed.emit(6.0)
    assert pg.settings.freeze_frames == 6
    win.store.undo.undo()
    pump(app, 0.1)
    assert pg.settings.freeze_frames == 3 and f.value() == 3

    def check():
        pg._scan = frozen['res'].scan          # the log is read already: only the checks run
        assert pg.run()
        t0 = time.time()
        while (pg.busy or pg.result is None or pg.stale) and time.time() - t0 < 240:
            pump(app, 0.1)
        assert not pg.busy and not pg.stale
        return pg.result
    res = check()
    run = frozen['res'].stream('F')['frozen']['runs'][0]
    row = next(r for r in pg.timeline.rows if r['tag'] == 'F')
    assert [(x['held_s'], x['to_s'], x['repeats']) for x in row['frozen']] == [(run['held_s'], run['to_s'], 5)]
    assert '5 frozen frames' in row['tip']
    assert not next(r for r in pg.timeline.rows if r['tag'] == 'ML')['frozen']
    texts = [w.text.text() for w in pg.findings]
    assert any(t.startswith('F froze: 5 frames') for t in texts) and any(t.startswith('R: 1 frame') for t in texts)
    assert '6 frozen (F)' in pg.tiles['drops'].note
    # the amber run is painted in F's row, nowhere in ML's
    pg.timeline.resize(900, pg.timeline.minimumHeight())
    img = QImage(pg.timeline.size(), QImage.Format.Format_RGB32)
    pg.timeline.render(img)
    amber = QColor(STATUS['warning']).rgb()

    def amber_in(tag):
        i = next(k for k, r in enumerate(pg.timeline.rows) if r['tag'] == tag)
        y = pg.timeline.TOP + i * pg.timeline.ROW + pg.timeline.ROW // 2
        return sum(img.pixel(x, y) == amber for x in range(pg.timeline.LEFT, img.width() - 12))
    xa, xb = (pg.timeline._x(run['held_s']), pg.timeline._x(run['to_s']))
    assert amber_in('F') >= max(int(xb - xa) - 2, 2) and amber_in('ML') == 0
    # five repeats under a threshold of six: a warning
    pg._set('freeze 6', freeze_frames=6)
    assert pg.stale
    res = check()
    assert res.stream('F')['status'] == 'warning' and res.settings.freeze_frames == 6
