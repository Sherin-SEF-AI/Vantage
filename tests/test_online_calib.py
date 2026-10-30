"""Online self-calibration: time-varying extrinsics over a whole drive, with knocks told from flex.

Test drives are written by the sample dataset generator (core.dataset, nuScenes, a quarter of the sensor's size) from
the sample data rig with three cameras (F, MR, QR) and the roof lidar: a drive is two or more generated segments joined
one after the other (join_nuscenes below: timestamps and ego poses continued, tokens prefixed, every segment's
sample_data pointed at the first segment's calibration, which is the plan). A knock: the second segment rendered with QR
yawed 0.8°. Flex: segments driven at different speeds with F pitched by 0.04° per m/s of their speed. Each is found with
its time, size and the clean segments' poses.

The maths on its own, against independent references: the Euler Jacobian against its closed form, the chain smoother
against the dense least-squares solution of the same problem, the model selection on series of known steps, slopes and
pure noise, the excursion logic, the step-time refinement, the ground plane's RANSAC, the ground-shift Jacobian against
the geometry of a camera looking straight down, and the ground factor against two cameras rendered (core.synth) with a
known relative rotation over a marked floor. Then the exports (nuScenes rows per segment, the input untouched; the /tf
sidecar; JSON, CSV, PNG), determinism, the batch entry, cancel, and the page."""
import hashlib
import json
import math
import os
import shutil
import time
from types import SimpleNamespace

import numpy as np
import pytest

from vantage.core import dataset as DS
from vantage.core import drivelog as DL
from vantage.core import online_calib as OC
from vantage.core import rosbag as RB
from vantage.core import samples, synth
from vantage.core.analysis import Engine
from vantage.core.mathutil import BODY_TO_OPTICAL, rot_y, rot_z, rot_zyx
from vantage.core.project import AnalysisSettings

CAMS = ('F', 'MR', 'QR')
KNOCK = ('QR', (0.8, 0.0, 0.0))                 # yaw, pitch, roll (deg) of the second segment
FLEX_K = 0.04                                   # F pitch, deg per m/s of the segment's speed
FLEX_SPEEDS = (10.0, 50.0, 20.0, 45.0)          # km/h
SCALE = 0.25


def make_project(err=None, cams=CAMS):
    """The sample data rig with only cams visible (radars hidden); err {tag: (dyaw, dpitch, droll)} turns cameras."""
    p = samples.tigor_sample_project()
    v = next(v for v in p.variants if v.sensors)
    p.active_uid = v.uid
    for c in v.cameras:
        c.visible = c.tag in cams
        d = (err or {}).get(c.tag)
        if d:
            c.yaw, c.pitch, c.roll = c.yaw + d[0], c.pitch + d[1], c.roll + d[2]
    for s in v.sensors:
        if s.kind == 'radar':
            s.visible = False
    return p, v


def segment(out, eng, err=None, dur=4.0, speed=30.0):
    p, _ = make_project(err)
    st = DS.DatasetSettings.from_dict({'duration_s': dur, 'image_scale': SCALE, 'nuscenes': True, 'rosbag': False,
                                       'keyframe_hz': 2, 'speed_kmh': speed, 'seed': 1})
    return DS.generate(p, eng, st, out).nuscenes_dir


SHARED = ('sensor', 'calibrated_sensor', 'category', 'attribute', 'visibility', 'log', 'map')
TIMED = ('sample', 'sample_data', 'ego_pose')


def join_nuscenes(parts, out):
    """One nuScenes folder (out) of the drives parts (nuScenes dataroots) one after the other: every later part's
    timestamps moved to start one 50 ms lidar period after the previous part's end, its ego poses (and annotations)
    moved to continue where the previous part stopped, its tokens prefixed, its files linked under a prefixed name, its
    sample_data pointed at the first part's calibrated_sensor rows (the plan) channel by channel. Returns the start time
    (s after the first timestamp) of every part."""
    ver = DS.VERSION
    os.makedirs(os.path.join(out, ver), exist_ok=True)
    joined, starts = {}, []
    shift_us, offset, t_first = 0, np.zeros(3), None
    for pi, root in enumerate(parts):
        tabs = {}
        for n in os.listdir(os.path.join(root, ver)):
            if n.endswith('.json'):
                with open(os.path.join(root, ver, n)) as fh:
                    tabs[n[:-5]] = json.load(fh)
        if pi == 0:
            for n in SHARED:
                joined[n] = tabs.get(n, [])
            shared = {r['token'] for n in SHARED for r in tabs.get(n, [])}
            chan0 = {s['token']: s['channel'] for s in tabs['sensor']}
            cs0 = {chan0[c['sensor_token']]: c['token'] for c in tabs['calibrated_sensor']}
        chan = {s['token']: s['channel'] for s in tabs['sensor']}
        cs_ch = {c['token']: chan[c['sensor_token']] for c in tabs['calibrated_sensor']}
        t0 = min(int(r['timestamp']) for r in tabs['sample_data'])
        t1 = max(int(r['timestamp']) for r in tabs['sample_data'])
        t_first = t0 if t_first is None else t_first
        dt = (shift_us - (t0 - t_first)) if pi else 0
        starts.append((t0 + dt - t_first) / 1e6)
        eps = sorted(tabs['ego_pose'], key=lambda e: e['timestamp'])
        dp = offset - np.asarray(eps[0]['translation'], float) if pi else np.zeros(3)
        pre = f'p{pi}' if pi else ''
        for n, rows in tabs.items():
            if n in SHARED:
                continue
            for r in rows:
                r = dict(r)
                for k, v in list(r.items()):
                    if (k == 'token' or k.endswith('_token') or k in ('prev', 'next')) and isinstance(v, str):
                        if k == 'calibrated_sensor_token':
                            r[k] = cs0[cs_ch[v]]
                        elif k == 'log_token':
                            r[k] = joined['log'][0]['token']
                        elif v and v not in shared and pre:
                            r[k] = pre + v
                if n in TIMED:
                    r['timestamp'] = int(r['timestamp']) + dt
                if n in ('ego_pose', 'sample_annotation'):
                    r['translation'] = [float(a + b) for a, b in zip(r['translation'], dp)]
                if n == 'sample_data' and pre:
                    d, b = os.path.split(r['filename'])
                    r['filename'] = f'{d}/{pre}_{b}'
                joined.setdefault(n, []).append(r)
        for r in tabs['sample_data']:
            d, b = os.path.split(r['filename'])
            dst = os.path.join(out, d, f'{pre}_{b}' if pre else b)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if not os.path.exists(dst):
                try:
                    os.link(os.path.join(root, r['filename']), dst)
                except OSError:
                    shutil.copy2(os.path.join(root, r['filename']), dst)
        shift_us = (t1 + dt - t_first) + 50_000
        offset = np.asarray(eps[-1]['translation'], float) + dp
    for n, rows in joined.items():
        with open(os.path.join(out, ver, f'{n}.json'), 'w') as fh:
            json.dump(rows, fh)
    return starts


join_nuscenes.__test__ = False


def _tables_digest(ver):
    h = hashlib.sha256()
    for n in sorted(os.listdir(ver)):
        with open(os.path.join(ver, n), 'rb') as fh:
            h.update(n.encode() + fh.read())
    return h.hexdigest()


# ============================================================================ the maths on its own
def test_settings_round_trip_clamp_and_junk():
    st = OC.OnlineCalibSettings.from_dict({'window_s': 1e9, 'stride_s': float('nan'), 'frames_per_window': 2.6,
                                           'edges': 'no', 'fit_translation': 1, 'tolerance_deg': -3, 'path': 5,
                                           'huber_k': 'x', 'seed': True, 'unknown': 1})
    assert st.window_s == OC.LIMITS['window_s'][1] and st.stride_s == OC.OnlineCalibSettings().stride_s
    assert st.frames_per_window == 3 and st.edges is False and st.fit_translation is True
    assert st.tolerance_deg == OC.LIMITS['tolerance_deg'][0] and st.path == ''
    assert st.huber_k == OC.OnlineCalibSettings().huber_k and st.seed == 0
    assert OC.OnlineCalibSettings.from_dict(st.to_dict()) == st
    assert OC.OnlineCalibSettings.from_dict(None) == OC.OnlineCalibSettings()
    a = AnalysisSettings.from_dict({'cell_m': 0.05})            # an old project file without the slot
    assert OC.OnlineCalibSettings.from_dict(a.online_calib) == OC.OnlineCalibSettings()


def test_window_spans_cover_the_drive():
    assert OC.window_spans(5.0, 10.0, 5.0) == [(0.0, 5.0)]
    sp = OC.window_spans(30.0, 10.0, 10.0)
    assert sp == [(0.0, 10.0), (10.0, 20.0), (20.0, 30.0)]
    sp = OC.window_spans(31.0, 10.0, 10.0)                      # a second left over: the last window stretched
    assert sp[-1] == (20.0, 31.0) and len(sp) == 3
    sp = OC.window_spans(36.0, 10.0, 10.0)                      # six left over: one more window to the end
    assert sp[-1] == (26.0, 36.0) and len(sp) == 4
    sp = OC.window_spans(30.0, 10.0, 5.0)                       # overlapping
    assert sp[0] == (0.0, 10.0) and sp[1] == (5.0, 15.0) and sp[-1] == (20.0, 30.0)


def test_euler_jacobian_matches_its_closed_form():
    """ω = δyaw·z + δpitch·Rz(yaw)·y + δroll·Rz(yaw)·Ry(pitch)·x for R = Rz Ry Rx (pitch positive down)."""
    for ypr in ((0.0, 0.0, 0.0), (-90.0, 50.0, 0.0), (65.0, 45.0, 3.0), (180.0, 40.0, -2.0)):
        J = OC.jac_rot(ypr)
        y, p, _ = np.radians(ypr)
        ref = np.stack([[0.0, 0.0, 1.0], rot_z(y) @ [0.0, 1.0, 0.0], rot_z(y) @ rot_y(p) @ [1.0, 0.0, 0.0]], 1)
        assert np.allclose(J, ref, atol=1e-6)


def test_chain_smoother_equals_the_dense_least_squares():
    """The Kalman / RTS chain gives the same estimates and marginal covariances as the dense normal equations of the
    whole sliding-window problem (measurements, random walk, prior, a released walk at a step, known offsets)."""
    rng = np.random.default_rng(3)
    K, n = 7, 3
    t = np.cumsum(rng.uniform(0.5, 2.0, K))
    prior = np.array([2.0, 2.0, 0.1])
    q = np.array([0.05, 0.02, 0.003])
    infos, offs = [], rng.normal(0, 0.1, (K, n))
    for k in range(K):
        A = rng.normal(0, 1, (4, n)) * (0.0 if k == 3 else 1.0)       # window 3 measures nothing
        b = rng.normal(0, 1, 4)
        infos.append((A.T @ A, A.T @ b))
    breaks = {4: [1]}
    scale = 0.5
    m, P = OC.smooth_chain(infos, t, prior, q, breaks, offs, scale)
    N = K * n
    Lam = np.zeros((N, N))
    eta = np.zeros(N)
    Lam[:n, :n] += np.diag(1 / prior ** 2)
    big = (10 * prior) ** 2
    for k in range(K):
        L, e = infos[k]
        sl = slice(k * n, (k + 1) * n)
        Lam[sl, sl] += scale * L
        eta[sl] += scale * (e - L @ offs[k])
        if k:
            Qi = np.diag(1 / np.where(np.isin(np.arange(n), breaks.get(k, [])), big, q ** 2 * (t[k] - t[k - 1])))
            pl = slice((k - 1) * n, k * n)
            Lam[pl, pl] += Qi
            Lam[sl, sl] += Qi
            Lam[pl, sl] -= Qi
            Lam[sl, pl] -= Qi
    x = np.linalg.solve(Lam, eta).reshape(K, n)
    Cv = np.linalg.inv(Lam)
    assert np.allclose(m, x, atol=1e-9)
    for k in range(K):
        assert np.allclose(P[k], Cv[k * n:(k + 1) * n, k * n:(k + 1) * n], atol=1e-9)


def test_window_solve_downweights_an_outlying_factor():
    lay = OC.Layout([SimpleNamespace(tag='A', uid='a')], None, False)
    rows = OC.Rows(lay.n)
    H = np.eye(3)
    for z in ([0.1, 0.0, 0.0], [0.12, 0.0, 0.0], [0.08, 0.0, 0.0], [2.0, 0.0, 0.0]):
        rows.add(H, np.array(z), np.full(3, 0.02))
    A, b, g = rows.arrays()
    x, P, w = OC.solve_window(A, b, g, rows.ng, lay.prior, 1.5)
    assert x[0] == pytest.approx(0.1, abs=0.03) and w[3] < 0.1 and w[:3].min() > 0.5
    rows.add(np.eye(3), np.zeros(3), np.array([np.inf, np.nan, 0.0]))     # nothing usable: no factor
    assert rows.ng == 4


def test_model_selection_finds_steps_and_slopes_and_no_ghosts_in_noise():
    rng = np.random.default_rng(11)
    n = 40
    t = np.arange(n) + 0.5
    s = np.full(n, 0.03)
    y = 0.02 + np.where(t > 23, 0.5, 0.0) + rng.normal(0, 0.03, n)
    m = OC.select_model(t, y, s, {}, 1.0, 6.0, OC.MIN_SEG, 6, 1.5, 0.1, 0.1)
    assert len(m['steps']) == 1 and m['steps'][0]['j'] == 23 and m['steps'][0]['size'] == pytest.approx(0.5, abs=0.04)
    assert m['level'] == pytest.approx(0.02, abs=0.02) and not m['flex']
    speed = rng.uniform(2, 20, n)
    y = 0.1 + 0.02 * speed + rng.normal(0, 0.03, n)
    m = OC.select_model(t, y, s, {'speed': speed - speed.min(), 'accel': rng.normal(0, 1, n)}, 1.0, 6.0, OC.MIN_SEG,
                        6, 1.5, 0.1, 0.1)
    assert [f['cov'] for f in m['flex']] == ['speed'] and not m['steps']
    assert m['flex'][0]['slope'] == pytest.approx(0.02, abs=3 * m['flex'][0]['sigma'])
    ghosts = 0
    for seed in range(30):                                          # pure noise: a ghost now and then at most
        r = np.random.default_rng(100 + seed)
        y = r.normal(0, 0.03, n)
        m = OC.select_model(t, y, s * r.uniform(0.7, 1.5), {'speed': r.uniform(0, 15, n)}, 1.0, 6.0, OC.MIN_SEG, 6, 1.5,
                            0.1, 0.1)
        ghosts += bool(m['steps'] or m['flex'])
    assert ghosts <= 1
    # a slope two outlying windows make up does not survive leaving one out
    speed = np.r_[np.full(n - 2, 10.0) + rng.normal(0, 1, n - 2), [2.0, 2.5]]
    y = np.r_[rng.normal(0, 0.03, n - 2), [-0.4, -0.35]]
    m = OC.select_model(t, y, s, {'speed': speed - speed.min()}, 1.0, 6.0, OC.MIN_SEG, 6, 1.5, 0.1, 0.1)
    assert not m['flex']


def _events_of(y, sigma=0.03, frames=None):
    t = np.arange(len(y)) + 0.5
    st = OC.OnlineCalibSettings(window_s=1.0, stride_s=1.0)
    m = OC.select_model(t, y, np.full(len(y), sigma), {}, 1.0, 6.0, OC.MIN_SEG, 6, 1.5, 0.1, 0.1)
    m['idx'] = np.arange(len(y))
    cam = SimpleNamespace(tag='QR', uid='q')
    return OC._events({'QR': {'yaw': m}}, None, {'QR': frames or {}}, t, st, [], {}, [cam], None)


def test_a_step_that_returns_is_an_excursion_and_one_that_stays_a_knock():
    rng = np.random.default_rng(5)
    y = np.r_[np.zeros(8), np.full(6, 0.6), np.zeros(8)] + rng.normal(0, 0.03, 22)
    ev = _events_of(y)
    assert [e['kind'] for e in ev] == ['excursion']
    assert ev[0]['t_s'] == pytest.approx(8.0, abs=0.01) and ev[0]['t_end_s'] == pytest.approx(14.0, abs=0.01)
    assert 'returned at 00:14.0' in ev[0]['message']
    y = np.r_[np.zeros(8), np.full(6, 0.6), np.full(8, 0.3)] + rng.normal(0, 0.03, 22)
    ev = _events_of(y)
    assert sorted(e['kind'] for e in ev) == ['knock', 'knock']
    assert all(e['beyond_tolerance'] == (abs(e['size']['yaw']) > 0.3) for e in ev)


def test_the_step_time_comes_from_the_frames_around_it():
    rng = np.random.default_rng(2)
    ft = np.arange(0.125, 8.0, 0.25)                                # four frames a second
    fy = np.where(ft > 4.6, 0.8, 0.0) + rng.normal(0, 0.05, len(ft))
    tau, res = OC.refine_step(ft, fy, 4.5, 0.0, 0.8, 2.0, 7.0)
    assert tau == pytest.approx(4.625, abs=0.13) and res == pytest.approx(0.25)
    ev = _events_of(np.where(np.arange(8) + 0.5 > 4.6, 0.8, 0.0) + rng.normal(0, 0.03, 8),
                    frames={round(a, 4): [b, 0.0, 0.0] for a, b in zip(ft, fy)})
    assert ev[0]['t_s'] == pytest.approx(4.625, abs=0.13) and ev[0]['resolution_s'] == pytest.approx(0.25)


def test_the_ground_plane_is_found_through_clutter():
    rng = np.random.default_rng(1)
    xy = rng.uniform(-20, 20, (6000, 2))
    xy = xy[np.hypot(*xy.T) > 2.5]
    n = np.array([math.sin(math.radians(1.0)), 0.0, math.cos(math.radians(1.0))])
    z = -(0.03 + n[0] * xy[:, 0]) / n[2] + rng.normal(0, 0.01, len(xy))
    P = np.c_[xy, z]
    clutter = np.c_[rng.uniform(-15, 15, (1500, 2)), rng.uniform(0.0, 0.4, 1500)]
    nrm, d, k = OC.fit_ground_plane(np.vstack([P, clutter]), np.random.default_rng(0))
    assert math.degrees(math.acos(abs(nrm @ n))) < 0.05 and -d / nrm[2] == pytest.approx(-0.03, abs=0.005)
    assert OC.fit_ground_plane(P[:20], np.random.default_rng(0)) is None


def test_ground_shift_of_a_camera_looking_straight_down():
    """From height h looking straight down: a yaw turns the ground about the nadir (∂g/∂yaw = (−dy, dx)·π/180), a pitch
    moves the nadir point back by h·π/180 per degree (positive pitch tips the view backwards past the vertical)."""
    p, v = make_project()
    cam = v.by_tag('F')
    cam.x, cam.y, cam.z, cam.yaw, cam.pitch, cam.roll = 1.0, 0.5, 2.0, 0.0, 90.0, 0.0
    X = OC._CamGeo(p, cam, (384, 480))
    plane = (np.array([0.0, 0.0, 1.0]), 0.0)
    J = X.ground_shift(np.array([[1.0, 0.5, 0.0], [1.6, 0.1, 0.0]]), plane)
    k = math.pi / 180
    assert J[0, :, 1] == pytest.approx([-2.0 * k, 0.0], abs=1e-6)
    assert J[1, :, 0] == pytest.approx([0.4 * k, 0.6 * k], abs=1e-6)


def test_the_ground_factor_recovers_a_known_relative_rotation():
    """MR and QR rendered over a floor marked with short random strokes, QR turned by (0.6, 0.4, −0.3)° against its
    plan, MR as planned (and held there): the patches' displacements solved for QR give the turn within 1.5 σ on every
    axis, and the factor's information pins pitch best (the ground below a side camera)."""
    import cv2
    eng = Engine()
    p, v = make_project()
    eng.vehicle_geom(p)
    rng = np.random.default_rng(5)
    marks = []
    for _ in range(900):
        x, y = rng.uniform(-6, 9), rng.uniform(-9, 9)
        a, L = rng.uniform(0, math.pi), rng.uniform(0.15, 0.5)
        marks.append((x, y, x + L * math.cos(a), y + L * math.sin(a), rng.uniform(0.03, 0.08), 0.0))
    w = synth.World(centre=(1.3, 0.0), markings=marks, boxes=[], cylinders=[], wall_r=40.0, wall_h=3.0)
    MR, QR = v.by_tag('MR'), v.by_tag('QR')
    true = np.array([0.6, 0.4, -0.3])
    iA = synth.render_world(p, eng, MR, w, 480, 384, seed=1, noise=1.5)
    iB = synth.render_world(p, eng, QR, w, 480, 384, seed=2, noise=1.5,
                            pose=(rot_zyx(QR.yaw + true[0], QR.pitch + true[1], QR.roll + true[2]), QR.pos()))
    geos = {'MR': OC._CamGeo(p, MR, (384, 480)), 'QR': OC._CamGeo(p, QR, (384, 480))}
    pr = OC.ground_pairs(p, geos, eng, v, 8.0)[0]
    assert (pr['a'], pr['b']) == ('MR', 'QR') and len(pr['blocks']) >= 20
    got = OC.ground_factor(pr, geos['MR'], geos['QR'], cv2.cvtColor(iA, cv2.COLOR_RGB2GRAY),
                           cv2.cvtColor(iB, cv2.COLOR_RGB2GRAY), 0.0, 0.0, (np.array([0.0, 0.0, 1.0]), 0.0),
                           DL.Motion('none'), None, 1.5)
    assert len(got) >= 20
    lay = OC.Layout([MR, QR], None, False)
    rows = OC.Rows(lay.n)
    H = np.zeros((3, lay.n))
    H[np.arange(3), lay.rot['MR']] = 1.0
    rows.add(H, np.zeros(3), np.full(3, 1e-3))
    for g, d, s, ncc, JA, JB, V in got:
        H = np.zeros((2, lay.n))
        H[:, lay.rot['MR']] = JA
        H[:, lay.rot['QR']] = -JB
        rows.add(V.T @ H, V.T @ d, s)
    A, b, gid = rows.arrays()
    x, P, _ = OC.solve_window(A, b, gid, rows.ng, lay.prior, 1.5)
    est, sig = x[lay.rot['QR']], np.sqrt(np.diag(P))[lay.rot['QR']]
    assert np.all(np.abs(est - true) <= 1.5 * sig + 0.02), (est, sig)
    assert sig[1] == min(sig) and sig[1] < 0.1


# ============================================================================ a drive with a knock
@pytest.fixture(scope='module')
def knock(tmp_path_factory):
    """Two 4 s segments at 30 km/h, QR yawed 0.8° in the second; calibrated in 1 s windows of 4 frames."""
    base = tmp_path_factory.mktemp('online_knock')
    eng = Engine()
    a = segment(str(base / 'a'), eng, None)
    b = segment(str(base / 'b'), eng, {KNOCK[0]: KNOCK[1]})
    path = str(base / 'joined')
    starts = join_nuscenes([a, b], path)
    p, v = make_project()
    st = OC.OnlineCalibSettings(path=path, window_s=1.0, stride_s=1.0, frames_per_window=4, outline=True, ground=True)
    p.analysis.online_calib = st.to_dict()
    t0 = time.time()
    res = OC.run(p, eng, st)
    return {'path': path, 'starts': starts, 'p': p, 'v': v, 'eng': eng, 'res': res, 'seconds': time.time() - t0,
            'st': st, 'base': base}


def test_the_knock_is_found_with_its_time_size_and_the_clean_poses(knock):
    res = knock['res']
    t_step = knock['starts'][1]
    knocks = [e for e in res.events if e['kind'] == 'knock']
    assert len(knocks) == 1 and knocks[0]['sensor'] == 'QR' and 'yaw' in knocks[0]['axes']
    e = knocks[0]
    assert e['t_s'] == pytest.approx(t_step, abs=0.6)                            # within a window
    assert e['size']['yaw'] == pytest.approx(0.8, abs=0.2)
    assert e['beyond_tolerance'] and OC.exit_code(res) == 3
    assert 'QR' in e['message'] and 'yawed +0.' in e['message'] and 'stayed (knock)' in e['message']
    assert res.findings[0]['severity'] == 'serious' and res.findings[0]['kind'] == 'knock'
    assert not [x for x in res.events if x['sensor'] != 'QR']
    # the segments: QR before and after, the other cameras one segment as planned
    segs = res.segments['QR']
    assert len(segs) == 2 and segs[0]['t1'] == pytest.approx(e['t_s'])
    yaw = segs[0]['axes'].index('yaw')
    assert abs(segs[0]['delta'][yaw]) < 0.1 and segs[1]['delta'][yaw] == pytest.approx(0.8, abs=0.2)
    for tag in ('F', 'MR'):
        assert len(res.segments[tag]) == 1
        assert max(abs(d) for d in res.segments[tag][0]['delta'] if d is not None) < 0.1
    # the series jumps at the step, not across it
    d0, _ = OC.delta_at(res, 'QR', e['t_s'] - 0.01)
    d1, _ = OC.delta_at(res, 'QR', e['t_s'] + 0.01)
    assert d1[yaw] - d0[yaw] == pytest.approx(e['size']['yaw'], abs=0.1)
    # every window says what it measured and which axes that constrains
    assert len(res.windows) == 8 and all(w['frames'] >= 3 for w in res.windows)
    kinds = {m['kind'] for w in res.windows for m in w['measurements']}
    assert kinds == {'edges', 'outline', 'ground'}
    assert any(m['kind'] == 'ground' and m['constrains'] for w in res.windows for m in w['measurements'])
    assert {'MR', 'QR'} in [{q['a'], q['b']} for q in res.pairs]
    assert all(set(w['constrained']) == {'F', 'MR', 'QR'} for w in res.windows)
    assert knock['seconds'] < 90


def test_the_corrected_variant_takes_the_final_segment(knock):
    res, p = knock['res'], knock['p']
    v = OC.corrected_variant(p, res)
    assert v is not None and 'online calib from 00:04' in v.name
    qr = v.by_tag('QR')
    plan = knock['v'].by_tag('QR')
    assert qr.yaw == pytest.approx(plan.yaw + 0.8, abs=0.2) and qr.uid != plan.uid
    for tag in ('F', 'MR'):                                         # nothing significant to apply: as planned
        c, c0 = v.by_tag(tag), knock['v'].by_tag(tag)
        assert (c.yaw, c.pitch, c.roll) == pytest.approx((c0.yaw, c0.pitch, c0.roll), abs=1e-6)


def test_the_nuscenes_export_has_a_row_per_segment_and_leaves_the_input_alone(knock, tmp_path):
    res = knock['res']
    ver_in = os.path.join(knock['path'], DS.VERSION)
    before = _tables_digest(ver_in)
    info = OC.export_nuscenes(res, knock['p'], str(tmp_path / 'nus'))
    assert _tables_digest(ver_in) == before
    ver = os.path.join(info['path'], DS.VERSION)
    with open(os.path.join(ver, 'calibrated_sensor.json')) as fh:
        cal = json.load(fh)
    with open(os.path.join(ver, 'sample_data.json')) as fh:
        sds = json.load(fh)
    with open(os.path.join(ver, 'sensor.json')) as fh:
        chan = {s['token']: s['channel'] for s in json.load(fh)}
    new = [r for r in cal if 'vantage_online_calib' in r]
    assert info['rows'] == len(new) == 1 + 1 + 2                    # F, MR one segment each, QR two
    by = {r['token']: r for r in cal}
    t_step = next(e['t_s'] for e in res.events if e['kind'] == 'knock')
    qr = [sd for sd in sds if chan[by[sd['calibrated_sensor_token']]['sensor_token']] == 'CAM_QR']
    assert qr and info['repointed'] >= len(qr)
    for sd in qr:
        row = by[sd['calibrated_sensor_token']]
        seg = row['vantage_online_calib']['segment']
        t = (sd['timestamp'] * 1000 - res.span['start_ns']) / 1e9
        assert seg == (1 if t >= t_step else 0)
    # the rotation: the camera's optical frame of the plan turned by the segment's correction (vehicle frame)
    row = next(r for r in new if r['vantage_online_calib']['segment'] == 1 and
               chan[r['sensor_token']] == 'CAM_QR')
    sg = res.segments['QR'][1]
    plan = knock['v'].by_tag('QR')
    d = [sg['delta'][sg['axes'].index(a)] for a in OC.AXES]
    R_ref = rot_zyx(plan.yaw + d[0], plan.pitch + d[1], plan.roll + d[2]) @ BODY_TO_OPTICAL.T
    assert np.allclose(OC._quat_wxyz_R(row['rotation']), R_ref, atol=1e-6)
    lid = [sd for sd in sds if chan[by[sd['calibrated_sensor_token']]['sensor_token']].startswith('LIDAR')]
    assert lid and all('vantage_online_calib' not in by[sd['calibrated_sensor_token']] for sd in lid)
    assert set(info['linked']) >= {'samples', 'sweeps'}
    f0 = next(sd['filename'] for sd in qr)
    assert os.path.isfile(os.path.join(info['path'], f0))
    with pytest.raises(ValueError):
        OC.export_nuscenes(res, knock['p'], os.path.join(knock['path'], 'out'))    # never inside the input
    # the rewritten folder reads as a drive again, its channels intact
    assert DL.open_log(info['path']).counts == DL.open_log(knock['path']).counts


def test_the_tf_sidecar_follows_the_segments(knock, tmp_path):
    res = knock['res']
    out = OC.export_tf_bag(res, knock['p'], str(tmp_path / 'tf'))
    rd = RB.BagReader(str(tmp_path / 'tf'))
    try:
        assert set(rd.topics) == {'/tf', '/tf_static'}
        dyn, static = [], []
        for topic, t_ns, msg in rd.messages():
            (dyn if topic == '/tf' else static).append((t_ns, msg))
    finally:
        rd.close()
    kids = {tr['child_frame_id'] for _, m in static for tr in m['transforms']}
    assert {'L1_link', 'F_optical_frame', 'QR_optical_frame'} <= kids and 'QR_link' not in kids
    t_step = next(e['t_s'] for e in res.events if e['kind'] == 'knock')
    plan = knock['v'].by_tag('QR')
    yaws = []
    for t_ns, m in dyn:
        tr = next(x for x in m['transforms'] if x['child_frame_id'] == 'QR_link')
        assert tr['header']['frame_id'] == 'base_link'
        R = RB.transform_matrix(tr)[:3, :3]
        yaws.append(((t_ns - res.span['start_ns']) / 1e9, math.degrees(math.atan2(R[1, 0], R[0, 0])) - plan.yaw))
    yaws = np.array(yaws)
    assert out['stamps'] == len(dyn) >= res.span['duration_s'] * res.settings.tf_hz
    assert abs(np.median(yaws[yaws[:, 0] < t_step - 0.6, 1])) < 0.1
    assert np.median(yaws[yaws[:, 0] > t_step + 0.6, 1]) == pytest.approx(0.8, abs=0.2)
    near = yaws[np.abs(yaws[:, 0] - t_step) < 0.002]
    assert len(near) == 2 and near[1, 1] - near[0, 1] > 0.4                  # a jump 1 ms wide


def test_documents_files_and_determinism(knock, tmp_path):
    res = knock['res']
    doc = OC.to_doc(res, knock['p'])
    s = json.dumps(doc)
    assert doc['schema'] == OC.SCHEMA and doc['exit_code'] == 3 and doc['limits'] == OC.ASSUMPTIONS
    assert doc['settings'] == OC.OnlineCalibSettings.from_dict(doc['settings']).to_dict()
    assert set(doc['downstream']) == {'nuscenes', 'ros2', 'json'} and 'NaN' not in s
    w0 = doc['windows'][0]
    assert {m['kind'] for m in w0['measurements']} >= {'edges', 'ground'}
    assert all('constrains' in m for m in w0['measurements'])
    ext = doc['extrinsics']
    assert ext['schema'] == OC.EXTRINSICS_SCHEMA and len(ext['sensors']['QR']['segments']) == 2
    seg = ext['sensors']['QR']['segments'][1]
    assert seg['yaw_pitch_roll_deg'][0] == pytest.approx(knock['v'].by_tag('QR').yaw + 0.8, abs=0.2)
    assert len(ext['sensors']['QR']['interpolated']['t_s']) == len(res.windows)
    names = [n for n, _ in OC.files(res, knock['p'])]
    assert names == ['online_calib.json', 'online_calib_extrinsics.json', 'online_calib_timeline.csv',
                     'online_calib_timeline.png']
    csv_ = OC.to_csv(res).splitlines()
    assert csv_[0].split(',') == list(OC.CSV_COLUMNS) and len(csv_) == 1 + len(res.windows) * 9
    png = dict(OC.files(res, knock['p']))['online_calib_timeline.png']
    assert png[:8] == b'\x89PNG\r\n\x1a\n'
    paths = OC.export(res, knock['p'], str(tmp_path / 'exp'))
    assert any(q.endswith('_online_tf') for q in paths) and any(q.endswith('nuscenes') for q in paths)
    # the same drive and settings again (its scan kept): the same numbers bit for bit
    again = OC.run(knock['p'], knock['eng'], knock['st'], scan=res._scan)
    assert again.series['QR']['est'] == res.series['QR']['est'] and again.events[0]['t_s'] == res.events[0]['t_s']


def test_batch_cancel_and_errors(knock, tmp_path):
    p = knock['p']
    st = OC.OnlineCalibSettings(window_s=4.0, stride_s=4.0, frames_per_window=4, outline=False)
    res, code = OC.batch(p, knock['path'], str(tmp_path / 'b'), st)
    assert res is not None and code in (0, 3) and os.path.isfile(str(tmp_path / 'b' / 'online_calib.json'))
    assert OC.batch(p, str(tmp_path / 'nothing'), str(tmp_path / 'c'), st) == (None, 4)
    calls = []

    def cancel():
        calls.append(1)
        return len(calls) > 3
    assert OC.run(p, knock['eng'], st, cancel=cancel, path=knock['path']) is None
    with pytest.raises(ValueError):
        OC.run(p, knock['eng'], OC.OnlineCalibSettings())                     # no drive
    q, w = make_project(cams=())
    with pytest.raises(ValueError):
        OC.run(q, knock['eng'], st, path=knock['path'])                       # no camera has a topic


# ============================================================================ a drive with flex
@pytest.fixture(scope='module')
def flex(tmp_path_factory):
    """Four 2 s segments at 10, 50, 20 and 45 km/h, F pitched FLEX_K° per m/s of the segment's speed."""
    base = tmp_path_factory.mktemp('online_flex')
    eng = Engine()
    parts = [segment(str(base / f's{i}'), eng, {'F': (0.0, FLEX_K * v / 3.6, 0.0)}, dur=2.0, speed=v)
             for i, v in enumerate(FLEX_SPEEDS)]
    path = str(base / 'joined')
    join_nuscenes(parts, path)
    p, v = make_project()
    st = OC.OnlineCalibSettings(window_s=1.0, stride_s=1.0, frames_per_window=4, ground=True)
    return {'res': OC.run(p, eng, st, path=path), 'p': p}


def test_flex_with_speed_is_found_and_not_taken_for_knocks(flex):
    res = flex['res']
    fx = [e for e in res.events if e['kind'] == 'flex']
    assert len(fx) == 1 and fx[0]['sensor'] == 'F' and fx[0]['axes'] == ['pitch'] and fx[0]['covariate'] == 'speed'
    assert fx[0]['slope'] == pytest.approx(FLEX_K, abs=0.012)
    lo, hi = (v / 3.6 for v in (min(FLEX_SPEEDS), max(FLEX_SPEEDS)))
    assert fx[0]['amplitude'] == pytest.approx(0.5 * FLEX_K * (hi - lo), abs=0.08)
    assert 'varies ±' in fx[0]['message'] and 'with speed (flex)' in fx[0]['message']
    assert not [e for e in res.events if e['kind'] in ('knock', 'excursion')] and OC.exit_code(res) == 0
    # the smoothed pitch follows the speed: its spread over the windows is the flex's
    pitch = np.array(res.series['F']['est'])[:, 1]
    speed = np.array([w['covariates']['speed'] for w in res.windows])
    assert np.corrcoef(pitch, speed)[0, 1] > 0.9
    # the segment's pose at rest: the slowest speed driven
    sg = res.segments['F'][0]
    assert sg['delta'][1] == pytest.approx(FLEX_K * speed.min(), abs=0.06)


# ============================================================================ the page
from test_gui import app, win, pump  # noqa: E402,F401 (fixtures)


def test_page_runs_shows_applies_undoes_and_exports(app, win, knock, tmp_path):
    from vantage.ui.workspaces.online_calib_ws import OnlineCalibPage
    p, _ = make_project()
    win.store.set_project(p)
    pump(app, 0.3)
    pg = OnlineCalibPage(win.ctx)
    pg.resize(1400, 900)
    pg.show()
    pump(app, 0.2)
    assert not pg.b_run.isEnabled() and not pg.b_export.isEnabled() and not pg.b_apply.isEnabled()
    assert pg.open_path(knock['path'])
    pump(app, 0.1)
    assert pg.b_run.isEnabled() and pg.settings.path == DL.norm_path(knock['path'])
    # settings through their fields and switches, undoable
    pg.fields['window_s'].set_value(1.0)
    pg.fields['window_s'].committed.emit(1.0)
    pg.fields['stride_s'].set_value(1.0)
    pg.fields['stride_s'].committed.emit(1.0)
    assert pg.settings.window_s == 1.0 and pg.settings.stride_s == 1.0
    pg.switches['outline'].setChecked(True)
    assert pg.settings.outline is True
    win.store.undo.undo()
    pump(app, 0.1)
    assert pg.settings.outline is False and not pg.switches['outline'].isChecked()
    assert pg.run()
    assert pg.busy and pg.b_cancel.isEnabled() and not pg.b_run.isEnabled()
    t0 = time.time()
    while (pg.busy or pg.result is None) and time.time() - t0 < 240:
        pump(app, 0.1)
    res = pg.result
    assert res is not None and not pg.busy and 'Failed' not in pg.status.text()
    assert len(pg.plots) == 3 and pg.events.rowCount() == len(res.events) >= 1
    assert pg.events.item(0, 0).text() == 'QR' and pg.events.item(0, 1).text() == 'knock'
    assert len(pg.findings) >= len(res.findings)
    pg.plots[2].grab()                                               # paints (QR with its knock)
    assert pg.summary()['exit_code'] == 3
    # the final segment as a new variant, one undo step
    n = len(win.store.project.variants)
    assert pg.b_apply.isEnabled()
    uid = pg.apply_final()
    pump(app, 0.2)
    pj = win.store.project
    assert len(pj.variants) == n + 1 and pj.active_uid == uid and 'online calib' in pj.variant.name
    win.store.undo.undo()
    pump(app, 0.2)
    assert len(win.store.project.variants) == n
    paths = pg.export(str(tmp_path / 'exp'))
    assert any(q.endswith('online_calib.json') for q in paths) and all(os.path.exists(q) for q in paths)
    # cancel: a second run stopped straight away
    assert pg.run()
    pg.cancel()
    t0 = time.time()
    while pg.busy and time.time() - t0 < 120:
        pump(app, 0.1)
    assert not pg.busy and pg.status.text().startswith('Cancelled')
    # a new project drops the result
    win.store.set_project(samples.tigor_sample_project())
    pump(app, 0.2)
    assert pg.result is None and pg.events.rowCount() == 0 and not pg.plots
    pg.close()
    pg.deleteLater()
