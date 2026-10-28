"""The CARLA bridge: frames against the real carla 0.9.16 module's matrices, the fisheye remap against Vantage's lens
model, lidar resampling, the rig file, the bridge end to end against a fake carla module (tests/fake_carla), the
recording read back by the drive-log and as-built checks, tidy-up after a cancel or an error, and the page.

Tests marked carla_live run only when a CARLA server is reachable (VANTAGE_CARLA=host:port) and CARLA's client Python
is given (VANTAGE_CARLA_PYTHON); they are skipped otherwise."""
import importlib.util
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
FAKE_DIR = os.path.join(HERE, 'fake_carla')


def _fake():
    """The fake carla module, loaded under its own name (never shadowing a real carla)."""
    name = 'fake_carla_for_tests'
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, os.path.join(FAKE_DIR, 'carla', '__init__.py'))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _project(lidar_z=None, extra=True):
    from vantage.core import samples
    from vantage.core.project import Sensor
    p = samples.tigor_sample_project()
    v = next(v for v in p.variants if v.sensors)
    p.active_uid = v.uid
    if lidar_z is not None:
        next(s for s in v.sensors if s.kind == 'lidar').z = lidar_z
    if extra:
        v.sensors.append(Sensor(tag='IMU1', name='INS', kind='imu', model_id='ins_dual', x=1.0, y=0.0, z=0.5))
        v.sensors.append(Sensor(tag='G1', name='GNSS front', kind='gnss', model_id='gnss_survey', x=1.6, y=0.0, z=1.45))
        v.sensors.append(Sensor(tag='G2', name='GNSS rear', kind='gnss', model_id='gnss_survey', x=0.2, y=0.0, z=1.45))
    return p


# ======================================================================= frames
# carla.Transform(Location(0.5, -1.25, 2.0), Rotation(pitch, yaw, roll)).get_matrix()[:3][:3] from the real carla
# 0.9.16 client module (PyPI wheel), printed on 2026-10-27
REAL_CARLA = [
    ((20.02, 143.0, 99.25), [[-0.7503765, -0.1731187, -0.6379381], [0.5654493, 0.3317258, -0.7551325],
                             [0.3423481, -0.9273553, -0.1510294]]),
    ((-43.97, -71.94, 134.48), [[0.2231172, -0.8197103, 0.5275356], [-0.6842453, 0.2537475, 0.6836816],
                                [-0.6942816, -0.5135049, -0.5042675]]),
    ((-79.16, 115.64, 106.94), [[-0.0813794, 0.6692326, -0.7385831], [0.1695482, -0.7209446, -0.6719317],
                                [-0.9821562, -0.1799069, -0.0547971]]),
    ((-5.13, -70.91, -79.77), [[0.3257428, 0.196612, -0.9247894], [-0.9412208, -0.0250706, -0.3368605],
                               [-0.0894158, 0.9801608, 0.1768887]]),
    ((-39.22, -19.77, 1.64), [[0.7290601, 0.3210772, 0.6044674], [-0.2620466, 0.9467934, -0.1868522],
                              [-0.6322998, -0.0221722, 0.7744065]]),
    ((8.56, 178.38, 105.36), [[-0.9884653, -0.1359826, -0.0666714], [0.0279555, 0.2688347, -0.9627805],
                              [0.148845, -0.953539, -0.2619323]]),
    ((10.0, 20.0, 30.0), [[0.92541653, -0.21461014, -0.31232452], [0.33682406, 0.84349322, 0.41841206],
                          [0.17364818, -0.49240386, 0.8528685]]),
]


def test_carla_matrix_matches_the_real_carla_module():
    from vantage.sim import frames as FR
    for (p, y, r), M in REAL_CARLA:
        assert np.allclose(FR.carla_matrix(p, y, r), M, atol=2e-7)
    # Transform.transform(Location(1, 0, 0)) of Transform(Location(1, 2, 3), Rotation(10, 20, 30)), from carla 0.9.16
    T = FR.transform_matrix((1, 2, 3), (10, 20, 30))
    assert np.allclose(T @ [1, 0, 0, 1], [1.925416, 2.336824, 3.173648, 1], atol=1e-6)


def test_carla_euler_round_trip():
    from vantage.sim import frames as FR
    rng = np.random.default_rng(3)
    for p, y, r in rng.uniform([-89, -179, -179], [89, 179, 179], (200, 3)):
        p2, y2, r2 = FR.carla_euler(FR.carla_matrix(p, y, r))
        assert np.allclose([p2, y2, r2], [p, y, r], atol=1e-7)
    # gimbal lock keeps the rotation
    M = FR.carla_matrix(90.0, 30.0, 20.0)
    assert np.allclose(FR.carla_matrix(*FR.carla_euler(M)), M, atol=1e-9)


def test_vantage_pose_to_carla_closed_form_and_points():
    """carla yaw = −yaw, pitch = −pitch, roll = roll; and a sensor-frame point lands on the same vehicle point."""
    from vantage.core.mathutil import rot_zyx
    from vantage.sim import frames as FR
    rng = np.random.default_rng(5)
    for yaw, pitch, roll in rng.uniform([-179, -80, -179], [179, 80, 179], (100, 3)):
        R = rot_zyx(yaw, pitch, roll)
        p, y, r = FR.carla_rotation(R)
        assert np.allclose([p, y, r], [-pitch, FR._wrap(-yaw), FR._wrap(roll)], atol=1e-7)
        t = rng.uniform(-2, 2, 3)
        q = rng.uniform(-5, 5, 3)                                  # a point in the sensor frame (Vantage axes)
        veh = R @ q + t
        Tc = FR.transform_matrix(FR.carla_location(t), (p, y, r))
        assert np.allclose(Tc @ np.r_[FR.to_vantage(q), 1.0], np.r_[FR.to_vantage(veh), 1.0], atol=1e-9)
    # forward: a camera pitched down 40° in Vantage looks down in CARLA too (CARLA's z is up)
    p, y, r = FR.carla_rotation(rot_zyx(30.0, 40.0, 0.0))
    fwd = FR.carla_matrix(p, y, r)[:, 0]
    assert fwd[2] < 0 and fwd[1] < 0                               # down, and to the left (CARLA's −y)


def test_angular_velocity_mirror():
    """A rate in CARLA's (left-handed) axes maps to Vantage's as (−ωx, ωy, −ωz): checked by finite rotations."""
    from vantage.core.mathutil import rot_zyx
    from vantage.sim import frames as FR
    rng = np.random.default_rng(9)
    for _ in range(20):
        w = rng.normal(0, 0.5, 3)
        R0 = rot_zyx(*rng.uniform(-90, 90, 3))
        dt = 1e-4
        R1 = R0 @ _expm(w * dt)                                    # body rate w (Vantage axes)
        C0, C1 = FR.S @ R0 @ FR.S, FR.S @ R1 @ FR.S                # the same motion seen in CARLA's axes
        wc = FR.rotvec(C0.T @ C1) / dt                             # what a right-handed formula reads off CARLA's numbers
        assert np.allclose(FR.angular_to_vantage(wc), w, atol=1e-5)


def _expm(v):
    th = np.linalg.norm(v)
    if th < 1e-15:
        return np.eye(3)
    k = v / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


# ======================================================================= cameras: the fisheye remap
def _render_views(cam_doc, cam, intr_rot):
    """Each view's image of the direction field in the camera's optical frame, rendered as CARLA would: the view's
    carla.Rotation (from the rig file), CARLA's pinhole (f = W / 2 tan(fov/2), centre at the image centre), its
    actor axes (x forward, y right, z up) — independent of the remap's own matrices."""
    from vantage.core.mathutil import BODY_TO_OPTICAL
    from vantage.sim import frames as FR
    W2O = BODY_TO_OPTICAL @ intr_rot.T                             # vehicle -> camera optical
    at = cam_doc['atlas']
    atlas = np.zeros((at['height'], at['width'], 3), np.float32)
    for k, v in enumerate(cam_doc['views']):
        Wd, Hd = v['width'], v['height']
        f = Wd / (2.0 * math.tan(math.radians(v['fov_deg']) / 2))
        u, vv = np.meshgrid(np.arange(Wd, dtype=float), np.arange(Hd, dtype=float))
        d = np.stack([np.full_like(u, f), u - (Wd / 2 - 0.5), -(vv - (Hd / 2 - 0.5))], -1)
        d /= np.linalg.norm(d, axis=-1, keepdims=True)
        rot = v['carla']['rotation']
        Mc = FR.carla_matrix(rot['pitch'], rot['yaw'], rot['roll'])
        dv = FR.to_vantage(d @ Mc.T)                               # CARLA vehicle axes -> Vantage vehicle frame
        atlas[at['offsets'][k]:at['offsets'][k] + Hd, :Wd] = dv @ W2O.T
    return atlas


def test_fisheye_remap_reproduces_the_lens_model():
    """The remapped image of a direction field equals Vantage's own projection: at every pixel centre, and at random
    directions projected with Intrinsics.project_dirs (sub-pixel)."""
    import cv2
    from vantage.core import lens, simrig
    from vantage.core.mathutil import rot_zyx
    p = _project(extra=False)
    v = p.variant
    st = simrig.SimSettings.from_dict({'image_scale': 0.25})
    rig = simrig.build(p, st)
    for tag in ('F', 'ML'):
        cam = v.by_tag(tag)
        doc = next(c for c in rig.doc['cameras'] if c['tag'] == tag)
        assert len(doc['views']) == 5 and {x['name'] for x in doc['views']} == {'front', 'left', 'right', 'up', 'down'}
        intr = lens.resolve(p.module_for(cam), p.lens_for(cam))
        atlas = _render_views(doc, cam, rot_zyx(cam.yaw, cam.pitch, cam.roll))
        mx, my = rig.remaps[doc['remap']['x']], rig.remaps[doc['remap']['y']]
        out = cv2.remap(atlas, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        w, h = doc['image']['width'], doc['image']['height']
        d, ok = simrig.output_rays(intr, w, h)
        f_px = intr.f_px * w / intr.width
        got = out[ok] / np.linalg.norm(out[ok], axis=1, keepdims=True)
        err_px = np.degrees(np.arccos(np.clip(np.sum(got * d[ok], 1), -1, 1))) * math.pi / 180 * f_px
        assert ok.mean() > 0.6
        # cv2.remap reads at 1/32 px steps (its fixed-point maps): about 0.03 px, plus bilinear interpolation
        assert np.percentile(err_px, 99) < 0.1 and err_px.max() < 0.15, (tag, np.percentile(err_px, 99), err_px.max())
        # random directions through the lens model: sample the remapped image where Vantage projects them
        rng = np.random.default_rng(11)
        D = rng.normal(size=(4000, 3))
        D[:, 2] = np.abs(D[:, 2]) + 0.05
        D /= np.linalg.norm(D, axis=1, keepdims=True)
        u, vv, th, inimg = intr.project_dirs(D)
        sel = inimg & (u > 4) & (vv > 4) & (u < intr.width - 4) & (vv < intr.height - 4) & \
            (th < intr.max_theta - 0.02)
        uo = u[sel] * w / intr.width - 0.5                          # OpenCV coordinates of the output image
        vo = vv[sel] * h / intr.height - 0.5
        samp = cv2.remap(out, uo.astype(np.float32).reshape(-1, 1), vo.astype(np.float32).reshape(-1, 1),
                         cv2.INTER_LINEAR).reshape(-1, 3)
        samp /= np.linalg.norm(samp, axis=1, keepdims=True)
        e = np.arccos(np.clip(np.sum(samp * D[sel], 1), -1, 1)) * f_px
        assert sel.sum() > 1000 and np.percentile(e, 99) < 0.1, (tag, np.percentile(e, 99))


def test_cube_faces_feathered_across_their_seams():
    """Near a seam a pixel mixes both faces (smoothstep weights summing to 1, the first face at least a third): every
    face's sample is the same direction, so the feathered image still reproduces the lens model; and a step between
    two views (CARLA renders the views of a moving camera slightly differently) fades over the band instead of
    showing as a line."""
    import cv2
    from vantage.core import lens, simrig
    from vantage.core.mathutil import rot_zyx
    from vantage.sim import carla_bridge as CB
    p = _project(extra=False)
    v = p.variant
    rig = simrig.build(p, simrig.SimSettings.from_dict({'image_scale': 0.25}))
    cam = v.by_tag('F')
    doc = next(c for c in rig.doc['cameras'] if c['tag'] == 'F')
    assert doc['remap']['seam_blend_deg'] == simrig.SEAM_BLEND_DEG
    t = math.tan(math.radians(45 + simrig.SEAM_BLEND_DEG / 2))
    for x in doc['views']:                                  # each face reaches half the band past its seams
        assert x['width'] / 2 - simrig.FACE_MARGIN_PX >= x['f_px'] * t - 1e-6
    blend = tuple(rig.remaps[doc['remap']['blend'][k]] for k in ('idx', 'x', 'y', 'w'))
    idx, bx, by, bw = blend
    assert len(idx) > 1000 and np.allclose(bw.sum(1), 1, atol=1e-6) and bw[:, 0].min() > 1 / 3 - 1e-6
    assert np.all((bx[:, 0] >= 0) & (bw[:, 1] > 0)) and np.all((bx[:, 1] < 0) == (bw[:, 2] == 0))
    intr = lens.resolve(p.module_for(cam), p.lens_for(cam))
    atlas = _render_views(doc, cam, rot_zyx(cam.yaw, cam.pitch, cam.roll))
    mx, my = rig.remaps[doc['remap']['x']], rig.remaps[doc['remap']['y']]
    first = cv2.remap(atlas, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    w, h = doc['image']['width'], doc['image']['height']
    d, ok = simrig.output_rays(intr, w, h)
    f_px = intr.f_px * w / intr.width
    for j in range(2):                                      # each other face's sample: the pixel's own direction
        has = bx[:, j] >= 0
        smp = CB._sample(atlas, bx[has, j], by[has, j])
        smp /= np.linalg.norm(smp, axis=1, keepdims=True)
        e = np.arccos(np.clip(np.sum(smp * d.reshape(-1, 3)[idx[has]], 1), -1, 1)) * f_px
        assert has.sum() > 0 and np.percentile(e, 99) < 0.1 and e.max() < 0.15, (j, e.max())
    out = CB.feather(first.copy(), atlas, blend)
    got = out[ok] / np.linalg.norm(out[ok], axis=1, keepdims=True)
    e = np.arccos(np.clip(np.sum(got * d[ok], 1), -1, 1)) * f_px
    assert np.percentile(e, 99) < 0.1 and e.max() < 0.15
    # a step of 40 levels between the views: one line without feathering, a ramp with it
    at = doc['atlas']
    flat = np.zeros((at['height'], at['width'], 3), np.uint8)
    for k, x in enumerate(doc['views']):
        flat[at['offsets'][k]:at['offsets'][k] + x['height'], :x['width']] = 100 + 40 * (k % 2)
    hard = cv2.remap(flat, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    soft = CB.feather(hard.copy(), flat, blend)
    inner = cv2.erode(ok.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    jump = lambda img: max(np.abs(np.diff(img[..., 0].astype(int), axis=a))[(inner[:-1] if a == 0 else inner[:, :-1])
                                                                          ].max() for a in (0, 1))
    assert jump(hard) == 40 and jump(soft) <= 8, (jump(hard), jump(soft))


def test_lidar_points_moved_to_where_the_sensor_was_when_it_fired():
    """CARLA casts a step's slice at the step's end; carla_bridge.fired_at puts each point in the sensor's pose at its
    firing time: against poses built independently (translation and rotation interpolated over the step)."""
    from vantage.sim import carla_bridge as CB
    from vantage.sim import frames as FR
    W_prev = FR.transform_matrix((10.0, -3.0, 1.9), (0.4, 30.0, -0.2))
    W_cur = FR.transform_matrix((10.0 + 0.1 * math.cos(math.radians(30.5)), -3.0 + 0.1 * math.sin(math.radians(30.5)),
                                 1.905), (0.45, 31.0, -0.1))
    rng = np.random.default_rng(3)
    X = rng.uniform(-30, 30, (500, 3)) + np.array([10.0, -3.0, 0.0])         # world points
    P_cur = (np.linalg.inv(W_cur) @ np.c_[X, np.ones(len(X))].T).T[:, :3]    # as CARLA returns them (step's end)
    a = rng.uniform(0, 1, len(X))
    got = CB.fired_at(P_cur, W_prev, W_cur, a)
    T = np.linalg.inv(W_prev) @ W_cur
    w, V = np.linalg.eig(T[:3, :3])
    for i in range(0, len(X), 50):                                          # the pose at a: W_prev · exp(a log T)
        L = (V @ np.diag(np.log(w.astype(complex)) * a[i]) @ np.linalg.inv(V)).real
        Ta = np.eye(4)
        Ta[:3, :3] = _expm(np.array([L[2, 1], L[0, 2], L[1, 0]]))
        Ta[:3, 3] = a[i] * T[:3, 3]
        want = (np.linalg.inv(W_prev @ Ta) @ np.append(X[i], 1.0))[:3]
        assert np.allclose(got[i], want, atol=1e-9), (got[i], want)
    assert np.allclose(CB.fired_at(P_cur, W_prev, W_cur, np.ones(len(X))), P_cur, atol=1e-9)


def test_server_renders_on_the_nvidia_gpu_and_never_a_gpu_without_rendering(tmp_path, monkeypatch):
    """Rendering: the Vulkan loader pointed at NVIDIA's driver when it is loaded (an integrated GPU drives the
    desktop), unless the user chose a driver; no rendering: at no driver at all."""
    from vantage.sim import server as SV
    proc = tmp_path / 'nvidia_version'
    icd = tmp_path / 'icd.d'
    icd.mkdir()
    (icd / 'nvidia_icd.json').write_text('{}')
    assert SV.nvidia_vulkan(str(proc), (str(icd),)) == ''                   # no NVIDIA driver loaded
    proc.write_text('NVRM version')
    assert SV.nvidia_vulkan(str(proc), (str(tmp_path / 'none'), str(icd))) == str(icd / 'nvidia_icd.json')
    root = tmp_path / 'CARLA'
    (root / os.path.dirname(SV.BINARY)).mkdir(parents=True)
    (root / SV.BINARY).write_text('')
    ran = []

    class P:
        pid = 0

        def __init__(self, cmd, **kw):
            ran.append((cmd, kw['env']))

        def poll(self):
            return 0
    monkeypatch.setattr(SV.subprocess, 'Popen', P)
    monkeypatch.setattr(SV, 'port_open', lambda *a, **k: False)
    monkeypatch.setattr(SV, 'scoped', lambda cmd, memory_gb, unit='': list(cmd))      # (no systemd needed here)
    monkeypatch.setattr(SV, 'nvidia_vulkan', lambda: '/x/nvidia_icd.json')
    for var in ('VK_ICD_FILENAMES', 'VK_DRIVER_FILES'):
        monkeypatch.delenv(var, raising=False)
    SV.Server(str(root)).start()
    assert ran[-1][1]['VK_ICD_FILENAMES'] == ran[-1][1]['VK_DRIVER_FILES'] == '/x/nvidia_icd.json'
    assert '-RenderOffScreen' in ran[-1][0]
    SV.Server(str(root), no_render=True).start()
    assert ran[-1][1]['VK_ICD_FILENAMES'] == '/nonexistent.json' and '-nullrhi' in ran[-1][0]
    monkeypatch.setenv('VK_ICD_FILENAMES', '/mine.json')                     # the user's choice stands
    SV.Server(str(root)).start()
    assert ran[-1][1]['VK_ICD_FILENAMES'] == '/mine.json' and 'VK_DRIVER_FILES' not in ran[-1][1]


def test_a_stream_stamped_on_the_simulation_step_reads_its_rate():
    """A 15 Hz radar on 80 Hz steps (each sample at the nearest step: intervals of 62.5, 62.5 and 75 ms) reads 15 Hz in
    the drive-log check, not the 16 Hz of its median interval; drops still count and stay out of the rate."""
    from vantage.core import drivelog as DL
    t = np.round(np.arange(0, 6, 1 / 15.0) * 80) / 80 * 1e9
    s = DL.interval_stats(t.astype(np.int64), 1 / 15.0, 1.5)
    assert s['rate_hz'] == pytest.approx(15.0, rel=2e-3) and s['drops'] == 0 and s['median_ms'] == pytest.approx(62.5)
    s = DL.interval_stats(np.delete(t, [30, 31]).astype(np.int64), 1 / 15.0, 1.5)
    assert s['drops'] == 2 and s['rate_hz'] == pytest.approx(15.0, rel=5e-3)


def test_vignette_taken_out_through_the_measured_response():
    """tonemap: UE's vignette mask (its formula, the corners at √2 for any aspect), the mask each output pixel reads,
    and the correction: an image darkened in linear light by the mask (through the response) comes back within a
    level."""
    from vantage.sim import tonemap as TM
    assert TM.vignette_mask(1.0) == pytest.approx(1 / 1.16 ** 2) and TM.vignette_mask(0.0) == 1.0
    for w, h in ((640, 640), (960, 540), (300, 900)):
        r = TM.vignette_radius(np.array([-0.5, w - 0.5, (w - 1) / 2]), np.array([-0.5, h - 0.5, (h - 1) / 2]), w, h)
        assert r[0] == pytest.approx(math.sqrt(2)) and r[1] == pytest.approx(math.sqrt(2)) and r[2] == pytest.approx(0)
    g = TM.RESPONSE_LN
    assert g.shape == (3, 256) and np.all(np.diff(g, axis=1) > 0)
    # a 64 x 48 view and a second one below it in the atlas; output pixels reading their centres and corners
    sizes = [(64, 48), (32, 32)]
    mx = np.array([[31.5, -0.5, 15.5, -1.0]], np.float32)
    my = np.array([[23.5, -0.5, 48 + 15.5, -1.0]], np.float32)
    m = TM.mask_of_remap(mx, my, [0, 48], sizes)
    assert m[0, 0] == pytest.approx(1.0) and m[0, 1] == pytest.approx(TM.vignette_mask(math.sqrt(2)), rel=1e-5)
    assert m[0, 2] == pytest.approx(1.0) and m[0, 3] == 1.0
    # darken a gradient image by a mask through the response, then take the vignette out again
    z = np.tile(np.arange(5, 251, dtype=float), (3, 1)).T[None]                # (1, 246, 3)
    mask = np.linspace(TM.M_LO * 1.001, 1.0, z.shape[1])[None]
    dark = np.empty_like(z)
    for c in range(3):
        dark[..., c] = np.interp(np.interp(z[..., c], np.arange(256), g[c]) + np.log(mask), g[c], np.arange(256))
    back = TM.Devignette(mask).apply(np.rint(dark).astype(np.uint8))
    keep = dark.min(2) >= 1.0                                                     # not crushed to black by the darkening
    assert keep.mean() > 0.9 and np.abs(back.astype(float) - z)[keep].max() <= 1.5
    lut = TM.correction_lut()
    assert np.array_equal(lut[-1, 0], np.arange(256)) and np.all(lut[:, :, 0] == 0) and np.all(lut[:, :, 255] == 255)


def test_single_view_for_a_narrow_lens_and_rectified_images():
    from vantage.core import lens, simrig
    from vantage.core.project import Camera
    p = _project(extra=False)
    v = p.variant
    narrow = next(l for l in p.lenses.values() if l.sheet_fov and 0 < (l.sheet_fov[0] or 0) < 70)
    cam = v.cameras[0]
    cam.lens_id = narrow.id
    cam.module_id = narrow.module_id or cam.module_id
    rig = simrig.build(p, simrig.SimSettings.from_dict({'image_scale': 0.2}))
    doc = next(c for c in rig.doc['cameras'] if c['tag'] == cam.tag)
    assert len(doc['views']) == 1 and doc['views'][0]['name'] == 'front'
    assert isinstance(cam, Camera)
    # rectified: a pinhole output; every ray it needs within the views
    rig2 = simrig.build(p, simrig.SimSettings.from_dict({'image_scale': 0.2, 'undistort': True}))
    d2 = rig2.doc['cameras'][1]
    assert d2['image']['distortion_model'] == 'plumb_bob' and d2['image']['rectified']
    mx = rig2.remaps[d2['remap']['x']]
    assert (mx >= 0).mean() > 0.95
    del lens


# ======================================================================= lidar
def test_lidar_plan_even_measured_and_off_grid_beams():
    from vantage.core import simrig
    from vantage.core.sensors import SensorModel, builtin_sensor_models
    os1 = next(m for m in builtin_sensor_models() if m.id == 'ouster_os1_128')
    lp = simrig.lidar_plan(os1)
    assert lp['channels'] == 128 and lp['exact'] and lp['elevation_error_deg'] == 0.0
    # measured beams on a 0.125° grid (dense near the horizon, coarse elsewhere): every one exactly on a channel
    beams = sorted(set([-25.0 + 1.0 * k for k in range(16)] + [-9.0 + 0.125 * k for k in range(96)] +
                       [3.0 + 1.0 * k for k in range(13)]))
    m = SensorModel('x', 'Measured', 'lidar', beams=beams, v_min=-25, v_max=15)
    lp = simrig.lidar_plan(m, max_channels=1024)
    g = (lp['upper_fov'] - lp['lower_fov']) / (lp['channels'] - 1)
    assert lp['exact'] and abs(g - 0.125) < 1e-12 and lp['channels'] == 321
    el_of = {ring: lp['upper_fov'] - j * g for j, ring in enumerate(lp['keep']) if ring >= 0}
    assert sorted(el_of) == list(range(len(beams)))
    assert np.allclose([el_of[r] for r in range(len(beams))], beams, atol=1e-9)
    # off any grid within the limit: nearest channels, the error stated and under half a channel
    rng = np.random.default_rng(2)
    odd = -20.0 + np.cumsum(rng.uniform(0.31, 1.2, 40))           # at least 0.31° apart, on no grid
    m2 = SensorModel('y', 'Odd', 'lidar', beams=list(odd))
    lp2 = simrig.lidar_plan(m2, max_channels=200)
    g2 = (lp2['upper_fov'] - lp2['lower_fov']) / (lp2['channels'] - 1)
    assert not lp2['exact'] and 0 < lp2['elevation_error_deg'] <= g2 / 2 + 1e-12
    got = sorted(lp2['upper_fov'] - j * g2 for j, r in enumerate(lp2['keep']) if r >= 0)
    assert np.max(np.abs(np.array(got) - odd)) <= lp2['elevation_error_deg'] + 1e-12
    # two beams closer than the grid still get channels of their own (the error says how far they moved)
    m3 = SensorModel('z', 'Close', 'lidar', beams=[-10.0, 0.0, 0.01, 0.3, 10.0])
    lp3 = simrig.lidar_plan(m3, max_channels=40)
    rings = [r for r in lp3['keep'] if r >= 0]
    assert sorted(rings) == list(range(5)) and lp3['elevation_error_deg'] < 1.0


def test_tick_plan_and_schedules():
    from vantage.core import simrig
    from vantage.sim.carla_bridge import schedule
    st = simrig.SimSettings()
    tp = simrig.tick_plan([(10.0, 1024, 1.31e6)], [('F', 10.0), ('RD1', 15.0), ('RD2', 20.0)], st)
    assert tp['tick_hz'] == 80.0 and tp['per_lidar'] == [(8, 1)] and tp['inexact'] == ['RD1']
    tp = simrig.tick_plan([(10.0, 1024, 1.31e6)], [], simrig.SimSettings.from_dict({'tick_hz': 100}))
    assert tp['tick_hz'] == 100.0 and tp['per_lidar'] == [(10, 5)]            # 5120 columns thinned to 1024
    s = schedule(15.0, 80.0, 160)
    assert abs(len(s) - (2 * 15 + 1)) <= 1 and np.max(np.abs(np.diff(s) / 80.0 - 1 / 15.0)) <= 1 / 80.0
    assert np.array_equal(schedule(20.0, 80.0, 16), np.arange(0, 17, 4))
    assert np.array_equal(schedule(200.0, 80.0, 5), np.arange(6))


def test_radar_cone_holds_the_zone():
    from vantage.core import simrig
    for ah, eh in ((9.0, 7.0), (45.0, 7.0), (75.0, 10.0)):
        c = simrig.radar_cone(ah, eh)
        a, b = math.tan(math.radians(c['horizontal_fov'] / 2)), math.tan(math.radians(c['vertical_fov'] / 2))
        az, el = np.meshgrid(np.linspace(-ah, ah, 41), np.linspace(-eh, eh, 21))
        y = np.tan(np.radians(az))
        z = np.tan(np.radians(el)) * np.sqrt(1 + y * y)             # elevation from the horizontal plane
        assert np.all((y / a) ** 2 + (z / b) ** 2 <= 1 + 1e-9)
        assert 0.05 < c['inside'] <= 1.0


# ======================================================================= the rig file
def test_settings_tolerant_and_rig_document():
    from vantage.core import simrig
    st = simrig.SimSettings.from_dict({'duration_s': 'x', 'traffic': -5, 'image_scale': 7, 'weather': 'Snowy',
                                       'vehicle': 'boat', 'camera_rate': 'fast', 'tick_hz': 0.5, 'nuscenes': 'no'})
    assert st.duration_s == 20.0 and st.traffic == 0 and st.image_scale == 1.0 and st.weather == 'project'
    assert st.vehicle == 'auto' and st.camera_rate == 'lidar' and st.tick_hz == 1.0 and st.nuscenes is False
    assert simrig.SimSettings.from_dict(None) == simrig.SimSettings()
    p = _project()
    rig = simrig.build(p, simrig.SimSettings.from_dict({'image_scale': 0.1}))
    d = rig.doc
    assert d['schema'] == simrig.SCHEMA and d['carla_version'] == '0.9.16'
    assert d['simulation']['tick_hz'] == 80.0 and d['simulation']['warmup_ticks'] % 8 == 0
    assert d['vehicle']['blueprint'] in simrig.CARLA_VEHICLES and d['vehicle']['size_error'] < 0.1
    assert [c['tag'] for c in d['cameras']] == ['F', 'R', 'ML', 'MR', 'QL', 'QR']
    assert {r['channel'] for r in d['nuscenes']['sensor']} >= {'CAM_F', 'LIDAR_L1', 'RADAR_RD1'}
    L1 = d['lidars'][0]
    assert L1['carla']['rotation'] == {'pitch': 0.0, 'yaw': 180.0, 'roll': 0.0} and L1['mount_turn_deg'] == -180.0
    json.dumps(d)                                                  # plain JSON
    s = simrig.summary(rig)
    assert s['tick_hz'] == 80.0 and len(s['cameras']) == 6
    names = [n for n, _ in simrig.files(rig)]
    assert names == ['sim_rig.json', 'sim_remap.npz']
    # nothing to simulate
    p2 = _project(extra=False)
    p2.variant.cameras.clear()
    p2.variant.sensors.clear()
    with pytest.raises(ValueError):
        simrig.build(p2)


def test_weather_mapping():
    from vantage.core import simrig
    p = _project(extra=False)
    w = simrig.carla_weather(p, simrig.SimSettings.from_dict({'weather': 'HardRainNoon'}))
    assert w['preset'] == 'HardRainNoon' and w['params'] == {}
    fog = simrig.carla_weather(p, simrig.SimSettings.from_dict({'weather': 'vantage:fog'}))['params']
    clear = simrig.carla_weather(p, simrig.SimSettings.from_dict({'weather': 'vantage:clear'}))['params']
    rain = simrig.carla_weather(p, simrig.SimSettings.from_dict({'weather': 'vantage:heavy_rain'}))['params']
    night = simrig.carla_weather(p, simrig.SimSettings.from_dict({'weather': 'vantage:night'}))
    # fog: the density whose fog has the condition's MOR for road users, as measured on CARLA 0.9.16 (150 m → 46.7)
    assert fog['fog_density'] == pytest.approx(simrig.FOG_MOR_M / 150.0, abs=0.06) and fog['fog_distance'] == 0.0
    assert simrig.fog_mor_of(simrig.fog_density_for(150.0)) == pytest.approx(150.0, rel=1e-6)
    assert simrig.fog_density_for(30.0) == simrig.FOG_DENSITY_MAX and simrig.fog_density_for(0.0) == 0.0
    assert clear['fog_density'] <= 5 and clear['precipitation'] == 0
    assert rain['precipitation'] == 100.0 and rain['precipitation_deposits'] == 90.0
    assert night['params']['sun_altitude_angle'] == -90.0 and night['night']
    snow = simrig.carla_weather(p, simrig.SimSettings.from_dict({'weather': 'vantage:snow'}))
    assert any('no snow' in n for n in snow['notes'])


def test_vehicle_match():
    from vantage.core import simrig
    p = _project(extra=False)
    m = simrig.match_vehicle(p)
    sc = [c['score'] for c in m['candidates']]
    assert sc == sorted(sc) and m['blueprint'] == m['candidates'][0]['blueprint'] and m['how'] == 'nearest by size'
    assert simrig.match_vehicle(p, 'vehicle.tesla.model3')['blueprint'] == 'vehicle.tesla.model3'


# ======================================================================= the bridge against the fake carla
def _args(rig, out, **kw):
    from vantage.sim import carla_bridge as CB
    a = CB.parser().parse_args([os.path.join(out, 'rig', 'sim_rig.json'), out])
    for k, v in kw.items():
        setattr(a, k, v)
    return CB.resolve_args(a, rig)


@pytest.fixture(scope='module')
def fake_run(tmp_path_factory):
    """One 1.2 s drive of the sample's data rig (plus an IMU and two GNSS antennas, the lidar on a mast) through the
    bridge against the fake carla."""
    from vantage.core import simrig
    from vantage.sim import carla_bridge as CB
    fc = _fake()
    fc.reset()
    out = str(tmp_path_factory.mktemp('carla_fake') / 'run')
    p = _project(lidar_z=2.0)
    rig = simrig.build(p, simrig.SimSettings.from_dict({'image_scale': 0.1, 'duration_s': 1.2, 'traffic': 6,
                                                        'warmup_s': 0.25, 'seed': 4}))
    simrig.export(rig, p, os.path.join(out, 'rig'))
    doc, remaps = CB.load_rig(os.path.join(out, 'rig', 'sim_rig.json'))
    b = CB.Bridge(fc, doc, remaps, out, _args(doc, out), None, None)
    rep = b.run()
    return {'out': out, 'report': rep, 'project': p, 'rig': doc, 'log': list(fc.LOG), 'world': fc.STATE['world']}


def test_bridge_spawns_records_and_tidies_up(fake_run):
    rep, log, out = fake_run['report'], fake_run['log'], fake_run['out']
    assert rep['status'] == 'ok' and rep['traffic'] == 6
    spawned = {e[1] for e in log if e[0] == 'spawn'}
    destroyed = {e[1] for e in log if e[0] == 'destroy'}
    assert spawned and spawned == destroyed                        # everything spawned is destroyed
    w = fake_run['world']
    assert not w._actors and w.applied[0] == (True, 0.0125) and w.applied[-1] == (False, None)
    assert ('tm_sync', True) in log and log[[i for i, e in enumerate(log) if e[0] == 'tm_sync'][-1]] == ('tm_sync', False)
    assert ('tm_seed', 4) in log
    # the ego: the rig's CARLA vehicle, hero; every sensor attached to it with the rig's transform + the ego origin
    ego = next(e for e in log if e[0] == 'spawn' and e[3].get('role_name') == 'hero')
    assert ego[2] == fake_run['rig']['vehicle']['blueprint']
    origin = rep['vehicle']['ego_origin_in_actor_m']
    lid = next(e for e in log if e[0] == 'spawn' and e[2] == 'sensor.lidar.ray_cast')
    L = fake_run['rig']['lidars'][0]
    assert lid[5] == ego[1] and np.allclose(lid[4][:3], np.add(L['carla']['location'], origin), atol=1e-9)
    assert lid[3]['channels'] == '128' and float(lid[3]['dropoff_general_rate']) == 0.0
    cams = [e for e in log if e[0] == 'spawn' and e[2] == 'sensor.camera.rgb']
    assert len(cams) == sum(len(c['views']) for c in fake_run['rig']['cameras'])
    assert all(float(e[3]['lens_k']) == 0.0 and float(e[3]['motion_blur_intensity']) == 0.0 for e in cams)
    # what was written
    for f in ('run.json', 'dataset.json', 'README.md'):
        assert os.path.isfile(os.path.join(out, f))
    ds = json.load(open(os.path.join(out, 'dataset.json')))
    assert ds['schema'] == 'vantage.dataset/1' and ds['generator'] == 'carla'
    ch = {c['channel'] or c['topic']: c for c in ds['channels']}
    assert ch['LIDAR_L1']['frames'] == 12 and ch['CAM_F']['frames'] == ch['CAM_F']['planned_frames'] == 13
    assert ch['/imu1']['frames'] == 97 and ch['/g1']['frames'] == 13
    assert all(c['frames'] == c['planned_frames'] for c in ds['channels'])


def test_nuscenes_tables_are_consistent(fake_run):
    out = fake_run['out']
    V = os.path.join(out, 'nuscenes', 'v1.0-vantage')
    T = {n: json.load(open(os.path.join(V, n + '.json'))) for n in (
        'attribute', 'calibrated_sensor', 'category', 'ego_pose', 'instance', 'log', 'map', 'sample',
        'sample_annotation', 'sample_data', 'scene', 'sensor', 'visibility')}
    tok = {n: {r['token'] for r in rows} for n, rows in T.items()}
    for r in T['sample_data']:
        assert r['calibrated_sensor_token'] in tok['calibrated_sensor'] and r['ego_pose_token'] in tok['ego_pose']
        assert r['sample_token'] in tok['sample'] and os.path.isfile(os.path.join(out, 'nuscenes', r['filename']))
        assert (r['prev'] == '' or r['prev'] in tok['sample_data']) and (r['next'] == '' or r['next'] in tok['sample_data'])
    for r in T['calibrated_sensor']:
        assert r['sensor_token'] in tok['sensor']
    for r in T['sample_annotation']:
        assert r['instance_token'] in tok['instance'] and r['sample_token'] in tok['sample']
        assert r['visibility_token'] in tok['visibility']
    assert len(T['sample']) == 2 and T['scene'][0]['nbr_samples'] == 2
    keys = [r for r in T['sample_data'] if r['is_key_frame']]
    assert {r['sample_token'] for r in keys} == tok['sample']
    assert os.path.isfile(os.path.join(out, 'nuscenes', T['map'][0]['filename']))
    # the lidar frame is right: keyframe points fall inside the other vehicles' boxes
    assert sum(r['num_lidar_pts'] for r in T['sample_annotation']) > 50


def _q2R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def test_lidar_ground_points_land_on_the_checker_they_hit_in_the_images(fake_run):
    """End to end through independent paths: the fake's lidar (CARLA's ray-cast convention) and its cameras (CARLA's
    pinhole views of a 2 m ground checker), the frames conversion, the remap through Vantage's lens and the nuScenes
    calibration: lidar ground points (the bag's, each moved with the ego pose at its own firing time) projected into
    camera F with cv2.fisheye fall on pixels of the checker colour their world position has. Within 10 m, where a
    square spans several pixels of the 10 % images: camera F sits 0.62 m above the ground (the fake's cars rest on
    it, as CARLA's do), and farther a pixel spans more than a 2 m square along the range."""
    import cv2
    from vantage.core import rosbag as RB
    out = fake_run['out']
    V = os.path.join(out, 'nuscenes', 'v1.0-vantage')
    T = {n: json.load(open(os.path.join(V, n + '.json'))) for n in ('calibrated_sensor', 'ego_pose', 'sample_data',
                                                                    'sensor')}
    off = np.array(json.load(open(os.path.join(out, 'dataset.json')))['scene']['global_offset_m'])
    ego = sorted(T['ego_pose'], key=lambda r: r['timestamp'])
    et = np.array([e['timestamp'] for e in ego], float)
    ex = np.array([e['translation'] for e in ego])
    eyaw = np.unwrap([2 * math.atan2(e['rotation'][3], e['rotation'][0]) for e in ego])
    egos = {e['token']: e for e in ego}
    sens = {r['token']: r['channel'] for r in T['sensor']}
    cal = {r['token']: dict(r, channel=sens[r['sensor_token']]) for r in T['calibrated_sensor']}
    cam = [r for r in T['sample_data'] if cal[r['calibrated_sensor_token']]['channel'] == 'CAM_F']
    cl = next(c for c in cal.values() if c['channel'] == 'LIDAR_L1')
    bag = os.path.join(out, 'rosbag', fake_run['rig']['output']['bag_name'])
    agree = total = near_ok = near = 0
    for m in RB.BagReader(bag).messages(topics=['/l1/points']):
        msg = m[2]
        a = RB.pointcloud2_array(msg)
        stamp_us = RB.stamp_ns(msg['header']['stamp']) / 1000.0
        tus = stamp_us - 1e5 + a['t'].astype(float) * 1e6        # each point's firing time (the sweep ends at the stamp)
        Pe = np.c_[a['x'], a['y'], a['z']].astype(float) @ _q2R(cl['rotation']).T + cl['translation']
        x = np.stack([np.interp(tus, et, ex[:, k]) for k in range(3)], -1)
        yaw = np.interp(tus, et, eyaw)
        c, s_ = np.cos(yaw), np.sin(yaw)
        Pg = np.c_[c * Pe[:, 0] - s_ * Pe[:, 1], s_ * Pe[:, 0] + c * Pe[:, 1], Pe[:, 2]] + x
        Pg = Pg[np.abs(Pg[:, 2]) < 0.05]                           # on the ground
        C = min(cam, key=lambda r: abs(r['timestamp'] - stamp_us))
        cc, ec = cal[C['calibrated_sensor_token']], egos[C['ego_pose_token']]
        Pc = ((Pg - ec['translation']) @ _q2R(ec['rotation']) - cc['translation']) @ _q2R(cc['rotation'])
        rng = np.linalg.norm(Pc, axis=1)
        sel = (Pc[:, 2] > 0.3) & (rng < 10.0)
        Pg, Pc, rng = Pg[sel], Pc[sel], rng[sel]
        if not len(Pc):
            continue
        K = np.array(cc['camera_intrinsic'])
        D = np.array(cc['vantage_distortion']['coefficients'], float)
        uv, _ = cv2.fisheye.projectPoints(Pc.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, D)
        uv = uv.reshape(-1, 2)
        col = cv2.imread(os.path.join(out, 'nuscenes', C['filename']), cv2.IMREAD_COLOR).astype(int)
        img = cv2.cvtColor(col.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        h, w = img.shape
        world = Pg - off                                           # global = mirror(CARLA world) + offset
        xw, yw = world[:, 0], -world[:, 1]
        fx, fy = (xw / 2.0) % 1.0, (yw / 2.0) % 1.0
        # not within 0.3 m of a square's edge: about two pixels of the 10 % images on the grazing ground camera F sees
        # (at 30 % images, 0.2 m: 99.985 % agree within 8 m, so what the band leaves out is pixel size, not geometry)
        away = (np.minimum(fx, 1 - fx) > 0.15) & (np.minimum(fy, 1 - fy) > 0.15)
        inside = (uv[:, 0] > 2) & (uv[:, 1] > 2) & (uv[:, 0] < w - 3) & (uv[:, 1] < h - 3) & away
        par = ((np.floor(xw / 2.0) + np.floor(yw / 2.0)) % 2 == 0)[inside]
        iy, ix = np.round(uv[inside, 1]).astype(int), np.round(uv[inside, 0]).astype(int)
        g = img[iy, ix]
        sat = col[iy, ix].max(1) - col[iy, ix].min(1)
        # a checker pixel (grey), not a car body or the sky (both coloured): the lidar on its mast sees ground the low
        # cameras see a car in front of
        keep = ((g > 130) | (g < 95)) & (sat < 30)
        ok = ((g > 110) == par) & keep
        agree += int(ok.sum())
        total += int(keep.sum())
        nr = rng[inside] < 8.0
        near_ok += int((ok & nr).sum())
        near += int((keep & nr).sum())
    assert total > 500 and agree / total > 0.95, (agree, total)
    assert near > 100 and near_ok / near > 0.98, (near_ok, near)


def test_recording_reads_back_in_the_drive_log_and_as_built_checks(fake_run):
    from vantage.core import asbuilt as AB
    from vantage.core import drivelog as DL
    p = fake_run['project']
    out = fake_run['out']
    bag = os.path.join(out, 'rosbag', fake_run['rig']['output']['bag_name'])
    for path, src in ((bag, 'rosbag'), (os.path.join(out, 'nuscenes'), 'nuscenes')):
        res = DL.run(p, path)
        assert res is not None and res.source == src
        streams = {e['tag']: e for e in res.streams}
        assert streams['F']['topic'] and streams['L1']['topic'] and streams['RD2']['topic']
        assert streams['F']['status'] == 'ok' and streams['RD2']['status'] == 'ok'
        assert abs(streams['L1']['rate_hz'] - 10.0) < 0.01
        assert any('Vantage wrote' in f['message'] for f in res.findings)
        assert res.sync['lidar'] is not None and abs(res.sync['lidar']['phase_error_deg']) < 1.0
    sc = AB.scan_recording(bag, frames=4, min_motion_m=0.0)
    assert sc is not None and sc.source == 'rosbag'
    assert {'/f/image_raw/compressed', '/ml/image_raw/compressed'} <= set(sc.streams)
    assert all(len(s.frames) >= 2 for s in sc.streams.values())
    # the as-built check runs on it; the outline comparison cannot hold (CARLA's car is not the project's body),
    # but its lidar estimate of each camera's rotation is the plan's within half a degree
    from vantage.core.analysis import Engine
    res = AB.run(p, Engine(), AB.AsBuiltSettings(path=bag, frames=6, min_motion_m=0.0))
    s = AB.summary(res)
    assert s['verdict'] in AB.VERDICTS and len(s['per_camera']) == 6
    est = [r['lidar_ypr_deg'] for r in s['per_camera'] if r.get('lidar_ypr_deg')]
    assert est and max(abs(v) for e in est for v in e) < 0.5, est


def test_bridge_cancel_and_error_tidy_up(tmp_path):
    from vantage.core import simrig
    from vantage.sim import carla_bridge as CB
    fc = _fake()
    p = _project(lidar_z=2.0, extra=False)
    rig = simrig.build(p, simrig.SimSettings.from_dict({'image_scale': 0.05, 'duration_s': 1.0, 'traffic': 2,
                                                        'warmup_s': 0.1}))
    for case in ('cancel', 'error'):
        fc.reset()
        out = str(tmp_path / case)
        simrig.export(rig, p, os.path.join(out, 'rig'))
        doc, remaps = CB.load_rig(os.path.join(out, 'rig', 'sim_rig.json'))
        calls = {'n': 0}

        def count():
            calls['n'] += 1
            return calls['n']
        cancel = (lambda: count() > 50) if case == 'cancel' else None
        b = CB.Bridge(fc, doc, remaps, out, _args(doc, out), None, cancel)
        if case == 'error':                                         # the lidar stops delivering mid-run
            orig = b._get

            def broken(key, frame, timeout):
                if key[0] == 'lidar' and count() > 40:
                    raise RuntimeError('No data from lidar/L1')
                return orig(key, frame, timeout)
            b._get = broken
        with pytest.raises(CB.Cancelled if case == 'cancel' else RuntimeError):
            b.run()
        b.rec.cleanup()
        assert not fc.STATE['world']._actors                       # all destroyed
        assert fc.STATE['world'].applied[-1] == (False, None)      # settings restored
        assert not os.path.exists(os.path.join(out, 'nuscenes'))
        assert not os.path.exists(os.path.join(out, 'ground_truth'))  # the ground truth too
        assert not os.listdir(os.path.join(out, 'rosbag')) if os.path.isdir(os.path.join(out, 'rosbag')) else True


def test_bridge_command_line_with_the_fake(tmp_path):
    """python -m vantage.sim.carla_bridge as the user runs it (here with the fake carla on PYTHONPATH)."""
    from vantage.core import simrig
    p = _project(lidar_z=2.0, extra=False)
    p.variant.cameras[:] = p.variant.cameras[:1]
    rig = simrig.build(p, simrig.SimSettings.from_dict({'image_scale': 0.05, 'duration_s': 0.5, 'traffic': 1,
                                                        'warmup_s': 0.1}))
    simrig.export(rig, p, str(tmp_path / 'rig'))
    env = dict(os.environ, PYTHONPATH=FAKE_DIR + os.pathsep + REPO)
    r = subprocess.run([sys.executable, '-m', 'vantage.sim.carla_bridge', str(tmp_path / 'rig' / 'sim_rig.json'),
                        str(tmp_path / 'out'), '--town', 'Town01', '--weather', 'ClearNoon', '--seed', '2',
                        '--progress-json'], env=env, capture_output=True, text=True, timeout=600, cwd=str(tmp_path))
    assert r.returncode == 0, r.stderr[-2000:]
    lines = [json.loads(x) for x in r.stdout.splitlines() if x.startswith('{')]
    assert lines and lines[-1]['progress'] == 1.0
    rep = json.load(open(tmp_path / 'out' / 'run.json'))
    assert rep['status'] == 'ok' and rep['town'] == 'Town01' and rep['weather'] == {'preset': 'ClearNoon'}


# ======================================================================= round 8: ground truth, faults, scenarios, actors
def _mini_run(tmp, settings=None, cams=('F',), radars=(), extra=False, lidar=True, seed=4, **kw):
    """A small drive through the bridge against the fake carla: the cameras, radars and extras asked for, the lidar on
    a mast. Returns {'out', 'report', 'rig', 'log', 'world', 'project'}."""
    from vantage.core import simrig
    from vantage.sim import carla_bridge as CB
    fc = _fake()
    fc.reset()
    p = _project(lidar_z=2.0, extra=extra)
    v = p.variant
    v.cameras[:] = [c for c in v.cameras if c.tag in cams]
    v.sensors[:] = [s for s in v.sensors if (s.kind == 'lidar' and lidar) or s.tag in radars or
                    (extra and s.kind in ('imu', 'gnss'))]
    d = {'image_scale': 0.08, 'duration_s': 1.2, 'traffic': 4, 'warmup_s': 0.25, 'seed': seed}
    d.update(settings or {})
    rig = simrig.build(p, simrig.SimSettings.from_dict(d))
    out = str(tmp)
    simrig.export(rig, p, os.path.join(out, 'rig'))
    doc, remaps = CB.load_rig(os.path.join(out, 'rig', 'sim_rig.json'))
    b = CB.Bridge(fc, doc, remaps, out, _args(doc, out, **kw), None, None)
    rep = b.run()
    return {'out': out, 'report': rep, 'rig': doc, 'log': list(fc.LOG), 'world': fc.STATE['world'], 'project': p,
            'bridge': b}


def _tables(out):
    V = os.path.join(out, 'nuscenes', 'v1.0-vantage')
    return {n[:-5]: json.load(open(os.path.join(V, n))) for n in os.listdir(V) if n.endswith('.json')}


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0


def test_ground_truth_index_masks_objects_and_coco(fake_run):
    """The index read back: every written camera frame has its label images at the image's size; the instance image
    holds exactly the frame's objects (their pixel counts and tight boxes); objects are the spawned vehicles (their
    actor ids), the map's parked car (by its level box), the traffic light's head (the light's actor id) and the sign;
    the instance masks agree with the 3D boxes through the lens (the fake's objects are boxes); the COCO files say the
    same."""
    import cv2
    from vantage.core import groundtruth as GT
    out = fake_run['out']
    d = GT.read(out)
    assert d is not None and d['schema'] == GT.SCHEMA and [c['name'] for c in d['classes']][:3] == ['unlabelled',
                                                                                                     'road', 'sidewalk']
    ds = json.load(open(os.path.join(out, 'dataset.json')))
    frames_of = {c['channel']: c['frames'] for c in ds['channels'] if c['kind'] == 'camera'}
    spawned = {e[1] for e in fake_run['log'] if e[0] == 'spawn' and e[2].startswith('vehicle.')}
    ego = next(e[1] for e in fake_run['log'] if e[0] == 'spawn' and e[3].get('role_name') == 'hero')
    fc = _fake()
    ious, kinds = [], set()
    for tag, cam in d['cameras'].items():
        assert len(cam['frames']) == frames_of[cam['channel']] > 0
        for f in cam['frames'][::3]:
            assert os.path.isfile(os.path.join(out, f['image'])) and f['sample_data_token']
            sem, inst, rng = GT.load_semantic(d, f), GT.load_instance(d, f), GT.load_range(d, f)
            assert sem.shape == inst.shape == rng.shape == (cam['height'], cam['width'])
            assert set(np.unique(sem)) <= set(range(30)) and rng.max() < GT.RANGE_MAX_M
            ids = set(np.unique(inst)) - {0}
            assert ids == {o['id'] for o in f['objects']}
            for o in f['objects']:
                ys, xs = np.nonzero(inst == o['id'])
                assert o['pixels'] == len(ys) and o['bbox'] == [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]
                assert o['actor_id'] != ego
                if o['id'] < 1000:
                    assert o['actor_id'] in spawned and o['class'] == 'car' and o['category'].startswith('vehicle.')
                    kinds.add('vehicle')
                elif o['class'] == 'traffic_light':
                    assert o['actor_id'] == fc.LIGHT_ID and o['category'] == 'static.manmade'
                    kinds.add('light')
                elif o['class'] == 'traffic_sign':
                    assert o['actor_id'] is None
                    kinds.add('sign')
                else:
                    assert o['class'] == 'car' and o['actor_id'] is None      # a parked car of the map
                    kinds.add('parked')
                if o['pixels'] >= 30 and o['occluded'] == 0 and o['truncated'] == 0:
                    ious.append(_iou(o['bbox'], o['bbox_full']))
                b = o['box3d']
                assert len(b['center']) == 3 and len(b['size']) == 3 and o['range_m'] > 0
    assert kinds == {'vehicle', 'light', 'sign', 'parked'}, kinds
    assert len(ious) > 10 and np.median(ious) > 0.9 and min(ious) > 0.7, (len(ious), np.median(ious), min(ious))
    # COCO: one image per frame, an annotation per object, its RLE the object's mask
    coco = json.load(open(os.path.join(out, 'ground_truth', 'coco', 'CAM_F.json')))
    fr = d['cameras']['F']['frames']
    assert len(coco['images']) == len(fr) and len(coco['annotations']) == sum(len(f['objects']) for f in fr)
    cats = {c['id']: c['name'] for c in coco['categories']}
    k = 0
    for i, f in enumerate(fr):
        inst = GT.load_instance(d, f) if f['objects'] else None
        for o in f['objects']:
            a = coco['annotations'][k]
            k += 1
            x0, y0, x1, y1 = o['bbox']
            assert a['image_id'] == i + 1 and a['bbox'] == [x0, y0, x1 - x0, y1 - y0] and cats[a['category_id']] == \
                o['class'] and a['area'] == o['pixels'] and a['actor_id'] == o['actor_id']
            assert np.array_equal(GT.rle_decode(a['segmentation']), inst == o['id'])
    del cv2


def test_range_image_matches_the_lidar(fake_run):
    """The range images against the lidar, two independent paths: lidar points on the static ground (labelled so by
    lidarseg), each moved into the global frame with the ego pose at its own firing time (the bag's t), then into
    camera F at its nearest frame's time through the nuScenes poses and projected through the rig's lens: the range
    PNG at their pixel is the distance along that pixel's ray to the ground through the lidar point."""
    from vantage.core import groundtruth as GT
    from vantage.core import rosbag as RB
    out = fake_run['out']
    d = GT.read(out)
    T = _tables(out)
    sens = {r['token']: r['channel'] for r in T['sensor']}
    cal = {r['token']: dict(r, channel=sens[r['sensor_token']]) for r in T['calibrated_sensor']}
    egos = {e['token']: e for e in T['ego_pose']}
    ego = sorted(T['ego_pose'], key=lambda r: r['timestamp'])
    et = np.array([e['timestamp'] for e in ego], float)
    ex = np.array([e['translation'] for e in ego])
    eyaw = np.unwrap([2 * math.atan2(e['rotation'][3], e['rotation'][0]) for e in ego])
    sd = {r['token']: r for r in T['sample_data']}
    cl = next(c for c in cal.values() if c['channel'] == 'LIDAR_L1')
    cam = next(c for c in fake_run['rig']['cameras'] if c['tag'] == 'F')
    lens = cam['lens']
    from vantage.sim import carla_bridge as CB
    _, remaps = CB.load_rig(os.path.join(out, 'rig', 'sim_rig.json'))
    cg = GT.CameraGT(cam, remaps)                                  # each pixel's ray (the remap's own geometry)
    frames = d['cameras']['F']['frames']
    sweeps = {sd[s['sample_data_token']]['timestamp']: s for s in d['lidars']['L1']['sweeps']}
    bag = os.path.join(out, 'rosbag', fake_run['rig']['output']['bag_name'])
    noise = fake_run['rig']['output']['lidar_noise_cm'] / 100.0
    errs, zs = [], []
    for m in RB.BagReader(bag).messages(topics=['/l1/points']):
        a = RB.pointcloud2_array(m[2])
        stamp_us = RB.stamp_ns(m[2]['header']['stamp']) // 1000
        lab = GT.load_lidarseg(d, sweeps[stamp_us])
        assert len(lab) == len(a)
        keep = np.isin(lab, [GT.NUSC_INDEX['flat.driveable_surface'], GT.NUSC_INDEX['flat.sidewalk']])
        tus = stamp_us - 1e5 + a['t'][keep].astype(float) * 1e6     # each point's firing time
        Pe = np.c_[a['x'], a['y'], a['z']][keep].astype(float) @ _q2R(cl['rotation']).T + cl['translation']
        x = np.stack([np.interp(tus, et, ex[:, k]) for k in range(3)], -1)
        yaw = np.interp(tus, et, eyaw)
        c, s_ = np.cos(yaw), np.sin(yaw)
        Pg = np.c_[c * Pe[:, 0] - s_ * Pe[:, 1], s_ * Pe[:, 0] + c * Pe[:, 1], Pe[:, 2]] + x
        f = min(frames, key=lambda f: abs(f['timestamp'] - stamp_us))
        C = sd[f['sample_data_token']]
        cc, ec = cal[C['calibrated_sensor_token']], egos[C['ego_pose_token']]
        Pc = ((Pg - ec['translation']) @ _q2R(ec['rotation']) - cc['translation']) @ _q2R(cc['rotation'])
        u, v, _, ok = GT.project(lens, Pc)
        rng = GT.load_range(d, f)
        h, w = rng.shape
        ok &= (u >= 1) & (v >= 1) & (u < w - 1) & (v < h - 1) & (np.linalg.norm(Pc, axis=1) < 20.0)
        sem = GT.load_semantic(d, f)
        iu, iv = u[ok].astype(int), v[ok].astype(int)
        ground = np.isin(sem[iv, iu], [1, 2]) & (rng[iv, iu] > 0)
        # the lidar point's ground plane (its normal: global z) along the pixel's own ray: the range there
        R_gc = _q2R(ec['rotation']) @ _q2R(cc['rotation'])          # camera optical -> global
        n = R_gc.T @ np.array([0.0, 0.0, 1.0])
        dpx = cg.rays[iv, iu].astype(float)
        r_exp = (Pc[ok] @ n) / (dpx @ n)
        # the lidar's range noise moves the point along its own ray: its height by σ · u_z, the range along the
        # camera's grazing ray by that / (n · d)
        L_g = np.c_[c * cl['translation'][0] - s_ * cl['translation'][1],
                    s_ * cl['translation'][0] + c * cl['translation'][1], np.full(len(c), cl['translation'][2])] + x
        u_l = Pg - L_g
        u_l /= np.linalg.norm(u_l, axis=1, keepdims=True)
        sig = noise * np.abs(u_l[ok, 2]) / np.abs(dpx @ n) + 1.0 / GT.RANGE_SCALE
        e = (rng[iv, iu] - r_exp)[ground]
        errs.append(e)
        zs.append(e / sig[ground])
    e, z = np.concatenate(errs), np.concatenate(zs)
    # no bias, and the scatter the lidar's noise predicts (|z| of a normal distribution: median 0.67, 90 % 1.64)
    assert len(e) > 2000 and abs(np.median(e)) < 0.003, (len(e), np.median(e))
    assert np.median(np.abs(z)) < 0.9 and np.percentile(np.abs(z), 90) < 2.0, \
        (np.median(np.abs(z)), np.percentile(np.abs(z), 90))


def test_lidarseg_and_panoptic_in_the_official_layout(fake_run):
    """nuScenes-lidarseg and -panoptic as the devkit reads them (lidarseg.json / panoptic.json rows token =
    sample_data_token, <token>_lidarseg.bin uint8 per point, <token>_panoptic.npz 'data' uint16 = index · 1000 +
    instance, category.json with 'index'), and right: ground points are ground, the points in the annotated vehicles'
    boxes are vehicles with their instance, the parked car's points vehicle.car with instance 500 + its index."""
    from vantage.core import groundtruth as GT
    out = fake_run['out']
    T = _tables(out)
    sd = {r['token']: r for r in T['sample_data']}
    sens = {r['token']: r['channel'] for r in T['sensor']}
    cal = {r['token']: sens[r['sensor_token']] for r in T['calibrated_sensor']}
    lidar_sd = [r for r in T['sample_data'] if cal[r['calibrated_sensor_token']] == 'LIDAR_L1']
    assert {r['sample_data_token'] for r in T['lidarseg']} == {r['token'] for r in lidar_sd}
    cats = sorted(T['category'], key=lambda r: r['index'])
    assert [c['index'] for c in cats][:32] == list(range(32)) and [c['name'] for c in cats][:32] == \
        [n for _, n, _ in GT.NUSCENES_LIDARSEG]
    assert {r['category_token'] for r in T['instance']} <= {c['token'] for c in cats}
    ann = {}
    for a in T['sample_annotation']:
        ann.setdefault(a['sample_token'], []).append(a)
    inst_cat = {r['token']: next(c['name'] for c in cats if c['token'] == r['category_token']) for r in T['instance']}
    vehicle_ok = vehicle_n = ground_ok = ground_n = 0
    for row, prow in zip(T['lidarseg'], T['panoptic']):
        assert row['token'] == row['sample_data_token'] == prow['sample_data_token']
        assert row['filename'] == f'lidarseg/v1.0-vantage/{row["token"]}_lidarseg.bin'
        assert prow['filename'] == f'panoptic/v1.0-vantage/{prow["token"]}_panoptic.npz'
        r = sd[row['sample_data_token']]
        pts = np.fromfile(os.path.join(out, 'nuscenes', r['filename']), np.float32).reshape(-1, 5)
        lab = np.fromfile(os.path.join(out, 'nuscenes', row['filename']), np.uint8)
        with np.load(os.path.join(out, 'nuscenes', prow['filename'])) as z:
            assert z.files == ['data'] and z['data'].dtype == np.uint16
            pan = z['data']
        assert len(lab) == len(pts) == len(pan) and np.array_equal(pan // 1000, lab)
        inst = pan % 1000
        assert np.all(inst[~np.isin(lab, list(GT.THING_INDICES))] == 0)
        ground = pts[:, 2] < -2.0 + 0.05                       # the mast lidar 2 m above the ground (sensor frame)
        ground_ok += int(np.isin(lab[ground], [24, 26]).sum())
        ground_n += int(ground.sum())
        if not r['is_key_frame']:
            continue
        # annotated vehicle boxes at the keyframe: their points are vehicles, one instance per box
        cl = next(c for c in T['calibrated_sensor'] if c['token'] == r['calibrated_sensor_token'])
        ep = next(e for e in T['ego_pose'] if e['token'] == r['ego_pose_token'])
        Pg = (pts[:, :3] @ _q2R(cl['rotation']).T + cl['translation']) @ _q2R(ep['rotation']).T + ep['translation']
        for a in ann.get(r['sample_token'], []):
            W, Lg, H = a['size']
            q = (Pg - a['translation']) @ _q2R(a['rotation'])
            # on its surface (the box grown 5 cm), not the ground under it
            m = (np.abs(q[:, 0]) < Lg / 2 + 0.05) & (np.abs(q[:, 1]) < W / 2 + 0.05) & (q[:, 2] > -H / 2 + 0.1) & \
                (q[:, 2] < H / 2 + 0.05)
            if m.sum() < 5:
                continue
            want = inst_cat[a['instance_token']]
            vehicle_ok += int((lab[m] == GT.NUSC_INDEX[want]).sum())
            vehicle_n += int(m.sum())
            ids = set(inst[m][lab[m] == GT.NUSC_INDEX[want]].tolist())
            assert len(ids) == 1 and ids.pop() > 0                 # one instance per box
    assert ground_n > 1000 and ground_ok / ground_n > 0.99
    assert vehicle_n > 50 and vehicle_ok / vehicle_n > 0.99, (vehicle_ok, vehicle_n)
    d = GT.read(out)
    assert [s['sample_data_token'] for s in d['lidars']['L1']['sweeps']] == [r['token'] for r in T['lidarseg']]
    rep = fake_run['report']['ground_truth']['lidar_labels']['L1']
    assert rep['matched_in_order_share'] == 1.0 and rep['max_direction_error_deg'] < 1e-4 and \
        abs(rep['range_rms_diff_m'] - 0.015) < 0.002                # the ray-cast lidar's noise only


def test_faults_change_what_carla_records_and_the_calibration_states_the_plan(tmp_path):
    """Every fault kind on a small rig: the camera's views spawned at its rotated mount (the labels follow the image),
    the radar moved; the calibration the plan's; the camera stamped 5 ms late with its ego pose at the stamp; lidar
    sweeps 2 and 5 never written and its ranges 0.3 m long; the radar frozen (its scans repeat the last one before);
    half the IMU's samples dropped; run.json says each as applied and README.md says the drive is faulty."""
    from vantage.core import groundtruth as GT
    from vantage.sim import frames as FR
    from vantage.sim.carla_bridge import pose_matrix
    faults = [{'tag': 'F', 'kind': 'rotation', 'deg': [0.0, 2.0, 0.0]}, {'tag': 'F', 'kind': 'clock_offset', 'ms': 5},
              {'tag': 'L1', 'kind': 'drop', 'frames': [2, 5]}, {'tag': 'L1', 'kind': 'range_bias', 'm': 0.3},
              {'tag': 'RD1', 'kind': 'freeze', 'from_s': 0.3, 'to_s': 0.7},
              {'tag': 'RD1', 'kind': 'translation', 'm': [0.1, 0.0, 0.0]}, {'tag': 'IMU1', 'kind': 'drop', 'rate': 0.5}]
    r = _mini_run(tmp_path / 'f', {'faults': faults}, radars=('RD1',), extra=True)
    out, rep, rig = r['out'], r['report'], r['rig']
    fi = {(f['tag'], f['kind']): f for f in rep['faults_injected']}
    assert len(fi) == 7
    rot = fi[('F', 'rotation')]
    assert rot['pose_delta_deg'] == [0.0, 2.0, 0.0] and rot['angle_deg'] == pytest.approx(2.0)
    assert rot['actual_pose']['pitch'] == pytest.approx(rot['plan_pose']['pitch'] + 2.0)
    # the views CARLA spawned: the planned views turned by R_plan · R(fault) · R_plan⁻¹
    cam = next(c for c in rig['cameras'] if c['tag'] == 'F')
    Rp, Ra = pose_matrix(cam['pose_vehicle']), pose_matrix(rot['actual_pose'])
    views = [e for e in r['log'] if e[0] == 'spawn' and e[2] == 'sensor.camera.rgb']
    for e, v in zip(views, cam['views']):
        got = FR.vantage_matrix(e[4][3], e[4][4], e[4][5])
        rr = v['carla']['rotation']
        assert np.allclose(got, Ra @ Rp.T @ FR.vantage_matrix(rr['pitch'], rr['yaw'], rr['roll']), atol=1e-6)
    # the calibration states the plan: the rig's own rows, unchanged
    T = _tables(out)
    assert sorted(T['calibrated_sensor'], key=lambda x: x['token']) == \
        sorted([x for x in rig['nuscenes']['calibrated_sensor'] if x['token'] in
                {c['token'] for c in T['calibrated_sensor']}], key=lambda x: x['token'])
    # the labels follow the image: objects' masks still match their boxes through the lens at the ACTUAL mount
    d = GT.read(out)
    ious = [_iou(o['bbox'], o['bbox_full']) for f in d['cameras']['F']['frames'] for o in f['objects']
            if o['pixels'] >= 20 and o['occluded'] == 0 and o['truncated'] == 0]
    assert ious and np.median(ious) > 0.85
    # stamps: the camera's 5 ms late, its ego pose the one at its stamp (between the steps, interpolated)
    sens = {s['token']: s['channel'] for s in T['sensor']}
    cal = {c['token']: sens[c['sensor_token']] for c in T['calibrated_sensor']}
    by_ch = {}
    for s in T['sample_data']:
        by_ch.setdefault(cal[s['calibrated_sensor_token']], []).append(s)
    cam_c = r['bridge'].P['chans']
    plan_cam = next(c for c in cam_c if c.kind == 'camera')
    t_cam = sorted(s['timestamp'] for s in by_ch['CAM_F'])
    epoch = rig['output']['epoch_us']
    assert t_cam == [epoch + int(round(t * 1e6)) + 5000 for t in plan_cam.stamps]
    from vantage.core import rosbag as RB
    bag = os.path.join(out, 'rosbag', rig['output']['bag_name'])
    odom = [(RB.stamp_ns(m[2]['header']['stamp']) / 1e3, m[2]['pose']['pose']['position'])
            for m in RB.BagReader(bag).messages(topics=['/odom'])]
    ot = np.array([o[0] for o in odom])
    ox = np.array([[o[1]['x'], o[1]['y']] for o in odom])
    ego = {e['token']: e for e in T['ego_pose']}
    for s in by_ch['CAM_F']:
        tr = ego[s['ego_pose_token']]['translation']
        want = [np.interp(s['timestamp'], ot, ox[:, k]) for k in range(2)]
        assert np.allclose(tr[:2], want, atol=0.03)                # (odometry at 50 Hz: within its interpolation)
    # drops: L1's sweeps 2 and 5 never written; the IMU about half its samples
    plan_l = next(c for c in cam_c if c.kind == 'lidar')
    t_l = sorted(s['timestamp'] for s in by_ch['LIDAR_L1'])
    assert t_l == [epoch + int(round(t * 1e6)) for k, t in enumerate(plan_l.stamps) if k not in (2, 5)]
    ds = json.load(open(os.path.join(out, 'dataset.json')))
    ch = {c['channel'] or c['topic']: c for c in ds['channels']}
    assert ch['LIDAR_L1']['frames'] == ch['LIDAR_L1']['planned_frames'] - 2
    assert 0.3 < ch['/imu1']['frames'] / ch['/imu1']['planned_frames'] < 0.7
    assert fi[('IMU1', 'drop')]['dropped_frames'] and fi[('L1', 'drop')]['dropped_frames'] == [2, 5]
    # the range bias: ground points (sensor 2 m above the ground) lie (r + 0.3) / r further along their rays
    pts = []
    for s in by_ch['LIDAR_L1']:
        pts.append(np.fromfile(os.path.join(out, 'nuscenes', s['filename']), np.float32).reshape(-1, 5)[:, :3])
    P = np.concatenate(pts).astype(float)
    rr = np.linalg.norm(P, axis=1)
    g = (P[:, 2] < -1.9) & (P[:, 2] > -2.6) & (rr < 15)
    bias = rr[g] * (1.0 - 2.0 / -P[g, 2])                          # z = −2 r' / r, r' = r + b  ->  b = r' (1 − 2/|z|)
    assert abs(np.median(bias) - 0.3) < 0.02, np.median(bias)
    # the freeze: RD1's scans in the window are the scan just before it, byte for byte
    rd = sorted(by_ch['RADAR_RD1'], key=lambda s: s['timestamp'])
    data = [open(os.path.join(out, 'nuscenes', s['filename']), 'rb').read() for s in rd]
    fz = fi[('RD1', 'freeze')]
    assert fz['frozen_frames'] and all(data[k] == data[fz['held_frame']] for k in fz['frozen_frames'])
    assert data[fz['frozen_frames'][-1] + 1] != data[fz['held_frame']]
    assert fi[('RD1', 'translation')]['position_delta_m'] == [0.1, 0.0, 0.0]
    assert 'deliberately faulty' in open(os.path.join(out, 'README.md')).read()
    assert ds['faults_injected'] == rep['faults_injected']


def test_scenarios_play_out_deterministically(tmp_path):
    """Each built-in scenario on the fake: its actors where it puts them, its events when they happen; the same seed
    gives the same drive; the ground truth names the scenario's actors (a walker is a pedestrian, a cyclist a
    bicycle with its rider)."""
    from vantage.core import groundtruth as GT
    runs = {}
    for name, params in (('lead_vehicle_brake', {'brake_at_s': 0.4, 'gap_m': 12.0}),
                         ('cut_in', {'cut_at_s': 0.2, 'cutter_offset_m': 6.0, 'cut_gap_m': 0.0}),
                         ('pedestrian_crossing', {'distance_m': 10.0, 'walk_speed_mps': 3.0, 'start_at_s': 0.0,
                                                  'walker': 'child'}),
                         ('cyclist_crossing', {'distance_m': 10.0, 'start_at_s': 0.0}),
                         ('parked', {})):
        runs[name] = _mini_run(tmp_path / name, {'scenario': {'name': name, 'params': params},
                                                 'duration_s': 4.0 if name == 'cut_in' else 2.5, 'traffic': 3})
    rep = runs['lead_vehicle_brake']['report']['scenario']
    ev = {e['what']: e for e in rep['events']}
    assert rep['actors'][0]['role'] == 'lead' and ev['lead_brake']['t_s'] == pytest.approx(0.4, abs=0.0125)
    assert 11.0 < ev['lead_brake']['gap_m'] < 13.0 and ev['lead_stopped']['t_s'] > 0.4
    assert rep['start']['straight_ahead_m'] > 30 and runs['lead_vehicle_brake']['report']['traffic'] == 0
    rep = runs['cut_in']['report']['scenario']
    ev = {e['what']: e for e in rep['events']}
    assert ev['lane_change_ordered']['t_s'] == pytest.approx(0.2, abs=0.0125)
    assert abs(ev['lane_change_ordered']['lateral_m']) == pytest.approx(3.5, abs=0.2)
    assert ev['cut_in']['t_s'] > 0.2 and ev['cut_in']['longitudinal_m'] > 0
    assert ev['cut_in_complete']['t_s'] > ev['cut_in']['t_s'] and abs(ev['cut_in_complete']['lateral_m']) < 0.3
    for name, cls, cat in (('pedestrian_crossing', 'pedestrian', 'human.pedestrian.child'),
                           ('cyclist_crossing', 'bicycle', 'vehicle.bicycle')):
        rep = runs[name]['report']['scenario']
        ev = {e['what']: e for e in rep['events']}
        assert ev['start']['t_s'] == 0.0 and ev['enters_ego_lane']['t_s'] > 0.0
        aid = rep['actors'][0]['actor_id']
        d = GT.read(runs[name]['out'])
        objs = [o for f in d['cameras']['F']['frames'] for o in f['objects'] if o['actor_id'] == aid]
        assert objs and {o['class'] for o in objs} == {cls} and {o['category'] for o in objs} == {cat}
        T = _tables(runs[name]['out'])
        assert cat in {c['name'] for c in T['category'] if c['token'] in {i['category_token'] for i in T['instance']}}
    # parked: the ego stands still, no traffic, no actors
    T = _tables(runs['parked']['out'])
    xy = np.array([e['translation'][:2] for e in T['ego_pose']])
    assert np.ptp(xy, axis=0).max() < 1e-6 and runs['parked']['report']['traffic'] == 0
    # the same seed, the same drive
    again = _mini_run(tmp_path / 'again', {'scenario': {'name': 'lead_vehicle_brake', 'params': {
        'brake_at_s': 0.4, 'gap_m': 12.0}}, 'duration_s': 2.5, 'traffic': 3})
    a, b = runs['lead_vehicle_brake'], again
    assert a['report']['scenario']['events'] == b['report']['scenario']['events']
    assert _tables(a['out'])['ego_pose'] == _tables(b['out'])['ego_pose']


def test_walkers_on_the_sidewalks(tmp_path):
    """Pedestrians on CARLA's walker AI in a free drive: as many as asked, walking (their AI started, then stopped
    before anything is destroyed), annotated and labelled as pedestrians; the same seed puts them in the same places."""
    from vantage.core import groundtruth as GT
    runs = [_mini_run(tmp_path / k, {'walkers': 4, 'traffic': 2}, cams=('F', 'R', 'ML', 'MR')) for k in ('a', 'b')]
    r = runs[0]
    assert r['report']['walkers'] == 4 and r['report']['traffic'] == 2
    log = r['log']
    ctrl = [e[1] for e in log if e[0] == 'spawn' and e[2] == 'controller.ai.walker']
    started = [e[1] for e in log if e[0] == 'walker_ai_start']
    stopped = [i for i, e in enumerate(log) if e[0] == 'walker_ai_stop']
    destroyed = [i for i, e in enumerate(log) if e[0] == 'destroy']
    assert len(ctrl) == 4 and sorted(started) == sorted(ctrl) and max(stopped) < min(destroyed)
    assert ('pedestrians_seed', 4) in log
    T = _tables(r['out'])
    cats = {c['token']: c['name'] for c in T['category']}
    peds = {i['token'] for i in T['instance'] if cats[i['category_token']].startswith('human.pedestrian')}
    rows = [a for a in T['sample_annotation'] if a['instance_token'] in peds]
    from vantage.sim.record import token
    ped_attr = {token('vantage', 'attribute', n) for n in ('pedestrian.moving', 'pedestrian.standing')}
    assert rows and all(len(a['attribute_tokens']) == 1 and a['attribute_tokens'][0] in ped_attr for a in rows)
    assert any(a['attribute_tokens'][0] == token('vantage', 'attribute', 'pedestrian.moving') for a in rows)
    d = GT.read(r['out'])
    classes = {o['class'] for c in d['cameras'].values() for f in c['frames'] for o in f['objects']
               if o['category'].startswith('human.pedestrian')}
    assert classes == {'pedestrian'}
    other = _tables(runs[1]['out'])
    assert [a['translation'] for a in T['sample_annotation'] if a['instance_token'] in peds] == \
        [a['translation'] for a in other['sample_annotation'] if a['instance_token'] in peds]


def test_actors_scene_places_every_actor_and_labels_it(tmp_path):
    """ACTORS.json around a still ego at a given spawn point: every actor where its pose says (box bottom centre and
    heading in the ego's vehicle frame), a traffic light reported as not spawnable, no traffic; the ground truth names
    each actor by its actor id with its category."""
    from vantage.core import groundtruth as GT
    actors = [{'id': 'kid', 'kind': 'child', 'pose': {'x': 7.0, 'y': 0.8, 'yaw': 90.0}},
              {'id': 'man', 'kind': 'pedestrian', 'pose': {'x': -6.0, 'y': -1.0, 'yaw': 0.0}},
              {'id': 'bike', 'kind': 'cyclist', 'pose': {'x': 4.0, 'y': 3.0, 'yaw': -90.0}},
              {'id': 'car', 'kind': 'car', 'pose': {'x': 12.0, 'y': -0.5, 'yaw': 180.0}},
              {'id': 'cone', 'kind': 'cone', 'pose': {'x': 5.0, 'y': -2.0}},
              {'id': 'bar', 'kind': 'barrier', 'pose': {'x': 9.0, 'y': 2.5, 'yaw': 30.0}},
              {'id': 'sign', 'kind': 'sign', 'pose': {'x': 3.0, 'y': -4.0, 'yaw': 90.0}},
              {'id': 'tl', 'kind': 'traffic_light', 'pose': {'x': 15.0, 'y': 4.0}},
              {'id': 'a2', 'kind': 'blueprint:vehicle.audi.a2', 'pose': {'x': -12.0, 'y': 0.0, 'z': 0.0, 'yaw': 0.0}},
              {'id': 'mini', 'kind': 'blueprint:vehicle.mini.cooper_s', 'pose': {'x': -20.0, 'y': 2.0, 'yaw': 0.0}}]
    r = _mini_run(tmp_path / 'scene', {'actors': actors, 'spawn_point': 3, 'traffic': 5},
                  cams=('F', 'R', 'ML', 'MR'))
    rep = r['report']
    assert rep['traffic'] == 0 and rep['start']['spawn_point'] == 3 and rep['scenario']['name'] == 'actors'
    rows = {x['id']: x for x in rep['actors']}
    assert rows['tl']['actor_id'] is None and 'traffic-light' in rows['tl']['error']
    for a in actors:
        if a['kind'] == 'traffic_light':
            continue
        row = rows[a['id']]
        assert row['actor_id'] is not None and row['error_m'] < 1e-3 and abs(row['error_z_m']) < 1e-3
        assert abs(row['error_yaw_deg']) < 1e-3, row
    assert rows['kid']['blueprint'] in ('walker.pedestrian.0009', 'walker.pedestrian.0010')
    assert rows['a2']['blueprint'] == 'vehicle.audi.a2' and rows['cone']['blueprint'] == 'static.prop.trafficcone01'
    assert rows['mini']['blueprint'] == 'vehicle.mini.cooper_s' and rows['mini']['error_m'] < 1e-3
    # a turned prop: its own size, not the world-axes box CARLA 0.9.16 reports for it
    assert rows['sign']['bbox']['size_lwh'] == [1.064, 0.128, 2.16]
    T = _tables(r['out'])
    xy = np.array([e['translation'][:2] for e in T['ego_pose']])
    assert np.ptp(xy, axis=0).max() < 1e-6                         # the ego stands still
    d = GT.read(r['out'])
    seen = {}
    for c in d['cameras'].values():
        for f in c['frames']:
            for o in f['objects']:
                if o['actor_id'] is not None:
                    seen.setdefault(o['actor_id'], set()).add((o['class'], o['category']))
    want = {'kid': ('pedestrian', 'human.pedestrian.child'), 'man': ('pedestrian', 'human.pedestrian.adult'),
            'bike': ('bicycle', 'vehicle.bicycle'), 'car': ('car', 'vehicle.car'),
            'cone': ('dynamic', 'movable_object.trafficcone'), 'bar': ('static', 'movable_object.barrier'),
            'sign': ('traffic_sign', 'static.manmade'), 'a2': ('car', 'vehicle.car')}
    for k, (cls, cat) in want.items():
        assert (cls, cat) in seen.get(rows[k]['actor_id'], set()), (k, seen.get(rows[k]['actor_id']))
    sizes = {tuple(o['box3d']['size']) for c in d['cameras'].values() for f in c['frames'] for o in f['objects']
             if o['actor_id'] == rows['sign']['actor_id']}
    assert sizes == {(1.064, 0.128, 2.16)}                        # the turned sign's own box in the ground truth
    cats = {c['token']: c['name'] for c in T['category']}
    annotated = {cats[i['category_token']] for i in T['instance']}
    assert {'human.pedestrian.child', 'human.pedestrian.adult', 'vehicle.bicycle', 'movable_object.trafficcone',
            'movable_object.barrier'} <= annotated and 'static.manmade' not in annotated


def test_the_bridge_never_hangs_on_a_dead_server(tmp_path, monkeypatch):
    """A server that stops answering mid-run (the fake's tick never returns, as a crashed server's client call): the
    watchdog ends the bridge after twice --timeout with run.json saying so (exit 4) and what was written removed; a
    SIGTERM while it hangs ends it within the grace (exit 3); a run over the render budget is refused before it
    starts (exit 2)."""
    import signal as SG
    from vantage.core import simrig
    from vantage.sim import carla_bridge as CB
    p = _project(lidar_z=2.0, extra=False)
    p.variant.cameras[:] = p.variant.cameras[:1]
    rig = simrig.build(p, simrig.SimSettings.from_dict({'image_scale': 0.05, 'duration_s': 3.0, 'traffic': 1,
                                                        'warmup_s': 0.1}))
    simrig.export(rig, p, str(tmp_path / 'rig'))
    base = [sys.executable, '-m', 'vantage.sim.carla_bridge', str(tmp_path / 'rig' / 'sim_rig.json')]
    env = dict(os.environ, PYTHONPATH=FAKE_DIR + os.pathsep + REPO, FAKE_CARLA_HANG_AT_TICK='60')
    t0 = time.monotonic()
    r = subprocess.run(base + [str(tmp_path / 'hang'), '--timeout', '2'], env=env, capture_output=True, text=True,
                       timeout=120, cwd=str(tmp_path))
    took = time.monotonic() - t0
    rep = json.load(open(tmp_path / 'hang' / 'run.json'))
    assert r.returncode == 4 and rep['status'] == 'failed' and 'stopped answering' in rep['error'] and took < 40
    assert rep['server_stopped'] and not (tmp_path / 'hang' / 'nuscenes').exists()
    pr = subprocess.Popen(base + [str(tmp_path / 'sig'), '--timeout', '60'], env=env, stdout=subprocess.DEVNULL,
                          stderr=subprocess.PIPE, text=True, cwd=str(tmp_path))
    end = time.monotonic() + 60
    while time.monotonic() < end and not (tmp_path / 'sig' / 'nuscenes').exists():
        time.sleep(0.2)
    time.sleep(3.0)                                               # (hung at its 60th step by now)
    t0 = time.monotonic()
    pr.send_signal(SG.SIGTERM)
    code = pr.wait(timeout=60)
    assert code == 3 and time.monotonic() - t0 < CB.SIGNAL_GRACE_S + 10
    assert json.load(open(tmp_path / 'sig' / 'run.json'))['status'] == 'cancelled'
    r = subprocess.run(base + [str(tmp_path / 'big'), '--render-budget-mpx', '0.001'], env=env, capture_output=True,
                       text=True, timeout=120, cwd=str(tmp_path))
    assert r.returncode == 2 and 'over the budget' in r.stderr and not (tmp_path / 'big').exists()
    monkeypatch.setattr(CB, 'RENDER_BUDGET_MPX', 0.001)          # the plan says so before the run is started
    st = simrig.SimSettings.from_dict({'image_scale': 0.05, 'duration_s': 3.0, 'traffic': 1})
    assert any('The bridge refuses this run' in n for n in simrig.build(p, st).notes)


def test_tidying_up_stops_at_a_dead_server():
    """A server that died mid-run times every call out (120 s each): the bridge's tidy-up asks once, then stops."""
    from vantage.sim import carla_bridge as CB
    calls = []

    class Dead:
        type_id = 'vehicle.audi.a2'

        def __getattr__(self, name):
            def call(*_a):
                calls.append(name)
                raise RuntimeError('time-out of 120000ms while waiting for the simulator')
            return call

    class World(Dead):
        def get_settings(self):
            calls.append('get_settings')
            raise RuntimeError('time-out of 120000ms while waiting for the simulator')
    b = CB.Bridge.__new__(CB.Bridge)
    b.a = type('A', (), {'tm_port': 8200})()
    b.actors = [Dead() for _ in range(16)]
    b.sensors = [(Dead(), 'camera', {}, 0) for _ in range(30)]
    b.tm, b.original = None, object()
    b._tidy(World())
    assert calls == ['get_settings']


def test_tidy_after_a_killed_bridge(tmp_path):
    """A bridge killed mid-run (no tidying) leaves the server synchronous with its actors; tidy_server (--tidy)
    destroys what bridges spawned, by their role names, and nothing else, and puts the world back in asynchronous
    mode."""
    from vantage.core import simrig
    from vantage.sim import carla_bridge as CB
    fc = _fake()
    fc.reset()
    p = _project(lidar_z=2.0, extra=False)
    p.variant.cameras[:] = p.variant.cameras[:1]
    rig = simrig.build(p, simrig.SimSettings.from_dict({'image_scale': 0.05, 'duration_s': 1.0, 'traffic': 3,
                                                        'walkers': 2, 'warmup_s': 0.1,
                                                        'scenario': {'name': 'cyclist_crossing'}}))
    out = str(tmp_path / 'killed')
    simrig.export(rig, p, os.path.join(out, 'rig'))
    doc, remaps = CB.load_rig(os.path.join(out, 'rig', 'sim_rig.json'))
    b = CB.Bridge(fc, doc, remaps, out, _args(doc, out), None, None)
    b._tidy = lambda world: None                                  # killed: nothing tidied

    def killed(*_a, **_k):
        raise SystemExit('killed')
    b._loop = killed
    with pytest.raises(SystemExit):
        b.run()
    w = fc.STATE['world']
    lib = w.get_blueprint_library()
    bp = lib.find('vehicle.tesla.model3')
    bp.set_attribute('role_name', 'hero')                         # another client's car and sensor on the server
    theirs = w.spawn_actor(bp, fc.Transform(fc.Location(120.0, 0.0, 0.3), fc.Rotation()))
    cam = w.spawn_actor(lib.find('sensor.camera.rgb'), fc.Transform(), attach_to=theirs)
    ours = [a for a in w._actors.values() if CB._role(a).startswith('vantage_')]
    assert w.get_settings().synchronous_mode and len(ours) > 10
    res = CB.tidy_server(fc, '127.0.0.1', 2100)
    assert set(w._actors) == {theirs.id, cam.id} and not w.get_settings().synchronous_mode
    n_veh = len([e for e in fc.LOG if e[0] == 'spawn' and e[2].startswith('vehicle.')]) - 1      # (not theirs)
    assert res['was_synchronous'] and res['walkers'] == 2 and res['controllers'] == 2 and res['vehicles'] == n_veh == 2
    assert res['sensors'] == len([e for e in fc.LOG if e[0] == 'spawn' and e[2].startswith('sensor.')]) - 1
    # the command line: --tidy needs a port; with one it says what it did
    env = dict(os.environ, PYTHONPATH=FAKE_DIR + os.pathsep + REPO)
    base = [sys.executable, '-m', 'vantage.sim.carla_bridge', '--tidy']
    r = subprocess.run(base, env=env, capture_output=True, text=True, timeout=120, cwd=str(tmp_path))
    assert r.returncode == 2 and '--port' in r.stderr
    r = subprocess.run(base + ['--port', '2100'], env=env, capture_output=True, text=True, timeout=120,
                       cwd=str(tmp_path))
    assert r.returncode == 0 and json.loads(r.stdout)['synchronous'] is False


def test_bridge_command_line_flags(tmp_path):
    """--faults, --scenario with --scenario-params, --ground-truth and --actors none on the command line over the
    rig's settings; a fault naming no sensor of the rig fails before anything starts (exit 2)."""
    from vantage.core import simrig
    p = _project(lidar_z=2.0, extra=False)
    p.variant.cameras[:] = p.variant.cameras[:1]
    p.variant.sensors[:] = [s for s in p.variant.sensors if s.kind == 'lidar']
    rig = simrig.build(p, simrig.SimSettings.from_dict({'image_scale': 0.05, 'duration_s': 0.8, 'traffic': 1,
                                                        'warmup_s': 0.1, 'actors': [
                                                            {'id': 'x', 'kind': 'cone', 'pose': {'x': 5, 'y': 0}}]}))
    simrig.export(rig, p, str(tmp_path / 'rig'))
    assert (tmp_path / 'rig' / 'actors.json').is_file() and not (tmp_path / 'rig' / 'faults.json').exists()
    (tmp_path / 'faults.json').write_text(json.dumps([{'tag': 'L1', 'kind': 'drop', 'frames': [1]}]))
    env = dict(os.environ, PYTHONPATH=FAKE_DIR + os.pathsep + REPO)
    base = [sys.executable, '-m', 'vantage.sim.carla_bridge', str(tmp_path / 'rig' / 'sim_rig.json')]
    r = subprocess.run(base + [str(tmp_path / 'out'), '--faults', str(tmp_path / 'faults.json'), '--scenario',
                               'lead_vehicle_brake', '--scenario-params', '{"brake_at_s": 0.2}', '--ground-truth',
                               'semantic,range', '--actors', 'none'], env=env, capture_output=True, text=True,
                       timeout=600, cwd=str(tmp_path))
    assert r.returncode == 0, r.stderr[-2000:]
    rep = json.load(open(tmp_path / 'out' / 'run.json'))
    assert rep['scenario']['name'] == 'lead_vehicle_brake' and rep['scenario']['params']['brake_at_s'] == 0.2
    assert rep['actors'] == [] and [f['kind'] for f in rep['faults_injected']] == ['drop']
    assert rep['ground_truth']['camera_kinds'] == ['semantic_segmentation', 'depth']
    assert not rep['ground_truth']['lidar_labels'] and not (tmp_path / 'out' / 'nuscenes' / 'lidarseg').exists()
    from vantage.core import groundtruth as GT
    d = GT.read(str(tmp_path / 'out'))
    f = d['cameras']['F']['frames'][0]
    assert f['instance'] is None and f['objects'] == [] and f['semantic'] and f['range']
    bad = tmp_path / 'bad.json'
    bad.write_text(json.dumps([{'tag': 'NOPE', 'kind': 'drop', 'rate': 0.1}]))
    r = subprocess.run(base + [str(tmp_path / 'out2'), '--faults', str(bad)], env=env, capture_output=True,
                       text=True, timeout=120, cwd=str(tmp_path))
    assert r.returncode == 2 and 'no sensor NOPE' in r.stderr and not (tmp_path / 'out2').exists()


def test_range_only_ground_truth(tmp_path):
    """Only the range asked for: the depth camera alone beside each view, range images and no labels or objects."""
    from vantage.core import groundtruth as GT
    r = _mini_run(tmp_path / 'r', {'ground_truth': {'semantic': False, 'instance': False, 'range': True,
                                                    'lidar_labels': False}, 'duration_s': 0.5})
    assert r['report']['ground_truth']['camera_kinds'] == ['depth']
    kinds = {e[2] for e in r['log'] if e[0] == 'spawn' and e[2].startswith('sensor.')}
    assert 'sensor.camera.depth' in kinds and not kinds & {'sensor.camera.instance_segmentation',
                                                            'sensor.camera.semantic_segmentation',
                                                            'sensor.lidar.ray_cast_semantic'}
    d = GT.read(r['out'])
    f = d['cameras']['F']['frames'][0]
    assert f['semantic'] is None and f['instance'] is None and f['objects'] == [] and GT.load_range(d, f).max() > 1
    assert d['lidars'] == {}


def test_round8_settings_and_rig_blocks():
    """SimSettings: ground truth, faults, the scenario and an actors scene, tolerant of junk; the rig's lens block,
    the ground truth's cost, the faults checked against the rig's sensors."""
    from vantage.core import simrig
    st = simrig.SimSettings.from_dict({
        'ground_truth': {'semantic': 'no', 'range': 0, 'junk': 1},
        'faults': [{'tag': 'F', 'kind': 'rotation', 'deg': [0, 99, 0]}, {'tag': 'F', 'kind': 'warp'}, 'x',
                   {'tag': 'L1', 'kind': 'drop', 'rate': 3}, {'tag': 'RD1', 'kind': 'freeze', 'from_s': 2, 'to_s': 1}],
        'scenario': {'name': 'cut_in', 'params': {'speed_kmh': 900, 'side': 'up', 'nope': 1}},
        'actors': [{'id': 'a', 'kind': 'child', 'pose': {'x': 1, 'y': 'q'}}, {'kind': 'cone', 'pose': {'x': 2, 'y': 0}},
                   {'kind': 'tank', 'pose': {'x': 0, 'y': 0}}], 'spawn_point': -9})
    assert st.ground_truth == {'semantic': False, 'instance': True, 'range': False, 'lidar_labels': True}
    assert st.faults == [{'tag': 'F', 'kind': 'rotation', 'deg': [0.0, 30.0, 0.0]},
                         {'tag': 'L1', 'kind': 'drop', 'rate': 0.9}]
    assert st.scenario == {'name': 'cut_in', 'params': {'speed_kmh': 90.0, 'side': 'left'}}
    assert st.actors == [{'id': 'a2', 'kind': 'cone', 'pose': {'x': 2.0, 'y': 0.0, 'z': 0.0, 'yaw': 0.0}}]
    assert st.spawn_point == -1 and simrig.SimSettings.from_dict(st.to_dict()) == st
    assert simrig.SimSettings.from_dict({'walkers': 999}).walkers == 300 and simrig.SimSettings().walkers == 0
    assert simrig.SimSettings.from_dict({'scenario': 'parked'}).scenario == {'name': 'parked', 'params': {}}
    p = _project(extra=False)
    rig = simrig.build(p, simrig.SimSettings.from_dict({'image_scale': 0.1, 'faults': [
        {'tag': 'F', 'kind': 'range_bias', 'm': 0.1}, {'tag': 'ZZ', 'kind': 'drop', 'rate': 0.1},
        {'tag': 'RD1', 'kind': 'clock_offset', 'ms': 3}]}), tables=False)
    d = rig.doc
    assert d['faults'] == [{'tag': 'RD1', 'kind': 'clock_offset', 'ms': 3.0}]     # range_bias: no camera's; ZZ: none
    assert any('does not apply to F' in n for n in rig.notes) and any('no sensor ZZ' in n for n in rig.notes)
    g = d['ground_truth']
    assert g['camera_kinds'] == ['instance_segmentation', 'depth']
    assert g['gt_render_mpx'] == pytest.approx(2 * g['render_mpx'], abs=0.01)
    assert all(c['lens']['model'] == 'lens' and c['gt_render_mpx'] == pytest.approx(2 * c['render_mpx'], abs=2e-3)
               for c in d['cameras'])                              # (each rounded to 1 000 px)
    assert d['scenario']['name'] == 'free_drive' and d['actors'] == {'items': [], 'spawn_point': -1,
                                                                     'frame': d['actors']['frame']}
    s = simrig.summary(rig)
    assert s['ground_truth']['instance'] and s['scenario'] == 'free_drive' and s['faults'] == 1
    rig2 = simrig.build(p, simrig.SimSettings.from_dict({'image_scale': 0.1, 'ground_truth': {
        'semantic': True, 'instance': False, 'range': False, 'lidar_labels': False}}), tables=False)
    assert rig2.doc['ground_truth']['camera_kinds'] == ['semantic_segmentation']


# ======================================================================= tests against a real CARLA server
def _live():
    spec = os.environ.get('VANTAGE_CARLA', '')
    py = os.environ.get('VANTAGE_CARLA_PYTHON', '')
    if not spec or not py:
        return None
    host, _, port = spec.partition(':')
    from vantage.sim import server as SV
    return (host or '127.0.0.1', int(port or 2000), py) if SV.port_open(host or '127.0.0.1', int(port or 2000)) \
        else None


carla_live = pytest.mark.skipif(_live() is None, reason='no CARLA server (set VANTAGE_CARLA=host:port and '
                                                        'VANTAGE_CARLA_PYTHON to CARLA\'s client Python)')


@carla_live
def test_live_carla_recording(tmp_path):
    """A short drive on the real server: lidar, radar, GNSS and IMU always; the cameras when VANTAGE_CARLA_CAMERAS=1
    (a server that renders); then the drive-log check: streams healthy, and with cameras every camera that can be
    checked holds against the lidar."""
    from vantage.core import drivelog as DL
    from vantage.core import simrig
    host, port, py = _live()
    cams = os.environ.get('VANTAGE_CARLA_CAMERAS') == '1'
    p = _project()
    rig = simrig.build(p, simrig.SimSettings.from_dict({'image_scale': 0.5, 'duration_s': 6.0, 'traffic': 20,
                                                        'port': port}))
    simrig.export(rig, p, str(tmp_path / 'rig'))
    env = dict(os.environ, PYTHONPATH=REPO)
    args = [py, '-m', 'vantage.sim.carla_bridge', str(tmp_path / 'rig' / 'sim_rig.json'), str(tmp_path / 'out'),
            '--host', host, '--port', str(port)] + ([] if cams else ['--no-cameras'])
    t = time.time()
    r = subprocess.run(args, env=env, capture_output=True, text=True, timeout=3000, cwd=str(tmp_path))
    assert r.returncode == 0, r.stderr[-3000:]
    rep = json.load(open(tmp_path / 'out' / 'run.json'))
    assert rep['status'] == 'ok' and rep['counts']['sample_data'] > 0
    print('live run', round(time.time() - t, 1), 's', rep['performance'])
    bag = str(tmp_path / 'out' / 'rosbag' / rig.doc['output']['bag_name'])
    res = DL.run(p, bag)
    st = {e['tag']: e for e in res.streams}
    assert st['L1']['status'] in ('ok', 'warning') and abs(st['L1']['rate_hz'] - 10.0) < 0.05
    ann = json.load(open(tmp_path / 'out' / 'nuscenes' / 'v1.0-vantage' / 'sample_annotation.json'))
    moving = [a for a in ann if a['num_lidar_pts'] > 0]
    assert moving, 'no lidar point in any annotated box: the lidar frame would be wrong'
    if cams:
        held = [x for x in res.extrinsics if x.get('verdict') == 'ok']
        bad = [x for x in res.extrinsics if x.get('verdict') == 'rotated']
        assert held and not bad, [(x['tag'], x.get('verdict'), x.get('delta_deg')) for x in res.extrinsics]


# ======================================================================= the page
from test_gui import app, win, pump, settle, _wait   # noqa: E402,F401  (fixtures)


def _open(win, name, cls):
    if name not in win._page_widgets:
        pg = cls(win.ctx)
        win.pages.addWidget(pg)
        win._page_widgets[name] = pg
    win.set_page(name)
    return win._page_widgets[name]


def test_sim_page_plan_edit_undo_run_and_open_checks(app, win, tmp_path, monkeypatch):
    from vantage.sim import server as SV
    from vantage.ui.workspaces.sim_ws import SimPage
    p = win.store.project
    v = next(v for v in p.variants if v.sensors)

    def activate(pj):
        pj.active_uid = v.uid
    win.store.edit('variants', activate, 'data rig')
    pump(app, 0.3)
    pg = _open(win, 'sim', SimPage)
    _wait(app, lambda: pg.rig is not None, 120)
    assert pg.table.rowCount() == len([c for c in p.variant.cameras if c.visible]) + len(p.variant.sensors)
    assert pg.tiles['step'].value == '80'
    # edit: the duration field, as a user types it; undo puts it back
    f, scale = pg.fields['duration_s']
    f.lineEdit().selectAll()
    from PyQt6.QtTest import QTest
    from PyQt6.QtCore import Qt
    QTest.keyClicks(f.lineEdit(), '1')
    QTest.keyClick(f.lineEdit(), Qt.Key.Key_Return)
    pump(app, 0.2)
    assert pg.settings.duration_s == 1.0
    win.store.undo.undo()
    pump(app, 0.2)
    assert pg.settings.duration_s == 20.0
    # a quick run against the fake carla: small images, one camera, a mast lidar
    def fn(pj):
        pj.analysis.sim = {'duration_s': 0.5, 'image_scale': 0.05, 'traffic': 1, 'warmup_s': 0.1}
        vv = pj.variant
        vv.cameras[:] = vv.cameras[:1]
        next(s for s in vv.sensors if s.kind == 'lidar').z = 2.0
    win.store.edit(['variants', 'analysis'], fn, 'test setup')
    _wait(app, lambda: pg.rig is not None and len(pg.rig.doc['cameras']) == 1, 120)
    monkeypatch.setattr(SV, 'port_open', lambda *a, **k: True)
    pg.python = sys.executable
    pg.extra_pythonpath = FAKE_DIR
    out = str(tmp_path / 'carla_run')
    assert pg.run(out)
    _wait(app, lambda: not pg.busy, 600)
    assert pg.result is not None and pg.result['status'] == 'ok', pg.g_status.text()
    assert 'samples' in pg.out.toPlainText() and pg.b_drivelog.isEnabled()
    assert pg.open_in('drivelog')
    assert win.page_name() == 'drivelog'
    dl = win.page('drivelog')
    _wait(app, lambda: dl.result is not None, 300)
    assert dl.result.metrics['streams'] > 0


def test_a_bridge_that_dies_without_its_report_leaves_no_half_recording(app, win, tmp_path):
    """CARLA's client library aborts the bridge when the server dies mid-run, so no run.json: the page removes what it
    wrote (as the bridge does on an error), keeps the rig and the log, writes run.json with the reason, and shows no
    path in full (from ~, shortened)."""
    from vantage.core import simrig as SR
    from vantage.sim import server as SV
    from vantage.ui.workspaces.sim_ws import SimPage
    pg = _open(win, 'sim', SimPage)
    out = tmp_path / 'drive'
    for d in ('nuscenes/samples', 'rosbag/bag', 'rig'):
        (out / d).mkdir(parents=True)
    for f in ('dataset.json', 'README.md', 'bridge.log', 'rig/sim_rig.json', 'rosbag/bag/metadata.yaml'):
        (out / f).write_text('x')
    import socket
    with socket.socket() as sk:
        sk.bind(('127.0.0.1', 0))
        port = sk.getsockname()[1]                                  # nothing answers there
    assert not SV.port_open('127.0.0.1', port)
    doc = pg._no_report(str(out), -6, port)
    assert doc['status'] == 'failed' and doc['server_stopped'] and 'server stopped during the run' in doc['error']
    assert sorted(os.listdir(out)) == ['bridge.log', 'rig', 'run.json'] and (out / 'rig' / 'sim_rig.json').exists()
    from vantage.ui.workspaces.sim_ws import short_path
    assert SR.read_run(str(out))['error'] == doc['error'] and doc['error'].endswith(short_path(str(out / 'bridge.log')))
    assert 'tidied' not in doc                                      # (the server is gone: nothing to tidy)
    # the server still up (a bridge killed by its memory cap): the page tidies it (carla_bridge --tidy)
    import pytest as _pt
    mp = _pt.MonkeyPatch()
    try:
        mp.setattr(SV, 'port_open', lambda *a, **k: True)
        pg.python, pg.extra_pythonpath = sys.executable, FAKE_DIR
        out2 = tmp_path / 'drive2'
        (out2 / 'rig').mkdir(parents=True)
        doc2 = pg._no_report(str(out2), -9, 2100)
    finally:
        mp.undo()
    assert not doc2['server_stopped'] and doc2['tidied']['synchronous'] is False, doc2


def test_sim_page_ground_truth_scenario_faults_and_actors(app, win, tmp_path):
    """The page's round-8 sections driven like a user: a ground-truth switch, the scenario chooser and its fields,
    the faults table (add, type values, change kind and sensor, remove), an actors scene from a file; every edit one
    undo step; the command line says all of it."""
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    from vantage.sim import carla_bridge as CB
    from vantage.ui.workspaces.sim_ws import SimPage, fault_text, parse_fault
    p = win.store.project
    v = next(v for v in p.variants if v.sensors)

    def activate(pj):
        pj.active_uid = v.uid
        pj.analysis.sim = {}
    win.store.edit(['variants', 'analysis'], activate, 'data rig')
    pump(app, 0.3)
    pg = _open(win, 'sim', SimPage)
    _wait(app, lambda: pg.rig is not None, 120)
    assert 'Renders' in pg.gt_cost.text() and 'semantic lidar' in pg.gt_cost.text()
    # ground truth: a switch as a user clicks it; undo puts it back
    t = pg.gt_toggles['range']
    assert t.isChecked()
    t.click()
    pump(app, 0.2)
    assert pg.settings.ground_truth == {'semantic': True, 'instance': True, 'range': False, 'lidar_labels': True}
    win.store.undo.undo()
    pump(app, 0.2)
    assert pg.settings.ground_truth['range'] and pg.gt_toggles['range'].isChecked()
    # the scenario: cut-in, its fields appear; a speed typed, a side chosen
    pg.scen.setCurrentIndex(pg.scen.findData('cut_in'))
    pump(app, 0.2)
    assert pg.settings.scenario == {'name': 'cut_in', 'params': {}} and set(pg.scen_fields) == \
        set(CB.SCENARIOS['cut_in'][2]) and 'next lane' in pg.scen_about.text()
    f = pg.scen_fields['speed_kmh']
    f.lineEdit().selectAll()
    QTest.keyClicks(f.lineEdit(), '50')
    QTest.keyClick(f.lineEdit(), Qt.Key.Key_Return)
    pump(app, 0.2)
    pg.scen_fields['side'].buttons['right'].click()
    pump(app, 0.2)
    assert pg.settings.scenario == {'name': 'cut_in', 'params': {'speed_kmh': 50.0, 'side': 'right'}}
    win.store.undo.undo()
    pump(app, 0.2)
    assert pg.settings.scenario['params'] == {'speed_kmh': 50.0} and pg.scen_fields['side'].value() == 'left'
    # faults
    pg.b_add_fault.click()
    pump(app, 0.2)
    first = next(c.tag for c in v.cameras if c.visible)
    assert pg.settings.faults == [{'tag': first, 'kind': 'rotation', 'deg': [0.0, 0.5, 0.0]}]
    assert pg.faults.rowCount() == 1
    ed = pg.faults.cellWidget(0, 2)
    ed.setText('0, 1.5, -0.2')
    ed.editingFinished.emit()
    pump(app, 0.2)
    assert pg.settings.faults[0]['deg'] == [0.0, 1.5, -0.2]
    ed = pg.faults.cellWidget(0, 2)
    ed.setText('oops')
    ed.editingFinished.emit()
    pump(app, 0.2)
    assert pg.settings.faults[0]['deg'] == [0.0, 1.5, -0.2] and '3 numbers' in pg.fault_status.text()
    kc = pg.faults.cellWidget(0, 1)
    kc.setCurrentIndex(kc.findData('drop'))
    pump(app, 0.2)
    assert pg.settings.faults == [{'tag': first, 'kind': 'drop', 'rate': 0.05}]
    sc = pg.faults.cellWidget(0, 0)
    sc.setCurrentIndex(sc.findData('L1'))
    pump(app, 0.2)
    assert pg.settings.faults == [{'tag': 'L1', 'kind': 'drop', 'rate': 0.05}]
    kc = pg.faults.cellWidget(0, 1)
    assert kc.findData('range_bias') >= 0                     # a lidar's kinds
    pg.faults.cellWidget(0, 3).click()
    pump(app, 0.2)
    assert pg.settings.faults == [] and pg.faults.rowCount() == 0
    win.store.undo.undo()
    pump(app, 0.2)
    assert pg.settings.faults == [{'tag': 'L1', 'kind': 'drop', 'rate': 0.05}] and pg.faults.rowCount() == 1
    for f in ({'tag': 'F', 'kind': 'rotation', 'deg': [0.0, -0.5, 1.25]}, {'tag': 'F', 'kind': 'translation',
              'm': [0.01, 0.0, -0.02]}, {'tag': 'F', 'kind': 'clock_offset', 'ms': -7.5},
              {'tag': 'L1', 'kind': 'range_bias', 'm': 0.05}, {'tag': 'R', 'kind': 'drop', 'frames': [3, 7]},
              {'tag': 'R', 'kind': 'drop', 'rate': 0.25}, {'tag': 'L1', 'kind': 'freeze', 'from_s': 2.0, 'to_s': 3.5}):
        assert parse_fault(f['tag'], f['kind'], fault_text(f)) == f
    # an actors scene from a file (an unknown kind left out)
    path = tmp_path / 'ACTORS.json'
    path.write_text(json.dumps([{'id': 'kid', 'kind': 'child', 'pose': {'x': 5, 'y': 0}},
                                {'kind': 'tank', 'pose': {'x': 1, 'y': 1}}]))
    assert pg.load_actors(str(path))
    pump(app, 0.2)
    assert pg.settings.actors == [{'id': 'kid', 'kind': 'child', 'pose': {'x': 5.0, 'y': 0.0, 'z': 0.0, 'yaw': 0.0}}]
    assert '1 actors' in pg.actors_label.text() and '1 child' in pg.actors_label.text()
    cmd = pg.command_line('/r/sim_rig.json')
    assert '--actors /r/actors.json' in cmd and '--faults /r/faults.json' in cmd and '--scenario ' not in cmd
    assert '--ground-truth semantic,instance,range,lidar_labels' in cmd
    pg.b_clear_actors.click()
    pump(app, 0.2)
    cmd = pg.command_line('/r/sim_rig.json')
    assert pg.settings.actors == [] and '--scenario cut_in' in cmd and \
        """--scenario-params '{"speed_kmh":50.0}'""" in cmd and '--actors' not in cmd
    # Export rig writes the lists the command line names
    from vantage.core import simrig as SR
    names = [n for n, _ in SR.files(SR.build(win.store.project.snapshot(), pg.settings, tables=False))]
    assert names == ['sim_rig.json', 'sim_remap.npz', 'faults.json']
    win.store.edit('analysis', lambda pj: setattr(pj.analysis, 'sim', {}), 'reset')
    pump(app, 0.2)


def test_open_run_shows_a_finished_drive_and_its_ground_truth(app, win, fake_run, tmp_path):
    """Open run… on a drive recorded before (the bridge's own, as a headless --carla-run makes one): refused for a
    folder without run.json or with another schema; else its report as after a run, the Drive-log check, As-built
    check and Detection quality buttons on it, and the Ground truth tab: every camera with frames, each layer the
    drive's own label images in colour pixel for pixel (checked against core.groundtruth), the boxes, the object table
    sorted by range, a row or a click on an object highlighting it, the opacity blending, the lidar from above; a drive
    without ground truth says so; a failed run shows its report with the checks off; a new project clears it."""
    import shutil
    from PyQt6.QtCore import QPointF, Qt
    from PyQt6.QtGui import QImage
    from vantage.core import groundtruth as GT
    from vantage.ui import gtview as GV
    from vantage.ui.workspaces.sim_ws import SimPage
    out = fake_run['out']
    pg = _open(win, 'sim', SimPage)
    empty = tmp_path / 'empty'
    empty.mkdir()
    assert not pg.open_run(str(empty)) and 'holds no run.json' in pg.g_status.text()
    (empty / 'run.json').write_text('{"schema": "something/1"}')
    assert not pg.open_run(str(empty)) and 'not a CARLA run' in pg.g_status.text()
    assert pg.result is None and not pg.b_drivelog.isEnabled() and pg.gt.canvas.img is None
    # the drive: its report and its checks
    assert pg.open_run(out)
    pump(app, 0.2)
    assert pg.out_dir == os.path.normpath(out) and pg.result['status'] == 'ok'
    from vantage.core import simrig as SR
    assert pg.out.toPlainText() == SimPage.report_text(SR.read_run(out)) and 'Ground truth:' in pg.out.toPlainText()
    assert pg.b_drivelog.isEnabled() and pg.b_asbuilt.isEnabled() and pg.b_detect.isEnabled()
    assert 'camera frames with ground truth' in pg.g_status.text()
    # the ground truth: the tab given room, every camera that has frames
    pg.show_ground_truth()
    pump(app, 0.2)
    assert pg.out_tabs.currentWidget() is pg.gt and pg.right.sizes()[1] >= 0.6 * sum(pg.right.sizes())
    gt = pg.gt
    doc = GT.read(out)
    tags = [t for t, c in doc['cameras'].items() if c['frames']]
    assert gt.cameras() == tags and len(tags) == 6
    tag = tags[0]
    assert gt.set_camera(tag)
    k = gt.busiest_frame(1)
    gt.set_frame(k)
    f = GT.frames(doc, tag)[k]
    assert f['objects'] and gt.frame_index() == k
    img = GT.load_image(doc, f)

    def shown():
        q = gt.canvas.img.convertToFormat(QImage.Format.Format_RGB888)
        a = np.frombuffer(q.constBits().asstring(q.sizeInBytes()), np.uint8).reshape(q.height(), q.bytesPerLine())
        return a[:, :q.width() * 3].reshape(q.height(), q.width(), 3)
    assert np.array_equal(shown(), img)
    assert gt.set_layer('semantic')
    sem = GT.load_semantic(doc, f)
    assert np.array_equal(shown(), GT.colour_semantic(sem))
    present = [c['name'] for c in GT.classes_in(sem)]
    assert [t.split('  ')[0] for _, t, _ in gt.legend.rows] == [n.replace('_', ' ') for n in present]
    assert gt.set_layer('instance')
    inst = GT.load_instance(doc, f)
    assert np.array_equal(shown(), GT.colour_instance(inst))
    gt.set_opacity(40)                                           # 40 % of the labels over the image
    blend = (img.astype(np.float32) * 0.6 + GT.colour_instance(inst).astype(np.float32) * 0.4).astype(np.uint8)
    assert np.abs(shown().astype(int) - blend.astype(int)).max() <= 1
    gt.set_opacity(100)
    assert gt.set_layer('range')
    rng = GT.load_range(doc, f)
    near, far = GT.range_limits(rng)
    t = GT.range_position(rng, near, far)
    a = shown()
    ok = ~np.isnan(t)
    assert np.array_equal(a[ok], GV.RANGE_LUT[np.rint(t[ok] * 255).astype(int)])
    assert gt.legend.bar is not None and gt.legend.bar[3].endswith(f'{near:g}–{far:g} m)')
    # boxes: tight and full of every object, labelled in their class; the table nearest first
    gt.set_layer('image')
    gt.set_boxes('both')
    objs = f['objects']
    n_full = sum(1 for o in objs if o.get('bbox_full'))
    assert len(gt.canvas.shapes) == len(objs) + n_full
    assert sum(1 for s in gt.canvas.shapes if s['dashed']) == n_full
    gt.set_boxes('none')
    assert gt.canvas.shapes == []
    gt.set_boxes('tight')
    assert gt.table.rowCount() == len(objs)
    ranges = [gt.table.item(r, 2).data(Qt.ItemDataRole.UserRole + 1) for r in range(gt.table.rowCount())]
    assert ranges == sorted(ranges)
    # a row clicked highlights its object: its box and its pixels
    big = max(objs, key=lambda o: o['pixels'])
    r = next(r for r in range(gt.table.rowCount()) if gt.table.item(r, 1).text() == str(big['id']))
    gt.table.selectRow(r)
    pump(app, 0.1)
    assert gt.selected == big['id'] and [s['hi'] for s in gt.canvas.shapes].count(True) == 1
    m = inst == big['id']
    edge = GV.outline(m)                                       # outlined: its border and just outside it
    assert (edge & m).any() and (edge & ~m).any() and (m & ~edge).any()
    assert (shown()[edge] == 255).all() and (shown()[~edge] == img[~edge]).all()
    # a click on another object in the image selects it; on no object clears it
    other = next(o for o in sorted(objs, key=lambda o: -o['pixels']) if o['id'] != big['id'])
    ys, xs = np.nonzero(inst == other['id'])
    gt.canvas.picked.emit(float(xs[0]) + 0.5, float(ys[0]) + 0.5)
    pump(app, 0.1)
    assert gt.selected == other['id'] and gt.table.item(gt.table.selectionModel().selectedRows()[0].row(), 1).text() \
        == str(other['id'])
    ys, xs = np.nonzero(inst == 0)
    gt.canvas.picked.emit(float(xs[0]) + 0.5, float(ys[0]) + 0.5)
    pump(app, 0.1)
    assert gt.selected is None and not gt.table.selectionModel().selectedRows()
    # zoom: to a box and back
    x0, y0, x1, y1 = big['bbox']
    gt.canvas.zoom_to(x0, y0, x1, y1)
    assert gt.canvas.zoom > 1.0
    gt.canvas.reset_zoom()
    assert gt.canvas.zoom == 1.0
    # the lidar from above: the sweep nearest the frame, in its classes
    assert gt.set_layer('lidar')
    assert 'from above' in gt.canvas.caption and gt.legend.rows and len(gt.canvas.shapes) == \
        sum(1 for o in objs if o.get('box3d'))
    # the frames: the slider, loaded when shown, a few kept
    for j in range(len(GT.frames(doc, tag))):
        gt.set_frame(j)
    assert len(gt._cache) <= GV.CACHE_FRAMES * 3
    # Detection quality opens it
    assert pg.open_detection() and win.page_name() == 'detection'
    assert os.path.normpath(win.page('detection').settings.recordings[0]) == os.path.normpath(out)
    # and the drive-log check reads it and runs
    assert pg.open_in('drivelog') and win.page_name() == 'drivelog'
    dl = win.page('drivelog')
    _wait(app, lambda: dl.result is not None, 300)
    assert dl.result.metrics['streams'] > 0
    win.set_page('sim')
    # a drive without ground truth: said
    nogt = tmp_path / 'nogt'
    nogt.mkdir()
    shutil.copy(os.path.join(out, 'run.json'), nogt / 'run.json')
    assert pg.open_run(str(nogt)) and not pg.gt.has_ground_truth()
    assert pg.gt.canvas.empty == GV.EMPTY_NO_GT and pg.gt.canvas.img is None and 'no ground truth' in \
        pg.g_status.text() and not pg.b_detect.isEnabled()
    # a failed run: its report, no checks
    bad = tmp_path / 'failed'
    bad.mkdir()
    (bad / 'run.json').write_text(json.dumps({'schema': 'vantage.sim_run/1', 'status': 'failed',
                                              'error': 'the CARLA server stopped answering'}))
    assert pg.open_run(str(bad)) and 'did not end well' in pg.g_status.text()
    assert '"failed"' in pg.out.toPlainText() and not pg.b_drivelog.isEnabled() and pg.gt.canvas.img is None
    # a new project: the drive goes
    assert pg.open_run(out)
    from vantage.core import samples
    win.store.set_project(samples.tigor_sample_project())
    pump(app, 0.3)
    assert pg.result is None and pg.out_dir == '' and pg.gt.doc is None and pg.out.toPlainText() == ''
