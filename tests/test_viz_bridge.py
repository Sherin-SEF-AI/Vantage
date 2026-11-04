"""The Foxglove / Rerun bridge (core/viz_bridge.py): the scene's geometry, the MCAP file of Foxglove's JSON schemas
read back by our reader and by Foxglove's own mcap package, every message valid against its embedded schema, the
camera intrinsics and transforms against the analysis, the ego mask against the cube map, and Rerun through a fake
rerun module (no viewer, no network)."""
import json
import math
import sys
import types

import numpy as np
import pytest

from vantage.core import samples
from vantage.core import viz_bridge as VB
from vantage.core.analysis import Engine, camera_frames
from vantage.core.lens import resolve


@pytest.fixture(scope='module')
def rig():
    p = samples.tigor_sample_project()
    p.active_uid = next(v.uid for v in p.variants if v.sensors)
    e = Engine()
    r = e.analyse(p)
    return p, e, r, VB.scene_entities(p, e, result=r)


def test_scene_entities_geometry(rig):
    p, e, r, ents = rig
    assert ents['frame'] == 'vehicle'
    cams = [c for c in p.variant.cameras if c.visible]
    assert [c['tag'] for c in ents['cameras']] == [c.tag for c in cams]
    for c, cam in zip(ents['cameras'], cams):
        it = resolve(p.module_for(cam), p.lens_for(cam))
        R, W2O = camera_frames(cam)
        assert np.allclose(c['R'], W2O.T) and np.allclose(c['R'] @ c['R'].T, np.eye(3), atol=1e-9)
        assert np.allclose(c['pos'], cam.pos())
        assert c['K'][0, 0] == pytest.approx(it.f_px) and c['K'][0, 2] == pytest.approx(it.cx - 0.5)
        # the field outline: every point frustum_m from the camera, inside the lens's field
        d = c['outline'] - c['pos']
        assert np.allclose(np.linalg.norm(d, axis=1), 2.0, atol=1e-6)
        opt = d @ c['R']
        theta = np.arctan2(np.hypot(opt[:, 0], opt[:, 1]), opt[:, 2])
        assert theta.max() <= it.max_theta + 1e-6
        # the outline projects onto the image border (or the image circle) through the real lens
        u, v, _, ok = it.project_dirs(opt / np.linalg.norm(opt, axis=1, keepdims=True))
        on_edge = (np.minimum.reduce([u, v, it.width - u, it.height - v]) < 1.0) | (theta > it.max_theta - 1e-4)
        assert on_edge.all()
        if it.projection == 'kannala_brandt':
            assert c['distortion_model'] == 'kannala_brandt' and c['D'] == pytest.approx(list(it.kb))
        fp = c['footprint']
        rc = next(x for x in r.cams if x.uid == cam.uid)
        assert fp.shape == (len(rc.segments), 2, 3) and np.allclose(fp[..., :2], rc.segments)
    lid = next(s for s in ents['sensors'] if s['kind'] == 'lidar')
    assert lid['rings'] and all(np.allclose(ring[:, 2], 0.0, atol=1e-9) for _, ring in lid['rings'])
    rad = next(s for s in ents['sensors'] if s['kind'] == 'radar')
    assert rad['wedges']
    tf = {t['child']: t for t in ents['transforms']}
    assert {f'{c["tag"]}_optical' for c in ents['cameras']} <= set(tf) and lid['tag'] in tf
    b = ents['blind']
    assert b is not None and len(b['centres']) > 0 and b['half'] == pytest.approx([0.1, 0.1, 0.05])


def test_ego_mask_matches_the_cube_map(rig):
    p, e, r, ents = rig
    c = next(c for c in ents['cameras'] if c['tag'] == 'F')
    cam = p.variant.by_tag('F')
    it = resolve(p.module_for(cam), p.lens_for(cam))
    R, W2O = camera_frames(cam)
    m = c['mask']
    h, w = m.shape
    assert w == 480 and h == round(480 * it.height / it.width) and 0.02 < m.mean() < 0.6
    # a pixel in the mask's lower middle sees the bumper (ego); the image centre sees ground or sky
    rng = np.random.default_rng(1)
    for _ in range(40):
        y, x = int(rng.integers(h)), int(rng.integers(w))
        d, th = it.unproject(np.array([(x + 0.5) / w * it.width]), np.array([(y + 0.5) / h * it.height]))
        if th[0] > it.max_theta:
            assert not m[y, x]
            continue
        dw = np.asarray(d, float) @ W2O
        bd, _ = e.cube(cam.pos()).lookup(dw)
        tg = -cam.pos()[2] / dw[0, 2] if dw[0, 2] < -1e-6 else math.inf
        assert m[y, x] == bool(np.isfinite(bd[0]) and bd[0] < tg)


def test_foxglove_mcap_reads_back_and_validates(rig, tmp_path):
    p, e, r, ents = rig
    path = VB.write_mcap(str(tmp_path / 'scene.mcap'), VB.foxglove_messages(ents))
    msgs = VB.read_mcap(path)
    topics = {t for t, *_ in msgs}
    assert {'/tf', '/vantage/scene', '/F/camera_info', '/F/ego_mask', '/L1/ground_rings'} <= topics
    assert all(enc == 'jsonschema' and menc == 'json' for _, _, enc, menc, _ in msgs)
    names = {t: n for t, n, *_ in msgs}
    assert names['/tf'] == 'foxglove.FrameTransforms' and names['/vantage/scene'] == 'foxglove.SceneUpdate'
    assert names['/F/camera_info'] == 'foxglove.CameraCalibration' and names['/F/ego_mask'] == 'foxglove.CompressedImage'
    assert names['/L1/ground_rings'] == 'foxglove.PointCloud'
    # every message is valid against its schema (the embedded Foxglove JSON schemas)
    js = pytest.importorskip('jsonschema')
    sch = VB.schemas()
    for t, n, _, _, m in msgs:
        js.validate(m, sch[n])
    # the content
    by = {t: m for t, _, _, _, m in msgs}
    tf = {x['child_frame_id']: x for x in by['/tf']['transforms']}
    f = next(c for c in ents['cameras'] if c['tag'] == 'F')
    q = tf['F_optical']['rotation']
    from vantage.core.rosbag import quat_matrix
    assert np.allclose(quat_matrix([q['x'], q['y'], q['z'], q['w']]), f['R'], atol=1e-9)
    ci = by['/F/camera_info']
    assert ci['distortion_model'] == 'kannala_brandt' and len(ci['D']) == 4 and len(ci['K']) == 9 and len(ci['P']) == 12
    import base64
    import cv2
    img = cv2.imdecode(np.frombuffer(base64.b64decode(by['/F/ego_mask']['data']), np.uint8), cv2.IMREAD_GRAYSCALE)
    assert img.shape == f['mask'].shape and np.array_equal(img > 127, f['mask'])
    pc = by['/L1/ground_rings']
    P = np.frombuffer(base64.b64decode(pc['data']), '<f4').reshape(-1, 3)
    lid = next(s for s in ents['sensors'] if s['tag'] == 'L1')
    assert np.allclose(P, np.concatenate([r_ for _, r_ in lid['rings']]), atol=1e-5)
    ids = {en['id'] for en in by['/vantage/scene']['entities']}
    assert {'cameras/F/field', 'cameras/F/footprint', 'sensors/L1', 'blind_volume/voxels'} <= ids
    assert any(i.startswith('vehicle/') for i in ids)


def test_the_mcap_file_opens_in_foxglove_s_reader(rig, tmp_path):
    mc = pytest.importorskip('mcap.reader')
    p, e, r, ents = rig
    path = VB.write_mcap(str(tmp_path / 'scene.mcap'), VB.foxglove_messages(ents))
    with open(path, 'rb') as fh:
        rd = mc.make_reader(fh)
        summ = rd.get_summary()
        assert summ is not None and summ.statistics.message_count == len(VB.read_mcap(path))
        names = {s.name for s in summ.schemas.values()}
        assert {'foxglove.SceneUpdate', 'foxglove.FrameTransforms', 'foxglove.CameraCalibration'} <= names
        n = 0
        for schema, channel, message in rd.iter_messages():
            assert schema.encoding == 'jsonschema' and channel.message_encoding == 'json'
            json.loads(message.data)
            n += 1
        assert n == summ.statistics.message_count


def test_export_and_live_without_the_packages(rig, tmp_path, monkeypatch):
    p, e, r, ents = rig
    path = VB.export_foxglove_mcap(p, e, str(tmp_path / 'x' / 's.mcap'), result=r, volume=False)
    assert VB.read_mcap(path)
    monkeypatch.setitem(sys.modules, 'foxglove', None)
    monkeypatch.setitem(sys.modules, 'rerun', None)
    assert not VB.foxglove_available() and not VB.rerun_available()
    live, msg = VB.serve_foxglove(p, e, result=r)
    assert live is None and 'pip install foxglove-sdk' in msg
    assert 'pip install rerun-sdk' in VB.stream_rerun(p, e, result=r)


class _Arch:
    def __init__(self, kind, *a, **k):
        self.kind, self.a, self.k = kind, a, k


def _fake_module(name, calls, kinds):
    m = types.ModuleType(name)
    for k in kinds:
        setattr(m, k, (lambda kind: (lambda *a, **kw: _Arch(kind, *a, **kw)))(k))
    return m


def test_rerun_through_a_fake_module(rig, tmp_path, monkeypatch):
    p, e, r, ents = rig
    calls = []
    rr = _fake_module('rerun', calls, ['Mesh3D', 'LineStrips3D', 'Transform3D', 'Pinhole', 'Image', 'Points3D',
                                       'Boxes3D', 'TextDocument', 'Quaternion'])
    rr.ViewCoordinates = types.SimpleNamespace(RIGHT_HAND_Z_UP='RHZU', RDF='RDF')
    rr.init = lambda app, **k: calls.append(('init', app, k))
    rr.spawn = lambda **k: calls.append(('spawn', k))
    rr.connect_grpc = lambda url=None, **k: calls.append(('connect', url))
    rr.save = lambda path, **k: calls.append(('save', path))
    rr.log = lambda path, arch, static=False: calls.append(('log', path, arch, static))
    monkeypatch.setitem(sys.modules, 'rerun', rr)
    assert VB.rerun_available()
    out = VB.stream_rerun(p, e, mode='save', path=str(tmp_path / 's.rrd'), result=r)
    assert out.startswith('Sent ') and calls[0][0] == 'init' and calls[1] == ('save', str(tmp_path / 's.rrd'))
    logs = {c[1]: c[2] for c in calls if c[0] == 'log'}
    assert all(c[3] for c in calls if c[0] == 'log')                    # a static scene
    assert logs['vehicle'] == 'RHZU'
    f = next(c for c in ents['cameras'] if c['tag'] == 'F')
    tr = logs['vehicle/cameras/F']
    assert tr.kind == 'Transform3D' and np.allclose(tr.k['translation'], f['pos']) and np.allclose(tr.k['mat3x3'],
                                                                                                   f['R'])
    ph = logs['vehicle/cameras/F/image']
    assert ph.kind == 'Pinhole' and ph.k['camera_xyz'] == 'RDF' and np.allclose(ph.k['image_from_camera'], f['K'])
    assert ph.k['resolution'] == [f['width'], f['height']]
    assert logs['vehicle/cameras/F/image/ego_mask'].kind == 'Image'
    assert logs['vehicle/fields/F'].kind == 'LineStrips3D' and logs['vehicle/sensors/L1'].kind == 'Boxes3D'
    assert logs['vehicle/sensors/L1/ground_rings'].kind == 'LineStrips3D'
    assert logs['vehicle/blind_volume/voxels'].kind == 'Boxes3D'
    assert any(k.startswith('vehicle/mesh/') and v.kind == 'Mesh3D' for k, v in logs.items())
    assert 'kannala_brandt' in logs['vehicle/cameras/F/lens'].a[0]
    # spawn and connect
    calls.clear()
    VB.stream_rerun(p, e, mode='connect', url='rerun+http://127.0.0.1:9876/proxy', result=r, volume=False)
    assert ('connect', 'rerun+http://127.0.0.1:9876/proxy') in calls
    calls.clear()
    VB.stream_rerun(p, e, mode='spawn', result=r, volume=False)
    assert calls[1][0] == 'spawn'
    with pytest.raises(ValueError):
        VB.stream_rerun(p, e, mode='save', result=r)


def test_live_foxglove_through_a_fake_sdk(rig, monkeypatch):
    p, e, r, ents = rig
    logged, made = [], {}
    fx = types.ModuleType('foxglove')

    class Server:
        stopped = False

        def stop(self):
            Server.stopped = True

    class Channel:
        def __init__(self, topic, schema=None, **k):
            made[topic] = schema

        def log(self, msg, log_time=None):
            logged.append(msg)

    fx.start_server = lambda port=8765, **k: Server()
    fx.Channel = Channel
    monkeypatch.setitem(sys.modules, 'foxglove', fx)
    live, msg = VB.serve_foxglove(p, e, port=8799, result=r, volume=False)
    try:
        assert 'ws://127.0.0.1:8799' in msg and '/vantage/scene' in made
        assert made['/vantage/scene']['title'] == 'foxglove.SceneUpdate'
        import time
        t = time.time()
        while len(logged) < len(made) and time.time() - t < 5:
            time.sleep(0.02)
        assert len(logged) >= len(made)
    finally:
        live.stop()
    assert Server.stopped


def test_png_and_quaternion_helpers():
    import cv2
    m = np.zeros((5, 7), bool)
    m[1:3, 2:6] = True
    img = cv2.imdecode(np.frombuffer(VB._png(m), np.uint8), cv2.IMREAD_GRAYSCALE)
    assert np.array_equal(img > 127, m)
    from vantage.core.rosbag import quat_matrix
    rng = np.random.default_rng(0)
    for _ in range(20):
        A = np.linalg.qr(rng.normal(size=(3, 3)))[0]
        A *= np.sign(np.linalg.det(A))
        assert np.allclose(quat_matrix(VB._quat(A)), A, atol=1e-9)
