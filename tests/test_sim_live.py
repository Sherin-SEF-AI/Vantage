"""CARLA as a live rig (vantage.sim.live and the CARLA page's live panel) against the fake carla (tests/fake_carla): the
MJPEG streams OpenCV reads, rig health's frame rate on them, a rotation fault against the camera planned at the turned
pose and against the drift indicator, the delivery faults (freeze, drop, delay) in rig health's verdicts, tidy-up of
CARLA and of the project's streams, the command line, and the panel driven like a user.

carla_live tests run only against a real server (VANTAGE_CARLA=host:port and VANTAGE_CARLA_PYTHON, as
tests/test_sim.py)."""
import contextlib
import dataclasses
import importlib.util
import json
import math
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
FAKE_DIR = os.path.join(HERE, 'fake_carla')
BOX_BP = 'vehicle.citroen.c3'
_open_url = urllib.request.build_opener(urllib.request.ProxyHandler({})).open


def _fake():
    """The fake carla module, loaded under its own name (never shadowing a real carla); shared with test_sim."""
    name = 'fake_carla_for_tests'
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, os.path.join(FAKE_DIR, 'carla', '__init__.py'))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


NARROW = 'h03r2'                  # a 37° lens: one pinhole view per frame (the fake renders a view in ~15 ms)


def _sample(tags=('F',), fps=None, lens=None):
    """The sample's data rig with only the given cameras visible (a slower module, a narrow lens: the fake keeps real
    time on a busy machine)."""
    from vantage.core import samples
    p = samples.tigor_sample_project()
    v = next(v for v in p.variants if v.sensors)
    p.active_uid = v.uid
    for c in v.cameras:
        c.visible = c.tag in tags
        if lens:
            c.lens_id = lens
    if fps is not None:
        p.modules['sturdecam31'] = dataclasses.replace(p.modules['sturdecam31'], fps=float(fps))
    p.analysis.sim = {'traffic': 0, 'warmup_s': 0.1, 'port': 2300, 'tm_port': 8400}
    return p


def _box(x0, x1, y0, y1, z0, z1):
    from vantage.core.mesh import Mesh
    V = np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)], float)
    F = [[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4],
         [1, 5, 7], [1, 7, 3]]
    return Mesh(V, np.array(F))


def _box_project(pitch=40.0):
    """A body that is exactly the fake ego's box (the CARLA vehicle's), a fisheye 4 cm in front of its front face
    looking down: the planned body edge is the one the fake renders, so the drift indicator starts near zero."""
    from vantage.core.project import Camera, Project, Variant
    _, L, W, H, rear, _ = _fake().VEHICLES[BOX_BP]
    x0, x1 = -L / 2 - rear, L / 2 - rear                     # the rig's origin is the rear axle
    p = Project()
    p.name = 'Box rig'
    p.vehicle.mesh = _box(x0, x1, -W / 2, W / 2, 0.0, H)
    v = Variant(name='Box')
    p.variants = [v]
    p.active_uid = v.uid
    v.cameras = [Camera(tag='F', name='Front', x=x1 + 0.04, y=0.0, z=0.8, yaw=0.0, pitch=pitch, roll=0.0)]
    p.analysis.sim = {'traffic': 0, 'warmup_s': 0.1, 'vehicle': BOX_BP, 'port': 2300, 'tm_port': 8400}
    return p


@contextlib.contextmanager
def _session(p, tmp, args=(), scale=0.1, carla=None):
    """A live rig on the fake carla in this process: (live, client)."""
    from vantage.core import simrig as SR
    from vantage.sim import carla_bridge as CB
    from vantage.core import carla_live as CL
    from vantage.sim import live as LV
    fc = carla or _fake()
    rig = CL.build_rig(p, scale=scale)
    out = os.path.join(str(tmp), 'rig')
    SR.export(rig, p, out)
    doc, remaps = CB.load_rig(os.path.join(out, 'sim_rig.json'))
    a = LV.resolve_args(LV.parser().parse_args([os.path.join(out, 'sim_rig.json')] + list(args)), doc)
    if hasattr(fc, 'reset'):
        fc.reset()
    events = []
    live = LV.LiveRig(fc, doc, remaps, a, events.append)
    srv = LV.serve(live)
    th = threading.Thread(target=live.run, daemon=True)
    th.start()
    client = LV.LiveClient(srv.server_address[1])
    live.events, live.thread, live.port = events, th, srv.server_address[1]
    try:
        yield live, client
    finally:
        live.request_stop('test over')
        th.join(60)
        LV.close_server(srv, live)


@contextlib.contextmanager
def _process(p, tmp, args=(), scale=0.1):
    """The live server as the panel runs it, in its own process (python -m vantage.sim.live with the fake carla on
    PYTHONPATH): timing measured without this process's work in the way. Yields the client."""
    from vantage.core import simrig as SR
    from vantage.core import carla_live as CL
    from vantage.sim import live as LV
    rig = CL.build_rig(p, scale=scale)
    SR.export(rig, p, os.path.join(str(tmp), 'rig'))
    cmd, env = CL.command(sys.executable, os.path.join(str(tmp), 'rig', 'sim_rig.json'), 0,
                          ['--parent-pid', os.getpid()] + list(args), scope=False, extra_path=FAKE_DIR)
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=str(tmp))
    try:
        ev = LV.parse_event(proc.stdout.readline())
        assert ev and ev['event'] == 'listening', proc.stderr.read()[-2000:] if proc.poll() is not None else ev
        yield LV.LiveClient(ev['port'])
    finally:
        if proc.poll() is None:
            try:
                LV.LiveClient(ev['port']).stop()
            except Exception:                                       # noqa: BLE001
                pass
            try:
                proc.communicate(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()


def _wait(cond, timeout=60.0, step=0.05):
    t = time.time()
    while time.time() - t < timeout:
        if cond():
            return True
        time.sleep(step)
    return cond()


def _streaming(client, tags=None, timeout=60.0):
    def ok():
        s = client.status()
        return s['state'] == 'streaming' and all(c['served'] > 0 for t, c in s['cameras'].items()
                                                 if tags is None or t in tags)
    assert _wait(ok, timeout), client.status()


def _frame(client, tag='F', after=2):
    """The newest frame (BGR) after `after` more frames were served."""
    import cv2
    n0 = client.status()['cameras'][tag]['served']
    assert _wait(lambda: client.status()['cameras'][tag]['served'] >= n0 + after, 60)
    data = _open_url(f'http://127.0.0.1:{client.port}/{tag}.jpg', timeout=5).read()
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def _applied(client, tag, kind):
    return _wait(lambda: any(f['kind'] == kind and f.get('state') == 'applied'
                             for f in client.status()['cameras'][tag]['faults']), 30)


# ======================================================================= faults, mounts and the step (no server)
def test_faults_checked_normalised_and_described():
    from vantage.core import carla_live as CL
    from vantage.sim import live as LV
    tags = ['F', 'R']
    assert LV.parse_fault({'tag': 'F', 'deg': [0, 0.5, 0]}, tags) == {'tag': 'F', 'kind': 'rotation',
                                                                       'deg': [0.0, 0.5, 0.0]}
    assert LV.parse_fault({'tag': 'R', 'm': [0.01, 0, 0]}, tags)['kind'] == 'translation'
    assert LV.parse_fault({'tag': 'R', 'kind': 'drop'}, tags)['rate'] == 0.3
    assert LV.parse_fault({'tag': 'R', 'kind': 'delay'}, tags)['ms'] == 100.0
    assert LV.parse_fault({'tag': 'R', 'kind': 'clock_offset', 'ms': 5}, tags)['ms'] == 5.0
    assert LV.parse_fault({'tag': 'F', 'kind': 'freeze', 's': 2}, tags) == {'tag': 'F', 'kind': 'freeze', 's': 2.0}
    for bad, words in (({'tag': 'X', 'deg': [0, 1, 0]}, 'No camera'), ({'tag': 'F', 'kind': 'melt'}, 'Unknown'),
                       ({'tag': 'F', 'deg': [0, 1]}, 'three numbers'), ({'tag': 'F', 'deg': [0, math.nan, 0]}, 'deg'),
                       ({'tag': 'F', 'deg': [0, 31, 0]}, 'between'),
                       ({'tag': 'F', 'kind': 'drop', 'rate': 1.5}, 'rate'),
                       ({'tag': 'F', 'kind': 'delay', 'ms': -1}, 'ms'), ({'tag': 'F', 'm': [0, 0, 2.5]}, 'between'),
                       ({'tag': 'F', 'kind': 'freeze', 's': 'soon'}, 'number'), ([1, 2], 'JSON object'),
                       ({'tag': 'F'}, 'Unknown')):
        with pytest.raises(ValueError, match=words):
            LV.parse_fault(bad, tags)
    assert LV.describe_fault({'kind': 'rotation', 'deg': [0, 0.5, 0]}) == 'turned pitch +0.5°'
    assert LV.describe_fault({'kind': 'translation', 'm': [0.02, 0, 0], 's': 3}) == 'moved x +2 cm for 3 s'
    assert LV.describe_fault({'kind': 'drop', 'rate': 0.25}) == 'dropping 25 %'
    assert LV.describe_fault({'kind': 'delay', 'ms': 40}) == '40 ms late'


def test_a_turned_view_is_the_camera_turned_about_its_own_axes():
    """turned_view against independent constructions: the camera planned at the turned pose (yaw and pitch faults, as
    rot_zyx composes them), a point fixed in the camera, and the translation in the vehicle frame."""
    from vantage.core.mathutil import BODY_TO_OPTICAL, rot_zyx
    from vantage.sim import frames as FR
    from vantage.core import carla_live as CL
    from vantage.sim import live as LV
    rng = np.random.default_rng(11)

    def view(cam, F):
        R = rot_zyx(*cam)
        B = R @ BODY_TO_OPTICAL.T @ F @ BODY_TO_OPTICAL               # a view's body frame, as core.simrig makes it
        p, y, r = FR.carla_rotation(B)
        return {'location': [1.0, -0.5, 0.8], 'rotation': {'pitch': p, 'yaw': y, 'roll': r}}, B

    def B_of(v):
        r = v['rotation']
        return FR.vantage_matrix(r['pitch'], r['yaw'], r['roll'])
    for _ in range(30):
        yaw, pitch = rng.uniform(-170, 170), rng.uniform(-60, 60)
        F = rot_zyx(*rng.uniform(-90, 90, 3))
        v, B = view((yaw, pitch, 0.0), F)
        d = rng.uniform(-5, 5)
        pose = {'yaw': yaw, 'pitch': pitch, 'roll': 0.0}
        # a pitch fault on a camera without roll is the camera planned d further down
        assert np.allclose(B_of(LV.turned_view(v, pose, [0, d, 0])), view((yaw, pitch + d, 0.0), F)[1], atol=1e-6)
        # a yaw fault on a level camera is the camera planned turned by d
        lv, _ = view((yaw, 0.0, 0.0), F)
        assert np.allclose(B_of(LV.turned_view(lv, dict(pose, pitch=0.0), [d, 0, 0])),
                           view((yaw + d, 0.0, 0.0), F)[1], atol=1e-6)
        # any fault: a point fixed in the camera's frame lands at R·rot_zyx(deg)·q
        roll = rng.uniform(-30, 30)
        v, B = view((yaw, pitch, roll), F)
        deg = rng.uniform(-3, 3, 3)
        R = rot_zyx(yaw, pitch, roll)
        q = rng.normal(size=3)                                        # in the camera's body frame
        q_view = B.T @ R @ q                                          # the same point in the view's body frame
        B2 = B_of(LV.turned_view(v, {'yaw': yaw, 'pitch': pitch, 'roll': roll}, deg))
        assert np.allclose(B2 @ q_view, R @ rot_zyx(*deg) @ q, atol=1e-6)
        assert LV.turned_view(v, pose, deg)['location'] == v['location']    # about the optical centre
    moved = LV.turned_view(v, pose, None, [0.02, 0.01, -0.03])
    assert np.allclose(moved['location'], np.add(v['location'], FR.carla_location([0.02, 0.01, -0.03])))
    assert moved['rotation'] == {k: float(x) for k, x in v['rotation'].items()}


def test_live_step_is_a_whole_multiple_of_every_camera_rate():
    from vantage.core import carla_live as CL
    from vantage.sim import live as LV
    assert LV.live_tick_hz([30.0]) == 30.0
    assert LV.live_tick_hz([30.0, 25.0]) == 150.0
    assert LV.live_tick_hz([30.0, 20.0, 30.0]) == 60.0
    assert LV.live_tick_hz([7.5, 15.0]) == 15.0
    assert LV.live_tick_hz([]) == 10.0 and LV.live_tick_hz([0.0, 12.0]) == 12.0
    assert LV.live_tick_hz([30.0, 29.97]) == 30.0                    # no common step under 400 Hz: the fastest
    rig = CL.build_rig(_sample(('F', 'R')), scale=0.05)
    d = rig.doc
    assert [c['tag'] for c in d['cameras']] == ['F', 'R'] and not d['lidars'] and not d['radars']
    assert d['simulation']['tick_hz'] == 30.0 and all(c['capture']['every_ticks'] == 1 for c in d['cameras'])
    assert d['cameras'][0]['image']['width'] == 96 and d['cameras'][0]['image']['sensor_size'] == [1920, 1536]
    with pytest.raises(ValueError, match='No visible camera'):
        CL.build_rig(_sample(('F',)), cameras=['nope'])


# ======================================================================= the server against the fake carla
def test_mjpeg_streams_that_opencv_reads_and_the_http_api(tmp_path):
    import cv2
    from vantage.core import carla_live as CL
    from vantage.sim import live as LV
    with _session(_sample(('F', 'R')), tmp_path, ['--noise', '1']) as (live, client):
        _streaming(client)
        s = client.status()
        assert s['schema'] == LV.SCHEMA and s['tick_hz'] == 30.0 and s['town']
        c = s['cameras']['F']
        assert c['url'] == '/F.mjpg' and c['size'] == [1920, 1536] and c['render_size'] == [192, 154]
        assert c['views'] == 5 and c['latency_ms']['median'] > 0 and c['mount'] == {'deg': None, 'm': None}
        # OpenCV opens the stream as the Live rig page does (core.capture: cv2.VideoCapture(url))
        cap = cv2.VideoCapture(client.url('F'))
        assert cap.isOpened()
        got = []
        t = time.time()
        while len(got) < 5 and time.time() - t < 30:
            ok, f = cap.read()
            assert ok
            got.append(f)
        cap.release()
        assert got[-1].shape == (1536, 1920, 3)
        # the fisheye the right way up: sky above (blue over red), the ground and the car's front below
        top, bottom = got[-1][:150].reshape(-1, 3).mean(0), got[-1][-150:].reshape(-1, 3).mean(0)
        assert top[0] > top[2] + 40 and bottom[2] > bottom[0]
        assert len({f.tobytes() for f in got}) == len(got)          # noise: never two identical frames
        # the raw multipart stream: boundary, length, frame numbers increasing, served stamps on this clock
        with _open_url(f'http://127.0.0.1:{live.port}/R.mjpg', timeout=10) as r:
            assert r.headers['Content-Type'] == f'multipart/x-mixed-replace; boundary={LV.BOUNDARY}'
            frames = []
            for _ in range(3):
                assert r.readline().strip() == f'--{LV.BOUNDARY}'.encode()
                head = {}
                while True:
                    line = r.readline().strip()
                    if not line:
                        break
                    k, _, v = line.decode().partition(':')
                    head[k.strip().lower()] = v.strip()
                data = r.read(int(head['content-length']))
                assert r.readline() == b'\r\n' and data[:2] == b'\xff\xd8'
                assert abs(float(head['x-published']) - time.monotonic()) < 5
                frames.append(int(head['x-frame']))
        assert frames == sorted(frames) and len(set(frames)) == 3
        assert cv2.imdecode(np.frombuffer(_open_url(f'http://127.0.0.1:{live.port}/F.jpg').read(), np.uint8),
                            cv2.IMREAD_COLOR).shape == (1536, 1920, 3)
        assert '<img src="/F.mjpg"' in _open_url(f'http://127.0.0.1:{live.port}/').read().decode()

        def http_error(req):
            try:
                _open_url(req, timeout=5)
            except urllib.error.HTTPError as e:
                return e.code, json.loads(e.read())['error']
            return 200, ''
        code, err = http_error(f'http://127.0.0.1:{live.port}/X.mjpg')
        assert code == 404 and 'no such stream' in err
        assert http_error(urllib.request.Request(f'http://127.0.0.1:{live.port}/stop', data=b'{}',
                                                 method='POST'))[0] == 415   # not JSON: a web page cannot send it
        assert http_error(urllib.request.Request(f'http://127.0.0.1:{live.port}/status',
                                                 headers={'Host': 'evil.example:80'}))[0] == 403
        with pytest.raises(LV.LiveError, match='No camera'):
            client.fault({'tag': 'Q', 'deg': [0, 1, 0]})
        with pytest.raises(LV.LiveError, match='between'):
            client.fault({'tag': 'F', 'kind': 'delay', 'ms': 1e6})
        assert live.state == 'streaming'


def test_rig_health_reads_the_live_cameras_at_their_rate(tmp_path):
    """core.rig_health on the live server's URLs: each camera at its module's rate (10 fps here so the fake keeps real
    time), its sensor size, the cameras in step, no fault."""
    from vantage.core import rig_health as rh
    from vantage.core import carla_live as CL
    from vantage.sim import live as LV
    p = _sample(('F', 'R'), fps=10.0, lens=NARROW)
    with _process(p, tmp_path, scale=0.08) as client:
        _streaming(client)
        assert client.status()['tick_hz'] == 10.0 and client.status()['cameras']['F']['views'] == 1
        p.streams = {t: client.url(t) for t in ('F', 'R')}
        snap = rh.run_for(p, [c for c in p.variant.cameras if c.visible], 4.0)
        s = client.status()
    assert s['realtime_factor'] == pytest.approx(1.0, abs=0.1), s    # (the fake kept real time)
    for t in ('F', 'R'):
        c = snap['cameras'][t]
        assert c['kind'] == 'http' and c['size_ok'] and c['frames'] >= 30, c
        rate = (c['frames'] - 1) / c['seconds']                     # frames over the watched seconds
        assert rate == pytest.approx(10.0, rel=0.1) and rate == pytest.approx(s['cameras'][t]['fps'], rel=0.1), c
        assert c['fps'] == pytest.approx(10.0, rel=0.25) and c['dropped_share'] < 0.1, c   # (the last few averaged)
        assert not any('frozen' in r or 'stalled' in r for r in c['reasons']), c
    assert snap['rig']['sync_skew_ms'] < 15


def _set_transform(self, tr):
    """Actor.set_transform as CARLA 0.9.16 does it (measured on a server): an attached actor's transform relative to its
    parent, which it keeps following; a free actor's world transform."""
    if self.parent is not None:
        self._rel = tr._m()
    else:
        self._m_world = tr._m()


@pytest.mark.parametrize('how', ['in place', 'spawned again'])
def test_a_rotation_fault_is_the_camera_turned_and_moves_the_drift_indicator(tmp_path, monkeypatch, how):
    """On the box rig: the planned body edge is the fake's, so rig health's indicator starts at a quantum; turning the
    camera 1° and 2° down moves it about 9 and 17 sensor pixels (the lens's ~15 px per degree at its centre, less off
    axis); the 2° fault is, pixel for pixel, the camera planned 2° further down; clearing it gives the planned image
    back exactly. The views are turned in place with set_transform (CARLA's clients have it) or, without it, spawned
    again."""
    import cv2
    from vantage.core import rig_health as rh
    from vantage.core.analysis import Engine
    fc = _fake()
    if how == 'in place' and not hasattr(fc.Actor, 'set_transform'):
        monkeypatch.setattr(fc.Actor, 'set_transform', _set_transform, raising=False)
    elif how == 'spawned again':
        for cls in fc.Actor.__mro__:
            if 'set_transform' in vars(cls):
                monkeypatch.delattr(cls, 'set_transform')
        assert not hasattr(fc.Actor, 'set_transform')
    p = _box_project(40.0)
    ev = rh.expected_view(p, Engine(), p.variant.cameras[0])
    drift = lambda f: rh.alignment(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), ev)       # noqa: E731
    with _session(p, tmp_path / 'a', ['--still', '--noise', '0'], scale=0.25) as (live, client):
        _streaming(client)
        f0 = _frame(client)
        out = {}
        for d in (1.0, 2.0):
            f = client.fault({'tag': 'F', 'deg': [0, d, 0]})
            assert f['kind'] == 'rotation' and f['text'] == f'turned pitch +{d:g}°'
            assert _applied(client, 'F', 'rotation')
            out[d] = _frame(client, after=2)
        st = client.status()
        assert st['cameras']['F']['mount']['deg'] == [0.0, 2.0, 0.0] and st['cameras']['F']['remounts'] == 2
        assert st['cameras']['F']['mount_method'] == how and len(st['faults']) == 1    # the 2° replaced the 1°
        cams = [e for e in fc.LOG if e[0] == 'spawn' and e[2] == 'sensor.camera.rgb']
        assert len(cams) == (5 if how == 'in place' else 15)
        assert client.clear(tag='F')[0]['deg'] == [0.0, 2.0, 0.0]
        assert _wait(lambda: client.status()['cameras']['F']['mount']['deg'] is None, 30)
        back = _frame(client, after=2)
        log = live.fault_log()
    with _session(_box_project(42.0), tmp_path / 'b', ['--still', '--noise', '0'], scale=0.25) as (live2, client2):
        _streaming(client2)
        planned42 = _frame(client2)
    assert drift(f0) <= 3.0
    assert 6.0 <= drift(out[1.0]) <= 12.0 and 13.0 <= drift(out[2.0]) <= 21.0, (drift(out[1.0]), drift(out[2.0]))
    assert drift(out[2.0]) > drift(out[1.0]) > rh.HealthSettings().drift_warn_px - 6
    assert np.abs(out[2.0].astype(int) - planned42).mean() < 0.2
    assert np.abs(f0.astype(int) - planned42).mean() > 3.0
    assert np.array_equal(back, f0)
    assert [r['state'] for r in log] == ['replaced', 'cleared'] and all(r.get('applied_s') is not None for r in log)


def test_body_masks_are_carlas_own_car_and_make_the_drift_indicator_work(tmp_path):
    """The live server's body masks (CARLA's own car per camera, instance segmentation at the planned mount): on the box
    rig the planned body itself; on the sample, whose Tigor body is not the CARLA car's, the drift indicator measured
    against the project's body does not follow a 1° and 2° turn, while against CARLA's car it grows with it."""
    import cv2
    from vantage.core import rig_health as rh
    from vantage.core.analysis import Engine
    from vantage.core import carla_live as CL
    from vantage.sim import live as LV
    eng = Engine()

    def run(p, sub, tag):
        with _session(p, tmp_path / sub, ['--still', '--noise', '0'], scale=0.25) as (live, client):
            _streaming(client)
            c = client.status()['cameras'][tag]
            assert c['body_mask']['url'] == f'/{tag}.body.png' and c['body_mask']['size'] == c['render_size']
            ev_c = CL.body_view(client.body(tag), c['size'], ('carla', tag))
            ev_p = rh.expected_view(p, eng, p.variant.by_tag(tag))
            frames = [_frame(client, tag)]
            for d in (1.0, 2.0):
                client.fault({'tag': tag, 'deg': [0, d, 0]})
                assert _applied(client, tag, 'rotation')
                frames.append(_frame(client, tag, after=2))
        rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
        iou = (ev_c.body & ev_p.body).sum() / max((ev_c.body | ev_p.body).sum(), 1)
        return iou, [rh.alignment(f, ev_p) for f in rgb], [rh.alignment(f, ev_c) for f in rgb], ev_c
    iou, plan, carla, ev = run(_box_project(40.0), 'box', 'F')
    assert iou > 0.95 and ev.sensor == (1920, 1536) and len(ev.edge) > 300 and ev.contours
    assert carla[0] <= 3.0 and carla[1] - carla[0] >= 5.0 and carla[2] - carla[1] >= 5.0, carla
    iou, plan, carla, _ = run(_sample(('F',)), 'tigor', 'F')
    assert iou < 0.5                                                # the project's body is not CARLA's car
    assert carla[0] <= 3.0 and carla[1] - carla[0] >= 4.0 and carla[2] - carla[1] >= 4.0, carla
    assert not (plan[1] > plan[0] and plan[2] > plan[1]), plan       # (what the Live rig page measures today)
    with _session(_sample(('F',)), tmp_path / 'off', ['--no-body-masks'], scale=0.05) as (live, client):
        _streaming(client)
        assert client.status()['cameras']['F']['body_mask'] is None
        with pytest.raises(LV.LiveError, match='no body mask'):
            client.body('F')


def test_freeze_drop_and_delay_show_in_rig_health(tmp_path):
    from vantage.core import rig_health as rh
    p = _sample(('F', 'R'), fps=10.0, lens=NARROW)
    with _process(p, tmp_path, scale=0.08) as client:
        _streaming(client)
        p.streams = {t: client.url(t) for t in ('F', 'R')}
        mon = rh.RigMonitor()
        mon.configure(p, [c for c in p.variant.cameras if c.visible])
        mon.start()
        try:
            def settle(seconds):
                mon.reset_stats()
                t = time.time()
                while time.time() - t < seconds:
                    mon.poll()
                    time.sleep(0.05)
                return mon.snapshot()
            assert _wait(lambda: all(s.stats.frames for s in mon.streams.values()), 30)
            base = settle(2.5)
            assert all(base['cameras'][t]['status'] in ('ok', 'warning') for t in ('F', 'R')), base['cameras']
            assert base['rig']['sync_skew_ms'] < 15
            client.fault({'tag': 'F', 'kind': 'freeze'})
            snap = settle(2.0)
            assert snap['cameras']['F']['status'] == 'fault' and any('frozen' in r for r in
                                                                     snap['cameras']['F']['reasons'])
            assert snap['cameras']['R']['frozen_frames'] < 3 and snap['cameras']['F']['fps'] > 7
            client.clear(tag='F')
            client.fault({'tag': 'R', 'kind': 'drop', 'rate': 0.5})
            snap = settle(3.0)
            r = snap['cameras']['R']
            assert r['dropped_share'] > 0.2 and r['status'] == 'fault' and any('dropped' in x for x in r['reasons'])
            assert snap['cameras']['F']['frozen_frames'] < 3
            assert client.status()['cameras']['R']['dropped_by_fault'] > 5
            client.clear()
            client.fault({'tag': 'R', 'kind': 'delay', 'ms': 35})
            snap = settle(3.0)
            assert snap['rig']['sync_skew_ms'] == pytest.approx(35.0, abs=10.0), snap['rig']
            time.sleep(2.5)                                         # the server's 5 s window: delayed frames only
            lat = client.status()['cameras']
            assert lat['R']['latency_ms']['median'] - lat['F']['latency_ms']['median'] == pytest.approx(35.0, abs=8)
            client.clear()
        finally:
            mon.stop()


def test_a_session_for_the_batch_mode(tmp_path):
    """run_session, what the batch mode runs: the live server, rig health on its streams with a fault injected on
    time, the drift against CARLA's car, the files, the server stopped; and no CARLA server: a clear error."""
    from vantage.core import simrig as SR
    from vantage.core import carla_live as CL
    from vantage.sim import live as LV
    p = _sample(('F', 'R'), fps=10.0, lens=NARROW)
    res = CL.run_session(p, 3.0, sys.executable, str(tmp_path / 'run'), scale=0.08, extra_path=FAKE_DIR, scope=False,
                         check_server=False, faults=[(1.0, {'tag': 'R', 'kind': 'freeze'})])
    assert res['status'] == 'ok' and res['faults'][0]['kind'] == 'freeze' and 0.9 <= res['faults'][0]['at_s'] < 1.6
    cams = res['rig_health']['cameras']
    assert cams['R']['status'] == 'fault' and any('frozen' in r for r in cams['R']['reasons'])
    assert not any('frozen' in r for r in cams['F']['reasons']) and cams['F']['frames'] > 15
    assert cams['F']['drift_checked'] and cams['F']['drift_against'] == 'carla_body'
    assert res['live']['stop_reason'] == 'stop requested over HTTP' and res['live']['fault_log'][0]['tag'] == 'R'
    for f in ('rig/sim_rig.json', 'rig/sim_remap.npz', 'live.json', 'rig_health.json', 'live.log'):
        assert (tmp_path / 'run' / f).is_file(), f
    assert json.loads((tmp_path / 'run' / 'rig_health.json').read_text())['cameras']['R']['status'] == 'fault'
    with socket.socket() as sk:
        sk.bind(('127.0.0.1', 0))
        free = sk.getsockname()[1]                                   # nothing answers there
    with pytest.raises(RuntimeError, match='No CARLA server'):
        CL.run_session(p, 1.0, sys.executable, str(tmp_path / 'none'), settings=SR.SimSettings.from_dict(
            {'port': free}))


@pytest.mark.parametrize('how', ['in place', 'spawned again'])
def test_stop_tidies_up_carla_and_the_server(tmp_path, monkeypatch, how):
    fc = _fake()
    if how == 'in place' and not hasattr(fc.Actor, 'set_transform'):
        monkeypatch.setattr(fc.Actor, 'set_transform', _set_transform, raising=False)
    elif how == 'spawned again':
        for cls in fc.Actor.__mro__:
            if 'set_transform' in vars(cls):
                monkeypatch.delattr(cls, 'set_transform')
    p = _sample(('F',))
    p.analysis.sim['traffic'] = 2
    with _session(p, tmp_path, ['--report', str(tmp_path / 'live.json')], scale=0.05) as (live, client):
        _streaming(client)
        client.fault({'tag': 'F', 'deg': [0.5, 0, 0], 's': 0.3})        # a timed fault: re-mounted twice
        assert _wait(lambda: not client.status()['faults'], 30)
        assert _wait(lambda: client.status()['cameras']['F']['remounts'] == 2, 30)
        assert client.stop() == {'state': 'stopping'}
        live.thread.join(60)
        assert live.state == 'stopped' and live.stop_reason == 'stop requested over HTTP'
        rep = live.report()
    log = list(fc.LOG)
    spawned = {e[1] for e in log if e[0] == 'spawn'}
    destroyed = {e[1] for e in log if e[0] == 'destroy'}
    cams = [e for e in log if e[0] == 'spawn' and e[2] == 'sensor.camera.rgb']
    assert len(cams) == (5 if how == 'in place' else 15) and spawned == destroyed     # 5 views (3 mounts), all gone
    w = fc.STATE['world']
    assert not w._actors and w.applied[0] == (True, 1 / 30) and w.applied[-1] == (False, None)
    assert [e for e in log if e[0] == 'tm_sync'][-1] == ('tm_sync', False)
    with pytest.raises(OSError):
        socket.create_connection(('127.0.0.1', live.port), 1.0).close()
    f = rep['fault_log'][0]
    assert f['state'] == 'cleared' and f['ended_wall'] - f['since_wall'] >= 0.3 - 1e-3 and f['s'] == 0.3
    assert rep['traffic'] == 2 and rep['vehicle'] and rep['cameras']['F']['served'] > 0


def test_a_server_that_does_not_answer_fails_cleanly(tmp_path):
    from vantage.core import carla_live as CL
    from vantage.sim import live as LV

    class Down:
        class Client:
            def __init__(self, host, port):
                pass

            def set_timeout(self, s):
                pass

            def get_server_version(self):
                raise RuntimeError('time-out of 1000ms while waiting for the simulator')
    with _session(_sample(('F',)), tmp_path, scale=0.05, carla=Down) as (live, client):
        assert _wait(lambda: client.status()['state'] == 'failed', 10)
        s = client.status()
        assert 'time-out' in s['error'] and s['cameras']['F']['served'] == 0
        with pytest.raises(LV.LiveError, match='failed'):
            client.fault({'tag': 'F', 'kind': 'freeze'})
        assert any(e.get('state') == 'failed' for e in live.events)


def test_command_line_stops_when_asked_and_when_its_parent_exits(tmp_path):
    """python -m vantage.sim.live as the panel runs it (here with the fake carla on PYTHONPATH): the port it picked,
    streaming, a stop over HTTP, and a stop when --parent-pid is gone; the report written in both cases."""
    from vantage.core import simrig as SR
    from vantage.core import carla_live as CL
    from vantage.sim import live as LV
    p = _sample(('F',))
    rig = CL.build_rig(p, scale=0.05)
    SR.export(rig, p, str(tmp_path / 'rig'))
    for how in ('http', 'parent'):
        parent = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(6)']) if how == 'parent' else None
        rep = tmp_path / f'live-{how}.json'
        cmd, env = CL.command(sys.executable, str(tmp_path / 'rig' / 'sim_rig.json'), 0,
                              ['--report', str(rep), '--parent-pid', parent.pid if parent else os.getpid()],
                              scope=False, extra_path=FAKE_DIR)
        assert cmd[:4] == [sys.executable, '-m', 'vantage.sim.live', str(tmp_path / 'rig' / 'sim_rig.json')]
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                cwd=str(tmp_path))
        try:
            ev = LV.parse_event(proc.stdout.readline())
            assert ev['event'] == 'listening' and ev['port'] > 0 and ev['streams']['F'].endswith('/F.mjpg')
            client = LV.LiveClient(ev['port'])
            _streaming(client)
            if how == 'http':
                client.stop()
            else:
                parent.wait(30)
            out, err = proc.communicate(timeout=60)
        finally:
            if proc.poll() is None:
                proc.kill()
        assert proc.returncode == 0, err[-2000:]
        last = [LV.parse_event(x) for x in out.splitlines() if LV.parse_event(x)][-1]
        assert last['event'] == 'stopped' and last['state'] == 'stopped'
        doc = json.loads(rep.read_text())
        assert doc['schema'] == LV.SCHEMA and doc['cameras']['F']['served'] > 0
        assert doc['stop_reason'] == ('stop requested over HTTP' if how == 'http' else 'the app that started it exited')


# ======================================================================= against a real CARLA server
def _live_server():
    spec, py = os.environ.get('VANTAGE_CARLA', ''), os.environ.get('VANTAGE_CARLA_PYTHON', '')
    if not spec or not py:
        return None
    host, _, port = spec.partition(':')
    from vantage.sim import server as SV
    return (host or '127.0.0.1', int(port or 2000), py) if SV.port_open(host or '127.0.0.1', int(port or 2000)) \
        else None


carla_live = pytest.mark.skipif(_live_server() is None, reason='no CARLA server (set VANTAGE_CARLA=host:port and '
                                                               'VANTAGE_CARLA_PYTHON to CARLA\'s client Python)')


@carla_live
def test_live_on_a_real_carla_server(tmp_path):
    """The front camera live on the real server: rig health reads it, a 1° turn re-mounts it and the frames go on."""
    from vantage.core import rig_health as rh
    from vantage.core import simrig as SR
    from vantage.core import carla_live as CL
    from vantage.sim import live as LV
    host, port, py = _live_server()
    tm = int(os.environ.get('VANTAGE_CARLA_TM', port + 6100))
    p = _sample(('F',))
    rig = CL.build_rig(p, scale=0.25)
    SR.export(rig, p, str(tmp_path / 'rig'))
    cmd, env = CL.command(py, str(tmp_path / 'rig' / 'sim_rig.json'), 0,
                          ['--host', host, '--port', port, '--tm-port', tm, '--traffic', 10,
                           '--parent-pid', os.getpid()],
                          memory_gb=8.0)
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=str(tmp_path))
    try:
        ev = LV.parse_event(proc.stdout.readline())
        client = LV.LiveClient(ev['port'], timeout=10)
        _streaming(client, timeout=300)
        p.streams = {'F': client.url('F')}
        snap = rh.run_for(p, [p.variant.by_tag('F')], 8.0)
        assert snap['cameras']['F']['frames'] > 10 and snap['cameras']['F']['size_ok']
        client.fault({'tag': 'F', 'deg': [0, 1, 0]})
        assert _applied(client, 'F', 'rotation')
        n = client.status()['cameras']['F']['served']
        assert _wait(lambda: client.status()['cameras']['F']['served'] > n + 5, 60)
        print('real CARLA live:', json.dumps({k: v for k, v in client.status()['cameras']['F'].items()
                                              if k in ('fps', 'latency_ms', 'make_ms', 'tick_ms')}))
        client.stop()
        proc.wait(120)
    finally:
        if proc.poll() is None:
            proc.terminate()
            proc.wait(60)
    assert proc.returncode == 0


# ======================================================================= the panel
from test_gui import app, win, pump, settle, _wait as _gwait   # noqa: E402,F401  (fixtures)


def _open(win, name, cls):
    if name not in win._page_widgets:
        pg = cls(win.ctx)
        win.pages.addWidget(pg)
        win._page_widgets[name] = pg
    win.set_page(name)
    return win._page_widgets[name]


def test_live_panel_start_faults_and_stop(app, win, monkeypatch, tmp_path):
    """Driven like a user: Start live (a CARLA server already answering: here the fake carla), the streams pointed at
    the live server as one undoable edit, the Live rig page open and reading them with the fault window; back on the
    panel, rig health's verdict in its table turns to a fault when the stream is frozen and back when cleared; a
    rotation is re-mounted; Stop live tidies up, puts the streams back (undoable too) and keeps the session report."""
    from vantage import workdir
    from vantage.sim import server as SV
    from vantage.ui.workspaces import sim_live as SL
    monkeypatch.setattr(SV, 'port_open', lambda *a, **k: True)      # never a real server (nor the user's on 2000)
    st = win.store

    def setup(pj):
        v = next(v for v in pj.variants if v.sensors)
        pj.active_uid = v.uid
        pj.modules['sturdecam31'] = dataclasses.replace(pj.modules['sturdecam31'], fps=10.0)
        for c in v.cameras:
            c.lens_id = NARROW if c.tag == 'F' else c.lens_id        # R stays a fisheye (5 views)
        pj.analysis.sim = {'traffic': 0, 'warmup_s': 0.1, 'port': 2300, 'tm_port': 8400}
        pj.streams = {'R': 'synthetic'}
    st.edit(['variants', 'analysis', 'streams', 'library'], setup, 'live test setup')
    win.ctx.settings.setValue(SL.KEY_PYTHON, sys.executable)
    workdir.put(win.ctx.settings, str(tmp_path / 'work'))
    pump(app, 0.2)
    pg = _open(win, 'carla_live', lambda ctx: SL.SimLivePanel(ctx))
    pg.extra_pythonpath = FAKE_DIR
    assert set(pg.cam_boxes) == {c.tag for c in st.variant.cameras if c.visible} and pg.b_start.isEnabled()
    assert not pg.controls.b_freeze.isEnabled() and not pg.b_faults.isEnabled()
    pg.set_cameras(['F', 'R'])
    pg.scale.set_value(8.0)
    pg.scale.committed.emit(8.0)
    assert win.ctx.settings.value(SL.KEY_SCALE) == pytest.approx(0.08)
    n_undo = st.undo.count()
    pg.b_start.click()
    assert pg.state == 'starting' and not pg.b_start.isEnabled() and not pg.cam_boxes['F'].isEnabled()
    _gwait(app, lambda: pg.state in ('streaming', 'failed'), 180)
    assert pg.state == 'streaming', pg.status_line.text.text()
    urls = {t: f'http://127.0.0.1:{pg.port}/{t}.mjpg' for t in ('F', 'R')}
    assert {t: st.project.streams.get(t) for t in ('F', 'R')} == urls
    assert st.undo.count() == n_undo + 1 and 'CARLA live' in st.undo.undoText()
    # the Live rig page opened and reads the live streams; the fault window is up
    assert win.page_name() == 'rig'
    rig = win.page('rig')
    assert rig.running
    _gwait(app, lambda: rig.monitor.preview('F') is not None and rig.monitor.preview('R') is not None, 60)
    assert pg._fault_window is not None and pg._fault_window.isVisible()
    assert pg._fault_window.controls.b_freeze.isEnabled()
    # back on the panel: its own rig health table
    win.set_page('carla_live')
    pump(app, 0.3)
    assert not rig.running and pg.monitor.running
    col = {h: i for i, (h, _) in enumerate(SL.COLS)}

    def cell(tag, h):
        for r in range(pg.table.rowCount()):
            if pg.table.item(r, 0) and pg.table.item(r, 0).text() == tag:
                return pg.table.item(r, col[h]).text()
        return ''
    _gwait(app, lambda: cell('F', 'Health') in ('ok', 'warning') and cell('R', 'Health') in ('ok', 'warning'), 60)
    _gwait(app, lambda: cell('F', 'Served') not in ('', '—') and cell('F', 'Latency').endswith('ms'), 30)
    # freeze F with the panel's button: rig health calls it a fault, and recovers when cleared
    i = pg.controls.cam.findData('F')
    pg.controls.cam.setCurrentIndex(i)
    pg.controls.b_freeze.click()
    _gwait(app, lambda: cell('F', 'Fault') == 'frozen', 30)

    def tip(tag):
        return next(pg.table.item(r, col['Health']).toolTip() for r in range(pg.table.rowCount())
                    if pg.table.item(r, 0).text() == tag)
    _gwait(app, lambda: 'frozen' in tip('F'), 30)                    # ten identical frames: a fault
    assert cell('F', 'Health') == 'fault' and 'frozen' not in tip('R')     # (R's own verdict varies with the load)
    pg.controls.b_clear.click()
    _gwait(app, lambda: cell('F', 'Fault') == '', 30)
    _gwait(app, lambda: 'frozen' not in tip('F'), 30)
    # turn R 1° down: CARLA re-mounts it, the stream goes on, the drift indicator is computed on it
    pg.controls.cam.setCurrentIndex(pg.controls.cam.findData('R'))
    pg.controls.rotate()
    _gwait(app, lambda: (pg.live or {}).get('cameras', {}).get('R', {}).get('mount', {}).get('deg') == [0.0, 1.0, 0.0],
           30)
    _gwait(app, lambda: 'R' in pg.carla_views and 'F' in pg.carla_views, 30)  # CARLA's car: what drift compares
    _gwait(app, lambda: pg.monitor.streams['R'].drift_checked, 60)
    assert cell('R', 'Fault') == 'turned pitch +1°'
    h = pg.health()['cameras']['R']
    assert h['drift_checked'] and h['drift_against'] == 'carla_body'
    sm = pg.summary()
    assert sm['state'] == 'streaming' and sm['cameras']['R']['faults'] == ['turned pitch +1°']
    assert sm['cameras']['F']['size'] == [1920, 1536] and 'http' not in json.dumps(sm)
    # a bad fault is refused with the server's reason
    msgs = []
    win.ctx.message.connect(lambda t, ms: msgs.append(t))
    pg.inject({'tag': 'F', 'kind': 'delay', 'ms': 99999})
    _gwait(app, lambda: any('between' in m for m in msgs), 20)
    # Stop: CARLA tidied up, the streams put back as one more undoable edit, the session report kept
    proc = pg.proc
    pg.b_stop.click()
    assert pg.state == 'stopping' and st.project.streams == {'R': 'synthetic'}
    assert not pg._fault_window.isVisible()
    _gwait(app, lambda: pg.state == 'off', 120)
    assert proc.returncode == 0 and pg.proc is None
    assert st.undo.count() == n_undo + 2 and 'put back' in st.undo.undoText()
    assert pg.report and pg.report['stop_reason'] == 'stop requested over HTTP'
    assert [f['kind'] for f in pg.report['fault_log']] == ['freeze', 'rotation']
    assert os.path.isfile(os.path.join(pg.folder, 'rig', 'sim_rig.json'))
    assert 'Stopped' in pg.status_line.text.text() and pg.b_start.isEnabled()
    st.undo.undo()
    assert st.project.streams['F'] == urls['F']
    st.undo.undo()
    assert st.project.streams == {'R': 'synthetic'}
