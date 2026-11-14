"""The edge watchdog (vantage.edge, vantage.core.edge_daemon): its configuration, the knock and health checks on
recordings with faults written in, every alert channel against local fakes, the service unit, a footprint without Qt,
and the page."""
import http.server
import json
import math
import os
import socket
import subprocess
import sys
import threading
import time
import types
import urllib.request

import numpy as np
import pytest

from vantage.core import edge_daemon as ED
from vantage.core import samples
from vantage.edge import config as CF
from vantage.edge.knock import StepDetector, angles_about, bucket, translation_from_rays

CARLA = os.path.join(os.environ['VANTAGE_CARLA_DRIVE'], 'nuscenes') if os.environ.get('VANTAGE_CARLA_DRIVE') else ''
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope='module')
def proj():
    p = samples.tigor_sample_project()
    p.active_uid = next(v for v in p.variants if v.sensors).uid
    return p


@pytest.fixture(scope='module')
def cfg(proj):
    return ED.export_config(proj)


def _small_cfg(cfg, tags=('ML', 'F'), fps=None):
    c = json.loads(json.dumps(cfg))
    c['cameras'] = [x for x in c['cameras'] if x['tag'] in tags]
    for x in c['cameras']:
        if fps:
            x['fps'] = fps
    c['alerts']['http']['port'] = 0
    c['alerts']['log']['path'] = ''
    return c


# ---------------------------------------------------------------------------------------------- configuration
def test_config_carries_the_design(proj, cfg):
    var = proj.variant
    assert cfg['schema'] == 'vantage.edge_config/1'
    assert [c['tag'] for c in cfg['cameras']] == [c.tag for c in var.cameras if c.visible]
    # every camera of the sample has a GMSL port: its default source is that port's V4L2 device
    assert all(c['source'].startswith('/dev/video') and c['source_from'].startswith('GMSL port')
               for c in cfg['cameras'])
    assert set(cfg['planned_offsets']) == {c.tag for c in var.cameras}         # the sync plan's offsets
    assert sum('body_mask' in c for c in cfg['cameras']) >= 4                  # the plan sees the body
    doc = json.loads(ED.config_json(cfg))
    back = CF.load(doc)
    assert back['knock']['threshold_deg'] == 1.0 and back['alerts']['http']['port'] == 8088
    # the radial lens table reproduces Vantage's lens model
    from vantage.core.lens import resolve
    c = var.cameras[2]
    intr = resolve(proj.module_for(c), proj.lens_for(c))
    lens = CF.RadialLens(next(x for x in cfg['cameras'] if x['tag'] == c.tag)['lens'])
    uu, vv = np.meshgrid(np.linspace(50, intr.width - 50, 15), np.linspace(50, intr.height - 50, 12))
    d_ref, th_ref = intr.unproject(uu.ravel() + 0.5, vv.ravel() + 0.5)        # sensor coordinates: centre at +0.5
    d, th = lens.unproject(uu.ravel(), vv.ravel())
    ok = th_ref < intr.max_theta - 0.02
    ang = np.degrees(np.arccos(np.clip(np.sum(d[ok] * d_ref[ok], 1), -1, 1)))
    assert ang.max() < 0.002
    u, v, inside = lens.project(d[ok])
    assert inside.all() and np.abs(u - uu.ravel()[ok]).max() < 0.01 and np.abs(v - vv.ravel()[ok]).max() < 0.01
    # a stream at half the size scales
    d2, _ = lens.unproject(uu.ravel()[ok] / 2 - 0.25, vv.ravel()[ok] / 2 - 0.25, intr.width // 2, intr.height // 2)
    assert np.abs(d2 - d[ok]).max() < 1e-9


def test_settings_tolerant_and_sources(proj):
    st = ED.EdgeSettings.from_dict({'knock_deg': 'x', 'analysis_hz': 1e9, 'http_port': -5, 'gpio_backend': 'laser',
                                    'webhook_url': 'ftp://nope', 'sources': {'F': ' rtsp://cam/f ', 'R': ''},
                                    'health': {'fps_warn': 0.8, 'bogus': 1}, 'knock_enabled': 'yes'})
    assert st.knock_deg == 1.0 and st.analysis_hz == 30.0 and st.http_port == 0
    assert st.gpio_backend == 'none' and st.webhook_url == '' and st.sources == {'F': 'rtsp://cam/f'}
    assert st.health == {'fps_warn': 0.8} and st.knock_enabled is True
    assert ED.EdgeSettings.from_dict(None) == ED.EdgeSettings()
    assert ED.EdgeSettings.from_dict(st.to_dict()) == st
    p = proj.snapshot()
    p.analysis.edge_daemon = st.to_dict()
    p.streams = {'R': 'rtsp://cam/r', 'ML': 'synthetic:30'}
    c = {x['tag']: x for x in ED.export_config(p, body_masks=False)['cameras']}
    assert c['F']['source'] == 'rtsp://cam/f' and c['F']['source_from'] == 'settings'
    assert c['R']['source'] == 'rtsp://cam/r' and c['R']['source_from'] == 'live view'
    assert c['ML']['source'].startswith('/dev/video')            # a synthetic preview is no source on the car


def test_config_load_rejects_broken():
    with pytest.raises(ValueError):
        CF.load({'schema': 'vantage.other/1', 'cameras': [{'tag': 'F'}]})
    with pytest.raises(ValueError):
        CF.load({'cameras': []})
    with pytest.raises(ValueError):
        CF.load({'cameras': [{'tag': 'F'}, {'tag': 'F'}]})
    with pytest.raises(ValueError):
        CF.load({'cameras': [{'tag': 'F', 'lens': {'width': 10, 'height': 10, 'cx': 5, 'cy': 5,
                                                   'theta': [0, 0.1, 0.05], 'r_px': [0, 1, 2]}}]})
    c = CF.load({'cameras': [{'tag': 'F', 'source': '/dev/video0', 'fps': 'junk'}],
                 'alerts': {'raise_s': -4, 'mqtt': {'port': 'x', 'host': 'broker'}}, 'knock': {'z_min': 'bad'}})
    assert c['cameras'][0]['fps'] == 30.0 and c['alerts']['raise_s'] == 0.0
    assert c['alerts']['mqtt'] == {'host': 'broker', 'port': 1883, 'topic': 'vantage/edge', 'user': '',
                                   'keepalive_s': 60}
    assert c['knock']['z_min'] == 4.0


def test_systemd_unit_and_install_files(proj, tmp_path):
    st = ED.EdgeSettings(memory_mb=300, cpu_percent=60, python='/opt/py/bin/python3', config_path='/etc/v/e.json')
    u = ED.systemd_unit(st, 'rig A')
    for line in ('ExecStart=/opt/py/bin/python3 -m vantage.edge --config /etc/v/e.json', 'NoNewPrivileges=yes',
                 'ProtectSystem=strict', 'MemoryMax=300M', 'CPUQuota=60%', 'SupplementaryGroups=video gpio',
                 'Restart=on-failure', 'ReadWritePaths=/var/log/vantage', 'LogsDirectory=vantage',
                 'WantedBy=multi-user.target', 'User=vantage'):
        assert line in u.splitlines(), line
    paths = ED.export_install(proj, str(tmp_path))
    assert sorted(os.path.basename(p) for p in paths) == ['README.md', 'edge.json', 'vantage-edge.service']
    assert CF.load(str(tmp_path / 'edge.json'))['rig'].startswith(proj.name)
    assert 'systemctl enable --now vantage-edge' in (tmp_path / 'README.md').read_text()


def test_real_systemd_unit_verifies(tmp_path):
    exe = '/usr/bin/systemd-analyze'
    if not os.path.exists(exe):
        pytest.skip('systemd-analyze not installed')
    st = ED.EdgeSettings(python=sys.executable, log_path=str(tmp_path / 'log' / 'edge.log'),
                         service_user=os.environ.get('USER', 'root'))
    unit = tmp_path / 'vantage-edge.service'
    unit.write_text(ED.systemd_unit(st))
    r = subprocess.run([exe, 'verify', '--user', str(unit)], capture_output=True, text=True, timeout=60)
    bad = [l for l in (r.stderr + r.stdout).splitlines() if 'vantage-edge.service' in l and
           ('Unknown' in l or 'Invalid' in l or 'Failed to parse' in l or 'bad-setting' in l)]
    assert not bad, bad


# ---------------------------------------------------------------------------------------------- the maths
def test_translation_from_rays_recovers_motion():
    rng = np.random.default_rng(3)
    for share, p95, worst in ((0.0, 0.5, 1.0), (0.15, 1.0, 2.0)):
        errs = []
        for k in range(40):
            d = rng.normal(size=3)
            d[2] = abs(d[2]) + 0.2
            d /= np.linalg.norm(d)
            P = np.column_stack([rng.uniform(-15, 15, 250), rng.uniform(-3, 3, 250), rng.uniform(3, 60, 250)])
            X2 = P - 0.7 * d
            keep = X2[:, 2] > 1
            r1 = P[keep] / np.linalg.norm(P[keep], axis=1, keepdims=True)
            r2 = X2[keep] / np.linalg.norm(X2[keep], axis=1, keepdims=True) + rng.normal(0, 0.3 / 300, X2[keep].shape)
            bad = rng.uniform(size=len(r2)) < share                          # mistracks and moving traffic
            r2[bad] = r2[bad] + rng.normal(0, 0.05, (bad.sum(), 3))
            r2 /= np.linalg.norm(r2, axis=1, keepdims=True)
            out = translation_from_rays(r1, r2, thresh=1 / 300, seed=k)
            errs.append(math.degrees(math.acos(min(1.0, float(out[0] @ d)))))   # the sign comes out right too
            assert out[2] > 0.7
        assert np.percentile(errs, 95) < p95 and max(errs) < worst, (share, np.percentile(errs, 95), max(errs))
    assert translation_from_rays(np.eye(3)[:3], np.eye(3)[:3]) is None


def test_step_detector_and_angles():
    rng = np.random.default_rng(1)
    sd = StepDetector(baseline_s=3, window_s=2, hold_s=1, threshold=1.0, z_min=4)
    t = np.arange(0, 20, 0.2)
    first = None
    for ti in t:
        v = rng.normal(0, 0.4, 2) + (np.array([1.5, 0.0]) if ti >= 10 else 0)
        sd.add(ti, v)
        if sd.knock and first is None:
            first = sd.knock['t']
    assert first is not None and 10.0 < first < 13.5
    assert abs(sd.last['size'] - 1.5) < 0.4
    quiet = StepDetector(3, 2, 1, 1.0, 4)
    for ti in t:
        quiet.add(ti, rng.normal(0, 0.4, 2))
    assert quiet.knock is None and quiet.state == 'ok'
    # an excursion shorter than the hold is not a knock
    ex = StepDetector(3, 1, 2, 1.0, 4)
    for ti in t:
        ex.add(ti, np.array([3.0, 0]) if 8 <= ti < 9 else rng.normal(0, 0.2, 2))
    assert ex.knock is None
    ref = np.array([0.0, 0.0, 1.0])
    a = angles_about(np.array([math.sin(math.radians(2)), 0, math.cos(math.radians(2))]), ref)
    assert abs(abs(a).max() - 2.0) < 1e-9 and abs(a).min() < 1e-9
    p = np.array([[x, y] for y in range(0, 80, 2) for x in range(0, 100, 2)], float)
    kept = bucket(p, (80, 100), rows=4, cols=5, per=3)
    assert len(kept) == 4 * 5 * 3


def test_knock_injection_is_a_rotation(cfg):
    from vantage.edge.inject import BODY_TO_OPTICAL, Injector, rot_zyx
    from vantage.core.mathutil import rot_zyx as rot_ref
    assert np.allclose(rot_zyx(10, -20, 5), rot_ref(10, -20, 5))
    c = next(x for x in cfg['cameras'] if x['tag'] == 'F')
    lens = CF.RadialLens(c['lens'])
    w, h = 192, 154
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.stack([(xx * 7 + yy * 3) % 251, (xx * 3) % 253, (yy * 5) % 247], -1).astype(np.uint8)
    inj = Injector([{'kind': 'knock', 'camera': 'F', 'at_s': 1.0, 'pitch': 1.5}], {'F': c['lens']})
    assert inj('F', 0.0, img)[0][1] is img                                  # before the knock: untouched
    out = inj('F', 1.5, img)[0][1]
    mx, my = next(iter(inj.maps.values()))
    M = BODY_TO_OPTICAL @ rot_zyx(0, 1.5, 0) @ BODY_TO_OPTICAL.T
    for (u, v) in ((96, 77), (40, 40), (150, 120)):
        d, _ = lens.unproject([u], [v], w, h)
        u0, v0, _ = lens.project(d @ M.T, w, h)
        assert abs(mx[v, u] - u0[0]) < 1e-3 and abs(my[v, u] - v0[0]) < 1e-3
    assert out.shape == img.shape and not np.array_equal(out, img)


# ---------------------------------------------------------------------------------------------- dry runs
FAULTS = [{'kind': 'knock', 'camera': 'ML', 'at_s': 6.0, 'yaw': 2.5},
          {'kind': 'freeze', 'camera': 'F', 'at_s': 5.0, 'dur_s': 3.0},
          {'kind': 'stall', 'camera': 'QR', 'at_s': 7.0, 'dur_s': 0.0},
          {'kind': 'dark', 'camera': 'R', 'at_s': 4.0, 'dur_s': 4.0},
          {'kind': 'drop', 'camera': 'MR', 'at_s': 4.0, 'dur_s': 5.0, 'keep_every': 3}]


def test_dry_run_catches_every_fault_on_a_synthetic_drive(proj):
    res = ED.dry_run(proj, None, FAULTS, seconds=12.0, fps=10.0, width=320)
    m = res.metrics
    assert m['faults'] == 5 and m['caught'] == 5, res.faults
    assert m['false_alarms'] == 0, [(a['camera'], a['check'], a['message']) for a in res.false_alarms]
    by = {f['kind']: f for f in res.faults}
    assert by['freeze']['check'] == 'frozen' and by['stall']['check'] == 'stall' and by['dark']['check'] == 'dark'
    assert by['drop']['check'] in ('drops', 'frame_rate')
    assert by['knock']['check'] == 'knock' and by['knock']['delay_s'] < 5.5
    assert abs(by['knock']['size_error_deg']) < 1.2
    assert all(f['delay_s'] < 5.5 for f in res.faults)
    assert ED.exit_code(res) == 0
    s = ED.summary(res)
    json.dumps(s)
    doc = ED.to_doc(res, proj)
    assert doc['schema'] == ED.SCHEMA and doc['frame'].startswith('vehicle')
    assert 'false alarm' not in ED.to_csv(res)


def test_dry_run_clean_drive_is_quiet(proj, tmp_path):
    res = ED.dry_run(proj, None, [], seconds=10.0, fps=10.0, width=320, seed=4)
    assert res.report['alerts'] == [] and res.metrics['false_alarms'] == 0
    assert all(c['status'] == 'ok' for c in res.report['cameras'].values())
    assert all(c['knock']['state'] == 'ok' for c in res.report['cameras'].values())
    paths = ED.export(res, proj, str(tmp_path))
    names = sorted(os.path.basename(p) for p in paths)
    assert names == ['README.md', 'edge.json', 'edge_alerts.csv', 'edge_dry_run.json', 'vantage-edge.service']


def _write_recording(cfg, root, seconds=6.0, fps=10.0, width=192, kinds=('images', 'avi')):
    import cv2
    rec = ED.synthetic_drive(cfg, seconds, fps, 20.0, width, seed=2)
    out = {}
    for tag, s in rec.items():
        if 'images' in kinds:
            d = root / 'images' / f'CAM_{tag}'
            d.mkdir(parents=True, exist_ok=True)
            for t, f in s.frames():
                us = 1767268800000000 + int(round(t * 1e6)) + (3000 if tag == 'F' else 0)
                cv2.imwrite(str(d / f'drive__CAM_{tag}__{us}.png'), f)
        if 'avi' in kinds:
            (root / 'avi').mkdir(exist_ok=True)
            p = str(root / 'avi' / f'{tag}.avi')
            w = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, round(width * 1536 / 1920)))
            for t, f in s.frames():
                w.write(f)
            w.release()
            out[tag] = p
    return out


def test_recordings_folders_videos_and_bags(cfg, tmp_path):
    from vantage.core import rosbag as RB
    from vantage.edge.sources import open_recording
    c = _small_cfg(cfg, ('ML', 'F'), fps=10.0)
    _write_recording(c, tmp_path)
    s, kind = open_recording(str(tmp_path / 'images'), ['ML', 'F', 'QR'])
    assert kind == 'nuScenes folder' or kind == 'image folders'
    assert sorted(s) == ['F', 'ML'] and s['F'].stamped and len(s['F']) == 60
    assert abs(s['F'].times[0] - 0.003) < 1e-9 and s['ML'].times[0] == 0.0            # one clock for all
    v, kind = open_recording(str(tmp_path / 'avi'), ['ML', 'F'])
    assert kind == 'video files' and sorted(v) == ['F', 'ML'] and abs(v['F'].fps - 10.0) < 1e-6
    # a ROS 2 bag of compressed images, topics named after the cameras
    import cv2
    bag = tmp_path / 'bag'
    with RB.BagWriter(str(bag), compression='none') as w:
        for tag in ('ml', 'f'):
            w.add_topic(f'/{tag}/image_raw/compressed', 'sensor_msgs/msg/CompressedImage')
        for t, f in s['ML'].frames():
            t_ns = 1_000_000_000 + int(t * 1e9)
            for tag in ('ml', 'f'):
                ok, buf = cv2.imencode('.jpg', f)
                w.write(f'/{tag}/image_raw/compressed', t_ns, RB.compressed_image(buf.tobytes(), t_ns, tag))
    b, kind = open_recording(str(bag), ['ML', 'F', 'R'])
    assert kind == 'ROS 2 bag' and sorted(b) == ['F', 'ML']
    frames = list(b['ML'].frames())
    assert len(frames) == 60 and abs(frames[1][0] - 0.1) < 1e-6 and frames[0][1].shape[2] == 3
    rep = __import__('vantage.edge.daemon', fromlist=['dry_run']).dry_run(
        c, str(bag), [{'kind': 'freeze', 'camera': 'F', 'at_s': 2.0, 'dur_s': 3.5}])
    assert [a['check'] for a in rep['alerts'] if a['camera'] == 'F'] == ['frozen']


def test_live_file_streams_freeze_fires(cfg, tmp_path):
    """The watchdog reading video files as live streams (capture.LiveSource, played at their own rate)."""
    import cv2
    from vantage.edge.daemon import Watchdog
    from vantage.edge.inject import Injector
    c = _small_cfg(cfg, ('ML', 'F'), fps=20.0)
    rec = ED.synthetic_drive(c, 4.0, 20.0, 20.0, 160, seed=1)
    inj = Injector([{'kind': 'freeze', 'camera': 'F', 'at_s': 1.0, 'dur_s': 3.0}])
    for x in c['cameras']:
        p = str(tmp_path / f'{x["tag"]}.avi')
        w = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*'MJPG'), 20.0, (160, 128))
        for t, f in rec[x['tag']].frames():
            for _, g in inj(x['tag'], t, f):
                w.write(g)
        w.release()
        x['source'], x['width'], x['height'] = p, 160, 128
    c['knock']['enabled'] = False
    c['alerts']['raise_s'] = 0.5
    c['alerts']['grace_s'] = 0.5
    wd = Watchdog(c, notify=False)
    wd.run(seconds=3.6, period=0.1)
    rep = wd.report()
    got = {(a['camera'], a['check']) for a in rep['alerts']}
    assert ('F', 'frozen') in got, rep['alerts']
    assert not any(cam == 'ML' for cam, _ in got), rep['alerts']
    assert rep['cameras']['ML']['frames'] >= 40
    assert rep['process']['peak_rss_mb'] is None or rep['process']['peak_rss_mb'] > 10


# ---------------------------------------------------------------------------------------------- alert channels
class _Broker:
    """A fake MQTT broker: accepts, answers CONNECT with CONNACK, records PUBLISH packets; closes the first connection
    after its first PUBLISH (the client must reconnect)."""

    def __init__(self):
        from vantage.edge.mqtt import read_packet
        self.read_packet = read_packet
        self.sock = socket.socket()
        self.sock.bind(('127.0.0.1', 0))
        self.sock.listen(4)
        self.port = self.sock.getsockname()[1]
        self.got, self.connects = [], []
        threading.Thread(target=self.run, daemon=True).start()

    def run(self):
        n = 0
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            n += 1
            try:
                h, body = self.read_packet(conn)
                self.connects.append(body)
                conn.sendall(b'\x20\x02\x00\x00')
                while True:
                    h, body = self.read_packet(conn)
                    if h >> 4 == 3:
                        tl = int.from_bytes(body[:2], 'big')
                        self.got.append((body[2:2 + tl].decode(), body[2 + tl:], bool(h & 1)))
                        if n == 1:
                            break
                    elif h >> 4 == 14:
                        break
            except (ConnectionError, OSError):
                pass
            conn.close()


def test_mqtt_publisher_against_a_fake_broker(monkeypatch):
    from vantage.edge.mqtt import MqttPublisher, connect_packet, _varlen
    assert _varlen(0) == b'\x00' and _varlen(127) == b'\x7f' and _varlen(128) == b'\x80\x01' and \
        _varlen(16383) == b'\xff\x7f'
    pkt = connect_packet('vantage-edge-x', 30, 'u', 'p')
    assert pkt[0] == 0x10 and b'MQTT' in pkt and pkt[pkt.index(b'MQTT') + 5] == 0xC2
    monkeypatch.setenv('VANTAGE_EDGE_MQTT_PASSWORD', 'secret')
    b = _Broker()
    m = MqttPublisher('127.0.0.1', b.port, 'vantage-edge-test', user='rig')
    m.publish('vantage/rig/alerts', '{"a": 1}')
    t = time.time()
    while not b.got and time.time() - t < 10:
        time.sleep(0.02)
    # the broker hung up after that message; QoS 0 may lose what is sent into the dead connection, so keep
    # publishing until one arrives on the second connection (the publisher noticed and reconnected)
    k = 0
    while len(b.connects) < 2 or not any(tp.endswith('/status') for tp, _, _ in b.got):
        k += 1
        m.publish('vantage/rig/status', b'{"s": %d}' % k, retain=True)
        time.sleep(0.2)
        assert time.time() - t < 30, (b.got, b.connects, m.error)
    m.close()
    assert ('vantage/rig/alerts', b'{"a": 1}', False) in b.got
    assert any(tp == 'vantage/rig/status' and retain for tp, _, retain in b.got)
    assert len(b.connects) >= 2 and m.error == 'stopped'
    assert b'secret' in b.connects[0] and b'vantage-edge-test' in b.connects[0]
    b.sock.close()


def test_notifier_webhook_log_gpio(tmp_path):
    from vantage.edge.alerts import AlertBook, Notifier
    got = []

    class H(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            got.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            self.send_response(204)
            self.end_headers()

        def log_message(self, *a):
            pass
    srv = http.server.ThreadingHTTPServer(('127.0.0.1', 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    al = CF.load({'cameras': [{'tag': 'F'}], 'alerts': {
        'webhook': {'url': f'http://127.0.0.1:{srv.server_address[1]}/hook'},
        'log': {'path': str(tmp_path / 'logs' / 'edge.log')},
        'gpio': {'backend': 'file', 'path': str(tmp_path / 'gpio'), 'pattern': 'steady'}}})['alerts']
    n = Notifier(al, 'rig A')
    book = AlertBook(raise_s=1.0, clear_s=2.0)
    assert book.update(0.0, {('F', 'frozen'): ('fault', 'frozen: 12 identical frames')}) == []
    ev = book.update(1.2, {('F', 'frozen'): ('fault', 'frozen: 14 identical frames')})
    assert [e['event'] for e in ev] == ['raised'] and ev[0]['id'] == 1
    n.send(ev, book.any_fault())
    t = time.time()
    while (not got or (tmp_path / 'gpio').read_text().strip() != '1') and time.time() - t < 5:
        time.sleep(0.05)
    assert got and got[0]['camera'] == 'F' and got[0]['event'] == 'raised' and got[0]['rig'] == 'rig A'
    assert (tmp_path / 'gpio').read_text().strip() == '1'
    assert book.update(2.0, {}) == []                              # gone for less than clear_s
    ev = book.update(3.5, {})
    assert [e['event'] for e in ev] == ['cleared']
    n.send(ev, book.any_fault())
    t = time.time()
    while (tmp_path / 'gpio').read_text().strip() != '0' and time.time() - t < 5:
        time.sleep(0.05)
    assert (tmp_path / 'gpio').read_text().strip() == '0'
    ch = n.channels()
    n.close()
    srv.shutdown()
    log = (tmp_path / 'logs' / 'edge.log').read_text()
    assert 'ALERT F frozen fault' in log and 'CLEAR F frozen' in log
    assert ch['webhook']['sent'] >= 1 and ch['gpio']['backend'] == 'file'
    # a blip shorter than raise_s never raises
    b2 = AlertBook(1.0, 2.0)
    assert b2.update(0.0, {('F', 'dark'): ('warning', 'dark')}) == []
    assert b2.update(0.6, {}) == [] and b2.update(1.2, {('F', 'dark'): ('warning', 'dark')}) == []
    assert b2.counter == 0


def test_http_status_endpoints(cfg):
    from vantage.edge import status_http
    from vantage.edge.daemon import Watchdog
    c = _small_cfg(cfg, ('ML', 'F'), fps=10.0)
    for x in c['cameras']:
        x['width'], x['height'] = 160, 128
    c['alerts']['grace_s'] = 0.0
    wd = Watchdog(c, notify=False, replay=True)
    rec = ED.synthetic_drive(c, 3.0, 10.0, 20.0, 160, seed=5)
    from vantage.edge.sources import merged
    wd.replay_frames(merged(rec), end_s=0.5)
    srv = status_http.serve(wd, '127.0.0.1', 0)
    base = f'http://127.0.0.1:{srv.server_address[1]}'
    try:
        get = lambda p: urllib.request.urlopen(base + p, timeout=5)
        st = json.loads(get('/status.json').read())
        assert st['schema'] == 'vantage.edge_status/1' and set(st['cameras']) == {'ML', 'F'}
        assert st['cameras']['F']['frames'] == 30 and 'knock' in st['cameras']['F']
        assert json.loads(get('/alerts.json').read())['alerts'] == []
        met = get('/metrics').read().decode()
        assert 'vantage_edge_fps{rig=' in met and 'camera="ML"' in met
        assert 'vantage_edge_alerts{severity="fault"} 0' in met
        page = get('/').read().decode()
        assert '<table>' in page and 'ML' in page
        assert get('/healthz').read() == b'ok\n'
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(urllib.request.Request(base + '/status.json', data=b'{}', method='POST'), timeout=5)
        assert e.value.code == 405
    finally:
        srv.shutdown()
        srv.server_close()


def test_gpio_backends_degrade():
    from vantage.edge import gpio
    msgs = []
    assert gpio.make({'backend': 'none'}) is None
    assert gpio.make({'backend': 'gpiod', 'line': -1}, log=msgs.append) is None and msgs
    assert gpio.make({'backend': 'file'}, log=msgs.append) is None
    assert gpio.make({'backend': 'auto', 'line': -1}) is None
    assert set(gpio.available()) == {'gpiod_module', 'gpioset', 'chips'}


def test_ros2_source_with_a_fake_rclpy(monkeypatch):
    """The ROS 2 glue against a stand-in for rclpy and sensor_msgs (no ROS here)."""
    import cv2
    from vantage.edge.sources import Ros2Source
    subs = []

    class Node:
        def create_subscription(self, typ, topic, cb, qos):
            subs.append((typ, topic, cb))

        def destroy_node(self):
            pass
    img = np.zeros((48, 64, 3), np.uint8)
    img[10:20, 5:30] = (255, 0, 0)                      # RGB red
    ok, buf = cv2.imencode('.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    msg = types.SimpleNamespace(format='png', data=buf.tobytes())
    state = {'ok': False}

    def spin_once(node, timeout_sec=0.1):
        if subs and not state.get('sent'):
            state['sent'] = True
            subs[0][2](msg)
        time.sleep(0.01)
    rclpy = types.ModuleType('rclpy')
    rclpy.ok = lambda: state['ok']
    rclpy.init = lambda: state.update(ok=True)
    rclpy.create_node = lambda name: Node()
    rclpy.spin_once = spin_once
    qos = types.ModuleType('rclpy.qos')
    qos.qos_profile_sensor_data = object()
    sm, smm = types.ModuleType('sensor_msgs'), types.ModuleType('sensor_msgs.msg')
    smm.Image, smm.CompressedImage = type('Image', (), {}), type('CompressedImage', (), {})
    for name, mod in (('rclpy', rclpy), ('rclpy.qos', qos), ('sensor_msgs', sm), ('sensor_msgs.msg', smm)):
        monkeypatch.setitem(sys.modules, name, mod)
    got = []
    src = Ros2Source('ros2:/cam/f/image_raw/compressed', on_frame=lambda f, t: got.append(f))
    src.start()
    t = time.time()
    while not got and time.time() - t < 5:
        time.sleep(0.02)
    src.stop()
    assert subs[0][0] is smm.CompressedImage and subs[0][1] == '/cam/f/image_raw/compressed'
    assert got and got[0].shape == (48, 64, 3) and tuple(got[0][15, 10]) == (255, 0, 0) and src.error == ''


# ---------------------------------------------------------------------------------------------- footprint, command line
def test_command_line_and_no_qt(cfg, tmp_path):
    c = _small_cfg(cfg, ('ML', 'F'), fps=10.0)
    _write_recording(c, tmp_path, seconds=5.0, kinds=('images',))
    (tmp_path / 'edge.json').write_text(json.dumps(c))
    (tmp_path / 'faults.json').write_text(json.dumps([{'kind': 'freeze', 'camera': 'ML', 'at_s': 1.0,
                                                       'dur_s': 3.5}]))
    env = dict(os.environ, PYTHONPATH=REPO)
    env.pop('QT_QPA_PLATFORM', None)
    code = ('import sys, runpy\nsys.argv = ["vantage.edge"] + sys.argv[1:]\n'
            'try:\n    runpy.run_module("vantage.edge", run_name="__main__")\nexcept SystemExit as e:\n'
            '    bad = sorted(m for m in sys.modules if m.startswith(("PyQt", "OpenGL", "vantage.ui")))\n'
            '    print("MODULES", bad)\n    raise\n')
    r = subprocess.run([sys.executable, '-c', code, '--config', str(tmp_path / 'edge.json'), '--dry-run',
                        str(tmp_path / 'images'), '--inject', str(tmp_path / 'faults.json'), '--report',
                        str(tmp_path / 'run.json')], capture_output=True, text=True, timeout=300, env=env)
    assert r.returncode == 1, r.stderr                         # a fault was raised
    assert 'MODULES []' in r.stdout, r.stdout
    assert 'frozen' in r.stdout
    rep = json.loads((tmp_path / 'run.json').read_text())
    assert rep['schema'] == 'vantage.edge_run/1' and [a['check'] for a in rep['alerts']] == ['frozen']
    r = subprocess.run([sys.executable, '-m', 'vantage.edge', '--config', str(tmp_path / 'edge.json'), '--check'],
                       capture_output=True, text=True, timeout=120, env=env)
    assert r.returncode == 0 and 'ML' in r.stdout
    r = subprocess.run([sys.executable, '-m', 'vantage.edge', '--config', str(tmp_path / 'nope.json')],
                       capture_output=True, text=True, timeout=120, env=env)
    assert r.returncode == 2


def test_validation_checks_pass():
    from vantage.validation.suite import Outcome
    specs = ED.validation_checks()
    assert {s['id'] for s in specs} == {'edge.travel_direction', 'edge.knock_injection', 'edge.fault_detection'}

    class Ctx:
        quick = True

        def rng(self, cid, k=0):
            return np.random.default_rng([7, len(cid), k])

        def sample(self):
            return samples.tigor_sample_project()
    for s in specs:
        assert {'id', 'model', 'title', 'quantity', 'reference', 'tolerance', 'unit', 'refs', 'fn', 'cost'} <= set(s)
        out = s['fn'](Ctx())
        assert isinstance(out, Outcome) and out.ok and out.error <= s['tolerance'], (s['id'], out)


@pytest.mark.skipif(not os.path.isdir(CARLA), reason='the CARLA drive is not on this computer')
def test_real_carla_drive_faults():
    p = samples.tigor_sample_project()
    var = next(v for v in p.variants if v.sensors)
    faults = [{'kind': 'knock', 'camera': 'ML', 'at_s': 6.0, 'yaw': 2.0},
              {'kind': 'knock', 'camera': 'QL', 'at_s': 5.0, 'yaw': -1.5, 'pitch': 1.0},
              {'kind': 'freeze', 'camera': 'R', 'at_s': 5.0, 'dur_s': 3.0},
              {'kind': 'stall', 'camera': 'QR', 'at_s': 8.0, 'dur_s': 0.0}]
    res = ED.dry_run(p, CARLA, faults, variant=var)
    assert res.metrics['caught'] == 4 and res.metrics['false_alarms'] == 0, (res.faults, res.false_alarms)
    clean = ED.dry_run(p, CARLA, [], variant=var)
    assert clean.report['alerts'] == []


# ---------------------------------------------------------------------------------------------- the page
from test_gui import app, pump, settle, win  # noqa: E402,F401  (fixtures)


def _open(win, name, cls):
    if name not in win._page_widgets:
        pg = cls(win.ctx)
        win.pages.addWidget(pg)
        win._page_widgets[name] = pg
    win.set_page(name)
    return win._page_widgets[name]


def test_page_settings_config_and_dry_run(app, win, monkeypatch):
    from vantage.ui.workspaces.edge_ws import EdgePage
    pg = _open(win, 'edge', EdgePage)
    pg.activate()
    t = time.time()
    while pg.cfg is None and time.time() - t < 60:
        pump(app, 0.05)
    assert pg.cfg is not None and pg.cam_table.rowCount() == len(pg.cfg['cameras'])
    assert json.loads(pg.cfg_view.toPlainText())['schema'] == 'vantage.edge_config/1'
    assert 'ExecStart=' in pg.unit_view.toPlainText()
    # a threshold edited like a user, then undone
    f = pg.fields['knock_deg']
    f.lineEdit().selectAll()
    from PyQt6.QtTest import QTest
    QTest.keyClicks(f.lineEdit(), '2.5')
    from PyQt6.QtCore import Qt as _Qt
    QTest.keyClick(f.lineEdit(), _Qt.Key.Key_Return)
    pump(app, 0.2)
    assert win.store.project.analysis.edge_daemon.get('knock_deg') == 2.5
    win.store.undo.undo()
    pump(app, 0.3)
    assert not win.store.project.analysis.edge_daemon.get('knock_deg')
    # a camera's source typed into the table
    pg.cam_table.item(0, 1).setText('rtsp://10.0.0.5/f')
    pump(app, 0.3)
    tag = pg.cam_table.item(0, 0).text()
    assert win.store.project.analysis.edge_daemon['sources'] == {tag: 'rtsp://10.0.0.5/f'}
    t = time.time()
    while pg.cfg['cameras'][0]['source'] != 'rtsp://10.0.0.5/f' and time.time() - t < 30:
        pump(app, 0.05)
    assert pg.cfg['cameras'][0]['source'] == 'rtsp://10.0.0.5/f'
    # the GPIO backend through its segmented control
    pg.gpio.buttons['file'].click()
    pump(app, 0.2)
    assert win.store.project.analysis.edge_daemon['gpio_backend'] == 'file'
    # a dry run on a short synthetic drive with the typical faults
    real = ED.dry_run
    monkeypatch.setattr(ED, 'dry_run', lambda *a, **k: real(*a, **dict(k, seconds=10.0, width=256)))
    pg._preset()
    assert pg.fault_table.rowCount() == len(pg.faults()) == 4
    assert pg.run()
    t = time.time()
    while pg.result is None and time.time() - t < 240:
        pump(app, 0.1)
    assert pg.result is not None
    assert pg.score_table.rowCount() == 4 and pg.alert_table.rowCount() >= 3
    assert pg.result.metrics['caught'] >= 3
    assert pg.b_export.isEnabled()
    settle(app, win)
