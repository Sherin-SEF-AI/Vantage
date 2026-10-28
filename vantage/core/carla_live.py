"""CARLA as a live rig, the app's side: the rig the live server drives (vantage.sim.live), its command line, a whole
live session with rig health watching its streams (the batch mode's), and the body-edge plan of CARLA's own car that
rig health's drift indicator compares the live frames with. vantage.sim.live itself runs in CARLA's Python with only
the bridge's sources, so what needs the rest of Vantage (rig health, the rig planner) lives here."""
from __future__ import annotations

import json
import os
import threading
import time

import numpy as np

from ..sim import live as LV


def body_view(png: bytes, sensor_size, key=(), width: int = 640):
    """rig_health's ExpectedView of CARLA's own car from a live server's /<TAG>.body.png, at rig health's working
    width: the plan the body-edge drift indicator (rig_health.alignment) compares a frame with, when the camera rides on
    CARLA's car rather than the project's. The edge, its normals and the outline are made as rig_health.expected_view
    makes them from its planned body; no horizon."""
    import cv2
    from . import rig_health as rh
    m = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise ValueError('not a body mask image')
    W, H = int(sensor_size[0]), int(sensor_size[1])
    w = min(int(width), W)
    h = max(1, round(w * H / W))
    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    valid, body = m >= 64, m >= 192
    k = np.ones((3, 3), np.uint8)
    inner = cv2.erode(body.astype(np.uint8), k).astype(bool)
    deep = cv2.erode(valid.astype(np.uint8), k, iterations=3).astype(bool)
    edge = body & ~inner & deep
    jj, ii = np.nonzero(edge)
    soft = cv2.GaussianBlur(body.astype(np.float32), (0, 0), 1.5)
    n = np.c_[cv2.Sobel(soft, cv2.CV_32F, 1, 0)[jj, ii], cv2.Sobel(soft, cv2.CV_32F, 0, 1)[jj, ii]]
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    sx, sy = W / w, H / h
    inside = deep.copy()
    inside[[0, -1], :] = False
    inside[:, [0, -1]] = False
    contours = []
    cs, _ = cv2.findContours(body.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for c in cs:
        pts = c.reshape(-1, 2)
        keep = inside[pts[:, 1], pts[:, 0]]
        if keep.all():
            runs = [np.vstack([pts, pts[:1]])]
        else:
            k0 = int(np.argmin(keep))
            pts, keep = np.roll(pts, -k0, 0), np.roll(keep, -k0)
            cut = np.flatnonzero(np.diff(keep.astype(np.int8)))
            runs = [pts[a + 1:b + 1] for a, b in zip(cut[::2], cut[1::2])]
            if len(cut) % 2:
                runs.append(pts[cut[-1] + 1:])
        for r in runs:
            if len(r) >= 3:
                r = cv2.approxPolyDP(r.reshape(-1, 1, 2).astype(np.int32), 0.8, False).reshape(-1, 2).astype(float)
                contours.append(np.c_[(r[:, 0] + 0.5) * sx, (r[:, 1] + 0.5) * sy])
    return rh.ExpectedView(w, h, (W, H), valid, body, np.c_[ii, jj], n, contours, [], tuple(key))


def build_rig(project, settings=None, cameras=None, scale: float = 0.25, variant=None, progress=None, cancel=None):
    """The simulator rig for live streaming (core.simrig.SimRig): the variant's cameras (only `cameras`, tags, when
    given), each at its module's frame rate, the step a whole multiple of every rate (LV.live_tick_hz), rendered at
    `scale` of the sensor size; the lidars, radars, GNSS and IMUs left out. Raises ValueError when no camera is left."""
    from . import simrig as SR
    st = settings if settings is not None else SR.settings_of(project)
    snap = project.snapshot()
    if variant is not None:
        snap.active_uid = variant if isinstance(variant, str) else variant.uid
    v = snap.variant
    want = set(cameras) if cameras else None
    for c in v.cameras:
        if want is not None and c.tag not in want:
            c.visible = False
    for s in v.sensors:
        s.visible = False
    cams = [c for c in v.cameras if c.visible]
    if not cams:
        raise ValueError('No visible camera to stream' + (f' among {", ".join(sorted(want))}' if want else '') + '.')
    rates = [float(snap.module_for(c).fps or 0.0) for c in cams]
    d = st.to_dict()
    d.update(camera_rate='module', image_scale=float(scale), tick_hz=LV.live_tick_hz(rates), nuscenes=False,
             rosbag=False)
    if 'ground_truth' in d:                          # a recording's extras: live has its own faults, no scenario
        d['ground_truth'] = {k: False for k in (d['ground_truth'] or {})}
    for k, v in (('faults', []), ('actors', []), ('scenario', {'name': 'free_drive', 'params': {}})):
        if k in d:
            d[k] = v
    return SR.build(snap, SR.SimSettings.from_dict(d), progress=progress, cancel=cancel)


def command(python: str, rig_path: str, http_port: int = 0, args=(), memory_gb: float = 4.0, scope: bool = True,
            extra_path: str = '') -> tuple[list, dict]:
    """(command, environment) running the live server in CARLA's Python with Vantage's source on PYTHONPATH, in a
    systemd scope capped at memory_gb (no swap)."""
    from ..sim import server as SV
    src = SV.vantage_source()
    if not src:
        raise RuntimeError('The live rig needs Vantage\'s source folder (a frozen build has none): run it from a '
                           'source checkout.')
    cmd = [python, '-m', 'vantage.sim.live', rig_path, '--http-port', str(int(http_port))] + [str(x) for x in args]
    if scope:
        cmd = SV.scoped(cmd, memory_gb)
    env = {k: v for k, v in os.environ.items() if not k.startswith(('PYTHON', 'QT_', 'VIRTUAL_ENV'))}
    env['PYTHONPATH'] = (extra_path + os.pathsep if extra_path else '') + src
    env['PYTHONUNBUFFERED'] = '1'
    return cmd, env


def run_session(project, seconds: float, python: str, out_dir: str, settings=None, cameras=None, scale: float = 0.25,
                faults=(), serve_size: str = 'sensor', extra_path: str = '', scope: bool = True,
                check_server: bool = True, start_timeout: float = 600.0, progress=None) -> dict:
    """A live session without the app (the batch mode, scripts): the live server on the settings' CARLA server, rig
    health (core.rig_health) on its streams for `seconds` after the first frames, each fault of `faults` ((at_s, fault)
    pairs) injected at its time, the drift indicator of every camera's last frame against CARLA's own car (its body
    mask), then the live server stopped. Writes rig/, live.json and rig_health.json into out_dir and returns
    {'schema', 'status': 'ok' | 'failed', 'error', 'rig_health': snapshot, 'live': the server's report, 'faults'}.
    A RuntimeError when no CARLA server answers (check_server) or the live server does not stream."""
    import subprocess
    from . import rig_health as rh
    from . import simrig as SR
    from ..sim import server as SV
    st = settings if settings is not None else SR.settings_of(project)
    if check_server and not SV.port_open('127.0.0.1', st.port):
        raise RuntimeError(f'No CARLA server answers on port {st.port}.')
    prog = progress or (lambda f, text='': None)
    os.makedirs(out_dir, exist_ok=True)
    prog(0.0, 'Writing the live rig')
    rig = build_rig(project, st, cameras, scale)
    SR.export(rig, project, os.path.join(out_dir, 'rig'))
    tags = [c['tag'] for c in rig.doc['cameras']]
    rep_path = os.path.join(out_dir, 'live.json')
    cmd, env = command(python, os.path.join(out_dir, 'rig', 'sim_rig.json'), 0,
                       ['--port', st.port, '--tm-port', st.tm_port, '--timeout', f'{st.timeout_s:g}', '--serve-size',
                        serve_size, '--parent-pid', os.getpid(), '--report', rep_path], scope=scope,
                       extra_path=extra_path)
    log = open(os.path.join(out_dir, 'live.log'), 'w', encoding='utf-8')
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=log, stdin=subprocess.DEVNULL, text=True,
                            cwd=out_dir)
    log.close()
    client, mon, injected = None, None, []
    try:
        ev = LV.parse_event(proc.stdout.readline() or '{}')
        if not ev or ev.get('event') != 'listening':
            raise RuntimeError('The live server did not start (see live.log).')
        threading.Thread(target=lambda: [None for _ in proc.stdout], daemon=True).start()
        client = LV.LiveClient(ev['port'], timeout=10.0)
        t0 = time.monotonic()
        while True:
            s = client.status()
            if s['state'] == 'failed' or proc.poll() is not None:
                raise RuntimeError(f'The live server failed: {s.get("error") or "see live.log"}')
            if s['state'] == 'streaming' and all(c['served'] for c in s['cameras'].values()):
                break
            if time.monotonic() - t0 > start_timeout:
                raise RuntimeError(f'The live server did not stream within {start_timeout:g} s.')
            prog(0.05, f'Live server: {s["state"]}')
            time.sleep(0.25)
        p = project.snapshot()
        p.streams = {t: client.url(t) for t in tags}
        cams = [c for c in p.variant.cameras if c.tag in tags]
        views = {t: body_view(client.body(t), s['cameras'][t]['size'], ('carla', t)) for t in tags
                 if s['cameras'][t].get('body_mask')}
        mon = rh.RigMonitor()
        mon.configure(p, cams, planned=rh.schedule_offsets(rig.doc))   # the live rig's own schedule: what it delivers
        mon.start()
        t1 = time.monotonic()
        while time.monotonic() - t1 < 30.0 and not all(x.stats.frames for x in mon.streams.values()):
            mon.poll()
            time.sleep(0.05)
        mon.reset_stats()
        todo = sorted(((float(at), f) for at, f in faults), key=lambda x: x[0])
        t1 = time.monotonic()
        while (el := time.monotonic() - t1) < float(seconds):
            while todo and el >= todo[0][0]:
                at, f = todo.pop(0)
                injected.append(dict(client.fault(f), at_s=round(el, 3)))
            mon.poll()
            prog(0.1 + 0.85 * el / max(float(seconds), 1e-9), f'{el:.0f} s of {float(seconds):g} s')
            time.sleep(0.05)
        for t, x in mon.streams.items():
            f = x.source.latest()
            ev_ = views.get(t)
            if f is not None and ev_ is not None:
                x.drift, x.drift_checked = rh.alignment(f, ev_), True
        snap = rh.jsonable(mon.snapshot())
        for t, c in snap['cameras'].items():
            c['drift_against'] = 'carla_body' if t in views else None
        snap.update(project=project.name, variant=project.variant.name, seconds=float(seconds),
                    time=time.strftime('%Y-%m-%dT%H:%M:%S'), settings=mon.settings.to_dict())
    finally:
        if mon is not None:
            mon.stop()
        if client is not None and proc.poll() is None:
            try:
                client.stop()
            except (OSError, LV.LiveError):
                pass
        try:
            proc.wait(120)
        except subprocess.TimeoutExpired:
            proc.terminate()
            proc.wait(30)
    with open(os.path.join(out_dir, 'rig_health.json'), 'w', encoding='utf-8') as fh:
        json.dump(snap, fh, indent=1, default=LV._jsonable)
    try:
        with open(rep_path, encoding='utf-8') as fh:
            live = json.load(fh)
    except (OSError, ValueError):
        live = None
    prog(1.0, 'Done')
    return {'schema': LV.SCHEMA, 'status': 'ok' if proc.returncode == 0 else 'failed',
            'error': '' if proc.returncode == 0 else f'the live server exited with {proc.returncode}',
            'rig_health': snap, 'live': live, 'faults': injected, 'out_dir': out_dir}
