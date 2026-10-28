"""CARLA as a live rig: the planned cameras driven in CARLA in real time and served as MJPEG over HTTP, so Vantage's
Live rig page (core.rig_health, which opens HTTP streams through OpenCV) watches them as it watches a real rig, with
faults injected and cleared while it runs.

    python -m vantage.sim.live sim_rig.json --port 2000 --http-port 8765 [--town Town10HD --traffic 20 --seed 1]

in CARLA's own Python (the carla wheel, numpy, OpenCV; Vantage's source on PYTHONPATH), against a CARLA 0.9.16 server
that renders. sim_rig.json and sim_remap.npz come from core.simrig; vantage.core.carla_live.build_rig() makes the
live one: the chosen cameras at their modules' frame rates, the simulation step a whole multiple of every rate. Only the cameras are
simulated live (rig health watches camera streams); lidars, radars, GNSS and IMUs are the recording bridge's.

What it does
------------
1. Serves HTTP on 127.0.0.1 at once (--http-port 0: a free port, printed as {"event": "listening", "port": N}):
     GET  /<TAG>.mjpg     the camera's frames as MJPEG (multipart/x-mixed-replace), each client gets the newest frame
     GET  /<TAG>.jpg      the newest frame
     GET  /<TAG>.body.png CARLA's own car as the camera sees it at its planned mount (255: the ego's body, 128: the
                          rest of the image, 0: outside it), at the rendered size: what rig health's body-edge drift
                          indicator should compare with here, since CARLA's car is not the project's
                          (vantage.core.carla_live.body_view())
     GET  /status         JSON (vantage.sim_live/1): state, real-time factor, per camera served fps, latency, faults
     POST /fault          {"tag": "F", "deg": [yaw, pitch, roll]} or {"tag", "kind": rotation | translation | freeze
                          | drop | delay | clock_offset, ...}: the fault as applied, with its id
     POST /fault/clear    {"tag": "F"} | {"id": 3} | {} (every fault)
     POST /stop           stop and tidy up
   POSTs must be JSON (a web page cannot send them without the browser asking first) and the Host must be local.
2. Connects, loads the town, synchronous mode at the rig's step, the weather (a CARLA preset or the rig's mapped
   Vantage condition); spawns the ego (the rig's CARLA vehicle, role 'hero') and --traffic vehicles on the traffic
   manager's autopilot (seeded; --still parks the ego), lets them drive --warmup seconds, reads where the rig's origin
   (the rear-axle centre on the ground) lies in the CARLA vehicle (its rear wheels and bounding box, as the recording
   bridge), renders each camera's views once with sensor.camera.instance_segmentation at the planned mount and keeps
   the ego's own pixels (G + 256·B is the actor id, measured on a 0.9.16 server: core.groundtruth.decode_instance),
   remapped to the output image by its nearest view pixel (the body masks; --no-body-masks skips them), and attaches
   every camera's pinhole views at their rig poses, each camera spawned so that CARLA renders it on the steps of its
   schedule.
3. Steps paced to the wall clock: step i is due dt after step i − 1. When the server cannot keep up, the step runs
   late and the lag is dropped (never a burst to catch up): /status gives the real-time factor reached. Each camera
   frame is made with the recording bridge's own functions: the views stacked and remapped through the rig's tables
   (cv2.remap, bilinear), Unreal's vignette taken out in linear light (vantage.sim.tonemap), the cube faces feathered
   across their seams (carla_bridge.feather); then a little sensor noise (--noise: CARLA renders none, and a still
   scene would read as a frozen camera), resized to the module's sensor size (--serve-size sensor, the default: rig
   health checks the resolution against the module; the detail is the rig's image scale), JPEG.
4. Faults, applied between steps, reversible, logged with their simulation and wall times:
   * rotation, deg = [yaw, pitch, roll] degrees about the camera's own axes (x along its view, y left, z up;
     intrinsic Z-Y-X, pitch positive down, as Vantage's poses): every view of the camera is turned about their common
     optical centre before the next step, so CARLA renders the turned camera (not a warp of its image). The views are
     moved in place with Actor.set_transform, which on a sensor attached to a vehicle sets its transform relative to
     the vehicle and keeps it attached (measured on a 0.9.16 server: relative pose within 1e-4, still following the
     vehicle when it moves) and allocates nothing. Spawning them again instead is what a client without
     set_transform gets (on the camera's next frame step, so its schedule is kept); in two runs here, beside another
     CARLA server on the same GPU, the throughput fell to less than half within 20 s of such a re-spawn and did not
     recover while the run lasted (the cause was not isolated; with set_transform it never happened);
   * translation, m = [x, y, z] metres in the vehicle frame (ISO 8855): the same, moved;
   * freeze: the stream repeats the frame it showed last; drop: each frame is left out with probability rate;
     delay (clock_offset: the same, by its live meaning): every frame is served ms later than it was made.
   s = seconds clears a fault by itself after that long. One rotation and one translation per camera at a time: a new
   one replaces the old.
5. Stops on POST /stop, SIGINT / SIGTERM, after --duration seconds, or when --parent-pid exits (the app that started
   it), and tidies up as the recording bridge does: the vehicles off the autopilot, a step, every sensor stopped,
   everything spawned destroyed, a step, the traffic manager and the world back to their original asynchronous
   settings; then the HTTP server closes. --report writes the session (live.json) at the end.

Latency is measured from the start of the step that rendered a frame to the frame being served: CARLA's render and
delivery, the remap and encoding, and any delay fault. A client's own read and decode come on top (on one machine,
about a frame's decode).
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import heapq
import json
import math
import os
import queue
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import quote, unquote, urlsplit

import numpy as np

from . import frames as FR

SCHEMA = 'vantage.sim_live/1'
RIG_SCHEMA = 'vantage.sim_rig/1'
BOUNDARY = 'vantageframe'
KINDS = ('rotation', 'translation', 'freeze', 'drop', 'delay', 'clock_offset')
MOUNT_KINDS = ('rotation', 'translation')
MAX_DEG = 30.0                    # largest fault rotation per axis
MAX_M = 2.0                       # largest fault translation per axis (the recording bridge's limit)
MAX_MS = 5000.0                   # longest delay
MAX_S = 3600.0                    # longest timed fault
MAX_LAG_S = 0.25                  # a step this late drops the pacing debt instead of catching up
WINDOW_S = 5.0                    # served fps and latency over this window
FRAME_WAIT_S = 2.0                # longest wait for a due frame's views (the first frame of a mount: --timeout)
IN_FLIGHT = 2                     # frames of one camera being made at once; another one is skipped (and counted)
MAX_BODY = 65536
LIMITS_TEXT = ('Cameras only (no lidar, radar, GNSS or IMU live). Each frame is one instant with a manual exposure '
               '(no rolling shutter or motion blur). Frame rates are what the server reaches in real time: when it '
               'cannot keep up, frames come less often, as they would from an overloaded rig. CARLA renders and reads '
               'back every pinhole view in turn (a fisheye is 5 views); measured on a 0.9.16 server (RTX 5080): one '
               'fisheye at 30 fps with views of 231 px, 23 fps at 924 px, 10 fps at 1848 px; six at 462 px, 2.8 fps '
               'each. '
               'Fewer cameras, narrower lenses and a smaller render size all help. Served at the module\'s '
               'sensor size from views rendered at the rig\'s image scale. The noise is uniform, a few levels: not a '
               'sensor noise model. CARLA\'s car body is not the project\'s: the live panel measures the body-edge '
               'drift indicator against CARLA\'s own car as each camera sees it at its planned mount (the body masks); '
               'against the project\'s body it reads the two bodies\' difference, not the camera.')


class Cancelled(Exception):
    pass


class LiveError(RuntimeError):
    """An error the live server answered with (its message is for the user)."""


# =========================================================================== faults
def _num(v, lo, hi, name) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise ValueError(f'{name} must be a number.') from None
    if not math.isfinite(x) or x < lo or x > hi:
        raise ValueError(f'{name} must be between {lo:g} and {hi:g}.')
    return x


def _vec(v, lim, name) -> list:
    if not isinstance(v, (list, tuple)) or len(v) != 3:
        raise ValueError(f'{name} must be three numbers.')
    return [_num(x, -lim, lim, name) for x in v]


def parse_fault(d, tags) -> dict:
    """A live fault checked and normalised: {'tag', 'kind', and 'deg' | 'm' | 'rate' | 'ms', optional 's'}. Raises
    ValueError (the message says what is wrong) for anything else; values out of range are refused, not clamped."""
    if not isinstance(d, dict):
        raise ValueError('A fault is a JSON object, e.g. {"tag": "F", "deg": [0, 0.5, 0]}.')
    tag = d.get('tag')
    if not isinstance(tag, str) or tag not in tags:
        raise ValueError(f'No camera {tag!r} in the live rig (cameras: {", ".join(tags) or "none"}).')
    kind = d.get('kind') or ('rotation' if 'deg' in d else 'translation' if 'm' in d else '')
    if kind not in KINDS:
        raise ValueError(f'Unknown fault kind {kind!r} (one of {", ".join(KINDS)}).')
    out = {'tag': tag, 'kind': kind}
    if kind == 'rotation':
        out['deg'] = _vec(d.get('deg'), MAX_DEG, 'deg')
    elif kind == 'translation':
        out['m'] = _vec(d.get('m'), MAX_M, 'm')
    elif kind == 'drop':
        out['rate'] = _num(d.get('rate', 0.3), 0.0, 1.0, 'rate')
    elif kind in ('delay', 'clock_offset'):
        out['ms'] = _num(d.get('ms', 100.0), 0.0, MAX_MS, 'ms')
    if d.get('s') is not None:
        out['s'] = _num(d['s'], 0.0, MAX_S, 's')
    return out


def describe_fault(f) -> str:
    """A fault in a few words, for tables and messages."""
    k = f.get('kind')
    if k == 'rotation':
        y, p, r = f['deg']
        txt = 'turned ' + ', '.join(f'{n} {v:+g}°' for n, v in (('yaw', y), ('pitch', p), ('roll', r)) if v) \
            if any(f['deg']) else 'turned 0°'
    elif k == 'translation':
        txt = 'moved ' + ', '.join(f'{n} {100 * v:+g} cm' for n, v in zip('xyz', f['m']) if v) if any(f['m']) \
            else 'moved 0 cm'
    elif k == 'freeze':
        txt = 'frozen'
    elif k == 'drop':
        txt = f'dropping {100 * f["rate"]:g} %'
    else:
        txt = f'{f["ms"]:g} ms late'
    if f.get('s'):
        txt += f' for {f["s"]:g} s'
    return txt


def turned_view(view: dict, cam_pose: dict, deg=None, m=None) -> dict:
    """A view's CARLA pose ({'location': rig-relative, CARLA axes; 'rotation': carla.Rotation fields}) after its camera
    is turned by deg = (yaw, pitch, roll) about the camera's own axes and moved by m (vehicle frame, metres). The
    camera's frame R becomes R · rot_zyx(deg), so each view's body frame B becomes R · rot_zyx(deg) · Rᵀ · B; the views
    turn about their common optical centre."""
    from ..core.mathutil import rot_zyx
    loc = [float(v) for v in view['location']]
    if m is not None:
        loc = [a + b for a, b in zip(loc, FR.carla_location(m))]
    r = view['rotation']
    rot = {'pitch': float(r['pitch']), 'yaw': float(r['yaw']), 'roll': float(r['roll'])}
    if deg is not None and any(float(x) for x in deg):
        B = FR.vantage_matrix(rot['pitch'], rot['yaw'], rot['roll'])
        Rc = rot_zyx(float(cam_pose['yaw']), float(cam_pose['pitch']), float(cam_pose['roll']))
        p, y, rr = FR.carla_rotation(Rc @ rot_zyx(*[float(x) for x in deg]) @ Rc.T @ B)
        rot = {'pitch': p, 'yaw': y, 'roll': rr}
    return {'location': loc, 'rotation': rot}


def live_tick_hz(rates, hi: float = 400.0) -> float:
    """The simulation step of a live rig: the smallest whole multiple of the fastest camera rate that every rate
    divides (30 → 30, 30 and 25 → 150), else the fastest rate (the others then round to whole steps)."""
    r = sorted({round(float(x), 6) for x in rates if x and float(x) > 0}) or [10.0]
    top = r[-1]
    for k in range(1, int(hi // top) + 1):
        R = top * k
        if all(abs(R / x - round(R / x)) < 1e-6 for x in r):
            return float(R)
    return float(top)


# =========================================================================== serving
class Hub:
    """The newest JPEG of one camera, for every client of its stream."""

    def __init__(self, tag):
        self.tag = tag
        self.cond = threading.Condition()
        self.seq = 0
        self.jpg = b''
        self.meta = {}
        self.last_frame = -1
        self.closed = False
        self.clients = 0
        self.pub = collections.deque(maxlen=2000)      # (served at, latency s, components)

    def publish(self, jpg: bytes, meta: dict) -> bool:
        """Serve a frame; False for one older than the frame already served (made out of order) or after close."""
        with self.cond:
            if self.closed or meta['frame'] <= self.last_frame:
                return False
            now = time.monotonic()
            self.last_frame = meta['frame']
            self.seq += 1
            self.jpg = jpg
            self.meta = dict(meta, published=now)
            self.pub.append((now, now - meta['t_step'], (meta['t_tick'] - meta['t_step'],
                                                         meta['t_views'] - meta['t_tick'],
                                                         meta['t_done'] - meta['t_views'])))
            self.cond.notify_all()
            return True

    def wait(self, after: int, timeout: float):
        """(seq, jpg, meta) of a frame newer than `after`, or None (timeout or closed)."""
        with self.cond:
            self.cond.wait_for(lambda: self.seq != after or self.closed, timeout)
            if self.closed or self.seq == after:
                return None
            return self.seq, self.jpg, self.meta

    def close(self):
        with self.cond:
            self.closed = True
            self.cond.notify_all()


class Delayer(threading.Thread):
    """Serves frames at a later time (the delay fault)."""

    def __init__(self):
        super().__init__(name='live-delay', daemon=True)
        self.cond = threading.Condition()
        self.heap = []
        self.k = 0
        self.done = False

    def put(self, due, hub, jpg, meta):
        with self.cond:
            self.k += 1
            heapq.heappush(self.heap, (due, self.k, hub, jpg, meta))
            self.cond.notify()

    def run(self):
        while True:
            with self.cond:
                while True:
                    if self.done:
                        return
                    if not self.heap:
                        self.cond.wait(0.5)
                        continue
                    left = self.heap[0][0] - time.monotonic()
                    if left <= 0:
                        break
                    self.cond.wait(min(left, 0.5))
                _, _, hub, jpg, meta = heapq.heappop(self.heap)
            hub.publish(jpg, meta)

    def close(self):
        with self.cond:
            self.done = True
            self.heap.clear()
            self.cond.notify()


class _Handler(BaseHTTPRequestHandler):
    server_version = 'VantageLive/1'
    protocol_version = 'HTTP/1.0'          # one request per connection; a stream ends when the connection closes

    def log_message(self, fmt, *args):
        pass

    def _host_ok(self) -> bool:
        """Only requests addressed to this machine (a page elsewhere cannot rebind its name to it)."""
        h = (self.headers.get('Host') or '').strip().lower()
        if h.startswith('['):
            h = h[1:h.find(']')]
        elif ':' in h:
            h = h.rsplit(':', 1)[0]
        return h in ('', '127.0.0.1', 'localhost', '::1', str(self.server.server_address[0]))

    def _send(self, code, data: bytes, ctype):
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        try:
            self.wfile.write(data)
        except OSError:
            pass

    def _json(self, code, doc):
        self._send(code, (json.dumps(doc, default=_jsonable) + '\n').encode('utf-8'), 'application/json')

    def do_GET(self):
        if not self._host_ok():
            return self._json(403, {'error': 'requests must be addressed to this machine'})
        live = self.server.live
        path = unquote(urlsplit(self.path).path)
        if path in ('/status', '/status.json'):
            return self._json(200, live.status())
        if path == '/faults':
            return self._json(200, {'faults': live.active_faults(), 'log': live.fault_log()})
        if path in ('/', '/index.html'):
            return self._send(200, live.index_html().encode('utf-8'), 'text/html; charset=utf-8')
        if path.endswith('.body.png') and path[1:-9] in live.cams:
            png = live.cams[path[1:-9]].body_png
            if not png:
                return self._json(404, {'error': f'{path[1:-9]}: no body mask (not made yet, or turned off)'})
            return self._send(200, png, 'image/png')
        name, _, ext = path[1:].rpartition('.')
        hub = live.hubs.get(name)
        if hub is None or ext not in ('mjpg', 'jpg', 'jpeg'):
            return self._json(404, {'error': f'no such stream: {path}', 'streams': [f'/{quote(t, safe="")}.mjpg'
                                                                                     for t in live.hubs]})
        if ext != 'mjpg':
            with hub.cond:
                jpg = hub.jpg
            if not jpg:
                return self._json(503, {'error': f'{name}: no frame yet'})
            return self._send(200, jpg, 'image/jpeg')
        self.send_response(200)
        self.send_header('Content-Type', f'multipart/x-mixed-replace; boundary={BOUNDARY}')
        self.send_header('Cache-Control', 'no-cache, no-store')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Connection', 'close')
        self.end_headers()
        self.connection.settimeout(10.0)             # a client that stops reading is dropped
        with hub.cond:
            hub.clients += 1
        try:
            seq = 0
            while not self.server.closing:
                got = hub.wait(seq, 1.0)
                if got is None:
                    if hub.closed:
                        break
                    continue
                seq, jpg, meta = got
                head = (f'--{BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpg)}\r\n'
                        f'X-Frame: {meta["frame"]}\r\nX-Sim-Time: {meta["sim_s"]:.6f}\r\n'
                        f'X-Published: {meta["published"]:.6f}\r\n\r\n').encode('ascii')
                self.wfile.write(head + jpg + b'\r\n')
        except (OSError, ValueError):
            pass
        finally:
            with hub.cond:
                hub.clients -= 1

    def do_POST(self):
        if not self._host_ok():
            return self._json(403, {'error': 'requests must be addressed to this machine'})
        if 'application/json' not in (self.headers.get('Content-Type') or '').lower():
            return self._json(415, {'error': 'send JSON (Content-Type: application/json)'})
        try:
            n = int(self.headers.get('Content-Length') or 0)
        except ValueError:
            n = -1
        if n < 0 or n > MAX_BODY:
            return self._json(413, {'error': 'body too large'})
        try:
            body = json.loads(self.rfile.read(n) or b'{}') if n else {}
        except ValueError:
            return self._json(400, {'error': 'the body is not JSON'})
        live = self.server.live
        path = urlsplit(self.path).path
        try:
            if path == '/fault':
                return self._json(200, {'fault': live.add_fault(body)})
            if path == '/fault/clear':
                body = body if isinstance(body, dict) else {}
                return self._json(200, {'cleared': live.clear_faults(body.get('tag'), body.get('id'))})
            if path == '/stop':
                live.request_stop('stop requested over HTTP')
                return self._json(202, {'state': 'stopping'})
        except ValueError as e:
            return self._json(400, {'error': str(e)})
        return self._json(404, {'error': f'no such action: {path}'})


def serve(live, host='127.0.0.1', port=0) -> ThreadingHTTPServer:
    """Start the live server's HTTP side on its own thread; .server_address[1] is the port."""
    srv = ThreadingHTTPServer((host, int(port)), _Handler)
    srv.daemon_threads = True
    srv.live = live
    srv.closing = False
    threading.Thread(target=srv.serve_forever, kwargs={'poll_interval': 0.2}, name='live-http', daemon=True).start()
    return srv


def close_server(srv, live=None):
    srv.closing = True
    for hub in (live.hubs.values() if live is not None else ()):
        hub.close()
    srv.shutdown()
    srv.server_close()


# =========================================================================== the live rig (CARLA's side)
class _Cam:
    """One camera's live state."""

    def __init__(self, doc, every, phase, serve_size, noise, seed):
        self.doc = doc
        self.tag = doc['tag']
        self.n = max(int(every), 1)
        self.first = int(phase) % self.n + 1          # the first step (1, 2, …) the camera renders on
        self.views = []                               # [(actor, queue)] of the views now mounted
        self.gen = 0
        self.pending = {}                             # CARLA frame -> {view index: image}
        self.mount = (None, None)                     # (deg, m) mounted now
        self.want = None                              # (deg, m) to mount on the next frame step
        self.wait_long = True                         # the first frame of a mount may take a while
        self.inflight = 0
        self.last_sent = -1
        self.counts = {'rendered': 0, 'skipped_busy': 0, 'dropped_by_fault': 0, 'missed': 0, 'remounts': 0}
        self.method = ''                              # how the last fault mount was made: 'in place' | 'spawned again'
        self.frozen = None                            # the JPEG a freeze repeats
        self.body_png = b''                           # CARLA's own car at the planned mount (see the module doc)
        self.body_info = None
        self.rng = np.random.default_rng([int(seed), sum(map(ord, self.tag)), 1])      # the drop fault's draws
        im = doc['image']
        self.render_size = (int(im['width']), int(im['height']))
        self.serve_size = tuple(int(v) for v in im.get('sensor_size') or self.render_size) \
            if serve_size == 'sensor' else self.render_size
        lv = int(round(noise))
        if lv > 0:
            g = np.random.default_rng([int(seed), sum(map(ord, self.tag))])
            w, h = self.render_size
            self.noise = [g.integers(0, 2 * lv + 1, (h, w, 3), dtype=np.uint8) for _ in range(4)]
        else:
            self.noise = []
        self.k = 0

    def due(self, i) -> bool:
        return i >= self.first and (i - self.first) % self.n == 0


class LiveRig:
    """The live rig on a CARLA server. run() blocks until it is stopped; status() and the fault calls are safe from any
    thread (the HTTP server's)."""

    def __init__(self, carla, rig: dict, remaps: dict, args, emit=None):
        from . import carla_bridge as CB
        from . import tonemap as TM
        self.CB, self.TM = CB, TM
        self.carla = carla
        self.rig = rig
        self.remaps = remaps
        self.a = args
        self.emit = emit or (lambda d: None)
        self.lock = threading.RLock()
        self.stop_evt = threading.Event()
        self.stop_reason = ''
        self.state = 'starting'
        self.error = ''
        self.notes = []
        self.R = float(rig['simulation']['tick_hz'])
        self.dt = 1.0 / self.R
        self.actors = []                  # vehicles, in spawn order
        self.sensor_actors = []           # every camera view mounted (and not yet destroyed)
        self.tm = None
        self.world = None
        self.original = None
        self.ego = None
        self.O = np.zeros(3)
        self.steps = 0
        self.t_stream = None
        self.timeline = collections.deque(maxlen=4000)  # (wall, step) for the real-time factor
        self.lag_resets = 0
        self.behind_s = 0.0
        self.info = {}
        self.faults = {}                  # id -> active fault
        self.log = []                     # every fault, as applied and cleared
        self.next_id = 1
        self.pool = None
        self.delayer = Delayer()
        self.delayer.start()
        want = set(getattr(args, 'cameras', None) or [])
        docs = [c for c in rig['cameras'] if not want or c['tag'] in want]
        if not docs:
            raise ValueError('No camera to stream: the rig has none' + (' of ' + ', '.join(sorted(want)) if want
                                                                        else '') + '.')
        self.cams = {}
        for c in docs:
            cap = c['capture']
            cam = _Cam(c, cap['every_ticks'], cap['phase_ticks'], getattr(args, 'serve_size', 'sensor'),
                       float(getattr(args, 'noise', 2.0)), int(getattr(args, 'seed', 1) or 0))
            cam.devignette, cam.blend = self._tables(c)
            self.cams[c['tag']] = cam
        self.hubs = {t: Hub(t) for t in self.cams}
        if not getattr(args, 'keep_vignette', False):
            self._note('Unreal\'s vignette (in every pinhole view) taken out of each pixel in linear light through '
                       'CARLA\'s measured camera response (vantage.sim.tonemap), as the recording bridge does.')
        if float(getattr(args, 'noise', 0.0)) > 0:
            self._note(f'Uniform noise of 0…{2 * int(round(float(args.noise)))} levels added per frame (CARLA renders '
                       'none; a still scene would read as frozen).')

    # ------------------------------------------------------------------ helpers
    def _note(self, text):
        with self.lock:
            if text not in self.notes:
                self.notes.append(text)

    def _set_state(self, s, **extra):
        with self.lock:
            self.state = s
        self.emit(dict({'event': 'state', 'state': s}, **extra))

    def _tables(self, c):
        """(devignette layers or None, blend tables or None) of a camera, as the recording bridge makes them."""
        TM = self.TM
        blend = None
        b = c['remap'].get('blend')
        if b and b['idx'] in self.remaps:
            blend = tuple(self.remaps[b[k]] for k in ('idx', 'x', 'y', 'w'))
        if getattr(self.a, 'keep_vignette', False):
            return None, blend
        sizes = [(v['width'], v['height']) for v in c['views']]
        offs = c['atlas']['offsets']
        layers = [TM.Devignette(TM.mask_of_remap(self.remaps[c['remap']['x']], self.remaps[c['remap']['y']], offs,
                                                 sizes))]
        if blend is not None:
            m = TM.mask_of_remap(blend[1], blend[2], offs, sizes)
            layers += [TM.Devignette(m[:, j][None, :]) for j in range(m.shape[1])]
        return layers, blend

    def request_stop(self, reason='stopped'):
        with self.lock:
            if not self.stop_reason:
                self.stop_reason = reason
        self.stop_evt.set()

    def _check(self):
        if self.stop_evt.is_set():
            raise Cancelled()

    def _parent_gone(self) -> bool:
        pid = int(getattr(self.a, 'parent_pid', 0) or 0)
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        return False

    # ------------------------------------------------------------------ faults (any thread)
    def add_fault(self, d) -> dict:
        f = parse_fault(d, list(self.cams))
        with self.lock:
            if self.state in ('stopping', 'stopped', 'failed'):
                raise ValueError(f'The live rig is {self.state}.')
            c = self.cams[f['tag']]
            if f['kind'] in MOUNT_KINDS:              # one of each per camera: a new one replaces the old
                for old in [x for x in self.faults.values() if x['tag'] == f['tag'] and x['kind'] == f['kind']]:
                    self._end(old, 'replaced')
            f.update(id=self.next_id, since_s=round(self._sim_t(), 4), since_wall=time.time(),
                     until=(time.monotonic() + f['s']) if f.get('s') else None, text=describe_fault(f))
            self.next_id += 1
            if f['kind'] in MOUNT_KINDS:
                f['state'] = 'mounting'
                c.want = self._mount_of(f['tag'], extra=f)
            else:
                f['state'] = 'applied'
                f['applied_s'] = f['since_s']
            if f['kind'] == 'freeze':
                with self.hubs[f['tag']].cond:
                    c.frozen = self.hubs[f['tag']].jpg or None
            self.faults[f['id']] = f
            self.log.append(dict(f))
        self.emit({'event': 'fault', 'fault': _public(f)})
        return _public(f)

    def clear_faults(self, tag=None, fid=None) -> list:
        if tag is not None and tag not in self.cams:
            raise ValueError(f'No camera {tag!r} in the live rig.')
        with self.lock:
            if fid is not None:
                try:
                    fid = int(fid)
                except (TypeError, ValueError):
                    raise ValueError('id must be a fault\'s number.') from None
            gone = [f for f in self.faults.values() if (tag is None or f['tag'] == tag) and
                    (fid is None or f['id'] == fid)]
            for f in gone:
                self._end(f, 'cleared')
            for t in {f['tag'] for f in gone if f['kind'] in MOUNT_KINDS}:
                self.cams[t].want = self._mount_of(t)
        out = [_public(f) for f in gone]
        if out:
            self.emit({'event': 'cleared', 'faults': out})
        return out

    def _end(self, f, how):
        """(lock held) End an active fault: logged with when."""
        self.faults.pop(f['id'], None)
        for row in self.log:
            if row['id'] == f['id']:
                row.update(state=how, ended_s=round(self._sim_t(), 4), ended_wall=time.time())
        if f['kind'] == 'freeze':
            self.cams[f['tag']].frozen = None

    def _mount_of(self, tag, extra=None):
        """(lock held) The (deg, m) mount fault of a camera from its active faults (plus one being added)."""
        deg = m = None
        for f in list(self.faults.values()) + ([extra] if extra else []):
            if f['tag'] != tag:
                continue
            if f['kind'] == 'rotation':
                deg = f['deg']
            elif f['kind'] == 'translation':
                m = f['m']
        return (deg, m)

    def active_faults(self) -> list:
        with self.lock:
            return [_public(f) for f in self.faults.values()]

    def fault_log(self) -> list:
        with self.lock:
            return [_public(f) for f in self.log]

    def _expire(self):
        now = time.monotonic()
        with self.lock:
            due = [f for f in self.faults.values() if f.get('until') is not None and now >= f['until']]
        for f in due:
            self.clear_faults(fid=f['id'])

    def _delivery(self, tag) -> dict:
        """(lock held) The delivery faults on a camera now."""
        out = {'drop': 0.0, 'freeze': False, 'delay_ms': 0.0}
        for f in self.faults.values():
            if f['tag'] != tag:
                continue
            if f['kind'] == 'drop':
                out['drop'] = max(out['drop'], f['rate'])
            elif f['kind'] == 'freeze':
                out['freeze'] = True
            elif f['kind'] in ('delay', 'clock_offset'):
                out['delay_ms'] += f['ms']
        return out

    # ------------------------------------------------------------------ status (any thread)
    def _sim_t(self) -> float:
        return self.steps * self.dt

    def status(self) -> dict:
        now = time.monotonic()
        with self.lock:
            tl = [x for x in self.timeline if x[0] >= now - WINDOW_S]
            rtf = ((tl[-1][1] - tl[0][1]) * self.dt / (tl[-1][0] - tl[0][0])) if len(tl) >= 2 and \
                tl[-1][0] > tl[0][0] else None
            cams = {}
            for tag, c in self.cams.items():
                hub = self.hubs[tag]
                with hub.cond:
                    pub = [x for x in hub.pub if x[0] >= now - WINDOW_S]
                    clients, served, meta = hub.clients, hub.seq, dict(hub.meta)
                fps = (len(pub) - 1) / (pub[-1][0] - pub[0][0]) if len(pub) >= 2 and pub[-1][0] > pub[0][0] else 0.0
                lat = np.array([x[1] for x in pub]) * 1000.0
                comp = np.array([x[2] for x in pub]) * 1000.0 if pub else np.zeros((0, 3))
                deg, m = c.mount
                cams[tag] = {
                    'url': f'/{quote(tag, safe="")}.mjpg', 'channel': c.doc.get('channel'),
                    'rate_hz': round(self.R / c.n, 6), 'module_fps': float(c.doc['capture'].get('rate_hz', 0.0)),
                    'fps': round(fps, 3), 'served': served, **c.counts, 'clients': clients,
                    'size': list(c.serve_size), 'render_size': list(c.render_size),
                    'views': len(c.doc['views']), 'render_mpx': c.doc.get('render_mpx'),
                    'latency_ms': _stats(lat), 'tick_ms': _stats(comp[:, 0]), 'views_ms': _stats(comp[:, 1]),
                    'make_ms': _stats(comp[:, 2]),
                    'last_frame_s': meta.get('sim_s'), 'mount': {'deg': deg, 'm': m}, 'mount_method': c.method,
                    'body_mask': dict(c.body_info, url=f'/{quote(tag, safe="")}.body.png') if c.body_info else None,
                    'faults': [_public(f) for f in self.faults.values() if f['tag'] == tag]}
            return {'schema': SCHEMA, 'state': self.state, 'error': self.error, 'stop_reason': self.stop_reason,
                    'tick_hz': self.R, 'steps': self.steps, 'sim_s': round(self._sim_t(), 4),
                    'wall_s': round(now - self.t_stream, 3) if self.t_stream is not None else 0.0,
                    'realtime_factor': round(rtf, 4) if rtf is not None else None,
                    'lag_resets': self.lag_resets, 'behind_s': round(self.behind_s, 3),
                    **self.info, 'cameras': cams, 'faults': [_public(f) for f in self.faults.values()],
                    'notes': list(self.notes), 'limits': LIMITS_TEXT,
                    'clock': 'latency: from the start of the step that rendered a frame to the frame being served '
                             '(time.monotonic on this machine)'}

    def index_html(self) -> str:
        imgs = ''.join(f'<figure><img src="/{quote(t, safe="")}.mjpg" width="480"><figcaption>{_esc(t)}</figcaption>'
                       f'</figure>' for t in self.cams)
        return ('<!doctype html><meta charset="utf-8"><title>Vantage live rig</title><style>body{font:14px sans-serif;'
                'background:#111;color:#ddd}figure{display:inline-block;margin:6px}</style>'
                f'<h3>Vantage live rig · {_esc(self.state)}</h3>{imgs}<p><a href="/status">status</a></p>')

    # ------------------------------------------------------------------ the run
    def run(self) -> int:
        """Connect, stream until stopped, tidy up. 0 when stopped as asked, 2 on an error."""
        carla, a = self.carla, self.a
        world = None
        try:
            self._set_state('connecting')
            client = carla.Client(a.host, int(a.port))
            client.set_timeout(float(a.timeout))
            self.info['server_version'] = client.get_server_version()
            self.info['client_version'] = client.get_client_version()
            world = client.get_world()
            town = a.town or self.rig['scenario']['town']
            cur = world.get_map().name.rsplit('/', 1)[-1]
            if town and cur not in (town, town + '_Opt'):
                self._check()
                world = client.load_world(town)
                cur = world.get_map().name.rsplit('/', 1)[-1]
            self.info['town'] = cur
            self.world = world
            self.original = world.get_settings()
            self._client = client
            try:
                self._setup(world)
                self._loop(world)
            finally:
                self._set_state('stopping')
                self._tidy(world)
            return 0
        except Cancelled:
            return 0
        except Exception as e:                         # noqa: BLE001 (reported in /status, the event and the report)
            import traceback
            with self.lock:
                self.error = f'{type(e).__name__}: {e}'
                self.info['traceback'] = traceback.format_exc()
            self._set_state('failed', error=self.error)
            print(f'{type(e).__name__}: {e}', file=sys.stderr, flush=True)
            return 2
        finally:
            if self.pool is not None:
                self.pool.shutdown(wait=True)
            self.delayer.close()
            with self.lock:
                if self.state != 'failed':
                    self.state = 'stopped'

    def _setup(self, world):
        carla, a, rig = self.carla, self.a, self.rig
        seed = int(a.seed)
        rng = np.random.default_rng([seed, 7919])
        s = world.get_settings()
        s.synchronous_mode = True
        s.fixed_delta_seconds = self.dt
        world.apply_settings(s)
        weather, night = self._weather(world)
        self.info['weather'] = weather
        lib = world.get_blueprint_library()
        spawns = list(world.get_map().get_spawn_points())
        if not spawns:
            raise RuntimeError('The map has no spawn points.')
        order = list(rng.permutation(len(spawns)))
        if a.spawn_point is not None:
            if not 0 <= int(a.spawn_point) < len(spawns):
                raise ValueError(f'--spawn-point {a.spawn_point}: the map has {len(spawns)} spawn points.')
            order = [int(a.spawn_point)] + [k for k in order if k != int(a.spawn_point)]
        veh = rig['vehicle']['blueprint'] if a.vehicle in (None, '', 'auto') else a.vehicle
        bp = self._bp(lib, veh, {'role_name': 'hero'})
        used = 0
        for k in order:
            used += 1
            self.ego = world.try_spawn_actor(bp, spawns[k])
            if self.ego is not None or a.spawn_point is not None:
                break
        if self.ego is None:
            raise RuntimeError('No free spawn point for the ego vehicle.')
        self.actors.append(self.ego)
        cars = [b for b in lib.filter('vehicle.*') if (not b.has_attribute('number_of_wheels') or
                                                        b.get_attribute('number_of_wheels').as_int() == 4)]
        traffic = []
        for k in order[used:]:
            if len(traffic) >= int(a.traffic):
                break
            b = cars[int(rng.integers(len(cars)))]
            if b.has_attribute('role_name'):
                b.set_attribute('role_name', 'autopilot')
            v = world.try_spawn_actor(b, spawns[k])
            if v is not None:
                traffic.append(v)
                self.actors.append(v)
        if len(traffic) < int(a.traffic):
            self._note(f'{len(traffic)} of {a.traffic} traffic vehicles found a free spawn point.')
        world.tick()
        if traffic or not a.still:
            self.tm = self._client.get_trafficmanager(int(a.tm_port))
            self.tm.set_synchronous_mode(True)
            self.tm.set_random_device_seed(seed)
            for v in traffic:
                v.set_autopilot(True, int(a.tm_port))
            if not a.still:
                self.ego.set_autopilot(True, int(a.tm_port))
        if night and hasattr(carla, 'VehicleLightState'):
            VLS = carla.VehicleLightState
            flags = VLS.Position | (VLS.HighBeam if rig['weather'].get('beam') == 'high' else VLS.LowBeam)
            try:
                self.ego.set_light_state(VLS(flags))
            except TypeError:
                self.ego.set_light_state(flags)
        self.info.update(traffic=len(traffic), still=bool(a.still), seed=seed, vehicle=self.ego.type_id)
        self._set_state('warming up')
        warm = float(a.warmup if a.warmup is not None else rig['scenario'].get('warmup_s', 2.0))
        for k in range(max(int(round(warm * self.R)), 1)):
            self._check()
            world.tick(float(a.timeout))
            if k % 20 == 0 and self._parent_gone():
                self.request_stop('the app that started it exited')
        self._ego_origin()
        if getattr(a, 'body_masks', True):
            self._body_masks(world, lib)
        self.pool = cf.ThreadPoolExecutor(max_workers=max(2, min(8, (os.cpu_count() or 2) - 1)),
                                          thread_name_prefix='live-make')

    def _body_masks(self, world, lib):
        """CARLA's own car as each camera sees it at its planned mount: every view rendered once by an instance
        segmentation camera (one step), the ego's actor id kept, each output pixel taking its nearest view pixel
        through the rig's remap tables. Served as /<TAG>.body.png (see the module doc)."""
        from ..core import groundtruth as GT
        try:
            lib.find('sensor.camera.instance_segmentation')
        except (IndexError, RuntimeError):
            self._note('No instance segmentation camera on this server: no body masks.')
            return
        views = {}
        try:
            for c in self.cams.values():
                for vi, v in enumerate(c.doc['views']):
                    bp = self._bp(lib, 'sensor.camera.instance_segmentation',
                                  {'image_size_x': v['width'], 'image_size_y': v['height'], 'fov': v['fov_deg'],
                                   'sensor_tick': 0.0, 'role_name': f'vantage_body_{c.tag}'})
                    act = world.spawn_actor(bp, self._view_tf(c, v, (None, None)), attach_to=self.ego)
                    self.sensor_actors.append(act)
                    q = queue.Queue()
                    act.listen(q.put)
                    views[(c.tag, vi)] = (act, q)
            frame = int(world.tick(float(self.a.timeout)))
            got = {}
            end = time.monotonic() + float(self.a.timeout)
            for key, (_, q) in views.items():
                while key not in got:
                    left = end - time.monotonic()
                    if left <= 0:
                        raise RuntimeError('the instance segmentation views did not arrive')
                    try:
                        img = q.get(timeout=min(left, 0.5))
                    except queue.Empty:
                        self._check()
                        continue
                    if int(img.frame) >= frame:
                        got[key] = img
            ego = int(self.ego.id)
            import cv2
            for c in self.cams.values():
                at = c.doc['atlas']
                ids = np.zeros((int(at['height']), int(at['width'])), np.int64)
                ids[:] = -1
                for vi, v in enumerate(c.doc['views']):
                    img = got[(c.tag, vi)]
                    _, a = GT.decode_instance(GT.bgra(img.raw_data, img.width, img.height))
                    y0 = int(at['offsets'][vi])
                    ids[y0:y0 + a.shape[0], :a.shape[1]] = a
                mx, my = self.remaps[c.doc['remap']['x']], self.remaps[c.doc['remap']['y']]
                inside = mx >= -0.5                         # (−1: the remap tables' 'nothing')
                xi = np.clip(np.rint(mx), 0, ids.shape[1] - 1).astype(np.int64)
                yi = np.clip(np.rint(my), 0, ids.shape[0] - 1).astype(np.int64)
                body = inside & (ids[yi, xi] == ego)
                m = np.where(body, 255, np.where(inside, 128, 0)).astype(np.uint8)
                ok, buf = cv2.imencode('.png', m)
                c.body_png = buf.tobytes() if ok else b''
                c.body_info = {'pixels': int(body.sum()), 'share': round(float(body.sum()) / max(int(inside.sum()), 1),
                                                                          4), 'size': [m.shape[1], m.shape[0]]}
            self._note('Body masks: CARLA\'s own car in each camera at its planned mount (instance segmentation), '
                       'served as /<TAG>.body.png for the body-edge drift indicator.')
        except Cancelled:
            raise
        except Exception as e:                         # noqa: BLE001 (no masks: the stream goes on without)
            self._note(f'No body masks ({type(e).__name__}: {e}).')
        finally:
            for act, _ in views.values():
                for fn in (act.stop, act.destroy):
                    try:
                        fn()
                    except Exception:                  # noqa: BLE001
                        pass
                if act in self.sensor_actors:
                    self.sensor_actors.remove(act)

    def _bp(self, lib, bid, attrs):
        bp = lib.find(bid)
        for k, v in attrs.items():
            if bp.has_attribute(k):
                bp.set_attribute(k, str(v))
        return bp

    def _weather(self, world):
        """(weather report, night) as the recording bridge sets it: a CARLA preset, or the rig's mapped condition."""
        carla, wx = self.carla, self.rig['weather']
        preset = self.a.weather or wx.get('preset') or ''
        if preset and preset != 'rig':
            if not hasattr(carla.WeatherParameters, preset):
                raise ValueError(f'Unknown CARLA weather preset {preset!r}.')
            world.set_weather(getattr(carla.WeatherParameters, preset))
            return {'preset': preset}, preset.endswith('Night')
        p = dict(wx['params'])
        base = carla.WeatherParameters.ClearNoon
        kw = {k: getattr(base, k) for k in ('scattering_intensity', 'mie_scattering_scale', 'rayleigh_scattering_scale',
                                            'dust_storm') if hasattr(base, k)}
        kw.update(p)
        world.set_weather(carla.WeatherParameters(**kw))
        return {'vantage': wx.get('label'), 'params': p}, bool(wx.get('night'))

    def _ego_origin(self):
        """Where the rig's origin (the rear-axle centre on the ground) lies in the CARLA vehicle's own frame: the rear
        wheels' centre and the bounding box's bottom, as the recording bridge reads it."""
        ego = self.ego
        bb = ego.bounding_box
        Minv = np.linalg.inv(np.array(ego.get_transform().get_matrix(), float))
        rear_x = None
        try:
            pts = np.array([[w.position.x, w.position.y, w.position.z] for w in ego.get_physics_control().wheels],
                           float) / 100.0
            loc = (Minv @ np.c_[pts, np.ones(len(pts))].T).T[:, :3]
            if len(loc) >= 4:
                rear_x = float(np.mean(np.sort(loc[:, 0])[:2]))
        except Exception as e:                         # noqa: BLE001 (a vehicle without wheels' data)
            self._note(f'Wheel positions not read ({type(e).__name__}); the rear axle from the rig\'s table.')
        if rear_x is None:
            rear_x = float((self.rig['vehicle'].get('carla') or {}).get('rear_axle_x', -bb.extent.x * 0.6))
        self.O = np.array([rear_x, 0.0, float(bb.location.z - bb.extent.z)])
        self.info['ego_origin_in_actor_m'] = [round(float(v), 4) for v in self.O]
        self.info['bounding_box_lwh_m'] = [round(2 * bb.extent.x, 4), round(2 * bb.extent.y, 4),
                                           round(2 * bb.extent.z, 4)]

    # ------------------------------------------------------------------ cameras
    def _view_tf(self, c, v, mount):
        """The carla.Transform (relative to the ego) of one view of a camera at the planned mount turned / moved by
        mount = (deg, m)."""
        pose = turned_view(v['carla'], c.doc['pose_vehicle'], *mount)
        loc = np.asarray(pose['location'], float) + self.O
        r = pose['rotation']
        return self.carla.Transform(self.carla.Location(x=float(loc[0]), y=float(loc[1]), z=float(loc[2])),
                                    self.carla.Rotation(pitch=float(r['pitch']), yaw=float(r['yaw']),
                                                        roll=float(r['roll'])))

    def _remount(self, c, mount) -> bool:
        """Turn / move a camera's mounted views in place (Actor.set_transform, relative to the ego: see the module doc);
        False when the actors cannot (they are then spawned again)."""
        if not c.views or not all(callable(getattr(a, 'set_transform', None)) for a, _ in c.views):
            return False
        for (act, _), v in zip(c.views, c.doc['views']):
            act.set_transform(self._view_tf(c, v, mount))
        self._mounted(c, mount, 'in place')
        return True

    def _mounted(self, c, mount, method):
        """After a camera was mounted: its mount, how (a fault's re-mount counted), its mount faults applied."""
        with self.lock:
            if not (method == 'spawned again' and c.gen == 1):          # (the first mount is no re-mount)
                c.counts['remounts'] += 1
                c.method = method
            c.mount = tuple(mount)
            for f in self.faults.values():
                if f['tag'] == c.tag and f['kind'] in MOUNT_KINDS and f.get('state') == 'mounting':
                    f['state'] = 'applied'
                    f['applied_s'] = round(self._sim_t(), 4)
                    for row in self.log:
                        if row['id'] == f['id']:
                            row.update(state='applied', applied_s=f['applied_s'])

    def _mount(self, world, lib, c, mount):
        """(Re)spawn a camera's views at the planned mount turned / moved by mount = (deg, m)."""
        self._unmount(c)
        tick = c.n * self.dt - self.dt * 1e-4          # a hair under the period: Unreal carries the remainder over
        ex = self.rig['output']
        views = []
        for vi, v in enumerate(c.doc['views']):
            attrs = {'image_size_x': v['width'], 'image_size_y': v['height'], 'fov': v['fov_deg'],
                     'sensor_tick': tick, 'lens_k': 0.0, 'lens_kcube': 0.0, 'lens_circle_multiplier': 0.0,
                     'lens_circle_falloff': 0.0, 'motion_blur_intensity': 0.0, 'motion_blur_max_distortion': 0.0,
                     'lens_flare_intensity': 0.0, 'bloom_intensity': 0.0, 'chromatic_aberration_intensity': 0.0,
                     'blur_amount': 0.0, 'blur_radius': 0.0, 'exposure_mode': ex.get('exposure', 'manual'),
                     'exposure_compensation': ex.get('exposure_ev', 0.0), 'role_name': f'vantage_live_{c.tag}'}
            bp = self._bp(lib, 'sensor.camera.rgb', attrs)
            act = world.spawn_actor(bp, self._view_tf(c, v, mount), attach_to=self.ego)
            self.sensor_actors.append(act)
            q = queue.Queue()
            act.listen(q.put)
            views.append((act, q))
        with self.lock:
            c.views = views
            c.gen += 1
            c.pending = {}
            c.wait_long = True
        self._mounted(c, mount, 'spawned again')

    def _unmount(self, c):
        for act, _ in c.views:
            for fn in (act.stop, act.destroy):
                try:
                    fn()
                except Exception:                      # noqa: BLE001 (gone already)
                    pass
            if act in self.sensor_actors:
                self.sensor_actors.remove(act)
        c.views = []

    def _collect(self, c, frame, deadline):
        """Move what the views delivered into c.pending; for a due frame, wait until every view of it is in (or each
        view moved past it: CARLA skipped it, or the deadline)."""
        nv = len(c.views)
        while True:
            for vi, (_, q) in enumerate(c.views):
                while True:
                    try:
                        img = q.get_nowait()
                    except queue.Empty:
                        break
                    c.pending.setdefault(int(img.frame), {})[vi] = img
            if frame is None or len(c.pending.get(frame, {})) >= nv:
                return
            if all(any(f > frame and vi in v for f, v in c.pending.items()) for vi in range(nv)):
                c.counts['missed'] += 1
                return
            if time.monotonic() > deadline or self.stop_evt.is_set():
                c.counts['missed'] += 1
                if not self.stop_evt.is_set():
                    self._note(f'{c.tag}: frames that did not arrive in time are counted as missed.')
                return
            time.sleep(0.0005)

    def _dispatch(self, c, frames_meta):
        """Hand every complete frame of a camera to the pool (the newest first in line; older partial ones go)."""
        done = sorted(f for f, v in c.pending.items() if len(v) >= len(c.views))
        if not done:
            return
        for f in [f for f in c.pending if f < done[-1]]:
            if f not in done:
                c.pending.pop(f)
        for f in done:
            views = c.pending.pop(f)
            meta = frames_meta.get(f)
            if meta is None or f <= c.last_sent:
                continue
            c.counts['rendered'] += 1
            with self.lock:
                if c.inflight >= IN_FLIGHT:
                    c.counts['skipped_busy'] += 1
                    continue
                c.inflight += 1
            c.last_sent = f
            m = dict(meta, frame=f, t_views=time.monotonic())
            self.pool.submit(self._make, c, views, m)

    def _make(self, c, views, meta):
        """(a pool thread) One served frame: the views remapped, devignetted, feathered, noise, size, JPEG."""
        import cv2
        try:
            at = c.doc['atlas']
            atlas = np.zeros((int(at['height']), int(at['width']), 3), np.uint8)
            for vi, v in enumerate(c.doc['views']):
                img = views[vi]
                a = np.frombuffer(img.raw_data, np.uint8).reshape(int(img.height), int(img.width), 4)
                y0 = int(at['offsets'][vi])
                atlas[y0:y0 + a.shape[0], :a.shape[1]] = a[..., :3]
            out = cv2.remap(atlas, self.remaps[c.doc['remap']['x']], self.remaps[c.doc['remap']['y']],
                            cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            dv = c.devignette
            if dv is not None:
                out = dv[0].apply(out)
            out = self.CB.feather(out, atlas, c.blend, dv[1:] if dv is not None else None)
            if c.noise:
                out = cv2.add(out, c.noise[meta['frame'] % len(c.noise)])
            if c.serve_size != c.render_size:
                out = cv2.resize(out, c.serve_size, interpolation=cv2.INTER_LINEAR)
            ok, buf = cv2.imencode('.jpg', out, [int(cv2.IMWRITE_JPEG_QUALITY), int(self.a.jpeg_quality)])
            if not ok:
                raise RuntimeError('JPEG encoding failed')
            meta['t_done'] = time.monotonic()
            self._deliver(c, buf.tobytes(), meta)
        except Exception as e:                         # noqa: BLE001 (one bad frame must not stop the stream)
            self._note(f'{c.tag}: a frame could not be made ({type(e).__name__}: {e}).')
        finally:
            with self.lock:
                c.inflight -= 1

    def _deliver(self, c, jpg, meta):
        """Serve a frame through the camera's delivery faults (drop, freeze, delay)."""
        hub = self.hubs[c.tag]
        with self.lock:
            f = self._delivery(c.tag)
            if f['drop'] > 0 and c.rng.random() < f['drop']:
                c.counts['dropped_by_fault'] += 1
                return
            if f['freeze']:
                if c.frozen is None:
                    c.frozen = jpg
                jpg = c.frozen
            delay = f['delay_ms']
        if delay > 0:
            self.delayer.put(meta['t_done'] + delay / 1000.0, hub, jpg, meta)
        else:
            hub.publish(jpg, meta)

    # ------------------------------------------------------------------ the loop
    def _loop(self, world):
        a = self.a
        lib = world.get_blueprint_library()
        timeout = float(a.timeout)
        dur = float(a.duration or 0.0)
        frames_meta = {}
        t_ref, i_ref = None, 0
        i = 0
        t_parent = 0.0
        self._set_state('streaming', cameras=list(self.cams), tick_hz=self.R)
        self.t_stream = time.monotonic()
        while not self.stop_evt.is_set():
            i += 1
            # mounts: the first on the camera's first frame step; a fault's at once, in place (or the views spawned
            # again on the camera's next frame step, so its schedule is kept)
            for c in self.cams.values():
                with self.lock:
                    want, c.want = c.want, None
                if not c.views:
                    if c.due(i):
                        self._mount(world, lib, c, want if want is not None else c.mount)
                    elif want is not None:
                        with self.lock:
                            c.want = c.want or want
                elif want is not None and not self._remount(c, want):
                    if c.due(i):
                        self._mount(world, lib, c, want)
                    else:
                        with self.lock:
                            c.want = c.want or want
            # pace the step to the wall clock; when late, drop the debt rather than catch up in a burst
            now = time.monotonic()
            if t_ref is None:
                t_ref, i_ref = now, i
            target = t_ref + (i - i_ref) * self.dt
            if target > now:
                if self.stop_evt.wait(target - now):
                    break
            elif now - target > MAX_LAG_S:
                with self.lock:
                    self.lag_resets += 1
                    self.behind_s += now - target
                t_ref, i_ref = now, i
            t_step = time.monotonic()
            frame = int(world.tick(timeout))
            t_tick = time.monotonic()
            with self.lock:
                self.steps = i
                self.timeline.append((t_tick, i))
            frames_meta[frame] = {'t_step': t_step, 't_tick': t_tick, 'sim_s': i * self.dt, 'step': i}
            for f in [f for f in frames_meta if f < frame - 4 * max(c.n for c in self.cams.values()) - 8]:
                frames_meta.pop(f)
            for c in self.cams.values():
                if not c.views:
                    continue
                due = c.due(i)
                wait = timeout if c.wait_long else FRAME_WAIT_S
                self._collect(c, frame if due else None, time.monotonic() + wait)
                if due and frame in c.pending and len(c.pending[frame]) >= len(c.views):
                    c.wait_long = False
                self._dispatch(c, frames_meta)
            self._expire()
            if dur > 0 and time.monotonic() - self.t_stream >= dur:
                self.request_stop(f'--duration {dur:g} s reached')
            if time.monotonic() - t_parent > 1.0:
                t_parent = time.monotonic()
                if self._parent_gone():
                    self.request_stop('the app that started it exited')

    def _tidy(self, world):
        """Stop and destroy everything spawned and restore the settings, in the recording bridge's order (a vehicle
        destroyed while on the autopilot makes the traffic manager touch a dead actor)."""
        def quiet(fn, *args):
            try:
                return fn(*args)
            except Exception:                          # noqa: BLE001 (tidying up after anything)
                return None
        client = getattr(self, '_client', None)
        if client is not None:                         # a server that died must not hold each call for --timeout
            quiet(client.set_timeout, min(float(self.a.timeout), 10.0))
        s = quiet(world.get_settings)
        sync = bool(getattr(s, 'synchronous_mode', False)) if s is not None else False
        if self.tm is not None:
            for v in self.actors:
                quiet(v.set_autopilot, False, int(self.a.tm_port))
        if sync:
            quiet(world.tick)
        for act in list(self.sensor_actors):
            quiet(act.stop)
        for act in reversed(list(self.sensor_actors)):
            quiet(act.destroy)
        self.sensor_actors = []
        for c in self.cams.values():
            c.views = []
        for v in reversed(self.actors):
            quiet(v.destroy)
        self.actors = []
        if sync:
            quiet(world.tick)
        if self.tm is not None:
            quiet(self.tm.set_synchronous_mode, False)
        if self.original is not None:
            quiet(world.apply_settings, self.original)

    def report(self) -> dict:
        doc = self.status()
        doc['fault_log'] = self.fault_log()
        doc['rig'] = {'project': self.rig.get('project'), 'variant': self.rig.get('variant'),
                      'carla_version': self.rig.get('carla_version')}
        doc['time'] = time.strftime('%Y-%m-%dT%H:%M:%S')
        if doc.get('state') != 'failed':
            doc.pop('traceback', None)
        return doc


def _stats(x) -> dict | None:
    x = np.asarray(x, float)
    if not len(x):
        return None
    return {'median': round(float(np.median(x)), 2), 'p95': round(float(np.percentile(x, 95)), 2),
            'max': round(float(np.max(x)), 2), 'n': int(len(x))}


def _public(f) -> dict:
    return {k: v for k, v in f.items() if k != 'until'}


def _esc(s) -> str:
    return str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)


# =========================================================================== the app's side (no carla, no Qt)
def stream_url(port: int, tag: str, host: str = '127.0.0.1') -> str:
    return f'http://{host}:{int(port)}/{quote(tag, safe="")}.mjpg'


class LiveClient:
    """Talks to a running live server (standard library only; never through a proxy)."""

    def __init__(self, port: int, host: str = '127.0.0.1', timeout: float = 3.0):
        import urllib.request
        self.host, self.port, self.timeout = host, int(port), float(timeout)
        self._open = urllib.request.build_opener(urllib.request.ProxyHandler({})).open
        self._Request = urllib.request.Request

    def url(self, tag: str) -> str:
        return stream_url(self.port, tag, self.host)

    def _req(self, method, path, body=None) -> dict:
        import urllib.error
        data = json.dumps(body if body is not None else {}).encode('utf-8') if method == 'POST' else None
        req = self._Request(f'http://{self.host}:{self.port}{path}', data=data, method=method,
                            headers={'Content-Type': 'application/json'} if data is not None else {})
        try:
            with self._open(req, timeout=self.timeout) as r:
                return json.loads(r.read() or b'{}')
        except urllib.error.HTTPError as e:
            try:
                msg = json.loads(e.read() or b'{}').get('error') or str(e)
            except ValueError:
                msg = str(e)
            raise LiveError(msg) from None

    def status(self) -> dict:
        return self._req('GET', '/status')

    def fault(self, fault: dict) -> dict:
        return self._req('POST', '/fault', fault)['fault']

    def clear(self, tag=None, fid=None) -> list:
        body = {}
        if tag is not None:
            body['tag'] = tag
        if fid is not None:
            body['id'] = fid
        return self._req('POST', '/fault/clear', body)['cleared']

    def stop(self) -> dict:
        return self._req('POST', '/stop', {})

    def body(self, tag: str) -> bytes:
        """The camera's body mask (/<TAG>.body.png): PNG bytes."""
        import urllib.error
        try:
            url = f'http://{self.host}:{self.port}/{quote(tag, safe="")}.body.png'
            with self._open(url, timeout=self.timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            raise LiveError(f'{tag}: no body mask ({e.code})') from None


def parse_event(line: str):
    """The event dict of a line the live server printed, or None."""
    try:
        d = json.loads(line)
    except ValueError:
        return None
    return d if isinstance(d, dict) and 'event' in d else None


# =========================================================================== command line
def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog='python -m vantage.sim.live',
                                 description='Drive a Vantage rig (sim_rig.json from core.simrig) in CARLA in real '
                                             'time '
                                             'and serve every camera as MJPEG over HTTP, with live fault injection.')
    ap.add_argument('rig', help='sim_rig.json (sim_remap.npz next to it)')
    ap.add_argument('--http-port', type=int, default=0, help='HTTP port (0: a free one, printed)')
    ap.add_argument('--http-host', default='127.0.0.1', help='HTTP address (default: this machine only)')
    ap.add_argument('--host', default='127.0.0.1', help='the CARLA server')
    ap.add_argument('--port', type=int, default=None, help='its RPC port (default: the rig\'s)')
    ap.add_argument('--tm-port', type=int, default=None, help='the traffic manager\'s port (default: the rig\'s)')
    ap.add_argument('--town', default=None, help='CARLA map (default: the rig\'s)')
    ap.add_argument('--weather', default=None, help='a CARLA weather preset or "rig" (default: the rig\'s)')
    ap.add_argument('--traffic', type=int, default=None, help='other vehicles on autopilot (default: the rig\'s)')
    ap.add_argument('--seed', type=int, default=None,
                    help='spawn choices and the traffic manager (default: the rig\'s)')
    ap.add_argument('--vehicle', default=None, help='the ego\'s CARLA blueprint (default: the rig\'s choice)')
    ap.add_argument('--spawn-point', type=int, default=None, help='the ego at this spawn point of the map')
    ap.add_argument('--still', action='store_true', help='the ego stays parked (no autopilot)')
    ap.add_argument('--warmup', type=float, default=None, help='seconds of traffic before the cameras attach')
    ap.add_argument('--duration', type=float, default=0.0, help='stop after this many seconds (0: when asked)')
    ap.add_argument('--timeout', type=float, default=60.0, help='seconds to wait for the server or a first frame')
    ap.add_argument('--cameras', default='', help='only these camera tags, comma separated (default: every one)')
    ap.add_argument('--serve-size', choices=('sensor', 'render'), default='sensor',
                    help='frames at the module\'s sensor size (default) or as rendered (the rig\'s image scale)')
    ap.add_argument('--jpeg-quality', type=int, default=85, help='JPEG quality of the served frames')
    ap.add_argument('--noise', type=float, default=2.0, help='uniform noise 0…2N levels per frame (0: none)')
    ap.add_argument('--keep-vignette', action='store_true', help='keep Unreal\'s vignette in every pinhole view')
    ap.add_argument('--no-body-masks', dest='body_masks', action='store_false',
                    help='do not render CARLA\'s own car per camera at the start (/<TAG>.body.png)')
    ap.add_argument('--parent-pid', type=int, default=0, help='stop when this process exits')
    ap.add_argument('--report', default='', help='write the session (live.json) here at the end')
    return ap


def resolve_args(a, rig: dict):
    """Fill what was left out from the rig's scenario and server settings."""
    sc, srv = rig['scenario'], rig.get('server', {})
    a.traffic = int(a.traffic if a.traffic is not None else sc['traffic'])
    a.seed = int(a.seed if a.seed is not None else sc['seed'])
    a.port = int(a.port if a.port is not None else srv.get('port', 2000))
    a.tm_port = int(a.tm_port if a.tm_port is not None else srv.get('tm_port', 8100))
    if a.weather is None and sc.get('weather') and sc['weather'] != 'project' and \
            not str(sc['weather']).startswith('vantage:'):
        a.weather = sc['weather']
    a.cameras = [t.strip() for t in str(a.cameras or '').split(',') if t.strip()]
    a.jpeg_quality = min(max(int(a.jpeg_quality), 30), 100)
    a.noise = min(max(float(a.noise), 0.0), 20.0)
    return a


def _emit(d):
    print(json.dumps(d, default=_jsonable), flush=True)


def main(argv=None) -> int:
    a = parser().parse_args(argv)
    from .carla_bridge import load_rig
    rig, remaps = load_rig(a.rig)
    resolve_args(a, rig)
    live = LiveRig(None, rig, remaps, a, _emit)

    def on_signal(*_):
        live.request_stop('signal')
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    srv = serve(live, a.http_host, a.http_port)
    port = srv.server_address[1]
    _emit({'event': 'listening', 'port': port, 'url': f'http://{a.http_host}:{port}/',
           'streams': {t: stream_url(port, t, a.http_host) for t in live.cams}})
    code = 2
    try:
        import carla
        live.carla = carla
        code = live.run()
    except Exception as e:                             # noqa: BLE001 (no carla module, …)
        live.error = f'{type(e).__name__}: {e}'
        live.state = 'failed'
        print(live.error, file=sys.stderr, flush=True)
    finally:
        if a.report:
            try:
                tmp = a.report + '.tmp'
                with open(tmp, 'w', encoding='utf-8') as fh:
                    json.dump(live.report(), fh, indent=1, default=_jsonable)
                os.replace(tmp, a.report)
            except OSError as e:
                print(f'Cannot write {a.report}: {e}', file=sys.stderr, flush=True)
        close_server(srv, live)
    _emit({'event': 'stopped', 'state': live.state, 'reason': live.stop_reason, 'error': live.error, 'code': code})
    return code


if __name__ == '__main__':
    sys.exit(main())
