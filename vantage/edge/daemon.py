"""The watchdog: reads the rig's camera streams, checks their health and whether a camera has been knocked, and alerts.

Health is vantage.core.rig_health's (its RigMonitor reads the streams and reconnects them, its StreamStats and verdict
judge frame rate, dropped frames, stalls, frozen images, exposure, sharpness and resolution, its capture_offsets the
skew between cameras against the sync plan's offsets that the configuration carries). Knocks are vantage.edge.knock's.
Conditions become alerts through vantage.edge.alerts (debounced), shown on the local status page
(vantage.edge.status_http) and sent to the configured channels.

Bounded cost. Every frame costs rig health's 320 px grey thumbnail; knock analysis runs on at most analysis_hz frames
per camera per second, downscaled to analysis_width, on one worker thread fed by a queue of one frame per camera
(a frame arriving while that camera's previous one waits replaces it: the analysis never falls behind, it thins out).
Memory: per stream the newest frame, a preview and a 2000-entry timing history; per camera two analysis images and the
learned body template; the alert history is capped (500). Nothing grows with the running time.

Clocks. Live, times are time.monotonic() when a frame arrived. A dry run (replay) uses the recording's own stamps as
the clock and runs as fast as the computer can (deterministic: the same recording and faults give the same alerts), or
at the recording's pace with realtime=True.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import types

import numpy as np

from ..core import rig_health as RH
from . import config as CF
from .alerts import AlertBook, Notifier
from .knock import KnockMonitor, RigConsensus

STATUS_SCHEMA = 'vantage.edge_status/1'
REPORT_SCHEMA = 'vantage.edge_run/1'
CHECK_LABEL = {'stall': 'stalled', 'frame_rate': 'frame rate', 'drops': 'dropped frames', 'frozen': 'frozen image',
               'dark': 'too dark', 'bright': 'too bright', 'sharpness': 'blurred or covered',
               'resolution': 'resolution',
               'stream': 'stream', 'knock': 'knocked', 'sync': 'sync skew', 'drift': 'body edge off the plan',
               'host': 'every stream at once'}


def classify(text: str) -> str:
    """The check a rig_health verdict reason belongs to."""
    t = text.lower()
    for key, words in (('stall', ('stalled', 'last frame')), ('frame_rate', (' fps of ',)), ('drops', ('dropped',)),
                       ('frozen', ('frozen',)), ('dark', ('too dark', 'underexposed')),
                       ('bright', ('too bright', 'overexposed')), ('sharpness', ('blurred',)),
                       ('resolution', ('the module is',)), ('drift', ('body edge',))):
        if any(w in t for w in words):
            return key
    return 'stream'


def usage() -> dict:
    """This process's CPU time (s) and memory (current and peak resident MB, from /proc on Linux)."""
    t = os.times()
    out = {'cpu_s': t.user + t.system, 'rss_mb': None, 'peak_rss_mb': None}
    try:
        with open('/proc/self/status') as fh:
            for line in fh:
                if line.startswith('VmRSS:'):
                    out['rss_mb'] = int(line.split()[1]) / 1024.0
                elif line.startswith('VmHWM:'):
                    out['peak_rss_mb'] = int(line.split()[1]) / 1024.0
    except OSError:
        import resource
        out['peak_rss_mb'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    return out


class _Module:
    def __init__(self, fps, w, h):
        self.fps, self.width_px, self.height_px = fps, w, h


class Watchdog:
    def __init__(self, cfg, notify: bool = True, replay: bool = False):
        self.cfg = cfg = CF.load(cfg)
        self.replay = replay
        self.health = RH.HealthSettings.from_dict(cfg['health'])
        self.kst = cfg['knock']
        al = cfg['alerts']
        self.book = AlertBook(al['raise_s'], al['clear_s'])
        self.grace_s = al['grace_s']
        self._settled = {}                           # tag -> the stream start whose statistics were reset
        self.end_status = None
        self.notifier = Notifier(al, cfg['rig']) if notify else None
        self.cams = {c['tag']: c for c in cfg['cameras']}
        mods = {}
        for c in cfg['cameras']:
            ln = c.get('lens') or {}
            mods[c['tag']] = _Module(c['fps'], int(c.get('width') or ln.get('width') or 0) or None,
                                     int(c.get('height') or ln.get('height') or 0) or None)
        shim = types.SimpleNamespace(streams={c['tag']: c['source'] or f'(none:{c["tag"]})' for c in cfg['cameras']},
                                     module_for=lambda c: mods[c.tag])
        cams = [types.SimpleNamespace(tag=c['tag'], uid=c.get('uid') or c['tag']) for c in cfg['cameras']]
        for m in mods.values():
            if m.width_px is None:          # no size known: the resolution check is left out
                m.width_px = m.height_px = 0
        self.monitor = RH.RigMonitor(self.health)
        self.monitor.configure(shim, cams, planned=cfg['planned_offsets'] or {})
        for tag, s in self.monitor.streams.items():
            if mods[tag].width_px == 0:
                s.stats.expected = None
            s.source.on_frame = self._hook(tag, s.source.on_frame)
            if s.spec.startswith('ros2:'):
                from .sources import Ros2Source
                s.source = Ros2Source(s.spec, on_frame=s.source.on_frame)
        self.knock = {c['tag']: KnockMonitor(c, self.kst, seed=i) for i, c in enumerate(cfg['cameras'])
                      if self.kst['enabled']}
        self.consensus = RigConsensus(self.kst['window_s'])
        if len(self.knock) >= 3:                     # a common motion is told from a knock with three cameras or more
            for k in self.knock.values():
                k.consensus = self.consensus
        self.last_an = {}
        self.next_due = {}                           # tag -> when its next frame is analysed
        self.ring = {}                               # tag -> [(frame, t)]: candidates for the pair
        self.queue = {}                              # tag -> (grey, t, w, h): the frame waiting for analysis
        self.qlock = threading.Condition()
        self.analysed = 0
        self.skipped = 0
        self.stop_ev = threading.Event()
        self.state = {}
        self.slock = threading.Lock()
        self.t_start = None
        self.cpu0 = None
        self.timeline = []                           # (t, {tag: status}) every timeline_s of the clock
        self.timeline_s = 0.5
        self._last_tl = None
        self.worker = None
        self.http = None

    # ---------------------------------------------------------------- frames
    def _hook(self, tag, inner):
        def on_frame(frame, t):
            if inner is not None:
                inner(frame, t)
            self.offer(tag, frame, t)
        return on_frame

    def offer(self, tag, frame, t):
        """A frame for knock analysis (called for every frame). At most analysis_hz frames per camera are analysed,
        each with the frame about pair_s before it; only the frames that can be that partner are kept (references,
        a few per camera)."""
        if tag not in self.knock:
            return
        due = self.next_due.get(tag)
        ring = self.ring.setdefault(tag, [])
        pair = self.kst['pair_s']
        if due is None or t < due - 1e-9:
            if due is None or t >= due - pair - 0.6 / max(self.cams[tag]['fps'], 1.0):
                ring.append((frame, t))
                if len(ring) > 8:
                    del ring[0]
            if due is None:
                self.next_due[tag] = t + pair
            return
        prev = min((x for x in ring if x[1] < t - 1e-9), key=lambda x: abs((t - x[1]) - pair), default=None)
        ring.clear()
        ring.append((frame, t))
        self.next_due[tag] = t + 1.0 / max(self.kst['analysis_hz'], 1e-3)
        if prev is None:
            return
        if self.replay:
            self._analyse(tag, prev, frame, t)
            return
        with self.qlock:
            if tag in self.queue:
                self.skipped += 1                    # the previous one was not analysed in time: thinned out
            self.queue[tag] = (prev, frame, t)
            self.qlock.notify()

    def _small(self, frame):
        import cv2
        h, w = frame.shape[:2]
        aw = int(min(self.kst['analysis_width'], w))
        g = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if aw != w:
            g = cv2.resize(g, (aw, max(1, round(h * aw / w))), interpolation=cv2.INTER_AREA)
        return g

    def _analyse(self, tag, prev, frame, t):
        try:
            h, w = frame.shape[:2]
            g0 = self._small(prev[0]) if prev[0].shape == frame.shape else None
            self.knock[tag].feed(self._small(frame), t, w, h, None if g0 is None else (g0, prev[1]))
        except Exception as e:                       # noqa: BLE001 - one bad frame never stops the watchdog
            self.knock[tag].reject[f'error: {type(e).__name__}'] += 1
        self.analysed += 1

    def _work(self):
        while not self.stop_ev.is_set():
            with self.qlock:
                while not self.queue and not self.stop_ev.is_set():
                    self.qlock.wait(0.5)
                if self.stop_ev.is_set():
                    return
                tag = min(self.queue, key=lambda k: self.queue[k][2])      # the oldest waiting frame first
                prev, frame, t = self.queue.pop(tag)
            self._analyse(tag, prev, frame, t)

    # ---------------------------------------------------------------- evaluation
    def conditions(self, now: float, snap: dict) -> dict:
        seen = {}
        for tag, s in self.monitor.streams.items():
            c = snap['cameras'].get(tag)
            if c is None:
                continue
            err = s.error if (s.retry_at is not None or (not s.source.running and s.source.error)) else ''
            status, reasons = RH.verdict(c, self.health, s.started, now, err, None)
            settling = now - s.started < self.grace_s            # a stream's first seconds: its timing settles
            for sev, text in reasons:
                if settling and classify(text) in ('frame_rate', 'drops', 'stall'):
                    continue
                if sev in ('fault', 'warning'):
                    key = (tag, classify(text))
                    if key not in seen or (sev == 'fault' and seen[key][0] != 'fault'):
                        seen[key] = (sev, text)
            k = self.knock.get(tag)
            if k is not None:
                ks = k.status()
                if ks['state'] == 'knock':
                    seen[(tag, 'knock')] = ('fault', f'knocked: the {ks["method"]} moved {ks["size_deg"]:.2f}° '
                                                     f'(seen at {ks["at_s"] - (self.t_start or 0):.1f} s)')
        # every stream losing frames at the same moment is the computer falling behind (or one shared link), not
        # each camera: one rig alert instead of one per camera
        timing = ('drops', 'frame_rate', 'stall')
        live = [t for t in self.monitor.streams if snap['cameras'].get(t, {}).get('frames')]
        hit = [t for t in live if any((t, k) in seen for k in timing)]
        if len(live) >= 3 and len(hit) >= max(3, 0.75 * len(live)):
            kinds = sorted({k for t in hit for k in timing if (t, k) in seen})
            for t in hit:
                for k in timing:
                    seen.pop((t, k), None)
            cpu = (self.state or {}).get('process', {}).get('cpu_percent_of_one_core')
            seen[('rig', 'host')] = ('warning', f'{len(hit)} of {len(live)} streams were late at the same moment '
                                                f'({", ".join(CHECK_LABEL[k] for k in kinds)}): the computer fell '
                                                'behind or a shared link stalled'
                                     + (f' (the watchdog uses {cpu:.0f} % of a core)' if cpu is not None else ''))
        r = snap['rig']
        if r.get('sync_status') in ('warning', 'fault') and now - (self.t_start or now) >= self.grace_s:
            seen[('rig', 'sync')] = (r['sync_status'], f'capture skew {r["sync_skew_ms"]:.1f} ms '
                                                       f'({r["sync_skew_camera"]}) against the '
                                                       f'{"sync plan" if r["sync_against"] == "plan" else "others"}')
        return seen

    def tick(self, now: float | None = None) -> list:
        now = time.monotonic() if now is None else now
        if self.t_start is None:
            self.t_start = now
        for key, s in self.monitor.streams.items():
            # a stream's first seconds (connecting, the first frames bunched or late) are not its timing: the
            # statistics start again once the grace period is over
            if s.started is not None and now - s.started >= self.grace_s and self._settled.get(key) != s.started:
                self._settled[key] = s.started
                if self.grace_s > 0:
                    with s.stats.lock:                   # a frozen image is not timing: its count carries on
                        keep = (s.stats.prev_small, s.stats.frozen_run)
                    s.stats.reset()
                    with s.stats.lock:
                        s.stats.prev_small, s.stats.frozen_run = keep
        snap = self.monitor.snapshot(now)
        seen = self.conditions(now, snap)
        events = self.book.update(now, seen)
        if self.notifier is not None and (events or self.notifier.gpio is not None):
            self.notifier.send(events, self.book.any_fault())
        cams = {}
        for tag, c in snap['cameras'].items():
            d = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in c.items() if k not in ('drift_px',)}
            if tag in self.knock:
                d['knock'] = self.knock[tag].status()
            d['alerts'] = sorted(k[1] for k in self.book.active if k[0] == tag)
            cams[tag] = d
        u = usage()
        if self.cpu0 is None:
            self.cpu0 = (u['cpu_s'], time.monotonic())
        wall_s = max(time.monotonic() - self.cpu0[1], 1e-6)
        doc = {'schema': STATUS_SCHEMA, 'rig': self.cfg['rig'], 'clock_s': round(float(now - self.t_start), 3),
               'replay': self.replay, 'cameras': cams, 'rig_health': snap['rig'],
               'alerts': self.book.to_list(active_only=True),
               'counts': {'faults': sum(1 for a in self.book.active.values() if a['severity'] == 'fault'),
                          'warnings': sum(1 for a in self.book.active.values() if a['severity'] == 'warning'),
                          'raised_total': self.book.counter},
               'analysis': {'analysed': self.analysed, 'thinned': self.skipped, 'hz': self.kst['analysis_hz']},
               'process': {'cpu_percent_of_one_core': round(100.0 * (u['cpu_s'] - self.cpu0[0]) / wall_s, 1),
                           'rss_mb': u['rss_mb'], 'peak_rss_mb': u['peak_rss_mb']},
               'channels': self.notifier.channels() if self.notifier is not None else {}}
        with self.slock:
            self.state = doc
        if self._last_tl is None or now - self._last_tl >= self.timeline_s - 1e-9:
            self._last_tl = now
            self.timeline.append((round(now - self.t_start, 3),
                                  {tag: (c['status'], sorted(k[1] for k in self.book.active if k[0] == tag))
                                   for tag, c in snap['cameras'].items()}))
            if len(self.timeline) > 20000:
                del self.timeline[:1000]
        return events

    def status(self) -> dict:
        with self.slock:
            return dict(self.state)

    # ---------------------------------------------------------------- running
    def serve_http(self):
        h = self.cfg['alerts']['http']
        if int(h.get('port', 0)) <= 0:
            return None
        from .status_http import serve
        self.http = serve(self, h.get('bind', '127.0.0.1'), int(h['port']))
        return self.http

    def run(self, seconds: float | None = None, period: float = 0.5, status_every: float = 5.0):
        """Live: read the streams until stop() (or for seconds)."""
        self.worker = threading.Thread(target=self._work, name='edge-analysis', daemon=True)
        self.worker.start()
        self.monitor.start()
        t0 = time.monotonic()
        self.t_start = t0
        last_status = 0.0
        if self.notifier is not None:
            self.notifier.info(f'watching {len(self.cams)} camera(s) of {self.cfg["rig"]}')
        try:
            while not self.stop_ev.is_set():
                now = time.monotonic()
                if seconds is not None and now - t0 >= seconds:
                    break
                self.monitor.poll(now)
                self.tick(now)
                if self.notifier is not None and now - last_status >= status_every:
                    last_status = now
                    self.notifier.status(self.status())
                self.stop_ev.wait(period)
        finally:
            self.tick()
            self.close()

    def replay_frames(self, items, realtime: bool = False, step: float = 0.1, end_s: float | None = None):
        """Dry run: items yields (t, tag, frame) in time order (vantage.edge.sources.merged)."""
        now = None
        t_first = None
        wall0 = time.monotonic()
        for s in self.monitor.streams.values():
            s.started = 0.0
        for t, tag, f in items:
            if self.stop_ev.is_set():
                break
            if t_first is None:
                t_first = t
                self.t_start = t
                for s in self.monitor.streams.values():
                    s.started = t
                now = t
            while now + step <= t:
                now += step
                self.tick(now)
            if realtime:
                d = (t - t_first) - (time.monotonic() - wall0)
                if d > 0:
                    time.sleep(d)
            s = self.monitor.streams.get(tag)
            if s is None:
                continue
            s.stats.add(f, t)
            self.offer(tag, f, t)
        if now is not None:
            self.tick(now)
            self.end_status = self.status()          # the cameras as the recording ends (before its tail)
            stop = now + (end_s or 0.0)
            while now + step <= stop + 1e-9:
                now += step
                self.tick(now)
            self.tick(now)
        return now

    def stop(self):
        self.stop_ev.set()
        with self.qlock:
            self.qlock.notify_all()

    def close(self):
        self.stop()
        self.monitor.stop(wait=True)
        if self.http is not None:
            self.http.shutdown()
            self.http.server_close()
            self.http = None
        if self.notifier is not None:
            self.notifier.close()

    def report(self) -> dict:
        """The run's record: what was raised and cleared, each camera's last state, the process's cost."""
        st = self.status()
        t0 = self.t_start or 0.0
        alerts = [dict(a, raised_s=round(a['raised_t'] - t0, 3),
                       cleared_s=None if a.get('cleared_t') is None else round(a['cleared_t'] - t0, 3))
                  for a in self.book.to_list()]
        return {'schema': REPORT_SCHEMA, 'rig': self.cfg['rig'], 'replay': self.replay, 'clock_start': t0,
                'seconds': st.get('clock_s'), 'alerts': alerts,
                'cameras': (self.end_status or st).get('cameras', {}),
                'rig_health': (self.end_status or st).get('rig_health'),
                'analysis': st.get('analysis'), 'process': st.get('process'),
                'timeline': [{'t': t, 'cameras': {k: {'status': s, 'alerts': a} for k, (s, a) in c.items()}}
                             for t, c in self.timeline]}


# ---------------------------------------------------------------------------------------------- dry runs
def recording_rates(series: dict) -> dict:
    """{tag: frame rate} of each camera's recording, from its stamps (median interval)."""
    out = {}
    for tag, s in series.items():
        times = getattr(s, 'times', None)
        if times and len(times) > 2:
            dt = float(np.median(np.diff(times)))
            if dt > 0:
                out[tag] = 1.0 / dt
        elif getattr(s, 'fps', None):
            out[tag] = float(s.fps)
    return out


def recorded_offsets(path: str) -> dict | None:
    """The cameras' planned capture offsets of a simulated drive (its rig/sim_rig.json), or None."""
    for p in (os.path.join(path, 'rig', 'sim_rig.json'), os.path.join(os.path.dirname(path), 'rig', 'sim_rig.json')):
        if os.path.isfile(p):
            with open(p, encoding='utf-8') as fh:
                return RH.schedule_offsets(json.load(fh))
    return None


def dry_run(cfg, recording: str, faults=None, realtime: bool = False, nominal: str = 'recording',
            notify: bool = False, tail_s: float = 0.0, progress=None, cancel=None) -> dict:
    """Run the watchdog over a recording, with faults written into it (vantage.edge.inject). nominal 'recording'
    judges each camera's frame rate against the recording's own rate (a 10 Hz drive of a 30 fps camera is not a
    fault); 'config' against the configuration's. Returns the run's report plus 'recording' and 'faults'."""
    from .inject import Injector, clean
    from .sources import merged, open_recording
    cfg = CF.load(cfg)
    tags = [c['tag'] for c in cfg['cameras']]
    if isinstance(recording, dict):                  # {tag: series} already open (a synthetic drive)
        series, kind = {t: s for t, s in recording.items() if t in tags}, 'frames'
        recording = '(frames)'
    else:
        series, kind = open_recording(recording, tags)
    if not series:
        raise ValueError(f'no frames of {", ".join(tags)} found in {recording}')
    notes = []
    cfg = json.loads(json.dumps(cfg))
    cfg['cameras'] = [c for c in cfg['cameras'] if c['tag'] in series]
    missing = [t for t in tags if t not in series]
    if missing:
        notes.append(f'not in the recording: {", ".join(missing)}')
    if nominal == 'recording':
        rates = recording_rates(series)
        for c in cfg['cameras']:
            r = rates.get(c['tag'])
            if r and abs(r - c['fps']) > 0.02 * c['fps']:
                notes.append(f'{c["tag"]}: recorded at {r:.2f} fps (configured {c["fps"]:g}); judged against the '
                             'recording\'s rate')
                c['fps'] = round(r, 4)
    po = recorded_offsets(recording) if os.path.exists(recording) else None
    if po is not None:
        cfg['planned_offsets'] = {k: list(v) for k, v in po.items()}
        notes.append('capture skew judged against the simulated rig\'s recorded schedule (rig/sim_rig.json)')
    sizes = {}
    for tag, s in series.items():
        for _, f in s.frames():
            sizes[tag] = (f.shape[1], f.shape[0])
            break
    if nominal == 'recording':
        for c in cfg['cameras']:
            sz = sizes.get(c['tag'])
            if sz and (c.get('width'), c.get('height')) != sz:
                notes.append(f'{c["tag"]}: recorded at {sz[0]}×{sz[1]} (the sensor is {c.get("width")}×'
                             f'{c.get("height")}); judged against the recording\'s size')
                c['width'], c['height'] = sz
    lenses = {c['tag']: c['lens'] for c in cfg['cameras'] if c.get('lens')}
    inj = Injector(faults, lenses)
    wd = Watchdog(cfg, notify=notify, replay=True)
    n = [0]
    total = sum(len(s) for s in series.values()) or 1

    def items():
        for it in merged(series, inj):
            n[0] += 1
            if progress and n[0] % 25 == 0:
                progress(min(0.99, n[0] / total), f'{n[0]} frames')
            if cancel and cancel():
                wd.stop()
                return
            yield it
    t_wall = time.monotonic()
    u0 = usage()
    wd.replay_frames(items(), realtime=realtime, end_s=tail_s)
    u1 = usage()
    if wd.notifier is not None:
        wd.notifier.close()
    rep = wd.report()
    rep.update({'recording': {'path': recording, 'kind': kind, 'cameras': sorted(series), 'frames': n[0],
                              'sizes': {k: list(v) for k, v in sizes.items()}},
                'faults': clean(faults), 'notes': notes, 'cancelled': bool(cancel and cancel()),
                'cost': {'wall_s': round(time.monotonic() - t_wall, 3), 'cpu_s': round(u1['cpu_s'] - u0['cpu_s'], 3),
                         'peak_rss_mb': u1['peak_rss_mb']}})
    return rep


# ---------------------------------------------------------------------------------------------- command line
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog='python -m vantage.edge',
                                 description='Vantage edge watchdog: rig health and knock detection on the car\'s '
                                             'camera streams, with local alerts. It only reads and alerts.')
    ap.add_argument('--config', required=True, help='the configuration Vantage exported (edge.json)')
    ap.add_argument('--dry-run', metavar='RECORDING', help='replay a recording (nuScenes folder, image folders, '
                                                           'video files or a ROS 2 bag) instead of the live streams')
    ap.add_argument('--inject', metavar='JSON', help='faults to write into the dry run: a JSON list or a file')
    ap.add_argument('--realtime', action='store_true', help='dry run at the recording\'s own pace')
    ap.add_argument('--seconds', type=float, help='stop after this long (live)')
    ap.add_argument('--report', metavar='PATH', help='write the run\'s record (JSON) here at the end')
    ap.add_argument('--no-http', action='store_true', help='no status page')
    ap.add_argument('--check', action='store_true', help='check the configuration and exit')
    ap.add_argument('--threads', type=int, default=1, help='OpenCV worker threads (default 1: a predictable, bounded '
                                                            'share of the CPU)')
    a = ap.parse_args(argv)
    try:
        import cv2
        cv2.setNumThreads(max(1, int(a.threads)))
    except ImportError:
        print('vantage.edge: OpenCV is needed (pip install opencv-python-headless)', file=sys.stderr)
        return 2
    try:
        cfg = CF.load(a.config)
    except (OSError, ValueError) as e:
        print(f'vantage.edge: {e}', file=sys.stderr)
        return 2
    if a.check:
        print(f'{cfg["rig"]}: {len(cfg["cameras"])} camera(s): ' + ', '.join(
            f'{c["tag"]} {c["source"] or "(no source)"}' for c in cfg['cameras']))
        return 0
    if a.no_http:
        cfg['alerts']['http']['port'] = 0
    if a.dry_run:
        faults = []
        if a.inject:
            txt = open(a.inject, encoding='utf-8').read() if os.path.isfile(a.inject) else a.inject
            faults = json.loads(txt)
        try:
            rep = dry_run(cfg, a.dry_run, faults, realtime=a.realtime, notify=True)
        except (OSError, ValueError) as e:
            print(f'vantage.edge: {e}', file=sys.stderr)
            return 2
        for al in rep['alerts']:
            print(f'{al["raised_s"]:9.2f}  {al["camera"]:>6} '
                  f'{al["check"]:<10} {al["severity"]:<7} {al["message"]}')
        if a.report:
            with open(a.report, 'w', encoding='utf-8') as fh:
                json.dump(rep, fh, indent=1, default=str)
        return 1 if any(x['severity'] == 'fault' for x in rep['alerts']) else 0
    wd = Watchdog(cfg, notify=True)
    if not a.no_http:
        try:
            srv = wd.serve_http()
            if srv is not None:
                print(f'status page: http://{srv.server_address[0]}:{srv.server_address[1]}/', flush=True)
        except OSError as e:
            print(f'vantage.edge: no status page ({e})', file=sys.stderr)
    import signal

    def _stop(*_):
        wd.stop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _stop)
        except ValueError:
            pass
    wd.run(a.seconds)
    rep = wd.report()
    if a.report:
        with open(a.report, 'w', encoding='utf-8') as fh:
            json.dump(rep, fh, indent=1, default=str)
    return 1 if any(x['severity'] == 'fault' for x in rep['alerts']) else 0
