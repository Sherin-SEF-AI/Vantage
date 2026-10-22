"""Live health of a data-collection rig's camera streams (no Qt).

Per stream, from the frames and the times they arrived: frame rate, interval jitter, dropped frames, the age of the
newest frame, the resolution against the module's sensor, brightness and clipping, sharpness and frozen images; per
rig, the spread of the cameras' capture times. Capture times are the software clock when a frame reached Vantage
(``time.monotonic()``), not sensor timestamps: a network or encoder delay shows up in them.

The drift check compares a live image with what the camera should see according to the plan: the body's silhouette
and the horizon, computed through the camera's lens model. A camera that has rotated or moved shows its body edge
somewhere else. The alignment score (median distance from the planned body edge to the nearest edge in the image, in
sensor pixels) is an indicator of that, not a calibration: the installation check measures the pose.

Stream sources are stored per camera tag in ``project.streams``: an RTSP/HTTP URL, a V4L2 device, a GStreamer pipeline,
a video file, or ``synthetic`` / ``synthetic:FPS`` for frames of the parking scene rendered through the camera's lens.
"""
from __future__ import annotations

import collections
import math
import threading
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from .analysis import camera_frames
from .capture import LiveSource
from .lens import resolve

SCHEMA = 'vantage.rig_health/1'
SMALL_W = 320                      # width of the grey thumbnail the image statistics use
PREVIEW_W = 640                    # width of the preview the UI shows (made at most PREVIEW_HZ times a second)
PREVIEW_HZ = 10.0
CLOCK_NOTE = ('capture times are when frames reached Vantage (software clock; a synthetic source stamps its own '
              'schedule), not sensor timestamps: network and encoder delays are included')


@dataclass
class HealthSettings:
    """Thresholds of the verdicts. The frame rate a camera should deliver is its module's fps."""
    fps_warn: float = 0.9              # measured / nominal below this: warning
    fps_fault: float = 0.5             # ... below this: fault
    drop_factor: float = 1.5           # an interval longer than this × the nominal one has dropped frames in it
    drops_warn: float = 0.01           # share of frames dropped in the window
    drops_fault: float = 0.05
    age_warn_s: float = 0.5            # newest frame older than this: warning
    age_fault_s: float = 2.0           # ... than this: fault (stalled)
    connect_s: float = 8.0             # no frame this long after starting: fault
    dark_mean: float = 20.0            # mean grey level below this: warning
    bright_mean: float = 235.0         # ... above this: warning
    clip_warn: float = 0.10            # share of pixels at 253..255 (or 0..2 for under-exposure, ×3)
    sharp_warn: float = 8.0            # variance of the Laplacian of the 320 px grey image below this: warning
    freeze_frames: int = 10            # this many identical consecutive frames: fault
    sync_warn_ms: float = 20.0         # median capture-time offset between cameras
    sync_fault_ms: float = 50.0
    drift_warn_px: float = 12.0        # alignment score above this: warning (the camera may have moved)
    window_s: float = 5.0              # statistics window

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d) -> 'HealthSettings':
        s = HealthSettings()
        for k, v in (d or {}).items():
            if k in HealthSettings.__dataclass_fields__:
                try:
                    setattr(s, k, type(getattr(s, k))(v))
                except (TypeError, ValueError):
                    pass
        return s


# ====================================================================================== statistics
def _grey_small(frame: np.ndarray, width=SMALL_W) -> np.ndarray:
    import cv2
    h, w = frame.shape[:2]
    sw = min(width, w)
    sh = max(1, round(h * sw / w))
    g = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    return cv2.resize(g, (sw, sh), interpolation=cv2.INTER_AREA)


def _small_stats(g: np.ndarray) -> dict:
    import cv2
    return {'mean': float(g.mean()), 'clip_dark': float((g <= 2).mean()), 'clip_bright': float((g >= 253).mean()),
            'sharpness': float(cv2.Laplacian(g, cv2.CV_64F).var())}


def image_stats(frame: np.ndarray) -> dict:
    """Brightness, clipping (share of pixels at 0..2 and 253..255) and sharpness (variance of the Laplacian) of one
    frame, on a 320 px grey thumbnail."""
    return _small_stats(_grey_small(frame))


class StreamStats:
    """Statistics of one stream; add() is called on the reader thread, summary() on any other."""

    def __init__(self, nominal_fps=30.0, expected_size=None, settings: HealthSettings | None = None):
        self.nominal = float(nominal_fps) if nominal_fps and nominal_fps > 0 else 30.0
        self.expected = tuple(expected_size) if expected_size else None
        self.st = settings or HealthSettings()
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        with getattr(self, 'lock', threading.Lock()):
            self.frames = 0
            self.first_t = None
            self.last_t = None
            self.ema_dt = None
            self.times = collections.deque(maxlen=2000)
            self.intervals = collections.deque(maxlen=2000)
            self.size = None
            self.img = {}
            self.prev_small = None
            self.frozen_run = 0                     # consecutive frames identical to the one before

    def add(self, frame: np.ndarray, t: float):
        small = _grey_small(frame)
        img = _small_stats(small)
        with self.lock:
            if self.last_t is not None:
                dt = max(1e-6, t - self.last_t)
                self.intervals.append((t, dt))
                self.ema_dt = dt if self.ema_dt is None else 0.9 * self.ema_dt + 0.1 * dt
            else:
                self.first_t = t
            self.last_t = t
            self.times.append(t)
            self.frames += 1
            self.size = (int(frame.shape[1]), int(frame.shape[0]))
            same = self.prev_small is not None and np.array_equal(self.prev_small, small)
            self.frozen_run = self.frozen_run + 1 if same else 0
            self.prev_small = small
            self.img = img

    def summary(self, now: float | None = None) -> dict:
        now = time.monotonic() if now is None else now
        st = self.st
        with self.lock:
            win = [dt for (t, dt) in self.intervals if t >= now - st.window_s] if self.intervals else []
            if not win and self.intervals:
                win = [dt for _, dt in list(self.intervals)[-30:]]
            nominal_dt = 1.0 / self.nominal
            gaps = [dt for dt in win if dt > st.drop_factor * nominal_dt]
            dropped = int(sum(max(0, round(dt / nominal_dt) - 1) for dt in gaps))
            n = len(win)
            out = {
                'frames': self.frames,
                'fps': (1.0 / self.ema_dt) if self.ema_dt else 0.0,
                'nominal_fps': self.nominal,
                'jitter_ms': float(np.std(win) * 1000) if n >= 2 else 0.0,
                'dropped': dropped,
                'dropped_share': dropped / (n + dropped) if n + dropped else 0.0,
                'age_s': (now - self.last_t) if self.last_t is not None else None,
                'size': list(self.size) if self.size else None,
                'expected_size': list(self.expected) if self.expected else None,
                'size_ok': (self.size == self.expected) if (self.size and self.expected) else None,
                'frozen_frames': self.frozen_run,
                'seconds': (self.last_t - self.first_t) if self.first_t is not None and self.last_t is not None else 0.0,
                **self.img,
            }
        return out

    def recent_times(self, since: float) -> np.ndarray:
        with self.lock:
            return np.array([t for t in self.times if t >= since], float)


def verdict(s: dict, settings: HealthSettings, started: float | None = None, now: float | None = None,
            error: str = '', drift_px: float | None = None) -> tuple[str, list]:
    """('ok' | 'warning' | 'fault' | 'starting', [(severity, text)]) for one camera's summary."""
    st = settings
    now = time.monotonic() if now is None else now
    reasons = []
    if not s.get('frames'):
        if error:
            return 'fault', [('fault', error)]
        if started is not None and now - started > st.connect_s:
            return 'fault', [('fault', f'no frames after {now - started:.0f} s')]
        return 'starting', [('info', 'connecting…')]
    age = s.get('age_s')
    if error:
        reasons.append(('fault', error))
    if age is not None and age > st.age_fault_s:
        reasons.append(('fault', f'stalled: no frame for {age:.1f} s'))
    elif age is not None and age > st.age_warn_s:
        reasons.append(('warning', f'last frame {age * 1000:.0f} ms old'))
    if s.get('seconds', 0) >= 1.0 and age is not None and age <= st.age_fault_s:
        r = s['fps'] / s['nominal_fps'] if s['nominal_fps'] else 1.0
        if r < st.fps_fault:
            reasons.append(('fault', f'{s["fps"]:.1f} fps of {s["nominal_fps"]:g}'))
        elif r < st.fps_warn:
            reasons.append(('warning', f'{s["fps"]:.1f} fps of {s["nominal_fps"]:g}'))
        if s['dropped_share'] > st.drops_fault:
            reasons.append(('fault', f'{s["dropped_share"] * 100:.1f} % frames dropped'))
        elif s['dropped_share'] > st.drops_warn:
            reasons.append(('warning', f'{s["dropped_share"] * 100:.1f} % frames dropped'))
    if s.get('frozen_frames', 0) >= st.freeze_frames:
        reasons.append(('fault', f'frozen: {s["frozen_frames"] + 1} identical frames'))
    if s.get('size_ok') is False:
        reasons.append(('warning', f'{s["size"][0]}×{s["size"][1]}, the module is {s["expected_size"][0]}×{s["expected_size"][1]}'))
    m = s.get('mean')
    if m is not None:
        if m < st.dark_mean:
            reasons.append(('warning', f'too dark (mean {m:.0f})'))
        elif m > st.bright_mean:
            reasons.append(('warning', f'too bright (mean {m:.0f})'))
        if s.get('clip_bright', 0) > st.clip_warn:
            reasons.append(('warning', f'overexposed: {s["clip_bright"] * 100:.0f} % clipped'))
        if s.get('clip_dark', 0) > 3 * st.clip_warn:
            reasons.append(('warning', f'underexposed: {s["clip_dark"] * 100:.0f} % black'))
        if s.get('sharpness', 1e9) < st.sharp_warn:
            reasons.append(('warning', f'blurred or covered lens (sharpness {s["sharpness"]:.1f})'))
    if drift_px is not None and drift_px > st.drift_warn_px:
        reasons.append(('warning', f'body edge {drift_px:.0f} px off the plan: the camera may have moved'))
    worst = 'fault' if any(r[0] == 'fault' for r in reasons) else 'warning' if any(r[0] == 'warning' for r in reasons) \
        else 'ok'
    return worst, reasons


def sync_skew(times: dict) -> tuple[float | None, str]:
    """(median capture-time offset in ms of the worst camera to the first one, its key), from each camera's recent
    capture times. Cameras that are triggered together show ~0; free-running ones up to half a frame interval."""
    keys = [k for k, t in times.items() if len(t) >= 3]
    if len(keys) < 2:
        return None, ''
    ref = np.sort(times[keys[0]])
    worst, who = 0.0, ''
    for k in keys[1:]:
        t = np.asarray(times[k], float)
        i = np.clip(np.searchsorted(ref, t), 1, len(ref) - 1)
        d = np.minimum(np.abs(t - ref[i - 1]), np.abs(t - ref[i]))
        med = float(np.median(d)) * 1000
        if med > worst:
            worst, who = med, k
    return worst, who


# ====================================================================================== sources
def describe(spec: str) -> str:
    """'synthetic', 'rtsp', 'http', 'device', 'gstreamer', 'file' or '' for a stream spec."""
    import os
    s = (spec or '').strip()
    if not s:
        return ''
    if s.lower().startswith('synthetic'):
        return 'synthetic'
    low = s.lower()
    if low.startswith('rtsp://'):
        return 'rtsp'
    if low.startswith(('http://', 'https://')):
        return 'http'
    if '!' in s:
        return 'gstreamer'
    if s.isdigit() or s.startswith('/dev/video'):
        return 'device'
    if os.path.exists(s):
        return 'file'
    return 'stream'


def shown(spec: str) -> str:
    """A stream spec for labels, reports and the undo history: a password in a URL is replaced by ***."""
    from .credentials import mask
    return mask(spec)


def synthetic_fps(spec: str, default: float) -> float:
    try:
        return float(spec.split(':', 1)[1]) if ':' in spec else float(default)
    except ValueError:
        return float(default)


class SyntheticSource:
    """A stand-in camera: the parking scene rendered once through the camera's lens (on this source's thread), then
    delivered at `fps` with fresh sensor noise per frame, stamped with its time on the camera's clock (frames the host
    is too busy to make are skipped, and show as drops). Same interface as capture.LiveSource."""

    def __init__(self, render, fps=30.0, size=None, noise=2, spec='synthetic', on_frame=None):
        self.render = render
        self.fps = max(0.5, float(fps))
        self.size_wanted = tuple(size) if size else None
        self.noise = max(1, int(noise))
        self.spec = spec
        self.on_frame = on_frame
        self.error = ''
        self.frames = 0
        self.size = None
        self.stamp = 0.0
        self._frame = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

    @property
    def running(self):
        return self._thread is not None and self._thread.is_alive()

    def start(self):
        if self.running and not self._stop.is_set():
            return
        self._stop = threading.Event()
        self.error = ''
        self._thread = threading.Thread(target=self._run, args=(self._stop,), name='synthetic', daemon=True)
        self._thread.start()

    def _run(self, stop):
        import cv2
        try:
            base = self.render()
        except Exception as e:
            self.error = f'cannot render the synthetic view: {e}'
            return
        if base.ndim == 2:
            base = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB)
        if self.size_wanted and (base.shape[1], base.shape[0]) != self.size_wanted:
            base = cv2.resize(base, self.size_wanted, interpolation=cv2.INTER_LINEAR)
        n = self.noise
        base = np.minimum(base, 254 - 2 * n).astype(np.uint8)            # + noise + flicker stays below 256
        rng = np.random.default_rng(7)
        pats = [rng.integers(0, 2 * n + 1, base.shape[:2], dtype=np.uint8)[..., None] for _ in range(4)]
        period = 1.0 / self.fps
        nxt = time.monotonic()
        k = 0
        while not stop.is_set():
            f = base + pats[k % len(pats)]
            if k % 7 == 3:                                       # a little flicker, like auto-exposure hunting
                f += 1
            k += 1
            t = nxt          # the frame's time on the camera's own clock: a busy host does not make it late
            if t > time.monotonic():
                t = time.monotonic()
            with self._lock:
                self._frame = f
                self.frames += 1
                self.size = (f.shape[1], f.shape[0])
                self.stamp = t
            cb = self.on_frame
            if cb is not None:
                try:
                    cb(f, t)
                except Exception:
                    pass
            nxt += period
            delay = nxt - time.monotonic()
            if delay < -period:                                  # fell behind: do not burst to catch up
                nxt = time.monotonic()
            elif delay > 0:
                stop.wait(delay)

    def latest(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def stop(self, wait=True):
        self._stop.set()
        t = self._thread
        if wait and t is not None and t is not threading.current_thread():
            t.join(timeout=1.5)
        self._thread = None


def synthetic_render(project, engine, cam, pose=None, width=768):
    """A callable that renders the camera's view of the parking scene (at `width`, upscaled to the sensor later)."""
    def render():
        from .synth import parking_world, render_capture
        intr = resolve(project.module_for(cam), project.lens_for(cam))
        w = min(width, intr.width)
        h = max(1, round(w * intr.height / intr.width))
        return render_capture(project, engine, cam, w, h, pose=pose, world=parking_world(project), seed=cam.slot,
                              noise=0.0)
    return render


# ====================================================================================== the rig
@dataclass
class _Stream:
    key: str                        # camera tag
    uid: str
    spec: str
    source: object
    stats: StreamStats
    started: float = 0.0
    retry_at: float | None = None
    backoff: float = 1.0
    error: str = ''
    preview: object = None
    preview_t: float = 0.0
    drift: float | None = None      # alignment indicator of a recent frame (None: not checked, or no body edge seen)
    drift_checked: bool = False
    plock: threading.Lock = field(default_factory=threading.Lock)


class RigMonitor:
    """Reads every configured stream on its own thread and keeps the statistics. poll() reconnects dropped streams
    with a growing back-off; snapshot() is the state the page, the batch mode and the assistant read."""

    def __init__(self, settings: HealthSettings | None = None):
        self.settings = settings or HealthSettings()
        self.streams: dict = {}
        self.unconfigured: list = []
        self.running = False
        self.t0 = 0.0

    # ---------------------------------------------------------------- set-up
    def configure(self, project, cams, engine_factory=None):
        """cams: the cameras to watch; their sources come from project.streams[tag]. Cameras without one are listed
        as not configured. engine_factory() gives an engine for synthetic sources (rendered on their own threads)."""
        self.stop(wait=False)
        self.streams = {}
        self.update(project, cams, engine_factory)

    def update(self, project, cams, engine_factory=None):
        """Follow the project: streams whose source changed restart (when running), the others keep their state."""
        want = {c.tag: ((getattr(project, 'streams', {}) or {}).get(c.tag, '').strip(), c) for c in cams}
        for key in list(self.streams):
            if key not in want or want[key][0] != self.streams[key].spec:
                self.streams.pop(key).source.stop(wait=False)
        self.unconfigured = [k for k, (spec, _) in want.items() if not spec]
        now = time.monotonic()
        out = {}
        for key, (spec, c) in want.items():
            if not spec:
                continue
            mod = project.module_for(c)
            s = self.streams.get(key)
            if s is None:
                s = _Stream(key, c.uid, spec, None, StreamStats(mod.fps, (mod.width_px, mod.height_px), self.settings))
                s.source = self._make(s, project, c, engine_factory)
                if self.running:
                    s.started = now
                    s.source.start()
            else:
                s.stats.nominal = float(mod.fps) if mod.fps and mod.fps > 0 else 30.0
                s.stats.expected = (mod.width_px, mod.height_px)
            out[key] = s
        self.streams = out                                       # in the cameras' order

    def _make(self, s, project, cam, engine_factory):
        def on_frame(frame, t, s=s):
            s.stats.add(frame, t)
            if t - s.preview_t >= 1.0 / PREVIEW_HZ:
                import cv2
                h, w = frame.shape[:2]
                pw = min(PREVIEW_W, w)
                pv = cv2.resize(frame, (pw, max(1, round(h * pw / w))), interpolation=cv2.INTER_AREA)
                with s.plock:
                    s.preview, s.preview_t = pv, t
        if describe(s.spec) == 'synthetic':
            mod = project.module_for(cam)
            eng = engine_factory() if engine_factory else None
            if eng is None:
                from .analysis import Engine
                eng = Engine()
            return SyntheticSource(synthetic_render(project.snapshot(), eng, cam), synthetic_fps(s.spec, mod.fps),
                                   (mod.width_px, mod.height_px), spec=s.spec, on_frame=on_frame)
        return LiveSource(s.spec, on_frame=on_frame)

    # ---------------------------------------------------------------- running
    def start(self):
        now = time.monotonic()
        self.t0 = now
        for s in self.streams.values():
            s.started, s.retry_at, s.backoff, s.error = now, None, 1.0, ''
            s.stats.reset()
            s.source.start()
        self.running = True

    def stop(self, wait=True):
        """wait=False (the UI) only signals the readers; each exits on its own when its current read returns."""
        for s in self.streams.values():
            if s.source is not None:
                s.source.stop(wait)
        self.running = False

    def reset_stats(self):
        now = time.monotonic()
        for s in self.streams.values():
            s.stats.reset()
            s.started = now

    def poll(self, now: float | None = None):
        """Reconnect streams whose reader stopped (could not open, stopped delivering, or hangs in a read), after
        1, 2, 4 … 15 s."""
        if not self.running:
            return
        now = time.monotonic() if now is None else now
        hang = max(3 * self.settings.age_fault_s, self.settings.connect_s)
        for s in self.streams.values():
            src = s.source
            if src.running:
                last = s.stats.last_t
                if last is not None and now - last < 1.0:
                    s.backoff = 1.0                              # delivering again
                    continue
                if now - max(s.started, last or 0.0) < hang:
                    continue
                src.stop(False)                                  # alive but silent: drop it and reconnect
                src.error = src.error or f'no frame for {now - max(s.started, last or 0.0):.0f} s'
            if s.retry_at is None:
                s.error = src.error or 'the stream stopped'
                s.retry_at = now + s.backoff
                s.backoff = min(15.0, s.backoff * 2)
            elif now >= s.retry_at:
                s.retry_at = None
                s.started = now
                src.start()

    def preview(self, key):
        s = self.streams.get(key)
        if s is None:
            return None
        with s.plock:
            return s.preview

    def latest(self, key):
        s = self.streams.get(key)
        return s.source.latest() if s is not None else None

    def snapshot(self, now: float | None = None) -> dict:
        now = time.monotonic() if now is None else now
        st = self.settings
        cams = {}
        times = {}
        for key, s in self.streams.items():
            summ = s.stats.summary(now)
            err = s.error if (s.retry_at is not None or (not s.source.running and s.source.error)) else ''
            if err and s.retry_at is not None:
                err += f' · retry in {max(0.0, s.retry_at - now):.0f} s'
            status, reasons = verdict(summ, st, s.started, now, err, s.drift)
            cams[key] = {'camera': key, 'source': shown(s.spec), 'kind': describe(s.spec), 'status': status,
                         'reasons': [r[1] for r in reasons if r[0] != 'info'], 'drift_px': s.drift,
                         'drift_checked': s.drift_checked, **summ}
            if summ['frames'] and (summ['age_s'] or 0) < st.age_fault_s:
                times[key] = s.stats.recent_times(now - st.window_s)
        for tag in self.unconfigured:
            cams[tag] = {'camera': tag, 'source': '', 'kind': '', 'status': 'off', 'reasons': ['no stream configured'],
                         'frames': 0}
        skew, who = sync_skew(times)
        live = [c for c in cams.values() if c['status'] in ('ok', 'warning')]
        fps = [(c['fps'], k) for k, c in cams.items() if c.get('frames') and c.get('fps')]
        rig = {'cameras': len(cams), 'streaming': len(live),
               'faults': sum(1 for c in cams.values() if c['status'] == 'fault'),
               'warnings': sum(1 for c in cams.values() if c['status'] == 'warning'),
               'not_configured': sum(1 for c in cams.values() if c['status'] == 'off'),
               'worst_fps': min(fps)[0] if fps else None, 'worst_fps_camera': min(fps)[1] if fps else None,
               'sync_skew_ms': skew, 'sync_skew_camera': who or None,
               'sync_status': None if skew is None else 'fault' if skew > st.sync_fault_ms else
               'warning' if skew > st.sync_warn_ms else 'ok',
               'seconds': now - self.t0 if self.running else 0.0, 'clock': CLOCK_NOTE}
        return {'schema': SCHEMA, 'cameras': cams, 'rig': rig}


# ====================================================================================== drift check
@dataclass
class ExpectedView:
    """What a camera should see according to the plan, at a working resolution."""
    w: int
    h: int
    sensor: tuple
    valid: np.ndarray               # (h, w) inside the image circle
    body: np.ndarray                # (h, w) the vehicle (and rig) body
    edge: np.ndarray                # (K, 2) int working-pixel coordinates of the body's edge (not the circle's)
    normal: np.ndarray              # (K, 2) unit normals of the edge, pointing into the body
    contours: list                  # [(N, 2)] polylines of the body's outline, sensor pixels
    horizon: list                   # [(N, 2)] horizon, sensor pixels
    key: tuple = ()


def view_key(project, cam) -> tuple:
    m = project.vehicle.mesh
    return (cam.x, cam.y, cam.z, cam.yaw, cam.pitch, cam.roll, cam.module_id, cam.lens_id, bool(cam.calib),
            id(m), m.n_tris if m is not None else 0)


def expected_view(project, engine, cam, width=640) -> ExpectedView:
    import cv2
    intr = resolve(project.module_for(cam), project.lens_for(cam))
    W, H = int(intr.width), int(intr.height)
    w = min(width, W)
    h = max(1, round(w * H / W))
    us = (np.arange(w) + 0.5) / w * W
    vs = (np.arange(h) + 0.5) / h * H
    U, V = np.meshgrid(us, vs)
    d_opt, th = intr.unproject(U.ravel(), V.ravel())
    R, W2O = camera_frames(cam)
    dw = d_opt @ W2O
    pos = cam.pos()
    valid = th <= intr.max_theta
    with np.errstate(divide='ignore', invalid='ignore'):
        tg = np.where(dw[:, 2] < -1e-6, -pos[2] / dw[:, 2], np.inf)
    body = np.zeros(len(dw), bool)
    if engine.vehicle_geom(project) is not None:
        depth, _ = engine.cube(pos).lookup(dw)
        body = valid & np.isfinite(depth) & (depth < tg)
    valid = valid.reshape(h, w)
    body = body.reshape(h, w)
    k = np.ones((3, 3), np.uint8)
    inner = cv2.erode(body.astype(np.uint8), k).astype(bool)
    deep = cv2.erode(valid.astype(np.uint8), k, iterations=3).astype(bool)
    edge = body & ~inner & deep
    jj, ii = np.nonzero(edge)
    soft = cv2.GaussianBlur(body.astype(np.float32), (0, 0), 1.5)
    n = np.c_[cv2.Sobel(soft, cv2.CV_32F, 1, 0)[jj, ii], cv2.Sobel(soft, cv2.CV_32F, 0, 1)[jj, ii]]
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    sx, sy = W / w, H / h
    # the outline, without the stretches where the body only meets the image circle or the frame
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
            k0 = int(np.argmin(keep))                                   # start the walk off the body's edge
            pts, keep = np.roll(pts, -k0, 0), np.roll(keep, -k0)
            cut = np.flatnonzero(np.diff(keep.astype(np.int8)))
            runs = [pts[a + 1:b + 1] for a, b in zip(cut[::2], cut[1::2])]
            if len(cut) % 2:
                runs.append(pts[cut[-1] + 1:])
        for r in runs:
            if len(r) >= 3:
                r = cv2.approxPolyDP(r.reshape(-1, 1, 2).astype(np.int32), 0.8, False).reshape(-1, 2).astype(float)
                contours.append(np.c_[(r[:, 0] + 0.5) * sx, (r[:, 1] + 0.5) * sy])
    az = np.radians(np.arange(0.0, 360.0, 0.5))
    dirs = np.c_[np.cos(az), np.sin(az), np.zeros_like(az)]
    u, v, _, ok = intr.project_dirs(dirs @ W2O.T)
    horizon, cur = [], []
    for a, b, o in zip(u, v, ok):
        if o:
            cur.append((a, b))
        elif len(cur) > 1:
            horizon.append(np.array(cur))
            cur = []
        else:
            cur = []
    if len(cur) > 1:
        horizon.append(np.array(cur))
    return ExpectedView(w, h, (W, H), valid, body, np.c_[ii, jj], n, contours, horizon, view_key(project, cam))


def alignment(frame: np.ndarray, ev: ExpectedView, search=0.05, min_support=0.3) -> float | None:
    """How far (sensor pixels, median) the body edge in the live image lies from the planned one: along each planned
    edge point's normal, the nearest gradient peak above the image's noise. Small when the camera sits where it was
    planned. None when the plan shows no body edge, or too little of it has an edge in the image (the body and what
    is behind it look alike, or the lens is covered)."""
    import cv2
    if len(ev.edge) < 20:
        return None
    g = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    g = cv2.resize(g, (ev.w, ev.h), interpolation=cv2.INTER_AREA).astype(np.float32)
    g = cv2.GaussianBlur(g, (0, 0), 1.0)
    G = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    G[~ev.valid] = 0
    r = max(4, int(search * ev.w))
    ks = np.arange(-r, r + 1)
    P = ev.edge[:, None, :] + ks[None, :, None] * ev.normal[:, None, :]
    S = G[np.clip(np.round(P[..., 1]).astype(int), 0, ev.h - 1), np.clip(np.round(P[..., 0]).astype(int), 0, ev.w - 1)]
    floor = max(8.0, 3.0 * float(np.median(G[ev.valid])))
    peak = np.zeros_like(S, bool)
    peak[:, 1:-1] = (S[:, 1:-1] >= S[:, :-2]) & (S[:, 1:-1] >= S[:, 2:]) & (S[:, 1:-1] > floor)
    best = np.where(peak, np.abs(ks)[None, :], 1 << 20).argmin(1)
    ok = peak[np.arange(len(S)), best]
    if ok.mean() < min_support:
        return None
    return float(np.median(np.abs(ks[best[ok]]))) * ev.sensor[0] / ev.w


# ====================================================================================== batch
def run_for(project, cams, seconds, settings=None, engine_factory=None, connect_timeout=20.0, progress=None) -> dict:
    """Watch every configured stream for `seconds` (after the streams deliver their first frame, or the connect
    timeout) and return the snapshot, with the drift indicator of each camera's last frame."""
    mon = RigMonitor(settings)
    mon.configure(project, cams, engine_factory)
    mon.start()
    try:
        t_end = time.monotonic() + connect_timeout
        while time.monotonic() < t_end:
            mon.poll()
            if all(s.stats.frames > 0 for s in mon.streams.values()):
                break
            time.sleep(0.05)
        mon.reset_stats()
        t_end = time.monotonic() + max(0.5, float(seconds))
        while time.monotonic() < t_end:
            mon.poll()
            if progress:
                progress(1 - (t_end - time.monotonic()) / max(0.5, float(seconds)))
            time.sleep(0.05)
        eng = engine_factory() if engine_factory else None
        if eng is None:
            from .analysis import Engine
            eng = Engine()
        by_tag = {c.tag: c for c in cams}
        for key, s in mon.streams.items():
            f = s.source.latest()
            if f is not None and key in by_tag:
                try:
                    s.drift, s.drift_checked = alignment(f, expected_view(project, eng, by_tag[key])), True
                except Exception:
                    s.drift = None
        snap = mon.snapshot()
    finally:
        mon.stop()
    snap['project'] = project.name
    snap['variant'] = project.variant.name
    snap['seconds'] = float(seconds)
    snap['settings'] = mon.settings.to_dict()
    snap['time'] = time.strftime('%Y-%m-%dT%H:%M:%S')
    return snap


def jsonable(doc):
    """Snapshot for JSON: numpy values to floats, the ages rounded."""
    def fix(o):
        if isinstance(o, dict):
            return {k: fix(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [fix(v) for v in o]
        if isinstance(o, (np.floating, float)):
            v = float(o)
            return round(v, 4) if math.isfinite(v) else None
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.bool_):
            return bool(o)
        return o
    return fix(doc)
