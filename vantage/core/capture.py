"""Image sources for the installation check: files on disk and live camera streams.

A live source is anything OpenCV can open: a V4L2 device (/dev/video0 or just 0), a GStreamer pipeline
(contains '!', e.g. the nvarguscamerasrc / v4l2src pipelines of a Jetson GMSL deserialiser), an RTSP/HTTP URL
or a video file. A background thread keeps only the newest frame so a grab is never stale.

GStreamer pipelines: the OpenCV wheels from PyPI (and so the downloadable bundles) are built without GStreamer. Then
the pipeline runs in the system's own gst-launch-1.0 (GstPipe), which on a Jetson has NVIDIA's plug-ins
(nvarguscamerasrc, nvvidconv): Vantage appends a conversion to raw BGR and a sink writing to a pipe it reads.
"""
from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import threading
import time
from collections import deque

import numpy as np

from .credentials import mask

IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.pgm', '.ppm')


def load_image(path: str) -> np.ndarray:
    """RGB (or grey) uint8 array. Reads through numpy so non-ASCII paths work on every platform."""
    import cv2
    data = np.fromfile(path, np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f'Not an image OpenCV can read: {os.path.basename(path)}')
    if img.dtype != np.uint8:
        # scale by the type's range, never per image (that would shift the black level and differ camera to camera)
        if np.issubdtype(img.dtype, np.integer):
            img = (img.astype(np.float64) * (255.0 / np.iinfo(img.dtype).max)).round()
        else:
            img = np.asarray(img, np.float64) * (255.0 if np.nanmax(img) <= 1.0 else 1.0)
        img = np.clip(np.nan_to_num(img), 0, 255).astype(np.uint8)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB if img.shape[2] == 4 else cv2.COLOR_BGR2RGB)
    return img


def save_image(path: str, img: np.ndarray):
    import cv2
    ext = os.path.splitext(path)[1] or '.png'
    out = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(ext, out)
    if not ok:
        raise ValueError(f'Cannot encode {ext} images')
    buf.tofile(path)


def match_folder(folder: str, tags) -> dict:
    """tag -> image path for files named after the camera tag (F.png, cam_QR.jpg, ML-capture.tif ...).
    An exact stem match wins over a stem that only contains the tag as a separate word."""
    import re
    files = [f for f in sorted(os.listdir(folder)) if f.lower().endswith(IMAGE_EXTS)]
    out = {}
    for tag in tags:
        exact = [f for f in files if os.path.splitext(f)[0].lower() == tag.lower()]
        word = [f for f in files if re.search(rf'(^|[^a-z0-9]){re.escape(tag.lower())}([^a-z0-9]|$)', os.path.splitext(f)[0].lower())]
        pick = (exact or word or [None])[0]
        if pick:
            out[tag] = os.path.join(folder, pick)
    return out


def default_spec(port: int) -> str:
    """Device a camera on GMSL port n usually appears as on a Jetson with a quad deserialiser."""
    return f'/dev/video{max(0, port - 1)}'


# ------------------------------------------------------------------ GStreamer pipelines without OpenCV's GStreamer
GST_SINK = 'vantagesink'
GST_TIMEOUT_S = 15.0                # a Jetson's camera pipeline can take a few seconds to negotiate
_GST_CV = []


def opencv_has_gstreamer() -> bool:
    """Whether this OpenCV was built with GStreamer (the PyPI wheels are not)."""
    if not _GST_CV:
        import cv2
        line = next((l for l in cv2.getBuildInformation().splitlines() if l.strip().startswith('GStreamer:')), '')
        _GST_CV.append('YES' in line.split(':', 1)[-1].upper())
    return _GST_CV[0]


def gst_pipeline(spec: str, fd: int) -> list[str]:
    """gst-launch-1.0's arguments for an OpenCV-style pipeline: its appsink (if any) replaced by a conversion to raw
    BGR and a sink that writes the frames to file descriptor fd."""
    s = re.sub(r'!\s*appsink\b[^!]*$', '', spec.strip()).strip().rstrip('!').strip()
    return shlex.split(s) + ['!', 'videoconvert', '!', 'video/x-raw,format=BGR', '!', 'fdsink', f'name={GST_SINK}',
                             f'fd={fd}', 'sync=false']


_CAPS = re.compile(rf'{GST_SINK}\.GstPad:sink: caps = (.*)')


def parse_caps(text: str) -> dict | None:
    """width, height and frame rate from the caps gst-launch -v reports for the sink's pad, or None."""
    m = _CAPS.search(text)
    if not m:
        return None
    caps = m.group(1)
    w, h = re.search(r'width=\(int\)(\d+)', caps), re.search(r'height=\(int\)(\d+)', caps)
    if not (w and h):
        return None
    fr = re.search(r'framerate=\(fraction\)(\d+)/(\d+)', caps)
    fps = int(fr.group(1)) / int(fr.group(2)) if fr and int(fr.group(2)) else 0.0
    return {'width': int(w.group(1)), 'height': int(h.group(1)), 'fps': fps}


class GstPipe:
    """cv2.VideoCapture's isOpened / read / get / set / release for a GStreamer pipeline run by the system's
    gst-launch-1.0. Frames arrive as raw BGR on a pipe (rows padded to 4 bytes, as GStreamer lays them out); the
    frame size comes from the caps gst-launch -v prints. Raises RuntimeError with GStreamer's message when the
    pipeline does not start."""

    def __init__(self, spec: str, timeout: float = GST_TIMEOUT_S):
        from ..sysenv import system_env
        gst = shutil.which('gst-launch-1.0')
        if not gst:
            raise RuntimeError('this OpenCV has no GStreamer support and gst-launch-1.0 is not installed '
                               '(install gstreamer1.0-tools)')
        r, w = os.pipe()
        try:
            self.proc = subprocess.Popen([gst, '-v', *gst_pipeline(spec, w)], stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, pass_fds=(w,),
                                         env=system_env(), text=True, errors='replace', bufsize=1)
        except OSError:
            os.close(r)
            raise
        finally:
            os.close(w)
        self._fd = r
        self.caps = None
        self.lines = deque(maxlen=60)
        ready = threading.Event()
        threading.Thread(target=self._drain, args=(ready,), name='gst-launch', daemon=True).start()
        ready.wait(timeout)
        if self.caps is None:
            err = [l for l in self.lines if 'ERROR' in l or 'erroneous' in l or 'no element' in l]
            self.release()
            raise RuntimeError((err[0] if err else f'no frame format from the pipeline within {timeout:g} s').strip())
        self.width, self.height = self.caps['width'], self.caps['height']
        self.stride = (self.width * 3 + 3) // 4 * 4
        self.frame_bytes = self.stride * self.height

    def _drain(self, ready):
        """Read gst-launch's messages (it blocks on a full pipe otherwise); the sink's caps set the frame size."""
        try:
            for line in self.proc.stdout:
                self.lines.append(line.rstrip())
                if self.caps is None:
                    c = parse_caps(line)
                    if c is not None:
                        self.caps = c
                        ready.set()
        except (OSError, ValueError):
            pass
        finally:
            ready.set()

    def isOpened(self) -> bool:
        return self._fd is not None

    def read(self):
        if self._fd is None:
            return False, None
        buf = bytearray(self.frame_bytes)
        view, n = memoryview(buf), 0
        try:
            while n < self.frame_bytes:
                k = os.readv(self._fd, [view[n:]])
                if k <= 0:
                    return False, None
                n += k
        except OSError:
            return False, None
        rows = np.frombuffer(buf, np.uint8).reshape(self.height, self.stride)
        return True, np.ascontiguousarray(rows[:, :self.width * 3]).reshape(self.height, self.width, 3)

    def get(self, prop) -> float:
        import cv2
        return {cv2.CAP_PROP_FRAME_WIDTH: float(getattr(self, 'width', 0)),
                cv2.CAP_PROP_FRAME_HEIGHT: float(getattr(self, 'height', 0)),
                cv2.CAP_PROP_FPS: float((self.caps or {}).get('fps', 0.0))}.get(prop, 0.0)

    def set(self, prop, value) -> bool:
        return False

    def release(self):
        fd, self._fd = self._fd, None
        if fd is not None:
            os.close(fd)                    # the sink's next write fails: the pipeline stops
        p = self.proc
        if p.poll() is None:
            p.terminate()
            try:
                p.wait(2.0)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait(2.0)


def _open(spec: str):
    import cv2
    from .credentials import resolve
    s = resolve(spec.strip())              # a password kept out of the project file is put back here
    if s.isdigit():
        return cv2.VideoCapture(int(s))
    if '!' in s:
        return cv2.VideoCapture(s, cv2.CAP_GSTREAMER) if opencv_has_gstreamer() else GstPipe(s)
    if s.startswith('/dev/video'):
        return cv2.VideoCapture(s, cv2.CAP_V4L2)
    return cv2.VideoCapture(s)


class LiveSource:
    """Background reader for one stream; latest() returns a copy of the newest RGB frame."""

    def __init__(self, spec: str, on_frame=None):
        self.spec = spec
        self.error = ''
        self.frames = 0
        self.size = None
        self.stamp = 0.0               # time.monotonic() when the newest frame arrived
        self.on_frame = on_frame       # callable(frame, stamp), called on the reader thread for every frame
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
        # a fresh event per reader: a reader still stuck in read() after stop() keeps its own (set) event and exits
        self._stop = threading.Event()
        self.error = ''
        self._thread = threading.Thread(target=self._run, args=(self._stop,), name=f'live:{mask(self.spec)}', daemon=True)
        self._thread.start()

    def _run(self, stop=None):
        stop = stop or self._stop
        import cv2
        try:
            cap = _open(self.spec)
        except Exception as e:
            self.error = mask(f'Cannot open {self.spec}: {e}')       # errors reach labels, reports and the assistant
            return
        if cap is None or not cap.isOpened():
            self.error = f'Cannot open {mask(self.spec)}'
            return
        fails = 0
        is_file = os.path.isfile(self.spec)
        fps = cap.get(cv2.CAP_PROP_FPS) if is_file else 0
        period = 1.0 / fps if is_file and 1 <= fps <= 240 else 1 / 30      # a file plays at its own frame rate
        nxt = time.monotonic()
        try:
            while not stop.is_set():
                ok, f = cap.read()
                if not ok or f is None:
                    fails += 1
                    if is_file and fails < 3:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    if fails > 60:
                        self.error = f'{mask(self.spec)} stopped delivering frames'
                        break
                    time.sleep(0.03)
                    continue
                fails = 0
                if f.ndim == 3:
                    f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                if stop.is_set():
                    break
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
                    except Exception:           # a statistics hook must never stop the stream
                        pass
                if is_file:
                    nxt += period
                    delay = nxt - time.monotonic()
                    if delay < -period:
                        nxt = time.monotonic()
                    elif delay > 0:
                        stop.wait(delay)
        finally:
            cap.release()

    def latest(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def stop(self, wait=True):
        """wait=False returns at once (a reader blocked in read() exits by itself when the read returns)."""
        self._stop.set()
        t = self._thread
        if wait and t is not None and t is not threading.current_thread():
            t.join(timeout=1.5)
        self._thread = None
