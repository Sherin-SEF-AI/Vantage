"""Image sources for the installation check: files on disk and live camera streams.

A live source is anything OpenCV can open: a V4L2 device (/dev/video0 or just 0), a GStreamer pipeline
(contains '!', e.g. the nvarguscamerasrc / v4l2src pipelines of a Jetson GMSL deserialiser), an RTSP/HTTP URL
or a video file. A background thread keeps only the newest frame so a grab is never stale.
"""
from __future__ import annotations

import os
import threading
import time

import numpy as np

IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.pgm', '.ppm')


def load_image(path: str) -> np.ndarray:
    """RGB (or grey) uint8 array. Reads through numpy so non-ASCII paths work on every platform."""
    import cv2
    data = np.fromfile(path, np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f'Not an image OpenCV can read: {os.path.basename(path)}')
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
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


def _open(spec: str):
    import cv2
    s = spec.strip()
    if s.isdigit():
        return cv2.VideoCapture(int(s))
    if '!' in s:
        return cv2.VideoCapture(s, cv2.CAP_GSTREAMER)
    if s.startswith('/dev/video'):
        return cv2.VideoCapture(s, cv2.CAP_V4L2)
    return cv2.VideoCapture(s)


class LiveSource:
    """Background reader for one stream; latest() returns a copy of the newest RGB frame."""

    def __init__(self, spec: str):
        self.spec = spec
        self.error = ''
        self.frames = 0
        self.size = None
        self._frame = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

    @property
    def running(self):
        return self._thread is not None and self._thread.is_alive()

    def start(self):
        if self.running:
            return
        self._stop.clear()
        self.error = ''
        self._thread = threading.Thread(target=self._run, name=f'live:{self.spec}', daemon=True)
        self._thread.start()

    def _run(self):
        import cv2
        try:
            cap = _open(self.spec)
        except Exception as e:
            self.error = f'Cannot open {self.spec}: {e}'
            return
        if cap is None or not cap.isOpened():
            self.error = f'Cannot open {self.spec}'
            return
        fails = 0
        is_file = os.path.isfile(self.spec)
        try:
            while not self._stop.is_set():
                ok, f = cap.read()
                if not ok or f is None:
                    fails += 1
                    if is_file and fails < 3:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    if fails > 60:
                        self.error = f'{self.spec} stopped delivering frames'
                        break
                    time.sleep(0.03)
                    continue
                fails = 0
                if f.ndim == 3:
                    f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                with self._lock:
                    self._frame = f
                    self.frames += 1
                    self.size = (f.shape[1], f.shape[0])
                if is_file:
                    time.sleep(1 / 30)
        finally:
            cap.release()

    def latest(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def stop(self):
        self._stop.set()
        t = self._thread
        if t is not None and t is not threading.current_thread():
            t.join(timeout=1.5)
        self._thread = None
