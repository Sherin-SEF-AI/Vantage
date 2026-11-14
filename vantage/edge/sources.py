"""Where the watchdog's frames come from.

Live (the watchdog in the car): every source vantage.core.capture opens (a V4L2 device, a GStreamer pipeline, an
RTSP/HTTP URL, a video file, which plays at its own frame rate and starts again at its end), and ROS 2 image topics,
spec 'ros2:/topic' (sensor_msgs/Image or sensor_msgs/CompressedImage), through rclpy when the computer has ROS 2
(sourced before the watchdog starts). Capture times are the computer's monotonic clock when a frame arrived, as in
vantage.core.rig_health.

Recordings (dry runs): open_recording(path, tags) finds each camera's frames in
- a nuScenes folder (samples/ and sweeps/ with CAM_<tag> channels, or a CAM_<tag> folder itself), images named with
  their microsecond stamp (…__<µs>.jpg), as Vantage's sample datasets and CARLA drives write them;
- a folder of images per camera (a sub-folder named after the tag), stamped by a 16- or 19-digit number in the name
  (µs or ns) or, without one, by frame number at the configured rate;
- one video file per camera (named after its tag, any container OpenCV reads), stamped by frame number at the file's
  rate;
- a ROS 2 bag (vantage.core.rosbag: MCAP or SQLite, no ROS installation), image topics matched to tags by the topic's
  name (/ml/image_raw/compressed is ML), stamped with each message's header stamp.
Each reader yields (t seconds, RGB frame) lazily, so a long drive is never held in memory.
"""
from __future__ import annotations

import glob
import heapq
import os
import re
import threading
import time

import numpy as np

IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.pgm', '.ppm')
VIDEO_EXTS = ('.avi', '.mp4', '.mkv', '.mov', '.m4v', '.webm', '.mjpeg', '.mjpg')
_STAMP = re.compile(r'(?<!\d)(\d{16}|\d{19})(?!\d)')


def _tokens(s: str) -> list:
    return [t for t in re.split(r'[^a-z0-9]+', s.lower()) if t]


def _has_tag(name: str, tag: str) -> bool:
    a, b = _tokens(name), _tokens(tag)
    return bool(b) and any(a[k:k + len(b)] == b for k in range(len(a) - len(b) + 1))


def _read_rgb(path):
    import cv2
    img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class ImageSeries:
    def __init__(self, files, fps=10.0):
        stamps = []
        for f in files:
            m = _STAMP.search(os.path.basename(f))
            stamps.append(None if m is None else int(m.group(1)) * (1000 if len(m.group(1)) == 16 else 1))   # ns
        if files and all(s is not None for s in stamps):
            order = np.argsort(stamps, kind='stable')
            self.files = [files[i] for i in order]
            self.stamps_ns = [stamps[i] for i in order]                 # integers: no rounding at 1.7e18 ns
            self.times = [(x - self.stamps_ns[0]) * 1e-9 for x in self.stamps_ns]
            self.stamped = True
        else:
            self.files = sorted(files)
            self.stamps_ns = None
            self.times = [i / fps for i in range(len(self.files))]
            self.stamped = False

    def __len__(self):
        return len(self.files)

    def frames(self):
        for t, f in zip(self.times, self.files):
            img = _read_rgb(f)
            if img is not None:
                yield t, img


class VideoSeries:
    def __init__(self, path, fps=None):
        import cv2
        self.path = path
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError(f'cannot open {os.path.basename(path)}')
        f = cap.get(cv2.CAP_PROP_FPS)
        self.fps = float(fps or (f if 1 <= f <= 1000 else 30.0))
        self.n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        self.stamped = False

    def __len__(self):
        return self.n

    def frames(self):
        import cv2
        cap = cv2.VideoCapture(self.path)
        i = 0
        try:
            while True:
                ok, f = cap.read()
                if not ok or f is None:
                    break
                yield i / self.fps, cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                i += 1
        finally:
            cap.release()


class BagSeries:
    def __init__(self, path, topic):
        self.path, self.topic = path, topic
        from ..core.rosbag import BagReader
        with BagReader(path) as b:
            self.n = int(b.counts.get(topic, 0))
            self.start_ns = b.start_ns
        self.stamped = True

    def __len__(self):
        return self.n

    def frames(self):
        from ..core.rosbag import BagReader, image_array, stamp_ns
        with BagReader(self.path) as b:
            for _, t_ns, msg in b.messages([self.topic]):
                try:
                    img = image_array(msg, to_rgb=True)
                except Exception:                    # noqa: BLE001 - an undecodable frame is skipped
                    continue
                st = stamp_ns(msg['header']['stamp']) if isinstance(msg.get('header'), dict) else 0
                yield ((st or t_ns) - self.start_ns) * 1e-9, img


def _bag_topics(path):
    from ..core.rosbag import BagReader
    with BagReader(path) as b:
        return {t: ty for t, ty in b.topics.items() if ty.endswith(('/Image', '/CompressedImage'))}


def _is_bag(path) -> bool:
    if os.path.isfile(path):
        return path.lower().endswith(('.mcap', '.db3'))
    return os.path.isdir(path) and (os.path.isfile(os.path.join(path, 'metadata.yaml')) or
                                    bool(glob.glob(os.path.join(path, '*.mcap')) +
                                         glob.glob(os.path.join(path, '*.db3'))))


def open_recording(path: str, tags, fps=10.0) -> tuple[dict, str]:
    """({tag: series}, kind) of the cameras found in a recording; cameras not found are left out."""
    path = os.path.abspath(path)
    tags = list(tags)
    out = {}
    if _is_bag(path):
        topics = _bag_topics(path)
        for tag in tags:
            cand = [t for t in topics if _has_tag(t.split('/')[1] if t.count('/') > 1 else t, tag)]
            cand = cand or [t for t in topics if _has_tag(t, tag)]
            if cand:
                out[tag] = BagSeries(path, sorted(cand, key=len)[0])
        return out, 'ROS 2 bag'
    if os.path.isfile(path) and path.lower().endswith(VIDEO_EXTS):
        if len(tags) == 1:
            return {tags[0]: VideoSeries(path)}, 'video'
        stem = os.path.splitext(os.path.basename(path))[0]
        tag = next((t for t in tags if _has_tag(stem, t)), None)
        return ({tag: VideoSeries(path)} if tag else {}), 'video'
    if not os.path.isdir(path):
        raise ValueError(f'{path}: not a recording Vantage can read (a folder, a video file or a ROS 2 bag)')
    nus = [os.path.join(r, d) for r in (path, os.path.join(path, 'nuscenes')) for d in ('samples', 'sweeps')
           if os.path.isdir(os.path.join(r, d))]
    kind = 'nuScenes folder' if nus else 'image folders'
    entries = os.listdir(path)
    for tag in tags:
        files = []
        for base in nus:
            for d in os.listdir(base):
                if d.upper() in (f'CAM_{tag.upper()}', tag.upper()):
                    files += [f for f in glob.glob(os.path.join(base, d, '*')) if f.lower().endswith(IMAGE_EXTS)]
        if not files:
            for d in entries:
                full = os.path.join(path, d)
                if os.path.isdir(full) and d.upper() in (tag.upper(), f'CAM_{tag.upper()}'):
                    files = [f for f in glob.glob(os.path.join(full, '*')) if f.lower().endswith(IMAGE_EXTS)]
                    break
        if files:
            out[tag] = ImageSeries(files, fps)
            continue
        vids = [os.path.join(path, f) for f in entries if f.lower().endswith(VIDEO_EXTS)
                and _has_tag(os.path.splitext(f)[0], tag)]
        if vids:
            out[tag] = VideoSeries(sorted(vids, key=len)[0])
            kind = 'video files'
    if not out and len(tags) == 1:
        imgs = [os.path.join(path, f) for f in entries if f.lower().endswith(IMAGE_EXTS)]
        if imgs:
            out[tags[0]] = ImageSeries(imgs, fps)
    stamped = [s for s in out.values() if isinstance(s, ImageSeries) and s.stamped and s.stamps_ns]
    if stamped:                                     # one clock for every camera: the first stamp of any is 0
        t0 = min(s.stamps_ns[0] for s in stamped)
        for s in stamped:
            s.times = [(x - t0) * 1e-9 for x in s.stamps_ns]
    return out, kind


def merged(series: dict, transform=None):
    """(t, tag, frame) of every camera in time order. transform(tag, t, frame) -> list of (t, frame) (a fault injection;
    [] drops the frame, several items repeat it; times must keep increasing per camera)."""
    order = {tag: i for i, tag in enumerate(series)}

    def gen(tag, s):
        for t, f in s.frames():
            for t2, f2 in (transform(tag, t, f) if transform else ((t, f),)):
                yield t2, tag, f2
    return heapq.merge(*[gen(tag, s) for tag, s in series.items()], key=lambda x: (x[0], order[x[1]]))


# ---------------------------------------------------------------------------------------------------- ROS 2 live
class Ros2Source:
    """A live ROS 2 image topic through rclpy (ROS 2 sourced in the environment). The same interface as
    vantage.core.capture.LiveSource: start, stop, running, error, latest, on_frame(frame, t)."""

    def __init__(self, spec: str, on_frame=None):
        self.spec = spec
        self.topic = spec.split(':', 1)[1] if ':' in spec else spec
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
        if self.running:
            return
        self._stop = threading.Event()
        self.error = ''
        self._thread = threading.Thread(target=self._run, args=(self._stop,), name=f'ros2:{self.topic}', daemon=True)
        self._thread.start()

    def _msg(self, msg):
        from ..core.rosbag import image_array
        if hasattr(msg, 'format') and not hasattr(msg, 'encoding'):
            d = {'format': msg.format, 'data': bytes(msg.data)}
        else:
            d = {'encoding': msg.encoding, 'height': msg.height, 'width': msg.width, 'step': msg.step,
                 'is_bigendian': msg.is_bigendian, 'data': bytes(msg.data)}
        img = image_array(d, to_rgb=True)
        t = time.monotonic()
        with self._lock:
            self._frame, self.size, self.stamp = img, (img.shape[1], img.shape[0]), t
            self.frames += 1
        cb = self.on_frame
        if cb is not None:
            try:
                cb(img, t)
            except Exception:                         # noqa: BLE001 - a statistics hook never stops the stream
                pass

    def _run(self, stop):
        try:
            import rclpy
            from rclpy.qos import qos_profile_sensor_data
            from sensor_msgs.msg import CompressedImage, Image
        except ImportError as e:
            self.error = f'ROS 2 (rclpy) is not available: {e}; source ROS 2 before starting the watchdog'
            return
        try:
            if not rclpy.ok():
                rclpy.init()
            node = rclpy.create_node('vantage_edge_' + re.sub(r'[^a-z0-9_]', '_', self.topic.lower()).strip('_'))
            typ = CompressedImage if self.topic.rstrip('/').endswith('compressed') else Image
            node.create_subscription(typ, self.topic, self._msg, qos_profile_sensor_data)
            while not stop.is_set():
                rclpy.spin_once(node, timeout_sec=0.1)
            node.destroy_node()
        except Exception as e:                        # noqa: BLE001
            self.error = f'ROS 2 topic {self.topic}: {e}'

    def latest(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def stop(self, wait=True):
        self._stop.set()
        t = self._thread
        if wait and t is not None and t is not threading.current_thread():
            t.join(timeout=1.5)
        self._thread = None
