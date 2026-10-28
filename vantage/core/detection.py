"""An off-the-shelf 2D object detector for the detection-quality study (core.detection_eval): YOLOX, a COCO detector
published under the Apache License 2.0, run through OpenCV's DNN module (cv2.dnn), so Vantage needs no new dependency.

The model
---------
YOLOX (Ge, Liu, Wang, Li and Sun, "YOLOX: Exceeding YOLO Series in 2021", arXiv:2107.08430; Megvii,
github.com/Megvii-BaseDetection/YOLOX, Apache-2.0) in the ONNX exports of its release 0.1.1rc0 (2021-08-18). The model
files are never part of Vantage: the app downloads one on request into the user's data folder (models_dir()) from
its pinned URL and keeps it only when its size and SHA-256 match the values below. COCO val2017 AP (0.5:0.95) is the
publisher's figure from YOLOX's README, not measured here.

The ONNX graph takes a BGR image as float32 0–255 (no normalisation), NCHW, any size that is a multiple of 32 (checked
here with OpenCV 4.13 and 5.0), and gives per anchor point x, y, w, h (not decoded), the objectness and the 80 COCO
class probabilities (both after a sigmoid). As YOLOX's own demo (demo/ONNXRuntime, yolox/utils/demo_utils.py):
* preprocessing: the image scaled by r = min(W_in / w, H_in / h) (bilinear), placed top-left on a canvas of grey 114;
* decoding at strides 8, 16, 32 (grid order row-major per stride): centre = (offset + cell) · stride, size =
  exp(log size) · stride;
* score = objectness · class probability, kept above keep_score, non-maximum suppression per class (IoU nms_iou,
  cv2.dnn.NMSBoxesBatched), boxes divided by r and clipped to the image; at most max_dets per image, the highest
  scores.
Input size: 'auto' runs each image at its own resolution (its sides rounded up to a multiple of 32; images larger than
max_input_px on their longer side are scaled down), so the detector sees what the camera recorded; a fixed size runs
the model's usual square letterbox (640 for S and M, 416 for Tiny and Nano).

Classes
-------
COCO's classes the study uses map to Vantage's: person → pedestrian, bicycle → cyclist, motorcycle → motorcyclist, car,
truck, bus, traffic light, stop sign. The ground truth's classes (core.groundtruth, CARLA's semantic tags) map the same
way: pedestrian; bicycle and rider → cyclist; motorcycle → motorcyclist; car, truck, bus, traffic_light; a
traffic_sign is a stop sign only when its object says so (its CARLA type), otherwise it is an ignore region for the
stop-sign class (COCO knows no other sign). A COCO detector boxes a cyclist's person and bicycle separately while the
ground truth boxes the actor (bicycle and rider): with fuse_riders a person detection that overlaps a bicycle or
motorcycle detection (at least FUSE_IOA of the person's box inside the bike's, its centre over the bike's width)
joins it: one cyclist (motorcyclist) box, their union, with the bike's score. 'vehicle' groups car, truck and bus,
evaluated without telling them apart (a pickup boxed as a truck still counts).

Speed (measured here, 2026-10-27, 24 CPU threads, cv2.dnn's own CPU backend; the forward pass only): YOLOX-S at
640 × 640 93 ms with OpenCV 4.13, 40 ms with OpenCV 5.0; at 960 × 960 227 / 102 ms. YOLOX-Tiny at 416 × 416 32 /
11 ms; YOLOX-M at 640 × 640 184 ms (4.13). The OpenCL target is offered when OpenCV has OpenCL; PyPI's OpenCV has no
CUDA backend.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sys
import threading
import time
from dataclasses import asdict, dataclass

import numpy as np

SCHEMA = 'vantage.detections/1'
ENV_DIR = 'VANTAGE_MODEL_DIR'          # overrides the models folder (tests, shared installs)

COCO_CLASSES = (
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
    'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
    'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
    'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
    'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')
# the 80 classes' ids in the COCO annotation files (1–90 with gaps), for results in COCO's detection format
COCO_IDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33,
            34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
            62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90)

# Vantage's detection classes (the study's rows), in order
CLASSES = ('pedestrian', 'cyclist', 'motorcyclist', 'car', 'truck', 'bus', 'traffic_light', 'stop_sign')
CLASS_LABEL = {'pedestrian': 'Pedestrian', 'cyclist': 'Cyclist', 'motorcyclist': 'Motorcyclist', 'car': 'Car',
               'truck': 'Truck', 'bus': 'Bus', 'traffic_light': 'Traffic light', 'stop_sign': 'Stop sign',
               'vehicle': 'Vehicle (car, truck, bus)'}
COCO_TO_CLASS = {'person': 'pedestrian', 'bicycle': 'cyclist', 'motorcycle': 'motorcyclist', 'car': 'car',
                 'truck': 'truck', 'bus': 'bus', 'traffic light': 'traffic_light', 'stop sign': 'stop_sign'}
# ground-truth class names (core.groundtruth's class table, CARLA 0.9.16's tags) -> Vantage's
GT_TO_CLASS = {'pedestrian': 'pedestrian', 'bicycle': 'cyclist', 'rider': 'cyclist', 'motorcycle': 'motorcyclist',
               'car': 'car', 'truck': 'truck', 'bus': 'bus', 'traffic_light': 'traffic_light',
               'traffic_sign': 'stop_sign'}
GROUPS = {'vehicle': ('car', 'truck', 'bus')}     # evaluated without telling the members apart
EVAL_CLASSES = CLASSES + tuple(GROUPS)
RIDER_OF = {'cyclist': 'bicycle', 'motorcyclist': 'motorcycle'}
FUSE_IOA = 0.25                         # share of a person's box inside a bike's box that makes it the rider

_COCO_INDEX = {n: i for i, n in enumerate(COCO_CLASSES)}
MAPPED_COCO = np.array([_COCO_INDEX[n] for n in COCO_TO_CLASS], int)                 # COCO index of each mapped class
MAPPED_CLASS = np.array([CLASSES.index(COCO_TO_CLASS[n]) for n in COCO_TO_CLASS], int)  # its Vantage class index
CLASS_OF_COCO = np.full(len(COCO_CLASSES), -1, int)            # COCO index -> Vantage class index (-1: unused)
CLASS_OF_COCO[MAPPED_COCO] = MAPPED_CLASS


@dataclass(frozen=True)
class ModelSpec:
    id: str
    name: str
    file: str
    url: str
    sha256: str
    size: int                    # bytes
    input_px: int                # the square size it was trained and published at
    coco_ap: float               # COCO val2017 AP 0.5:0.95, the publisher's figure
    params_m: float
    gflops: float
    licence: str = 'Apache-2.0'
    licence_url: str = 'https://github.com/Megvii-BaseDetection/YOLOX/blob/main/LICENSE'
    source: str = 'https://github.com/Megvii-BaseDetection/YOLOX/releases/tag/0.1.1rc0'
    family: str = 'yolox'
    note: str = ''

    def to_dict(self):
        return asdict(self)


_REL = 'https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/'
MODELS = {
    'yolox_s': ModelSpec('yolox_s', 'YOLOX-S', 'yolox_s.onnx', _REL + 'yolox_s.onnx',
                         'c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063', 35858002, 640, 40.5, 9.0,
                         26.8, note='the default: a good balance of accuracy and speed on a CPU'),
    'yolox_tiny': ModelSpec('yolox_tiny', 'YOLOX-Tiny', 'yolox_tiny.onnx', _REL + 'yolox_tiny.onnx',
                            '427cc366d34e27ff7a03e2899b5e3671425c262ea2291f88bb942bc1cc70b0f7', 20219662, 416, 32.8,
                            5.06, 6.45, note='about 3× faster, less accurate'),
    'yolox_nano': ModelSpec('yolox_nano', 'YOLOX-Nano', 'yolox_nano.onnx', _REL + 'yolox_nano.onnx',
                            'c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d', 3659407, 416, 25.8,
                            0.91, 1.08, note='an embedded-class model'),
    'yolox_m': ModelSpec('yolox_m', 'YOLOX-M', 'yolox_m.onnx', _REL + 'yolox_m.onnx',
                         '21ff6cfdeb53b013bac2249599e55f00bff3cfdfdab37ed7a4620818c1d15b3f', 101259744, 640, 46.9,
                         25.3, 73.8, note='more accurate, about 2× slower than S'),
}
DEFAULT_MODEL = 'yolox_s'
LICENCE_TEXT = ('YOLOX by Megvii, Apache License 2.0 (github.com/Megvii-BaseDetection/YOLOX). The model file is '
                'downloaded from YOLOX\'s GitHub release on request, checked against its SHA-256 and kept in your '
                'data folder; it is not part of Vantage.')
BACKENDS = ('cpu', 'opencl')
LIMITS = {'input_px': (0, 1920), 'max_input_px': (320, 3840), 'keep_score': (0.001, 0.99), 'nms_iou': (0.1, 0.95),
          'max_dets': (1, 1000), 'threads': (0, 64)}
INTS = ('input_px', 'max_input_px', 'max_dets', 'threads')


# =========================================================================== settings
@dataclass
class DetectorSettings:
    model: str = DEFAULT_MODEL
    input_px: int = 0             # 0 = auto: each image at its own size (sides up to a multiple of 32); else square
    max_input_px: int = 1280      # auto: a larger image is scaled down to this on its longer side
    keep_score: float = 0.05      # detections kept (for the precision–recall curves); the operating point is higher
    nms_iou: float = 0.45         # YOLOX's own
    max_dets: int = 100           # per image, as COCO's evaluation
    fuse_riders: bool = True
    backend: str = 'cpu'
    threads: int = 0              # OpenCV's threads (0 = its default)

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d) -> 'DetectorSettings':
        s = DetectorSettings()
        d = d if isinstance(d, dict) else {}
        if d.get('model') in MODELS:
            s.model = d['model']
        if d.get('backend') in BACKENDS:
            s.backend = d['backend']
        v = d.get('fuse_riders')
        if isinstance(v, (bool, int)) and not isinstance(v, float):
            s.fuse_riders = bool(v)
        elif isinstance(v, str):
            s.fuse_riders = v.strip().lower() in ('1', 'true', 'yes', 'on')
        for k, (lo, hi) in LIMITS.items():
            if k not in d or isinstance(d[k], bool):
                continue
            try:
                x = float(d[k])
            except (TypeError, ValueError):
                continue
            if not math.isfinite(x):
                continue
            x = min(max(x, lo), hi)
            setattr(s, k, int(round(x)) if k in INTS else x)
        if s.input_px:
            s.input_px = max(32, int(round(s.input_px / 32.0)) * 32)
        return s

    def key(self, sha256='') -> str:
        """What the detections depend on (the cache key's settings part)."""
        d = self.to_dict()
        d.pop('threads', None)
        d.pop('backend', None)            # CPU and OpenCL agree to float tolerance; a cache serves both
        return json.dumps([d, sha256], sort_keys=True)


# =========================================================================== where models live
def data_dir() -> str:
    """Vantage's per-user data folder: %LOCALAPPDATA%\\Vantage (Windows), ~/Library/Application Support/Vantage
    (macOS), $XDG_DATA_HOME/Vantage, by default ~/.local/share/Vantage (Linux)."""
    if sys.platform.startswith('win'):
        root = os.environ.get('LOCALAPPDATA') or os.environ.get('APPDATA') or os.path.expanduser('~')
    elif sys.platform == 'darwin':
        root = os.path.expanduser('~/Library/Application Support')
    else:
        root = os.environ.get('XDG_DATA_HOME') or os.path.join(os.path.expanduser('~'), '.local', 'share')
    return os.path.join(root, 'Vantage')


def models_dir() -> str:
    """Where downloaded models are kept: $VANTAGE_MODEL_DIR, else <data folder>/models."""
    return os.path.abspath(os.path.expanduser(os.environ.get(ENV_DIR) or os.path.join(data_dir(), 'models')))


def spec_of(model) -> ModelSpec:
    if isinstance(model, ModelSpec):
        return model
    if model not in MODELS:
        raise KeyError(f'Unknown detector model {model!r}; known: {", ".join(MODELS)}')
    return MODELS[model]


def model_path(model, folder=None) -> str:
    return os.path.join(folder or models_dir(), spec_of(model).file)


_SHA_CACHE: dict = {}
_SHA_LOCK = threading.Lock()


def file_sha256(path, chunk=1 << 20) -> str:
    """SHA-256 of a file (hex), cached on its path, size and modification time."""
    st = os.stat(path)
    key = (os.path.abspath(path), st.st_size, st.st_mtime_ns)
    with _SHA_LOCK:
        if key in _SHA_CACHE:
            return _SHA_CACHE[key]
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(chunk), b''):
            h.update(block)
    out = h.hexdigest()
    with _SHA_LOCK:
        _SHA_CACHE[key] = out
    return out


def _mb(n) -> str:
    return f'{n / 1e6:.1f} MB'


def status(model=DEFAULT_MODEL, folder=None) -> dict:
    """{'id', 'name', 'path', 'present', 'verified', 'size', 'expected_size', 'sha256', 'licence', 'url', 'text'}:
    whether the model is in the models folder and its checksum matches (a wrong file is never used)."""
    sp = spec_of(model)
    path = model_path(sp, folder)
    out = {'id': sp.id, 'name': sp.name, 'path': path, 'present': False, 'verified': False, 'size': 0,
           'expected_size': sp.size, 'sha256': sp.sha256, 'licence': sp.licence, 'licence_url': sp.licence_url,
           'url': sp.url, 'source': sp.source}
    if not os.path.isfile(path):
        out['text'] = f'{sp.name} is not downloaded ({_mb(sp.size)}, {sp.licence}).'
        return out
    out['present'] = True
    out['size'] = os.path.getsize(path)
    if out['size'] != sp.size:
        out['text'] = (f'{sp.name}: the file in {os.path.dirname(path)} has {out["size"]:,} bytes, not '
                       f'{sp.size:,}: download it again.')
        return out
    try:
        ok = file_sha256(path) == sp.sha256
    except OSError as e:
        out['text'] = f'{sp.name}: {e}'
        return out
    out['verified'] = ok
    out['text'] = (f'{sp.name} ready ({_mb(sp.size)}, SHA-256 checked, {sp.licence}).' if ok else
                   f'{sp.name}: the file\'s SHA-256 does not match the published model: download it again.')
    return out


def _ssl_context():
    import ssl
    ctx = ssl.create_default_context()
    try:                            # bundles do not always find the system certificates; certifi does
        import certifi
        ctx.load_verify_locations(certifi.where())
    except Exception:               # noqa: BLE001
        pass
    return ctx


def download(model=DEFAULT_MODEL, progress=None, cancel=None, folder=None, timeout=60.0) -> str | None:
    """Download a model from its pinned URL into the models folder; returns its path (None when cancelled). The file
    is written as <name>.part and only renamed into place when its size and SHA-256 match the published ones;
    anything else raises (ValueError for a mismatch, OSError for the network) and leaves nothing behind."""
    import urllib.request
    from .. import __version__
    sp = spec_of(model)
    folder = folder or models_dir()
    os.makedirs(folder, exist_ok=True)
    final = os.path.join(folder, sp.file)
    part = final + '.part'
    stop = cancel or (lambda: False)
    req = urllib.request.Request(sp.url, headers={'User-Agent': f'Vantage/{__version__}'})
    h = hashlib.sha256()
    n = 0
    t0 = time.perf_counter()
    try:
        kw = {'context': _ssl_context()} if sp.url.lower().startswith('https:') else {}
        with urllib.request.urlopen(req, timeout=timeout, **kw) as r, open(part, 'wb') as fh:
            total = int(r.headers.get('Content-Length') or 0) or sp.size
            while True:
                if stop():
                    raise _Cancelled()
                block = r.read(1 << 20)
                if not block:
                    break
                fh.write(block)
                h.update(block)
                n += len(block)
                if n > sp.size * 1.01 + (1 << 20):
                    raise ValueError(f'{sp.name}: the download is larger than the published {sp.size:,} bytes.')
                if progress is not None:
                    dt = max(time.perf_counter() - t0, 1e-3)
                    progress(min(n / max(total, 1), 1.0), f'{_mb(n)} of {_mb(total)} · {n / dt / 1e6:.1f} MB/s')
        if n != sp.size:
            raise ValueError(f'{sp.name}: downloaded {n:,} bytes, the published model has {sp.size:,}.')
        if h.hexdigest() != sp.sha256:
            raise ValueError(f'{sp.name}: the download\'s SHA-256 ({h.hexdigest()[:16]}…) is not the published '
                             f'model\'s ({sp.sha256[:16]}…): not kept.')
        os.replace(part, final)
    except _Cancelled:
        _rm(part)
        return None
    except BaseException:
        _rm(part)
        raise
    with _SHA_LOCK:
        st = os.stat(final)
        _SHA_CACHE[(os.path.abspath(final), st.st_size, st.st_mtime_ns)] = sp.sha256
    return final


class _Cancelled(Exception):
    pass


def _rm(path):
    try:
        os.remove(path)
    except OSError:
        pass


def import_file(path, folder=None) -> str:
    """Copy a model file someone downloaded by hand (an offline computer) into the models folder; it must be one of
    the published models (by SHA-256). Returns the model's id; ValueError otherwise."""
    sha = file_sha256(path)
    sp = next((m for m in MODELS.values() if m.sha256 == sha), None)
    if sp is None:
        raise ValueError(f'{os.path.basename(path)} is none of the published models (its SHA-256 matches none of '
                         + ', '.join(m.file for m in MODELS.values()) + ').')
    folder = folder or models_dir()
    os.makedirs(folder, exist_ok=True)
    dst = os.path.join(folder, sp.file)
    if os.path.abspath(dst) != os.path.abspath(path):
        shutil.copyfile(path, dst + '.part')
        os.replace(dst + '.part', dst)
    return sp.id


def remove(model, folder=None) -> bool:
    p = model_path(model, folder)
    if os.path.isfile(p):
        os.remove(p)
        return True
    return False


def opencl_available() -> bool:
    try:
        import cv2
        return bool(cv2.ocl.haveOpenCL())
    except Exception:               # noqa: BLE001
        return False


# =========================================================================== pre- and post-processing (numpy only)
def input_size(w: int, h: int, st: DetectorSettings) -> tuple[int, int]:
    """(W, H) of the network input for a w × h image."""
    if st.input_px:
        return st.input_px, st.input_px
    s = min(1.0, st.max_input_px / max(w, h, 1))
    return max(32, int(math.ceil(w * s / 32.0)) * 32), max(32, int(math.ceil(h * s / 32.0)) * 32)


def letterbox(img, W: int, H: int):
    """YOLOX's preprocessing: (blob 1×3×H×W float32 BGR 0–255, r) — the image scaled by r = min(W/w, H/h) placed
    top-left on grey 114."""
    import cv2
    h, w = img.shape[:2]
    r = min(W / w, H / h)
    nw, nh = int(w * r), int(h * r)
    canvas = np.full((H, W, 3), 114, np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    rs = img if (nw, nh) == (w, h) else cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas[:nh, :nw] = rs[:, :, :3]
    return np.ascontiguousarray(canvas.transpose(2, 0, 1)[None], dtype=np.float32), r


_GRIDS: dict = {}


def _grid(W: int, H: int):
    key = (W, H)
    if key not in _GRIDS:
        g, s = [], []
        for stride in (8, 16, 32):
            hs, ws = H // stride, W // stride
            xv, yv = np.meshgrid(np.arange(ws), np.arange(hs))
            g.append(np.stack((xv, yv), 2).reshape(-1, 2))
            s.append(np.full((hs * ws, 1), stride))
        _GRIDS[key] = (np.concatenate(g).astype(np.float32), np.concatenate(s).astype(np.float32))
    return _GRIDS[key]


def decode(raw, W: int, H: int):
    """YOLOX's raw output (1 × N × 85 or N × 85) at input W × H -> (boxes N × 4 x0 y0 x1 y1 in input px, objectness N,
    class probabilities N × 80)."""
    p = np.asarray(raw, np.float32).reshape(-1, 85)
    grids, strides = _grid(W, H)
    if len(grids) != len(p):
        raise ValueError(f'The model gave {len(p)} predictions for a {W} × {H} input, expected {len(grids)}.')
    xy = (p[:, :2] + grids) * strides
    wh = np.exp(np.minimum(p[:, 2:4], 30.0)) * strides
    boxes = np.concatenate([xy - wh / 2, xy + wh / 2], 1)
    return boxes, p[:, 4], p[:, 5:]


def empty() -> dict:
    return {'boxes': np.zeros((0, 4), np.float32), 'scores': np.zeros(0, np.float32), 'coco': np.zeros(0, np.int16),
            'cls': np.zeros(0, np.int8)}


def _nms_batched(boxes, scores, cls, thr):
    """Indices kept by non-maximum suppression within each class (cv2.dnn.NMSBoxesBatched)."""
    import cv2
    if not len(boxes):
        return np.zeros(0, int)
    xywh = np.column_stack([boxes[:, :2], boxes[:, 2:] - boxes[:, :2]]).astype(np.float64)
    keep = cv2.dnn.NMSBoxesBatched(xywh.tolist(), scores.astype(float).tolist(), cls.astype(int).tolist(), 0.0,
                                   float(thr))
    return np.asarray(keep, int).reshape(-1)


def postprocess(boxes, obj, probs, r: float, w: int, h: int, st: DetectorSettings) -> dict:
    """Decoded predictions -> detections of the mapped classes in image px: score = objectness × class probability
    above keep_score, NMS per class, boxes / r clipped to the image, the max_dets best, riders fused."""
    sc = obj[:, None] * probs[:, MAPPED_COCO]                     # N × mapped classes
    ii, jj = np.nonzero(sc > st.keep_score)
    if not len(ii):
        return empty()
    b = boxes[ii] / r
    s = sc[ii, jj]
    coco = MAPPED_COCO[jj]
    keep = _nms_batched(b, s, coco, st.nms_iou)
    b, s, coco = b[keep], s[keep], coco[keep]
    b[:, [0, 2]] = np.clip(b[:, [0, 2]], 0, w)
    b[:, [1, 3]] = np.clip(b[:, [1, 3]], 0, h)
    ok = (b[:, 2] - b[:, 0] > 0.5) & (b[:, 3] - b[:, 1] > 0.5)
    b, s, coco = b[ok], s[ok], coco[ok]
    order = np.argsort(-s, kind='mergesort')[:st.max_dets]
    cls = CLASS_OF_COCO[coco[order]].astype(np.int8)
    out = {'boxes': b[order].astype(np.float32), 'scores': s[order].astype(np.float32),
           'coco': coco[order].astype(np.int16), 'cls': cls}
    return fuse_riders(out) if st.fuse_riders else out


def fuse_riders(d: dict) -> dict:
    """A person detection that rides a bicycle or motorcycle detection joins it: the union box as one cyclist
    (motorcyclist) with the bike's score. A bike takes at most one person (the most overlapped), bikes in score order;
    a person that rides nothing stays a pedestrian."""
    cls = d['cls']
    pi = CLASSES.index('pedestrian')
    bikes = np.nonzero(np.isin(cls, (CLASSES.index('cyclist'), CLASSES.index('motorcyclist'))))[0]
    persons = np.nonzero(cls == pi)[0]
    if not len(bikes) or not len(persons):
        return d
    b = d['boxes'].astype(np.float64)
    used = set()
    out_boxes = b.copy()
    drop = []
    for k in bikes[np.argsort(-d['scores'][bikes], kind='mergesort')]:
        bx = b[k]
        best, best_ioa = -1, FUSE_IOA
        for j in persons:
            if j in used:
                continue
            p = b[j]
            iw = min(p[2], bx[2]) - max(p[0], bx[0])
            ih = min(p[3], bx[3]) - max(p[1], bx[1])
            if iw <= 0 or ih <= 0:
                continue
            ioa = iw * ih / max((p[2] - p[0]) * (p[3] - p[1]), 1e-9)
            cx = (p[0] + p[2]) / 2
            if ioa >= best_ioa and bx[0] <= cx <= bx[2]:
                best, best_ioa = j, ioa
        if best >= 0:
            used.add(best)
            drop.append(best)
            p = b[best]
            out_boxes[k] = [min(p[0], bx[0]), min(p[1], bx[1]), max(p[2], bx[2]), max(p[3], bx[3])]
    if not drop:
        return d
    keep = np.setdiff1d(np.arange(len(cls)), drop)
    return {'boxes': out_boxes[keep].astype(np.float32), 'scores': d['scores'][keep], 'coco': d['coco'][keep],
            'cls': d['cls'][keep]}


# =========================================================================== the detector
class Detector:
    """One loaded model. detect(img) -> {'boxes' n × 4 (x0 y0 x1 y1, image px), 'scores', 'coco' (COCO class index),
    'cls' (index into CLASSES), 'ms' (forward + decoding)}. Thread-safe (a lock round the network)."""

    def __init__(self, path, spec: ModelSpec | None = None, settings: DetectorSettings | None = None):
        import cv2
        self.spec = spec or MODELS[DEFAULT_MODEL]
        self.settings = settings or DetectorSettings()
        self.path = path
        self.net = cv2.dnn.readNetFromONNX(path)
        self.backend = 'cpu'
        if self.settings.backend == 'opencl' and opencl_available():
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
            self.backend = 'opencl'
        else:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        if self.settings.threads:
            cv2.setNumThreads(int(self.settings.threads))
        self.lock = threading.Lock()
        self.opencv = cv2.__version__

    def forward(self, blob):
        with self.lock:
            self.net.setInput(blob)
            return self.net.forward()

    def detect(self, img) -> dict:
        t0 = time.perf_counter()
        h, w = img.shape[:2]
        W, H = input_size(w, h, self.settings)
        blob, r = letterbox(img, W, H)
        raw = self.forward(blob)
        boxes, obj, probs = decode(raw, W, H)
        out = postprocess(boxes, obj, probs, r, w, h, self.settings)
        out['ms'] = (time.perf_counter() - t0) * 1000.0
        out['input'] = (W, H)
        return out

    __call__ = detect

    def describe(self) -> dict:
        return {'model': self.spec.id, 'name': self.spec.name, 'sha256': self.spec.sha256, 'licence': self.spec.licence,
                'opencv': self.opencv, 'backend': self.backend, 'settings': self.settings.to_dict()}


def load(settings: DetectorSettings | None = None, folder=None) -> Detector:
    """The settings' model, loaded (FileNotFoundError when it is not downloaded, ValueError when its checksum does not
    match: a modified or truncated file is never used)."""
    st = settings or DetectorSettings()
    sp = spec_of(st.model)
    s = status(sp, folder)
    if not s['present']:
        raise FileNotFoundError(f'{sp.name} is not downloaded: Simulate ▸ Detection quality ▸ Download '
                                f'({_mb(sp.size)}).')
    if not s['verified']:
        raise ValueError(s['text'])
    return Detector(s['path'], sp, st)


def to_coco_results(dets: dict, image_id) -> list:
    """Detections in COCO's results format ([{'image_id', 'category_id', 'bbox' [x, y, w, h], 'score'}]), COCO's own
    category ids (riders fused into a bike keep the bike's id)."""
    out = []
    for b, s, c in zip(dets['boxes'], dets['scores'], dets['coco']):
        out.append({'image_id': image_id, 'category_id': COCO_IDS[int(c)],
                    'bbox': [round(float(b[0]), 2), round(float(b[1]), 2), round(float(b[2] - b[0]), 2),
                             round(float(b[3] - b[1]), 2)], 'score': round(float(s), 5)})
    return out
