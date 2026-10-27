"""As-built check: does each real camera see what the plan says it sees, and what do the plan's coverage figures become
with the rig as it was really built?

Input: a recording of a short drive from the real rig: a ROS 2 bag (core.rosbag: a rosbag2 folder, an .mcap or a .db3
file), a nuScenes-format folder (as core.drivelog reads it) or a plain folder of images per camera (one subfolder per
camera, or file names starting with the camera's tag: F_000123.jpg). Image streams are mapped to the variant's cameras
as the drive-log check maps them (core.drivelog: the tag in the topic, folder or channel name); the mapping can be
edited and the edits are saved in the settings per recording path.

Reading. One streaming pass reads the recording and never holds it in memory: at `frames` instants spread over the
drive, one image of every camera stream is kept (grey, at most ANALYSIS_W wide, lossless), but only when the vehicle
has moved since the stream's last kept frame: by min_motion_m (or 3° of heading) from odometry (nav_msgs/Odometry,
nuScenes ego poses) or GNSS fixes; without them, or with only an IMU (turning, not travel), by the frames themselves
(the mean change of a small grey copy against the last kept frame, min_change grey levels). Images of any size are
mapped to the module's sensor through their scale, as the drive-log check does.

Per mapped camera:
1. Static (ego) mask. Per pixel, over the kept frames (each first matched to the median frame by a gain and an offset:
   auto exposure), the median M and the 90th percentile D of |frame - M|. A pixel is static when D stays under
   static_sigma x the noise (from the stillest tenth of the image; never under COMPRESSION_GREY grey levels), or under
   RING_SHARE of the largest D within 4 px (the vehicle's first pixels next to a moving edge ring with a share of that
   edge's change through compression and blur). Static is not yet the vehicle: plain road, sky and anything else
   without texture stay the same too. Static areas are split along the static edges of M (Canny), and each piece is
   attributed to the vehicle when most of it lies within the search band of the planned silhouette (the plan's body
   grown by search_deg). A piece mostly outside the band that also reaches deep inside the planned body is the vehicle
   and scenery run together with no static edge between them: deep inside is the vehicle, the band there undecided
   (neither counted for the mask nor used for the fit). A piece that does not reach in is scenery parted from the
   vehicle by a static edge; that edge is taken as the vehicle's outline when the scenery shows more temporal
   variation next to it than the vehicle does (road noise and mottle, which paint does not have), else the band stays
   undecided (it may be a line on the body). Small vehicle pieces are dropped, small holes filled, small specks of the
   world inside the vehicle or the band made undecided. A fisheye's dark area outside the image circle is found
   separately (flat, still and dark, grown from the image corners); its edge is located along rays from a rough
   circle where the median image leaves the exterior's level by half the step to what lies inside, and fitted with a
   circle (RANSAC, then least squares): centre and radius.
2. Simulated mask: the same camera from the plan, its pixels' rays looked up in the vehicle's occlusion cube
   (engine.cube, as core.synth renders it: the body, the rig parts and whatever else the engine puts in the cube) through
   the lens model (core.lens.resolve with the camera's unit calibration); the predicted image circle from the lens's
   max_theta.
3. Comparison: mask IoU over the decided pixels, the mean and 95th percentile distance between the two outlines
   (midpoints between neighbouring pixels on either side, each real one moved onto the median image's edge when an
   isolated edge lies within 1.5 px), in pixels and degrees, the image-circle centre and radius error. Then the pose
   correction: the rotation (yaw, pitch, roll added to the plan's; optionally a translation) that best lays the
   simulated outline onto the real one, with a uniform outline offset fitted beside it (blur and the threshold grow or
   shrink a mask all round; fitted, it cannot pass for a rotation): a two-stage grid on the distance transform of the
   real outline, then iteratively reweighted Gauss-Newton on point-to-line distances (each simulated outline point
   against the nearest real one along its normal; Cauchy weights, shrinking from 6 to 1 px), started from the grid's
   best and from the plan, the lower robust cost winning; a pose beyond the search range is no answer. The standard
   errors are a cluster-robust sandwich (outline points in CELL_PX cells as clusters) combined with a floor of SIGMA_Q
   px per cell (the pixel grid, the cube's texels, the mesh's facets); an axis whose floor alone exceeds OBS_DEG is not
   constrained by the outline in view and stays as planned. An axis is off when it is precise (standard error within
   two thirds of the tolerance) and beyond the tolerance by more than three standard errors; one beyond the tolerance
   but not precise makes the camera uncertain. A camera whose outline is too short, mostly undecided, or pins no axis
   is 'not checkable': no number is made up.
4. Optionally the drive-log check's rotation of each camera against the lidar (core.drivelog, run on the same recording
   when it has a lidar topic) as a second, independent estimate; both are shown and whether they agree within their
   errors. For a camera its outline cannot check, a confident drive-log estimate stands in (marked so).
5. As built: a copy of the variant with every confidently measured correction applied (cameras off by more than the
   tolerance on a constrained axis, the correction within the tolerance's precision), the coverage analysis on plan and
   as built, and their differences: blind ground, the share seen by two cameras, the shares at the minimum and target
   density, each camera's visible ground; the coverage requirements of the scorecard (core.requirements) on both.
   Verdict per camera and overall: the plan's coverage holds, holds with corrections or does not hold (or not
   checkable).

Frames: the vehicle frame (ISO 8855, rear axle); angles in degrees, intrinsic Z-Y-X, pitch positive down.
"""
from __future__ import annotations

import copy
import csv
import datetime as _dt
import io
import json
import math
import os
import re
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from . import drivelog as DL
from .mathutil import BODY_TO_OPTICAL, euler_zyx, rot_zyx
from .pngtext import put_text

SCHEMA = 'vantage.asbuilt/1'
ANALYSIS_W = 960            # images are analysed at most this wide (px)
THUMB_W = 640               # the colour frame kept per camera for the overlay
IMAGE_EXT = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.pgm', '.ppm', '.webp')
VERDICTS = ('holds', 'holds_with_corrections', 'does_not_hold', 'not_checkable')
VERDICT_TEXT = {'holds': 'Holds', 'holds_with_corrections': 'Holds with corrections', 'does_not_hold': 'Does not hold',
                'not_checkable': 'Not checkable'}
SEVERITIES = ('critical', 'serious', 'warning', 'good')
SOURCES = ('rosbag', 'nuscenes', 'images')
SOURCE_TEXT = {'rosbag': 'ROS 2 bag', 'nuscenes': 'nuScenes folder', 'images': 'image folder'}
AXES = ('yaw', 'pitch', 'roll')
TAXES = ('x', 'y', 'z')
OBS_DEG = 0.5               # an axis the outline pins down less well than this (from the pixel grid alone) stays as planned
OBS_M = 0.03                # the same for a translation axis (m)
SIGMA_Q = 0.35              # px: an outline cell's own error from the pixel grid, the cube's texels and the mesh's facets
CELL_PX = 24                # px: outline points in one cell share their errors (the sandwich's clusters, the floor)
VEH_SHARE = 0.6             # a static piece is the vehicle's when this share of it lies within the plan's band
MIN_OUTLINE_PX = 60         # px of simulated outline below which a camera is not checkable this way
MIN_OBSERVED = 0.25         # share of the simulated outline that must have a real outline in reach
MIN_FRAMES = 4
TURN_DEG = 3.0              # a heading change of this much counts as moving
COMPRESSION_GREY = 6.0      # the least grey-level deviation a static pixel may show (compression ringing)
RING_SHARE = 0.2            # … or this share of the largest deviation within 4 px (ringing next to a moving edge)
MAPPINGS_KEPT = 20
LIMITS = {'frames': (4, 60), 'min_motion_m': (0.0, 50.0), 'min_change': (0.5, 60.0), 'static_sigma': (2.0, 30.0),
          'edge_contrast': (2.0, 80.0), 'min_area_pct': (0.0, 5.0), 'search_deg': (0.5, 10.0),
          'tol_boundary_px': (0.2, 20.0), 'tol_boundary_deg': (0.02, 5.0), 'tol_rotation_deg': (0.02, 5.0),
          'tol_translation_m': (0.001, 0.5), 'tol_circle_pct': (0.1, 20.0), 'tol_blind_m2': (0.0, 10.0),
          'tol_multi_pct': (0.0, 50.0), 'tol_density_pct': (0.0, 50.0), 'tol_visible_pct': (0.0, 100.0)}
INTS = ('frames',)
BOOLS = ('fit_translation', 'lidar_check')
ASSUMPTIONS = (
    'The ego mask is what stays the same while the vehicle moves: it needs motion and texture in the scene. Plain road, '
    'sky and other surfaces without texture stay the same too; they are told from the vehicle by the static edges '
    'between them and by lying outside the search band of the planned silhouette, and where neither decides, that '
    'stretch of the outline is left out. Lighting, reflections in the paint and dirt on the lens change the mask. The '
    'simulated mask is only as good as the body mesh (a CAD mesh gives exact edges, a photogrammetry or generic body '
    'does not) and the occlusion cube it is looked up in (256 texels per face, about 0.35°). A camera that sees little '
    'of the vehicle constrains its rotation poorly, and the outline of a nearly straight edge pins only the rotation '
    'across it; such axes are reported as not constrained and left as planned. Translation is weakly observable (a '
    'distant edge moves the same for a turn as for a shift) and is fitted only when asked. The drive-log estimate is '
    'relative to the lidar, as that check is.')


# =========================================================================== settings
@dataclass
class AsBuiltSettings:
    path: str = ''                     # the recording last checked
    variant_uid: str = ''              # the variant to check against; '' = the active one
    frames: int = 24                   # frames kept per camera, spread over the drive
    min_motion_m: float = 0.5          # travel between kept frames (odometry, GNSS); 0 = every target frame
    min_change: float = 3.0            # without odometry or GNSS: mean grey-level change between kept frames
    static_sigma: float = 5.0          # a pixel varying less than this many noise sigmas is static
    edge_contrast: float = 10.0        # grey levels of a static edge that splits static areas
    min_area_pct: float = 0.05         # vehicle pieces (and holes in them) smaller than this share of the image are dropped
    search_deg: float = 3.0            # the band around the planned silhouette and the rotation search, ±
    fit_translation: bool = False      # also fit a translation (weakly observable)
    lidar_check: bool = True           # also run the drive-log rotation check against the lidar
    tol_boundary_px: float = 1.5       # mean outline distance a camera may show and hold, px …
    tol_boundary_deg: float = 0.3      # … and degrees
    tol_rotation_deg: float = 0.3      # a confident correction beyond this is a mounting error
    tol_translation_m: float = 0.02
    tol_circle_pct: float = 2.0        # image circle centre or radius off by more than this share of the radius
    tol_blind_m2: float = 0.05         # coverage: more blind ground than this over the plan's does not hold
    tol_multi_pct: float = 1.0         # … the share seen by two cameras dropping by more than these points
    tol_density_pct: float = 1.0       # … the shares at the minimum and target density
    tol_visible_pct: float = 3.0       # … a camera's visible ground changing by more than this share of the plan's
    mappings: dict = field(default_factory=dict)   # {recording path: {stream: tag ('' = no camera)}}: the edits only

    def to_dict(self):
        d = asdict(self)
        d['mappings'] = {p: dict(m) for p, m in self.mappings.items()}
        return d

    @staticmethod
    def from_dict(d: dict | None) -> 'AsBuiltSettings':
        """Tolerant of junk: unknown keys ignored, numbers clamped to LIMITS (NaN ignored), switches checked."""
        s = AsBuiltSettings()
        d = d if isinstance(d, dict) else {}
        if isinstance(d.get('path'), str):
            s.path = d['path'][:4096]
        if isinstance(d.get('variant_uid'), str):
            s.variant_uid = d['variant_uid'][:64]
        for k in BOOLS:
            v = d.get(k)
            if isinstance(v, bool) or (isinstance(v, int) and not isinstance(v, float)):
                setattr(s, k, bool(v))
            elif isinstance(v, str):
                setattr(s, k, v.strip().lower() in ('1', 'true', 'yes', 'on'))
        for k, (lo, hi) in LIMITS.items():
            if k not in d or isinstance(d[k], bool):
                continue
            try:
                v = float(d[k])
            except (TypeError, ValueError):
                continue
            if math.isfinite(v):
                v = min(max(v, lo), hi)
                setattr(s, k, int(round(v)) if k in INTS else v)
        maps = d.get('mappings')
        if isinstance(maps, dict):
            for p, m in list(maps.items())[-MAPPINGS_KEPT:]:
                if isinstance(p, str) and isinstance(m, dict):
                    s.mappings[p] = {t[:256]: g[:32] for t, g in m.items() if isinstance(t, str) and isinstance(g, str)}
        return s

    def overrides(self, path) -> dict:
        return dict(self.mappings.get(norm_path(path), {})) if path else {}


def norm_path(path) -> str:
    return DL.norm_path(path)


def settings_of(project) -> AsBuiltSettings:
    return AsBuiltSettings.from_dict(getattr(project.analysis, 'asbuilt', None))


# =========================================================================== recordings
def _natural(name):
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r'(\d+)', name)]


def _is_image(name) -> bool:
    return name.lower().endswith(IMAGE_EXT)


def image_streams(path) -> dict | None:
    """{stream name: [image files in name order]} of a plain image folder: each subfolder holding images is a stream
    named by the subfolder; images directly in the folder are grouped by the part of their name before the first
    '_', '-', '.' or space (F_000123.jpg -> stream F). Streams need two images or more. None when the folder holds no
    such streams (or is not a folder)."""
    p = norm_path(path)
    if not os.path.isdir(p):
        return None
    out = {}
    try:
        names = sorted(os.listdir(p), key=_natural)
    except OSError:
        return None
    for n in names:
        q = os.path.join(p, n)
        if os.path.isdir(q):
            try:
                files = [os.path.join(q, f) for f in sorted(os.listdir(q), key=_natural)
                         if _is_image(f) and os.path.isfile(os.path.join(q, f))]
            except OSError:
                continue
            if len(files) >= 2:
                out[n] = files
        elif _is_image(n) and os.path.isfile(q):
            m = re.match(r'([^_\-. ]+)[_\-. ]', n)
            if m:
                out.setdefault(m.group(1), []).append(q)
    out = {k: v for k, v in out.items() if len(v) >= 2}
    return out or None


def recording_kind(path) -> str:
    """'rosbag', 'nuscenes' or 'images'. FileNotFoundError when the path does not exist, ValueError when it is none of
    them."""
    p = norm_path(path)
    if not p or not os.path.exists(p):
        raise FileNotFoundError(f'Not found: {path}')
    if os.path.isfile(p):
        if p.lower().endswith(('.mcap', '.db3')):
            return 'rosbag'
        raise ValueError(f'{os.path.basename(p)} is not a recording: open a ROS 2 bag (.mcap, .db3 or a rosbag2 '
                         'folder), a nuScenes folder or a folder of images per camera.')
    if DL.is_nuscenes(p):
        return 'nuscenes'
    if os.path.isfile(os.path.join(p, 'metadata.yaml')) or any(n.lower().endswith(('.mcap', '.db3'))
                                                               for n in os.listdir(p)):
        return 'rosbag'
    if image_streams(p):
        return 'images'
    raise ValueError(f'{os.path.basename(p) or p} holds no recording: no rosbag2 files (.mcap, .db3), no nuScenes '
                     'tables and no images per camera (a subfolder per camera, or file names starting with its tag).')


IMAGE_TYPE = 'sensor_msgs/msg/Image'


def open_recording(path) -> DL.LogInfo:
    """The streams of a recording and their message counts, read cheaply (no images): a drivelog.LogInfo (source
    'rosbag', 'nuscenes' or 'images'; an image folder's streams are typed as images)."""
    kind = recording_kind(path)
    if kind != 'images':
        return DL.open_log(path)
    streams = image_streams(path) or {}
    return DL.LogInfo(norm_path(path), 'images', {k: IMAGE_TYPE for k in streams},
                      {k: len(v) for k, v in streams.items()}, 0, 0)


def recording_key(path) -> tuple:
    """What a scan was made from: the path and the size and modification time of its files."""
    p = norm_path(path)
    try:
        kind = recording_kind(p)
    except (OSError, ValueError):
        return p, ()
    if kind != 'images':
        key = DL.log_key(p)
        if kind == 'nuscenes':                      # the images live beside the tables
            nd = DL._nus_dirs(p)
            if nd is not None:
                files = []
                for sub in ('samples', 'sweeps'):
                    root = os.path.join(nd[1], sub)
                    if os.path.isdir(root):
                        try:
                            st = os.stat(root)
                            files.append((sub, int(st.st_mtime)))
                        except OSError:
                            pass
                key = (key[0], key[1] + tuple(files))
        return key
    files = []
    for name, lst in sorted((image_streams(p) or {}).items()):
        for f in lst:
            try:
                st = os.stat(f)
                files.append((os.path.relpath(f, p), st.st_size, int(st.st_mtime)))
            except OSError:
                pass
    return p, tuple(files)


@dataclass
class Frame:
    t_s: float                         # seconds after the recording's start (an image folder: the file's place)
    grey: np.ndarray                   # uint8 (ha, wa) at the analysis size
    size: tuple                        # (w, h) of the recorded image
    travel_m: float | None = None      # displacement from the stream's first kept frame (odometry / GNSS)


@dataclass
class StreamScan:
    topic: str
    type: str
    count: int = 0
    size: tuple | None = None          # (w, h) of the first decoded image
    frames: list = field(default_factory=list)
    jpeg: bytes = b''                  # colour thumbnail of the middle kept frame
    tsize: tuple = (0, 0)
    skipped_still: int = 0             # target frames passed over because the vehicle had not moved
    decode_errors: int = 0


@dataclass
class RecScan:
    path: str
    source: str
    key: tuple
    params: tuple                      # (frames, min_motion_m, min_change) it was read with
    start_ns: int
    end_ns: int
    streams: dict                      # stream -> StreamScan (image streams only)
    topics: dict                       # every topic -> (type, messages) (for the mapping table)
    motion: str = 'frames'             # what told motion: 'odometry' | 'GNSS' | 'IMU and frames' | 'frames'
    notes: list = field(default_factory=list)
    seconds: float = 0.0

    @property
    def duration_s(self) -> float:
        return max(0.0, (self.end_ns - self.start_ns) / 1e9)


def _rgb8(img, encoding='', scale=None):
    """An image as uint8 RGB (or grey): Bayer debayered, deeper images scaled (by `scale`, else their own 99.5th
    percentile), alpha dropped, YUV 4:2:2 taken as its luma. Returns (array, scale used)."""
    import cv2
    a = np.asarray(img)
    enc = (encoding or '').lower()
    if a.ndim == 2 and enc.startswith('bayer_'):
        code = {'bayer_rggb8': cv2.COLOR_BayerBG2RGB, 'bayer_bggr8': cv2.COLOR_BayerRG2RGB,
                'bayer_gbrg8': cv2.COLOR_BayerGR2RGB, 'bayer_grbg8': cv2.COLOR_BayerGB2RGB}.get(enc)
        if code is not None:
            a = cv2.cvtColor(a, code)
    if a.dtype != np.uint8:
        f = a.astype(np.float32)
        if scale is None:
            scale = float(np.nanpercentile(f, 99.5)) or 1.0
        a = np.clip(np.nan_to_num(f) / scale * 255.0, 0, 255).astype(np.uint8)
    if a.ndim == 3 and a.shape[2] == 4:
        a = a[..., :3]
    if a.ndim == 3 and a.shape[2] == 2:
        a = a[..., 1] if enc.startswith('uyvy') or enc == 'yuv422' else a[..., 0]
    if a.ndim == 3 and a.shape[2] == 1:
        a = a[..., 0]
    return np.ascontiguousarray(a), scale


def _grey(a):
    import cv2
    g = a if a.ndim == 2 else cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    h, w = g.shape
    s = min(1.0, ANALYSIS_W / w)
    wa, ha = max(8, int(round(w * s))), max(8, int(round(h * s)))
    if (wa, ha) != (w, h):
        g = cv2.resize(g, (wa, ha), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(g)


def _thumb(a) -> tuple[bytes, tuple]:
    import cv2
    rgb = a if a.ndim == 3 else np.repeat(a[..., None], 3, 2)
    h, w = rgb.shape[:2]
    s = min(1.0, THUMB_W / w)
    wt, ht = max(8, int(round(w * s))), max(8, int(round(h * s)))
    th = cv2.resize(rgb, (wt, ht), interpolation=cv2.INTER_AREA) if (wt, ht) != (w, h) else rgb
    ok, buf = cv2.imencode('.jpg', np.ascontiguousarray(th[..., ::-1]), [cv2.IMWRITE_JPEG_QUALITY, 88])
    return (buf.tobytes() if ok else b''), (wt, ht)


def _small(g):
    """A 64 px wide blurred grey copy: the frame-difference motion test."""
    import cv2
    h, w = g.shape
    ws, hs = 64, max(4, int(round(64 * h / w)))
    return cv2.GaussianBlur(cv2.resize(g, (ws, hs), interpolation=cv2.INTER_AREA).astype(np.float32), (0, 0), 1.0)


class _Mover:
    """Where the vehicle is (odometry or GNSS positions in metres, heading from odometry or an IMU's yaw rate) as the
    recording streams past, to tell whether it moved between two kept frames."""

    def __init__(self, kind):
        self.kind = kind               # 'odometry' | 'GNSS' | 'IMU' | 'none'
        self.pos = None                # latest (x, y, z) m
        self.yaw = None                # latest heading, deg
        self._ll0 = None
        self._t_imu = None

    @property
    def has_travel(self) -> bool:
        return self.kind in ('odometry', 'GNSS')

    def feed(self, t_ns, row):
        """One decoded small message (core.drivelog._decode_small's tuple)."""
        if self.kind == 'odometry':
            x, y, z, qx, qy, qz, qw = row[:7]
            if all(math.isfinite(v) for v in (x, y, z)):
                self.pos = np.array([x, y, z], float)
            if all(math.isfinite(v) for v in (qx, qy, qz, qw)) and (qx or qy or qz or qw):
                self.yaw = math.degrees(math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))
        elif self.kind == 'GNSS':
            status, lat, lon, alt = row[:4]
            if status >= 0 and math.isfinite(lat) and math.isfinite(lon):
                if self._ll0 is None:
                    self._ll0 = (lat, lon)
                k = 6378137.0 * math.pi / 180.0
                self.pos = np.array([(lon - self._ll0[1]) * k * math.cos(math.radians(self._ll0[0])),
                                     (lat - self._ll0[0]) * k, alt if math.isfinite(alt) else 0.0])
        elif self.kind == 'IMU':
            gz = row[2]
            if self._t_imu is not None and math.isfinite(gz):
                dt = (t_ns - self._t_imu) / 1e9
                if 0 < dt < 1.0:
                    self.yaw = (self.yaw or 0.0) + math.degrees(gz) * dt
            elif self.yaw is None:
                self.yaw = 0.0
            self._t_imu = t_ns

    def state(self):
        return (None if self.pos is None else self.pos.copy(), self.yaw)

    def moved(self, last, min_motion_m) -> tuple[bool | None, float | None]:
        """(moved since `last` (a state()), displacement m); moved None when this source cannot tell."""
        if last is None:
            return True, 0.0
        p0, y0 = last
        d = None
        if self.pos is not None and p0 is not None:
            d = float(np.linalg.norm(self.pos[:2] - p0[:2]))
        turned = (self.yaw is not None and y0 is not None and abs((self.yaw - y0 + 180.0) % 360.0 - 180.0) >= TURN_DEG)
        if self.has_travel and d is not None:
            return (d >= min_motion_m or turned), d
        if turned:
            return True, d
        return None, d


class _Picker:
    """Per stream: which images to keep. Target instants spread over the stream; at each, the first image at or after
    it whose vehicle moved since the last kept one (or whose picture changed, when motion is not known) is kept; a
    target with no such image before the next one is passed over."""

    def __init__(self, n, t0, t1, mover, st_min_motion, st_min_change):
        self.tt = np.array([t0 + (t1 - t0) * k / max(n, 1) for k in range(n)], float)
        self.k = 0
        self.mover = mover
        self.min_motion, self.min_change = st_min_motion, st_min_change
        self.last = None               # mover state at the last kept frame
        self.first = None
        self.last_small = None
        self.skipped = 0

    def due(self, t) -> bool:
        while self.k + 1 < len(self.tt) and t >= self.tt[self.k + 1]:
            self.k += 1                # the window passed without a kept frame
            self.skipped += 1
        return self.k < len(self.tt) and t >= self.tt[self.k]

    def motion_ok(self):
        """True / False when the motion source decides, None when the picture has to (frame differences)."""
        if self.min_motion <= 0 and self.mover.has_travel:
            return True
        ok, _ = self.mover.moved(self.last, self.min_motion)
        return ok

    def picture_ok(self, small) -> bool:
        if self.last_small is None or small.shape != self.last_small.shape:
            return True
        return float(np.mean(np.abs(small - self.last_small))) >= self.min_change

    def keep(self, small):
        self.last = self.mover.state()
        if self.first is None:
            self.first = self.last
        self.last_small = small
        self.k += 1

    def travel(self):
        if self.first is None or self.first[0] is None or self.mover.pos is None:
            return None
        return float(np.linalg.norm(self.mover.pos[:2] - self.first[0][:2]))


def scan_recording(path, frames: int = 24, min_motion_m: float = 0.5, min_change: float = 3.0, progress=None,
                   cancel=None) -> RecScan | None:
    """Read a recording once (see the module docstring); None when cancelled. FileNotFoundError / ValueError when it is
    not a recording."""
    t0 = time.perf_counter()
    p = norm_path(path)
    kind = recording_kind(p)
    params = (int(frames), float(min_motion_m), float(min_change))
    fn = {'rosbag': _scan_bag, 'nuscenes': _scan_nuscenes, 'images': _scan_images}[kind]
    sc = fn(p, params, progress, cancel)
    if sc is None:
        return None
    sc.key = recording_key(p)
    sc.seconds = time.perf_counter() - t0
    for s in sc.streams.values():
        if s.frames:
            mid = s.frames[len(s.frames) // 2]
            if not s.jpeg:
                s.jpeg, s.tsize = _thumb(mid.grey)
    return sc


def _keep_frame(ss: StreamScan, pk: _Picker, a_rgb, t_s, small):
    g = _grey(a_rgb)
    h, w = a_rgb.shape[:2]
    if ss.size is None:
        ss.size = (int(w), int(h))
    pk.keep(small)
    ss.frames.append(Frame(float(t_s), g, (int(w), int(h)), pk.travel()))
    n = len(pk.tt)
    if len(ss.frames) == (n + 1) // 2 or not ss.jpeg:  # the colour thumbnail of a frame near the middle
        ss.jpeg, ss.tsize = _thumb(a_rgb)


def _consider(ss: StreamScan, pk: _Picker, t_ns, t_s, get_rgb):
    """One image of a stream at log time t_ns: keep it when it is due and the vehicle moved. get_rgb() decodes it
    (called only when needed); returns False when it would not decode."""
    if not pk.due(t_ns):
        return True
    ok = pk.motion_ok()
    if ok is False:
        ss.skipped_still += 1
        return True
    try:
        a = get_rgb()
    except (ValueError, KeyError, TypeError):
        ss.decode_errors += 1
        return False
    if a is None or not getattr(a, 'size', 0):
        ss.decode_errors += 1
        return False
    small = _small(_grey(a))
    if ok is None and not pk.picture_ok(small):
        ss.skipped_still += 1
        return True
    _keep_frame(ss, pk, a, t_s, small)
    return True


def _motion_topic(types, counts):
    """(topic, kind) of the best motion source of a bag: odometry, then GNSS fixes, then an IMU."""
    for kind, word in (('odom', 'odometry'), ('fix', 'GNSS'), ('imu', 'IMU')):
        cands = [t for t, ty in types.items() if DL.topic_kind(ty) == kind and counts.get(t, 0) > 1]
        if cands:
            return max(cands, key=lambda t: counts.get(t, 0)), word
    return None, 'none'


def _scan_bag(path, params, progress, cancel) -> RecScan | None:
    from . import rosbag as RB
    n, min_motion, min_change = params
    rd = RB.BagReader(path)
    try:
        types = {t: DL.norm_type(ty) for t, ty in rd.topics.items()}
        counts = dict(rd.counts)
        t0, t1 = int(rd.start_ns), int(rd.end_ns)
        images = [t for t, ty in types.items() if DL.topic_kind(ty) == 'image']
        mtopic, mkind = _motion_topic(types, counts)
        mover = _Mover(mkind)
        streams = {t: StreamScan(t, types[t]) for t in images}
        notes = []
        if images:
            pickers = {t: _Picker(n, t0, t1, mover, min_motion, min_change) for t in images}
            scales = {}
            want = images + ([mtopic] if mtopic else [])
            span = max(t1 - t0, 1)
            k_all = 0
            for topic, t_ns, raw in rd.messages(want, decode=False):
                k_all += 1
                if k_all % 64 == 0:
                    if cancel is not None and cancel():
                        return None
                    if progress is not None:
                        progress(min(max((t_ns - t0) / span, 0.0), 1.0), 'reading the recording')
                if topic == mtopic:
                    try:
                        msg = rd.decode(topic, raw)
                        mover.feed(t_ns, DL._decode_small(DL.topic_kind(types[topic]), msg))
                    except (ValueError, KeyError, TypeError):
                        pass
                    continue
                ss = streams[topic]
                ss.count += 1

                def get(topic=topic, raw=raw):
                    msg = rd.decode(topic, raw)
                    img = RB.image_array(msg, to_rgb=True)
                    a, scales[topic] = _rgb8(img, msg.get('encoding', ''), scales.get(topic))
                    return a
                _consider(ss, pickers[topic], t_ns, (t_ns - t0) / 1e9, get)
        notes += [f'Bag: {w}' for w in getattr(rd, 'warnings', [])[:10]]
        sc = RecScan(path, 'rosbag', (), params, t0, t1, streams, {t: (ty, counts.get(t, 0)) for t, ty in types.items()},
                     {'odometry': 'odometry', 'GNSS': 'GNSS', 'IMU': 'IMU and frames', 'none': 'frames'}[mkind],
                     notes)
        return sc
    finally:
        rd.close()


def _scan_nuscenes(path, params, progress, cancel) -> RecScan | None:
    import cv2
    n, min_motion, min_change = params
    nd = DL._nus_dirs(path)
    ver, root = nd
    tables = DL._nus_tables(ver, ('sensor', 'calibrated_sensor', 'sample_data', 'ego_pose'))
    chans = DL._nus_channels(tables)
    by = {}
    for sd in tables['sample_data']:
        ch = chans.get(sd.get('calibrated_sensor_token'))
        if ch is None or not ch[0]:
            continue
        by.setdefault(ch, []).append(sd)
    all_ts = [int(sd['timestamp']) for v in by.values() for sd in v]
    t0, t1 = min(all_ts, default=0) * 1000, max(all_ts, default=0) * 1000
    eps = sorted(tables['ego_pose'], key=lambda e: e['timestamp'])
    ep_t = np.array([int(e['timestamp']) * 1000 for e in eps], np.int64)
    ep_p = np.array([e['translation'] for e in eps], float).reshape(-1, 3)
    ep_y = np.unwrap(np.array([math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
                               for q in (e['rotation'] for e in eps)], float)) if eps else np.zeros(0)
    streams, topics = {}, {}
    cams = [(ch, sds) for (ch, mod), sds in sorted(by.items()) if mod == 'camera']
    for (ch, mod), sds in by.items():
        topics[ch] = (DL.NUS_TYPE.get(mod, 'other'), len(sds))
    for ci, (ch, sds) in enumerate(cams):
        if cancel is not None and cancel():
            return None
        sds = sorted(sds, key=lambda s: s['timestamp'])
        mover = _Mover('odometry' if len(eps) > 1 else 'none')
        pk = _Picker(n, t0, t1, mover, min_motion, min_change)
        ss = streams[ch] = StreamScan(ch, DL.NUS_TYPE['camera'], len(sds))
        for s in sds:
            t_ns = int(s['timestamp']) * 1000
            if len(eps) > 1:
                mover.pos = np.array([np.interp(t_ns, ep_t, ep_p[:, k]) for k in range(3)])
                mover.yaw = math.degrees(float(np.interp(t_ns, ep_t, ep_y)))
            f = os.path.join(root, s.get('filename', ''))

            def get(f=f):
                img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
                if img is None:
                    raise ValueError(f'cannot read {f}')
                if img.ndim == 3:
                    img = np.ascontiguousarray(img[..., ::-1][..., :3])
                return _rgb8(img)[0]
            _consider(ss, pk, t_ns, (t_ns - t0) / 1e9, get)
        if progress is not None:
            progress((ci + 1) / max(len(cams), 1), f'reading {ch}')
    notes = ['nuScenes: motion from the ego poses.'] if len(eps) > 1 else []
    return RecScan(ver, 'nuscenes', (), params, t0, t1, streams, topics, 'odometry' if len(eps) > 1 else 'frames',
                   notes)


def _name_time(path) -> float | None:
    """A time in seconds from a file name holding an epoch stamp (10 digits s, 13 ms, 16 µs or 19 ns)."""
    m = re.findall(r'(\d{10,19})', os.path.basename(path))
    if not m:
        return None
    v = m[-1]
    return int(v) / {10: 1, 13: 1e3, 16: 1e6, 19: 1e9}.get(len(v), 10 ** (len(v) - 10))


def _scan_images(path, params, progress, cancel) -> RecScan | None:
    import cv2
    n, min_motion, min_change = params
    streams_f = image_streams(path) or {}
    streams, topics = {}, {}
    notes = ['Image folder: no motion data, so the frames are kept by how much the picture changed; times are the '
             'files\' places in name order unless their names hold a time stamp.']
    for ci, (name, files) in enumerate(sorted(streams_f.items())):
        if cancel is not None and cancel():
            return None
        times = [_name_time(f) for f in files]
        stamped = all(t is not None for t in times) and len(set(times)) == len(times)
        ts = [(t - times[0]) for t in times] if stamped else [k / 10.0 for k in range(len(files))]
        mover = _Mover('none')
        pk = _Picker(n, 0, len(files), mover, min_motion, min_change)
        ss = streams[name] = StreamScan(name, IMAGE_TYPE, len(files))
        topics[name] = (IMAGE_TYPE, len(files))
        scale = [None]
        for k, f in enumerate(files):
            def get(f=f):
                img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
                if img is None:
                    raise ValueError(f'cannot read {f}')
                if img.ndim == 3:
                    img = np.ascontiguousarray(img[..., ::-1][..., :3])
                a, scale[0] = _rgb8(img, '', scale[0])
                return a
            if not _consider(ss, pk, k, ts[k], get):
                notes.append(f'{name}: {os.path.basename(f)} would not decode.')
        if progress is not None:
            progress((ci + 1) / max(len(streams_f), 1), f'reading {name}')
    dur = max((max(s.frames[-1].t_s for s in streams.values() if s.frames), 0.0)) if any(
        s.frames for s in streams.values()) else 0.0
    return RecScan(path, 'images', (), params, 0, int(dur * 1e9), streams, topics, 'frames', notes)


# =========================================================================== mapping
def mapping_for(var, scan_or_info, overrides: dict) -> dict:
    """stream -> camera tag: the drive-log check's automatic mapping (tag in the name), then the user's edits; image
    streams and cameras only."""
    if isinstance(scan_or_info, RecScan):
        info = DL.LogInfo(scan_or_info.path, scan_or_info.source, {t: ty for t, (ty, _) in scan_or_info.topics.items()},
                          {t: c for t, (_, c) in scan_or_info.topics.items()}, scan_or_info.start_ns,
                          scan_or_info.end_ns)
    else:
        info = scan_or_info
    m = DL.mapping_for(var, info, overrides or {})
    cams = {c.tag for c in var.cameras if c.visible}
    return {t: g for t, g in m.items() if g in cams and DL.topic_kind(info.topics.get(t, '')) == 'image'}


def compatible_tags(var, msg_type) -> list:
    """Camera tags an image stream can be mapped to (none for other streams)."""
    return [t for t in DL.compatible_tags(var, msg_type) if var.by_tag(t) is not None]


# =========================================================================== per-camera geometry
def _intr(project, cam):
    from .lens import resolve
    return resolve(project.module_for(cam), project.lens_for(cam))


def _rays(intr, ashape, ss=1):
    """Optical-frame rays and theta of the analysis pixels' sample points (ss x ss per pixel): pixel i's centre is at
    analysis coordinate i, its samples at (j + 0.5) / ss - 0.5."""
    ha, wa = ashape
    sx, sy = wa / intr.width, ha / intr.height
    xs = (np.arange(wa * ss) + 0.5) / ss - 0.5
    ys = (np.arange(ha * ss) + 0.5) / ss - 0.5
    U, V = np.meshgrid((xs + 0.5) / sx, (ys + 0.5) / sy)
    d, th = intr.unproject(U.ravel(), V.ravel())
    return d, th


def _W2O(cam, d=(0.0, 0.0, 0.0)):
    return BODY_TO_OPTICAL @ rot_zyx(cam.yaw + d[0], cam.pitch + d[1], cam.roll + d[2]).T


def sim_mask(project, cam, intr, ashape, engine, delta=(0.0, 0.0, 0.0), shift=(0.0, 0.0, 0.0), ss=1):
    """(body, inside the lens circle, body depth m) per analysis sample (ss x ss per pixel) for the camera at the plan
    pose plus the rotation delta (deg) and the shift (m, vehicle frame): the pixel's ray looked up in the occlusion
    cube at the camera's position, as core.synth renders the own vehicle."""
    from .synth import NEAR_CLIP
    ha, wa = ashape
    d, th = _rays(intr, ashape, ss)
    circle = (th <= intr.max_theta).reshape(ha * ss, wa * ss)
    vg = engine.vehicle_geom(project) if engine is not None else None
    if vg is None:
        z = np.zeros((ha * ss, wa * ss), bool)
        return z, circle, np.full(z.shape, np.inf, np.float32)
    pos = cam.pos() + np.asarray(shift, float)
    bd = engine.cube(pos).lookup(d @ _W2O(cam, delta))[0]
    body = (np.isfinite(bd) & (bd > NEAR_CLIP)).reshape(ha * ss, wa * ss) & circle
    return body, circle, np.where(body.ravel(), bd, np.inf).reshape(ha * ss, wa * ss).astype(np.float32)


def predicted_circle(intr, ashape) -> dict:
    """The lens model's image circle in analysis pixels: centre, radius (x; y by the pixel aspect) and whether it lies
    inside the image anywhere."""
    from .lens import r_of_theta
    ha, wa = ashape
    sx, sy = wa / intr.width, ha / intr.height
    r = float(r_of_theta(intr.max_theta, intr.f_mm, intr.projection, intr.kb)) / intr.pixel_mm
    cx, cy = intr.cx * sx - 0.5, intr.cy * sy - 0.5
    rx, ry = r * sx, r * intr.aspect * sy
    corners = np.array([[-0.5, -0.5], [wa - 0.5, -0.5], [-0.5, ha - 0.5], [wa - 0.5, ha - 0.5]])
    reach = np.hypot((corners[:, 0] - cx) / rx, (corners[:, 1] - cy) / ry)
    return {'cx': cx, 'cy': cy, 'r': float(rx), 'ry': float(ry), 'visible': bool(reach.max() > 1.0),
            'r_sensor_px': r, 'max_half_deg': math.degrees(intr.max_theta)}


def _pairs(A, B):
    """Midpoints (x, y) between 4-neighbour pixels one in A, the other in B, and the unit normal from A towards B."""
    pts, nrm = [], []
    for a, b, off, n_ in ((A[:, :-1], B[:, 1:], (0.5, 0.0), (1.0, 0.0)), (B[:, :-1], A[:, 1:], (0.5, 0.0), (-1.0, 0.0)),
                          (A[:-1, :], B[1:, :], (0.0, 0.5), (0.0, 1.0)), (B[:-1, :], A[1:, :], (0.0, 0.5), (0.0, -1.0))):
        ys, xs = np.nonzero(a & b)
        pts.append(np.c_[xs + off[0], ys + off[1]])
        nrm.append(np.tile(n_, (len(xs), 1)))
    return np.concatenate(pts).astype(np.float64), np.concatenate(nrm).astype(np.float64)


def _smooth_normals(A, pts, nrm, sigma=1.5):
    """The pairs' normals from the gradient of A blurred by sigma (out of A), where it is clear; else the axis one."""
    import cv2
    if not len(pts):
        return nrm
    f = cv2.GaussianBlur(A.astype(np.float32), (0, 0), sigma)
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    h, w = A.shape
    x = np.clip(pts[:, 0], 0, w - 1.001)
    y = np.clip(pts[:, 1], 0, h - 1.001)
    x0, y0 = x.astype(int), y.astype(int)
    fx, fy = x - x0, y - y0
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)

    def bil(g):
        return (g[y0, x0] * (1 - fx) * (1 - fy) + g[y0, x1] * fx * (1 - fy) + g[y1, x0] * (1 - fx) * fy
                + g[y1, x1] * fx * fy)
    n = -np.c_[bil(gx), bil(gy)]
    L = np.linalg.norm(n, axis=1)
    good = L > 0.02
    out = nrm.copy()
    out[good] = n[good] / L[good, None]
    flip = np.sum(out * nrm, axis=1) < 0
    out[flip] = nrm[flip]
    return out


def _contours(mask, step=1.0) -> list:
    """The outlines of a mask as float32 (K, 2) polylines in analysis pixels (closed)."""
    import cv2
    if mask is None or not mask.any():
        return []
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cs:
        if len(c) < 3:
            continue
        c = cv2.approxPolyDP(c, step * 0.5, True) if step else c
        out.append(c.reshape(-1, 2).astype(np.float32))
    return out


# =========================================================================== the frames' statistics
def frame_stats(frames, block=48):
    """(median M, 90th percentile of |frame - M|, noise sigma) per pixel of the kept frames (grey uint8, one size),
    each frame first matched to the median by a gain and an offset fitted on the pixels it shares with it (auto
    exposure). Row blocks keep the memory small."""
    import cv2
    A = np.stack(frames).astype(np.float32)               # (N, h, w): at most 60 x 960 x 768
    N, h, w = A.shape
    M = np.median(A, 0)
    if N >= 4:
        for _ in range(2):
            changed = False
            for k in range(N):
                d = np.abs(A[k] - M)
                sel = (d < 8.0) & (M > 12) & (M < 243)
                if sel.sum() < 500:
                    continue
                x, y = M[sel], A[k][sel]
                X = np.c_[x, np.ones_like(x)]
                a, b = np.linalg.lstsq(X, y, rcond=None)[0]
                if abs(a - 1.0) > 0.01 or abs(b) > 0.5:
                    A[k] = (A[k] - b) / max(a, 1e-3)
                    changed = True
            if not changed:
                break
            M = np.median(A, 0)
    q = min(N - 1, max(0, int(math.ceil(0.9 * N)) - 1))
    D = np.empty((h, w), np.float32)
    for r0 in range(0, h, block):
        blk = np.abs(A[:, r0:r0 + block] - M[r0:r0 + block])
        D[r0:r0 + block] = np.partition(blk, q, axis=0)[q]
    del A
    D = cv2.medianBlur(D, 3)
    # the noise: the stillest tenth of the image (for Gaussian noise the 90th percentile of |x - median| is 1.645
    # sigma); pixels the moving world reaches through blur or compression are caught by the threshold, and the outline
    # is put back on the median image's edge afterwards (refine_outline)
    mid = (M > 12) & (M < 243)
    base = float(np.percentile(D[mid], 10)) if mid.sum() > 100 else float(np.percentile(D, 10))
    sigma = max(base / 1.645, 0.5)
    return M, D, sigma


def detect_circle(M, static, sigma, contrast=10.0) -> dict | None:
    """The dark, flat, still area outside a fisheye's image circle, grown from the image corners, and the circle fitted
    to its edge (the edge points put on the median image's intensity step first: compression rings next to the sharp
    edge and makes the grown area ragged). None when no corner is dark, flat and still, or the edge is not a circle's
    (the circle covers the sensor; a dark part of the vehicle in a corner)."""
    import cv2
    h, w = M.shape
    k = max(3, min(h, w) // 48)
    med = float(np.median(M[h // 4:3 * h // 4, w // 4:3 * w // 4]))      # the picture's middle: inside any circle
    gm = np.hypot(cv2.Sobel(M, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(M, cv2.CV_32F, 0, 1, ksize=3)) / 8.0
    seeds = []
    for ys, xs in ((slice(0, k), slice(0, k)), (slice(0, k), slice(w - k, w)), (slice(h - k, h), slice(0, k)),
                   (slice(h - k, h), slice(w - k, w))):
        patch = M[ys, xs]
        v = float(np.median(patch))
        if (float(patch.std()) <= max(2.0, 3 * sigma) and static[ys, xs].mean() >= 0.9 and v <= 72.0
                and v <= med - 15.0):
            seeds.append((ys.start + k // 2, xs.start + k // 2, v))
    if not seeds:
        return None
    tol = max(8.0, 4.0 * sigma)
    ext = np.zeros((h, w), bool)
    for yy, xx, v in seeds:
        cand = (static & (np.abs(M - v) <= tol) & (gm <= max(4.0, 3.0 * sigma))).astype(np.uint8)
        n_, lab = cv2.connectedComponents(cand, connectivity=4)
        if lab[yy, xx] > 0:
            ext |= lab == lab[yy, xx]
    if ext.sum() < 0.002 * h * w:
        return None
    ext = cv2.morphologyEx(ext.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0
    inside = ~ext
    pts, nrm = _pairs(inside, ext)
    away = (pts[:, 0] > 1) & (pts[:, 0] < w - 2) & (pts[:, 1] > 1) & (pts[:, 1] < h - 2)
    pts = pts[away]
    if len(pts) < 20:
        return None
    rough = fit_circle(pts, tol=3.0)
    if rough is None:
        return None
    v = float(np.median([s_[2] for s_ in seeds]))
    edge = _circle_edge(M, v, (rough['cx'], rough['cy']), rough['r'], max(tol, contrast / 2.0))
    fit = fit_circle(edge, tol=1.0) if len(edge) >= 20 else None
    if fit is None or fit['rms_px'] > 1.5 or fit['inlier_share'] < 0.6 or fit['arc_deg'] < 60.0:
        return None
    # the exterior: everything beyond the fitted circle that the grown area reached, and the ring of compression
    # noise next to the edge
    yy, xx = np.mgrid[0:h, 0:w]
    beyond = np.hypot(xx - fit['cx'], yy - fit['cy']) > fit['r']
    return {'exterior': beyond & (cv2.dilate(ext.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0), 'fit': fit}


def _circle_edge(M, v, centre, r0, tol, n=720, span=12.0):
    """Points on a lens circle's edge: along rays from the rough centre, scanned from outside in, where the median
    image first leaves the exterior's level v by half the local step to what lies inside (at least tol); rays whose
    inside is as dark as the exterior, or whose outside is not flat, give none."""
    h, w = M.shape
    th = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    t = np.arange(r0 + span, r0 - span - 1e-9, -0.25)
    X = centre[0] + np.cos(th)[:, None] * t[None, :]
    Y = centre[1] + np.sin(th)[:, None] * t[None, :]
    ok = ((X >= 0) & (Y >= 0) & (X <= w - 1) & (Y <= h - 1)).all(1)
    X, Y, th = X[ok], Y[ok], th[ok]
    if not len(X):
        return np.zeros((0, 2))
    x0 = np.clip(X.astype(int), 0, w - 2)
    y0 = np.clip(Y.astype(int), 0, h - 2)
    fx, fy = X - x0, Y - y0
    vals = (M[y0, x0] * (1 - fx) * (1 - fy) + M[y0, x0 + 1] * fx * (1 - fy) + M[y0 + 1, x0] * (1 - fx) * fy
            + M[y0 + 1, x0 + 1] * fx * fy)
    dev = np.abs(vals - v)
    inner = np.median(dev[:, -16:], axis=1)                 # the last 4 px of each ray: inside the circle
    thr = np.maximum(tol, 0.5 * inner)
    above = dev >= thr[:, None]
    i = np.argmax(above, axis=1)
    good = above.any(1) & (i > 4) & (inner >= 2 * tol)
    rows = np.nonzero(good)[0]
    rows = rows[np.array([dev[r, :i[r]].max() < thr[r] for r in rows], bool)] if len(rows) else rows
    if not len(rows):
        return np.zeros((0, 2))
    ii = i[rows]
    a, b = dev[rows, ii - 1], dev[rows, ii]
    f = np.clip((thr[rows] - a) / np.maximum(b - a, 1e-9), 0.0, 1.0)
    rr = t[ii - 1] + (t[ii] - t[ii - 1]) * f
    return np.c_[centre[0] + np.cos(th[rows]) * rr, centre[1] + np.sin(th[rows]) * rr]


def fit_circle(pts, iters=300, tol=1.5, seed=0) -> dict | None:
    """Circle (cx, cy, r) through edge points: RANSAC on three points, then least squares (geometric) on the inliers.
    Also the rms, the inliers and the angle the points span (a short arc pins the circle poorly)."""
    from scipy.optimize import least_squares
    P = np.asarray(pts, float)
    if len(P) < 5:
        return None
    rng = np.random.default_rng(seed)
    best, best_n = None, 0
    for _ in range(iters):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        A = np.array([[b[0] - a[0], b[1] - a[1]], [c[0] - a[0], c[1] - a[1]]]) * 2
        y = np.array([b @ b - a @ a, c @ c - a @ a])
        if abs(np.linalg.det(A)) < 1e-6:
            continue
        ctr = np.linalg.solve(A, y)
        r = np.linalg.norm(a - ctr)
        n = int(np.sum(np.abs(np.linalg.norm(P - ctr, axis=1) - r) <= tol))
        if n > best_n:
            best, best_n = (ctr, r), n
    if best is None:
        return None
    inl = np.abs(np.linalg.norm(P - best[0], axis=1) - best[1]) <= 3 * tol
    Q = P[inl]

    def res(x):
        return np.linalg.norm(Q - x[:2], axis=1) - x[2]
    sol = least_squares(res, np.r_[best[0], best[1]], loss='soft_l1', f_scale=tol)
    cx, cy, r = (float(v) for v in sol.x)
    rr = res(sol.x)
    ang = np.degrees(np.arctan2(Q[:, 1] - cy, Q[:, 0] - cx))
    hist = np.histogram(ang, bins=36, range=(-180, 180))[0] > 0
    return {'cx': cx, 'cy': cy, 'r': abs(r), 'rms_px': float(np.sqrt(np.mean(rr ** 2))), 'points': int(len(Q)),
            'inlier_share': float(len(Q) / len(P)), 'arc_deg': float(hist.sum() * 10.0)}


# =========================================================================== the static (ego) mask
def static_threshold(sigma, st: AsBuiltSettings) -> float:
    """Grey levels a pixel's 90th-percentile deviation may reach and the pixel still count as static: static_sigma
    noise sigmas, but never under COMPRESSION_GREY (a JPEG or H.264 picture rings next to a moving edge by that much,
    and a smooth surface's noise is compressed away, so the noise alone would set it too tight)."""
    return max(st.static_sigma * sigma, COMPRESSION_GREY)


VEH, WORLD, UND, OUT = 1, 2, 3, 4      # labels: the vehicle, the world (moving or static scenery), undecided, outside


def classify(M, D, sigma, body_plan, inside, band_px, st: AsBuiltSettings) -> dict:
    """Label every pixel VEH / WORLD / UND / OUT from the frames' statistics and the plan's silhouette (see the module
    docstring). body_plan: the plan's body mask; inside: inside the (measured, else predicted) image circle; band_px:
    the search band's width."""
    import cv2
    from scipy import ndimage
    h, w = M.shape
    T = static_threshold(sigma, st)
    # static: under the threshold, or a pixel the moving world next to it only reaches through compression or blur
    # (a fifth of the largest deviation within 4 px, and under four times the threshold): the vehicle's first pixels
    # next to a moving edge ring with a share of that edge's change, the world's own pixels change as much as their
    # neighbours
    near_max = cv2.dilate(D, np.ones((9, 9), np.uint8))
    static = ((D <= T) | ((D <= 4.0 * T) & (D <= RING_SHARE * near_max))) & inside
    # static edges of the median image (Canny on a light blur): they split static areas
    m8 = np.clip(cv2.GaussianBlur(M, (0, 0), 0.7), 0, 255).astype(np.uint8)
    hi = 4.0 * st.edge_contrast * 0.7
    edges = cv2.Canny(m8, 0.5 * hi, hi, L2gradient=True) > 0
    barrier = cv2.dilate((edges & static).astype(np.uint8), np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)) > 0
    barrier &= static
    free = static & ~barrier
    n, lab = cv2.connectedComponents(free.astype(np.uint8), connectivity=4)
    dt_out = cv2.distanceTransform((~body_plan).astype(np.uint8), cv2.DIST_L2, 5)
    dt_in = cv2.distanceTransform(body_plan.astype(np.uint8), cv2.DIST_L2, 5)
    outer = body_plan | (dt_out <= band_px)
    inner = body_plan & (dt_in > band_px)
    area = np.bincount(lab.ravel(), minlength=n)
    in_outer = np.bincount(lab.ravel(), weights=outer.ravel().astype(float), minlength=n)
    share = np.where(area > 0, in_outer / np.maximum(area, 1), 0.0)
    veh_comp = share >= VEH_SHARE
    veh_comp[0] = False
    lab_veh = veh_comp[lab]
    labels = np.full((h, w), WORLD, np.uint8)
    scenery = free & ~lab_veh
    labels[free & lab_veh] = VEH
    # static pieces mostly outside the band (plain road, sky). One reaching deep inside the plan's body is the vehicle
    # and the scenery run together where no static edge parts them: deep inside is the vehicle (no turn within the
    # band moves it off), the band undecided. One that does not is scenery parted from the vehicle by a static edge;
    # the edge is the vehicle's outline when the scenery shows more temporal variation near it than the vehicle does
    # (road noise and mottle, which paint does not have), else it may be a line on the body with more body beyond
    # it, and the band stays undecided.
    in_inner = np.bincount(lab.ravel(), weights=inner.ravel().astype(float), minlength=n)
    merged = (~veh_comp) & (in_inner >= np.maximum(0.01 * area, 50))
    merged[0] = False
    lab_merged = merged[lab]
    labels[scenery & inner] = VEH
    labels[scenery & lab_merged & outer & ~inner] = UND
    pure = scenery & ~lab_merged & outer & ~inner
    if pure.any() and barrier.any() and (free & lab_veh).any():
        near = cv2.distanceTransform((~barrier).astype(np.uint8), cv2.DIST_L2, 3) <= 4.0
        dv = D[near & free & lab_veh & outer]
        d_veh = float(np.median(dv)) if len(dv) > 20 else None
        ids = np.unique(lab[pure])
        for c in ids:
            sel = (lab == c) & outer & ~inner
            ds = D[sel & near]
            if d_veh is None or len(ds) < 10 or not (np.median(ds) >= 1.4 * d_veh and
                                                      np.median(ds) - d_veh >= 0.5 * sigma):
                labels[sel] = UND
    elif pure.any():
        labels[pure] = UND
    # the barrier pixels (static themselves) take the label of the nearest static pixel off the barrier
    if barrier.any() and free.any():
        idx = ndimage.distance_transform_edt(~free, return_distances=False, return_indices=True)
        labels[barrier] = labels[idx[0][barrier], idx[1][barrier]]
    labels[~inside] = OUT
    # clean the vehicle: drop small pieces, fill small holes
    a_min = max(4, int(st.min_area_pct / 100.0 * h * w))
    veh = labels == VEH
    n2, lab2, stats, _ = cv2.connectedComponentsWithStats(veh.astype(np.uint8), connectivity=8)
    small = np.zeros(n2, bool)
    small[1:] = stats[1:, cv2.CC_STAT_AREA] < a_min
    labels[small[lab2] & veh] = WORLD
    veh = labels == VEH
    n3, lab3, stats3, _ = cv2.connectedComponentsWithStats((~veh & inside).astype(np.uint8), connectivity=4)
    if n3 > 1:
        border = np.zeros(n3, bool)
        border[np.unique(np.r_[lab3[0], lab3[-1], lab3[:, 0], lab3[:, -1]])] = True
        hole = np.zeros(n3, bool)
        hole[1:] = (stats3[1:, cv2.CC_STAT_AREA] < a_min) & ~border[1:]
        # a hole touching the circle's edge is not a hole
        if (~inside).any():
            edge = cv2.dilate((~inside).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            hole[np.unique(lab3[edge])] = False
        labels[hole[lab3]] = VEH
    # small islands of the world inside the vehicle or the undecided band (a speck that changed): unknown, not an
    # outline
    world = labels == WORLD
    n4, lab4, stats4, _ = cv2.connectedComponentsWithStats(world.astype(np.uint8), connectivity=4)
    if n4 > 1:
        edge = np.zeros(n4, bool)
        edge[np.unique(np.r_[lab4[0], lab4[-1], lab4[:, 0], lab4[:, -1]])] = True
        if (~inside).any():
            ring = cv2.dilate((~inside).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            edge[np.unique(lab4[ring])] = True
        speck = np.zeros(n4, bool)
        speck[1:] = (stats4[1:, cv2.CC_STAT_AREA] < a_min) & ~edge[1:]
        labels[speck[lab4] & world] = UND
    return {'labels': labels, 'static': static, 'edges': edges & static, 'threshold': T,
            'static_pct': 100.0 * float(static.sum()) / max(int(inside.sum()), 1),
            'scenery_pct': 100.0 * float((static & (labels == WORLD)).sum()) / max(int(inside.sum()), 1),
            'undecided_pct': 100.0 * float((labels == UND).sum()) / max(int(inside.sum()), 1)}


# =========================================================================== the outline fit
NPAR = 7                    # the fit's parameters: yaw, pitch, roll (deg), x, y, z (m), the outline offset (px)


class _Outline:
    """The plan's simulated outline as vehicle-frame directions from the camera (and points at the body's depth, for a
    translation) and the real outline (refined midpoints and normals) with a k-d tree."""

    def __init__(self, cam, intr, ashape, sim_pts, sim_dirs, sim_P, real_pts, real_nrm):
        from scipy.spatial import cKDTree
        self.cam, self.intr, self.ashape = cam, intr, ashape
        ha, wa = ashape
        self.sx, self.sy = wa / intr.width, ha / intr.height
        self.D, self.P = sim_dirs, sim_P
        self.plan_pts = sim_pts
        self.Q, self.Nq = real_pts, real_nrm
        self.tree = cKDTree(real_pts) if len(real_pts) else None
        self.pos = cam.pos()

    def subset(self, n):
        """The same outline with at most n simulated points (evenly picked): the coarse search."""
        if len(self.D) <= n:
            return self
        keep = np.linspace(0, len(self.D) - 1, n).astype(int)
        sub = _Outline.__new__(_Outline)
        sub.__dict__.update(self.__dict__)
        sub.D, sub.P, sub.plan_pts = self.D[keep], self.P[keep], self.plan_pts[keep]
        return sub

    def project(self, x):
        """Analysis-pixel positions of the simulated outline for the pose part of x (dyaw, dpitch, droll[, dx, dy,
        dz]) and which land inside the lens circle."""
        W2O = _W2O(self.cam, x[:3])
        if len(x) > 3 and np.any(np.asarray(x[3:6], float) != 0):
            V = self.P - (self.pos + np.asarray(x[3:6], float))
            V = V / np.maximum(np.linalg.norm(V, axis=1), 1e-9)[:, None]
        else:
            V = self.D
        u, v, th, ok = self.intr.project_dirs(V @ W2O.T)
        ok &= th <= self.intr.max_theta
        return np.c_[u * self.sx - 0.5, v * self.sy - 0.5], ok

    def match(self, p, ok, gate, offset=0.0):
        """Nearest real outline point per projected point: index, point-to-line residual (px, along the real normal,
        less the outline offset), distance and whether it is in reach (distance under gate, the real outline passing
        within 2.5 px of the point's normal line)."""
        dist, j = self.tree.query(p, distance_upper_bound=gate)
        good = ok & np.isfinite(dist)
        j = np.where(good, j, 0)
        dv = p - self.Q[j]
        r0 = np.sum(dv * self.Nq[j], axis=1)
        tang = np.sqrt(np.maximum(np.sum(dv * dv, axis=1) - r0 * r0, 0.0))
        good &= tang <= 2.5
        return j, r0 - offset, np.where(good, dist, np.inf), good


def _robust_chamfer(ol: _Outline, dtmap, x, c):
    """Mean Cauchy cost of the distance from the simulated outline at x to the real one (bilinear distance map)."""
    p, ok = ol.project(x)
    h, w = dtmap.shape
    xi, yi = p[:, 0], p[:, 1]
    inb = ok & (xi >= 0) & (yi >= 0) & (xi <= w - 1) & (yi <= h - 1)
    x0 = np.clip(xi.astype(int), 0, w - 2)
    y0 = np.clip(yi.astype(int), 0, h - 2)
    fx, fy = np.clip(xi - x0, 0, 1), np.clip(yi - y0, 0, 1)
    d = (dtmap[y0, x0] * (1 - fx) * (1 - fy) + dtmap[y0, x0 + 1] * fx * (1 - fy) + dtmap[y0 + 1, x0] * (1 - fx) * fy
         + dtmap[y0 + 1, x0 + 1] * fx * fy)
    cap = math.log1p(9.0)                                   # beyond 3 c: no real outline there (the same as unseen)
    s = np.where(inb, np.minimum(np.log1p((d / c) ** 2), cap), cap)
    return float(s.mean())


def _cost(ol: _Outline, x, c=1.0):
    """The robust cost of a pose: per simulated outline point the Cauchy cost of its point-to-line residual (less the
    offset), capped at 3 c, and the cap for a point with no real outline in reach."""
    p, ok = ol.project(x[:6])
    j, r, dist, good = ol.match(p, ok, 4.0 * c + 2.0, x[6])
    cap = math.log1p(9.0)
    return float(np.mean(np.where(good, np.minimum(np.log1p((r / c) ** 2), cap), cap)))


def _jacobian(ol: _Outline, x, j, cols, h_rot=0.01, h_t=0.002):
    """d(residual)/dx per point for fixed matches j, the columns asked for: the projected point's motion along the
    real normal for the pose, -1 for the offset."""
    out = []
    for k in cols:
        if k == 6:
            out.append(-np.ones(len(j)))
            continue
        hk = h_rot if k < 3 else h_t
        e = np.zeros(6)
        e[k] = hk
        pp, _ = ol.project(x[:6] + e)
        pm, _ = ol.project(x[:6] - e)
        out.append(np.sum((pp - pm) / (2 * hk) * ol.Nq[j], axis=1))
    return np.stack(out, 1)


def _irls(ol: _Outline, x0, free, c_steps=(6.0, 3.0, 1.5, 1.0), iters=25):
    """Gauss-Newton on Cauchy-weighted point-to-line residuals; free: which of the NPAR parameters move. Returns the
    parameters and whether it settled with enough points."""
    x = np.asarray(x0, float).copy()
    fi = np.nonzero(free)[0]
    done = False
    lim = np.array([1.0, 1.0, 1.0, 0.05, 0.05, 0.05, 1.0])[fi]          # deg, m, px per step
    tiny = np.array([2e-4, 2e-4, 2e-4, 2e-5, 2e-5, 2e-5, 1e-3])[fi]
    for c in c_steps:
        gate = 4.0 * c + 2.0
        for _ in range(iters):
            p, ok = ol.project(x[:6])
            j, r, dist, good = ol.match(p, ok, gate, x[6])
            if good.sum() < 10:
                return x, False
            w = np.where(good, 1.0 / (1.0 + (r / c) ** 2), 0.0)
            J = _jacobian(ol, x, j, fi)
            H = J.T @ (J * w[:, None])
            g = J.T @ (w * r)
            lam = 1e-6 * max(float(np.trace(H)), 1e-9)
            try:
                step = -np.linalg.solve(H + lam * np.eye(len(fi)), g)
            except np.linalg.LinAlgError:
                return x, False
            step = np.clip(step, -lim, lim)
            x[fi] += step
            done = True
            if np.all(np.abs(step) < tiny):
                break
    return x, done


def fit_outline(ol: _Outline, dtmap, search_deg, translation=False) -> dict:
    """The pose correction laying the simulated outline onto the real one (see the module docstring): x holds yaw,
    pitch, roll (deg), the shift x, y, z (m, held at 0 unless translation) and the outline offset (px: the real outline
    lying that far outside the simulated one all round, as blur or the static threshold make it; fitted so it cannot
    pass for a rotation)."""
    S = float(search_deg)
    c0 = 6.0
    sub = ol.subset(1500)
    # stage 1: a coarse grid of rotations on the distance map; stage 2: finer around the best three
    g1 = np.linspace(-S, S, 7)
    grid = np.array(np.meshgrid(g1, g1, g1, indexing='ij')).reshape(3, -1).T
    sc = np.array([_robust_chamfer(sub, dtmap, d, c0) for d in grid])
    picks = []
    step1 = g1[1] - g1[0]
    for i in np.argsort(sc):
        if all(np.abs(grid[i] - grid[j]).max() > 1.5 * step1 for j in picks):
            picks.append(i)
        if len(picks) == 3:
            break
    g2 = np.linspace(-step1, step1, 5)
    fine = np.array(np.meshgrid(g2, g2, g2, indexing='ij')).reshape(3, -1).T
    best, bval = np.zeros(3), _robust_chamfer(sub, dtmap, np.zeros(3), c0 / 2)
    for i in picks:
        for d in grid[i] + fine:
            v = _robust_chamfer(sub, dtmap, d, c0 / 2)
            if v < bval:
                best, bval = d, v
    free = np.array([True, True, True] + [bool(translation)] * 3 + [True])
    # from the grid's best and from the plan: the lower final cost wins (a grid point can sit on a wrong match)
    tries = []
    starts = [(best, (6.0, 3.0, 1.5, 1.0)), (best, (2.0, 1.0))] + ([(np.zeros(3), (2.0, 1.0))] if np.any(best) else [])
    for start, steps in starts:
        xs, oks = _irls(ol, np.r_[start, np.zeros(NPAR - 3)], free, c_steps=steps)
        if oks and np.all(np.abs(xs[:3]) <= S + 0.5):
            tries.append((_cost(ol, xs), xs))
    out = {'x': np.zeros(NPAR), 'free': free.copy(), 'converged': bool(tries), 'translation': bool(translation)}
    if not tries:
        out['why'] = f'no pose within ±{S:g}° lines the outlines up'
        return out
    x = min(tries, key=lambda t: t[0])[1]
    ok = True
    # observability: the pixel grid's error alone, per axis; refit with the loose axes held at the plan
    obs_lim = np.array([OBS_DEG] * 3 + [OBS_M] * 3 + [np.inf])
    for _ in range(3):
        se, se_floor, n_used, cells = _errors(ol, x, free)
        loose = free & ((se_floor > obs_lim) | ~np.isfinite(se_floor))
        loose[6] = False
        if not loose.any():
            break
        free = free & ~loose
        x[~free & (np.arange(NPAR) < 6)] = 0.0
        if not free[:6].any():
            break
        x, ok = _irls(ol, x, free, c_steps=(3.0, 1.5, 1.0))
        if not ok or np.any(np.abs(x[:3]) > S + 0.5):
            out.update({'x': np.zeros(NPAR), 'converged': False, 'why': f'no pose within ±{S:g}° lines the outlines up'})
            return out
    if free[:6].any():
        se, se_floor, n_used, cells = _errors(ol, x, free)
    else:
        se, se_floor, n_used, cells = np.full(NPAR, np.inf), np.full(NPAR, np.inf), 0, 0
    out.update({'x': x, 'free': free, 'stderr': se, 'stderr_floor': se_floor, 'points': n_used, 'cells': cells,
                'converged': ok})
    return out


def _errors(ol: _Outline, x, free):
    """(standard error, its pixel-grid floor, points used, cells) per parameter at x: a cluster-robust sandwich of the
    Cauchy-weighted point-to-line residuals (CELL_PX cells as clusters) combined with the floor SIGMA_Q px per cell.
    Infinite for a parameter held fixed or not constrained at all."""
    fi = np.nonzero(free)[0]
    out = np.full(NPAR, np.inf)
    floor = np.full(NPAR, np.inf)
    p, ok = ol.project(x[:6])
    j, r, dist, good = ol.match(p, ok, 4.0, x[6])
    if good.sum() < 10 or not len(fi):
        return out, floor, int(good.sum()), 0
    c = 1.0
    w = np.where(good, 1.0 / (1.0 + (r / c) ** 2), 0.0)
    J = _jacobian(ol, x, j, fi)
    H = J.T @ (J * w[:, None])
    cell = (np.floor(p[:, 0] / CELL_PX).astype(np.int64) * 100003 + np.floor(p[:, 1] / CELL_PX).astype(np.int64))
    u, inv = np.unique(cell[good], return_inverse=True)
    K = len(u)
    G = (J * (w * r)[:, None])[good]
    Gc = np.stack([np.bincount(inv, weights=G[:, k], minlength=K) for k in range(len(fi))], 1)
    Hi = np.linalg.pinv(H)
    Mt = Gc.T @ Gc * (K / max(K - 1, 1))
    sand = np.diag(Hi @ Mt @ Hi)
    # the floor: each cell's mean residual uncertain by SIGMA_Q px (the pixel grid), cells independent
    Jw = (J * w[:, None])[good]
    Jc = np.stack([np.bincount(inv, weights=Jw[:, k], minlength=K) for k in range(len(fi))], 1)
    wc = np.bincount(inv, weights=w[good], minlength=K)
    Jm = Jc / np.maximum(wc, 1e-9)[:, None]
    Hc = Jm.T @ Jm
    flo = np.diag(np.linalg.pinv(Hc)) * SIGMA_Q ** 2
    ev, V = np.linalg.eigh(Hc)                     # a direction the cells do not pin at all
    if K < len(fi) or ev.min() <= 1e-9 * max(ev.max(), 1e-12):
        flo = np.where(np.abs(V[:, np.argmin(ev)]) > 0.2, np.inf, flo)
    out[fi] = np.sqrt(np.maximum(sand, 0.0) + flo)
    floor[fi] = np.sqrt(flo)
    return out, floor, int(good.sum()), K


def refine_outline(M, pts, nrm, contrast, reach=1.5):
    """Each real outline point moved along its normal to the median image's edge when there is one within reach (px):
    the centroid of the intensity gradient's magnitude above half its peak, on the samples connected to the peak.
    The static threshold puts the outline on the pixel grid and, where blur or compression lets the moving world leak
    into the vehicle's first pixels, a pixel inside; the edge of the median image is where the vehicle ends. Points
    without a clear edge (contrast / 3 grey levels per pixel) stay. Returns the points and how many moved."""
    h, w = M.shape
    if not len(pts):
        return pts, 0
    t = np.arange(-reach - 1.0, reach + 1.0 + 1e-9, 0.25)
    X = pts[:, None, 0] + t[None, :] * nrm[:, None, 0]
    Y = pts[:, None, 1] + t[None, :] * nrm[:, None, 1]
    inside = (X >= 0) & (Y >= 0) & (X <= w - 1) & (Y <= h - 1)
    Xc, Yc = np.clip(X, 0, w - 1.001), np.clip(Y, 0, h - 1.001)
    x0, y0 = Xc.astype(int), Yc.astype(int)
    fx, fy = Xc - x0, Yc - y0
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    v = (M[y0, x0] * (1 - fx) * (1 - fy) + M[y0, x1] * fx * (1 - fy) + M[y1, x0] * (1 - fx) * fy
         + M[y1, x1] * fx * fy)
    g = np.abs(np.gradient(v, 0.25, axis=1))
    core = np.abs(t) <= reach
    gc = np.where(core[None, :] & inside, g, 0.0)
    k = np.argmax(gc, axis=1)
    peak = gc[np.arange(len(pts)), k]
    half = gc >= 0.5 * peak[:, None]
    # the samples above half the peak connected to it
    idx = np.arange(len(t))[None, :]
    run = half.copy()
    left = np.maximum.accumulate(np.where(~half & (idx < k[:, None]), idx, -1), axis=1)
    right = np.minimum.accumulate(np.where(~half & (idx > k[:, None]), idx, len(t))[:, ::-1], axis=1)[:, ::-1]
    lo = left[np.arange(len(pts)), k]
    hi = right[np.arange(len(pts)), k]
    run &= (idx > lo[:, None]) & (idx < hi[:, None])
    wgt = np.where(run, gc, 0.0)
    tc = np.sum(wgt * t[None, :], axis=1) / np.maximum(wgt.sum(1), 1e-12)
    # an isolated edge only: well above the gradient just beyond reach on either side (texture in the world's
    # median, a line on the body)
    outer = np.where(inside & ~core[None, :], g, 0.0)
    g_in = outer[:, t < 0].max(axis=1)
    g_out = outer[:, t > 0].max(axis=1)
    ok = ((peak >= contrast / 3.0) & (peak >= 2.0 * np.maximum(g_in, g_out)) & (k > 0) & (k < len(t) - 1)
          & np.isfinite(tc))
    out = pts.copy()
    out[ok] += tc[ok, None] * nrm[ok]
    return out, int(ok.sum())


def _deg_per_px(intr, ashape, pts, nrm):
    """The angle (deg) one analysis pixel spans along each point's normal."""
    ha, wa = ashape
    sx, sy = wa / intr.width, ha / intr.height
    a = pts - 0.5 * nrm
    b = pts + 0.5 * nrm
    da, _ = intr.unproject((a[:, 0] + 0.5) / sx, (a[:, 1] + 0.5) / sy)
    db, _ = intr.unproject((b[:, 0] + 0.5) / sx, (b[:, 1] + 0.5) / sy)
    return np.degrees(np.arccos(np.clip(np.sum(da * db, axis=1), -1.0, 1.0)))


def _outline_distance(ol: _Outline, x, band_px, offset=0.0):
    """Mean and 95th percentile distance (px and deg) from the simulated outline at x to the real one (less a uniform
    offset, px), over the simulated points with a real outline in reach (band_px), and that share of the outline
    (observed)."""
    p, ok = ol.project(np.asarray(x, float)[:6])
    if ol.tree is None:
        return None
    j, r, dist, good = ol.match(p, ok, band_px, offset)
    n = int(ok.sum())
    if not good.any():
        return {'mean_px': None, 'p95_px': None, 'mean_deg': None, 'p95_deg': None, 'observed_pct': 0.0, 'points': n}
    a = np.abs(r[good])
    deg = a * _deg_per_px(ol.intr, ol.ashape, p[good], ol.Nq[j[good]])
    return {'mean_px': float(a.mean()), 'p95_px': float(np.percentile(a, 95)), 'mean_deg': float(deg.mean()),
            'p95_deg': float(np.percentile(deg, 95)), 'observed_pct': 100.0 * float(good.sum()) / max(n, 1),
            'points': n}


def _iou(real_veh, sim_body, decided):
    a = real_veh & decided
    b = sim_body & decided
    u = int((a | b).sum())
    return float((a & b).sum()) / u if u else None


# =========================================================================== one camera
def _row(cam, topic):
    return {'uid': cam.uid, 'tag': cam.tag, 'topic': topic, 'frames': 0, 'size': None, 'module_size': None,
            'ashape': None, 'check': 'not_checkable', 'reason': '', 'verdict': 'not_checkable', 'noise': None,
            'static_pct': None, 'scenery_pct': None, 'undecided_pct': None, 'circle': None, 'iou_plan': None,
            'iou_fit': None, 'outline_plan': None, 'outline_fit': None, 'outline_px': 0, 'fit': None,
            'correction_deg': [0.0, 0.0, 0.0], 'correction_m': [0.0, 0.0, 0.0], 'apply': False, 'lidar': None,
            'agree': None, 'coverage': None, 'contours': {}, 'jpeg': b'', 'median_jpeg': b'', 'tsize': (0, 0),
            'plan_deg': [cam.yaw, cam.pitch, cam.roll], 'plan_pos': [cam.x, cam.y, cam.z], 'travel_m': None}


def check_camera(project, cam, ss: StreamScan | None, st: AsBuiltSettings, engine) -> dict:
    """The static mask, the simulated mask, their comparison and the pose correction of one camera."""
    import cv2
    row = _row(cam, ss.topic if ss is not None else None)
    if ss is None:
        row['reason'] = 'no image stream is mapped to it'
        return row
    row['jpeg'], row['tsize'] = ss.jpeg, ss.tsize
    frames = ss.frames
    row['frames'] = len(frames)
    if len(frames) < MIN_FRAMES:
        row['reason'] = (f'{len(frames)} frame{"s" * (len(frames) != 1)} taken while moving (at least {MIN_FRAMES} '
                         'are needed): ' + ('the vehicle hardly moved' if ss.skipped_still else 'the stream is short'))
        return row
    shapes = {}
    for f in frames:
        shapes[f.grey.shape] = shapes.get(f.grey.shape, 0) + 1
    ashape = max(shapes, key=shapes.get)
    frames = [f for f in frames if f.grey.shape == ashape]
    w0, h0 = next(f.size for f in frames)
    mod = project.module_for(cam)
    W, H = int(mod.width_px), int(mod.height_px)
    row['size'], row['module_size'], row['ashape'] = (w0, h0), (W, H), ashape
    row['travel_m'] = max((f.travel_m for f in frames if f.travel_m is not None), default=None)
    if abs(w0 * H - h0 * W) > 0.01 * max(w0 * H, h0 * W):
        row['reason'] = f'the images ({w0} × {h0}) do not have the module\'s aspect ({W} × {H})'
        return row
    if engine is None or engine.vehicle_geom(project) is None:
        row['reason'] = 'the project has no vehicle body to simulate the mask from'
        return row
    intr = _intr(project, cam)
    ha, wa = ashape
    M, D, sigma = frame_stats([f.grey for f in frames])
    row['noise'] = sigma
    ok_m, buf = cv2.imencode('.jpg', np.clip(M, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 85])
    row['median_jpeg'] = buf.tobytes() if ok_m else b''
    static0 = D <= static_threshold(sigma, st)
    # the image circle: predicted and measured
    pc = predicted_circle(intr, ashape)
    det = detect_circle(M, static0, sigma, st.edge_contrast)
    body_plan, circle_plan, _ = sim_mask(project, cam, intr, ashape, engine)
    circ = {'predicted': {'cx': pc['cx'], 'cy': pc['cy'], 'r': pc['r']}, 'visible_predicted': pc['visible'],
            'measured': None, 'visible': det is not None and det.get('fit') is not None}
    inside = circle_plan.copy()
    if det is not None:
        inside = ~det['exterior']
        if det.get('fit') is not None:
            f = det['fit']
            circ['measured'] = {k: f[k] for k in ('cx', 'cy', 'r', 'rms_px', 'points', 'arc_deg')}
            circ['centre_error_px'] = float(math.hypot(f['cx'] - pc['cx'], f['cy'] - pc['cy']))
            circ['radius_error_px'] = float(f['r'] - pc['r'])
            circ['radius_error_pct'] = 100.0 * circ['radius_error_px'] / max(pc['r'], 1e-9)
            circ['centre_error_pct'] = 100.0 * circ['centre_error_px'] / max(pc['r'], 1e-9)
            from .lens import dr_dtheta
            drdt = float(dr_dtheta(intr.max_theta, intr.f_mm, intr.projection, intr.kb)) / intr.pixel_mm * (wa / W)
            circ['radius_error_deg'] = math.degrees(circ['radius_error_px'] / max(drdt, 1e-9))
            circ['sensor_scale'] = W / wa
    row['circle'] = circ
    # the search band around the plan's silhouette: search_deg at the image's finest scale, plus a few pixels
    ppd = intr.f_px * (wa / W) * math.pi / 180.0
    band = float(st.search_deg * ppd + 3.0)
    cl = classify(M, D, sigma, body_plan, inside, band, st)
    labels = cl['labels']
    row.update({'static_pct': cl['static_pct'], 'scenery_pct': cl['scenery_pct'],
                'undecided_pct': cl['undecided_pct'], 'threshold': cl['threshold']})
    veh = labels == VEH
    world = labels == WORLD
    decided = (labels == VEH) | (labels == WORLD)
    row['contours'] = {'real': _contours(veh), 'plan': _contours(body_plan), 'undecided': _contours(labels == UND)}
    row['_labels'] = labels                   # kept for the overlay picture (dropped from exports)
    row['iou_plan'] = _iou(veh, body_plan, decided)
    # the outlines: simulated (sub-pixel samples at the plan) and real
    body2, circle2, depth2 = sim_mask(project, cam, intr, ashape, engine, ss=2)
    vis2 = cv2.erode(circle2.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    vis2[:2, :], vis2[-2:, :], vis2[:, :2], vis2[:, -2:] = False, False, False, False
    sp2, sn2 = _pairs(body2 & vis2, ~body2 & vis2)
    real_pts, real_nrm = _pairs(veh, world)
    if len(real_pts):
        real_nrm = _smooth_normals(veh, real_pts, real_nrm)
        real_pts, moved = refine_outline(M, real_pts, real_nrm, st.edge_contrast)
        row['refined_pct'] = 100.0 * moved / len(real_pts)
    outline_px = len(sp2) / 2.0
    row['outline_px'] = outline_px
    if outline_px < MIN_OUTLINE_PX:
        row['reason'] = (f'the plan shows {outline_px:.0f} px of vehicle outline in this camera (at least '
                         f'{MIN_OUTLINE_PX} are needed): it sees too little of the vehicle to be checked this way')
        return row
    if len(real_pts) < 20:
        row['reason'] = 'no outline between the vehicle and the moving world was found in the frames'
        return row
    sp = (sp2 + 0.5) / 2.0 - 0.5                           # sub-pixel samples -> analysis pixels
    xs2 = np.clip(np.round(sp2[:, 0] - 0.5 * sn2[:, 0]).astype(int), 0, body2.shape[1] - 1)
    ys2 = np.clip(np.round(sp2[:, 1] - 0.5 * sn2[:, 1]).astype(int), 0, body2.shape[0] - 1)
    depth = depth2[ys2, xs2]
    sx, sy = wa / W, ha / H
    d_opt, _ = intr.unproject((sp[:, 0] + 0.5) / sx, (sp[:, 1] + 0.5) / sy)
    dirs = d_opt @ _W2O(cam)
    P = cam.pos() + dirs * np.where(np.isfinite(depth), depth, 5.0)[:, None]
    ol = _Outline(cam, intr, ashape, sp, dirs, P, real_pts, real_nrm)
    row['outline_plan'] = _outline_distance(ol, np.zeros(6), band)
    obs = (row['outline_plan'] or {}).get('observed_pct') or 0.0
    if obs < 100.0 * MIN_OBSERVED:
        row['reason'] = (f'only {obs:.0f} % of the planned outline has a real outline within reach (the vehicle\'s '
                         'edge against plain road or sky, or a vehicle far from the plan)')
        return row
    dtmap = cv2.distanceTransform((~_raster(real_pts, ashape)).astype(np.uint8), cv2.DIST_L2, 5)
    fit = fit_outline(ol, dtmap, st.search_deg, st.fit_translation)
    row['fit'] = _fit_doc(fit)
    if not fit.get('converged') or not np.any(fit['free'][:6]):
        row['reason'] = (fit.get('why') or 'the outline fit did not settle' if not fit.get('converged') else
                         'the outline in view constrains no axis of the rotation')
        return row
    x = fit['x']
    row['check'] = 'fitted'
    row['outline_fit'] = _outline_distance(ol, x, band, x[6])
    row['offset_px'] = float(x[6])
    body_fit, _, _ = sim_mask(project, cam, intr, ashape, engine, tuple(x[:3]), tuple(x[3:6]))
    row['iou_fit'] = _iou(veh, body_fit, decided)
    row['contours']['fitted'] = _contours(body_fit)
    # per axis: precise (constrained to a third of the tolerance … two thirds), off (precise, beyond the tolerance and
    # three standard errors), suspect (beyond the tolerance and three standard errors but not precise)
    se = np.asarray(fit['stderr'], float)[:6]
    tol = np.array([st.tol_rotation_deg] * 3 + [st.tol_translation_m] * 3)
    free = fit['free'][:6]
    precise = free & (se <= tol / 1.5)
    sig = np.abs(x[:6]) > 3.0 * se
    off = precise & (np.abs(x[:6]) > tol) & sig
    suspect = free & ~precise & (np.abs(x[:6]) > tol) & sig
    row['confident'] = [bool(v) for v in precise]
    row['beyond'] = [bool(v) for v in off]
    names = AXES + TAXES
    unit = ['°'] * 3 + [' m'] * 3
    if off.any():
        corr = np.where(precise & sig, x[:6], 0.0)
        row['correction_deg'] = [float(v) for v in corr[:3]]
        row['correction_m'] = [float(v) for v in corr[3:6]]
        row['apply'] = True
    elif suspect.any():
        row['check'] = 'uncertain'
        row['reason'] = (', '.join(f'{names[k]} {x[k]:+.2f}{unit[k]} ± {se[k]:.2f}{unit[k]}' for k in range(6)
                                   if suspect[k])
                         + ' looks beyond the tolerance but is not measured precisely enough to correct (the outline '
                           'in view is short or straight)')
    elif not precise[:3].any():
        row['check'] = 'uncertain'
        row['reason'] = (f'the outline in view pins no rotation axis to ±{st.tol_rotation_deg / 1.5:.2f}° (best '
                         + ', '.join(f'{names[k]} ± {se[k]:.2f}°' for k in range(3) if free[k]) + ')')
    return row


def _raster(pts, ashape):
    ha, wa = ashape
    m = np.zeros((ha, wa), bool)
    if len(pts):
        xi = np.clip(np.round(pts[:, 0]).astype(int), 0, wa - 1)
        yi = np.clip(np.round(pts[:, 1]).astype(int), 0, ha - 1)
        m[yi, xi] = True
    return m


def _fit_doc(fit):
    x = np.asarray(fit.get('x', np.zeros(NPAR)), float)
    se = np.asarray(fit.get('stderr', np.full(NPAR, np.inf)), float)
    fl = np.asarray(fit.get('stderr_floor', np.full(NPAR, np.inf)), float)
    free = np.asarray(fit.get('free', np.zeros(NPAR, bool)), bool)
    fin = lambda a: [float(v) if np.isfinite(v) else None for v in a]      # noqa: E731
    tr = bool(fit.get('translation'))
    return {'delta_deg': fin(x[:3]), 'stderr_deg': fin(se[:3]), 'floor_deg': fin(fl[:3]),
            'constrained': [bool(v) for v in free[:3]],
            'shift_m': fin(x[3:6]) if tr else None, 'stderr_m': fin(se[3:6]) if tr else None,
            'constrained_t': [bool(v) for v in free[3:6]] if tr else None,
            'offset_px': float(x[6]), 'stderr_offset_px': fin(se[6:7])[0],
            'points': int(fit.get('points', 0)), 'cells': int(fit.get('cells', 0)),
            'converged': bool(fit.get('converged'))}


# =========================================================================== the drive-log estimate
def _lidar_estimates(project, var, path, overrides, st, engine, progress, cancel) -> tuple[dict, str]:
    """({camera tag: the drive-log check's rotation row}, note) from core.drivelog run on the same recording when it
    has a lidar topic; ({}, why not) otherwise."""
    try:
        info = DL.open_log(path)
    except Exception as e:                      # noqa: BLE001 (a recording drivelog cannot read: say why)
        return {}, f'The drive-log check could not read the recording: {e}'
    dst = DL.DriveLogSettings.from_dict(getattr(project.analysis, 'drivelog', None))
    maps = dict(dst.overrides(path))
    maps.update(overrides or {})
    st_dl = DL.DriveLogSettings(path=norm_path(path), variant_uid=var.uid, camera_stamp=dst.camera_stamp,
                                lidar_stamp=dst.lidar_stamp, frames_per_camera=max(4, min(st.frames, 12)),
                                tolerance_deg=st.tol_rotation_deg, search_deg=min(max(st.search_deg, 0.2), 5.0),
                                sync_sweeps=40, mappings={norm_path(path): maps})
    m = DL.mapping_for(var, info, maps)
    lidars = {s.tag for s in var.sensors if s.visible and s.kind == 'lidar'}
    if not any(g in lidars for g in m.values()):
        return {}, 'No lidar of the variant has a topic in the recording: the drive-log estimate is not made.'
    try:
        res = DL.run(project, path, st_dl, progress, cancel, engine, var)
    except Exception as e:                      # noqa: BLE001 (the outline check stands on its own: say why)
        return {}, f'The drive-log estimate failed: {type(e).__name__}: {e}'
    if res is None:
        return None, ''
    return {r['tag']: r for r in res.extrinsics}, ''


def _agree(row, lid) -> bool | None:
    """Whether the outline fit and the lidar estimate agree within their errors (2 sigma combined, plus 0.05°) on the
    axes both pin down; None when they share none."""
    fit = row.get('fit') or {}
    d1, s1, c1 = fit.get('delta_deg'), fit.get('stderr_deg'), fit.get('constrained')
    d2, s2, c2 = lid.get('delta_deg'), lid.get('stderr_deg'), lid.get('observable')
    if row.get('check') != 'fitted' or d2 is None or lid.get('confidence') not in ('high', 'medium'):
        return None
    ok, n = True, 0
    for k in range(3):
        if not (c1 and c1[k] and c2 and c2[k]) or s1[k] is None or s2 is None or s2[k] is None:
            continue
        n += 1
        if abs(d1[k] - d2[k]) > 2.0 * math.hypot(s1[k], s2[k]) + 0.05:
            ok = False
    return ok if n else None


# =========================================================================== as built and coverage
def asbuilt_variant(project, res, date: str | None = None):
    """A copy of the checked variant named '<variant> · as built <date>' with every confident correction applied (the
    outline fit's, else a confident drive-log rotation): new uids, a corrected camera and its mirror partner unlinked
    (the built rig is not symmetric). None when there is nothing to apply or the variant is gone."""
    from .project import Variant, new_uid
    src = project.variant_by_uid(res.variant_uid)
    if src is None:
        return None
    rows = {r['tag']: r for r in res.cameras if r.get('apply')}
    if not rows:
        return None
    v = Variant.from_dict(src.to_dict())
    v.uid = new_uid()
    v.name = f'{src.name} · as built {date or res.date}'
    umap = {c.uid: new_uid() for c in v.cameras}
    for c in v.cameras:
        c.uid = umap[c.uid]
        c.link_uid = umap.get(c.link_uid, '')
    for s in v.sensors:
        s.uid = new_uid()
    v.seams = {'|'.join(umap.get(u, u) for u in k.split('|')): a for k, a in v.seams.items()}
    done = []
    for c in v.cameras:
        r = rows.get(c.tag)
        if r is None:
            continue
        d, t = r['correction_deg'], r.get('correction_m') or [0.0, 0.0, 0.0]
        c.yaw, c.pitch, c.roll = euler_zyx(rot_zyx(c.yaw + d[0], c.pitch + d[1], c.roll + d[2]))
        c.yaw, c.pitch, c.roll = round(c.yaw, 4), round(c.pitch, 4), round(c.roll, 4)
        c.x, c.y, c.z = round(c.x + t[0], 5), round(c.y + t[1], 5), round(c.z + t[2], 5)
        if c.link_uid:
            partner = next((x for x in v.cameras if x.uid == c.link_uid), None)
            if partner is not None:
                partner.link_uid = ''
            c.link_uid = ''
        txt = f'{c.tag} yaw {d[0]:+.2f}°, pitch {d[1]:+.2f}°, roll {d[2]:+.2f}°'
        if any(abs(x) > 0 for x in t):
            txt += f', shift {t[0] * 1000:+.0f}/{t[1] * 1000:+.0f}/{t[2] * 1000:+.0f} mm'
        done.append(txt + (' (drive-log estimate)' if r.get('source') == 'lidar' else ''))
    v.note = (f'As built, from the recording {os.path.basename(res.path)} ({res.date}): ' + '; '.join(done) + '.')
    return v


def _with_variant(project, v):
    q = copy.copy(project)
    q.variants = list(project.variants) + [v]
    q.active_uid = v.uid
    return q


COV_KEYS = (('blind_m2', 'Blind ground', 'm²', 'lower'), ('covered_pct', 'ROI covered', '%', 'higher'),
            ('multi_pct', 'Seen by two or more cameras', '%', 'higher'),
            ('min_ok_pct', 'At the minimum density', '%', 'higher'),
            ('target_ok_pct', 'At the target density', '%', 'higher'),
            ('min_density', 'Minimum density', 'px/cm', 'higher'))


def coverage_compare(project, var, asb, engine, st: AsBuiltSettings) -> dict | None:
    """The coverage analysis on the plan and on the as-built copy (the plan again when nothing was corrected) and the
    differences, each judged against the settings' tolerances; per camera its visible ground. None without a body."""
    if engine is None or engine.vehicle_geom(project) is None:
        return None
    a = engine.analyse(project, variant=var)
    if a is None:
        return None
    pa = project if asb is None else _with_variant(project, asb)
    b = a if asb is None else engine.analyse(pa, variant=asb)
    tol = {'blind_m2': st.tol_blind_m2, 'multi_pct': st.tol_multi_pct, 'min_ok_pct': st.tol_density_pct,
           'target_ok_pct': st.tol_density_pct}
    rows = []
    for key, label, unit, better in COV_KEYS:
        va, vb = float(a.metrics[key]), float(b.metrics[key])
        delta = vb - va
        worse = -delta if better == 'higher' else delta
        t = tol.get(key)
        rows.append({'key': key, 'label': label, 'unit': unit, 'plan': va, 'as_built': vb, 'delta': delta,
                     'tolerance': t, 'holds': None if t is None else bool(worse <= t + 1e-9)})
    per = {}
    by_tag_b = {c.tag: c.uid for c in (asb.cameras if asb is not None else var.cameras)}
    for c in var.cameras:
        if not c.visible:
            continue
        sa = a.per_cam.get(c.uid, {}).get('seen_m2', 0.0)
        sb = b.per_cam.get(by_tag_b.get(c.tag, ''), {}).get('seen_m2', 0.0)
        dp = 100.0 * (sb - sa) / sa if sa > 1e-9 else (0.0 if sb <= 1e-9 else math.inf)
        per[c.tag] = {'plan_m2': float(sa), 'as_built_m2': float(sb), 'delta_m2': float(sb - sa), 'delta_pct': float(dp),
                      'holds': bool(abs(dp) <= st.tol_visible_pct + 1e-9)}
    blind_a = a.grid.roi & (a.count == 0)
    blind_b = b.grid.roi & (b.count == 0)
    return {'rows': rows, 'cameras': per, 'holds': all(r['holds'] is not False for r in rows),
            'requirements': _requirements(project, var, pa, asb, engine),
            '_maps': {'roi': a.grid.roi, 'body': a.grid.body, 'blind_plan': blind_a, 'blind_as_built': blind_b,
                      'count_plan': a.count, 'count_as_built': b.count}}


def _requirements(project, var, pa, asb, engine) -> dict | None:
    """The scorecard's coverage requirements (the user's; without any, those the plan implies) on plan and as built:
    cheap, the coverage analysis is cached."""
    from . import requirements as RQ
    items = [i for i in RQ.settings_of(project).items if RQ.METRICS.get(i['metric']) is not None and
             RQ.METRICS[i['metric']].study == 'coverage']
    origin = 'scorecard'
    if not items:
        items = [i for i in RQ.from_plan(project, var) if RQ.METRICS.get(i['metric']) is not None and
                 RQ.METRICS[i['metric']].study == 'coverage']
        origin = 'plan'
    items = RQ.RequirementsSettings.from_dict({'items': items}).items       # ids and defaults as the scorecard has them
    if not items:
        return None
    out = {'origin': origin, 'items': []}
    res = {}
    for name, pj, uid in (('plan', project, var.uid), ('as_built', pa, (asb or var).uid)):
        r = RQ.run(pj, engine, RQ.RequirementsSettings(items=items, variant=uid), issues_list=False)
        res[name] = r
    for k, it in enumerate(items):
        a, b = res['plan'].items[k], res['as_built'].items[k]
        out['items'].append({'id': it['id'], 'text': RQ.describe(it, project), 'metric': it['metric'],
                             'priority': it['priority'], 'plan': a.status, 'plan_value': a.value,
                             'as_built': b.status, 'as_built_value': b.value})
    out['plan'] = res['plan'].verdict
    out['as_built'] = res['as_built'].verdict
    return out


# =========================================================================== results
@dataclass
class AsBuiltResult:
    settings: AsBuiltSettings
    path: str
    source: str
    variant: str
    variant_uid: str
    date: str                       # YYYY-MM-DD of the check
    recording: dict                 # duration_s, streams, frames, motion source
    mapping: dict                   # stream -> tag
    cameras: list                   # per visible camera (see check_camera)
    coverage: dict | None
    verdict: str                    # holds | holds_with_corrections | does_not_hold | not_checkable
    findings: list                  # [{'severity', 'message', 'tags', 'uids'}], worst first
    metrics: dict
    asbuilt: object = None          # the as-built Variant (None when nothing was corrected)
    notes: list = field(default_factory=list)
    scan: RecScan | None = None
    seconds: float = 0.0

    def camera(self, tag):
        return next((r for r in self.cameras if r['tag'] == tag), None)


def _finding(out, sev, message, tags=(), uids=()):
    out.append({'severity': sev, 'message': message, 'tags': list(tags), 'uids': list(uids)})


def _variant(project, variant, st):
    if variant is not None and not isinstance(variant, str):
        return variant
    uid = variant if isinstance(variant, str) else st.variant_uid
    return (project.variant_by_uid(uid) if uid else None) or project.variant


def run(project, engine=None, settings: AsBuiltSettings | None = None, variant=None, progress=None, cancel=None,
        path=None, scan: RecScan | None = None) -> AsBuiltResult | None:
    """Check a recording (the settings' path by default) against a variant's plan (the settings' variant, else the
    active one). A scan of the same files read with the same frame settings is used again (pass it as scan). None
    when cancelled. ValueError (a message for the user) when there is no recording, no image in it, no camera to
    check or no vehicle body."""
    t_start = time.perf_counter()
    st = settings or settings_of(project)
    path = norm_path(path or st.path)
    if not path:
        raise ValueError('No recording given: open a ROS 2 bag, a nuScenes folder or a folder of images per camera.')
    var = _variant(project, variant, st)
    if engine is None:
        from .analysis import Engine
        engine = Engine()
    if engine.vehicle_geom(project) is None:
        raise ValueError('The project has no vehicle body: the simulated masks need one (import or generate a body).')
    cams = [c for c in var.cameras if c.visible]
    if not cams:
        raise ValueError(f'{var.name} has no visible camera to check.')
    lidar_on = st.lidar_check

    def prog(lo, hi):
        if progress is None:
            return None
        return lambda f, text='': progress(lo + (hi - lo) * max(0.0, min(1.0, f)), text)
    params = (int(st.frames), float(st.min_motion_m), float(st.min_change))
    if scan is None or scan.key != recording_key(path) or scan.params != params:
        scan = scan_recording(path, *params, progress=prog(0.0, 0.55 if lidar_on else 0.7), cancel=cancel)
        if scan is None:
            return None
    if not scan.streams:
        raise ValueError(f'{os.path.basename(path)} holds no images (no Image or CompressedImage topic, no camera '
                         'channel, no image files): the as-built check needs camera frames.')
    if not any(s.count for s in scan.streams.values()):
        raise ValueError(f'{os.path.basename(path)}: its image streams hold no messages.')
    mapping = mapping_for(var, scan, st.overrides(path))
    rev = {g: t for t, g in mapping.items()}
    out, rows = [], []
    p_cam = prog(0.55 if lidar_on else 0.7, 0.75 if lidar_on else 0.95)
    for i, c in enumerate(cams):
        if cancel is not None and cancel():
            return None
        if p_cam is not None:
            p_cam(i / len(cams), f'camera {c.tag}')
        t = rev.get(c.tag)
        rows.append(check_camera(project, c, scan.streams.get(t) if t else None, st, engine))
    if cancel is not None and cancel():
        return None
    # ---- the drive-log estimate
    notes = list(scan.notes)
    lid = {}
    if lidar_on and scan.source in ('rosbag', 'nuscenes'):
        lid, why = _lidar_estimates(project, var, path, st.overrides(path), st, engine, prog(0.75, 0.95), cancel)
        if lid is None:
            return None
        if why:
            notes.append(why)
    elif lidar_on:
        notes.append('An image folder has no lidar: the drive-log estimate is not made.')
    for r in rows:
        e = lid.get(r['tag'])
        if e is not None and e.get('delta_deg') is not None:
            r['lidar'] = {'delta_deg': e['delta_deg'], 'stderr_deg': e.get('stderr_deg'),
                          'observable': e.get('observable'), 'confidence': e.get('confidence'),
                          'verdict': e.get('verdict'), 'lidar': e.get('lidar'), 'reason': e.get('reason') or '',
                          'frames': e.get('frames', 0), 'apply': bool(e.get('apply')),
                          'correction_deg': e.get('correction_deg')}
            r['agree'] = _agree(r, r['lidar'])
            if r['check'] != 'fitted' and e.get('confidence') in ('high', 'medium') and \
                    e.get('verdict') in ('ok', 'rotated'):
                r['source'] = 'lidar'                     # the outline cannot tell: the drive-log estimate stands in
                if e.get('apply') and e.get('correction_deg'):
                    r['correction_deg'] = [float(v) for v in e['correction_deg']]
                    r['apply'] = True
        elif e is not None:
            r['lidar'] = {'delta_deg': None, 'verdict': e.get('verdict'), 'reason': e.get('reason') or '',
                          'confidence': e.get('confidence')}
    # ---- as built
    date = _dt.date.today().isoformat()
    res = AsBuiltResult(st, path, scan.source, var.name, var.uid, date, {}, mapping, rows, None, 'not_checkable', out,
                        {}, None, notes, scan)
    asb = asbuilt_variant(project, res, date)
    res.asbuilt = asb
    if progress is not None:
        progress(0.96, 'coverage')
    cov = coverage_compare(project, var, asb, engine, st)
    res.coverage = cov
    _judge(res, st, out)
    order = {s: i for i, s in enumerate(SEVERITIES)}
    out.sort(key=lambda f: order.get(f['severity'], 9))
    frames = sum(len(s.frames) for s in scan.streams.values())
    res.recording = {'duration_s': scan.duration_s, 'streams': len(scan.streams), 'frames': frames,
                     'motion': scan.motion, 'source': scan.source, 'scan_seconds': scan.seconds}
    fitted = [r for r in rows if r['check'] == 'fitted']
    worst = max(fitted, key=lambda r: max(abs(v) for v in r['correction_deg']), default=None)
    res.metrics = {
        'cameras': len(rows), 'checked': len(fitted), 'not_checkable': sum(r['verdict'] == 'not_checkable' for r in rows),
        'corrected': [r['tag'] for r in rows if r.get('apply')],
        'does_not_hold': [r['tag'] for r in rows if r['verdict'] == 'does_not_hold'],
        'worst_correction_deg': (max(abs(v) for v in worst['correction_deg']) if worst is not None else None),
        'worst_correction_tag': worst['tag'] if worst is not None and worst.get('apply') else None,
        'mean_iou_plan': _mean([r['iou_plan'] for r in fitted]), 'mean_iou_fit': _mean([r['iou_fit'] for r in fitted]),
        'critical': sum(f['severity'] == 'critical' for f in out),
        'warnings': sum(f['severity'] in ('warning', 'serious') for f in out)}
    if cov is not None:
        for row in cov['rows']:
            res.metrics[f'delta_{row["key"]}'] = row['delta']
    res.seconds = time.perf_counter() - t_start
    res.metrics['seconds'] = res.seconds
    if progress is not None:
        progress(1.0, 'done')
    return res


def _mean(v):
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else None


def _judge(res: AsBuiltResult, st: AsBuiltSettings, out):
    """Per-camera and overall verdicts and the findings."""
    cov = res.coverage
    per = (cov or {}).get('cameras', {})
    for r in res.cameras:
        tag, uids = r['tag'], [r['uid']]
        c = r.get('circle') or {}
        if c.get('measured') is not None:
            e = max(abs(c.get('centre_error_pct', 0.0)), abs(c.get('radius_error_pct', 0.0)))
            if e > st.tol_circle_pct:
                _finding(out, 'warning', f'{tag}: the image circle is off the lens model by {c["centre_error_px"]:.1f} '
                                         f'px at its centre and {c["radius_error_px"]:+.1f} px in radius '
                                         f'({e:.1f} % of the radius, tolerance {st.tol_circle_pct:g} %): a decentred '
                                         'lens, another lens, or a lens model to recalibrate.', [tag], uids)
        if r['check'] != 'fitted':
            lid = r.get('lidar') or {}
            if r.get('source') == 'lidar' and r.get('apply'):        # the drive-log estimate stands in
                pc = per.get(tag)
                r['verdict'] = 'holds_with_corrections' if pc is None or pc['holds'] else 'does_not_hold'
                if pc is not None:
                    r['coverage'] = pc
                d = r['correction_deg']
                _finding(out, 'serious' if r['verdict'] != 'does_not_hold' else 'critical',
                         f'{tag}: not checkable from its outline ({r["reason"]}); the drive-log check finds it rotated '
                         f'by yaw {d[0]:+.2f}°, pitch {d[1]:+.2f}°, roll {d[2]:+.2f}° against the lidar '
                         f'({lid.get("confidence")} confidence), applied to the as-built copy.'
                         + (f' Its visible ground changes by {pc["delta_pct"]:+.1f} %.' if pc is not None else ''),
                         [tag], uids)
            elif r.get('source') == 'lidar':
                r['verdict'] = 'holds'
                _finding(out, 'good', f'{tag}: not checkable from its outline ({r["reason"]}); the drive-log check '
                                      f'finds it within {st.tol_rotation_deg:g}° of the plan against the lidar '
                                      f'({lid.get("confidence")} confidence).', [tag], uids)
            else:
                r['verdict'] = 'not_checkable'
                _finding(out, 'warning', f'{tag}: not checkable ({r["reason"]}).', [tag], uids)
            continue
        fit = r['fit']
        ofit = r.get('outline_fit') or {}
        unexplained = (ofit.get('mean_px') is not None and (ofit['mean_px'] > 2.0 * st.tol_boundary_px and
                                                             ofit['mean_deg'] > 2.0 * st.tol_boundary_deg))
        pc = per.get(tag)
        if pc is not None:
            r['coverage'] = pc
        conf = r.get('confident') or [True] * 3
        loose = [AXES[k] for k in range(3) if not fit['constrained'][k] or not conf[k]]
        if unexplained:
            r['verdict'] = 'does_not_hold'
            _finding(out, 'critical', f'{tag}: its real outline is {ofit["mean_px"]:.1f} px ({ofit["mean_deg"]:.2f}°) '
                                      'from the simulated one on average even after the best rotation: something in '
                                      'view the plan does not have (a part, a different body), or dirt on the lens.',
                     [tag], uids)
        elif r.get('apply'):
            d = r['correction_deg']
            se = fit['stderr_deg']
            axes = ', '.join(f'{AXES[k]} {d[k]:+.2f}° ± {se[k]:.2f}°' for k in range(3) if d[k])
            if r.get('correction_m') and any(r['correction_m']):
                axes += ', shift ' + '/'.join(f'{v * 1000:+.0f}' for v in r['correction_m']) + ' mm'
            holds = pc is None or pc['holds']
            r['verdict'] = 'holds_with_corrections' if holds else 'does_not_hold'
            cov_txt = (f' Its visible ground changes by {pc["delta_pct"]:+.1f} % ({pc["delta_m2"]:+.2f} m²)'
                       + (', within' if holds else ', beyond') + f' the {st.tol_visible_pct:g} % tolerance.'
                       if pc is not None else '')
            _finding(out, 'serious' if holds else 'critical',
                     f'{tag} is mounted off the plan: {axes} (tolerance {st.tol_rotation_deg:g}°)'
                     + (f'; {" and ".join(loose)} not pinned to the tolerance by the outline in view' if loose else '')
                     + '.' + cov_txt, [tag], uids)
        else:
            r['verdict'] = 'holds'
            ob = r['outline_fit'] or {}
            iou = f'IoU {r["iou_fit"]:.3f}, ' if r.get('iou_fit') is not None else ''
            _finding(out, 'good', f'{tag} matches the plan: {iou}outline '
                                  f'{(ob.get("mean_px") or 0):.2f} px ({(ob.get("mean_deg") or 0):.3f}°) on average'
                                  + (f'; {" and ".join(loose)} not pinned to the tolerance by the outline in view'
                                     if loose else '') + '.', [tag], uids)
        off = r.get('offset_px')
        if off is not None and abs(off) > st.tol_boundary_px:
            _finding(out, 'warning', f'{tag}: the real outline lies {abs(off):.1f} px '
                                     f'{"inside" if off > 0 else "outside"} the simulated one all round (blur in the '
                                     'lens, a body bigger or smaller than the mesh, or the static threshold); the pose '
                                     'fit allows for it.', [tag], uids)
        lid = r.get('lidar')
        if r.get('agree') is False:
            _finding(out, 'warning', f'{tag}: the outline fit and the drive-log check against the lidar '
                                     f'({lid.get("lidar") or "lidar"}) disagree beyond their errors: a lidar mounted '
                                     'off the plan shows as the same rotation on every camera.', [tag], uids)
    checked = [r for r in res.cameras if r['verdict'] != 'not_checkable']
    if not checked:
        res.verdict = 'not_checkable'
    elif any(r['verdict'] == 'does_not_hold' for r in checked) or (cov is not None and not cov['holds']):
        res.verdict = 'does_not_hold'
    elif any(r['verdict'] == 'holds_with_corrections' for r in checked):
        res.verdict = 'holds_with_corrections'
    else:
        res.verdict = 'holds'
    if cov is not None and res.asbuilt is not None:
        bad = [row for row in cov['rows'] if row['holds'] is False]
        if bad:
            _finding(out, 'critical', 'With the rig as built the coverage changes beyond the tolerances: '
                     + '; '.join(f'{row["label"].lower()} {row["plan"]:.2f} → {row["as_built"]:.2f} {row["unit"]}'
                                 for row in bad) + '.')
        else:
            _finding(out, 'good', 'With the corrections the plan\'s coverage figures hold: '
                     + '; '.join(f'{row["label"].lower()} {row["delta"]:+.2f} {row["unit"]}' for row in cov['rows']
                                 if row['tolerance'] is not None) + '.')
    n_nc = len(res.cameras) - len(checked)
    if checked and n_nc:
        _finding(out, 'warning', f'{n_nc} camera{"s" * (n_nc != 1)} could not be checked: the verdict covers the '
                                 f'other {len(checked)}.')


# =========================================================================== output
def _r(v, d=3):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return round(f, d) + 0.0 if math.isfinite(f) else None


def _rl(v, d=3):
    return None if v is None else [_r(x, d) for x in v]


def summary(res: AsBuiltResult) -> dict:
    """Compact figures (the assistant, the batch report header): no images, no paths beyond the recording's name."""
    m = res.metrics
    cov = res.coverage or {}
    return {
        'recording': os.path.basename(res.path), 'source': res.source, 'variant': res.variant,
        'verdict': res.verdict, 'verdict_text': VERDICT_TEXT[res.verdict],
        'duration_s': _r(res.recording.get('duration_s'), 2), 'motion_from': res.recording.get('motion'),
        'cameras': m.get('cameras'), 'checked': m.get('checked'), 'corrected': m.get('corrected'),
        'does_not_hold': m.get('does_not_hold'),
        'worst_correction_deg': _r(m.get('worst_correction_deg'), 3), 'worst_correction_camera':
            m.get('worst_correction_tag'),
        'per_camera': [{'camera': r['tag'], 'verdict': r['verdict'], 'iou': _r(r.get('iou_fit') or r.get('iou_plan'), 3),
                        'outline_mean_px': _r((r.get('outline_plan') or {}).get('mean_px'), 2),
                        'correction_ypr_deg': _rl(r['correction_deg'], 3) if r.get('apply') else None,
                        'stderr_ypr_deg': _rl((r.get('fit') or {}).get('stderr_deg'), 3),
                        'lidar_ypr_deg': _rl((r.get('lidar') or {}).get('delta_deg'), 3), 'agree': r.get('agree'),
                        'reason': r.get('reason') or None} for r in res.cameras],
        'coverage': [{'figure': row['label'], 'unit': row['unit'], 'plan': _r(row['plan'], 3),
                      'as_built': _r(row['as_built'], 3), 'delta': _r(row['delta'], 3), 'holds': row['holds']}
                     for row in cov.get('rows', [])],
        'findings': [{'severity': f['severity'], 'message': f['message']} for f in res.findings
                     if f['severity'] != 'good'],
        'assumptions': ASSUMPTIONS}


def _outline_doc(o):
    if not o:
        return None
    return {'mean_px': _r(o.get('mean_px'), 3), 'p95_px': _r(o.get('p95_px'), 3), 'mean_deg': _r(o.get('mean_deg'), 4),
            'p95_deg': _r(o.get('p95_deg'), 4), 'observed_pct': _r(o.get('observed_pct'), 1)}


def _camera_doc(r) -> dict:
    c = r.get('circle') or {}
    fit = r.get('fit') or {}
    lid = r.get('lidar') or {}
    scale = c.get('sensor_scale') or 1.0
    circ = None
    if c:
        circ = {'predicted': {k: _r(v, 2) for k, v in (c.get('predicted') or {}).items()},
                'inside_the_image': c.get('visible_predicted'),
                'measured': {k: _r(v, 3) for k, v in (c.get('measured') or {}).items()} if c.get('measured') else None,
                'centre_error_px': _r(c.get('centre_error_px'), 3), 'radius_error_px': _r(c.get('radius_error_px'), 3),
                'radius_error_pct': _r(c.get('radius_error_pct'), 3), 'radius_error_deg': _r(c.get('radius_error_deg'), 4),
                'centre_error_sensor_px': _r((c.get('centre_error_px') or 0) * scale, 2) if c.get('measured') else None}
    return {'camera': r['tag'], 'stream': r['topic'], 'verdict': r['verdict'], 'check': r['check'],
            'reason': r.get('reason') or None, 'frames': r['frames'],
            'image_size': list(r['size']) if r.get('size') else None,
            'analysis_size': [r['ashape'][1], r['ashape'][0]] if r.get('ashape') else None,
            'noise_grey_levels': _r(r.get('noise'), 3), 'static_pct': _r(r.get('static_pct'), 2),
            'static_scenery_pct': _r(r.get('scenery_pct'), 2), 'undecided_pct': _r(r.get('undecided_pct'), 2),
            'image_circle': circ, 'iou_plan': _r(r.get('iou_plan'), 4), 'iou_fitted': _r(r.get('iou_fit'), 4),
            'outline_plan': _outline_doc(r.get('outline_plan')), 'outline_fitted': _outline_doc(r.get('outline_fit')),
            'planned_outline_px': _r(r.get('outline_px'), 1),
            'fit': {'ypr_deg': _rl(fit.get('delta_deg'), 4), 'stderr_ypr_deg': _rl(fit.get('stderr_deg'), 4),
                    'constrained': fit.get('constrained'), 'shift_m': _rl(fit.get('shift_m'), 5),
                    'stderr_shift_m': _rl(fit.get('stderr_m'), 5), 'shift_constrained': fit.get('constrained_t'),
                    'outline_offset_px': _r(fit.get('offset_px'), 3),
                    'outline_points': fit.get('points'), 'cells': fit.get('cells')} if fit else None,
            'correction_ypr_deg': _rl(r['correction_deg'], 4), 'correction_shift_m': _rl(r.get('correction_m'), 5),
            'applied': bool(r.get('apply')), 'correction_source': r.get('source', 'outline') if r.get('apply') else None,
            'plan_ypr_deg': _rl(r['plan_deg'], 4), 'plan_position_m': _rl(r['plan_pos'], 4),
            'lidar_estimate': {'ypr_deg': _rl(lid.get('delta_deg'), 4), 'stderr_ypr_deg': _rl(lid.get('stderr_deg'), 4),
                               'constrained': lid.get('observable'), 'confidence': lid.get('confidence'),
                               'verdict': lid.get('verdict'), 'relative_to': lid.get('lidar'),
                               'reason': lid.get('reason') or None} if lid else None,
            'estimates_agree': r.get('agree'),
            'visible_ground': {k: _r(v, 3) if not isinstance(v, bool) else v for k, v in r['coverage'].items()}
            if r.get('coverage') else None}


def to_doc(res: AsBuiltResult, project=None) -> dict:
    """The JSON export."""
    cov = res.coverage
    return {
        'schema': SCHEMA, 'project': getattr(project, 'name', ''), 'variant': res.variant, 'date': res.date,
        'recording': {'path': res.path, 'source': res.source, **{k: _r(v, 3) if isinstance(v, float) else v
                                                                for k, v in res.recording.items()}},
        'frame': 'vehicle (ISO 8855, rear axle)',
        'units': 'angles degrees (yaw, pitch, roll intrinsic Z-Y-X, pitch positive down, corrections added to the '
                 'plan\'s), lengths metres, image positions in analysis pixels (the recorded image at most '
                 f'{ANALYSIS_W} px wide, pixel centres at integers) unless stated',
        'settings': {k: v for k, v in res.settings.to_dict().items() if k != 'mappings'},
        'assumptions': ASSUMPTIONS, 'notes': list(res.notes),
        'verdict': res.verdict, 'verdict_text': VERDICT_TEXT[res.verdict],
        'mapping': [{'stream': t, 'camera': g} for t, g in sorted(res.mapping.items())],
        'cameras': [_camera_doc(r) for r in res.cameras],
        'as_built_variant': ({'name': res.asbuilt.name, 'note': res.asbuilt.note,
                              'cameras': [{'camera': c.tag, 'position_m': [_r(c.x, 5), _r(c.y, 5), _r(c.z, 5)],
                                           'ypr_deg': [_r(c.yaw, 4), _r(c.pitch, 4), _r(c.roll, 4)]}
                                          for c in res.asbuilt.cameras if c.visible]}
                             if res.asbuilt is not None else None),
        'coverage': ({'figures': [{'figure': row['key'], 'label': row['label'], 'unit': row['unit'],
                                   'plan': _r(row['plan'], 4), 'as_built': _r(row['as_built'], 4),
                                   'delta': _r(row['delta'], 4), 'tolerance': row['tolerance'], 'holds': row['holds']}
                                  for row in cov['rows']],
                      'cameras': [{'camera': t, **{k: (_r(v, 4) if not isinstance(v, bool) else v)
                                                   for k, v in d.items()}} for t, d in cov['cameras'].items()],
                      'holds': cov['holds'], 'requirements': cov.get('requirements')} if cov else None),
        'metrics': {k: (_r(v, 4) if isinstance(v, float) else v) for k, v in res.metrics.items()},
        'findings': [{'severity': f['severity'], 'message': f['message'], 'cameras': f['tags']} for f in res.findings],
    }


CSV_COLUMNS = ('camera', 'stream', 'verdict', 'frames', 'iou_plan', 'iou_fitted', 'outline_mean_px', 'outline_p95_px',
               'outline_mean_deg', 'outline_fitted_mean_px', 'fit_yaw_deg', 'fit_pitch_deg', 'fit_roll_deg',
               'se_yaw_deg', 'se_pitch_deg', 'se_roll_deg', 'applied', 'lidar_yaw_deg', 'lidar_pitch_deg',
               'lidar_roll_deg', 'agree', 'circle_centre_error_px', 'circle_radius_error_px', 'visible_ground_plan_m2',
               'visible_ground_as_built_m2', 'reason')


def to_csv(res: AsBuiltResult) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator='\n')
    w.writerow(CSV_COLUMNS)
    for r in res.cameras:
        d = _camera_doc(r)
        f = d['fit'] or {}
        op, of = d['outline_plan'] or {}, d['outline_fitted'] or {}
        fy = f.get('ypr_deg') or [None] * 3
        se = f.get('stderr_ypr_deg') or [None] * 3
        li = (d['lidar_estimate'] or {}).get('ypr_deg') or [None] * 3
        c = d['image_circle'] or {}
        vg = d['visible_ground'] or {}
        vals = [r['tag'], r['topic'] or '', r['verdict'], r['frames'], d['iou_plan'], d['iou_fitted'],
                op.get('mean_px'), op.get('p95_px'), op.get('mean_deg'), of.get('mean_px'), *fy, *se, d['applied'],
                *li, r.get('agree'), c.get('centre_error_px'), c.get('radius_error_px'), vg.get('plan_m2'),
                vg.get('as_built_m2'), r.get('reason') or '']
        w.writerow(['' if v is None else v for v in vals])
    return buf.getvalue()


# ---------------------------------------------------------------- pictures
REAL_COLOR = '#3987e5'           # the real (measured) outline
SIM_COLOR = '#d95926'            # the simulated outline (plan or fitted)
CIRCLE_REAL = '#199e70'
CIRCLE_SIM = '#c98500'
UND_COLOR = '#898781'


def _bgr(hex_):
    return int(hex_[5:7], 16), int(hex_[3:5], 16), int(hex_[1:3], 16)


def overlay_layers(r, which='plan', size=None) -> dict:
    """What the overlay draws for one camera, scaled to a picture of `size` (default its thumbnail's): polylines of
    the real and simulated outline ('plan' or 'fitted'), the undecided stretches, and the circles (cx, cy, r)."""
    if not r.get('ashape'):
        return {'real': [], 'sim': [], 'undecided': [], 'circle_real': None, 'circle_sim': None}
    ha, wa = r['ashape']
    wt, ht = size or r['tsize']
    kx, ky = wt / wa, ht / ha

    def sc(lines):
        return [np.c_[(ln[:, 0] + 0.5) * kx - 0.5, (ln[:, 1] + 0.5) * ky - 0.5] for ln in lines]
    con = r.get('contours') or {}
    sim = con.get('fitted') if which == 'fitted' and con.get('fitted') is not None else con.get('plan', [])
    c = r.get('circle') or {}

    def circ(d):
        if not d:
            return None
        return ((d['cx'] + 0.5) * kx - 0.5, (d['cy'] + 0.5) * ky - 0.5, d['r'] * kx)
    return {'real': sc(con.get('real', [])), 'sim': sc(sim or []), 'undecided': sc(con.get('undecided', [])),
            'circle_real': circ(c.get('measured')),
            'circle_sim': circ(c.get('predicted')) if c.get('visible_predicted') or c.get('measured') else None}


def overlay_png(res: AsBuiltResult, tag) -> bytes | None:
    """The camera's frame with the real outline and the simulated one at the plan (left) and fitted (right), the image
    circles, a legend and the figures."""
    import cv2
    r = res.camera(tag)
    if r is None or not r.get('jpeg'):
        return None
    base = cv2.imdecode(np.frombuffer(r['jpeg'], np.uint8), cv2.IMREAD_COLOR)
    if base is None:
        return None
    panels = []
    for which in ('plan', 'fitted'):
        img = (base * 0.75).astype(np.uint8)
        L = overlay_layers(r, which, (base.shape[1], base.shape[0]))
        for ln in L['undecided']:
            cv2.polylines(img, [np.round(ln * 4).astype(np.int32)], True, _bgr(UND_COLOR), 1, cv2.LINE_AA, shift=2)
        for ln in L['real']:
            cv2.polylines(img, [np.round(ln * 4).astype(np.int32)], True, _bgr(REAL_COLOR), 2, cv2.LINE_AA, shift=2)
        for ln in L['sim']:
            cv2.polylines(img, [np.round(ln * 4).astype(np.int32)], True, _bgr(SIM_COLOR), 1, cv2.LINE_AA, shift=2)
        for key, col in (('circle_real', CIRCLE_REAL), ('circle_sim', CIRCLE_SIM)):
            if L[key] is not None:
                x, y, rr = L[key]
                cv2.circle(img, (int(round(x * 4)), int(round(y * 4))), int(round(rr * 4)), _bgr(col), 1, cv2.LINE_AA,
                           shift=2)
        fit = r.get('fit') or {}
        if which == 'plan':
            txt = f'{r["tag"]}  plan   IoU {r["iou_plan"]:.3f}' if r.get('iou_plan') is not None else f'{r["tag"]}  plan'
        elif r['check'] == 'fitted':
            d = fit.get('delta_deg') or [0, 0, 0]
            txt = (f'{r["tag"]}  fitted  yaw {d[0]:+.2f} pitch {d[1]:+.2f} roll {d[2]:+.2f} deg'
                   + (f'  IoU {r["iou_fit"]:.3f}' if r.get('iou_fit') is not None else ''))
        else:
            txt = f'{r["tag"]}  not fitted'
            img[:] = (img * 0.5).astype(np.uint8)
        cv2.rectangle(img, (0, 0), (img.shape[1], 22), (30, 29, 28), -1)
        put_text(img, txt, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (236, 235, 230), 1, cv2.LINE_AA)
        panels.append(img)
    out = np.concatenate([panels[0], np.full((panels[0].shape[0], 6, 3), 30, np.uint8), panels[1]], 1)
    foot = np.full((26, out.shape[1], 3), (30, 29, 28), np.uint8)
    x = 8
    for text, col in (('real outline', REAL_COLOR), ('simulated', SIM_COLOR), ('image circle measured', CIRCLE_REAL),
                      ('lens model', CIRCLE_SIM), ('undecided', UND_COLOR)):
        cv2.line(foot, (x, 13), (x + 18, 13), _bgr(col), 2, cv2.LINE_AA)
        put_text(foot, text, (x + 24, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (183, 194, 195), 1, cv2.LINE_AA)
        x += 34 + int(cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0][0])
    put_text(foot, VERDICT_TEXT[r['verdict']], (out.shape[1] - 170, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (236, 235, 230), 1, cv2.LINE_AA)
    ok, buf = cv2.imencode('.png', np.concatenate([out, foot], 0))
    return buf.tobytes() if ok else None


def coverage_png(res: AsBuiltResult, px=600) -> bytes | None:
    """The ROI from above, forward up: ground blind in both (red), blind only as built (dark red), seen only as built
    (blue), seen by one camera (light) or by two or more (darker), the body grey."""
    import cv2
    cov = res.coverage
    if not cov or '_maps' not in cov:
        return None
    m = cov['_maps']
    roi, body = m['roi'], m['body']
    img = np.empty(roi.shape + (3,), np.uint8)
    img[:] = (244, 243, 239)
    cb = m['count_as_built']
    img[roi & (cb == 1)] = (183, 211, 246)
    img[roi & (cb >= 2)] = (109, 167, 236)
    img[roi & m['blind_plan'] & m['blind_as_built']] = (208, 59, 59)
    img[roi & ~m['blind_plan'] & m['blind_as_built']] = (120, 20, 20)
    img[roi & m['blind_plan'] & ~m['blind_as_built']] = (24, 79, 149)
    img[body] = (120, 119, 114)
    img = img[::-1, ::-1].transpose(1, 0, 2)
    k = max(1, int(px / max(img.shape[:2])))
    img = np.repeat(np.repeat(img, k, 0), k, 1)
    ok, buf = cv2.imencode('.png', np.ascontiguousarray(img[..., ::-1]))
    return buf.tobytes() if ok else None


def _safe(tag):
    return re.sub(r'[^0-9A-Za-z_-]+', '_', tag)


def files(res: AsBuiltResult, project=None) -> list:
    """[(name, bytes)]: asbuilt_check.json, asbuilt_cameras.csv, asbuilt_coverage.png and an overlay PNG per camera
    with a frame (asbuilt_overlay_<tag>.png)."""
    out = [('asbuilt_check.json', (json.dumps(to_doc(res, project), indent=1, ensure_ascii=False) + '\n')
            .encode('utf-8')),
           ('asbuilt_cameras.csv', to_csv(res).encode('utf-8'))]
    cp = coverage_png(res)
    if cp:
        out.append(('asbuilt_coverage.png', cp))
    for r in res.cameras:
        data = overlay_png(res, r['tag'])
        if data:
            out.append((f'asbuilt_overlay_{_safe(r["tag"])}.png', data))
    return out


def export(res: AsBuiltResult, project, out_dir) -> list[str]:
    """Write files(res) into out_dir; returns their paths."""
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for name, data in files(res, project):
        p = os.path.join(out_dir, name)
        with open(p, 'wb') as fh:
            fh.write(data)
        paths.append(p)
    return paths


# =========================================================================== the Issues list
_LAST = {}                          # (project name, variant uid) -> (inputs key, the verdicts): the last check run


def _issue_key(project, var, st: AsBuiltSettings) -> str:
    cams = [(c.tag, round(c.x, 5), round(c.y, 5), round(c.z, 5), round(c.yaw, 4), round(c.pitch, 4),
             round(c.roll, 4), c.module_id, c.lens_id, json.dumps(c.calib, sort_keys=True, default=str), c.visible)
            for c in var.cameras]
    s = {k: v for k, v in st.to_dict().items() if k not in ('variant_uid',)}
    return json.dumps([norm_path(st.path), var.uid, cams, s], sort_keys=True, default=str)


def remember(project, res: AsBuiltResult) -> None:
    """Keep the result for the Issues list while the plan and the settings stay as they were checked."""
    var = project.variant_by_uid(res.variant_uid)
    if var is None:
        return
    slim = {'path': res.path, 'cameras': [{'tag': r['tag'], 'uid': r['uid'], 'verdict': r['verdict'],
                                           'correction_deg': list(r['correction_deg'])} for r in res.cameras]}
    _LAST[(project.name, var.uid)] = (_issue_key(project, var, res.settings), slim)     # no frames kept
    while len(_LAST) > 8:
        _LAST.pop(next(iter(_LAST)))


def issues(project, vg=None) -> list:
    """The last as-built check's problems for the active variant while its inputs are unchanged (cheap: a key
    compared): cameras whose coverage does not hold (critical), cameras mounted off (serious)."""
    from . import rules as R
    var = project.variant
    hit = _LAST.get((project.name, var.uid))
    if hit is None:
        return []
    st = settings_of(project)
    if hit[0] != _issue_key(project, var, st):
        return []
    res = hit[1]
    out = []
    for r in res['cameras']:
        if r['verdict'] == 'does_not_hold':
            out.append(R.Issue('critical', f'As built, {r["tag"]} does not hold the plan\'s coverage '
                                           f'({os.path.basename(res["path"])}).', r['tag'], [r['uid']]))
        elif r['verdict'] == 'holds_with_corrections':
            d = r['correction_deg']
            out.append(R.Issue('serious', f'As built, {r["tag"]} is mounted off the plan by yaw {d[0]:+.2f}°, pitch '
                                          f'{d[1]:+.2f}°, roll {d[2]:+.2f}°: calibrate with the as-built pose.',
                               r['tag'], [r['uid']]))
    return out
