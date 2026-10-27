"""Drive-log check: a recorded drive checked against the rig plan, closing the loop plan → build → verify.

Input: a ROS 2 bag (core.rosbag: a rosbag2 directory with .mcap or .db3 files, or one such file) or a nuScenes-format
folder (its tables and samples/sweeps; sample_data timestamps are µs). Topics (nuScenes channels) are mapped to the
variant's visible sensors by the tag in the topic name or the header frame (`/F/image_raw/compressed`, frame
`F_optical_frame` → camera F; `/L1/points` → lidar L1; `CAM_F` → camera F) and by the message type (Image and
CompressedImage → a camera, PointCloud2 → a lidar or radar, Imu → an IMU, NavSatFix → a GNSS antenna or INS). The
mapping can be edited and the edits are saved in the settings per bag path.

Reading. One streaming pass reads the bag and never holds it in memory: every message's size, log time and header stamp
(read straight from the CDR bytes, without decoding the message), every n-th lidar sweep for the phase fit, the IMU,
GNSS and odometry messages, and, at frames_per_camera instants spread over the drive, one image of every camera topic
and the point-cloud sweeps around it (their depth edges and a thinned copy of their points). That scan does not depend
on the mapping or the thresholds, so editing them checks again without reading the bag again.

Checks, each with findings ok / warning / fault:
1. Streams: per mapped sensor the message count, the measured rate (the mean interval between header stamps, drops
   left out) against
   the plan's (the module's fps, the model's rate_hz), drops (intervals longer than drop_factor × the nominal period:
   frames missing, the longest gap, the share of time missing), jitter, latency (log time − header stamp; negative
   means a clock problem), stamps going backwards, the share of the bag the stream covers, the image size against the
   module's sensor, the points per cloud against the model's points per frame; topics no sensor claims and visible
   sensors without a topic.
2. Data rate per topic (message bytes over the topic's span) against the recording budget (core.recording): for a
   camera the bits per pixel really recorded and the compression against raw pixels; the bag's GB per hour.
3. Sync against the plan (core.timing). The reference lidar's azimuth over time is fitted per sampled sweep from the
   points' own time field when present (t / time / timestamp …, relative or absolute, its unit detected), else from
   the sweep's header stamp (the first or last point is taken as the head's direction at the stamp) and the spin model.
   Point times that would lie after their message was logged count from the sweep start while the stamp marks its end:
   they are read one period earlier.
   Its phase at the sync pulses (multiples of its period on the bag clock: PTP top of second) is the measured phase
   lock, against Sensor.phase_deg. For every camera frame the exposure centre (the stamp moved by the camera_stamp
   convention, using the timing plan's readout and exposure) minus the time the sweep crossed the camera's view is its
   measured offset, modulo the camera's frame period like the plan's misalignment; the part a lidar phase error
   explains is separated from the camera's own trigger error. Within each trigger line, the stamp skew of each camera
   against the line's first camera.
4. Extrinsics (targetless): for each sampled instant the lidar points within half a rotation of the camera's exposure
   (by their own times, or by the times the spin model gives their azimuths; motion compensated with odometry when
   the log has it (nuScenes: its ego poses), rotation only with an IMU, else not), their depth edges
   (where the range jumps between neighbouring points of a ring or of neighbouring rings, the foreground side
   continuous) projected through the camera's lens model at the plan's extrinsics; the score is the edges' mean
   attraction to the image's edges (for the coarse grid a Gaussian of the distance to the nearest Canny edge, for the
   refinement the image's normalised gradient magnitude, which peaks exactly on an edge). Image edges on the own
   vehicle (its pixels by the cube map, grown by BODY_DIL) are left out, and so are the lidar edges that project there
   or within three widths of the refinement's smoothing of it, and those the vehicle hides from the camera (a mirror
   camera's own mirror hides building edges the roof lidar sees over it): an edge the image cannot show would
   otherwise pull the rotation away from the vehicle. The vehicle is the project's, with the variant's rig and
   housings, whether or not the caller passes an engine. A rotation correction of the
   camera (yaw, pitch, roll within ±search_deg; a coarse grid, then Nelder-Mead on the sharper map) maximises it over all
   sampled frames, under a weak prior towards small corrections. The standard error is the larger of a cluster-robust
   sandwich estimate (image regions as clusters) and the spread of each frame's own optimum; an axis whose score
   peak is too flat (its width from central differences of at least a pixel), or whose error is over twice the
   tolerance, is not pinned down and stays as planned. A camera is
   'rotated' when a pinned-down axis exceeds the tolerance with high or medium confidence. Images whose pixel (at the
   image centre) spans more than the tolerance — a sample dataset at a tenth of the sensor's size — give no verdict
   ('uncertain', saying why): they cannot resolve the rotation they would be judged by.
5. GNSS and IMU: the IMU's gravity magnitude and gyro bias when at rest (0.5 s windows with little rotation and
   steady acceleration), the gravity direction against the planned mounting, the fix status shares and the reported
   position error of each NavSatFix stream.

Frames: the vehicle frame (ISO 8855, rear axle). Lidar points are taken in the lidar's own frame (REP 103) and moved
into the vehicle frame with the plan's extrinsics; the lidar is the reference, so a rotation found for a camera is
relative to the lidar.
"""
from __future__ import annotations

import csv
import io
import json
import math
import os
import re
import struct
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from .mathutil import BODY_TO_OPTICAL, euler_zyx, rot_zyx
from .pngtext import put_text

SCHEMA = 'vantage.drive_log_check/1'
SEVERITIES = ('fault', 'warning', 'ok')
STAMPS = ('mid_exposure', 'trigger', 'readout_end')
STAMP_LABEL = {'mid_exposure': 'Exposure middle', 'trigger': 'Trigger', 'readout_end': 'End of readout'}
LIDAR_STAMPS = ('auto', 'start', 'end')
LIDAR_STAMP_LABEL = {'auto': 'Auto', 'start': 'Sweep start', 'end': 'Sweep end'}
ANALYSIS_W = 960            # images are checked at most this wide (px)
THUMB_W = 640               # overlay thumbnails kept per sampled frame
EDGE_MAX = 6000             # lidar depth-edge points kept per sweep
KEEP_MAX = 30000            # other points kept per sweep (occlusion test, overlay)
FIT_MAX = 20000             # points used for one sweep's azimuth fit
VIEW_MAX = 2500             # edge points per camera frame in the search
SHOW_MAX = 8000             # points per camera frame in the overlay
EDGE_JUMP_M = 0.3           # a depth edge: the range jumps by this much …
EDGE_JUMP_REL = 0.06        # … and by this share of the nearer range
IMU_MAX = 120000            # IMU messages decoded (every n-th beyond)
ODOM_MAX = 60000
TIME_FIELDS = ('t', 'time', 'timestamp', 'time_stamp', 'timeoffset', 'time_offset', 'offset_time', 'ts')
RING_FIELDS = ('ring', 'channel', 'laser_id', 'line', 'row')
MAPPINGS_KEPT = 20
PRIOR_SCALE = 2.0           # the prior on a correction: a Gaussian of PRIOR_SCALE × the search range
THETA_SHARE = 0.97          # edges beyond this share of the lens circle's half angle are left out
BODY_DIL = 7               # px (analysis): the own vehicle's pixels grown by this much hide image edges
OBSERVABLE_DEG = 1.5        # an axis whose score peak is wider than this (1 sigma) is not pinned down
LIMITS = {'frames_per_camera': (2, 200), 'tolerance_deg': (0.01, 10.0), 'search_deg': (0.2, 5.0),
          'drop_factor': (1.05, 10.0), 'sync_tol_ms': (0.05, 100.0), 'sync_sweeps': (5, 5000)}
INTS = ('frames_per_camera', 'sync_sweeps')
ASSUMPTIONS = (
    'Sync pulses are taken at multiples of the lidar period on the log\'s clock (PTP, top of the second), and lidar '
    'clouds in the lidar\'s own frame. Without a time per point, a sweep\'s header stamp is taken as the moment the '
    'head faces the cloud\'s first point (its last for a stamp at the sweep end). Camera stamps mark what the Camera '
    'stamps setting says, with the readout and exposure of the sync plan. The rotation check finds each camera\'s '
    'rotation relative to the lidar from depth edges against image edges: translation is not observable this way at '
    'road distances, the edges of round objects move with the viewpoint, and a view without depth edges (open road) '
    'gives no answer. Rolling-shutter skew is not modelled. Odometry poses are taken as the vehicle frame\'s; with only '
    'an IMU the vehicle\'s turning is compensated, not its travel.')

STAMPED = {
    'sensor_msgs/msg/Image', 'sensor_msgs/msg/CompressedImage', 'sensor_msgs/msg/PointCloud2', 'sensor_msgs/msg/Imu',
    'sensor_msgs/msg/NavSatFix', 'sensor_msgs/msg/CameraInfo', 'sensor_msgs/msg/TimeReference',
    'sensor_msgs/msg/PointCloud', 'sensor_msgs/msg/LaserScan', 'sensor_msgs/msg/MagneticField',
    'sensor_msgs/msg/FluidPressure', 'sensor_msgs/msg/Temperature', 'nav_msgs/msg/Odometry',
    'geometry_msgs/msg/TransformStamped', 'geometry_msgs/msg/PoseStamped', 'geometry_msgs/msg/TwistStamped',
    'geometry_msgs/msg/PoseWithCovarianceStamped', 'geometry_msgs/msg/TwistWithCovarianceStamped',
    'geometry_msgs/msg/Vector3Stamped', 'geometry_msgs/msg/QuaternionStamped', 'visualization_msgs/msg/Marker'}
TYPE_KIND = {'sensor_msgs/msg/Image': 'image', 'sensor_msgs/msg/CompressedImage': 'image',
             'sensor_msgs/msg/PointCloud2': 'points', 'sensor_msgs/msg/Imu': 'imu',
             'sensor_msgs/msg/NavSatFix': 'fix', 'nav_msgs/msg/Odometry': 'odom',
             'sensor_msgs/msg/CameraInfo': 'camera_info'}
SENSOR_KINDS_FOR = {'image': ('camera',), 'points': ('lidar', 'radar'), 'imu': ('imu',), 'fix': ('gnss', 'imu')}
KIND_WORD = {'camera': 'camera', 'lidar': 'lidar', 'radar': 'radar', 'imu': 'IMU', 'gnss': 'GNSS antenna'}


# =========================================================================== settings
@dataclass
class DriveLogSettings:
    path: str = ''                     # the bag or nuScenes folder last checked
    variant_uid: str = ''              # the variant to check against; '' = the active one
    camera_stamp: str = 'mid_exposure'  # what a camera's header stamp marks: mid_exposure | trigger | readout_end
    lidar_stamp: str = 'auto'          # without point times, a sweep's stamp marks its start | end (auto: bag start,
                                       # nuScenes end)
    frames_per_camera: int = 20        # sampled instants for the extrinsic check, spread over the drive
    tolerance_deg: float = 0.3         # a camera rotated by more than this is flagged
    search_deg: float = 2.0            # rotation corrections searched within ± this
    drop_factor: float = 1.5           # an interval this many times the nominal period is a drop
    sync_tol_ms: float = 2.0           # sync offsets beyond this are flagged
    sync_sweeps: int = 300             # lidar sweeps fitted for the phase lock, spread over the drive
    mappings: dict = field(default_factory=dict)   # {bag path: {topic: tag ('' = no rig sensor)}}: the edits only

    def to_dict(self):
        d = asdict(self)
        d['mappings'] = {p: dict(m) for p, m in self.mappings.items()}
        return d

    @staticmethod
    def from_dict(d: dict | None) -> 'DriveLogSettings':
        s = DriveLogSettings()
        d = d if isinstance(d, dict) else {}
        if isinstance(d.get('path'), str):
            s.path = d['path'][:4096]
        if isinstance(d.get('variant_uid'), str):
            s.variant_uid = d['variant_uid'][:64]
        if d.get('camera_stamp') in STAMPS:
            s.camera_stamp = d['camera_stamp']
        if d.get('lidar_stamp') in LIDAR_STAMPS:
            s.lidar_stamp = d['lidar_stamp']
        for k, (lo, hi) in LIMITS.items():
            if k in d:
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
                    s.mappings[p] = {t: g[:32] for t, g in m.items() if isinstance(t, str) and isinstance(g, str)}
        return s

    def overrides(self, path) -> dict:
        return dict(self.mappings.get(norm_path(path), {})) if path else {}


def norm_path(path) -> str:
    return os.path.normpath(os.path.abspath(os.path.expanduser(str(path)))) if path else ''


def stamp_offset_ms(convention: str, readout_ms: float, exposure_ms: float) -> float:
    """ms from a camera's header stamp to the middle row's exposure centre (core.timing's rolling-shutter model: rows
    read top to bottom over readout_ms after the trigger, each exposed exposure_ms, so the middle row's exposure is
    centred at trigger + readout/2 + exposure/2; the last row ends at trigger + readout + exposure)."""
    if convention == 'trigger':
        return readout_ms / 2 + exposure_ms / 2
    if convention == 'readout_end':
        return -(readout_ms / 2 + exposure_ms / 2)
    return 0.0


# =========================================================================== message types and raw CDR
def norm_type(t) -> str:
    """'sensor_msgs/Image' (ROS 1 style) → 'sensor_msgs/msg/Image'."""
    t = str(t or '').strip()
    parts = t.split('/')
    return f'{parts[0]}/msg/{parts[1]}' if len(parts) == 2 else t


def topic_kind(msg_type) -> str:
    return TYPE_KIND.get(norm_type(msg_type), 'other')


def cdr_header(raw):
    """(stamp ns, frame_id, payload offset after the header, endian) of a message whose first field is a
    std_msgs/Header, read from its CDR bytes (4-byte encapsulation header first); None when they do not hold one."""
    if raw is None or len(raw) < 16:
        return None
    e = '<' if raw[1] in (1, 3) else '>'
    try:
        sec, nsec, n = struct.unpack_from(e + 'iII', raw, 4)
    except struct.error:
        return None
    if nsec >= 1_000_000_000 or n > 4096 or 16 + n > len(raw):
        return None
    fid = bytes(raw[16:16 + n]).split(b'\0', 1)[0].decode('utf-8', 'replace')
    return sec * 1_000_000_000 + nsec, fid, 12 + n, e


def cdr_dims(raw, head):
    """(height, width) of an Image or PointCloud2 from its CDR bytes: the two uint32 right after the header."""
    if head is None:
        return None
    off = (head[2] + 3) & ~3                       # payload offset, aligned to 4
    try:
        h, w = struct.unpack_from(head[3] + 'II', raw, 4 + off)
    except struct.error:
        return None
    return int(h), int(w)


def cdr_compressed_dims(raw, head):
    """(width, height) of a CompressedImage from its CDR bytes: the JPEG or PNG header inside its data field."""
    if head is None:
        return None
    e = head[3]
    try:
        off = 4 + ((head[2] + 3) & ~3)
        n, = struct.unpack_from(e + 'I', raw, off)                  # format string
        off = 4 + ((off - 4 + 4 + n + 3) & ~3)
        m, = struct.unpack_from(e + 'I', raw, off)                  # data length
    except struct.error:
        return None
    start = off + 4
    if m <= 0 or start + min(m, 16) > len(raw):
        return None
    return jpeg_size(raw, start, start + m)


# =========================================================================== lidar clouds
def _rel_times(tv, stamp_ns: int, period_s: float = 0.1):
    """A cloud's per-point time field as seconds after its header stamp: absolute times (s, ms, µs or ns since the
    epoch, within 2 s of the stamp) or relative ones (the unit whose spread best matches a sweep). None when the field
    holds nothing usable."""
    tv = np.asarray(tv, np.float64)
    ok = np.isfinite(tv)
    if ok.sum() < 2:
        return None
    v = tv[ok]
    span = float(v.max() - v.min())
    if span <= 0:
        return None
    st = stamp_ns / 1e9
    med = float(np.median(v))
    for scale in (1.0, 1e-3, 1e-6, 1e-9):
        if abs(med * scale - st) < 2.0:
            if tv.dtype == np.float32:
                return None                              # float32 epoch seconds are useless (128 s steps)
            return np.where(ok, (tv - st / scale) * scale, np.nan)
    best, cost = None, None
    for scale in (1.0, 1e-3, 1e-6, 1e-9):
        s = span * scale
        if 1e-4 <= s <= 2.0:
            c = abs(math.log(s / max(period_s, 1e-3)))
            if cost is None or c < cost:
                best, cost = scale, c
    return None if best is None else np.where(ok, tv * best, np.nan)


def _rings_from_elevation(el):
    """Ring ids from point elevations (degrees) when the cloud says nothing: runs of occupied 0.02° bins."""
    if not len(el):
        return np.zeros(0, int)
    lo = float(el.min())
    b = np.floor((el - lo) / 0.02).astype(np.int64)
    cnt = np.bincount(b)
    occ = cnt > max(2, len(el) * 2e-5)
    lab = np.cumsum(np.r_[occ[0], occ[1:] & ~occ[:-1]])
    return lab[b]


def depth_edges(P, rho, ring, dt=None):
    """Depth-edge points of a lidar cloud (sensor frame): where the range jumps between neighbouring points of a ring
    (consecutive in azimuth) or of neighbouring rings (nearest azimuth), by more than EDGE_JUMP_M and EDGE_JUMP_REL of
    the nearer range, while the nearer point's surface continues on its own side (no grazing ground or noise). Each
    edge point lies half way between the two directions at the nearer range, so a sampled edge carries no half-step
    bias. Returns (points (K, 3), their times or None)."""
    n = len(P)
    if n < 50:
        return np.zeros((0, 3)), (None if dt is None else np.zeros(0))
    az = np.degrees(np.arctan2(P[:, 1], P[:, 0]))
    uniq, rid = np.unique(ring, return_inverse=True)
    order = np.lexsort((az, rid))
    r_s, a_s = rid[order], az[order]
    same = r_s[1:] == r_s[:-1]
    da = np.diff(a_s)
    pos = same & (da > 1e-9)
    step = float(np.median(da[pos])) if pos.any() else 0.2
    gap = 3.0 * step + 1e-6
    ok = same & (da <= gap)
    nxt = np.full(n, -1)
    prv = np.full(n, -1)
    i, j = order[:-1][ok], order[1:][ok]
    nxt[i] = j
    prv[j] = i
    # close each ring that goes all round
    first = np.r_[0, np.nonzero(~same)[0] + 1]
    last = np.r_[np.nonzero(~same)[0], n - 1]
    wrap = (a_s[first] + 360.0 - a_s[last]) <= gap
    nxt[order[last[wrap]]] = order[first[wrap]]
    prv[order[first[wrap]]] = order[last[wrap]]
    # neighbouring rings by mean elevation
    el = np.degrees(np.arcsin(np.clip(P[:, 2] / np.maximum(rho, 1e-9), -1, 1)))
    mel = np.bincount(rid, weights=el) / np.maximum(np.bincount(rid), 1)
    rorder = np.argsort(mel)
    up = np.full(n, -1)
    dn = np.full(n, -1)
    starts = {int(r_s[f]): (f, l + 1) for f, l in zip(first, last)}
    spacing = np.diff(mel[rorder])
    sp_med = float(np.median(spacing)) if len(spacing) else 1.0
    for ra, rb, d_el in zip(rorder[:-1], rorder[1:], spacing):
        if d_el > 4 * max(sp_med, 0.05) or ra not in starts or rb not in starts:
            continue
        fa, la = starts[int(ra)]
        fb, lb = starts[int(rb)]
        ia, ib = order[fa:la], order[fb:lb]
        for src, dst, link in ((ia, ib, up), (ib, ia, dn)):
            a_src, a_dst = az[src], az[dst]
            k = np.searchsorted(a_dst, a_src)
            k0 = np.clip(k - 1, 0, len(dst) - 1)
            k1 = np.clip(k, 0, len(dst) - 1)
            d0 = np.abs(a_src - a_dst[k0])
            d1 = np.abs(a_dst[k1] - a_src)
            kk = np.where(d0 <= d1, k0, k1)
            dd = np.minimum(d0, d1)
            m = dd <= gap
            link[src[m]] = dst[kk[m]]
    U = P / np.maximum(rho, 1e-9)[:, None]
    out_p, out_t = [], []
    for a_, b_, before, after in ((np.arange(n), nxt, prv, nxt), (np.arange(n), up, dn, up)):
        a = a_[b_ >= 0]
        b = b_[a]
        ra, rb = rho[a], rho[b]
        near_a = ra < rb
        fg = np.where(near_a, a, b)
        bg = np.where(near_a, b, a)
        other = np.where(near_a, before[a], after[b])
        near = np.minimum(ra, rb)
        jump = np.abs(ra - rb)
        sel = (jump > np.maximum(EDGE_JUMP_M, EDGE_JUMP_REL * near)) & (other >= 0) & (near > 1.0) & (near < 80.0)
        sel &= np.abs(rho[fg] - rho[np.maximum(other, 0)]) < jump / 3.0
        if not sel.any():
            continue
        f, g = fg[sel], bg[sel]
        d = U[f] + U[g]
        d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
        out_p.append(d * rho[f][:, None])
        if dt is not None:
            out_t.append(dt[f])
    if not out_p:
        return np.zeros((0, 3)), (None if dt is None else np.zeros(0))
    E = np.concatenate(out_p)
    return E, (np.concatenate(out_t) if dt is not None else None)


def fit_sweep(az, dt):
    """Linear fit of a sweep's head azimuth over time from its points: az (deg, sensor frame) and dt (s after the
    stamp). Time-binned circular means give a robust start, then all points refine it. {'tm' (s), 'a' (deg at tm,
    unwrapped), 'b' (deg/s), 'rms', 'span_deg', 'n'} or None when the points do not make a turning sweep."""
    ok = np.isfinite(az) & np.isfinite(dt)
    az, dt = az[ok], dt[ok]
    if len(dt) < 200:
        return None
    if len(dt) > FIT_MAX:
        k = np.linspace(0, len(dt) - 1, FIT_MAX).astype(int)
        az, dt = az[k], dt[k]
    t0, t1 = float(dt.min()), float(dt.max())
    if t1 - t0 <= 1e-6:
        return None
    nb = 120
    bi = np.minimum(((dt - t0) / (t1 - t0) * nb).astype(int), nb - 1)
    c = np.bincount(bi, weights=np.cos(np.radians(az)), minlength=nb)
    s = np.bincount(bi, weights=np.sin(np.radians(az)), minlength=nb)
    cnt = np.bincount(bi, minlength=nb)
    tb = np.bincount(bi, weights=dt, minlength=nb) / np.maximum(cnt, 1)
    use = (cnt >= 3) & (np.hypot(c, s) > 0.5 * np.maximum(cnt, 1))
    if use.sum() < 8:
        return None
    ab = np.degrees(np.unwrap(np.arctan2(s[use], c[use])))
    tm = (t0 + t1) / 2
    b0, a0 = np.polyfit(tb[use] - tm, ab, 1)
    for _ in range(2):
        pred = a0 + b0 * (dt - tm)
        res = (az - pred + 180.0) % 360.0 - 180.0
        mad = float(np.median(np.abs(res - np.median(res))))
        keep = np.abs(res) <= max(4.0 * 1.4826 * mad, 0.5)
        if keep.sum() < 100:
            return None
        A = np.c_[np.ones(int(keep.sum())), dt[keep] - tm]
        y = pred[keep] + res[keep]
        (a0, b0), *_ = np.linalg.lstsq(A, y, rcond=None)
    res = (az - (a0 + b0 * (dt - tm)) + 180.0) % 360.0 - 180.0
    span = abs(b0) * (t1 - t0)
    if span < 270.0:
        return None
    return {'tm': float(tm), 'a': float(a0), 'b': float(b0), 'rms': float(np.sqrt(np.mean(res[keep] ** 2))),
            'span_deg': float(span), 'n': int(keep.sum()), 't0': t0, 't1': t1}


def _cloud_xyz(arr):
    names = arr.dtype.names or ()
    if not all(k in names for k in ('x', 'y', 'z')):
        return None
    return np.stack([arr['x'].astype(np.float64), arr['y'].astype(np.float64), arr['z'].astype(np.float64)], 1)


def process_cloud(arr, stamp_ns: int, height: int = 1, width: int = 0, period_s: float = 0.1, edges=True) -> dict:
    """What the checks keep of one decoded cloud (a structured array of its points): counts, the per-point times (s
    after the stamp) when it has a time field, the sweep's azimuth fit, its depth edges and a thinned copy of its
    points (both sensor frame, float32)."""
    if arr.ndim > 1:                           # an organised cloud (rows × columns)
        height, width = arr.shape[0], arr.shape[1]
        arr = arr.reshape(-1)
    P = _cloud_xyz(arr)
    out = {'stamp_ns': int(stamp_ns), 'n': int(len(arr)), 'n_valid': 0, 'has_time': False, 'fit': None,
           'az_first': None, 'az_last': None, 'E': np.zeros((0, 3), np.float32), 'E_dt': None,
           'K': np.zeros((0, 3), np.float32), 'K_dt': None}
    if P is None or not len(P):
        return out
    names = arr.dtype.names
    rho = np.linalg.norm(P, axis=1)
    valid = np.isfinite(rho) & (rho > 0.05)
    idx = np.nonzero(valid)[0]
    out['n_valid'] = int(len(idx))
    if not len(idx):
        return out
    P, rho = P[idx], rho[idx]
    tf = next((f for f in TIME_FIELDS if f in names), None)
    dt = None
    if tf is not None:
        dt = _rel_times(arr[tf][idx], stamp_ns, period_s)
        if dt is not None and not np.isfinite(dt).all():
            dt = np.where(np.isfinite(dt), dt, np.nanmedian(dt))
    out['has_time'] = dt is not None
    out['dt_max'] = float(np.max(dt)) if dt is not None and len(dt) else None
    az = np.degrees(np.arctan2(P[:, 1], P[:, 0]))
    xy = np.hypot(P[:, 0], P[:, 1]) > 1.0
    # the first and last fired column: an organised cloud's first / last column, else the first / last few points
    # (a dense cloud is in firing order, one column after the other)
    cand = np.nonzero(xy)[0] if xy.any() else np.arange(len(idx))
    if height > 1 and width > 0:
        col = idx[cand] % width
        heads = (cand[col == col.min()], cand[col == col.max()])
    else:
        heads = (cand[:3], cand[-3:])
    for key, sel in zip(('az_first', 'az_last'), heads):
        a = np.radians(az[sel])
        out[key] = float(np.degrees(np.arctan2(np.sin(a).mean(), np.cos(a).mean())))
    if dt is not None:
        out['fit'] = fit_sweep(az[xy], dt[xy])
    if edges:
        rf = next((f for f in RING_FIELDS if f in names), None)
        if rf is not None:
            ring = arr[rf][idx].astype(np.int64)
        elif height > 1 and width > 0:
            ring = idx // width
        else:
            ring = _rings_from_elevation(np.degrees(np.arcsin(np.clip(P[:, 2] / rho, -1, 1))))
        E, E_dt = depth_edges(P, rho, ring, dt)
        if len(E) > EDGE_MAX:
            k = np.linspace(0, len(E) - 1, EDGE_MAX).astype(int)
            E, E_dt = E[k], (E_dt[k] if E_dt is not None else None)
        out['E'] = E.astype(np.float32)
        out['E_dt'] = None if E_dt is None else E_dt.astype(np.float32)
        k = np.linspace(0, len(P) - 1, min(len(P), KEEP_MAX)).astype(int)
        out['K'] = P[k].astype(np.float32)
        out['K_dt'] = None if dt is None else dt[k].astype(np.float32)
    return out


def sweep_phase(sw: dict, rate_hz: float, sense: int, stamp_ref: str = 'start', use_time=True):
    """(phase deg in [0, 360), source) of one processed sweep: the sensor-frame azimuth the head faces at the sync
    pulse nearest the sweep (multiples of the period on the bag clock). From the azimuth fit of the points' own times
    when there is one (and use_time), else from the stamp: the head faces the first point (start stamp) or the last
    (end stamp) at the stamp, turning at rate_hz in the sense given (+1 counter-clockwise)."""
    T_ns = 1e9 / rate_hz
    if use_time and sw.get('fit') is not None:
        f = sw['fit']
        t_m = sw['stamp_ns'] + f['tm'] * 1e9
        k = round(t_m / T_ns)
        d = (k * T_ns - sw['stamp_ns']) / 1e9 - f['tm']
        return (f['a'] + f['b'] * d) % 360.0, 'point times'
    az0 = sw['az_first'] if stamp_ref == 'start' else sw['az_last']
    if az0 is None:
        return None, ''
    k = round(sw['stamp_ns'] / T_ns)
    d = (k * T_ns - sw['stamp_ns']) / 1e9
    return (az0 + sense * 360.0 * rate_hz * d) % 360.0, 'stamps'


def _wrap(x, c):
    """x folded into [-c/2, c/2)."""
    return (np.asarray(x, float) + c / 2) % c - c / 2


def _circ_mean(deg):
    a = np.radians(np.asarray(deg, float))
    return float(np.degrees(np.arctan2(np.sin(a).mean(), np.cos(a).mean())))


def _circ_median(x, c):
    """Median of values that wrap with period c, centred on their circular mean first."""
    x = np.asarray(x, float)
    if not len(x):
        return None
    mu = _circ_mean(x * 360.0 / c) * c / 360.0
    return float(mu + np.median(_wrap(x - mu, c)))


# =========================================================================== images
def image_sample(img, stamp_ns: int, log_ns: int, encoding: str = '') -> dict:
    """What the checks keep of one camera image: its size, its grey image at the analysis size (at most ANALYSIS_W
    wide, JPEG quality 95) for the edge maps, and a colour JPEG thumbnail for the overlay."""
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
        hi = float(np.percentile(f, 99.5)) or 1.0
        a = np.clip(f / hi * 255.0, 0, 255).astype(np.uint8)
    if a.ndim == 3 and a.shape[2] == 4:
        a = a[..., :3]
    if a.ndim == 3 and a.shape[2] == 2:                        # yuv422: the luma bytes
        a = a[..., 1] if enc.startswith('uyvy') or enc == 'yuv422' else a[..., 0]
    h, w = a.shape[:2]
    rgb = a if a.ndim == 3 else np.repeat(a[..., None], 3, 2)
    grey = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2GRAY)
    s = min(1.0, ANALYSIS_W / w)
    wa, ha = max(8, int(round(w * s))), max(8, int(round(h * s)))
    g = cv2.resize(grey, (wa, ha), interpolation=cv2.INTER_AREA) if (wa, ha) != (w, h) else grey
    ok1, gbuf = cv2.imencode('.jpg', g, [cv2.IMWRITE_JPEG_QUALITY, 95])
    ts = min(1.0, THUMB_W / w)
    wt, ht = max(8, int(round(w * ts))), max(8, int(round(h * ts)))
    th = cv2.resize(rgb, (wt, ht), interpolation=cv2.INTER_AREA) if (wt, ht) != (w, h) else rgb
    ok2, buf = cv2.imencode('.jpg', np.ascontiguousarray(th[..., ::-1]), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return {'stamp_ns': int(stamp_ns), 'log_ns': int(log_ns), 'size': (int(w), int(h)), 'ashape': (ha, wa),
            'grey': gbuf.tobytes() if ok1 else b'', 'jpeg': buf.tobytes() if ok2 else b'', 'tsize': (wt, ht)}


def jpeg_size(data, start: int = 0, end: int | None = None) -> tuple | None:
    """(width, height) from a JPEG's SOF marker (or a PNG's IHDR) in data[start:end], read in place (no copy)."""
    end = len(data) if end is None else min(end, len(data))
    if end - start >= 24 and bytes(data[start:start + 8]) == b'\x89PNG\r\n\x1a\n':
        w, h = struct.unpack_from('>II', data, start + 16)
        return int(w), int(h)
    if end - start < 4 or data[start] != 0xFF or data[start + 1] != 0xD8:
        return None
    i = start + 2
    sof = (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF)
    while i + 9 <= end:
        if data[i] != 0xFF:
            i += 1
            continue
        m = data[i + 1]
        if m == 0xFF:
            i += 1
            continue
        if m in (0xD8, 0x01) or 0xD0 <= m <= 0xD7:
            i += 2
            continue
        L, = struct.unpack_from('>H', data, i + 2)
        if m in sof:
            h, w = struct.unpack_from('>HH', data, i + 5)
            return int(w), int(h)
        if m == 0xDA:                                  # start of scan without a frame header: give up
            return None
        i += 2 + L
    return None


# =========================================================================== the scan
@dataclass
class TopicScan:
    topic: str
    type: str
    kind: str
    count: int = 0
    bytes: int = 0
    log_ns: np.ndarray | None = None
    stamp_ns: np.ndarray | None = None      # header stamps, log order (None: the messages carry none)
    frame_id: str = ''
    dims: dict = field(default_factory=dict)     # images: (w, h) -> messages seen at that size
    cloud_n: np.ndarray | None = None       # clouds: points per message (height × width)
    fields: tuple = ()
    sweeps: list = field(default_factory=list)   # clouds: processed sampled sweeps (no points kept)
    data: dict = field(default_factory=dict)     # imu / fix / odom: decoded arrays
    stamp_ref: str = 'start'                # a cloud's stamp marks its sweep start (bags) or end (nuScenes)
    info: dict = field(default_factory=dict)     # camera_info: the first message's size and model


@dataclass
class Scan:
    path: str
    source: str                             # 'rosbag' | 'nuscenes'
    key: tuple
    start_ns: int
    end_ns: int
    topics: dict                            # topic -> TopicScan
    targets: list                           # [{'t_ns', 'images': {topic: sample}, 'clouds': {topic: [sweep]}}]
    n_frames: int
    sync_sweeps: int
    seconds: float = 0.0
    notes: list = field(default_factory=list)

    @property
    def duration_s(self) -> float:
        return max(0.0, (self.end_ns - self.start_ns) / 1e9)


def log_key(path) -> tuple:
    """What a scan was made from: the path and the size and modification time of every file in it."""
    p = norm_path(path)
    files = []
    if os.path.isdir(p):
        for root, _, names in os.walk(p):
            for n in sorted(names):
                if n.endswith(('.mcap', '.db3', '.yaml', '.json')):
                    q = os.path.join(root, n)
                    try:
                        st = os.stat(q)
                        files.append((os.path.relpath(q, p), st.st_size, int(st.st_mtime)))
                    except OSError:
                        pass
    elif os.path.exists(p):
        st = os.stat(p)
        files.append((os.path.basename(p), st.st_size, int(st.st_mtime)))
    return p, tuple(files)


def is_nuscenes(path) -> bool:
    return _nus_dirs(path) is not None


def _nus_dirs(path):
    """(version folder with the tables, dataroot) of a nuScenes folder given as either; None when it is not one."""
    p = norm_path(path)
    if not os.path.isdir(p):
        return None
    if os.path.isfile(os.path.join(p, 'sample_data.json')):
        return p, os.path.dirname(p)
    for n in sorted(os.listdir(p)):
        q = os.path.join(p, n)
        if os.path.isdir(q) and os.path.isfile(os.path.join(q, 'sample_data.json')):
            return q, p
    return None


@dataclass
class LogInfo:
    path: str
    source: str
    topics: dict                # topic -> type
    counts: dict
    start_ns: int
    end_ns: int


def open_log(path) -> LogInfo:
    """The topics and their message counts of a bag or nuScenes folder, read cheaply (no messages)."""
    p = norm_path(path)
    if not os.path.exists(p):
        raise FileNotFoundError(f'Not found: {p}')
    nd = _nus_dirs(p)
    if nd is not None:
        tables = _nus_tables(nd[0], ('sensor', 'calibrated_sensor', 'sample_data'))
        chans = _nus_channels(tables)
        topics, counts, ts = {}, {}, []
        for sd in tables['sample_data']:
            ch = chans.get(sd.get('calibrated_sensor_token'))
            if ch is None:
                continue
            topics[ch[0]] = NUS_TYPE.get(ch[1], 'other')
            counts[ch[0]] = counts.get(ch[0], 0) + 1
            ts.append(int(sd.get('timestamp', 0)))
        return LogInfo(p, 'nuscenes', topics, counts, min(ts, default=0) * 1000, max(ts, default=0) * 1000)
    from . import rosbag as RBG
    rd = RBG.BagReader(p)
    try:
        return LogInfo(p, 'rosbag', {t: norm_type(ty) for t, ty in rd.topics.items()}, dict(rd.counts),
                       int(rd.start_ns), int(rd.end_ns))
    finally:
        rd.close()


def scan_log(path, n_frames: int = 20, sync_sweeps: int = 300, progress=None, cancel=None) -> Scan | None:
    """Read a bag or nuScenes folder once (see the module docstring); None when cancelled."""
    t0 = time.perf_counter()
    nd = _nus_dirs(path)
    sc = (_scan_nuscenes(nd, n_frames, sync_sweeps, progress, cancel) if nd is not None else
          _scan_bag(norm_path(path), n_frames, sync_sweeps, progress, cancel))
    if sc is not None:
        sc.seconds = time.perf_counter() - t0
        sc.path, sc.key = norm_path(path), log_key(path)       # as given (a nuScenes dataroot or its version folder)
    return sc


def _targets(t0, t1, n):
    return [int(t0 + (t1 - t0) * (k + 0.5) / n) for k in range(n)]


def _decode_small(kind, msg):
    """One IMU, NavSatFix or Odometry message as a flat tuple of floats."""
    if kind == 'imu':
        w, a = msg['angular_velocity'], msg['linear_acceleration']
        return (w['x'], w['y'], w['z'], a['x'], a['y'], a['z'])
    if kind == 'fix':
        cov = list(np.asarray(msg.get('position_covariance', [0.0] * 9), float).ravel()[:9]) + [0.0] * 9
        return (msg['status']['status'], msg['latitude'], msg['longitude'], msg['altitude'],
                cov[0], cov[4], cov[8], msg.get('position_covariance_type', 0))
    if kind == 'odom':
        p, q = msg['pose']['pose']['position'], msg['pose']['pose']['orientation']
        v, w = msg['twist']['twist']['linear'], msg['twist']['twist']['angular']
        return (p['x'], p['y'], p['z'], q['x'], q['y'], q['z'], q['w'], v['x'], v['y'], v['z'], w['x'], w['y'],
                w['z'])
    return ()


SMALL_COLS = {'imu': ('gx', 'gy', 'gz', 'ax', 'ay', 'az'),
              'fix': ('status', 'lat', 'lon', 'alt', 'cov_e', 'cov_n', 'cov_u', 'cov_type'),
              'odom': ('x', 'y', 'z', 'qx', 'qy', 'qz', 'qw', 'vx', 'vy', 'vz', 'wx', 'wy', 'wz')}


def _scan_bag(path, n_frames, sync_sweeps, progress, cancel) -> Scan | None:
    from . import rosbag as RBG
    rd = RBG.BagReader(path)
    try:
        types = {t: norm_type(ty) for t, ty in rd.topics.items()}
        counts = dict(rd.counts)
        t0, t1 = int(rd.start_ns), int(rd.end_ns)
        tops = {t: TopicScan(t, ty, topic_kind(ty)) for t, ty in types.items()}
        acc = {t: {'log': [], 'stamp': [], 'n': [], 'small': [], 'st': []} for t in types}
        targets = [{'t_ns': t, 'images': {}, 'clouds': {}} for t in _targets(t0, t1, n_frames)]
        tt = np.array([t['t_ns'] for t in targets], np.int64)
        img_next = {t: 0 for t, ts in tops.items() if ts.kind == 'image'}
        cloud = {t: {'next': 0, 'prev': [], 'windows': [], 'cache': {}, 'i': 0}
                 for t, ts in tops.items() if ts.kind == 'points'}
        stride = {t: max(1, counts.get(t, 0) // max(sync_sweeps, 1)) for t in cloud}
        small_stride = {t: max(1, counts.get(t, 0) // (IMU_MAX if ts.kind == 'imu' else ODOM_MAX))
                        for t, ts in tops.items() if ts.kind in ('imu', 'fix', 'odom')}
        seen = {}                                        # topic of another type -> whether it carries a header
        bad = {}                                         # topic -> messages that would not decode
        unsized = {}                                     # image topic -> messages whose size was not readable
        span = max(t1 - t0, 1)

        def dec(topic, raw):
            """A message decoded with the bag's own definition; None (counted) when it does not decode."""
            try:
                msg = rd.decode(topic, raw)
            except ValueError:
                msg = None
            if not isinstance(msg, dict):
                bad[topic] = bad.get(topic, 0) + 1
                return None
            return msg
        n_all = 0

        def cloud_sweep(topic, raw, stamp, k_msg, edges, log):
            st_ = cloud[topic]
            if edges and k_msg in st_['cache']:
                return st_['cache'][k_msg]
            msg = dec(topic, raw)
            if msg is None:
                return None
            try:
                arr = RBG.pointcloud2_array(msg)
            except (ValueError, KeyError, TypeError):
                bad[topic] = bad.get(topic, 0) + 1
                return None
            if not tops[topic].fields:
                tops[topic].fields = tuple(arr.dtype.names or ())
            per = _median_period(acc[topic]['stamp'] or acc[topic]['log']) or 0.1
            sw = process_cloud(arr, stamp, int(msg.get('height', 1)), int(msg.get('width', 0)), per, edges)
            sw['log_ns'] = int(log)
            if edges:
                st_['cache'][k_msg] = sw
            return sw

        def close_window(topic, w):
            k, msgs = w
            sws = []
            for (k_msg, stamp, raw, log) in msgs:
                sw = cloud_sweep(topic, raw, stamp, k_msg, True, log)
                if sw is not None:
                    sws.append(sw)
            targets[k]['clouds'][topic] = sws

        for topic, t_ns, raw in rd.messages(decode=False):
            n_all += 1
            if n_all % 256 == 0:
                if cancel is not None and cancel():
                    return None
                if progress is not None:
                    progress(min(max((t_ns - t0) / span, 0.0), 1.0), 'reading the bag')
            ts = tops[topic]
            a = acc[topic]
            raw = bytes(raw) if not isinstance(raw, (bytes, bytearray)) else raw
            ts.count += 1
            ts.bytes += len(raw)
            a['log'].append(int(t_ns))
            if ts.type in STAMPED:
                head = cdr_header(raw)
            else:                                        # another type: a header only if the first one's is plausible
                if topic not in seen:
                    h = cdr_header(raw)
                    seen[topic] = bool(h is not None and abs(h[0] - t_ns) < 10 * 86400 * 10 ** 9
                                       and h[1].isprintable())
                head = cdr_header(raw) if seen[topic] else None
            stamp = head[0] if head is not None else int(t_ns)
            if head is not None:
                a['stamp'].append(head[0])
                if not ts.frame_id:
                    ts.frame_id = head[1]
            k_msg = ts.count - 1
            if ts.kind == 'image':
                if ts.type == 'sensor_msgs/msg/Image':
                    hw = cdr_dims(raw, head)
                    wh = (hw[1], hw[0]) if hw is not None else None
                else:
                    wh = cdr_compressed_dims(raw, head)
                if wh is not None:
                    ts.dims[wh] = ts.dims.get(wh, 0) + 1
                else:
                    unsized[topic] = unsized.get(topic, 0) + 1
                k = img_next[topic]
                if (k < len(targets) and t_ns >= tt[k]) or ts.count == 1:
                    msg = dec(topic, raw)
                    if msg is not None:
                        try:
                            img = RBG.image_array(msg, to_rgb=True)
                        except (ValueError, KeyError, TypeError):
                            img = None
                            bad[topic] = bad.get(topic, 0) + 1
                        if img is not None and getattr(img, 'size', 0):
                            smp = image_sample(img, stamp, t_ns, msg.get('encoding', ''))
                            if unsized.get(topic) and smp['size'] not in ts.dims:
                                ts.dims[smp['size']] = unsized[topic]           # its size from the decoded image
                            if k < len(targets) and t_ns >= tt[k]:
                                targets[k]['images'][topic] = smp
                    while img_next[topic] < len(targets) and tt[img_next[topic]] <= t_ns:
                        img_next[topic] += 1
            elif ts.kind == 'points':
                hw = cdr_dims(raw, head)
                a['n'].append(hw[0] * hw[1] if hw is not None else -1)
                cs = cloud[topic]
                if k_msg % stride[topic] == 0:
                    sw = cloud_sweep(topic, raw, stamp, k_msg, False, t_ns)
                    if sw is not None:
                        ts.sweeps.append({k_: v for k_, v in sw.items() if k_ not in ('E', 'E_dt', 'K', 'K_dt')})
                item = (k_msg, stamp, raw, t_ns)
                for w in cs['windows']:
                    w[1].append(item)
                while cs['next'] < len(targets) and tt[cs['next']] <= t_ns:
                    cs['windows'].append((cs['next'], list(cs['prev']) + [item]))
                    cs['next'] += 1
                done = [w for w in cs['windows'] if len(w[1]) >= 4]
                for w in done:
                    close_window(topic, w)
                    cs['windows'].remove(w)
                cs['prev'] = (cs['prev'] + [item])[-2:]
                live = {m[0] for w in cs['windows'] for m in w[1]} | {m[0] for m in cs['prev']}
                for kk in [kk for kk in cs['cache'] if kk not in live]:
                    del cs['cache'][kk]
            elif ts.kind in ('imu', 'fix', 'odom'):
                if k_msg % small_stride[topic] == 0:
                    msg = dec(topic, raw)
                    if msg is not None:
                        try:
                            a['small'].append(_decode_small(ts.kind, msg))
                            a['st'].append(stamp)
                        except (KeyError, TypeError, ValueError):
                            pass
            elif ts.kind == 'camera_info' and not ts.info:
                msg = dec(topic, raw)
                if msg is not None:
                    ts.info = {'width': int(msg.get('width', 0)), 'height': int(msg.get('height', 0)),
                               'distortion_model': str(msg.get('distortion_model', '')),
                               'k': [float(v) for v in np.asarray(msg.get('k', [0] * 9), float).ravel()[:9]]}
        for topic, cs in cloud.items():
            for w in cs['windows']:
                close_window(topic, w)
            cs['windows'] = []
            cs['cache'] = {}
        _finish_topics(tops, acc)
        notes = [f'{t}: {n} message{"s" * (n != 1)} would not decode and were skipped.' for t, n in sorted(bad.items())]
        for t, ts in tops.items():
            if ts.kind == 'points':
                note = _time_origin(ts, [tg['clouds'].get(t, []) for tg in targets])
                if note:
                    notes.append(note)
        notes += [f'Bag: {w}' for w in getattr(rd, 'warnings', [])[:20]]
        for t, n in bad.items():
            tops[t].info['decode_errors'] = n
        return Scan(path, 'rosbag', log_key(path), t0, t1, tops, targets, n_frames, sync_sweeps, notes=notes)
    finally:
        rd.close()


def _time_origin(ts, window_sweeps) -> str:
    """Point times that count from the sweep start while the stamp marks its end (relative to the stamp they would
    lie after the message was logged, which no measurement can): moved back by one period. Returns a note when so."""
    sws = [s for s in ts.sweeps if s.get('dt_max') is not None and s.get('log_ns')]
    stamps = ts.stamp_ns if ts.stamp_ns is not None else ts.log_ns
    per = interval_stats(stamps)['median_ms'] if stamps is not None and len(stamps) > 2 else None
    if not sws or not per:
        return ''
    T = per / 1000.0
    excess = [(s['stamp_ns'] + s['dt_max'] * 1e9 - s['log_ns']) / 1e9 for s in sws]
    if float(np.median(excess)) <= 0.5 * T:
        return ''
    ts.info['time_shift_s'] = -T
    seen = set()
    for s in list(ts.sweeps) + [s for lst in window_sweeps for s in lst]:
        if id(s) in seen:
            continue
        seen.add(id(s))
        if s.get('fit') is not None:
            s['fit'] = dict(s['fit'], tm=s['fit']['tm'] - T, t0=s['fit']['t0'] - T, t1=s['fit']['t1'] - T)
        for k in ('E_dt', 'K_dt'):
            if s.get(k) is not None:
                s[k] = (s[k] - np.float32(T)).astype(np.float32)
        if s.get('dt_max') is not None:
            s['dt_max'] -= T
    return (f'{ts.topic}: the points\' times count from the sweep start while the stamp marks its end (they would '
            f'lie after the message was logged): read as {T * 1000:.0f} ms earlier.')


def _median_period(ns_list):
    if len(ns_list) < 3:
        return None
    d = np.diff(np.asarray(ns_list[-50:], np.int64)) / 1e9
    d = d[d > 0]
    return float(np.median(d)) if len(d) else None


def _finish_topics(tops, acc):
    for t, ts in tops.items():
        a = acc[t]
        ts.log_ns = np.asarray(a['log'], np.int64)
        ts.stamp_ns = np.asarray(a['stamp'], np.int64) if len(a['stamp']) == len(a['log']) and a['stamp'] else None
        if ts.kind == 'points':
            ts.cloud_n = np.asarray(a['n'], np.int64)
        if ts.kind in SMALL_COLS and a['small']:
            arr = np.asarray(a['small'], np.float64)
            ts.data = {'t_ns': np.asarray(a['st'], np.int64)}
            for i, c in enumerate(SMALL_COLS[ts.kind]):
                ts.data[c] = arr[:, i]


# ---------------------------------------------------------------- nuScenes
NUS_TYPE = {'camera': 'sensor_msgs/msg/CompressedImage', 'lidar': 'sensor_msgs/msg/PointCloud2',
            'radar': 'sensor_msgs/msg/PointCloud2'}


def _nus_tables(ver, names):
    out = {}
    for n in names:
        q = os.path.join(ver, f'{n}.json')
        with open(q, encoding='utf-8') as fh:
            out[n] = json.load(fh)
    return out


def _nus_channels(tables):
    """calibrated_sensor token -> (channel, modality)."""
    sens = {s['token']: (s.get('channel', ''), s.get('modality', '')) for s in tables['sensor']}
    return {c['token']: sens.get(c.get('sensor_token'), ('', '')) for c in tables['calibrated_sensor']}


def _read_pcd_bin(path):
    """A nuScenes lidar .pcd.bin (float32 x y z intensity ring) as a structured array."""
    a = np.fromfile(path, np.float32)
    n = len(a) // 5
    a = a[:n * 5].reshape(n, 5)
    out = np.zeros(n, [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('intensity', 'f4'), ('ring', 'u2')])
    out['x'], out['y'], out['z'], out['intensity'] = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    out['ring'] = np.clip(np.round(a[:, 4]), 0, 65535).astype(np.uint16)
    return out


def _pcd_points(path):
    """POINTS of a .pcd file's header (a radar file); None when it has none."""
    try:
        with open(path, 'rb') as fh:
            head = fh.read(2048).split(b'DATA', 1)[0].decode('ascii', 'replace')
    except OSError:
        return None
    m = re.search(r'^POINTS\s+(\d+)', head, re.M)
    return int(m.group(1)) if m else None


def _scan_nuscenes(nd, n_frames, sync_sweeps, progress, cancel) -> Scan | None:
    import cv2
    ver, root = nd
    tables = _nus_tables(ver, ('sensor', 'calibrated_sensor', 'sample_data', 'ego_pose'))
    chans = _nus_channels(tables)
    by = {}
    for sd in tables['sample_data']:
        ch = chans.get(sd.get('calibrated_sensor_token'))
        if ch is None or not ch[0]:
            continue
        by.setdefault(ch, []).append(sd)
    tops = {}
    all_ts = [int(sd['timestamp']) for v in by.values() for sd in v]
    t0 = min(all_ts, default=0) * 1000
    t1 = max(all_ts, default=0) * 1000
    targets = [{'t_ns': t, 'images': {}, 'clouds': {}} for t in _targets(t0, t1, n_frames)]
    n_ch = max(len(by), 1)
    for ci, ((ch, mod), sds) in enumerate(sorted(by.items())):
        if cancel is not None and cancel():
            return None
        sds.sort(key=lambda s: s['timestamp'])
        ty = NUS_TYPE.get(mod, 'other')
        ts = TopicScan(ch, ty, topic_kind(ty), frame_id=ch, stamp_ref='end')
        ts.count = len(sds)
        stamps = np.array([int(s['timestamp']) * 1000 for s in sds], np.int64)
        ts.log_ns, ts.stamp_ns = stamps.copy(), stamps
        files = [os.path.join(root, s.get('filename', '')) for s in sds]
        sizes = np.array([os.path.getsize(f) if os.path.isfile(f) else 0 for f in files], np.int64)
        ts.bytes = int(sizes.sum())
        if mod == 'camera':
            for s in sds:
                key = (int(s.get('width') or 0), int(s.get('height') or 0))
                if key[0] and key[1]:
                    ts.dims[key] = ts.dims.get(key, 0) + 1
            for k, tg in enumerate(targets):
                j = int(np.searchsorted(stamps, tg['t_ns']))
                if j >= len(sds):
                    continue
                img = cv2.imread(files[j], cv2.IMREAD_UNCHANGED)
                if img is None:
                    continue
                if img.ndim == 3:
                    img = img[..., ::-1][..., :3]
                smp = image_sample(img, int(stamps[j]), int(stamps[j]))
                tg['images'][ch] = smp
                if not ts.dims:
                    ts.dims[smp['size']] = ts.count
        elif mod == 'lidar':
            ts.cloud_n = sizes // 20
            ts.fields = ('x', 'y', 'z', 'intensity', 'ring')
            per = float(np.median(np.diff(stamps))) / 1e9 if len(stamps) > 2 else 0.05
            st = max(1, len(sds) // max(sync_sweeps, 1))
            for j in range(0, len(sds), st):
                if os.path.isfile(files[j]):
                    sw = process_cloud(_read_pcd_bin(files[j]), int(stamps[j]), edges=False, period_s=per)
                    ts.sweeps.append({k_: v for k_, v in sw.items() if k_ not in ('E', 'E_dt', 'K', 'K_dt')})
            for tg in targets:
                j = int(np.searchsorted(stamps, tg['t_ns']))
                sws = []
                for jj in range(max(j - 2, 0), min(j + 2, len(sds))):
                    if os.path.isfile(files[jj]):
                        sws.append(process_cloud(_read_pcd_bin(files[jj]), int(stamps[jj]), period_s=per))
                tg['clouds'][ch] = sws
        elif mod == 'radar':
            ts.cloud_n = np.array([(_pcd_points(f) or 0) if os.path.isfile(f) else 0 for f in files], np.int64)
            ts.fields = ('x', 'y', 'z')
        tops[ch] = ts
        if progress is not None:
            progress((ci + 1) / n_ch, f'reading {ch}')
    # ego poses as odometry
    eps = sorted(tables['ego_pose'], key=lambda e: e['timestamp'])
    if eps:
        q = np.array([e['rotation'] for e in eps], float)                  # w x y z
        p = np.array([e['translation'] for e in eps], float)
        t = np.array([int(e['timestamp']) * 1000 for e in eps], np.int64)
        v = np.zeros_like(p)
        if len(t) > 1:
            dtt = np.gradient(t / 1e9)
            v = np.gradient(p, axis=0) / np.maximum(dtt, 1e-6)[:, None]
        ts = TopicScan('ego_pose', 'nav_msgs/msg/Odometry', 'odom', count=len(eps), frame_id='ego_pose')
        ts.log_ns, ts.stamp_ns = t.copy(), t
        ts.data = {'t_ns': t, 'x': p[:, 0], 'y': p[:, 1], 'z': p[:, 2], 'qx': q[:, 1], 'qy': q[:, 2], 'qz': q[:, 3],
                   'qw': q[:, 0], 'vx': v[:, 0], 'vy': v[:, 1], 'vz': v[:, 2], 'wx': np.zeros(len(t)),
                   'wy': np.zeros(len(t)), 'wz': np.zeros(len(t))}
        tops['ego_pose'] = ts
    return Scan(ver, 'nuscenes', log_key(ver), t0, t1, tops, targets, n_frames, sync_sweeps,
                notes=['nuScenes: stamps are the recorded times (no separate log time, so latency is not checked); '
                       'a lidar sample\'s stamp is its sweep end.'])


# =========================================================================== mapping topics to sensors
def _tokens(text) -> list:
    return [t for t in re.split(r'[^0-9A-Za-z]+', str(text or '').upper()) if t]


def _sensors(var) -> list:
    """(uid, tag, kind, obj) of the variant's visible cameras and sensors."""
    out = [(c.uid, c.tag, 'camera', c) for c in var.cameras if c.visible]
    out += [(s.uid, s.tag, s.kind, s) for s in var.sensors if s.visible]
    return out


def auto_map(var, topics: dict) -> dict:
    """topic -> tag for the topics whose name or frame names a visible sensor of a kind their messages fit
    (topics: {topic: (type, frame_id, messages)}). One topic per sensor (the one with the most messages); a single
    leftover topic of a kind and a single leftover sensor of that kind are paired."""
    sens = _sensors(var)
    claims = {}
    for topic, (ty, frame, n) in topics.items():
        kinds = SENSOR_KINDS_FOR.get(topic_kind(ty), ())
        if not kinds:
            continue
        tok_t, tok_f = set(_tokens(topic)), set(_tokens(frame))
        best, score, tie = None, 0, False
        for uid, tag, kind, _ in sens:
            if kind not in kinds:
                continue
            T = tag.upper()
            sc = (len(T) * 2 + 1) if T in tok_t else (len(T) * 2) if T in tok_f else 0
            if sc > score:
                best, score, tie = tag, sc, False
            elif sc == score and sc > 0:
                tie = True
        if best is not None and not tie:
            claims.setdefault(best, []).append((n, topic))
    out = {}
    for tag, lst in claims.items():
        lst.sort(key=lambda r: (-r[0], r[1]))
        out[lst[0][1]] = tag
    taken = set(out.values())
    for tk in ('image', 'points', 'imu', 'fix'):
        left_t = [t for t, (ty, _, _) in topics.items() if topic_kind(ty) == tk and t not in out]
        left_s = [tag for _, tag, kind, _ in sens if kind in SENSOR_KINDS_FOR[tk][:1] and tag not in taken]
        if len(left_t) == 1 and len(left_s) == 1:
            out[left_t[0]] = left_s[0]
            taken.add(left_s[0])
    return out


def mapping_for(var, scan_or_info, overrides: dict) -> dict:
    """The effective mapping: automatic, then the user's edits ('' = not a rig sensor); only tags of visible sensors
    whose kind fits the topic's messages."""
    if isinstance(scan_or_info, Scan):
        tops = {t: (ts.type, ts.frame_id, ts.count) for t, ts in scan_or_info.topics.items()}
    else:
        tops = {t: (ty, '', scan_or_info.counts.get(t, 0)) for t, ty in scan_or_info.topics.items()}
    m = auto_map(var, tops)
    kinds = {tag: kind for _, tag, kind, _ in _sensors(var)}
    for topic, tag in (overrides or {}).items():
        if topic not in tops:
            continue
        if not tag:
            m.pop(topic, None)
        elif kinds.get(tag) in SENSOR_KINDS_FOR.get(topic_kind(tops[topic][0]), ()):
            for t in [t for t, g in m.items() if g == tag]:
                del m[t]
            m[topic] = tag
    return m


def compatible_tags(var, msg_type) -> list:
    """Tags a topic of this type can be mapped to."""
    kinds = SENSOR_KINDS_FOR.get(topic_kind(msg_type), ())
    return [tag for _, tag, kind, _ in _sensors(var) if kind in kinds]


# =========================================================================== results
@dataclass
class DriveLogResult:
    settings: DriveLogSettings
    path: str
    source: str
    variant: str
    variant_uid: str
    span: dict                      # start_ns, end_ns, duration_s, messages, bytes
    topics: list                    # per topic: topic, type, kind, messages, bytes, tag (mapped or '')
    mapping: dict                   # topic -> tag
    streams: list                   # per visible sensor (see _stream)
    data: dict                      # 'rows' per mapped sensor, 'totals'
    sync: dict
    extrinsics: list                # per camera
    imu: list
    gnss: list
    findings: list                  # [{'severity', 'section', 'message', 'tags', 'uids'}], worst first
    metrics: dict
    overlays: dict = field(default_factory=dict)    # camera uid -> frames for the overlay viewer
    gaps: dict = field(default_factory=dict)        # tag -> [(start s, end s)] after the bag start
    notes: list = field(default_factory=list)
    scan: Scan | None = None
    seconds: float = 0.0

    def stream(self, tag):
        return next((s for s in self.streams if s['tag'] == tag), None)

    def extrinsic(self, tag):
        return next((e for e in self.extrinsics if e['tag'] == tag), None)

    def worst(self) -> str:
        return self.findings[0]['severity'] if self.findings else 'ok'


def _finding(out, sev, section, message, tags=(), uids=()):
    out.append({'severity': sev, 'section': section, 'message': message, 'tags': list(tags), 'uids': list(uids)})


DATASET_SCHEMA = 'vantage.dataset/1'


def vantage_dataset(path) -> dict | None:
    """The dataset.json of a drive written by Vantage's sample-dataset generator (core.dataset), found from the bag
    folder, its .mcap/.db3 file or the nuScenes folder (they sit in <out>/rosbag/<name>/ and <out>/nuscenes/, the JSON
    in <out>); None for any other log."""
    p = os.path.abspath(path)
    for _ in range(4):
        f = os.path.join(p, 'dataset.json')
        if os.path.isfile(f):
            try:
                with open(f, encoding='utf-8') as fh:
                    doc = json.load(fh)
            except (OSError, ValueError):
                doc = None
            if isinstance(doc, dict) and doc.get('schema') == DATASET_SCHEMA:
                return doc
        up = os.path.dirname(p)
        if up == p:
            break
        p = up
    return None


def recorded_camera(doc, project, cam, period_ms) -> tuple[float | None, tuple | None]:
    """(frame rate, image size) a Vantage sample dataset wrote this camera with: one frame per lidar rotation or the
    module's rate (settings camera_rate), at image_scale of the sensor's size (as core.dataset sizes them)."""
    st = (doc or {}).get('settings') or {}
    mod = project.module_for(cam)
    hz = 1000.0 / period_ms if (st.get('camera_rate', 'lidar') == 'lidar' and period_ms) else (float(mod.fps or 0)
                                                                                              or None)
    k = float(st.get('image_scale') or 1.0)
    size = (max(16, int(round(mod.width_px * k))), max(16, int(round(mod.height_px * k))))
    return hz, size


def _expected_hz(project, kind, obj) -> float | None:
    if kind == 'camera':
        fps = float(project.module_for(obj).fps or 0)
        return fps if fps > 0 else None
    m = project.sensor_model_for(obj)
    r = float(m.rate_hz or 0)
    return r if r > 0 else None


def interval_stats(t_ns, nominal_s=None, drop_factor=1.5) -> dict:
    """Rate, drops and jitter of a stream from its stamps (ns, in the order received): intervals over drop_factor ×
    the nominal period as drops (each missing round(gap / period) − 1 frames), the rate from the mean of the others
    (a stream stamped on a coarser clock, a 15 Hz radar on 80 Hz ticks, alternates 62.5 and 75 ms: its median interval
    says 16 Hz, the mean 15), their jitter, stamps not after the previous one."""
    t = np.asarray(t_ns, np.int64)
    out = {'n': int(len(t)), 'median_ms': None, 'rate_hz': None, 'drops': 0, 'drop_events': 0, 'longest_gap_ms': None,
           'missing_s': 0.0, 'missing_share': 0.0, 'drop_share': 0.0, 'jitter_ms': None, 'jitter_p95_ms': None,
           'nonmonotonic': 0, 'gaps': []}
    if len(t) < 2:
        return out
    out['nonmonotonic'] = int(np.count_nonzero(np.diff(t) <= 0))
    ts = np.sort(t)
    d = np.diff(ts) / 1e9
    pos = d[d > 0]
    if not len(pos):
        return out
    med = float(np.median(pos))
    # drops against the plan's period, unless the stream runs at another rate (that is reported on its own)
    nom = float(nominal_s) if nominal_s and abs(med - nominal_s) <= 0.05 * nominal_s else med
    out['median_ms'] = med * 1000.0
    out['rate_hz'] = 1.0 / med
    gap = d > drop_factor * nom
    miss = np.maximum(np.round(d[gap] / nom) - 1, 1).astype(int)
    out['drops'] = int(miss.sum())
    out['drop_events'] = int(gap.sum())
    out['longest_gap_ms'] = float(d.max() * 1000.0)
    span = (ts[-1] - ts[0]) / 1e9 + nom
    out['missing_s'] = float(np.sum(d[gap] - nom))
    out['missing_share'] = out['missing_s'] / span if span > 0 else 0.0
    out['drop_share'] = out['drops'] / (len(t) + out['drops'])
    normal = d[~gap & (d > 0)]
    if len(normal):
        out['rate_hz'] = 1.0 / float(np.mean(normal))
    if len(normal) > 1:
        out['jitter_ms'] = float(np.std(normal) * 1000.0)
        out['jitter_p95_ms'] = float(np.percentile(np.abs(normal - nom), 95) * 1000.0)
    g0 = np.nonzero(gap)[0]
    out['gaps'] = [(ts[i] / 1e9, ts[i + 1] / 1e9) for i in g0[:2000]]          # s since the epoch
    return out


def _points_expected(project, s, engine):
    """(points per frame of the model, the share the vehicle blocks or None)."""
    m = project.sensor_model_for(s)
    per = m.pps() / m.rate_hz if m.is_lidar and m.rate_hz > 0 else None
    blocked = None
    if per and engine is not None:
        try:
            from . import sensor_coverage as SC
            if engine.vehicle_geom(project) is not None:
                blocked = float(SC._Lidar(s, m, engine.cube(s.pos())).blocked.mean())
        except Exception:
            blocked = None
    return per, blocked


def _stream(project, var, uid, tag, kind, obj, topic, scan, st, engine, recorded=None) -> dict:
    """Stream health of one sensor (see the module docstring). recorded: (rate, size) a Vantage sample dataset wrote
    this camera with, which then replaces the module's."""
    e = {'uid': uid, 'tag': tag, 'kind': kind, 'topic': topic, 'type': '', 'messages': 0,
         'expected_hz': _expected_hz(project, kind, obj), 'status': 'ok', 'problems': [], 'notes': []}
    if kind == 'camera':
        mod = project.module_for(obj)
        e['expected_size'] = (int(mod.width_px), int(mod.height_px))
        if recorded is not None:
            e['expected_hz'], e['expected_size'] = recorded[0] or e['expected_hz'], recorded[1] or e['expected_size']
    if topic is None:
        return e
    ts = scan.topics[topic]
    e['type'], e['messages'] = ts.type, ts.count
    stamps = ts.stamp_ns if ts.stamp_ns is not None else ts.log_ns
    e['stamped'] = ts.stamp_ns is not None
    nom = 1.0 / e['expected_hz'] if e['expected_hz'] else None
    iv = interval_stats(stamps, nom, st.drop_factor)
    gaps = iv.pop('gaps')
    e.update(iv)
    base = scan.start_ns / 1e9
    e['gaps'] = [(a - base, b - base) for a, b in gaps]
    if len(stamps):                                    # the timeline's bar, on the gaps' clock
        e['span_s'] = ((int(stamps.min()) - scan.start_ns) / 1e9, (int(stamps.max()) - scan.start_ns) / 1e9)
    if len(ts.log_ns):
        e['first_s'] = (int(ts.log_ns.min()) - scan.start_ns) / 1e9
        e['last_s'] = (int(ts.log_ns.max()) - scan.start_ns) / 1e9
        dur = scan.duration_s
        per = nom or ((iv['median_ms'] or 0.0) / 1000.0)
        e['coverage'] = min((e['last_s'] - e['first_s'] + per) / dur, 1.0) if dur > 0 else 1.0
    if ts.stamp_ns is not None and scan.source == 'rosbag' and len(ts.log_ns) == len(ts.stamp_ns):
        lat = (ts.log_ns - ts.stamp_ns) / 1e6
        e['latency_ms'] = {'median': float(np.median(lat)), 'p95': float(np.percentile(lat, 95)),
                           'min': float(lat.min()), 'max': float(lat.max())}
    if kind == 'camera' and ts.dims:
        e['sizes'] = sorted(ts.dims.items(), key=lambda kv: -kv[1])
        e['size'] = e['sizes'][0][0]
    if ts.kind == 'points' and ts.cloud_n is not None and len(ts.cloud_n):
        cn = ts.cloud_n[ts.cloud_n >= 0]
        e['points'] = float(np.median(cn)) if len(cn) else None
        sw = [s for s in ts.sweeps if s.get('n')]
        if sw:
            e['valid_share'] = float(np.median([s['n_valid'] / max(s['n'], 1) for s in sw]))
        if kind == 'lidar':
            e['points_expected'], e['blocked_share'] = _points_expected(project, obj, engine)
    return e


def _judge_stream(e, st, scan, out):
    """Findings of one stream; sets its status."""
    tag, kind = e['tag'], e['kind']
    word = KIND_WORD.get(kind, kind)
    sev = []

    def add(s, msg):
        sev.append(s)
        e['problems'].append(msg)
        _finding(out, s, 'streams', msg, [tag], [e['uid']])
    if e['topic'] is None:
        if kind == 'gnss':
            e['status'] = 'ok'
            e['notes'].append('no topic of its own (its signal reaches the recorder through the GNSS receiver)')
            return
        add('fault', f'{tag}: no topic in the log for this {word}.')
        e['status'] = 'fault'
        return
    exp = e['expected_hz']
    rate = e.get('rate_hz')
    if rate is None:
        add('fault', f'{tag}: {e["messages"]} message{"s" * (e["messages"] != 1)} on {e["topic"]}: too few to '
                     'measure a rate.')
    elif exp:
        dev = abs(rate - exp) / exp
        if dev > 0.05:
            add('fault', f'{tag} runs at {rate:.2f} Hz, the plan says {exp:g} Hz ({e["topic"]}).')
        elif dev > 0.01:
            add('warning', f'{tag} runs at {rate:.2f} Hz against the plan\'s {exp:g} Hz.')
    if e.get('drops'):
        share = e['drop_share'] * 100
        msg = (f'{tag} dropped {e["drops"]} frame{"s" * (e["drops"] != 1)} ({share:.1f} %) in {e["drop_events"]} '
               f'gap{"s" * (e["drop_events"] != 1)}; the longest {e["longest_gap_ms"]:.0f} ms, '
               f'{e["missing_share"] * 100:.1f} % of its time missing.')
        add('fault' if share > 1.0 else 'warning', msg)
    if e.get('nonmonotonic'):
        add('fault', f'{tag}: {e["nonmonotonic"]} stamp{"s" * (e["nonmonotonic"] != 1)} not after the one before '
                     '(duplicated or going back).')
    lat = e.get('latency_ms')
    if lat is not None:
        if lat['min'] < -1.0:
            add('fault', f'{tag}: header stamps up to {-lat["min"]:.1f} ms after the time the message was logged: the '
                         'sensor\'s clock and the recorder\'s are not in sync.')
        elif lat['median'] > 100.0 + (1000.0 / exp if exp and e['kind'] in ('lidar', 'radar') else 0.0):
            add('warning', f'{tag}: logged {lat["median"]:.0f} ms after its stamp (median).')
    if e.get('stamped') is False:
        add('warning', f'{tag}: messages without a header stamp; rates come from the log time.')
    cov = e.get('coverage')
    if cov is not None and cov < 0.9 and scan.duration_s > 1.0:
        add('warning', f'{tag} covers {cov * 100:.0f} % of the drive (from {e["first_s"]:.1f} s to '
                       f'{e["last_s"]:.1f} s of {scan.duration_s:.1f} s).')
    jit = e.get('jitter_p95_ms')
    if jit is not None and exp and jit > 0.1 * 1000.0 / exp:
        add('warning', f'{tag}: intervals jitter by up to {jit:.1f} ms (p95) on a {1000.0 / exp:.1f} ms period.')
    if kind == 'camera' and e.get('size') is not None:
        w, h = e['size']
        W, H = e['expected_size']
        if (w, h) != (W, H):
            if abs(w / h - W / H) <= 0.01 * W / H:                  # the same aspect: binned or scaled
                add('warning', f'{tag}: images are {w} × {h}, the module\'s sensor is {W} × {H} (scaled by '
                               f'{w / W:.2f}).')
            else:
                add('fault', f'{tag}: images are {w} × {h}, the module\'s sensor is {W} × {H}: a different aspect '
                             '(cropped or another mode).')
        if len(e.get('sizes', [])) > 1:
            add('warning', f'{tag}: images of {len(e["sizes"])} different sizes.')
    if kind == 'lidar' and e.get('points') is not None and e.get('points_expected'):
        r = e['points'] / e['points_expected']
        blk = e.get('blocked_share')
        b = f'; the plan expects {blk * 100:.0f} % of them on the vehicle' if blk is not None else ''
        if r > 1.05:
            add('warning', f'{tag}: {e["points"]:,.0f} points per cloud, {r:.2f}× the model\'s '
                           f'{e["points_expected"]:,.0f} per frame (another mode or dual returns?).')
        elif r < 0.3:
            add('warning', f'{tag}: {e["points"]:,.0f} points per cloud, {r * 100:.0f} % of the model\'s '
                           f'{e["points_expected"]:,.0f} per frame{b}.')
        elif r < 0.95:
            e['notes'].append(f'{r * 100:.0f} % of the model\'s points per frame: a dense cloud without the '
                              f'no-return points{b}')
    e['status'] = 'fault' if 'fault' in sev else 'warning' if 'warning' in sev else 'ok'


# =========================================================================== data rate
def _data(project, var, streams, scan, out) -> dict:
    from . import recording as RB
    bud = RB.budget(project, var)
    ecu = project.ecu
    rows = []
    for e in streams:
        if e['topic'] is None:
            continue
        ts = scan.topics[e['topic']]
        if not ts.count:
            continue
        per = (1.0 / e['expected_hz']) if e.get('expected_hz') else ((e.get('median_ms') or 0.0) / 1000.0)
        span = (int(ts.log_ns.max()) - int(ts.log_ns.min())) / 1e9 + per if len(ts.log_ns) else 0.0
        mbps = ts.bytes * 8 / 1e6 / span if span > 0 else None
        r = {'tag': e['tag'], 'uid': e['uid'], 'kind': e['kind'], 'topic': e['topic'], 'messages': ts.count,
             'bytes': ts.bytes, 'mbps': mbps, 'mb_s': mbps / 8.0 if mbps is not None else None,
             'gb_h': mbps / 8.0 * 3.6 if mbps is not None else None, 'bytes_per_msg': ts.bytes / ts.count}
        row = bud.row(e['tag'])
        if row is not None and row['recorded']:
            r['planned_mbps'] = row['mbps']
            r['planned_format'] = row['format']
            r['ratio'] = mbps / row['mbps'] if mbps is not None and row['mbps'] > 0 else None
        if e['kind'] == 'camera' and e.get('size'):
            w, h = e['size']
            r['bpp'] = r['bytes_per_msg'] * 8.0 / (w * h)
            r['planned_bpp'] = bud.settings.camera_bpp(ecu)
            r['compression'] = (w * h * ecu.bpp / 8.0) / r['bytes_per_msg'] if r['bytes_per_msg'] > 0 else None
            r['compressed'] = ts.type == 'sensor_msgs/msg/CompressedImage'
        rows.append(r)
        ratio = r.get('ratio')
        if ratio is not None and ratio > 1.25:
            extra = (f' ({r["bpp"]:.2f} bit/px against the plan\'s {r["planned_bpp"]:g})' if 'bpp' in r else '')
            _finding(out, 'warning', 'data', f'{e["tag"]} records {mbps:,.1f} Mbit/s, {ratio:.1f}× the budget\'s '
                                              f'{r["planned_mbps"]:,.1f}{extra}: the plan\'s figure is low.',
                     [e['tag']], [e['uid']])
    total_bytes = sum(ts.bytes for ts in scan.topics.values() if ts.topic != 'ego_pose')
    dur = scan.duration_s
    mb_s = total_bytes / 1e6 / dur if dur > 0 else None
    tot = {'bytes': total_bytes, 'duration_s': dur, 'mb_s': mb_s, 'gb_h': mb_s * 3.6 if mb_s is not None else None,
           'planned_gb_h': bud.totals['gb_h'], 'planned_mb_s': bud.totals['mb_s'],
           'storage_tb': bud.settings.storage_tb, 'write_mb_s': bud.settings.write_mb_s,
           'write_headroom': bud.settings.write_headroom}
    if mb_s:
        tot['hours_per_set'] = bud.settings.storage_tb * 1e6 / mb_s / 3600.0
        tot['ratio'] = mb_s / bud.totals['mb_s'] if bud.totals['mb_s'] > 0 else None
        if mb_s * bud.settings.write_headroom > bud.settings.write_mb_s:
            _finding(out, 'fault', 'data', f'The log was written at {mb_s:,.0f} MB/s: with the {bud.settings.write_headroom:g}× '
                                           f'headroom the recorder\'s {bud.settings.write_mb_s:,.0f} MB/s sustained '
                                           'write is not enough.')
        if tot['ratio'] is not None and tot['ratio'] > 1.25:
            _finding(out, 'warning', 'data', f'The log holds {tot["gb_h"]:,.0f} GB per hour, {tot["ratio"]:.1f}× the '
                                             f'budget\'s {bud.totals["gb_h"]:,.0f}: one {bud.settings.storage_tb:g} TB '
                                             f'disk set lasts {tot["hours_per_set"]:.1f} h.')
        else:
            _finding(out, 'ok', 'data', f'The log holds {tot["gb_h"]:,.1f} GB per hour (the budget: '
                                        f'{bud.totals["gb_h"]:,.1f}); one {bud.settings.storage_tb:g} TB disk set '
                                        f'lasts {tot["hours_per_set"]:.1f} h at this rate.')
    return {'rows': rows, 'totals': tot}


# =========================================================================== motion between lidar points and images
class Motion:
    """The vehicle's motion from odometry (pose over time) or an IMU (angular rate only), to move lidar points measured
    at their own times into the vehicle frame at a camera's exposure."""

    def __init__(self, kind='none', t=None, pos=None, ypr=None, omega=None, speed=None, source=''):
        self.kind, self.t, self.pos, self.ypr, self.omega, self.speed, self.source = \
            kind, t, pos, ypr, omega, speed, source

    @staticmethod
    def from_scan(project, var, scan, mapping) -> 'Motion':
        base = scan.start_ns
        odo = [ts for ts in scan.topics.values() if ts.kind == 'odom' and ts.data and len(ts.data['t_ns']) > 1]
        if odo:
            ts = max(odo, key=lambda x: len(x.data['t_ns']))
            d = ts.data
            o = np.argsort(d['t_ns'], kind='stable')
            t = (d['t_ns'][o] - base) / 1e9
            q = np.stack([d['qx'], d['qy'], d['qz'], d['qw']], 1)[o]
            ypr = np.radians(np.array([euler_zyx(_quat_R(qq)) for qq in q]))
            ypr = np.unwrap(ypr, axis=0)
            pos = np.stack([d['x'], d['y'], d['z']], 1)[o]
            v = np.stack([d['vx'], d['vy'], d['vz']], 1)[o]
            speed = np.linalg.norm(v, axis=1)
            if not np.isfinite(speed).all() or (speed == 0).all() and len(t) > 2:
                vv = np.gradient(pos, axis=0) / np.maximum(np.gradient(t), 1e-6)[:, None]
                speed = np.where(np.isfinite(speed) & (speed > 0), speed, np.linalg.norm(vv, axis=1))
            return Motion('odometry', t, pos, ypr, speed=(t, speed), source=ts.topic)
        tag_of = {t: g for t, g in mapping.items()}
        imus = [(ts, var_sensor(var, tag_of.get(ts.topic))) for ts in scan.topics.values()
                if ts.kind == 'imu' and ts.data and len(ts.data['t_ns']) > 1]
        imus = [(ts, s) for ts, s in imus if s is not None] or [(ts, None) for ts, _ in imus]
        speed = _gnss_speed(scan)
        if imus:
            ts, s = imus[0]
            R = rot_zyx(s.yaw, s.pitch, s.roll) if s is not None else np.eye(3)
            d = ts.data
            o = np.argsort(d['t_ns'], kind='stable')
            w = np.stack([d['gx'], d['gy'], d['gz']], 1)[o] @ R.T
            return Motion('imu', (d['t_ns'][o] - base) / 1e9, omega=w, speed=speed, source=ts.topic)
        return Motion('none', speed=speed)

    @property
    def label(self) -> str:
        return {'odometry': 'odometry (rotation and travel)', 'imu': 'IMU (rotation only)',
                'none': 'none'}[self.kind]

    def speed_at(self, t_s) -> float | None:
        if self.speed is None:
            return None
        t, v = self.speed
        if not len(t):
            return None
        return float(np.interp(t_s, t, v))

    def apply(self, P, t_pts, t_c):
        """Points P (K, 3) of the vehicle frame at their times t_pts (s after the bag start) → the vehicle frame at
        t_c."""
        if self.kind == 'none' or t_pts is None or not len(P):
            return P
        dt = np.asarray(t_pts, float) - t_c
        if self.kind == 'imu':
            sel = (self.t >= t_c - 0.1) & (self.t <= t_c + 0.1)
            w = self.omega[sel].mean(0) if sel.any() else self.omega[np.argmin(np.abs(self.t - t_c))]
            out = np.empty_like(P)
            bins = np.round(dt / 0.002).astype(int)
            for b in np.unique(bins):
                m = bins == b
                out[m] = P[m] @ _rotvec_R(w * float(b) * 0.002).T
            return out
        Rc, pc = self._pose(t_c)
        out = np.empty_like(P)
        bins = np.round(dt / 0.002).astype(int)
        for b in np.unique(bins):
            m = bins == b
            Ri, pi = self._pose(t_c + float(b) * 0.002)
            out[m] = (P[m] @ Ri.T + pi - pc) @ Rc
        return out

    def _pose(self, t):
        ypr = np.array([np.interp(t, self.t, self.ypr[:, k]) for k in range(3)])
        p = np.array([np.interp(t, self.t, self.pos[:, k]) for k in range(3)])
        return rot_zyx(*np.degrees(ypr)), p


def var_sensor(var, tag):
    if not tag:
        return None
    return next((s for s in var.sensors if s.tag == tag), None)


def _quat_R(q):
    x, y, z, w = q
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _rotvec_R(v):
    th = float(np.linalg.norm(v))
    if th < 1e-12:
        return np.eye(3)
    k = np.asarray(v, float) / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def _R_rotvec(R):
    c = max(-1.0, min(1.0, (np.trace(R) - 1) / 2))
    th = math.acos(c)
    if th < 1e-12:
        return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(th))
    return v * th


def _gnss_speed(scan):
    """(t s, speed m/s) from the fixes of the first NavSatFix stream with a fix, or None."""
    for ts in scan.topics.values():
        if ts.kind != 'fix' or not ts.data or len(ts.data['t_ns']) < 3:
            continue
        d = ts.data
        ok = (d['status'] >= 0) & np.isfinite(d['lat']) & np.isfinite(d['lon'])
        if ok.sum() < 3:
            continue
        t = (d['t_ns'][ok] - scan.start_ns) / 1e9
        o = np.argsort(t)
        t = t[o]
        lat, lon = np.radians(d['lat'][ok][o]), np.radians(d['lon'][ok][o])
        Re = 6378137.0
        x = (lon - lon[0]) * Re * math.cos(float(lat[0]))
        y = (lat - lat[0]) * Re
        dt = np.gradient(t)
        v = np.hypot(np.gradient(x), np.gradient(y)) / np.maximum(dt, 1e-3)
        k = max(1, min(len(v) // 4, 5))
        v = np.convolve(v, np.ones(k) / k, mode='same')
        return t, v
    return None


# =========================================================================== sync
def _phase_lock(project, s, ts, st, scan) -> dict | None:
    """The measured phase lock of a mapped spinning lidar from its sampled sweeps."""
    m = project.sensor_model_for(s)
    f = float(m.rate_hz)
    sense_m = 1 if m.rotation == 'ccw' else -1
    sw = [x for x in ts.sweeps if x.get('n_valid')]
    if not sw or f <= 0:
        return None
    fits = [x for x in sw if x.get('fit') is not None]
    use_time = len(fits) >= max(1, len(sw) // 3)
    stamps = ts.stamp_ns if ts.stamp_ns is not None else ts.log_ns
    per = interval_stats(stamps)['median_ms']
    if use_time:
        b = float(np.median([x['fit']['b'] for x in fits]))
        f_meas, sense = abs(b) / 360.0, (1 if b > 0 else -1)
        sw = fits
    else:
        f_meas, sense = (1000.0 / per if per else f), sense_m
    f_use = f if abs(f_meas - f) <= 0.02 * f else f_meas
    ref = st.lidar_stamp if st.lidar_stamp != 'auto' else ts.stamp_ref
    ph, tk = [], []
    for x in sw:
        p, src = sweep_phase(x, f_use, sense, ref, use_time)
        if p is None:
            continue
        ph.append(p)
        tm = x['fit']['tm'] if use_time else 0.0
        tk.append((x['stamp_ns'] - scan.start_ns) / 1e9 + tm)
    if not ph:
        return None
    o = np.argsort(tk)
    tk = np.asarray(tk)[o]
    ph = np.degrees(np.unwrap(np.radians(np.asarray(ph)[o])))
    med = float(np.round(_circ_median(ph, 360.0), 6) % 360.0)
    spread = float(np.std(_wrap(ph - med, 360.0)))
    err = float(_wrap(med - float(s.phase_deg or 0.0), 360.0))
    late_ms = -sense * err / (360.0 * f_use) * 1000.0         # the sweep passes every direction this much later
    return {'uid': s.uid, 'tag': s.tag, 'topic': ts.topic, 'rate_hz': f_meas, 'plan_rate_hz': f, 'use_rate_hz': f_use,
            'sense': sense, 'plan_sense': sense_m, 'phase_deg': med, 'plan_phase_deg': float(s.phase_deg or 0.0),
            'phase_error_deg': err, 'late_ms': late_ms, 'spread_deg': spread,
            'spread_ms': spread / (360.0 * f_use) * 1000.0, 'sweeps': len(ph),
            'source': 'point times' if use_time else f'header stamps ({ref} of the sweep) and the spin model',
            'rms_deg': float(np.median([x['fit']['rms'] for x in fits])) if use_time else None,
            '_t': tk, '_phi': ph}


def head_azimuth(lock, t_abs_ns):
    """The sensor-frame azimuth (deg) the lidar head faces at absolute times (ns), from its measured phase lock (the
    phase at the sync pulses interpolated over the drive, the measured rate and sense)."""
    t_abs_ns = np.asarray(t_abs_ns, np.int64)
    f = lock['use_rate_hz']
    T_ns = int(round(1e9 / f))
    rem = t_abs_ns % T_ns
    rem = np.where(rem >= T_ns // 2, rem - T_ns, rem) / 1e9               # s since the nearest sync pulse
    base = lock['_base_ns']
    phi = np.interp((t_abs_ns - base) / 1e9, lock['_t'], lock['_phi'])
    return phi + lock['sense'] * 360.0 * f * rem


def _sync(project, var, plan, rev, scan, st, out) -> dict:
    """Measured sync against the timing plan (see the module docstring)."""
    res = {'reason': plan.reason, 'lidar': None, 'lidars': [], 'cameras': [], 'groups': [],
           'stamp_convention': st.camera_stamp,
           'readout_ms': plan.settings.readout_ms, 'exposure_ms': plan.settings.exposure_ms}
    conv_ms = stamp_offset_ms(st.camera_stamp, plan.settings.readout_ms, plan.settings.exposure_ms)
    locks = {}
    for s in var.sensors:
        if not s.visible or s.kind != 'lidar' or not project.sensor_model_for(s).spinning or s.tag not in rev:
            continue
        L = _phase_lock(project, s, scan.topics[rev[s.tag]], st, scan)
        if L is not None:
            L['_base_ns'] = scan.start_ns
            locks[s.uid] = L
            res['lidars'].append(L)
    tol = st.sync_tol_ms
    for L in res['lidars']:
        if abs(L['rate_hz'] - L['plan_rate_hz']) > 0.02 * L['plan_rate_hz']:
            _finding(out, 'fault', 'sync', f'{L["tag"]} turns at {L["rate_hz"]:.2f} Hz, the plan says '
                                           f'{L["plan_rate_hz"]:g} Hz.', [L['tag']], [L['uid']])
        if L['source'] == 'point times' and L['sense'] != L['plan_sense']:
            _finding(out, 'fault', 'sync', f'{L["tag"]} turns {"counter-" if L["sense"] > 0 else ""}clockwise '
                                           '(seen from its top), the model says the other way.', [L['tag']],
                     [L['uid']])
        if abs(L['late_ms']) > tol:
            _finding(out, 'fault', 'sync',
                     f'{L["tag"]} is phase-locked at {L["phase_deg"]:.1f}°, the plan says {L["plan_phase_deg"]:.1f}°: '
                     f'its sweep passes every direction {abs(L["late_ms"]):.1f} ms '
                     f'{"later" if L["late_ms"] > 0 else "earlier"} than planned (from {L["source"]}).',
                     [L['tag']], [L['uid']])
        elif L['spread_ms'] > tol:
            pass
        else:
            _finding(out, 'ok', 'sync', f'{L["tag"]} is phase-locked at {L["phase_deg"]:.1f}° (plan '
                                        f'{L["plan_phase_deg"]:.1f}°, {abs(L["late_ms"]):.2f} ms off) over '
                                        f'{L["sweeps"]} sweeps.', [L['tag']], [L['uid']])
        if L['spread_ms'] > tol:
            _finding(out, 'warning', 'sync', f'{L["tag"]}\'s phase wanders by {L["spread_deg"]:.1f}° '
                                             f'({L["spread_ms"]:.1f} ms, standard deviation): not phase-locked?',
                     [L['tag']], [L['uid']])
    ref = locks.get(plan.lidar.uid) if plan.lidar is not None else None
    if plan.lidar is not None:
        res['lidar'] = ref
        if ref is None:
            res['reason'] = (f'{plan.lidar.tag} (the plan\'s reference lidar) has no topic in the log.' if
                             plan.lidar.tag not in rev else
                             f'{plan.lidar.tag}\'s sweeps could not be fitted (no points with a direction).')
    cams = {c.uid: c for c in var.cameras}
    for e in plan.cams:
        topic = rev.get(e['tag'])
        row = {'uid': e['uid'], 'tag': e['tag'], 'topic': topic, 'group': e['sync'] or None,
               'planned_ms': e.get('misalign_ms'), 'cycle_ms': e.get('cycle_ms'), 'alignment': e.get('alignment'),
               'frames': 0, 'status': 'ok'}
        res['cameras'].append(row)
        if topic is None:
            row['status'] = None
            continue
        ts = scan.topics[topic]
        stamps = np.sort(ts.stamp_ns if ts.stamp_ns is not None else ts.log_ns)
        row['frames'] = int(len(stamps))
        if ref is None or e.get('azimuth_deg') is None or not len(stamps):
            continue
        mid = stamps + int(round(conv_ms * 1e6))
        az = head_azimuth(ref, mid)
        f = ref['use_rate_hz']
        T_ms = 1000.0 / f
        off = _wrap(az - e['azimuth_deg'], 360.0) / (ref['sense'] * 360.0 * f) * 1000.0
        cyc = e.get('cycle_ms') or T_ms
        off = _wrap(off, cyc)
        row['offset_ms'] = _circ_median(off, cyc)
        row['abs_p95_ms'] = float(np.percentile(np.abs(off), 95))
        if e.get('misalign_ms') is not None and e.get('alignment') != 'drifts':
            dev = _wrap(off - e['misalign_ms'], cyc)
            mu = _circ_median(dev, cyc)
            dev = _wrap(dev - mu, cyc) + mu
            own = dev + ref['late_ms']
            row['deviation_ms'] = float(np.median(dev))
            row['own_ms'] = float(np.median(own))
            row['lidar_part_ms'] = -ref['late_ms']
            row['spread_ms'] = float(np.percentile(dev, 95) - np.percentile(dev, 5))
            row['p95_ms'] = float(np.percentile(np.abs(e['misalign_ms'] + dev), 95))
        else:
            row['spread_ms'] = float(np.percentile(off, 95) - np.percentile(off, 5))
    # ---- trigger lines: stamp skew against the line's first camera
    byc = {r['uid']: r for r in res['cameras']}
    for g in plan.groups:
        mem = [u for u in g['uids'] if byc.get(u, {}).get('topic')]
        if len(mem) < 2:
            continue
        ref_c = byc[mem[0]]
        rs = np.sort(_stamps(scan.topics[ref_c['topic']]))
        per = []
        for u in mem[1:]:
            r = byc[u]
            s_ = np.sort(_stamps(scan.topics[r['topic']]))
            j = np.clip(np.searchsorted(rs, s_), 1, len(rs) - 1)
            near = np.where(np.abs(rs[j] - s_) < np.abs(rs[j - 1] - s_), rs[j], rs[j - 1])
            sk = (s_ - near) / 1e6
            half = (g['period_ms'] or 100.0) / 2
            sk = sk[np.abs(sk) < half]
            if not len(sk):
                continue
            d = {'tag': r['tag'], 'uid': u, 'median_ms': float(np.median(sk)),
                 'p95_ms': float(np.percentile(np.abs(sk), 95)), 'frames': int(len(sk))}
            r['skew_ms'], r['skew_p95_ms'], r['skew_ref'] = d['median_ms'], d['p95_ms'], ref_c['tag']
            per.append(d)
        res['groups'].append({'letter': g['letter'], 'reference': ref_c['tag'], 'cameras': per,
                              'worst_p95_ms': max((d['p95_ms'] for d in per), default=None)})
    # ---- findings per camera
    bad = []
    for r in res['cameras']:
        if r['topic'] is None:
            continue
        msg, sev = [], None
        own = r.get('own_ms')
        if own is not None and abs(own) > tol:
            sev = 'fault' if abs(own) > 2 * tol else 'warning'
            msg.append(f'{r["tag"]} fires {abs(own):.1f} ms {"late" if own > 0 else "early"} against the sync plan')
        sk = r.get('skew_ms')
        if sk is not None and abs(sk) > tol:
            sev = sev or ('fault' if abs(sk) > 2 * tol else 'warning')
            msg.append(f'{abs(sk):.1f} ms {"after" if sk > 0 else "before"} {r["skew_ref"]} on trigger line '
                       f'{r["group"]}')
        if msg:
            text = msg[0] + (f' ({msg[1]})' if len(msg) > 1 else '') if own is not None and abs(own) > tol else \
                f'{r["tag"]} fires ' + msg[-1]
            _finding(out, sev, 'sync', text + '.', [r['tag']], [r['uid']])
            r['status'] = sev
            bad.append(r['tag'])
        elif r.get('spread_ms') is not None and r.get('deviation_ms') is not None and r['spread_ms'] > 2 * tol:
            _finding(out, 'warning', 'sync', f'{r["tag"]}\'s offset to the sweep varies by {r["spread_ms"]:.1f} ms '
                                             '(5–95 %): trigger jitter or a drifting clock.', [r['tag']], [r['uid']])
            r['status'] = 'warning'
            bad.append(r['tag'])
    measured = [r for r in res['cameras'] if r.get('own_ms') is not None]
    if measured and not bad:
        w = max(measured, key=lambda r: abs(r['own_ms']))
        _finding(out, 'ok', 'sync', f'Every camera fires within {abs(w["own_ms"]):.2f} ms of the sync plan (worst '
                                    f'{w["tag"]}).')
    if plan.lidar is not None and ref is None and res['reason']:
        _finding(out, 'warning', 'sync', 'Sync against the sweep not checked: ' + res['reason'])
    return res


def _stamps(ts):
    return ts.stamp_ns if ts.stamp_ns is not None else ts.log_ns


# =========================================================================== extrinsics
def _bilinear(flat, H, W, fi, x, y):
    ok = (x >= 0) & (x <= W - 1) & (y >= 0) & (y <= H - 1)
    x0 = np.clip(np.floor(x), 0, W - 2).astype(np.int64)
    y0 = np.clip(np.floor(y), 0, H - 2).astype(np.int64)
    fx = np.clip(x - x0, 0, 1)
    fy = np.clip(y - y0, 0, 1)
    i = fi * (H * W) + y0 * W + x0
    v = (flat[i] * (1 - fx) * (1 - fy) + flat[i + 1] * fx * (1 - fy) + flat[i + W] * (1 - fx) * fy
         + flat[i + W + 1] * fx * fy)
    return np.where(ok, v, 0.0)


class _Scorer:
    """Mean edge attraction of the lidar depth edges projected through the camera at a rotation correction."""

    def __init__(self, cam, intr, sx, sy, Q, fi, maps, limit):
        self.ypr = (cam.yaw, cam.pitch, cam.roll)
        self.intr, self.sx, self.sy = intr, sx, sy
        self.Q, self.fi = Q, fi
        self.F, self.H, self.W = maps.shape
        self.flat = maps.reshape(-1)                 # uint8: sampled as floats point by point
        self.limit = limit

    def project(self, d, Q):
        y, p, r = self.ypr
        W2O = BODY_TO_OPTICAL @ rot_zyx(y + d[0], p + d[1], r + d[2]).T
        u, v, th, ok = self.intr.project_dirs(Q @ W2O.T)
        return u * self.sx - 0.5, v * self.sy - 0.5, ok

    def values(self, d, sel=None):
        """Each point's attraction (0..1) at correction d."""
        Q = self.Q if sel is None else self.Q[sel]
        fi = self.fi if sel is None else self.fi[sel]
        x, y, ok = self.project(np.asarray(d, float), Q)
        return np.where(ok, _bilinear(self.flat, self.H, self.W, fi, x, y), 0.0) / 255.0

    def __call__(self, d, sel=None):
        d = np.asarray(d, float)
        if np.abs(d).max() > self.limit:
            return 0.0
        n = len(self.Q) if sel is None else int(np.count_nonzero(sel)) if sel.dtype == bool else len(sel)
        if not n:
            return 0.0
        return float(self.values(d, sel).sum() / n)


def _sandwich(fn, x, clusters, h=0.05):
    """Cluster-robust standard error (deg) of the correction x maximising the mean score: H⁻¹ M H⁻¹ with H the
    Hessian of the mean and M the covariance of its gradient, the per-point gradients summed within each cluster
    (image regions per frame: neighbouring points share their errors). Infinite along a direction the edges do not
    pin down."""
    e = np.eye(3) * h
    vals = [(fn.values(x + e[k]) - fn.values(x - e[k])) / (2 * h) for k in range(3)]
    G = np.stack(vals, 1)                                        # (N, 3) per-point gradients
    N = len(G)
    if N < 10:
        return np.full(3, np.inf)
    u, inv = np.unique(clusters, return_inverse=True)
    Gc = np.stack([np.bincount(inv, weights=G[:, k], minlength=len(u)) for k in range(3)], 1)
    Mt = Gc.T @ Gc / (N * N)
    f0 = fn(x)
    Hm = np.zeros((3, 3))
    fp = [fn(x + e[i]) for i in range(3)]
    fm = [fn(x - e[i]) for i in range(3)]
    for i in range(3):
        Hm[i, i] = (fp[i] - 2 * f0 + fm[i]) / (h * h)
        for j in range(i + 1, 3):
            Hm[i, j] = Hm[j, i] = (fn(x + e[i] + e[j]) - fn(x + e[i] - e[j]) - fn(x - e[i] + e[j])
                                   + fn(x - e[i] - e[j])) / (4 * h * h)
    w, V = np.linalg.eigh(-Hm)
    w = np.maximum(w, 1e-6 * max(float(np.abs(w).max()), 1e-12))     # flat or not a peak: a vast error that way
    Hi = V @ np.diag(1.0 / w) @ V.T
    return np.sqrt(np.maximum(np.diag(Hi @ Mt @ Hi), 0.0))


def _peak_width(fn, x, px_deg=0.0):
    """Per axis, the marginal width (deg, one sigma of a Gaussian fitted to the log score) of the score's peak at x and
    whether the axis is pinned down (width under OBSERVABLE_DEG): the Hessian of log score by central differences of
    0.15°, at least one pixel (px_deg): the score samples its maps bilinearly, linear between pixel centres, so a
    shorter step measures where the peak falls within its pixel rather than its width."""
    h = max(0.15, float(px_deg))
    f0 = fn(x)
    if f0 <= 0:
        return [math.inf] * 3, [False] * 3
    L = lambda d: math.log(max(fn(d), 1e-12))          # noqa: E731
    l0 = math.log(f0)
    Hm = np.zeros((3, 3))
    e = np.eye(3) * h
    lp = [L(x + e[i]) for i in range(3)]
    lm = [L(x - e[i]) for i in range(3)]
    for i in range(3):
        Hm[i, i] = (lp[i] - 2 * l0 + lm[i]) / (h * h)
        for j in range(i + 1, 3):
            Hm[i, j] = Hm[j, i] = (L(x + e[i] + e[j]) - L(x + e[i] - e[j]) - L(x - e[i] + e[j])
                                   + L(x - e[i] - e[j])) / (4 * h * h)
    A = -Hm
    w, V = np.linalg.eigh(A)
    w = np.maximum(w, 1e-9)
    var = (V ** 2) @ (1.0 / w)
    width = np.sqrt(var)
    return [float(v) for v in width], [bool(v <= OBSERVABLE_DEG) for v in width]


def _nm(fn, x0, step, xatol=0.002):
    from scipy.optimize import minimize
    x0 = np.asarray(x0, float)
    simplex = np.vstack([x0, x0 + np.diag([step] * 3)])
    r = minimize(lambda x: -fn(x), x0, method='Nelder-Mead',
                 options={'xatol': xatol, 'fatol': 1e-7, 'maxiter': 600, 'initial_simplex': simplex})
    return np.asarray(r.x, float), -float(r.fun)


def _camera_masks(project, cam, intr, ashape, sx, sy, engine, var=None):
    """(inside the lens circle, on the own vehicle) per analysis pixel; the vehicle with var's rig housings."""
    import cv2
    ha, wa = ashape
    xs = (np.arange(wa) + 0.5) / sx
    ys = (np.arange(ha) + 0.5) / sy
    U, V = np.meshgrid(xs, ys)
    d, th = intr.unproject(U.ravel(), V.ravel())
    circle = (th <= THETA_SHARE * intr.max_theta).reshape(ha, wa)
    body = None
    if engine is not None and engine.vehicle_geom(project, var) is not None:
        W2O = BODY_TO_OPTICAL @ rot_zyx(cam.yaw, cam.pitch, cam.roll).T
        bd = engine.cube(cam.pos()).lookup(d @ W2O)[0]
        body = (np.isfinite(bd) & (bd > 0.05)).reshape(ha, wa)
        if body.any():
            body = cv2.dilate(body.astype(np.uint8), np.ones((BODY_DIL, BODY_DIL), np.uint8)) > 0
    return circle, body


def _edge_maps(smp, circle, body, tau_c, tau_f):
    """(coarse, fine) attraction maps (uint8) of one image at the analysis size. Coarse: a Gaussian of the distance
    to the nearest Canny edge (a wide basin for the grid search). Fine: the gradient magnitude of the image smoothed
    by tau_f, scaled by its 99th percentile: it peaks exactly on an edge (Canny's thinning puts an edge up to half a
    pixel to one side), so the refinement is free of that bias. Edges outside the lens circle and on the own vehicle
    are left out. None when the image shows no edges."""
    import cv2
    g = cv2.imdecode(np.frombuffer(smp['grey'], np.uint8), cv2.IMREAD_GRAYSCALE)
    if g is None:
        return None
    keep = circle.copy()
    if body is not None:
        keep &= ~body
    gb = cv2.GaussianBlur(g, (0, 0), 1.0)
    mag = np.hypot(cv2.Sobel(gb, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(gb, cv2.CV_32F, 0, 1, ksize=3))
    hi = max(float(np.percentile(mag[keep], 92)) if keep.any() else 0.0, 8.0)
    E = (cv2.Canny(gb, 0.4 * hi, hi, L2gradient=True) > 0) & keep
    if E.sum() < 20:
        return None
    dt = cv2.distanceTransform((~E).astype(np.uint8), cv2.DIST_L2, 5)
    coarse = (255.0 * np.exp(-0.5 * (dt / tau_c) ** 2)).astype(np.uint8)
    gf = cv2.GaussianBlur(g.astype(np.float32), (0, 0), tau_f)
    gm = np.hypot(cv2.Sobel(gf, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(gf, cv2.CV_32F, 0, 1, ksize=3))
    gm[~keep] = 0.0
    top = float(np.percentile(gm[keep], 99)) if keep.any() else 0.0
    if top <= 1e-6:
        return None
    fine = (255.0 * np.clip(gm / top, 0.0, 1.0)).astype(np.uint8)
    return coarse, fine


def _spin_times(P, sw, ref, sense, rate):
    """Firing times (s after the stamp) of a sweep's points from their azimuths and the spin model, for clouds without
    a time per point: the head faces the first point at a start stamp (the last at an end stamp) and turns at rate Hz
    in the sense given."""
    az = np.degrees(np.arctan2(P[:, 1], P[:, 0]))
    if ref == 'start':
        return ((sense * (az - (sw['az_first'] or 0.0))) % 360.0) / (360.0 * rate)
    return -((sense * ((sw['az_last'] or 0.0) - az)) % 360.0) / (360.0 * rate)


def _frame_points(tg, topic, t_c_ns, T_s, ref, R_L, p_L, motion, base_ns, sense=-1, rate=10.0):
    """The lidar's depth edges and kept points around one camera exposure, in the vehicle frame at that exposure: the
    points measured within half a rotation of it, by their own times or, without, by the times the spin model gives
    their azimuths. Also each edge point's |time difference| to the exposure (s)."""
    sws = tg['clouds'].get(topic) or []
    if not sws:
        return None
    t_c = (t_c_ns - base_ns) / 1e9
    Es, Et, Ks, Kt = [], [], [], []
    for s in sws:
        if s['has_time'] and s['E_dt'] is not None:
            e_dt, k_dt = s['E_dt'].astype(np.float64), s['K_dt'].astype(np.float64)
        else:
            e_dt = _spin_times(s['E'].astype(np.float64), s, ref, sense, rate)
            k_dt = _spin_times(s['K'].astype(np.float64), s, ref, sense, rate)
        te = (s['stamp_ns'] - base_ns) / 1e9 + e_dt
        tk = (s['stamp_ns'] - base_ns) / 1e9 + k_dt
        me, mk = np.abs(te - t_c) <= T_s / 2, np.abs(tk - t_c) <= T_s / 2
        Es.append(s['E'][me])
        Et.append(te[me])
        Ks.append(s['K'][mk])
        Kt.append(tk[mk])
    E = np.concatenate(Es).astype(np.float64) if Es else np.zeros((0, 3))
    K = np.concatenate(Ks).astype(np.float64) if Ks else np.zeros((0, 3))
    te, tk = np.concatenate(Et) if Et else np.zeros(0), np.concatenate(Kt) if Kt else np.zeros(0)
    E = motion.apply(E @ R_L.T + p_L, te, t_c)
    K = motion.apply(K @ R_L.T + p_L, tk, t_c)
    return E, K, np.abs(te - t_c)


def _on_mask(Q, W2O, intr, sx, sy, mask):
    """Which points (camera-relative, vehicle frame) project at the plan pose into the analysis-size mask."""
    out = np.zeros(len(Q), bool)
    if not len(Q):
        return out
    u, v, _, ok = intr.project_dirs(Q @ W2O.T)
    ha, wa = mask.shape
    xi, yi = np.floor(u * sx).astype(np.int64), np.floor(v * sy).astype(np.int64)
    ok &= (xi >= 0) & (xi < wa) & (yi >= 0) & (yi < ha)
    out[ok] = mask[yi[ok], xi[ok]]
    return out


def _visible(Qe, Qk, W2O, intr, sx, sy, ha, wa, cube):
    """Which edge points (and kept points) the camera can see at the plan pose: in front, inside the lens circle and
    the image, not behind the own vehicle, not behind nearer lidar points (a coarse depth buffer)."""
    def proj(Q):
        u, v, th, ok = intr.project_dirs(Q @ W2O.T)
        ok &= th <= THETA_SHARE * intr.max_theta
        return u * sx - 0.5, v * sy - 0.5, ok
    xe, ye, oe = proj(Qe)
    xk, yk, ok_ = proj(Qk)
    re, rk = np.linalg.norm(Qe, axis=1), np.linalg.norm(Qk, axis=1)
    if cube is not None:
        for Q, r, o in ((Qe, re, oe), (Qk, rk, ok_)):
            if len(Q):
                bd = cube.lookup(Q / np.maximum(r, 1e-9)[:, None])[0]
                o &= ~(np.isfinite(bd) & (bd > 0.05) & (bd < r - 0.05))
    cell = 8.0
    gw, gh = int(wa / cell) + 2, int(ha / cell) + 2
    z = np.full(gw * gh, np.inf)
    for x, y, r, o in ((xe, ye, re, oe), (xk, yk, rk, ok_)):
        if o.any():
            ci = (np.clip((x[o] / cell).astype(int), 0, gw - 1) + np.clip((y[o] / cell).astype(int), 0, gh - 1) * gw)
            np.minimum.at(z, ci, r[o])
    zz = z.reshape(gh, gw)
    zmin = zz.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            zmin = np.minimum(zmin, np.roll(np.roll(zz, dy, 0), dx, 1))
    zmin = zmin.ravel()

    def front(x, y, r, o):
        ci = np.clip((x / cell).astype(int), 0, gw - 1) + np.clip((y / cell).astype(int), 0, gh - 1) * gw
        return o & (r <= zmin[ci] * 1.05 + 0.3)
    return front(xe, ye, re, oe), front(xk, yk, rk, ok_)


def _extrinsic(project, var, cam, topic, lid, lid_topic, scan, motion, st, conv_ms, engine, out_overlays) -> dict:
    """The targetless rotation check of one camera against the lidar (see the module docstring)."""
    from .lens import resolve
    mod = project.module_for(cam)
    intr = resolve(mod, project.lens_for(cam))
    W, H = int(mod.width_px), int(mod.height_px)
    res = {'uid': cam.uid, 'tag': cam.tag, 'topic': topic, 'lidar': lid.tag, 'frames': 0, 'points': 0,
           'delta_deg': None, 'stderr_deg': None, 'agree': None, 'sharpness_pct': None, 'score_plan': None,
           'score_corrected': None, 'confidence': None, 'verdict': 'not checked', 'reason': '', 'apply': False,
           'dt_ms': None, 'compensation': motion.label, 'plan_deg': [cam.yaw, cam.pitch, cam.roll]}
    smps = [(k, tg['images'][topic]) for k, tg in enumerate(scan.targets) if topic in tg['images']]
    if not smps:
        res['reason'] = 'no sampled image'
        return res
    shapes = {}
    for _, s in smps:
        shapes[s['ashape']] = shapes.get(s['ashape'], 0) + 1
    ashape = max(shapes, key=shapes.get)
    w0, h0 = next(s['size'] for _, s in smps if s['ashape'] == ashape)
    if abs(w0 * H - h0 * W) > 0.01 * max(w0 * H, h0 * W):
        res['reason'] = f'the images ({w0} × {h0}) do not have the module\'s aspect ({W} × {H})'
        return res
    ha, wa = ashape
    sx, sy = wa / W, ha / H
    m = project.sensor_model_for(lid)
    T_s = 1.0 / m.rate_hz if m.rate_hz > 0 else 0.1
    R_L, p_L = rot_zyx(lid.yaw, lid.pitch, lid.roll), lid.pos()
    lts = scan.topics[lid_topic]
    ref = st.lidar_stamp if st.lidar_stamp != 'auto' else lts.stamp_ref
    circle, body = _camera_masks(project, cam, intr, ashape, sx, sy, engine, var)
    cube = engine.cube(cam.pos()) if engine is not None and engine.vehicle_geom(project, var) is not None else None
    ppd = intr.f_px * sx * math.pi / 180.0                     # analysis px per degree at the image centre
    tau_c = float(np.clip(1.0 * ppd, 2.0, 16.0))
    tau_f = float(np.clip(0.15 * ppd, 1.0, 4.0))
    hide = None                                         # lidar edges near the vehicle's (masked) pixels: left out,
    if body is not None and body.any():                 # out to where the fine map still feels the mask (3 τ)
        import cv2
        k = 2 * int(math.ceil(3 * tau_f)) + 1
        hide = cv2.dilate(body.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
    W2O = BODY_TO_OPTICAL @ rot_zyx(cam.yaw, cam.pitch, cam.roll).T
    pos = cam.pos()
    lo = hi = None
    mesh = project.vehicle.mesh
    if mesh is not None and getattr(mesh, 'n_tris', 0):
        lo, hi = np.asarray(mesh.bbox[0], float) - 0.15, np.asarray(mesh.bbox[1], float) + 0.15
    frames, Qs, fis, mc, mf, dts = [], [], [], [], [], []
    for k, smp in smps:
        if smp['ashape'] != ashape:
            continue
        t_c = smp['stamp_ns'] + int(round(conv_ms * 1e6))
        got = _frame_points(scan.targets[k], lid_topic, t_c, T_s, ref, R_L, p_L, motion, scan.start_ns,
                            1 if m.rotation == 'ccw' else -1, m.rate_hz if m.rate_hz > 0 else 10.0)
        if got is None:
            continue
        E, K, adt = got
        if lo is not None:
            keep = ~np.all((E >= lo) & (E <= hi), axis=1)
            E, adt = E[keep], adt[keep]
            K = K[~np.all((K >= lo) & (K <= hi), axis=1)]
        Qe, Qk = E - pos, K - pos
        ve, vk = _visible(Qe, Qk, W2O, intr, sx, sy, ha, wa, cube)
        if hide is not None:                            # an edge the image cannot show (its edges there are left out)
            ve &= ~_on_mask(Qe, W2O, intr, sx, sy, hide)    # would pull the rotation away from the vehicle
        Qe, adt = Qe[ve], adt[ve]
        if len(Qe) < 20:
            continue
        dtm = float(np.median(adt))
        if len(Qe) > VIEW_MAX:
            Qe = Qe[np.linspace(0, len(Qe) - 1, VIEW_MAX).astype(int)]
        maps = _edge_maps(smp, circle, body, tau_c, tau_f)
        if maps is None:
            continue
        fi = len(frames)
        Qs.append(Qe)
        fis.append(np.full(len(Qe), fi))
        mc.append(maps[0])
        mf.append(maps[1])
        dts.append(dtm)
        Qk = Qk[vk]
        if len(Qk) > SHOW_MAX:
            Qk = Qk[np.linspace(0, len(Qk) - 1, SHOW_MAX).astype(int)]
        frames.append({'t_s': (t_c - scan.start_ns) / 1e9, 'jpeg': smp['jpeg'], 'tsize': smp['tsize'],
                       'Q': Qk.astype(np.float32), 'rng': np.linalg.norm(Qk, axis=1).astype(np.float32),
                       'edges': len(Qe)})
    res['frames'] = len(frames)
    res['points'] = int(sum(len(q) for q in Qs))
    res['dt_ms'] = float(np.median(dts) * 1000.0) if dts else None
    out_overlays[cam.uid] = {'tag': cam.tag, 'frames': frames, 'module_size': (W, H), 'plan': [cam.yaw, cam.pitch,
                                                                                               cam.roll],
                             'delta': None}
    if not frames:
        res['reason'] = 'no lidar depth edges in the camera\'s view'
        return res
    Q = np.concatenate(Qs)
    fi = np.concatenate(fis)
    S = st.search_deg
    coarse = _Scorer(cam, intr, sx, sy, Q, fi, np.stack(mc), S + 0.5)
    fine = _Scorer(cam, intr, sx, sy, Q, fi, np.stack(mf), S + 0.5)
    del mc, mf
    sp = PRIOR_SCALE * S                                # a weak prior: small mounting errors are likelier

    def post(fn, sel=None):
        return lambda d: fn(d, sel) * math.exp(-float(np.dot(d, d)) / (2 * sp * sp))
    sel = np.linspace(0, len(Q) - 1, min(len(Q), 15000)).astype(int)
    n = int(math.ceil(S / 0.5))
    g = np.linspace(-S, S, 2 * n + 1)
    grid = np.array(np.meshgrid(g, g, g, indexing='ij')).reshape(3, -1).T
    pc = post(coarse, sel)
    sc = np.array([pc(d) for d in grid])
    picks = []
    step = g[1] - g[0] if len(g) > 1 else S
    for i in np.argsort(-sc):
        if all(np.abs(grid[i] - grid[j]).max() > 1.5 * step for j in picks):
            picks.append(i)
        if len(picks) == 3:
            break
    starts = []
    for i in picks:                                     # refine the best grid points on the coarse map …
        x, v = _nm(post(coarse, sel), grid[i], step / 2, 0.02)
        starts.append((v, x))
    starts.sort(key=lambda r: -r[0])
    best, bval = None, -1.0
    for v0, x in starts[:2]:                            # … and the best two on the fine map with every point
        if best is not None and v0 < 0.9 * starts[0][0]:
            break
        x, v = _nm(post(fine), x, 0.1)
        if v > bval:
            best, bval = x, v
    s0, s1 = fine(np.zeros(3)), fine(best)
    res['score_plan'], res['score_corrected'] = s0, s1
    px_deg = 1.0 / max(ppd, 1e-9)               # one pixel's angle at the image centre
    width, observable = _peak_width(fine, best, px_deg)
    xs, ys, _ = fine.project(best, Q)
    clusters = fi * 64 + np.clip((xs / wa * 8).astype(int), 0, 7) * 8 + np.clip((ys / ha * 8).astype(int), 0, 7)
    se_sw = _sandwich(fine, best, clusters)
    F = len(frames)
    per = []
    for f in range(F):
        m_ = fi == f
        if m_.sum() >= 20:
            x, _ = _nm(post(fine, m_), best, 0.1, 0.005)
            per.append(x)
    per = np.array(per) if per else np.zeros((0, 3))
    tol = st.tolerance_deg
    obs = np.array(observable)
    stderr = se_sw.copy()
    agree = None
    if len(per) >= 2:
        sig = 1.4826 * np.median(np.abs(per - best), axis=0)
        stderr = np.maximum(stderr, sig / math.sqrt(len(per)))
        agree = float(np.mean(np.all((np.abs(per - best) <= tol) | ~obs, axis=1)))
    obs &= stderr <= 2 * tol
    observable = [bool(o) for o in obs]
    res.update({'delta_deg': [float(v) for v in best], 'stderr_deg': [float(v) if np.isfinite(v) else None
                                                                       for v in stderr], 'agree': agree,
                'width_deg': [float(v) for v in width], 'observable': [bool(o) for o in observable],
                'per_frame_deg': per.tolist()})
    se = float(np.max(stderr[obs])) if obs.any() else math.inf
    if not obs.any() or px_deg > tol:            # images coarser than the tolerance cannot judge it
        conf = 'low'
    elif F >= 3 and agree is not None and agree >= 0.6 and se <= tol / 3 and s1 > s0 - 1e-9:
        conf = 'high'
    elif F >= 2 and agree is not None and agree >= 0.4 and se <= tol / 1.5:
        conf = 'medium'
    else:
        conf = 'low'
    res['confidence'] = conf
    big = float(np.abs(np.where(obs, best, 0.0)).max())
    names = ('yaw', 'pitch', 'roll')
    if conf == 'low':
        res['verdict'] = 'uncertain'
        why = []
        if F < 3:
            why.append(f'{F} frame{"s" * (F != 1)} with edges')
        if agree is not None and agree < 0.6:
            why.append(f'{agree * 100:.0f} % of the frames agree')
        if not obs.any():
            why.append('no axis is pinned down by the edges in view')
        if px_deg > tol:
            why.append(f'the images resolve {px_deg:.2f}° per pixel, coarser than the {tol:g}° tolerance')
        elif se > tol / 1.5 and obs.any():
            why.append(f'±{se:.2f}° standard error')
        res['reason'] = ', '.join(why)
    elif big > tol:
        res['verdict'] = 'rotated'
    else:
        res['verdict'] = 'ok'
    if conf != 'low' and not obs.all():
        res['reason'] = (', '.join(n_ for n_, o in zip(names, observable) if not o) +
                         ' not pinned down by the edges in view (left as planned)')
    res['apply'] = conf in ('high', 'medium') and res['verdict'] == 'rotated'
    best = np.where(obs, best, 0.0)
    res['correction_deg'] = [float(v) for v in best]
    res['corrected_deg'] = [cam.yaw + best[0], cam.pitch + best[1], cam.roll + best[2]]
    Rp = rot_zyx(cam.yaw, cam.pitch, cam.roll)
    res['rotvec_vehicle_deg'] = [float(v) for v in np.degrees(_R_rotvec(rot_zyx(*res['corrected_deg']) @ Rp.T))]
    out_overlays[cam.uid]['delta'] = res['correction_deg']
    return res


def _extrinsics(project, var, rev, scan, motion, st, plan, engine, out, overlays, progress, cancel) -> list:
    from .timing import TimingSettings
    tm = TimingSettings.from_dict(getattr(project.analysis, 'timing', None))
    conv_ms = stamp_offset_ms(st.camera_stamp, tm.readout_ms, tm.exposure_ms)
    lid = None
    if plan.lidar is not None:
        lid = var.sensor(plan.lidar.uid)
    if lid is None or lid.tag not in rev:
        lid = next((s for s in var.sensors if s.visible and s.kind == 'lidar' and s.tag in rev), None)
    rows = []
    cams = [c for c in var.cameras if c.visible]
    if lid is None:
        for c in cams:
            rows.append({'uid': c.uid, 'tag': c.tag, 'topic': rev.get(c.tag), 'verdict': 'not checked',
                         'reason': 'no lidar topic', 'delta_deg': None, 'confidence': None, 'apply': False})
        if cams:
            _finding(out, 'warning', 'extrinsics', 'Camera rotations not checked: no lidar of the variant has a topic '
                                                   'in the log.')
        return rows
    tol = st.tolerance_deg
    for i, c in enumerate(cams):
        if cancel is not None and cancel():
            return None
        if progress is not None:
            progress(i / max(len(cams), 1), f'camera {c.tag}')
        topic = rev.get(c.tag)
        if topic is None:
            rows.append({'uid': c.uid, 'tag': c.tag, 'topic': None, 'verdict': 'not checked', 'reason': 'no topic',
                         'delta_deg': None, 'confidence': None, 'apply': False})
            continue
        r = _extrinsic(project, var, c, topic, lid, rev[lid.tag], scan, motion, st, conv_ms, engine, overlays)
        rows.append(r)
        if r['verdict'] == 'rotated':
            d, obs = r['delta_deg'], r['observable']
            sev = 'fault' if r['confidence'] == 'high' else 'warning'
            axes = ', '.join(f'{n} {d[k]:+.2f}°' for k, n in enumerate(('yaw', 'pitch', 'roll')) if obs[k])
            loose = [n for k, n in enumerate(('yaw', 'pitch', 'roll')) if not obs[k]]
            agree = f', {r["agree"] * 100:.0f} % agree' if r.get('agree') is not None else ''
            _finding(out, sev, 'extrinsics',
                     f'{c.tag} is rotated against the plan (relative to {lid.tag}): {axes}'
                     + (f'; {" and ".join(loose)} not pinned down by the edges in view' if loose else '')
                     + f' (tolerance {tol:g}°, {r["confidence"]} confidence, {r["frames"]} frames{agree}).',
                     [c.tag], [c.uid])
        elif r['verdict'] == 'uncertain':
            _finding(out, 'warning', 'extrinsics', f'{c.tag}: rotation not determined ({r["reason"]}).', [c.tag],
                     [c.uid])
        elif r['verdict'] == 'not checked' and r.get('reason'):
            _finding(out, 'warning', 'extrinsics', f'{c.tag}: rotation not checked ({r["reason"]}).', [c.tag],
                     [c.uid])
    okc = [r for r in rows if r['verdict'] == 'ok']
    if okc:
        w = max(okc, key=lambda r: max(abs(v) for v in r['delta_deg']))
        _finding(out, 'ok', 'extrinsics', f'{", ".join(r["tag"] for r in okc)} match the plan within {tol:g}° '
                                          f'(largest: {w["tag"]} {max(abs(v) for v in w["delta_deg"]):.2f}°).')
    conf = [r for r in rows if r.get('confidence') in ('high', 'medium') and r.get('rotvec_vehicle_deg')]
    if len(conf) >= 3:
        V = np.array([r['rotvec_vehicle_deg'] for r in conf])
        mean = V.mean(0)
        if np.linalg.norm(mean) > tol and np.linalg.norm(V - mean, axis=1).max() < 0.5 * np.linalg.norm(mean):
            _finding(out, 'warning', 'extrinsics',
                     f'{len(conf)} cameras show the same rotation ({np.linalg.norm(mean):.2f}° about the vehicle '
                     f'axis {", ".join(f"{v:+.2f}" for v in mean / np.linalg.norm(mean))}): the lidar {lid.tag} is '
                     'more likely rotated than every camera.', [lid.tag], [lid.uid])
    if any(r.get('dt_ms') for r in rows) and motion.kind != 'odometry':
        dtm = float(np.median([r['dt_ms'] for r in rows if r.get('dt_ms')]))
        v = motion.speed_at(scan.duration_s / 2) if motion.speed is not None else None
        tail = (f'; at the drive\'s {v:.1f} m/s that is {v * dtm / 10:.1f} cm of travel' if v else
                '; at 10 m/s that would be ' + f'{dtm:.0f} cm of travel')
        _finding(out, 'ok' if (v is not None and v * dtm / 1000 < 0.02) else 'warning', 'extrinsics',
                 f'Lidar points were matched to images {dtm:.1f} ms apart (median) without travel compensation '
                 f'({motion.label}){tail}.')
    return rows


# =========================================================================== GNSS and IMU
def _imu_check(project, s, ts, scan, motion, out) -> dict:
    d = ts.data
    r = {'uid': s.uid if s else '', 'tag': s.tag if s else ts.topic, 'topic': ts.topic, 'samples': 0,
         'stationary_s': 0.0, 'gravity': None, 'gyro_bias_dps': None, 'tilt_deg': None, 'status': 'ok'}
    if not d or len(d.get('t_ns', [])) < 10:
        return r
    t = (d['t_ns'] - scan.start_ns) / 1e9
    o = np.argsort(t)
    t = t[o]
    w = np.stack([d['gx'], d['gy'], d['gz']], 1)[o]
    a = np.stack([d['ax'], d['ay'], d['az']], 1)[o]
    r['samples'] = int(len(t))
    an = np.linalg.norm(a, axis=1)
    wn = np.linalg.norm(w, axis=1)
    win = np.floor((t - t[0]) / 0.5).astype(int)
    nw = int(win.max()) + 1
    cnt = np.bincount(win, minlength=nw)
    mw = np.bincount(win, weights=wn, minlength=nw) / np.maximum(cnt, 1)
    ma = np.bincount(win, weights=an, minlength=nw) / np.maximum(cnt, 1)
    sa = np.sqrt(np.maximum(np.bincount(win, weights=an ** 2, minlength=nw) / np.maximum(cnt, 1) - ma ** 2, 0))
    still = (cnt >= 5) & (mw < 0.02) & (sa < 0.1 * np.maximum(ma / 9.81, 0.1))
    if motion.speed is not None:
        tw = t[0] + (np.arange(nw) + 0.5) * 0.5
        sp = np.array([motion.speed_at(x) or 0.0 for x in tw])
        still &= sp < 0.2
    sel = still[win]
    r['stationary_s'] = float(still.sum() * 0.5)
    tag = r['tag']
    uids = [r['uid']] if r['uid'] else []
    if sel.sum() >= 5:
        g = float(np.median(an[sel]))
        r['gravity'] = g
        bias = w[sel].mean(0)
        r['gyro_bias_dps'] = [float(v) for v in np.degrees(bias)]
        if s is not None:
            up = rot_zyx(s.yaw, s.pitch, s.roll).T @ np.array([0.0, 0.0, 1.0])
            m_ = a[sel].mean(0)
            c = float(np.dot(m_ / max(np.linalg.norm(m_), 1e-9), up))
            r['tilt_deg'] = math.degrees(math.acos(max(-1.0, min(1.0, c))))
        if abs(g - 1.0) < 0.1:
            _finding(out, 'fault', 'gnss_imu', f'{tag}: accelerations read {g:.3f} at rest: in g, not m/s².', [tag],
                     uids)
            r['status'] = 'fault'
        elif abs(g - 9.81) > 0.15:
            _finding(out, 'warning', 'gnss_imu', f'{tag}: gravity reads {g:.2f} m/s² at rest (9.81 expected): a scale '
                                                 'error or a bias.', [tag], uids)
            r['status'] = 'warning'
        bmax = float(np.abs(np.degrees(bias)).max())
        if bmax > 1.0:
            _finding(out, 'warning', 'gnss_imu', f'{tag}: gyro bias at rest up to {bmax:.2f}°/s.', [tag], uids)
            r['status'] = 'warning' if r['status'] == 'ok' else r['status']
        if r['tilt_deg'] is not None and r['tilt_deg'] > 3.0 and r['status'] != 'fault':
            _finding(out, 'warning', 'gnss_imu', f'{tag}: gravity points {r["tilt_deg"]:.1f}° off the planned '
                                                 'mounting\'s down (the IMU is mounted otherwise, or the vehicle stood '
                                                 'on a slope).', [tag], uids)
        if r['status'] == 'ok':
            _finding(out, 'ok', 'gnss_imu', f'{tag}: {g:.3f} m/s² at rest, gyro bias {bmax:.3f}°/s, over '
                                            f'{r["stationary_s"]:.1f} s standing still.', [tag], uids)
    else:
        _finding(out, 'ok', 'gnss_imu', f'{tag}: no period standing still: gravity and gyro bias not checked.', [tag],
                 uids)
    return r


def _fix_check(s, ts, out) -> dict:
    d = ts.data
    tag = s.tag if s is not None else ts.topic
    r = {'uid': s.uid if s is not None else '', 'tag': tag, 'topic': ts.topic, 'fixes': 0, 'status': 'ok'}
    if not d or not len(d.get('t_ns', [])):
        return r
    st_ = d['status']
    n = len(st_)
    r['fixes'] = int(n)
    r['no_fix_share'] = float(np.mean(st_ < 0))
    r['fix_share'] = float(np.mean(st_ == 0))
    r['sbas_share'] = float(np.mean(st_ == 1))
    r['rtk_share'] = float(np.mean(st_ == 2))
    known = d['cov_type'] > 0
    if known.any():
        h = np.sqrt(np.maximum((d['cov_e'][known] + d['cov_n'][known]) / 2, 0))
        r['h_sigma_m'] = float(np.median(h))
        r['v_sigma_m'] = float(np.median(np.sqrt(np.maximum(d['cov_u'][known], 0))))
    uids = [r['uid']] if r['uid'] else []
    if r['no_fix_share'] > 0.05:
        r['status'] = 'warning'
        _finding(out, 'warning', 'gnss_imu', f'{tag}: no fix in {r["no_fix_share"] * 100:.0f} % of {n} messages.',
                 [tag], uids)
    else:
        txt = f'{tag}: a fix in {100 - r["no_fix_share"] * 100:.0f} % of {n} messages'
        if r['rtk_share'] > 0:
            txt += f' ({r["rtk_share"] * 100:.0f} % with ground-based corrections)'
        if r.get('h_sigma_m') is not None:
            txt += f', {r["h_sigma_m"] * 100:.1f} cm horizontal (1σ, as reported)'
        _finding(out, 'ok', 'gnss_imu', txt + '.', [tag], uids)
    return r


# =========================================================================== run
def _variant(project, variant, st):
    if variant is not None and not isinstance(variant, str):
        return variant
    uid = variant if isinstance(variant, str) else st.variant_uid
    return (project.variant_by_uid(uid) if uid else None) or project.variant


def run(project, path=None, settings: DriveLogSettings | None = None, progress=None, cancel=None, engine=None,
        variant=None, scan: Scan | None = None) -> DriveLogResult | None:
    """Check a drive log (a bag or a nuScenes folder; the settings' path by default) against a variant's plan (the
    settings' variant, else the active one). engine: the analysis engine whose cube maps give the vehicle's blocking
    (of the lidar's points, of each camera's view); one is made when none is given, so a script gets the answer the
    page gets. A scan of the same files from an earlier run is used again (pass it as scan). None when cancelled."""
    t_start = time.perf_counter()
    if engine is None:                  # without the own vehicle, lidar edges a mirror camera's own mirror hides
        from .analysis import Engine    # (the roof lidar sees over it) were matched to the mirror's image edge: a
        engine = Engine()               # correctly mounted MR came out 0.4° off and 'rotated'
    st = settings or DriveLogSettings.from_dict(getattr(project.analysis, 'drivelog', None))
    path = norm_path(path or st.path)
    if not path:
        raise ValueError('No drive log given: open a ROS 2 bag or a nuScenes folder.')
    var = _variant(project, variant, st)

    def prog(lo, hi):
        if progress is None:
            return None
        return lambda f, text='': progress(lo + (hi - lo) * max(0.0, min(1.0, f)), text)
    if scan is None or scan.key != log_key(path) or scan.n_frames != st.frames_per_camera or \
            scan.sync_sweeps != st.sync_sweeps:
        scan = scan_log(path, st.frames_per_camera, st.sync_sweeps, prog(0.0, 0.75), cancel)
        if scan is None:
            return None
    if cancel is not None and cancel():
        return None
    from . import timing as TM
    out = []
    mapping = mapping_for(var, scan, st.overrides(path))
    rev = {g: t for t, g in mapping.items()}
    topics = [{'topic': t, 'type': ts.type, 'kind': ts.kind, 'messages': ts.count, 'bytes': ts.bytes,
               'frame_id': ts.frame_id, 'tag': mapping.get(t, '')}
              for t, ts in sorted(scan.topics.items()) if t != 'ego_pose']
    # ---- 1. streams (a drive Vantage wrote is judged by how it was written: rate and image size)
    plan = TM.plan(project, var)
    written = vantage_dataset(path)
    if written is not None:
        ws = written.get('settings') or {}
        _finding(out, 'ok', 'streams',
                 f'A sample dataset Vantage wrote ({written.get("variant", "")}, {ws.get("scene", "")}, '
                 f'{ws.get("duration_s", "?")} s): cameras are checked at the rate and size it was written with '
                 f'({"one frame per lidar rotation" if ws.get("camera_rate", "lidar") == "lidar" else "the module rate"}, '
                 f'{float(ws.get("image_scale") or 1.0):g} of the sensor\'s size), not the modules\'.')
    streams = []
    for uid, tag, kind, obj in _sensors(var):
        rec = recorded_camera(written, project, obj, plan.period_ms) if (written is not None and kind == 'camera') \
            else None
        e = _stream(project, var, uid, tag, kind, obj, rev.get(tag), scan, st, engine, rec)
        _judge_stream(e, st, scan, out)
        streams.append(e)
    for t in topics:
        n_bad = scan.topics[t['topic']].info.get('decode_errors', 0)
        if n_bad and t['tag']:
            _finding(out, 'warning', 'streams', f'{t["tag"]}: {n_bad} message{"s" * (n_bad != 1)} on {t["topic"]} '
                                                'would not decode and were skipped.', [t['tag']])
    for t in topics:
        if not t['tag'] and t['kind'] in ('image', 'points'):
            _finding(out, 'warning', 'mapping', f'{t["topic"]} ({t["messages"]} {"images" if t["kind"] == "image" else "clouds"}) '
                                                'is mapped to no sensor of the variant.')
    ok_streams = [e for e in streams if e['status'] == 'ok' and e['topic']]
    if ok_streams:
        _finding(out, 'ok', 'streams', f'{len(ok_streams)} of {len(streams)} streams healthy: '
                                       f'{", ".join(e["tag"] for e in ok_streams)}.')
    gaps = {e['tag']: e.get('gaps', []) for e in streams if e['topic']}
    # ---- 2. data rate
    data = _data(project, var, streams, scan, out)
    # ---- 3. sync
    sync = _sync(project, var, plan, rev, scan, st, out)
    # ---- 4. extrinsics
    motion = Motion.from_scan(project, var, scan, mapping)
    overlays = {}
    ext = _extrinsics(project, var, rev, scan, motion, st, plan, engine, out, overlays, prog(0.75, 0.98), cancel)
    if ext is None:
        return None
    # ---- 5. GNSS and IMU
    imu, gnss = [], []
    for t, ts in scan.topics.items():
        if ts.kind == 'imu':
            imu.append(_imu_check(project, var_sensor(var, mapping.get(t)), ts, scan, motion, out))
        elif ts.kind == 'fix':
            gnss.append(_fix_check(var_sensor(var, mapping.get(t)), ts, out))
    order = {s: i for i, s in enumerate(SEVERITIES)}
    out.sort(key=lambda f: order.get(f['severity'], 9))
    for L in sync['lidars']:
        L.pop('_t', None), L.pop('_phi', None), L.pop('_base_ns', None)
    n_bad = sum(e['status'] != 'ok' for e in streams)
    drops = [e for e in streams if e.get('drops')]
    synced = [r for r in sync['cameras'] if r.get('own_ms') is not None]
    rot = [r for r in ext if r.get('delta_deg') is not None and r.get('confidence') in ('high', 'medium')]
    metrics = {
        'streams': len(streams), 'streams_ok': len(streams) - n_bad, 'streams_faulty': n_bad,
        'missing': [e['tag'] for e in streams if e['topic'] is None and e['kind'] != 'gnss'],
        'drops': int(sum(e['drops'] for e in drops)),
        'worst_drop': (max(drops, key=lambda e: e['drop_share'])['tag'] if drops else None),
        'worst_drop_share': max((e['drop_share'] for e in drops), default=0.0),
        'worst_sync_ms': (max((abs(r['own_ms']) for r in synced), default=None)),
        'worst_sync_tag': (max(synced, key=lambda r: abs(r['own_ms']))['tag'] if synced else None),
        'lidar_phase_error_deg': sync['lidar']['phase_error_deg'] if sync.get('lidar') else None,
        'worst_rotation_deg': (max((max(abs(v) for v in r['delta_deg']) for r in rot), default=None)),
        'worst_rotation_tag': (max(rot, key=lambda r: max(abs(v) for v in r['delta_deg']))['tag'] if rot else None),
        'corrections': [r['tag'] for r in ext if r.get('apply')],
        'gb_per_hour': data['totals'].get('gb_h'), 'planned_gb_per_hour': data['totals'].get('planned_gb_h'),
        'faults': sum(f['severity'] == 'fault' for f in out),
        'warnings': sum(f['severity'] == 'warning' for f in out)}
    span = {'start_ns': scan.start_ns, 'end_ns': scan.end_ns, 'duration_s': scan.duration_s,
            'messages': sum(ts.count for t, ts in scan.topics.items() if t != 'ego_pose'),
            'bytes': data['totals']['bytes']}
    res = DriveLogResult(st, path, scan.source, var.name, var.uid, span, topics, mapping, streams, data, sync, ext,
                         imu, gnss, out, metrics, overlays, gaps, list(scan.notes), scan)
    res.seconds = time.perf_counter() - t_start
    metrics['seconds'] = res.seconds
    metrics['scan_seconds'] = scan.seconds
    if progress is not None:
        progress(1.0, 'done')
    return res


# =========================================================================== corrections
def corrected_variant(project, res: DriveLogResult, tags=None):
    """A copy of the checked variant named '<name> · drive-log corrected' with the rotation corrections applied to the
    cameras found rotated beyond the tolerance with high or medium confidence (or to the given tags), on the axes the
    edges pin down; its cameras and sensors get new uids, a corrected camera and its mirror partner are unlinked (the
    as-built rig is not symmetric). None when there is nothing to apply."""
    from .project import Variant, new_uid
    src = project.variant_by_uid(res.variant_uid)
    if src is None:
        return None
    pick = {r['tag']: r for r in res.extrinsics if r.get('delta_deg') is not None and
            (r['tag'] in tags if tags is not None else r.get('apply'))}
    if not pick:
        return None
    v = Variant.from_dict(src.to_dict())
    v.uid = new_uid()
    v.name = f'{src.name} · drive-log corrected'
    umap = {}
    for c in v.cameras:
        umap[c.uid] = new_uid()
    for c in v.cameras:
        c.uid = umap[c.uid]
        c.link_uid = umap.get(c.link_uid, '')
    for s in v.sensors:
        s.uid = new_uid()
    v.seams = {'|'.join(umap.get(u, u) for u in k.split('|')): a for k, a in v.seams.items()}
    done = []
    for c in v.cameras:
        r = pick.get(c.tag)
        if r is None:
            continue
        d = r.get('correction_deg') or r['delta_deg']
        c.yaw, c.pitch, c.roll = euler_zyx(rot_zyx(c.yaw + d[0], c.pitch + d[1], c.roll + d[2]))
        c.yaw, c.pitch, c.roll = round(c.yaw, 4), round(c.pitch, 4), round(c.roll, 4)
        if c.link_uid:
            partner = next((x for x in v.cameras if x.uid == c.link_uid), None)
            if partner is not None:
                partner.link_uid = ''
            c.link_uid = ''
        done.append(f'{c.tag} yaw {d[0]:+.2f}°, pitch {d[1]:+.2f}°, roll {d[2]:+.2f}°')
    v.note = (f'Camera rotations corrected from the drive log {os.path.basename(res.path)} (relative to the lidar '
              f'{next((r["lidar"] for r in res.extrinsics if r.get("lidar")), "")}): ' + '; '.join(done) + '.')
    return v


# =========================================================================== output
def _r(v, d=3):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return round(f, d) + 0.0 if math.isfinite(f) else None


def _rl(v, d=3):
    return None if v is None else [_r(x, d) for x in v]


def summary(res: DriveLogResult) -> dict:
    """Compact figures (the assistant, the batch report header): no images, no paths beyond the log's name."""
    m = res.metrics
    return {
        'log': os.path.basename(res.path), 'source': res.source, 'variant': res.variant,
        'duration_s': _r(res.span['duration_s'], 2), 'messages': res.span['messages'],
        'streams_ok': m['streams_ok'], 'streams': m['streams'], 'missing': m['missing'],
        'dropped_frames': m['drops'], 'worst_drop': m['worst_drop'],
        'worst_drop_pct': _r(m['worst_drop_share'] * 100, 2),
        'gb_per_hour': _r(m['gb_per_hour'], 2), 'planned_gb_per_hour': _r(m['planned_gb_per_hour'], 2),
        'lidar_phase_error_deg': _r(m['lidar_phase_error_deg'], 2),
        'worst_sync_ms': _r(m['worst_sync_ms'], 2), 'worst_sync_camera': m['worst_sync_tag'],
        'worst_rotation_deg': _r(m['worst_rotation_deg'], 3), 'worst_rotation_camera': m['worst_rotation_tag'],
        'streams_table': [{'sensor': e['tag'], 'kind': e['kind'], 'topic': e['topic'], 'status': e['status'],
                           'rate_hz': _r(e.get('rate_hz'), 3), 'expected_hz': e.get('expected_hz'),
                           'drops': e.get('drops', 0)} for e in res.streams],
        'sync': [{'camera': r['tag'], 'planned_ms': _r(r.get('planned_ms'), 2), 'offset_ms': _r(r.get('offset_ms'), 2),
                  'own_ms': _r(r.get('own_ms'), 2), 'skew_ms': _r(r.get('skew_ms'), 2)}
                 for r in res.sync['cameras'] if r.get('topic')],
        'rotations': [{'camera': r['tag'], 'verdict': r['verdict'], 'confidence': r.get('confidence'),
                       'yaw_pitch_roll_deg': _rl(r.get('delta_deg'), 3)} for r in res.extrinsics],
        'findings': [{'severity': f['severity'], 'message': f['message']} for f in res.findings
                     if f['severity'] != 'ok'],
        'assumptions': ASSUMPTIONS}


def _stream_doc(e):
    lat = e.get('latency_ms')
    return {'sensor': e['tag'], 'kind': e['kind'], 'topic': e['topic'], 'type': e.get('type') or None,
            'status': e['status'], 'messages': e['messages'], 'expected_hz': e.get('expected_hz'),
            'rate_hz': _r(e.get('rate_hz'), 4), 'median_interval_ms': _r(e.get('median_ms'), 3),
            'dropped_frames': e.get('drops', 0), 'drop_events': e.get('drop_events', 0),
            'drop_pct': _r(e.get('drop_share', 0) * 100, 3), 'longest_gap_ms': _r(e.get('longest_gap_ms'), 2),
            'missing_time_pct': _r(e.get('missing_share', 0) * 100, 3), 'jitter_ms': _r(e.get('jitter_ms'), 3),
            'jitter_p95_ms': _r(e.get('jitter_p95_ms'), 3), 'nonmonotonic_stamps': e.get('nonmonotonic', 0),
            'latency_ms': None if lat is None else {k: _r(v, 3) for k, v in lat.items()},
            'coverage_pct': _r(e.get('coverage', 0) * 100 if e.get('coverage') is not None else None, 2),
            'image_size': list(e['size']) if e.get('size') else None,
            'expected_size': list(e['expected_size']) if e.get('expected_size') else None,
            'points_per_cloud': _r(e.get('points'), 1), 'points_expected': _r(e.get('points_expected'), 1),
            'blocked_share_planned': _r(e.get('blocked_share'), 4), 'problems': e['problems'], 'notes': e['notes']}


def to_doc(res: DriveLogResult, project=None) -> dict:
    """The JSON export."""
    def lock(L):
        return None if L is None else {
            'lidar': L['tag'], 'topic': L['topic'], 'source': L['source'], 'sweeps': L['sweeps'],
            'rate_hz': _r(L['rate_hz'], 5), 'plan_rate_hz': L['plan_rate_hz'],
            'rotation': 'ccw' if L['sense'] > 0 else 'cw', 'phase_deg': _r(L['phase_deg'], 3),
            'plan_phase_deg': _r(L['plan_phase_deg'], 3), 'phase_error_deg': _r(L['phase_error_deg'], 3),
            'sweep_late_ms': _r(L['late_ms'], 3), 'phase_spread_deg': _r(L['spread_deg'], 3),
            'fit_rms_deg': _r(L.get('rms_deg'), 3)}
    return {
        'schema': SCHEMA,
        'project': getattr(project, 'name', ''), 'variant': res.variant,
        'log': {'path': res.path, 'source': res.source, 'duration_s': _r(res.span['duration_s'], 4),
                'messages': res.span['messages'], 'bytes': res.span['bytes'], 'start_ns': res.span['start_ns'],
                'end_ns': res.span['end_ns']},
        'frame': 'vehicle (ISO 8855, rear axle)',
        'units': 'times ms unless stated, angles degrees, rates Hz, data decimal (MB = 1e6 bytes, Mbit/s = 1e6 bit/s)',
        'settings': {k: v for k, v in res.settings.to_dict().items() if k != 'mappings'},
        'assumptions': ASSUMPTIONS,
        'notes': list(res.notes),
        'topics': [{'topic': t['topic'], 'type': t['type'], 'messages': t['messages'], 'bytes': t['bytes'],
                    'frame_id': t['frame_id'] or None, 'sensor': t['tag'] or None} for t in res.topics],
        'streams': [_stream_doc(e) for e in res.streams],
        'data': {'sensors': [{'sensor': r['tag'], 'topic': r['topic'], 'mbit_s': _r(r['mbps']),
                              'gb_per_hour': _r(r['gb_h']), 'bytes_per_message': _r(r['bytes_per_msg'], 1),
                              'planned_mbit_s': _r(r.get('planned_mbps')), 'ratio_to_plan': _r(r.get('ratio')),
                              'bits_per_pixel': _r(r.get('bpp'), 4), 'planned_bits_per_pixel': r.get('planned_bpp'),
                              'compression_vs_raw': _r(r.get('compression'), 2)} for r in res.data['rows']],
                 'totals': {k: _r(v, 4) if isinstance(v, float) else v for k, v in res.data['totals'].items()}},
        'sync': {'camera_stamps': res.sync['stamp_convention'], 'readout_ms': res.sync['readout_ms'],
                 'exposure_ms': res.sync['exposure_ms'], 'no_sweep_check': res.sync['reason'] or None,
                 'reference_lidar': lock(res.sync.get('lidar')),
                 'lidars': [lock(L) for L in res.sync['lidars']],
                 'conventions': {'offset': 'exposure centre of the middle row minus the time the reference lidar\'s '
                                           'sweep crosses the camera\'s view, modulo the camera\'s frame period '
                                           '(positive: the exposure comes after the sweep)',
                                 'own': 'the offset\'s deviation from the plan with the lidar\'s phase error taken '
                                        'out: the camera\'s own trigger error',
                                 'skew': 'stamp minus the nearest stamp of the trigger line\'s first camera'},
                 'cameras': [{'camera': r['tag'], 'topic': r['topic'], 'group': r['group'], 'frames': r['frames'],
                              'planned_misalignment_ms': _r(r.get('planned_ms')),
                              'measured_offset_ms': _r(r.get('offset_ms')),
                              'deviation_ms': _r(r.get('deviation_ms')), 'own_ms': _r(r.get('own_ms')),
                              'lidar_phase_part_ms': _r(r.get('lidar_part_ms')), 'p95_ms': _r(r.get('p95_ms')),
                              'spread_5_95_ms': _r(r.get('spread_ms')), 'skew_ms': _r(r.get('skew_ms')),
                              'skew_p95_ms': _r(r.get('skew_p95_ms')), 'skew_against': r.get('skew_ref')}
                             for r in res.sync['cameras']],
                 'trigger_lines': [{'line': g['letter'], 'reference': g['reference'],
                                    'worst_p95_ms': _r(g['worst_p95_ms']),
                                    'cameras': [{'camera': d['tag'], 'median_ms': _r(d['median_ms']),
                                                 'p95_ms': _r(d['p95_ms']), 'frames': d['frames']}
                                                for d in g['cameras']]} for g in res.sync['groups']]},
        'extrinsics': {'method': 'lidar depth edges against image Canny edges, rotation correction of the camera '
                                 '(yaw, pitch, roll added to the plan\'s; intrinsic Z-Y-X, pitch positive down) '
                                 'relative to the lidar; translation is not observable this way',
                       'tolerance_deg': res.settings.tolerance_deg, 'search_deg': res.settings.search_deg,
                       'cameras': [{'camera': r['tag'], 'topic': r.get('topic'), 'lidar': r.get('lidar'),
                                    'verdict': r['verdict'], 'reason': r.get('reason') or None,
                                    'confidence': r.get('confidence'), 'frames': r.get('frames', 0),
                                    'edge_points': r.get('points', 0),
                                    'correction_ypr_deg': _rl(r.get('delta_deg'), 4),
                                    'standard_error_deg': _rl(r.get('stderr_deg'), 4),
                                    'frames_agreeing_pct': _r(r['agree'] * 100, 1) if r.get('agree') is not None
                                    else None,
                                    'peak_drop_pct_at_0_1_deg': _rl(r.get('sharpness_pct'), 3),
                                    'score_plan': _r(r.get('score_plan'), 4),
                                    'score_corrected': _r(r.get('score_corrected'), 4),
                                    'plan_ypr_deg': _rl(r.get('plan_deg'), 4),
                                    'corrected_ypr_deg': _rl(r.get('corrected_deg'), 4),
                                    'time_difference_ms': _r(r.get('dt_ms'), 2),
                                    'motion_compensation': r.get('compensation'), 'apply': r.get('apply', False)}
                                   for r in res.extrinsics]},
        'imu': [{'sensor': r['tag'], 'topic': r['topic'], 'samples': r['samples'],
                 'stationary_s': _r(r['stationary_s'], 2), 'gravity_m_s2': _r(r['gravity'], 4),
                 'gyro_bias_deg_s': _rl(r['gyro_bias_dps'], 4), 'gravity_tilt_deg': _r(r['tilt_deg'], 3),
                 'status': r['status']} for r in res.imu],
        'gnss': [{'sensor': r['tag'], 'topic': r['topic'], 'fixes': r['fixes'],
                  'no_fix_pct': _r(r.get('no_fix_share', 0) * 100, 2),
                  'ground_corrected_pct': _r(r.get('rtk_share', 0) * 100, 2),
                  'h_sigma_m': _r(r.get('h_sigma_m'), 4), 'v_sigma_m': _r(r.get('v_sigma_m'), 4),
                  'status': r['status']} for r in res.gnss],
        'metrics': {k: (_r(v, 4) if isinstance(v, float) else v) for k, v in res.metrics.items()},
        'findings': [{'severity': f['severity'], 'section': f['section'], 'message': f['message'],
                      'sensors': f['tags']} for f in res.findings],
    }


CSV_COLUMNS = ('sensor', 'kind', 'topic', 'status', 'messages', 'expected_hz', 'rate_hz', 'dropped_frames',
               'drop_pct', 'longest_gap_ms', 'jitter_ms', 'latency_median_ms', 'coverage_pct', 'image_size',
               'points_per_cloud', 'mbit_s', 'planned_mbit_s')


def to_csv(res: DriveLogResult) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator='\n')
    w.writerow(CSV_COLUMNS)
    rows = {r['tag']: r for r in res.data['rows']}
    for e in res.streams:
        d = _stream_doc(e)
        r = rows.get(e['tag'], {})
        lat = d['latency_ms'] or {}
        vals = [d['sensor'], d['kind'], d['topic'] or '', d['status'], d['messages'], d['expected_hz'],
                d['rate_hz'], d['dropped_frames'], d['drop_pct'], d['longest_gap_ms'], d['jitter_ms'],
                lat.get('median'), d['coverage_pct'], 'x'.join(map(str, d['image_size'])) if d['image_size'] else '',
                d['points_per_cloud'], _r(r.get('mbps')), _r(r.get('planned_mbps'))]
        w.writerow(['' if v is None else v for v in vals])
    return buf.getvalue()


# ---------------------------------------------------------------- pictures
SEV_BGR = {'fault': (59, 59, 208), 'warning': (25, 178, 250), 'ok': (12, 163, 12)}
CAM_COLORS = ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#008300', '#9085e9', '#e66767']
KIND_COLORS = {'lidar': '#0aa5b8', 'radar': '#a0743c', 'imu': '#9085e9', 'gnss': '#c98500'}


def _bgr(hex_):
    return int(hex_[5:7], 16), int(hex_[3:5], 16), int(hex_[1:3], 16)


def stream_color(project_var, e) -> str:
    """Hex colour of a stream's row: a camera's slot colour, a sensor its kind's."""
    if e['kind'] == 'camera' and project_var is not None:
        c = next((c for c in project_var.cameras if c.uid == e['uid']), None)
        if c is not None:
            return CAM_COLORS[c.slot % 8]
    return KIND_COLORS.get(e['kind'], '#898781')


def range_colors(r, lo=1.0, hi=40.0):
    """RGB uint8 (K, 3) for ranges: a perceptually ordered ramp from near (yellow) to far (blue), log scale."""
    t = np.clip((np.log(np.maximum(r, 1e-3)) - math.log(lo)) / (math.log(hi) - math.log(lo)), 0, 1)
    stops = np.array([[253, 231, 37], [94, 201, 98], [33, 145, 140], [59, 82, 139], [68, 1, 84]], float)
    x = t * (len(stops) - 1)
    i = np.minimum(x.astype(int), len(stops) - 2)
    f = (x - i)[:, None]
    return (stops[i] * (1 - f) + stops[i + 1] * f).astype(np.uint8)


def project_overlay(project, var, ov, frame, delta=None, size=None):
    """Pixel positions (x, y) in a thumbnail of `size` (default the frame's) and colours of the frame's points at the
    plan (delta None) or the corrected rotation, and which fall inside."""
    from .lens import resolve
    cam = next((c for c in var.cameras if c.tag == ov['tag']), None)
    if cam is None or not len(frame['Q']):
        return np.zeros(0), np.zeros(0), np.zeros((0, 3), np.uint8)
    intr = resolve(project.module_for(cam), project.lens_for(cam))
    y, p, r = ov['plan']
    d = delta if delta is not None else (0.0, 0.0, 0.0)
    W2O = BODY_TO_OPTICAL @ rot_zyx(y + d[0], p + d[1], r + d[2]).T
    u, v, th, ok = intr.project_dirs(frame['Q'].astype(np.float64) @ W2O.T)
    W, H = ov['module_size']
    wt, ht = size or frame['tsize']
    x, yy = u * wt / W - 0.5, v * ht / H - 0.5
    ok &= th <= intr.max_theta
    return x[ok], yy[ok], range_colors(frame['rng'][ok])


def overlay_png(project, var, res: DriveLogResult, uid, k=None) -> bytes | None:
    """The plan's and the corrected projection of one sampled frame side by side (points coloured by range)."""
    import cv2
    ov = res.overlays.get(uid)
    if not ov or not ov['frames']:
        return None
    frames = ov['frames']
    k = len(frames) // 2 if k is None else max(0, min(k, len(frames) - 1))
    fr = frames[k]
    base = cv2.imdecode(np.frombuffer(fr['jpeg'], np.uint8), cv2.IMREAD_COLOR)
    if base is None:
        return None
    ext = next((e for e in res.extrinsics if e['uid'] == uid), {})
    panels = []
    for label, delta in (('plan', None), ('corrected', ov.get('delta'))):
        img = base.copy()
        if label == 'corrected' and delta is None:
            img[:] = (img * 0.35).astype(np.uint8)
        else:
            x, y, col = project_overlay(project, var, ov, fr, delta)
            for xi, yi, c in zip(np.round(x).astype(int), np.round(y).astype(int), col):
                cv2.circle(img, (int(xi), int(yi)), 1, (int(c[2]), int(c[1]), int(c[0])), -1, cv2.LINE_AA)
        d = delta or [0.0, 0.0, 0.0]
        txt = 'plan' if label == 'plan' else (f'corrected  yaw {d[0]:+.2f}  pitch {d[1]:+.2f}  roll {d[2]:+.2f} deg'
                                              if delta is not None else 'no correction')
        cv2.rectangle(img, (0, 0), (img.shape[1], 22), (30, 29, 28), -1)
        put_text(img, f'{ov["tag"]}  {txt}', (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (236, 235, 230), 1,
                    cv2.LINE_AA)
        panels.append(img)
    out = np.concatenate([panels[0], np.full((panels[0].shape[0], 6, 3), 30, np.uint8), panels[1]], 1)
    foot = np.full((24, out.shape[1], 3), (30, 29, 28), np.uint8)
    s0, s1 = ext.get('score_plan'), ext.get('score_corrected')
    line = f'frame {k + 1} of {len(frames)} at {fr["t_s"]:.2f} s   {fr["edges"]} depth-edge points'
    if s0 is not None and s1 is not None:
        line += f'   edge score {s0:.3f} -> {s1:.3f}   {ext.get("verdict", "")} ({ext.get("confidence") or "-"})'
    put_text(foot, line, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (183, 194, 195), 1, cv2.LINE_AA)
    ok, buf = cv2.imencode('.png', np.concatenate([out, foot], 0))
    return buf.tobytes() if ok else None


def timeline_png(res: DriveLogResult, var=None, width=1200) -> bytes:
    """One row per stream over the drive: where it has data, its drops (red, at least 1 px wide) and a missing stream
    as a hatched row."""
    import cv2
    rows = res.streams
    head, rh, left = 40, 22, 90
    H = head + rh * max(len(rows), 1) + 34
    img = np.full((H, width, 3), (30, 29, 28), np.uint8)
    font, aa = cv2.FONT_HERSHEY_SIMPLEX, cv2.LINE_AA
    dur = max(res.span['duration_s'], 1e-6)
    x0, x1 = left, width - 16
    X = lambda s: int(round(x0 + (x1 - x0) * s / dur))          # noqa: E731
    put_text(img, f'Drops and gaps per stream  {os.path.basename(res.path)}  {dur:.1f} s', (12, 24), font, 0.55,
                (236, 235, 230), 1, aa)
    for i, e in enumerate(rows):
        y = head + i * rh
        col = _bgr(stream_color(var, e))
        put_text(img, e['tag'], (12, y + 15), font, 0.45, col, 1, aa)
        if e['topic'] is None:
            for xx in range(x0, x1, 8):
                cv2.line(img, (xx, y + 17), (xx + 6, y + 5), (90, 88, 84), 1, aa)
            continue
        a, b = (X(v) for v in (e.get('span_s') or (e.get('first_s', 0.0), e.get('last_s', dur))))
        cv2.rectangle(img, (a, y + 6), (max(b, a + 1), y + 16), tuple(int(0.55 * c) for c in col), -1)
        for g0, g1 in e.get('gaps', []):
            ga, gb = X(g0), X(g1)
            cv2.rectangle(img, (ga, y + 3), (max(gb, ga + 1), y + 19), SEV_BGR['fault'], -1)
    yb = head + rh * max(len(rows), 1) + 8
    cv2.line(img, (x0, yb), (x1, yb), (129, 135, 137), 1)
    step = next(s for s in (0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 30, 60, 120, 300, 600, 1800, 3600, 1e9) if dur / s <= 12)
    t = 0.0
    while t <= dur + 1e-9:
        cv2.line(img, (X(t), yb), (X(t), yb + 4), (129, 135, 137), 1)
        put_text(img, f'{t:g} s', (X(t) - 10, yb + 18), font, 0.38, (129, 135, 137), 1, aa)
        t += step
    ok, buf = cv2.imencode('.png', img)
    return buf.tobytes()


def files(res: DriveLogResult, project=None) -> list:
    """[(name, bytes)]: drive_log_check.json, drive_log_streams.csv, drive_log_timeline.png and an overlay PNG per
    checked camera (drive_log_overlay_<tag>.png)."""
    var = project.variant_by_uid(res.variant_uid) if project is not None else None
    out = [('drive_log_check.json', (json.dumps(to_doc(res, project), indent=1, ensure_ascii=False) + '\n')
            .encode('utf-8')),
           ('drive_log_streams.csv', to_csv(res).encode('utf-8')),
           ('drive_log_timeline.png', timeline_png(res, var))]
    if project is not None and var is not None:
        for uid, ov in res.overlays.items():
            data = overlay_png(project, var, res, uid)
            if data:
                out.append((f'drive_log_overlay_{re.sub(r"[^0-9A-Za-z_-]+", "_", ov["tag"])}.png', data))
    return out


def export(res: DriveLogResult, project, out_dir) -> list[str]:
    """Write files(res) into out_dir; returns their paths."""
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for name, data in files(res, project):
        p = os.path.join(out_dir, name)
        with open(p, 'wb') as fh:
            fh.write(data)
        paths.append(p)
    return paths
