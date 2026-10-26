"""Synthetic sample dataset: a short drive recorded by exactly the planned rig, written as a nuScenes version folder
and a ROS 2 bag, so the team can test its ingestion pipeline, labelling tools and training loaders against the real
sensor set, frames, timing and calibration before any hardware exists (core.drivescene simulates the drive).

What is written (out_dir):
* nuscenes/  a devkit-loadable version folder v1.0-vantage with all 13 tables, samples/<CHANNEL>/ for keyframes and
  sweeps/<CHANNEL>/ for the rest (.jpg, lidar .pcd.bin: float32 x y z intensity ring, radar .pcd with nuScenes'
  binary header), and maps/<token>.png, the drivable area at 0.1 m per pixel. Load it with
  NuScenes(version='v1.0-vantage', dataroot='<out_dir>/nuscenes').
* rosbag/<name>/  a rosbag2 bag (MCAP): images, camera_info, lidar points, radar detections, IMU, GNSS fixes, /tf,
  /tf_static, /odom and the ground truth as markers and JSON.
* dataset.json and README.md: what was written, counts, sizes and the conventions.

Calibration: the calibrated_sensor rows are the nuscenes_calib export's (core.exporters, rear-axle ego frame, the same
tokens); a camera's camera_intrinsic is that K scaled to the written image size. Images are the real lens's images
(fisheye; the export's vantage_distortion holds the Kannala-Brandt coefficients) or, with undistort, rectified
pinhole images whose camera_intrinsic is exact. GNSS antennas and IMUs are not nuScenes sensors: they drive the bag's
GNSS and IMU topics, while ego_pose is the exact simulated pose at each sample_data's timestamp. Channels are named as
the export names them (CAM_<TAG>, LIDAR_<TAG>, RADAR_<TAG>), or with channel_names = 'nuscenes' as nuScenes does
(CAM_FRONT … by heading, the highest lidar LIDAR_TOP, RADAR_FRONT … by corner) for loaders and devkit helpers that
expect those names; the rows are the same either way.

Time: t = 0 is a sync pulse at EPOCH (2026-01-01 12:00:00 UTC); nuScenes timestamps are integer microseconds, bag
stamps nanoseconds, from the same instants. A camera frame's timestamp is the middle row's exposure centre (trigger
+ readout/2 + exposure/2, core.timing's settings); cameras on a trigger line fire at the sync plan's trigger times
(core.timing), the others free-run from t = 0. A lidar sweep starts at a sync pulse and is stamped at its end (as
nuScenes); each point's t field in the bag is seconds from the sweep start. A radar scan and IMU / GNSS samples are
stamped at their measurement time. Keyframes (nuScenes samples) are every n-th sweep of the first lidar (n = its rate
/ keyframe_hz), else every n-th frame of the first camera; every other channel's keyframe is its sample_data nearest
in time.

Frames: the nuScenes global frame and the bag's odom frame are the scene's world frame (x east, y north, z up, metres,
origin at origin_lat / origin_lon / origin_alt, WGS84); ego / base_link is the vehicle frame (ISO 8855, rear-axle
centre on the ground). Bag frame ids are the URDF export's link names: <tag>_link for sensor bodies and lidars,
radars, GNSS antennas and IMUs (REP 103 axes), <tag>_optical_frame for camera images.

Weather: DatasetSettings.weather names a core.weather condition ('' follows the project's selected one, 'clear' unless
set): fog, rain, snow and night show in the camera images, the lidar loses far returns and gains false near ones, the
radar's reach and RCS shrink with the rain's loss (core.drivescene). 'Clear' writes exactly what it wrote before weather
existed; another condition is named in dataset.json (scene.weather) and the README.

Limits: see core.drivescene (synthetic geometry, simple materials, no weather): the scene is for testing a pipeline,
not for training.
"""
from __future__ import annotations

import concurrent.futures as cf
import datetime as _dt
import json
import math
import os
import shutil
import threading
import time
import zlib
from dataclasses import asdict, dataclass, field

import numpy as np

from . import drivescene as DS
from . import exporters as EX
from . import timing as TM
from . import weather as WX
from .mathutil import BODY_TO_OPTICAL, quat_from_matrix, rot_zyx

SCHEMA = 'vantage.dataset/1'
VERSION = 'v1.0-vantage'
EPOCH_US = 1_767_268_800_000_000            # 2026-01-01 12:00:00 UTC
SCENES = ('urban', 'highway')
SCENE_LABEL = {'urban': 'Urban street', 'highway': 'Highway'}
CAMERA_RATES = ('lidar', 'module')
CHANNEL_NAMES = ('tags', 'nuscenes')        # CAM_<TAG> as the nuscenes_calib export, or nuScenes' own names
# nuScenes' camera channels and the heading each faces (degrees, counter-clockwise from forward)
NUSC_CAMERAS = (('CAM_FRONT', 0.0), ('CAM_FRONT_LEFT', 55.0), ('CAM_FRONT_RIGHT', -55.0), ('CAM_BACK_LEFT', 110.0),
                ('CAM_BACK_RIGHT', -110.0), ('CAM_BACK', 180.0))
MARKER = '.vantage-dataset'                 # marks a nuscenes/ folder this module wrote (it may clear it again)
MAP_RES = 0.1                               # m per map pixel (the devkit's MapMask resolution)
LIMITS = {'duration_s': (0.5, 30.0), 'speed_kmh': (0.0, 130.0), 'seed': (0, 999999), 'image_scale': (0.05, 1.0),
          'keyframe_hz': (0.2, 20.0), 'jpeg_quality': (30, 100), 'lidar_noise_cm': (0.0, 20.0),
          'origin_lat': (-85.0, 85.0), 'origin_lon': (-180.0, 180.0), 'origin_alt': (-500.0, 9000.0)}
INTS = ('seed', 'jpeg_quality')
BOOLS = ('nuscenes', 'rosbag', 'rolling_shutter', 'drop_self_hits', 'undistort')
ASSUMPTIONS = ('Synthetic geometry (boxes, cylinders, spheres), simple materials, one fixed sun, no weather, flat '
               'road, '
               'no suspension motion; actors at constant speeds. Lidar: one return per beam, typical material '
               'reflectivities, R_max(ρ) = range_m·√(ρ/0.10) with ρ scaled by the incidence cosine, Gaussian range '
               'noise. Radar: one detection per object at its nearest point, typical noise. IMU: typical MEMS '
               'noise and bias. GNSS: white RTK-like noise. The scene is for testing a pipeline (ingestion, labelling '
               'tools, loaders), not for training.')
# estimate() figures, measured on the sample data rig at the defaults (the real numbers depend on the scene, the
# settings and the machine)
JPEG_BITS_PER_PX = 0.95                     # at quality 90; scaled by quality below
LIDAR_RETURNS = 0.53                        # share of beams that return in the urban scene
BAG_POINT_BYTES = (15.0, 22.0)              # a lidar point in the bag with zstd chunks, without
SEC_PER_MPX = 0.67                          # rendering, per megapixel on one worker while the others run
SEC_PER_MRAY = 0.7                          # lidar, per million beams
SEC_PER_SCAN = 0.011                        # radar, per scan
SEC_PER_SAMPLE = 0.3                        # annotations (visibility rays), per keyframe
WORKERS = 3                                 # camera frames rendered at once (each needs ~150 MB at half size)

CATEGORIES = [
    ('human.pedestrian.adult', 'Adult subcategory.'), ('human.pedestrian.child', 'Child subcategory.'),
    ('human.pedestrian.wheelchair', 'Wheelchairs.'), ('human.pedestrian.stroller', 'Strollers.'),
    ('human.pedestrian.personal_mobility', 'Small electric or self-propelled vehicles.'),
    ('human.pedestrian.police_officer', 'Police officer.'),
    ('human.pedestrian.construction_worker', 'Construction worker.'), ('animal', 'All animals.'),
    ('vehicle.car', 'Passenger cars, vans and SUVs.'), ('vehicle.motorcycle', 'Motorcycles and scooters.'),
    ('vehicle.bicycle', 'Bicycles; the box includes the rider.'), ('vehicle.bus.bendy', 'Articulated buses.'),
    ('vehicle.bus.rigid', 'Rigid buses.'), ('vehicle.truck', 'Trucks and lorries.'),
    ('vehicle.construction', 'Construction vehicles.'), ('vehicle.emergency.ambulance', 'Ambulances.'),
    ('vehicle.emergency.police', 'Police vehicles.'), ('vehicle.trailer', 'Trailers.'),
    ('movable_object.barrier', 'Temporary road barriers.'), ('movable_object.trafficcone', 'Traffic cones.'),
    ('movable_object.pushable_pullable', 'Objects pushed or pulled by a person.'),
    ('movable_object.debris', 'Debris on the road.'), ('static_object.bicycle_rack', 'Bicycle racks.')]
ATTRIBUTES = [('vehicle.moving', 'Vehicle is moving.'), ('vehicle.stopped', 'Vehicle is stationary, with a driver.'),
              ('vehicle.parked', 'Vehicle is parked.'), ('cycle.with_rider', 'There is a rider on the bicycle.'),
              ('cycle.without_rider', 'There is no rider on the bicycle.'),
              ('pedestrian.sitting_lying_down', 'The pedestrian is sitting or lying down.'),
              ('pedestrian.standing', 'The pedestrian is standing.'),
              ('pedestrian.moving', 'The pedestrian is moving.')]
VISIBILITY = [('1', 'v0-40', 'visibility of whole object is between 0 and 40%'),
              ('2', 'v40-60', 'visibility of whole object is between 40 and 60%'),
              ('3', 'v60-80', 'visibility of whole object is between 60 and 80%'),
              ('4', 'v80-100', 'visibility of whole object is between 80 and 100%')]
RADAR_DTYPE = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('dyn_prop', 'i1'), ('id', '<i2'), ('rcs', '<f4'),
                        ('vx', '<f4'), ('vy', '<f4'), ('vx_comp', '<f4'), ('vy_comp', '<f4'),
                        ('is_quality_valid', 'i1'), ('ambig_state', 'i1'), ('x_rms', 'i1'), ('y_rms', 'i1'),
                        ('invalid_state', 'i1'), ('pdh0', 'i1'), ('vx_rms', 'i1'), ('vy_rms', 'i1')])
PCD_HEAD = ('# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\n'
            'FIELDS x y z dyn_prop id rcs vx vy vx_comp vy_comp is_quality_valid ambig_state x_rms y_rms '
            'invalid_state pdh0 vx_rms vy_rms\nSIZE 4 4 4 1 2 4 4 4 4 4 1 1 1 1 1 1 1 1\n'
            'TYPE F F F I I F F F F F I I I I I I I I\nCOUNT 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1\n'
            'WIDTH {n}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {n}\nDATA binary\n')
LIDAR_PC2 = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('intensity', '<f4'), ('ring', '<u2'), ('t', '<f4')])
RADAR_PC2 = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('v_r', '<f4'), ('rcs', '<f4')])
TF_HZ = 50.0
GNSS_HZ = 10.0
GT_HZ = 10.0
MARKER_RGBA = {'vehicle.car': (0.20, 0.55, 0.95, 0.45), 'vehicle.truck': (0.55, 0.35, 0.95, 0.45),
               'human.pedestrian.adult': (0.95, 0.35, 0.20, 0.55), 'vehicle.bicycle': (0.95, 0.75, 0.10, 0.55)}


# =========================================================================== settings
@dataclass
class DatasetSettings:
    duration_s: float = 8.0
    speed_kmh: float = 30.0
    scene: str = 'urban'               # urban | highway
    seed: int = 1
    image_scale: float = 0.5           # images at this share of the sensor's size (1 = full)
    camera_rate: str = 'lidar'         # lidar: one frame per lidar rotation, as the sync plan triggers; module: its fps
    channel_names: str = 'tags'        # tags: CAM_<TAG> (the nuscenes_calib export's); nuscenes: CAM_FRONT, LIDAR_TOP …
    keyframe_hz: float = 2.0
    jpeg_quality: int = 90
    nuscenes: bool = True
    rosbag: bool = True
    rolling_shutter: bool = False
    lidar_noise_cm: float = 1.5
    drop_self_hits: bool = True
    undistort: bool = False            # rectified pinhole images (camera_intrinsic exact) instead of the fisheye's
    origin_lat: float = 13.0827
    origin_lon: float = 80.2707
    origin_alt: float = 7.0
    weather: str = ''                  # a core.weather condition id; '' = the project's selected condition

    def to_dict(self):
        d = asdict(self)
        if not d['weather']:            # the default stays out: settings, tokens and dataset.json as before weather
            del d['weather']
        return d

    @staticmethod
    def from_dict(d: dict | None) -> 'DatasetSettings':
        """Tolerant of junk: unknown keys ignored, numbers clamped to LIMITS, words checked."""
        s = DatasetSettings()
        d = d if isinstance(d, dict) else {}
        if d.get('scene') in SCENES:
            s.scene = d['scene']
        if d.get('camera_rate') in CAMERA_RATES:
            s.camera_rate = d['camera_rate']
        if d.get('channel_names') in CHANNEL_NAMES:
            s.channel_names = d['channel_names']
        if d.get('weather') in WX.IDS:
            s.weather = d['weather']
        for k in BOOLS:
            v = d.get(k)
            if isinstance(v, (bool, int)) and not isinstance(v, float):
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
        return s


@dataclass
class DatasetResult:
    settings: DatasetSettings
    out_dir: str
    nuscenes_dir: str                  # '' when not written
    bag_dir: str                       # '' when not written
    project: str
    variant: str
    variant_uid: str
    channels: list                     # per sensor: {'kind', 'tag', 'channel', 'topic', 'frames', 'keyframes', 'bytes'}
    counts: dict                       # nuScenes table rows, bag messages, annotations
    sizes: dict                        # bytes: nuscenes, rosbag, total
    scene: dict                        # the scene's description
    seconds: float
    notes: list = field(default_factory=list)
    bag: dict | None = None            # BagWriter.close()
    weather: dict | None = None        # the core.weather condition (None: clear)


# =========================================================================== helpers
def _project_for(project, variant):
    """The project with variant (a Variant, its uid or None for the active one) active: a shallow copy if needed."""
    if isinstance(variant, str):
        variant = project.variant_by_uid(variant)
    if variant is None or variant is project.variant:
        return project
    import copy
    q = copy.copy(project)
    q.variants = list(project.variants) + ([] if any(v is variant for v in project.variants) else [variant])
    q.active_uid = variant.uid
    return q


def _us(t: float) -> int:
    return EPOCH_US + int(round(t * 1e6))


def _ns(t: float) -> int:
    return EPOCH_US * 1000 + int(round(t * 1e9))


def _ros_name(tag: str) -> str:
    """A ROS name token of a tag: lower case, letters, digits and '_', not starting with a digit."""
    t = ''.join(ch if ch.isalnum() else '_' for ch in tag.lower()).strip('_') or 'sensor'
    return t if t[0].isalpha() else 's_' + t


def _quat_wxyz(R) -> list:
    x, y, z, w = quat_from_matrix(np.asarray(R, float))
    if w < 0:
        x, y, z, w = -x, -y, -z, -w
    return [EX._r(w, 10), EX._r(x, 10), EX._r(y, 10), EX._r(z, 10)]


def _yaw_quat(yaw) -> list:
    return [EX._r(math.cos(yaw / 2), 10), 0.0, 0.0, EX._r(math.sin(yaw / 2), 10)]


def _ego_dims(project) -> tuple:
    """(length, x of the rear end) of the ego in its frame (typical figures without a body)."""
    try:
        m = project.vehicle.measured() if project.vehicle.mesh is not None else {}
    except Exception:
        m = {}
    rear = float(m.get('rear_x', -0.8))
    front = float(m.get('front_x', 3.2))
    return max(front - rear, 1.0), rear


_CACHE = {}
_CACHE_LOCK = threading.Lock()


def _cached(key, make, keep=6):
    with _CACHE_LOCK:
        if key in _CACHE:
            v = _CACHE.pop(key)
            _CACHE[key] = v
            return v
    v = make()
    with _CACHE_LOCK:
        _CACHE[key] = v
        while len(_CACHE) > keep:
            _CACHE.pop(next(iter(_CACHE)))
    return v


def scene_for(project, st: DatasetSettings) -> DS.Scene:
    """The drive scene for these settings (cached: previews at other times reuse it)."""
    L, rear = _ego_dims(project)
    key = ('scene', st.scene, round(st.speed_kmh, 6), round(st.duration_s, 6), int(st.seed), round(L, 4),
           round(rear, 4))
    return _cached(key, lambda: DS.build_scene(st.scene, st.speed_kmh, st.duration_s, st.seed, L, rear))


# =========================================================================== the plan: what fires when
@dataclass
class _Chan:
    kind: str                          # camera lidar radar imu gnss
    sensor: object
    tag: str                           # file / frame safe tag (the exports')
    channel: str                       # nuScenes channel ('' for imu, gnss)
    topic: str                         # bag topic prefix '/<tag>'
    times: np.ndarray                  # trigger / sweep start / sample times, s
    stamps: np.ndarray                 # what is written: exposure centre / sweep end / sample time, s
    rate: float                        # Hz (nominal)
    info: dict = field(default_factory=dict)


def _plan(project, variant, st: DatasetSettings) -> dict:
    """Every sensor's events over the drive, the keyframe (sample) times and notes. Raises ValueError when the variant
    has nothing that records."""
    T = float(st.duration_s)
    tp = TM.plan(project, variant)
    readout = tp.settings.readout_ms / 1000.0
    exposure = tp.settings.exposure_ms / 1000.0
    period = tp.period_ms / 1000.0 if tp.period_ms else None
    ctags = EX.safe_tags(variant)
    stags = EX.sensor_tags(variant)
    notes, chans = [], []
    for c in variant.cameras:
        if not c.visible:
            continue
        mod = project.module_for(c)
        e = tp.cam(c.uid) or {}
        fps = float(mod.fps or 0.0)
        trig = e.get('trigger_ms')
        if period is not None and st.camera_rate == 'lidar':
            dt = period
            off = (trig / 1000.0) if (e.get('triggered') and trig is not None) else 0.0
        else:
            if fps <= 0:
                notes.append(f'{c.tag}: its module has no frame rate; the camera is left out.')
                continue
            dt = 1.0 / fps
            off = ((trig / 1000.0) % dt) if (period is not None and e.get('triggered') and trig is not None) else 0.0
        last = T - readout - exposure - off              # the last frame's readout ends within the drive
        k = np.arange(int(math.floor(last / dt + 1e-9)) + 1) if last >= 0 else np.zeros(0, int)
        times = off + k * dt
        tag = ctags[c.uid]
        chans.append(_Chan('camera', c, tag, f'CAM_{tag.upper()}', '/' + _ros_name(tag), times,
                           times + readout / 2 + exposure / 2, 1.0 / dt,
                           {'triggered': bool(e.get('triggered')) and trig is not None, 'trigger_ms': trig,
                            'period_s': dt, 'offset_s': off}))
    for s in variant.sensors:
        if not s.visible:
            continue
        m = project.sensor_model_for(s)
        tag = stags[s.uid]
        rate = float(m.rate_hz or 0.0)
        if s.kind == 'gnss':
            rate = GNSS_HZ
        if rate <= 0:
            notes.append(f'{s.tag}: no rate in its model; left out.')
            continue
        if s.kind == 'lidar':
            P = 1.0 / rate
            k = np.arange(int(math.floor(T / P + 1e-9)))
            chans.append(_Chan('lidar', s, tag, f'LIDAR_{tag.upper()}', '/' + _ros_name(tag), k * P, (k + 1) * P, rate))
        else:
            k = np.arange(int(math.floor(T * rate + 1e-9)) + 1)
            name = {'radar': f'RADAR_{tag.upper()}'}.get(s.kind, '')
            chans.append(_Chan(s.kind, s, tag, name, '/' + _ros_name(tag), k / rate, k / rate, rate))
    data = [c for c in chans if c.kind in ('camera', 'lidar', 'radar')]
    if not data:
        raise ValueError('Nothing to record: the variant has no visible camera, lidar or radar.')
    empty = [c for c in data if not len(c.stamps)]
    for c in empty:
        notes.append(f'{c.sensor.tag}: no frame fits in {T:g} s; left out.')
    chans = [c for c in chans if len(c.stamps)]
    data = [c for c in chans if c.kind in ('camera', 'lidar', 'radar')]
    if not data:
        raise ValueError('Nothing to record: no frame fits in the drive; make it longer.')
    for c in data:
        c.info['export_channel'] = c.channel
    if st.channel_names == 'nuscenes':
        notes += _nuscenes_names(project, data)
    # keyframes: every n-th event of the first lidar, else the first camera, else the first radar
    ref = next((c for c in data if c.kind == 'lidar'), None) or next((c for c in data if c.kind == 'camera'), None) \
        or data[0]
    slowest = min(c.rate for c in data)
    kf = min(st.keyframe_hz, slowest, ref.rate)
    n = max(1, int(round(ref.rate / kf)))
    if abs(ref.rate / n - st.keyframe_hz) > 1e-6:
        notes.append(f'Keyframes at {ref.rate / n:.3g} Hz: every {n} frame{"s" * (n > 1)} of {ref.sensor.tag}'
                     + (f' (no faster than the slowest sensor, {slowest:g} Hz).' if kf < st.keyframe_hz - 1e-9
                        else '.'))
    samples = ref.stamps[n - 1::n] if len(ref.stamps) >= n else ref.stamps[:1]
    keys = {}
    for c in data:
        j = np.abs(c.stamps[None, :] - samples[:, None]).argmin(1)
        seen = {}
        for i, jj in enumerate(j):                       # each sample_data is the keyframe of one sample at most
            if jj in seen:
                if abs(c.stamps[jj] - samples[i]) < abs(c.stamps[jj] - samples[seen[jj]]):
                    seen[jj] = i
            else:
                seen[jj] = i
        keys[c.channel] = {int(jj): i for jj, i in seen.items()}
        if len(seen) < len(samples):
            notes.append(f'{c.channel}: {len(samples) - len(seen)} sample(s) without a keyframe of this channel.')
    return {'chans': chans, 'samples': samples, 'keys': keys, 'ref': ref, 'keyframe_hz': ref.rate / n,
            'readout': readout, 'exposure': exposure, 'period': period, 'timing': tp, 'notes': notes}


def _nuscenes_names(project, data) -> list:
    """Rename the channels as nuScenes names them, for loaders and devkit helpers that expect its names: each camera
    the nuScenes camera nearest its heading (closest pairs first), the highest lidar LIDAR_TOP, the radar facing
    forward at the front RADAR_FRONT and the others RADAR_<FRONT|BACK>_<LEFT|RIGHT> by their corner. A sensor left
    over keeps its tag's channel. Returns notes."""
    from .sensors import heading
    notes = []
    cams = [c for c in data if c.kind == 'camera']
    pairs = sorted((abs((c.sensor.yaw - yaw + 180.0) % 360.0 - 180.0), i, name)
                   for i, c in enumerate(cams) for name, yaw in NUSC_CAMERAS)
    done, used = set(), set()
    for _, i, name in pairs:
        if i not in done and name not in used:
            cams[i].channel = name
            done.add(i)
            used.add(name)
    lids = [c for c in data if c.kind == 'lidar']
    if lids:
        max(lids, key=lambda c: c.sensor.z).channel = 'LIDAR_TOP'
    rads = [c for c in data if c.kind == 'radar']
    if rads:
        xs = [c.sensor.x for c in rads]
        mid = (min(xs) + max(xs)) / 2
        front = [c for c in rads if abs(heading(c.sensor)) < 30.0]
        taken = set()
        if front:
            f = max(front, key=lambda c: c.sensor.x)
            f.channel = 'RADAR_FRONT'
            taken.add(f.channel)
        for c in rads:
            if c.channel == 'RADAR_FRONT':
                continue
            name = f'RADAR_{"FRONT" if c.sensor.x >= mid else "BACK"}_{"LEFT" if c.sensor.y >= 0 else "RIGHT"}'
            if name not in taken:
                c.channel = name
                taken.add(name)
    left = [c.channel for c in data if c.channel == c.info['export_channel'] and c.kind != 'lidar' or
            (c.kind == 'lidar' and c.channel != 'LIDAR_TOP')]
    if left:
        notes.append(f'No nuScenes name left for {", ".join(left)}: they keep their tags\' channel names.')
    return notes


def _camera_setup(project, ctx_model, scale: float, undistort: bool):
    """(w, h, K scaled to the image, distortion model, coefficients) of a camera's dataset images."""
    intr = ctx_model.intr
    w = max(16, int(round(intr.width * scale)))
    h = max(16, int(round(intr.height * scale)))
    sx, sy = w / intr.width, h / intr.height
    K = np.array([[ctx_model.fx * sx, 0.0, (ctx_model.cx + 0.5) * sx - 0.5],
                  [0.0, ctx_model.fy * sy, (ctx_model.cy + 0.5) * sy - 0.5], [0.0, 0.0, 1.0]])
    pin = EX._pinhole(ctx_model)
    if undistort or pin:
        return w, h, K, 'plumb_bob', [0.0] * 5
    return w, h, K, 'equidistant', [float(k) for k in ctx_model.k]


# =========================================================================== estimate
def estimate(project, settings: DatasetSettings | None = None, variant=None) -> dict:
    """Frames per sensor, approximate bytes and seconds for a generation with these settings (no rendering)."""
    st = settings or DatasetSettings.from_dict(getattr(project.analysis, 'dataset', None))
    p = _project_for(project, variant)
    v = p.variant
    try:
        plan = _plan(p, v, st)
    except ValueError as e:
        return {'ok': False, 'reason': str(e), 'frames': {}, 'bytes': 0, 'seconds': 0.0}
    ctx = EX._context(p, None, EX.ExportOptions(frame='rear_axle'))
    models = {cm.cam.uid: cm for cm in ctx.models}
    frames, px, rays, scans, nbytes = {}, 0.0, 0.0, 0, 0.0
    outs = int(st.nuscenes) + int(st.rosbag)
    qf = (st.jpeg_quality / 90.0) ** 2.5
    import importlib.util
    bag_pt = BAG_POINT_BYTES[0] if importlib.util.find_spec('zstandard') else BAG_POINT_BYTES[1]
    for c in plan['chans']:
        n = len(c.stamps)
        frames[c.channel or c.topic] = n
        if c.kind == 'camera':
            w, h, *_ = _camera_setup(p, models[c.sensor.uid], st.image_scale, st.undistort)
            px += w * h * n
            nbytes += outs * w * h * JPEG_BITS_PER_PX * qf / 8.0 * n
        elif c.kind == 'lidar':
            m = p.sensor_model_for(c.sensor)
            r = m.n_channels() * (m.columns() if m.spinning else len(DS.SN.scan_azimuths(m)))
            rays += r * n
            nbytes += r * n * LIDAR_RETURNS * (20.0 * st.nuscenes + bag_pt * st.rosbag)
        elif c.kind == 'radar':
            scans += n
            nbytes += n * (20 * 43 * st.nuscenes + 20 * 20 * st.rosbag + 600)
        else:
            nbytes += n * 400 * st.rosbag
    nbytes += st.rosbag * st.duration_s * (TF_HZ * 2 * 300 + GT_HZ * 30 * 400)
    workers = max(1, min(WORKERS, os.cpu_count() or 1))
    render = px / 1e6 * SEC_PER_MPX / workers * (1.3 if st.rolling_shutter else 1.0)
    sensors = rays / 1e6 * SEC_PER_MRAY + scans * SEC_PER_SCAN          # on the main thread, beside the rendering
    secs = max(render, sensors) + len(plan['samples']) * SEC_PER_SAMPLE + 2.0
    return {'ok': True, 'reason': '', 'frames': frames, 'samples': len(plan['samples']),
            'keyframe_hz': plan['keyframe_hz'], 'images': int(sum(len(c.stamps) for c in plan['chans']
                                                                   if c.kind == 'camera')),
            'sweeps': int(sum(len(c.stamps) for c in plan['chans'] if c.kind == 'lidar')), 'scans': scans,
            'megapixels': px / 1e6, 'lidar_beams': rays, 'bytes': nbytes, 'seconds': secs, 'notes': plan['notes']}


# =========================================================================== preview
def preview(project, engine, settings: DatasetSettings | None = None, t: float = 0.0, variant=None,
            width: int = 320) -> dict:
    """One instant of the drive for the page: camera images at preview size, the lidar points of the sweep around t
    and the radar detections (in the ego frame at t, x forward, y left), the ground-truth boxes and the road edges."""
    st = settings or DatasetSettings.from_dict(getattr(project.analysis, 'dataset', None))
    p = _project_for(project, variant)
    v = p.variant
    plan = _plan(p, v, st)
    t = float(min(max(t, 0.0), st.duration_s))
    scene = scene_for(p, st)
    if engine is None:
        from .analysis import Engine
        engine = Engine()
    engine.vehicle_geom(p)
    wx = WX.scene_weather(p, st.weather or None, engine)
    ctx = EX._context(p, None, EX.ExportOptions(frame='rear_axle'))
    models = {cm.cam.uid: cm for cm in ctx.models}
    x, y, psi = (float(a) for a in scene.drive.pose(t))
    Rw = DS._rz(psi)

    def to_ego(P):
        P = np.atleast_2d(P)
        return (P - np.array([x, y, 0.0])) @ Rw
    out = {'t': t, 'cameras': [], 'lidar': np.zeros((0, 4), np.float32), 'radar': [], 'boxes': [], 'road': [],
           'notes': plan['notes'], 'speed_kmh': float(scene.drive.speed(t)) * 3.6}
    for c in (c for c in plan['chans'] if c.kind == 'camera'):
        cm = models[c.sensor.uid]
        scale = width / cm.intr.width
        w, h, K, _, _ = _camera_setup(p, cm, scale, st.undistort)
        cs = _camera_sim(p, engine, c.sensor, w, h, st.undistort, K)
        img = cs.render(scene, t - plan['readout'] / 2 - plan['exposure'] / 2, plan['readout'], plan['exposure'],
                        st.rolling_shutter, noise=0.0, wx=wx)
        out['cameras'].append({'tag': c.sensor.tag, 'slot': c.sensor.slot, 'image': img, 'channel': c.channel})
    pts = []
    for c in (c for c in plan['chans'] if c.kind == 'lidar'):
        ls = _lidar_sim(p, engine, c.sensor)
        t0 = math.floor(t / ls.period) * ls.period
        sw = ls.sweep(scene, t0, st.lidar_noise_cm / 100.0, st.drop_self_hits, seed=1, wx=wx)
        # each point at its own firing pose -> world -> the ego frame at t (no motion skew in the picture)
        tc = t0 + sw['t'].astype(float)
        xs, ys, ps_ = scene.drive.pose(tc)
        Pv = sw['xyz'].astype(float) @ ls.R_vs.T + ls.pos
        c_, s_ = np.cos(ps_), np.sin(ps_)
        Pw = np.stack([xs + c_ * Pv[:, 0] - s_ * Pv[:, 1], ys + s_ * Pv[:, 0] + c_ * Pv[:, 1], Pv[:, 2]], 1)
        pts.append(np.c_[to_ego(Pw), sw['intensity']])
    if pts:
        out['lidar'] = np.vstack(pts).astype(np.float32)
    for c in (c for c in plan['chans'] if c.kind == 'radar'):
        rs = _radar_sim(p, engine, c.sensor)
        tk = c.stamps[np.abs(c.stamps - t).argmin()]
        det = rs.scan(scene, float(tk), seed=1, wx=wx)
        q = np.c_[det['x'], det['y'], det['z']] @ rs.R_vs.T + rs.pos
        out['radar'].append({'tag': c.sensor.tag, 'points': q, 'v_r': det['v_r']})
    for b in scene.boxes_at(t, near=(x, y)):
        L, W, H = b['size']
        cy, sy = math.cos(b['yaw']), math.sin(b['yaw'])
        q = np.array([[L / 2, W / 2], [-L / 2, W / 2], [-L / 2, -W / 2], [L / 2, -W / 2]])
        cw = np.c_[b['centre'][0] + q[:, 0] * cy - q[:, 1] * sy, b['centre'][1] + q[:, 0] * sy + q[:, 1] * cy,
                   np.zeros(4)]
        out['boxes'].append({'name': b['name'], 'category': b['category'], 'corners': to_ego(cw)[:, :2],
                             'moving': b['moving']})
    s0, _ = scene.road.to_sd(np.array([x]), np.array([y]))
    ss = np.linspace(float(s0[0]) - 90.0, float(s0[0]) + 90.0, 181)
    rd = scene.road
    for d in sorted({rd.e_lanes, rd.e_kerb, rd.e_pave} - {0.0}):
        for sgn in (1, -1):
            xx, yy, _ = rd.to_world(ss, sgn * d)
            out['road'].append(to_ego(np.c_[xx, yy, np.zeros(len(ss))])[:, :2])
    return out


def _camera_sim(project, engine, cam, w, h, rectified, K):
    vg = engine.vehicle_geom(project)
    key = ('cam', json.dumps(cam.to_dict(), sort_keys=True, default=str),
           json.dumps(project.module_for(cam).to_dict(), default=str),
           json.dumps(project.lens_for(cam).to_dict(), default=str), w, h, bool(rectified),
           None if K is None else tuple(np.round(np.asarray(K).ravel(), 9)), id(vg))
    return _cached(key, lambda: DS.CameraSim(project, engine, cam, w, h, rectified, K), keep=16)


def _lidar_sim(project, engine, s):
    vg = engine.vehicle_geom(project)
    key = ('lidar', json.dumps(s.to_dict(), sort_keys=True), json.dumps(project.sensor_model_for(s).to_dict()), id(vg))
    return _cached(key, lambda: DS.LidarSim(project, engine, s), keep=16)


def _radar_sim(project, engine, s):
    return DS.RadarSim(project, engine, s)


# =========================================================================== output folders
def bag_name(project, variant=None) -> str:
    v = _project_for(project, variant).variant
    return 'vantage_' + EX._slug(f'{project.name}_{v.name}').replace('-', '_')


def _check_dirs(out_dir, name, st):
    """(nuscenes folder, bag folder); ValueError when either holds files this module did not write."""
    nus = os.path.join(out_dir, 'nuscenes')
    bag = os.path.join(out_dir, 'rosbag', name)
    if os.path.exists(out_dir) and not os.path.isdir(out_dir):
        raise ValueError(f'{out_dir} is a file: choose a folder.')
    if st.nuscenes and os.path.isdir(nus) and os.listdir(nus) and not os.path.exists(os.path.join(nus, MARKER)):
        raise ValueError(f'{nus} already holds files Vantage did not write: choose an empty folder.')
    if st.rosbag and os.path.isdir(bag):
        if [f for f in os.listdir(bag) if not (f == 'metadata.yaml' or f.endswith('.mcap'))]:
            raise ValueError(f'{bag} holds other files: choose an empty folder.')
    return nus, bag


def check_out_dir(out_dir, settings: DatasetSettings, project, variant=None) -> None:
    """Raise ValueError (a message for the user) when a generation into out_dir would fail at once: nothing to record,
    no output selected, or a folder holding files this module did not write. Writes nothing."""
    if not (settings.nuscenes or settings.rosbag):
        raise ValueError('Neither nuScenes nor the ROS 2 bag is selected: nothing to write.')
    p = _project_for(project, variant)
    _plan(p, p.variant, settings)
    _check_dirs(os.path.abspath(out_dir), bag_name(project, variant), settings)


def _prepare(out_dir, name, st, created):
    """Make the output folders, clearing only what an earlier run of this module wrote there."""
    nus, bag = _check_dirs(out_dir, name, st)
    os.makedirs(out_dir, exist_ok=True)
    if st.nuscenes:
        if os.path.isdir(nus):
            for sub in ('samples', 'sweeps', 'maps', VERSION):
                shutil.rmtree(os.path.join(nus, sub), ignore_errors=True)
        os.makedirs(os.path.join(nus, VERSION), exist_ok=True)
        with open(os.path.join(nus, MARKER), 'w') as fh:
            fh.write('Written by Vantage (core.dataset); a new run clears samples/, sweeps/, maps/ and '
                     f'{VERSION}/ here.\n')
        created.append(nus)
    if st.rosbag:
        if os.path.isdir(bag):
            shutil.rmtree(bag)
        created.append(bag)
    return nus, bag


def _cleanup(created):
    for path in created:
        if os.path.basename(path) == 'nuscenes':
            for sub in ('samples', 'sweeps', 'maps', VERSION, MARKER):
                q = os.path.join(path, sub)
                if os.path.isdir(q):
                    shutil.rmtree(q, ignore_errors=True)
                elif os.path.exists(q):
                    os.remove(q)
            try:
                os.rmdir(path)
            except OSError:
                pass
        elif os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)


# =========================================================================== nuScenes files
def _pcd_radar(det: dict) -> bytes:
    n = len(det['x'])
    a = np.zeros(max(n, 1), RADAR_DTYPE)
    if n:
        for k in ('x', 'y', 'z', 'rcs', 'vx', 'vy', 'vx_comp', 'vy_comp'):
            a[k] = det[k]
        a['dyn_prop'] = det['dyn_prop']
        a['id'] = np.arange(n)
        a['is_quality_valid'] = 1
        a['ambig_state'] = 3              # unambiguous Doppler
        a['invalid_state'] = 0            # valid
        a['pdh0'] = 1                     # false alarm probability < 25 %
        a['x_rms'] = a['y_rms'] = a['vx_rms'] = a['vy_rms'] = 3
    else:                                 # nuScenes writes an empty scan as one NaN point
        for k in ('x', 'y', 'z', 'rcs', 'vx', 'vy', 'vx_comp', 'vy_comp'):
            a[k] = np.nan
    # a trailing newline: the devkit's reader wants a byte after the last point
    return PCD_HEAD.format(n=len(a)).encode('ascii') + a.tobytes() + b'\n'


def _lidar_bin(sw: dict) -> bytes:
    a = np.empty((len(sw['xyz']), 5), np.float32)
    a[:, :3] = sw['xyz']
    a[:, 3] = sw['intensity']
    a[:, 4] = sw['ring']
    return a.tobytes()


# =========================================================================== generation
def generate(project, engine, settings: DatasetSettings | dict | None, out_dir: str, progress=None, cancel=None,
             variant=None) -> DatasetResult | None:
    """Simulate the drive and write the dataset into out_dir (nuscenes/, rosbag/<name>/, dataset.json, README.md).
    Returns None when cancelled (what was written is removed again). Raises ValueError when there is nothing to
    record or out_dir holds files this module did not write."""
    t_start = time.perf_counter()
    st = settings if isinstance(settings, DatasetSettings) else DatasetSettings.from_dict(
        settings if settings is not None else getattr(project.analysis, 'dataset', None))
    if not (st.nuscenes or st.rosbag):
        raise ValueError('Neither nuScenes nor the ROS 2 bag is selected: nothing to write.')
    p = _project_for(project, variant)
    v = p.variant
    plan = _plan(p, v, st)
    notes = list(plan['notes'])
    if engine is None:
        from .analysis import Engine
        engine = Engine()
    engine.vehicle_geom(p)
    wx = WX.scene_weather(p, st.weather or None, engine)
    if wx is not None:
        notes.append(f'Weather: {wx.condition.label()} (core.weather; typical models).')

    def prog(f, text=''):
        if progress is not None:
            progress(min(max(float(f), 0.0), 1.0), text)

    def stop():
        return cancel is not None and cancel()
    prog(0.0, 'Building the scene')
    scene = scene_for(p, st)
    ctx = EX._context(p, None, EX.ExportOptions(frame='rear_axle'))
    models = {cm.cam.uid: cm for cm in ctx.models}
    RB = None
    if st.rosbag:
        try:
            from . import rosbag as RB
        except ImportError:
            notes.append('ROS 2 bag not written: core.rosbag is not available.')
    out_dir = os.path.abspath(out_dir)
    created = []
    nus, bag_dir = _prepare(out_dir, bag_name(p), st, created)
    try:
        res = _generate(p, v, engine, st, plan, scene, ctx, models, RB, out_dir, nus, bag_dir, notes, prog, stop,
                        t_start, wx)
    except BaseException:
        _cleanup(created)
        raise
    if res is None:
        _cleanup(created)
    return res


def _generate(p, v, engine, st, plan, scene, ctx, models, RB, out_dir, nus, bag_dir, notes, prog, stop, t_start,
              wx=None):
    chans = plan['chans']
    samples = plan['samples']
    readout, exposure = plan['readout'], plan['exposure']
    key_st = dict(st.to_dict(), weather=wx.condition.to_doc()) if wx is not None else \
        {k: v for k, v in st.to_dict().items() if k != 'weather'}          # clear: the tokens as before weather
    run_key = EX._token('vantage', 'dataset', p.name, v.name, json.dumps(key_st, sort_keys=True))
    sample_tok = [EX._token('vantage', 'sample', run_key, _us(t)) for t in samples]
    sample_us = [_us(t) for t in samples]
    # ---- calibrated sensors: the nuscenes_calib export's rows (visible sensors), intrinsics scaled to the images
    calib = dict(EX._nuscenes(ctx))
    cal_rows = {r['sensor_token']: r for r in json.loads(calib['nuscenes/calibrated_sensor.json'])}
    sens_rows = {r['channel']: r for r in json.loads(calib['nuscenes/sensor.json'])}
    sims, cam_setup = {}, {}
    sensor_tab, cal_tab = [], []
    for c in chans:
        if c.kind == 'camera':
            cm = models[c.sensor.uid]
            w, h, K, dmodel, dcoef = _camera_setup(p, cm, st.image_scale, st.undistort)
            cam_setup[c.channel] = (w, h, K, dmodel, dcoef)
            sims[c.channel] = DS.CameraSim(p, engine, c.sensor, w, h, st.undistort, K)
        elif c.kind == 'lidar':
            sims[c.channel] = DS.LidarSim(p, engine, c.sensor)
        elif c.kind == 'radar':
            sims[c.channel] = DS.RadarSim(p, engine, c.sensor)
        if c.channel:
            srow = dict(sens_rows[c.info['export_channel']], channel=c.channel)
            row = dict(cal_rows[srow['token']])
            if c.kind == 'camera':
                w, h, K, dmodel, dcoef = cam_setup[c.channel]
                row['camera_intrinsic'] = [[EX._r(K[i, j], 6) for j in range(3)] for i in range(3)]
                vd = dict(row.get('vantage_distortion', {}))
                vd['image_size'] = [w, h]
                vd['image_scale'] = [EX._r(w / cm.intr.width, 9), EX._r(h / cm.intr.height, 9)]
                if st.undistort and dmodel == 'plumb_bob' and vd.get('model') == 'kannala_brandt':
                    vd['rectified_from'] = {'model': 'kannala_brandt', 'coefficients': vd.get('coefficients', [])}
                    vd['model'] = 'none (rectified pinhole images)'
                    vd['coefficients'] = []
                row['vantage_distortion'] = vd
            sensor_tab.append(srow)
            cal_tab.append(row)
            c.info['cs_token'] = row['token']
    # ---- sample_data tokens and chains
    for c in chans:
        if not c.channel:
            continue
        c.info['tokens'] = [EX._token('vantage', 'sample_data', run_key, c.channel, _us(t)) for t in c.stamps]
        nxt = np.searchsorted(np.array(sample_us), [_us(t) for t in c.stamps], 'left')
        c.info['sweep_sample'] = np.minimum(nxt, len(samples) - 1)
    # ---- writers
    logname = 'vantage-' + _dt.datetime.fromtimestamp(EPOCH_US / 1e6, _dt.timezone.utc).strftime('%Y-%m-%d-%H-%M-%S') \
        + '+0000'
    bag = None
    if st.rosbag and RB is not None:
        bag = RB.BagWriter(bag_dir, overwrite=True)
        bag.add_topic('/tf_static', 'tf2_msgs/msg/TFMessage')
        bag.add_topic('/tf', 'tf2_msgs/msg/TFMessage')
        bag.add_topic('/odom', 'nav_msgs/msg/Odometry')
        bag.add_topic('/ground_truth/markers', 'visualization_msgs/msg/MarkerArray')
        bag.add_topic('/ground_truth/boxes', 'std_msgs/msg/String')
        for c in chans:
            if c.kind == 'camera':
                bag.add_topic(c.topic + '/image_raw/compressed', 'sensor_msgs/msg/CompressedImage')
                bag.add_topic(c.topic + '/camera_info', 'sensor_msgs/msg/CameraInfo')
            elif c.kind == 'lidar':
                bag.add_topic(c.topic + '/points', 'sensor_msgs/msg/PointCloud2')
            elif c.kind == 'radar':
                bag.add_topic(c.topic + '/detections', 'sensor_msgs/msg/PointCloud2')
            elif c.kind == 'imu':
                bag.add_topic(c.topic + '/imu', 'sensor_msgs/msg/Imu')
            elif c.kind == 'gnss':
                bag.add_topic(c.topic + '/fix', 'sensor_msgs/msg/NavSatFix')
        bag.write('/tf_static', _ns(0.0), {'transforms': _static_tfs(ctx, chans, RB, _ns(0.0))})
    # ---- every event in time order (ties: cameras first, they are the slowest and run on the pool)
    order_kind = {'camera': 0, 'lidar': 1, 'radar': 2, 'imu': 3, 'gnss': 4, 'tf': 5, 'gt': 6}
    events = []
    for ci, c in enumerate(chans):
        for k, ts in enumerate(c.stamps):
            events.append((float(ts), order_kind[c.kind], ci, k))
    if bag is not None:
        for k in range(int(math.floor(st.duration_s * TF_HZ + 1e-9)) + 1):
            events.append((k / TF_HZ, order_kind['tf'], -1, k))
        gt_times = next((c.stamps for c in chans if c.kind == 'lidar'), None)
        if gt_times is None:
            gt_times = np.arange(int(math.floor(st.duration_s * GT_HZ + 1e-9)) + 1) / GT_HZ
        for k, ts in enumerate(gt_times):
            events.append((float(ts), order_kind['gt'], -2, k))
    events.sort()
    cost = []
    for ts, ok, ci, k in events:
        if ci < 0:
            cost.append(0.002)
            continue
        c = chans[ci]
        if c.kind == 'camera':
            w, h = cam_setup[c.channel][:2]
            cost.append(w * h / 1e6 * SEC_PER_MPX / WORKERS)
        elif c.kind == 'lidar':
            cost.append(sims[c.channel].C * sims[c.channel].J / 1e6 * SEC_PER_MRAY)
        elif c.kind == 'radar':
            cost.append(SEC_PER_SCAN)
        else:
            cost.append(0.001)
    total_cost = sum(cost) + len(samples) * SEC_PER_SAMPLE + 0.5
    done_cost = 0.0
    # IMU errors (constant bias per unit), GNSS noise
    imu_err = {}
    for ci, c in enumerate(chans):
        if c.kind == 'imu':
            imu_err[ci] = DS.imu_errors(st.seed * 1000 + ci, len(c.stamps))
    ins = next((p.sensor_model_for(s) for s in v.sensors if s.visible and s.kind == 'imu'
                and p.sensor_model_for(s).pos_sigma_cm > 0), None)
    sig_h = (ins.pos_sigma_cm / 100.0) if ins is not None else 0.02
    gnss_rng = np.random.default_rng([int(st.seed), 4242])
    # ---- outputs gathered while writing
    sd_rows, ego_rows = [], []
    counts_lidar, counts_radar = {}, {}            # (channel, k) -> {obj: points} for keyframes
    per_chan = {c.channel or c.topic: {'kind': c.kind, 'tag': c.sensor.tag, 'channel': c.channel,
                                        'topic': c.topic, 'frames': 0, 'keyframes': 0, 'bytes': 0} for c in chans}
    nbytes = {'nuscenes': 0, 'rosbag': 0}

    def write_file(rel, data):
        path = os.path.join(nus, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as fh:
            fh.write(data)
        nbytes['nuscenes'] += len(data)

    def sd_record(c, k, ext, data, width=0, height=0):
        """Write one sample_data file and its rows (sample_data, ego_pose)."""
        ts = float(c.stamps[k])
        us = _us(ts)
        key = plan['keys'][c.channel].get(k)
        is_key = key is not None
        folder = 'samples' if is_key else 'sweeps'
        rel = f'{folder}/{c.channel}/{logname}__{c.channel}__{us}.{ext}'
        write_file(rel, data)
        tok = c.info['tokens'][k]
        x, y, psi = (float(a) for a in scene.drive.pose(ts))
        ego_rows.append({'token': tok, 'timestamp': us, 'rotation': _yaw_quat(psi),
                         'translation': [EX._r(x, 6), EX._r(y, 6), 0.0]})
        sd_rows.append({'token': tok, 'sample_token': sample_tok[key if is_key else int(c.info['sweep_sample'][k])],
                        'ego_pose_token': tok, 'calibrated_sensor_token': c.info['cs_token'], 'timestamp': us,
                        'fileformat': ext.split('.')[0], 'is_key_frame': is_key, 'height': int(height),
                        'width': int(width), 'filename': rel,
                        'prev': c.info['tokens'][k - 1] if k > 0 else '',
                        'next': c.info['tokens'][k + 1] if k + 1 < len(c.stamps) else ''})
        per_chan[c.channel]['keyframes'] += int(is_key)
        return is_key

    def bag_write(topic, t_ns, msg):
        bag.write(topic, t_ns, msg)

    # camera frames render on a small pool, ahead of the writer; everything is written in event order
    workers = max(1, min(WORKERS, os.cpu_count() or 1))
    pool = cf.ThreadPoolExecutor(max_workers=workers)
    futures = {}
    cam_events = [i for i, e in enumerate(events) if e[2] >= 0 and chans[e[2]].kind == 'camera']
    next_cam = [0]

    def render_job(c, k):
        sim = sims[c.channel]
        img = sim.render(scene, float(c.times[k]), readout, exposure, st.rolling_shutter, noise=2.0,
                         seed=[int(st.seed), zlib.crc32(c.channel.encode()), int(k)], wx=wx)
        import cv2
        ok, buf = cv2.imencode('.jpg', np.ascontiguousarray(img[..., ::-1]),
                               [int(cv2.IMWRITE_JPEG_QUALITY), int(st.jpeg_quality)])
        if not ok:
            raise RuntimeError('JPEG encoding failed')
        return buf.tobytes()

    def feed(upto):
        while next_cam[0] < len(cam_events) and len(futures) < 2 * workers and cam_events[next_cam[0]] <= upto:
            i = cam_events[next_cam[0]]
            _, _, ci, k = events[i]
            futures[i] = pool.submit(render_job, chans[ci], k)
            next_cam[0] += 1
    try:
        for i, (ts, _, ci, k) in enumerate(events):
            if stop():
                for f in futures.values():
                    f.cancel()
                if bag is not None:
                    bag.close()
                return None
            feed(i + 400)
            t_ns = _ns(ts)
            if ci == -1:                                       # /tf and /odom
                x, y, psi = (float(a) for a in scene.drive.pose(ts))
                kin = scene.drive.kinematics(ts)
                T = np.eye(4)
                T[:3, :3] = DS._rz(psi)
                T[:3, 3] = (x, y, 0.0)
                bag_write('/tf', t_ns, {'transforms': [RB.transform_stamped(T, t_ns, 'odom', 'base_link')]})
                bag_write('/odom', t_ns, {
                    'header': RB.header(t_ns, 'odom'), 'child_frame_id': 'base_link',
                    'pose': {'pose': {'position': {'x': x, 'y': y, 'z': 0.0}, 'orientation': RB.quat_msg(T[:3, :3])}},
                    'twist': {'twist': {'linear': {'x': float(kin['v']), 'y': 0.0, 'z': 0.0},
                                        'angular': {'x': 0.0, 'y': 0.0, 'z': float(kin['omega'])}}}})
            elif ci == -2:                                     # ground truth for viewers
                x, y, _ = (float(a) for a in scene.drive.pose(ts))
                boxes = scene.boxes_at(ts, near=(x, y))
                markers = [RB.box_marker(b['centre'], DS._rz(b['yaw']), b['size'], t_ns, 'odom', 'ground_truth',
                                         b['id'], MARKER_RGBA.get(b['category'], (0.8, 0.8, 0.8, 0.5)),
                                         int(1.5e9 / GT_HZ)) for b in boxes]
                bag_write('/ground_truth/markers', t_ns, {'markers': markers})
                doc = {'stamp_ns': t_ns, 'frame_id': 'odom', 'boxes': [
                    {'id': b['id'], 'instance_token': EX._token('vantage', 'instance', run_key, b['id']),
                     'category': b['category'], 'name': b['name'], 'centre': [round(q, 4) for q in b['centre']],
                     'size_lwh': list(b['size']), 'yaw': round(b['yaw'], 6),
                     'velocity': [round(q, 4) for q in b['velocity']]} for b in boxes]}
                bag_write('/ground_truth/boxes', t_ns, {'data': json.dumps(doc, separators=(',', ':'))})
            else:
                c = chans[ci]
                pc = per_chan[c.channel or c.topic]
                pc['frames'] += 1
                if c.kind == 'camera':
                    jpg = futures.pop(i).result()
                    pc['bytes'] += len(jpg)
                    w, h, K, dmodel, dcoef = cam_setup[c.channel]
                    if st.nuscenes:
                        sd_record(c, k, 'jpg', jpg, w, h)
                    if bag is not None:
                        frame = f'{c.tag}_optical_frame'
                        bag_write(c.topic + '/image_raw/compressed', t_ns, RB.compressed_image(jpg, t_ns, frame))
                        bag_write(c.topic + '/camera_info', t_ns, RB.camera_info(w, h, dmodel, dcoef, K, None, None,
                                                                                  t_ns, frame))
                elif c.kind == 'lidar':
                    sim = sims[c.channel]
                    sw = sim.sweep(scene, float(c.times[k]), st.lidar_noise_cm / 100.0, st.drop_self_hits,
                                   seed=[int(st.seed), 104729, ci, k], wx=wx)
                    pc['bytes'] += len(sw['xyz']) * 20
                    if st.nuscenes:
                        if sd_record(c, k, 'pcd.bin', _lidar_bin(sw)):
                            counts_lidar[(c.channel, k)] = _count(sw['obj'])
                    elif plan['keys'][c.channel].get(k) is not None:
                        counts_lidar[(c.channel, k)] = _count(sw['obj'])
                    if bag is not None:
                        a = np.empty(len(sw['xyz']), LIDAR_PC2)
                        a['x'], a['y'], a['z'] = sw['xyz'][:, 0], sw['xyz'][:, 1], sw['xyz'][:, 2]
                        a['intensity'], a['ring'], a['t'] = sw['intensity'], sw['ring'], sw['t']
                        bag_write(c.topic + '/points', t_ns, RB.pointcloud2(a, t_ns, f'{c.tag}_link'))
                elif c.kind == 'radar':
                    det = sims[c.channel].scan(scene, float(c.times[k]), seed=[int(st.seed), 15485863, ci, k],
                                               wx=wx)
                    pc['bytes'] += len(det['x']) * 43
                    if st.nuscenes:
                        if sd_record(c, k, 'pcd', _pcd_radar(det)):
                            counts_radar[(c.channel, k)] = _count(det['obj'])
                    elif plan['keys'][c.channel].get(k) is not None:
                        counts_radar[(c.channel, k)] = _count(det['obj'])
                    if bag is not None:
                        a = np.empty(len(det['x']), RADAR_PC2)
                        for f in ('x', 'y', 'z', 'v_r', 'rcs'):
                            a[f] = det[f]
                        bag_write(c.topic + '/detections', t_ns, RB.pointcloud2(a, t_ns, f'{c.tag}_link'))
                elif c.kind == 'imu' and bag is not None:
                    bag_write(c.topic + '/imu', t_ns, _imu_msg(RB, scene, c, k, ts, t_ns, imu_err[ci], p, v,
                                                               [int(st.seed), ci, k]))
                elif c.kind == 'gnss' and bag is not None:
                    P = DS.antenna_world(scene.drive, c.sensor, ts)[0]
                    P = P + gnss_rng.normal(0, 1, 3) * np.array([sig_h, sig_h, 2 * sig_h])
                    lat, lon, alt = DS.enu_to_geodetic(P, st.origin_lat, st.origin_lon, st.origin_alt)
                    bag_write(c.topic + '/fix', t_ns, {
                        'header': RB.header(t_ns, f'{c.tag}_link'),
                        'status': {'status': 2, 'service': 15},          # GBAS (RTK) fix; GPS, GLONASS, BeiDou, Galileo
                        'latitude': float(lat[0]), 'longitude': float(lon[0]), 'altitude': float(alt[0]),
                        'position_covariance': [sig_h ** 2, 0.0, 0.0, 0.0, sig_h ** 2, 0.0, 0.0, 0.0,
                                                (2 * sig_h) ** 2],
                        'position_covariance_type': 2})
            done_cost += cost[i]
            if i % 8 == 0 or i == len(events) - 1:
                c = chans[ci] if ci >= 0 else None
                prog(0.95 * done_cost / total_cost,
                     f'{c.sensor.tag} · {ts:.2f} s' if c is not None else f'{ts:.2f} s')
    except BaseException:
        if bag is not None:
            try:
                bag.close()
            except Exception:
                pass
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    # ---- annotations, instances and the tables
    if stop():
        if bag is not None:
            bag.close()
        return None
    prog(0.95, 'Annotations')
    ann_rows, inst = _annotations(scene, chans, samples, sample_tok, run_key, plan, sims, counts_lidar, counts_radar,
                                  stop)
    if ann_rows is None:
        if bag is not None:
            bag.close()
        return None
    bag_res = bag.close() if bag is not None else None
    if bag_res is not None:
        nbytes['rosbag'] = int(bag_res['bytes']) + os.path.getsize(os.path.join(bag_dir, 'metadata.yaml'))
        for c in chans:
            pc = per_chan[c.channel or c.topic]
            if c.kind == 'camera':
                pc['messages'] = 2 * len(c.stamps)
            else:
                pc['messages'] = len(c.stamps)
    counts = {'samples': len(samples), 'sample_data': len(sd_rows), 'ego_poses': len(ego_rows),
              'annotations': len(ann_rows), 'instances': len(inst), 'bag_messages': bag_res['messages'] if bag_res
              else 0}
    if st.nuscenes:
        prog(0.97, 'nuScenes tables')
        _write_tables(p, v, st, scene, nus, logname, run_key, samples, sample_tok, sample_us, sensor_tab, cal_tab,
                      sd_rows, ego_rows, ann_rows, inst, write_file)
    res = DatasetResult(st, out_dir, nus if st.nuscenes else '', bag_dir if bag_res is not None else '', p.name,
                        v.name, v.uid, list(per_chan.values()), counts,
                        {'nuscenes': nbytes['nuscenes'], 'rosbag': nbytes['rosbag'],
                         'total': nbytes['nuscenes'] + nbytes['rosbag']},
                        _scene_doc(scene, st, plan, wx), 0.0, notes, bag_res,
                        None if wx is None else wx.condition.to_doc())
    res.seconds = time.perf_counter() - t_start
    export(res, p, out_dir)
    prog(1.0, 'Done')
    return res


def _count(obj) -> dict:
    ids, n = np.unique(np.asarray(obj)[np.asarray(obj) >= 0], return_counts=True)
    return {int(i): int(k) for i, k in zip(ids, n)}


def _static_tfs(ctx, chans, RB, t_ns) -> list:
    """base_link -> every camera body and sensor; camera body -> its optical frame (the URDF export's links)."""
    out = []
    for c in chans:
        if c.kind == 'camera':
            cm = next(m for m in ctx.models if m.cam.uid == c.sensor.uid)
            T = np.eye(4)
            T[:3, :3] = cm.R
            T[:3, 3] = c.sensor.pos() - ctx.origin
            out.append(RB.transform_stamped(T, t_ns, 'base_link', f'{c.tag}_link'))
            To = np.eye(4)
            To[:3, :3] = BODY_TO_OPTICAL.T
            out.append(RB.transform_stamped(To, t_ns, f'{c.tag}_link', f'{c.tag}_optical_frame'))
        else:
            out.append(RB.transform_stamped(EX._T_vehicle_sensor(ctx, c.sensor), t_ns, 'base_link', f'{c.tag}_link'))
    return out


def _imu_msg(RB, scene, c, k, ts, t_ns, err, p, v, seed) -> dict:
    ba, bg, na, ng = err
    f, w = DS.imu_truth(scene.drive, c.sensor, ts)
    f = f[0] + ba + na[k]
    w = w[0] + bg + ng[k]
    m = p.sensor_model_for(c.sensor)
    sa, sg = DS.IMU_NOISE['accel_sigma'], DS.IMU_NOISE['gyro_sigma']
    msg = {'header': RB.header(t_ns, f'{c.tag}_link'),
           'angular_velocity': {'x': float(w[0]), 'y': float(w[1]), 'z': float(w[2])},
           'angular_velocity_covariance': [sg * sg, 0, 0, 0, sg * sg, 0, 0, 0, sg * sg],
           'linear_acceleration': {'x': float(f[0]), 'y': float(f[1]), 'z': float(f[2])},
           'linear_acceleration_covariance': [sa * sa, 0, 0, 0, sa * sa, 0, 0, 0, sa * sa]}
    if m.antennas > 0:                     # a GNSS/INS reports its attitude in the world (ENU) frame
        _, _, psi = (float(a) for a in scene.drive.pose(ts))
        att = math.radians(max(m.attitude_sigma_deg, 0.01))
        gn = [s for s in v.sensors if s.visible and s.kind == 'gnss']
        base = float(np.linalg.norm(gn[0].pos() - gn[1].pos())) if len(gn) >= 2 else 0.0
        hs = m.heading_sigma(base) if m.antennas >= 2 and base > 0 else float('nan')
        head = math.radians(hs) if math.isfinite(hs) else math.radians(0.5)
        rng = np.random.default_rng(seed)
        R = DS._rz(psi + rng.normal(0, head)) @ rot_zyx(0.0, math.degrees(rng.normal(0, att)),
                                                         math.degrees(rng.normal(0, att))) @ DS.SN.rotation(c.sensor)
        msg['orientation'] = RB.quat_msg(R)
        msg['orientation_covariance'] = [att * att, 0, 0, 0, att * att, 0, 0, 0, head * head]
    else:
        msg['orientation'] = {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 0.0}
        msg['orientation_covariance'] = [-1.0, 0, 0, 0, 0, 0, 0, 0, 0]          # REP 145: no orientation
    return msg


def _attribute(b) -> str:
    sp = math.hypot(b['velocity'][0], b['velocity'][1])
    cat = b['category']
    if cat.startswith('vehicle.bicycle'):
        return 'cycle.with_rider'
    if cat.startswith('human.pedestrian'):
        return 'pedestrian.moving' if sp > 0.3 else 'pedestrian.standing'
    if not b['moving']:
        return 'vehicle.parked'
    return 'vehicle.moving' if sp > 0.3 else 'vehicle.stopped'


def _annotations(scene, chans, samples, sample_tok, run_key, plan, sims, counts_lidar, counts_radar, stop):
    """sample_annotation rows (with visibility and point counts) and instance rows."""
    cams = [sims[c.channel] for c in chans if c.kind == 'camera']
    keyk = {}
    for c in chans:
        if c.kind in ('lidar', 'radar'):
            for k, i in plan['keys'][c.channel].items():
                keyk.setdefault(i, []).append((c.kind, c.channel, k))
    rows, by_inst = [], {}
    for i, ts in enumerate(samples):
        if stop():
            return None, None
        x, y, _ = (float(a) for a in scene.drive.pose(float(ts)))
        nl, nr = {}, {}
        for kind, ch, k in keyk.get(i, ()):
            src = counts_lidar if kind == 'lidar' else counts_radar
            dst = nl if kind == 'lidar' else nr
            for oid, n in src.get((ch, k), {}).items():
                dst[oid] = dst.get(oid, 0) + n
        for b in scene.boxes_at(float(ts), near=(x, y)):
            vis = DS.visibility(scene, float(ts), b, cams) if cams else 0.0
            L, W, H = b['size']
            tok = EX._token('vantage', 'sample_annotation', run_key, b['id'], _us(float(ts)))
            inst = EX._token('vantage', 'instance', run_key, b['id'])
            rows.append({'token': tok, 'sample_token': sample_tok[i], 'instance_token': inst,
                         'visibility_token': DS.visibility_bin(vis),
                         'attribute_tokens': [EX._token('vantage', 'attribute', _attribute(b))],
                         'translation': [EX._r(q, 6) for q in b['centre']], 'size': [W, L, H],
                         'rotation': _yaw_quat(b['yaw']), 'prev': '', 'next': '',
                         'num_lidar_pts': int(nl.get(b['id'], 0)), 'num_radar_pts': int(nr.get(b['id'], 0))})
            by_inst.setdefault(inst, (b['category'], []))[1].append(len(rows) - 1)
    inst_rows = []
    for inst, (cat, idx) in by_inst.items():
        for a, b in zip(idx[:-1], idx[1:]):
            rows[a]['next'] = rows[b]['token']
            rows[b]['prev'] = rows[a]['token']
        inst_rows.append({'token': inst, 'category_token': EX._token('vantage', 'category', cat),
                          'nbr_annotations': len(idx), 'first_annotation_token': rows[idx[0]]['token'],
                          'last_annotation_token': rows[idx[-1]]['token']})
    return rows, inst_rows


def _map_png(scene) -> tuple[bytes, tuple]:
    """The drivable area (the carriageway with its parking strips or shoulders) as an 8-bit PNG at MAP_RES, its
    bottom-left pixel corner at the world origin (as the devkit's MapMask reads it)."""
    import cv2
    x1, y1 = scene.extent[2], scene.extent[3]
    W, H = int(math.ceil(x1 / MAP_RES)), int(math.ceil(y1 / MAP_RES))
    img = np.zeros((H, W), np.uint8)
    rd = scene.road
    s = np.arange(-150.0, rd.L1 + rd.Larc + 1500.0, 0.5)
    xl, yl, _ = rd.to_world(s, rd.e_kerb)
    xr, yr, _ = rd.to_world(s, -rd.e_kerb)
    keep = (np.minimum(xl, xr) > -60) & (np.maximum(xl, xr) < x1 + 60) & (np.minimum(yl, yr) > -60) & \
        (np.maximum(yl, yr) < y1 + 60)
    if keep.any():
        poly = np.r_[np.c_[xl[keep], yl[keep]], np.c_[xr[keep], yr[keep]][::-1]]
        px = np.c_[poly[:, 0] / MAP_RES, H - poly[:, 1] / MAP_RES]
        cv2.fillPoly(img, [np.round(px * 4).astype(np.int32)], 255, lineType=cv2.LINE_8, shift=2)
    ok, buf = cv2.imencode('.png', img, [int(cv2.IMWRITE_PNG_COMPRESSION), 6])
    return buf.tobytes(), (W, H)


def _write_tables(p, v, st, scene, nus, logname, run_key, samples, sample_tok, sample_us, sensor_tab, cal_tab, sd_rows,
                  ego_rows, ann_rows, inst_rows, write_file):
    log_tok = EX._token('vantage', 'log', run_key)
    scene_tok = EX._token('vantage', 'scene', run_key)
    map_tok = EX._token('vantage', 'map', run_key)
    sample_rows = []
    for i, tok in enumerate(sample_tok):
        sample_rows.append({'token': tok, 'timestamp': sample_us[i], 'prev': sample_tok[i - 1] if i else '',
                            'next': sample_tok[i + 1] if i + 1 < len(sample_tok) else '', 'scene_token': scene_tok})
    png, _ = _map_png(scene)
    write_file(f'maps/{map_tok}.png', png)
    vehicle = p.vehicle.spec_name or p.vehicle.name or 'vehicle'
    tables = {
        'attribute': [{'token': EX._token('vantage', 'attribute', n), 'name': n, 'description': d}
                      for n, d in ATTRIBUTES],
        'calibrated_sensor': cal_tab,
        'category': [{'token': EX._token('vantage', 'category', n), 'name': n, 'description': d}
                     for n, d in CATEGORIES],
        'ego_pose': sorted(ego_rows, key=lambda r: r['timestamp']),
        'instance': inst_rows,
        'log': [{'token': log_tok, 'logfile': logname, 'vehicle': vehicle,
                 'date_captured': _dt.datetime.fromtimestamp(EPOCH_US / 1e6, _dt.timezone.utc).strftime('%Y-%m-%d'),
                 'location': f'vantage-{st.scene}'}],
        'map': [{'token': map_tok, 'log_tokens': [log_tok], 'category': 'semantic_prior',
                 'filename': f'maps/{map_tok}.png'}],
        'sample': sample_rows,
        'sample_annotation': ann_rows,
        'sample_data': sorted(sd_rows, key=lambda r: r['timestamp']),
        'scene': [{'token': scene_tok, 'log_token': log_tok, 'nbr_samples': len(sample_rows),
                   'first_sample_token': sample_tok[0], 'last_sample_token': sample_tok[-1], 'name': 'scene-0001',
                   'description': f'Vantage synthetic {SCENE_LABEL[st.scene].lower()} drive, '
                                  f'{st.speed_kmh:g} km/h, seed {st.seed}; {p.name} / {v.name}'}],
        'sensor': sensor_tab,
        'visibility': [{'token': t, 'level': lv, 'description': d} for t, lv, d in VISIBILITY],
    }
    for name, rows in tables.items():
        write_file(f'{VERSION}/{name}.json', (json.dumps(rows, indent=1) + '\n').encode('utf-8'))


def _scene_doc(scene, st, plan, wx=None) -> dict:
    rd = scene.road
    doc = {'kind': st.scene, 'seed': st.seed, 'duration_s': st.duration_s, 'speed_kmh': st.speed_kmh,
           'ego_travel_m': round(float(scene.drive.travel(st.duration_s)), 3),
           'bend': {'direction': 'left' if rd.turn > 0 else 'right', 'radius_m': rd.R,
                    'angle_deg': round(math.degrees(rd.theta), 3), 'starts_at_m': round(rd.L1, 3)},
           'objects': len(scene.objects), 'annotated_actors': sum(1 for o in scene.objects if o.category),
           'keyframe_hz': plan['keyframe_hz'], 'samples': len(plan['samples']),
           'camera_readout_ms': plan['readout'] * 1000.0, 'camera_exposure_ms': plan['exposure'] * 1000.0,
           'origin': {'lat': st.origin_lat, 'lon': st.origin_lon, 'alt': st.origin_alt, 'datum': 'WGS84'}}
    if wx is not None:                  # clear: the scene as before weather existed
        doc['weather'] = dict(wx.condition.to_doc(), extinction_550nm_per_km=round(wx.sigma_vis * 1000.0, 4),
                              lidar_extinction_per_km=round(wx.sigma_lidar * 1000.0, 4),
                              lidar_false_return_share=round(wx.false_share, 4),
                              radar_attenuation_db_km=round(wx.radar_db_km, 4),
                              lens_drop_share={wx.tags.get(u, u): round(v, 4) for u, v in wx.drops.items()},
                              headlamps_m=[[round(float(q), 3) for q in L] for L in wx.lamps])
    return doc


def run(project, engine=None, settings=None, variant=None, progress=None, cancel=None, out_dir=None):
    """generate() with the module contract's arguments; out_dir defaults to <project folder>/exports/<variant>/dataset
    (a saved project)."""
    if out_dir is None:
        v = _project_for(project, variant).variant
        out_dir = os.path.join(EX.default_out_dir(project, v), 'dataset')
    return generate(project, engine, settings, out_dir, progress, cancel, variant)


# =========================================================================== reports
def _mb(n) -> float:
    return round(n / 1e6, 2)


def summary(res: DatasetResult) -> dict:
    """Compact figures (the assistant, the batch log)."""
    return {'out_dir': res.out_dir, 'nuscenes': res.nuscenes_dir or None, 'rosbag': res.bag_dir or None,
            'variant': res.variant, 'scene': res.settings.scene, 'duration_s': res.settings.duration_s,
            'speed_kmh': res.settings.speed_kmh, 'seed': res.settings.seed,
            'samples': res.counts['samples'], 'sample_data': res.counts['sample_data'],
            'annotations': res.counts['annotations'], 'instances': res.counts['instances'],
            'bag_messages': res.counts['bag_messages'],
            'size_mb': {k: _mb(v) for k, v in res.sizes.items()}, 'seconds': round(res.seconds, 1),
            'channels': [{'channel': c['channel'] or c['topic'], 'kind': c['kind'], 'frames': c['frames'],
                          'keyframes': c['keyframes']} for c in res.channels],
            'notes': list(res.notes)}


def load_snippet(res: DatasetResult) -> str:
    lines = []
    if res.nuscenes_dir:
        lines += ['from nuscenes.nuscenes import NuScenes',
                  f"nusc = NuScenes(version='{VERSION}', dataroot={res.nuscenes_dir!r})"]
    if res.bag_dir:
        lines += [f'ros2 bag info {res.bag_dir}']
    return '\n'.join(lines)


def to_doc(res: DatasetResult, project=None) -> dict:
    """dataset.json: what was written, where, the settings, the scene and the conventions."""
    st = res.settings
    return {
        'schema': SCHEMA, 'project': res.project, 'variant': res.variant,
        'written': {'nuscenes': os.path.relpath(res.nuscenes_dir, res.out_dir) if res.nuscenes_dir else None,
                    'nuscenes_version': VERSION if res.nuscenes_dir else None,
                    'rosbag': os.path.relpath(res.bag_dir, res.out_dir) if res.bag_dir else None},
        'settings': st.to_dict(), 'scene': res.scene, 'counts': res.counts,
        'bytes': res.sizes, 'seconds': round(res.seconds, 2),
        'channels': res.channels,
        'bag_topics': (res.bag or {}).get('topics', {}),
        'conventions': {
            'global_frame': 'world: x east, y north, z up, metres, origin at settings origin_lat/lon/alt (WGS84); '
                            'the nuScenes global frame and the bag\'s odom frame',
            'ego_frame': 'vehicle, ISO 8855: x forward, y left, z up, origin at the rear-axle centre on the ground '
                         '(nuScenes ego, bag base_link)',
            'time': f'microseconds (nuScenes) / nanoseconds (bag) since the Unix epoch; t = 0 of the drive is '
                    f'{EPOCH_US} us, a sync pulse',
            'camera_timestamp': 'exposure centre of the middle row: trigger + readout/2 + exposure/2',
            'lidar_timestamp': 'end of the sweep (nuScenes); points in the sensor frame at their own firing time '
                               '(not motion compensated); bag field t = seconds from the sweep start',
            'lidar_fields': 'nuScenes .pcd.bin float32 x y z intensity ring (ring 0 = the lowest beam); bag '
                            'PointCloud2 x y z intensity (float32) ring (uint16) t (float32)',
            'radar_fields': 'nuScenes .pcd 18 fields (vx, vy: radial velocity decomposed in the radar frame; '
                            '_comp: with the ego motion removed); bag PointCloud2 x y z v_r rcs (dBsm)',
            'images': 'rectified pinhole, camera_intrinsic exact' if st.undistort else
                      'the real lens (fisheye): camera_intrinsic + vantage_distortion (Kannala-Brandt) as the '
                      'nuscenes_calib export, scaled to the image size',
            'bag_frames': '<tag>_link (sensor bodies, REP 103), <tag>_optical_frame (camera images): the URDF '
                          'export\'s link names'},
        'assumptions': ASSUMPTIONS + ('' if not res.weather else ' Weather: ' + WX.ASSUMPTIONS),
        'scene_limits': DS.__doc__.split('Modelling limits:')[1].strip() if 'Modelling limits:' in (DS.__doc__ or '')
        else '',
        'notes': res.notes,
    }


def _readme(res: DatasetResult) -> str:
    st = res.settings
    rows = '\n'.join(f'| {c["channel"] or c["topic"]} | {c["kind"]} | {c["frames"]} | {c["keyframes"]} |'
                     for c in res.channels)
    bag_line = f'- ROS 2 bag: `{os.path.relpath(res.bag_dir, res.out_dir)}/` ({res.counts["bag_messages"]} messages)' \
        if res.bag_dir else '- ROS 2 bag: not written'
    nus_line = (f'- nuScenes: `{os.path.relpath(res.nuscenes_dir, res.out_dir)}/` version `{VERSION}`: '
                f'{res.counts["samples"]} samples, {res.counts["sample_data"]} sample_data, '
                f'{res.counts["annotations"]} annotations of {res.counts["instances"]} instances') \
        if res.nuscenes_dir else '- nuScenes: not written'
    wx_line = (' no\nweather. Not for training.' if not res.weather else
               f'\nweather: {res.weather["label"]} (typical models, core.weather). Not for training.')
    return f"""# Vantage synthetic drive · {res.project} / {res.variant}

A {st.duration_s:g} s {SCENE_LABEL[st.scene].lower()} drive at {st.speed_kmh:g} km/h (seed {st.seed}) recorded by the
planned rig, for testing ingestion, labelling tools and loaders. **Synthetic**: simple geometry and materials,{wx_line}

{nus_line}
{bag_line}

| Channel | Kind | Frames | Keyframes |
|---|---|---:|---:|
{rows}

Load:

```
{load_snippet(res)}
```

Conventions:
- Global frame (nuScenes global, bag `odom`): x east, y north, z up, metres; origin at {st.origin_lat:.6f},
  {st.origin_lon:.6f}, {st.origin_alt:g} m (WGS84). Ego frame (`base_link`): x forward, y left, z up at the rear-axle
  centre on the ground.
- Camera timestamp: the middle row's exposure centre. Lidar: the sweep's end; points are in the sensor frame at their
  own firing time (no motion compensation), bag field `t` = seconds from the sweep start. Radar, IMU, GNSS: the
  measurement time.
- calibrated_sensor rows are the nuscenes_calib export's; camera_intrinsic is scaled to the image size. Channels:
  {'nuScenes names (CAM_FRONT …, LIDAR_TOP, RADAR_FRONT …)' if st.channel_names == 'nuscenes' else
   'the rig tags (CAM_<TAG>, LIDAR_<TAG>, RADAR_<TAG>), as the export'}.
  {'Images are rectified pinhole images: camera_intrinsic is exact.' if st.undistort else
   'Images are the real (fisheye) lens: undistort with vantage_distortion (Kannala-Brandt) before using the 3x3.'}
- Bag frame ids are the URDF export's link names (`<tag>_link`, `<tag>_optical_frame`).

{ASSUMPTIONS}{'' if not res.weather else chr(10) + chr(10) + 'Weather: ' + WX.ASSUMPTIONS}
"""


def files(res: DatasetResult, project=None) -> list:
    """[(name, bytes)]: dataset.json and README.md (the manifest; the data itself is written by generate)."""
    return [('dataset.json', (json.dumps(to_doc(res, project), indent=1, default=str) + '\n').encode('utf-8')),
            ('README.md', _readme(res).encode('utf-8'))]


def export(res: DatasetResult, project, out_dir) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    out = []
    for name, data in files(res, project):
        path = os.path.join(out_dir, name)
        with open(path, 'wb') as fh:
            fh.write(data)
        out.append(path)
    return out
