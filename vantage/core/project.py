"""Project data model and .vantage persistence (zip: project.json + mesh.npz + thumb.png)."""
from __future__ import annotations

import copy
import io
import json
import math
import time
import uuid
import zipfile
from dataclasses import dataclass, field, asdict

import numpy as np

from .lens import Module, Lens, builtin_modules, builtin_lenses
from .mesh import Mesh, Part, Zone
from .sensors import SensorModel, builtin_sensor_models, KINDS as SENSOR_KINDS
from . import credentials as _credentials

FORMAT_VERSION = 1
MAX_CAMERAS = 8
FRAME_PRESETS = {
    'rear_axle': 'ISO 8855 · rear axle',
    'front_axle': 'ISO 8855 · front axle',
    'front_bumper': 'ISO 8855 · front bumper',
    'centre': 'ISO 8855 · vehicle centre',
    'custom': 'Custom…',
}


def new_uid() -> str:
    return uuid.uuid4().hex[:10]


# ---------------------------------------------------------------- entities
@dataclass
class Camera:
    uid: str = field(default_factory=new_uid)
    tag: str = 'C'
    name: str = 'Camera'
    location: str = ''
    slot: int = 0                      # colour slot 0..7 (follows the camera, never the order)
    x: float = 0.0
    y: float = 0.0
    z: float = 1.0
    yaw: float = 0.0
    pitch: float = 30.0
    roll: float = 0.0
    module_id: str = 'sturdecam31'
    lens_id: str = 'h05r2'
    port: int = 0                      # GMSL port 1..8, 0 = unassigned
    sync: str = 'A'
    cable_m: float = 3.0
    standoff_mm: float = 12.0
    bracket: str = 'Flat plate · 25 mm'
    snap_normal: bool = True
    locked: bool = False
    visible: bool = True
    link_uid: str = ''                 # mirrored partner
    route_extra_m: float = 0.0
    route_note: str = ''
    calib: dict = field(default_factory=dict)   # unit calibration (core.calib record); overrides the lens model

    def pos(self):
        return np.array([self.x, self.y, self.z])

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        known = Camera.__dataclass_fields__.keys()
        return Camera(**{k: v for k, v in d.items() if k in known})


@dataclass
class Sensor:
    """A lidar, radar, GNSS antenna or IMU of the rig, in the vehicle frame (core.sensors has the conventions). The
    position is its optical centre (a GNSS antenna's phase centre, an IMU's axis origin); yaw/pitch/roll as a camera's,
    pitch positive down."""
    uid: str = field(default_factory=new_uid)
    tag: str = 'L1'
    name: str = 'Lidar'
    kind: str = 'lidar'                # lidar | radar | gnss | imu
    model_id: str = 'ouster_os1_128'
    x: float = 1.5
    y: float = 0.0
    z: float = 1.6
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    phase_deg: float = 0.0             # spinning lidar: sensor-frame azimuth it faces at the top of each sync period
    port: str = ''                     # network port or link on the recorder
    visible: bool = True
    locked: bool = False

    def pos(self):
        return np.array([self.x, self.y, self.z])

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        known = Sensor.__dataclass_fields__.keys()
        s = Sensor(**{k: v for k, v in d.items() if k in known})
        s.kind = s.kind if s.kind in SENSOR_KINDS else 'lidar'
        for k in ('x', 'y', 'z', 'yaw', 'pitch', 'roll', 'phase_deg'):
            try:
                v = float(getattr(s, k))
            except (TypeError, ValueError):
                v = float('nan')
            setattr(s, k, v if math.isfinite(v) else getattr(Sensor(), k))
        s.phase_deg %= 360.0
        return s


def mirror_props(src: Camera, dst: Camera, keys):
    for k in keys:
        if k == 'y':
            dst.y = -src.y
        elif k == 'yaw':
            dst.yaw = -src.yaw
        elif k == 'roll':
            dst.roll = -src.roll
        elif k in ('x', 'z', 'pitch', 'module_id', 'lens_id', 'standoff_mm', 'bracket', 'snap_normal', 'cable_m'):
            # not the trigger line ('sync'): partners on opposite sides are swept at different times
            setattr(dst, k, getattr(src, k))


@dataclass
class Variant:
    uid: str = field(default_factory=new_uid)
    name: str = 'Variant'
    note: str = ''
    cameras: list = field(default_factory=list)
    sensors: list = field(default_factory=list)     # lidars, radars, GNSS antennas and IMUs (Sensor)
    seams: dict = field(default_factory=dict)       # 'uidA|uidB' -> angle deg (manual)
    blend_m: float = 0.20
    created: float = field(default_factory=time.time)

    def cam(self, uid):
        for c in self.cameras:
            if c.uid == uid:
                return c
        return None

    def by_tag(self, tag):
        for c in self.cameras:
            if c.tag == tag:
                return c
        return None

    def sensor(self, uid):
        for s in self.sensors:
            if s.uid == uid:
                return s
        return None

    def tags(self) -> set:
        """Every tag in use, cameras and sensors."""
        return {c.tag for c in self.cameras} | {s.tag for s in self.sensors}

    def free_slot(self):
        used = {c.slot for c in self.cameras}
        for i in range(MAX_CAMERAS):
            if i not in used:
                return i
        return 0

    def to_dict(self):
        return {'uid': self.uid, 'name': self.name, 'note': self.note, 'cameras': [c.to_dict() for c in self.cameras],
                'sensors': [s.to_dict() for s in self.sensors],
                'seams': dict(self.seams), 'blend_m': self.blend_m, 'created': self.created}

    @staticmethod
    def from_dict(d):
        v = Variant(uid=d.get('uid', new_uid()), name=d.get('name', 'Variant'), note=d.get('note', ''),
                    cameras=[Camera.from_dict(c) for c in d.get('cameras', [])],
                    sensors=[Sensor.from_dict(x) for x in d.get('sensors', [])], seams=dict(d.get('seams', {})),
                    blend_m=float(d.get('blend_m', 0.2)), created=float(d.get('created', time.time())))
        return v


@dataclass
class ROI:
    mode: str = 'offset'               # 'offset' | 'polygon'
    distance: float = 2.0
    measured_from: str = 'outline'     # 'outline' | 'bbox'
    corners: str = 'round'             # 'round' | 'square'
    exclude_under_body: bool = True
    polygon: list = field(default_factory=list)   # [(x, y), ...]

    def to_dict(self):
        d = asdict(self)
        d['polygon'] = [list(p) for p in self.polygon]
        return d

    @staticmethod
    def from_dict(d):
        r = ROI(**{k: v for k, v in d.items() if k in ROI.__dataclass_fields__})
        r.polygon = [tuple(p) for p in r.polygon]
        return r


@dataclass
class Target:
    uid: str = field(default_factory=new_uid)
    name: str = 'T1'
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    pattern: str = 'checkerboard'      # checkerboard | charuco | aprilgrid
    rows: int = 6
    cols: int = 6
    square_mm: float = 100.0

    @property
    def size_m(self):
        return (self.cols * self.square_mm / 1000.0, self.rows * self.square_mm / 1000.0)

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        return Target(**{k: v for k, v in d.items() if k in Target.__dataclass_fields__})


@dataclass
class RigPart:
    """A solid of the data-collection rig on the vehicle (roof rack, sensor mast, crossbar, enclosure), in the vehicle
    frame. It blocks cameras like the body does (occludes) and cameras can be placed on it (mountable)."""
    uid: str = field(default_factory=new_uid)
    name: str = 'Rig part'
    kind: str = 'box'                  # box | cylinder (vertical) | bar (horizontal, along y before yaw)
    x: float = 0.0                     # centre, m
    y: float = 0.0
    z: float = 1.6
    length: float = 0.4                # box: along x; bar: its length
    width: float = 0.4                 # box: along y
    height: float = 0.2                # box, cylinder: along z
    radius: float = 0.05               # cylinder, bar
    yaw: float = 0.0                   # degrees about z
    occludes: bool = True
    mountable: bool = True
    visible: bool = True

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        return RigPart(**{k: v for k, v in d.items() if k in RigPart.__dataclass_fields__})


@dataclass
class Obstacle:
    uid: str = field(default_factory=new_uid)
    name: str = 'Pole'
    x: float = 0.0
    y: float = 0.0
    height: float = 1.0
    radius: float = 0.05

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        return Obstacle(**{k: v for k, v in d.items() if k in Obstacle.__dataclass_fields__})


@dataclass
class ECU:
    name: str = 'Jetson AGX Orin'
    location: str = 'Centre console, under the rear seat'
    x: float = 1.00
    y: float = 0.0
    z: float = 0.35
    ports: int = 8
    route_factor: float = 1.08          # routing slack over the Manhattan length
    service_loop_m: float = 0.25
    link_gbps: float = 6.0              # GMSL2 forward channel per camera link
    csi_gbps_per_deser: float = 10.0    # deserializer CSI-2 output: 4 lanes x 2.5 Gbps
    bpp: int = 16                       # YUV 4:2:2 output
    overhead: float = 1.10              # blanking and protocol overhead on the pixel payload

    def deserializer(self, port: int) -> str:
        if port <= 0:
            return '—'
        return f"deserializer {'A' if port <= 4 else 'B'}, link {((port - 1) % 4) + 1}"

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        return ECU(**{k: v for k, v in d.items() if k in ECU.__dataclass_fields__})


@dataclass
class AnalysisSettings:
    cell_m: float = 0.05
    min_density: float = 3.0
    target_density: float = 5.0
    live: bool = True
    heatmap: str = 'none'              # none | coverage | density | redundancy
    obstacle_mode: str = 'ground'      # ground | pole05 | pole10 | child
    tol_deg: float = 5.0
    tol_mm: float = 100.0
    load_cases: list = field(default_factory=list)   # [[name, front drop mm, rear drop mm], ...]
    bowl: dict = field(default_factory=dict)          # 3D surround-view bowl shape (core.bowl.BowlShape fields)
    volume: dict = field(default_factory=dict)        # blind volume study (core.volume.VolumeSettings fields)
    guidelines: dict = field(default_factory=dict)    # parking guidelines (core.guidelines.GuidelineSettings fields)
    stitched: dict = field(default_factory=dict)      # standing objects in the stitched view (core.stitched_objects.Settings)
    assistant: dict = field(default_factory=dict)     # the assistant's project brief: {'brief': text}
    perception: dict = field(default_factory=dict)    # perception range study (core.perception.PerceptionSettings)
    sensor_coverage: dict = field(default_factory=dict)   # lidar and radar coverage (core.sensor_coverage)
    timing: dict = field(default_factory=dict)        # sync and timing plan (core.timing)
    recording: dict = field(default_factory=dict)     # recording budget (core.recording)
    gnss: dict = field(default_factory=dict)          # GNSS and IMU study (core.gnss)
    power: dict = field(default_factory=dict)         # power budget (core.power)
    exposure: dict = field(default_factory=dict)      # sun glare and dirt exposure (core.exposure)
    brackets: dict = field(default_factory=dict)      # mount bracket generator (core.brackets)
    dataset: dict = field(default_factory=dict)       # synthetic sample dataset (core.dataset)
    lidar_calib: dict = field(default_factory=dict)   # lidar-camera calibration (core.lidar_calib)
    drivelog: dict = field(default_factory=dict)      # drive-log check (core.drivelog)
    requirements: dict = field(default_factory=dict)  # requirements scorecard (core.requirements)

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        return AnalysisSettings(**{k: v for k, v in d.items() if k in AnalysisSettings.__dataclass_fields__})


@dataclass
class TargetSettings:
    pattern: str = 'checkerboard'
    rows: int = 6
    cols: int = 6
    square_mm: float = 100.0
    cams_per_target: int = 2
    min_density: float = 3.0

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        return TargetSettings(**{k: v for k, v in d.items() if k in TargetSettings.__dataclass_fields__})


@dataclass
class PhotoFrame:
    name: str
    pose: list | None = None           # 4x4 world->camera (OpenCV optical), row-major
    points: list = field(default_factory=list)   # [(landmark, u, v)]
    rms: float = 0.0

    def to_dict(self):
        return {'name': self.name, 'pose': self.pose, 'points': [list(p) for p in self.points], 'rms': self.rms}

    @staticmethod
    def from_dict(d):
        return PhotoFrame(d['name'], d.get('pose'), [tuple(p) for p in d.get('points', [])], float(d.get('rms', 0)))


@dataclass
class PhotoSet:
    source: str = ''                   # folder or .zip path
    frames: list = field(default_factory=list)
    width: int = 0
    height: int = 0
    f_px: float = 0.0
    cx: float = 0.0
    cy: float = 0.0
    k1: float = 0.0
    rms: float = 0.0
    method: str = ''
    solved_at: str = ''
    landmarks: dict = field(default_factory=dict)   # name -> [x, y, z] vehicle frame
    current: int = 0

    def to_dict(self):
        d = asdict(self)
        d['frames'] = [f.to_dict() for f in self.frames]
        return d

    @staticmethod
    def from_dict(d):
        p = PhotoSet(**{k: v for k, v in d.items() if k in PhotoSet.__dataclass_fields__ and k != 'frames'})
        p.frames = [PhotoFrame.from_dict(f) for f in d.get('frames', [])]
        return p


@dataclass
class Vehicle:
    name: str = 'Vehicle'
    spec_name: str = ''
    source: str = ''                   # file path or 'template:<key>'
    template: str = ''
    import_units: str = 'm'
    import_up: str = '+Z'
    import_forward: str = '+X'
    occluder_target: int = 8000
    wheel_centres: dict = field(default_factory=dict)   # 'FL' -> [x, y, z]
    spec: dict = field(default_factory=dict)            # datasheet dims
    landmarks: dict = field(default_factory=dict)
    mesh: Mesh | None = None
    imported_tris: int = 0

    def meta(self):
        return {'name': self.name, 'spec_name': self.spec_name, 'source': self.source, 'template': self.template,
                'import_units': self.import_units, 'import_up': self.import_up, 'import_forward': self.import_forward,
                'occluder_target': self.occluder_target, 'wheel_centres': self.wheel_centres, 'spec': self.spec,
                'landmarks': {k: list(v) for k, v in self.landmarks.items()}, 'imported_tris': self.imported_tris,
                'parts': [p.to_dict() for p in self.mesh.parts] if self.mesh else [],
                'zones': [z.to_dict() for z in self.mesh.zones] if self.mesh else [],
                # undo state only (project.json leaves it out; mesh.npz stores it): editing zones replaces this array
                # on the shared mesh, so the state keeps the one that matches its zone list
                'face_zone': self.mesh.face_zone if self.mesh else None}

    # measured dimensions from the mesh ------------------------------------------------
    def measured(self) -> dict:
        if self.mesh is None or self.mesh.n_tris == 0:
            return {}
        m = self.mesh
        body = np.array([p.name not in ('Mirrors',) and p.role != 'ignored' for p in m.parts], bool)[m.face_part]
        v = m.tri[body].reshape(-1, 3) if body.any() else m.tri.reshape(-1, 3)
        lo, hi = v.min(axis=0), v.max(axis=0)
        out = {'Length': float(hi[0] - lo[0]), 'Width': float(hi[1] - lo[1]), 'Height': float(hi[2] - lo[2])}
        wc = self.wheel_centres
        if all(k in wc for k in ('FL', 'RL')):
            out['Wheelbase'] = float(abs(wc['FL'][0] - wc['RL'][0]))
        nonwheel = np.array([p.role not in ('wheels', 'ignored') for p in m.parts], bool)[m.face_part]
        if nonwheel.any() and 'Wheelbase' in out:
            c = m.centroids[nonwheel]
            xr, xf = wc['RL'][0], wc['FL'][0]
            mid = (c[:, 0] > xr + 0.45) & (c[:, 0] < xf - 0.45)
            if mid.any():
                out['Ground clearance'] = float(m.tri[nonwheel][mid][:, :, 2].min())
        out['front_x'] = float(hi[0])
        out['rear_x'] = float(lo[0])
        return out


@dataclass
class LogEntry:
    t: float
    kind: str
    text: str


class Project:
    def __init__(self):
        self.name = 'untitled'
        self.path = ''
        self.vehicle = Vehicle()
        self.modules: dict[str, Module] = {m.id: m for m in builtin_modules()}
        self.lenses: dict[str, Lens] = {l.id: l for l in builtin_lenses()}
        self.sensor_models: dict[str, SensorModel] = {m.id: m for m in builtin_sensor_models()}
        self.variants: list[Variant] = []
        self.active_uid = ''
        self.roi = ROI()
        self.analysis = AnalysisSettings()
        self.targets: list[Target] = []
        self.target_settings = TargetSettings()
        self.obstacles: list[Obstacle] = []
        self.rig: list[RigPart] = []           # the data-collection rig mounted on the vehicle
        self.ecu = ECU()
        self.photos = PhotoSet()
        self.streams: dict[str, str] = {}               # camera tag -> stream source (URL, device, file, 'synthetic')
        self.fleet: dict = {}                           # vehicles built to this design and their calibrations (core.fleet)
        self.frame = 'rear_axle'
        self.custom_origin = [0.0, 0.0, 0.0]
        self.units = 'm'
        self.log: list[LogEntry] = []
        self.saved_at = 0.0
        self.created = time.time()
        self.thumb_png: bytes = b''

    # -------------------------------------------------------------- access
    def snapshot(self) -> 'Project':
        """Copy safe to hand to a worker thread (the mesh is shared; it is never mutated in place)."""
        q = copy.copy(self)
        q.vehicle = copy.copy(self.vehicle)             # its own fields (the mesh is still shared)
        for k in ('wheel_centres', 'spec', 'landmarks'):
            setattr(q.vehicle, k, copy.deepcopy(getattr(self.vehicle, k)))
        q.variants = [Variant.from_dict(v.to_dict()) for v in self.variants]
        for k in ('roi', 'analysis', 'targets', 'target_settings', 'obstacles', 'rig', 'ecu', 'lenses', 'modules',
                  'sensor_models', 'custom_origin', 'streams', 'fleet'):
            setattr(q, k, copy.deepcopy(getattr(self, k)))
        q.photos = self.photos
        q.log = []
        return q

    @property
    def variant(self) -> Variant:
        for v in self.variants:
            if v.uid == self.active_uid:
                return v
        if self.variants:
            self.active_uid = self.variants[0].uid
            return self.variants[0]
        v = Variant(name='Variant 1')
        self.variants.append(v)
        self.active_uid = v.uid
        return v

    def variant_by_uid(self, uid):
        for v in self.variants:
            if v.uid == uid:
                return v
        return None

    def module_for(self, cam: Camera) -> Module:
        return self.modules.get(cam.module_id) or next(iter(self.modules.values()))

    def lens_for(self, cam: Camera) -> Lens:
        """The lens model in use: the camera's own unit calibration when it has one, else the library lens."""
        base = self.nominal_lens_for(cam)
        if cam.calib:
            from .calib import unit_lens
            return unit_lens(base, self.module_for(cam), cam.calib)
        return base

    def nominal_lens_for(self, cam: Camera) -> Lens:
        return self.lenses.get(cam.lens_id) or next(iter(self.lenses.values()))

    def sensor_model_for(self, s: Sensor) -> SensorModel:
        """The sensor's library model; a missing one falls back to the first of the same kind."""
        kind = s.kind if s.kind in SENSOR_KINDS else 'lidar'
        m = self.sensor_models.get(s.model_id)
        if m is not None and m.kind == kind:
            return m
        return next((m for m in self.sensor_models.values() if m.kind == kind), None) or \
            next(m for m in builtin_sensor_models() if m.kind == kind)

    def add_log(self, kind, text):
        self.log.append(LogEntry(time.time(), kind, text))
        if len(self.log) > 2000:
            del self.log[:500]

    # -------------------------------------------------------------- frames
    def origin_offset(self) -> np.ndarray:
        """World (rear-axle) coordinates of the display origin."""
        meas = self.vehicle.measured() if self.vehicle.mesh is not None else {}
        wc = self.vehicle.wheel_centres
        if self.frame == 'front_axle':
            return np.array([wc['FL'][0] if 'FL' in wc else 0.0, 0, 0])
        if self.frame == 'front_bumper':
            return np.array([meas.get('front_x', 0.0), 0, 0])
        if self.frame == 'centre':
            return np.array([(meas.get('front_x', 0.0) + meas.get('rear_x', 0.0)) / 2, 0, 0])
        if self.frame == 'custom':
            return np.array(self.custom_origin, float)
        return np.zeros(3)

    def to_display(self, p):
        return np.asarray(p, float) - self.origin_offset()

    def from_display(self, p):
        return np.asarray(p, float) + self.origin_offset()

    @property
    def frame_label(self):
        return FRAME_PRESETS.get(self.frame, self.frame)

    # -------------------------------------------------------------- sections for undo
    def section(self, key: str):
        if key.startswith('variant:'):
            v = self.variant_by_uid(key[8:])
            return v.to_dict() if v else None
        if key == 'variants':
            return {'variants': [v.to_dict() for v in self.variants], 'active': self.active_uid}
        if key == 'roi':
            return self.roi.to_dict()
        if key == 'analysis':
            return self.analysis.to_dict()
        if key == 'targets':
            return {'targets': [t.to_dict() for t in self.targets], 'settings': self.target_settings.to_dict()}
        if key == 'obstacles':
            return [o.to_dict() for o in self.obstacles]
        if key == 'rig':
            return [r.to_dict() for r in self.rig]
        if key == 'ecu':
            return self.ecu.to_dict()
        if key == 'library':
            return {'modules': {k: m.to_dict() for k, m in self.modules.items()},
                    'lenses': {k: l.to_dict() for k, l in self.lenses.items()},
                    'sensors': {k: m.to_dict() for k, m in self.sensor_models.items()}}
        if key == 'frame':
            return {'frame': self.frame, 'custom': list(self.custom_origin), 'units': self.units}
        if key == 'name':
            return self.name
        if key == 'photos':
            return self.photos.to_dict()
        if key == 'streams':
            return dict(self.streams)
        if key == 'fleet':
            return copy.deepcopy(self.fleet)
        if key == 'vehicle':
            # mesh is shared by reference (immutable once built); meta is deep-copied
            return {'meta': copy.deepcopy(self.vehicle.meta()), 'vehicle': self.vehicle}
        raise KeyError(key)

    def restore(self, key: str, state):
        if key.startswith('variant:'):
            uid = key[8:]
            nv = Variant.from_dict(state)
            for i, v in enumerate(self.variants):
                if v.uid == uid:
                    self.variants[i] = nv
                    return
            self.variants.append(nv)
        elif key == 'variants':
            self.variants = [Variant.from_dict(v) for v in state['variants']]
            self.active_uid = state['active']
        elif key == 'roi':
            self.roi = ROI.from_dict(state)
        elif key == 'analysis':
            self.analysis = AnalysisSettings.from_dict(state)
        elif key == 'targets':
            self.targets = [Target.from_dict(t) for t in state['targets']]
            self.target_settings = TargetSettings.from_dict(state['settings'])
        elif key == 'obstacles':
            self.obstacles = [Obstacle.from_dict(o) for o in state]
        elif key == 'rig':
            self.rig = [RigPart.from_dict(r) for r in state]
        elif key == 'ecu':
            self.ecu = ECU.from_dict(state)
        elif key == 'library':
            self.modules = {k: Module.from_dict(m) for k, m in state['modules'].items()}
            self.lenses = {k: Lens.from_dict(l) for k, l in state['lenses'].items()}
            if 'sensors' in state:
                self.sensor_models = {k: SensorModel.from_dict(m) for k, m in state['sensors'].items()}
        elif key == 'frame':
            self.frame, self.custom_origin, self.units = state['frame'], list(state['custom']), state['units']
        elif key == 'name':
            self.name = state
        elif key == 'photos':
            self.photos = PhotoSet.from_dict(state)
        elif key == 'streams':
            self.streams = dict(state)
        elif key == 'fleet':
            self.fleet = copy.deepcopy(state)
        elif key == 'vehicle':
            veh = state['vehicle']
            meta = state['meta']
            for k in ('name', 'spec_name', 'source', 'template', 'import_units', 'import_up', 'import_forward',
                      'occluder_target', 'wheel_centres', 'spec', 'imported_tris'):
                setattr(veh, k, copy.deepcopy(meta[k]))
            veh.landmarks = {k: tuple(v) for k, v in meta['landmarks'].items()}
            if veh.mesh is not None and meta.get('zones') is not None:
                veh.mesh.zones = [Zone.from_dict(z) for z in meta['zones']]
                veh.mesh.parts = [Part.from_dict(p) for p in meta['parts']] or veh.mesh.parts
            fz = meta.get('face_zone')
            if veh.mesh is not None and fz is not None and len(fz) == veh.mesh.n_tris:
                veh.mesh.face_zone = np.array(fz, np.int16)
            self.vehicle = veh
        else:
            raise KeyError(key)

    # -------------------------------------------------------------- persistence
    def to_json(self) -> dict:
        return {
            'format': 'vantage', 'version': FORMAT_VERSION, 'name': self.name, 'created': self.created,
            'saved_at': time.time(), 'vehicle': {k: v for k, v in self.vehicle.meta().items() if k != 'face_zone'},
            'modules': {k: m.to_dict() for k, m in self.modules.items()},
            'lenses': {k: l.to_dict() for k, l in self.lenses.items()},
            'sensor_models': {k: m.to_dict() for k, m in self.sensor_models.items()},
            'variants': [v.to_dict() for v in self.variants], 'active': self.active_uid,
            'roi': self.roi.to_dict(), 'analysis': self.analysis.to_dict(),
            'targets': [t.to_dict() for t in self.targets], 'target_settings': self.target_settings.to_dict(),
            'obstacles': [o.to_dict() for o in self.obstacles], 'rig': [r.to_dict() for r in self.rig],
            'ecu': self.ecu.to_dict(),
            'photos': self.photos.to_dict(), 'frame': self.frame, 'custom_origin': self.custom_origin,
            'units': self.units,
            # passwords in stream URLs go to the user's private store, never into the file (core/credentials.py)
            'streams': {k: _credentials.remember(v) for k, v in self.streams.items()},
            'fleet': self.fleet,
            'log': [[e.t, e.kind, e.text] for e in self.log[-400:]],
        }

    def save(self, path: str, thumb_png: bytes | None = None):
        thumb_png = thumb_png if thumb_png is not None else self.thumb_png     # re-saving keeps the stored thumbnail
        data = self.to_json()
        buf = io.BytesIO()
        m = self.vehicle.mesh
        if m is not None:
            np.savez_compressed(buf, vertices=m.vertices.astype(np.float32), faces=m.faces.astype(np.int32),
                                face_part=m.face_part, face_zone=m.face_zone)
        tmp = path + '.tmp'
        with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as z:
            z.writestr('project.json', json.dumps(data, indent=1))
            if m is not None:
                z.writestr('mesh.npz', buf.getvalue())
            if thumb_png:
                z.writestr('thumb.png', thumb_png)
                self.thumb_png = thumb_png
        import os
        os.replace(tmp, path)
        self.path = path
        self.saved_at = data['saved_at']

    @staticmethod
    def load(path: str) -> 'Project':
        import os
        if os.path.getsize(path) > 2_000_000_000:
            raise ValueError('File is larger than 2 GB.')
        with zipfile.ZipFile(path) as z:
            names = set(z.namelist())
            if 'project.json' not in names:
                raise ValueError('Not a Vantage project (project.json missing).')
            data = json.loads(z.read('project.json').decode('utf-8'))
            mesh_bytes = z.read('mesh.npz') if 'mesh.npz' in names else None
            thumb = z.read('thumb.png') if 'thumb.png' in names else b''
        if data.get('format') != 'vantage':
            raise ValueError('Not a Vantage project.')
        p = Project()
        p.path = path
        p.name = data.get('name', 'untitled')
        p.created = data.get('created', time.time())
        p.saved_at = data.get('saved_at', 0)
        vm = data.get('vehicle', {})
        veh = Vehicle()
        for k in ('name', 'spec_name', 'source', 'template', 'import_units', 'import_up', 'import_forward',
                  'occluder_target', 'wheel_centres', 'spec', 'imported_tris'):
            if k in vm:
                setattr(veh, k, vm[k])
        veh.landmarks = {k: tuple(v) for k, v in vm.get('landmarks', {}).items()}
        if mesh_bytes:
            with np.load(io.BytesIO(mesh_bytes), allow_pickle=False) as npz:
                parts = [Part.from_dict(d) for d in vm.get('parts', [])] or [Part('Body')]
                zones = [Zone.from_dict(d) for d in vm.get('zones', [])]
                fp = npz['face_part']
                fp = np.clip(fp, 0, len(parts) - 1)
                veh.mesh = Mesh(npz['vertices'].astype(np.float64), npz['faces'], fp, parts, npz['face_zone'], zones)
        p.vehicle = veh
        p.modules = {k: Module.from_dict(m) for k, m in data.get('modules', {}).items()} or p.modules
        p.lenses = {k: Lens.from_dict(l) for k, l in data.get('lenses', {}).items()} or p.lenses
        # the built-in sensors stay available in older projects; a saved model (edited or custom) wins
        p.sensor_models.update({k: SensorModel.from_dict(m) for k, m in (data.get('sensor_models') or {}).items()})
        p.variants = [Variant.from_dict(v) for v in data.get('variants', [])]
        p.active_uid = data.get('active', p.variants[0].uid if p.variants else '')
        p.roi = ROI.from_dict(data.get('roi', {}))
        p.analysis = AnalysisSettings.from_dict(data.get('analysis', {}))
        p.targets = [Target.from_dict(t) for t in data.get('targets', [])]
        p.target_settings = TargetSettings.from_dict(data.get('target_settings', {}))
        p.obstacles = [Obstacle.from_dict(o) for o in data.get('obstacles', [])]
        p.rig = [RigPart.from_dict(r) for r in data.get('rig', [])]
        p.ecu = ECU.from_dict(data.get('ecu', {}))
        p.photos = PhotoSet.from_dict(data.get('photos', {}))
        p.streams = {str(k): str(v) for k, v in (data.get('streams') or {}).items() if str(v).strip()}
        p.fleet = data.get('fleet') if isinstance(data.get('fleet'), dict) else {}
        p.frame = data.get('frame', 'rear_axle')
        p.custom_origin = data.get('custom_origin', [0, 0, 0])
        p.units = data.get('units', 'm')
        p.log = [LogEntry(*e) for e in data.get('log', [])]
        p.thumb_png = thumb
        return p


def read_thumbnail(path: str) -> tuple[bytes, dict]:
    """Cheap peek for the start window: (thumb png bytes, summary dict)."""
    try:
        with zipfile.ZipFile(path) as z:
            names = set(z.namelist())
            data = json.loads(z.read('project.json').decode('utf-8'))
            thumb = z.read('thumb.png') if 'thumb.png' in names else b''
        var = next((v for v in data.get('variants', []) if v.get('uid') == data.get('active')), None)
        return thumb, {'vehicle': data.get('vehicle', {}).get('name', ''),
                       'cameras': len(var['cameras']) if var else 0,
                       'variants': len(data.get('variants', [])), 'template': data.get('vehicle', {}).get('template', '')}
    except Exception:
        return b'', {}
