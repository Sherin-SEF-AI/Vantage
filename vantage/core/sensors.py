"""Lidars and radars of the rig: the sensor library and their geometry.

A sensor (core.project.Sensor) sits in the vehicle frame like a camera: x forward, y left, z up, origin at the
rear-axle centre on the ground; its orientation is intrinsic Z-Y-X yaw/pitch/roll in degrees with pitch positive
*down*, and its own frame has x along its forward axis, y left, z up (ROS REP 103). The position is the sensor's
optical centre (a spinning lidar's beam origin), not its base.

Lidar beams: a channel fires at a fixed elevation (degrees above the sensor's xy plane); a spinning lidar sweeps all
channels through 360° of azimuth (counter-clockwise from the sensor's +x, the ISO convention), a fixed-field lidar
(scan 'solid') through its horizontal field centred on +x. A non-repetitive scanner (scan 'nonrepetitive', e.g. a
Livox) is modelled as an even grid with the datasheet's points per frame over its field.

Radar: one or more zones (a long-range beam and a wide near beam, say), each a range on a car-sized target and the
half-angles of its field in azimuth and elevation. Ranges for smaller targets scale with the fourth root of the radar
cross-section (the radar equation).

The library figures are nominal values from public datasheets and product pages (radars: typical classes, not a
product); every model says where its figures come from in `note`. Check them against your unit's datasheet.
"""
from __future__ import annotations

import csv
import io
import json
import math
from dataclasses import asdict, dataclass, field

import numpy as np

from .mathutil import rot_zyx

KINDS = ('lidar', 'radar')
SCANS = ('spinning', 'solid', 'nonrepetitive')
SCAN_LABEL = {'spinning': 'Spinning', 'solid': 'Fixed field', 'nonrepetitive': 'Non-repetitive', 'fixed': 'Fixed field'}
REF_RCS = 10.0                       # m², the car-sized target a radar datasheet range is given for
MIN_FOOT_M = 0.002                   # a sensor's footprint never shrinks below 2 mm in the plan view


@dataclass
class SensorModel:
    """One lidar or radar product (or a typical class of radar). Angles in degrees, ranges in metres."""
    id: str
    name: str = 'Lidar'
    kind: str = 'lidar'              # lidar | radar
    vendor: str = ''
    scan: str = 'spinning'           # lidar: spinning | solid | nonrepetitive; radar: fixed
    channels: int = 32               # lidar beams (rings); a non-repetitive scanner: rows of its equivalent grid
    v_min: float = -16.0             # elevation of the lowest beam
    v_max: float = 15.0              # elevation of the highest beam
    beams: list = field(default_factory=list)    # measured beam elevations (overrides the even v_min..v_max spread)
    h_fov: float = 360.0             # horizontal field, centred on the sensor's +x
    h_res: float = 0.2               # azimuth step between a channel's points at rate_hz
    rate_hz: float = 10.0            # frames (rotations) per second
    range_m: float = 100.0           # lidar: range on a dark (10 %) target; radar: on a car (REF_RCS)
    range_max_m: float = 0.0         # rated maximum range on bright targets (0 = not given)
    range_min_m: float = 0.5         # nearest measured range
    points_per_s: float = 0.0        # datasheet single-return point rate (0 = from channels, h_fov, h_res, rate)
    bytes_per_point: float = 12.0    # recorded bytes per point as the driver delivers them (packet payload)
    rotation: str = 'cw'             # spinning lidar seen from above: cw | ccw
    zones: list = field(default_factory=list)    # radar: [[range_m, az_half_deg, el_half_deg], ...]
    data_mbps: float = 0.0           # radar: output data rate, Mbit/s
    interface: str = '1000BASE-T1'
    time_sync: str = 'PTP'           # PTP (IEEE 1588 / gPTP) | PPS + NMEA | none
    size_mm: tuple = (87.0, 87.0, 74.0)    # spinning: diameter, diameter, height; others: depth (x), width (y), height
    optical_z_mm: float = 37.0       # optical centre above the base
    power_w: float = 0.0
    note: str = ''                   # where the figures come from
    builtin: bool = False

    def to_dict(self):
        d = asdict(self)
        d['size_mm'] = list(self.size_mm)
        d['beams'] = [float(b) for b in self.beams]
        d['zones'] = [list(map(float, z)) for z in self.zones]
        return d

    @staticmethod
    def from_dict(d):
        d = dict(d)
        if 'size_mm' in d:
            d['size_mm'] = tuple(float(v) for v in d['size_mm'])
        known = SensorModel.__dataclass_fields__
        m = SensorModel(**{k: v for k, v in d.items() if k in known})
        for k, f in known.items():                  # a hand-edited file may hold "12" for 12
            if f.type in ('float', 'int') and not isinstance(getattr(m, k), (int, float)):
                try:
                    v = float(getattr(m, k))
                except (TypeError, ValueError):
                    v = float('nan')
                setattr(m, k, (int(v) if f.type == 'int' else v) if math.isfinite(v) else getattr(SensorModel(m.id), k))
        m.kind = m.kind if m.kind in KINDS else 'lidar'
        m.rotation = m.rotation if m.rotation in ('cw', 'ccw') else 'cw'
        for k in ('rate_hz', 'h_res', 'h_fov', 'range_m'):           # these divide or bound everything downstream
            if getattr(m, k) <= 0:
                setattr(m, k, getattr(SensorModel(m.id), k))
        m.h_fov = min(m.h_fov, 360.0)
        m.beams = [float(b) for b in (m.beams or []) if _finite(b)]
        m.zones = [[float(z[0]), float(z[1]), float(z[2])] for z in (m.zones or [])
                   if len(z) >= 3 and all(_finite(v) for v in z[:3]) and z[0] > 0]
        return m

    # ------------------------------------------------------------ derived figures
    @property
    def is_lidar(self):
        return self.kind == 'lidar'

    @property
    def spinning(self):
        return self.is_lidar and self.scan == 'spinning'

    def elevations(self) -> np.ndarray:
        """Beam elevations, degrees, lowest first."""
        if self.beams:
            return np.sort(np.asarray(self.beams, float))
        n = max(int(self.channels), 1)
        if n == 1:
            return np.array([(self.v_min + self.v_max) / 2])
        return np.linspace(self.v_min, self.v_max, n)

    def n_channels(self) -> int:
        return len(self.beams) if self.beams else max(int(self.channels), 1)

    def columns(self) -> int:
        """Points per channel per frame."""
        return max(int(round(self.h_fov / max(self.h_res, 1e-3))), 1)

    def pps(self) -> float:
        """Points per second (single return)."""
        if not self.is_lidar:
            return 0.0
        if self.points_per_s > 0:
            return float(self.points_per_s)
        return float(self.n_channels() * self.columns() * max(self.rate_hz, 0.0))

    def data_bps(self) -> float:
        """Recorded data rate, bit/s: lidar points × bytes per point; a radar's stated output."""
        if self.is_lidar:
            return self.pps() * max(self.bytes_per_point, 0.0) * 8.0
        return max(self.data_mbps, 0.0) * 1e6

    def v_fov(self) -> tuple[float, float]:
        """(lowest, highest) elevation of the field."""
        if self.is_lidar:
            e = self.elevations()
            return float(e[0]), float(e[-1])
        el = max((z[2] for z in self.zones), default=0.0)
        return -el, el

    def az_half(self) -> float:
        """Half the horizontal field, degrees (180 for a spinning lidar)."""
        if self.is_lidar:
            return 180.0 if self.h_fov >= 359.999 else self.h_fov / 2
        return max((z[1] for z in self.zones), default=0.0)

    def max_range(self) -> float:
        if self.is_lidar:
            return float(self.range_m)
        return max((z[0] for z in self.zones), default=0.0)

    def v_step(self) -> float:
        """Mean spacing between neighbouring beams, degrees."""
        e = self.elevations()
        return float(np.mean(np.diff(e))) if len(e) > 1 else 0.0

    def summary(self) -> str:
        if self.is_lidar:
            lo, hi = self.v_fov()
            return (f'{self.n_channels()} ch · {self.h_fov:.0f}° × {hi - lo:.1f}° · {self.range_m:.0f} m @10% · '
                    f'{self.pps() / 1e6:.2f} Mpts/s')
        z = ', '.join(f'{r:.0f} m ±{a:.0f}°' for r, a, _ in self.zones)
        return f'{z} · {self.data_mbps:g} Mbit/s'


def _finite(v) -> bool:
    try:
        return math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


# ======================================================================= library
_DATASHEET = 'Nominal figures from the public datasheet; check them against your unit\'s.'


def builtin_sensor_models() -> list[SensorModel]:
    return [
        SensorModel('ouster_os1_128', 'OS1-128 (Rev 7)', 'lidar', 'Ouster', 'spinning', 128, -22.5, 22.5,
                    h_fov=360.0, h_res=360.0 / 1024, rate_hz=10.0, range_m=90.0, range_min_m=0.5,
                    points_per_s=1_310_720, bytes_per_point=12.0, rotation='cw', interface='1000BASE-T',
                    time_sync='PTP', size_mm=(87.0, 87.0, 74.0), optical_z_mm=36.18, power_w=20.0,
                    note=f'{_DATASHEET} 1024 × 10 Hz mode: 90 m on a 10 % target in 100 klx sunlight; '
                         '2048 columns or 20 Hz double the points.', builtin=True),
        SensorModel('hesai_ot128', 'OT128', 'lidar', 'Hesai', 'spinning', 128, -25.0, 15.0,
                    h_fov=360.0, h_res=360.0 / 2700, rate_hz=10.0, range_m=200.0, range_max_m=230.0, range_min_m=0.3,
                    points_per_s=3_456_000, bytes_per_point=4.5, rotation='cw', interface='1000BASE-T1',
                    time_sync='PTP', size_mm=(110.0, 110.0, 118.0), optical_z_mm=60.0, power_w=18.0,
                    note=f'{_DATASHEET} The real channels are denser near the horizon (0.125° there); modelled as an '
                         'even spread unless beam angles are imported.', builtin=True),
        SensorModel('hesai_at128', 'AT128', 'lidar', 'Hesai', 'solid', 128, -12.5, 12.9,
                    h_fov=120.0, h_res=0.1, rate_hz=10.0, range_m=200.0, range_min_m=0.5,
                    points_per_s=1_536_000, bytes_per_point=4.5, interface='1000BASE-T1', time_sync='PTP',
                    size_mm=(110.0, 137.0, 48.0), optical_z_mm=24.0, power_w=18.0,
                    note=f'{_DATASHEET} Hybrid solid-state: 120° × 25.4° field, 0.1° × 0.2°.', builtin=True),
        SensorModel('hesai_xt32', 'XT32', 'lidar', 'Hesai', 'spinning', 32, -16.0, 15.0,
                    h_fov=360.0, h_res=0.18, rate_hz=10.0, range_m=80.0, range_max_m=120.0, range_min_m=0.05,
                    points_per_s=640_000, bytes_per_point=4.5, rotation='cw', interface='1000BASE-T',
                    time_sync='PTP', size_mm=(100.0, 100.0, 76.0), optical_z_mm=38.0, power_w=10.0,
                    note=f'{_DATASHEET} 80 m on a 10 % target, 120 m rated; 1° between channels.', builtin=True),
        SensorModel('velodyne_vlp16', 'VLP-16 (Puck)', 'lidar', 'Velodyne', 'spinning', 16, -15.0, 15.0,
                    h_fov=360.0, h_res=0.2, rate_hz=10.0, range_m=100.0, range_min_m=0.5,
                    points_per_s=300_000, bytes_per_point=3.25, rotation='cw', interface='100BASE-TX',
                    time_sync='PPS + NMEA', size_mm=(103.3, 103.3, 71.7), optical_z_mm=37.7, power_w=8.0,
                    note=f'{_DATASHEET} Up to 100 m (no reflectivity given); 2° between channels; 1206-byte packets '
                         'of 384 points.', builtin=True),
        SensorModel('livox_mid360', 'Mid-360', 'lidar', 'Livox', 'nonrepetitive', 40, -7.0, 52.0,
                    h_fov=360.0, h_res=0.72, rate_hz=10.0, range_m=40.0, range_max_m=70.0, range_min_m=0.1,
                    points_per_s=200_000, bytes_per_point=14.0, interface='100BASE-TX', time_sync='PTP',
                    size_mm=(65.0, 65.0, 60.0), optical_z_mm=47.0, power_w=6.5,
                    note=f'{_DATASHEET} Non-repetitive scan, modelled as an even grid of 20 000 points per 10 Hz '
                         'frame; 70 m on an 80 % target; shorter range near the top of the field.', builtin=True),
        SensorModel('radar_long', 'Long-range radar · typical', 'radar', '', 'fixed',
                    zones=[[250.0, 9.0, 7.0], [70.0, 45.0, 7.0]], range_min_m=0.2, rate_hz=15.0, data_mbps=1.0,
                    interface='CAN-FD / 100BASE-T1', time_sync='PTP', size_mm=(37.0, 137.0, 91.0), optical_z_mm=45.0,
                    power_w=6.6, note='A typical 77 GHz long-range radar (far beam ±9° to 250 m, near beam ±45° to '
                                      '70 m on a car), not a product: enter your radar\'s datasheet figures.',
                    builtin=True),
        SensorModel('radar_imaging', '4D imaging radar · typical', 'radar', '', 'fixed',
                    zones=[[300.0, 50.0, 15.0]], range_min_m=0.2, rate_hz=20.0, data_mbps=30.0,
                    interface='1000BASE-T1', time_sync='PTP', size_mm=(40.0, 140.0, 100.0), optical_z_mm=50.0,
                    power_w=23.0, note='A typical 77 GHz 4D imaging radar (±50° × ±15°, 300 m on a car), not a '
                                       'product: enter your radar\'s datasheet figures.', builtin=True),
        SensorModel('radar_corner', 'Corner radar · typical', 'radar', '', 'fixed',
                    zones=[[100.0, 75.0, 10.0]], range_min_m=0.2, rate_hz=20.0, data_mbps=2.0,
                    interface='CAN-FD / 100BASE-T1', time_sync='PTP', size_mm=(30.0, 80.0, 70.0), optical_z_mm=35.0,
                    power_w=4.0, note='A typical 77 GHz corner radar (±75°, 100 m on a car), not a product: enter '
                                      'your radar\'s datasheet figures.', builtin=True),
    ]


def default_model_id(kind: str) -> str:
    return 'ouster_os1_128' if kind == 'lidar' else 'radar_long'


# ======================================================================= frames and beams
def rotation(s) -> np.ndarray:
    """Sensor -> vehicle rotation."""
    return rot_zyx(s.yaw, s.pitch, s.roll)


def directions(elev_deg, az_deg) -> np.ndarray:
    """Unit vectors in the sensor frame for every (elevation, azimuth) pair: shape (E, A, 3)."""
    e = np.radians(np.asarray(elev_deg, float))[:, None]
    a = np.radians(np.asarray(az_deg, float))[None, :]
    return np.stack([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e) * np.ones_like(a)], axis=-1)


def scan_azimuths(model: SensorModel, step: float | None = None) -> np.ndarray:
    """Azimuths of the scan (sensor frame, degrees, counter-clockwise from +x)."""
    step = float(step or model.h_res)
    half = model.az_half()
    if half >= 180.0:
        return np.arange(-180.0, 180.0, step)
    n = max(int(round(2 * half / step)), 1)
    return -half + (np.arange(n) + 0.5) * (2 * half / n)


def beam_dirs(s, model: SensorModel, step: float | None = None) -> np.ndarray:
    """World (vehicle-frame) unit vectors of the lidar's beams, (channels, azimuths, 3)."""
    return directions(model.elevations(), scan_azimuths(model, step)) @ rotation(s).T


def to_sensor(s, P) -> np.ndarray:
    """Vehicle-frame points (K, 3) in the sensor frame."""
    return (np.asarray(P, float) - np.array([s.x, s.y, s.z])) @ rotation(s)


def angles(s, P):
    """(range, azimuth deg, elevation deg) of vehicle-frame points (K, 3) seen from the sensor."""
    q = to_sensor(s, P)
    r = np.linalg.norm(q, axis=1)
    az = np.degrees(np.arctan2(q[:, 1], q[:, 0]))
    el = np.degrees(np.arcsin(np.clip(q[:, 2] / np.maximum(r, 1e-12), -1, 1)))
    return r, az, el


def in_field(s, model: SensorModel, P, rcs: float | None = None) -> np.ndarray:
    """Whether each vehicle-frame point (K, 3) lies in the sensor's field and range (no occlusion test). A radar takes
    the target's cross-section (m², default the car the datasheet range is given for)."""
    r, az, el = angles(s, P)
    if model.is_lidar:
        lo, hi = model.v_fov()
        half = model.az_half()
        ok = (el >= lo - 1e-9) & (el <= hi + 1e-9) & (r >= model.range_min_m) & (r <= model.range_m)
        return ok if half >= 180.0 else ok & (np.abs(az) <= half + 1e-9)
    scale = radar_scale(rcs)
    ok = np.zeros(len(r), bool)
    for rng, ah, eh in model.zones:
        ok |= (np.abs(az) <= ah) & (np.abs(el) <= eh) & (r <= rng * scale) & (r >= model.range_min_m)
    return ok


def radar_scale(rcs: float | None) -> float:
    """Range factor for a target of radar cross-section rcs (m²) against the datasheet's car: (rcs / 10)^¼."""
    if rcs is None:
        return 1.0
    return float(max(rcs, 0.0) / REF_RCS) ** 0.25


def ground_rings(s, model: SensorModel, az_step: float = 1.0, far: float | None = None) -> list:
    """Where each lidar channel meets the ground (z = 0), as polylines in the vehicle frame: [(channel, (N, 3))], one
    polyline per unbroken run of azimuths. Beams that point up or reach the ground beyond the range are left out."""
    far = model.range_m if far is None else far
    D = beam_dirs(s, model, az_step)
    pos = np.array([s.x, s.y, s.z])
    out = []
    closed = model.az_half() >= 180.0
    for c in range(D.shape[0]):
        d = D[c]
        with np.errstate(divide='ignore', invalid='ignore'):
            t = np.where(d[:, 2] < -1e-6, -pos[2] / d[:, 2], np.inf)
        ok = np.isfinite(t) & (t <= far) & (t >= model.range_min_m)
        if not ok.any():
            continue
        P = pos + d * np.where(ok, t, 0.0)[:, None]
        if closed and ok.all():
            out.append((c, np.vstack([P, P[:1]])))
            continue
        run = []
        for i in range(len(ok)):
            if ok[i]:
                run.append(P[i])
            elif len(run) > 1:
                out.append((c, np.array(run)))
                run = []
            else:
                run = []
        if len(run) > 1:
            out.append((c, np.array(run)))
    return out


def radar_wedges(s, model: SensorModel, rcs: float | None = None, samples: int = 24) -> list:
    """Each radar zone's field in plan as a closed polygon (N, 2) in the vehicle frame: apex at the sensor, an arc at
    the zone's range (for the target's cross-section) about the sensor's heading."""
    out = []
    k = radar_scale(rcs)
    R = rotation(s)
    fwd = R[:, 0]
    head = math.degrees(math.atan2(fwd[1], fwd[0]))
    for rng, ah, _ in model.zones:
        a = np.radians(head + np.linspace(-ah, ah, samples))
        r = rng * k
        arc = np.c_[s.x + r * np.cos(a), s.y + r * np.sin(a)]
        out.append(np.vstack([[s.x, s.y], arc]))
    return out


def heading(s) -> float:
    """Direction of the sensor's +x in plan, degrees counter-clockwise from the vehicle's +x."""
    f = rotation(s)[:, 0]
    return math.degrees(math.atan2(f[1], f[0]))


# ======================================================================= body
def body_box(s, model: SensorModel):
    """(centre (3,), half sizes (3,), rotation (3, 3), round) of the sensor housing: a cylinder for a spinning lidar
    (round = True, half sizes radius, radius, half height), a box otherwise; the optical centre sits optical_z_mm above
    the base."""
    dx, dy, h = (max(float(v), 1.0) / 1000.0 for v in model.size_mm)
    R = rotation(s)
    base_to_centre = h / 2 - model.optical_z_mm / 1000.0
    c = np.array([s.x, s.y, s.z]) + R @ np.array([0.0, 0.0, base_to_centre])
    return c, np.array([dx / 2, dy / 2, h / 2]), R, model.spinning or model.scan == 'nonrepetitive'


def footprint(s, model: SensorModel) -> np.ndarray:
    """The housing's outline seen from above (closed polygon (n, 2))."""
    c, half, R, rnd = body_box(s, model)
    if rnd:
        t = np.linspace(0, 2 * math.pi, 24, endpoint=False)
        r = max(half[0], MIN_FOOT_M)
        return np.c_[c[0] + r * np.cos(t), c[1] + r * np.sin(t)]
    q = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], float) * np.maximum(half[:2], MIN_FOOT_M)
    a = math.radians(heading(s))
    rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    return q @ rot.T + c[:2]


# ======================================================================= placement
def next_tag(kind: str, taken) -> str:
    """'L1', 'L2' … for lidars, 'RD1' … for radars, skipping every tag already used by a camera or sensor."""
    taken = {t.upper() for t in taken}
    stem = 'L' if kind == 'lidar' else 'RD'
    i = 1
    while f'{stem}{i}' in taken:
        i += 1
    return f'{stem}{i}'


def default_pose(kind: str, model: SensorModel, mesh=None, rig=()) -> dict:
    """A sensible first pose: a lidar on top of the highest mountable rig part (a mast head, an enclosure) or on the
    roof; a radar in the middle of the front bumper, facing forward at 0.5 m."""
    from . import rig as rigm
    oz = model.optical_z_mm / 1000.0
    if kind == 'lidar':
        tops = [p for p in rig if p.visible and p.mountable and p.kind in ('box', 'cylinder')]
        if tops:
            p = max(tops, key=lambda q: q.z + q.height / 2)
            return {'x': round(p.x, 3), 'y': round(p.y, 3), 'z': round(p.z + p.height / 2 + oz, 3),
                    'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0}
        xf, xr, _, top = rigm.roof(mesh)
        return {'x': round((xf + xr) / 2, 3), 'y': 0.0, 'z': round(top + oz, 3), 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0}
    if mesh is not None and mesh.n_tris:
        lo, hi = mesh.bbox
        c = mesh.centroids
        band = (np.abs(c[:, 2] - 0.5) < 0.15) & (np.abs(c[:, 1]) < 0.3)
        front = float(c[band, 0].max()) if band.any() else float(hi[0])
        return {'x': round(front + model.size_mm[0] / 2000.0, 3), 'y': 0.0, 'z': 0.5, 'yaw': 0.0, 'pitch': 0.0,
                'roll': 0.0}
    return {'x': 3.6, 'y': 0.0, 'z': 0.5, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0}


# ======================================================================= beam angle files
def read_beam_angles(path: str) -> list[float]:
    """Beam elevations (degrees) from a lidar's own calibration: Ouster metadata JSON (beam_altitude_angles), a Hesai
    angle-correction CSV (Channel, Elevation, Azimuth), a Velodyne ROS calibration YAML (vert_correction, radians) or a
    plain list with one angle per line."""
    with open(path, encoding='utf-8-sig', errors='replace') as f:
        text = f.read()
    return parse_beam_angles(text)


def parse_beam_angles(text: str) -> list[float]:
    t = text.strip()
    if not t:
        raise ValueError('The file is empty.')
    if t[0] in '{[':
        try:
            d = json.loads(t)
        except ValueError:
            d = None
        if d is not None:
            found = _find_key(d, 'beam_altitude_angles')
            if found is None and isinstance(d, list):
                found = d
            if found is None:
                raise ValueError('No beam_altitude_angles in this JSON (an Ouster metadata file has them).')
            return _check([float(v) for v in found])
    if 'vert_correction' in t:
        import yaml
        d = yaml.safe_load(t)
        lasers = (d or {}).get('lasers') or []
        vals = sorted(((int(l.get('laser_id', i)), float(l['vert_correction'])) for i, l in enumerate(lasers)
                       if 'vert_correction' in l))
        return _check([math.degrees(v) for _, v in vals])
    rows = list(csv.reader(io.StringIO(t)))
    head = [h.strip().lower() for h in rows[0]] if rows else []
    if any('elevation' in h for h in head):
        col = next(i for i, h in enumerate(head) if 'elevation' in h)
        return _check([float(r[col]) for r in rows[1:] if len(r) > col and r[col].strip()])
    vals = []
    for line in t.replace(',', '\n').splitlines():
        line = line.strip()
        if line and not line.startswith('#'):
            vals.append(float(line))
    return _check(vals)


def _find_key(d, key):
    if isinstance(d, dict):
        if key in d:
            return d[key]
        for v in d.values():
            r = _find_key(v, key)
            if r is not None:
                return r
    return None


def _check(vals: list[float]) -> list[float]:
    vals = [v for v in vals if math.isfinite(v)]
    if len(vals) < 2:
        raise ValueError('Fewer than two beam angles found.')
    if max(abs(v) for v in vals) > 90:
        raise ValueError('Beam angles beyond ±90°: not elevations in degrees.')
    return sorted(vals)
