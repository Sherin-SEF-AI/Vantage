"""Lidars, radars, GNSS antennas and IMUs of the rig: the sensor library and their geometry.

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

GNSS antenna: its position is the antenna's phase centre; it needs open sky above its elevation mask. IMU (or a
GNSS/INS unit): its position is the centre of its measurement axes, its frame its x/y/z axes; an INS with two antenna
inputs (antennas = 2) gets its heading from the baseline between two GNSS antennas, heading_sigma_deg at
heading_baseline_m (the error shrinks with a longer baseline).

The library holds real products with the figures their makers state in public datasheets, manuals and product pages
(LIBRARY_NOTE); every model says where its figures come from in `note` and `url`, and what its maker does not state.
Check them against your unit's datasheet and calibration.
"""
from __future__ import annotations

import csv
import io
import json
import math
from dataclasses import asdict, dataclass, field

import numpy as np

from .mathutil import rot_zyx

KINDS = ('lidar', 'radar', 'gnss', 'imu')
KIND_LABEL = {'lidar': 'Lidar', 'radar': 'Radar', 'gnss': 'GNSS antenna', 'imu': 'IMU'}
KIND_NOUN = {'lidar': 'lidar', 'radar': 'radar', 'gnss': 'GNSS antenna', 'imu': 'IMU'}     # in running text
SCANS = ('spinning', 'solid', 'nonrepetitive')
SCAN_LABEL = {'spinning': 'Spinning', 'solid': 'Fixed field', 'nonrepetitive': 'Non-repetitive', 'fixed': 'Fixed field'}
REF_RCS = 10.0                       # m², the car-sized target a radar datasheet range is given for
MIN_FOOT_M = 0.002                   # a sensor's footprint never shrinks below 2 mm in the plan view


RANGING = {'lidar': ('fmcw',), 'radar': ('mimo',)}     # SensorModel.ranging besides '' (the kind's usual model)

@dataclass
class SensorModel:
    """One lidar, radar, GNSS antenna or IMU product. Angles in degrees, ranges in metres."""
    id: str
    name: str = 'Lidar'
    kind: str = 'lidar'              # lidar | radar | gnss | imu
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
    data_mbps: float = 0.0           # radar, IMU: output data rate, Mbit/s
    interface: str = '1000BASE-T1'
    time_sync: str = 'PTP'           # PTP (IEEE 1588 / gPTP) | PPS + NMEA | none
    size_mm: tuple = (87.0, 87.0, 74.0)    # spinning: diameter, diameter, height; others: depth (x), width (y), height
    optical_z_mm: float = 37.0       # optical centre above the base
    power_w: float = 0.0
    mask_deg: float = 10.0           # GNSS antenna: elevation mask, degrees above its horizon (the receiver's setting)
    lna_gain_db: float = 0.0         # GNSS antenna: its LNA gain, dB (0 = not given: the cable study's setting)
    antennas: int = 0                # IMU / INS: GNSS antenna inputs (0 none, 1 position, 2 position + heading)
    heading_sigma_deg: float = 0.0   # INS with two antennas: heading error (1 sigma) at heading_baseline_m
    heading_baseline_m: float = 2.0
    pos_sigma_cm: float = 0.0        # INS: horizontal position error (1 sigma) with RTK corrections
    attitude_sigma_deg: float = 0.0  # INS: roll and pitch error (1 sigma)
    accel_noise_density: float = 0.0  # IMU: accelerometer white noise, m/s²/√Hz (Kalibr's accelerometer_noise_density)
    accel_random_walk: float = 0.0   # IMU: accelerometer bias random walk, m/s³/√Hz
    gyro_noise_density: float = 0.0  # IMU: gyro white noise, rad/s/√Hz (an ARW in °/√h × π/180 / 60)
    gyro_random_walk: float = 0.0    # IMU: gyro bias random walk, rad/s²/√Hz (0 in any of the four: not given)
    note: str = ''                   # where the figures come from
    url: str = ''                    # the source page or datasheet
    builtin: bool = False
    ranging: str = ''                # '' = the kind's usual model (lidar: pulsed time of flight; radar: the zone
                                     # model); lidar 'fmcw' (coherent: range and radial velocity, core.fmcw_lidar);
                                     # radar 'mimo' (a virtual array, core.imaging_radar)
    params: dict = field(default_factory=dict)   # the ranging model's own datasheet figures (fmcw: velocity range and
                                     # resolution, wavelength …; mimo: carrier, bandwidth, the virtual array …), owned by
                                     # the module named above; a figure not stated is absent, never guessed

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
        m.antennas = min(max(int(m.antennas), 0), 2)
        m.ranging = m.ranging if m.ranging in RANGING.get(m.kind, ()) else ''
        m.params = dict(m.params) if isinstance(m.params, dict) else {}
        for k in ('accel_noise_density', 'accel_random_walk', 'gyro_noise_density', 'gyro_random_walk'):
            setattr(m, k, max(float(getattr(m, k)), 0.0))
        m.heading_baseline_m = m.heading_baseline_m if m.heading_baseline_m > 0 else 2.0
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
    def is_radar(self):
        return self.kind == 'radar'

    @property
    def has_field(self):
        """Lidars and radars see a field; GNSS antennas and IMUs do not."""
        return self.kind in ('lidar', 'radar')

    @property
    def spinning(self):
        return self.is_lidar and self.scan == 'spinning'

    def heading_sigma(self, baseline_m: float) -> float:
        """INS heading error (1 sigma, degrees) with two antennas baseline_m apart: the datasheet figure scaled by
        heading_baseline_m / baseline (the relative carrier-phase position error does not depend on the baseline)."""
        if self.heading_sigma_deg <= 0 or baseline_m <= 1e-6:
            return float('nan')
        return float(self.heading_sigma_deg * self.heading_baseline_m / baseline_m)

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
        """Recorded data rate, bit/s: lidar points × bytes per point; a radar's or IMU's stated output; nothing for a
        GNSS antenna (its receiver's output is the INS's)."""
        if self.is_lidar:
            return self.pps() * max(self.bytes_per_point, 0.0) * 8.0
        if self.kind == 'gnss':
            return 0.0
        return max(self.data_mbps, 0.0) * 1e6

    def v_fov(self) -> tuple[float, float]:
        """(lowest, highest) elevation of the field."""
        if self.is_lidar:
            e = self.elevations()
            return float(e[0]), float(e[-1])
        if self.kind == 'gnss':
            return float(self.mask_deg), 90.0
        el = max((z[2] for z in self.zones), default=0.0)
        return -el, el

    def az_half(self) -> float:
        """Half the horizontal field, degrees (180 for a spinning lidar)."""
        if self.is_lidar:
            return 180.0 if self.h_fov >= 359.999 else self.h_fov / 2
        if self.kind == 'gnss':
            return 180.0
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
        if self.kind == 'gnss':
            return f'mask {self.mask_deg:g}° · {self.power_w:g} W'
        if self.kind == 'imu':
            parts = [f'{self.rate_hz:g} Hz']
            if self.antennas:
                parts.append(f'{self.antennas} antenna' + ('s' if self.antennas > 1 else ''))
            if self.heading_sigma_deg > 0:
                parts.append(f'heading {self.heading_sigma_deg:g}° @ {self.heading_baseline_m:g} m')
            if self.pos_sigma_cm > 0:
                parts.append(f'RTK {self.pos_sigma_cm:g} cm')
            return ' · '.join(parts)
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
CHECKED = '2026-11-04'             # when the vendors' pages and datasheets were read for the figures below
LIBRARY_NOTE = (
    'Built-in models: real products, with the figures their makers state in public datasheets, user manuals and '
    f'product pages (read {CHECKED}; each model names its source in its note and url). A figure the maker does not '
    'state is said so in the note; where a modelled figure had to be read off a drawing or converted (a range on a '
    'stated cross-section to the 10 m² car the radar ranges here are given for, say), the note says how. Check them '
    'against your own unit\'s datasheet and calibration.')

# the beam elevations of RoboSense's RS-Ruby Plus as its user guide lists them (v1.1, Appendix E, Table 36), lowest first
_RUBY_PLUS_BEAMS = (
    -25.1, -19.64, -16.07, -13.61, -11.78, -10.37, -9.27, -8.38, -7.67, -7.17, -6.87, -6.67, -6.52, -6.4, -6.31,
    -6.21, -6.1, -6.01, -5.91, -5.81, -5.71, -5.6, -5.51, -5.41, -5.3, -5.2, -5.1, -5.0, -4.9, -4.8, -4.7, -4.6, -4.5,
    -4.4, -4.3, -4.2, -4.1, -4.0, -3.9, -3.8, -3.7, -3.6, -3.5, -3.4, -3.3, -3.2, -3.1, -3.0, -2.9, -2.8, -2.7, -2.6,
    -2.5, -2.39, -2.3, -2.2, -2.09, -2.0, -1.9, -1.8, -1.69, -1.59, -1.49, -1.39, -1.29, -1.19, -1.09, -0.99, -0.89,
    -0.79, -0.69, -0.59, -0.49, -0.39, -0.29, -0.19, -0.09, 0.01, 0.11, 0.21, 0.31, 0.41, 0.51, 0.61, 0.71, 0.81,
    0.91, 1.01, 1.11, 1.21, 1.31, 1.41, 1.51, 1.61, 1.71, 1.82, 1.91, 2.02, 2.12, 2.22, 2.32, 2.41, 2.52, 2.62, 2.72,
    2.82, 2.92, 3.02, 3.12, 3.22, 3.32, 3.42, 3.52, 3.62, 3.72, 3.82, 3.97, 4.17, 4.42, 4.72, 5.07, 5.48, 5.98, 6.58,
    7.43, 9.02, 11.53, 15.04)
# HDL-32E: 32 beams from -30.67° to +10.67°, 1.33° apart (user manual 63-9113, Table 9-2)
_HDL32_BEAMS = tuple(round(-30.67 + i * 41.34 / 31, 2) for i in range(32))
# HDL-64E S2 / S3: the manual states the spacing, not a table: 32 upper lasers from +2° down to -8.33° at 1/3°, 32
# lower lasers from -8.83° to -24.33° at 1/2° (each unit's own angles are in its db.xml)
_HDL64_BEAMS = tuple(sorted([round(2.0 - i / 3, 2) for i in range(32)] + [round(-8.83 - i * 0.5, 2) for i in range(32)]))


def _src(url, text) -> str:
    return f'{text} Source: {url} (read {CHECKED}).'


_OUSTER = 'https://data.ouster.io/downloads/datasheets/'
_OUSTER_FW = 'https://data.ouster.io/downloads/software-user-manual/firmware-user-manual-v3.1.0.pdf'
_HESAI = 'https://www.hesaitech.com/product/'
_VELO = 'https://web.archive.org/web/2017/http://velodynelidar.com/docs/'
_AUMOVIO = 'https://engineering-solutions.aumovio.com/wp-content/uploads/2025/10/'
_SMS = 'https://www.smartmicro.com/wp-content/uploads/2026/03/'
_NO_SYNC = ('The datasheet states no time synchronisation (no PTP): the app stamps its data on arrival at the host.')
_NO_RATE = 'The datasheet gives no output data rate (smartmicro states up to 255 targets per cycle).'


def _lidars() -> list[SensorModel]:
    os1 = _OUSTER + 'datasheet-rev7-v3p2-os1.pdf'
    return [
        SensorModel('ouster_os1_128', 'OS1-128 (Rev 7)', 'lidar', 'Ouster', 'spinning', 128, -21.2, 21.2,
                    h_fov=360.0, h_res=360.0 / 1024, rate_hz=10.0, range_m=90.0, range_max_m=170.0, range_min_m=0.5,
                    points_per_s=1_310_720, bytes_per_point=12.0, rotation='cw', interface='1000BASE-T',
                    time_sync='PTP', size_mm=(87.0, 87.0, 74.2), optical_z_mm=38.195, power_w=16.0, url=os1,
                    note=_src(os1, '42.4° vertical field (+21.2° to −21.2°), 90 m on a 10 % target in 100 klx '
                              'sunlight, 14–20 W (16 W nominal), 87 mm across and 74.2 mm high with the thermal cap. '
                              'The beam origin, 38.195 mm above the base, is read from the datasheet\'s dimension '
                              'drawing. The datasheet\'s point rate is 5,242,880 /s (2048 columns, two returns); the '
                              '1024 × 10 Hz single-return mode used here gives 1,310,720 /s, 12 bytes per point in '
                              'the default packet profile (firmware manual §5.1.3). Each unit\'s beam angles are in '
                              'its metadata: import them for the exact rings.'), builtin=True),
        SensorModel('ouster_os0_128', 'OS0-128 (Rev 7)', 'lidar', 'Ouster', 'spinning', 128, -45.4, 45.4,
                    h_fov=360.0, h_res=360.0 / 1024, rate_hz=10.0, range_m=35.0, range_max_m=75.0, range_min_m=0.5,
                    points_per_s=1_310_720, bytes_per_point=12.0, rotation='cw', interface='1000BASE-T',
                    time_sync='PTP', size_mm=(87.0, 87.0, 74.2), optical_z_mm=38.195, power_w=16.0,
                    url=_OUSTER + 'datasheet-rev7-v3p2-os0.pdf',
                    note=_src(_OUSTER + 'datasheet-rev7-v3p2-os0.pdf',
                              'Wide-field lidar: 90.8° vertical field, 35 m on a 10 % target, 75 m on 80 %, 14–20 W '
                              '(16 W nominal). Beam origin 38.195 mm above the base (dimension drawing). 1024 × 10 Hz '
                              'single return (1,310,720 points/s, derived from the mode; the datasheet states the '
                              'dual-return maximum).'), builtin=True),
        SensorModel('ouster_os2_128', 'OS2-128 (Rev 7)', 'lidar', 'Ouster', 'spinning', 128, -11.25, 11.25,
                    h_fov=360.0, h_res=360.0 / 1024, rate_hz=10.0, range_m=200.0, range_max_m=350.0,
                    range_min_m=0.8, points_per_s=1_310_720, bytes_per_point=12.0, rotation='cw',
                    interface='1000BASE-T', time_sync='PTP', size_mm=(119.6, 119.6, 102.9), optical_z_mm=78.296,
                    power_w=24.0, url=_OUSTER + 'datasheet-rev7-v2p5-os2.pdf',
                    note=_src(_OUSTER + 'datasheet-rev7-v2p5-os2.pdf',
                              'Long-range lidar: 22.5° vertical field, 200 m on a 10 % target, 350 m on 80 %, '
                              '119.6 mm across and 102.9 mm high, 1.1 kg. 18–24 W with no nominal figure: the top is '
                              'used. Beam origin 78.296 mm above the base (dimension drawing). 1024 × 10 Hz single '
                              'return (1,310,720 points/s, derived from the mode).'), builtin=True),
        SensorModel('hesai_ot128', 'OT128', 'lidar', 'Hesai', 'spinning', 128, -25.0, 15.0,
                    h_fov=360.0, h_res=360.0 / 2700, rate_hz=10.0, range_m=200.0, range_max_m=230.0, range_min_m=0.3,
                    points_per_s=3_456_000, bytes_per_point=4.5, rotation='cw', interface='1000BASE-T1',
                    time_sync='PTP', size_mm=(118.0, 118.0, 132.3), optical_z_mm=60.0, power_w=29.0,
                    url=_HESAI + 'ot128/',
                    note=_src(_HESAI + 'ot128/',
                              '40° vertical field (−24.77° to +14.99° in Hesai\'s published angle-correction file), '
                              '200 m on a 10 % target, 29 W, 118 mm across and 132.3 mm high. Not on Hesai\'s product '
                              'page: the beam origin height (60 mm is an estimate; take it from your unit\'s user '
                              'manual). The channels are denser near the horizon (0.125° there); modelled as an even '
                              'spread unless beam angles are imported.'), builtin=True),
        SensorModel('hesai_at128', 'AT128', 'lidar', 'Hesai', 'solid', 128, -12.5, 12.9,
                    h_fov=120.0, h_res=0.1, rate_hz=10.0, range_m=210.0, range_min_m=0.5,
                    points_per_s=1_536_000, bytes_per_point=4.5, interface='1000BASE-T1', time_sync='PTP',
                    size_mm=(114.0, 136.0, 49.0), optical_z_mm=24.0, power_w=13.5, url=_HESAI + 'at128/',
                    note=_src(_HESAI + 'at128/',
                              'Hybrid solid-state: 120° × 25.4° field, 0.1° × 0.2°, 210 m on a 10 % target, 13.5 W, '
                              '136 wide × 114 deep × 49 mm high, 940 g. Not on Hesai\'s product page: the optical '
                              'centre height (24 mm, half the height, is an estimate).'), builtin=True),
        SensorModel('hesai_xt32', 'XT32', 'lidar', 'Hesai', 'spinning', 32, -16.0, 15.0,
                    h_fov=360.0, h_res=0.18, rate_hz=10.0, range_m=80.0, range_max_m=120.0, range_min_m=0.05,
                    points_per_s=640_000, bytes_per_point=4.5, rotation='cw', interface='1000BASE-T',
                    time_sync='PTP', size_mm=(103.2, 103.2, 76.0), optical_z_mm=38.0, power_w=10.0,
                    url=_HESAI + 'xt32/',
                    note=_src(_HESAI + 'xt32/',
                              '31° vertical field (1° between channels), 80 m on a 10 % target, 120 m rated, 800 g. '
                              'Hesai\'s page prints the size as "Φ 76 × H 103.2 mm", taken as transposed (103.2 mm '
                              'across, 76 mm high: the unit is a squat puck). Not on the product page: power (10 W) '
                              'and the beam origin height (38 mm, half the height); check them in the user '
                              'manual.'), builtin=True),
        SensorModel('velodyne_vlp16', 'VLP-16 (Puck)', 'lidar', 'Velodyne', 'spinning', 16, -15.0, 15.0,
                    h_fov=360.0, h_res=0.2, rate_hz=10.0, range_m=100.0, range_min_m=0.5,
                    points_per_s=300_000, bytes_per_point=3.25, rotation='cw', interface='100BASE-TX',
                    time_sync='PPS + NMEA', size_mm=(103.3, 103.3, 71.7), optical_z_mm=37.7, power_w=8.0,
                    url='https://ouster.com/downloads/velodyne-downloads',
                    note=_src('https://ouster.com/downloads/velodyne-downloads (datasheet 63-9229, user manual '
                              '63-9243)',
                              'Up to 100 m (no reflectivity given); 2° between channels; the data origin is 37.7 mm '
                              'above the base (user manual §9.1); 1248-byte frames of 384 points.'), builtin=True),
        SensorModel('velodyne_hdl32e', 'HDL-32E', 'lidar', 'Velodyne', 'spinning', 32, -30.67, 10.67,
                    beams=list(_HDL32_BEAMS), h_fov=360.0, h_res=0.166, rate_hz=10.0, range_m=100.0,
                    range_min_m=1.0, points_per_s=695_000, bytes_per_point=3.25, rotation='cw',
                    interface='100BASE-TX', time_sync='PPS + NMEA', size_mm=(85.3, 85.3, 144.2),
                    optical_z_mm=90.81, power_w=12.0,
                    url=_VELO + 'datasheet/97-0038_Rev%20K_%20HDL-32E_Datasheet_Web.pdf',
                    note=_src(_VELO + 'datasheet/97-0038_Rev%20K_%20HDL-32E_Datasheet_Web.pdf (and user manual '
                              '63-9113)',
                              'The lidar of the nuScenes dataset. 32 beams from −30.67° to +10.67° (the manual\'s '
                              'table), up to 100 m (no reflectivity given; the older manual says 1–70 m), about '
                              '695,000 points/s, 12 W, 85.3 mm across and 144.2 mm high, 1 kg. The data origin is '
                              '90.8 mm above the base (manual §9.1). A legacy product: the Velodyne documents are '
                              'cited from velodynelidar.com through the Internet Archive.'), builtin=True),
        SensorModel('velodyne_hdl64e_s2', 'HDL-64E S2', 'lidar', 'Velodyne', 'spinning', 64, -24.33, 2.0,
                    beams=list(_HDL64_BEAMS), h_fov=360.0, h_res=0.1728, rate_hz=10.0, range_m=50.0,
                    range_max_m=120.0, range_min_m=0.9, points_per_s=1_333_000, bytes_per_point=3.25,
                    rotation='cw', interface='100BASE-TX', time_sync='PPS + NMEA', size_mm=(223.5, 231.1, 257.3),
                    optical_z_mm=206.5, power_w=48.0,
                    url=_VELO + 'manuals/63-HDL64ES2h%20HDL-64E%20S2%20CD%20Users%20Manual.pdf',
                    note=_src(_VELO + 'manuals/63-HDL64ES2h%20HDL-64E%20S2%20CD%20Users%20Manual.pdf',
                              'The lidar family of the KITTI dataset (its paper names the HDL-64E without the '
                              'variant). 64 beams: +2° to −8.33° at 1/3° and −8.83° to −24.33° at 1/2° (the '
                              'manual\'s spacing; each unit\'s angles are in its db.xml), 50 m on pavement (about '
                              '10 %), 120 m on cars, about 1.33 million points/s at 0.1728° and 10 Hz. 12 V at 4 A '
                              '(48 W). Two laser blocks with their own optical centres, 206.5 mm and 134.1 mm above '
                              'the base (outline drawing 86-0105): the upper block\'s is used. 13 kg.'),
                    builtin=True),
        SensorModel('velodyne_hdl64e_s3', 'HDL-64E S3', 'lidar', 'Velodyne', 'spinning', 64, -24.33, 2.0,
                    beams=list(_HDL64_BEAMS), h_fov=360.0, h_res=0.1728, rate_hz=10.0, range_m=50.0,
                    range_max_m=120.0, range_min_m=0.9, points_per_s=1_333_000, bytes_per_point=3.25,
                    rotation='cw', interface='100BASE-TX', time_sync='PPS + NMEA', size_mm=(223.5, 231.1, 283.0),
                    optical_z_mm=229.4, power_w=48.0,
                    url=_VELO + 'manuals/63-HDL64ES3%20REV%20J%20MANUAL,USERS%20AND%20PROGRAM%20GUIDE,'
                                'HDL-64E%20S3.pdf',
                    note=_src(_VELO + 'manuals/ (HDL-64E S3 user manual Rev J)',
                              'The same beam spacing as the S2 (manual Appendix L), 50 m on pavement (about 10 %), '
                              '120 m on cars, about 1.333 million points/s. At 12 V it draws 4 A (48 W). Two laser '
                              'blocks, 229.4 mm and 157.0 mm above the base (outline drawing 86-0104): the upper '
                              'block\'s is used. 13.6 kg.'), builtin=True),
        SensorModel('robosense_ruby_plus', 'RS-Ruby Plus', 'lidar', 'RoboSense', 'spinning', 128, -25.1, 15.04,
                    beams=list(_RUBY_PLUS_BEAMS), h_fov=360.0, h_res=0.2, rate_hz=10.0, range_m=240.0,
                    range_max_m=250.0, range_min_m=0.4, points_per_s=2_304_000, bytes_per_point=3.25,
                    rotation='cw', interface='1000BASE-T1', time_sync='PTP', size_mm=(125.0, 125.0, 128.0),
                    optical_z_mm=73.5, power_w=27.0, url='https://www.robosense.ai/en/rslidar/RubyPlus',
                    note=_src('https://www.robosense.ai/en/rslidar/RubyPlus (and the Ruby Plus User Guide v1.1)',
                              '128 beams at the user guide\'s angles (Appendix E), 2,304,000 points/s at 0.2° and '
                              '10 Hz, 27 W, 125 mm across and 128 mm high, 1.92 kg. 240 m on a 10 % target holds on '
                              '32 of the 128 channels only (the steep ones reach 70 m): the guide lists each '
                              'channel\'s range. Origin 73.5 mm above the base (user guide §4.1).'), builtin=True),
        SensorModel('livox_mid360', 'Mid-360', 'lidar', 'Livox', 'nonrepetitive', 40, -7.0, 52.0,
                    h_fov=360.0, h_res=0.72, rate_hz=10.0, range_m=40.0, range_max_m=70.0, range_min_m=0.1,
                    points_per_s=200_000, bytes_per_point=14.0, interface='100BASE-TX', time_sync='PTP',
                    size_mm=(65.0, 65.0, 60.0), optical_z_mm=47.0, power_w=6.5,
                    url='https://www.livoxtech.com/mid-360/specs',
                    note=_src('https://www.livoxtech.com/mid-360/specs',
                              'Non-repetitive scan, modelled as an even grid of 20,000 points per 10 Hz frame; 40 m '
                              'on a 10 % target, 70 m on 80 %; shorter range near the top of the field. Origin 47 mm '
                              'above the base (user manual drawing).'), builtin=True),
        SensorModel('livox_hap', 'HAP', 'lidar', 'Livox', 'nonrepetitive', 108, -12.5, 12.5,
                    h_fov=120.0, h_res=0.18, rate_hz=10.0, range_m=150.0, range_max_m=200.0, range_min_m=0.5,
                    points_per_s=452_000, bytes_per_point=14.0, interface='100BASE-TX', time_sync='PTP',
                    size_mm=(131.6, 105.0, 65.0), optical_z_mm=23.0, power_w=12.0,
                    url='https://www.livoxtech.com/hap/specs',
                    note=_src('https://www.livoxtech.com/hap/specs (and the HAP user manual)',
                              '120° × 25° non-repetitive field, modelled as an even grid of 45,200 points per 10 Hz '
                              'frame; 150 m on a 10 % target (about 125 m near the vertical edges), 200 m on 35 %, '
                              '12 W, 1.12 kg, gPTP. Livox gives the vertical field only as 25°: it is taken as '
                              'centred (±12.5°), as the manual draws it. Origin 23 mm above the base (manual §5).'),
                    builtin=True),
    ]


def _radars() -> list[SensorModel]:
    ars408 = _AUMOVIO + '121118_FS_ARS-408-21_EN_HS.pdf'
    ars548 = _AUMOVIO + 'FS_ARS-548-RDI_EN.pdf'
    srr308 = _AUMOVIO + '121118_FS_SRR308-21_EN_HS.pdf'
    conti = (' Continental\'s industrial radars are now sold by AUMOVIO Engineering Solutions. The datasheet gives the '
             'size as "L × W × H": taken as width × height × depth (the thin side faces the road).')
    return [
        SensorModel('conti_ars408_21', 'ARS 408-21', 'radar', 'Continental', 'fixed',
                    zones=[[250.0, 9.0, 7.0], [70.0, 45.0, 10.0], [20.0, 60.0, 10.0]], range_min_m=0.2,
                    rate_hz=13.9, data_mbps=0.5, interface='CAN (500 kbit/s)', time_sync='none',
                    size_mm=(31.0, 138.0, 91.0), optical_z_mm=45.5, power_w=6.6, url=ars408,
                    note=_src(ars408, 'The radar of the nuScenes dataset; 77 GHz, 2D (no elevation measured). Far '
                              'beam ±9° to 250 m, near beam ±45° to 70 m (the datasheet\'s "70 m/100 m" is not '
                              'explained: 70 m is used) and ±60° to 20 m; elevation 14° far, 20° near (taken as full '
                              'angles). The ranges are on "natural targets", none stated: taken as the car the app\'s '
                              'radar ranges are for. Cycle about 72 ms (13.9 Hz; the introduction says 17 scans/s). '
                              'Output over one 500 kbit/s CAN bus: the bus\'s 0.5 Mbit/s is used as its data rate '
                              '(an upper bound). 6.6 W typical, 12 W peak, 320 g, IP6K9K. ' + _NO_SYNC + conti),
                    builtin=True),
        SensorModel('conti_ars548_rdi', 'ARS 548 RDI', 'radar', 'Continental', 'fixed',
                    zones=[[301.0, 50.0, 4.0], [100.0, 50.0, 14.0]], range_min_m=0.2, rate_hz=20.0,
                    interface='100BASE-T1', time_sync='none', size_mm=(39.0, 137.0, 90.0), optical_z_mm=45.0,
                    power_w=18.0, url=ars548,
                    note=_src(ars548, '4D long-range radar, 77 GHz: 0.2–301 m (Continental\'s 2022 edition; the '
                              'current datasheet\'s range line is garbled), ±50° output field, elevation ±4° at 300 m '
                              'and ±14° within 100 m (two zones here). Range accuracy ±0.15 m, resolution 0.22 m. '
                              'Cycle about 50 ms. BroadR-Reach (100BASE-T1) Ethernet; the data rate is not stated. '
                              '18 W typical, 23 W maximum, 526 g. ' + _NO_SYNC + conti), builtin=True),
        SensorModel('conti_srr308_21', 'SRR 308-21', 'radar', 'Continental', 'fixed',
                    zones=[[120.0, 75.0, 12.0]], range_min_m=0.3, rate_hz=25.0, data_mbps=0.5,
                    interface='CAN (500 kbit/s)', time_sync='none', size_mm=(25.0, 80.0, 89.3), optical_z_mm=44.6,
                    power_w=3.9, url=srr308,
                    note=_src(srr308, 'Corner radar, 24 GHz (the only Continental corner radar with a public '
                              'datasheet), 2D: ±75° measured (±90° detection only), 95 m on a 6 dBsm target and 80 m '
                              'on 3 dBsm. Both convert to 120 m on the 10 m² car the app\'s radar ranges are for '
                              '(range ∝ RCS^¼, the radar equation). Elevation ±12° at −6 dB (the product page writes '
                              '±23° at −20 dB). Range accuracy ±0.2 m beyond 2 m; angle ±2° at ±30°, ±5° at ±75°. '
                              'Cycle 40 ms. 500 kbit/s CAN (0.5 Mbit/s used, an upper bound). 3.9 W, 172 g. Size '
                              'without the mount. ' + _NO_SYNC + conti), builtin=True),
        SensorModel('smartmicro_drvegrd152', 'DRVEGRD 152', 'radar', 'smartmicro', 'fixed',
                    zones=[[178.0, 50.0, 10.0]], range_min_m=0.9, rate_hz=18.2, interface='100BASE-T1',
                    time_sync='none', size_mm=(20.3, 143.0, 97.0), optical_z_mm=48.5, power_w=6.0,
                    url=_SMS + 'DRVEGRD_152_Automotive_Data_Sheet.pdf',
                    note=_src(_SMS + 'DRVEGRD_152_Automotive_Data_Sheet.pdf',
                              '4D imaging radar, 77 GHz, in its long-range mode: ±50° × ±10°, 178 m on a 10 dBsm '
                              'reflector (the datasheet\'s "sensitivity on passenger car"), from 0.9 m; the '
                              'medium-range mode reaches 65 m from 0.33 m (add it as a copy). Update cycle ≤ 55 ms. '
                              'Under 6 W, 310 g, IP67, 97 × 143 × 20.3 mm. ' + _NO_RATE + ' ' + _NO_SYNC),
                    builtin=True),
        SensorModel('smartmicro_drvegrd171', 'DRVEGRD 171', 'radar', 'smartmicro', 'fixed',
                    zones=[[180.0, 50.0, 10.0]], range_min_m=1.2, rate_hz=18.2, interface='100BASE-T1',
                    time_sync='none', size_mm=(25.7, 143.0, 97.0), optical_z_mm=48.5, power_w=7.0,
                    url=_SMS + 'DRVEGRD_171_Automotive_Data_Sheet.pdf',
                    note=_src(_SMS + 'DRVEGRD_171_Automotive_Data_Sheet.pdf',
                              '4D imaging radar, 77 GHz, in its extra-long-range mode: ±50° × ±10°, 180 m on a '
                              'passenger car (240 m maximum; the product page gives 240 m at 10 dBsm), from 1.2 m; '
                              'the other modes reach 98 m and 42 m. Update cycle ≤ 55 ms. 7 W, 455 g, '
                              '97 × 143 × 25.7 mm. ' + _NO_RATE + ' ' + _NO_SYNC), builtin=True),
        SensorModel('smartmicro_drvegrd169', 'DRVEGRD 169', 'radar', 'smartmicro', 'fixed',
                    zones=[[95.0, 65.0, 7.5]], range_min_m=1.3, rate_hz=14.3, interface='100BASE-T1',
                    time_sync='none', size_mm=(17.7, 79.0, 97.0), optical_z_mm=48.5, power_w=4.0,
                    url=_SMS + 'DRVEGRD_169_Automotive_Data_Sheet.pdf',
                    note=_src(_SMS + 'DRVEGRD_169_Automotive_Data_Sheet.pdf',
                              '4D corner radar, 77 GHz, in its long-range mode: ±65° × ±7.5°, 95 m on a 10 dBsm '
                              'reflector (130 m maximum), from 1.3 m; the shorter modes reach 55 m, 19 m and 9.5 m '
                              '(±70° × ±14°). Update cycle ≤ 70 ms. 4 W, 186 g, 97 × 79 × 17.7 mm. ' + _NO_RATE + ' '
                              + _NO_SYNC), builtin=True),
    ]


def builtin_sensor_models() -> list[SensorModel]:
    return _lidars() + _radars() + _gnss_imu() + _ranging_library()


RANGING_LIBRARIES = ('fmcw_lidar', 'imaging_radar')   # each offers library() -> [SensorModel] from its datasheets


def _ranging_library() -> list:
    import importlib
    out = []
    for name in RANGING_LIBRARIES:
        try:
            mod = importlib.import_module(f'{__package__}.{name}')
        except ImportError:
            continue
        fn = getattr(mod, 'library', None)
        if fn is not None:
            out += list(fn())
    return out


def _gnss_imu() -> list[SensorModel]:
    hx = 'https://hexagondownloads.blob.core.windows.net/public/Novatel/assets/Documents/Papers/'
    mask = (' The elevation mask is the receiver\'s setting, not the antenna\'s (10° here; set yours).')
    return [
        SensorModel('novatel_gnss850', 'VEXXIS GNSS-850', 'gnss', 'NovAtel', 'fixed', rate_hz=10.0, mask_deg=10.0,
                    interface='TNC (coax)', time_sync='none', size_mm=(176.0, 176.0, 55.0), optical_z_mm=51.7,
                    lna_gain_db=29.0, url=hx + 'GNSS-850-Product-Sheet/GNSS-850-Product-Sheet.pdf',
                    note=_src(hx + 'GNSS-850-Product-Sheet/GNSS-850-Product-Sheet.pdf (and the GNSS-800 Series User '
                                   'Guide)',
                              'Multi-constellation survey antenna for a vehicle roof (GPS L1/L2/L5, GLONASS, Galileo '
                              'E1/E5/E6, BeiDou B1/B2/B3, L-band), 29 dB LNA, 176 mm across and 55 mm high, 507 g, '
                              'IP69K. Phase centre 51.7 mm (L1) and 52.0 mm (L2) up, from the user guide\'s table for '
                              'the GNSS-800 series (taken from the mount\'s base). Powered by its receiver through the '
                              'coax: 55 mA typical at 3.8–18 V.' + mask), builtin=True),
        SensorModel('ublox_ann_mb_00', 'ANN-MB-00', 'gnss', 'u-blox', 'fixed', rate_hz=10.0, mask_deg=10.0,
                    interface='SMA (coax)', time_sync='none', size_mm=(60.0, 82.0, 22.5), optical_z_mm=8.9,
                    lna_gain_db=28.0,
                    url='https://content.u-blox.com/sites/default/files/documents/ANN-MB_DataSheet_UBX-18049862.pdf',
                    note=_src('https://content.u-blox.com/sites/default/files/documents/ANN-MB_DataSheet_UBX-18049862'
                              '.pdf',
                              'Compact dual-band (L1/L2, E1/E5b, B1I/B2I) patch antenna on a magnetic base, 28 dB '
                              'LNA, 82 × 60 × 22.5 mm with its screw tabs, 173 g with its 5 m RG-174 cable, IP67. '
                              'Phase centre 8.9 mm (L1) and 7.6 mm (L2) above its base, measured on a 120 mm ground '
                              'plane: without one (a plastic roof) it is more prone to multipath. Powered through the '
                              'coax: 15 mA typical at 5 V.' + mask), builtin=True),
        SensorModel('oxts_rt3000_v4', 'RT3000 v4', 'imu', 'OxTS', 'fixed', rate_hz=100.0, interface='100BASE-TX',
                    time_sync='PPS + NMEA', size_mm=(120.0, 120.0, 71.0), optical_z_mm=35.5, power_w=6.0,
                    antennas=2, heading_sigma_deg=0.04, heading_baseline_m=2.0, pos_sigma_cm=1.0,
                    attitude_sigma_deg=0.01, accel_noise_density=2.0e-4, gyro_noise_density=3.4907e-5,
                    url='https://www.oxts.com/solutions/inertial-navigation-solutions/navigation-hardware/rt3000-v4/',
                    note=_src('https://www.oxts.com/solutions/inertial-navigation-solutions/navigation-hardware/'
                              'rt3000-v4/ (datasheet via its distributor Positics)',
                              'Dual-antenna MEMS GNSS/INS (OxTS\'s RT3003 was KITTI\'s): 100 Hz (250 Hz optional), '
                              'heading 0.04° (1σ) at a 2 m baseline, 1 cm position (CEP) and 0.01° roll and pitch '
                              'with RTK corrections and a wheel-speed input. Gyro random walk 0.12°/√h (3.49e-5 '
                              'rad/s/√Hz), accelerometer 0.012 m/s/√h (2.0e-4 m/s²/√Hz); bias stability 0.8°/h (not '
                              'a random walk: the bias walks are not given). 120 × 120 × 71 mm, 690 g, 6 W. PPS in '
                              'and out; PTP is an option. The position is the centre of the measurement axes marked '
                              'on the unit (taken at mid-height here); the output data rate is not stated.'),
                    builtin=True),
        SensorModel('novatel_pwrpak7d_e1', 'PwrPak7D-E1 (SPAN)', 'imu', 'NovAtel', 'fixed', rate_hz=200.0,
                    interface='100BASE-TX', time_sync='PTP', size_mm=(147.0, 125.0, 55.0), optical_z_mm=27.5,
                    power_w=4.15, antennas=2, heading_sigma_deg=0.08, heading_baseline_m=2.0, pos_sigma_cm=1.0,
                    attitude_sigma_deg=0.02, accel_noise_density=8.333e-4, gyro_noise_density=2.9089e-5,
                    url=hx + 'PwrPak7D-E1-PS/PwrPak7D-E1-PS.pdf',
                    note=_src(hx + 'PwrPak7D-E1-PS/PwrPak7D-E1-PS.pdf',
                              'Dual-antenna GNSS/INS with an Epson MEMS IMU: 200 Hz, heading 0.08° (RMS) at 2 m '
                              '(0.05° at 4 m), RTK 1 cm + 1 ppm (RMS), roll and pitch 0.02°. Gyro 0.1°/√h (2.91e-5 '
                              'rad/s/√Hz), accelerometer 0.05 m/s/√h (8.33e-4 m/s²/√Hz); bias stability 3.5°/h. PPS '
                              'out, PTP and gPTP. 147 × 125 × 55 mm, 510 g, 4.15 W. The output data rate is not '
                              'stated.'), builtin=True),
        SensorModel('novatel_span_cpt7', 'SPAN CPT7', 'imu', 'NovAtel', 'fixed', rate_hz=200.0,
                    interface='100BASE-TX', time_sync='PTP', size_mm=(90.0, 60.0, 60.0), optical_z_mm=30.0,
                    power_w=9.0, antennas=2, heading_sigma_deg=0.08, heading_baseline_m=2.0, pos_sigma_cm=1.0,
                    attitude_sigma_deg=0.01, accel_noise_density=1.0e-3, gyro_noise_density=1.7453e-5,
                    url=hx + 'SPAN-CPT7-PS/SPAN-CPT7-PS.pdf',
                    note=_src(hx + 'SPAN-CPT7-PS/SPAN-CPT7-PS.pdf',
                              'Tactical-grade GNSS/INS (Honeywell HG4930 MEMS IMU) for reference trajectories: '
                              '200 Hz, heading 0.08° (RMS) at 2 m, RTK 1 cm + 1 ppm, roll and pitch 0.01° (0.003° '
                              'post-processed). Gyro 0.06°/√h (1.75e-5 rad/s/√Hz), accelerometer 0.06 m/s/√h '
                              '(1.0e-3 m/s²/√Hz); bias stability 0.45°/h. PTP and gPTP. 90 × 60 × 60 mm, 500 g, '
                              '9 W.'), builtin=True),
        SensorModel('sbg_ellipse_d', 'Ellipse-D', 'imu', 'SBG Systems', 'fixed', rate_hz=200.0,
                    interface='RS-422 / CAN', time_sync='PPS + NMEA', size_mm=(46.0, 45.0, 32.0), optical_z_mm=16.0,
                    power_w=1.05, antennas=2, heading_sigma_deg=0.2, heading_baseline_m=2.0, pos_sigma_cm=1.0,
                    attitude_sigma_deg=0.05, accel_noise_density=5.590e-4, gyro_noise_density=5.2360e-5,
                    url='https://www.sbg-systems.com/ins/ellipse-d/',
                    note=_src('https://www.sbg-systems.com/ins/ellipse-d/',
                              'Miniature dual-antenna GNSS/INS: 200 Hz, heading 0.2° at a baseline over 2 m, RTK '
                              '1 cm + 1 ppm, roll and pitch 0.05° (RMS, land, with an odometer). Gyro 0.18°/√h '
                              '(5.24e-5 rad/s/√Hz), accelerometer 57 µg/√Hz (5.59e-4 m/s²/√Hz); bias stability '
                              '7°/h. PPS and event inputs; no PTP. 46 × 45 × 32 mm, 65 g, under 1.05 W (the page\'s '
                              'headline says 6 W).'), builtin=True),
        SensorModel('sbg_ekinox_d', 'Ekinox-D', 'imu', 'SBG Systems', 'fixed', rate_hz=200.0,
                    interface='100BASE-TX', time_sync='PTP', size_mm=(86.0, 100.0, 75.0), optical_z_mm=37.5,
                    power_w=5.9, antennas=2, heading_sigma_deg=0.02, heading_baseline_m=4.0, pos_sigma_cm=1.0,
                    attitude_sigma_deg=0.015, accel_noise_density=3.923e-4, gyro_noise_density=4.0724e-5,
                    url='https://www.sbg-systems.com/ins/ekinox-d/',
                    note=_src('https://www.sbg-systems.com/ins/ekinox-d/ (and the Ekinox hardware manual v1.4)',
                              'Tactical MEMS GNSS/INS: 200 Hz, heading 0.02° (1σ) at a baseline over 4 m, RTK 1 cm '
                              '+ 0.5 ppm, roll and pitch 0.015° (land, with an odometer). Gyro 0.14°/√h (4.07e-5 '
                              'rad/s/√Hz), accelerometer 40 µg/√Hz for the ±14 g grade (3.92e-4 m/s²/√Hz; 7 µg/√Hz '
                              'for ±8 g); bias stability 0.5°/h. PTP grandmaster and PPS. 100 × 86 × 75 mm, 610 g, '
                              '5.9 W.'), builtin=True),
        SensorModel('xsens_mti630', 'MTi-630 AHRS', 'imu', 'Movella (Xsens)', 'fixed', rate_hz=400.0,
                    interface='RS-232 / CAN', time_sync='PPS + NMEA', size_mm=(31.5, 28.0, 13.0), optical_z_mm=6.5,
                    power_w=0.5, antennas=0, attitude_sigma_deg=0.2, accel_noise_density=5.884e-4,
                    gyro_noise_density=1.2217e-4,
                    url='https://www.xsens.com/hubfs/MTi%20product%20leaflet_07022025_MTi-630.pdf',
                    note=_src('https://www.xsens.com/hubfs/MTi%20product%20leaflet_07022025_MTi-630.pdf',
                              'MEMS IMU and AHRS without GNSS: up to 2 kHz (400 Hz strapdown-integrated output, '
                              'used here), roll and pitch 0.2° RMS. Gyro noise 0.007°/s/√Hz (1.22e-4 rad/s/√Hz), '
                              'accelerometer 60 µg/√Hz (5.88e-4 m/s²/√Hz); in-run bias 8°/h and 10–15 µg. SyncIn '
                              'and SyncOut. 28 × 31.5 × 13 mm, 8.9 g, under 0.5 W.'), builtin=True),
        SensorModel('bosch_bmi088', 'BMI088', 'imu', 'Bosch Sensortec', 'fixed', rate_hz=400.0,
                    interface='SPI', time_sync='none', size_mm=(4.5, 3.0, 0.95), optical_z_mm=0.5, power_w=0.0155,
                    antennas=0, accel_noise_density=1.863e-3, gyro_noise_density=2.4435e-4,
                    url='https://www.bosch-sensortec.com/media/boschsensortec/downloads/datasheets/'
                        'bst-bmi088-ds001.pdf',
                    note=_src('https://www.bosch-sensortec.com/media/boschsensortec/downloads/datasheets/'
                              'bst-bmi088-ds001.pdf',
                              'A bare MEMS IMU chip (on flight controllers and many sensor boards), not a boxed unit: '
                              'gyro 100–2000 Hz, accelerometer up to 1600 Hz (400 Hz used). Gyro noise 0.014°/s/√Hz '
                              '(2.44e-4 rad/s/√Hz), accelerometer 190 µg/√Hz on z (1.86e-3 m/s²/√Hz; 160 on x and '
                              'y). No bias stability, weight or power figure: 15.5 mW is its stated supply currents '
                              '(5 mA gyro, 150 µA accelerometer) at 3.0 V. The 4.5 × 3 mm package goes on your own '
                              'board.'), builtin=True),
    ]


# library models that were generic placeholders in earlier versions ("· typical", no product), and the real model a project
# that used one moves to (Project.from_file; the design log says so)
RETIRED = {'radar_long': 'conti_ars408_21', 'radar_imaging': 'smartmicro_drvegrd152',
           'radar_corner': 'smartmicro_drvegrd169', 'gnss_survey': 'novatel_gnss850', 'gnss_patch': 'ublox_ann_mb_00',
           'ins_dual': 'oxts_rt3000_v4', 'ins_fog': 'novatel_span_cpt7', 'imu_mems': 'xsens_mti630'}


def default_model_id(kind: str) -> str:
    return {'lidar': 'ouster_os1_128', 'radar': 'conti_ars408_21', 'gnss': 'novatel_gnss850',
            'imu': 'oxts_rt3000_v4'}.get(
        kind, 'ouster_os1_128')


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


def in_field(s, model: SensorModel, P, rcs: float | None = None, zone_scale=None) -> np.ndarray:
    """Whether each vehicle-frame point (K, 3) lies in the sensor's field and range (no occlusion test). A radar takes
    the target's cross-section (m², default the car the datasheet range is given for) and optionally a range factor
    per zone (zone_scale: what it looks through, core.materials.radar_range_factors; None = the datasheet's)."""
    r, az, el = angles(s, P)
    if not model.has_field:
        return np.zeros(len(r), bool)
    if model.is_lidar:
        lo, hi = model.v_fov()
        half = model.az_half()
        ok = (el >= lo - 1e-9) & (el <= hi + 1e-9) & (r >= model.range_min_m) & (r <= model.range_m)
        return ok if half >= 180.0 else ok & (np.abs(az) <= half + 1e-9)
    scale = radar_scale(rcs)
    ok = np.zeros(len(r), bool)
    for k, (rng, ah, eh) in enumerate(model.zones):
        f = zone_scale[k] if zone_scale is not None and k < len(zone_scale) else 1.0
        ok |= (np.abs(az) <= ah) & (np.abs(el) <= eh) & (r <= rng * scale * f) & (r >= model.range_min_m)
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


def radar_wedges(s, model: SensorModel, rcs: float | None = None, samples: int = 24, zone_scale=None) -> list:
    """Each radar zone's field in plan as a closed polygon (N, 2) in the vehicle frame: apex at the sensor, an arc at
    the zone's range (for the target's cross-section, × zone_scale's factor when given) about the sensor's heading."""
    out = []
    k = radar_scale(rcs)
    R = rotation(s)
    fwd = R[:, 0]
    head = math.degrees(math.atan2(fwd[1], fwd[0]))
    for i, (rng, ah, _) in enumerate(model.zones):
        a = np.radians(head + np.linspace(-ah, ah, samples))
        r = rng * k * (zone_scale[i] if zone_scale is not None and i < len(zone_scale) else 1.0)
        arc = np.c_[s.x + r * np.cos(a), s.y + r * np.sin(a)]
        out.append(np.vstack([[s.x, s.y], arc]))
    return out


def heading(s) -> float:
    """Direction of the sensor's +x in plan, degrees counter-clockwise from the vehicle's +x."""
    f = rotation(s)[:, 0]
    return math.degrees(math.atan2(f[1], f[0]))


# ======================================================================= body
def body_box(s, model: SensorModel):
    """(centre (3,), half sizes (3,), rotation (3, 3), round) of the sensor housing: a cylinder for a spinning lidar or
    a GNSS antenna (round = True, half sizes radius, radius, half height), a box otherwise; the optical centre (an
    antenna's phase centre, an IMU's axis origin) sits optical_z_mm above the base."""
    dx, dy, h = (max(float(v), 1.0) / 1000.0 for v in model.size_mm)
    R = rotation(s)
    base_to_centre = h / 2 - model.optical_z_mm / 1000.0
    c = np.array([s.x, s.y, s.z]) + R @ np.array([0.0, 0.0, base_to_centre])
    return c, np.array([dx / 2, dy / 2, h / 2]), R, model.spinning or model.scan == 'nonrepetitive' or model.kind == 'gnss'


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
TAG_STEM = {'lidar': 'L', 'radar': 'RD', 'gnss': 'GN', 'imu': 'IMU'}


def next_tag(kind: str, taken) -> str:
    """'L1', 'L2' … for lidars, 'RD1' … for radars, 'GN1' … for GNSS antennas, 'IMU1' … for IMUs, skipping every tag
    already used by a camera or sensor."""
    taken = {t.upper() for t in taken}
    stem = TAG_STEM.get(kind, 'S')
    i = 1
    while f'{stem}{i}' in taken:
        i += 1
    return f'{stem}{i}'


def surface_top(x: float, y: float, mesh=None, rig=()) -> float:
    """Height of the highest surface at (x, y) seen from above: the body (a ray straight down) or the top of a visible
    rig part over that point."""
    from . import rig as rigm
    top = 0.0
    if mesh is not None and mesh.n_tris:
        t, f = mesh.raycast(np.array([[x, y, float(mesh.bbox[1][2]) + 1.0]]), np.array([[0.0, 0.0, -1.0]]))
        if f[0] >= 0:
            top = float(mesh.bbox[1][2]) + 1.0 - float(t[0])
    for p in rig:
        if not p.visible:
            continue
        poly = rigm.footprint(p)
        if _in_poly(x, y, poly):
            top = max(top, p.z + (p.radius if p.kind == 'bar' else p.height / 2))
    return top


def _in_poly(x, y, poly) -> bool:
    inside = False
    n = len(poly)
    for i in range(n):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


ANTENNA_CLEAR_M = 0.55          # a new GNSS antenna keeps this far from a lidar or mast (near-field multipath)


def _antenna_spot(roof, antennas, lidars, rig) -> tuple[float, float]:
    """Where the next GNSS antenna goes on the roof: the first on the centreline near the rear edge, the second near
    the front edge (the longest heading baseline), a third in between; each spot keeps ANTENNA_CLEAR_M from every
    lidar and upright rig part (moving sideways when the centreline is taken) and from the antennas already placed."""
    xf, xr, hw, _ = roof
    inset = min(0.05, max(xf - xr, 0.0) / 4)
    xs = np.linspace(xr + inset, xf - inset, 41)
    ys = [0.0] + [k * f * hw for f in (0.3, 0.55, 0.8) for k in (1, -1)]
    posts = [(s.x, s.y, 0.05) for s in lidars] + [(p.x, p.y, p.radius) for p in rig
                                                    if p.visible and p.kind == 'cylinder']
    near = [(a.x, a.y, 0.0) for a in antennas]

    def clear(x, y):
        return all(math.hypot(x - px, y - py) - r >= ANTENNA_CLEAR_M for px, py, r in posts + near)
    n = len(antennas)
    order = (lambda c: (abs(c[1]), c[0])) if n == 0 else (lambda c: (abs(c[1]), -c[0])) if n == 1 else \
        (lambda c: (abs(c[1]), abs(c[0] - (xf + xr) / 2)))
    cands = sorted(((float(x), float(y)) for x in xs for y in ys), key=order)
    return next((c for c in cands if clear(*c)), cands[0])


def default_pose(kind: str, model: SensorModel, mesh=None, rig=(), others=()) -> dict:
    """A sensible first pose: a lidar on top of the highest mountable rig part (a mast head, an enclosure) or on the
    roof; a radar in the middle of the front bumper, facing forward at 0.5 m; a GNSS antenna on the roof (or the rack
    over it) near its rear edge, the second one near its front edge for the longest heading baseline, both clear of
    the lidars and masts (_antenna_spot); an IMU on the
    cabin floor above the rear axle (the vehicle origin), so its lever arm to the vehicle frame is vertical only.
    `others` are the variant's sensors already placed."""
    from . import rig as rigm
    oz = model.optical_z_mm / 1000.0
    if kind == 'gnss':
        x, y = _antenna_spot(rigm.roof(mesh), [s for s in others if s.kind == 'gnss'],
                             [s for s in others if s.kind == 'lidar'], rig)
        return {'x': round(x, 3), 'y': y, 'z': round(surface_top(x, y, mesh, rig) + oz, 3), 'yaw': 0.0, 'pitch': 0.0,
                'roll': 0.0}
    if kind == 'imu':
        return {'x': 0.0, 'y': 0.0, 'z': 0.45, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0}
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
