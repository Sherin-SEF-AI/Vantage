"""Custom hardware: the specification sheet of every piece of kit Vantage plans with, and the figures it actually reads.

A field is listed here only when some part of Vantage reads it (`used_by` names who): a camera module's pixel pitch
drives every density map, its width × height × frame rate the link budget and the recording budget, its power the power
budget; a lidar's beam angles drive its ground rings and coverage, its point rate and bytes per point the recording
budget; a radar's zones its coverage; an INS's heading figures the pose-error budget; a recorder's sustained write the
recording verdict; a switch's sync interval and residence jitter the clock tree; a DC-DC converter's rating and
efficiency the power budget. A figure left empty is "not stated" (0 or '' in the app object), exactly as the built-in
library does for its products, and the studies say so rather than invent one.

Kinds:

- sensors and cameras (they go into the project's own library, never the built-in one): ``camera_module``, ``lens``,
  ``lidar``, ``radar``, ``gnss`` (antenna), ``ins`` (GNSS/INS with antenna inputs), ``imu``;
- rig hardware (one selected item per role drives the studies; ``project.hardware``): ``compute``, ``recorder``,
  ``switch``, ``time_server``, ``converter``.

Selecting a rig item writes its figures into the settings the studies already read (``project.ecu``,
``analysis.power``, ``analysis.recording``), so every study, export and the requirements scorecard use them; the
figures it replaced are kept and put back when the selection is cleared. The clock tree takes the switch's and the
time server's timing figures as its typical values, sourced to the item. With nothing selected everything behaves as
before.
"""
from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, field

SCHEMA = 'vantage.hardware/1'
SENSOR_KINDS = ('camera_module', 'lens', 'lidar', 'radar', 'gnss', 'ins', 'imu')
RIG_ROLES = ('compute', 'recorder', 'switch', 'time_server', 'converter')
KIND_LABEL = {'camera_module': 'Camera module', 'lens': 'Lens', 'lidar': 'Lidar', 'radar': 'Radar',
              'gnss': 'GNSS antenna', 'ins': 'GNSS/INS', 'imu': 'IMU', 'compute': 'Compute unit',
              'recorder': 'Recorder', 'switch': 'Network switch', 'time_server': 'Time server / GNSS timing receiver',
              'converter': 'DC-DC converter / power supply'}


@dataclass(frozen=True)
class Field:
    key: str
    label: str
    unit: str = ''
    type: str = 'float'            # float | int | str | bool | choice | vec2 | vec3 | list | zones
    required: bool = False
    lo: float | None = None
    hi: float | None = None
    default: object = None         # None = not stated (the app object's own "not given" value)
    used_by: tuple = ()
    help: str = ''
    choices: tuple = ()


def F(key, label, unit='', type='float', required=False, lo=None, hi=None, default=None, used_by=(), help='',
      choices=()):
    return Field(key, label, unit, type, required, lo, hi, default, tuple(used_by), help, tuple(choices))


_ID = F('id', 'Id', type='str', help='A short unique id (letters, digits, _); made from the name when empty')
_NAME = F('name', 'Name', type='str', required=True, used_by=('library', 'reports', 'exports'),
          help='The product name as the maker writes it')
_VENDOR = F('vendor', 'Maker', type='str', used_by=('library', 'reports'))
_URL = F('url', 'Source', type='str', used_by=('library',), help='The datasheet or product page the figures come from')
_NOTE = F('note', 'Note', type='str', used_by=('library',), help='Where the figures come from and any caveat')
_SIZE = F('size_mm', 'Size', 'mm', 'vec3', used_by=('3D housings', 'housings study', 'brackets', 'harness'),
          help='Depth (x) × width (y) × height; a spinning lidar: diameter × diameter × height')
_OPTZ = F('optical_z_mm', 'Optical centre above the base', 'mm', lo=0, hi=1000,
          used_by=('sensor pose', 'coverage', 'exports'), help='Where the beams or the phase centre start, above the '
                                                               'mounting face')
_POWER = F('power_w', 'Power, steady', 'W', lo=0, hi=2000, used_by=('power budget', 'harness voltage drop'))
_IFACE = F('interface', 'Interface', type='str', used_by=('harness cable type', 'recording link load', 'clock tree'),
           help='e.g. 1000BASE-T1, 100BASE-T1, 1000BASE-T, CAN (500 kbit/s), RS-232, SPI')
_SYNC = F('time_sync', 'Time sync', type='choice', choices=('PTP', 'PPS + NMEA', 'none'), default='none',
          used_by=('sync plan', 'clock tree'), help='How it timestamps: PTP (IEEE 1588 / gPTP), a PPS + NMEA '
                                                     'input, or none (the host stamps it on arrival)')
_RATE = F('rate_hz', 'Output rate', 'Hz', lo=0.1, hi=4000, used_by=('sync plan', 'clock tree', 'drive-log check',
                                                                    'sample dataset'))
_MBPS = F('data_mbps', 'Output data rate', 'Mbit/s', lo=0, hi=10000, used_by=('recording budget', 'link load'),
          help='0 = not stated: the recording budget counts it as 0 and says so')
_IMU_NOISE = (
    F('accel_noise_density', 'Accelerometer noise density', 'm/s²/√Hz', lo=0, hi=1,
      used_by=('Kalibr imu.yaml', 'sample dataset IMU noise'), help='Kalibr accelerometer_noise_density (VRW)'),
    F('accel_random_walk', 'Accelerometer bias random walk', 'm/s³/√Hz', lo=0, hi=1, used_by=('Kalibr imu.yaml',)),
    F('gyro_noise_density', 'Gyro noise density', 'rad/s/√Hz', lo=0, hi=1,
      used_by=('Kalibr imu.yaml', 'sample dataset IMU noise'), help='An ARW in °/√h × π/180 / 60'),
    F('gyro_random_walk', 'Gyro bias random walk', 'rad/s²/√Hz', lo=0, hi=1, used_by=('Kalibr imu.yaml',)),
)

FIELDS = {
    'camera_module': (
        _ID, _NAME, _VENDOR,
        F('sensor', 'Image sensor', type='str', used_by=('library', 'reports'), help='e.g. Sony IMX390'),
        F('width_px', 'Width', 'px', 'int', True, 16, 20000, used_by=('coverage density', 'link budget',
                                                                       'recording budget', 'every export')),
        F('height_px', 'Height', 'px', 'int', True, 16, 20000, used_by=('coverage density', 'link budget',
                                                                         'recording budget', 'every export')),
        F('pixel_um', 'Pixel pitch', 'µm', 'float', True, 0.3, 30, used_by=('every density map', 'lens model',
                                                                            'exports'),
          help='Sensor width = width × pitch: the lens model maps rays to the sensor through it'),
        F('fps', 'Frame rate', 'fps', 'float', True, 0.1, 1000, used_by=('link budget', 'recording budget',
                                                                         'sync plan', 'clock tree')),
        F('shutter', 'Shutter', type='choice', choices=('rolling', 'global', ''), default='',
          used_by=('sync plan (rolling-shutter skew)', 'motion warp')),
        F('interface', 'Interface', type='str', default='GMSL2', used_by=('harness', 'link budget', 'clock tree'),
          help='GMSL2, GMSL3, FPD-Link III/IV, MIPI CSI-2, GigE, USB 3'),
        F('hw_sync', 'Hardware trigger input', type='bool', default=True, used_by=('sync plan',)),
        F('size_mm', 'Size', 'mm', 'vec3', used_by=('3D model', 'housings study', 'brackets', 'mount clearance'),
          help='Width × height (across the optical axis) × length along it'),
        F('weight_g', 'Weight', 'g', lo=0, hi=10000, used_by=('thermal and vibration', 'suspension load')),
        F('cables_m', 'Cable lengths offered', 'm', 'list', used_by=('cable check', 'harness'),
          help='The module cables you can buy, e.g. 3, 15'),
        F('power_w', 'Power', 'W', lo=0, hi=100, used_by=('power budget',),
          help='0 = not stated: the power budget counts 0 W and lists the camera'),
        F('dynamic_range_db', 'Dynamic range', 'dB', lo=0, hi=200, used_by=('exposure study (sun glare)',)),
        F('ip_rating', 'IP rating', type='str', used_by=('library',)),
        F('temp_c', 'Operating temperature', '°C', 'vec2', used_by=('thermal and vibration',)),
        F('chroma', 'Colour', type='choice', choices=('colour', 'monochrome', 'RGB-IR', ''), default='',
          used_by=('sample dataset', 'exports')),
        F('optical_format', 'Optical format', type='str', used_by=('library',)),
        _URL, _NOTE),
    'lens': (
        _ID, _NAME,
        F('module_id', 'Characterised on module', type='str', used_by=('lens model (sensor size)',),
          help='The module id it was measured on; needed to fit its field of view'),
        F('label', 'Label', type='str', used_by=('library',), help='wide / narrow / fisheye'),
        F('focal_mm', 'Focal length', 'mm', lo=0.1, hi=300, used_by=('lens model', 'every density map', 'exports'),
          help='Needed unless the field of view is stated'),
        F('projection', 'Projection', type='choice', default='equisolid',
          choices=('rectilinear', 'equidistant', 'equisolid', 'stereographic', 'orthographic', 'kannala_brandt'),
          used_by=('lens model', 'exports (OpenCV fisheye / pinhole)')),
        F('kb', 'Kannala–Brandt k1…k4', '', 'list', used_by=('lens model', 'Kalibr / OpenCV fisheye exports'),
          help='With projection kannala_brandt; left empty they are fitted to the stated field of view'),
        F('sheet_fov', 'Stated field of view H, V, D', '°', 'vec3', used_by=('lens fit', 'FOV check'),
          help='The maker\'s horizontal, vertical, diagonal field (0 = not stated)'),
        F('max_half_deg', 'Image-circle half angle', '°', lo=0, hi=135, used_by=('lens model (image circle)',)),
        F('f_number', 'Aperture', 'f/', lo=0, hi=64, used_by=('depth of field', 'exposure')),
        F('focus_near_m', 'Near focus', 'm', lo=0, hi=100, used_by=('in-focus coverage',)),
        F('pp_offset_px', 'Principal point offset', 'px', 'vec2', used_by=('lens model', 'exports')),
        _NOTE),
    'lidar': (
        _ID, _NAME, _VENDOR,
        F('scan', 'Scan pattern', type='choice', choices=('spinning', 'solid', 'nonrepetitive'), default='spinning',
          used_by=('coverage', 'lidar points', 'clock tree (sweep timing)')),
        F('channels', 'Channels', 'beams', 'int', True, 1, 1024, used_by=('ground rings', 'coverage', 'recording')),
        F('v_min', 'Lowest beam', '°', 'float', True, -90, 90, used_by=('ground rings', 'coverage', 'blind volume')),
        F('v_max', 'Highest beam', '°', 'float', True, -90, 90, used_by=('coverage',)),
        F('beams', 'Measured beam elevations', '°', 'list', used_by=('ground rings', 'coverage'),
          help='Each channel\'s elevation (overrides the even spread); as many as there are channels'),
        F('h_fov', 'Horizontal field', '°', 'float', True, 1, 360, default=360.0, used_by=('coverage',)),
        F('h_res', 'Azimuth step', '°', 'float', True, 0.01, 10, used_by=('coverage', 'occupancy density',
                                                                         'fusion density', 'recording')),
        F('rate_hz', 'Rotation / frame rate', 'Hz', 'float', True, 0.1, 100, used_by=('sync plan', 'clock tree',
                                                                                       'recording', 'motion warp')),
        F('range_m', 'Range on a 10 % target', 'm', 'float', True, 1, 2000, used_by=('coverage', 'perception range',
                                                                                    'weather')),
        F('range_max_m', 'Rated maximum range', 'm', lo=0, hi=5000, used_by=('library',)),
        F('range_min_m', 'Minimum range', 'm', lo=0, hi=50, default=0.5, used_by=('coverage (near blind zone)',)),
        F('points_per_s', 'Point rate (single return)', 'pts/s', lo=0, hi=1e8,
          used_by=('recording budget',), help='0 = from channels × 360 / azimuth step × rate'),
        F('bytes_per_point', 'Bytes per point (packets)', 'B', lo=0, hi=256, default=12.0,
          used_by=('recording budget',)),
        F('rotation', 'Rotation seen from above', type='choice', choices=('cw', 'ccw'), default='cw',
          used_by=('sync plan', 'timing wedges', 'motion warp')),
        _IFACE, _SYNC, _SIZE, _OPTZ, _POWER, _URL, _NOTE),
    'radar': (
        _ID, _NAME, _VENDOR,
        F('zones', 'Detection zones', '', 'zones', True,
          used_by=('coverage', 'materials (fascia loss)', 'weather', 'suspension'),
          help='[[range on a 10 m² car (m), azimuth half-angle (°), elevation half-angle (°)], …], far zone first'),
        F('range_min_m', 'Minimum range', 'm', lo=0, hi=50, default=0.5, used_by=('coverage',)),
        _RATE, _MBPS, _IFACE, _SYNC, _SIZE, _OPTZ, _POWER, _URL, _NOTE),
    'gnss': (
        _ID, _NAME, _VENDOR,
        F('mask_deg', 'Elevation mask', '°', lo=0, hi=45, default=10.0, used_by=('GNSS sky study',)),
        F('lna_gain_db', 'LNA gain', 'dB', lo=0, hi=60, used_by=('cable harness RF budget',),
          help='0 = not stated: the harness study\'s own setting applies'),
        _RATE, _IFACE, _SIZE, _OPTZ, _POWER, _URL, _NOTE),
    'ins': (
        _ID, _NAME, _VENDOR,
        F('antennas', 'GNSS antenna inputs', '', 'int', True, 1, 2, default=2,
          used_by=('GNSS/IMU study', 'clock tree (PPS source)'), help='1 = position, 2 = position + heading'),
        F('heading_sigma_deg', 'Heading error, 1σ', '°', lo=0, hi=10, used_by=('pose-error budget',)),
        F('heading_baseline_m', 'at antenna baseline', 'm', lo=0.1, hi=10, default=2.0, used_by=('pose-error budget',)),
        F('pos_sigma_cm', 'Position error (RTK), 1σ', 'cm', lo=0, hi=1000, used_by=('pose-error budget',)),
        F('attitude_sigma_deg', 'Roll and pitch error, 1σ', '°', lo=0, hi=10, used_by=('pose-error budget',)),
        *_IMU_NOISE, _RATE, _MBPS, _IFACE, _SYNC, _SIZE, _OPTZ, _POWER, _URL, _NOTE),
    'imu': (
        _ID, _NAME, _VENDOR, *_IMU_NOISE, _RATE, _MBPS, _IFACE, _SYNC, _SIZE, _OPTZ, _POWER, _URL, _NOTE),
    # ---------------------------------------------------------------- rig hardware (project.hardware)
    'compute': (
        _ID, _NAME, _VENDOR,
        F('ports', 'Camera serial links', 'links', 'int', True, 1, 64, default=8,
          used_by=('port checks', 'link budget', 'clock tree (deserializers)'),
          help='GMSL / FPD-Link ports; 4 per deserializer (A = 1–4, B = 5–8, …)'),
        F('link_gbps', 'Forward channel per link', 'Gbit/s', 'float', True, 0.1, 100, default=6.0,
          used_by=('link budget', 'recording link load'), help='GMSL2: 6, GMSL3: 12'),
        F('csi_gbps_per_deser', 'Deserializer CSI-2 output', 'Gbit/s', 'float', True, 0.1, 200, default=10.0,
          used_by=('link budget',), help='e.g. 4 lanes × 2.5 Gbit/s'),
        F('bpp', 'Pixel format on the link', 'bit/px', 'int', False, 1, 64, default=16,
          used_by=('link budget', 'recording budget (raw)'), help='YUV 4:2:2 = 16, RAW12 = 12'),
        F('overhead', 'Link overhead', '×', lo=1.0, hi=3.0, default=1.10, used_by=('link budget',),
          help='Blanking and protocol on top of the pixel payload'),
        F('power_w', 'Power (top power mode)', 'W', 'float', True, 0, 2000, used_by=('power budget',)),
        F('location', 'Where it is mounted', type='str', used_by=('reports', 'clock tree')),
        F('x', 'Mount x', 'm', lo=-20, hi=40, used_by=('cable routing', 'harness', 'power cable drop')),
        F('y', 'Mount y', 'm', lo=-5, hi=5, used_by=('cable routing', 'harness')),
        F('z', 'Mount z', 'm', lo=-1, hi=5, used_by=('cable routing', 'harness')),
        F('route_factor', 'Routing slack', '×', lo=1.0, hi=3.0, default=1.08, used_by=('cable lengths',)),
        F('service_loop_m', 'Service loop per cable', 'm', lo=0, hi=5, default=0.25, used_by=('cable lengths',)),
        _URL, _NOTE),
    'recorder': (
        _ID, _NAME, _VENDOR,
        F('write_mb_s', 'Sustained write', 'MB/s', 'float', True, 1, 1e6,
          used_by=('recording budget verdict',), help='After any SLC cache is full: the figure that matters'),
        F('storage_tb', 'Capacity of one disk set', 'TB', 'float', True, 0.01, 10000,
          used_by=('recording budget (disk sets per shift)',)),
        F('power_w', 'Power', 'W', 'float', True, 0, 2000, used_by=('power budget',)),
        F('interface', 'Interfaces', type='str', used_by=('reports',), help='e.g. 2 × 10GBASE-T, PCIe Gen4 x8'),
        _URL, _NOTE),
    'switch': (
        _ID, _NAME, _VENDOR,
        F('ports', 'Ports', '', 'int', False, 1, 256, used_by=('reports',)),
        F('port_gbps', 'Port speed', 'Gbit/s', lo=0.01, hi=400, used_by=('reports',)),
        F('uplink_gbps', 'Uplink to the recorder', 'Gbit/s', 'float', True, 0.01, 1000,
          used_by=('recording budget (uplink load)',)),
        F('power_w', 'Power', 'W', 'float', True, 0, 2000, used_by=('power budget', 'clock tree (a switch in the '
                                                                                    'tree when > 0)')),
        F('ptp', 'gPTP / 802.1AS time-aware bridge', type='bool', default=True, used_by=('clock tree',)),
        F('sync_ms', 'Sync interval', 'ms', lo=1, hi=4000, default=125.0, used_by=('clock tree',),
          help='IEEE 802.1AS default 125 ms (logSyncInterval −3)'),
        F('jitter_ns', 'Residence-time jitter, 1σ', 'ns', lo=0, hi=1e6, used_by=('clock tree',)),
        F('asym_ns', 'Link asymmetry', 'ns', lo=-1e6, hi=1e6, used_by=('clock tree',)),
        F('delay_ns', 'Path delay', 'ns', lo=0, hi=1e7, used_by=('clock tree',)),
        _URL, _NOTE),
    'time_server': (
        _ID, _NAME, _VENDOR,
        F('power_w', 'Power', 'W', 'float', True, 0, 500, used_by=('power budget', 'clock tree (a grandmaster in the '
                                                                                   'tree when > 0)')),
        F('ppm', 'Holdover oscillator', 'ppm', lo=0, hi=1000, used_by=('clock tree (holdover)',),
          help='TCXO ≈ 0.5, OCXO ≈ 0.01'),
        F('aging_ppb_day', 'Oscillator ageing', 'ppb/day', lo=0, hi=1e5, used_by=('clock tree (holdover)',)),
        F('servo_hz', 'PPS discipline bandwidth', 'Hz', lo=0.001, hi=10, used_by=('clock tree',)),
        F('pps_ns', 'PPS error while tracking, 1σ', 'ns', lo=0, hi=1e6, used_by=('clock tree',)),
        F('priority', 'gPTP priority1', '', lo=0, hi=255, used_by=('clock tree (grandmaster election)',)),
        _URL, _NOTE),
    'converter': (
        _ID, _NAME, _VENDOR,
        F('rated_w', 'Rated output', 'W', 'float', True, 1, 100000, used_by=('power budget (converter load)',)),
        F('efficiency', 'Efficiency', '', 'float', True, 0.3, 1.0, used_by=('power budget (from the vehicle, heat)',)),
        F('output_v', 'Output (rig bus)', 'V', 'float', True, 3, 1000, default=12.0,
          used_by=('power budget (currents, cable drop, fuses)',)),
        F('input_min_v', 'Input range, min', 'V', lo=0, hi=1000, used_by=('reports',)),
        F('input_max_v', 'Input range, max', 'V', lo=0, hi=1000, used_by=('reports',)),
        F('in_enclosure', 'Inside the enclosure', type='bool', default=True, used_by=('thermal (enclosure heat)',)),
        _URL, _NOTE),
}

# rig role -> {spec key: (settings section, settings key)}: where a selection's figures go
_APPLY = {
    'compute': {'name': ('ecu', 'name'), 'ports': ('ecu', 'ports'), 'link_gbps': ('ecu', 'link_gbps'),
                'csi_gbps_per_deser': ('ecu', 'csi_gbps_per_deser'), 'bpp': ('ecu', 'bpp'),
                'overhead': ('ecu', 'overhead'), 'location': ('ecu', 'location'), 'x': ('ecu', 'x'),
                'y': ('ecu', 'y'), 'z': ('ecu', 'z'), 'route_factor': ('ecu', 'route_factor'),
                'service_loop_m': ('ecu', 'service_loop_m'), 'power_w': ('power', 'compute_w')},
    'recorder': {'write_mb_s': ('recording', 'write_mb_s'), 'storage_tb': ('recording', 'storage_tb'),
                 'power_w': ('power', 'recorder_w')},
    'switch': {'uplink_gbps': ('recording', 'uplink_gbps'), 'power_w': ('power', 'switch_w')},
    'time_server': {'power_w': ('power', 'ptp_w')},
    'converter': {'rated_w': ('power', 'dcdc_w'), 'efficiency': ('power', 'dcdc_eff'), 'output_v': ('power', 'bus_v'),
                  'in_enclosure': ('power', 'dcdc_in_enclosure')},
}
# rig role -> clock-tree node kind and the spec keys that become its typical values
_CLOCK = {'switch': ('switch', ('sync_ms', 'jitter_ns', 'asym_ns', 'delay_ns')),
          'time_server': ('server', ('ppm', 'aging_ppb_day', 'servo_hz', 'priority'))}


def kinds() -> tuple:
    return SENSOR_KINDS + RIG_ROLES


def fields(kind) -> tuple:
    if kind not in FIELDS:
        raise KeyError(f'unknown hardware kind {kind!r}: one of {", ".join(kinds())}')
    return FIELDS[kind]


def required(kind) -> list:
    return [f.key for f in fields(kind) if f.required]


def template(kind) -> dict:
    """A fillable spec sheet: every field at its default (None = not stated), with a sibling help map giving each
    field's label, unit, range, whether it is required and who reads it."""
    spec = {f.key: (list(f.default) if isinstance(f.default, tuple) else f.default) for f in fields(kind)}
    help_ = {f.key: {'label': f.label, 'unit': f.unit, 'type': f.type, 'required': f.required,
                     'range': [f.lo, f.hi] if f.lo is not None or f.hi is not None else None,
                     'choices': list(f.choices) or None, 'used_by': list(f.used_by), 'help': f.help}
             for f in fields(kind)}
    return {'schema': SCHEMA, 'kind': kind, 'spec': spec, 'help': help_}


def _empty(v) -> bool:
    return v is None or v == '' or (isinstance(v, (list, tuple)) and not len(v))


def _num(v):
    if isinstance(v, bool):
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _vec(v, n):
    if not isinstance(v, (list, tuple)) or len(v) != n:
        return None
    out = [_num(x) for x in v]
    return None if any(x is None for x in out) else out


def unwrap(spec) -> tuple:
    """(kind or None, spec dict) of a spec sheet as written by template(), or a bare spec dict."""
    if isinstance(spec, dict) and 'spec' in spec and isinstance(spec['spec'], dict):
        return spec.get('kind'), dict(spec['spec'])
    return None, dict(spec or {})


def validate(kind, spec) -> tuple[list, list]:
    """(errors, warnings) of a spec: missing required figures, wrong types, out-of-range values, and consistency
    (lidar point rate against its scan figures, radar zones, a lens's focal length or field, a camera's sensor size
    against its optical format, a recorder's capacity against its write rate, a converter's efficiency)."""
    _, spec = unwrap(spec)
    errors, warns = [], []
    for f in fields(kind):
        v = spec.get(f.key)
        if _empty(v):
            if f.required:
                errors.append(f'{f.label}: required')
            continue
        if f.type in ('float', 'int'):
            x = _num(v)
            if x is None:
                errors.append(f'{f.label}: a number is needed')
                continue
            if f.type == 'int' and abs(x - round(x)) > 1e-9:
                errors.append(f'{f.label}: a whole number is needed')
            if (f.lo is not None and x < f.lo) or (f.hi is not None and x > f.hi):
                errors.append(f'{f.label}: {x:g} {f.unit} is outside {f.lo:g}…{f.hi:g}')
        elif f.type == 'bool' and not isinstance(v, bool):
            errors.append(f'{f.label}: true or false')
        elif f.type == 'choice' and v not in f.choices:
            errors.append(f'{f.label}: one of {", ".join(c or "(not stated)" for c in f.choices)}')
        elif f.type in ('vec2', 'vec3') and _vec(v, 2 if f.type == 'vec2' else 3) is None:
            errors.append(f'{f.label}: {2 if f.type == "vec2" else 3} numbers')
        elif f.type == 'list' and (not isinstance(v, (list, tuple)) or any(_num(x) is None for x in v)):
            errors.append(f'{f.label}: a list of numbers')
        elif f.type == 'zones':
            ok = isinstance(v, (list, tuple)) and all(_vec(z, 3) is not None for z in v)
            if not ok:
                errors.append(f'{f.label}: [[range m, azimuth half °, elevation half °], …]')
            else:
                for i, (r, az, el) in enumerate(v):
                    if not (0 < r <= 2000 and 0 < az <= 180 and 0 < el <= 90):
                        errors.append(f'{f.label} {i + 1}: range 0–2000 m, azimuth half 0–180°, elevation half 0–90°')
                rs = [float(z[0]) for z in v]
                if rs != sorted(rs, reverse=True):
                    warns.append(f'{f.label}: the far zone usually comes first')
    if errors:
        return errors, warns
    g = lambda k, d=0.0: _num(spec.get(k)) if _num(spec.get(k)) is not None else d      # noqa: E731
    if kind == 'lidar':
        if g('v_min') >= g('v_max'):
            errors.append('Lowest beam must be below the highest')
        beams = spec.get('beams') or []
        if beams and len(beams) != int(g('channels')):
            errors.append(f'Measured beam elevations: {len(beams)} given for {int(g("channels"))} channels')
        if beams and (min(beams) < g('v_min') - 0.5 or max(beams) > g('v_max') + 0.5):
            warns.append('Measured beams reach beyond the lowest/highest beam')
        if g('points_per_s') > 0 and spec.get('scan', 'spinning') == 'spinning':
            exp = g('channels') * g('h_fov', 360.0) / max(g('h_res'), 1e-9) * g('rate_hz')
            if exp > 0 and not 0.45 <= g('points_per_s') / exp <= 2.2:
                warns.append(f'Point rate {g("points_per_s"):,.0f}/s against channels × field / step × rate = '
                             f'{exp:,.0f}/s: check the azimuth step or the return mode')
        if g('range_max_m') and g('range_max_m') < g('range_m'):
            warns.append('Rated maximum range is below the 10 % range')
    elif kind == 'camera_module':
        w_mm = g('width_px') * g('pixel_um') / 1000
        h_mm = g('height_px') * g('pixel_um') / 1000
        of = str(spec.get('optical_format') or '')
        frac = _optical_inch(of)
        if frac:
            diag = math.hypot(w_mm, h_mm)
            nominal = frac * 16.0                         # the "1-inch = 16 mm diagonal" convention
            if not 0.6 <= diag / nominal <= 1.4:
                warns.append(f'Pixel pitch × resolution gives a {diag:.2f} mm diagonal; a {of} sensor is about '
                             f'{nominal:.1f} mm')
        if g('power_w') == 0:
            warns.append('Power not stated: the power budget counts 0 W for it')
        if not spec.get('shutter'):
            warns.append('Shutter not stated: the sync plan treats it as rolling')
    elif kind == 'lens':
        fov = _vec(spec.get('sheet_fov'), 3) or [0, 0, 0]
        if g('focal_mm') <= 0 and not any(fov):
            errors.append('A focal length or a stated field of view is needed')
        if spec.get('projection') == 'kannala_brandt' and _empty(spec.get('kb')) and not any(fov):
            errors.append('Kannala–Brandt needs its k1…k4, or a stated field of view to fit them to')
        if spec.get('projection') == 'kannala_brandt' and _empty(spec.get('kb')) and not spec.get('module_id'):
            errors.append('Fitting k1…k4 to the field of view needs the module it was characterised on')
        if not _empty(spec.get('kb')) and len(spec.get('kb')) != 4:
            errors.append('Kannala–Brandt: four terms k1…k4')
        if spec.get('projection') == 'rectilinear' and max(fov) >= 160:
            errors.append('A rectilinear lens cannot see 160° or more')
    elif kind == 'radar':
        if g('data_mbps') == 0:
            warns.append('Output data rate not stated: the recording budget counts 0')
    elif kind in ('ins', 'imu'):
        noise = [g(k) for k in ('accel_noise_density', 'accel_random_walk', 'gyro_noise_density', 'gyro_random_walk')]
        if 0 < sum(1 for x in noise if x > 0) < 4:
            warns.append('Only some IMU noise figures: Kalibr\'s example fills the rest, and imu.yaml says so')
        if kind == 'ins' and int(g('antennas')) == 2 and g('heading_sigma_deg') == 0:
            warns.append('Dual-antenna heading accuracy not stated')
    elif kind == 'recorder':
        hours = g('storage_tb') * 1e6 / max(g('write_mb_s'), 1e-9) / 3600
        if hours < 0.25:
            warns.append(f'At full write speed it fills in {hours * 60:.0f} minutes')
    elif kind == 'converter':
        if g('input_min_v') and g('input_max_v') and g('input_min_v') >= g('input_max_v'):
            errors.append('Input range: min must be below max')
        if g('efficiency') > 0.99:
            warns.append('An efficiency above 99 % is unusual for a DC-DC converter')
    elif kind == 'switch':
        if spec.get('ptp') is False:
            warns.append('Not a time-aware bridge: PTP across it is not modelled as gPTP')
    return errors, warns


def _optical_inch(s: str) -> float:
    """'1/2.7"' -> 1/2.7; '2/3"' -> 2/3; 0 when not an optical format."""
    import re
    m = re.match(r'\s*(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*(?:"|in|inch|″)', s)
    if m:
        return float(m.group(1)) / float(m.group(2))
    return 0.0


def _slug(s):
    import re
    t = re.sub(r'[^a-z0-9]+', '_', (s or '').lower()).strip('_')
    return t or 'custom'


def _clean(kind, spec) -> dict:
    """The spec's stated values converted to the app's types; not-stated fields left out."""
    out = {}
    for f in fields(kind):
        v = spec.get(f.key)
        if _empty(v):
            continue
        if f.type == 'float':
            v = float(v)
        elif f.type == 'int':
            v = int(round(float(v)))
        elif f.type in ('vec2', 'vec3', 'list'):
            v = tuple(float(x) for x in v)
        elif f.type == 'zones':
            v = [[float(x) for x in z] for z in v]
        out[f.key] = v
    return out


def from_spec(kind, spec):
    """The app object of a valid spec: lens.Module / lens.Lens, sensors.SensorModel, or a rig item dict."""
    k2, spec = unwrap(spec)
    kind = kind or k2
    errs, _ = validate(kind, spec)
    if errs:
        raise ValueError('; '.join(errs))
    d = _clean(kind, spec)
    d['id'] = d.get('id') or _slug(d.get('name'))
    if kind == 'camera_module':
        from .lens import Module
        if 'size_mm' not in d:
            d['size_mm'] = (0.0, 0.0, 0.0)
        return Module.from_dict(dict(d, builtin=False))
    if kind == 'lens':
        return _lens(d, spec)
    if kind in ('lidar', 'radar', 'gnss', 'ins', 'imu'):
        from .sensors import SensorModel
        sk = {'ins': 'imu'}.get(kind, kind)
        if kind == 'imu':
            d['antennas'] = 0
        if kind == 'radar':
            d.setdefault('scan', 'fixed')
            d['range_m'] = float(d['zones'][0][0])
        if 'size_mm' not in d:
            d['size_mm'] = (0.0, 0.0, 0.0)
        d.setdefault('time_sync', 'none')
        return SensorModel.from_dict(dict(d, kind=sk, builtin=False))
    return dict(d, kind=kind)


def _lens(d, spec):
    from . import lens as L
    fov = tuple(d.get('sheet_fov', (0.0, 0.0, 0.0)))
    proj = d.get('projection', 'equisolid')
    if proj == 'kannala_brandt' and 'kb' not in d:
        m = _module_for(spec.get('module_id'))
        sw, sh = m.width_px * m.pixel_um / 1000, m.height_px * m.pixel_um / 1000
        half = (sw / 2, sh / 2, math.hypot(sw, sh) / 2)
        base, f_fit, _ = L.best_base(fov, half)
        f = d.get('focal_mm') or f_fit
        d['kb'] = tuple(float(x) for x in L.fit_sheet(f, fov, half, base=base))
        d['focal_mm'] = f
        d['note'] = (d.get('note', '') + f' Kannala–Brandt terms fitted to the stated field of view on '
                     f'{m.name} (base {base}).').strip()
    elif 'focal_mm' not in d and any(fov):
        m = _module_for(spec.get('module_id'))
        sw, sh = m.width_px * m.pixel_um / 1000, m.height_px * m.pixel_um / 1000
        d['focal_mm'] = float(L.focal_from_fov(fov, (sw / 2, sh / 2, math.hypot(sw, sh) / 2), proj))
    return L.Lens.from_dict(dict(d, builtin=False))


_MODULES_EXTRA = {}


def _module_for(mid):
    from .lens import builtin_modules
    if mid in _MODULES_EXTRA:
        return _MODULES_EXTRA[mid]
    for m in builtin_modules():
        if m.id == mid:
            return m
    raise ValueError(f'module {mid!r} is not in the library or the project')


def to_spec(obj, kind=None) -> dict:
    """The spec dict of an app object (Module, Lens, SensorModel or a rig item dict), keyed by the registry's fields."""
    from .lens import Lens, Module
    from .sensors import SensorModel
    if isinstance(obj, Module):
        kind, d = 'camera_module', obj.to_dict()
    elif isinstance(obj, Lens):
        kind, d = 'lens', obj.to_dict()
    elif isinstance(obj, SensorModel):
        d = obj.to_dict()
        kind = kind or (obj.kind if obj.kind != 'imu' else ('ins' if obj.antennas > 0 else 'imu'))
    else:
        d = dict(obj)
        kind = kind or d.get('kind')
    out = {}
    for f in fields(kind):
        v = d.get(f.key)
        if isinstance(v, tuple):
            v = list(v)
        if isinstance(v, float) and math.isinf(v):
            v = None
        out[f.key] = v
    return out


# ---------------------------------------------------------------- the project
def store(project) -> dict:
    """project.hardware: {'items': {role: {id: spec}}, 'selected': {role: id}, 'replaced': {role: {section.key: old}}}"""
    h = getattr(project, 'hardware', None)
    if not isinstance(h, dict):
        h = {}
        project.hardware = h
    h.setdefault('items', {})
    h.setdefault('selected', {})
    h.setdefault('replaced', {})
    return h


def add_to_project(project, kind, spec, select=True) -> str:
    """Add a custom item: a sensor model into project.sensor_models, a camera module or lens into the project's
    library, a rig item into project.hardware (and select it for its role, which applies its figures). Returns its
    id (made unique)."""
    k2, raw = unwrap(spec)
    kind = kind or k2
    obj = from_spec(kind, raw)
    if kind == 'camera_module':
        obj.id = _unique(obj.id, project.modules)
        project.modules[obj.id] = obj
        _MODULES_EXTRA[obj.id] = obj
        return obj.id
    if kind == 'lens':
        if obj.module_id and obj.module_id in project.modules:
            _MODULES_EXTRA[obj.module_id] = project.modules[obj.module_id]
        obj.id = _unique(obj.id, project.lenses)
        project.lenses[obj.id] = obj
        return obj.id
    if kind in ('lidar', 'radar', 'gnss', 'ins', 'imu'):
        obj.id = _unique(obj.id, project.sensor_models)
        project.sensor_models[obj.id] = obj
        return obj.id
    h = store(project)
    items = h['items'].setdefault(kind, {})
    iid = _unique(obj['id'], items)
    obj['id'] = iid
    items[iid] = obj
    if select:
        select_item(project, kind, iid)
    return iid


def _unique(iid, taken):
    base, n, out = iid, 2, iid
    while out in taken:
        out = f'{base}_{n}'
        n += 1
    return out


def selected(project, role):
    """The selected rig item of a role (its spec dict), or None."""
    h = getattr(project, 'hardware', None) or {}
    iid = (h.get('selected') or {}).get(role)
    return ((h.get('items') or {}).get(role) or {}).get(iid) if iid else None


def select_item(project, role, iid):
    """Select a rig item for its role (None clears it): its figures go into the settings the studies read; what they
    replaced is kept and put back when the selection is cleared or changed."""
    if role not in RIG_ROLES:
        raise KeyError(role)
    h = store(project)
    _restore(project, role)
    if iid is None:
        h['selected'].pop(role, None)
        return
    item = h['items'].get(role, {}).get(iid)
    if item is None:
        raise KeyError(iid)
    h['selected'][role] = iid
    old = {}
    for key, (sec, sk) in _APPLY.get(role, {}).items():
        if key not in item or _empty(item[key]):
            continue
        old[f'{sec}.{sk}'] = _get(project, sec, sk)
        _set(project, sec, sk, item[key])
    h['replaced'][role] = old


def _restore(project, role):
    h = store(project)
    for path, v in (h['replaced'].pop(role, None) or {}).items():
        sec, sk = path.split('.', 1)
        _set(project, sec, sk, v, remove=isinstance(v, str) and v == _MISSING)


_MISSING = '__not_set__'          # the settings key was absent before the selection: removed again on restore


def _get(project, sec, key):
    if sec == 'ecu':
        return getattr(project.ecu, key)
    d = getattr(project.analysis, sec, None) or {}
    return copy.deepcopy(d[key]) if key in d else _MISSING


def _set(project, sec, key, v, remove=False):
    if sec == 'ecu':
        setattr(project.ecu, key, type(getattr(project.ecu, key))(v) if not isinstance(v, str) else v)
        return
    d = dict(getattr(project.analysis, sec, None) or {})
    if remove:
        d.pop(key, None)
    else:
        d[key] = v
    setattr(project.analysis, sec, d)


def sources(project) -> dict:
    """{role: 'name (custom hardware)'} of the selected rig items: what the studies list among their assumptions."""
    out = {}
    for role in RIG_ROLES:
        it = selected(project, role)
        if it is not None:
            out[role] = f'{KIND_LABEL[role]}: {it.get("name")} (project hardware' + \
                        (f', {it["url"]}' if it.get('url') else '') + ')'
    return out


def clock_params(project) -> dict:
    """{clock-tree node kind: {param: (value, source)}} from the selected switch and time server."""
    out = {}
    for role, (node, keys) in _CLOCK.items():
        it = selected(project, role)
        if it is None:
            continue
        src = f'{it.get("name")} (project hardware)'
        vals = {k: (float(it[k]), src) for k in keys if not _empty(it.get(k))}
        if role == 'time_server' and not _empty(it.get('pps_ns')):
            out.setdefault('gnss', {})['pps_ns'] = (float(it['pps_ns']), src)
        if vals:
            out[node] = vals
    return out


def library_items(project) -> list:
    """[(kind, id, name, builtin)] of the project's custom sensors, cameras and rig hardware (for the Library panel)."""
    out = []
    for mid, m in project.modules.items():
        if not m.builtin:
            out.append(('camera_module', mid, m.name, False))
    for lid, ln in project.lenses.items():
        if not ln.builtin:
            out.append(('lens', lid, ln.name, False))
    for sid, s in project.sensor_models.items():
        if not s.builtin:
            k = s.kind if s.kind != 'imu' else ('ins' if s.antennas > 0 else 'imu')
            out.append((k, sid, s.name, False))
    h = getattr(project, 'hardware', None) or {}
    for role in RIG_ROLES:
        for iid, it in ((h.get('items') or {}).get(role) or {}).items():
            out.append((role, iid, it.get('name', iid), False))
    return out


def dump_template(kind) -> str:
    return json.dumps(template(kind), indent=2, ensure_ascii=False)
