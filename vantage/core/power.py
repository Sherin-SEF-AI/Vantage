"""Power budget of the rig: what every device draws, whether the DC-DC converter, the fuses and the power cables
carry it, whether the alternator keeps up while driving, how long the auxiliary battery runs the rig with the engine
off, and how much heat the enclosure must shed.

The supply. The vehicle's 12 V system (the alternator while the engine runs, on an EV its own HV-to-12 V converter;
an auxiliary battery when it does not) feeds one DC-DC converter. Its output is the rig's bus (12, 24 or 48 V), split
into fused circuits that feed the devices.

Loads, steady and peak (every visible device of the variant):
- a camera: its module's power figure, or the typical camera_w (a GMSL camera powered over its coax) when the module
  gives none. The deserializer on the ECU supplies it, so it counts on the compute circuit unless moved.
- a lidar, radar, GNSS antenna or IMU: its library model's power figure (nominal). A GNSS antenna is powered over its
  coax by the receiver; it counts on the GNSS · IMU circuit.
- compute, recorder, network switch and the time server (PTP grandmaster): the settings' figures; other loads (fans,
  displays, a laptop) as the user lists them, each with its own steady and peak power and circuit.
Peak: a lidar spins up at lidar_peak × its steady power for a few seconds (plus a cold-start heater when one is set),
everything else switches on at peak_factor × its steady power; the peak assumes everything switches on at once.

Cables. Each lidar, radar and IMU has its own power cable from the distribution next to the ECU: its length is the
Manhattan route like a camera's (rules.route_length: × route_factor + the service loop) unless set, its gauge
cable_awg unless set. Copper at 20 °C; the drop is over both conductors (there and back). A device draws constant
power (its own regulator), so its current rises as its input voltage falls: V = (Vbus + √(Vbus² − 4·P·R)) / 2, and
when Vbus² < 4·P·R the cable cannot deliver the power at all (the voltage collapses). Cameras, GNSS antennas and the
loads inside the enclosure have no cable modelled.

Fuses. A circuit's steady current should stay within 80 % of its fuse and its peak within the fuse; blade fuses carry
short inrush (they blow within seconds at about twice their rating), so a peak over the fuse is a warning and a peak
over twice the fuse critical. A circuit without a fitted rating gets the smallest standard fuse that satisfies both.

Limits: the wiring is the Manhattan route, connectors and the shared feed to each fuse are not modelled; the DC-DC
efficiency is one figure at every load; the battery's usable energy does not depend on the rate or temperature; the
power over coax converters inside the ECU are part of the camera figure.

Units: W, A, V, Wh, minutes, metres. Cheap (no geometry): computed again on every edit.
"""
from __future__ import annotations

import csv
import io
import json
import math
import os
from dataclasses import asdict, dataclass, field

from . import rules as R

SCHEMA = 'vantage.power_budget/1'
BUS_VOLTAGES = (12.0, 24.0, 48.0)
CIRCUITS = ('compute', 'cameras', 'lidar', 'radar', 'gnss_imu', 'recorder', 'network', 'other')
CIRCUIT_LABEL = {'compute': 'Compute', 'cameras': 'Cameras', 'lidar': 'Lidars', 'radar': 'Radars',
                 'gnss_imu': 'GNSS · IMU', 'recorder': 'Recorder', 'network': 'Network', 'other': 'Other'}
CIRCUIT_NAME = {'compute': 'compute', 'cameras': 'camera', 'lidar': 'lidar', 'radar': 'radar',      # in a sentence
                'gnss_imu': 'GNSS · IMU', 'recorder': 'recorder', 'network': 'network', 'other': 'other loads\''}
KINDS = ('camera', 'lidar', 'radar', 'gnss', 'imu', 'compute', 'recorder', 'switch', 'ptp', 'other')
KIND_LABEL = {'camera': 'Camera', 'lidar': 'Lidar', 'radar': 'Radar', 'gnss': 'GNSS antenna', 'imu': 'IMU',
              'compute': 'Compute', 'recorder': 'Recorder', 'switch': 'Switch', 'ptp': 'Time server',
              'other': 'Other'}
DEFAULT_CIRCUIT = {'camera': 'compute', 'lidar': 'lidar', 'radar': 'radar', 'gnss': 'gnss_imu', 'imu': 'gnss_imu',
                   'compute': 'compute', 'recorder': 'recorder', 'switch': 'network', 'ptp': 'network',
                   'other': 'other'}
# the fixed loads: (key, setting, tag, model)
FIXED = (('compute', 'compute_w', 'Compute', ''), ('recorder', 'recorder_w', 'Recorder', 'Data recorder'),
         ('switch', 'switch_w', 'Switch', 'Network switch'), ('ptp', 'ptp_w', 'PTP', 'Time server · PTP grandmaster'))
CABLED = ('lidar', 'radar', 'imu')         # devices with their own power cable from the distribution
SUPPLY = {'camera': 'over its coax from the deserializer', 'gnss': 'over its coax from the receiver',
          'lidar': 'own power cable', 'radar': 'own power cable', 'imu': 'own power cable',
          'compute': 'at the distribution', 'recorder': 'at the distribution', 'switch': 'at the distribution',
          'ptp': 'at the distribution', 'other': 'at the distribution'}
# standard automotive fuse ratings (blade ATO/ATC to 40 A, MAXI above), A
STANDARD_FUSES_A = (1.0, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0, 60.0, 70.0, 80.0,
                    100.0, 120.0)
CONTINUOUS = 0.8                 # a fuse carries this share of its rating continuously
INRUSH_BLOW = 2.0                # a peak this many times the fuse blows a blade fuse within seconds
RHO_CU = 1.7241e-8               # Ω·m, annealed copper (IACS 100 %) at 20 °C
AWG_RANGE = (4, 28)
SEVERITIES = ('critical', 'serious', 'warning', 'note', 'good')
ISSUE_SEVERITIES = ('critical', 'serious', 'warning')
DEFAULT_ENCLOSURE = ('compute', 'recorder', 'switch', 'ptp')
MAX_OTHER = 40
ASSUMPTIONS = ('Library power figures are nominal (public datasheets, or a typical class), not measured on your '
               'units; a camera module without a figure uses the typical camera power. The peak factors (lidar '
               'spin-up, switch-on of everything else) are typical, the peak assumes everything switches on at once, '
               'and a lidar\'s cold-start heater counts in its peak only. The alternator\'s spare current is your '
               'figure: what the vehicle\'s own loads leave. Cable resistance is copper at 20 °C (about 0.4 % more '
               'per °C warmer); each device draws constant power, so its current rises as its input voltage falls. '
               'The DC-DC efficiency is one figure at every load; the auxiliary battery feeds the converter\'s input '
               'with the engine off, its usable energy independent of rate and temperature.')
# accepted range of every number in the settings (values outside are clamped)
LIMITS = {'camera_w': (0.0, 50.0), 'lidar_peak': (1.0, 10.0), 'peak_factor': (1.0, 10.0),
          'lidar_heater_w': (0.0, 500.0), 'compute_w': (0.0, 5000.0), 'recorder_w': (0.0, 5000.0),
          'switch_w': (0.0, 1000.0), 'ptp_w': (0.0, 500.0), 'dcdc_w': (1.0, 50000.0), 'dcdc_eff': (0.5, 1.0),
          'alternator_a': (0.0, 1000.0), 'alternator_v': (6.0, 60.0), 'battery_wh': (0.0, 100000.0),
          'battery_dod': (0.05, 1.0), 'runtime_target_min': (0.0, 1440.0), 'max_drop_pct': (0.1, 50.0),
          'min_input_v': (0.0, 60.0)}
LOAD_W = (0.0, 5000.0)           # an other load's steady and peak power
CABLE_M = (0.0, 200.0)           # a set cable length
FUSE_A = (0.1, 1000.0)           # a fitted fuse rating


def _num(v, default=0.0) -> float:
    """A figure as a finite, non-negative float (a hand-edited library entry may hold anything)."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) and v >= 0 else default


def _float(v):
    """A finite float, or None."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _clamp(v, lo_hi):
    return min(max(v, lo_hi[0]), lo_hi[1])


def _bool(v, default=False) -> bool:
    if isinstance(v, (bool, int)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in ('1', 'true', 'yes', 'on')
    return default


def _awg(v):
    v = _float(v)
    return None if v is None else int(round(_clamp(v, AWG_RANGE)))


@dataclass
class PowerSettings:
    bus_v: float = 12.0                # the rig's bus: the DC-DC converter's output (12 / 24 / 48 V)
    camera_w: float = 1.5              # a camera whose module gives no figure (GMSL, powered over its coax)
    lidar_peak: float = 2.0            # a lidar's spin-up × its steady power, for a few seconds
    peak_factor: float = 1.2           # everything else at switch-on × its steady power
    lidar_heater_w: float = 0.0        # a lidar's cold-start heater on top of its peak (0 = none)
    compute_w: float = 60.0            # Jetson AGX Orin in its top power mode
    recorder_w: float = 40.0
    switch_w: float = 15.0
    ptp_w: float = 5.0                 # time server / PTP grandmaster
    other: list = field(default_factory=list)       # [[name, steady W, peak W, circuit, in enclosure], ...]
    dcdc_w: float = 600.0              # the converter's rated output
    dcdc_eff: float = 0.92
    dcdc_in_enclosure: bool = True     # its losses heat the enclosure
    alternator_a: float = 50.0         # spare current the vehicle's own loads leave, at alternator_v
    alternator_v: float = 13.8
    battery_wh: float = 1000.0         # auxiliary battery
    battery_dod: float = 0.8           # the share of it that is usable
    runtime_target_min: float = 30.0   # engine off: a parked calibration session
    cable_awg: int = 18
    max_drop_pct: float = 5.0          # a power cable's drop at the steady current, % of the bus
    min_input_v: float = 9.0           # the lowest input voltage the devices accept
    fuses: dict = field(default_factory=dict)       # circuit -> fitted fuse A (absent: sized automatically)
    circuit_of: dict = field(default_factory=dict)  # device key (uid or fixed load) -> circuit (absent: by kind)
    cables: dict = field(default_factory=dict)      # device uid -> {'m': length, 'awg': gauge} (absent: routed)
    enclosure: list = field(default_factory=lambda: list(DEFAULT_ENCLOSURE))   # device keys inside the enclosure

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d: dict | None) -> 'PowerSettings':
        s = PowerSettings()
        d = d if isinstance(d, dict) else {}
        for k, lim in LIMITS.items():
            if k in d:
                v = _float(d[k])
                if v is not None:
                    setattr(s, k, _clamp(v, lim))
        v = _float(d.get('bus_v'))
        if v is not None and v > 0:
            s.bus_v = min(BUS_VOLTAGES, key=lambda b: abs(b - v))
        if 'cable_awg' in d and _awg(d['cable_awg']) is not None:
            s.cable_awg = _awg(d['cable_awg'])
        if 'dcdc_in_enclosure' in d:
            s.dcdc_in_enclosure = _bool(d['dcdc_in_enclosure'], True)
        other = d.get('other')
        for row in (other if isinstance(other, (list, tuple)) else []):
            if len(s.other) >= MAX_OTHER:
                break
            if not isinstance(row, (list, tuple)) or not row:
                continue
            name = str(row[0]).strip()[:60] or 'Load'
            steady = _clamp(_num(row[1] if len(row) > 1 else 0.0), LOAD_W)
            peak = _clamp(_num(row[2] if len(row) > 2 else steady, steady), LOAD_W)
            circ = row[3] if len(row) > 3 and row[3] in CIRCUITS else 'other'
            s.other.append([name, steady, max(peak, steady), circ, _bool(row[4]) if len(row) > 4 else False])
        fuses = d.get('fuses')
        for k, v in (fuses.items() if isinstance(fuses, dict) else ()):
            v = _float(v)
            if k in CIRCUITS and v is not None and v > 0:
                s.fuses[k] = _clamp(v, FUSE_A)
        circ = d.get('circuit_of')
        for k, v in (circ.items() if isinstance(circ, dict) else ()):
            if isinstance(k, str) and v in CIRCUITS:
                s.circuit_of[k] = v
        cables = d.get('cables')
        for k, v in (cables.items() if isinstance(cables, dict) else ()):
            if not isinstance(k, str) or not isinstance(v, dict):
                continue
            c = {}
            m = _float(v.get('m'))
            if m is not None:
                c['m'] = _clamp(m, CABLE_M)
            if _awg(v.get('awg')) is not None:
                c['awg'] = _awg(v['awg'])
            if c:
                s.cables[k] = c
        enc = d.get('enclosure')
        if isinstance(enc, (list, tuple)):
            s.enclosure = list(dict.fromkeys(k for k in enc if isinstance(k, str)))
        return s

    def peak_of(self, kind: str, steady: float) -> float:
        """A device's peak power from its steady power (not for other loads, which give theirs)."""
        if kind == 'lidar':
            return steady * self.lidar_peak + self.lidar_heater_w
        return steady * self.peak_factor


@dataclass
class PowerResult:
    settings: PowerSettings
    rows: list                  # per device: dict (see run)
    circuits: list              # per circuit with a load: dict (see run)
    totals: dict
    findings: list              # [{'severity', 'kind', 'message', 'objects', 'tags', 'uids'}], worst first
    variant: str = ''
    variant_uid: str = ''

    def row(self, tag):
        return next((r for r in self.rows if r['tag'] == tag), None)

    def circuit(self, name):
        return next((c for c in self.circuits if c['circuit'] == name), None)

    def worst(self) -> str:
        return next((f['severity'] for f in self.findings if f['severity'] in ISSUE_SEVERITIES), 'good')


# =========================================================================== electrics
def awg_diameter_mm(awg: float) -> float:
    """Conductor diameter of an AWG gauge (ASTM B258): 0.127 mm × 92^((36 − n) / 39)."""
    return 0.127 * 92.0 ** ((36.0 - awg) / 39.0)


def awg_ohm_per_km(awg: float) -> float:
    """Resistance of one copper conductor of an AWG gauge at 20 °C, Ω/km (18 AWG: 20.95)."""
    area_m2 = math.pi / 4.0 * (awg_diameter_mm(awg) * 1e-3) ** 2
    return RHO_CU / area_m2 * 1000.0


def supply(v_bus: float, p_w: float, r_ohm: float):
    """A constant-power load p_w fed from v_bus through r_ohm (there and back): (voltage at the load, current). The
    voltage is None when the cable cannot deliver the power (v_bus² < 4·p·r); the current is then the one at the
    collapse point (half the bus voltage)."""
    if p_w <= 0:
        return v_bus, 0.0
    if r_ohm <= 0:
        return v_bus, p_w / v_bus
    disc = v_bus * v_bus - 4.0 * p_w * r_ohm
    if disc < 0:
        return None, 2.0 * p_w / v_bus
    v = (v_bus + math.sqrt(disc)) / 2.0
    return v, p_w / v


def auto_fuse(steady_a: float, peak_a: float):
    """The smallest standard fuse with the steady current within 80 % and the peak within the rating; None when even
    the largest is too small."""
    need = max(steady_a / CONTINUOUS, peak_a)
    return next((f for f in STANDARD_FUSES_A if f >= need - 1e-9), None)


def route_length(project, s) -> float:
    """A device's power cable from the distribution next to the ECU: routed along its data cable (core.harness) when
    routing is on, else the Manhattan route like a camera's (× the routing slack, + the service loop), metres."""
    from . import harness
    r = harness.routed_length(project, s)
    if r is not None:
        return r
    e = project.ecu
    d = abs(s.x - e.x) + abs(s.y - e.y) + abs(s.z - e.z)
    d = d if math.isfinite(d) else 0.0
    return float(d * _num(e.route_factor, 1.0) + _num(e.service_loop_m))


def cable_check(st: PowerSettings, steady_w: float, peak_w: float, length_m: float, awg: int) -> dict:
    """The power cable of one device: round-trip resistance, voltage and current at the device at steady and peak
    power, the drops and loss, and its status."""
    r = 2.0 * length_m * awg_ohm_per_km(awg) / 1000.0
    vs, i_s = supply(st.bus_v, steady_w, r)
    vp, i_p = supply(st.bus_v, peak_w, r)
    drop = None if vs is None else st.bus_v - vs
    pdrop = None if vp is None else st.bus_v - vp
    out = {'cable_ohm': r, 'v_steady': vs, 'v_peak': vp, 'steady_a': i_s, 'peak_a': i_p,
           'drop_v': drop, 'drop_pct': None if drop is None else drop / st.bus_v * 100.0,
           'peak_drop_v': pdrop, 'peak_drop_pct': None if pdrop is None else pdrop / st.bus_v * 100.0,
           'loss_w': st.bus_v * i_s - steady_w if vs is not None else None}
    if vs is None or vp is None:
        sev, why = 'serious', 'collapse'
    elif out['drop_pct'] > st.max_drop_pct + 1e-9:
        sev, why = 'serious', 'drop'
    elif vp < st.min_input_v - 1e-9:
        sev, why = 'serious', 'min_input'
    elif out['peak_drop_pct'] > st.max_drop_pct + 1e-9:
        sev, why = 'warning', 'peak_drop'
    else:
        sev, why = 'good', ''
    out['cable_severity'], out['cable_problem'] = sev, why
    return out


def fix_awg(st: PowerSettings, steady_w: float, peak_w: float, length_m: float):
    """The thinnest gauge (largest AWG number) that passes every cable check, or None."""
    for awg in range(AWG_RANGE[1], AWG_RANGE[0] - 1, -1):
        if cable_check(st, steady_w, peak_w, length_m, awg)['cable_severity'] == 'good':
            return awg
    return None


def fuse_severity(steady_a: float, peak_a: float, fuse_a) -> str:
    if not fuse_a:
        return 'critical'
    if steady_a > fuse_a + 1e-9 or peak_a > INRUSH_BLOW * fuse_a + 1e-9:
        return 'critical'
    if peak_a > fuse_a + 1e-9 or steady_a > CONTINUOUS * fuse_a + 1e-9:
        return 'warning'
    return 'good'


# =========================================================================== the budget
def _device(st, key, uid, tag, kind, model, source, steady, peak, slot, circuit=None, enclosure=None):
    return {'key': key, 'uid': uid, 'tag': tag, 'kind': kind, 'model': model, 'source': source,
            'steady_w': float(steady), 'peak_w': float(max(peak, steady)), 'slot': int(slot),
            'circuit': circuit or st.circuit_of.get(key) or DEFAULT_CIRCUIT[kind],
            'enclosure': (key in st.enclosure) if enclosure is None else bool(enclosure), 'supply': SUPPLY[kind],
            'cable_m': None, 'cable_set': False, 'awg': None, 'cable_ohm': None, 'v_steady': st.bus_v,
            'v_peak': st.bus_v, 'steady_a': steady / st.bus_v, 'peak_a': max(peak, steady) / st.bus_v,
            'drop_v': None, 'drop_pct': None, 'peak_drop_v': None, 'peak_drop_pct': None, 'loss_w': None,
            'cable_severity': None, 'cable_problem': ''}


def run(project, engine=None, settings: PowerSettings | None = None, variant=None, progress=None,
        cancel=None) -> PowerResult:
    """The power budget of a variant (default the active one). Rows: every visible camera, lidar, radar, GNSS antenna
    and IMU, then compute, recorder, switch and time server (when not 0 W) and the other loads, each with key (uid or
    fixed key), tag, kind, model, source (module | typical | library | missing | setting | user), steady_w, peak_w,
    circuit, steady_a and peak_a (at the device), enclosure, and for a cabled device cable_m, awg, cable_ohm, v_steady,
    v_peak, drop_v/drop_pct, peak_drop_v/peak_drop_pct, loss_w and cable_severity. Cheap; engine, progress and cancel
    are accepted for the module contract and not needed."""
    st = settings or PowerSettings.from_dict(getattr(project.analysis, 'power', None))
    v = variant or project.variant
    rows = []
    for c in v.cameras:
        if not c.visible:
            continue
        m = project.module_for(c)
        w = _num(m.power_w)
        steady = w if w > 0 else st.camera_w
        rows.append(_device(st, c.uid, c.uid, c.tag, 'camera', m.name, 'module' if w > 0 else 'typical', steady,
                            st.peak_of('camera', steady), c.slot))
    count = {}
    for s in v.sensors:
        if not s.visible:
            continue
        m = project.sensor_model_for(s)
        kind = m.kind
        w = _num(m.power_w)
        r = _device(st, s.uid, s.uid, s.tag, kind, f'{m.vendor} {m.name}'.strip(), 'library' if w > 0 else 'missing',
                    w, st.peak_of(kind, w), count.get(kind, 0))
        count[kind] = count.get(kind, 0) + 1
        if kind in CABLED:
            own = st.cables.get(s.uid, {})
            r['cable_set'] = 'm' in own
            r['cable_m'] = own['m'] if 'm' in own else route_length(project, s)
            r['awg'] = own.get('awg', st.cable_awg)
            r.update(cable_check(st, r['steady_w'], r['peak_w'], r['cable_m'], r['awg']))
        rows.append(r)
    for i, (key, attr, tag, model) in enumerate(FIXED):
        w = getattr(st, attr)
        if w > 0:
            rows.append(_device(st, key, '', tag, key, model or project.ecu.name, 'setting', w, st.peak_of(key, w), i))
    for i, (name, steady, peak, circ, enc) in enumerate(st.other):
        rows.append(_device(st, f'other:{i}', '', name, 'other', 'Other load', 'user', steady, peak, i, circ, enc))
    circuits = _circuits(st, rows)
    res = PowerResult(st, rows, circuits, _totals(st, rows, circuits), [], v.name, v.uid)
    res.findings = _findings(res)
    return res


def _circuits(st, rows) -> list:
    out = []
    for name in CIRCUITS:
        devs = [r for r in rows if r['circuit'] == name]
        if not devs:
            continue
        sa = sum(r['steady_a'] for r in devs)
        pa = sum(r['peak_a'] for r in devs)
        auto = auto_fuse(sa, pa)
        fuse = st.fuses.get(name) or auto
        out.append({'circuit': name, 'label': CIRCUIT_LABEL[name], 'tags': [r['tag'] for r in devs],
                    'uids': [r['uid'] for r in devs if r['uid']], 'steady_w': sum(r['steady_w'] for r in devs),
                    'peak_w': sum(r['peak_w'] for r in devs), 'steady_a': sa, 'peak_a': pa, 'fuse_a': fuse,
                    'fuse_auto': name not in st.fuses, 'suggested_fuse_a': auto,
                    'load': sa / fuse if fuse else None, 'peak_load': pa / fuse if fuse else None,
                    'severity': fuse_severity(sa, pa, fuse)})
    return out


def _totals(st, rows, circuits) -> dict:
    steady = sum(r['steady_w'] for r in rows)
    peak = sum(r['peak_w'] for r in rows)
    bus_a = sum(r['steady_a'] for r in rows)
    bus_pa = sum(r['peak_a'] for r in rows)
    out_w, out_pw = st.bus_v * bus_a, st.bus_v * bus_pa          # the loads and their cables' losses
    in_w, in_pw = out_w / st.dcdc_eff, out_pw / st.dcdc_eff
    usable = st.battery_wh * st.battery_dod
    spare = st.alternator_a * st.alternator_v
    margin = spare - in_w
    if margin < 0:
        drain = usable / -margin * 60.0
    else:
        drain = math.inf
    runtime = usable / in_w * 60.0 if in_w > 0 else math.inf
    loss = in_w - out_w
    inside = [r for r in rows if r['enclosure']]
    heat = sum(r['steady_w'] for r in inside) + (loss if st.dcdc_in_enclosure else 0.0)
    rank = {s: i for i, s in enumerate(SEVERITIES)}
    worst = min(circuits, key=lambda c: (rank[c['severity']], -(c['load'] if c['load'] is not None else math.inf)),
                default=None)                              # the worst status first, then the busiest
    return {'steady_w': steady, 'peak_w': peak, 'cable_loss_w': out_w - steady, 'bus_a': bus_a, 'bus_peak_a': bus_pa,
            'dcdc_output_w': out_w, 'dcdc_output_peak_w': out_pw, 'dcdc_load': out_w / st.dcdc_w,
            'dcdc_peak_load': out_pw / st.dcdc_w, 'dcdc_input_w': in_w, 'dcdc_input_peak_w': in_pw,
            'dcdc_loss_w': loss, 'input_a': in_w / st.alternator_v, 'alternator_spare_w': spare,
            'alternator_margin_w': margin, 'alternator_load': in_w / spare if spare > 0 else math.inf,
            'battery_usable_wh': usable, 'battery_drain_min': drain, 'runtime_min': runtime,
            'heat_w': heat, 'heat_tags': [r['tag'] for r in inside],
            'worst_circuit': worst['circuit'] if worst else None,
            'by_kind_w': {k: sum(r['steady_w'] for r in rows if r['kind'] == k) for k in KINDS},
            'counts': {k: sum(1 for r in rows if r['kind'] == k) for k in KINDS}}


def _finding(out, sev, kind, message, rows=(), objects=''):
    tags = [r['tag'] for r in rows]
    out.append({'severity': sev, 'kind': kind, 'message': message, 'tags': tags,
                'uids': [r['uid'] for r in rows if r.get('uid')], 'objects': objects or ', '.join(tags)})


def _pct(x):
    return f'{x * 100:.0f} %'


def _findings(res: PowerResult) -> list:
    st, t, rows = res.settings, res.totals, res.rows
    out = []
    bus = f'{st.bus_v:g} V'
    # ---- the DC-DC converter
    if t['dcdc_output_w'] > st.dcdc_w + 1e-9:
        _finding(out, 'critical', 'dcdc',
                 f'The DC-DC converter is over its rating: the rig draws {t["dcdc_output_w"]:,.0f} W steady from its '
                 f'{st.dcdc_w:,.0f} W output ({_pct(t["dcdc_load"])}). Use a bigger converter or split the circuits '
                 'over two.', objects='DC-DC converter')
    elif t['dcdc_output_peak_w'] > st.dcdc_w + 1e-9:
        _finding(out, 'warning', 'dcdc',
                 f'At switch-on the rig draws {t["dcdc_output_peak_w"]:,.0f} W for a few seconds, over the DC-DC '
                 f'converter\'s {st.dcdc_w:,.0f} W ({_pct(t["dcdc_peak_load"])}). Check its overload rating or switch '
                 'the lidars on after the rest.', objects='DC-DC converter')
    else:
        _finding(out, 'good', 'dcdc',
                 f'The DC-DC converter carries the rig: {t["dcdc_output_w"]:,.0f} W steady '
                 f'({_pct(t["dcdc_load"])}) and {t["dcdc_output_peak_w"]:,.0f} W at switch-on of its '
                 f'{st.dcdc_w:,.0f} W; it draws {t["dcdc_input_w"]:,.0f} W from the vehicle at '
                 f'{st.dcdc_eff * 100:g} % efficiency.')
    # ---- the alternator while driving
    spare, margin = t['alternator_spare_w'], t['alternator_margin_w']
    have = f'{spare:,.0f} W spare ({st.alternator_a:g} A at {st.alternator_v:g} V)'
    if margin < -1e-9:
        tail = (f'The auxiliary battery drains while driving: empty after {t["battery_drain_min"]:,.0f} min.'
                if t['battery_usable_wh'] > 0 else 'With no auxiliary battery the vehicle\'s own battery drains.')
        _finding(out, 'serious', 'alternator',
                 f'The alternator cannot carry the rig: it has {have} and the DC-DC converter draws '
                 f'{t["dcdc_input_w"]:,.0f} W. {tail}', objects='Alternator')
    elif spare > 0 and t['alternator_load'] > CONTINUOUS + 1e-9:
        _finding(out, 'warning', 'alternator',
                 f'The rig takes {_pct(t["alternator_load"])} of the alternator\'s {have}: {margin:,.0f} W left. At '
                 'idle an alternator gives less; check the figure at idle.', objects='Alternator')
    else:
        _finding(out, 'good', 'alternator',
                 f'The alternator carries the rig with {margin:,.0f} W to spare: {t["dcdc_input_w"]:,.0f} W of its '
                 f'{have}.')
    # ---- engine off, on the auxiliary battery
    rt = t['runtime_min']
    if t['dcdc_input_w'] > 0:
        if rt < st.runtime_target_min - 1e-9:
            need = st.runtime_target_min / 60.0 * t['dcdc_input_w'] / st.battery_dod
            have_b = (f'on the auxiliary battery ({t["battery_usable_wh"]:,.0f} of {st.battery_wh:,.0f} Wh usable)'
                      if st.battery_wh > 0 else 'with no auxiliary battery')
            _finding(out, 'warning', 'runtime',
                     f'Engine off, the rig runs {rt:,.0f} min {have_b}, short of the {st.runtime_target_min:g} min '
                     f'target. A battery of {need:,.0f} Wh ({st.battery_dod * 100:g} % usable) would do.',
                     objects='Auxiliary battery')
        elif st.battery_wh > 0:
            _finding(out, 'good', 'runtime',
                     f'Engine off, the rig runs {rt:,.0f} min ({rt / 60:.1f} h) on the auxiliary battery, over the '
                     f'{st.runtime_target_min:g} min target.')
    # ---- circuits and their fuses
    for c in res.circuits:
        devs = [r for r in rows if r['circuit'] == c['circuit']]
        name, fuse, rec = f'The {CIRCUIT_NAME[c["circuit"]]} circuit', c['fuse_a'], c['suggested_fuse_a']
        sa, pa = c['steady_a'], c['peak_a']
        fit = (f'Fit a {rec:g} A fuse (and wire rated for it)' if rec else
               'No standard fuse holds it: split the circuit')
        if fuse is None:
            _finding(out, 'critical', 'circuit',
                     f'{name} draws {sa:.1f} A steady and {pa:.1f} A at switch-on: more than a '
                     f'{STANDARD_FUSES_A[-1]:g} A fuse carries. Split it over more circuits or raise the bus voltage.',
                     devs, f'{c["label"]} circuit')
        elif c['severity'] == 'critical' and sa > fuse + 1e-9:
            _finding(out, 'critical', 'circuit',
                     f'{name} draws {sa:.1f} A steady, over its {fuse:g} A fuse: the fuse blows. {fit}'
                     + (' or split the circuit.' if rec else '.'), devs, f'{c["label"]} circuit')
        elif c['severity'] == 'critical':
            _finding(out, 'critical', 'circuit',
                     f'{name} draws {pa:.1f} A at switch-on, more than twice its {fuse:g} A fuse: a blade fuse blows '
                     f'within seconds. {fit}' + (', or switch the lidars on after the rest.' if rec else '.'), devs,
                     f'{c["label"]} circuit')
        elif c['severity'] == 'warning' and pa > fuse + 1e-9:
            _finding(out, 'warning', 'circuit',
                     f'{name} peaks at {pa:.1f} A at switch-on, {_pct(c["peak_load"])} of its {fuse:g} A fuse. Fuses '
                     'carry short inrush: check the fuse\'s time-current curve against the few seconds of spin-up'
                     + (f', or fit a {rec:g} A fuse.' if rec else '.'), devs, f'{c["label"]} circuit')
        elif c['severity'] == 'warning':
            _finding(out, 'warning', 'circuit',
                     f'{name} draws {sa:.1f} A steady, {_pct(c["load"])} of its {fuse:g} A fuse: above the '
                     f'{CONTINUOUS * 100:.0f} % a fuse carries continuously. {fit}.', devs, f'{c["label"]} circuit')
    known = [c for c in res.circuits if c['load'] is not None]
    if known and all(c['severity'] == 'good' for c in res.circuits):
        b = max(known, key=lambda c: c['load'])
        _finding(out, 'good', 'circuit',
                 f'Every circuit is within its fuse: the busiest, the {CIRCUIT_NAME[b["circuit"]]} circuit, draws '
                 f'{b["steady_a"]:.1f} A steady of its {b["fuse_a"]:g} A fuse ({_pct(b["load"])}) and '
                 f'{b["peak_a"]:.1f} A at switch-on'
                 + (' (fuse sized automatically).' if b['fuse_auto'] else '.'))
    # ---- power cables
    cabled = [r for r in rows if r['cable_m'] is not None]
    for r in cabled:
        what = f'{r["cable_m"]:.1f} m of {r["awg"]} AWG'
        prob = r['cable_problem']
        if not prob:
            continue
        fix = fix_awg(st, r['steady_w'], r['peak_w'], r['cable_m'])
        tip = f' Use {fix} AWG or a shorter route.' if fix is not None and fix < r['awg'] else \
            ' Shorten the route or raise the bus voltage.'
        if prob == 'collapse':
            _finding(out, 'serious', 'cable',
                     f'{r["tag"]}: {what} ({r["cable_ohm"]:.2f} Ω there and back) cannot deliver '
                     f'{r["peak_w"] if r["v_steady"] is not None else r["steady_w"]:.0f} W at {bus}: the voltage '
                     f'collapses{" at switch-on" if r["v_steady"] is not None else ""}.{tip}', [r])
        elif prob == 'drop':
            _finding(out, 'serious', 'cable',
                     f'{r["tag"]}: its power cable ({what}) drops {r["drop_pct"]:.1f} % ({r["drop_v"]:.2f} V at '
                     f'{r["steady_a"]:.2f} A), over the {st.max_drop_pct:g} % limit.{tip}', [r])
        elif prob == 'min_input':
            _finding(out, 'serious', 'cable',
                     f'{r["tag"]}: at switch-on its input falls to {r["v_peak"]:.1f} V ({r["peak_a"]:.1f} A over '
                     f'{what}), below the {st.min_input_v:g} V it needs.{tip}', [r])
        elif prob == 'peak_drop':
            _finding(out, 'warning', 'cable',
                     f'{r["tag"]}: at switch-on its power cable ({what}) drops {r["peak_drop_pct"]:.1f} % for a few '
                     f'seconds ({r["v_peak"]:.1f} V at the device), over the {st.max_drop_pct:g} % limit.{tip}', [r])
    if cabled and all(r['cable_severity'] == 'good' for r in cabled):
        b = max(cabled, key=lambda r: r['drop_pct'])
        _finding(out, 'good', 'cable',
                 f'Every power cable is within the {st.max_drop_pct:g} % drop: the worst, {b["tag"]}, drops '
                 f'{b["drop_pct"]:.1f} % ({b["cable_m"]:.1f} m of {b["awg"]} AWG) and keeps {b["v_peak"]:.1f} V at '
                 'switch-on.')
    # ---- devices without a power figure
    typical = [r for r in rows if r['source'] == 'typical']
    for model in dict.fromkeys(r['model'] for r in typical):
        rs = [r for r in typical if r['model'] == model]
        _finding(out, 'note', 'no_figure',
                 f'The {model} module gives no power figure, so {", ".join(r["tag"] for r in rs)} count'
                 f'{"s" if len(rs) == 1 else ""} the typical {st.camera_w:g} W of a GMSL camera powered over its coax. '
                 'Enter the module\'s figure in the camera library.', rs)
    for r in rows:
        if r['source'] == 'missing':
            _finding(out, 'note', 'no_figure',
                     f'{r["tag"]} ({r["model"]}): its model gives no power figure, so it counts 0 W. Enter it in the '
                     'sensor library.', [r])
    # ---- heat in the enclosure
    if t['heat_w'] > 0:
        parts = ', '.join(t['heat_tags'])
        if st.dcdc_in_enclosure and t['dcdc_loss_w'] > 0:
            parts = (parts + ' and ' if parts else '') + f'the DC-DC converter\'s {t["dcdc_loss_w"]:,.0f} W loss'
        _finding(out, 'note', 'heat', f'The enclosure must shed {t["heat_w"]:,.0f} W of heat: {parts}.',
                 objects='Enclosure')
    order = {s: i for i, s in enumerate(SEVERITIES)}
    out.sort(key=lambda f: order.get(f['severity'], 9))          # stable: the order above within a severity
    return out


# =========================================================================== the Issues list
def issues(project, vg=None) -> list:
    """The budget's problems for the Issues list (cheap: no geometry): the DC-DC converter over its rating, the
    alternator short, the engine-off runtime under target, a circuit over its fuse or its 80 %, a power cable's drop."""
    res = run(project)
    return [R.Issue(f['severity'], f['message'], f['objects'] or f'{res.variant} · power', list(f['uids']))
            for f in res.findings if f['severity'] in ISSUE_SEVERITIES]


# =========================================================================== output
def _r(v, d=3):
    if v is None:
        return None
    v = float(v)
    return round(v, d) if math.isfinite(v) else None


def summary(res: PowerResult) -> dict:
    """Compact figures (the assistant, the batch report header)."""
    t, st = res.totals, res.settings
    w = res.circuit(t['worst_circuit']) if t['worst_circuit'] else None
    return {'bus_v': st.bus_v, 'steady_w': _r(t['steady_w'], 1), 'peak_w': _r(t['peak_w'], 1),
            'dcdc_output_w': _r(t['dcdc_output_w'], 1), 'dcdc_input_w': _r(t['dcdc_input_w'], 1),
            'dcdc_rating_w': st.dcdc_w, 'dcdc_load_pct': _r(t['dcdc_load'] * 100, 1),
            'dcdc_peak_load_pct': _r(t['dcdc_peak_load'] * 100, 1),
            'alternator_spare_w': _r(t['alternator_spare_w'], 1),
            'alternator_margin_w': _r(t['alternator_margin_w'], 1),
            'battery_drain_while_driving_min': _r(t['battery_drain_min'], 1),
            'engine_off_runtime_min': _r(t['runtime_min'], 1), 'runtime_target_min': st.runtime_target_min,
            'enclosure_heat_w': _r(t['heat_w'], 1),
            'worst_circuit': None if w is None else {
                'circuit': w['circuit'], 'fuse_a': w['fuse_a'], 'fuse_auto': w['fuse_auto'],
                'steady_a': _r(w['steady_a'], 2), 'peak_a': _r(w['peak_a'], 2),
                'load_pct': _r(w['load'] * 100, 1) if w['load'] is not None else None},
            'per_device': [{'tag': r['tag'], 'kind': r['kind'], 'steady_w': _r(r['steady_w'], 2),
                            'peak_w': _r(r['peak_w'], 2), 'circuit': r['circuit'], 'current_a': _r(r['steady_a'], 2),
                            'cable_drop_pct': _r(r['drop_pct'], 2)} for r in res.rows],
            'findings': [{'severity': f['severity'], 'message': f['message']} for f in res.findings
                         if f['severity'] != 'good'],
            'assumptions': ASSUMPTIONS}


def _row_doc(r) -> dict:
    return {'tag': r['tag'], 'kind': r['kind'], 'model': r['model'], 'power_source': r['source'],
            'steady_w': _r(r['steady_w']), 'peak_w': _r(r['peak_w']), 'circuit': r['circuit'], 'supply': r['supply'],
            'steady_a': _r(r['steady_a']), 'peak_a': _r(r['peak_a']), 'cable_m': _r(r['cable_m']),
            'cable_length_set': r['cable_set'] if r['cable_m'] is not None else None, 'awg': r['awg'],
            'cable_ohm': _r(r['cable_ohm'], 4), 'drop_v': _r(r['drop_v']), 'drop_pct': _r(r['drop_pct'], 2),
            'peak_drop_v': _r(r['peak_drop_v']), 'peak_drop_pct': _r(r['peak_drop_pct'], 2),
            'device_v_steady': _r(r['v_steady']) if r['cable_m'] is not None else None,
            'device_v_peak': _r(r['v_peak']) if r['cable_m'] is not None else None,
            'cable_status': r['cable_severity'], 'in_enclosure': r['enclosure']}


def _circuit_doc(c) -> dict:
    return {'circuit': c['circuit'], 'label': c['label'], 'devices': c['tags'], 'fuse_a': c['fuse_a'],
            'fuse_auto': c['fuse_auto'], 'suggested_fuse_a': c['suggested_fuse_a'],
            'steady_w': _r(c['steady_w']), 'peak_w': _r(c['peak_w']), 'steady_a': _r(c['steady_a']),
            'peak_a': _r(c['peak_a']), 'load_pct': _r(c['load'] * 100, 2) if c['load'] is not None else None,
            'peak_load_pct': _r(c['peak_load'] * 100, 2) if c['peak_load'] is not None else None,
            'status': c['severity']}


def to_doc(res: PowerResult, project=None) -> dict:
    """The JSON export: every device's power, current and cable, the circuits and their fuses, the supply's totals
    and the findings."""
    t, st = res.totals, res.settings
    return {
        'schema': SCHEMA,
        'project': getattr(project, 'name', ''), 'variant': res.variant,
        'units': {'power': 'W', 'current': 'A', 'voltage': 'V', 'energy': 'Wh', 'time': 'min', 'length': 'm',
                  'resistance': 'ohm'},
        'frame': 'vehicle (ISO 8855, rear axle): cable routes from the ECU position',
        'settings': st.to_dict(),
        'assumptions': ASSUMPTIONS,
        'devices': [_row_doc(r) for r in res.rows],
        'circuits': [_circuit_doc(c) for c in res.circuits],
        'totals': {'steady_w': _r(t['steady_w']), 'peak_w': _r(t['peak_w']), 'cable_loss_w': _r(t['cable_loss_w']),
                   'bus_v': st.bus_v, 'bus_steady_a': _r(t['bus_a']), 'bus_peak_a': _r(t['bus_peak_a']),
                   'dcdc_output_w': _r(t['dcdc_output_w']), 'dcdc_output_peak_w': _r(t['dcdc_output_peak_w']),
                   'dcdc_rating_w': st.dcdc_w, 'dcdc_load_pct': _r(t['dcdc_load'] * 100, 2),
                   'dcdc_peak_load_pct': _r(t['dcdc_peak_load'] * 100, 2), 'dcdc_efficiency': st.dcdc_eff,
                   'dcdc_input_w': _r(t['dcdc_input_w']), 'dcdc_input_peak_w': _r(t['dcdc_input_peak_w']),
                   'dcdc_loss_w': _r(t['dcdc_loss_w']), 'vehicle_v': st.alternator_v,
                   'vehicle_input_a': _r(t['input_a']), 'alternator_spare_w': _r(t['alternator_spare_w']),
                   'alternator_margin_w': _r(t['alternator_margin_w']),
                   'battery_usable_wh': _r(t['battery_usable_wh']),
                   'battery_drain_while_driving_min': _r(t['battery_drain_min']),
                   'engine_off_runtime_min': _r(t['runtime_min']), 'runtime_target_min': st.runtime_target_min,
                   'enclosure_heat_w': _r(t['heat_w']), 'enclosure_devices': t['heat_tags'],
                   'by_kind_w': {k: _r(x) for k, x in t['by_kind_w'].items()}},
        'findings': [{'severity': f['severity'], 'kind': f['kind'], 'message': f['message'], 'objects': f['objects'],
                      'devices': f['tags']} for f in res.findings],
    }


CSV_COLUMNS = ('tag', 'kind', 'model', 'power_source', 'steady_w', 'peak_w', 'circuit', 'steady_a', 'peak_a',
               'cable_m', 'awg', 'cable_ohm', 'drop_v', 'drop_pct', 'peak_drop_pct', 'device_v_peak', 'in_enclosure')
CIRCUIT_COLUMNS = ('circuit', 'devices', 'fuse_a', 'fuse_auto', 'steady_w', 'peak_w', 'steady_a', 'peak_a',
                   'load_pct', 'peak_load_pct', 'status')


def _blank(v):
    return '' if v is None else v


def to_csv(res: PowerResult) -> str:
    """One row per device and a total row (the total's current is the bus's: the loads and their cables' losses)."""
    t = res.totals
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator='\n')
    w.writerow(CSV_COLUMNS)
    for r in res.rows:
        d = _row_doc(r)
        w.writerow([_blank(d[k]) for k in CSV_COLUMNS])
    w.writerow(['total', '', f'DC-DC output {_r(t["dcdc_output_w"])} W', '', _r(t['steady_w']), _r(t['peak_w']), 'bus',
                _r(t['bus_a']), _r(t['bus_peak_a']), '', '', '', '', '', '', '', ''])
    return buf.getvalue()


def circuits_csv(res: PowerResult) -> str:
    """One row per circuit with a load: its devices (space-separated tags), fuse, currents and loads."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator='\n')
    w.writerow(CIRCUIT_COLUMNS)
    for c in res.circuits:
        d = _circuit_doc(c)
        d['devices'] = ' '.join(c['tags'])
        w.writerow([_blank(d[k]) for k in CIRCUIT_COLUMNS])
    return buf.getvalue()


def files(res: PowerResult, project=None) -> list:
    """[(name, bytes)]: power_budget.json, power_budget.csv (devices) and power_circuits.csv."""
    return [('power_budget.json', (json.dumps(to_doc(res, project), indent=1) + '\n').encode('utf-8')),
            ('power_budget.csv', to_csv(res).encode('utf-8')),
            ('power_circuits.csv', circuits_csv(res).encode('utf-8'))]


def export(res: PowerResult, project, out_dir) -> list[str]:
    """Write the files into out_dir; returns their paths."""
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for name, data in files(res, project):
        path = os.path.join(out_dir, name)
        with open(path, 'wb') as fh:
            fh.write(data)
        paths.append(path)
    return paths
