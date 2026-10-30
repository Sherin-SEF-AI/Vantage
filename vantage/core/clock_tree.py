"""Clock tree and time sync: how the rig's clocks are distributed from GNSS to every sensor, what each sensor's
timestamps are off by when the distribution drifts or loses lock, and how far that moves a lidar point in a camera
image.

The tree (build_topology, every parameter editable per node in analysis.clock_tree 'nodes')
- GNSS receiver: its PPS edge is UTC within pps_ns (1σ) while it tracks satellites; none during a PPS dropout.
- Grandmaster: the time server when the power budget lists one (core.power ptp_w > 0), else the compute. It is
  disciplined to the PPS by a clock servo (one update a second). A second grandmaster-capable clock (the compute, when
  the time server is the grandmaster) takes over by the best master clock algorithm (BMCA, lowest priority1 wins)
  after the announce timeout when the grandmaster is lost; a better one coming back takes over at once.
- Ethernet switch (a gPTP time-aware bridge, when the power budget lists one): its ports' sync interval, its
  residence-time jitter and its uplink's path delay and asymmetry.
- Endpoints: lidars, radars and the IMU stamp with their own PTP-disciplined clock (stamping 'ptp'), their own clock
  disciplined to the PPS ('pps') or the host's receive time ('host': the compute's clock plus a jitter and a latency
  left uncorrected), from the library model's time_sync. A spinning lidar's head is phase-locked to its own clock (as
  core.timing assumes), so its clock error moves both its timestamps and its sweep.
- GMSL2 deserializers (ports 1-4 on A, 5-8 on B, as core.timing): an FSYNC timer on its own crystal, either locked to
  the compute's clock (re-phased every rephase_s, counting a whole number of ticks per frame in between; 0 = every
  edge scheduled on the compute clock) or free-running. Each FSYNC edge falls on a timer tick.
- Cameras on a trigger line fire on their deserializer's FSYNC edges at the sync plan's trigger times (core.timing);
  the driver stamps a frame either with its scheduled trigger time ('trigger') or with its arrival time ('arrival':
  the compute's clock plus a jitter). The stamp refers to the middle row's exposure centre (trigger + readout/2 +
  exposure/2, core.timing), so with 'trigger' stamping the error is the FSYNC edge's: the deserializer clock's error
  minus the tick rounding, minus a late pulse; a missed pulse labels the next frame with the missed pulse's time, one
  frame period early, for that one frame (the driver resynchronises on the frame counter). Cameras without a hardware
  trigger input or a trigger line free-run and are stamped on arrival.

Clocks. A clock's error x(t) is its reading minus true time. A servo (a PI loop of the node's bandwidth, damping 0.7,
updated every sync interval, or every second against the PPS) steers it to its reference: the measured offset is
x − x_ref minus half the summed asymmetry of the links on the way (gPTP assumes symmetric links) plus noise of the
links' jitter added in quadrature. The step policy steps an offset larger than step_us at once (the clock jumps, back
in time when it was ahead); the slew policy never steps and limits the correction to slew_ppm. A clock without a
reference (PPS lost, no grandmaster, free-running) holds over: it keeps its learned frequency and drifts by its
oscillator's holdover stability ppm (a normal draw per Monte Carlo run, 1σ) plus aging: x = x0 + y·t + ½·a·t².

Faults (analysis.clock_tree 'events', on a timeline from 0 to duration_s): PPS dropout for a time (an urban canyon or
a tunnel), grandmaster loss (for a time, or for good), a link asymmetry step on a node's uplink, a missed or a late
FSYNC pulse on a deserializer (all its cameras) or one camera. A seeded Monte Carlo (runs × the timeline) gives every
sensor's timestamp error against true time and against the reference lidar (p50, p95, max of the magnitude).

Projection. A timestamp error Δt between a lidar point and a camera image is a vehicle motion the pipeline does not
account for: at speed v and yaw rate ω the vehicle turns ω·Δt about its vertical axis and moves along the arc
(translation v·Δt for a straight drive). A static point P seen by the lidar is, in the vehicle frame at the camera's
instant, R(ω·Δt)ᵀ·(P − T); both are projected through the camera's real lens model (core.lens) and the pixel distance
is the error: rotation dominates far away, translation (parallax) close by. Points lie on each pixel's ray at a range
from the camera or on the ground plane (z = 0). With one timestamp per image a rolling shutter adds (row/H − ½)·readout
to each row's Δt (off by default: rows compensated).

Conventions: ISO 8855 vehicle frame (x forward, y left, z up, origin at the rear-axle centre on the ground); times in
seconds from the scenario's start; errors reported in µs (stamp minus true time); image errors in px (OpenCV pixel
centres on integers).

Limits. Typical values are class figures (oscillator, PHY and receiver datasheets, IEEE 802.1AS-2020 defaults) or
heuristics, each labelled. Link asymmetry adds half of itself whichever way a link is crossed. Switches relay time
without a servo of their own (802.1AS transparent relaying), and the compute's system clock is its PTP clock
(phc2sys's own small error is left out). The holdover frequency is a fresh draw at the first holdover and stays (its
aging goes on); temperature transients, oscillator noise (Allan deviation) and servo state machines beyond step/slew
are not modelled. The simulation steps at the shortest sync interval (at most 125 ms); the PPS updates on the nearest
step. FSYNC to exposure latency is taken as known and constant. The motion is a planar arc at constant speed and yaw
rate (no pitch, roll or sideslip), the world static.
"""
from __future__ import annotations

import csv
import io
import json
import math
import os
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from . import sensors as SN
from . import timing as TM
from .analysis import camera_frames
from .lens import resolve

SCHEMA = 'vantage.clock_tree/1'
SEVERITIES = ('critical', 'serious', 'warning', 'note', 'good')
KINDS = ('gnss', 'server', 'compute', 'switch', 'deser', 'camera', 'lidar', 'radar', 'imu')
KIND_LABEL = {'gnss': 'GNSS receiver', 'server': 'Time server', 'compute': 'Compute', 'switch': 'Ethernet switch',
              'deser': 'Deserializer', 'camera': 'Camera', 'lidar': 'Lidar', 'radar': 'Radar', 'imu': 'IMU'}
SENSOR_KINDS = ('lidar', 'radar', 'imu')
DAMPING = 0.7                    # the servo's damping ratio
MAX_WT = 0.6                     # a discrete PI loop's ω·Ts at most (bandwidth capped for stability)
DEFAULT_SYNC_MS = 125.0          # IEEE 802.1AS-2020 default sync interval (logSyncInterval −3)
ACCURACY_US = 1.0                # IEEE 802.1AS-2020 Annex B: within 1 µs over up to 7 hops (its accuracy goal)
MAX_SAMPLES = 1_500_000          # sensor stamps evaluated per sensor at most (runs × frames; frames strided beyond)
GROUND_MAX_M = 200.0             # a ray reaching the ground further than this counts as missing it
GRID = (41, 33)                  # image sample grid (as core.timing's motion figures)

# key: (label, unit, lo, hi, decimals, tooltip)
PARAMS = {
    'pps_ns': ('PPS error', 'ns', 0.0, 1e6, 1, 'The PPS edge against UTC while the receiver tracks satellites, 1σ'),
    'ppm': ('Holdover', 'ppm', 0.0, 1000.0, 3, 'Frequency error the oscillator drifts at once it loses its reference '
                                                '(1σ); a free-running clock drifts at it from the start'),
    'aging_ppb_day': ('Aging', 'ppb/day', 0.0, 1e5, 1, 'The oscillator\'s frequency ageing: adds ½·a·t² in holdover'),
    'servo_hz': ('Servo', 'Hz', 0.001, 10.0, 3, 'Bandwidth of the clock servo (a PI loop, damping 0.7) that steers '
                                                'this clock to its reference'),
    'policy': ('Offset policy', '', None, None, 0, 'Step: an offset larger than the step threshold is stepped at once '
                                                   '(the clock jumps). Slew: never step, correct at most the slew '
                                                   'limit'),
    'step_us': ('Step above', 'µs', 0.0, 1e7, 1, 'With the step policy an offset larger than this is stepped'),
    'slew_ppm': ('Slew limit', 'ppm', 1.0, 1e6, 0, 'The largest frequency correction the servo applies'),
    'priority': ('Priority1', '', 0.0, 255.0, 0, 'BMCA priority1: the lowest grandmaster-capable clock is the '
                                                 'grandmaster; 255 = never grandmaster'),
    'delay_ns': ('Path delay', 'ns', 0.0, 1e7, 0, 'Mean path delay of the link to its parent: measured by peer delay '
                                                  'and removed, shown for reference'),
    'asym_ns': ('Asymmetry', 'ns', -1e6, 1e6, 0, 'One way of the link longer than the other: gPTP assumes symmetric '
                                                 'links, so every clock synchronised across it is off by half'),
    'jitter_ns': ('Jitter', 'ns', 0.0, 1e7, 1, 'Timestamp or residence-time jitter (1σ) this hop adds to every offset '
                                               'measured across it'),
    'sync_ms': ('Sync interval', 'ms', 1.0, 4000.0, 2, 'Sync messages on this switch\'s ports: the clocks behind it '
                                                       'update this often'),
    'fsync': ('FSYNC clock', '', None, None, 0, 'Locked: the FSYNC timer follows the compute\'s clock. Free: it runs '
                                                'on its own crystal'),
    'tick_us': ('Timer tick', 'µs', 0.0, 1e5, 3, 'Every FSYNC edge falls on a tick of the timer (0 = exact)'),
    'rephase_s': ('Re-phase', 's', 0.0, 3600.0, 2, 'The locked timer counts a whole number of ticks per frame on its '
                                                   'own crystal and is re-phased to the compute clock this often (0 = '
                                                   'every edge scheduled on the compute clock)'),
    'stamping': ('Stamping', '', None, None, 0, 'Where the timestamp comes from'),
    'host_us': ('Host jitter', 'µs', 0.0, 1e6, 1, 'Jitter (1σ) of a host receive timestamp'),
    'latency_us': ('Latency left', 'µs', -1e6, 1e6, 1, 'Fixed latency a host receive timestamp does not remove'),
    'readout_ms': ('Readout', 'ms', 0.0, 100.0, 2, 'Rolling-shutter readout from the first row to the last'),
    'exposure_ms': ('Exposure', 'ms', 0.001, 100.0, 3, 'Exposure time of each row'),
    'rate_hz': ('Rate', 'Hz', 0.1, 2000.0, 2, 'Frames or messages a second'),
}
CHOICES = {'policy': (('step', 'Step'), ('slew', 'Slew')),
           'fsync': (('locked', 'Locked'), ('free', 'Free-running')),
           'stamping': (('trigger', 'Trigger time'), ('arrival', 'Arrival'), ('ptp', 'PTP clock'),
                        ('pps', 'PPS-disciplined'), ('host', 'Host receive time'))}
STAMPING = {'camera': ('trigger', 'arrival'), 'lidar': ('ptp', 'pps', 'host'), 'radar': ('ptp', 'pps', 'host'),
            'imu': ('ptp', 'pps', 'host')}
_SERVO = ('ppm', 'aging_ppb_day', 'servo_hz', 'policy', 'step_us', 'slew_ppm')
_LINK = ('delay_ns', 'asym_ns', 'jitter_ns')
KIND_PARAMS = {
    'gnss': ('pps_ns',),
    'server': _SERVO + ('priority',),
    'compute': _SERVO + ('priority',) + _LINK,
    'switch': ('sync_ms',) + _LINK,
    'deser': ('fsync', 'ppm', 'aging_ppb_day', 'tick_us', 'rephase_s'),
    'camera': ('stamping', 'readout_ms', 'exposure_ms', 'host_us', 'latency_us'),
    'lidar': ('stamping', 'rate_hz') + _SERVO + _LINK + ('host_us', 'latency_us'),
    'radar': ('stamping', 'rate_hz') + _SERVO + _LINK + ('host_us', 'latency_us'),
    'imu': ('stamping', 'rate_hz') + _SERVO + _LINK + ('host_us', 'latency_us'),
}

# typical values and where they come from
S_XO = 'typical crystal oscillator (XO): ±10–50 ppm over temperature (oscillator datasheets)'
S_XO_AGE = 'typical XO ageing ±3–5 ppm a year (oscillator datasheets)'
S_TCXO = 'typical TCXO: ±0.5–2 ppm over temperature (TCXO datasheets); an OCXO holds ±0.01–0.1 ppm'
S_TCXO_AGE = 'typical TCXO ageing about ±1 ppm a year (TCXO datasheets)'
S_PPS = 'typical: timing-grade GNSS receivers state 5–30 ns (1σ) PPS accuracy in their datasheets'
S_GM_SERVO = 'heuristic: a PPS discipline loop settles over tens of seconds (0.02–0.1 Hz)'
S_SERVO = ('heuristic: linuxptp\'s PI servo (kp 0.7, ki 0.3 per update) is about 0.1 Hz at one update a second; '
           'faster sync allows more')
S_STEP = ('heuristic: linuxptp steps offsets over first_step_threshold, 20 µs, at start (step_threshold, for later '
          'offsets, is 0: never, by default)')
S_SLEW = 'the Linux kernel\'s adjtimex frequency limit, 500 ppm'
S_PRIO_SERVER = 'IEEE 802.1AS-2020 8.6.2.1: priority1 246 for a network infrastructure time-aware system'
S_PRIO_COMPUTE = 'IEEE 802.1AS-2020 8.6.2.1: priority1 248 for other time-aware systems (255: not grandmaster-capable)'
S_SYNC = ('IEEE 802.1AS-2020 default sync interval, 125 ms (logSyncInterval −3); the automotive profile allows '
          '31.25–125 ms')
S_DELAY = 'heuristic: a few metres of cable (5 ns/m) and two PHYs of about 150 ns'
S_ASYM = 'heuristic: PHY and cable asymmetry of a few to tens of ns; 802.1AS assumes none'
S_JIT_GBE = 'heuristic: hardware timestamps at 125 MHz (1000BASE-T/T1) resolve 8 ns, taken as the 1σ jitter'
S_JIT_FE = 'heuristic: hardware timestamps at 25 MHz (100BASE-T1/TX) resolve 40 ns, taken as the 1σ jitter'
S_JIT_BRIDGE = 'heuristic: residence-time jitter of a hardware time-aware bridge, a few to tens of ns'
S_TICK = 'heuristic: a microcontroller timer at 1 MHz; an FPGA at 100 MHz ticks every 0.01 µs'
S_REPHASE = 'heuristic: the FSYNC timer re-phased on each PPS second'
S_FSYNC = 'heuristic: the deserializer\'s frame sync generated from the compute\'s PTP clock'
S_HOST = 'heuristic: a host receive timestamp: tens to hundreds of µs of jitter (interrupt latency, CAN bus load)'
S_CAM_HOST = 'heuristic: frame-arrival interrupt jitter of tens of µs'
S_LAT = 'heuristic: the driver removes the nominal latency'
S_TIMING = 'Analysis ▸ Sync and timing (its readout and exposure)'
S_TRIGGER = 'heuristic: the driver stamps each frame with its scheduled trigger time'
S_RATE = 'the library model\'s rate'


def _typical(kind, model=None, interface='') -> dict:
    """key -> (value, source) for a node of a kind."""
    gbe = '1000' in interface or 'GMSL' in interface
    servo = {'ppm': (20.0, S_XO), 'aging_ppb_day': (10.0, S_XO_AGE), 'servo_hz': (0.25, S_SERVO),
             'policy': ('step', S_STEP), 'step_us': (20.0, S_STEP), 'slew_ppm': (500.0, S_SLEW)}
    link = {'delay_ns': (300.0, S_DELAY), 'asym_ns': (10.0, S_ASYM),
            'jitter_ns': (8.0, S_JIT_GBE) if gbe else (40.0, S_JIT_FE)}
    host = {'host_us': (100.0, S_HOST), 'latency_us': (0.0, S_LAT)}
    if kind == 'gnss':
        return {'pps_ns': (20.0, S_PPS)}
    if kind == 'server':
        return {'ppm': (0.5, S_TCXO), 'aging_ppb_day': (3.0, S_TCXO_AGE), 'servo_hz': (0.05, S_GM_SERVO),
                'policy': ('slew', 'heuristic: a grandmaster slews so its time never jumps'),
                'step_us': (1000.0, S_STEP), 'slew_ppm': (500.0, S_SLEW), 'priority': (246.0, S_PRIO_SERVER)}
    if kind == 'compute':
        return dict(servo, priority=(248.0, S_PRIO_COMPUTE), delay_ns=(300.0, S_DELAY), asym_ns=(10.0, S_ASYM),
                    jitter_ns=(8.0, S_JIT_GBE))
    if kind == 'switch':
        return {'sync_ms': (DEFAULT_SYNC_MS, S_SYNC), 'delay_ns': (300.0, S_DELAY), 'asym_ns': (10.0, S_ASYM),
                'jitter_ns': (10.0, S_JIT_BRIDGE)}
    if kind == 'deser':
        return {'fsync': ('locked', S_FSYNC), 'ppm': (20.0, S_XO), 'aging_ppb_day': (10.0, S_XO_AGE),
                'tick_us': (1.0, S_TICK), 'rephase_s': (1.0, S_REPHASE)}
    if kind == 'camera':
        return {'stamping': ('trigger', S_TRIGGER), 'readout_ms': (20.0, S_TIMING), 'exposure_ms': (2.0, S_TIMING),
                'host_us': (50.0, S_CAM_HOST), 'latency_us': (0.0, S_LAT)}
    ts = getattr(model, 'time_sync', 'PTP') if model is not None else 'PTP'
    st = 'ptp' if ts == 'PTP' else 'pps' if 'PPS' in ts else 'host'
    rate = float(getattr(model, 'rate_hz', 10.0) or 10.0)
    return dict(servo, **link, **host, stamping=(st, f'the library model\'s time sync ({ts})'),
                rate_hz=(rate, S_RATE))


# ---------------------------------------------------------------- events
EVENT_KINDS = ('pps_dropout', 'gm_loss', 'asymmetry', 'fsync_missed', 'fsync_late')
EVENT_LABEL = {'pps_dropout': 'PPS dropout', 'gm_loss': 'Grandmaster lost', 'asymmetry': 'Link asymmetry',
               'fsync_missed': 'Missed FSYNC pulse', 'fsync_late': 'Late FSYNC pulse'}
EVENT_HELP = {'pps_dropout': 'No PPS for a time (an urban canyon, a tunnel): the clocks it disciplines hold over',
              'gm_loss': 'The grandmaster drops off the network for a time (0 = for good): BMCA hands over to the '
                         'next grandmaster-capable clock after the announce timeout',
              'asymmetry': 'A node\'s uplink becomes asymmetric by a number of ns (a cable or PHY change)',
              'fsync_missed': 'One FSYNC pulse is missed on a deserializer (all its cameras) or a camera',
              'fsync_late': 'One FSYNC pulse comes late by a number of ms'}
EVENT_FIELDS = {'pps_dropout': ('dur_s',), 'gm_loss': ('dur_s',), 'asymmetry': ('node', 'value'),
                'fsync_missed': ('node',), 'fsync_late': ('node', 'value')}
EVENT_UNIT = {'asymmetry': 'ns', 'fsync_late': 'ms'}
MAX_EVENTS = 24


def clean_event(d) -> dict | None:
    """A tolerant event: {'kind', 't_s', 'dur_s', 'node', 'value'}; None for junk."""
    if not isinstance(d, dict) or d.get('kind') not in EVENT_KINDS:
        return None
    k = d['kind']
    out = {'kind': k, 't_s': 0.0, 'dur_s': 0.0, 'node': '', 'value': 0.0}
    for key, lo, hi in (('t_s', 0.0, 3600.0), ('dur_s', 0.0, 3600.0), ('value', -1e6, 1e6)):
        v = _num(d.get(key))
        if v is not None:
            out[key] = min(max(v, lo), hi)
    if isinstance(d.get('node'), str):
        out['node'] = d['node'][:60]
    if k == 'pps_dropout' and out['dur_s'] <= 0:
        out['dur_s'] = 30.0
    if k == 'asymmetry' and out['value'] == 0:
        out['value'] = 200.0
    if k == 'fsync_late':
        out['value'] = min(max(out['value'], 0.0), 1000.0) or 2.0
    return out


def describe_event(e: dict, topo: 'Topology | None' = None) -> str:
    """'PPS dropout at 20 s for 60 s'."""
    k = e['kind']
    who = ''
    if e.get('node'):
        n = topo.node(e['node']) if topo is not None else None
        who = f' on {n.label if n is not None else e["node"]}'
    if k == 'pps_dropout':
        return f'PPS dropout at {e["t_s"]:g} s for {e["dur_s"]:g} s'
    if k == 'gm_loss':
        return f'Grandmaster lost at {e["t_s"]:g} s' + (f' for {e["dur_s"]:g} s' if e['dur_s'] > 0 else ' for good')
    if k == 'asymmetry':
        return f'Link asymmetry {e["value"]:+g} ns{who} from {e["t_s"]:g} s'
    if k == 'fsync_missed':
        return f'Missed FSYNC pulse{who} at {e["t_s"]:g} s'
    return f'FSYNC pulse {e["value"]:g} ms late{who} at {e["t_s"]:g} s'


PRESETS = (('nominal', 'Nominal'), ('urban', 'Urban canyon'), ('gm_loss', 'Grandmaster loss'),
           ('asymmetry', 'Link asymmetry'), ('fsync', 'FSYNC faults'), ('all', 'All faults'))


def preset_events(name: str, topo: 'Topology', duration_s: float = 120.0) -> list:
    """A scenario's events for this topology: nominal (none), an urban canyon (60 s without PPS from a sixth of the
    way in, and a missed FSYNC pulse on the first deserializer), a grandmaster lost for good at a quarter, a 500 ns
    asymmetry step on the reference lidar's uplink at a quarter, FSYNC faults (a missed and a 2 ms late pulse), or all
    of them."""
    D = float(duration_s)
    des = [n.id for n in topo.nodes if n.kind == 'deser']
    lid = topo.lidar or next((n.id for n in topo.nodes if n.kind in SENSOR_KINDS and n.p.get('stamping') == 'ptp'), '')
    out = []
    if name in ('urban', 'all'):
        out.append({'kind': 'pps_dropout', 't_s': round(D / 6, 1), 'dur_s': round(min(60.0, D / 2), 1)})
    if name in ('gm_loss', 'all'):
        out.append({'kind': 'gm_loss', 't_s': round(D / 4 if name == 'gm_loss' else 0.6 * D, 1), 'dur_s': 0.0})
    if name in ('asymmetry', 'all') and lid:
        out.append({'kind': 'asymmetry', 't_s': round(D / 4 if name == 'asymmetry' else 0.3 * D, 1), 'node': lid,
                    'value': 500.0})
    if name in ('urban', 'fsync', 'all') and des:
        out.append({'kind': 'fsync_missed', 't_s': round(0.75 * D, 1), 'node': des[0]})
    if name in ('fsync', 'all') and des:
        out.append({'kind': 'fsync_late', 't_s': round(0.85 * D, 1), 'node': des[-1], 'value': 2.0})
    return [clean_event(e) for e in out]


# ---------------------------------------------------------------- settings
LIMITS = {'seed': (0, 2 ** 31 - 1), 'runs': (1, 400), 'duration_s': (5.0, 600.0), 'speed_kmh': (0.0, 300.0),
          'yaw_dps': (-180.0, 180.0), 'range_m': (0.5, 500.0), 'threshold_px': (0.01, 1000.0),
          'time_budget_us': (0.001, 1e7), 'holdover_need_s': (0.0, 86400.0), 'bmca_s': (0.0, 60.0),
          'slip_ms': (-1000.0, 1000.0)}
INTS = ('seed', 'runs')
SURFACES = ('range', 'ground')


def _num(v):
    if isinstance(v, bool):
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _bool(v, default):
    if isinstance(v, (bool, int)) and not isinstance(v, float):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in ('1', 'true', 'yes', 'on')
    return default


def clean_value(key, value):
    """A node parameter's value cleaned (numbers clamped to PARAMS, choices checked); None when unusable."""
    if key not in PARAMS:
        return None
    if key in CHOICES:
        return value if isinstance(value, str) and value in dict(CHOICES[key]) else None
    v = _num(value)
    if v is None:
        return None
    lo, hi = PARAMS[key][2], PARAMS[key][3]
    v = min(max(v, lo), hi)
    return float(round(v)) if key == 'priority' else v


@dataclass
class ClockTreeSettings:
    seed: int = 1                    # Monte Carlo seed
    runs: int = 100                  # Monte Carlo runs over the timeline
    duration_s: float = 120.0        # the scenario timeline
    events: list = field(default_factory=lambda: [
        {'kind': 'pps_dropout', 't_s': 20.0, 'dur_s': 60.0, 'node': '', 'value': 0.0},
        {'kind': 'fsync_missed', 't_s': 90.0, 'dur_s': 0.0, 'node': 'deser:A', 'value': 0.0}])
    nodes: dict = field(default_factory=dict)    # node id -> {param: value}: the user's edits over the defaults
    speed_kmh: float = 50.0          # vehicle speed for the projection error
    yaw_rate_dps: float = 25.0       # yaw rate (a turn) for the projection error
    surface: str = 'range'           # points at range_m from the camera, or on the ground
    range_m: float = 20.0
    rows_compensated: bool = True    # False: one timestamp per image, rows off by (row/H − ½)·readout
    threshold_px: float = 1.0        # the most projection error the dataset accepts (worst over the image, p95)
    time_budget_us: float = 100.0    # how far from UTC the rig's time may drift (holdover budget)
    holdover_need_s: float = 60.0    # the longest PPS outage the rig must ride through within the budget
    bmca_s: float = 3.0              # announce timeout before a backup grandmaster takes over
    slip_ms: float = 1.5             # the slip calculator's time error
    preset: str = 'urban'            # the scenario preset the events came from ('' = edited by hand)

    @property
    def yaw_dps(self) -> float:
        return self.yaw_rate_dps

    @property
    def speed_mps(self) -> float:
        return self.speed_kmh / 3.6

    def to_dict(self):
        return {'seed': self.seed, 'runs': self.runs, 'duration_s': self.duration_s,
                'events': [dict(e) for e in self.events], 'nodes': {k: dict(v) for k, v in self.nodes.items()},
                'speed_kmh': self.speed_kmh, 'yaw_rate_dps': self.yaw_rate_dps, 'surface': self.surface,
                'range_m': self.range_m, 'rows_compensated': self.rows_compensated,
                'threshold_px': self.threshold_px, 'time_budget_us': self.time_budget_us,
                'holdover_need_s': self.holdover_need_s, 'bmca_s': self.bmca_s, 'slip_ms': self.slip_ms,
                'preset': self.preset}

    @staticmethod
    def from_dict(d) -> 'ClockTreeSettings':
        """Tolerant: unknown keys dropped, numbers clamped to LIMITS, bad values keep the defaults."""
        s = ClockTreeSettings()
        d = d if isinstance(d, dict) else {}
        for k, (lo, hi) in LIMITS.items():
            key = 'yaw_rate_dps' if k == 'yaw_dps' else k
            v = _num(d.get(key))
            if v is None:
                continue
            v = min(max(v, lo), hi)
            setattr(s, key, int(round(v)) if k in INTS else v)
        if isinstance(d.get('surface'), str) and d['surface'] in SURFACES:
            s.surface = d['surface']
        s.rows_compensated = _bool(d.get('rows_compensated'), s.rows_compensated)
        if isinstance(d.get('events'), list):
            s.events = [e for e in (clean_event(x) for x in d['events'][:MAX_EVENTS * 4]) if e is not None][:MAX_EVENTS]
        else:
            s.events = [clean_event(e) for e in s.events]
        if isinstance(d.get('nodes'), dict):
            for nid, kv in list(d['nodes'].items())[:400]:
                if not isinstance(nid, str) or not isinstance(kv, dict):
                    continue
                ok = {k: clean_value(k, v) for k, v in kv.items() if isinstance(k, str)}
                ok = {k: v for k, v in ok.items() if v is not None}
                if ok:
                    s.nodes[nid[:60]] = ok
        if isinstance(d.get('preset'), str):
            s.preset = d['preset'] if d['preset'] in dict(PRESETS) else ''
        for e in s.events:
            e['t_s'] = min(e['t_s'], s.duration_s)
        return s


def settings_of(project) -> ClockTreeSettings:
    return ClockTreeSettings.from_dict(getattr(project.analysis, 'clock_tree', None))


# ---------------------------------------------------------------- topology
@dataclass
class Node:
    id: str                          # 'gnss', 'server', 'compute', 'switch', 'deser:A', 'cam:<uid>', 'sensor:<uid>'
    kind: str
    label: str
    parent: str                      # '' for the root
    uids: list = field(default_factory=list)
    slot: int = -1                   # a camera's colour slot
    p: dict = field(default_factory=dict)          # resolved parameters
    typical: dict = field(default_factory=dict)    # key -> (typical value, source)
    edited: tuple = ()               # keys the user changed
    info: dict = field(default_factory=dict)       # fixed facts (fps, trigger line, model, interface …)
    note: str = ''

    @property
    def clock(self) -> bool:
        """Has a clock of its own in the simulation (switches relay, cameras and host-stamped sensors borrow)."""
        if self.kind in ('server', 'compute', 'deser'):
            return True
        return self.kind in SENSOR_KINDS and self.p.get('stamping') in ('ptp', 'pps')

    def params(self) -> tuple:
        return KIND_PARAMS.get(self.kind, ())


@dataclass
class Topology:
    nodes: list
    notes: list
    lidar: str = ''                  # the reference lidar's node id ('' = none)
    gm: str = ''                     # the grandmaster at the start
    variant_uid: str = ''

    def node(self, nid) -> Node | None:
        return next((n for n in self.nodes if n.id == nid), None)

    def children(self, nid) -> list:
        return [n for n in self.nodes if n.parent == nid]

    def ancestors(self, nid) -> list:
        """[nid, parent, grandparent, …, root]."""
        out, seen = [], set()
        n = self.node(nid)
        while n is not None and n.id not in seen:
            out.append(n.id)
            seen.add(n.id)
            n = self.node(n.parent) if n.parent else None
        return out

    def path_links(self, a, b) -> list:
        """The PTP links between nodes a and b: the child node ids of the tree edges on the way (PPS cables left
        out)."""
        A, B = self.ancestors(a), self.ancestors(b)
        common = next((x for x in A if x in B), None)
        links = A[:A.index(common)] if common in A else A
        links += B[:B.index(common)] if common in B else B
        return [x for x in links if self.node(x) is not None and self.node(x).parent not in ('gnss', '')]

    def sync_ms(self, nid) -> float:
        """The sync interval a PTP clock updates at: its nearest switch's (default 125 ms)."""
        for a in self.ancestors(nid)[1:]:
            n = self.node(a)
            if n is not None and n.kind == 'switch':
                return float(n.p['sync_ms'])
        return DEFAULT_SYNC_MS

    def gm_capable(self) -> list:
        return sorted((n for n in self.nodes if n.kind in ('server', 'compute') and n.p.get('priority', 255) < 255),
                      key=lambda n: (n.p['priority'], n.id != 'server'))


def _variant(project, variant):
    var = project.variant_by_uid(variant) if isinstance(variant, str) else variant
    return var or project.variant


def _power(project):
    from .power import PowerSettings
    return PowerSettings.from_dict(getattr(project.analysis, 'power', None))


def build_topology(project, variant=None, settings: ClockTreeSettings | None = None) -> Topology:
    """The rig's clock tree with every parameter resolved (the typical defaults, then the settings' edits)."""
    st = settings or settings_of(project)
    var = _variant(project, variant)
    pw = _power(project)
    tst = TM.TimingSettings.from_dict(getattr(project.analysis, 'timing', None))
    nodes, notes = [], []

    def add(nid, kind, label, parent, uids=(), slot=-1, model=None, interface='', info=None, note='', force=None):
        typ = _typical(kind, model, interface)
        if kind == 'camera':
            typ['readout_ms'] = (tst.readout_ms, S_TIMING)
            typ['exposure_ms'] = (tst.exposure_ms, S_TIMING)
        p = {k: v for k, (v, _) in typ.items()}
        edits = st.nodes.get(nid, {})
        edited = []
        for k in KIND_PARAMS[kind]:
            if k in edits:
                v = clean_value(k, edits[k])
                if k == 'stamping' and v not in STAMPING.get(kind, ()):
                    v = None
                if v is not None:
                    p[k] = v
                    edited.append(k)
        for k, v in (force or {}).items():
            p[k] = v
        n = Node(nid, kind, label, parent, list(uids), slot, p, typ, tuple(edited), dict(info or {}), note)
        nodes.append(n)
        return n

    gn = [s for s in var.sensors if s.visible and s.kind == 'gnss']
    ins = [s for s in var.sensors if s.visible and s.kind == 'imu' and project.sensor_model_for(s).antennas > 0]
    have_gnss = bool(gn or ins)
    add('gnss', 'gnss', 'GNSS receiver' + (f' · {" ".join(s.tag for s in gn + ins)}' if have_gnss else ' · assumed'),
        '', [s.uid for s in gn + ins],
        note='' if have_gnss else 'No GNSS antenna in the variant: a GNSS timing receiver is assumed.')
    if not have_gnss:
        notes.append('The variant has no GNSS antenna: a GNSS timing receiver giving the PPS is assumed.')
    server = pw.ptp_w > 0
    switch = pw.switch_w > 0
    ecu = project.ecu
    if server:
        add('server', 'server', 'Time server', 'gnss', info={'role': 'grandmaster'})
    eth_root = 'server' if server else 'compute'
    if switch:
        add('switch', 'switch', 'Ethernet switch', eth_root)
    add('compute', 'compute', f'{ecu.name}' if ecu.name else 'Compute',
        ('switch' if switch else 'server') if server else 'gnss', info={'location': ecu.location})
    if not server:
        nodes[-1].typical['priority'] = (246.0, S_PRIO_SERVER)
        if 'priority' not in nodes[-1].edited:
            nodes[-1].p['priority'] = 246.0
    eth = 'switch' if switch else eth_root
    # deserializers and cameras
    plan = None
    try:
        plan = TM.plan(project, var, tst)
    except Exception:                    # the sync plan is only for the trigger phases; without it they start at 0
        plan = None
    cams = [c for c in var.cameras if c.visible]
    by_d = {}
    for c in cams:
        d = TM.deserializer_of(c.port) or 'A'
        by_d.setdefault(d, []).append(c)
    for d in sorted(by_d):
        add(f'deser:{d}', 'deser', f'Deserializer {d}', 'compute',
            info={'ports': f'GMSL ports {"1–4" if d == "A" else "5–8"}'})
    for d in sorted(by_d):
        for c in by_d[d]:
            mod = project.module_for(c)
            line = TM._letter(c)
            trig = bool(mod.hw_sync) and bool(line)
            pc = plan.cam(c.uid) if plan is not None else None
            tms = pc.get('trigger_ms') if pc is not None else None
            note = ''
            if not c.port:
                note = f'{c.tag} has no GMSL port; counted on deserializer A.'
            if not trig:
                note = (note + ' ' if note else '') + (
                    f'{c.tag}: the module has no hardware trigger input; it free-runs and is stamped on arrival.'
                    if not mod.hw_sync else f'{c.tag} is on no trigger line; it free-runs and is stamped on arrival.')
            n = add(f'cam:{c.uid}', 'camera', f'{c.tag} · {c.name}', f'deser:{d}', [c.uid], c.slot,
                    info={'tag': c.tag, 'fps': float(mod.fps), 'line': line, 'triggered': trig, 'port': c.port,
                          'trigger_ms': float(tms) if tms is not None else 0.0, 'module': mod.name,
                          'shutter': mod.shutter or 'not given'},
                    note=note.strip(), force=None if trig else {'stamping': 'arrival'})
            if note:
                notes.append(note.strip())
    # lidars, radars, IMUs
    for s in var.sensors:
        if not s.visible or s.kind not in SENSOR_KINDS:
            continue
        m = project.sensor_model_for(s)
        nid = f'sensor:{s.uid}'
        typ = _typical(s.kind, m, m.interface)
        stamping = clean_value('stamping', st.nodes.get(nid, {}).get('stamping'))
        if stamping not in STAMPING[s.kind]:
            stamping = typ['stamping'][0]
        parent = eth if stamping == 'ptp' else 'gnss' if stamping == 'pps' else 'compute'
        add(nid, s.kind, f'{s.tag} · {m.name}', parent, [s.uid], model=m, interface=m.interface,
            info={'tag': s.tag, 'model': m.name, 'interface': m.interface, 'time_sync': m.time_sync,
                  'spinning': bool(m.spinning) if s.kind == 'lidar' else False})
    order = {'gnss': 0, 'server': 1, 'switch': 2, 'compute': 3}
    nodes.sort(key=lambda n: (order.get(n.id, 4 if n.kind in SENSOR_KINDS else 5 if n.kind == 'deser' else 6)))
    topo = Topology(nodes, notes, variant_uid=var.uid)
    if plan is not None and plan.lidar is not None:
        topo.lidar = f'sensor:{plan.lidar.uid}'
    else:
        topo.lidar = next((n.id for n in nodes if n.kind == 'lidar'), '')
    cap = topo.gm_capable()
    topo.gm = cap[0].id if cap else ''
    if not cap:
        notes.append('No grandmaster-capable clock (every priority1 is 255): every PTP clock free-runs.')
    elif len(cap) == 1:
        notes.append(f'{cap[0].label} is the only grandmaster-capable clock: losing it leaves every PTP clock in '
                     f'holdover.')
    return topo


# ---------------------------------------------------------------- the simulation
@dataclass
class Sim:
    """One seeded Monte Carlo over the timeline: clock errors (s) on the time grid, per run."""
    topo: Topology
    settings: ClockTreeSettings
    events: list
    dt: float
    t: np.ndarray                    # (N+1,) s
    X: dict                          # clock node id -> (R, N+1) error just before its servo update at each step
    draws: dict                      # node id -> {'y': (R,) holdover frequency, 'a': (R,) aging s/s², 'eps': (R,)}
    steps: dict                      # node id -> list of (t, (R,) step sizes s, 0 where none)
    holdover: dict                   # node id -> list of [start, end] s spent without a reference
    gm: list                         # the grandmaster's id at each step ('' = none)
    servo: dict                      # node id -> {'bandwidth_hz', 'capped', 'ts_s'}
    seconds: float = 0.0

    @property
    def runs(self) -> int:
        return next(iter(self.X.values())).shape[0] if self.X else self.settings.runs

    def at(self, nid, times) -> np.ndarray:
        """A clock's error at times (M,) by linear interpolation: (R, M)."""
        Xg = self.X[nid]
        N = Xg.shape[1] - 1
        f = np.clip(np.asarray(times, float) / self.dt, 0.0, N)
        i = np.minimum(np.floor(f).astype(np.intp), N - 1)
        w = f - i
        return Xg[:, i] * (1.0 - w) + Xg[:, i + 1] * w


def _in(t, a, b) -> bool:
    return a <= t < b


def _gm_sequence(topo: Topology, events, t, bmca_s) -> list:
    """The active grandmaster at each time ('' while BMCA waits out the announce timeout or none is left)."""
    cap = [n.id for n in topo.gm_capable()]
    losses = [e for e in events if e['kind'] == 'gm_loss']
    lost_spans = {}                      # node -> [(start, end)]
    cur = cap[0] if cap else ''
    # who is lost by each loss event: the grandmaster active at its time (worked out in time order)
    seq, wait_until = [], None
    pending = sorted(losses, key=lambda e: e['t_s'])

    def lost(nid, tt):
        return any(a <= tt < b for a, b in lost_spans.get(nid, ()))

    def best(tt):
        return next((n for n in cap if not lost(n, tt)), '')
    for tt in t:
        while pending and pending[0]['t_s'] <= tt + 1e-12:
            e = pending.pop(0)
            if cur:
                end = e['t_s'] + e['dur_s'] if e['dur_s'] > 0 else math.inf
                lost_spans.setdefault(cur, []).append((e['t_s'], end))
        b = best(tt)
        if b != cur:
            if cur and lost(cur, tt):
                if wait_until is None:
                    start = max(a for a, bb in lost_spans[cur] if a <= tt < bb)
                    wait_until = start + bmca_s
                if tt + 1e-12 >= wait_until:
                    cur, wait_until = b, None
                else:
                    seq.append('')
                    continue
            else:                        # a better grandmaster is back (or appears): it takes over at once
                cur, wait_until = b, None
        seq.append(cur)
    return seq


def _asym_steps(events) -> dict:
    """node id -> [(t, extra asymmetry ns)] from the asymmetry events."""
    out = {}
    for e in events:
        if e['kind'] == 'asymmetry' and e['node']:
            out.setdefault(e['node'], []).append((e['t_s'], e['value']))
    return out


def simulate(topo: Topology, st: ClockTreeSettings, events=None, progress=None, cancel=None) -> Sim | None:
    """The clocks over the timeline, every run at once (vectorised over runs). None when cancelled."""
    t0 = time.perf_counter()
    events = [clean_event(e) for e in (st.events if events is None else events)]
    events = [e for e in events if e is not None]
    rng = np.random.default_rng(st.seed)
    R = int(st.runs)
    D = float(st.duration_s)
    clocks = [n for n in topo.nodes if n.clock and n.kind != 'deser']
    syncs = [topo.sync_ms(n.id) for n in clocks if n.parent != 'gnss'] or [DEFAULT_SYNC_MS]
    dt = min(min(syncs) / 1000.0, 0.125)
    dt = max(dt, D / 40000.0, 0.005)
    N = max(int(round(D / dt)), 2)
    dt = D / N
    tg = np.arange(N + 1) * dt
    k_pps = max(int(round(1.0 / dt)), 1)
    gm_seq = _gm_sequence(topo, events, tg, st.bmca_s)
    drops = [(e['t_s'], e['t_s'] + e['dur_s']) for e in events if e['kind'] == 'pps_dropout']
    asym_ev = _asym_steps(events)
    pps_sigma = topo.node('gnss').p['pps_ns'] * 1e-9 if topo.node('gnss') else 0.0
    # draws, in the tree's order (deserializers included), so the stream does not depend on the run
    draws = {}
    for n in topo.nodes:
        if n.clock:
            y = rng.normal(0.0, n.p['ppm'] * 1e-6, R)
            a = (rng.integers(0, 2, R) * 2 - 1) * n.p['aging_ppb_day'] * 1e-9 / 86400.0
            eps = rng.random(R)
            draws[n.id] = {'y': y, 'a': a, 'eps': eps}
    state = {}
    servo = {}
    for n in clocks:
        pps = n.parent == 'gnss'
        Ts = k_pps * dt if pps else max(int(round(topo.sync_ms(n.id) / 1000.0 / dt)), 1) * dt
        w_raw = 2 * math.pi * n.p['servo_hz'] * Ts
        w = min(w_raw, MAX_WT)
        servo[n.id] = {'bandwidth_hz': w / (2 * math.pi * Ts), 'capped': w_raw > MAX_WT, 'ts_s': Ts}
        state[n.id] = {'x': np.zeros(R), 'I': np.zeros(R), 'u': np.zeros(R), 'hold': False, 'th1': None,
                       'k_up': max(int(round(Ts / dt)), 1), 'Ts': Ts, 'Kp': 2 * DAMPING * w / Ts,
                       'Ki': w * w / (Ts * Ts), 'pps': pps}
    X = {n.id: np.empty((R, N + 1)) for n in clocks}
    steps = {n.id: [] for n in clocks}
    hold_spans = {n.id: [] for n in topo.nodes if n.clock}
    links = {}

    def bias_sigma(nid, ref, tt):
        key = (nid, ref)
        if key not in links:
            ls = topo.path_links(nid, ref)
            links[key] = (ls, math.sqrt(sum(topo.node(x).p['jitter_ns'] ** 2 for x in ls)) * 1e-9)
        ls, sig = links[key]
        b = 0.0
        for x in ls:
            a = topo.node(x).p['asym_ns'] + sum(v for ts, v in asym_ev.get(x, ()) if ts <= tt)
            b += a / 2.0
        return b * 1e-9, sig

    pps_now = np.zeros(R)
    for k in range(N + 1):
        tt = tg[k]
        if k % k_pps == 0:
            pps_now = rng.normal(0.0, pps_sigma, R)
        dropped = any(_in(tt, a, b) for a, b in drops)
        gm = gm_seq[k]
        now = {nid: s_['x'] for nid, s_ in state.items()}  # every clock at t_k, before any moves on to t_k+1
        for n in clocks:
            s = state[n.id]
            X[n.id][:, k] = s['x']
            if k == N:
                continue
            if k % s['k_up'] == 0:
                if s['pps']:
                    ref = None if dropped else 'pps'
                elif n.id == gm or not gm:
                    ref = None
                else:
                    ref = gm
                was = s['hold']
                if ref is None:
                    s['hold'] = True
                else:
                    s['hold'] = False
                    if ref == 'pps':
                        e = s['x'] - pps_now
                    else:
                        b, sig = bias_sigma(n.id, ref, tt)
                        e = s['x'] - now[ref] - b + rng.normal(0.0, sig, R)
                    if n.p['policy'] == 'step':
                        big = np.abs(e) > n.p['step_us'] * 1e-6
                        if big.any():
                            steps[n.id].append((tt, np.where(big, -e, 0.0)))
                            s['x'] = np.where(big, s['x'] - e, s['x'])
                            e = np.where(big, 0.0, e)
                    I_new = s['I'] + s['Ki'] * s['Ts'] * e
                    u = s['Kp'] * e + I_new
                    if n.p['policy'] == 'slew':
                        lim = n.p['slew_ppm'] * 1e-6
                        clip = np.abs(u) > lim
                        u = np.clip(u, -lim, lim)
                        I_new = np.where(clip, s['I'], I_new)
                    s['I'], s['u'] = I_new, u
                if s['hold'] and not was:
                    hold_spans[n.id].append([tt, None])
                    if s['th1'] is None:
                        s['th1'] = tt
                elif was and not s['hold']:
                    hold_spans[n.id][-1][1] = tt
            dr = draws[n.id]
            yo = 0.0 if s['th1'] is None else dr['y'] + dr['a'] * (tt + dt / 2 - s['th1'])
            s['x'] = s['x'] + dt * (yo - (s['I'] if s['hold'] else s['u']))
        if progress is not None and k % 64 == 0:
            progress(0.7 * k / N)
        if cancel is not None and k % 64 == 0 and cancel():
            return None
    for n in clocks:
        if hold_spans[n.id] and hold_spans[n.id][-1][1] is None:
            hold_spans[n.id][-1][1] = D
    # deserializer FSYNC clocks
    for n in topo.nodes:
        if n.kind != 'deser':
            continue
        dr = draws[n.id]
        if n.p['fsync'] == 'free' or 'compute' not in X:
            X[n.id] = dr['y'][:, None] * tg[None, :] + 0.5 * dr['a'][:, None] * tg[None, :] ** 2
            hold_spans[n.id] = [[0.0, D]]
        elif n.p['rephase_s'] <= 0:
            X[n.id] = X['compute']
        else:
            tr = np.floor(tg / n.p['rephase_s'] + 1e-9) * n.p['rephase_s']
            kr = np.minimum(np.round(tr / dt).astype(np.intp), N)
            X[n.id] = X['compute'][:, kr] + dr['y'][:, None] * (tg - tr)[None, :]
    return Sim(topo, st, events, dt, tg, X, draws, steps, hold_spans, gm_seq, servo, time.perf_counter() - t0)


# ---------------------------------------------------------------- sensor timestamps
def frame_times(node: Node, duration_s: float) -> np.ndarray:
    """The sensor's frame (or message) times over the timeline, s: a camera's at its trigger phase from the sync plan
    every frame period, the others' at their rate from 0."""
    if node.kind == 'camera':
        fps = max(node.info.get('fps', 30.0), 0.1)
        T = 1.0 / fps
        ph = (node.info.get('trigger_ms', 0.0) / 1000.0) % T
    else:
        T = 1.0 / max(node.p.get('rate_hz', 10.0), 0.1)
        ph = 0.0
    n = int(math.floor((duration_s - ph) / T - 1e-9)) + 1
    return ph + np.arange(max(n, 0)) * T


def _event_frames(sim: Sim, node: Node, times) -> dict:
    """frame index -> ('missed' | 'late', value s) for the FSYNC events that reach this camera."""
    out = {}
    for e in sim.events:
        if e['kind'] not in ('fsync_missed', 'fsync_late') or e['node'] not in (node.id, node.parent):
            continue
        j = int(np.searchsorted(times, e['t_s'] - 1e-12))
        if j < len(times):
            out[j] = ('missed', 0.0) if e['kind'] == 'fsync_missed' else ('late', e['value'] / 1000.0)
    return out


def quantisation(times, frame_s, tick_s, rephase_s, eps, free=False, cols=None) -> np.ndarray:
    """Each FSYNC edge's offset from its ideal time (R, len(cols)), s, for a timer of tick tick_s: every edge on the
    nearest tick (rephase_s 0), or a whole number of ticks per frame counted from the last re-phase (each rephase_s,
    or never when the timer runs free). eps (R,) is the tick grid's phase against true time, in ticks; cols the frame
    indices wanted (all by default)."""
    times = np.asarray(times, float)
    cols = np.arange(len(times)) if cols is None else np.asarray(cols, np.intp)
    R = len(eps)
    if tick_s <= 0 or not len(cols):
        return np.zeros((R, len(cols)))
    off = (np.asarray(eps, float) * tick_s)[:, None]

    def nearest(tau):
        return tick_s * np.round((tau[None, :] - off) / tick_s) + off - tau[None, :]
    if rephase_s <= 0 and not free:
        return nearest(times[cols])
    if free:
        j0 = np.zeros(len(cols), np.intp)
    else:
        tr = np.floor(times[cols] / rephase_s + 1e-9) * rephase_s
        j0 = np.searchsorted(times, tr - 1e-12)
    Nt = max(int(round(frame_s / tick_s)), 1)
    return nearest(times[j0]) + (cols - j0)[None, :] * (Nt * tick_s - frame_s)


def stamps(sim: Sim, nid: str, stride: int = 1, keep=()) -> tuple:
    """(times (M,), stamp error (R, M) s: the stamp minus the true capture time) of a sensor node; frames strided
    (every stride-th, plus the frames in keep and those an FSYNC event hits)."""
    topo, st = sim.topo, sim.settings
    n = topo.node(nid)
    times = frame_times(n, st.duration_s)
    ev = _event_frames(sim, n, times) if n.kind == 'camera' else {}
    idx = np.arange(0, len(times), max(int(stride), 1))
    extra = [j for j in list(ev) + list(keep) if 0 <= j < len(times)]
    if extra:
        idx = np.union1d(idx, np.array(extra, np.intp))
    tt = times[idx]
    R = sim.runs
    sub = np.random.default_rng([int(st.seed), _hash(_seed_key(n))])  # per sensor, so strides do not shift others
    if n.kind == 'camera':
        if n.p['stamping'] == 'arrival':
            s = sim.at('compute', tt) + n.p['latency_us'] * 1e-6 + sub.normal(0.0, n.p['host_us'] * 1e-6, (R, len(tt)))
            return tt, s
        d = topo.node(n.parent)
        frame = 1.0 / max(n.info.get('fps', 30.0), 0.1)
        q = quantisation(times, frame, d.p['tick_us'] * 1e-6, d.p['rephase_s'], sim.draws[d.id]['eps'],
                         free=d.p['fsync'] == 'free', cols=idx)
        s = sim.at(d.id, tt) - q
        for j, (what, v) in ev.items():              # a missed pulse: the next edge's image carries this stamp
            col = int(np.searchsorted(idx, j))
            s[:, col] -= frame if what == 'missed' else v
        return tt, s
    if n.p['stamping'] == 'host':
        s = sim.at('compute', tt) + n.p['latency_us'] * 1e-6 + sub.normal(0.0, n.p['host_us'] * 1e-6, (R, len(tt)))
        return tt, s
    return tt, sim.at(nid, tt) + sub.normal(0.0, n.p['jitter_ns'] * 1e-9, (R, len(tt)))


def _seed_key(n: Node) -> str:
    """A node's name for seeding its random stream: its kind and label (the sensor's tag), not its id, which carries
    the sensor's uid: a project built again (the sample, an import) gets new uids and must draw the same numbers."""
    return f'{n.kind}:{n.label}'


def _hash(s: str) -> int:
    h = 2166136261
    for ch in s.encode('utf-8'):
        h = ((h ^ ch) * 16777619) & 0xFFFFFFFF
    return h


def _stats(a) -> dict:
    a = np.abs(np.asarray(a, float)).ravel()
    if not len(a):
        return {'p50': None, 'p95': None, 'max': None}
    p50, p95 = np.percentile(a, [50, 95])
    return {'p50': float(p50), 'p95': float(p95), 'max': float(a.max())}


def _series(tt, err, D, bins) -> dict:
    """Per time bin: p50, p95 and max of |err| over runs and the frames in the bin (s)."""
    a = np.abs(err)
    b = np.minimum((tt / D * bins).astype(np.intp), bins - 1)
    p50 = np.full(bins, np.nan)
    p95 = np.full(bins, np.nan)
    mx = np.full(bins, np.nan)
    order = np.argsort(b, kind='stable')
    bs = b[order]
    cuts = np.searchsorted(bs, np.arange(bins + 1))
    for i in range(bins):
        cols = order[cuts[i]:cuts[i + 1]]
        if len(cols):
            v = a[:, cols].ravel()
            p50[i], p95[i] = np.percentile(v, [50, 95])
            mx[i] = v.max()
    return {'p50': p50, 'p95': p95, 'max': mx}


# ---------------------------------------------------------------- projection
def vehicle_motion(dt_s, speed_mps, yaw_rate_rad):
    """The vehicle's pose change over dt (arrays broadcast): rotation angle θ = ω·dt about z and the translation along
    the arc (v/ω·sin θ, v/ω·(1 − cos θ), 0) (v·dt along x when ω·dt is 0), in the frame at the start."""
    dt = np.asarray(dt_s, float)
    th = yaw_rate_rad * dt
    small = np.abs(th) < 1e-9
    w = yaw_rate_rad if abs(yaw_rate_rad) > 0 else 1.0
    tx = np.where(small, speed_mps * dt * (1 - th * th / 6), speed_mps / w * np.sin(th))
    ty = np.where(small, speed_mps * dt * th / 2, speed_mps / w * (1 - np.cos(th)))
    return th, np.stack([tx, ty, np.zeros_like(tx)], -1)


def _moved(P, th, T):
    """Static points P (K, 3) seen from the vehicle after it turned th and moved T (per point or shared):
    R(th)ᵀ·(P − T)."""
    c, s = np.cos(th), np.sin(th)
    Q = P - T
    return np.stack([c * Q[..., 0] + s * Q[..., 1], -s * Q[..., 0] + c * Q[..., 1], Q[..., 2]], -1)


def image_samples(intr, grid=GRID):
    """Pixel samples (u, v) over the image inside the lens circle, the principal point first (as core.timing)."""
    gu, gv = grid
    U, V = np.meshgrid(np.linspace(0.5, intr.width - 0.5, gu), np.linspace(0.5, intr.height - 0.5, gv))
    u = np.r_[intr.cx, U.ravel()]
    v = np.r_[intr.cy, V.ravel()]
    d, th = intr.unproject(u, v)
    keep = th <= intr.max_theta + 1e-9
    keep[0] = True
    return u, v, d, keep


def projection_error(project, cam, dt_s, speed_mps, yaw_rate_dps, surface='range', range_m=20.0, grid=GRID,
                     readout_ms=0.0, rows_compensated=True) -> dict:
    """How far a static point moves in the camera image when the lidar-camera time difference is wrong by dt_s at
    speed and yaw rate: per image sample, the pixel distance between the point projected where the stamps put it and
    where it is (the real lens model). Points on each pixel's ray range_m from the camera, or on the ground (z = 0).
    {'u', 'v', 'err' (px, NaN without a point), 'du', 'dv', 'centre_px' (None when the centre has no point),
    'worst_px', 'P' (vehicle frame)}."""
    mod, lens = project.module_for(cam), project.lens_for(cam)
    intr = resolve(mod, lens)
    _, W2O = camera_frames(cam)
    u, v, d, keep = image_samples(intr, grid)
    D = d @ W2O                                           # vehicle-frame ray directions
    c = cam.pos()
    if surface == 'ground':
        with np.errstate(divide='ignore', invalid='ignore'):
            r = np.where(D[:, 2] < -1e-9, -c[2] / D[:, 2], np.nan)
        r = np.where((r > 0) & (r <= GROUND_MAX_M), r, np.nan)
    else:
        r = np.full(len(u), float(range_m))
    ok = keep & np.isfinite(r)
    P = c + np.where(ok, r, 0.0)[:, None] * D
    dt = np.full(len(u), float(dt_s))
    if not rows_compensated and readout_ms:
        dt = dt + (v / intr.height - 0.5) * readout_ms / 1000.0
    th, T = vehicle_motion(dt, speed_mps, math.radians(yaw_rate_dps))
    Q = _moved(P, th, T)
    ua, va, _, _ = intr.project_dirs((P - c) @ W2O.T)
    ub, vb, _, _ = intr.project_dirs((Q - c) @ W2O.T)
    du, dv = ub - ua, vb - va
    err = np.where(ok, np.hypot(du, dv), np.nan)
    with np.errstate(invalid='ignore'):
        worst = float(np.nanmax(err)) if np.isfinite(err).any() else None
    return {'u': u, 'v': v, 'err': err, 'du': np.where(ok, du, np.nan), 'dv': np.where(ok, dv, np.nan),
            'centre_px': float(err[0]) if ok[0] else None, 'worst_px': worst, 'P': P, 'ok': ok,
            'grid': grid, 'width': intr.width, 'height': intr.height}


def slip(project, cam, dt_ms, speed_kmh, yaw_rate_dps, surface='range', range_m=20.0) -> dict:
    """The slip calculator: a dt_ms time error at speed_kmh turning yaw_rate_dps; the rotation ω·Δt (deg), the
    translation v·Δt (mm) and the error at the image centre and worst over the image (px)."""
    pe = projection_error(project, cam, dt_ms / 1000.0, speed_kmh / 3.6, yaw_rate_dps, surface, range_m)
    return {'rotation_deg': yaw_rate_dps * dt_ms / 1000.0, 'translation_mm': speed_kmh / 3.6 * dt_ms,
            'centre_px': pe['centre_px'], 'worst_px': pe['worst_px']}


# ---------------------------------------------------------------- the study
@dataclass
class ClockTreeResult:
    settings: ClockTreeSettings
    topology: Topology
    events: list
    sensors: list                    # per sensor node: stats against true time and the reference lidar
    clocks: list                     # per clock node: error against true time, steps, holdover
    series: dict                     # {'t': (bins,) s, node id -> {'p50','p95','max'} (bins,) s; 'rel:<id>' vs lidar}
    projection: list                 # per camera: Δt against the lidar and the image error it makes
    findings: list
    metrics: dict
    variant: str = ''
    variant_uid: str = ''
    seconds: float = 0.0
    sim: Sim | None = None           # the clocks (for the page and tests; not exported)

    def sensor(self, nid):
        return next((s for s in self.sensors if s['id'] == nid), None)

    def camera(self, uid_or_id):
        return next((c for c in self.projection if c['id'] == uid_or_id or c['uid'] == uid_or_id), None)

    def worst(self) -> str:
        return self.findings[0]['severity'] if self.findings else 'good'


def holdover_time(ppm, aging_ppb_day, budget_s, k=1.96) -> float:
    """Seconds a clock in holdover stays within budget_s at k σ of its frequency (ppm, 1σ) plus its aging:
    the root of k·σ·t + ½·a·t² = budget (inf for a perfect oscillator)."""
    y = k * ppm * 1e-6
    a = aging_ppb_day * 1e-9 / 86400.0
    if a <= 0:
        return budget_s / y if y > 0 else math.inf
    return (-y + math.sqrt(y * y + 2 * a * budget_s)) / a


def run(project, engine=None, settings: ClockTreeSettings | None = None, variant=None, progress=None,
        cancel=None) -> ClockTreeResult | None:
    """The clock tree study: the topology, the Monte Carlo over the scenario, every sensor's timestamp error against
    true time and the reference lidar, and the projection error it makes in each camera. engine is accepted for the
    module contract and not needed. None when cancelled."""
    t0 = time.perf_counter()
    st = settings or settings_of(project)
    var = _variant(project, variant)
    prog = progress or (lambda f: None)
    topo = build_topology(project, var, st)
    sim = simulate(topo, st, progress=prog, cancel=cancel)
    if sim is None:
        return None
    R, D = sim.runs, st.duration_s
    bins = int(min(240, max(20, D * 2)))
    series = {'t': (np.arange(bins) + 0.5) * D / bins}
    sens = [n for n in topo.nodes if n.kind in SENSOR_KINDS + ('camera',)]
    lid = topo.node(topo.lidar) if topo.lidar else None
    rows = []
    rel = {}
    for i, n in enumerate(sens):
        M = len(frame_times(n, D))
        stride = max(int(math.ceil(R * M / MAX_SAMPLES)), 1)
        tt, s = stamps(sim, n.id, stride)
        row = {'id': n.id, 'kind': n.kind, 'label': n.label, 'tag': n.info.get('tag', n.label), 'uids': n.uids,
               'slot': n.slot, 'stamping': n.p.get('stamping'), 'frames': M, 'stride': stride,
               'true': _stats(s), 'vs_lidar': None}
        series[n.id] = _series(tt, s, D, bins)
        if lid is not None and n.id != lid.id:
            _, sl = _lidar_at(sim, lid, tt)
            d = s - sl
            row['vs_lidar'] = _stats(d)
            rel[n.id] = d
            series['rel:' + n.id] = _series(tt, d, D, bins)
        if n.kind == 'camera':
            frame = 1.0 / max(n.info.get('fps', 30.0), 0.1)
            row['frames_off'] = float(np.mean(np.sum(np.abs(s) > frame / 2, axis=1)))
        rows.append(row)
        prog(0.7 + 0.15 * (i + 1) / max(len(sens), 1))
        if cancel is not None and cancel():
            return None
    clocks = []
    for n in topo.nodes:
        if n.id not in sim.X:
            continue
        x = sim.X[n.id]
        st_ = sim.steps.get(n.id, [])
        sizes = np.concatenate([s for _, s in st_]) if st_ else np.zeros(0)
        big = sizes[np.abs(sizes) > 0]
        clocks.append({'id': n.id, 'kind': n.kind, 'label': n.label, 'true': _stats(x),
                       'holdover': [[a, b] for a, b in sim.holdover.get(n.id, [])],
                       'steps_per_run': float(sum(np.count_nonzero(s) for _, s in st_) / R),
                       'max_step_us': float(np.abs(big).max() * 1e6) if len(big) else 0.0,
                       'backward': bool((big < 0).any()) if len(big) else False,
                       'servo': sim.servo.get(n.id)})
        series['clock:' + n.id] = _series(sim.t, x, D, bins)
    projection = _projection(project, var, topo, st, rows, rel)
    prog(0.95)
    res = ClockTreeResult(st, topo, sim.events, rows, clocks, series, projection, [], {}, var.name, var.uid,
                          sim=sim)
    res.metrics = _metrics(res)
    res.findings = _findings(res, project)
    res.seconds = time.perf_counter() - t0
    with _LOCK:
        if len(_LAST) > 8:
            _LAST.pop(next(iter(_LAST)))
        _LAST[_key(project, var, st)] = [f for f in res.findings if f['kind'] == 'projection']
    prog(1.0)
    return res


def _lidar_at(sim: Sim, lid: Node, tt):
    """The reference lidar's stamp error at times tt (its clock interpolated, its stamp jitter added)."""
    sub = np.random.default_rng([int(sim.settings.seed), _hash(_seed_key(lid)), 7])
    R = sim.runs
    if lid.p['stamping'] == 'host':
        return tt, sim.at('compute', tt) + lid.p['latency_us'] * 1e-6 + sub.normal(0.0, lid.p['host_us'] * 1e-6,
                                                                                   (R, len(tt)))
    return tt, sim.at(lid.id, tt) + sub.normal(0.0, lid.p['jitter_ns'] * 1e-9, (R, len(tt)))


def _projection(project, var, topo: Topology, st: ClockTreeSettings, rows, rel) -> list:
    out = []
    plan = None
    try:
        plan = TM.plan(project, var)
    except Exception:                    # the plan's misalignment is extra information only
        plan = None
    for row in rows:
        if row['kind'] != 'camera':
            continue
        n = topo.node(row['id'])
        cam = var.cam(n.uids[0]) if hasattr(var, 'cam') else next(c for c in var.cameras if c.uid == n.uids[0])
        stat = row['vs_lidar'] or row['true']
        ref = 'lidar' if row['vs_lidar'] else 'true time'
        e = {'id': n.id, 'uid': cam.uid, 'tag': cam.tag, 'slot': cam.slot, 'against': ref,
             'dt_p50_ms': stat['p50'] * 1e3, 'dt_p95_ms': stat['p95'] * 1e3, 'dt_max_ms': stat['max'] * 1e3}
        for key, dt in (('p95', stat['p95']), ('max', stat['max'])):
            pe = projection_error(project, cam, dt, st.speed_mps, st.yaw_rate_dps, st.surface, st.range_m,
                                  readout_ms=n.p['readout_ms'], rows_compensated=st.rows_compensated)
            e[f'centre_{key}_px'] = pe['centre_px']
            e[f'worst_{key}_px'] = pe['worst_px']
        pc = plan.cam(cam.uid) if plan is not None else None
        mis = pc.get('misalign_ms') if pc is not None else None
        e['plan_misalign_ms'] = mis
        if mis is not None:
            pe = projection_error(project, cam, abs(mis) / 1000.0, st.speed_mps, st.yaw_rate_dps, st.surface,
                                  st.range_m)
            e['plan_worst_px'] = pe['worst_px']
            e['plan_centre_px'] = pe['centre_px']
            e['plan_travel_cm'] = pc.get('travel_cm')
        else:
            e['plan_worst_px'] = e['plan_centre_px'] = e['plan_travel_cm'] = None
        out.append(e)
    return out


def _metrics(res: ClockTreeResult) -> dict:
    st, topo = res.settings, res.topology
    pj = res.projection
    w95 = max(pj, key=lambda c: c['worst_p95_px'] or 0.0, default=None)
    wmx = max(pj, key=lambda c: c['worst_max_px'] or 0.0, default=None)
    gm = topo.node(topo.gm) if topo.gm else None
    hold = holdover_time(gm.p['ppm'], gm.p['aging_ppb_day'], st.time_budget_us * 1e-6) if gm is not None else 0.0
    gmc = next((c for c in res.clocks if c['id'] == topo.gm), None)
    rel = [s['vs_lidar']['p95'] for s in res.sensors if s['vs_lidar'] is not None]
    return {'worst_p95_px': w95['worst_p95_px'] if w95 else None, 'worst_p95_tag': w95['tag'] if w95 else None,
            'worst_max_px': wmx['worst_max_px'] if wmx else None, 'worst_max_tag': wmx['tag'] if wmx else None,
            'worst_dt_p95_ms': max((c['dt_p95_ms'] for c in pj), default=None),
            'worst_dt_max_ms': max((c['dt_max_ms'] for c in pj), default=None),
            'gm': gm.label if gm else None, 'holdover_s': hold,
            'gm_true_p95_us': gmc['true']['p95'] * 1e6 if gmc else None,
            'gm_true_max_us': gmc['true']['max'] * 1e6 if gmc else None,
            'worst_rel_p95_us': max(rel) * 1e6 if rel else None,
            'cameras': len(pj), 'sensors': len(res.sensors), 'runs': st.runs}


# ---------------------------------------------------------------- findings and issues
def _finding(out, sev, kind, message, uids=()):
    out.append({'severity': sev, 'kind': kind, 'message': message, 'uids': list(uids)})


def fmt_time(s) -> str:
    """A time error in words: '41 ns', '3.2 µs', '1.25 ms'."""
    if s is None:
        return '—'
    a = abs(s)
    if a < 1e-6:
        return f'{s * 1e9:.0f} ns'
    if a < 1e-3:
        return f'{s * 1e6:.1f} µs' if a < 1e-4 else f'{s * 1e6:.0f} µs'
    return f'{s * 1e3:.2f} ms' if a < 0.01 else f'{s * 1e3:.1f} ms'


def _motion_words(st):
    where = 'on the ground' if st.surface == 'ground' else f'{st.range_m:g} m away'
    turn = f' turning {abs(st.yaw_rate_dps):g}°/s' if st.yaw_rate_dps else ''
    return f'at {st.speed_kmh:g} km/h{turn}, points {where}'


def _findings(res: ClockTreeResult, project) -> list:
    out = []
    st, topo, m = res.settings, res.topology, res.metrics
    # projection
    for c in res.projection:
        w95, wmx = c['worst_p95_px'], c['worst_max_px']
        against = 'the lidar' if c['against'] == 'lidar' else 'true time'
        if w95 is not None and w95 > st.threshold_px:
            _finding(out, 'serious', 'projection',
                     f'{c["tag"]}: its timestamps are off {against} by up to {c["dt_p95_ms"]:.3g} ms in 95 % of '
                     f'frames ({c["dt_max_ms"]:.3g} ms at the worst): lidar points land up to {w95:.1f} px off in '
                     f'its image ({wmx:.1f} px at the worst) {_motion_words(st)}; the limit is {st.threshold_px:g} '
                     f'px.', [c['uid']])
        elif wmx is not None and wmx > st.threshold_px:
            row = res.sensor(c['id']) or {}
            which = ('the frame a missed FSYNC pulse mislabels' if row.get('frames_off', 0) >= 0.5 else
                     'the worst frame')
            _finding(out, 'warning', 'projection',
                     f'{c["tag"]}: in 95 % of frames lidar points land within {w95:.2f} px, but {which} is '
                     f'{wmx:.1f} px off ({c["dt_max_ms"]:.3g} ms against {against}) {_motion_words(st)}; the limit '
                     f'is {st.threshold_px:g} px.', [c['uid']])
    # holdover
    gm = topo.node(topo.gm) if topo.gm else None
    gmc = next((c for c in res.clocks if c['id'] == topo.gm), None)
    if gm is not None:
        hold = m['holdover_s']
        drop = [e for e in res.events if e['kind'] == 'pps_dropout']
        p95 = m['gm_true_p95_us']
        ex = gmc['true']['max'] * 1e6 if gmc else None
        span = max((e['dur_s'] for e in drop), default=0.0)
        if drop and ex is not None and ex > st.time_budget_us:
            _finding(out, 'serious', 'holdover',
                     f'{gm.label} drifts up to {ex:.0f} µs from UTC during the {span:g} s PPS dropout ({p95:.1f} µs '
                     f'p95 over the run): over the {st.time_budget_us:g} µs budget. Its oscillator ({gm.p["ppm"]:g} '
                     f'ppm) holds the budget for {hold:.0f} s (95 %).', [])
        elif hold < st.holdover_need_s:
            _finding(out, 'serious', 'holdover',
                     f'{gm.label} holds UTC within {st.time_budget_us:g} µs for {hold:.0f} s without PPS (95 %, '
                     f'{gm.p["ppm"]:g} ppm): less than the {st.holdover_need_s:g} s the rig must ride through. An '
                     f'oscillator of {_needed_ppm(st):.2g} ppm or better holds it.', [])
        else:
            _finding(out, 'good', 'holdover',
                     f'{gm.label} holds UTC within {st.time_budget_us:g} µs for {hold:.0f} s without PPS (95 %, '
                     f'{gm.p["ppm"]:g} ppm), more than the {st.holdover_need_s:g} s needed.', [])
    # FSYNC clocks running free
    for n in topo.nodes:
        if n.kind == 'deser' and n.p['fsync'] == 'free':
            cams = [s for s in res.sensors if s['kind'] == 'camera' and topo.node(s['id']).parent == n.id]
            w = max((s['vs_lidar'] or s['true'])['p95'] for s in cams) if cams else None
            _finding(out, 'warning', 'free_fsync',
                     f'{n.label}\'s FSYNC timer runs free on its own crystal ({n.p["ppm"]:g} ppm): its cameras drift '
                     f'{fmt_time(w)} (p95) from the lidar over {st.duration_s:g} s. Lock it to the compute\'s clock.',
                     [u for s in cams for u in s['uids']])
    # missed / late FSYNC pulses
    for e in res.events:
        if e['kind'] in ('fsync_missed', 'fsync_late'):
            n = topo.node(e['node'])
            if n is None:
                _finding(out, 'note', 'event', f'{describe_event(e)}: no such deserializer or camera in this variant.')
                continue
            cams = [x for x in topo.nodes if x.kind == 'camera' and (x.id == n.id or x.parent == n.id)]
            trig = [x for x in cams if x.p['stamping'] == 'trigger']
            if e['kind'] == 'fsync_missed':
                if trig:
                    fr = 1000.0 / max(trig[0].info.get('fps', 30.0), 0.1)
                    _finding(out, 'warning', 'fsync',
                             f'A missed FSYNC pulse on {n.label} at {e["t_s"]:g} s labels the next frame of '
                             f'{", ".join(x.info.get("tag", x.label) for x in trig)} with the missed pulse\'s time: '
                             f'one frame {fr:.1f} ms off. Check the frame counter against the trigger count.',
                             [u for x in trig for u in x.uids])
                else:
                    _finding(out, 'note', 'fsync', f'A missed FSYNC pulse on {n.label} at {e["t_s"]:g} s drops a '
                                                   f'frame; its cameras are stamped on arrival, so no frame is '
                                                   f'mislabelled.', [u for x in cams for u in x.uids])
            else:
                _finding(out, 'note' if not trig else 'warning', 'fsync',
                         f'An FSYNC pulse {e["value"]:g} ms late on {n.label} at {e["t_s"]:g} s exposes one frame '
                         f'{e["value"]:g} ms after its stamp' + (' (trigger-time stamping).' if trig else
                                                                 '; arrival stamps follow it.'),
                         [u for x in cams for u in x.uids])
    # steps backwards
    back = sorted((c for c in res.clocks if c['backward'] and c['max_step_us'] > 0), key=lambda c: -c['max_step_us'])
    if back:
        who = ', '.join(f'{topo.node(c["id"]).info.get("tag") or c["label"]} {c["max_step_us"]:.0f} µs'
                        for c in back[:6]) + (f' and {len(back) - 6} more' if len(back) > 6 else '')
        _finding(out, 'note', 'step',
                 f'{len(back)} clock{"s" * (len(back) != 1)} step back in time (at most: {who}): a pipeline that '
                 f'needs monotonic timestamps should use the slew policy.',
                 [u for c in back for u in topo.node(c['id']).uids])
    # gPTP accuracy goal in lock
    rel = [s for s in res.sensors if s['vs_lidar'] is not None and s['stamping'] == 'ptp']
    worst = max(rel, key=lambda s: s['vs_lidar']['p50'], default=None)
    if worst is not None and worst['vs_lidar']['p50'] * 1e6 > ACCURACY_US:
        _finding(out, 'note', 'gptp',
                 f'{worst["tag"]} is {fmt_time(worst["vs_lidar"]["p50"])} from the lidar half the time: over the '
                 f'1 µs IEEE 802.1AS-2020 aims for across 7 hops. Check the link asymmetry and the jitter.',
                 worst['uids'])
    for s in res.sensors:
        if s['stamping'] == 'host' and s['kind'] != 'camera' and s['vs_lidar'] is not None:
            _finding(out, 'note', 'host',
                     f'{s["tag"]} is stamped on the host\'s receive time: {fmt_time(s["vs_lidar"]["p95"])} from the '
                     f'lidar (p95). A PTP-capable interface stamps it at the source.', s['uids'])
    if not topo.lidar:
        _finding(out, 'note', 'no_lidar', 'No lidar in this variant: the cameras\' errors are against true time.')
    if not topo.gm_capable():
        _finding(out, 'warning', 'no_gm', 'No grandmaster-capable clock: every PTP clock free-runs.')
    elif len(topo.gm_capable()) == 1 and any(e['kind'] == 'gm_loss' for e in res.events):
        _finding(out, 'warning', 'no_backup',
                 f'{topo.gm_capable()[0].label} is the only grandmaster-capable clock: when it is lost every PTP '
                 f'clock holds over on its own crystal and they drift apart.')
    if not any(f['kind'] == 'projection' for f in out) and res.projection:
        _finding(out, 'good', 'projection',
                 f'Every camera keeps lidar points within {st.threshold_px:g} px ({m["worst_max_px"]:.2f} px at the '
                 f'worst, {m["worst_max_tag"]}) {_motion_words(st)}.')
    rank = {s: i for i, s in enumerate(SEVERITIES)}
    out.sort(key=lambda f: rank[f['severity']])
    return out


def _needed_ppm(st) -> float:
    return st.time_budget_us / max(st.holdover_need_s, 1e-9) / 1.96


_LOCK = threading.Lock()
_LAST: dict = {}


def _key(project, var, st) -> str:
    return json.dumps([var.uid, st.to_dict(), [c.to_dict() for c in var.cameras], [s.to_dict() for s in var.sensors],
                       getattr(project.analysis, 'timing', {}), getattr(project.analysis, 'power', {}),
                       project.ecu.to_dict()], sort_keys=True, default=str)


def issues(project, vg=None) -> list:
    """Cheap checks for the Issues list: the grandmaster's holdover against the time budget, FSYNC timers running
    free, and the last run's serious projection findings (95 % of frames over the limit) for exactly these inputs (no
    Monte Carlo here)."""
    from .rules import Issue
    st = settings_of(project)
    var = project.variant
    topo = build_topology(project, var, st)
    out = []
    gm = topo.node(topo.gm) if topo.gm else None
    if gm is not None:
        hold = holdover_time(gm.p['ppm'], gm.p['aging_ppb_day'], st.time_budget_us * 1e-6)
        if hold < st.holdover_need_s:
            out.append(Issue('warning', f'{gm.label} holds UTC within {st.time_budget_us:g} µs for only {hold:.0f} s '
                                        f'without PPS ({gm.p["ppm"]:g} ppm, 95 %): the rig must ride through '
                                        f'{st.holdover_need_s:g} s.', f'{var.name} · clock tree', []))
    for n in topo.nodes:
        if n.kind == 'deser' and n.p['fsync'] == 'free':
            uids = [u for c in topo.children(n.id) for u in c.uids]
            out.append(Issue('warning', f'{n.label}\'s FSYNC timer runs free ({n.p["ppm"]:g} ppm): its cameras drift '
                                        f'from the lidar.', f'{var.name} · clock tree', uids))
    with _LOCK:
        last = _LAST.get(_key(project, var, st))
    for f in last or ():                 # only what most frames suffer: a single faulty frame is the scenario's
        if f['severity'] in ('critical', 'serious'):
            out.append(Issue(f['severity'], f['message'], f'{var.name} · clock tree', list(f['uids'])))
    return out


# ---------------------------------------------------------------- overlays
CAM_RGBA = [tuple(int(h[i:i + 2], 16) / 255.0 for i in (1, 3, 5)) for h in TM.CAM_COLORS]


def _dt_for(result, cam_uid, which='max'):
    if result is None:
        return None
    c = result.camera(cam_uid)
    return None if c is None else c[f'dt_{which}_ms'] / 1000.0


def overlay_segments(project, result=None, cam_uid=None, dt_s=None, scale=1.0, az_step=2.0, rings=8,
                     settings: ClockTreeSettings | None = None) -> list:
    """Where the reference lidar's points land with and without the slip, in the vehicle frame: [(p0_xyz, p1_xyz,
    rgba)], p0 where a point is at the camera's instant, p1 where the slipped stamps put it (the displacement times
    scale, to make millimetres visible). Points on the lidar's beams (every rings-th ring, every az_step degrees) at
    the settings' range or where they meet the ground, kept when the camera (cam_uid; every camera without one) sees
    them; coloured by the camera. dt_s defaults to the camera's worst Δt in the result (else the settings' slip)."""
    st = settings or (result.settings if result is not None else settings_of(project))
    var = project.variant_by_uid(result.variant_uid) if result is not None and result.variant_uid else project.variant
    var = var or project.variant
    lid = next((s for s in var.sensors if s.visible and s.kind == 'lidar'
                and (result is None or f'sensor:{s.uid}' == result.topology.lidar)), None)
    if lid is None:
        lid = next((s for s in var.sensors if s.visible and s.kind == 'lidar'), None)
    if lid is None:
        return []
    m = project.sensor_model_for(lid)
    el = m.elevations()[::max(int(rings), 1)]
    az = SN.scan_azimuths(m, max(az_step, m.h_res))
    Dirs = (SN.directions(el, az) @ SN.rotation(lid).T).reshape(-1, 3)
    o = lid.pos()
    with np.errstate(divide='ignore', invalid='ignore'):
        rg = np.where(Dirs[:, 2] < -1e-9, -o[2] / Dirs[:, 2], np.inf)
    r = np.minimum(rg, st.range_m)
    P = o + r[:, None] * Dirs
    cams = [c for c in var.cameras if c.visible and (cam_uid is None or c.uid == cam_uid)]
    out = []
    taken = np.zeros(len(P), bool)
    for c in cams:
        dt = dt_s if dt_s is not None else _dt_for(result, c.uid)
        if dt is None:
            dt = st.slip_ms / 1000.0
        intr = resolve(project.module_for(c), project.lens_for(c))
        _, W2O = camera_frames(c)
        _, _, _, ok = intr.project_dirs((P - c.pos()) @ W2O.T)
        sel = ok & ~taken
        if not sel.any():
            continue
        taken |= sel
        th, T = vehicle_motion(dt, st.speed_mps, math.radians(st.yaw_rate_dps))
        Q = _moved(P[sel], float(th), T)
        P1 = Q + (P[sel] - Q) * scale
        rgba = CAM_RGBA[c.slot % 8] + (0.9,)
        out += [(tuple(map(float, a)), tuple(map(float, b)), rgba) for a, b in zip(Q, P1)]
    return out


def image_segments(project, result, cam_uid, dt_s=None, grid=(13, 10), scale=1.0) -> list:
    """The error field over a camera's image: [((u0, v0), (u1, v1), err_px)], from where a point is (the
    displacement times scale) to where the slipped stamps put it, on a coarse grid of pixel samples."""
    st = result.settings if result is not None else None
    var = project.variant_by_uid(result.variant_uid) if result is not None and result.variant_uid else project.variant
    cam = next((c for c in (var or project.variant).cameras if c.uid == cam_uid), None)
    if cam is None:
        return []
    st = st or settings_of(project)
    dt = dt_s if dt_s is not None else _dt_for(result, cam_uid)
    if dt is None:
        dt = st.slip_ms / 1000.0
    node = result.topology.node(f'cam:{cam_uid}') if result is not None else None
    ro = node.p['readout_ms'] if node is not None else 0.0
    pe = projection_error(project, cam, dt, st.speed_mps, st.yaw_rate_dps, st.surface, st.range_m, grid,
                          readout_ms=ro, rows_compensated=st.rows_compensated)
    out = []
    for i in range(1, len(pe['u'])):
        if not pe['ok'][i] or not math.isfinite(pe['err'][i]):
            continue
        u0, v0 = float(pe['u'][i]), float(pe['v'][i])        # where the stamps put the point (on this pixel's ray)
        out.append(((u0 + float(pe['du'][i]) * scale, v0 + float(pe['dv'][i]) * scale), (u0, v0),
                    float(pe['err'][i])))
    return out


# ---------------------------------------------------------------- output
def _r(v, d=3):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return round(v, d) + 0.0 if math.isfinite(v) else None


def _us(st_):
    return None if st_ is None else {k: _r(v * 1e6 if v is not None else None, 3) for k, v in st_.items()}


ASSUMPTIONS = ('A clock\'s error is its reading minus true time; a PI servo (damping 0.7) steers it to its reference '
               'every sync interval (every second against the PPS), the measured offset carrying half the summed '
               'link asymmetry and the links\' jitter in quadrature. Without a reference a clock keeps its learned '
               'frequency and drifts by its holdover stability (a normal draw per run) plus aging. Cameras on a '
               'trigger line fire on their deserializer\'s FSYNC edges, each on a timer tick; trigger-time stamps '
               'carry the deserializer clock\'s error minus the tick rounding. The projection error is the pixel '
               'distance between a static point projected where the stamps put it and where it is after the '
               'vehicle\'s planar arc (speed, yaw rate) over the time error, through the real lens model.')


def summary(res: ClockTreeResult) -> dict:
    """Compact figures (the assistant, the batch report header)."""
    m = res.metrics
    return {'grandmaster': m['gm'], 'runs': res.settings.runs, 'duration_s': res.settings.duration_s,
            'events': [describe_event(e, res.topology) for e in res.events],
            'worst_projection_px_p95': _r(m['worst_p95_px'], 2), 'worst_projection_px_max': _r(m['worst_max_px'], 2),
            'worst_camera': m['worst_max_tag'], 'worst_dt_ms_p95': _r(m['worst_dt_p95_ms'], 4),
            'holdover_s': _r(m['holdover_s'], 1), 'gm_error_us_max': _r(m['gm_true_max_us'], 2),
            'findings': [f['message'] for f in res.findings if f['severity'] != 'good']}


def to_doc(res: ClockTreeResult, project=None) -> dict:
    """The JSON export: the settings (every default filled in), the tree with each node's parameters and where its
    typical values come from, the events, every sensor's timestamp error, the clocks, the projection table and the
    findings."""
    st, topo = res.settings, res.topology
    return {
        'schema': SCHEMA, 'project': getattr(project, 'name', ''), 'variant': res.variant,
        'frame': 'vehicle (ISO 8855, rear axle)',
        'units': {'time': 's from the scenario start', 'error': 'µs (stamp minus true time; statistics of its '
                                                                 'magnitude)',
                  'dt': 'ms (camera stamp error minus the reference lidar\'s)', 'image': 'px', 'speed': 'km/h',
                  'yaw_rate': 'deg/s (positive turning left)', 'range': 'm from the camera along each pixel\'s ray'},
        'assumptions': ASSUMPTIONS, 'limits': __doc__.split('Limits. ')[1].strip(),
        'settings': st.to_dict(),
        'topology': {'grandmaster': topo.gm, 'reference_lidar': topo.lidar, 'notes': list(topo.notes),
                     'nodes': [{'id': n.id, 'kind': n.kind, 'label': n.label, 'parent': n.parent or None,
                                'params': {k: n.p[k] for k in n.params()},
                                'units': {k: PARAMS[k][1] for k in n.params() if PARAMS[k][1]},
                                'edited': list(n.edited),
                                'typical': {k: {'value': n.typical[k][0], 'source': n.typical[k][1]}
                                            for k in n.params() if k in n.typical},
                                'info': n.info, 'note': n.note or None} for n in topo.nodes]},
        'events': [dict(e, description=describe_event(e, topo)) for e in res.events],
        'sensors': [{'id': s['id'], 'tag': s['tag'], 'kind': s['kind'], 'stamping': s['stamping'],
                     'frames': s['frames'], 'stride': s['stride'], 'against_true_us': _us(s['true']),
                     'against_lidar_us': _us(s['vs_lidar']),
                     **({'frames_off_per_run': _r(s.get('frames_off'), 3)} if s['kind'] == 'camera' else {})}
                    for s in res.sensors],
        'clocks': [{'id': c['id'], 'label': c['label'], 'against_true_us': _us(c['true']),
                    'holdover_s': [[_r(a, 3), _r(b, 3)] for a, b in c['holdover']],
                    'steps_per_run': _r(c['steps_per_run'], 3), 'max_step_us': _r(c['max_step_us'], 3),
                    'steps_back': c['backward'],
                    'servo': {k: (_r(v, 5) if isinstance(v, float) else v) for k, v in (c['servo'] or {}).items()}
                    or None} for c in res.clocks],
        'projection': [{k: (_r(v, 4) if isinstance(v, float) else v) for k, v in c.items() if k not in ('id',)}
                       for c in res.projection],
        'metrics': {k: (_r(v, 4) if isinstance(v, float) else v) for k, v in res.metrics.items()},
        'findings': [{'severity': f['severity'], 'kind': f['kind'], 'message': f['message']} for f in res.findings],
    }


SENSOR_COLUMNS = ('sensor', 'kind', 'stamping', 'true_p50_us', 'true_p95_us', 'true_max_us', 'lidar_p50_us',
                  'lidar_p95_us', 'lidar_max_us')
PROJ_COLUMNS = ('camera', 'against', 'dt_p50_ms', 'dt_p95_ms', 'dt_max_ms', 'centre_p95_px', 'worst_p95_px',
                'centre_max_px', 'worst_max_px', 'plan_misalign_ms', 'plan_worst_px')


def sensors_csv(res: ClockTreeResult) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator='\n')
    w.writerow(SENSOR_COLUMNS)
    for s in res.sensors:
        t, l_ = _us(s['true']), _us(s['vs_lidar']) or {}
        w.writerow([s['tag'], s['kind'], s['stamping'], t['p50'], t['p95'], t['max'], l_.get('p50'), l_.get('p95'),
                    l_.get('max')])
    return buf.getvalue()


def projection_csv(res: ClockTreeResult) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator='\n')
    w.writerow(PROJ_COLUMNS)
    for c in res.projection:
        w.writerow([c['tag'], c['against']] + [_r(c[k], 4) for k in PROJ_COLUMNS[2:]])
    return buf.getvalue()


def png(res: ClockTreeResult, size=(1200, 640)) -> bytes:
    """The scenario as PNG: each sensor's p95 timestamp error against the reference lidar (or true time) over the
    timeline on a log scale, the events marked, and the projection table."""
    import cv2
    W, H = size
    img = np.full((H, W, 3), (25, 26, 26), np.uint8)
    ink, ink2, muted, grid = (236, 239, 240), (183, 194, 195), (129, 135, 137), (44, 44, 42)
    font, aa = cv2.FONT_HERSHEY_SIMPLEX, cv2.LINE_AA

    def text(t, x, y, col, s=0.45):
        cv2.putText(img, t, (int(x), int(y)), font, s, col, 1, aa)
    st, m = res.settings, res.metrics
    text(f'Clock tree  {res.variant}  {st.runs} runs x {st.duration_s:g} s, seed {st.seed}', 16, 26, ink, 0.6)
    w = m['worst_max_px']
    text(f'worst projection {w:.2f} px ({m["worst_max_tag"]}) at {st.speed_kmh:g} km/h, {st.yaw_rate_dps:g} deg/s'
         if w is not None else 'no cameras', 16, 50, ink2)
    L, T_, Rr, B = 70, 70, W - 260, H - 60
    lo, hi = -9.0, -1.0                                   # log10 seconds: 1 ns .. 100 ms
    D = st.duration_s
    X = lambda t: L + (Rr - L) * t / D
    Y = lambda s: B - (B - T_) * (min(max(math.log10(max(s, 1e-12)), lo), hi) - lo) / (hi - lo)
    for e in res.events:
        a = X(e['t_s'])
        b = X(min(e['t_s'] + max(e['dur_s'], 0.0), D)) if e['kind'] in ('pps_dropout', 'gm_loss') else a + 2
        if e['kind'] == 'gm_loss' and e['dur_s'] <= 0:
            b = X(D)
        cv2.rectangle(img, (int(a), T_), (int(max(b, a + 2)), B), (52, 48, 40), -1)
        text(EVENT_LABEL[e['kind']], a + 3, T_ + 14, muted, 0.38)
    for k in range(int(lo), int(hi) + 1):
        y = Y(10.0 ** k)
        cv2.line(img, (L, int(y)), (Rr, int(y)), grid, 1)
        lbl = {-9: '1 ns', -8: '10 ns', -7: '100 ns', -6: '1 us', -5: '10 us', -4: '100 us', -3: '1 ms',
               -2: '10 ms', -1: '100 ms'}[k]
        text(lbl, 8, y + 4, muted, 0.38)
    cv2.rectangle(img, (L, T_), (Rr, B), (70, 70, 66), 1)
    tser = res.series['t']
    for s in res.sensors:
        key = 'rel:' + s['id'] if 'rel:' + s['id'] in res.series else s['id']
        ser = res.series[key]['p95']
        col = TM._bgr(TM.CAM_COLORS[s['slot'] % 8]) if s['kind'] == 'camera' else (
            (200, 200, 196) if s['kind'] == 'lidar' else (140, 170, 200) if s['kind'] == 'radar' else (120, 200, 160))
        pts = [(int(X(t)), int(Y(v))) for t, v in zip(tser, ser) if math.isfinite(v)]
        if len(pts) > 1:
            cv2.polylines(img, [np.array(pts, np.int32)], False, col, 1, aa)
    for i in range(0, 7):
        t = D * i / 6
        text(f'{t:.0f} s', X(t) - 10, B + 18, muted, 0.38)
    text('p95 timestamp error against the reference lidar (true time for the lidar)', L, H - 18, muted, 0.4)
    y = T_
    text('camera   dt p95   px p95   px max', Rr + 14, y, ink2, 0.4)
    for c in res.projection[:16]:
        y += 20
        text(f'{c["tag"]:<6}', Rr + 14, y, TM._bgr(TM.CAM_COLORS[c['slot'] % 8]), 0.4)
        text(f'{c["dt_p95_ms"]:.3g} ms  {(c["worst_p95_px"] or 0):.2f}  {(c["worst_max_px"] or 0):.2f}',
             Rr + 64, y, ink, 0.4)
    ok, buf = cv2.imencode('.png', img)
    return buf.tobytes() if ok else b''


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


def files(res: ClockTreeResult, project=None) -> list:
    """[(name, bytes)]: clock_tree.json, clock_tree_sensors.csv, clock_tree_projection.csv and clock_tree.png."""
    return [('clock_tree.json', (json.dumps(to_doc(res, project), indent=1, ensure_ascii=False,
                                            default=_json_default) + '\n').encode('utf-8')),
            ('clock_tree_sensors.csv', sensors_csv(res).encode('utf-8')),
            ('clock_tree_projection.csv', projection_csv(res).encode('utf-8')),
            ('clock_tree.png', png(res))]


def export(res: ClockTreeResult, project, out_dir) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for name, data in files(res, project):
        path = os.path.join(out_dir, name)
        with open(path, 'wb') as fh:
            fh.write(data)
        paths.append(path)
    return paths
