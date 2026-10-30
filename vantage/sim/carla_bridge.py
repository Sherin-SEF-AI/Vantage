"""Run a Vantage rig in the CARLA simulator and record it as Vantage's sample dataset records a drive.

    python -m vantage.sim.carla_bridge sim_rig.json OUT --town Town10HD --weather ClearNoon --duration 20 \\
        --traffic 30 --seed 1

in CARLA's own Python environment (the carla wheel, numpy, OpenCV; the Vantage source on PYTHONPATH), against a
CARLA 0.9.16 server. sim_rig.json and sim_remap.npz come from core.simrig (Vantage ▸ Simulate ▸ CARLA, or its
export); the module doc there says what CARLA does and how the rig maps onto it.

What a run does
---------------
1. Connects (--host, --port), loads the town when another is open, puts the world and the traffic manager (--tm-port)
   in synchronous mode at the rig's step (fixed_delta_seconds = 1 / tick_hz), seeds the traffic manager and the
   spawn choices with --seed, sets the weather (a CARLA preset, or the rig's mapped Vantage condition).
2. Spawns the ego (the rig's CARLA vehicle, role 'hero') and --traffic vehicles at seeded spawn points, all on the
   traffic manager's autopilot, and lets them drive --warmup seconds.
3. Reads where the project's ego origin (the rear-axle centre on the ground) lies in the CARLA vehicle: the rear
   wheels' centre (VehiclePhysicsControl.wheels, world centimetres) and the bounding box's bottom; attaches every
   sensor there by its rig pose (cameras as their pinhole views, each radar zone as a radar).
4. Steps a whole number of lidar rotations of warm-up (the radar's and IMU's first steps are garbage), the sync pulse
   t = 0 falling where the lidar's angle is back at its start; each camera is spawned so its first frame lands one
   period before its planned exposure centre. Then steps --duration seconds, writing as it goes:
   * cameras: the views of a frame stacked and remapped through the rig's tables (cv2.remap, bilinear), Unreal's
     vignette taken out of every sample in linear light (vantage.sim.tonemap: else each pinhole view of a fisheye
     darkens towards its own edges; --keep-vignette keeps it), the cube faces feathered across their seams (the
     rig's blend tables: within 1.5° of a seam a pixel mixes both faces), JPEG;
   * lidar: each step's slice into the sensor's planned frame (mirror y, undo the mount's turn), the even channels
     kept on the model's beams (ring 0 the lowest), the azimuth thinned back to the model's grid when cast finer,
     points nearer than its minimum range and (optionally) inside the own vehicle's box dropped; a sweep per rotation,
     stamped at its end, each point's t its firing time from its azimuth, and the point where the sensor was then:
     CARLA casts a step's slice of the rotation at one instant, the step's end, so each point is moved into the
     sensor's pose at its firing time (its motion over the step interpolated), as a spinning lidar records it —
     else stamps and geometry disagree by up to a step (at 8 m/s 5 cm on average);
   * radar: the steps of a scan merged, detections outside the zone's azimuth × elevation dropped, the radial
     velocity compensated with the radar's own motion (from its positions, as CARLA computes the relative one);
   * IMU: CARLA's accelerometer (Unreal axes mirrored) and the angular rate from the vehicle's poses (central
     difference, into the IMU's frame; CARLA's gyroscope is not in the sensor frame), a typical MEMS noise and bias;
     a GNSS/INS also its attitude with its stated errors; GNSS: CARLA's fixes with the rig's noise;
   * /tf and /odom at 50 Hz, ground truth (every actor's box within 60 m) at each lidar sweep, keyframe annotations
     with their lidar and radar point counts and a visibility from camera rays (World.cast_ray) to the box's faces.
   * ground truth (--ground-truth, the rig's ground_truth; vantage.core.groundtruth): beside every pinhole view
     CARLA's instance camera (its red channel is the semantic tag, measured equal to the semantic camera's; green +
     256 × blue the actor id) and depth camera, same size, field, pose and step; per camera frame the semantic,
     instance and range PNGs (labels at the nearest view pixel, never mixed; the range along each output pixel's own
     ray, inverse depth interpolated within one surface) and the objects (the bridge's actors by their id, the map's
     parked vehicles by their level boxes, every traffic-light head (TrafficLight.get_light_boxes) and sign claiming
     the static pixels inside its box): tight and full boxes, truncation, occlusion, range, box3d; COCO files;
     beside every lidar a semantic lidar (the same rays, checked every step) whose labels follow the recorded points
     through the same drops and order: nuScenes-lidarseg and -panoptic; ground_truth/index.json;
   * faults (--faults, the rig's): rotation and translation move the mounts CARLA spawns, clock_offset shifts the
     stamps (the ego pose at the stamp), drop leaves frames out, freeze repeats the last frame, range_bias lengthens
     the lidar's and radar's rays; the calibration keeps stating the plan; run.json faults_injected as applied;
   * a scenario (--scenario, SCENARIOS: the ego and its actors on a straight lane chosen by the seed, events in
     run.json scenario) or an actors scene (--actors: the ego still with the actors placed around it, no traffic);
     pedestrians on CARLA's walker AI (--walkers).
   * SUMO traffic (--traffic-source sumo, the rig's traffic_source; vantage.sim.sumo_cosim): instead of the traffic
     manager's vehicles, Eclipse SUMO drives the traffic, stepped once before every synchronous tick through CARLA's
     own co-simulation package (Co-Simulation/Sumo/sumo_integration); its vehicles are CARLA actors (role
     vantage_sumo), so they have boxes, instance ids and lidar labels like any other; the ego and a scenario's
     vehicles are mirrored into SUMO after each tick; run.json traffic_source says what SUMO did. Walkers stay on
     CARLA's walker AI.
5. Writes the nuScenes tables, dataset.json, README.md and run.json (this run's report), then tidies up in all cases:
   stops and destroys every sensor and vehicle it spawned, restores the world's settings and the traffic manager's
   asynchronous mode. A cancel (SIGINT / SIGTERM) or an error removes what was written; run.json then says why.

Progress goes to stdout as JSON lines ({"progress": f, "text": ...}) with --progress-json, else as text.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import math
import os
import queue
import signal
import sys
import time

if __name__ == '__main__' and hasattr(signal, 'pthread_sigmask') and hasattr(signal, 'sigtimedwait'):
    # numpy's OpenBLAS starts its threads on import, before main() runs: block SIGINT and SIGTERM first, so that every
    # thread inherits the block and only the watchdog takes them (on such a thread a SIGTERM would end the bridge at
    # once, and a SIGINT become a KeyboardInterrupt the watchdog never sees, CARLA left untidied either way)
    signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})

import numpy as np

from ..core import groundtruth as GT
from . import frames as FR
from . import tonemap as TM
from .record import Channel, Recorder, token

RUN_SCHEMA = 'vantage.sim_run/1'
RIG_SCHEMA = 'vantage.sim_rig/1'
TF_HZ = 50.0
SELF_MARGIN_M = 0.05
VIS_TOL_M = 0.3
MAP_MARGIN_M = 500.0
MAP_MAX_PX = 12000
MAX_CAMERA_JOBS = 6              # camera frames in flight (being encoded and labelled): beyond, the run waits for them
# the cameras' rendered megapixels per frame (images and ground truth) a run is refused beyond (--render-budget-mpx): a
# 0.9.16 server capped at 12 GB crashed (signal 11) at 306 Mpx, six 1920 × 1536 fisheyes with ground truth; 77 ran
RENDER_BUDGET_MPX = 200.0
SIGNAL_GRACE_S = 15.0            # after SIGINT / SIGTERM: this long to tidy up, else the watchdog ends the process
# role names of everything a bridge spawns (the ego: CARLA's 'hero', with sensors of the bridge's): tidy_server removes
# only these after a bridge was killed
ROLE_PREFIX = 'vantage_'
ROLE_SENSOR, ROLE_TRAFFIC, ROLE_WALKER, ROLE_TARGET = 'vantage_sensor', 'vantage_traffic', 'vantage_walker', \
    'vantage_target'
ROLE_SUMO = 'vantage_sumo'          # the vehicles SUMO drives (sumo_cosim.ROLE_SUMO)
TRAFFIC_SOURCES = ('carla', 'sumo')


class Cancelled(Exception):
    pass


# =========================================================================== the contracts: faults, scenarios, actors
# A fault changes what CARLA records (the actual mount, the stamps, the delivery); the calibration outputs (nuScenes
# calibrated_sensor, tf, dataset.json, the bag) keep stating the plan, as on a real faulty rig.
FAULT_KINDS = {
    # kind: (sensor kinds it applies to, about)
    'rotation': (('camera', 'lidar', 'radar', 'imu', 'gnss'),
                 'deg [yaw, pitch, roll]: the mount turned in the sensor\'s own frame (x forward, y left, z up; '
                 'intrinsic Z-Y-X, pitch positive down)'),
    'translation': (('camera', 'lidar', 'radar', 'imu', 'gnss'), 'm [x, y, z]: the mount moved, in the vehicle frame'),
    'clock_offset': (('camera', 'lidar', 'radar', 'imu', 'gnss'),
                     'ms: data captured at t is stamped t + ms (nuScenes and bag; its ego pose is the one at the stamp)'),
    'drop': (('camera', 'lidar', 'radar', 'imu', 'gnss'), 'rate (0–0.9, seeded) or frames [k, …]: frames never written'),
    'freeze': (('camera', 'lidar', 'radar', 'imu', 'gnss'),
               'from_s, to_s: frames stamped in [from, to) repeat the last one before (stuck sensor)'),
    'range_bias': (('lidar', 'radar'), 'm: every return moved along its ray by m'),
}
FAULT_LIMITS = {'deg': 30.0, 'm': 2.0, 'ms': 1000.0, 'rate': 0.9, 's': 3600.0, 'bias': 5.0}
SCENARIOS = {
    # name: (label, about, {param: ('num', default, lo, hi, unit, tip) | ('choice', default, options, tip)})
    'free_drive': ('Free drive', 'The ego and the traffic on the traffic manager\'s autopilot (the town\'s rules).', {}),
    'lead_vehicle_brake': (
        'Lead vehicle brakes', 'A vehicle drives ahead of the ego in its lane and brakes hard; the ego follows on the '
                               'autopilot and brakes for it.',
        {'speed_kmh': ('num', 30.0, 5.0, 90.0, 'km/h', 'Speed of both vehicles before the brake'),
         'gap_m': ('num', 20.0, 6.0, 80.0, 'm', 'Distance from the ego\'s front to the lead\'s rear at the start'),
         'brake_at_s': ('num', 4.0, 0.0, 600.0, 's', 'When the lead brakes (from the start of the recording)'),
         'traffic': ('num', 0.0, 0.0, 100.0, 'vehicles', 'Other traffic on the autopilot')}),
    'cut_in': (
        'Cut-in', 'A faster vehicle in the next lane changes into the ego\'s lane just ahead of it.',
        {'speed_kmh': ('num', 30.0, 5.0, 90.0, 'km/h', 'The ego\'s speed'),
         'cutter_speed_kmh': ('num', 45.0, 5.0, 120.0, 'km/h', 'The cutting-in vehicle\'s speed'),
         'cutter_offset_m': ('num', 10.0, -40.0, 40.0, 'm', 'Where the cutter starts along the road from the ego '
                                                              '(centre to centre; negative: behind)'),
         'cut_at_s': ('num', 2.0, 0.0, 600.0, 's', 'The cutter changes lanes from this time (from the start of the '
                                                         'recording) …'),
         'cut_gap_m': ('num', 4.0, -5.0, 60.0, 'm', '… once its rear is this far ahead of the ego\'s front'),
         'side': ('choice', 'left', ('left', 'right'), 'The lane the cutter comes from (the other one when it has '
                                                         'none)'),
         'traffic': ('num', 0.0, 0.0, 100.0, 'vehicles', 'Other traffic on the autopilot')}),
    'pedestrian_crossing': (
        'Pedestrian crossing', 'A pedestrian walks across the road ahead of the ego; the ego on the autopilot brakes '
                               'for walkers in its path.',
        {'ego_speed_kmh': ('num', 25.0, 5.0, 60.0, 'km/h', 'The ego\'s speed'),
         'distance_m': ('num', 30.0, 5.0, 120.0, 'm', 'Where the pedestrian waits ahead of the ego (along its lane)'),
         'walk_speed_mps': ('num', 1.4, 0.3, 4.0, 'm/s', 'Walking speed'),
         'start_at_s': ('num', 1.0, 0.0, 600.0, 's', 'When the pedestrian starts walking'),
         'side': ('choice', 'right', ('right', 'left'), 'The side of the road the pedestrian starts from'),
         'walker': ('choice', 'adult', ('adult', 'child'), 'Adult or child'),
         'traffic': ('num', 0.0, 0.0, 100.0, 'vehicles', 'Other traffic on the autopilot')}),
    'cyclist_crossing': (
        'Cyclist crossing', 'A cyclist rides across the road ahead of the ego at a constant speed.',
        {'ego_speed_kmh': ('num', 25.0, 5.0, 60.0, 'km/h', 'The ego\'s speed'),
         'distance_m': ('num', 35.0, 5.0, 120.0, 'm', 'Where the cyclist waits ahead of the ego (along its lane)'),
         'ride_speed_mps': ('num', 4.0, 1.0, 10.0, 'm/s', 'Riding speed'),
         'start_at_s': ('num', 1.0, 0.0, 600.0, 's', 'When the cyclist sets off'),
         'side': ('choice', 'right', ('right', 'left'), 'The side of the road the cyclist starts from'),
         'traffic': ('num', 0.0, 0.0, 100.0, 'vehicles', 'Other traffic on the autopilot')}),
    'parked': ('Parked', 'The ego stands still on a straight road with no traffic.', {}),
}
ACTOR_KINDS = ('child', 'pedestrian', 'cyclist', 'car', 'truck', 'traffic_light', 'sign', 'cone', 'barrier')
# blueprint preferences per kind (the first the server has); walkers by their 'age' attribute, chosen by the seed
ACTOR_BLUEPRINTS = {
    'cyclist': ('vehicle.diamondback.century', 'vehicle.bh.crossbike', 'vehicle.gazelle.omafiets'),
    'car': ('vehicle.toyota.prius', 'vehicle.tesla.model3', 'vehicle.audi.a2', 'vehicle.citroen.c3'),
    'truck': ('vehicle.carlamotors.carlacola', 'vehicle.carlamotors.european_hgv'),
    'traffic_light': (),                    # CARLA 0.9.16 has no traffic-light blueprint to spawn
    'sign': ('static.prop.streetsign', 'static.prop.streetsign01', 'static.prop.streetsign04'),
    'cone': ('static.prop.trafficcone01', 'static.prop.trafficcone02', 'static.prop.constructioncone'),
    'barrier': ('static.prop.streetbarrier', 'static.prop.chainbarrier'),
}
MAX_ACTORS = 200


def _num(v, lo, hi):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return min(max(x, lo), hi) if math.isfinite(x) else None


def _vec(v, n, lim):
    if not isinstance(v, (list, tuple)) or len(v) != n:
        return None
    out = [_num(x, -lim, lim) for x in v]
    return None if any(x is None for x in out) else out


def clean_faults(items, sensors: dict | None = None) -> tuple[list, list]:
    """(faults, problems): the fault list kept to what is valid (numbers clamped; sensors: {tag: kind} of the rig, when
    known, to check each tag and that the kind applies to it)."""
    out, bad = [], []
    if not isinstance(items, list):
        return out, ([] if items in (None, '', {}) else ['The faults are not a list.'])
    for i, f in enumerate(items[:64]):
        if not isinstance(f, dict):
            bad.append(f'Fault {i + 1}: not an object.')
            continue
        tag, kind = str(f.get('tag', '')).strip(), str(f.get('kind', '')).strip()
        if kind not in FAULT_KINDS:
            bad.append(f'Fault {i + 1}: unknown kind {kind!r} (one of {", ".join(FAULT_KINDS)}).')
            continue
        if not tag:
            bad.append(f'Fault {i + 1}: no sensor tag.')
            continue
        if sensors is not None:
            if tag not in sensors:
                bad.append(f'Fault {i + 1}: no sensor {tag} in the rig.')
                continue
            if sensors[tag] not in FAULT_KINDS[kind][0]:
                bad.append(f'Fault {i + 1}: {kind} does not apply to {tag} (a {sensors[tag]}).')
                continue
        g = {'tag': tag, 'kind': kind}
        if kind == 'rotation':
            g['deg'] = _vec(f.get('deg'), 3, FAULT_LIMITS['deg'])
        elif kind == 'translation':
            g['m'] = _vec(f.get('m'), 3, FAULT_LIMITS['m'])
        elif kind == 'clock_offset':
            g['ms'] = _num(f.get('ms'), -FAULT_LIMITS['ms'], FAULT_LIMITS['ms'])
        elif kind == 'range_bias':
            g['m'] = _num(f.get('m'), -FAULT_LIMITS['bias'], FAULT_LIMITS['bias'])
        elif kind == 'drop':
            if isinstance(f.get('frames'), list):
                fr = sorted({int(x) for x in f['frames'][:100000] if isinstance(x, (int, float)) and
                             not isinstance(x, bool) and math.isfinite(x) and x >= 0})
                g['frames'] = fr
            else:
                g['rate'] = _num(f.get('rate'), 0.0, FAULT_LIMITS['rate'])
        elif kind == 'freeze':
            a, b = _num(f.get('from_s'), 0.0, FAULT_LIMITS['s']), _num(f.get('to_s'), 0.0, FAULT_LIMITS['s'])
            g['from_s'], g['to_s'] = a, b
            if a is not None and b is not None and b <= a:
                bad.append(f'Fault {i + 1}: a freeze ends after it starts.')
                continue
        if any(v is None for v in g.values()):
            bad.append(f'Fault {i + 1} ({kind} {tag}): a value is missing or not a number.')
            continue
        out.append(g)
    return out, bad


def clean_scenario(name, params=None) -> tuple[dict, list]:
    """({'name', 'params'} with every parameter of the scenario, clamped, defaults filled in; problems)."""
    bad = []
    if name not in SCENARIOS:
        if name not in (None, ''):
            bad.append(f'Unknown scenario {name!r}: free_drive.')
        name = 'free_drive'
    params = params if isinstance(params, dict) else {}
    out = {}
    for k, spec in SCENARIOS[name][2].items():
        v = params.get(k)
        if spec[0] == 'num':
            x = _num(v, spec[2], spec[3]) if v is not None and not isinstance(v, bool) else None
            out[k] = spec[1] if x is None else x
        else:
            out[k] = v if v in spec[2] else spec[1]
    for k in params:
        if k not in out:
            bad.append(f'{name}: no parameter {k!r}.')
    return {'name': name, 'params': out}, bad


def clean_actors(items) -> tuple[list, list]:
    """(actors, problems): ACTORS.json's list kept to valid entries: {'id', 'kind', 'pose': {'x', 'y', 'z', 'yaw'}}
    (metres and degrees in the ego's vehicle frame)."""
    out, bad, seen = [], [], set()
    if not isinstance(items, list):
        return out, ([] if items in (None, '', {}) else ['The actors are not a list.'])
    for i, a in enumerate(items[:MAX_ACTORS]):
        if not isinstance(a, dict):
            bad.append(f'Actor {i + 1}: not an object.')
            continue
        kind = str(a.get('kind', '')).strip()
        if kind not in ACTOR_KINDS and not (kind.startswith('blueprint:') and len(kind) > 10):
            bad.append(f'Actor {i + 1}: unknown kind {kind!r}.')
            continue
        aid = str(a.get('id', '') or f'a{i + 1}').strip()[:64]
        if aid in seen:
            bad.append(f'Actor {i + 1}: the id {aid} is used twice.')
            continue
        pose = a.get('pose') if isinstance(a.get('pose'), dict) else {}
        p = {'x': _num(pose.get('x'), -500.0, 500.0), 'y': _num(pose.get('y'), -500.0, 500.0),
             'z': _num(pose.get('z', 0.0), -20.0, 50.0), 'yaw': _num(pose.get('yaw', 0.0), -360.0, 360.0)}
        if p['x'] is None or p['y'] is None:
            bad.append(f'Actor {i + 1} ({aid}): its pose needs x and y.')
            continue
        p = {k: (0.0 if v is None else v) for k, v in p.items()}
        seen.add(aid)
        out.append({'id': aid, 'kind': kind, 'pose': p})
    if isinstance(items, list) and len(items) > MAX_ACTORS:
        bad.append(f'Only the first {MAX_ACTORS} actors are kept.')
    return out, bad


GT_KEYS = ('semantic', 'instance', 'range', 'lidar_labels')


def clean_ground_truth(d) -> dict:
    """{semantic, instance, range, lidar_labels: bool}; everything on by default."""
    d = d if isinstance(d, dict) else {}
    out = {}
    for k in GT_KEYS:
        v = d.get(k, True)
        out[k] = v.strip().lower() in ('1', 'true', 'yes', 'on') if isinstance(v, str) else bool(v)
    return out


def gt_camera_kinds(gt: dict) -> list:
    """The CARLA cameras rendered beside every RGB view for the ground truth: the instance camera carries the semantic
    tag too (measured: its red channel is the semantic camera's at every pixel), the depth camera gives the range and
    the objects' occlusion."""
    out = []
    if gt.get('instance'):
        out.append('instance_segmentation')
    elif gt.get('semantic'):
        out.append('semantic_segmentation')
    if gt.get('range') or gt.get('instance'):
        out.append('depth')
    return out


# =========================================================================== inputs
def load_rig(path: str) -> tuple[dict, dict]:
    """The rig document and its remap tables (sim_remap.npz next to it)."""
    with open(path, encoding='utf-8') as fh:
        rig = json.load(fh)
    if not isinstance(rig, dict) or rig.get('schema') != RIG_SCHEMA:
        raise ValueError(f'{path} is not a Vantage simulator rig ({RIG_SCHEMA}).')
    remaps = {}
    npz = os.path.join(os.path.dirname(os.path.abspath(path)), 'sim_remap.npz')
    if rig.get('cameras'):
        if not os.path.isfile(npz):
            raise ValueError(f'{npz} is missing: export the rig again (it holds the cameras\' remap tables).')
        with np.load(npz, allow_pickle=False) as z:
            remaps = {k: np.ascontiguousarray(z[k], np.int64 if z[k].dtype.kind in 'iu' else np.float32)
                      for k in z.files}
    return rig, remaps


def plan(rig: dict, duration: float, cameras=True, faults=(), seed=0) -> dict:
    """Every channel's planned stamps over the drive (s from the sync pulse), the keyframes and each channel's
    keyframe assignment, as core.dataset plans them; the delivery faults marked on their channels (frames dropped,
    a clock offset, a freeze, a range bias), the keyframes among the frames that are written."""
    R = float(rig['simulation']['tick_hz'])
    T = float(duration)
    N = int(round(T * R))
    chans = []
    for c in (rig['cameras'] if cameras else []):
        cap = c['capture']
        n, q = int(cap['every_ticks']), int(cap['phase_ticks'])
        ticks = np.arange(q, N + 1, n)
        chans.append(Channel('camera', c, c['channel'], ticks / R, R / n))
    for L in rig['lidars']:
        m = int(L['steps_per_rotation'])
        k = np.arange(N // m)
        chans.append(Channel('lidar', L, L['channel'], (k + 1) * m / R, R / m))
    for kind, docs in (('radar', rig['radars']), ('imu', rig['imus']), ('gnss', rig['gnss'])):
        for x in docs:
            ticks = schedule(float(x['rate_hz']), R, N)
            ch = Channel(kind, x, x['channel'] if kind == 'radar' else '', ticks / R, min(float(x['rate_hz']), R))
            ch.ticks = ticks
            chans.append(ch)
    chans = [c for c in chans if len(c.stamps)]
    by_tag = {}
    for c in chans:
        by_tag.setdefault(c.doc['tag'], []).append(c)
    for f in faults or ():
        for c in by_tag.get(f['tag'], []):
            n = len(c.stamps)
            if f['kind'] == 'drop':
                if 'frames' in f:
                    c.dropped |= {int(k) for k in f['frames'] if 0 <= int(k) < n}
                else:
                    r = np.random.default_rng([int(seed), 4241] + [ord(ch) for ch in f['tag']]).random(n)
                    c.dropped |= set(np.flatnonzero(r < float(f['rate'])).tolist())
            elif f['kind'] == 'clock_offset':
                c.offset_s += float(f['ms']) / 1000.0
            elif f['kind'] == 'freeze':
                c.freeze = (float(f['from_s']), float(f['to_s']))
            elif f['kind'] == 'range_bias':
                c.range_bias += float(f['m'])
    data = [c for c in chans if c.kind in ('camera', 'lidar', 'radar')]
    if not data:
        raise ValueError('Nothing to record: no camera, lidar or radar frame fits in the drive.')
    ref = next((c for c in data if c.kind == 'lidar'), None) or next((c for c in data if c.kind == 'camera'), None) \
        or data[0]
    kf_hz = float(rig['output']['keyframe_hz'])
    slowest = min(c.rate for c in data)
    kf = min(kf_hz, slowest, ref.rate)
    n = max(1, int(round(ref.rate / kf)))
    samples = ref.stamps[n - 1::n] if len(ref.stamps) >= n else ref.stamps[:1]
    for c in data:
        okk = np.array([k for k in range(len(c.stamps)) if k not in c.dropped], int)
        if not len(okk):
            c.keys = {}
            continue
        j = okk[np.abs(c.stamps[okk][None, :] - samples[:, None]).argmin(1)]
        seen = {}
        for i, jj in enumerate(j):
            if jj in seen:
                if abs(c.stamps[jj] - samples[i]) < abs(c.stamps[jj] - samples[seen[jj]]):
                    seen[jj] = i
            else:
                seen[jj] = i
        c.keys = {int(jj): i for jj, i in seen.items()}
    return {'chans': chans, 'samples': samples, 'keyframe_hz': ref.rate / n, 'ref': ref, 'ticks': N}


def schedule(rate: float, R: float, N: int) -> np.ndarray:
    """Steps (0..N, from the sync pulse) a sensor of this rate samples at: every n-th step when the rate divides the
    step rate, else the step nearest each 1/rate (the rate exact on average), every step when it is faster."""
    if rate <= 0 or rate >= R:
        return np.arange(0, N + 1)
    n = R / rate
    if abs(n - round(n)) < 1e-9:
        return np.arange(0, N + 1, int(round(n)))
    k = np.arange(int(math.floor(N / n)) + 1)
    return np.unique(np.minimum(np.round(k * n).astype(int), N))


# =========================================================================== geometry helpers
def _loc(carla, xyz):
    return carla.Location(x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2]))


def _tf(carla, location, rotation):
    return carla.Transform(_loc(carla, location),
                           carla.Rotation(pitch=float(rotation['pitch']), yaw=float(rotation['yaw']),
                                          roll=float(rotation['roll'])))


def _m(t) -> np.ndarray:
    """4x4 (CARLA axes) of a carla.Transform."""
    return np.array(t.get_matrix(), float)


S4 = np.diag([1.0, -1.0, 1.0, 1.0])


def _mirror(M: np.ndarray) -> np.ndarray:
    """A 4x4 between CARLA frames as one between the mirrored (Vantage-axes) frames."""
    return S4 @ M @ S4


def _kb_project(K, D, model, P):
    """Pixel coordinates (OpenCV) and the angle off the axis of optical-frame points (K, 3), cv2.fisheye's model for
    'equidistant', a plain pinhole otherwise."""
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    rho = np.hypot(x, y)
    th = np.arctan2(rho, z)
    if model == 'equidistant':
        k1, k2, k3, k4 = (list(D) + [0, 0, 0, 0])[:4]
        t2 = th * th
        thd = th * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))
        s = np.where(rho > 1e-12, thd / np.maximum(rho, 1e-12), 0.0)
        u, v = K[0, 0] * x * s + K[0, 2], K[1, 1] * y * s + K[1, 2]
    else:
        zz = np.where(z > 1e-6, z, np.nan)
        u, v = K[0, 0] * x / zz + K[0, 2], K[1, 1] * y / zz + K[1, 2]
    return u, v, th


def _sample(img, x, y, cols=1024):
    """img (h, w, 3) read bilinearly at the points (x, y) (M,): (M, 3). The points go through cv2.remap as a block
    of rows (it takes images under 32767 pixels a side); outside the image reads 0."""
    import cv2
    m = len(x)
    rows = max(-(-m // cols), 1)
    px = np.full(rows * cols, -1.0, np.float32)
    py = np.full(rows * cols, -1.0, np.float32)
    px[:m], py[:m] = x, y
    out = cv2.remap(img, px.reshape(rows, cols), py.reshape(rows, cols), cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out.reshape(-1, img.shape[2])[:m]


def fired_at(P, W_prev, W_cur, a) -> np.ndarray:
    """Points P (n, 3) in a sensor's frame at the end of a step (W_cur: world <- sensor then; W_prev: a step
    earlier) as the sensor saw them at the fraction a (n,) of the step: in its pose then, the motion over the step
    interpolated at constant velocity (translation a·t, rotation a·θ about the step's axis). a = 1 leaves them."""
    T = np.linalg.inv(W_prev) @ W_cur                    # sensor at the end -> sensor at the start
    R, t = T[:3, :3], T[:3, 3]
    q = P @ R.T + t                                      # the points in the start pose's frame
    cos = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    th = math.acos(cos)
    a = np.asarray(a, float)[:, None]
    q = q - a * t
    if th < 1e-12:
        return q
    k = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * math.sin(th))
    phi = -a * th                                        # undo a·θ: into the pose at the firing time
    return q * np.cos(phi) + np.cross(k, q) * np.sin(phi) + k * (q @ k)[:, None] * (1.0 - np.cos(phi))


def feather(out, atlas, blend, devignette=None):
    """The output image (h, w, 3) uint8, read from each pixel's first face, with the cube faces feathered across their
    seams (blend: idx, x, y, w of core.simrig.blend_tables; None leaves it as it is). devignette: a tonemap.Devignette
    per other face's samples, applied before mixing (each sample's vignette is its own view's)."""
    if blend is None or not len(blend[0]):
        return out
    idx, bx, by, bw = blend
    flat = out.reshape(-1, out.shape[2])
    acc = flat[idx].astype(np.float32) * bw[:, :1]
    for j in range(bx.shape[1]):
        smp = _sample(atlas, bx[:, j], by[:, j])
        if devignette is not None:
            smp = devignette[j].apply(smp[None])[0]
        acc += smp.astype(np.float32) * bw[:, 1 + j:2 + j]
    flat[idx] = np.clip(np.rint(acc), 0, 255).astype(out.dtype) if out.dtype == np.uint8 else acc
    return out


# =========================================================================== the bridge
class Bridge:
    def __init__(self, carla, rig: dict, remaps: dict, out_dir: str, args, progress=None, cancel=None):
        self.carla = carla
        self.rig = rig
        self.remaps = remaps
        self.out_dir = os.path.abspath(out_dir)
        self.a = args
        self.progress = progress or (lambda f, text='': None)
        self.cancel = cancel or (lambda: False)
        self.notes = []
        self.timing = {}
        self.actors = []                 # everything spawned, destroyed in reverse at the end
        self.sensors = []                # (actor, kind, doc, view or zone index) of every sensor
        self.devignette = {}             # camera tag -> tonemap.Devignette per layer (the first face's, the blends')
        self.R = float(rig['simulation']['tick_hz'])
        self.dt = 1.0 / self.R
        self.rec = None
        self.report = {'schema': RUN_SCHEMA, 'status': 'starting', 'project': rig.get('project'),
                       'variant': rig.get('variant'), 'out_dir': self.out_dir}
        self.gt = getattr(args, 'gt', None) or clean_ground_truth(rig.get('ground_truth', {}))
        self.gt_kinds = gt_camera_kinds(self.gt) if not getattr(args, 'no_cameras', False) else []
        self.faults = list(getattr(args, 'fault_list', None) or [])
        self.scen = getattr(args, 'scen', None) or clean_scenario('free_drive')[0]
        self.actor_list = list(getattr(args, 'actor_list', None) or [])
        self.faults_injected = []
        self.spawn_index = {}            # CARLA actor id -> 1, 2, … in the order the bridge spawned them (not the ego)
        self.actor_cat = {}              # CARLA actor id -> nuScenes category (the actors the bridge spawned)
        self.scenario = None
        self.scene_actors = []           # the actors scene's report rows
        self.cam_gt = {}                 # camera tag -> core.groundtruth.CameraGT
        self.local_bb = {}               # actor id -> its box as measured unturned (static props)
        self.assoc = {}
        self.sem_lidar = {}
        self.world_actor_ids = set()
        self._gt_seconds = 0.0
        self.part_tags = {}              # 'traffic_light' / 'traffic_sign' -> raw CARLA tag counts of their pixels
        self.traffic_source = getattr(args, 'traffic_source', None) or 'carla'
        self.cosim = None                # vantage.sim.sumo_cosim.CoSim when SUMO drives the traffic
        self.sumo_ids = {}               # CARLA actor id -> SUMO vehicle id (the vehicles SUMO drives)
        import threading
        self._lock = threading.Lock()
        if self.faults:
            self.rig = rig = json.loads(json.dumps(rig))     # the mounts change: never the caller's document
            self._apply_mount_faults()

    # ------------------------------------------------------------------ faults on the mounts
    def _docs(self):
        for kind, key in (('camera', 'cameras'), ('lidar', 'lidars'), ('radar', 'radars'), ('imu', 'imus'),
                          ('gnss', 'gnss')):
            for d in self.rig.get(key, []):
                yield kind, d

    def _apply_mount_faults(self):
        """Rotation and translation faults move the mounts CARLA spawns (each doc's 'carla' pose and every camera
        view's) and give the sensor a 'pose_actual'; 'pose_vehicle', the nuScenes rows and tf_static keep the plan."""
        docs = {d['tag']: (kind, d) for kind, d in self._docs()}
        for f in self.faults:
            if f['kind'] not in ('rotation', 'translation') or f['tag'] not in docs:
                continue
            kind, d = docs[f['tag']]
            pa = dict(d.get('pose_actual') or d['pose_vehicle'])
            R_plan = pose_matrix(d['pose_vehicle'])
            R_now = pose_matrix(pa)
            pos = np.array([pa['x'], pa['y'], pa['z']], float)
            if f['kind'] == 'rotation':
                R_new = R_now @ pose_matrix({'yaw': f['deg'][0], 'pitch': f['deg'][1], 'roll': f['deg'][2]})
            else:
                R_new = R_now
                pos = pos + np.asarray(f['m'], float)
            ypr = euler_zyx(R_new)
            new = {'x': float(pos[0]), 'y': float(pos[1]), 'z': float(pos[2]), 'yaw': ypr[0], 'pitch': ypr[1],
                   'roll': ypr[2]}
            D = R_new @ R_now.T                               # vehicle-frame rotation taking the old mount to the new
            loc = FR.carla_location(pos)
            if kind == 'camera':
                for v in d['views']:
                    r = v['carla']['rotation']
                    Rv = D @ FR.vantage_matrix(r['pitch'], r['yaw'], r['roll'])
                    p_, y_, r_ = FR.carla_rotation(Rv)
                    v['carla'] = {'location': [round(x, 6) for x in loc],
                                  'rotation': {'pitch': p_, 'yaw': y_, 'roll': r_}}
            else:
                r = d['carla']['rotation']
                Rc = D @ FR.vantage_matrix(r['pitch'], r['yaw'], r['roll'])
                p_, y_, r_ = FR.carla_rotation(Rc)
                d['carla'] = {'location': [round(x, 6) for x in loc], 'rotation': {'pitch': p_, 'yaw': y_, 'roll': r_}}
            d['pose_actual'] = new
            plan = d['pose_vehicle']
            row = dict(f, sensor_kind=kind, plan_pose=dict(plan), actual_pose={k: round(v, 6) for k, v in new.items()},
                       pose_delta_deg=[round(_wrap180(new[k] - plan[k]), 6) for k in ('yaw', 'pitch', 'roll')],
                       position_delta_m=[round(new[k] - plan[k], 6) for k in 'xyz'],
                       calibration='the outputs state the plan (pose_vehicle); CARLA records the actual mount')
            if f['kind'] == 'rotation':
                row['frame'] = ('the sensor\'s own frame (x forward, y left, z up): intrinsic Z-Y-X yaw, pitch, roll, '
                                'pitch positive down; R_actual = R_plan · R(deg)')
                row['angle_deg'] = round(math.degrees(np.linalg.norm(FR.rotvec(R_new @ R_plan.T))), 6)
            else:
                row['frame'] = 'the vehicle frame (ISO 8855): position_actual = position_plan + m'
            self.faults_injected.append(row)

    def pose_actual(self, d) -> dict:
        return d.get('pose_actual') or d['pose_vehicle']

    # ------------------------------------------------------------------ helpers
    def _alive(self, phase=None):
        """A sign of life for the watchdog (main): the server answered; phase: setup, run, finish, tidy."""
        life = getattr(self, 'life', None)
        if life is not None:
            life['t'] = time.monotonic()
            if phase:
                life['phase'] = phase

    def _check(self):
        self._alive()
        if self.cancel():
            raise Cancelled()

    def _note(self, text):
        if text not in self.notes:
            self.notes.append(text)

    def _bp(self, lib, bid, attrs: dict):
        bp = lib.find(bid)
        for k, v in attrs.items():
            if bp.has_attribute(k):
                bp.set_attribute(k, str(v))
        return bp

    # ------------------------------------------------------------------ SUMO traffic (vantage.sim.sumo_cosim)
    def _tick(self, world, timeout=None, recording=False):
        """One synchronous step: with SUMO traffic, SUMO steps first and its vehicles move in CARLA, and the bridge's
        own vehicles are mirrored into SUMO after it (run_synchronization.py's order)."""
        cs = getattr(self, 'cosim', None)
        if cs is not None:
            cs.before_tick()
        frame = world.tick(timeout) if timeout is not None else world.tick()
        if cs is not None:
            cs.after_tick(recording)
        return frame

    def _start_sumo(self, client, world, n, mode):
        """SUMO started as the background traffic (n vehicles at once, or the insertion period), stepped with every
        tick from here on; nothing to start when the scene has no traffic."""
        from . import sumo_cosim as SUMO
        a = self.a
        if int(getattr(a, 'walkers', 0) or 0):
            self._note(SUMO.WALKERS_NOTE)
        if n <= 0 and mode in ('actors', 'parked'):
            self._note('SUMO traffic not started: ' + ('an actors scene' if mode == 'actors' else 'the parked scene')
                       + ' has no traffic.')
            return
        if n <= 0:
            self._note('SUMO traffic not started: no traffic asked for.')
            return
        warm_s = float(a.warmup if a.warmup is not None else self.rig['scenario'].get('warmup_s', 2.0))
        horizon = warm_s + float(a.duration) + (int(self.rig['simulation']['warmup_ticks']) + 2) / self.R + 30.0

        def mine():                                    # the bridge's own CARLA vehicles, in spawn order
            return [x for x in self.actors if x.type_id.startswith('vehicle.') and int(x.id) not in self.sumo_ids]

        def spawned(actor, sid):
            self.sumo_ids[int(actor.id)] = sid
            self._own(actor)

        def gone(aid):
            self.actors = [x for x in self.actors if int(x.id) != int(aid)]
        self.progress(0.02, 'Starting SUMO')
        self.cosim = SUMO.CoSim(self.carla, client, world, a.sumo, seed=int(a.seed), dt=self.dt, vehicles=n,
                                horizon_s=horizon, work_dir=os.path.join(self.out_dir, 'sumo'),
                                cache_dir=getattr(a, 'sumo_cache', '') or '',
                                carla_root=getattr(a, 'carla_root', '') or '', on_spawn=spawned, on_destroy=gone,
                                carla_vehicles=mine, check=self._check, note=self._note)
        self.cosim.start()

    # ------------------------------------------------------------------ the run
    def run(self) -> dict:
        t_wall = time.perf_counter()
        carla = self.carla
        a = self.a
        if self.traffic_source == 'sumo':              # refused before CARLA is touched when SUMO is not there
            from . import sumo_cosim as SUMO
            self._sumo_pre = SUMO.preflight(a.sumo, getattr(a, 'carla_root', '') or '')
        client = carla.Client(a.host, int(a.port))
        client.set_timeout(float(a.timeout))
        self.report['client_version'] = client.get_client_version()
        self.report['server_version'] = client.get_server_version()
        if str(self.report['server_version']).split('-')[0] != str(self.report['client_version']).split('-')[0]:
            self._note(f'Server {self.report["server_version"]} and client {self.report["client_version"]} differ.')
        world = client.get_world()
        town = a.town or self.rig['scenario']['town']
        cur = world.get_map().name.rsplit('/', 1)[-1]
        if town and cur not in (town, town + '_Opt'):
            self.progress(0.01, f'Loading {town}')
            world = client.load_world(town)
            cur = world.get_map().name.rsplit('/', 1)[-1]
        self.report['town'] = cur
        self.world = world
        self.original = world.get_settings()
        self.tm = client.get_trafficmanager(int(a.tm_port))
        try:
            return self._run(client, world, t_wall)
        except BaseException as e:
            import traceback
            self.report['traceback'] = traceback.format_exc()
            print(f'{type(e).__name__}: {e}', file=sys.stderr, flush=True)
            raise
        finally:
            self._tidy(world)

    def _run(self, client, world, t_wall):
        carla = self.carla
        a = self.a
        rig = self.rig
        seed = int(a.seed)
        rng = np.random.default_rng([seed, 7919])
        s = world.get_settings()
        s.synchronous_mode = True
        s.fixed_delta_seconds = self.dt
        world.apply_settings(s)
        self.tm.set_synchronous_mode(True)
        self.tm.set_random_device_seed(seed)
        # ---- weather
        wx = rig['weather']
        preset = a.weather or wx.get('preset') or ''
        if preset and preset != 'rig':
            if not hasattr(carla.WeatherParameters, preset):
                raise ValueError(f'Unknown CARLA weather preset {preset!r}.')
            world.set_weather(getattr(carla.WeatherParameters, preset))
            weather = {'preset': preset}
            night = preset.endswith('Night')
        else:
            p = dict(wx['params'])
            base = carla.WeatherParameters.ClearNoon
            kw = {k: getattr(base, k) for k in ('scattering_intensity', 'mie_scattering_scale',
                                                'rayleigh_scattering_scale', 'dust_storm') if hasattr(base, k)}
            kw.update(p)
            world.set_weather(carla.WeatherParameters(**kw))
            weather = {'vantage': wx.get('label'), 'params': p}
            night = bool(wx.get('night'))
        self.report['weather'] = weather
        # ---- the map's level boxes (parked vehicles, traffic lights, signs): read before any vehicle is spawned,
        # which the same call lists
        self._level = self._level_boxes(world)
        try:
            self.world_actor_ids = {int(x.id) for x in world.get_actors()}
        except Exception:                              # noqa: BLE001
            self.world_actor_ids = set()
        # ---- the ego, the scenario's actors, the traffic
        lib = world.get_blueprint_library()
        spawns = list(world.get_map().get_spawn_points())
        if not spawns:
            raise RuntimeError('The map has no spawn points.')
        order = rng.permutation(len(spawns))
        veh = rig['vehicle']['blueprint'] if a.vehicle in (None, '', 'auto') else a.vehicle
        bp = self._bp(lib, veh, {'role_name': 'hero'})
        self.mode = mode = 'actors' if self.actor_list else self.scen['name']
        tm_port = int(a.tm_port)
        start = None
        if mode == 'free_drive' and int(getattr(a, 'spawn_point', -1)) < 0:
            ego, used = None, 0
            for i in order:
                used += 1
                ego = world.try_spawn_actor(bp, spawns[i])
                if ego is not None:
                    break
            if ego is None:
                raise RuntimeError('No free spawn point for the ego vehicle.')
            free = [spawns[i] for i in order[used:]]
            n_traffic = int(a.traffic)
        else:
            ego, start = self._spawn_ego_straight(world, bp, spawns, rng)
            n_traffic = int(a.traffic) if mode == 'free_drive' else \
                0 if mode in ('actors', 'parked') else int(round(self.scen['params'].get('traffic', 0)))
            free = None
        self.actors.append(ego)
        self.ego = ego
        if mode not in ('free_drive', 'actors', 'parked'):
            self.scenario = Scenario(self, world, lib, start, rng)
            self.scenario.spawn()
        if free is None:                               # away from the ego and the scenario's actors
            keep = [self._loc_np(ego.get_transform().location)] + \
                [self._loc_np(x.get_transform().location) for x in self.actors[1:]]
            free = [spawns[i] for i in order if min(np.linalg.norm(self._loc_np(spawns[i].location) - k)
                                                      for k in keep) > 40.0]
        cars = [b for b in lib.filter('vehicle.*') if (not b.has_attribute('number_of_wheels') or
                                                        b.get_attribute('number_of_wheels').as_int() == 4)]
        traffic = []
        sumo_n = n_traffic if self.traffic_source == 'sumo' else 0
        for sp in free:
            if len(traffic) >= n_traffic or sumo_n:
                break
            b = cars[int(rng.integers(len(cars)))]
            if b.has_attribute('role_name'):
                b.set_attribute('role_name', ROLE_TRAFFIC)
            v = world.try_spawn_actor(b, sp)
            if v is not None:
                traffic.append(v)
                self._own(v)
        n_walk = int(getattr(a, 'walkers', 0) or 0) if mode not in ('actors', 'parked') else 0
        walkers = self._spawn_walkers(world, lib, n_walk, rng, seed) if n_walk else []
        if n_walk and len(walkers) < n_walk:
            self._note(f'{len(walkers)} of {n_walk} walkers found a free place on the sidewalks.')
        self.report['walkers'] = len(walkers)
        # ground-truth ids by spawn order, not CARLA's actor ids: the same seed gives the same tokens on every run
        self.gt_id = self.spawn_index
        if self.traffic_source == 'sumo':
            self._start_sumo(client, world, sumo_n, mode)
        elif len(traffic) < n_traffic:
            self._note(f'{len(traffic)} of {n_traffic} traffic vehicles found a free spawn point.')
        self._tick(world)
        for v in traffic:
            v.set_autopilot(True, tm_port)
        if mode == 'free_drive':
            ego.set_autopilot(True, tm_port)
        elif self.scenario is not None:
            self.scenario.start()
        else:
            self._hold(ego)
        if night and hasattr(carla, 'VehicleLightState'):
            VLS = carla.VehicleLightState
            flags = VLS.Position | (VLS.HighBeam if wx.get('beam') == 'high' else VLS.LowBeam)
            try:
                ego.set_light_state(VLS(flags))
            except TypeError:
                ego.set_light_state(flags)
        self.report['traffic'] = len(traffic)
        warm_s = float(a.warmup if a.warmup is not None else rig['scenario'].get('warmup_s', 2.0))
        self.progress(0.03, 'Traffic settling')
        for _ in range(int(round(warm_s * self.R))):
            self._check()
            self._tick(world)
        # ---- where the ego origin lies in the CARLA vehicle
        self._measure_ego()
        self._mount_check(world)
        if mode == 'actors':
            self._place_actors(world, lib, rng)
        # ---- sensors (cameras later, at their spawn steps)
        P = plan(rig, float(a.duration), cameras=not a.no_cameras, faults=self.faults, seed=seed)
        self.P = P
        self._fault_plan_report(P)
        if a.no_cameras and rig['cameras']:
            self._note('Cameras left out (--no-cameras): every annotation\'s visibility is then "1" (0–40 %), as no '
                       'camera saw it.')
        self._spawn_sensors(world, lib)
        cam_docs = [] if a.no_cameras else rig['cameras']
        warm = int(rig['simulation']['warmup_ticks'])
        spawn_at = {}
        for c in cam_docs:
            n, q = int(c['capture']['every_ticks']), int(c['capture']['phase_ticks'])
            spawn_at.setdefault(max(warm + q - n, 1), []).append(c)
        # ---- output
        key_parts = ['vantage', 'carla', rig['project'], rig['variant'], self.report['town'], seed, float(a.duration),
                     int(a.traffic), json.dumps(weather, sort_keys=True)]
        if mode != 'free_drive' or self.faults:        # (a plain drive keeps the tokens it always had)
            key_parts.append(json.dumps({'scenario': self.scen, 'actors': self.actor_list, 'faults': self.faults},
                                        sort_keys=True))
        if getattr(self, 'cosim', None) is not None:   # (and a CARLA-traffic drive too)
            so = dict(self.a.sumo, binary=None)        # (the tokens never depend on where files lie)
            so['net'] = os.path.basename(so['net'])
            key_parts.append(json.dumps({'traffic_source': 'sumo', 'sumo': so}, sort_keys=True))
        run_key = token(*key_parts)
        nus_rows = rig['nuscenes']
        chan_names = {c.channel for c in P['chans'] if c.channel}
        extra = {'sensor_rows': [r for r in nus_rows['sensor'] if r['channel'] in chan_names],
                 'cal_rows': []}
        tok_of = {r['token'] for r in extra['sensor_rows']}
        extra['cal_rows'] = [r for r in nus_rows['calibrated_sensor'] if r['sensor_token'] in tok_of]
        self.rec = Recorder(rig, self.out_dir, P, run_key, extra, bool(rig['output']['nuscenes']),
                            bool(rig['output']['rosbag']), gt=self._gt_spec())
        self._loop(world, warm, spawn_at, cam_docs, rng, t_wall)
        return self.report

    def _gt_spec(self) -> dict:
        """What the Recorder writes for the ground truth, with the notes index.json carries."""
        cams = bool(self.gt_kinds) and bool(self.rig['cameras']) and not self.a.no_cameras
        lid = bool(self.gt.get('lidar_labels')) and bool(self.rig['lidars'])
        notes = []
        if cams:
            notes.append('Camera labels from CARLA\'s ' + ' and '.join(k.replace('_', ' ') + ' camera' for k in
                                                                       self.gt_kinds) +
                         ' beside every pinhole view (same size, field, pose and step), read at the nearest view pixel '
                         'through the camera\'s remap (never interpolated or feathered: where faces are feathered, the '
                         'face with the higher weight).')
            if 'instance_segmentation' in self.gt_kinds:
                notes.append('The instance camera\'s red channel is the semantic tag (the same as the semantic '
                             'camera\'s at every pixel, measured on 0.9.16), green + 256 × blue the CARLA actor id; an '
                             'object is an actor the bridge spawned (by that id) or a map object (a parked vehicle, a '
                             'traffic light, a sign) whose level box holds at least half of its pixels\' 3D points.')
            if not self.gt.get('instance'):
                notes.append('No instance camera: no objects and no ego class (the ego\'s pixels are car).')
        if lid:
            notes.append('Lidar labels from a semantic lidar beside every lidar (same channels, azimuths, rate and '
                         'steps): the same rays in the same order, checked every step (points per channel and every '
                         'direction within 0.01°), else matched by azimuth and counted (run.json '
                         'ground_truth.lidar_labels); kept and ordered exactly as the recorded points.')
        if any(f['kind'] == 'freeze' for f in self.faults):
            notes.append('A frozen camera frame (frame.frozen) repeats an earlier image; its labels are the true scene '
                         'at its time.')
        return {'enabled': cams or lid, 'camera': cams, 'lidar': lid, 'notes': notes}

    # ------------------------------------------------------------------ the scene: spawning, the actors scene
    def _own(self, actor, cat=None):
        """Keep an actor the bridge spawned (destroyed at the end) and give it a ground-truth index (1, 2, … in spawn
        order) and its nuScenes category."""
        self.actors.append(actor)
        self.spawn_index[int(actor.id)] = len(self.spawn_index) + 1
        self.actor_cat[int(actor.id)] = cat or GT.category_of(actor.type_id, getattr(actor, 'attributes', {}) or {})

    @staticmethod
    def _loc_np(loc) -> np.ndarray:
        return np.array([loc.x, loc.y, loc.z], float)

    def _hold(self, vehicle):
        """A vehicle standing still: off the autopilot, hand brake on."""
        try:
            vehicle.apply_control(self.carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))
        except Exception:                              # noqa: BLE001 (a vehicle without controls)
            pass

    def _straight(self, m, loc, ahead: float, behind: float = 0.0):
        """(waypoint, straight metres ahead, straight metres behind) of the lane at loc: how far it runs on within 3°
        of its heading and outside junctions, up to ahead / behind."""
        wp = m.get_waypoint(loc)
        if wp is None:
            return None, 0.0, 0.0
        yaw0 = wp.transform.rotation.yaw

        def run(step_fn, limit):
            d, cur = 0.0, wp
            while d < limit:
                nx = step_fn(cur)
                if len(nx) != 1:
                    break
                cur = nx[0]
                if getattr(cur, 'is_junction', False) or abs(_wrap180(cur.transform.rotation.yaw - yaw0)) > 3.0:
                    break
                d += 2.0
            return d
        return wp, run(lambda w: w.next(2.0), ahead), run(lambda w: w.previous(2.0), behind) if behind > 0 else 0.0

    def _spawn_ego_straight(self, world, bp, spawns, rng):
        """The ego at the spawn point asked for (--spawn-point), else at one chosen by the seed among those on a lane
        straight long enough for the scenario (the straightest when none is): (actor, start info)."""
        need_a, need_b, adj = self._need_straight()
        m = world.get_map()
        n = len(spawns)
        sp_req = int(getattr(self.a, 'spawn_point', -1))
        if sp_req >= 0:
            if sp_req >= n:
                raise RuntimeError(f'Spawn point {sp_req} does not exist ({n} on this map).')
            cands = [sp_req]
        else:
            rows = []
            for i, sp in enumerate(spawns):
                wp, a_m, b_m = self._straight(m, sp.location, need_a, need_b)
                if wp is None or getattr(wp, 'is_junction', False):
                    continue
                if adj and self._adjacent(wp, self.scen['params'].get('side', 'left'), need_a) is None:
                    continue
                rows.append((i, min(a_m / need_a, 1.0) + (min(b_m / need_b, 1.0) if need_b else 1.0)))
            good = [i for i, sc in rows if sc >= 2.0 - 1e-9]
            if good:
                cands = [good[k] for k in rng.permutation(len(good))]
            else:
                cands = [i for i, _ in sorted(rows, key=lambda r: -r[1])] or list(rng.permutation(n))
                self._note(f'No spawn point has {need_a:.0f} m of straight lane ahead for the {self.mode} scene: the '
                           'straightest one is used.')
        for i in cands:
            ego = world.try_spawn_actor(bp, spawns[int(i)])
            if ego is not None:
                wp, a_m, b_m = self._straight(m, spawns[int(i)].location, need_a, need_b)
                start = {'spawn_point': int(i), 'waypoint': wp, 'straight_ahead_m': a_m, 'straight_behind_m': b_m,
                         'road_id': getattr(wp, 'road_id', None), 'lane_id': getattr(wp, 'lane_id', None),
                         'chosen': 'asked for' if sp_req >= 0 else 'by the seed on a straight lane'}
                self.report['start'] = {k: v for k, v in start.items() if k != 'waypoint'}
                return ego, start
        raise RuntimeError('No free spawn point for the ego vehicle.' if sp_req < 0 else
                           f'Spawn point {sp_req} is taken.')

    def _need_straight(self):
        """(metres of straight lane ahead, behind, whether an adjacent lane is needed) for the scene."""
        p = self.scen['params']
        warm = float(self.a.warmup if self.a.warmup is not None else self.rig['scenario'].get('warmup_s', 2.0))
        name = self.scen['name'] if self.mode != 'actors' else 'actors'
        if name == 'lead_vehicle_brake':
            return min(p['gap_m'] + p['speed_kmh'] / 3.6 * (warm + p['brake_at_s']) + 25.0, 300.0), 0.0, False
        if name == 'cut_in':                           # to the end of the lane change (about 3 s)
            return min(max(p['cutter_offset_m'], 0.0) + p['cutter_speed_kmh'] / 3.6 * (warm + p['cut_at_s'] + 3.0)
                       + 10.0, 300.0), max(-p['cutter_offset_m'], 0.0) + 10.0, True
        if name in ('pedestrian_crossing', 'cyclist_crossing'):
            return p['distance_m'] + 15.0, 0.0, False
        if name == 'actors':
            ahead = max([a['pose']['x'] for a in self.actor_list] + [0.0])
            behind = max([-a['pose']['x'] for a in self.actor_list] + [0.0])
            return max(ahead + 10.0, 20.0), behind + 5.0, False
        return 30.0, 10.0, False                       # parked, or a free drive at a given spawn point

    def _adjacent(self, wp, side='left', ahead=0.0):
        """The waypoint of the next driving lane in the same direction on that side (else the other side) whose
        marking lets a vehicle in it change into wp's lane (Waypoint.lane_change, checked every 10 m over `ahead`
        metres), or None: (waypoint, side)."""
        for sd in (side, 'right' if side == 'left' else 'left'):
            try:
                w = wp.get_left_lane() if sd == 'left' else wp.get_right_lane()
            except Exception:                          # noqa: BLE001
                w = None
            if w is None or str(getattr(w, 'lane_type', '')).split('.')[-1] != 'Driving':
                continue
            if (w.lane_id > 0) != (wp.lane_id > 0):
                continue
            back = 'Right' if sd == 'left' else 'Left'          # from that lane into wp's: the other way round
            ok, cur, d = True, w, 0.0
            while True:
                lc = str(getattr(cur, 'lane_change', 'Both')).split('.')[-1]
                if lc not in (back, 'Both'):
                    ok = False
                    break
                if d >= ahead:
                    break
                nx = cur.next(10.0)
                if not nx:
                    break
                cur, d = nx[0], d + 10.0
            if ok:
                return w, sd
        return None

    def _actor_blueprint(self, lib, kind, rng):
        """(blueprint id, problem) for an actors-scene kind."""
        if kind.startswith('blueprint:'):
            bid = kind.split(':', 1)[1]
            try:
                lib.find(bid)
            except Exception:                          # noqa: BLE001
                return None, f'no blueprint {bid} on the server'
            return bid, ''
        if kind in ('child', 'pedestrian'):
            want = 'child' if kind == 'child' else 'adult'
            ids = sorted(b.id for b in lib.filter('walker.pedestrian.*')
                         if b.has_attribute('age') and b.get_attribute('age').as_str() == want)
            if not ids:
                return None, f'no {want} walker blueprint on the server'
            return ids[int(rng.integers(len(ids)))], ''
        have = {b.id for b in lib}
        for bid in ACTOR_BLUEPRINTS.get(kind, ()):
            if bid in have:
                return bid, ''
        if kind == 'traffic_light':
            return None, 'CARLA 0.9.16 has no traffic-light blueprint that can be spawned (the map\'s own lights are ' \
                         'level actors)'
        return None, f'none of {", ".join(ACTOR_BLUEPRINTS.get(kind, ()))} on the server'

    def _place_actors(self, world, lib, rng):
        """The actors scene: every actor at its pose in the ego's vehicle frame (the ego stands still), its box's
        bottom centre on the pose and its heading the pose's yaw; walkers stand (no controller), everything else has
        its physics off so it stays exactly where it was put. Then a short settle and every actor's pose measured."""
        carla = self.carla
        M = _m(self.ego.get_transform())
        rows = []
        placed = []
        for a in self.actor_list:
            self._check()
            row = {'id': a['id'], 'kind': a['kind'], 'requested': dict(a['pose']), 'actor_id': None, 'blueprint': None}
            bid, why = self._actor_blueprint(lib, a['kind'], rng)
            if bid is None:
                row['error'] = why
                rows.append(row)
                self._note(f'Actor {a["id"]} ({a["kind"]}) not spawned: {why}.')
                continue
            bp = lib.find(bid)
            if bp.has_attribute('role_name'):
                bp.set_attribute('role_name', ROLE_TARGET)
            if bp.has_attribute('is_invincible'):
                bp.set_attribute('is_invincible', 'false')
            pose = a['pose']
            p_act = np.asarray(FR.carla_location([pose['x'], pose['y'], pose['z']]), float) + self.O
            p_w = (M @ np.r_[p_act, 1.0])[:3]
            gp = getattr(world, 'ground_projection', None)
            if gp is not None:                         # the ground under its place, before anything is there
                try:
                    hit = gp(_loc(carla, p_w + [0.0, 0.0, 3.0]), 10.0)
                    if hit is not None:
                        row['ground_gap_m'] = round(float(p_w[2] - hit.location.z), 4)
                except Exception:                      # noqa: BLE001
                    pass
            R_w = M[:3, :3] @ FR.carla_matrix(0.0, -float(pose['yaw']), 0.0)
            pit, yw, rl = FR.carla_euler(R_w)
            prop = bid.startswith('static.')
            act = None
            for lift in (0.6, 1.5, 3.0):                # (a prop unturned: 0.9.16 gives its box in world axes)
                act = world.try_spawn_actor(bp, _tf(carla, p_w + [0.0, 0.0, lift],
                                                    {'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0} if prop else
                                                    {'pitch': pit, 'yaw': yw, 'roll': rl}))
                if act is not None:
                    break
            if act is None:
                row['error'] = 'the spawn collided with something at that place'
                rows.append(row)
                self._note(f'Actor {a["id"]} ({a["kind"]}) not spawned: its place is taken.')
                continue
            self._own(act)
            bb = act.bounding_box
            if prop:                                   # its own box, read unturned (see _box_of)
                self.local_bb[int(act.id)] = bb
            walker = act.type_id.startswith('walker.')
            if not walker:
                try:
                    act.set_simulate_physics(False)
                except Exception:                      # noqa: BLE001
                    pass
            o = p_w - R_w @ np.array([bb.location.x, bb.location.y, bb.location.z - bb.extent.z])
            act.set_transform(_tf(carla, o, {'pitch': pit, 'yaw': yw, 'roll': rl}))
            if walker:
                try:
                    act.apply_control(carla.WalkerControl(speed=0.0))
                except Exception:                      # noqa: BLE001
                    pass
            row.update(actor_id=int(act.id), blueprint=act.type_id)
            rows.append(row)
            placed.append((row, act))
        for _ in range(max(int(round(0.5 * self.R)), 3)):       # let the walkers settle on their feet
            self._check()
            self._tick(self.world)
        snap = self.world.get_snapshot()
        Mi = np.linalg.inv(_mirror(_m(self.ego.get_transform())) @ self.T_actor_ego)
        for row, act in placed:
            s_ = snap.find(act.id)
            Ma = _mirror(_m((s_ or act).get_transform()))
            bb = self._box_of(act)
            Rv = (Mi @ Ma)[:3, :3]
            c = (Mi @ Ma @ np.r_[FR.to_vantage([bb.location.x, bb.location.y, bb.location.z]), 1.0])[:3]
            bottom = c - Rv @ np.array([0.0, 0.0, bb.extent.z])
            yaw = math.degrees(math.atan2(Rv[1, 0], Rv[0, 0]))
            got = {'x': round(float(bottom[0]), 4), 'y': round(float(bottom[1]), 4), 'z': round(float(bottom[2]), 4),
                   'yaw': round(yaw, 3)}
            req = row['requested']
            row.update(pose_vehicle=got, bbox={'size_lwh': [round(2 * bb.extent.x, 4), round(2 * bb.extent.y, 4),
                                                            round(2 * bb.extent.z, 4)],
                                               'center_in_actor': [round(bb.location.x, 4), round(-bb.location.y, 4),
                                                                   round(bb.location.z, 4)]},
                       error_m=round(float(math.hypot(got['x'] - req['x'], got['y'] - req['y'])), 4),
                       error_z_m=round(got['z'] - req['z'], 4),
                       error_yaw_deg=round(_wrap180(got['yaw'] - req['yaw']), 3))
        self.scene_actors = rows
        self.report['actors'] = rows
        spawned = sum(1 for r in rows if r.get('actor_id') is not None)
        self._note(f'An actors scene: {spawned} of {len(rows)} actors spawned around the ego, which stands still; no '
                   'traffic.')

    def _spawn_walkers(self, world, lib, n, rng, seed) -> list:
        """n pedestrians at random places of the map's navigation mesh (its sidewalks and crossings), each walking to
        another on CARLA's walker AI (controller.ai.walker) at 1.0–1.6 m/s; deterministic with the seed (the server's
        pedestrian random generator seeded with it, the blueprints and speeds from the run's)."""
        carla = self.carla
        if hasattr(world, 'set_pedestrians_seed'):
            world.set_pedestrians_seed(int(seed))
        bps = sorted(lib.filter('walker.pedestrian.*'), key=lambda b: b.id)
        if not bps:
            self._note('No walker blueprint on the server: no pedestrians.')
            return []
        out, tries = [], 0
        while len(out) < n and tries < 4 * n:
            tries += 1
            loc = world.get_random_location_from_navigation()
            if loc is None:
                continue
            bp = bps[int(rng.integers(len(bps)))]
            if bp.has_attribute('is_invincible'):
                bp.set_attribute('is_invincible', 'false')
            if bp.has_attribute('role_name'):
                bp.set_attribute('role_name', ROLE_WALKER)
            w = world.try_spawn_actor(bp, carla.Transform(carla.Location(x=loc.x, y=loc.y, z=loc.z + 1.0)))
            if w is not None:
                self._own(w)
                out.append(w)
        world.tick()
        ctrl_bp = lib.find('controller.ai.walker')
        self.walker_ctrls = []
        for w in out:
            c = world.spawn_actor(ctrl_bp, carla.Transform(), attach_to=w)
            self.actors.append(c)
            self.walker_ctrls.append(c)
        world.tick()
        for c in self.walker_ctrls:
            c.start()
            c.go_to_location(world.get_random_location_from_navigation())
            c.set_max_speed(float(1.0 + 0.6 * rng.random()))
        return out

    def _world_of(self, p_vehicle) -> np.ndarray:
        """A point of the ego's vehicle frame in CARLA's world (CARLA axes), now."""
        M = _m(self.ego.get_transform())
        return (M @ np.r_[np.asarray(FR.carla_location(p_vehicle), float) + self.O, 1.0])[:3]

    def _fault_plan_report(self, P):
        """faults_injected rows of the delivery faults, from the plan's channels (what will be dropped or frozen)."""
        by_tag = {}
        for c in P['chans']:
            by_tag.setdefault(c.doc['tag'], []).append(c)
        for f in self.faults:
            if f['kind'] in ('rotation', 'translation'):
                continue
            row = dict(f)
            for c in by_tag.get(f['tag'], []):
                row['channel'] = c.channel or c.topic
                row['planned_frames'] = len(c.stamps)
                if f['kind'] == 'drop':
                    row['dropped_frames'] = sorted(int(k) for k in c.dropped)
                    row['dropped_stamps_s'] = [round(float(c.stamps[k]), 6) for k in row['dropped_frames']]
                elif f['kind'] == 'freeze':
                    fr = [k for k in range(len(c.stamps)) if c.frozen(k)]
                    row['frozen_frames'] = fr
                    row['held_frame'] = (fr[0] - 1) if fr and fr[0] > 0 else (fr[0] if fr else None)
                    row['ground_truth'] = 'the true scene at each frame\'s time (the image repeats an earlier one)' \
                        if c.kind == 'camera' else 'the labels of the repeated points'
                elif f['kind'] == 'clock_offset':
                    row['applies_to'] = ('every stamp of the channel (nuScenes and bag): data captured at t is stamped '
                                         't + ms; its nuScenes ego pose is the one at the stamp')
                elif f['kind'] == 'range_bias':
                    row['applies_to'] = 'every return moved along its ray by m (range + m)'
            self.faults_injected.append(row)
        if self.faults:
            self._note(f'This drive is deliberately faulty: {len(self.faults)} fault'
                       f'{"s" * (len(self.faults) != 1)} injected (run.json faults_injected); the calibration states '
                       'the plan.')

    # ------------------------------------------------------------------ the ego
    def _measure_ego(self):
        ego = self.ego
        bb = ego.bounding_box
        M = _m(ego.get_transform())
        Minv = np.linalg.inv(M)
        rear_x = None
        try:
            wheels = ego.get_physics_control().wheels
            pts = np.array([[w.position.x, w.position.y, w.position.z] for w in wheels], float) / 100.0
            loc = (Minv @ np.c_[pts, np.ones(len(pts))].T).T[:, :3]
            if len(loc) >= 4:
                xs = np.sort(loc[:, 0])
                rear_x = float(np.mean(xs[:2]))
        except Exception as e:                       # noqa: BLE001 (a vehicle without wheels' data)
            self._note(f'Wheel positions not read ({type(e).__name__}); the rear axle from the rig\'s table.')
        table = (self.rig['vehicle'].get('carla') or {})
        if rear_x is None:
            rear_x = float(table.get('rear_axle_x', -bb.extent.x * 0.6))
        z0 = float(bb.location.z - bb.extent.z)
        self.O = np.array([rear_x, 0.0, z0])          # the ego origin in the actor frame, CARLA axes
        self.T_actor_ego = np.eye(4)                  # ego (Vantage axes) -> actor (Vantage axes)
        self.T_actor_ego[:3, 3] = FR.to_vantage(self.O)
        self.bb = (np.array([bb.location.x, bb.location.y, bb.location.z]), np.array([bb.extent.x, bb.extent.y,
                                                                                    bb.extent.z]))
        size = [2 * bb.extent.x, 2 * bb.extent.y, 2 * bb.extent.z]
        self.report['vehicle'] = {'blueprint': ego.type_id, 'bounding_box_lwh_m': [round(v, 4) for v in size],
                                  'ego_origin_in_actor_m': [round(v, 4) for v in self.O],
                                  'rig_choice': self.rig['vehicle'].get('blueprint'),
                                  'project': self.rig['vehicle'].get('project')}

    def _mount_check(self, world):
        """Which sensors sit inside the CARLA vehicle: inside its bounding box, and then how deep inside its body (a
        ray cast towards it from outside the box, from above for a roof sensor and else horizontally along its view,
        meets the body first: the first hit inside the box). The CARLA body is not the project's: a camera inside it
        sees the inside of the car."""
        M = _m(self.ego.get_transform())
        c, e = self.bb
        out = []
        docs = [('camera', d) for d in self.rig['cameras']] + [('lidar', d) for d in self.rig['lidars']] + \
            [('radar', d) for d in self.rig['radars']] + [('gnss', d) for d in self.rig['gnss']]
        for kind, d in docs:
            pv = self.pose_actual(d)
            loc = np.asarray(FR.carla_location([pv['x'], pv['y'], pv['z']]), float) + self.O
            inside_box = bool(np.all(np.abs(loc - c) <= e))
            row = {'tag': d['tag'], 'kind': kind, 'inside_bounding_box': inside_box, 'inside_body_m': None}
            if inside_box:
                if kind in ('lidar', 'gnss'):                  # roof sensors: from above
                    h = np.array([0.0, 0.0, 1.0])
                else:                                          # horizontally along the view
                    R_v = FR.rot_z(pv['yaw']) @ _rot_y(pv['pitch']) @ _rot_x(pv['roll'])
                    fwd = FR.to_vantage(R_v[:, 0])
                    h = np.array([fwd[0], fwd[1], 0.0])
                    if np.linalg.norm(h) < 0.2:                # looking straight up or down: outwards from the centre
                        h = np.array([loc[0] - c[0], loc[1] - c[1], 0.0])
                h /= max(np.linalg.norm(h), 1e-9)
                reach = float(2.0 * np.max(e) + 1.0)           # from outside the box
                p_w = (M @ np.r_[loc, 1.0])[:3]
                o_w = (M @ np.r_[loc + reach * h, 1.0])[:3]
                try:
                    hits = world.cast_ray(_loc(self.carla, o_w), _loc(self.carla, p_w))
                except Exception:                             # noqa: BLE001
                    hits = None
                own = []
                for hpt in (hits or []):                       # hits on the ego's own body: inside its box
                    hw = np.array([hpt.location.x, hpt.location.y, hpt.location.z])
                    ha = (np.linalg.inv(M) @ np.r_[hw, 1.0])[:3]
                    if np.all(np.abs(ha - c) <= e + 0.02):
                        own.append(hw)
                if own:
                    depth = reach - float(np.linalg.norm(own[0] - o_w))
                    row['inside_body_m'] = round(max(depth, 0.0), 4)
                    if depth > 0.005 and kind == 'camera':
                        self._note(f'{d["tag"]}: {100 * depth:.1f} cm inside the CARLA vehicle\'s body (its body is '
                                   'not the project\'s): its images show the inside of the car.')
                elif hits is not None:
                    row['inside_body_m'] = 0.0             # the ray reaches it without meeting the body
            out.append(row)
        self.report['mounts'] = out

    # ------------------------------------------------------------------ sensors
    def _attach(self, world, bp, loc_rel, rotation):
        """Spawn a sensor at a rig location (CARLA axes, relative to the ego origin) on the ego."""
        if bp.has_attribute('role_name'):
            bp.set_attribute('role_name', ROLE_SENSOR)  # (what tidy_server knows the bridge's sensors by)
        loc = np.asarray(loc_rel, float) + self.O
        a = world.spawn_actor(bp, _tf(self.carla, loc, rotation), attach_to=self.ego)
        self.actors.append(a)
        return a

    def _spawn_sensors(self, world, lib):
        self.queues = {}
        self.sem_lidar = {}
        for L in self.rig['lidars']:
            bp = self._bp(lib, 'sensor.lidar.ray_cast', L['blueprint'])
            act = self._attach(world, bp, L['carla']['location'], L['carla']['rotation'])
            self._listen(act, ('lidar', L['tag']))
            self.sensors.append((act, 'lidar', L, 0))
            if self.gt.get('lidar_labels'):
                # the semantic lidar: the same channels, fields, rate and steps, spawned on the same step at the same
                # mount, so it casts the very same rays (no noise, no drop-off: it has neither)
                sb = self._bp(lib, 'sensor.lidar.ray_cast_semantic', {k: L['blueprint'][k] for k in (
                    'channels', 'upper_fov', 'lower_fov', 'points_per_second', 'rotation_frequency', 'range',
                    'horizontal_fov', 'sensor_tick')})
                sa = self._attach(world, sb, L['carla']['location'], L['carla']['rotation'])
                self._listen(sa, ('lidar_sem', L['tag']))
                self.sensors.append((sa, 'lidar_sem', L, 0))
                self.sem_lidar[L['tag']] = {'slices': 0, 'points': 0, 'count_mismatch': 0, 'max_dir_err_deg': 0.0,
                                            'range_diff_sq': 0.0, 'range_diff_n': 0, 'fallback_slices': 0}
        for r in self.rig['radars']:
            for zi, z in enumerate(r['zones']):
                bp = self._bp(lib, 'sensor.other.radar', z['blueprint'])
                act = self._attach(world, bp, r['carla']['location'], r['carla']['rotation'])
                self._listen(act, ('radar', r['tag'], zi))
                self.sensors.append((act, 'radar', r, zi))
        for x in self.rig['imus']:
            sig = x['noise']['accel_sigma']
            bp = self._bp(lib, 'sensor.other.imu', {'sensor_tick': 0.0, 'noise_accel_stddev_x': sig,
                                                    'noise_accel_stddev_y': sig, 'noise_accel_stddev_z': sig,
                                                    'noise_seed': int(self.a.seed) * 31 + len(self.sensors)})
            act = self._attach(world, bp, x['carla']['location'], x['carla']['rotation'])
            self._listen(act, ('imu', x['tag']))
            self.sensors.append((act, 'imu', x, 0))
        for g in self.rig['gnss']:
            bp = self._bp(lib, 'sensor.other.gnss', dict(g['blueprint'], noise_seed=int(self.a.seed) * 37 +
                                                         len(self.sensors)))
            act = self._attach(world, bp, g['carla']['location'], g['carla']['rotation'])
            self._listen(act, ('gnss', g['tag']))
            self.sensors.append((act, 'gnss', g, 0))

    def _listen(self, act, key):
        q = queue.Queue()
        self.queues[key] = q
        act.listen(q.put)

    def _spawn_camera(self, world, lib, c):
        n = int(c['capture']['every_ticks'])
        tick = n * self.dt - self.dt * 1e-4           # a hair under the period: Unreal carries the remainder over
        ex = self.rig['output']
        for vi, v in enumerate(c['views']):
            attrs = {'image_size_x': v['width'], 'image_size_y': v['height'], 'fov': v['fov_deg'],
                     'sensor_tick': tick, 'lens_k': 0.0, 'lens_kcube': 0.0, 'lens_circle_multiplier': 0.0,
                     'lens_circle_falloff': 0.0, 'motion_blur_intensity': 0.0, 'motion_blur_max_distortion': 0.0,
                     'lens_flare_intensity': 0.0, 'bloom_intensity': 0.0, 'chromatic_aberration_intensity': 0.0,
                     'blur_amount': 0.0, 'blur_radius': 0.0,
                     'exposure_mode': ex.get('exposure', 'manual'), 'exposure_compensation': ex.get('exposure_ev', 0.0)}
            bp = self._bp(lib, 'sensor.camera.rgb', attrs)
            act = self._attach(world, bp, v['carla']['location'], v['carla']['rotation'])
            self._listen(act, ('camera', c['tag'], vi))
            self.sensors.append((act, 'camera', c, vi))
            for kind in self.gt_kinds:                 # the ground truth: the same view, field, pose and tick
                gb = self._bp(lib, 'sensor.camera.' + kind, {
                    'image_size_x': v['width'], 'image_size_y': v['height'], 'fov': v['fov_deg'], 'sensor_tick': tick,
                    'lens_k': 0.0, 'lens_kcube': 0.0, 'lens_circle_multiplier': 0.0, 'lens_circle_falloff': 0.0})
                ga = self._attach(world, gb, v['carla']['location'], v['carla']['rotation'])
                self._listen(ga, ('camera', c['tag'], vi, kind))
                self.sensors.append((ga, 'camera_gt', c, (vi, kind)))
        if self.gt_kinds and c['tag'] not in self.cam_gt:
            self.cam_gt[c['tag']] = GT.CameraGT(c, self.remaps)
        if not getattr(self.a, 'keep_vignette', False):
            sizes = [(v['width'], v['height']) for v in c['views']]
            offs = c['atlas']['offsets']
            layers = [TM.Devignette(TM.mask_of_remap(self.remaps[c['remap']['x']], self.remaps[c['remap']['y']],
                                                     offs, sizes))]
            bl = self._blend(c)
            if bl is not None:
                m = TM.mask_of_remap(bl[1], bl[2], offs, sizes)
                layers += [TM.Devignette(m[:, j][None, :]) for j in range(m.shape[1])]
            self.devignette[c['tag']] = layers
            self._note('Cameras: Unreal\'s vignette (intensity 0.4, in every pinhole view) taken out of each pixel in '
                       'linear light through CARLA\'s measured camera response (vantage.sim.tonemap).')

    def _get(self, key, frame, timeout):
        """The item of sensor `key` for this frame (older ones are dropped)."""
        q = self.queues[key]
        end = time.monotonic() + timeout
        while True:
            left = end - time.monotonic()
            if left <= 0:
                raise RuntimeError(f'No data from {"/".join(map(str, key))} for frame {frame} within {timeout:g} s.')
            try:
                item = q.get(timeout=min(left, 0.5))
            except queue.Empty:
                self._check()
                continue
            self._alive()
            if item.frame == frame:
                return item
            if item.frame > frame:
                raise RuntimeError(f'{"/".join(map(str, key))} skipped frame {frame} (got {item.frame}).')

    # ------------------------------------------------------------------ the recording loop
    def _loop(self, world, warm, spawn_at, cam_docs, rng, t_wall):
        rig = self.rig
        rec = self.rec
        R, dt = self.R, self.dt
        N = self.P['ticks']
        lib = world.get_blueprint_library()
        chan_of = {(c.kind, c.doc['tag']): c for c in self.P['chans']}
        for c in self.P['chans']:
            if hasattr(c, 'ticks'):
                c.tick_index = {int(tk): k for k, tk in enumerate(c.ticks)}
        timeout = float(self.a.timeout)
        # global frame: CARLA's world mirrored, shifted so the drive starts at (MAP_MARGIN_M, MAP_MARGIN_M)
        M0 = _mirror(_m(self.ego.get_transform())) @ self.T_actor_ego
        self.offset = np.round(np.array([MAP_MARGIN_M, MAP_MARGIN_M, 0.0]) - np.array([M0[0, 3], M0[1, 3], 0.0]))
        self.G = np.eye(4)
        self.G[:3, 3] = self.offset
        # per-sensor state
        lid_state = {L['tag']: self._new_sweep() for L in rig['lidars']}
        rad_state = {r['tag']: [] for r in rig['radars']}
        cam_pending = {c['tag']: {} for c in cam_docs}
        imu_pending = {}
        imu_bias = {x['tag']: np.random.default_rng([int(self.a.seed), 991, i]).normal(
            0, x['noise']['gyro_bias'], 3) for i, x in enumerate(rig['imus'])}
        imu_rng = {x['tag']: np.random.default_rng([int(self.a.seed), 992, i]) for i, x in enumerate(rig['imus'])}
        self.held = {}                    # (kind, tag) -> the last content written (a freeze repeats it)
        ego_hist = {}
        radar_prev = {}
        frame0 = None
        pool = cf.ThreadPoolExecutor(max_workers=max(1, min(3, os.cpu_count() or 1)))
        cam_jobs = []                     # (future, chan, k, t, T_ego) in submission order
        tf_every = max(1, int(round(R / TF_HZ)))
        gt_ticks = set()
        ref = self.P['ref']
        for t in (ref.stamps if ref.kind == 'lidar' else np.arange(0, N + 1, max(1, int(round(R / 10.0)))) / R):
            gt_ticks.add(int(round(t * R)))
        sample_ticks = {int(round(t * R)): i for i, t in enumerate(self.P['samples'])}
        boxes_at = {}                     # rel tick -> boxes (for keyframe point counts)
        key_ticks = set()
        for c in self.P['chans']:
            if c.kind == 'lidar':
                m = int(c.doc['steps_per_rotation'])
                for k in c.keys:
                    key_ticks.update(range(k * m + 1, (k + 1) * m + 1))
            elif c.kind == 'radar':
                for k in c.keys:
                    end = int(c.ticks[k])
                    start = int(c.ticks[k - 1]) if k > 0 else end - max(int(round(R / float(c.doc['rate_hz']))), 1)
                    key_ticks.update(range(start + 1, end + 1))
        ann_rows, instances = [], {}
        static_boxes = self._static_boxes(world)
        counts_l, counts_r = {}, {}
        perf = {'tick_s': 0.0, 'wait_s': 0.0, 'camera_s': 0.0, 'lidar_s': 0.0, 'ground_truth_s': 0.0}
        cam_frames = {c['tag']: {'expected': 0, 'got': 0, 'unexpected': 0, 'dropped': 0} for c in cam_docs}
        phase_meas = {}
        gt_boxes_at = {}                  # absolute tick -> the objects' boxes for the cameras' ground truth
        self.assoc = {c['tag']: {} for c in cam_docs}
        total = warm + N
        t_rec0 = None
        for i in range(1, total + 1):
            self._check()
            for c in spawn_at.get(i, ()):
                self._spawn_camera(world, lib, c)
            t0 = time.perf_counter()
            self._alive('run')
            frame = self._tick(world, timeout, i >= warm)
            self._alive()
            perf['tick_s'] += time.perf_counter() - t0
            if frame0 is None:
                frame0 = frame - 1                     # tick i ran frame frame0 + i
            snap = world.get_snapshot()
            rel = i - warm
            t = rel * dt
            if rel == 0:
                t_rec0 = time.perf_counter()
            es = snap.find(self.ego.id)
            M_actor = _m(es.get_transform())
            T_ego = self.G @ _mirror(M_actor) @ self.T_actor_ego
            v_world = np.array([es.get_velocity().x, es.get_velocity().y, es.get_velocity().z])
            ego_hist[i] = (M_actor, T_ego, v_world)
            ego_hist.pop(i - 3, None)
            rec.pose(t, T_ego)
            if self.scenario is not None:
                self.scenario.step(rel, t, snap)
            t1 = time.perf_counter()
            items = {}
            for act, kind, doc, idx in self.sensors:
                if kind in ('camera', 'camera_gt'):
                    continue
                key = (kind, doc['tag']) if kind not in ('radar',) else (kind, doc['tag'], idx)
                items[key] = self._get(key, frame, timeout)
            perf['wait_s'] += time.perf_counter() - t1
            boxes = None
            if rel in gt_ticks or rel in key_ticks or rel in sample_ticks:
                boxes = self._boxes(snap, T_ego) + [b for b in static_boxes if self._near(b, T_ego)]
            if boxes is not None and rel in key_ticks:
                boxes_at[rel] = boxes
            # ---- cameras: collect what arrived; wait for the frames due now
            any_due = False
            for c in cam_docs:
                st = cam_pending[c['tag']]
                n = int(c['capture']['every_ticks'])
                spawn_i = max(warm + int(c['capture']['phase_ticks']) - n, 1)
                due = i >= spawn_i and (i - spawn_i) % n == 0
                any_due |= due and 0 <= rel <= N
                if due and 0 <= rel <= N:
                    cam_frames[c['tag']]['expected'] += 1
                self._collect_camera(c, st, frame if due else None, timeout)
            if any_due and self.gt_kinds:
                gt_boxes_at[i] = self._gt_boxes(snap, T_ego)
            for c in cam_docs:
                st = cam_pending[c['tag']]
                n = int(c['capture']['every_ticks'])
                spawn_i = max(warm + int(c['capture']['phase_ticks']) - n, 1)
                nkeys = len(self._view_keys(c))
                for f in sorted(st):
                    if len(st[f]) < nkeys:
                        continue
                    views = st.pop(f)
                    ti = f - frame0
                    tt = (ti - warm) * dt
                    if ti < spawn_i or (ti - spawn_i) % n:
                        cam_frames[c['tag']]['unexpected'] += 1
                    if tt < -1e-9 or tt > float(self.a.duration) + 1e-9:
                        continue
                    ch = chan_of[('camera', c['tag'])]
                    k = int(np.argmin(np.abs(ch.stamps - tt)))
                    if abs(ch.stamps[k] - tt) > dt / 2:
                        self._note(f'{c["tag"]}: a frame at {tt:.4f} s, off its planned times (CARLA\'s schedule).')
                        continue
                    if k in ch.dropped:
                        cam_frames[c['tag']]['dropped'] += 1
                        continue
                    cam_frames[c['tag']]['got'] += 1
                    T_e = ego_hist.get(ti, (None, T_ego))[1]
                    cam_jobs.append((pool.submit(self._camera_frame, c, views, gt_boxes_at.get(ti)), ch, k, tt, T_e))
            for ti in [x for x in gt_boxes_at if x < i - 4 * max([int(c['capture']['every_ticks'])
                                                                  for c in cam_docs] + [1])]:
                gt_boxes_at.pop(ti, None)
            while len(cam_jobs) > MAX_CAMERA_JOBS:     # back-pressure: never more frames in flight than this
                fut, ch, k, tt, T_e = cam_jobs.pop(0)
                t2 = time.perf_counter()
                self._write_camera(ch, k, tt, fut.result(), T_e)
                perf['camera_s'] += time.perf_counter() - t2
            while cam_jobs and cam_jobs[0][0].done():
                fut, ch, k, tt, T_e = cam_jobs.pop(0)
                t2 = time.perf_counter()
                self._write_camera(ch, k, tt, fut.result(), T_e)
                perf['camera_s'] += time.perf_counter() - t2
            # ---- radar steps (the steps before t = 0 that belong to the first scan count too), scans
            for r in rig['radars']:
                Ws = self._sensor_world(M_actor, r)
                p_now = Ws[:3, 3]
                v_rad = (p_now - radar_prev.get(r['tag'], p_now)) / dt
                radar_prev[r['tag']] = p_now
                ch = chan_of[('radar', r['tag'])]
                first = int(ch.ticks[0]) - max(int(round(self.R / float(r['rate_hz']))), 1)
                if rel <= first or i < 3:              # CARLA's first steps after spawning are garbage
                    continue
                det = self._radar_step(r, [items[('radar', r['tag'], zi)] for zi in range(len(r['zones']))], Ws, v_rad,
                                       boxes_at.get(rel), ch.range_bias)
                rad_state[r['tag']].append(det)
                k = ch.tick_index.get(rel)
                if k is not None:
                    merged = self._merge_dets(rad_state[r['tag']])
                    rad_state[r['tag']] = []
                    if k in ch.dropped:
                        continue
                    merged = self._hold_or_keep(('radar', r['tag']), ch, k, merged)
                    rec.radar(ch, k, ch.stamps[k] + ch.offset_s, merged, None if ch.offset_s else T_ego)
                    if k in ch.keys:
                        counts_r[(r['tag'], k)] = merged['counts']
            if rel < 0:
                continue
            # ---- lidar slices
            t3 = time.perf_counter()
            for L in rig['lidars']:
                m = int(L['steps_per_rotation'])
                if rel <= 0:
                    continue
                self._lidar_slice(L, items[('lidar', L['tag'])], lid_state[L['tag']], M_actor,
                                  boxes_at.get(rel), rel, phase_meas, (ego_hist.get(i - 1) or (None,))[0],
                                  items.get(('lidar_sem', L['tag'])), static_boxes)
                if rel % m == 0:
                    k = rel // m - 1
                    ch = chan_of[('lidar', L['tag'])]
                    st = lid_state[L['tag']]
                    if k < len(ch.stamps) and k not in ch.dropped:
                        sw = self._sweep(L, st, ch)
                        sw = self._hold_or_keep(('lidar', L['tag']), ch, k, sw)
                        rec.lidar(ch, k, ch.stamps[k] + ch.offset_s, sw['xyz'], sw['int'], sw['ring'], sw['t'],
                                  None if ch.offset_s else T_ego)
                        if sw.get('lidarseg') is not None:
                            rec.gt_lidar(ch, k, sw['lidarseg'], sw['panoptic'])
                        if k in ch.keys:
                            counts_l[(L['tag'], k)] = dict(st['counts'])
                    lid_state[L['tag']] = self._new_sweep()
            perf['lidar_s'] += time.perf_counter() - t3
            # ---- IMU (one step late: the rate is a central difference), GNSS
            for x in rig['imus']:
                ch = chan_of[('imu', x['tag'])]
                pend = imu_pending.pop(x['tag'], None)
                if pend is not None:                   # the previous step's sample, now its central difference exists
                    self._imu(x, pend, ego_hist, ch, imu_bias[x['tag']], imu_rng[x['tag']])
                if rel in ch.tick_index:
                    imu_pending[x['tag']] = (rel, items[('imu', x['tag'])], i)
                    if i == total:
                        self._imu(x, imu_pending.pop(x['tag']), ego_hist, ch, imu_bias[x['tag']], imu_rng[x['tag']])
            for g in rig['gnss']:
                ch = chan_of[('gnss', g['tag'])]
                k = ch.tick_index.get(rel)
                if k is not None and k not in ch.dropped:
                    m_ = items[('gnss', g['tag'])]
                    fix = self._hold_or_keep(('gnss', g['tag']), ch, k, (m_.latitude, m_.longitude, m_.altitude))
                    rec.gnss(ch, t + ch.offset_s, fix[0], fix[1], fix[2], float(g['sigma_h_m']))
            # ---- tf, odometry, ground truth, annotations
            if rel % tf_every == 0:
                Rb = T_ego[:3, :3]
                v_body = Rb.T @ FR.to_vantage(v_world)
                rec.tf(t, T_ego, v_body, self._rate(ego_hist, i))
            if rel in gt_ticks and boxes is not None:
                rec.ground_truth(t, boxes, 1.5 / max(ref.rate, 1.0))
            if rel in sample_ticks and boxes is not None:
                self._annotate(sample_ticks[rel], t, boxes, T_ego, ann_rows, instances, counts_l, counts_r)
            if i % 4 == 0 or i == total:
                self.progress(0.05 + 0.9 * i / total, f'{max(t, 0.0):.2f} s of {float(self.a.duration):g} s')
        # ---- the rest of the cameras
        for fut, ch, k, tt, T_e in cam_jobs:
            self._write_camera(ch, k, tt, fut.result(), T_e)
        pool.shutdown(wait=True)
        perf['ground_truth_s'] = round(self._gt_seconds, 2)
        for c in cam_docs:
            if cam_pending[c['tag']]:
                self._note(f'{c["tag"]}: {len(cam_pending[c["tag"]])} frame(s) missing a view at the end.')
        self._finish(ann_rows, instances, perf, cam_frames, phase_meas, t_wall, t_rec0)

    def _hold_or_keep(self, key, ch, k, content):
        """A freeze fault: within its window the channel repeats the last content written before it (the first frame
        of the window holds itself when none came before); outside, the content is written and remembered."""
        if ch.freeze is not None and ch.frozen(k):
            if key not in self.held:
                self.held[key] = content
            return self.held[key]
        if ch.freeze is not None:
            self.held[key] = content
        return content

    def _write_camera(self, ch, k, tt, res, T_e):
        jpeg = self._hold_or_keep(('camera', ch.doc['tag']), ch, k, res['jpeg'])
        t_w = tt + ch.offset_s
        name = self.rec.camera(ch, k, t_w, jpeg, None if ch.offset_s else T_e)
        if res.get('gt') is not None:
            self.rec.gt_camera(ch, k, t_w, res['gt'], name)

    def _sensor_world(self, M_actor, doc) -> np.ndarray:
        """4x4 (CARLA axes) world <- sensor."""
        loc = np.asarray(doc['carla']['location'], float) + self.O
        rot = doc['carla']['rotation']
        return M_actor @ FR.transform_matrix(loc, (rot['pitch'], rot['yaw'], rot['roll']))

    def _view_keys(self, c) -> list:
        """The images a camera frame is made of: (view, None) for each RGB view, (view, kind) for its ground truth."""
        return [(vi, None) for vi in range(len(c['views']))] + \
            [(vi, k) for vi in range(len(c['views'])) for k in self.gt_kinds]

    def _collect_camera(self, c, st, frame, timeout):
        """Move what the views delivered into st[frame][(view, kind)]; when frame is due, wait for all its images."""
        keys = self._view_keys(c)
        nv = len(keys)
        end = time.monotonic() + timeout
        while True:
            for vk in keys:
                q = self.queues.get(('camera', c['tag'], vk[0]) if vk[1] is None else ('camera', c['tag']) + vk)
                if q is None:
                    continue
                while True:
                    try:
                        img = q.get_nowait()
                    except queue.Empty:
                        break
                    st.setdefault(img.frame, {})[vk] = img
            if frame is None or len(st.get(frame, {})) >= nv:
                return
            if all(any(f > frame and vk in v for f, v in st.items()) for vk in keys):
                return                                 # every view moved past it: CARLA skipped this one
            if time.monotonic() > end:
                raise RuntimeError(f'Camera {c["tag"]}: frame {frame} did not arrive within {timeout:g} s.')
            self._check()
            time.sleep(0.002)

    def _blend(self, c):
        """The camera's seam feathering (idx, x, y, w from core.simrig.blend_tables), or None."""
        b = c['remap'].get('blend')
        if not b or b['idx'] not in self.remaps:
            return None
        return tuple(self.remaps[b[k]] for k in ('idx', 'x', 'y', 'w'))

    def _camera_frame(self, c, views, gtb) -> dict:
        """(a worker thread) A camera frame: the JPEG from its RGB views and, with ground truth on, its label images
        and objects."""
        out = {'jpeg': self._camera_jpeg(c, views)}
        if self.gt_kinds:
            t0 = time.perf_counter()
            out['gt'] = self._camera_gt(c, views, gtb)
            with self._lock:
                self._gt_seconds += time.perf_counter() - t0
        return out

    def _camera_gt(self, c, views, gtb) -> dict:
        """The label images of a camera frame (core.groundtruth.CameraGT: nearest view pixel, range along each output
        pixel's ray) and its objects (tight boxes from the instance mask, full boxes and truncation from the 3D boxes
        through the lens, occlusion from the range image), from the camera's actual mount."""
        cg = self.cam_gt[c['tag']]
        at = c['atlas']
        ah, aw = int(at['height']), int(at['width'])
        packs, atl = {}, {}                            # kind -> the views' pixels as packed uint32, stacked as the atlas
        for kind in self.gt_kinds:
            a32 = np.zeros((ah, aw), np.uint32)
            for vi, v in enumerate(c['views']):
                y0 = int(at['offsets'][vi])
                img = views[(vi, kind)]
                a32[y0:y0 + int(img.height), :int(img.width)] = GT.packed(img.raw_data, img.width, img.height)
            atl[kind] = a32
            packs[kind] = cg.gather(a32)               # decoded only where an output pixel reads
        raw, tags = None, None
        if 'instance_segmentation' in packs:
            tags, raw = GT.unpack_instance(packs['instance_segmentation'])
        elif 'semantic_segmentation' in packs:
            tags = GT.unpack_instance(packs['semantic_segmentation'])[0]
        sem = GT.CLASS_OF_TAG[tags] if tags is not None else None
        h, w = cg.h, cg.w
        rng = None
        if 'depth' in atl:                             # (the instance camera tells one surface from another)
            seg = atl.get('instance_segmentation', atl.get('semantic_segmentation'))
            rng = cg.range_from_packed(atl['depth'], seg)
        pa = self.pose_actual(c)
        pose = {'R': pose_matrix(pa), 'pos': np.array([pa['x'], pa['y'], pa['z']], float)}
        objs, coco = [], []
        inst_png = None
        if raw is not None:
            gtb = gtb or {'known': {}, 'level': [], 'parts': []}
            found, sem = GT.objects_in_frame(sem, raw, rng, cg.rays, pose, c['lens'], gtb['known'], gtb['level'],
                                             int(self.ego.id), self.assoc[c['tag']], parts=gtb.get('parts', ()))
            inst = np.zeros((h, w), np.uint16)
            for f in found:
                inst.ravel()[f['pix']] = f['obj']['reg']
            for f in found:                            # what CARLA had tagged the lights' and signs' pixels
                if f.get('raw_tags') is not None:
                    name = GT.CLASS_NAMES[int(f['obj']['part_class'])]
                    with self._lock:
                        acc = self.part_tags.setdefault(name, np.zeros(GT.EGO_CLASS + 1, np.int64))
                        acc += f['raw_tags']
            for f in sorted(found, key=lambda f: f['obj']['reg']):
                o = f['obj']
                geo = GT.object_geometry(o, pose, c['lens'], w, h, cg.rays, rng, pix=f['pix'])
                Rb = np.asarray(o['R'], float)
                aid = o.get('actor_id')
                if aid is None and 'part_class' not in o:  # a map object that is an actor too (its instance id)
                    rids = [r for r in f['raw_ids'] if r in self.world_actor_ids]
                    aid = rids[0] if rids else None
                objs.append({'id': int(o['reg']), 'actor_id': aid, 'instance_token': o.get('instance_token'),
                             'category': o['category'], 'class': GT.CLASS_NAMES.get(f['class'], 'unlabelled'),
                             'bbox': f['bbox'], 'pixels': f['pixels'], **geo,
                             'box3d': {'center': [round(float(x), 4) for x in o['center']],
                                       'size': [round(float(x), 4) for x in o['size']],
                                       'yaw_deg': round(math.degrees(math.atan2(Rb[1, 0], Rb[0, 0])), 4)}})
                coco.append(GT.rle_of_pixels(f['pix'], h, w))
            inst_png = _png(inst)
        rng_png = _png(GT.range_png(rng)) if (rng is not None and self.gt.get('range')) else None
        sem_png = _png(sem.astype(np.uint8)) if sem is not None and (self.gt.get('semantic') or
                                                                     self.gt.get('instance')) else None
        return {'semantic': sem_png, 'instance': inst_png, 'range': rng_png, 'objects': objs, 'coco': coco,
                'width': w, 'height': h}

    def _camera_jpeg(self, c, views) -> bytes:
        import cv2
        at = c['atlas']
        atlas = np.zeros((int(at['height']), int(at['width']), 3), np.uint8)
        for vi, v in enumerate(c['views']):
            img = views[(vi, None)]
            a = np.frombuffer(img.raw_data, np.uint8).reshape(int(img.height), int(img.width), 4)
            y0 = int(at['offsets'][vi])
            atlas[y0:y0 + a.shape[0], :a.shape[1]] = a[..., :3]
        out = cv2.remap(atlas, self.remaps[c['remap']['x']], self.remaps[c['remap']['y']], cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        dv = self.devignette.get(c['tag'])
        if dv is not None:
            out = dv[0].apply(out)
        out = feather(out, atlas, self._blend(c), dv[1:] if dv is not None else None)
        ok, buf = cv2.imencode('.jpg', out, [int(cv2.IMWRITE_JPEG_QUALITY), int(self.rig['output']['jpeg_quality'])])
        if not ok:
            raise RuntimeError('JPEG encoding failed')
        return buf.tobytes()

    @staticmethod
    def _new_sweep() -> dict:
        return {'pts': [], 'int': [], 'ring': [], 't': [], 'counts': {}, 'tag': [], 'idx': [], 'static': []}

    def _sweep(self, L, st, ch) -> dict:
        """A finished sweep in firing order (column by column), with its range bias and labels."""
        xyz = np.concatenate(st['pts']) if st['pts'] else np.zeros((0, 3), np.float32)
        it = np.concatenate(st['int']) if st['int'] else np.zeros(0, np.float32)
        rg = np.concatenate(st['ring']) if st['ring'] else np.zeros(0, np.uint16)
        tp = np.concatenate(st['t']) if st['t'] else np.zeros(0, np.float32)
        order = np.lexsort((rg, tp))                   # firing order, as a lidar sends them: column by column
        out = {'xyz': xyz[order], 'int': it[order], 'ring': rg[order], 't': tp[order], 'lidarseg': None,
               'panoptic': None}
        if ch.range_bias:
            r = np.linalg.norm(out['xyz'], axis=1)
            out['xyz'] = (out['xyz'] * ((r + ch.range_bias) / np.maximum(r, 1e-6))[:, None]).astype(np.float32)
        if self.gt.get('lidar_labels') and L['tag'] in self.sem_lidar and self.rec.do_nus:
            tag = np.concatenate(st['tag'])[order] if st['tag'] else np.zeros(0, np.uint32)
            idx = np.concatenate(st['idx'])[order] if st['idx'] else np.zeros(0, np.uint32)
            stat = np.concatenate(st['static'])[order] if st['static'] else np.zeros(0, np.int32)
            out['lidarseg'], out['panoptic'] = self._point_labels(tag, idx, stat)
        return out

    def _point_labels(self, tag, idx, stat) -> tuple[np.ndarray, np.ndarray]:
        """(nuScenes-lidarseg index uint8, nuScenes-panoptic uint16) per point: an actor the bridge spawned gives its
        own category (a rider its two-wheeler's), the ego vehicle.ego, the rest its tag's category; things get their
        instance (a spawned actor's index, a map's parked vehicle 500 + its index), stuff instance 0."""
        tag = np.asarray(tag, np.int64)
        idx = np.asarray(idx, np.int64)
        lab = GT.NUSC_OF_TAG[np.clip(tag, 0, 255)].astype(np.int64)
        inst = np.zeros(len(tag), np.int64)
        thing_tag = np.isin(tag, list(GT.THING_TAGS))
        for aid in np.unique(idx[thing_tag & (idx > 0)]).tolist():
            cat = self.actor_cat.get(int(aid))
            if cat is None or cat not in GT.NUSC_INDEX:
                continue
            m = thing_tag & (idx == aid)
            lab[m] = GT.NUSC_INDEX[cat]
            k = self.spawn_index.get(int(aid), 0)
            inst[m] = k if 0 < k < 500 else 0
        m = (stat > 0) & (idx == 0) & thing_tag
        inst[m] = np.where(stat[m] - 1 < 499, 500 + stat[m] - 1, 0)
        ego = idx == int(self.ego.id)
        lab[ego] = GT.NUSC_INDEX['vehicle.ego']
        inst[ego] = 0
        inst[~np.isin(lab, list(GT.THING_INDICES))] = 0
        return lab.astype(np.uint8), (lab * GT.PANOPTIC_DIV + inst).astype(np.uint16)

    def _semantic_match(self, L, raw, counts, sem) -> tuple[np.ndarray, np.ndarray]:
        """(tag, object id) per point of a ray-cast lidar slice from the semantic lidar's slice: the same rays in the
        same order (checked: the same points per channel, every direction within 0.01°); else matched per channel by
        azimuth (each point takes the semantic point nearest in azimuth, within half a step) and counted."""
        S = GT.semantic_lidar(sem.raw_data)
        nch = len(counts)
        sc = np.array([sem.get_point_count(ch) for ch in range(nch)], int)
        stats = self.sem_lidar[L['tag']]
        stats['slices'] += 1
        stats['points'] += int(len(raw))
        P = raw[:, :3].astype(float)
        Q = np.c_[S['x'], S['y'], S['z']].astype(float) if len(S) else np.zeros((0, 3))
        if np.array_equal(sc, counts) and len(Q) == len(P):
            rp, rq = np.linalg.norm(P, axis=1), np.linalg.norm(Q, axis=1)
            cosang = np.sum(P * Q, axis=1) / np.maximum(rp * rq, 1e-9)
            err = float(np.degrees(np.arccos(np.clip(cosang.min(), -1.0, 1.0)))) if len(P) else 0.0
            if err <= 0.01:
                stats['max_dir_err_deg'] = max(stats['max_dir_err_deg'], err)
                stats['range_diff_sq'] += float(np.sum((rp - rq) ** 2))
                stats['range_diff_n'] += int(len(P))
                return S['tag'].astype(np.uint32), S['idx'].astype(np.uint32)
        stats['fallback_slices'] += 1
        stats['count_mismatch'] += int(abs(int(sc.sum()) - int(counts.sum())))
        tag = np.zeros(len(P), np.uint32)
        idx = np.zeros(len(P), np.uint32)
        cols = int(L['columns']) * int(L['oversample'])
        half = float(L['blueprint']['horizontal_fov']) / cols / 2.0
        a0 = np.r_[0, np.cumsum(counts)]
        b0 = np.r_[0, np.cumsum(sc)]
        for ch in range(nch):
            p_ = P[a0[ch]:a0[ch + 1]]
            q_ = Q[b0[ch]:b0[ch + 1]]
            if not len(p_) or not len(q_):
                continue
            ap = np.degrees(np.arctan2(p_[:, 1], p_[:, 0]))
            aq = np.degrees(np.arctan2(q_[:, 1], q_[:, 0]))
            o = np.argsort(aq)
            j = np.clip(np.searchsorted(aq[o], ap), 1, len(o) - 1) if len(o) > 1 else np.zeros(len(ap), int)
            cand = np.stack([o[np.maximum(j - 1, 0)], o[np.minimum(j, len(o) - 1)]], 1)
            d = np.abs(((aq[cand] - ap[:, None]) + 180.0) % 360.0 - 180.0)
            best = cand[np.arange(len(ap)), np.argmin(d, 1)]
            ok = np.min(d, 1) <= half
            sl = slice(a0[ch], a0[ch + 1])
            tag[sl] = np.where(ok, S['tag'][b0[ch]:b0[ch + 1]][best], 0)
            idx[sl] = np.where(ok, S['idx'][b0[ch]:b0[ch + 1]][best], 0)
        return tag, idx

    def _lidar_slice(self, L, meas, st, M_actor, boxes, rel, phase_meas, M_prev=None, sem=None, static_boxes=()):
        bpd = L['blueprint']
        raw = np.frombuffer(meas.raw_data, np.float32).reshape(-1, 4)
        nch = int(bpd['channels'])
        counts = np.array([meas.get_point_count(ch) for ch in range(nch)], int)
        if counts.sum() != len(raw):
            raise RuntimeError(f'{L["tag"]}: the point counts per channel do not add up.')
        chan = np.repeat(np.arange(nch), counts)
        keep = np.asarray(L['keep'], int)
        ring = keep[chan]
        sel = ring >= 0
        P = raw[:, :3].astype(float)
        cols = int(L['columns']) * int(L['oversample'])
        hfov = float(bpd['horizontal_fov'])
        step = hfov / cols
        az = np.degrees(np.arctan2(P[:, 1], P[:, 0]))
        g = np.round((az + hfov / 2.0) / step).astype(np.int64) % cols
        if int(L['oversample']) > 1:
            sel &= (g % int(L['oversample'])) == 0
        r = np.linalg.norm(P, axis=1)
        sel &= r >= float(L['range_min_m'])
        if rel == 1 and sel.any() and L['tag'] not in phase_meas:     # the first slice after the sync pulse
            phase_meas[L['tag']] = float(np.min(g[sel])) * step
        # into the planned sensor frame: mirror y, undo the mount's turn about z
        Rt = FR.rot_z(float(L['mount_turn_deg']))
        Q = FR.to_vantage(P) @ Rt.T
        if self.rig['output'].get('drop_self_hits', True):
            W_actor = FR.transform_matrix(np.asarray(L['carla']['location'], float) + self.O,
                                          (L['carla']['rotation']['pitch'], L['carla']['rotation']['yaw'],
                                           L['carla']['rotation']['roll']))
            Pa = P @ W_actor[:3, :3].T + W_actor[:3, 3]
            c, e = self.bb
            inside = np.all(np.abs(Pa - c) <= e + SELF_MARGIN_M, axis=1)
            sel &= ~inside
        tp = (g.astype(float) / cols) / float(L['rate_hz'])           # firing time from the sweep's start
        Qf = Q
        if M_prev is not None:                       # each point where the sensor was when it fired (see the doc)
            j = (rel - 1) % int(L['steps_per_rotation'])
            a = np.clip((tp - j * self.dt) / self.dt, 0.0, 1.0)
            Qf = FR.to_vantage(fired_at(P, self._sensor_world(M_prev, L), self._sensor_world(M_actor, L), a)) @ Rt.T
        st['pts'].append(Qf[sel].astype(np.float32))
        st['int'].append((255.0 * raw[sel, 3]).astype(np.float32))
        st['ring'].append(ring[sel].astype(np.uint16))
        st['t'].append(tp[sel].astype(np.float32))
        if sem is not None:
            tg, ix = self._semantic_match(L, raw, counts, sem)
            tg, ix = tg[sel], ix[sel]
            st['tag'].append(tg)
            st['idx'].append(ix)
            stat = np.zeros(len(tg), np.int32)
            cand = (ix == 0) & np.isin(tg, list(GT.VEHICLE_TAGS))
            if cand.any() and static_boxes:            # a map's parked vehicle (level mesh, no actor id): its box
                Ws = _mirror(self._sensor_world(M_actor, L))
                Rt4 = np.eye(4)
                Rt4[:3, :3] = Rt
                Pc = (self.G @ Ws @ np.linalg.inv(Rt4) @ np.c_[Q[sel][cand], np.ones(int(cand.sum()))].T).T[:, :3]
                hit = np.zeros(len(Pc), np.int32)
                for b in static_boxes:
                    if b.get('level_k') is None or np.hypot(*(b['centre'][:2] - Pc[0, :2])) > 150.0:
                        continue
                    m = (hit == 0) & _in_box(Pc, b, 0.15)
                    hit[m] = int(b['level_k']) + 1
                stat[cand] = hit
            st['static'].append(stat)
        if boxes is not None and sel.any():
            Ws = _mirror(self._sensor_world(M_actor, L))
            Rt4 = np.eye(4)
            Rt4[:3, :3] = Rt
            Pg = (self.G @ Ws @ np.linalg.inv(Rt4) @ np.c_[Q[sel], np.ones(int(sel.sum()))].T).T[:, :3]
            for b in boxes:
                n = int(_in_box(Pg, b).sum())
                if n:
                    st['counts'][b['id']] = st['counts'].get(b['id'], 0) + n

    def _radar_step(self, r, meas_list, Ws, v_rad_world, boxes, range_bias=0.0) -> dict:
        xs = []
        for zi, meas in enumerate(meas_list):
            for d in meas:
                xs.append((zi, d.depth, d.azimuth, d.altitude, d.velocity))
        if not xs:
            return {'P': np.zeros((0, 3)), 'v': np.zeros(0), 'counts': {}, 'u': np.zeros((0, 3)), 'vc': np.zeros(0)}
        A = np.array(xs, float)
        zi, dep, az, alt, vel = A[:, 0].astype(int), A[:, 1] + float(range_bias), A[:, 2], A[:, 3], A[:, 4]
        u_c = np.stack([np.cos(alt) * np.cos(az), np.cos(alt) * np.sin(az), np.sin(alt)], 1)   # CARLA axes
        u = FR.to_vantage(u_c)
        azv = np.degrees(np.arctan2(u[:, 1], u[:, 0]))
        elv = np.degrees(np.arcsin(np.clip(u[:, 2], -1, 1)))
        zones = r['zones']
        ah = np.array([zones[k]['az_half_deg'] for k in zi])
        eh = np.array([zones[k]['el_half_deg'] for k in zi])
        rng = np.array([zones[k]['range'] for k in zi])
        ok = (np.abs(azv) <= ah + 1e-9) & (np.abs(elv) <= eh + 1e-9) & (dep <= rng) & (dep >= float(r['range_min_m']))
        u, dep, vel, u_c = u[ok], dep[ok], vel[ok], u_c[ok]
        P = u * dep[:, None]
        # CARLA's velocity is relative to the radar; with its own motion added back it is the target's radial speed
        u_world = u_c @ Ws[:3, :3].T
        vc = vel + u_world @ v_rad_world
        counts = {}
        if boxes is not None and len(P):
            Pw = (self.G @ _mirror(Ws) @ np.c_[P, np.ones(len(P))].T).T[:, :3]
            for b in boxes:
                n = int(_in_box(Pw, b, 0.2).sum())
                if n:
                    counts[b['id']] = counts.get(b['id'], 0) + n
        return {'P': P, 'v': vel, 'vc': vc, 'u': u, 'counts': counts}

    @staticmethod
    def _merge_dets(steps) -> dict:
        P = np.concatenate([s['P'] for s in steps]) if steps else np.zeros((0, 3))
        u = np.concatenate([s['u'] for s in steps]) if steps else np.zeros((0, 3))
        v = np.concatenate([s['v'] for s in steps]) if steps else np.zeros(0)
        vc = np.concatenate([s['vc'] for s in steps]) if steps else np.zeros(0)
        counts = {}
        for s in steps:
            for k, n in s['counts'].items():
                counts[k] = counts.get(k, 0) + n
        n = len(P)
        return {'x': P[:, 0], 'y': P[:, 1], 'z': P[:, 2], 'v_r': v, 'rcs': np.zeros(n),
                'vx': v * u[:, 0], 'vy': v * u[:, 1], 'vx_comp': vc * u[:, 0], 'vy_comp': vc * u[:, 1],
                'dyn_prop': np.where(np.abs(vc) < 0.3, 1, 0).astype(np.int8), 'counts': counts}

    def _rate(self, hist, i) -> np.ndarray:
        """The vehicle's angular rate (Vantage body axes, rad/s) at step i: a central difference of its poses, a
        backward one at the newest step."""
        a, b, span = hist.get(i - 1), hist.get(i + 1), 2
        if b is None:
            a, b, span = hist.get(i - 1), hist.get(i), 1
        if a is None or b is None:
            return np.zeros(3)
        Ra, Rb = a[1][:3, :3], b[1][:3, :3]
        Rm = hist[i][1][:3, :3] if i in hist else Rb
        w_world = Ra @ FR.rotvec(Ra.T @ Rb) / (span * self.dt)
        return Rm.T @ w_world

    def _imu(self, x, pend, hist, ch, bias, rng):
        rel, meas, i = pend
        t = rel * self.dt
        k = ch.tick_index.get(rel)
        if k is not None and k in ch.dropped:
            return
        acc = FR.to_vantage([meas.accelerometer.x, meas.accelerometer.y, meas.accelerometer.z])
        R_m = FR.vantage_matrix(x['carla']['rotation']['pitch'], x['carla']['rotation']['yaw'],
                                x['carla']['rotation']['roll'])
        w = R_m.T @ self._rate(hist, i) + bias + rng.normal(0.0, x['noise']['gyro_sigma'], 3)
        R_or = None
        att = head = 0.0
        if int(x.get('antennas', 0)) > 0 and i in hist:
            att = math.radians(max(float(x.get('attitude_sigma_deg', 0.0)), 0.01))
            gn = self.rig['gnss']
            base = float(np.linalg.norm(np.subtract([gn[0]['pose_vehicle'][k] for k in 'xyz'],
                                                    [gn[1]['pose_vehicle'][k] for k in 'xyz']))) if len(gn) >= 2 \
                else 0.0
            hs = float(x.get('heading_sigma_deg', 0.0)) * float(x.get('heading_baseline_m', 2.0)) / base \
                if base > 0 and float(x.get('heading_sigma_deg', 0.0)) > 0 and int(x['antennas']) >= 2 else 0.5
            head = math.radians(hs)
            from_noise = FR.rot_z(math.degrees(rng.normal(0, head)))
            Rg = hist[i][1][:3, :3]
            tilt = _rot_xy(rng.normal(0, att), rng.normal(0, att))
            R_or = from_noise @ Rg @ tilt @ R_m
        if k is not None:
            acc, w, R_or = self._hold_or_keep(('imu', x['tag']), ch, k, (acc, w, R_or))
        self.rec.imu(ch, t + ch.offset_s, acc, w, R_or, att, head, x['noise'])

    # ------------------------------------------------------------------ ground truth
    def _category(self, a) -> str:
        return self.actor_cat.get(int(a.id)) or GT.category_of(a.type_id, getattr(a, 'attributes', {}) or {})

    def _boxes(self, snap, T_ego, rng=None) -> list:
        """Every other actor the bridge spawned (vehicles, walkers, props) within the annotation range (or rng m):
        global centre, rotation, size, velocity."""
        out = []
        rng = float(rng if rng is not None else self.rig['output'].get('gt_range_m', 60.0))
        ex = T_ego[:3, 3]
        for a in self.actors:
            if a is self.ego or not a.type_id.startswith(('vehicle.', 'walker.', 'static.prop.')):
                continue
            s = snap.find(a.id)
            if s is None:
                continue
            M = _mirror(_m(s.get_transform()))
            bb = self._box_of(a)
            c_loc = FR.to_vantage([bb.location.x, bb.location.y, bb.location.z])
            Mg = self.G @ M
            centre = Mg[:3, :3] @ c_loc + Mg[:3, 3]
            if math.hypot(centre[0] - ex[0], centre[1] - ex[1]) > rng:
                continue
            Rg = Mg[:3, :3]
            vel = s.get_velocity()
            sid = getattr(self, 'sumo_ids', {}).get(int(a.id))
            if sid is not None and math.hypot(vel.x, vel.y) < 1e-6:     # kinematic: CARLA reports no velocity
                vv = self.cosim.velocity(a.id, s.get_transform().rotation.yaw) if self.cosim is not None else None
                v = FR.to_vantage(vv) if vv is not None else FR.to_vantage([vel.x, vel.y, vel.z])
            else:
                v = FR.to_vantage([vel.x, vel.y, vel.z])
            out.append({'id': int(self.gt_id.get(a.id, 100_000 + a.id)), 'actor_id': int(a.id),
                        'category': self._category(a), 'name': a.type_id, 'centre': centre,
                        'R': Rg, 'yaw': math.atan2(Rg[1, 0], Rg[0, 0]),
                        'size_lwh': [2 * bb.extent.x, 2 * bb.extent.y, 2 * bb.extent.z], 'velocity': v,
                        'moving': float(np.hypot(v[0], v[1])) > 0.3, 'static': False})
            if sid is not None:
                out[-1]['sumo_id'] = sid
        return out

    def _box_of(self, a):
        """An actor's bounding box in its own frame: CARLA's, except a static prop's, which 0.9.16 gives in the world's
        axes (a sign turned 90° reports its length and width swapped: measured): the one read when it stood
        unturned."""
        return self.local_bb.get(int(a.id)) or a.bounding_box

    # (label, nuScenes category, length range m, annotated in nuScenes): the map's level boxes the bridge reads
    LEVEL_LABELS = (('Car', 'vehicle.car', (2.5, 7.0), True), ('Truck', 'vehicle.truck', (3.5, 20.0), True),
                    ('Bus', 'vehicle.bus.rigid', (6.0, 20.0), True), ('Motorcycle', 'vehicle.motorcycle', (1.2, 3.2), True),
                    ('Bicycle', 'vehicle.bicycle', (1.0, 2.6), True), ('TrafficLight', 'static.manmade', (0.05, 8.0), False),
                    ('TrafficSigns', 'static.manmade', (0.05, 8.0), False))

    def _level_boxes(self, world) -> list:
        """The map's own objects (level geometry, not actors) from World.get_level_bbs, in CARLA's world: parked
        vehicles (annotated as vehicle.parked), traffic lights and signs (camera ground-truth objects only). Read
        before the run spawns any vehicle (the call lists the vehicles' boxes too); boxes of a vehicle's size only."""
        COL = getattr(self.carla, 'CityObjectLabel', None)
        if COL is None or not hasattr(world, 'get_level_bbs'):
            return []
        owner = {}                                     # a light head's box centre -> its traffic light's actor id
        try:
            for a in world.get_actors().filter('traffic.traffic_light'):
                for b in (a.get_light_boxes() if hasattr(a, 'get_light_boxes') else []):
                    owner[tuple(np.round([b.location.x, b.location.y, b.location.z], 2))] = int(a.id)
        except Exception:                              # noqa: BLE001
            pass
        out = []
        for label, cat, (lo, hi), annotated in self.LEVEL_LABELS:
            lab = getattr(COL, label, None)
            if lab is None:
                continue
            try:
                bbs = list(world.get_level_bbs(lab))
            except Exception:                          # noqa: BLE001
                self._note(f'Map objects ({label}) not annotated: World.get_level_bbs failed.')
                continue
            for b in bbs:
                L, W, H = 2 * b.extent.x, 2 * b.extent.y, 2 * b.extent.z
                if annotated and not (lo <= L <= hi and 0.3 <= W <= 4.0 and 0.5 <= H <= 5.0):
                    continue
                if not annotated and not (max(L, W, H) <= hi):
                    continue
                out.append({'label': label, 'category': cat, 'annotated': annotated,
                            'loc': [b.location.x, b.location.y, b.location.z],
                            'rot': (b.rotation.pitch, b.rotation.yaw, b.rotation.roll), 'size': [L, W, H],
                            'actor_id': owner.get(tuple(np.round([b.location.x, b.location.y, b.location.z], 2)))
                            if label == 'TrafficLight' else None})
        heads = sum(1 for e in out if e['label'] == 'TrafficLight')
        if heads:
            self._note(f'{heads} traffic-light heads (each light\'s TrafficLight.get_light_boxes, '
                       f'{sum(1 for e in out if e["actor_id"] is not None)} matched to their light) and '
                       f'{sum(1 for e in out if e["label"] == "TrafficSigns")} signs of the map are camera ground-truth '
                       'objects of their own: each takes the static pixels whose 3D point lies inside its box.')
        return out

    def _static_boxes(self, world) -> list:
        """The map's level boxes in the global frame: the vehicles as ground-truth boxes (vehicle.parked, ids 1 000 000 +
        their index), and every box (lights and signs too) for the cameras' ground truth (self._level_global)."""
        out = []
        self._level_global = []
        for k, e in enumerate(getattr(self, '_level', [])):
            Rr = FR.vantage_matrix(*e['rot'])
            centre = FR.to_vantage(e['loc']) + self.offset
            b = {'id': 1_000_000 + k, 'category': e['category'], 'name': f'parked {e["label"].lower()} (map)',
                 'centre': centre, 'R': Rr, 'yaw': math.atan2(Rr[1, 0], Rr[0, 0]), 'size_lwh': list(e['size']),
                 'velocity': np.zeros(3), 'moving': False, 'static': True, 'level_k': k, 'label': e['label'],
                 'actor_id': e.get('actor_id')}
            self._level_global.append(b)
            if e['annotated']:
                out.append(b)
        if out:
            self._note(f'{len(out)} parked vehicles of the map (its level boxes) annotated as vehicle.parked.')
        return out

    def _gt_boxes(self, snap, T_ego) -> dict:
        """The objects a camera's ground truth can name, in the vehicle frame at this step: {'known': {CARLA actor id:
        object} (the actors the bridge spawned), 'level': [object] (the map's level boxes within 100 m)}."""
        Ti = np.linalg.inv(T_ego)
        known = {}
        boxes = self._boxes(snap, T_ego, 250.0)        # every spawned actor a camera may see, not only near ones
        run_key = self.rec.run_key
        for b in boxes:
            aid = b['actor_id']
            known[aid] = {'key': ('actor', aid), 'actor_id': aid, 'reg': int(self.spawn_index.get(aid, 0)) or 999,
                          'category': b['category'], 'center': Ti[:3, :3] @ b['centre'] + Ti[:3, 3],
                          'R': Ti[:3, :3] @ b['R'], 'size': list(b['size_lwh']),
                          'instance_token': token('vantage', 'instance', run_key, b['id'])}
        level, parts = [], []
        ex = T_ego[:3, 3]
        for b in getattr(self, '_level_global', []):
            if math.hypot(b['centre'][0] - ex[0], b['centre'][1] - ex[1]) > 100.0:
                continue
            o = {'key': ('level', b['level_k']), 'level_label': b['label'], 'actor_id': b.get('actor_id'),
                 'reg': 1000 + int(b['level_k']), 'category': b['category'],
                 'center': Ti[:3, :3] @ b['centre'] + Ti[:3, 3], 'R': Ti[:3, :3] @ b['R'], 'size': list(b['size_lwh']),
                 'instance_token': None}
            if b['label'] in GT.PART_CLASS:            # a light's head, a sign: claims its pixels by its box
                o['part_class'] = GT.PART_CLASS[b['label']]
                parts.append(o)
            else:
                o['instance_token'] = token('vantage', 'instance', run_key, b['id'])
                level.append(o)
        return {'known': known, 'level': level, 'parts': parts}

    def _near(self, b, T_ego) -> bool:
        rng = float(self.rig['output'].get('gt_range_m', 60.0))
        return math.hypot(b['centre'][0] - T_ego[0, 3], b['centre'][1] - T_ego[1, 3]) <= rng

    def _annotate(self, si, t, boxes, T_ego, rows, instances, counts_l, counts_r):
        """sample_annotation rows of sample si (the moving actors' boxes and the parked cars near the ego)."""
        nl, nr = {}, {}
        for c in self.P['chans']:
            for k, i in c.keys.items():
                if i != si:
                    continue
                src = counts_l if c.kind == 'lidar' else counts_r if c.kind == 'radar' else None
                dst = nl if c.kind == 'lidar' else nr
                if src is None:
                    continue
                for oid, n in src.get((c.doc['tag'], k), {}).items():
                    dst[oid] = dst.get(oid, 0) + n
        run_key = self.rec.run_key
        us = self.rec.us(t)
        for b in boxes:
            vis = self._visibility(b, T_ego)
            L, W, H = b['size_lwh']
            tok = token('vantage', 'sample_annotation', run_key, b['id'], us)
            inst = token('vantage', 'instance', run_key, b['id'])
            cat = b['category']
            if cat.startswith(('vehicle.bicycle', 'vehicle.motorcycle')):
                attr = 'cycle.with_rider' if not b['static'] else 'cycle.without_rider'
            elif cat.startswith('human'):
                attr = 'pedestrian.moving' if b['moving'] else 'pedestrian.standing'
            elif not cat.startswith('vehicle.'):
                attr = None                            # movable objects carry no attribute in nuScenes
            elif b['static']:
                attr = 'vehicle.parked'
            else:
                attr = 'vehicle.moving' if b['moving'] else 'vehicle.stopped'
            if cat not in GT.NUSC_INDEX or cat.startswith(('static.', 'flat.')) or cat in ('noise', 'vehicle.ego'):
                continue                               # (a street sign the actors scene put there: no nuScenes box)
            rows.append({'token': tok, 'sample_token': self.rec.sample_tok[si], 'instance_token': inst,
                         'visibility_token': '1' if vis < 0.4 else '2' if vis < 0.6 else '3' if vis < 0.8 else '4',
                         'attribute_tokens': [token('vantage', 'attribute', attr)] if attr else [],
                         'translation': [round(float(q), 6) + 0.0 for q in b['centre']],
                         'size': [round(W, 4), round(L, 4), round(H, 4)],
                         'rotation': [round(math.cos(b['yaw'] / 2), 10), 0.0, 0.0, round(math.sin(b['yaw'] / 2), 10)],
                         'prev': '', 'next': '', 'num_lidar_pts': int(nl.get(b['id'], 0)),
                         'num_radar_pts': int(nr.get(b['id'], 0))})
            instances.setdefault(inst, (b['category'], []))[1].append(len(rows) - 1)

    def _visibility(self, b, T_ego) -> float:
        """Share of the box's face points facing the ego that a camera sees: in its image, and the first thing a ray
        from it meets (World.cast_ray) within VIS_TOL_M of the point."""
        L, W, H = b['size_lwh']
        P, N = _box_points(L, W, H)
        Pw = b['centre'] + P @ b['R'].T
        Nw = N @ b['R'].T
        ego_c = T_ego[:3, :3] @ np.array([1.2, 0.0, 1.0]) + T_ego[:3, 3]
        face = np.sum(Nw * (ego_c - Pw), axis=1) > 0
        if not face.any():
            return 0.0
        Pw, Nw = Pw[face], Nw[face]
        seen = np.zeros(len(Pw), bool)
        cams = [] if self.a.no_cameras else self.rig['cameras']
        for c in cams:
            pv = c['pose_vehicle']
            R_c = FR.rot_z(pv['yaw']) @ _rot_y(pv['pitch']) @ _rot_x(pv['roll'])
            pos_g = T_ego[:3, :3] @ np.array([pv['x'], pv['y'], pv['z']]) + T_ego[:3, 3]
            W2O = FR.BODY_TO_OPTICAL @ (T_ego[:3, :3] @ R_c).T
            Q = (Pw - pos_g) @ W2O.T
            im = c['image']
            u, v, th = _kb_project(np.array(im['K']), im['D'], im['distortion_model'], Q)
            ok = (u >= 0) & (u < im['width']) & (v >= 0) & (v < im['height']) & \
                (th <= math.radians(float(im['max_theta_deg']))) & np.isfinite(u) & ~seen
            for j in np.nonzero(ok)[0]:
                if self._clear(pos_g, Pw[j]):
                    seen[j] = True
        return float(seen.mean())

    def _clear(self, a_g, b_g) -> bool:
        if getattr(self, '_no_cast', False):
            return True
        w = self.world
        a_c = FR.to_vantage(a_g - self.offset)        # global -> CARLA world (mirror back)
        b_c = FR.to_vantage(b_g - self.offset)
        d = b_c - a_c
        dist = float(np.linalg.norm(d))
        if dist < 1e-6:
            return True
        start = a_c + d / dist * min(VIS_TOL_M, dist / 2)
        try:
            hits = w.cast_ray(_loc(self.carla, start), _loc(self.carla, b_c))
        except Exception:                              # noqa: BLE001 (a server without cast_ray)
            self._no_cast = True
            self._note('Visibility from the cameras\' fields only: World.cast_ray is not available.')
            return True
        if not hits:
            return True
        h = hits[0].location
        return float(np.linalg.norm(np.array([h.x, h.y, h.z]) - start)) >= dist - min(VIS_TOL_M, dist / 2) - VIS_TOL_M

    # ------------------------------------------------------------------ the end
    def _map_png(self, world):
        """The drivable area (the map's lanes, World Map.generate_waypoints) at 0.1 m per pixel, its bottom-left
        pixel corner at the global origin; None when the map gives no waypoints."""
        import cv2
        try:
            wps = world.get_map().generate_waypoints(2.0)
        except Exception:                              # noqa: BLE001
            return None
        if not wps:
            return None
        pts, widths, yaws = [], [], []
        for w in wps:
            tr = w.transform
            pts.append([tr.location.x, tr.location.y])
            widths.append(float(getattr(w, 'lane_width', 3.5)))
            yaws.append(-math.radians(tr.rotation.yaw))
        P = np.array(pts, float)
        P[:, 1] *= -1.0
        P += self.offset[:2]
        hi = np.max(P, axis=0) + 20.0
        Wpx = int(min(math.ceil(max(hi[0], 1.0) / 0.1), MAP_MAX_PX))
        Hpx = int(min(math.ceil(max(hi[1], 1.0) / 0.1), MAP_MAX_PX))
        img = np.zeros((Hpx, Wpx), np.uint8)
        for (x, y), wd, yw in zip(P, widths, yaws):
            if x < -5 or y < -5 or x > Wpx * 0.1 + 5 or y > Hpx * 0.1 + 5:
                continue
            c, s = math.cos(yw), math.sin(yw)
            corners = np.array([[x + c * dx - s * dy, y + s * dx + c * dy] for dx, dy in
                                ((-1.3, -wd / 2), (1.3, -wd / 2), (1.3, wd / 2), (-1.3, wd / 2))])
            px = np.c_[corners[:, 0] / 0.1, Hpx - corners[:, 1] / 0.1]
            cv2.fillConvexPoly(img, np.round(px * 4).astype(np.int32), 255, lineType=cv2.LINE_8, shift=2)
        ok, buf = cv2.imencode('.png', img, [int(cv2.IMWRITE_PNG_COMPRESSION), 6])
        if Wpx >= MAP_MAX_PX or Hpx >= MAP_MAX_PX:
            self._note(f'The map mask is cut at {MAP_MAX_PX * 0.1:.0f} m from the global origin.')
        return buf.tobytes() if ok else None

    def _finish(self, ann_rows, instances, perf, cam_frames, phase_meas, t_wall, t_rec0):
        self._alive('finish')                         # (no server calls here: the watchdog waits)
        rec = self.rec
        inst_rows = []
        for inst, (cat, idx) in instances.items():
            for a, b in zip(idx[:-1], idx[1:]):
                ann_rows[a]['next'] = ann_rows[b]['token']
                ann_rows[b]['prev'] = ann_rows[a]['token']
            inst_rows.append({'token': inst, 'category_token': token('vantage', 'category', cat),
                              'nbr_annotations': len(idx), 'first_annotation_token': ann_rows[idx[0]]['token'],
                              'last_annotation_token': ann_rows[idx[-1]]['token']})
        self.progress(0.96, 'Writing the tables')
        bag_res = rec.close_bag()
        map_png = self._map_png(self.world) if rec.do_nus else None
        if rec.do_nus and map_png is None:
            import cv2
            ok, buf = cv2.imencode('.png', np.zeros((16, 16), np.uint8))
            map_png = buf.tobytes()
            self._note('The map gave no waypoints: an empty map mask.')
        rig = self.rig
        sim_s = float(self.a.duration)
        wall = time.perf_counter() - t_wall
        rec_wall = time.perf_counter() - t_rec0 if t_rec0 is not None else wall
        lid_rep = []
        for L in rig['lidars']:
            ph = phase_meas.get(L['tag'])
            lid_rep.append({'tag': L['tag'], 'carla_channels': L['blueprint']['channels'], 'rings': L['rings'],
                            'azimuth_step_deg': L['azimuth_step_deg'], 'oversample': L['oversample'],
                            'first_azimuth_of_rotation_deg': ph,
                            'note': 'the first kept laser of a rotation, in CARLA\'s angle from its start (0 = on '
                                    'the grid)'})
        scene = {'town': self.report.get('town'), 'carla_version': self.report.get('server_version'),
                 'weather': self.report.get('weather', {}).get('preset') or self.report.get('weather', {}).get('vantage'),
                 'duration_s': sim_s, 'traffic': self.report.get('traffic'), 'seed': int(self.a.seed),
                 'vehicle': self.report.get('vehicle', {}).get('blueprint'), 'tick_hz': self.R,
                 'warmup_ticks': rig['simulation']['warmup_ticks'], 'keyframe_hz': self.P['keyframe_hz'],
                 'samples': len(self.P['samples']), 'global_offset_m': [float(v) for v in self.offset],
                 'ego_origin_in_actor_m': self.report.get('vehicle', {}).get('ego_origin_in_actor_m')}
        lab = {}
        for tag, st in getattr(self, 'sem_lidar', {}).items():
            lab[tag] = {'slices': st['slices'], 'points': st['points'],
                        'matched_in_order_share': round(1.0 - st['fallback_slices'] / max(st['slices'], 1), 6),
                        'fallback_slices': st['fallback_slices'], 'count_mismatch_points': st['count_mismatch'],
                        'max_direction_error_deg': st['max_dir_err_deg'],
                        'range_rms_diff_m': round(math.sqrt(st['range_diff_sq'] / st['range_diff_n']), 5)
                        if st['range_diff_n'] else None,
                        'note': 'the ray-cast lidar against the semantic lidar, same rays: the range differs by the '
                                'ray-cast lidar\'s noise only'}
        parts = {}
        for name, cnt in self.part_tags.items():
            tot = int(cnt.sum())
            parts[name] = {'pixels': tot, 'carla_tags': {GT.CLASS_NAMES[k]: round(int(v) / max(tot, 1), 5)
                                                         for k, v in enumerate(cnt) if v},
                           'labelled_before': round(int(cnt[GT.CLASS_ID[name]]) / max(tot, 1), 5),
                           'labelled_after': 1.0,
                           'note': f'pixels of the {name.replace("_", " ")}s\' boxes (a light\'s heads: its '
                                   'get_light_boxes; a sign: its level box) and the light\'s own pixels next to them: '
                                   'the share CARLA had tagged ' + name + ', and after (all of them)'}
        gtr = {'settings': dict(self.gt), 'camera_kinds': list(self.gt_kinds), 'seconds': round(self._gt_seconds, 2),
               'lidar_labels': lab, 'lights_and_signs': parts, 'index': GT.INDEX if (self.gt_kinds or lab) else None}
        scen = self.scenario.report() if self.scenario is not None else \
            {'name': 'actors' if self.mode == 'actors' else self.mode, 'params': dict(self.scen['params']),
             'start': self.report.get('start'), 'actors': [], 'events': []}
        self.report.update({
            'faults_injected': self.faults_injected, 'scenario': scen, 'ground_truth': gtr,
            'actors': self.scene_actors})
        if getattr(self, 'cosim', None) is not None:   # SUMO drove the traffic: its block (only then)
            tr = self.cosim.report()
            self.report['traffic_source'] = tr
            self.report['traffic'] = tr['vehicles_spawned']
            scene['traffic'] = tr['vehicles_spawned']
        self.report.update({
            'status': 'ok', 'scene': scene, 'cameras': cam_frames, 'lidars': lid_rep,
            'performance': {'wall_s': round(wall, 2), 'recording_wall_s': round(rec_wall, 2),
                            'simulated_s': sim_s, 'steps': self.P['ticks'],
                            'steps_per_wall_s': round(self.P['ticks'] / max(rec_wall, 1e-6), 2),
                            'realtime_factor': round(sim_s / max(rec_wall, 1e-6), 4),
                            **{k: round(v, 2) for k, v in perf.items()}},
            'notes': self.notes + list(rig.get('notes', []))})
        rec.notes = list(self.notes)
        doc = rec.finish(ann_rows, inst_rows, map_png, scene, self.report, bag_res)
        self.report['counts'] = doc['counts']
        self.report['bytes'] = doc['bytes']
        self.report['channels'] = doc['channels']
        _write_json(os.path.join(self.out_dir, 'run.json'), self.report)
        self.progress(1.0, 'Done')

    def _tidy(self, world):
        """Stop and destroy everything spawned and restore the settings: always, also after an error. The order
        matters: a vehicle still on the traffic manager's autopilot when it is destroyed makes the manager's thread
        touch a dead actor (CARLA then aborts the client), so the vehicles leave the autopilot first, a step lets the
        manager drop them, the sensors stop listening, everything is destroyed, a step completes that, and only then do
        the manager and the world go back to asynchronous mode."""
        dead = []
        self._alive('tidy')
        cs = getattr(self, 'cosim', None)
        if cs is not None:                             # SUMO stopped, CARLA's lights given back (never raises)
            try:
                cs.close(ok=getattr(self, 'report', {}).get('status') == 'ok')
            except Exception:                          # noqa: BLE001
                pass

        def quiet(fn, *args):
            if dead:                                   # the server stopped answering: nothing more to ask it
                return None
            self._alive()
            try:
                return fn(*args)
            except Exception as e:                     # noqa: BLE001 (tidying up after anything)
                if 'time-out' in str(e) or 'timeout' in str(e).lower():
                    dead.append(str(e))
                return None
        sync = bool(getattr(world.get_settings(), 'synchronous_mode', False)) if quiet(world.get_settings) else False
        tm_port = int(getattr(self.a, 'tm_port', 8000))
        for a in self.actors:                          # walkers' AI controllers stop before anything is destroyed
            if a.type_id.startswith('controller.'):
                quiet(a.stop)
        for a in self.actors:
            if a.type_id.startswith('vehicle.'):
                quiet(a.set_autopilot, False, tm_port)
                if hasattr(a, 'disable_constant_velocity'):
                    quiet(a.disable_constant_velocity)
        if sync:
            quiet(world.tick)
        for act, kind, doc, idx in self.sensors:
            quiet(act.stop)
        for a in reversed(self.actors):
            quiet(a.destroy)
        self.actors = []
        self.sensors = []
        if sync:
            quiet(world.tick)
        if getattr(self, 'tm', None) is not None:
            quiet(self.tm.set_synchronous_mode, False)
        quiet(world.apply_settings, self.original)


# =========================================================================== scenarios
class Scenario:
    """A built-in scenario (SCENARIOS), deterministic with the seed: its actors spawned at waypoints relative to the
    ego's start (a spawn point on a straight lane, chosen by the seed), driven by the traffic manager (a desired
    speed, no lane changes of its own, traffic lights ignored: a scenario plays out on its own) or directly (a brake,
    a forced lane change, a walker's control, a bicycle's constant velocity), and its events: what happened when
    (seconds from the start of the recording) and the distances then."""

    def __init__(self, bridge, world, lib, start, rng):
        self.b = bridge
        self.world, self.lib, self.start_info, self.rng = world, lib, start, rng
        self.carla = bridge.carla
        self.name = bridge.scen['name']
        self.p = dict(bridge.scen['params'])
        self.tm = bridge.tm
        self.tm_port = int(bridge.a.tm_port)
        self.events = []
        self.roles = {}
        self.info = {}
        self.done = set()
        self.meas = {}

    # ------------------------------------------------------------------ helpers
    def _event(self, t, what, **kw):
        self.done.add(what)
        self.events.append(dict({'t_s': round(float(t), 4), 'what': what},
                                **{k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v)
                                   for k, v in kw.items()}))

    @staticmethod
    def _fwd(tr) -> np.ndarray:
        a = math.radians(tr.rotation.yaw)
        return np.array([math.cos(a), math.sin(a), 0.0])

    @staticmethod
    def _right(tr) -> np.ndarray:                     # CARLA's axes are left-handed: right of yaw 0 is +y
        a = math.radians(tr.rotation.yaw)
        return np.array([-math.sin(a), math.cos(a), 0.0])

    @staticmethod
    def _loc(x) -> np.ndarray:
        L = x.location if hasattr(x, 'location') else x
        return np.array([L.x, L.y, L.z], float)

    @staticmethod
    def _speed(snap_actor) -> float:
        v = snap_actor.get_velocity()
        return float(math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z))

    def _ahead(self, wp, d):
        if d < -1e-3:
            pv = wp.previous(-d)
            return pv[0] if pv else None
        if d < 1e-3:
            return wp
        nx = wp.next(d)
        return nx[0] if nx else None

    def _bp_id(self, kind):
        bid, why = self.b._actor_blueprint(self.lib, kind, self.rng)
        if bid is None:
            raise RuntimeError(f'The {self.name} scenario needs a {kind}: {why}.')
        return bid

    def _spawn(self, bid, loc, yaw, role, lift=0.3):
        carla = self.carla
        bp = self.lib.find(bid)
        if bp.has_attribute('role_name'):
            bp.set_attribute('role_name', ROLE_PREFIX + 'scenario_' + role)
        act = None
        for extra in (0.0, 0.4, 1.0):
            act = self.world.try_spawn_actor(bp, carla.Transform(
                carla.Location(x=float(loc[0]), y=float(loc[1]), z=float(loc[2]) + lift + extra),
                carla.Rotation(pitch=0.0, yaw=float(yaw), roll=0.0)))
            if act is not None:
                break
        if act is None:
            raise RuntimeError(f'The {self.name} scenario could not spawn its {role}: the place is taken.')
        self.b._own(act)
        self.roles[role] = act
        return act

    def _carriageway(self, wp) -> tuple[float, float]:
        """Lateral offsets (m, along the lane's right) of the outer edges of the road's driving lanes at wp, both
        directions: CARLA's right lane moves away from the road's centre, its left one towards it and across."""
        r = self._right(wp.transform)
        c0 = self._loc(wp.transform)
        spans = [(0.0, float(wp.lane_width))]
        seen = {wp.lane_id}
        for first in ('right', 'left'):
            cur, step = wp, first
            for _ in range(10):
                try:
                    nxt = cur.get_right_lane() if step == 'right' else cur.get_left_lane()
                except Exception:                      # noqa: BLE001
                    nxt = None
                if nxt is None or nxt.lane_id in seen or str(getattr(nxt, 'lane_type', '')).split('.')[-1] != 'Driving':
                    break
                if (nxt.lane_id > 0) != (cur.lane_id > 0):
                    step = 'right'                     # across the centre: away from it from now on
                seen.add(nxt.lane_id)
                spans.append((float((self._loc(nxt.transform) - c0) @ r), float(nxt.lane_width)))
                cur = nxt
        return min(o - w / 2 for o, w in spans), max(o + w / 2 for o, w in spans)

    def _drive(self, v, kmh):
        v.set_autopilot(True, self.tm_port)
        for fn, args in (('auto_lane_change', (v, False)), ('ignore_lights_percentage', (v, 100.0)),
                         ('ignore_signs_percentage', (v, 100.0)), ('set_desired_speed', (v, float(kmh)))):
            f = getattr(self.tm, fn, None)
            if f is not None:
                try:
                    f(*args)
                except Exception:                      # noqa: BLE001
                    pass

    # ------------------------------------------------------------------ the scenario
    def spawn(self):
        b = self.b
        wp0 = self.start_info['waypoint']
        if wp0 is None:
            raise RuntimeError(f'The {self.name} scenario needs the ego on a lane.')
        L_ego = 2.0 * float(b.ego.bounding_box.extent.x)
        if self.name == 'lead_vehicle_brake':
            bid = self._bp_id('car')
            wp = self._ahead(wp0, self.p['gap_m'] + L_ego / 2 + 2.3)
            if wp is None:
                raise RuntimeError('The lane ends before the lead vehicle\'s place.')
            lead = self._spawn(bid, self._loc(wp.transform), wp.transform.rotation.yaw, 'lead')
            d = self.p['gap_m'] + L_ego / 2 + float(lead.bounding_box.extent.x)    # its real length
            wp = self._ahead(wp0, d) or wp
            tr = wp.transform
            lead.set_transform(self.carla.Transform(self.carla.Location(x=tr.location.x, y=tr.location.y,
                                                                         z=tr.location.z + 0.3), tr.rotation))
            self.info['lead_start_m'] = round(d, 3)
        elif self.name == 'cut_in':
            adj = b._adjacent(wp0, self.p['side'], 60.0)
            if adj is None:
                raise RuntimeError('The ego\'s lane has no lane beside it in the same direction whose marking allows '
                                   'a lane change into it.')
            wpa, side = adj
            if side != self.p['side']:
                b._note(f'Cut-in: no lane on the {self.p["side"]}; the cutter comes from the {side}.')
            wpc = self._ahead(wpa, self.p['cutter_offset_m'])
            if wpc is None:
                raise RuntimeError('The next lane ends before the cutter\'s place.')
            self._spawn(self._bp_id('car'), self._loc(wpc.transform), wpc.transform.rotation.yaw, 'cutter')
            self.info.update(side=side, lane_width_m=float(wp0.lane_width))
        elif self.name in ('pedestrian_crossing', 'cyclist_crossing'):
            wpx = self._ahead(wp0, self.p['distance_m'])
            if wpx is None:
                raise RuntimeError('The lane ends before the crossing.')
            lo, hi = self._carriageway(wpx)
            side = self.p['side']
            r = self._right(wpx.transform)
            c0 = self._loc(wpx.transform)
            s0, s1 = (hi + 1.0, lo - 1.0) if side == 'right' else (lo - 1.0, hi + 1.0)
            d = r * (1.0 if s1 > s0 else -1.0)
            yaw = math.degrees(math.atan2(d[1], d[0]))
            if self.name == 'pedestrian_crossing':
                act = self._spawn(self._bp_id('child' if self.p['walker'] == 'child' else 'pedestrian'),
                                  c0 + r * s0, yaw, 'pedestrian', lift=1.0)
            else:
                act = self._spawn(self._bp_id('cyclist'), c0 + r * s0, yaw, 'cyclist')
                self.b._hold(act)
            self.info.update(origin=c0, right=r, dir=d, from_m=s0, to_m=s1, lane_width_m=float(wpx.lane_width),
                             road_m=[round(lo, 3), round(hi, 3)])
        self.report_start = {k: v for k, v in self.start_info.items() if k != 'waypoint'}

    def start(self):
        """After the first step: the vehicles on the autopilot at their speeds (the crossing actor waits)."""
        b = self.b
        p = self.p
        if self.name == 'lead_vehicle_brake':
            self._drive(b.ego, p['speed_kmh'])
            self._drive(self.roles['lead'], p['speed_kmh'])
        elif self.name == 'cut_in':
            self._drive(b.ego, p['speed_kmh'])
            self._drive(self.roles['cutter'], p['cutter_speed_kmh'])
        else:
            self._drive(b.ego, p['ego_speed_kmh'])
            for fn in ('ignore_walkers_percentage', 'ignore_vehicles_percentage'):
                f = getattr(self.tm, fn, None)
                if f is not None:
                    try:
                        f(b.ego, 0.0)
                    except Exception:                  # noqa: BLE001
                        pass

    def step(self, rel, t, snap):
        """One simulation step (t seconds from the start of the recording; negative while the sensors warm up)."""
        b = self.b
        carla = self.carla
        es = snap.find(b.ego.id)
        if es is None:
            return
        tr_e = es.get_transform()
        pe, fe = self._loc(tr_e), self._fwd(tr_e)
        v_e = self._speed(es)
        L_ego = float(b.ego.bounding_box.extent.x)
        if self.name == 'lead_vehicle_brake':
            lead = self.roles['lead']
            ls = snap.find(lead.id)
            if ls is None:
                return
            gap = float((self._loc(ls.get_transform()) - pe) @ fe) - L_ego - float(lead.bounding_box.extent.x)
            if 'lead_brake' not in self.done and t >= self.p['brake_at_s']:
                lead.set_autopilot(False, self.tm_port)
                lead.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0))
                self._event(t, 'lead_brake', gap_m=gap, ego_speed_mps=v_e, lead_speed_mps=self._speed(ls))
            if 'lead_brake' in self.done:
                self.meas['min_gap_m'] = round(min(self.meas.get('min_gap_m', math.inf), gap), 4)
                if 'lead_stopped' not in self.done and self._speed(ls) < 0.1:
                    self._event(t, 'lead_stopped', gap_m=gap)
                if 'ego_stopped' not in self.done and v_e < 0.1:
                    self._event(t, 'ego_stopped', gap_m=gap)
        elif self.name == 'cut_in':
            cut = self.roles['cutter']
            cs = snap.find(cut.id)
            if cs is None:
                return
            pc = self._loc(cs.get_transform())
            wp_e = self.world.get_map().get_waypoint(tr_e.location)
            lat = float((pc - self._loc(wp_e.transform)) @ self._right(wp_e.transform)) if wp_e is not None else \
                float((pc - pe) @ self._right(tr_e))
            lon = float((pc - pe) @ fe)
            gap = lon - L_ego - float(cut.bounding_box.extent.x)
            if 'lane_change_ordered' not in self.done and t >= self.p['cut_at_s'] and gap >= self.p['cut_gap_m']:
                # (CARLA 0.9.16's traffic manager left forced lane changes undone on Town10HD, asked every step on a
                # lane whose marking allows them: the cutter is steered itself, then handed back)
                cut.set_autopilot(False, self.tm_port)
                self.info['target_lane'] = (wp_e.road_id, wp_e.lane_id) if wp_e is not None else None
                self._event(t, 'lane_change_ordered', longitudinal_m=lon, gap_m=gap, lateral_m=lat,
                            ego_speed_mps=v_e, cutter_speed_mps=self._speed(cs))
            if 'lane_change_ordered' in self.done and 'cut_in_complete' not in self.done:
                self._steer_cutter(cut, cs, t, lat)
            if 'lane_change_ordered' in self.done and 'cut_in' not in self.done and \
                    abs(lat) < 0.5 * self.info.get('lane_width_m', 3.5):
                self._event(t, 'cut_in', longitudinal_m=lon, gap_m=lon - L_ego - float(cut.bounding_box.extent.x),
                            ego_speed_mps=v_e, cutter_speed_mps=self._speed(cs))
            if 'cut_in' in self.done and lon > 0:
                self.meas['min_gap_m'] = round(min(self.meas.get('min_gap_m', math.inf),
                                                   lon - L_ego - float(cut.bounding_box.extent.x)), 4)
        else:
            role = 'pedestrian' if self.name == 'pedestrian_crossing' else 'cyclist'
            act = self.roles[role]
            sa = snap.find(act.id)
            if sa is None:
                return
            pa = self._loc(sa.get_transform())
            s_lat = float((pa - self.info['origin']) @ self.info['right'])
            dist = float(np.linalg.norm((pa - pe)[:2]))
            if 'start' not in self.done and t >= self.p['start_at_s']:
                d = self.info['dir']
                if role == 'pedestrian':
                    act.apply_control(carla.WalkerControl(direction=carla.Vector3D(x=float(d[0]), y=float(d[1]), z=0.0),
                                                          speed=float(self.p['walk_speed_mps'])))
                else:
                    act.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0, hand_brake=False))
                    act.enable_constant_velocity(carla.Vector3D(x=float(self.p['ride_speed_mps']), y=0.0, z=0.0))
                self._event(t, 'start', distance_m=dist, ego_speed_mps=v_e)
            if 'start' in self.done and 'done' not in self.done:
                half = 0.5 * self.info['lane_width_m']
                if 'enters_ego_lane' not in self.done and abs(s_lat) < half:
                    self._event(t, 'enters_ego_lane', distance_m=dist, ego_speed_mps=v_e)
                if 'enters_ego_lane' in self.done and 'leaves_ego_lane' not in self.done and abs(s_lat) >= half:
                    self._event(t, 'leaves_ego_lane', distance_m=dist, ego_speed_mps=v_e)
                if (s_lat - self.info['to_m']) * np.sign(self.info['to_m'] - self.info['from_m']) >= 0:
                    if role == 'pedestrian':
                        act.apply_control(carla.WalkerControl(speed=0.0))
                    else:
                        act.disable_constant_velocity()
                        b._hold(act)
                    self._event(t, 'done', distance_m=dist)
                if 'ego_stopped' not in self.done and v_e < 0.1 and float((pa - pe) @ fe) > 0:
                    self._event(t, 'ego_stopped', distance_m=dist)
                self.meas['min_distance_m'] = round(min(self.meas.get('min_distance_m', math.inf), dist), 4)

    def _steer_cutter(self, cut, cs, t, lat):
        """One step of the cutter's lane change: pure pursuit of a point 6–15 m ahead on the lane beside it towards the
        ego until it is in the ego's lane, then on its own lane (the front wheels' steering limit from its physics), its
        speed held at the scenario's; back on the autopilot once it has run within 0.3 m of its lane's centre and 3° of
        its heading for half a second, or after 8 s."""
        carla = self.carla
        m = self.world.get_map()
        tr = cs.get_transform()
        p = self._loc(tr)
        wp = m.get_waypoint(tr.location)
        if wp is None:
            return
        own = wp                                       # the lane it is in (the road's segments change ids: by lane)
        if 'cut_in' not in self.done:                  # until it is in: the lane beside it, towards the ego
            go_right = self.info.get('side') == 'left'
            try:
                w2 = wp.get_right_lane() if go_right else wp.get_left_lane()
            except Exception:                          # noqa: BLE001
                w2 = None
            wp = w2 if w2 is not None and (w2.lane_id > 0) == (wp.lane_id > 0) else wp
        v = self._speed(cs)
        ld = min(max(0.6 * v + 4.0, 6.0), 15.0)
        nx = wp.next(ld)
        if not nx:
            return
        q = self._loc(nx[0].transform) - p
        h = self._fwd(tr)
        alpha = math.atan2(h[0] * q[1] - h[1] * q[0], h[0] * q[0] + h[1] * q[1])   # CARLA's axes: + to the right
        if 'max_steer' not in self.info:
            try:                                       # wheels front left, front right, rear left, rear right (cm)
                wh = cut.get_physics_control().wheels
                self.info['max_steer'] = math.radians(float(max(w.max_steer_angle for w in wh)))
                pts = np.array([[w.position.x, w.position.y] for w in wh], float) / 100.0
                self.info['wheelbase'] = float(np.linalg.norm(pts[:2].mean(0) - pts[2:4].mean(0)))
            except Exception:                          # noqa: BLE001
                self.info['max_steer'], self.info['wheelbase'] = math.radians(70.0), 2.7
        delta = math.atan(2.0 * self.info['wheelbase'] * math.sin(alpha) / ld)
        e = self.p['cutter_speed_kmh'] / 3.6 - v
        cut.apply_control(carla.VehicleControl(throttle=float(min(max(0.2 + 0.3 * e, 0.0), 1.0)) if e > -0.3 else 0.0,
                                               brake=float(min(-0.3 * e, 1.0)) if e < -1.5 else 0.0,
                                               steer=float(min(max(delta / self.info['max_steer'], -1.0), 1.0))))
        off = float((p - self._loc(own.transform)) @ self._right(own.transform))   # from its own lane's centre
        turn = abs(_wrap180(tr.rotation.yaw - own.transform.rotation.yaw))
        settled = 'cut_in' in self.done and abs(off) < 0.3 and turn < 3.0
        self.info['settled'] = self.info.get('settled', 0) + 1 if settled else 0
        lat = off if 'cut_in' in self.done else lat
        started = next(e_['t_s'] for e_ in self.events if e_['what'] == 'lane_change_ordered')
        if self.info['settled'] >= 0.5 / self.b.dt or t - started > 8.0:     # (lateral_m: from its lane's centre)
            self._drive(cut, self.p['cutter_speed_kmh'])
            self._event(t, 'cut_in_complete' if self.info['settled'] else 'cut_in_abandoned', lateral_m=lat)
            self.done.add('cut_in_complete')

    def report(self) -> dict:
        spec = SCENARIOS[self.name]
        return {'name': self.name, 'label': spec[0], 'about': spec[1], 'params': self.p,
                'start': getattr(self, 'report_start', None),
                'actors': [{'role': r, 'actor_id': int(a.id), 'blueprint': a.type_id,
                            'gt_id': int(self.b.spawn_index.get(int(a.id), 0))} for r, a in self.roles.items()],
                'geometry': {k: (np.round(v, 4).tolist() if isinstance(v, np.ndarray) else v)
                             for k, v in self.info.items()},
                'events': self.events, 'measures': self.meas,
                'notes': ['Times from the start of the recording (t = 0, the sync pulse); the vehicles\' speeds are '
                          'the traffic manager\'s desired speeds (traffic lights and signs ignored).']}


# =========================================================================== small helpers
def _rot_x(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)


def _rot_y(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], float)


def _rot_xy(ax, ay):
    return _rot_y(math.degrees(ay)) @ _rot_x(math.degrees(ax))


def pose_matrix(p) -> np.ndarray:
    """Body -> vehicle rotation of a Vantage pose (yaw, pitch, roll degrees; Rz·Ry·Rx, pitch positive down)."""
    return FR.rot_z(float(p['yaw'])) @ _rot_y(float(p['pitch'])) @ _rot_x(float(p['roll']))


def euler_zyx(R) -> list:
    """[yaw, pitch, roll] degrees of a rotation (the inverse of pose_matrix; pitch in [−90, 90])."""
    R = np.asarray(R, float)
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, -float(R[2, 0])))))
    if abs(R[2, 0]) < 1.0 - 1e-12:
        yaw = math.degrees(math.atan2(R[1, 0], R[0, 0]))
        roll = math.degrees(math.atan2(R[2, 1], R[2, 2]))
    else:
        yaw = math.degrees(math.atan2(-R[0, 1], R[1, 1]))
        roll = 0.0
    return [yaw, pitch, roll]


def _wrap180(a: float) -> float:
    return (float(a) + 180.0) % 360.0 - 180.0


def _in_box(P, b, margin=0.1) -> np.ndarray:
    q = (P - b['centre']) @ b['R']
    L, W, H = b['size_lwh']
    return (np.abs(q[:, 0]) <= L / 2 + margin) & (np.abs(q[:, 1]) <= W / 2 + margin) & \
        (np.abs(q[:, 2]) <= H / 2 + margin)


def _box_points(L, W, H):
    """Corner and face-centre points of a box about its centre and their outward normals (corners: the diagonal)."""
    P, N = [], []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                P.append([sx * L / 2, sy * W / 2, sz * H / 2])
                n = np.array([sx, sy, sz], float)
                N.append(n / np.linalg.norm(n))
    for ax, half in ((0, L / 2), (1, W / 2), (2, H / 2)):
        for sgn in (-1, 1):
            p = [0.0, 0.0, 0.0]
            p[ax] = sgn * half
            n = [0.0, 0.0, 0.0]
            n[ax] = float(sgn)
            P.append(p)
            N.append(n)
    return np.array(P), np.array(N)


def _png(a) -> bytes:
    import cv2
    ok, buf = cv2.imencode('.png', np.ascontiguousarray(a), [int(cv2.IMWRITE_PNG_COMPRESSION), 1])
    if not ok:
        raise RuntimeError('PNG encoding failed')
    return buf.tobytes()


def _write_json(path, doc):
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        fh.write(json.dumps(doc, indent=1, default=_jsonable) + '\n')
    os.replace(tmp, path)


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)


def render_mpx(rig: dict, a) -> float:
    """Megapixels the run's cameras render per frame: their views, once for the image and once per ground-truth
    camera kind."""
    if getattr(a, 'no_cameras', False):
        return 0.0
    kinds = gt_camera_kinds(getattr(a, 'gt', None) or clean_ground_truth(rig.get('ground_truth', {})))
    return float(sum(c.get('render_mpx', 0.0) for c in rig.get('cameras', []))) * (1 + len(kinds))


def _watchdog(b, stop, life, sigs, timeout):
    """The bridge's watchdog thread: takes SIGINT / SIGTERM (blocked in the main thread, so they arrive even while
    a call into CARLA's client blocks it) and sets the cancel flag; ends the process when the main thread does not
    finish within SIGNAL_GRACE_S of a signal (a second signal: at once), or when the server gives no sign of life for
    twice the timeout while the run talks to it (CARLA's client then hangs in a call: seen when a server crashed).
    run.json says why; what was written is removed (exit 3 cancelled, 4 the server stopped answering)."""
    limit = 2.0 * max(float(timeout), 1.0)
    while life.get('phase') != 'done':
        got = signal.sigtimedwait(sigs, 0.5) if sigs else time.sleep(0.5)
        now = time.monotonic()
        if got is not None:
            if stop['flag']:
                stop['at'] = now - SIGNAL_GRACE_S - 1.0     # a second signal: no more waiting
            else:
                stop['flag'] = True
                stop['at'] = now
        why = None
        if stop['at'] is not None and now - stop['at'] > SIGNAL_GRACE_S:
            why = ('cancelled', 'stopped by a signal; the CARLA server did not answer while tidying up, so it may be '
                                'left in synchronous mode with the run\'s actors (carla_bridge --tidy --port P)')
        elif life.get('phase') in ('setup', 'run', 'tidy') and now - life['t'] > limit:
            why = ('failed', f'the CARLA server stopped answering (no reply for {limit:g} s): the run was aborted; the '
                             'server may be left in synchronous mode with its actors (carla_bridge --tidy --port P)')
        if why is None:
            continue
        status, msg = why
        try:
            if getattr(b, 'cosim', None) is not None:
                b.cosim.kill()
        except Exception:                              # noqa: BLE001
            pass
        try:
            if b.rec is not None:
                b.rec.cleanup()
            b.report.update(status=status, error=msg, notes=b.notes, server_stopped=status == 'failed')
            _write_json(os.path.join(b.out_dir, 'run.json'), b.report)
        except Exception:                              # noqa: BLE001 (ending regardless)
            pass
        print(f'{"Cancelled" if status == "cancelled" else "Failed"}: {msg}', file=sys.stderr, flush=True)
        sys.stdout.flush()
        os._exit(3 if status == 'cancelled' else 4)


# =========================================================================== tidying a server after a killed bridge
def _role(actor) -> str:
    try:
        return str((getattr(actor, 'attributes', None) or {}).get('role_name', ''))
    except Exception:                                  # noqa: BLE001
        return ''


def tidy_server(carla, host: str, port: int, timeout: float = 20.0) -> dict:
    """After a bridge was killed (SIGKILL, its memory cap) the server stays in synchronous mode with its actors: this
    stops and destroys what a bridge spawned and nothing else — the sensors named ROLE_SENSOR, the vehicles, walkers
    and props whose role name starts with ROLE_PREFIX, a 'hero' only when it carries such a sensor, the AI controllers
    of those walkers — and puts the world back in asynchronous mode (variable step). Returns what it did."""
    client = carla.Client(host, int(port))
    client.set_timeout(float(timeout))
    world = client.get_world()
    # a client knows the actors from the world snapshots it is sent, and a synchronous server nobody ticks sends none:
    # measured on 0.9.16, a fresh client listed 0 of the 51 actors a killed bridge left. One step first.
    if world.get_settings().synchronous_mode:
        world.tick()
    else:
        world.wait_for_tick()
    acts = list(world.get_actors())

    def quiet(fn, *args):
        try:
            return fn(*args)
        except Exception:                              # noqa: BLE001
            return None
    sensors = [a for a in acts if a.type_id.startswith('sensor.') and _role(a) == ROLE_SENSOR]
    egos = {a.parent.id for a in sensors if getattr(a, 'parent', None) is not None and _role(a.parent) == 'hero'}
    mine = [a for a in acts if not a.type_id.startswith(('sensor.', 'controller.')) and
            (_role(a).startswith(ROLE_PREFIX) or a.id in egos)]
    walkers = {a.id for a in mine if a.type_id.startswith('walker.')}
    ctrls = [a for a in acts if a.type_id.startswith('controller.') and getattr(a, 'parent', None) is not None and
             a.parent.id in walkers]
    st = world.get_settings()
    sync = bool(st.synchronous_mode)
    for c in ctrls:
        quiet(c.stop)
    for a in mine:
        if a.type_id.startswith('vehicle.'):
            quiet(a.set_autopilot, False)
    for s in sensors:
        quiet(s.stop)
    if sync:
        quiet(world.tick)
    gone = sum(1 for a in ctrls + sensors + mine if quiet(a.destroy))
    if any(_role(a) == ROLE_SUMO for a in mine):       # a SUMO run may have frozen the lights: CARLA runs them again
        for tl in [a for a in acts if a.type_id == 'traffic.traffic_light' and hasattr(a, 'freeze')]:
            quiet(tl.freeze, False)
    if sync:
        quiet(world.tick)
    st = world.get_settings()
    st.synchronous_mode = False
    st.fixed_delta_seconds = None
    world.apply_settings(st)
    return {'destroyed': gone, 'sensors': len(sensors), 'controllers': len(ctrls),
            'vehicles': sum(1 for a in mine if a.type_id.startswith('vehicle.')), 'walkers': len(walkers),
            'props': sum(1 for a in mine if a.type_id.startswith('static.')), 'was_synchronous': sync,
            'synchronous': False}


# =========================================================================== command line
def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog='python -m vantage.sim.carla_bridge',
                                 description='Run a Vantage rig (sim_rig.json from core.simrig) in CARLA and record it '
                                             'as a nuScenes folder and a ROS 2 bag.')
    ap.add_argument('rig', nargs='?', help='sim_rig.json (sim_remap.npz next to it)')
    ap.add_argument('out', nargs='?', help='output folder (nuscenes/, rosbag/, dataset.json, run.json)')
    ap.add_argument('--render-budget-mpx', type=float, default=RENDER_BUDGET_MPX, metavar='MPX',
                    help='refuse a run whose cameras render more megapixels per frame (images and ground truth) than '
                         f'this (default {RENDER_BUDGET_MPX:g}: a 0.9.16 server under 12 GB crashed at 306); 0: no limit')
    ap.add_argument('--tidy', action='store_true',
                    help='no run: after a bridge was killed, destroy what bridges spawned on the server at --port (only '
                         'those) and put it back in asynchronous mode')
    ap.add_argument('--town', default=None, help='CARLA map (default: the rig\'s)')
    ap.add_argument('--weather', default=None, help='a CARLA weather preset (ClearNoon …) or "rig" for the rig\'s '
                                                    'mapped Vantage condition (default: the rig\'s)')
    ap.add_argument('--duration', type=float, default=None, help='seconds recorded (default: the rig\'s)')
    ap.add_argument('--traffic', type=int, default=None, help='other vehicles on autopilot (default: the rig\'s)')
    ap.add_argument('--seed', type=int, default=None, help='spawn choices and the traffic manager (default: the rig\'s)')
    ap.add_argument('--vehicle', default=None, help='the ego\'s CARLA blueprint (default: the rig\'s choice)')
    ap.add_argument('--warmup', type=float, default=None, help='seconds of traffic before recording')
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=None, help='the server\'s RPC port (default: the rig\'s, 2000)')
    ap.add_argument('--tm-port', type=int, default=None, help='the traffic manager\'s port (default: the rig\'s, 8100)')
    ap.add_argument('--timeout', type=float, default=60.0, help='seconds to wait for the server or a sensor')
    ap.add_argument('--no-cameras', action='store_true', help='leave the cameras out (a server without rendering)')
    ap.add_argument('--keep-vignette', action='store_true',
                    help='keep Unreal\'s vignette in every pinhole view (seams where a fisheye\'s views meet)')
    ap.add_argument('--progress-json', action='store_true', help='progress as JSON lines on stdout')
    ap.add_argument('--ground-truth', default=None, metavar='LIST',
                    help='ground truth to record: a comma list of semantic, instance, range, lidar_labels, or "none" '
                         '(default: the rig\'s, all of them)')
    ap.add_argument('--faults', default=None, metavar='FAULTS.json',
                    help='inject faults: a JSON list of {"tag", "kind", ...} (kinds: ' + ', '.join(FAULT_KINDS) +
                         '); "none" for none (default: the rig\'s)')
    ap.add_argument('--scenario', default=None, choices=list(SCENARIOS),
                    help='a built-in scenario (default: the rig\'s, free_drive)')
    ap.add_argument('--scenario-params', default=None, metavar='JSON', help='the scenario\'s parameters as a JSON '
                                                                            'object (the rest: their defaults)')
    ap.add_argument('--actors', default=None, metavar='ACTORS.json',
                    help='an actors scene: the ego still with these actors around it, no traffic (a JSON list of '
                         '{"id", "kind", "pose": {"x", "y", "z", "yaw"}} in the ego\'s vehicle frame); "none" for none')
    ap.add_argument('--walkers', type=int, default=None, metavar='N',
                    help='pedestrians walking the sidewalks on CARLA\'s walker AI (default: the rig\'s, 0; not in an '
                         'actors scene or the parked scenario)')
    ap.add_argument('--spawn-point', type=int, default=None, metavar='N',
                    help='the ego\'s spawn point (default: the rig\'s; −1: chosen by the seed, on a straight road '
                         'for a scenario or an actors scene)')
    ap.add_argument('--traffic-source', default=None, choices=list(TRAFFIC_SOURCES),
                    help='who drives the other vehicles: CARLA\'s traffic manager or SUMO co-simulated in lockstep '
                         '(default: the rig\'s, carla)')
    ap.add_argument('--sumo-net', default=None, metavar='FILE',
                    help='SUMO traffic: "auto" (CARLA\'s bundled net for Town01, 04, 05, else generated from the map), '
                         '"generate", or a .net.xml or .sumocfg (default: the rig\'s)')
    ap.add_argument('--sumo-cache', default=None, metavar='DIR',
                    help='SUMO traffic: where nets generated from a map\'s OpenDRIVE are cached (default: the working '
                         'folder\'s sumo-nets)')
    ap.add_argument('--carla-root', default=None, metavar='DIR',
                    help='SUMO traffic: the CARLA folder holding Co-Simulation/Sumo (default: found next to this '
                         'Python\'s environment)')
    return ap


def _load_json_arg(value, what):
    """A JSON argument: a file path or inline JSON."""
    if value is None:
        return None
    if isinstance(value, str) and os.path.isfile(value):
        with open(value, encoding='utf-8') as fh:
            return json.load(fh)
    try:
        return json.loads(value)
    except ValueError as e:
        raise ValueError(f'{what}: neither a file nor JSON ({e}).') from None


def resolve_args(a, rig: dict):
    """Fill the arguments left out from the rig's scenario and server settings."""
    sc = rig['scenario']
    srv = rig.get('server', {})
    a.duration = float(a.duration if a.duration is not None else sc['duration_s'])
    a.traffic = int(a.traffic if a.traffic is not None else sc['traffic'])
    a.walkers = max(0, min(int(a.walkers if a.walkers is not None else sc.get('walkers', 0)), 300))
    a.seed = int(a.seed if a.seed is not None else sc['seed'])
    a.port = int(a.port if a.port is not None else srv.get('port', 2000))
    a.tm_port = int(a.tm_port if a.tm_port is not None else srv.get('tm_port', 8100))
    if a.weather is None and sc.get('weather') and sc['weather'] not in ('project',) and \
            not str(sc['weather']).startswith('vantage:'):
        a.weather = sc['weather']
    # ground truth, faults, the scenario, an actors scene: the command line over the rig's
    gt = clean_ground_truth(rig.get('ground_truth', {}))
    if a.ground_truth is not None:
        want = {x.strip() for x in str(a.ground_truth).split(',') if x.strip()}
        unknown = want - set(GT_KEYS) - {'none', 'all'}
        if unknown:
            raise ValueError(f'--ground-truth: unknown {", ".join(sorted(unknown))} (one of {", ".join(GT_KEYS)}).')
        gt = {k: ('all' in want or k in want) and 'none' not in want for k in GT_KEYS}
    a.gt = gt
    kinds = {c['tag']: 'camera' for c in rig.get('cameras', [])}
    kinds.update({x['tag']: k for k, key in (('lidar', 'lidars'), ('radar', 'radars'), ('imu', 'imus'),
                                             ('gnss', 'gnss')) for x in rig.get(key, [])})
    if a.faults is not None and str(a.faults).strip().lower() == 'none':
        raw = []
    else:
        raw = _load_json_arg(a.faults, '--faults') if a.faults is not None else rig.get('faults', [])
    a.fault_list, bad = clean_faults(raw, kinds)
    if bad:
        raise ValueError('Faults: ' + ' '.join(bad))
    params = _load_json_arg(a.scenario_params, '--scenario-params') if a.scenario_params is not None else None
    name = a.scenario or sc.get('name', 'free_drive')
    if params is None:
        params = sc.get('params', {}) if name == sc.get('name', 'free_drive') else {}
    a.scen, bad = clean_scenario(name, params)
    if bad:
        raise ValueError('Scenario: ' + ' '.join(bad))
    if a.actors is not None and str(a.actors).strip().lower() == 'none':
        items = []
    else:
        items = _load_json_arg(a.actors, '--actors') if a.actors is not None else \
            (rig.get('actors') or {}).get('items', [])
    a.actor_list, bad = clean_actors(items)
    if bad:
        raise ValueError('Actors: ' + ' '.join(bad))
    if a.spawn_point is None:
        a.spawn_point = int((rig.get('actors') or {}).get('spawn_point', -1))
    # who drives the traffic: the rig's traffic_source block (absent: CARLA's traffic manager), the command line over it
    ts = rig.get('traffic_source') or {}
    src = getattr(a, 'traffic_source', None) or ts.get('source') or 'carla'
    if getattr(a, 'sumo_net', None) and not getattr(a, 'traffic_source', None):
        src = 'sumo'                                   # (--sumo-net alone asks for SUMO traffic)
    if src not in TRAFFIC_SOURCES:
        raise ValueError(f'Traffic source {src!r}: one of {", ".join(TRAFFIC_SOURCES)}.')
    a.traffic_source = src
    if src == 'sumo':
        from . import sumo_cosim as SUMO
        d = dict(ts.get('sumo') or {})
        if getattr(a, 'sumo_net', None):
            n = str(a.sumo_net)
            d['net'] = n if n in SUMO.NET_CHOICES else os.path.abspath(os.path.expanduser(n))
        a.sumo, bad = SUMO.clean_sumo(d)
        if bad:
            raise ValueError('SUMO: ' + '; '.join(bad) + '.')
    else:
        a.sumo = None
    return a


def main(argv=None) -> int:
    a = parser().parse_args(argv)
    if a.tidy:
        if a.port is None:
            print('Failed: --tidy needs --port (the server to tidy).', file=sys.stderr)
            return 2
        import carla
        try:
            res = tidy_server(carla, a.host, a.port, a.timeout)
        except Exception as e:                         # noqa: BLE001
            print(f'Failed: {type(e).__name__}: {e}', file=sys.stderr)
            return 2
        print(json.dumps(res), flush=True)
        return 0
    if not a.rig or not a.out:
        print('Failed: a rig file and an output folder are needed (or --tidy).', file=sys.stderr)
        return 2
    try:
        rig, remaps = load_rig(a.rig)
        resolve_args(a, rig)
        mpx = render_mpx(rig, a)
        if a.render_budget_mpx > 0 and mpx > a.render_budget_mpx:
            raise ValueError(f'the cameras would render {mpx:.0f} Mpx per frame (images and ground truth), over the '
                             f'budget of {a.render_budget_mpx:g} (a 0.9.16 server capped at 12 GB crashed at 306): '
                             'lower the image size, record fewer cameras\' ground truth, or raise --render-budget-mpx '
                             'for a bigger server')
    except (OSError, ValueError) as e:
        print(f'Failed: {e}', file=sys.stderr)
        return 2
    stop = {'flag': False, 'at': None}
    life = {'t': time.monotonic(), 'phase': 'setup'}
    sigs = {signal.SIGINT, signal.SIGTERM}
    waited = hasattr(signal, 'pthread_sigmask') and hasattr(signal, 'sigtimedwait')
    if waited:                                         # taken by the watchdog thread, even while CARLA's call blocks
        signal.pthread_sigmask(signal.SIG_BLOCK, sigs)  # (before carla's own threads start: they inherit it)
    else:
        def on_signal(*_):
            stop['flag'] = True
            stop['at'] = time.monotonic()
        signal.signal(signal.SIGINT, on_signal)
        signal.signal(signal.SIGTERM, on_signal)
    last = [0.0]

    def progress(f, text=''):
        now = time.monotonic()
        if f < 1.0 and now - last[0] < 0.25:
            return
        last[0] = now
        if a.progress_json:
            print(json.dumps({'progress': round(float(f), 4), 'text': text}), flush=True)
        else:
            print(f'{100 * f:5.1f} %  {text}', flush=True)
    import carla
    b = Bridge(carla, rig, remaps, a.out, a, progress, lambda: stop['flag'])
    b.life = life
    os.makedirs(b.out_dir, exist_ok=True)
    import threading
    threading.Thread(target=_watchdog, args=(b, stop, life, sigs if waited else None, float(a.timeout)),
                     daemon=True).start()
    try:
        rep = b.run()
    except Cancelled:
        if b.rec is not None:
            b.rec.cleanup()
        b.report.update(status='cancelled', notes=b.notes)
        _write_json(os.path.join(b.out_dir, 'run.json'), b.report)
        print('Cancelled: what was written is removed.', file=sys.stderr)
        life['phase'] = 'done'
        return 3
    except Exception as e:                             # noqa: BLE001 (reported to the caller in run.json)
        if b.rec is not None:
            b.rec.cleanup()
        b.report.update(status='failed', error=f'{type(e).__name__}: {e}', notes=b.notes)
        _write_json(os.path.join(b.out_dir, 'run.json'), b.report)
        print(f'Failed: {type(e).__name__}: {e}', file=sys.stderr)
        life['phase'] = 'done'
        return 2
    life['phase'] = 'done'
    c = rep.get('counts', {})
    gt = (f', ground truth for {c["gt_frames"]} camera frames ({c.get("gt_objects", 0)} objects) and '
          f'{c.get("lidarseg", 0)} lidar sweeps') if c.get('gt_frames') or c.get('lidarseg') else ''
    print(f'Recorded {rep["scene"]["duration_s"]:g} s: {c.get("sample_data", 0)} sample_data, '
          f'{c.get("bag_messages", 0)} bag messages, {c.get("annotations", 0)} annotations{gt} into {b.out_dir} '
          f'({rep["performance"]["wall_s"]:.0f} s).', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
