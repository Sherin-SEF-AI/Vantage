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

import numpy as np

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


class Cancelled(Exception):
    pass


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


def plan(rig: dict, duration: float, cameras=True) -> dict:
    """Every channel's planned stamps over the drive (s from the sync pulse), the keyframes and each channel's
    keyframe assignment, as core.dataset plans them."""
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
        j = np.abs(c.stamps[None, :] - samples[:, None]).argmin(1)
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

    # ------------------------------------------------------------------ helpers
    def _check(self):
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

    # ------------------------------------------------------------------ the run
    def run(self) -> dict:
        t_wall = time.perf_counter()
        carla = self.carla
        a = self.a
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
        # ---- the map's parked cars (level geometry): read before any vehicle is spawned, which the same call lists
        self._level_cars = self._level_boxes(world)
        # ---- vehicles
        lib = world.get_blueprint_library()
        spawns = list(world.get_map().get_spawn_points())
        if not spawns:
            raise RuntimeError('The map has no spawn points.')
        order = rng.permutation(len(spawns))
        veh = rig['vehicle']['blueprint'] if a.vehicle in (None, '', 'auto') else a.vehicle
        bp = self._bp(lib, veh, {'role_name': 'hero'})
        ego = None
        used = 0
        for i in order:
            used += 1
            ego = world.try_spawn_actor(bp, spawns[i])
            if ego is not None:
                break
        if ego is None:
            raise RuntimeError('No free spawn point for the ego vehicle.')
        self.actors.append(ego)
        self.ego = ego
        cars = [b for b in lib.filter('vehicle.*') if (not b.has_attribute('number_of_wheels') or
                                                        b.get_attribute('number_of_wheels').as_int() == 4)]
        traffic = []
        for i in order[used:]:
            if len(traffic) >= int(a.traffic):
                break
            b = cars[int(rng.integers(len(cars)))]
            if b.has_attribute('role_name'):
                b.set_attribute('role_name', 'autopilot')
            v = world.try_spawn_actor(b, spawns[i])
            if v is not None:
                traffic.append(v)
                self.actors.append(v)
        # ground-truth ids by spawn order, not CARLA's actor ids: the same seed gives the same tokens on every run
        self.gt_id = {v.id: k + 1 for k, v in enumerate(traffic)}
        if len(traffic) < int(a.traffic):
            self._note(f'{len(traffic)} of {a.traffic} traffic vehicles found a free spawn point.')
        world.tick()
        for v in traffic:
            v.set_autopilot(True, int(a.tm_port))
        ego.set_autopilot(True, int(a.tm_port))
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
            world.tick()
        # ---- where the ego origin lies in the CARLA vehicle
        self._measure_ego()
        self._mount_check(world)
        # ---- sensors (cameras later, at their spawn steps)
        P = plan(rig, float(a.duration), cameras=not a.no_cameras)
        self.P = P
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
        run_key = token('vantage', 'carla', rig['project'], rig['variant'], self.report['town'], seed,
                        float(a.duration), int(a.traffic), json.dumps(weather, sort_keys=True))
        nus_rows = rig['nuscenes']
        chan_names = {c.channel for c in P['chans'] if c.channel}
        extra = {'sensor_rows': [r for r in nus_rows['sensor'] if r['channel'] in chan_names],
                 'cal_rows': []}
        tok_of = {r['token'] for r in extra['sensor_rows']}
        extra['cal_rows'] = [r for r in nus_rows['calibrated_sensor'] if r['sensor_token'] in tok_of]
        self.rec = Recorder(rig, self.out_dir, P, run_key, extra, bool(rig['output']['nuscenes']),
                            bool(rig['output']['rosbag']))
        self._loop(world, warm, spawn_at, cam_docs, rng, t_wall)
        return self.report

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
            pv = d['pose_vehicle']
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
        loc = np.asarray(loc_rel, float) + self.O
        a = world.spawn_actor(bp, _tf(self.carla, loc, rotation), attach_to=self.ego)
        self.actors.append(a)
        return a

    def _spawn_sensors(self, world, lib):
        self.queues = {}
        for L in self.rig['lidars']:
            bp = self._bp(lib, 'sensor.lidar.ray_cast', L['blueprint'])
            act = self._attach(world, bp, L['carla']['location'], L['carla']['rotation'])
            self._listen(act, ('lidar', L['tag']))
            self.sensors.append((act, 'lidar', L, 0))
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
        lid_state = {L['tag']: {'pts': [], 'int': [], 'ring': [], 't': [], 'counts': {}} for L in rig['lidars']}
        rad_state = {r['tag']: [] for r in rig['radars']}
        cam_pending = {c['tag']: {} for c in cam_docs}
        imu_pending = {}
        imu_bias = {x['tag']: np.random.default_rng([int(self.a.seed), 991, i]).normal(
            0, x['noise']['gyro_bias'], 3) for i, x in enumerate(rig['imus'])}
        imu_rng = {x['tag']: np.random.default_rng([int(self.a.seed), 992, i]) for i, x in enumerate(rig['imus'])}
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
        perf = {'tick_s': 0.0, 'wait_s': 0.0, 'camera_s': 0.0, 'lidar_s': 0.0}
        cam_frames = {c['tag']: {'expected': 0, 'got': 0, 'unexpected': 0} for c in cam_docs}
        phase_meas = {}
        total = warm + N
        t_rec0 = None
        for i in range(1, total + 1):
            self._check()
            for c in spawn_at.get(i, ()):
                self._spawn_camera(world, lib, c)
            t0 = time.perf_counter()
            frame = world.tick(timeout)
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
            t1 = time.perf_counter()
            items = {}
            for act, kind, doc, idx in self.sensors:
                if kind == 'camera':
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
            for c in cam_docs:
                st = cam_pending[c['tag']]
                n = int(c['capture']['every_ticks'])
                spawn_i = max(warm + int(c['capture']['phase_ticks']) - n, 1)
                due = i >= spawn_i and (i - spawn_i) % n == 0
                if due and 0 <= rel <= N:
                    cam_frames[c['tag']]['expected'] += 1
                self._collect_camera(c, st, frame if due else None, timeout)
                for f in sorted(st):
                    if len(st[f]) < len(c['views']):
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
                    cam_frames[c['tag']]['got'] += 1
                    T_e = ego_hist.get(ti, (None, T_ego))[1]
                    cam_jobs.append((pool.submit(self._camera_jpeg, c, views), ch, k, tt, T_e))
            while cam_jobs and cam_jobs[0][0].done():
                fut, ch, k, tt, T_e = cam_jobs.pop(0)
                t2 = time.perf_counter()
                rec.camera(ch, k, tt, fut.result(), T_e)
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
                                       boxes_at.get(rel))
                rad_state[r['tag']].append(det)
                k = ch.tick_index.get(rel)
                if k is not None:
                    merged = self._merge_dets(rad_state[r['tag']])
                    rad_state[r['tag']] = []
                    rec.radar(ch, k, ch.stamps[k], merged, T_ego)
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
                                  boxes_at.get(rel), rel, phase_meas, (ego_hist.get(i - 1) or (None,))[0])
                if rel % m == 0:
                    k = rel // m - 1
                    ch = chan_of[('lidar', L['tag'])]
                    st = lid_state[L['tag']]
                    if k < len(ch.stamps):
                        xyz = np.concatenate(st['pts']) if st['pts'] else np.zeros((0, 3), np.float32)
                        it = np.concatenate(st['int']) if st['int'] else np.zeros(0, np.float32)
                        rg = np.concatenate(st['ring']) if st['ring'] else np.zeros(0, np.uint16)
                        tp = np.concatenate(st['t']) if st['t'] else np.zeros(0, np.float32)
                        order = np.lexsort((rg, tp))       # firing order, as a lidar sends them: column by column
                        rec.lidar(ch, k, ch.stamps[k], xyz[order], it[order], rg[order], tp[order], T_ego)
                        if k in ch.keys:
                            counts_l[(L['tag'], k)] = dict(st['counts'])
                    lid_state[L['tag']] = {'pts': [], 'int': [], 'ring': [], 't': [], 'counts': {}}
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
                if rel in ch.tick_index:
                    m_ = items[('gnss', g['tag'])]
                    rec.gnss(ch, t, m_.latitude, m_.longitude, m_.altitude, float(g['sigma_h_m']))
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
            rec.camera(ch, k, tt, fut.result(), T_e)
        pool.shutdown(wait=True)
        for c in cam_docs:
            if cam_pending[c['tag']]:
                self._note(f'{c["tag"]}: {len(cam_pending[c["tag"]])} frame(s) missing a view at the end.')
        self._finish(ann_rows, instances, perf, cam_frames, phase_meas, t_wall, t_rec0)

    # ------------------------------------------------------------------ per-sensor processing
    def _sensor_world(self, M_actor, doc) -> np.ndarray:
        """4x4 (CARLA axes) world <- sensor."""
        loc = np.asarray(doc['carla']['location'], float) + self.O
        rot = doc['carla']['rotation']
        return M_actor @ FR.transform_matrix(loc, (rot['pitch'], rot['yaw'], rot['roll']))

    def _collect_camera(self, c, st, frame, timeout):
        """Move what the views delivered into st[frame][view]; when frame is due, wait for all its views."""
        nv = len(c['views'])
        end = time.monotonic() + timeout
        while True:
            for vi in range(nv):
                q = self.queues.get(('camera', c['tag'], vi))
                if q is None:
                    continue
                while True:
                    try:
                        img = q.get_nowait()
                    except queue.Empty:
                        break
                    st.setdefault(img.frame, {})[vi] = img
            if frame is None or len(st.get(frame, {})) >= nv:
                return
            if all(any(f > frame and vi in v for f, v in st.items()) for vi in range(nv)):
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

    def _camera_jpeg(self, c, views) -> bytes:
        import cv2
        at = c['atlas']
        atlas = np.zeros((int(at['height']), int(at['width']), 3), np.uint8)
        for vi, v in enumerate(c['views']):
            img = views[vi]
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

    def _lidar_slice(self, L, meas, st, M_actor, boxes, rel, phase_meas, M_prev=None):
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
        if boxes is not None and sel.any():
            Ws = _mirror(self._sensor_world(M_actor, L))
            Rt4 = np.eye(4)
            Rt4[:3, :3] = Rt
            Pg = (self.G @ Ws @ np.linalg.inv(Rt4) @ np.c_[Q[sel], np.ones(int(sel.sum()))].T).T[:, :3]
            for b in boxes:
                n = int(_in_box(Pg, b).sum())
                if n:
                    st['counts'][b['id']] = st['counts'].get(b['id'], 0) + n

    def _radar_step(self, r, meas_list, Ws, v_rad_world, boxes) -> dict:
        xs = []
        for zi, meas in enumerate(meas_list):
            for d in meas:
                xs.append((zi, d.depth, d.azimuth, d.altitude, d.velocity))
        if not xs:
            return {'P': np.zeros((0, 3)), 'v': np.zeros(0), 'counts': {}, 'u': np.zeros((0, 3)), 'vc': np.zeros(0)}
        A = np.array(xs, float)
        zi, dep, az, alt, vel = A[:, 0].astype(int), A[:, 1], A[:, 2], A[:, 3], A[:, 4]
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
        self.rec.imu(ch, t, acc, w, R_or, att, head, x['noise'])

    # ------------------------------------------------------------------ ground truth
    def _category(self, a) -> str:
        tid = a.type_id
        attrs = getattr(a, 'attributes', {}) or {}
        base = attrs.get('base_type', '')
        if tid.startswith('walker.'):
            return 'human.pedestrian.adult'
        if 'ambulance' in tid:
            return 'vehicle.emergency.ambulance'
        if 'police' in tid:
            return 'vehicle.emergency.police'
        if base in ('bicycle',):
            return 'vehicle.bicycle'
        if base in ('motorcycle',):
            return 'vehicle.motorcycle'
        if base in ('truck',) or 'firetruck' in tid:
            return 'vehicle.truck'
        if base in ('bus',) or 'fusorosa' in tid:
            return 'vehicle.bus.rigid'
        return 'vehicle.car'

    def _boxes(self, snap, T_ego) -> list:
        """Every other vehicle's box within the annotation range: global centre, rotation, size, velocity."""
        out = []
        rng = float(self.rig['output'].get('gt_range_m', 60.0))
        ex = T_ego[:3, 3]
        for a in self.actors:
            if a is self.ego or not a.type_id.startswith(('vehicle.', 'walker.')):
                continue
            s = snap.find(a.id)
            if s is None:
                continue
            M = _mirror(_m(s.get_transform()))
            bb = a.bounding_box
            c_loc = FR.to_vantage([bb.location.x, bb.location.y, bb.location.z])
            Mg = self.G @ M
            centre = Mg[:3, :3] @ c_loc + Mg[:3, 3]
            if math.hypot(centre[0] - ex[0], centre[1] - ex[1]) > rng:
                continue
            Rg = Mg[:3, :3]
            v = FR.to_vantage([s.get_velocity().x, s.get_velocity().y, s.get_velocity().z])
            out.append({'id': int(self.gt_id.get(a.id, 100_000 + a.id)), 'category': self._category(a),
                        'name': a.type_id, 'centre': centre,
                        'R': Rg, 'yaw': math.atan2(Rg[1, 0], Rg[0, 0]),
                        'size_lwh': [2 * bb.extent.x, 2 * bb.extent.y, 2 * bb.extent.z], 'velocity': v,
                        'moving': float(np.hypot(v[0], v[1])) > 0.3, 'static': False})
        return out

    def _level_boxes(self, world) -> list:
        """The map's parked cars (level geometry, not actors) from World.get_level_bbs(Car), in CARLA's world: read
        before the run spawns any vehicle (the call lists the vehicles' boxes too). Car-sized boxes only."""
        label = getattr(getattr(self.carla, 'CityObjectLabel', None), 'Car', None)
        if label is None or not hasattr(world, 'get_level_bbs'):
            return []
        try:
            bbs = list(world.get_level_bbs(label))
        except Exception:                              # noqa: BLE001
            self._note('Parked cars not annotated: World.get_level_bbs failed.')
            return []
        out = []
        for b in bbs:
            L, W, H = 2 * b.extent.x, 2 * b.extent.y, 2 * b.extent.z
            if 2.5 <= L <= 7.0 and 1.3 <= W <= 2.8 and 1.0 <= H <= 3.5:
                out.append(([b.location.x, b.location.y, b.location.z], (b.rotation.pitch, b.rotation.yaw,
                                                                         b.rotation.roll), [L, W, H]))
        return out

    def _static_boxes(self, world) -> list:
        """The map's parked cars as ground-truth boxes in the global frame (vehicle.parked)."""
        out = []
        for k, (loc, rot, size) in enumerate(getattr(self, '_level_cars', [])):
            Rr = FR.vantage_matrix(*rot)
            centre = FR.to_vantage(loc) + self.offset
            out.append({'id': 1_000_000 + k, 'category': 'vehicle.car', 'name': 'parked car (map)',
                        'centre': centre, 'R': Rr, 'yaw': math.atan2(Rr[1, 0], Rr[0, 0]), 'size_lwh': list(size),
                        'velocity': np.zeros(3), 'moving': False, 'static': True})
        if out:
            self._note(f'{len(out)} parked cars of the map (its level boxes) annotated as vehicle.parked.')
        return out

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
            if b['category'].startswith('vehicle.bicycle') or b['category'].startswith('vehicle.motorcycle'):
                attr = 'cycle.with_rider'
            elif b['category'].startswith('human'):
                attr = 'pedestrian.moving' if b['moving'] else 'pedestrian.standing'
            elif b['static']:
                attr = 'vehicle.parked'
            else:
                attr = 'vehicle.moving' if b['moving'] else 'vehicle.stopped'
            rows.append({'token': tok, 'sample_token': self.rec.sample_tok[si], 'instance_token': inst,
                         'visibility_token': '1' if vis < 0.4 else '2' if vis < 0.6 else '3' if vis < 0.8 else '4',
                         'attribute_tokens': [token('vantage', 'attribute', attr)],
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
        def quiet(fn, *args):
            try:
                return fn(*args)
            except Exception:                          # noqa: BLE001 (tidying up after anything)
                return None
        sync = bool(getattr(world.get_settings(), 'synchronous_mode', False)) if quiet(world.get_settings) else False
        tm_port = int(getattr(self.a, 'tm_port', 8000))
        for a in self.actors:
            if a.type_id.startswith('vehicle.'):
                quiet(a.set_autopilot, False, tm_port)
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


# =========================================================================== command line
def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog='python -m vantage.sim.carla_bridge',
                                 description='Run a Vantage rig (sim_rig.json from core.simrig) in CARLA and record it '
                                             'as a nuScenes folder and a ROS 2 bag.')
    ap.add_argument('rig', help='sim_rig.json (sim_remap.npz next to it)')
    ap.add_argument('out', help='output folder (nuscenes/, rosbag/, dataset.json, run.json)')
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
    return ap


def resolve_args(a, rig: dict):
    """Fill the arguments left out from the rig's scenario and server settings."""
    sc = rig['scenario']
    srv = rig.get('server', {})
    a.duration = float(a.duration if a.duration is not None else sc['duration_s'])
    a.traffic = int(a.traffic if a.traffic is not None else sc['traffic'])
    a.seed = int(a.seed if a.seed is not None else sc['seed'])
    a.port = int(a.port if a.port is not None else srv.get('port', 2000))
    a.tm_port = int(a.tm_port if a.tm_port is not None else srv.get('tm_port', 8100))
    if a.weather is None and sc.get('weather') and sc['weather'] not in ('project',) and \
            not str(sc['weather']).startswith('vantage:'):
        a.weather = sc['weather']
    return a


def main(argv=None) -> int:
    a = parser().parse_args(argv)
    rig, remaps = load_rig(a.rig)
    resolve_args(a, rig)
    stop = {'flag': False}

    def on_signal(*_):
        stop['flag'] = True
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
    os.makedirs(b.out_dir, exist_ok=True)
    try:
        rep = b.run()
    except Cancelled:
        if b.rec is not None:
            b.rec.cleanup()
        b.report.update(status='cancelled', notes=b.notes)
        _write_json(os.path.join(b.out_dir, 'run.json'), b.report)
        print('Cancelled: what was written is removed.', file=sys.stderr)
        return 3
    except Exception as e:                             # noqa: BLE001 (reported to the caller in run.json)
        if b.rec is not None:
            b.rec.cleanup()
        b.report.update(status='failed', error=f'{type(e).__name__}: {e}', notes=b.notes)
        _write_json(os.path.join(b.out_dir, 'run.json'), b.report)
        print(f'Failed: {type(e).__name__}: {e}', file=sys.stderr)
        return 2
    c = rep.get('counts', {})
    print(f'Recorded {rep["scene"]["duration_s"]:g} s: {c.get("sample_data", 0)} sample_data, '
          f'{c.get("bag_messages", 0)} bag messages, {c.get("annotations", 0)} annotations into {b.out_dir} '
          f'({rep["performance"]["wall_s"]:.0f} s).', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
