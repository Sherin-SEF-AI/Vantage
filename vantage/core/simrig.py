"""The rig as a simulator needs it: every camera, lidar, radar, GNSS antenna and IMU of a variant converted to CARLA's
frame with its intrinsics, the simulation's step and each sensor's schedule, written as sim_rig.json and
sim_remap.npz for the CARLA bridge (vantage.sim.carla_bridge), which needs only numpy, OpenCV and carla.

CARLA (researched for 0.9.16, the current stable release, September 2025; checked against its source at the 0.9.16
tag and a 0.9.16 server running here without rendering)
--------------------------------------------------------------------------------------------------------------------
* Release: Unreal Engine 4.26, Linux package CARLA_0.9.16.tar.gz (8.3 GB, unpacks to 19 GB; Town01–05 and Town10HD;
  AdditionalMaps_0.9.16.tar.gz adds Town06, 07, 11, 12, 13, 15). No checksum is published; the download's multipart
  S3 ETag (173c05807cd942bfb81f40fcfd59430b-125, 64 MiB parts) was recomputed and matched. Python client wheels on PyPI
  and in PythonAPI/carla/dist: CPython 3.10, 3.11 and 3.12, manylinux_2_31 x86-64 (and Windows). CARLA 0.10.0 (UE5,
  December 2024) exists but is a separate line with fewer maps; 0.9.16 is the release marked latest.
* Server: CarlaUE4.sh (CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping CarlaUE4) with -RenderOffScreen to render
  without a window (GPU still needed: UE 4.26 is Vulkan only), -carla-rpc-port (2000; streaming on the next port),
  -quality-level=Epic|Low (Low: no post-processing or shadows, 50 m draw distance), -nosound. -nullrhi runs with no
  rendering at all: lidar, radar, GNSS and IMU still work (they are physics ray casts and actor kinematics) but a
  camera stops the server (seen here: it crashed on the first capture), and -quality-level=Low with -nullrhi crashes at
  start-up. The traffic manager listens on its own port (8000 by default; this machine already has a service on 8000).
* Synchronous mode: WorldSettings.synchronous_mode with fixed_delta_seconds; the client's World.tick() advances one
  step and returns its frame; the traffic manager must be put in synchronous mode too (set_synchronous_mode) and is
  seeded by set_random_device_seed. A sensor with sensor_tick 0 runs on every step; with sensor_tick T it runs on the
  first step after it is spawned and then by Unreal's tick interval, which carries the remainder over (measured here:
  0.1 s at 0.01 s steps fires after 11, then every 10 steps; 0.0999 s fires every 10 steps from the start).
* Weather: carla.WeatherParameters (cloudiness, precipitation, precipitation_deposits, wind_intensity 0–100;
  sun_azimuth_angle, sun_altitude_angle in degrees; fog_density 0–100, fog_distance m, fog_falloff; wetness 0–100;
  scattering, Mie and Rayleigh scales; dust_storm) and the presets ClearNoon … HardRainNight, DustStorm. Fog and rain
  are rendered effects for the RGB camera only: they do not change lidar or radar returns. There is no snow.
  Measured here (round 8, a still camera, linear light): fog_density sets an extinction proportional to it, a MOR of
  FOG_MOR_M / fog_density for objects 0.2–2.5 m above the road (a height fog: denser along the road surface itself);
  precipitation renders water on the lens, not extinction: contrast −12, −21, −34 % at every range for 30, 60, 100
  and the image centre blurred (its sharpness ×0.74, ×0.38, ×0.32).
* sensor.camera.rgb: a pinhole camera only, square pixels, image_size_x/_y, fov = the horizontal field (f = W / (2
  tan(fov/2)), principal point at the image centre), BGRA 8-bit raw_data; a lens-distortion post-process (lens_k,
  lens_kcube, lens_circle_*) that is no calibrated lens model; exposure_mode manual | histogram, gamma, motion blur,
  bloom, lens flare. A fisheye beyond what one pinhole view can hold is therefore rendered as several pinhole views
  from one optical centre and remapped through Vantage's own lens model (below).
* sensor.lidar.ray_cast: channels evenly spaced from upper_fov (channel 0) to lower_fov (the last), inclusive; on each
  step it casts round(points_per_second·dt / channels) points per channel at azimuth steps of rotation_frequency ·
  horizontal_fov · dt / that count, starting where the last step stopped; a laser's azimuth is
  fmod(angle + step·i, horizontal_fov) − horizontal_fov/2, so at angle 0 it points backwards and it turns clockwise
  seen from above (Unreal's yaw grows to the right). Points (x, y, z, intensity float32) come in the sensor's own
  CARLA axes, channel by channel (LidarMeasurement.get_point_count(channel)); intensity is only the atmospheric
  attenuation exp(−a·r), not a reflectivity; missed rays give no point; dropoff_* drop points at random (set to 0
  here) and noise_stddev adds range noise along the ray. Arbitrary beam elevations are not possible: the channels are
  evenly spaced.
* sensor.other.radar: per step points_per_second·dt rays cast at random inside an elliptical cone (horizontal_fov ×
  vertical_fov, denser near the axis), range; RadarDetection: depth (m), azimuth and altitude (radians, azimuth
  positive to the right), velocity (m/s, the hit actor's velocity relative to the radar's own along the ray: positive
  moving away); no RCS. The first step after spawning has no previous position: its velocities are garbage.
* sensor.other.gnss: latitude, longitude (degrees), altitude (m) from the map's geo-reference; noise_*_stddev and bias
  in degrees (lat, lon) and metres (alt). sensor.other.imu: accelerometer (m/s², gravity included, from the sensor's
  own positions over three steps, so the first two steps are garbage), gyroscope (rad/s), compass; 0.9.16's gyroscope
  rotates the vehicle's world-frame rate by the sensor's mounting rotation instead of into its frame
  (InertialMeasurementUnit.cpp, ComputeGyroscope), so the bridge takes the rate from the vehicle's poses instead.
* Traffic manager: Client.get_trafficmanager(port), Vehicle.set_autopilot(True, port), synchronous and seeded as
  above; vehicles only here (no pedestrians). A vehicle still on the autopilot when it is destroyed makes the
  manager's thread touch a dead actor and the client aborts (seen here): vehicles leave the autopilot first.
* Seen on the 0.9.16 server (no rendering, Town10HD): the first step after a sensor spawns gives a radar velocity
  and an IMU acceleration of thousands (no previous position); LidarMeasurement.horizontal_angle stays 0, so the
  bridge reads azimuths from the points; World.get_level_bbs(Car) lists the map's parked cars and also every spawned
  vehicle; the same seed on the same server gives bit-identical lidar, radar and poses; without cameras the sample's
  data rig (an OS1-128, five radars) records at 68 steps/s of 80 Hz (0.86× real time) on this machine.

What this module makes
----------------------
* Frames: vantage.sim.frames (x, −y, z; carla yaw = −yaw, pitch = −pitch, roll = roll, exact for any rotation by
  conjugation, tested against matrices from the carla 0.9.16 module). Sensor positions are relative to the ego's
  rear-axle centre on the ground; the bridge adds where that lies in the CARLA vehicle's own frame (its rear wheels
  and bounding box, read at run time).
* Cameras: each module's image at image_scale (the sample dataset's size and intrinsics, core.dataset) from one or
  more pinhole views: one view when every ray of the image is within 60° of the axis, else the cube faces the image
  needs (front, left, right, up, down, back), each SEAM_BLEND_DEG / 2 (1.5°) past its seams, with every ray at least
  1.5 px inside its face.
  The remap tables (sim_remap.npz) hold, per output pixel, where it reads in the views stacked into one atlas image:
  Vantage's exact lens (project.lens_for, with the camera's unit calibration) unprojects each pixel centre and the
  pinhole geometry of its view projects it; pixels outside the image circle stay black. The views' resolution matches
  the lens's own at the image centre times face_oversample. All views of a camera render at the same instant with the
  same manual exposure, no motion blur, bloom or lens flare (each would differ per view and show as seams), and CARLA's
  lens-distortion effect off (lens_k = lens_kcube = lens_circle_multiplier = 0). Measured on the 0.9.16 server (RTX
  5080, Town10HD), and handled by the bridge:
  - Unreal's default vignette (intensity 0.4, applied in linear light before the tone curve; no blueprint attribute)
    darkens every view towards its own edges: vantage.sim.tonemap takes it out of each sample in linear light through
    the camera response, measured from exposure brackets (Debevec–Malik).
  - The views of a moving camera render slightly differently (temporal anti-aliasing reprojects each view's history
    along its own image motion; static views agree): on the sample, a step of about 4 levels where two faces meet.
    blend_tables feathers the faces across a SEAM_BLEND_DEG band at each seam; the step became under 1 level.
  - Geometry: lidar points projected with the pinhole convention here (f = W / 2 tan(fov/2), centre at W/2 − 0.5 in
    OpenCV coordinates) fall on CARLA's depth image within 0.33 mm (median), and the colour image's edges lie on its
    semantic segmentation's within 0.1 px, static and moving.
* Lidar: CARLA's even channels are laid over the model's beam elevations: the coarsest even grid from the lowest to
  the highest beam that has every beam on it (within 0.0001°), up to max_lidar_channels; the channels between beams
  are cast and dropped (said, with the cost). When no grid within that limit fits, each beam takes its nearest channel
  and the largest elevation error is stated. The azimuth step is the model's exactly: points per channel per step =
  columns / steps per rotation, the step rate chosen so that is a whole number, or the step made finer by an integer
  factor and the points thinned back to the model's grid. The sensor is mounted turned about its z axis so CARLA's
  start angle falls on the model's phase at each sync pulse (rounded to the azimuth grid); points come back rotated
  into the sensor's planned frame (REP 103).
* Radar: one CARLA radar per zone of the model (its range on a car, its azimuth and elevation field), all at the
  radar's pose; a scan merges every step of the radar's period. CARLA's field is an elliptical cone, Vantage's zone a
  rectangle in azimuth × elevation: the cone is made just large enough to hold the rectangle (radar_cone) and
  detections outside the rectangle are dropped; the rays per step are raised by the share that falls outside, so a
  scan has about radar_points rays inside the zone.
* GNSS and IMU at their poses. A radar, IMU or GNSS rate that is no whole number of steps: each sample at the step
  nearest its time (the rate exact on average, the jitter within half a step); one faster than the step: one sample
  per step. Both are stated.
* Schedule: the simulation step is a multiple of the lidar rates (else of the cameras'), chosen in [min_tick_hz,
  max_tick_hz] (or tick_hz) for the fewest extra lidar rays, then the most sensors whose rate is a whole number of
  steps, then the fewest steps. Cameras run at the lidar rate with the sync plan's trigger times (core.timing), or at
  the module's rate; each frame is rendered at its planned exposure centre rounded to a step.
* Vehicle: the CARLA vehicle nearest the project's (length, width, height, wheelbase), from a table measured on the
  0.9.16 server (CARLA_VEHICLES); its body is not the project's, so the cameras see a different car.

* Ground truth, faults, a scenario, an actors scene (round 8): SimSettings.ground_truth {semantic, instance, range,
  lidar_labels} (the cameras rendered for it, and their cost in Mpx, in the rig's ground_truth block; each camera's
  'lens' block lets the bridge project 3D boxes through Vantage's exact lens model), faults (a list, checked against
  the rig's sensors), scenario {name, params}, actors (ACTORS.json's list), spawn_point, walkers; the contracts and
  their checks are vantage.sim.carla_bridge's (FAULT_KINDS, SCENARIOS, ACTOR_KINDS, clean_*).

Limits: CARLA renders its own world and its own vehicle; lidar and radar ignore the weather; lidar intensity is not a
reflectivity; the radar has no RCS and no noise model of Vantage's; a camera frame is one instant (no rolling shutter,
no motion blur); trigger offsets and rates round to the simulation step.
"""
from __future__ import annotations

import datetime as _dt
import io
import json
import math
import os
from dataclasses import asdict, dataclass, field

import numpy as np

from .lens import resolve
from .mathutil import BODY_TO_OPTICAL, rot_zyx
from ..sim import carla_bridge as CB
from ..sim import frames as FR

SCHEMA = 'vantage.sim_rig/1'
RUN_SCHEMA = 'vantage.sim_run/1'
CARLA_VERSION = '0.9.16'
TOWNS = ('Town10HD', 'Town01', 'Town02', 'Town03', 'Town04', 'Town05')     # in the 0.9.16 base package
CARLA_WEATHERS = ('ClearNoon', 'CloudyNoon', 'WetNoon', 'WetCloudyNoon', 'SoftRainNoon', 'MidRainyNoon',
                  'HardRainNoon', 'ClearSunset', 'CloudySunset', 'WetSunset', 'WetCloudySunset', 'SoftRainSunset',
                  'MidRainSunset', 'HardRainSunset', 'ClearNight', 'CloudyNight', 'WetNight', 'WetCloudyNight',
                  'SoftRainNight', 'MidRainyNight', 'HardRainNight', 'DustStorm')
WEATHER_KEYS = ('cloudiness', 'precipitation', 'precipitation_deposits', 'wind_intensity', 'sun_azimuth_angle',
                'sun_altitude_angle', 'fog_density', 'fog_distance', 'fog_falloff', 'wetness')
CAMERA_RATES = ('lidar', 'module')
CHANNEL_NAMES = ('tags', 'nuscenes')
EXPOSURES = ('manual', 'histogram')
QUALITIES = ('Epic', 'Low')
# blueprint id -> (base type, length, width (mirrors included), height, wheelbase, rear axle x in the actor frame), m:
# measured on a CARLA 0.9.16 server (no rendering) on 2026-10-27: each blueprint spawned on Town10HD, its bounding box
# and wheel positions read (the actor's origin is on the ground under the box centre)
CARLA_VEHICLES = {
    'vehicle.audi.a2': ('car', 3.705, 1.789, 1.549, 2.506, -1.256),
    'vehicle.audi.etron': ('car', 4.856, 2.033, 1.649, 2.900, -1.386),
    'vehicle.audi.tt': ('car', 4.181, 1.994, 1.385, 2.642, -1.400),
    'vehicle.bmw.grandtourer': ('car', 4.611, 2.242, 1.667, 2.935, -1.443),
    'vehicle.carlamotors.carlacola': ('truck', 5.204, 2.627, 2.467, 2.665, -1.206),
    'vehicle.carlamotors.european_hgv': ('truck', 7.936, 2.891, 3.462, 4.995, -2.678),
    'vehicle.carlamotors.firetruck': ('truck', 8.468, 2.891, 3.827, 4.995, -2.678),
    'vehicle.chevrolet.impala': ('car', 5.357, 2.033, 1.411, 2.982, -1.329),
    'vehicle.citroen.c3': ('car', 3.988, 1.851, 1.617, 2.684, -1.410),
    'vehicle.dodge.charger_2020': ('car', 5.008, 1.882, 1.535, 3.044, -1.433),
    'vehicle.dodge.charger_police': ('car', 4.974, 2.038, 1.542, 2.992, -1.501),
    'vehicle.dodge.charger_police_2020': ('car', 5.238, 1.930, 1.638, 3.044, -1.433),
    'vehicle.ford.ambulance': ('van', 6.366, 2.351, 2.431, 3.770, -1.661),
    'vehicle.ford.crown': ('car', 5.366, 1.801, 1.575, 2.916, -1.143),
    'vehicle.ford.mustang': ('car', 4.718, 1.895, 1.301, 2.886, -1.324),
    'vehicle.jeep.wrangler_rubicon': ('car', 3.866, 1.905, 1.878, 2.308, -1.017),
    'vehicle.lincoln.mkz_2017': ('car', 4.902, 2.128, 1.511, 2.871, -1.394),
    'vehicle.lincoln.mkz_2020': ('car', 4.892, 1.837, 1.490, 2.860, -1.389),
    'vehicle.mercedes.coupe': ('car', 5.027, 2.152, 1.651, 3.289, -1.519),
    'vehicle.mercedes.coupe_2020': ('car', 4.674, 1.812, 1.442, 2.833, -1.287),
    'vehicle.mercedes.sprinter': ('van', 5.915, 1.988, 2.561, 3.662, -1.729),
    'vehicle.micro.microlino': ('car', 2.207, 1.481, 1.376, 1.525, -0.841),
    'vehicle.mini.cooper_s': ('car', 3.806, 1.970, 1.475, 2.511, -1.367),
    'vehicle.mini.cooper_s_2021': ('car', 4.553, 2.097, 1.767, 3.010, -1.640),
    'vehicle.mitsubishi.fusorosa': ('bus', 10.273, 3.944, 4.253, 5.630, -2.615),
    'vehicle.nissan.micra': ('car', 3.633, 1.845, 1.501, 2.490, -1.317),
    'vehicle.nissan.patrol': ('car', 4.605, 1.932, 1.855, 2.839, -1.380),
    'vehicle.nissan.patrol_2021': ('car', 5.566, 2.150, 2.045, 3.279, -1.500),
    'vehicle.seat.leon': ('car', 4.193, 1.816, 1.474, 2.736, -1.420),
    'vehicle.tesla.cybertruck': ('truck', 6.274, 2.390, 2.098, 4.075, -1.907),
    'vehicle.tesla.model3': ('car', 4.792, 2.163, 1.488, 3.005, -1.386),
    'vehicle.toyota.prius': ('car', 4.514, 2.007, 1.525, 2.819, -1.399),
    'vehicle.volkswagen.t2': ('van', 4.480, 2.069, 2.038, 2.444, -1.277),
    'vehicle.volkswagen.t2_2021': ('van', 4.442, 1.775, 1.987, 2.544, -1.299),
}
VEHICLE_NOTE = ('Bounding boxes and wheel positions measured on a CARLA 0.9.16 server on 2026-10-27; a box\'s width '
                'includes the mirrors.')
# cube faces: columns are the face's optical x, y, z in the camera's optical frame (z = where the face looks)
FACES = {
    'front': np.eye(3),
    'right': np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]),
    'left': np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
    'down': np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]),
    'up': np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]),
    'back': np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]),
}
SINGLE_VIEW_DEG = 60.0          # a camera whose rays all lie within this of its axis renders as one pinhole view
FACE_MARGIN_PX = 1.5            # every ray lands at least this far inside its cube face (bilinear reads stay inside)
SEAM_BLEND_DEG = 3.0            # cube faces overlap across each seam by this much and are feathered there
VIEW_MAX_PX = 4096              # largest view CARLA is asked to render (per side)
GT_RANGE_M = 60.0               # actors annotated within this horizontal distance of the ego (as core.drivescene)
EARTH_R = 6378137.0
LIMITS = {'duration_s': (1.0, 600.0), 'traffic': (0, 300), 'seed': (0, 999999), 'image_scale': (0.05, 1.0),
          'face_oversample': (0.5, 2.0), 'jpeg_quality': (30, 100), 'keyframe_hz': (0.2, 20.0),
          'tick_hz': (0.0, 400.0), 'min_tick_hz': (1.0, 400.0), 'max_tick_hz': (1.0, 400.0),
          'lidar_noise_cm': (0.0, 20.0), 'max_lidar_channels': (2, 1024), 'radar_points': (50, 20000),
          'exposure_ev': (-5.0, 5.0), 'warmup_s': (0.0, 30.0), 'memory_gb': (4.0, 64.0), 'port': (1024, 65000),
          'tm_port': (1024, 65000), 'timeout_s': (5.0, 600.0), 'spawn_point': (-1, 100000), 'walkers': (0, 300)}
INTS = ('traffic', 'seed', 'jpeg_quality', 'max_lidar_channels', 'radar_points', 'port', 'tm_port', 'spawn_point',
        'walkers')
SCENARIOS = tuple(CB.SCENARIOS)
SERVER_MPX_WARN = 150.0          # rendered Mpx per frame above which the plan warns (measured: 77 ran, 306 crashed)
GT_NOTE = ('Ground truth beside every pinhole view: CARLA\'s instance camera (its red channel is the semantic tag, its '
           'green and blue the actor id) and depth camera, same size, field, pose and step; labels read the nearest '
           'view pixel through the remap (never interpolated), the range runs along each output pixel\'s own ray. A '
           'semantic lidar beside every lidar (same channels, azimuths and steps) labels each recorded point.')
BOOLS = ('nuscenes', 'rosbag', 'drop_self_hits', 'undistort')
ASSUMPTIONS = (
    'CARLA renders its own world, traffic and ego vehicle (the nearest CARLA vehicle by size, not the project\'s '
    'body). Cameras: pinhole views from one optical centre remapped through Vantage\'s lens model; each frame is one '
    'instant (no rolling shutter or motion blur) with a manual exposure. Lidar: CARLA\'s even channels resampled to '
    'the beam elevations, the azimuth grid exact; one return per beam, intensity = CARLA\'s atmospheric attenuation '
    '(not reflectivity); no weather effect. Radar: CARLA\'s random rays in an elliptical cone per zone, no RCS. Rates '
    'and trigger times round to the simulation step. Weather: a Vantage condition\'s fog becomes CARLA\'s fog by a '
    'calibration measured on CARLA 0.9.16 (a visibility of about 7000 m / fog_density for road users, ±15 %; denser '
    'along the road surface), its light and rain by documented heuristics (CARLA publishes no calibration of them); '
    'CARLA\'s rain is water on the lens (contrast lost at every range, the image centre blurred), not extinction with '
    'range; fog and rain show in the camera images only; CARLA has no snow.')


# =========================================================================== settings
@dataclass
class SimSettings:
    town: str = 'Town10HD'
    weather: str = 'project'           # 'project' (the selected weather condition), 'vantage:<id>' or a CARLA preset
    duration_s: float = 20.0
    traffic: int = 30                  # other vehicles on autopilot
    walkers: int = 0                   # pedestrians on CARLA's walker AI (a free drive, a scenario's background)
    seed: int = 1
    vehicle: str = 'auto'              # 'auto' (nearest by size) or a CARLA blueprint id
    camera_rate: str = 'lidar'         # lidar: one frame per lidar rotation at the sync plan's trigger; module: its fps
    image_scale: float = 0.5
    face_oversample: float = 1.0       # view resolution against the lens's own at the image centre
    undistort: bool = False            # rectified pinhole images (camera_intrinsic exact) instead of the real lens's
    jpeg_quality: int = 90
    exposure: str = 'manual'           # manual: every view of a camera alike; histogram: CARLA's auto per view (seams)
    exposure_ev: float = 0.0           # CARLA's exposure_compensation
    keyframe_hz: float = 2.0
    channel_names: str = 'tags'
    nuscenes: bool = True
    rosbag: bool = True
    lidar_noise_cm: float = 1.5
    drop_self_hits: bool = True
    max_lidar_channels: int = 256
    radar_points: int = 1500           # rays per scan per zone
    tick_hz: float = 0.0               # simulation steps per second; 0 = chosen (see the module doc)
    min_tick_hz: float = 40.0
    max_tick_hz: float = 100.0
    warmup_s: float = 2.0              # traffic settles before the sensors are attached
    quality: str = 'Epic'              # server quality level
    memory_gb: float = 12.0            # the server's memory cap (its own systemd scope)
    port: int = 2000
    tm_port: int = 8100
    timeout_s: float = 60.0
    # round 8: ground truth, faults, scenarios, an actors scene (contracts in vantage.sim.carla_bridge)
    ground_truth: dict = field(default_factory=lambda: CB.clean_ground_truth({}))
    faults: list = field(default_factory=list)          # [{'tag', 'kind', ...}] (CB.FAULT_KINDS)
    scenario: dict = field(default_factory=lambda: {'name': 'free_drive', 'params': {}})
    actors: list = field(default_factory=list)          # ACTORS.json's list: an actors scene around the ego, still
    spawn_point: int = -1                               # the ego's spawn point (−1: a straight road, by the seed)

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d) -> 'SimSettings':
        """Tolerant of junk: unknown keys ignored, numbers clamped to LIMITS, words checked."""
        s = SimSettings()
        d = d if isinstance(d, dict) else {}
        if isinstance(d.get('town'), str) and d['town'].strip():
            s.town = d['town'].strip()[:64]
        w = d.get('weather')
        if isinstance(w, str) and (w == 'project' or w in CARLA_WEATHERS or (w.startswith('vantage:') and len(w) < 64)):
            s.weather = w
        v = d.get('vehicle')
        if isinstance(v, str) and (v == 'auto' or (v.startswith('vehicle.') and len(v) < 80)):
            s.vehicle = v
        for k, opts in (('camera_rate', CAMERA_RATES), ('channel_names', CHANNEL_NAMES), ('exposure', EXPOSURES),
                        ('quality', QUALITIES)):
            if d.get(k) in opts:
                setattr(s, k, d[k])
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
                x = float(d[k])
            except (TypeError, ValueError):
                continue
            if math.isfinite(x):
                x = min(max(x, lo), hi)
                setattr(s, k, int(round(x)) if k in INTS else x)
        if 0.0 < s.tick_hz < 1.0:
            s.tick_hz = 1.0
        s.max_tick_hz = max(s.max_tick_hz, s.min_tick_hz)
        if 'ground_truth' in d:
            s.ground_truth = CB.clean_ground_truth(d['ground_truth'])
        s.faults = CB.clean_faults(d.get('faults'))[0]
        sc = d.get('scenario')
        if isinstance(sc, str):
            sc = {'name': sc}
        if isinstance(sc, dict):
            s.scenario = CB.clean_scenario(sc.get('name'), sc.get('params'))[0]
            s.scenario['params'] = {k: v for k, v in s.scenario['params'].items()
                                    if k in (sc.get('params') or {})}       # only what was set (defaults implied)
        s.actors = CB.clean_actors(d.get('actors'))[0]
        return s


def settings_of(project) -> SimSettings:
    return SimSettings.from_dict(getattr(project.analysis, 'sim', None))


# =========================================================================== weather
def _interp(x, xs, ys):
    return float(np.interp(float(x), xs, ys))


# CARLA 0.9.16's fog against visibility, measured (round 8: Town10HD, a still pinhole camera 1.6 m up with the
# bridge's camera attributes, no traffic, every weather field but fog_density fixed, fog_distance 0, fog_falloff 0.1,
# two views down long straight lanes; the frames in linear light through vantage.sim.tonemap's response). The
# contrast of texture edges 0.2–2.5 m above the road (where road users are) against the fog-free frame falls as
# exp(−b·r) with b ∝ fog_density, 3.4e-4 to 5.2e-4 per metre per unit over densities 10–100 (median 4.3e-4): so the
# WMO MOR (5 % contrast, Koschmieder: MOR = 2.996 / b) is FOG_MOR_M / fog_density, FOG_MOR_M = 7000 m (±15 %: the
# views and the densities differ by that much). The fog is a height fog: rays that end on the road surface, and low
# objects such as hedges, pass through denser fog (from density 40 up their contrast is 0.15–0.75 of the fog-free
# one already at 5–20 m).
FOG_MOR_M = 7000.0
FOG_DENSITY_MAX = 100.0              # CARLA's largest fog_density: a MOR below FOG_MOR_M / 100 = 70 m is not reached


def fog_density_for(visibility_m: float) -> float:
    """CARLA fog_density (fog_distance 0, fog_falloff 0.1) whose fog has the given MOR for road users (FOG_MOR_M)."""
    if not visibility_m or visibility_m <= 0:
        return 0.0
    return float(min(FOG_DENSITY_MAX, FOG_MOR_M / float(visibility_m)))


def fog_mor_of(fog_density: float) -> float | None:
    """The MOR (m) CARLA's fog of this fog_density gives road users (the inverse of fog_density_for); None for 0."""
    return FOG_MOR_M / float(fog_density) if fog_density and fog_density > 0 else None


def carla_weather(project, settings: SimSettings) -> dict:
    """{'source', 'label', 'preset' (or ''), 'params' (WeatherParameters fields), 'night', 'beam', 'notes'} for the
    settings' weather: a CARLA preset as it is, or a Vantage condition (core.weather) mapped by these heuristics
    (CARLA publishes no calibration of its fog or rain against visibility or rain rate):
    * light: day → sun 45° up (CARLA's noon presets), night → −90° (its night presets; CARLA's street lights stay on);
    * cloud: log-linear in the ambient illuminance, 30 klx → 5 (ClearNoon), 5 klx → 60 (CloudyNoon), 1 klx → 100;
    * rain: precipitation and wind piecewise linear in the rate, 2.5 mm/h (upper end of light rain, AMS) → 30
      (SoftRain), 7.6 mm/h (upper end of moderate) → 60 (MidRain), 25 mm/h → 100 (HardRain); puddles 50 → 90 as the
      presets; wetness as CARLA's night presets (0 by day, as its noon presets);
    * fog: fog_density = FOG_MOR_M / visibility, from the camera (fog_distance 0), falloff 0.1: measured on CARLA
      0.9.16 (FOG_MOR_M's note), so the fog's MOR for road users is the condition's (150 m → 46.7; it used to be a
      log-linear guess, 10 km → 3, 50 m → 100, which put 79.9 on a 150 m fog: a MOR of about 90 m, 60 m along the
      road surface, and the weather sweep's car recall fell from 26 to 3 %); below 70 m CARLA's densest fog (100)
      is used and said;
    * snow: none in CARLA: rendered as cloud and wet roads, and said."""
    w = settings.weather
    if w in CARLA_WEATHERS:
        return {'source': 'carla', 'label': w, 'preset': w, 'params': {}, 'night': w.endswith('Night'),
                'beam': 'low', 'notes': [f'CARLA weather preset {w}.']}
    from . import weather as WX
    cid = w.split(':', 1)[1] if w.startswith('vantage:') else None
    st = WX.settings_of(project)
    try:
        c = WX.resolve_condition(project, cid if cid in WX.IDS else None, st)
    except Exception:
        c = WX.PRESET['clear']
    notes = []
    night = c.light == WX.NIGHT
    lux = max(float(c.ambient_lux), 1.0)
    cloud = _interp(math.log10(lux), [3.0, math.log10(5000.0), math.log10(30000.0)], [100.0, 60.0, 5.0]) \
        if not night else 5.0
    rain = float(c.rain_mm_h)
    prec = _interp(rain, [0.0, 2.5, 7.6, 25.0], [0.0, 30.0, 60.0, 100.0]) if rain > 0 else 0.0
    wind = max(10.0, _interp(rain, [0.0, 2.5, 7.6, 25.0], [10.0, 30.0, 60.0, 100.0]))
    deposits = _interp(rain, [0.0, 2.5, 7.6, 25.0], [0.0, 50.0, 60.0, 90.0]) if rain > 0 else 0.0
    if rain > 0:
        cloud = max(cloud, _interp(rain, [0.0, 2.5, 7.6, 25.0], [5.0, 20.0, 60.0, 100.0]))
    wet = (_interp(rain, [0.0, 2.5, 7.6, 25.0], [0.0, 60.0, 80.0, 100.0]) if rain > 0 else 0.0) if night else 0.0
    vis = float(c.visibility_m or 0.0)
    fog = 3.0 if not night else 60.0
    if vis > 0:
        fog = fog_density_for(vis)
        got = fog_mor_of(fog)
        notes.append(f'Fog {vis:g} m → fog_density {fog:.1f} (measured on CARLA 0.9.16: a MOR of {FOG_MOR_M:g} m / '
                     'fog_density for road users, ±15 %; denser along the road surface; camera images only)'
                     + (f'; CARLA\'s densest fog gives {got:.0f} m, not {vis:g} m' if got and got > vis * 1.05 else '')
                     + '.')
    if float(c.snow_mm_h) > 0:
        cloud = max(cloud, 80.0)
        deposits = max(deposits, 40.0)
        notes.append('CARLA has no snow: rendered as heavy cloud and wet roads.')
    if rain > 0:
        notes.append(f'Rain {rain:g} mm/h → precipitation {prec:.0f} (anchored on CARLA\'s soft / mid / hard rain '
                     'presets by the AMS rain classes; camera images only, the lidar and radar are not affected).')
    if night:
        notes.append(f'Night: sun 90° below the horizon, the ego\'s {c.beam} beam on; CARLA\'s street lights stay on '
                     f'(the condition\'s {c.ambient_lux:g} lx is not reproduced).')
    params = {'cloudiness': round(cloud, 1), 'precipitation': round(prec, 1),
              'precipitation_deposits': round(deposits, 1), 'wind_intensity': round(wind, 1),
              'sun_azimuth_angle': -1.0, 'sun_altitude_angle': 45.0 if not night else -90.0,
              'fog_density': round(fog, 1), 'fog_distance': 0.0 if vis > 0 else (0.75 if not night else 75.0),
              'fog_falloff': 0.1 if not night or vis > 0 else 1.0, 'wetness': round(wet, 1)}
    return {'source': 'vantage', 'label': c.label() if hasattr(c, 'label') else c.name, 'condition': c.id,
            'preset': '', 'params': params, 'night': night, 'beam': c.beam, 'notes': notes}


# =========================================================================== vehicle
def match_vehicle(project, choice: str = 'auto') -> dict:
    """The CARLA vehicle for the ego: the one asked for, or the 4-wheeled one nearest the project's length, width,
    height and wheelbase (the mean of the relative differences of the figures the project has; a CARLA box's width
    includes the mirrors, so the body's width is compared with 92 % of it)."""
    meas = {}
    try:
        meas = project.vehicle.measured() if project.vehicle.mesh is not None else {}
    except Exception:
        meas = {}
    want = {k: float(meas[k]) for k in ('Length', 'Width', 'Height', 'Wheelbase') if k in meas and meas[k] > 0}

    def score(row):
        _, L, W, H, wb, _ = row
        got = {'Length': L, 'Width': W * 0.92, 'Height': H, 'Wheelbase': wb}
        if not want:
            return 0.0
        return float(np.mean([abs(got[k] - v) / v for k, v in want.items()]))
    ranked = sorted(CARLA_VEHICLES.items(), key=lambda kv: (score(kv[1]), kv[0]))
    if choice != 'auto':
        bp = choice
        how = 'chosen'
    elif want:
        bp = ranked[0][0]
        how = 'nearest by size'
    else:
        bp = 'vehicle.citroen.c3'
        how = 'the project has no body: a typical small car'
    row = CARLA_VEHICLES.get(bp)
    out = {'blueprint': bp, 'how': how, 'project': {k.lower(): round(v, 3) for k, v in want.items()},
           'note': VEHICLE_NOTE, 'candidates': [{'blueprint': k, 'score': round(score(r), 4)} for k, r in ranked[:4]]}
    if row is not None:
        out['carla'] = {'base_type': row[0], 'length': row[1], 'width': row[2], 'height': row[3], 'wheelbase': row[4],
                        'rear_axle_x': row[5]}
        out['size_error'] = round(score(row), 4) if want else None
    else:
        out['carla'] = None
        out['size_error'] = None
    return out


# =========================================================================== cameras: views and remap
@dataclass
class View:
    name: str
    F: np.ndarray            # columns: the view's optical x, y, z in the camera's optical frame
    width: int
    height: int
    f: float                 # px

    @property
    def fov_deg(self) -> float:
        """CARLA's fov attribute: the horizontal field over the full image width."""
        return math.degrees(2.0 * math.atan(self.width / (2.0 * self.f)))

    def project(self, d_view):
        """View-frame directions (K, 3) -> OpenCV pixel coordinates (centres on integers)."""
        z = d_view[..., 2]
        return (self.width / 2.0 - 0.5 + self.f * d_view[..., 0] / z,
                self.height / 2.0 - 0.5 + self.f * d_view[..., 1] / z)


def output_size(intr, scale: float) -> tuple[int, int]:
    """The image size the sample dataset writes (core.dataset._camera_setup)."""
    return max(16, int(round(intr.width * scale))), max(16, int(round(intr.height * scale)))


def output_rays(intr, w: int, h: int, K_pinhole=None):
    """Optical-frame unit rays (h, w, 3) of the output image's pixel centres and whether each sees anything: through
    the lens (Vantage's coordinates, pixel i covering [i, i+1) of the sensor scaled to w x h) or, with K_pinhole, a
    rectified pinhole image."""
    sx, sy = intr.width / w, intr.height / h
    uu, vv = np.meshgrid(np.arange(w, dtype=float), np.arange(h, dtype=float))
    if K_pinhole is not None:
        K = np.asarray(K_pinhole, float)
        x = (uu - K[0, 2]) / K[0, 0]
        y = (vv - K[1, 2]) / K[1, 1]
        d = np.stack([x, y, np.ones_like(x)], -1)
        d /= np.linalg.norm(d, axis=-1, keepdims=True)
        theta = np.arccos(np.clip(d[..., 2], -1, 1))
        return d, theta <= min(intr.max_theta, math.radians(89.0))
    d, theta = intr.unproject((uu + 0.5) * sx, (vv + 0.5) * sy)
    d = d / np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-12)
    return d, (theta <= intr.max_theta) & np.isfinite(theta)


def plan_views(d, ok, f_out: float, oversample: float = 1.0) -> list:
    """The pinhole views a camera's rays need (see the module doc): one view within SINGLE_VIEW_DEG, else cube faces."""
    D = d[ok]
    if not len(D):
        return []
    f_v = max(f_out * oversample, 8.0)
    ang = np.degrees(np.arccos(np.clip(D[:, 2], -1, 1)))
    if float(ang.max()) <= SINGLE_VIEW_DEG:
        ax = float(np.max(np.abs(D[:, 0] / D[:, 2])))
        ay = float(np.max(np.abs(D[:, 1] / D[:, 2])))
        W = 2 * int(math.ceil(f_v * ax + FACE_MARGIN_PX + 1))
        H = 2 * int(math.ceil(f_v * ay + FACE_MARGIN_PX + 1))
        if max(W, H) > VIEW_MAX_PX:
            k = VIEW_MAX_PX / max(W, H)
            W, H, f_v = 2 * int(W * k / 2), 2 * int(H * k / 2), f_v * k
        return [View('front', FACES['front'], W, H, f_v)]
    t = math.tan(math.radians(45.0 + SEAM_BLEND_DEG / 2.0))  # a face reaches half the band past each of its seams
    N = 2 * int(math.ceil(f_v * t + FACE_MARGIN_PX))       # face centre density f = (N/2 − margin) / t ≥ f_v
    N = min(N, VIEW_MAX_PX)
    f = (N / 2.0 - FACE_MARGIN_PX) / t
    best = np.argmax(np.stack([D @ FACES[k][:, 2] for k in FACES], 1), 1)
    used = sorted(set(best.tolist()))
    names = list(FACES)
    return [View(names[i], FACES[names[i]], N, N, f) for i in used]


def remap_tables(d, ok, views: list) -> tuple[np.ndarray, np.ndarray, list]:
    """(map_x, map_y float32 (h, w), atlas offsets): where each output pixel reads in the views stacked vertically
    (view k's rows start at offsets[k]); −1 where the pixel sees nothing. cv2.remap(atlas, map_x, map_y,
    INTER_LINEAR, BORDER_CONSTANT) gives the image."""
    h, w = ok.shape
    mx = np.full((h, w), -1.0, np.float32)
    my = np.full((h, w), -1.0, np.float32)
    offsets, y0 = [], 0
    for v in views:
        offsets.append(y0)
        y0 += v.height
    if not views:
        return mx, my, offsets
    D = d[ok]
    if len(views) == 1:
        k = np.zeros(len(D), int)
    else:
        k = np.argmax(np.stack([D @ v.F[:, 2] for v in views], 1), 1)
    u = np.zeros(len(D))
    vv = np.zeros(len(D))
    for i, v in enumerate(views):
        sel = k == i
        dv = D[sel] @ v.F
        uu, vy = v.project(dv)
        u[sel] = uu
        vv[sel] = vy + offsets[i]
    mx[ok] = u.astype(np.float32)
    my[ok] = vv.astype(np.float32)
    return mx, my, offsets


def blend_tables(d, ok, views: list, offsets: list, band_deg: float = SEAM_BLEND_DEG):
    """Feathering across the cube faces' seams, or None for a single view: (idx (M,) int32, the flat indices of the
    output pixels within band_deg / 2 of a seam; x, y (M, 2) float32, where they read their second and third face in
    the atlas (−1: none); w (M, 3) float32, the weights of the face remap_tables reads and of those two, summing to 1).
    A face's weight is a smoothstep of the ray's angle inside it from its seams with the other rendered faces: ½ on a
    seam, 1 half the band inside, 0 half the band outside. The views of a camera differ slightly where CARLA renders
    them in motion (its temporal anti-aliasing reprojects each view's history along that view's own image motion):
    feathered, a difference of a few levels fades over the band instead of stepping at a line."""
    if len(views) < 2:
        return None
    D = d[ok]
    n = np.stack([v.F[:, 2] for v in views])
    s = np.full((len(D), len(views)), np.inf)
    for i in range(len(views)):
        for j in range(len(views)):
            if i != j and abs(float(n[i] @ n[j])) < 0.5:               # neighbours (not the same or opposite face)
                m = (n[i] - n[j]) / np.linalg.norm(n[i] - n[j])
                s[:, i] = np.minimum(s[:, i], np.degrees(np.arcsin(np.clip(D @ m, -1.0, 1.0))))
    t = np.clip((s + band_deg / 2.0) / band_deg, 0.0, 1.0)
    w = t * t * (3.0 - 2.0 * t)
    U = np.full(w.shape, -1.0)
    V = np.full(w.shape, -1.0)
    for i, v in enumerate(views):
        dv = D @ v.F
        front = dv[:, 2] > 1e-9
        uu, vy = v.project(np.where(front[:, None], dv, 1.0))
        inside = front & (uu >= FACE_MARGIN_PX - 0.5) & (uu <= v.width - 0.5 - FACE_MARGIN_PX) & \
            (vy >= FACE_MARGIN_PX - 0.5) & (vy <= v.height - 0.5 - FACE_MARGIN_PX)
        w[~inside, i] = 0.0
        U[:, i], V[:, i] = uu, vy + offsets[i]
    first = np.argmax(D @ n.T, 1)                                         # the face remap_tables reads
    tot = w.sum(1)
    many = (np.count_nonzero(w, 1) >= 2) & (tot > 0)
    rows = np.flatnonzero(many)
    W = w[rows] / tot[rows, None]
    k0 = first[rows]
    order = np.argsort(-W, 1)
    out_w = np.zeros((len(rows), 3), np.float32)
    out_x = np.full((len(rows), 2), -1.0, np.float32)
    out_y = np.full((len(rows), 2), -1.0, np.float32)
    r = np.arange(len(rows))
    out_w[:, 0] = W[r, k0]
    col = np.zeros(len(rows), int)
    for c in range(order.shape[1]):
        kc = order[:, c]
        take = (kc != k0) & (W[r, kc] > 0) & (col < 2)
        out_w[take, 1 + col[take]] = W[take, kc[take]]
        out_x[take, col[take]] = U[rows[take], kc[take]]
        out_y[take, col[take]] = V[rows[take], kc[take]]
        col += take
    out_w /= out_w.sum(1, keepdims=True)                     # (a weight beyond two other faces: none on a cube)
    idx = np.flatnonzero(ok.ravel())[rows].astype(np.int32)
    return idx, out_x, out_y, out_w


def view_sizes_cost(views) -> float:
    """Megapixels a camera frame renders."""
    return sum(v.width * v.height for v in views) / 1e6


def bridge_memory_gb(cams: list, gt_kinds: list, workers: int = 3) -> float:
    """The bridge's peak memory (GB, an estimate with a 30 % margin) for the rig's cameras: the remap tables (about
    11 bytes per output pixel), the ground truth's per-pixel tables (57), the frames in flight (CB.MAX_CAMERA_JOBS of
    every camera's views, 4 bytes per rendered pixel per camera kind) and each worker's temporaries; 0.5 GB besides."""
    k = len(gt_kinds)
    out = sum(c['image']['width'] * c['image']['height'] for c in cams)
    rend = [c['render_mpx'] * 1e6 for c in cams]
    per_frame = (sum(rend) / max(len(rend), 1)) * 4 * (1 + k)
    worker = max(rend, default=0.0) * 4 * (k + 1) + max((c['image']['width'] * c['image']['height'] for c in cams),
                                                         default=0) * (60 if k else 20)
    b = 0.5e9 + out * (11 + (57 if k else 0)) + CB.MAX_CAMERA_JOBS * per_frame + workers * worker
    return round(1.3 * b / 1e9, 2)


# =========================================================================== lidar
def lidar_plan(model, max_channels: int = 256, tol_deg: float = 1e-4) -> dict:
    """CARLA's channels for a lidar model's beams (see the module doc): {'channels', 'upper_fov', 'lower_fov',
    'keep' (CARLA channel -> ring, −1 dropped; ring 0 the lowest beam), 'elevation_error_deg', 'exact', 'note'}."""
    el = np.sort(np.asarray(model.elevations(), float))
    n = len(el)
    if n == 1:
        return {'channels': 1, 'upper_fov': float(el[0]), 'lower_fov': float(el[0]), 'keep': [0],
                'elevation_error_deg': 0.0, 'exact': True, 'note': 'One beam.'}
    lo, hi = float(el[0]), float(el[-1])
    span = hi - lo
    found = None
    for N in range(n - 1, max(int(max_channels), n) ):
        g = span / N
        k = np.round((el - lo) / g)
        if np.max(np.abs(lo + k * g - el)) <= tol_deg and len(np.unique(k)) == n:
            found = (N, k.astype(int), 0.0)
            break
    if found is None:
        N = max(int(max_channels), n) - 1
        g = span / N
        k = np.round((el - lo) / g).astype(int)
        for i in range(1, n):                         # beams closer than the grid: the next free channel up …
            k[i] = max(k[i], k[i - 1] + 1)
        for i in range(n - 1, -1, -1):                # … and back down from the top so none passes the last
            k[i] = min(k[i], N - (n - 1 - i))
            if i < n - 1:
                k[i] = min(k[i], k[i + 1] - 1)
        found = (N, k, float(np.max(np.abs(lo + k * g - el))))
    N, k, err = found
    ch = N + 1
    keep = [-1] * ch
    for ring, kk in enumerate(k):
        keep[N - int(kk)] = ring                      # CARLA channel 0 is the highest
    if err == 0.0 and ch == n:
        note = f'{n} evenly spaced beams: CARLA\'s channels exactly.'
    elif err == 0.0:
        note = (f'{n} beams on an even {span / N:.4f}° grid: {ch} CARLA channels cast, {ch - n} between the beams '
                f'dropped ({ch / n:.2f}× the rays); every elevation exact.')
    else:
        note = (f'{n} beams on no even grid within {ch} channels: each takes its nearest of {ch} CARLA channels, '
                f'elevation error up to {err:.4f}°.')
    return {'channels': ch, 'upper_fov': hi, 'lower_fov': lo, 'keep': keep, 'elevation_error_deg': err,
            'exact': err == 0.0, 'note': note}


def radar_cone(az_half: float, el_half: float, samples: int = 40000) -> dict:
    """CARLA's radar field for a zone: its rays end on an ellipse of half-tangents a, b at the range, so the ellipse
    a = √2·tan(az_half), b = √2·tan(el_half) / cos(az_half) holds the zone's whole azimuth × elevation rectangle (the
    elevation is measured from the horizontal plane, so the rectangle reaches tan(el)/cos(az) at its sides); detections
    outside the rectangle are dropped. inside: the share of CARLA's rays (radius uniform, as CARLA draws them) that land
    in the rectangle, from a fixed sample."""
    ta, te = math.tan(math.radians(min(az_half, 88.0))), math.tan(math.radians(min(el_half, 88.0)))
    a = min(math.sqrt(2.0) * ta, math.tan(math.radians(89.0)))
    b = min(math.sqrt(2.0) * te / math.cos(math.radians(min(az_half, 88.0))), math.tan(math.radians(89.0)))
    rng = np.random.default_rng(12345)
    r = rng.uniform(0.0, 1.0, samples)
    phi = rng.uniform(0.0, 2.0 * math.pi, samples)
    y, z = a * r * np.cos(phi), b * r * np.sin(phi)
    az = np.degrees(np.arctan(y))
    el = np.degrees(np.arctan2(z, np.sqrt(1.0 + y * y)))
    inside = float(np.mean((np.abs(az) <= az_half) & (np.abs(el) <= el_half)))
    return {'horizontal_fov': 2.0 * math.degrees(math.atan(a)), 'vertical_fov': 2.0 * math.degrees(math.atan(b)),
            'inside': max(inside, 1e-3)}


def _columns(model) -> int:
    return max(int(round(model.h_fov / max(model.h_res, 1e-6))), 1)


# =========================================================================== schedule
def _ratio_ok(R: float, rate: float) -> bool:
    n = R / rate
    return rate > 0 and abs(n - round(n)) < 1e-9 and round(n) >= 1


def tick_plan(lidars: list, others: list, settings: SimSettings) -> dict:
    """The simulation step (see the module doc). lidars: [(rate_hz, columns, rays per second at the model's grid)];
    others: [(name, rate_hz)]. Returns {'tick_hz', 'per_lidar': [(m steps per rotation, q oversampling)],
    'inexact': [names], 'candidates'}."""
    lid_rates = [r for r, _, _ in lidars if r > 0]
    base_rates = lid_rates or [r for _, r in others if r > 0] or [10.0]
    if settings.tick_hz > 0:
        cands = [float(settings.tick_hz)]
    else:
        base = max(base_rates)
        lo, hi = settings.min_tick_hz, settings.max_tick_hz
        cands = [base * k for k in range(1, 2000) if base * k <= hi + 1e-9]
        cands = [R for R in cands if all(_ratio_ok(R, r) for r in base_rates)]
        cands = [R for R in cands if R >= lo - 1e-9] or cands[-1:] or [base]
    best = None
    for R in cands:
        per, cost = [], 0.0
        for rate, cols, rays in lidars:
            m = max(int(round(R / rate)), 1)
            q = next((q for q in range(1, 17) if (cols * q) % m == 0), None)
            if q is None:
                q = 16
            per.append((m, q))
            cost += q * rays
        inexact = [n for n, r in others if r > 0 and not _ratio_ok(R, r)]
        key = (round(cost / 1e5), len(inexact), -R)       # equal cost: the finer step (trigger times, IMU rate)
        if best is None or key < best[0]:
            best = (key, R, per, inexact)
    _, R, per, inexact = best
    return {'tick_hz': float(R), 'per_lidar': per, 'inexact': inexact, 'candidates': cands}


# =========================================================================== the rig
@dataclass
class SimRig:
    settings: SimSettings
    project: str
    variant: str
    variant_uid: str
    doc: dict                          # sim_rig.json
    remaps: dict                       # npz key -> float32 array
    notes: list = field(default_factory=list)
    seconds: float = 0.0


def _variant(project, variant):
    if isinstance(variant, str):
        variant = project.variant_by_uid(variant)
    return variant or project.variant


def _with_variant(project, v):
    from . import dataset as DSET
    return DSET._project_for(project, v)


def _pose(o) -> dict:
    return {'x': float(o.x), 'y': float(o.y), 'z': float(o.z), 'yaw': float(o.yaw), 'pitch': float(o.pitch),
            'roll': float(o.roll)}


def _carla_pose(pos, R) -> dict:
    p, y, r = FR.carla_rotation(R)
    return {'location': [round(v, 6) for v in FR.carla_location(pos)],
            'rotation': {'pitch': round(p, 6), 'yaw': round(y, 6), 'roll': round(r, 6)}}


def _m4(T) -> list:
    return [[round(float(v), 10) for v in row] for row in np.asarray(T, float)]


def build(project, settings: SimSettings | None = None, variant=None, progress=None, cancel=None,
          tables: bool = True) -> SimRig | None:
    """The simulator rig of a variant (the active one by default). tables=False leaves the remap tables out (a
    summary: the views are still planned). Raises ValueError when nothing records."""
    import time
    from . import dataset as DSET
    from . import exporters as EX
    from . import timing as TM
    t0 = time.perf_counter()
    st = settings or settings_of(project)
    v = _variant(project, variant)
    p = _with_variant(project, v)
    v = p.variant

    def prog(f, text=''):
        if progress is not None:
            progress(min(max(float(f), 0.0), 1.0), text)

    def stopped():
        return cancel is not None and cancel()
    notes = []
    tp = TM.plan(p, v)
    readout = tp.settings.readout_ms / 1000.0
    exposure = tp.settings.exposure_ms / 1000.0
    period = tp.period_ms / 1000.0 if tp.period_ms else None
    ctags, stags = EX.safe_tags(v), EX.sensor_tags(v)
    cams = [c for c in v.cameras if c.visible]
    sens = [s for s in v.sensors if s.visible]
    lid = [s for s in sens if s.kind == 'lidar']
    if not cams and not any(s.kind in ('lidar', 'radar') for s in sens):
        raise ValueError('Nothing to simulate: the variant has no visible camera, lidar or radar.')
    # ---- rates and the schedule
    cam_rate = {}
    for c in cams:
        e = tp.cam(c.uid) or {}
        fps = float(p.module_for(c).fps or 0.0)
        trig = e.get('trigger_ms')
        trig_ok = bool(e.get('triggered')) and trig is not None
        if period is not None and st.camera_rate == 'lidar':
            dt, off = period, (trig / 1000.0 if trig_ok else 0.0)
        else:
            if fps <= 0:
                notes.append(f'{c.tag}: its module has no frame rate; the camera is left out.')
                continue
            dt = 1.0 / fps
            off = ((trig / 1000.0) % dt) if (period is not None and trig_ok) else 0.0
        cam_rate[c.uid] = (1.0 / dt, off + readout / 2 + exposure / 2, trig_ok, trig)
    cams = [c for c in cams if c.uid in cam_rate]
    lid_info = []
    for s in lid:
        m = p.sensor_model_for(s)
        lp = lidar_plan(m, st.max_lidar_channels)
        cols = _columns(m)
        lid_info.append((float(m.rate_hz), cols, lp['channels'] * cols * float(m.rate_hz)))
    others = [(c.tag, cam_rate[c.uid][0]) for c in cams] + \
        [(s.tag, float(DSET.GNSS_HZ if s.kind == 'gnss' else p.sensor_model_for(s).rate_hz))
         for s in sens if s.kind != 'lidar']
    tick = tick_plan(lid_info, others, st)
    R = tick['tick_hz']
    dt = 1.0 / R
    cam_tags = {c.tag for c in cams}
    for name in tick['inexact']:
        rate = dict(others)[name]
        if name in cam_tags:
            notes.append(f'{name}: {rate:g} Hz is no whole number of {R:g} Hz simulation steps: recorded at '
                         f'{R / max(round(R / rate), 1):.4g} Hz.')
        elif rate > R:
            notes.append(f'{name}: {rate:g} Hz is faster than the {R:g} Hz simulation step: recorded at {R:g} Hz (one '
                         'sample per step).')
        else:
            notes.append(f'{name}: {rate:g} Hz on {R:g} Hz steps: each sample at the step nearest its time (within '
                         f'{500.0 / R:.2f} ms), the rate exact on average.')
    prog(0.05, 'Cameras')
    # ---- export context (the nuScenes rows and frames are the dataset's)
    ctx = EX._context(p, None, EX.ExportOptions(frame='rear_axle'))
    models = {cm.cam.uid: cm for cm in ctx.models}
    calib = dict(EX._nuscenes(ctx))
    cal_rows = {r['sensor_token']: r for r in json.loads(calib['nuscenes/calibrated_sensor.json'])}
    sens_rows = {r['channel']: r for r in json.loads(calib['nuscenes/sensor.json'])}
    # channels as core.dataset names them (nuScenes names optional)
    chans = []
    for c in cams:
        tag = ctags[c.uid]
        chans.append(DSET._Chan('camera', c, tag, f'CAM_{tag.upper()}', '/' + DSET._ros_name(tag), np.zeros(1),
                                np.zeros(1), cam_rate[c.uid][0]))
    for s in sens:
        tag = stags[s.uid]
        name = {'lidar': f'LIDAR_{tag.upper()}', 'radar': f'RADAR_{tag.upper()}'}.get(s.kind, '')
        chans.append(DSET._Chan(s.kind, s, tag, name, '/' + DSET._ros_name(tag), np.zeros(1), np.zeros(1),
                                float(p.sensor_model_for(s).rate_hz or 0.0)))
    for c in chans:
        c.info['export_channel'] = c.channel
    if st.channel_names == 'nuscenes':
        notes += DSET._nuscenes_names(p, [c for c in chans if c.kind in ('camera', 'lidar', 'radar')])
    by_uid = {c.sensor.uid: c for c in chans}

    def nus_rows(ch):
        if not ch.channel:
            return None, None
        srow = dict(sens_rows[ch.info['export_channel']], channel=ch.channel)
        return srow, dict(cal_rows[srow['token']])
    remaps = {}
    doc_cams = []
    sensor_tab, cal_tab, tfs = [], [], []
    gt = dict(st.ground_truth)
    gt_kinds = CB.gt_camera_kinds(gt)
    for i, c in enumerate(cams):
        if stopped():
            return None
        prog(0.05 + 0.6 * i / max(len(cams), 1), f'Remap tables · {c.tag}')
        cm = models[c.uid]
        ch = by_uid[c.uid]
        intr = resolve(p.module_for(c), p.lens_for(c))
        w, h, K, dmodel, dcoef = DSET._camera_setup(p, cm, st.image_scale, st.undistort)
        d, ok = output_rays(intr, w, h, K if (st.undistort and intr.projection != 'rectilinear') else None)
        f_out = float(intr.f_px) * w / intr.width
        views = plan_views(d, ok, f_out, st.face_oversample)
        key = ctags[c.uid]
        offs = list(np.cumsum([0] + [vw.height for vw in views[:-1]]).astype(int).tolist()) if views else []
        blend = None
        if tables:
            mx, my, offs = remap_tables(d, ok, views)
            remaps[f'{key}__map_x'] = mx
            remaps[f'{key}__map_y'] = my
            blend = blend_tables(d, ok, views, offs)
            if blend is not None:
                for name, arr in zip(('idx', 'x', 'y', 'w'), blend):
                    remaps[f'{key}__blend_{name}'] = arr
        R_cam = rot_zyx(c.yaw, c.pitch, c.roll)
        O2V = R_cam @ BODY_TO_OPTICAL.T                         # optical -> vehicle
        if st.undistort and intr.projection != 'rectilinear':
            lens_doc = {'model': 'pinhole', 'K': _m4(K), 'max_theta_deg': math.degrees(intr.max_theta),
                        'image_size': [w, h]}
        else:
            lens_doc = {'model': 'lens', 'projection': intr.projection, 'f_px': float(intr.f_px),
                        'kb': [float(x) for x in intr.kb], 'cx': float(intr.cx), 'cy': float(intr.cy),
                        'aspect': float(intr.aspect), 'max_theta_deg': math.degrees(intr.max_theta),
                        'sensor_size': [int(intr.width), int(intr.height)], 'image_size': [w, h]}
        doc_views = []
        for vw in views:
            R_body = O2V @ vw.F @ BODY_TO_OPTICAL               # the view's body frame (x along it) -> vehicle
            doc_views.append({'name': vw.name, 'width': vw.width, 'height': vw.height, 'f_px': vw.f,
                              'fov_deg': vw.fov_deg, 'optical': _m4(vw.F),
                              'carla': _carla_pose(c.pos(), R_body)})
        rate, centre, trig_ok, trig = cam_rate[c.uid]
        n = max(int(round(R / rate)), 1)
        q = int(round(centre * R)) % n
        rec_c = q / R                                       # the recorded exposure centre nearest the plan's
        rec_c += round((centre - rec_c) * R / n) * n / R
        srow, row = nus_rows(ch)
        if row is not None:
            row['camera_intrinsic'] = [[EX._r(K[a, b], 6) for b in range(3)] for a in range(3)]
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
        T = np.eye(4)
        T[:3, :3] = cm.R
        T[:3, 3] = c.pos() - ctx.origin
        tfs.append({'parent': 'base_link', 'child': f'{key}_link', 'matrix': _m4(T)})
        To = np.eye(4)
        To[:3, :3] = BODY_TO_OPTICAL.T
        tfs.append({'parent': f'{key}_link', 'child': f'{key}_optical_frame', 'matrix': _m4(To)})
        doc_cams.append({
            'uid': c.uid, 'tag': c.tag, 'safe_tag': key, 'name': c.name, 'channel': ch.channel,
            'topic': ch.topic, 'frame_id': f'{key}_optical_frame', 'pose_vehicle': _pose(c),
            'image': {'width': w, 'height': h, 'K': _m4(K), 'distortion_model': dmodel, 'D': list(dcoef),
                      'rectified': bool(st.undistort), 'sensor_size': [intr.width, intr.height],
                      'lens_projection': intr.projection, 'max_theta_deg': math.degrees(intr.max_theta)},
            'views': doc_views, 'atlas': {'offsets': offs, 'width': max((vw.width for vw in views), default=0),
                                          'height': int(sum(vw.height for vw in views))},
            'remap': dict({'x': f'{key}__map_x', 'y': f'{key}__map_y'},
                          **({'blend': {n: f'{key}__blend_{n}' for n in ('idx', 'x', 'y', 'w')},
                              'seam_blend_deg': SEAM_BLEND_DEG} if len(views) > 1 else {})),
            'capture': {'rate_hz': rate, 'every_ticks': n, 'phase_ticks': q, 'recorded_hz': R / n,
                        'exposure_centre_s': centre, 'recorded_centre_s': rec_c,
                        'offset_error_ms': 1000.0 * (rec_c - centre), 'triggered': trig_ok, 'trigger_ms': trig},
            'lens': lens_doc,
            'render_mpx': round(view_sizes_cost(views), 3),
            'gt_render_mpx': round(view_sizes_cost(views) * len(gt_kinds), 3)})
        if abs(R / n - rate) > 1e-6:
            notes.append(f'{c.tag}: {rate:g} Hz recorded at {R / n:.4g} Hz (a whole number of steps).')
    prog(0.7, 'Lidars, radars, GNSS and IMU')
    doc_lid, doc_rad, doc_imu, doc_gnss = [], [], [], []
    li = 0
    for s in sens:
        m = p.sensor_model_for(s)
        ch = by_uid[s.uid]
        key = stags[s.uid]
        Rs = rot_zyx(s.yaw, s.pitch, s.roll)
        T = EX._T_vehicle_sensor(ctx, s)
        tfs.append({'parent': 'base_link', 'child': f'{key}_link', 'matrix': _m4(T)})
        srow, row = nus_rows(ch)
        if row is not None:
            sensor_tab.append(srow)
            cal_tab.append(row)
        base = {'uid': s.uid, 'tag': s.tag, 'safe_tag': key, 'name': s.name, 'model': f'{m.vendor} {m.name}'.strip(),
                'channel': ch.channel, 'topic': ch.topic, 'frame_id': f'{key}_link', 'pose_vehicle': _pose(s)}
        if s.kind == 'lidar':
            lp = lidar_plan(m, st.max_lidar_channels)
            mm, q = tick['per_lidar'][li]
            li += 1
            cols = _columns(m)
            step = m.h_fov / cols
            # CARLA's head points backwards (azimuth 180°) at angle 0, the sync pulse: turn the mount so it faces the
            # planned phase, on the azimuth grid
            turn = (float(s.phase_deg) - 180.0) if m.h_fov >= 359.999 else 0.0
            turn = round(turn / step) * step
            Rm = Rs @ FR.rot_z(turn)
            pps = int(round(lp['channels'] * cols * q * float(m.rate_hz)))
            per_tick = cols * q / mm
            if abs(per_tick - round(per_tick)) > 1e-9:
                notes.append(f'{s.tag}: {cols} columns do not split into {mm} steps per rotation: the azimuth step '
                             'is rounded by CARLA.')
            noise = st.lidar_noise_cm / 100.0
            doc_lid.append(dict(base, carla=_carla_pose(s.pos(), Rm), mount_turn_deg=turn, blueprint={
                'channels': lp['channels'], 'upper_fov': lp['upper_fov'], 'lower_fov': lp['lower_fov'],
                'points_per_second': pps, 'rotation_frequency': float(m.rate_hz), 'range': float(m.range_m),
                'horizontal_fov': float(m.h_fov), 'dropoff_general_rate': 0.0, 'dropoff_intensity_limit': 1.0,
                'dropoff_zero_intensity': 0.0, 'noise_stddev': noise, 'atmosphere_attenuation_rate': 0.004,
                'sensor_tick': 0.0}, keep=lp['keep'], rings=m.n_channels(), elevation_error_deg=lp['elevation_error_deg'],
                plan_note=lp['note'], columns=cols, azimuth_step_deg=step, oversample=q, steps_per_rotation=mm,
                rate_hz=float(m.rate_hz), range_min_m=float(m.range_min_m), scan=m.scan,
                rotation='clockwise' if m.rotation == 'cw' else 'counter-clockwise',
                rays_per_s=lp['channels'] * cols * q * float(m.rate_hz)))
            if m.rotation != 'cw' and m.spinning:
                notes.append(f'{s.tag}: the model turns counter-clockwise; CARLA\'s lidar turns clockwise (the points\' '
                             'times run the other way round).')
            if q > 1:
                notes.append(f'{s.tag}: the azimuth step is cast {q}× finer and thinned back to {step:.4f}° '
                             f'({q}× the rays) so it is exact at {R:g} Hz steps.')
            notes.append(f'{s.tag}: {lp["note"]}')
        elif s.kind == 'radar':
            rate = float(m.rate_hz or 0.0) or 10.0
            per_scan = max(R / rate, 1.0)                 # steps per scan (on average)
            zones = []
            for zi, (rng, ah, eh) in enumerate(m.zones):
                cone = radar_cone(ah, eh)
                pts_tick = max(int(math.ceil(st.radar_points / cone['inside'] / per_scan)), 1)
                zones.append({'range': float(rng), 'az_half_deg': float(ah), 'el_half_deg': float(eh),
                              'inside_share': cone['inside'],
                              'blueprint': {'horizontal_fov': cone['horizontal_fov'], 'vertical_fov': cone['vertical_fov'],
                                            'range': float(rng), 'points_per_second': int(pts_tick * R),
                                            'sensor_tick': 0.0}})
            doc_rad.append(dict(base, carla=_carla_pose(s.pos(), Rs), zones=zones, rate_hz=rate,
                                recorded_hz=min(rate, R), range_min_m=float(m.range_min_m)))
        elif s.kind == 'imu':
            rate = float(m.rate_hz or 0.0) or 100.0
            from . import drivescene as DS
            doc_imu.append(dict(base, carla=_carla_pose(s.pos(), Rs), rate_hz=rate, recorded_hz=min(rate, R),
                                antennas=int(m.antennas), attitude_sigma_deg=float(m.attitude_sigma_deg),
                                heading_sigma_deg=float(m.heading_sigma_deg),
                                heading_baseline_m=float(m.heading_baseline_m), noise=dict(DS.IMU_NOISE)))
        elif s.kind == 'gnss':
            ins = next((p.sensor_model_for(x) for x in sens if x.kind == 'imu'
                        and p.sensor_model_for(x).pos_sigma_cm > 0), None)
            sig_h = (ins.pos_sigma_cm / 100.0) if ins is not None else 0.02
            sig_deg = math.degrees(sig_h / EARTH_R)
            doc_gnss.append(dict(base, carla=_carla_pose(s.pos(), Rs), rate_hz=DSET.GNSS_HZ,
                                 recorded_hz=min(DSET.GNSS_HZ, R), sigma_h_m=sig_h, blueprint={
                                     'noise_lat_stddev': sig_deg, 'noise_lon_stddev': sig_deg,
                                     'noise_alt_stddev': 2.0 * sig_h, 'sensor_tick': 0.0}))
    # ---- warm-up: a whole number of lidar rotations, at least 3 steps (the radar's and IMU's first steps are garbage)
    # and one camera period before the sync pulse
    mm_all = [l['steps_per_rotation'] for l in doc_lid] or [1]
    lcm = 1
    for x in mm_all:
        lcm = lcm * x // math.gcd(lcm, x)
    need = max([3] + [c['capture']['every_ticks'] + 1 for c in doc_cams])
    warm = lcm * int(math.ceil(need / lcm))
    veh = match_vehicle(p, st.vehicle)
    wx = carla_weather(p, st)
    notes += wx['notes']
    # ---- ground truth, faults, the scenario, an actors scene
    kinds = {c['tag']: 'camera' for c in doc_cams}
    kinds.update({x['tag']: k for k, docs in (('lidar', doc_lid), ('radar', doc_rad), ('imu', doc_imu),
                                              ('gnss', doc_gnss)) for x in docs})
    faults, fbad = CB.clean_faults(st.faults, kinds)
    notes += fbad
    scen, sbad = CB.clean_scenario(st.scenario.get('name'), st.scenario.get('params'))
    notes += sbad
    actors, abad = CB.clean_actors(st.actors)
    notes += abad
    if actors and scen['name'] != 'free_drive':
        notes.append(f'An actors scene: the {scen["name"]} scenario is not run (the ego stands still with the actors).')
    rgb_mpx = sum(c['render_mpx'] for c in doc_cams)
    gt_mpx = sum(c['gt_render_mpx'] for c in doc_cams)
    gt_doc = dict(gt, camera_kinds=gt_kinds, render_mpx=round(rgb_mpx, 3), gt_render_mpx=round(gt_mpx, 3),
                  lidar_sensors=len(doc_lid) if gt.get('lidar_labels') else 0, note=GT_NOTE)
    mem = bridge_memory_gb(doc_cams, gt_kinds)
    if mem > 3.5:
        notes.append(f'The bridge needs about {mem:.1f} GB (its cap is set from this; full-size images with ground '
                     'truth are the heavy case).')
    if rgb_mpx + gt_mpx > CB.RENDER_BUDGET_MPX:
        notes.append(f'The bridge refuses this run: {rgb_mpx + gt_mpx:.0f} Mpx per frame is over its render budget of '
                     f'{CB.RENDER_BUDGET_MPX:g} (a 0.9.16 server capped at 12 GB crashed at 306 Mpx; --render-budget-mpx '
                     'raises it for a bigger server).')
    elif rgb_mpx + gt_mpx > SERVER_MPX_WARN:
        notes.append(f'CARLA renders {rgb_mpx + gt_mpx:.0f} Mpx per frame of all cameras: more than the server may '
                     f'hold (a 0.9.16 server capped at 12 GB crashed at 306 Mpx: six 1920 × 1536 fisheyes with ground '
                     f'truth; 77 Mpx, the same at half size, ran). Lower the image size, or record the ground truth of '
                     'fewer cameras or none.')
    if gt.get('lidar_labels') and doc_lid and not st.nuscenes:
        notes.append('Lidar labels are written as nuScenes-lidarseg and -panoptic files: they need the nuScenes '
                     'output (on).')
    if gt_mpx:
        notes.append(f'Ground truth renders {gt_mpx:.1f} Mpx per frame of all cameras beside the images\' '
                     f'{rgb_mpx:.1f} ({" + ".join(gt_kinds)}).')
    # keyframes: every n-th sweep of the first lidar (else frame of the first camera), as core.dataset
    ref_rate = doc_lid[0]['rate_hz'] if doc_lid else (doc_cams[0]['capture']['recorded_hz'] if doc_cams else
                                                      (doc_rad[0]['recorded_hz'] if doc_rad else 1.0))
    meas = {}
    try:
        meas = p.vehicle.measured() if p.vehicle.mesh is not None else {}
    except Exception:
        meas = {}
    doc = {
        'schema': SCHEMA, 'carla_version': CARLA_VERSION, 'project': p.name, 'variant': v.name, 'variant_uid': v.uid,
        'created': _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat(),
        'frames': {
            'vehicle': 'ISO 8855: x forward, y left, z up, metres, origin at the rear-axle centre on the ground; '
                       'yaw/pitch/roll intrinsic Z-Y-X degrees, pitch positive down',
            'carla': 'Unreal (left-handed): x forward, y right, z up, metres; carla.Rotation(pitch: nose up +, yaw: '
                     'towards +y +, roll: right side down +) degrees, matrix Transform::GetMatrix',
            'conversion': 'location (x, −y, z); rotation by conjugation with diag(1, −1, 1): for a yaw/pitch/roll '
                          'pose, carla yaw = −yaw, pitch = −pitch, roll = roll',
            'sensor_locations': 'relative to the ego origin (rear-axle centre on the ground) in CARLA\'s axes; the '
                                'bridge adds where that origin lies in the CARLA vehicle\'s own frame'},
        'vehicle': dict(veh, name=p.vehicle.spec_name or p.vehicle.name, measured={
            k: round(float(meas[k]), 4) for k in ('Length', 'Width', 'Height', 'Wheelbase', 'front_x', 'rear_x')
            if k in meas}),
        'simulation': {'tick_hz': R, 'dt_s': dt, 'warmup_ticks': warm, 'tick_candidates': tick['candidates'],
                       'bridge_memory_gb': mem, 'render_mpx_per_frame': round(rgb_mpx + gt_mpx, 3),
                       'timing': {'lidar_period_s': period, 'readout_s': readout, 'exposure_s': exposure}},
        'scenario': {'town': st.town, 'weather': st.weather, 'duration_s': st.duration_s, 'traffic': st.traffic,
                     'seed': st.seed, 'warmup_s': st.warmup_s, 'name': scen['name'], 'params': scen['params'],
                     'walkers': st.walkers},
        'ground_truth': gt_doc,
        'faults': faults,
        'actors': {'items': actors, 'spawn_point': int(st.spawn_point),
                   'frame': 'vehicle (ISO 8855, rear axle on the ground): x, y the target\'s ground-contact centre '
                            '(its box centre on the ground), z its lowest point, yaw its heading; metres, degrees'},
        'server': {'port': st.port, 'tm_port': st.tm_port, 'quality': st.quality, 'memory_gb': st.memory_gb,
                   'timeout_s': st.timeout_s},
        'weather': wx,
        'output': {'image_scale': st.image_scale, 'jpeg_quality': st.jpeg_quality, 'nuscenes': st.nuscenes,
                   'rosbag': st.rosbag, 'channel_names': st.channel_names, 'bag_name': DSET.bag_name(p),
                   'keyframe_hz': st.keyframe_hz, 'keyframe_ref_hz': ref_rate, 'lidar_noise_cm': st.lidar_noise_cm,
                   'drop_self_hits': st.drop_self_hits, 'camera_rate': st.camera_rate, 'undistort': st.undistort,
                   'exposure': st.exposure, 'exposure_ev': st.exposure_ev, 'gt_range_m': GT_RANGE_M,
                   'dataset_version': DSET.VERSION, 'epoch_us': DSET.EPOCH_US},
        'cameras': doc_cams, 'lidars': doc_lid, 'radars': doc_rad, 'imus': doc_imu, 'gnss': doc_gnss,
        'nuscenes': {'sensor': sensor_tab, 'calibrated_sensor': cal_tab, 'categories': DSET.CATEGORIES,
                     'attributes': DSET.ATTRIBUTES, 'visibility': DSET.VISIBILITY},
        'tf_static': tfs,
        'assumptions': ASSUMPTIONS,
        'notes': notes,
    }
    prog(1.0, 'Done')
    return SimRig(st, p.name, v.name, v.uid, doc, remaps, notes, time.perf_counter() - t0)


def run(project, engine=None, settings=None, variant=None, progress=None, cancel=None) -> SimRig | None:
    """build() with the module contract's arguments (no engine needed)."""
    st = settings if isinstance(settings, SimSettings) else (SimSettings.from_dict(settings) if settings is not None
                                                            else settings_of(project))
    return build(project, st, variant, progress, cancel)


# =========================================================================== reports
def summary(rig: SimRig) -> dict:
    d = rig.doc
    return {'project': rig.project, 'variant': rig.variant, 'carla_version': d['carla_version'],
            'bridge_memory_gb': d['simulation'].get('bridge_memory_gb'),
            'tick_hz': d['simulation']['tick_hz'], 'warmup_ticks': d['simulation']['warmup_ticks'],
            'vehicle': d['vehicle']['blueprint'], 'vehicle_size_error': d['vehicle'].get('size_error'),
            'weather': d['weather']['label'],
            'cameras': [{'tag': c['tag'], 'views': [v['name'] for v in c['views']], 'image': [c['image']['width'],
                         c['image']['height']], 'render_mpx': c['render_mpx'], 'hz': round(c['capture']['recorded_hz'], 4)}
                        for c in d['cameras']],
            'lidars': [{'tag': l['tag'], 'carla_channels': l['blueprint']['channels'], 'rings': l['rings'],
                        'elevation_error_deg': l['elevation_error_deg'], 'oversample': l['oversample'],
                        'rays_per_s': l['rays_per_s']} for l in d['lidars']],
            'radars': [{'tag': r['tag'], 'zones': len(r['zones']), 'hz': round(r['recorded_hz'], 4)} for r in d['radars']],
            'imus': [x['tag'] for x in d['imus']], 'gnss': [x['tag'] for x in d['gnss']],
            'ground_truth': {k: d['ground_truth'][k] for k in CB.GT_KEYS},
            'gt_render_mpx': d['ground_truth']['gt_render_mpx'], 'render_mpx': d['ground_truth']['render_mpx'],
            'scenario': d['scenario']['name'], 'faults': len(d['faults']), 'actors': len(d['actors']['items']),
            'notes': list(rig.notes)}


def to_doc(rig: SimRig, project=None) -> dict:
    return rig.doc


def remap_npz(rig: SimRig) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(buf, **rig.remaps)
    return buf.getvalue()


def files(rig: SimRig, project=None) -> list:
    """[(name, bytes)]: sim_rig.json and sim_remap.npz (what the bridge reads); faults.json and actors.json too when
    the rig has faults or an actors scene (the same lists sim_rig.json carries, for the bridge's --faults and
    --actors)."""
    out = [('sim_rig.json', (json.dumps(rig.doc, indent=1) + '\n').encode('utf-8')),
           ('sim_remap.npz', remap_npz(rig))]
    if rig.doc.get('faults'):
        out.append(('faults.json', (json.dumps(rig.doc['faults'], indent=1) + '\n').encode('utf-8')))
    if rig.doc.get('actors', {}).get('items'):
        out.append(('actors.json', (json.dumps(rig.doc['actors']['items'], indent=1) + '\n').encode('utf-8')))
    return out


def export(rig: SimRig, project, out_dir) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    out = []
    for name, data in files(rig, project):
        path = os.path.join(out_dir, name)
        with open(path + '.tmp', 'wb') as fh:
            fh.write(data)
        os.replace(path + '.tmp', path)
        out.append(path)
    return out


def read_run(out_dir) -> dict | None:
    """The bridge's run.json in out_dir (vantage.sim_run/1), or None."""
    path = os.path.join(out_dir, 'run.json')
    try:
        with open(path, encoding='utf-8') as fh:
            doc = json.load(fh)
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) and doc.get('schema') == RUN_SCHEMA else None
