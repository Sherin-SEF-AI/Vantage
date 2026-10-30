# The simulator bridge

A simulator bridge drives one rig variant of a Vantage project in a simulator and records the drive the way Vantage's
sample dataset records one, so that Vantage's checks run on it unchanged. Vantage ships a bridge for CARLA 0.9.16
(`vantage/sim/carla_bridge.py`, with `record.py`, `frames.py` and `tonemap.py`). This page is the contract between
Vantage and any bridge: the rig file Vantage writes, what a bridge writes back, the timing it keeps, and what each of
Vantage's checks reads. [`sim_rig.schema.json`](sim_rig.schema.json) and [`sim_run.schema.json`](sim_run.schema.json)
describe the two JSON documents of the contract as JSON Schemas (draft 2020-12). `tests/test_sim_contract.py` writes a
rig file as `--carla-rig` does, runs the CARLA bridge against the fake `carla` module in `tests/fake_carla`, validates
both documents against the schemas and checks that every field this page's tables name is in the files, so the page and
the schemas follow the code.

Fields and behaviour specific to CARLA are marked **CARLA** below. A bridge for another simulator (Isaac Sim, a custom
Unreal build) skips them and keeps the rest. Paths in the tables use `a.b` for a key inside an object, `a[]` for every
element of a list and `a.{tag}` for every value of an object keyed by a name; a bare name after a path is a sibling of
that path's last key.

## The flow

```
vantage project.vantage --variant V --carla-rig OUT/rig/      # sim_rig.json, sim_remap.npz
<bridge> OUT/rig/sim_rig.json OUT/                            # nuscenes/, rosbag/, ground_truth/, dataset.json, run.json
vantage project.vantage --variant V --drive-log OUT/nuscenes report/
```

(`V`: the variant's name or 1-based index.)

Vantage's own bridge is `python -m vantage.sim.carla_bridge OUT/rig/sim_rig.json OUT/`, run in CARLA's Python with
Vantage's source on `PYTHONPATH`. When Vantage starts a drive itself (Simulate ▸ CARLA ▸ Run, `--carla-run`, campaigns,
critical targets in CARLA) it writes the rig into `OUT/rig/` and runs the bridge on `OUT/rig/sim_rig.json` with `OUT` as
the output folder; detection quality and critical targets in CARLA read the rig back from `OUT/rig/sim_rig.json`. A
bridge run by hand keeps the rig there as well.

## Conventions

- **Vehicle frame**: ISO 8855: x forward, y left, z up; metres; origin at the centre of the rear axle, on the ground
  (the nuScenes ego frame, the bag's `base_link`). Every `pose_vehicle`, actor pose, `box3d` and the calibration rows
  are in this frame.
- **Orientation**: `yaw`, `pitch`, `roll` in degrees, intrinsic Z-Y-X, **pitch positive down**; with all three zero a
  sensor looks along +x. `R = rot_zyx(yaw, pitch, roll)` (`vantage/core/mathutil.py`) takes the sensor's own frame into
  the vehicle frame.
- **Sensor frames**: a lidar, radar, IMU or GNSS antenna's own frame is x forward, y left, z up (REP 103) at its
  optical centre (an antenna's phase centre, an IMU's axis origin). A camera's optical frame is OpenCV's (x right,
  y down, z forward): `p_optical = BODY_TO_OPTICAL · Rᵀ · (p_vehicle − position)` with
  `BODY_TO_OPTICAL = [[0, −1, 0], [0, 0, −1], [1, 0, 0]]`.
- **Units**: metres, seconds, degrees unless the key says otherwise (`_ms`, `_us`, `_hz`, `_px`, `_cm`, `_deg`,
  `_mpx`, `_gb`); rates in Hz; radar velocities in m/s; IMU rates in rad/s and accelerations in m/s².
- **Pixels**: OpenCV coordinates, pixel centres on integers (`K`, `camera_intrinsic`, the remap tables, the view
  pinholes), except the `lens` block and the ground truth's boxes, which use pixel-edge coordinates (pixel `i` covers
  `[i, i+1)`; the centre of a 1920-pixel row is 960).
- **Time**: simulation time. `t = 0` is the drive's first sync pulse; a stamp is `output.epoch_us + t` in microseconds
  (nuScenes) or nanoseconds (the bag), since 1970-01-01 UTC.
- **Names**: a sensor's `tag` (`F`, `ML`, `L1`, `RD3`) is its name in the project. Its nuScenes channel
  (`CAM_<TAG>`, `LIDAR_<TAG>`, `RADAR_<TAG>`), bag topic prefix (`/<tag in lower case>`) and frame ids
  (`<safe_tag>_link`, `<safe_tag>_optical_frame`) are given in the rig file; a bridge uses them as given. The checks
  find a stream's sensor by the tag in its channel, topic or frame id.
- **CARLA**: Unreal's frame is left-handed (x forward, y right, z up). A position maps as `(x, −y, z)`; a rotation by
  conjugation with `diag(1, −1, 1)`, which for a plain pose is `carla yaw = −yaw`, `pitch = −pitch`, `roll = roll`
  (`vantage/sim/frames.py`).

## The rig file: `sim_rig.json`

Written by `vantage project.vantage [--variant NAME] --carla-rig OUT_DIR`, by Simulate ▸ CARLA ▸ Export rig…, or in
Python by `core.simrig.build(project, settings)` and `core.simrig.export(rig, project, out_dir)`. The folder holds:

| file | what it holds |
|---|---|
| `sim_rig.json` | the rig: UTF-8 JSON, schema `vantage.sim_rig/1` |
| `sim_remap.npz` | the cameras' remap tables (`numpy.savez_compressed`; open with `allow_pickle=False`). Always written; the CARLA bridge needs it when the rig has cameras. |
| `faults.json`, `actors.json` | the rig's `faults` and `actors.items` again, for the CARLA bridge's `--faults` and `--actors`. Present when not empty. |

A batch run adds a `provenance` block (`docs/project-format.md`); a bridge ignores it. The settings that shape the rig
are the project's `analysis.sim` (`docs/project-format.md`), with the batch's `--carla-*` options over them.

```python
import json
import numpy as np

rig = json.load(open('OUT/rig/sim_rig.json', encoding='utf-8'))
assert rig['schema'] == 'vantage.sim_rig/1'
remap = np.load('OUT/rig/sim_remap.npz', allow_pickle=False)
for cam in rig['cameras']:
    p = cam['pose_vehicle']
    print(cam['tag'], cam['channel'], p['x'], p['y'], p['z'], p['yaw'], p['pitch'], p['roll'], cam['lens']['projection'])
```

### Top level

| key | meaning |
|---|---|
| `schema` | `"vantage.sim_rig/1"` |
| `carla_version` | **CARLA**: the release the CARLA blocks were planned for (`0.9.16`) |
| `project`, `variant`, `variant_uid` | the project's name, the variant's name and uid |
| `created` | when the file was written, ISO 8601 UTC |
| `frames.vehicle`, `carla`, `conversion`, `sensor_locations` | the frames above, in words |
| `assumptions` | what a CARLA run assumes, in prose (copied into `dataset.json` and `README.md`) |
| `notes` | what planning the rig found worth saying: rates rounded to the step, the lidar's channel plan, the render cost |

### The schedule

| key | meaning |
|---|---|
| `simulation.tick_hz`, `dt_s` | the simulation step: every planned stamp is a whole number of steps from the sync pulse. Chosen as a multiple of the lidar rates (else the cameras') in the settings' range, for the fewest extra lidar rays, then the most sensors on whole steps, then the finest step. |
| `simulation.warmup_ticks` | steps run with the sensors attached before `t = 0`: a whole number of lidar rotations, at least 3 steps and one camera period |
| `simulation.tick_candidates` | the step rates considered |
| `simulation.timing.lidar_period_s`, `readout_s`, `exposure_s` | the sync plan (`core/timing.py`): the reference lidar's period (null without a lidar), the cameras' readout and exposure |
| `simulation.bridge_memory_gb`, `render_mpx_per_frame` | **CARLA**: the bridge's estimated peak memory; megapixels rendered per frame of all cameras |

### Scenario, weather, vehicle, server

| key | meaning |
|---|---|
| `scenario.duration_s` | seconds recorded from `t = 0` |
| `scenario.seed` | seeds the spawn choices, the traffic and every random choice of the run (a seeded drop fault too) |
| `scenario.traffic`, `walkers` | other vehicles on the autopilot; pedestrians on the walker AI |
| `scenario.warmup_s` | seconds of traffic before the sensors are attached |
| `scenario.name`, `params` | the built-in scenario (`free_drive`, `lead_vehicle_brake`, `cut_in`, `pedestrian_crossing`, `cyclist_crossing`, `parked`) and the parameters that were set; `carla_bridge.SCENARIOS` lists every parameter with its default and range |
| `scenario.town`, `weather` | **CARLA**: the map; the weather setting (`project`, `vantage:<id>` or a CARLA preset) |
| `weather.source`, `label`, `condition` | a CARLA preset (`carla`) or a Vantage weather condition (`vantage`, its id in `condition`) and its name |
| `weather.preset` | **CARLA**: the preset; empty for a mapped condition |
| `weather.params.cloudiness`, `precipitation`, `precipitation_deposits`, `wind_intensity`, `wetness`, `fog_density`, `fog_distance`, `fog_falloff`, `sun_azimuth_angle`, `sun_altitude_angle` | **CARLA**: `WeatherParameters` for a mapped condition (empty for a preset); `core/simrig.py` `carla_weather` says how a condition maps (fog by a measured calibration, light and rain by stated heuristics) |
| `weather.night`, `beam` | night: the ego's headlights on, `low` or `high` beam |
| `weather.notes` | what the mapping could not reproduce |
| `vehicle.name`, `measured`, `project` | the project's vehicle: its model name, its body measured in the vehicle frame (`Length`, `Width`, `Height`, `Wheelbase`, `front_x`, `rear_x`), the same four sizes in lower case (empty without a body) |
| `vehicle.blueprint`, `how`, `size_error`, `note` | **CARLA**: the CARLA vehicle nearest the project's body by size, how it was chosen, its mean relative size difference |
| `vehicle.candidates[].blueprint`, `score` | **CARLA**: the four nearest and their size differences |
| `vehicle.carla.base_type`, `length`, `width`, `height`, `wheelbase`, `rear_axle_x` | **CARLA**: the chosen vehicle's measured box and rear axle (m, actor frame); null for a blueprint Vantage has not measured |
| `server.port`, `tm_port`, `quality`, `memory_gb`, `timeout_s` | **CARLA**: the server and traffic manager the run talks to |

The rig carries no body mesh: a simulator renders its own ego vehicle. The vehicle frame's origin must be placed on
the simulated vehicle's rear-axle centre on the ground (the CARLA bridge reads its wheels and box and reports where in
`run.json` `vehicle.ego_origin_in_actor_m`).

### Traffic: SUMO (`traffic_source`)

**CARLA**. Present only when Eclipse SUMO drives the other vehicles (`analysis.sim.traffic_source` = `sumo`, the batch's
`--carla-traffic sumo`); without it CARLA's traffic manager drives them, and the rig file is exactly what it was before
SUMO existed. The CARLA bridge co-simulates SUMO in lockstep through CARLA's own co-simulation package
(`CARLA_ROOT/Co-Simulation/Sumo/sumo_integration`, as `run_synchronization.py` uses it; `vantage/sim/sumo_cosim.py`):
SUMO steps once before every synchronous CARLA step, the vehicles it inserts become CARLA actors (physics off, moved to
SUMO's pose every step: role `vantage_sumo`), and the ego and a scenario's vehicles are mirrored into SUMO after the
step so its traffic reacts to them. Walkers stay on CARLA's walker AI: CARLA's co-simulation does not synchronise
SUMO's pedestrians, and SUMO's vehicles do not see CARLA's walkers.

| key | meaning |
|---|---|
| `traffic_source.source` | `sumo` |
| `traffic_source.vehicles` | vehicles in the network at once (`scenario.traffic` for a free drive, a scenario's `traffic` parameter; 0 for an actors scene or `parked`: SUMO is not started) |
| `traffic_source.sumo.net` | `auto`: CARLA's bundled net for Town01, Town04, Town05 (`examples/net`), else one generated from the map's OpenDRIVE (`World.get_map().to_opendrive()` through CARLA's `util/netconvert_carla.py`) and cached by the OpenDRIVE's SHA-256; `generate`: generated for a bundled town too; or the path of the user's `.net.xml` or `.sumocfg` (its own demand) |
| `traffic_source.sumo.demand`, `period_s` | `count`: `vehicles` in the network at once (SUMO's `--max-num-vehicles`, plus the mirrored vehicles); `period`: one trip inserted every `period_s`, at most `vehicles` at once. Trips by SUMO's `randomTrips.py` with `--seed` = `seed` |
| `traffic_source.sumo.classes` | SUMO vehicle classes drawn (`passenger`, `evehicle`, `authority`, `truck`, `motorcycle`, `bicycle`): each trip's vType is one of CARLA's measured vTypes of these classes that the server has as a blueprint (CARLA's `data/vtypes.json`), drawn from the seed |
| `traffic_source.sumo.tls` | who runs the traffic lights (`run_synchronization.py --tls-manager`): `carla` (CARLA's lights, copied into SUMO each step) or `sumo` (CARLA's lights frozen and set from SUMO's each step). CARLA's bundled nets have no traffic lights: there SUMO's vehicles do not stop at lights whichever owns them |
| `traffic_source.sumo.binary` | `auto` (the `sumo` package's `SUMO_HOME/bin/sumo` in CARLA's Python: headless, never `sumo-gui`) or a path |
| `traffic_source.net_kind` | what the net will be for `scenario.town`: `bundled`, `generated`, `user` or `user_config` |
| `traffic_source.step_s`, `seed` | SUMO's step (`simulation.dt_s`, or the fewest whole steps that are whole milliseconds: SUMO's clock) and seed (`scenario.seed`) |
| `traffic_source.bundled_towns`, `limits` | the towns with a bundled net; the co-simulation's limits in words |

### Output settings

| key | meaning |
|---|---|
| `output.nuscenes`, `rosbag` | write the nuScenes folder; write the ROS 2 bag |
| `output.dataset_version` | the nuScenes version folder, `v1.0-vantage` |
| `output.bag_name` | the bag's folder under `rosbag/` |
| `output.epoch_us` | `t = 0` in microseconds since 1970 UTC (2026-01-01 12:00:00 UTC; a whole second, so the sync pulses fall on multiples of the lidar period) |
| `output.image_scale`, `jpeg_quality` | images are this share of the sensor's size (the result is in `cameras[].image`); their JPEG quality |
| `output.camera_rate` | `lidar`: one frame per lidar rotation at the sync plan's trigger; `module`: the module's frame rate |
| `output.undistort` | rectified pinhole images instead of the real lens's (the `lens` block of a camera whose lens is not rectilinear is then a pinhole) |
| `output.channel_names` | `tags` (`CAM_<TAG>` …) or `nuscenes` (`CAM_FRONT`, `LIDAR_TOP` …); the channels in the sensor entries are already the chosen ones. The checks map streams by tag, so drives meant for them keep `tags`. |
| `output.keyframe_hz`, `keyframe_ref_hz` | nuScenes samples per second asked for; the rate of the stream they are taken from |
| `output.lidar_noise_cm` | range noise (1 σ) along each lidar ray |
| `output.drop_self_hits` | drop lidar points inside the ego's own box (5 cm margin) |
| `output.gt_range_m` | actors are annotated within this horizontal distance of the ego (60 m) |
| `output.exposure`, `exposure_ev` | **CARLA**: manual exposure (every view of a camera alike) or CARLA's histogram auto exposure; exposure compensation |

### Cameras

Each entry of `cameras` is one visible camera of the variant.

| key | meaning |
|---|---|
| `cameras[].uid`, `tag`, `safe_tag`, `name` | identity in the project; `safe_tag` is the tag as frame ids and remap keys use it |
| `cameras[].channel`, `topic`, `frame_id` | its nuScenes channel, bag topic prefix and image frame id (`<safe_tag>_optical_frame`) |
| `cameras[].pose_vehicle` | the optical centre and orientation, vehicle frame (the plan) |
| `cameras[].image.width`, `height` | the image the bridge writes, px |
| `cameras[].image.K`, `distortion_model`, `D` | the calibration the outputs state at that size (OpenCV coordinates): `equidistant` is Kannala-Brandt k1..k4 as `cv2.fisheye` uses it (ROS's name for it), `plumb_bob` a pinhole with zero coefficients. For a lens whose projection is not Kannala-Brandt these coefficients are a fit; the `lens` block is exact. |
| `cameras[].image.rectified`, `sensor_size`, `lens_projection`, `max_theta_deg` | rectified images or not; the module's sensor [w, h] px; the lens model's projection; its field, the largest angle off the axis that reaches the image |
| `cameras[].capture.rate_hz`, `recorded_hz`, `every_ticks`, `phase_ticks` | frames at the steps `phase_ticks + k · every_ticks` from the sync pulse: `recorded_hz = tick_hz / every_ticks` (the planned `rate_hz` rounded to whole steps) |
| `cameras[].capture.exposure_centre_s`, `recorded_centre_s`, `offset_error_ms` | the sync plan's exposure centre after the pulse, the step it is rounded to, and the difference |
| `cameras[].capture.triggered`, `trigger_ms` | the sync plan triggers the camera; its trigger after the pulse (null when not triggered) |
| `cameras[].lens` | the lens model to render through (below) |
| `cameras[].render_mpx`, `gt_render_mpx` | **CARLA**: megapixels its views render, and their ground-truth views |
| `cameras[].views` | **CARLA**, reusable: the pinhole views the image is remapped from (below) |
| `cameras[].atlas`, `remap` | **CARLA**, reusable: how the views are stacked and remapped (below) |

### The lens model

`cameras[].lens` is Vantage's own lens model of the camera (`core/lens.py`, with the unit's calibration when the
project has one); `core/groundtruth.py` `project()` is a numpy implementation of it. An output pixel `(i, j)` (pixel
edges) sees the ray that the model sends to the sensor point `((i + ½)·W/w, (j + ½)·H/h)`, `[W, H] = sensor_size`,
`[w, h] = image_size`. Pixels whose ray is beyond `max_theta_deg` see nothing and are black.

| key | meaning |
|---|---|
| `cameras[].lens.model` | `lens` (the real lens) or `pinhole` (rectified images, `output.undistort`) |
| `cameras[].lens.projection` | image height for a ray θ off the axis: `rectilinear` f tan θ, `equidistant` f θ, `equisolid` 2f sin(θ/2), `stereographic` 2f tan(θ/2), `orthographic` f sin θ, `kannala_brandt` f θ (1 + k1 θ² + k2 θ⁴ + k3 θ⁶ + k4 θ⁸) |
| `cameras[].lens.f_px`, `kb` | f in sensor pixels; k1..k4 |
| `cameras[].lens.cx`, `cy`, `aspect` | the principal point in the sensor's pixel-edge coordinates; fy / fx. A ray `(x, y, z)` in the optical frame with `ρ = √(x² + y²)` lands at `u = cx + x·r(θ)/ρ`, `v = cy + y·r(θ)/ρ · aspect`, then scaled by `w/W`, `h/H`. |
| `cameras[].lens.max_theta_deg`, `sensor_size`, `image_size` | the field; the sensor [W, H] and the image [w, h] |
| `cameras[].lens.K` | `pinhole` only: the 3×3 matrix at the image size, OpenCV coordinates, up to `max_theta_deg` (at most 89°) |

A simulator that renders this lens natively renders it. One that renders pinholes only does what the CARLA bridge does:

| key | meaning |
|---|---|
| `cameras[].views[].name` | `front` for a single view (every ray within 60° of the axis), else the cube faces the image needs: `front`, `right`, `left`, `up`, `down`, `back` |
| `cameras[].views[].width`, `height`, `f_px`, `fov_deg` | a pinhole of that size with focal length `f_px`, principal point at the centre (`W/2 − 0.5`, OpenCV); `fov_deg = 2 atan(W / 2f)`, CARLA's `fov` |
| `cameras[].views[].optical` | 3×3, columns the view's optical x, y, z in the camera's optical frame: a view ray `d` is the camera ray `optical · d`. All views share the camera's optical centre and are taken at the same instant. Each cube face reaches 1.5° past its seams. |
| `cameras[].views[].carla` | **CARLA**: the view's `location` (relative to the ego origin, CARLA axes) and `rotation` (`carla.Rotation`) |
| `cameras[].atlas.offsets`, `width`, `height` | the views stacked vertically into one image: view `k` at rows `offsets[k]…`, columns from 0 |
| `cameras[].remap.x`, `y` | keys in `sim_remap.npz`: float32 (h, w), where each image pixel reads in the atlas (−1 outside the field); `cv2.remap(atlas, x, y, INTER_LINEAR, BORDER_CONSTANT)` gives the image |
| `cameras[].remap.blend.idx`, `x`, `y`, `w` | keys of the seam feathering (cameras of several views): `idx` int32 (M,) flat pixel indices within `seam_blend_deg / 2` of a seam; `x`, `y` float32 (M, 2) where they read their second and third face (−1: none); `w` float32 (M, 3) the weights of the remapped value and those two, summing to 1 (`carla_bridge.feather`) |
| `cameras[].remap.seam_blend_deg` | the band the faces are feathered over (3°) |

### Lidars

| key | meaning |
|---|---|
| `lidars[].uid`, `tag`, `safe_tag`, `name`, `model` | identity; vendor and model |
| `lidars[].channel`, `topic`, `frame_id` | nuScenes channel, bag topic prefix, frame id (`<safe_tag>_link`) |
| `lidars[].pose_vehicle` | the optical centre and orientation (the plan) |
| `lidars[].rate_hz`, `steps_per_rotation` | rotations per second; steps per rotation (`tick_hz / rate_hz`) |
| `lidars[].rings` | the model's beams; ring 0 is the lowest |
| `lidars[].columns`, `azimuth_step_deg` | firings per rotation and the azimuth step (`h_fov / columns`) |
| `lidars[].range_min_m` | points nearer than this are dropped |
| `lidars[].scan`, `rotation` | `spinning`, `solid` or `nonrepetitive`; the model's direction seen from above (`clockwise`, `counter-clockwise`) |
| `lidars[].mount_turn_deg` | the turn about the sensor's z that puts CARLA's start direction (backwards) on the planned phase: for a 360° lidar the head faces azimuth `180 + mount_turn_deg` (sensor frame, rounded to the azimuth grid) at every sync pulse; 0 otherwise |
| `lidars[].blueprint.channels`, `upper_fov`, `lower_fov` | **CARLA**: channels evenly spaced from `upper_fov` (channel 0) to `lower_fov`, degrees |
| `lidars[].keep` | per blueprint channel the ring it records, −1 for a channel cast and dropped. Ring `r`'s elevation is `upper_fov − c·(upper_fov − lower_fov)/(channels − 1)` for the channel `c` with `keep[c] = r`: the model's beam exactly when `elevation_error_deg` is 0. |
| `lidars[].elevation_error_deg`, `plan_note` | the largest beam elevation error of the channel grid, and the plan in words |
| `lidars[].blueprint.range`, `horizontal_fov`, `rotation_frequency` | the model's range (m), horizontal field (°) and rate, as CARLA attributes |
| `lidars[].blueprint.points_per_second`, `noise_stddev`, `atmosphere_attenuation_rate`, `dropoff_general_rate`, `dropoff_intensity_limit`, `dropoff_zero_intensity`, `sensor_tick` | **CARLA**: `sensor.lidar.ray_cast` attributes (noise = `output.lidar_noise_cm`, no random drop-off, every step) |
| `lidars[].oversample`, `rays_per_s` | **CARLA**: the azimuth cast this many times finer and thinned back to the grid; rays cast per second |
| `lidars[].carla` | **CARLA**: the mount, turned by `mount_turn_deg` |

### Radars

| key | meaning |
|---|---|
| `radars[].uid`, `tag`, `safe_tag`, `name`, `model`, `channel`, `topic`, `frame_id`, `pose_vehicle` | as for a lidar |
| `radars[].rate_hz`, `recorded_hz` | scans per second planned, and recorded (at most one per step) |
| `radars[].range_min_m` | detections nearer than this are dropped |
| `radars[].zones[].range`, `az_half_deg`, `el_half_deg` | a detection zone: a rectangle in azimuth × elevation (degrees from the radar's axis, elevation from its horizontal plane) up to `range` m; a model may have several |
| `radars[].zones[].inside_share` | **CARLA**: the share of the cone's rays inside the rectangle |
| `radars[].zones[].blueprint.horizontal_fov`, `vertical_fov`, `range`, `points_per_second`, `sensor_tick` | **CARLA**: the `sensor.other.radar` cone that holds the zone (detections outside the rectangle are dropped) |
| `radars[].carla` | **CARLA**: the mount |

### IMUs and GNSS antennas

| key | meaning |
|---|---|
| `imus[].uid`, `tag`, `safe_tag`, `name`, `model`, `channel`, `topic`, `frame_id`, `pose_vehicle` | as for a lidar; `channel` is empty (bag only: nuScenes has no IMU sensor) |
| `imus[].rate_hz`, `recorded_hz` | samples per second planned, and recorded (at most one per step) |
| `imus[].noise.accel_sigma`, `gyro_sigma`, `accel_bias`, `gyro_bias` | white noise per sample (m/s², rad/s) and the scale of a constant bias drawn per run |
| `imus[].antennas`, `attitude_sigma_deg`, `heading_sigma_deg`, `heading_baseline_m` | a GNSS/INS (`antennas` > 0) reports its orientation with these errors; 0: an IMU alone, no orientation |
| `imus[].carla` | **CARLA**: the mount |
| `gnss[].uid`, `tag`, `safe_tag`, `name`, `model`, `channel`, `topic`, `frame_id`, `pose_vehicle` | as for an IMU |
| `gnss[].rate_hz`, `recorded_hz`, `sigma_h_m` | fixes per second (10); horizontal noise (1 σ, m; the vertical is twice it) |
| `gnss[].blueprint.noise_lat_stddev`, `noise_lon_stddev`, `noise_alt_stddev`, `sensor_tick` | **CARLA**: the same noise in degrees and metres |
| `gnss[].carla` | **CARLA**: the mount |

### Faults

A fault changes what the simulator records; the calibration the outputs state (nuScenes `calibrated_sensor`, `/tf_static`,
`camera_info`, `dataset.json`) stays the plan's, as on a real faulty rig. That is what makes it a known answer.
`carla_bridge.FAULT_KINDS` lists the kinds and the sensors they apply to; `clean_faults` checks and clamps them.

| key | meaning |
|---|---|
| `faults[].tag`, `kind` | the sensor; `rotation`, `translation`, `clock_offset`, `drop`, `freeze`, `range_bias` |
| `faults[].deg` | `rotation`: [yaw, pitch, roll] degrees in the sensor's own frame: `R_actual = R_plan · rot_zyx(deg)` (±30°) |
| `faults[].m` | `translation`: [x, y, z] m in the vehicle frame, added to the position (±2 m); `range_bias`: m added to every lidar or radar range (±5 m) |
| `faults[].ms` | `clock_offset`: data captured at t is stamped t + ms (nuScenes and bag; its nuScenes ego pose is the one at the stamp) |
| `faults[].rate`, `frames` | `drop`: a share of the frames (0–0.9: planned frame `k` is dropped when `numpy.random.default_rng([seed, 4241] + [ord(c) for c in tag]).random(n)[k] < rate`), or the planned frame indices; dropped frames are never written |
| `faults[].from_s`, `to_s` | `freeze`: frames stamped in [from, to) repeat the last frame written before (the first holds itself when none came before) |

### The actors scene

| key | meaning |
|---|---|
| `actors.items[].id`, `kind` | a unique id (a critical target's uid when Vantage made the scene); `child`, `pedestrian`, `cyclist`, `car`, `truck`, `traffic_light`, `sign`, `cone`, `barrier` or **CARLA** `blueprint:<id>` |
| `actors.items[].pose.x`, `y`, `z`, `yaw` | vehicle frame: the actor's box bottom centre (x, y its ground-contact centre, z its lowest point, m) and heading (degrees from +x, counter-clockwise) |
| `actors.spawn_point` | **CARLA**: the ego's spawn point; −1: a straight road chosen by the seed |
| `actors.frame` | the frame, in words |

With actors the ego stands still on a straight road with them around it and no traffic; a scenario is then not run.

### Ground-truth options

| key | meaning |
|---|---|
| `ground_truth.semantic`, `instance`, `range`, `lidar_labels` | what to record: semantic and instance images and the objects per camera frame, range images, nuScenes-lidarseg and -panoptic labels per lidar sweep |
| `ground_truth.lidar_sensors` | lidars labelled |
| `ground_truth.camera_kinds`, `render_mpx`, `gt_render_mpx`, `note` | **CARLA**: the label cameras rendered beside every view, the cost, how the labels are made |

### nuScenes rows and static transforms

| key | meaning |
|---|---|
| `nuscenes.sensor[].token`, `channel`, `modality` | the `sensor.json` rows, one per camera, lidar and radar |
| `nuscenes.calibrated_sensor[].token`, `sensor_token`, `translation`, `rotation` | the `calibrated_sensor.json` rows: `p_ego = R(rotation) · p_sensor + translation`, rotation `[w, x, y, z]`; a camera's sensor frame is its optical frame |
| `nuscenes.calibrated_sensor[].camera_intrinsic` | 3×3 at the image size (cameras); empty for lidars and radars |
| `nuscenes.calibrated_sensor[].vantage_distortion.model`, `coefficients`, `image_size`, `image_scale`, `camera`, `name`, `source` | cameras: `kannala_brandt` (k1..k4 for `cv2.fisheye`), `none (pinhole)` or `none (rectified pinhole images)`; the image size and its scale from the sensor; the camera's tag and place; `library_lens` or `unit_calibration` |
| `nuscenes.calibrated_sensor[].vantage_distortion.rectified_from.model`, `coefficients` | with rectified images: the lens they were rectified from |
| `nuscenes.calibrated_sensor[].vantage_sensor.tag`, `model`, `kind` | lidars and radars |
| `nuscenes.categories`, `attributes`, `visibility` | `[name, description]` of the nuScenes categories and attributes the drive may use; `[token, level, description]` of the four visibility bins |
| `tf_static[].parent`, `child`, `matrix` | `/tf_static`: 4×4 parent ← child, `base_link` → `<safe_tag>_link` (each sensor body) and `<safe_tag>_link` → `<safe_tag>_optical_frame` (each camera) |

A bridge copies these rows as they are for the channels it writes: they are the plan, and a fault never changes them.

## What a bridge writes

Into the output folder `OUT` (`record.py` writes it this way; `check_out_dir` refuses a `nuscenes/` it did not write):

```
OUT/
  rig/                          the rig folder (Vantage puts it here when it starts the drive)
  nuscenes/
    .vantage-dataset            marker: a later run may clear samples/, sweeps/, maps/, v1.0-vantage/ here
    v1.0-vantage/*.json         the tables
    samples/<CHANNEL>/…         keyframe files; sweeps/<CHANNEL>/… the others
    maps/<token>.png            the map mask (0.1 m per pixel)
    lidarseg/v1.0-vantage/<token>_lidarseg.bin, panoptic/v1.0-vantage/<token>_panoptic.npz
  rosbag/<bag_name>/            rosbag2, MCAP storage, CDR messages, metadata.yaml
  ground_truth/index.json, ground_truth/<CHANNEL>/…png, ground_truth/coco/<CHANNEL>.json
  dataset.json
  README.md
  run.json
```

A file is named `<log>__<CHANNEL>__<timestamp µs>.<ext>` with `<log>` = `vantage-YYYY-MM-DD-HH-MM-SS+0000` of the epoch.
`nuscenes/` and `rosbag/` follow `output.nuscenes` and `output.rosbag`. On a failure or a cancel the bridge removes
what it wrote and leaves only `run.json` saying why.

### Data files

| file | contents |
|---|---|
| camera `.jpg` | the image at `cameras[].image` size through the `lens` model, JPEG at `output.jpeg_quality`. One instant per frame (global shutter, no motion blur). |
| lidar `.pcd.bin` | float32 × 5 per point: x, y, z (m, the lidar's own frame), intensity, ring (ring 0 the lowest beam), in firing order (column by column). Each point is placed where the sensor was when it fired. Points nearer than `range_min_m` and, with `drop_self_hits`, inside the ego's box are left out. **CARLA**: intensity = 255 × CARLA's atmospheric attenuation, not a reflectivity. |
| radar `.pcd` | nuScenes' 18-field binary PCD: x, y, z (radar frame), dyn_prop (1 standing: a compensated speed under 0.3 m/s, else 0), id, rcs (0: the simulator gives none), vx, vy (the radial velocity along the ray, relative to the radar, positive away, as a vector in the radar's x-y), vx_comp, vy_comp (with the radar's own motion removed), and the quality fields. An empty scan is one NaN point, as nuScenes writes it. The steps of a scan are merged. |

### nuScenes: `nuscenes/v1.0-vantage/`

The 13 tables of nuScenes v1.0, plus `lidarseg` and `panoptic` with lidar labels. Tokens are 32-hex md5 hashes of the
parts of the drive (`record.token`): the same seed and settings give the same tokens. Vantage's checks read only the
rows marked **read**; the others are for the nuScenes devkit.

| key | meaning |
|---|---|
| `sensor[].token`, `channel`, `modality` | **read**: the rig's rows for the channels written |
| `calibrated_sensor[].token`, `sensor_token` | **read** (drive-log, as-built, campaigns: the tokens only) |
| `calibrated_sensor[].translation`, `rotation`, `camera_intrinsic`, `vantage_distortion` | **read** by detection quality (pinhole views) and the ground-truth viewer (lidar points into the vehicle frame): the rig's rows, the plan |
| `sample_data[].token`, `calibrated_sensor_token`, `timestamp`, `filename` | **read**: one row per file; `timestamp` µs, `filename` relative to `nuscenes/` |
| `sample_data[].width`, `height` | **read**: the image size (0 for lidar and radar) |
| `sample_data[].ego_pose_token`, `sample_token`, `fileformat`, `is_key_frame`, `prev`, `next` | the ego pose row (same token as the sample_data), the sample it belongs to (its keyframe, else the next sample), `jpg` / `pcd`, and the channel's chain |
| `ego_pose[].token`, `timestamp`, `translation`, `rotation` | **read**: global ← ego at the sample_data's stamp (a clock-offset fault: at the shifted stamp), rotation `[w, x, y, z]` |
| `sample[].token`, `timestamp`, `prev`, `next`, `scene_token` | the keyframes |
| `scene[].token`, `log_token`, `nbr_samples`, `first_sample_token`, `last_sample_token`, `name`, `description` | one scene per drive |
| `log[].token`, `logfile`, `vehicle`, `date_captured`, `location` | one log |
| `map[].token`, `log_tokens`, `category`, `filename` | the map mask |
| `sample_annotation[].token`, `sample_token`, `instance_token`, `visibility_token`, `attribute_tokens`, `translation`, `size`, `rotation`, `prev`, `next`, `num_lidar_pts`, `num_radar_pts` | every actor within `gt_range_m` at each keyframe: box centre (global), size [w, l, h], yaw quaternion, visibility bin from camera rays to the box's faces, points of the keyframe sweep and scan inside the box |
| `instance[].token`, `category_token`, `nbr_annotations`, `first_annotation_token`, `last_annotation_token` | one per annotated actor |
| `category[].token`, `name`, `description`, `index` | the 32 nuScenes-lidarseg categories with their `index`, then any other the drive uses |
| `attribute[].token`, `name`, `description` | the rig's attributes |
| `visibility[].token`, `level`, `description` | the four visibility bins |
| `lidarseg[].token`, `sample_data_token`, `filename` | lidar labels: uint8 nuScenes-lidarseg index per point of the sweep, in its `.pcd.bin` order |
| `panoptic[].token`, `sample_data_token`, `filename` | `.npz` with `data`: uint16 category index × 1000 + instance id per point |

The global frame is right-handed, z up, metres, fixed to the world. **CARLA**: CARLA's world with y mirrored, shifted
by `run.json` `scene.global_offset_m` so the map mask starts at its bottom-left corner.

### The bag: `rosbag/<bag_name>/`

A rosbag2 folder (`core/rosbag.py` writes MCAP; Vantage reads MCAP and SQLite `.db3`). Every message's log time equals
its header stamp. `<topic>` is the sensor's `topic`.

| topic | type | contents |
|---|---|---|
| `/tf_static` | `tf2_msgs/msg/TFMessage` | once at `t = 0`: the rig's `tf_static` |
| `/tf` | `tf2_msgs/msg/TFMessage` | 50 Hz: `odom` → `base_link`, the ego pose in the global frame |
| `/odom` | `nav_msgs/msg/Odometry` | 50 Hz: frame `odom`, child `base_link`; the pose and the body's velocity and angular rate in `base_link` |
| `<topic>/image_raw/compressed` | `sensor_msgs/msg/CompressedImage` | the camera's JPEG, frame `<safe_tag>_optical_frame` |
| `<topic>/camera_info` | `sensor_msgs/msg/CameraInfo` | `cameras[].image`: width, height, distortion_model, D, K |
| `<topic>/points` | `sensor_msgs/msg/PointCloud2` | a lidar sweep stamped at its end, frame `<safe_tag>_link`: x, y, z, intensity (float32), ring (uint16), t (float32, seconds from the sweep's start, the firing time) |
| `<topic>/detections` | `sensor_msgs/msg/PointCloud2` | a radar scan, frame `<safe_tag>_link`: x, y, z, v_r (radial velocity relative to the radar, positive away), rcs (float32) |
| `<topic>/imu` | `sensor_msgs/msg/Imu` | frame `<safe_tag>_link`: angular velocity (rad/s) and specific force (m/s², gravity included: +9.81 up at rest) in the IMU's frame, noise covariances; a GNSS/INS also its orientation, else `orientation_covariance[0] = −1` (REP 145) |
| `<topic>/fix` | `sensor_msgs/msg/NavSatFix` | frame `<safe_tag>_link`: WGS84 latitude, longitude (°), altitude (m) of the antenna, status 2, diagonal covariance (σh², σh², (2σh)²), type 2 |
| `/ground_truth/markers` | `visualization_msgs/msg/MarkerArray` | the actors' boxes (frame `odom`) at each sweep of the first lidar |
| `/ground_truth/boxes` | `std_msgs/msg/String` | the same as JSON: `stamp_ns`, `frame_id`, `boxes` (`id`, `instance_token`, `category`, `name`, `centre`, `size_lwh`, `yaw`, `velocity`; `sumo_id` for a vehicle SUMO drives) |

### `dataset.json`

The drive's description in the sample dataset's schema (`vantage.dataset/1`). The drive-log check looks for it in the
recording's folder and up to three folders above it: with it, cameras are judged at the rate and size they were
written with; without it, at the module's frame rate and full sensor size (a reduced `image_scale` then shows as a
finding).

| key | meaning |
|---|---|
| `schema`, `generator` | `vantage.dataset/1` (**read**); `carla` for a CARLA drive |
| `project`, `variant` | **read** by the drive-log check and detection quality |
| `settings.camera_rate`, `image_scale`, `scene`, `duration_s` | **read** by the drive-log check (`output.camera_rate`, `output.image_scale`; `scene` names the source, `carla`) |
| `settings.seed`, `channel_names`, `keyframe_hz`, `jpeg_quality`, `nuscenes`, `rosbag`, `lidar_noise_cm`, `drop_self_hits`, `undistort` | the rest of the output settings |
| `written.nuscenes`, `nuscenes_version`, `rosbag` | the folders written, relative to `OUT` (null when not written) |
| `scene` | `run.json` `scene` |
| `counts`, `bytes`, `channels`, `faults_injected` | as in `run.json` |
| `bag_topics` | messages per topic |
| `ground_truth.index`, `schema`, `cameras`, `lidars` | the index's path, its schema and the tags it covers (null without ground truth) |
| `conventions.global_frame`, `ego_frame`, `time`, `camera_timestamp`, `lidar_timestamp`, `lidar_fields`, `radar_fields`, `images`, `bag_frames` | the conventions of this page, in words |
| `assumptions`, `notes` | the rig's assumptions; the bridge's notes |
| `carla` | **CARLA**: the run report as in `run.json` |

### Ground truth: `ground_truth/index.json`

Schema `vantage.ground_truth/1` (`core/groundtruth.py`, whose module text describes the files in full). Paths in it are
relative to `OUT`.

| key | meaning |
|---|---|
| `schema`, `project`, `variant` | `vantage.ground_truth/1` and the rig's names |
| `classes[].id`, `name`, `nuscenes`, `nuscenes_index`, `color`, `thing` | the class table: ids 0–28 are CARLA 0.9.16's semantic tags (`carla.CityObjectLabel`), 29 is `ego` (the vehicle carrying the rig); each class's nuScenes-lidarseg category, palette colour and whether it has instances |
| `classes[].carla_tags`, `carla_label` | **CARLA**: the tags of a class. A bridge for another simulator maps its own labels onto these classes. |
| `cameras.{tag}.channel`, `width`, `height` | per camera |
| `cameras.{tag}.frames[].sample_data_token`, `timestamp`, `image` | the frame's image (its sample_data and path) |
| `cameras.{tag}.frames[].semantic`, `instance`, `range` | label images, null when not recorded: uint8 class id per pixel; uint16 object id (0 none); uint16 = round(range m · 256) along the pixel's own ray from the optical centre (0 nothing within 256 m). Same size as the image, pixel for pixel; labels never interpolated. |
| `cameras.{tag}.frames[].bag.topic`, `stamp_ns` | the image's bag message, with a bag |
| `cameras.{tag}.frames[].objects[].id`, `actor_id`, `instance_token`, `category`, `class` | an object: its id in the instance image (stable over the drive), the simulator's actor id, the nuScenes instance (null unless annotated at a keyframe), nuScenes category, class name |
| `cameras.{tag}.frames[].objects[].bbox`, `pixels`, `bbox_full`, `truncated`, `occluded`, `range_m` | tight box of its visible pixels `[x0, y0, x1, y1]` (pixel edges), their count, the 3D box's projected extent (null outside the field), the shares outside the image and hidden, range from the optical centre to the box centre |
| `cameras.{tag}.frames[].objects[].box3d.center`, `size`, `yaw_deg` | the 3D box in the vehicle frame at the frame's time: centre, [l, w, h], heading |
| `lidars.{tag}.channel` | per labelled lidar |
| `lidars.{tag}.sweeps[].sample_data_token`, `timestamp`, `is_key_frame`, `points`, `lidarseg`, `panoptic` | per labelled sweep: its sample_data, point count and label files |
| `nuscenes_lidarseg_categories[].index`, `name` | the lidarseg categories |
| `conventions`, `notes` | the conventions in words; how the labels were made |

A frame of a camera with a freeze fault carries `frozen: true` (the image repeats an earlier one; the labels are the
true scene). With a rotation or translation fault the labels match the image (the actual mount).
`ground_truth/coco/<CHANNEL>.json` holds the same objects as COCO detection files (bbox `[x, y, w, h]`, compressed RLE masks).

### The run report: `run.json`

Schema `vantage.sim_run/1`. Vantage takes a folder with a `run.json` of this schema for a simulated drive
(`core.simrig.read_run`); with `status` `ok` the drive is complete. [`sim_run.schema.json`](sim_run.schema.json) says
which fields an `ok` report must have; fields marked **CARLA** may be left out by another bridge.

| key | meaning |
|---|---|
| `schema`, `status` | `vantage.sim_run/1`; `ok`, `failed` or `cancelled` |
| `error`, `traceback` | a failure: `<Exception>: <message>` and the traceback |
| `exit_code` | written by Vantage, not the bridge, when a bridge it started ended without a report: its exit code (negative for a signal) |
| `project`, `variant`, `out_dir` | the rig's names; the output folder as the bridge saw it |
| `town`, `traffic`, `walkers` | the map; the vehicles and pedestrians spawned |
| `weather.preset`, `vantage`, `params` | the weather applied: a preset by name, or a condition's label with its parameters |
| `client_version`, `server_version` | **CARLA**: the client's and server's versions |
| `vehicle.blueprint`, `bounding_box_lwh_m`, `rig_choice`, `project` | the simulated ego: what it is, its box [l, w, h] m, what the rig asked for, the project body's sizes |
| `vehicle.ego_origin_in_actor_m` | **CARLA**: where the vehicle frame's origin lies in the actor's frame (CARLA axes), measured from its wheels and box |
| `mounts[].tag`, `kind`, `inside_bounding_box`, `inside_body_m` | **CARLA**: whether a sensor's mount lies inside the simulated vehicle's box and how deep inside its body (its body is not the project's: a camera inside it sees the inside of the car) |
| `start.spawn_point`, `straight_ahead_m`, `straight_behind_m`, `road_id`, `lane_id`, `chosen` | a scenario's or actors scene's start on a straight lane |
| `scene.town`, `weather`, `duration_s`, `seed`, `traffic`, `vehicle`, `tick_hz`, `warmup_ticks`, `keyframe_hz`, `samples` | the drive (also `dataset.json` `scene`); detection quality pairs recordings with the same scenario, seed, town and duration |
| `scene.global_offset_m` | the shift from the simulator's world to the global frame |
| `scene.carla_version`, `ego_origin_in_actor_m` | **CARLA** |
| `cameras.{tag}.expected`, `got`, `unexpected`, `dropped` | per camera: frames due, written, off the schedule, dropped by a fault |
| `lidars[].tag`, `carla_channels`, `rings`, `azimuth_step_deg`, `oversample`, `first_azimuth_of_rotation_deg`, `note` | **CARLA**: per lidar, its channel plan and the first laser's azimuth from the start angle (0: on the grid) |
| `channels[].kind`, `tag`, `channel`, `topic`, `frames`, `keyframes`, `bytes`, `planned_frames` | every stream: frames written of those planned |
| `counts.samples`, `sample_data`, `ego_poses`, `annotations`, `instances`, `bag_messages`, `gt_frames`, `gt_objects`, `lidarseg` | what was written |
| `bytes.nuscenes`, `rosbag`, `ground_truth`, `total` | bytes written |
| `performance.simulated_s` | seconds of drive recorded (campaigns read it) |
| `performance.wall_s`, `recording_wall_s`, `steps`, `steps_per_wall_s`, `realtime_factor`, `tick_s`, `wait_s`, `camera_s`, `lidar_s`, `ground_truth_s` | how long it took |
| `notes` | the bridge's notes, then the rig's |

The known answers and the scene:

| key | meaning |
|---|---|
| `faults_injected[].tag`, `kind`, `deg`, `m`, `ms`, `rate`, `frames`, `from_s`, `to_s` | each fault as the rig gives it (empty list: a clean drive) |
| `faults_injected[].sensor_kind`, `plan_pose`, `actual_pose`, `pose_delta_deg`, `position_delta_m`, `angle_deg`, `frame`, `calibration` | a mount fault: the planned and actual pose (vehicle frame), actual − plan in yaw, pitch, roll (wrapped) and x, y, z, the rotation's angle, the frame its values are in |
| `faults_injected[].channel`, `planned_frames`, `applies_to` | a delivery fault: its stream and planned frames |
| `faults_injected[].dropped_frames`, `dropped_stamps_s` | `drop`: the planned frame indices never written and their times |
| `faults_injected[].frozen_frames`, `held_frame`, `ground_truth` | `freeze`: the frame indices that repeat and the one they repeat |
| `scenario.name`, `label`, `about`, `params` | the scenario played (`actors` for an actors scene), every parameter with its default filled in |
| `scenario.start.spawn_point`, `straight_ahead_m`, `straight_behind_m`, `road_id`, `lane_id`, `chosen` | as `start` (null for a free drive) |
| `scenario.actors[].role`, `actor_id`, `blueprint`, `gt_id` | the scenario's actors and their object ids in the ground truth |
| `scenario.events[].t_s`, `what` | what happened when (`lead_brake`, `cut_in_complete` …, with the figures of the moment) |
| `scenario.geometry`, `measures`, `notes` | distances set up; figures measured over the drive (`min_gap_m`, `min_distance_m`) |
| `actors[].id`, `kind`, `requested`, `actor_id`, `blueprint`, `error` | each actor of an actors scene as asked for, and spawned (`actor_id` = the ground truth's `objects[].actor_id`) or why not |
| `actors[].requested.x`, `y`, `z`, `yaw` | the pose asked for (`actors.items[].pose` of the rig) |
| `actors[].pose_vehicle.x`, `y`, `z`, `yaw` | as placed, measured after settling: box bottom centre and heading, vehicle frame |
| `actors[].ground_gap_m`, `error_m`, `error_z_m`, `error_yaw_deg` | the place's height over the ground under it; the placing error (horizontal, vertical, heading) |
| `actors[].bbox.size_lwh`, `center_in_actor` | the actor's box [l, w, h] m and its centre in the actor's frame |
| `ground_truth.settings.semantic`, `instance`, `range`, `lidar_labels` | what was recorded |
| `ground_truth.index`, `seconds` | `ground_truth/index.json` (null without ground truth); the time spent on labels |
| `ground_truth.camera_kinds` | **CARLA**: the label cameras |
| `ground_truth.lidar_labels.{tag}.slices`, `points`, `matched_in_order_share`, `fallback_slices`, `count_mismatch_points`, `max_direction_error_deg`, `range_rms_diff_m`, `note` | how the labelling lidar's rays matched the recorded lidar's (**CARLA**: a semantic lidar beside each lidar) |
| `ground_truth.lights_and_signs.{kind}.pixels`, `carla_tags`, `labelled_before`, `labelled_after`, `note` | **CARLA**: traffic-light and sign pixels claimed from their boxes |

SUMO traffic (**CARLA**; only when the rig has `traffic_source`). SUMO's vehicles are CARLA actors the bridge spawned,
so they have boxes, instance ids, lidar labels and nuScenes annotations like any other; `traffic_source.actors` names
each by its SUMO id and type (the ground truth's `objects[].actor_id` is its `actor_id`; the bag's
`/ground_truth/boxes` carry its `sumo_id`). Their velocity in the annotations and boxes is SUMO's speed along their
heading (CARLA reports none for an actor it does not simulate).

| key | meaning |
|---|---|
| `traffic_source.source`, `sumo_version`, `package_version`, `traci_api`, `binary` | `sumo`; SUMO's version (TraCI), the eclipse-sumo package's, the TraCI API level, the binary run |
| `traffic_source.net.kind`, `town`, `file`, `sha256`, `seconds` | the net run (`bundled`, `generated`, `user`, `user_config`), its file name and SHA-256, the seconds to get it |
| `traffic_source.net.cached`, `opendrive_sha256` | a generated net: taken from the cache or made now; the OpenDRIVE's SHA-256 (the cache key) |
| `traffic_source.demand.mode`, `vehicles`, `period_s`, `trips`, `classes`, `vtypes` | the demand: `count`, `period` or `user` (a `.sumocfg`'s own); vehicles at once; the trips' period and number; the classes and vTypes drawn |
| `traffic_source.demand.routes_file`, `routes_sha256`, `generator`, `seconds` | the trips written (`OUT/sumo/demand.rou.xml`, kept with `run.sumocfg` and SUMO's log) and how |
| `traffic_source.demand.config`, `config_sha256` | a user `.sumocfg`: its name and SHA-256 |
| `traffic_source.seed`, `step_s`, `steps_per_sumo_step`, `max_vehicles` | SUMO's `--seed`; its step and how many simulation steps it spans (SUMO's clock is whole milliseconds: 2 at 80 Hz; between SUMO's steps its vehicles move on along their heading at its speed); its `--max-num-vehicles` |
| `traffic_source.tls.owner`, `linked`, `frozen_in_carla` | who ran the lights, how many of CARLA's lights the net links, whether CARLA's were frozen for SUMO to set |
| `traffic_source.vehicles_spawned`, `arrived`, `not_spawned` | SUMO vehicles spawned in CARLA, those that ended their trip (destroyed), those CARLA could not take |
| `traffic_source.vehicles_in_carla.at_start`, `peak`, `mean`, `note` | SUMO vehicles in CARLA from `t = 0` to the end |
| `traffic_source.teleports`, `collisions` | what SUMO reported: vehicles it teleported (a jam it gave up on), collisions |
| `traffic_source.mirrored[].actor_id`, `sumo_id`, `blueprint`, `role`, `sumo_type` | the bridge's own CARLA vehicles (the ego, a scenario's) mirrored into SUMO |
| `traffic_source.actors[].sumo_id`, `sumo_type`, `vclass`, `actor_id`, `blueprint`, `spawned_t_s`, `arrived_t_s` | every vehicle SUMO inserted: its id, vType and class, the CARLA actor and blueprint it became (null when not spawned), when (SUMO's time) |
| `traffic_source.timing.net_s`, `demand_s`, `launch_s`, `step_s`, `sync_s` | seconds getting the net, writing the demand, starting SUMO; stepping SUMO and moving vehicles both ways over the run |
| `traffic_source.walkers`, `limits` | the co-simulation's limits in words |

## Timing and synchronisation

The CARLA bridge runs the world in synchronous mode at `simulation.tick_hz`: every sensor samples the same world state
at a step, and every stamp is a whole number of steps from the sync pulse. A bridge for another simulator keeps the
same schedule; the checks measure the drive against it.

1. Before `t = 0`: `scenario.warmup_s` of traffic, then `simulation.warmup_ticks` steps with the sensors attached, so
   that every 360° lidar is back at its start direction at `t = 0` (**CARLA**: the radar's and IMU's first steps are
   garbage, hence at least 3).
2. The drive is `N = round(duration_s · tick_hz)` steps.
3. Camera frame `k` is taken at step `phase_ticks + k · every_ticks` (`≤ N`) and stamped there: the sync plan's
   exposure centre, rounded to a step (`offset_error_ms`). The drive-log check reads a camera stamp as the middle of the
   exposure (its default `camera_stamp`).
4. Lidar sweep `k` covers one rotation from `k / rate_hz` and is stamped at its end, `(k + 1) · steps_per_rotation /
   tick_hz`; there are `N // steps_per_rotation` sweeps. At every sync pulse (`t` a multiple of the period) the head
   faces azimuth `180 + mount_turn_deg`, the planned `phase_deg`. A point's bag `t` is its firing time after the
   sweep's start, and the point is where the sensor was at that time.
5. Radars, IMUs and GNSS antennas sample at every `tick_hz / rate_hz`-th step when that is whole, else at the step
   nearest each `k / rate_hz` (`carla_bridge.schedule`: the rate exact on average), at every step when they are faster
   than the step. A radar scan merges the steps since the last scan and is stamped at its last.
6. `/tf` and `/odom` every `round(tick_hz / 50)` steps; ground-truth boxes at each sweep of the first lidar (else at
   10 Hz).
7. Keyframes (nuScenes samples): every `n`-th sweep of the first lidar, `n = round(rate / min(keyframe_hz, slowest
   stream, rate))` (else of the first camera); each channel's keyframe is its written frame nearest the sample's time.
8. Each `sample_data` has an `ego_pose` at its own stamp.
9. **CARLA**, SUMO traffic: from the moment the traffic would be spawned, SUMO steps before the synchronous step, its
   vehicles are moved in CARLA before the step and the bridge's own vehicles are mirrored into SUMO after it
   (`run_synchronization.py`'s order), so every sensor sees SUMO's state of that step. SUMO's clock is whole
   milliseconds: when `dt_s` is one (20 Hz: 50 ms) SUMO steps with every step; else every `k` steps, the fewest whose
   time is whole milliseconds (80 Hz: `k` = 2, a 25 ms SUMO step), and on the steps between SUMO's vehicles move on
   along their heading at SUMO's speed (`traffic_source.step_s`, `steps_per_sumo_step`).

What the drive-log check measures on this: each stream's rate, drops and frozen frames from the stamps and contents;
the lidar's phase at the sync pulses (multiples of its period on the recording's clock, so `epoch_us` stays a whole
second) against the plan's; each camera's exposure centre against the moment the lidar sweeps its view, against the
sync plan; each camera's rotation against the lidar. A stamp off this schedule is what a `clock_offset` fault is, and
the check reports it as one.

## What each check needs

| check | run it | needs | uses when present |
|---|---|---|---|
| Drive-log check (`core/drivelog.py`) | `vantage project.vantage --variant V --drive-log OUT/nuscenes report/` or `… --drive-log OUT/rosbag/<bag_name> report/` | nuScenes: `sensor.json`, `calibrated_sensor.json`, `sample_data.json`, `ego_pose.json` and the files. Or the bag: camera, lidar and radar topics named by tag, with header stamps and frame ids. Lidar points x, y, z in the lidar's frame. | `dataset.json` (cameras judged as written); a lidar time field `t` and `ring`; `/odom` or ego poses (motion compensation); IMU and NavSatFix topics |
| As-built check (`core/asbuilt.py`) | `--as-built-check OUT/rosbag/<bag_name> report/` (or `OUT/nuscenes`) | camera images with motion between them (odometry, ego poses or fixes, else image change) | a lidar (its second estimate, through the drive-log check). It compares each camera's view of the vehicle with the project's body: a simulator that renders another car makes the outline comparison not checkable. |
| Ground-truth summary | `vantage --ground-truth-summary OUT [gt.json]` (no project) | `ground_truth/index.json` | `run.json` `status`, `scene`, `ground_truth`, `scenario`, `faults_injected`, `actors` |
| Detection quality (`core/detection_eval.py`) | `vantage project.vantage --detection-quality det/ --detection-recording OUT` | `ground_truth/index.json` with camera frames: `image`, `objects` (`bbox`, `class`, `pixels`, `occluded`, `truncated`, `range_m`, `box3d`, `actor_id` or `instance_token`) | `range` (ranges the false detections); `instance` and `nuscenes/v1.0-vantage/` `sensor` and `calibrated_sensor` (pinhole views); `rig/sim_rig.json` `cameras[].image.max_theta_deg`; `run.json` `scene` and `scenario.name` (paired comparisons); `dataset.json` `project`, `variant` |
| Campaigns (`core/simcampaign.py`) | `vantage project.vantage --campaign out/` | per drive: `run.json` `status`, `performance.simulated_s`, `faults_injected` (the known answers); the nuScenes tables and files; ground truth for the detection evaluator and the rig-health reference drive (class `ego` pixels) | — |
| Critical targets in CARLA (`core/critical_carla.py`) | `vantage project.vantage --critical-carla out/`, or `--critical-carla out/ --critical-carla-drives DIR` for drives recorded before | an actors scene with ground truth: `rig/sim_rig.json` `cameras[]` (`tag`, `uid`, `lens`, `pose_vehicle`); `run.json` `actors[]` (`id`, `actor_id`, `pose_vehicle`, `bbox.size_lwh`); the instance, semantic and range images and `objects[].actor_id` | `run.json` `mounts[].inside_body_m` |
| Open run… (Simulate ▸ CARLA) | the app | `run.json` (`schema`, `status`) | the ground truth (its viewer); `rosbag/<first folder>` else `nuscenes/` for the drive-log and as-built buttons |

Campaigns and critical targets start the bridge themselves. The CARLA bridge's command line is
`python -m vantage.sim.carla_bridge RIG OUT --progress-json` with `--port`, `--tm-port`, `--timeout`, `--seed`,
`--duration`, `--traffic`, `--scenario`, `--scenario-params JSON`, `--faults FILE|none`, `--ground-truth LIST|none` and
`--actors FILE|none` over the rig's own values (and `--traffic-source carla|sumo`, `--sumo-net auto|generate|FILE`,
`--sumo-cache DIR`, `--carla-root DIR` for SUMO traffic); progress is JSON lines `{"progress": f, "text": …}` on stdout; the exit
code is 0 when done, 2 when it failed, 3 when cancelled; SIGINT or SIGTERM cancels it (it tidies the simulator up and
removes what it wrote). Another simulator's bridge plugs into them through Python: `core.simcampaign.run(…,
runner=fn)` with `fn(rig_json, out_dir, args, progress, cancel)` returning an exit code (`args` is that command line's
options), and `core.critical_carla.run(…, runner=fn)` with `fn(rig_json, out_dir, progress, cancel)`; with a runner,
neither looks for a CARLA server.

## Validating a bridge

On a drive of a variant whose rig file the bridge was given:

1. The report: `jsonschema.validate(json.load(open('OUT/run.json')), json.load(open('docs/sim_run.schema.json')))`.
2. `vantage --ground-truth-summary OUT` (with ground truth): exit 0, frames, objects and sweeps per camera, lidar and
   class.
3. `vantage project.vantage --variant V --drive-log OUT/nuscenes dl-nus/` and the same on `OUT/rosbag/<bag_name>`:
   every visible sensor mapped to its stream and healthy, the finding "A sample dataset Vantage wrote …", the lidar's
   phase error well under a degree, every camera's rotation against the lidar within the tolerance; exit 0 on a clean
   drive. A drive with a fault should show that fault (exit 3).
4. `vantage project.vantage --variant V --as-built-check OUT/rosbag/<bag_name> ab/`.
5. With ground truth: `vantage project.vantage --detection-quality det/ --detection-recording OUT`.
6. In the app: Simulate ▸ CARLA ▸ Open run… on `OUT`.

The reference is Vantage's own bridge against `tests/fake_carla` (a `carla` module with CARLA 0.9.16's frames, sensor
conventions and ground truth over a flat world). `tests/test_sim.py` runs it (`fake_run`, `_mini_run`) and checks the
nuScenes tables' consistency, the lidar points against the images, the drive-log and as-built checks on the recording
(`test_recording_reads_back_in_the_drive_log_and_as_built_checks`), the ground truth (`test_ground_truth_index_…`,
`test_lidarseg_and_panoptic_in_the_official_layout`), faults, scenarios and actors scenes; `tests/test_sim_contract.py`
validates its documents against the schemas. `tests/test_sumo.py` runs the bridge with SUMO traffic against
`tests/fake_sumo` (a fake `traci`, `sumolib` and `sumo` package, a fake SUMO process, and a stand-in for CARLA's
`sumo_integration` package; CARLA's real package too when a CARLA folder is found).

## Versioning

Each document carries its schema: `vantage.sim_rig/1` (the rig file), `vantage.sim_run/1` (the report),
`vantage.dataset/1` (`dataset.json`), `vantage.ground_truth/1` (the index); the nuScenes folder is the version
`v1.0-vantage`. Vantage compares the schema string exactly: the CARLA bridge refuses a rig file of another one
(`load_rig`), and a `run.json`, `dataset.json` or index of another one is not taken for a drive, a Vantage dataset or
ground truth. Everything else is read with defaults, so fields are added within a version and an older reader ignores
them; a change an older reader would misread needs a new version string, which an older Vantage then refuses rather
than misreads. The schemas here describe version 1 as this Vantage writes it and are strict (an unknown key is an
error), so that they document every field.
