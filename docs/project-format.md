# The `.vantage` project format

A Vantage project is one file with the extension `.vantage`. This page describes what is in it, the units and frames
of every number, and how the format is versioned. [`project.schema.json`](project.schema.json) is the same description
as a JSON Schema (draft 2020-12); `tests/test_provenance.py` validates the sample project and every template against
it, so the schema and this page follow the code.

## The file

A `.vantage` file is a ZIP archive with up to three members:

| member | what it holds |
|---|---|
| `project.json` | everything but the vehicle's triangles: the vehicle's description, the component library, the variants with their cameras and sensors, the analysis settings, boards, obstacles, the rig, the ECU, reference photos, live-stream sources, the fleet and a short history. UTF-8 JSON. |
| `mesh.npz` | the vehicle body as NumPy arrays (`numpy.savez_compressed`): `vertices` float32 (V, 3), metres in the vehicle frame; `faces` int32 (F, 3), vertex indices; `face_part` int32 (F,), index into `vehicle.parts`; `face_zone` int16 (F,), index into `vehicle.zones` (−1: no zone). Absent when the project has no vehicle body. |
| `thumb.png` | a picture of the vehicle and rig for the start window. Optional. |

Vantage writes the archive to a temporary file next to the target and renames it into place, so a crash never leaves
half a project. A reader should refuse files over 2 GB and must open `mesh.npz` with `allow_pickle=False`.

Reading a project without Vantage:

```python
import io, json, zipfile
import numpy as np

with zipfile.ZipFile('rig.vantage') as z:
    project = json.loads(z.read('project.json'))
    mesh = np.load(io.BytesIO(z.read('mesh.npz')), allow_pickle=False) if 'mesh.npz' in z.namelist() else None

variant = next(v for v in project['variants'] if v['uid'] == project['active'])
for cam in variant['cameras']:
    print(cam['tag'], cam['x'], cam['y'], cam['z'], cam['yaw'], cam['pitch'], cam['roll'])
```

## Conventions

- **Vehicle frame**: ISO 8855: x forward, y left, z up; metres; origin at the centre of the rear axle, on the ground.
  Every position in the file is in this frame. `frame` and `custom_origin` only choose the origin the app shows and
  exports in (`rear_axle`, `front_axle`, `front_bumper`, `centre`, `custom`); the file always stores rear-axle
  coordinates.
- **Orientation**: yaw, pitch, roll in degrees, applied as intrinsic Z-Y-X rotations, **pitch positive down** (a
  camera pitched 30° looks 30° below the horizon). With yaw = pitch = roll = 0 a camera's optical axis points along +x;
  its OpenCV optical frame (x right, y down, z forward) is `BODY_TO_OPTICAL · Rᵀ` with `R = rot_zyx(yaw, pitch, roll)`
  (`vantage/core/mathutil.py`). A lidar's or radar's own frame is x forward, y left, z up (REP 103) at its optical
  centre, turned the same way.
- **Units**: metres for positions and lengths unless the key says otherwise (`_mm`, `_cm`, `_deg`, `_px`, `_um`, `_w`,
  `_g`, `_hz`, `_mbps`, `_gbps`); degrees for angles; seconds since 1970-01-01 UTC for times (`created`, `saved_at`,
  the log). `units` is only the unit the app displays.
- **Pixels**: OpenCV coordinates, pixel centres on integers (the centre of a 1920-pixel-wide image is x = 959.5).
- **uids**: cameras, sensors, variants, boards, obstacles and rig parts carry a `uid`, ten random hexadecimal digits.
  They are identities other entries refer to (a camera's `link_uid`, a variant's seam keys `uidA|uidB`, `active`,
  settings that name a variant); they carry no meaning and differ every time a sample or template is created.
- **Tags**: a camera's or sensor's `tag` (`F`, `ML`, `L1`, `RD3`) is its short name, unique within its variant; files
  Vantage exports name sensors by tag.

## Versioning

`format` is always `"vantage"` and `version` is the format version, 1 since Vantage 0.1. Vantage adds fields without
changing the version and reads files tolerantly:

- a missing field takes its default, so files from older versions open as they were saved (an old project without
  sensors, rig parts or a study's settings gets none or the defaults);
- an unknown field is ignored, so a file from a newer version opens in an older one, without the newer fields;
- the component library is stored in full (`modules`, `lenses`, `sensor_models`); on reading, the entries saved in
  the file win over the built-in ones of the same id, and built-ins the file does not have are added. A project keeps
  the lens figures it was planned with even when a later Vantage revises its library.

`version` is to be raised only by a change an older reader would misread (Vantage 0.2 reads any version and relies
on the tolerance above). The schema is strict (an unknown key is an error) so that it documents every field Vantage
writes; it describes version 1 as Vantage 0.2 writes it.

## project.json

### Top level

| key | type | meaning |
|---|---|---|
| `format`, `version` | string, integer | `"vantage"`, 1 (required) |
| `name` | string | the project's name |
| `created`, `saved_at` | number | creation and save times |
| `vehicle` | object | the vehicle (below) |
| `modules`, `lenses`, `sensor_models` | objects by id | the component library |
| `variants` | array | rig layouts; `active` is the uid of the one shown when the project opens |
| `roi` | object | the ground region the coverage figures are computed on |
| `analysis` | object | analysis settings and every study's settings |
| `targets`, `target_settings` | array, object | calibration boards on the ground and the defaults for new ones |
| `obstacles` | array | poles and other standing objects around the vehicle |
| `rig` | array | solids of the data-collection rig (racks, masts, enclosures) |
| `ecu` | object | the computer the cameras connect to |
| `photos` | object | reference photos of the vehicle fitted to the body |
| `frame`, `custom_origin`, `units` | | display and default export frame; display units |
| `streams` | object | live-stream source per camera tag |
| `fleet` | object | vehicles built to this design and their calibrations |
| `log` | array | the last 400 history entries `[time, kind, text]` |

### vehicle

`name`, `spec_name` (model name as shown), `template` (the generated body's template key, else empty), `source` (the
imported file's path on the computer it was imported on, or `template:<key>`), `import_units` (`m`, `mm`, `cm`, `in`),
`import_up` and `import_forward` (the imported model's axes, `±X`, `±Y`, `±Z`), `occluder_target` (triangles the body
is simplified to for occlusion), `imported_tris`, `wheel_centres` (`FL`, `FR`, `RL`, `RR` → [x, y, z] m), `spec`
(datasheet dimensions in metres: `Length`, `Width`, `Height`, `Wheelbase`, `Ground clearance`), `landmarks` (named
points on the body, m), `parts` and `zones`.

A **part** groups body triangles (`face_part` in `mesh.npz`): `name`, `material` (as imported), `alpha` (opacity),
`color` (RGB 0–1), `role` (`occluder · mountable`, `occluder · not mountable` for glass, `wheels`, `ignored`), and
optionally `surface` and `surface_mm`, the surface material class and its thickness for the radar fascia study.
A **zone** is a painted mount area (`face_zone`): `name`, `kind` (`allowed`, `forbidden`), `auto`.

### The component library

A **module** (camera module): `id`, `name`, `vendor`, `sensor`, `width_px`, `height_px`, `pixel_um` (pixel pitch,
µm), `optical_format`, `interface`, `fps`, `hw_sync`, `size_mm` [w, h, length], `weight_g`, `cables_m` (lengths
offered), `power_w`, `shutter`, `dynamic_range_db`, `ip_rating`, `temp_c` [min, max], `chroma`, `url`, `builtin`,
`note` (where the figures come from).

A **lens**: `id`, `name`, `label`, `module_id` (the module it was characterised on, empty: any), `focal_mm`,
`f_number` (0: not given), `focus_near_m` (0: not given), `focus_far_m` (null: infinity), `projection`
(`rectilinear`, `equidistant`, `equisolid`, `stereographic`, `orthographic`, or `kannala_brandt` with `kb` = k1..k4),
`sheet_fov` [H, V, D] degrees, `max_half_deg` (the image circle's half angle; 0: from the diagonal), `pp_offset_px`
(principal point offset from the sensor centre, px), `aspect` (fy / fx), `measured` ([θ°, image height mm] samples the
model was fitted to), `fit_rms_px`, `builtin`, `note`.

A **sensor model** (lidar, radar, GNSS antenna, IMU): `id`, `name`, `kind`, `vendor`; lidar: `scan` (`spinning`,
`solid`, `nonrepetitive`), `channels`, `v_min`/`v_max` or measured `beams` (elevations, °), `h_fov`, `h_res`,
`rate_hz`, `rotation` (`cw`, `ccw` seen from above), `points_per_s`, `bytes_per_point`; ranges `range_m` (lidar: on a
10 % target; radar: on a 10 m² car), `range_max_m`, `range_min_m`; radar: `zones` [[range m, azimuth half-angle °,
elevation half-angle °]], `data_mbps`; GNSS: `mask_deg`; IMU/INS: `antennas`, `heading_sigma_deg` at
`heading_baseline_m`, `pos_sigma_cm`, `attitude_sigma_deg`, noise densities `accel_noise_density` (m/s²/√Hz),
`accel_random_walk` (m/s³/√Hz), `gyro_noise_density` (rad/s/√Hz), `gyro_random_walk` (rad/s²/√Hz); and for all:
`interface`, `time_sync`, `size_mm`, `optical_z_mm` (optical centre above the base), `power_w`, `builtin`, `note`.
Library figures are datasheet values or typical classes, as each entry's `note` says.

### variants

A **variant** is one rig layout: `uid`, `name`, `note`, `cameras`, `sensors`, `seams` (manual stitching seams:
`"uidA|uidB"` → azimuth in degrees about the body centre, counter-clockwise from +x), `blend_m` (seam blend width, m),
`created`.

A **camera**: `uid`, `tag`, `name`, `location`, `slot` (colour, 0–7), position `x`, `y`, `z` (its optical centre, m)
and `yaw`, `pitch`, `roll` (°), `module_id`, `lens_id`, `port` (GMSL port 1–8, 0: unassigned), `sync` (trigger line
A–H), `cable_m`, `standoff_mm` (optical centre off the surface), `bracket`, `snap_normal`, `locked`, `visible`,
`link_uid` (the mirrored partner), `route_extra_m`, `route_note`, and `calib`: the unit's own lens calibration (empty:
the library lens) with `model`, `width`, `height`, `fx`, `fy`, `cx`, `cy` (OpenCV pixels), `k` (Kannala–Brandt k1..k4,
the OpenCV fisheye model), `rms_px`, `serial`, `source`, `views`, `date`. A calibration overrides the lens model.

A **sensor**: `uid`, `tag`, `name`, `kind` (`lidar`, `radar`, `gnss`, `imu`), `model_id`, position `x`, `y`, `z` (its
optical centre; a GNSS antenna's phase centre; an IMU's axis origin) and `yaw`, `pitch`, `roll`, `phase_deg` (a
spinning lidar's azimuth at the top of each sync period, 0–360), `port`, `visible`, `locked`.

### roi

`mode` (`offset`: a band `distance` metres around the body, `measured_from` the `outline` or the `bbox`, with `round`
or `square` `corners`; `polygon`: the ground `polygon` [[x, y], …] m), `exclude_under_body`.

### analysis

`cell_m` (ground grid cell, m), `min_density` and `target_density` (px/cm), `obstacle_mode` (`ground`, or the height
checked: `pole05`, `pole10`, `child`), `tol_deg` and `tol_mm` (mounting tolerance), `load_cases` ([name, front axle
drop mm, rear axle drop mm]), `live` and `heatmap` (view settings), and one object per study, empty for its defaults:

| key | study (module in `vantage/core/`) |
|---|---|
| `bowl` | 3D surround-view bowl (`bowl.py`) |
| `volume` | blind volume (`volume.py`) |
| `guidelines` | parking guidelines (`guidelines.py`) |
| `stitched` | objects in the stitched view (`stitched_objects.py`) |
| `assistant` | the assistant's project brief |
| `perception` | perception range (`perception.py`) |
| `sensor_coverage` | lidar and radar coverage (`sensor_coverage.py`) |
| `timing` | sync and timing plan (`timing.py`) |
| `recording` | recording budget (`recording.py`) |
| `gnss` | GNSS and IMU (`gnss.py`) |
| `power` | power budget (`power.py`) |
| `exposure` | sun glare and dirt (`exposure.py`) |
| `brackets` | mount brackets (`brackets.py`) |
| `dataset` | sample dataset (`dataset.py`) |
| `lidar_calib` | lidar–camera calibration (`lidar_calib.py`) |
| `drivelog` | drive-log check (`drivelog.py`) |
| `requirements` | requirements scorecard (`requirements.py`) |
| `housings` | housings in view (`housings.py`) |
| `asbuilt` | as-built check (`asbuilt.py`) |
| `suspension` | suspension and loads (`suspension.py`) |
| `critical` | critical targets (`critical.py`) |
| `terrain` | road and body motion (`terrain.py`) |
| `observability` | calibration observability (`observability.py`) |
| `materials` | materials, glass and wipers (`materials.py`) |
| `weather` | weather and night (`weather.py`) |
| `harness` | cable harness (`harness.py`) |
| `automation` | what to regenerate on a change (`automation.py`; never a command: commands stay on the computer) |
| `sim` | simulator runs (`simrig.py`, `vantage/sim/`) |
| `experiments` | designed experiments, sensitivity and uncertainty (`experiments.py`) |
| `detection` | detection quality per rig on simulated drives (`detection_eval.py`) |
| `campaign` | simulation campaigns: injected faults, weather sweeps, rig comparisons (`simcampaign.py`) |
| `critical_carla` | critical targets placed as CARLA scenes (`critical_carla.py`) |
| `copilot` | the Copilot's approval records: the plans the user approved and their runs (`vantage/assistant/copilot.py`) |

Each module's settings class (its `from_dict`) reads its object tolerantly: unknown keys are dropped, numbers clamped
to the module's `LIMITS`, missing ones defaulted.

`analysis.drivelog` (`DriveLogSettings` in `drivelog.py`): `path` (the bag or nuScenes folder last checked),
`variant_uid` (empty: the active variant), `camera_stamp` (what a camera's header stamp marks: `mid_exposure`,
`trigger`, `readout_end`), `lidar_stamp` (`auto`, `start`, `end`), `frames_per_camera` (2–200, default 20),
`tolerance_deg` (0.01–10, default 0.3), `search_deg` (0.2–5, default 2), `drop_factor` (1.05–10, default 1.5),
`freeze_frames` (an integer, 1–10000, default 3: this many repeats in a row of one image, identical content, make a
camera frozen, a fault; fewer are a warning), `sync_tol_ms` (0.05–100, default 2), `sync_sweeps` (5–5000, default
300) and `mappings` (per bag path, the topic → sensor tag edits; the last 20 logs are kept).

`analysis.experiments` (`ExperimentSettings` in `experiments.py`): `name`, `variant` (the uid of the variant varied;
empty for the active one), `factors` (each: `id`, `kind` — `camera`, `sensor`, `cameras` (every visible camera),
`roi`, `load_case` or `weather` —, `target` (a camera's or sensor's uid), `attr` (`x` `y` `z` `yaw` `pitch` `roll`,
`lens`, `model`, `distance`, `case`, `condition`), `mode` (`value` or `offset`), `lo`, `hi`, `levels`, `choices` (the
levels of a categorical factor), `dist` (`uniform`, `normal`, `truncnormal`) with `mean` and `sd`, `mirror` (the
mirror partner follows), `name`), `responses` (each: `metric` (a requirements metric id), `params`, `op` and
`target`), `design` (`factorial`, `oat`, `lhs`, `sobol`, `morris`, `montecarlo`), `samples`, `sobol_n`, `morris_r`,
`levels`, `morris_levels`, `seed`, `bootstrap`, `confidence`, `workers`.

`analysis.sim` (`SimSettings` in `simrig.py`, the CARLA run): `town`, `weather` (`project`: the selected weather
condition, `vantage:<id>` or a CARLA preset), `duration_s`, `traffic`, `seed`, `vehicle` (`auto` or a CARLA
blueprint), `camera_rate` (`lidar` or `module`), `image_scale`, `face_oversample`, `undistort`, `jpeg_quality`,
`exposure` (`manual` or `histogram`), `exposure_ev`, `keyframe_hz`, `channel_names` (`tags` or `nuscenes`),
`nuscenes`, `rosbag`, `lidar_noise_cm`, `drop_self_hits`, `max_lidar_channels`, `radar_points`, `tick_hz` (0: chosen),
`min_tick_hz`, `max_tick_hz`, `warmup_s`, `quality` (`Epic` or `Low`), `memory_gb` (the server's memory cap), `port`,
`tm_port`, `timeout_s`, `walkers` (pedestrians on CARLA's walker AI), `ground_truth` (`semantic`, `instance`,
`range`, `lidar_labels`: booleans, all on by default), `faults` (a list, each `tag` and `kind` — `rotation` with `deg`
[yaw, pitch, roll] in the sensor's own frame, `translation` with `m` [x, y, z] in the vehicle frame, `clock_offset`
with `ms`, `drop` with `rate` or `frames`, `freeze` with `from_s` and `to_s`, `range_bias` with `m` —; contracts in
`vantage/sim/carla_bridge.py`), `scenario` (`name` — `free_drive`, `lead_vehicle_brake`, `cut_in`,
`pedestrian_crossing`, `cyclist_crossing`, `parked` — and `params`, only the ones set), `actors` (an actors scene: each
`id`, `kind` — `child`, `pedestrian`, `cyclist`, `car`, `truck`, `traffic_light`, `sign`, `cone`, `barrier` or
`blueprint:<id>` — and `pose` `x`, `y`, `z` (m), `yaw` (°) in the vehicle frame), `spawn_point` (−1: a straight road
chosen by the seed). Where CARLA is installed and its Python stay on the computer (the app's settings), never in
the project; so do the live rig's cameras, render size and served size.

`analysis.detection` (`DetectionSettings` in `detection_eval.py`): `recordings` (absolute paths of CARLA run folders
with ground truth, the first the reference, as `drivelog.path`), the detector — `model` (`yolox_s`, `yolox_tiny`,
`yolox_nano`, `yolox_m`; the model file itself is downloaded into the user's data folder, never into the project),
`input_px` (0: each image at its own size), `max_input_px`, `keep_score`, `nms_iou`, `max_dets`, `fuse_riders`,
`backend` (`cpu`, `opencl`), `threads` —, the matching — `score` (the operating point), `iou`, `iou_small`, `small_px`,
`min_px`, `min_pixels`, `max_occluded`, `max_truncated` —, `range_edges` (m), `height_edges` (px), `classes`,
`cameras` (tags; empty: every camera), `frame_step`, the pinhole views — `views` (`fisheye`, `pinhole`, `both`),
`view_fov_deg`, `view_pitch_deg`, `view_yaws`, `view_width`, `view_aspect` —, the statistics — `bootstrap`, `block_s`,
`confidence`, `seed`, `min_gt` —, and `cache` (keep the detections in each recording).

`analysis.campaign` (`CampaignSettings` in `simcampaign.py`; empty: the fault-detectability template fitted to the
active variant): `name`, `template` (`fault_detectability`, `weather_sweep`, `rig_ab`, `custom`), `variants` (uids;
empty: the active one), `seeds`, `weathers` (CARLA presets or `vantage:<condition>`), `scenarios` (each `name` and
`params`), `fault_sets` (each `kind` — as `analysis.sim.faults` —, `tags`, `axes` for rotations (`yaw`, `pitch`,
`roll`) and translations (`x`, `y`, `z`), `magnitudes` in degrees, centimetres, milliseconds, per cent of the frames or
seconds by kind, `signs` (`alternate`, `positive`, `negative`, `random`)), `clean` (a clean drive per seed and
setting), `multiplex` (one condition per sensor per drive), `base` (`analysis.sim` keys over the project's for every
drive), `evaluators` (of `drivelog`, `rig_health`, `asbuilt`, `detection`, `image_stats`, `returns`), `asbuilt_on`
(`none`, `clean`, `all`), `drivelog`, `health` and `detection` (those checks' settings over their defaults),
`target` (the detection probability a threshold is quoted at), `confidence`, `server` (`running`: one already on the
port; `own`: started and stopped by the campaign), `out_dir` (empty: the working folder's
`<project>/campaigns/<name>`), `max_drives`.

`analysis.critical_carla` (`CriticalCarlaSettings` in `critical_carla.py`): `town`, `weather`, `spawn_points` (a
list; −1: a straight road chosen by the seed), `seed`, `duration_s` (1–10), `image_scale` (0.1–1, default 0.5),
`frames` (1–20, the last frames of each drive measured), `group` (targets that no camera sees overlap share a drive),
`vehicle` (`auto` or a `vehicle.*` blueprint), `actors` (target uid → `auto`, `standin`, `none` or
`blueprint:<id>`).

`analysis.copilot` (`vantage/assistant/copilot.py`): `plans`, the approval records of the plans the user approved
(newest last, at most 60), each `vantage.copilot_record/1`: `id` (the plan's, kept by its revisions), `version`,
`title`, `rationale`, `request` (the user's words that led to it), `provider` (`groq`, `claude`), `model`, `steps`
(each `kind` — `carla_run`, `campaign`, `detection`, `critical_carla`, `carla_live`, `experiment`,
`sample_dataset`, `study`, `export` —, `label`, `settings`: the full settings of the page that runs it as its settings
class keeps them, `sources`: per flat key where the value came from, `request`, `default`, `copilot` or `limit`, with
a reason, `warnings`, `notes`, `estimate`, and `settings_sha256` of `{kind, settings}`), `settings_sha256` of the
steps' `[{kind, settings}]`, `approved_utc`, `status` (`approved`, `running`, `done`, `failed`, `cancelled`) and
`runs` (each `started_utc`, `finished_utc`, `via`: `app`, `rerun` or `batch`, `status`, and per step `status`,
`message`, `output`, `summary`). A page's Run reached from a plan runs only when the record verifies (both hashes
match its steps) and its step's settings are exactly the page's; batch `--run-plan ID` runs one. The records are an
audit trail: undo never removes one, and the content hash leaves them out (their runs change as they run).

### Boards, obstacles, rig, ECU

A **target** (calibration board on the ground): `uid`, `name`, centre `x`, `y` (m), `yaw` (°), `pattern`
(`checkerboard`, `charuco`, `aprilgrid`), `rows`, `cols`, `square_mm`. `target_settings` holds the defaults for new
boards plus `cams_per_target` and `min_density`.

An **obstacle**: `uid`, `name`, `x`, `y`, `height`, `radius` (m).

A **rig part**: `uid`, `name`, `kind` (`box`: `length` along x, `width` along y, `height`; `cylinder`: vertical,
`radius`, `height`; `bar`: horizontal along y, `radius`, `length`), centre `x`, `y`, `z`, `yaw`, `occludes`,
`mountable`, `visible`.

The **ecu**: `name`, `location`, `x`, `y`, `z`, `ports`, `route_factor` (routing slack over the Manhattan length),
`service_loop_m`, `link_gbps` (GMSL2 forward channel per link), `csi_gbps_per_deser`, `bpp` (bits per pixel received,
16 = YUV 4:2:2), `overhead` (blanking and protocol factor).

### photos, streams, fleet, log

`photos`: `source` (the photos' folder or .zip on the computer that fitted them), `frames` ([`name`, `pose` 4×4
world-to-camera in OpenCV optical axes, `points` [[landmark, u, v]], `rms`]), the photo camera's `width`, `height`,
`f_px`, `cx`, `cy`, `k1`, the fit's `rms` and `method`, `solved_at`, `landmarks` (m) and `current` (the photo shown).

`streams`: camera tag → source (an RTSP/HTTP URL, a device, a file, `synthetic`). A URL never carries its password in
the file: Vantage moves passwords to a private store on the computer (`stream-credentials.json`, readable only by the
user) and puts them back when it opens the stream.

`fleet`: `vehicles` ([`id`, `name`, `vin`, `note`, `records`: measured poses of some sensors on one date, in the
vehicle frame]) and `settings` (`core/fleet.py`).

`log`: the last 400 entries of the project's history, `[time, kind, text]`.

## What is not in the file

Stream passwords (above), the assistant's API key (the system keyring or a private file), automation commands and
their approval, the window layout and preferences (the app's settings on the computer), recent files. A project can
be shared without them.

## Content hash and provenance

Every JSON file Vantage exports and every batch output carries a `"provenance"` block (`vantage/core/provenance.py`)
that names the project by a **content hash**: SHA-256 of the canonical JSON (keys sorted, no spaces, UTF-8,
integral numbers as integers) of the project as Vantage holds it in memory, with every uid replaced by the place of
its object (`variants[4].cameras[2]`: so a sample or template built twice hashes the same), the mesh arrays hashed by
dtype, shape and bytes, and stream URLs without passwords. Left out as volatile or specific to one computer:
`saved_at`, the `created` times, `log`, `active` (the variant is recorded on its own), the view settings
`analysis.live` and `analysis.heatmap`, `analysis.assistant`, `analysis.copilot`, `analysis.automation`,
`vehicle.source`, `photos.source` and `photos.current`. The block's `plan` field names the Copilot plan that produced
an output (`{plan, version, step}`), else null. The block also gives one shorter hash per section (`vehicle`, `mesh`, `library`, `variants`,
`roi`, `analysis`, …), so `vantage PROJECT --reproduce RESULT.json` can say which part of a project differs.

An imported or generated body is held in memory with 64-bit vertices and stored with 32-bit ones, so a project's
content hash (and, in the last digits, its figures) can change once, at its first save; from then on, saving and
reopening keeps both.

## Validating a file

```python
import json, zipfile
import jsonschema                      # pip install jsonschema

schema = json.load(open('docs/project.schema.json'))
with zipfile.ZipFile('rig.vantage') as z:
    jsonschema.validate(json.loads(z.read('project.json')), schema)
```
