# Changelog

Versions follow [semantic versioning](https://semver.org). The top entry is what the release notes on GitHub show.

## Unreleased

**What's new**

- **Clock tree and time sync** (Analysis ▸ Clock tree and time sync…, `--clock-tree`): the rig's PPS / gPTP / FSYNC time
  distribution with editable oscillators, servos, links and timers (typical values with sources), a fault timeline
  (PPS dropout, grandmaster loss with BMCA, link asymmetry, missed and late FSYNC pulses) as a seeded Monte Carlo,
  every sensor's timestamp error against true time and the lidar, and the projection error it makes in each camera
  through the real lens model, with a heat map over the image and 3D overlay segments.
- **Exposure metering and HDR seams** (Analysis ▸ Exposure metering and HDR seams…): an auto-exposure metering mask
  per camera through its real lens (no body, rig, housings, sky or the dark beyond the image circle), exported as
  PNG, a 0–15 metering grid and weighted AE rectangles (libargus `setAeRegions`); and a clear-sky photometric model
  that predicts each camera's exposure over the campaign and at low sun, a tunnel exit and night, the step across
  each stitch seam and what the bird's-eye gain balancing leaves of it. Batch `--isp OUT_DIR`.
- **Calibration coach** (Calibrate ▸ Live rig ▸ Calibration coach): live next-best-view coaching for lens and
  lidar–camera calibration with a ghost board over the camera's image, per-parameter lock bars from the Fisher
  information, sharp/duplicate gating, a simulated technician, and hand-off of the views to Lens calibration or
  Lidar–camera calibration; batch `--calib-coach`.
- **SUMO traffic in CARLA**: Simulate ▸ CARLA ▸ Traffic runs Eclipse SUMO co-simulated in lockstep (CARLA's own
  co-simulation package) as the background traffic: bundled, generated (cached) or your own net, seeded demand, CARLA
  or SUMO running the lights; SUMO's vehicles carry ground truth and run.json names them (`traffic_source`). Batch
  `--carla-traffic sumo`, `--sumo-net FILE` (exit 7 when SUMO is not in CARLA's Python). SUMO's own time on a 12 s
  drive: 4 s (Town04) to 11 s (Town10HD).
- **Gradient refinement of the rig** (Optimise placement ▸ Refine with gradients): L-BFGS-B with torch gradients on a
  smooth surrogate of the coverage objective (blind ground, ground below the minimum density, critical targets,
  objects on a seam), annealed from soft to sharp, from the current variant or the Pareto candidates; every result
  checked with the exact engine; Add as variant; batch `--refine-rig`.
- **Thermal and vibration drift** (Analysis ▸ Thermal and vibration drift…): how the midday sun warps the rig and how
  the road shakes it. Every camera's and sensor's move and angle change from its garage calibration hour by hour in
  full sun (the exposure study's sun, a clear sky, each rig part's temperatures from an energy balance, expansion,
  bowing and the aluminium rack against the steel roof), camera-lidar and stereo pairs in pixels through the lens;
  each mount's first mode, flex and blur on an ISO 8608 road through a half-car from the suspension tab, flagged in
  the body and wheel-hop bands; cobblestones, an expansion joint and a speed bump. Batch `--thermo-vib`.
- **Log replay** (Simulate ▸ Log replay…): a recorded drive rebuilt as 3D Gaussians (from its lidar and images, or a
  3DGS .ply), validated on held-out frames and sweeps, re-rendered through a proposed rig's lenses, body, rolling
  shutter and lidars, and detection compared between the recorded and the proposed rig per class and range; batch
  `--log-replay DRIVE_DIR OUT_DIR` with `--replay-variant`.
- **Online self-calibration** (Calibrate ▸ Online self-calibration…): every camera's extrinsics relative to the lidar
  over a whole recorded drive, window by window; a knock that stays told from an excursion that comes back and from
  flex with speed, braking, turning or the time of day; the calibration over time as JSON, a nuScenes folder with one
  calibrated_sensor row per segment and a ROS 2 /tf sidecar bag; `--online-calib DRIVE OUT_DIR` (exit 3 on a knock
  beyond the tolerance).

## 0.6.0 — 2026-10-29

The assistant now recommends what to do next, live, about what you are working on: the camera you selected, the edit
you just made, the page you are on, the job that just finished. Worked out on this computer; nothing runs or changes
until you choose it.

**What's new**

- **Recommended actions in the assistant, live as you work.** Worked out on this computer from Vantage's own results; computing them sends nothing.
  - **The live strip** at the top of the assistant panel follows what you are doing: the open page, the selection (a camera, a sensor, a rig part, a target, a requirement, a plan), your last edit with the analysis before and after it, and a job that has just finished. It shows one to three recommendations about that thing, a moment (300 ms) after it changes, with stable ids so nothing jumps.
    - *For QR · 3D and top views*: *QR alone sees 3.00 m²: check what its loss or a mounting error costs*. With a camera's minimum density below the target, copies with its pitch changed by ±2° and ±4° are analysed in the background, and a step that raises it without adding blind ground is named: *QR's minimum density is 3.44 px/cm (target 5): try pitch -4°*, with the simulated figures.
    - After an edit: *Your last change added 0.155 m² blind ground: undo it* (it shows History and names the step; it never undoes) or *ask why*.
    - On the CARLA page: *No server on port 2000: start it*; *Image scale 1 with ground truth exceeds the render budget: use 0.8* (the rig, built again at 0.8, renders 196 Mpx); *Check the last CARLA drive with the drive-log check*; *Score detection on the last drive*.
    - After a job: the drive-log check finds a camera rotated → *Ask why ML is off by …°*; detection quality has a weak cell → *Generate data where recall is weak: person 20–40 m in …* (a plan).
  - **The empty chat** lists the design's own recommendations, most useful first: blind ground (the largest patch, where, the nearest camera), a camera's minimum density, a seam where a standing child vanishes, the Issues by name, the worst single-camera mounting error, failing requirements (or none written), perception sectors short of their range, a lidar with no calibration plan, a stale analysis, CARLA never run, the last drive not checked or not scored, a campaign that misses your detection target, a plan waiting or failed, an unhealthy live stream; general starters when little is wrong. Five show; More shows the rest.
  - **Recommended (n)** in the panel's header opens the full list at any time.
  - **Next steps** under each answer: the recommendations worked out again, and up to three follow-ups from the model through the new `suggest_next` tool (Groq and Claude; a strict tool on Claude), checked and deduplicated. A plan card offers revisions from its own warnings and estimate (*Halve the duration: 20 → 10 s*, *Use half-size images: scale 1 → 0.5*, *Add a clear-weather drive to compare*) and, after it ran, the checks that fit it.
  - Each button says what it does: *Ask* and *Start goal* send the prompt as typing would (consent and the daily cap apply), *Plan it* asks the Copilot for a plan card that still needs approval, *Open* only shows a page or points at its button. No recommendation runs anything or changes the design by itself.
  - × dismisses one for this project file (kept in the app's settings, never in the project) until its figures change by more than 15 %.

**Fixed**

- The dividers between the window's panels were 2 px wide and the colour of the background, so a panel was hard to
  resize: they are 6 px wide and light up under the pointer. The assistant panel can be made narrower (from about
  315 px).

**What it does**

Everything in 0.5.0 and before (below): place cameras, lidars, radars, GNSS antennas and IMUs on the vehicle and the
rig; prove what they record; calibrate and check them against recordings; hand the calibration to the data pipeline;
drive the rig in CARLA with ground truth, faults, detection quality and campaigns; and let the Copilot plan any of it
for your approval. Now the assistant also tells you, as you work, what is worth doing next and why. Free software
under the GNU General Public License, version 3 or later.

**Downloads**

| System | File | Needs |
|---|---|---|
| Windows | `Vantage-0.6.0-windows-x64-setup.exe` (installer), or the `.zip` | Windows 10 or 11, x64 |
| macOS | `Vantage-0.6.0-macos-arm64.zip` | macOS 14 or later, Apple silicon |
| Linux x86_64 | `Vantage-0.6.0-linux-x86_64.tar.gz` | glibc 2.35 or later (Ubuntu 22.04, Debian 12, Fedora 36 and newer) |
| Linux arm64 | `Vantage-0.6.0-linux-arm64.tar.gz` | glibc 2.35 or later: NVIDIA Jetson Orin with JetPack 6, Ubuntu 22.04 arm64 and newer |

`SHA256SUMS.txt` lists each file's SHA-256. Every bundle carries its own Python and libraries and can check itself
on your machine: the *Vantage self-test* entry in the Start menu on Windows, `vantage --self-test` on Linux once
installed, `/Applications/Vantage.app/Contents/MacOS/vantage-cli --self-test` on macOS. On a Jetson, GStreamer
camera pipelines need `gst-launch-1.0` (JetPack installs it; else `sudo apt install gstreamer1.0-tools`).

**Unsigned builds**

The downloads are not code-signed, so the system warns the first time:

- **Windows** (SmartScreen, "Windows protected your PC"): choose *More info*, then *Run anyway*.
- **macOS** ("Vantage cannot be opened" or "Apple could not verify Vantage"): open *System Settings ▸ Privacy &
  Security* and choose *Open Anyway* next to the message about Vantage (on macOS 14 you can instead Control-click the
  app and choose *Open*). If macOS says the app is damaged, run in Terminal:
  `xattr -dr com.apple.quarantine /Applications/Vantage.app`.
- **Linux**: nothing to approve; check the download against `SHA256SUMS.txt`.

**Known limits**

- Validated in simulation and on synthetic captures only; the first fitted rig will tell how close the planned poses,
  the installation check and the calibrations come in practice.
- Lidar, radar, GNSS and IMU figures in the library are nominal datasheet values (the radars are typical classes, not
  specific products); check them against your units.
- The drift indicator of the live rig health is an indicator, not a measurement: it misses a rotation along a body
  edge (a mirror camera's yaw) and featureless edges, and frame times are when frames reached Vantage, not sensor time.
- STEP import needs the optional gmsh package, which the downloads do not include: run from source with
  `pip install .[step]` for it.
- No Intel Mac or Windows on Arm build (Windows on Arm runs the x64 build emulated); run from source there.
- The Linux arm64 bundle needs glibc 2.35 or later: JetPack 6 on a Jetson (Ubuntu 22.04), not JetPack 5 (Ubuntu
  20.04; run from source there). It carries Qt 6.7 rather than 6.11, because PyQt6's later arm64 builds need Ubuntu
  24.04; the whole test suite passes on both. It was built and self-tested on GitHub's arm64 machines, not yet on a
  Jetson; on a Jetson, OpenGL comes from NVIDIA's driver.
- Without OpenGL 2.1 the 3D view falls back to a slower software renderer.
- ROS 2 bags: rosbag2's whole-file compression (`.mcap.zstd`) is not read (`ros2 bag decompress` first), nor XCDR2
  messages.
- The assistant sends what it reads of the design (figures, issues, the project brief) to Groq; it is off until you
  give it a key, and it proposes design changes for you to apply rather than making them.
- Since 0.2.0: the road surface is used by the coverage analysis and the road study; the optimiser, tolerance study,
  perception range, stitched-objects study, installation check and critical targets still assume flat ground. The
  material, weather, headlamp and cable figures are typical values with their sources stated, not measurements of your
  vehicle, parts or units. The automation command runs with your own permissions, unsandboxed.
- Since 0.3.0: CARLA's cameras need a GPU. CARLA's lidar has no reflectivity and its radar no cross-section.
- Since 0.4.0:
  - CARLA renders a fisheye's views one after another: all six live fisheyes reach about 2.8 fps. One camera at a small
    size keeps real time.
  - **CARLA's car is not your vehicle's body.** The as-built check cannot judge mounts on a CARLA drive. The cameras
    share a small common pitch of −0.04 to −0.10° there.
  - **CARLA's lidar and radar ignore weather.**
  - **Ground truth at full image size** exceeds the render budget, so use image scale 0.5.
  - **The fog calibration** covers one town and CARLA 0.9.16.
  - **Detector:** YOLOX sees fisheye images as they are; pinhole views cut from them find more cars.
- Since 0.5.0:
  - Groq's prices are not bundled: a Groq conversation shows a cost only once you enter Groq's prices in Preferences.
  - An estimate is an estimate: CARLA minutes come from this machine's earlier drives (the development machine's
    figures before the first one), the disk from the sample dataset's estimator for the same rig with the ground truth
    as uncompressed PNG (an upper bound).
  - In batch mode a CARLA step needs a server already running (`--run-plan` never starts one), and a live-rig step with
    no duration watches until 5 s after its last fault (10 s without faults).
  - Claude's effort changes between a planning turn and a question; Anthropic documents that a change of the top-level
    effort invalidates the prompt cache of the conversation's messages, so such a turn reads fewer cached tokens.
  - Tested against Groq's real API in a recorded session and against a local stand-in of Anthropic's API: the first
    real Claude request (its betas, strict tools and streamed tool input) has not been made in development.
- New in 0.6.0:
  - The pitch suggestion for a selected camera tries only ±2° and ±4° of pitch, and only while the assistant panel is
    open.
  - Whether the last CARLA drive was checked or scored is known for the current session only, and the CARLA server's
    state only once the CARLA page has been opened.
  - The perception-range and requirements recommendations use those pages' last results; they do not run the studies.
- The validation suite checks the models on the cases it builds against independent references; it does not
  replace measuring a fitted rig.

## 0.5.0 — 2026-10-28

The assistant becomes the Copilot: say what you want run, and it answers with a plan you check, change and approve
before anything runs. And it can now run on Claude as well as Groq.

**What's new**

- **The Copilot** (Window ▸ Assistant, `/plan …`; Window ▸ Copilot plans… lists every plan). Ask for a run in plain words, for example *a 10 s drive in Town10
  at dusk in rain, 30 cars, 10 walkers, a pedestrian crossing, ground truth on, half-size images*, and the Copilot
  answers with a **plan**: one or more steps, each a CARLA drive, a simulation campaign, detection quality, critical
  targets in CARLA, the CARLA live rig (with faults on a schedule), a designed experiment, a sample dataset, a study
  or an export. The plan card shows:
  - every value, grouped by section, with where it came from (*your request*, *default*, *chosen by the Copilot*,
    *forced by a limit*) and why;
  - the documented limits that apply (*CARLA's lidar ignores rain*, *the as-built check cannot judge mounts on
    CARLA*, an image scale lowered to fit the bridge's render budget);
  - an estimate from Vantage's own estimators: CARLA minutes (from this machine's measured drives when there are
    any), frames and labels, disk, drives, design points;
  - what changed after a follow-up (*more rain*, *only F*, *halve it* revise the plan; they do not start over).

  Every value is said in words, the page's own: *F pitch 25–45° (5 levels) · F lens h05r2, h03r2*, *Blind ground in
  the ROI (m²), less is better*, *F turned pitch +1°*, *at 2 s: QR frozen* (the JSON stays in the value's tooltip).
  *Forced by a limit* means the page changed what was asked (*F pitch 25–95° → 25–90°*, *asked 200; the scenario
  allows 5–90, so 90*); the page filling in its defaults or writing a camera's tag as its uid is still *your request*.
  A setting the step's design, scenario or study does not use (samples on a full factorial, a free drive's traffic in
  a scenario, the step-rate range with a fixed step) is greyed in its own *Not used* section, never forced and not a
  decision of the plan. A follow-up's changes are listed one per setting: *Traffic 30 → 20 vehicles*, *Weather
  HardRainSunset → Vantage fog*, *Factors: added F lens h05r2, h03r2*.

  Double-click a value to change it with the page's own kind of field. A plan that does not validate (the page's
  rules, the render budget, the memory free, a CARLA port of 2000–2010 that is not the project's own, a missing
  detector model) cannot be approved.
- **Nothing runs until you approve.** **Approve and run** checks the plan again, writes an approval record into the
  project (`analysis.copilot`: the plan, your words, the provider and model, the settings and their SHA-256, the time,
  the status) and runs each step through its page's own Run; a CARLA step starts the CARLA page's server as the page
  does, on the project's port. A page's Run reached from a plan refuses without the record. An approved plan never
  changes: an edit is a new version that needs approval again. Undo never removes a record. **Open on its page** fills
  the page's fields as one undo step and runs nothing.
- **Runs you can trace and repeat.** Every output's provenance names the plan (a new `plan` field), and the plan's
  folder holds `copilot_plan.json` with each step's result. **More ▸ Copilot plans** runs an approved plan again from
  its stored settings without asking the model; batch **`--run-plan ID [OUT_DIR]`** does the same headless (exit 10
  when there is no approved, intact plan of that ID or a step no longer validates).
- **Claude as a provider** (Preferences ▸ Assistant ▸ Provider), beside Groq, each with its own key, models, usage and
  cost. Claude runs through Anthropic's official Python SDK: `claude-opus-5-5` by default or `claude-sonnet-5-5`,
  adaptive thinking at effort high when planning and medium for questions, strict tools whose every input is checked
  before it runs, prompt caching (the usage line shows the cache reads), an append-only conversation that passes the
  model's thinking back unchanged, and, on by default, Anthropic's server-side fallback when a request is declined. A
  refusal is shown with its reason and runs nothing.
- **Privacy and cost.** **More ▸ What was sent** shows the last request as it left, with this computer's folders shown
  as `~`; images, recordings, point clouds, keys, stream passwords and the automation command are never sent. The
  first question to a provider asks once. The usage line shows tokens and an estimated cost per conversation (Claude's
  prices built in with their source date; Groq's entered in Preferences); an optional daily token cap stops sending
  when it is used up.

**Changed**

- A Copilot plan puts traffic asked for with a CARLA scenario into the scenario's own Other traffic, the one the
  drive uses: *30 cars and a pedestrian crossing* drives with 30 cars (the page's Traffic is a free drive's and shows
  as not used).
- The assistant no longer writes a sample dataset or runs an experiment itself: both are plan steps you approve.
- The downloads include Anthropic's Python SDK (`anthropic` 2.3.0). From source: `pip install --user
  "anthropic==2.3.0"` or the new `[claude]` extra (where the system Python is externally managed, as on Ubuntu 23.04
  and later, `pip install --user` is refused: use a virtual environment); without it the Groq provider works as
  before.

**Fixed**

- The assistant's lookups showed `&#x27;` for an apostrophe in their names (*Read the Copilot's catalogue*).

- The CARLA page's top tiles no longer cover *The rig in CARLA* (and the live rig's notes no longer cover its fault
  row) in a 900 px window or with the assistant open: each pane of the column scrolls when it is short, and the plan
  pane gets the height its notes need when the first plan shows.

**What it does**

Everything in 0.4.0 and before (below): place cameras, lidars, radars, GNSS antennas and IMUs on the vehicle and the
rig; prove what they record; calibrate and check them against recordings; hand the calibration to the data pipeline;
drive the rig in CARLA with ground truth, faults, detection quality and campaigns. Now the Copilot plans any of it
from your words, and nothing runs until you approve the plan. Free software under the GNU General Public License,
version 3 or later.

**Downloads**

| System | File | Needs |
|---|---|---|
| Windows | `Vantage-0.5.0-windows-x64-setup.exe` (installer), or the `.zip` | Windows 10 or 11, x64 |
| macOS | `Vantage-0.5.0-macos-arm64.zip` | macOS 14 or later, Apple silicon |
| Linux x86_64 | `Vantage-0.5.0-linux-x86_64.tar.gz` | glibc 2.35 or later (Ubuntu 22.04, Debian 12, Fedora 36 and newer) |
| Linux arm64 | `Vantage-0.5.0-linux-arm64.tar.gz` | glibc 2.35 or later: NVIDIA Jetson Orin with JetPack 6, Ubuntu 22.04 arm64 and newer |

`SHA256SUMS.txt` lists each file's SHA-256. Every bundle carries its own Python and libraries and can check itself
on your machine: the *Vantage self-test* entry in the Start menu on Windows, `vantage --self-test` on Linux once
installed, `/Applications/Vantage.app/Contents/MacOS/vantage-cli --self-test` on macOS. On a Jetson, GStreamer
camera pipelines need `gst-launch-1.0` (JetPack installs it; else `sudo apt install gstreamer1.0-tools`).

**Unsigned builds**

The downloads are not code-signed, so the system warns the first time:

- **Windows** (SmartScreen, "Windows protected your PC"): choose *More info*, then *Run anyway*.
- **macOS** ("Vantage cannot be opened" or "Apple could not verify Vantage"): open *System Settings ▸ Privacy &
  Security* and choose *Open Anyway* next to the message about Vantage (on macOS 14 you can instead Control-click the
  app and choose *Open*). If macOS says the app is damaged, run in Terminal:
  `xattr -dr com.apple.quarantine /Applications/Vantage.app`.
- **Linux**: nothing to approve; check the download against `SHA256SUMS.txt`.

**Known limits**

- Validated in simulation and on synthetic captures only; the first fitted rig will tell how close the planned poses,
  the installation check and the calibrations come in practice.
- Lidar, radar, GNSS and IMU figures in the library are nominal datasheet values (the radars are typical classes, not
  specific products); check them against your units.
- The drift indicator of the live rig health is an indicator, not a measurement: it misses a rotation along a body
  edge (a mirror camera's yaw) and featureless edges, and frame times are when frames reached Vantage, not sensor time.
- STEP import needs the optional gmsh package, which the downloads do not include: run from source with
  `pip install .[step]` for it.
- No Intel Mac or Windows on Arm build (Windows on Arm runs the x64 build emulated); run from source there.
- The Linux arm64 bundle needs glibc 2.35 or later: JetPack 6 on a Jetson (Ubuntu 22.04), not JetPack 5 (Ubuntu
  20.04; run from source there). It carries Qt 6.7 rather than 6.11, because PyQt6's later arm64 builds need Ubuntu
  24.04; the whole test suite passes on both. It was built and self-tested on GitHub's arm64 machines, not yet on a
  Jetson; on a Jetson, OpenGL comes from NVIDIA's driver.
- Without OpenGL 2.1 the 3D view falls back to a slower software renderer.
- ROS 2 bags: rosbag2's whole-file compression (`.mcap.zstd`) is not read (`ros2 bag decompress` first), nor XCDR2
  messages.
- The assistant sends what it reads of the design (figures, issues, the project brief) to Groq; it is off until you
  give it a key, and it proposes design changes for you to apply rather than making them.
- Since 0.2.0: the road surface is used by the coverage analysis and the road study; the optimiser, tolerance study,
  perception range, stitched-objects study, installation check and critical targets still assume flat ground. The
  material, weather, headlamp and cable figures are typical values with their sources stated, not measurements of your
  vehicle, parts or units. The automation command runs with your own permissions, unsandboxed.
- Since 0.3.0: CARLA's cameras need a GPU. CARLA's lidar has no reflectivity and its radar no cross-section.
- Since 0.4.0:
  - CARLA renders a fisheye's views one after another: all six live fisheyes reach about 2.8 fps. One camera at a small
    size keeps real time.
  - **CARLA's car is not your vehicle's body.** The as-built check cannot judge mounts on a CARLA drive. The cameras
    share a small common pitch of −0.04 to −0.10° there.
  - **CARLA's lidar and radar ignore weather.**
  - **Ground truth at full image size** exceeds the render budget, so use image scale 0.5.
  - **The fog calibration** covers one town and CARLA 0.9.16.
  - **Detector:** YOLOX sees fisheye images as they are; pinhole views cut from them find more cars.
- New in 0.5.0:
  - Groq's prices are not bundled: a Groq conversation shows a cost only once you enter Groq's prices in Preferences.
  - An estimate is an estimate: CARLA minutes come from this machine's earlier drives (the development machine's
    figures before the first one), the disk from the sample dataset's estimator for the same rig with the ground truth
    as uncompressed PNG (an upper bound).
  - In batch mode a CARLA step needs a server already running (`--run-plan` never starts one), and a live-rig step with
    no duration watches until 5 s after its last fault (10 s without faults).
  - Claude's effort changes between a planning turn and a question; Anthropic documents that a change of the top-level
    effort invalidates the prompt cache of the conversation's messages, so such a turn reads fewer cached tokens.
  - Tested against Groq's real API in a recorded session and against a local stand-in of Anthropic's API: the first
    real Claude request (its betas, strict tools and streamed tool input) has not been made in development.
- The validation suite checks the models on the cases it builds against independent references; it does not
  replace measuring a fitted rig.

## 0.4.0 — 2026-10-28

The rig in CARLA as a source of labelled training data and as a test bench: ground truth for every sensor, faults
of known size, standard scenarios, detection quality per rig, critical targets placed as CARLA scenes, CARLA as a
live rig, and campaigns that measure how small a fault Vantage's own checks catch.

**What's new**

- **Ground truth from CARLA** (Simulate ▸ CARLA, `--carla-run … --ground-truth`). Every camera frame gets:
  - a semantic image, an instance image and a range image, rendered by CARLA and taken through the same fisheye remap
    as the colour image;
  - its objects, each with a tight 2D box from its pixels, the full box of its 3D box through the lens, truncation,
    occlusion, range and the 3D box;
  - COCO files.

  Every lidar sweep gets nuScenes-lidarseg and nuScenes-panoptic labels from CARLA's semantic lidar, matched point
  for point. On a real drive:
  - lidar points inside vehicle boxes are 100 % labelled vehicle;
  - ranges agree with the lidar to −0.9 mm median;
  - traffic lights and signs cover 100 % of their pixels.

  `--ground-truth-summary` reads a drive's labels back. Simulate ▸ CARLA's **Ground truth** tab shows them frame by
  frame:
  - semantic labels in CARLA's palette;
  - instances, one colour per object;
  - range on a log scale;
  - the lidar sweep from above in its lidarseg classes;
  - the tight and full boxes, and an object table whose rows highlight the object.

  **Open run…** shows any drive recorded before, with its report and its checks.
- **Faults of known size** (`--faults`):
  - a camera or sensor turned or moved;
  - a clock offset;
  - dropped or frozen frames.

  CARLA records the faulty rig while every calibration file states the plan, exactly as a real faulty rig would, and
  run.json lists what was injected.
- **Scenarios** (`--scenario`): a lead vehicle braking, a cut-in, a pedestrian or cyclist crossing, the ego parked,
  each deterministic with the seed. **Actor scenes** (`--actors`) place walkers, cyclists, cars and props at given
  poses around a parked ego.
- **Detection quality per rig** (Simulate ▸ Detection quality, `--detection-quality`).
  - **The detector:** an off-the-shelf detector (YOLOX-S, Apache-2.0, downloaded on request and never bundled) runs
    through OpenCV on the recorded images.
  - **The scoring:** recall, precision and AP per camera, class and range, against CARLA's ground truth; the
    detection range; recall against pixel height; bootstrap intervals; rig against rig; the fisheye against pinhole
    views cut from it.
  - **Example:** on the sample, pitching every camera 20° up raises vehicle recall at 20–40 m by 12 points and
    lowers it beyond 40 m by 7.
  - **In designed experiments:** detection recall can be a response, so an experiment can ask which pitch
    maximises pedestrian recall.
- **Simulation campaigns** (Simulate ▸ Simulation campaigns, `--campaign`): many CARLA drives, each varied in one
  known way (a fault, a weather, a rig, a scenario, a seed), scored by the drive-log check, rig health, detection and
  image statistics.
  - **Fault detectability:** detection probability against fault size with Wilson intervals, the smallest fault
    caught 90 % of the time, false alarms and estimation accuracy.
  - **Weather sweeps.**
  - **Rig comparisons.**

  On 49 real drives, the drive-log check catches a camera turned 0.35° (front) or 0.52° (mirror) 90 % of the time
  with no false alarm on the clean cameras. The as-built check, which cannot judge a mount on CARLA, is listed as not
  applicable rather than counted as a false alarm. It found every clock offset from 1 ms, every frame drop and every frozen
  interval.
- **Critical targets in CARLA** (Simulate ▸ Critical targets in CARLA, the critical-targets page's In CARLA…): each
  target becomes the nearest CARLA actor (a child or adult walker, a cyclist, a car, a stand-in for a sign), and each
  camera's view of it is measured from CARLA's ground truth and compared with the critical-targets study. On the
  sample, all 30 camera views agree, and pixel heights agree to 2.6 % median. Drives are found at any folder depth.
- **CARLA as a live rig** (Simulate ▸ CARLA live rig, `--carla-live`): CARLA streams every camera as MJPEG, the Live
  rig page watches it as it watches a real rig, and faults can be injected while it streams. Body drift is measured
  against CARLA's own car.

**Fixed**

- **The drive-log check's rotation search** could stay at the plan on textured scenes, where its coarse map is
  nearly flat; on one CARLA route a camera turned 1° read −0.08° with high confidence. It now also searches from its
  fine map, accepting the result only when significantly more frames agree.
- **Kerb edges** seen at a grazing angle, which the lidar sees from higher up, biased the result by up to 0.4°; they
  are now left out.
- **Frames that split into two estimates** now lower the confidence or give no verdict.
- **The drive-log check now finds frozen cameras** (runs of identical images, with their times). The Drive-log
  page draws them in amber on the timeline, and a new Freeze repeats setting sets how many identical frames count.
- **Rig health's capture skew** is judged against the sync plan's trigger offsets: a rig with planned staggered
  triggers is no longer flagged. Live CARLA streams are judged against the live rig's own schedule.
- **The as-built check** reported a lidar-based rotation the lidar did not pin down; it now shows only the axes
  measured.
- **CARLA fog:** a Vantage visibility was mapped to CARLA fog about 1.6× too thin. It is now calibrated: visibility
  ≈ 7000 m / fog density.
- **The CARLA bridge:**
  - it no longer hangs when the server dies (a watchdog ends the run within twice its timeout);
  - it refuses runs above a render budget that crashes the server;
  - it has a `--tidy` mode that clears what a killed run left behind;
  - one CARLA job runs at a time across the app;
  - a cancel always reaches its tidy-up: numpy's worker threads, started before the bridge set its signals
    aside, could take the stop signal and end the bridge without tidying the server.
- **A campaign keeps its drives** when a newer Vantage words its rig file differently: only the rig's content
  keys the drives.

**Changed**

- Vantage's weather model's 5 % contrast threshold is documented as a human observer's: for a detector its fog
  ranges are an upper bound. CARLA renders rain as water on the lens rather than as loss with range.
- `--export` without `--formats` writes the default formats; the two new ones (`sim_campaign`,
  `detection_quality`) are written only when named. 47 export formats, 34 pages.
- New batch flags and exit codes:
  - 8: the detector model is missing;
  - 9: a campaign is incomplete.

**What it does**

Everything in 0.3.0 and before (below): place cameras, lidars, radars, GNSS antennas and IMUs on the vehicle and the
rig; prove what they record; calibrate and check them against recordings; hand the calibration to the data pipeline;
run designed experiments, drive the rig in CARLA, validate every model and trace every number. Now CARLA also labels
every frame, injects faults of known size, scores a detector per rig, stages critical targets, streams as a live rig
and runs campaigns that measure how small a fault Vantage's own checks catch. Free software under the GNU General
Public License, version 3 or later.

**Downloads**

| System | File | Needs |
|---|---|---|
| Windows | `Vantage-0.4.0-windows-x64-setup.exe` (installer), or the `.zip` | Windows 10 or 11, x64 |
| macOS | `Vantage-0.4.0-macos-arm64.zip` | macOS 14 or later, Apple silicon |
| Linux x86_64 | `Vantage-0.4.0-linux-x86_64.tar.gz` | glibc 2.35 or later (Ubuntu 22.04, Debian 12, Fedora 36 and newer) |
| Linux arm64 | `Vantage-0.4.0-linux-arm64.tar.gz` | glibc 2.35 or later: NVIDIA Jetson Orin with JetPack 6, Ubuntu 22.04 arm64 and newer |

`SHA256SUMS.txt` lists each file's SHA-256. Every bundle carries its own Python and libraries and can check itself
on your machine: the *Vantage self-test* entry in the Start menu on Windows, `vantage --self-test` on Linux once
installed, `/Applications/Vantage.app/Contents/MacOS/vantage-cli --self-test` on macOS. On a Jetson, GStreamer
camera pipelines need `gst-launch-1.0` (JetPack installs it; else `sudo apt install gstreamer1.0-tools`).

**Unsigned builds**

The downloads are not code-signed, so the system warns the first time:

- **Windows** (SmartScreen, "Windows protected your PC"): choose *More info*, then *Run anyway*.
- **macOS** ("Vantage cannot be opened" or "Apple could not verify Vantage"): open *System Settings ▸ Privacy &
  Security* and choose *Open Anyway* next to the message about Vantage (on macOS 14 you can instead Control-click the
  app and choose *Open*). If macOS says the app is damaged, run in Terminal:
  `xattr -dr com.apple.quarantine /Applications/Vantage.app`.
- **Linux**: nothing to approve; check the download against `SHA256SUMS.txt`.

**Known limits**

- Validated in simulation and on synthetic captures only; the first fitted rig will tell how close the planned poses,
  the installation check and the calibrations come in practice.
- Lidar, radar, GNSS and IMU figures in the library are nominal datasheet values (the radars are typical classes, not
  specific products); check them against your units.
- The drift indicator of the live rig health is an indicator, not a measurement: it misses a rotation along a body
  edge (a mirror camera's yaw) and featureless edges, and frame times are when frames reached Vantage, not sensor time.
- STEP import needs the optional gmsh package, which the downloads do not include: run from source with
  `pip install .[step]` for it.
- No Intel Mac or Windows on Arm build (Windows on Arm runs the x64 build emulated); run from source there.
- The Linux arm64 bundle needs glibc 2.35 or later: JetPack 6 on a Jetson (Ubuntu 22.04), not JetPack 5 (Ubuntu
  20.04; run from source there). It carries Qt 6.7 rather than 6.11, because PyQt6's later arm64 builds need Ubuntu
  24.04; the whole test suite passes on both. It was built and self-tested on GitHub's arm64 machines, not yet on a
  Jetson; on a Jetson, OpenGL comes from NVIDIA's driver.
- Without OpenGL 2.1 the 3D view falls back to a slower software renderer.
- ROS 2 bags: rosbag2's whole-file compression (`.mcap.zstd`) is not read (`ros2 bag decompress` first), nor XCDR2
  messages.
- The assistant sends what it reads of the design (figures, issues, the project brief) to Groq; it is off until you
  give it a key, and it proposes design changes for you to apply rather than making them.
- Since 0.2.0: the road surface is used by the coverage analysis and the road study; the optimiser, tolerance study,
  perception range, stitched-objects study, installation check and critical targets still assume flat ground. The
  material, weather, headlamp and cable figures are typical values with their sources stated, not measurements of your
  vehicle, parts or units. The automation command runs with your own permissions, unsandboxed.
- Since 0.3.0: CARLA's cameras need a GPU. CARLA's lidar has no reflectivity and its radar no cross-section.
- New in 0.4.0:
  - CARLA renders a fisheye's views one after another: all six live fisheyes reach about 2.8 fps. One camera at a small
    size keeps real time.
  - **CARLA's car is not your vehicle's body.** The as-built check cannot judge mounts on a CARLA drive. The cameras
    share a small common pitch of −0.04 to −0.10° there.
  - **CARLA's lidar and radar ignore weather.**
  - **Ground truth at full image size** exceeds the render budget, so use image scale 0.5.
  - **The fog calibration** covers one town and CARLA 0.9.16.
  - **Detector:** YOLOX sees fisheye images as they are; pinhole views cut from them find more cars.
- The validation suite checks the models on the cases it builds against independent references; it does not
  replace measuring a fitted rig.

## 0.3.0 — 2026-10-27

Research grade: designed experiments over the rig, the planned rig in the CARLA simulator, a validation suite with a
technical reference for every model, and a provenance record on every number Vantage writes.

**What's new**

- **Designed experiments** (Analysis ▸ Experiments): vary camera and sensor poses, lenses and models, the mount
  height, the ROI distance, the load case or the weather over a full factorial, a one-at-a-time sweep, a Latin
  hypercube, Sobol/Saltelli or Morris design, or propagate input distributions by Monte Carlo; the responses are the
  requirements scorecard's metrics (now including critical targets and calibration observability). Sobol first-order
  and total indices with bootstrap intervals (checked against the Ishigami function's exact indices), Morris μ*/σ, main
  effects, two-factor surfaces, Pareto fronts and the best design, added as a variant in one click. Points are cached
  and a cancelled run resumes; cheap metrics take about 3 ms a point. `vantage --experiment spec.json OUT` runs one
  headless.
- **Simulate in CARLA** (Simulate ▸ CARLA): the planned rig in the free CARLA simulator (0.9.16) on the CARLA car
  nearest the vehicle's size, in its towns, traffic and weather, recorded as a nuScenes folder and a ROS 2 bag in the
  sample dataset's layout with ground-truth boxes. Fisheye cameras are rendered as pinhole cube faces and remapped
  through the lens model (within 0.1 px), lidar beams lie on CARLA's channels with the exact azimuth step, each radar
  zone is a CARLA radar; the faces are feathered across their seams, Unreal's vignette is taken out in linear light
  through CARLA's measured camera response, and each lidar point is placed where the sensor was when it fired. The
  server runs headless in its own memory-capped scope (on the NVIDIA GPU of a computer with two); the drive then opens
  in the drive-log check. On an RTX 5080 the sample's six fisheyes, lidar and five radars record 6 s in 28 s, and the
  drive-log check finds every camera within 0.25° of its plan. `--carla-rig` writes the rig for CARLA's own Python, `--carla-run` records a drive (exit 7 when no server
  answers or the run fails).
- **Validation suite** (Help ▸ Validate the models…, `vantage --validate`): 69 checks of 20 models against references
  that share none of their code — OpenCV, analytic footprints, exact ray casting, ITU-R and UN R112 tables, published
  worked examples, Monte Carlo — each with its quantity, reference, measured error, tolerance and verdict, as HTML, JSON
  and CSV; exit 6 when a check fails. The website's **technical reference** (methods.html) documents each model's
  equations, frames, assumptions, parameters and limits with its literature and standards, and the suite's report is
  published next to it.
- **Provenance and reproducing a result**: every JSON Vantage writes (exports, page exports, batch outputs) ends with
  a `provenance` block — the version and a hash of the code, the libraries, the project's and variant's content
  hashes, the study's settings with every default filled in, its seeds, the batch command and each input's SHA-256,
  and a hash of the result. `vantage project.vantage --reproduce result.json` runs it again and compares every number,
  exactly or within `--tolerance`, and says why a result does not reproduce (exit 6). Every random number now comes
  from a seeded generator. The `.vantage` format is documented in `docs/project-format.md` with a JSON Schema, and
  `CITATION.cff` says how to cite Vantage.

**Fixed**

- The body occlusion maps stored, per direction, the distance to the centre of the triangle drawn there rather than
  to the surface itself: 15 % off in the median and over 3× in 5 % of directions (which directions are hidden was
  right). They now store the exact distance (0.15 % median error), in the same time. Two results move with it:
  - a radar outside the body (an open engine bay) no longer treats a surface just ahead of it as its bumper fascia;
    the sample's radar coverage is unchanged;
  - the drive-log check leaves out lidar edges that land on the camera's own vehicle, where the image's edges are
    masked: they pulled the rotation off. On the sample dataset's drive, a correctly mounted mirror camera's apparent
    error falls from up to 0.9° to 0.36°.
- The calibration observability check's agreement band was lopsided: it could reject a correct prediction that came
  out lower than the simulated spread. It is now symmetric.
- The tolerance study and the joint solve could give different numbers for the same project in two sessions (mirrored
  pairs and boards were ordered by random ids); they now follow the project's order.
- The snow attenuation uses ITU-R P.1814's printed constants (a 0.1 % change).
- The drive-log check measured a stream's rate from its median interval: a stream stamped on a coarser clock (a
  15 Hz radar on 80 Hz ticks, intervals of 62.5 and 75 ms) read 16 Hz and was flagged. It now uses the mean of the
  regular intervals, drops left out.
- The drive-log check run from a script (Python, without the analysis engine) did not model the own vehicle: lidar
  edges a mirror camera's own mirror hides were matched to the mirror's image edge, and a correctly mounted mirror
  camera came out 0.4° off and "rotated". It now always models the vehicle, with the checked variant's rig. Whether
  an axis is pinned down is now judged over at least a pixel, so on small images a well-agreed axis is no longer
  called uncertain.
- The road study said "N of M cases pass the load cases' limits" when it counted the cases that exceed them.
- Typographic characters (°, ±, ×, →) in exported PNG images came out as "??".

**Changed**

- `--export` without `--formats` writes the default formats; the two new ones (`experiment`, `sim_rig`) are written
  only when named. 45 export formats, 31 pages.
- New batch exit codes: 6 (a result does not reproduce, or a validation check failed) and 7 (a CARLA run failed).

**What it does**

Everything in 0.2.0 and before (below): place cameras, lidars, radars, GNSS antennas and IMUs on the vehicle and the
rig; prove what they record; calibrate and check them against recordings; hand the calibration to the data pipeline.
Now with designed experiments over the design, the rig in the CARLA simulator, every model validated against
independent references, and a provenance record on every number. Free software under the GNU General Public License,
version 3 or later.

**Downloads**

| System | File | Needs |
|---|---|---|
| Windows | `Vantage-0.3.0-windows-x64-setup.exe` (installer), or the `.zip` | Windows 10 or 11, x64 |
| macOS | `Vantage-0.3.0-macos-arm64.zip` | macOS 14 or later, Apple silicon |
| Linux x86_64 | `Vantage-0.3.0-linux-x86_64.tar.gz` | glibc 2.35 or later (Ubuntu 22.04, Debian 12, Fedora 36 and newer) |
| Linux arm64 | `Vantage-0.3.0-linux-arm64.tar.gz` | glibc 2.35 or later: NVIDIA Jetson Orin with JetPack 6, Ubuntu 22.04 arm64 and newer |

`SHA256SUMS.txt` lists each file's SHA-256. Every bundle carries its own Python and libraries and can check itself
on your machine: the *Vantage self-test* entry in the Start menu on Windows, `vantage --self-test` on Linux once
installed, `/Applications/Vantage.app/Contents/MacOS/vantage-cli --self-test` on macOS. On a Jetson, GStreamer
camera pipelines need `gst-launch-1.0` (JetPack installs it; else `sudo apt install gstreamer1.0-tools`).

**Unsigned builds**

The downloads are not code-signed, so the system warns the first time:

- **Windows** (SmartScreen, "Windows protected your PC"): choose *More info*, then *Run anyway*.
- **macOS** ("Vantage cannot be opened" or "Apple could not verify Vantage"): open *System Settings ▸ Privacy &
  Security* and choose *Open Anyway* next to the message about Vantage (on macOS 14 you can instead Control-click the
  app and choose *Open*). If macOS says the app is damaged, run in Terminal:
  `xattr -dr com.apple.quarantine /Applications/Vantage.app`.
- **Linux**: nothing to approve; check the download against `SHA256SUMS.txt`.

**Known limits**

- Validated in simulation and on synthetic captures only; the first fitted rig will tell how close the planned poses,
  the installation check and the calibrations come in practice.
- Lidar, radar, GNSS and IMU figures in the library are nominal datasheet values (the radars are typical classes, not
  specific products); check them against your units.
- The drift indicator of the live rig health is an indicator, not a measurement: it misses a rotation along a body
  edge (a mirror camera's yaw) and featureless edges, and frame times are when frames reached Vantage, not sensor time.
- STEP import needs the optional gmsh package, which the downloads do not include: run from source with
  `pip install .[step]` for it.
- No Intel Mac or Windows on Arm build (Windows on Arm runs the x64 build emulated); run from source there.
- The Linux arm64 bundle needs glibc 2.35 or later: JetPack 6 on a Jetson (Ubuntu 22.04), not JetPack 5 (Ubuntu
  20.04; run from source there). It carries Qt 6.7 rather than 6.11, because PyQt6's later arm64 builds need Ubuntu
  24.04; the whole test suite passes on both. It was built and self-tested on GitHub's arm64 machines, not yet on a
  Jetson; on a Jetson, OpenGL comes from NVIDIA's driver.
- Without OpenGL 2.1 the 3D view falls back to a slower software renderer.
- ROS 2 bags: rosbag2's whole-file compression (`.mcap.zstd`) is not read (`ros2 bag decompress` first), nor XCDR2
  messages.
- The assistant sends what it reads of the design (figures, issues, the project brief) to Groq; it is off until you
  give it a key, and it proposes design changes for you to apply rather than making them.
- Since 0.2.0: the road surface is used by the coverage analysis and the road study; the optimiser, tolerance study,
  perception range, stitched-objects study, installation check and critical targets still assume flat ground. The
  material, weather, headlamp and cable figures are typical values with their sources stated, not measurements of your
  vehicle, parts or units. The automation command runs with your own permissions, unsandboxed.
- New in 0.3.0: CARLA's cameras need a GPU. CARLA's car is not your vehicle's body, so the as-built outline check reports a CARLA
  drive as not checkable; the drive-log rotation check is the cross-check (on CARLA drives the cameras share a pitch of
  about −0.1° whose cause is not known yet; the views' mounting, CARLA's projection and its lidar check out exactly). CARLA's lidar has no reflectivity and its radar no cross-section.
- The validation suite checks the models on the cases it builds against independent references; it does not
  replace measuring a fitted rig.

## 0.2.0 — 2026-10-26

Seven new studies for the rig on the road and in the workshop, and a more accurate installation check and lidar–camera
calibration.

**What's new**

- **Critical targets** (Analysis ▸ Critical targets): place what the rig must see — a child 2 m from the bumper, a
  pedestrian, a car, a traffic light, a sign, a kerb, a lane marking on a curve — and see which cameras, lidars and
  radars resolve each one; then the inverse: how far each camera's yaw, pitch, roll and position may move before a
  target is lost, as a mounting-tolerance sheet for the bracket and the install. The placement optimiser can require
  or weight the targets and adds the best three layouts as variants.
- **Road and body motion** (Analysis ▸ Road and body motion): brake dive, acceleration squat and cornering roll from
  the suspension model, with each forward camera's horizon shift; ramps, crests, garage ramps, speed bumps, kerbs and
  crowned roads in the coverage analysis, the car standing on the road and the road hiding what lies beyond a crest.
- **Calibration observability** (Calibrate ▸ Calibration observability): how precisely the planned boards will pin
  each camera's pose (and camera pairs, lidar–camera pairs, radars with corner reflectors), the weak axis and why,
  and the board moves that help most, checked against simulated calibrations; lidar points and scan lines per object
  and per pixel in each camera's view.
- **Materials, glass and wipers** (Analysis ▸ Materials, glass and wipers): materials on the vehicle's parts; what a
  radar loses through its bumper fascia, paint or a metal grille (and the lidar and radar coverage now uses it: on
  the sample the corner radars lose 2.7 dB through the painted fascia), cavity resonance and ghosting risk; refraction
  and dashboard glare for a camera behind the windscreen; the wiper sweep against each windscreen camera's view.
- **Weather and night** (Analysis ▸ Weather and night): overcast, night with headlamps or street lights, rain, fog and
  snow in the perception range study and in the rendered sample dataset (fog, darkness and headlamp pools in the
  camera images, dropped and false lidar points, weakened radar).
- **Cable harness** (Analysis ▸ Cable harness): every cable routed along the body to an entry grommet and on to the
  compute box, against its link's limit (GMSL2, Automotive Ethernet, CAN, GNSS RF loss, power), as a harness table;
  a stereo pair or trigger line split across deserializers is flagged.
- **Automation** (Export ▸ Automation): on save, regenerate the chosen calibration files and run your own test
  command (approved by you, kept in your settings, never in a project), with the changed files and a history;
  `--watch` for a headless loop and GitHub Actions / GitLab CI templates.

**Fixed**

- The installation check's joint solve stopped short of its optimum (37 of 40 simulated checks) and spread 1.2–3.3×
  wider than its own uncertainty; it now converges every time, matches its uncertainty, and takes 0.15 s instead of
  3.4 s.
- The lidar–camera calibration's predicted precision was 3–5× too optimistic and its solve had a small bias (x −3.3
  mm); both are fixed and the solve is about 9× faster.
- The import wizard counted the mirrors in the vehicle's width unless the part was named exactly "Mirrors"; a camera
  placed on the roof was named "Side right"; a radar deep in the nose was not flagged as inside the body.

**What it does**

Everything in 0.1.0 and 0.1.1 (below): place cameras, lidars, radars, GNSS antennas and IMUs on the vehicle and the
rig; prove what they record; calibrate and check them against recordings; hand the calibration to the data pipeline.
Free software under the GNU General Public License, version 3 or later.

**Downloads**

| System | File | Needs |
|---|---|---|
| Windows | `Vantage-0.2.0-windows-x64-setup.exe` (installer), or the `.zip` | Windows 10 or 11, x64 |
| macOS | `Vantage-0.2.0-macos-arm64.zip` | macOS 14 or later, Apple silicon |
| Linux x86_64 | `Vantage-0.2.0-linux-x86_64.tar.gz` | glibc 2.35 or later (Ubuntu 22.04, Debian 12, Fedora 36 and newer) |
| Linux arm64 | `Vantage-0.2.0-linux-arm64.tar.gz` | glibc 2.35 or later: NVIDIA Jetson Orin with JetPack 6, Ubuntu 22.04 arm64 and newer |

`SHA256SUMS.txt` lists each file's SHA-256. Every bundle carries its own Python and libraries and can check itself
on your machine: the *Vantage self-test* entry in the Start menu on Windows, `vantage --self-test` on Linux once
installed, `/Applications/Vantage.app/Contents/MacOS/vantage-cli --self-test` on macOS. On a Jetson, GStreamer
camera pipelines need `gst-launch-1.0` (JetPack installs it; else `sudo apt install gstreamer1.0-tools`).

**Unsigned builds**

The downloads are not code-signed, so the system warns the first time:

- **Windows** (SmartScreen, "Windows protected your PC"): choose *More info*, then *Run anyway*.
- **macOS** ("Vantage cannot be opened" or "Apple could not verify Vantage"): open *System Settings ▸ Privacy &
  Security* and choose *Open Anyway* next to the message about Vantage (on macOS 14 you can instead Control-click the
  app and choose *Open*). If macOS says the app is damaged, run in Terminal:
  `xattr -dr com.apple.quarantine /Applications/Vantage.app`.
- **Linux**: nothing to approve; check the download against `SHA256SUMS.txt`.

**Known limits**

- Validated in simulation and on synthetic captures only; the first fitted rig will tell how close the planned poses,
  the installation check and the calibrations come in practice.
- Lidar, radar, GNSS and IMU figures in the library are nominal datasheet values (the radars are typical classes, not
  specific products); check them against your units.
- The drift indicator of the live rig health is an indicator, not a measurement: it misses a rotation along a body
  edge (a mirror camera's yaw) and featureless edges, and frame times are when frames reached Vantage, not sensor time.
- STEP import needs the optional gmsh package, which the downloads do not include: run from source with
  `pip install .[step]` for it.
- No Intel Mac or Windows on Arm build (Windows on Arm runs the x64 build emulated); run from source there.
- The Linux arm64 bundle needs glibc 2.35 or later: JetPack 6 on a Jetson (Ubuntu 22.04), not JetPack 5 (Ubuntu
  20.04; run from source there). It carries Qt 6.7 rather than 6.11, because PyQt6's later arm64 builds need Ubuntu
  24.04; the whole test suite passes on both. It was built and self-tested on GitHub's arm64 machines, not yet on a
  Jetson; on a Jetson, OpenGL comes from NVIDIA's driver.
- Without OpenGL 2.1 the 3D view falls back to a slower software renderer.
- ROS 2 bags: rosbag2's whole-file compression (`.mcap.zstd`) is not read (`ros2 bag decompress` first), nor XCDR2
  messages.
- The assistant sends what it reads of the design (figures, issues, the project brief) to Groq; it is off until you
  give it a key, and it proposes design changes for you to apply rather than making them.
- New in 0.2.0: the road surface is used by the coverage analysis and the road study; the optimiser, tolerance study,
  perception range, stitched-objects study, installation check and critical targets still assume flat ground. The
  material, weather, headlamp and cable figures are typical values with their sources stated, not measurements of your
  vehicle, parts or units. The automation command runs with your own permissions, unsandboxed.

## 0.1.1 — 2026-10-25

Vantage on the vehicle: a Linux arm64 download for the data-collection computer itself (an NVIDIA Jetson Orin with
JetPack 6), so the live rig health, the drive-log check and the as-built check run in the field right after a drive,
and camera pipelines that open from the downloads.

**What's new**

- **Linux arm64** (`Vantage-0.1.1-linux-arm64.tar.gz`): built and smoke-tested on Ubuntu 22.04 arm64, the base of
  JetPack 6; installs per user like the x86_64 bundle.
- **GStreamer pipelines from the downloads.** The OpenCV wheels the downloads carry have no GStreamer, so a pipeline
  such as `nvarguscamerasrc sensor-id=0 ! … ! appsink` or `v4l2src device=/dev/video0 ! …` could not open. Vantage
  now runs it in the system's own `gst-launch-1.0` (on a Jetson, the one with NVIDIA's plug-ins) and reads the frames
  back raw; V4L2 devices (`/dev/video0`) open directly as before. Programs of the system that Vantage starts from the
  bundle (GStreamer, ffmpeg, the desktop tools) now get the session's environment instead of the bundle's libraries.
- `--self-test` reports how cameras open on the machine: V4L2, GStreamer (it runs a test pipeline), RTSP/HTTP and
  video files.

**Fixed**

- The installation check and lens calibration found board corners that depended on what had run before in the
  session: OpenCV's checkerboard detector draws on OpenCV's global random generator, and with the OpenCV 5 the
  downloads carry, a check after another camera's check could match one board's grid 75 px off (the joint solve then
  set that board aside). Every detection now starts the generator from the same seed, so the same image always gives
  the same corners.
- A theme or UI-scale change after a chart, plot or panel had been closed called into the deleted widget: an error
  in the log with Qt 6.11, a crash with Qt 6.7 (the Linux arm64 bundle's). Those widgets now disconnect when they go.

**What it does**

Everything in 0.1.0 (below): place cameras, lidars, radars, GNSS antennas and IMUs on the vehicle and the rig; prove
what they record (coverage with every occluder, blind volume, perception range, lidar and radar coverage, timing,
recording and power budgets, GNSS sky, sun and dirt, load cases from the suspension); calibrate and check them
(lens calibration, installation check, lidar–camera calibration, live rig health, drive-log check, as-built check,
fleet); and hand the calibration to the data pipeline (nuScenes, KITTI, ROS, Kalibr and more, each checked against
its tool). Free software under the GNU General Public License, version 3 or later.

**Downloads**

| System | File | Needs |
|---|---|---|
| Windows | `Vantage-0.1.1-windows-x64-setup.exe` (installer), or the `.zip` | Windows 10 or 11, x64 |
| macOS | `Vantage-0.1.1-macos-arm64.zip` | macOS 14 or later, Apple silicon |
| Linux x86_64 | `Vantage-0.1.1-linux-x86_64.tar.gz` | glibc 2.35 or later (Ubuntu 22.04, Debian 12, Fedora 36 and newer) |
| Linux arm64 | `Vantage-0.1.1-linux-arm64.tar.gz` | glibc 2.35 or later: NVIDIA Jetson Orin with JetPack 6, Ubuntu 22.04 arm64 and newer |

`SHA256SUMS.txt` lists each file's SHA-256. Every bundle carries its own Python and libraries and can check itself
on your machine: the *Vantage self-test* entry in the Start menu on Windows, `vantage --self-test` on Linux once
installed, `/Applications/Vantage.app/Contents/MacOS/vantage-cli --self-test` on macOS. On a Jetson, GStreamer
camera pipelines need `gst-launch-1.0` (JetPack installs it; else `sudo apt install gstreamer1.0-tools`).

**Unsigned builds**

The downloads are not code-signed, so the system warns the first time:

- **Windows** (SmartScreen, "Windows protected your PC"): choose *More info*, then *Run anyway*.
- **macOS** ("Vantage cannot be opened" or "Apple could not verify Vantage"): open *System Settings ▸ Privacy &
  Security* and choose *Open Anyway* next to the message about Vantage (on macOS 14 you can instead Control-click the
  app and choose *Open*). If macOS says the app is damaged, run in Terminal:
  `xattr -dr com.apple.quarantine /Applications/Vantage.app`.
- **Linux**: nothing to approve; check the download against `SHA256SUMS.txt`.

**Known limits**

- Validated in simulation and on synthetic captures only; the first fitted rig will tell how close the planned poses,
  the installation check and the calibrations come in practice.
- Lidar, radar, GNSS and IMU figures in the library are nominal datasheet values (the radars are typical classes, not
  specific products); check them against your units.
- The drift indicator of the live rig health is an indicator, not a measurement: it misses a rotation along a body
  edge (a mirror camera's yaw) and featureless edges, and frame times are when frames reached Vantage, not sensor time.
- STEP import needs the optional gmsh package, which the downloads do not include: run from source with
  `pip install .[step]` for it.
- No Intel Mac or Windows on Arm build (Windows on Arm runs the x64 build emulated); run from source there.
- The Linux arm64 bundle needs glibc 2.35 or later: JetPack 6 on a Jetson (Ubuntu 22.04), not JetPack 5 (Ubuntu
  20.04; run from source there). It carries Qt 6.7 rather than 6.11, because PyQt6's later arm64 builds need Ubuntu
  24.04; the whole test suite passes on both. It was built and self-tested on GitHub's arm64 machines, not yet on a
  Jetson; on a Jetson, OpenGL comes from NVIDIA's driver.
- Without OpenGL 2.1 the 3D view falls back to a slower software renderer.
- ROS 2 bags: rosbag2's whole-file compression (`.mcap.zstd`) is not read (`ros2 bag decompress` first), nor XCDR2
  messages.
- The assistant sends what it reads of the design (figures, issues, the project brief) to Groq; it is off until you
  give it a key, and it proposes design changes for you to apply rather than making them.

## 0.1.0 — 2026-10-24

The first public release. Vantage designs the sensor rigs of data-collection vehicles, the cars that record the camera,
lidar and radar data self-driving models are trained on, and proves in simulation what the rig will record before any
hardware is fitted. It is 0.1 because everything has been validated in simulation and on synthetic captures, not yet
on a fitted rig's real captures.

**What it does**

- **Vehicle and rig.** Import the vehicle (.glb .gltf .obj .stl .ply .fbx) or start from a template or the Tata Tigor EV
  sample; add the rig as simple solids (roof rack, sensor mast, enclosure, crossbar). Place cameras from a library of
  17 e-con Systems camera modules and 20 lenses (figures from e-con's product pages, each lens reproducing its stated
  field of view; rectilinear, fisheye and measured Kannala–Brandt models), lidars, radars, GNSS antennas and IMUs, with
  move and rotate handles in 3D and mirrored pairs.
- **What the rig records.** Live ground coverage, pixel density, redundancy and stitch seams with exact occlusion per
  camera by the body, the rig and every sensor housing and other camera module (a radar or lidar in a camera's view
  is flagged with the share of the image it hides); the 3D blind volume; perception range per object and direction; lidar and radar coverage and what the
  vehicle blocks; camera triggers against the lidar sweep; the recording and power budgets; GNSS sky view and lever
  arms; sun glare and lens dirt over a campaign; load cases from a suspension model (springs, tyres, bump stops, measured
  ride-height curves) and the real loads, with every camera's move against its calibration load; Monte Carlo mounting tolerance; a placement optimiser;
  variant comparison; a requirements scorecard with a design-review pack.
- **Calibration.** Lens calibration from checkerboard views, the end-of-line installation check from one image per
  camera with a joint extrinsic solve, lidar–camera calibration, live rig health from RTSP/HTTP, V4L2 and GStreamer
  streams, a drive-log check of recorded ROS 2 bags and nuScenes folders, an as-built check that compares each real
  camera's view of the vehicle in a recording with the simulated one, fits the mounting error and says whether the
  plan's coverage still holds, and a fleet of vehicles calibrated over time.
- **Hand-off.** Exports: mount sheet and reports (PDF, HTML), extrinsics (JSON, YAML, CSV), OpenCV, ROS 2 URDF and
  static TF, Jetson camera config, nuScenes, KITTI, ROS camera_info and Kalibr calibration files (each checked against
  the tool that reads it: the nuScenes and KITTI devkits, pykitti, Kalibr's config reader, ROS 2's camera_info parser), bird's-eye remap
  tables with C++ and Python reference stitchers, a 3D surround-view bowl, parking guidelines, DXF and STEP mount
  points, 3D-printable brackets (STL). A synthetic sample dataset (nuScenes folder and ROS 2 bag) recorded by exactly
  the planned rig. ROS 2 bags are read (MCAP, SQLite) and written (MCAP) without a ROS installation.
- **Batch mode** for scripts and CI: the studies, checks and exports from the command line, with exit codes for failed
  checks (`vantage project.vantage --check review/` runs the requirements scorecard). An optional assistant: a chat
  about the design through a Groq API key of your own. Dark and light themes, undo for every edit.

Vantage is free software under the GNU General Public License, version 3 or later (see LICENSE).

**Downloads**

| System | File | Needs |
|---|---|---|
| Windows | `Vantage-0.1.0-windows-x64-setup.exe` (installer), or the `.zip` | Windows 10 or 11, x64 |
| macOS | `Vantage-0.1.0-macos-arm64.zip` | macOS 14 or later, Apple silicon |
| Linux | `Vantage-0.1.0-linux-x86_64.tar.gz` | x86_64 with glibc 2.35 or later (Ubuntu 22.04, Debian 12, Fedora 36 and newer) |

`SHA256SUMS.txt` lists each file's SHA-256. Every bundle carries its own Python and libraries and can check itself
on your machine: the *Vantage self-test* entry in the Start menu on Windows, `vantage --self-test` on Linux once
installed, `/Applications/Vantage.app/Contents/MacOS/vantage-cli --self-test` on macOS.

**Unsigned builds**

The downloads are not code-signed, so the system warns the first time:

- **Windows** (SmartScreen, "Windows protected your PC"): choose *More info*, then *Run anyway*.
- **macOS** ("Vantage cannot be opened" or "Apple could not verify Vantage"): open *System Settings ▸ Privacy &
  Security* and choose *Open Anyway* next to the message about Vantage (on macOS 14 you can instead Control-click the
  app and choose *Open*). If macOS says the app is damaged, run in Terminal:
  `xattr -dr com.apple.quarantine /Applications/Vantage.app`.
- **Linux**: nothing to approve; check the download against `SHA256SUMS.txt`.

**Known limits**

- Validated in simulation and on synthetic captures only; the first fitted rig will tell how close the planned poses,
  the installation check and the calibrations come in practice.
- Lidar, radar, GNSS and IMU figures in the library are nominal datasheet values (the radars are typical classes, not
  specific products); check them against your units.
- The drift indicator of the live rig health is an indicator, not a measurement: it misses a rotation along a body
  edge (a mirror camera's yaw) and featureless edges, and frame times are when frames reached Vantage, not sensor time.
- STEP import needs the optional gmsh package, which the downloads do not include: run from source with
  `pip install .[step]` for it.
- No Intel Mac, Windows on Arm or Linux arm64 build yet (Windows on Arm runs the x64 build emulated); run from source
  there.
- Without OpenGL 2.1 the 3D view falls back to a slower software renderer.
- ROS 2 bags: rosbag2's whole-file compression (`.mcap.zstd`) is not read (`ros2 bag decompress` first), nor XCDR2
  messages.
- The assistant sends what it reads of the design (figures, issues, the project brief) to Groq; it is off until you
  give it a key, and it proposes design changes for you to apply rather than making them.
