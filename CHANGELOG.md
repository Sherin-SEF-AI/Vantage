# Changelog

Versions follow [semantic versioning](https://semver.org). The top entry is what the release notes on GitHub show.

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
