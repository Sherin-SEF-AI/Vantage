# Developing Vantage

Running the tests and where the code is.

## Tests

```bash
QT_QPA_PLATFORM=offscreen python3 -m pytest tests -q
```

Core engine, loaders (all formats, malformed files), exporters (OpenCV reprojection checked against cv2.fisheye), icons,
installation check (known mounting errors recovered from synthetic captures), unit lens calibration (a simulated unit
recovered from synthetic board views; every calibration file format), the joint solve (misplaced boards), exposure
balancing and multi-band blending, the C++ and Python reference stitchers (compiled and compared pixel by pixel with
the app's stitcher; the C++ test is skipped without g++), critical targets (pixels and envelopes of a pinhole camera
against closed forms, reported limits checked by moving the camera just inside and just outside them), materials
(multilayer transmission against the explicit sum of a slab's bounces, a measured painted bumper plate reproduced, the
glass's ray shift against the textbook plate formula, wiper sweeps against polar geometry), weather (ITU-R P.838-3
against its own table, P.840 against a Rayleigh derivation, the lidar and radar range equations against root finding,
the headlamp pattern against UN R112's test points; clear weather gives the old results bit for bit), the road (each
primitive against its closed form, the ray march against brute force, the road hidden beyond a crest against the
tangent from the camera, brake dive and roll against hand calculations; flat ground gives the old analysis bit for
bit), the cable harness (geodesics on a plane and unfolded across a box's edge, run lengths rebuilt from their parts,
the RF budget and the power drop against the power budget's), calibration observability (Jacobians against finite
differences, covariances against hand-built information matrices, predictions against Monte Carlo runs of the real
solves), automation (the command's approval, environment and exit code, the process group stopped, changes detected by
content, pruning that never leaves the folder, `--watch`, the CI templates checked against batch mode's flags), housings (depths against exact ray–box and ray–cylinder
hits, a lidar's shadow on the ground against its analytic projection, no body+rig cube rebuilt when a sensor moves),
the as-built check (synthetic recordings from the sample-dataset generator with cameras mounted wrong on purpose: the
injected pitch, roll and yaw recovered within their stated errors), suspension load cases (hand calculations of springs
and tyres in series, bump stops, measured curves, roll from off-centre loads), the dataset calibration files against
the tools that read them (the KITTI devkit's projection, pykitti, Kalibr's config reader and camera models, ROS 2's
camera_info parser and image_geometry; the tool-based tests run when `VANTAGE_DEVKITS` points at them), every e-con
module and lens against its stated fields of view, GStreamer pipelines through the system's gst-launch-1.0 (frames
read back exactly, padded rows, errors; run where gstreamer1.0-tools is installed), packaging (the spec collects every page, the release workflow,
install.sh and uninstall.sh, the self-test), stitching, load cases, link budget, rig health (statistics
on known sequences, synthetic and video-file streams, the drift indicator; no device or network stream is opened),
stream passwords kept out of project files, rig parts (occlusion, mounting, URDF), perception range (ranges against
the lens model's pixel heights), lidars and radars (ring radii, fields and radar range against closed-form geometry,
beam-angle files, misplacement issues, every sensor frame through the URDF, nuScenes and KITTI files), lidar and radar
coverage (points on objects against the beam geometry, roof blocking, verdicts), the sync plan (sweep crossings,
triggers, groups, blur against hand-worked cases), the recording budget (rates, links, disks), the dataset exports (reprojection through every written file), batch mode, the shader
renderer (run headless through EGL in a subprocess and checked against the CPU visibility test; skipped where there is no
EGL OpenGL), the 3D surround view, designed experiments (Sobol indices against the Ishigami function's analytic values
and their interval coverage over seeds, Morris on linear and interacting functions, Monte Carlo intervals; responses
against the studies run by hand on project copies; cancel and resume), the validation suite (a quick run of every
check, injected model faults caught by the checks meant to catch them, seeded reproducibility), provenance (content
hashes independent of random ids, results bit for bit in two processes, an audit of the sources for unseeded random
numbers, `--reproduce` on every batch output), the CARLA bridge (against a fake `carla` module that records what the
client asks for: frames, lidar resampling, the rig file, a drive end to end read back by the drive-log and as-built
checks, the ground truth (masks, COCO, range images against the lidar, lidarseg in the devkit's layout), faults
changing what is recorded while the calibration states the plan, scenarios and actors scenes, a server that stops
answering; a live test runs where `VANTAGE_CARLA` names a server), the live rig (MJPEG read back by OpenCV, faults in
rig health's verdicts, the tidy-up), detection quality (YOLOX's decoding on hand-built outputs, matching and AP against
hand cases and pycocotools, bootstrap coverage, a recording with known answers end to end), simulation campaigns
(Wilson intervals' exact coverage, a logistic threshold recovered, a fault campaign through the fake bridge, cancel and
resume) and critical targets in CARLA (the measurement on an analytic frame, the comparison through the fake bridge),
regression tests for every bug found in review
(`test_regressions_*.py`), and GUI tests that drive the app with real mouse and keyboard events. The validation suite
itself (`vantage --validate`) checks the models against independent references.

## Layout

```
vantage/core   engine: lens models, mesh, loaders, body generator, occlusion, analysis, tolerance, optimiser,
               targets, photos, rules, exporters, project model, synthetic captures (synth), installation
               check (install), joint extrinsic solve (joint), unit lens calibration (calib), stitching,
               exposure balancing and remap tables (stitch), 3D surround-view bowl (bowl), camera visibility
               (visibility), load cases, image sources (capture), stream passwords (credentials), perception
               range (perception), rig parts (rig), live stream health and drift (rig_health), lidar and radar
               library and geometry (sensors), lidar and radar coverage (sensor_coverage), sync and timing plan
               (timing), recording budget (recording), housings as camera occluders (housings), as-built
               check against a recording (asbuilt), suspension and loads (suspension), road primitives and
               body motion (terrain), critical targets and mounting envelopes (critical), materials, radar
               through the fascia and glass (materials), windscreen wipers (wipers), weather and night
               (weather), cable routing and link limits (harness), calibration observability
               (observability), regenerating exports and running the user's checks (automation), designed
               experiments and sensitivity (experiments), provenance blocks and reproducing results
               (provenance), the rig as a simulator needs it (simrig), CARLA's ground truth (groundtruth), the
               detector and its evaluation (detection, detection_eval), simulation campaigns (simcampaign),
               critical targets in CARLA (critical_carla), the live rig's app side (carla_live), and the rest of
               the studies (gnss, power,
               exposure, brackets, dataset, drivescene, rosbag, lidar_calib, drivelog, fleet, requirements)
vantage/validation  the validation suite: every model against independent references, the report (`--validate`)
vantage/sim    the CARLA bridge, run in CARLA's own Python (carla_bridge, record, frames, tonemap), the live rig
               server (live) and the server (server)
vantage/runtime  reference bird's-eye stitchers shipped with the remap tables (C++17, Python)
vantage/batch  command-line batch mode
vantage/mcp    the MCP server for Claude (server, the tools: service, changes as operations: edits, a window or a
               file: session); the window's side is vantage/ui/mcp_bridge
vantage/selftest  --self-test: checks an installation (analysis, exports, cameras, bags, plugins, OpenGL, every page)
                  without a window
vantage/sysenv  the environment for system programs started from the bundle (GStreamer, ffmpeg, desktop tools)
vantage/ui     Qt UI: main window, docks, viewports (render_gl: the shader renderer), plan views, camera images,
               charts, dialogs, workspaces, the bowl controller
tests/         pytest suites (tests/fake_carla: a fake carla module for the bridge's tests)
docs/          the user guide (user-guide.md, generated from site/guide.html), the .vantage project format
               (project-format.md) and its JSON Schema; the simulator bridge contract (simulator-bridge.md) and its schemas
site/          the website: index, the guide (guide.html), docs, the technical reference (methods.html) and the published validation report
packaging/     PyInstaller spec, build script, icons, Linux install scripts and desktop entry,
               Windows installer (Inno Setup); .github/workflows/build.yml builds them on GitHub
```

Fonts: IBM Plex Sans and IBM Plex Mono, SIL Open Font License 1.1 (`vantage/resources/fonts`).
