# Vantage · sensor rig planner for autonomous-driving data collection

A desktop app for designing the sensor rigs of data-collection vehicles, the cars that record the camera data
self-driving models are trained on. Import a 3D model of the vehicle, place the cameras on it, and prove in simulation
what the rig will record before any hardware is fitted: what each camera sees through its real lens model, where the
body blocks it, how many pixels land on the ground and on objects, where cameras overlap, and how sensitive the layout is
to mounting errors and load. Then calibrate the real cameras from their live streams (RTSP/HTTP, V4L2, GStreamer
pipelines such as a Jetson's GMSL cameras), check every installed camera against the plan, and hand the calibration to the
data pipeline (OpenCV, ROS 2, Kalibr and more). Cameras today; lidar and radar are planned.

The same engine also serves production surround-view systems: stitched bird's-eye remap tables, a 3D bowl view, parking
guidelines and where standing people vanish at stitch seams.

**Website and documentation:** <https://sherin-sef-ai.github.io/Vantage/> (feature videos recorded from the app, the full documentation, batch-mode reference). The site's source is `site/`; it is published by `.github/workflows/pages.yml`.

It opens with the **Tata Tigor EV** sample: the body is lofted from the profile measured on the 19-frame studio spin
(wheel-centre calibration, 1.6 px RMS), the cameras are e-con Systems STURDeCAM31 (Sony ISX031, 1920 × 1536, 3.0 µm) with
the H05R2 wide and H03R2 narrow lenses, and four layout variants are included.

The studio photos themselves are not part of the repository. To use the photo-matching features with them, put
`Tata-Tigor-EV-360.zip` in `data/` (or next to the project folder) or set `VANTAGE_TIGOR_PHOTOS` to its path; without
it the sample opens normally with the photo poses but no images.

## Install

**Linux, no Python needed.** Download `Vantage-<version>-linux-x86_64.tar.gz`, then:

```bash
tar xf Vantage-*-linux-x86_64.tar.gz
cd Vantage-*-linux-x86_64
./install.sh            # per user: ~/.local/opt/vantage, a menu entry, the .vantage file type, ~/.local/bin/vantage
vantage --self-test     # checks analysis, OpenCV, fonts, the OpenGL renderer and the main window
```

`~/.local/opt/vantage/uninstall.sh` removes it again (projects and settings are kept). Without installing, run
`./Vantage` from the extracted folder.

**Windows, no Python needed.** Run `Vantage-<version>-windows-x64-setup.exe`. It installs for the current user without
administrator rights, adds a Start menu entry and opens `.vantage` projects. `Vantage.exe` is the window;
`vantage-cli.exe` in the same folder is for batch mode and `--self-test` from a terminal. The `.zip` is the same program
without the installer.

**macOS.** Unzip `Vantage-<version>-macos-arm64.zip` and move `Vantage.app` to Applications. The app is not signed: the
first time, right-click it and choose Open. Batch mode: `Vantage.app/Contents/MacOS/vantage-cli`.

**With Python 3.10 or later:**

```bash
python3 -m pip install .            # from this folder; add [step] for STEP import through gmsh
vantage                             # the window (vantage-gui on Windows starts it without a console)
vantage project.vantage --analyse report.json
```

**Building the downloads.** Push a tag (`git tag v1.0.0 && git push origin v1.0.0`) or start *Build installers* in the
repository's Actions tab: GitHub builds and smoke-tests all three and attaches them to the run. To build for the machine
you are on:

```bash
python3 -m pip install -r requirements.txt pyinstaller
python3 packaging/build.py --smoke  # dist/: the bundle, the archive and on Windows the installer (needs Inno Setup 6)
```

## Run

```bash
cd vantage
python3 -m pip install -r requirements.txt     # PyQt6, numpy, scipy, PyOpenGL, PyYAML, opencv-python
python3 run_vantage.py                          # start window
python3 run_vantage.py --sample                 # straight into the Tigor EV sample
python3 run_vantage.py car.glb                  # import a model
python3 run_vantage.py project.vantage          # open a project
```

Options: `--theme dark|light|system`, `--scale 1.15`, `--software-3d` (QPainter 3D view instead of OpenGL),
`--template sedan|hatchback|suv|van|truck|tigor`, `--self-test`, `--version`.

### Batch mode (no window)

```bash
python3 run_vantage.py project.vantage --analyse report.json              # metrics, seams, issues, link budget, targets,
                                                                         # objects in the stitched view + seam advice
python3 run_vantage.py project.vantage --load-cases -                     # load cases as JSON on stdout
python3 run_vantage.py project.vantage --blind-volume volume.json         # 3D blind volume, unseen objects, reach rose
python3 run_vantage.py project.vantage --export out/ --formats extrinsics_json,opencv_yaml,bev_lut
python3 run_vantage.py project.vantage --install-check captures/ --report check.json \
        --lut-out out/bev_lut --as-built built.vantage                    # end-of-line check from one image per camera
python3 run_vantage.py project.vantage --calibrate-lens views/ --camera F --board 9x6 --square 60 \
        --serial SN1234 --calib-out F.yaml --save calibrated.vantage      # one camera's own lens, from board views
python3 run_vantage.py project.vantage --import-calib camchain.yaml F.yaml --save calibrated.vantage
```

`--variant NAME|N` picks a variant, `--frame`, `--units m|mm` and `--angles deg|rad` set the export conventions, and
`--pos-tol`, `--ang-tol`, `--min-boards` the installation tolerances. The installation check solves all cameras jointly
with the board placement unless `--no-joint`; `--board-tol MM` is how far a board may lie off its planned place
(default 20). The stitched PNG is exposure-balanced unless `--no-balance`, `--blend multiband` for multi-band seams.
Captures are matched to cameras by tag in the file name (`F.png`, `cam_QR.jpg`, `ML-capture.tif`); imported
calibrations by the camera name or ROS topic in the file, the file name, or the serial number. Exit code: 0 fine, 2 the
plan has critical issues (with `--analyse` or `--export`), 3 a camera has to be re-mounted or could not be checked,
4 a lens calibration failed or an imported calibration matched no camera (or did not fit it); when several apply, the
highest is returned.

## What is in it

| Area | Features |
|---|---|
| Vehicle import (Ctrl+I) | .glb .gltf .obj .stl .ply .fbx (binary and ASCII), .step via the optional gmsh package. Wizard: units (guessed from the bounding box), up/forward axes and mirroring, inside-out meshes flipped automatically, wheel centres (auto from wheel parts or picked on the mesh), ground plane, origin preset, spec check against datasheet dimensions, occluder simplification, mount zones painted with a brush, glass detected from materials and transparency. |
| Camera modules | Library of modules and lenses; module editor with datasheet fields, projection models (rectilinear, equidistant, equisolid, stereographic, measured Kannala–Brandt), a live r(θ) plot, distortion table, image circle on the sensor, derived values and datasheet-vs-model FOV check. Import from JSON, CSV or OpenCV YAML. |
| Placement | Place on surface (C, Alt-click for a mirrored pair), drag from the library, move gizmo (G) and rotate rings (R), drag along the body with surface snapping, aim at a ground point (Camera ▸ Aim ▸ At ground point, then click in 3D or the top view), linked L/R pairs, scrubbable number fields that accept expressions and units, editable camera spreadsheet with paste from Excel, coordinate frames (rear axle, front axle, front bumper, centre, custom) in m or mm. |
| Analysis | Live, incremental ground coverage on a 5/2/1 cm grid inside an ROI (true offset or polygon): blind area, redundancy, best-camera pixel density, focus, per-camera unique/overlap areas, "if removed" figures, stitch seams, worst single-camera error, obstacle poles and vertical targets. Body occlusion is exact per camera (cube depth maps). |
| Views | 3D (OpenGL, QPainter fallback), top coverage/density/redundancy maps, simulated fisheye camera images, stitched bird's-eye with draggable seams, bowl view, 1/2/4 viewport layouts, photo-matched shading (the active 3D view looks through a calibrated reference photo with the model drawn translucent over it; [ and ] step through the photos). |
| 3D rendering | Shader renderer (GLSL 1.20, falls back to fixed-function GL, then to QPainter): geometry kept on the GPU, smooth normals with crease angles, hemisphere + key + rim lighting with specular, a shadow map for the vehicle on the ground and on itself, contact shadow (ambient occlusion) under the body, a ground plane that fades into the background, MSAA. **Camera light** (View ▸ Overlays, Alt+3): every camera's real lens model and body occlusion evaluated per pixel, so the body and the ground show what each camera sees (overlaps hatched) live while a camera is dragged; the software renderer computes the same on the CPU. Animated view changes; **Render image…** (Ctrl+Shift+R, up to 4× the view size, vector overlays stay crisp) and **Turntable video…** (MP4 with ffmpeg, else PNG frames). |
| 3D surround view (Camera views ▸ bowl) | The view an in-car display shows: camera images projected through each lens onto a bowl around the car (flat floor, curved wall; floor, wall height, reach, profile and squareness are tunable and saved with the project) with the seams, exposure balancing and blending of the bird's-eye view, and the car model in the middle; typical display viewpoints. Images come from a synthetic parking scene rendered through every camera (bay markings, parked cars, a pillar, a bollard, the project's obstacles, a garage wall), from the installation check's captures, or just the camera sectors. Export the bowl as OBJ + texture with per-vertex remap tables (sensor pixel and weight per camera) for an ECU renderer. |
| Parking guidelines (Camera views ▸ steering bar; View ▸ Overlays ▸ Parking guidelines, Alt+G) | What a surround-view display draws while parking, from a kinematic bicycle model of the vehicle (wheelbase and track from the wheel centres, body width without the mirrors from the mesh): static width lines and distance markers, the trajectory of the bumper corners for the steering angle with red / yellow / green distance bands and ticks, and the **swept path** of the whole body with its mirrors. Reverse or forward; a steering slider (left of centre steers left; the views follow while dragging, one undo step per drag) shows the road-wheel and steering-wheel angle and turning radius. Drawn through each camera's real lens (fisheye curvature, image circle, body occlusion, unit calibrations) over the camera images, on the bird's-eye view, in the bowl, and on the ground in the 3D and top views; obstacles inside the swept path are flagged with the distance at which the body (or, for tall ones, a mirror) would touch them. Export per camera for the ECU: a JSON table of pixel polylines with colours and labels for every steering angle from full lock left to full lock right (sensor pixel convention as the remap tables), the ground polylines in the export frame, and a PNG preview over the synthetic parking scene. |
| Studies | Monte Carlo tolerance study with sensitivity tornado and pinned worst cases; evolutionary placement optimiser scored under tolerance with a Pareto front; side-by-side variant comparison with Δ density map; calibration target planner with auto-place and suggestions; reference-photo matching (spin scrubber, landmarks, OpenCV calibration, matched 3D view). |
| Installation check (Calibrate ▸ Installation check…, Ctrl+Shift+I) | One image per camera from a file, a live stream (V4L2 device, GStreamer pipeline, RTSP/HTTP URL; preview and grab) or a synthetic capture rendered from the plan with an optional simulated mounting error. The floor boards are cut out of each fisheye image as rectified pinhole patches (exact lens model), the chessboard corners found with OpenCV, and the installed pose solved by PnP on the bearing rays; boards that do not agree with the rest are dropped and the detection is repeated around the solved pose. Per camera: boards, corners, RMS, Δx Δy Δz, Δψ Δθ Δφ and a verdict against the tolerance (OK / adjust / re-mount). The captures are stitched into the bird's-eye view with the planned and with the measured poses side by side; save the report, the PNG, the remap tables, or apply the measured poses as an *as-built* variant. |
| Unit lens calibration (Calibrate ▸ Lens calibration…, Ctrl+Shift+L) | Every camera can carry its own intrinsics instead of the library lens: calibrate from 10–20 views of a hand-held checkerboard (image files, a live stream, or a synthetic set rendered through a simulated unit), fitted as cv::fisheye (Kannala–Brandt, fx/fy/cx/cy/k1–k4). Boards the raw fisheye distorts too much are found in rectified views and mapped back exactly; poor views are dropped. Shows per-view RMS, corner coverage over the image with advice, parameters against the library lens, and how far the library lens would put a ray on this unit. Import calibrations made elsewhere (OpenCV YAML/XML, ROS camera_info, Kalibr camchain, JSON), matched to cameras by name, topic, file name or serial. The unit calibration replaces the library lens in the analysis, the installation check, the remap tables and every export (which says so). |
| Joint extrinsic solve | The installation check refines all cameras together with each board's placement on the floor (x, y, yaw under a placement prior) in one sparse least-squares problem, so cameras that share a board agree at the seams even when the boards were laid a centimetre or two off. It reports the seam agreement before and after (floor disagreement of corners seen by two cameras), the estimated board positions, names boards that lie beyond their tolerance, and sets aside a board observation no placement explains. |
| Surround view | Bird's-eye remap tables per camera (float32 sensor x/y per output pixel + uint8 blend weight, body mask, JSON header with usage, plain-text manifest) from the planned or measured poses: ownership follows the analysis seams with a linear blend, holes fall back to the best-density camera, and body occlusion is honoured. Exposure balancing (per-camera, per-channel gains from the shared ground, Brown–Lowe) and multi-band (Laplacian pyramid) seam blending. **Reference stitcher** export: a dependency-free C++17 program (`vbev::load / estimate_gains / stitch`, OpenMP) and a numpy + OpenCV Python program for stills or live V4L2/GStreamer/RTSP sources (CUDA remap on a Jetson), both matching Vantage's own stitcher to within a grey level. |
| Objects in the stitched view (heatmap *Objects*; Camera views ▸ Stitch seams) | The bird's-eye view assumes everything lies on the ground, so a standing object is drawn by each camera as a smear away from the camera, and near a seam both cameras' smears can land in the other camera's sector: the object vanishes from the display although both raw images show it. A pole (0.5 m pole, 0.9 m child, 1.8 m adult or any height, 0.3 m across; saved with the project) is stood on every ROI cell and followed through the same ownership and blend as the remap tables: the map shows how much of it the stitched view shows, where it **vanishes at a seam** and where it is **cut off** (its smear leaves the view, or it is taller than the camera), with hotspots, an Issues warning, and the hovered point's smears drawn per camera. **Suggest seams** searches each seam ±15° for angles that make it vanish on less ground without leaving more ground below the minimum density; **Apply** moves them as one undo step. The bird's-eye view can show the same zones and smears (*Standing objects*). Export: `stitched_objects` (JSON in the export frame and units, with the hotspots and the seam advice, and the map as PNG). |
| Load cases (Analysis ▸ Load cases…) | Axle drops per load case turn into body pitch and drop; for each case the blind area, minimum density, worst ground shift of the kerb-calibrated stitched view and worst seam ghost between neighbouring cameras. The preset drops are examples: enter the vehicle's suspension data. |
| 3D blind volume (Analysis ▸ Blind volume…) | Where around the vehicle something can stand unseen by every camera, not just on the ground: the ROI columns are sampled at every height (default 5 cm steps up to 2 m) with the ground analysis' visibility test (lens model, sensor edges, body occlusion). Blind volume and share of the space, the height up to which the ground is covered, and for objects standing around the car (0.5 m pole, 0.9 m child, 1.8 m adult) the area where none is seen and how far from the body it can be. A reach rose (how far from the body something can hide, per azimuth), a height slider over a plan slice, a map of the smallest object seen on each cell, and the blind space drawn in 3D as a translucent cut-away up to the slider's height, on its page and as a 3D overlay (View ▸ Overlays). Export: OBJ surface + JSON (`vantage.blind_volume/1`). |
| Assistant (Window ▸ Assistant, Ctrl+J) | A chat about the open design with a model hosted on Groq (OpenAI-compatible API; GPT-OSS 120B by default, any chat model the key may use, with its reasoning effort). The model reads the design only through Vantage's own tools: the overview, coverage and per-camera figures, the Issues list, one camera in detail, why a point is or is not seen by each camera (field of view, image frame, body occlusion, density), blind and low-density regions, the seams, and the studies' results. It checks a change by re-running the analysis on a copy, runs studies on demand (blind volume, objects in the stitched view with seam advice, load cases, mounting tolerance), compares variants, and searches a camera's pose (pattern search within ranges for the least blind ground, the best minimum density or redundancy, optionally keeping a minimum density; negligible gains are not proposed). It can point at things in the app (select cameras, frame a ground point with a marker, switch heatmaps and overlays, open pages) but can only *propose* design changes: a proposal shows the camera, seam-angle or threshold changes (mirrored partners included) and the predicted effect, and is applied as one undo step, applied to a copy as a **new variant**, or discarded; after applying, the card shows the measured result next to the prediction. **Agent mode** (the Agent switch, or `/goal …`): give a goal and the assistant plans it as a live checklist, works through it with a larger budget (30 rounds, 4 minutes; Esc stops it, and it writes up what it has when the budget runs out) and ends with the best proposal and a report. A **project brief** (requirements and constraints, saved with the project) is sent with every question. Slash commands: `/goal`, `/optimise QR`, `/compare`, `/study`, `/show`, `/review`, `/issues`, `/brief`, `/new`. Answers stream with the model's reasoning (collapsible); every lookup is listed with exactly what was sent. *Ask* in the Issues list explains an issue. Set up in Preferences ▸ Assistant: the API key (tested against Groq; kept in the system keyring with the optional `keyring` package, otherwise in a private file in the settings folder, never in projects; `GROQ_API_KEY` overrides it), the model and the reasoning effort. What leaves the machine: camera poses, lens and module names and analysis figures; never the vehicle model, images or files. |
| Rules | Issues list: blind ground, cable routed length vs module cable, GMSL port conflicts, GMSL2 link and deserializer CSI-2 budget (editable link rate, bit depth and overhead in the Compute inspector), cameras off the body or on forbidden zones, edge clearance, density below threshold. |
| Export (Ctrl+E) | Mount sheet PDF, report HTML/PDF, extrinsics JSON/YAML/CSV, OpenCV fisheye YAML per camera, ROS 2 URDF + static TF launch file, Jetson AGX Orin camera config, bird's-eye remap tables, reference stitcher sources, 3D surround-view bowl, parking guideline tables, objects in the stitched view (JSON + map), blind volume (OBJ + JSON), viewport / bird's-eye / camera-view PNGs, mount points as DXF and STEP. |
| Workflow | Undo/redo with History, command palette (Ctrl+K), keyboard shortcuts (?), workspaces, dark and light themes, UI scale, `.vantage` project files (zip of JSON + mesh + thumbnail), start window with recent projects and templates. |

Conventions: ISO 8855 (x forward, y left, z up), origin at the rear-axle centre on the ground; camera orientation is intrinsic
Z-Y-X yaw/pitch/roll with pitch positive down. Pixel density is √(radial × tangential sampling) / range, in px/cm.
Exported and imported intrinsics use OpenCV pixel coordinates (pixel centres on integers: the centre of a 1920-wide
sensor is cx = 959.5); the remap tables use sensor coordinates with pixel centres at i + 0.5, as their header says.

## Tests

```bash
QT_QPA_PLATFORM=offscreen python3 -m pytest tests -q
```

Core engine, loaders (all formats, malformed files), exporters (OpenCV reprojection checked against cv2.fisheye), icons,
installation check (known mounting errors recovered from synthetic captures), unit lens calibration (a simulated unit
recovered from synthetic board views; every calibration file format), the joint solve (misplaced boards), exposure
balancing and multi-band blending, the C++ and Python reference stitchers (compiled and compared pixel by pixel with
the app's stitcher; the C++ test is skipped without g++), stitching, load cases, link budget, batch mode, the shader
renderer (run headless through EGL in a subprocess and checked against the CPU visibility test; skipped where there is no
EGL OpenGL), the 3D surround view, regression tests for every bug found in review (`test_regressions_*.py`), and GUI
tests that drive the app with real mouse and keyboard events.

## Layout

```
vantage/core   engine: lens models, mesh, loaders, body generator, occlusion, analysis, tolerance, optimiser,
               targets, photos, rules, exporters, project model, synthetic captures (synth), installation
               check (install), joint extrinsic solve (joint), unit lens calibration (calib), stitching,
               exposure balancing and remap tables (stitch), 3D surround-view bowl (bowl), camera visibility
               (visibility), load cases, image sources (capture)
vantage/runtime  reference bird's-eye stitchers shipped with the remap tables (C++17, Python)
vantage/batch  command-line batch mode
vantage/selftest  --self-test: checks an installation without a window
vantage/ui     Qt UI: main window, docks, viewports (render_gl: the shader renderer), plan views, camera images,
               charts, dialogs, workspaces, the bowl controller
tests/         pytest suites
packaging/     PyInstaller spec, build script, icons, Linux install scripts and desktop entry,
               Windows installer (Inno Setup); .github/workflows/build.yml builds them on GitHub
```

Fonts: IBM Plex Sans and IBM Plex Mono, SIL Open Font License 1.1 (`vantage/resources/fonts`).
