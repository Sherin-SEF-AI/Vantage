# Vantage · surround-view camera planner

A PyQt6 desktop app for planning surround-view (360°) camera rigs on vehicles. Import a 3D model of the car and the camera
specifications, place cameras on the body, and see what each camera sees, where the ground is blind, how dense the pixels
are, where the stitched bird's-eye view seams, and how sensitive the rig is to mounting errors. Export a mount sheet and
calibration files.

It opens with the **Tata Tigor EV** sample: the body is lofted from the profile measured on the 19-frame studio spin
(wheel-centre calibration, 1.6 px RMS), the cameras are e-con Systems STURDeCAM31 (Sony ISX031, 1920 × 1536, 3.0 µm) with
the H05R2 wide and H03R2 narrow lenses, and four layout variants are included.

The studio photos themselves are not part of the repository. To use the photo-matching features with them, put
`Tata-Tigor-EV-360.zip` in `data/` (or next to the project folder) or set `VANTAGE_TIGOR_PHOTOS` to its path; without
it the sample opens normally with the photo poses but no images.

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
`--template sedan|hatchback|suv|van|truck|tigor`.

### Batch mode (no window)

```bash
python3 run_vantage.py project.vantage --analyse report.json              # metrics, seams, issues, link budget, targets
python3 run_vantage.py project.vantage --load-cases -                     # load cases as JSON on stdout
python3 run_vantage.py project.vantage --export out/ --formats extrinsics_json,opencv_yaml,bev_lut
python3 run_vantage.py project.vantage --install-check captures/ --report check.json \
        --lut-out out/bev_lut --as-built built.vantage                    # end-of-line check from one image per camera
```

`--variant NAME|N` picks a variant, `--frame`, `--units m|mm` and `--angles deg|rad` set the export conventions, and
`--pos-tol`, `--ang-tol`, `--min-boards` the installation tolerances. Captures are matched to cameras by tag in the file
name (`F.png`, `cam_QR.jpg`, `ML-capture.tif`). Exit code: 0 fine, 2 the plan has critical issues (with `--analyse` or
`--export`), 3 a camera has to be re-mounted or could not be checked.

## What is in it

| Area | Features |
|---|---|
| Vehicle import (Ctrl+I) | .glb .gltf .obj .stl .ply .fbx (binary and ASCII), .step via the optional gmsh package. Wizard: units (guessed from the bounding box), up/forward axes and mirroring, inside-out meshes flipped automatically, wheel centres (auto from wheel parts or picked on the mesh), ground plane, origin preset, spec check against datasheet dimensions, occluder simplification, mount zones painted with a brush, glass detected from materials and transparency. |
| Camera modules | Library of modules and lenses; module editor with datasheet fields, projection models (rectilinear, equidistant, equisolid, stereographic, measured Kannala–Brandt), a live r(θ) plot, distortion table, image circle on the sensor, derived values and datasheet-vs-model FOV check. Import from JSON, CSV or OpenCV YAML. |
| Placement | Place on surface (C, Alt-click for a mirrored pair), drag from the library, move gizmo (G) and rotate rings (R), drag along the body with surface snapping, aim at a ground point (Camera ▸ Aim ▸ At ground point, then click in 3D or the top view), linked L/R pairs, scrubbable number fields that accept expressions and units, editable camera spreadsheet with paste from Excel, coordinate frames (rear axle, front axle, front bumper, centre, custom) in m or mm. |
| Analysis | Live, incremental ground coverage on a 5/2/1 cm grid inside an ROI (true offset or polygon): blind area, redundancy, best-camera pixel density, focus, per-camera unique/overlap areas, "if removed" figures, stitch seams, worst single-camera error, obstacle poles and vertical targets. Body occlusion is exact per camera (cube depth maps). |
| Views | 3D (OpenGL, QPainter fallback), top coverage/density/redundancy maps, simulated fisheye camera images, stitched bird's-eye with draggable seams, bowl view, 1/2/4 viewport layouts, photo-matched shading (the active 3D view looks through a calibrated reference photo with the model drawn translucent over it; [ and ] step through the photos). |
| Studies | Monte Carlo tolerance study with sensitivity tornado and pinned worst cases; evolutionary placement optimiser scored under tolerance with a Pareto front; side-by-side variant comparison with Δ density map; calibration target planner with auto-place and suggestions; reference-photo matching (spin scrubber, landmarks, OpenCV calibration, matched 3D view). |
| Installation check (Calibrate ▸ Installation check…, Ctrl+Shift+I) | One image per camera from a file, a live stream (V4L2 device, GStreamer pipeline, RTSP/HTTP URL; preview and grab) or a synthetic capture rendered from the plan with an optional simulated mounting error. The floor boards are cut out of each fisheye image as rectified pinhole patches (exact lens model), the chessboard corners found with OpenCV, and the installed pose solved by PnP on the bearing rays; boards that do not agree with the rest are dropped and the detection is repeated around the solved pose. Per camera: boards, corners, RMS, Δx Δy Δz, Δψ Δθ Δφ and a verdict against the tolerance (OK / adjust / re-mount). The captures are stitched into the bird's-eye view with the planned and with the measured poses side by side; save the report, the PNG, the remap tables, or apply the measured poses as an *as-built* variant. |
| Surround view | Bird's-eye remap tables per camera (float32 sensor x/y per output pixel + uint8 blend weight, body mask, JSON header with usage) from the planned or measured poses: ownership follows the analysis seams with a linear blend, holes fall back to the best-density camera, and body occlusion is honoured. |
| Load cases (Analysis ▸ Load cases…) | Axle drops per load case turn into body pitch and drop; for each case the blind area, minimum density, worst ground shift of the kerb-calibrated stitched view and worst seam ghost between neighbouring cameras. The preset drops are examples: enter the vehicle's suspension data. |
| Rules | Issues list: blind ground, cable routed length vs module cable, GMSL port conflicts, GMSL2 link and deserializer CSI-2 budget (editable link rate, bit depth and overhead in the Compute inspector), cameras off the body or on forbidden zones, edge clearance, density below threshold. |
| Export (Ctrl+E) | Mount sheet PDF, report HTML/PDF, extrinsics JSON/YAML/CSV, OpenCV fisheye YAML per camera, ROS 2 URDF + static TF launch file, Jetson AGX Orin camera config, bird's-eye remap tables, viewport / bird's-eye / camera-view PNGs, mount points as DXF and STEP. |
| Workflow | Undo/redo with History, command palette (Ctrl+K), keyboard shortcuts (?), workspaces, dark and light themes, UI scale, `.vantage` project files (zip of JSON + mesh + thumbnail), start window with recent projects and templates. |

Conventions: ISO 8855 (x forward, y left, z up), origin at the rear-axle centre on the ground; camera orientation is intrinsic
Z-Y-X yaw/pitch/roll with pitch positive down. Pixel density is √(radial × tangential sampling) / range, in px/cm.

## Tests

```bash
QT_QPA_PLATFORM=offscreen python3 -m pytest tests -q
```

Core engine, loaders (all formats, malformed files), exporters (OpenCV reprojection checked against cv2.fisheye), icons,
installation check (known mounting errors recovered from synthetic captures), stitching, load cases, link budget, batch mode,
and GUI tests that drive the app with real mouse and keyboard events.

## Layout

```
vantage/core   engine: lens models, mesh, loaders, body generator, occlusion, analysis, tolerance, optimiser,
               targets, photos, rules, exporters, project model, synthetic captures (synth), installation
               check (install), stitching and remap tables (stitch), load cases, image sources (capture)
vantage/batch  command-line batch mode
vantage/ui     Qt UI: main window, docks, viewports, plan views, camera images, charts, dialogs, workspaces
tests/         pytest suites
```

Fonts: IBM Plex Sans and IBM Plex Mono, SIL Open Font License 1.1 (`vantage/resources/fonts`).
