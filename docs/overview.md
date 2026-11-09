# Vantage: overview, features and capabilities

**Vantage** is a free, open-source (GPL-3.0) desktop application for **designing, proving, calibrating and checking the
sensor rigs of autonomous-driving data-collection vehicles**. These are the cars fitted with cameras, lidars, radars,
GNSS and IMUs that record the data self-driving models are trained on.

This document describes what the application is, who it is for, and everything it can do. The short version is in
[§1](#1-what-vantage-is). The rest is a reference you can read in any order.

| | |
|---|---|
| Current release | **0.8.1** (2026-11-03). |
| Platforms | Windows 10/11 x64 · macOS 14+ (Apple silicon) · Linux x86_64 · Linux arm64 (NVIDIA Jetson Orin, JetPack 6) |
| Built with | Python 3.10+, PyQt6, NumPy, SciPy, OpenCV, PyOpenGL (GLSL shader renderer) |
| Size | about 188,000 lines of Python; 84 test files, about 2,050 tests; a validation suite of 71 model checks |
| Website and docs | <https://sherin-sef-ai.github.io/Vantage/> |
| Licence | GNU GPL v3 or later. The reference stitchers that ship with the remap tables are Apache-2.0, and everything Vantage exports from your project is yours. |

---

## Contents

1. [What Vantage is](#1-what-vantage-is)
2. [Who it is for and the problem it solves](#2-who-it-is-for-and-the-problem-it-solves)
3. [The lifecycle it covers](#3-the-lifecycle-it-covers)
4. [The sample project](#4-the-sample-project)
5. [Vehicle and rig modelling](#5-vehicle-and-rig-modelling)
6. [Placing and editing sensors](#6-placing-and-editing-sensors)
7. [Coverage analysis, the core engine](#7-coverage-analysis-the-core-engine)
8. [Views, 3D rendering and navigation](#8-views-3d-rendering-and-navigation)
9. [Surround view: bird's-eye, bowl and parking guidelines](#9-surround-view-birds-eye-bowl-and-parking-guidelines)
10. [Engineering studies](#10-engineering-studies)
11. [Design exploration: tolerance, optimiser, variants, experiments](#11-design-exploration-tolerance-optimiser-variants-experiments)
12. [Requirements scorecard and design-review pack](#12-requirements-scorecard-and-design-review-pack)
13. [Calibration and installation](#13-calibration-and-installation)
14. [Field checks: live rig, drive logs, as-built, fleet](#14-field-checks-live-rig-drive-logs-as-built-fleet)
15. [Simulation: synthetic datasets and CARLA](#15-simulation-synthetic-datasets-and-carla)
16. [Exports and data hand-off](#16-exports-and-data-hand-off)
17. [The AI assistant and Copilot](#17-the-ai-assistant-and-copilot)
18. [Automation, CI and batch mode](#18-automation-ci-and-batch-mode)
19. [Trust: validation, provenance and tests](#19-trust-validation-provenance-and-tests)
20. [Everyday workflow and user interface](#20-everyday-workflow-and-user-interface)
21. [Installing and running](#21-installing-and-running)
22. [Conventions](#22-conventions)
23. [Architecture](#23-architecture)
24. [Assumptions and known limits](#24-assumptions-and-known-limits)
25. [Glossary](#25-glossary)

---

## 1. What Vantage is

Vantage is a **sensor-rig planner and checker**: it covers the rig's design, simulation, calibration and field checks in one app.

1. **Design.** Import a 3D model of the vehicle, then put cameras (with their real lenses), lidars, radars, GNSS
   antennas, IMUs and the rig hardware (roof rack, mast, enclosures) on it.
2. **Prove in simulation**, before any hardware is fitted, what the rig will record:
   - what each camera sees through its real lens model;
   - where the body, the rig and the other sensors' housings block it;
   - how many pixels land on the ground and on objects;
   - where cameras overlap;
   - how far each object class is detected in every direction, in every weather;
   - how sensitive all of this is to mounting errors, load and road shape.
3. **Engineer the rest of the rig:**
   - timing against the lidar sweep;
   - data rates and disks;
   - power, fuses and cables;
   - GNSS sky view and lever arms;
   - sun glare and dirt;
   - radar through the bumper paint;
   - printable mounting brackets.
4. **Simulate it end to end** in the CARLA driving simulator, with fully labelled ground truth, injected faults of
   known size, standard scenarios, detector quality per rig and multi-drive campaigns.
5. **Calibrate and install the real rig:**
   - lens calibration per unit;
   - the end-of-line installation check from one image per camera;
   - joint multi-camera extrinsics;
   - lidar–camera calibration;
   - a calibration plan with its predicted precision.
6. **Check the real rig in the field:**
   - live stream health;
   - recorded drives checked against the plan;
   - the as-built coverage;
   - drift across a fleet of vehicles over months.
7. **Hand off to the data pipeline.** Calibration goes out in the formats the tools read (OpenCV, ROS 2, URDF/TF,
   nuScenes, KITTI, Kalibr, COCO, MCAP bags), each checked against the tool that reads it.

The same engine also serves **production surround-view systems**:

- bird's-eye remap tables and reference stitchers;
- a 3D bowl view;
- parking guidelines;
- where standing people vanish at stitch seams.

Two things set it apart:

- **Every figure is traceable.** Each model is validated against an independent reference. Each output carries a
  provenance block, and any result can be re-run and compared number by number.
- **The AI assistant cannot run anything without approval.** It reads the design only through Vantage's own tools and
  only *proposes* changes. Its Copilot turns plain-language requests into plans that you approve before anything runs.

---

## 2. Who it is for and the problem it solves

**Who it is for**

- Perception and data-collection teams building recording vehicles.
- Sensor and calibration engineers.
- Surround-view (SVM) engineers.
- Fleet operations teams who maintain rigs over time.
- Anyone who has to justify a rig layout in a design review.

**The problem.** Rig layouts are usually decided by intuition, CAD screenshots and trial fits. The costly mistakes are
found late, after the hardware is bought, bracketed and calibrated, or even after a dataset has been recorded:

- a blind patch by the front wheel;
- a lidar that spends a third of its beams on the roof;
- a camera whose horizon jumps when the boot is loaded;
- a recorder that cannot keep up;
- a radar behind metallic paint;
- a camera that moved on its mount half-way through a campaign.

**What Vantage does about it.** It turns each of those into a number you can see while you are designing and a check
you can run in the field, with the same models end to end. The plan, the simulation, the installed rig and the
recorded data are all compared against one another.

---

## 3. The lifecycle it covers

```
 Design ─► Analyse ─► Explore ─► Review ─► Simulate ─► Build ─► Calibrate ─► Check ─► Operate ─► Hand off
   │          │          │          │          │          │          │           │         │          │
 vehicle   coverage   tolerance  require-   sample     brackets   lens /      live rig  fleet     calibration
 import,   density,   optimiser  ments      dataset,   harness,   install /   health,   drift,    files for
 sensors,  blind      variants,  score-     CARLA      power,     joint /     drive-log as-built  nuScenes,
 rig parts volume...  experiments card, PDF drives...  recording  lidar-cam   check     variants  KITTI, ROS...
```

| Phase | Menu | Main tools |
|---|---|---|
| Design | Vehicle, Camera | Import wizard, rig parts, sensor libraries, placement tools, camera spreadsheet |
| Analyse | Analysis | Live coverage, the Issues list, 20+ study pages |
| Explore | Analysis | Tolerance Monte Carlo, placement optimiser, variant comparison, designed experiments |
| Review | Analysis ▸ Requirements scorecard | Pass/fail requirements, design-review PDF |
| Simulate | Export ▸ Sample dataset; Simulate | Synthetic nuScenes + ROS 2 dataset; CARLA drives, ground truth, faults, scenarios, campaigns |
| Build | Analysis | Mount brackets (STL), cable harness, power budget, recording budget |
| Calibrate | Calibrate | Lens calibration, installation check, joint solve, lidar–camera calibration, observability |
| Check | Calibrate | Live rig health, drive-log check, as-built check |
| Operate | Calibrate ▸ Fleet | Many vehicles, many dates: drift trends and fleet outliers |
| Hand off | Export | 30+ export formats, automation on every change, CI templates |

---

## 4. The sample project

Vantage opens with a complete sample, the **Tata Tigor EV** (`--sample`). This sample is the reference for every example figure in this document.

**The body** is lofted from the profile measured on a 19-frame studio spin (wheel-centre calibration, 1.6 px RMS).

**The cameras** are e-con Systems **STURDeCAM31** (Sony ISX031, 1920 × 1536, 3.0 µm pixels) with H05R2 wide and H03R2
narrow lenses.

**Five layout variants** are included:

| Variant | Description |
|---|---|
| 6-cam · v3 | Recommended layout: no blind ground within 2 m |
| 4-cam | Best aim found for four cameras |
| 4-cam, mirror cameras 5° off | Tolerance case: mirror units 5° off and 10 cm back |
| 6-cam front-heavy | An extra pair on the front bumper corners |
| Data rig · 6-cam + lidar + radars | Six cameras, a 128-channel roof lidar (Ouster OS1-128) and five radars (one long-range, four corner) |

**Other templates** (`--template`): the vehicle library's 13 real vehicles (core/vehicles.py), by id. Each has its
maker's stated dimensions and sources, and a body that lofts its family's profile to them. Three (Lincoln MKZ, Toyota
Prius, Lexus RX 450h) have freely licensed 3D models that Vantage downloads on request and imports with the stated
figures. The old names sedan, hatchback, suv, van and truck open the MKZ, ZOE, RX 450h, Sprinter and Cascadia; `tigor`
is the sample.

**The studio photos** are not in the repository. With them supplied (via `data/` or `VANTAGE_TIGOR_PHOTOS`), the
photo-matching features work. Without them, the sample opens normally with the photo poses but no images.

---

## 5. Vehicle and rig modelling

### 5.1 Vehicle import (Ctrl+I)

**Formats:**

- `.glb`, `.gltf`, `.obj`, `.stl`, `.ply` and `.fbx` (binary and ASCII);
- `.step` through the optional `gmsh` package.

**The import wizard:**

- **Units:** guessed from the bounding box.
- **Axes:** up and forward axes, and mirroring.
- **Mesh repair:** inside-out meshes are flipped automatically.
- **Wheel centres:** found automatically from wheel parts, or picked on the mesh.
- **Reference:** ground plane and origin preset.
- **Spec check:** against the datasheet dimensions.
- **Simplification:** occluders are simplified.
- **Mount zones:** painted with a brush, as allowed or forbidden.
- **Glass:** detected from materials and transparency.

### 5.2 Rig parts (Vehicle ▸ Add rig part)

The data-collection rig is modelled as simple solids:

- a **roof rack** (two crossbars and a platform);
- a **sensor mast** (a tube with a head 0.5 m above the roof);
- a **sensor enclosure**;
- a **crossbar**.

Each part is sized and placed from the vehicle's roof in one undo step.

**Occluding parts** block cameras exactly as the body does. That applies everywhere:

- the coverage;
- the blind volume;
- the camera images and the synthetic captures;
- the assistant's *why is this point not seen*.

**Mountable parts** carry sensors: you can place a sensor on them and drag it along them with snapping.

Rig parts are exported:

- to the ROS 2 URDF, as fixed links with geometry;
- to the extrinsics, as a `rig` list;
- to the report, as a table.

### 5.3 Camera modules and lenses

**The library** holds 36 camera modules and 52 lenses, every one a real product:

- **e-con Systems** (22 modules):
  - **STURDeCAM** automotive cameras: 20, 21, 25, 29, 31, 31FSH (with a heated fisheye), 34, 39, 57, 81, 84 and 88;
  - **NileCAM** GMSL2 board cameras: 21, 56, 81, 85, 130 and 210;
  - **NeduCAM25**, **NeduCAM81**, **Nedu4CAM57** and **Nedu4CAM88** (FPD-Link).
- **Other makers** (14 modules):
  - **GMSL cameras:** Leopard Imaging LI-IMX390, LI-IMX490 and LI-IMX728; D3 Embedded Discovery PRO IMX390 and
    IMX728; Sekonix SF3324 and SF3325 (NVIDIA DRIVE's AR0231 cameras).
  - **The datasets' cameras:** nuScenes' Basler acA1600-60gc with its 5.5 mm lens, and KITTI's FLIR Flea2 colour and
    greyscale cameras with their 4 mm lens.
  - **Machine-vision cameras:** Teledyne FLIR Blackfly S (GigE and USB 3) and LUCID Triton (IP67 GigE), with a real
    C-mount lens each (Computar M0814-MP2, Fujinon HF6XA-5M).

**What each module records:**

- the maker's stated figures: sensor, resolution, pixel pitch, frame rate, shutter, dynamic range, IP rating,
  temperature, power, size and cables, with the page and the date it was read;
- a figure the maker does not state is left out and listed as not given, never filled in. The power budget counts
  such a camera 0 W and lists it.
- a lens keeps its maker's stated focal length and aperture, and its distortion terms are fitted to the stated field
  of view. A C-mount lens known only by its focal length is modelled as rectilinear at that focal length;
- the source page and the date it was read;
- the figures the vendor does not give.

**What each lens reproduces:** the vendor's stated horizontal, vertical and diagonal field of view, within 0.5°. Focal
lengths are not published, so they are derived from the field of view, and the distortion between the stated angles is
modelled.

**Projection models:**

- rectilinear;
- equidistant;
- equisolid;
- stereographic;
- measured Kannala–Brandt.

**The module editor:**

- the datasheet fields;
- a live r(θ) plot and a distortion table;
- the image circle on the sensor;
- the derived values;
- a datasheet-versus-model field-of-view check;
- the source of every figure.

You can import modules from JSON, CSV or OpenCV YAML.

### 5.4 Lidars and radars (Vehicle ▸ Add lidar / Add radar)

**The sensor library** holds real products only, with the figures their makers state in public datasheets, user
manuals and product pages (read 2026-11-04). Each model's note names its source, says what the maker does not state
and how any modelled figure was read or converted.

**Lidars:**

| Model | Channels and field | Range |
|---|---|---|
| Ouster OS1-128 Rev 7 | 128 channels, ±21.2° | 90 m (10 % target) |
| Ouster OS0-128 Rev 7 | 128 channels, ±45.4° | 35 m (10 %) |
| Ouster OS2-128 Rev 7 | 128 channels, ±11.25° | 200 m (10 %) |
| Hesai OT128 | 128 channels, −25° to +15° | 200 m (10 %) |
| Hesai AT128 | fixed 120° × 25.4° field | 210 m (10 %) |
| Hesai XT32 | 32 channels, 1° spacing | 80 m (10 %) |
| Velodyne VLP-16 | 16 channels, ±15° | up to 100 m |
| Velodyne HDL-32E (nuScenes) | 32 channels, −30.67° to +10.67° | up to 100 m |
| Velodyne HDL-64E S2 and S3 (KITTI) | 64 channels, −24.33° to +2° | 50 m (pavement), 120 m (cars) |
| RoboSense RS-Ruby Plus | 128 channels, −25.1° to +15.04° | 240 m (10 %, on 32 channels) |
| Livox Mid-360 | non-repetitive 360° × 59° | 40 m (10 %) |
| Livox HAP | non-repetitive 120° × 25° | 150 m (10 %) |

**Radars** (77 GHz unless noted; ranges on a 10 m² car):

| Model | Zones | Interface |
|---|---|---|
| Continental ARS 408-21 (nuScenes) | ±9° to 250 m, ±45° to 70 m, ±60° to 20 m | CAN |
| Continental ARS 548 RDI (4D) | ±50° × ±4° to 301 m, ±50° × ±14° to 100 m | 100BASE-T1 |
| Continental SRR 308-21 (24 GHz corner) | ±75° × ±12° to 120 m | CAN |
| smartmicro DRVEGRD 152 (4D) | ±50° × ±10° to 178 m | 100BASE-T1 |
| smartmicro DRVEGRD 171 (4D) | ±50° × ±10° to 180 m | 100BASE-T1 |
| smartmicro DRVEGRD 169 (4D corner) | ±65° × ±7.5° to 95 m | 100BASE-T1 |

No radar datasheet states its time synchronisation, so the sync plan stamps radar data when it reaches the host. No
Ethernet radar states its output data rate: the recording budget counts it as 0 and says so. A CAN radar counts its
bus's 500 kbit/s, an upper bound.

**How a lidar is modelled:** as a channel fan, with:

- beam elevations;
- azimuth step;
- frame rate;
- range on a 10 % target;
- points per second;
- bytes per point;
- spin direction.

**Measured beam angles** can be imported from:

- an Ouster metadata JSON;
- a Hesai angle-correction CSV;
- a Velodyne ROS calibration YAML;
- a plain list.

**How a radar is modelled:** as one or more zones (range on a car, ±azimuth, ±elevation). Smaller targets are reached
at the fourth root of their radar cross-section. A range a datasheet gives on another target is converted the same
way: the SRR 308's 95 m on 6 dBsm is 120 m on the car.

**Placement and display:**

- A lidar goes on top of the highest mountable rig part, or on the roof.
- A radar goes in the middle of the front bumper.
- Both are drawn in 3D (the housing, the lidar's ground rings, the radar's field edges) and in the top view.

**Issues are raised for:**

- a lidar inside the body or a rig part;
- a radar more than 15 cm behind the bumper surface;
- a sensor below the ground;
- a duplicate tag.

### 5.5 GNSS antennas and IMUs

**The library** holds real products (the position is an antenna's phase centre, an IMU's the centre of its axes):

- **antennas:** NovAtel VEXXIS GNSS-850 (survey, 29 dB LNA, phase centre 51.7 mm up) and u-blox ANN-MB-00 (patch,
  28 dB, 8.9 mm). The harness's RF budget uses the antenna's stated LNA gain.
- **dual-antenna GNSS/INS:** OxTS RT3000 v4 (heading 0.04° at 2 m), NovAtel PwrPak7D-E1 (0.08° at 2 m), SBG
  Ellipse-D (0.2° at 2 m).
- **high-grade GNSS/INS:** NovAtel SPAN CPT7 (Honeywell HG4930) and SBG Ekinox-D (0.02° at 4 m).
- **IMUs:** Xsens MTi-630 and the Bosch BMI088 chip.

Gyro and accelerometer noise densities are the stated figures, converted exactly to Kalibr's units (an ARW in °/√h
× π/180/60, a VRW in m/s/√h ÷ 60). No maker states a bias random walk (they state bias stability, which is not one),
so Kalibr's `imu.yaml` takes its documented example there and says so. The synthetic sample dataset draws its IMU
noise from the chosen model's density.

**The figures each model holds:**

- heading error at a stated baseline;
- RTK position error;
- roll and pitch error;
- output rate;
- antenna inputs;
- the accelerometer and gyro noise densities and bias random walks, as Kalibr's `imu.yaml` needs them.

**Automatic placement:**

- the first antenna near the rear of the roof;
- the second near the front, for the longest heading baseline;
- the IMU on the cabin floor above the rear axle.

### 5.6 Housings block cameras

Every lidar, radar, GNSS and IMU housing, and every other camera's module body, blocks each camera like the body does.
Everything sees the housings:

- the coverage, density and blind volume;
- the perception range;
- the synthetic captures and the sample dataset;
- the stitched and bowl views;
- the camera light;
- the installation check.

For each camera, the **share of its image** each housing covers is reported:

| Share of the image | Severity |
|---|---|
| over 2 %, or in the image centre | warning |
| over 10 % | serious |
| the lens inside another housing | critical |

Export: `housings.json` and `.csv`, with the coverage with and without housings.

---

## 6. Placing and editing sensors

**Placement:**

- **Place on surface** (C). Alt-click places a mirrored left/right pair.
- **Drag from the library** onto the vehicle.

**Move (G) and rotate (R) gizmos** work on a camera, sensor or rig part:

- an arrow per axis and a square for the horizontal plane;
- rotation rings;
- the change is read out beside the cursor;
- Ctrl snaps to 1 cm or 5°;
- Shift gives fine steps;
- Esc cancels without an undo step.

**Other ways to place:**

- **Drag along the body** with surface snapping.
- **Aim at a ground point** (Camera ▸ Aim ▸ At ground point).
- **Linked left/right pairs** move together.

**Entering values:**

- **Number fields** can be scrubbed, and accept expressions and units.
- **The camera spreadsheet** is editable and takes paste from Excel.

**Coordinate frames:** rear axle, front axle, front bumper, centre or custom, in m or mm.

---

## 7. Coverage analysis, the core engine

**How it works:**

- **Live and incremental.** The ground coverage is computed on a 5, 2 or 1 cm grid, inside a region of interest (ROI):
  a true offset from the body, or a polygon.
- **Exact occlusion per camera.** The body, the rig parts and the housings are handled exactly, with cube depth maps.

**What it reports:**

- **Blind area.**
- **Redundancy.**
- **Pixel density:** the best camera's px/cm, and focus.
- **Per camera:** the area it sees alone, the overlap areas, and "if removed" figures.
- **Stitch seams** and the worst single-camera error.
- **Obstacles:** poles and vertical targets.

### The Issues list (rules)

Issues are raised live. Clicking one selects its camera or sensor. The rules cover:

- **Coverage and placement:**
  - blind ground;
  - density below the threshold;
  - cameras off the body or on forbidden zones;
  - edge clearance;
  - misplaced lidars, radars and GNSS antennas.
- **Links and cables:**
  - GMSL port conflicts;
  - GMSL2 link and deserializer CSI-2 bandwidth;
  - cable routed length against the module's cable;
  - cables over their link's limit;
  - GNSS cables without RF margin;
  - stereo pairs and trigger lines split over two deserializers.
- **Recording and power:**
  - the recorder's write speed, sensor links, uplink, and a disk set that fills before the shift ends;
  - the DC-DC converter, fuses, alternator, cable drop and engine-off runtime.
- **GNSS:** sky view, multipath and heading baseline.
- **Exposure and materials:**
  - lenses highly exposed to dirt;
  - radars looking into metal or losing 3 dB or more through the fascia;
  - high radar multipath risk;
  - glare and unwiped views behind glass.
- **Critical targets:**
  - a target no camera resolves;
  - a mount tighter than the install tolerance.
- **Road:** a wheel lifting, or the body or a sensor touching the road.
- **Calibration:** cameras the planned calibration boards cannot pin.
- **Fleet:** vehicles out of tolerance.
- **Requirements:** failing must-requirements whose metric is cheap to evaluate.
- **Automation:** dropped automation settings and failed automated runs.

---

## 8. Views, 3D rendering and navigation

### 8.1 Views

**3D viewport:** OpenGL, with a QPainter fallback. The layout can be 1, 2 or 4 viewports.

**Top maps:**

- **Density:** smooth, on a colour-blind-safe blue scale starting at the project's minimum. Ground below the minimum is
  grey and hatched. Iso-lines mark the minimum and the target density.
- **Redundancy:** in three ordered steps.
- **Best camera:** in crisp cells.
- **Blind ground:** red and hatched.
- A legend for every mode.

**Hover read-out** (crosshair):

- the value under the cursor and its state against the thresholds;
- every camera that sees the point, with its px/cm, range and off-axis angle.

Hovering over a camera's bar in the Coverage panel lights up the ground only that camera sees.

**Other views:**

- **Charts**, with exact-value tooltips and labelled thresholds.
- **Simulated fisheye camera images.**
- **Stitched bird's-eye view**, with draggable seams.
- **Bowl view.**
- **Photo-matched shading:** the 3D view looks through a calibrated reference photo, with the model drawn translucent
  over it. The [ and ] keys step through the photos.

### 8.2 The 3D renderer

The renderer is a GLSL shader renderer. It falls back to fixed-function GL, then to QPainter.

**Geometry and materials:**

- **Geometry on the GPU**, with smooth normals and crease angles.
- **Lighting in linear space.** Data colours (heat maps, camera light) keep their exact legend colours.
- **Materials** from the mesh's parts:
  - clear-coated paint, with a broad metallic base lobe, reflecting a studio environment with three long overhead
    softboxes;
  - see-through glass;
  - matte tyres;
  - black trim;
  - lenses;
  - rig metal.

**Shadows and backdrop:**

- **Sky occlusion** darkens wheel arches, mirrors and the joints between sensors and the roof.
- **Shadows** are soft and Poisson-filtered, cast by the body, the rig and every housing, plus a contact shadow under
  the body.
- **A studio backdrop** sits on the real horizon.

**Floor grid**:

- drawn in the ground shader;
- anti-aliased lines every 0.5 m and 1 m, from the display frame's origin;
- the lines fade where they crowd;
- the x axis is red and the y axis green.

**Ground maps** have anti-aliased edges.

**Lighting off**: with Shadows and lighting off, the floor and backdrop stay; only the shadows and sky
occlusion go.

**Labels overlay**: camera tags and target names on small plates. Where plates would overlap, the
selected or nearer one wins.

**Sensors at their real size and pose:**

- a lidar with its window band;
- a radar with its radome;
- a GNSS dome;
- an IMU with its axes;
- each camera module with a lens ring in its colour.

**Overlays** (View ▸ Overlays):

- frustums;
- footprints;
- camera light;
- shadows and lighting;
- occlusion rays;
- seams;
- ROI;
- calibration targets;
- dimensions;
- grid;
- labels;
- blind volume;
- parking guidelines;
- lidar rings and radar fields;
- lidar points;
- GNSS sky;
- mount brackets;
- critical targets;
- cable harness;
- wiper sweep.

The overlays that need more explanation:

- **Camera light** (Alt+3) evaluates every camera's real lens model and body occlusion per pixel. The body and the
  ground show what each camera sees, with overlaps hatched, live while a camera is dragged.
  - **On the body**, the camera's colour tints the diffuse shading only. The clear-coat reflections and highlights
    stay on top, so the panels keep their shape, and a thin rim in the camera's colour marks the silhouette.
  - **Edges:** the image and field edges are anti-aliased over 1.5 screen pixels. Occlusion edges are soft over one
    texel of the occlusion cube map.
  - **On the ground:** inside the analysis ROI the colours are exactly the legend's. Outside it, the light fades over
    6 m to 15 % of its strength, so the floor reads as floor again while you can still see where a camera reaches.
- **Lidar points** shows one simulated sweep per lidar, coloured by range. Beams the vehicle stops are drawn in red;
  on the sample, 36 % of the roof lidar's beams.
- **GNSS sky** draws a dome over each antenna, coloured where the sky is blocked.

**Heat maps:** none, coverage, density, redundancy, and objects in the stitched view.

**Shading:** solid, wireframe and X-ray.

**Output:**

- **Render image…** (Ctrl+Shift+R): up to 4× the view size, with crisp vector overlays.
- **Turntable video…**: MP4 through ffmpeg, or PNG frames.

### 8.3 3D navigation

**Mouse:**

- You orbit about the point under the cursor (a dot marks it), so what you grab stays put.
- The wheel zooms towards the surface under the cursor.

**View bar**: four buttons at the top-left of each 3D view, each opening the same choices as the View
menu:

- **View:** perspective, top, front, side, rear, through a camera; frame; render.
- **Shading.**
- **Overlays:** the menu stays open while you toggle several.
- **Heat map.**

**Navigation control** (top-right):

- an axis ball to drag-orbit, or click an axis to look along it;
- pan and zoom handles;
- frame all;
- perspective or orthographic.

**Touchpad:**

- two fingers orbit;
- Shift with two fingers pans;
- pinch, or Ctrl with two fingers, zooms.

**Keyboard:**

- arrows orbit (Alt for small steps);
- Shift with arrows pans;
- + and − zoom;
- 7 is top, 1 is front, Ctrl+1 is rear, 3 is side;
- 0 is the default view;
- 5 toggles orthographic.

**Mouse styles** (Preferences ▸ 3D navigation): Blender-style (middle-drag orbits) or CAD-style (right-drag orbits).

---

### 8.9 Research tools in the 3D view

Everything here is computed by `core/research_maps.py` and `core/motion_footprint.py` and drawn by
`ui/research_overlays.py`.

- **Legend and live metrics:** the bottom-left card.
- **Picture-in-picture:** a through-the-lens view of the camera under the pointer (`camview.render_camera`), with
  the cursor as a crosshair.
- **Alt-drag:** moves a seam (one undo step, with a badge of Δangle, Δblind and Δvanishing child area) or a guideline
  distance marker.
- **BEV features heat map:** a 200 × 200 grid of 0.5 m cells. The value is the cell's footprint in pixels through
  each seeing camera's lens ÷ stride² (16), summed; cells below 1 are starved.
- **Stereo σ heat map:** σ = √((σθ₁r₁)² + (σθ₂r₂)²) / sin γ for the best camera pair at 0.5 px.
- **Fusion density:** camera pixels per lidar point along each lidar ring.
- **Timing wedges:** the lidar azimuths sampled more than 10 ms from each camera's exposure, from the sync plan.
- **Motion:** the footprints and lidar rings carried by a constant speed and yaw rate over each point's own sample
  time (trigger, rolling-shutter row, mid-exposure, sweep azimuth).
- **Probe tool:** a pedestrian, car box or ChArUco board. For each camera it gives the box in pixels, the incidence,
  dr/dθ and the worst ground shift for a 0.2° extrinsic drift.

### 8.10 Bridges to the AV research stack

The export registry's `_BRIDGES` connect modules that write their own folders; each is a normal format
(`--export <id>`) and none is in the default set:

| Module | Export ids |
|---|---|
| `core/plucker.py` | `plucker_rays` |
| `core/occupancy.py` | `occupancy_density` |
| `core/rectify.py` | `rectify_maps`, `rectify_canonical` |
| `core/rig_gap.py` | `rig_gap`, and Analysis ▸ Compare rigs |
| `core/stress.py` | `stress_test` |
| `core/gsplat_bridge.py` | `gsplat_bridge` |
| `core/viz_bridge.py` | `foxglove_mcap`, plus Stream to Rerun / Foxglove |
| `core/stack_exports.py` | `stack_autoware`, `stack_navsim`, `stack_hugsim`, `stack_mmdet3d`, `stack_openpcdet`, `stack_lerobot` |

`core/reference_rigs.py` holds published AV layouts. Every figure is stated, measured or estimated, with its source;
only rigs with positions are placed.

## 9. Surround view: bird's-eye, bowl and parking guidelines

### 9.1 Bird's-eye stitching and remap tables

**Remap tables** are made per camera. Each holds:

- a float32 sensor x/y per output pixel;
- a uint8 blend weight;
- a body mask;
- a JSON header and a manifest.

They are built from the planned or the measured poses.

**How pixels are assigned:**

- ownership follows the analysis seams, with a linear blend;
- holes fall back to the best-density camera;
- body occlusion is honoured.

**Exposure balancing** uses per-camera, per-channel gains from the shared ground (Brown–Lowe). **Multi-band blending**
uses a Laplacian pyramid.

**Reference stitchers:**

- a dependency-free C++17 program (OpenMP);
- a NumPy + OpenCV Python program for stills or live V4L2, GStreamer or RTSP sources (CUDA remap on a Jetson).

Both match Vantage's own stitcher to within a grey level.

### 9.2 Objects in the stitched view

**Why objects vanish.** A bird's-eye view assumes everything is on the ground. Standing objects become smears, and near
a seam both smears can land in the other camera's sector, so the object vanishes from the display.

**How Vantage finds it:**

1. A pole (0.5 m, 0.9 m child, 1.8 m adult, or any height) is stood on every ROI cell.
2. It is followed through the same ownership and blending as the remap tables.
3. The map shows where it **vanishes at a seam** or is **cut off**.

**Suggest seams** searches ±15° per seam for less vanishing ground without losing density. **Apply** moves the seams in
one undo step.

Claude does the same over MCP. `get_seams` gives each seam's vanish area and advised angle, with every angle tried.
`simulate_changes` compares angles by the vanish area per seam. `edit_project` sets a seam
(`{"op": "set", "target": "seam", "id": "QR|MR", "props": {"angle_deg": -107}}`) or hands it back to the automatic
placement (`"props": {"auto": true}`).

### 9.3 3D surround view (bowl)

**What it is.** The view an in-car display shows: the camera images projected onto a bowl (a flat floor and a curved
wall, all tunable), with seams, exposure balancing and blending, and the car model in the middle.

**Image sources:**

- a synthetic parking scene (bays, parked cars, a pillar, a bollard, a garage wall);
- the installation check's captures;
- camera sectors only.

**Export:** OBJ + texture, with per-vertex remap tables for an ECU renderer.

### 9.4 Parking guidelines

**The model.** A kinematic bicycle model of the vehicle, with the wheelbase and track from the wheel centres and the body
width from the mesh.

**What is drawn:**

- static width lines and distance markers;
- the trajectory of the bumper corners, with red, yellow and green distance bands;
- the **swept path** of the whole body including the mirrors.

**Controls:**

- reverse or forward;
- a steering slider, showing the road-wheel and steering-wheel angles and the turning radius.

**Where the guidelines are drawn:** through each camera's real lens over the camera images, on the bird's-eye view, in
the bowl, and on the ground in the 3D and top views.

**Obstacles** inside the swept path are flagged with the distance at which the body or a mirror would touch them.

**Export, per camera:**

- a JSON table of pixel polylines for every steering angle;
- the ground polylines;
- a PNG preview.

---

## 10. Engineering studies

Each study is a page under **Analysis**. Every study has these in common:

- it runs in the background, usually in under a second;
- it re-runs after edits while its page is open;
- it saves its settings with the project, undoably;
- it feeds the Issues list;
- it exports JSON with a provenance block (plus CSV and PNG);
- it has a batch-mode flag;
- most can be run or read by the assistant.

| Study | The question it answers | Key outputs and examples from the sample | Batch flag |
|---|---|---|---|
| **Load cases and suspension** | How far do the cameras move when the car is loaded? | Masses at points on a body on four corner springs: wheel rate in series with the tyre, bump stops, measured wheel-load curves, anti-roll bars. Per case: corner drops, pitch, roll and every sensor's move in mm and degrees, plus the blind area, stitched-ground shift and seam ghost. Reads suspension files in JSON, YAML or CSV with unit conversion. | `--load-cases`, `--suspension` |
| **Road and body motion** | What does the rig see on a real road, and while braking or cornering? | Ramps, crests, garage ramps, speed bumps, kerbs and crowned roads as one height field; ray-marched visibility; the distance each bump or kerb is first seen; body and sensor clearance; brake dive, squat and roll as horizon shifts in pixels. Sample: 0.8 g braking pitches the body 2.25° and moves F's horizon 36 px. | `--terrain` |
| **Thermal and vibration drift** | How far do the midday sun and the road move each camera and sensor from where it was calibrated? | Hour by hour on the hottest campaign day: each rig part's temperatures from an energy balance, expansion, bowing and the rack against the steel roof; every device's turn and move, camera-lidar and stereo pairs in pixels. On an ISO 8608 road through a half-car: each mount's first mode against the body and wheel-hop bands, flex and blur; cobblestones, an expansion joint, a speed bump. Sample: worst drift 1.1 mm (RD1), pairs under 0.04 px, softest mount 109 Hz; with a roof rack and a 0.5 m mast the lidar's mast mode is 62.5 Hz. | `--thermo-vib` |
| **3D blind volume** | Where can something *stand* unseen, not just lie on the ground? | ROI columns sampled at every height (5 cm steps to 2 m): blind volume, the area where a 0.5 m pole, a child or an adult is unseen, a reach rose, the smallest object seen per cell, a 3D cut-away. Export: OBJ + JSON. | `--blind-volume` |
| **Perception range** | How far out does the rig resolve cars, pedestrians, cyclists and signs, in every direction? | 360 directions × 120 ranges to 250 m, the minimum pixel height through the real lens model, required range per sector with pass/fail, stereo pairs with depth resolution at 10, 20 and 50 m. | `--perception-range` |
| **Weather and night** | What do rain, fog, snow and night cost each sensor? | Eight conditions. Cameras: Koschmieder contrast, ITU-R P.1814, headlamp beam (UN R112 class B) against a photon budget, lens drops. Lidar: two-way extinction. Radar: ITU-R P.838-3 and P.840. Sample: an unlit night leaves pedestrians resolved to 19 m ahead (66 m by day); heavy rain takes the long-range radar from 250 to 195 m. | `--weather`, `--condition` |
| **Critical targets** | Does the rig resolve *this* child 2 m in front of the bumper, and how much mounting error can each camera take? | Per target and camera: share seen, pixels across, focus, lidar points, radars. The inverse: the allowed yaw, pitch, roll, x, y and z error before a target is lost (to 0.01° and 0.1 mm), all six together on a 729-pose grid, and `mounting_envelopes.csv` for mechanical design. | `--critical-targets` |
| **Lidar and radar coverage** | What do the lidars and radars add, and where is perception redundant, single or a gap? | Points on the vehicle per channel and direction, the ground blind zone, ground density, points on objects from the real beam elevations, a redundant / single / gap verdict per sector and class, camera–lidar overlap for labelling. Sample: 34 % of the roof lidar's points land on the roof; raising it 0.6 m loses none. | `--sensor-coverage` |
| **Sync and timing** | When should each camera fire against the spinning lidar? | The sweep crossing per camera (parallax included), triggers centring a rolling-shutter exposure, misalignment per hardware trigger line, Suggest groups / Apply, motion blur and rolling-shutter skew. Sample: six cameras on one line are up to 12.4 ms off; four lines bring that under 1 ms. | `--sync-plan` |
| **Clock tree and time sync** | How far off are the sensors' timestamps when PPS drops out, the grandmaster is lost or an FSYNC pulse is missed, and what does that do to lidar points in the images? | The PPS / gPTP / FSYNC tree with editable oscillators, servos, links and timer ticks (typical values with sources), a fault timeline as a seeded Monte Carlo, every sensor's error against true time and the lidar, and the pixel error through the real lens model. Sample: a 60 s PPS dropout leaves the time server 81 µs off UTC at worst; cameras stay within 25 µs of the lidar (0.04 px at 50 km/h turning 25°/s); one missed FSYNC pulse mislabels one frame by 33 ms (58 px). | `--clock-tree` |
| **Recording budget** | How much data, and can the recorder keep up? | Rates per sensor (raw, JPEG, H.265, lidar packets or clouds), link loads, uplink, write speed with 30 % headroom, GB/h, TB per shift, hours per disk set. Sample: an 8 TB set holds 2 h of six raw cameras. | `--recording-budget` |
| **Power budget** | Does the vehicle's supply carry the rig? | Steady and inrush power per device, DC-DC load on a 12, 24 or 48 V bus, alternator margin, engine-off runtime, enclosure heat, fused circuits, cable drop at constant power with the gauge that fixes it. | `--power-budget` |
| **Cable harness** | How long is every cable, and does it fit its link? | Routes on the body's surface graph (avoiding glass, doors and shut lines) to entry grommets, then inside to the compute box; GMSL2, FPD-Link, automotive Ethernet, CAN FD and RS-422 limits; a GNSS coax RF budget; power-pair drop. Sample: 18 cables, 56.2 m. | `--harness` |
| **GNSS and IMU** | Do the antennas see the sky, and what lever arms does the INS need? | Sky-view ray tracing with each blocker named, multipath near field, heading baseline and error, lever arms (FLU and FRD), a pose error budget per sensor. Sample: 100 % sky, a 1.15 m baseline giving 0.17° heading, sensor positions known to 32 mm horizontally. | `--gnss` |
| **Sun glare and dirt** | When does the sun shine into each camera over a campaign, and which lens gets dirtiest? | The NOAA solar algorithm (checked to 0.003°), location, dates, driving window, time zone, headings from a GPX or CSV track; the share of time with the sun in or near the image, a month × hour matrix, the sun's path in the image; a dirt and spray heuristic per lens. Sample: the front camera has the sun in its image 7.2 % of the driving time in Munich. | `--exposure` |
| **Exposure metering and HDR seams** | What should each camera's auto-exposure meter, and how far do neighbouring cameras' exposures drift apart across the bird's-eye seams? | A metering mask per camera through the real lens (no body, rig, housings, sky or the dark beyond the image circle) as PNG, a 0–15 metering grid and weighted AE rectangles (libargus `setAeRegions`); a clear-sky photometric model over the Sun glare campaign and at low sun, a tunnel exit and night: each camera's exposure and the step across each seam before and after the stitcher's gain balancing. Sample: the masks remove 20–32 % of each image; over a year in Munich the seams step by up to 1.5 EV and balancing leaves under 0.1 EV; a tunnel exit spreads the cameras over 3.7 EV. | `--isp` |
| **Materials, glass and wipers** | What do the bumper paint, the windscreen and the wipers do to the sensors? | 77 GHz multilayer transmission through the fascia and paint (solid, metallic, pearl, repaired), the derated radar range, the fascia's thickness tuning; cameras behind glass (light lost, ray shift, dashboard reflection); a wiper sweep per windscreen camera. Sample: the corner radars lose 2.7 dB (95 → 82 m), and a repaired metallic bumper costs 8.2 dB. | `--materials` |
| **Mount brackets** | Can I print a bracket that holds each camera at exactly its planned pose? | A 3D-printable bracket per camera: hole patterns, a surface-fitted base, VHB or screws, tube clamps for rig bars; checks for image blockage, pose accuracy, wall thickness and stiffness; watertight STL. | `--brackets` |
| **Calibration observability** | How precisely will the planned calibration pin each sensor? | Fisher information per camera (checkerboard, ChArUco, AprilGrid), 1σ per axis, the weakest direction explained in plain words, suggested board moves with their gain, a Monte Carlo *Verify*, lidar–camera fusion density. | `--observability` |

### Risk-weighted coverage (Analysis ▸ Risk-weighted coverage)

**What it asks:** a rig can cover almost all of the ground and still miss where crashes start. The study places the
other road user where it is in the 0.5–4 s before a conflict and tests the cameras there with the perception range's
pixel rule.

**Sources**, each figure traced to its table and page:

- NHTSA's pre-crash scenario statistics: Najm et al. 2007 (DOT HS 810 767), Swanson et al. 2019 (DOT HS 812 745,
  Table 12) and the 2016 pedestrian table (DOT HS 812 312), weighted by crash frequency, cost, fatal crashes or
  functional years lost;
- the 39 scenarios of Euro NCAP's 2026 Crash Avoidance protocols (Frontal, Low Speed, Lane Departure v1.2), with their
  speeds, paths, impact points and turn paths, weighted equally.

**Measured on the sample's data rig** (crash frequency, resolved by a camera): 99 % of the ground but 88 % of the
pedestrian, 84 % of the cyclist, 93 % of the vehicle and 79 % of the motorcyclist pre-crash exposure. The gaps are
front 0–10 m (turning-across-path pedestrians in the last second) and front 60–80+ m (the oncoming tests).

**Limits:** cameras only, clear weather, constant speeds and typical sizes; NHTSA's rows are US crashes mapped to the
Euro NCAP template of the same configuration. A scorecard metric, *risk-weighted coverage ≥ X % for C*, checks it.

---

## 11. Design exploration: tolerance, optimiser, variants, experiments

### Tolerance study

- A **Monte Carlo** study of mounting errors.
- A **sensitivity tornado**.
- **Pinned worst cases.**

### Placement optimiser

**Method:**

- evolutionary search;
- scored under tolerance;
- a Pareto front of options.

**Critical targets** can be required, or unresolved ones weighed in the score. The best three candidates are added as
variants.

**Next release:**

- targets as constraints;
- mirrored camera units switched together;
- locked cameras;
- common tolerance draws.

**Refine with gradients**: the Optimise page's second tab tunes every camera's x, y, z, yaw, pitch
and roll (and optionally the stitch seams' angles) continuously with L-BFGS-B on a smooth surrogate of the coverage
objective (blind ground, ground below the minimum density, unresolved critical targets, objects on a seam), annealed
from soft to sharp. Cameras stay on their mount zone, locked cameras stay, mirrored pairs move together; every layout
is checked with the exact engine and the best exact one kept. On the sample's data rig the blind ground falls from 0.183 to 0.145 m² (0.170 m² with the seam angles free); on the 4-cam variant the ground below 3 px/cm falls from 2.72
to 2.26 m². Batch `--refine-rig` (`--refine-from`).

### Variant comparison

Variants are compared side by side, with a **Δ density map**.

### Designed experiments (Analysis ▸ Experiments)

**Factors:**

- any sensor's x, y, z, yaw, pitch or roll;
- lens or model choice;
- the rig's mount height;
- the ROI;
- the load case;
- the weather.

Mirror partners can follow.

**Responses:** any metric of the requirements scorecard.

**Designs:**

- full factorial;
- one at a time;
- Latin hypercube;
- Sobol/Saltelli;
- Morris;
- Monte Carlo.

**Analysis:**

- Sobol first-order and total indices, with bootstrap intervals. These are checked against the Ishigami function.
- Morris μ* and σ.
- Main effects.
- Response surfaces.
- The Pareto front.
- The best design, added as a variant.
- Monte Carlo uncertainty, with a Wilson share meeting a target.

**Running:**

- in the background;
- every point is cached, so a cancelled run resumes;
- batch: `--experiment spec.json OUT`.

### Design agent (Analysis ▸ Design agent)

**In:** a typed brief and the programme's documents (.txt, .md, .pdf with a built-in reader, .docx, .html). Every
sentence keeps its page and paragraph. Requirements and constraints (a budget, sensor counts, a mounting tolerance) are
read by the brief parser, or by your Groq or Claude key with strict JSON; each row quotes its sentence, a figure that
is not in the quote is rejected, and statements no study can check are listed.

**Out:** 3–7 complete rigs on a Pareto front of cost (your prices: Vantage ships none), steady power, the worst
requirement margin and detection (the learned model's prediction when one is trained, else the geometric resolved share,
labelled so). The search swaps lenses and modules, adds or removes cameras, lidars and radars and re-aims, seeded from
your variants and published reference rigs. Each option is re-checked by every study and in random builds within the
stated tolerance, explains each sensor (what fails without it) and critiques itself. Creating one is one undo step,
logged with why the others were not chosen; large change sets ask for an approval.

**Measured on the sample** (brief: pedestrians at 80 m ahead and 30 m all round, under 300 W, a $6k budget, 2 mm
tolerance; prices entered by a stated rule for the test only): five options, all passing every must requirement and
the robust check; the cheapest drops the lidar and radars the brief does not need and gives F a narrow lens. With the
ODD test fixture as a PDF, no option meets *no blind ground*: the agent says so rather than hiding it. A run of 160
rigs takes 2–3 minutes. The LLM path was tested only against fake servers.

### Learned detection model (Analysis ▸ Learned detection model)

**What it is:** a model that predicts a rig's detection per class and region from its geometry alone, with an
uncertainty that is checked. The features are physics (pixels on the object through the real lens, cameras seeing a
region, body occlusion, lidar points, radar coverage); the model is a Gaussian process per class on the empirical logit
of recall, over a monotone logistic prior in pixels.

**Training data:** perception records (`vantage.perception_record/1`) from Rig perception, Foundation models and Robust
perception, in a store in your data folder (or beside the project). Records export anonymised for pooling.

**Calibration** is measured by leaving out a whole town (every scene set recorded there), a scene set or a rig,
against two baselines (the logistic in pixels alone and a constant). On a campaign of 1050 records (136 rigs, 48 CARLA
scene sets in six towns, noon and sunset), leaving a town out: pedestrian RMSE of recall 0.230 (logistic 0.307,
constant 0.411), 95 % interval coverage 0.965, 68 % coverage 0.705; vehicle 0.239 (0.281, 0.384), 0.959, 0.714. It
beats the baseline in every town. The gain comes from towns and scene sets: with one training town it is no better
than the baseline, an eighth of the records does as well as all of them, and choosing rigs by expected improvement did
not beat random choice. Calibrated for CARLA noon and sunset, YOLOX-S lifted to 3D, camera rigs within the training
resolutions and ranges within 60 m; on whole rigs driven at a higher resolution its 95 % intervals covered 0.90.

**Uses:** predict a variant without simulating it, rank which variants to simulate next (expected improvement), a term
in the placement optimiser (*Learned detection* weight), and the design agent's detection axis.

---

## 12. Requirements scorecard and design-review pack

**What it is.** Analysis ▸ Requirements scorecard turns the programme's written requirements into measurable
pass/fail checks on the app's own studies.

**What a requirement can measure.** Each requirement picks a metric from a registry. The metrics cover:

- perception range per class and sector;
- blind ground;
- density and overlap;
- the Issues count;
- lidar and radar gaps;
- the recording budget (hours per disk, write speed, uplink, link load);
- the power budget (runtime, DC-DC load, alternator, circuits, cable drop);
- blind volume;
- objects vanishing in the stitched view;
- tolerance Monte Carlo;
- GNSS (sky, heading, pose error);
- sun glare and dirt;
- fleet drift;
- brackets;
- the faulty streams, drops, sync and camera rotations of a recorded test drive.

Each requirement also has an operator and a target, is a **must** or a **should**, and can carry a note.

**Ways to add requirements:**

- **Add from the plan**, from what the plan already sets.
- **Templates** of typical data-rig requirements.
- **Suggest from the brief**: Vantage reads the project brief for phrasings such as "pedestrians to 40 m", "≤ 1 ms
  sync" or "8 hours per disk" and proposes them for review. It never adds them by itself.

**Evaluation:**

- cheap metrics are evaluated live;
- **Run** computes the studies the requirements need in the background.

**What it shows:**

- the value;
- the margin;
- the status;
- the evidence, with a link to the study page;
- results marked **stale** after a design change.

**Verdict:** the scorecard passes when every must-requirement passes.

**Export:**

- `vantage.requirements/1` JSON;
- CSV;
- **`review_pack.pdf`**: a title page with the verdict, the scorecard, a page per study with its key figures and plot,
  the Issues list, and every study's assumptions and limits.

**In CI:** batch `--check` exits **5** when a must-requirement fails.

---

## 13. Calibration and installation

### 13.1 Unit lens calibration (Ctrl+Shift+L)

**Input:** 10 to 20 views of a hand-held checkerboard, from image files, a live stream or a simulated unit.

**Fit:** `cv::fisheye` (Kannala–Brandt).

**Handling difficult views:**

- boards that are too distorted in the raw fisheye are found in rectified views and mapped back exactly;
- poor views are dropped.

**What it shows:**

- the RMS per view;
- corner coverage, with advice;
- a comparison with the library lens.

**Importing calibrations made elsewhere:**

- formats: OpenCV YAML or XML, ROS `camera_info`, a Kalibr camchain, JSON;
- matched to cameras by name, ROS topic, file name or serial number.

**Effect:** the unit calibration replaces the library lens everywhere (the analysis, the checks, the remap tables and
every export).

### 13.2 Installation check (Ctrl+Shift+I)

The end-of-line check, from **one image per camera**.

**Image sources:**

- a file;
- a live stream (V4L2, GStreamer, RTSP or HTTP);
- a synthetic capture with an optional simulated mounting error.

**The solve:**

1. Floor boards are cut out of each fisheye image as rectified pinhole patches.
2. The corners are found with OpenCV.
3. The pose is solved by PnP on the bearing rays.
4. Outlier boards are dropped and the detection is repeated around the solved pose.

**Per camera:**

- boards, corners and RMS;
- Δx, Δy, Δz, Δψ, Δθ, Δφ;
- a verdict: **OK / adjust / re-mount**.

**Pick corners** is the fallback when glare, a half-hidden board or a dirty lens defeats the
detection:

- the camera's image is enlarged, with a ring on every board corner where the plan puts it;
- you drag the rings onto the real corners, with a 4× magnifier under the pointer;
- four or more picks give the whole pose, two or three the rotation with the position as planned;
- the picked pose then looks for the boards again, and the boards it finds replace their picks;
- a pick far off the solve is named, as most likely the wrong corner;
- batch mode takes the picks as `--install-picks picks.json`.

**What you get:**

- the captures stitched with the planned and the measured poses, side by side;
- the report;
- PNGs and remap tables;
- or the measured poses applied as an *as-built* variant.

### 13.3 Joint extrinsic solve

**What it does:** refines all the cameras together with each board's placement on the floor (x, y and yaw under a
prior), in one sparse least-squares problem. Cameras sharing a board agree at the seams even when the boards were laid
a centimetre or two off.

**What it reports:**

- seam agreement before and after;
- the estimated board positions;
- the boards beyond their tolerance.

### 13.4 Lidar–camera calibration

**Pair ranking:** camera–lidar pairs are ranked by how much of the camera's image the lidar's beams cover.

**The pose plan** places each board where:

- both sensors see all of it;
- at least four lidar channels cross it;
- its squares are big enough;
- its orientation adds most to pinning all six degrees of freedom.

**Captures:**

- **Synthetic:** the real lens and the real beams, with an optional injected error.
- **A folder of real pairs:** PCD, KITTI `.bin`, nuScenes `.pcd.bin`, PLY, `.npy` or CSV.

**The solve:**

- one robust solve gives T_camera_lidar with its 1σ, the Δ against the plan, and a verdict;
- undetermined directions are named in plain words, for example *"the translation along z is not determined"*.

**What you get:**

- a lidar-on-image overlay as visual proof;
- apply to the lidar, or as a variant;
- JSON export (both directions, matrix and quaternion, covariance).

### 13.5 Calibration observability

How firmly the planned calibration will pin every sensor, before anything is captured. See
[§10](#10-engineering-studies).

---

## 14. Field checks: live rig, drive logs, as-built, fleet

### 14.1 Live rig health (Calibrate ▸ Live rig)

**What it does:** watches every camera's stream at once before a collection drive.

**Sources:**

- RTSP or HTTP;
- V4L2;
- a GStreamer pipeline;
- a video file;
- *Synthetic*: the parking scene rendered through the lens, with noise.

**Per camera:**

- frame rate against the module's;
- jitter;
- dropped frames;
- frame age;
- resolution;
- brightness and clipping;
- sharpness (variance of the Laplacian);
- frozen images;
- a verdict: **OK / warning / fault**, with its reasons.

**For the rig:**

- the cameras that are up;
- the worst frame rate;
- the sync skew between cameras, judged against the sync plan's trigger stagger.

**Over each image:**

- the planned body outline and horizon;
- an edge-alignment drift indicator.

**Reliability:**

- the readers run on their own threads and reconnect with back-off;
- **Save frames** writes the images and `rig_health.json`;
- **Check installation** hands the frames to the installation check.

**Passwords** in stream URLs are never stored in projects. They go to a private `stream-credentials.json` and are shown
as `***`.

**Calibration coach**: a second mode of the Live rig page (also **Calibrate ▸ Calibration coach…**).
While a technician holds a checkerboard in front of one live camera, the coach keeps each sharp, new view and adds up
the Fisher information its corners give on the lens (or, with a lidar, on where the lidar sits); lock bars turn green
as each part reaches the requirement, and a ghost board drawn through the exact lens model shows the next most useful
pose, with the move said in words. Finish hands the views to Lens calibration or Lidar–camera calibration. On the
sample's data rig, following the coach pins the front fisheye's lens to 0.1° in 25 views where five random hand-held
sessions do not in 40; the rear camera–roof lidar pair reaches 0.3° and 10 mm in 8 views against a median of 21.
Batch `--calib-coach`.

### 14.2 Drive-log check

**Input:** a recorded drive (a ROS 2 bag or a nuScenes folder), checked against the plan.

**What it checks:**

- **Streams:** presence, rate, drops, latency, image size, frozen cameras.
- **Data rate:** against the recording budget, with real compression.
- **Sync:** the lidar's phase lock and each camera's trigger error.
- **Rotations:** each camera's rotation relative to the lidar, from depth edges against image edges, with a
  confidence. Proven on CARLA: a front camera's rotation found 90 % of the time from 0.35°, a mirror camera's from
  0.52°, and no clean camera flagged.
- **GNSS/IMU:** gravity, gyro bias at rest, fix shares.

**Output:** the confident rotations can be applied as a corrected variant.

### 14.3 As-built check

The question: *does each real camera see the vehicle where the plan says it does, and does the coverage still hold as
built?*

**Input:** a short drive from the real rig (a ROS 2 bag, a nuScenes folder or image folders).

**The steps:**

1. **Ego mask.** What stays the same while the vehicle moves is the vehicle. Undecidable pixels are left out, not
   guessed.
2. **Against the plan:** the simulated mask through the lens model, the overlap, the outline distance, and the image
   circle.
3. **Pose correction.** The yaw, pitch, roll (and optionally a shift) that align the outlines, with standard errors.
   Axes the outline cannot constrain stay as planned, and a camera that sees too little of the vehicle is *not
   checkable*.
4. **As built.** The coverage analysis is re-run on the corrected copy.

**Verdict:** holds / holds with corrections / does not hold.

**Online self-calibration** (Calibrate ▸ Online self-calibration…): follows every camera's rotation
relative to the lidar through a whole recorded drive, window by window, from the drive's own depth edges against image
edges (the vehicle's outline and the shared ground are optional factors, off by default: on CARLA drives they gave
false knocks). A smoother joins the windows and a model search (BIC) tells a **knock** that stays from an excursion
that comes back and from **flex** with speed, braking, turning or the time of day. On Vantage's synthetic drives five
injected knocks of 0.3–0.8° were all found (size within 0.12°, time within 0.75 s); on a CARLA drive joined from a
clean and a faulted run, the front camera's 0.5° pitch fault came out as a +0.39° knock 0.2 s from the join. Exports
the calibration over time as JSON, a nuScenes folder with one calibrated_sensor row per segment and a ROS 2 /tf
sidecar bag. Batch `--online-calib DRIVE OUT_DIR` (exit 3 on a knock beyond the tolerance).

### 14.4 Fleet

**Vehicles and records.** Many vehicles with the same rig are re-calibrated over months. Records come from:

- installation check reports;
- extrinsics JSON, YAML or CSV;
- nuScenes calibration;
- a Kalibr camchain;
- variants;
- manual entry.

**Analysis:**

- **Per sensor:** the latest Δ and a verdict.
- **Drift:** a least-squares trend per 30 days (with a t-test), the days until the tolerance is exceeded, and jump
  detection.
- **Across the fleet:** medians and robust z-scores, for example *"V4: ML is yawed 2.0° left of the fleet median"*.
- **When most vehicles share an offset, the plan is wrong:** **Variant from the fleet median** makes the as-built plan.

**Views:**

- a vehicle × sensor matrix;
- drift charts;
- findings.

**Export:** `vantage.fleet/1`.

### 14.5 Fleet twin (Calibrate ▸ Fleet twin)

**What it follows:** every vehicle's sensors over time, from the Fleet page's records, online self-calibrations of
recorded drives (imported, or read from a watch folder per vehicle as drives arrive) and rig-health drift (an
indicator, never a measurement).

**The model:** per axis a smooth local linear trend (Harvey), Kalman filter and smoother, fitted across the fleet by
maximum likelihood: each source's σ scale and offset, a drift-rate prior per sensor position. Knocks by a generalised
likelihood-ratio test with a stated false-alarm rate. The forecast is a first-passage probability against the critical
targets' inverse tolerances (else the mounting tolerance): a recalibrate-by date, a schedule, outlier vehicles.

**Back into the design:** the measured drift as tolerance inputs (`fleet_twin_tolerances.json`); *Use as mounting
tolerance* sets the tolerance studies to it, and Robust perception can draw its builds from it.

**Measured:** on 50 real online calibrations of CARLA drives, knocks of 0.45–0.8° were found 9 of 12 times at the
first drive after them, with no false knock; on a labelled long-term fixture using CARLA's real calibration errors,
120-day forecasts covered the truth 92 % of the time at 90 %. The drives' dates and their order on a vehicle are a
fixture: no real fleet data was used.

---

## 15. Simulation: synthetic datasets and CARLA

### 15.1 Sample dataset (Export ▸ Sample dataset)

**What it is:** a short synthetic drive recorded by exactly the planned rig. It is for testing ingestion, labelling
tools and loaders before the hardware exists.

**The scene:** a seeded street or highway, with lanes, kerbs, buildings, trees, lights, signs, parked cars, traffic,
cyclists and pedestrians.

**How each sensor is simulated:**

- **Cameras:** through the real lens model, at the sync plan's trigger times, with optional rolling shutter.
- **Lidars:** column by column at the phase-locked azimuth, with reflectivity, noise, intensity, ring and per-point
  time.
- **Radars:** with Doppler and RCS.
- **IMU:** with lever-arm terms, noise and bias.
- **GNSS:** fixes with RTK-like noise.

**Output:**

- a devkit-loadable **nuScenes** folder: 13 tables, keyframes and sweeps, a map, boxes with visibility;
- a **ROS 2 bag**.

**Verification:**

- checked with the nuScenes devkit (projections match to 0.0001 px) and with rosbags;
- the sample data rig takes about 1.5 min and writes about 270 MB.

### 15.2 Simulate in CARLA (Simulate ▸ CARLA)

**What it does:** drives the planned rig in the CARLA 0.9.16 simulator. Every camera, lidar, radar, GNSS and IMU goes on
the CARLA car nearest the vehicle's size, in CARLA's town, traffic and weather. The drive is recorded synchronously as
nuScenes plus a ROS 2 bag, in the sample dataset's layout.

**How the sensors are rendered:**

- **Fisheyes** as pinhole cube faces remapped through the lens model (within 0.1 px):
  - the seams are feathered;
  - Unreal's vignette is removed in linear light;
  - each camera is metered once before the drive (mean level 118, fixed, every view alike; manual and CARLA's auto
    exposure stay available);
  - each lidar point is placed where the sensor was when it fired.
- **Lidar beams** exactly on CARLA's channels.
- **Radar zones** as CARLA radars.

**Validation:**

- lidar points fall on CARLA's depth image within 0.33 mm;
- the image edges fall on its segmentation within 0.1 px;
- on an RTX 5080, the sample's rig records 6 s of driving in 28 s.

**Running:**

- the server is started headless in a memory-capped scope;
- the results open in the drive-log check;
- **Open run…** reopens earlier drives.

**SUMO traffic**: Eclipse SUMO can drive the other vehicles instead of CARLA's traffic manager,
co-simulated in lockstep through CARLA's own co-simulation package. SUMO's vehicles are CARLA actors, so they are in
the ground truth and the annotations; the net is CARLA's bundled one (Town01, 04, 05; no traffic lights), one
generated from the map and cached, or your own. SUMO must be installed in CARLA's Python (the page shows the command;
Vantage never runs it); walkers stay on CARLA's walker AI. On a 12 s drive SUMO's own time was 4 s (Town04) to 11 s
(Town10HD). Batch `--carla-traffic sumo`, `--sumo-net FILE`.

### 15.3 CARLA ground truth

**Per camera frame:**

- semantic, instance and range images in the output camera's own fisheye geometry;
- tight and full 2D boxes;
- truncation, occlusion and range;
- 3D boxes;
- COCO files.

**Per lidar point:** the nuScenes-lidarseg class and panoptic instance, from a semantic lidar casting the same rays.

**Measured:**

- 602 frames with 10,165 objects in 10 s;
- ray-for-ray lidar agreement of 99.75 %;
- the range images agree with the lidar to a 2.2 cm median.

**Viewer:** a frame-by-frame Ground truth tab.

### 15.4 Faults with known answers

**Injectable faults:**

- a sensor turned or moved on its mount;
- a late clock;
- dropped or frozen frames;
- long lidar or radar ranges.

**What makes them "known answers":**

- every calibration output still states the plan, just as a real faulty rig looks;
- `run.json` records the truth.

**Verified:** a 0.5° pitch shows in the images as 0.5001°.

### 15.5 Scenarios and actors scenes

**Scenarios**, each seeded and repeatable:

- free drive;
- a lead vehicle braking;
- a car cutting in;
- a pedestrian or cyclist crossing;
- parked.

Every event's time and distance is logged.

**Actors scenes:** children, pedestrians, cyclists, cars, cones, signs or any blueprint, placed around the parked ego
within a millimetre.

### 15.6 CARLA live rig

**What it does:** the planned cameras are driven in CARLA in real time and served as MJPEG streams, so the Live rig page
watches them as it would a real rig.

**Faults on a schedule:**

- a camera turned or moved;
- a frozen stream;
- dropped or late frames.

**Measured:** one fisheye at 12.5 % render size runs at 30 fps with 30 ms latency.

### 15.7 Detection quality

**Method:** YOLOX (Apache-2.0, downloaded on request and SHA-256-verified) runs through OpenCV on CARLA drives with
ground truth. Detections are matched as COCO's evaluation does; the AP matches pycocotools.

**Results:**

- recall, precision and AP per camera, class and range;
- each camera's detection range;
- recall against pixel height;
- rig-level recall;
- rig variants compared with paired block-bootstrap intervals.

**Example:** cameras pitched 20° up change vehicle recall at 20–40 m by +12.4 points [+4, +21], and beyond 40 m by
−6.8 [−12, −2].

**Log replay** (Simulate ▸ Log replay): a recorded drive (a nuScenes folder such as a CARLA run, or a ROS 2 bag) is rebuilt as 3D Gaussians from its lidar
sweeps, coloured and refined against its camera images (or a 3DGS `.ply` is imported); objects with 3D boxes are
rebuilt in their boxes. The scene is rendered through the recorded rig, to check it on frames and sweeps it never saw,
and through a proposed rig with its own lenses, body, housings, rolling shutter and lidars; the same detector then
runs on both. On a 10 s CARLA drive of the sample's data rig: 257 000 Gaussians built in about 2 minutes on a GPU,
held-out frames at 20.2 dB PSNR and lidar sweeps at a median range error of 8.4 cm; raising the roof lidar 0.3 m takes
its own-roof hits from 35 % of the beams to none. Against the drive's real frames at the same 50 moments, YOLOX-S
finds 57.6 % of the vehicles on the re-render and 59.2 % of the same 255 vehicles on the real images (cars 74 %
against 81 %). Objects the lidar did not rebuild are not scored: the re-render shows 259 of the 313 vehicles the
ground truth shows (79 of 128 beyond 40 m), which leaves an easier set: its own vehicle recall is 56 % where the real
frames give 49 %. Traffic lights are mostly not rebuilt (63 of 349) and their recall is not assessed. The recorded rig
is rendered with the same rolling shutter as the proposed one (its own when the drive states one, else its modules'),
so identical cameras give identical figures. The images are re-rendered, not real: compare rigs, not absolute figures.
Batch `--log-replay DRIVE_DIR OUT_DIR` (`--replay-variant`).

### 15.8 Simulation campaigns

**What a campaign is:** many CARLA drives, each varied in one known way, every one scored by Vantage's own checks.

**Templates:**

- **Fault detectability:** the detection probability per fault size, false alarms, and the smallest fault found 90 %
  of the time (from a logistic fit with a bootstrap interval).
- **Weather sweep.**
- **Rig A vs B.**

**Running:** campaigns are cached and resumable, and their cost is estimated up front.

**What the first real run did:** it exposed weaknesses in Vantage's own checks, which were fixed at the root and the
drives re-scored.

### 15.9 Critical targets in CARLA

**What it does:** the project's critical targets are stood in CARLA as the nearest actors, then compared camera by
camera with the critical-targets study.

**Results on the sample:**

- 30 of 30 views agree (95 % interval 88.6–100 %);
- the median pixel error is +2.6 %.

### 15.10 Rig perception (Simulate ▸ Rig perception)

**What it measures:** how well a 3D detector finds each class in each region around the car (8 sectors × 0–10 … 80+ m),
per rig, on the same CARLA scenes. Ground truth is every annotated object in range, so an object a rig cannot see
counts against it. Matching and AP are nuScenes'; intervals come from a block bootstrap over time, paired when two
rigs drove the same scene; only differences whose interval excludes zero are findings, with the count expected by
chance.

**Detectors:** YOLOX's boxes lifted to 3D through the exact lens model (ground contact and a class size prior, fused
across cameras, triangulated where rays cross), or a team's own BEV detector as a command approved on this computer.
No pretrained torch BEV model qualified: their weights are trained on non-commercial datasets.

**Measured:** on the round-8 pair (data rig against cameras 20° up, Town10HD) the pitched rig finds 5.1 points more
vehicles [2.8, 7.1] on the fisheye images; on pinhole views cut from the same images the difference disappears. On a
fresh 6-scene set (12 drives, 56 min of CARLA), removing the quarter cameras costs 2.3 points of pedestrian recall
[0.2, 4.9]: 5 of 64 comparisons significant where 3.2 are expected by chance, weak evidence and said so. A COCO
detector on CARLA renders: compare rigs, not absolute performance.

### 15.11 Re-drive (Simulate ▸ Log replay ▸ Re-drive)

**What it answers:** do we lose our data if we change the rig? The recorded drive is re-rendered through the proposed
rig; each pixel's expected error is predicted from nine evidence features (never seen, novel-view angle, distance to a
recorded viewpoint, moving object, sky …) by an additive isotonic fit calibrated on frames the reconstruction never saw.
Each camera gets its share within an error budget, a verdict, the reasons per region, and what the old rig saw that the
new one no longer does. The re-shot drive is written as a nuScenes folder with ground truth, confidence maps and a ROS 2
bag.

**Measured** against the real recording of the 20°-up rig: the expected error matched within 0.7 grey levels on five
of six cameras (the front camera, which sees the car's own shadow, under-predicts by half); vehicle recall on the
re-shot drive 90 % against 86 % real, pedestrians 22 % against 83 % (people are rebuilt as rigid bodies). No real
third-party log was used: none downloadable without registration had camera_info, point clouds, tf and odometry.

### 15.12 Foundation models (Simulate ▸ Foundation models)

**What it measures:** how far a modern detector finds each class with each camera, and how far monocular depth is
usable, on rendered frames with ground truth, beside Vantage's geometric pixels-on-target range. OWLv2 and Grounding
DINO (open vocabulary, your own class queries), YOLOX-S as the baseline and Depth Anything V2 Small, each downloaded on
request from a pinned revision with every file SHA-256 checked (models without a commercial-use licence only on an
opt-in). P(detect) against range by logistic regression with cluster-bootstrap intervals gives R50, R90 and the pixel
height the detector needs.

**Measured** in a CARLA range sweep with metered exposure on the sample's front fisheye (60 drives): YOLOX-S finds half
the pedestrians out to 49 m [44–54] where the pixel rule promises 68 m (it needs about 34 px, not 24), on the side
camera 27 m [23–35] against 60 m, and cyclists only to about 12–14 m against 58–64 m (a lower bound: bike and rider are
matched as one); monocular depth stays within 10 % only to about 5 m. Models trained on photographs, measured on renders.

### 15.13 Robust perception (Simulate ▸ Robust perception)

**What it measures:** the rig's tolerances carried to detection. Each build draws every camera's true pose from the
mounting tolerance (the installation check's, the analysis tolerance or the fleet twin's measured drift), the thermal
study's drift, the road vibration and the calibration residual; the perception keeps its calibration. A rotation
remaps the recorded image exactly through its lens; what lies beyond the recording is marked missing, never invented.
YOLOX runs on every build: recall, BEV error, out-of-view and ghosts per class and region with the band over builds,
optionally Rig perception's 3D metric, and Sobol indices for the source and camera that drive the spread.

**Measured** on the paired CARLA drives (24 builds): with each car calibrated, vehicle rig recall varies 0.834–0.881
between builds and the body's road pitch and heave drive the BEV error; on the design calibration the ghost rate goes
from 0.12 to 0.63 and mounting dominates. One scenario and one seed: build variation, not scene sampling.

---

## 16. Exports and data hand-off

**Export ▸ Export (Ctrl+E)** writes any combination of the following.

**Mechanical and documentation:**

- mount sheet PDF;
- report (HTML or PDF);
- mount points as DXF or STEP;
- mount brackets (STL, JSON and a sheet);
- viewport, bird's-eye and camera-view PNGs.

**Calibration:**

- extrinsics (JSON, YAML or CSV);
- OpenCV fisheye YAML per camera;
- ROS 2 URDF + static TF launch file;
- Jetson AGX Orin camera configuration.

**Datasets** (each checked against the tool that reads it):

- **nuScenes** `sensor` and `calibrated_sensor` tables, checked with the devkit.
- **KITTI:**
  - `calib_cam_to_cam`;
  - `calib_velo_to_cam`;
  - `calib_imu_to_velo`;
  - per-camera object calibration files, read by mmdetection3d and OpenPCDet.

  The KITTI devkit lands points within 0.001 px, and pykitti reads the files exactly.
- **ROS `camera_info`**, parsed exactly by ROS 2. image_proc rectifies within 0.001 px.
- **Kalibr camchain:** pinhole-equi, or double sphere, omni or extended unified for wide lenses (fitted within
  0.25 px). With an IMU, `T_cam_imu` and `imu.yaml` are added.

**Surround view:**

- remap tables;
- the reference stitchers (C++ and Python);
- the bowl (OBJ + texture);
- parking guideline tables;
- objects in the stitched view.

**Studies** (JSON + CSV + PNG):

- blind volume;
- perception range;
- lidar and radar coverage;
- sync;
- recording;
- power;
- GNSS and IMU;
- sun and dirt;
- fleet;
- housings;
- load cases;
- requirements and the review pack;
- observability;
- harness;
- terrain;
- critical targets;
- materials;
- weather.

**Pipelines:** CI templates for GitHub Actions and GitLab CI.

### ROS 2 bags without ROS

**Writing:** rosbag2 MCAP, with zstd, lz4 or no compression, and full indexes and CRCs. The bags open in
`ros2 bag` (Humble and later), Foxglove and rosbags.

**Reading:**

- MCAP and SQLite `.db3`, including split bags;
- streamed;
- the bag's own message definitions are used, so vendor types decode too;
- truncated recordings are read up to the last complete chunk.

---

## 17. The AI assistant and Copilot

**Window ▸ Assistant** (Ctrl+J) opens a chat about the open design.

### 17.1 Providers

Providers are chosen in Preferences ▸ Assistant:

- **Groq** (OpenAI-compatible; GPT-OSS 120B by default).
- **Claude**, through Anthropic's official SDK (`claude-opus-5-5` by default, or `claude-sonnet-5-5`):
  - adaptive thinking;
  - strict, schema-checked tools;
  - prompt caching;
  - server-side compaction;
  - fallback when a request is declined.

**Keys:**

- stored in the system keyring, or in a private settings file;
- never in projects;
- `GROQ_API_KEY` and `ANTHROPIC_API_KEY` override them.

### 17.2 What the assistant can do

**How it reads the design:** only through Vantage's own tools:

- the overview;
- coverage and per-camera figures;
- the Issues list;
- *why a point is or is not seen*;
- blind and low-density regions;
- seams;
- every study and page result.

**What it can do:**

- check a change on a copy;
- run the read-only studies;
- compare variants;
- search a camera's pose;
- read the perception studies' last results (rig perception, re-drive, foundation models, robust perception) and
  compute the cheap ones (risk-weighted coverage, the fleet twin and its tolerances, the learned model's predictions
  and which rigs to simulate next);
- read requirements from a brief and run the design agent on a copy (`read_requirements`, `design_rig`): an option is
  created only by you, on the page.

**What it can change:** nothing directly. It only **proposes** design changes. You apply a proposal as one undo step or
as a new variant, and it is then measured against the prediction.

**Agent mode** (`/goal …`) works towards a design goal with a larger budget.

**Slash commands:**

- `/plan`;
- `/goal`;
- `/optimise QR`;
- `/compare`;
- `/study`;
- `/show`;
- `/review`;
- `/issues`;
- `/brief`;
- `/new`.

### 17.3 The Copilot: plans you approve

**Say what you want run**, for example:

- *"a 10 s drive in Town10 at dusk in rain, 30 cars, a pedestrian crossing, ground truth on, half-size images"*;
- *"a fault campaign: turn F and ML by 0.1 to 1°, three seeds"*;
- *"an experiment: F pitch 25–45° by 5° × the two lenses"*.

**The answer is a plan:** steps of 18 kinds: CARLA drive, campaign, detection, critical targets, live rig,
experiment, sample dataset, study, export, and the perception studies (rig perception, foundation-model range, robust
perception, re-drive, design agent, train the learned model, learned-model campaign, risk-weighted coverage, fleet
twin). An approved study step runs in the background, can be cancelled, and its result lands on the study's page;
the design agent's step never creates a variant. Without a model, `/plan …` maps plain requests to these kinds by
Vantage's own rules.

**What each step shows:**

- every value, in the page's own words;
- where each value came from (your request, the default, the Copilot's choice, or a limit) and why;
- unused settings, greyed out;
- the documented limits that apply;
- an estimate (CARLA minutes, frames, disk).

**Validation:** a plan that does not validate cannot be approved, for example one over the render budget or memory,
using a wrong port, or with a missing model.

**Revisions:** follow-ups such as *"more rain"* or *"halve it"* revise the plan, with a diff per setting.

**Approve and run:**

1. The plan is re-checked.
2. An **approval record** is written into the project: the plan, your words, the model, the settings and their
   SHA-256, and the time. Undo never removes it.
3. Each step runs through its page's own Run.

**Guarantees:**

- an approved plan never changes;
- every output's provenance names the plan;
- **Open on its page** fills the fields only, and runs nothing;
- **Reject** drops the plan;
- plans can be re-run from the stored settings (More ▸ Copilot plans, or `--run-plan ID`) without asking the model
  again.

### 17.4 Recommended actions

**Where they come from:** worked out on your computer from Vantage's own results. Nothing is sent.

**Where they appear:**

- **On every page**, the strip knows what the page is for, what it needs first and where its result is. It leads with
  that page's next step: a missing prerequisite and where to fix it, the page's Run (a plan for CARLA, GPU and training
  pages), up to three of its findings to ask about, the next page of the workflow, its export and *What can I do
  here?*. It refreshes 60 ms after you change page; the model reads the same through `get_page_context`.
- **A live strip** that follows what you are doing. Examples:
  - *"QR alone sees 3.00 m²"*;
  - *"Your last change added 0.155 m² blind ground: undo it"*;
  - *"No server on port 2000: start it"*;
  - *"Generate data where recall is weak"*.
- **The empty chat**, which lists the design's own open points.
- **Next steps chips** after each answer.

**What the buttons do:** they only ask, plan, or open a page. None runs or changes anything by itself.

### 17.5 New in 0.7.0

**Vision (Claude):** the assistant can look at:

- the coverage map;
- a close-up of a gap;
- a camera's synthetic view.

Each picture is drawn from the analysis with a caption of the figures, and shown as an enlargeable thumbnail.

**Automatic model choice:**

- quick questions go to a fast model;
- goals, plans, optimisation and picture questions go to Claude Opus 5.5;
- the fallback is said once;
- each answer is labelled with its model and why, and can be re-asked with the other model.

**Whole-rig optimisation** (`optimise_rig`):

- targets as constraints (density, blind area, camera count, critical targets);
- a time budget of up to 90 s;
- up to five Pareto options, ready to propose.

**`add_camera_options`:** where one more camera or a mirrored pair would close the largest gap.

**Design log:**

- decisions, rejected ideas, constraints and notes, kept in the project;
- the model can only propose entries;
- rejected changes are left out of the recommendations.

**Answer-quality benchmark** (`python -m vantage.assistant.evals`):

- 48 questions in ten categories, graded deterministically against the app's own numbers;
- the cost is shown first, and nothing is sent without `--yes`.

### 17.6 Privacy and cost

**What is sent:**

- your text;
- the plan settings;
- small tool results (poses, model names, figures).

**What is never sent:**

- recordings or point clouds;
- keys;
- stream passwords;
- the automation command;
- local folder paths.

Images are never sent either, except the rendered analysis pictures, which need Claude.

**Controls:**

- **More ▸ What was sent** shows the last request.
- The first request to a provider asks once.
- The usage line shows tokens and the estimated cost.
- An optional daily token cap stops sending.

---

### 17.7 Claude over MCP and the Claude Code panel

Vantage is also an **MCP server** (`vantage --mcp`, standard library only). That means Claude Code or Claude Desktop,
on a Claude subscription, reads and changes rig designs. Add it once with
`claude mcp add --scope user vantage -- vantage --mcp`.

- **With a window open**, Claude works on that window's project. Each change is one undo step named "Claude · …", and
  the reply carries the analysis before and after it. Its proposals, plans and design-log entries appear in the
  assistant panel for the user. The bridge listens on 127.0.0.1 with a token only the user can read (Preferences ▸
  General ▸ Claude).
- **With no window open**, Claude works on a `.vantage` file. It is analysed as the app does it and saved after each
  change, with the original kept as `.bak`.
- **Tools (74).**
  - The assistant's tools.
  - `edit_project`: anything the user can change in the design, stitch seams included, checked as a whole before
    anything changes.
  - `run_batch`: any of batch mode's 94 job options, on the design as it is now, in a run folder beside it.
  - Undo and redo of Claude's own steps only; save, export, recent activity, project data and status.
- **What stays the user's:** locked devices, unit calibrations, stream sources, the fleet, photos, the vehicle model,
  Copilot approvals and the automation settings. CARLA runs, model downloads and the automation command are not
  available to Claude.

The **assistant panel** switches between the in-app assistant (Groq, or Claude through the API) and **Claude Code ·
MCP**, a terminal in the same panel (Linux and macOS, with the optional `pyte` package). Claude Code starts there by
itself and works on that window first.

- **The terminal:** colours, box drawing, copy and paste, scrollback, and the alternate screen and mouse reports of
  full-screen programs (the wheel scrolls Claude's conversation).
- **Calls to action:** the app's recommendations hand their request to Claude Code.
- **History:** Claude Code resumes its own conversations, and every in-app assistant conversation is kept on the
  computer per project, to read, copy, delete or continue.

## 18. Automation, CI and batch mode

### 18.1 Automation (Export ▸ Automation)

**What it does:** regenerates what downstream needs whenever the design changes, then runs your own checks.

**What the project keeps:**

- the export formats;
- the output folder, inside the project's folder;
- the variant;
- the requirements check;
- the trigger: on save, after a quiet period, or by hand.

**The command to run** (`make perception-test`, say):

- is kept only in your own settings, never in the project file;
- is shown in full;
- is approved before its first run, and again after any change.

**Environment variables** the command receives:

- `VANTAGE_EXPORT_DIR`;
- `VANTAGE_PROJECT`;
- `VANTAGE_VARIANT`;
- `VANTAGE_SUMMARY` (every file with its SHA-256, what changed, the verdict);
- `VANTAGE_CHECK_EXIT`.

**Behaviour:**

- a time limit and Stop;
- live output;
- changes detected by content;
- stale files pruned;
- a history of runs.

**Headless:** `--watch project.vantage --on-change "cmd"`.

### 18.2 Batch mode (no window)

Every study and check has a command-line form, for CI boxes and for the Jetson on the vehicle. A selection:

```bash
vantage project.vantage --analyse report.json            # metrics, seams, issues, link budget, targets
vantage project.vantage --export out/ --formats extrinsics_json,opencv_yaml,bev_lut
vantage project.vantage --check review/                  # requirements scorecard + review pack (CI gate)
vantage project.vantage --install-check captures/ --report check.json --as-built built.vantage
vantage project.vantage --calibrate-lens views/ --camera F --board 9x6 --square 60
vantage project.vantage --rig-health 10 health.json       # watch every stream for 10 s
vantage project.vantage --drive-log drive/ report/        # a recorded drive against the plan
vantage project.vantage --as-built-check drive/ report/
vantage project.vantage --perception-range ranges.json --condition night
vantage project.vantage --dataset out/                    # synthetic nuScenes + ROS 2 bag
vantage project.vantage --carla-run drive/ --carla-scenario pedestrian_crossing --carla-ground-truth all
vantage project.vantage --campaign campaign/ --campaign-template fault_detectability
vantage project.vantage --experiment spec.json experiment/
vantage project.vantage --reproduce ranges.json check.json --tolerance 'blind_m2=0.005'
vantage --validate report/                                # every model against its reference
vantage project.vantage --run-plan P3F9A21C plan-run/     # a Copilot plan approved in the app
```

**Exit codes** are designed for CI gates. When several apply, the highest is returned.

| Code | Meaning |
|---|---|
| 0 | Fine |
| 2 | The plan has critical issues (or the study's own critical finding: recorder can't keep up, fuse over rating, antenna without sky, etc.) |
| 3 | A camera must be re-mounted or could not be checked; a stream is faulty; the drive shows a fault; coverage doesn't hold as built |
| 4 | A calibration failed, or an input could not be read |
| 5 | `--check`: a must-requirement fails or cannot be evaluated |
| 6 | `--reproduce` does not reproduce, or a `--validate` check fails |
| 7 | CARLA: no server answers, or the run fails |
| 8 | The detector model is missing or does not verify |
| 9 | A campaign drive failed or is still to do |
| 10 | `--run-plan`: no approved, intact plan of that ID |
| 130 | Interrupted (Ctrl+C) |

---

## 19. Trust: validation, provenance and tests

### 19.1 Validation suite (Help ▸ Validate the models, `vantage --validate`)

**71 quantitative checks** of every model, each against a reference that shares none of its code:

| Model | Reference |
|---|---|
| Lens models | OpenCV |
| Coverage | analytic footprints |
| Occlusion | exact ray casting |
| Rain attenuation | the ITU-R P.838-3 table |
| Headlamps | the UN R112 test points |
| Sun position | the SPA paper |
| Bumper fascia | a measured plate |
| Suspension | hand calculations |
| Observability | Monte Carlo solves |
| Experiments | the Ishigami function |
| Dataset files | their formulae |

**The report:**

- HTML, JSON and CSV;
- per check: the quantity, the reference, the error, the tolerance and pass or fail;
- seeded and reproducible;
- exit 6 on a failure.

**Technical reference:** `site/methods.html` documents each model's equations, assumptions, parameters, sources and
limits.

### 19.2 Provenance and reproducibility

**Every JSON Vantage writes** ends with a `provenance` block. It holds:

- the Vantage version and a hash of its code (with the git commit and whether the code is modified);
- the library versions;
- the platform;
- content hashes of the project and of each of its sections;
- the study settings with every default filled in;
- the seeds;
- the batch command, with paths reduced to file names;
- input file hashes;
- the time;
- a hash of the result.

It never holds a path, user name or host name.

**Re-running a result:** `--reproduce result.json` runs the study again and compares every number, exactly or within a
tolerance. It says why a result does not reproduce:

- another version;
- other code or libraries;
- a changed project section;
- a changed input;
- an edited result.

**Determinism:**

- all randomness is seeded, and a test audits the sources for unseeded random numbers;
- results are bit-for-bit identical across processes.

**Documentation:**

- the `.vantage` format is documented in `docs/project-format.md`, with a JSON Schema;
- `CITATION.cff` says how to cite Vantage.

### 19.3 Tests

**About 1,400 pytest tests** check the models:

- **against closed-form results,** for example lens projection, ring radii and sync crossings;
- **against the external tools that read the output:** the nuScenes devkit, the KITTI devkit, pykitti, Kalibr's reader,
  ROS 2's parsers, image_geometry and pycocotools;
- **against synthetic data with known answers:** injected mounting errors are recovered, and simulated units are
  calibrated.

**Other coverage:**

- a fake `carla` module for the CARLA bridge;
- EGL-headless shader-render checks;
- GUI tests driving the app with real mouse and keyboard events;
- a regression test for every bug found in review.

**Self-test:** `vantage --self-test` checks an installation without opening a window:

- it analyses the sample;
- writes every export;
- writes and reads a ROS 2 bag;
- loads Qt's plugins;
- draws with OpenGL and with the software renderer;
- builds every page.

---

## 20. Everyday workflow and user interface

**Editing:**

- **Undo and redo** with a History panel. Every edit is one undoable step, including the assistant's applied proposals.
- **Command palette** (Ctrl+K).
- **Shortcut sheet** (?).

**Layout and appearance:**

- **Workspaces** and docks: the Outliner, the Inspector, the Issues list, the Coverage panel and the Assistant.
- **Dark and light themes**, and a UI scale. The fonts are IBM Plex Sans and Mono.

**Project files:** **`.vantage` projects** are a zip of JSON, the mesh and a thumbnail, and are documented with a schema.

**Start window:** recent projects and templates.

**Working folder** (default `~/Documents/Vantage`): datasets, validation reports and runs go here and are never
overwritten.

---

## 21. Installing and running

### Downloads

Each download carries its own Python, so nothing else needs installing.

| System | File | Notes |
|---|---|---|
| Windows | `Vantage-<v>-windows-x64-setup.exe` or `.zip` | A per-user install; no administrator rights needed. Unsigned: SmartScreen ▸ More info ▸ Run anyway. |
| macOS | `Vantage-<v>-macos-arm64.zip` | Unsigned: System Settings ▸ Privacy & Security ▸ Open Anyway. |
| Linux x86_64 | `Vantage-<v>-linux-x86_64.tar.gz` | `./install.sh` adds a menu entry, the `.vantage` file type and `~/.local/bin/vantage`. |
| Linux arm64 | `Vantage-<v>-linux-arm64.tar.gz` | Jetson Orin with JetPack 6: runs the live rig and the checks on the vehicle itself. |

`SHA256SUMS.txt` comes with every release.

### From source (Python 3.10+)

```bash
python3 -m pip install .            # extras: [step] [bags] [keyring] [claude]
vantage                             # the window
vantage --sample                    # straight into the Tigor EV sample
vantage car.glb                     # import a model
vantage --install-desktop           # Linux: app-menu entry and .vantage file type
vantage --desktop-shortcut          # Linux: the same plus a desktop launcher
vantage --self-test                 # check the installation without a window
```

**Options:**

- `--theme dark|light|system`;
- `--scale 1.15`;
- `--software-3d`;
- `--template <vehicle id>|tigor` (or the old sedan, hatchback, suv, van, truck);
- `--version`.

### Cameras on a Jetson

**Opening cameras:**

- **V4L2 devices** (`/dev/video0`, as e-con's GMSL drivers expose them);
- **GStreamer pipelines** such as `nvarguscamerasrc … ! appsink`. Vantage runs the pipeline in the system's
  `gst-launch-1.0` (with NVIDIA's plug-ins) and reads the frames back.

**JetPack 5:** too old for the bundle; run from source there.

---

## 22. Conventions

**Vehicle frame:**

- **ISO 8855:** x forward, y left, z up.
- The origin is at the rear-axle centre, on the ground.

**Camera orientation:**

- intrinsic Z-Y-X yaw, pitch and roll;
- **pitch is positive down**.

**Pixel density:** √(radial × tangential sampling) / range, in **px/cm**.

**Intrinsics** use OpenCV pixel coordinates, with pixel centres on integers (for a 1920-wide sensor, cx = 959.5).

**Remap tables** use sensor coordinates with pixel centres at i + 0.5.

**Export frames:**

- the frame can be chosen: rear axle, front axle, front bumper, centre or custom;
- units in m or mm;
- angles in degrees or radians.

---

## 23. Architecture

```
vantage/core        the engine: lens models, mesh and loaders, occlusion, coverage analysis, tolerance, optimiser,
                    every study (perception, weather, terrain, critical, materials, harness, gnss, power, recording,
                    timing, exposure, brackets, observability, experiments, requirements, ...), calibration
                    (install, joint, calib, lidar_calib), field checks (rig_health, drivelog, asbuilt, fleet),
                    stitching and bowl, datasets and ROS 2 bags, provenance, CARLA app side (simrig, groundtruth,
                    detection, simcampaign, critical_carla, carla_live)
vantage/assistant   providers (Groq, Claude), the agent loop, tools, Copilot plans and tools, recommendations,
                    routing, vision, design log (memory), rig tools, the evals benchmark
vantage/validation  the validation suite and its report
vantage/sim         the CARLA bridge (runs in CARLA's own Python), the live-rig server
vantage/runtime     reference bird's-eye stitchers shipped with the remap tables (C++17, Python)
vantage/ui          the Qt UI: main window, docks, 3D viewports and the shader renderer (render_gl), plan views,
                    camera views, charts, dialogs, one workspace page per study
vantage/batch.py    command-line batch mode
vantage/selftest.py --self-test
tests/              pytest suites (tests/fake_carla: a fake carla module)
docs/               the .vantage project format and its JSON Schema
site/               the website: the guide (guide.html), documentation and technical reference (methods.html)
packaging/          PyInstaller spec, build script, icons, Linux install scripts, Windows installer
```

**Design principles that run through the code:**

- **One engine, many consumers.** The same visibility test and lens model feed the analysis, the studies, the
  synthetic images, the checks and the exports.
- **Undo for everything.** Every edit, including applied suggestions and assistant proposals, is one undo step.
- **Typical values are labelled typical.** Library figures carry their source and date. What is a heuristic says so.
- **Nothing outward-facing without approval.** Automation commands, Copilot plans and AI requests each need explicit
  consent.

---

## 24. Assumptions and known limits

Vantage states its limits wherever they apply. The main ones:

**Library figures**

- Every library model is a real product. Its figures are what its maker states publicly (datasheets, manuals, product
  pages), not measurements of your unit; what a maker does not state is left out and listed, never filled in.
- Makers state no radar time synchronisation, no Ethernet radar data rate and no IMU bias random walk: the app says so
  where it needs one. Hesai's datasheets could not be downloaded, so the OT128's and AT128's beam-origin heights and
  the XT32's power are estimates, marked in their notes.
- Replace them with your unit's measured figures.

**Weather model**

- The camera contrast threshold (5 %) is a human observer's, so fog ranges are an **upper bound** for a detector.
- In calibrated CARLA fog, YOLOX found cars at 10–20 m 19 % of the time against 46 % in clear weather, where the model
  predicts no loss.
- The wiper and dirt models are heuristics, with stated weights.

**Live timestamps**

- Live sync skew uses **software timestamps** (when frames reached Vantage), not sensor time.
- The drift indicator is an **indicator**, not a measurement.

**Synthetic data**

- The sample dataset has synthetic geometry and materials.
- It is for **pipeline tests, not training**.

**CARLA**

- CARLA's car is not the project's body, so the as-built check marks CARLA mounts **not checkable**.
- CARLA's rain is water on the lens, not extinction with range.
- CARLA's lidar and radar ignore the weather.
- No traffic lights, kerbs or lane markings can be spawned.
- The cameras need a GPU.
- A run is refused above **200 Mpx per frame**.

**Detector**

- Detection quality uses a COCO detector on CARLA renders.
- It is for **comparing** rigs, not for certifying a production detector.

**Other models**

- Sun studies assume a clear sky and a flat horizon.
- The cable harness does not model the interior beyond its trim routes.

**PyTorch**

- Gradient refinement and log replay need PyTorch, which the downloads leave out for its size; they say so when
  started. Install from source with the `[torch]` extra for them. Log replay is practical only on an NVIDIA GPU.

**Unsigned builds**

- The installers are not signed.
- The OS warnings are expected.

---

**Round 14 (perception, re-drive, design agent, learned model, fleet twin)**

- The detection figures are a COCO detector's (and two open-vocabulary detectors') on CARLA renders: they compare rigs,
  cameras and models; they are not a stack's absolute performance. No qualifying pretrained BEV model was found.
- The learned model was measured on CARLA renders (six towns, noon and sunset, no cyclists, to 60 m) with one
  detector: its intervals hold on an unseen town there, not on real roads, at night or for other detectors. Vantage
  ships no trained model: train it on your own records.
- Re-drive was validated on CARLA drives only; no real third-party log was used.
- The fleet twin's knock and forecast figures come from real CARLA online calibrations arranged in time by a labelled
  fixture: no real fleet data was used.
- The design agent's prices in the measured run were entered for the test by a stated rule; Vantage ships no prices.

## 25. Glossary

| Term | Meaning |
|---|---|
| **ROI** | Region of interest: the ground area around the vehicle the analysis covers |
| **Blind ground** | Ground in the ROI that no camera sees |
| **Pixel density (px/cm)** | How many pixels a camera puts on a centimetre of ground or object |
| **Redundancy** | How many cameras see a point |
| **Seam** | The boundary between two cameras' sectors in the stitched bird's-eye view |
| **Variant** | One layout of the rig within a project; variants can be compared |
| **Footprint** | The ground a camera sees |
| **Camera light** | An overlay that lights the scene with each camera's real field of view, so you see exactly what it sees |
| **Extrinsics / intrinsics** | A sensor's pose on the vehicle / a camera's lens and sensor parameters |
| **Kannala–Brandt** | The fisheye lens model used by OpenCV's `cv::fisheye` |
| **PnP** | Perspective-n-Point: solving a camera's pose from known 3D points and their image positions |
| **GMSL2 / FPD-Link** | Serial camera links over coax, typical in automotive camera systems |
| **Lever arm** | The vector from the IMU to another sensor or antenna |
| **As-built** | The rig as actually installed, as opposed to as planned |
| **Provenance** | The block in every output that records exactly what produced it |
| **Copilot plan** | A run proposed by the assistant, with every setting and its reason, run only after approval |
| **Bowl view** | A 3D surround view: the camera images projected onto a bowl around the car |
| **nuScenes / KITTI** | Public autonomous-driving dataset formats that training pipelines read |
| **MCAP** | The container format of ROS 2 bags used by Vantage |

---

*This overview was written for Vantage 0.7.0. The full, always-current reference is in
the README and on the website: <https://sherin-sef-ai.github.io/Vantage/>.*
