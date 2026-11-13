# Vantage · sensor rig planner for autonomous-driving data collection

A desktop app for designing the sensor rigs of data-collection vehicles, the cars that record the camera data
self-driving models are trained on. Import a 3D model of the vehicle, place the cameras, lidars and radars on it, and
prove in simulation what the rig will record before any hardware is fitted. Then calibrate the real cameras, check the
installed rig and its recorded drives against the plan, and hand the calibration to the data pipeline.

**Website and documentation:** <https://sherin-sef-ai.github.io/Vantage/>, with feature videos recorded from the app,
[a step-by-step guide with an example for every feature](https://sherin-sef-ai.github.io/Vantage/guide.html) and the
full reference. A 70-minute tour of the app is on YouTube: <https://youtu.be/s0TM2iDErBI>.

## What it does

- **Plan the rig.** Each camera is modelled through its real lens: what it sees, where the body, the rig and the other
  sensors' housings block it, and how many pixels land on the ground and on objects. You also see where cameras overlap,
  and how sensitive the layout is to mounting errors, load and road motion. Lidars, radars, GNSS and IMUs sit on the
  same rig, using real products from their makers' datasheets.
- **Engineer it.** Brackets ready to 3D-print, the cable harness, the power budget, the clock tree and trigger plan, and
  the recording budget. A requirements scorecard you can run as a CI gate.
- **Drive it in CARLA.** Every frame comes labelled (semantic, instance, range, 2D and 3D boxes) through each fisheye's
  own lens model. Inject faults of known size, run standard scenarios and campaigns, and measure detection quality per
  rig and per region around the car.
- **Calibrate and verify.** Unit lens calibration, camera-lidar calibration with an observability plan, an installation
  check, online calibration over a recorded drive, and a drive-log check of the real rig against the plan.
- **Hand it off.** Exports to OpenCV, ROS 2 (URDF, TF and bags), Kalibr, nuScenes, KITTI, Autoware and more. Each one is
  checked against the tool that reads it, and every number carries provenance.
- **Ask for it.** The **Copilot** (Groq or Claude, with your own key) turns a request into a plan with every value and
  its reason, and nothing runs until you approve it. Claude Code and Claude Desktop can also work on a design over MCP
  (97 tools, resources, prompts and background jobs).

The same engine serves production surround-view systems: stitched bird's-eye remap tables, a 3D bowl view, parking
guidelines, and where standing people vanish at stitch seams.

Every feature, with the figures measured with it, is listed in [docs/features.md](docs/features.md).

## The sample

Vantage opens with the **Tata Tigor EV** sample:
- **Body:** lofted from the profile measured on a 19-frame studio spin (wheel-centre calibration, 1.6 px RMS).
- **Cameras:** e-con Systems STURDeCAM31 (Sony ISX031, 1920 × 1536, 3.0 µm), with the H05R2 wide and H03R2 narrow lenses.
- **Variants:** five layouts. The fifth, *Data rig*, adds a 128-channel roof lidar and five radars to the six cameras.

The studio photos are not part of the repository. To use the photo-matching features with them, put
`Tata-Tigor-EV-360.zip` in `data/` (or next to the project folder), or set `VANTAGE_TIGOR_PHOTOS` to its path. Without
it, the sample opens normally with the photo poses but no images.

## Install

Download the file for your system from the [latest release](https://github.com/Sherin-SEF-AI/Vantage/releases/latest).
Each one carries its own Python and libraries, so there is nothing else to install.

| System | File | Needs |
|---|---|---|
| Windows | `Vantage-<version>-windows-x64-setup.exe` (installer) or `…-windows-x64.zip` | Windows 10 or 11, x64 |
| macOS | `Vantage-<version>-macos-arm64.zip` | macOS 14 or later, Apple silicon |
| Linux | `Vantage-<version>-linux-x86_64.tar.gz` | x86_64, glibc 2.35 or later (Ubuntu 22.04, Debian 12, Fedora 36 and newer) |
| Linux arm64 | `Vantage-<version>-linux-arm64.tar.gz` | arm64, glibc 2.35 or later: NVIDIA Jetson Orin with JetPack 6, Ubuntu 22.04 arm64 and newer |

- **Windows:** run the setup. It installs for the current user without administrator rights.
- **macOS:** move `Vantage.app` to Applications. The app is not signed, so the first time, choose **Open Anyway** in
  *System Settings ▸ Privacy & Security*.
- **Linux:** unpack the file and run `./install.sh`.
- **Check the install:** `vantage --self-test` checks the installation without opening a window, and ends with
  `self-test passed`.

**From source, with Python 3.10 or later:** `python3 -m pip install .`, then `vantage`.

**On a Jetson AGX Orin with JetPack 7:** `packaging/jetson/setup.sh` installs from source with PyTorch on the GPU and
the system's GStreamer-enabled OpenCV for GMSL cameras (details in [docs/install.md](docs/install.md)).

Each system's details, GMSL cameras on a Jetson, checking a download and building the downloads are in
[docs/install.md](docs/install.md).

## A first session

The full walk-through, with every feature, where it lives and a worked example, is the guide:
<https://sherin-sef-ai.github.io/Vantage/guide.html> (also in this repository as
[`docs/user-guide.md`](docs/user-guide.md)). In brief, on the bundled sample:

1. **Open the sample** (`vantage --sample`). The active variant, *6-cam · v3*, has six STURDeCAM31 cameras
   (`F`, `R`, `ML`, `MR`, `QL`, `QR`). The Coverage panel reads:
   - 0.010 m² blind out of a 34.46 m² ROI (99.97 % covered);
   - 53.0 % seen by two or more cameras;
   - 3.12 px/cm at the coarsest.

   The Issues tab has three entries: the blind ground, the disk sets per shift and a 0.9 m child vanishing at the
   `ML | QL` seam.
2. **Change a camera** in the inspector, and the analysis follows. Raising `F`'s aim from 40° to 35° makes the blind ground
   0.038 m². `Ctrl+Z` undoes it.
3. **Fix the seams.** *Camera views* (`Ctrl+Shift+B`) ▸ *Stitch seams* ▸ *Suggest seams* moves three seams, and the
   area where the child vanishes falls from 1.22 m² to 1.18 m².
4. **Compare layouts.** *Layout ▸ Compare variants…*: *4-cam* is blind on 0.073 m² with 6.9 % redundancy, and
   *4-cam, mirror cameras 5° off* on 0.623 m².
5. **Add sensors and run the studies** on *Data rig* (`--variant 5`):
   - the front radar hides 10.7 % of `F`'s image (blind ground 0.183 m²);
   - cars are resolved to 70.1 m ahead against the 80 m asked;
   - the worst camera fires 12.41 ms off the lidar sweep;
   - an 8 TB disk set holds 2.0 h.
6. **Sign off and hand over:**
   - **Sign off** with *Analysis ▸ Requirements scorecard* (`--check review/`, exit 5 on a failed must).
   - **Build** with the brackets and harness.
   - **Calibrate and verify** from the Calibrate menu.
   - **Export** the calibration (`--export out/ --formats nuscenes_calib,ros2,opencv_yaml`).

## From the command line

Every analysis, study, export and check also runs without a window, with exit codes for CI:

```bash
vantage project.vantage --analyse report.json                    # metrics, seams, issues, link budget, targets
vantage project.vantage --export out/ --formats opencv_yaml,ros2  # calibration files for the data pipeline
vantage project.vantage --check review/                          # the requirements scorecard: exit 5 on a failed must
vantage --mcp                                                    # an MCP server for Claude Code or Claude Desktop
```

The full reference for batch mode, its exit codes and Claude over MCP is in
[docs/command-line.md](docs/command-line.md).

## Documentation

| Page | What is in it |
|---|---|
| [Guide](https://sherin-sef-ai.github.io/Vantage/guide.html) ([docs/user-guide.md](docs/user-guide.md)) | A step-by-step guide with a worked example for every feature |
| [docs/features.md](docs/features.md) | Every feature, where it is in the app and the figures measured with it |
| [docs/overview.md](docs/overview.md) | The models behind the figures, their references and their known limits |
| [docs/install.md](docs/install.md) | Installing on each system, running from source, building the downloads |
| [docs/install-guide.md](docs/install-guide.md) | Calibrate ▸ Install guide: fitting a rig to a car live, stage by stage, with the mathematics behind it |
| [docs/command-line.md](docs/command-line.md) | Batch mode, exit codes and Claude over MCP |
| [docs/simulator-bridge.md](docs/simulator-bridge.md) | The CARLA bridge: rig file, run folder and ground truth |
| [docs/project-format.md](docs/project-format.md) | The `.vantage` project file and its schema |
| [docs/development.md](docs/development.md) | Running the tests and where the code is |
| [CHANGELOG.md](CHANGELOG.md) | What changed in each version |

`Help ▸ Validate the models…` (or `vantage --validate report/`) checks every model against independent references
(closed forms, OpenCV, ITU-R tables, published examples, exact ray casting, Monte Carlo) and writes the report. To cite
Vantage, use [CITATION.cff](CITATION.cff).

## Licence

Vantage is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as
published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. It is
distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See [LICENSE](LICENSE) for the full text. The downloads also carry
the licences of the libraries they bundle (PyQt6 and Qt, NumPy, SciPy, OpenCV and others) in their `licenses` folder.

**Exceptions, for production use.** The reference bird's-eye stitchers in `vantage/runtime/` (`bev_stitch.cpp`,
`bev_stitch.py`, `CMakeLists.txt` and their README), which the remap-table export copies next to the tables, are
licensed under the [Apache License 2.0](vantage/runtime/LICENSE) instead, so they can be built into proprietary and
production software (an ECU's surround-view pipeline, a recorder) without opening its source. And what Vantage exports
from your project (remap tables, calibration files, URDF and TF, dataset tables, bowl meshes, guideline tables,
brackets, reports, images) is the program's output, not part of Vantage: it is yours, under no licence of Vantage's,
and can be used anywhere.
