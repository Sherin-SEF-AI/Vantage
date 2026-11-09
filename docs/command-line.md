# Vantage from the command line

Batch mode (every analysis, study, export and check without a window, with its exit codes) and Claude over MCP. The same reference is on the website: <https://sherin-sef-ai.github.io/Vantage/docs.html>.

## Batch mode (no window)

```bash
python3 run_vantage.py project.vantage --analyse report.json              # metrics, seams, issues, link budget, targets,
                                                                         # objects in the stitched view + seam advice
python3 run_vantage.py project.vantage --load-cases -                     # load cases as JSON on stdout
python3 run_vantage.py project.vantage --suspension-data corners.csv --suspension cases.json   # load cases from the
                                                                         # suspension and the loads: corner drops, roll
python3 run_vantage.py project.vantage --housings housings.json           # what housings hide in each camera's image
python3 run_vantage.py project.vantage --blind-volume volume.json         # 3D blind volume, unseen objects, reach rose
python3 run_vantage.py project.vantage --perception-range ranges.json --variant 2   # detection range per object and
                                                                         # direction, sector verdicts, stereo pairs
python3 run_vantage.py project.vantage --export out/ --formats extrinsics_json,opencv_yaml,bev_lut
python3 run_vantage.py project.vantage --install-check captures/ --report check.json \
        --lut-out out/bev_lut --as-built built.vantage                    # end-of-line check from one image per camera
python3 run_vantage.py project.vantage --install-check captures/ --install-picks picks.json   # with board corners
                                                                         # picked by hand where glare beats the detection
python3 run_vantage.py project.vantage --calibrate-lens views/ --camera F --board 9x6 --square 60 \
        --serial SN1234 --calib-out F.yaml --save calibrated.vantage      # one camera's own lens, from board views
python3 run_vantage.py project.vantage --import-calib camchain.yaml F.yaml --save calibrated.vantage
python3 run_vantage.py project.vantage --rig-health 10 health.json       # watch every camera's stream for 10 s
python3 run_vantage.py project.vantage --sensor-coverage coverage.json   # lidar/radar reach, blocked beams, verdicts
python3 run_vantage.py project.vantage --sync-plan sync.json             # camera triggers against the lidar sweep
python3 run_vantage.py project.vantage --clock-tree clock/               # every sensor's timestamp error over a fault
                                                                         # scenario, and its pixels in the images
python3 run_vantage.py project.vantage --recording-budget budget.json    # data rate, links, write speed, hours per disk
python3 run_vantage.py project.vantage --power-budget power.json         # loads, DC-DC, fuses, power cables, runtime
python3 run_vantage.py project.vantage --gnss gnss.json                  # antennas' sky, heading baseline, lever arms
python3 run_vantage.py project.vantage --exposure exposure.json          # sun in each camera over a campaign, dirt
python3 run_vantage.py project.vantage --isp metering/                  # AE metering masks, grids and regions,
                                                                         # exposure steps across the seams
python3 run_vantage.py project.vantage --brackets out/                   # a printable STL bracket per camera
python3 run_vantage.py project.vantage --dataset out/                    # a synthetic drive: nuScenes + ROS 2 bag
python3 run_vantage.py project.vantage --drive-log drive/ report/        # a recorded drive against the plan
python3 run_vantage.py project.vantage --as-built-check drive/ report/   # each camera's view of the vehicle in a
                                                                         # recording against the plan; coverage as built
python3 run_vantage.py project.vantage --online-calib drive.mcap online/   # every camera against the lidar over a
                                                                         # drive: knocks, flex, calibration over time
python3 run_vantage.py project.vantage --fleet-import V3 check.json --fleet fleet/ --save project.vantage
python3 run_vantage.py project.vantage --check review/                   # requirements scorecard + review pack (CI)
python3 run_vantage.py project.vantage --lidar-calib out/ --lidar-calib-pair F L1 --lidar-calib-folder boards/
python3 run_vantage.py project.vantage --observability obs/ --verify 30  # how firmly the planned boards pin each
                                                                         # camera, checked by Monte Carlo
python3 run_vantage.py project.vantage --calib-coach coach/             # hand-held board views the calibration
                                                                         # coach needs, against random views
python3 run_vantage.py project.vantage --harness harness.json            # every cable routed along the body
python3 run_vantage.py project.vantage --terrain road.json               # coverage on the road, bumps first seen,
                                                                         # clearance, brake dive, squat and roll
python3 run_vantage.py project.vantage --thermo-vib drift/              # sun warping and road vibration of the rig
python3 run_vantage.py project.vantage --critical-targets envelopes/     # who resolves each critical target, and
                                                                         # each camera's mounting envelope
python3 run_vantage.py project.vantage --materials materials.json        # radars through the fascia, glass, wipers
python3 run_vantage.py project.vantage --weather weather.json            # rain, fog, snow and night compared
python3 run_vantage.py project.vantage --refine-rig refined.json --refine-from 2   # gradient refinement of a
                                                                         # variant's poses, checked exactly
python3 run_vantage.py project.vantage --perception-range night.json --condition night
python3 run_vantage.py --watch project.vantage --on-change "make perception-test"   # regenerate the exports and
                                                                         # run your check on every change
python3 run_vantage.py project.vantage --ci-templates ci/                # GitHub Actions and GitLab CI jobs
python3 run_vantage.py project.vantage --reproduce ranges.json check.json --tolerance 'blind_m2=0.005'
                                                                         # run a result again, compare every number
python3 run_vantage.py project.vantage --experiment experiment_spec.json experiment/   # a designed experiment
python3 run_vantage.py --validate report/ --validate-quick               # every model against its reference
python3 run_vantage.py project.vantage --carla-rig carla/                # the rig as the CARLA bridge reads it
python3 run_vantage.py project.vantage --carla-run drive/                # drive it in a running CARLA server
python3 run_vantage.py project.vantage --carla-run drive/ --carla-scenario pedestrian_crossing \
        --carla-faults faults.json --carla-ground-truth all                # a scenario, faults, ground truth
python3 run_vantage.py project.vantage --carla-run drive/ --carla-traffic sumo --sumo-net generate   # SUMO traffic
python3 run_vantage.py --ground-truth-summary drive/                     # a drive's ground truth in brief
python3 run_vantage.py project.vantage --carla-live 20 live/ --live-cameras F --live-fault '5:{"tag": "F", "deg": [0, 1, 0]}'
                                                                         # the cameras live in CARLA, rig health on them
python3 run_vantage.py project.vantage --detection-quality det/ --detection-recording drive/ --detection-download
python3 run_vantage.py project.vantage --campaign campaign/ --campaign-template fault_detectability
python3 run_vantage.py project.vantage --critical-carla critical/        # critical targets stood in CARLA
python3 run_vantage.py project.vantage --log-replay drive/ replay/ --replay-variant 2   # a recorded drive
                                                                         # re-rendered through another rig
python3 run_vantage.py project.vantage --run-plan P3F9A21C plan-run/     # a Copilot plan you approved in the app
python3 run_vantage.py project.vantage --rig-perception perception/ --perception-recording drive-a/:A \
        --perception-recording drive-b/:B                                # 3D detection per class and region, A vs B
python3 run_vantage.py project.vantage --redrive drive/ redrive/ --redrive-variant 2 --reshoot   # how much of a
                                                                         # recorded drive a new rig can reuse
python3 run_vantage.py project.vantage --read-requirements extraction.json odd.pdf --design design/   # requirements
                                                                         # from documents, then trade-off rigs
python3 run_vantage.py project.vantage --perf-model model/ --perf-save  # train the learned detection model
python3 run_vantage.py project.vantage --suggest-simulations 4          # which variants to simulate next
python3 run_vantage.py project.vantage --foundation-range fm/ --foundation-recording drive/   # detection and depth
                                                                         # range of foundation models per camera
python3 run_vantage.py project.vantage --robust-perception robust/ --robust-recording drive/ --robust-min-recall 0.8
python3 run_vantage.py project.vantage --risk-coverage risk.json --risk-weight fatal --risk-min pedestrian=85
python3 run_vantage.py project.vantage --fleet-twin-online V3 online.json --fleet-twin twin/ --as-of 2026-11-01
```

`--variant NAME|N` picks a variant, `--frame`, `--units m|mm` and `--angles deg|rad` set the export conventions, and
`--pos-tol`, `--ang-tol`, `--min-boards` the installation tolerances. The installation check solves all cameras jointly
with the board placement unless `--no-joint`; `--board-tol MM` is how far a board may lie off its planned place
(default 20). `--install-picks FILE.json` gives board corners picked by hand for cameras the detection fails on, `{"F": [{"board": "T1", "corner": 0, "x": 812.5, "y": 640.0}, …]}` in pixels of the camera's image (corners 0–3 go round the board from its −u, −v corner: the rear right one of a board at yaw 0), or a saved installation report with `picked_corners`. The stitched PNG is exposure-balanced unless `--no-balance`, `--blend multiband` for multi-band seams.
`--rig-health SECONDS [OUT.json]` reads the streams set on the Live rig page (saved in the project), waits for their
first frames, then measures for SECONDS and writes `vantage.rig_health/1` (default stdout). `--condition ID`
(clear, overcast, night, night_lit, light_rain, heavy_rain, fog or snow) sets the weather `--perception-range`,
`--dataset` and `--weather` use (default: the project's). `--watch PROJECT.vantage` regenerates the formats of the project's automation settings
(Export ▸ Automation; `--export DIR`, `--formats` and `--variant` override them) once at the start and after every
change of the file, runs `--on-change` through the shell in the project's folder, and stops after `--watch-runs N`
runs (default: when stopped). The command comes from the command line only, never from the project file.
`--reproduce RESULT.json [OUT.json]` runs again what wrote a result (from its provenance block: the same project given
first, input files next to the result or in this folder) and compares every number, exactly or within
`--tolerance [PATTERN=]VALUE[%]` per quantity. `--experiment SPEC.json OUT_DIR` runs a designed experiment exported
from Analysis ▸ Experiments and resumes from OUT_DIR's journal after an interruption. `--validate [OUT_DIR]` writes
the validation suite's report (default `./validation`; with a project its vehicle and exports are checked too),
`--validate-models lens,weather` picks models, `--validate-quick` is a 30-second smoke run. `--carla-rig OUT_DIR`
writes the variant as the CARLA bridge reads it; `--carla-run OUT_DIR` drives it in the CARLA server already running on
the port of Simulate ▸ CARLA's settings (`--carla-python PY`: CARLA's client Python, else the one the page chose).
With either, `--carla-scenario NAME` and `--carla-scenario-params JSON`, `--carla-faults FAULTS.json`,
`--carla-actors ACTORS.json` (a file or the JSON itself; `none` clears the project's), `--carla-ground-truth
all|none|semantic,instance,range,lidar_labels`, `--carla-walkers N` and `--carla-spawn-point N` go into the rig over
the project's CARLA settings, each checked against the rig first. `--ground-truth-summary RUN_DIR [OUT.json]` reads a
drive's ground truth (no project needed). `--carla-live SECONDS [OUT_DIR]` drives the cameras live (`--live-cameras
F,R`, `--live-scale 0.25`, `--live-fault AT_S:JSON` repeatable) and writes rig health's verdicts (`rig_health.json`) and
the live server's report. `--detection-quality OUT_DIR` evaluates the detector on the project's recordings or on
`--detection-recording RUN_DIR` (repeatable, the first the reference); `--detection-download` fetches the model first.
`--campaign OUT_DIR` runs the project's simulation campaign or `--campaign-template
fault_detectability|weather_sweep|rig_ab`, its drives in the campaign's own folder when it names one, else
`OUT_DIR/drives`, and resumes from what is done; `--campaign-phase drive|evaluate|analyse` runs part of it (only
`drive` needs CARLA). `--critical-carla OUT_DIR` stands the critical targets in CARLA (drives in `OUT_DIR/drives`),
`--critical-carla-drives DIR` compares drives recorded before without CARLA (one drive, or the drives at any depth
under DIR, each named by its path there). `--run-plan ID[:VERSION] [OUT_DIR]` runs a Copilot plan approved in the
app from the approval record in the project (its stored settings; no model is asked), each step through the option
that runs that kind into `OUT_DIR/step-<n>-<kind>` (default: the plan's folder in the working folder), every output's
provenance naming the plan (a perception-study step runs its study headless; `--save` keeps what a fleet-twin watch
step adds); CARLA steps need a server already running, as `--carla-run` does. A modifier without its
option is refused.
`--export` without `--formats` leaves out the experiment, the CARLA rig, the simulation campaign, the detection
quality (minutes, tens of MB), the calibration coach, the gradient refinement, the log replay, rig perception, re-drive,
foundation models, robust perception, the design agent, the learned model and its perception records; the last two export what is on disk (the campaign's drives and checks, the recordings'
cached detections) and drive nothing.
Captures are matched to cameras by tag in the file name (`F.png`, `cam_QR.jpg`, `ML-capture.tif`); imported
calibrations by the camera name or ROS topic in the file, the file name, or the serial number. Exit code: 0 fine, 2 the
plan has critical issues (with `--analyse` or `--export`; with `--recording-budget`, the recorder or a sensor's link
cannot keep up; with `--power-budget`, the DC-DC converter or a fuse is over its rating; with `--gnss`, an antenna has no
sky; with `--brackets`, a camera cannot take a printed bracket as planned; with `--suspension`, a load case lifts a wheel
or exceeds the gross vehicle mass; with `--housings`, a camera sits inside a housing; with `--critical-targets`, no camera
resolves a critical target; with `--materials`, a radar looks into metal over half its field; with `--terrain`, the body
or a sensor would touch the road or no camera resolves a speed bump or kerb; with `--thermo-vib`, a mount resonates in the
body or wheel-hop band or a device drifts in full sun beyond the calibration tolerance; with `--harness`, a cable is longer than its
link allows, no module cable reaches or a GNSS cable has no RF margin; with `--observability` and calibration targets
planned, a camera sees no board whole; with `--calib-coach`, no hand-held board pose fits the camera; with `--refine-rig`, the variant has no vehicle body or
visible camera to refine; with `--isp`, a stitch seam keeps an exposure step over twice the limit after gain
balancing), 3 a camera has to be re-mounted or could not
be checked (or, with `--rig-health`, a camera's stream is faulty or not set; with `--drive-log`, the recording shows a
fault; with `--as-built-check`, the plan's coverage does not hold as built; with `--online-calib`, a camera was knocked
beyond the tolerance; with `--fleet`, a sensor needs re-mounting in its vehicle's latest record; with `--lidar-calib`, the lidar has to be
re-mounted or its pose is not determined; with `--clock-tree`, lidar points land further off than the limit in 95 % of
frames or the grandmaster cannot hold UTC within the time budget), 4 a lens calibration failed or an
imported calibration matched no camera (or did not fit it), or a fleet record, drive log, as-built recording, suspension data file or lidar calibration capture could not be read or the
sample dataset not written (or the `--experiment` spec cannot be read or does not fit the project, or `--reproduce`
cannot try the result, or `--calib-coach` has no visible camera or, in lidar mode, no camera–lidar pair, or `--refine-from` names no
variant, or the `--online-calib` drive cannot be read or no camera has a topic in it), 5 with `--check` a must requirement fails or cannot be evaluated, 6 with `--reproduce` the
result does not reproduce or with `--validate` a check of the models fails, 7 with `--carla-run`, `--carla-live`,
`--campaign` or `--critical-carla` no CARLA server answers, CARLA's client Python is missing or the run fails (or,
with `--carla-traffic sumo` or `--sumo-net`, SUMO is not in CARLA's Python) (2 with
`--carla-rig` or `--carla-run` when the variant has nothing to simulate or a `--carla-*` option does not fit it, or
its cameras would render more than the bridge's 200 Mpx a frame; with
`--ground-truth-summary` when the drive has no ground truth; with `--detection-quality` when there is no recording with
ground truth to evaluate; with `--campaign` when the design has nothing to drive or names a sensor the variant lacks;
with `--critical-carla` when no target can be stood in CARLA or no drive is found; 3 with `--carla-live` when rig health
finds a camera faulty and with `--critical-carla` when a camera view disagrees with the study; 3 with `--log-replay` when the
reconstruction does not validate, 4 when its drive cannot be read), 8 with
`--detection-quality` (or a `--campaign` whose checks include detection) the detector model is missing or does not
verify, 9 with `--campaign` a drive failed or is still to do (run it again: what is done is reused), 10 with
`--run-plan` when the project holds no approved, intact plan of that ID or a step no longer validates (approve it again
in the app; otherwise `--run-plan` returns the first failing step's own code, and a step that reports findings, 3, 5 or
6, lets the plan go on); when several apply, the highest is returned. Ctrl+C during a CARLA run returns 130 once CARLA is tidied up. `--watch` exits with its last run's code: the command's own when it ran, else 5 when the
requirements check fails, 4 when the project could not be read or a format not written, 130 when stopped during the
command.

The perception, design and fleet options (each also on its page; a modifier without its option is refused):

| Option | What it writes | Exit codes |
|---|---|---|
| `--rig-perception OUT_DIR` (`--perception-recording DIR[:RIG]` repeatable, `--perception-backend lifted\|external:NAME`, `--perception-records`) | `rig_perception.json`, region and comparison CSVs, findings, a map per class and rig, perception records | 4 no recording or one unreadable, 8 the 2D detector model missing, 9 the external detector not set up or not approved here |
| `--perception-scene-set SET OUT_DIR` (`--scene-variants`, `--scene-seeds`, `--carla-python`) | the scene set's drives in `OUT_DIR/drives`, `scene_set.json` | 2 the set does not fit, 7 no CARLA server, 9 a drive failed |
| `--redrive DRIVE_DIR OUT_DIR` (`--redrive-variant`, `--reshoot`) | `redrive.json`, camera and region CSVs, polar maps; with `--reshoot` the re-shot drive in `OUT_DIR/reshot` | 3 the proposed rig's verdict is re-collect, 4 the drive cannot be read or PyTorch is missing |
| `--read-requirements OUT.json FILE…` | the brief parser's requirements and constraints, each quoting its sentence; set as the design agent's rows | 2 nothing to read, 4 a document cannot be read |
| `--design OUT_DIR` (`--design-evaluations N`, `--design-seed S`) | the design options (JSON, CSV, sensors CSV, PNG) | 2 no accepted requirement, 5 no option meets every must requirement and constraint |
| `--perf-model OUT_DIR` (`--perf-save`) | the learned model's calibration and predictions; `--perf-save` keeps the model for the page and the optimiser | 3 too few records to train, 4 the store cannot be read |
| `--perf-records-import FILE`, `--perf-records-export FILE` (`--no-anonymise`) | records added to (or written from) the learned model's store, anonymised unless asked | 4 the file cannot be read; 2 nothing to export |
| `--suggest-simulations K` | which variants to evaluate for real next, JSON on stdout | 3 no trained model |
| `--foundation-range OUT_DIR` (`--foundation-recording`, `--foundation-models`, `--foundation-depth`, `--foundation-download`) | ranges per camera, class and model, depth per camera, figures, examples, perception records | 2 nothing to evaluate (the CARLA sweep runs from the page), 4 PyTorch or transformers missing, 8 a model missing |
| `--robust-perception OUT_DIR` (`--robust-recording`, `--robust-mode`, `--robust-samples`, `--robust-sobol`, `--robust-no-detect`, `--robust-min-recall`) | `robust_perception.json`, region and sensitivity CSVs, a figure per recording | 2 no recording, 3 a 5th-percentile rig recall below `--robust-min-recall`, 8 the detector model missing |
| `--risk-coverage [OUT.json\|DIR]` (`--risk-weight`, `--risk-metric`, `--risk-min CLASS=PCT`) | the risk-weighted coverage per class, regions, weights and maps | 3 a class below its `--risk-min` |
| `--fleet-twin OUT_DIR` (`--as-of`), `--fleet-twin-online VEHICLE FILE…` (`--fleet-twin-date`), `--fleet-twin-watch` (`--watch-runs`, `--watch-interval`) | the twin (JSON, CSV, schedule, PNG, tolerance inputs); online calibrations added to a vehicle; watch folders read headless | 3 a sensor due or beyond its limit, 4 it cannot run, or a file or vehicle not found |

## Claude over MCP (Claude Code, Claude Desktop)

Vantage is an MCP server, so Claude (with a Claude subscription in Claude Code or Claude Desktop, or any MCP client)
can read and change rig designs. Add it once:

```bash
claude mcp add --scope user vantage -- python3 /path/to/vantage/run_vantage.py --mcp     # from source
claude mcp add --scope user vantage -- vantage --mcp                                     # installed (Linux)
```

On Windows use `vantage-cli.exe --mcp` and on macOS `/Applications/Vantage.app/Contents/MacOS/vantage-cli --mcp`.
Claude Desktop takes the same command in `claude_desktop_config.json` under `mcpServers`.

- **With a Vantage window open**, Claude works on the project in it. Each change is one undo step named
  "Claude · …" (Edit ▸ Undo takes it back). The analysis reruns before Claude gets the figures before and after.
  Proposals, Copilot plans and design-log entries Claude makes appear in the Assistant panel for you to apply,
  approve or accept. The window listens on 127.0.0.1 only and checks a token kept in a file only you can read.
  Preferences ▸ General ▸ Claude turns it off.
- **With no window open**, Claude works on a `.vantage` file. Pass it with `--mcp-project FILE`, or Claude opens one
  with the `open_project` tool. Each change is analysed and saved, and the file as it was is kept as `FILE.bak`.
  Claude will not open a file that a window has open, nor overwrite one that changed on disk.

**Tools.**
- Everything the in-app assistant reads and runs: the design, the coverage, the studies, simulate and compare,
  `look` (pictures), and the Copilot's plan tools.
- `edit_project`, which changes anything you can change in the design as one undo step: cameras, sensors, rig parts
  (with the presets), variants, stitch seams, ROI, ECU, analysis and study settings, targets, obstacles, the brief and
  the design log. Each call is checked as a whole before anything changes. A seam is set by hand
  (`{"op": "set", "target": "seam", "id": "QR|MR", "props": {"angle_deg": -107}}`) or handed back to the automatic
  placement (`"props": {"auto": true}`). `get_seams` gives the advised angles, and `simulate_changes` compares them by
  where a standing object vanishes at each seam.
- `undo_last_change`, which only ever undoes Claude's own last step, and `redo_change`.
- `run_batch`, which runs any of batch mode's 94 job options on the project as it is now, unsaved changes included,
  in a run folder beside the project. That covers studies, checks of drive logs, as-built recordings and installation
  photos, lens calibration and import, online calibration, log replay, refinement, experiments, validation and
  reproducing results. It returns the exit code and its meaning, the files written and the small JSON ones. CARLA
  runs, model downloads and the automation command are not available through it.
- `save_project`, `export_files`, `get_recent_activity` (what you changed in the window), `get_project_data` and
  `vantage_status`.

**Claude inside the window.** The assistant panel on the right (Ctrl+J) has a switch at the top. **Groq assistant**
is the in-app assistant: Groq, or Claude through the API, set in Preferences ▸ Assistant; it proposes changes for you
to apply. **Claude Code · MCP** is a terminal in the same panel. Claude Code starts there by itself the first time,
on your Claude subscription, and works on *this* window even when several are open, so its changes appear beside
the conversation (Ctrl+\` goes straight to it). The terminal is your shell, in the project's folder. It handles
colours, Claude Code's interface, copy and paste (Ctrl+Shift+C / Ctrl+Shift+V) and scrollback (the wheel, or
Shift+PageUp). Keys go to the shell while it has focus. It needs the optional `pyte` package
(`sudo apt install python3-pyte`, `pip install pyte`, or the `[terminal]` extra; the Linux and macOS downloads
include it) and runs on Linux and macOS.

**What Claude cannot change.** Locked cameras and sensors (Claude may lock one, never unlock it), unit calibrations,
stream sources, the fleet, photos, the vehicle model, the Copilot's approval records and the automation settings.
Runs such as CARLA drives still go through a Copilot plan that you approve in the window.
