"""Batch options of round 17's studies (batch.py adds them to its parser and runs them after round 14's).

Each option's exit codes are in its help; the batch's code is the largest of every option's."""
from __future__ import annotations

import json
import os
import sys

ROUND17_OPTIONS = ('thermal', 'event_cameras', 'event_drive', 'fmcw', 'imaging_radar', 'sotif', 'cleaning',
                   'framing', 'fea', 'aero', 'trigger_firmware', 'edge_config', 'edge_dry_run',
                   'thermal_intrinsics', 'usd_export', 'ego_composite', 'neural_replay',
                   'diff', 'lab_publish', 'lab_pull', 'lab_sync', 'lab_pooled_twin')


def add_round17_arguments(g):
    g.add_argument('--thermal', metavar='OUT_DIR',
                   help='thermal cameras at night (Analysis ▸ Thermal cameras at night): detection, recognition and '
                        'identification range of a pedestrian, cyclist and car per camera on a clear night, in fog, '
                        'rain and on a hot road, thermal (LWIR) cameras against visible ones in UN R112 headlamp '
                        'light; writes thermal.json, CSVs and PNGs into OUT_DIR; exit 2 when a thermal camera looks '
                        'through glass with no IR window')
    g.add_argument('--event-cameras', metavar='OUT_DIR',
                   help='event cameras (Analysis ▸ Event cameras): each event camera\'s event rate and data rate '
                        'over a rendered drive against its link, motion blur against the RGB smear and a tunnel exit; '
                        'a virtual EVK4 beside the front camera when the rig has none; JSON, CSVs and PNGs into '
                        'OUT_DIR; exit 3 when an event camera\'s data is over its link')
    g.add_argument('--event-drive', metavar='DRIVE_DIR',
                   help='a CARLA drive\'s recorded events (events/index.json): per camera the event rate and CARLA\'s '
                        'DVS against the model, as JSON on stdout; exit 2 when the drive has no events')
    g.add_argument('--fmcw', metavar='OUT_DIR',
                   help='the FMCW lidar study (Analysis ▸ FMCW lidar): radial velocity per point, ego velocity and '
                        'moving points from one sweep, sun and cross-talk against a time-of-flight lidar, on the '
                        'page\'s scene (or --fmcw-recording); writes JSON, CSVs, the 4D clouds as PCD and PLY and a '
                        'ROS 2 bag into OUT_DIR; exit 3 when an FMCW lidar\'s velocity range is below the scene\'s '
                        'closing speed, 2 when there is no lidar or the recording cannot be read')
    g.add_argument('--fmcw-recording', metavar='DRIVE_DIR',
                   help='with --fmcw: a CARLA drive the bridge wrote (ground truth and lidar sweeps)')
    g.add_argument('--fmcw-sweeps', type=int, metavar='N', help='with --fmcw: at most N sweeps of the recording')
    g.add_argument('--imaging-radar', nargs='?', const='-', metavar='OUT.json|DIR',
                   help='imaging radar (Analysis ▸ Imaging radar): every radar as a 4D MIMO imaging radar: '
                        'resolution and sidelobes, detection range per class over the road, the bumper\'s and road\'s '
                        'ghosts, mounting advice; a DIR gets JSON, CSVs and a PNG; exit 2 on a grating lobe or a '
                        'serious cavity ghost inside the field, 4 when the variant has no radar')
    g.add_argument('--imaging-radar-as', metavar='MODEL_ID',
                   help='with --imaging-radar: evaluate every radar as this imaging radar from the library')
    g.add_argument('--sotif', nargs='?', const='-', metavar='OUT.json|DIR',
                   help='the SOTIF matrix (Analysis ▸ SOTIF matrix): ISO 21448 triggering conditions against the '
                        'rig\'s functional insufficiencies and the ODD speed limit (UN R157\'s stopping model); a '
                        'DIR gets JSON, CSVs, HTML and PDF; exit 3 when the ODD speed limit is below the target, 2 '
                        'with no '
                        'visible camera. Supports a safety case; not a certification')
    g.add_argument('--sotif-speed', type=float, metavar='KMH', help='with --sotif: the target speed')
    g.add_argument('--sotif-model', choices=('r157', 'r157_fsm', 'aashto', 'custom'),
                   help='with --sotif: the stopping model')
    g.add_argument('--sotif-basis', choices=('two', 'any', 'cameras', 'lidars', 'radars'),
                   help='with --sotif: which sensors must see the object (two: two sensor types)')
    g.add_argument('--cleaning', nargs='?', const='-', metavar='OUT.json|DIR',
                   help='cleaning and de-icing (Analysis ▸ Cleaning and de-icing): coverage, cleans per hour, tank '
                        'over the shift, heaters at speed, power; exit 3 on a serious finding (a tank that does not '
                        'last the shift, a heater that cannot clear ice, a nozzle that misses its window)')
    g.add_argument('--cleaning-propose', action='store_true',
                   help='with --cleaning: a starting layout when the project has no devices (washers, air and heaters)')
    g.add_argument('--framing', metavar='OUT_DIR',
                   help='the rig frame (Analysis ▸ Frame): a T-slot aluminium frame from the roof rig with a cut '
                        'list, stock plan, BOM, CNC plate DXFs, STL parts and a STEP assembly into OUT_DIR; exit 3 '
                        'on a serious finding (a sensor sunk into the frame, a cut longer than stock), 2 when there is '
                        'nothing to frame')
    g.add_argument('--fea', metavar='OUT_DIR',
                   help='frame modes (Analysis ▸ Frame modes): a quadratic-tetrahedron modal analysis of the frame, '
                        'the road jitter per sensor and the crosswind case (about a minute and 2 GB); exit 3 when a '
                        'mode lies in the body-bounce or wheel-hop band, 2 when there is no frame')
    g.add_argument('--fea-mesh', type=float, metavar='MM', help='with --fea: the element size (default 25 mm)')
    g.add_argument('--aero', metavar='OUT_DIR',
                   help='aerodynamics (Analysis ▸ Aerodynamics): the rig\'s drag, the EV range it costs, vortex '
                        'lock-in against the frame\'s modes and the crosswind load per sensor (not CFD); exit 3 on a '
                        'range or lock-in warning, 2 when nothing stands above the roof')
    g.add_argument('--trigger-firmware', metavar='OUT_DIR',
                   help='trigger-box firmware (Embedded ▸ Trigger firmware) from the sync plan: RP2040 (C, PIO), '
                        'STM32F411 (C), RP2040 (Rust) and Jetson (libgpiod) projects, each one\'s scheduling core '
                        'compiled and simulated here against a crystal and a GNSS PPS; exit 3 on a plan problem or a '
                        'failed check, 8 when a toolchain to prove a target is missing (the files are written)')
    g.add_argument('--edge-config', metavar='OUT_DIR',
                   help='the edge watchdog\'s install files (Embedded ▸ Edge watchdog): edge.json, the systemd unit '
                        'and a README')
    g.add_argument('--edge-dry-run', metavar='OUT_DIR',
                   help='run the edge watchdog over --edge-recording (default: a synthetic drive of the cameras) with '
                        '--edge-faults written in; writes the alerts JSON and CSV; exit 3 when a fault is missed or a '
                        'false alarm is raised, 4 when the recording cannot be read')
    g.add_argument('--edge-recording', metavar='PATH',
                   help='with --edge-dry-run: a nuScenes folder, image folders, videos or a ROS 2 bag')
    g.add_argument('--edge-faults', metavar='FAULTS.json',
                   help='with --edge-dry-run: [{kind: knock|freeze|drop|stall|dark, camera, at_s, …}]')
    g.add_argument('--thermal-intrinsics', metavar='OUT_DIR',
                   help='thermal intrinsics (Calibrate ▸ Thermal intrinsics): each camera\'s intrinsics fitted '
                        'against temperature from its calibrations at several temperatures; JSON, CSV, PNG, OpenCV and '
                        'Kalibr files; exit 3 when a camera drifts past the limit over its operating range, 2 when no '
                        'camera has calibrations at two temperatures')
    g.add_argument('--usd-export', metavar='OUT_DIR',
                   help='the vehicle, the rig and every camera, lidar and radar as an OpenUSD stage (Omniverse, Isaac '
                        'Sim; cameras with exact intrinsics and NVIDIA\'s fisheye schema) and the rig as an NVIDIA '
                        'NCore V4 store (NuRec\'s input); exit 4 when --usd-splats holds no scene (the stage is still '
                        'written)')
    g.add_argument('--usd-splats', metavar='CACHE_DIR',
                   help='with --usd-export: a log replay\'s cache folder; its Gaussians go in as a USD ParticleField')
    g.add_argument('--ego-composite', metavar='DRIVE_DIR',
                   help='composite the project\'s body, rig and housings into a recorded CARLA drive with ground truth '
                        '(images, lidar, labels) in place of CARLA\'s car, into --ego-out (default <drive>-ego); with '
                        '--ego-check also the as-built check (exact body) and the drift indicator on the copy; exit 3 '
                        'when the check finds a camera to re-mount, 4 when the drive cannot be read or has no ground '
                        'truth')
    g.add_argument('--ego-out', metavar='DIR', help='with --ego-composite: where the composited copy goes')
    g.add_argument('--ego-check', action='store_true', help='with --ego-composite: check the copy too')
    g.add_argument('--neural-replay', metavar='OUT_DIR',
                   help="neural replay (Simulate ▸ Log replay ▸ Neural replay) on the log replay's drive: the exact-"
                        'lens (3DGUT) projection against the affine one, articulated walkers, traffic-light states, a '
                        'lidar opacity and relit frames, each measured (held-out PSNR, lidar error, detection recall '
                        'against the recorded images); minutes on a GPU; exit 3 when the chosen projection validates '
                        'worse than the other on this drive, 4 when there is no drive or PyTorch is missing')
    g.add_argument('--neural-projection', choices=('ut', 'ewa'), help='with --neural-replay: the projection used')
    g.add_argument('--neural-relight', metavar='CONDITIONS',
                   help='with --neural-replay: comma-separated relit conditions (wet, night, snow)')
    g.add_argument('--diff', metavar='OTHER',
                   help='what OTHER (a .vantage file, a project folder or path@REV of a git repository) changes '
                        'against this project, by meaning: cameras moved by mm and turned by degrees, lenses, '
                        'settings, the fleet; the text on stdout, --diff-json OUT for the JSON; exit 1 when they '
                        'differ (0 when the same)')
    g.add_argument('--diff-json', metavar='OUT.json', help='with --diff: also write the diff as JSON')
    g.add_argument('--lab-publish', nargs=2, metavar=('URL', 'PROJECT_ID'),
                   help='publish the project to a lab server as a new immutable version (the token from the keyring, '
                        'lab-tokens.json or VANTAGE_LAB_TOKEN); exit 3 when refused (401, 403, or 409: someone '
                        'published first; sync instead), 4 when the server cannot be reached')
    g.add_argument('--lab-message', metavar='TEXT', help='with --lab-publish or --lab-sync: the version\'s message')
    g.add_argument('--lab-pull', nargs=3, metavar=('URL', 'PROJECT_ID[@REF]', 'DEST'),
                   help='pull a version (head, or a number) from a lab server into DEST (name.vantage.d: a project '
                        'folder; else a .vantage file); exit 3 refused, 4 unreachable')
    g.add_argument('--lab-sync', nargs=2, metavar=('URL', 'PROJECT_ID'),
                   help='bring this project file and the lab together (publish, pull, or a three-way merge when both '
                        'changed; the result saved over the project); exit 5 on conflicts (nothing published), 3 '
                        'refused, 4 unreachable')
    g.add_argument('--lab-pooled-twin', nargs=2, metavar=('URL', 'OUT_DIR'),
                   help='the fleet twin of every fleet the lab shares with your team, re-based onto this project\'s '
                        'plan; writes the fleet twin\'s files into OUT_DIR; exit 3 refused, 4 unreachable')
    g.add_argument('--lab-projects', metavar='IDS', help='with --lab-pooled-twin: comma-separated lab project ids')


def run_round17(project, eng, args, helpers) -> int:
    """Every round-17 option given. helpers: batch's {'out', 'stamped', 'progress'}."""
    code = 0
    if args.diff:
        code = max(code, _diff(project, args))
    if args.thermal:
        code = max(code, _thermal(project, eng, args, helpers))
    if args.framing:
        code = max(code, _framing(project, eng, args, helpers))
    if args.fea:
        code = max(code, _fea(project, eng, args, helpers))
    if args.aero:
        code = max(code, _aero(project, eng, args, helpers))
    if args.event_cameras:
        code = max(code, _event_cameras(project, eng, args, helpers))
    if args.event_drive:
        code = max(code, _event_drive(args))
    if args.fmcw:
        code = max(code, _fmcw(project, eng, args, helpers))
    if args.trigger_firmware:
        code = max(code, _trigger_firmware(project, eng, args, helpers))
    if args.edge_config:
        code = max(code, _edge_config(project, args))
    if args.edge_dry_run:
        code = max(code, _edge_dry_run(project, args, helpers))
    if args.thermal_intrinsics:
        code = max(code, _thermal_intrinsics(project, eng, args, helpers))
    if args.imaging_radar:
        code = max(code, _imaging_radar(project, eng, args, helpers))
    if args.usd_export:
        code = max(code, _usd_export(project, eng, args, helpers))
    if args.ego_composite:
        code = max(code, _ego_composite(project, eng, args, helpers))
    if args.neural_replay:
        code = max(code, _neural_replay(project, eng, args, helpers))
    if args.sotif:
        code = max(code, _sotif(project, eng, args, helpers))
    if args.cleaning:
        code = max(code, _cleaning(project, eng, args, helpers))
    if args.lab_sync:
        code = max(code, _lab(lambda: _lab_sync(project, args)))
    if args.lab_publish:
        code = max(code, _lab(lambda: _lab_publish(project, args)))
    if args.lab_pull:
        code = max(code, _lab(lambda: _lab_pull(args)))
    if args.lab_pooled_twin:
        code = max(code, _lab(lambda: _lab_pooled_twin(project, eng, args, helpers)))
    return code


# =========================================================================== thermal
def _thermal(project, eng, args, H) -> int:
    """--thermal OUT_DIR (core.thermal_ir). 2: a thermal camera behind glass with no IR window."""
    from .core import thermal_ir as TH
    res = TH.run(project, eng, progress=H['progress']('thermal'))
    print('', file=sys.stderr)
    for pth in H['stamped'](TH.export(res, project, args.thermal), '--thermal', args.thermal):
        print(pth)
    s = TH.summary(res)
    def best(x):
        return f'{x["camera"]} {x["range_m"]:.0f} m' if x else 'none'
    for c in s['conditions'].values():
        print(f'thermal: {c["label"]}: pedestrian first seen by {c["winner"] or "no camera"}; thermal '
              f'{best(c.get("thermal"))}, visible (low beam) {best(c.get("visible_low_beam"))}', file=sys.stderr)
    for f in res.findings:
        if f['severity'] not in ('good', 'note'):
            print(f'  {f["severity"]:<8} {f["message"]}', file=sys.stderr)
    crit = any(f['severity'] == 'critical' for f in res.findings) or any(
        i.severity == 'critical' for i in TH.issues(project, eng.vehicle_geom(project) if eng is not None else None))
    return 2 if crit else 0


def _write(doc_or_files, dest, H, option, export_fn):
    """dest '-' or *.json: the JSON document; else a folder of the study's files."""
    if dest == '-' or dest.lower().endswith('.json'):
        H['out'](doc_or_files(), dest)
    else:
        for pth in H['stamped'](export_fn(dest), option, dest):
            print(pth)


# =========================================================================== frame, modes, aerodynamics
def _findings(findings) -> list:
    out = []
    for f in findings:
        if f['severity'] not in ('good', 'note'):
            print(f'  {f["severity"]:<8} {f["message"]}', file=sys.stderr)
            out.append(f)
    return out


def _framing(project, eng, args, H) -> int:
    """--framing OUT_DIR (core.framing). 3: a serious finding; 2: nothing to frame."""
    from .core import framing as FR
    res = FR.run(project, eng)
    if res is None or not res.members:
        print('framing: nothing to frame: add a roof rack or a sensor mast (Vehicle ▸ Add rig part)', file=sys.stderr)
        return 2
    for pth in H['stamped'](FR.export(res, project, args.framing), '--framing', args.framing):
        print(pth)
    s = FR.summary(res)
    print(f'framing: {s["members"]} members, {s["profile_m"]} m of profile, {s["plates"]} plates, {s["feet"]} feet, '
          f'{s["frame_kg"]} kg', file=sys.stderr)
    bad = _findings(res.findings)
    return 3 if any(f['severity'] in ('critical', 'serious') for f in bad) else 0


def _fea(project, eng, args, H) -> int:
    """--fea OUT_DIR (core.fea, with core.aero's wind case). 3: a mode in the body-bounce or wheel-hop band; 2: no
    frame."""
    from .core import aero as AE
    from .core import fea as FE
    from .core import framing as FR
    if not FR.synthesise(project).members:
        print('fea: no frame: add a roof rack or a sensor mast (Vehicle ▸ Add rig part)', file=sys.stderr)
        return 2
    st = FE.settings_of(project)
    if args.fea_mesh:
        d = st.to_dict()
        d['mesh_mm'] = args.fea_mesh
        st = FE.FEASettings.from_dict(d)
    res = FE.run(project, eng, st, progress=H['progress']('frame modes'), wind=AE.run(project, eng))
    print('', file=sys.stderr)
    if res is None or not res.modes:
        print('fea: no modes (no frame to mesh)', file=sys.stderr)
        return 2
    for pth in H['stamped'](FE.export(res, project, args.fea), '--fea', args.fea):
        print(pth)
    s = FE.summary(res)
    print(f'fea: {s["dofs"]} DOFs, modes ' + ', '.join(f'{f:.1f}' for f in s['modes_hz'][:6]) + ' Hz', file=sys.stderr)
    bad = _findings(res.findings)
    return 3 if any(f['kind'] == 'resonance' for f in bad) else 0


def _aero(project, eng, args, H) -> int:
    """--aero OUT_DIR (core.aero). 3: a range or lock-in warning; 2: nothing above the roof."""
    from .core import aero as AE
    res = AE.run(project, eng)
    if res is None or not res.elements:
        print('aero: nothing above the roof: no rig drag', file=sys.stderr)
        return 2
    for pth in H['stamped'](AE.export(res, project, args.aero), '--aero', args.aero):
        print(pth)
    for sp in AE.summary(res)['speeds']:
        print(f'aero: {sp["kmh"]:g} km/h: {sp["drag_n"]} N, {sp["range_loss_pct"]} % range', file=sys.stderr)
    bad = _findings(res.findings)
    return 3 if any(f['kind'] in ('range', 'lock_in') for f in bad) else 0


# =========================================================================== event cameras
def _event_cameras(project, eng, args, H) -> int:
    """--event-cameras OUT_DIR (core.event_camera). 3: an event camera's data over its link."""
    from .core import event_camera as EC
    res = EC.run(project, eng, progress=H['progress']('event cameras'))
    print('', file=sys.stderr)
    if res is None:
        return 130
    for pth in H['stamped'](EC.export(res, project, args.event_cameras), '--event-cameras', args.event_cameras):
        print(pth)
    bad = []
    for f in res.findings:
        if f['severity'] not in ('good', 'note'):
            print(f'  {f["severity"]:<8} {f["message"]}', file=sys.stderr)
            bad.append(f)
    return 3 if any(f['severity'] in ('critical', 'serious') for f in bad) else 0


def _event_drive(args) -> int:
    """--event-drive DRIVE_DIR (core.event_camera). 2: no events in the drive."""
    from .core import event_camera as EC
    doc = EC.load_recording(args.event_drive)
    if not doc:
        print(f'event drive: {args.event_drive} has no events/index.json', file=sys.stderr)
        return 2
    print(json.dumps(EC.recording_summary(doc), indent=1, ensure_ascii=False, default=str))
    return 0


# =========================================================================== FMCW lidar
def _fmcw(project, eng, args, H) -> int:
    """--fmcw OUT_DIR (core.fmcw_lidar). 3: an FMCW lidar's velocity range below the closing speed; 2: no lidar, or the
    recording cannot be read."""
    from .core import fmcw_lidar as FM
    if not any(s.visible and s.kind == 'lidar' for s in project.variant.sensors):
        print('fmcw: no visible lidar in the variant', file=sys.stderr)
        return 2
    d = FM.settings_of(project).to_dict()
    if args.fmcw_recording:
        if not os.path.isdir(args.fmcw_recording):
            print(f'fmcw: {args.fmcw_recording}: not a folder', file=sys.stderr)
            return 2
        d.update(source='recording', recording=os.path.abspath(args.fmcw_recording))
    if args.fmcw_sweeps:
        d['max_sweeps'] = args.fmcw_sweeps
    try:
        res = FM.run(project, eng, FM.FmcwSettings.from_dict(d), progress=H['progress']('fmcw'))
    except (OSError, ValueError) as e:
        print(f'\nfmcw: {e}', file=sys.stderr)
        return 2
    print('', file=sys.stderr)
    paths = FM.export(res, project, args.fmcw)
    bag = FM.write_bag(res, args.fmcw)
    for pth in H['stamped'](paths, '--fmcw', args.fmcw) + ([bag] if bag else []):
        print(pth)
    for r in FM.summary(res)['lidars']:
        print(f'fmcw: {r["tag"]} ({r["model"]}{"" if r["fmcw"] else ", as " + str(r["evaluated_as"])}): ego error '
              f'{r["ego_error_mean_mps"]} m/s, moving recall {r["moving_recall"]}, precision {r["moving_precision"]}',
              file=sys.stderr)
    iss = FM.issues(project)
    for i in iss:
        print(f'  {i.severity:<8} {i.message}', file=sys.stderr)
    return 3 if iss else 0


# =========================================================================== imaging radar
def _imaging_radar(project, eng, args, H) -> int:
    """--imaging-radar [OUT.json|DIR] (core.imaging_radar). 2: a grating lobe or a serious cavity ghost in the field;
    4: no radar."""
    from .core import imaging_radar as IR
    if not any(s.visible and s.kind == 'radar' for s in project.variant.sensors):
        print('imaging radar: no visible radar in the variant', file=sys.stderr)
        return 4
    d = IR.settings_of(project).to_dict()
    if args.imaging_radar_as:
        d['swap_model'] = args.imaging_radar_as
    res = IR.run(project, eng, IR.ImagingRadarSettings.from_dict(d))
    _write(lambda: IR.to_doc(res, project), args.imaging_radar, H, '--imaging-radar',
           lambda dst: IR.export(res, project, dst))
    for r in IR.summary(res)['radars']:
        print(f'imaging radar: {r["tag"]} ({r["model"]}): {r["az_res_deg"]}° × {r["el_res_deg"]}°, sidelobes '
              f'{r["psl_az_db"]} dB, {r["grating_lobes"]} grating lobes, {r["cavity_ghosts_in_field"]} cavity ghosts '
              'in the field', file=sys.stderr)
    bad = [f for f in res.findings if f['kind'] == 'grating' or (f['kind'] == 'cavity_ghost'
                                                                 and f['severity'] in ('critical', 'serious'))]
    for f in bad:
        print(f'  {f["severity"]:<8} {f["message"]}', file=sys.stderr)
    return 2 if bad else 0


# =========================================================================== the user's body in CARLA, OpenUSD
def _usd_export(project, eng, args, H) -> int:
    """--usd-export OUT_DIR (core.usd_export). 4: --usd-splats holds no scene."""
    from .core import usd_export as UE
    st = UE.settings_of(project)
    code = 0
    if args.usd_splats:
        if not os.path.isfile(os.path.join(args.usd_splats, 'scene.npz')):
            print(f'usd export: {args.usd_splats} holds no log-replay scene (scene.npz): written without splats',
                  file=sys.stderr)
            code = 4
        else:
            d = st.to_dict()
            d['splats'] = os.path.abspath(args.usd_splats)
            st = UE.UsdSettings.from_dict(d)
    res = UE.run(project, eng, st)
    paths = UE.export(res, project, args.usd_export)
    for pth in H['stamped'](paths, '--usd-export', args.usd_export):
        if '.zarr' + os.sep not in pth:
            print(pth)
    s = UE.summary(res)
    print(f'usd export: {s["name"]}.usda ({s["usda_bytes"]:,} bytes), {s["ncore_files"]} NCore files; fisheye '
          f'cameras: {", ".join(s["cameras_fisheye"]) or "none"}', file=sys.stderr)
    for n in s['notes']:
        print(f'  note: {n}', file=sys.stderr)
    return code


def _ego_composite(project, eng, args, H) -> int:
    """--ego-composite DRIVE_DIR (core.ego_composite). 3: the check finds a camera to re-mount; 4: the drive cannot be
    read or has no ground truth."""
    from .core import ego_composite as EC
    d = EC.settings_of(project).to_dict()
    d.update(drive=os.path.abspath(args.ego_composite), out_dir=os.path.abspath(args.ego_out) if args.ego_out else '')
    try:
        res = EC.run(project, eng, EC.EgoSettings.from_dict(d), progress=H['progress']('compositing'),
                     check=bool(args.ego_check))
    except (OSError, ValueError) as e:
        print(f'\nego composite: {e}', file=sys.stderr)
        return 4
    print('', file=sys.stderr)
    if res is None:
        return 130
    s = EC.summary(res)
    out = s.get('out_dir') or ''
    for pth in H['stamped'](EC.export(res, project, out), '--ego-composite', out):
        print(pth)
    for t, c in s['cameras'].items():
        print(f'ego composite: {t}: body {c.get("body_share")}, {c.get("frame_ms_median")} ms a frame', file=sys.stderr)
    chk = (s.get('check') or {}).get('cameras') or {}
    for t, c in chk.items():
        print(f'  check {t}: {c.get("verdict")}', file=sys.stderr)
    return 3 if any(c.get('flagged') for c in chk.values()) else 0


# =========================================================================== neural replay
def _neural_replay(project, eng, args, H) -> int:
    """--neural-replay OUT_DIR (core.neural_replay). 3: the chosen projection validates worse than the other; 4: no
    drive, or PyTorch missing."""
    from .core import neural_replay as NR
    d = NR.settings_of(project).to_dict()
    if args.neural_projection:
        d['projection'] = args.neural_projection
    if args.neural_relight:
        conds = [c.strip() for c in args.neural_relight.split(',') if c.strip()]
        bad = [c for c in conds if c not in ('wet', 'night', 'snow')]
        if bad:
            raise SystemExit(f'--neural-relight {args.neural_relight!r}: wet, night or snow')
        d['relight_conditions'] = conds
    try:
        res = NR.run(project, eng, NR.NeuralSettings.from_dict(d), progress=H['progress']('neural replay'))
    except ValueError as e:
        print(f'\nneural replay: {e}', file=sys.stderr)
        return 4
    print('', file=sys.stderr)
    if res is None:
        return 130
    for pth in H['stamped'](NR.export(res, project, args.neural_replay), '--neural-replay', args.neural_replay):
        print(pth)
    s = NR.summary(res)
    print(f'neural replay: PSNR {s["psnr_ut_db"]} dB (UT) / {s["psnr_ewa_db"]} dB (affine); pedestrian recall '
          f'{s["pedestrian_recall_rigid"]} rigid, {s["pedestrian_recall_articulated"]} articulated, '
          f'{s["pedestrian_recall_recorded"]} recorded', file=sys.stderr)
    other = 'psnr_ewa_db' if s['projection'] == 'ut' else 'psnr_ut_db'
    mine, other = s.get(f'psnr_{s["projection"]}_db'), s.get(other)
    return 3 if mine is not None and other is not None and mine < other else 0


# =========================================================================== embedded and thermal intrinsics
def _trigger_firmware(project, eng, args, H) -> int:
    """--trigger-firmware OUT_DIR (core.trigger_firmware). 3: a plan problem or a failed check; 8: a toolchain to prove
    a target is missing."""
    from .core import trigger_firmware as FW
    res = FW.run(project, eng, progress=H['progress']('trigger firmware'))
    print('', file=sys.stderr)
    if res is None:
        return 130
    paths = FW.export(res, project, args.trigger_firmware)
    for pth in H['stamped'](paths, '--trigger-firmware', args.trigger_firmware):
        if os.sep + 'firmware' + os.sep not in pth:      # the project trees: listed by the summary
            print(pth)
    s = FW.summary(res)
    print(f'trigger firmware: {len(s["lines"])} lines, {len(paths)} files', file=sys.stderr)
    for t, d in s['targets'].items():
        print(f'  {t}: host check {d["host_check"]}, locked max {d["locked_max_error_ns"]} ns, holdover max '
              f'{d["holdover_max_error_ns"]} ns', file=sys.stderr)
    for p_ in s['problems']:
        print(f'  problem: {p_}', file=sys.stderr)
    return {0: 0, 1: 3, 2: 8, 3: 130}[FW.exit_code(res)]


def _edge_config(project, args) -> int:
    from .core import edge_daemon as ED
    for pth in ED.export_install(project, args.edge_config):
        print(pth)
    return 0


def _edge_dry_run(project, args, H) -> int:
    """--edge-dry-run OUT_DIR (core.edge_daemon). 3: a fault missed or a false alarm; 4: the recording unreadable."""
    from .core import edge_daemon as ED
    faults = None
    if args.edge_faults:
        try:
            with open(args.edge_faults, encoding='utf-8') as fh:
                faults = json.load(fh)
        except (OSError, ValueError) as e:
            raise SystemExit(f'--edge-faults {args.edge_faults}: {e}')
    try:
        res = ED.dry_run(project, args.edge_recording or None, faults, progress=H['progress']('edge dry run'))
    except (OSError, ValueError) as e:
        print(f'\nedge dry run: {e}', file=sys.stderr)
        return 4
    print('', file=sys.stderr)
    if res is None:
        return 4
    for pth in H['stamped'](ED.export(res, project, args.edge_dry_run), '--edge-dry-run', args.edge_dry_run):
        print(pth)
    m = res.metrics
    print(f'edge dry run: {m.get("caught", 0)} of {m.get("faults", 0)} faults caught, {m.get("false_alarms")} false '
          f'alarms', file=sys.stderr)
    return {0: 0, 1: 3, 2: 4}[ED.exit_code(res)]


def _thermal_intrinsics(project, eng, args, H) -> int:
    """--thermal-intrinsics OUT_DIR (core.thermal_intrinsics). 3: a drift past the limit; 2: nothing to fit."""
    from .core import thermal_intrinsics as TI
    res = TI.run(project, eng)
    if res is None or not res.curves:
        print('thermal intrinsics: no camera has calibrations at two or more temperatures', file=sys.stderr)
        return 2
    for pth in H['stamped'](TI.export(res, project, args.thermal_intrinsics), '--thermal-intrinsics',
                            args.thermal_intrinsics):
        print(pth)
    for f in res.findings:
        print(f'  {f["severity"]:<8} {f["message"]}', file=sys.stderr)
    return {0: 0, 1: 3, 2: 2}[TI.exit_code(res)]


# =========================================================================== SOTIF and cleaning
def _sotif(project, eng, args, H) -> int:
    """--sotif [OUT.json|DIR] (core.sotif). 3: the ODD speed limit is below the target; 2: no visible camera."""
    from .core import sotif as SO
    if not any(c.visible for c in project.variant.cameras):
        print('sotif: no visible camera in the variant', file=sys.stderr)
        return 2
    d = SO.settings_of(project).to_dict()
    for k, a in (('target_kmh', args.sotif_speed), ('stop_model', args.sotif_model), ('basis', args.sotif_basis)):
        if a is not None:
            d[k] = a
    res = SO.run(project, eng, SO.SotifSettings.from_dict(d), progress=H['progress']('sotif'))
    print('', file=sys.stderr)
    _write(lambda: SO.to_doc(res, project), args.sotif, H, '--sotif', lambda dst: SO.export(res, project, dst))
    s = SO.summary(res)
    lim = s['odd_speed_kmh']
    print(f'sotif: ODD speed limit {"none" if lim is None else format(lim, ".1f") + " km/h"} against '
          f'{s["target_kmh"]:g} km/h ({s["stop_model"]}, basis {s["basis"]}), bound by {s["binding"] or "nothing"}'
          f'{" · " + s["binding_class"] if s.get("binding_class") else ""}', file=sys.stderr)
    for f in s['findings']:
        if f['severity'] not in ('good', 'note'):
            print(f'  {f["severity"]:<8} {f["message"]}', file=sys.stderr)
    return 0 if s['meets_target'] else 3


def _cleaning(project, eng, args, H) -> int:
    """--cleaning [OUT.json|DIR] (core.cleaning). 3: a serious or critical finding."""
    from .core import cleaning as CL
    st = CL.settings_of(project)
    if args.cleaning_propose and not st.devices:
        d = st.to_dict()
        d['devices'] = CL.propose_layout(project)
        st = CL.CleaningSettings.from_dict(d)
        print(f'cleaning: proposed {len(st.devices)} devices (not saved: --save keeps them)', file=sys.stderr)
        project.analysis.cleaning = st.to_dict()
    res = CL.run(project, eng, st)
    _write(lambda: CL.to_doc(res, project), args.cleaning, H, '--cleaning', lambda dst: CL.export(res, project, dst))
    s = CL.summary(res)
    print(f'cleaning: {s["devices"]} devices · tank {s["tank_runtime_h"]} h over a {s["shift_h"]} h shift · '
          f'{s["average_w"]} W average, {s["peak_w"]} W peak', file=sys.stderr)
    for f in s['findings']:
        if f['severity'] not in ('good', 'note'):
            print(f'  {f["severity"]:<8} {f["message"]}', file=sys.stderr)
    return 3 if any(f['severity'] in ('critical', 'serious') for f in s['findings']) else 0


# =========================================================================== project folders and the lab
def _diff(project, args) -> int:
    """--diff OTHER (core.project_dir). 1: they differ; 0: the same."""
    from .core import project_dir as PD
    try:
        other, label = PD.load_spec(args.diff)
    except (PD.ProjectDirError, OSError, ValueError, KeyError) as e:
        raise SystemExit(f'--diff {args.diff}: {e}')
    doc = PD.diff(project, other, project.name or 'this project', label)
    sys.stdout.write(PD.diff_text(doc))
    if args.diff_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.diff_json)), exist_ok=True)
        with open(args.diff_json, 'w', encoding='utf-8') as fh:
            json.dump(doc, fh, indent=1, ensure_ascii=False)
        print(f'wrote {args.diff_json}', file=sys.stderr)
    return 0 if doc['same'] else 1


def _lab(fn) -> int:
    from .lab.client import LabClientError
    try:
        return fn()
    except LabClientError as e:
        print(f'lab: {e}', file=sys.stderr)
        return 4 if e.status == 0 else 3


def _client(url):
    from .lab.client import LabClient
    tok = os.environ.get('VANTAGE_LAB_TOKEN') or None
    return LabClient(url, token=tok)


def _lab_publish(project, args) -> int:
    url, pid = args.lab_publish
    doc = _client(url).publish(project, pid, args.lab_message or '')
    print(f'lab: published {pid} version {doc.get("number")}'
          f'{" (already there)" if doc.get("existing") else ""}', file=sys.stderr)
    return 0


def _lab_pull(args) -> int:
    url, spec, dest = args.lab_pull
    pid, _, ref = spec.partition('@')
    p = _client(url).pull(pid, ref or 'head', dest)
    print(f'lab: pulled {pid} version {p.lab_version["number"]} into {dest}', file=sys.stderr)
    return 0


def _lab_sync(project, args) -> int:
    url, pid = args.lab_sync
    out = _client(url).sync(project, pid, args.lab_message or '')
    print(f'lab: {out["status"]} (version {out["version"]})', file=sys.stderr)
    if out['status'] == 'conflicts':
        for c in out['conflicts'][:20]:
            print(f'  conflict: {c if isinstance(c, str) else json.dumps(c, ensure_ascii=False)[:160]}',
                  file=sys.stderr)
        return 5
    if out['status'] in ('pulled', 'merged') and getattr(project, 'path', ''):
        out['project'].save(project.path)
        print(f'lab: saved {os.path.basename(project.path)}', file=sys.stderr)
    return 0


def _lab_pooled_twin(project, eng, args, H) -> int:
    from .core import fleet_twin as FT
    url, out_dir = args.lab_pooled_twin
    ids = [s.strip() for s in (args.lab_projects or '').split(',') if s.strip()] or None
    res, doc = _client(url).pooled_twin(project, ids, engine=eng, progress=H['progress']('pooled fleet twin'))
    print('', file=sys.stderr)
    for pth in H['stamped'](FT.export(res, project, out_dir), '--lab-pooled-twin', out_dir):
        print(pth)
    print(f'lab: pooled {len(doc.get("sources") or [])} fleet(s)'
          f'{", skipped " + str(len(doc["skipped"])) if doc.get("skipped") else ""}', file=sys.stderr)
    return 0
