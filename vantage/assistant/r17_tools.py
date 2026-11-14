"""The assistant's tools for round 17's studies (thermal cameras, event cameras, FMCW lidar, imaging radar, neural
replay, the ego body in CARLA, USD, framing, FEA, aero, trigger firmware, the edge daemon, SOTIF, cleaning, the lab).

Each study here is cheap (a second or a few) and runs on the snapshot, or reads the page's last result where a run is
long; none writes anything. Changes go through MCP's write tools (mcp/r17.py) or the page, as one undo step. Every
result is plain JSON in the units the module states; none carries a file path."""
from __future__ import annotations


def _T():
    from . import tools
    return tools


def _variant(tc, name):
    from .r14_tools import _variant as v
    return v(tc, name)


# =========================================================================== thermal cameras at night
def get_thermal_night(tc, condition=None, cls=None, level=None, beam_model=None, variant=None, **_):
    """The thermal study on the snapshot (about a second): per condition which camera sees a pedestrian first, each
    camera's detection range, the findings; with condition/class/level, every camera's range and what limits it."""
    from ..core import thermal_ir as TH
    try:
        v = _variant(tc, variant)
    except ValueError as e:
        return {'error': str(e)}
    st = TH.settings_of(tc.project)
    if beam_model:
        d = st.to_dict()
        d['beam_model'] = beam_model
        st = TH.ThermalSettings.from_dict(d)
    res = TH.run(tc.project, tc.engine, st, variant=v, cancel=_cancel(tc))
    if res is None:
        return {'error': 'cancelled'}
    out = TH.summary(res)
    if not res.cams:
        out['note'] = 'the variant has no visible camera'
    if condition or cls or level:
        cid, c, lv = condition or 'clear_night', cls or 'pedestrian', level or 'detection'
        out['ranges'] = [{'camera': r['tag'], 'modality': r.get('modality'), 'beam': r['beam'] or None,
                          'range_m': TH._r(r['range_m'], 1), 'limited_by': r['limited_by']}
                         for r in res.rows if r['condition'] == cid and r['class'] == c and r['level'] == lv]
        out['asked'] = {'condition': cid, 'class': c, 'level': lv}
    out['meaning'] = ('ranges are what each camera resolves (an upper bound for a trained detector): thermal cameras '
                      'from Planck radiometry, the ISO 7730 clothing temperature and the camera\'s NETD; visible '
                      'cameras in UN R112 low and high beam with a photon budget. Typical models with cited figures, '
                      'not scene measurements. A thermal camera behind glass sees nothing (glass is opaque from 8 to '
                      '14 µm). add_thermal_camera puts a Boson+ 640 on the front bumper.')
    return out


def _with(mod, settings_cls, project, **over):
    """The module's settings with the given fields changed (None leaves a field as it is)."""
    d = mod.settings_of(project).to_dict()
    d.update({k: v for k, v in over.items() if v is not None})
    return getattr(mod, settings_cls).from_dict(d)


# =========================================================================== event cameras
EVENT_FAST = {'duration_s': 0.1, 'render_scale': 0.15, 'tunnel_hz': 120}


def get_event_cameras(tc, fast=True, variant=None, **_):
    """The event camera study on the snapshot: per event camera (a virtual one beside the RGB camera when the rig has
    none) its event and data rate and the link's load, the edge it places where the RGB camera smears, and the tunnel
    exit. fast (default): a short, coarse render (seconds); fast false: the page's settings (about 30 s)."""
    from ..core import event_camera as EC
    try:
        v = _variant(tc, variant)
    except ValueError as e:
        return {'error': str(e)}
    st = _with(EC, 'EventSettings', tc.project, **(EVENT_FAST if fast else {}))
    res = EC.run(tc.project, tc.engine, st, variant=v, cancel=_cancel(tc))
    if res is None:
        return {'error': 'cancelled'}
    out = EC.summary(res)
    out['fast'] = bool(fast)
    return _scrub_paths(out)


# =========================================================================== FMCW lidar
def get_fmcw_lidar(tc, source=None, scene=None, ego_speed_kmh=None, counterpart=None, max_sweeps=None, variant=None,
                   **_):
    """The FMCW study on the snapshot (a scene well under a second; the project's recording a few seconds): per lidar
    the ego-velocity error, moving-point recall and precision, the range kept in sunlight and the cross-talk, against a
    time-of-flight lidar."""
    from ..core import fmcw_lidar as FM
    try:
        v = _variant(tc, variant)
    except ValueError as e:
        return {'error': str(e)}
    if not any(s.visible and s.kind == 'lidar' for s in v.sensors):
        return {'error': 'no visible lidar in the variant (Vehicle ▸ Add lidar; the library has FMCW models)'}
    st = _with(FM, 'FmcwSettings', tc.project, source=source, scene=scene, ego_speed_kmh=ego_speed_kmh,
               counterpart=counterpart, max_sweeps=max_sweeps)
    if st.source == 'recording' and not st.recording:
        return {'error': 'no recording set on the FMCW lidar page (a CARLA drive folder)'}
    res = FM.run(tc.project, tc.engine, st, variant=v, cancel=_cancel(tc))
    return {'error': 'cancelled'} if res is None else _scrub_paths(FM.summary(res))


def _scrub_paths(doc):
    """No paths, and plain JSON: infinite or NaN numbers (a sidelobe level of a one-element axis) become None."""
    import math
    from ..core import provenance as PV

    def plain(o):
        if isinstance(o, dict):
            return {k: plain(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [plain(v) for v in o]
        if hasattr(o, 'item') and not isinstance(o, (str, bytes)):
            o = o.item() if getattr(o, 'size', 1) == 1 else o.tolist()
        return None if isinstance(o, float) and not math.isfinite(o) else o
    return plain(PV.scrub(doc))


# =========================================================================== imaging radar
def get_imaging_radar(tc, radar=None, as_model=None, window=None, variant=None, **_):
    """Every radar as a 4D imaging radar on the snapshot (about a second): resolution and sidelobes, detection range per
    class over the road, the bumper's and road's ghosts; with radar, its beam figures, ranges, ghosts and advice."""
    from ..core import imaging_radar as IR
    try:
        v = _variant(tc, variant)
    except ValueError as e:
        return {'error': str(e)}
    if not any(s.visible and s.kind == 'radar' for s in v.sensors):
        return {'error': 'no visible radar in the variant (Vehicle ▸ Add radar)'}
    st = _with(IR, 'ImagingRadarSettings', tc.project, swap_model=as_model, window=window)
    res = IR.run(tc.project, tc.engine, st, variant=v, cancel=_cancel(tc), maps=False)
    if res is None:
        return {'error': 'cancelled'}
    out = IR.summary(res)
    if radar:
        r = next((x for x in res.radars if x['tag'] == radar), None)
        if r is None:
            return {'error': f'no radar {radar!r}: ' + ', '.join(x['tag'] for x in res.radars)}
        b = r['beam'] or {}
        out['radar'] = {'tag': r['tag'], 'model': r['model'],
                        'beam': {k: b.get(k) for k in ('az_res_deg', 'el_res_deg', 'psl_az_db', 'psl_el_db',
                                                        'range_res_m', 'grating') if k in b},
                        'continuous_m': {k: c['continuous_m'] for k, c in r['ground']['classes'].items()},
                        'fades_m': {k: c['fades'][:6] for k, c in r['ground']['classes'].items()},
                        'ghosts_in_field': [g for g in ((r['cavity'] or {}).get('ghosts') or [])
                                            if g.get('in_field') and g.get('listed')][:8],
                        'advice': IR.advice(r)}
    return _scrub_paths(out)


# =========================================================================== neural replay
def get_neural_replay(tc, **_):
    """The log replay's neural settings (projection, walkers, lights, lidar opacity, relighting) and which optional
    models are downloaded; it never runs the study (minutes on a GPU: the Log replay page's Neural replay tab)."""
    from ..core import neural_replay as NR
    from ..core import relight as RL
    out = {'settings': {k: v for k, v in NR.settings_of(tc.project).to_dict().items() if k != 'smpl_path'}}
    for name, fn in (('pose_model', NR.pose_status), ('body_model_anny', NR.anny_status),
                     ('relight_diffusion', RL.diffusion_status)):
        try:
            st = fn(verify=False)
            out[name] = {k: st.get(k) for k in ('name', 'present', 'verified', 'licence', 'expected_size',
                                                'commercial') if k in st}
        except Exception as e:                         # noqa: BLE001 — a status never breaks a question
            out[name] = {'error': type(e).__name__}
    out['how'] = ('Simulate ▸ Log replay ▸ Neural replay ▸ Measure (minutes on a GPU), or over MCP: start_job with '
                  'run_batch ["--neural-replay", "DIR"]; downloads go through a plan the user approves')
    return _scrub_paths(out)


# =========================================================================== the user's body in CARLA
def get_ego_composite(tc, **_):
    """The last drive the project's body was composited into (Simulate ▸ Composite my body into a drive…, or batch
    --ego-composite): per camera the share of the image the body covers and the cost a frame, and the check on the
    copy (as-built verdicts with the exact body, fitted turns, drift). Reads the result file; never runs."""
    import json as _json
    import os as _os
    from ..core import ego_composite as EC
    st = EC.settings_of(tc.project)
    out = st.out_dir or (st.drive.rstrip('/\\') + '-ego' if st.drive else '')
    path = _os.path.join(out, 'ego_composite.json') if out else ''
    if not path or not _os.path.isfile(path):
        return {'result': None, 'live_compositing': st.live,
                'how': 'Simulate ▸ Composite my body into a drive… (a CARLA drive with ground truth; about 80 s for '
                       'a 12 s six-camera drive), or over MCP: start_job with run_batch '
                       '["--ego-composite", DRIVE, "--ego-check"]'}
    try:
        with open(path, encoding='utf-8') as fh:
            doc = _json.load(fh)
    except (OSError, ValueError) as e:
        return {'error': f'the composited drive\'s result cannot be read: {type(e).__name__}'}
    comp, chk = doc.get('composite') or {}, doc.get('check') or {}
    res = {'cameras': {t: {k: c.get(k) for k in ('frames', 'body_share', 'hole_px', 'filled_px', 'frame_ms_median')}
                       for t, c in (comp.get('cameras') or {}).items()},
           'check': {'asbuilt_verdict': chk.get('asbuilt_verdict'),
                     'cameras': {t: {k: c.get(k) for k in ('verdict', 'fit_deg', 'error_deg', 'flagged',
                                                            'drift_px_median')}
                                 for t, c in (chk.get('cameras') or {}).items()}} if chk else None,
           'live_compositing': st.live}
    return _scrub_paths(res)


# =========================================================================== embedded: firmware, the edge watchdog
def get_trigger_firmware(tc, groups=None, simulate=False, **_):
    """The trigger-box firmware plan from the sync plan: each line's cameras, rate and first pulse, and per target
    (RP2040 C/PIO, STM32F411, RP2040 Rust, Jetson) its files; with simulate, each core compiled and simulated here
    against a 30 ppm crystal and a jittered PPS (5–15 s)."""
    from ..core import trigger_firmware as FW
    st = _with(FW, 'FirmwareSettings', tc.project, groups=groups)
    res = FW.run(tc.project, tc.engine, st, cancel=_cancel(tc), simulate=bool(simulate))
    if res is None:
        return {'error': 'cancelled'}
    out = FW.summary(res)
    out['meaning'] = ('the boards produce trigger pulses and nothing else; the projects are generated, not built for '
                      'the boards here (each README lists the toolchain). export_files with the trigger_firmware '
                      'format writes them.')
    return out


def get_edge_config(tc, variant=None, **_):
    """The in-car watchdog's configuration for the variant: the cameras, their stream sources and nominal rates, the
    alert outputs and thresholds (no passwords: a stream password stays in the keyring)."""
    from ..core import edge_daemon as ED
    try:
        v = _variant(tc, variant)
    except ValueError as e:
        return {'error': str(e)}
    st = ED.settings_of(tc.project)
    cfg = ED.export_config(tc.project, v, st, engine=tc.engine, body_masks=False)
    keep = {k: cfg[k] for k in cfg if k not in ('body_masks',)}
    return _scrub_paths(keep)


def edge_dry_run(tc, faults=None, seconds=None, variant=None, **_):
    """The watchdog over a synthetic drive of the variant's cameras with the given faults written in: each fault
    caught or missed and how fast, and the false alarms (seconds of compute)."""
    from ..core import edge_daemon as ED
    try:
        v = _variant(tc, variant)
    except ValueError as e:
        return {'error': str(e)}
    from ..edge.inject import KINDS
    fl = []
    for f in faults or []:
        if not isinstance(f, dict) or f.get('kind') not in KINDS:
            return {'error': f'a fault is {{kind: {"|".join(KINDS)}, camera, at_s, …}}'}
        fl.append(dict(f))
    res = ED.dry_run(tc.project, None, fl or None, variant=v, seconds=float(seconds or 12.0), cancel=_cancel(tc))
    if res is None:
        return {'error': 'cancelled'}
    out = ED.summary(res)
    out['recording'] = 'synthetic drive' if not out.get('recording') else 'the recording'
    return _scrub_paths(out)


def get_thermal_intrinsics(tc, camera=None, temperature_c=None, **_):
    """Each camera's intrinsics against temperature, from its calibrations at several temperatures; with camera and
    temperature_c, that camera's intrinsics compensated to the temperature."""
    from ..core import thermal_intrinsics as TI
    res = TI.run(tc.project, tc.engine, cancel=_cancel(tc))
    if res is None:
        return {'error': 'cancelled'}
    out = TI.summary(res)
    if camera is not None:
        cam = next((c for c in tc.project.variant.cameras if c.tag == camera or c.uid == camera), None)
        if cam is None:
            return {'error': f'no camera {camera!r}'}
        cur = res.curve(cam.uid)
        if cur is None:
            out['camera'] = {'tag': cam.tag, 'note': 'no calibrations at two or more temperatures for this camera'}
        elif temperature_c is not None:
            rec = TI.compensate(cur, float(temperature_c))
            t = float(temperature_c)
            out['camera'] = {'tag': cam.tag, 'temperature_c': t, 'record': {k: v for k, v in rec.items()
                                                                            if k not in ('serial', 'source')},
                             'at_temperature': {q: round(float(cur.value(q, t)), 6) for q in TI.PARAMS}}
    return out


# =========================================================================== SOTIF and cleaning
def get_sotif(tc, target_kmh=None, stop_model=None, basis=None, odd=None, variant=None, **_):
    """The SOTIF matrix on the snapshot (a few seconds): the ODD speed limit and what binds it, per triggering
    condition its maximum safe speed and verdict, ISO 21448's areas, the findings."""
    from ..core import sotif as SO
    try:
        v = _variant(tc, variant)
    except ValueError as e:
        return {'error': str(e)}
    st = _with(SO, 'SotifSettings', tc.project, target_kmh=target_kmh, stop_model=stop_model, basis=basis, odd=odd)
    res = SO.run(tc.project, tc.engine, st, variant=v, cancel=_cancel(tc))
    return {'error': 'cancelled'} if res is None else SO.summary(res)


def get_cleaning(tc, ambient_c=None, speed_kmh=None, dirt=None, propose=False, variant=None, **_):
    """The cleaning and de-icing layout on the snapshot (milliseconds): coverage per camera and lidar, tank runtime
    over the shift, heater temperature and ice-clearing time, power; with propose, a starting layout when the project
    has none."""
    from ..core import cleaning as CL
    try:
        v = _variant(tc, variant)
    except ValueError as e:
        return {'error': str(e)}
    st = _with(CL, 'CleaningSettings', tc.project, ambient_c=ambient_c, speed_kmh=speed_kmh, dirt=dirt)
    proposed = False
    if propose and not st.devices:
        d = st.to_dict()
        d['devices'] = CL.propose_layout(tc.project)
        st, proposed = CL.CleaningSettings.from_dict(d), True
    res = CL.run(tc.project, tc.engine, st, variant=v, cancel=_cancel(tc))
    if res is None:
        return {'error': 'cancelled'}
    out = CL.summary(res)
    if proposed:
        out['proposed_layout'] = ('a starting layout, not applied: set analysis.cleaning.devices with edit_project '
                                  'or use the Cleaning page\'s Propose button')
    return out


# =========================================================================== frame, modes, aerodynamics
def get_frame(tc, variant=None, **_):
    """The T-slot frame synthesised from the rig on the snapshot (well under a second): members, plates, feet, mass,
    the cut list and stock bars, the findings."""
    from ..core import framing as FR
    try:
        v = _variant(tc, variant)
    except ValueError as e:
        return {'error': str(e)}
    res = FR.run(tc.project, tc.engine, variant=v, cancel=_cancel(tc))
    if res is None:
        return {'error': 'cancelled'}
    out = FR.summary(res)
    out['cut_list'] = [{k: r.get(k) for k in ('item', 'profile', 'part', 'length_mm', 'quantity', 'ends') if k in r}
                       for r in res.cut_list][:40]
    if not res.members:
        out['note'] = 'nothing to frame: add a roof rack or a sensor mast (Vehicle ▸ Add rig part)'
    return out


def get_frame_modes(tc, **_):
    """The frame's modes from the Frame modes page's last run for this design (a run meshes about 100k degrees of
    freedom: about a minute and 2 GB, so the assistant never starts one), else what is set up."""
    from ..core import fea as FE
    res = FE.cached(tc.project)
    if res is None:
        s = (getattr(tc, 'studies', None) or {}).get('fea')
        if s:
            return s
        return {'result': None, 'settings': FE.settings_of(tc.project).to_dict(),
                'how': 'Analysis ▸ Frame modes ▸ Run (a background job, about a minute); over MCP, '
                       'start_job with run_batch ["--fea", "DIR"]'}
    return FE.summary(res)


def get_aerodynamics(tc, variant=None, **_):
    """The rig's drag, EV range loss, vortex lock-in and wind load on the snapshot (milliseconds; the frame's modes
    when the Frame modes page has them, else beam estimates). Not CFD."""
    from ..core import aero as AE
    try:
        v = _variant(tc, variant)
    except ValueError as e:
        return {'error': str(e)}
    res = AE.run(tc.project, tc.engine, variant=v, cancel=_cancel(tc))
    if res is None:
        return {'error': 'cancelled'}
    out = AE.summary(res)
    out['meaning'] = ('published bluff-body drag coefficients times the rig\'s own frontal areas (not CFD); range loss '
                      'on the vehicle\'s settings (a 40 kWh EV by default); lock-in where a member sheds vortices '
                      'at one of the frame\'s modes, with its Scruton number')
    return out


def _cancel(tc):
    ev = getattr(tc, 'cancel', None)
    return (lambda: ev.is_set()) if ev is not None else None


def _tools():
    from .tools import Tool
    S = {'type': 'string'}
    return [
        Tool('get_thermal_night', 'Compute thermal and night ranges',
             'Thermal (LWIR) cameras against visible cameras at night: per condition (clear_night, fog, rain, '
             'hot_road) which camera detects a pedestrian first and how far, each camera\'s pedestrian detection '
             'range, the hot-road contrast inversion and glass blocking LWIR; with condition, class and level, every '
             'camera\'s range and what limits it (pixels, thermal contrast, light). About a second. Use it for '
             '"how far do we see a pedestrian at night", "would a thermal camera help in fog".',
             {'condition': {**S, 'enum': ['clear_night', 'fog', 'rain', 'hot_road']},
              'class': {**S, 'enum': ['pedestrian', 'cyclist', 'car']},
              'level': {**S, 'enum': ['detection', 'recognition', 'identification']},
              'beam_model': {**S, 'enum': ['typical', 'r112_min'],
                             'description': 'a typical lamp or the dimmest UN R112 allows'},
              'variant': {**S, 'description': 'variant name (default the active one)'}},
             lambda tc, **k: get_thermal_night(tc, cls=k.pop('class', None), **k), 'compute'),
        Tool('get_event_cameras', 'Compute the event camera study',
             'Neuromorphic event cameras on the rig (a virtual Prophesee EVK4 beside the front camera when it has '
             'none): event rate and data rate over a rendered drive and whether the link keeps up (dropped events), '
             'how sharply a moving edge is placed where the RGB exposure smears it, and a tunnel exit with a stopped '
             'car ahead where the RGB exposure catches up. fast (default): seconds; fast false: the page\'s settings '
             '(about 30 s; start_job over MCP).',
             {'fast': {'type': 'boolean'}, 'variant': {**S, 'description': 'variant name (default the active one)'}},
             get_event_cameras, 'compute'),
        Tool('get_fmcw_lidar', 'Compute the FMCW lidar study',
             'What a coherent FMCW lidar adds over time-of-flight: per lidar the radial velocity of every return, '
             'the car\'s own velocity from one sweep (its error), moving-point recall and precision (and the points '
             'moving across the ray, which Doppler cannot see), the range kept in sunlight and the false points from '
             'the sun and another lidar. A time-of-flight lidar is evaluated as the chosen FMCW model at its pose. On '
             'a scripted scene (under a second) or the page\'s CARLA drive (seconds).',
             {'source': {**S, 'enum': ['scene', 'recording']}, 'scene': {**S, 'enum': ['urban', 'highway']},
              'ego_speed_kmh': {'type': 'number'},
              'counterpart': {**S, 'description': 'the FMCW model a time-of-flight lidar is compared as'},
              'max_sweeps': {'type': 'integer'},
              'variant': {**S, 'description': 'variant name (default the active one)'}}, get_fmcw_lidar, 'compute'),
        Tool('get_frame', 'Design the rig frame',
             'The T-slot aluminium frame Vantage builds from the roof rig (cited 80/20, IPS and Framing Technology '
             'profiles): members, plates under each sensor, roof-rail feet, its mass, the cut list and stock bars, '
             'sensors left unframed, the findings. Under a second. add_frame_to_rig (MCP) puts it into the rig.',
             {'variant': {**S, 'description': 'variant name (default the active one)'}}, get_frame, 'compute'),
        Tool('get_frame_modes', 'Read the frame modes',
             'The frame\'s lowest modes (Hz, part, motion) and each sensor\'s road jitter through them beside the '
             'one-mode beam estimate, from the Frame modes page\'s last run for this design; without one, how to run '
             'it (about a minute). Cheap: it never runs the solver.', {}, get_frame_modes),
        Tool('get_aerodynamics', 'Compute the rig aerodynamics',
             'The rig\'s drag and CdA at 100 and 130 km/h, the EV range it costs, vortex-shedding lock-in speeds '
             'against the frame\'s modes, and the crosswind load on each sensor. From cited coefficients, not CFD. '
             'Milliseconds.',
             {'variant': {**S, 'description': 'variant name (default the active one)'}}, get_aerodynamics, 'compute'),
        Tool('get_imaging_radar', 'Compute the imaging radar study',
             'Every radar as a 4D MIMO imaging radar: the virtual array\'s azimuth and elevation resolution, '
             'sidelobes and grating lobes (uniform, Hann or Taylor taper), detection range per class over the road '
             'with two-ray fades, the ghosts the bumper cavity and the road make, and mounting advice (gap, tilt, '
             'fascia thickness, pitch). as_model evaluates every radar as that imaging radar (nothing changes). '
             'About a second.',
             {'radar': {**S, 'description': 'one radar\'s tag: its beam, ranges, ghosts and advice'},
              'as_model': {**S, 'description': 'an imaging radar model id from the library'},
              'window': {**S, 'enum': ['uniform', 'hann', 'taylor']},
              'variant': {**S, 'description': 'variant name (default the active one)'}}, get_imaging_radar,
             'compute'),
        Tool('get_neural_replay', 'Read the neural replay',
             'The log replay\'s neural settings (exact-lens 3DGUT projection or affine, articulated or rigid walkers, '
             'traffic-light states, lidar opacity, relighting to wet, night or snow), which optional models are '
             'downloaded (ViTPose-B, the Anny body, InstructPix2Pix), and how to measure it. Change the settings with edit_project (analysis.neural_replay); a run is long work.', {},
             get_neural_replay),
        Tool('get_ego_composite', 'Read the composited drive',
             'The last CARLA drive the project\'s own body was composited into (in place of CARLA\'s car): per '
             'camera the body\'s share of the image, and the check on the copy (the as-built verdict with the exact '
             'body outline, fitted turns, the drift indicator); else how to make one. Cheap: it reads the result.',
             {}, get_ego_composite),
        Tool('get_trigger_firmware', 'Plan the trigger firmware',
             'Trigger-box firmware from the sync plan: each trigger line\'s cameras, rate and first pulse after the '
             'GNSS PPS, and per target (RP2040 in C with PIO, STM32F411, RP2040 in Rust, Jetson libgpiod) the files; '
             'with simulate, each scheduling core compiled and simulated here (5–15 s): the worst pulse error with a '
             '30 ppm crystal and PPS jitter, and after a PPS dropout. The boards only produce trigger pulses.',
             {'groups': {**S, 'enum': ['plan', 'suggested']}, 'simulate': {'type': 'boolean'}},
             get_trigger_firmware, 'compute'),
        Tool('get_edge_config', 'Read the edge watchdog config',
             'The in-car watchdog\'s configuration for the variant: cameras, stream sources and rates, thresholds, '
             'alert outputs (status page, MQTT, webhook, log, buzzer). No passwords.',
             {'variant': {**S, 'description': 'variant name (default the active one)'}}, get_edge_config),
        Tool('edge_dry_run', 'Dry-run the edge watchdog',
             'Run the in-car watchdog over a synthetic drive of the variant\'s cameras with faults written in '
             '(knock, freeze, drop, stall, dark: {kind, camera, at_s, yaw|pitch|roll, dur_s, gain, keep_every}): '
             'which it caught, how fast, and any false alarm. Seconds.',
             {'faults': {'type': 'array', 'items': {'type': 'object'}}, 'seconds': {'type': 'number'},
              'variant': {**S, 'description': 'variant name (default the active one)'}}, edge_dry_run, 'compute'),
        Tool('get_thermal_intrinsics', 'Read the thermal intrinsics',
             'How each camera\'s fx, fy, cx, cy and distortion move with temperature (from its calibrations at '
             'several temperatures: per °C slopes, drift over the operating range); with camera and temperature_c, '
             'its intrinsics compensated to that temperature.',
             {'camera': {**S, 'description': 'a camera tag'}, 'temperature_c': {'type': 'number'}},
             get_thermal_intrinsics),
        Tool('get_sotif', 'Compute the SOTIF matrix',
             'ISO 21448 triggering conditions (rain, fog and snow bins, night, low sun, lens dirt, the blind volume, '
             'housings, a tunnel exit; thermal and event cameras when fitted) against the rig\'s functional '
             'insufficiencies, each measured by an existing study, and the ODD speed limit from UN R157\'s stopping '
             'model per road surface: what binds it, each condition\'s maximum safe speed and verdict, the known and '
             'unknown areas. A few seconds. Supports a safety case; not a certification.',
             {'target_kmh': {'type': 'number', 'description': 'the speed the ODD should support'},
              'stop_model': {**S, 'enum': ['r157', 'r157_fsm', 'aashto', 'custom']},
              'basis': {**S, 'enum': ['two', 'any', 'cameras', 'lidars', 'radars'],
                        'description': 'which sensors must see the object (two: two sensor types)'},
              'odd': {'type': 'array', 'items': S, 'description': 'the triggering condition ids inside the ODD'},
              'variant': {**S, 'description': 'variant name (default the active one)'}}, get_sotif, 'compute'),
        Tool('get_cleaning', 'Compute cleaning and de-icing',
             'Washer and air nozzles and lens heaters on the cameras and lidars: each window\'s spray coverage and '
             'where it misses, cleans per hour, the tank\'s runtime over a shift, each heater\'s lens temperature '
             'and ice-clearing time at the speed and ambient, the power they add. Milliseconds. propose: a starting '
             'layout when the project has none (not applied).',
             {'ambient_c': {'type': 'number'}, 'speed_kmh': {'type': 'number'},
              'dirt': {**S, 'enum': ['dust', 'spray', 'mud', 'salt', 'insects', 'rain']},
              'propose': {'type': 'boolean'},
              'variant': {**S, 'description': 'variant name (default the active one)'}}, get_cleaning, 'compute'),
    ]


def register(T=None):
    """Add TOOLS_R17 to tools.TOOLS and tools.BY_NAME once (idempotent)."""
    T = T or _T()
    have = {t.name for t in T.TOOLS}
    for t in TOOLS_R17:
        if t.name not in have:
            T.TOOLS.append(t)
        T.BY_NAME[t.name] = T.BY_NAME.get(t.name, t)


TOOLS_R17 = _tools()
