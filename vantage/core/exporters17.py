"""Export formats of round 17's studies, added to exporters.FORMATS: one table entry per format, so each study's
files come from its own module's files(result, project), the same as its page's Export button writes.

An entry: id -> dict(folder, group, label, ext, names(project, variant) -> [file names] ([] = nothing to export),
files(project, variant, engine) -> [(name, bytes)], preview(files) -> text, default)."""
from __future__ import annotations

import json


# =========================================================================== thermal cameras at night
def _thermal_names(p, v):
    if not any(c.visible for c in v.cameras):
        return []
    return ['thermal.json', 'thermal_ranges.csv', 'thermal_curves.csv', 'thermal_ranges.png', 'thermal_cnr.png']


def _thermal_files(p, v, eng):
    from . import thermal_ir as TH
    res = TH.run(p, eng, variant=v)
    return TH.files(res, p) if res is not None else []


def _thermal_preview(files):
    doc = json.loads(files[0][1])
    lines = []
    for c in doc.get('conditions') or []:
        rows = [r for r in doc.get('ranges') or [] if r.get('condition') == c.get('id')
                and r.get('class') == 'pedestrian' and r.get('level') == 'detection' and r.get('beam') in ('', 'low')]
        best = sorted(rows, key=lambda r: -(r.get('range_m') or 0))[:3]
        lines.append(f'{c.get("label") or c.get("id")}: ' + ', '.join(
            f'{r["tag"]}{" (low beam)" if r.get("beam") else ""} {r["range_m"]:.0f} m' for r in best
            if r.get('range_m') is not None))
    lines += [f'{f["severity"]}: {f["message"]}' for f in doc.get('findings') or [] if f['severity'] != 'good'][:6]
    return 'Pedestrian detection range per condition (the three farthest cameras):\n' + '\n'.join(lines)


# =========================================================================== event cameras
def _event_files(p, v, eng):
    from . import event_camera as EC
    res = EC.run(p, eng, variant=v)
    return EC.files(res, p) if res is not None else []


# =========================================================================== FMCW lidar
def _lidars(v):
    return [s for s in v.sensors if s.visible and s.kind == 'lidar']


def _fmcw_files(p, v, eng):
    from . import fmcw_lidar as FM
    res = FM.run(p, eng, variant=v)
    return FM.files(res, p) if res is not None else []


def _fmcw_preview(files):
    doc = json.loads(files[0][1])
    lines = []
    for r in doc.get('lidars') or []:
        sm = r.get('summary') or {}
        lines.append(f'{r.get("tag")} ({r.get("name")}{"" if r.get("fmcw") else ", evaluated as FMCW"}): moving '
                     f'recall {sm.get("recall")}, precision {sm.get("precision")}, ego error '
                     f'{sm.get("ego_error_mean_mps")} m/s')
    return '\n'.join(lines) or f'{len(files)} files'


# =========================================================================== imaging radar
def _radar_files(p, v, eng):
    from . import imaging_radar as IR
    res = IR.run(p, eng, variant=v)
    return IR.files(res, p) if res is not None else []


# =========================================================================== neural replay
def _neural_names(p, v):
    from . import splat_reproject as SP
    st = SP.settings_of(p)
    if not st.drive or not SP.is_drive(st.drive) or SP.torch_missing():
        return []
    return ['neural_replay.json', 'neural_replay_perception.csv', 'neural_replay_projection.csv',
            'neural_replay_walkers.png', 'neural_replay_relight.png']


def _neural_files(p, v, eng):
    from . import neural_replay as NR
    res = NR.run(p, eng, variant=v)
    return NR.files(res, p) if res is not None else []


# =========================================================================== OpenUSD
def _usd_files(p, v, eng):
    from . import usd_export as UE
    res = UE.run(p, eng, variant=v)
    return UE.files(res, p) if res is not None else []


def _usd_names(p, v):
    from . import usd_export as UE
    n = UE.settings_of(p).name
    return [f'{n}.usda', f'{n}.json']               # (and the NCore V4 store's files beside them when set)


# =========================================================================== embedded and thermal intrinsics
def _fw_files(p, v, eng):
    from . import trigger_firmware as FW
    res = FW.run(p, eng, variant=v)
    return FW.files(res, p) if res is not None else []


def _fw_preview(files):
    doc = json.loads(files[0][1])
    lines = [f'line {ln["line"]}: {", ".join(ln["cameras"])} at {ln["rate_hz"]} Hz' for ln in doc.get('lines') or []]
    lines += [f'{t}: {d.get("name")}' for t, d in (doc.get('targets') or {}).items()]
    return '\n'.join(lines) + f'\n\n{len(files)} files (four firmware projects)'


def _edge_files(p, v, eng):
    from . import edge_daemon as ED
    return ED.install_files(p, v)


def _ti_names(p, v):
    from . import thermal_intrinsics as TI
    sw = TI.sweeps_of(p)
    return ['thermal_curves.json', 'thermal_curves.csv'] if any(len(sw.get(c.uid) or []) >= 2
                                                                for c in v.cameras) else []


def _ti_files(p, v, eng):
    from . import thermal_intrinsics as TI
    res = TI.run(p, eng, variant=v)
    return TI.files(res, p) if res is not None else []


# =========================================================================== SOTIF and cleaning
def _cams(v):
    return any(c.visible for c in v.cameras)


def _sotif_files(p, v, eng):
    from . import sotif as SO
    res = SO.run(p, eng, variant=v)
    return SO.files(res, p) if res is not None else []


def _sotif_preview(files):
    doc = json.loads(files[0][1])
    odd = doc.get('odd') or {}
    head = (f'ODD speed limit {odd.get("speed_kmh") or 0:.1f} km/h (target {doc.get("settings", {}).get("target_kmh")} '
            f'km/h), bound by {odd.get("binding") or "nothing"}')
    rows = [f'{r.get("label")}: {r["max_kmh"]:.0f} km/h ({r.get("verdict")})' for r in odd.get('table') or []
            if isinstance(r.get('max_kmh'), (int, float))][:12]
    return head + '\n' + '\n'.join(rows) + '\n\n' + (doc.get('disclaimer') or '')


def _cleaning_names(p, v):
    from . import cleaning as CL
    if not (_cams(v) or any(s.visible and s.kind == 'lidar' for s in v.sensors)):
        return []
    return ['cleaning.json', 'cleaning.csv', 'cleaning.png'] if CL.settings_of(p).devices else []


def _cleaning_files(p, v, eng):
    from . import cleaning as CL
    res = CL.run(p, eng, variant=v)
    return CL.files(res, p) if res is not None else []


def _cleaning_preview(files):
    doc = json.loads(files[0][1])
    t = doc.get('totals') or {}
    lines = [f'{len(doc.get("devices") or [])} devices · tank {t.get("runtime_h", 0):.1f} h over a '
             f'{t.get("shift_h")} h shift · {t.get("average_w", 0):.1f} W average, {t.get("peak_w", 0):.0f} W peak']
    lines += [f'{f["severity"]}: {f["message"]}' for f in doc.get('findings') or [] if f['severity'] != 'good'][:8]
    return '\n'.join(lines)


# =========================================================================== frame, modes, aerodynamics
def _roof(p, v):
    return any(r.visible for r in getattr(p, 'rig', []) or []) or any(s.visible for s in v.sensors)


def _framed(p, v) -> bool:
    """A frame can be built from this variant's roof rig (the synthesis is cheap, well under a second)."""
    if not _roof(p, v):
        return False
    from . import framing as FR
    try:
        return bool(FR.synthesise(p, variant=v).members)
    except Exception:                          # noqa: BLE001 — no frame: nothing to export
        return False


def _framing_files(p, v, eng):
    from . import framing as FR
    res = FR.run(p, eng, variant=v)
    return FR.files(res, p) if res is not None and res.members else []


def _fea_files(p, v, eng):
    from . import aero as AE
    from . import fea as FE
    res = FE.cached(p, v) or FE.run(p, eng, variant=v, wind=AE.run(p, eng, variant=v))
    return FE.files(res, p) if res is not None and res.modes else []


def _aero_files(p, v, eng):
    from . import aero as AE
    res = AE.run(p, eng, variant=v)
    return AE.files(res, p) if res is not None else []


def _findings_preview(files):
    doc = json.loads(files[0][1])
    lines = [f'{f["severity"]}: {f["message"]}' for f in doc.get('findings') or []
             if isinstance(f, dict) and f.get('severity') in ('critical', 'serious', 'warning', 'good')][:10]
    return '\n'.join(lines) or f'{len(files)} files'


FORMATS17 = {
    'thermal': dict(folder='thermal', group='Documents',
                    label='Thermal cameras at night: DRI ranges, R112 photon budget',
                    ext='JSON, CSV, PNG', names=_thermal_names, files=_thermal_files, preview=_thermal_preview),
    'event_cameras': dict(folder='event_cameras', group='Documents', default=False,     # renders a drive: ~30 s
                          label='Event cameras: event rate, bandwidth, motion blur, tunnel exit',
                          ext='JSON, CSV, PNG',
                          names=lambda p, v: (['event_cameras.json', 'event_cameras.csv']
                                              + [f'event_rate_{c.tag}.csv' for c in v.cameras if c.visible
                                                 and getattr(p.module_for(c), 'modality', 'rgb') == 'event']
                                              if _cams(v) else []),
                          files=_event_files, preview=_findings_preview),
    'fmcw': dict(folder='fmcw', group='Research', default=False,
                 label='FMCW lidar: 4D clouds with radial velocity, sun, cross-talk',
                 ext='JSON, CSV, PCD, PLY',
                 names=lambda p, v: (['fmcw.json', 'fmcw_lidars.csv', 'fmcw_sun.csv']
                                     + [f'fmcw_{s.tag}.{e}' for s in _lidars(v) for e in ('pcd', 'ply')]
                                     if _lidars(v) else []),
                 files=_fmcw_files, preview=_fmcw_preview),
    'neural_replay': dict(folder='neural_replay', group='Research', default=False,      # minutes on a GPU
                          label='Neural replay: exact-lens projection, articulated walkers, lights, relighting',
                          ext='JSON, CSV, PNG', names=_neural_names, files=_neural_files,
                          preview=_findings_preview),
    'usd_rig': dict(folder='usd', group='Research', default=False,
                    label='OpenUSD rig (Omniverse, Isaac Sim) and NVIDIA NCore V4 (NuRec)', ext='USDA, JSON, Zarr',
                    names=_usd_names, files=_usd_files,
                    preview=lambda files: files[0][1].decode('utf-8')[:2000]),
    'imaging_radar': dict(folder='imaging_radar', group='Documents',
                          label='Imaging radar: arrays, resolution, ranges over the road, cavity ghosts',
                          ext='JSON, CSV, PNG',
                          names=lambda p, v: (['imaging_radar.json', 'imaging_radar.csv', 'imaging_radar_ghosts.csv',
                                               'imaging_radar.png']
                                              if any(s.visible and s.kind == 'radar' for s in v.sensors) else []),
                          files=_radar_files, preview=_findings_preview),
    'trigger_firmware': dict(folder='trigger_firmware', group='Embedded',
                             label='Trigger-box firmware: RP2040 (C, Rust), STM32F411, Jetson projects',
                             ext='JSON, CSV, C, Rust, CMake', default=False,      # compiles and simulates: 5–15 s
                             names=lambda p, v: (['trigger_firmware.json', 'trigger_schedule.csv']   # + firmware/
                                                 if _cams(v) else []),
                             files=_fw_files, preview=_fw_preview),
    'edge_install': dict(folder='edge', group='Embedded',
                         label='Edge watchdog: configuration, systemd unit, README', ext='JSON, service, MD',
                         names=lambda p, v: ['edge.json', 'vantage-edge.service', 'README.md'] if _cams(v) else [],
                         files=_edge_files, preview=lambda files: files[0][1].decode('utf-8')[:1500]),
    'thermal_intrinsics': dict(folder='thermal_intrinsics', group='Calibration and robotics',
                               label='Thermal intrinsics: curves against temperature (JSON, CSV, OpenCV, Kalibr)',
                               ext='JSON, CSV, PNG, YAML', names=_ti_names, files=_ti_files,
                               preview=lambda files: files[1][1].decode('utf-8')[:1500] if len(files) > 1 else ''),
    'sotif': dict(folder='sotif', group='Documents', label='SOTIF matrix and the ODD speed bound (ISO 21448)',
                  ext='JSON, CSV, HTML, PDF',
                  names=lambda p, v: (['sotif.json', 'sotif_matrix.csv', 'sotif_odd.csv', 'sotif_report.html',
                                       'sotif_report.pdf'] if _cams(v) else []),
                  files=_sotif_files, preview=_sotif_preview),
    'framing': dict(folder='framing', group='Mount points',
                    label='Rig frame: cut list, BOM, CNC plates (DXF), STL parts, STEP assembly',
                    ext='JSON, CSV, PDF, DXF, STL, STEP, PNG',
                    # (the names that are always written; a plate DXF and an STL per part, and the BOM's PDF when a
                    # Qt application runs, come with them)
                    names=lambda p, v: (['framing.json', 'framing_cutlist.csv', 'framing_bom.csv', 'framing.step',
                                         'plates.dxf', 'framing.png'] if _framed(p, v) else []),
                    files=_framing_files, preview=_findings_preview),
    'fea': dict(folder='fea', group='Mount points', label='Frame modes: modes, road jitter per sensor, wind case',
                ext='JSON, CSV, PNG', default=False,
                names=lambda p, v: ['fea.json', 'fea_modes.csv', 'fea_jitter.csv'] if _framed(p, v) else [],
                files=_fea_files, preview=_findings_preview),
    'aero': dict(folder='aero', group='Documents', label='Aerodynamics: rig drag, EV range loss, lock-in (not CFD)',
                 ext='JSON, CSV, PNG',
                 names=lambda p, v: (['aero.json', 'aero_drag.csv', 'aero_shedding.csv', 'aero.png']
                                     if _roof(p, v) else []),
                 files=_aero_files, preview=_findings_preview),
    'cleaning': dict(folder='cleaning', group='Documents',
                     label='Cleaning and de-icing: coverage, tank, heaters, power',
                     ext='JSON, CSV, PNG', names=_cleaning_names, files=_cleaning_files, preview=_cleaning_preview),
}


def export_formats(ExportFormat):
    """The ExportFormat rows (each a folder of files)."""
    return [ExportFormat(k, d['group'], d['label'], d['ext'],
                         (lambda d: lambda p, v: [f'{d["folder"]}/{n}' for n in d['names'](p, v)])(d), 'binary',
                         multi=True, default=d.get('default', True)) for k, d in FORMATS17.items()]


def files(fmt_id, project, variant, engine) -> list:
    d = FORMATS17[fmt_id]
    if not d['names'](project, variant):
        return []
    return [(f'{d["folder"]}/{n}', b) for n, b in d['files'](project, variant, engine)]


def preview(fmt_id, files_, base, size_text) -> tuple:
    if not files_:
        return f'{base}/', 'Nothing to export in this variant.', 0
    total = sum(len(b) for _, b in files_)
    try:
        text = FORMATS17[fmt_id]['preview'](files_)
    except Exception as e:                          # noqa: BLE001 — a preview never breaks the dialog
        text = f'({type(e).__name__}: {e})'
    listing = ', '.join(n.rsplit('/', 1)[-1] for n, _ in files_)
    return f'{base}/{files_[0][0]}', f'{len(files_)} files · {size_text(total)}: {listing}\n\n{text}', total
