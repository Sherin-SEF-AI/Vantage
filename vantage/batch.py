"""Headless batch mode: analyse, export, run load cases and check an installation from the command line.

    vantage project.vantage --analyse report.json
    vantage project.vantage --export out/ --formats extrinsics_json,opencv_yaml,bev_lut
    vantage project.vantage --install-check captures/ --as-built built.vantage
    vantage project.vantage --calibrate-lens views/ --camera F --board 9x6 --square 60 --save out.vantage
    vantage project.vantage --import-calib camchain.yaml F.yaml --save out.vantage
    vantage --sample --load-cases -
    vantage project.vantage --suspension-data corners.csv --suspension cases.json
    vantage project.vantage --blind-volume volume.json
    vantage project.vantage --perception-range ranges.json --variant 2
    vantage project.vantage --rig-health 10 health.json
    vantage project.vantage --sensor-coverage coverage.json
    vantage project.vantage --housings housings.json
    vantage project.vantage --sync-plan sync.json --units mm
    vantage project.vantage --recording-budget budget.json
    vantage project.vantage --power-budget power.json
    vantage project.vantage --gnss gnss.json
    vantage project.vantage --exposure exposure.json
    vantage project.vantage --brackets out/
    vantage project.vantage --dataset out/
    vantage project.vantage --drive-log drive.mcap report/
    vantage project.vantage --as-built-check drive/ report/
    vantage project.vantage --check review/
    vantage project.vantage --lidar-calib out/ --lidar-calib-pair F L1 --lidar-calib-folder boards/
    vantage project.vantage --fleet-import V3 install_report.json --fleet fleet/ --save project.vantage
    vantage project.vantage --terrain road.json
    vantage project.vantage --harness harness.json
    vantage project.vantage --observability observability/ --verify 30
    vantage project.vantage --critical-targets envelopes/
    vantage project.vantage --materials materials.json
    vantage project.vantage --weather weather.json
    vantage project.vantage --perception-range night.json --condition night
    vantage --watch project.vantage --on-change "make perception-test"
    vantage project.vantage --ci-templates ci/
    vantage project.vantage --reproduce ranges.json check.json --tolerance 1e-9 'blind_m2=0.005' '*_pct=0.1%'
    vantage project.vantage --experiment experiment_spec.json experiment/
    vantage --validate report/ --validate-quick
    vantage project.vantage --carla-rig carla/
    vantage project.vantage --carla-run drive/ --carla-python ~/carla/venv/bin/python

Every JSON object a batch run writes (and every JSON the exporters write) carries a "provenance" block
(core.provenance): the version and code, the libraries, the project's and variant's content hashes, the study's
settings and seeds, this command with every path reduced to its file name, the inputs' hashes, the time and the
result's own hash. --reproduce runs again what wrote a result (the same project given again, its command and settings)
and reports per number whether it reproduces, exactly or within --tolerance, and why not.

The exit code is 0 when everything is fine, 2 when an analysis or export finds critical issues in the plan (or the
recording budget finds the recorder or a sensor link cannot keep up, or the power budget finds the DC-DC converter or a
fuse over its rating, the GNSS study finds an antenna with no sky, a camera cannot take a printed bracket as planned,
a suspension load case lifts a wheel or exceeds the gross vehicle mass, a camera sits inside a housing, no camera
resolves a critical target, a radar looks into metal over half its field, on the road the body or a sensor would
touch it or no camera resolves a speed bump or kerb, or a routed cable is longer than its link allows, no module cable
reaches or a GNSS cable has no RF margin, or with calibration targets planned a camera sees no board whole),
3 when an installation check finds a camera that has to be re-mounted (or could not be solved), the rig health check
finds a camera faulty (or without a stream), the fleet has a sensor to re-mount in its latest record or a drive log
shows a fault or a lidar has to be re-mounted (or its pose is not determined by the board poses) or the as-built check
finds the plan's coverage does not hold, and 4 when a lens calibration fails, an imported calibration matches no
camera, a fleet record, a drive log, an as-built check's recording, a suspension data file or a lidar calibration's
captures cannot be read or the
sample dataset cannot be written, and 5 when, with --check, a must requirement fails or cannot be evaluated.
With --watch the exit code is the last run's: the command's own when it ran, else 5 when the requirements check fails,
4 when the project could not be read or a format not written, 130 when stopped during the command.
With --reproduce: 0 when the result reproduces (exactly or within the tolerances), 6 when it does not, 4 when it
cannot be tried (no provenance, the project or an input file not found, a live measurement). With --validate: 6 when a
check of the models against its reference fails. With --experiment: 4 when the spec cannot be read or does not fit the
project. With --carla-rig or --carla-run: 2 when the variant has nothing to simulate; with --carla-run, 7 when no CARLA
server answers, CARLA's client Python is missing or the run fails.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import re
import sys
import time

import numpy as np


def _out(doc, dest):
    doc = _with_provenance(doc, dest)
    text = json.dumps(doc, indent=1, ensure_ascii=False, default=_jsonable)
    if not dest or dest == '-':
        print(text)
    else:
        os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
        with open(dest, 'w', encoding='utf-8') as fh:
            fh.write(text + '\n')
        print(f'wrote {dest}', file=sys.stderr)


def _with_provenance(doc, dest):
    """doc with its provenance block (core.provenance) as the last key while a batch run records one."""
    from .core import provenance as PV
    b = PV.current_batch()
    if not isinstance(doc, dict) or b is None:
        return doc
    return PV.stamp(doc, b.get('project'), output=PV.batch_output(doc, dest))


def _stamped(paths, option, folder):
    """The files a folder output wrote, each JSON object among them stamped with its provenance; returns paths."""
    from .core import provenance as PV
    b = PV.current_batch()
    if b is None or not folder:
        return paths
    base = os.path.abspath(folder)
    for p in paths:
        if str(p).lower().endswith('.json') and os.path.isfile(p):
            rel = os.path.relpath(os.path.abspath(p), base).replace(os.sep, '/')
            PV.stamp_path(p, b.get('project'), output={'option': option, 'name': rel, 'in_folder': True})
    return paths


def _stamp_file(path, option):
    """A JSON file written by a batch option (not through _out) stamped with its provenance."""
    from .core import provenance as PV
    b = PV.current_batch()
    if b is not None and path and os.path.isfile(path):
        PV.stamp_path(path, b.get('project'), output={'option': option, 'name': os.path.basename(path),
                                                      'in_folder': False})


def _jsonable(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f'{type(o).__name__} is not JSON serialisable')


def _r(v, dp=3):
    return round(float(v), dp)


def load_project(args):
    from .core import samples
    from .core.project import Project
    if args.sample:
        return samples.tigor_sample_project()
    if args.template:
        return samples.template_project(args.template)
    if args.path and args.path.lower().endswith('.vantage'):
        return Project.load(args.path)
    if args.path:
        raise SystemExit('Batch mode needs a .vantage project (import vehicle models in the app).')
    raise SystemExit('Give a .vantage project, --sample or --template.')


def pick_variant(project, name):
    if not name:
        return
    for i, v in enumerate(project.variants):
        if v.name == name or v.uid == name or str(i + 1) == name:
            project.active_uid = v.uid
            return
    raise SystemExit(f'No variant {name!r}. Variants: ' + ', '.join(v.name for v in project.variants))


def analysis_doc(project, res, eng) -> dict:
    from .core import rules
    from .core.targets import evaluate_targets
    vg = eng.vehicle_geom(project)
    mounts = {}
    for c in project.variant.cameras:
        try:
            mounts[c.uid] = rules.mount_info(project, c, vg)
        except Exception:
            pass
    issues = rules.collect_issues(project, res, vg, mounts)
    doc = {'schema': 'vantage.analysis/1', 'project': project.name, 'variant': project.variant.name,
           'time': time.strftime('%Y-%m-%dT%H:%M:%S'), 'cameras': [], 'issues': [], 'seams': [], 'link_budget': []}
    if res is not None:
        m = res.metrics
        doc['metrics'] = {'roi_m2': _r(m['roi_area']), 'blind_m2': _r(m['blind_m2']), 'covered_pct': _r(m['covered_pct'], 2),
                          'multi_pct': _r(m['multi_pct'], 2), 'min_density_px_cm': _r(m['min_density'], 2),
                          'min_ok_pct': _r(m['min_ok_pct'], 2), 'target_ok_pct': _r(m['target_ok_pct'], 2),
                          'in_focus_pct': _r(m['in_focus_pct'], 2), 'cell_m': _r(m['cell'], 3)}
        tags = {c.uid: c.tag for c in project.variant.cameras}
        doc['seams'] = [{'between': [tags.get(s.a, s.a), tags.get(s.b, s.b)], 'azimuth_deg': _r(s.angle, 1),
                         'density_px_cm': _r(s.density, 2), 'auto': bool(s.auto)} for s in res.seams]
    for c in project.variant.cameras:
        row = {'tag': c.tag, 'port': c.port, 'module': c.module_id, 'lens': c.lens_id,
               'pose': {'x': c.x, 'y': c.y, 'z': c.z, 'yaw': c.yaw, 'pitch': c.pitch, 'roll': c.roll},
               'route_m': _r(rules.route_length(project, c), 2), 'cable_m': c.cable_m,
               'gbps': _r(rules.camera_gbps(project, c), 3)}
        if res is not None and c.uid in res.per_cam:
            p = res.per_cam[c.uid]
            row.update(seen_m2=_r(p['seen_m2']), unique_m2=_r(p['unique_m2']), min_density_px_cm=_r(p['min_dens'], 2),
                       if_removed_blind_m2=_r(p['if_removed_blind']))
        doc['cameras'].append(row)
    doc['issues'] = [{'severity': i.severity, 'message': i.message, 'objects': i.objects} for i in issues]
    doc['link_budget'] = [{'deserializer': n, 'ports': ports, 'cameras': tags_, 'used_gbps': _r(used, 3),
                           'capacity_gbps': cap, 'load_pct': _r(100 * used / cap if cap else 0, 1)}
                          for n, ports, tags_, used, cap, worst, link in rules.link_budget(project)]
    if res is not None:
        from .core import stitched_objects as so
        sres = so.analyse(project, eng, res)
        doc['stitched_objects'] = so.to_doc(sres, so.advise(project, eng, res))
        iss = so.issue(sres)
        if iss is not None:
            issues = sorted(issues + [iss], key=lambda i: rules.SEVERITY_ORDER.get(i.severity, 9))
            doc['issues'] = [{'severity': i.severity, 'message': i.message, 'objects': i.objects} for i in issues]
    if project.targets:
        st = evaluate_targets(eng, project)
        names = {t.uid: t.name for t in project.targets}
        doc['targets'] = [{'name': names.get(s.uid, s.uid), 'ok': bool(s.ok),
                           'seen_by': [{'tag': tag, 'visible': _r(frac, 2), 'min_density': _r(d, 2)}
                                       for _, tag, frac, d in s.seen_by]} for s in st]
    return doc, issues


def load_cases_doc(project, eng) -> dict:
    from .core.loadcases import evaluate, cases_of
    rs = evaluate(eng, project, cases_of(project), cell=max(project.analysis.cell_m, 0.05))
    return {'schema': 'vantage.load_cases/1', 'variant': project.variant.name,
            'note': 'Axle drops are the project\'s load cases (or examples when none are set).',
            'cases': [{'name': r.name, 'front_drop_mm': r.front_mm, 'rear_drop_mm': r.rear_mm, 'pitch_deg': _r(r.pitch_deg),
                       'blind_m2': _r(r.blind_m2), 'delta_blind_m2': _r(r.delta_blind), 'min_density': _r(r.min_density, 2),
                       'ground_shift_cm': _r(r.ground_shift_cm, 1), 'seam_ghost_cm': _r(r.seam_ghost_cm, 1),
                       'status': r.status} for r in rs]}


def blind_volume_doc(project, eng, res) -> dict:
    from .core import volume as vol
    st = vol.VolumeSettings.from_dict(project.analysis.volume)
    vr = vol.analyse_volume(project, eng, res, st.height, st.dz, st.cell or None)
    if vr is None:
        raise SystemExit('The blind volume needs a vehicle model.')
    return vol.to_doc(vr, project)


def perception_range_doc(project, eng, res) -> dict:
    from .core import perception as PR
    pr = PR.analyse(project, eng, res, PR.PerceptionSettings.from_dict(project.analysis.perception))
    return PR.to_doc(pr, project)


def install_check(project, eng, res, folder, st, lut_out=None, balance=True, blend='linear'):
    from .core import capture as cap
    from .core.install import check_all
    from .core.joint import pose_dict, report
    from .core.mathutil import rot_zyx
    cams = [c for c in project.variant.cameras if c.visible]
    found = cap.match_folder(folder, [c.tag for c in cams])
    images, unreadable = {}, {}
    for c in cams:
        if c.tag in found:
            try:
                images[c.uid] = cap.load_image(found[c.tag])
            except Exception as e:      # one bad file must not stop the check of the others
                unreadable[c.uid] = str(e)
    results, jr = check_all(project, cams, images, st)
    rows = []
    for c in cams:
        row = {'tag': c.tag, 'planned': {'x': c.x, 'y': c.y, 'z': c.z, 'yaw': c.yaw, 'pitch': c.pitch, 'roll': c.roll}}
        r = results.get(c.uid)
        if r is None:
            row.update(status='serious', message=f'Could not read {os.path.basename(found[c.tag])}: {unreadable[c.uid]}'
                       if c.uid in unreadable else 'No image for this camera in the folder')
            rows.append(row)
            continue
        row.update(image=os.path.basename(found[c.tag]), status=r.status, message=r.message,
                   boards=[b.name for b in r.boards], corners=r.n_corners)
        if c.calib:
            row['lens'] = {'unit_calibration': c.calib.get('serial') or True, 'rms_px': c.calib.get('rms_px')}
        if r.pos is not None:
            row['measured'] = {'x': _r(r.pos[0], 4), 'y': _r(r.pos[1], 4), 'z': _r(r.pos[2], 4), 'yaw': _r(r.yaw),
                               'pitch': _r(r.pitch), 'roll': _r(r.roll)}
            row['delta_mm'] = [_r(v, 1) for v in r.dpos_mm]
            row['delta_deg'] = [_r(v, 3) for v in r.dang_deg]
            row['rms_px'] = _r(r.rms_px)
            row['rotation_error_deg'] = _r(r.rot_err_deg)
            if r.single is not None:
                row['own_solve'] = pose_dict(*r.single)
        rows.append(row)
    doc = {'schema': 'vantage.install_check/1', 'project': project.name, 'variant': project.variant.name,
           'time': time.strftime('%Y-%m-%dT%H:%M:%S'), 'folder': os.path.abspath(folder),
           'tolerance': {'position_mm': st.pos_tol_mm, 'angle_deg': st.ang_tol_deg, 'boards': st.min_boards},
           'cameras': rows}
    if jr is not None and jr.poses:
        doc['joint'] = report(jr)
    if lut_out and res is not None:
        from .core.stitch import build_lut, export_lut, stitch, estimate_gains
        poses = {u: (rot_zyx(r.yaw, r.pitch, r.roll), r.pos) for u, r in results.items() if r.pos is not None}
        lut = build_lut(project, eng, res, res=0.02, poses=poses)
        paths = _stamped(export_lut(lut, lut_out, project), '--lut-out', lut_out)
        gains = estimate_gains(lut, images) if balance else None
        bev = stitch(lut, images, gains=gains, blend=blend)
        cap.save_image(os.path.join(lut_out, 'birds-eye-measured.png'), bev)
        doc['remap_tables'] = os.path.abspath(lut_out)
        doc['remap_files'] = len(paths) + 1
        if gains:
            tags = {c.uid: c.tag for c in cams}
            doc['exposure_gains'] = {tags[u]: [_r(x, 4) for x in g] for u, g in gains.items()}
    return doc, results


def calibrate_lens(project, args) -> tuple[dict, bool]:
    """--calibrate-lens: one camera's unit calibration from a folder of board views."""
    from .core import calib, capture as cap
    cam = project.variant.by_tag(args.camera or '')
    if cam is None:
        raise SystemExit('--calibrate-lens needs --camera TAG. Cameras: ' + ', '.join(c.tag for c in project.variant.cameras))
    try:
        cols, rows = (int(x) for x in args.board.lower().split('x'))
    except ValueError:
        raise SystemExit(f'--board {args.board!r}: give inner corners as COLSxROWS, e.g. 9x6')
    folder = args.calibrate_lens
    files = [f for f in sorted(os.listdir(folder)) if f.lower().endswith(cap.IMAGE_EXTS)]
    images, unreadable = [], []
    for f in files:
        try:
            images.append((f, cap.load_image(os.path.join(folder, f))))
        except Exception as e:
            unreadable.append({'image': f, 'error': str(e)})
    m, base = project.module_for(cam), project.nominal_lens_for(cam)
    r = calib.calibrate(images, (cols, rows), args.square, m, base, serial=args.serial or '', source='images')
    doc = {'schema': 'vantage.lens_calibration/1', 'camera': cam.tag, 'folder': os.path.abspath(folder),
           'board': {'inner_corners': [cols, rows], 'square_mm': args.square}, 'ok': r.ok, 'message': r.message,
           'advice': r.advice, 'unreadable': unreadable, 'views': [{'image': v.name, 'used': v.used, 'corners': 0 if v.corners is None else len(v.corners),
                                          'rms_px': None if not v.used else _r(v.rms, 4), 'note': v.note} for v in r.views]}
    if r.ok:
        doc['calibration'] = calib.clean_record(r.record)
        doc['compared_with_library'] = [dict(zip(('parameter', 'library', 'unit', 'difference'), row))
                                        for row in calib.compare_rows(m, base, r.record)]
        cam.calib = calib.clean_record(r.record)
        if args.calib_out:
            with open(args.calib_out, 'w', encoding='utf-8') as fh:
                fh.write(calib.to_opencv_yaml(r.record, cam.tag) if not args.calib_out.lower().endswith('.json') else
                         calib.to_json(project, [cam]))
            _stamp_file(args.calib_out, '--calib-out')
            print(f'wrote {args.calib_out}', file=sys.stderr)
    return doc, r.ok


def import_calibrations(project, paths) -> tuple[dict, bool]:
    """--import-calib: unit calibrations from files, matched to the cameras by name, file name or serial."""
    from .core import calib
    recs, errors = [], []
    for pth in paths:
        try:
            recs += calib.read_file(pth)
        except Exception as e:
            errors.append({'file': pth, 'error': str(e)})
    cams = list(project.variant.cameras)
    matched, unmatched = calib.match(recs, cams)
    rows = []
    for c in cams:
        r = matched.get(c.uid)
        if r is None:
            continue
        probs = calib.problems(r, project.module_for(c), project.nominal_lens_for(c))
        fatal = calib.fatal_problems(r, project.module_for(c))
        row = {'camera': c.tag, 'file': os.path.basename(r.get('file', '')), 'name_in_file': r.get('hint', ''),
               'fx': _r(r['fx'], 3), 'cx': _r(r['cx'], 3), 'cy': _r(r['cy'], 3), 'warnings': probs}
        if fatal:
            errors.append(dict(row, error='; '.join(fatal)))
            continue
        c.calib = calib.clean_record(r)
        rows.append(row)
    doc = {'schema': 'vantage.calibration_import/1', 'applied': rows, 'errors': errors,
           'unmatched': [{'file': os.path.basename(r.get('file', '')), 'name_in_file': r.get('hint', '')} for r in unmatched]}
    return doc, bool(rows) and not unmatched and not errors


def as_built_variant(project, results):
    from .core.mathutil import wrap180
    from .core.project import Variant, new_uid
    cur = project.variant
    v = Variant.from_dict(cur.to_dict())
    v.uid = new_uid()
    v.name = f'{cur.name} · as-built'
    for c in v.cameras:
        r = results.get(c.uid)
        c.uid = new_uid()
        if r is not None and r.pos is not None:
            c.x, c.y, c.z = (round(float(t), 4) for t in r.pos)
            c.yaw, c.pitch, c.roll = round(wrap180(r.yaw), 3), round(r.pitch, 3), round(wrap180(r.roll), 3)
        c.link_uid = ''
    v.seams = {}
    project.variants.append(v)
    project.active_uid = v.uid
    return v


def parser():
    """The batch command line as the app reads it (the project, --sample, --template and the batch options): what a
    provenance block's command is written from and read back with."""
    import argparse
    ap = argparse.ArgumentParser(prog='vantage', add_help=False)
    ap.add_argument('path', nargs='?')
    ap.add_argument('--sample', action='store_true')
    ap.add_argument('--template')
    add_arguments(ap)
    return ap


def run(args) -> int:
    """A batch run, recorded for the provenance blocks of what it writes (core.provenance); --reproduce checks one."""
    from .core import provenance as PV
    PV.begin_batch(args, parser())
    try:
        return reproduce(args) if getattr(args, 'reproduce', None) else _run(args)
    finally:
        PV.end_batch()


def _run(args) -> int:
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv[:1])
    from .ui.theme import load_fonts
    load_fonts()
    if args.watch:
        from .core import automation as AU
        return AU.watch_cli(args)
    if args.on_change:
        raise SystemExit('--on-change runs with --watch PROJECT.vantage.')
    if args.verify and args.observability is None:
        raise SystemExit('--verify checks the predictions of --observability.')
    if (args.validate_models or args.validate_quick) and args.validate is None:
        raise SystemExit('--validate-models and --validate-quick go with --validate.')
    if args.carla_python and not args.carla_run:
        raise SystemExit('--carla-python goes with --carla-run.')
    if args.validate is not None and not (args.path or args.sample or args.template):
        return _validate(args, None)                    # no project: the checks build their own references
    from .core import exporters as ex
    from .core.analysis import Engine
    from .core.install import InstallSettings
    project = load_project(args)
    from .core import provenance as PV
    PV.note_project(project)            # its content hash as given, before this command changes it
    pick_variant(project, args.variant)
    if args.condition:                  # the weather the perception range, the sample dataset and --weather use
        project.analysis.weather = dict(project.analysis.weather or {}, condition=args.condition)
    code = 0
    cal_docs = []
    if args.import_calib:
        cdoc, ok = import_calibrations(project, args.import_calib)
        cal_docs.append(('import', cdoc))
        code = code if ok else 4
    if args.calibrate_lens:
        cdoc, ok = calibrate_lens(project, args)
        cal_docs.append(('calibration', cdoc))
        code = code if ok else 4
    if args.suspension_data:
        from .core import suspension as SU
        try:
            sst, what, warns = SU.load_file(args.suspension_data, SU.settings_for(project))
        except (OSError, ValueError) as e:
            print(f'suspension data: {e}', file=sys.stderr)
            code = max(code, 4)
        else:
            project.analysis.suspension = sst.to_dict()
            print(f'read the {what} from {os.path.basename(args.suspension_data)}', file=sys.stderr)
            for w in warns:
                print(f'  note: {w}', file=sys.stderr)
    if cal_docs:
        doc_ = cal_docs[0][1] if len(cal_docs) == 1 else {'schema': 'vantage.batch_calibration/1', **dict(cal_docs)}
        _out(doc_, '-' if not args.report or args.install_check else args.report)
    eng = Engine()
    t0 = time.time()
    res = eng.analyse(project)
    print(f'analysed {project.variant.name}: {len(project.variant.cameras)} cameras in {time.time() - t0:.1f} s',
          file=sys.stderr)
    doc, issues = analysis_doc(project, res, eng)
    if any(i.severity == 'critical' for i in issues) and (args.analyse is not None or args.export):
        code = max(code, 2)
    if args.analyse is not None:
        _out(doc, args.analyse)
    if args.load_cases is not None:
        _out(load_cases_doc(project, eng), args.load_cases)
    if args.housings is not None:
        from .core import housings as HS
        hres = HS.run(project, eng)
        _out(HS.to_doc(hres, project), args.housings)
        for iv in hres.cameras:
            print(f'  {iv.tag:<4} {HS.in_view_text(iv)[0]}', file=sys.stderr)
        if any(iv.inside for iv in hres.cameras):
            code = max(code, 2)
    if args.suspension is not None:
        from .core import suspension as SU
        sr = SU.run(project, eng, cell=max(project.analysis.cell_m, 0.05))
        _out(SU.to_doc(sr, project), args.suspension)
        for f in sr.findings:
            if f['severity'] not in ('good', 'note'):
                print(f'  {f["severity"]:<8} {f["message"]}', file=sys.stderr)
        if any(f['kind'] in ('lift', 'gross') for f in sr.findings):
            code = max(code, 2)
    if args.blind_volume is not None:
        _out(blind_volume_doc(project, eng, res), args.blind_volume)
    if args.perception_range is not None:
        _out(perception_range_doc(project, eng, res), args.perception_range)
    if args.sensor_coverage is not None:
        from .core import sensor_coverage as SC
        ctx = ex._context(project, None, ex.ExportOptions(frame=args.frame, units=args.units, angles=args.angles,
                                                          engine=eng))
        cres = SC.run(project, eng)
        _out(ex.sensor_coverage_doc(ctx, cres), args.sensor_coverage)
        m = cres.metrics
        print(f'  sector checks: ' + ', '.join(f'{m[v]} {v}' for v in SC.VERDICTS), file=sys.stderr)
        for L_ in cres.lidars:
            print(f'  {L_.tag}: {100 * L_.blocked:.0f} % of its points land on the vehicle', file=sys.stderr)
    if args.sync_plan is not None:
        from .core import timing as TM
        ctx = ex._context(project, None, ex.ExportOptions(frame=args.frame, units=args.units, angles=args.angles))
        sdoc = ex.sync_plan_doc(ctx)
        _out(sdoc, args.sync_plan)
        s = TM.summary(TM.plan(project))
        if s.get('no_sweep_plan'):
            print(f'  no sweep plan: {s["no_sweep_plan"]}', file=sys.stderr)
        else:
            print(f'  worst misalignment {s["worst_misalignment_ms"]} ms ({s["worst_travel_cm"]} cm at '
                  f'{s["speed_kmh"]:g} km/h); suggested lines: {s["suggested_worst_misalignment_ms"]} ms', file=sys.stderr)
    if args.recording_budget is not None:
        from .core import recording as RB
        rb = RB.budget(project)
        _out(RB.to_doc(rb, project), args.recording_budget)
        for f in rb.findings:
            print(f'  {f["severity"]:<8} {f["message"]}', file=sys.stderr)
        if any(f['severity'] == 'critical' for f in rb.findings):
            code = max(code, 2)
    if args.power_budget is not None:
        from .core import power as PW
        pb = PW.run(project)
        _out(PW.to_doc(pb, project), args.power_budget)
        for f in pb.findings:
            if f['severity'] != 'good':
                print(f'  {f["severity"]:<8} {f["message"]}', file=sys.stderr)
        if any(f['severity'] == 'critical' for f in pb.findings):
            code = max(code, 2)
    if args.exposure is not None:
        from .core import exposure as XP
        xres = XP.run(project, eng)
        _out(XP.to_doc(xres, project), args.exposure)
        if args.exposure not in ('', '-'):
            png_path = os.path.splitext(args.exposure)[0] + '.png'
            with open(png_path, 'wb') as fh:
                fh.write(XP.png(xres))
            print(f'wrote {png_path}', file=sys.stderr)
        for f in xres.findings:
            if f['severity'] != 'good':
                print(f'  {f["severity"]:<8} {f["message"]}', file=sys.stderr)
    if args.gnss is not None:
        from .core import gnss as GN
        gr = GN.run(project, eng)
        _out(GN.to_doc(gr, project), args.gnss)
        for f in gr.findings:
            if f['severity'] not in ('good', 'note'):
                print(f'  {f["severity"]:<8} {f["message"]}', file=sys.stderr)
        if any(f['severity'] == 'critical' for f in gr.findings):
            code = max(code, 2)
    if args.observability is not None:
        from .core import observability as OB
        ores = OB.run(project, eng)
        if args.verify:
            OB.monte_carlo(ores, trials=args.verify)
        dest = args.observability
        if dest not in ('', '-') and not dest.lower().endswith('.json'):
            for pth in _stamped(OB.export(ores, project, dest), '--observability', dest):
                print(f'wrote {pth}', file=sys.stderr)
        else:
            _out(OB.to_doc(ores, project), dest)
        for c in ores.cameras:
            print(f'  {c["tag"]:<4} {c["verdict"]:<8} ' + (f'worst {c["worst_axis"]} {c["ratio"]:.2f}× the requirement'
                  if c['sigma'] else 'sees no board whole'), file=sys.stderr)
        if ores.mc is not None:
            print(f'  Monte Carlo: {"the predictions hold" if ores.mc["agree"] else "the spread differs"}',
                  file=sys.stderr)
        if project.targets and any(c['verdict'] == 'critical' for c in ores.cameras):
            code = max(code, 2)
    if args.harness is not None:
        from .core import harness as HN
        hn = HN.run(project, eng)
        _out(HN.to_doc(hn, project), args.harness)
        for f in hn.findings:
            if f['severity'] != 'good':
                print(f'  {f["severity"]:<8} {f["message"]}', file=sys.stderr)
        if any(f['kind'] in ('limit', 'rf', 'option') and f['severity'] == 'serious' for f in hn.findings):
            code = max(code, 2)
    if args.terrain is not None:
        from .core import terrain as TR
        tr = TR.run(project, eng)
        _out(TR.to_doc(tr, project), args.terrain)
        for f in tr.findings:
            if f['severity'] not in ('good', 'note'):
                print(f'  {f["severity"]:<8} {f["message"]}', file=sys.stderr)
        if any(f['kind'] in ('touch', 'device', 'unseen') and f['severity'] in ('serious', 'critical')
               for f in tr.findings):
            code = max(code, 2)
    if args.critical_targets is not None:
        from .core import critical as CR
        cres = CR.run(project, eng)
        dest = args.critical_targets
        if dest not in ('', '-') and not dest.lower().endswith('.json'):
            for pth in _stamped(CR.export(cres, project, dest), '--critical-targets', dest):
                print(f'wrote {pth}', file=sys.stderr)
        else:
            _out(CR.to_doc(cres, project), dest)
        if not cres.targets:
            print('  no critical targets: place them in Analysis ▸ Critical targets', file=sys.stderr)
        for r in cres.targets:
            print(f'  {r.name:<40} ' + (f'resolved by {", ".join(r.met_by)}' if r.cameras_ok else 'NOT RESOLVED'),
                  file=sys.stderr)
        if any(not r.cameras_ok for r in cres.targets):
            code = max(code, 2)
    if args.materials is not None:
        from .core import materials as MT
        mres = MT.run(project, eng)
        _out(MT.to_doc(mres, project), args.materials)
        for r in MT.summary(mres)['radars']:
            loss = r['two_way_loss_db']
            print(f'  {r["tag"]:<4} {r["mount"]:<8} ' + (f'{r["looks_through"]}, ' if r['looks_through'] else '')
                  + (f'{loss:.1f} dB two-way, ' if loss is not None else '')
                  + 'range ' + ' / '.join('–' if x is None else f'{x:.0f}' for x in r['range_m'])
                  + f' m on a car, multipath risk {r["multipath_risk"]}', file=sys.stderr)
        for f in mres.findings:
            if f['severity'] != 'good':
                print(f'  {f["severity"]:<8} {f["message"]}', file=sys.stderr)
        if any(f['severity'] == 'critical' for f in mres.findings):
            code = max(code, 2)
    if args.weather is not None:
        from .core import weather as WX
        wres = WX.run(project, eng)
        _out(WX.to_doc(wres, project), args.weather)
        for f in wres.findings:
            if f['severity'] not in ('good', 'note'):
                print(f'  {f["severity"]:<8} {f["message"]}', file=sys.stderr)
    if args.export:
        if args.formats:
            ids = [f.strip() for f in args.formats.split(',') if f.strip()]
        else:
            ids = [f.id for f in ex.FORMATS if f.group != 'Images' and f.default]
        bad = [i for i in ids if i in ('viewport_png', 'bev_png', 'camera_views_png')]
        if bad:
            raise SystemExit(f'{", ".join(bad)} are rendered by the app window; export them from the app.')
        opts = ex.ExportOptions(out_dir=args.export, frame=args.frame, units=args.units, angles=args.angles,
                                issues=issues, engine=eng)
        try:
            paths = ex.export(ids, project, res, opts)
        except ValueError as e:
            raise SystemExit(str(e))
        for p in paths:
            print(p)
    if args.install_check:
        st = InstallSettings(pos_tol_mm=args.pos_tol, ang_tol_deg=args.ang_tol, min_boards=args.min_boards,
                             joint=not args.no_joint, board_sigma_mm=args.board_tol,
                             board_sigma_deg=max(0.2, args.board_tol / 20.0))
        idoc, results = install_check(project, eng, res, args.install_check, st, args.lut_out, balance=not args.no_balance,
                                      blend=args.blend)
        _out(idoc, args.report or '-')
        if any(r['status'] != 'good' and r['status'] != 'warning' for r in idoc['cameras']):
            code = max(code, 3)
        if args.as_built:
            v = as_built_variant(project, results)
            project.save(args.as_built)
            print(f'saved {args.as_built} with variant {v.name}', file=sys.stderr)
    if args.rig_health:
        hdoc = rig_health_doc(project, eng, args.rig_health)
        _out(hdoc, args.rig_health[1] if len(args.rig_health) > 1 else '-')
        if any(c['status'] in ('fault', 'off') for c in hdoc['cameras'].values()):
            code = max(code, 3)
    if args.brackets:
        from .core import brackets as BR
        bres = BR.run(project, eng)
        paths = _stamped(BR.export(bres, project, args.brackets), '--brackets', args.brackets)
        print(f'wrote {len(paths)} files to {os.path.join(args.brackets, "brackets")}', file=sys.stderr)
        for b in bres.brackets:
            print(f'  {b.tag:<4} {b.status:<9} {b.message}', file=sys.stderr)
        if any(b.status in ('standoff', 'floating', 'inside', 'face') or b.worst() in ('critical', 'serious')
               for b in bres.brackets):
            code = max(code, 2)
    if args.dataset is not None:
        from .core import dataset as DSET
        try:
            dres = DSET.generate(project, eng, None, args.dataset,
                                 lambda f, text='': print(f'\r  dataset {f * 100:5.1f} % {text:<28}', end='',
                                                          file=sys.stderr))
            print(file=sys.stderr)
            _stamped([os.path.join(dres.out_dir, 'dataset.json')], '--dataset', dres.out_dir)
            s = DSET.summary(dres)
            print(f'  dataset: {s["samples"]} samples, {s["annotations"]} annotations, {s["size_mb"]["total"]} MB '
                  f'in {s["seconds"]} s → {dres.out_dir}', file=sys.stderr)
            for n in dres.notes:
                print(f'  note: {n}', file=sys.stderr)
        except ValueError as e:
            print(f'  dataset: {e}', file=sys.stderr)
            code = max(code, 4)
    if args.drive_log is not None:
        from .core import drivelog as DLG
        dest = args.drive_log[1] if len(args.drive_log) > 1 else '-'
        try:
            lres = DLG.run(project, args.drive_log[0], engine=eng, variant=project.variant)
        except (OSError, ValueError) as e:
            print(f'drive log: {e}', file=sys.stderr)
            code = max(code, 4)
        else:
            if dest not in ('', '-') and not dest.lower().endswith('.json'):
                for pth in _stamped(DLG.export(lres, project, dest), '--drive-log', dest):
                    print(f'wrote {pth}', file=sys.stderr)
            else:
                _out(DLG.to_doc(lres, project), dest)
            for f in lres.findings:
                if f['severity'] != 'ok':
                    print(f'  {f["severity"]:<8} {f["message"]}', file=sys.stderr)
            if any(f['severity'] == 'fault' for f in lres.findings):
                code = max(code, 3)
    if args.as_built_check is not None:
        from .core import asbuilt as ABL
        dest = args.as_built_check[1] if len(args.as_built_check) > 1 else '-'
        try:
            ares = ABL.run(project, eng, ABL.settings_of(project), project.variant, path=args.as_built_check[0])
        except (OSError, ValueError) as e:
            print(f'as-built check: {e}', file=sys.stderr)
            code = max(code, 4)
        else:
            if dest not in ('', '-') and not dest.lower().endswith('.json'):
                for pth in _stamped(ABL.export(ares, project, dest), '--as-built-check', dest):
                    print(f'wrote {pth}', file=sys.stderr)
            else:
                _out(ABL.to_doc(ares, project), dest)
            for f in ares.findings:
                if f['severity'] != 'good':
                    print(f'  {f["severity"]:<8} {f["message"]}', file=sys.stderr)
            print(f'as-built check: {ABL.VERDICT_TEXT.get(ares.verdict, ares.verdict)}', file=sys.stderr)
            if args.save and ares.asbuilt is not None:
                project.variants.append(ares.asbuilt)
            if ares.verdict == 'does_not_hold':
                code = max(code, 3)
    if args.check is not None:
        from .core import requirements as RQ
        dest = None if args.check == '-' else args.check
        cdoc, ccode = RQ.check(project, eng, dest, variant=project.variant if args.variant else None)
        if dest is None:
            _out(cdoc, '-')
        elif not dest.lower().endswith('.json'):
            _stamped([os.path.join(dest, 'requirements.json')], '--check', dest)
            print(f'wrote {dest}/requirements.json, requirements.csv, review_pack.pdf', file=sys.stderr)
        else:
            _stamp_file(dest, '--check')
        for r in cdoc['requirements']:
            print(f'  {r["status"]:<13} {r["priority"]:<6} {r["id"]:<4} {r["text"]}'
                  + (f' ({r["reason"]})' if r['reason'] else ''), file=sys.stderr)
        c = cdoc['counts']
        print(f'requirements: {cdoc["verdict_text"]} · must {c["must"]["pass"]}/{c["must"]["total"]} · should '
              f'{c["should"]["pass"]}/{c["should"]["total"]}', file=sys.stderr)
        code = max(code, ccode)
    if args.lidar_calib:
        from .core import lidar_calib as LC
        st = LC.LidarCalibSettings.from_dict(project.analysis.lidar_calib)
        cam, lid = args.lidar_calib_pair or (st.camera, st.lidar)
        try:
            lres = (LC.calibrate_folder(project, cam, lid, args.lidar_calib_folder, st) if args.lidar_calib_folder
                    else LC.run(project, eng, LC.LidarCalibSettings.from_dict(dict(st.to_dict(), camera=cam,
                                                                                    lidar=lid))))
        except (ValueError, OSError) as e:
            print(f'lidar calibration: {e}', file=sys.stderr)
            code = max(code, 4)
        else:
            for pth in _stamped(LC.export(lres, project, args.lidar_calib), '--lidar-calib', args.lidar_calib):
                print(pth)
            s = LC.summary(lres)
            print(f'  {s.get("camera")} – {s.get("lidar")}: {lres.verdict}', file=sys.stderr)
            if lres.verdict in ('serious', 'unobservable', 'failed'):
                code = max(code, 3)
    if args.fleet_import:
        from .core import fleet as FL
        if len(args.fleet_import) < 2:
            raise SystemExit('--fleet-import needs a vehicle name and at least one file.')
        fdoc = FL.import_files(project, args.fleet_import[0], args.fleet_import[1:])
        _out(fdoc, '-')
        for e in fdoc['errors']:
            print(f'  could not read {e["file"]}: {e["error"]}', file=sys.stderr)
        if fdoc['errors'] or not fdoc['imported']:
            code = max(code, 4)
    if args.fleet is not None:
        from .core import fleet as FL
        fr = FL.run(project)
        if args.fleet == '-' or args.fleet.lower().endswith('.json'):
            _out(FL.to_doc(fr, project), args.fleet)
        else:
            for path in _stamped(FL.export(fr, project, args.fleet), '--fleet', args.fleet):
                print(path)
        for f in fr.findings:
            if f['severity'] != 'good':
                print(f'  {f["severity"]:<8} {f["message"]}', file=sys.stderr)
        if fr.flagged('re-mount'):
            code = max(code, 3)
    if args.ci_templates:
        from .core import automation as AU
        try:
            for pth in AU.write_ci(project, args.ci_templates):
                print(pth)
        except ValueError as e:
            raise SystemExit(str(e))
    if args.experiment:
        code = max(code, _experiment(project, eng, args))
    if args.carla_rig:
        code = max(code, _carla_rig(project, args))
    if args.carla_run:
        code = max(code, _carla_run(project, args))
    if args.validate is not None:
        code = max(code, _validate(args, project, eng))
    if args.save:
        project.save(args.save)
        print(f'saved {args.save}', file=sys.stderr)
    return code


def _experiment(project, eng, args) -> int:
    """--experiment SPEC OUT: a designed experiment (core.experiments) into OUT; it resumes from OUT's journal."""
    from .core import experiments as EXP
    spec, dest = args.experiment
    last = [-1.0]

    def progress(f, text=''):
        if f - last[0] >= 0.02 or f >= 1.0:
            last[0] = f
            print(f'\r  experiment {f * 100:5.1f} % {text:<48}', end='', file=sys.stderr)
    edoc, ecode = EXP.batch(project, spec, dest, eng, variant=project.variant if args.variant else None,
                            progress=progress)
    print(file=sys.stderr)
    if ecode:
        print(f'experiment: {edoc.get("error")}', file=sys.stderr)
        return ecode
    for pth in _stamped([os.path.join(dest, n) for n in edoc.get('files', [])], '--experiment', dest):
        print(pth)
    best = edoc.get('best') or {}
    print(f'experiment: {edoc["design_label"]}, {edoc["points"]} points ({edoc["cached"]} resumed) in '
          f'{edoc["seconds"]} s' + (f'; best point {best["point"]}: {best["design"]}' if best else ''), file=sys.stderr)
    for r in edoc.get('responses', []):
        top = ', '.join(f'{d["factor"]} ({d["score"]})' for d in r.get('drivers', [])[:3])
        print(f'  {r["label"]}: {r["evaluated"]} points' + (f'; drivers {top}' if top else ''), file=sys.stderr)
    for w in edoc.get('warnings', []):
        print(f'  note: {w}', file=sys.stderr)
    return 0


def _carla_rig(project, args) -> int:
    """--carla-rig OUT_DIR: the variant as the CARLA bridge reads it (sim_rig.json, sim_remap.npz); 2 when it has
    nothing to simulate."""
    from .core import simrig as SR
    try:
        rig = SR.build(project)
    except ValueError as e:
        print(f'carla rig: {e}', file=sys.stderr)
        return 2
    for pth in _stamped(SR.export(rig, project, args.carla_rig), '--carla-rig', args.carla_rig):
        print(pth)
    s = SR.summary(rig)
    print(f'carla rig: {len(s["cameras"])} cameras, {len(s["lidars"])} lidars, {len(s["radars"])} radars on '
          f'{s["vehicle"]} at {s["tick_hz"]:g} Hz; {s["weather"]}', file=sys.stderr)
    for n in s['notes']:
        print(f'  note: {n}', file=sys.stderr)
    return 0


def _carla_python(args) -> str:
    """CARLA's client Python: --carla-python, else the one Simulate ▸ CARLA chose on this computer, else the venv
    next to a CARLA folder found in the usual places."""
    if args.carla_python:
        return os.path.abspath(os.path.expanduser(args.carla_python))
    try:
        from PyQt6.QtCore import QSettings
        saved = str(QSettings('Vantage', 'Vantage').value('sim/python', '') or '')
    except Exception:                       # noqa: BLE001 — no settings: look for it
        saved = ''
    if saved and os.path.isfile(saved):
        return saved
    from .sim import server as SV
    return next((py for py in (SV.client_python(r) for r in SV.candidate_roots()) if py), '')


def _carla_run(project, args) -> int:
    """--carla-run OUT_DIR: the variant driven in the CARLA server already running on the project's port (Simulate ▸
    CARLA's settings), recorded into OUT_DIR as a nuScenes folder and a ROS 2 bag by the bridge in CARLA's own Python;
    2 when there is nothing to simulate, 7 when no server answers, the client Python is missing or the run fails."""
    import subprocess
    from .core import simrig as SR
    from .sim import server as SV
    st = SR.settings_of(project)
    out = os.path.abspath(args.carla_run)
    try:
        rig = SR.build(project, st)
    except ValueError as e:
        print(f'carla run: {e}', file=sys.stderr)
        return 2
    py = _carla_python(args)
    if not py or not os.path.isfile(py):
        print('carla run: no Python for CARLA\'s client: give --carla-python (a Python with the carla package, numpy, '
              'OpenCV and PyYAML)', file=sys.stderr)
        return 7
    if not SV.port_open('127.0.0.1', st.port):
        print(f'carla run: no CARLA server answers on port {st.port}: start one (Simulate ▸ CARLA ▸ Start, or '
              f'CarlaUE4.sh -RenderOffScreen -carla-rpc-port={st.port})', file=sys.stderr)
        return 7
    rig_dir = os.path.join(out, 'rig')
    SR.export(rig, project, rig_dir)
    stale = os.path.join(out, 'run.json')
    if os.path.isfile(stale):
        os.remove(stale)                    # an earlier run's report must not stand for this one
    try:
        cmd, env = SV.bridge_command(py, os.path.join(rig_dir, 'sim_rig.json'), out,
                                     ['--port', str(st.port), '--tm-port', str(st.tm_port),
                                      '--timeout', f'{st.timeout_s:g}'])
    except RuntimeError as e:
        print(f'carla run: {e}', file=sys.stderr)
        return 7
    log_path = os.path.join(out, 'bridge.log')
    print(f'carla run: {st.duration_s:g} s in {st.town}, {st.traffic} vehicles, seed {st.seed} → {out}',
          file=sys.stderr)
    with open(log_path, 'w', encoding='utf-8') as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=log, text=True, env=env, cwd=out)
        try:
            for line in proc.stdout:
                pr = SV.parse_progress(line)
                if pr is not None:
                    print(f'\r  carla {pr[0] * 100:5.1f} % {pr[1]:<40}', end='', file=sys.stderr)
                else:
                    log.write(line)
            proc.wait()
        except KeyboardInterrupt:           # the bridge got the same Ctrl+C: it tidies CARLA up
            proc.wait()
            print(file=sys.stderr)
            return 130
    print(file=sys.stderr)
    doc = SR.read_run(out)
    if doc is None:
        print(f'carla run: the bridge exited with {proc.returncode} and wrote no report: see {log_path}',
              file=sys.stderr)
        return 7
    _stamped([p_ for p_ in (os.path.join(out, 'run.json'), os.path.join(out, 'dataset.json')) if os.path.isfile(p_)],
             '--carla-run', out)
    if doc.get('status') != 'ok':
        print(f'carla run: {doc.get("status")}: ' + str(doc.get('error') or 'see run.json').splitlines()[0],
              file=sys.stderr)
        return 7
    perf, sc = doc.get('performance') or {}, doc.get('scene') or {}
    print(f'carla run: {sc.get("samples", "?")} samples, {perf.get("steps", "?")} steps in {perf.get("wall_s", "?")} s '
          f'written to {out} (nuscenes/, rosbag/, run.json)', file=sys.stderr)
    for n in doc.get('notes') or []:
        print(f'  note: {n}', file=sys.stderr)
    return 0


def _validate(args, project, eng=None) -> int:
    """--validate [OUT]: every model checked against an independent reference (vantage.validation), the report into
    OUT (default ./validation); 6 when a check fails. With a project its vehicle and exports are checked too."""
    from . import validation as VA
    st = VA.ValidationSettings.from_dict({'models': args.validate_models or (), 'quick': bool(args.validate_quick)})
    if args.validate_models:
        unknown = [m for m in args.validate_models.split(',') if m.strip() and m.strip() not in VA.MODEL_IDS]
        if unknown:
            raise SystemExit(f'--validate-models: unknown model {", ".join(unknown)}; known: '
                             + ', '.join(VA.MODEL_IDS))
    sel = VA.specs(st)
    print(f'validation: {len(sel)} checks' + (' (quick)' if st.quick else ''), file=sys.stderr)
    res = VA.run(project, eng, settings=st, variant=project.variant if (project is not None and args.variant) else None,
                 progress=lambda f: print(f'\r  validation {f * 100:5.1f} %', end='', file=sys.stderr))
    print(file=sys.stderr)
    for pth in _stamped(VA.export(res, project, args.validate), '--validate', args.validate):
        if pth.endswith(('validation.html', 'validation.json')):
            print(pth)
    for c in res.checks:
        err = f'{c.error:.3g}' if c.error == c.error else 'n/a'
        print(f'  {"pass" if c.passed else "FAIL"}  {c.id:<36} {err} / {c.tolerance:g} {c.unit}'
              + (f'  ({c.exception})' if c.exception else ''), file=sys.stderr)
    print(f'validation: {res.passed}/{len(res.checks)} checks passed in {res.seconds:.0f} s; report '
          f'{os.path.join(args.validate, "validation.html")}', file=sys.stderr)
    return VA.exit_code(res)


def rig_health_doc(project, eng, spec) -> dict:
    """--rig-health SECONDS [OUT.json]: watch every camera's stream (project.streams) and report its health."""
    from .core import rig_health as rh
    try:
        seconds = float(spec[0])
    except ValueError:
        raise SystemExit(f'--rig-health needs the seconds to measure first, not {spec[0]!r}.')
    if len(spec) > 2 or (len(spec) == 2 and spec[1].lower().endswith('.vantage')):
        raise SystemExit('--rig-health takes SECONDS and at most one OUT.json (put the project before it).')
    cams = [c for c in project.variant.cameras if c.visible]
    print(f'rig health: watching {sum(1 for c in cams if project.streams.get(c.tag))} of {len(cams)} cameras for '
          f'{seconds:g} s', file=sys.stderr)
    doc = rh.jsonable(rh.run_for(project, cams, seconds, engine_factory=eng.share))
    for k, c in doc['cameras'].items():
        print(f'  {k:<4} {c["status"]:<8} ' + ('; '.join(c['reasons']) or f'{c.get("fps", 0):.1f} fps'), file=sys.stderr)
    return doc


# ---------------------------------------------------------------------------------------------------- --reproduce
# options that write a result: a re-run keeps only the one that wrote the result being checked
_PRODUCERS = ('analyse', 'load_cases', 'blind_volume', 'perception_range', 'sensor_coverage', 'sync_plan',
              'recording_budget', 'power_budget', 'gnss', 'exposure', 'fleet', 'brackets', 'dataset', 'drive_log',
              'housings', 'suspension', 'as_built_check', 'check', 'lidar_calib', 'observability', 'harness', 'terrain',
              'critical_targets', 'materials', 'weather', 'export', 'install_check', 'rig_health', 'ci_templates',
              'experiment', 'validate', 'carla_rig')
# what a re-run never does: save a project, write files beside the result, watch a project, run a command
_NEVER = ('report', 'lut_out', 'as_built', 'calib_out', 'save', 'watch', 'on_change', 'reproduce', 'tolerance',
          'carla_run', 'carla_python')
# producers whose input comes first and their output second
_IN_OUT = ('drive_log', 'as_built_check', 'experiment')


class _CannotReproduce(Exception):
    pass


def _find(name, search, sha=None):
    """An input or project recorded by its file name: as given, or next to the result, in this folder or the
    project's. With the recorded SHA-256, the candidate whose content matches it comes first (an output folder can
    hold a file of the input's name, such as the experiment spec an --experiment run writes beside its results)."""
    from .core import provenance as PV
    cands = ([os.path.abspath(name)] if name and os.path.exists(name) else []) + \
        [os.path.join(d, os.path.basename(str(name))) for d in search]
    cands = [c for i, c in enumerate(cands) if os.path.exists(c) and c not in cands[:i]]
    if sha and len(cands) > 1:
        same = next((c for c in cands if PV.path_digest(c).get('sha256') == sha), None)
        if same is not None:
            return same
    return cands[0] if cands else None


def _reproduction(prov, args, tmp, search):
    """(Namespace of the re-run, the file it writes the result to (None: standard output), its command) from a
    result's provenance block."""
    from .core import provenance as PV
    out = dict(prov.get('output') or {})
    cmd = prov.get('command')
    if not cmd:                                     # an export made in the app: the batch exporter makes it again
        st = (prov.get('settings') or {}).get('export') or {}
        if not st.get('format') or not out.get('name'):
            raise _CannotReproduce('the result names no batch command and no export format')
        var = prov.get('variant') or {}
        proj = (prov.get('project') or {}).get('file')
        cmd = ['vantage'] + ([proj] if proj else []) + \
            (['--variant', str(var['index'])] if var.get('index') else []) + \
            ['--export', 'out', '--formats', st['format'], '--frame', str(st.get('frame') or 'rear_axle'),
             '--units', str(st.get('units') or 'm'), '--angles', str(st.get('angles') or 'deg')]
        out.update(option='--export', in_folder=True)
        if not proj and not args.path:
            raise _CannotReproduce(f'the project was not saved when {out["name"]} was exported: give it, '
                                   f'vantage PROJECT.vantage --reproduce …')
    opt = (out.get('option') or '').lstrip('-').replace('-', '_')
    if opt == 'rig_health':
        raise _CannotReproduce('the rig health check measures live streams: it cannot be run again')
    if opt == 'reproduce':
        raise _CannotReproduce('this is a reproduction report: reproduce the result it checked instead')
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            ns = parser().parse_args([str(x) for x in cmd[1:]])
    except SystemExit:
        raise _CannotReproduce('the recorded command cannot be read by this version: '
                               + ' '.join(str(x) for x in cmd)) from None
    if not opt or not hasattr(ns, opt) or getattr(ns, opt) in (None, False):
        raise _CannotReproduce('the result does not say which option wrote it')
    # the project: the one given, else the recorded one found by its name (or the sample or template)
    if prov.get('project') is None:
        pass
    elif args.path:
        ns.path, ns.sample, ns.template = args.path, False, None
    elif ns.path:
        found = _find(ns.path, search)
        if found is None:
            raise _CannotReproduce(f'the project {ns.path} is not next to the result nor in this folder: give it, '
                                   f'vantage PROJECT.vantage --reproduce …')
        ns.path = found
    # only the option that wrote the result; nothing saved, watched or run
    for d in _PRODUCERS:
        if d != opt:
            setattr(ns, d, None)
    for d in _NEVER:
        setattr(ns, d, None)
    if opt == 'export':
        fmt = ((prov.get('settings') or {}).get('export') or {}).get('format') or out.get('format')
        if fmt:
            ns.formats = fmt
    # input files by their recorded names
    missing = []
    for dest, how in PV.INPUT_ARGS.items():
        v = getattr(ns, dest, None)
        if dest in ('path', 'watch') or not v:
            continue
        vals = list(v) if isinstance(v, (list, tuple)) else [v]
        idx = [0] if how == 'first' else range(1, len(vals)) if how == 'rest' else range(len(vals))
        opt_ = '--' + dest.replace('_', '-')
        for i in idx:
            sha = next((x.get('sha256') for x in prov.get('inputs') or []
                        if x.get('option') == opt_ and x.get('name') == os.path.basename(str(vals[i]))), None)
            p = _find(vals[i], search, sha)
            if p is None:
                missing.append(str(vals[i]))
            else:
                vals[i] = p
        setattr(ns, dest, vals if isinstance(v, (list, tuple)) else vals[0])
    if missing:
        raise _CannotReproduce('input not found next to the result nor in this folder: ' + ', '.join(missing))
    # where the re-run writes the result
    name = str(out.get('name') or '-')
    if out.get('in_folder'):
        value = os.path.join(tmp, 'out')
        target = os.path.join(value, *name.split('/'))
    else:
        value = target = os.path.join(tmp, name if name.lower().endswith('.json') else 'result.json')
    if opt in _IN_OUT:
        setattr(ns, opt, [getattr(ns, opt)[0], value])
    elif opt in ('install_check', 'calibrate_lens', 'import_calib'):
        ns.report = value
    elif opt == 'fleet_import':
        target = None                               # it writes to the standard output only
    else:
        setattr(ns, opt, value)
    return ns, target, PV.command_of(ns, parser())


def _doc_in(text, schema):
    """The JSON object with this schema printed in text (an output that goes to the standard output only)."""
    dec = json.JSONDecoder()
    for m in re.finditer(r'(?m)^\{', text):
        try:
            doc, _ = dec.raw_decode(text, m.start())
        except ValueError:
            continue
        if isinstance(doc, dict) and doc.get('schema') == schema:
            return doc
    return None


def reproduce(args) -> int:
    """--reproduce RESULT.json [OUT.json]: run again what wrote a result, compare every number (core.provenance) and
    write the report; 0 reproduces, 6 does not, 4 cannot be tried."""
    import tempfile
    from .core import provenance as PV
    spec = args.reproduce
    if len(spec) > 2 or (len(spec) == 2 and spec[1].lower().endswith('.vantage')):
        raise SystemExit('--reproduce takes RESULT.json and at most one OUT.json (put the project before it).')
    src, dest = spec[0], (spec[1] if len(spec) > 1 else '-')
    try:
        tol = PV.parse_tolerances(args.tolerance or [])
    except ValueError as e:
        raise SystemExit(f'--tolerance {e}')
    name = os.path.basename(src)
    try:
        with open(src, encoding='utf-8') as fh:
            rec = json.load(fh)
    except (OSError, ValueError) as e:
        print(f'reproduce: cannot read {name}: {e}', file=sys.stderr)
        return 4
    prov = rec.get(PV.KEY) if isinstance(rec, dict) else None
    if not isinstance(prov, dict) or prov.get('schema') != PV.SCHEMA:
        print(f'reproduce: {name} has no provenance block: it was written by an earlier Vantage or another program',
              file=sys.stderr)
        return 4
    search = [os.path.dirname(os.path.abspath(src)), os.getcwd()] + \
        ([os.path.dirname(os.path.abspath(args.path))] if args.path else [])
    var = prov.get('variant') or {}
    print(f'reproduce {name}: {prov.get("study") or "result"} of "{var.get("name")}", written '
          f'{prov.get("created_utc")} by Vantage {(prov.get("vantage") or {}).get("version")}', file=sys.stderr)
    new, given, error, cmd = None, None, None, None
    with tempfile.TemporaryDirectory(prefix='vantage-reproduce-') as tmp:
        try:
            ns, target, cmd = _reproduction(prov, args, tmp, search)
        except _CannotReproduce as e:
            error = str(e)
        if error is None and prov.get('project') is not None:      # (a result of no project, e.g. a validation)
            try:
                project = load_project(ns)
            except (SystemExit, OSError, ValueError) as e:
                error = f'the project cannot be read: {e}'
            else:
                PV.note_project(project)            # this report's own provenance names the project checked
                given = PV.project_digest(project)
        if error is None:
            print('  re-run: ' + ' '.join(cmd), file=sys.stderr)
            buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf):
                    run(ns)
            except SystemExit as e:
                error = f'the re-run stopped: {e}'
            if buf.getvalue().strip():
                print(buf.getvalue().rstrip(), file=sys.stderr)
            if error is None:
                try:
                    if target and os.path.isfile(target):
                        with open(target, encoding='utf-8') as fh:
                            new = json.load(fh)
                    else:
                        new = _doc_in(buf.getvalue(), rec.get('schema'))
                except (OSError, ValueError) as e:
                    error = f'the re-run\'s result cannot be read: {e}'
                if new is None and error is None:
                    error = 'the re-run wrote no result'
    rep = PV.report(name, rec, new, tol, now_provenance=(new or {}).get(PV.KEY) if isinstance(new, dict) else None,
                    project_now=given, error=error, command=cmd)
    _out(rep, dest)
    c = rep.get('counts')
    if c:
        print(f'  {c["numbers"]} numbers: {c["exact"]} exact, {c["within"]} within tolerance, {c["differs"]} differ; '
              f'{c["texts_changed"]} texts changed, {c["missing"]} missing, {c["extra"]} new', file=sys.stderr)
    print(f'  {rep["message"]}', file=sys.stderr)
    ok = rep['verdict'] in ('reproduces', 'reproduces_within_tolerance')
    for r in rep.get('reasons') or []:
        print(f'  {"note" if ok else "why"}: {r}', file=sys.stderr)
    if rep['verdict'] == 'cannot_reproduce':
        return 4
    return 0 if ok else PV.EXIT_NOT_REPRODUCED


def add_arguments(ap):
    g = ap.add_argument_group('batch mode (no window)')
    g.add_argument('--analyse', nargs='?', const='-', metavar='OUT.json', help='write the analysis summary as JSON (- = stdout)')
    g.add_argument('--load-cases', nargs='?', const='-', metavar='OUT.json', help='evaluate the load cases')
    g.add_argument('--blind-volume', nargs='?', const='-', metavar='OUT.json',
                   help='3D blind volume: space up to the study height no camera sees, undetectable objects, reach rose')
    g.add_argument('--perception-range', nargs='?', const='-', metavar='OUT.json',
                   help='how far out the rig resolves each object class per direction, sector verdicts, stereo pairs')
    g.add_argument('--sensor-coverage', nargs='?', const='-', metavar='OUT.json',
                   help='lidar and radar coverage: what the vehicle blocks, detection range per object and sector '
                        'against the cameras\', camera-lidar overlap')
    g.add_argument('--sync-plan', nargs='?', const='-', metavar='OUT.json',
                   help='camera trigger offsets against the lidar sweep, trigger lines, motion blur and rolling shutter')
    g.add_argument('--recording-budget', nargs='?', const='-', metavar='OUT.json',
                   help='data rate of every camera, lidar and radar, links, write speed and hours per disk set')
    g.add_argument('--power-budget', nargs='?', const='-', metavar='OUT.json',
                   help='power of every device, the DC-DC converter, fuses and power cables, the alternator margin '
                        'and the engine-off runtime')
    g.add_argument('--gnss', nargs='?', const='-', metavar='OUT.json',
                   help='GNSS antennas\' sky view, the heading baseline, the INS lever arms and the pose error budget')
    g.add_argument('--exposure', nargs='?', const='-', metavar='OUT.json',
                   help='sun glare and dirt exposure of the active variant (JSON; with a file, OUT.png next to it)')
    g.add_argument('--fleet', nargs='?', const='-', metavar='OUT.json|DIR',
                   help='the fleet against the plan: latest offsets, drift and vehicles that differ from the rest '
                        '(a folder gets fleet.json, fleet.csv and fleet.png)')
    g.add_argument('--fleet-import', nargs='+', metavar=('VEHICLE', 'FILE'),
                   help='add records to the fleet vehicle VEHICLE (added if new) from installation check reports, '
                        'extrinsics JSON/YAML/CSV, nuScenes tables or Kalibr camchains; keep them with --save')
    g.add_argument('--brackets', metavar='DIR', help='generate a 3D-printable mount bracket per camera into '
                   'DIR/brackets (STL per camera, brackets.json, brackets_sheet.png)')
    g.add_argument('--dataset', metavar='OUT_DIR',
                   help='write a synthetic drive recorded by the rig as a nuScenes version folder and a ROS 2 bag '
                        '(the project\'s sample-dataset settings; about 1.5 min at the defaults)')
    g.add_argument('--drive-log', nargs='+', metavar=('LOG', 'OUT'),
                   help='check a recorded drive (ROS 2 bag folder/.mcap/.db3 or nuScenes folder) against the plan: '
                        'OUT.json, or a folder for the JSON, the CSV, the timeline and overlay PNGs (default: stdout)')
    g.add_argument('--housings', nargs='?', const='-', metavar='OUT.json',
                   help='what the sensor housings and the other cameras\' modules hide in each camera\'s image, and '
                        'the ground coverage with and without them; exit 2 when a camera sits inside a housing')
    g.add_argument('--suspension', nargs='?', const='-', metavar='OUT.json',
                   help='load cases from the suspension and the loads (Analysis ▸ Load cases): corner drops, pitch and '
                        'roll, every camera\'s and sensor\'s move against the calibration load, and the ground figures; '
                        'exit 2 when a case lifts a wheel or exceeds the gross vehicle mass')
    g.add_argument('--suspension-data', metavar='FILE',
                   help='read suspension data (JSON, YAML, a CSV parameter sheet or ride-height table) into the project '
                        'first (kept with --save); exit 4 when it cannot be read')
    g.add_argument('--as-built-check', nargs='+', metavar=('REC', 'OUT'),
                   help='check a recording of the real rig (ROS 2 bag folder/.mcap/.db3, nuScenes folder or a folder '
                        'of images per camera) against the plan: each camera\'s outline of the vehicle, the pose '
                        'corrections and the coverage as built; OUT.json, or a folder for the JSON, the CSV and the '
                        'PNGs (default: stdout); with --save the as-built variant is kept; exit 3 when the plan\'s '
                        'coverage does not hold')
    g.add_argument('--check', nargs='?', const='-', metavar='OUT',
                   help='evaluate the project\'s requirements (Analysis ▸ Requirements scorecard); OUT.json gets the '
                        'scorecard, a folder the JSON, CSV and the design-review pack (PDF); exit 5 when a must '
                        'requirement fails or cannot be evaluated')
    g.add_argument('--lidar-calib', metavar='OUT_DIR',
                   help='calibrate a lidar to a camera from board captures (synthetic from the plan, or real pairs '
                        'with --lidar-calib-folder); writes the JSON and an overlay per pose')
    g.add_argument('--lidar-calib-pair', nargs=2, metavar=('CAMERA', 'LIDAR'),
                   help='the camera and lidar tags (default: the Lidar-camera calibration page\'s, else the best pair)')
    g.add_argument('--lidar-calib-folder', metavar='DIR',
                   help='image + point-cloud pairs matched by name (pose01.png + pose01.pcd / .bin / .ply / .npy / .csv)')
    g.add_argument('--observability', nargs='?', const='-', metavar='OUT.json|DIR',
                   help='calibration observability (Calibrate ▸ Calibration observability): each camera\'s 1σ from '
                        'the planned boards, the camera pairs, lidar–camera pairs, radars and the fusion density; a '
                        'folder gets the JSON, CSV and PNGs; exit 2 when a camera sees no board whole')
    g.add_argument('--verify', type=int, default=0, metavar='N',
                   help='with --observability: check the predictions with N Monte Carlo trials (about 4 s each)')
    g.add_argument('--harness', nargs='?', const='-', metavar='OUT.json',
                   help='every cable routed along the body: lengths against each link\'s limit, GNSS cable loss, '
                        'power-pair drop and the paths; exit 2 when a cable is longer than its link allows, no module '
                        'cable reaches or a GNSS cable has no RF margin')
    g.add_argument('--terrain', nargs='?', const='-', metavar='OUT.json',
                   help='the road and body motion study (Analysis ▸ Road and body motion): coverage on the road '
                        'against flat ground, how far the road ahead stays in view, when bumps and kerbs are first '
                        'seen, clearance driving over them, and brake dive, squat and cornering roll with each '
                        'camera\'s horizon shift; exit 2 when the body or a sensor would touch the road or a bump or '
                        'kerb is never resolved')
    g.add_argument('--critical-targets', nargs='?', const='-', metavar='OUT.json|DIR',
                   help='critical targets (Analysis ▸ Critical targets): which cameras resolve each and every camera\'s '
                        'mounting envelope (OUT.json or -: the JSON; a folder: the JSON, CSVs and PNGs); exit 2 when '
                        'no camera resolves a target')
    g.add_argument('--materials', nargs='?', const='-', metavar='OUT.json',
                   help='what each radar looks through (two-way loss, derated range, multipath risk), the cameras '
                        'behind glass and the wipers; exit 2 when a radar looks into metal over half its field')
    g.add_argument('--weather', nargs='?', const='-', metavar='OUT.json',
                   help='rain, fog, snow and night: every condition compared (range per object ahead, each camera, '
                        'lidar and radar against clear)')
    from .core.weather import IDS as WEATHER_IDS
    g.add_argument('--condition', choices=WEATHER_IDS,
                   help='the weather for --perception-range, --dataset and --weather (default: the project\'s)')
    g.add_argument('--export', metavar='DIR', help='export into DIR')
    g.add_argument('--formats', help='comma-separated export format ids (default: every non-image format but the '
                                     'experiment and the CARLA rig)')
    g.add_argument('--variant', help='variant name or 1-based index (default: the active one)')
    g.add_argument('--frame', help='export frame: rear_axle, front_axle, front_bumper, ground_centre, custom')
    g.add_argument('--units', default='m', choices=['m', 'mm'])
    g.add_argument('--angles', default='deg', choices=['deg', 'rad'])
    g.add_argument('--install-check', metavar='DIR', help='check the installation from one image per camera in DIR '
                                                          '(file names contain the camera tag)')
    g.add_argument('--report', metavar='OUT.json', help='installation check report (default: stdout)')
    g.add_argument('--lut-out', metavar='DIR', help='write bird\'s-eye remap tables for the measured poses into DIR')
    g.add_argument('--as-built', metavar='OUT.vantage', help='save the project with an as-built variant of measured poses')
    g.add_argument('--pos-tol', type=float, default=10.0, help='installation position tolerance, mm (default 10)')
    g.add_argument('--ang-tol', type=float, default=1.0, help='installation angle tolerance, degrees (default 1)')
    g.add_argument('--min-boards', type=int, default=1, help='boards a camera must see (default 1)')
    g.add_argument('--no-joint', action='store_true', help='solve each camera on its own (no joint solve with the boards)')
    g.add_argument('--board-tol', type=float, default=20.0, help='how far a board may lie off its planned place, mm (default 20)')
    g.add_argument('--no-balance', action='store_true', help='stitch without exposure balancing')
    g.add_argument('--blend', default='linear', choices=['linear', 'multiband'], help='seam blend of the stitched PNG')
    g.add_argument('--calibrate-lens', metavar='DIR', help='calibrate one camera\'s lens from the board views in DIR')
    g.add_argument('--camera', metavar='TAG', help='camera to calibrate')
    g.add_argument('--board', default='9x6', help='calibration board inner corners, COLSxROWS (default 9x6)')
    g.add_argument('--square', type=float, default=60.0, help='calibration board square, mm (default 60)')
    g.add_argument('--serial', help='serial number to store with the calibration')
    g.add_argument('--calib-out', metavar='FILE', help='also write the calibration (.yaml OpenCV, .json Vantage)')
    g.add_argument('--import-calib', nargs='+', metavar='FILE', help='apply unit calibrations from OpenCV/ROS/Kalibr/JSON files')
    g.add_argument('--save', metavar='OUT.vantage', help='save the project (with calibrations applied)')
    g.add_argument('--rig-health', nargs='+', metavar=('SECONDS', 'OUT.json'),
                   help='watch every camera\'s stream (set on the Live rig page) for SECONDS and report frame rate, '
                        'drops, exposure, sharpness and sync (default: stdout)')
    from .core import automation as AU
    AU.add_arguments(g)          # --watch, --on-change, --watch-runs, --watch-interval, --ci-templates
    g.add_argument('--experiment', nargs=2, metavar=('SPEC.json', 'OUT_DIR'),
                   help='run a designed experiment (a spec exported from Analysis ▸ Experiments) into OUT_DIR: the '
                        'points (CSV), the analysis (JSON) and the plots; resumes from OUT_DIR\'s journal after an '
                        'interruption; exit 4 when the spec cannot be read or does not fit the project')
    g.add_argument('--validate', nargs='?', const='validation', metavar='OUT_DIR',
                   help='check every model against independent references (closed forms, OpenCV, ITU-R tables, '
                        'published examples, exact ray casting, Monte Carlo) and write the report into OUT_DIR '
                        '(default ./validation); with a project its vehicle and exports are checked too; exit 6 when '
                        'a check fails')
    g.add_argument('--validate-models', metavar='IDS', help='with --validate: comma-separated model ids (default: all)')
    g.add_argument('--validate-quick', action='store_true',
                   help='with --validate: fewer Monte Carlo trials and samples (about 30 s), tolerances unchanged')
    g.add_argument('--carla-rig', metavar='OUT_DIR',
                   help='write the variant as the CARLA bridge reads it (sim_rig.json, sim_remap.npz) into OUT_DIR; '
                        'exit 2 when it has nothing to simulate')
    g.add_argument('--carla-run', metavar='OUT_DIR',
                   help='drive the variant in the CARLA server running on the port of Simulate ▸ CARLA\'s settings '
                        '(2000 unless set) and record a nuScenes folder and a ROS 2 bag into OUT_DIR; exit 7 when no '
                        'server answers or the run fails')
    g.add_argument('--carla-python', metavar='PYTHON',
                   help='with --carla-run: CARLA\'s client Python (default: the one Simulate ▸ CARLA chose, else the '
                        'venv next to a CARLA folder found)')
    g.add_argument('--reproduce', nargs='+', metavar=('RESULT.json', 'OUT.json'),
                   help='run again what wrote RESULT.json (a batch output or an exported JSON, from its provenance '
                        'block; give the same project first, input files next to the result or in this folder) and '
                        'report per number whether it reproduces and why not (OUT.json, default: stdout); exit 6 when '
                        'it does not reproduce, 4 when it cannot be tried')
    g.add_argument('--tolerance', nargs='+', metavar='SPEC',
                   help='with --reproduce: how far a number may differ, [PATTERN=]VALUE[%%]: VALUE in the quantity\'s '
                        'own unit, or %% of the recorded value; PATTERN matches the number\'s path or key '
                        '(blind_m2=0.005, \'metrics.*=1e-6\', \'*_pct=0.1%%\'); a bare VALUE for the rest '
                        '(default: exact)')


def wanted(args) -> bool:
    return any(getattr(args, k) not in (None, False) for k in ('analyse', 'load_cases', 'export', 'install_check',
                                                               'calibrate_lens', 'import_calib', 'blind_volume',
                                                               'perception_range', 'rig_health',
                                                               'recording_budget', 'sync_plan', 'sensor_coverage',
                                                               'power_budget', 'gnss', 'exposure', 'fleet',
                                                               'fleet_import', 'brackets', 'dataset',
                                                               'drive_log', 'as_built_check', 'check',
                                                               'suspension', 'suspension_data', 'housings',
                                                               'lidar_calib', 'observability', 'harness', 'terrain',
                                                               'critical_targets', 'materials',
                                                               'weather', 'watch', 'on_change', 'ci_templates',
                                                               'reproduce', 'experiment', 'validate', 'carla_rig',
                                                               'carla_run'))
