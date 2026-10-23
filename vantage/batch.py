"""Headless batch mode: analyse, export, run load cases and check an installation from the command line.

    vantage project.vantage --analyse report.json
    vantage project.vantage --export out/ --formats extrinsics_json,opencv_yaml,bev_lut
    vantage project.vantage --install-check captures/ --as-built built.vantage
    vantage project.vantage --calibrate-lens views/ --camera F --board 9x6 --square 60 --save out.vantage
    vantage project.vantage --import-calib camchain.yaml F.yaml --save out.vantage
    vantage --sample --load-cases -
    vantage project.vantage --blind-volume volume.json
    vantage project.vantage --perception-range ranges.json --variant 2
    vantage project.vantage --rig-health 10 health.json
    vantage project.vantage --sensor-coverage coverage.json
    vantage project.vantage --sync-plan sync.json --units mm
    vantage project.vantage --recording-budget budget.json
    vantage project.vantage --power-budget power.json
    vantage project.vantage --gnss gnss.json
    vantage project.vantage --exposure exposure.json
    vantage project.vantage --brackets out/
    vantage project.vantage --dataset out/
    vantage project.vantage --drive-log drive.mcap report/
    vantage project.vantage --check review/
    vantage project.vantage --lidar-calib out/ --lidar-calib-pair F L1 --lidar-calib-folder boards/
    vantage project.vantage --fleet-import V3 install_report.json --fleet fleet/ --save project.vantage

The exit code is 0 when everything is fine, 2 when an analysis or export finds critical issues in the plan (or the
recording budget finds the recorder or a sensor link cannot keep up, or the power budget finds the DC-DC converter or a
fuse over its rating, the GNSS study finds an antenna with no sky, or a camera cannot take a printed bracket as planned),
3 when an installation check finds a camera that has to be re-mounted (or could not be solved), the rig health check
finds a camera faulty (or without a stream), the fleet has a sensor to re-mount in its latest record or a drive log
shows a fault or a lidar has to be re-mounted (or its pose is not determined by the board poses), and 4 when a lens
calibration fails, an imported calibration matches no camera, a fleet record, a drive log or a lidar calibration's captures cannot be read or the
sample dataset cannot be written, and 5 when, with --check, a must requirement fails or cannot be evaluated.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np


def _out(doc, dest):
    text = json.dumps(doc, indent=1, ensure_ascii=False, default=_jsonable)
    if not dest or dest == '-':
        print(text)
    else:
        os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
        with open(dest, 'w', encoding='utf-8') as fh:
            fh.write(text + '\n')
        print(f'wrote {dest}', file=sys.stderr)


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
        paths = export_lut(lut, lut_out, project)
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


def run(args) -> int:
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv[:1])
    from .ui.theme import load_fonts
    load_fonts()
    from .core import exporters as ex
    from .core.analysis import Engine
    from .core.install import InstallSettings
    project = load_project(args)
    pick_variant(project, args.variant)
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
    if args.export:
        if args.formats:
            ids = [f.strip() for f in args.formats.split(',') if f.strip()]
        else:
            ids = [f.id for f in ex.FORMATS if f.group != 'Images']
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
        paths = BR.export(bres, project, args.brackets)
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
                for pth in DLG.export(lres, project, dest):
                    print(f'wrote {pth}', file=sys.stderr)
            else:
                _out(DLG.to_doc(lres, project), dest)
            for f in lres.findings:
                if f['severity'] != 'ok':
                    print(f'  {f["severity"]:<8} {f["message"]}', file=sys.stderr)
            if any(f['severity'] == 'fault' for f in lres.findings):
                code = max(code, 3)
    if args.check is not None:
        from .core import requirements as RQ
        dest = None if args.check == '-' else args.check
        cdoc, ccode = RQ.check(project, eng, dest, variant=project.variant if args.variant else None)
        if dest is None:
            _out(cdoc, '-')
        elif not dest.lower().endswith('.json'):
            print(f'wrote {dest}/requirements.json, requirements.csv, review_pack.pdf', file=sys.stderr)
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
            for pth in LC.export(lres, project, args.lidar_calib):
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
            for path in FL.export(fr, project, args.fleet):
                print(path)
        for f in fr.findings:
            if f['severity'] != 'good':
                print(f'  {f["severity"]:<8} {f["message"]}', file=sys.stderr)
        if fr.flagged('re-mount'):
            code = max(code, 3)
    if args.save:
        project.save(args.save)
        print(f'saved {args.save}', file=sys.stderr)
    return code


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
    g.add_argument('--export', metavar='DIR', help='export into DIR')
    g.add_argument('--formats', help='comma-separated export format ids (default: every non-image format)')
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


def wanted(args) -> bool:
    return any(getattr(args, k) not in (None, False) for k in ('analyse', 'load_cases', 'export', 'install_check',
                                                               'calibrate_lens', 'import_calib', 'blind_volume',
                                                               'perception_range', 'rig_health',
                                                               'recording_budget', 'sync_plan', 'sensor_coverage',
                                                               'power_budget', 'gnss', 'exposure', 'fleet',
                                                               'fleet_import', 'brackets', 'dataset',
                                                               'drive_log', 'check', 'lidar_calib'))
