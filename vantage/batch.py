"""Headless batch mode: analyse, export, run load cases and check an installation from the command line.

    vantage project.vantage --analyse report.json
    vantage project.vantage --export out/ --formats extrinsics_json,opencv_yaml,bev_lut
    vantage project.vantage --install-check captures/ --as-built built.vantage
    vantage project.vantage --calibrate-lens views/ --camera F --board 9x6 --square 60 --save out.vantage
    vantage project.vantage --import-calib camchain.yaml F.yaml --save out.vantage
    vantage --sample --load-cases -
    vantage project.vantage --blind-volume volume.json

The exit code is 0 when everything is fine, 2 when an analysis or export finds critical issues in the plan,
3 when an installation check finds a camera that has to be re-mounted (or could not be solved) and 4 when a lens
calibration fails or an imported calibration matches no camera.
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
    if args.save:
        project.save(args.save)
        print(f'saved {args.save}', file=sys.stderr)
    return code


def add_arguments(ap):
    g = ap.add_argument_group('batch mode (no window)')
    g.add_argument('--analyse', nargs='?', const='-', metavar='OUT.json', help='write the analysis summary as JSON (- = stdout)')
    g.add_argument('--load-cases', nargs='?', const='-', metavar='OUT.json', help='evaluate the load cases')
    g.add_argument('--blind-volume', nargs='?', const='-', metavar='OUT.json',
                   help='3D blind volume: space up to the study height no camera sees, undetectable objects, reach rose')
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


def wanted(args) -> bool:
    return any(getattr(args, k) not in (None, False) for k in ('analyse', 'load_cases', 'export', 'install_check',
                                                               'calibrate_lens', 'import_calib', 'blind_volume'))
