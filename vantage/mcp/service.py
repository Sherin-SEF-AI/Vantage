"""The tools Claude gets over MCP, and running one on a session (the open Vantage window or a project file).

The same code runs in both places: in the app (vantage.ui.mcp_bridge, on a worker thread: everything that touches the
window goes through the session) and in the MCP process on a file (vantage.mcp.session.FileSession). A session gives:

    mode                 'live' | 'file'
    context()            a ToolContext of the project analysed as it is now (the assistant's tools read it)
    project()            the project itself (read only here)
    edit(keys, fn, text) fn(project) as one undoable step; the analysis follows
    undo() / redo()      Claude's own last step only; a reason when that is not possible
    save(path)           'saved …' or a reason
    activity(limit)      what happened lately: the undo history with who did what, the page, the selection
    show(req), proposal(prop), plan(plan), log(entry)   the window only (file sessions say so)

The assistant's tools come along as they are (reading the design, studies, simulate, compare, look, the Copilot's
plan tools), except two that only make sense inside the app's own chat (update_plan, suggest_next). Then
edit_project changes anything the user can change in the design, one undo step per call; undo, save, export and
activity complete it. import_vehicle and import_layout bring in a model and a layout; set_view, render_view and
turntable_video drive the window's 3D view; start_job, get_job, cancel_job and list_jobs (run by the MCP server)
run any tool in the background."""
from __future__ import annotations

import base64
import json
import os
import threading

from ..assistant import tools as T
from . import edits as E

PROTOCOL_NOTE = ('Vantage plans the cameras, lidars, radars, GNSS and IMUs of a vehicle data-collection rig. Units: '
                 'metres and degrees, in the ISO 8855 vehicle frame (x forward, y left, z up, origin at the rear-axle '
                 'centre on the ground); pitch is positive down. Start with get_overview, then read before you change '
                 'anything. edit_project changes the design as one undo step and returns what the analysis says '
                 'before and after; simulate_changes tries a change without making it. Locked cameras and sensors are '
                 'the user\'s: leave them. In the Vantage window every change shows at once and the user can undo it; '
                 'on a project file each change is saved with a backup of the file as it was first opened.')

SKIP = ('update_plan', 'suggest_next')
NEEDS_WINDOW = ('show_in_app', 'propose_changes', 'propose_requirements', 'add_design_log', 'propose_plan')

_OP_ITEM = {'type': 'object', 'properties': {
    'op': {'type': 'string', 'enum': list(E.OPS), 'description': 'set fields, add, remove, activate a variant, '
                                                                 'link or unlink mirrored cameras, aim a camera'},
    'target': {'type': 'string', 'enum': list(E.TARGETS)},
    'id': {'type': 'string', 'description': 'which one: a camera or sensor tag, a rig part, obstacle or target name, '
                                            'a variant name (set/remove/activate; for add variant: the one to copy), '
                                            'a seam\'s two camera tags as get_seams lists them ("QR|MR")'},
    'props': {'type': 'object', 'description': 'the fields to set (get_project_data lists them); set seam: angle_deg '
                                               '(degrees about the body centre) or auto: true (back to the automatic '
                                               'placement); add sensor: kind (lidar|radar|gnss|imu), model_id and pose '
                                               'optional; add log: kind (decision|rejected|constraint|note), text, '
                                               'about (tags); aim camera: at [x, y] (ground) or [x, y, z], or '
                                               'along: "normal" (the body\'s outward normal), and/or level_roll: true'},
    'preset': {'type': 'string', 'enum': ['roof_rack', 'sensor_mast', 'enclosure', 'bar'],
               'description': 'add rig_part: a preset sized from the roof and fitted with the parts there'},
    'variant': {'type': 'string', 'description': 'camera/sensor ops: which variant (default: the active one)'},
    'partner': {'type': 'string', 'description': 'link: the other camera\'s tag'},
    'mirror': {'type': 'boolean', 'description': 'set camera: mirror the change to its linked partner (default true)'},
    'copy': {'type': 'boolean', 'description': 'add variant: copy the active (or id) variant (default true)'},
    'activate': {'type': 'boolean', 'description': 'add variant: make it active (default true)'}},
    'required': ['op', 'target']}


class Tool:
    def __init__(self, name, title, description, params, fn=None, required=(), read_only=True, destructive=False,
                 assistant=None, router=False, idempotent=False, window=False):
        self.name, self.title, self.description = name, title, description
        self.params, self.fn, self.required = params, fn, tuple(required)
        self.read_only, self.destructive, self.assistant = read_only, destructive, assistant
        self.router = router                      # run by the MCP server itself (background jobs), not a session
        self.idempotent = idempotent or read_only
        self.window = window                      # needs the Vantage window open

    def schema(self) -> dict:
        s = {'type': 'object', 'properties': self.params}
        if self.required:
            s['required'] = list(self.required)
        return s

    def listing(self) -> dict:
        return {'name': self.name, 'title': self.title, 'description': self.description, 'inputSchema': self.schema(),
                'annotations': {'title': self.title, 'readOnlyHint': self.read_only,
                                'destructiveHint': self.destructive, 'idempotentHint': self.idempotent,
                                'openWorldHint': False}}


class NeedsWindow(Exception):
    pass


# =========================================================================== results
def text(doc, error=False) -> dict:
    """An MCP result. Whole JSON (the assistant's tools cut their own text for the in-app chat; these never are); a
    JSON object also goes out as structuredContent (with the same JSON as text, for clients that read only text)."""
    body = doc if isinstance(doc, str) else json.dumps(doc, ensure_ascii=False, default=str, separators=(',', ':'))
    out = {'content': [{'type': 'text', 'text': body}], 'isError': error}
    st = structured(body)
    if st is not None:
        out['structuredContent'] = st
    return out


def structured(body):
    """The JSON object in a tool's text, or None (prose, a list, or not JSON)."""
    if not isinstance(body, str) or not body.lstrip().startswith('{'):
        return None
    try:
        doc = json.loads(body)
    except ValueError:
        return None
    return doc if isinstance(doc, dict) else None


def _figures(tc) -> dict:
    res = tc.result
    if res is None:
        return {}
    m = res.metrics
    return {'blind_m2': round(m['blind_m2'], 4), 'covered_pct': round(m['covered_pct'], 2),
            'multi_pct': round(m['multi_pct'], 2), 'min_density_px_cm': round(m['min_density'], 2),
            'min_ok_pct': round(m['min_ok_pct'], 2), 'issues': len(tc.issues),
            'critical_issues': sum(1 for i in tc.issues if i.severity == 'critical')}


def _issue_lines(tc) -> list:
    return [f'{i.severity}: {i.message}' for i in tc.issues]


# =========================================================================== MCP-only tools
def _status(s, **_):
    p = s.project()
    return {'mode': s.mode, 'where': s.where(), 'project': p.name, 'variant': p.variant.name,
            'variants': [v.name for v in p.variants], 'cameras': [c.tag for c in p.variant.cameras],
            'sensors': [x.tag for x in p.variant.sensors], 'rig_parts': [r.name for r in p.rig],
            'unsaved_changes': s.unsaved(), 'note': PROTOCOL_NOTE}


def _data(s, sections=None, **_):
    return E.data(s.project(), sections)


def _moves_seams(ops) -> bool:
    """Operations that move the stitch seams: a seam itself, or a camera (the seams follow the cameras)."""
    return isinstance(ops, list) and any(isinstance(o, dict) and o.get('target') in ('seam', 'camera') for o in ops)


def _edit(s, ops=None, **_):
    before = s.context()
    seams = list(before.result.seams) if before.result is not None else None    # what seam operations name
    stitched = _moves_seams(ops)
    p0 = before.project.snapshot() if stitched else None       # as it is, before the trial changes the snapshot
    trial = before.project                     # a snapshot: the operations are tried on it first
    try:
        keys, lines = E.apply(trial, ops, seams)
    except E.EditError as e:
        return {'error': str(e), 'changed': 'nothing'}
    s0 = T.stitched_for(before, p0, before.result, 'now') if stitched else None
    label = E.title(lines)
    s.edit(keys, lambda p: E.apply(p, ops, seams), label)
    after = s.context()
    old = set(_issue_lines(before))
    now = _issue_lines(after)
    out = {'changed': lines, 'undo_step': label, 'before': _figures(before), 'after': _figures(after),
           'new_issues': [i for i in now if i not in old], 'resolved_issues': sorted(old - set(now))}
    if stitched:
        s1 = T.stitched_for(after, after.project, after.result, 'now')
        out['stitched_view'] = T.stitched_compare(p0, before.result, s0, after.project, after.result, s1)
    out['note'] = 'undo_last_change takes it back' + ('; not saved yet (save_project)' if s.unsaved() else
                                                      '; saved to the file')
    return out


def _undo(s, **_):
    return {'result': s.undo()}


def _redo(s, **_):
    return {'result': s.redo()}


def _save(s, path=None, **_):
    return {'result': s.save(path)}


def _activity(s, limit=20, **_):
    return s.activity(max(1, min(int(limit or 20), 100)))


def _export(s, formats=None, folder=None, **_):
    from ..core import exporters as X
    every = [{'id': f.id, 'group': f.group, 'label': f.label, 'default': f.default} for f in X.FORMATS]
    if not formats:
        return {'formats': every, 'note': 'name formats (ids) and a folder to write them'}
    ids = {f.id for f in X.FORMATS}
    bad = [f for f in formats if f not in ids]
    if bad:
        return {'error': f'unknown format(s) {", ".join(bad)}', 'formats': sorted(ids)}
    tc = s.context()
    p = tc.project
    if not folder:
        base = os.path.dirname(p.path) if p.path else (tc.working_dir or os.path.expanduser('~'))
        folder = os.path.join(base, f'{p.name}-export')
    folder = os.path.abspath(os.path.expanduser(folder))
    os.makedirs(folder, exist_ok=True)
    opts = X.ExportOptions(out_dir=folder, frame=p.frame, units=p.units, issues=tc.issues, engine=tc.engine,
                           worst_single=tc.worst)
    try:
        paths = X.export(list(formats), p, tc.result, opts)
    except ValueError as e:
        return {'error': str(e)}
    return {'folder': folder, 'files': [os.path.relpath(x, folder) for x in paths]}


def _batch(s, args=None, help=False, timeout_s=600, _cancel=None, **_):
    from . import batchrun as B
    if help or not args:
        return B.help_doc()
    why = B.check(args)
    if why:
        return {'error': why}
    tc = s.context()
    p = tc.project
    base = os.path.dirname(p.path) if p.path and os.access(os.path.dirname(p.path), os.W_OK) else \
        os.path.join(tc.working_dir or os.path.expanduser('~'), p.name)
    os.makedirs(base, exist_ok=True)
    return B.run(p, args, base, timeout_s, cancel=_cancel)


def _open(s, path=None, sample=False, template=None, **_):
    return s.open(path=path, sample=bool(sample), template=template)


def _import_vehicle(s, path=None, units=None, up=None, forward=None, flip_lr=False, origin='rear_axle',
                    ground='tyre', name=None, keep_cameras=True, **_):
    from ..core import loaders
    from ..core import vehicle_import as VI
    if not path:
        return {'error': 'path: the model file (.glb .gltf .obj .stl .ply .dae .fbx; .step with gmsh)'}
    try:
        veh, rep = VI.import_vehicle(path, units=units, up=up, forward=forward, flip_lr=bool(flip_lr),
                                     origin=origin or 'rear_axle', ground=ground or 'tyre', name=name)
    except (loaders.MeshLoadError, ValueError, OSError) as e:
        return {'error': str(e), 'changed': 'nothing'}
    before = s.context()
    keep = bool(keep_cameras)
    label = E.title([f'imported {rep["file"]} ({rep["triangles"]} tris, {rep["units"].split()[0]}, up '
                     f'{veh.import_up}, forward {veh.import_forward})'])

    def fn(p):
        p.vehicle = veh
        if not keep:
            p.variant.cameras = []
        if p.photos.landmarks:
            for k, c in veh.wheel_centres.items():
                p.photos.landmarks[f'Wheel centre {k}'] = list(c)
        p.add_log('import', f'{rep["file"]} · {veh.mesh.n_tris} tris · units {veh.import_units} · up {veh.import_up} '
                            f'· forward {veh.import_forward}')
    s.edit(['vehicle', 'variant', 'photos'], fn, label)
    after = s.context()
    if s.mode == 'live':
        s.show({'action': 'frame_all'})
    return {'imported': rep, 'undo_step': label, 'cameras_kept': keep, 'before': _figures(before),
            'after': _figures(after),
            'note': 'Check the report: a guessed unit, up or forward axis can be given explicitly. undo_last_change '
                    'takes it back.'}


def _import_layout(s, path=None, name=None, **_):
    from ..core import samples
    from ..core.project import Variant
    from ..ui.dialogs.misc import import_layout_file
    if not path:
        return {'error': 'path: a CSV (tag,name,x,y,z,yaw,pitch[,roll,lens,port]) or the extrinsics JSON export'}
    path = os.path.abspath(os.path.expanduser(path))
    p = s.project()
    try:
        cams = import_layout_file(path, p)
    except (OSError, ValueError, KeyError, TypeError) as e:
        return {'error': f'{os.path.basename(path)}: {e}', 'changed': 'nothing'}
    v = Variant(name=name or os.path.splitext(os.path.basename(path))[0])
    v.cameras = cams
    samples.link_pairs(v)
    label = E.title([f'imported layout {v.name!r} ({len(cams)} cameras) as a new variant'])

    def fn(pj):
        pj.variants.append(v)
        pj.active_uid = v.uid
    s.edit(['variants'], fn, label)
    after = s.context()
    return {'variant': v.name, 'cameras': [{'tag': c.tag, 'x': c.x, 'y': c.y, 'z': c.z, 'yaw': c.yaw,
                                             'pitch': c.pitch, 'roll': c.roll, 'lens_id': c.lens_id} for c in cams],
            'active': True, 'undo_step': label, 'after': _figures(after)}


VIEWS = ('persp', 'top', 'side', 'front', 'rear', 'top3d', 'bev', 'bowl', 'through')


def _window_only(s, what):
    if s.mode != 'live':
        raise NeedsWindow(f'{what} draws in the Vantage window, and none is open: open the project in Vantage '
                          '(look draws the analysis without a window)')


def _view_req(view=None, camera=None, orbit=None, overlays=None, heatmap=None, frame=None, page=None):
    req = {}
    if view is not None:
        if view not in VIEWS:
            raise ValueError(f'view: one of {", ".join(VIEWS)}')
        if view == 'through' and not camera:
            raise ValueError('view through: camera (its tag)')
        req['view'] = view
    if camera:
        req['camera'] = str(camera)
    if orbit is not None:
        if not isinstance(orbit, dict):
            raise ValueError('orbit: {az, el, distance} (degrees, degrees, metres)')
        o = {}
        for k, lo, hi in (('az', -720.0, 720.0), ('el', -89.0, 89.99), ('distance', 0.5, 500.0)):
            if k in orbit:
                v = float(orbit[k])
                if not lo <= v <= hi:
                    raise ValueError(f'orbit.{k}: {lo:g} to {hi:g}')
                o[k] = v
        req['orbit'] = o
    if overlays is not None:
        from ..ui.context import OVERLAYS
        keys = {k for k, *_ in OVERLAYS}
        if not isinstance(overlays, dict) or any(k not in keys for k in overlays):
            raise ValueError(f'overlays: {{key: true|false}} with keys {", ".join(sorted(keys))}')
        req['overlays'] = {k: bool(v) for k, v in overlays.items()}
    if heatmap is not None:
        from ..ui.context import HEATMAPS
        if heatmap not in {k for k, _ in HEATMAPS}:
            raise ValueError(f'heatmap: one of {", ".join(k for k, _ in HEATMAPS)}')
        req['heatmap'] = heatmap
    if frame is not None:
        if frame not in ('all', 'selection'):
            raise ValueError('frame: all | selection')
        req['frame'] = frame
    return req


def _set_view(s, view=None, camera=None, orbit=None, overlays=None, heatmap=None, frame=None, **_):
    _window_only(s, 'set_view')
    try:
        req = _view_req(view, camera, orbit, overlays, heatmap, frame)
    except (ValueError, TypeError) as e:
        return {'error': str(e)}
    if not req:
        return {'error': 'give view, camera, orbit, overlays, heatmap or frame'}
    return s.view(req)


def _render_view(s, path=None, scale=2, transparent=False, view=None, camera=None, orbit=None, overlays=None,
                 heatmap=None, inline=True, **_):
    _window_only(s, 'render_view')
    try:
        req = _view_req(view, camera, orbit, overlays, heatmap)
    except (ValueError, TypeError) as e:
        return {'error': str(e)}
    sc = int(scale or 2)
    if not 1 <= sc <= 4:
        return {'error': 'scale: 1 to 4 (times the view\'s size)'}
    req.update(scale=sc, transparent=bool(transparent), path=_out_path(s, path, 'render', '.png'))
    out = s.render(req)
    if inline and isinstance(out, dict) and out.get('path'):
        out['_image'] = out['path']
    return out


def _turntable(s, path=None, seconds=8, fps=30, scale=1, **_):
    _window_only(s, 'turntable_video')
    secs, f = float(seconds or 8), int(fps or 30)
    if not 1 <= secs <= 60 or not 5 <= f <= 60:
        return {'error': 'seconds: 1 to 60; fps: 5 to 60'}
    sc = float(scale or 1)
    if not 0.5 <= sc <= 2:
        return {'error': 'scale: 0.5 to 2'}
    return s.turntable({'path': _out_path(s, path, 'turntable', '.mp4'), 'frames': int(round(secs * f)), 'fps': f,
                        'scale': sc})


def _out_path(s, path, kind, ext):
    import time as _t
    if path:
        out = os.path.abspath(os.path.expanduser(path))
    else:
        p = s.project()
        base = os.path.dirname(p.path) if p.path and os.access(os.path.dirname(p.path), os.W_OK) else \
            os.path.join(os.path.expanduser('~'), 'Vantage')
        out = os.path.join(base, f'{p.name}-{kind}-{_t.strftime("%Y%m%d-%H%M%S")}{ext}')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    return out


MCP_TOOLS = [
    Tool('vantage_status', 'Vantage status',
         'Where Claude is working (the open Vantage window or a project file), the project, its variants, the active '
         'variant\'s cameras, sensors and rig parts, whether there are unsaved changes, and the conventions '
         '(units, frame, signs). Call it first.', {}, _status),
    Tool('get_project_data', 'Read the editable project data',
         'The fields edit_project can set, with their current values: cameras, sensors, rig, obstacles, targets, '
         'variants, roi, ecu, analysis, project, library (the module, lens and sensor-model ids to choose from).',
         {'sections': {'type': 'array', 'items': {'type': 'string', 'enum': ['cameras', 'sensors', 'rig', 'obstacles',
                                                                           'targets', 'variants', 'roi', 'ecu',
                                                                           'analysis', 'project', 'library']},
                       'description': 'default: cameras, sensors, rig, variants'}}, _data),
    Tool('edit_project', 'Change the design',
         'Change the design: a list of operations applied together as ONE undo step, checked first (a bad '
         'operation changes nothing and says why). Set any field of a camera (pose, lens, module, port, trigger line, '
         'cable, bracket, name, visible), sensor, rig part, obstacle, calibration target, variant, ROI, ECU, analysis '
         'or study settings, the project (name, frame, units, brief); set a stitch seam\'s angle by hand '
         '({"op": "set", "target": "seam", "id": "QR|MR", "props": {"angle_deg": -107}}) or return it to the '
         'automatic placement ({"auto": true}) as the Stitch seams page does; add cameras, sensors (default pose by '
         'kind), '
         'rig parts (presets roof_rack, sensor_mast, enclosure, bar, or a custom box/cylinder/bar), obstacles, '
         'targets, variants (a copy of the active one by default), design-log entries; remove them; activate a '
         'variant; link or unlink mirrored cameras (a change to one then mirrors to the other). Returns the lines '
         'changed and the coverage figures and issues before and after; for seam and camera operations also the '
         'stitched view before and after (per seam and in total, where the standing object vanishes). Locked cameras '
         'and sensors cannot change.',
         {'ops': {'type': 'array', 'items': _OP_ITEM, 'description': 'the operations, in order'}},
         _edit, required=('ops',), read_only=False, destructive=True),
    Tool('undo_last_change', 'Undo Claude\'s last change',
         'Undo the last change, when it was Claude\'s (edit_project). A change the user made after it is never undone '
         'from here.', {}, _undo, read_only=False),
    Tool('redo_change', 'Redo', 'Redo the change undo_last_change took back.', {}, _redo, read_only=False),
    Tool('save_project', 'Save the project',
         'Save the project. In the window: File ▸ Save (a project never saved asks for a path: give path). On a '
         'file: changes are saved as they are made; path saves a copy there and works on it from then on.',
         {'path': {'type': 'string', 'description': 'a .vantage path (optional)'}}, _save, read_only=False,
         destructive=True),
    Tool('get_recent_activity', 'What changed lately',
         'The latest steps of the undo history (who made each: the user or Claude), the page open in the window and '
         'the selection: what the user did since you last looked.',
         {'limit': {'type': 'integer', 'description': 'steps (default 20)'}}, _activity),
    Tool('export_files', 'Export calibration and plans',
         'Write the design in the formats the app exports (calibration files for nuScenes, KITTI, ROS camera_info, '
         'Kalibr; mounting sheets; BOM; the remap tables …) to a folder. Without formats: the list of format ids.',
         {'formats': {'type': 'array', 'items': {'type': 'string'}}, 'folder': {'type': 'string'}}, _export,
         read_only=False),
    Tool('run_batch', 'Run a Vantage job',
         'Run any of Vantage\'s batch jobs on the project as it is now (unsaved changes included; the project itself '
         'is not changed): studies (--perception-range, --sensor-coverage, --clock-tree, --thermo-vib, --isp, --gnss, '
         '--exposure, --terrain, --harness …), checks on files (--drive-log RECORDING, --as-built-check, '
         '--install-check IMAGES, --online-calib DRIVE, --lidar-calib, --log-replay, --detection-quality), '
         'calibration (--calibrate-lens, --import-calib, then --save OUT.vantage), --refine-rig, --experiment, '
         'perception and the fleet (--risk-coverage, --fleet-twin, --rig-perception, --robust-perception, --redrive, '
         '--foundation-range, --perf-model, --read-requirements, --design), '
         '--validate, --reproduce, --export with --formats, --rig-health. Runs in a folder of its own beside the '
         'project; returns the exit code and its meaning, the output\'s end, the files written and the small JSON ones. '
         'help: true lists every option with its arguments. CARLA runs, downloads and the automation command are not '
         'here: CARLA runs go through propose_plan for the user to approve.',
         {'args': {'type': 'array', 'items': {'type': 'string'},
                   'description': 'the command-line words after the project, e.g. ["--perception-range", '
                                  '"range.json"]; relative outputs land in the run folder'},
          'help': {'type': 'boolean', 'description': 'list the options instead of running'},
          'timeout_s': {'type': 'integer', 'description': 'stop after this many seconds (default 600, at most 3600)'}},
         _batch, read_only=False),
    Tool('open_project', 'Open a project',
         'Work on another project. With the Vantage window open, Claude works on the project in it (open others there); '
         'without a window: path opens a .vantage file, sample the Tata Tigor EV sample, template a body '
         '(sedan, hatchback, suv, van, truck) — unsaved until save_project gets a path. A file opened here is worked on '
         'even while a window is open; with no arguments, Claude goes back to the window.',
         {'path': {'type': 'string'}, 'sample': {'type': 'boolean'}, 'template': {'type': 'string'}}, _open,
         read_only=False),
    Tool('import_vehicle', 'Import a vehicle model',
         'Replace the vehicle with a 3D model file (.glb .gltf .obj .stl .ply .dae .fbx; .step with gmsh), as File ▸ '
         'Import ▸ Vehicle model does: units, up and forward axes from the file or guessed (give them when the '
         'report says a guess is wrong), turned into the vehicle frame, wheels found, ground on the tyres and the '
         'origin at the rear-axle centre, glass made a forbidden mount zone. One undo step; the cameras stay unless '
         'keep_cameras is false. Returns the import report and the coverage before and after.',
         {'path': {'type': 'string', 'description': 'the model file'},
          'units': {'type': 'string', 'enum': ['m', 'mm', 'cm', 'in']},
          'up': {'type': 'string', 'enum': ['+X', '-X', '+Y', '-Y', '+Z', '-Z']},
          'forward': {'type': 'string', 'enum': ['+X', '-X', '+Y', '-Y', '+Z', '-Z']},
          'flip_lr': {'type': 'boolean', 'description': 'mirror left and right (a model exported mirrored)'},
          'origin': {'type': 'string', 'enum': ['rear_axle', 'front_axle', 'front_bumper', 'centre', 'file']},
          'ground': {'type': 'string', 'enum': ['tyre', 'lowest', 'file']},
          'name': {'type': 'string'},
          'keep_cameras': {'type': 'boolean', 'description': 'keep the active variant\'s cameras (default true)'}},
         _import_vehicle, required=('path',), read_only=False, destructive=True),
    Tool('import_layout', 'Import a camera layout',
         'A camera layout from a CSV (tag,name,x,y,z,yaw,pitch[,roll,lens,port]; headers such as x_mm or yaw_rad '
         'give units) or the extrinsics JSON export, as a new variant made active (File ▸ Import ▸ Layout). One '
         'undo step.',
         {'path': {'type': 'string'}, 'name': {'type': 'string', 'description': 'the variant\'s name (default: the '
                                                                                 'file\'s)'}},
         _import_layout, required=('path',), read_only=False),
    Tool('set_view', 'Set the 3D view',
         'Set what the Vantage window shows (never changes the design): view (persp | top | side | front | rear | '
         'top3d | bev | bowl | through, with camera for through: the view through that camera\'s lens), orbit '
         '{az, el, distance} for the 3D view, overlays {key: true|false} (camlight, frustums, footprints, seams, roi, '
         'grid, sensors, lidarpts, brackets, harness, volume …), heatmap (none | coverage | density | redundancy | '
         'stitched), frame (all | selection). Needs the window.',
         {'view': {'type': 'string', 'enum': list(VIEWS)}, 'camera': {'type': 'string'},
          'orbit': {'type': 'object', 'properties': {'az': {'type': 'number'}, 'el': {'type': 'number'},
                                                     'distance': {'type': 'number'}}},
          'overlays': {'type': 'object', 'additionalProperties': {'type': 'boolean'}},
          'heatmap': {'type': 'string'}, 'frame': {'type': 'string', 'enum': ['all', 'selection']}},
         _set_view, window=True),
    Tool('render_view', 'Render the 3D view',
         'Render the window\'s 3D view as Render image does (the shader renderer, overlays crisp) at 1–4 times its '
         'size to a PNG, after an optional view, orbit, overlays and heatmap as in set_view; returns the file and the '
         'picture. Needs the window (look draws the analysis without one).',
         {'path': {'type': 'string', 'description': 'a .png path (default: next to the project)'},
          'scale': {'type': 'integer', 'minimum': 1, 'maximum': 4}, 'transparent': {'type': 'boolean'},
          'view': {'type': 'string', 'enum': list(VIEWS)}, 'camera': {'type': 'string'},
          'orbit': {'type': 'object'}, 'overlays': {'type': 'object'}, 'heatmap': {'type': 'string'},
          'inline': {'type': 'boolean', 'description': 'return the picture too (default true)'}},
         _render_view, read_only=False, window=True),
    Tool('turntable_video', 'Turntable video',
         'One orbit of the 3D view around its target as an MP4 (ffmpeg) or PNG frames, as View ▸ Turntable video '
         'does. Needs the window; long ones are best run with start_job.',
         {'path': {'type': 'string'}, 'seconds': {'type': 'number'}, 'fps': {'type': 'integer'},
          'scale': {'type': 'number'}},
         _turntable, read_only=False, window=True),
    Tool('start_job', 'Start a background job',
         'Run any other tool in the background and return at once with a job id: for long studies (run_batch, '
         'run_study, optimise_rig, design_rig, run_validation, calibrate_lidar_camera, turntable_video …). '
         'get_job reads its progress and result, cancel_job stops it. Jobs run one after another on the project as '
         'it is when each starts.',
         {'tool': {'type': 'string', 'description': 'the tool to run'},
          'arguments': {'type': 'object', 'description': 'its arguments'},
          'label': {'type': 'string'}},
         router=True, read_only=False, required=('tool',)),
    Tool('get_job', 'Read a background job',
         'A job\'s state (queued, running, done, failed, cancelled), how long it has run and, when done, its result. '
         'wait_s waits up to that long for it to finish first.',
         {'job_id': {'type': 'string'}, 'wait_s': {'type': 'number', 'minimum': 0, 'maximum': 300}},
         router=True, required=('job_id',)),
    Tool('cancel_job', 'Cancel a background job', 'Stop a queued or running job (a batch run is stopped at once; a '
         'study stops at its next check).', {'job_id': {'type': 'string'}}, router=True, read_only=False,
         required=('job_id',)),
    Tool('list_jobs', 'List background jobs', 'Every job of this session, newest first, with its state.', {},
         router=True),
]


def _assistant_tools():
    out = []
    for t in T.TOOLS:
        if t.name in SKIP:
            continue
        desc = t.description
        if t.name in NEEDS_WINDOW:
            desc += ' (Needs the Vantage window open: it shows there for the user.)'
        if t.name == 'propose_changes':
            desc += ' To make a change yourself, use edit_project.'
        if t.name == 'get_seams':
            desc += (' Set a seam with edit_project: {"op": "set", "target": "seam", "id": "QR|MR", "props": '
                     '{"angle_deg": -107}}; {"auto": true} returns it to the automatic placement '
                     '(seam_advice.edit_project_ops applies the advice).')
        out.append(Tool(t.name, t.label, desc, t.params, None, t.required, assistant=t))
    return out


TOOLS = MCP_TOOLS + _assistant_tools()
BY_NAME = {t.name: t for t in TOOLS}


def listing() -> list:
    return [t.listing() for t in TOOLS]


# =========================================================================== running one
_EDIT_LOCK = threading.Lock()


def _picture(pic) -> dict:
    return {'type': 'image', 'data': base64.b64encode(pic.data).decode('ascii'), 'mimeType': pic.media_type}


def _file_picture(path, max_side=1568):
    """An image file as MCP image content, scaled down to max_side pixels on its long side (what a model reads)."""
    try:
        from PyQt6.QtCore import QBuffer, QByteArray, QIODevice, Qt
        from PyQt6.QtGui import QImage
        img = QImage(path)
        if img.isNull():
            return None
        if max(img.width(), img.height()) > max_side:
            img = img.scaled(max_side, max_side, Qt.AspectRatioMode.KeepAspectRatio,
                             Qt.TransformationMode.SmoothTransformation)
        ba = QByteArray()
        buf = QBuffer(ba)
        buf.open(QIODevice.OpenModeFlag.WriteOnly)
        img.save(buf, 'PNG')
        return {'type': 'image', 'data': base64.b64encode(bytes(ba)).decode('ascii'), 'mimeType': 'image/png'}
    except Exception:                              # the picture is extra: the file is what was asked for
        return None


def call(session, name: str, args, cancel=None) -> dict:
    """Run tool name with args on the session: an MCP tools/call result ({'content': [...], 'isError': bool}).
    cancel: a threading.Event the caller sets to stop it (a batch run is killed; a study stops at its next check)."""
    t = BY_NAME.get(name)
    if t is None:
        return text({'error': f'unknown tool {name!r}'}, True)
    if t.router:
        return text({'error': f'{name} is run by the MCP server, not the window'}, True)
    args = {k: v for k, v in (args or {}).items() if v is not None} if isinstance(args, dict) else {}
    try:
        if t.assistant is None:
            with _EDIT_LOCK if not t.read_only else _nolock():
                out = t.fn(session, _cancel=cancel, **args)
            pic = out.pop('_image', None) if isinstance(out, dict) else None
            res = text(out, isinstance(out, dict) and 'error' in out)
            if pic:
                img = _file_picture(pic)
                if img is not None:
                    res['content'].append(img)
            return res
        problems = T.validate_args(name, args)
        if problems:
            return text({'error': 'the input does not match the tool: ' + '; '.join(problems[:6])}, True)
        if name in NEEDS_WINDOW and session.mode != 'live':
            return text({'error': f'{name} shows its result in the Vantage window, and none is open. Open the project in '
                                  'Vantage, or use edit_project to make the change on the file.'}, True)
        tc = session.context()
        tc.vision = True
        tc.cancel = cancel if cancel is not None else threading.Event()
        if session.mode == 'live':
            tc.on_ui, tc.on_proposal, tc.on_copilot, tc.on_log = (session.show, session.proposal, session.plan,
                                                                  session.log)
        body, ok, pics = T.run_full(tc, name, args)
        content = [{'type': 'text', 'text': body}] + [_picture(p) for p in pics]
        out = {'content': content, 'isError': not ok}
        st = structured(body)
        if st is not None:
            out['structuredContent'] = st
        return out
    except NeedsWindow as e:
        return text({'error': str(e)}, True)
    except Exception as e:                          # a tool never takes the server down
        return text({'error': f'{type(e).__name__}: {e}'}, True)


class _nolock:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def dumps(doc) -> str:
    return json.dumps(doc, ensure_ascii=False, default=str)
