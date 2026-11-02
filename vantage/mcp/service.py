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

The assistant's 66 tools come along as they are (reading the design, studies, simulate, compare, look, the Copilot's
plan tools), except two that only make sense inside the app's own chat (update_plan, suggest_next). Then
edit_project changes anything the user can change in the design, one undo step per call; undo, save, export and
activity complete it."""
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
                                                                 'link or unlink mirrored cameras'},
    'target': {'type': 'string', 'enum': list(E.TARGETS)},
    'id': {'type': 'string', 'description': 'which one: a camera or sensor tag, a rig part, obstacle or target name, '
                                            'a variant name (set/remove/activate; for add variant: the one to copy)'},
    'props': {'type': 'object', 'description': 'the fields to set (get_project_data lists them); add sensor: kind '
                                               '(lidar|radar|gnss|imu), model_id and pose optional; add log: kind '
                                               '(decision|rejected|constraint|note), text, about (tags)'},
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
                 assistant=None):
        self.name, self.title, self.description = name, title, description
        self.params, self.fn, self.required = params, fn, tuple(required)
        self.read_only, self.destructive, self.assistant = read_only, destructive, assistant

    def schema(self) -> dict:
        s = {'type': 'object', 'properties': self.params}
        if self.required:
            s['required'] = list(self.required)
        return s

    def listing(self) -> dict:
        return {'name': self.name, 'title': self.title, 'description': self.description, 'inputSchema': self.schema(),
                'annotations': {'title': self.title, 'readOnlyHint': self.read_only,
                                'destructiveHint': self.destructive, 'openWorldHint': False}}


class NeedsWindow(Exception):
    pass


# =========================================================================== results
def text(doc, error=False) -> dict:
    return {'content': [{'type': 'text', 'text': doc if isinstance(doc, str) else T.dump(doc)}], 'isError': error}


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


def _edit(s, ops=None, **_):
    before = s.context()
    trial = before.project                     # a snapshot: the operations are tried on it first
    try:
        keys, lines = E.apply(trial, ops)
    except E.EditError as e:
        return {'error': str(e), 'changed': 'nothing'}
    label = E.title(lines)
    s.edit(keys, lambda p: E.apply(p, ops), label)
    after = s.context()
    old = set(_issue_lines(before))
    now = _issue_lines(after)
    return {'changed': lines, 'undo_step': label, 'before': _figures(before), 'after': _figures(after),
            'new_issues': [i for i in now if i not in old], 'resolved_issues': sorted(old - set(now)),
            'note': 'undo_last_change takes it back' + ('; not saved yet (save_project)' if s.unsaved() else
                                                        '; saved to the file')}


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


def _open(s, path=None, sample=False, template=None, **_):
    return s.open(path=path, sample=bool(sample), template=template)


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
         'or study settings, the project (name, frame, units, brief); add cameras, sensors (default pose by kind), '
         'rig parts (presets roof_rack, sensor_mast, enclosure, bar, or a custom box/cylinder/bar), obstacles, '
         'targets, variants (a copy of the active one by default), design-log entries; remove them; activate a '
         'variant; link or unlink mirrored cameras (a change to one then mirrors to the other). Returns the lines '
         'changed and the coverage figures and issues before and after. Locked cameras and sensors cannot change.',
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
    Tool('open_project', 'Open a project',
         'Work on another project. With the Vantage window open, Claude works on the project in it (open others there); '
         'without a window: path opens a .vantage file, sample the Tata Tigor EV sample, template a body '
         '(sedan, hatchback, suv, van, truck) — unsaved until save_project gets a path. A file opened here is worked on '
         'even while a window is open; with no arguments, Claude goes back to the window.',
         {'path': {'type': 'string'}, 'sample': {'type': 'boolean'}, 'template': {'type': 'string'}}, _open,
         read_only=False),
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


def call(session, name: str, args) -> dict:
    """Run tool name with args on the session: an MCP tools/call result ({'content': [...], 'isError': bool})."""
    t = BY_NAME.get(name)
    if t is None:
        return text({'error': f'unknown tool {name!r}'}, True)
    args = {k: v for k, v in (args or {}).items() if v is not None} if isinstance(args, dict) else {}
    try:
        if t.assistant is None:
            with _EDIT_LOCK if not t.read_only else _nolock():
                out = t.fn(session, **args)
            return text(out, isinstance(out, dict) and 'error' in out)
        problems = T.validate_args(name, args)
        if problems:
            return text({'error': 'the input does not match the tool: ' + '; '.join(problems[:6])}, True)
        if name in NEEDS_WINDOW and session.mode != 'live':
            return text({'error': f'{name} shows its result in the Vantage window, and none is open. Open the project in '
                                  'Vantage, or use edit_project to make the change on the file.'}, True)
        tc = session.context()
        tc.vision = True
        tc.cancel = threading.Event()
        if session.mode == 'live':
            tc.on_ui, tc.on_proposal, tc.on_copilot, tc.on_log = (session.show, session.proposal, session.plan,
                                                                  session.log)
        body, ok, pics = T.run_full(tc, name, args)
        content = [{'type': 'text', 'text': body}] + [_picture(p) for p in pics]
        return {'content': content, 'isError': not ok}
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
