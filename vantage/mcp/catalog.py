"""The MCP server's resources, prompts and argument completions.

Resources are the project as Claude reads it, addressable by URI, each read through the tool that computes it (so a
resource is always the project as it is now, in the window or the file):

    vantage://status            where Claude works, the project, its variants, cameras and sensors
    vantage://overview          the design: vehicle, cameras, sensors, rig, coverage
    vantage://project-data      every field edit_project can set, with its value
    vantage://coverage, vantage://issues, vantage://seams, vantage://requirements, vantage://design-log,
    vantage://activity, vantage://page, vantage://camera-library, vantage://validation-checks,
    vantage://export-formats, vantage://batch-options
    vantage://view/top.png      the bird's-eye coverage picture

and templates:

    vantage://camera/{tag}      one camera in detail
    vantage://study/{name}      a study's last result (get_<name>: sync_plan, power_budget, rig_perception …)
    vantage://view/{view}       a picture from the analysis: top, camera (with ?camera=TAG through look)
    vantage://docs/{name}       the documentation (README, overview, user-guide, command-line …) where it is installed
    vantage://job/{job_id}      a background job's state and result

The project resources change with the project: a client that subscribes is told (notifications/resources/updated)
when the project changes in the window or on the file.

Prompts are the common workflows, each a message that names the tools in the order they work best."""
from __future__ import annotations

import base64
import json
import os
import re

from . import service as SV

JSON = 'application/json'

STATIC = [
    ('vantage://status', 'status', 'Status', 'Where Claude works (the window or a file), the project, its variants, '
     'cameras, sensors and rig parts, unsaved changes and the conventions', 'vantage_status', {}),
    ('vantage://overview', 'overview', 'Design overview', 'The vehicle, every camera and sensor with its pose and '
     'lens, the rig, the ROI and the analysis thresholds (vantage://coverage has the figures)', 'get_overview', {}),
    ('vantage://project-data', 'project-data', 'Editable project data', 'Every field edit_project can set, with its '
     'current value, section by section', 'get_project_data',
     {'sections': ['cameras', 'sensors', 'rig', 'obstacles', 'targets', 'variants', 'roi', 'ecu', 'analysis',
                   'project']}),
    ('vantage://coverage', 'coverage', 'Ground coverage', 'Blind area, coverage, redundancy, density, per camera',
     'get_coverage', {}),
    ('vantage://issues', 'issues', 'Issues', 'The Issues list: severity, message, cameras', 'get_issues', {}),
    ('vantage://seams', 'seams', 'Stitch seams', 'The bird\'s-eye seams with their angles and advice', 'get_seams', {}),
    ('vantage://requirements', 'requirements', 'Requirements', 'Each requirement with its value, target, margin '
     'and verdict', 'get_requirements', {}),
    ('vantage://design-log', 'design-log', 'Design log', 'Decisions, rejected ideas and constraints',
     'get_design_log', {}),
    ('vantage://activity', 'activity', 'Recent activity', 'The latest undo steps (the user\'s and Claude\'s), the '
     'page open and the selection', 'get_recent_activity', {}),
    ('vantage://page', 'page', 'The page open', 'What the page the user has open is for and what it needs',
     'get_page_context', {}),
    ('vantage://camera-library', 'camera-library', 'Camera library', 'Every camera module and lens to choose from',
     'get_camera_library', {}),
    ('vantage://validation-checks', 'validation-checks', 'Validation checks', 'The validation suite\'s models and '
     'checks with their references and tolerances', 'list_validation_checks', {}),
    ('vantage://export-formats', 'export-formats', 'Export formats', 'Every format export_files writes',
     'export_files', {}),
    ('vantage://batch-options', 'batch-options', 'Batch options', 'Every batch option run_batch runs, with its '
     'arguments', 'run_batch', {'help': True}),
    ('vantage://view/top.png', 'view-top', 'Bird\'s-eye coverage picture', 'The coverage seen from above, drawn '
     'from the analysis', 'look', {'view': 'top'}),
]
STATIC_BY_URI = {r[0]: r for r in STATIC}
PROJECT_URIS = tuple(u for u, *_ in STATIC if u not in ('vantage://camera-library', 'vantage://validation-checks',
                                                        'vantage://export-formats', 'vantage://batch-options'))

TEMPLATES = [
    {'uriTemplate': 'vantage://camera/{tag}', 'name': 'camera', 'title': 'A camera',
     'description': 'One camera in detail: pose, lens, field of view, coverage, seams, issues', 'mimeType': JSON},
    {'uriTemplate': 'vantage://study/{name}', 'name': 'study', 'title': 'A study\'s result',
     'description': 'The last result of a study (sync_plan, power_budget, recording_budget, rig_perception, '
                    'risk_coverage, fleet_twin …: the get_<name> tools without arguments)', 'mimeType': JSON},
    {'uriTemplate': 'vantage://view/{view}', 'name': 'view', 'title': 'A picture from the analysis',
     'description': 'top (bird\'s-eye coverage) or camera-{TAG} (the view through a camera)', 'mimeType': 'image/png'},
    {'uriTemplate': 'vantage://docs/{name}', 'name': 'docs', 'title': 'Documentation',
     'description': 'Vantage\'s documentation pages, where installed (README, overview, user-guide, command-line, '
                    'features, project-format, simulator-bridge, install, development)', 'mimeType': 'text/markdown'},
    {'uriTemplate': 'vantage://job/{job_id}', 'name': 'job', 'title': 'A background job',
     'description': 'A job\'s state and, when done, its result', 'mimeType': JSON},
]


def study_names() -> list:
    return sorted(t.name[4:] for t in SV.TOOLS if t.name.startswith('get_') and not t.required and not t.router)


def docs_dir():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return root if os.path.isfile(os.path.join(root, 'README.md')) else None


def doc_files() -> dict:
    root = docs_dir()
    if root is None:
        return {}
    out = {'README': os.path.join(root, 'README.md')}
    d = os.path.join(root, 'docs')
    if os.path.isdir(d):
        for n in sorted(os.listdir(d)):
            if n.endswith('.md'):
                out[n[:-3]] = os.path.join(d, n)
    if os.path.isfile(os.path.join(root, 'CHANGELOG.md')):
        out['CHANGELOG'] = os.path.join(root, 'CHANGELOG.md')
    return out


def resource_list() -> list:
    out = [{'uri': u, 'name': n, 'title': t, 'description': d,
            'mimeType': 'image/png' if u.endswith('.png') else JSON} for u, n, t, d, *_ in STATIC]
    for name, path in doc_files().items():
        out.append({'uri': f'vantage://docs/{name}', 'name': f'docs-{name}', 'title': f'Documentation: {name}',
                    'description': 'Vantage documentation', 'mimeType': 'text/markdown',
                    'size': os.path.getsize(path)})
    return out


class NotFound(Exception):
    pass


def resolve(uri: str):
    """('tool', name, args) | ('doc', path) | ('job', id) for a resource URI; NotFound otherwise."""
    if uri in STATIC_BY_URI:
        r = STATIC_BY_URI[uri]
        return ('tool', r[4], dict(r[5]))
    m = re.fullmatch(r'vantage://camera/([^/?#]+)', uri)
    if m:
        return ('tool', 'get_camera', {'camera': m.group(1)})
    m = re.fullmatch(r'vantage://study/([a-z0-9_]+)', uri)
    if m:
        if m.group(1) not in study_names():
            raise NotFound(f'no study {m.group(1)!r} (vantage://study/ takes: {", ".join(study_names())})')
        return ('tool', 'get_' + m.group(1), {})
    m = re.fullmatch(r'vantage://view/(top|camera-([^/?#.]+))(?:\.png)?', uri)
    if m:
        return ('tool', 'look', {'view': 'top'} if m.group(1) == 'top' else {'view': 'camera', 'camera': m.group(2)})
    m = re.fullmatch(r'vantage://docs/([A-Za-z0-9_.-]+)', uri)
    if m:
        f = doc_files().get(m.group(1))
        if f is None:
            raise NotFound(f'no documentation page {m.group(1)!r} here' + (
                f' (pages: {", ".join(doc_files())})' if doc_files() else ' (the docs are not installed with this '
                                                                         'Vantage: see the website)'))
        return ('doc', f)
    m = re.fullmatch(r'vantage://job/([A-Za-z0-9_-]+)', uri)
    if m:
        return ('job', m.group(1))
    raise NotFound(f'no resource {uri}')


def contents_from_result(uri: str, res: dict) -> list:
    """An MCP tool result as resource contents: text as JSON (or plain text), pictures as PNG blobs."""
    out = []
    for c in res.get('content', []):
        if c.get('type') == 'image':
            out.append({'uri': uri, 'mimeType': c.get('mimeType', 'image/png'), 'blob': c['data']})
        elif c.get('type') == 'text':
            body = c.get('text', '')
            out.append({'uri': uri, 'mimeType': JSON if SV.structured(body) is not None or
                        body.lstrip().startswith('[') else 'text/plain', 'text': body})
    return out


def doc_contents(uri, path) -> list:
    with open(path, encoding='utf-8') as fh:
        return [{'uri': uri, 'mimeType': 'text/markdown', 'text': fh.read()}]


def blob(data: bytes, uri, mime) -> dict:
    return {'uri': uri, 'mimeType': mime, 'blob': base64.b64encode(data).decode('ascii')}


# =========================================================================== prompts
PROMPTS = [
    {'name': 'design_review', 'title': 'Review the rig design',
     'description': 'A full review of the design: coverage, blind regions, seams, issues, requirements, and what to '
                    'change, each change tried before it is proposed',
     'arguments': [{'name': 'focus', 'description': 'what matters most (e.g. pedestrians close to the car, parking, '
                                                    'highway range); optional', 'required': False}]},
    {'name': 'plan_rig_from_brief', 'title': 'Plan a rig from a brief',
     'description': 'From a written brief to requirements, a design-agent search and a chosen layout',
     'arguments': [{'name': 'brief', 'description': 'the brief: what the rig records, where, what must be seen',
                    'required': True}]},
    {'name': 'close_blind_spots', 'title': 'Close the blind spots',
     'description': 'Find the blind regions and close them with the fewest changes, each tried first',
     'arguments': [{'name': 'max_cameras', 'description': 'how many cameras the rig may have at most; optional',
                    'required': False}]},
    {'name': 'calibration_plan', 'title': 'Plan the calibration',
     'description': 'Calibration targets, observability, camera–lidar calibration and the clock tree',
     'arguments': []},
    {'name': 'verify_recording', 'title': 'Check a recorded drive against the plan',
     'description': 'A recorded drive (ROS 2 bag or nuScenes folder) checked against the plan: streams, timing, '
                    'mounting, online calibration',
     'arguments': [{'name': 'path', 'description': 'the recording', 'required': True}]},
    {'name': 'perception_study', 'title': 'Perception in the loop',
     'description': 'How well detection works on this rig and where crashes start, through a plan the user approves',
     'arguments': [{'name': 'classes', 'description': 'classes that matter (e.g. pedestrian, cyclist); optional',
                    'required': False}]},
    {'name': 'compare_variants', 'title': 'Compare the layout variants',
     'description': 'The project\'s variants side by side, with a recommendation and its reasons', 'arguments': []},
    {'name': 'explain_issue', 'title': 'Explain an issue',
     'description': 'Why an issue in the Issues list happens and the smallest change that fixes it',
     'arguments': [{'name': 'issue', 'description': 'the issue\'s text (or a camera tag)', 'required': True}]},
    {'name': 'import_and_start', 'title': 'Start from a vehicle model',
     'description': 'Import a vehicle model, check the import, and lay out a first camera rig',
     'arguments': [{'name': 'path', 'description': 'the model file', 'required': True},
                   {'name': 'cameras', 'description': 'how many cameras (default 6); optional', 'required': False}]},
]
PROMPTS_BY_NAME = {p['name']: p for p in PROMPTS}

_TEXT = {
    'design_review': (
        'Review this Vantage rig design{focus}.\n\n'
        '1. vantage_status, then get_overview and get_coverage.\n'
        '2. get_issues and find_blind_regions; explain_point for any region that matters.\n'
        '3. get_seams (with its advice) and get_requirements (run: true if they are stale).\n'
        '4. For every change worth making, simulate_changes first and keep only what improves the figures without '
        'new issues.\n'
        '5. Summarise: what is good, what is not, and the changes ranked by effect, each with its before and after '
        'figures. Make no change without asking; when asked, edit_project applies them as one undo step and '
        'render_view (or look without the window) shows the result.'),
    'plan_rig_from_brief': (
        'Plan a camera rig from this brief:\n\n{brief}\n\n'
        '1. read_requirements on the brief; show the requirements it reads and ask which to accept '
        '(propose_requirements puts them on the scorecard in the window).\n'
        '2. design_rig on the accepted requirements (start_job if it may take long; get_job for the result).\n'
        '3. compare the options it gives (compare_variants, predict_detection where a learned model exists) and '
        'explain the trade-offs.\n'
        '4. When the user picks one, apply it (edit_project) and check get_requirements.'),
    'close_blind_spots': (
        'Close the blind spots of this design{maxcam}.\n\n'
        '1. find_blind_regions (largest first) and explain_point on each to see which cameras miss it and why.\n'
        '2. Try, with simulate_changes: aiming an existing camera (an aim operation at the region), a lens change, '
        'moving a camera; then add_camera_options for one more camera.\n'
        '3. Prefer the smallest change; keep only changes that leave no new issues.\n'
        '4. Propose the best one with its figures; when the user agrees, edit_project, then look (top) to show it.'),
    'calibration_plan': (
        'Plan the calibration of this rig.\n\n'
        '1. get_calibration_observability: per camera the 1-sigma pose from the planned boards and the weakest '
        'direction; apply its suggested board moves only with the user.\n'
        '2. plan_lidar_calibration for the camera–lidar pairs, then calibrate_lidar_camera on the best pair.\n'
        '3. get_sync_plan and get_clock_tree: trigger times against the lidar sweep and the timestamp errors.\n'
        '4. Summarise the calibration procedure step by step with the figures it should reach.'),
    'verify_recording': (
        'Check the recording at {path} against the plan.\n\n'
        '1. run_batch with ["--drive-log", "{path}"] (start_job for a long recording); read its JSON.\n'
        '2. If it is a drive of the built rig: run_batch ["--online-calib", "{path}"] for knocks and flex.\n'
        '3. get_drive_log_check and get_online_calib for the results the window keeps.\n'
        '4. Report per stream and camera what is fine, what is off and what to do (re-mount, recalibrate, '
        're-record).'),
    'perception_study': (
        'Study how well perception works on this rig{classes}.\n\n'
        '1. get_rig_perception, get_risk_coverage and get_perf_model: what has been measured already.\n'
        '2. plan_perception_scene_set and suggest_simulations for what to run next; estimate_plan and validate_plan '
        'for its cost.\n'
        '3. propose_plan so the user approves the CARLA runs in the window (they never start from here).\n'
        '4. When results are in, explain per class and region where detection falls short and why.'),
    'compare_variants': (
        'Compare the layout variants of this project.\n\n'
        '1. vantage_status for the variants, compare_variants for their figures.\n'
        '2. For the two best: get_seams, get_recording_budget and get_power_budget (activate each with '
        'edit_project only if the user agrees, then back).\n'
        '3. Recommend one with its reasons and what it costs.'),
    'explain_issue': (
        'Explain this issue: {issue}\n\n'
        '1. get_issues to find it; get_camera for the cameras it names; explain_point where it is a place.\n'
        '2. Say why it happens in plain words, with the figures.\n'
        '3. Find the smallest fix with simulate_changes and give its before and after.'),
    'import_and_start': (
        'Start a project from the vehicle model at {path}.\n\n'
        '1. import_vehicle with the path; read its report (units, up, forward, wheels, size). If the size or axes '
        'look wrong, import again with them given.\n'
        '2. get_overview; if there are no cameras, add_camera_options or optimise_rig for {cameras} cameras; else '
        'check the coverage.\n'
        '3. render_view (or look without the window) to show the result.'),
}


def prompt_messages(name, args) -> dict:
    p = PROMPTS_BY_NAME.get(name)
    if p is None:
        raise NotFound(f'no prompt {name!r}')
    args = args or {}
    missing = [a['name'] for a in p['arguments'] if a.get('required') and not str(args.get(a['name']) or '').strip()]
    if missing:
        raise ValueError(f'prompt {name}: {", ".join(missing)} needed')
    vals = {
        'focus': f' with this focus: {args["focus"]}' if args.get('focus') else '',
        'brief': args.get('brief', ''),
        'maxcam': f' (at most {args["max_cameras"]} cameras)' if args.get('max_cameras') else '',
        'path': args.get('path', ''),
        'classes': f' for {args["classes"]}' if args.get('classes') else '',
        'issue': args.get('issue', ''),
        'cameras': args.get('cameras') or 6,
    }
    text = _TEXT[name].format(**vals)
    return {'description': p['description'],
            'messages': [{'role': 'user', 'content': {'type': 'text', 'text': text}}]}


# =========================================================================== completion
def complete(ref: dict, argument: dict, lookup) -> list:
    """Values for one argument of a prompt or resource template, filtered by what was typed. lookup(kind) gives
    project values: 'cameras', 'variants', 'jobs'."""
    name, typed = argument.get('name', ''), str(argument.get('value', ''))
    kind = ref.get('type')
    vals = []
    if kind == 'ref/resource':
        tpl = ref.get('uri', '')
        if '{tag}' in tpl:
            vals = lookup('cameras')
        elif '{name}' in tpl and 'study' in tpl:
            vals = study_names()
        elif '{name}' in tpl and 'docs' in tpl:
            vals = list(doc_files())
        elif '{view}' in tpl:
            vals = ['top'] + [f'camera-{t}' for t in lookup('cameras')]
        elif '{job_id}' in tpl:
            vals = lookup('jobs')
    elif kind == 'ref/prompt':
        if name in ('issue',):
            vals = lookup('issues')
        elif name == 'cameras' or name == 'max_cameras':
            vals = [str(n) for n in range(1, 9)]
        elif name == 'classes':
            vals = ['pedestrian', 'cyclist', 'car', 'truck', 'motorcycle', 'pedestrian, cyclist']
        elif name == 'path':
            vals = _paths(typed)
    low = typed.lower()
    vals = [v for v in vals if str(v).lower().startswith(low)] or [v for v in vals if low in str(v).lower()]
    return [str(v) for v in vals][:100]


def _paths(typed) -> list:
    """File-name completion for a path argument."""
    base = os.path.expanduser(typed) if typed else os.getcwd() + os.sep
    d, stem = (base, '') if base.endswith(os.sep) else os.path.split(base)
    try:
        names = sorted(os.listdir(d or '.'))
    except OSError:
        return []
    out = []
    for n in names:
        if n.startswith('.') or not n.startswith(stem):
            continue
        full = os.path.join(d, n)
        out.append(full + (os.sep if os.path.isdir(full) else ''))
    return out[:100]


def dumps(doc) -> str:
    return json.dumps(doc, ensure_ascii=False, default=str)
