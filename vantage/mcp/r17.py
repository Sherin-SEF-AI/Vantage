"""MCP tools of round 17's studies. Each change is one undo step ("Claude · …") on the session, as edit_project's
are, and returns what changed; the lab tools only read. The other read tools are the assistant's
(assistant/r17_tools.py), offered here as every assistant tool is."""
from __future__ import annotations


def _add_thermal_camera(s, module_id=None, lens_id=None, **_):
    from ..core import thermal_ir as TH
    from . import edits as E
    from .service import _figures
    tc = s.context()
    mid, lid = module_id or 'flir_bosonplus_640', lens_id or 'bosonplus640_as32'
    from ..core.lens import builtin_modules, builtin_lenses
    mods = {m.id: m for m in builtin_modules()}
    if mid not in mods and mid not in s.project().modules:
        return {'error': f'no module {mid!r}', 'changed': 'nothing',
                'thermal_modules': sorted(k for k, m in mods.items() if TH.is_thermal(m))}
    if not TH.is_thermal(mods.get(mid) or s.project().modules.get(mid)):
        return {'error': f'{mid!r} is not a thermal (LWIR) module', 'changed': 'nothing'}
    lenses = {l.id: l for l in builtin_lenses()}
    lz = lenses.get(lid) or s.project().lenses.get(lid)
    if lz is None or (getattr(lz, 'module_id', mid) or mid) != mid:
        return {'error': f'lens {lid!r} does not fit {mid!r}',
                'lenses': sorted(k for k, l in lenses.items() if getattr(l, 'module_id', '') == mid),
                'changed': 'nothing'}
    out = {}
    label = E.title([f'added a thermal camera ({mid}, {lid}) on the front bumper'])

    def fn(pj):
        cam = TH.add_thermal_camera(pj, mid, lid, engine=tc.engine)
        out.update(tag=cam.tag, x=cam.x, y=cam.y, z=cam.z, yaw=cam.yaw, pitch=cam.pitch, module_id=mid, lens_id=lid)
    try:
        s.edit(['variant', 'library'], fn, label)
    except ValueError as e:
        return {'error': str(e), 'changed': 'nothing'}
    return {'camera': out, 'undo_step': label, 'after': _figures(s.context()),
            'next': 'get_thermal_night reads how far it sees at night'}


def _add_frame_to_rig(s, **_):
    from ..core import framing as FR
    from . import edits as E
    from .service import _figures
    res = FR.synthesise(s.project())
    if not res.members:
        return {'error': 'nothing to frame: add a roof rack or a sensor mast first', 'changed': 'nothing'}
    label = E.title([f'added the T-slot frame to the rig ({len(res.members)} members, {len(res.plates)} plates)'])
    out = {}

    def fn(pj):
        before = {p.uid for p in pj.rig}
        FR.apply_to_rig(pj, FR.synthesise(pj))
        out['parts'] = [p.name for p in pj.rig if p.uid not in before]
    s.edit(['rig', 'analysis'], fn, label)
    return {'added': out.get('parts', []), 'undo_step': label, 'summary': FR.summary(res),
            'after': _figures(s.context())}


# =========================================================================== project folders and the lab
def _diff_projects(s, a=None, b=None, **_):
    """What b changes against a, by meaning (core.project_dir.diff); either may be path@REV of a git repository;
    without a, the project Claude works on; without b, a's working copy (a must then be path@REV)."""
    import os
    from ..core import project_dir as PD

    def load(spec):
        spec = os.path.expanduser(spec)                # ~/rigs/x.vantage.d@HEAD as well as a plain path
        rev = '@' in spec and not os.path.exists(spec)
        path = spec.rsplit('@', 1)[0] if rev else spec
        path = os.path.abspath(path)
        if not (path.lower().endswith('.vantage') or PD.is_project_dir(path) or PD.wants_dir(path)):
            raise ValueError(f'{spec}: a .vantage file, a project folder or path@REV')
        return PD.load_spec(path + '@' + spec.rsplit('@', 1)[1] if rev else path)
    try:
        if a:
            pa, la = load(a)
        else:
            pa, la = s.project(), 'this project'
        if b:
            pb, lb = load(b)
        elif a and '@' in a:
            pb, lb = load(a.rsplit('@', 1)[0])
            lb += ' (working copy)'
        else:
            pb, lb = s.project(), 'this project'
    except (PD.ProjectDirError, OSError, ValueError, KeyError) as e:
        return {'error': str(e)}
    doc = PD.diff(pa, pb, la, lb)
    text = PD.diff_text(doc, max_lines=200)
    return {'same': doc['same'], 'text': text, 'summary': doc.get('summary')}


def _lab(url):
    from ..lab.client import LabClient
    return LabClient(url)


def _lab_err(e):
    return {'error': str(e), 'status': getattr(e, 'status', None),
            'hint': 'the token is the one saved on the Lab page (or VANTAGE_LAB_TOKEN)'}


def _lab_versions(s, url=None, project=None, **_):
    from ..lab.client import LabClientError
    if not url or not project:
        return {'error': 'url (the lab server) and project (its project id) are needed'}
    try:
        return {'project': project, 'versions': _lab(url).versions(project)}
    except LabClientError as e:
        return _lab_err(e)


def _lab_compare(s, url=None, project=None, a=None, b=None, **_):
    from ..lab.client import LabClientError
    if not url or not project or a is None or b is None:
        return {'error': 'url, project and two versions a and b (numbers or head) are needed'}
    try:
        return _lab(url).compare(project, a, b)
    except LabClientError as e:
        return _lab_err(e)


def _lab_pooled_twin(s, url=None, projects=None, **_):
    from ..core import fleet_twin as FT
    from ..lab.client import LabClientError
    if not url:
        return {'error': 'url (the lab server) is needed'}
    tc = s.context()
    try:
        res, doc = _lab(url).pooled_twin(tc.project, projects or None, engine=tc.engine)
    except LabClientError as e:
        return _lab_err(e)
    if res is None:
        return {'error': 'cancelled'}
    out = FT.summary(res)
    out['pooled'] = {'sources': len(doc.get('sources') or []), 'skipped': doc.get('skipped') or []}
    return out


def tools(Tool):
    S = {'type': 'string'}
    return [
        Tool('add_thermal_camera', 'Add a thermal camera',
             'Put a thermal (LWIR) camera on the active variant\'s front bumper, looking ahead and level, as the '
             'Thermal cameras page\'s Add button does (default a Teledyne FLIR Boson+ 640 with the 32° AS32 lens). '
             'One undo step. Move it with edit_project; get_thermal_night says how far it sees.',
             {'module_id': {**S, 'description': 'a thermal module of the library (default flir_bosonplus_640)'},
              'lens_id': {**S, 'description': 'one of its lenses (default bosonplus640_as32)'}},
             _add_thermal_camera, read_only=False),
        Tool('add_frame_to_rig', 'Add the frame to the rig',
             'Put the T-slot frame Vantage synthesises from the roof rig (get_frame) into the rig as parts, as the '
             'Frame page\'s Add to rig does: earlier frame parts are replaced and the parts it stands for hidden. '
             'One undo step.', {}, _add_frame_to_rig, read_only=False),
        Tool('diff_projects', 'Compare two projects',
             'What one project changes against another, by meaning: cameras and sensors moved by mm and turned by '
             'degrees, lenses and modules swapped, settings, requirements, the fleet. a and b are .vantage files, '
             'project folders or path@REV in a git repository (rig.vantage.d@HEAD~1); without a, the project open '
             'here; with only a = path@REV, against its working copy. Nothing changes.',
             {'a': {**S, 'description': 'the base (default: the project open here)'},
              'b': {**S, 'description': 'the changed one (default: the project open here, or a\'s working copy)'}},
             _diff_projects),
        Tool('lab_versions', 'List lab versions',
             'The version history of a project on the team\'s lab server (number, digest, message, who, when). Uses '
             'the token saved on the Lab page. Nothing changes; publishing is the user\'s, on the Lab page.',
             {'url': {**S, 'description': 'the lab server, e.g. http://lab.local:8765'},
              'project': {**S, 'description': 'the lab project id'}}, _lab_versions, required=('url', 'project')),
        Tool('lab_compare', 'Compare lab versions',
             'What changed between two versions of a project on the lab server, by meaning (as diff_projects).',
             {'url': S, 'project': S, 'a': {'type': ['integer', 'string'], 'description': 'a version number or head'},
              'b': {'type': ['integer', 'string']}}, _lab_compare, required=('url', 'project', 'a', 'b')),
        Tool('lab_pooled_twin', 'Pooled fleet twin',
             'The fleet twin over every fleet the lab shares with the user\'s team, re-based onto this project\'s '
             'plan: urgency counts, the recalibration schedule, knocks, outliers. Seconds to a minute; long pools '
             'are best run with start_job. Nothing changes.',
             {'url': S, 'projects': {'type': 'array', 'items': S, 'description': 'lab project ids (default: all)'}},
             _lab_pooled_twin, required=('url',)),
    ]
