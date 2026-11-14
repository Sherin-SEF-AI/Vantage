"""Unpacked project folders: a project as a directory of stable, sorted, diff-friendly text files that git can diff and
merge; a semantic diff of two projects or two git revisions; a three-way merge of project JSON (a git merge driver).

The folder (`name.vantage.d/`, or any folder the user picks)
-------------------------------------------------------------
    project.json              the project's own fields (name, vehicle description, ROI, boards, obstacles, rig parts,
                              ECU, frame, streams without passwords, hardware) and which mesh file holds the body
    library/modules.json      the component library, one file per kind, entries by id
    library/lenses.json
    library/sensor_models.json
    variants/<name>.json      one file per variant (its cameras, sensors, seams); "order" keeps the variants' order
    analysis/base.json        the base analysis settings (grid cell, densities, tolerances, load cases)
    analysis/<slot>.json      one file per study whose settings are not the defaults (an empty slot has no file)
    requirements.json         the requirements scorecard's settings (analysis.requirements)
    design_log.jsonl          the design log (analysis.assistant_log), one entry per line
    fleet.json                vehicles built to the design and their calibration records
    photos.json               reference photos fitted to the body
    log.jsonl                 the history, one entry [time, kind, text] per line
    mesh.npz | mesh.obj       the vehicle body: NumPy arrays (as in the .vantage file, written byte-for-byte the same
                              every time) or, when asked and the mesh is small, a text OBJ (faces grouped part/zone)
    thumb.png                 the picture the start window shows
    .gitattributes            binaries marked, the line files merged by union, JSON by the Vantage merge driver
    .gitignore                the per-user files below
    .vantage-view.json        which variant is shown (per user: kept out of git, so two people never conflict on it)

Every JSON file Vantage writes carries "$vantage": "<kind>/1" (the key sorts first). Keys are sorted, two-space-free
one-item-per-line indentation, numbers as Python writes them (the shortest text that reads back as the same double, so
a value survives save and load exactly), short lists of numbers on one line, UTF-8, LF line ends, a final newline.
Nothing that changes on every save is written: no save time, and a file whose bytes would not change is not rewritten
(git sees no change; the mesh's NPZ has fixed member times). Saving never deletes a file it did not write: a file is
removed only when it carries Vantage's mark and the project no longer has what it held (a deleted variant, a study
reset to its defaults). Saving into a folder that holds other files but no Vantage project is refused unless asked.

Round trip: a .vantage file saved as a folder and back is the same project (the same project.json content apart from
the save time, the same mesh arrays, the same content hash, core.provenance.project_digest). validate_round_trip()
proves it on a project; tests/test_project_dir.py on the sample and the templates.

Git
---
Two people who change different variants change different files; git merges them with no help. The history and the
design log are JSON lines merged by git's union driver (both sides' new lines kept). For everything else, an optional
merge driver (git_setup(folder) or `vantage diff --git-setup FOLDER` registers it in that repository's .git/config)
merges JSON three ways by meaning: objects key by key, cameras, sensors, variants and every list of objects with a uid
(or id) by identity, so two people moving different cameras of one variant, or changing different settings of one
study, merge cleanly; the same value changed two ways is a conflict (ours is kept, git marks the file, the conflicts
are printed). The same driver merges zipped .vantage files (project.json by meaning, the mesh from the side that
changed it). Without the driver git's own line merge applies, which the one-item-per-line layout makes work in most
cases.

`vantage diff A B` (main(['diff', A, B])): a semantic diff of two projects: .vantage files, folders, or either at a git
revision (`path@REV`, read with read-only git commands): variants added, removed, renamed; cameras and sensors added,
removed, moved (by how much, in mm, and turned, in degrees, with the rotation angle between the two orientations), lens
and module changes, every other field changed; the library, the vehicle and its mesh, every study's settings, the
fleet, the requirements and the design log. As text and as JSON (schema vantage.project_diff/1). Exit code 0: the
same, 1: they differ, 2: a project could not be read.

Frames: positions in the vehicle frame (ISO 8855, rear axle), metres in the files, millimetres in the diff; angles in
degrees, intrinsic Z-Y-X, pitch positive down.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import zipfile

import numpy as np

LAYOUT = 'vantage.dir/1'
DIFF_SCHEMA = 'vantage.project_diff/1'
MERGE_SCHEMA = 'vantage.project_merge/1'
DIR_SUFFIX = '.vantage.d'
MARK = '$vantage'
VIEW_FILE = '.vantage-view.json'
LAB_FILE = '.vantage-lab.json'          # the lab client's sync state (vantage.lab.client): per user, never published
TEXT_MESH_MAX_VERTICES = 200_000        # a text mesh (OBJ) is offered up to this size: beyond it the text is huge
NPZ_TIME = (1980, 1, 1, 0, 0, 0)        # the mesh NPZ's member times: fixed, so the same mesh is the same bytes
ARRAYS = ('vertices', 'faces', 'face_part', 'face_zone')
DIALOG_TEXT = 'Vantage project folder (name.vantage.d)'
FRAME_TEXT = 'vehicle (ISO 8855, rear axle): x forward, y left, z up; angles intrinsic Z-Y-X, pitch positive down'
EXIT_SAME, EXIT_DIFFERENT, EXIT_ERROR = 0, 1, 2
INLINE_LIST = 100                      # a list of plain values up to this many characters is written on one line
GITATTRIBUTES = """\
# Vantage project folder (written by Vantage; edit freely, Vantage leaves an edited copy alone)
*.json text eol=lf merge=vantage
*.jsonl text eol=lf merge=union
*.obj text eol=lf
mesh.npz binary
thumb.png binary
"""
GITIGNORE = """\
# Vantage project folder (written by Vantage): per-user files, never shared
.vantage-view.json
.vantage-lab.json
*.vantage-tmp
"""
ASSUMPTIONS = (
    'The folder holds exactly what a .vantage file holds; only the save time is left out (it would change every '
    'file on every save) and the variant on show is a per-user file kept out of git. Git merges files, not '
    'meaning: two people changing the same value two ways is a conflict that a person resolves (the merge driver '
    'keeps ours and lists them). The mesh and the thumbnail are binary: two people re-importing the body is a '
    'conflict. A folder copied while Vantage saves it can hold a mix of old and new files (each file is replaced '
    'atomically, project.json last).')

_MARK_RE = re.compile(rb'^\s*\{\s*"\$vantage"\s*:', re.S)
_SLUG = re.compile(r'[^a-z0-9]+')


class ProjectDirError(ValueError):
    """A folder or revision that cannot be read or written as a project; the message says why."""


# =========================================================================== paths
def wants_dir(path: str) -> bool:
    """Is path meant as a project folder? An existing directory, or a name ending in .vantage.d."""
    p = str(path or '').rstrip('/\\')
    return bool(p) and (os.path.isdir(p) or p.lower().endswith(DIR_SUFFIX))


def is_project_dir(path: str) -> bool:
    """A folder holding a Vantage project (its project.json says so)."""
    f = os.path.join(str(path or ''), 'project.json')
    if not os.path.isfile(f):
        return False
    try:
        with open(f, 'rb') as fh:
            head = fh.read(4096)
        return bool(_MARK_RE.match(head)) or b'"format"' in head and b'"vantage"' in head
    except OSError:
        return False


def is_project_path(path: str) -> bool:
    """A .vantage file or a project folder."""
    return bool(path) and (os.path.isfile(path) and str(path).lower().endswith('.vantage') or is_project_dir(path))


def name_from_path(path: str) -> str:
    """The project's name from its file or folder: 'rig' for rig.vantage and rig.vantage.d."""
    b = os.path.basename(str(path).rstrip('/\\'))
    for suf in (DIR_SUFFIX, '.vantage'):
        if b.lower().endswith(suf):
            return b[:-len(suf)] or b
    return b


def default_dir_for(path: str) -> str:
    """rig.vantage -> rig.vantage.d beside it."""
    p = str(path).rstrip('/\\')
    if p.lower().endswith('.vantage'):
        return p + '.d'
    return p if p.lower().endswith(DIR_SUFFIX) else p + DIR_SUFFIX


def mtime(path: str) -> float:
    """The newest modification time of a project (a file's, or the newest of a folder's project files)."""
    if not os.path.isdir(path):
        return os.path.getmtime(path)
    t = os.path.getmtime(path)
    for rel in list_files(path):
        try:
            t = max(t, os.path.getmtime(os.path.join(path, rel)))
        except OSError:
            pass
    return t


def backup(path: str) -> str:
    """Copy a project as it is to <path>.bak (a file) or <path>.bak/ (a folder's project files only). Returns it."""
    import shutil
    dst = str(path).rstrip('/\\') + '.bak'
    if os.path.isdir(path):
        for rel in list_files(path, include_view=True):
            d = os.path.join(dst, rel)
            os.makedirs(os.path.dirname(d), exist_ok=True)
            shutil.copy2(os.path.join(path, rel), d)
    else:
        shutil.copy2(path, dst)
    return dst


# =========================================================================== stable JSON
def _plain(o):
    from .provenance import plain
    return plain(o)


def _scalar(o) -> bool:
    return o is None or isinstance(o, (bool, int, float, str))


def _fmt(o, ind: int) -> str:
    pad, pad1 = '  ' * ind, '  ' * (ind + 1)
    if isinstance(o, dict):
        if not o:
            return '{}'
        items = sorted(o.items(), key=lambda kv: str(kv[0]))
        return '{\n' + ',\n'.join(f'{pad1}{json.dumps(str(k), ensure_ascii=False)}: {_fmt(v, ind + 1)}'
                                  for k, v in items) + '\n' + pad + '}'
    if isinstance(o, list):
        if not o:
            return '[]'
        if all(_scalar(x) for x in o):
            one = json.dumps(o, ensure_ascii=False, allow_nan=True, separators=(', ', ': '))
            if len(one) <= INLINE_LIST and all(not isinstance(x, str) or len(x) < 40 for x in o):
                return one
        if all(isinstance(x, list) and all(_scalar(y) for y in x) for x in o):
            rows = [json.dumps(x, ensure_ascii=False, allow_nan=True, separators=(', ', ': ')) for x in o]
            if all(len(r) <= INLINE_LIST for r in rows):
                return '[\n' + ',\n'.join(pad1 + r for r in rows) + '\n' + pad + ']'
        return '[\n' + ',\n'.join(pad1 + _fmt(x, ind + 1) for x in o) + '\n' + pad + ']'
    return json.dumps(o, ensure_ascii=False, allow_nan=True)


def dumps_stable(o) -> bytes:
    """The stable text of a JSON value: keys sorted, one item per line, short plain lists on one line, numbers as
    Python's shortest round-trip text, UTF-8, LF, a final newline. The same value always gives the same bytes."""
    return (_fmt(_plain(o), 0) + '\n').encode('utf-8')


def _doc(kind: str, body: dict) -> bytes:
    return dumps_stable({MARK: f'{kind}/1', **body})


def _lines(rows) -> bytes:
    """JSON lines: one compact value per line, keys sorted (union-mergeable)."""
    return ''.join(json.dumps(r, ensure_ascii=False, allow_nan=True, sort_keys=True, separators=(',', ':')) + '\n'
                   for r in _plain(list(rows))).encode('utf-8')


def _read_lines(data: bytes, what: str) -> list:
    out = []
    for i, line in enumerate(data.decode('utf-8').splitlines()):
        s = line.strip()
        if not s:
            continue
        if s.startswith(('<<<<<<<', '=======', '>>>>>>>')):
            raise ProjectDirError(f'{what} has git conflict markers (line {i + 1}): finish the merge first')
        try:
            out.append(json.loads(s))
        except ValueError:
            raise ProjectDirError(f'{what}: line {i + 1} is not JSON') from None
    return out


def _read_json(data: bytes, what: str):
    try:
        return json.loads(data.decode('utf-8'))
    except (ValueError, UnicodeDecodeError) as e:
        text = data[:200000].decode('utf-8', 'replace')
        if re.search(r'^(<<<<<<<|>>>>>>>) ', text, re.M):
            raise ProjectDirError(f'{what} has git conflict markers: finish the merge first (or register Vantage\'s '
                                  'merge driver: vantage diff --git-setup)') from None
        raise ProjectDirError(f'{what} is not JSON: {e}') from None


def slug(name: str, fallback: str = 'variant') -> str:
    """A file name from a variant's name: lower-case ASCII letters, digits and dashes."""
    import unicodedata
    s = unicodedata.normalize('NFKD', str(name)).encode('ascii', 'ignore').decode().lower()
    s = _SLUG.sub('-', s).strip('-')[:60].strip('-')
    return s or fallback


# =========================================================================== the mesh
def _npz_bytes(arrays: dict) -> bytes:
    """An NPZ NumPy reads (np.load) whose bytes depend only on the arrays: fixed member order and times."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
        for k in ARRAYS:
            b = io.BytesIO()
            np.lib.format.write_array(b, np.ascontiguousarray(arrays[k]), allow_pickle=False)
            zi = zipfile.ZipInfo(k + '.npy', date_time=NPZ_TIME)
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = 0o644 << 16
            z.writestr(zi, b.getvalue(), compresslevel=6)
    return buf.getvalue()


def mesh_arrays(mesh) -> dict | None:
    """The arrays a project file stores (as Project.save writes them: float32 vertices, int32 faces)."""
    if mesh is None:
        return None
    return {'vertices': np.asarray(mesh.vertices).astype(np.float32), 'faces': np.asarray(mesh.faces).astype(np.int32),
            'face_part': np.asarray(mesh.face_part), 'face_zone': np.asarray(mesh.face_zone)}


def arrays_digest(arrays: dict | None) -> str | None:
    if arrays is None:
        return None
    h = hashlib.sha256()
    for k in ARRAYS:
        a = np.ascontiguousarray(arrays[k])
        a = a.astype(a.dtype.newbyteorder('<'), copy=False)
        h.update(f'{k}:{a.dtype.str}:{a.shape}\n'.encode())
        h.update(a.tobytes())
    return h.hexdigest()


def _f32(v) -> str:
    return np.format_float_positional(np.float32(v), unique=True, trim='-')


def obj_bytes(arrays: dict) -> bytes:
    """The mesh as a Wavefront OBJ: vertices as the shortest text that reads back as the same float32, faces 1-based,
    grouped by part and zone ('g part3_zone-1') in face order, the arrays' dtypes in a header comment."""
    v, f = arrays['vertices'], arrays['faces']
    fp, fz = np.asarray(arrays['face_part']), np.asarray(arrays['face_zone'])
    out = [f'# Vantage vehicle mesh: vehicle frame (ISO 8855, rear axle), metres; {len(v)} vertices, {len(f)} faces',
           f'# vantage-dtypes face_part={fp.dtype.str} face_zone={fz.dtype.str}']
    out += [f'v {_f32(a)} {_f32(b)} {_f32(c)}' for a, b, c in v]
    grp = None
    for i, (a, b, c) in enumerate(np.asarray(f, np.int64) + 1):
        g = (int(fp[i]), int(fz[i]))
        if g != grp:
            out.append(f'g part{g[0]}_zone{g[1]}')
            grp = g
        out.append(f'f {a} {b} {c}')
    return ('\n'.join(out) + '\n').encode('ascii')


_GROUP = re.compile(r'part(-?\d+)_zone(-?\d+)$')


def obj_arrays(data: bytes) -> dict:
    """The arrays of an OBJ written by obj_bytes (another OBJ: triangles only, every face part 0, no zone)."""
    vs, fs, parts, zones = [], [], [], []
    g = (0, -1)
    dt = {'face_part': '<i4', 'face_zone': '<i2'}
    for line in data.decode('utf-8', 'replace').splitlines():
        if line.startswith('v '):
            vs.append(line.split()[1:4])
        elif line.startswith('f '):
            idx = [int(t.split('/')[0]) for t in line.split()[1:]]
            if len(idx) < 3:
                continue
            for k in range(1, len(idx) - 1):            # a polygon as a fan of triangles
                fs.append((idx[0], idx[k], idx[k + 1]))
                parts.append(g[0])
                zones.append(g[1])
        elif line.startswith('g '):
            m = _GROUP.search(line.strip())
            g = (int(m.group(1)), int(m.group(2))) if m else (0, -1)
        elif line.startswith('# vantage-dtypes'):
            for t in line.split()[2:]:
                k, _, v = t.partition('=')
                if k in dt and v:
                    dt[k] = v
    if not vs or not fs:
        raise ProjectDirError('mesh.obj holds no triangles')
    v = np.array(vs, dtype=np.float32)
    f = np.array(fs, dtype=np.int64)
    f = np.where(f < 0, f + len(v) + 1, f) - 1                # OBJ: negative indices count from the end
    if f.min() < 0 or f.max() >= len(v):
        raise ProjectDirError('mesh.obj: a face names a vertex that is not there')
    return {'vertices': v, 'faces': f.astype(np.int32), 'face_part': np.array(parts, dtype=np.dtype(dt['face_part'])),
            'face_zone': np.array(zones, dtype=np.dtype(dt['face_zone']))}


# =========================================================================== project <-> files
def _analysis_split(analysis: dict) -> tuple:
    """(base settings, {slot: settings} of the non-empty dict slots, requirements, design log)."""
    base, slots = {}, {}
    req, dlog = {}, []
    for k, v in analysis.items():
        if k == 'requirements':
            req = v if isinstance(v, dict) else {}
        elif k == 'assistant_log':
            dlog = v if isinstance(v, list) else []
        elif isinstance(v, dict):
            if v:
                slots[k] = v
        else:
            base[k] = v
    return base, slots, req, dlog


def _variant_files(variants: list) -> dict:
    """{relative path: variant dict with its order}: file names from the names, unique (a uid suffix on a clash)."""
    out, taken = {}, set()
    for i, v in enumerate(variants):
        s = slug(v.get('name', ''))
        name = s
        if name in taken:
            name = f'{s}-{str(v.get("uid", ""))[:6] or i + 1}'
        k = 2
        while name in taken:
            name = f'{s}-{k}'
            k += 1
        taken.add(name)
        out[f'variants/{name}.json'] = dict(v, order=i)
    return out


def files_of(project, thumb_png: bytes | None = None, text_mesh: bool = False) -> dict:
    """{relative path: bytes}: the project as the folder's files (everything save_dir writes, the view file and the
    git files included). The same project gives the same bytes."""
    data = _plain(project.to_json())
    data.pop('saved_at', None)
    out = {}
    lib = {k: data.pop(k, {}) for k in ('modules', 'lenses', 'sensor_models')}
    for k, v in lib.items():
        out[f'library/{k}.json'] = _doc('library', {'kind': k, 'items': v})
    for rel, v in _variant_files(data.pop('variants', [])).items():
        out[rel] = _doc('variant', v)
    base, slots, req, dlog = _analysis_split(data.pop('analysis', {}))
    out['analysis/base.json'] = _doc('analysis', {'slot': 'base', 'settings': base})
    for k, v in slots.items():
        out[f'analysis/{k}.json'] = _doc('analysis', {'slot': k, 'settings': v})
    if req:
        out['requirements.json'] = _doc('requirements', {'settings': req})
    out['design_log.jsonl'] = _lines(dlog)
    out['fleet.json'] = _doc('fleet', {'fleet': data.pop('fleet', {}) or {}})
    out['photos.json'] = _doc('photos', {'photos': data.pop('photos', {})})
    out['log.jsonl'] = _lines(data.pop('log', []))
    active = data.pop('active', '')
    out[VIEW_FILE] = _doc('view', {'active': active})
    arrays = mesh_arrays(project.vehicle.mesh)
    mesh = None
    if arrays is not None:
        text = bool(text_mesh) and len(arrays['vertices']) <= TEXT_MESH_MAX_VERTICES
        mesh = {'file': 'mesh.obj' if text else 'mesh.npz', 'sha256': arrays_digest(arrays),
                'vertices': int(len(arrays['vertices'])), 'faces': int(len(arrays['faces']))}
        out[mesh['file']] = obj_bytes(arrays) if text else _npz_bytes(arrays)
    data['layout'] = LAYOUT
    data['mesh'] = mesh
    out['project.json'] = _doc('project', data)
    thumb = project.thumb_png if thumb_png is None else thumb_png
    if thumb:
        out['thumb.png'] = bytes(thumb)
    out['.gitattributes'] = GITATTRIBUTES.encode()
    out['.gitignore'] = GITIGNORE.encode()
    return out


SHARED_SKIP = (VIEW_FILE, LAB_FILE)


def publishable(files: dict) -> dict:
    """The files of files_of() that are the project (what the lab publishes): the per-user view file left out."""
    return {k: v for k, v in files.items() if k not in SHARED_SKIP}


def _known(rel: str) -> bool:
    if rel in ('project.json', 'requirements.json', 'design_log.jsonl', 'fleet.json', 'photos.json', 'log.jsonl',
               'mesh.npz', 'mesh.obj', 'thumb.png', VIEW_FILE):
        return True
    parts = rel.split('/')
    return len(parts) == 2 and parts[0] in ('library', 'variants', 'analysis') and parts[1].endswith('.json')


def list_files(folder: str, include_view: bool = True) -> list:
    """The project files a folder holds (relative, '/'-separated), the files Vantage reads."""
    out = []
    for rel in ('project.json', 'requirements.json', 'design_log.jsonl', 'fleet.json', 'photos.json', 'log.jsonl',
                'mesh.npz', 'mesh.obj', 'thumb.png') + ((VIEW_FILE,) if include_view else ()):
        if os.path.isfile(os.path.join(folder, rel)):
            out.append(rel)
    for sub in ('library', 'variants', 'analysis'):
        d = os.path.join(folder, sub)
        if os.path.isdir(d):
            out += sorted(f'{sub}/{n}' for n in os.listdir(d) if n.endswith('.json')
                          and os.path.isfile(os.path.join(d, n)))
    return out


def read_files(folder: str) -> dict:
    """{relative path: bytes} of a folder's project files."""
    out = {}
    for rel in list_files(folder):
        with open(os.path.join(folder, rel), 'rb') as fh:
            out[rel] = fh.read()
    return out


def _marked(data: bytes) -> bool:
    return bool(_MARK_RE.match(data[:4096]))


def data_from_files(files: dict, what: str = 'the folder') -> tuple:
    """(project.json-shaped data as Project.to_json writes it, mesh arrays or None, thumbnail bytes, notes) from a
    folder's files."""
    if 'project.json' not in files:
        raise ProjectDirError(f'{what} holds no project.json: not a Vantage project folder')
    top = _read_json(files['project.json'], 'project.json')
    if not isinstance(top, dict) or top.get('format') != 'vantage':
        raise ProjectDirError('project.json is not a Vantage project')
    notes = []
    data = {k: v for k, v in top.items() if k not in (MARK, 'layout', 'mesh')}
    for k in ('modules', 'lenses', 'sensor_models'):
        rel = f'library/{k}.json'
        d = _read_json(files[rel], rel) if rel in files else {}
        data[k] = d.get('items') if isinstance(d, dict) and isinstance(d.get('items'), dict) else {}
    vs = []
    for rel in sorted(r for r in files if r.startswith('variants/')):
        d = _read_json(files[rel], rel)
        if not isinstance(d, dict) or not str(d.get(MARK, '')).startswith('variant/'):
            notes.append(f'{rel} is not a Vantage variant: left out')
            continue
        order = d.pop('order', None)
        d.pop(MARK, None)
        vs.append((order if isinstance(order, (int, float)) and not isinstance(order, bool) else 1e9, rel, d))
    vs.sort(key=lambda t: (t[0], t[1]))
    data['variants'] = [d for _, _, d in vs]
    analysis = {}
    for rel in sorted(r for r in files if r.startswith('analysis/')):
        d = _read_json(files[rel], rel)
        if not isinstance(d, dict) or not str(d.get(MARK, '')).startswith('analysis/'):
            notes.append(f'{rel} is not a Vantage analysis file: left out')
            continue
        slot = d.get('slot') or rel[len('analysis/'):-len('.json')]
        st = d.get('settings')
        if slot == 'base':
            if isinstance(st, dict):
                analysis.update(st)
        else:
            analysis[slot] = st if st is not None else {}
    if 'requirements.json' in files:
        d = _read_json(files['requirements.json'], 'requirements.json')
        analysis['requirements'] = d.get('settings', {}) if isinstance(d, dict) else {}
    if 'design_log.jsonl' in files:
        rows, seen = [], set()
        for e in _read_lines(files['design_log.jsonl'], 'design_log.jsonl'):
            key = e.get('id') if isinstance(e, dict) else None
            if key is not None and key in seen:                 # a union merge of an entry edited on both sides
                rows = [r for r in rows if not (isinstance(r, dict) and r.get('id') == key)]
            seen.add(key)
            rows.append(e)
        analysis['assistant_log'] = rows
    data['analysis'] = analysis
    if 'fleet.json' in files:
        d = _read_json(files['fleet.json'], 'fleet.json')
        data['fleet'] = d.get('fleet', {}) if isinstance(d, dict) else {}
    if 'photos.json' in files:
        d = _read_json(files['photos.json'], 'photos.json')
        data['photos'] = d.get('photos', {}) if isinstance(d, dict) else {}
    if 'log.jsonl' in files:
        rows = [r for r in _read_lines(files['log.jsonl'], 'log.jsonl') if isinstance(r, list) and len(r) == 3]
        rows.sort(key=lambda r: (r[0] if isinstance(r[0], (int, float)) else 0))   # a union merge interleaves
        data['log'] = rows
    view = _read_json(files[VIEW_FILE], VIEW_FILE) if VIEW_FILE in files else {}
    uids = [v.get('uid') for v in data['variants']]
    act = view.get('active') if isinstance(view, dict) else None
    data['active'] = act if act in uids else (uids[0] if uids else '')
    data['saved_at'] = 0.0
    arrays = None
    mesh = top.get('mesh') if isinstance(top.get('mesh'), dict) else None
    name = (mesh or {}).get('file') or ('mesh.npz' if 'mesh.npz' in files else 'mesh.obj' if 'mesh.obj' in files
                                        else None)
    if name in files:
        if name == 'mesh.obj':
            arrays = obj_arrays(files[name])
        else:
            with np.load(io.BytesIO(files[name]), allow_pickle=False) as npz:
                arrays = {k: npz[k] for k in ARRAYS}
        if mesh and mesh.get('sha256') and arrays_digest(arrays) != mesh['sha256']:
            notes.append(f'{name} is not the mesh project.json records (a merge took them from different sides?): '
                         'the file\'s mesh is used')
    elif mesh:
        notes.append(f'{mesh.get("file")} is missing: the project has no vehicle body')
    return data, arrays, files.get('thumb.png', b''), notes


def project_from_files(files: dict, path: str = '', what: str = 'the folder'):
    """A Project from a folder's files (read_files, or a git revision's, or a lab version's)."""
    from .project import Project
    data, arrays, thumb, notes = data_from_files(files, what)
    p = Project.from_data(data, arrays, thumb, path)
    for n in notes:
        p.add_log('file', n)
    return p


def load_dir(path: str):
    """Project.load for a folder."""
    if not os.path.isdir(path):
        raise ProjectDirError(f'{path} is not a folder')
    p = project_from_files(read_files(path), path=os.path.abspath(path), what=os.path.basename(path))
    try:
        p.saved_at = os.path.getmtime(os.path.join(path, 'project.json'))
    except OSError:
        pass
    return p


def _write_atomic(path: str, data: bytes) -> bool:
    """Write only when the bytes differ; through a temporary file renamed into place. True when written."""
    try:
        with open(path, 'rb') as fh:
            if fh.read() == data:
                return False
    except OSError:
        pass
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.vantage-tmp'
    with open(tmp, 'wb') as fh:
        fh.write(data)
    os.replace(tmp, path)
    return True


def _ours_to_remove(folder: str, keep: set) -> list:
    """Files of the folder Vantage wrote (its mark, or a JSON-lines/mesh file it names) that the project no longer
    has: the only files a save removes."""
    out = []
    for rel in list_files(folder, include_view=False):
        if rel in keep or not _known(rel):
            continue
        full = os.path.join(folder, rel)
        if rel.endswith('.json'):
            try:
                with open(full, 'rb') as fh:
                    if _marked(fh.read(4096)):
                        out.append(rel)
            except OSError:
                pass
        elif rel in ('mesh.npz', 'mesh.obj'):
            out.append(rel)                 # the other mesh format, after a switch (project.json names the one used)
    return out


def check_folder(path: str, into_existing: bool = False) -> str | None:
    """Why a project cannot be saved into this folder, or None: a folder that is not empty must already be a Vantage
    project folder (or into_existing, e.g. the root of a git repository chosen on purpose)."""
    if os.path.exists(path) and not os.path.isdir(path):
        return f'{path} is a file, not a folder'
    if not os.path.isdir(path) or is_project_dir(path) or into_existing:
        return None
    try:
        others = [n for n in os.listdir(path) if n not in ('.git', '.gitattributes', '.gitignore')]
    except OSError as e:
        return str(e)
    if others:
        return (f'{os.path.basename(path.rstrip("/"))} holds other files and no Vantage project: choose an empty '
                'folder or a project folder (nothing was written)')
    return None


def save_dir(project, path: str, thumb_png: bytes | None = None, text_mesh: bool | None = None,
             into_existing: bool = False) -> dict:
    """Write the project as a folder (Project.save does this for a folder path). text_mesh None keeps the format the
    folder has (mesh.obj stays text), else NPZ. Returns {'written': [...], 'removed': [...], 'unchanged': n}."""
    path = os.path.abspath(str(path).rstrip('/\\'))
    why = check_folder(path, into_existing)
    if why:
        raise ProjectDirError(why)
    if text_mesh is None:
        text_mesh = os.path.isfile(os.path.join(path, 'mesh.obj')) and not os.path.isfile(os.path.join(path, 'mesh.npz'))
    files = files_of(project, thumb_png, text_mesh=text_mesh)
    os.makedirs(path, exist_ok=True)
    written, unchanged = [], 0
    for rel in ('.gitattributes', '.gitignore'):
        full = os.path.join(path, rel)
        if os.path.exists(full):            # the user's own, or one they may have edited: theirs to keep
            unchanged += 1
        elif _write_atomic(full, files[rel]):
            written.append(rel)
    order = sorted((r for r in files if r not in ('.gitattributes', '.gitignore', 'project.json')),
                   key=lambda r: (r.startswith('.'), r)) + ['project.json']
    for rel in order:
        if _write_atomic(os.path.join(path, *rel.split('/')), files[rel]):
            written.append(rel)
        else:
            unchanged += 1
    removed = []
    for rel in _ours_to_remove(path, set(files)):
        try:
            os.remove(os.path.join(path, *rel.split('/')))
            removed.append(rel)
        except OSError:
            pass
    project.path = path
    project.saved_at = __import__('time').time()
    if thumb_png:
        project.thumb_png = thumb_png
    return {'written': written, 'removed': removed, 'unchanged': unchanged}


def read_thumbnail(path: str) -> tuple:
    """core.project.read_thumbnail for a folder: (thumb png bytes, summary dict)."""
    try:
        with open(os.path.join(path, 'project.json'), 'rb') as fh:
            top = json.loads(fh.read().decode('utf-8'))
        thumb = b''
        if os.path.isfile(os.path.join(path, 'thumb.png')):
            with open(os.path.join(path, 'thumb.png'), 'rb') as fh:
                thumb = fh.read()
        files = {}
        for r in list_files(path):
            if r.startswith('variants/') or r == VIEW_FILE:
                with open(os.path.join(path, r), 'rb') as fh:
                    files[r] = fh.read()
        vs = []
        for r in sorted(x for x in files if x.startswith('variants/')):
            try:
                vs.append(json.loads(files[r]))
            except ValueError:
                pass
        view = json.loads(files[VIEW_FILE]) if VIEW_FILE in files else {}
        var = next((v for v in vs if v.get('uid') == view.get('active')), vs[0] if vs else None)
        return thumb, {'vehicle': top.get('vehicle', {}).get('name', ''),
                       'cameras': len(var.get('cameras', [])) if var else 0, 'variants': len(vs),
                       'template': top.get('vehicle', {}).get('template', '')}
    except Exception:                                   # noqa: BLE001  a peek for the start window: never fails
        return b'', {}


def convert(src: str, dst: str, text_mesh: bool | None = None) -> str:
    """A .vantage file to a folder or back (by dst: a folder path or name.vantage.d, else a .vantage file)."""
    from .project import Project
    p = Project.load(src)
    if wants_dir(dst):
        save_dir(p, dst, text_mesh=text_mesh)
    else:
        p.save(dst)
    return dst


def content_equal(a, b) -> dict:
    """Are two projects the same content? {'equal', 'digest_a', 'digest_b', 'json_equal', 'mesh_equal', 'differs'}
    (project.json as written, the save time left out; the mesh arrays as stored; core.provenance's content hash)."""
    from .provenance import project_digest
    ja, jb = _plain(a.to_json()), _plain(b.to_json())
    for j in (ja, jb):
        j.pop('saved_at', None)
    differs = [k for k in sorted(set(ja) | set(jb)) if ja.get(k) != jb.get(k)]
    ma, mb = arrays_digest(mesh_arrays(a.vehicle.mesh)), arrays_digest(mesh_arrays(b.vehicle.mesh))
    da, db = project_digest(a)['sha256'], project_digest(b)['sha256']
    return {'equal': not differs and ma == mb and da == db, 'json_equal': not differs, 'mesh_equal': ma == mb,
            'digest_a': da, 'digest_b': db, 'differs': differs}


# =========================================================================== git (read-only, and the driver setup)
def _git_env() -> dict:
    env = dict(os.environ)
    env.pop('GIT_DIR', None)
    env['GIT_TERMINAL_PROMPT'] = '0'
    env['GIT_OPTIONAL_LOCKS'] = '0'                     # a read never takes the index lock
    env['LC_ALL'] = 'C'
    return env


def _git(cwd: str, *args, check=True) -> bytes:
    try:
        r = subprocess.run(['git', '-C', cwd, *args], capture_output=True, env=_git_env(), timeout=120)
    except FileNotFoundError:
        raise ProjectDirError('git is not installed') from None
    except subprocess.TimeoutExpired:
        raise ProjectDirError(f'git {args[0]} did not answer in 120 s') from None
    if check and r.returncode != 0:
        raise ProjectDirError(f'git {" ".join(args[:3])}: {r.stderr.decode("utf-8", "replace").strip()[:400]}')
    return r.stdout


def git_files_at(path: str, rev: str) -> tuple:
    """('dir', {relative path: bytes}) or ('file', bytes): the project at path as it was at git revision rev
    (read-only commands: rev-parse, cat-file, ls-tree)."""
    path = os.path.abspath(path)
    base = path if os.path.isdir(path) else os.path.dirname(path)
    while base and not os.path.isdir(base):
        base = os.path.dirname(base)
    top = _git(base, 'rev-parse', '--show-toplevel').decode().strip()
    rel = os.path.relpath(path, top).replace(os.sep, '/')
    rel = '' if rel == '.' else rel
    spec = f'{rev}:{rel}'
    kind = _git(top, 'cat-file', '-t', spec).decode().strip()
    if kind == 'blob':
        return 'file', _git(top, 'cat-file', 'blob', spec)
    if kind != 'tree':
        raise ProjectDirError(f'{spec} is a {kind}, not a project')
    listing = _git(top, 'ls-tree', '-r', '-z', '--full-tree', rev, '--', rel or '.')
    out = {}
    for ent in listing.split(b'\0'):
        if not ent:
            continue
        meta, name = ent.split(b'\t', 1)
        mode, typ, sha = meta.decode().split()
        name = name.decode('utf-8')
        sub = name[len(rel) + 1:] if rel else name
        if typ == 'blob' and _known(sub):
            out[sub] = _git(top, 'cat-file', 'blob', sha)
    return 'dir', out


def load_spec(spec: str, rev: str | None = None):
    """A project from 'path', 'path@REV' (a git revision) or path with rev. Returns (Project, label)."""
    from .project import Project
    path = spec
    if rev is None and not os.path.exists(spec) and '@' in spec:
        path, rev = spec.rsplit('@', 1)
    label = f'{os.path.basename(path.rstrip("/"))}@{rev}' if rev else os.path.basename(path.rstrip('/'))
    if not rev:
        if not os.path.exists(path):
            raise ProjectDirError(f'{path} does not exist')
        return Project.load(path), label
    kind, got = git_files_at(path, rev)
    if kind == 'dir':
        return project_from_files(got, path='', what=label), label
    with tempfile.TemporaryDirectory(prefix='vantage-rev-') as td:
        f = os.path.join(td, 'rev.vantage')
        with open(f, 'wb') as fh:
            fh.write(got)
        p = Project.load(f)
        p.path = ''
    return p, label


def driver_command() -> str:
    """The merge driver's command line for .git/config (%O base, %A ours (the result), %B theirs, %P the path)."""
    exe = sys.executable or 'python3'
    if getattr(sys, 'frozen', False):
        return f'"{exe}" --merge-file %O %A %B %P'
    return f'"{exe}" -m vantage.core.project_dir merge-file %O %A %B %P'


def git_setup(folder: str, command: str | None = None) -> dict:
    """Register the merge driver in the git repository that holds folder (its .git/config, this repository only) and
    make sure the folder has its .gitattributes. A change to the user's repository: only on the user's request."""
    top = _git(folder, 'rev-parse', '--show-toplevel').decode().strip()
    cmd = command or driver_command()
    _git(top, 'config', 'merge.vantage.name', 'Vantage project: three-way merge by meaning')
    _git(top, 'config', 'merge.vantage.driver', cmd)
    ga = os.path.join(folder, '.gitattributes')
    wrote = False
    if not os.path.exists(ga):
        wrote = _write_atomic(ga, GITATTRIBUTES.encode())
    return {'repository': top, 'driver': cmd, 'gitattributes_written': wrote}


# =========================================================================== three-way merge
_MISSING = object()
ID_KEYS = ('uid', 'id')


def _ident(x):
    if isinstance(x, dict):
        for k in ID_KEYS:
            v = x.get(k)
            if isinstance(v, str) and v:
                return (k, v)
    return None


def _keyed(lst) -> dict | None:
    """{identity: item} when every item of the list has a distinct uid (or id); else None."""
    out = {}
    for x in lst:
        k = _ident(x)
        if k is None or k in out:
            return None
        out[k] = x
    return out


def _union_rows(base, ours, theirs) -> list:
    """Rows of both sides (the history log: [t, kind, text]): ours, then theirs' new rows, less the rows either side
    removed from the base, ordered by time (a stable sort keeps each side's own order)."""
    def key(r):
        return json.dumps(r, sort_keys=True)
    kb, ko, kt = {key(r) for r in base}, {key(r) for r in ours}, {key(r) for r in theirs}
    gone = (kb - kt) | (kb - ko)
    out, seen = [], set()
    for r in list(ours) + list(theirs):
        k = key(r)
        if k in seen or k in gone:
            continue
        seen.add(k)
        out.append(r)
    out.sort(key=lambda r: r[0] if isinstance(r, list) and r and _num(r[0]) else 0)
    return out


UNION_PATHS = ('log',)


def merge3(base, ours, theirs, path: str = '', conflicts: list | None = None):
    """The three-way merge of JSON values by meaning: objects key by key, lists of objects with a uid or id by
    identity (order: ours, theirs' new items after their predecessor), the history log by union; a value changed two
    ways is a conflict ({'path', 'base', 'ours', 'theirs'} appended to conflicts) and keeps ours. _MISSING marks an
    absent key. Returns the merged value (or _MISSING)."""
    conflicts = conflicts if conflicts is not None else []
    if ours == theirs:
        return ours
    if ours == base:
        return theirs
    if theirs == base:
        return ours
    if path in UNION_PATHS and all(isinstance(x, list) for x in (ours, theirs)):
        return _union_rows(base if isinstance(base, list) else [], ours, theirs)
    if isinstance(ours, dict) and isinstance(theirs, dict) and (isinstance(base, dict) or base is _MISSING):
        b = base if isinstance(base, dict) else {}
        out = {}
        for k in list(ours) + [k for k in theirs if k not in ours]:
            v = merge3(b.get(k, _MISSING), ours.get(k, _MISSING), theirs.get(k, _MISSING),
                       f'{path}.{k}' if path else str(k), conflicts)
            if v is not _MISSING:
                out[k] = v
        return out
    if isinstance(ours, list) and isinstance(theirs, list) and (isinstance(base, list) or base is _MISSING):
        b = base if isinstance(base, list) else []
        kb, ko, kt = _keyed(b), _keyed(ours), _keyed(theirs)
        if kb is not None and ko is not None and kt is not None and (ours or theirs):
            order = list(ko)
            tl = list(kt)
            for i, k in enumerate(tl):            # theirs' items ours lacks (new, or deleted by ours: merge3 decides)
                if k in ko:
                    continue
                prev = next((tl[j] for j in range(i - 1, -1, -1) if tl[j] in order), None)
                order.insert(order.index(prev) + 1 if prev is not None else 0, k)
            out = []
            for k in order:
                name = f'{path}[{k[1]}]'
                v = merge3(kb.get(k, _MISSING), ko.get(k, _MISSING), kt.get(k, _MISSING), name, conflicts)
                if v is not _MISSING:
                    out.append(v)
            return out
    conflicts.append({'path': path or '.', 'base': None if base is _MISSING else base,
                      'ours': None if ours is _MISSING else ours, 'theirs': None if theirs is _MISSING else theirs})
    return ours


def _merge_lines(base: bytes, ours: bytes, theirs: bytes) -> bytes:
    """A JSON-lines file: ours, then theirs' new lines (union, as git's union driver)."""
    def rows(b):
        return [ln for ln in b.decode('utf-8', 'replace').splitlines() if ln.strip()]
    bo, oo, to = rows(base), rows(ours), rows(theirs)
    gone = (set(bo) - set(to)) | (set(bo) - set(oo))
    out, seen = [], set()
    for r in oo + to:
        if r in seen or r in gone:
            continue
        seen.add(r)
        out.append(r)
    return ('\n'.join(out) + '\n').encode('utf-8') if out else b''


def merge_bytes(base: bytes, ours: bytes, theirs: bytes, name: str = '') -> tuple:
    """(merged bytes, conflicts) of one file of a project folder, or of a zipped .vantage (by its content)."""
    if name.endswith('.jsonl'):
        return _merge_lines(base, ours, theirs), []
    if ours[:2] == b'PK' and theirs[:2] == b'PK':
        return merge_zip(base, ours, theirs)
    conflicts = []
    b = _read_json(base, f'{name} (base)') if base.strip() else _MISSING
    o = _read_json(ours, f'{name} (ours)')
    t = _read_json(theirs, f'{name} (theirs)')
    saved = None
    if isinstance(o, dict) and isinstance(t, dict) and ('saved_at' in o or 'saved_at' in t):
        saved = max(x.pop('saved_at', 0) or 0 for x in (o, t))      # a save time: the newer, never a conflict
        if isinstance(b, dict):
            b.pop('saved_at', None)
    m = merge3(b, o, t, '', conflicts)
    if saved is not None and isinstance(m, dict):
        m['saved_at'] = saved
    if isinstance(m, dict) and MARK in m:
        return dumps_stable(m), conflicts
    return json.dumps(_plain(m), indent=1).encode('utf-8') + b'\n', conflicts


def merge_zip(base: bytes, ours: bytes, theirs: bytes) -> tuple:
    """Three zipped .vantage files merged: project.json by meaning (merge3; the save time the newer one), the mesh
    and the thumbnail from the side that changed them (both changed: a conflict, ours kept)."""
    def read(b):
        if not b:
            return {}
        with zipfile.ZipFile(io.BytesIO(b)) as z:
            return {n: z.read(n) for n in z.namelist()}
    zb, zo, zt = read(base), read(ours), read(theirs)
    conflicts = []
    jb = json.loads(zb['project.json']) if 'project.json' in zb else _MISSING
    jo, jt = json.loads(zo['project.json']), json.loads(zt['project.json'])
    so, st = jo.get('saved_at', 0), jt.get('saved_at', 0)
    for j in (jb, jo, jt):
        if isinstance(j, dict):
            j.pop('saved_at', None)
    m = merge3(jb, jo, jt, '', conflicts)
    m['saved_at'] = max(so or 0, st or 0)
    out = io.BytesIO()
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('project.json', json.dumps(m, indent=1))
        for n in ('mesh.npz', 'thumb.png'):
            b, o, t = zb.get(n), zo.get(n), zt.get(n)
            pick = o if o == t or t == b else t if o == b else o
            if o != t and o != b and t != b:
                if n == 'mesh.npz':
                    conflicts.append({'path': n, 'base': 'binary', 'ours': 'binary', 'theirs': 'binary'})
            if pick:
                z.writestr(n, pick)
    return out.getvalue(), conflicts


def merge_file(base_path: str, ours_path: str, theirs_path: str, name: str = '') -> int:
    """The git merge driver: merges into ours_path; 0 when clean, 1 with conflicts (listed on stderr)."""
    def rd(p):
        try:
            with open(p, 'rb') as fh:
                return fh.read()
        except OSError:
            return b''
    try:
        merged, conflicts = merge_bytes(rd(base_path), rd(ours_path), rd(theirs_path), name or ours_path)
    except (ProjectDirError, ValueError, KeyError, zipfile.BadZipFile) as e:
        print(f'vantage merge: {name}: {e}', file=sys.stderr)
        return 1
    with open(ours_path, 'wb') as fh:
        fh.write(merged)
    for c in conflicts[:50]:
        print(f'vantage merge: {name}: {c["path"]} changed both ways: ours {json.dumps(c["ours"])[:120]}, '
              f'theirs {json.dumps(c["theirs"])[:120]} (ours kept)', file=sys.stderr)
    return 1 if conflicts else 0


# =========================================================================== semantic diff
POSE = ('x', 'y', 'z', 'yaw', 'pitch', 'roll')
MOVE_EPS_MM = 1e-6
TURN_EPS_DEG = 1e-9
MAX_CHANGES = 400


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _same(a, b) -> bool:
    if _num(a) and _num(b):
        return a == b or (isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b))
    return a == b


def leaf_changes(a, b, path: str = '', out: list | None = None, limit: int = MAX_CHANGES) -> list:
    """[{'path', 'a', 'b'}] of every leaf that differs (lists of objects with a uid or id matched by identity;
    a missing side is None)."""
    out = [] if out is None else out
    if len(out) >= limit or (a is _MISSING and b is _MISSING):
        return out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            leaf_changes(a.get(k, _MISSING), b.get(k, _MISSING), f'{path}.{k}' if path else str(k), out, limit)
        return out
    if isinstance(a, list) and isinstance(b, list):
        ka, kb = _keyed(a), _keyed(b)
        if ka is not None and kb is not None and (a or b) and not (all(_scalar(x) for x in a + b)):
            for k in list(ka) + [k for k in kb if k not in ka]:
                leaf_changes(ka.get(k, _MISSING), kb.get(k, _MISSING), f'{path}[{k[1]}]', out, limit)
            return out
        if all(_scalar(x) for x in a + b) or len(a) != len(b):
            if not (len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))):
                out.append({'path': path, 'a': a, 'b': b})
            return out
        for i, (x, y) in enumerate(zip(a, b)):
            leaf_changes(x, y, f'{path}[{i}]', out, limit)
        return out
    if a is _MISSING or b is _MISSING or not _same(a, b):
        out.append({'path': path, 'a': None if a is _MISSING else a, 'b': None if b is _MISSING else b,
                    **({'added': True} if a is _MISSING else {'removed': True} if b is _MISSING else {})})
    return out


def _rotation_deg(pa: dict, pb: dict) -> float:
    from .mathutil import rot_zyx
    Ra = rot_zyx(pa['yaw'], pa['pitch'], pa['roll'])
    Rb = rot_zyx(pb['yaw'], pb['pitch'], pb['roll'])
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return float(math.degrees(math.acos(min(1.0, max(-1.0, c)))))


def _pose_change(a: dict, b: dict) -> dict | None:
    from .mathutil import wrap180
    try:
        pa = {k: float(a[k]) for k in POSE}
        pb = {k: float(b[k]) for k in POSE}
    except (KeyError, TypeError, ValueError):
        return None
    d = {f'd{k}_mm': (pb[k] - pa[k]) * 1000.0 for k in ('x', 'y', 'z')}
    d.update({f'd{k}_deg': wrap180(pb[k] - pa[k]) for k in ('yaw', 'pitch', 'roll')})
    moved = math.sqrt(sum(d[f'd{k}_mm'] ** 2 for k in ('x', 'y', 'z')))
    turned = _rotation_deg(pa, pb)
    if moved <= MOVE_EPS_MM and turned <= TURN_EPS_DEG and all(abs(d[f'd{k}_deg']) <= TURN_EPS_DEG
                                                                for k in ('yaw', 'pitch', 'roll')):
        return None
    return {'moved_mm': moved, 'rotation_deg': turned, **d}


def _match(a_items: list, b_items: list) -> list:
    """[(a or None, b or None)]: by uid, then by tag (or name) among the rest."""
    pairs, used = [], set()
    bu = {x.get('uid'): i for i, x in enumerate(b_items) if x.get('uid')}
    rest_a = []
    for x in a_items:
        j = bu.get(x.get('uid'))
        if j is not None and j not in used:
            used.add(j)
            pairs.append((x, b_items[j]))
        else:
            rest_a.append(x)
    for x in rest_a:
        key = x.get('tag') or x.get('name')
        j = next((i for i, y in enumerate(b_items) if i not in used and key and (y.get('tag') or y.get('name')) == key),
                 None)
        if j is not None:
            used.add(j)
            pairs.append((x, b_items[j]))
        else:
            pairs.append((x, None))
    pairs += [(None, y) for i, y in enumerate(b_items) if i not in used]
    return pairs


def _items_diff(a_items, b_items, kind: str) -> list:
    out = []
    for a, b in _match(a_items, b_items):
        if a is None or b is None:
            x = a or b
            out.append({'kind': kind, 'tag': x.get('tag', ''), 'name': x.get('name', ''), 'uid': x.get('uid', ''),
                        'status': 'added' if a is None else 'removed',
                        'pose': {k: x.get(k) for k in POSE}})
            continue
        pc = _pose_change(a, b)
        skip = set(POSE) | {'uid'}
        fields = [c for c in leaf_changes({k: v for k, v in a.items() if k not in skip},
                                          {k: v for k, v in b.items() if k not in skip})]
        if pc is None and not fields:
            continue
        row = {'kind': kind, 'tag': b.get('tag', ''), 'name': b.get('name', ''), 'uid': b.get('uid', ''),
               'status': 'changed', 'move': pc, 'fields': fields}
        if a.get('tag') != b.get('tag'):
            row['was'] = a.get('tag')
        out.append(row)
    return out


def _variant_diff(a: dict | None, b: dict | None) -> dict | None:
    if a is None or b is None:
        x = a or b
        return {'name': x.get('name', ''), 'uid': x.get('uid', ''), 'status': 'added' if a is None else 'removed',
                'cameras': len(x.get('cameras', [])), 'sensors': len(x.get('sensors', []))}
    items = _items_diff(a.get('cameras', []), b.get('cameras', []), 'camera') + \
        _items_diff(a.get('sensors', []), b.get('sensors', []), 'sensor')
    other = leaf_changes({k: v for k, v in a.items() if k not in ('cameras', 'sensors', 'created', 'uid')},
                         {k: v for k, v in b.items() if k not in ('cameras', 'sensors', 'created', 'uid')})
    if not items and not other:
        return None
    row = {'name': b.get('name', ''), 'uid': b.get('uid', ''), 'status': 'changed', 'items': items, 'fields': other}
    if a.get('name') != b.get('name'):
        row['was'] = a.get('name')
    return row


def _uid_names(data: dict) -> dict:
    """uid -> a readable name (variant names, camera and sensor tags) for the diff's text."""
    out = {}
    for v in data.get('variants', []):
        out[v.get('uid')] = f'variant "{v.get("name")}"'
        for c in v.get('cameras', []) + v.get('sensors', []):
            out[c.get('uid')] = c.get('tag')
    return out


def diff(a, b, a_label: str = 'A', b_label: str = 'B') -> dict:
    """The semantic diff of two projects (Project objects): what B changes against A."""
    from .provenance import project_digest
    ja, jb = _plain(a.to_json()), _plain(b.to_json())
    for j in (ja, jb):
        for k in ('saved_at', 'created', 'log'):
            j.pop(k, None)
    da, db = project_digest(a), project_digest(b)
    out = {'schema': DIFF_SCHEMA, 'frame': FRAME_TEXT,
           'units': {'move': 'mm', 'angles': 'deg', 'positions in values': 'm'},
           'a': {'label': a_label, 'name': a.name, 'sha256': da['sha256']},
           'b': {'label': b_label, 'name': b.name, 'sha256': db['sha256']},
           'same_content': da['sha256'] == db['sha256'],
           'sections_changed': sorted(k for k in set(da['sections']) | set(db['sections'])
                                      if da['sections'].get(k) != db['sections'].get(k))}
    variants = []
    for va, vb in _match(ja.get('variants', []), jb.get('variants', [])):
        d = _variant_diff(va, vb)
        if d is not None:
            variants.append(d)
    out['variants'] = variants
    out['active'] = None if ja.get('active') == jb.get('active') else {
        'a': _uid_names(ja).get(ja.get('active'), ja.get('active')),
        'b': _uid_names(jb).get(jb.get('active'), jb.get('active'))}
    lib = {}
    for k in ('modules', 'lenses', 'sensor_models'):
        A, B = ja.get(k, {}), jb.get(k, {})
        ch = {'added': sorted(set(B) - set(A)), 'removed': sorted(set(A) - set(B)),
              'changed': {i: leaf_changes(A[i], B[i]) for i in sorted(set(A) & set(B)) if A[i] != B[i]}}
        if ch['added'] or ch['removed'] or ch['changed']:
            lib[k] = ch
    out['library'] = lib
    an_a, an_b = ja.get('analysis', {}), jb.get('analysis', {})
    analysis = []
    for k in sorted(set(an_a) | set(an_b)):
        if k in ('assistant_log',):
            continue
        ch = leaf_changes(an_a.get(k, _MISSING), an_b.get(k, _MISSING), k)
        if ch:
            analysis.append({'slot': k, 'changes': ch})
    out['analysis'] = analysis
    la, lb = an_a.get('assistant_log', []) or [], an_b.get('assistant_log', []) or []
    ia = {e.get('id') for e in la if isinstance(e, dict)}
    ib = {e.get('id') for e in lb if isinstance(e, dict)}
    out['design_log'] = {'added': [e for e in lb if isinstance(e, dict) and e.get('id') not in ia],
                         'removed': [e for e in la if isinstance(e, dict) and e.get('id') not in ib]}
    veh = leaf_changes({k: v for k, v in ja.get('vehicle', {}).items()},
                       {k: v for k, v in jb.get('vehicle', {}).items()})
    ma, mb = arrays_digest(mesh_arrays(a.vehicle.mesh)), arrays_digest(mesh_arrays(b.vehicle.mesh))
    out['vehicle'] = {'changes': veh, 'mesh_changed': ma != mb,
                      'mesh': {'a': _mesh_info(a), 'b': _mesh_info(b)} if ma != mb else None}
    fa = {v['id']: v for v in _fleet_vehicles(ja.get('fleet'))}
    fb = {v['id']: v for v in _fleet_vehicles(jb.get('fleet'))}
    out['fleet'] = {
        'vehicles_added': [fb[k]['name'] for k in fb if k not in fa],
        'vehicles_removed': [fa[k]['name'] for k in fa if k not in fb],
        'records_added': sum(len({r.get('id') for r in fb[k]['records']} - {r.get('id') for r in fa[k]['records']})
                             for k in fb if k in fa) + sum(len(fb[k]['records']) for k in fb if k not in fa),
        'records_removed': sum(len({r.get('id') for r in fa[k]['records']} - {r.get('id') for r in fb[k]['records']})
                               for k in fa if k in fb) + sum(len(fa[k]['records']) for k in fa if k not in fb),
        'settings': leaf_changes((ja.get('fleet') or {}).get('settings', _MISSING),
                                 (jb.get('fleet') or {}).get('settings', _MISSING), 'settings')}
    sections = []
    for k in ('name', 'roi', 'targets', 'target_settings', 'obstacles', 'rig', 'ecu', 'photos', 'frame',
              'custom_origin', 'units', 'streams', 'hardware'):
        ch = leaf_changes(ja.get(k, _MISSING), jb.get(k, _MISSING), k)
        if ch:
            sections.append({'section': k, 'changes': ch})
    for k in sorted((set(ja) | set(jb)) - {'name', 'roi', 'targets', 'target_settings', 'obstacles', 'rig', 'ecu',
                                           'photos', 'frame', 'custom_origin', 'units', 'streams', 'hardware',
                                           'variants', 'active', 'modules', 'lenses', 'sensor_models', 'analysis',
                                           'vehicle', 'fleet', 'format', 'version'}):
        ch = leaf_changes(ja.get(k, _MISSING), jb.get(k, _MISSING), k)
        if ch:
            sections.append({'section': k, 'changes': ch})
    out['sections'] = sections
    s = {'variants_added': sum(v['status'] == 'added' for v in variants),
         'variants_removed': sum(v['status'] == 'removed' for v in variants),
         'variants_changed': sum(v['status'] == 'changed' for v in variants)}
    items = [i for v in variants if v['status'] == 'changed' for i in v['items']]
    s.update({'cameras_added': sum(i['kind'] == 'camera' and i['status'] == 'added' for i in items),
              'cameras_removed': sum(i['kind'] == 'camera' and i['status'] == 'removed' for i in items),
              'cameras_moved': sum(i['kind'] == 'camera' and bool(i.get('move')) for i in items),
              'sensors_added': sum(i['kind'] == 'sensor' and i['status'] == 'added' for i in items),
              'sensors_removed': sum(i['kind'] == 'sensor' and i['status'] == 'removed' for i in items),
              'sensors_moved': sum(i['kind'] == 'sensor' and bool(i.get('move')) for i in items),
              'lens_changes': sum(any(f['path'] in ('lens_id', 'module_id', 'model_id') for f in i.get('fields', []))
                                  for i in items),
              'settings_changed': len(analysis), 'library_changed': len(lib),
              'mesh_changed': out['vehicle']['mesh_changed']})
    s['changes'] = (s['variants_added'] + s['variants_removed'] + len(items) + len(analysis) + len(lib)
                    + len(sections) + len(veh) + int(s['mesh_changed']) + len(out['design_log']['added'])
                    + len(out['design_log']['removed']) + len(out['fleet']['vehicles_added'])
                    + len(out['fleet']['vehicles_removed']) + out['fleet']['records_added']
                    + out['fleet']['records_removed'] + len(out['fleet']['settings'])
                    + sum(len(v.get('fields', [])) for v in variants if v['status'] == 'changed'))
    out['summary'] = s
    out['same'] = s['changes'] == 0 and out['active'] is None
    return out


def _mesh_info(p) -> dict | None:
    m = p.vehicle.mesh
    return None if m is None else {'vertices': int(len(m.vertices)), 'faces': int(len(m.faces))}


def _fleet_vehicles(f) -> list:
    from . import fleet as FL
    return FL.vehicles(f if isinstance(f, dict) else {})


def _v(x) -> str:
    if isinstance(x, float):
        return f'{x:.6g}'
    if isinstance(x, (dict, list)):
        t = json.dumps(x, ensure_ascii=False, separators=(',', ':'))
        return t if len(t) <= 80 else t[:77] + '…'
    return json.dumps(x, ensure_ascii=False) if isinstance(x, str) else str(x)


def _change_text(c: dict) -> str:
    if c.get('added'):
        return f'{c["path"]} = {_v(c["b"])} (new)'
    if c.get('removed'):
        return f'{c["path"]} removed (was {_v(c["a"])})'
    return f'{c["path"]}: {_v(c["a"])} → {_v(c["b"])}'


def _move_text(m: dict) -> str:
    parts = []
    if m['moved_mm'] > MOVE_EPS_MM:
        parts.append(f'moved {m["moved_mm"]:.1f} mm (x {m["dx_mm"]:+.1f}, y {m["dy_mm"]:+.1f}, z {m["dz_mm"]:+.1f})')
    turns = [f'{k} {m[f"d{k}_deg"]:+.2f}°' for k in ('yaw', 'pitch', 'roll') if abs(m[f'd{k}_deg']) > TURN_EPS_DEG]
    if turns:
        parts.append(f'turned {m["rotation_deg"]:.2f}° ({", ".join(turns)})')
    return ', '.join(parts)


def diff_text(doc: dict, max_lines: int = 400) -> str:
    """The diff as plain text, one change per line."""
    L = [f'{doc["a"]["label"]} → {doc["b"]["label"]}']
    if doc['same']:
        L.append('No difference.')
        return '\n'.join(L) + '\n'
    s = doc['summary']
    L.append(f'{s["changes"]} change(s): {s["variants_changed"]} variant(s) changed, {s["variants_added"]} added, '
             f'{s["variants_removed"]} removed; {s["cameras_moved"]} camera(s) and {s["sensors_moved"]} sensor(s) '
             f'moved; {s["settings_changed"]} study setting(s) changed.')
    for v in doc['variants']:
        if v['status'] != 'changed':
            L.append(f'Variant "{v["name"]}" {v["status"]} ({v["cameras"]} cameras, {v["sensors"]} sensors)')
            continue
        L.append(f'Variant "{v["name"]}"' + (f' (renamed from "{v["was"]}")' if v.get('was') else '') + ':')
        for c in v['fields']:
            if c['path'] != 'name':
                L.append(f'  {_change_text(c)}')
        for i in v['items']:
            who = f'{i["kind"]} {i["tag"]}'
            if i['status'] != 'changed':
                p = i['pose']
                L.append(f'  {who} {i["status"]} at x {p.get("x")}, y {p.get("y")}, z {p.get("z")} m')
                continue
            bits = []
            if i.get('was'):
                bits.append(f'renamed from {i["was"]}')
            if i.get('move'):
                bits.append(_move_text(i['move']))
            for f in i['fields']:
                if f['path'] == 'tag':
                    continue
                bits.append(_change_text(f))
            L.append(f'  {who}: ' + '; '.join(bits))
    if doc.get('active'):
        L.append(f'Shown variant: {doc["active"]["a"]} → {doc["active"]["b"]}')
    for k, ch in doc['library'].items():
        L.append(f'Library {k}: ' + '; '.join(
            ([f'added {", ".join(ch["added"])}'] if ch['added'] else []) +
            ([f'removed {", ".join(ch["removed"])}'] if ch['removed'] else []) +
            [f'{i}: ' + ', '.join(_change_text(c) for c in cs[:6]) for i, cs in ch['changed'].items()]))
    for a in doc['analysis']:
        for c in a['changes']:
            L.append(f'Analysis {_change_text(c)}')
    for e in doc['design_log']['added']:
        L.append(f'Design log + {e.get("kind", "")}: {str(e.get("text", ""))[:100]}')
    for e in doc['design_log']['removed']:
        L.append(f'Design log − {e.get("kind", "")}: {str(e.get("text", ""))[:100]}')
    veh = doc['vehicle']
    for c in veh['changes']:
        L.append(f'Vehicle {_change_text(c)}')
    if veh['mesh_changed']:
        mi = veh['mesh']
        L.append('Vehicle body changed' + (f' ({mi["a"]["faces"] if mi["a"] else 0} → '
                                           f'{mi["b"]["faces"] if mi["b"] else 0} faces)' if mi else ''))
    f = doc['fleet']
    for n in f['vehicles_added']:
        L.append(f'Fleet: vehicle {n} added')
    for n in f['vehicles_removed']:
        L.append(f'Fleet: vehicle {n} removed')
    if f['records_added'] or f['records_removed']:
        L.append(f'Fleet: {f["records_added"]} record(s) added, {f["records_removed"]} removed')
    for c in f['settings']:
        L.append(f'Fleet {_change_text(c)}')
    for sct in doc['sections']:
        for c in sct['changes']:
            L.append(_change_text(c))
    if len(L) > max_lines:
        L = L[:max_lines] + [f'… and {len(L) - max_lines} more lines (the JSON has them all)']
    return '\n'.join(L) + '\n'


# =========================================================================== validation
def validate_round_trip(project, work: str, text_mesh: bool = False) -> dict:
    """Zip -> folder -> zip on project (in work): the three loads' content compared. {'equal', 'steps': [...]}."""
    from .project import Project
    z1 = os.path.join(work, 'a.vantage')
    d1 = os.path.join(work, 'a.vantage.d')
    z2 = os.path.join(work, 'b.vantage')
    project.save(z1)
    p1 = Project.load(z1)
    save_dir(p1, d1, text_mesh=text_mesh)
    p2 = Project.load(d1)
    p2.save(z2)
    p3 = Project.load(z2)
    c12, c13 = content_equal(p1, p2), content_equal(p1, p3)
    again = files_of(p2, text_mesh=text_mesh)
    stable = all(again[r] == open(os.path.join(d1, *r.split('/')), 'rb').read() for r in again
                 if r not in ('.gitattributes', '.gitignore'))
    return {'equal': c12['equal'] and c13['equal'] and stable, 'zip_to_dir': c12, 'dir_to_zip': c13,
            'stable_bytes': stable, 'digest': c12['digest_a']}


def validation_checks() -> list:
    """Checks for vantage/validation (Spec-compatible dicts; fn(ctx) -> suite.Outcome): the folder round trip, the
    semantic diff of known edits and the three-way merge of two edits."""
    def _outcome(error, detail, note=''):
        from ..validation.suite import Outcome
        return Outcome(error=float(error), detail=detail, note=note)

    def round_trip(ctx):
        p = ctx.project
        w1, w2 = os.path.join(ctx.tmp(), 'project_dir_npz'), os.path.join(ctx.tmp(), 'project_dir_obj')
        os.makedirs(w1, exist_ok=True)
        os.makedirs(w2, exist_ok=True)
        r = validate_round_trip(p, w1)
        r2 = validate_round_trip(p, w2, text_mesh=True) if p.vehicle.mesh is not None \
            and len(p.vehicle.mesh.vertices) <= TEXT_MESH_MAX_VERTICES else {'equal': True}
        bad = int(not r['equal']) + int(not r2['equal'])
        return _outcome(bad, {'npz': {k: r[k] for k in ('equal', 'stable_bytes', 'digest')},
                              'obj': {'equal': r2['equal']}, 'differs': r['zip_to_dir']['differs']})

    def known_edits(ctx):
        from .project import Project
        p = ctx.sample()
        q = Project.from_data(_plain(p.to_json()), mesh_arrays(p.vehicle.mesh), b'', '')
        v = next(x for x in q.variants if x.sensors)
        cam = v.cameras[0]
        cam.x += 0.050
        cam.z += 0.014
        cam.yaw += 2.0
        v.cameras[1].lens_id = next(k for k in q.lenses if k != v.cameras[1].lens_id)
        s = v.sensors[0]
        s.pitch += 1.5
        q.analysis.tol_mm = 42.0
        d = diff(p, q)
        it = {i['tag']: i for vv in d['variants'] if vv['status'] == 'changed' for i in vv['items']}
        mv = it[cam.tag]['move']
        exp_move = math.hypot(50.0, 14.0)
        err = max(abs(mv['moved_mm'] - exp_move), abs(mv['dyaw_deg'] - 2.0), abs(mv['rotation_deg'] - 2.0),
                  abs(it[s.tag]['move']['rotation_deg'] - 1.5))
        lens_ok = any(f['path'] == 'lens_id' for f in it[v.cameras[1].tag]['fields'])
        tol_ok = any(c['path'] == 'tol_mm' for a in d['analysis'] for c in a['changes'])
        counts_ok = d['summary']['cameras_moved'] == 1 and d['summary']['sensors_moved'] == 1
        from ..validation.suite import Outcome
        return Outcome(error=err, detail={'moved_mm': mv['moved_mm'], 'expected_mm': exp_move,
                                          'rotation_deg': mv['rotation_deg'], 'summary': d['summary']},
                       ok=lens_ok and tol_ok and counts_ok)

    def merge_two_edits(ctx):
        p = ctx.sample()
        base = _plain(p.to_json())
        ours, theirs = copy.deepcopy(base), copy.deepcopy(base)
        ours['variants'][0]['cameras'][0]['x'] += 0.1
        theirs['variants'][0]['cameras'][1]['yaw'] += 3.0
        theirs['variants'][-1]['name'] = 'renamed'
        conflicts = []
        m = merge3(base, ours, theirs, '', conflicts)
        expect = copy.deepcopy(ours)
        expect['variants'][0]['cameras'][1]['yaw'] += 3.0
        expect['variants'][-1]['name'] = 'renamed'
        wrong = len(leaf_changes(m, expect))
        c2 = []
        o2, t2 = copy.deepcopy(base), copy.deepcopy(base)
        o2['variants'][0]['cameras'][0]['x'] = 1.0
        t2['variants'][0]['cameras'][0]['x'] = 2.0
        merge3(base, o2, t2, '', c2)
        return _outcome(wrong + abs(len(conflicts)) + abs(len(c2) - 1),
                        {'wrong_leaves': wrong, 'conflicts_clean_case': len(conflicts), 'conflicts_both_ways': len(c2)})

    common = {'model': 'formats', 'refs': (), 'cost': 'cheap'}
    return [
        dict(common, id='formats.project_dir_round_trip', title='Project folder round trip (zip → folder → zip)',
             quantity='projects whose content differs after the round trip (project.json, mesh arrays, content hash)',
             reference='the project as loaded from its .vantage file (identity: nothing may change)', tolerance=0,
             unit='projects', fn=round_trip, cost='study'),
        dict(common, id='formats.project_diff_known_edits', title='Semantic diff of known edits',
             quantity='largest error of the reported move (mm) and turn (°) against the edits made',
             reference='the edits applied: a camera moved 50 mm forward and 14 mm up and yawed 2°, a lidar pitched '
                       '1.5°, a lens swapped, a tolerance changed', tolerance=1e-6, unit='mm or °', fn=known_edits),
        dict(common, id='formats.project_merge_two_edits', title='Three-way merge of two edits',
             quantity='leaves that differ from the hand-merged result plus conflict-count errors',
             reference='the two edits applied by hand (different cameras: no conflict; the same value two ways: one '
                       'conflict)', tolerance=0, unit='count', fn=merge_two_edits),
    ]


# =========================================================================== command line
def main(argv=None) -> int:
    """vantage diff A B | merge-file O A B [P] | convert SRC DST [--text-mesh] | git-setup FOLDER."""
    ap = argparse.ArgumentParser(prog='vantage-project', description='Vantage project folders: diff, merge, convert')
    sub = ap.add_subparsers(dest='cmd', required=True)
    d = sub.add_parser('diff', help='semantic diff of two projects (files, folders, or path@REV)')
    d.add_argument('a')
    d.add_argument('b', nargs='?', help='default: the working copy of A (when A is path@REV)')
    d.add_argument('--json', metavar='OUT', help='also write the diff as JSON (- for stdout instead of the text)')
    d.add_argument('--git-setup', action='store_true', help='register the merge driver in A\'s git repository')
    m = sub.add_parser('merge-file', help='git merge driver: %%O %%A %%B %%P')
    m.add_argument('base')
    m.add_argument('ours')
    m.add_argument('theirs')
    m.add_argument('name', nargs='?', default='')
    c = sub.add_parser('convert', help='a .vantage file to a folder or back')
    c.add_argument('src')
    c.add_argument('dst')
    c.add_argument('--text-mesh', action='store_true', help='the mesh as text (OBJ) when it is small')
    g = sub.add_parser('git-setup', help='register the merge driver in the folder\'s git repository')
    g.add_argument('folder')
    args = ap.parse_args(argv)
    try:
        if args.cmd == 'merge-file':
            return merge_file(args.base, args.ours, args.theirs, args.name)
        if args.cmd == 'convert':
            print(convert(args.src, args.dst, text_mesh=args.text_mesh or None))
            return 0
        if args.cmd == 'git-setup':
            print(json.dumps(git_setup(args.folder), indent=1))
            return 0
        if args.git_setup:
            print(json.dumps(git_setup(args.a.rsplit('@', 1)[0] if not os.path.exists(args.a) else args.a), indent=1))
            return 0
        pa, la = load_spec(args.a)
        if args.b:
            pb, lb = load_spec(args.b)
        else:
            path = args.a.rsplit('@', 1)[0]
            pb, lb = load_spec(path)
            lb = f'{lb} (working copy)'
        doc = diff(pa, pb, la, lb)
    except (ProjectDirError, OSError, ValueError, KeyError, zipfile.BadZipFile) as e:
        print(f'vantage diff: {e}', file=sys.stderr)
        return EXIT_ERROR
    if args.json == '-':
        sys.stdout.write(json.dumps(doc, indent=1, ensure_ascii=False) + '\n')
    else:
        sys.stdout.write(diff_text(doc))
        if args.json:
            with open(args.json, 'w', encoding='utf-8') as fh:
                json.dump(doc, fh, indent=1, ensure_ascii=False)
    return EXIT_SAME if doc['same'] else EXIT_DIFFERENT


if __name__ == '__main__':
    sys.exit(main())
