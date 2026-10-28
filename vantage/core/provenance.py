"""Provenance: every number Vantage writes traceable to its inputs, its method and its version, and reproducible.

Every JSON object the exporters write (core.exporters) and every batch output (vantage.batch) carries a block under
"provenance" (schema vantage.provenance/1), added after everything else in the document so the rest stays as it was:

- vantage: the version; code_sha256, a SHA-256 of the package's Python sources (the code that computed the numbers,
  the same from a git checkout, a pip install or an unpacked source tree; None in the bundled app, which ships no
  sources); and, when run from a git checkout, the commit, `git describe` and whether the app code (vantage/) differs
  from that commit (dirty).
- runtime: Python, NumPy, SciPy, OpenCV, Qt and PyQt versions, the BLAS NumPy was built against, the platform, the
  processor count and the BLAS thread settings. Never a path, a user name or a host name.
- project: its name, the project file's name (never its folder), sha256 (the content hash below), one shorter hash per
  section (to say what differs), and in batch mode input_sha256 when the command changed the project before the study
  ran (--import-calib, --calibrate-lens, --suspension-data, --condition, --fleet-import).
- variant: name, 1-based index, uid and its own content hash.
- study, output (the option or export format and the file's name inside its folder), settings (the study's own
  settings with every default filled in, plus the base analysis settings and the region of interest), seeds (every
  random seed in those settings), command (batch: the command line with every path reduced to its file name; the
  user's own --on-change command is never recorded), inputs (batch: each input file or folder by name, size and
  SHA-256), created_utc, and result_sha256.

Content hash (project.sha256): SHA-256 of the canonical JSON of the project as Vantage holds it in memory: keys
sorted, no spaces, UTF-8, integral numbers written as integers (so 2 and 2.0 hash alike), every uid replaced by the
place its object has in the project ('variants[4].cameras[2]': uids are random identities, so a sample or template
built twice hashes the same), the vehicle mesh as its arrays (dtype, shape and bytes), stream URLs without passwords.
Left out as volatile or machine-specific: saved_at, created times, the log, the active variant (recorded on its own),
the view toggles (analysis.live, analysis.heatmap), the assistant's brief, the automation settings, where the vehicle
model and the photos came from (vehicle.source, photos.source: paths of this machine) and the photo shown.

Result hash (result_sha256): SHA-256 of the canonical JSON of the document without its provenance block, with every
date-time stamp and every computation time (the numbers under a key named 'seconds' or ending in '_seconds')
blanked. Two runs that compute the
same numbers have the same result hash; verify(doc) checks that a file still holds what Vantage wrote.

Reproduce (vantage PROJECT --reproduce RESULT.json, in vantage.batch): re-runs what produced a result and compares
every number (exactly, or within a tolerance per quantity: compare()), and says why it does not reproduce (a
different version, code, library, project section, variant or input).

Determinism: every random number generator in vantage/ is numpy's PCG64 (np.random.default_rng) built from a seed (a
study setting, recorded here, or a constant in the code, covered by code_sha256); OpenCV's RNG is seeded before the
one OpenCV call that uses it (install.find_sb). tests/test_provenance.py audits the sources for unseeded use.
"""
from __future__ import annotations

import datetime as _dt
import functools
import hashlib
import importlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import tempfile
import threading

import numpy as np

SCHEMA = 'vantage.provenance/1'
REPRODUCE_SCHEMA = 'vantage.reproduce/1'
KEY = 'provenance'
EXIT_NOT_REPRODUCED = 6             # batch --reproduce: the numbers do not reproduce (4: it cannot be tried)
PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # .../vantage
CHECKOUT_DIR = os.path.dirname(PACKAGE_DIR)
INPUT_HASH_LIMIT = 1 << 30          # bytes of an input hashed in full; beyond, the head and tail of each big file
SAMPLE_BYTES = 16 << 20
RESULT_HASH_TEXT = ('SHA-256 of the canonical JSON of this document without its provenance block (keys sorted, no '
                    'spaces, UTF-8, integral numbers as integers), date-time stamps and computation times (the '
                    'numbers under keys named "seconds" or ending in "_seconds") blanked')
CONTENT_HASH_TEXT = ('SHA-256 of the canonical JSON of the project in memory: uids replaced by their objects\' places, '
                     'the mesh arrays by dtype, shape and bytes; times, the log, the active variant, view toggles, '
                     'the assistant brief, automation settings and source paths left out')
ASSUMPTIONS = (
    'Bit-for-bit reproduction needs the same code (code_sha256), the same library versions and the same kind of '
    'processor and BLAS: across machines floating-point results can differ in their last digits, so give '
    '--tolerance there. The content hash describes the project in memory: an imported or generated vehicle body is '
    'stored with 32-bit vertices, so a project\'s hash (and its figures, slightly) can change once at its first save. '
    'Inputs larger than 1 GiB are hashed from the first and last 16 MiB of each big file. Live measurements (rig '
    'health) cannot be reproduced. Time stamps and computation times are not compared; uids (random identities) are '
    'compared by their order of appearance. An export made in the app is re-made by the batch exporter.')

_TS = re.compile(r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?')
_UID = re.compile(r'(?<![0-9a-f])[0-9a-f]{10}(?![0-9a-f])')
# an absolute path inside a string (not a URL): its folders are dropped, its last part kept
_ABS = re.compile(r'(?<![\w:/.\\-])(?:[A-Za-z]:[\\/]|~[\\/]|/)(?:[^\s"\'<>|*?\\/]+[\\/])+([^\s"\'<>|*?\\/]*)')
_PATH_KEYS = ('path', 'source', 'file', 'folder', 'dir', 'out_dir', 'project_file')
_UI_ONLY = ('live', 'heatmap', 'assistant', 'automation')
_BASE_ANALYSIS = ('cell_m', 'min_density', 'target_density', 'obstacle_mode', 'tol_deg', 'tol_mm', 'load_cases')

# project.analysis key -> (core module, settings accessor): a function of the project or a settings class
_SETTINGS = {
    'volume': ('volume', 'VolumeSettings'), 'perception': ('perception', 'PerceptionSettings'),
    'sensor_coverage': ('sensor_coverage', 'CoverageSettings'), 'timing': ('timing', 'TimingSettings'),
    'recording': ('recording', 'RecordingSettings'), 'gnss': ('gnss', 'GnssSettings'),
    'power': ('power', 'PowerSettings'), 'exposure': ('exposure', 'ExposureSettings'),
    'brackets': ('brackets', 'BracketSettings'), 'dataset': ('dataset', 'DatasetSettings'),
    'lidar_calib': ('lidar_calib', 'LidarCalibSettings'), 'drivelog': ('drivelog', 'DriveLogSettings'),
    'requirements': ('requirements', 'settings_of'), 'housings': ('housings', 'settings_of'),
    'asbuilt': ('asbuilt', 'settings_of'), 'suspension': ('suspension', 'settings_for'),
    'critical': ('critical', 'settings_of'), 'terrain': ('terrain', 'settings_of'),
    'observability': ('observability', 'settings_of'), 'materials': ('materials', 'settings_of'),
    'weather': ('weather', 'settings_of'), 'harness': ('harness', 'settings_of'), 'fleet': ('fleet', 'settings_of'),
    'guidelines': ('guidelines', 'GuidelineSettings'), 'stitched': ('stitched_objects', 'Settings'),
    'bowl': ('bowl', 'BowlShape'), 'experiments': ('experiments', 'settings_of'), 'sim': ('simrig', 'settings_of'),
    'detection': ('detection_eval', 'settings_of'), 'campaign': ('simcampaign', 'settings_of'),
    'critical_carla': ('critical_carla', 'settings_of'),
}
# study (a schema's name or an export format id) -> the project.analysis settings it reads besides the base ones
STUDY_SETTINGS = {
    'analysis': (), 'load_cases': (), 'blind_volume': ('volume',), 'perception_range': ('perception', 'weather'),
    'sensor_coverage': ('sensor_coverage', 'materials', 'weather'), 'sync_plan': ('timing',),
    'recording_budget': ('recording',), 'power_budget': ('power',), 'gnss_imu': ('gnss',), 'exposure': ('exposure',),
    'housings': ('housings',), 'suspension': ('suspension',), 'suspension_load_cases': ('suspension',),
    'observability': ('observability',), 'harness': ('harness',), 'terrain': ('terrain', 'suspension'),
    'critical_targets': ('critical', 'perception'), 'materials': ('materials',),
    'weather': ('weather', 'perception', 'sensor_coverage'), 'requirements': ('requirements',), 'fleet': ('fleet',),
    'brackets': ('brackets',), 'dataset': ('dataset', 'weather'), 'lidar_camera_calibration': ('lidar_calib',),
    'drive_log_check': ('drivelog',), 'asbuilt': ('asbuilt', 'housings'), 'stitched_objects': ('stitched',),
    'guidelines': ('guidelines',), 'bowl_mesh': ('bowl',), 'bowl': ('bowl',), 'reproduce': (),
    'experiment': ('experiments',), 'experiments': ('experiments',), 'experiment_spec': ('experiments',),
    'sim_rig': ('sim', 'weather'), 'sim_run': ('sim', 'weather'), 'validation': (),
    'detection_quality': ('detection',), 'sim_campaign': ('campaign', 'sim', 'weather', 'detection'),
    'critical_carla': ('critical_carla', 'critical', 'perception', 'sim'), 'ground_truth_summary': (),
    'sim_live': ('sim',),
}
# a batch output's schema name -> the option that writes it
SCHEMA_OPTION = {
    'analysis': 'analyse', 'load_cases': 'load_cases', 'blind_volume': 'blind_volume',
    'perception_range': 'perception_range', 'sensor_coverage': 'sensor_coverage', 'sync_plan': 'sync_plan',
    'recording_budget': 'recording_budget', 'power_budget': 'power_budget', 'gnss_imu': 'gnss',
    'exposure': 'exposure', 'housings': 'housings', 'suspension': 'suspension', 'observability': 'observability',
    'harness': 'harness', 'terrain': 'terrain', 'critical_targets': 'critical_targets', 'materials': 'materials',
    'weather': 'weather', 'install_check': 'install_check', 'rig_health': 'rig_health',
    'drive_log_check': 'drive_log', 'asbuilt': 'as_built_check', 'requirements': 'check', 'fleet': 'fleet',
    'fleet_import': 'fleet_import', 'lens_calibration': 'calibrate_lens', 'calibration_import': 'import_calib',
    'batch_calibration': 'calibrate_lens', 'brackets': 'brackets', 'lidar_camera_calibration': 'lidar_calib',
    'dataset': 'dataset', 'reproduce': 'reproduce', 'experiments': 'experiment', 'experiment_spec': 'experiment',
    'sim_rig': 'carla_rig', 'sim_run': 'carla_run', 'validation': 'validate',
    'detection_quality': 'detection_quality', 'sim_campaign': 'campaign', 'critical_carla': 'critical_carla',
    'ground_truth_summary': 'ground_truth_summary', 'sim_live': 'carla_live',
}


# ====================================================================== canonical JSON and hashes
def _jsonable(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, frozenset)):
        return sorted(o, key=str)
    if isinstance(o, bytes):
        return o.hex()
    return str(o)


def plain(o):
    """o as a JSON reader sees it: numpy scalars and arrays as numbers and lists, tuples as lists, keys as strings."""
    return json.loads(json.dumps(o, default=_jsonable, ensure_ascii=False, allow_nan=True))


def _canon(o):
    if o is None or isinstance(o, (bool, str)):
        return o
    if isinstance(o, float):
        if math.isfinite(o) and o.is_integer() and abs(o) < 2 ** 53:
            return int(o)              # 2.0 and 2 are the same number (and -0.0 is 0)
        return o
    if isinstance(o, int):
        return int(o)
    if isinstance(o, dict):
        return {str(k): _canon(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_canon(v) for v in o]
    return _canon(plain(o))


def canonical(o) -> bytes:
    """The canonical JSON of o: keys sorted, no spaces, UTF-8, integral numbers as integers, NaN as NaN."""
    return json.dumps(_canon(plain(o)), sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=True).encode('utf-8')


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def neutral(doc):
    """doc without its provenance block, every date-time stamp and every computation time blanked: the numbers under
    a key named 'seconds' or ending in '_seconds' (scan_seconds, a dict of study_seconds …)."""
    def blank(v):
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return None
        if isinstance(v, dict):
            return {k: blank(x) for k, x in v.items()}
        if isinstance(v, list):
            return [blank(x) for x in v]
        return v

    def walk(o):
        if isinstance(o, dict):
            return {k: (blank(v) if isinstance(k, str) and (k == 'seconds' or k.endswith('_seconds')) else walk(v))
                    for k, v in o.items()}
        if isinstance(o, list):
            return [walk(v) for v in o]
        if isinstance(o, str):
            return _TS.sub('<time>', o)
        return o
    o = plain(doc)
    if isinstance(o, dict):
        o.pop(KEY, None)
    return walk(o)


def result_digest(doc) -> str:
    """The result hash of a document (RESULT_HASH_TEXT)."""
    return sha256(canonical(neutral(doc)))


def verify(doc) -> dict:
    """Does the document still hold what Vantage wrote? {'ok': True/False/None (no provenance), 'recorded', 'actual'}."""
    prov = doc.get(KEY) if isinstance(doc, dict) else None
    rec = prov.get('result_sha256') if isinstance(prov, dict) else None
    actual = result_digest(doc) if isinstance(doc, dict) else None
    return {'ok': None if rec is None else rec == actual, 'recorded': rec, 'actual': actual}


# ====================================================================== the project's content hash
def _uid_places(o, path='', out=None) -> dict:
    """uid -> the place of the object that carries it, in document order ('variants[4].cameras[2]')."""
    out = {} if out is None else out
    if isinstance(o, dict):
        u = o.get('uid')
        if isinstance(u, str) and u and u not in out:
            out[u] = path or '.'
        for k, v in o.items():
            _uid_places(v, f'{path}.{k}' if path else str(k), out)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            _uid_places(v, f'{path}[{i}]', out)
    return out


def _renumber(o, places: dict):
    """o with every known uid (alone, in 'a|b' seam keys or inside other text) replaced by its place."""
    def s(t):
        if t in places:
            return places[t]
        return _UID.sub(lambda m: places.get(m.group(0), m.group(0)), t) if len(t) >= 10 else t
    if isinstance(o, dict):
        return {s(k): _renumber(v, places) for k, v in o.items()}
    if isinstance(o, list):
        return [_renumber(v, places) for v in o]
    if isinstance(o, str):
        return s(o)
    return o


_MESH_CACHE: list = []              # [(arrays kept alive, digest)] of the last few meshes hashed
_MESH_LOCK = threading.Lock()


def mesh_digest(mesh) -> str | None:
    """SHA-256 of the mesh arrays the analysis uses (vertices, faces, face_part, face_zone) by dtype, shape and bytes."""
    if mesh is None:
        return None
    arrays = tuple(getattr(mesh, k, None) for k in ('vertices', 'faces', 'face_part', 'face_zone'))
    with _MESH_LOCK:
        for held, dig in _MESH_CACHE:
            if all(a is b for a, b in zip(held, arrays)):
                return dig
    h = hashlib.sha256()
    for name, a in zip(('vertices', 'faces', 'face_part', 'face_zone'), arrays):
        if a is None:
            h.update(f'{name}:none\n'.encode())
            continue
        a = np.ascontiguousarray(a)
        a = a.astype(a.dtype.newbyteorder('<'), copy=False)
        h.update(f'{name}:{a.dtype.str}:{a.shape}\n'.encode())
        h.update(a.tobytes())
    dig = h.hexdigest()
    with _MESH_LOCK:
        _MESH_CACHE.insert(0, (arrays, dig))
        del _MESH_CACHE[4:]
    return dig


def _strip_password(url) -> str:
    try:
        from .credentials import split
        return split(str(url))[0]
    except Exception:
        return re.sub(r'(://[^/:@\s]+):[^@/\s]*@', r'\1@', str(url))


def project_content(project) -> dict:
    """The project's content by section, as hashed (uids not yet replaced)."""
    veh = project.vehicle
    meta = veh.meta()
    return {
        'name': project.name,
        'vehicle': {k: v for k, v in meta.items() if k not in ('face_zone', 'source')},
        'mesh': mesh_digest(veh.mesh),
        'library': {'modules': {k: m.to_dict() for k, m in project.modules.items()},
                    'lenses': {k: m.to_dict() for k, m in project.lenses.items()},
                    'sensor_models': {k: m.to_dict() for k, m in project.sensor_models.items()}},
        'variants': [{k: v for k, v in var.to_dict().items() if k != 'created'} for var in project.variants],
        'roi': project.roi.to_dict(),
        'analysis': {k: v for k, v in project.analysis.to_dict().items() if k not in _UI_ONLY},
        'targets': {'targets': [t.to_dict() for t in project.targets],
                    'settings': project.target_settings.to_dict()},
        'obstacles': [o.to_dict() for o in project.obstacles],
        'rig': [r.to_dict() for r in project.rig],
        'ecu': project.ecu.to_dict(),
        'photos': {k: v for k, v in project.photos.to_dict().items() if k not in ('source', 'current')},
        'frame': {'frame': project.frame, 'custom_origin': list(project.custom_origin), 'units': project.units},
        'streams': {str(k): _strip_password(v) for k, v in (project.streams or {}).items()},
        'fleet': project.fleet or {},
    }


def project_digest(project) -> dict:
    """{'sha256': the content hash, 'sections': {section: its hash's first 16 hex digits}}."""
    content = plain(project_content(project))
    content = _renumber(content, _uid_places(content))
    return {'sha256': sha256(canonical(content)),
            'sections': {k: sha256(canonical(v))[:16] for k, v in content.items()}}


def variant_digest(variant) -> str:
    """The variant's own content hash: its uids by their place inside the variant, its creation time left out."""
    d = plain({k: v for k, v in variant.to_dict().items() if k != 'created'})
    places = {u: 'variant' + (p if p == '.' else '.' + p) for u, p in _uid_places(d).items()}
    return sha256(canonical(_renumber(d, places)))


# ====================================================================== the environment
def _git_env() -> dict:
    try:
        from ..sysenv import system_env
        env = system_env()
    except Exception:
        env = dict(os.environ)
    env = {k: v for k, v in env.items() if not k.startswith('GIT_')}
    env['GIT_OPTIONAL_LOCKS'] = '0'                 # never write the index (a read-only look)
    env['LC_ALL'] = 'C'
    return env


def _git(*args) -> str | None:
    try:
        r = subprocess.run(['git', '-C', CHECKOUT_DIR, *args], capture_output=True, text=True, timeout=5,
                           env=_git_env(), creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    return r.stdout if r.returncode == 0 else None


def _head_from_files() -> str | None:
    """The checked-out commit read from .git without git (HEAD, a loose ref or packed-refs)."""
    gd = os.path.join(CHECKOUT_DIR, '.git')
    try:
        if os.path.isfile(gd):                      # a worktree: 'gitdir: <path>'
            with open(gd, encoding='utf-8') as fh:
                gd = os.path.join(CHECKOUT_DIR, fh.read().split(':', 1)[1].strip())
        with open(os.path.join(gd, 'HEAD'), encoding='utf-8') as fh:
            head = fh.read().strip()
        if not head.startswith('ref:'):
            return head if re.fullmatch(r'[0-9a-f]{40,64}', head) else None
        ref = head[4:].strip()
        common = gd
        if os.path.isfile(os.path.join(gd, 'commondir')):
            with open(os.path.join(gd, 'commondir'), encoding='utf-8') as fh:
                common = os.path.join(gd, fh.read().strip())
        for base in (gd, common):
            p = os.path.join(base, *ref.split('/'))
            if os.path.isfile(p):
                with open(p, encoding='utf-8') as fh:
                    return fh.read().strip() or None
            pr = os.path.join(base, 'packed-refs')
            if os.path.isfile(pr):
                with open(pr, encoding='utf-8') as fh:
                    for line in fh:
                        parts = line.split()
                        if len(parts) == 2 and parts[1] == ref:
                            return parts[0]
    except (OSError, IndexError, ValueError):
        return None
    return None


@functools.lru_cache(maxsize=1)
def git_state() -> dict | None:
    """{'commit', 'describe', 'dirty'} when Vantage runs from a git checkout (dirty: the app code, vantage/, differs
    from the commit; None when git is not installed), else None."""
    if not os.path.exists(os.path.join(CHECKOUT_DIR, '.git')):
        return None
    out = _git('rev-parse', 'HEAD')
    commit = out.strip() if out else _head_from_files()
    if not commit:
        return None
    desc = _git('describe', '--tags', '--always', '--abbrev=10')
    st = _git('status', '--porcelain', '--untracked-files=normal', '--', os.path.basename(PACKAGE_DIR))
    return {'commit': commit, 'describe': desc.strip() if desc else None,
            'dirty': None if st is None else bool(st.strip())}


@functools.lru_cache(maxsize=1)
def code_digest() -> str | None:
    """SHA-256 over every .py file of the package (relative path and bytes, in path order); None without sources."""
    h, n = hashlib.sha256(), 0
    for root, dirs, files in os.walk(PACKAGE_DIR):
        dirs[:] = sorted(d for d in dirs if d != '__pycache__' and not d.startswith('.'))
        for f in sorted(files):
            if not f.endswith('.py'):
                continue
            p = os.path.join(root, f)
            rel = os.path.relpath(p, PACKAGE_DIR).replace(os.sep, '/')
            try:
                with open(p, 'rb') as fh:
                    data = fh.read()
            except OSError:
                continue
            h.update(f'{rel}\0{len(data)}\0'.encode())
            h.update(data)
            n += 1
    return h.hexdigest() if n else None


def _version_of(mod: str, attr: str = '__version__') -> str | None:
    try:
        return str(getattr(importlib.import_module(mod), attr))
    except Exception:
        return None


def _blas() -> str | None:
    try:
        cfg = np.show_config(mode='dicts') or {}
        b = (cfg.get('Build Dependencies') or {}).get('blas') or {}
        name, ver = b.get('name'), b.get('version')
        return ' '.join(str(x) for x in (name, ver) if x) or None
    except Exception:
        return None


@functools.lru_cache(maxsize=1)
def environment() -> dict:
    """{'vantage': version, code hash, git; 'runtime': library versions, platform, threads} (no paths)."""
    from .. import __version__
    qt = pyqt = None
    try:
        from PyQt6.QtCore import PYQT_VERSION_STR, qVersion
        qt, pyqt = qVersion(), PYQT_VERSION_STR
    except Exception:
        pass
    threads = {k: os.environ[k] for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')
               if os.environ.get(k)}
    return {
        'vantage': {'version': __version__, 'code_sha256': code_digest(), 'git': git_state(),
                    'frozen': bool(getattr(sys, 'frozen', False))},
        'runtime': {'python': platform.python_version(), 'implementation': platform.python_implementation(),
                    'numpy': np.__version__, 'scipy': _version_of('scipy'), 'opencv': _version_of('cv2'),
                    'qt': qt, 'pyqt': pyqt, 'blas': _blas(),
                    'platform': platform.platform(), 'machine': platform.machine(),
                    'cpu_count': os.cpu_count(), 'threads': threads},
    }


# ====================================================================== paths out
def reduce_path(s):
    """A path's last part (its file or folder name); '-' and empty stay."""
    if not isinstance(s, str) or s in ('', '-'):
        return s
    t = re.split(r'[\\/]+', s.rstrip('\\/'))
    return t[-1] if t and t[-1] else s


def _needles(extra=()) -> list:
    """Folders of this machine that must never appear: home, working, temporary, Python and Vantage folders (only
    those two or more levels deep, so '/usr' or '/tmp' alone never match ordinary text)."""
    out = set()
    for p in (os.path.expanduser('~'), _cwd(), tempfile.gettempdir(), sys.prefix, sys.base_prefix, sys.exec_prefix,
              os.path.dirname(sys.executable or ''), CHECKOUT_DIR, PACKAGE_DIR, *extra):
        if p and isinstance(p, str):
            p = os.path.normpath(p)
            if len([x for x in re.split(r'[\\/]+', p) if x]) >= 2:
                out.add(p)
                out.add(p.replace('\\', '/'))
    return sorted(out, key=len, reverse=True)


def _cwd() -> str:
    try:
        return os.getcwd()
    except OSError:
        return ''


def scrub(o, extra=()):
    """o with every path of this machine reduced to its last part: values under path-like keys, absolute paths inside
    text, and any remaining mention of the home, working, temporary, Python or Vantage folders."""
    needles = _needles(extra)

    def text(t: str, pathish=False) -> str:
        if pathish and ('/' in t or '\\' in t) and '://' not in t:
            t = reduce_path(t)
        t = _ABS.sub(lambda m: m.group(1) or '…', t)
        for n in needles:
            if n in t:
                t = re.sub(re.escape(n) + r'([\\/][^\s"\'<>|]*)?(?=$|[\s"\'<>|,;)])',
                           lambda m: reduce_path(m.group(1) or '') or '…', t)
        return t

    def walk(v, key=''):
        if isinstance(v, dict):
            return {k: walk(x, str(k)) for k, x in v.items()}
        if isinstance(v, list):
            return [walk(x, key) for x in v]
        if isinstance(v, str):
            return text(v, key.lower() in _PATH_KEYS or key.lower().endswith(('_path', '_file', '_dir', '_folder')))
        return v
    return walk(o)


# ====================================================================== settings and seeds
def effective_settings(project, key: str) -> dict:
    """project.analysis.<key> with every default filled in (the study's settings class), else the stored dict."""
    raw = getattr(project.analysis, key, None)
    spec = _SETTINGS.get(key)
    if spec is not None:
        try:
            mod = importlib.import_module(f'{__package__}.{spec[0]}')
            acc = getattr(mod, spec[1])
            st = acc(project) if spec[1].startswith('settings_') else acc.from_dict(raw or {})
            return plain(st.to_dict())
        except Exception:
            pass
    return plain(raw) if isinstance(raw, dict) else {}


def study_settings(project, study: str, export: dict | None = None) -> dict:
    a = project.analysis
    out = {'analysis': plain({k: getattr(a, k, None) for k in _BASE_ANALYSIS}), 'roi': plain(project.roi.to_dict())}
    for k in STUDY_SETTINGS.get(study, ()):
        out[k] = effective_settings(project, k)
    if export:
        out['export'] = dict(export)
    return out


def seeds_of(settings: dict) -> dict:
    """Every random seed in the settings: {'observability.seed': 1, …}."""
    out = {}

    def walk(o, path):
        if isinstance(o, dict):
            for k, v in o.items():
                p = f'{path}.{k}' if path else str(k)
                if 'seed' in str(k).lower() and isinstance(v, (int, float)) and not isinstance(v, bool):
                    out[p] = v
                else:
                    walk(v, p)
        elif isinstance(o, list):
            for i, v in enumerate(o):
                walk(v, f'{path}[{i}]')
    walk(settings, '')
    return out


# ====================================================================== batch runs
_BATCH: list = []                   # the batch runs in progress (a reproduction runs one inside another)


def begin_batch(args, parser) -> dict:
    """Start recording a batch run: its command (paths reduced to file names) and its inputs."""
    ctx = {'args': args, 'command': command_of(args, parser), 'inputs': input_digests(args), 'project': None,
           'input_sha256': None}
    _BATCH.append(ctx)
    return ctx


def note_project(project):
    """The batch run's project as loaded, before the command changes it."""
    if _BATCH:
        _BATCH[-1]['project'] = project
        _BATCH[-1]['input_sha256'] = project_digest(project)['sha256']


def end_batch():
    if _BATCH:
        _BATCH.pop()


def current_batch() -> dict | None:
    return _BATCH[-1] if _BATCH else None


_NAME_ARGS = {'variant', 'frame', 'units', 'angles', 'condition', 'formats', 'camera', 'board', 'serial', 'blend',
              'lidar_calib_pair', 'template', 'validate_models', 'carla_scenario', 'carla_ground_truth',
              'live_cameras', 'live_fault', 'campaign_template', 'campaign_phase'}
_SKIP_ARGS = {'reproduce', 'tolerance', 'on_change', 'help', 'version'}
# input options: which of their values are files or folders to hash
INPUT_ARGS = {'path': 'all', 'install_check': 'all', 'calibrate_lens': 'all', 'import_calib': 'all',
              'suspension_data': 'all', 'drive_log': 'first', 'as_built_check': 'first', 'lidar_calib_folder': 'all',
              'fleet_import': 'rest', 'watch': 'all', 'experiment': 'first', 'ground_truth_summary': 'first',
              'detection_recording': 'all', 'critical_carla_drives': 'all', 'carla_faults': 'json',
              'carla_actors': 'json', 'carla_scenario_params': 'json'}
# ('json': a file or the JSON itself on the command line; only a file is an input to hash)


def command_of(args, parser) -> list:
    """The command line that runs args again: options in the parser's order, defaults left out, every path reduced to
    its file name; --on-change (a command of the user's) is never recorded."""
    import argparse
    argv = ['vantage']
    for a in parser._actions:
        if a.dest in _SKIP_ARGS or a.dest == argparse.SUPPRESS or not hasattr(args, a.dest):
            continue
        v = getattr(args, a.dest)
        if v is None or v is False or v == a.default:
            continue
        vals = list(v) if isinstance(v, (list, tuple)) else [v]
        if a.dest not in _NAME_ARGS:
            keep_first = a.dest in ('fleet_import', 'rig_health', 'carla_live')     # a vehicle's name, seconds
            vals = [x if (keep_first and i == 0) else reduce_path(x) for i, x in enumerate(vals)]
        vals = [str(x) for x in vals]
        if not a.option_strings:
            argv += vals
        elif a.nargs == 0:
            argv.append(max(a.option_strings, key=len))
        else:
            argv += [max(a.option_strings, key=len)] + vals
    return argv


def _input_values(args, dest):
    v = getattr(args, dest, None)
    if v in (None, False, ''):
        return []
    vals = list(v) if isinstance(v, (list, tuple)) else [v]
    how = INPUT_ARGS[dest]
    if how == 'json':
        return [x for x in vals if os.path.isfile(str(x))]
    return vals[:1] if how == 'first' else vals[1:] if how == 'rest' else vals


def input_digests(args) -> list:
    """Each input file or folder of a batch command by name, kind, size and SHA-256 (the project file's content is
    the project hash)."""
    out = []
    for dest in INPUT_ARGS:
        if dest in ('path', 'watch'):
            continue
        for v in _input_values(args, dest):
            d = path_digest(str(v))
            d['option'] = '--' + dest.replace('_', '-')
            out.append(d)
    return out


def _file_sha(path: str, size: int, budget: list) -> tuple[str, bool]:
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        if size <= budget[0]:
            budget[0] -= size
            for chunk in iter(lambda: fh.read(1 << 20), b''):
                h.update(chunk)
            return h.hexdigest(), True
        h.update(f'sampled\0{size}\0'.encode())
        h.update(fh.read(SAMPLE_BYTES))
        fh.seek(max(0, size - SAMPLE_BYTES))
        h.update(fh.read(SAMPLE_BYTES))
    return h.hexdigest(), False


def path_digest(path: str, limit: int = INPUT_HASH_LIMIT) -> dict:
    """{'name', 'kind': file|folder|missing, 'bytes', 'files', 'sha256', 'hashed': full|sampled}. A folder's hash
    covers every file's relative path, size and hash."""
    name = reduce_path(path)
    budget = [limit]
    try:
        if os.path.isfile(path):
            size = os.path.getsize(path)
            dig, full = _file_sha(path, size, budget)
            return {'name': name, 'kind': 'file', 'bytes': size, 'files': 1, 'sha256': dig,
                    'hashed': 'full' if full else 'sampled'}
        if os.path.isdir(path):
            h, total, n, full = hashlib.sha256(), 0, 0, True
            for root, dirs, files in os.walk(path):
                dirs.sort()
                for f in sorted(files):
                    p = os.path.join(root, f)
                    try:
                        size = os.path.getsize(p)
                        dig, ok = _file_sha(p, size, budget)
                    except OSError:
                        continue
                    rel = os.path.relpath(p, path).replace(os.sep, '/')
                    h.update(f'{rel}\0{size}\0{dig}\n'.encode())
                    total, n, full = total + size, n + 1, full and ok
            return {'name': name, 'kind': 'folder', 'bytes': total, 'files': n, 'sha256': h.hexdigest(),
                    'hashed': 'full' if full else 'sampled'}
    except OSError:
        pass
    return {'name': name, 'kind': 'missing', 'bytes': 0, 'files': 0, 'sha256': None, 'hashed': None}


def batch_output(doc, dest, folder=None, name=None, option=None) -> dict:
    """Which option of the running batch command wrote this document and under what name: {'option', 'name',
    'in_folder'} (name '-': the standard output; in a folder output, the file's path inside it)."""
    b = current_batch()
    args = b['args'] if b else None
    opt = option or SCHEMA_OPTION.get(_schema_stem(doc))
    if args is not None and option is None and (opt is None or getattr(args, opt, None) in (None, False)):
        for k, v in vars(args).items():
            vals = v if isinstance(v, (list, tuple)) else [v]
            if dest not in (None, '', '-') and dest in vals:
                opt = k
                break
    flag = '--' + opt.replace('_', '-') if opt else None
    if folder is not None:
        return {'option': flag, 'name': name or reduce_path(dest), 'in_folder': True}
    return {'option': flag, 'name': '-' if dest in (None, '', '-') else reduce_path(dest), 'in_folder': False}


def _schema_stem(doc) -> str:
    s = doc.get('schema') if isinstance(doc, dict) else None
    m = re.match(r'vantage\.([a-z0-9_]+)/\d+$', s or '')
    return m.group(1) if m else ''


# ====================================================================== the block
def _utc(when=None) -> str:
    t = when or _dt.datetime.now(_dt.timezone.utc)
    if t.tzinfo is None:
        t = t.astimezone()
    return t.astimezone(_dt.timezone.utc).replace(microsecond=0).strftime('%Y-%m-%dT%H:%M:%SZ')


def block(project, doc=None, *, variant=None, study: str | None = None, export: dict | None = None,
          output: dict | None = None, when=None, cache: dict | None = None) -> dict:
    """The provenance block for doc (a JSON object) computed from project (None: a result of no project, such as a
    validation run: no project, variant or settings). study: a schema's name or an export format id (default: the
    doc's schema); export: {'format', 'frame', 'units', 'angles'} for an exported file; output: where it was written;
    when: the time of the run (aware datetime; default now); cache: a dict kept for the files of one export (the
    project hash is computed once)."""
    env = environment()
    b = current_batch()
    study = study or _schema_stem(doc) or ''
    proj = var = None
    settings = {}
    if project is not None:
        key = ('provenance', id(project))
        pd = cache.get(key) if cache is not None else None
        if pd is None:
            pd = project_digest(project)
            if cache is not None:
                cache[key] = pd
        proj = {'name': project.name, 'file': reduce_path(project.path) or None, 'sha256': pd['sha256'],
                'sections': pd['sections']}
        if b and b.get('project') is project and b.get('input_sha256') and b['input_sha256'] != pd['sha256']:
            proj['input_sha256'] = b['input_sha256']
        v = variant if variant is not None else (project.variant if project.variants else None)
        if v is not None:
            idx = next((i for i, x in enumerate(project.variants) if x is v or x.uid == v.uid), None)
            var = {'name': v.name, 'index': None if idx is None else idx + 1, 'uid': v.uid,
                   'sha256': variant_digest(v)}
        settings = study_settings(project, study, export)
    elif export:
        settings = {'export': dict(export)}
    out = {
        'schema': SCHEMA,
        'vantage': env['vantage'],
        'runtime': env['runtime'],
        'interface': 'batch' if b else 'app',
        'project': proj,
        'variant': var,
        'study': study,
        'output': output,
        'settings': settings,
        'seeds': seeds_of(settings),
        'command': list(b['command']) if b else None,
        'inputs': [dict(x) for x in b['inputs']] if b else [],
        'created_utc': _utc(when),
        'result_sha256': result_digest(doc) if doc is not None else None,
        'result_hash': RESULT_HASH_TEXT,
        'content_hash': CONTENT_HASH_TEXT,
    }
    path = getattr(project, 'path', '') if project is not None else ''
    return scrub(out, extra=(os.path.dirname(os.path.abspath(path)),) if path else ())


def stamp(doc: dict, project, **kw) -> dict:
    """doc with its provenance block as the last key (an existing one is replaced); doc itself is not changed."""
    out = {k: v for k, v in doc.items() if k != KEY}
    out[KEY] = block(project, out, **kw)
    return out


def _insert(text: str, blk: dict) -> str | None:
    """text (one JSON object) with "provenance": blk added before its closing brace, keeping its formatting."""
    s = text.rstrip()
    if not s.endswith('}'):
        return None
    tail = text[len(s):]
    head = s[:-1].rstrip()
    empty = head.lstrip().startswith('{') and head.strip() == '{'
    m = re.match(r'\s*\{\r?\n([ \t]+)"', s)
    if m:
        ind = m.group(1)
        nl = '\r\n' if '\r\n' in s[:4096] else '\n'
        body = json.dumps(blk, indent=ind, ensure_ascii=False).replace('\n', nl + ind)
        return head + ('' if empty else ',') + nl + ind + json.dumps(KEY) + ': ' + body + nl + '}' + tail
    sep, colon = (', ', ': ') if re.search(r'"\s*:\s', s[:4096]) else (',', ':')
    body = json.dumps(blk, ensure_ascii=False, separators=(sep, colon))
    return head + ('' if empty else sep) + json.dumps(KEY) + colon + body + '}' + tail


def stamp_bytes(data: bytes, project, **kw) -> bytes:
    """A JSON file's bytes with the provenance block added (its other bytes unchanged); anything that is not one
    JSON object (a list such as a nuScenes table, YAML, a binary file) is returned as it is."""
    try:
        text = data.decode('utf-8')
    except (UnicodeDecodeError, AttributeError):
        return data
    if not text.lstrip().startswith('{'):
        return data
    try:
        doc = json.loads(text)
    except ValueError:
        return data
    if not isinstance(doc, dict):
        return data
    blk = block(project, doc, **kw)
    new = None if KEY in doc else _insert(text, blk)
    if new is not None:
        try:
            back = json.loads(new)
            ok = isinstance(back, dict) and list(back)[-1] == KEY and len(back) == len(doc) + 1 and \
                result_digest(back) == blk['result_sha256']
        except ValueError:
            ok = False
        if ok:
            return new.encode('utf-8')
    m = re.match(r'\s*\{\r?\n([ \t]+)"', text)          # rewrite: an existing block, or text we could not extend
    doc = {k: v for k, v in doc.items() if k != KEY}
    doc[KEY] = blk
    return (json.dumps(doc, indent=m.group(1) if m else None, ensure_ascii=False) + ('\n' if text.endswith('\n')
                                                                                      else '')).encode('utf-8')


def stamp_files(files, project, output: dict | None = None, **kw) -> list:
    """[(relative path, bytes)] with every JSON object stamped; output gets each file's path as 'name'."""
    out = []
    for rel, data in files:
        if str(rel).lower().endswith('.json'):
            data = stamp_bytes(data, project, output=dict(output or {}, name=rel, in_folder=True), **kw)
        out.append((rel, data))
    return out


def stamp_path(path: str, project, **kw) -> bool:
    """Stamp a JSON file on disk in place (written atomically); False when it is not a JSON object."""
    try:
        with open(path, 'rb') as fh:
            data = fh.read()
    except OSError:
        return False
    new = stamp_bytes(data, project, **kw)
    if new is data or new == data:
        return False
    tmp = path + '.part'
    with open(tmp, 'wb') as fh:
        fh.write(new)
    os.replace(tmp, path)
    return True


def stamp_paths(paths, project, folder=None, **kw) -> list:
    """Stamp every JSON object among files already written (a module's export() into folder); returns paths. For
    exports that do not go through core.exporters, e.g. a page's Export button: PV.stamp_paths(X.export(res, p, d),
    p, d)."""
    base = os.path.abspath(folder) if folder else None
    given = kw.pop('output', None) or {}
    for path in paths:
        if str(path).lower().endswith('.json') and os.path.isfile(path):
            out = dict(given)
            if base:
                out.update(name=os.path.relpath(os.path.abspath(path), base).replace(os.sep, '/'), in_folder=True)
            stamp_path(path, project, output=out or None, **kw)
    return list(paths)


# ====================================================================== comparing results
def parse_tolerances(specs) -> list:
    """--tolerance specs: '[PATTERN=]VALUE[%]'. VALUE is absolute in the quantity's own unit, or with % relative to
    the recorded value. PATTERN matches a number's path ('metrics.blind_m2', 'cameras[*].seen_m2', '*_pct') or its
    key's name; the first pattern that matches applies; a bare VALUE applies to every number no pattern matches.
    Returns [(pattern or None, value, relative)]."""
    out = []
    for s in specs or ():
        for part in str(s).split(','):
            part = part.strip()
            if not part:
                continue
            pat, _, val = part.rpartition('=') if '=' in part else ('', '', part)
            rel = val.endswith('%')
            try:
                x = float(val[:-1] if rel else val)
            except ValueError:
                raise ValueError(f'{part!r}: give [PATTERN=]VALUE or VALUE%, e.g. blind_m2=0.005 or 0.1%') from None
            if not math.isfinite(x) or x < 0:
                raise ValueError(f'{part!r}: a tolerance is a finite number, 0 or more')
            out.append((pat.strip() or None, x / 100.0 if rel else x, rel))
    return out


def _glob(pat: str):
    return re.compile(''.join('.*' if c == '*' else '.' if c == '?' else re.escape(c) for c in pat))


def tolerance_for(path: str, tolerances: list) -> tuple[float, bool, str | None]:
    """(value, relative, pattern) for the number at path: the first matching pattern, else the bare default, else
    exact."""
    last = re.sub(r'\[\d+\]$', '', path.rsplit('.', 1)[-1])
    for pat, x, rel in tolerances:
        if pat is not None:
            rx = _glob(pat)
            if rx.fullmatch(path) or rx.fullmatch(last) or rx.fullmatch(re.sub(r'\[\d+\]', '', path)):
                return x, rel, pat
    default = [t for t in tolerances if t[0] is None]
    if default:
        return default[-1][1], default[-1][2], '*'
    return 0.0, False, None


def _leaves(o, path=''):
    if isinstance(o, dict):
        if not o:
            yield path, {}
        for k, v in o.items():
            yield from _leaves(v, f'{path}.{k}' if path else str(k))
    elif isinstance(o, list):
        if not o:
            yield path, []
        for i, v in enumerate(o):
            yield from _leaves(v, f'{path}[{i}]')
    else:
        yield path, o


def _uids_in_order(doc) -> dict:
    """10-hex uid tokens in strings, numbered by first appearance (identities, not results)."""
    out = {}
    for _, v in _leaves(doc):
        if isinstance(v, str):
            for m in _UID.finditer(v):
                out.setdefault(m.group(0), f'<uid{len(out) + 1}>')
    return out


def _isnum(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def compare(recorded, reproduced, tolerances=()) -> dict:
    """Every number and text of two results (provenance, time stamps and computation times left out; uids compared by
    order of appearance). Returns {'counts', 'numbers': [[path, recorded, reproduced, status, |diff|, tolerance]],
    'texts': [[path, recorded, reproduced]] (changed only), 'missing', 'extra', 'worst', 'verdict'}; status is
    exact, within, differs, missing or extra; verdict reproduces, reproduces_within_tolerance or
    does_not_reproduce."""
    A, B = neutral(recorded), neutral(reproduced)
    ua, ub = _uids_in_order(A), _uids_in_order(B)
    la = dict(_leaves(A))
    lb = dict(_leaves(B))
    numbers, texts, missing, extra = [], [], [], []
    counts = {'numbers': 0, 'exact': 0, 'within': 0, 'differs': 0, 'texts': 0, 'texts_changed': 0, 'missing': 0,
              'extra': 0}
    for path, a in la.items():
        if path not in lb:
            missing.append(path)
            if _isnum(a):
                numbers.append([path, a, None, 'missing', None, None])
                counts['numbers'] += 1
            counts['missing'] += 1
            continue
        b = lb[path]
        if _isnum(a) and _isnum(b):
            counts['numbers'] += 1
            fa, fb = float(a), float(b)
            if fa == fb or (math.isnan(fa) and math.isnan(fb)):
                numbers.append([path, a, b, 'exact', 0.0, None])
                counts['exact'] += 1
                continue
            d = abs(fa - fb) if math.isfinite(fa) and math.isfinite(fb) else math.inf
            x, rel, pat = tolerance_for(path, tolerances)
            lim = x * abs(fa) if rel else x
            ok = d <= lim
            numbers.append([path, a, b, 'within' if ok else 'differs', d, (f'{x * 100:g} %' if rel else x)
                            if pat else 0])
            counts['within' if ok else 'differs'] += 1
            continue
        counts['texts'] += 1
        sa = _UID.sub(lambda m: ua.get(m.group(0), m.group(0)), a) if isinstance(a, str) else a
        sb = _UID.sub(lambda m: ub.get(m.group(0), m.group(0)), b) if isinstance(b, str) else b
        if sa != sb and not (_isnum(sa) and _isnum(sb)):
            texts.append([path, a, b])
            counts['texts_changed'] += 1
    for path, b in lb.items():
        if path not in la:
            extra.append(path)
            counts['extra'] += 1
            if _isnum(b):
                numbers.append([path, None, b, 'extra', None, None])
    bad = [r for r in numbers if r[3] in ('differs', 'missing', 'extra')]

    def rel_err(r):
        a, d = r[1], r[4]
        if d is None:
            return math.inf
        return d / abs(a) if _isnum(a) and a else d
    worst = sorted((r for r in numbers if r[3] in ('within', 'differs')), key=rel_err, reverse=True)[:20]
    if bad or texts or missing or extra:
        verdict = 'does_not_reproduce'
    elif counts['within']:
        verdict = 'reproduces_within_tolerance'
    else:
        verdict = 'reproduces'
    return {'counts': counts, 'numbers': numbers, 'texts': texts, 'missing': missing, 'extra': extra,
            'worst': worst, 'verdict': verdict}


VERDICT_TEXT = {'reproduces': 'reproduces exactly', 'reproduces_within_tolerance': 'reproduces within the tolerances',
                'does_not_reproduce': 'does not reproduce', 'cannot_reproduce': 'cannot be reproduced'}


def differences(recorded: dict, now: dict) -> list:
    """What differs between two provenance blocks, as sentences (the reasons a result may not reproduce)."""
    out = []
    rv, nv = recorded.get('vantage') or {}, now.get('vantage') or {}
    if rv.get('version') != nv.get('version'):
        out.append(f'a different Vantage version: {rv.get("version")} then, {nv.get("version")} now')
    if rv.get('code_sha256') and nv.get('code_sha256') and rv['code_sha256'] != nv['code_sha256']:
        g0, g1 = rv.get('git') or {}, nv.get('git') or {}
        where = ''
        if g0.get('commit') and g1.get('commit'):
            where = (f' (commit {g0["commit"][:10]}{" with local changes" if g0.get("dirty") else ""} then, '
                     f'{g1["commit"][:10]}{" with local changes" if g1.get("dirty") else ""} now)')
        out.append('different code: the app\'s sources differ' + where)
    rr, nr = recorded.get('runtime') or {}, now.get('runtime') or {}
    libs = [f'{k} {rr.get(k)} → {nr.get(k)}' for k in ('python', 'numpy', 'scipy', 'opencv', 'qt', 'blas')
            if rr.get(k) != nr.get(k)]
    if libs:
        out.append('different libraries: ' + ', '.join(libs))
    if rr.get('platform') != nr.get('platform') or rr.get('machine') != nr.get('machine'):
        out.append(f'a different platform: {rr.get("platform")} then, {nr.get("platform")} now')
    rp, np_ = recorded.get('project') or {}, now.get('project') or {}
    if rp.get('sha256') != np_.get('sha256'):
        secs = [k for k in sorted(set(rp.get('sections') or {}) | set(np_.get('sections') or {}))
                if (rp.get('sections') or {}).get(k) != (np_.get('sections') or {}).get(k)]
        out.append('the project differs' + (f': {", ".join(secs)}' if secs else ''))
    rvar, nvar = recorded.get('variant') or {}, now.get('variant') or {}
    if rvar.get('sha256') != nvar.get('sha256'):
        out.append(f'the variant differs ("{rvar.get("name")}" then, "{nvar.get("name")}" now)')
    ri = {(i.get('option'), i.get('name')): i.get('sha256') for i in recorded.get('inputs') or []}
    ni = {(i.get('option'), i.get('name')): i.get('sha256') for i in now.get('inputs') or []}
    changed = [n for k, n in ((k, k[1]) for k in ri) if k in ni and ri[k] != ni[k]]
    if changed:
        out.append('different input files: ' + ', '.join(changed))
    return out


def report(name: str, recorded_doc: dict, reproduced_doc: dict | None, tolerances=(), *, now_provenance=None,
           project_now: dict | None = None, error: str | None = None, command=None) -> dict:
    """The reproduction report (schema vantage.reproduce/1) of the result file name: verdict, the reasons, what was
    the same, the integrity of the recorded file and every number compared."""
    prov = recorded_doc.get(KEY) or {}
    integ = verify(recorded_doc)
    doc = {'schema': REPRODUCE_SCHEMA, 'result': name, 'study': prov.get('study'),
           'recorded': {k: prov.get(k) for k in ('vantage', 'runtime', 'project', 'variant', 'output', 'command',
                                                 'inputs', 'created_utc', 'result_sha256')},
           'integrity': {'intact': integ['ok'], 'recorded_sha256': integ['recorded'], 'file_sha256': integ['actual']},
           'command': command, 'tolerances': [{'pattern': p or '*', 'value': (f'{x * 100:g} %' if r else x)}
                                              for p, x, r in tolerances],
           'assumptions': ASSUMPTIONS}
    reasons = []
    if integ['ok'] is False:
        reasons.append('the result file was changed after Vantage wrote it (its result hash does not match)')
    now = now_provenance or {}
    if now:
        doc['now'] = {k: now.get(k) for k in ('vantage', 'runtime', 'project', 'variant', 'inputs')}
    if project_now is not None:
        rp = prov.get('project') or {}
        want = rp.get('input_sha256') or rp.get('sha256')
        doc['project_given'] = {'sha256': project_now.get('sha256'), 'same_as_recorded': project_now.get('sha256') == want}
    if error is not None or reproduced_doc is None:
        doc.update(verdict='cannot_reproduce', verdict_text=VERDICT_TEXT['cannot_reproduce'],
                   message=error or 'The re-run wrote no result.', reasons=reasons + ([error] if error else []))
        return doc
    cmp_ = compare(recorded_doc, reproduced_doc, tolerances)
    diffs = differences(prov, now) if now else []
    if project_now is not None and not doc['project_given']['same_as_recorded'] and \
            not any(d.startswith('the project differs') for d in diffs):
        diffs.append('the project file given is not the one recorded (its content hash differs)')
    same_hash = bool(now.get('result_sha256')) and now.get('result_sha256') == prov.get('result_sha256')
    c = cmp_['counts']
    verdict = cmp_['verdict']
    if verdict == 'reproduces':
        msg = (f'Reproduces exactly: all {c["numbers"]} numbers are the same' +
               (' and the reproduced result has the same SHA-256.' if same_hash else '.'))
    elif verdict == 'reproduces_within_tolerance':
        msg = (f'Reproduces within the tolerances: {c["within"]} of {c["numbers"]} numbers differ, none by more than '
               'allowed.')
    else:
        parts = []
        if c['differs']:
            parts.append(f'{c["differs"]} of {c["numbers"]} numbers differ beyond their tolerance')
        if c['texts_changed']:
            parts.append(f'{c["texts_changed"]} texts changed')
        if c['missing'] or c['extra']:
            parts.append(f'{c["missing"]} values missing and {c["extra"]} new')
        w = next((r for r in cmp_['worst'] if r[3] == 'differs'), None)
        msg = 'Does not reproduce: ' + '; '.join(parts) + (f' (largest: {w[0]}, {w[1]} then, {w[2]} now)' if w else '')
        msg += '.'
        if not diffs and integ['ok'] is not False:
            diffs.append('nothing recorded differs (version, code, libraries, project, variant, inputs): a source of '
                         'non-determinism; please report it')
    doc.update(verdict=verdict, verdict_text=VERDICT_TEXT[verdict], message=msg, reasons=reasons + diffs,
               same={'result_sha256': same_hash,
                     'vantage': not any(d.startswith(('a different Vantage', 'different code')) for d in diffs),
                     'runtime': not any(d.startswith(('different libraries', 'a different platform')) for d in diffs),
                     'project': not any(d.startswith(('the project', )) for d in diffs),
                     'variant': not any(d.startswith('the variant') for d in diffs),
                     'inputs': not any(d.startswith('different input') for d in diffs)},
               reproduced_sha256=now.get('result_sha256'), counts=c, worst=cmp_['worst'], numbers=cmp_['numbers'],
               texts=cmp_['texts'], missing=cmp_['missing'], extra=cmp_['extra'])
    return doc


def summary(rep: dict) -> dict:
    """The report in short (for the assistant and the batch log): verdict, message, reasons, counts, worst five."""
    return {'result': rep.get('result'), 'study': rep.get('study'), 'verdict': rep.get('verdict'),
            'message': rep.get('message'), 'reasons': list(rep.get('reasons') or []), 'counts': rep.get('counts'),
            'worst': [{'path': r[0], 'recorded': r[1], 'reproduced': r[2], 'status': r[3]}
                      for r in (rep.get('worst') or [])[:5]]}
