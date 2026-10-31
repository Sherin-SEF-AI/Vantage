"""Automation: when the design changes, regenerate what downstream needs and run the user's own checks.

Two halves, kept apart on purpose:

- What to regenerate lives in the project (``analysis.automation``, AutomationSettings): whether to automate, the
  trigger (on save, after a quiet period of changes, or by hand), the export formats, the folder they go to (relative
  to the project file and never outside its folder), the variant and conventions, whether to run the requirements
  check. **It never holds a command.** A project file can only make Vantage write export files into a folder next to
  itself.
- The command to run afterwards (``make perception-test``, a script) lives only in the user's own settings
  (CommandStore: QSettings on this computer, keyed by the project file's path). It is shown in full and must be
  approved by the user before it first runs for that project file; changing it withdraws the approval. A project moved
  to another path has no command until the user sets one there.

A run (run()): analyse the variant, export each format into the folder, run the requirements check into
``<folder>/requirements``, remove the files the previous run wrote and this one did not (a camera removed, a format
dropped; only files Vantage itself wrote, from the local history), write ``vantage_summary.json``, then, when a command
is set and approved, run it through the system shell in the project's folder with these environment variables:

    VANTAGE_EXPORT_DIR   the export folder (absolute)
    VANTAGE_PROJECT      the project file (absolute)       VANTAGE_PROJECT_DIR  its folder
    VANTAGE_VARIANT      the variant's name                VANTAGE_VARIANT_UID  its id
    VANTAGE_SUMMARY      vantage_summary.json (absolute)   VANTAGE_TRIGGER      save | change | manual | watch
    VANTAGE_CHECK_EXIT   the requirements check's exit code (0, or 5 when a must requirement fails; empty when off)
    VANTAGE_RUN_ID       this run's id                     VANTAGE_UNSAVED      1 when unsaved edits are exported

Its output, exit code and duration and the files that changed (by content, SHA-256, inside the export folder: by the
regeneration and by the command) are recorded; the History keeps the last runs per project file in the user's config
folder (never in the project).

Headless: watch() polls a project file and runs on every change (batch ``--watch project.vantage --on-change "cmd"``),
passing the command's exit code through. ci_files() writes GitHub Actions and GitLab CI templates that run the batch
checks on the project in its repository and keep the calibration files and the review pack as artifacts.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import os
import re
import secrets
import signal
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field

SCHEMA = 'vantage.automation_run/1'              # vantage_summary.json and a run's JSON export
HISTORY_SCHEMA = 'vantage.automation_history/1'
SUMMARY_NAME = 'vantage_summary.json'
REQUIREMENTS_DIR = 'requirements'
TRIGGERS = ('save', 'change', 'manual')
TRIGGER_LABEL = {'save': 'On save', 'change': 'After changes', 'manual': 'By hand', 'watch': 'Watch'}
DEFAULT_OUT_DIR = 'exports/automation'
# the calibration files a data pipeline reads: extrinsics, per-camera intrinsics, ROS, dataset devkits
DEFAULT_FORMATS = ('extrinsics_json', 'opencv_yaml', 'ros_camera_info', 'ros2', 'nuscenes_calib', 'kitti_calib',
                   'kalibr_camchain')
UI_ONLY_GROUPS = ('Images',)                     # rendered by the app window: not available headless
LIMITS = {'quiet_s': (1.0, 3600.0)}
COMMAND_LIMITS = {'timeout_min': (0.1, 1440.0)}
DEFAULT_TIMEOUT_MIN = 30.0
OUTPUT_KEEP = 200_000                            # characters of a run's output kept in memory
OUTPUT_KEEP_HISTORY = 64_000                     # … and in the history file
HISTORY_KEEP = 30                                # runs per project file
HISTORY_PROJECTS = 200
SCAN_FILES = 20_000                              # the export folder is compared up to this many files …
SCAN_BYTES = 4_000_000_000                       # … and bytes
EXIT_ERROR = 4                                   # the regeneration failed (batch: cannot be read or written)
EXIT_TIMEOUT = 124                               # the command ran over its time limit (as coreutils timeout)
EXIT_CANCELLED = 130                             # stopped by the user (as a shell's Ctrl+C)
STATUS_LABEL = {'running': 'Running', 'passed': 'Passed', 'failed': 'Failed', 'error': 'Error',
                'cancelled': 'Stopped', 'timeout': 'Timed out'}
STATUS_SEVERITY = {'passed': 'good', 'failed': 'serious', 'error': 'critical', 'timeout': 'serious',
                   'cancelled': 'warning', 'running': None}
COMMAND_STATE_TEXT = {'none': 'no command set', 'off': 'command switched off for this project',
                      'not_approved': 'command not approved yet', 'pending': 'command to run', 'ran': 'command ran',
                      'skipped': 'command not run: the regeneration did not finish'}
ENV_HELP = (('VANTAGE_EXPORT_DIR', 'the export folder'), ('VANTAGE_PROJECT', 'the project file'),
            ('VANTAGE_PROJECT_DIR', 'its folder'), ('VANTAGE_VARIANT', 'the variant\'s name'),
            ('VANTAGE_VARIANT_UID', 'its id'), ('VANTAGE_SUMMARY', 'vantage_summary.json'),
            ('VANTAGE_TRIGGER', 'save, change, manual or watch'),
            ('VANTAGE_CHECK_EXIT', 'the requirements check\'s exit code (5: a must requirement fails)'),
            ('VANTAGE_RUN_ID', 'this run\'s id'), ('VANTAGE_UNSAVED', '1 when unsaved edits were exported'))
ASSUMPTIONS = ('The command runs through the system shell with your permissions, in the project\'s folder, with the '
               'environment of your session plus the VANTAGE_ variables; Vantage cannot sandbox it. It is kept in '
               'your own settings for this project file and runs only after you approved it there. Changed files are '
               'compared by content (SHA-256) inside the export folder only, up to 20 000 files and 4 GB. A run after '
               'changes exports the design as it is in the window, unsaved edits included (VANTAGE_UNSAVED=1), while '
               'VANTAGE_PROJECT names the file on disk.')


# =========================================================================== small helpers
def norm_path(path: str) -> str:
    """A project file's identity: absolute, symlinks resolved, case-folded where the file system folds case."""
    return os.path.normcase(os.path.realpath(os.path.abspath(os.path.expanduser(str(path)))))


def _config_dir() -> str:
    from .credentials import _dir
    return _dir()


def _now_iso(t: float | None = None) -> str:
    return _dt.datetime.fromtimestamp(time.time() if t is None else t).astimezone().replace(
        microsecond=0).isoformat()


def _plain(text, limit=200) -> str:
    """Text from a project made safe for one line, an environment variable or a comment: no control characters."""
    return re.sub(r'[\x00-\x1f\x7f]+', ' ', str(text or '')).strip()[:limit]


def _write_private(path: str, data: bytes):
    """Write a file only the user can read (0600), atomically."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f'{path}.{os.getpid()}.tmp'
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'wb') as fh:
        fh.write(data)
    os.replace(tmp, path)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def formats_available() -> list:
    """[(id, group, label, ext_label)] of every export format a run can write (the app window's images excluded)."""
    from . import exporters as ex
    return [(f.id, f.group, f.label, f.ext_label) for f in ex.FORMATS if f.group not in UI_ONLY_GROUPS]


def clean_folder(text) -> tuple[str | None, str]:
    """(folder relative to the project, '') or (None, why not). A folder must stay inside the project's folder: no
    absolute path, no drive, no ~, no '..', not the project folder itself, no characters Windows forbids."""
    s = str(text if text is not None else '').strip().replace('\\', '/')
    if not s:
        return None, 'no folder is given'
    if s.startswith('/') or s.startswith('~') or re.match(r'^[A-Za-z]:', s):
        return None, f'"{_plain(s, 80)}" is not relative to the project: give a folder inside the project\'s folder'
    parts = [x for x in s.split('/') if x not in ('', '.')]
    if not parts:
        return None, 'the folder must be inside the project\'s folder, not the folder itself'
    if any(x == '..' for x in parts):
        return None, f'"{_plain(s, 80)}" leaves the project\'s folder (..)'
    if any(ch < ' ' or ch in ':*?"<>|\x7f' for x in parts for ch in x):
        return None, f'"{_plain(s, 80)}" has characters a folder name cannot have'
    if any(x.lower() == '.git' for x in parts):
        return None, f'"{_plain(s, 80)}" is inside a repository\'s .git folder'
    if any(x.endswith(('.', ' ')) for x in parts):
        return None, f'"{_plain(s, 80)}": a folder name cannot end in a dot or a space'
    out = '/'.join(parts)
    if len(out) > 200:
        return None, 'the folder name is too long'
    return out, ''


def _inside(base_real: str, real: str, strict=False) -> bool:
    """real lies inside base_real (both resolved); strict: and is not base_real itself. False across drives."""
    try:
        return os.path.commonpath([base_real, real]) == base_real and not (strict and real == base_real)
    except ValueError:
        return False


def resolve_out_dir(project_path: str, folder: str) -> str:
    """The absolute export folder for a project file and a relative folder; ValueError when the project has no file
    yet or the folder (through '..' or a symbolic link) would lie outside the project's folder."""
    if not project_path:
        raise ValueError('Save the project first: the export folder is relative to the project file.')
    rel, why = clean_folder(folder)
    if rel is None:
        raise ValueError(f'Export folder: {why}.')
    base = os.path.dirname(os.path.abspath(project_path))
    out = os.path.normpath(os.path.join(base, *rel.split('/')))
    if not _inside(os.path.realpath(base), os.path.realpath(out), strict=True):
        raise ValueError(f'Export folder: "{rel}" leads outside the project\'s folder (a symbolic link).')
    return out


def check_links(out_dir: str, base: str):
    """ValueError when a symbolic link inside out_dir leads outside base (the project's folder): exports would write
    through it, outside the folder the project may write to."""
    rb = os.path.realpath(base)
    for root, dirs, names in os.walk(out_dir, followlinks=False):
        for name in dirs + names:
            path = os.path.join(root, name)
            if os.path.islink(path):
                real = os.path.realpath(path)
                if not _inside(rb, real):
                    rel = os.path.relpath(path, out_dir).replace(os.sep, '/')
                    raise ValueError(f'Export folder: "{rel}" is a symbolic link to {real}, outside the project\'s '
                                     'folder; Vantage does not write through it.')


def pick_variant(project, key):
    """The variant a uid or a name names, else None."""
    key = str(key or '')
    if not key:
        return None
    return project.variant_by_uid(key) or next((v for v in project.variants if v.name == key), None)


# =========================================================================== project settings (never a command)
@dataclass
class AutomationSettings:
    enabled: bool = False              # automate this project at all (empty settings: off, as in older projects)
    trigger: str = 'save'              # save | change | manual
    quiet_s: float = 10.0              # after changes: run once nothing changed for this long
    formats: list = field(default_factory=lambda: list(DEFAULT_FORMATS))
    out_dir: str = DEFAULT_OUT_DIR     # relative to the project file's folder, never outside it
    variant: str = ''                  # a variant uid; '' = the active one
    frame: str = ''                    # export frame; '' = the project's
    units: str = 'm'                   # m | mm
    angles: str = 'deg'                # deg | rad
    check: bool = True                 # run the requirements check into <out_dir>/requirements
    prune: bool = True                 # remove files the previous run wrote and this one did not
    run_command: bool = True           # run this computer's approved command afterwards (the command is not here)

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d) -> 'AutomationSettings':
        """Tolerant of junk: unknown keys ignored (a 'command' among them: a project never carries one), numbers
        clamped to LIMITS, unknown formats dropped, an unusable folder replaced by the default."""
        s = AutomationSettings()
        d = d if isinstance(d, dict) else {}
        for k in ('enabled', 'check', 'prune', 'run_command'):
            v = _bool(d.get(k))
            if v is not None:
                setattr(s, k, v)
        if d.get('trigger') in TRIGGERS:
            s.trigger = d['trigger']
        for k, (lo, hi) in LIMITS.items():
            if k in d:
                try:
                    v = float(d[k])
                except (TypeError, ValueError):
                    continue
                if math.isfinite(v):
                    setattr(s, k, min(max(v, lo), hi))
        if isinstance(d.get('formats'), (list, tuple)):
            known = {f[0] for f in formats_available()}
            out = []
            for f in d['formats']:
                if isinstance(f, str) and f in known and f not in out:
                    out.append(f)
            s.formats = out
        if 'out_dir' in d:
            rel, _ = clean_folder(d['out_dir'])
            if rel is not None:
                s.out_dir = rel
        if isinstance(d.get('variant'), str):
            s.variant = _plain(d['variant'], 64)
        from .project import FRAME_PRESETS
        if d.get('frame') == '' or d.get('frame') in FRAME_PRESETS:
            s.frame = d['frame']
        if d.get('units') in ('m', 'mm'):
            s.units = d['units']
        if d.get('angles') in ('deg', 'rad'):
            s.angles = d['angles']
        return s


def _bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
        return bool(v)
    if isinstance(v, str) and v.strip().lower() in ('1', 'true', 'yes', 'on', '0', 'false', 'no', 'off'):
        return v.strip().lower() in ('1', 'true', 'yes', 'on')
    return None


def settings_of(project) -> AutomationSettings:
    return AutomationSettings.from_dict(getattr(project.analysis, 'automation', None))


def settings_problems(d) -> list[str]:
    """What from_dict had to drop from stored settings (for the Issues list and the page)."""
    if not isinstance(d, dict) or not d:
        return []
    out = []
    if 'out_dir' in d:
        rel, why = clean_folder(d['out_dir'])
        if rel is None:
            out.append(f'the export folder was not usable ({why}); {DEFAULT_OUT_DIR} is used')
    if isinstance(d.get('formats'), (list, tuple)):
        known = {f[0] for f in formats_available()}
        bad = [str(f) for f in d['formats'] if not (isinstance(f, str) and f in known)]
        if bad:
            out.append('unknown or window-only export formats are skipped: ' + ', '.join(_plain(b, 40) for b in bad[:6]))
    if 'command' in d or 'commands' in d:
        out.append('the project file names a command: it is ignored (commands are kept in your own settings only)')
    return out


# =========================================================================== the user's command (this computer only)
@dataclass
class CommandConfig:
    path: str                          # the project file it belongs to (normalised)
    command: str = ''
    timeout_min: float = DEFAULT_TIMEOUT_MIN
    approved: str = ''                 # approval digest of (path, command); empty = not approved
    approved_at: float = 0.0
    set_at: float = 0.0

    @property
    def timeout_s(self) -> float:
        return self.timeout_min * 60.0


def approval_digest(path: str, command: str) -> str:
    """What an approval covers: this command for this project file. Any change to either withdraws it."""
    return hashlib.sha256(f'vantage-automation\0{norm_path(path)}\0{command}'.encode('utf-8')).hexdigest()


class CommandStore:
    """The user's commands, one per project file, in the user's own settings (a QSettings, or any object with
    value / setValue / remove), never in a project. Stored as JSON under automation/commands/<hash of the path>."""
    GROUP = 'automation/commands'

    def __init__(self, settings):
        self.settings = settings

    def key(self, path: str) -> str:
        return f'{self.GROUP}/{hashlib.sha1(norm_path(path).encode("utf-8")).hexdigest()[:20]}'

    def get(self, path: str) -> CommandConfig | None:
        if not path:
            return None
        raw = self.settings.value(self.key(path), '')
        try:
            d = json.loads(raw) if isinstance(raw, str) and raw else None
        except ValueError:
            d = None
        if not isinstance(d, dict) or d.get('path') != norm_path(path) or not isinstance(d.get('command'), str):
            return None
        cfg = CommandConfig(d['path'], d['command'])
        try:
            t = float(d.get('timeout_min', DEFAULT_TIMEOUT_MIN))
            cfg.timeout_min = min(max(t, COMMAND_LIMITS['timeout_min'][0]), COMMAND_LIMITS['timeout_min'][1]) \
                if math.isfinite(t) else DEFAULT_TIMEOUT_MIN
        except (TypeError, ValueError):
            pass
        cfg.approved = str(d.get('approved') or '')
        cfg.approved_at = float(d.get('approved_at') or 0) if isinstance(d.get('approved_at'), (int, float)) else 0.0
        cfg.set_at = float(d.get('set_at') or 0) if isinstance(d.get('set_at'), (int, float)) else 0.0
        return cfg

    def _put(self, cfg: CommandConfig):
        self.settings.setValue(self.key(cfg.path), json.dumps(asdict(cfg)))
        if hasattr(self.settings, 'sync'):
            self.settings.sync()

    def put(self, path: str, command: str, timeout_min: float | None = None) -> CommandConfig | None:
        """Set (or, with an empty command, remove) the command for a project file. A changed command is not approved
        until approve(); an unchanged one keeps its approval."""
        command = str(command or '').strip()
        if not command:
            self.remove(path)
            return None
        old = self.get(path)
        cfg = CommandConfig(norm_path(path), command, old.timeout_min if old else DEFAULT_TIMEOUT_MIN,
                            set_at=time.time())
        if timeout_min is not None:
            try:
                t = float(timeout_min)
            except (TypeError, ValueError):
                t = cfg.timeout_min
            if math.isfinite(t):
                cfg.timeout_min = min(max(t, COMMAND_LIMITS['timeout_min'][0]), COMMAND_LIMITS['timeout_min'][1])
        if old is not None and old.command == command and self.is_approved(path):
            cfg.approved, cfg.approved_at, cfg.set_at = old.approved, old.approved_at, old.set_at
        self._put(cfg)
        return cfg

    def approve(self, path: str, command: str | None = None) -> CommandConfig | None:
        """The user confirmed this command (shown in full) for this project file. With command given, only that exact
        text is approved (it must be the stored one)."""
        cfg = self.get(path)
        if cfg is None or (command is not None and command.strip() != cfg.command):
            return None
        cfg.approved, cfg.approved_at = approval_digest(path, cfg.command), time.time()
        self._put(cfg)
        return cfg

    def revoke(self, path: str):
        cfg = self.get(path)
        if cfg is not None:
            cfg.approved, cfg.approved_at = '', 0.0
            self._put(cfg)

    def is_approved(self, path: str) -> bool:
        cfg = self.get(path)
        return bool(cfg and cfg.command and cfg.approved == approval_digest(path, cfg.command))

    def approved_command(self, path: str) -> CommandConfig | None:
        """The command to run for a project file: only a set and approved one."""
        cfg = self.get(path)
        return cfg if cfg is not None and cfg.command and cfg.approved == approval_digest(path, cfg.command) else None

    def remove(self, path: str):
        self.settings.remove(self.key(path))
        if hasattr(self.settings, 'sync'):
            self.settings.sync()


# =========================================================================== the export folder: contents and changes
# Exports carry the time they were written (an "exported" stamp, KITTI's calib_time, a study's timing), NPZ archives
# their members' times and PDFs their creation time, so bytes differ on every run. A change is therefore judged on a
# fingerprint: text with its time stamps and timings masked, a zip archive by its members' names, sizes and CRCs; a
# PDF cannot be compared (its creation time is also inside compressed page streams) and counts as rewritten.
_VOLATILE = re.compile(rb'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?'
                       rb'|\b\d{1,2}-[A-Z][a-z]{2}-\d{4} \d{2}:\d{2}:\d{2}'
                       rb'|"seconds": ?-?[0-9][0-9.eE+-]*')
TEXT_MAX = 32_000_000


@dataclass
class Scan:
    files: dict                        # relative path ('/') -> (size, sha256, fingerprint; None for a PDF)
    complete: bool = True


def fingerprint(path: str) -> tuple[int, str, str | None]:
    """(size, SHA-256 of the bytes, fingerprint for change detection; None when the file cannot be compared)."""
    import zipfile
    size = os.path.getsize(path)
    with open(path, 'rb') as fh:
        head = fh.read(8)
        fh.seek(0)
        if size <= TEXT_MAX:
            data = fh.read()
            raw = hashlib.sha256(data).hexdigest()
        else:
            h = hashlib.sha256()
            for chunk in iter(lambda: fh.read(1 << 20), b''):
                h.update(chunk)
            raw, data = h.hexdigest(), None
    if head.startswith(b'%PDF'):
        return size, raw, None
    if head.startswith(b'PK\x03\x04'):
        try:
            with zipfile.ZipFile(path) as z:
                members = sorted((i.filename, i.CRC, i.file_size) for i in z.infolist())
            return size, raw, hashlib.sha256(json.dumps(members).encode('utf-8')).hexdigest()
        except (zipfile.BadZipFile, OSError, ValueError):
            return size, raw, raw
    if data is not None and b'\0' not in data[:8192]:
        return size, raw, hashlib.sha256(_VOLATILE.sub(b'<t>', data)).hexdigest()
    return size, raw, raw


def scan(folder: str) -> Scan:
    """Every regular file in folder (symbolic links and *.part/*.tmp left out) with its size, SHA-256 and
    fingerprint; at most SCAN_FILES files and SCAN_BYTES bytes (then complete is False)."""
    out, n, total, complete = {}, 0, 0, True
    if not os.path.isdir(folder):
        return Scan({})
    for root, dirs, names in os.walk(folder, followlinks=False):
        dirs.sort()
        for name in sorted(names):
            if name.endswith(('.part', '.tmp')):
                continue
            path = os.path.join(root, name)
            try:
                if os.path.islink(path) or not os.path.isfile(path):
                    continue
                size = os.path.getsize(path)
                if n >= SCAN_FILES or total + size > SCAN_BYTES:
                    complete = False
                    continue
                entry = fingerprint(path)
            except OSError:
                continue
            n += 1
            total += size
            out[os.path.relpath(path, folder).replace(os.sep, '/')] = entry
    return Scan(out, complete)


def diff(before: Scan, after: Scan) -> dict:
    """{'added', 'changed', 'rewritten' (a PDF written again: cannot be compared), 'removed': sorted relative paths,
    'unchanged': count, 'complete': bool}. Changed means a different fingerprint (time stamps do not count)."""
    a, b = before.files, after.files
    out = {'added': [], 'changed': [], 'rewritten': [], 'removed': sorted(k for k in a if k not in b),
           'unchanged': 0, 'complete': before.complete and after.complete}
    for k in sorted(b):
        if k not in a:
            out['added'].append(k)
        elif a[k][2] is None or b[k][2] is None:
            if a[k][1] == b[k][1]:
                out['unchanged'] += 1
            else:
                out['rewritten'].append(k)
        elif a[k][2] != b[k][2]:
            out['changed'].append(k)
        else:
            out['unchanged'] += 1
    return out


def _no_change() -> dict:
    return {'added': [], 'changed': [], 'rewritten': [], 'removed': [], 'unchanged': 0, 'complete': True}


def prune(out_dir: str, previous: list, written: list) -> list[str]:
    """Remove the files the previous run wrote (previous, from the local history) that this run did not: only regular
    files inside out_dir, never through '..' or a symbolic link; empty folders left behind go too. Returns what went."""
    base = os.path.realpath(out_dir)
    keep = set(written)
    gone = []
    for rel in sorted(set(previous or ()) - keep):
        rel = str(rel).replace('\\', '/')
        parts = rel.split('/')
        if not rel or rel.startswith('/') or re.match(r'^[A-Za-z]:', rel) or any(x in ('', '.', '..') for x in parts):
            continue
        path = os.path.join(out_dir, *parts)
        if not _inside(base, os.path.realpath(path), strict=True):
            continue
        try:
            if os.path.islink(path) or not os.path.isfile(path):
                continue
            os.remove(path)
        except OSError:
            continue
        gone.append(rel)
        d = os.path.dirname(path)
        while _inside(base, os.path.realpath(d), strict=True):
            try:
                os.rmdir(d)                 # only when empty
            except OSError:
                break
            d = os.path.dirname(d)
    return gone


# =========================================================================== one run
@dataclass
class RunRecord:
    id: str
    trigger: str
    project: str = ''                  # the project's name
    project_path: str = ''             # its file (absolute)
    variant: str = ''
    variant_uid: str = ''
    out_dir: str = ''                  # absolute
    started: float = 0.0
    finished: float = 0.0
    status: str = 'running'            # running | passed | failed | error | cancelled | timeout
    message: str = ''
    unsaved: bool = False              # unsaved edits were exported (a run after changes)
    formats: list = field(default_factory=list)
    format_errors: dict = field(default_factory=dict)   # id -> message
    written: list = field(default_factory=list)          # relative paths this run's regeneration wrote
    pruned: list = field(default_factory=list)
    changed: dict = field(default_factory=_no_change)    # by the regeneration (against the folder before it)
    command_changed: dict = field(default_factory=_no_change)   # by the command (against the folder after export)
    requirements: dict | None = None   # verdict, verdict_text, exit_code, musts, shoulds, failing
    issues: dict = field(default_factory=dict)            # severity -> count, 'list' -> [{severity, message, objects}]
    metrics: dict = field(default_factory=dict)
    conventions: dict = field(default_factory=dict)       # the export frame, units and angles
    notes: list = field(default_factory=list)
    command: str = ''                  # the command that ran (kept on this computer only, never exported)
    command_state: str = 'none'        # none | off | not_approved | pending | ran
    exit_code: int | None = None       # the command's
    export_s: float = 0.0
    command_s: float = 0.0
    output: str = ''                   # the command's output (tail)
    output_n: int = 0                  # characters of output so far (the tail drops the oldest)
    summary_path: str = ''

    @staticmethod
    def new(trigger: str, project=None) -> 'RunRecord':
        t = time.time()
        rid = time.strftime('%Y%m%d-%H%M%S', time.localtime(t)) + '-' + secrets.token_hex(2)
        r = RunRecord(rid, trigger, started=t)
        if project is not None:
            r.project, r.project_path = project.name, os.path.abspath(project.path) if project.path else ''
        return r

    @property
    def seconds(self) -> float:
        return max(0.0, (self.finished or time.time()) - self.started)

    @property
    def done(self) -> bool:
        return self.status != 'running'

    def n_changed(self) -> int:
        """Files added, changed or removed by the regeneration (a rewritten PDF not counted)."""
        c = self.changed
        return len(c.get('added', ())) + len(c.get('changed', ())) + len(c.get('removed', ()))

    def to_dict(self, output_keep=OUTPUT_KEEP_HISTORY) -> dict:
        d = asdict(self)
        if len(d['output']) > output_keep:
            d['output'] = '[… earlier output not kept …]\n' + d['output'][-output_keep:]
        return d

    @staticmethod
    def from_dict(d: dict) -> 'RunRecord | None':
        if not isinstance(d, dict) or not isinstance(d.get('id'), str):
            return None
        r = RunRecord(d['id'], str(d.get('trigger', 'manual')))
        for k, v in d.items():
            if k in RunRecord.__dataclass_fields__ and k not in ('id', 'trigger'):
                setattr(r, k, v)
        for k in ('changed', 'command_changed'):
            r_ = getattr(r, k)
            setattr(r, k, {**_no_change(), **(r_ if isinstance(r_, dict) else {})})
        for k in ('written', 'pruned', 'formats', 'notes'):
            if not isinstance(getattr(r, k), list):
                setattr(r, k, [])
        if r.status == 'running':                 # the app closed during the run
            r.status, r.message = 'cancelled', r.message or 'Vantage closed during the run.'
        return r


class _Cancelled(Exception):
    pass


def exit_code_of(rec: RunRecord) -> int:
    """The batch exit code of a run: the command's own when it ran; 4 when the regeneration failed, 124 on a time
    out, 130 when stopped; without a command 5 when the requirements check fails, else 0."""
    if rec.status == 'error':
        return EXIT_ERROR
    if rec.status == 'timeout':
        return EXIT_TIMEOUT
    if rec.status == 'cancelled':
        return EXIT_CANCELLED
    if rec.command_state == 'ran' and rec.exit_code is not None:
        return int(rec.exit_code)
    if rec.requirements and rec.requirements.get('exit_code'):
        return int(rec.requirements['exit_code'])
    return 0


def command_env(rec: RunRecord) -> dict:
    """The VANTAGE_ variables the command gets (values from the project made one-line)."""
    req = rec.requirements
    return {'VANTAGE_EXPORT_DIR': rec.out_dir, 'VANTAGE_PROJECT': rec.project_path,
            'VANTAGE_PROJECT_DIR': os.path.dirname(rec.project_path) if rec.project_path else '',
            'VANTAGE_VARIANT': _plain(rec.variant), 'VANTAGE_VARIANT_UID': _plain(rec.variant_uid, 64),
            'VANTAGE_SUMMARY': rec.summary_path, 'VANTAGE_TRIGGER': rec.trigger,
            'VANTAGE_CHECK_EXIT': '' if req is None else str(int(req.get('exit_code') or 0)),
            'VANTAGE_RUN_ID': rec.id, 'VANTAGE_UNSAVED': '1' if rec.unsaved else '0'}


def run(project, engine=None, settings: AutomationSettings | None = None, variant=None, progress=None, cancel=None,
        *, command: str | None = None, command_state: str | None = None, timeout_s: float | None = None,
        trigger: str = 'manual', on_output=None, previous: RunRecord | None = None, result=None,
        out_dir: str | None = None, record: RunRecord | None = None, unsaved: bool = False) -> RunRecord:
    """Regenerate the exports of project (its file must be saved: the folder is relative to it) and, when command is
    given, run it afterwards. command must come from the caller's trusted source (the user's approved command, a
    command line); this function never reads one from the project. previous: the last run into the same folder (its
    written files are pruned); result: an analysis result of the variant to reuse; out_dir: an absolute folder that
    overrides the settings' (batch --export). Never raises: failures are in the record's status and message."""
    st = settings or settings_of(project)
    rec = record or RunRecord.new(trigger, project)
    rec.unsaved = bool(unsaved)
    cancel = cancel or (lambda: False)
    progress = progress or (lambda f, text='': None)
    rec.command_state = command_state or ('pending' if command else 'none')
    keep_active = project.active_uid
    after = None
    try:
        after = _regenerate(project, engine, st, variant, progress, cancel, rec, previous, result, out_dir)
    except _Cancelled:
        rec.status, rec.message = 'cancelled', 'Stopped before the files were all written.'
    except Exception as e:                          # noqa: BLE001  one bad run must not take the app down
        rec.status, rec.message = 'error', str(e) if isinstance(e, ValueError) else f'{type(e).__name__}: {e}'
    finally:
        project.active_uid = keep_active
    rec.export_s = time.time() - rec.started
    if command and rec.status == 'running':
        progress(0.96, 'running your command')
        rec.command_state = 'ran'
        _run_command(rec, command, timeout_s, cancel, on_output, after)
    elif command:
        rec.command_state = 'skipped'
    if rec.status == 'running':
        if rec.command_state == 'ran':
            rec.status = 'passed' if rec.exit_code == 0 else 'failed'
        else:
            rec.status = 'failed' if rec.requirements and rec.requirements.get('exit_code') else 'passed'
    rec.message = rec.message or _message(rec)
    rec.finished = time.time()
    progress(1.0, '')
    return rec


def _regenerate(project, engine, st, variant, progress, cancel, rec, previous, result, out_dir):
    from . import exporters as ex
    from . import rules
    if not project.path and not out_dir:
        raise ValueError('Save the project first: the export folder is relative to the project file.')
    out = os.path.abspath(out_dir) if out_dir else resolve_out_dir(project.path, st.out_dir)
    rec.out_dir = out
    if project.path:
        rec.project_path = os.path.abspath(project.path)
    rec.project = project.name
    # ---- the variant
    v = variant if hasattr(variant, 'uid') else pick_variant(project, variant or st.variant)
    if v is None and (variant or st.variant):
        rec.notes.append(f'The variant "{_plain(variant or st.variant, 64)}" is not in the project: the active one, '
                         f'"{_plain(project.variant.name, 64)}", is exported.')
    v = v or project.variant
    project.active_uid = v.uid
    rec.variant, rec.variant_uid = v.name, v.uid
    os.makedirs(out, exist_ok=True)
    if not out_dir:
        check_links(out, os.path.dirname(os.path.abspath(project.path)))
    before = scan(out)
    if cancel():
        raise _Cancelled
    # ---- the analysis (exports, the Issues list in the summary)
    progress(0.02, 'analysing')
    eng = engine
    if eng is None:
        from .analysis import Engine
        eng = Engine()
    res = result if result is not None and getattr(result, 'variant_uid', None) == v.uid else eng.analyse(project)
    vg = eng.vehicle_geom(project)
    mounts = {}
    for c in v.cameras:
        try:
            mounts[c.uid] = rules.mount_info(project, c, vg)
        except Exception:                          # noqa: BLE001  as batch mode: a camera off the mesh has none
            pass
    try:
        issues = rules.collect_issues(project, res, vg, mounts)
    except Exception as e:                          # noqa: BLE001
        issues = []
        rec.notes.append(f'The Issues list could not be computed: {e}')
    rec.issues = {s: sum(1 for i in issues if i.severity == s) for s in ('critical', 'serious', 'warning')}
    rec.issues['list'] = [{'severity': i.severity, 'message': i.message, 'objects': i.objects} for i in issues[:200]]
    if res is not None:
        m = res.metrics
        rec.metrics = {'blind_m2': _r(m.get('blind_m2')), 'covered_pct': _r(m.get('covered_pct'), 2),
                       'min_density_px_cm': _r(m.get('min_density'), 2), 'cell_m': _r(m.get('cell'), 3)}
    if cancel():
        raise _Cancelled
    # ---- the exports, one format at a time (a failing format is named; the command then does not run)
    fmts = [f for f in st.formats if not (st.check and f == 'requirements')]
    rec.formats = list(fmts)
    frame = st.frame or project.frame
    rec.conventions = {'frame': frame, 'frame_text': f'vehicle, ISO 8855 (x forward, y left, z up), origin: '
                                                     f'{frame.replace("_", " ")}',
                       'units': st.units, 'angles': st.angles}
    opts = ex.ExportOptions(out_dir=out, frame=frame, units=st.units, angles=st.angles, issues=issues, engine=eng)
    written = []
    for i, fid in enumerate(fmts):
        progress(0.25 + 0.5 * i / max(1, len(fmts)), f'exporting {fid}')
        try:
            paths = ex.export([fid], project, res, opts)
        except Exception as e:                      # noqa: BLE001
            rec.format_errors[fid] = f'{type(e).__name__}: {e}' if not isinstance(e, ValueError) else str(e)
            continue
        written += [os.path.relpath(p, out).replace(os.sep, '/') for p in paths]
        if cancel():
            raise _Cancelled
    # ---- the requirements check
    if st.check:
        from . import requirements as RQ
        progress(0.76, 'checking the requirements')
        span = lambda f, text='': progress(0.76 + 0.16 * min(max(f, 0.0), 1.0), text or 'checking the requirements')  # noqa: E731
        rres = RQ.run(project, eng, variant=v.uid if (variant or st.variant) else None, progress=span, cancel=cancel)
        if rres is None:
            raise _Cancelled
        paths = RQ.export(rres, project, os.path.join(out, REQUIREMENTS_DIR))
        written += [os.path.relpath(p, out).replace(os.sep, '/') for p in paths]
        c = rres.counts()
        rec.requirements = {'verdict': rres.verdict, 'verdict_text': RQ.VERDICT_TEXT[rres.verdict],
                            'exit_code': RQ.EXIT_FAIL if rres.verdict in ('fail', 'incomplete') else 0,
                            'variant': rres.variant,
                            'musts': f'{c["must"]["pass"]}/{c["must"]["total"]}',
                            'shoulds': f'{c["should"]["pass"]}/{c["should"]["total"]}',
                            'failing': [r.item['id'] for r in rres.items
                                        if r.item['priority'] == 'must' and r.status != 'pass']}
    # ---- files the previous run wrote and this one did not (not after a failed format: nothing is final then)
    rec.written = sorted(set(written)) + [SUMMARY_NAME]
    if rec.format_errors:
        rec.status = 'error'
        rec.message = 'Could not export ' + '; '.join(f'{k}: {m}' for k, m in rec.format_errors.items())
    elif st.prune and previous is not None and previous.out_dir and norm_path(previous.out_dir) == norm_path(out):
        rec.pruned = prune(out, previous.written, rec.written)
    # ---- what changed (the summary itself, new every run, not counted), then the summary the command reads
    progress(0.94, 'writing the summary')
    mid = scan(out)
    rec.changed = diff(_without_summary(before), _without_summary(mid))
    rec.summary_path = os.path.join(out, SUMMARY_NAME)
    rec.export_s = time.time() - rec.started
    data = (json.dumps(to_doc(rec, project, mid), indent=1, ensure_ascii=False) + '\n').encode('utf-8')
    tmp = rec.summary_path + '.part'
    with open(tmp, 'wb') as fh:
        fh.write(data)
    os.replace(tmp, rec.summary_path)
    mid.files[SUMMARY_NAME] = fingerprint(rec.summary_path)      # as the scan after the command will see it
    return mid


def _without_summary(sc: Scan) -> Scan:
    return Scan({k: v for k, v in sc.files.items() if k != SUMMARY_NAME}, sc.complete)


def _r(v, d=3):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return round(v, d) if math.isfinite(v) else None


def _kill(proc):
    """Stop the command and everything it started (its own process group / session)."""
    if proc.poll() is not None:
        return
    try:
        if os.name == 'posix':
            os.killpg(proc.pid, signal.SIGTERM)
        else:
            subprocess.run(['taskkill', '/F', '/T', '/PID', str(proc.pid)], capture_output=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        proc.wait(3)
    except subprocess.TimeoutExpired:
        try:
            if os.name == 'posix':
                os.killpg(proc.pid, signal.SIGKILL)
            else:
                proc.kill()
        except OSError:
            pass
        try:
            proc.wait(3)
        except subprocess.TimeoutExpired:
            pass


def _run_command(rec: RunRecord, command: str, timeout_s, cancel, on_output, out_before: Scan | None = None):
    """Run the command through the system shell in the project's folder, the output (stdout and stderr together)
    streamed to on_output in chunks and kept in rec.output (tail); stopped (with what it started) on cancel or after
    timeout_s. Fills exit_code, status (timeout / cancelled), command_s and command_changed."""
    from ..sysenv import system_env
    from .credentials import mask
    env = system_env()
    env.update(command_env(rec))
    cwd = os.path.dirname(rec.project_path) if rec.project_path else rec.out_dir
    rec.command = command
    lock = threading.Lock()
    chunks: list[str] = []
    out_before = out_before if out_before is not None else scan(rec.out_dir)
    t0 = time.time()

    def emit(text):
        text = mask(text)
        rec.output = (rec.output + text)[-OUTPUT_KEEP:]
        rec.output_n += len(text)
        if on_output is not None:
            try:
                on_output(text)
            except Exception:                       # noqa: BLE001  a closed page must not stop the command
                pass

    def flush():
        with lock:
            text = ''.join(chunks)
            chunks.clear()
        if text:
            emit(text)

    kw = {'start_new_session': True} if os.name == 'posix' else \
        {'creationflags': getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0)}
    try:
        proc = subprocess.Popen(command, shell=True, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **kw)
    except OSError as e:
        rec.status, rec.message = 'error', f'The command could not start: {e}'
        rec.command_s = time.time() - t0
        return

    def reader():
        for line in iter(proc.stdout.readline, b''):
            with lock:
                chunks.append(line.decode('utf-8', errors='replace'))
        proc.stdout.close()
    th = threading.Thread(target=reader, daemon=True)
    th.start()
    deadline = t0 + timeout_s if timeout_s and timeout_s > 0 else None
    try:
        while proc.poll() is None:
            if cancel():
                rec.status = 'cancelled'
                break
            if deadline is not None and time.time() > deadline:
                rec.status = 'timeout'
                break
            time.sleep(0.05)
            flush()
    finally:
        if proc.poll() is None:                     # stopped, timed out, or the watch was interrupted
            if rec.status == 'running':
                rec.status = 'cancelled'
            _kill(proc)
        th.join(5)
        flush()
    rc = proc.returncode
    if rec.status == 'running':
        rec.exit_code = 128 - rc if rc is not None and rc < 0 else rc      # killed by signal N: 128 + N, as a shell
    elif rec.status == 'timeout':
        rec.message = f'The command ran over its {timeout_s / 60:g} min limit and was stopped.'
        emit(f'\n[Vantage stopped the command after {timeout_s / 60:g} min]\n')
    else:
        rec.message = 'The command was stopped.'
        emit('\n[Vantage stopped the command]\n')
    rec.command_s = time.time() - t0
    rec.command_changed = diff(out_before, scan(rec.out_dir))


def _message(rec: RunRecord) -> str:
    c, files = rec.changed, len(rec.written)
    parts = []
    if rec.status != 'error':
        what = [f'{len(c[k])} {w}' for k, w in (('added', 'new'), ('changed', 'changed'), ('removed', 'removed'))
                if c.get(k)]
        parts.append(f'{files} file{"s" if files != 1 else ""} written' + (': ' + ', '.join(what) if what else
                                                                            ', nothing changed'))
    if rec.requirements:
        parts.append(f'requirements: {rec.requirements["verdict_text"].lower()}')
    if rec.command_state == 'ran' and rec.exit_code is not None:
        parts.append(f'command exit {rec.exit_code}')
    elif rec.command_state in ('not_approved', 'off'):
        parts.append(COMMAND_STATE_TEXT[rec.command_state])
    text = '; '.join(parts)
    return text[:1].upper() + text[1:] + '.' if text else ''


# =========================================================================== output
def summary(rec: RunRecord) -> dict:
    """Compact figures of a run (the assistant, the batch log): never the command's text or its output."""
    c = rec.changed
    return {'status': rec.status, 'status_text': STATUS_LABEL.get(rec.status, rec.status), 'trigger': rec.trigger,
            'when': _now_iso(rec.started), 'seconds': _r(rec.seconds, 1), 'variant': rec.variant,
            'export_folder': rec.out_dir, 'files_written': len(rec.written), 'files_changed': rec.n_changed(),
            'changed_files': (c['added'] + c['changed'])[:20], 'removed_files': c['removed'][:20],
            'rewritten_documents': len(c.get('rewritten', ())),
            'format_errors': dict(rec.format_errors),
            'requirements': (rec.requirements or {}).get('verdict_text'),
            'issues': {k: v for k, v in rec.issues.items() if k != 'list'},
            'command': COMMAND_STATE_TEXT.get(rec.command_state, rec.command_state), 'exit_code': rec.exit_code,
            'command_seconds': _r(rec.command_s, 1) if rec.command_state == 'ran' else None,
            'exit_code_batch': exit_code_of(rec), 'message': rec.message, 'notes': list(rec.notes)}


def to_doc(rec: RunRecord, project=None, out_scan: Scan | None = None) -> dict:
    """A run as JSON (vantage_summary.json, written before the command runs, and a run's export): the project,
    variant, folder, formats, the files written (with SHA-256 when out_scan is given) and what changed, the
    requirements check, the Issues list, the environment the command gets. Never the command's text."""
    from .. import __version__
    files = [{'path': p, **({'bytes': out_scan.files[p][0], 'sha256': out_scan.files[p][1]}
                            if out_scan is not None and p in out_scan.files else {})} for p in rec.written]
    return {
        'schema': SCHEMA, 'vantage': __version__,
        'project': rec.project or getattr(project, 'name', ''), 'project_file': rec.project_path,
        'variant': rec.variant, 'variant_uid': rec.variant_uid, 'run_id': rec.id, 'trigger': rec.trigger,
        'started': _now_iso(rec.started), 'unsaved_edits': rec.unsaved,
        'conventions': dict(rec.conventions),
        'export_folder': rec.out_dir, 'formats': list(rec.formats), 'format_errors': dict(rec.format_errors),
        'files': files, 'pruned': list(rec.pruned),
        'changed': {k: v for k, v in rec.changed.items()},
        'requirements': rec.requirements,
        'issues': {k: v for k, v in rec.issues.items()},
        'metrics': dict(rec.metrics),
        'command': {'state': rec.command_state, 'exit_code': rec.exit_code,
                    'seconds': _r(rec.command_s, 3) if rec.command_state == 'ran' else None,
                    'changed': rec.command_changed if rec.command_state == 'ran' else None},
        'status': rec.status, 'message': rec.message, 'notes': list(rec.notes),
        'environment': {k: v for k, v in command_env(rec).items()},
        'assumptions': ASSUMPTIONS,
    }


def files(rec: RunRecord, project=None) -> list:
    """[(name, bytes)]: automation_run.json (the run, without the command's text) and automation_run.log (its
    output)."""
    doc = to_doc(rec, project)
    doc['seconds'] = _r(rec.seconds, 3)
    doc['exit_code_batch'] = exit_code_of(rec)
    head = (f'# Vantage automation run {rec.id} · {TRIGGER_LABEL.get(rec.trigger, rec.trigger)} · '
            f'{_now_iso(rec.started)} · {STATUS_LABEL.get(rec.status, rec.status)}'
            + (f' · exit {rec.exit_code}' if rec.exit_code is not None else '') + '\n')
    return [('automation_run.json', (json.dumps(doc, indent=1, ensure_ascii=False) + '\n').encode('utf-8')),
            ('automation_run.log', (head + rec.output).encode('utf-8'))]


def export(rec: RunRecord, project, out_dir) -> list[str]:
    """Write automation_run.json and automation_run.log into out_dir; returns their paths."""
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for name, data in files(rec, project):
        path = os.path.join(out_dir, name)
        with open(path, 'wb') as fh:
            fh.write(data)
        paths.append(path)
    return paths


# =========================================================================== the history (this computer)
class History:
    """The last runs per project file, in the user's config folder (automation-history.json, readable only by the
    user). Read again only when the file changed."""

    def __init__(self, path: str | None = None):
        self.path = path or os.path.join(_config_dir(), 'automation-history.json')
        self._cache = (None, {})
        self._lock = threading.Lock()

    def _load(self) -> dict:
        try:
            st = os.stat(self.path)
            key = (st.st_mtime_ns, st.st_size)
        except OSError:
            return {}
        if self._cache[0] == key:
            return self._cache[1]
        try:
            with open(self.path, encoding='utf-8') as fh:
                d = json.load(fh)
            projects = d.get('projects') if isinstance(d, dict) else None
            projects = projects if isinstance(projects, dict) else {}
        except (OSError, ValueError):
            projects = {}
        self._cache = (key, projects)
        return projects

    def runs(self, project_path: str) -> list[RunRecord]:
        """Newest first."""
        if not project_path:
            return []
        with self._lock:
            entry = self._load().get(norm_path(project_path))
        rows = entry.get('runs') if isinstance(entry, dict) else None
        out = [RunRecord.from_dict(d) for d in (rows if isinstance(rows, list) else [])]
        return [r for r in reversed(out) if r is not None]

    def last(self, project_path: str, out_dir: str | None = None, complete=False) -> RunRecord | None:
        """The newest run (into out_dir when given; with complete, only one whose regeneration finished: what the
        next run prunes against)."""
        for r in self.runs(project_path):
            if complete and r.status not in ('passed', 'failed'):
                continue
            if out_dir is None or (r.out_dir and norm_path(r.out_dir) == norm_path(out_dir)):
                return r
        return None

    def add(self, rec: RunRecord):
        if not rec.project_path:
            return
        with self._lock:
            projects = dict(self._load())
            key = norm_path(rec.project_path)
            entry = projects.get(key) if isinstance(projects.get(key), dict) else {'path': rec.project_path}
            rows = [d for d in entry.get('runs', []) if isinstance(d, dict) and d.get('id') != rec.id]
            rows.append(rec.to_dict())
            entry = {'path': rec.project_path, 'runs': rows[-HISTORY_KEEP:], 'last': rec.started}
            projects[key] = entry
            if len(projects) > HISTORY_PROJECTS:            # forget the projects not run for longest
                for k in sorted(projects, key=lambda k: projects[k].get('last', 0))[:len(projects) - HISTORY_PROJECTS]:
                    projects.pop(k, None)
            _write_private(self.path, (json.dumps({'schema': HISTORY_SCHEMA, 'projects': projects}) + '\n')
                           .encode('utf-8'))

    def clear(self, project_path: str):
        with self._lock:
            projects = dict(self._load())
            if projects.pop(norm_path(project_path), None) is not None:
                _write_private(self.path, (json.dumps({'schema': HISTORY_SCHEMA, 'projects': projects}) + '\n')
                               .encode('utf-8'))


_HISTORY: History | None = None


def default_history() -> History:
    """The history in the current config folder (a new one when XDG_CONFIG_HOME / APPDATA changed)."""
    global _HISTORY
    path = os.path.join(_config_dir(), 'automation-history.json')
    if _HISTORY is None or _HISTORY.path != path:
        _HISTORY = History(path)
    return _HISTORY


# =========================================================================== the Issues list
def issues(project, vg=None, history: History | None = None) -> list:
    """Cheap (settings and a cached file read): stored automation settings Vantage had to drop, and the last run on
    this computer failing (its command's exit code, a failed export, a time out). Nothing for a project without
    automation settings."""
    from . import rules as R
    raw = getattr(project.analysis, 'automation', None)
    if not raw:
        return []
    out = [R.Issue('warning', f'Automation: {p}.', 'Automation', []) for p in settings_problems(raw)]
    st = AutomationSettings.from_dict(raw)
    if st.enabled and project.path:
        last = (history or default_history()).last(project.path)
        if last is not None and last.status in ('failed', 'error', 'timeout'):
            when = time.strftime('%d %b %H:%M', time.localtime(last.started))
            what = (f'its command exited with {last.exit_code}' if last.status == 'failed' and last.command_state == 'ran'
                    else 'the requirements check failed' if last.status == 'failed'
                    else 'it ran over its time limit' if last.status == 'timeout' else last.message.rstrip('.'))
            out.append(R.Issue('warning', f'The last automated run ({TRIGGER_LABEL.get(last.trigger, last.trigger)}, '
                                          f'{when}) failed: {what}. See Automation.', 'Automation', []))
    return out


def overview(project, history: History | None = None, commands: CommandStore | None = None) -> dict:
    """The automation of a project for the assistant: settings, whether this computer has a command set and approved
    (never its text), and the last runs' summaries."""
    st = settings_of(project)
    cfg = commands.get(project.path) if commands is not None and project.path else None
    runs = (history or default_history()).runs(project.path) if project.path else []
    return {'settings': {k: v for k, v in st.to_dict().items()},
            'trigger': TRIGGER_LABEL[st.trigger] if st.enabled else 'off',
            'saved': bool(project.path),
            'command': ('not set' if cfg is None else 'set and approved' if commands.is_approved(project.path)
                        else 'set, not approved yet') if commands is not None else 'unknown',
            'last_runs': [summary(r) for r in runs[:5]],
            'meaning': 'what Vantage regenerates when the design changes (formats, folder, requirements check) and '
                       'the outcome of the user\'s own command afterwards; the command itself lives only in the '
                       'user\'s settings and is never shown or run by the assistant'}


# =========================================================================== headless: watch a project file
class Debouncer:
    """Fires once `quiet` seconds after the last poke (clock injectable for tests)."""

    def __init__(self, quiet: float, clock=time.monotonic):
        self.quiet, self.clock = float(quiet), clock
        self._t = None

    def poke(self):
        self._t = self.clock()

    @property
    def pending(self) -> bool:
        return self._t is not None

    def ready(self) -> bool:
        """True once when the quiet period has passed since the last poke."""
        if self._t is not None and self.clock() - self._t >= self.quiet:
            self._t = None
            return True
        return False


def file_signature(path: str):
    """(mtime_ns, size) of a file, None when it is missing."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size)


def _file_hash(path: str) -> str | None:
    try:
        h = hashlib.sha256()
        with open(path, 'rb') as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b''):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


_APP = None


def ensure_app():
    """A QApplication (offscreen unless set otherwise) with Vantage's fonts: the PDF exports draw with Qt. Kept alive
    here (an application nobody holds is destroyed at once)."""
    global _APP
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = _APP = QApplication(sys.argv[:1])
    from ..ui.theme import load_fonts
    load_fonts()
    return app


def watch(path: str, command: str | None = None, *, out_dir: str | None = None, formats=None, variant=None,
          interval: float = 1.0, quiet: float = 1.0, runs: int = 0, timeout_s: float | None = None,
          log=None, out=None, stop=None) -> int:
    """Watch a project file: regenerate (the project's automation settings; out_dir, formats and variant override
    them) and run command after every change, once the file has been quiet for `quiet` seconds; also once at the start.
    Ends after `runs` runs (0: never) or when stop() is true (or on Ctrl+C / SIGTERM through KeyboardInterrupt).
    Returns the last run's exit code (exit_code_of: the command's own when it ran). command comes from the command line,
    never from the project."""
    from .analysis import Engine
    from .project import Project
    path = os.path.abspath(path)
    log = log or (lambda s: print(f'vantage: {s}', file=sys.stderr, flush=True))
    out = out or (lambda s: (sys.stdout.write(s), sys.stdout.flush()))
    stop = stop or (lambda: False)
    ensure_app()
    eng = Engine()
    previous: RunRecord | None = None
    last_code, n, last_hash = 0, 0, None
    deb = Debouncer(quiet)
    sig = file_signature(path)
    due = True                                         # once at the start
    log(f'watching {path}' + (f'; on change: {command}' if command else '; no command: regenerating only'))
    try:
        while not stop():
            if due or deb.ready():
                due = False
                cur = file_signature(path)
                h = _file_hash(path) if cur is not None else None
                if cur is None:
                    log(f'{os.path.basename(path)} is missing: waiting for it')
                elif h == last_hash:
                    pass                                # touched, not changed
                else:
                    last_hash = h
                    rec, code = _watch_once(path, command, eng, out_dir, formats, variant, timeout_s, previous, out,
                                            log, Project)
                    n += 1
                    last_code = code
                    if rec is not None and rec.status in ('passed', 'failed'):
                        previous = rec
                    if runs and n >= runs:
                        break
            if stop():
                break
            time.sleep(max(0.05, float(interval)))
            s = file_signature(path)
            if s != sig:
                sig = s
                deb.poke()
    except KeyboardInterrupt:
        log('stopped')
    return last_code


def _watch_once(path, command, eng, out_dir, formats, variant, timeout_s, previous, out, log, Project):
    try:
        project = Project.load(path)
    except Exception as e:                              # noqa: BLE001  half-written, not a project, unreadable
        log(f'cannot read {os.path.basename(path)}: {e}')
        return None, EXIT_ERROR
    st = settings_of(project)
    if formats is not None:
        st = AutomationSettings.from_dict(dict(st.to_dict(), formats=list(formats)))
        unknown = [f for f in formats if f not in st.formats]
        if unknown:
            log('skipping unknown or window-only formats: ' + ', '.join(unknown))
    rec = run(project, eng, st, variant, command=command, trigger='watch', on_output=out, previous=previous,
              out_dir=out_dir, timeout_s=timeout_s)
    code = exit_code_of(rec)
    folder = rec.out_dir
    try:
        folder = os.path.relpath(rec.out_dir) if rec.out_dir else ''
    except ValueError:
        pass
    log(f'run {rec.id}: {STATUS_LABEL.get(rec.status, rec.status).lower()} · {rec.message} · {folder} · '
        f'{rec.seconds:.1f} s · exit {code}')
    for note in rec.notes:
        log(f'  note: {note}')
    return rec, code


# =========================================================================== CI templates
def repo_root(path: str) -> str | None:
    """The repository a file lies in (the nearest folder above it holding .git), else None."""
    d = os.path.dirname(os.path.abspath(path))
    while True:
        if os.path.exists(os.path.join(d, '.git')):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            return None
        d = parent


def _yq(s: str) -> str:
    """A YAML double-quoted scalar (JSON strings are valid YAML)."""
    return json.dumps(str(s), ensure_ascii=False)


def ci_files(project, settings: AutomationSettings | None = None, version: str | None = None,
             root: str | None = None, now: float | None = None) -> list:
    """[(name, bytes)]: a GitHub Actions workflow and a GitLab CI job that install this Vantage version, export the
    calibration files (the project's automation formats) and run the requirements check on the project in its
    repository, keeping both as artifacts; and a README saying where they go. The project's path is taken relative to
    its repository (the nearest folder above it with .git; root overrides), else assumed at the repository root."""
    from .. import __version__
    st = settings or settings_of(project)
    version = version or __version__
    path = os.path.abspath(project.path) if project.path else f'{project.name or "project"}.vantage'
    root = root or (repo_root(path) if project.path else None)
    rel = os.path.relpath(path, root).replace(os.sep, '/') if root else os.path.basename(path)
    if '${{' in rel or '\n' in rel:
        raise ValueError('The project\'s path cannot be put in a CI file: rename the file or its folder.')
    v = pick_variant(project, st.variant) if st.variant else None
    vflag = ''
    if v is not None:
        idx = project.variants.index(v) + 1
        vflag = f' --variant {v.uid}' if re.fullmatch(r'[A-Za-z0-9_-]{1,64}', v.uid) else f' --variant {idx}'
    conv = ''.join([f' --frame {st.frame}' if st.frame else '', ' --units mm' if st.units == 'mm' else '',
                    ' --angles rad' if st.angles == 'rad' else ''])
    fmts = ','.join(st.formats)
    stamp = _dt.datetime.fromtimestamp(time.time() if now is None else now).strftime('%Y-%m-%d')
    pip = f'vantage @ git+https://github.com/Sherin-SEF-AI/Vantage@v{version}'
    libs = 'libegl1 libgl1 libxkbcommon0 libfontconfig1 libdbus-1-3 libglib2.0-0'
    vname = _plain(v.name if v is not None else project.variant.name, 80)
    export_cmd = f'vantage "$VANTAGE_PROJECT"{vflag}{conv} --export "$VANTAGE_OUT/calibration" --formats {fmts}'
    check_cmd = f'vantage "$VANTAGE_PROJECT"{vflag} --check "$VANTAGE_OUT/review"'
    head = [f'# Vantage design checks for {_plain(project.name, 80)} ({rel}), generated by Vantage {version} on {stamp}.',
            f'# Variant: {vname}' + ('' if v is not None else ' (the project\'s active variant as saved)') + '.',
            '# It installs Vantage, '
            + ('exports the calibration files (exit 2 fails the job when the plan has critical issues)'
               if fmts else 'exports nothing (no formats chosen)')
            + (', runs the requirements check (exit 5 fails it when a must requirement fails or cannot be evaluated)'
               if st.check else '')
            + ' and keeps the files as artifacts.',
            '# Batch mode reads the project; it never runs anything the project names. No display needed (offscreen).']
    # ---------------------------------------------------------------- GitHub Actions
    gh = head[:1] + ['# Put it in the repository as .github/workflows/vantage.yml.'] + head[1:] + [
        '', 'name: Vantage design checks', '',
        'on:', '  push:', '    paths:', f'      - {_yq(rel)}', '  pull_request:', '    paths:', f'      - {_yq(rel)}',
        '  workflow_dispatch:', '',
        'permissions:', '  contents: read', '',
        'jobs:', '  vantage:', '    runs-on: ubuntu-22.04', '    timeout-minutes: 60', '    env:',
        '      QT_QPA_PLATFORM: offscreen', f'      VANTAGE_PROJECT: {_yq(rel)}', '      VANTAGE_OUT: vantage-out',
        '    steps:', '      - uses: actions/checkout@v7', '', '      - uses: actions/setup-python@v7', '        with:',
        '          python-version: "3.12"', '',
        '      - name: Qt libraries for batch mode', '        run: |', '          sudo apt-get update',
        f'          sudo apt-get install -y --no-install-recommends {libs}', '',
        f'      - name: Install Vantage {version}', '        id: install',
        f'        run: python -m pip install {_yq(pip)}', '']
    if fmts:
        gh += ['      - name: Calibration files', f'        run: {export_cmd}', '']
    if st.check:
        gh += ['      - name: Requirements check', "        if: ${{ !cancelled() && steps.install.outcome == 'success' }}",
               f'        run: {check_cmd}', '']
    if fmts:
        gh += ['      - name: Keep the calibration files', '        if: ${{ !cancelled() }}',
               '        uses: actions/upload-artifact@v7', '        with:', '          name: vantage-calibration',
               '          path: vantage-out/calibration', '          if-no-files-found: warn', '']
    if st.check:
        gh += ['      - name: Keep the review pack', '        if: ${{ !cancelled() }}',
               '        uses: actions/upload-artifact@v7', '        with:', '          name: vantage-review',
               '          path: vantage-out/review', '          if-no-files-found: warn', '']
    # ---------------------------------------------------------------- GitLab CI
    gl_rel = rel.replace('$', '$$')                    # GitLab expands $ in variables
    script = ['rc=0']
    if fmts:
        script.append(f'{export_cmd} || rc=$?')
    if st.check:
        script.append(f'{check_cmd} || {{ c=$?; if [ "$c" -gt "$rc" ]; then rc=$c; fi; }}')
    script.append('exit "$rc"')
    gl = head[:1] + ['# Include it from your .gitlab-ci.yml (include: - local: vantage.gitlab-ci.yml) or use it as '
                     'the .gitlab-ci.yml.'] + head[1:] + [
        '', 'vantage-checks:', '  image: python:3.12-slim', '  variables:', '    QT_QPA_PLATFORM: offscreen',
        f'    VANTAGE_PROJECT: {_yq(gl_rel)}', '    VANTAGE_OUT: vantage-out',
        '    PIP_DISABLE_PIP_VERSION_CHECK: "1"', '  rules:', '    - if: $CI_PIPELINE_SOURCE == "web"',
        '    - changes:', f'        - {_yq(rel)}',
        '  before_script:', '    - apt-get update',
        f'    - apt-get install -y --no-install-recommends git {libs}',
        f'    - python -m pip install {_yq(pip)}', '  script:'] + [f'    - {_yq(s)}' for s in script] + [
        '  artifacts:', '    when: always', '    name: vantage-calibration', '    paths:', '      - vantage-out/',
        '    expire_in: 30 days', '']
    where = (f'`{rel}` relative to the repository at `{root}`' if root else
             f'`{rel}`: no repository (a folder with `.git`) was found above the project, so the templates assume the '
             'project at the repository root; change VANTAGE_PROJECT and the paths if it lies elsewhere')
    readme = '\n'.join([
        f'# Vantage CI templates for {_plain(project.name, 80)}', '',
        f'Generated by Vantage {version} on {stamp} for {where}.', '',
        '- **GitHub Actions**: copy `github/vantage.yml` to `.github/workflows/vantage.yml`. It runs on a push or pull '
        'request that changes the project, and by hand (Actions ▸ Vantage design checks ▸ Run workflow).',
        '- **GitLab CI**: copy `gitlab/vantage.gitlab-ci.yml` into the repository and add '
        '`include: - local: vantage.gitlab-ci.yml` to `.gitlab-ci.yml` (or use it as `.gitlab-ci.yml`). It runs when '
        'the project changes and when a pipeline is started by hand.', '',
        'What the job does:', '',
        f'1. Installs Vantage {version} from its GitHub tag with the Qt libraries batch mode needs (no display).',
        (f'2. Exports the calibration files ({", ".join(st.formats)}) into `vantage-out/calibration`; exit 2 fails '
         'the job when the plan has critical issues.' if fmts else '2. Exports nothing: no formats are chosen on the '
                                                                   'Automation page.'),
        ('3. Runs the requirements check into `vantage-out/review` (requirements.json, requirements.csv, '
         'review_pack.pdf); exit 5 fails the job when a must requirement fails or cannot be evaluated.' if st.check
         else '3. The requirements check is off on the Automation page, so the job does not run it.'),
        '4. Keeps both folders as artifacts, also when a step failed.', '',
        f'Variant: {vname}' + (f' (`--variant {vflag.split()[-1]}`).' if vflag else ' (the active variant as saved).'),
        '', 'The templates run Vantage\'s batch mode on the project; neither they nor Vantage run anything the project '
        'file names. Your own commands (the Automation page) stay on your computer.', ''])
    return [('ci/github/vantage.yml', ('\n'.join(gh)).encode('utf-8')),
            ('ci/gitlab/vantage.gitlab-ci.yml', ('\n'.join(gl)).encode('utf-8')),
            ('ci/README.md', readme.encode('utf-8'))]


def write_ci(project, out_dir: str, settings: AutomationSettings | None = None) -> list[str]:
    """Write the CI templates into out_dir (ci/…); returns their paths."""
    paths = []
    for name, data in ci_files(project, settings):
        path = os.path.join(out_dir, *name.split('/'))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as fh:
            fh.write(data)
        paths.append(path)
    return paths


# =========================================================================== batch mode
def add_arguments(g):
    """The batch options (added to batch mode's group)."""
    g.add_argument('--watch', metavar='PROJECT.vantage',
                   help='watch a project file: on every change (and once at the start) regenerate the exports of its '
                        'automation settings (Export ▸ Automation) and run --on-change; exits with the last run\'s '
                        'code (the command\'s own when one ran)')
    g.add_argument('--on-change', metavar='CMD',
                   help='with --watch: a command to run after each regeneration, through the shell in the project\'s '
                        'folder, with VANTAGE_EXPORT_DIR, VANTAGE_PROJECT, VANTAGE_VARIANT, VANTAGE_SUMMARY … set')
    g.add_argument('--watch-runs', type=int, default=0, metavar='N',
                   help='with --watch: stop after N runs (1: regenerate and run once, for CI); default: until stopped')
    g.add_argument('--watch-interval', type=float, default=1.0, metavar='S',
                   help='with --watch: how often to look at the file, seconds (default 1)')
    g.add_argument('--ci-templates', metavar='DIR',
                   help='write a GitHub Actions workflow and a GitLab CI job that run the batch checks on the project '
                        'in its repository (DIR/ci/…)')


def watch_cli(args) -> int:
    """batch --watch: the loop, with SIGTERM ending it like Ctrl+C. --export overrides the folder, --formats the
    formats, --variant the variant."""
    if not str(args.watch).lower().endswith('.vantage'):
        raise SystemExit('--watch needs a .vantage project file.')
    if not os.path.exists(args.watch):
        raise SystemExit(f'--watch: {args.watch} does not exist.')
    try:
        old = signal.signal(signal.SIGTERM, _raise_interrupt)
    except (ValueError, OSError, AttributeError):        # not the main thread, or no SIGTERM here
        old = None
    fmts = [f.strip() for f in args.formats.split(',') if f.strip()] if getattr(args, 'formats', None) else None
    try:
        return watch(args.watch, getattr(args, 'on_change', None) or None,
                     out_dir=getattr(args, 'export', None) or None, formats=fmts,
                     variant=getattr(args, 'variant', None) or None,
                     interval=max(0.1, float(getattr(args, 'watch_interval', 1.0) or 1.0)),
                     runs=max(0, int(getattr(args, 'watch_runs', 0) or 0)))
    finally:
        if old is not None:
            signal.signal(signal.SIGTERM, old)


def _raise_interrupt(*_):
    raise KeyboardInterrupt
