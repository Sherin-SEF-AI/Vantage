"""run_batch: Claude runs any of Vantage's batch jobs on the project as it is now (MCP tool, vantage.mcp.service).

The project (the window's, or the file's, with its unsaved changes) is saved to a run folder, and batch mode
(vantage.batch, the same code as `vantage project.vantage --perception-range …`) runs on it in a process of its own,
in that folder: relative output paths land there. The reply has the exit code and what it means, the end of the
output, every file written and the content of the small JSON ones.

Allowed: every batch option except those that start CARLA or a live CARLA rig (a Copilot plan the user approves in
the window runs those), download a model, run the user's automation command or its watchers, or choose another
project (--sample, --template: the run is always on this project)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

DENY = {'--carla-run': 'starts CARLA: propose a Copilot plan (propose_plan) for the user to approve',
        '--carla-live': 'starts a live CARLA rig: propose a Copilot plan',
        '--campaign': 'runs CARLA drives: propose a Copilot plan',
        '--critical-carla': 'runs CARLA drives: propose a Copilot plan',
        '--run-plan': 'runs an approved Copilot plan: the user runs it from the window',
        '--detection-download': 'downloads a model: the user does it on the Detection quality page',
        '--watch': "runs the user's automation command", '--on-change': "the user's automation command",
        '--watch-runs': 'a watcher that never ends', '--watch-interval': 'a watcher that never ends',
        '--perception-scene-set': 'runs CARLA drives: the user drives a scene set from Simulate ▸ Rig perception',
        '--foundation-download': 'downloads models: the user does it on the Foundation models page',
        '--fleet-twin-watch': 'a watcher that never ends',
        '--perception-records': "writes into the user's own record store: the user adds records on the page",
        '--perf-records-import': "writes into the user's own record store: the user imports on the page",
        '--perf-save': "replaces the user's saved model: the user trains and saves it on the page",
        '--sample': 'the run is on this project', '--template': 'the run is on this project',
        '--self-test': 'not a job on the project',
        '--lab-publish': 'sends the project to a lab server: the user publishes on File ▸ Lab',
        '--lab-sync': 'sends the project to a lab server and may change it: the user syncs on File ▸ Lab',
        '--lab-pull': "writes a lab version into the user's files: the user pulls on File ▸ Lab",
        '--lab-pooled-twin': 'reads other teams\' fleets from a lab server: lab_pooled_twin does it'}
EXIT = {0: 'done: fine',
        2: 'done: the plan has critical issues (or the job\'s own failure condition, e.g. a fuse over its rating, a '
           'cable too long, an antenna with no sky); see the Issues in the output',
        3: 'done: a camera or sensor has to be re-mounted or could not be checked (a drive log, as-built check, '
           'online calibration, lidar calibration or rig health found a fault)',
        4: 'a calibration failed or matched no camera, or an input could not be read or an output not written',
        5: 'done: the requirements check failed (or no design option meets every must requirement)',
        6: 'done, with findings (see the output)',
        7: 'no CARLA server answered (or CARLA\'s Python is missing)',
        8: 'a model the job needs is not downloaded (the detector, a foundation model)',
        9: 'an external detector is not set up or approved here, or a drive failed',
        1: 'failed: an error (see the output)', 130: 'stopped', -9: 'stopped: it ran longer than timeout_s',
        -15: 'cancelled'}
JSON_MAX = 60_000                       # characters of JSON outputs returned in full, all together
TAIL = 4000


def _options() -> dict:
    """option -> (metavar/nargs, help) for every batch option Claude may use."""
    import argparse
    from ..batch import add_arguments
    ap = argparse.ArgumentParser(add_help=False)
    add_arguments(ap)
    out = {}
    for a in ap._actions:                               # noqa: SLF001 — argparse keeps them only here
        for o in a.option_strings:
            if o.startswith('--') and o not in DENY:
                mv = a.metavar if a.metavar else (None if a.nargs == 0 else a.dest.upper())
                if isinstance(mv, tuple):
                    mv = ' '.join(mv)
                takes = 'flag' if a.nargs == 0 or isinstance(a, (argparse._StoreTrueAction,  # noqa: SLF001
                                                                   argparse._StoreFalseAction)) else (mv or 'VALUE')
                out[o] = (takes, (a.help or '').replace('%%', '%'))
    return out


def help_doc() -> dict:
    return {'options': {o: f'{t} — {h}' if t != 'flag' else f'(flag) — {h}' for o, (t, h) in sorted(_options().items())},
            'not_allowed': DENY,
            'note': 'args are the command-line words after the project, e.g. ["--perception-range", "range.json"] or '
                    '["--drive-log", "/data/drive1", "check.json", "--variant", "Data rig"]. Relative output paths '
                    'land in the run folder. Several jobs can go in one run.'}


def check(args) -> str:
    """'' when args are batch options Claude may use, else why not."""
    if not isinstance(args, list) or not args or not all(isinstance(a, (str, int, float)) for a in args):
        return 'args: a list of the command-line words (strings), e.g. ["--perception-range", "range.json"]'
    allowed = _options()
    for a in map(str, args):
        if a.startswith('--'):
            name = a.split('=', 1)[0]
            if name in DENY:
                return f'{name} is not available here: {DENY[name]}'
            if name not in allowed:
                return f'{name} is not a batch option (run_batch with help: true lists them)'
    if not any(str(a).startswith('--') for a in args):
        return 'args name no job (an option such as --analyse OUT.json)'
    return ''


def command(project_file: str, args) -> tuple[list, dict]:
    """The command and environment of the batch process: this same Vantage (the bundle's console program, or this
    Python with the vantage package)."""
    env = dict(os.environ, QT_QPA_PLATFORM='offscreen')
    if getattr(sys, 'frozen', False):
        here = os.path.dirname(sys.executable)
        cli = os.path.join(here, 'vantage-cli.exe' if os.name == 'nt' else 'vantage-cli')
        exe = [cli if os.path.exists(cli) else sys.executable]
    else:
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        env['PYTHONPATH'] = root + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
        exe = [sys.executable, '-m', 'vantage']
    return exe + [project_file] + [str(a) for a in args], env


def run(project, args, base_dir: str, timeout_s=600, cancel=None) -> dict:
    """Run the batch job on a copy of project in a new folder under base_dir. cancel: a threading.Event; set, the
    job's process is stopped (exit code -15, 'cancelled')."""
    why = check(args)
    if why:
        return {'error': why}
    stamp = time.strftime('%Y%m%d-%H%M%S')
    folder = os.path.join(base_dir, f'claude-run-{stamp}')
    n = 2
    while os.path.exists(folder):
        folder = os.path.join(base_dir, f'claude-run-{stamp}-{n}')
        n += 1
    os.makedirs(folder)
    pfile = os.path.join(folder, 'project.vantage')
    project.save(pfile)                                 # (a snapshot: the user's project is not touched)
    cmd, env = command(pfile, args)
    t0 = time.time()
    limit = max(10, min(int(timeout_s or 600), 3600))
    with open(os.path.join(folder, '.stdout'), 'w+', encoding='utf-8', errors='replace') as fo, \
            open(os.path.join(folder, '.stderr'), 'w+', encoding='utf-8', errors='replace') as fe:
        proc = subprocess.Popen(cmd, cwd=folder, env=env, stdout=fo, stderr=fe, text=True)
        code = None
        while code is None:
            try:
                code = proc.wait(timeout=0.25)
            except subprocess.TimeoutExpired:
                if cancel is not None and cancel.is_set():
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                    code = -15
                elif time.time() - t0 > limit:
                    proc.kill()
                    proc.wait()
                    code = -9
        fo.seek(0)
        fe.seek(0)
        out, err = fo.read(), fe.read()
    for f in ('.stdout', '.stderr'):
        try:
            os.remove(os.path.join(folder, f))
        except OSError:
            pass
    files, budget, docs = [], JSON_MAX, {}
    for root, _, names in os.walk(folder):
        for nm in sorted(names):
            f = os.path.join(root, nm)
            if f == pfile:
                continue
            rel = os.path.relpath(f, folder)
            size = os.path.getsize(f)
            files.append({'file': rel, 'bytes': size})
            if nm.endswith('.json') and size <= budget:
                try:
                    with open(f, encoding='utf-8') as fh:
                        docs[rel] = json.load(fh)
                    budget -= size
                except (OSError, ValueError):
                    pass
    outside = [str(a) for a in args if os.path.isabs(str(a)) and not str(a).startswith(folder)]
    return {'exit_code': code, 'meaning': EXIT.get(code, 'finished with an unusual code: see the output'),
            'seconds': round(time.time() - t0, 1), 'folder': folder, 'files': files[:200], 'json': docs,
            'stdout_end': out[-TAIL:], 'stderr_end': err[-TAIL:],
            'note': ('The run used a copy of the project; it is unchanged. ' +
                     ('Some outputs were given absolute paths outside the run folder: ' + ', '.join(outside[:5])
                      if outside else '') +
                     (' --save wrote a new project file: open_project it to work on it.' if '--save' in map(str, args)
                      else '')).strip()}
