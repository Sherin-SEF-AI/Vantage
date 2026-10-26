"""Automation: regenerate the exports when the design changes and run the user's own command (core.automation,
ui.workspaces.automation_ws). The command is a harmless script written into the test's own folder: it records the
VANTAGE_ environment it got, reads the summary, prints a few lines and exits with the code it is given."""
import hashlib
import json
import os
import shlex
import stat
import sys
import threading
import time
import zipfile

import pytest
import yaml

from vantage.core import automation as AU
from vantage.core import samples
from vantage.core.project import Project

FAKE = r'''import json, os, sys, time
args = sys.argv[1:]
code = int(args[0]) if args else 0
sleep = float(args[1]) if len(args) > 1 else 0.0
env = {k: v for k, v in os.environ.items() if k.startswith('VANTAGE_')}
here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(here, 'runs.jsonl'), 'a') as fh:
    fh.write(json.dumps({'env': env, 'cwd': os.getcwd(), 'pid': os.getpid(), 'code': code}) + '\n')
print('fake check: reading', os.path.basename(env['VANTAGE_SUMMARY']), flush=True)
summary = json.load(open(env['VANTAGE_SUMMARY']))
print('summary schema', summary['schema'], 'files', len(summary['files']), flush=True)
with open(os.path.join(env['VANTAGE_EXPORT_DIR'], 'fake_result.txt'), 'w') as fh:
    fh.write('ok %d\n' % code)
for i in range(3):
    print('line', i, flush=True)
if sleep:
    time.sleep(sleep)
sys.exit(code)
'''


def fake_command(folder, code=0, sleep=0.0):
    """The harmless test command and where it logs its runs."""
    script = os.path.join(str(folder), 'fake_check.py')
    if not os.path.exists(script):
        with open(script, 'w') as fh:
            fh.write(FAKE)
    return f'"{sys.executable}" "{script}" {code} {sleep:g}', os.path.join(str(folder), 'runs.jsonl')


def runs_of(log):
    if not os.path.exists(log):
        return []
    with open(log) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def data_rig(path=None):
    p = samples.tigor_sample_project()
    p.active_uid = next(v for v in p.variants if v.sensors).uid
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        p.save(path)
    return p


@pytest.fixture(scope='module')
def eng():
    from vantage.core.analysis import Engine
    AU.ensure_app()
    return Engine()


@pytest.fixture()
def proj(tmp_path):
    return data_rig(str(tmp_path / 'proj' / 'rig.vantage'))


def _sha(path):
    with open(path, 'rb') as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:                                        # a zombie still answers kill(0)
        with open(f'/proc/{pid}/stat') as fh:
            return fh.read().split(')')[-1].split()[0] != 'Z'
    except OSError:
        return False


# ====================================================================== settings: never a command
def test_settings_are_tolerant_and_never_carry_a_command():
    d = AU.AutomationSettings.from_dict(None)
    assert d.enabled is False and d.trigger == 'save' and d.out_dir == AU.DEFAULT_OUT_DIR
    assert d.formats == list(AU.DEFAULT_FORMATS) and d.check and d.prune and d.run_command
    assert AU.AutomationSettings.from_dict('junk').to_dict() == d.to_dict()
    raw = {'enabled': 'yes', 'trigger': 'bogus', 'quiet_s': 1e9, 'units': 'mm', 'angles': 'gradians',
           'formats': ['extrinsics_json', 'viewport_png', 'nope', 'extrinsics_json', 3], 'out_dir': '../x',
           'frame': 'front_axle', 'variant': 'v\nx', 'command': 'touch PWNED', 'check': 0, 'prune': 'off'}
    s = AU.AutomationSettings.from_dict(raw)
    assert s.enabled is True and s.trigger == 'save' and s.quiet_s == AU.LIMITS['quiet_s'][1]
    assert s.formats == ['extrinsics_json']                  # the window-only and unknown ones dropped, no repeat
    assert s.out_dir == AU.DEFAULT_OUT_DIR and s.units == 'mm' and s.angles == 'deg' and s.frame == 'front_axle'
    assert s.variant == 'v x' and s.check is False and s.prune is False
    assert 'command' not in s.to_dict() and 'touch' not in json.dumps(s.to_dict())
    probs = AU.settings_problems(raw)
    assert any('folder' in x for x in probs) and any('viewport_png' in x for x in probs)
    assert any('command' in x and 'ignored' in x for x in probs)
    assert AU.AutomationSettings.from_dict({'quiet_s': float('nan')}).quiet_s == 10.0
    assert AU.AutomationSettings.from_dict({'quiet_s': -5}).quiet_s == AU.LIMITS['quiet_s'][0]
    assert AU.AutomationSettings.from_dict({'formats': []}).formats == []
    assert AU.AutomationSettings.from_dict(s.to_dict()).to_dict() == s.to_dict()
    assert AU.settings_problems({}) == [] and AU.settings_problems(None) == []


@pytest.mark.parametrize('text, want', [
    ('exports/auto', 'exports/auto'), ('exports\\auto', 'exports/auto'), ('./a/./b/', 'a/b'), ('ci out', 'ci out'),
    ('/abs', None), ('C:/x', None), ('c:x', None), ('~/x', None), ('..', None), ('a/../b', None), ('', None),
    ('.', None), ('a:b', None), ('a\nb', None), ('.git/hooks', None), ('a/.GIT', None), ('x' * 300, None),
    ('a./b', None), ('a ', 'a')])
def test_the_export_folder_stays_inside_the_project_folder(text, want):
    rel, why = AU.clean_folder(text)
    assert rel == want and (bool(why) == (want is None))


def test_resolve_out_dir_and_links_out_of_the_project(tmp_path, proj, eng):
    p = Project.load(proj.path)
    base = os.path.dirname(p.path)
    assert AU.resolve_out_dir(p.path, 'a/b') == os.path.join(base, 'a', 'b')
    with pytest.raises(ValueError, match='Save the project'):
        AU.resolve_out_dir('', 'a')
    outside = tmp_path / 'outside'
    outside.mkdir()
    os.symlink(str(outside), os.path.join(base, 'linked'))
    with pytest.raises(ValueError, match='outside'):
        AU.resolve_out_dir(p.path, 'linked/x')
    # a link inside the export folder that leads out: nothing is written, the command does not run
    out = os.path.join(base, 'exports', 'automation')
    os.makedirs(out)
    os.symlink(str(outside), os.path.join(out, 'opencv'))
    cmd, log = fake_command(tmp_path)
    rec = AU.run(p, eng, command=cmd)
    assert rec.status == 'error' and 'symbolic link' in rec.message and rec.command_state == 'skipped'
    assert list(outside.iterdir()) == [] and runs_of(log) == [] and AU.exit_code_of(rec) == AU.EXIT_ERROR


# ====================================================================== the user's command store
class DictSettings:
    def __init__(self):
        self.d = {}

    def value(self, k, default=None):
        return self.d.get(k, default)

    def setValue(self, k, v):
        self.d[k] = v

    def remove(self, k):
        self.d.pop(k, None)


def test_a_command_runs_only_once_approved_for_that_project_file(tmp_path):
    from PyQt6.QtCore import QSettings
    a, b = str(tmp_path / 'a.vantage'), str(tmp_path / 'b.vantage')
    for f in (a, b):
        open(f, 'w').close()
    for settings in (DictSettings(), QSettings(str(tmp_path / 'user.ini'), QSettings.Format.IniFormat)):
        cs = AU.CommandStore(settings)
        assert cs.get('') is None and cs.get(a) is None and cs.approved_command(a) is None
        cfg = cs.put(a, '  make perception-test  ')
        assert cfg.command == 'make perception-test' and not cs.is_approved(a) and cs.approved_command(a) is None
        assert cs.approve(a, 'something else') is None and not cs.is_approved(a)    # only the stored text
        assert cs.approve(a, 'make perception-test') is not None and cs.is_approved(a)
        # the approval covers this text for this file (an independent digest)
        want = hashlib.sha256(f'vantage-automation\0{AU.norm_path(a)}\0make perception-test'.encode()).hexdigest()
        assert cs.get(a).approved == want and cs.approved_command(a).command == 'make perception-test'
        assert cs.put(a, 'make perception-test', 5).approved == want and cs.is_approved(a)   # unchanged: kept
        assert cs.get(a).timeout_min == 5
        cs.put(a, 'make perception-test && rm -rf build')                      # changed: approval withdrawn
        assert not cs.is_approved(a)
        cs.approve(a)
        # the same file through a symbolic link is the same project file
        link = str(tmp_path / f'link-{type(settings).__name__}.vantage')
        os.symlink(a, link)
        assert cs.is_approved(link)
        # another file has none; a record copied from a to b's key is refused (it names a's path)
        assert cs.get(b) is None
        settings.setValue(cs.key(b), settings.value(cs.key(a)))
        assert cs.get(b) is None and not cs.is_approved(b)
        # b's own record carrying a's approval digest is not approved
        cs.put(b, 'make perception-test && rm -rf build')
        rec = json.loads(settings.value(cs.key(b)))
        rec['approved'] = cs.get(a).approved
        settings.setValue(cs.key(b), json.dumps(rec))
        assert not cs.is_approved(b)
        settings.setValue(cs.key(b), '{not json')
        assert cs.get(b) is None
        cs.revoke(a)
        assert not cs.is_approved(a)
        cs.remove(a)
        assert cs.get(a) is None
        assert cs.put(a, '   ') is None and cs.get(a) is None                    # empty: removed
    # QSettings keeps it for the next session
    s1 = QSettings(str(tmp_path / 'user2.ini'), QSettings.Format.IniFormat)
    AU.CommandStore(s1).put(a, 'echo hi')
    AU.CommandStore(s1).approve(a)
    s2 = QSettings(str(tmp_path / 'user2.ini'), QSettings.Format.IniFormat)
    assert AU.CommandStore(s2).is_approved(a)


# ====================================================================== a run
def test_a_run_regenerates_and_hands_the_command_its_environment(tmp_path, proj, eng):
    p = Project.load(proj.path)
    cmd, log = fake_command(tmp_path, code=3)
    chunks = []
    rec = AU.run(p, eng, command=cmd, on_output=chunks.append)
    base = os.path.dirname(os.path.abspath(p.path))
    out = os.path.join(base, 'exports', 'automation')
    assert rec.status == 'failed' and rec.exit_code == 3 and AU.exit_code_of(rec) == 3 and rec.command_state == 'ran'
    (r,) = runs_of(log)
    ours = dict(AU.ENV_HELP)
    env = {k: v for k, v in r['env'].items() if k in ours}      # the rest is the session's own environment
    assert all(r['env'][k] == os.environ[k] for k in r['env'] if k not in ours)
    assert env == {'VANTAGE_EXPORT_DIR': out, 'VANTAGE_PROJECT': os.path.abspath(p.path), 'VANTAGE_PROJECT_DIR': base,
                   'VANTAGE_VARIANT': p.variant.name, 'VANTAGE_VARIANT_UID': p.variant.uid,
                   'VANTAGE_SUMMARY': os.path.join(out, 'vantage_summary.json'), 'VANTAGE_TRIGGER': 'manual',
                   'VANTAGE_CHECK_EXIT': '0', 'VANTAGE_RUN_ID': rec.id, 'VANTAGE_UNSAVED': '0'}
    assert os.path.realpath(r['cwd']) == os.path.realpath(base)
    # what the command printed, in order, and that it could read the summary
    text = ''.join(chunks)
    assert text == rec.output and 'summary schema vantage.automation_run/1' in text
    assert [ln for ln in text.splitlines() if ln.startswith('line')] == ['line 0', 'line 1', 'line 2']
    # the files: the default calibration formats, the requirements check, the summary
    ex = json.load(open(os.path.join(out, 'extrinsics.json')))
    assert sorted(c['tag'] for c in ex['cameras']) == sorted(c.tag for c in p.variant.cameras)
    for t in ('F', 'ML', 'R'):
        assert os.path.exists(os.path.join(out, 'opencv', f'{t}.yaml'))
    for f in ('requirements.json', 'requirements.csv', 'review_pack.pdf'):
        assert f'requirements/{f}' in rec.written and os.path.exists(os.path.join(out, 'requirements', f))
    assert rec.requirements['verdict'] == 'empty' and rec.requirements['exit_code'] == 0
    on_disk = sorted(os.path.relpath(os.path.join(d, f), out).replace(os.sep, '/')
                     for d, _, fs in os.walk(out) for f in fs)
    assert on_disk == sorted(rec.written + ['fake_result.txt'])
    # the summary names every file with its SHA-256 (checked against the files themselves)
    doc = json.load(open(os.path.join(out, 'vantage_summary.json')))
    assert doc['schema'] == AU.SCHEMA and doc['run_id'] == rec.id and doc['variant_uid'] == p.variant.uid
    assert doc['conventions']['frame'] == p.frame and doc['conventions']['units'] == 'm'
    assert doc['environment'] == env and doc['command']['state'] == 'pending'
    hashed = [f for f in doc['files'] if 'sha256' in f]
    assert len(hashed) == len(rec.written) - 1                       # all but the summary itself
    for f in hashed:
        assert f['sha256'] == _sha(os.path.join(out, f['path'])) and f['bytes'] == os.path.getsize(
            os.path.join(out, f['path']))
    assert set(doc['changed']['added']) == set(rec.written) - {'vantage_summary.json'}
    # what the command itself wrote
    assert rec.command_changed['added'] == ['fake_result.txt'] and rec.command_changed['changed'] == []
    # the command's text is never in what is written or summarised
    assert cmd not in json.dumps(doc) and 'fake_check' not in json.dumps(AU.summary(rec))
    names = dict(AU.files(rec, p))
    assert set(names) == {'automation_run.json', 'automation_run.log'}
    assert b'fake_check' not in names['automation_run.json'] and b'line 2' in names['automation_run.log']
    paths = AU.export(rec, p, str(tmp_path / 'runexport'))
    assert json.load(open(paths[0]))['exit_code_batch'] == 3


def test_changes_are_judged_by_content_not_time_stamps(tmp_path, proj, eng):
    p = Project.load(proj.path)
    st = AU.AutomationSettings(formats=['extrinsics_json', 'opencv_yaml', 'kitti_calib', 'report_pdf'])
    r1 = AU.run(p, eng, st)
    assert r1.status == 'passed' and r1.changed['added'] and not r1.changed['changed']
    time.sleep(1.1)                             # every time stamp in the files moves on
    r2 = AU.run(p, eng, st, previous=r1)
    out = r2.out_dir
    assert r2.status == 'passed' and r2.changed['changed'] == [] and r2.changed['added'] == []
    assert r2.changed['rewritten'] == sorted(['report.pdf', 'requirements/review_pack.pdf'])
    assert r2.n_changed() == 0 and r2.changed['unchanged'] == len(r2.written) - 1 - 2
    assert 'nothing changed' in r2.message
    # the KITTI file did change on disk (its calib_time), yet counts as unchanged
    assert 'calib_time' in open(os.path.join(out, 'kitti', 'calib_cam_to_cam.txt')).read()
    # move one camera: its own files change, the other cameras' do not
    before_ml = _sha(os.path.join(out, 'opencv', 'ML.yaml'))
    f = p.variant.by_tag('F')
    f.x += 0.05
    r3 = AU.run(p, eng, st, previous=r2)
    assert 'opencv/F.yaml' in r3.changed['changed'] and 'extrinsics.json' in r3.changed['changed']
    assert 'opencv/ML.yaml' not in r3.changed['changed'] and _sha(os.path.join(out, 'opencv', 'ML.yaml')) == before_ml


def test_fingerprints_ignore_time_stamps_only(tmp_path):
    a, b, c = tmp_path / 'a.yaml', tmp_path / 'b.yaml', tmp_path / 'c.yaml'
    a.write_text('# exported 2026-10-26T13:07:56+05:30\ncalib_time: 26-Oct-2026 13:07:56\nx: 1.0\n"seconds": 0.021\n')
    b.write_text('# exported 2026-10-27T09:00:01+05:30\ncalib_time: 27-Oct-2026 09:00:01\nx: 1.0\n"seconds": 0.5\n')
    c.write_text('# exported 2026-10-26T13:07:56+05:30\ncalib_time: 26-Oct-2026 13:07:56\nx: 1.5\n"seconds": 0.021\n')
    fa, fb, fc = AU.fingerprint(str(a)), AU.fingerprint(str(b)), AU.fingerprint(str(c))
    assert fa[1] != fb[1] and fa[2] == fb[2] and fa[2] != fc[2]
    # a zip archive (an NPZ) written at another time: the same members, the same fingerprint
    za, zb = str(tmp_path / 'a.npz'), str(tmp_path / 'b.npz')
    for path, when in ((za, (2020, 1, 1, 0, 0, 0)), (zb, (2024, 5, 6, 7, 8, 10))):
        with zipfile.ZipFile(path, 'w') as z:
            z.writestr(zipfile.ZipInfo('m.npy', when), b'same bytes')
    assert AU.fingerprint(za)[1] != AU.fingerprint(zb)[1] and AU.fingerprint(za)[2] == AU.fingerprint(zb)[2]
    pdf = tmp_path / 'x.pdf'
    pdf.write_bytes(b'%PDF-1.4 ...')
    assert AU.fingerprint(str(pdf))[2] is None
    before, after = AU.Scan({'x.pdf': (1, 'aa', None), 'k': (1, 's1', 'f')}), AU.Scan({'x.pdf': (1, 'bb', None),
                                                                                         'k': (2, 's2', 'f')})
    d = AU.diff(before, after)
    assert d['rewritten'] == ['x.pdf'] and d['changed'] == [] and d['unchanged'] == 1


def test_stale_files_go_and_only_those_vantage_wrote(tmp_path, proj, eng):
    p = Project.load(proj.path)
    r1 = AU.run(p, eng, AU.AutomationSettings(formats=['extrinsics_json', 'opencv_yaml'], check=False))
    out = r1.out_dir
    with open(os.path.join(out, 'notes.txt'), 'w') as fh:
        fh.write('mine')
    os.makedirs(os.path.join(out, 'opencv', 'mine'))
    keep = os.path.join(out, 'opencv', 'mine', 'keep.yaml')
    open(keep, 'w').close()
    r2 = AU.run(p, eng, AU.AutomationSettings(formats=['extrinsics_json'], check=False), previous=r1)
    gone = sorted(f'opencv/{c.tag}.yaml' for c in p.variant.cameras)
    assert r2.pruned == gone and r2.changed['removed'] == gone
    assert os.path.exists(os.path.join(out, 'notes.txt')) and os.path.exists(keep)
    assert not any(os.path.exists(os.path.join(out, g)) for g in gone)
    # the next run without the opencv folder's own files leaves it (it still holds the user's file)
    assert os.path.isdir(os.path.join(out, 'opencv'))
    # pruning off: nothing goes
    r3 = AU.run(p, eng, AU.AutomationSettings(formats=['extrinsics_json', 'opencv_yaml'], check=False), previous=r2)
    r4 = AU.run(p, eng, AU.AutomationSettings(formats=['extrinsics_json'], check=False, prune=False), previous=r3)
    assert r4.pruned == [] and all(os.path.exists(os.path.join(out, g)) for g in gone)
    # prune() itself never leaves the folder, never follows a link
    victim = tmp_path / 'victim.txt'
    victim.write_text('do not delete')
    os.symlink(str(victim), os.path.join(out, 'link.txt'))
    assert AU.prune(out, ['../victim.txt', str(victim), 'link.txt', '../../victim.txt', ''], []) == []
    assert victim.exists() and os.path.islink(os.path.join(out, 'link.txt'))


def test_a_project_file_can_never_make_vantage_run_a_command(tmp_path, eng):
    path = str(tmp_path / 'proj' / 'evil.vantage')
    p = data_rig()
    marker = tmp_path / 'PWNED'
    evil = f'touch "{marker}"'
    p.analysis.automation = {'enabled': True, 'trigger': 'save', 'run_command': True, 'command': evil,
                             'on_change': evil, 'commands': [evil], 'out_dir': 'out'}
    os.makedirs(os.path.dirname(path))
    p.save(path)
    q = Project.load(path)
    assert q.analysis.automation['command'] == evil                    # kept in the file as data …
    rec = AU.run(q, eng)                                               # … never run
    assert rec.status == 'passed' and rec.command_state == 'none' and not marker.exists()
    assert AU.watch(path, None, runs=1, interval=0.1) == 0 and not marker.exists()
    probs = [i.message for i in AU.issues(q)]
    assert any('names a command' in m for m in probs)
    # nor can it name the folder of a command the user approved for it: the folder must stay in the project's folder
    q.analysis.automation = {'enabled': True, 'out_dir': '../../outside'}
    assert AU.settings_of(q).out_dir == AU.DEFAULT_OUT_DIR


def test_time_limit_and_stop_end_the_command_and_what_it_started(tmp_path, proj, eng):
    p = Project.load(proj.path)
    st = AU.AutomationSettings(formats=['extrinsics_json'], check=False)
    pidfile = tmp_path / 'grandchild.pid'
    cmd = f'sleep 30 & echo $! > "{pidfile}"; wait'
    t = time.time()
    rec = AU.run(p, eng, st, command=cmd, timeout_s=1.0)
    assert rec.status == 'timeout' and AU.exit_code_of(rec) == AU.EXIT_TIMEOUT and time.time() - t < 15
    assert 'Vantage stopped the command' in rec.output and rec.exit_code is None
    gpid = int(pidfile.read_text())
    deadline = time.time() + 5
    while _alive(gpid) and time.time() < deadline:
        time.sleep(0.05)
    assert not _alive(gpid)                     # the whole process group went
    # stopped by hand while the command runs
    cmd2, log = fake_command(tmp_path, code=0, sleep=30)
    cancel = threading.Event()
    result = {}

    def go():
        result['rec'] = AU.run(p, eng, st, command=cmd2, cancel=cancel.is_set)
    w = threading.Thread(target=go)
    w.start()
    deadline = time.time() + 60
    while not runs_of(log) and time.time() < deadline:
        time.sleep(0.05)
    pid = runs_of(log)[0]['pid']
    cancel.set()
    w.join(20)
    rec2 = result['rec']
    assert rec2.status == 'cancelled' and AU.exit_code_of(rec2) == AU.EXIT_CANCELLED
    deadline = time.time() + 5
    while _alive(pid) and time.time() < deadline:
        time.sleep(0.05)
    assert not _alive(pid)


def test_a_failed_regeneration_skips_the_command(tmp_path, proj, eng, monkeypatch):
    cmd, log = fake_command(tmp_path)
    q = data_rig()                              # never saved: no folder to write into
    rec = AU.run(q, eng, command=cmd)
    assert rec.status == 'error' and 'Save the project first' in rec.message and runs_of(log) == []
    from vantage.core import exporters as ex
    real = ex.export

    def flaky(ids, *a, **k):
        if 'opencv_yaml' in ids:
            raise RuntimeError('disk on fire')
        return real(ids, *a, **k)
    monkeypatch.setattr(ex, 'export', flaky)
    p = Project.load(proj.path)
    first = AU.run(p, eng, AU.AutomationSettings(formats=['extrinsics_json'], check=False))
    rec = AU.run(p, eng, AU.AutomationSettings(formats=['extrinsics_json', 'opencv_yaml'], check=False),
                 command=cmd, previous=first)
    assert rec.status == 'error' and 'opencv_yaml' in rec.format_errors and 'disk on fire' in rec.message
    assert rec.command_state == 'skipped' and runs_of(log) == [] and rec.pruned == []
    assert AU.exit_code_of(rec) == AU.EXIT_ERROR


def test_the_requirements_check_decides_without_a_command(tmp_path, proj, eng):
    from vantage.core import requirements as RQ
    p = Project.load(proj.path)
    # a must requirement the sample cannot meet: blind ground at most 0 m²
    st = RQ.settings_of(p)
    items = [{'id': 'R1', 'text': 'Less than no blind ground', 'priority': 'must', 'metric': 'coverage.blind_m2',
              'params': {}, 'op': '<=', 'target': -1.0}]
    p.analysis.requirements = dict(st.to_dict(), items=items)
    rec = AU.run(p, eng, AU.AutomationSettings(formats=[]))
    assert rec.requirements['verdict'] == 'fail' and rec.requirements['failing'] == ['R1'] and rec.status == 'failed'
    assert AU.exit_code_of(rec) == RQ.EXIT_FAIL
    cmd, log = fake_command(tmp_path, code=0)
    rec = AU.run(p, eng, AU.AutomationSettings(formats=[]), command=cmd)
    assert runs_of(log)[0]['env']['VANTAGE_CHECK_EXIT'] == str(RQ.EXIT_FAIL)
    assert rec.status == 'passed' and AU.exit_code_of(rec) == 0           # the command's own verdict passes through


# ====================================================================== history, issues, debounce
def test_history_is_private_and_bounded(tmp_path):
    h = AU.History(str(tmp_path / 'cfg' / 'history.json'))
    proj_a, proj_b = str(tmp_path / 'a.vantage'), str(tmp_path / 'b.vantage')
    for i in range(AU.HISTORY_KEEP + 5):
        r = AU.RunRecord(f'r{i:02d}', 'save', project_path=proj_a, started=1000.0 + i, finished=1001.0 + i,
                         status='passed' if i % 3 else 'cancelled', out_dir=str(tmp_path / 'out'),
                         written=['a.json'], output='x' * (AU.OUTPUT_KEEP_HISTORY + 10) if i == 34 else 'ok')
        h.add(r)
    h.add(AU.RunRecord('other', 'manual', project_path=proj_b, started=5.0, status='failed'))
    runs = h.runs(proj_a)
    assert [r.id for r in runs] == [f'r{i:02d}' for i in range(AU.HISTORY_KEEP + 4, 4, -1)]
    assert runs[0].output.startswith('[… earlier output not kept …]') and len(runs[0].output) < AU.OUTPUT_KEEP_HISTORY + 40
    assert stat.S_IMODE(os.stat(h.path).st_mode) == 0o600
    assert h.last(proj_a).id == 'r34' and h.last(proj_a, complete=True).id == 'r34'
    assert h.last(proj_a, complete=True, out_dir=str(tmp_path / 'elsewhere')) is None
    h.add(AU.RunRecord('r99', 'save', project_path=proj_a, started=3000.0, status='cancelled'))
    assert h.last(proj_a).id == 'r99' and h.last(proj_a, complete=True).id == 'r34'
    assert [r.id for r in h.runs(proj_b)] == ['other']
    # a run still marked running (the app closed during it) reads back as stopped
    h.add(AU.RunRecord('r100', 'save', project_path=proj_b, started=10.0))
    assert h.runs(proj_b)[0].status == 'cancelled'
    h.clear(proj_b)
    assert h.runs(proj_b) == []
    with open(h.path, 'w') as fh:
        fh.write('{broken')
    assert AU.History(h.path).runs(proj_a) == [] and h.runs(proj_a) == []
    assert AU.RunRecord.from_dict({'nope': 1}) is None
    r = AU.RunRecord.from_dict({'id': 'x', 'changed': 'junk', 'written': 'junk'})
    assert r.changed['added'] == [] and r.written == []


def test_issues_name_a_failed_last_run(tmp_path):
    h = AU.History(str(tmp_path / 'h.json'))
    p = data_rig()
    assert AU.issues(p, history=h) == []
    p.path = str(tmp_path / 'p.vantage')
    p.analysis.automation = {'enabled': True}
    assert AU.issues(p, history=h) == []
    h.add(AU.RunRecord('a', 'save', project_path=p.path, started=time.time(), status='passed'))
    assert AU.issues(p, history=h) == []
    h.add(AU.RunRecord('b', 'save', project_path=p.path, started=time.time(), status='failed', command_state='ran',
                       exit_code=2))
    (i,) = AU.issues(p, history=h)
    assert i.severity == 'warning' and 'exited with 2' in i.message and 'On save' in i.message
    p.analysis.automation = {'enabled': False}
    assert AU.issues(p, history=h) == []        # off: a past failure is not this design's problem
    p.analysis.automation = {'out_dir': '/etc'}
    assert any('folder' in x.message for x in AU.issues(p, history=h))
    t = time.time()
    for _ in range(50):
        AU.issues(p, history=h)
    assert (time.time() - t) / 50 < 0.03


def test_debouncer_fires_once_after_the_quiet_period():
    now = [0.0]
    d = AU.Debouncer(5.0, clock=lambda: now[0])
    assert not d.ready() and not d.pending
    for t in (0.0, 3.0, 6.0):                   # three changes, each inside the last one's quiet period
        now[0] = t
        d.poke()
    for t in (7.0, 10.9):
        now[0] = t
        assert not d.ready()
    now[0] = 11.0
    assert d.ready() and not d.ready() and not d.pending


# ====================================================================== headless: --watch
def test_watch_reruns_on_change_and_passes_the_exit_code_through(tmp_path, proj, eng):
    path = proj.path
    cmd, log = fake_command(tmp_path, code=7)
    t0 = time.time()
    seen = []

    def edit():                                 # once the first run's command ran, change the project on disk
        while len(runs_of(log)) < 1 and time.time() - t0 < 90:
            time.sleep(0.05)
        q = Project.load(path)
        q.variant.by_tag('F').x += 0.05
        q.save(path)
    th = threading.Thread(target=edit)
    th.start()
    code = AU.watch(path, cmd, runs=2, interval=0.1, quiet=0.3, out=seen.append, log=lambda s: None,
                    stop=lambda: time.time() - t0 > 120)
    th.join(10)
    assert code == 7
    rows = runs_of(log)
    assert len(rows) == 2 and all(r['env']['VANTAGE_TRIGGER'] == 'watch' for r in rows)
    assert rows[0]['env']['VANTAGE_RUN_ID'] != rows[1]['env']['VANTAGE_RUN_ID']
    doc = json.load(open(rows[1]['env']['VANTAGE_SUMMARY']))
    assert 'opencv/F.yaml' in doc['changed']['changed'] and 'opencv/ML.yaml' not in doc['changed']['changed']
    assert ''.join(seen).count('line 2') == 2


def test_watch_waits_out_an_unreadable_file_and_overrides(tmp_path, eng):
    path = str(tmp_path / 'proj' / 'rig.vantage')
    os.makedirs(os.path.dirname(path))
    with open(path, 'w') as fh:
        fh.write('half written')
    msgs = []
    assert AU.watch(path, None, runs=1, interval=0.1, log=msgs.append) == AU.EXIT_ERROR
    assert any('cannot read' in m for m in msgs)
    t0 = time.time()
    msgs.clear()

    def fix():                                  # once the watch has met the broken file, write a good one
        while not any('cannot read' in m for m in msgs) and time.time() - t0 < 60:
            time.sleep(0.05)
        data_rig(path)
    th = threading.Thread(target=fix)
    th.start()
    out = str(tmp_path / 'ci-out')
    code = AU.watch(path, None, runs=2, interval=0.1, quiet=0.2, log=msgs.append, out_dir=out,
                    formats=['extrinsics_json', 'viewport_png'], stop=lambda: time.time() - t0 > 90)
    th.join(5)
    assert code == 0 and os.path.exists(os.path.join(out, 'extrinsics.json'))
    assert any('viewport_png' in m for m in msgs)                       # skipped, and said so
    assert not os.path.exists(os.path.join(os.path.dirname(path), 'exports'))


def test_batch_options(tmp_path, proj):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('path', nargs='?')
    from vantage import batch
    batch.add_arguments(ap)
    if not any(a.dest == 'watch' for a in ap._actions):                 # until batch mode adds them itself
        AU.add_arguments(ap.add_argument_group('automation'))
    cmd, log = fake_command(tmp_path, code=4)
    args = ap.parse_args(['--watch', proj.path, '--on-change', cmd, '--watch-runs', '1', '--formats',
                          'extrinsics_json'])
    assert AU.watch_cli(args) == 4 and len(runs_of(log)) == 1
    with pytest.raises(SystemExit):
        AU.watch_cli(ap.parse_args(['--watch', str(tmp_path / 'nope.vantage')]))


# ====================================================================== CI templates
def _template_args(line):
    """The vantage arguments of a template's command line, with the job's variables filled in."""
    words = shlex.split(line.replace('$VANTAGE_PROJECT', 'rigs/rig.vantage').replace('$VANTAGE_OUT', 'vantage-out'))
    assert words[0] == 'vantage'
    return words[1:words.index('||')] if '||' in words else words[1:]


def test_ci_templates_run_the_batch_checks_in_the_repository(tmp_path):
    import argparse
    repo = tmp_path / 'repo'
    (repo / '.git').mkdir(parents=True)
    path = str(repo / 'rigs' / 'rig.vantage')
    p = data_rig(path)
    v = p.variant
    st = AU.AutomationSettings(formats=['extrinsics_json', 'opencv_yaml'], variant=v.uid, units='mm', check=True)
    files = dict(AU.ci_files(p, st, version='0.1.1', now=0))
    assert set(files) == {'ci/github/vantage.yml', 'ci/gitlab/vantage.gitlab-ci.yml', 'ci/README.md'}
    gh = yaml.safe_load(files['ci/github/vantage.yml'])
    on = gh.get('on', gh.get(True))             # YAML 1.1 reads the key "on" as true
    assert on['push']['paths'] == ['rigs/rig.vantage'] and on['pull_request']['paths'] == ['rigs/rig.vantage']
    assert 'workflow_dispatch' in on and gh['permissions'] == {'contents': 'read'}
    job = gh['jobs']['vantage']
    assert job['env']['VANTAGE_PROJECT'] == 'rigs/rig.vantage' and job['env']['QT_QPA_PLATFORM'] == 'offscreen'
    runs = [s.get('run', '') for s in job['steps']]
    export = f'vantage "$VANTAGE_PROJECT" --variant {v.uid} --units mm --export "$VANTAGE_OUT/calibration" ' \
             f'--formats extrinsics_json,opencv_yaml'
    check = f'vantage "$VANTAGE_PROJECT" --variant {v.uid} --check "$VANTAGE_OUT/review"'
    assert export in runs and check in runs
    assert any('git+https://github.com/Sherin-SEF-AI/Vantage@v0.1.1' in r for r in runs)
    uploads = [s for s in job['steps'] if s.get('uses', '').startswith('actions/upload-artifact')]
    assert [u['with']['path'] for u in uploads] == ['vantage-out/calibration', 'vantage-out/review']
    assert all(u['if'] == '${{ !cancelled() }}' for u in uploads)
    gl = yaml.safe_load(files['ci/gitlab/vantage.gitlab-ci.yml'])['vantage-checks']
    assert gl['variables']['VANTAGE_PROJECT'] == 'rigs/rig.vantage' and gl['artifacts']['when'] == 'always'
    assert gl['script'][0] == 'rc=0' and gl['script'][-1] == 'exit "$rc"'
    assert gl['script'][1] == export + ' || rc=$?' and gl['script'][2].startswith(check + ' || {')
    assert gl['rules'][1]['changes'] == ['rigs/rig.vantage']
    # every flag the templates use is one batch mode takes
    ap = argparse.ArgumentParser()
    ap.add_argument('path', nargs='?')
    from vantage import batch
    batch.add_arguments(ap)
    for line in (export, check):
        a = ap.parse_args(_template_args(line))
        assert a.path == 'rigs/rig.vantage' and a.variant == v.uid
    assert ap.parse_args(_template_args(export)).units == 'mm'
    readme = files['ci/README.md'].decode()
    assert 'rigs/rig.vantage' in readme and '.github/workflows/vantage.yml' in readme
    # without the check, and outside any repository
    st2 = AU.AutomationSettings(formats=['extrinsics_json'], check=False)
    lone = data_rig(str(tmp_path / 'lone' / 'my rig$1.vantage'))
    files2 = dict(AU.ci_files(lone, st2, root=None))
    gh2 = yaml.safe_load(files2['ci/github/vantage.yml'])
    assert not any('--check' in s.get('run', '') for s in gh2['jobs']['vantage']['steps'])
    gl2 = yaml.safe_load(files2['ci/gitlab/vantage.gitlab-ci.yml'])['vantage-checks']
    assert gl2['variables']['VANTAGE_PROJECT'] == 'my rig$$1.vantage'            # GitLab expands $: doubled
    assert gh2['jobs']['vantage']['env']['VANTAGE_PROJECT'] == 'my rig$1.vantage'
    assert 'no repository' in files2['ci/README.md'].decode()
    paths = AU.write_ci(lone, str(tmp_path / 'written'))
    assert len(paths) == 3 and all(os.path.exists(x) for x in paths)
    lone.path = str(tmp_path / '${{ secrets.TOKEN }}.vantage')
    with pytest.raises(ValueError):
        AU.ci_files(lone, st2)


# ====================================================================== the page
from test_gui import app, win, pump, settle, _wait     # noqa: E402,F401  (fixtures)


def _open(win, name='automation'):
    from vantage.ui.workspaces.automation_ws import AutomationPage
    if name not in win._page_widgets:
        pg = AutomationPage(win.ctx)
        win.pages.addWidget(pg)
        win._page_widgets[name] = pg
    win.set_page(name)
    return win._page_widgets[name]


@pytest.fixture()
def page(app, win, tmp_path):
    """The automation page on the sample's data rig, saved into the test's folder."""
    p = win.store.project
    p.active_uid = next(v for v in p.variants if v.sensors).uid
    p.path = str(tmp_path / 'proj' / 'rig.vantage')
    os.makedirs(os.path.dirname(p.path))
    win.store.edit('name', lambda q: setattr(q, 'name', 'rig'), 'name')
    assert win.save()
    settle(app, win)
    pg = _open(win)
    pump(app, 0.2)
    yield pg
    pg.ctl.shutdown()


def _answer(app, allow, seen=None):
    """Click Allow or Cancel in the approval dialog as soon as it is open."""
    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import QApplication
    from vantage.ui.workspaces.automation_ws import ApproveDialog

    tries = [0]

    def go():
        d = QApplication.activeModalWidget()
        if not isinstance(d, ApproveDialog):
            d = next((w for w in QApplication.topLevelWidgets() if isinstance(w, ApproveDialog) and w.isVisible()),
                     None)
        if d is None:
            tries[0] += 1
            if tries[0] < 200:
                QTimer.singleShot(25, go)
            return
        if seen is not None:
            seen.append(d.text.toPlainText())
        (d.allow_b if allow else d.cancel_b).click()
    QTimer.singleShot(25, go)


def _wait_idle(app, ctl, timeout=120):
    _wait(app, lambda: not ctl.busy, timeout)
    pump(app, 0.2)


def _type(app, w, text):
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    w.setFocus()
    w.selectAll()
    QTest.keyClicks(w, text)
    QTest.keyClick(w, Qt.Key.Key_Return)
    pump(app, 0.1)


def test_page_settings_undo_and_approving_the_command(app, win, page, tmp_path):
    pg, store = page, win.store
    ctl = pg.ctl
    assert win.page_name() == 'automation' and pg.isVisible() and store.project.analysis.automation == {}
    assert not pg.enabled.isChecked() and pg.b_run.isEnabled() and not pg.b_stop.isEnabled()
    pg.enabled.click()
    pump(app, 0.1)
    assert store.project.analysis.automation['enabled'] is True
    pg.trigger.buttons['change'].click()
    pump(app, 0.1)
    assert AU.settings_of(store.project).trigger == 'change' and pg.quiet.isEnabled()
    # a folder outside the project is refused and said so; one inside is kept
    _type(app, pg.folder, '../escape')
    assert AU.settings_of(store.project).out_dir == AU.DEFAULT_OUT_DIR and pg.folder_problem.isVisible()
    _type(app, pg.folder, 'out/ci')
    assert AU.settings_of(store.project).out_dir == 'out/ci' and not pg.folder_problem.isVisible()
    pg.format_boxes['report_html'].click()
    pump(app, 0.1)
    assert 'report_html' in AU.settings_of(store.project).formats
    pg.units.buttons['mm'].click()
    pump(app, 0.1)
    assert AU.settings_of(store.project).units == 'mm'
    first = store.project.variants[0]
    i = pg.variant.findData(first.uid)
    pg.variant.setCurrentIndex(i)
    pg.variant.activated.emit(i)                # as a pick from the list
    pump(app, 0.1)
    assert AU.settings_of(store.project).variant == first.uid and pg.variant.currentData() == first.uid
    for _ in range(6):
        store.undo.undo()
    pump(app, 0.1)
    assert store.project.analysis.automation == {} and not pg.enabled.isChecked()
    assert pg.folder.text() == AU.DEFAULT_OUT_DIR and not pg.format_boxes['report_html'].isChecked()
    assert pg.variant.currentIndex() == 0 and pg.units.value() == 'm'
    # the command: typed, saved, shown in full for approval; Cancel leaves it unapproved
    cmd, log = fake_command(tmp_path)
    pg.cmd.setPlainText(cmd)
    seen = []
    _answer(app, False, seen)
    pg.b_save_cmd.click()
    pump(app, 0.2)
    assert seen == [cmd] and ctl.command().command == cmd and not ctl.commands.is_approved(store.project.path)
    assert pg.cmd_state.text.text().startswith('Not approved') and pg.b_approve.isVisible()
    _answer(app, True, seen)
    pg.b_approve.click()
    pump(app, 0.2)
    assert ctl.commands.is_approved(store.project.path) and pg.cmd_state.text.text().startswith('Approved')
    assert not pg.b_approve.isVisible()
    # changed text: approval withdrawn, asked again
    pg.cmd.setPlainText(cmd + ' --more')
    _answer(app, False, seen)
    pg.b_save_cmd.click()
    pump(app, 0.2)
    assert seen[-1] == cmd + ' --more' and not ctl.commands.is_approved(store.project.path)
    # the command lives in the user's settings, never in the project or its file
    assert win.save()
    assert 'fake_check' not in json.dumps(store.project.to_json())
    with zipfile.ZipFile(store.project.path) as z:
        assert b'fake_check' not in z.read('project.json')
    assert 'fake_check' in str(win.ctx.settings.value(ctl.commands.key(store.project.path)))
    pg.b_remove_cmd.click()
    pump(app, 0.1)
    assert ctl.command() is None and pg.cmd.toPlainText() == '' and runs_of(log) == []


def test_saving_runs_the_approved_command_and_keeps_a_history(app, win, page, tmp_path):
    pg, store, ctl = page, win.store, page.ctl
    cmd, log = fake_command(tmp_path, code=0)
    pg.enabled.click()
    pump(app, 0.1)
    ctl.set_command(cmd)
    assert ctl.approve(cmd)
    pump(app, 0.1)
    f = store.variant.by_tag('F')
    store.set_cam(f.uid, x=f.x + 0.05)
    assert win.save()                           # dirty → clean with a new saved time: a save
    _wait(app, lambda: len(runs_of(log)) == 1 and not ctl.busy)
    pump(app, 0.3)
    env = runs_of(log)[0]['env']
    assert env['VANTAGE_TRIGGER'] == 'save' and env['VANTAGE_UNSAVED'] == '0'
    assert env['VANTAGE_EXPORT_DIR'] == os.path.join(os.path.dirname(store.project.path), 'exports', 'automation')
    rec = ctl.runs()[0]
    assert rec.status == 'passed' and rec.exit_code == 0 and pg.history.rowCount() == 1
    assert pg.tiles['result'].value == 'Passed' and pg.tiles['exit'].value == '0'
    assert 'line 2' in pg.out.toPlainText() and pg.files.rowCount() >= len(rec.written) - 1
    assert pg.tiles['req'].value == 'No requirements'
    # saved again with nothing changed: MainWindow.save → on_saved runs (the lead wires the call); here by hand
    store.project.save(store.project.path)
    assert ctl.on_saved(store.project.path)
    _wait(app, lambda: len(runs_of(log)) == 2 and not ctl.busy)
    pump(app, 0.3)
    assert ctl.runs()[0].n_changed() == 0 and pg.history.rowCount() == 2
    # undo back to the saved state is not a save
    store.set_cam(f.uid, x=f.x + 0.10)
    store.undo.undo()
    pump(app, 0.5)
    assert store.undo.isClean() and not ctl.busy and len(runs_of(log)) == 2
    # a changed command is not approved: the files are regenerated, the command does not run
    ctl.set_command(cmd + ' 0')
    store.set_cam(f.uid, x=f.x + 0.02)
    assert win.save()
    _wait(app, lambda: len(ctl.runs()) == 3 and not ctl.busy)
    pump(app, 0.3)
    rec = ctl.runs()[0]
    assert rec.command_state == 'not_approved' and len(runs_of(log)) == 2 and rec.status == 'passed'
    assert pg.tiles['exit'].sev == 'warning' and 'not approved' in pg.tiles['exit'].note
    # the history survives a new session (the user's config folder)
    assert [r.id for r in AU.default_history().runs(store.project.path)] == [r.id for r in ctl.runs()]
    # picking an older run shows its output
    pg.history.selectRow(2)
    pump(app, 0.1)
    assert pg._shown.id == ctl.runs()[2].id and 'line 0' in pg.out.toPlainText()
    paths = pg.export(str(tmp_path / 'run-export'))
    assert len(paths) == 2 and json.load(open(paths[0]))['run_id'] == ctl.runs()[2].id


def test_changes_run_once_after_the_quiet_period(app, win, page, tmp_path):
    pg, store, ctl = page, win.store, page.ctl
    cmd, log = fake_command(tmp_path)
    ctl.set_command(cmd)
    ctl.approve(cmd)
    store.edit('analysis', lambda q: setattr(q.analysis, 'automation', {'enabled': True, 'trigger': 'change',
                                                                        'quiet_s': 1.0, 'formats': ['extrinsics_json'],
                                                                        'check': False}), 'automation on')
    pump(app, 1.5)
    assert runs_of(log) == [] and not ctl.busy          # editing the automation itself is not a design change
    f = store.variant.by_tag('F')
    for k in range(3):                          # three edits, each well inside the quiet period
        store.set_cam(f.uid, x=f.x + 0.01)
        pump(app, 0.4)
    assert runs_of(log) == [] and not ctl.busy
    _wait(app, lambda: len(runs_of(log)) == 1 and not ctl.busy, 60)
    pump(app, 1.6)
    rows = runs_of(log)
    assert len(rows) == 1 and rows[0]['env']['VANTAGE_TRIGGER'] == 'change' and rows[0]['env']['VANTAGE_UNSAVED'] == '1'
    assert ctl.runs()[0].unsaved and any('Unsaved edits' in w.text.text() for w in pg._note_widgets)
    # the exported design is the one in the window, unsaved edit included
    ex = json.load(open(os.path.join(rows[0]['env']['VANTAGE_EXPORT_DIR'], 'extrinsics.json')))
    fx = next(c for c in ex['cameras'] if c['tag'] == 'F')
    assert fx['position'][0] == pytest.approx(store.variant.by_tag('F').x, abs=1e-6)
    assert store.variant.by_tag('F').x == pytest.approx(f.x, abs=1e-9) and not store.undo.isClean()


def test_stop_decline_and_another_project(app, win, page, tmp_path):
    pg, store, ctl = page, win.store, page.ctl
    path = store.project.path
    cmd, log = fake_command(tmp_path, code=0, sleep=30)
    store.edit('analysis', lambda q: setattr(q.analysis, 'automation', {'formats': ['extrinsics_json'],
                                                                        'check': False}), 'automation')
    ctl.set_command(cmd)
    # Run now with a command nobody approved: it is shown first; declined, nothing runs
    _answer(app, False)
    assert pg.run_now() is False
    pump(app, 0.2)
    assert not ctl.busy and ctl.runs() == [] and runs_of(log) == []
    # approved in the dialog: it runs; Stop ends it and what it started
    _answer(app, True)
    assert pg.run_now() is True
    _wait(app, lambda: len(runs_of(log)) == 1, 60)
    assert pg.b_stop.isEnabled() and not pg.b_run.isEnabled()
    pid = runs_of(log)[0]['pid']
    t = time.time()
    pg.b_stop.click()
    _wait_idle(app, ctl, 20)
    assert time.time() - t < 10 and ctl.runs()[0].status == 'cancelled' and pg.tiles['result'].value == 'Stopped'
    deadline = time.time() + 5
    while _alive(pid) and time.time() < deadline:
        time.sleep(0.05)
    assert not _alive(pid)
    # another project: no command, nothing to show; opening a project is not a save
    other = data_rig(str(tmp_path / 'other' / 'other.vantage'))
    other.analysis.automation = {'enabled': True, 'trigger': 'save'}
    other.save(other.path)
    store.set_cam(store.variant.cameras[0].uid, x=store.variant.cameras[0].x + 0.01)     # dirty, then replaced
    win.store.set_project(Project.load(other.path))
    pump(app, 0.5)
    assert not ctl.busy and ctl.runs() == [] and pg.history.rowCount() == 0
    assert pg.cmd.toPlainText() == '' and ctl.command() is None and pg.enabled.isChecked()
    # back to the first project: its command and its history come back
    win.store.set_project(Project.load(path))
    pump(app, 0.5)
    assert pg.cmd.toPlainText() == cmd and ctl.commands.is_approved(path) and pg.history.rowCount() == 1
    # an unsaved project cannot run (the folder and the command belong to its file)
    win.store.set_project(data_rig())
    pump(app, 0.3)
    assert not pg.b_run.isEnabled() and not pg.cmd.isEnabled() and pg.run_now() is False
    assert 'Save the project' in pg.status.text()
