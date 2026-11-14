"""Project folders (core.project_dir): the round trip zip -> folder -> zip, stable bytes, safe saving, git merges of
two people's edits (with and without Vantage's merge driver), the semantic diff and the three-way merge.

git runs only inside each test's own temporary repository, with the user's git configuration shut out."""
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

from vantage.core import project_dir as PD
from vantage.core import samples
from vantage.core.project import Project
from vantage.core.provenance import project_digest

HAVE_GIT = shutil.which('git') is not None
needs_git = pytest.mark.skipif(not HAVE_GIT, reason='needs git')


def _env(tmp):
    env = dict(os.environ)
    for k in list(env):
        if k.startswith('GIT_'):
            env.pop(k)
    env.update({'GIT_CONFIG_GLOBAL': os.devnull, 'GIT_CONFIG_NOSYSTEM': '1', 'GIT_AUTHOR_NAME': 'Test',
                'GIT_AUTHOR_EMAIL': 'test@example.invalid', 'GIT_COMMITTER_NAME': 'Test',
                'GIT_COMMITTER_EMAIL': 'test@example.invalid', 'HOME': str(tmp), 'GIT_TERMINAL_PROMPT': '0'})
    return env


def git(repo, *args, check=True):
    r = subprocess.run(['git', '-C', str(repo), *args], capture_output=True, text=True, env=_env(repo))
    if check and r.returncode != 0:
        raise AssertionError(f'git {args}: {r.stdout}\n{r.stderr}')
    return r


def _repo(tmp_path):
    repo = tmp_path / 'repo'
    repo.mkdir(parents=True)
    git(repo, 'init', '-q', '-b', 'main')
    return repo


@pytest.fixture(scope='module')
def sample_file(tmp_path_factory):
    d = tmp_path_factory.mktemp('sample')
    p = samples.tigor_sample_project()
    p.fleet = {'vehicles': [{'id': 'v1', 'name': 'V1', 'records': [
        {'id': 'r1', 'date': '2026-01-05', 'kind': 'manual', 'poses': {'F': {'x': 3.2, 'y': 0.0, 'z': 0.6, 'yaw': 0.0,
                                                                              'pitch': 40.0, 'roll': 0.0}}}]}]}
    p.analysis.assistant_log = [{'id': 'e1', 'kind': 'decision', 'text': 'Front camera stays in the grille',
                                 'about': [], 'date': '2026-01-04'}]
    p.analysis.recording = {'hours_per_day': 6.0}
    p.analysis.requirements = {'items': [{'id': 'q1', 'text': 'see a child at 2 m'}]}
    p.streams = {'F': 'rtsp://admin:hunter2@10.0.0.5:554/stream1'}
    p.thumb_png = b'\x89PNG\r\n\x1a\nfake'
    f = str(d / 'sample.vantage')
    p.save(f)
    return f


# ----------------------------------------------------------------------------------------------- round trip
@pytest.mark.parametrize('key', ['tigor', 'sedan', 'suv'])
@pytest.mark.parametrize('text_mesh', [False, True])
def test_round_trip_zip_folder_zip_is_identical(tmp_path, key, text_mesh):
    p = samples.template_project(key)
    r = PD.validate_round_trip(p, str(tmp_path), text_mesh=text_mesh)
    assert r['equal'], (r['zip_to_dir'], r['dir_to_zip'])
    assert r['stable_bytes']
    assert r['zip_to_dir']['digest_a'] == r['zip_to_dir']['digest_b'] == r['dir_to_zip']['digest_b']
    names = PD.list_files(str(tmp_path / 'a.vantage.d'))
    assert ('mesh.obj' in names) == text_mesh and ('mesh.npz' in names) != text_mesh


def test_round_trip_of_a_full_project(tmp_path, sample_file):
    p1 = Project.load(sample_file)
    d = tmp_path / 'rig.vantage.d'
    p1.save(str(d))                                       # Project.save: a .vantage.d path is a folder
    assert d.is_dir() and (d / 'project.json').is_file()
    p2 = Project.load(str(d))                             # Project.load: a folder
    c = PD.content_equal(p1, p2)
    assert c['equal'], c
    assert p2.thumb_png == p1.thumb_png and p2.active_uid == p1.active_uid
    assert p2.analysis.assistant_log == p1.analysis.assistant_log
    assert p2.analysis.requirements == p1.analysis.requirements and p2.fleet == p1.fleet
    names = set(PD.list_files(str(d)))
    for n in ('requirements.json', 'design_log.jsonl', 'fleet.json', 'analysis/recording.json', 'thumb.png',
              'library/lenses.json', '.vantage-view.json'):
        assert n in names
    assert len([n for n in names if n.startswith('variants/')]) == len(p1.variants)
    z = tmp_path / 'back.vantage'
    p2.save(str(z))
    p3 = Project.load(str(z))
    assert PD.content_equal(p1, p3)['equal']
    assert project_digest(p1) == project_digest(p3)


def test_stable_bytes_no_save_time_and_unchanged_files_not_rewritten(tmp_path, sample_file):
    p = Project.load(sample_file)
    d = str(tmp_path / 'x.vantage.d')
    r1 = PD.save_dir(p, d)
    assert 'project.json' in r1['written']
    before = {n: os.stat(os.path.join(d, n)).st_mtime_ns for n in PD.list_files(d)}
    r2 = PD.save_dir(Project.load(d), d)                # loaded and saved again: nothing to write
    assert r2['written'] == [] and r2['removed'] == []
    assert before == {n: os.stat(os.path.join(d, n)).st_mtime_ns for n in PD.list_files(d)}
    txt = open(os.path.join(d, 'project.json'), encoding='utf-8').read()
    assert 'saved_at' not in txt and txt.startswith('{\n  "$vantage": "project/1"') and txt.endswith('}\n')
    # the same project gives the same bytes; numbers read back exactly
    p.variants[0].cameras[0].x = 0.1 + 0.2
    a, b = PD.files_of(p), PD.files_of(p)
    assert a == b
    q = PD.project_from_files(a)
    assert q.variants[0].cameras[0].x == 0.1 + 0.2
    # the mesh NPZ is the same bytes every time (fixed member times)
    assert PD._npz_bytes(PD.mesh_arrays(p.vehicle.mesh)) == PD._npz_bytes(PD.mesh_arrays(p.vehicle.mesh))


def test_one_edit_changes_only_its_own_file(sample_file):
    p = Project.load(sample_file)
    a = PD.files_of(p)
    v = p.variants[2]
    v.cameras[0].yaw += 1.0
    b = PD.files_of(p)
    changed = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    assert changed == [next(k for k in b if k.startswith('variants/') and json.loads(b[k])['uid'] == v.uid)]
    p.analysis.tol_mm = 33.0
    c = PD.files_of(p)
    assert sorted(k for k in c if c[k] != b.get(k)) == ['analysis/base.json']


def test_saving_never_deletes_files_it_did_not_write(tmp_path, sample_file):
    p = Project.load(sample_file)
    d = tmp_path / 'p.vantage.d'
    PD.save_dir(p, str(d))
    (d / 'README.md').write_text('notes')
    (d / 'variants' / 'notes.json').write_text('{"mine": true}\n')       # not Vantage's: no mark
    (d / 'analysis' / 'keep.json').write_text('{"x": 1}\n')
    gone = p.variants.pop(1)
    p.analysis.recording = {}
    r = PD.save_dir(p, str(d))
    removed = set(r['removed'])
    assert 'analysis/recording.json' in removed
    assert any(x.startswith('variants/') for x in removed) and len(removed) == 2
    assert (d / 'README.md').exists() and (d / 'variants' / 'notes.json').exists() and (d / 'analysis' / 'keep.json').exists()
    q = Project.load(str(d))
    assert [v.uid for v in q.variants] == [v.uid for v in p.variants] and gone.uid not in {v.uid for v in q.variants}
    assert any('notes.json is not a Vantage variant' in e.text for e in q.log)
    # the user's own .gitattributes is theirs
    (d / '.gitattributes').write_text('* text\n')
    PD.save_dir(p, str(d))
    assert (d / '.gitattributes').read_text() == '* text\n'


def test_a_folder_of_other_files_is_refused(tmp_path, sample_file):
    p = Project.load(sample_file)
    d = tmp_path / 'home'
    d.mkdir()
    (d / 'thesis.tex').write_text('x')
    with pytest.raises(PD.ProjectDirError, match='holds other files'):
        p.save(str(d))
    assert sorted(os.listdir(d)) == ['thesis.tex']
    PD.save_dir(p, str(d), into_existing=True)                 # on purpose (e.g. a repository's root)
    assert (d / 'project.json').exists() and (d / 'thesis.tex').exists()
    with pytest.raises(PD.ProjectDirError, match='a file'):
        PD.save_dir(p, str(d / 'thesis.tex'))


def test_paths_names_thumbnail_and_switching_mesh_format(tmp_path, sample_file):
    assert PD.wants_dir('a/rig.vantage.d') and not PD.wants_dir('a/rig.vantage')
    assert PD.name_from_path('/x/rig.vantage.d/') == 'rig' and PD.name_from_path('/x/rig.vantage') == 'rig'
    assert PD.default_dir_for('/x/rig.vantage') == '/x/rig.vantage.d'
    p = Project.load(sample_file)
    d = tmp_path / 'rig.vantage.d'
    p.save(str(d), text_mesh=True)
    assert (d / 'mesh.obj').exists() and not (d / 'mesh.npz').exists()
    p.save(str(d))                                              # None keeps the folder's text mesh
    assert (d / 'mesh.obj').exists()
    p.save(str(d), text_mesh=False)
    assert (d / 'mesh.npz').exists() and not (d / 'mesh.obj').exists()
    assert PD.is_project_dir(str(d)) and PD.is_project_path(str(d)) and PD.is_project_path(sample_file)
    thumb, info = PD.read_thumbnail(str(d))
    assert thumb == p.thumb_png and info['variants'] == len(p.variants) and info['cameras'] == len(p.variant.cameras)
    assert PD.mtime(str(d)) >= os.path.getmtime(d / 'project.json')
    b = PD.backup(str(d))
    assert os.path.isfile(os.path.join(b, 'project.json'))
    assert PD.content_equal(Project.load(b), Project.load(str(d)))['equal']


def test_obj_reader_takes_a_plain_obj(tmp_path):
    a = PD.obj_arrays(b'v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\nf -4 -2 -1\n')
    assert a['faces'].tolist() == [[0, 1, 2], [0, 2, 3], [0, 2, 3]]
    assert a['vertices'].dtype == np.float32 and a['face_part'].tolist() == [0, 0, 0]
    with pytest.raises(PD.ProjectDirError):
        PD.obj_arrays(b'v 0 0 0\nf 1 2 3\n')


def test_conflict_markers_are_named(tmp_path, sample_file):
    p = Project.load(sample_file)
    d = tmp_path / 'c.vantage.d'
    PD.save_dir(p, str(d))
    f = next((d / 'variants').iterdir())
    f.write_text('<<<<<<< HEAD\n{}\n=======\n{}\n>>>>>>> theirs\n')
    with pytest.raises(PD.ProjectDirError, match='conflict markers'):
        Project.load(str(d))


def test_no_passwords_in_the_folder(tmp_path, sample_file):
    p = Project.load(sample_file)
    assert p.streams['F'] == 'rtsp://admin@10.0.0.5:554/stream1'
    p.streams['B'] = 'rtsp://root:s3cr3t-pw@10.0.0.6/live'
    files = PD.files_of(p)
    blob = b''.join(files.values())
    assert b's3cr3t-pw' not in blob and b'hunter2' not in blob


# ----------------------------------------------------------------------------------------------- three-way merge
def test_merge3_by_identity_and_conflicts():
    base = {'cams': [{'uid': 'a', 'x': 1}, {'uid': 'b', 'x': 2}], 's': 1, 't': 'k'}
    ours = {'cams': [{'uid': 'a', 'x': 5}, {'uid': 'b', 'x': 2}, {'uid': 'c', 'x': 9}], 's': 1, 't': 'k'}
    theirs = {'cams': [{'uid': 'z', 'x': 0}, {'uid': 'a', 'x': 1}, {'uid': 'b', 'x': 7}], 's': 2, 't': 'k'}
    c = []
    m = PD.merge3(base, ours, theirs, '', c)
    assert c == []
    assert m == {'cams': [{'uid': 'z', 'x': 0}, {'uid': 'a', 'x': 5}, {'uid': 'b', 'x': 7}, {'uid': 'c', 'x': 9}],
                 's': 2, 't': 'k'}
    # deleted on one side, unchanged on the other: deleted; changed both ways: ours and a conflict
    c = []
    m = PD.merge3(base, {'cams': [{'uid': 'a', 'x': 1}], 's': 3, 't': 'k'},
                  {'cams': [{'uid': 'a', 'x': 1}, {'uid': 'b', 'x': 2}], 's': 4, 't': 'k'}, '', c)
    assert m == {'cams': [{'uid': 'a', 'x': 1}], 's': 3, 't': 'k'}
    assert [x['path'] for x in c] == ['s'] and c[0]['theirs'] == 4
    # deleted on one side, changed on the other: a conflict
    c = []
    PD.merge3(base, {'cams': [{'uid': 'a', 'x': 1}], 's': 1, 't': 'k'},
              {'cams': [{'uid': 'a', 'x': 1}, {'uid': 'b', 'x': 8}], 's': 1, 't': 'k'}, '', c)
    assert [x['path'] for x in c] == ['cams[b]']
    # the history log is a union, in time order
    c = []
    m = PD.merge3({'log': [[1, 'e', 'a']]}, {'log': [[1, 'e', 'a'], [3, 'e', 'ours']]},
                  {'log': [[1, 'e', 'a'], [2, 'e', 'theirs']]}, '', c)
    assert m['log'] == [[1, 'e', 'a'], [2, 'e', 'theirs'], [3, 'e', 'ours']] and not c


def test_merge_of_two_project_json_edits_and_zips(tmp_path, sample_file):
    p = Project.load(sample_file)
    base = json.dumps(p.to_json()).encode()
    o, t = Project.load(sample_file), Project.load(sample_file)
    o.variants[0].cameras[0].x += 0.1
    o.add_log('edit', 'ours')
    t.variants[0].cameras[0].y += 0.2                         # the same camera, another field
    t.analysis.tol_deg = 2.0
    t.add_log('edit', 'theirs')
    merged, conf = PD.merge_bytes(base, json.dumps(o.to_json()).encode(), json.dumps(t.to_json()).encode(),
                                  'project.json')
    assert not conf
    m = json.loads(merged)
    cam = m['variants'][0]['cameras'][0]
    assert cam['x'] == o.variants[0].cameras[0].x and cam['y'] == t.variants[0].cameras[0].y
    assert m['analysis']['tol_deg'] == 2.0 and {'ours', 'theirs'} <= {e[2] for e in m['log']}
    # zipped .vantage files: the same, mesh kept
    zb, zo, zt = (tmp_path / 'b.vantage', tmp_path / 'o.vantage', tmp_path / 't.vantage')
    p.save(str(zb))
    o.save(str(zo))
    t.save(str(zt))
    rc = PD.merge_file(str(zb), str(zo), str(zt), 'rig.vantage')
    assert rc == 0
    q = Project.load(str(zo))
    assert q.variants[0].cameras[0].x == o.variants[0].cameras[0].x and q.analysis.tol_deg == 2.0
    assert q.vehicle.mesh is not None and q.vehicle.mesh.n_tris == p.vehicle.mesh.n_tris


# ----------------------------------------------------------------------------------------------- git
def _commit(repo, msg):
    git(repo, 'add', '-A')
    git(repo, 'commit', '-q', '-m', msg)


@needs_git
def test_git_merges_two_people_editing_different_variants(tmp_path, sample_file):
    repo = _repo(tmp_path)
    d = repo / 'rig.vantage.d'
    p = Project.load(sample_file)
    p.save(str(d))
    _commit(repo, 'base')
    assert '.vantage-view.json' not in git(repo, 'ls-files').stdout       # per user: not shared
    git(repo, 'checkout', '-q', '-b', 'alice')
    a = Project.load(str(d))
    a.active_uid = a.variants[0].uid
    a.variants[0].cameras[0].x += 0.05
    a.add_log('edit', 'Alice moved F')
    a.save(str(d))
    _commit(repo, 'alice')
    git(repo, 'checkout', '-q', 'main')
    b = Project.load(str(d))
    b.active_uid = b.variants[-1].uid
    b.variants[-1].sensors[0].z += 0.1
    b.variants[-1].note = 'Bob raised the lidar'
    b.add_log('edit', 'Bob raised L1')
    b.analysis.recording = {'hours_per_day': 8.0}
    b.save(str(d))
    _commit(repo, 'bob')
    r = git(repo, 'merge', '-q', '--no-edit', 'alice', check=False)
    assert r.returncode == 0, r.stdout + r.stderr                         # git's own merge, no driver needed
    m = Project.load(str(d))
    assert m.variants[0].cameras[0].x == a.variants[0].cameras[0].x
    assert m.variants[-1].sensors[0].z == b.variants[-1].sensors[0].z and m.analysis.recording == {'hours_per_day': 8.0}
    texts = [e.text for e in m.log]
    assert 'Alice moved F' in texts and 'Bob raised L1' in texts
    # the semantic diff of the two revisions
    doc = PD.diff(*[PD.load_spec(str(d), rev)[0] for rev in ('HEAD~1', 'HEAD')])
    assert doc['summary']['cameras_moved'] == 1 and doc['summary']['sensors_moved'] == 0


@needs_git
def test_merge_driver_merges_two_fields_of_one_camera(tmp_path, sample_file):
    """x and y of the same camera are adjacent lines: git's line merge conflicts; Vantage's driver does not."""
    def run(with_driver):
        repo = _repo(tmp_path / ('drv' if with_driver else 'plain'))
        d = repo / 'rig.vantage.d'
        Project.load(sample_file).save(str(d))
        if with_driver:
            env_cmd = f'"{sys.executable}" -m vantage.core.project_dir merge-file %O %A %B %P'
            info = PD.git_setup(str(d), command=env_cmd)
            assert os.path.realpath(info['repository']) == os.path.realpath(repo)
        _commit(repo, 'base')
        git(repo, 'checkout', '-q', '-b', 'other')
        q = Project.load(str(d))
        q.variants[0].cameras[0].y += 0.25
        q.save(str(d))
        _commit(repo, 'other')
        git(repo, 'checkout', '-q', 'main')
        q = Project.load(str(d))
        q.variants[0].cameras[0].x += 0.5
        q.save(str(d))
        _commit(repo, 'main')
        return repo, d, git(repo, 'merge', '-q', '--no-edit', 'other', check=False)
    _, _, r = run(False)
    assert r.returncode != 0 and 'CONFLICT' in r.stdout + r.stderr
    old_pp = os.environ.get('PYTHONPATH', '')
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(PD.__file__))))
    os.environ['PYTHONPATH'] = os.pathsep.join(x for x in (root, old_pp) if x)
    try:
        repo, d, r = run(True)
    finally:
        os.environ['PYTHONPATH'] = old_pp
    assert r.returncode == 0, r.stdout + r.stderr
    m = Project.load(str(d))
    base = Project.load(sample_file).variants[0].cameras[0]
    assert m.variants[0].cameras[0].x == base.x + 0.5 and m.variants[0].cameras[0].y == base.y + 0.25


@needs_git
def test_diff_command_on_files_folders_and_revisions(tmp_path, sample_file, capsys):
    repo = _repo(tmp_path)
    d = repo / 'rig.vantage.d'
    p = Project.load(sample_file)
    p.save(str(d))
    _commit(repo, 'base')
    v = next(x for x in p.variants if x.sensors)
    v.cameras[0].x += 0.050
    v.cameras[0].z += 0.014
    v.cameras[0].yaw += 2.0
    old_lens = v.cameras[1].lens_id
    v.cameras[1].lens_id = next(k for k in p.lenses if k != old_lens)
    from vantage.core.project import Sensor
    v.sensors.append(Sensor(tag='G1', name='GNSS', kind='gnss', model_id='', x=1.0, y=0.0, z=1.6))
    p.variants[1].name = 'Four cameras'
    p.variants.pop(2)
    p.analysis.tol_mm = 42.0
    p.save(str(d))
    assert PD.main(['diff', f'{d}@HEAD', str(d)]) == PD.EXIT_DIFFERENT
    text = capsys.readouterr().out
    assert f'camera {v.cameras[0].tag}: moved 51.9 mm (x +50.0, y +0.0, z +14.0), turned 2.00° (yaw +2.00°)' in text
    assert f'lens_id: "{old_lens}" → "{v.cameras[1].lens_id}"' in text
    assert 'sensor G1 added' in text and '(renamed from "4-cam")' in text and 'removed (' in text
    assert 'Analysis tol_mm: 100 → 42' in text
    out = tmp_path / 'd.json'
    assert PD.main(['diff', f'{d}@HEAD', '--json', str(out)]) == PD.EXIT_DIFFERENT
    doc = json.loads(out.read_text())
    assert doc['schema'] == PD.DIFF_SCHEMA and doc['summary']['variants_removed'] == 1
    assert doc['summary']['sensors_added'] == 1 and doc['summary']['lens_changes'] == 1
    capsys.readouterr()
    assert PD.main(['diff', str(d), str(d)]) == PD.EXIT_SAME
    assert 'No difference' in capsys.readouterr().out
    z = tmp_path / 'z.vantage'
    Project.load(str(d)).save(str(z))
    assert PD.main(['diff', str(z), str(d)]) == PD.EXIT_SAME                 # a file against a folder
    assert PD.main(['diff', str(tmp_path / 'nope.vantage'), str(d)]) == PD.EXIT_ERROR
    # a zipped .vantage tracked in git, at a revision
    zr = repo / 'old.vantage'
    Project.load(sample_file).save(str(zr))
    _commit(repo, 'zip')
    q, label = PD.load_spec(f'{zr}@HEAD')
    assert label == 'old.vantage@HEAD' and PD.content_equal(q, Project.load(sample_file))['equal']


def test_diff_known_values_and_nothing_for_the_same(sample_file):
    a = Project.load(sample_file)
    assert PD.diff(a, Project.load(sample_file))['same']
    b = Project.load(sample_file)
    s = next(x for x in b.variants if x.sensors).sensors[0]
    s.pitch += 1.5
    s.yaw -= 0.5
    d = PD.diff(a, b)
    mv = next(i for v in d['variants'] if v['status'] == 'changed' for i in v['items'])['move']
    from vantage.core.mathutil import rot_zyx
    Ra = rot_zyx(s.yaw + 0.5, s.pitch - 1.5, s.roll)
    Rb = rot_zyx(s.yaw, s.pitch, s.roll)
    expect = np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1)))
    assert mv['rotation_deg'] == pytest.approx(expect, abs=1e-9) and mv['moved_mm'] == 0
    assert mv['dpitch_deg'] == pytest.approx(1.5) and mv['dyaw_deg'] == pytest.approx(-0.5)


class _Ctx:
    def __init__(self, tmp):
        self.project = samples.tigor_sample_project()
        os.makedirs(tmp, exist_ok=True)
        self._tmp = str(tmp)

    def sample(self):
        return samples.tigor_sample_project()

    def tmp(self):
        return self._tmp


def test_validation_checks(tmp_path):
    checks = PD.validation_checks()
    assert {c['id'] for c in checks} == {'formats.project_dir_round_trip', 'formats.project_diff_known_edits',
                                         'formats.project_merge_two_edits'}
    for c in checks:
        out = c['fn'](_Ctx(tmp_path / c['id']))
        assert out.error <= c['tolerance'] and out.ok, (c['id'], out)
        assert {'model', 'title', 'quantity', 'reference', 'unit', 'refs', 'cost'} <= set(c)
