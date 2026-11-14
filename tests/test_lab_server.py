"""The lab server (vantage.lab): its REST API in process on localhost (a free port the system picks), token auth and
per-team access, immutable digest-addressed versions, concurrent publishes, sync with a three-way merge, the pooled
fleet twin, pooled perception records, and that no password, key or token ever reaches the server."""
import copy
import hashlib
import json
import os
import threading

import numpy as np
import pytest

from vantage.core import fleet as FL
from vantage.core import fleet_twin as FT
from vantage.core import perf_model as PM
from vantage.core import project_dir as PD
from vantage.core import samples
from vantage.core.project import Project
from vantage.lab import client as LC
from vantage.lab import server as LS
from vantage.lab.store import LabStore, tree_digest


@pytest.fixture()
def lab(tmp_path):
    srv = LS.make_server(str(tmp_path / 'lab'), '127.0.0.1', 0, quiet=True)
    th = LS.serve_in_thread(srv)
    st = srv.store
    toks = {'admin': st.add_token('ops', 'admin', 'root')['token'],
            'alice': st.add_token('alpha', 'member', 'alice')['token'],
            'bob': st.add_token('alpha', 'member', 'bob')['token'],
            'carol': st.add_token('beta', 'member', 'carol')['token'],
            'rita': st.add_token('beta', 'reader', 'rita')['token']}
    yield srv, toks, {k: LC.LabClient(srv.url, v) for k, v in toks.items()}
    srv.shutdown()
    srv.server_close()
    th.join(5)


def _status(fn, *a, **k):
    try:
        fn(*a, **k)
    except LC.LabClientError as e:
        return e.status
    return 200


@pytest.fixture(scope='module')
def sample(tmp_path_factory):
    """The sample as a file holds it (float32 mesh): what a pulled copy is compared with."""
    f = str(tmp_path_factory.mktemp('s') / 'sample.vantage')
    samples.tigor_sample_project().save(f)
    return Project.load(f)


# ----------------------------------------------------------------------------------------------- auth and access
def test_health_auth_refusals_and_roles(lab):
    srv, toks, c = lab
    assert c['alice'].health()['service'] == 'vantage-lab'
    assert LC.LabClient(srv.url, '').health()['api'] == 1                     # no token needed
    assert c['alice'].whoami() == {'team': 'alpha', 'role': 'member', 'token': 2, 'name': 'alice'}
    assert _status(LC.LabClient(srv.url, 'vlab_not-a-token').whoami) == 401
    assert _status(LC.LabClient(srv.url, None).request, 'GET', 'whoami', auth=False) == 401
    with pytest.raises(LC.LabClientError, match='no token'):
        LC.LabClient(srv.url, '').whoami()
    pa = c['alice'].register('Data rig', share_fleet=False)
    assert pa['id'] == 'alpha/data-rig' and pa['head'] is None
    assert _status(c['bob'].register, 'Data rig') == 409                      # the name is taken
    assert _status(c['rita'].register, 'mine') == 403                         # readers cannot register
    assert _status(c['carol'].project, 'alpha/data-rig') == 403               # another team's private project
    assert [p['id'] for p in c['carol'].projects()] == []
    assert _status(c['carol'].patch, 'alpha/data-rig', readers=['beta']) == 403
    assert _status(c['alice'].project, 'alpha/nope') == 404
    assert _status(c['alice'].request, 'GET', 'tokens') == 403                # admin only
    assert len(c['admin'].request('GET', 'tokens')['tokens']) == 5
    new = c['admin'].request('POST', 'tokens', {'team': 'gamma', 'role': 'member', 'name': 'gil'})
    assert LC.LabClient(srv.url, new['token']).whoami()['team'] == 'gamma'
    c['admin'].request('DELETE', f'tokens/{new["id"]}')
    assert _status(LC.LabClient(srv.url, new['token']).whoami) == 401          # revoked
    assert _status(c['alice'].request, 'DELETE', 'projects') == 405
    assert _status(c['alice'].request, 'GET', 'nothing/here') == 404
    # sharing: beta reads alpha's project once alpha lets it
    c['alice'].patch('alpha/data-rig', readers=['beta'])
    assert c['carol'].project('alpha/data-rig')['readers'] == ['beta']
    assert c['rita'].project('alpha/data-rig')['id'] == 'alpha/data-rig'
    # nothing secret in the store: tokens only as their SHA-256
    db = open(os.path.join(srv.store.root, 'lab.sqlite3'), 'rb').read()
    for t in toks.values():
        assert t.encode() not in db and hashlib.sha256(t.encode()).hexdigest().encode() in db


# ----------------------------------------------------------------------------------------------- versions
def test_publish_pull_compare_and_immutability(lab, sample, tmp_path):
    srv, toks, c = lab
    a = c['alice']
    pid = a.register('rig')['id']
    p = copy.deepcopy(sample)
    v1 = a.publish(p, pid, 'first', parent=0)
    files = LC.files_to_publish(p)
    assert v1['number'] == 1 and not v1['existing'] and v1['digest'] == LC.digest_of(files)
    assert set(v1['manifest']) == set(files) and '.vantage-view.json' not in files
    assert v1['upload']['uploaded'] == len(set(LC.manifest_of(files).values()))
    again = a.publish(copy.deepcopy(p), pid, 'same again')                   # the same files: the same version
    assert again['existing'] and again['number'] == 1 and again['upload']['uploaded'] == 0
    v = next(x for x in p.variants if x.sensors)
    v.cameras[0].x += 0.1
    v.cameras[0].yaw -= 1.0
    v2 = a.publish(p, pid, 'moved F')
    assert v2['number'] == 2 and v2['parent'] == 1
    assert v2['upload']['uploaded'] == 1                                     # only the changed variant's file
    assert [x['number'] for x in a.versions(pid)] == [1, 2]
    assert a.version(pid, v1['digest'][:16])['number'] == 1                  # digest-addressed
    q = a.pull(pid, 1, dest=str(tmp_path / 'pulled.vantage.d'))
    assert PD.content_equal(q, sample)['equal']
    assert LC.sync_state(str(tmp_path / 'pulled.vantage.d'))['version'] == 1
    q2 = a.pull(pid, 'head')
    assert PD.content_equal(q2, p)['equal']
    cmp = c['alice'].compare(pid, 1, 2)
    assert cmp['diff']['summary']['cameras_moved'] == 1
    assert f'camera {v.cameras[0].tag}: moved 100.0 mm (x +100.0, y +0.0, z +0.0), turned 1.00° (yaw -1.00°)' in \
        cmp['text']
    # a version is immutable: its digest is its files' and the store holds each file once, by SHA-256
    man = a.version(pid, 1)['manifest']
    assert tree_digest(man) == v1['digest']
    for path, (sha, size) in man.items():
        blob = open(srv.store.blob_path(sha), 'rb').read()
        assert hashlib.sha256(blob).hexdigest() == sha and len(blob) == size
    # a stale parent is refused
    stale = copy.deepcopy(sample)
    stale.analysis.tol_mm = 5.0
    assert _status(a.publish_files, pid, LC.files_to_publish(stale), 'stale', parent=1) == 409
    assert a.publish_files(pid, LC.files_to_publish(sample), 'old files', parent=1)['existing']   # v1's files: v1


def test_bad_uploads_and_manifests_are_refused(lab):
    srv, toks, c = lab
    a = c['alice']
    pid = a.register('x')['id']
    assert _status(a.request, 'PUT', 'blobs/' + '0' * 64, b'not that') == 400
    assert _status(c['rita'].request, 'PUT', 'blobs/' + hashlib.sha256(b'x').hexdigest(), b'x') == 403
    sha = hashlib.sha256(b'{}').hexdigest()
    a.request('PUT', f'blobs/{sha}', b'{}')
    for man in ({'../etc/passwd': sha, 'project.json': sha}, {'/abs.json': sha, 'project.json': sha},
                {'a\\b.json': sha, 'project.json': sha}, {'variants/x.json': sha}, {}):
        assert _status(a.request, 'POST', f'projects/{pid}/versions', {'manifest': man, 'parent': 0}) == 400
    try:
        a.request('POST', f'projects/{pid}/versions', {'manifest': {'project.json': 'a' * 64}, 'parent': 0})
    except LC.LabClientError as e:
        assert e.status == 400 and e.extra['missing'] == ['a' * 64]
    # another team cannot read a file of a private project, even knowing its SHA-256
    a.request('POST', f'projects/{pid}/versions', {'manifest': {'project.json': sha}, 'parent': 0})
    assert _status(c['carol'].request, 'GET', f'blobs/{sha}', raw=True) == 404
    assert a.request('GET', f'blobs/{sha}', raw=True) == b'{}'
    assert _status(c['carol'].request, 'POST', f'projects/{pid}/versions', {'manifest': {'project.json': sha}}) == 403


def test_two_publishes_at_once_are_consistent(lab, sample):
    srv, toks, c = lab
    pid = c['alice'].register('race')['id']
    c['alice'].publish(copy.deepcopy(sample), pid, 'base', parent=0)
    edits = []
    for i in range(2):
        p = copy.deepcopy(sample)
        p.variants[i].cameras[0].z += 0.01 * (i + 1)
        edits.append(LC.files_to_publish(p))
    for f in edits:                                                           # files up first: the race is the publish
        c['alice'].upload(f)
    barrier = threading.Barrier(2)
    out = [None, None]

    def go(i, who, parent):
        barrier.wait()
        try:
            out[i] = c[who].request('POST', f'projects/{pid}/versions',
                                    {'manifest': LC.manifest_of(edits[i]), 'parent': parent, 'message': who})
        except LC.LabClientError as e:
            out[i] = e.status
    ths = [threading.Thread(target=go, args=(i, w, 1)) for i, w in enumerate(('alice', 'bob'))]
    [t.start() for t in ths]
    [t.join() for t in ths]
    won = [x for x in out if isinstance(x, dict)]
    assert len(won) == 1 and won[0]['number'] == 2 and out.count(409) == 1     # one lands, the other must merge
    # without a parent check every publish lands, numbered densely; identical content is one version
    n0 = 2
    files = []
    for i in range(6):
        p = copy.deepcopy(sample)
        p.analysis.tol_mm = 10.0 + i
        files.append(LC.files_to_publish(p))
        c['alice'].upload(files[-1])
    files.append(files[0])                                                    # the same files twice
    res = [None] * len(files)
    barrier = threading.Barrier(len(files))

    def go2(i):
        barrier.wait()
        res[i] = c['alice' if i % 2 else 'bob'].request(
            'POST', f'projects/{pid}/versions', {'manifest': LC.manifest_of(files[i]), 'parent': None})
    ths = [threading.Thread(target=go2, args=(i,)) for i in range(len(files))]
    [t.start() for t in ths]
    [t.join() for t in ths]
    nums = sorted({r['number'] for r in res})
    assert nums == list(range(n0 + 1, n0 + 7))
    assert res[0]['number'] == res[-1]['number']
    assert [v['number'] for v in c['alice'].versions(pid)] == list(range(1, n0 + 7))


def test_sync_merges_two_people_editing_different_variants(lab, sample, tmp_path):
    srv, toks, c = lab
    pid = c['alice'].register('shared')['id']
    da, db = str(tmp_path / 'alice.vantage.d'), str(tmp_path / 'bob.vantage.d')
    p = copy.deepcopy(sample)
    p.save(da)
    r = c['alice'].sync(Project.load(da), pid, 'first')
    assert r['status'] == 'published' and r['version'] == 1
    c['bob'].pull(pid, 'head', dest=db)
    assert c['bob'].sync(Project.load(db), pid)['status'] == 'up to date'
    pa = Project.load(da)
    pa.variants[0].cameras[0].x += 0.05
    pa.save(da)
    assert c['alice'].sync(pa, pid, 'Alice moves F')['status'] == 'published'
    pb = Project.load(db)
    pb.variants[-1].sensors[0].z += 0.1
    pb.thumb_png = b'\x89PNG another picture'
    pb.save(db)
    r = c['bob'].sync(pb, pid, 'Bob raises L1')
    assert r['status'] == 'merged' and r['version'] == 3 and not r['conflicts']
    r['project'].save(db)
    m = Project.load(db)
    assert m.variants[0].cameras[0].x == pa.variants[0].cameras[0].x
    assert m.variants[-1].sensors[0].z == pb.variants[-1].sensors[0].z
    r = c['alice'].sync(Project.load(da), pid)
    assert r['status'] == 'pulled' and r['version'] == 3
    r['project'].save(da)
    assert PD.content_equal(Project.load(da), Project.load(db))['equal']
    # the same value changed two ways: conflicts, nothing published
    pa, pb = Project.load(da), Project.load(db)
    pa.analysis.tol_mm, pb.analysis.tol_mm = 11.0, 22.0
    pa.save(da)
    pb.save(db)
    c['alice'].sync(pa, pid)
    r = c['bob'].sync(pb, pid)
    assert r['status'] == 'conflicts' and r['conflicts'][0]['file'] == 'analysis/base.json'
    assert [v['number'] for v in c['bob'].versions(pid)] == [1, 2, 3, 4]


# ----------------------------------------------------------------------------------------------- pooled fleet twin
def _fleet(project, tag, n_veh, seed, rate_deg_per_day, prefix):
    """Vehicles with dated records of tag measured against project's plan: offsets, a yaw drift, noise."""
    rng = np.random.default_rng(seed)
    plan = {p['tag']: p for p in FL.plan_sensors(project, FL.plan_variant(project))}[tag]['pose']
    vs = []
    for i in range(n_veh):
        off = rng.normal(0, 0.002, 3)
        recs = []
        for k, day in enumerate((0, 30, 60, 90, 120)):
            pose = dict(plan)
            pose['x'] += off[0] + rng.normal(0, 0.0005)
            pose['y'] += off[1] + rng.normal(0, 0.0005)
            pose['z'] += off[2] + rng.normal(0, 0.0005)
            pose['yaw'] += rate_deg_per_day * day + rng.normal(0, 0.01)
            recs.append({'id': f'{prefix}{i}r{k}', 'date': f'2026-0{1 + day // 30}-0{1 + k}', 'kind': 'extrinsics',
                         'poses': {tag: pose}, 'source': f'/home/someone/{prefix}/calib-{k}.json'})
        vs.append({'id': f'{prefix}{i}', 'name': f'V{i + 1}', 'records': recs})
    return {'vehicles': vs}


def _projects():
    a = samples.tigor_sample_project()
    a.active_uid = a.variants[0].uid
    tag = a.variant.cameras[0].tag
    a.fleet = _fleet(a, tag, 3, 1, 0.002, 'a')
    b = samples.tigor_sample_project()
    b.active_uid = b.variants[0].uid
    b.variant.cameras[0].x += 0.10                                            # team B built F 10 cm further forward
    b.variant.cameras[0].yaw += 1.0
    b.fleet = _fleet(b, tag, 2, 2, 0.003, 'b')
    return a, b, tag


ST = FT.FleetTwinSettings(limit_source='analysis', as_of='2026-06-01')


def test_pooled_twin_rebases_and_equals_the_single_project_twin():
    a, b, tag = _projects()
    q, notes = FT.pooled_project(a, [FT.pool_source(a, 'A'), FT.pool_source(b, 'B')], ST)
    assert len(FL.vehicles(q)) == 5 and not notes
    plan = FL.plan_sensors(q, FT.plan_variant(q, ST))
    pooled = FT.fleet_observations(q, plan, FT.plan_variant(q, ST))
    own = FT.fleet_observations(b, FL.plan_sensors(b, FT.plan_variant(b, ST)), FT.plan_variant(b, ST))
    for vid in ('b0', 'b1'):                        # B's deltas against B's plan are the deltas in the pooled twin
        x = np.array([o['d'] for o in own[(vid, tag)]])
        y = np.array([o['d'] for o in pooled[(f'B/{vid}', tag)]])
        assert np.allclose(x, y, atol=1e-9)
    assert all('/home/someone' not in r['source'] for v in FT.pool_source(b, 'B')['vehicles'] for r in v['records'])
    # one source pooled is the project's own twin (the vehicles only renamed)
    r1 = FT.run(a, settings=ST, variant=FT.plan_variant(a, ST))
    r2 = FT.run_pooled(a, [FT.pool_source(a, 'A')], settings=ST)
    assert r2.metrics['pooled'] == [{'label': 'A', 'vehicles': 3, 'records': 15, 'observations': 0}]
    for (vid, t), c1 in r1.cells.items():
        c2 = r2.cells[(f'A/{vid}', t)]
        assert c1['urgency'] == c2['urgency'] and c1['p_now'] == pytest.approx(c2['p_now'], abs=1e-12)
    r3 = FT.run_pooled(a, [FT.pool_source(a, 'A'), FT.pool_source(b, 'B')], settings=ST)
    assert r3.metrics['vehicles'] == 5 and FT.POOL_TEXT in r3.notes


def test_pooled_fleet_through_the_lab_with_access_per_team(lab):
    srv, toks, c = lab
    a, b, tag = _projects()
    pa = c['alice'].register('fleet-a')['id']
    pb = c['carol'].register('fleet-b', share_fleet=False)['id']
    c['alice'].publish(a, pa, 'fleet A', parent=0)
    c['carol'].publish(b, pb, 'fleet B', parent=0)
    doc = c['alice'].pool()
    assert [s['project'] for s in doc['sources']] == ['alpha/fleet-a']            # B is beta's and not readable
    c['carol'].patch(pb, readers=['alpha'])
    doc = c['alice'].pool()
    assert [s['project'] for s in doc['sources']] == ['alpha/fleet-a']
    assert doc['skipped'] == [{'project': 'beta/fleet-b', 'why': 'its team does not share its fleet'}]
    c['carol'].patch(pb, share_fleet=True)
    res, doc = c['alice'].pooled_twin(a, settings=ST)
    assert sorted(s['project'] for s in doc['sources']) == ['alpha/fleet-a', 'beta/fleet-b']
    assert res.metrics['vehicles'] == 5
    names = sorted(v['name'] for v in res.vehicles)
    assert names[0].startswith('alpha/fleet-a@1 · V') and names[-1].startswith('beta/fleet-b@1 · V')
    assert [s['project'] for s in c['carol'].pool()['sources']] == ['beta/fleet-b']   # alpha shares nothing
    assert [s['project'] for s in c['alice'].pool(['beta/fleet-b'])['sources']] == ['beta/fleet-b']


# ----------------------------------------------------------------------------------------------- records
def _record(i):
    rig = {'vehicle': {'name': 'Our test car', 'length_m': 4.0, 'width_m': 1.8, 'height_m': 1.5},
           'cameras': [{'tag': f'C{i}', 'module': '', 'lens': '', 'width': 640, 'height': 480, 'x': 1.0 + 0.01 * i,
                        'y': 0.0, 'z': 1.2, 'yaw': 0.0, 'pitch': 10.0, 'roll': 0.0, 'hfov_deg': 90.0,
                        'vfov_deg': 70.0, 'max_theta_deg': 60.0, 'projection': 'rectilinear'}], 'sensors': []}
    from vantage.core import regions as RG
    g = RG.RegionGrid()
    m = {'pedestrian': {rid: PM.metrics_cell(10, 5 + i % 3, 1) for _, _, rid, _ in g.regions()}}
    return PM.make_record(source='synthetic-test', scene={'kind': 'synthetic', 'town': f'T{i}', 'seed': i},
                          detector={'id': 'analytic', 'backend': 'external:test'}, rig=rig, grid=g, metrics=m)


def test_records_opt_in_pooled_and_anonymised(lab, tmp_path):
    srv, toks, c = lab
    pid = c['alice'].register('perception')['id']
    docs = [_record(i) for i in range(5)]
    assert _status(c['alice'].push_records, pid, docs) == 403                  # not opted in
    c['alice'].patch(pid, share_records=True)
    assert _status(c['rita'].push_records, pid, docs) == 403
    assert _status(c['carol'].push_records, pid, docs) == 403                  # not beta's project
    r = c['alice'].push_records(pid, docs)
    assert r == {'added': 5, 'duplicates': 0, 'rejected': 0, 'errors': []}
    assert c['bob'].push_records(pid, docs)['duplicates'] == 5
    raw = c['alice'].request('POST', 'records', {'project': pid, 'records': docs})        # not anonymised
    assert raw['rejected'] == 5 and 'anonymised' in raw['errors'][0]['errors'][0]
    store = PM.Store(str(tmp_path / 'mine.jsonl'))
    got = c['carol'].pull_records(store)                                       # every team learns from everyone's runs
    assert got['received'] == 5 and got['added'] == 5
    assert all(d['anonymised'] and 'name' not in d['rig']['vehicle'] for d in store.records())
    assert {d['origin_id'] for d in store.records()} == {d['id'] for d in docs}
    assert c['carol'].pull_records(store)['added'] == 0


def test_project_records_picks_this_projects_records(tmp_path, sample):
    p = copy.deepcopy(sample)
    store = PM.Store(str(tmp_path / 's.jsonl'))
    mine = PM.make_record(source='synthetic-test', scene={'kind': 'synthetic', 'town': 'T', 'seed': 1},
                          detector={'id': 'x', 'backend': 'external:t'}, rig=PM.rig_block(p, p.variants[0]),
                          grid=_record(0)['grid'], metrics=_record(0)['metrics'])
    store.add(mine)
    store.add(_record(1))
    assert [d['id'] for d in LC.project_records(p, store)] == [mine['id']]


# ----------------------------------------------------------------------------------------------- privacy
def test_no_passwords_keys_or_tokens_reach_the_server(lab, sample, tmp_path):
    srv, toks, c = lab
    from vantage.assistant import secrets as SEC
    api_key = 'gsk_TESTKEY_0123456789abcdefghij'
    SEC.save(api_key, 'groq')
    p = copy.deepcopy(sample)
    p.streams = {'F': 'rtsp://admin:pw-hunter2@10.1.2.3:554/live', 'B': 'http://u:pw-swordfish@cam.local/mjpg'}
    p.save(str(tmp_path / 'secret.vantage'))
    p = Project.load(str(tmp_path / 'secret.vantage'))
    assert LC.save_token(srv.url, toks['alice']) in ('keyring', 'file')
    cl = LC.LabClient(srv.url)                                                # the token from the private store
    assert cl.token == toks['alice']
    pid = cl.register('secret')['id']
    sent = LC.files_to_publish(p)
    v = cl.publish(p, pid, 'with streams', parent=0)
    assert set(v['manifest']) == set(sent)                                    # exactly what was shown
    everything = b''
    for root, _, names in os.walk(srv.store.root):
        for n in names:
            everything += open(os.path.join(root, n), 'rb').read()
    for needle in ('pw-hunter2', 'pw-swordfish', api_key, toks['alice'], toks['admin']):
        assert needle.encode() not in everything, needle
    assert b'rtsp://admin@10.1.2.3:554/live' in b''.join(sent.values())       # the stream is there, its password not
    LC.forget_token(srv.url)
    assert LC.load_token(srv.url) is None


# ----------------------------------------------------------------------------------------------- command line
def test_command_line_init_tokens_backup(tmp_path, capsys, sample):
    root = str(tmp_path / 'lab')
    assert LS.main(['init', '--root', root, '--team', 'Core Team']) == 0
    first = json.loads(capsys.readouterr().out)
    assert first['team'] == 'core-team' and first['token'].startswith('vlab_')
    assert LS.main(['token', '--root', root, '--team', 'core-team', '--role', 'member', '--name', 'm']) == 0
    tok = json.loads(capsys.readouterr().out)['token']
    st = LabStore(root)
    who = st.authenticate(tok)
    pid = st.register(who, 'p')['id']
    files = LC.files_to_publish(sample)
    for b in files.values():
        import io
        st.put_blob(who, hashlib.sha256(b).hexdigest(), io.BytesIO(b), len(b), 1 << 30)
    st.publish(who, pid, LC.manifest_of(files), 'v1', 0)
    assert LS.main(['backup', '--root', root, str(tmp_path / 'bk')]) == 0
    capsys.readouterr()
    bk = LabStore(str(tmp_path / 'bk'))
    doc = bk.version(bk.authenticate(tok), pid, 'head')
    assert PD.content_equal(PD.project_from_files(bk.version_files(doc)), sample)['equal']
    assert LS.main(['stats', '--root', root]) == 0
    assert json.loads(capsys.readouterr().out)['versions'] == 1
    assert LS.main(['revoke', '--root', root, '2']) == 0
    with pytest.raises(Exception):
        st.authenticate(tok)
    assert LS.main(['token', '--root', str(tmp_path / 'none'), '--team', 'x']) == 2   # no lab there


# ----------------------------------------------------------------------------------------------- the page
from test_gui import app, win, pump, settle, _wait   # noqa: E402,F401  (fixtures)


def _open(win, name, cls):
    if name not in win._page_widgets:
        pg = cls(win.ctx)
        win.pages.addWidget(pg)
        win._page_widgets[name] = pg
    win.set_page(name)
    return win._page_widgets[name]


def test_lab_page_connects_publishes_shows_history_diffs_pulls_and_pools(app, win, lab, tmp_path, monkeypatch):
    from vantage.ui.workspaces.lab_ws import LabPage
    srv, toks, c = lab
    a, b, tag = _projects()
    c['carol'].publish(b, c['carol'].register('fleet-b', share_fleet=True, readers=['alpha'])['id'], 'B', parent=0)
    path = str(tmp_path / 'page.vantage.d')
    a.save(path)
    win.store.set_project(Project.load(path))
    pg = _open(win, 'lab', LabPage)
    pg.url.setText(srv.url)
    pg.token.setText(toks['alice'])
    pg.b_save_token.click()
    assert LC.load_token(srv.url) == toks['alice'] and pg.token.text() == ''
    pg.b_connect.click()
    _wait(app, lambda: pg.who is not None and not pg.busy, 10)
    assert pg.who['team'] == 'alpha' and 'alpha' in pg.status_text
    pg.new_name.setText('page rig')
    pg.b_register.click()
    _wait(app, lambda: pg.pid == 'alpha/page-rig' and not pg.busy, 10)
    assert pg.files_table.rowCount() == len(LC.files_to_publish(win.store.project))   # what will be sent
    pg.message.setText('first from the page')
    pg.b_publish.click()
    _wait(app, lambda: pg.history.rowCount() == 1 and not pg.busy, 10)
    win.store.edit(['variant'], lambda p: setattr(p.variant.cameras[0], 'x', p.variant.cameras[0].x + 0.2),
                   'move F')
    win.store.project.save(path)
    pg.message.setText('moved F')
    pg.b_publish.click()
    _wait(app, lambda: pg.history.rowCount() == 2 and not pg.busy, 10)
    pg.history.selectRow(0)                                                   # newest first: version 2
    _wait(app, lambda: 'moved 200.0 mm' in pg.diff_text.toPlainText() and not pg.busy, 10)
    pg.share_fleet.setChecked(True)
    _wait(app, lambda: not pg.busy and c['alice'].project('alpha/page-rig')['share_fleet'], 10)
    assert pg.b_pool.isEnabled()      # a button is off only while a request runs
    pg.b_pool.click()
    _wait(app, lambda: pg.pool_result is not None and not pg.busy, 60)
    assert pg.pool_result.metrics['vehicles'] == 5 and pg.pool_table.rowCount() >= 2
    monkeypatch.setattr(win, '_maybe_save', lambda: True, raising=False)
    pg.history.selectRow(1)                                                   # version 1
    pg.b_open.click()
    _wait(app, lambda: pg.busy is False and win.store.project.lab_version['number'] == 1
                 if hasattr(win.store.project, 'lab_version') else False, 10)
    assert win.store.project.variant.cameras[0].x == pytest.approx(a.variant.cameras[0].x)
