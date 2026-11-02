"""Claude over MCP (vantage.mcp): the stdio protocol, edits as operations, a project file without the window, and the
open window through its bridge (vantage.ui.mcp_bridge)."""
import json
import os
import subprocess
import sys
import threading
import time

import pytest

from vantage.core import samples
from vantage.mcp import edits as E
from vantage.mcp import service as SV
from vantage.mcp import session as SE
from vantage.mcp.server import Router, Server

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope='module')
def sample():
    return samples.tigor_sample_project()


def _proj(sample):
    return sample.snapshot()


def _call(router, name, **args):
    r = router.call(name, args)
    doc = json.loads(r['content'][0]['text'])
    return doc, r['isError']


# ======================================================================= protocol
def test_initialize_lists_tools_and_answers_errors_as_json_rpc():
    srv = Server(Router(mode='file'), out=None)
    r = srv.handle({'jsonrpc': '2.0', 'id': 1, 'method': 'initialize',
                    'params': {'protocolVersion': '2025-06-18', 'capabilities': {}}})
    assert r['result']['protocolVersion'] == '2025-06-18'
    assert r['result']['serverInfo']['name'] == 'vantage' and 'ISO 8855' in r['result']['instructions']
    r = srv.handle({'jsonrpc': '2.0', 'id': 2, 'method': 'initialize', 'params': {'protocolVersion': '1999-01-01'}})
    assert r['result']['protocolVersion'] == '2025-11-25'                  # ours, for a version we do not know
    tools = srv.handle({'jsonrpc': '2.0', 'id': 3, 'method': 'tools/list'})['result']['tools']
    names = {t['name'] for t in tools}
    assert {'edit_project', 'undo_last_change', 'get_overview', 'simulate_changes', 'look', 'propose_plan'} <= names
    assert not names & set(SV.SKIP)
    edit = next(t for t in tools if t['name'] == 'edit_project')
    assert edit['annotations']['readOnlyHint'] is False and edit['inputSchema']['required'] == ['ops']
    assert next(t for t in tools if t['name'] == 'get_coverage')['annotations']['readOnlyHint'] is True
    assert srv.handle({'jsonrpc': '2.0', 'method': 'notifications/initialized'}) is None
    assert srv.handle({'jsonrpc': '2.0', 'id': 4, 'method': 'nope'})['error']['code'] == -32601


def test_stdio_server_in_a_process_edits_a_project_file(tmp_path, sample):
    path = str(tmp_path / 'tigor.vantage')
    sample.save(path)
    msgs = [{'jsonrpc': '2.0', 'id': 1, 'method': 'initialize', 'params': {'protocolVersion': '2025-06-18'}},
            {'jsonrpc': '2.0', 'method': 'notifications/initialized'},
            {'jsonrpc': '2.0', 'id': 2, 'method': 'tools/call',
             'params': {'name': 'edit_project', 'arguments': {'ops': [
                 {'op': 'set', 'target': 'camera', 'id': 'F', 'props': {'pitch': 35}}]}}}]
    env = dict(os.environ, XDG_CONFIG_HOME=str(tmp_path / 'cfg'))
    p = subprocess.run([sys.executable, os.path.join(ROOT, 'run_vantage.py'), '--mcp', '--mcp-project', path],
                       input='\n'.join(json.dumps(m) for m in msgs) + '\n', capture_output=True, text=True,
                       env=env, timeout=600)
    lines = [json.loads(x) for x in p.stdout.splitlines()]
    assert [d['id'] for d in lines] == [1, 2]                                 # stdout holds the protocol only
    doc = json.loads(lines[1]['result']['content'][0]['text'])
    assert doc['changed'] == ['camera F: pitch 40 → 35'], doc
    from vantage.core.project import Project
    assert Project.load(path).variant.by_tag('F').pitch == 35
    assert Project.load(path + '.bak').variant.by_tag('F').pitch == 40     # the file as it was


# ======================================================================= edits
def test_set_mirrors_to_the_partner_and_checks_every_value(sample):
    p = _proj(sample)
    keys, lines = E.apply(p, [{'op': 'set', 'target': 'camera', 'id': 'ML', 'props': {'pitch': 38, 'yaw': 70}}])
    assert keys == [f'variant:{p.active_uid}'] and 'mirrored to MR' in lines[0]
    assert (p.variant.by_tag('MR').pitch, p.variant.by_tag('MR').yaw) == (38, -70)
    for op, why in [({'op': 'set', 'target': 'camera', 'id': 'F', 'props': {'pitch': 'down'}}, 'a number'),
                    ({'op': 'set', 'target': 'camera', 'id': 'F', 'props': {'pitch': 120}}, 'between'),
                    ({'op': 'set', 'target': 'camera', 'id': 'F', 'props': {'lens_id': 'nope'}}, 'not in the library'),
                    ({'op': 'set', 'target': 'camera', 'id': 'F', 'props': {'calib': {}}}, 'cannot set calib'),
                    ({'op': 'set', 'target': 'camera', 'id': 'F', 'props': {'tag': 'R'}}, 'taken'),
                    ({'op': 'set', 'target': 'camera', 'id': 'Q9', 'props': {'pitch': 1}}, 'no camera'),
                    ({'op': 'set', 'target': 'analysis', 'props': {'copilot': {}}}, 'cannot set copilot'),
                    ({'op': 'set', 'target': 'analysis', 'props': {'automation': {}}}, 'cannot set automation'),
                    ({'op': 'set', 'target': 'roi', 'props': {'mode': 'polygon', 'polygon': [[0, 0]]}}, 'at least 3'),
                    ({'op': 'remove', 'target': 'camera'}, 'id'),
                    ({'op': 'fly', 'target': 'camera'}, 'one of')]:
        with pytest.raises(E.EditError, match=why):
            E.apply(_proj(sample), [op])


def test_locked_cameras_stay_the_users(sample):
    p = _proj(sample)
    p.variant.by_tag('F').locked = True
    for op in ({'op': 'set', 'target': 'camera', 'id': 'F', 'props': {'pitch': 30}},
               {'op': 'remove', 'target': 'camera', 'id': 'F'}):
        with pytest.raises(E.EditError, match='locked'):
            E.apply(p, [op])
    with pytest.raises(E.EditError, match='only the user unlocks'):
        E.apply(_proj(sample), [{'op': 'set', 'target': 'camera', 'id': 'R', 'props': {'locked': False}}])
    E.apply(p, [{'op': 'set', 'target': 'camera', 'id': 'R', 'props': {'locked': True}}])     # locking is fine
    assert p.variant.by_tag('R').locked


def test_add_remove_link_and_variants(sample):
    p = _proj(sample)
    n_var = len(p.variants)
    keys, lines = E.apply(p, [
        {'op': 'add', 'target': 'variant', 'props': {'name': 'Claude try'}},
        {'op': 'add', 'target': 'sensor', 'props': {'kind': 'radar'}},
        {'op': 'add', 'target': 'camera', 'props': {'tag': 'X1', 'x': 1.0, 'y': 0.8, 'z': 1.0, 'yaw': 90}},
        {'op': 'link', 'target': 'camera', 'id': 'X1', 'partner': 'F'},
        {'op': 'remove', 'target': 'camera', 'id': 'ML'},
        {'op': 'add', 'target': 'rig_part', 'preset': 'roof_rack'},
        {'op': 'add', 'target': 'obstacle', 'props': {'x': 4, 'y': 0, 'height': 0.5}},
        {'op': 'set', 'target': 'project', 'props': {'brief': 'Cover 2 m all round'}},
        {'op': 'set', 'target': 'analysis', 'props': {'perception': {'object': 'car'}}},
        {'op': 'add', 'target': 'log', 'props': {'kind': 'decision', 'text': 'Mirrors at 40° pitch'}}])
    assert set(keys) == {'variants', 'rig', 'obstacles', 'analysis'}, keys
    v = p.variant
    assert len(p.variants) == n_var + 1 and v.name == 'Claude try'
    orig = sample.variant
    assert {c.uid for c in v.cameras}.isdisjoint({c.uid for c in orig.cameras})       # a copy with its own uids
    assert v.by_tag('ML') is None and v.by_tag('MR').link_uid == ''                     # its partner let go
    assert not any(orig.by_tag('ML').uid in k for k in v.seams)                   # nor any seam of it
    assert v.by_tag('X1').link_uid == v.by_tag('F').uid and v.by_tag('X1').port > 0
    assert [s.kind for s in v.sensors][-1] == 'radar'
    assert p.analysis.assistant['brief'] == 'Cover 2 m all round'
    assert p.analysis.perception['object'] == 'car'
    assert p.analysis.assistant_log[-1]['text'] == 'Mirrors at 40° pitch'
    assert any(r.name.lower().startswith('roof rack') for r in p.rig)
    _, lines = E.apply(p, [{'op': 'activate', 'target': 'variant', 'id': '1'},
                           {'op': 'remove', 'target': 'variant', 'id': 'Claude try'}])
    assert len(p.variants) == n_var and p.variant.uid == sample.variants[0].uid


def test_project_data_lists_what_can_be_set(sample):
    d = E.data(sample, ['cameras', 'library', 'analysis', 'project'])
    cam = d['cameras'][0]
    assert set(E.CAMERA_FIELDS) <= set(cam) and 'mirrored_with' in cam
    assert {'h05r2', 'h03r2'} <= {l['id'] for l in d['library']['lenses']}
    assert 'copilot' not in d['analysis'] and 'automation' not in d['analysis']['study_settings']
    assert 'error' in E.data(sample, ['nope'])


# ======================================================================= stitch seams
@pytest.fixture(scope='module')
def analysed(sample):
    from vantage.core.analysis import Engine
    e = Engine()
    return e, e.analyse(sample)


def test_a_seam_is_set_by_hand_or_handed_back_to_the_automatic_placement(sample, analysed):
    _, res = analysed
    p = _proj(sample)
    tags = {c.uid: c.tag for c in p.variant.cameras}
    s0, s1 = res.seams[0], res.seams[1]
    name = f'{tags[s0.a]}|{tags[s0.b]}'
    key = f'{s0.a}|{s0.b}'
    assert s0.auto and key not in p.variant.seams
    keys, lines = E.apply(p, [{'op': 'set', 'target': 'seam', 'id': name, 'props': {'angle_deg': s0.angle - 4.04}}],
                          res.seams)
    assert keys == [f'variant:{p.variant.uid}'] and p.variant.seams[key] == round(s0.angle - 4.04, 1)
    assert lines == [f'seam {tags[s0.a]} | {tags[s0.b]}: {s0.angle:.1f}° automatic → {s0.angle - 4.04:.1f}° by hand']
    # either order of the tags, in the same list as another seam; then back to automatic
    rev = f'{tags[s0.b]} | {tags[s0.a]}'
    pin = {'op': 'set', 'target': 'seam', 'id': f'{tags[s1.a]}|{tags[s1.b]}', 'props': {'auto': False}}
    _, lines = E.apply(p, [{'op': 'set', 'target': 'seam', 'id': rev, 'props': {'auto': True}}, pin], res.seams)
    assert key not in p.variant.seams and lines[0].endswith('→ automatic')
    assert p.variant.seams[f'{s1.a}|{s1.b}'] == round(s1.angle, 1)             # auto: false pins it where it is
    assert E.apply(p, [{'op': 'set', 'target': 'seam', 'id': name, 'props': {'auto': True}}], res.seams)[1] == \
        [f'seam {tags[s0.a]} | {tags[s0.b]}: no change']
    other = res.seams[2]
    for op, why in (({'props': {'angle_deg': s0.angle}, 'id': f'{tags[s0.a]}|{tags[other.b]}'}, 'no seam'),
                    ({'props': {'angle_deg': 'left'}}, 'a number'),
                    ({'props': {'angle_deg': float('nan')}}, 'a number'),
                    ({'props': {'angle_deg': 1.0, 'auto': True}}, 'not both'),
                    ({'props': {'auto': 'yes'}}, 'true or false'),
                    ({'props': {'blend_m': 0.2}}, 'cannot set blend_m'),
                    ({'props': {}}, 'angle_deg'),
                    ({'props': {'angle_deg': s1.angle + 2}}, 'not between the seams either side'),
                    ({'op': 'remove'}, 'is set'),
                    ({'props': {'angle_deg': 1.0}, 'variant': 'Front-heavy'}, 'active variant')):
        q = _proj(sample)
        o = {'op': 'set', 'target': 'seam', 'id': name, **op}
        if o.get('variant'):
            E.apply(q, [{'op': 'add', 'target': 'variant', 'props': {'name': 'Front-heavy'}, 'activate': False}])
        with pytest.raises(E.EditError, match=why):
            E.apply(q, [o], res.seams)
    with pytest.raises(E.EditError, match='no seams'):
        E.apply(_proj(sample), [{'op': 'set', 'target': 'seam', 'id': name, 'props': {'auto': True}}])
    with pytest.raises(E.EditError, match='target.: "seam"'):           # the shape that was refused before
        E.apply(_proj(sample), [{'op': 'set', 'target': 'variant', 'props': {'seams': [{'between': ['QR', 'MR']}]}}])


def test_edit_project_sets_a_seam_with_the_stitched_view_before_and_after(tmp_path, sample, monkeypatch):
    from vantage.core.project import Project
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'cfg'))
    path = str(tmp_path / 'p.vantage')
    sample.save(path)
    r = Router(path, 'file')
    seams, err = _call(r, 'get_seams', advise=False)
    assert not err and 'seam_advice' not in seams
    row = seams['seams'][0]
    name = '|'.join(row['between'])
    assert row['placed'] == 'automatic' and row['object_vanishes_m2'] is not None
    assert seams['object'] == '0.9 m child'
    assert seams['object_vanishes_m2'] == pytest.approx(sum(x['object_vanishes_m2'] for x in seams['seams']), abs=0.01)
    ang = row['angle_deg'] - 9
    doc, err = _call(r, 'edit_project', ops=[{'op': 'set', 'target': 'seam', 'id': name, 'props': {'angle_deg': ang}}])
    assert not err, doc
    sv = doc['stitched_view']
    first = sv['seams'][0]
    assert first['between'] == row['between'] and first['angle_deg'] == [row['angle_deg'], round(ang, 1)]
    assert first['placed'] == ['automatic', 'by hand']
    assert first['object_vanishes_m2'][0] == row['object_vanishes_m2']
    assert sv['object_vanishes_m2'][0] == seams['object_vanishes_m2']
    after, _ = _call(r, 'get_seams', advise=False)                       # what the edit reported is what is there now
    assert after['seams'][0]['angle_deg'] == round(ang, 1) and after['seams'][0]['placed'] == 'by hand'
    assert sv['object_vanishes_m2'][1] == after['object_vanishes_m2']
    assert first['object_vanishes_m2'][1] == after['seams'][0]['object_vanishes_m2']
    p = Project.load(path)
    assert round(ang, 1) in p.variant.seams.values()                           # saved, one undo step
    doc, _ = _call(r, 'undo_last_change')
    assert doc['result'].startswith('undid: Claude · seam') and not Project.load(path).variant.seams
    _call(r, 'redo_change')
    doc, err = _call(r, 'edit_project', ops=[{'op': 'set', 'target': 'seam', 'id': name, 'props': {'auto': True}}])
    assert not err and doc['stitched_view']['seams'][0]['placed'] == ['by hand', 'automatic']
    assert not Project.load(path).variant.seams
    doc, err = _call(r, 'edit_project', ops=[{'op': 'set', 'target': 'camera', 'id': 'F', 'props': {'pitch': 1}},
                                             {'op': 'set', 'target': 'variant', 'props': {'seams': {}}}])
    assert err and 'target": "seam"' in doc['error'] and doc['changed'] == 'nothing'
    doc, err = _call(r, 'edit_project', ops=[{'op': 'set', 'target': 'camera', 'id': 'F', 'props': {'pitch': 38}}])
    assert not err and 'stitched_view' in doc                                  # a camera moves the seams too


def test_simulate_changes_compares_seam_angles_by_where_the_object_vanishes(tmp_path, sample, analysed, monkeypatch):
    from vantage.core import stitched_objects as so
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'cfg'))
    r = Router(None, 'file')
    _call(r, 'open_project', sample=True)
    seams, _ = _call(r, 'get_seams', advise=False)
    row = seams['seams'][0]
    out = {}
    for d in (-8, 0, 8):
        doc, err = _call(r, 'simulate_changes', seams=[{'between': row['between'], 'angle_deg': row['angle_deg'] + d}])
        assert not err, doc
        out[d] = doc
    sv = out[-8]['stitched_view']
    assert sv['object'] == '0.9 m child' and len(sv['seams']) == len(seams['seams'])
    assert [s['object_vanishes_m2'][0] for s in sv['seams']] == [s['object_vanishes_m2'] for s in seams['seams']]
    assert out[-8]['effect']['object_vanishes_m2'] == {'before': seams['object_vanishes_m2'],
                                                       'after': sv['object_vanishes_m2'][1]}
    # the same computation as the stitched-objects analysis of the changed design
    e, res = analysed
    q = _proj(sample)
    s0 = res.seams[0]
    q.variant.seams[f'{s0.a}|{s0.b}'] = round(row['angle_deg'] + 8, 1)
    e.vehicle_geom(q)
    r1 = e.analyse(q)
    sres = so.analyse(q, e, r1)
    got = out[8]['stitched_view']
    assert got['object_vanishes_m2'][1] == round(sres.metrics['vanish_m2'], 3)
    assert [s['object_vanishes_m2'][1] for s in got['seams']] == [round(x['vanish_m2'], 3) for x in sres.seams]
    assert got['seams'][0]['density_px_cm'][1] == round(r1.seams[0].density, 2)
    # the angles differ where it matters: the vanish area at the moved seam, its density
    firsts = {d: o['stitched_view']['seams'][0] for d, o in out.items()}
    assert len({f['object_vanishes_m2'][1] for f in firsts.values()} | {f['density_px_cm'][1] for f in firsts.values()}
               ) > 2
    assert firsts[0]['placed'] == ['automatic', 'by hand'] and firsts[0]['angle_deg'][1] == row['angle_deg']
    assert out[0]['stitched_view']['issue']['change'] in ('unchanged', 'changed', 'none')
    # back to automatic is a change only for a seam placed by hand
    doc, err = _call(r, 'simulate_changes', seams=[{'between': row['between'], 'auto': True}])
    assert err and 'automatically already' in doc['error']
    _call(r, 'edit_project', ops=[{'op': 'set', 'target': 'seam', 'id': '|'.join(row['between']),
                                   'props': {'angle_deg': row['angle_deg'] + 8}}])
    doc, err = _call(r, 'simulate_changes', seams=[{'between': row['between'], 'auto': True}])
    assert not err and doc['changes'][0]['to'] == 'automatic'
    assert doc['stitched_view']['seams'][0]['placed'] == ['by hand', 'automatic']
    assert doc['stitched_view']['seams'][0]['angle_deg'][1] == row['angle_deg']
    # a camera change moves the seams: reported too
    doc, err = _call(r, 'simulate_changes', changes=[{'camera': 'F', 'dyaw': 6}])
    assert not err and len(doc['stitched_view']['seams']) == len(seams['seams'])


def test_get_seams_advises_angles_that_edit_project_applies(tmp_path, sample, analysed, monkeypatch):
    from vantage.core import stitched_objects as so
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'cfg'))
    r = Router(None, 'file')
    _call(r, 'open_project', sample=True)
    doc, err = _call(r, 'get_seams')
    assert not err
    e, res = analysed
    adv = so.advise(_proj(sample), e, res)                                    # the Stitch seams page's Suggest seams
    assert adv.changes
    tags = {c.uid: c.tag for c in sample.variant.cameras}
    for row, x in zip(doc['seams'], adv.seams):
        assert row['between'] == [tags[x.a], tags[x.b]]
        assert row['advised_angle_deg'] == round(x.best, 1) and row['advice'] == ('move' if x.moved else 'keep')
        assert [row['angle_deg'], round(x.vanish_before, 3)] in [t[:2] for t in row['tried']]   # where it is now
        assert all(len(t) == 3 for t in row['tried']) and len(row['tried']) > 5
    sa = doc['seam_advice']
    assert sa['vanish_m2_now'] == round(adv.vanish_now, 3) and sa['vanish_m2_after'] == round(adv.vanish_after, 3)
    assert sa['vanish_m2_after'] < sa['vanish_m2_now']
    assert {o['id']: o['props']['angle_deg'] for o in sa['edit_project_ops']} == \
        {f'{tags[k.split("|")[0]]}|{tags[k.split("|")[1]]}': v for k, v in adv.changes.items()}
    assert len(sa['propose_changes_seams']) == len(sa['edit_project_ops'])
    ed, err = _call(r, 'edit_project', ops=sa['edit_project_ops'])          # Apply: one undo step
    assert not err, ed
    assert len(ed['changed']) == len(adv.changes) and all('by hand' in x for x in ed['changed'])
    assert ed['stitched_view']['object_vanishes_m2'][1] < ed['stitched_view']['object_vanishes_m2'][0]
    doc, _ = _call(r, 'undo_last_change')
    assert doc['result'].startswith('undid')
    names = {t['name']: t for t in SV.listing()}
    desc = names['get_seams']['description']
    assert 'advised angle' in desc and '"target": "seam"' in desc
    assert '"auto": true' in names['edit_project']['description'] and 'seam' in \
        names['edit_project']['inputSchema']['properties']['ops']['items']['properties']['target']['enum']
    assert 'object_vanishes_m2' in names['simulate_changes']['description']


# ======================================================================= a file
def test_file_session_saves_each_change_keeps_a_backup_and_undoes(tmp_path, sample, monkeypatch):
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'cfg'))
    path = str(tmp_path / 'p.vantage')
    sample.save(path)
    r = Router(path, 'file')
    doc, err = _call(r, 'edit_project', ops=[{'op': 'set', 'target': 'camera', 'id': 'R', 'props': {'pitch': 30}}])
    assert not err and doc['before']['blind_m2'] is not None and 'after' in doc
    from vantage.core.project import Project
    assert Project.load(path).variant.by_tag('R').pitch == 30 and os.path.exists(path + '.bak')
    doc, err = _call(r, 'edit_project', ops=[{'op': 'set', 'target': 'camera', 'id': 'R', 'props': {'pitch': 400}},
                                             {'op': 'set', 'target': 'camera', 'id': 'F', 'props': {'pitch': 1}}])
    assert err and doc['changed'] == 'nothing'
    assert Project.load(path).variant.by_tag('F').pitch == 40                 # the whole list or nothing
    doc, _ = _call(r, 'get_recent_activity')
    assert doc['steps_this_session'][-1]['by'] == 'claude'
    doc, _ = _call(r, 'undo_last_change')
    assert doc['result'].startswith('undid') and Project.load(path).variant.by_tag('R').pitch == 40
    os.utime(path, (time.time() + 5, time.time() + 5))                     # someone else saved it meanwhile
    doc, err = _call(r, 'edit_project', ops=[{'op': 'set', 'target': 'camera', 'id': 'R', 'props': {'pitch': 31}}])
    assert err and 'changed on disk' in doc['error']
    doc, err = _call(r, 'show_in_app', action='frame_all')
    assert err and 'Vantage window' in doc['error']                         # the window's tools say what they need


def test_a_file_open_in_a_window_is_left_to_the_window(tmp_path, sample, monkeypatch):
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'cfg'))
    path = str(tmp_path / 'p.vantage')
    sample.save(path)
    os.makedirs(SE.bridge_dir())
    with open(os.path.join(SE.bridge_dir(), f'{os.getpid()}.json'), 'w') as f:
        json.dump({'pid': os.getpid(), 'port': 1, 'token': 't', 'path': path, 'started': 1}, f)
    assert 'open in a Vantage window' in SE.FileSession().open(path=path)['error']
    with open(os.path.join(SE.bridge_dir(), '999999999.json'), 'w') as f:      # a window that is gone
        json.dump({'pid': 999999999, 'port': 1, 'token': 't', 'path': '', 'started': 2}, f)
    assert [w['pid'] for w in SE.windows()] == [os.getpid()]


# ======================================================================= batch jobs
def test_run_batch_runs_jobs_on_the_project_as_it_is(tmp_path, sample, monkeypatch):
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'cfg'))
    path = str(tmp_path / 'p.vantage')
    sample.save(path)
    r = Router(path, 'file')
    _call(r, 'edit_project', ops=[{'op': 'set', 'target': 'camera', 'id': 'F', 'props': {'pitch': 33}}])
    doc, err = _call(r, 'run_batch', args=['--analyse', 'a.json', '--sync-plan', 'sync.json'])
    assert not err and doc['exit_code'] in (0, 2), doc
    assert {'a.json', 'sync.json'} <= {f['file'] for f in doc['files']}
    pose = next(c['pose'] for c in doc['json']['a.json']['cameras'] if c['tag'] == 'F')
    assert pose['pitch'] == 33                                                     # the project as it is now
    assert os.path.dirname(doc['folder']) == str(tmp_path)                         # beside the project
    from vantage.core.project import Project
    assert Project.load(path).variant.by_tag('F').pitch == 33                      # (the edit, saved by the edit)
    doc, err = _call(r, 'run_batch', help=True)
    assert '--perception-range' in doc['options'] and '--carla-run' not in doc['options']
    for args, why in ((['--carla-run', 'x'], 'propose_plan'), (['--watch'], 'automation'),
                      (['--sample', '--analyse', 'a.json'], 'this project'), (['--bogus'], 'not a batch option'),
                      (['a.json'], 'no job')):
        doc, err = _call(r, 'run_batch', args=args)
        assert err and why in doc['error'], (args, doc)


# ======================================================================= the window
def test_claude_works_in_the_open_window_as_undo_steps(app_win):
    app, w = app_win
    from vantage.ui.mcp_bridge import enabled
    assert enabled(w.ctx.settings) and w.ctx.mcp.running
    info = json.load(open(w.ctx.mcp.file))
    assert info['pid'] == os.getpid() and len(info['token']) >= 32
    if os.name == 'posix':
        assert os.stat(w.ctx.mcp.file).st_mode & 0o077 == 0                   # the user's alone
    router = Router(mode='live')
    out = {}

    def talk():
        out['status'] = _call(router, 'vantage_status')
        out['edit'] = _call(router, 'edit_project', ops=[
            {'op': 'set', 'target': 'camera', 'id': 'QL', 'props': {'pitch': 44}}])
        out['bad'] = _call(router, 'edit_project', ops=[{'op': 'remove', 'target': 'camera', 'id': 'NOPE'}])
        out['coverage'] = _call(router, 'get_coverage')
        out['propose'] = _call(router, 'propose_changes', changes=[{'camera': 'F', 'dpitch': -2}], title='Tip F up',
                               rationale='test')
        out['activity'] = _call(router, 'get_recent_activity')
        out['batch'] = _call(router, 'run_batch', args=['--analyse', 'live.json'])
        out['undo'] = _call(router, 'undo_last_change')
        bad = SE.LiveLink(dict(info, token='wrong'))
        try:
            bad.request('call', name='vantage_status', args={})
        except SE.LinkError as e:
            out['refused'] = str(e)

    t = threading.Thread(target=talk)
    t.start()
    end = time.time() + 300
    while t.is_alive() and time.time() < end:
        _pump(app, 0.05)
    assert not t.is_alive()
    st, err = out['status']
    assert not err and st['mode'] == 'live' and 'QL' in st['cameras']
    ed, err = out['edit']
    assert not err and ed['changed'][0].startswith('camera QL: pitch 50 → 44'), ed
    assert ed['after']['blind_m2'] is not None
    assert out['bad'][1] and out['bad'][0]['changed'] == 'nothing'
    assert not out['coverage'][1]
    assert not out['propose'][1] and w.ctx.assistant.proposals                 # a card for the user to apply
    steps = out['activity'][0]['steps']
    assert steps[-1]['by'] == 'claude' and steps[-1]['text'].startswith('Claude · camera QL')
    bd, err = out['batch']
    assert not err and bd['exit_code'] in (0, 2) and 'live.json' in bd['json'], bd
    assert 'critical' in bd['meaning'] or bd['exit_code'] == 0
    assert next(c['pose']['pitch'] for c in bd['json']['live.json']['cameras'] if c['tag'] == 'QL') == 44   # unsaved edit
    assert out['undo'][0]['result'].startswith('undid')
    assert w.store.project.variant.by_tag('QL').pitch == 50 and w.store.project.variant.by_tag('QR').pitch == 50
    assert 'wrong token' in out['refused']
    # a step of the user's is never undone from Claude's side
    w.store.set_cam(w.store.project.variant.by_tag('F').uid, pitch=39)
    t = threading.Thread(target=lambda: out.__setitem__('undo2', _call(router, 'undo_last_change')))
    t.start()
    while t.is_alive():
        _pump(app, 0.05)
    assert "the user's" in out['undo2'][0]['result'] and w.store.project.variant.by_tag('F').pitch == 39
    # a seam, set in the window as the Stitch seams page sets it (one undo step of Claude's)
    res = w.ctx.analysis.result
    s0 = res.seams[0]
    tags = {c.uid: c.tag for c in w.store.project.variant.cameras}

    def seam():
        out['seam'] = _call(router, 'edit_project', ops=[{'op': 'set', 'target': 'seam',
                                                          'id': f'{tags[s0.a]}|{tags[s0.b]}',
                                                          'props': {'angle_deg': s0.angle - 5}}])
    t = threading.Thread(target=seam)
    t.start()
    while t.is_alive():
        _pump(app, 0.05)
    sd, err = out['seam']
    assert not err and sd['stitched_view']['seams'][0]['placed'] == ['automatic', 'by hand'], sd
    assert w.store.project.variant.seams == {f'{s0.a}|{s0.b}': round(s0.angle - 5, 1)}
    assert w.store.undo.command(w.store.undo.index() - 1).text().startswith('Claude · seam')
    w.store.undo.undo()
    assert not w.store.project.variant.seams
    w.ctx.mcp.stop()
    assert not os.path.exists(info and os.path.join(SE.bridge_dir(), f'{os.getpid()}.json'))


def _pump(app, s):
    from PyQt6.QtCore import QEventLoop
    t = time.time()
    while time.time() - t < s:
        app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 20)
        time.sleep(0.005)


@pytest.fixture()
def app_win():
    from PyQt6.QtWidgets import QApplication
    from vantage.ui.mainwindow import MainWindow
    from vantage.ui.viewport3d import configure_default_format
    configure_default_format()
    app = QApplication.instance() or QApplication([])
    w = MainWindow(samples.tigor_sample_project(), soft_3d=True)
    w.resize(1400, 900)
    w.show()
    end = time.time() + 90
    while (w.ctx.analysis.result is None or not w.ctx.mcp.running) and time.time() < end:
        _pump(app, 0.05)
    yield app, w
    while w.ctx.analysis._busy or w.ctx.analysis.timer.isActive():
        _pump(app, 0.05)
    w.ctx.mcp.stop()
    w.store.undo.setClean()
    w.close()
    w.deleteLater()
    _pump(app, 0.2)
