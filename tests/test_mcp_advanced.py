"""Claude over MCP, the protocol beyond tools (resources and templates with subscriptions, prompts, completion,
logging, progress, cancellation, structured content), background jobs, the headless vehicle import, layout import,
the aim operation, and the window's view, render and turntable tools."""
import io
import json
import math
import os
import queue
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

from vantage.core import samples
from vantage.mcp import catalog as CT
from vantage.mcp import edits as E
from vantage.mcp import service as SV
from vantage.mcp.server import Router, Server

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope='module')
def sample():
    return samples.tigor_sample_project()


def car_obj(path, scale=1000.0, yup=True):
    """A car-sized model: a 4.2 × 1.8 × 1.2 m body on four 0.3 m wheels named wheel_*, wheelbase 2.6 m with the rear
    axle at x = 0; written in millimetres, with the file's y up and -z forward when yup (as many exporters write)."""
    V = []

    def box(x0, x1, y0, y1, z0, z1):
        b = len(V)
        V.extend((x, y, z) for x in (x0, x1) for y in (y0, y1) for z in (z0, z1))
        quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
        return [(b + q[0], b + q[1], b + q[2]) for q in quads] + [(b + q[0], b + q[2], b + q[3]) for q in quads]

    def cyl(cx, cz, y0, y1, r, n=24):
        b = len(V)
        for i in range(n):
            a = 2 * math.pi * i / n
            V.append((cx + r * math.cos(a), y0, cz + r * math.sin(a)))
            V.append((cx + r * math.cos(a), y1, cz + r * math.sin(a)))
        f = []
        for i in range(n):
            j = (i + 1) % n
            f += [(b + 2 * i, b + 2 * j, b + 2 * j + 1), (b + 2 * i, b + 2 * j + 1, b + 2 * i + 1)]
        V.append((cx, y0, cz))
        V.append((cx, y1, cz))
        c0, c1 = len(V) - 2, len(V) - 1
        for i in range(n):
            j = (i + 1) % n
            f += [(c0, b + 2 * j, b + 2 * i), (c1, b + 2 * i + 1, b + 2 * j + 1)]
        return f
    groups = [('body', box(-0.8, 3.4, -0.9, 0.9, 0.25, 1.45))]
    for nm, x, y0, y1 in (('wheel_RL', 0.0, 0.7, 0.92), ('wheel_RR', 0.0, -0.92, -0.7), ('wheel_FL', 2.6, 0.7, 0.92),
                          ('wheel_FR', 2.6, -0.92, -0.7)):
        groups.append((nm, cyl(x, 0.3, y0, y1, 0.3)))
    with open(path, 'w') as fh:
        for x, y, z in V:
            p = (-y, z, -x) if yup else (x, y, z)
            fh.write('v ' + ' '.join(f'{c * scale:.3f}' for c in p) + '\n')
        for nm, fs in groups:
            fh.write(f'o {nm}\n')
            for a, b, c in fs:
                fh.write(f'f {a + 1} {b + 1} {c + 1}\n')


class _Out(io.StringIO):
    """The server's stdout in-process: lines as they are written, safe across threads."""

    def __init__(self):
        super().__init__()
        self.q = queue.Queue()

    def write(self, s):
        for line in s.splitlines():
            if line.strip():
                self.q.put(json.loads(line))
        return len(s)

    def flush(self):
        pass


def _answer(out, mid, timeout=120):
    end, other = time.time() + timeout, []
    try:
        while time.time() < end:
            d = out.q.get(timeout=max(0.01, end - time.time()))
            if d.get('id') == mid and 'method' not in d:
                return d
            other.append(d)
    finally:
        for d in other:
            out.q.put(d)
    raise TimeoutError(mid)


def _ask(srv, out, mid, method, params=None):
    r = srv.handle({'jsonrpc': '2.0', 'id': mid, 'method': method, 'params': params or {}})
    return r if r is not None else _answer(out, mid)


# ======================================================================= protocol, in process
def test_initialize_offers_resources_prompts_completion_and_logging():
    out = _Out()
    srv = Server(Router(mode='file'), out=out)
    caps = srv.handle({'jsonrpc': '2.0', 'id': 1, 'method': 'initialize',
                       'params': {'protocolVersion': '2025-11-25', 'capabilities': {}}})['result']['capabilities']
    assert caps['resources'] == {'subscribe': True, 'listChanged': False}
    assert 'prompts' in caps and 'completions' in caps and 'logging' in caps
    tools = {t['name']: t for t in srv.handle({'jsonrpc': '2.0', 'id': 2, 'method': 'tools/list'})['result']['tools']}
    assert {'import_vehicle', 'import_layout', 'set_view', 'render_view', 'turntable_video', 'start_job', 'get_job',
            'cancel_job', 'list_jobs'} <= set(tools)
    assert tools['get_coverage']['annotations']['idempotentHint'] is True
    assert tools['import_vehicle']['annotations']['destructiveHint'] is True
    assert 'aim' in tools['edit_project']['inputSchema']['properties']['ops']['items']['properties']['op']['enum']
    res = srv.handle({'jsonrpc': '2.0', 'id': 3, 'method': 'resources/list'})['result']['resources']
    assert {'vantage://status', 'vantage://overview', 'vantage://project-data', 'vantage://view/top.png'} <= \
        {r['uri'] for r in res}
    tpl = srv.handle({'jsonrpc': '2.0', 'id': 4, 'method': 'resources/templates/list'})['result']['resourceTemplates']
    assert {t['uriTemplate'] for t in tpl} >= {'vantage://camera/{tag}', 'vantage://study/{name}',
                                                'vantage://job/{job_id}'}
    prompts = srv.handle({'jsonrpc': '2.0', 'id': 5, 'method': 'prompts/list'})['result']['prompts']
    assert len(prompts) >= 8 and all(p['description'] for p in prompts)
    got = srv.handle({'jsonrpc': '2.0', 'id': 6, 'method': 'prompts/get',
                      'params': {'name': 'plan_rig_from_brief', 'arguments': {'brief': 'urban robotaxi'}}})['result']
    assert got['messages'][0]['role'] == 'user' and 'urban robotaxi' in got['messages'][0]['content']['text']
    assert srv.handle({'jsonrpc': '2.0', 'id': 7, 'method': 'prompts/get',
                       'params': {'name': 'plan_rig_from_brief'}})['error']['code'] == -32602
    assert srv.handle({'jsonrpc': '2.0', 'id': 8, 'method': 'logging/setLevel',
                       'params': {'level': 'loud'}})['error']['code'] == -32602
    assert srv.handle({'jsonrpc': '2.0', 'id': 9, 'method': 'resources/subscribe',
                       'params': {'uri': 'vantage://nope'}})['error']['code'] == -32002


def test_resources_read_completion_and_structured_content(tmp_path, sample, monkeypatch):
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'cfg'))
    path = str(tmp_path / 'p.vantage')
    sample.save(path)
    out = _Out()
    srv = Server(Router(path, mode='file'), out=out)
    r = _ask(srv, out, 1, 'resources/read', {'uri': 'vantage://overview'})
    ct = r['result']['contents'][0]
    assert ct['mimeType'] == 'application/json' and 'cameras' in json.loads(ct['text'])
    r = _ask(srv, out, 2, 'resources/read', {'uri': 'vantage://camera/F'})
    assert '"F"' in r['result']['contents'][0]['text']
    r = _ask(srv, out, 3, 'resources/read', {'uri': 'vantage://study/power_budget'})
    assert r['result']['contents'][0]['mimeType'] == 'application/json'
    r = _ask(srv, out, 4, 'resources/read', {'uri': 'vantage://view/top.png'})
    assert any('blob' in c and c['mimeType'] == 'image/png' for c in r['result']['contents'])
    assert _ask(srv, out, 5, 'resources/read', {'uri': 'vantage://study/nope'})['error']['code'] == -32002
    if CT.doc_files():
        r = _ask(srv, out, 6, 'resources/read', {'uri': 'vantage://docs/README'})
        assert r['result']['contents'][0]['mimeType'] == 'text/markdown'
    r = _ask(srv, out, 7, 'completion/complete', {'ref': {'type': 'ref/resource', 'uri': 'vantage://camera/{tag}'},
                                                  'argument': {'name': 'tag', 'value': 'Q'}})
    assert r['result']['completion']['values'] == ['QL', 'QR']
    r = _ask(srv, out, 8, 'completion/complete', {'ref': {'type': 'ref/resource', 'uri': 'vantage://study/{name}'},
                                                  'argument': {'name': 'name', 'value': 'recor'}})
    assert r['result']['completion']['values'] == ['recording_budget']
    r = _ask(srv, out, 9, 'tools/call', {'name': 'get_coverage', 'arguments': {}})
    st = r['result']['structuredContent']
    assert st == json.loads(r['result']['content'][0]['text']) and 'blind_m2' in st


def test_subscription_is_told_when_the_project_changes_and_logs_follow_the_level(tmp_path, sample, monkeypatch):
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'cfg'))
    monkeypatch.setattr('vantage.mcp.server.WATCH_S', 0.2)
    path = str(tmp_path / 'p.vantage')
    sample.save(path)
    out = _Out()
    srv = Server(Router(path, mode='file'), out=out)
    _ask(srv, out, 1, 'tools/call', {'name': 'vantage_status', 'arguments': {}})
    assert out.q.empty()                                                      # no logs before a level is set
    assert _ask(srv, out, 2, 'resources/subscribe', {'uri': 'vantage://overview'})['result'] == {}
    assert _ask(srv, out, 3, 'logging/setLevel', {'level': 'info'})['result'] == {}
    time.sleep(0.5)
    r = _ask(srv, out, 4, 'tools/call', {'name': 'edit_project', 'arguments': {'ops': [
        {'op': 'set', 'target': 'camera', 'id': 'F', 'props': {'pitch': 36}}]}})
    assert not r['result']['isError']
    notes, end = [], time.time() + 10
    while time.time() < end and not ({'notifications/resources/updated', 'notifications/message'} <=
                                      {n.get('method') for n in notes}):
        try:
            notes.append(out.q.get(timeout=0.5))
        except queue.Empty:
            pass
    assert {'uri': 'vantage://overview'} in [n['params'] for n in notes if n['method'] == 'notifications/resources/updated']
    logs = [n['params'] for n in notes if n['method'] == 'notifications/message']
    assert any(lg['data'].get('tool') == 'edit_project' and lg['level'] == 'info' for lg in logs)
    _ask(srv, out, 5, 'resources/unsubscribe', {'uri': 'vantage://overview'})
    srv._stop.set()


def test_calls_keep_their_order_reads_together_writes_alone():
    from vantage.mcp.server import _Order
    o, log = _Order(), []

    def run(name, write, hold):
        t = o.enter(write)
        log.append(('start', name))
        time.sleep(hold)
        log.append(('end', name))
        o.leave(t)
    ths = []
    for name, write, hold in (('r1', False, 0.3), ('r2', False, 0.1), ('w', True, 0.1), ('r3', False, 0.05)):
        th = threading.Thread(target=run, args=(name, write, hold))
        th.start()
        ths.append(th)
        time.sleep(0.02)
    for th in ths:
        th.join()
    i = {e: n for n, e in enumerate(log)}
    assert i[('start', 'r2')] < i[('end', 'r1')]                              # reads side by side
    assert i[('start', 'w')] > max(i[('end', 'r1')], i[('end', 'r2')])        # a write after every call before it
    assert i[('start', 'r3')] > i[('end', 'w')]                               # and before every call after it


# ======================================================================= jobs, progress and cancellation (stdio)
def test_stdio_progress_cancellation_and_background_jobs(tmp_path, sample):
    path = str(tmp_path / 'p.vantage')
    sample.save(path)
    env = dict(os.environ, XDG_CONFIG_HOME=str(tmp_path / 'cfg'), QT_QPA_PLATFORM='offscreen')
    p = subprocess.Popen([sys.executable, os.path.join(ROOT, 'run_vantage.py'), '--mcp', '--mcp-project', path],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, env=env,
                         bufsize=1)
    got, cv = [], threading.Condition()

    def reader():
        for line in p.stdout:
            with cv:
                got.append(json.loads(line))
                cv.notify_all()
    threading.Thread(target=reader, daemon=True).start()

    def send(doc):
        p.stdin.write(json.dumps(doc) + '\n')
        p.stdin.flush()

    def answer(mid, timeout=300):
        end = time.time() + timeout
        with cv:
            while time.time() < end:
                hit = next((d for d in got if d.get('id') == mid and 'method' not in d), None)
                if hit:
                    return hit
                cv.wait(0.2)
        raise TimeoutError(mid)

    def tool(mid, name, **args):
        send({'jsonrpc': '2.0', 'id': mid, 'method': 'tools/call', 'params': {'name': name, 'arguments': args}})
        return json.loads(answer(mid)['result']['content'][0]['text'])
    try:
        send({'jsonrpc': '2.0', 'id': 1, 'method': 'initialize', 'params': {'protocolVersion': '2025-06-18'}})
        answer(1)
        # a slow call with a progress token: progress every PROGRESS_S; cancelled, it gets no answer
        send({'jsonrpc': '2.0', 'id': 2, 'method': 'tools/call', 'params': {
            'name': 'run_batch', 'arguments': {'args': ['--validate', 'val']}, '_meta': {'progressToken': 'p2'}}})
        end = time.time() + 60
        while time.time() < end and sum(d.get('method') == 'notifications/progress' for d in got) < 2:
            time.sleep(0.2)
        prog = [d['params'] for d in got if d.get('method') == 'notifications/progress']
        assert len(prog) >= 2 and prog[0]['progressToken'] == 'p2' and prog[1]['progress'] > prog[0]['progress']
        send({'jsonrpc': '2.0', 'method': 'notifications/cancelled', 'params': {'requestId': 2}})
        assert tool(3, 'vantage_status')['mode'] == 'file'                     # the server goes on
        time.sleep(1.0)
        assert not any(d.get('id') == 2 for d in got)                           # no answer to a cancelled call
        # a background job: started, waited for, its result
        j = tool(4, 'start_job', tool='run_batch', arguments={'args': ['--recording-budget', 'rb.json']})
        assert j['state'] in ('queued', 'running') and j['job_id'].startswith('job-')
        send({'jsonrpc': '2.0', 'id': 5, 'method': 'tools/call',
              'params': {'name': 'get_job', 'arguments': {'job_id': j['job_id'], 'wait_s': 240}}})
        r = answer(5)['result']
        assert r['structuredContent']['job']['state'] == 'done', r
        assert r['structuredContent']['result']['exit_code'] in (0, 2, 6)
        # a long job cancelled while it runs
        j2 = tool(6, 'start_job', tool='run_batch', arguments={'args': ['--validate', 'val2']})
        time.sleep(3)
        assert tool(7, 'cancel_job', job_id=j2['job_id'])['state'] in ('running', 'cancelled')
        assert tool(8, 'get_job', job_id=j2['job_id'], wait_s=60)['state'] == 'cancelled'
        jobs = tool(9, 'list_jobs')['jobs']
        assert [x['state'] for x in jobs] == ['cancelled', 'done']
        assert tool(10, 'start_job', tool='get_job')['error'].startswith('a job runs a tool')
    finally:
        p.stdin.close()
        p.wait(120)


def test_a_batch_run_is_stopped_when_cancelled(tmp_path, sample):
    from vantage.mcp import batchrun as B
    ev = threading.Event()
    threading.Timer(2.0, ev.set).start()
    t0 = time.time()
    r = B.run(sample.snapshot(), ['--validate', 'v'], str(tmp_path), 600, cancel=ev)
    assert r['exit_code'] == -15 and r['meaning'] == 'cancelled' and time.time() - t0 < 30


# ======================================================================= imports and aiming
def test_import_vehicle_guesses_units_and_axes_finds_wheels_and_puts_the_origin_on_the_rear_axle(tmp_path):
    from vantage.core.vehicle_import import import_vehicle
    f = str(tmp_path / 'car.obj')
    car_obj(f, 1000.0, True)
    v, rep = import_vehicle(f, units='mm', up='+Y', forward='-Z')
    assert rep['units'] == 'mm (given)' and rep['wheels_found'] == ['FL', 'FR', 'RL', 'RR']
    assert rep['size_m'] == {'length': 4.2, 'width': 1.84, 'height': 1.45} and rep['wheelbase_m'] == 2.6
    assert np.allclose(v.wheel_centres['RL'], [0.0, 0.92, 0.3]) and np.allclose(v.wheel_centres['FR'], [2.6, -0.92, 0.3])
    lo, hi = v.mesh.bbox
    assert abs(lo[2]) < 1e-6 and abs(lo[0] + 0.8) < 1e-6                     # on the ground; 0.8 m rear overhang
    g, rep2 = import_vehicle(f)                                               # mm and y up guessed from the size
    assert rep2['units'].startswith('mm (guessed') and rep2['up'].startswith('+Y (guessed')
    assert rep2['size_m'] == rep['size_m']
    with pytest.raises(ValueError):
        import_vehicle(f, up='+Y', forward='-Y')
    car_obj(str(tmp_path / 'tiny.obj'), 1.0 / 1000, False)                   # kilometres: no unit makes it a car
    _, rep3 = import_vehicle(str(tmp_path / 'tiny.obj'))
    assert 'check' in rep3


def test_import_vehicle_and_layout_over_mcp_are_undo_steps(tmp_path, sample, monkeypatch):
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'cfg'))
    path = str(tmp_path / 'p.vantage')
    sample.save(path)
    car_obj(str(tmp_path / 'car.obj'))
    (tmp_path / 'layout.csv').write_text('tag,name,x_mm,y_mm,z_mm,yaw,pitch\nF,Front,3600,0,1300,0,30\n'
                                         'R,Rear,-900,0,1300,180,30\nL,Left,1500,950,1500,90,45\n')
    router = Router(path, mode='file')

    def call(tool, **a):
        r = router.call(tool, a)
        return json.loads(r['content'][0]['text']), r['isError']
    d, err = call('import_layout', path=str(tmp_path / 'layout.csv'), name='From CSV')
    assert not err and d['variant'] == 'From CSV' and [c['tag'] for c in d['cameras']] == ['F', 'R', 'L']
    assert call('vantage_status')[0]['variant'] == 'From CSV'
    d, err = call('import_vehicle', path=str(tmp_path / 'car.obj'), units='mm', up='+Y', forward='-Z')
    assert not err and d['imported']['wheels_found'] == ['FL', 'FR', 'RL', 'RR'] and d['cameras_kept']
    assert d['undo_step'].startswith('Claude · imported car.obj')
    from vantage.core.project import Project
    assert Project.load(path).vehicle.wheel_centres['FL'] == [2.6, 0.92, 0.3]
    assert call('undo_last_change')[0]['result'].startswith('undid')
    assert Project.load(path).vehicle.name == sample.vehicle.name                # the Tigor again
    d, err = call('import_vehicle', path=str(tmp_path / 'nope.glb'))
    assert err and d['changed'] == 'nothing'
    d, err = call('set_view', view='persp')
    assert err and 'window' in d['error']


def test_aim_points_a_camera_mirrors_it_and_levels_its_roll(sample):
    p = sample.snapshot()
    f = p.variant.by_tag('F')
    keys, lines = E.apply(p, [{'op': 'aim', 'target': 'camera', 'id': 'F', 'props': {'at': [f.x + 5.0, f.y, 0.0]}}])
    assert lines[0].startswith('aimed camera F') and f.yaw == 0.0
    assert abs(f.pitch - round(math.degrees(math.atan2(f.z, 5.0)), 1)) < 0.06
    ml, mr = p.variant.by_tag('ML'), p.variant.by_tag('MR')
    E.apply(p, [{'op': 'aim', 'target': 'camera', 'id': 'ML', 'props': {'at': [ml.x, ml.y + 3.0], 'level_roll': True}}])
    assert abs(ml.yaw - 90.0) < 0.06 and ml.roll == 0.0
    assert abs(mr.yaw + 90.0) < 0.06 and mr.pitch == ml.pitch                  # the linked partner mirrors it
    E.apply(p, [{'op': 'aim', 'target': 'camera', 'id': 'QR', 'props': {'along': 'normal'}}])
    assert p.variant.by_tag('QR').pitch >= 0.0
    for bad in ({'at': [1.0]}, {'at': [f.x, f.y, f.z]}, {'along': 'sky'}, {}):
        with pytest.raises(E.EditError):
            E.apply(p, [{'op': 'aim', 'target': 'camera', 'id': 'F', 'props': bad}])
    with pytest.raises(E.EditError):
        E.apply(p, [{'op': 'aim', 'target': 'sensor', 'id': 'L1', 'props': {'at': [1, 1]}}])


def test_catalog_resolves_every_listed_resource_and_template():
    for r in CT.resource_list():
        assert CT.resolve(r['uri'])
    assert CT.resolve('vantage://camera/QL') == ('tool', 'get_camera', {'camera': 'QL'})
    assert CT.resolve('vantage://view/camera-F') == ('tool', 'look', {'view': 'camera', 'camera': 'F'})
    assert CT.resolve('vantage://study/sync_plan') == ('tool', 'get_sync_plan', {})
    assert CT.resolve('vantage://job/job-12') == ('job', 'job-12')
    for name in CT.study_names():
        assert SV.BY_NAME[f'get_{name}'].required == ()
    for p in CT.PROMPTS:                                                     # every prompt renders with its arguments
        args = {a['name']: 'x' for a in p['arguments']}
        msg = CT.prompt_messages(p['name'], args)['messages'][0]['content']['text']
        assert len(msg) > 80 and '{' not in msg.replace('{"', '')
    named = {t.name for t in SV.TOOLS}
    for text in CT._TEXT.values():                                            # prompts name only real tools
        for word in set(w.strip('.,;:()') for w in text.split()):
            if '_' in word and word.islower() and not word.startswith('--') and word.isidentifier() and \
                    word not in ('max_cameras',):
                assert word in named or word in ('edit_project',), word


# ======================================================================= the window
def test_view_render_and_turntable_in_the_window(app_win, tmp_path):
    app, w = app_win
    router = Router(mode='live')
    out = {}

    def talk():
        out['view'] = json.loads(router.call('set_view', {'view': 'persp', 'orbit': {'az': 30, 'el': 25},
                                                          'overlays': {'camlight': True}, 'heatmap': 'coverage'})
                                 ['content'][0]['text'])
        out['bad'] = router.call('set_view', {'view': 'sideways'})
        out['render'] = router.call('render_view', {'path': str(tmp_path / 'r.png'), 'scale': 1})
        out['through'] = json.loads(router.call('set_view', {'view': 'through', 'camera': 'F'})['content'][0]['text'])
        out['back'] = router.call('set_view', {'view': 'persp'})
        out['turn'] = json.loads(router.call('turntable_video', {'path': str(tmp_path / 't'), 'seconds': 1, 'fps': 5,
                                                                 'scale': 0.5})['content'][0]['text'])
    t = threading.Thread(target=talk)
    t.start()
    end = time.time() + 300
    while t.is_alive() and time.time() < end:
        _pump(app, 0.05)
    assert not t.is_alive()
    v = out['view']
    assert v['done']['view'] == 'persp' and v['done']['orbit']['az'] == 30 and v['page'] == 'viewports'
    assert w.ctx.overlays.get('camlight') is True and w.ctx.heatmap == 'coverage'
    assert out['bad']['isError']
    r = out['render']
    assert not r['isError'] and os.path.isfile(tmp_path / 'r.png')
    doc = json.loads(r['content'][0]['text'])
    assert doc['pixels'][0] > 100 and any(c['type'] == 'image' for c in r['content'])
    assert out['through']['done']['camera'] == 'F'
    assert out['turn']['frames'] == 5 and os.path.exists(out['turn']['path'])
    w.ctx.set_overlay('camlight', False)


def _pump(app, s):
    end = time.time() + s
    while time.time() < end:
        app.processEvents()
        time.sleep(0.005)


@pytest.fixture(scope='module')
def app_win():
    from PyQt6.QtWidgets import QApplication
    from vantage.ui.mainwindow import MainWindow
    from vantage.ui.viewport3d import configure_default_format
    configure_default_format()
    app = QApplication.instance() or QApplication([])
    w = MainWindow(samples.tigor_sample_project(), soft_3d=True)
    w.resize(1200, 800)
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
