"""Pictures for the model (assistant/vision.py, the look tool): the coverage map draws each blind cell where the analysis
has it, the right way up; the caption's figures are the analysis's; every picture fits the size limits; the Claude
request carries the picture as an image block inside the tool_result (against tests/fake_anthropic.py, never the real
API); Groq is not offered look and never receives an image; the transcript shows what the model saw as a thumbnail."""
import base64
import json
import os
import threading
import time

import numpy as np
import pytest

from vantage.assistant import agent as A
from vantage.assistant import tools as T
from vantage.assistant import vision as VI
from vantage.core import rules, samples
from vantage.core.analysis import Engine


@pytest.fixture(autouse=True)
def _keep_assistant_settings():
    """The assistant's settings (and keys) as they were before each test: these tests change consent, the automatic
    choice and the keys, which other test modules in the same session must not inherit."""
    from PyQt6.QtCore import QSettings
    from vantage.assistant import providers as _PR
    from vantage.assistant import secrets as _K
    s = QSettings('Vantage', 'Vantage')
    s.beginGroup('assistant')
    before = {k: s.value(k) for k in s.allKeys()}
    s.endGroup()
    keys = {p: _K.load(p) for p in _PR.PROVIDERS}
    yield
    s.beginGroup('assistant')
    s.remove('')
    for k, v in before.items():
        s.setValue(k, v)
    s.endGroup()
    for p, (k, src) in keys.items():
        if k and src != 'environment':
            _K.save(k, p)
        elif not k:
            _K.delete(p)


@pytest.fixture(scope='module')
def eng():
    return Engine()


@pytest.fixture(scope='module')
def gappy(eng):
    """The sample with camera F switched off: a large blind patch in front of the car."""
    p = samples.tigor_sample_project()
    next(c for c in p.variant.cameras if c.tag == 'F').visible = False
    return p, eng.analyse(p)


@pytest.fixture(scope='module')
def sample(eng):
    p = samples.tigor_sample_project()
    res = eng.analyse(p)
    vg = eng.vehicle_geom(p)
    mounts = {c.uid: rules.mount_info(p, c, vg) for c in p.variant.cameras}
    return p, eng, res, mounts, rules.collect_issues(p, res, vg, mounts)


def _decode(pic):
    import cv2
    arr = cv2.imdecode(np.frombuffer(pic.data, np.uint8), cv2.IMREAD_COLOR)
    assert arr is not None
    return arr[..., ::-1]                    # RGB


def _near_marks(pic, project, px, py, r=16):
    """Is the pixel near something drawn over the cells (a camera ring and its tag)?"""
    for c in project.variant.cameras:
        if c.visible:
            cx, cy = pic.world_to_px(c.x, c.y)
            if abs(cx - px) < r + 30 and abs(cy - py) < r:
                return True
    return False


# ============================================================================================ rendering
def test_the_map_draws_every_blind_cell_where_the_analysis_has_it(gappy):
    p, res = gappy
    g = res.grid
    blind = g.roi & (res.count == 0)
    assert blind.sum() * g.cell ** 2 > 0.3, 'F off must leave a large blind patch'
    pic = VI.render_map(p, res)
    img = _decode(pic)
    jj, ii = np.nonzero(blind)
    hits = checked = 0
    for j, i in zip(jj, ii):
        px, py = pic.world_to_px(g.xs[i], g.ys[j])
        if _near_marks(pic, p, px, py):
            continue
        checked += 1
        hits += int(np.abs(img[int(py), int(px)].astype(int) - VI.BLIND).max() <= 6)
    assert checked > 0.9 * len(jj) and hits >= 0.9 * checked, (hits, checked)   # the outline crosses edge cells
    # covered cells far from any marking are never drawn red, body cells are body grey
    cov = np.argwhere(g.roi & (res.count > 0))
    for j, i in cov[::97]:
        px, py = pic.world_to_px(g.xs[i], g.ys[j])
        if not _near_marks(pic, p, px, py, 40):
            assert np.abs(img[int(py), int(px)].astype(int) - VI.BLIND).max() > 30
    body = np.argwhere(g.body)
    jb, ib = body[len(body) // 2]
    px, py = pic.world_to_px(g.xs[ib], g.ys[jb])
    assert np.abs(img[int(py), int(px)].astype(int) - VI.BODY).max() <= 6
    # the red area in the picture is the blind area (an independent count of pixels)
    red = (np.abs(img.astype(int) - VI.BLIND).max(axis=2) <= 6).sum()
    s = pic.transform[2]
    assert red / s ** 2 == pytest.approx(res.metrics['blind_m2'], rel=0.06)


def test_the_map_is_the_right_way_up(gappy):
    p, res = gappy
    pic = VI.render_map(p, res)
    pos = {c.tag: pic.world_to_px(c.x, c.y) for c in p.variant.cameras}
    assert pos['F'][1] < pos['ML'][1] < pos['QL'][1] < pos['R'][1]            # the front at the top
    assert pos['ML'][0] < pos['MR'][0] and pos['QL'][0] < pos['QR'][0]        # the left on the left
    # the blind patch is in front of the car: above F's row
    b = pic.info['blind_patches'][0]
    assert pic.world_to_px(*b['centre'])[1] < pos['ML'][1]
    assert b['centre'][0] > p.variant.by_tag('ML').x


def test_caption_figures_are_the_analysis(gappy):
    p, res = gappy
    m = res.metrics
    for layer in VI.LAYERS:
        pic = VI.render_map(p, res, layer)
        i = pic.info
        assert i['blind_m2'] == pytest.approx(m['blind_m2'], abs=1e-4)
        assert i['covered_pct'] == pytest.approx(m['covered_pct'], abs=0.01)
        assert i['min_density_px_cm'] == pytest.approx(m['min_density'], abs=1e-3)
        assert i['multi_pct'] == pytest.approx(m['multi_pct'], abs=0.01)
        assert f'blind {m["blind_m2"]:.3f} m²' in pic.caption and f'covered {m["covered_pct"]:.1f} %' in pic.caption
        assert 'F' not in i['cameras'] and 'ML' in i['cameras']
    # the numbered patches: largest first, and all of them together are the blind area (an independent count)
    from scipy import ndimage
    g = res.grid
    lab, n = ndimage.label(g.roi & (res.count == 0), structure=np.ones((3, 3)))
    areas = sorted((np.bincount(lab.ravel())[1:] * g.cell ** 2).tolist(), reverse=True)
    got = [b['area_m2'] for b in pic.info['blind_patches']]
    assert got == pytest.approx(areas[:len(got)], abs=1e-4) and got == sorted(got, reverse=True)
    if n <= 6:
        assert sum(got) == pytest.approx(m['blind_m2'], abs=1e-3)
    assert f'1: {got[0]:.3f} m²' in pic.caption


def test_closeup_counts_the_cells_in_its_window(gappy):
    p, res = gappy
    g = res.grid
    b = VI.render_map(p, res).info['blind_patches'][0]
    x, y = b['centre']
    pic = VI.render_map(p, res, centre=(x, y), radius_m=0.8)
    X, Y = np.meshgrid(g.xs, g.ys)
    # the window is whole cells around the point: cells whose centre lies within radius + half a cell
    win = (np.abs(X - x) <= 0.8 + g.cell) & (np.abs(Y - y) <= 0.8 + g.cell)
    i0 = int(np.floor((x - 0.8 - g.x0) / g.cell))
    i1 = int(np.ceil((x + 0.8 - g.x0) / g.cell))
    j0 = int(np.floor((y - 0.8 - g.y0) / g.cell))
    j1 = int(np.ceil((y + 0.8 - g.y0) / g.cell))
    sub = (g.roi & (res.count == 0))[j0:j1, i0:i1]
    assert pic.info['window_blind_m2'] == pytest.approx(sub.sum() * g.cell ** 2, abs=1e-4)
    assert pic.info['window_blind_m2'] <= (g.roi & (res.count == 0) & win).sum() * g.cell ** 2 + 1e-9
    assert pic.view == 'closeup' and pic.transform[2] > VI.render_map(p, res).transform[2] * 2   # larger
    img = _decode(pic)
    d2 = np.where(g.roi & (res.count == 0), (X - x) ** 2 + (Y - y) ** 2, np.inf)
    bj, bi = np.unravel_index(np.argmin(d2), X.shape)
    px, py = pic.world_to_px(g.xs[bi], g.ys[bj])              # the blind cell nearest the marked point
    assert np.abs(img[int(py), int(px)].astype(int) - VI.BLIND).max() <= 6    # is left visible
    with pytest.raises(ValueError):
        VI.render_map(p, res, centre=(80.0, 80.0), radius_m=1.0)


def test_camera_view_is_the_synthetic_render(sample):
    import cv2
    from vantage.core import synth
    p, e, res, *_ = sample
    cam = p.variant.by_tag('F')
    pic = VI.render_camera(p, e, cam, res, (180.0, 120.0))
    assert pic.media_type == 'image/jpeg' and pic.data[:3] == b'\xff\xd8\xff'
    img = _decode(pic).astype(float)
    ref = synth.render_world(p, e, cam, synth.parking_world(p), pic.width, pic.height, seed=0, noise=0.0)
    ref = np.asarray(ref, float)[..., :3]
    body = (slice(40, None), slice(None))                         # below the label strip
    assert np.abs(img[body] - ref[body]).mean() < 4.0             # the same picture, JPEG-compressed
    d = res.per_cam[cam.uid]
    assert pic.info['seen_m2'] == pytest.approx(d['seen_m2'], abs=1e-3)
    assert pic.info['only_this_camera_m2'] == pytest.approx(d['unique_m2'], abs=1e-3)
    assert f'{d["seen_m2"]:.2f} m²' in pic.caption and 'Not a real capture' in pic.caption
    del cv2


def test_every_picture_fits_the_limits(sample, gappy):
    p, e, res, *_ = sample
    pics = [VI.render_map(p, res, 'density'), VI.render_map(p, res, 'redundancy'),
            VI.render_map(*gappy, centre=(3.8, 0.0), radius_m=5.0), VI.render_map(p, res, centre=(0, 1), radius_m=0.25),
            VI.render_camera(p, e, p.variant.by_tag('ML'), res)]
    for pic in pics:
        assert max(pic.width, pic.height) <= VI.MAX_EDGE and pic.width * pic.height <= VI.MAX_PIXELS
        assert len(pic.data) < 400_000, (pic.view, len(pic.data))              # far below the API's 5 MB
        assert pic.tokens == int(np.ceil(pic.width * pic.height / 750)) <= 1700
        h, w = _decode(pic).shape[:2]
        assert (w, h) == (pic.width, pic.height)
        assert len(pic.caption) < 2000 and pic.summary()['bytes'] == len(pic.data)
    assert VI.fit(4000, 3000) == (1280, 960) and VI.fit(800, 600) == (800, 600)
    assert VI.image_tokens(1280, 960) == 1639


# ============================================================================================ the tool
def _tc(sample, **kw):
    p, e, res, mounts, issues = sample
    return T.ToolContext(p.snapshot(), res, e.share(), issues, mounts, **kw)


def test_look_returns_the_picture_only_to_a_model_that_takes_images(sample):
    tc = _tc(sample, vision=True)
    text, ok, pics = T.run_full(tc, 'look', {'view': 'top'})
    out = json.loads(text)
    assert ok and len(pics) == 1 and out['picture']['width_px'] == pics[0].width
    assert out['caption'] == pics[0].caption and '_pictures' not in out and 'iVBOR' not in text
    assert out['figures']['blind_m2'] == pytest.approx(sample[2].metrics['blind_m2'], abs=1e-4)
    text, ok, pics = T.run_full(tc, 'look', {'view': 'camera', 'camera': 'QR'})
    assert ok and pics[0].media_type == 'image/jpeg'
    text, ok, pics = T.run_full(tc, 'look', {'view': 'closeup', 'x': -0.8, 'y': 0.0, 'radius_m': 1.0})
    assert ok and pics[0].view == 'closeup'
    # without image input: the caption and the figures, no picture
    tc2 = _tc(sample, vision=False)
    for args in ({'view': 'top'}, {'view': 'camera', 'camera': 'F'}, {'view': 'closeup', 'x': -0.8, 'y': 0.0}):
        text, ok, pics = T.run_full(tc2, 'look', args)
        out = json.loads(text)
        assert ok and pics == [] and out['picture'] is None and '(No picture' in out['caption'], args
    assert json.loads(T.run_full(tc2, 'look', {'view': 'top'})[0])['figures']['blind_m2'] == \
        pytest.approx(sample[2].metrics['blind_m2'], abs=1e-4)
    # bad input is an error the model can fix, never an exception
    for args in ({'view': 'camera', 'camera': 'NOPE'}, {'view': 'closeup'}, {'view': 'sideways'}):
        text, ok, pics = T.run_full(tc, 'look', args)
        assert not ok and 'error' in json.loads(text) and pics == []
    assert T.run(tc, 'look', {'view': 'top'})[1]                          # the text-only entry point still works


# ============================================================================================ providers
def test_groq_is_not_offered_look_and_never_gets_an_image(sample, monkeypatch):
    from fake_groq import FakeGroq
    srv = FakeGroq().start()
    monkeypatch.setenv('VANTAGE_ASSISTANT_URL', srv.url)
    try:
        s = A.Session('test-key', A.Settings('openai/gpt-oss-120b'))
        assert not s.vision and not any(C.vision for C in __import__('vantage.assistant.client',
                                                                     fromlist=['KNOWN']).KNOWN.values())
        srv.script += [srv.tool_call('look', {'view': 'top'}, 'c1'), srv.answer('From the figures.')]
        ev = []
        s.run_turn('Where is the ground blind?', _tc(sample), ev.append, threading.Event())
        assert ev[-1].data['ok']
        names = [t['function']['name'] for t in srv.requests[0]['tools']]
        assert 'look' not in names and len(names) == len(T.TOOLS) - 1
        assert 'pictures need Claude' in srv.requests[0]['messages'][0]['content']
        tool_msg = srv.requests[1]['messages'][-1]
        assert tool_msg['role'] == 'tool' and '(No picture' in json.loads(tool_msg['content'])['caption']
        done = [e for e in ev if e.kind == 'tool_done']
        assert done and done[0].data['pictures'] == []
        for _, body in srv.raw:
            assert b'iVBORw0KGgo' not in body and b'/9j/' not in body and b'"image"' not in body
    finally:
        srv.stop()


def test_claude_gets_the_picture_as_an_image_block_in_the_tool_result(sample, monkeypatch):
    pytest.importorskip('anthropic', reason='the anthropic SDK is not installed in this Python')
    from fake_anthropic import FakeAnthropic
    from vantage.assistant import claude as CL
    srv = FakeAnthropic(key='sk-ant-test-key').start()
    monkeypatch.setenv('VANTAGE_CLAUDE_URL', srv.url)
    try:
        s = CL.ClaudeSession('sk-ant-test-key', A.Settings('claude-opus-5-5'))
        assert s.vision and 'look' in [t['name'] for t in s.tools]
        look = next(t for t in s.tools if t['name'] == 'look')
        assert look.get('strict') is True and look['input_schema']['additionalProperties'] is False
        srv.script += [srv.reply(tools=[('look', {'view': 'top'}, 'toolu_a'),
                                        ('look', {'view': 'camera', 'camera': 'F'}, 'toolu_b')]),
                       srv.reply(text='The blind patch is below the rear bumper.')]
        ev = []
        s.run_turn('Where is the ground blind?', _tc(sample), ev.append, threading.Event())
        assert ev[-1].data['ok'], [e.data for e in ev if e.kind == 'error']
        user = srv.requests[1]['messages'][-1]
        assert user['role'] == 'user' and [b['type'] for b in user['content']] == ['tool_result', 'tool_result']
        for b, mt, sig in zip(user['content'], ('image/png', 'image/jpeg'), (b'\x89PNG', b'\xff\xd8\xff')):
            assert [c['type'] for c in b['content']] == ['text', 'image']
            assert json.loads(b['content'][0]['text'])['picture']['format'] == mt
            src = b['content'][1]['source']
            assert src['type'] == 'base64' and src['media_type'] == mt
            assert base64.b64decode(src['data']).startswith(sig)
        assert [i['media_type'] for i in srv.images] == ['image/png', 'image/jpeg']
        assert all(i['where'] == 'tool_result' and max(i['width'], i['height']) <= VI.MAX_EDGE for i in srv.images)
        # the inspector shows the request with the image data shortened; the history keeps the picture
        sent = json.dumps(s.last_payload)
        assert 'base64, shortened here' in sent and 'iVBORw0KGgo' not in sent
        assert s.history()[2]['content'][0]['content'][1]['type'] == 'image'
        pics = [p for e in ev if e.kind == 'tool_done' for p in e.data['pictures']]
        assert [p.view for p in pics] == ['top', 'camera']
        # the next question resends the picture unchanged (append-only history)
        srv.script += [srv.reply(text='ok')]
        s.run_turn('Thanks', _tc(sample), ev.append, threading.Event())
        assert srv.requests[2]['messages'][2] == srv.requests[1]['messages'][2]
    finally:
        srv.stop()


def test_the_fake_server_refuses_a_bad_image():
    pytest.importorskip('anthropic', reason='the anthropic SDK is not installed in this Python')
    from fake_anthropic import FakeAnthropic
    f = FakeAnthropic()
    ok = {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png',
                                      'data': base64.b64encode(b'\x89PNG\r\n\x1a\n' + b'\0\0\0\r' + b'IHDR'
                                                               + (100).to_bytes(4, 'big') + (50).to_bytes(4, 'big')
                                                               ).decode()}}
    msg = lambda b: {'messages': [{'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 't',  # noqa
                                                               'content': [b]}]}]}
    assert f.check_images(msg(ok)) == '' and f.images[-1]['width'] == 100
    bad = json.loads(json.dumps(ok))
    bad['source']['media_type'] = 'image/jpeg'
    assert 'not a image/jpeg' in f.check_images(msg(bad))
    bad['source'] = {'type': 'url', 'url': 'http://x'}
    assert 'unsupported' in f.check_images(msg(bad))


# ============================================================================================ the transcript
from test_gui import app, win, pump, _wait  # noqa: E402,F401  (fixtures)


def test_the_transcript_shows_what_the_model_saw(app, win):
    from PyQt6.QtCore import QPoint, Qt
    from PyQt6.QtTest import QTest
    from vantage.ui.assistant import AssistantTurn, PictureDialog, PictureThumb
    win.show_assistant()
    pump(app, 0.2)
    p, res = win.store.project, win.ctx.analysis.result
    pic = VI.render_map(p, res)
    turn = AssistantTurn(win.ctx.assistant)
    win.assistant_panel._add(turn)
    turn.on(A.Event('tool_start', {'id': 'c1', 'name': 'look', 'label': 'Look at the design', 'kind': 'look',
                                   'args': {'view': 'top'}}))
    turn.on(A.Event('tool_done', {'id': 'c1', 'name': 'look', 'result': '{}', 'ok': True, 'ms': 5,
                                  'pictures': [pic]}))
    turn.on(A.Event('done', {'ok': True}))
    pump(app, 0.2)
    th = turn.findChild(PictureThumb)
    assert th is not None and th.ok and th.isVisible()
    pm = th.img.pixmap()
    assert max(pm.width(), pm.height()) == PictureThumb.THUMB
    assert f'{pic.width}×{pic.height} px' in th.meta.text() and th.toolTip() == pic.caption
    QTest.mouseClick(th, Qt.MouseButton.LeftButton, pos=QPoint(10, 10))
    pump(app, 0.2)
    d = th.dialog
    assert isinstance(d, PictureDialog) and d.isVisible()
    assert d.img.pixmap().width() == pic.width and d.caption.text() == pic.caption
    d.close()
