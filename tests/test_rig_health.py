"""Live rig health: the statistics on known sequences, the stream sources, the drift indicator, the Live rig page and
the batch flag. Only files and synthetic sources: no device or network stream is opened."""
import json
import time

import numpy as np
import pytest

from vantage.core import rig_health as rh


def _frames(n=3, size=(640, 512), lo=60, hi=180, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.integers(lo, hi, (size[1], size[0], 3), dtype=np.uint8) for _ in range(n)]


def _feed(st, times, frames):
    for k, t in enumerate(times):
        st.add(frames[k % len(frames)], t)
    return st.summary(now=times[-1])


def _texts(reasons):
    return ' | '.join(r[1] for r in reasons)


# ----------------------------------------------------------------------------- statistics
def test_known_frame_rate_and_jitter():
    st = rh.StreamStats(25.0, (640, 512))
    s = _feed(st, [k / 25 for k in range(100)], _frames())
    assert s['fps'] == pytest.approx(25.0, abs=0.01) and s['jitter_ms'] < 0.01 and s['dropped'] == 0
    assert s['size'] == [640, 512] and s['size_ok'] and s['frozen_frames'] == 0
    status, reasons = rh.verdict(s, st.st, 0.0, 99 / 25)
    assert status == 'ok' and not reasons
    # intervals alternating 30 / 50 ms: 25 fps on average, 10 ms jitter, no drops (both under 1.5 x 40 ms)
    t = np.cumsum([0.03 if k % 2 else 0.05 for k in range(100)])
    s = _feed(rh.StreamStats(25.0), list(t), _frames())
    assert s['jitter_ms'] == pytest.approx(10.0, abs=0.1) and s['dropped'] == 0
    assert s['fps'] == pytest.approx(25.0, rel=0.25)


def test_dropped_frames_and_low_frame_rate():
    frames = _frames()
    every10 = [k / 30 for k in range(100) if k % 10 != 5]          # one frame in ten missing
    s = _feed(rh.StreamStats(30.0), every10, frames)
    assert s['dropped'] == 10 and s['dropped_share'] == pytest.approx(0.1, abs=0.005)
    status, reasons = rh.verdict(s, rh.HealthSettings(), 0.0, every10[-1])
    assert status == 'fault' and '% frames dropped' in _texts(reasons)
    one = [k / 30 for k in range(100) if k != 50]                    # a single drop in 99 frames: a warning
    s = _feed(rh.StreamStats(30.0), one, frames)
    assert s['dropped'] == 1
    status, reasons = rh.verdict(s, rh.HealthSettings(), 0.0, one[-1])
    assert status == 'warning' and '1.0 % frames dropped' in _texts(reasons)
    slow = [k / 12 for k in range(40)]                               # 12 fps from a 30 fps module
    s = _feed(rh.StreamStats(30.0), slow, frames)
    assert s['fps'] == pytest.approx(12.0, abs=0.05)
    status, reasons = rh.verdict(s, rh.HealthSettings(), 0.0, slow[-1])
    assert status == 'fault' and '12.0 fps of 30' in _texts(reasons)
    s = _feed(rh.StreamStats(30.0), [k / 25 for k in range(60)], frames)    # 25 of 30: under 90 %, a warning
    assert rh.verdict(s, rh.HealthSettings(), 0.0, 59 / 25)[0] == 'warning'


def test_frozen_stalled_and_starting():
    one = _frames(1)
    s = _feed(rh.StreamStats(30.0), [k / 30 for k in range(30)], one)       # the same image 30 times
    assert s['frozen_frames'] == 29
    status, reasons = rh.verdict(s, rh.HealthSettings(), 0.0, 29 / 30)
    assert status == 'fault' and 'frozen' in _texts(reasons)
    st = rh.StreamStats(30.0)
    s = _feed(st, [k / 30 for k in range(60)], _frames())
    s = st.summary(now=59 / 30 + 3.0)                                       # nothing for 3 s
    status, reasons = rh.verdict(s, rh.HealthSettings(), 0.0, 59 / 30 + 3.0)
    assert status == 'fault' and 'stalled' in _texts(reasons)
    empty = rh.StreamStats(30.0).summary(now=5.0)
    assert rh.verdict(empty, rh.HealthSettings(), 4.0, 5.0)[0] == 'starting'
    assert rh.verdict(empty, rh.HealthSettings(), 0.0, 20.0) == ('fault', [('fault', 'no frames after 20 s')])
    assert rh.verdict(empty, rh.HealthSettings(), 4.0, 5.0, error='Cannot open x')[1][0][1] == 'Cannot open x'


def test_exposure_blur_and_resolution():
    import cv2
    times = [k / 30 for k in range(40)]
    bright = [np.full((512, 640, 3), 254, np.uint8) for _ in range(2)]
    bright[1][:8] = 250                                                    # not frozen
    s = _feed(rh.StreamStats(30.0), times, bright)
    status, reasons = rh.verdict(s, rh.HealthSettings(), 0.0, times[-1])
    assert status == 'warning' and 'overexposed' in _texts(reasons) and 'too bright' in _texts(reasons)
    dark = [np.full((512, 640, 3), v, np.uint8) for v in (1, 2)]
    s = _feed(rh.StreamStats(30.0), times, dark)
    assert 'too dark' in _texts(rh.verdict(s, rh.HealthSettings(), 0.0, times[-1])[1])
    # a sharp checkerboard and the same image through a badly focused (or fogged) lens
    yy, xx = np.mgrid[0:512, 0:640]
    board = (((xx // 40) + (yy // 40)) % 2 * 160 + 50).astype(np.uint8)
    sharp = [np.dstack([np.roll(board, k, 1)] * 3) for k in range(2)]
    blurred = [cv2.GaussianBlur(f, (0, 0), 25) for f in sharp]
    a = _feed(rh.StreamStats(30.0), times, sharp)
    b = _feed(rh.StreamStats(30.0), times, blurred)
    assert a['sharpness'] > 100 * b['sharpness'] and b['sharpness'] < rh.HealthSettings().sharp_warn
    assert rh.verdict(a, rh.HealthSettings(), 0.0, times[-1])[0] == 'ok'
    status, reasons = rh.verdict(b, rh.HealthSettings(), 0.0, times[-1])
    assert status == 'warning' and 'blurred' in _texts(reasons)
    assert rh.image_stats(sharp[1])['sharpness'] == pytest.approx(a['sharpness'])          # of the newest frame
    # a stream at the wrong resolution for its module
    s = _feed(rh.StreamStats(30.0, (1920, 1536)), times, _frames(size=(640, 512)))
    assert s['size_ok'] is False
    status, reasons = rh.verdict(s, rh.HealthSettings(), 0.0, times[-1])
    assert status == 'warning' and '640×512, the module is 1920×1536' in _texts(reasons)
    # every threshold lives in the settings
    st = rh.HealthSettings.from_dict({'sharp_warn': 0.0, 'fps_warn': 'x', 'unknown': 1})
    assert st.sharp_warn == 0.0 and st.fps_warn == 0.9 and rh.verdict(b, st, 0.0, times[-1])[0] == 'ok'
    assert rh.HealthSettings.from_dict(st.to_dict()) == st


def test_sync_skew():
    a = np.arange(60) / 30
    assert rh.sync_skew({'F': a}) == (None, '')
    skew, who = rh.sync_skew({'F': a, 'R': a + 0.0004, 'ML': a + 0.010})
    assert who == 'ML' and skew == pytest.approx(10.0, abs=0.01)
    # free-running cameras: up to half a frame interval apart
    skew, _ = rh.sync_skew({'F': a, 'R': a + 0.5 / 30})
    assert skew == pytest.approx(1000 / 60, abs=0.01)
    st = rh.StreamStats(30.0)
    _feed(st, list(a), _frames())
    assert len(st.recent_times(1.0)) == 30


# ----------------------------------------------------------------------------- the project
def test_streams_are_saved_and_undoable(tmp_path):
    from vantage.core import samples
    from vantage.core.project import Project
    p = samples.tigor_sample_project()
    p.photos.source = ''
    assert p.streams == {}
    p.streams = {'F': 'rtsp://10.0.0.5/front', 'R': 'synthetic:15'}
    before = p.section('streams')
    p.streams['ML'] = '/dev/video2'
    p.restore('streams', before)
    assert p.streams == {'F': 'rtsp://10.0.0.5/front', 'R': 'synthetic:15'}
    assert p.snapshot().streams == p.streams and p.snapshot().streams is not p.streams
    path = str(tmp_path / 'rig.vantage')
    p.save(path)
    assert Project.load(path).streams == p.streams
    assert rh.describe('rtsp://x') == 'rtsp' and rh.describe('synthetic:15') == 'synthetic'
    assert rh.describe('v4l2src ! appsink') == 'gstreamer' and rh.describe('/dev/video0') == 'device'
    assert rh.describe(path) == 'file' and rh.describe('') == ''
    assert rh.synthetic_fps('synthetic:15', 30) == 15 and rh.synthetic_fps('synthetic', 30) == 30
    assert rh.shown('rtsp://admin:s3cret@10.0.0.5:554/s1') == 'rtsp://admin:***@10.0.0.5:554/s1'
    assert rh.shown('rtsp://10.0.0.5/s1') == 'rtsp://10.0.0.5/s1'


# ----------------------------------------------------------------------------- sources and the monitor
def _video(path, size=(640, 512), n=24, fps=30.0, seed=1):
    import cv2
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), fps, size)
    assert vw.isOpened()
    yy, xx = np.mgrid[0:size[1], 0:size[0]]
    rng = np.random.default_rng(seed)
    for k in range(n):
        g = ((((xx + 3 * k) // 32) + (yy // 32)) % 2 * 120 + 60).astype(np.int16) + rng.integers(-4, 5, (size[1], size[0]))
        vw.write(np.dstack([np.clip(g, 0, 255).astype(np.uint8)] * 3))
    vw.release()
    return str(path)


def test_synthetic_source_delivers_at_its_rate():
    calls = []
    base = _frames(1, (320, 256))[0]

    def render():
        calls.append(1)
        return base
    got = []
    src = rh.SyntheticSource(render, fps=40, size=(640, 512), on_frame=lambda f, t: got.append((f, t)))
    src.start()
    t0 = time.monotonic()
    while len(got) < 40 and time.monotonic() - t0 < 5:
        time.sleep(0.02)
    src.stop()
    assert not src.running and len(calls) == 1
    assert len(got) >= 40 and got[0][0].shape == (512, 640, 3) and src.size == (640, 512)
    dt = np.diff([t for _, t in got[:40]])
    assert np.median(dt) == pytest.approx(1 / 40, abs=0.004)
    assert not np.array_equal(got[0][0], got[1][0])                     # fresh sensor noise: never frozen
    f1, f2 = src.latest(), src.latest()
    assert f1 is not f2 and f1.shape == (512, 640, 3)
    src.start()                                                          # restarts after a stop
    time.sleep(0.2)
    assert src.running
    src.stop(wait=False)
    bad = rh.SyntheticSource(lambda: 1 / 0, fps=10)
    bad.start()
    time.sleep(0.2)
    assert not bad.running and 'cannot render' in bad.error


def test_monitor_files_errors_and_reconnect(tmp_path):
    from vantage.core import samples
    p = samples.tigor_sample_project()
    p.photos.source = ''
    cams = [c for c in p.variant.cameras if c.visible]
    good = _video(tmp_path / 'F.avi')
    p.streams = {'F': good, 'R': str(tmp_path / 'missing.avi')}
    doc = rh.run_for(p, cams, 1.5, connect_timeout=5)
    cf, cr = doc['cameras']['F'], doc['cameras']['R']
    assert doc['schema'] == rh.SCHEMA and doc['rig']['clock'] == rh.CLOCK_NOTE
    assert cf['kind'] == 'file' and cf['frames'] > 15 and cf['fps'] == pytest.approx(30, rel=0.35)
    assert cf['size'] == [640, 512] and cf['size_ok'] is False and cf['status'] in ('warning', 'fault')
    assert any('the module is 1920×1536' in r for r in cf['reasons'])
    assert cr['status'] == 'fault' and 'Cannot open' in ' '.join(cr['reasons'])
    assert {doc['cameras'][c.tag]['status'] for c in cams if c.tag not in ('F', 'R')} == {'off'}
    assert doc['rig']['faults'] >= 1 and doc['rig']['not_configured'] == len(cams) - 2
    json.dumps(rh.jsonable(doc))
    # reconnect after 1, 2, 4 s
    mon = rh.RigMonitor()
    mon.configure(p, cams)
    mon.start()
    s = mon.streams['R']
    t0 = time.monotonic()
    while s.source.running and time.monotonic() - t0 < 5:
        time.sleep(0.02)
    now = time.monotonic()
    mon.poll(now)
    assert s.retry_at == pytest.approx(now + 1.0) and s.backoff == 2.0 and 'Cannot open' in s.error
    assert 'retry in' in ' '.join(mon.snapshot(now)['cameras']['R']['reasons'])
    mon.poll(now + 0.5)
    assert s.retry_at is not None                                          # not yet
    mon.poll(now + 1.1)
    assert s.retry_at is None
    while s.source.running and time.monotonic() - t0 < 10:
        time.sleep(0.02)
    mon.poll(now + 1.2)
    assert s.retry_at == pytest.approx(now + 1.2 + 2.0) and s.backoff == 4.0
    # changing one camera's source restarts only that stream
    fsrc = mon.streams['F'].source
    p.streams['R'] = good
    mon.update(p, cams)
    assert mon.streams['F'].source is fsrc and mon.streams['R'].spec == good and mon.streams['R'].source.running
    mon.stop()
    assert not any(st.source.running for st in mon.streams.values())


# ----------------------------------------------------------------------------- drift
@pytest.fixture(scope='module')
def planned():
    from vantage.core import samples
    from vantage.core.analysis import Engine
    p = samples.tigor_sample_project()
    p.photos.source = ''
    e = Engine()
    e.analyse(p)
    return p, e


def test_drift_rotated_camera_scores_worse(planned):
    from vantage.core.mathutil import rot_zyx
    p, e = planned
    warn = rh.HealthSettings().drift_warn_px
    for tag in ('F', 'QL'):
        c = p.variant.by_tag(tag)
        ev = rh.expected_view(p, e, c)
        assert len(ev.edge) > 100 and ev.contours and ev.horizon and ev.sensor == (1920, 1536)
        assert all(np.all((h[:, 0] >= 0) & (h[:, 0] <= 1920)) for h in ev.horizon)
        plan = rh.synthetic_render(p, e, c)()
        moved = rh.synthetic_render(p, e, c, pose=(rot_zyx(c.yaw + 5, c.pitch, c.roll), c.pos()))()
        a, b = rh.alignment(plan, ev), rh.alignment(moved, ev)
        assert a is not None and b is not None and a < warn < b, (tag, a, b)
    # a lens covered by a uniform surface: no edge to compare with
    assert rh.alignment(np.full((1536, 1920, 3), 90, np.uint8), ev) is None
    s = _feed(rh.StreamStats(30.0), [k / 30 for k in range(40)], _frames())
    status, reasons = rh.verdict(s, rh.HealthSettings(), 0.0, 39 / 30, drift_px=b)
    assert status == 'warning' and 'may have moved' in _texts(reasons)


# ----------------------------------------------------------------------------- batch
def test_batch_rig_health(planned, tmp_path):
    import argparse
    import cv2
    from vantage import batch
    from vantage.core import samples
    # a recording of the front camera (the parking scene through its lens, with sensor noise)
    pp, e = planned
    view = rh.synthetic_render(pp, e, pp.variant.by_tag('F'))()
    view = np.minimum(cv2.resize(view, (1920, 1536)), 250)
    rng = np.random.default_rng(3)
    vw = cv2.VideoWriter(str(tmp_path / 'F.avi'), cv2.VideoWriter_fourcc(*'MJPG'), 10.0, (1920, 1536))
    for _ in range(20):
        vw.write((view + rng.integers(0, 5, view.shape[:2], dtype=np.uint8)[..., None])[..., ::-1])
    vw.release()
    p = samples.tigor_sample_project()
    p.photos.source = ''
    for c in p.variant.cameras:
        c.visible = c.tag in ('F', 'R')
        p.module_for(c).fps = 10.0                  # a slow rig keeps the timing clear of a busy test machine
    p.streams = {'F': str(tmp_path / 'F.avi'), 'R': 'synthetic'}
    proj = str(tmp_path / 'rig.vantage')
    p.save(proj)
    ap = argparse.ArgumentParser()
    ap.add_argument('path', nargs='?')
    ap.add_argument('--sample', action='store_true')
    ap.add_argument('--template')
    batch.add_arguments(ap)
    out = tmp_path / 'health.json'
    args = ap.parse_args([proj, '--rig-health', '2', str(out)])
    assert batch.wanted(args) and args.rig_health == ['2', str(out)]
    assert batch.run(args) == 0
    doc = json.loads(out.read_text())
    assert doc['schema'] == 'vantage.rig_health/1' and set(doc['cameras']) == {'F', 'R'}
    assert doc['cameras']['F']['size_ok'] and doc['cameras']['R']['kind'] == 'synthetic'
    assert all(c['status'] in ('ok', 'warning') for c in doc['cameras'].values()), doc['cameras']
    assert doc['rig']['sync_skew_ms'] is not None and 'software clock' in doc['rig']['clock']
    assert doc['cameras']['F']['drift_checked'] and doc['cameras']['F']['drift_px'] < 12
    # a camera without a stream fails the check
    p.variant.by_tag('ML').visible = True
    p.save(proj)
    assert batch.run(ap.parse_args([proj, '--rig-health', '1', str(out)])) == 3
    assert json.loads(out.read_text())['cameras']['ML']['status'] == 'off'
    with pytest.raises(SystemExit):
        batch.run(ap.parse_args(['--sample', '--rig-health', '1', 'other.vantage']))


# ----------------------------------------------------------------------------- the page
from test_gui import app, win, pump, _wait     # noqa: E402,F401  (fixtures)


def _open(app, win):
    win.actions_by_id['live_rig'].trigger()
    pump(app, 0.2)
    pg = win.page('rig')
    assert win.page_name() == 'rig' and pg.isVisible()
    return pg


def test_live_rig_page_synthetic(app, win, tmp_path):
    from vantage.assistant import tools as T
    from vantage.core import samples
    st = win.store
    tc = win.ctx.assistant._context()
    assert T.get_rig_health(tc)['streaming'] is False
    st.edit('streams', lambda p: p.streams.update({'F': 'synthetic', 'R': 'synthetic:15'}), 'Streams')
    pg = _open(app, win)
    f, r, ml = (st.variant.by_tag(t) for t in ('F', 'R', 'ML'))
    assert pg.rows[f.uid][0].currentData() == 'synthetic' and pg.rows[r.uid][1].text() == '15'
    assert pg.rows[ml.uid][0].currentData() == 'none' and 'Stopped' in pg.summary.text()
    pg.b_start.click()
    assert pg.running and pg.b_start.text() == 'Stop'
    _wait(app, lambda: pg.monitor.preview('F') is not None and pg.monitor.preview('R') is not None, 60)
    n0 = pg.snap['cameras']['F']['frames']
    pump(app, 2.0)
    cam = pg.snap['cameras']
    assert cam['F']['frames'] > n0 + 20 and cam['F']['status'] in ('ok', 'warning')
    assert cam['F']['size'] == [1920, 1536] and cam['F']['size_ok']
    assert cam['R']['fps'] == pytest.approx(15, rel=0.2) and cam['R']['status'] == 'fault'
    assert cam['ML']['status'] == 'off' and pg.snap['rig']['not_configured'] == len(pg.tiles) - 2
    assert 'up' in pg.summary.text() and 'worst' in pg.summary.text() and 'sync skew' in pg.summary.text()
    # the planned body outline and horizon are drawn over the image, the indicator is in the statistics
    _wait(app, lambda: 'F' in pg.views and 'R' in pg.views, 60)
    assert pg.views['F'].contours and pg.views['F'].horizon
    _wait(app, lambda: pg.snap['cameras']['F'].get('drift_checked'), 30)
    assert pg.snap['cameras']['F']['drift_px'] < 12
    img = pg.tiles[f.uid].grab().toImage()
    assert img.width() > 100
    # the assistant reads the same statistics
    out = T.get_rig_health(win.ctx.assistant._context())
    assert out['streaming'] and out['cameras']['R']['status'] == 'fault' and 'sync_skew_ms' in out['rig']
    # a camera's source set on its tile is an undoable project edit and starts while running
    pg.set_kind(ml.uid, 'synthetic')
    assert st.project.streams['ML'] == 'synthetic' and 'ML' in pg.monitor.streams
    _wait(app, lambda: pg.monitor.preview('ML') is not None, 60)
    st.undo.undo()
    pump(app, 0.1)
    assert 'ML' not in st.project.streams and 'ML' not in pg.monitor.streams
    assert pg.rows[ml.uid][0].currentData() == 'none'
    pg._edit_done(r.uid, '20')
    assert st.project.streams['R'] == 'synthetic:20'
    # hiding the page stops every reader, showing it again starts them
    srcs = [s.source for s in pg.monitor.streams.values()]
    win.set_page('viewports')
    pump(app, 0.2)
    assert not pg.running and pg._resume and not pg.timer.isActive()
    _wait(app, lambda: not any(s.running for s in srcs), 10)
    win.set_page('rig')
    pump(app, 0.2)
    assert pg.running
    # a new project stops it for good
    srcs = [s.source for s in pg.monitor.streams.values()]
    st.set_project(samples.tigor_sample_project())
    pump(app, 0.2)
    assert not pg.running and not pg._resume and not pg.monitor.streams
    _wait(app, lambda: not any(s.running for s in srcs), 10)
    assert T.get_rig_health(win.ctx.assistant._context())['streaming'] is False


def test_live_rig_files_save_and_check_installation(app, win, tmp_path):
    st = win.store
    f, r, ml = (st.variant.by_tag(t) for t in ('F', 'R', 'ML'))
    vf, vr = _video(tmp_path / 'F.avi', (1920, 1536), 12), _video(tmp_path / 'R.avi', (1920, 1536), 12, seed=2)
    st.edit('streams', lambda p: p.streams.update({'F': vf, 'R': vr}), 'Streams')
    pg = _open(app, win)
    assert pg.rows[f.uid][0].currentData() == 'stream' and pg.rows[f.uid][1].text() == vf
    assert not pg.save_frames(str(tmp_path / 'none'))                  # nothing streamed yet
    pg.start()
    _wait(app, lambda: pg.monitor.preview('F') is not None and pg.monitor.preview('R') is not None, 30)
    pump(app, 1.5)
    cf = pg.snap['cameras']['F']
    assert cf['kind'] == 'file' and cf['size_ok'] and cf['fps'] == pytest.approx(30, rel=0.35)
    paths = pg.save_frames(str(tmp_path / 'out'))
    names = sorted(p.rsplit('/', 1)[1] for p in paths)
    assert names == ['F.png', 'R.png', 'rig_health.json']
    doc = json.loads((tmp_path / 'out' / 'rig_health.json').read_text())
    assert doc['schema'] == rh.SCHEMA and doc['frames'] == {'F': 'F.png', 'R': 'R.png'}
    assert doc['cameras']['F']['frames'] > 10 and doc['rig']['clock'] == rh.CLOCK_NOTE
    from vantage.core.capture import load_image
    assert load_image(str(tmp_path / 'out' / 'F.png')).shape == (1536, 1920, 3)
    # hand the frames to the installation check
    assert pg.check_installation()
    pump(app, 0.3)
    ip = win.page('install')
    assert win.page_name() == 'install' and not pg.running
    assert ip.sources[f.uid]['kind'] == 'file' and ip.images[f.uid].shape == (1536, 1920, 3)
    assert ip.sources[r.uid]['kind'] == 'file' and ml.uid not in ip.images
    assert ip.rows[f.uid][0].currentData() == 'file'
    # the check's live field uses the same stream setting, and writes it back
    assert ip.sources[f.uid]['spec'] == vf
    vm = _video(tmp_path / 'ML.avi')
    ip.sources[ml.uid]['kind'] = 'live'
    ip._edit_done(ml.uid, vm)
    assert st.project.streams['ML'] == vm and ip.live[ml.uid].spec == vm
    ip.set_source(ml.uid, 'file')                                      # stops the preview
    st.undo.undo()
    pump(app, 0.1)
    assert 'ML' not in st.project.streams and ip.sources[ml.uid]['spec'] != vm and ml.uid not in ip.live
    win.set_page('rig')
    pump(app, 0.3)
    assert pg.running                                                  # was streaming when it was left
    pg.stop()
    assert not pg.running


def test_stream_passwords_stay_out_of_the_project_file(tmp_path):
    """rtsp://user:password@host is saved as rtsp://user@host; the password lives in the user's private store."""
    import os
    import stat
    import zipfile
    from vantage.core import credentials as C
    from vantage.core import samples
    from vantage.core.capture import _open  # noqa: F401  (resolves through the store)
    from vantage.core.project import Project
    p = samples.template_project('sedan')
    p.streams = {'F': 'rtsp://admin:s3cr%t@10.0.0.5:554/stream1', 'R': '/dev/video1', 'ML': 'rtsp://cam.local/live'}
    path = str(tmp_path / 'rig.vantage')
    p.save(path)
    with zipfile.ZipFile(path) as z:
        raw = b''.join(z.read(n) for n in z.namelist() if n.endswith('.json'))
    assert b's3cr%t' not in raw and b'rtsp://admin@10.0.0.5:554/stream1' in raw
    store = os.path.join(C._dir(), C.FILE)
    assert os.environ['XDG_CONFIG_HOME'] in store                   # the test's folder, not the user's
    if os.name == 'posix':
        assert stat.S_IMODE(os.stat(store).st_mode) == 0o600
    q = Project.load(path)
    assert q.streams['F'] == 'rtsp://admin@10.0.0.5:554/stream1'
    assert q.streams['R'] == '/dev/video1' and q.streams['ML'] == 'rtsp://cam.local/live'
    assert C.resolve(q.streams['F']) == 'rtsp://admin:s3cr%t@10.0.0.5:554/stream1'
    assert C.resolve('rtsp://other@10.0.0.5:554/x') == 'rtsp://other@10.0.0.5:554/x'
    C.forget(q.streams['F'])
    assert C.resolve(q.streams['F']) == q.streams['F']


def test_stream_errors_hide_the_password(monkeypatch):
    """A stream that fails to open reports its URL without the password: the error reaches labels, the health report
    and the assistant (which sends it to Groq)."""
    from vantage.core import capture
    from vantage.core.credentials import mask
    url = 'rtsp://admin:s3cr%t@10.0.0.5:554/stream1'
    assert mask(f'Cannot open {url}: timeout') == 'Cannot open rtsp://admin:***@10.0.0.5:554/stream1: timeout'
    assert mask('rtsp://cam.local/live and http://u@h/x') == 'rtsp://cam.local/live and http://u@h/x'

    def boom(spec):
        raise RuntimeError(f'could not connect to {spec}')
    for opener in (boom, lambda spec: None):
        monkeypatch.setattr(capture, '_open', opener)
        src = capture.LiveSource(url)
        src.start()
        src._thread.join(5)
        assert src.error and 's3cr%t' not in src.error and 'admin:***@10.0.0.5' in src.error
        assert 's3cr%t' not in src._thread.name
    from types import SimpleNamespace
    from vantage.assistant import tools as T
    h = {'rig': {'up': 0, 'faults': 1}, 'cameras': {'F': {
        'status': 'fault', 'source': mask(url), 'reasons': [f'Cannot open {mask(url)}: timeout', '0.0 fps of 30']}}}
    out = T.get_rig_health(SimpleNamespace(rig_health=h, project=SimpleNamespace(streams={'F': url})))
    assert '10.0.0.5' not in json.dumps(out) and 'admin' not in json.dumps(out)
    assert out['cameras']['F']['reasons'] == ['Cannot open <stream URL> timeout', '0.0 fps of 30']
