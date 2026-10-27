"""Website clip 'carla': Simulate ▸ CARLA… on the sample's data rig, a real CARLA run with its camera images.

The page finds the CARLA 0.9.16 installation of the machine that records (where the page itself looks for one, or
VANTAGE_CARLA_ROOT), linked into the throw-away home as ~/carla/<its folder> with its client Python's venv next to it
as ~/carla/venv; the links are removed at the end. The machine needs a GPU: the cameras are rendered.

The clip: the plan (an 80 Hz step, each fisheye camera as five pinhole views remapped through its lens, the roof
lidar's 128 beams on CARLA's channels, the radar cones, the Audi A2) and the rig in CARLA table; Start, which runs the
server headless in its own systemd scope capped at 12 GB (the page's Memory cap), on the NVIDIA GPU; Run, a 4 s drive
through Town10HD at ClearNoon with 20 vehicles of traffic, seed 1, images at half the sensor size, shown as a
time-lapse; then Drive-log check on the recording: the six cameras' rotation verdicts against the lidar and the
lidar's depth edges over a CARLA fisheye image. The server is stopped by the page at the end, whatever happens. The
recording goes to the working folder in the throw-away home (shown from ~). The project is never saved.
"""
from __future__ import annotations

import os
import time

from PyQt6.QtCore import QRectF

# the scenario, saved with the project as the page's settings (Simulate ▸ CARLA's fields show it)
SCENARIO = {'town': 'Town10HD', 'weather': 'ClearNoon', 'duration_s': 4.0, 'traffic': 20, 'seed': 1,
            'image_scale': 0.5, 'memory_gb': 12.0, 'quality': 'Epic'}


def setup(p):
    p.analysis.sim = dict(SCENARIO)


def find_carla() -> str:
    """The recording machine's CARLA folder: VANTAGE_CARLA_ROOT, else the first the page would find in the real home
    folder (the clip runs with a throw-away one), else ''."""
    import pwd
    from vantage.sim import server as SV
    root = os.environ.get('VANTAGE_CARLA_ROOT', '')
    if SV.is_carla_root(root):
        return root
    home = os.environ['HOME']
    os.environ['HOME'] = pwd.getpwuid(os.getuid()).pw_dir
    try:
        roots = SV.candidate_roots()
    finally:
        os.environ['HOME'] = home
    return roots[0] if roots else ''


def link_carla(home):
    """~/carla/<the CARLA folder> and ~/carla/venv (its client Python's venv, next to it) in the throw-away home,
    linked to the real ones; returns the links made."""
    root = find_carla()
    if not root:
        return []
    made = []
    os.makedirs(os.path.join(home, 'carla'), exist_ok=True)
    for src in (root, os.path.join(os.path.dirname(root), 'venv')):
        dst = os.path.join(home, 'carla', os.path.basename(src))
        if os.path.isdir(src) and not os.path.lexists(dst):
            os.symlink(src, dst)
            made.append(dst)
    return made


def lapse(r, cond, video_s, guess_s, timeout):
    """Record while a background job runs, as a time-lapse of about video_s seconds of video when it takes guess_s of
    wall time; a longer wait speeds up further, so it stays short. The recorder's chip shows the measured speed-up."""
    t0, f0 = time.time(), r.f
    speed = max(1.0, guess_s / video_s)
    while not cond() and time.time() - t0 < timeout:
        ts = time.time()
        r.tick()
        if time.time() - t0 > guess_s:
            speed *= 1.03
        time.sleep(max(0.015, speed / r.fps - (time.time() - ts)))
        ratio = (time.time() - t0) / max(1e-6, (r.f - f0) / r.fps)
        r.timelapse = int(round(ratio)) if ratio >= 1.6 else 0
    r.timelapse = 0
    r.tick()
    return time.time() - t0


def _report_gpu(srv):
    """(printed for the record, not shown) the GPU the server renders on."""
    import subprocess
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.used', '--format=csv,noheader'],
                             capture_output=True, text=True, timeout=20).stdout.strip()
        apps = [x.strip() for x in subprocess.run(['nvidia-smi'], capture_output=True, text=True,
                                                  timeout=20).stdout.splitlines() if 'Carla' in x]
        print(f'  GPU: {out}; {apps}', flush=True)
    except (OSError, subprocess.SubprocessError) as e:
        print(f'  GPU: {e}', flush=True)


def _report_memory(srv):
    """(printed for the record, not shown) the server scope's memory: its cap and its peak."""
    import subprocess
    try:
        out = subprocess.run(['systemctl', '--user', 'show', '-p', 'MemoryMax,MemoryPeak,MemoryCurrent',
                              srv.unit + '.scope'], capture_output=True, text=True, timeout=20).stdout.split()
        print(f'  server scope: {out}', flush=True)
    except (OSError, subprocess.SubprocessError) as e:
        print(f'  server scope: {e}', flush=True)


def clip(stage, r):
    links = link_carla(stage.home)
    if not links:
        raise RuntimeError('no CARLA installation found (set VANTAGE_CARLA_ROOT)')
    held = {}
    try:
        _clip(stage, r, held)
    finally:
        pg = held.get('page')
        if pg is not None:
            if pg.server is not None:
                _report_memory(pg.server)
            pg.cancel()
            t = time.time()
            while pg.busy and time.time() - t < 60:
                stage.app.processEvents()
                time.sleep(0.05)
            pg.stop_server_now()                      # the page's own stop: SIGTERM to the server's process group
        for x in links:
            os.unlink(x)


def _clip(stage, r, held):
    from vantage.sim import server as SV
    w = stage._rig_window(setup)
    r.win = w
    r.chapter('Simulate in CARLA', 'The planned rig, driven in CARLA')
    r.menu('Simulate', 'CARLA')
    pg = held['page'] = w.page('sim')
    r.wait_for(lambda: pg.rig is not None and pg._check is not None, timeout=90)
    if pg.rig is None or not (pg._check or {}).get('ok'):
        raise RuntimeError(f'the CARLA plan or its Python is not ready: {pg._check}')
    if SV.port_open('127.0.0.1', pg.settings.port):
        raise RuntimeError('a server already answers on the port: the clip starts its own')
    dl = w.page('drivelog')
    stage.tilde_path_label(dl)
    r.hold(0.4)
    # the plan: step rate, rendering, lidar rays, ego vehicle, and the rig in CARLA (each camera's views, the lidar's
    # channels, the radar cones)
    t = r.wrect(pg.table)
    rows = pg.table.rowCount()
    h = pg.table.rowViewportPosition(rows - 1) + pg.table.rowHeight(rows - 1) + pg.table.horizontalHeader().height()
    top = r.wrect(pg.tiles['step']).y()
    r.zoom(QRectF(t.x(), top, t.width() * 0.62, t.y() + min(t.height(), h) + 30 - top), 0.7, margin=8)
    r.hold(2.0)
    # the installation, the server and the scenario; Start: headless, capped at 12 GB, on the NVIDIA GPU
    left = r.wrect(pg.root_label)
    seed = r.wrect(pg.fields['seed'][0])
    r.zoom(QRectF(left.x(), left.y() - 30, left.width(), seed.bottom() - left.y() + 36), 0.7, margin=8)
    r.hold(0.3)
    r.click(pg.b_start, dur=0.5)
    if pg.server is None:
        raise RuntimeError('the server did not start')
    _mark(r, 'Start')
    waited = lapse(r, lambda: pg.server is None or not pg.server.running or SV.port_open('127.0.0.1', pg.server.port),
                   2.4, 2.4, 240)
    if not pg.server.ready():
        raise RuntimeError(f'the server never answered ({waited:.0f} s)')
    print(f'  server answering after {waited:.1f} s', flush=True)
    _report_gpu(pg.server)
    pg._refresh_server()                              # the status line now, not at its next second
    r.hold(0.9)
    # Run: the drive, recorded as a nuScenes folder and a ROS 2 bag
    run = r.wrect(pg.b_run)
    out = r.wrect(pg.out)
    r.zoom(QRectF(run.x(), run.y() - 6, 980, out.bottom() - run.y() + 6), 0.7, margin=8)
    r.click(pg.b_run, dur=0.4)
    if not pg.busy:
        raise RuntimeError(f'the run did not start: {pg.g_status.text()}')
    _mark(r, 'Run')
    waited = lapse(r, lambda: not pg.busy, 3.4, 25.0, 900)
    res = pg.result or {}
    if res.get('status') != 'ok':
        raise RuntimeError(f'the run failed: {pg.g_status.text()}')
    print(f'  run: {waited:.1f} s; {res.get("performance")}; {res.get("counts")}', flush=True)
    _mark(r, 'report')
    r.hold(1.9)
    # Drive-log check: the recording against the plan
    r.click(pg.b_drivelog, dur=0.4)
    r.zoom(None, 0.45, wait=False)
    waited = lapse(r, lambda: dl.result is not None and not dl.busy, 1.6, 13.0, 600)
    if dl.result is None:
        raise RuntimeError(f'the drive-log check gave no result: {dl.status_text}')
    print(f'  drive-log check: {waited:.1f} s; {dl.status_text}', flush=True)
    for x in dl.result.extrinsics:
        print(f'    {x["tag"]:3s} {x["verdict"]:10s} {x.get("confidence")} delta {x.get("delta_deg")} '
              f'se {x.get("stderr_deg")}', flush=True)
    r.hold(0.5)
    # the extrinsics: F's image from CARLA with the lidar's depth edges over it (the buildings, bollards and railing
    # along its top), then every camera's verdict
    i = dl.cam_pick.findText('F')
    if i >= 0:
        dl.cam_pick.setCurrentIndex(i)
    bar = dl.tabs.tabBar()
    r.click(bar, bar.tabRect(2).center(), dur=0.5)
    r.hold(0.5)
    ir = dl.overlay.image_rect()
    o = r.wpos(dl.overlay, ir.topLeft())
    wz = max(ir.width() * 0.62, 482.0)              # at least the recorder's narrowest zoom: no widening about it
    r.zoom(QRectF(o.x() + 1, o.y() + 1, wz, wz * 9 / 16), 0.8, margin=0)
    _mark(r, 'overlay')
    r.hold(2.3)
    e = r.wrect(dl.ext)
    eh = dl.ext.horizontalHeader().height() + dl.ext.rowCount() * dl.ext.rowHeight(0)
    r.zoom(QRectF(e.x(), e.y(), dl.ext.horizontalHeader().length(), eh), 0.7, margin=12)
    _mark(r, 'verdicts')
    r.hold(2.3)
    r.zoom(None, 0.45)
    r.cursor_on = False
    r.hold(0.4)


def _mark(r, what):
    print(f'  {r.f / r.fps:5.1f} s: {what}', flush=True)
