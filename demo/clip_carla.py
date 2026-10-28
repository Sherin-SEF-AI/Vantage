"""Website clip 'carla': Simulate ▸ CARLA… opening a real CARLA drive of the sample's data rig made before the clip
(Open run…), its report, and its ground truth frame by frame in the page's Ground truth tab.

Made in two steps, so that a CARLA server and the recording never share the machine's memory:
1. the drive, recorded before the clip by Vantage's own Run without a window: `vantage PROJECT --carla-run OUT` in the
   CARLA server of the project's port (Simulate ▸ CARLA's settings as setup() sets them: Town10HD, ClearNoon, 4 s, 20
   vehicles, seed 1, images at half the sensor size, ground truth on), the server then stopped. OUT is a folder
   <variant>-<YYYYmmdd-HHMMSS> in the folder VANTAGE_CLIP_RUNS names, as the CARLA page names its runs;
2. this clip, with no server: that folder is linked into the throw-away working folder as the project's CARLA runs
   (~/Documents/Vantage/<project>/carla), where Open run…'s dialog starts.

The clip opens the drive with Open run… (the folder the dialog would be given, passed to the page's open_run: the
dialog itself is the desktop's), shows the report as after a run, then the Ground truth tab on the camera frame holding
the most labelled objects, zoomed to them: the image with the objects' tight boxes, then both boxes and a vehicle
picked in the object table and highlighted, the instances (one colour per object) laid over the image, the range on
its log colour scale, the lidar sweep from above in its nuScenes-lidarseg classes, and last the semantic labels in
CARLA's palette with the legend of the classes present (the poster). Detection quality on the drive is the
'detection' clip's: with it this one would run past 25 s.

The CARLA folder and its client venv are linked into the throw-away home as ~/carla (the page checks them; removed
at the end). While the clip records, the app can neither start a CARLA server nor probe CARLA's default ports
(2000–2010, traffic manager 8000/8100); the page's settings name port CARLA_PORT, where nothing needs to run. The
project is never saved.
"""
from __future__ import annotations

import os

from PyQt6.QtCore import QPointF, QRectF

CARLA_PORT = int(os.environ.get('VANTAGE_CLIP_CARLA_PORT', '2300'))
CARLA_TM_PORT = int(os.environ.get('VANTAGE_CLIP_CARLA_TM_PORT', str(CARLA_PORT + 6100)))

# the scenario, saved with the project as the page's settings (Simulate ▸ CARLA's fields show it)
SCENARIO = {'town': 'Town10HD', 'weather': 'ClearNoon', 'duration_s': 4.0, 'traffic': 20, 'seed': 1,
            'image_scale': 0.5, 'memory_gb': 12.0, 'quality': 'Epic', 'port': CARLA_PORT, 'tm_port': CARLA_TM_PORT,
            'ground_truth': {'semantic': True, 'instance': True, 'range': True, 'lidar_labels': True}}


def need(var, what) -> str:
    """A folder the clip needs, named by an environment variable (the recording machine's, never in the repo)."""
    v = os.environ.get(var, '')
    if not v or not os.path.isdir(v):
        raise RuntimeError(f'set {var} to {what}')
    return v


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


def link_runs(project, working):
    """The project's CARLA runs folder in the throw-away working folder, linked to the folder on disk that
    VANTAGE_CLIP_RUNS names: a drive with ground truth is too big for a memory-backed temporary folder."""
    from vantage import workdir
    src = need('VANTAGE_CLIP_RUNS', 'a folder on disk for the CARLA runs the clips record')
    dst = workdir.project_dir(project, working, 'carla')
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    os.symlink(src, dst)
    return dst


def click_tab(r, tabs, title):
    bar = tabs.tabBar()
    i = next(k for k in range(tabs.count()) if tabs.tabText(k) == title)
    r.click(bar, QPointF(bar.tabRect(i).center()), 0.5, real=False)
    tabs.setCurrentIndex(i)
    r.hold(0.2)


FORBIDDEN = set(range(2000, 2011)) | {8000, 8100}       # CARLA's defaults: possibly another user's server


def guard_ports(SV):
    """While the clip runs, the app never starts a CARLA server and never probes CARLA's default ports; returns a
    function that undoes it."""
    start, probe = SV.Server.start, SV.port_open

    def refuse(self, *a, **k):
        raise RuntimeError('the clip never starts a CARLA server: start one outside the recording')

    def port_open(host='127.0.0.1', port=2000, timeout=0.5):
        if int(port) in FORBIDDEN:
            raise RuntimeError(f'the app asked for port {port}, one of CARLA\'s defaults')
        return probe(host, port, timeout)
    SV.Server.start, SV.port_open = refuse, port_open

    def undo():
        SV.Server.start, SV.port_open = start, probe
    return undo


def finished_run(folder) -> str:
    """The newest drive with ground truth in folder (its name as the CARLA page names a run), or ''."""
    runs = sorted(d for d in (os.path.join(folder, x) for x in os.listdir(folder))
                  if os.path.isfile(os.path.join(d, 'ground_truth', 'index.json')) and
                  os.path.isfile(os.path.join(d, 'run.json')))
    return runs[-1] if runs else ''


def clip(stage, r):
    from vantage.sim import server as SV
    if CARLA_PORT in FORBIDDEN or CARLA_TM_PORT in FORBIDDEN:
        raise RuntimeError(f'refusing CARLA port {CARLA_PORT} / {CARLA_TM_PORT}: CARLA\'s default ports may be '
                           'another user\'s server')
    runs = need('VANTAGE_CLIP_RUNS', 'the folder holding the drive made by `vantage PROJECT --carla-run`')
    if not finished_run(runs):
        raise RuntimeError(f'no CARLA drive with ground truth in {runs}: make one first (see the module docstring)')
    undo = guard_ports(SV)
    links = link_carla(stage.home)
    if not links:
        undo()
        raise RuntimeError('no CARLA installation found (set VANTAGE_CARLA_ROOT)')
    held = {'links': links}
    try:
        _clip(stage, r, held)
    finally:
        for x in links:
            os.unlink(x)
        undo()


def best_frame(doc):
    """(camera tag, frame index) of the frame holding the most labelled objects of at least 250 visible pixels (a tie:
    the most such pixels)."""
    from vantage.core import groundtruth as GT
    best = (-1, -1, '', 0)
    for tag in doc.get('cameras', {}):
        for k, f in enumerate(GT.frames(doc, tag)):
            px = [o['pixels'] for o in f.get('objects', []) if o['pixels'] >= 250]
            best = max(best, (len(px), sum(px), tag, k))
    return best[2], best[3]


def slide(r, slider, to, dur=0.9):
    """Drag a slider's handle to a value, the pointer on it."""
    a = slider.value()
    g = r.wrect(slider)
    span = max(slider.maximum() - slider.minimum(), 1)

    def x_of(v):
        return g.x() + 9 + (g.width() - 18) * (v - slider.minimum()) / span
    r.move(QPointF(x_of(a), g.center().y()), 0.4)
    r.ripple()
    n = max(2, round(dur * r.fps))
    for k in range(1, n + 1):
        v = round(a + (to - a) * k / n)
        slider.setValue(v)
        r.cur = r.cur * 0 + (x_of(v), g.center().y())
        r.tick()


def objects_box(frame, min_px=250, pad=40):
    """The image rectangle (x0, y0, x1, y1) around a frame's objects of at least min_px pixels, padded."""
    bb = [o['bbox'] for o in frame['objects'] if o['pixels'] >= min_px]
    return (min(b[0] for b in bb) - pad, min(b[1] for b in bb) - pad, max(b[2] for b in bb) + pad,
            max(b[3] for b in bb) + pad)


def _clip(stage, r, held):
    from vantage.core import groundtruth as GT
    w = stage._rig_window(setup)
    r.win = w
    r.chapter('Simulate in CARLA', 'A finished drive opened: its report and its ground truth')
    r.menu('Simulate', 'CARLA')
    pg = w.page('sim')
    held['links'].append(link_runs(w.store.project, pg.working()))
    pg._show_where()
    r.wait_for(lambda: pg.rig is not None and pg._check is not None, timeout=90)
    if pg.rig is None or not (pg._check or {}).get('ok'):
        raise RuntimeError(f'the CARLA plan or its Python is not ready: {pg._check}')
    st = pg.settings
    if (st.port, st.tm_port) != (CARLA_PORT, CARLA_TM_PORT) or pg.server is not None:
        raise RuntimeError(f'the page is set to port {st.port} / {st.tm_port}, not {CARLA_PORT} / {CARLA_TM_PORT}')
    drive = finished_run(pg.runs_folder())          # through the link: shown from ~
    if not drive:
        raise RuntimeError('no finished drive in the project\'s runs folder')
    r.hold(0.2)
    # Open run…: the drive made before, in the project's runs folder
    r.click(pg.b_open, dur=0.6, real=False)
    if not pg.open_run(drive):
        raise RuntimeError(f'Open run… refused the drive: {pg.g_status.text()}')
    print(f'  opened: {pg.g_status.text()}', flush=True)
    _mark(r, 'opened')
    # its report, as after a run
    a, o = r.wrect(pg.b_drivelog), r.wrect(pg.out)
    r.zoom(QRectF(a.x(), a.y() - 8, o.width() * 0.62, 330), 0.35, margin=8)
    r.hold(1.4)
    r.zoom(None, 0.3, wait=False)
    # the ground truth, frame by frame: the frame with the most labelled objects, zoomed to them
    click_tab(r, pg.out_tabs, 'Ground truth')
    gt = pg.gt
    doc = GT.read(drive)
    tag, k = best_frame(doc)
    frame = GT.frames(doc, tag)[k]
    gt.set_camera(tag)
    gt.set_boxes('tight')
    box = objects_box(frame)
    gt.canvas.zoom_to(*box)                         # the view stays zoomed while the slider runs to the frame
    stage.settle(0.3)
    view = r.wrect(gt)
    r.zoom(QRectF(view.x(), view.y(), view.width(), view.height()), 0.4, margin=4)
    slide(r, gt.slider, k, 0.7)
    gt.show_frame()
    print(f'  frame {tag} {k}: {gt.canvas.caption}; objects >= 250 px: '
          f'{[(o["class"], o["id"], o["pixels"]) for o in frame["objects"] if o["pixels"] >= 250]}', flush=True)
    _mark(r, 'image, tight boxes')
    r.hold(0.8)
    # both boxes, and a vehicle picked in the table: highlighted
    r.click(gt.boxes.buttons['both'], dur=0.4)
    car = max((o for o in frame['objects'] if o['class'] in ('car', 'truck', 'bus')), key=lambda o: o['pixels'])
    row = next(i for i in range(gt.table.rowCount()) if gt.table.item(i, 1).text() == str(car['id']))
    gt.table.scrollToItem(gt.table.item(row, 0))
    cell = gt.table.visualItemRect(gt.table.item(row, 0))
    r.click(gt.table.viewport(), QPointF(cell.center()), 0.5)
    if gt.selected != car['id']:
        raise RuntimeError(f'the table row did not select {car["class"]} {car["id"]}')
    _mark(r, f'selected {car["class"]} {car["id"]}')
    r.hold(1.3)
    # the instances over the image (the opacity), the range, the lidar from above
    r.click(gt.layer.buttons['instance'], dur=0.45)
    _mark(r, 'instance')
    slide(r, gt.opacity, 60, 0.45)
    gt.show_frame()
    r.hold(1.1)
    gt.opacity.setValue(100)
    r.click(gt.layer.buttons['range'], dur=0.45)
    _mark(r, 'range')
    r.hold(1.5)
    r.click(gt.layer.buttons['lidar'], dur=0.45)
    _mark(r, 'lidar')
    r.hold(1.4)
    # last, the semantic labels with the legend of the classes present (the poster)
    r.click(gt.layer.buttons['semantic'], dur=0.45, hold=0.0)
    gt.canvas.zoom_to(*box)
    gt.select_object(None)
    gt.set_boxes('tight')
    _mark(r, 'semantic')
    r.cursor_on = False
    r.hold(3.0)


def _mark(r, what):
    print(f'  {r.f / r.fps:5.1f} s: {what}', flush=True)
