"""Website clip 'detection': Simulate ▸ Detection quality… comparing two rigs on drives recorded in CARLA 0.9.16.

The same pedestrian-crossing scene (seed 1, 20 vehicles, 12 s, Town10HD) recorded with ground truth by the sample's
data rig and by the same rig with every camera pitched 20° further up. YOLOX-S runs through OpenCV on every fisheye
image of both drives; its detections were cached in each recording when the drives were first evaluated, so Run
matches them to the ground truth in seconds. The tiles (vehicles, every camera), the rig recall against distance of
both rigs with the paired differences (+12.4 points at 20–40 m, −6.8 beyond 40 m), then one frame of the second rig
with its labelled objects and the detections drawn over it.

The model is the recording machine's YOLOX-S (the folder VANTAGE_CLIP_MODELS names), linked into the throw-away
home's data folder and checked by its SHA-256 as the page always does; nothing is downloaded. The drives are working
copies whose sensor files link to the recordings (the folder VANTAGE_CLIP_DETECT names), linked into
~/Documents/Vantage/<project>/carla/. The links are
removed at the end. The project is never saved.
"""
from __future__ import annotations

import os

from PyQt6.QtCore import QPointF, QRectF

RUNS = ('data-rig-6-cam-lidar-radars', 'data-rig-cameras-20-up')


def need(var, what) -> str:
    """A folder the clip needs, named by an environment variable (the recording machine's, never in the repo)."""
    v = os.environ.get(var, '')
    if not v or not os.path.isdir(v):
        raise RuntimeError(f'set {var} to {what}')
    return v


def link_data(home) -> tuple[list, list]:
    """The two drives in the throw-away working folder and the models folder in its data folder (links)."""
    data = need('VANTAGE_CLIP_DETECT', 'the folder holding the two drives ' + ' and '.join(RUNS))
    models = need('VANTAGE_CLIP_MODELS', 'the folder holding yolox_s.onnx')
    runs_dir = os.path.join(home, 'Documents', 'Vantage', 'tata-tigor-ev', 'carla')
    os.makedirs(runs_dir, exist_ok=True)
    made, runs = [], []
    for name in RUNS:
        dst = os.path.join(runs_dir, name)
        os.symlink(os.path.join(data, name), dst)
        made.append(dst)
        runs.append(dst)
    mdir = os.path.join(home, '.local', 'share', 'Vantage')
    os.makedirs(mdir, exist_ok=True)
    os.symlink(models, os.path.join(mdir, 'models'))
    made.append(os.path.join(mdir, 'models'))
    return made, runs


def click_tab(r, tabs, title):
    bar = tabs.tabBar()
    i = next(k for k in range(tabs.count()) if tabs.tabText(k) == title)
    r.click(bar, QPointF(bar.tabRect(i).center()), 0.5, real=False)
    tabs.setCurrentIndex(i)
    r.hold(0.2)


def clip(stage, r):
    made, runs = link_data(stage.home)
    old = os.environ.get('VANTAGE_MODEL_DIR')
    os.environ['VANTAGE_MODEL_DIR'] = os.path.join(stage.home, '.local', 'share', 'Vantage', 'models')
    try:
        _clip(stage, r, runs)
    finally:
        if old is None:
            os.environ.pop('VANTAGE_MODEL_DIR', None)
        else:
            os.environ['VANTAGE_MODEL_DIR'] = old
        for x in made:
            os.unlink(x)


def _clip(stage, r, runs):
    from vantage.core import detection_eval as DE

    def setup(p):
        p.analysis.detection = DE.DetectionSettings.from_dict({'recordings': list(runs)}).to_dict()
    w = stage._rig_window(setup)
    r.win = w
    r.chapter('Detection quality', 'Will a detector find what each rig records?')
    r.menu('Simulate', 'Detection quality')
    pg = w.page('detection')
    stage.settle(2)
    r.hold(0.3)
    # the model (checked by its SHA-256) and the two drives, the first the reference
    m, rec = r.wrect(pg.model_status), r.wrect(pg.recs)
    top = min(m.y(), rec.y()) - 64
    r.zoom(QRectF(m.x(), top, max(m.width(), rec.width()), rec.bottom() - top + 8), 0.7, margin=10)
    r.hold(1.8)
    r.zoom(None, 0.5)
    r.click(pg.b_run, dur=0.6)
    if not pg.busy:
        raise RuntimeError(f'the run did not start: {pg.status_text if hasattr(pg, "status_text") else ""}')
    r.wait_for(lambda: not pg.busy and pg.result is not None, timeout=900, speed=3)
    res = pg.result
    if res is None:
        raise RuntimeError('no result')
    for c in res.comparisons:
        if c['class'] == 'vehicle' and c['mode'] == 'rig':
            print(f'  rig {c["range_bin"]}: {c.get("recall_reference")} -> {c.get("recall")} '
                  f'd {c.get("recall_difference")} {c.get("recall_difference_ci")}', flush=True)
    r.hold(0.3)
    # the tiles: vehicles over every camera of the reference rig
    a = r.wrect(pg.tiles['objects'])
    b = r.wrect(pg.tiles['rig'])
    r.zoom(QRectF(a.x(), a.y() - 40, b.right() - a.x(), b.bottom() - a.y() + 44), 0.6, margin=8)
    r.hold(1.6)
    # rig against rig: recall against distance and the paired differences
    click_tab(r, pg.tabs, 'Compare')
    c = r.wrect(pg.rig_chart)
    t = r.wrect(pg.cmp_table)
    r.zoom(QRectF(c.x(), c.y(), c.width(), t.y() + t.height() * 0.62 - c.y()), 0.7, margin=6)
    r.hold(3.2)
    # a frame of the second rig (cameras 20° up): labelled objects (found green, missed orange) and the detections
    click_tab(r, pg.tabs, 'Frames')
    if pg.rec_pick.count() > 1:
        pg.rec_pick.setCurrentIndex(1)
    i = pg.cam_pick.findData('F')
    if i >= 0:
        pg.cam_pick.setCurrentIndex(i)
    pg.show_frame(best=True)
    r.hold(0.3)
    f = r.wrect(pg.frame_view)
    r.zoom(f, 0.7, margin=4)
    r.hold(2.6)
    r.zoom(None, 0.5)
    r.cursor_on = False
    r.hold(0.4)
