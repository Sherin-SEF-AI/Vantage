"""Website clip 'asbuilt': Calibrate ▸ As-built check on a recording of the sample's data rig as built.

Set-up (about 15 s, nothing filmed): a 2.5 s synthetic drive of the data rig as built is written as a ROS 2 bag by
core.dataset (images at 35 % of the sensor size, as tests/test_asbuilt.py does), with the left mirror camera ML
pitched 2° down and rolled 1° and the rear camera R turned 1.5° in yaw; the plan keeps the nominal poses. The bag goes
to ~/drives/rig_test_drive in the recording's throw-away home folder.

The clip: open the recording, Run (a time-lapse while it reads and fits), the verdict tiles, ML's real outline (blue)
against the simulated one (orange) at the plan's pose and at the fitted one, the per-camera table with the fitted
correction and its standard errors (R's from the drive-log check against the lidar: its bumper is the plain road's
grey, so its outline cannot tell), then the coverage plan against as built.
"""
from __future__ import annotations

import os
import shutil
import tempfile

from PyQt6.QtCore import QRectF

ERR = {'ML': (0.0, 2.0, 1.0), 'R': (1.5, 0.0, 0.0)}      # yaw, pitch, roll (deg) of the rig as built against the plan
NAME = 'rig_test_drive'


def make_drive(dest):
    """The recording of the rig as built: a rosbag2 folder at dest."""
    from vantage.core import dataset as DS, samples
    from vantage.core.analysis import Engine
    p = samples.tigor_sample_project()
    p.photos.source = ''
    v = next(v for v in p.variants if v.sensors)
    p.active_uid = v.uid
    for c in v.cameras:
        d = ERR.get(c.tag)
        if d:
            c.yaw, c.pitch, c.roll = c.yaw + d[0], c.pitch + d[1], c.roll + d[2]
    st = DS.DatasetSettings.from_dict({'duration_s': 2.5, 'image_scale': 0.35, 'nuscenes': False, 'rosbag': True,
                                       'keyframe_hz': 2})
    work = tempfile.mkdtemp(prefix='vantage-asbuilt-gen-')
    try:
        gen = DS.generate(p, Engine(), st, work)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.move(gen.bag_dir, dest)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def clip(stage, r):
    drives = os.path.join(stage.home, 'drives')
    bag = os.path.join(drives, NAME)
    try:
        make_drive(bag)
        w = stage._rig_window()
        r.win = w
        r.chapter('As-built check', 'The real rig against the plan, from a recording')
        r.menu('Calibrate', 'As-built check')
        pg = w.page('asbuilt')
        stage.tilde_path_label(pg)
        r.hold(0.3)
        r.click(pg.b_dir, real=False, dur=0.5)        # the folder dialog cannot be filmed: open the bag directly
        pg.open_path(bag)
        r.hold(0.7)
        r.click(pg.b_run, dur=0.5)
        r.wait_for(lambda: pg.result is not None and not pg.busy, timeout=300, speed=3)
        r.hold(0.8)
        tiles = r.wrect(pg.tiles['verdict']).united(r.wrect(pg.tiles['blind']))
        r.zoom(tiles, 0.7, margin=16)
        r.hold(1.3)
        r.zoom(None, 0.5)
        # ML: the real outline (blue) against the simulated one (orange) at the plan's pose, then at the fitted one
        i = pg.cam_pick.findData('ML')
        if i >= 0:
            pg.cam_pick.setCurrentIndex(i)
        r.hold(0.3)
        ir = pg.view.image_rect()
        tl = r.wpos(pg.view, ir.topLeft())
        r.zoom(QRectF(tl.x() + ir.width() * 0.45, tl.y(), ir.width() * 0.55, ir.height()), 0.8, margin=6)
        r.hold(1.4)
        r.click(pg.mode.buttons['fitted'], dur=0.4)
        r.hold(1.4)
        r.zoom(None, 0.5)
        # the table: the fitted corrections with their standard errors, R's from the lidar
        t = pg.cams
        g = r.wrect(t)
        last = t.columnViewportPosition(10) + t.columnWidth(10)
        rows = QRectF(g.x(), g.y(), last + 8, t.horizontalHeader().height() + t.rowCount() * t.rowHeight(0) + 4)
        r.zoom(rows, 0.7, margin=10)
        r.hold(2.0)
        r.zoom(None, 0.5)
        # the coverage: plan against as built
        bar = pg.tabs.tabBar()
        r.click(bar, bar.tabRect(1).center(), dur=0.5)
        r.hold(0.6)
        cov = r.wrect(pg.cov_line).united(r.wrect(pg.vis))
        r.zoom(cov, 0.7, margin=12)
        r.hold(2.0)
        r.zoom(None, 0.5)
        r.hold(0.6)
    finally:
        shutil.rmtree(drives, ignore_errors=True)
