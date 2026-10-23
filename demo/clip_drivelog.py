"""Website clip: Calibrate ▸ Drive-log check on a recorded drive of the sample's data rig.

Set-up (a temp folder, about 20 s): a short synthetic drive of the data rig as built is written as a ROS 2 bag by
core.dataset, with the front camera F mounted 0.6° off its planned yaw (F, not a camera at either end of the trigger
line's spread, so the line's planned trigger stays as it is); then stream faults are put into the bag: MR loses every
tenth frame, R's stamps run 7 ms late (its trigger fires late), QR's topic is missing. The images are rendered at a
quarter of the sensor size for speed and scaled back up, so their size matches the camera module.

The clip: open the bag on the page, run, the streams (MR's drops on the timeline, QR missing), the sync table (R 7 ms
late against the plan), then the overlay of F's lidar points at the plan's rotation and at the correction found.
"""
from __future__ import annotations

import os
import shutil
import tempfile

import numpy as np
from PyQt6.QtCore import QRectF

YAW_CAM, YAW_OFF = 'F', 0.6     # degrees: F as built against the plan
LATE_MS = 7.0           # R's trigger
DROP_EVERY = 10         # MR loses every tenth frame


def make_bag(work) -> str:
    """The faulty drive as a rosbag2 folder in `work`; returns its path."""
    import cv2
    from vantage.core import dataset as DS, rosbag as RB, samples
    from vantage.core.analysis import Engine
    p = samples.tigor_sample_project()
    p.photos.source = ''
    v = next(v for v in p.variants if v.sensors)
    p.active_uid = v.uid
    v.by_tag(YAW_CAM).yaw += YAW_OFF                    # as built: not as planned
    st = DS.DatasetSettings.from_dict({'duration_s': 1.5, 'image_scale': 0.25, 'camera_rate': 'module',
                                       'nuscenes': False, 'jpeg_quality': 90, 'seed': 3})
    gen = os.path.join(work, 'generated')
    DS.generate(p, Engine(), st, gen)
    src = next(os.path.join(r, d) for r, ds, _ in os.walk(gen) for d in ds
               if os.path.isfile(os.path.join(r, d, 'metadata.yaml')))
    out = os.path.join(work, 'data_rig_drive')
    W, H = p.module_for(v.cameras[0]).width_px, p.module_for(v.cameras[0]).height_px
    rd = RB.BagReader(src)
    seen = {}
    with RB.BagWriter(out, compression='none') as bw:
        for t, ty in rd.topics.items():
            if not t.startswith('/qr/'):
                bw.add_topic(t, ty)
        for topic, t_ns, raw in rd.messages(decode=False):
            if topic.startswith('/qr/'):
                continue                                       # QR's stream never reached the recorder
            if not topic.endswith('/image_raw/compressed'):
                bw.write(topic, t_ns, raw)
                continue
            k = seen[topic] = seen.get(topic, -1) + 1
            if topic.startswith('/mr/') and k % DROP_EVERY == DROP_EVERY // 2:
                continue
            msg = rd.decode(topic, raw)
            img = cv2.imdecode(np.frombuffer(msg['data'], np.uint8), cv2.IMREAD_COLOR)
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
            msg['data'] = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
            if topic.startswith('/r/'):
                s = RB.stamp_ns(msg['header']['stamp']) + int(LATE_MS * 1e6)
                msg['header']['stamp'] = RB.stamp(s)
                t_ns += int(LATE_MS * 1e6)
            bw.write(topic, t_ns, msg)
    rd.close()
    shutil.rmtree(gen, ignore_errors=True)
    return out


def clip(stage, r):
    work = tempfile.mkdtemp(prefix='vantage-drivelog-clip-')
    try:
        bag = make_bag(work)

        def setup(p):
            p.analysis.drivelog = {'frames_per_camera': 8}
        w = stage._rig_window(setup)
        r.win = w
        r.chapter('Drive log', 'A recorded drive checked against the plan')
        r.menu('Calibrate', 'Drive-log check')
        pg = w.page('drivelog')
        r.hold(0.3)
        r.click(pg.b_dir, real=False, dur=0.5)        # the folder dialog cannot be filmed: open the bag directly
        pg.open_path(bag)
        r.hold(0.6)
        r.click(pg.b_run, dur=0.5)
        r.wait_for(lambda: pg.result is not None and not pg.busy, timeout=240, speed=3)
        r.hold(1.0)
        # the streams: MR's gaps on the timeline, QR missing
        tl = r.wrect(pg.timeline)
        r.zoom(QRectF(tl.x(), tl.y(), tl.width() * 0.75, tl.height()), 0.7, margin=24)
        r.hold(1.4)
        r.zoom(None, 0.5)
        # the sync table: R's trigger 7 ms late against the plan
        bar = pg.tabs.tabBar()
        r.click(bar, bar.tabRect(1).center(), dur=0.5)
        r.hold(0.2)
        t = pg.sync
        rows = QRectF(r.wrect(t).x(), r.wrect(pg.lidar_line).y(), t.horizontalHeader().length(),
                      r.wrect(t).y() - r.wrect(pg.lidar_line).y() + t.horizontalHeader().height()
                      + t.rowCount() * t.rowHeight(0))
        r.zoom(rows, 0.7, margin=16)
        r.hold(1.8)
        r.zoom(None, 0.5)
        # the overlay: F's lidar points at the plan's rotation, then at the correction found
        pg.cam_pick.setCurrentIndex(pg.cam_pick.findText(YAW_CAM))
        r.click(bar, bar.tabRect(2).center(), dur=0.5)
        r.hold(0.5)
        ir = pg.overlay.image_rect()                   # the buildings along the top of the fisheye image
        tl = r.wpos(pg.overlay, ir.topLeft())
        wz = max(ir.width() * 0.62, 482.0)           # at least the recorder's narrowest zoom: no widening about it
        r.zoom(QRectF(tl.x() + 1, tl.y() + 1, wz, wz * 9 / 16), 0.8, margin=0)
        r.hold(1.2)
        r.click(pg.mode.buttons['corrected'], dur=0.4)
        r.hold(1.3)
        r.click(pg.mode.buttons['plan'], dur=0.3)
        r.hold(0.8)
        r.click(pg.mode.buttons['corrected'], dur=0.3)
        r.hold(1.0)
        r.zoom(None, 0.5)
        r.hold(1.6)
    finally:
        shutil.rmtree(work, ignore_errors=True)
