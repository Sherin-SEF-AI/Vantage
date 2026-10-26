"""35 Sample dataset and ROS 2 bags: Export ▸ Sample dataset on the data rig with two GNSS antennas and an INS
(stage._nav_window): the preview at two instants (the six camera images, the lidar's top view with the ground-truth
boxes), the outputs, the estimate at the defaults, the drive shortened to 2 s at a quarter of the image size on the
form, Generate into the working folder (~/Documents/Vantage in the recording's throw-away home; a marked time-lapse),
the summary with the devkit snippet (paths from ~). The written dataset is removed afterwards.

Figures read from the page while recording (r.expect), the same as core.dataset.estimate gives in a light run: at the
defaults (8 s, half-size images) 281 MB in about 86 s ('approximate, this machine', the page says); 2 s at 25 %:
54.6 MB. The written size and time come from the summary the page shows after the run.
"""

META = {
    'id': 'dataset',
    'part': 5,
    'title': 'Sample dataset and ROS 2 bags',
    'promise': 'A drive recorded by exactly this rig, before the hardware exists.',
    'menu': 'Export ▸ Sample dataset',
}

LINES = [
    ("Long before the rig is built, the data pipeline needs recordings to test with. The sample dataset is a short "
     "synthetic drive, recorded by exactly this rig: six cameras, the lidar, five radars, two GNSS antennas and an INS.",
     "Long before the rig is built, the data pipeline needs recordings to test with. The sample dataset is a short "
     "synthetic drive, recorded by exactly this rig: six cameras, the lidar, five radars, two G N S S antennas, and an "
     "I N S."),
    ("A seeded street with a bend: lanes, kerbs, buildings, trees, parked cars, traffic, cyclists and pedestrians. "
     "Scrub to any instant to see the camera images, and the lidar from above with the ground-truth boxes.",
     "A seeded street with a bend: lanes, kerbs, buildings, trees, parked cars, traffic, cyclists, and pedestrians. "
     "Scrub to any instant to see the camera images, and the lidar from above, with the ground truth boxes."),
    ("The cameras render through their real lens models at the sync plan's trigger times, with the body and rig "
     "hiding what lies behind them; the lidar fires column by column from wherever the vehicle is at that instant.",
     "The cameras render through their real lens models at the sync plan's trigger times, with the body and rig "
     "hiding what lies behind them. The lidar fires column by column, from wherever the vehicle is at that instant."),
    ("It is written as a nuScenes folder the devkit loads, and as a ROS 2 bag. Vantage writes and reads the bags "
     "itself, with no ROS installation.",
     "It is written as a new scenes folder the dev kit loads, and as a ross two bag. Vantage writes and reads the bags "
     "itself, with no ross installation."),
    ("The page estimates the cost first: at the defaults, an eight-second drive with half-size images, about 280 "
     "megabytes and a minute and a half here.",
     "The page estimates the cost first: at the defaults, an eight second drive with half size images, about two "
     "hundred and eighty megabytes, and a minute and a half, here."),
    ("For this video, two seconds at a quarter of the image size: about 55 megabytes.",
     "For this video, two seconds, at a quarter of the image size: about fifty-five megabytes."),
    ("Generate writes into a new folder of the working folder, and never overwrites one. It runs in the background, "
     "with Cancel.",
     "Generate writes into a new folder of the working folder, and never overwrites one. It runs in the background, "
     "with cancel."),
    ("The summary lists what was written, and the snippet that loads it with the nuScenes devkit.",
     "The summary lists what was written, and the snippet that loads it with the new scenes dev kit."),
    ("Synthetic geometry and one sun, no weather: a dataset for testing a pipeline, not for training a model.",
     "Synthetic geometry and one sun, no weather: a dataset for testing a pipeline, not for training a model."),
]


def scene(stage, r):
    import os
    import shutil
    from PyQt6.QtCore import QPointF, QRectF
    from PyQt6.QtTest import QTest
    from PyQt6.QtCore import Qt
    import clip_dataset as CDS
    import helpers_tourc as H

    w = stage._nav_window()
    r.win = w
    st = w.store
    stage.tilde(w.msg_label)
    vp = stage.vp
    out = os.path.join(stage.home, 'Documents', 'Vantage')

    # 0 - open the page
    r.lower_third()
    r.say(0)
    r.cursor_on = False
    stage.orbit(r, vp, 26, 3.6, d_el=4)
    r.cursor_on = True
    r.until_word(0, 'The sample dataset')
    r.menu('Export', 'Sample dataset')
    pg = w.page('dataset')
    stage.tilde(pg.out)
    r.wait_for(lambda: pg.preview_data is not None and not pg._pbusy, timeout=120)
    r.crossfade(0.4)
    r.finish_speech(0.2)

    # 1 - the scene; scrub
    r.say(1)
    CDS._scrub_to(r, pg, 30)
    r.hold(0.3)
    top = r.wrect(pg.top)
    r.zoom(QRectF(top.x(), top.y(), top.width(), top.height() * 0.95), 1.0, margin=20)
    r.hold(max(1.5, r.speech_left() - 4.0))
    r.zoom(None, 0.8)
    CDS._scrub_to(r, pg, 65)                         # into the bend
    r.finish_speech(0.2)

    # 2 - cameras and lidar
    r.say(2)
    r.cursor_on = False
    r.zoom(r.wrect(pg.grid), 1.0, margin=12)
    r.hold(max(1.5, r.speech_left() - 3.8))
    r.zoom(r.wrect(pg.top), 1.0, margin=12)
    r.finish_speech(0.2)
    r.zoom(None, 0.8)

    # 3 - the outputs: nuScenes and a ROS 2 bag
    r.say(3)
    r.cursor_on = True
    tg = H.union(r.wrect(pg.toggles['nuscenes']), r.wrect(pg.toggles['rosbag']))
    r.zoom(QRectF(tg.x() - 40, tg.y() - 120, 520, 300), 1.0)
    r.callout(lambda: H.union(r.wrect(pg.toggles['nuscenes']), r.wrect(pg.toggles['rosbag'])), dur=max(2.4, r.speech_left()),
              pad=4)
    r.finish_speech(0.2)
    r.zoom(None, 0.8)

    # 4 - the estimate at the defaults
    T = pg.tiles
    print('estimate at the defaults:', {k: (t.value, t.unit, t.note) for k, t in T.items()}, flush=True)
    r.expect((T['size'].value, T['size'].unit), ('281.2', 'MB'), 'size at the defaults')
    r.expect((T['time'].value, T['time'].unit), ('~86', 's'), 'time at the defaults')
    r.say(4)
    t0, t3 = r.wrect(T['frames']), r.wrect(T['time'])
    r.zoom(QRectF(t0.x() - 10, t0.y() - 20, t3.right() - t0.x() + 20, t0.height() + 40), 1.0)
    r.callout(T['size'], dur=2.2, pad=3, dim=0.15)
    r.callout(T['time'], dur=max(2.0, r.speech_left()), pad=3, dim=0.0)
    r.finish_speech(0.2)
    r.zoom(None, 0.8)

    # 5 - 2 s at 25 %: the estimate follows
    r.say(5)
    f_dur, _ = pg.fields['duration_s']
    f_img, _ = pg.fields['image_scale']
    fr = H.union(r.wrect(f_dur), r.wrect(f_img))
    r.zoom(QRectF(fr.x() - 160, fr.y() - 60, 560, fr.height() + 160), 0.9)
    r.enter(f_dur, '2')
    r.enter(f_img, '25')
    stage.settle(1)
    s = pg.settings
    r.expect((s.duration_s, s.image_scale), (2.0, 0.25), 'the shorter drive')
    r.zoom(None, 0.8)
    print('estimate now:', {k: (t.value, t.unit, t.note) for k, t in T.items()}, flush=True)
    r.expect((T['size'].value, T['size'].unit), ('54.6', 'MB'), 'size now')
    r.figure('Size', float(T['size'].value), T['size'].unit, 1, dur=2.8, src=T['size'], start=281.2)
    r.finish_speech(0.4)

    # 6 - Generate: a time-lapse
    r.say(6)
    r.click(pg.b_generate, real=False)
    pg.b_generate.click()
    r.move(r.wpos(pg.top), 0.6, wait=False)
    r.cursor_on = False
    H.wait_job(r, lambda: not pg.busy, timeout=600, fast=8.0)
    stage.settle(1)
    r.expect(pg.result is not None, True, 'the dataset written')
    print('summary:', pg.out.toPlainText()[:600], flush=True)
    r.finish_speech(0.2)

    # 7 - the summary and the snippet
    r.say(7)
    box = r.wrect(pg.out)
    r.zoom(QRectF(box.x(), box.y() - 26, 980, 300), 1.0, margin=12)
    r.hold(max(1.6, r.speech_left() - 0.2))
    r.finish_speech(0.2)

    # 8 - limits
    r.say(8)
    r.zoom(None, 0.8)
    lr = r.wrect(pg.limits)
    r.zoom(QRectF(lr.x() - 20, lr.y() - 40, lr.width() + 260, lr.height() + 80), 1.0)
    r.callout(pg.limits, dur=max(2.0, r.speech_left()), pad=4, dim=0.25)
    r.finish_speech(0.6)
    r.zoom(None, 0.8)
    r.hold(0.4)
    shutil.rmtree(out, ignore_errors=True)
