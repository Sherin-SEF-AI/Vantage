"""32 Drive-log check: Calibrate ▸ Drive-log check on a recorded drive of the sample's data rig. The drive is
demo/clip_drivelog.make_bag's (not filmed, about 20 s): a 1.5 s synthetic drive written as a ROS 2 bag by the sample
dataset, with faults put in on purpose (MR loses every tenth frame, R's stamps 7 ms late, QR's topic missing) and F
mounted 0.6 deg off in yaw; it goes to ~/drives/data_rig_drive in the recording's throw-away home (shown from ~).

Figures read from the page while recording (r.expect), the same as core.drivelog.run gives in a light run: 10 of 12
streams healthy; 4 dropped frames, the worst MR (9.1 %); QR has no topic; worst sync offset 7.0 ms (R fires 7.0 ms
late, after F on line A); L1 phase-locked at 0.0 deg; worst camera rotation 0.53 deg (F: yaw +0.53, high confidence,
8 frames); the bag's injected yaw is 0.6 deg.
"""

META = {
    'id': 'drivelog',
    'part': 4,
    'title': 'Drive-log check',
    'promise': 'A recorded drive against the plan: streams, data, sync and rotations.',
    'menu': 'Calibrate ▸ Drive-log check',
}

LINES = [
    ("After a drive, the drive-log check reads the recording, a ROS 2 bag or a nuScenes folder, and checks it against "
     "the plan: the streams, the data, the sync, and where the cameras point.",
     "After a drive, the drive log check reads the recording, a ross two bag, or a new scenes folder, and checks it "
     "against the plan: the streams, the data, the sync, and where the cameras point."),
    ("This is a short synthetic drive of the data rig with faults put in on purpose: a camera that drops frames, one "
     "that fires late, one that never reached the recorder, and the front camera turned 0.6 degrees.",
     "This is a short synthetic drive of the data rig, with faults put in on purpose: a camera that drops frames, one "
     "that fires late, one that never reached the recorder, and the front camera turned zero point six degrees."),
    ("Topics are matched to the rig's sensors by the tag in their name or frame; where the names differ, the match "
     "is set in the table.",
     "Topics are matched to the rig's sensors by the tag in their name or frame. Where the names differ, the match is "
     "set in the table."),
    ("Run reads the log once, in the background and a chunk at a time, so long drives are fine.",
     "Run reads the log once, in the background, and a chunk at a time, so long drives are fine."),
    ("The tiles sum it up: ten of twelve streams healthy, four dropped frames, a camera 7 milliseconds out of sync, "
     "and a camera turned half a degree.",
     "The tiles sum it up: ten of twelve streams healthy, four dropped frames, a camera seven milliseconds out of "
     "sync, and a camera turned half a degree."),
    ("Streams: each sensor's rate against the plan, drops and the longest gap, jitter, latency and image size. The "
     "right quarter camera has no topic at all, and the timeline shows the right mirror camera's gaps.",
     "Streams: each sensor's rate against the plan, drops and the longest gap, jitter, latency, and image size. The "
     "right quarter camera has no topic at all, and the timeline shows the right mirror camera's gaps."),
    ("Sync: the lidar's measured phase lock against its plan, and each camera's trigger against the moment the sweep "
     "crosses its view. The rear camera fires 7 milliseconds late.",
     "Sync: the lidar's measured phase lock against its plan, and each camera's trigger, against the moment the sweep "
     "crosses its view. The rear camera fires seven milliseconds late."),
    ("Rotations: the lidar's depth edges are projected into each camera and lined up with the image's edges. Through "
     "the plan, the front camera's points sit off the buildings' edges; with the correction found, they line up.",
     "Rotations: the lidar's depth edges are projected into each camera, and lined up with the image's edges. "
     "Through the plan, the front camera's points sit off the buildings' edges. With the correction found, they line "
     "up."),
    ("It finds the front camera turned 0.53 degrees in yaw, with high confidence; the drive was made with 0.6. "
     "Translation cannot be measured this way, and an axis the edges do not pin down is left as planned.",
     "It finds the front camera turned zero point five three degrees in yaw, with high confidence; the drive was made "
     "with zero point six. Translation cannot be measured this way, and an axis the edges do not pin down is left as "
     "planned."),
    ("The findings say it all in words. Apply rotation corrections makes a corrected variant in one undo step, and "
     "Export writes the check as JSON, CSV and images.",
     "The findings say it all in words. Apply rotation corrections makes a corrected variant, in one undo step, and "
     "export writes the check as Jason, C S V, and images."),
]


def scene(stage, r):
    import os
    import shutil
    import tempfile
    from PyQt6.QtCore import QPointF, QRectF
    import clip_drivelog as CD
    import helpers_tourc as H

    work = tempfile.mkdtemp(prefix='vantage-drivelog-tour-')
    bag = CD.make_bag(work, os.path.join(stage.home, 'drives', 'data_rig_drive'))
    shutil.rmtree(work, ignore_errors=True)

    def setup(p):
        p.analysis.drivelog = {'frames_per_camera': 8}
    w = stage._rig_window(setup)
    r.win = w
    st = w.store
    stage.tilde(w.msg_label)
    vp = stage.vp

    # 0 - open the page
    r.lower_third()
    r.say(0)
    r.cursor_on = False
    stage.orbit(r, vp, 20, 3.4, d_el=4)
    r.cursor_on = True
    r.until_word(0, 'and checks it against')
    r.menu('Calibrate', 'Drive-log check')
    pg = w.page('drivelog')
    stage.tilde_path_label(pg)
    stage.settle(2)
    r.crossfade(0.4)
    r.finish_speech(0.2)

    # 1 - open the faulty drive
    r.say(1)
    r.click(pg.b_dir, real=False, dur=0.7)            # the folder dialog cannot be filmed: open the bag directly
    pg.open_path(bag)
    stage.settle(2)
    r.expect(pg.path_label.text(), '~/drives/data_rig_drive', 'the log shown')
    pl = r.wrect(pg.path_label)
    r.zoom(QRectF(pl.x() - 20, pl.y() - 60, 520, 260), 0.9)
    r.callout(pg.path_label, dur=max(2.4, r.speech_left() - 1.0), pad=4)
    r.finish_speech(0.2)

    # 2 - topics matched by tag
    r.say(2)
    tt = r.wrect(pg.topics)
    r.zoom(QRectF(tt.x() - 10, tt.y() - 40, tt.width() + 20, tt.height() + 50), 1.0)
    r.hold(max(1.5, r.speech_left() - 0.4))
    r.finish_speech(0.2)
    r.zoom(None, 0.8)

    # 3 - Run (real time)
    r.say(3)
    r.click(pg.b_run, real=False)
    pg.b_run.click()
    r.move(r.wpos(pg.tabs), 0.8)
    r.cursor_on = False
    H.wait_job(r, lambda: pg.result is not None and not pg.busy, timeout=600, fast=6.0)
    stage.settle(1)
    print('status:', pg.status.text(), flush=True)
    r.finish_speech(0.2)

    # 4 - the tiles
    T = pg.tiles
    vals = {k: (t.value, t.unit, t.note) for k, t in T.items()}
    print('tiles:', vals, flush=True)
    r.expect((T['streams'].value, T['streams'].unit), ('10', 'of 12'), 'streams tile')
    r.expect(T['drops'].value, '4', 'drops tile')
    r.expect((T['sync'].value, T['sync'].note.split(' ')[0]), ('7.0', 'R'), 'sync tile')
    r.expect((T['rotation'].value, T['rotation'].note.split(' ')[0]), ('0.53', 'F'), 'rotation tile')
    r.say(4)
    t0, t3 = r.wrect(T['streams']), r.wrect(T['rotation'])
    r.zoom(QRectF(t0.x() - 10, t0.y() - 20, t3.right() - t0.x() + 20, t0.height() + 40), 1.0)
    for k, word in (('streams', 'ten of twelve'), ('drops', 'four dropped'), ('sync', 'seven milliseconds'),
                    ('rotation', 'a camera turned')):
        r.until_word(4, word)
        r.callout(T[k], dur=1.8, pad=3, dim=0.15)
    r.finish_speech(0.4)
    r.zoom(None, 0.8)

    # 5 - streams: QR missing, MR's gaps on the timeline
    r.say(5)
    sr = r.wrect(pg.streams)
    r.zoom(QRectF(sr.x() - 10, sr.y() - 40, sr.width() + 20, sr.height() + 50), 1.0)
    qr = next(i for i in range(pg.streams.rowCount()) if pg.streams.item(i, 0).text() == 'QR')
    r.until_word(5, 'The right quarter camera')
    r.callout(H.cell(r, pg.streams, qr), dur=2.0, pad=2, dim=0.2)
    r.until_word(5, 'and the timeline')
    tl = r.wrect(pg.timeline)
    r.zoom(QRectF(tl.x(), tl.y(), tl.width() * 0.75, tl.height()), 0.9, margin=24)
    r.finish_speech(0.6)
    r.zoom(None, 0.8)

    # 6 - sync: R 7 ms late
    bar = pg.tabs.tabBar()
    r.say(6)
    r.cursor_on = True
    r.click(bar, bar.tabRect(1).center(), dur=0.6)
    stage.settle(1)
    sy = pg.sync
    rows = QRectF(r.wrect(sy).x(), r.wrect(pg.lidar_line).y(), sy.horizontalHeader().length(),
                  r.wrect(sy).y() - r.wrect(pg.lidar_line).y() + sy.horizontalHeader().height()
                  + sy.rowCount() * sy.rowHeight(0))
    r.zoom(rows, 1.0, margin=16)
    r.callout(pg.lidar_line, dur=2.4, pad=4, dim=0.15)
    ri = next(i for i in range(sy.rowCount()) if sy.item(i, 0).text() == 'R')
    r.expect(sy.item(ri, 5).text(), '7.00', "R's own trigger error")
    r.until_word(6, 'The rear camera fires')
    r.callout(H.cell(r, sy, ri), dur=max(2.0, r.speech_left() + 0.3), pad=2, dim=0.2)
    r.finish_speech(0.3)
    r.zoom(None, 0.8)

    # 7 - rotations: F's lidar points through the plan, then corrected
    r.say(7)
    pg.cam_pick.setCurrentIndex(pg.cam_pick.findText('F'))
    r.click(bar, bar.tabRect(2).center(), dur=0.6)
    stage.settle(1)
    ir = pg.overlay.image_rect()
    tl = r.wpos(pg.overlay, ir.topLeft())
    zw = min(ir.width(), 740.0)                   # the deepest zoom: the buildings along the top of the image
    r.zoom(QRectF(tl.x() + (ir.width() - zw) / 2, tl.y() + 2, zw, zw * 9 / 16), 1.0, margin=0)
    r.until_word(7, 'With the correction found')
    r.click(pg.mode.buttons['corrected'], dur=0.5)
    r.hold(max(1.2, r.speech_left() - 1.6))
    r.click(pg.mode.buttons['plan'], dur=0.3)
    r.hold(0.6)
    r.click(pg.mode.buttons['corrected'], dur=0.3)
    r.finish_speech(0.1)

    # 8 - the extrinsics table
    ex = pg.ext
    fi = next(i for i in range(ex.rowCount()) if ex.item(i, 0).text() == 'F')
    r.expect((ex.item(fi, 1).text(), ex.item(fi, 7).text()), ('+0.53', 'High'), "F's correction")
    r.say(8)
    r.zoom(None, 0.6)
    er = r.wrect(ex)
    r.zoom(QRectF(er.x() - 10, er.y() - 20, er.width() + 20, er.height() + 70), 1.0)
    r.callout(H.cell(r, ex, fi), dur=2.6, pad=2, dim=0.2)
    r.until_word(8, 'Translation cannot')
    r.callout(pg.ext_note, dur=max(2.0, r.speech_left() + 0.2), pad=4, dim=0.2)
    r.finish_speech(0.2)

    # 9 - findings; apply; export
    r.say(9)
    r.zoom(None, 0.8, wait=False)
    r.click(bar, bar.tabRect(3).center(), dur=0.6)
    stage.settle(1)
    fr = r.wrect(pg.find_host)
    r.zoom(QRectF(fr.x() - 10, fr.y() - 20, fr.width() + 20, 330), 1.0)
    r.hold(max(1.5, r.speech_left() - 5.5))
    r.zoom(None, 0.8)
    r.click(bar, bar.tabRect(2).center(), dur=0.5)
    stage.settle(1)
    r.until_word(9, 'Apply rotation corrections')
    r.click(pg.b_apply, real=False)
    pg.b_apply.click()
    stage.settle(3)
    print('variant now:', st.variant.name, flush=True)
    r.expect(st.variant.name.endswith('drive-log corrected'), True, 'the corrected variant')
    r.callout(w.variant_combo, dur=2.4, pad=4, side='below')
    r.until_word(9, 'export writes')
    r.callout(pg.b_export, dur=max(1.8, r.speech_left() + 0.3), pad=4)
    r.cursor_on = False
    r.finish_speech(0.8)
