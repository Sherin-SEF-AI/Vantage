"""31 Live rig health: Calibrate ▸ Live rig on the sample: every camera given a synthetic stand-in stream (the parking
scene through its lens at the module's frame rate, with sensor noise), Start, the tiles with their statistics and the
planned body outline and horizon, the rig line, the rear camera's stand-in typed down to 12 frames a second (its
measured rate 12 of the module's 30, a fault), Check installation handing the frames over.

Everything runs in real time on camera (r.wait_for at speed 1: the video keeps pace with the wall clock). The
statistics are what the monitor measured while the recorder was also drawing every frame, so no frame rate is narrated
as a figure; the verdicts are read from the monitor's snapshot (r.expect): R's status 'fault' with a frame-rate reason
once its stand-in runs at 12 fps against the module's 30 (HealthSettings.fps_fault 0.5).
"""

META = {
    'id': 'liverig',
    'part': 4,
    'title': 'Live rig health',
    'promise': 'Before a drive: is every camera streaming, sharp, and where it was calibrated?',
    'menu': 'Calibrate ▸ Live rig',
}

LINES = [
    ("Before a collection drive, every camera on the rig has to be streaming, exposed, sharp, and where it was "
     "calibrated. Live rig watches every stream at once.",
     "Before a collection drive, every camera on the rig has to be streaming, exposed, sharp, and where it was "
     "calibrated. Live rig watches every stream at once."),
    ("A camera's source is saved with the project: an RTSP or HTTP URL, a V4L2 device, a GStreamer pipeline or a "
     "video file. Here each camera gets a synthetic stand-in: the parking scene through its lens, with sensor noise.",
     "A camera's source is saved with the project: an R T S P or H T T P address, a V four L two device, a G streamer "
     "pipeline, or a video file. Here, each camera gets a synthetic stand-in: the parking scene through its lens, "
     "with sensor noise."),
    ("Start reads every stream on its own thread, and reconnects a dropped one by itself.",
     "Start reads every stream on its own thread, and reconnects a dropped one by itself."),
    ("Under each image: the measured frame rate against the module's, the interval jitter, dropped frames and the "
     "image size; then mean brightness, the clipped share, and sharpness, which falls on a fogged or covered lens.",
     "Under each image: the measured frame rate against the module's, the interval jitter, dropped frames, and the "
     "image size. Then mean brightness, the clipped share, and sharpness, which falls on a fogged or covered lens."),
    ("Dashed over each image: where the plan puts the body and the horizon. The edge alignment score says how far "
     "the real body edge lies from the planned one: a knocked camera shows it. An indicator, not a measurement.",
     "Dashed over each image: where the plan puts the body, and the horizon. The edge alignment score says how far "
     "the real body edge lies from the planned one: a knocked camera shows it. An indicator, not a measurement."),
    ("The rig line sums it up: cameras up, the worst frame rate, faults and warnings, and how far apart the cameras "
     "capture. Here the stand-ins share this computer with the screen recording, so they drop frames, and the page "
     "says so.",
     "The rig line sums it up: cameras up, the worst frame rate, faults and warnings, and how far apart the cameras "
     "capture. Here, the stand-ins share this computer with the screen recording, so they drop frames, and the page "
     "says so."),
    ("Now give the rear camera's stand-in twelve frames a second. Its frame rate falls to twelve of the module's "
     "thirty: below half is a fault.",
     "Now give the rear camera's stand-in twelve frames a second. Its frame rate falls to twelve, of the module's "
     "thirty: below half is a fault."),
    ("Save frames writes each camera's newest frame and the report. Check installation freezes them and opens the "
     "installation check with them, ready to run.",
     "Save frames writes each camera's newest frame, and the report. Check installation freezes them, and opens the "
     "installation check with them, ready to run."),
    ("The same check runs without a window: --rig-health watches the streams for a set time and fails a script when "
     "a camera is faulty.",
     "The same check runs without a window: rig health, in batch mode, watches the streams for a set time, and fails "
     "a script when a camera is faulty."),
]


def scene(stage, r):
    import time
    from PyQt6.QtCore import QPointF, QRectF
    import helpers_tourc as H

    w = stage.window()
    r.win = w
    st = w.store
    stage.tilde(w.msg_label)
    vp = stage.vp

    def real(seconds):                         # record in real time, the streams running
        t_end = time.time() + seconds
        r.wait_for(lambda: time.time() >= t_end, timeout=seconds + 60)

    # 0 - open the page
    r.lower_third()
    r.say(0)
    r.cursor_on = False
    stage.orbit(r, vp, 18, 3.0, d_el=4)
    r.cursor_on = True
    r.until_word(0, 'Live rig watches')
    r.menu('Calibrate', 'Live rig')
    pg = w.page('rig')
    stage.settle(2)
    r.crossfade(0.4)
    r.finish_speech(0.2)
    cams = [c for c in st.variant.cameras if c.visible]

    # 1 - sources: synthetic stand-ins
    r.say(1)
    c0, e5 = pg.rows[cams[0].uid][0], pg.rows[cams[-1].uid][1]
    g = H.union(r.wrect(c0), r.wrect(e5))
    r.zoom(QRectF(g.x() - 60, g.y() - 60, g.width() + 100, g.height() + 200), 1.0)
    r.until_word(1, 'Here, each camera')
    r.choose(c0, c0.findData('synthetic'))
    for c in cams[1:]:
        cb = pg.rows[c.uid][0]
        r.move_to(cb, None, 0.35)
        r.ripple()
        pg.set_kind(c.uid, 'synthetic')
        r.hold(0.15)
    r.expect(all(st.project.streams.get(c.tag) == 'synthetic' for c in cams), True, 'synthetic streams')
    r.finish_speech(0.2)
    r.zoom(None, 0.8)

    # From here the stand-ins stream in real time while the recorder draws: the live stretches compose only a few
    # frames a second (H.light_wait) so the recording does not starve the streams it films, and the camera moves
    # are dissolves (a cut under a crossfade) rather than animated zooms.
    def live(seconds):
        H.light_wait(r, max(0.2, seconds))

    def cut(target, margin=24):
        r.crossfade(0.45)
        r.zoom(target, 0, margin=margin)

    # 2 - Start
    r.say(2)
    r.click(pg.b_start, real=False)
    pg.b_start.click()
    r.cursor_on = False
    live(max(1.5, r.speech_left() + 0.2))

    # 3 - the tile footer statistics
    F = st.variant.by_tag('F')
    tile = pg.tiles[F.uid]
    r.say(3)
    tr = r.wrect(tile)
    cut(QRectF(tr.x(), tr.bottom() - 220, tr.width(), 230))
    live(r.speech_left() + 0.3)

    # 4 - the planned outline and horizon; the edge alignment indicator
    r.say(4)
    cut(tr, margin=6)
    live(max(2.5, r.speech_left() - 3.2))
    left = r.wrect(pg.show_edge)
    cut(QRectF(left.x() - 20, left.y() - 60, 520, 260))
    r.callout(lambda: H.union(r.wrect(pg.show_edge), r.wrect(pg.show_score)), dur=max(2.0, r.speech_left()), pad=4)
    live(r.speech_left() + 0.1)
    cut(None)

    # 5 - the rig line
    summ = pg.summary.text()
    print('summary:', summ, flush=True)
    up = int(summ.split(' of ', 1)[0]) if summ[:1].isdigit() else 0
    r.expect(up >= 1, True, 'cameras up')
    r.expect(('warning' in summ) or ('fault' in summ and not summ.endswith(' 0 faults')), True,
             'stand-ins dropping frames under the recording load (the narration says so)')
    r.say(5)
    sr = r.wrect(pg.summary)
    cut(QRectF(sr.x() - 20, sr.y() - 40, max(sr.width() + 60, 900), 240))
    r.callout(pg.summary, dur=max(2.6, r.speech_left()), pad=4, dim=0.25)
    live(r.speech_left() + 0.3)
    cut(None)

    # 6 - the rear camera at 12 fps: a fault
    R = st.variant.by_tag('R')
    r.say(6)
    ed = pg.rows[R.uid][1]
    er = r.wrect(ed)
    cut(QRectF(er.x() - 140, er.y() - 120, 560, 315))
    r.cursor_on = True
    r.enter(ed, '12', cps=6)
    if st.project.streams.get('R') != 'synthetic:12':
        ed.editingFinished.emit()
    r.expect(st.project.streams.get('R'), 'synthetic:12', "R's stand-in")
    r.cursor_on = False
    cut(None)
    t_end = time.time() + 30
    while pg.cam_state('R').get('status') != 'fault' and time.time() < t_end:
        live(0.5)
    sR = pg.cam_state('R')
    print('R state:', {k: sR.get(k) for k in ('status', 'fps', 'nominal_fps', 'reasons')}, flush=True)
    r.expect(sR.get('status'), 'fault', "R's verdict")
    r.expect(any(x.endswith('fps of 30') for x in sR.get('reasons', [])), True, "R's reason")
    r.expect(round(sR.get('fps') or 0), 12, "R's measured frame rate")
    cut(r.wrect(pg.tiles[R.uid]), margin=6)
    live(max(2.4, r.speech_left() + 0.6))
    cut(None)

    # 7 - Save frames; Check installation
    r.say(7)
    br = H.union(r.wrect(pg.b_save), r.wrect(pg.b_start))
    cut(QRectF(br.x() - 30, br.y() - 120, br.width() + 60, br.height() + 160))
    r.cursor_on = True
    r.move_to(pg.b_save, None, 0.6)
    r.callout(pg.b_save, dur=2.0, pad=4)
    live(1.6)
    r.until_word(7, 'Check installation')
    r.click(pg.b_check, real=False)
    pg.b_check.click()
    stage.settle(2)
    ip = w.page('install')
    r.expect(w.page_name(), 'install', 'the page shown')
    r.zoom(None, 0.0)
    r.crossfade(0.5)
    r.cursor_on = False
    r.hold(max(1.5, r.speech_left() - 1.0))
    r.finish_speech(0.2)

    # 8 - batch mode
    r.say(8)
    pg.stop()
    r.zoom(r.wrect(ip.tile_host), 1.2, margin=8)
    r.hold(max(1.5, r.speech_left() - 1.4))
    r.zoom(None, 0.9)
    r.finish_speech(0.8)
