"""30 Lidar–camera calibration: Calibrate ▸ Lidar–camera calibration on the sample's data rig, front camera F with the
roof lidar L1 (seed 4): the pairs ranked by beam coverage, the pose plan and its predicted uncertainty, a mounting
error injected into the synthetic capture (15 mm forward, 8 mm up, 0.6 deg of yaw), Capture and solve (a marked
time-lapse), the overlay through the plan and the solve, the tiles, the findings, the poses table, Apply.

Figures read from the page while recording (r.expect), the same as core.lidar_calib.run gives in a light run with
these settings: the roof lidar covers 15 % of F's image on the page (R's 17 %, the best); six poses, predicted from the planes
alone to +-0.9 / +-1.7 / +-3.4 mm and +-0.09 deg; result: L1 is 15 mm / 0.61 deg from its planned pose, adjust;
determined to +-2.1 mm / +-0.020 deg (1 sigma) from 6 poses; the solve is 2.4 mm / 0.023 deg from the true mounting.
"""

META = {
    'id': 'lidarcal',
    'part': 4,
    'title': 'Lidar–camera calibration',
    'promise': 'Where the lidar really sits, from a board both sensors see.',
    'menu': 'Calibrate ▸ Lidar–camera calibration',
}

LINES = [
    ("A lidar on the roof and a camera on the bumper have to agree on where things are. The lidar–camera calibration "
     "measures where the lidar really sits, from a checkerboard held up where both see it.",
     "A lidar on the roof, and a camera on the bumper, have to agree on where things are. The lidar camera calibration "
     "measures where the lidar really sits, from a checkerboard held up where both see it."),
    ("Pairs are ranked by how much of the camera's image the lidar's beams cover. Here the front camera, which the "
     "roof lidar covers on 15 per cent of its image.",
     "Pairs are ranked by how much of the camera's image the lidar's beams cover. Here, the front camera, which the "
     "roof lidar covers on fifteen per cent of its image."),
    ("The pose plan puts six boards where both sensors see all of the board, at least four lidar channels cross it "
     "and its squares stay big enough in the image, each turned so that together they pin all six degrees of freedom.",
     "The pose plan puts six boards where both sensors see all of the board, at least four lidar channels cross it, "
     "and its squares stay big enough in the image, each turned, so that together they pin all six degrees of "
     "freedom."),
    ("From the board planes alone, the plan predicts the lidar's position to within 3.4 millimetres and its angles "
     "to 0.09 degrees; the edges and the squares narrow it further.",
     "From the board planes alone, the plan predicts the lidar's position to within three point four millimetres, "
     "and its angles to zero point zero nine degrees. The edges and the squares narrow it further."),
    ("The captures here are synthetic, through the camera's lens and the lidar's beams with range noise; a folder of "
     "real image and point-cloud pairs works the same way. Inject an error: 15 millimetres forward, 8 up, 0.6 degrees "
     "of yaw.",
     "The captures here are synthetic, through the camera's lens and the lidar's beams, with range noise. A folder of "
     "real image and point cloud pairs works the same way. Inject an error: fifteen millimetres forward, eight up, "
     "zero point six degrees of yaw."),
    ("Capture and solve: the camera finds the corners through its lens model; the lidar gives the board's plane, its "
     "edges and, in the reflectivity, its squares. One robust solve fits the lidar and every board together.",
     "Capture and solve. The camera finds the corners through its lens model. The lidar gives the board's plane, its "
     "edges, and, in the reflectivity, its squares. One robust solve fits the lidar and every board together."),
    ("The overlay is the proof: the lidar's points drawn on the image through the plan miss the board; through the "
     "solve, they land on it.",
     "The overlay is the proof. The lidar's points, drawn on the image through the plan, miss the board. Through the "
     "solve, they land on it."),
    ("L1 is 15 millimetres and 0.61 degrees from its plan, outside the tolerance of 10 millimetres and 0.3 degrees: "
     "adjust it. It is determined to 2.1 millimetres and 0.02 degrees.",
     "L one is fifteen millimetres, and zero point six one degrees, from its plan, outside the tolerance of ten "
     "millimetres and zero point three degrees: adjust it. It is determined to two point one millimetres, and zero "
     "point zero two degrees."),
    ("The error was injected, so the solve can be checked: it is 2.4 millimetres and 0.023 degrees from the true "
     "mounting.",
     "The error was injected, so the solve can be checked: it is two point four millimetres, and zero point zero two "
     "three degrees, from the true mounting."),
    ("The table lists each pose with its residuals. A board that moved between the two captures is set aside, and "
     "boards all tilted the same way are reported as leaving a direction undetermined, not as a number.",
     "The table lists each pose, with its residuals. A board that moved between the two captures is set aside, and "
     "boards all tilted the same way are reported as leaving a direction undetermined, not as a number."),
    ("Apply moves the lidar to its measured pose in one undo step, or keep the plan with As a new variant. Export "
     "writes the transform both ways, with its uncertainty.",
     "Apply moves the lidar to its measured pose, in one undo step, or keep the plan with, as a new variant. Export "
     "writes the transform both ways, with its uncertainty."),
]


def scene(stage, r):
    from PyQt6.QtCore import QPointF, QRectF
    import helpers_tourc as H

    def setup(p):
        p.analysis.lidar_calib = {'camera': 'F', 'lidar': 'L1', 'seed': 4}
    w = stage._rig_window(setup)
    r.win = w
    st = w.store
    stage.tilde(w.msg_label)
    vp = stage.vp
    L1 = next(s for s in st.variant.sensors if s.tag == 'L1')

    # 0 - open the page
    r.lower_third()
    r.say(0)
    r.cursor_on = False
    stage.orbit(r, vp, 22, 3.6, d_el=4)
    r.cursor_on = True
    r.until_word(0, 'measures where the lidar')
    r.menu('Calibrate', 'Lidar–camera calibration')
    pg = w.page('lidar_calib')
    r.wait_for(lambda: pg.plan is not None and bool(pg.plan.get('poses')), timeout=120)
    stage.settle(1)
    r.crossfade(0.4)
    r.finish_speech(0.2)

    # 1 - the pair, ranked by coverage
    items = [pg.cam_box.itemText(i) for i in range(pg.cam_box.count())]
    print('camera list:', items, flush=True)
    r.expect(pg.cam_box.currentText(), 'F · 15 %', 'the camera picked')
    r.expect('R · 17 %' in items, True, 'the best overlap')
    r.say(1)
    cb = r.wrect(pg.cam_box)
    r.zoom(QRectF(cb.x() - 140, cb.y() - 120, 560, 420), 1.0)
    r.click(pg.cam_box, None, 0.7, real=False, hold=0.1)
    pg.cam_box.showPopup()
    r.hold(max(1.8, r.speech_left() - 1.6))
    pg.cam_box.hidePopup()
    r.hold(0.2)
    r.callout(lambda: H.union(r.wrect(pg.cam_box), r.wrect(pg.lid_box)), dur=max(1.4, r.speech_left() + 0.2), pad=4)
    r.finish_speech(0.2)
    r.zoom(None, 0.8)

    # 2 - the pose plan
    r.say(2)
    r.zoom(pg.plan_view, 1.0, margin=10)
    r.cursor_on = False
    r.hold(max(1.5, r.speech_left() - 0.6))
    r.finish_speech(0.2)

    # 3 - predicted uncertainty
    note = pg.plan_note.text()
    print('plan note:', note, flush=True)
    r.expect('±0.9 / ±1.7 / ±3.4 mm' in note and '±0.09°' in note, True, 'predicted uncertainty')
    r.say(3)
    nr = r.wrect(pg.plan_note)
    r.zoom(QRectF(nr.x() - 20, nr.y() - 120, nr.width() + 40, nr.height() + 150), 1.0)
    r.callout(pg.plan_note, dur=max(2.6, r.speech_left() + 0.2), pad=4, dim=0.25)
    r.finish_speech(0.2)
    r.zoom(None, 0.8)

    # 4 - synthetic captures; the injected error
    r.say(4)
    r.cursor_on = True
    src = r.wrect(pg.source)
    r.zoom(QRectF(src.x() - 20, src.y() - 40, 520, 300), 1.0, wait=False)
    r.move_to(pg.source, None, 0.8)
    r.callout(pg.source, dur=2.4, pad=4)
    r.until_word(4, 'Inject an error')
    e0 = r.wrect(pg.fields['err_x_mm'])
    r.zoom(QRectF(e0.x() - 160, e0.y() - 60, 560, 300), 0.9)
    for key, text in (('err_x_mm', '15'), ('err_z_mm', '8'), ('err_yaw_deg', '0.6')):
        r.enter(pg.fields[key], text)
        r.hold(0.3)
    stg = pg.settings
    r.expect((stg.err_x_mm, stg.err_z_mm, stg.err_yaw_deg), (15.0, 8.0, 0.6), 'injected error')
    r.finish_speech(0.2)
    r.zoom(None, 0.8)
    r.wait_for(lambda: pg.plan is not None and bool(pg.plan.get('poses')), timeout=120)

    # 5 - Capture and solve: a time-lapse
    r.say(5)
    r.click(pg.b_run, real=False)
    pg.b_run.click()
    r.move(r.wpos(pg.plan_view), 0.8)
    r.cursor_on = False
    r.callout(w.state_btn, 'Progress', dur=3.2, pad=4, side='below')      # the job runs in the background
    H.wait_job(r, lambda: pg.result is not None and not pg.busy, timeout=600, fast=15.0)
    stage.settle(1)
    r.zoom(None, 0.6, wait=False)
    res = pg.result
    print('result:', res.message, flush=True)
    r.finish_speech(0.2)

    # 6 - the overlay: plan against solved
    r.say(6)
    ov = r.wrect(pg.overlay)
    side = min(ov.width() - 12, ov.height() - 48)
    top = ov.y() - 34
    hh = 34 + 22 + side * 0.86
    r.zoom(QRectF(ov.center().x() - hh * 8 / 9, top, hh * 16 / 9, hh), 1.0, margin=0)
    r.cursor_on = True
    r.click(pg.mode.buttons['plan'], dur=0.6)
    r.until_word(6, 'Through the solve')
    r.click(pg.mode.buttons['solved'], dur=0.5)
    r.finish_speech(0.6)
    r.zoom(None, 0.8)

    # 7 - the tiles
    find = [f.text.text() for f in pg.findings]
    print('findings:', find, flush=True)
    r.expect(res.message, 'L1 is 15 mm / 0.61° from its planned pose: adjust the bracket, or apply the measured pose '
                          'to the variant.', 'the verdict')
    r.expect('Determined to ±2.1 mm / ±0.020° (1σ) from 6 poses.' in find, True, 'the uncertainty')
    r.say(7)
    t0, t1 = r.wrect(pg.tiles['verdict']), r.wrect(pg.tiles['sigma'])
    r.zoom(QRectF(t0.x() - 10, t0.y() - 20, t1.right() - t0.x() + 20, t1.bottom() - t0.y() + 40), 1.0)
    r.callout(pg.tiles['pos'], dur=2.0, pad=3, dim=0.2)
    r.callout(pg.tiles['ang'], dur=2.0, pad=3, dim=0.0)
    r.until_word(7, 'It is determined')
    r.callout(pg.tiles['sigma'], dur=max(2.0, r.speech_left() + 0.3), pad=3, dim=0.2)
    r.finish_speech(0.2)

    # 8 - against the truth: the findings
    r.expect('Synthetic: the solve is 2.4 mm / 0.023° from the true mounting.' in find, True, 'against the truth')
    r.say(8)
    fr = r.wrect(pg.find_host)
    r.zoom(QRectF(fr.x() - 20, fr.y() - 40, fr.width() + 40, 200), 1.0)
    r.callout(lambda: r.wrect(pg.findings[-1]), dur=max(2.4, r.speech_left() + 0.3), pad=4, dim=0.25)
    r.finish_speech(0.2)

    # 9 - the poses table
    r.say(9)
    tb = pg.table
    r.zoom(H.rows_rect(r, tb).adjusted(-10, -30, 10, 10), 1.0)
    r.hold(max(1.5, r.speech_left() - 0.4))
    r.finish_speech(0.2)
    r.zoom(None, 0.8)

    # 10 - Apply
    r.say(10)
    br = H.union(r.wrect(pg.b_export), r.wrect(pg.b_apply))
    r.zoom(QRectF(br.x() - 30, br.y() - 120, br.width() + 60, br.height() + 160), 1.0)
    r.cursor_on = True
    x0 = st.variant.sensor(L1.uid).x
    r.click(pg.b_apply, real=False)
    pg.b_apply.click()
    stage.settle(2)
    r.expect(round((st.variant.sensor(L1.uid).x - x0) * 1000), 15, 'L1 moved by the measured offset')
    r.callout(lambda: H.union(r.wrect(pg.b_variant), r.wrect(pg.b_apply)), dur=2.2, pad=4, dim=0.2)
    r.until_word(10, 'Export writes')
    r.callout(pg.b_export, dur=max(1.8, r.speech_left() + 0.3), pad=4, dim=0.2)
    r.cursor_on = False
    r.finish_speech(0.4)
    r.zoom(None, 0.8)
    r.hold(0.6)
