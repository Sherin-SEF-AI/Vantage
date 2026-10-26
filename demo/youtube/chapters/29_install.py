"""29 Installation check and joint solve: Calibrate ▸ Installation check on the sample (6-cam · v3, its eight floor
boards): the sources (file, live, synthetic), a simulated mounting error (seed 7: up to 1.5 deg and 15 mm per camera,
exposure +-20 %, vignetting 30 %), the tolerance, Run check (a marked time-lapse), the captures with planned boards
and detected corners, the table against the plan, the check against the simulated error, the joint solve and its
board table, the stitched bird's-eye with planned and measured poses, and Apply as as-built variant.

Figures read from the page while recording (r.expect / printed), the same as core.install.check_all gives in a light
run with the page's seed: all six cameras 'adjust' (F off by 11 mm / 1.4 deg); the simulated error recovered to
within 0.16 deg and 3.1 mm; the joint solve's seams agree to 0.8 mm.
"""

META = {
    'id': 'install',
    'part': 4,
    'title': 'Installation check and joint solve',
    'promise': 'One image per camera: is each one mounted where the plan says?',
    'menu': 'Calibrate ▸ Installation check  ·  Ctrl+Shift+I',
}

LINES = [
    ("With the boards on the floor, the installation check measures where each camera really sits, from one image per "
     "camera: an end-of-line check for every vehicle built.",
     "With the boards on the floor, the installation check measures where each camera really sits, from one image per "
     "camera: an end of line check, for every vehicle built."),
    ("Each image can come from a file, a live stream (a V4L2 device, a GStreamer pipeline or an RTSP URL) or, as "
     "here, a synthetic capture rendered from the plan.",
     "Each image can come from a file, a live stream, such as a V four L two device, a G streamer pipeline, or an R T "
     "S P address, or, as here, a synthetic capture rendered from the plan."),
    ("A synthetic capture can carry a simulated mounting error: here up to 1.5 degrees and 15 millimetres per camera, "
     "with different exposures and vignetting.",
     "A synthetic capture can carry a simulated mounting error: here, up to one and a half degrees, and fifteen "
     "millimetres per camera, with different exposures, and vignetting."),
    ("The tolerance is 10 millimetres and 1 degree. Within it a camera is OK; up to three times, adjust it or keep it "
     "as built; beyond that, re-mount it.",
     "The tolerance is ten millimetres and one degree. Within it, a camera is OK. Up to three times, adjust it, or "
     "keep it as built. Beyond that, re-mount it."),
    ("Run check renders the six captures, finds the boards in each and solves the poses, then builds the remap "
     "tables for both sets of poses.",
     "Run check renders the six captures, finds the boards in each, and solves the poses, then builds the remap "
     "tables for both sets of poses."),
    ("Each board is cut out of the fisheye image as a rectified patch through the exact lens model, and its corners "
     "found with OpenCV. Dashed is where the plan puts the board; the dots are the corners found.",
     "Each board is cut out of the fisheye image as a rectified patch, through the exact lens model, and its corners "
     "found with open C V. Dashed is where the plan puts the board. The dots are the corners found."),
    ("The table compares each camera with the plan: boards, corners, residual, the offsets in millimetres and "
     "degrees, and a verdict. The front camera is 11 millimetres and 1.4 degrees off: adjust it.",
     "The table compares each camera with the plan: boards, corners, residual, the offsets in millimetres and "
     "degrees, and a verdict. The front camera is eleven millimetres, and one point four degrees off: adjust it."),
    ("The error was simulated, so the page checks itself: the injected errors came back to within 0.16 degrees and "
     "3.1 millimetres.",
     "The error was simulated, so the page checks itself: the injected errors came back to within zero point one six "
     "degrees, and three point one millimetres."),
    ("The joint solve refines all cameras together with each board's place on the floor, so cameras that share a "
     "board agree at the seams: here to 0.8 millimetres.",
     "The joint solve refines all cameras together, with each board's place on the floor, so cameras that share a "
     "board agree at the seams: here, to zero point eight millimetres."),
    ("Stitched with the planned poses, the mounting errors break the lines at the seams; with the measured poses they "
     "join up again.",
     "Stitched with the planned poses, the mounting errors break the lines at the seams. With the measured poses, "
     "they join up again."),
    ("Save the report, the stitched PNG, or the remap tables the surround-view pipeline loads; or apply the measured "
     "poses as an as-built variant, in one undo step.",
     "Save the report, the stitched P N G, or the remap tables the surround view pipeline loads. Or apply the "
     "measured poses as an as-built variant, in one undo step."),
]


def scene(stage, r):
    from PyQt6.QtCore import QPointF, QRectF
    import helpers_tourc as H

    w = stage.window()
    r.win = w
    st = w.store
    stage.tilde(w.msg_label)
    vp = stage.vp

    # 0 - open the page
    r.lower_third()
    r.say(0)
    r.cursor_on = False
    stage.orbit(r, vp, 18, 3.2, d_el=4)
    r.cursor_on = True
    r.until_word(0, 'from one image per')
    r.menu('Calibrate', 'Installation check')
    pg = w.page('install')
    stage.settle(3)
    r.crossfade(0.4)
    r.finish_speech(0.2)
    cams = [c for c in st.variant.cameras if c.visible]

    # 1 - sources
    r.say(1)
    combo0 = pg.rows[cams[0].uid][0]
    g0 = r.wrect(pg.rows[cams[0].uid][0])
    g5 = r.wrect(pg.rows[cams[-1].uid][2])
    src = QRectF(g0.x() - 60, g0.y() - 40, g5.right() - g0.x() + 80, g5.bottom() - g0.y() + 90)
    r.zoom(src, 1.0)
    r.click(combo0, None, 0.7, real=False, hold=0.1)
    combo0.showPopup()
    r.hold(max(1.5, r.speech_left() - 3.0))
    combo0.hidePopup()
    r.hold(0.3)
    r.callout(lambda: H.union(r.wrect(pg.rows[cams[0].uid][0]), r.wrect(pg.rows[cams[-1].uid][2])), dur=max(2.0, r.speech_left()),
              pad=4)
    r.finish_speech(0.2)

    # 2 - simulated mounting error
    r.expect((round(pg.sim_ang.value(), 1), round(pg.sim_pos.value()), round(pg.sim_expo.value()),
              round(pg.sim_vig.value()), int(pg.sim_seed.value())), (1.5, 15, 20, 30, 7), 'simulated error settings')
    r.say(2)
    s0, s1 = r.wrect(pg.sim), r.wrect(pg.sim_res)
    r.zoom(QRectF(s0.x() - 20, s0.y() - 30, 520, s1.bottom() - s0.y() + 60), 1.0)
    r.callout(lambda: H.union(r.wrect(pg.sim_ang), r.wrect(pg.sim_pos)).adjusted(-120, 0, 0, 0), dur=2.6, pad=4)
    r.until_word(2, 'with different exposures')
    r.callout(lambda: H.union(r.wrect(pg.sim_expo), r.wrect(pg.sim_vig)).adjusted(-120, 0, 0, 0), dur=max(1.8, r.speech_left()),
              pad=4)
    r.finish_speech(0.2)

    # 3 - tolerance
    r.expect((round(pg.tol_pos.value()), round(pg.tol_ang.value(), 2)), (10, 1.0), 'tolerance')
    r.say(3)
    t0, t1 = r.wrect(pg.tol_pos), r.wrect(pg.min_boards)
    r.zoom(QRectF(t0.x() - 140, t0.y() - 40, 520, t1.bottom() - t0.y() + 110), 1.0)
    r.callout(lambda: H.union(r.wrect(pg.tol_pos), r.wrect(pg.tol_ang)).adjusted(-120, 0, 0, 0), dur=max(2.4, r.speech_left()),
              pad=4)
    r.finish_speech(0.2)
    r.zoom(None, 0.8)

    # 4 - Run check: a time-lapse
    r.say(4)
    r.click(pg.b_run, real=False)
    pg.b_run.click()
    r.move(r.wpos(pg.tile_host, QPointF(pg.tile_host.width() * 0.5, pg.tile_host.height() * 0.5)), 0.8)
    r.cursor_on = False
    r.wait_for(lambda: not pg.busy and bool(pg.results), timeout=900, speed=10)
    r.hold(0.4)
    status = pg.status.text()
    print('status:', status, flush=True)
    r.finish_speech(0.2)

    # 5 - the captures: planned boards dashed, corners found
    r.say(5)
    F = st.variant.by_tag('F')
    tile = pg.tiles[F.uid]
    r.zoom(r.wrect(pg.tile_host), 1.0, margin=8)
    r.hold(1.6)
    r.cursor_on = True
    r.click(tile, None, 0.7, real=False)
    pg.toggle_single(F.uid)
    stage.settle(1)
    r.crossfade(0.3)
    r.zoom(r.wrect(tile), 0.8, margin=4)
    r.hold(max(1.6, r.speech_left() - 1.2))
    pg.toggle_single(F.uid)
    stage.settle(1)
    r.crossfade(0.3)
    r.zoom(None, 0.8)
    r.finish_speech(0.2)

    # 6 - the table against the plan
    t = pg.table
    vals = {t.item(i, 0).text(): [t.item(i, j).text() for j in range(t.columnCount())] for i in range(t.rowCount())}
    for k, v in vals.items():
        print('row', k, v, flush=True)
    r.expect(vals['F'][11], 'Off by 11 mm / 1.4°: adjust the bracket or keep it as built', "F's verdict")
    r.expect(sum(1 for v in vals.values() if v[11].endswith('keep it as built')), 6, 'cameras to adjust')
    r.say(6)
    tr = r.wrect(t)
    r.zoom(QRectF(tr.x() - 10, tr.y() - 40, tr.width() + 20, tr.height() + 50), 1.0)
    r.until_word(6, 'The front camera')
    r.callout(H.cell(r, t, 0), dur=max(2.4, r.speech_left() + 0.3), pad=2, dim=0.25)
    r.finish_speech(0.2)

    # 7 - the check against the simulated error
    summ = pg.summary.text()
    print('summary:', summ, flush=True)
    r.expect(summ, 'simulated error recovered to within 0.16° · 3.1 mm', 'recovered')
    r.say(7)
    r.callout(pg.summary, dur=max(2.6, r.speech_left() + 0.3), pad=4, dim=0.3)
    r.finish_speech(0.3)
    r.zoom(None, 0.8)

    # 8 - the joint solve
    jl = pg.joint_lbl.text()
    print('joint:', jl, flush=True)
    r.expect('seams agree to 0.8 mm' in jl, True, 'joint seam agreement')
    r.say(8)
    jr = H.union(r.wrect(pg.joint_lbl), r.wrect(pg.btable))
    r.zoom(QRectF(jr.x() - 20, jr.y() - 30, jr.width() + 40, jr.height() + 60), 1.0)
    r.callout(pg.joint_lbl, dur=2.6, pad=4, dim=0.2)
    r.until_word(8, 'agree at the seams')
    r.callout(pg.btable, dur=max(2.0, r.speech_left() + 0.3), pad=4, dim=0.2)
    r.finish_speech(0.2)
    r.zoom(None, 0.8)

    # 9 - stitched: planned against measured
    r.say(9)
    br = r.wrect(pg.bev)
    r.zoom(QRectF(br.x() - 10, br.y() - 40, br.width() + 20, br.height() + 50), 1.0)
    r.cursor_on = True
    r.click(pg.which.buttons['planned'], dur=0.6)
    stage.settle(1)
    r.hold(max(1.8, r.speech_left() - 3.0))
    r.until_word(9, 'With the measured poses')
    r.click(pg.which.buttons['measured'], dur=0.5)
    stage.settle(1)
    r.finish_speech(0.6)
    r.zoom(None, 0.8)

    # 10 - outputs; as-built variant
    r.say(10)
    btns = H.union(r.wrect(pg.b_png), r.wrect(pg.b_apply))
    r.zoom(QRectF(btns.x() - 40, btns.y() - 120, btns.width() + 80, btns.height() + 160), 1.0)
    r.callout(lambda: H.union(r.wrect(pg.b_png), r.wrect(pg.b_lut)), dur=2.6, pad=4, dim=0.2)
    r.until_word(10, 'Or apply the measured')
    r.click(pg.b_apply, real=False)
    pg.b_apply.click()
    stage.settle(3)
    r.expect(st.variant.name, '6-cam · v3 · as-built', 'the as-built variant')
    r.zoom(None, 0.8)
    r.toolbar('view_persp')                       # the measured poses, active, in the main window
    stage.settle(3)
    r.crossfade(0.5)
    r.callout(w.variant_combo, dur=max(2.0, r.speech_left() + 0.4), pad=4, side='below')
    r.cursor_on = False
    r.finish_speech(1.0)
