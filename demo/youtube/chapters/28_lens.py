"""28 Unit lens calibration: Calibrate ▸ Lens calibration on the sample's front camera F (STURDeCAM31, H05R2 fisheye):
the board, the three sources, a synthetic set of 20 views from a simulated unit (seed 1: focal +-1.5 %, centre +-15
px), Calibrate, the views table and the corner coverage, the parameters against the library lens, the check against
the simulated truth, Apply to F (the Inspector's Unit calibration row), Save, and Import of calibrations made
elsewhere (the six ROS camera_info files Vantage writes for this rig, matched by camera name).

Figures read from the dialog while recording (r.expect), computed by core.calib exactly as the dialog does (checked
in a light run first): 20 of 20 views, RMS 0.083 px, corners cover 82 % of the image; cx -10.86 px, cy +12.98 px
against the library lens; the library lens puts a ray up to 20.4 px (2.20 deg) from where this unit images it within
69 deg off axis; against the simulated truth 0.86 px RMS. The calibration takes a few seconds and is shown in real
time.
"""

META = {
    'id': 'lens',
    'part': 4,
    'title': 'Unit lens calibration',
    'promise': 'Each camera’s own lens, measured from a hand-held checkerboard.',
    'menu': 'Calibrate ▸ Lens calibration  ·  Ctrl+Shift+L',
}

LINES = [
    ("Every lens is made to a tolerance: a real unit's focal length and image centre differ a little from the "
     "datasheet's. Lens calibration gives each camera its own intrinsics.",
     "Every lens is made to a tolerance: a real unit's focal length and image centre differ a little from the "
     "datasheet's. Lens calibration gives each camera its own intrinsics."),
    ("Pick the camera, here the front one, and the board: nine by six inner corners, with 60-millimetre squares.",
     "Pick the camera, here the front one, and the board: nine by six inner corners, with sixty millimetre squares."),
    ("Views come from image files, a live stream, or a synthetic set: a simulated unit, up to 1.5 per cent off in "
     "focal length and 15 pixels in its centre, seeing a hand-held board in twenty places.",
     "Views come from image files, a live stream, or a synthetic set: a simulated unit, up to one and a half per cent "
     "off in focal length, and fifteen pixels in its centre, seeing a hand-held board in twenty places."),
    ("Calibrate finds the board in every view, in rectified patches where the fisheye bends it too much, and fits the "
     "cv::fisheye model: focal lengths, centre and four distortion terms.",
     "Calibrate finds the board in every view, in rectified patches where the fisheye bends it too much, and fits the "
     "open C V fisheye model: focal lengths, centre, and four distortion terms."),
    ("All twenty views were used, with a residual of 0.083 pixels RMS, and the corners cover 82 per cent of the "
     "image: the tint shows where they fell.",
     "All twenty views were used, with a residual of zero point zero eight three pixels R M S, and the corners cover "
     "eighty-two per cent of the image. The tint shows where they fell."),
    ("The table lists each view with its corners and residual; a view the fit cannot use is set aside, with the "
     "reason.",
     "The table lists each view, with its corners and residual. A view the fit cannot use is set aside, with the "
     "reason."),
    ("Against the library lens, this unit's centre is 11 and 13 pixels off. Within the field the views cover, the "
     "library lens would put a ray up to 20.4 pixels, 2.2 degrees, from where this unit images it.",
     "Against the library lens, this unit's centre is eleven and thirteen pixels off. Within the field the views "
     "cover, the library lens would put a ray up to twenty point four pixels, two point two degrees, from where this "
     "unit images it."),
    ("The views were simulated, so the result can be checked against the truth: within 0.86 pixels RMS.",
     "The views were simulated, so the result can be checked against the truth: within zero point eight six pixels "
     "R M S."),
    ("Apply to F stores it on the camera in one undo step. From then on it replaces the library lens in the analysis, "
     "the installation check, the remap tables and every export, which says so.",
     "Apply to F stores it on the camera, in one undo step. From then on, it replaces the library lens in the "
     "analysis, the installation check, the remap tables, and every export, which says so."),
    ("Save writes it as OpenCV YAML or Vantage JSON. Calibrations made elsewhere import too: OpenCV, ROS camera_info, "
     "Kalibr or JSON, each matched to a camera by name, topic, file name or serial number.",
     "Save writes it as open C V yamel, or Vantage Jason. Calibrations made elsewhere import too: open C V, ross "
     "camera info, kalibur, or Jason, each matched to a camera by name, topic, file name, or serial number."),
]


def scene(stage, r):
    import os
    from PyQt6.QtCore import QPointF, QRectF
    from vantage.core import calib, exporters
    from vantage.ui.dialogs.lens_calib_dialog import ImportCalibDialog
    import helpers_tourc as H

    w = stage.window()
    r.win = w
    st = w.store
    stage.tilde(w.msg_label)
    F = st.variant.by_tag('F')
    st.select_cams([F.uid])
    stage.settle(3)
    vp = stage.vp

    # files for the import at the end (not filmed): the ROS camera_info files Vantage writes for this rig
    ros_dir = os.path.join(stage.home, 'calibrations', 'ros')
    exporters.export(['ros_camera_info'], st.project, w.ctx.analysis.result,
                     exporters.ExportOptions(out_dir=ros_dir))
    ros_files = sorted(os.path.join(dp, f) for dp, _, fs in os.walk(ros_dir) for f in fs if f.endswith('.yaml'))
    print('camera_info files:', [os.path.basename(f) for f in ros_files], flush=True)

    # 0 - why; open the dialog
    r.lower_third()
    r.say(0)
    r.cursor_on = False
    stage.orbit(r, vp, 16, 3.0, d_el=4)
    r.cursor_on = True
    r.until_word(0, 'Lens calibration gives')
    r.menu('Calibrate', 'Lens calibration')
    d = w._calib_dialog
    stage.extra.append(d)
    d.resize(1660, 960)                           # wider than its default, so the result table shows every column
    from PyQt6.QtWidgets import QSplitter
    d.findChild(QSplitter).setSizes([340, 700, 620])
    d.move(w.mapToGlobal(w.rect().center()) - d.rect().center())
    stage.settle(2)
    stage.film_centred(r, d)                      # the dialog on its own, centred on the stage background
    r.crossfade(0.5)
    r.finish_speech(0.2)
    dr = r.wrect(d)
    left = d.cam.parentWidget()

    # 1 - the camera and the board
    r.say(1)
    lr = r.wrect(left)
    r.zoom(QRectF(lr.x() - 10, lr.y() - 10, lr.width() + 260, 360), 1.0)
    r.callout(lambda: H.union(r.wrect(d.cam), r.wrect(d.cam_info)), dur=2.2, pad=4)
    r.until_word(1, 'and the board')
    r.callout(lambda: H.union(r.wrect(d.cols), r.wrect(d.square)).adjusted(-130, 0, 0, 0), dur=max(2.0, r.speech_left()),
              pad=4)
    r.expect((int(d.cols.value()), int(d.rows.value()), round(d.square.value())), (9, 6, 60), 'board')
    r.finish_speech(0.2)

    # 2 - the sources; a synthetic set
    r.say(2)
    r.move_to(d.source, None, 0.7)
    r.hold(0.8)
    r.click(d.source.buttons['synthetic'], dur=0.5)
    stage.settle(1)
    r.expect((round(d.syn_f.value(), 1), round(d.syn_pp.value()), int(d.syn_n.value())), (1.5, 15, 20), 'synthetic set')
    r.until_word(2, 'a simulated unit')
    r.callout(lambda: H.union(r.wrect(d.syn_n), r.wrect(d.syn_seed)).adjusted(-130, 0, 0, 0), dur=3.0, pad=4)
    r.until_word(2, 'seeing a hand-held')
    b_render = H.button(d, 'Render views')
    r.click(b_render, real=False)
    b_render.click()
    r.zoom(None, 0.8, wait=False)
    H.wait_job(r, lambda: not d.busy and len(d.views) == 20, timeout=300, fast=4.0)
    r.hold(0.4)
    r.finish_speech(0.2)

    # 3 - Calibrate
    r.say(3)
    r.click(d.b_run, real=False)
    d.b_run.click()
    r.move(r.wpos(d.preview, QPointF(d.preview.width() * 0.7, d.preview.height() * 0.6)), 0.8)
    H.wait_job(r, lambda: not d.busy and d.result is not None, timeout=300, fast=6.0)
    r.cursor_on = False
    pr = r.wrect(d.preview)
    r.zoom(QRectF(pr.x() - 10, pr.y() - 30, pr.width() + 20, pr.height() + 40), 1.0, wait=False)
    r.finish_speech(0.2)

    # 4 - the result: 20 of 20, RMS, coverage
    summ = d.summary.text()
    print('summary:', summ, flush=True)
    r.expect(summ, '20 of 20 views · RMS 0.083 px · corners cover 82 % of the image', 'summary')
    r.say(4)
    r.zoom(None, 0.8)
    rr = r.wrect(d.summary)
    r.zoom(QRectF(rr.x() - 20, rr.y() - 40, 640, 380), 1.0)
    r.callout(d.summary, dur=3.2, pad=4, dim=0.25)
    r.until_word(4, 'The tint')
    r.zoom(None, 0.8)
    r.zoom(QRectF(pr.x() - 10, pr.y() - 30, pr.width() + 20, pr.height() + 40), 1.0)
    r.finish_speech(0.2)

    # 5 - the views table
    r.say(5)
    vt = d.vtable
    r.zoom(QRectF(r.wrect(vt).x() - 10, r.wrect(vt).y() - 20, r.wrect(vt).width() + 20, r.wrect(vt).height() + 30), 1.0)
    row = 13                                      # view_14
    r.click(vt.viewport(), QPointF(vt.columnViewportPosition(0) + 40, vt.rowViewportPosition(row) + vt.rowHeight(row) / 2),
            dur=0.7)
    stage.settle(1)
    r.callout(H.cell(r, vt, row), dur=max(2.0, r.speech_left() + 0.2), pad=2, dim=0.2)
    r.finish_speech(0.2)
    r.zoom(None, 0.8)

    # 6 - against the library lens
    pt = d.ptable
    rows = {pt.item(i, 0).text(): [pt.item(i, j).text() for j in range(4)] for i in range(pt.rowCount())}
    print('parameters:', rows, flush=True)
    r.expect(rows['cx px'][3], '-10.86', 'cx difference')
    r.expect(rows['cy px'][3], '+12.98', 'cy difference')
    diff = d.diff.text()
    print('diff:', diff, flush=True)
    r.expect('up to 20.4 px' in diff and '2.20°' in diff, True, 'library lens against this unit')
    r.expect('(up to 69° off axis)' in diff, True, 'field the views cover')
    r.say(6)
    tr = r.wrect(pt)
    r.zoom(QRectF(tr.x() - 20, tr.y() - 20, tr.width() + 40, r.wrect(d.diff).bottom() - tr.y() + 50), 1.0)
    r.callout(lambda: H.union(H.cell(r, pt, 2)(), H.cell(r, pt, 3)()), dur=2.6, pad=2, dim=0.2)
    r.until_word(6, 'Within the field')
    r.callout(d.diff, dur=max(2.4, r.speech_left() + 0.2), pad=4, dim=0.2)
    r.finish_speech(0.2)

    # 7 - against the simulated truth
    r.expect('Against the simulated truth: 0.86 px RMS' in diff, True, 'against the truth')
    r.say(7)
    r.callout(d.diff, dur=max(2.4, r.speech_left() + 0.3), pad=4, dim=0.3)
    r.finish_speech(0.3)
    r.zoom(None, 0.8)

    # 8 - Apply to F; the Inspector row
    r.say(8)
    r.cursor_on = True
    r.click(d.b_apply, real=False)
    d.b_apply.click()
    stage.wait_analysis(r)
    r.expect(bool(st.cam(F.uid).calib), True, 'F carries its calibration')
    r.hold(0.4)
    r.callout(d.cam_info, dur=2.2, pad=4)
    r.hold(1.6)
    d.hide()
    r.zoom(None, 0.0)
    stage.film_window(r, w)
    r.crossfade(0.5)
    stage.settle(2)
    page = w.inspector.pages['camera']
    hd = page.s_opt.header
    r.zoom(QRectF(r.wrect(hd).x() - 260, r.wrect(hd).y() - 60, 560, 315), 0.9)
    if not page.s_opt.body.isVisible():
        r.click(hd, QPointF(60, hd.height() / 2), 0.6)
        stage.settle(1)
    cr = r.wrect(page.calib_v)
    r.zoom(QRectF(cr.x() - 260, cr.y() - 170, 560, 315), 0.8)
    print('unit calibration row:', page.calib_v.v.text(), flush=True)
    r.expect(page.calib_v.v.text(), 'SIM0001', "the Inspector's unit calibration row")
    r.callout(page.calib_v, dur=max(2.0, r.speech_left() + 0.3), pad=4)
    r.finish_speech(0.2)
    r.zoom(None, 0.8)

    # 9 - Save; Import of calibrations made elsewhere
    r.say(9)
    d.show()
    stage.film_centred(r, d)
    r.crossfade(0.4)
    sv = r.wrect(d.b_save)
    r.zoom(QRectF(sv.x() - 200, sv.y() - 200, 640, 360), 0.8)
    r.click(d.b_save, real=False)                 # the file dialog cannot be filmed
    r.hold(0.8)
    r.zoom(None, 0.6)
    d.hide()
    stage.film_window(r, w)
    r.crossfade(0.4)
    r.until_word(9, 'Calibrations made elsewhere')
    r.peek_menu('Calibrate', 'Import unit calibrations', hold=0.8)
    imp = ImportCalibDialog(w.ctx, ros_files, w)
    stage.extra.append(imp)
    imp.show()
    imp.resize(1560, 520)                         # wide enough for every column, the match check included (after
                                                  # show: showEvent fits a dialog to the offscreen screen)
    stage.settle(1)
    stage.film_centred(r, imp)
    r.crossfade(0.5)
    r.expect(sum(1 for cb in imp.combos if cb.currentData()), len(st.variant.cameras), 'files matched to cameras')
    r.cursor_on = False
    r.zoom(imp.table, 1.0, margin=40)
    r.hold(max(1.6, r.speech_left() - 0.6))
    r.zoom(None, 0.8)
    imp.hide()
    stage.film_window(r, w)
    r.crossfade(0.5)
    r.finish_speech(1.0)
