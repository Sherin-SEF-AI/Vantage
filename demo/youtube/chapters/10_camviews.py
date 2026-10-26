"""10 Camera views and the stitched bird's-eye: View ▸ Camera views (every camera's simulated image through its lens),
the stitched bird's-eye with its seams, a seam dragged (the seam table follows) and undone, standing objects that
vanish at the seams, Suggest seams and Apply, the object heights, and the surround-view exports.

Figures read from the app while recording (r.expect): six camera images; seams QR|MR, MR|F, F|ML, ML|QL, QL|R, R|QR;
a 0.9 m child vanishes on 1.22 m² (Coverage panel); Suggest seams: "3 seams · vanish area −9 %"; after Apply 1.18 m².
"""

META = {
    'id': 'camviews',
    'part': 2,
    'title': "Camera views and the stitched bird's-eye",
    'promise': 'What every lens sees, stitched, and where standing people vanish at the seams.',
    'menu': 'View ▸ Camera views, bird’s-eye, bowl  (Ctrl+Shift+B)',
}

LINES = [
    ("View ▸ Camera views shows every camera's simulated image side by side, through its real lens model: the fisheye "
     "curve, the image circle, and the body tinted in the camera's colour.",
     "View, camera views, shows every camera's simulated image side by side, through its real lens model: the fisheye "
     "curve, the image circle, and the body tinted in the camera's colour."),
    ("On each image: the ground grid, the region of interest, and the seams where it hands over to its neighbours.",
     "On each image: the ground grid, the region of interest, and the seams where it hands over to its neighbours."),
    ("On the right, the stitched bird's-eye view: which camera's image the display shows on each patch of ground, "
     "split at the seams, as the ECU's remap tables will stitch it.",
     "On the right, the stitched bird's-eye view: which camera's image the display shows on each patch of ground, "
     "split at the seams, as the E C U's remap tables will stitch it."),
    ("Drag a seam's handle to move it. The table below follows: each seam's angle, its blend width, and the pixel "
     "density where the cameras meet.",
     "Drag a seam's handle to move it. The table below follows: each seam's angle, its blend width, and the pixel "
     "density where the cameras meet."),
    ("A bird's-eye view assumes everything lies flat. A standing object is smeared away from each camera, and near a "
     "seam both smears can fall on the other camera's side: the object vanishes from the display, though both "
     "cameras see it.",
     "A bird's-eye view assumes everything lies flat. A standing object is smeared away from each camera, and near a "
     "seam, both smears can fall on the other camera's side: the object vanishes from the display, though both "
     "cameras see it."),
    ("On this layout, a 0.9-metre child can vanish on 1.22 square metres of the region.",
     "On this layout, a zero point nine metre child can vanish on one point two two square metres of the region."),
    ("Suggest seams searches each seam within 15 degrees for angles that hide it on less ground, without more ground "
     "below the minimum density: three seams, 9 per cent less.",
     "Suggest seams searches each seam within fifteen degrees, for angles that hide it on less ground, without more "
     "ground below the minimum density: three seams, nine per cent less."),
    ("Apply moves them as one undo step, and the region where the child can vanish shrinks to 1.18 square metres.",
     "Apply moves them as one undo step, and the region where the child can vanish shrinks to one point one eight "
     "square metres."),
    ("The object is a choice: a half-metre pole, a child, an adult, or any height.",
     "The object is a choice: a half metre pole, a child, an adult, or any height."),
    ("For the ECU, the remap tables, a reference stitcher in C++ and Python, and the objects map are exports.",
     "For the E C U, the remap tables, a reference stitcher in C plus plus and Python, and the objects map, are "
     "exports."),
]


def scene(stage, r):
    import time

    from PyQt6.QtCore import QPoint, QPointF, QRectF

    from helpers_tour_a import drag_sent
    from vantage.ui.dialogs.export_dialog import ExportDialog

    w = stage.window()
    r.win = w
    st = w.store
    app = stage.app
    rows = w.bottom.coverage.rows

    def shown(k):
        return rows[k].v.text()

    def pump(n=4):
        for _ in range(n):
            app.processEvents()
    ctl = w.ctx.stitched
    r.expect(shown('Object vanishes at seams'), '1.22', 'object vanishes at seams')

    # ---- 0: View ▸ Camera views
    r.cursor_on = True
    r.lower_third()
    r.say(0)
    r.hold(0.8)
    r.menu('View', 'Camera views')
    pg = w._page_widgets['camviews']
    stage.settle(5)
    t = time.time()
    while ctl.result is None and time.time() - t < 60:
        stage.app.processEvents()
        time.sleep(0.02)
    pump(10)
    r.crossfade(0.6)
    r.expect(len(pg.views), 6, 'camera images')
    heads = [pg.table.horizontalHeaderItem(j).text() for j in range(pg.table.columnCount())]
    print('seams:', heads, flush=True)
    r.expect(heads, ['QR | MR', 'MR | F', 'F | ML', 'ML | QL', 'QL | R', 'R | QR'], 'the seams')
    r.cursor_on = False
    r.hold(0.8)
    grid = r.wrect(pg.grid_w)
    r.zoom(grid, 1.0, margin=8)
    r.finish_speech(0.2)

    # ---- 1: one image close up
    r.say(1)
    r.zoom(pg.views[0], 1.0, margin=10)
    r.finish_speech(0.2)
    r.zoom(None, 0.8)

    # ---- 2: the stitched bird's-eye
    r.say(2)
    r.hold(0.4)
    r.zoom(pg.bev, 1.0, margin=8)
    r.finish_speech(0.2)

    # ---- 3: drag a seam handle; the table follows
    r.say(3)
    r.cursor_on = True
    res = w.ctx.analysis.result
    k = heads.index('ML | QL')
    s = res.seams[k]
    a, b = pg.bev._seam_segment(res, s.angle)
    q0 = pg.bev.w2s(*b)
    import math
    cx, cy = res.centre
    ang = math.radians(s.angle + 9.0)
    rad = math.hypot(b[0] - cx, b[1] - cy)
    q1 = pg.bev.w2s(cx + rad * math.cos(ang), cy + rad * math.sin(ang))
    angle0 = pg.table.item(0, k).text()
    drag_sent(r, pg.bev, q0, q1, 1.8)
    stage.wait_analysis(r)
    angle1 = pg.table.item(0, k).text()
    print('ML | QL angle:', angle0, '->', angle1, flush=True)
    r.zoom(None, 0.7)
    tb = r.wrect(pg.table)
    r.zoom(QRectF(tb.x() - 10, tb.y() - 40, tb.width() + 20, tb.height() + 50), 0.9)
    hh = pg.table.horizontalHeader()

    def col():
        x = hh.sectionViewportPosition(k)
        tl = r.wpos(hh, QPointF(x, 0))
        return QRectF(tl.x(), tl.y(), hh.sectionSize(k), hh.height() + pg.table.rowHeight(0) * 5)
    r.callout(col, '', dur=max(2.2, r.speech_left()), pad=2, dim=0.25)
    r.finish_speech(0.2)

    # ---- 4: standing objects
    r.say(4)
    r.zoom(None, 0.7, wait=False)
    st.undo.undo()                                     # back to the sample's seams
    stage.wait_analysis(r)
    r.wait_for(lambda: ctl.result is not None and not ctl._busy and not ctl.timer.isActive(), timeout=60)
    pump()
    r.expect(shown('Object vanishes at seams'), '1.22', 'object vanishes, seams as before')
    r.click(pg.obj_check, None, 0.7)
    if not pg.obj_check.isChecked():
        pg.obj_check.setChecked(True)
    pump()
    r.cursor_on = False
    r.zoom(pg.bev, 1.0, margin=8)
    r.hold(max(2.0, r.speech_left() - 1.0))
    r.zoom(None, 0.8)
    r.finish_speech(0.2)

    # ---- 5: 1.22 m²
    r.say(5)
    r.figure('Object vanishes at seams', float(shown('Object vanishes at seams')), 'm²', 2,
             dur=max(3.2, r.speech_left() + 0.6), src=rows['Object vanishes at seams'], note='0.9 m child')
    r.finish_speech(0.4)

    # ---- 6: Suggest seams
    r.say(6)
    r.cursor_on = True
    r.click(pg.b_suggest, None, 0.7, real=False)
    ctl.suggest()
    r.wait_for(lambda: not ctl.advising and ctl.advice is not None, timeout=120)
    note = pg.adv_note.text()
    print('advice:', note, '|', pg.adv_note.toolTip(), flush=True)
    r.expect(note, '3 seams · vanish area −9 %', 'the advice')
    tb = r.wrect(pg.table)
    r.zoom(QRectF(tb.x() - 10, tb.y() - 70, tb.width() + 20, tb.height() + 80), 0.9)
    r.callout(pg.adv_note, '', dur=max(2.2, r.speech_left() + 0.3), pad=4, dim=0.25)
    r.finish_speech(0.2)

    # ---- 7: Apply
    r.say(7)
    stage.mark_result()
    r.click(pg.b_apply, None, 0.6, real=False)
    ctl.apply_advice()
    stage.wait_analysis(r)
    r.wait_for(lambda: ctl.result is not None and not ctl._busy and not ctl.timer.isActive(), timeout=60)
    pump()
    r.zoom(None, 0.7)
    r.expect(shown('Object vanishes at seams'), '1.18', 'object vanishes after Apply')
    r.cursor_on = False
    r.figure('Object vanishes at seams', float(shown('Object vanishes at seams')), 'm²', 2,
             dur=max(3.0, r.speech_left() + 0.6), src=rows['Object vanishes at seams'], start=1.22, note='seams applied')
    r.finish_speech(0.4)

    # ---- 8: the object's height
    r.say(8)
    r.cursor_on = True
    r.click(pg.obj_combo, None, 0.6, real=False)
    pg.obj_combo.showPopup()
    r.hold(max(1.6, r.speech_left() - 0.2))
    pg.obj_combo.hidePopup()
    r.hold(0.2)
    r.finish_speech(0.1)

    # ---- 9: the exports
    r.say(9)
    r.key_combo('Ctrl+E', dur=1.2)
    d = ExportDialog(w.ctx, w, preselect='bev_stitcher')
    stage.extra.append(d)
    stage.tilde(d.out)
    d.show()
    pump(10)
    d.move(w.mapToGlobal(QPoint((w.width() - d.width()) // 2, (w.height() - d.height()) // 2)))
    pump()
    r.crossfade(0.4)
    r.zoom(d, 0.8, margin=20)
    r.hold(max(1.6, r.speech_left() - 0.6))
    d.hide()
    r.zoom(None, 0.0)
    r.crossfade(0.4)
    r.cursor_on = False
    r.finish_speech(1.0)
