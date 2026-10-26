"""18 Load cases and suspension: Analysis ▸ Load cases on the sample's data rig. The suspension (typical values for a
compact sedan scaled to the model, labelled so), the loads (seats, boot, roof, equipment, the rig's own parts), the
five cases with corner drops, pitch, roll and bump stops; the driver alone rolls the body; the full load moves the rear
camera most; Run adds the ground shift and the seam ghosting; calibrating at the driver + operator load instead.

Figures read from the app while recording (r.expect): the rig's parts 5.6 kg; Driver: roll -0.26°, largest move
16.2 mm; Full: rear corners drop 75.3 / 75.5 mm, R moves 84.9 mm, the body turns 1.16°; after Run (calibrated at
"Rig only, no occupants"): Full ghosts 34.8 cm; calibrated at "Driver + operator" and run again: Full ghosts 25.8 cm.
"""

META = {
    'id': 'loads',
    'part': 3,
    'title': 'Load cases and suspension',
    'promise': 'Crew, luggage and equipment move every sensor. By how much?',
    'menu': 'Analysis ▸ Load cases',
}

LINES = [
    ("A data-collection car is calibrated at one load, then driven with a crew, equipment and luggage aboard. Load "
     "cases show how far that moves every camera and sensor.",
     "A data collection car is calibrated at one load, then driven with a crew, equipment, and luggage aboard. Load "
     "cases show how far that moves every camera and sensor."),
    ("The body sits on four corner springs, each in series with its tyre, with a progressive bump stop. These are "
     "typical values for a compact sedan, and the dialog says so: replace them with your vehicle's data.",
     "The body sits on four corner springs, each in series with its tyre, with a progressive bump stop. These are "
     "typical values for a compact sedan, and the dialog says so: replace them with your vehicle's data."),
    ("The loads are masses at points: the seats, the boot, the roof and the recording equipment. The rig's own parts "
     "are estimated from their modules and housings: 5.6 kilograms here.",
     "The loads are masses at points: the seats, the boot, the roof, and the recording equipment. The rig's own parts "
     "are estimated from their modules and housings: five point six kilograms here."),
    ("Five cases, from the rig alone to five occupants and luggage: each corner's drop, the body's pitch and roll, "
     "and the bump stops touched.",
     "Five cases, from the rig alone, to five occupants and luggage: each corner's drop, the body's pitch and roll, "
     "and the bump stops touched."),
    ("The driver alone sits off-centre: the body rolls 0.26 degrees, and against the calibration load the cameras "
     "move by up to 16 mm.",
     "The driver alone sits off centre: the body rolls zero point two six degrees, and against the calibration load, "
     "the cameras move by up to sixteen millimetres."),
    ("Fully loaded, the rear drops 75 mm onto its bump stops: the rear camera moves 85 mm, and the whole body turns "
     "by 1.16 degrees.",
     "Fully loaded, the rear drops seventy-five millimetres, onto its bump stops: the rear camera moves eighty-five "
     "millimetres, and the whole body turns by one point one six degrees."),
    ("Run adds the ground figures of every case: how far the stitched ground shifts, and how far two cameras "
     "disagree at each seam.",
     "Run adds the ground figures of every case: how far the stitched ground shifts, and how far two cameras disagree "
     "at each seam."),
    ("Calibrated with the rig alone, the full load ghosts the seams by 35 cm. The findings say it plainly: calibrate "
     "at the load you record with, or log the load on every drive.",
     "Calibrated with the rig alone, the full load ghosts the seams by thirty-five centimetres. The findings say it "
     "plainly: calibrate at the load you record with, or log the load on every drive."),
    ("Calibrate with the driver and the operator aboard instead, and run again: the worst ghosting falls to 26 cm, "
     "with every move measured from there.",
     "Calibrate with the driver and the operator aboard instead, and run again: the worst ghosting falls to twenty-six "
     "centimetres, with every move measured from there."),
    ("The suspension and the cases are saved with the project; the batch mode and the assistant run them too.",
     "The suspension and the cases are saved with the project. The batch mode, and the assistant, run them too."),
]


def scene(stage, r):
    import numpy as np
    from PyQt6.QtCore import QPoint, QPointF, QRectF

    w = stage._rig_window()
    r.win = w
    r.cursor_on = False
    vp = stage.vp
    vp.cam.az, vp.cam.el = -36.0, 20.0
    w.viewarea.frame_all()
    stage.settle(2)
    r.cur = np.array([1200.0, 620.0])

    def row_of(t, start):
        return next(i for i in range(t.rowCount()) if t.item(i, 0) and t.item(i, 0).text().startswith(start))

    def cell(t, row, col):
        return r.local_rect(t.viewport(), QRectF(t.visualItemRect(t.item(row, col))))

    def click_row(t, row, col=0):
        rect = t.visualItemRect(t.item(row, col))
        r.click(t.viewport(), QPointF(rect.center().x(), rect.center().y()), 0.6)
        stage.settle(1)

    # ---- 0: the opener; the dialog from its menu
    r.lower_third()
    r.say(0)
    stage.orbit(r, vp, 24, 3.0, d_el=3)
    r.cursor_on = True
    # the menu walked like a user; the dialog is then opened by its action's own method and sized: the dialog fits
    # itself to the (offscreen) screen when it shows, which at the recording's 2x scale squeezes it
    r.peek_menu('Analysis', 'Load cases', hold=0.2)
    r.ripple()
    d = w.load_cases()
    d.resize(1240, 800)
    d.move(w.mapToGlobal(QPoint((w.width() - 1240) // 2, (w.height() - 800) // 2)))
    stage.extra.append(d)
    for _ in range(6):
        stage.app.processEvents()
    print('load-cases dialog:', d.size().width(), 'x', d.size().height(), flush=True)
    r.expect((d.width(), d.height()), (1240, 800), 'the dialog size')
    r.crossfade(0.4)
    r.finish_speech(0.1)
    bar = d.tabs.tabBar()

    # ---- 1: the suspension, typical values labelled typical
    r.say(1)
    r.click(bar, bar.tabRect(2).center(), dur=0.6)
    stage.settle(1)
    top = r.wrect(d.fields['kerb_mass']).united(r.wrect(d.stable))
    r.zoom(QRectF(top.x() - 80, top.y() - 30, top.width() + 160, top.height() + 40), 1.0, wait=False)
    r.move_to(d.stable.viewport(), QPointF(d.stable.viewport().width() * 0.4, 40), 0.8)
    r.until_word(1, 'typical values')
    tl = r.wrect(d.typical_lbl)
    r.zoom(QRectF(tl.x() - 10, tl.y() - 90, tl.width() * 0.75, 260), 0.8, wait=False)
    r.callout(d.typical_lbl, '', dur=max(2.4, r.speech_left() + 0.4), pad=4, dim=0.3)
    r.finish_speech(0.2)

    # ---- 2: the loads and the rig's own masses
    r.say(2)
    r.click(bar, bar.tabRect(1).center(), dur=0.6)
    stage.settle(1)
    r.zoom(r.wrect(d.itable).united(r.wrect(d.rtable)), 1.0, margin=12, wait=False)
    r.move_to(d.itable.viewport(), QPointF(d.itable.viewport().width() * 0.5, 60), 0.8)
    r.until_word(2, "The rig's own parts")
    r.expect(d.rig_lbl.text(), '13 parts · 5.6 kg', "the rig's estimated mass")
    r.callout(d.rtable, '', dur=2.2, pad=2, dim=0.25)
    r.until_word(2, 'five point six')
    r.figure("The rig's own parts", 5.6, 'kg', 1, dur=max(2.4, r.speech_left() + 0.6), src=d.rig_lbl)
    r.finish_speech(0.2)
    r.zoom(None, 0.8, wait=False)

    # ---- 3: the cases
    r.say(3)
    r.click(bar, bar.tabRect(0).center(), dur=0.6)
    stage.settle(1)
    ct = d.ctable
    cr = r.wrect(ct)
    r.zoom(QRectF(cr.x(), cr.y() - 50, cr.width() * 0.72, cr.height() + 60), 1.0)
    r.until_word(3, "each corner's drop")
    r.callout(lambda: QRectF(cell(ct, 0, 2)()).united(cell(ct, ct.rowCount() - 1, 5)()), '', dur=1.8, pad=3, dim=0.2)
    r.until_word(3, 'pitch and roll')
    r.callout(lambda: QRectF(cell(ct, 0, 6)()).united(cell(ct, ct.rowCount() - 1, 7)()), '', dur=1.6, pad=3, dim=0.2)
    r.until_word(3, 'bump stops')
    r.callout(lambda: QRectF(cell(ct, 0, 8)()).united(cell(ct, ct.rowCount() - 1, 8)()), '',
              dur=max(1.6, r.speech_left() + 0.3), pad=3, dim=0.2)
    r.finish_speech(0.1)

    # ---- 4: the driver alone
    r.say(4)
    i_drv = row_of(ct, 'Driver')
    r.expect(ct.item(i_drv, 0).text(), 'Driver', 'the driver row')
    r.expect(ct.item(i_drv, 7).text(), '−0.26', "the driver's roll")
    r.expect(ct.item(i_drv, 9).text(), '16.2', "the driver's largest move")
    r.zoom(QRectF(cr.x(), cr.y() - 50, cr.width(), cr.height() + 60), 0.8)
    click_row(ct, i_drv)
    r.callout(cell(ct, i_drv, 7), '', dur=2.2, pad=3, dim=0.2)
    r.until_word(4, 'the cameras move')
    r.figure('Largest move · driver', float(ct.item(i_drv, 9).text()), 'mm', 1, dur=max(2.6, r.speech_left() + 0.6),
             src=cell(ct, i_drv, 9), note='against the rig alone')
    r.finish_speech(0.2)

    # ---- 5: the full load
    r.say(5)
    i_full = row_of(ct, 'Full')
    r.expect(f'{ct.item(i_full, 4).text()}/{ct.item(i_full, 5).text()}', '75.3/75.5', 'rear drops at full load')
    r.expect(ct.item(i_full, 10).text(), '1.16', 'the body turns at full load')
    click_row(ct, i_full)
    r.callout(lambda: QRectF(cell(ct, i_full, 4)()).united(cell(ct, i_full, 5)()), '', dur=2.2, pad=3, dim=0.2)
    dt = d.dtable
    i_r = row_of(dt, 'R ')
    r.expect(dt.item(i_r, 4).text(), '84.9', "R's move at full load")
    r.until_word(5, 'the rear camera moves')
    dr = r.wrect(dt)
    r.zoom(QRectF(dr.x() - 10, dr.y() - 50, dr.width() + 20, min(dr.height(), 300) + 60), 0.8)
    r.callout(lambda: QRectF(cell(dt, i_r, 0)()).united(cell(dt, i_r, 7)()), '', dur=2.2, pad=3, dim=0.2)
    r.until_word(5, 'the whole body')
    r.zoom(QRectF(cr.x(), cr.y() - 50, cr.width(), cr.height() + 60), 0.8)
    r.callout(cell(ct, i_full, 10), '', dur=max(1.8, r.speech_left() + 0.3), pad=3, dim=0.2)
    r.finish_speech(0.2)

    # ---- 6: Run: the ground figures
    r.say(6)
    r.zoom(None, 0.8, wait=False)
    r.click(d.b_run, None, 0.7)
    r.wait_for(lambda: not d.busy and d.sres is not None, timeout=240)
    stage.settle(1)
    r.hold(0.4)
    r.zoom(QRectF(cr.x() + cr.width() * 0.45, cr.y() - 50, cr.width() * 0.55, cr.height() + 60), 1.0)
    r.callout(lambda: QRectF(cell(ct, 0, 12)()).united(cell(ct, ct.rowCount() - 1, 13)()), '',
              dur=max(2.0, r.speech_left() + 0.3), pad=3, dim=0.2)
    r.finish_speech(0.2)

    # ---- 7: the full load ghosts 35 cm; the findings
    r.say(7)
    ghost = ct.item(i_full, 13).text()
    print('ghost at full load (calibrated with the rig alone):', ghost, flush=True)
    r.expect(ghost, '34.8', 'ghost at full load')
    r.figure('Seam ghost · full load', float(ghost), 'cm', 1, dur=3.0, src=cell(ct, i_full, 13),
             note='calibrated with the rig alone')
    r.hold(3.0)
    r.until_word(7, 'The findings')
    fr = r.wrect(d.find_lbl)
    r.zoom(QRectF(fr.x() - 10, fr.y() - 20, fr.width() + 20, fr.height() + 40), 1.0)
    r.callout(d.find_lbl, '', dur=max(2.0, r.speech_left() + 0.4), pad=4, dim=0.25)
    r.finish_speech(0.2)

    # ---- 8: calibrated at driver + operator
    r.say(8)
    r.zoom(QRectF(r.wrect(d.cal_combo).x() - 140, r.wrect(d.cal_combo).y() - 60, 760, 300), 0.8)
    r.choose(d.cal_combo, d.cal_combo.findText('Driver + operator'))
    stage.settle(1)
    r.zoom(None, 0.8, wait=False)
    r.click(d.b_run, None, 0.7)
    r.wait_for(lambda: not d.busy and d.sres is not None and bool(ct.item(i_full, 13).text()), timeout=240)
    stage.settle(1)
    ghost2 = ct.item(i_full, 13).text()
    print('ghost at full load (calibrated with driver + operator):', ghost2, flush=True)
    r.expect(ghost2, '25.8', 'ghost at full load, calibrated with driver + operator')
    r.zoom(QRectF(cr.x() + cr.width() * 0.45, cr.y() - 50, cr.width() * 0.55, cr.height() + 60), 1.0)
    r.figure('Seam ghost · full load', float(ghost2), 'cm', 1, dur=max(3.0, r.speech_left() + 0.8),
             src=cell(ct, i_full, 13), start=34.8, note='calibrated with driver + operator')
    r.finish_speech(0.3)

    # ---- 9: saved with the project; batch and assistant
    r.say(9)
    r.zoom(None, 0.9)
    r.hold(max(0.8, r.speech_left() - 2.6))
    from PyQt6.QtWidgets import QPushButton
    b_close = next(b for b in d.findChildren(QPushButton) if b.text() == 'Close')
    r.click(b_close, None, 0.6, real=False)
    d.hide()
    r.crossfade(0.5)
    r.cursor_on = False
    stage.orbit(r, vp, 30, max(2.0, r.speech_left() + 0.8), d_el=2)
    r.finish_speech(0.8)
