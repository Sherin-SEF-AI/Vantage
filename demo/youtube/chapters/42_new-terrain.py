"""42 Road and body motion (episode v020, chapter 2): Analysis ▸ Road and body motion on the sample (6-cam · v3). A
crest added (a 300 m vertical curve 5 m ahead into an 8 % fall): the road ahead in view to 19.8 m against 60 m on flat
ground, the side-on profile and its hover read-out; the radius typed to 150 m: 14.5 m. A speed bump (80 mm × 0.6 m,
2 m ahead): first resolved 13 m out by F, its far side hidden, 9.2 cm of body clearance driving over it; typed 150 mm:
2.2 cm, a warning; undone. A kerb 0.5 m right of the car: the ground each camera loses behind it. The road with the
heat map in the main 3D view. Body motion: the inputs (CG estimated 0.57 m, typical anti-dive, anti-lift, anti-squat,
roll centres), hard braking 0.8 g (pitch +2.25°, F's horizon 36 px up), cornering 0.5 g (roll 2.55°, 5.1°/g), the
lateral slider dragged with its live read-out, the ground figures of the load cases (hard braking: 69.0 cm of stitched
ground shift, 35.4 cm of seam ghost), anti-dive typed to 50 %: pitch +1.92°. The limits and the hooks.

Figures read from the app while recording (r.expect), the same as core.terrain gives on the sample: see the r.expect
calls; every one is a text the page shows (tiles, tables, read-out, findings).
"""

META = {
    'id': 'new-terrain',
    'part': 1,
    'title': 'Road and body motion',
    'promise': 'The rig on a real road: crests, speed bumps, kerbs, braking and cornering.',
    'menu': 'Analysis ▸ Road and body motion…',
}

LINES = [
    # 0
    ("Every coverage figure so far stood the car on flat ground. On the road it meets crests, speed bumps, kerbs and "
     "ramps, and its body pitches and rolls as it brakes and turns, while the cameras were calibrated at rest.",
     "Every coverage figure so far stood the car on flat ground. On the road it meets crests, speed bumps, kerbs, and "
     "ramps, and its body pitches and rolls as it brakes and turns, while the cameras were calibrated at rest."),
    # 1
    ("Analysis ▸ Road and body motion puts the car on a real road. On the left, the road and the body motion; in the "
     "middle, the coverage on this road and the road side on; on the right, what changes against flat ground.",
     "Analysis, road and body motion, puts the car on a real road. On the left, the road and the body motion. In the "
     "middle, the coverage on this road, and the road side on. On the right, what changes against flat ground."),
    # 2
    ("Six primitives build the road: a ramp, a crest, a garage ramp, a speed bump, a kerb and a crowned road. Add a "
     "crest: a 300-metre vertical curve starting 5 metres ahead, into an 8 per cent fall.",
     "Six primitives build the road: a ramp, a crest, a garage ramp, a speed bump, a kerb, and a crowned road. Add a "
     "crest: a three hundred metre vertical curve, starting five metres ahead, into an eight per cent fall."),
    # 3
    ("The road ahead now stays in view to 19.8 metres, against 60 on flat ground. The profile shows why: beyond the "
     "brow the road, in red, would be seen on flat ground, and the brow hides it.",
     "The road ahead now stays in view to nineteen point eight metres, against sixty on flat ground. The profile shows "
     "why: beyond the brow the road, in red, would be seen on flat ground, and the brow hides it."),
    # 4
    ("The radius sets how soon. Type 150 metres, a sharper crest, and the road ahead is in view to only 14.5.",
     "The radius sets how soon. Type a hundred and fifty metres, a sharper crest, and the road ahead is in view to "
     "only fourteen point five."),
    # 5
    ("Add a speed bump, 80 millimetres high and 0.6 metres long, 2 metres ahead. F first resolves it 13 metres out, "
     "its height at least 3 pixels tall; farther, the brow hides it. Its own far side hides a strip of road.",
     "Add a speed bump, eighty millimetres high and zero point six metres long, two metres ahead. F first resolves it "
     "thirteen metres out, its height at least three pixels tall. Farther, the brow hides it. Its own far side hides "
     "a strip of road."),
    # 6
    ("Driving over it, the body clears the bump by 9.2 centimetres. Make it 150 millimetres high: the clearance "
     "falls to 2.2, a warning here and in the Issues list. Undo puts it back.",
     "Driving over it, the body clears the bump by nine point two centimetres. Make it a hundred and fifty "
     "millimetres high: the clearance falls to two point two, a warning here, and in the issues list. Undo puts it "
     "back."),
    # 7
    ("Add a kerb, 120 millimetres high, half a metre right of the car. Cameras on the road compares each camera with "
     "flat ground: MR and QR each lose about a quarter of a square metre behind its face.",
     "Add a kerb, a hundred and twenty millimetres high, half a metre right of the car. Cameras on the road compares "
     "each camera with flat ground: M R and Q R each lose about a quarter of a square metre, behind its face."),
    # 8
    ("The whole coverage analysis now uses the road: heat maps, figures and Issues. In 3D, the car stands on it, "
     "with the heat map on the road, the brow, the bump and the kerb in place.",
     "The whole coverage analysis now uses the road: heat maps, figures, and issues. In 3D, the car stands on it, "
     "with the heat map on the road, the brow, the bump, and the kerb in place."),
    # 9
    ("One switch goes back to flat ground and keeps the road for later; with no primitive, every result is exactly "
     "what it was.",
     "One switch goes back to flat ground, and keeps the road for later. With no primitive, every result is exactly "
     "what it was."),
    # 10
    ("Body motion puts braking, accelerating and cornering through the suspension of the load cases. The weight acts "
     "at the centre of gravity, estimated at 0.57 metres; anti-dive, anti-lift, anti-squat and the roll centres say "
     "how much the linkage takes. All are typical until you enter yours.",
     "Body motion puts braking, accelerating, and cornering through the suspension of the load cases. The weight acts "
     "at the centre of gravity, estimated at zero point five seven metres. Anti-dive, anti-lift, anti-squat, and the "
     "roll centres say how much the linkage takes. All are typical until you enter yours."),
    # 11
    ("Hard braking at 0.8 g pitches the body 2.25 degrees nose down, against the load the cameras were calibrated "
     "at, and moves the horizon 36 pixels up F's image.",
     "Hard braking at zero point eight g pitches the body two point two five degrees nose down, against the load the "
     "cameras were calibrated at, and moves the horizon thirty-six pixels up F's image."),
    # 12
    ("Cornering at half a g rolls it 2.55 degrees: 5.1 degrees per g. The sliders try any mix of braking and "
     "turning, with the read-out live while you drag; the whole study follows when you let go.",
     "Cornering at half a g rolls it two point five five degrees: five point one degrees per g. The sliders try any "
     "mix of braking and turning, with the read-out live while you drag. The whole study follows when you let go."),
    # 13
    ("Each case also goes through the load cases' ground model. Braking hard, the stitched ground shifts 69 "
     "centimetres and two cameras disagree at a seam by 35: record the body's attitude with the data to correct it.",
     "Each case also goes through the load cases' ground model. Braking hard, the stitched ground shifts sixty-nine "
     "centimetres, and two cameras disagree at a seam by thirty-five. Record the body's attitude with the data, to "
     "correct it."),
    # 14
    ("The inputs matter: an anti-dive of 50 per cent instead of the typical 20 takes the braking pitch from 2.25 to "
     "1.92 degrees.",
     "The inputs matter: an anti-dive of fifty per cent, instead of the typical twenty, takes the braking pitch from "
     "two point two five, to one point nine two degrees."),
    # 15
    ("The limits: the road is one-dimensional profiles added together, with no bends or potholes; a bump counts as "
     "seen by its pixel height alone; the tyres are rigid; the body motion is steady, with no dampers. Studies that "
     "build their own ground, like the optimiser, stay on flat ground.",
     "The limits: the road is one dimensional profiles added together, with no bends or potholes. A bump counts as "
     "seen by its pixel height alone. The tyres are rigid. The body motion is steady, with no dampers. Studies that "
     "build their own ground, like the optimiser, stay on flat ground."),
    # 16
    ("The Issues list flags a lifting wheel, the body touching the road, and a sensor within 5 cm of it. Export "
     "writes JSON, with the motion cases and the profile as CSV and PNG. In batch mode, --terrain exits with 0 on this "
     "road, and with 2 once a 200 mm bump would touch the body. The assistant can run the study too.",
     "The issues list flags a lifting wheel, the body touching the road, and a sensor within five centimetres of it. "
     "Export writes Jason, with the motion cases and the profile as C S V and P N G. In batch mode, terrain exits with "
     "zero on this road, and with two once a two hundred millimetre bump would touch the body. The assistant can run "
     "the study too."),
]


def scene(stage, r):
    import numpy as np
    from PyQt6.QtCore import QPointF, QRectF
    from PyQt6.QtWidgets import QStyle, QStyleOptionSlider

    import helpers_tourc as H
    from vantage.core import terrain as TR
    from vantage.ui.workspaces import terrain_ws as TWS

    # The page re-runs the study by itself after an edit while its last run took under 3 s (about 1 s here). The final
    # recording composes 2x frames on the same machine and can slow a run past that, which would stop the re-runs on
    # camera: the recording raises the limit.
    TWS.AUTO_MAX_S = 60.0

    w = stage.window()
    r.win = w
    app = stage.app
    st = w.store

    def pump(n=4):
        for _ in range(n):
            app.processEvents()

    # ---- 0: flat ground, the sample
    vp = stage.vp
    vp.cam.az, vp.cam.el = -38.0, 16.0
    vp.update()
    stage.settle(2)
    r.cursor_on = False
    r.cur = np.array([1180.0, 640.0])
    r.lower_third()
    r.say(0)
    stage.orbit(r, vp, 30, max(4.0, r.line(0) - 0.8), d_el=5)
    r.finish_speech(0.2)

    # ---- 1: the page
    r.say(1)
    r.cursor_on = True
    r.menu('Analysis', 'Road and body motion')
    pg = w.page('terrain')
    ctl = pg.ctl
    r.crossfade(0.4)
    stage.settle(1)

    def done(n):
        res = ctl.result
        return (res is not None and not ctl.busy and not ctl.stale and len(res.features) == n
                and not ctl.timer.isActive())
    r.wait_for(lambda: done(0), timeout=120)
    pump(4)
    r.cursor_on = False
    L = r.wrect(pg.chart).x() - 14
    cen = r.wrect(pg.plan).united(r.wrect(pg.chart))
    rig = r.wrect(pg.tiles['blind']).united(r.wrect(pg.tabs))
    r.until_word(1, 'On the left')
    r.callout((0, 60, L, 1000), '', dur=r.at_word(1, 'In the middle') - r.now(), pad=2, dim=0.3)
    r.until_word(1, 'In the middle')
    r.callout(cen, '', dur=r.at_word(1, 'On the right') - r.now(), pad=2, dim=0.3)
    r.until_word(1, 'On the right')
    r.callout(rig, '', dur=max(1.6, r.speech_left() + 0.2), pad=2, dim=0.3)
    r.finish_speech(0.2)

    # ---- 2: the primitives; a crest
    r.say(2)
    ab = H.union(*[r.wrect(b) for b in pg.add_buttons.values()])
    r.zoom(QRectF(ab.x() - 10, ab.y() - 90, ab.width() + 20, ab.height() + 130), 0.9)
    r.callout(ab, '', dur=max(1.6, r.at_word(2, 'Add a crest') - r.now() - 0.2), pad=4, dim=0.25)
    r.until_word(2, 'Add a crest')
    r.cursor_on = True
    r.click(pg.add_buttons['crest'], None, 0.6)
    r.wait_for(lambda: done(1), timeout=120)
    pump(4)
    f0 = TR.describe(st.project.analysis.terrain['features'][0])
    print('crest:', f0, flush=True)
    r.expect(f0, 'Crest R 300 m into -8 %, 5 m ahead', 'the crest')
    row = pg.rows[0]
    rr = r.wrect(row)
    r.zoom(QRectF(rr.x() - 8, rr.y() - 20, rr.width() + 16, rr.height() + 40), 0.8)
    r.cursor_on = False
    r.callout(row, '', dur=max(1.8, r.speech_left() + 0.2), pad=3, dim=0.25)
    r.finish_speech(0.2)

    # ---- 3: road ahead in view; the profile
    t = pg.tiles['ahead']
    print('ahead tile:', t.value, t.unit, t.note, flush=True)
    r.expect((t.value, t.note), ('19.8', 'flat ground 60.0 m'), 'road ahead in view')
    r.say(3)
    r.zoom(QRectF(r.wrect(t).x() - 20, r.wrect(t).y() - 20, 640, 360), 0.8)
    r.figure('Road ahead in view', 19.8, 'm', 1, dur=max(2.6, r.at_word(3, 'The profile') - r.now()), src=t,
             note='flat ground 60.0 m')
    r.until_word(3, 'The profile')
    pg.reach.buttons['60'].click()
    pump(2)
    ch = r.wrect(pg.chart)
    r.zoom(QRectF(ch.x() - 6, ch.y() - 30, ch.width() + 12, ch.height() + 36), 0.9)
    res = ctl.result
    ref = TR.references(st.project)
    _, x_lo, x_hi, sx, sz, P, box = pg.chart._geometry()
    q = P(ref['front_x'] + 30.0, float(np.interp(30.0, res.profiles['ahead']['d'], res.profiles['ahead']['h'])))
    r.cursor_on = True
    r.hover(pg.chart, q, 0.9)
    tip = pg.chart.tip
    print('profile tip:', tip, flush=True)
    r.hold(max(2.0, r.speech_left() - 0.4))
    r.leave(pg.chart)
    r.cursor_on = False
    r.finish_speech(0.2)

    # ---- 4: a sharper crest
    r.say(4)
    rr = r.wrect(row)
    r.zoom(QRectF(rr.x() - 8, rr.y() - 20, rr.width() + 16, rr.height() + 40), 0.7)
    r.cursor_on = True
    r.enter(row.fields['radius'][0], '150')
    pump()
    if abs(ctl.settings.features[0]['radius'] - 150.0) > 1e-6:      # no commit on Enter offscreen
        row.fields['radius'][0].committed.emit(150.0)
        pump()
    r.wait_for(lambda: done(1) and abs(ctl.result.features[0]['radius'] - 150.0) < 1e-6, timeout=120)
    pump(4)
    r.cursor_on = False
    t = pg.tiles['ahead']
    print('ahead tile:', t.value, t.note, flush=True)
    r.expect(t.value, '14.5', 'road ahead with R 150')
    r.zoom(QRectF(r.wrect(t).x() - 20, r.wrect(t).y() - 20, 640, 360), 0.8)
    r.figure('Road ahead in view', 14.5, 'm', 1, dur=max(2.2, r.speech_left() + 0.3), src=t, start=19.8,
             note='crest R 150 m')
    r.finish_speech(0.2)

    # ---- 5: a speed bump
    r.say(5)
    r.zoom(None, 0.6)
    r.cursor_on = True
    r.click(pg.add_buttons['bump'], None, 0.6)
    r.wait_for(lambda: done(2), timeout=120)
    pump(4)
    f1 = TR.describe(st.project.analysis.terrain['features'][1])
    r.expect(f1, 'Speed bump 80 mm × 0.6 m, 2 m ahead', 'the bump')
    tabs = pg.tabs
    bar = tabs.tabBar()
    r.click(bar, QPointF(bar.tabRect(1).center()), 0.5, real=False)
    tabs.setCurrentIndex(1)
    pump(2)
    tr_ = pg.t_road
    rowt = [tr_.item(1, c).text() for c in range(6)]
    print('bump row:', rowt, flush=True)
    r.expect(rowt[1:6], ['13.0', 'F', 'to bumper', '—', '9.2'], 'the bump in the table')
    r.cursor_on = False
    tb = r.wrect(tabs)
    r.zoom(QRectF(tb.x() - 6, tb.y() - 10, tb.width() + 12, 150), 0.9)
    r.until_word(5, 'F first resolves')
    r.callout(lambda: H.union(H.cell(r, tr_, 1, 1)(), H.cell(r, tr_, 1, 2)()), '', dur=max(2.4, r.at_word(
        5, 'Its own far side') - r.now() - 0.2), pad=2, dim=0.25)
    r.until_word(5, 'Its own far side')
    pg.reach.buttons['15'].click()
    pump(2)
    ch = r.wrect(pg.chart)
    r.zoom(QRectF(ch.x() - 6, ch.y() - 30, ch.width() + 12, ch.height() + 36), 0.8)
    r.hold(max(1.6, r.speech_left() + 0.2))
    r.finish_speech(0.2)

    # ---- 6: clearance; a taller bump; undo
    r.say(6)
    r.zoom(QRectF(tb.x() - 6, tb.y() - 10, tb.width() + 12, 150), 0.8)
    r.callout(H.cell(r, tr_, 1, 5), '', dur=max(1.8, r.at_word(6, 'Make it') - r.now() - 0.2), pad=2, dim=0.25)
    r.until_word(6, 'Make it')
    row_b = pg.rows[1]
    rb = r.wrect(row_b)
    r.zoom(QRectF(rb.x() - 8, rb.y() - 20, rb.width() + 16, rb.height() + 40), 0.6)
    r.cursor_on = True
    r.enter(row_b.fields['height'][0], '150')
    pump()
    if abs(ctl.settings.features[1]['height'] - 0.15) > 1e-9:
        row_b.fields['height'][0].committed.emit(150.0)
        pump()
    r.wait_for(lambda: done(2) and abs(ctl.result.features[1]['height'] - 0.15) < 1e-9, timeout=120)
    pump(4)
    r.cursor_on = False
    tcl = pg.tiles['clear']
    print('clearance tile:', tcl.value, tcl.unit, tcl.sev, flush=True)
    r.expect((tcl.value, tcl.unit, tcl.sev), ('2.2', 'cm', 'warning'), 'clearance over a 150 mm bump')
    warn = next(x for x in pg.findings if x.text.text().startswith('Driving over "Speed bump 150 mm'))
    r.expect(warn.text.text(), 'Driving over "Speed bump 150 mm × 0.6 m, 2 m ahead" the body clears the road by only '
                               '2.2 cm (underbody between the axles).', 'the clearance warning')
    r.zoom(QRectF(r.wrect(tcl).x() - 20, r.wrect(tcl).y() - 20, 640, 360), 0.7)
    r.figure('Body clearance', 2.2, 'cm', 1, dur=2.4, src=tcl, start=9.2, note='a 150 mm bump')
    r.hold(1.6)
    r.until_word(6, 'Undo')
    r.zoom(None, 0.6)
    r.key_combo('Ctrl+Z', dur=1.4)
    w.actions_by_id['undo'].trigger()
    pump(2)

    # ---- 7: a kerb; the cameras on the road (the undone bump's figures come back as the line starts)
    r.say(7)
    r.wait_for(lambda: done(2) and abs(ctl.result.features[1]['height'] - 0.08) < 1e-9, timeout=120)
    pump(4)
    r.expect(pg.tiles['clear'].value, '9.2', 'clearance back')
    r.cursor_on = True
    r.click(pg.add_buttons['kerb'], None, 0.6)
    r.wait_for(lambda: done(3), timeout=120)
    pump(4)
    f2 = TR.describe(st.project.analysis.terrain['features'][2])
    r.expect(f2, 'Kerb 120 mm, 0.5 m right of the car', 'the kerb')
    r.until_word(7, 'Cameras on the road')
    r.click(bar, QPointF(bar.tabRect(2).center()), 0.5, real=False)
    tabs.setCurrentIndex(2)
    pump(2)
    tc = pg.t_cams
    cams = {tc.item(i, 0).text(): tc.item(i, 3).text() for i in range(tc.rowCount())}
    print('hidden per camera:', cams, flush=True)
    r.expect((cams['MR'], cams['QR']), ('0.25', '0.28'), 'MR and QR lose ground behind the kerb')
    r.cursor_on = False
    r.zoom(QRectF(tb.x() - 6, tb.y() - 10, tb.width() + 12, 230), 0.9)
    hhc = tc.horizontalHeader()

    def ccol(c):
        def f():
            x = hhc.sectionViewportPosition(c)
            tl_ = r.wpos(hhc, QPointF(x, 0))
            return QRectF(tl_.x(), tl_.y(), hhc.sectionSize(c), hhc.height() + tc.rowHeight(0) * tc.rowCount())
        return f
    r.callout(ccol(3), '', dur=max(1.8, r.at_word(7, 'M R and Q R') - r.now()), pad=2, dim=0.25)
    r.until_word(7, 'M R and Q R')
    iMR = next(i for i in range(tc.rowCount()) if tc.item(i, 0).text() == 'MR')
    iQR = next(i for i in range(tc.rowCount()) if tc.item(i, 0).text() == 'QR')
    r.callout(lambda: H.union(H.cell(r, tc, iMR)(), H.cell(r, tc, iQR)()), '', dur=max(2.0, r.speech_left() + 0.2),
              pad=2, dim=0.25)
    r.finish_speech(0.2)
    r.zoom(None, 0.7)

    # ---- 8: the road in the main 3D view
    r.say(8)
    pc = r.wrect(pg.plan)
    r.zoom(pc, 0.8, margin=4)
    r.hold(max(1.4, r.at_word(8, 'In 3D') - r.now() - 1.2))
    r.cursor_on = True
    r.menu('View', 'Back to viewports')
    stage.settle(3)
    vp = stage.vp
    vp.cam.az, vp.cam.el = -70.0, 10.0
    vp.cam.dist *= 1.2
    vp.update()
    pump(4)
    r.crossfade(0.5)
    r.zoom(None, 0)
    r.cursor_on = False
    stage.orbit(r, vp, 100, max(4.5, r.speech_left() + 0.8), d_el=10)
    r.finish_speech(0.2)

    # ---- 9: the switch
    r.say(9)
    r.cursor_on = True
    r.menu('Analysis', 'Road and body motion')
    stage.settle(1)
    r.crossfade(0.4)
    en = r.wrect(pg.enabled)
    r.zoom(QRectF(en.x() - 10, en.y() - 80, 420, 200), 0.8)
    r.cursor_on = False
    r.callout(pg.enabled, '', dur=max(2.2, r.speech_left() + 0.2), pad=4, dim=0.25)
    r.finish_speech(0.2)

    # ---- 10: the body-motion inputs
    est = pg.cg_note.text()
    print('cg note:', est, flush=True)
    r.expect(est.startswith('CG height estimated: 0.57 m for the empty car.'), True, 'the CG estimate')
    d = ctl.settings.dynamics
    r.expect((d.anti_dive, d.anti_lift, d.anti_squat, d.rc_front, d.rc_rear), (20.0, 10.0, 15.0, 0.08, 0.2),
             'the typical inputs')
    r.say(10)
    dyn = H.union(*[r.wrect(x) for x in pg.dyn.values()])
    body = H.union(r.wrect(pg.load), dyn, r.wrect(pg.cg_note), r.wrect(pg.drive))
    r.zoom(QRectF(body.x() - 10, body.y() - 150, body.width() + 20, body.height() + 170), 1.0)
    r.until_word(10, 'estimated at')
    r.callout(pg.cg_note, '', dur=2.2, pad=3, dim=0.25)
    r.until_word(10, 'Anti-dive')
    r.callout(dyn, '', dur=max(2.4, r.speech_left() + 0.2), pad=3, dim=0.25)
    r.finish_speech(0.2)

    # ---- 11: hard braking
    r.say(11)
    r.cursor_on = True
    hb = H.button(pg, 'Hard braking')
    r.click(hb, None, 0.6)
    pump(2)
    print('readout:', pg.readout.text(), flush=True)
    ro_lines = pg.readout.text().split('\n')
    r.expect((ro_lines[0].startswith('Pitch +2.25° · roll '), ro_lines[1].startswith('F horizon 36 px up (+2.25°)')),
             (True, True), 'the read-out')
    r.wait_for(lambda: done(3), timeout=120)
    pump(4)
    tbk = pg.tiles['brake']
    print('brake tile:', tbk.value, tbk.unit, tbk.note, flush=True)
    r.expect((tbk.value, tbk.note), ('+2.25', 'F horizon 36 px'), 'the hard braking tile')
    r.cursor_on = False
    ro = r.wrect(pg.readout)
    r.zoom(QRectF(ro.x() - 10, ro.y() - 90, max(ro.width() + 20, 420), ro.height() + 110), 0.8)
    r.callout(pg.readout, '', dur=max(2.0, r.at_word(11, 'and moves the horizon') - r.now()), pad=3, dim=0.25)
    r.until_word(11, 'and moves the horizon')
    r.zoom(QRectF(r.wrect(tbk).x() - 20, r.wrect(tbk).y() - 20, 640, 360), 0.7)
    r.figure('F horizon, hard braking', 36, 'px', 0, dur=max(2.2, r.speech_left() + 0.4), src=tbk, note='up the image')
    r.finish_speech(0.2)

    # ---- 12: cornering; the slider
    tcn = pg.tiles['corner']
    print('corner tile:', tcn.value, tcn.note, flush=True)
    r.expect((tcn.value, tcn.note), ('+2.55', '5.1°/g roll gradient'), 'cornering')
    r.say(12)
    r.figure('Roll, cornering 0.5 g', 2.55, '°', 2, dur=max(2.4, r.at_word(12, 'The sliders') - r.now()), src=tcn,
             note='5.1°/g')
    r.until_word(12, 'The sliders')
    sl = pg.sl_lat
    sg = r.wrect(sl).united(r.wrect(pg.sl_long)).united(r.wrect(pg.readout))
    r.zoom(QRectF(sg.x() - 120, sg.y() - 20, sg.width() + 140, sg.height() + 40), 0.8)

    def slider_x(s, v):
        opt = QStyleOptionSlider()
        s.initStyleOption(opt)
        gr = s.style().subControlRect(QStyle.ComplexControl.CC_Slider, opt, QStyle.SubControl.SC_SliderGroove, s)
        hr = s.style().subControlRect(QStyle.ComplexControl.CC_Slider, opt, QStyle.SubControl.SC_SliderHandle, s)
        span = gr.width() - hr.width()
        return gr.x() + QStyle.sliderPositionFromValue(s.minimum(), s.maximum(), v, span) + hr.width() / 2
    y = sl.height() / 2
    r.cursor_on = True
    H.drag(r, sl, QPointF(slider_x(sl, sl.value()), y), QPointF(slider_x(sl, 45), y), dur=1.8)
    pump(2)
    if sl.value() != 45:
        sl.setValue(45)
        pg._commit_sliders()
    print('lat slider:', sl.value(), pg.readout.text(), flush=True)
    r.hold(0.4)
    H.drag(r, sl, QPointF(slider_x(sl, sl.value()), y), QPointF(slider_x(sl, 0), y), dur=1.2, approach=0.3)
    pump(2)
    if sl.value() != 0:                            # back to straight-line braking
        sl.setValue(0)
        pg._commit_sliders()
    r.expect((ctl.settings.long_g, ctl.settings.lat_g), (-0.8, 0.0), 'the sliders back at hard braking')
    r.cursor_on = False
    r.hold(max(0.6, r.speech_left() + 0.2))
    r.finish_speech(0.1)

    # ---- 13: the ground figures of each case
    r.say(13)
    r.wait_for(lambda: done(3), timeout=120)
    pump(4)
    r.click(bar, QPointF(bar.tabRect(0).center()), 0.5, real=False)
    tabs.setCurrentIndex(0)
    pump(2)
    tm = pg.t_motion
    k_hb = next(i for i in range(tm.rowCount()) if tm.item(i, 0).text() == 'Hard braking')
    hbrow = [tm.item(k_hb, c).text() for c in range(8)]
    print('hard braking row:', hbrow, flush=True)
    r.expect((hbrow[6], hbrow[7]), ('69.0', '35.4'), 'hard braking ground shift and ghost')
    r.zoom(QRectF(tb.x() - 6, tb.y() - 10, tb.width() + 12, 260), 0.9)
    hm = tm.horizontalHeader()

    def mcol(c0, c1):
        def f():
            x0 = hm.sectionViewportPosition(c0)
            x1 = hm.sectionViewportPosition(c1) + hm.sectionSize(c1)
            tl_ = r.wpos(hm, QPointF(x0, 0))
            return QRectF(tl_.x(), tl_.y(), x1 - x0, hm.height() + tm.rowHeight(0) * tm.rowCount())
        return f
    r.callout(mcol(5, 7), '', dur=max(2.0, r.at_word(13, 'Braking hard') - r.now()), pad=2, dim=0.25)
    r.until_word(13, 'Braking hard')
    r.callout(H.cell(r, tm, k_hb), '', dur=max(2.0, r.at_word(13, 'Record the body') - r.now()), pad=2, dim=0.25)
    ghost = next(x for x in pg.findings if 'cases exceed the load cases' in x.text.text())
    print('ghost finding:', ghost.text.text(), flush=True)
    r.expect(ghost.text.text().startswith('Against the calibration at "Rig only, no occupants" the stitched ground '
                                          'shifts up to 75.2 cm (Braking in a bend) and the seams ghost up to 35.4 cm '
                                          '(Hard braking); 7 of 8 cases exceed'), True, 'the ghost finding')
    r.until_word(13, 'Record the body')
    fz = r.wrect(pg.find_host.parentWidget())
    r.zoom(QRectF(fz.x() - 6, fz.y() - 30, fz.width() + 12, min(fz.height(), 260) + 40), 0.7)
    r.callout(ghost, '', dur=max(2.0, r.speech_left() + 0.2), pad=3, dim=0.25)
    r.finish_speech(0.2)
    r.zoom(None, 0.7)

    # ---- 14: anti-dive 50 %
    r.say(14)
    r.zoom(QRectF(dyn.x() - 10, dyn.y() - 40, dyn.width() + 20, dyn.height() + 60), 0.7)
    r.cursor_on = True
    r.enter(pg.dyn['anti_dive'], '50')
    pump()
    if abs(ctl.settings.dynamics.anti_dive - 50.0) > 1e-9:
        pg.dyn['anti_dive'].committed.emit(50.0)
        pump()
    r.wait_for(lambda: done(3) and abs(ctl.result.settings.dynamics.anti_dive - 50.0) < 1e-9, timeout=120)
    pump(4)
    r.cursor_on = False
    tbk = pg.tiles['brake']
    print('brake tile now:', tbk.value, tbk.note, flush=True)
    r.expect(tbk.value, '+1.92', 'pitch with 50 % anti-dive')
    r.zoom(QRectF(r.wrect(tbk).x() - 20, r.wrect(tbk).y() - 20, 640, 360), 0.7)
    r.figure('Pitch, hard braking', 1.92, '°', 2, dur=max(2.4, r.speech_left() + 0.4), src=tbk, start=2.25,
             note='anti-dive 50 %')
    r.finish_speech(0.2)
    r.zoom(None, 0.7)

    # ---- 15: the limits (the page's assumptions)
    r.say(15)
    asr = r.wrect(pg.assume)
    r.zoom(QRectF(asr.x() - 8, asr.y() - 10, asr.width() + 16, asr.height() + 20), 1.0)
    r.hold(max(2.0, r.speech_left() - 1.2))
    r.zoom(None, 0.8)
    r.finish_speech(0.2)

    # ---- 16: issues, export, batch (real runs, made now and not filmed), assistant
    import copy
    import os
    import site
    import sys
    import studio
    home = stage.home
    bin_ = os.path.join(home, 'bin')
    work = os.path.join(home, 'projects')
    os.makedirs(bin_, exist_ok=True)
    os.makedirs(work, exist_ok=True)
    with open(os.path.join(bin_, 'vantage'), 'w') as fh_:
        fh_.write(f'#!/bin/sh\nexec {sys.executable} {os.path.join(studio.ROOT, "run_vantage.py")} "$@"\n')
    os.chmod(os.path.join(bin_, 'vantage'), 0o755)
    env = {k: v for k, v in os.environ.items() if k not in ('QT_SCALE_FACTOR',)}
    env.update(PATH=bin_ + os.pathsep + env.get('PATH', ''), HOME=home, PYTHONUSERBASE=site.USER_BASE,
               QT_QPA_PLATFORM='offscreen')
    snap = st.project.snapshot()
    snap.save(os.path.join(work, 'road.vantage'), b'')
    tall = st.project.snapshot()
    ter = copy.deepcopy(tall.analysis.terrain)
    ter['features'][1]['height'] = 0.2
    tall.analysis.terrain = ter
    tall.save(os.path.join(work, 'road-200mm-bump.vantage'), b'')
    c1, l1, _ = H.run_timed('vantage road.vantage --terrain road.json', work, env, home)
    c2, l2, _ = H.run_timed('vantage road-200mm-bump.vantage --terrain bump.json', work, env, home)
    for c_, l_ in ((c1, l1), (c2, l2)):
        print('batch exit', c_, flush=True)
        for _, ln in l_:
            print('   ', ln, flush=True)
    r.expect((c1, c2), (0, 2), 'the batch exit codes')
    r.expect(any('would touch the road' in ln for _, ln in l2), True, 'the touch finding in batch')
    noise = (lambda ln: not ln.startswith(('This plugin does not', '[ WARN', 'qt.')))
    script = ([('cmd', 'vantage road.vantage --terrain road.json')] + H.timed_output(l1, keep=noise) +
              [('cmd', 'echo $?'), ('out', str(c1), 'ok'), ('wait', 0.6),
               ('cmd', 'vantage road-200mm-bump.vantage --terrain bump.json')] + H.timed_output(l2, keep=noise) +
              [('cmd', 'echo $?'), ('out', str(c2), 'warn')])
    r.say(16)
    r.cursor_on = True
    r.move_to(pg.b_export, None, 0.8)
    r.callout(pg.b_export, '', dur=2.0, pad=4, dim=0.25)
    r.hold(1.6)
    r.cursor_on = False
    r.until_word(16, 'In batch mode')
    span = H.terminal(r, script, title='~/projects')
    r.crossfade(0.5)
    r.hold(max(span + 0.8, r.speech_left() + 0.3))
    r.finish_speech(1.0)
