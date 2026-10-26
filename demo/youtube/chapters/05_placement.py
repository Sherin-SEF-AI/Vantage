"""05 Placing cameras: Place on surface with Alt-click (a mirrored pair on the front bumper corners), the move handles
(an arrow dragged with the read-out, then Esc), the rotate rings, dragging a camera along the body (surface snap, the
zone in the read-out), Camera ▸ Aim ▸ At ground point with the linked partner mirroring it, a number field taking a
value with a unit, scrubbing a field, the Cameras table, and the coordinate frame and units the values are shown in.

Every interaction is a real event sent to the 3D view or the Inspector. Figures read from the app while recording
(r.expect): the pair is tagged FL and FR; with them the ground seen by at least two cameras goes from 53.0 to 69.2 %;
the read-outs are the scene's own; FR mirrors FL's aim (its yaw is FL's negated); F's height typed as "600 mm" reads
0.600 m.
"""

META = {
    'id': 'placement',
    'part': 1,
    'title': 'Placing cameras',
    'promise': 'Place, move, turn and aim, with the analysis following every step.',
    'menu': 'Camera ▸ Place on surface (C)  ·  Move (G) · Rotate (R)  ·  Camera ▸ Aim',
}

LINES = [
    ("Place on surface, C: click the body and a camera lands a standoff off it, aimed out and down. Alt-click places "
     "a mirrored pair: here, on the front corners of the bumper.",
     "Place on surface, C: click the body, and a camera lands a standoff off it, aimed out and down. Alt-click places "
     "a mirrored pair: here, on the front corners of the bumper."),
    ("Two more cameras, and the ground seen by at least two of them rises from 53 to 69 per cent.",
     "Two more cameras, and the ground seen by at least two of them rises from fifty-three, to sixty-nine per cent."),
    ("G shows the selected camera's move handles: an arrow for each axis, and a square for the horizontal plane.",
     "G shows the selected camera's move handles: an arrow for each axis, and a square for the horizontal plane."),
    ("Drag an arrow and the read-out beside the cursor gives the change. Ctrl snaps to whole centimetres, Shift moves "
     "finely, and Esc puts it back without an undo step.",
     "Drag an arrow, and the read-out beside the cursor gives the change. Control snaps to whole centimetres, shift "
     "moves finely, and escape puts it back, without an undo step."),
    ("R swaps in the rotate rings: yaw about the vertical, pitch about the camera's own axis, with the aim drawn "
     "dashed.",
     "R swaps in the rotate rings: yaw about the vertical, pitch about the camera's own axis, with the aim drawn "
     "dashed."),
    ("Or drag the camera itself: it slides along the body, snapped to the surface, and the read-out names the zone it "
     "is on.",
     "Or drag the camera itself: it slides along the body, snapped to the surface, and the read-out names the zone "
     "it is on."),
    ("Camera ▸ Aim ▸ At ground point, then a click on the ground, turns it to look there; its linked partner on the "
     "right turns the mirror way.",
     "Camera, aim, at ground point, then a click on the ground, turns it to look there. Its linked partner on the "
     "right turns the mirror way."),
    ("Number fields take expressions and units: type 600 mm into the front camera's height and it reads 0.600 metres. "
     "Drag a field's letter to scrub it.",
     "Number fields take expressions and units: type six hundred millimetres into the front camera's height, and it "
     "reads zero point six metres. Drag a field's letter to scrub it."),
    ("The Cameras table holds every pose, for editing in place or pasting from a spreadsheet.",
     "The cameras table holds every pose, for editing in place, or pasting from a spreadsheet."),
    ("Values can be shown from the rear axle, the front axle, the front bumper or the centre, in metres or "
     "millimetres; the project itself stays in ISO 8855.",
     "Values can be shown from the rear axle, the front axle, the front bumper, or the centre, in metres or "
     "millimetres. The project itself stays in I S O eighty-eight fifty-five."),
]


def scene(stage, r):
    import numpy as np
    from PyQt6.QtCore import QPointF, QRectF, Qt
    from PyQt6.QtTest import QTest

    from helpers_tour_a import click_sent, drag_sent, project_pt

    w = stage.window()
    r.win = w
    st = w.store
    vp = stage.vp
    app = stage.app
    sc = w.viewarea.panes[0].scene
    rows = w.bottom.coverage.rows
    page = w.inspector.pages['camera']

    def shown(k):
        return rows[k].v.text()

    def pump(n=4):
        for _ in range(n):
            app.processEvents()
    r.expect(shown('Seen by ≥ 2 cameras'), '53.0', 'seen by two or more, six cameras')
    m = st.project.vehicle.mesh
    vg = w.ctx.analysis.engine.vehicle_geom(st.project)
    lo, hi = vg.bbox
    o = np.array([[hi[0] - 0.12, 2.0, 0.62]])
    d = np.array([[0.0, -1.0, 0.0]])
    t, f = m.raycast(o, d)
    corner = o[0] + d[0] * t[0]                     # the bumper's front-left corner, where the pair goes

    # ---- 0: Place on surface, Alt-click: a mirrored pair
    r.cursor_on = True
    r.lower_third()
    r.say(0)
    stage.fly(r, vp, az=32.0, el=20.0, dist=vp.cam.dist * 0.78, target=(2.2, 0.0, 0.45), dur=1.8)
    r.toolbar('tool_place')
    r.key_combo('C', dur=1.2)
    r.hold(0.4)
    glass = project_pt(r, vp, (2.15, 0.55, 1.15))        # over the windscreen: forbidden
    r.hover(vp, glass, 0.8)
    r.hold(0.6)
    q = project_pt(r, vp, corner)
    r.until_word(0, 'Alt-click')
    r.hover(vp, q, 0.8)
    r.key_combo('Alt+Click', dur=1.4)
    r.hold(0.2)
    stage.mark_result()
    click_sent(r, vp, q, 0.3, mods=Qt.KeyboardModifier.AltModifier)
    pump()
    tags = [c.tag for c in st.variant.cameras]
    print('cameras now:', tags, flush=True)
    r.expect(tags[-2:], ['FL', 'FR'], 'the new pair')
    stage.wait_analysis(r)
    r.finish_speech(0.1)
    r.toolbar('tool_select')

    # ---- 1: the effect: seen by two or more
    r.say(1)
    r.expect(shown('Seen by ≥ 2 cameras'), '69.2', 'seen by two or more, eight cameras')
    r.cursor_on = False
    r.figure('Seen by ≥ 2 cameras', float(shown('Seen by ≥ 2 cameras')), '%', 1, dur=max(3.4, r.speech_left() + 0.6),
             src=rows['Seen by ≥ 2 cameras'], start=53.0, note='8 cameras')
    stage.orbit(r, vp, -22, max(3.0, r.speech_left() + 0.2), d_el=3)
    r.finish_speech(0.3)

    fl = st.variant.by_tag('FL')
    fr = st.variant.by_tag('FR')

    # ---- 2: G: the move handles
    r.say(2)
    r.cursor_on = True
    st.select_cams([fl.uid])
    q = project_pt(r, vp, fl.pos())
    click_sent(r, vp, q, 0.7)
    r.toolbar('tool_move')
    r.key_combo('G', dur=1.2)
    pump()
    c = st.cam(fl.uid)
    stage.fly(r, vp, az=vp.cam.az + 10, el=24.0, dist=2.8, target=c.pos(), dur=1.4)
    flp = project_pt(r, vp, st.cam(fl.uid).pos())
    gz = r.wpos(vp, flp)
    r.zoom(QRectF(gz.x() - 430, gz.y() - 250, 860, 480), 1.0)
    r.finish_speech(0.2)

    # ---- 3: drag an arrow with the read-out; Esc puts it back
    r.say(3)
    pos, L, axes = sc._gizmo_axes(vp, st.cam(fl.uid))
    a0 = project_pt(r, vp, pos + axes[2] * L * 0.8)      # the z arrow: up the bumper's corner
    a1 = project_pt(r, vp, pos + axes[2] * L * 1.25)
    z_before = st.cam(fl.uid).z
    drag_sent(r, vp, a0, a1, 1.8, hold_end=1.2)
    r.leave(vp)
    r.hold(0.3)
    print('drag z:', z_before, '->', st.cam(fl.uid).z, flush=True)
    r.until_word(3, 'and escape')
    pos, L, axes = sc._gizmo_axes(vp, st.cam(fl.uid))
    b0 = project_pt(r, vp, pos + axes[1] * L * 0.8)
    b1 = project_pt(r, vp, pos + axes[1] * L * 1.8)
    n_undo = st.undo.count()
    y_before = st.cam(fl.uid).y

    def esc():
        r.key_combo('Esc', dur=1.2)
        QTest.keyClick(vp, Qt.Key.Key_Escape)
    drag_sent(r, vp, b0, b1, 1.6, during=(0.85, esc))
    r.leave(vp)
    pump()
    r.expect(st.undo.count(), n_undo, 'no undo step after Esc')
    r.expect(st.cam(fl.uid).y, y_before, 'Esc puts it back')
    r.finish_speech(0.2)

    # ---- 4: R: the rotate rings
    r.say(4)
    r.toolbar('tool_rotate')
    r.key_combo('R', dur=1.2)
    pump()
    c = st.cam(fl.uid)
    ring = sc._ring(vp, c, 0)
    s_, _, ok = vp.project(ring)
    k0 = int(np.argmax(s_[:, 1] * ok))                    # the ring's nearest point to the viewer (lowest on screen)
    pa = QPointF(*s_[k0])
    pb = QPointF(*s_[(k0 + 4) % len(ring)])
    r.hold(0.6)
    drag_sent(r, vp, pa, pb, 1.6, hold_end=1.0)
    r.leave(vp)
    print('yaw after the ring:', st.cam(fl.uid).yaw, 'FR', st.cam(fr.uid).yaw, flush=True)
    r.finish_speech(0.2)

    # ---- 5: drag the camera itself along the body
    r.say(5)
    r.toolbar('tool_select')
    pump()
    c = st.cam(fl.uid)
    p0 = project_pt(r, vp, c.pos())
    p1 = project_pt(r, vp, (c.x + 0.07, c.y - 0.30, c.z))
    before = c.pos()
    drag_sent(r, vp, p0, p1, 2.2, hold_end=1.4)
    r.leave(vp)
    print('after the surface drag:', before, '->', st.cam(fl.uid).pos(), st.cam(fl.uid).location, flush=True)
    r.expect(bool(np.linalg.norm(st.cam(fl.uid).pos() - before) > 0.05), True, 'the camera slid along the body')
    r.finish_speech(0.2)

    # ---- 6: Aim at a ground point; the partner mirrors it
    r.say(6)
    r.zoom(None, 0.8, wait=False)
    stage.fly(r, vp, az=32.0, el=22.0, dist=4.2, target=(2.4, 0.0, 0.4), dur=1.0)
    r.menu('Camera', 'Aim', 'At ground point')
    pump()
    gp = project_pt(r, vp, (hi[0] + 1.6, 1.6, 0.0))
    click_sent(r, vp, gp, 0.8)
    r.leave(vp)
    pump()
    c, cr = st.cam(fl.uid), st.cam(fr.uid)
    print('aimed FL yaw/pitch', c.yaw, c.pitch, '| FR', cr.yaw, cr.pitch, flush=True)
    r.expect(round(cr.yaw, 1), round(-c.yaw, 1), "FR mirrors FL's yaw")
    r.expect(round(cr.pitch, 1), round(c.pitch, 1), "FR mirrors FL's pitch")
    stage.wait_analysis(r)
    r.cursor_on = False
    r.until_word(6, 'Its linked partner')
    stage.orbit(r, vp, 30, max(2.0, r.speech_left() + 0.2), d_el=4)
    r.finish_speech(0.2)

    # ---- 7: a value with a unit; scrubbing
    r.say(7)
    r.cursor_on = True
    cf = st.variant.by_tag('F')
    st.select_cams([cf.uid])
    vp.cam.az, vp.cam.el = -38.0, 22.0                 # the whole car beside the Inspector again
    w.viewarea.frame_all()
    stage.settle(2)
    r.crossfade(0.4)
    insp = r.wrect(w.docks['inspector'])
    r.zoom(QRectF(insp.x() - 40, insp.y(), insp.width() + 40, 300), 0.9)
    r.enter(page.pz, '600 mm', cps=9)
    if abs(st.cam(cf.uid).z - 0.6) > 1e-6:
        page.pz.committed.emit(0.6)
    pump()
    r.expect(page.pz.text().strip(), '0.600', "F's height as the field shows it")
    r.callout(page.pz, '', dur=1.8, pad=3, dim=0.2)
    r.until_word(7, 'Drag a field')
    hdl = page.rp.handle
    hc = QPointF(hdl.width() / 2, hdl.height() / 2)
    import math as _m
    drag_sent(r, hdl, hc, hc, 1.6, path=lambda t: QPointF(hc.x() + 60 * _m.sin(_m.pi * t), hc.y()))
    r.finish_speech(0.2)
    r.zoom(None, 0.7)

    # ---- 8: the Cameras table
    r.say(8)
    w.bottom.show_tab('cameras')
    pump()
    r.crossfade(0.3)
    b = r.wrect(w.docks['bottom'])
    r.zoom(QRectF(b.x(), b.y(), b.width() * 0.6, b.height()), 0.9)
    r.move_to(w.bottom.cameras, QPointF(w.bottom.cameras.width() * 0.3, 60), 0.6)
    r.hold(max(1.6, r.speech_left()))
    r.zoom(None, 0.7)
    w.bottom.show_tab('coverage')
    r.finish_speech(0.2)

    # ---- 9: the frame and the units the values are shown in
    r.say(9)
    r.zoom(QRectF(insp.x() - 40, insp.y(), insp.width() + 40, 300), 0.9, wait=False)
    r.menu('View', 'Coordinate frame', 'ISO 8855 · front bumper')
    pump()
    r.hold(0.4)
    r.callout(page.row_pos, '', dur=2.0, pad=3, dim=0.2)
    r.hold(1.6)
    r.menu('View', 'Units', 'mm')
    pump()
    r.callout(page.row_pos, '', dur=max(1.6, r.speech_left() - 1.0), pad=3, dim=0.2)
    r.hold(max(1.2, r.speech_left() - 1.2))
    r.cursor_on = False
    r.zoom(None, 0.8)
    r.finish_speech(1.0)
