"""02 Importing a vehicle: the wizard, step by step, on a real file: the sample's own body written out as an OBJ the way
many modelling tools export a car (millimetres, Y up, the nose towards -Z, each wheel its own mesh, see-through glass;
helpers_tour_a.write_body_obj, into the throw-away home's Downloads folder, shown from ~).

Figures read from the wizard while recording (r.expect): 7 meshes, 6 040 triangles, units guessed millimetres with a
3.993 m long bounding box that "fits a passenger car"; roles body = occluder and mountable, glass = occluder and not
mountable, the four wheels = wheels; the forward guess (+Z, the longest axis) points at the tail and is flipped to -Z;
4 wheels found; the spec check: every dimension within the 5 mm tolerance; the occluder brought to 4 000 triangles
(3 920 triangles, 53.8 mm max deviation) and back to 8 000 (the 6 040-triangle body used whole); 1 glass part found; the import recorded in the History.
"""

META = {
    'id': 'import',
    'part': 1,
    'title': 'Importing a vehicle: the wizard',
    'promise': 'Any model, any units, any axes: brought in, measured and checked.',
    'menu': 'File ▸ Import ▸ Vehicle model…  (Ctrl+I)',
}

LINES = [
    ("Your own vehicle comes in with File ▸ Import ▸ Vehicle model, or Ctrl+I: glTF and GLB, OBJ, STL, PLY and FBX, "
     "and STEP files with the optional gmsh package.",
     "Your own vehicle comes in with file, import, vehicle model, or control I: G L T F and G L B, O B J, S T L, "
     "P L Y and F B X, and STEP files, with the optional G M S H package."),
    ("Here, the sample's own body, written out as an OBJ in millimetres with Y up, the way many modelling tools "
     "export a car.",
     "Here, the sample's own body, written out as an O B J file in millimetres, with Y up, the way many modelling "
     "tools export a car."),
    ("The wizard reads its seven meshes and guesses the units from the bounding box: read as millimetres, the body is "
     "3.99 metres long, the size of a passenger car.",
     "The wizard reads its seven meshes, and guesses the units from the bounding box: read as millimetres, the body "
     "is three point nine nine metres long, the size of a passenger car."),
    ("Each mesh gets a role from its name and material: the body can carry cameras, the glass blocks the view but "
     "cannot, and the wheels are found by name.",
     "Each mesh gets a role from its name and material: the body can carry cameras, the glass blocks the view but "
     "cannot, and the wheels are found by name."),
    ("Orient: up is the shortest axis and forward the longest, but this file's nose points the other way: the red "
     "x arrow points at the tail.",
     "Orient: up is the shortest axis, and forward the longest. But this file's nose points the other way: the red "
     "X arrow points at the tail."),
    ("One click flips forward, and the ISO 8855 triad matches the car: x forward, y left, z up.",
     "One click flips forward, and the I S O eighty-eight fifty-five triad matches the car: X forward, Y left, "
     "Z up."),
    ("Scale and origin: the four wheel centres come from the wheel meshes, or a click on each tyre. The ground "
     "passes through the tyre contact points, and the origin goes to the middle of the rear axle.",
     "Scale and origin: the four wheel centres come from the wheel meshes, or a click on each tyre. The ground "
     "passes through the tyre contact points, and the origin goes to the middle of the rear axle."),
    ("The spec check holds the body against the datasheet: length, width, height, wheelbase and ground clearance, "
     "each within the five-millimetre tolerance.",
     "The spec check holds the body against the datasheet: length, width, height, wheelbase, and ground clearance, "
     "each within the five millimetre tolerance."),
    ("Lines of sight are tested against a simplified copy of the body, the occluder. With a target of 4,000 "
     "triangles, it strays up to 53.8 millimetres from the full mesh.",
     "Lines of sight are tested against a simplified copy of the body, the occluder. With a target of four thousand "
     "triangles, it strays up to fifty-three point eight millimetres from the full mesh."),
    ("Too coarse for camera work: back to 8,000, and this 6,040-triangle body is used whole.",
     "Too coarse for camera work. Back to eight thousand, and this six thousand and forty triangle body is used "
     "whole."),
    ("Glass is found from its material and transparency, and made a forbidden zone.",
     "Glass is found from its material and transparency, and made a forbidden zone."),
    ("Your own zones are painted with a brush: name the zone, set the brush's size, and paint. Here, an allowed zone "
     "along the roof, for a rack.",
     "Your own zones are painted with a brush: name the zone, set the brush's size, and paint. Here, an allowed zone "
     "along the roof, for a rack."),
    ("The summary lists every choice. Import replaces the vehicle as one undo step; the cameras stay, and are checked "
     "against the new body.",
     "The summary lists every choice. Import replaces the vehicle as one undo step. The cameras stay, and are "
     "checked against the new body."),
    ("The body is in the window, the analysis has run again on it, and the History records the import with its "
     "units and axes.",
     "The body is in the window, the analysis has run again on it, and the history records the import, with its "
     "units and axes."),
    ("Later, the Vehicle menu reopens the same steps on the current body: align and scale, wheel centres, the ground "
     "plane and the zone brushes.",
     "Later, the vehicle menu reopens the same steps on the current body: align and scale, wheel centres, the ground "
     "plane, and the zone brushes."),
]

OCC = 4000


def scene(stage, r):
    import os
    import time

    import numpy as np
    from PyQt6.QtCore import QPoint, QPointF, QRectF
    from PyQt6.QtWidgets import QApplication, QPushButton

    from helpers_tour_a import drag_sent, write_body_obj
    from vantage.ui.dialogs.import_wizard import ImportWizard

    path = write_body_obj(os.path.join(stage.home, 'Downloads', 'tigor-body.obj'))
    w = stage.window()
    r.win = w
    st = w.store
    vp = stage.vp
    app = stage.app

    def pump(n=4):
        for _ in range(n):
            app.processEvents()

    # ---- 0: the menu, the shortcut
    r.cursor_on = True
    r.lower_third()
    r.say(0)
    r.hold(0.6)
    r.zoom(QRectF(0, 0, 820, 460), 0.8, wait=False)
    r.peek_menu('File', 'Import', 'Vehicle model', hold=1.6)
    r.key_combo('Ctrl+I', dur=1.6)
    wiz = ImportWizard(w.ctx, w)
    stage.extra.append(wiz)
    stage.tilde(wiz.f_meta)
    wiz.show()
    pump()
    wiz.move(w.mapToGlobal(QPoint((w.width() - wiz.width()) // 2, (w.height() - wiz.height()) // 2 + 10)))
    pump()
    r.crossfade(0.35)
    r.hold(0.5)
    r.zoom(wiz, 1.0, margin=20)
    r.finish_speech(0.1)

    # ---- 1: Browse... (a file dialog cannot be filmed: the click is shown, the file opened by the wizard's load())
    r.say(1)
    br = next(b for b in wiz.drop.findChildren(QPushButton) if b.text().startswith('Browse'))
    r.click(br, None, 0.8, real=False)
    r.hold(0.6)
    wiz.load(path)
    t = time.time()
    while wiz.lm is None and time.time() - t < 30:
        r.tick()
    pump()
    r.hold(0.4)
    r.callout(wiz.drop, '', dur=max(2.0, r.speech_left()), pad=2, dim=0.25)
    r.finish_speech(0.2)

    # ---- 2: seven meshes; the units from the bounding box
    meta = wiz.f_meta.text()
    print('wizard file line:', meta, '|', wiz.u_note.text(), flush=True)
    r.expect('· 7 meshes ·' in meta and '6 040 tris' in meta, True, 'meshes and triangles')
    r.expect(wiz.units, 'mm', 'units guessed')
    r.expect(wiz.u_note.text().startswith('guessed · bounding box 3.993 ×'), True, 'length read in millimetres')
    r.expect('fits a passenger car' in wiz.u_note.text(), True, 'passenger car')
    r.say(2)
    r.callout(wiz.parts, '', dur=2.4, pad=2, dim=0.25)
    r.until_word(2, 'guesses the units')
    r.move_to(wiz.u_combo, None, 0.7)
    r.callout(lambda: r.wrect(wiz.u_combo).united(r.wrect(wiz.u_note)), '', dur=max(2.6, r.speech_left() + 0.3), pad=4,
              dim=0.3)
    r.finish_speech(0.2)

    # ---- 3: the roles
    roles = {wiz.parts.item(i, 0).text(): wiz.parts.cellWidget(i, 3).currentText() for i in range(wiz.parts.rowCount())}
    print('roles:', roles, flush=True)
    r.expect(roles['body'], 'occluder · mountable', 'body role')
    r.expect(roles['glass'], 'occluder · not mountable', 'glass role')
    r.expect(sorted(v for k, v in roles.items() if k.startswith('wheel_')), ['wheels'] * 4, 'wheel roles')
    hh = wiz.parts.horizontalHeader()

    def role_col():
        x0 = hh.sectionViewportPosition(3)
        top = r.wpos(hh, QPointF(x0, 0))
        n = wiz.parts.rowCount()
        h = hh.height() + wiz.parts.rowViewportPosition(n - 1) + wiz.parts.rowHeight(n - 1)
        return QRectF(top.x(), top.y(), hh.sectionSize(3), h)
    r.say(3)
    r.move(role_col().center(), 0.8)
    r.callout(role_col, 'Role', dur=max(3.0, r.speech_left() - 0.4), pad=3, dim=0.3, side='left')
    r.finish_speech(0.2)

    # ---- 4: orient: the forward guess points at the tail
    r.say(4)
    r.click(wiz.b_next, None, 0.7)
    pump(6)
    r.crossfade(0.3)
    vo = wiz.vp_orient
    r.hold(0.6)
    r.expect((wiz.up, wiz.fwd), ('+Y', '+Z'), 'the guessed axes')

    def tip():
        s, _, ok = vo.project([np.array([0.6, 0.0, 0.0])])
        q = r.wpos(vo, QPointF(*s[0]))
        return QRectF(q.x() - 34, q.y() - 34, 68, 68)
    r.until_word(4, 'the red')
    r.callout(tip, 'x · towards the tail', dur=max(2.4, r.speech_left() + 0.2), style='circle', dim=0.3, side='above')
    r.finish_speech(0.2)
    # ---- 5: flip forward
    flip = next(b for b in wiz.findChildren(QPushButton) if b.text() == 'Forward')
    r.say(5)
    r.click(flip, None, 0.7)
    pump(4)
    r.crossfade(0.3)
    r.expect(wiz.fwd, '-Z', 'forward after the flip')
    r.hold(0.8)
    r.callout(tip, 'x · forward', dur=2.2, style='circle', dim=0.3, side='above')
    r.hold(1.0)
    r.callout(lambda: r.wrect(wiz.up_seg).united(r.wrect(wiz.fw_seg)), '', dur=max(1.8, r.speech_left()), pad=4, dim=0.3)
    r.finish_speech(0.2)

    # ---- 6: scale and origin: wheels, ground, origin
    r.say(6)
    r.click(wiz.b_next, None, 0.7)
    pump(6)
    r.crossfade(0.3)
    r.expect(len(wiz.wheels), 4, 'wheels found')
    rows = list(wiz.wheel_rows.values())

    def wheel_box():
        a = r.wrect(rows[0][0]).united(r.wrect(rows[-1][2]))
        return QRectF(a.x() - 30, a.y() - 4, a.width() + 34, a.height() + 8)
    r.hold(0.6)
    r.callout(wheel_box, '', dur=2.6, pad=2, dim=0.3)
    r.until_word(6, 'or a click')
    r.move_to(rows[0][2], None, 0.7)
    r.hold(0.8)
    r.until_word(6, 'The ground')
    r.callout(lambda: r.wrect(wiz.ground_c).united(r.wrect(wiz.origin_c)), '', dur=max(2.4, r.speech_left()), pad=4,
              dim=0.3)
    r.finish_speech(0.2)

    # ---- 7: the spec check
    specs = [[wiz.spec_t.item(i, j).text() if wiz.spec_t.item(i, j) else '' for j in range(5)] for i in range(5)]
    print('spec check:', specs, flush=True)
    r.expect([s[4] for s in specs], ['Within tolerance'] * 5, 'spec check')
    r.say(7)
    sp = r.wrect(wiz.spec_t)
    r.zoom(QRectF(sp.x(), sp.y() - 150, sp.width(), sp.height() + 150), 1.0, margin=16)
    r.callout(wiz.spec_t, '', dur=max(2.6, r.speech_left() - 1.0), pad=2, dim=0.2)
    r.finish_speech(0.2)
    r.zoom(wiz, 0.8, margin=20)

    # ---- 8: the occluder, simplified
    r.say(8)
    r.click(wiz.b_next, None, 0.7)
    pump(6)
    r.crossfade(0.3)
    sl = wiz.occ
    y = sl.height() / 2

    def sx(v):
        return (v - sl.minimum()) / (sl.maximum() - sl.minimum()) * (sl.width() - 16) + 8
    r.hold(0.6)
    r.until_word(8, 'With a target')
    r.drag(sl, QPointF(sx(sl.value()), y), QPointF(sx(OCC), y), 1.0)
    if sl.value() != OCC:                            # land exactly on the value the narration says
        sl.setValue(OCC)
        wiz._occ()
    pump()
    occ = wiz.occ_lbl.text()
    print('occluder:', occ, flush=True)
    r.expect(wiz.occ_target, OCC, 'occluder target')
    r.expect('· 53.8 mm max deviation' in occ, True, 'occluder deviation')
    r.callout(wiz.occ_lbl, '', dur=max(2.6, r.speech_left() + 0.6), pad=4, dim=0.3, side='below')
    r.finish_speech(0.3)
    # ---- 9: back to 8 000: the body is used whole
    r.say(9)
    r.drag(sl, QPointF(sx(sl.value()), y), QPointF(sx(8000), y), 0.9)
    if sl.value() != 8000:
        sl.setValue(8000)
        wiz._occ()
    pump()
    occ = wiz.occ_lbl.text()
    print('occluder:', occ, flush=True)
    r.expect(occ.startswith('6 040 tris · 100.0 %'), True, 'occluder at 8 000')
    r.callout(wiz.occ_lbl, '', dur=max(2.0, r.speech_left() + 0.3), pad=4, dim=0.3, side='below')
    r.finish_speech(0.3)

    # ---- 10: glass
    r.say(10)
    r.expect(wiz.glass_note.text().startswith('1 glass part found by material and transparency'), True, 'glass part')
    r.callout(wiz.glass_note, '', dur=max(2.4, r.speech_left()), pad=4, dim=0.3)
    vz = wiz.vp_zones
    r.hold(0.6)
    stage.fly(r, vz, az=-62.0, el=34.0, dist=vz.cam.dist * 1.25, dur=1.2)
    r.finish_speech(0.1)
    # ---- 11: a zone painted with the brush
    r.say(11)
    r.click(wiz.paint_seg.buttons['allowed'], None, 0.5)
    r.enter(wiz.zone_name, 'Roof rack', cps=18)
    r.enter(wiz.brush, '250', cps=12)
    if abs(wiz.brush.value() - 250) > 0.5:
        wiz.brush.set_value(250)
    m = wiz.world_mesh()
    lo, hi = m.bbox
    roof_z = hi[2]

    def roof_pt(x):
        s, _, ok = vz.project([np.array([x, 0.0, roof_z])])
        return QPointF(*s[0])
    a, b = roof_pt(0.15), roof_pt(1.55)
    drag_sent(r, vz, a, b, 1.8)
    pump()
    r.hold(0.8)
    zones = [(z.kind, z.name, int((wiz.face_zone == i).sum())) for i, z in enumerate(wiz.zones)]
    print('zones:', zones, flush=True)
    r.expect([z[:2] for z in zones], [('forbidden', 'Glass (1 part, auto)'), ('allowed', 'Roof rack')], 'zones')
    r.callout(wiz.zt, '', dur=max(1.8, r.speech_left()), pad=2, dim=0.25)
    r.finish_speech(0.2)

    # ---- 12: the summary; Import
    r.say(12)
    r.click(wiz.b_next, None, 0.7)
    pump(4)
    r.crossfade(0.3)
    r.hold(0.4)
    r.callout(wiz.sum_t, '', dur=2.6, pad=2, dim=0.2)
    r.until_word(12, 'Import replaces')
    r.move_to(wiz.b_next, None, 0.7)
    r.until_word(12, 'The cameras stay')
    r.callout(wiz.keep_cams, '', dur=max(1.8, r.speech_left() - 0.2), pad=4, dim=0.3)
    r.hold(max(0.6, r.speech_left() - 0.8))
    r.ripple()
    r.hold(0.2)
    stage.mark_result()
    wiz.finish()
    pump()
    r.zoom(None, 0.0)
    r.crossfade(0.5)
    r.finish_speech(0.1)

    # ---- 13: the body back in the window; the History
    r.say(13)
    stage.wait_analysis(r)
    r.cursor_on = False
    vp.cam.az, vp.cam.el = -40.0, 22.0
    w.viewarea.frame_all()
    stage.orbit(r, vp, 30, 2.6, d_el=3)
    r.expect(st.project.vehicle.import_units, 'mm', 'import units in the project')
    w.bottom.show_tab('history')
    pump()
    hist = r.wrect(w.docks['bottom'])
    r.crossfade(0.3)
    r.zoom(QRectF(hist.x(), hist.y(), 640, 110), 0.9, margin=12)
    r.hold(max(1.6, r.speech_left() + 0.2))
    r.zoom(None, 0.7)
    w.bottom.show_tab('coverage')
    r.finish_speech(0.1)

    # ---- 14: the Vehicle menu reopens the steps
    r.say(14)
    r.cursor_on = True
    r.zoom(QRectF(0, 0, 900, 506), 0.8, wait=False)
    r.peek_menu('Vehicle', 'Align, scale', hold=max(1.4, r.speech_left() - 2.4))
    r.zoom(None, 0.7)
    r.cursor_on = False
    r.finish_speech(1.0)
