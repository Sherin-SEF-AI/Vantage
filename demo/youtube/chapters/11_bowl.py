"""11 3D surround view: the bowl an in-car display shows, full frame (a chromeless view of the page's own bowl scene),
then its controls on the Camera views page: typical viewpoints, exposure balancing off and on, the wall height
changed (and put back), the camera sectors as the source, and the bowl export.

Every recomputation is shown in real time (r.wait_for on the bowl controller, no time-lapse). Figures read from the
app while recording (r.expect): the bowl's default shape (floor 1.6 m, wall 2.6 m, reach 4.5 m), "6 camera(s) on the
bowl", the wall set to 1.6 m.
"""

META = {
    'id': 'bowl',
    'part': 2,
    'title': '3D surround view',
    'promise': 'The bowl an in-car display shows, from the planned cameras.',
    'menu': 'View ▸ Camera views ▸ 3D surround view',
}

LINES = [
    ("The 3D surround view is what an in-car display shows: every camera's image projected through its lens onto a "
     "bowl around the car, a flat floor and a curved wall, with the car model in the middle.",
     "The 3D surround view is what an in-car display shows: every camera's image projected through its lens onto a "
     "bowl around the car, a flat floor, and a curved wall, with the car model in the middle."),
    ("The images are a synthetic parking scene rendered through each planned camera: bay markings, parked cars, a "
     "pillar, a bollard, a garage wall, and the project's own obstacles.",
     "The images are a synthetic parking scene, rendered through each planned camera: bay markings, parked cars, a "
     "pillar, a bollard, a garage wall, and the project's own obstacles."),
    ("It uses the stitched bird's-eye view's seams, its exposure balancing and its blending.",
     "It uses the stitched bird's-eye view's seams, its exposure balancing, and its blending."),
    ("On the Camera views page, the bowl has its own controls, starting with the typical display viewpoints: behind "
     "the car, over a front corner, and straight down from the top, as a parking display shows it.",
     "On the camera views page, the bowl has its own controls, starting with the typical display viewpoints: behind "
     "the car, over a front corner, and straight down from the top, as a parking display shows it."),
    ("Exposure balancing evens out the cameras' gains from the ground they share. Off, every camera keeps its own "
     "brightness; on again, the seams fade.",
     "Exposure balancing evens out the cameras' gains, from the ground they share. Off, every camera keeps its own "
     "brightness. On again, the seams fade."),
    ("The bowl's shape is saved with the project: floor, wall height, reach, profile and squareness. A lower wall, "
     "1.6 metres instead of 2.6, flattens the far scene.",
     "The bowl's shape is saved with the project: floor, wall height, reach, profile, and squareness. A lower wall, "
     "one point six metres instead of two point six, flattens the far scene."),
    ("The source can also be the installation check's real captures, or just each camera's sector, to see who covers "
     "what.",
     "The source can also be the installation check's real captures, or just each camera's sector, to see who covers "
     "what."),
    ("Export writes the bowl as an OBJ with its texture, and per-vertex remap tables, a sensor pixel and a weight per "
     "camera, for the ECU's renderer.",
     "Export writes the bowl as an O B J with its texture, and per vertex remap tables, a sensor pixel and a weight "
     "per camera, for the E C U's renderer."),
]


def scene(stage, r):
    import time

    from PyQt6.QtCore import QPointF, QRectF

    from helpers_tour_a import wheel

    w = stage.window()
    st = w.store
    app = stage.app

    def pump(n=4):
        for _ in range(n):
            app.processEvents()
    # set-up, not filmed: the Camera views page with its bowl worked out once
    w.set_page('camviews')
    pg = w._page_widgets['camviews']
    ctl = pg.bowl_scene.bowl
    split = pg.bev.parentWidget()                   # the page's right-hand splitter: more room for the bowl
    split.setSizes([300, 760])
    t = time.time()
    while ctl.draw is None and time.time() - t < 180:
        app.processEvents()
        time.sleep(0.05)
    stage.settle(3)
    sh = ctl.shape
    r.expect((sh.floor_m, sh.height_m, sh.reach_m), (1.6, 2.6, 4.5), 'the default bowl')
    r.expect(ctl.status, '6 camera(s) on the bowl', 'cameras on the bowl')
    bv = stage.full_view(pg.bowl_scene, cam_from=pg.bowl.cam)
    bv.cam.dist *= 0.80
    bv.cam.az, bv.cam.el = 150.0, 33.0
    pump()

    def bowl_ready(gen0):
        return lambda: ctl.draw is not None and ctl.draw.version != gen0 and not ctl._busy and not ctl.timer.isActive()

    # ---- 0-2: full frame
    r.win = bv
    r.cursor_on = False
    r.lower_third()
    r.say(0)
    stage.orbit(r, bv, 110, r.line(0) + 0.4, d_el=-6, linear=True)
    r.say(1)
    stage.fly(r, bv, az=bv.cam.az + 40, el=24.0, dist=bv.cam.dist * 0.72, dur=2.6)
    stage.orbit(r, bv, 60, max(2.0, r.speech_left() + 0.3), d_el=2, linear=True)
    r.say(2)
    stage.fly(r, bv, az=-90.0, el=72.0, dist=bv.cam.dist * 1.25, dur=2.4)
    stage.orbit(r, bv, 30, max(1.6, r.speech_left() + 0.3), linear=True)
    bv.hide()

    # ---- 3: the page; typical viewpoints
    r.win = w
    r.say(3)
    pump()
    r.crossfade(0.6)
    bw = pg.bowl.parentWidget()
    r.zoom(bw, 1.0, margin=6, wait=False)
    r.cursor_on = True
    r.choose(pg.bowl_views, 'Behind')
    r.hold(0.6)
    r.choose(pg.bowl_views, 'Front left')
    r.hold(0.6)
    r.choose(pg.bowl_views, 'Top')
    r.hold(0.8)
    r.finish_speech(0.1)

    # ---- 4: exposure balancing off and on
    r.say(4)
    r.choose(pg.bowl_views, 'Rear left')
    r.hold(0.6)
    wheel(r, pg.bowl, QPointF(pg.bowl.width() * 0.5, pg.bowl.height() * 0.55), -4, 0.8)   # zoom out a little
    r.leave(pg.bowl)
    g0 = ctl.draw.version
    r.click(pg.bowl_balance, None, 0.6)
    if pg.bowl_balance.isChecked():
        pg.bowl_balance.setChecked(False)
    r.wait_for(bowl_ready(g0), timeout=120)
    r.until_word(4, 'On again')
    g0 = ctl.draw.version
    r.click(pg.bowl_balance, None, 0.6)
    if not pg.bowl_balance.isChecked():
        pg.bowl_balance.setChecked(True)
    r.wait_for(bowl_ready(g0), timeout=120)
    r.finish_speech(0.2)

    # ---- 5: the wall height
    r.say(5)
    f = pg.bowl_fields['height_m']
    r.until_word(5, 'A lower wall')
    g0 = ctl.draw.version
    r.enter(f, '1.6', cps=8)
    if abs(ctl.shape.height_m - 1.6) > 1e-6:
        f.committed.emit(1.6)
    r.wait_for(bowl_ready(g0), timeout=120)
    r.expect(ctl.shape.height_m, 1.6, 'the wall height')
    r.callout(f, '', dur=1.6, pad=3, dim=0.2)
    r.hold(max(1.0, r.speech_left()))
    r.finish_speech(0.2)

    # ---- 6: sources
    r.say(6)
    st.undo.undo()                                   # the wall back to 2.6 m
    r.click(pg.bowl_src, None, 0.6, real=False)
    pg.bowl_src.showPopup()
    r.hold(1.4)
    i = pg.bowl_src.findData('sectors')
    pg.bowl_src.hidePopup()
    pg.bowl_src.setCurrentIndex(i)
    pump()
    r.hold(max(1.6, r.speech_left()))
    r.finish_speech(0.1)

    # ---- 7: export
    r.say(7)
    g0 = ctl.draw.version if ctl.draw is not None else None
    pg.bowl_src.setCurrentIndex(pg.bowl_src.findData('synthetic'))
    from PyQt6.QtWidgets import QPushButton
    exp = next(b for b in bw.findChildren(QPushButton) if b.text().startswith('Export'))
    r.move_to(exp, None, 0.8)
    r.callout(exp, '', dur=2.4, pad=4, dim=0.25)
    r.wait_for(lambda: ctl.draw is not None and not ctl._busy and not ctl.timer.isActive(), timeout=120)
    r.hold(max(0.6, r.speech_left() - 1.2))
    r.cursor_on = False
    r.zoom(None, 0.8)
    r.finish_speech(1.0)
