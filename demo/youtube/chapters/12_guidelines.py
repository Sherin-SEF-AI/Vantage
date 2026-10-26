"""12 Parking guidelines: Camera ▸ Parking guidelines (the overlay on, the camera views with the steering bar), the
bicycle model, the rear camera's static lines and trajectory, the steering slider dragged (every view follows), the
swept path hitting the sample's 0.5 m pole at full lock, forward, the ground lines in the 3D and top views, and the
export for the ECU.

Figures read from the app while recording (r.expect): wheelbase 2.45 m; at +20° road wheel: +300° steering wheel,
radius 6.73 m; reversing at full right lock (−35°): "the body would hit Pole 0.5 m after 1.77 m".
"""

META = {
    'id': 'guidelines',
    'part': 2,
    'title': 'Parking guidelines',
    'promise': 'Trajectory and swept path, drawn through every real lens.',
    'menu': 'Camera ▸ Parking guidelines…  ·  View ▸ Overlays ▸ Parking guidelines (Alt+G)',
}

LINES = [
    ("Parking guidelines are what a surround-view display draws while parking. Camera ▸ Parking guidelines turns "
     "them on, and opens the camera views with the steering bar.",
     "Parking guidelines are what a surround-view display draws while parking. Camera, parking guidelines, turns them "
     "on, and opens the camera views with the steering bar."),
    ("They come from a kinematic bicycle model: the 2.45-metre wheelbase and the track from the wheel centres, and "
     "the body's width without its mirrors from the mesh.",
     "They come from a kinematic bicycle model: the two point four five metre wheelbase and the track, from the wheel "
     "centres, and the body's width without its mirrors, from the mesh."),
    ("Reversing, the rear camera shows the width lines and distance markers, and the path of the bumper corners with "
     "red, yellow and green distance bands.",
     "Reversing, the rear camera shows the width lines and distance markers, and the path of the bumper corners, with "
     "red, yellow, and green distance bands."),
    ("Drag the steering slider: left of the centre steers left. At 20 degrees on the road wheels, the steering wheel "
     "is at 300, and the turning radius 6.73 metres.",
     "Drag the steering slider: left of the centre steers left. At twenty degrees on the road wheels, the steering "
     "wheel is at three hundred, and the turning radius six point seven three metres."),
    ("Every line goes through each camera's real lens, fisheye curve and image circle, and the body hides what lies "
     "behind it; the bird's-eye view and the bowl follow too.",
     "Every line goes through each camera's real lens, fisheye curve and image circle, and the body hides what lies "
     "behind it. The bird's-eye view and the bowl follow too."),
    ("The swept path is the ground the whole body passes over, mirrors included. Reversing at full lock to the right, "
     "the body would hit the half-metre pole after 1.77 metres.",
     "The swept path is the ground the whole body passes over, mirrors included. Reversing at full lock to the right, "
     "the body would hit the half metre pole after one point seven seven metres."),
    ("Forward works the same way. Path length, margin, full lock, the steering ratio and the markers are settings, "
     "saved with the project.",
     "Forward works the same way. Path length, margin, full lock, the steering ratio, and the markers are settings, "
     "saved with the project."),
    ("Alt+G draws the same lines on the ground in the 3D and top views.",
     "Alt G draws the same lines on the ground, in the 3D and top views."),
    ("For the ECU, the guidelines export per camera: pixel polylines with colours and labels for every steering "
     "angle from lock to lock, the ground lines, and a preview over the parking scene.",
     "For the E C U, the guidelines export per camera: pixel polylines with colours and labels, for every steering "
     "angle from lock to lock, the ground lines, and a preview over the parking scene."),
]


def scene(stage, r):
    from PyQt6.QtCore import QPoint, QPointF, QRectF

    from helpers_tour_a import drag_sent
    from vantage.ui.dialogs.export_dialog import ExportDialog

    w = stage.window()
    r.win = w
    st = w.store
    app = stage.app

    def pump(n=4):
        for _ in range(n):
            app.processEvents()

    # ---- 0: Camera ▸ Parking guidelines
    r.cursor_on = True
    r.lower_third()
    r.say(0)
    r.hold(0.8)
    r.menu('Camera', 'Parking guidelines')
    stage.settle(5)
    pg = w._page_widgets['camviews']
    pg.bev.parentWidget().setSizes([480, 580])     # the right-hand splitter: room for the bowl as well
    bar = pg.guide_bar
    gc = bar.gc
    pump(10)
    r.crossfade(0.6)
    r.hold(0.6)
    r.zoom(bar, 0.9, margin=12)
    r.finish_speech(0.2)

    # ---- 1: the bicycle model
    r.say(1)
    o = gc.outline()
    r.expect(f'{o.wheelbase:.2f}', '2.45', 'wheelbase')
    r.callout(bar.readout, '', dur=max(2.4, r.speech_left() - 1.0), pad=4, dim=0.25)
    r.finish_speech(0.2)
    r.zoom(None, 0.8)

    # ---- 2: the rear camera, reversing
    r.say(2)
    cams = st.variant.cameras
    i_r = next(i for i, c in enumerate(cams) if c.tag == 'R')
    rv = pg.views[i_r]
    r.zoom(rv, 1.0, margin=8)
    r.cursor_on = False
    r.finish_speech(0.2)

    # ---- 3: steering
    sl = bar.slider

    def handle_x(v):
        from PyQt6.QtWidgets import QStyle, QStyleOptionSlider
        opt = QStyleOptionSlider()
        sl.initStyleOption(opt)
        hr = sl.style().subControlRect(QStyle.ComplexControl.CC_Slider, opt, QStyle.SubControl.SC_SliderHandle, sl)
        span = sl.width() - hr.width()
        return hr.width() / 2 + span * (v - sl.minimum()) / (sl.maximum() - sl.minimum())
    y = sl.height() / 2
    r.say(3)
    r.zoom(QRectF(r.wrect(pg.grid_w).x(), r.wrect(bar).y(), r.wrect(pg.grid_w).width(),
                  r.wrect(pg.grid_w).height() + r.wrect(bar).height()), 1.0, wait=False)
    r.cursor_on = True
    drag_sent(r, sl, QPointF(handle_x(sl.value()), y), QPointF(handle_x(-200), y), 2.6)
    pump()
    if abs(gc.stored().angle_deg - 20.0) > 1e-6:
        gc.commit(20.0)
        pump()
    ro = bar.readout.text()
    print('readout at 20:', ro, flush=True)
    r.expect(ro, '+20.0° road wheel · +300° steering wheel · radius 6.73 m', 'the read-out at 20 degrees')
    r.callout(bar.readout, '', dur=max(2.2, r.speech_left()), pad=4, dim=0.2)
    r.finish_speech(0.2)

    # ---- 4: through every lens; bird's-eye and bowl
    r.say(4)
    r.cursor_on = False
    r.zoom(pg.grid_w, 1.0, margin=6)
    r.hold(max(1.6, r.at_word(4, "The bird's-eye") - r.now()))
    r.zoom(pg.bev, 1.0, margin=6)
    r.hold(1.0)
    r.zoom(pg.bowl.parentWidget(), 0.9, margin=6)
    r.finish_speech(0.2)
    r.zoom(None, 0.8, wait=False)

    # ---- 5: the swept path; full lock right hits the 0.5 m pole
    r.say(5)
    r.cursor_on = True
    r.hold(max(0.4, r.at_word(5, 'Reversing at full lock') - r.now() - 0.4))
    drag_sent(r, sl, QPointF(handle_x(sl.value()), y), QPointF(handle_x(sl.maximum()), y), 2.4)
    pump()
    if abs(gc.stored().angle_deg + 35.0) > 1e-6:
        gc.commit(-35.0)
        pump()
    stt = bar.status.text()
    print('status at full right lock:', stt, flush=True)
    r.expect(stt, 'Swept path, reversing 3.0 m: the body would hit Pole 0.5 m after 1.77 m', 'the swept-path status')
    r.zoom(pg.bev, 0.8, margin=6)
    r.hold(1.0)
    r.zoom(QRectF(r.wrect(bar).x(), r.wrect(bar).y(), r.wrect(bar).width() * 0.75, r.wrect(bar).height() + 20), 0.8)
    r.callout(bar.status, '', dur=max(2.0, r.speech_left() + 0.4), pad=4, dim=0.2)
    r.finish_speech(0.3)

    # ---- 6: forward; the settings
    r.say(6)
    r.zoom(bar, 0.8, margin=12, wait=False)
    r.click(bar.dir.buttons['forward'], None, 0.7)
    pump()
    r.hold(0.6)
    r.click(bar.centre, None, 0.6)
    pump()
    r.callout(lambda: r.wrect(bar.fields['path_m']).united(r.wrect(bar.marks)).adjusted(-60, 0, 0, 0), '',
              dur=max(2.2, r.speech_left()), pad=4, dim=0.2)
    r.finish_speech(0.2)

    # ---- 7: on the ground in the 3D and top views
    r.say(7)
    r.zoom(None, 0.6, wait=False)
    r.click(bar.dir.buttons['reverse'], None, 0.5)
    gc.commit(20.0)
    pump()
    r.key_combo('Alt+G', dur=1.2)
    r.menu('View', 'Back to viewports')
    stage.settle(3)
    vp = w.viewarea.panes[0].vp3d
    vp.cam.az, vp.cam.el = 150.0, 40.0
    w.viewarea.frame_all()
    pump()
    r.crossfade(0.5)
    r.cursor_on = False
    stage.orbit(r, vp, 40, max(2.4, r.speech_left() + 0.2), d_el=4)
    r.finish_speech(0.1)

    # ---- 8: the export
    r.say(8)
    r.key_combo('Ctrl+E', dur=1.2)
    d = ExportDialog(w.ctx, w, preselect='guidelines')
    stage.extra.append(d)
    stage.tilde(d.out)
    d.show()
    pump(10)
    d.move(w.mapToGlobal(QPoint((w.width() - d.width()) // 2, (w.height() - d.height()) // 2)))
    pump()
    r.crossfade(0.4)
    r.zoom(d, 0.8, margin=20)
    r.hold(max(2.0, r.speech_left() - 0.6))
    d.hide()
    r.zoom(None, 0.0)
    r.crossfade(0.4)
    r.finish_speech(1.0)
