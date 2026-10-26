"""03 Moving around in 3D and the renderer: orbit about the point under the cursor (its dot), the wheel zooming towards
the surface under the cursor, the navigation control (ball, hand, magnifier, axis bubbles, frame, perspective /
orthographic), the number keys and arrows, Preferences' mouse-button styles, the 1/2/4 viewport layouts (perspective,
top, through a camera, bird's-eye), the shading modes, the shader renderer's materials and shadows, camera light while
a camera turns, and Render image / Turntable video.

Every interaction is a real event sent to the 3D view (mouse drags, wheel notches, clicks on the control, key presses).
Figures: the render sizes are read from the app's own Render image choices for this view (1x to 4x its size).
"""

META = {
    'id': 'viewport',
    'part': 1,
    'title': 'Moving around in 3D and the renderer',
    'promise': 'Orbit, zoom, split, shade, and see what every lens sees.',
    'menu': 'View ▸ Viewport layout · Shading · Overlays  ·  View ▸ Render image…',
}

LINES = [
    ("The 3D view is where the rig is built. Drag with the left button to orbit: the view turns about the point under "
     "the cursor, marked with a dot, so what you grab stays put.",
     "The 3D view is where the rig is built. Drag with the left button to orbit: the view turns about the point under "
     "the cursor, marked with a dot, so what you grab stays put."),
    ("The wheel zooms towards the surface under the cursor: here, in to the front camera, and back out.",
     "The wheel zooms towards the surface under the cursor: here, in to the front camera, and back out."),
    ("For a touchpad, or a mouse without a middle button, the control in the corner does the same: drag the ball to "
     "orbit, drag the hand to pan, and drag the magnifier to zoom in and out.",
     "For a touchpad, or a mouse without a middle button, the control in the corner does the same: drag the ball to "
     "orbit, drag the hand to pan, and drag the magnifier to zoom in and out."),
    ("Click an axis to look along it, the side or the top. The frame button shows everything again, and the cube "
     "switches between perspective and orthographic.",
     "Click an axis to look along it, the side, or the top. The frame button shows everything again, and the cube "
     "switches between perspective and orthographic."),
    ("With the view focused, keys do it too: 7, 1 and 3 for top, front and side, 0 for the default view, 5 for "
     "orthographic, and the arrows to orbit.",
     "With the view focused, keys do it too: seven, one, and three for top, front, and side, zero for the default "
     "view, five for orthographic, and the arrows to orbit."),
    ("Preferences choose which mouse buttons orbit and pan: Blender-style, with the middle button, or CAD-style, with "
     "the right.",
     "Preferences choose which mouse buttons orbit and pan: Blender style, with the middle button, or CAD style, "
     "with the right."),
    ("The view splits two or four ways, each pane with its own camera: perspective, the top view, the view through a "
     "camera's lens, and the stitched bird's-eye.",
     "The view splits two or four ways, each pane with its own camera: perspective, the top view, the view through a "
     "camera's lens, and the stitched bird's-eye."),
    ("Shading switches between solid, wireframe to check the mesh, and x-ray, which sees through the body to the "
     "cameras and sensors behind it. Then back to solid.",
     "Shading switches between solid, wireframe to check the mesh, and X-ray, which sees through the body to the "
     "cameras and sensors behind it. Then back to solid."),
    ("The shader renderer gives the body real materials: clear-coated paint reflecting a studio, dark see-through "
     "glass and matte tyres, with sky occlusion in the wheel arches and soft shadows on the ground.",
     "The shader renderer gives the body real materials: clear-coated paint reflecting a studio, dark see-through "
     "glass, and matte tyres, with sky occlusion in the wheel arches, and soft shadows on the ground."),
    ("Camera light, Alt+3, runs every camera's real lens model and the body's occlusion per pixel: the body and the "
     "ground show what each camera sees, live while a camera turns.",
     "Camera light, alt three, runs every camera's real lens model, and the body's occlusion, per pixel: the body "
     "and the ground show what each camera sees, live, while a camera turns."),
    ("Render image saves the view at up to four times its size, with the overlays kept crisp, and Turntable video "
     "turns it once round into an MP4.",
     "Render image saves the view at up to four times its size, with the overlays kept crisp, and turntable video "
     "turns it once round, into an M P 4."),
]


def scene(stage, r):
    import math

    import numpy as np
    from PyQt6.QtCore import QPoint, QPointF, QRectF, Qt
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QInputDialog

    from helpers_tour_a import click_sent, drag_sent, project_pt, wheel
    from vantage.ui.navigation import BALL_R
    from vantage.ui.dialogs.misc import PreferencesDialog

    w = stage.window()
    r.win = w
    st = w.store
    vp = stage.vp
    app = stage.app
    w.ctx.set_overlay('camlight', False)
    stage.settle(3)
    print('overlays:', {k: v for k, v in w.ctx.overlays.items() if v}, flush=True)
    cam_f = st.variant.by_tag('F')

    def pump(n=4):
        for _ in range(n):
            app.processEvents()

    def body_pt(xyz):
        q = project_pt(r, vp, xyz)
        return q

    # ---- 0: orbit about the point under the cursor
    r.cursor_on = True
    r.lower_third()
    r.say(0)
    r.hold(0.8)
    a = body_pt((2.3, -0.45, 1.0))                     # the bonnet, right of centre
    drag_sent(r, vp, a, QPointF(a.x() - 240, a.y() + 30), 2.4, kick=True)
    r.leave(vp)
    r.hold(0.3)
    b = body_pt((0.0, -0.84, 0.35))                     # the rear wheel on the right
    if b is None or not (0 < b.x() < vp.width() and 0 < b.y() < vp.height()):
        b = QPointF(vp.width() * 0.45, vp.height() * 0.6)
    drag_sent(r, vp, b, QPointF(b.x() + 200, b.y() - 50), 2.2, kick=True)
    r.leave(vp)
    r.finish_speech(0.1)

    # ---- 1: the wheel zooms towards the surface under the cursor
    r.say(1)
    f = body_pt(cam_f.pos())
    r.hold(0.4)
    wheel(r, vp, f, 7, 1.6)
    r.hold(0.8)
    wheel(r, vp, f, -7, 1.4)
    r.leave(vp)
    r.finish_speech(0.1)

    # ---- 2: the navigation control: ball, hand, magnifier
    nav = vp.nav
    r.say(2)

    def nav_box():
        c = nav.centre()
        top = r.wpos(vp, QPointF(c.x() - BALL_R - 10, c.y() - BALL_R - 10))
        bot = r.wpos(vp, nav.button_centre(3))
        return QRectF(top.x(), top.y(), 2 * BALL_R + 20, bot.y() - top.y() + 22)
    r.callout(nav_box, '', dur=2.4, pad=2, dim=0.3)
    c = nav.centre()
    r.until_word(2, 'drag the ball')
    drag_sent(r, vp, QPointF(c.x() + 8, c.y() + 10), QPointF(c.x() - 70, c.y() + 18), 1.0)
    pan = nav.button_centre(0)
    r.until_word(2, 'drag the hand')
    drag_sent(r, vp, pan, QPointF(pan.x() - 90, pan.y() + 40), 0.8)
    zm = nav.button_centre(1)
    r.until_word(2, 'drag the magnifier')
    drag_sent(r, vp, zm, QPointF(zm.x(), zm.y() + 50), 0.8)
    r.leave(vp)
    r.finish_speech(0.1)

    # ---- 3: axis bubbles, frame, perspective / orthographic
    def bubble(k, sign):
        return next(q for _, kk, ss, q in nav.bubbles() if (kk, ss) == (k, sign))
    r.say(3)
    r.hold(0.3)
    click_sent(r, vp, bubble(1, 1), 0.7)              # look at the left side
    r.hold(1.3)
    click_sent(r, vp, bubble(2, 1), 0.6)              # the top
    r.hold(1.3)
    r.until_word(3, 'The frame button')
    click_sent(r, vp, nav.button_centre(2), 0.6)
    r.hold(1.0)
    r.until_word(3, 'and the cube')
    click_sent(r, vp, nav.button_centre(3), 0.6)
    r.hold(0.9)
    r.expect(vp.cam.ortho, True, 'orthographic after the cube')
    click_sent(r, vp, nav.button_centre(3), 0.5)
    r.hold(0.4)
    r.finish_speech(0.1)

    # ---- 4: keys
    r.say(4)
    r.move(QPointF(r.wpos(vp).x() + 120, r.wpos(vp).y() + 140), 0.6)
    for key, cap, gap in ((Qt.Key.Key_7, '7', 1.25), (Qt.Key.Key_1, '1', 1.15), (Qt.Key.Key_3, '3', 1.2),
                          (Qt.Key.Key_0, '0', 1.3), (Qt.Key.Key_5, '5', 0.9)):
        r.key_combo(cap, dur=gap)
        r.hold(0.15)
        QTest.keyClick(vp, key)
        r.hold(gap - 0.1)
    QTest.keyClick(vp, Qt.Key.Key_5)                  # back to perspective
    r.key_combo('←', dur=1.6)
    for _ in range(3):
        QTest.keyClick(vp, Qt.Key.Key_Left)
        r.hold(0.4)
    r.finish_speech(0.1)

    # ---- 5: Preferences: mouse-button styles
    r.say(5)
    r.peek_menu('File', 'Preferences', hold=0.4)
    d = PreferencesDialog(w.ctx, w)
    stage.extra.append(d)
    stage.tilde(d.working)
    d.show()
    pump()
    d.move(w.mapToGlobal(QPoint((w.width() - d.width()) // 2, (w.height() - d.height()) // 2)))
    pump()
    r.crossfade(0.3)
    r.zoom(d, 0.8, margin=40)
    r.click(d.nav, None, 0.6, real=False)
    r.until_word(5, 'or CAD style')
    d.nav.setCurrentIndex(1)                    # shown, not applied: the dialog is cancelled
    r.callout(d.nav, '', dur=max(1.6, r.speech_left()), pad=3, dim=0.25)
    r.hold(max(1.4, r.speech_left()))
    d.nav.setCurrentIndex(0)
    r.hold(0.2)
    d.hide()
    r.zoom(None, 0.0)
    r.crossfade(0.4)
    r.finish_speech(0.1)

    # ---- 6: split four ways
    r.say(6)
    st.select_cams([st.variant.by_tag('ML').uid])
    r.toolbar('split4')
    stage.settle(5)
    w.viewarea.frame_all()
    pump()
    r.crossfade(0.4)
    panes = w.viewarea.panes
    print('pane kinds:', [p.kind for p in panes], flush=True)
    r.cursor_on = False
    for i, word in enumerate(('perspective', 'the top view', 'the view through', "the stitched bird's-eye")):
        r.until_word(6, word)
        r.callout(panes[i], '', dur=1.6 if i < 3 else max(1.6, r.speech_left() + 0.4), pad=-3, dim=0.3)
    r.finish_speech(0.3)
    r.cursor_on = True
    r.toolbar('split1')
    st.select_cams([])
    stage.settle(3)
    vp.cam.az, vp.cam.el = -38.0, 22.0
    w.viewarea.frame_all()
    pump()
    r.crossfade(0.4)

    # ---- 7: shading
    r.say(7)
    r.menu('View', 'Shading', 'Wire')
    r.crossfade(0.3)
    r.until_word(7, 'and X-ray')
    r.menu('View', 'Shading', 'X-ray')
    r.crossfade(0.3)
    r.cursor_on = False
    stage.orbit(r, vp, 25, max(1.6, r.at_word(7, 'Then back') - r.now() - 0.4), d_el=2)
    r.cursor_on = True
    r.menu('View', 'Shading', 'Solid')
    r.crossfade(0.3)
    r.finish_speech(0.1)

    # ---- 8: the shader renderer: materials, sky occlusion, shadows
    r.say(8)
    r.cursor_on = False
    w.ctx.set_overlay('footprints', False)
    pump()
    r.crossfade(0.4)
    lo, hi = stage.car_box(w)
    stage.fly(r, vp, az=-58.0, el=9.0, dist=vp.cam.dist * 0.55, target=(2.25, 0.55, 0.5), dur=2.4)
    stage.orbit(r, vp, -26, max(3.0, r.speech_left() - 2.4), d_el=4)
    stage.fly(r, vp, az=-140.0, el=16.0, dist=vp.cam.dist * 1.6, target=(0.9, 0.0, 0.5), dur=2.4)
    r.finish_speech(0.1)

    # ---- 9: camera light, live while a camera turns
    r.say(9)
    r.cursor_on = True
    r.menu('View', 'Overlays', 'Camera light')
    r.key_combo('Alt+3', dur=1.4)
    w.ctx.set_overlay('footprints', True)
    stage.settle(3)
    r.crossfade(0.4)
    r.cursor_on = False
    stage.fly(r, vp, az=-30.0, el=26.0, dist=vp.cam.dist * 1.05, target=(1.2, 0.0, 0.4), dur=1.6)
    y0 = cam_f.yaw
    n = round(max(3.6, r.speech_left() - 0.6) * stage.fps)
    for k in range(1, n + 1):
        t = k / n
        st.set_cam(cam_f.uid, merge=True, yaw=round(y0 + 28 * math.sin(2 * math.pi * t), 1), text='Yaw')
        r.tick()
    st.set_cam(cam_f.uid, merge=True, yaw=y0, text='Yaw')
    r.tick()
    r.finish_speech(0.1)

    # ---- 10: Render image, Turntable video
    r.say(10)
    r.cursor_on = True
    r.peek_menu('View', 'Render image', hold=0.5)
    r.leave(vp)
    items = [f'{k}× · {vp.width() * k} × {vp.height() * k} px' for k in (1, 2, 3, 4)]
    print('render sizes:', items, flush=True)
    dlg = QInputDialog(w)
    dlg.setWindowTitle('Render image')
    dlg.setLabelText('Size')
    dlg.setComboBoxItems(items)
    dlg.setTextValue(items[1])
    stage.extra.append(dlg)
    dlg.show()
    pump()
    dlg.move(w.mapToGlobal(QPoint((w.width() - dlg.width()) // 2, (w.height() - dlg.height()) // 2)))
    pump()
    r.crossfade(0.3)
    r.zoom(dlg, 0.7, margin=120)
    r.hold(1.6)
    dlg.hide()
    r.zoom(None, 0.6)
    r.crossfade(0.3)
    r.until_word(10, 'and turntable')
    r.peek_menu('View', 'Turntable video', hold=max(0.8, r.speech_left() - 1.2))
    r.cursor_on = False
    r.finish_speech(1.0)
