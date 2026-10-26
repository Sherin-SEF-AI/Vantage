"""14 Perception range: Analysis ▸ Perception range on the sample (the object classes, the envelope plan, by camera,
the sector verdicts against the required ranges, the stereo pairs), then a forward camera added on the front edge of
the roof (Place on surface: tag T, level-ish at 15° down; Camera ▸ Change lens ▸ H03R2; tag FN and pitch 0 typed into the Cameras table) and the page
again: the front sector passes.

Figures read from the app while recording (r.expect): classes car 1.5 x 1.8 m / 20 px, pedestrian 1.7 x 0.5 m / 24 px,
cyclist 1.7 x 0.6 / 24 px, sign 0.6 x 0.6 m / 12 px; front required 80 m; car resolved to 70 m ahead; "28 of 32 sector
checks pass"; ML · QL overlap 59°, baseline 1.88 m; with the forward camera: car 138 m, sign 92 m ahead, "32 of 32".
"""

META = {
    'id': 'perception',
    'part': 2,
    'title': 'Perception range',
    'promise': 'How far out does the rig resolve a car, a person, a sign?',
    'menu': 'Analysis ▸ Perception range…',
}

LINES = [
    ("How far out does the rig resolve the objects a driving dataset needs? Analysis ▸ Perception range stands a car, "
     "a pedestrian, a cyclist and a traffic sign in 360 directions, at 120 ranges out to 250 metres.",
     "How far out does the rig resolve the objects a driving dataset needs? Analysis, perception range, stands a car, "
     "a pedestrian, a cyclist, and a traffic sign, in three hundred and sixty directions, at a hundred and twenty "
     "ranges, out to two hundred and fifty metres."),
    ("A camera resolves an object when it is in the image, not hidden by the body, and tall enough in pixels through "
     "the real lens: 20 for a car, 24 for a person, 12 for a sign. Every figure is editable, and classes can be added.",
     "A camera resolves an object when it is in the image, not hidden by the body, and tall enough in pixels, through "
     "the real lens: twenty for a car, twenty-four for a person, twelve for a sign. Every figure is editable, and "
     "classes can be added."),
    ("The plan draws each object's envelope on a square-root range scale, with rings at 10, 25, 50, 100 and 200 "
     "metres; By camera shows which camera resolves it where.",
     "The plan draws each object's envelope on a square-root range scale, with rings at ten, twenty-five, fifty, a "
     "hundred, and two hundred metres. By camera shows which camera resolves it where."),
    ("Each sector has a required range. In the worst direction ahead, this surround rig resolves a car to 70 metres, "
     "short of the 80 required: 28 of 32 checks pass.",
     "Each sector has a required range. In the worst direction ahead, this surround rig resolves a car to seventy "
     "metres, short of the eighty required: twenty-eight of thirty-two checks pass."),
    ("Camera pairs that both see a car 20 metres out are listed as stereo pairs, with their overlap, baseline, and "
     "depth resolution at 10, 20 and 50 metres.",
     "Camera pairs that both see a car twenty metres out are listed as stereo pairs, with their overlap, baseline, and "
     "depth resolution at ten, twenty, and fifty metres."),
    ("The fisheyes look down at the ground around the car. For range, add a forward camera on the front edge of the "
     "roof, where it starts 15 degrees down, and give it the narrow H03R2 lens.",
     "The fisheyes look down at the ground around the car. For range, add a forward camera on the front edge of the "
     "roof, where it starts fifteen degrees down, and give it the narrow H zero three R two lens."),
    ("In the Cameras table, a tag of its own, and a pitch of zero: level, looking straight ahead.",
     "In the cameras table, a tag of its own, and a pitch of zero: level, looking straight ahead."),
    ("Back on the page, it runs again in under a second: a car is now resolved to 138 metres ahead, a sign to 92, and "
     "all 32 checks pass.",
     "Back on the page, it runs again in under a second: a car is now resolved to a hundred and thirty-eight metres "
     "ahead, a sign to ninety-two, and all thirty-two checks pass."),
    ("The ranges, the verdicts and the stereo pairs export as JSON, with the plan as a PNG, and the assistant can run "
     "the study too.",
     "The ranges, the verdicts, and the stereo pairs export as JSON, with the plan as a P N G, and the assistant can "
     "run the study too."),
]


def scene(stage, r):
    import numpy as np
    from PyQt6.QtCore import QPointF, QRectF

    from helpers_tour_a import click_sent, edit_cell, project_pt

    w = stage.window()
    r.win = w
    st = w.store
    app = stage.app

    def pump(n=4):
        for _ in range(n):
            app.processEvents()

    def map_box():                                 # the envelope plan's middle (the map itself is as tall as the page)
        mw = r.wrect(pg.map)
        side = min(mw.width(), mw.height()) * 0.86
        return QRectF(mw.center().x() - side / 2, mw.center().y() - side / 2, side, side)

    # ---- 0: Analysis ▸ Perception range
    r.cursor_on = True
    r.lower_third()
    r.say(0)
    r.hold(1.0)
    r.menu('Analysis', 'Perception range')
    pg = w.page('perception')
    ctl = pg.ctl
    r.crossfade(0.4)
    r.wait_for(lambda: ctl.result is not None and not ctl.busy, timeout=120)
    pump(6)
    r.cursor_on = False
    r.zoom(map_box(), 0.9, margin=8)
    r.finish_speech(0.2)
    r.zoom(None, 0.7)

    # ---- 1: the classes
    cls = [[pg.classes.item(i, j).text() for j in range(4)] for i in range(pg.classes.rowCount())]
    print('classes:', cls, flush=True)
    r.expect(cls, [['Car', '1.50', '1.80', '20'], ['Pedestrian', '1.70', '0.50', '24'], ['Cyclist', '1.70', '0.60', '24'],
                   ['Traffic sign', '0.60', '0.60', '12']], 'object classes')
    r.say(1)
    cr = r.wrect(pg.classes)
    r.zoom(QRectF(cr.x() - 10, cr.y() - 60, cr.width() + 20, cr.height() + 120), 1.0)
    hh = pg.classes.horizontalHeader()

    def mincol():
        x = hh.sectionViewportPosition(3)
        tl = r.wpos(hh, QPointF(x, 0))
        return QRectF(tl.x(), tl.y(), hh.sectionSize(3), hh.height() + pg.classes.rowHeight(0) * 4)
    r.until_word(1, 'twenty for a car')
    r.callout(mincol, '', dur=max(2.4, r.speech_left() - 2.4), pad=2, dim=0.25)
    r.finish_speech(0.2)
    r.zoom(None, 0.8)

    # ---- 2: the plan; by camera
    r.say(2)
    r.zoom(map_box(), 1.0, margin=8)
    r.until_word(2, 'By camera')
    r.cursor_on = True
    r.click(pg.mode.buttons['cameras'], None, 0.6)
    pump()
    r.hold(max(1.4, r.speech_left()))
    r.finish_speech(0.2)

    # ---- 3: the sector verdicts
    v = pg.verdicts
    head = [v.horizontalHeaderItem(j).text() for j in range(v.columnCount())]
    car = [v.item(0, j).text() for j in range(v.columnCount())]
    print('verdicts:', head, car, pg.status.text(), flush=True)
    r.expect(head[1], 'Front 80', 'front requirement')
    r.expect(car[1], '70', 'car ahead')
    r.expect(pg.status.text().startswith('28 of 32 sector checks pass'), True, 'checks passing')
    r.say(3)
    r.click(pg.mode.buttons['classes'], None, 0.5)
    vr = r.wrect(v)
    r.zoom(QRectF(vr.x() - 10, vr.y() - 40, vr.width() + 20, vr.height() + 50), 1.0)
    vh = v.horizontalHeader()

    def front_col():
        x = vh.sectionViewportPosition(1)
        tl = r.wpos(vh, QPointF(x, 0))
        return QRectF(tl.x(), tl.y(), vh.sectionSize(1), vh.height() + v.rowHeight(0) * v.rowCount())
    r.until_word(3, 'In the worst direction')
    r.callout(front_col, '', dur=max(2.6, r.speech_left() - 2.0), pad=2, dim=0.25)
    r.until_word(3, 'twenty-eight')
    r.zoom(None, 0.7)
    r.callout(pg.status, '', dur=max(1.8, r.speech_left() + 0.3), pad=4, dim=0.3)
    r.finish_speech(0.2)

    # ---- 4: stereo pairs
    pr0 = [pg.pairs.item(0, j).text() for j in range(6)]
    print('first pair:', pr0, flush=True)
    r.expect(pr0[:3], ['ML · QL', '59', '1.88'], 'the first stereo pair')
    r.say(4)
    prr = r.wrect(pg.pairs)
    r.zoom(QRectF(prr.x() - 10, prr.y() - 40, prr.width() + 20, min(prr.height(), 260) + 50), 1.0)
    r.hold(max(1.6, r.speech_left()))
    r.finish_speech(0.2)
    r.zoom(None, 0.8)

    # ---- 5: a forward camera on the roof's front edge, narrow lens
    r.say(5)
    r.cursor_on = True
    r.menu('View', 'Back to viewports')
    stage.settle(2)
    vp = w.viewarea.panes[0].vp3d
    r.crossfade(0.4)
    stage.fly(r, vp, az=-35.0, el=28.0, dist=vp.cam.dist * 0.7, target=(1.6, 0.0, 1.0), dur=1.2)
    m = st.project.vehicle.mesh
    t_, f_ = m.raycast(np.array([[1.55, 0.0, 3.0]]), np.array([[0.0, 0.0, -1.0]]))
    roof = np.array([1.55, 0.0, 3.0 - t_[0]])
    r.toolbar('tool_place')
    q = project_pt(r, vp, roof)
    r.hover(vp, q, 0.6)
    stage.mark_result()
    click_sent(r, vp, q, 0.2)
    pump()
    cam = st.variant.cameras[-1]
    print('new camera:', cam.tag, cam.name, cam.location, cam.pos(), cam.yaw, cam.pitch, flush=True)
    r.expect((cam.tag, cam.yaw, cam.pitch), ('T', 0.0, 15.0), 'the roof camera as placed')
    r.toolbar('tool_select')
    st.select_cams([cam.uid])
    r.until_word(5, 'and give it')
    r.menu('Camera', 'Change lens', 'STURDeCAM31 ·', 'H03R2')
    pump()
    r.expect(st.cam(cam.uid).lens_id, 'h03r2', 'the narrow lens')
    r.finish_speech(0.1)

    # ---- 6: the Cameras table: tag and pitch
    r.say(6)
    w.bottom.show_tab('cameras')
    pump()
    tab = w.bottom.cameras
    view = tab.view
    row = next(i for i, c in enumerate(tab.model.cams) if c.uid == cam.uid)
    b = r.wrect(w.docks['bottom'])
    r.zoom(QRectF(b.x(), b.y(), b.width() * 0.62, b.height()), 0.9, wait=False)
    from vantage.ui.bottom.panels import COLS
    ci = [k for k, _ in COLS]
    edit_cell(r, view, row, ci.index('tag'), 'FN')
    edit_cell(r, view, row, ci.index('pitch'), '0')
    pump()
    c2 = st.cam(cam.uid)
    r.expect((c2.tag, c2.yaw, c2.pitch), ('FN', 0.0, 0.0), 'tag and pitch typed in')
    r.hold(0.4)
    r.zoom(None, 0.7)
    w.bottom.show_tab('coverage')
    r.finish_speech(0.1)

    # ---- 7: the page again: the front passes
    r.say(7)
    r.menu('Analysis', 'Perception range')
    r.crossfade(0.4)
    pump()
    r.wait_for(lambda: ctl.result is not None and not ctl.busy and not ctl.stale, timeout=120)
    pump(6)
    car = [v.item(0, j).text() for j in range(v.columnCount())]
    sign = [v.item(3, j).text() for j in range(v.columnCount())]
    print('after:', car, sign, pg.status.text(), flush=True)
    r.expect((car[1], sign[1]), ('138', '92'), 'car and sign ahead with the forward camera')
    r.expect(pg.status.text().startswith('32 of 32 sector checks pass'), True, 'all checks pass')
    vr = r.wrect(v)
    r.zoom(QRectF(vr.x() - 10, vr.y() - 40, vr.width() + 20, vr.height() + 50), 1.0)
    r.callout(front_col, '', dur=2.4, pad=2, dim=0.25)
    r.hold(2.0)
    r.zoom(None, 0.7)
    r.callout(pg.status, '', dur=max(1.6, r.speech_left() + 0.3), pad=4, dim=0.3)
    r.finish_speech(0.2)

    # ---- 8: export
    r.say(8)
    r.cursor_on = True
    r.move_to(pg.b_export, None, 0.7)
    r.callout(pg.b_export, '', dur=1.8, pad=4, dim=0.25)
    r.hold(1.4)
    r.cursor_on = False
    r.zoom(map_box(), 0.9, margin=8)
    r.hold(max(1.0, r.speech_left() - 0.6))
    r.zoom(None, 0.7)
    r.finish_speech(1.0)
