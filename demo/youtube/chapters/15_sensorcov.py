"""15 Lidar and radar coverage, on the sample's Data rig (roof lidar L1, OS1-128; radars RD1-RD5): what the roof
blocks of the lidar, its rings and ground blind zone (the cell's tooltip), the plan layers, the object thresholds, the
per-sector verdicts by sensor type (a car ahead: single; a pedestrian ahead: a gap), lidar points in each camera, the
radars' reach, and the lidar raised 0.6 m in the Inspector with the page following.

Figures read from the app while recording (r.expect): "28 redundant · 1 single · 3 gaps of 32"; L1 35.2 % on the car,
ring 3.8 m, blind from the body 8.5 m at best and 16.1 m at worst; car ahead camera 70, lidar 58, radar 102 m, single;
pedestrian ahead 66 / 51 / 49 m, gap; F's image 22 % covered at 20 m; RD1 a car at ≥250 m, a pedestrian at 120 m;
raised 0.6 m: 0.0 % on the car, ground reached from 2.9 to 4.4 m, F 33 %.
"""

META = {
    'id': 'sensorcov',
    'part': 2,
    'title': 'Lidar and radar coverage',
    'promise': 'What the lidars and radars add to the cameras, sector by sector.',
    'menu': 'Analysis ▸ Lidar and radar coverage…',
}

LINES = [
    ("Lidars and radars have coverage of their own. On the Data rig, with its roof lidar and five radars, Analysis ▸ "
     "Lidar and radar coverage works out where they reach and what the vehicle blocks.",
     "Lidars and radars have coverage of their own. On the data rig, with its roof lidar and five radars, analysis, "
     "lidar and radar coverage, works out where they reach, and what the vehicle blocks."),
    ("The roof lidar sits low on the roof: 35 per cent of every frame's points land on the roof itself.",
     "The roof lidar sits low on the roof: thirty-five per cent of every frame's points land on the roof itself."),
    ("So the nearest ground it reaches is 8.5 to 16 metres from the body, depending on the direction. Closer than "
     "that, the ground gets no lidar points at all.",
     "So the nearest ground it reaches is eight point five to sixteen metres from the body, depending on the "
     "direction. Closer than that, the ground gets no lidar points at all."),
    ("The plan shows the ground density in points per square metre per frame, or each channel's ring where the "
     "vehicle does not block it.",
     "The plan shows the ground density, in points per square metre per frame, or each channel's ring, where the "
     "vehicle does not block it."),
    ("Each object needs so many lidar points to be detected, 20 on a car and 8 on a pedestrian, and has a radar "
     "cross-section; it stands in 360 directions out to 250 metres.",
     "Each object needs so many lidar points to be detected, twenty on a car, and eight on a pedestrian, and has a "
     "radar cross-section. It stands in three hundred and sixty directions, out to two hundred and fifty metres."),
    ("Per sector, the shortest camera, lidar and radar range meets the required one. A car 80 metres ahead: the "
     "cameras reach 70, the lidar 58, only the radar 102. A single sensor type.",
     "Per sector, the shortest camera, lidar, and radar range meets the required one. A car eighty metres ahead: the "
     "cameras reach seventy, the lidar fifty-eight, only the radar a hundred and two. A single sensor type."),
    ("A pedestrian ahead is a gap: 66 metres by camera, 51 by lidar, 49 by radar, none of them 80.",
     "A pedestrian ahead is a gap: sixty-six metres by camera, fifty-one by lidar, forty-nine by radar. None of them "
     "eighty."),
    ("Lidar points in each camera gives the share of every image a lidar covers, for labelling and for "
     "cross-calibration: 22 per cent of the front camera's at 20 metres.",
     "Lidar points in each camera gives the share of every image a lidar covers, for labelling, and for "
     "cross-calibration: twenty-two per cent of the front camera's, at twenty metres."),
    ("Raise the lidar 60 centimetres on a riser, typed into the Inspector as plus 0.6.",
     "Raise the lidar sixty centimetres on a riser, typed into the inspector as plus zero point six."),
    ("The page runs again: nothing lands on the roof, the ground is reached from 2.9 to 4.4 metres out, and a third "
     "of the front camera's image gets lidar points.",
     "The page runs again: nothing lands on the roof, the ground is reached from two point nine to four point four "
     "metres out, and a third of the front camera's image gets lidar points."),
    ("The radars: the long-range one detects a car beyond 250 metres and a pedestrian at 120. All of it exports as "
     "JSON with the plan, and the assistant can run it.",
     "The radars: the long range one detects a car beyond two hundred and fifty metres, and a pedestrian at a hundred "
     "and twenty. All of it exports as JSON with the plan, and the assistant can run it."),
]


def scene(stage, r):
    from PyQt6.QtCore import QPointF, QRectF

    w = stage._rig_window()
    r.win = w
    st = w.store
    app = stage.app

    def pump(n=4):
        for _ in range(n):
            app.processEvents()

    def cell_rect(t, row, col):
        def f():
            rr = t.visualItemRect(t.item(row, col))
            tl = r.wpos(t.viewport(), QPointF(rr.topLeft()))
            return QRectF(tl.x(), tl.y(), rr.width(), rr.height())
        return f

    def row_rect(t, row):
        def f():
            a = t.visualItemRect(t.item(row, 0))
            b = t.visualItemRect(t.item(row, t.columnCount() - 1))
            tl = r.wpos(t.viewport(), QPointF(a.topLeft()))
            return QRectF(tl.x(), tl.y(), b.right() - a.left(), a.height())
        return f

    # ---- 0: the page
    r.cursor_on = True
    r.lower_third()
    r.say(0)
    vp = stage.vp
    stage.orbit(r, vp, 30, 2.6, d_el=4)
    r.until_word(0, 'analysis, lidar')
    r.menu('Analysis', 'Lidar and radar coverage')
    pg = w.page('coverage')
    ctl = pg.ctl
    r.crossfade(0.4)
    r.wait_for(lambda: ctl.result is not None and not ctl.busy, timeout=120)
    pump(6)
    stt = pg.status.text()
    print('status:', stt, flush=True)
    r.expect(stt.startswith('28 redundant · 1 single · 3 gaps of 32 sector checks'), True, 'the verdict count')
    r.cursor_on = False
    r.finish_speech(0.2)

    # ---- 1: on the car
    li = [pg.lidars.item(0, j).text() for j in range(7)]
    tip = pg.lidars.item(0, 3).toolTip()
    print('lidar row:', li, '|', tip, flush=True)
    r.expect(li[1:3], ['35.2', '3.8'], 'L1 on the car and its ring')
    r.expect(tip.startswith('From the body outline: 8.5 m at best, 16.1 m at worst'), True, 'L1 ground blind zone')
    r.say(1)
    lt = r.wrect(pg.lidars)
    r.zoom(QRectF(lt.x() - 10, lt.y() - 40, lt.width() + 20, lt.height() + 60), 1.0)
    r.callout(cell_rect(pg.lidars, 0, 1), '', dur=max(2.4, r.speech_left()), pad=3, dim=0.25)
    r.finish_speech(0.2)

    # ---- 2: the ground blind zone (the cell's tooltip)
    r.say(2)
    r.cursor_on = True
    cr = cell_rect(pg.lidars, 0, 3)()
    r.move(cr.center(), 0.7)
    r.hold(0.3)
    # the cell's tooltip (an offscreen tooltip lands at the screen's edge): its text on a callout instead
    r.callout(cell_rect(pg.lidars, 0, 3), tip.split(';')[0], dur=max(3.0, r.speech_left()), pad=3, dim=0.25,
              side='below')
    r.finish_speech(0.2)
    r.zoom(None, 0.8)

    # ---- 3: the plan: density, rings
    r.say(3)
    mw = r.wrect(pg.map)
    side = min(mw.width(), mw.height()) * 0.86
    r.zoom(QRectF(mw.center().x() - side / 2, mw.center().y() - side / 2, side, side), 0.9, margin=8)
    r.until_word(3, "or each channel's ring")
    r.click(pg.layer.buttons['rings'], None, 0.6)
    pump()
    r.hold(max(1.4, r.speech_left()))
    r.finish_speech(0.2)

    # ---- 4: object thresholds
    cls = [[pg.classes.item(i, j).text() for j in range(4)] for i in range(pg.classes.rowCount())]
    print('classes:', cls, flush=True)
    r.expect([c[2] for c in cls[:2]], ['20', '8'], 'lidar points needed')
    r.say(4)
    r.click(pg.layer.buttons['density'], None, 0.5)
    ct = r.wrect(pg.classes)
    r.zoom(QRectF(ct.x() - 10, ct.y() - 40, ct.width() + 20, ct.height() + 60), 1.0)
    r.callout(pg.classes, '', dur=max(2.6, r.speech_left() - 0.4), pad=2, dim=0.25)
    r.finish_speech(0.2)
    r.zoom(None, 0.8)

    # ---- 5: the sector verdicts: a car ahead
    t = pg.sectors
    front = [t.item(0, j).text() for j in range(6)]
    print('car ahead:', front, flush=True)
    r.expect(front, ['Front', '80', '70', '58', '102', 'Single'], 'car ahead')
    r.say(5)
    sr = r.wrect(pg.sectors)
    gr = r.wrect(pg.grid)
    r.zoom(QRectF(sr.x() - 10, gr.y() - 30, sr.width() + 20, sr.bottom() - gr.y() + 40), 1.0)
    r.until_word(5, 'A car eighty')
    r.callout(row_rect(t, 0), '', dur=max(2.8, r.speech_left()), pad=2, dim=0.25)
    r.finish_speech(0.2)

    # ---- 6: a pedestrian ahead: a gap
    r.say(6)
    r.cursor_on = True
    r.click(pg.pick.buttons['1'], None, 0.6)
    pump()
    front = [t.item(0, j).text() for j in range(6)]
    print('pedestrian ahead:', front, flush=True)
    r.expect(front, ['Front', '80', '66', '51', '49', 'Gap'], 'pedestrian ahead')
    r.callout(row_rect(t, 0), '', dur=max(2.4, r.speech_left() + 0.2), pad=2, dim=0.25)
    r.finish_speech(0.2)

    # ---- 7: lidar points in each camera
    ov = [pg.overlap.item(0, j).text() for j in range(6)]
    print('overlap F:', ov, flush=True)
    r.expect((ov[0], ov[2]), ('F', '22'), "F's image covered at 20 m")
    r.say(7)
    r.click(pg.pick.buttons['0'], None, 0.5)
    ot = r.wrect(pg.overlap)
    r.zoom(QRectF(ot.x() - 10, ot.y() - 40, ot.width() + 20, min(ot.height(), 220) + 60), 1.0)
    r.until_word(7, 'twenty-two per cent')
    r.callout(cell_rect(pg.overlap, 0, 2), '', dur=max(2.0, r.speech_left() + 0.3), pad=3, dim=0.25)
    r.finish_speech(0.2)
    r.zoom(None, 0.7)

    # ---- 8: raise the lidar 0.6 m in the Inspector
    r.say(8)
    r.menu('View', 'Back to viewports')
    stage.settle(2)
    r.crossfade(0.4)
    lid = next(s for s in st.variant.sensors if s.kind == 'lidar')
    row = r.tree_item(w.outliner, 'L1')
    rr = row()
    if rr is not None:
        r.move(QPointF(rr.x() + 70, rr.center().y()), 0.7)
        r.ripple()
    st.select_obj(('sensor', lid.uid))
    stage.settle(2)
    sp = w.inspector.pages['sensor']
    insp = r.wrect(w.docks['inspector'])
    r.zoom(QRectF(insp.x() - 40, insp.y(), insp.width() + 40, 300), 0.8, wait=False)
    z0 = lid.z
    r.enter(sp.pz, f'{z0:.3f}+0.6', cps=10)
    if abs(st.variant.sensor(lid.uid).z - (z0 + 0.6)) > 1e-4:
        sp.pz.committed.emit(round(z0 + 0.6, 3))
    pump()
    r.expect(f'{st.variant.sensor(lid.uid).z:.3f}', f'{z0 + 0.6:.3f}', 'the raised lidar')
    r.zoom(None, 0.7)
    r.finish_speech(0.1)

    # ---- 9: the page again
    r.say(9)
    r.menu('Analysis', 'Lidar and radar coverage')
    r.crossfade(0.4)
    pump()
    r.wait_for(lambda: ctl.result is not None and not ctl.busy and not ctl.stale, timeout=120)
    pump(6)
    li = [pg.lidars.item(0, j).text() for j in range(7)]
    tip = pg.lidars.item(0, 3).toolTip()
    ov = [pg.overlap.item(0, j).text() for j in range(6)]
    print('raised:', li, '|', tip, '|', ov, flush=True)
    r.expect(li[1], '0.0', 'on the car, raised')
    r.expect(tip.startswith('From the body outline: 2.9 m at best, 4.4 m at worst'), True, 'ground reached, raised')
    r.expect(ov[2], '33', "F's image covered, raised")
    lt = r.wrect(pg.lidars)
    r.zoom(QRectF(lt.x() - 10, lt.y() - 40, lt.width() + 20, lt.height() + 60), 1.0)
    r.callout(cell_rect(pg.lidars, 0, 1), '', dur=1.8, pad=3, dim=0.25)
    r.until_word(9, 'the ground is reached')
    r.callout(cell_rect(pg.lidars, 0, 3), '', dur=1.8, pad=3, dim=0.25)
    r.until_word(9, 'a third')
    r.zoom(None, 0.6)
    ot = r.wrect(pg.overlap)
    r.zoom(QRectF(ot.x() - 10, ot.y() - 40, ot.width() + 20, min(ot.height(), 220) + 60), 0.8)
    r.callout(cell_rect(pg.overlap, 0, 2), '', dur=max(1.6, r.speech_left() + 0.3), pad=3, dim=0.25)
    r.finish_speech(0.2)
    r.zoom(None, 0.7)

    # ---- 10: the radars; export
    rd = [pg.radars.item(0, j).text() for j in range(pg.radars.columnCount())]
    print('RD1:', rd, flush=True)
    r.expect(rd[1:3], ['≥250', '120'], 'RD1 car and pedestrian')
    r.say(10)
    rt = r.wrect(pg.radars)
    r.zoom(QRectF(rt.x() - 10, rt.y() - 40, rt.width() + 20, rt.height() + 60), 1.0)
    r.callout(row_rect(pg.radars, 0), '', dur=2.6, pad=2, dim=0.25)
    r.until_word(10, 'All of it exports')
    r.zoom(None, 0.7)
    r.cursor_on = True
    r.move_to(pg.b_export, None, 0.7)
    r.callout(pg.b_export, '', dur=max(1.6, r.speech_left()), pad=4, dim=0.25)
    r.cursor_on = False
    r.finish_speech(1.0)
