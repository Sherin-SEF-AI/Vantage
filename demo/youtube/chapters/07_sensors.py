"""07 Lidars, radars, GNSS and IMU: Vehicle ▸ Add lidar (the library), the lidar on the mast head with its ground
rings and Inspector figures, the model figures and Import beam angles, the simulated sweep (Lidar points), a
long-range radar on the bumper with its two zones, two GNSS antennas and a GNSS/INS, the heading baseline, the top
view, the Issues list on a radar pushed into the body, and the sample's Data rig.

Set-up (not filmed): a sensor mast on the roof (Vehicle ▸ Add rig part ▸ Sensor mast, as chapter 06 shows).
Figures read from the app while recording (r.expect): the lidar menu's six models; L1 on the mast head; its Inspector
line "128 ch · 360° × 45.0° · 90 m @10% · 1.31 Mpts/s" and "Nearest ground ring 5.3 m"; RD1's zones 250 m ±9° and
70 m ±45°; the Issues list's 14 % for RD1 in camera F; GN1-GN2 1.15 m and heading sigma 0.17°; RD1 25 cm back:
"Radar RD1 sits 231 mm behind the outer surface along its boresight"; the Data rig: one lidar and five radars.
"""

META = {
    'id': 'sensors',
    'part': 1,
    'title': 'Lidars, radars, GNSS and IMU',
    'promise': 'Every sensor on the rig, with its real field, beams and figures.',
    'menu': 'Vehicle ▸ Add lidar · Add radar · Add GNSS antenna · Add IMU',
}

LINES = [
    ("Lidars, radars, GNSS antennas and IMUs sit on the rig next to the cameras. Vehicle ▸ Add lidar lists the "
     "library: Ouster, Hesai, Velodyne and Livox, with nominal figures from their public datasheets.",
     "Lidars, radars, G N S S antennas, and I M Us sit on the rig, next to the cameras. Vehicle, add lidar, lists the "
     "library: Ouster, Hesai, Velodyne, and Livox, with nominal figures from their public datasheets."),
    ("A lidar goes on top of the highest mountable part, here the mast. The rings are where its downward beams meet "
     "the ground: the nearest, 5.3 metres from the lidar.",
     "A lidar goes on top of the highest mountable part, here, the mast. The rings are where its downward beams meet "
     "the ground: the nearest, five point three metres from the lidar."),
    ("The Inspector edits it like a camera, and sums up the model: 128 channels over 45 degrees, 90 metres on a "
     "10 per cent target, 1.31 million points a second.",
     "The inspector edits it like a camera, and sums up the model: a hundred and twenty-eight channels over forty-five "
     "degrees, ninety metres on a ten per cent target, one point three one million points a second."),
    ("Editing a built-in model makes an edited copy for this sensor. Import beam angles reads the measured channel "
     "elevations from an Ouster, Hesai or Velodyne calibration file.",
     "Editing a built-in model makes an edited copy for this sensor. Import beam angles reads the measured channel "
     "elevations from an Ouster, Hesai, or Velodyne calibration file."),
    ("Lidar points draws one simulated sweep, coloured by range: beams that the body or the rig stops are red.",
     "Lidar points draws one simulated sweep, coloured by range: beams that the body or the rig stops are red."),
    ("Radars are typical 77-gigahertz classes, not products. The long-range class goes to the middle of the front "
     "bumper, with a far beam to 250 metres and a near beam to 70.",
     "Radars are typical seventy-seven gigahertz classes, not products. The long range class goes to the middle of "
     "the front bumper, with a far beam to two hundred and fifty metres, and a near beam to seventy."),
    ("The Issues list already notices that its housing fills 14 per cent of the front camera's image: the next "
     "chapter is about exactly that.",
     "The issues list already notices that its housing fills fourteen per cent of the front camera's image: the next "
     "chapter is about exactly that."),
    ("GNSS antennas and an IMU go on the same way: the first antenna near the rear of the roof, the second near the "
     "front for the longest heading baseline, the INS on the cabin floor above the rear axle.",
     "G N S S antennas and an I M U go on the same way: the first antenna near the rear of the roof, the second near "
     "the front, for the longest heading baseline, and the I N S on the cabin floor, above the rear axle."),
    ("The INS's Inspector gives the baseline, 1.15 metres, and the heading error it allows: 0.17 degrees.",
     "The I N S's inspector gives the baseline, one point one five metres, and the heading error it allows: zero point "
     "one seven degrees."),
    ("From the top: the lidar's rings, the radar's fields and the antennas' baseline, each with its tag.",
     "From the top: the lidar's rings, the radar's fields, and the antennas' baseline, each with its tag."),
    ("Push the radar 25 centimetres back, and the Issues list says it sits 231 millimetres behind the outer surface "
     "along its boresight: a radar sees through a plastic fascia, not through the body.",
     "Push the radar twenty-five centimetres back, and the issues list says it sits two hundred and thirty-one "
     "millimetres behind the outer surface, along its boresight: a radar sees through a plastic fascia, not through "
     "the body."),
    ("The sample's Data rig variant carries a roof lidar and five radars; every sensor is exported with the cameras, "
     "to the extrinsics, the URDF, nuScenes, KITTI and the report.",
     "The sample's data rig variant carries a roof lidar and five radars. Every sensor is exported with the cameras: "
     "to the extrinsics, the U R D F, nuScenes, KITTI, and the report."),
]


def scene(stage, r):
    import numpy as np
    from PyQt6.QtCore import QPointF, QRectF

    from helpers_tour_a import project_pt
    from vantage.ui.sensorview import set_props

    w = stage.window()
    r.win = w
    st = w.store
    app = stage.app
    w.actions_by_id['rig_sensor_mast'].trigger()        # set-up, not filmed: a mast on the roof
    w.ctx.set_overlay('camlight', False)
    stage.settle(10)
    stage.mark_result()
    vp = stage.vp
    vp.cam.az, vp.cam.el = -38.0, 22.0
    w.viewarea.frame_all()
    stage.settle(2)
    page = w.inspector.pages['sensor']
    iss = w.bottom.issues

    def pump(n=4):
        for _ in range(n):
            app.processEvents()

    def issue_rows():
        iss.refresh()
        return [iss.t.item(i, 1).text() for i in range(iss.t.rowCount()) if iss.t.item(i, 1)]

    # ---- 0: Vehicle ▸ Add lidar: the library
    r.cursor_on = True
    r.lower_third()
    r.say(0)
    r.hold(0.6)
    r.peek_menu('Vehicle', 'Add lidar', 'Livox', hold=max(1.0, r.speech_left() - 6.4))
    act = r.menu('Vehicle', 'Add lidar', 'Ouster OS1-128')
    pump()
    lid = next(s for s in st.variant.sensors if s.kind == 'lidar')
    print('lidar:', lid.tag, lid.name, lid.pos(), flush=True)
    r.expect(lid.tag, 'L1', 'the lidar tag')
    stage.wait_analysis(r)
    r.finish_speech(0.1)

    # ---- 1: on the mast head; the ground rings
    r.say(1)
    r.cursor_on = False
    stage.fly(r, vp, az=-52.0, el=34.0, dist=vp.cam.dist * 1.5, target=(1.0, 0.0, 0.6), dur=2.2)
    info = page.info.text()
    print('L1 info:', info, flush=True)
    r.expect('Nearest ground ring 5.3 m from the lidar' in info, True, 'nearest ring')
    r.until_word(1, 'The rings')
    stage.orbit(r, vp, 30, max(3.0, r.speech_left() + 0.3), d_el=4)
    r.finish_speech(0.1)

    # ---- 2: the Inspector
    r.say(2)
    r.expect(info.startswith('128 ch · 360° × 45.0° · 90 m @10% · 1.31 Mpts/s'), True, 'the lidar summary')
    insp = r.wrect(w.docks['inspector'])
    r.zoom(QRectF(insp.x() - 40, insp.y(), insp.width() + 40, 420), 1.0)
    r.callout(page.info, '', dur=max(2.6, r.speech_left() - 0.6), pad=4, dim=0.25)
    r.finish_speech(0.2)

    # ---- 3: model figures, Import beam angles
    r.say(3)
    r.cursor_on = True
    sa = page.parentWidget()
    from PyQt6.QtWidgets import QScrollArea
    while sa is not None and not isinstance(sa, QScrollArea):
        sa = sa.parentWidget()
    if sa is not None:
        sa.ensureWidgetVisible(page.fig_note, 0, 40)
        pump()
    r.zoom(QRectF(insp.x() - 40, insp.y(), insp.width() + 40, insp.height()), 0.8)
    r.callout(page.fig_note, '', dur=2.4, pad=4, dim=0.25)
    r.until_word(3, 'Import beam angles')
    if sa is not None:
        sa.ensureWidgetVisible(page.b_beams, 0, 60)
        pump()
    r.move_to(page.b_beams, None, 0.7)
    r.callout(page.b_beams, '', dur=max(2.0, r.speech_left()), pad=4, dim=0.25)
    r.finish_speech(0.2)
    if sa is not None:
        sa.verticalScrollBar().setValue(0)
    r.zoom(None, 0.8)

    # ---- 4: Lidar points
    r.say(4)
    r.menu('View', 'Overlays', 'Lidar points')
    r.cursor_on = False
    ov = w.viewarea.panes[0].scene._ov3d()
    r.tick(3)
    r.wait_for(lambda: not ov.busy(), timeout=60)      # the sweep is worked out in the background, in real time
    print('sweeps blocked:', {v.tag: round(v.blocked, 3) for v in ov.sweeps.values()}, flush=True)
    r.tick(2)
    stage.fly(r, vp, az=-20.0, el=30.0, dist=vp.cam.dist * 0.75, target=(1.2, 0.0, 0.9), dur=1.4)
    stage.orbit(r, vp, 30, max(1.6, r.speech_left() + 0.2), d_el=3)
    r.finish_speech(0.1)
    w.ctx.set_overlay('lidarpts', False)
    pump()
    r.crossfade(0.5)

    # ---- 5: a long-range radar on the bumper
    r.say(5)
    r.cursor_on = True
    stage.fly(r, vp, az=28.0, el=22.0, dist=vp.cam.dist * 0.9, target=(2.6, 0.0, 0.5), dur=1.4)
    r.menu('Vehicle', 'Add radar', 'Long-range radar')
    pump()
    rad = next(s for s in st.variant.sensors if s.kind == 'radar')
    stage.wait_analysis(r)
    zones = [[page.zones.item(i, j).text() for j in range(4)] for i in range(page.zones.rowCount())]
    print('RD1 zones:', zones, flush=True)
    r.expect(zones, [['1', '250', '9', '7'], ['2', '70', '45', '7']], "RD1's zones")
    r.cursor_on = False
    q = project_pt(r, vp, rad.pos())
    rq = r.wpos(vp, q)
    r.callout((rq.x() - 40, rq.y() - 30, 80, 60), 'RD1 · long-range radar', dur=2.6, style='circle', side='above')
    r.hold(2.0)
    r.zoom(QRectF(insp.x() - 40, insp.y(), insp.width() + 40, 520), 0.9, wait=False)
    r.callout(page.zones, '', dur=max(2.0, r.speech_left()), pad=3, dim=0.25)
    r.finish_speech(0.2)
    r.zoom(None, 0.7)

    # ---- 6: the Issues list notices the radar in camera F
    r.say(6)
    w.bottom.show_tab('issues')
    pump()
    rows_ = issue_rows()
    print('issues:', rows_, flush=True)
    r.expect(any("radar RD1's housing covers 14 % of its image" in x for x in rows_), True, 'RD1 in F')
    bb = r.wrect(w.docks['bottom'])
    r.zoom(QRectF(bb.x(), bb.y() - 10, min(bb.width() * 0.55, 1020), bb.height() + 10), 0.9)
    r.callout(iss, '', dur=max(2.4, r.speech_left()), dim=0.0, pad=-6)
    r.finish_speech(0.2)
    r.zoom(None, 0.7)
    w.bottom.show_tab('coverage')

    # ---- 7: GNSS antennas and an INS
    r.say(7)
    r.cursor_on = True
    stage.fly(r, vp, az=-60.0, el=30.0, dist=vp.cam.dist * 1.1, target=(0.9, 0.0, 0.8), dur=1.2)
    r.menu('Vehicle', 'Add GNSS antenna', 'Survey GNSS antenna')
    r.hold(0.5)
    r.until_word(7, 'the second')
    r.menu('Vehicle', 'Add GNSS antenna', 'Survey GNSS antenna')
    r.until_word(7, 'and the I N S')
    r.menu('Vehicle', 'Add IMU', 'Dual-antenna GNSS/INS')
    stage.wait_analysis(r)
    print('sensors:', [(s.tag, s.kind, s.pos()) for s in st.variant.sensors], flush=True)
    r.finish_speech(0.1)

    # ---- 8: the baseline and the heading error
    r.say(8)
    ins = next(s for s in st.variant.sensors if s.kind == 'imu')
    st.select_obj(('sensor', ins.uid))
    stage.settle(2)
    info = page.info.text()
    print('INS info:', info, flush=True)
    r.expect('Heading baseline GN1–GN2 1.15 m: heading σ 0.17°.' in info, True, 'baseline and heading')
    r.zoom(QRectF(insp.x() - 40, insp.y(), insp.width() + 40, 420), 0.9)
    r.callout(page.info, '', dur=max(2.6, r.speech_left()), pad=4, dim=0.25)
    r.finish_speech(0.2)
    r.zoom(None, 0.7)

    # ---- 9: the top view
    r.say(9)
    st.select_cams([])
    w.set_view('top')
    stage.settle(3)
    plan = w.viewarea.panes[0].plan
    r.crossfade(0.4)
    r.cursor_on = False
    s0 = plan.scale
    n = round(2.4 * stage.fps)
    for k in range(1, n + 1):
        e = k / n
        e = e * e * (3 - 2 * e)
        plan.scale = s0 * (1 - 0.62 * e)
        plan._user_view = True
        plan.update()
        r.tick()
    r.hold(max(1.0, r.speech_left() - 1.6))
    pr = r.wrect(plan)
    r.zoom(QRectF(pr.center().x() - 380, pr.center().y() - 210, 760, 420), 1.0)
    r.hold(max(0.6, r.speech_left()))
    r.zoom(None, 0.7)
    w.set_view('persp')
    stage.settle(2)
    r.crossfade(0.4)

    # ---- 10: a radar pushed into the body
    r.say(10)
    r.cursor_on = True
    st.select_obj(('sensor', rad.uid))
    stage.settle(2)
    r.zoom(QRectF(insp.x() - 40, insp.y(), insp.width() + 40, 360), 0.8)
    x0 = rad.x
    r.enter(page.px, f'{x0:.3f} - 0.25', cps=12)
    if abs(st.variant.sensor(rad.uid).x - (x0 - 0.25)) > 1e-4:
        set_props(w.ctx, rad.uid, f'Moved {rad.tag}', x=round(x0 - 0.25, 4))
    stage.wait_analysis(r)
    r.zoom(None, 0.7)
    w.bottom.show_tab('issues')
    pump()
    rows_ = issue_rows()
    print('issues after:', rows_, flush=True)
    r.expect(any(x.startswith('Radar RD1 sits 231 mm behind the outer surface along its boresight') for x in rows_),
             True, 'the radar issue')
    r.zoom(QRectF(bb.x(), bb.y() - 10, min(bb.width() * 0.55, 1020), bb.height() + 10), 0.9)
    r.hold(max(1.6, r.speech_left()))
    r.zoom(None, 0.7)
    w.bottom.show_tab('coverage')
    r.finish_speech(0.2)

    # ---- 11: the Data rig
    r.say(11)
    data = next(v for v in st.project.variants if v.name.startswith('Data rig'))
    r.expect(sum(1 for s in data.sensors if s.kind == 'lidar'), 1, 'lidars in the Data rig')
    r.expect(sum(1 for s in data.sensors if s.kind == 'radar'), 5, 'radars in the Data rig')
    r.choose(w.variant_combo, w.variant_combo.findData(data.uid))
    stage.wait_analysis(r)
    vp.cam.az, vp.cam.el = -36.0, 20.0
    w.viewarea.frame_all()
    r.crossfade(0.5)
    r.cursor_on = False
    stage.orbit(r, vp, 60, max(4.0, r.speech_left() + 0.6), d_el=5)
    r.finish_speech(1.0)
