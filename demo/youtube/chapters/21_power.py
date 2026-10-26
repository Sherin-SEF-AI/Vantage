"""21 Power budget: Analysis ▸ Power budget on the sample's data rig with two GNSS antennas and a dual-antenna INS
(stage._nav_window). Every device's steady and peak power, the DC-DC converter, the alternator, the engine-off runtime,
the fused circuits; a 5 A fuse typed on the compute circuit is over its current (critical); the 24 V bus carries the
same power at half the current, and the fuse holds; the cable drops; the enclosure's heat.

Figures read from the app while recording (r.expect): 183 W steady, 236 W peak, 19 devices; DC-DC 31 % of 600 W;
alternator +491 W; engine-off runtime 241 min against a 30 min target; compute circuit 5.7 A against a typed 5 A fuse
(113 %); at 24 V 2.8 A (57 %); the worst cable drop at 24 V, L1, 0.2 %; enclosure heat 136 W.
"""

META = {
    'id': 'power',
    'part': 3,
    'title': 'Power budget',
    'promise': 'What the rig draws, and whether the supply carries it.',
    'menu': 'Analysis ▸ Power budget',
}

LINES = [
    ("Every sensor on the rig needs power, and the car has only so much to spare. The power budget checks the whole "
     "chain, from the alternator to each fuse and cable.",
     "Every sensor on the rig needs power, and the car has only so much to spare. The power budget checks the whole "
     "chain, from the alternator to each fuse and cable."),
    ("Each device draws its steady power: the module's figure for a camera, the library's nominal figures for the "
     "lidar, radars, antennas and INS, and typical figures for the computer, recorder, switch and time server.",
     "Each device draws its steady power: the module's figure for a camera, the library's nominal figures for the "
     "lidar, radars, antennas, and I N S, and typical figures for the computer, recorder, switch, and time server."),
    ("At switch-on the lidar's motor draws twice its steady power: 183 W steady for nineteen devices, 236 W at peak.",
     "At switch on, the lidar's motor draws twice its steady power: a hundred and eighty-three watts steady, for "
     "nineteen devices, and two hundred and thirty-six at peak."),
    ("The DC-DC converter runs at 31 per cent of its 600 W, the alternator keeps 491 W to spare, and with the engine "
     "off the auxiliary battery runs the rig for 241 minutes.",
     "The D C to D C converter runs at thirty-one per cent of its six hundred watts, the alternator keeps four hundred "
     "and ninety-one watts to spare, and with the engine off, the auxiliary battery runs the rig for two hundred and "
     "forty-one minutes."),
    ("Devices are grouped into fused circuits. Each fuse is sized automatically from the steady and peak currents, "
     "until you enter the one you actually fit.",
     "Devices are grouped into fused circuits. Each fuse is sized automatically from the steady and peak currents, "
     "until you enter the one you actually fit."),
    ("Type the 5 A fuse fitted to the compute circuit: the Jetson and six cameras draw 5.7 A, the fuse would blow, "
     "and the finding says what to fit instead.",
     "Type the five amp fuse fitted to the compute circuit: the Jetson and six cameras draw five point seven amps, the "
     "fuse would blow, and the finding says what to fit instead."),
    ("On a 24 V bus the same power needs half the current: the compute circuit draws 2.8 A, and the 5 A fuse holds.",
     "On a twenty-four volt bus, the same power needs half the current: the compute circuit draws two point eight "
     "amps, and the five amp fuse holds."),
    ("Every lidar, radar and INS power cable gets its voltage drop at constant power, against a 5 per cent limit: the "
     "worst, the lidar's, is now 0.2 per cent.",
     "Every lidar, radar, and I N S power cable gets its voltage drop at constant power, against a five per cent "
     "limit. The worst, the lidar's, is now zero point two per cent."),
    ("And the enclosure must shed 136 W of heat. Library figures are nominal, not measured: replace them with your "
     "own, and export the budget as JSON and CSV.",
     "And the enclosure must shed a hundred and thirty-six watts of heat. Library figures are nominal, not measured: "
     "replace them with your own, and export the budget as Jason and C S V."),
]


def scene(stage, r):
    import numpy as np
    from PyQt6.QtCore import QPointF, QRectF, Qt
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QLineEdit

    w = stage._nav_window()
    r.win = w
    r.cursor_on = False
    vp = stage.vp
    vp.cam.az, vp.cam.el = -40.0, 26.0
    w.viewarea.frame_all()
    stage.settle(2)
    r.cur = np.array([1150.0, 560.0])

    # ---- 0: the opener; the page from its menu
    r.lower_third()
    r.say(0)
    stage.orbit(r, vp, 30, 3.0, d_el=3)
    r.cursor_on = True
    r.menu('Analysis', 'Power budget')
    pg = w.page('power')
    stage.settle(2)
    r.crossfade(0.4)
    r.finish_speech(0.1)
    t, ct = pg.table, pg.circ
    tot = pg.result.totals

    def rect_of(tb, r0, c0, r1, c1):
        return lambda: QRectF(r.local_rect(tb.viewport(), QRectF(tb.visualItemRect(tb.item(r0, c0))))()).united(
            r.local_rect(tb.viewport(), QRectF(tb.visualItemRect(tb.item(r1, c1))))())

    def row(tb, tag):
        return next(i for i in range(tb.rowCount()) if tb.item(i, 0) and tb.item(i, 0).text() == tag)

    tiles = r.wrect(pg.tiles['load']).united(r.wrect(pg.tiles['heat']))
    tiles_zoom = QRectF(tiles.x(), tiles.y() - 10, tiles.width(), r.wrect(pg.bar).bottom() - tiles.y() + 20)

    # ---- 1: every device
    r.say(1)
    tr = r.wrect(t)
    r.zoom(QRectF(tr.x(), tr.y() - 20, tr.width() * 0.62, tr.height() + 20), 1.0)
    r.callout(rect_of(t, row(t, 'F'), 0, row(t, 'QR'), 3), '', dur=2.2, pad=3, dim=0.2)
    r.until_word(1, "the library's")
    r.callout(rect_of(t, row(t, 'L1'), 0, row(t, 'IMU1'), 3), '', dur=2.6, pad=3, dim=0.2)
    r.until_word(1, 'and typical figures')
    r.callout(rect_of(t, row(t, 'Compute'), 0, row(t, 'PTP'), 3), '', dur=max(1.8, r.speech_left() + 0.3), pad=3,
              dim=0.2)
    r.finish_speech(0.2)

    # ---- 2: steady and peak
    r.say(2)
    i_l1 = row(t, 'L1')
    r.expect(f'{t.item(i_l1, 3).text()}/{t.item(i_l1, 4).text()}', '20.0/40.0', "the lidar's steady and peak power")
    r.expect(pg.tiles['load'].value, '183', 'steady load')
    r.expect(pg.tiles['load'].note.startswith('236 W peak · 19 devices'), True, 'peak and device count')
    pk = rect_of(t, i_l1, 4, i_l1, 4)()
    r.zoom(QRectF(tr.x(), tr.y() - 20, pk.right() - tr.x() + 40, tr.height() + 20), 0.8)
    r.callout(rect_of(t, i_l1, 3, i_l1, 4), '', dur=2.4, pad=3, dim=0.2)
    r.until_word(2, 'a hundred and eighty-three')
    r.zoom(tiles_zoom, 1.0)
    r.figure('Rig load', float(pg.tiles['load'].value), 'W', 0, dur=max(3.0, r.speech_left() + 0.6),
             src=pg.tiles['load'], note='236 W peak · 19 devices')
    r.finish_speech(0.2)

    # ---- 3: converter, alternator, runtime
    r.say(3)
    r.expect(pg.tiles['dcdc'].value, '31', 'DC-DC load')
    r.expect(pg.tiles['alternator'].value, '+491', 'alternator margin')
    r.expect(pg.tiles['runtime'].value, '241', 'engine-off runtime')
    r.callout(pg.tiles['dcdc'], '', dur=2.4, pad=3, dim=0.2)
    r.until_word(3, 'the alternator keeps')
    r.callout(pg.tiles['alternator'], '', dur=2.2, pad=3, dim=0.2)
    r.until_word(3, 'with the engine off')
    r.figure('Engine-off runtime', float(pg.tiles['runtime'].value), 'min', 0, dur=max(2.8, r.speech_left() + 0.6),
             src=pg.tiles['runtime'], note='target 30 min')
    r.finish_speech(0.2)

    # ---- 4: the circuits
    r.say(4)
    cr = r.wrect(ct)
    r.zoom(QRectF(cr.x() - 10, cr.y() - 40, cr.width() + 20, cr.height() + 50), 1.0)
    r.until_word(4, 'Each fuse')
    r.callout(rect_of(ct, 0, 2, ct.rowCount() - 1, 2), '', dur=max(2.4, r.speech_left() + 0.4), pad=3, dim=0.2)
    r.finish_speech(0.2)

    # ---- 5: a 5 A fuse typed on the compute circuit
    r.say(5)
    i_c = row(ct, 'Compute')
    item = ct.item(i_c, 2)
    rect = ct.visualItemRect(item)
    r.click(ct.viewport(), QPointF(rect.center()), 0.7, real=False)
    ct.editItem(item)
    r.tick(3)
    ed = next((e for e in ct.viewport().findChildren(QLineEdit) if e.isVisible()), None)
    if ed is None:
        item.setText('5')
    else:
        ed.selectAll()
        r.hold(0.3)
        QTest.keyClicks(ed, '5')
        r.hold(0.3)
        QTest.keyClick(ed, Qt.Key.Key_Return)
    r.wait_for(lambda: pg.result.circuit('compute')['fuse_a'] == 5.0, timeout=10)
    stage.settle(1)
    r.move_to(ct.viewport(), QPointF(ct.viewport().width() * 0.5, ct.viewport().height() - 8), 0.4)
    r.expect(ct.item(i_c, 5).text(), '113 %', 'compute circuit on a 5 A fuse')
    r.expect(ct.item(i_c, 3).text(), '5.66', 'compute circuit current')
    r.callout(rect_of(ct, i_c, 0, i_c, 6), '', dur=2.6, pad=3, dim=0.2)
    r.until_word(5, 'the finding says')
    f0 = next(f for f in pg.findings if 'fuse blows' in f.text.text())
    fr = r.wrect(f0)
    r.zoom(QRectF(fr.x() - 10, fr.y() - 60, min(fr.width(), 700) + 20, 160), 1.0)
    r.callout(f0, '', dur=max(2.0, r.speech_left() + 0.5), pad=3, dim=0.2)
    r.finish_speech(0.2)

    # ---- 6: the 24 V bus
    r.say(6)
    r.zoom(None, 0.8, wait=False)
    r.click(pg.bus.buttons['24'], None, 0.8)
    r.wait_for(lambda: pg.result.settings.bus_v == 24.0, timeout=10)
    stage.settle(1)
    r.expect(ct.item(i_c, 3).text(), '2.83', 'compute circuit current at 24 V')
    r.expect(ct.item(i_c, 5).text(), '57 %', 'compute circuit load at 24 V')
    r.zoom(QRectF(cr.x() - 10, cr.y() - 40, cr.width() + 20, cr.height() + 50), 1.0)
    r.figure('Compute circuit', float(ct.item(i_c, 3).text()), 'A', 2, dur=max(3.0, r.speech_left() + 0.6),
             src=rect_of(ct, i_c, 3, i_c, 3), start=5.66, note='24 V bus · 5 A fuse')
    r.finish_speech(0.2)

    # ---- 7: the cable drops
    r.say(7)
    r.expect(t.item(i_l1, 9).text(), '0.2 %', "the lidar cable's drop at 24 V")
    r.zoom(QRectF(tr.x(), tr.y() - 20, tr.width(), tr.height() + 20), 1.0)
    r.callout(rect_of(t, i_l1, 7, row(t, 'IMU1'), 10), '', dur=2.8, pad=3, dim=0.2)
    r.until_word(7, 'The worst')
    r.callout(rect_of(t, i_l1, 9, i_l1, 9), '', dur=max(1.8, r.speech_left() + 0.4), pad=3, dim=0.2)
    r.finish_speech(0.2)

    # ---- 8: heat; nominal figures; export
    r.say(8)
    r.expect(pg.tiles['heat'].value, '136', 'enclosure heat')
    r.zoom(tiles_zoom, 1.0)
    r.figure('Enclosure heat', float(pg.tiles['heat'].value), 'W', 0, dur=3.0, src=pg.tiles['heat'])
    r.until_word(8, 'export the budget')
    r.zoom(QRectF(r.wrect(pg.b_export).x() - 280, r.wrect(pg.b_export).y() - 120, 420, 180), 0.8)
    r.callout(pg.b_export, '', dur=max(1.6, r.speech_left()), pad=4, dim=0.2)
    r.finish_speech(0.3)
    r.zoom(None, 0.8)
    r.cursor_on = False
    r.hold(0.8)
