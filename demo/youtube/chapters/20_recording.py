"""20 Recording budget: Analysis ▸ Recording budget on the sample's data rig (six cameras, the roof lidar, five radars).
Every sensor's data rate, the totals, the disk set, the sustained write, the links and the uplink; the cameras
recorded as H.265 instead of raw; the assumptions labelled.

Figures read from the app while recording (r.expect): raw: 1,111.2 MB/s (8,889 Mbit/s), 4,000 GB/h, one 8 TB disk
set holds 2.0 h of an 8 h shift; sustained write 1,444 of 1,500 MB/s; a camera's GMSL2 link 26 %, the lidar's link
13 %, the uplink 1 %; H.265: 223 Mbit/s, one disk set holds 79.7 h.
"""

META = {
    'id': 'recording',
    'part': 3,
    'title': 'Recording budget',
    'promise': 'How much the rig records, and whether the recorder keeps up.',
    'menu': 'Analysis ▸ Recording budget',
}

LINES = [
    ("A data rig records for hours on end. The recording budget works out how much data that is, and whether the "
     "links and the recorder keep up.",
     "A data rig records for hours on end. The recording budget works out how much data that is, and whether the "
     "links and the recorder keep up."),
    ("Every sensor gets its data rate: a camera's width, height and frame rate times its bits per pixel; the lidar's "
     "points per second times bytes per point; each radar's stated output.",
     "Every sensor gets its data rate: a camera's width, height, and frame rate, times its bits per pixel. The "
     "lidar's points per second, times bytes per point. Each radar's stated output."),
    ("Recorded raw, six cameras dominate: 1,111 megabytes a second in all, four terabytes for every hour of driving.",
     "Recorded raw, six cameras dominate: one thousand one hundred and eleven megabytes a second in all. Four "
     "terabytes, for every hour of driving."),
    ("So one 8 TB disk set lasts two hours of an eight-hour shift, and the finding says how many sets a shift "
     "needs: five.",
     "So one eight terabyte disk set lasts two hours, of an eight hour shift, and the finding says how many sets a "
     "shift needs: five."),
    ("The disks must also sustain the data rate with 30 per cent headroom: 1,444 of their 1,500 MB/s, close to the "
     "limit.",
     "The disks must also sustain the data rate, with thirty per cent headroom: one thousand four hundred and "
     "forty-four, of their fifteen hundred megabytes a second. Close to the limit."),
    ("Each link is checked too: a camera's GMSL2 link carries raw pixels whatever is recorded, 26 per cent of it; "
     "the lidar uses 13 per cent of its gigabit Ethernet, and the uplink to the recorder 1 per cent.",
     "Each link is checked too. A camera's G M S L two link carries raw pixels, whatever is recorded: twenty-six per "
     "cent of it. The lidar uses thirteen per cent of its gigabit Ethernet, and the uplink to the recorder, one per "
     "cent."),
    ("Record the cameras as H.265 instead, and the rate falls to 223 Mbit/s: one disk set now holds almost 80 hours.",
     "Record the cameras as H dot two six five instead, and the rate falls to two hundred and twenty-three megabits a "
     "second: one disk set now holds almost eighty hours."),
    ("JPEG and H.265 sizes are typical assumptions, and the page says so: replace them with figures measured on your "
     "encoder and your scenes.",
     "JPEG and H dot two six five sizes are typical assumptions, and the page says so: replace them with figures "
     "measured on your encoder, and your scenes."),
    ("The disks, the shift and the uplink are yours to set. The budget exports as JSON and CSV, and the batch mode "
     "and the assistant read it too.",
     "The disks, the shift, and the uplink are yours to set. The budget exports as Jason and C S V, and the batch "
     "mode and the assistant read it too."),
]


def scene(stage, r):
    import numpy as np
    from PyQt6.QtCore import QPointF, QRectF

    w = stage._rig_window()
    r.win = w
    r.cursor_on = False
    vp = stage.vp
    vp.cam.az, vp.cam.el = -32.0, 22.0
    w.viewarea.frame_all()
    stage.settle(2)
    r.cur = np.array([1150.0, 560.0])

    # ---- 0: the opener; the page from its menu
    r.lower_third()
    r.say(0)
    stage.orbit(r, vp, 30, 3.0, d_el=3)
    r.cursor_on = True
    r.menu('Analysis', 'Recording budget')
    pg = w.page('recording')
    stage.settle(2)
    r.crossfade(0.4)
    r.finish_speech(0.1)
    t = pg.table
    tot = pg.result.totals

    def rows_rect(first, last, c0=0, c1=None):
        c1 = t.columnCount() - 1 if c1 is None else c1
        return lambda: QRectF(r.local_rect(t.viewport(), QRectF(t.visualItemRect(t.item(first, c0))))()).united(
            r.local_rect(t.viewport(), QRectF(t.visualItemRect(t.item(last, c1))))())

    def find_row(tag):
        return next(i for i in range(t.rowCount()) if t.item(i, 0) and t.item(i, 0).text() == tag)

    # ---- 1: every sensor's rate
    r.say(1)
    tr = r.wrect(t)
    r.zoom(QRectF(tr.x(), tr.y() - 20, tr.width() * 0.7, tr.height() + 20), 1.0)
    i_f, i_l1, i_rd1 = find_row('F'), find_row('L1'), find_row('RD1')
    r.callout(rows_rect(i_f, find_row('QR'), 0, 5), '', dur=2.6, pad=3, dim=0.2)
    r.until_word(1, "The lidar's points")
    r.callout(rows_rect(i_l1, i_l1, 0, 5), '', dur=2.2, pad=3, dim=0.2)
    r.until_word(1, "Each radar's")
    r.callout(rows_rect(i_rd1, find_row('RD5'), 0, 5), '', dur=max(1.8, r.speech_left() + 0.3), pad=3, dim=0.2)
    r.finish_speech(0.2)

    # ---- 2: the totals
    r.say(2)
    r.expect(f'{tot["mb_s"]:,.1f}', '1,111.2', 'total MB/s, raw')
    r.expect(pg.tiles['hour'].value, '4,000', 'GB per hour, raw')
    tiles = r.wrect(pg.tiles['rate']).united(r.wrect(pg.tiles['uplink']))
    r.zoom(QRectF(tiles.x(), tiles.y() - 10, tiles.width(), r.wrect(pg.bar).bottom() - tiles.y() + 20), 1.0)
    r.callout(pg.bar, '', dur=2.4, pad=3, dim=0.2)
    r.until_word(2, 'one thousand one hundred')
    r.figure('Data rate', round(tot['mb_s']), 'MB/s', 0, dur=2.6, src=pg.tiles['rate'], note='six raw cameras')
    r.until_word(2, 'Four')
    r.callout(pg.tiles['hour'], '', dur=max(1.8, r.speech_left() + 0.4), pad=3, dim=0.2)
    r.finish_speech(0.2)

    # ---- 3: one disk set: 2 h; the finding
    r.say(3)
    r.expect(pg.tiles['disk'].value, '2.0', 'hours per disk set, raw')
    r.expect(tot['sets_per_shift'], 5, 'disk sets per shift')
    r.figure('One 8 TB disk set', float(pg.tiles['disk'].value), 'h', 1, dur=3.0, src=pg.tiles['disk'],
             note='of an 8 h shift')
    r.hold(3.0)
    f0 = next(f for f in pg.findings if 'disk set' in f.text.text())
    fr = r.wrect(f0)
    r.zoom(QRectF(fr.x() - 10, fr.y() - 40, min(fr.width(), 1100) + 20, 120), 1.0)
    r.callout(f0, '', dur=max(1.8, r.speech_left() + 0.4), pad=3, dim=0.2)
    r.finish_speech(0.2)

    # ---- 4: the sustained write
    r.say(4)
    r.expect(pg.tiles['write'].value, '1,444', 'write needed')
    r.zoom(QRectF(tiles.x(), tiles.y() - 10, tiles.width(), r.wrect(pg.bar).bottom() - tiles.y() + 20), 1.0)
    r.callout(pg.tiles['write'], '', dur=max(2.4, r.speech_left() + 0.4), pad=3, dim=0.2)
    r.finish_speech(0.2)

    # ---- 5: the links
    r.say(5)
    r.expect(t.item(i_f, 7).text(), '26 %', "a camera's GMSL2 load")
    r.expect(t.item(i_l1, 7).text(), '13 %', "the lidar's link load")
    r.expect(pg.tiles['uplink'].value, '1', 'uplink load')
    r.zoom(QRectF(tr.x(), tr.y() - 20, tr.width(), tr.height() + 20), 1.0)
    r.callout(rows_rect(i_f, find_row('QR'), 6, 7), '', dur=2.6, pad=3, dim=0.2)
    r.until_word(5, 'The lidar uses')
    r.callout(rows_rect(i_l1, i_l1, 6, 7), '', dur=2.0, pad=3, dim=0.2)
    r.until_word(5, 'and the uplink')
    i_tot = find_row('Total')
    r.callout(rows_rect(i_tot, i_tot, 6, 7), '', dur=max(1.6, r.speech_left() + 0.3), pad=3, dim=0.2)
    r.finish_speech(0.2)

    # ---- 6: H.265
    r.say(6)
    r.zoom(None, 0.8, wait=False)
    b = pg.cam_fmt.buttons['h265']
    r.click(b, None, 0.8)
    stage.settle(1)
    r.hold(0.3)
    tot2 = pg.result.totals
    r.expect(f'{tot2["mbps"]:,.0f}', '223', 'total Mbit/s, H.265')
    r.expect(pg.tiles['disk'].value, '79.7', 'hours per disk set, H.265')
    r.zoom(QRectF(tiles.x(), tiles.y() - 10, tiles.width(), r.wrect(pg.bar).bottom() - tiles.y() + 20), 1.0)
    r.figure('Data rate', tot2['mbps'], 'Mbit/s', 0, dur=2.6, src=pg.tiles['rate'], start=tot['mbps'],
             note='cameras as H.265')
    r.until_word(6, 'one disk set now')
    r.figure('One 8 TB disk set', float(pg.tiles['disk'].value), 'h', 1, dur=max(2.6, r.speech_left() + 0.6),
             src=pg.tiles['disk'], start=2.0)
    r.finish_speech(0.2)

    # ---- 7: the assumptions, labelled
    r.say(7)
    left = pg.cam_fmt
    lr = r.wrect(left)
    r.zoom(QRectF(lr.x() - 10, lr.y() - 40, 360, 260), 1.0)
    note = next(lb for lb in pg.findChildren(type(pg.status)) if lb.text().startswith('JPEG and H.265 sizes'))
    r.callout(note, '', dur=2.6, pad=3, dim=0.2)
    r.until_word(7, 'replace them')
    r.callout(lambda: r.wrect(pg.fields['jpeg_bpp'][0]).united(r.wrect(pg.fields['h265_bpp'][0])).adjusted(-120, 0, 0, 0),
              '', dur=max(1.8, r.speech_left() + 0.4), pad=3, dim=0.2)
    r.finish_speech(0.2)

    # ---- 8: the recorder's settings; export
    r.say(8)
    rec = r.wrect(pg.fields['storage_tb'][0]).united(r.wrect(pg.fields['link_warn'][0]))
    r.zoom(QRectF(rec.x() - 130, rec.y() - 30, 400, rec.height() + 60), 0.9)
    r.callout(lambda: r.wrect(pg.fields['storage_tb'][0]).united(r.wrect(pg.fields['uplink_gbps'][0])).adjusted(-125, 0, 0, 0),
              '', dur=2.4, pad=3, dim=0.2)
    r.until_word(8, 'The budget exports')
    r.zoom(QRectF(r.wrect(pg.b_export).x() - 280, r.wrect(pg.b_export).y() - 120, 420, 180), 0.8)
    r.callout(pg.b_export, '', dur=max(1.6, r.speech_left()), pad=4, dim=0.2)
    r.finish_speech(0.3)
    r.zoom(None, 0.8)
    r.cursor_on = False
    r.hold(0.8)
