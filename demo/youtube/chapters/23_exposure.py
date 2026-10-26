"""23 Sun glare and dirt: Analysis ▸ Sun glare and dirt on the sample's data rig. The campaign (Munich, a year, 8 to 18 h,
every 15 min of every 7th day, every heading); the table and the findings; a quarter camera's month × hour heatmap,
sky chart and sun paths; an east-west route typed as headings; the same route in Chennai; the dirt map and cleaning.

Figures read from the app while recording (r.expect): Munich, uniform headings: F and R 7.2 % of the driving time
with the sun in the image (264 h), Nov–Feb; F pitched 10° further down 3.0 %; route 90:1, 270:1: F 9.7 %; Chennai
(the preset) on the same route: F 2.9 %; dirt: R 0.49, the highest of the cameras.
"""

META = {
    'id': 'exposure',
    'part': 3,
    'title': 'Sun glare and dirt',
    'promise': 'When the sun shines into each camera, and which lenses get dirty.',
    'menu': 'Analysis ▸ Sun glare and dirt',
}

LINES = [
    ("A campaign drives for months. When does the sun shine straight into a camera, and which lenses get dirty? Sun "
     "glare and dirt answers both.",
     "A campaign drives for months. When does the sun shine straight into a camera, and which lenses get dirty? Sun "
     "glare and dirt answers both."),
    ("The sun's position comes from NOAA's solar position algorithm, for a place, a date range and a daily driving "
     "window: here Munich, a year, 8 a.m. to 6 p.m., every 15 minutes of every seventh day, on every heading.",
     "The sun's position comes from Noah's solar position algorithm, for a place, a date range, and a daily driving "
     "window: here Munich, a year, eight A M to six P M, every fifteen minutes of every seventh day, on every "
     "heading."),
    ("Each sample goes through every camera's real lens model and the body. The front and rear cameras have the sun "
     "in the image 7.2 per cent of the driving time: 264 hours, mostly November to February.",
     "Each sample goes through every camera's real lens model, and the body. The front and rear cameras have the sun "
     "in the image seven point two per cent of the driving time: two hundred and sixty-four hours, mostly November to "
     "February."),
    ("The findings say what would help: pitched ten degrees further down, the front camera's share falls to 3.0 per "
     "cent; a lens hood helps too.",
     "The findings say what would help: pitched ten degrees further down, the front camera's share falls to three per "
     "cent. A lens hood helps too."),
    ("Select a camera for its month-by-hour heatmap, its sky with the sun's density and its field of view, and the "
     "sun's path through its image at the equinox and both solstices.",
     "Select a camera for its month by hour heatmap, its sky, with the sun's density and its field of view, and the "
     "sun's path through its image, at the equinox and both solstices."),
    ("A route changes the answer. Type the headings of an east-west motorway: the study runs again at once, and the "
     "front camera's share rises to 9.7 per cent.",
     "A route changes the answer. Type the headings of an east west motorway: the study runs again at once, and the "
     "front camera's share rises to nine point seven per cent."),
    ("The same route in Chennai, thirteen degrees north: the sun stands higher, and the front camera's share falls "
     "to 2.9 per cent.",
     "The same route in Chennai, thirteen degrees north: the sun stands higher, and the front camera's share falls to "
     "two point nine per cent."),
    ("Dirt and spray are a heuristic with its weights stated: wheel spray, the rear wake, frontal impact, and each "
     "lens's attitude and height. The rear camera's lens is the dirtiest of the cameras, at 0.49.",
     "Dirt and spray are a heuristic, with its weights stated: wheel spray, the rear wake, frontal impact, and each "
     "lens's attitude and height. The rear camera's lens is the dirtiest of the cameras, at zero point four nine."),
    ("Every camera, lidar window and radar fascia gets a class and cleaning advice. Clear sky and a flat horizon are "
     "assumed.",
     "Every camera, lidar window, and radar fascia gets a class, and cleaning advice. Clear sky and a flat horizon are "
     "assumed."),
]


def scene(stage, r):
    import numpy as np
    from PyQt6.QtCore import QPointF, QRectF, Qt
    from PyQt6.QtTest import QTest

    w = stage._rig_window()
    r.win = w
    r.cursor_on = False
    vp = stage.vp
    vp.cam.az, vp.cam.el = -60.0, 18.0
    w.viewarea.frame_all()
    stage.settle(2)
    r.cur = np.array([1150.0, 560.0])

    # ---- 0: the opener; the page from its menu
    r.lower_third()
    r.say(0)
    stage.orbit(r, vp, 36, 3.2, d_el=4)
    r.cursor_on = True
    r.menu('Analysis', 'Sun glare and dirt')
    pg = w.page('exposure')
    ctl = pg.ctl
    r.crossfade(0.4)
    r.wait_for(lambda: ctl.result is not None and not ctl.busy, timeout=120)
    stage.settle(1)
    r.finish_speech(0.1)
    t = pg.table

    def row(tag):
        return next(i for i in range(t.rowCount()) if t.item(i, 0).text().split(' ')[0] == tag)

    def cells(r0, c0, r1, c1):
        return lambda: QRectF(r.local_rect(t.viewport(), QRectF(t.visualItemRect(t.item(r0, c0))))()).united(
            r.local_rect(t.viewport(), QRectF(t.visualItemRect(t.item(r1, c1))))())

    # ---- 1: the campaign
    r.say(1)
    f = pg.fields
    top = r.wrect(f['lat'])
    bot = r.wrect(f['days'])
    r.zoom(QRectF(top.x() - 120, top.y() - 70, 440, bot.bottom() - top.y() + 100), 1.0)
    r.until_word(1, 'here Munich')
    r.callout(lambda: r.wrect(f['lat']).united(r.wrect(f['lon'])).adjusted(-110, 0, 0, 0), '', dur=1.6, pad=3, dim=0.2)
    r.until_word(1, 'eight A M')
    r.callout(lambda: r.wrect(f['hour_from']).united(r.wrect(f['hour_to'])).adjusted(-110, 0, 0, 0), '', dur=1.8,
              pad=3, dim=0.2)
    r.until_word(1, 'every fifteen')
    r.callout(lambda: r.wrect(f['minutes']).united(r.wrect(f['days'])).adjusted(-110, 0, 0, 0), '',
              dur=max(1.6, r.speech_left() + 0.3), pad=3, dim=0.2)
    r.finish_speech(0.2)

    # ---- 2: the table: F and R 7.2 %
    r.say(2)
    i_f, i_r = row('F'), row('R')
    r.expect((t.item(i_f, 1).text(), t.item(i_r, 1).text()), ('7.2', '7.2'), 'F and R sun in the image, Munich')
    r.expect(t.item(i_f, 4).text(), 'Nov–Feb', 'worst months')
    tr = r.wrect(t)
    r.zoom(QRectF(tr.x(), tr.y() - 26, tr.width() * 0.62, 26 + 23 * 8), 1.0)
    r.until_word(2, 'seven point two')
    r.figure('Sun in the image · F and R', float(t.item(i_f, 1).text()), '%', 1, dur=3.0, src=cells(i_f, 1, i_r, 1),
             note='of the driving time · Munich')
    r.until_word(2, 'mostly November')
    r.callout(cells(i_f, 4, i_r, 5), '', dur=max(1.8, r.speech_left() + 0.4), pad=3, dim=0.2)
    r.finish_speech(0.2)

    # ---- 3: the findings
    r.say(3)
    f0 = next(x for x in pg.findings if x.text.text().startswith('F has the sun'))
    print('F finding:', f0.text.text(), flush=True)
    r.expect('264 h' in f0.text.text() and '10° down 3.0 %' in f0.text.text(), True, "F's finding")
    fr = r.wrect(f0)
    r.zoom(QRectF(fr.x() - 10, fr.y() - 50, min(fr.width(), 900) + 20, fr.height() + 100), 1.0)
    r.callout(f0, '', dur=max(2.6, r.speech_left() + 0.4), pad=3, dim=0.2)
    r.finish_speech(0.2)

    # ---- 4: one camera: heatmap, sky, sun paths
    r.say(4)
    r.zoom(None, 0.8, wait=False)
    i_ql = row('QL')
    r.click(t.viewport(), QPointF(t.visualItemRect(t.item(i_ql, 0)).center()), 0.7)
    stage.settle(1)
    r.move(r.wpos(pg.sky, QPointF(pg.sky.width() - 14, pg.sky.height() * 0.3)), 0.5)
    h, s, k = r.wrect(pg.heat), r.wrect(pg.sky), r.wrect(pg.tracks)
    r.zoom(QRectF(h.x(), h.y() - 4, s.right() - h.x(), s.bottom() - h.y() + 8), 1.0)
    r.until_word(4, "and the sun's path")
    r.zoom(QRectF(k.x() - 6, k.y() - 40, k.width() + 12, k.height() + 46), 1.0)
    r.hold(max(1.2, r.speech_left() + 0.3))
    r.finish_speech(0.2)

    # ---- 5: an east-west route typed as headings
    r.say(5)
    r.zoom(None, 0.8, wait=False)
    before = ctl.result
    ed = pg.head_text
    r.click(ed, QPointF(ed.width() * 0.6, ed.height() / 2), 0.7)
    QTest.keyClick(ed, Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
    r.tick(2)
    hr = r.wrect(ed)
    r.zoom(QRectF(hr.x() - 140, hr.y() - 160, 460, 260), 0.7)
    for ch in '90:1, 270:1':
        QTest.keyClicks(ed, ch)
        r.hold(0.09)
    QTest.keyClick(ed, Qt.Key.Key_Return)
    r.wait_for(lambda: ctl.result is not before and not ctl.busy and not ctl.stale, timeout=60)
    stage.settle(1)
    r.hold(0.4)
    r.expect(t.item(i_f, 1).text(), '9.7', 'F on the east-west route, Munich')
    r.zoom(QRectF(tr.x(), tr.y() - 26, tr.width() * 0.62, 26 + 23 * 8), 1.0)
    r.figure('Sun in the image · F', float(t.item(i_f, 1).text()), '%', 1, dur=max(2.8, r.speech_left() + 0.6),
             src=cells(i_f, 1, i_f, 1), start=7.2, note='east-west route · Munich')
    r.finish_speech(0.2)

    # ---- 6: the same route in Chennai
    r.say(6)
    r.zoom(None, 0.8, wait=False)
    before = ctl.result
    pr = r.wrect(pg.preset)
    r.choose(pg.preset, pg.preset.findText('Chennai'))
    r.wait_for(lambda: ctl.result is not before and not ctl.busy and not ctl.stale, timeout=60)
    stage.settle(1)
    r.expect(f'{pg.settings.lat:.2f}', '13.08', 'Chennai latitude')
    r.expect(t.item(i_f, 1).text(), '2.9', 'F on the east-west route, Chennai')
    r.zoom(QRectF(pr.x() - 120, pr.y() - 20, 460, 120), 0.8)
    r.callout(lambda: r.wrect(f['lat']).united(r.wrect(f['lon'])).adjusted(-110, 0, 0, 0), '', dur=2.0, pad=3, dim=0.2)
    r.hold(1.8)
    r.zoom(QRectF(tr.x(), tr.y() - 26, tr.width() * 0.62, 26 + 23 * 8), 0.8)
    r.figure('Sun in the image · F', float(t.item(i_f, 1).text()), '%', 1, dur=max(2.6, r.speech_left() + 0.6),
             src=cells(i_f, 1, i_f, 1), start=9.7, note='east-west route · Chennai')
    r.finish_speech(0.2)

    # ---- 7: dirt
    r.say(7)
    cams = [row(x) for x in ('F', 'R', 'ML', 'MR', 'QL', 'QR')]
    dirt = {i: float(t.item(i, 7).text()) for i in cams}
    r.expect(t.item(i_r, 7).text(), '0.49', "R's dirt score")
    r.expect(max(dirt, key=dirt.get), i_r, 'the dirtiest camera')
    r.zoom(r.wrect(pg.dirt), 1.0, margin=10)
    r.until_word(7, "The rear camera's lens")
    r.zoom(QRectF(tr.x() + tr.width() * 0.4, tr.y() - 26, tr.width() * 0.6, 26 + 23 * 13), 1.0)
    r.figure('Dirt · rear camera', dirt[i_r], '', 2, dur=max(2.6, r.speech_left() + 0.6), src=cells(i_r, 7, i_r, 8),
             note='0 clean · 1 dirtiest')
    r.finish_speech(0.2)

    # ---- 8: classes and cleaning advice; the assumptions
    r.say(8)
    cl = cells(0, 8, t.rowCount() - 1, 10)
    r.callout(lambda: cl().intersected(r.wrect(t.viewport())), '', dur=max(2.6, r.speech_left() + 0.4), pad=3, dim=0.2)
    r.finish_speech(0.3)
    r.zoom(None, 0.9)
    r.cursor_on = False
    r.hold(0.8)
