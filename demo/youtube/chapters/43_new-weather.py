"""43 Weather and night (episode v020, chapter 3): Analysis ▸ Weather and night on the sample's data rig: the
perception range page's Weather block and its comparison panel. The eight conditions and their figures; night on an
unlit road (the envelopes inside the dashed clear ones, how the light is modelled, the Ranges table against clear,
nothing resolved to the sides and behind), high beam typed in and Typical back, street lights; drops on the lenses in
heavy rain (the findings), each lidar and radar against clear in heavy rain and in 150 m fog (the Sensors tab), snow;
the fog thickened to 50 m; the sample dataset's preview rendered at night and in that fog; the Notes tab; the hooks.

Figures read from the app while recording (r.expect), the same as core.weather gives on the sample's data rig: the
Ranges table (straight ahead, m; sector checks) clear 70 / 66 / 66 / 47, 28/32; night unlit 29 / 19 / 19 / 47, 7/32;
night with high beam 70 / 66 / 66 / 47, 7/32; street lights 70 / 66 / 66 / 47, 28/32; heavy rain: drops on about 15 %
of F's and 20 % of R's image (heuristic findings), L1 90 → 76 m, RD1 250 → 195 m, RD2-RD5 100 → 89 m; fog 150 m: L1
90 → 40 m, false returns ≈ 11.3 % of beams; snow: L1 → 71 m, the radars unchanged; fog 50 m: car 40 m, pedestrian
32 m, 20/32, L1 23 m.
"""

META = {
    'id': 'new-weather',
    'part': 1,
    'title': 'Weather and night',
    'promise': 'How far each camera, lidar and radar still reaches in rain, fog, snow and at night.',
    'menu': 'Analysis ▸ Weather and night…',
}

LINES = [
    # 0
    ("The perception range measures how far the rig resolves a car, a pedestrian, a cyclist and a sign, on a clear "
     "day. Data is also collected at night, in rain, fog and snow, and each kind of sensor loses something different.",
     "The perception range measures how far the rig resolves a car, a pedestrian, a cyclist, and a sign, on a clear "
     "day. Data is also collected at night, in rain, fog, and snow, and each kind of sensor loses something "
     "different."),
    # 1
    ("Analysis ▸ Weather and night opens the perception range on its Weather block. Eight conditions: clear, the "
     "reference with no weather model at all; overcast; night on an unlit road and under street lights; light and "
     "heavy rain; fog; and snow.",
     "Analysis, weather and night, opens the perception range on its weather block. Eight conditions: clear, the "
     "reference, with no weather model at all; overcast; night on an unlit road, and under street lights; light and "
     "heavy rain; fog; and snow."),
    # 2
    ("Every condition but clear has figures you can change: day or night, low or high beam, the fog's visibility, the "
     "rain and snow rates, and the ambient light. They are saved with the project, and Typical puts them back.",
     "Every condition but clear has figures you can change: day or night, low or high beam, the fog's visibility, the "
     "rain and snow rates, and the ambient light. They are saved with the project, and typical puts them back."),
    # 3
    ("Night on an unlit road: a moonlit sky of 0.1 lux, and the headlamps on low beam. The envelopes shrink inside "
     "the dashed clear ones.",
     "Night, on an unlit road: a moonlit sky of zero point one lux, and the headlamps on low beam. The envelopes "
     "shrink inside the dashed clear ones."),
    # 4
    ("An object now counts only while enough of its height is lit to the camera's minimum luminance. The headlamps "
     "are a typical beam pattern shaped after UN R112; the minimum comes from a photon budget of the module's pixel "
     "size, the lens's f-number and the frame rate.",
     "An object now counts only while enough of its height is lit, to the camera's minimum luminance. The headlamps "
     "are a typical beam pattern, shaped after U N R one twelve. The minimum comes from a photon budget of the "
     "module's pixel size, the lens's f-number, and the frame rate."),
    # 5
    ("The panel compares every condition straight ahead. At night a pedestrian is resolved to 19 metres instead of "
     "66, a car to 29 instead of 70; the sign, retroreflective, still to 47. Seven of the 32 sector checks pass, "
     "against 28 by day.",
     "The panel compares every condition, straight ahead. At night, a pedestrian is resolved to nineteen metres "
     "instead of sixty-six, a car to twenty-nine instead of seventy. The sign, retroreflective, still to forty-seven. "
     "Seven of the thirty-two sector checks pass, against twenty-eight by day."),
    # 6
    ("To the sides and behind there is no light at all, and nothing is resolved there.",
     "To the sides and behind, there is no light at all, and nothing is resolved there."),
    # 7
    ("Switch to high beam: straight ahead the ranges are back to the daylight figures, a pedestrian to 66 metres, but "
     "the sides and the rear stay dark, and the checks stay at seven of 32. Typical puts the low beam back.",
     "Switch to high beam: straight ahead the ranges are back to the daylight figures, a pedestrian to sixty-six "
     "metres, but the sides and the rear stay dark, and the checks stay at seven of thirty-two. Typical puts the low "
     "beam back."),
    # 8
    ("Under street lights, about 10 lux, every range is back to the clear figures, 28 of 32: here the light is the "
     "limit, not the camera.",
     "Under street lights, about ten lux, every range is back to the clear figures, twenty-eight of thirty-two. Here "
     "the light is the limit, not the camera."),
    # 9
    ("Rain hardly shortens a camera's range; drops on the lens hide part of the image instead. In heavy rain, about "
     "15 per cent of F's image and 20 per cent of R's, from their dirt and spray exposure: a washer jet, an air knife "
     "or a hydrophobic coating would help.",
     "Rain hardly shortens a camera's range. Drops on the lens hide part of the image instead. In heavy rain, about "
     "fifteen per cent of F's image, and twenty per cent of R's, from their dirt and spray exposure: a washer jet, an "
     "air knife, or a hydrophobic coating would help."),
    # 10
    ("The lidar and the radars are another matter. Heavy rain, 25 millimetres an hour, takes the long-range radar "
     "from 250 to 195 metres, the corner radars from 100 to 89, and the roof lidar from 90 to 76.",
     "The lidar and the radars are another matter. Heavy rain, twenty-five millimetres an hour, takes the long range "
     "radar from two hundred and fifty to a hundred and ninety-five metres, the corner radars from a hundred to "
     "eighty-nine, and the roof lidar from ninety to seventy-six."),
    # 11
    ("Fog of 150 metres takes the lidar to 40 metres, with false returns from droplets on about 11 per cent of its "
     "beams. Snow takes it to 71, and costs the radars nothing.",
     "Fog of a hundred and fifty metres takes the lidar to forty metres, with false returns from droplets on about "
     "eleven per cent of its beams. Snow takes it to seventy-one, and costs the radars nothing."),
    # 12
    ("The fisheyes keep their range in that fog: their own pixel limit is shorter than the fog's contrast range. "
     "Thicken it to 50 metres, and contrast becomes the limit: a car is resolved to 40 metres, a pedestrian to 32, "
     "and the lidar reaches 23.",
     "The fisheyes keep their range in that fog: their own pixel limit is shorter than the fog's contrast range. "
     "Thicken it to fifty metres, and contrast becomes the limit: a car is resolved to forty metres, a pedestrian to "
     "thirty-two, and the lidar reaches twenty-three."),
    # 13
    ("The sample dataset renders the same conditions. At night, the headlamp pools lie ahead of F, with the "
     "camera's noise in the dark, and almost nothing shows behind or to the sides.",
     "The sample dataset renders the same conditions. At night, the headlamp pools lie ahead of F, with the camera's "
     "noise in the dark, and almost nothing shows behind or to the sides."),
    # 14
    ("In that 50-metre fog, the street fades to grey with distance; lidar points drop out, false near returns "
     "appear, and radar detections weaken.",
     "In that fifty metre fog, the street fades to grey with distance. Lidar points drop out, false near returns "
     "appear, and radar detections weaken."),
    # 15
    ("These are typical models, not measurements, and the Notes tab names them: Koschmieder's contrast law, ITU-R "
     "attenuation for rain, snow and fog. Oncoming glare, the headlamps' own veil in fog, wet-road reflections and "
     "spray are not modelled.",
     "These are typical models, not measurements, and the notes tab names them: Koschmieder's contrast law, I T U R "
     "attenuation for rain, snow, and fog. Oncoming glare, the headlamps' own veil in fog, wet road reflections, and "
     "spray, are not modelled."),
    # 16
    ("Clear gives exactly the results it gave before weather existed. The comparison exports as JSON, CSV and PNG; "
     "batch mode has --weather, and --condition sets the weather for the perception range and the sample dataset; "
     "the assistant runs it as a study.",
     "Clear gives exactly the results it gave before weather existed. The comparison exports as Jason, C S V, and "
     "P N G. Batch mode has weather, and condition sets the weather for the perception range and the sample dataset. "
     "The assistant runs it as a study."),
]


def scene(stage, r):
    import numpy as np
    from PyQt6.QtCore import QPointF, QRectF, Qt

    import helpers_tourc as H
    from clip_dataset import _scrub_to
    from clip_weather import click_tab, pick_combo, pick_row
    from vantage.core import perception as PR

    w = stage._rig_window()
    r.win = w
    app = stage.app
    stage.tilde(w.msg_label)

    def pump(n=4):
        for _ in range(n):
            app.processEvents()

    # ---- 0: the data rig
    vp = stage.vp
    vp.cam.az, vp.cam.el = -34.0, 18.0
    vp.update()
    stage.settle(2)
    r.cursor_on = False
    r.cur = np.array([1180.0, 640.0])
    r.lower_third()
    r.say(0)
    stage.orbit(r, vp, -26, max(4.0, r.line(0) - 0.8), d_el=4)
    r.finish_speech(0.2)

    # ---- 1: the page; the eight conditions
    r.say(1)
    r.cursor_on = True
    r.menu('Analysis', 'Weather and night')
    pg = w.page('perception')
    right = pg.wtabs.parentWidget().parentWidget()          # set-up: room for every sensor in the comparison
    right.setSizes([250, 300, 530])
    pump(2)
    r.crossfade(0.4)

    def ready(cid, test=None):
        res = pg.ctl.result
        ok = (res is not None and not pg.ctl.busy and not pg.ctl.stale and PR._condition(res).id == cid
              and pg.wres is not None and pg.wres.selected == cid and not pg._wbusy and not pg._wtimer.isActive())
        return ok and (test is None or test())
    r.wait_for(lambda: ready('clear'), timeout=180)
    pump(4)
    wxb = H.union(r.wrect(pg.cond), r.wrect(pg.b_typical), *[r.wrect(f) for f in pg.wx_fields.values()],
                  r.wrect(pg.light), r.wrect(pg.beam), r.wrect(pg.wx_note))
    r.zoom(QRectF(wxb.x() - 12, wxb.y() - 50, wxb.width() + 24, wxb.height() + 70), 1.0)
    r.until_word(1, 'Eight conditions')
    r.click(pg.cond, None, 0.6, real=False)
    pg.cond.showPopup()
    items = [pg.cond.itemText(i) for i in range(pg.cond.count())]
    print('conditions:', items, flush=True)
    r.expect(len(items), 8, 'eight conditions')
    view = pg.cond.view()
    for i in range(pg.cond.count()):
        rr = view.visualRect(view.model().index(i, 0))
        r.move_to(view.viewport(), QPointF(rr.center().x() * 0.5, rr.center().y()), 0.35)
        view.setCurrentIndex(view.model().index(i, 0))
        r.hold(max(0.3, min(1.1, (r.speech_left() - 0.6) / max(1, pg.cond.count() - i))))
    pg.cond.hidePopup()
    r.cursor_on = False
    r.finish_speech(0.2)

    # ---- 2: the figures
    r.say(2)
    flds = H.union(r.wrect(pg.light), r.wrect(pg.beam), *[r.wrect(f) for f in pg.wx_fields.values()])
    r.callout(flds, '', dur=max(2.4, r.at_word(2, 'They are saved') - r.now()), pad=4, dim=0.25)
    r.until_word(2, 'and typical puts')
    r.callout(pg.b_typical, '', dur=max(1.4, r.speech_left() + 0.2), pad=4, dim=0.25)
    r.finish_speech(0.2)

    # ---- 3: night on an unlit road
    r.say(3)
    r.cursor_on = True
    pick_combo(r, pg.cond, 'night')
    r.wait_for(lambda: ready('night'), timeout=180)
    pump(4)
    c = PR._condition(pg.ctl.result)
    r.expect((c.light, c.beam, c.ambient_lux), ('night', 'low', 0.1), 'night on an unlit road')
    r.cursor_on = False
    r.zoom(r.wrect(pg.map), 1.0, margin=8)
    r.hold(max(1.6, r.speech_left() + 0.3))
    r.finish_speech(0.2)

    # ---- 4: how the light is modelled
    r.say(4)
    r.hold(max(2.0, r.at_word(4, 'The headlamps') - r.now()))
    r.zoom(None, 0.8)
    r.zoom(QRectF(wxb.x() - 12, wxb.y() - 50, wxb.width() + 24, wxb.height() + 70), 0.8)
    r.callout(pg.wx_note, '', dur=max(2.4, r.speech_left() + 0.2), pad=4, dim=0.25)
    r.finish_speech(0.2)
    r.zoom(None, 0.7)

    # ---- 5: the comparison, straight ahead
    t = pg.wcond

    def row_of(cid):
        return next(i for i in range(t.rowCount()) if t.item(i, 0).data(Qt.ItemDataRole.UserRole) == cid)

    def cells(cid):
        k = row_of(cid)
        return [t.item(k, j).text() for j in range(1, t.columnCount())]
    head = [t.horizontalHeaderItem(j).text() for j in range(t.columnCount())]
    print('ranges head:', head, flush=True)
    for cid in ('clear', 'night', 'night_lit', 'heavy_rain', 'fog', 'snow'):
        print('  ', cid, cells(cid), flush=True)
    r.expect(head, ['Condition', 'Car', 'Pedestrian', 'Cyclist', 'Traffic sign', 'Sectors'], 'the columns')
    r.expect(cells('clear'), ['70', '66', '66', '47', '28/32'], 'clear straight ahead')
    r.expect(cells('night'), ['29', '19', '19', '47', '7/32'], 'night straight ahead')
    r.say(5)
    click_tab(r, pg.wtabs, 'Ranges')
    tw = r.wrect(pg.wtabs)
    r.zoom(QRectF(tw.x() - 6, tw.y() - 34, tw.width() + 12, t.height() + 70), 1.0)
    r.until_word(5, 'At night')
    r.callout(H.cell(r, t, row_of('night')), '', dur=max(2.4, r.at_word(5, 'Seven of') - r.now()), pad=2, dim=0.25)
    r.until_word(5, 'Seven of')
    r.callout(lambda: H.union(H.cell(r, t, row_of('clear'), 5)(), H.cell(r, t, row_of('night'), 5)()), '',
              dur=max(2.0, r.speech_left() + 0.2), pad=2, dim=0.25)
    r.finish_speech(0.2)
    r.zoom(None, 0.7)

    # ---- 6: nothing to the sides and behind
    r.say(6)
    r.zoom(r.wrect(pg.map), 0.8, margin=8)
    r.hold(max(1.6, r.speech_left() + 0.3))
    r.finish_speech(0.2)
    r.zoom(None, 0.7)

    # ---- 7: high beam; Typical
    r.say(7)
    r.zoom(QRectF(wxb.x() - 12, wxb.y() - 50, wxb.width() + 24, wxb.height() + 70), 0.7)
    r.cursor_on = True
    r.click(pg.beam.buttons['high'], None, 0.6)
    pump(2)
    r.wait_for(lambda: ready('night', lambda: PR._condition(pg.ctl.result).beam == 'high'
                             and pg.wres.condition('night').beam == 'high'), timeout=180)
    pump(4)
    print('  night high', cells('night'), flush=True)
    r.expect(cells('night'), ['70', '66', '66', '47', '7/32'], 'night with high beam')
    r.cursor_on = False
    r.zoom(QRectF(tw.x() - 6, tw.y() - 34, tw.width() + 12, t.height() + 70), 0.8)
    r.callout(H.cell(r, t, row_of('night')), '', dur=max(2.4, r.at_word(7, 'Typical puts') - r.now() - 0.3), pad=2,
              dim=0.25)
    r.until_word(7, 'Typical puts')
    r.zoom(QRectF(wxb.x() - 12, wxb.y() - 50, wxb.width() + 24, wxb.height() + 70), 0.7)
    r.cursor_on = True
    r.click(pg.b_typical, None, 0.6)
    pump(2)
    r.cursor_on = False

    # ---- 8: street lights (the low beam's figures come back meanwhile)
    r.say(8)
    r.zoom(QRectF(tw.x() - 6, tw.y() - 34, tw.width() + 12, t.height() + 70), 0.8, wait=False)
    r.wait_for(lambda: ready('night', lambda: PR._condition(pg.ctl.result).beam == 'low'
                             and pg.wres.condition('night').beam == 'low'), timeout=180)
    pump(4)
    r.expect(cells('night'), ['29', '19', '19', '47', '7/32'], 'night back on low beam')
    r.expect(cells('night_lit'), ['70', '66', '66', '47', '28/32'], 'street lights')
    r.callout(H.cell(r, t, row_of('night_lit')), '', dur=max(2.6, r.speech_left() + 0.2), pad=2, dim=0.25)
    r.finish_speech(0.2)

    # ---- 9: drops on the lenses (heavy rain's findings)
    fnd = pg.wfind
    ftexts = [fnd.item(i, 1).text() for i in range(fnd.rowCount())]
    print('findings:', ftexts, flush=True)
    k_f = next(i for i, x in enumerate(ftexts) if x.startswith("Heavy rain · 25 mm/h: about 15 % of F's image"))
    k_r = next(i for i, x in enumerate(ftexts) if x.startswith("Heavy rain · 25 mm/h: about 20 % of R's image"))
    r.expect(ftexts[k_f].endswith('a washer jet, an air knife or a hydrophobic coating.'), True, 'the suggestion')
    r.say(9)
    r.cursor_on = True
    click_tab(r, pg.wtabs, pg.wtabs.tabText(2))
    pump(2)
    fnd.scrollToItem(fnd.item(k_f, 1))
    pump(2)
    r.cursor_on = False
    r.until_word(9, 'In heavy rain')
    r.callout(lambda: H.union(H.cell(r, fnd, k_f)(), H.cell(r, fnd, k_r)()), '', dur=max(3.0, r.speech_left() + 0.2),
              pad=2, dim=0.25)
    r.finish_speech(0.2)

    # ---- 10: heavy rain: each lidar and radar
    r.say(10)
    r.cursor_on = True
    click_tab(r, pg.wtabs, 'Ranges')
    pick_row(r, t, 'heavy_rain')
    click_tab(r, pg.wtabs, 'Sensors')
    pump(2)
    s = pg.wsens

    def srow(tag):
        k = next(i for i in range(s.rowCount()) if s.item(i, 0).text() == tag)
        return k, [s.item(k, j).text() for j in range(6)]
    rows = {tag: srow(tag)[1] for tag in ('F', 'R', 'L1', 'RD1', 'RD2', 'RD5')}
    print('heavy rain sensors:', rows, flush=True)
    r.expect(rows['L1'][2:5], ['90', '76', '-15 %'], 'L1 in heavy rain')
    r.expect(rows['RD1'][2:5], ['250', '195', '-22 %'], 'RD1 in heavy rain')
    r.expect((rows['RD2'][2:4], rows['RD5'][2:4]), (['100', '89'], ['100', '89']), 'the corner radars in heavy rain')
    r.cursor_on = False
    r.zoom(QRectF(tw.x() - 6, tw.y() - 34, tw.width() + 12, s.rowHeight(0) * (s.rowCount() + 1) + 70), 0.9)
    r.until_word(10, 'takes the long')
    r.callout(H.cell(r, s, srow('RD1')[0]), '', dur=r.at_word(10, 'the corner radars') - r.now(), pad=2, dim=0.25)
    r.callout(lambda: H.union(H.cell(r, s, srow('RD2')[0])(), H.cell(r, s, srow('RD5')[0])()), '',
              dur=r.at_word(10, 'and the roof lidar') - r.now() + 0.2, pad=2, dim=0.25, wait=False)
    r.until_word(10, 'and the roof lidar')
    r.callout(H.cell(r, s, srow('L1')[0]), '', dur=max(1.8, r.speech_left() + 0.2), pad=2, dim=0.25)
    r.finish_speech(0.2)

    # ---- 11: fog 150 m and snow: the lidar
    r.say(11)
    r.cursor_on = True
    click_tab(r, pg.wtabs, 'Ranges')
    pick_row(r, t, 'fog')
    click_tab(r, pg.wtabs, 'Sensors')
    pump(2)
    l1 = srow('L1')[1]
    print('fog L1:', l1, flush=True)
    r.expect(l1[2:5], ['90', '40', '-55 %'], 'L1 in 150 m fog')
    r.expect('false returns ≈ 11.3 % (heuristic)' in l1[5], True, 'false returns in fog')
    r.cursor_on = False
    r.callout(H.cell(r, s, srow('L1')[0]), '', dur=max(2.4, r.at_word(11, 'Snow takes') - r.now()), pad=2, dim=0.25)
    r.until_word(11, 'Snow takes')
    r.cursor_on = True
    click_tab(r, pg.wtabs, 'Ranges')
    pick_row(r, t, 'snow')
    click_tab(r, pg.wtabs, 'Sensors')
    pump(2)
    l1 = srow('L1')[1]
    rd = srow('RD1')[1]
    print('snow L1:', l1, 'RD1:', rd, flush=True)
    r.expect((l1[3], rd[2:5]), ('71', ['250', '250', '+0 %']), 'snow: lidar 71 m, radar unchanged')
    r.cursor_on = False
    r.callout(lambda: H.union(H.cell(r, s, srow('L1')[0])(), H.cell(r, s, srow('RD5')[0])()), '',
              dur=max(1.8, r.speech_left() + 0.2), pad=2, dim=0.25)
    r.finish_speech(0.2)
    r.zoom(None, 0.7)

    # ---- 12: fog thickened to 50 m
    r.say(12)
    r.cursor_on = True
    pick_combo(r, pg.cond, 'fog')
    r.wait_for(lambda: ready('fog'), timeout=180)
    pump(2)
    r.until_word(12, 'Thicken it')
    r.zoom(QRectF(wxb.x() - 12, wxb.y() - 50, wxb.width() + 24, wxb.height() + 70), 0.7)
    r.enter(pg.wx_fields['visibility_m'], '50')
    pump(2)
    if pg._wst().selected().visibility_m != 50.0:                 # no commit on Enter offscreen
        pg.wx_fields['visibility_m'].committed.emit(50.0)
        pump(2)
    r.wait_for(lambda: ready('fog', lambda: PR._condition(pg.ctl.result).visibility_m == 50.0
                             and pg.wres.condition('fog').visibility_m == 50.0), timeout=180)
    pump(4)
    click_tab(r, pg.wtabs, 'Ranges')
    pump(2)
    fogc = cells('fog')
    print('fog 50:', fogc, flush=True)
    r.expect(fogc[:2] + fogc[4:], ['40', '32', '20/32'], 'fog of 50 m')
    pick_row(r, t, 'fog')
    r.cursor_on = False
    r.zoom(QRectF(tw.x() - 6, tw.y() - 34, tw.width() + 12, t.height() + 70), 0.8)
    r.callout(H.cell(r, t, row_of('fog')), '', dur=max(1.6, r.at_word(12, 'and the lidar') - r.now()), pad=2, dim=0.25)
    r.until_word(12, 'and the lidar')
    click_tab(r, pg.wtabs, 'Sensors')
    pump(2)
    l1 = srow('L1')[1]
    print('fog 50 L1:', l1, flush=True)
    r.expect(l1[3], '23', 'L1 in 50 m fog')
    r.callout(H.cell(r, s, srow('L1')[0]), '', dur=max(1.6, r.speech_left() + 0.2), pad=2, dim=0.25)
    r.finish_speech(0.2)
    r.zoom(None, 0.7)

    # ---- 13: the sample dataset at night
    r.say(13)
    r.cursor_on = True
    r.menu('Export', 'Sample dataset')
    dp = w.page('dataset')
    stage.tilde(dp.out)
    r.crossfade(0.4)

    def shown():
        return dp.preview_data is not None and not dp._pbusy and not dp.timer.isActive() and not dp._pstale
    r.wait_for(shown, timeout=120)
    pump(2)
    i_n = dp.weather.findData('night')
    r.choose(dp.weather, i_n)
    pump(2)
    r.expect(dp.settings.weather, 'night', 'the dataset at night')
    _scrub_to(r, dp, 20)
    r.wait_for(shown, timeout=120)
    pump(2)
    r.cursor_on = False
    r.zoom(r.wrect(dp.grid), 1.0, margin=10)
    r.hold(max(2.0, r.speech_left() + 0.3))
    r.finish_speech(0.2)

    # ---- 14: and in the 50 m fog
    r.say(14)
    r.zoom(None, 0.6)
    r.cursor_on = True
    r.choose(dp.weather, dp.weather.findData('fog'))
    pump(2)
    r.expect(dp.settings.weather, 'fog', 'the dataset in fog')
    r.wait_for(shown, timeout=120)
    pump(2)
    r.cursor_on = False
    r.zoom(r.wrect(dp.grid), 0.9, margin=10)
    r.hold(max(1.6, r.at_word(14, 'Lidar points') - r.now()))
    r.zoom(r.wrect(dp.top), 0.9, margin=10)
    r.hold(max(1.6, r.speech_left() + 0.3))
    r.finish_speech(0.2)
    r.zoom(None, 0.7)

    # ---- 15: the Notes tab
    r.say(15)
    r.cursor_on = True
    r.menu('Analysis', 'Weather and night')
    r.crossfade(0.4)
    pump(2)
    click_tab(r, pg.wtabs, 'Notes')
    pump(2)
    r.cursor_on = False
    r.zoom(QRectF(tw.x() - 6, tw.y() - 34, tw.width() + 12, tw.height() + 40), 1.0)
    r.hold(max(2.0, r.speech_left() - 1.0))
    r.zoom(None, 0.8)
    r.finish_speech(0.2)

    # ---- 16: the hooks
    r.say(16)
    r.cursor_on = True
    r.move_to(pg.b_wexport, None, 0.8)
    r.callout(pg.b_wexport, '', dur=2.2, pad=4, dim=0.25)
    r.hold(1.8)
    r.cursor_on = False
    r.zoom(r.wrect(pg.map), 1.0, margin=8)
    r.hold(max(1.0, r.speech_left() - 1.2))
    r.zoom(None, 0.8)
    r.finish_speech(1.0)
