"""44 Materials, glass and wipers (episode v020, chapter 4): Analysis ▸ Materials, glass and wipers on the sample's data
rig, with a camera FW added behind the windscreen for this example (the sample has none; it is the camera the
materials tests place: STURDeCAM21 at x 1.80, z 1.36, 5° down). What each radar looks through (RD1 in front of the
bumper; RD2-RD5 behind the painted PP fascia), the loss map, the paint switched to metallic and to a repaired metallic
bumper, the fascia thickness that reflects least typed under Parts, the body set to painted steel, the library; the
camera behind glass (light, centre shift, dashboard glare, the glare shield); the windscreen from inside with the
typical wipers, opposed arms and a shortened blade; findings, the 3D wiper overlay, limits; batch, export, assistant.

Figures read from the app while recording (r.expect): RD1 '0.00 dB' in front of the body; RD2 11.7 mm behind the
fascia, 3.00 mm (2.43 half waves), 29.2°, 2.67 dB, 100 -> 86 m, reflection -6.8 dB, moderate; least reflection at
3.61 mm; metallic 3.89 dB / 80 m; repaired 8.31 dB / 62 m; 3.61 mm typed: 1.88 dB, -34.8 dB, low; steel: blocked,
critical; FW 66.9°, 73 % (0.45 EV), 3.84 mm, 0.25 px at 3 m, 33 % dashboard (about 14 % contrast); windscreen 1.65 m
x 0.64 m, tandem wipers clear 63 %, FW 100 %; opposed 55 %, FW 1 %; driver blade 0.2 m: FW 7 %; the batch run's lines
and exit code; the export's four files; the assistant tool get_materials exists.
"""

META = {
    'id': 'new-materials',
    'part': 2,
    'title': 'Materials, glass and wipers',
    'promise': 'What every radar and camera looks through, and what it costs.',
    'menu': 'Analysis ▸ Materials, glass and wipers',
}

LINES = [
    # 0
    ("A radar behind a bumper sees through plastic and paint. A camera behind the windscreen sees through glass. Both "
     "cost range, light or accuracy, and on most rigs nobody checks.",
     "A radar behind a bumper sees through plastic and paint. A camera behind the windscreen sees through glass. Both "
     "cost range, light, or accuracy, and on most rigs, nobody checks."),
    # 1
    ("Version 0.2.0 gives every part of the vehicle a material, and works out what it does to each sensor.",
     "Version zero point two gives every part of the vehicle a material, and works out what it does to each sensor."),
    # 2
    ("On the left: the vehicle's paint, the radar's frequency and thresholds, how a one-part body is split, the glass "
     "and the wipers. The thresholds are typical design targets: set your own.",
     "On the left: the vehicle's paint, the radar's frequency and thresholds, how a one-part body is split, the glass, "
     "and the wipers. The thresholds are typical design targets: set your own."),
    # 3
    ("For each radar, rays over its whole field find the surface it looks out through. The loss is worked out layer by "
     "layer, plastic, primer, basecoat and clear coat, at each ray's angle.",
     "For each radar, rays over its whole field find the surface it looks out through. The loss is worked out layer by "
     "layer, plastic, primer, basecoat, and clear coat, at each ray's angle."),
    # 4
    ("RD1, the front radar, sits in front of the bumper: it looks through nothing and loses nothing. The four corner "
     "radars sit 11.7 millimetres behind the bumper fascia.",
     "R D one, the front radar, sits in front of the bumper: it looks through nothing, and loses nothing. The four "
     "corner radars sit eleven point seven millimetres behind the bumper fascia."),
    # 5
    ("That fascia is a typical 3 millimetre polypropylene bumper with a solid paint, met at 29 degrees. There and back, "
     "it costs 2.67 decibels.",
     "That fascia is a typical three millimetre polypropylene bumper, with a solid paint, met at twenty-nine degrees. "
     "There and back, it costs two point six seven decibels."),
    # 6
    ("Range goes with the fourth root of what gets through: a car the radar would see at 100 metres is now seen at 86.",
     "Range goes with the fourth root of what gets through: a car the radar would see at a hundred metres is now seen "
     "at eighty-six."),
    # 7
    ("Thickness is also counted in half wavelengths inside the plastic; whole numbers reflect least. At 2.43, this "
     "fascia sends 6.8 decibels back into the radar, above the minus 10 target: a moderate multipath risk.",
     "Thickness is also counted in half wavelengths inside the plastic; whole numbers reflect least. At two point four "
     "three, this fascia sends six point eight decibels back into the radar, above the minus ten target: a moderate "
     "multipath risk."),
    # 8
    ("Pick a radar to see its loss over the field: azimuth across, elevation up, darker where it loses more.",
     "Pick a radar to see its loss over the field: azimuth across, elevation up, darker where it loses more."),
    # 9
    ("Below the map: the cavity between the radar and the fascia, and the thickness nearby that would reflect least, "
     "3.61 millimetres.",
     "Below the map: the cavity between the radar and the fascia, and the thickness nearby that would reflect least, "
     "three point six one millimetres."),
    # 10
    ("Paint matters most. A metallic basecoat carries metal flakes: 3.89 decibels, and the car is seen at 80 metres.",
     "Paint matters most. A metallic basecoat carries metal flakes: three point eight nine decibels, and the car is "
     "seen at eighty metres."),
    # 11
    ("A repaired bumper, metallic paint sprayed over the factory coat, loses 8.31 decibels: 62 metres, and a serious "
     "finding with the fix.",
     "A repaired bumper, metallic paint sprayed over the factory coat, loses eight point three one decibels: sixty-two "
     "metres, and a serious finding, with the fix."),
    # 12
    ("Back on solid paint. Under Parts, every part of the model has a material, from its name and role. The sample's "
     "body is one part, so it is split by position.",
     "Back on solid paint. Under parts, every part of the model has a material, from its name and role. The sample's "
     "body is one part, so it is split by position."),
    # 13
    ("Its ends below the bumper line are the fascia, and a band between the side windows and the bonnet is the "
     "windscreen. Any part can be set by hand, with its thickness.",
     "Its ends below the bumper line are the fascia, and a band between the side windows and the bonnet is the "
     "windscreen. Any part can be set by hand, with its thickness."),
    # 14
    ("Type the 3.61 millimetres the radar table suggested: the reflection falls to minus 34.8 decibels, the loss to "
     "1.88, and the multipath risk to low.",
     "Type the three point six one millimetres the radar table suggested: the reflection falls to minus thirty-four "
     "point eight decibels, the loss to one point eight eight, and the multipath risk to low."),
    # 15
    ("Make the body painted steel instead, and the corner radars are blocked: 77 gigahertz does not pass metal. That "
     "is critical, and the finding says what to do: put the radar behind plastic, or in front of the panel.",
     "Make the body painted steel instead, and the corner radars are blocked: seventy-seven gigahertz does not pass "
     "metal. That is critical, and the finding says what to do: put the radar behind plastic, or in front of the "
     "panel."),
    # 16
    ("Undo puts the fascia back. The library lists each material's layers and its typical 77 gigahertz figures, each "
     "with its published source. They are not your vehicle's: measure your fascia and enter it.",
     "Undo puts the fascia back. The library lists each material's layers, and its typical seventy-seven gigahertz "
     "figures, each with its published source. They are not your vehicle's: measure your fascia, and enter it."),
    # 17
    ("Cameras behind glass. The sample has none, so for this example a camera, FW, sits behind the windscreen.",
     "Cameras behind glass. The sample has none, so for this example, a camera, F W, sits behind the windscreen."),
    # 18
    ("At 67 degrees to the glass, the laminated windscreen passes 73 per cent of the light: 0.45 stops darker.",
     "At sixty-seven degrees to the glass, the laminated windscreen passes seventy-three per cent of the light: zero "
     "point four five stops darker."),
    # 19
    ("Glass also slides every ray sideways: FW's centre appears 3.84 millimetres off. Calibrated through the glass, "
     "0.25 pixels are left at 3 metres.",
     "Glass also slides every ray sideways: F W's centre appears three point eight four millimetres off. Calibrated "
     "through the glass, zero point two five pixels are left at three metres."),
    # 20
    ("And the glass reflects 33 per cent of the dashboard into the image: veiling glare, about 14 per cent of the "
     "contrast on a sunlit road.",
     "And the glass reflects thirty-three per cent of the dashboard into the image: veiling glare, about fourteen per "
     "cent of the contrast on a sunlit road."),
    # 21
    ("Tick Glare shield when a black shield sits under the camera, and the warning goes.",
     "Tick glare shield when a black shield sits under the camera, and the warning goes."),
    # 22
    ("Below, the windscreen seen from inside, 1.65 by 0.64 metres. Typical tandem wipers sized from it clear 63 per "
     "cent of the glass, and all of FW's view.",
     "Below, the windscreen seen from inside, one point six five by zero point six four metres. Typical tandem wipers "
     "sized from it clear sixty-three per cent of the glass, and all of F W's view."),
    # 23
    ("Opposed arms clear 55 per cent, and FW's footprint turns red: 1 per cent of its view is wiped.",
     "Opposed arms clear fifty-five per cent, and F W's footprint turns red: one per cent of its view is wiped."),
    # 24
    ("Or enter your own arms. Back on tandem, shorten the driver's blade to 0.2 metres: only 7 per cent of FW's view is "
     "wiped, and the dirt that stays is scored by the exposure model.",
     "Or enter your own arms. Back on tandem, shorten the driver's blade to zero point two metres: only seven per cent "
     "of F W's view is wiped, and the dirt that stays is scored by the exposure model."),
    # 25
    ("Undo restores the wipers. Findings gathers it all; the Issues list gets a radar looking into metal, a loss of 3 "
     "decibels or more, a high multipath risk, glare without a shield, and a camera behind unwiped glass.",
     "Undo restores the wipers. Findings gathers it all; the issues list gets a radar looking into metal, a loss of "
     "three decibels or more, a high multipath risk, glare without a shield, and a camera behind unwiped glass."),
    # 26
    ("In the main view, View, Overlays, Wiper sweep draws the wipers' sweep on the windscreen.",
     "In the main view, View, Overlays, Wiper sweep, draws the wipers' sweep on the windscreen."),
    # 27
    ("The limits: surfaces are flat layers at each ray, with no curvature or edge diffraction; glass is plane-parallel; "
     "the automatic regions are geometric guesses.",
     "The limits: surfaces are flat layers at each ray, with no curvature, or edge diffraction; glass is "
     "plane-parallel; and the automatic regions are geometric guesses."),
    # 28
    ("In batch mode, --materials writes the study and prints a line per radar, then its findings. It exits with code "
     "2 when a radar looks into metal, so a CI job can stop on it.",
     "In batch mode, materials writes the study, and prints a line per radar, then its findings. It exits with code "
     "two when a radar looks into metal, so a C I job can stop on it."),
    # 29
    ("The export writes JSON, two CSV tables and a picture, and the assistant reads the same study with its get "
     "materials tool.",
     "The export writes Jason, two C S V tables, and a picture, and the assistant reads the same study with its get "
     "materials tool."),
]

ROOT_RUN = 'run_vantage.py'


def scene(stage, r):
    import os
    import site
    import sys
    import numpy as np
    from PyQt6.QtCore import QPointF, QRectF
    import helpers_tourc as H
    import helpers_tour_a as HA
    from common import W as W_
    from vantage.core import samples
    from vantage.core.project import Camera

    def windscreen_camera(p):
        p.variant.cameras.append(Camera(tag='FW', name='Windscreen', x=1.80, y=0.0, z=1.36, yaw=0.0, pitch=5.0,
                                        slot=6, port=7, module_id='sturdecam21', lens_id='sturdecam21_std'))

    # ---- the batch runs (real, not filmed): the sample's data rig as saved, in the throw-away home
    import studio
    home = stage.home
    bin_ = os.path.join(home, 'bin')
    work = os.path.join(home, 'projects')
    os.makedirs(bin_, exist_ok=True)
    os.makedirs(work, exist_ok=True)
    with open(os.path.join(bin_, 'vantage'), 'w') as fh:
        fh.write(f'#!/bin/sh\nexec {sys.executable} {os.path.join(studio.ROOT, ROOT_RUN)} "$@"\n')
    os.chmod(os.path.join(bin_, 'vantage'), 0o755)
    env = {k: v for k, v in os.environ.items() if k not in ('QT_SCALE_FACTOR',)}
    env.update(PATH=bin_ + os.pathsep + env.get('PATH', ''), HOME=home, PYTHONUSERBASE=site.USER_BASE,
               QT_QPA_PLATFORM='offscreen')
    q = samples.tigor_sample_project()
    q.photos.source = ''
    q.active_uid = next(v.uid for v in q.variants if v.sensors)
    q.save(os.path.join(work, 'data-rig.vantage'), b'')

    def run(cmd):
        code, lines, secs = H.run_timed(cmd, work, env, home)
        print(f'$ {cmd}  -> {code} in {secs:.1f} s', flush=True)
        for _, ln in lines:
            print('   ', ln, flush=True)
        return code, lines, secs
    c_mt, l_mt, _ = run('vantage data-rig.vantage --materials materials.json')
    r.expect(c_mt, 0, 'exit code of --materials on the data rig')
    txt = [ln for _, ln in l_mt]
    r.expect(any(ln.strip().startswith('RD2  behind') and '2.7 dB two-way, range 86 m' in ln for ln in txt), True,
             'the batch line for RD2')
    c_ex, l_ex, _ = run('vantage data-rig.vantage --export out/ --formats materials')
    c_ls, l_ls, _ = run('ls out/materials')
    r.expect(sorted(ln for _, ln in l_ls), sorted(['materials.json', 'materials_radars.csv', 'materials_glass.csv',
                                                   'materials.png']), 'the export files')
    from vantage.assistant import tools as AT
    r.expect('get_materials' in AT.BY_NAME, True, 'the assistant tool')
    noise = (lambda ln: not ln.startswith(('This plugin does not', '[ WARN')))

    # ---- the window: the data rig with FW behind the windscreen
    w = stage._rig_window(windscreen_camera)
    r.win = w
    vp = stage.vp
    vp.cam.az, vp.cam.el = -38.0, 20.0
    w.viewarea.frame_all()
    stage.settle(2)
    d0 = vp.cam.dist
    r.cur = np.array([1150.0, 560.0])
    r.cursor_on = False

    def rect_u(*targets):
        def f():
            rs = [r.target_rect(t) for t in targets]
            return None if any(x is None for x in rs) else H.union(*rs)
        return f

    def span(t, r0, c0, r1, c1):
        return rect_u(H.cell(r, t, r0, c0), H.cell(r, t, r1, c1))

    def grown(target, dx):
        def f():
            x = r.target_rect(target)
            return None if x is None else x.adjusted(-dx, -4, 4, 4)
        return f

    def row_of(t, tag):
        return next(i for i in range(t.rowCount()) if t.item(i, 0) is not None and t.item(i, 0).text() == tag)

    def click_tab(tabs, title):
        bar = tabs.tabBar()
        i = next(k for k in range(tabs.count()) if tabs.tabText(k).startswith(title))
        r.click(bar, QPointF(bar.tabRect(i).center()), 0.5, real=False)
        tabs.setCurrentIndex(i)
        r.hold(0.2)

    # ---- 0: the problem, over the 3D view
    r.lower_third()
    r.say(0)
    stage.fly(r, vp, az=-58.0, el=10.0, dist=vp.cam.dist * 0.45, target=(3.0, 0.0, 0.55), dur=4.0)
    stage.orbit(r, vp, 24, max(1.5, r.speech_left() - 0.2), d_el=4)
    r.finish_speech(0.1)

    # ---- 1: the page from its menu
    r.say(1)
    r.cursor_on = True
    r.hold(0.8)
    r.menu('Analysis', 'Materials, glass and wipers')
    pg = w.page('materials')
    ctl = pg.ctl

    def ready():
        return ctl.result is not None and not ctl.busy and not ctl.stale and not ctl.timer.isActive() and \
            pg._shown is ctl.result
    r.crossfade(0.4)
    r.wait_for(ready, timeout=60)
    r.finish_speech(0.2)
    rd = pg.radars

    # ---- 2: the settings on the left
    r.say(2)
    left = pg.paint.parentWidget()
    lr = r.wrect(left)
    r.zoom(QRectF(lr.x(), lr.y() + 90, 400, 680), 1.0)
    r.callout(pg.paint, '', dur=1.6, pad=4, dim=0.25)
    r.until_word(2, "the radar's frequency")
    f_rad = rect_u(pg.fields['radar_ghz'][0], pg.fields['min_gap_mm'][0])
    r.callout(grown(f_rad, 122), '', dur=2.0, pad=2, dim=0.25)
    r.until_word(2, 'how a one-part body')
    r.callout(rect_u(pg.auto, pg.fields['fascia_top_m'][0]), '', dur=1.6, pad=4, dim=0.25)
    r.until_word(2, 'the glass,')
    r.callout(grown(rect_u(pg.fields['dash_reflectance'][0], pg.fields['calib_m'][0]), 122), '', dur=1.4, pad=2,
              dim=0.25)
    r.until_word(2, 'and the wipers')
    r.callout(rect_u(pg.w_on, pg.min_wiped), '', dur=max(1.6, r.speech_left() + 0.3), pad=4, dim=0.25)
    r.finish_speech(0.2)
    r.zoom(None, 0.8)

    # ---- 3: rays over each radar's field
    r.say(3)
    tr = r.wrect(rd)
    table_zoom = QRectF(tr.x() - 4, tr.y() - 30, 900, 30 + rd.rowHeight(0) * 6 + 30)
    r.zoom(table_zoom, 1.0)
    r.callout(span(rd, 0, 1, 4, 1), '', dur=2.6,
              pad=3, dim=0.25)
    r.until_word(3, 'layer by layer')
    r.callout(span(rd, 0, 5, 4, 5), '',
              dur=max(2.0, r.speech_left() + 0.3), pad=3, dim=0.25)
    r.finish_speech(0.2)

    # ---- 4: RD1 in front; the corner radars behind the fascia
    i1, i2 = row_of(rd, 'RD1'), row_of(rd, 'RD2')
    r.expect(rd.item(i1, 1).text(), 'nothing (in front of the body)', 'what RD1 looks through')
    r.expect(rd.item(i1, 5).text(), '0.00 dB', "RD1's loss")
    r.expect(rd.item(i2, 2).text().split('  ')[0], '11.7 mm', "RD2's gap to the fascia")
    r.say(4)
    r.callout(H.cell(r, rd, i1, None), '', dur=2.8, pad=2, dim=0.25)
    r.until_word(4, 'The four')
    r.callout(span(rd, i2, None, i2 + 3, None), '',
              dur=1.8, pad=2, dim=0.25)
    r.until_word(4, 'eleven point seven')
    r.callout(span(rd, i2, 2, i2 + 3, 2), '',
              dur=max(1.8, r.speech_left() + 0.3), pad=3, dim=0.25)
    r.finish_speech(0.2)

    # ---- 5: the fascia, its angle and the loss
    r.expect(rd.item(i2, 1).text(), 'Bumper fascia (PP, painted) (solid)', "RD2's surface")
    r.expect(rd.item(i2, 3).text(), '3.00 mm  2.43', "the fascia's thickness")
    r.expect(rd.item(i2, 4).text(), '29.2°', 'incidence')
    r.expect(rd.item(i2, 5).text(), '2.67 dB', 'two-way loss, solid paint')
    r.say(5)
    r.callout(H.cell(r, rd, i2, 1), '', dur=2.4, pad=3, dim=0.25)
    r.until_word(5, 'twenty-nine')
    r.callout(H.cell(r, rd, i2, 4), '', dur=1.6, pad=3, dim=0.25)
    r.until_word(5, 'There and back')
    r.figure('Two-way loss', 2.67, 'dB', 2, dur=max(2.8, r.speech_left() + 0.8), src=H.cell(r, rd, i2, 5))
    r.finish_speech(0.2)

    # ---- 6: range with the fourth root
    r.expect(rd.item(i2, 6).text(), '86 m', "RD2's derated range")
    r.expect(rd.item(i2, 6).toolTip(), 'zone ±75°: 100 → 86 m (-14 %)', "RD2's range before and after")
    r.say(6)
    r.until_word(6, 'a hundred metres')
    r.figure('Car seen at', 86, 'm', 0, dur=max(3.0, r.speech_left() + 0.8), src=H.cell(r, rd, i2, 6), start=100,
             note='100 m with nothing in front')
    r.finish_speech(0.2)

    # ---- 7: half wavelengths, reflection, multipath
    r.expect(rd.item(i2, 7).text(), '-6.8 dB', 'reflection back into the radar')
    r.expect(rd.item(i2, 8).text(), 'moderate', 'multipath risk')
    r.say(7)
    r.callout(H.cell(r, rd, i2, 3), '', dur=3.2, pad=3, dim=0.25)
    r.until_word(7, 'sends six point eight')
    r.callout(H.cell(r, rd, i2, 7), '', dur=2.4, pad=3, dim=0.25)
    r.until_word(7, 'a moderate')
    r.callout(H.cell(r, rd, i2, 8), '', dur=max(1.6, r.speech_left() + 0.3), pad=3, dim=0.25)
    r.finish_speech(0.2)
    r.zoom(None, 0.8)

    # ---- 8: RD2's loss map, hovered
    r.say(8)
    vpr = rd.viewport()
    rr = rd.visualRect(rd.model().index(i2, 0))
    r.click(vpr, QPointF(rr.center()), 0.6, real=False)
    rd.selectRow(i2)
    r.tick(2)
    r.expect(pg.beam.rep['tag'], 'RD2', 'the radar picked')
    bm = pg.beam
    br = bm._rect()
    bw = r.wrect(bm)
    r.zoom(QRectF(bw.x() + br.left() + br.width() * 0.42, bw.y(), br.width() * 0.58 + 80, br.height() * 0.62),
           1.0, wait=False)
    r.until_word(8, 'azimuth across')
    r.glide(bm, QPointF(br.left() + br.width() * 0.5, br.top() + br.height() * 0.3),
            QPointF(br.left() + br.width() * 0.9, br.top() + br.height() * 0.3), dur=max(2.6, r.speech_left() - 0.2))
    r.finish_speech(0.2)
    r.leave(bm)
    r.cursor_on = False

    # ---- 9: the text under the map: cavity and best thickness
    best = ctl.result.radar('RD2')['best_thickness']
    r.expect(f'{best["thickness_mm"]:.2f}', '3.61', 'the thickness that reflects least')
    r.expect('Least reflection near this thickness: 3.61 mm (-35.5 dB, 0.71 dB two-way)' in pg.radar_text.text(), True,
             'shown under the map')
    r.say(9)
    tx = r.wrect(pg.radar_text)
    r.zoom(QRectF(tx.x(), tx.y() - 60, tx.width(), tx.height() + 80), 1.0)
    r.callout(pg.radar_text, '', dur=2.4, pad=4, dim=0.2)
    r.until_word(9, 'three point six one')
    r.figure('Least reflection at', 3.61, 'mm', 2, dur=max(2.4, r.speech_left() + 0.8), src=pg.radar_text,
             note='-35.5 dB back, 0.71 dB two-way')
    r.finish_speech(0.3)
    r.zoom(None, 0.8)
    r.cursor_on = True

    # ---- 10: metallic paint
    r.say(10)
    paint_zoom = QRectF(r.wrect(pg.paint).x() - 10, r.wrect(pg.paint).y() - 30, 1260, 230)
    r.click(pg.paint.buttons['metallic'], None, 0.7)
    r.wait_for(lambda: ready() and ctl.result.settings.paint == 'metallic', timeout=60)
    r.zoom(paint_zoom, 1.0)
    r.expect(rd.item(i2, 5).text(), '3.89 dB', 'metallic loss')
    r.expect(rd.item(i2, 6).text(), '80 m', 'metallic range')
    r.until_word(10, 'three point eight nine')
    r.callout(H.cell(r, rd, i2, 5), '', dur=2.2, pad=3, dim=0.2)
    r.until_word(10, 'eighty metres')
    r.figure('Car seen at', 80, 'm', 0, dur=max(2.4, r.speech_left() + 0.6), src=H.cell(r, rd, i2, 6), start=86,
             note='metallic paint')
    r.finish_speech(0.2)

    # ---- 11: a repaired metallic bumper
    r.say(11)
    r.click(pg.paint.buttons['repaired'], None, 0.7)
    r.wait_for(lambda: ready() and ctl.result.settings.paint == 'repaired', timeout=60)
    r.expect(rd.item(i2, 5).text(), '8.31 dB', 'repaired metallic loss')
    r.expect(rd.item(i2, 6).text(), '62 m', 'repaired metallic range')
    f_ser = [f for f in ctl.result.findings if f['severity'] == 'serious']
    r.expect(len(f_ser) >= 4 and 'Use no effect paint over the radar' in f_ser[0]['message'], True,
             'the serious finding and its fix')
    r.until_word(11, 'eight point three one')
    r.figure('Two-way loss', 8.31, 'dB', 2, dur=2.6, src=H.cell(r, rd, i2, 5), start=3.89, note='repaired metallic')
    r.until_word(11, 'a serious finding')
    bar = pg.tabs.tabBar()
    r.callout(r.local_rect(bar, QRectF(bar.tabRect(3))), '', dur=max(1.6, r.speech_left() + 0.3), pad=3, dim=0.2)
    r.finish_speech(0.2)
    r.zoom(None, 0.8)

    # ---- 12: back to solid; the Parts tab
    r.say(12)
    r.click(pg.paint.buttons['solid'], None, 0.6)
    r.wait_for(lambda: ready() and ctl.result.settings.paint == 'solid', timeout=60)
    r.expect(rd.item(i2, 5).text(), '2.67 dB', 'back on solid paint')
    r.until_word(12, 'Under parts')
    click_tab(pg.tabs, 'Parts')
    r.tick(3)
    pt = pg.parts
    ptr = r.wrect(pt)
    parts_zoom = QRectF(ptr.x() - 4, ptr.y() - 30, 1100, 30 + pt.rowHeight(0) * 5 + 50)
    r.zoom(parts_zoom, 1.0)
    r.callout(span(pt, 0, 3, 3, 5), '',
              dur=2.4, pad=3, dim=0.25)
    r.until_word(12, "The sample's")
    r.callout(H.cell(r, pt, 0, None), '', dur=max(1.8, r.speech_left() + 0.3), pad=2, dim=0.25)
    r.finish_speech(0.2)

    # ---- 13: the automatic regions; set by hand
    reg = pg.regions_text.text()
    r.expect(reg.startswith('Automatic regions on the painted body: bumper fascia:') and 'windscreen:' in reg, True,
             'the automatic regions')
    r.say(13)
    r.callout(pg.regions_text, '', dur=3.8, pad=4, dim=0.25)
    r.until_word(13, 'Any part')
    r.callout(span(pt, 0, 3, 0, 4), '',
              dur=max(1.8, r.speech_left() + 0.3), pad=3, dim=0.25)
    r.finish_speech(0.2)

    # ---- 14: 3.61 mm typed
    r.say(14)
    HA.edit_cell(r, pt, 0, 4, '3.61')
    r.wait_for(lambda: ready() and w.store.project.vehicle.mesh.parts[0].surface_mm == 3.61, timeout=60)
    r.hold(0.3)
    click_tab(pg.tabs, 'Radars')
    r.tick(2)
    r.expect(rd.item(i2, 3).text(), '3.61 mm  2.92', 'thickness typed')
    r.expect(rd.item(i2, 7).text(), '-34.8 dB', 'reflection at 3.61 mm')
    r.expect(rd.item(i2, 5).text(), '1.88 dB', 'loss at 3.61 mm')
    r.expect(rd.item(i2, 8).text(), 'low', 'risk at 3.61 mm')
    r.zoom(table_zoom, 1.0)
    r.until_word(14, 'minus thirty-four')
    r.callout(H.cell(r, rd, i2, 7), '', dur=1.8, pad=3, dim=0.2)
    r.until_word(14, 'the loss to')
    r.callout(H.cell(r, rd, i2, 5), '', dur=1.6, pad=3, dim=0.2)
    r.until_word(14, 'multipath risk')
    r.callout(H.cell(r, rd, i2, 8), '', dur=max(1.4, r.speech_left() + 0.3), pad=3, dim=0.2)
    r.finish_speech(0.2)
    r.zoom(None, 0.8)

    # ---- 15: painted steel: blocked, critical
    r.say(15)
    r.menu('Edit', 'Undo')
    r.wait_for(lambda: ready() and w.store.project.vehicle.mesh.parts[0].surface_mm == 0, timeout=60)
    click_tab(pg.tabs, 'Parts')
    r.tick(2)
    r.choose(pt.cellWidget(0, 3), 'Painted steel panel')
    r.wait_for(lambda: ready() and ctl.result.radar('RD2')['mount'] == 'blocked', timeout=60)
    click_tab(pg.tabs, 'Radars')
    r.tick(2)
    r.expect(rd.item(i2, 5).text(), 'blocked', 'RD2 behind steel')
    crit = [f for f in ctl.result.findings if f['severity'] == 'critical']
    r.expect(len(crit), 4, 'critical findings for the four corner radars')
    r.zoom(table_zoom, 1.0)
    r.callout(span(rd, i2, 1, i2 + 3, 6), '',
              dur=2.4, pad=3, dim=0.2)
    r.until_word(15, 'That is critical')
    r.zoom(None, 0.7)
    click_tab(pg.tabs, 'Findings')
    r.tick(2)
    f0 = pg.findings[0]
    r.expect(f0.text.text().startswith('Radar RD2 looks into painted steel panel (100 % of its field)'), True,
             'the critical finding')
    fz = r.wrect(f0)
    r.zoom(QRectF(fz.x() - 10, fz.y() - 40, min(fz.width(), 1100) + 20, 200), 0.9)
    r.callout(f0, '', dur=max(1.6, r.speech_left() + 0.4), pad=3, dim=0.2)
    r.finish_speech(0.2)
    r.zoom(None, 0.7, wait=False)

    # ---- 16: undo; the library
    r.say(16)
    r.menu('Edit', 'Undo')
    r.wait_for(lambda: ready() and ctl.result.radar('RD2')['mount'] == 'behind', timeout=60)
    click_tab(pg.tabs, 'Parts')
    r.tick(2)
    lib = pg.lib
    lr2 = r.wrect(lib)
    r.zoom(QRectF(lr2.x() - 4, lr2.y() - 30, 1500, 30 + lib.rowHeight(0) * 13 + 30), 1.0)
    r.callout(span(lib, 0, 2, 11, 4), '',
              dur=3.0, pad=3, dim=0.25)
    r.until_word(16, 'each with its')
    r.callout(span(lib, 0, 5, 11, 5), '',
              dur=max(2.0, r.speech_left() + 0.4), pad=3, dim=0.25)
    r.finish_speech(0.2)
    r.zoom(None, 0.8)

    # ---- 17: the camera behind the windscreen
    r.say(17)
    click_tab(pg.tabs, 'Glass and wipers')
    r.tick(2)
    cm = pg.cams
    r.expect(cm.item(0, 0).text(), 'FW', 'the windscreen camera')
    note = r.note('FW: a windscreen camera added for this example (the sample has none)')
    cr = r.wrect(cm)
    r.zoom(QRectF(cr.x() - 4, cr.y() - 30, 760, 90), 1.0)
    r.callout(span(cm, 0, 0, 0, 1), '', dur=max(2.0, r.speech_left() + 0.3), pad=2, dim=0.25)
    r.finish_speech(0.2)
    c2 = r.target_rect(H.cell(r, cm, 0, 2))
    cams_zoom = QRectF(c2.x() - 40, cr.y() - 30, cr.right() - c2.x() + 44, 90)
    r.zoom(cams_zoom, 0.8)

    # ---- 18: light through the glass
    r.expect(cm.item(0, 2).text(), '66.9°', 'incidence on the windscreen')
    r.expect(cm.item(0, 3).text(), '73 %  −0.45 EV', 'light passed')
    r.expect(cm.item(0, 1).text(), 'Laminated windscreen · 4.96 mm', 'the glass')
    r.say(18)
    r.callout(H.cell(r, cm, 0, 2), '', dur=2.2, pad=3, dim=0.2)
    r.until_word(18, 'seventy-three')
    r.figure('Light passed', 73, '%', 0, dur=max(2.6, r.speech_left() + 0.6), src=H.cell(r, cm, 0, 3),
             note='0.45 stops darker')
    r.finish_speech(0.2)

    # ---- 19: the apparent centre shift
    r.expect(cm.item(0, 4).text(), '3.84 mm', 'apparent centre shift')
    r.expect(cm.item(0, 5).text(), '0.25 px', 'error left at 3 m')
    r.say(19)
    r.until_word(19, "F W's centre")
    r.figure('Centre appears off by', 3.84, 'mm', 2, dur=2.8, src=H.cell(r, cm, 0, 4))
    r.until_word(19, 'Calibrated')
    r.callout(H.cell(r, cm, 0, 5), '', dur=max(2.0, r.speech_left() + 0.4), pad=3, dim=0.2)
    r.finish_speech(0.2)

    # ---- 20: dashboard glare
    c_fw = ctl.result.camera('FW')
    r.expect(cm.item(0, 6).text(), '33 %', 'dashboard reflection')
    r.expect(f'{(1 - c_fw["contrast_kept"]) * 100:.0f}', '14', 'contrast lost')
    r.expect('about 14 % contrast lost' in cm.item(0, 6).toolTip(), True, 'contrast lost in the tooltip')
    r.say(20)
    r.until_word(20, 'thirty-three')
    r.figure('Dashboard reflected', 33, '%', 0, dur=max(3.0, r.speech_left() + 0.6), src=H.cell(r, cm, 0, 6),
             note='about 14 % contrast lost on a sunlit road')
    r.finish_speech(0.2)

    # ---- 21: the glare shield
    n_warn = sum(f['severity'] == 'warning' for f in ctl.result.findings)
    r.say(21)
    box = cm.cellWidget(0, 8)
    r.click(box, QPointF(9, box.height() / 2), 0.7, real=False)
    box.setChecked(True)
    r.wait_for(lambda: ready() and ctl.result.camera('FW')['shield'], timeout=60)
    r.expect(sum(f['severity'] == 'warning' for f in ctl.result.findings), n_warn - 1, 'the glare warning gone')
    r.callout(span(cm, 0, 6, 0, 8), '',
              dur=max(1.8, r.speech_left() + 0.4), pad=3, dim=0.2)
    r.finish_speech(0.2)
    r.close_layer(note)
    r.zoom(None, 0.8)

    # ---- 22: the windscreen from inside, typical tandem wipers
    ws = pg.ws_text.text()
    r.expect(ws.startswith('Windscreen: 1.65 m across the bottom, 0.64 m up the glass'), True, 'the windscreen size')
    r.expect('The wipers clear 63 % of it.' in ws, True, 'the wiped share')
    r.expect(cm.item(0, 7).text(), '100 %', "FW's view wiped")
    r.say(22)
    gv = pg.glass
    gr = r.wrect(gv)
    glass_zoom = QRectF(gr.x(), gr.y() + gr.height() * 0.22, gr.width(), gr.height() * 0.62)
    r.zoom(glass_zoom, 1.0)
    r.until_word(22, 'Typical tandem')
    r.callout(rect_u(pg.arms_head, H.cell(r, pg.arms, 1, 6)), '', dur=2.2,
              pad=4, dim=0.2)
    r.zoom(QRectF(gr.x(), gr.y(), r.wrect(pg.arms).right() - gr.x(), gr.height()), 0.8)
    r.until_word(22, 'sixty-three')
    r.callout(pg.ws_text, '', dur=1.8, pad=3, dim=0.2)
    r.until_word(22, "all of F W's")
    r.callout(H.cell(r, cm, 0, 7), '', dur=max(1.4, r.speech_left() + 0.3), pad=3, dim=0.2)
    r.finish_speech(0.2)

    # ---- 23: opposed arms
    r.say(23)
    r.zoom(None, 0.6, wait=False)
    r.click(pg.layout_seg.buttons['opposed'], None, 0.7)
    r.wait_for(lambda: ready() and ctl.result.wipers.settings.layout == 'opposed', timeout=60)
    r.expect('The wipers clear 55 % of it.' in pg.ws_text.text(), True, 'opposed wipers')
    r.expect(cm.item(0, 7).text(), '1 %', "FW's view wiped, opposed")
    foot_zoom = QRectF(560, r.wrect(cm).y() - 30, W_ - 560, 790)
    r.zoom(foot_zoom, 1.0)
    r.until_word(23, 'one per cent')
    r.callout(H.cell(r, cm, 0, 7), '', dur=max(1.6, r.speech_left() + 0.4), pad=3, dim=0.2)
    r.finish_speech(0.2)

    # ---- 24: your own arms: the driver's blade shortened
    r.say(24)
    r.zoom(None, 0.6, wait=False)
    r.click(pg.layout_seg.buttons['tandem'], None, 0.7)
    r.wait_for(lambda: ready() and ctl.result.wipers.settings.layout == 'tandem', timeout=60)
    r.expect(pg.arms.item(0, 0).text(), 'Driver', "the driver's arm")
    ar = r.wrect(pg.arms)
    r.zoom(QRectF(ar.x() - 4, ar.y() - 30, ar.width() * 0.62, 120), 0.8)
    HA.edit_cell(r, pg.arms, 0, 4, '0.2')
    r.wait_for(lambda: ready() and ctl.result.wipers.settings.layout == 'custom', timeout=60)
    r.expect(cm.item(0, 7).text(), '7 %', "FW's view wiped with a 0.2 m driver's blade")
    r.zoom(foot_zoom, 1.0)
    r.until_word(24, 'only seven')
    pt_fw = QPointF(gv._cams[0][0]) if gv._cams else QPointF(gv.width() / 2, gv.height() / 2)
    r.hover(gv, pt_fw, 0.8)
    r.expect(gv.tip is not None and gv.tip[1] == 'FW · 7 % wiped', True, 'the footprint read-out')
    r.hold(max(1.0, r.speech_left() + 0.4))
    r.finish_speech(0.2)
    r.leave(gv)
    r.zoom(None, 0.7, wait=False)

    # ---- 25: undo; findings and issues
    r.say(25)
    r.menu('Edit', 'Undo')
    r.wait_for(lambda: ready() and ctl.result.wipers.settings.layout == 'tandem', timeout=60)
    click_tab(pg.tabs, 'Findings')
    r.tick(2)
    fh = r.wrect(pg.find_host.parentWidget())
    r.zoom(QRectF(fh.x() - 6, fh.y() - 36, min(fh.width(), 1300), 420), 1.0)
    r.hold(max(1.0, r.speech_left() - 1.0))
    r.finish_speech(0.2)
    r.zoom(None, 0.7)

    # ---- 26: back to the main view, the wiper sweep overlay
    r.say(26)
    r.menu('View', 'Back to viewports')
    stage.settle(2)
    r.crossfade(0.4)
    r.menu('View', 'Overlays', 'Wiper sweep')
    r.expect(w.ctx.overlays.get('wipers'), True, 'the wiper sweep overlay')
    r.cursor_on = False
    r.finish_speech(0.1)

    # ---- 27: limits, over the sweep on the windscreen
    r.say(27)
    stage.fly(r, vp, az=-14.0, el=24.0, dist=d0 * 0.48, target=(1.6, 0.0, 1.05), dur=3.0)

    def sweep_area():
        pts = [HA.project_pt(r, vp, xyz) for xyz in ((1.95, 0.75, 1.25), (1.95, -0.75, 1.25), (2.2, 0.0, 1.1),
                                                     (1.7, 0.0, 1.4))]
        if any(q is None for q in pts):
            return None
        a = [r.wpos(vp, q) for q in pts]
        xs, ys = [q.x() for q in a], [q.y() for q in a]
        return QRectF(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
    r.callout(sweep_area, 'Wiper sweep', dur=max(2.5, r.speech_left() + 0.2), style='circle', dim=0.3)
    stage.orbit(r, vp, 16, max(2.0, r.speech_left() + 0.4), d_el=4)
    r.finish_speech(0.3)

    # ---- 28: batch
    def term(cmd, code, lines):
        return [('cmd', cmd)] + H.timed_output(lines, keep=noise, tones=True) + \
            [('wait', 0.4), ('cmd', 'echo $?'), ('out', str(code), 'warn' if code else 'ok')]
    r.say(28)
    span = H.terminal(r, term('vantage data-rig.vantage --materials materials.json', c_mt, l_mt), title='~/projects')
    print(f'terminal 28: {span:.1f} s against {r.speech_left():.1f} s of speech', flush=True)
    r.crossfade(0.5)
    r.hold(max(span + 0.6, r.speech_left() + 0.2))
    r.finish_speech(0.2)

    # ---- 29: export and the assistant
    r.say(29)
    span = H.terminal(r, [('cmd', 'vantage data-rig.vantage --export out/ --formats materials')] +
                      H.timed_output(l_ex, keep=noise) + [('wait', 0.3), ('cmd', 'ls out/materials')] +
                      H.timed_output(l_ls), title='~/projects')
    print(f'terminal 29: {span:.1f} s against {r.speech_left():.1f} s of speech', flush=True)
    r.crossfade(0.4)
    r.hold(max(span + 0.6, r.speech_left() + 0.8))
    r.finish_speech(1.0)
