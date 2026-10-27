"""45 Cable harness (episode v020, chapter 5): Analysis ▸ Cable harness on the sample's data rig with two GNSS antennas
and a dual-antenna INS added where Vehicle ▸ Add puts them (stage._nav_window; the sample's data rig has none). How a
route is made (device, outer skin, entry point, inside along the trim), the routing costs, the entry points, the tiles,
the front camera's run (longer than its fitted cable), how to read the table, link reaches, power pairs, the GNSS RF
budget and a thinner coax, the bulkhead grommet switched off, the three views, the "routed lengths everywhere" switch
and the Issues list, the 3D overlay, the timing plan's deserializer rule, limits, batch, export and the assistant.

Figures read from the app while recording (r.expect): 22 cables (15 data and RF, 7 power), 71.5 m, data and RF 48.3 m,
+17 % on the estimate; F 3.64 m against an estimate of 2.95 m and a 3 m cable, 121 %, serious, "use the 15 m cable";
F along the body 1.25 m, inside 2.14 m, service loop 0.25 m; RD1 3.76 m of 100BASE-T1, 25 % of 15 m; L1's power pair
1.5 %; GN2 4.18 m of RG-58, 13 dB budget, 9.9 dB margin; RG-174: 8.2 dB; the bulkhead grommet off: F 4.69 m through the
front left door seal, 156 %; the Issues row "Routed cable 3.64 m is longer than the 3.00 m module cable"; the timing
note on line A (6 cameras over two deserializers); the batch run and its exit code; the export's three files; the
assistant tool get_cable_harness exists.
"""

META = {
    'id': 'new-harness',
    'part': 2,
    'title': 'Cable harness',
    'promise': 'Every cable routed along the body, checked against its link.',
    'menu': 'Analysis ▸ Cable harness',
}

LINES = [
    # 0
    ("Every sensor on a rig needs a cable, and every cable has to reach. Until now Vantage estimated each run along the "
     "axes. Version 0.2.0 routes every cable along the body itself.",
     "Every sensor on a rig needs a cable, and every cable has to reach. Until now, Vantage estimated each run along "
     "the axes. Version zero point two routes every cable along the body itself."),
    # 1
    ("Analysis, Cable harness: here on the data rig, with two GNSS antennas and an INS added for this example.",
     "Analysis, Cable harness: here on the data rig, with two G N S S antennas and an I N S added for this example."),
    # 2
    ("Each cable leaves its device, follows the body's outer skin to an entry point, a grommet or a door seal, and runs "
     "inside along the trim to the compute box.",
     "Each cable leaves its device, follows the body's outer skin to an entry point, a grommet or a door seal, and "
     "runs inside along the trim to the compute box."),
    # 3
    ("On the skin, the cheapest path keeps off glass, at 25 times the cost, and off doors, bonnet, tailgate and shut "
     "lines, at 10. Then it is pulled straight wherever it can lie flat.",
     "On the skin, the cheapest path keeps off glass, at twenty-five times the cost, and off doors, bonnet, tailgate, "
     "and shut lines, at ten. Then it is pulled straight wherever it can lie flat."),
    # 4
    ("Inside, the run is the path along the axes times the compute box's slack, plus a service loop. Set the compute "
     "box outside, on a roof rack, and its cables are reached along the surface.",
     "Inside, the run is the path along the axes, times the compute box's slack, plus a service loop. Set the compute "
     "box outside, on a roof rack, and its cables are reached along the surface."),
    # 5
    ("Entry points are placed from the body's shape: a roof grommet, the bulkhead grommet at the cowl, four door seals, "
     "the tailgate seal and a rear-panel grommet. Move them, switch them off, or limit the cables they take.",
     "Entry points are placed from the body's shape: a roof grommet, the bulkhead grommet at the cowl, four door "
     "seals, the tailgate seal, and a rear panel grommet. Move them, switch them off, or limit the cables they take."),
    # 6
    ("The rig has 22 cables, 71.5 metres in all. Its data and RF runs come to 48.3 metres: 17 per cent more than the "
     "old estimate.",
     "The rig has twenty-two cables, seventy-one point five metres in all. Its data and R F runs come to forty-eight "
     "point three metres: seventeen per cent more than the old estimate."),
    # 7
    ("The tightest run is the front camera's. Click its route: the cable climbs over the bonnet to the bulkhead "
     "grommet, then runs inside to the Jetson, 3.64 metres in all.",
     "The tightest run is the front camera's. Click its route: the cable climbs over the bonnet to the bulkhead "
     "grommet, then runs inside to the Jetson, three point six four metres in all."),
    # 8
    ("The estimate said 2.95, and the camera's fitted cable is 3 metres: 121 per cent of it. Serious, and the finding "
     "gives the fix: the module's 15 metre cable.",
     "The estimate said two point nine five, and the camera's fitted cable is three metres: a hundred and twenty-one "
     "per cent of it. Serious, and the finding gives the fix: the module's fifteen metre cable."),
    # 9
    ("Each row is one cable: its length, its share of what bounds it, its status, the cable type and the entry point, "
     "both yours to change, the limit, the old estimate, and an extra length you can type.",
     "Each row is one cable: its length, its share of what bounds it, its status, the cable type and the entry point, "
     "both yours to change, the limit, the old estimate, and an extra length you can type."),
    # 10
    ("Hover a length to see its parts: for the front camera, 1.25 metres along the body, 2.14 inside, and a quarter "
     "metre of service loop.",
     "Hover a length to see its parts: for the front camera, one point two five metres along the body, two point one "
     "four inside, and a quarter metre of service loop."),
    # 11
    ("Each link has its own reach, typical figures with their basis: GMSL2 coax 15 metres, automotive Ethernet 15, "
     "ordinary Ethernet 100. The front radar's 3.76 metres use a quarter of its 100BASE-T1.",
     "Each link has its own reach, typical figures with their basis: G M S L two coax, fifteen metres, automotive "
     "Ethernet fifteen, ordinary Ethernet a hundred. The front radar's three point seven six metres use a quarter of "
     "its hundred base T one."),
    # 12
    ("The lidar, radars and the IMU also have power pairs, checked as the power budget checks them: the lidar's drops "
     "1.5 per cent.",
     "The lidar, the radars, and the I M U also have power pairs, checked as the power budget checks them: the "
     "lidar's drops one point five per cent."),
    # 13
    ("A GNSS antenna's coax is an RF budget: the antenna's 28 decibel amplifier, less the 15 the receiver needs, leaves "
     "13 for the cable. GN2's 4.18 metres of RG-58 keep 9.9 decibels in hand.",
     "A G N S S antenna's coax is an R F budget: the antenna's twenty-eight decibel amplifier, less the fifteen the "
     "receiver needs, leaves thirteen for the cable. G N two's four point one eight metres of R G fifty-eight keep "
     "nine point nine decibels in hand."),
    # 14
    ("Choose thinner RG-174 coax, and the margin falls to 8.2 decibels: still enough.",
     "Choose thinner R G one seventy-four coax, and the margin falls to eight point two decibels: still enough."),
    # 15
    ("Undo. Now switch the bulkhead grommet off: the front camera's cable goes in through the front door seal instead, "
     "4.69 metres, 156 per cent of its cable.",
     "Undo. Now switch the bulkhead grommet off: the front camera's cable goes in through the front door seal "
     "instead: four point six nine metres, a hundred and fifty-six per cent of its cable."),
    # 16
    ("Undo again. View the routes from above, from the side, or in 3D, and drag to orbit. Dashed parts run inside the "
     "body or behind it, and clicking a route picks its row in the table.",
     "Undo again. View the routes from above, from the side, or in 3 D, and drag to orbit. Dashed parts run inside the "
     "body, or behind it, and clicking a route picks its row in the table."),
    # 17
    ("Use the routed lengths everywhere makes these lengths replace the estimate in the Issues list, the camera cable "
     "check and the power budget. It is off by default, so existing results stay as they were.",
     "Use the routed lengths everywhere makes these lengths replace the estimate in the issues list, the camera cable "
     "check, and the power budget. It is off by default, so existing results stay as they were."),
    # 18
    ("Back in the main view, the Issues list now flags the front camera's cable as serious, and View, Overlays, Cable "
     "harness draws every route on the vehicle, in the 3D view and in the plan.",
     "Back in the main view, the issues list now flags the front camera's cable as serious; and View, Overlays, Cable "
     "harness, draws every route on the vehicle, in the 3 D view, and in the plan."),
    # 19
    ("The timing plan has a new rule too: a stereo pair or a trigger line split across two deserializers is flagged "
     "with the port moves that fix it.",
     "The timing plan has a new rule too: a stereo pair, or a trigger line, split across two deserializers is flagged, "
     "with the port moves that fix it."),
    # 20
    ("On this rig, line A drives six cameras, more than one deserializer's four links: wire it to the frame-sync "
     "input of both.",
     "On this rig, line A drives six cameras, more than one deserializer's four links: wire it to the frame sync input "
     "of both."),
    # 21
    ("The limits: inside, the route follows the axes, as the model has no interior. Bend radii, connector sizes, "
     "bundles and EMC separation are not modelled, and link reaches are typical.",
     "The limits: inside, the route follows the axes, as the model has no interior. Bend radii, connector sizes, "
     "bundles, and E M C separation are not modelled, and link reaches are typical."),
    # 22
    ("In batch mode, --harness writes every run and prints the findings. It exits with code 2 when a cable is longer "
     "than its link allows, or a GNSS cable has no margin left.",
     "In batch mode, harness writes every run, and prints the findings. It exits with code two when a cable is longer "
     "than its link allows, or a G N S S cable has no margin left."),
    # 23
    ("The export writes JSON with every path in the vehicle frame, a CSV harness sheet and a drawing, and the "
     "assistant reads the same harness with its get cable harness tool.",
     "The export writes Jason, with every path in the vehicle frame, a C S V harness sheet, and a drawing; and the "
     "assistant reads the same harness with its get cable harness tool."),
]

ROOT_RUN = 'run_vantage.py'


def scene(stage, r):
    import os
    import site
    import sys
    import numpy as np
    from PyQt6.QtCore import QPointF, QRectF, Qt
    from PyQt6.QtWidgets import QToolTip
    import helpers_tourc as H
    import helpers_tour_a as HA
    from vantage.core import samples, sensors
    from vantage.core.project import Sensor
    from tour import ease

    def nav(p):
        """The data rig with two GNSS antennas and a dual-antenna INS where Vehicle ▸ Add puts them (as
        stage._nav_window adds them)."""
        p.active_uid = next(v.uid for v in p.variants if v.sensors)
        v = p.variant
        for kind in ('gnss', 'gnss', 'imu'):
            m = p.sensor_models[sensors.default_model_id(kind)]
            v.sensors.append(Sensor(kind=kind, model_id=m.id, tag=sensors.next_tag(kind, v.tags()), name=m.name,
                                    **sensors.default_pose(kind, m, p.vehicle.mesh, p.rig, v.sensors)))

    # ---- the batch runs (real, not filmed), in the throw-away home
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
    nav(q)
    q.save(os.path.join(work, 'nav-rig.vantage'), b'')

    def run(cmd):
        code, lines, secs = H.run_timed(cmd, work, env, home)
        print(f'$ {cmd}  -> {code} in {secs:.1f} s', flush=True)
        for _, ln in lines:
            print('   ', ln, flush=True)
        return code, lines, secs
    c_hn, l_hn, _ = run('vantage nav-rig.vantage --harness harness.json')
    r.expect(any('F: 3.64 m routed, longer than its 3 m cable: use the 15 m cable.' in ln for _, ln in l_hn), True,
             'the batch finding for F')
    c_ex, l_ex, _ = run('vantage nav-rig.vantage --export out/ --formats harness')
    c_ls, l_ls, _ = run('ls out/harness')
    r.expect(sorted(ln for _, ln in l_ls), ['harness.csv', 'harness.json', 'harness.png'], 'the export files')
    from PyQt6.QtGui import QImage
    sheet = QImage(os.path.join(work, 'out', 'harness', 'harness.png'))
    r.expect(sheet.isNull(), False, 'the exported drawing')
    from vantage.assistant import tools as AT
    r.expect('get_cable_harness' in AT.BY_NAME, True, 'the assistant tool')
    noise = (lambda ln: not ln.startswith(('This plugin does not', '[ WARN')))

    # ---- the window: the data rig with GNSS and INS
    w = stage._nav_window()
    r.win = w
    vp = stage.vp
    vp.cam.az, vp.cam.el = -38.0, 22.0
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

    def row_of(t, text):
        return next(i for i in range(t.rowCount()) if t.item(i, 0) is not None and t.item(i, 0).text() == text)

    # ---- 0: the problem, over the 3D view
    r.lower_third()
    r.say(0)
    stage.orbit(r, vp, 40, max(3.0, r.speech_left() - 0.3), d_el=4)
    r.finish_speech(0.1)

    # ---- 1: the page
    r.say(1)
    note = r.note('GN1, GN2 and the INS added for this example, where Vehicle ▸ Add puts them')
    r.cursor_on = True
    r.hold(0.6)
    r.menu('Analysis', 'Cable harness')
    pg = w.page('harness')
    ctl = pg.ctl
    tb = pg.table
    view = pg.view

    def ready():
        return ctl.result is not None and not ctl.busy and not ctl.stale and tb.rowCount() > 0
    r.crossfade(0.4)
    r.wait_for(ready, timeout=90)
    r.finish_speech(0.2)
    r.close_layer(note)

    def B_():
        return view._frame(view.width(), view.height())

    def entry_pt(name):
        e = next(e for e in ctl.result.entries if e['name'] == name)
        B, sc, ce = B_()
        xy, _ = view._project(np.array([e['pos']]), B, sc, ce, view.width(), view.height())
        return QPointF(*xy[0])

    def entry_box(name, rad=16):
        def f():
            if not view.isVisible():
                return None
            q_ = entry_pt(name)
            return r.local_rect(view, QRectF(q_.x() - rad, q_.y() - rad, 2 * rad, 2 * rad))()
        return f

    def route_box(rid):
        def f():
            xy = view._screen.get(rid)
            if xy is None or not view.isVisible():
                return None
            x0, y0 = xy.min(axis=0)
            x1, y1 = xy.max(axis=0)
            return r.local_rect(view, QRectF(x0, y0, max(x1 - x0, 6), max(y1 - y0, 6)))()
        return f

    vr = r.wrect(view)

    # ---- 2: how a route is made
    r.say(2)
    r.zoom(QRectF(vr.x() + vr.width() * 0.12, vr.y(), vr.width() * 0.76, vr.height()), 1.0)
    f_id = next(x['id'] for x in ctl.result.runs if x['tag'] == 'F' and x['what'] == 'data')
    r.until_word(2, 'to an entry point')
    r.callout(entry_box('Bulkhead grommet'), '', dur=1.8, style='circle', pad=4, dim=0.2)
    r.until_word(2, 'runs inside')
    r.callout(lambda: r.local_rect(view, QRectF(view.width() * 0.25, view.height() * 0.28, view.width() * 0.5,
                                                view.height() * 0.44))(), '', dur=max(1.8, r.speech_left() + 0.3),
              pad=4, dim=0.2)
    r.finish_speech(0.2)
    r.zoom(None, 0.8)

    # ---- 3: the costs; pulled straight
    r.say(3)
    gl = pg.fields['glass_cost'][0]
    lr = r.wrect(gl)
    r.zoom(QRectF(lr.x() - 150, lr.y() - 70, 440, 230), 1.0)
    r.until_word(3, 'twenty-five')
    r.callout(lambda: r.target_rect(gl).adjusted(-136, -3, 3, 3), '', dur=1.8, pad=2, dim=0.25)
    r.until_word(3, 'and off doors')
    r.callout(lambda: r.target_rect(pg.fields['moving_cost'][0]).adjusted(-136, -3, 3, 3), '', dur=2.2, pad=2,
              dim=0.25)
    r.until_word(3, 'pulled straight')
    r.callout(lambda: r.target_rect(pg.fields['slack'][0]).adjusted(-136, -3, 3, 3), '',
              dur=max(1.6, r.speech_left() + 0.3), pad=2, dim=0.25)
    r.finish_speech(0.2)

    # ---- 4: inside; the compute box
    r.say(4)
    r.callout(rect_u(pg.ecu_fields['route_factor'], pg.ecu_fields['service_loop_m']), '', dur=3.0, pad=4, dim=0.25)
    r.until_word(4, 'Set the compute box')
    r.callout(pg.compute, '', dur=max(2.0, r.speech_left() + 0.3), pad=4, dim=0.25)
    r.finish_speech(0.2)

    # ---- 5: entry points
    r.say(5)
    et = pg.entries
    er = r.wrect(et)
    r.zoom(QRectF(er.x() - 6, er.y() - 40, er.width() + 12, er.height() + 80), 1.0)
    r.callout(span(et, 0, 0, 7, 4), '', dur=4.5, pad=3, dim=0.25)
    r.until_word(5, 'Move them')
    r.callout(span(et, 0, 2, 7, 6), '', dur=max(2.4, r.speech_left() + 0.3), pad=3, dim=0.25)
    r.finish_speech(0.2)
    r.zoom(None, 0.8)

    # ---- 6: the tiles
    T = pg.tiles
    r.expect((T['cables'].value, T['cables'].note), ('22', '15 data and RF · 7 power'), 'cables')
    r.expect((T['total'].value, T['total'].note), ('71.5', 'data and RF 48.3 m · +17 % on the estimate'),
             'total length')
    r.say(6)
    t0, t1 = r.wrect(T['cables']), r.wrect(T['entries'])
    r.zoom(QRectF(t0.x(), t0.y() - 10, t1.right() - t0.x(), t0.height() + 20), 1.0)
    r.figure('Cables', 22, '', 0, dur=2.6, src=T['cables'], note='15 data and RF · 7 power')
    r.until_word(6, 'seventy-one point five')
    r.figure('In all', 71.5, 'm', 1, dur=2.2, src=T['total'])
    r.until_word(6, 'seventeen per cent')
    r.callout(T['total'], '', dur=max(1.6, r.speech_left() + 0.4), pad=3, dim=0.2)
    r.finish_speech(0.2)
    r.zoom(None, 0.8)

    # ---- 7: F's route, clicked in the view
    iF = row_of(tb, 'F · data')
    r.expect(tb.item(iF, 1).text(), '3.64', "F's routed length")
    r.expect(T['tightest'].note, 'F · 3.64 m of its 3 m cable', 'the tightest run')
    r.say(7)
    r.callout(T['tightest'], '', dur=2.0, pad=3, dim=0.2)
    r.zoom(QRectF(vr.x() + vr.width() * 0.12, vr.y(), vr.width() * 0.76, vr.height()), 1.0)
    r.until_word(7, 'Click its route')
    xy = view._screen[f_id]
    k = int(len(xy) * 0.75)
    pt = QPointF(*((xy[k] + xy[min(k + 1, len(xy) - 1)]) / 2))
    HA.click_sent(r, view, pt, dur=0.7)
    r.wait_for(lambda: pg.selected == f_id, timeout=10)
    r.tick(2)
    r.until_word(7, 'climbs over')
    r.click(pg.mode.buttons['side'], None, 0.6)
    r.tick(3)
    r.callout(route_box(f_id), '', dur=max(2.4, r.speech_left() + 0.3), pad=10, dim=0.2)
    r.finish_speech(0.2)

    # ---- 8: against the estimate and the fitted cable
    r.expect(tb.item(iF, 8).text(), '2.95', "F's Manhattan estimate")
    r.expect(tb.item(iF, 7).text(), '3 m cable', "F's fitted cable")
    r.expect((tb.item(iF, 2).text(), tb.item(iF, 3).text()), ('121 %', 'Serious'), "F's use and status")
    fF = next(f for f in pg.findings if f.text.text().startswith('F: '))
    r.expect(fF.text.text(), 'F: 3.64 m routed, longer than its 3 m cable: use the 15 m cable.', "F's finding")
    r.say(8)
    tr_ = r.wrect(tb)
    r.zoom(QRectF(tr_.x() - 4, tr_.y() - 30, tr_.width() * 0.98, 30 + tb.rowHeight(0) * 4 + 40), 1.0)
    r.callout(H.cell(r, tb, iF, 8), '', dur=1.8, pad=3, dim=0.2)
    r.until_word(8, "the camera's fitted")
    r.callout(H.cell(r, tb, iF, 7), '', dur=1.8, pad=3, dim=0.2)
    r.until_word(8, 'a hundred and twenty-one')
    r.figure('Of its cable', 121, '%', 0, dur=2.4, src=H.cell(r, tb, iF, 2), note='3.64 m on a 3 m cable')
    r.until_word(8, 'the finding gives')
    fr = r.wrect(fF)
    r.zoom(QRectF(fr.x() - 10, fr.y() - 60, 460, 160), 0.8)
    r.callout(fF, '', dur=max(1.8, r.speech_left() + 0.4), pad=3, dim=0.2)
    r.finish_speech(0.2)

    # ---- 9: reading the table
    r.say(9)
    r.zoom(QRectF(tr_.x() - 4, tr_.y() - 30, tr_.width() * 0.98, 30 + tb.rowHeight(0) * 6 + 40), 0.8)
    for word, c0, c1 in (('its length', 1, 1), ('its share', 2, 2), ('its status', 3, 3), ('the cable type', 4, 5),
                         ('the limit', 7, 7), ('the old estimate', 8, 8), ('and an extra', 9, 9)):
        r.until_word(9, word)
        r.callout(span(tb, 0, c0, 5, c1), '', dur=1.4 if word != 'and an extra' else max(1.4, r.speech_left() + 0.3),
                  pad=3, dim=0.25)
    r.finish_speech(0.2)

    # ---- 10: hover a length: its parts
    tipF = tb.item(iF, 1).toolTip()
    r.expect(tipF, 'along the body 1.25 m · inside 2.14 m · service loop 0.25 m · extra 0.00 m', "F's length parts")
    r.say(10)
    cF = tb.visualItemRect(tb.item(iF, 1))
    r.hover(tb.viewport(), QPointF(cF.center()), 0.7)
    QToolTip.showText(tb.viewport().mapToGlobal(cF.bottomLeft()), tipF, tb.viewport())
    r.hold(max(1.5, r.speech_left() + 0.3))
    QToolTip.hideText()
    r.finish_speech(0.2)

    # ---- 11: link reaches; RD1
    i_rd1 = row_of(tb, 'RD1 · data')
    r.expect((tb.item(i_rd1, 1).text(), tb.item(i_rd1, 2).text(), tb.item(i_rd1, 7).text()), ('3.76', '25 %', '15 m'),
             "RD1's run")
    r.expect(tb.cellWidget(i_rd1, 4).currentText(), '100BASE-T1 · 15 m', "RD1's link")
    r.say(11)
    r.zoom(QRectF(tr_.x() - 4, tr_.y() - 30, tr_.width() * 0.98, tr_.height() + 30), 0.9)
    r.callout(span(tb, 0, 4, 6, 4), '', dur=2.4, pad=3, dim=0.25)
    r.until_word(11, 'automotive')
    r.callout(span(tb, i_rd1, 4, i_rd1, 4), '', dur=1.4, pad=3, dim=0.25)
    r.until_word(11, 'ordinary Ethernet')
    r.callout(span(tb, row_of(tb, 'L1 · data'), 4, row_of(tb, 'L1 · data'), 4), '', dur=1.4, pad=3, dim=0.25)
    r.until_word(11, "The front radar's")
    r.callout(span(tb, i_rd1, 0, i_rd1, 7), '', dur=max(1.8, r.speech_left() + 0.3), pad=3, dim=0.25)
    r.finish_speech(0.2)

    # ---- 12: power pairs
    i_lp = row_of(tb, 'L1 · power')
    r.expect(tb.item(i_lp, 2).text(), '1.5 %', "the lidar's power-pair drop")
    r.say(12)
    r.callout(span(tb, i_lp, 0, i_lp, 7), '', dur=2.6, pad=3, dim=0.25)
    r.until_word(12, "lidar's drops")
    r.callout(H.cell(r, tb, i_lp, 2), '', dur=max(1.4, r.speech_left() + 0.3), pad=3, dim=0.25)
    r.finish_speech(0.2)

    # ---- 13: the GNSS RF budget
    i_g2 = row_of(tb, 'GN2 · RF')
    tb.scrollToItem(tb.item(i_g2, 0))
    r.tick(2)
    g2 = ctl.result.data_run(next(x['uid'] for x in ctl.result.runs if x['tag'] == 'GN2'))
    r.expect((tb.item(i_g2, 1).text(), tb.item(i_g2, 7).text()), ('4.18', '13 dB'), "GN2's run and budget")
    r.expect(f'{g2["margin_db"]:.1f}', '9.9', "GN2's margin")
    fG = next(f for f in pg.findings if 'GNSS cable' in f.text.text())
    r.expect(fG.text.text(), 'Every GNSS cable keeps its RF margin: the least, GN2, 9.9 dB over 4.2 m of RG-58 coax.',
             'the RF finding')
    r.say(13)
    rf_r = r.wrect(pg.rf)
    r.zoom(QRectF(rf_r.x() - 6, rf_r.y() - 30, rf_r.width() + 12, 150), 1.0)
    r.callout(rect_u(pg.fields['gnss_lna_db'][0], pg.fields['gnss_min_db'][0]), '', dur=3.2, pad=3, dim=0.25)
    r.until_word(13, 'G N two')
    r.zoom(QRectF(tr_.x() - 4, tr_.y() + tr_.height() * 0.35, tr_.width() * 0.98, tr_.height() * 0.65), 0.8)
    r.callout(span(tb, i_g2, 0, i_g2, 7), '', dur=1.8, pad=3, dim=0.25)
    r.until_word(13, 'nine point nine')
    fgr = r.wrect(fG)
    r.zoom(QRectF(fgr.x() - 10, fgr.y() - 60, 460, 170), 0.6)
    r.callout(fG, '', dur=max(1.4, r.speech_left() + 0.3), pad=3, dim=0.2)
    r.finish_speech(0.2)

    # ---- 14: RG-174
    r.say(14)
    r.zoom(QRectF(rf_r.x() - 6, rf_r.y() - 30, rf_r.width() + 12, 150), 0.8, wait=False)
    r.choose(pg.rf, 'RG-174')
    r.wait_for(lambda: ready() and ctl.result.settings.gnss_cable == 'rf_rg174', timeout=60)
    fG = next(f for f in pg.findings if 'GNSS cable' in f.text.text())
    r.expect(fG.text.text(), 'Every GNSS cable keeps its RF margin: the least, GN2, 8.2 dB over 4.2 m of RG-174 coax.',
             'the RF finding with RG-174')
    fgr = r.wrect(fG)
    r.zoom(QRectF(fgr.x() - 10, fgr.y() - 60, 460, 170), 0.8)
    r.callout(fG, '', dur=max(1.6, r.speech_left() + 0.4), pad=3, dim=0.2)
    r.finish_speech(0.2)

    # ---- 15: undo; the bulkhead grommet off
    r.say(15)
    r.zoom(None, 0.6, wait=False)
    r.menu('Edit', 'Undo')
    r.wait_for(lambda: ready() and ctl.result.settings.gnss_cable == 'rf_rg58', timeout=60)
    ib = row_of(et, 'Bulkhead grommet')
    r.zoom(QRectF(er.x() - 6, er.y() - 40, er.width() + 12, er.height() + 80), 0.8)
    on = et.item(ib, 5)
    rect = et.visualItemRect(on)
    r.click(et.viewport(), QPointF(rect.center()), 0.6, real=False)
    on.setCheckState(Qt.CheckState.Unchecked)
    r.wait_for(lambda: ready() and not any(e['enabled'] for e in ctl.result.entries
                                           if e['name'] == 'Bulkhead grommet'), timeout=60)
    iF = row_of(tb, 'F · data')
    r.expect((tb.item(iF, 1).text(), tb.item(iF, 2).text()), ('4.69', '156 %'), 'F without the bulkhead grommet')
    r.expect(tb.cellWidget(iF, 5).currentText(), 'Shortest (Door seal front left)', "F's new entry point")
    r.zoom(QRectF(vr.x(), vr.y(), vr.width(), tr_.bottom() - vr.y()), 0.9)
    r.until_word(15, 'front door seal')
    r.callout(route_box(f_id), '', dur=2.0, pad=10, dim=0.2)
    r.until_word(15, 'four point six nine')
    r.callout(span(tb, iF, 1, iF, 5), '', dur=max(1.8, r.speech_left() + 0.3), pad=3, dim=0.2)
    r.finish_speech(0.2)

    # ---- 16: undo; the three views
    r.say(16)
    r.zoom(None, 0.6, wait=False)
    r.menu('Edit', 'Undo')
    r.wait_for(lambda: ready() and all(e['enabled'] for e in ctl.result.entries), timeout=60)
    r.zoom(QRectF(vr.x(), vr.y() - 40, vr.width(), vr.height() + 40), 0.8)
    r.click(pg.mode.buttons['top'], None, 0.5)
    r.hold(0.8)
    r.click(pg.mode.buttons['side'], None, 0.5)
    r.hold(0.8)
    r.until_word(16, 'or in 3 D')
    r.click(pg.mode.buttons['3d'], None, 0.5)
    r.tick(3)
    r.cursor_on = False
    y0, n = view.yaw, round(max(1.6, r.speech_left() - 0.2) * stage.fps)
    view._dragging = True
    for k in range(1, n + 1):
        view.yaw = y0 - 50.0 * ease(k / n)
        view.update()
        r.tick()
    view._dragging = False
    view._refine()
    r.tick(2)
    r.finish_speech(0.2)
    r.cursor_on = True
    r.zoom(None, 0.8)

    # ---- 17: the switch
    r.say(17)
    r.zoom(QRectF(r.wrect(pg.enabled).x() - 10, r.wrect(pg.enabled).y() - 120, 560, 260), 0.8)
    r.click(pg.enabled, QPointF(12, pg.enabled.height() / 2), 0.7)
    if not ctl.settings.enabled:
        pg.enabled.click()
    r.wait_for(lambda: ready() and ctl.result.settings.enabled, timeout=60)
    r.callout(pg.enabled, '', dur=max(2.0, r.speech_left() + 0.3), pad=4, dim=0.25)
    r.finish_speech(0.2)
    r.zoom(None, 0.7)

    # ---- 18: back in the main view: the Issues list and the overlay
    r.say(18)
    r.menu('View', 'Back to viewports')
    stage.settle(2)
    r.crossfade(0.4)
    iss = w.bottom.issues
    tabs = w.bottom.tabs
    r.click(tabs, QPointF(tabs.tabRect(w.bottom.keys.index('issues')).center()), 0.6, real=False)
    w.bottom.show_tab('issues')
    r.wait_for(lambda: not w.ctx.analysis._busy and not w.ctx.analysis.timer.isActive(), timeout=60)
    for _ in range(4):
        stage.app.processEvents()
    iss.refresh()
    r.tick(3)
    rows = [(iss.t.item(i, 0).text(), iss.t.item(i, 1).text()) for i in range(iss.t.rowCount())]
    print('issues:', rows, flush=True)
    i_cab = next(i for i, x in enumerate(rows) if x[1].startswith('Routed cable 3.64 m is longer than the 3.00 m'))
    r.expect(rows[i_cab][0], 'Serious', 'the cable issue')
    bb = r.wrect(w.docks['bottom'])
    r.zoom(QRectF(bb.x(), bb.y() - 10, min(bb.width() * 0.7, 1300), bb.height() + 10), 0.9)
    a_ = iss.t.visualItemRect(iss.t.item(i_cab, 0))
    b_ = iss.t.visualItemRect(iss.t.item(i_cab, 1))
    r.callout(r.local_rect(iss.t.viewport(), QRectF(a_.left(), a_.top(), b_.right() - a_.left(), a_.height())), '',
              dur=2.6, pad=2, dim=0.25)
    r.until_word(18, 'View, Overlays')
    r.zoom(None, 0.6)
    r.menu('View', 'Overlays', 'Cable harness')
    r.expect(w.ctx.overlays.get('harness'), True, 'the cable harness overlay')
    r.cursor_on = False
    stage.fly(r, vp, az=-30.0, el=30.0, dist=d0 * 0.5, target=(1.2, 0.0, 0.9), dur=max(1.2, r.speech_left() + 0.1))
    r.finish_speech(0.2)

    # ---- 19, 20: the timing plan's deserializer rule
    r.say(19)
    r.cursor_on = True
    r.menu('Analysis', 'Sync and timing')
    tp = w.page('timing')
    stage.settle(2)
    r.crossfade(0.4)
    nl = None
    for k_ in range(tp.notes_box.count()):
        wd = tp.notes_box.itemAt(k_).widget()
        if wd is not None and hasattr(wd, 'text') and wd.text.text().startswith('Trigger line A drives 6 cameras'):
            nl = wd
    r.expect(nl is not None, True, 'the trigger-line note')
    r.expect(nl.text.text(), "Trigger line A drives 6 cameras, more than one deserializer's 4 links (A: F R ML MR; B: QL "
                             "QR): wire the line to the frame-sync input of both deserializers so their frames start "
                             "together.", 'the trigger-line note')
    r.finish_speech(0.2)
    r.say(20)
    nr = r.wrect(nl)
    r.zoom(QRectF(nr.x() - 10, nr.y() - 60, nr.width() + 20, nr.height() + 120), 1.0)
    r.callout(nl, '', dur=max(2.4, r.speech_left() + 0.4), pad=4, dim=0.25)
    r.finish_speech(0.2)
    r.zoom(None, 0.7)

    # ---- 21: limits, over the routes in 3D
    r.say(21)
    r.menu('View', 'Back to viewports')
    stage.settle(2)
    r.crossfade(0.4)
    r.cursor_on = False
    stage.orbit(r, vp, 40, max(2.5, r.speech_left() + 0.3), d_el=3)
    r.finish_speech(0.3)

    # ---- 22: batch
    def term(cmd, code, lines):
        return [('cmd', cmd)] + H.timed_output(lines, keep=noise, tones=True) + \
            [('wait', 0.4), ('cmd', 'echo $?'), ('out', str(code), 'warn' if code else 'ok')]
    r.say(22)
    sp = H.terminal(r, term('vantage nav-rig.vantage --harness harness.json', c_hn, l_hn), title='~/projects')
    print(f'terminal 22: {sp:.1f} s against {r.speech_left():.1f} s', flush=True)
    r.crossfade(0.5)
    r.hold(max(sp + 0.5, r.speech_left() + 0.2))
    r.finish_speech(0.2)

    # ---- 23: export, the drawing, the assistant
    r.say(23)
    sp = H.terminal(r, [('cmd', 'vantage nav-rig.vantage --export out/ --formats harness')] +
                    H.timed_output(l_ex, keep=noise) + [('wait', 0.3), ('cmd', 'ls out/harness')] +
                    H.timed_output(l_ls), title='~/projects')
    print(f'terminal 23: {sp:.1f} s against {r.speech_left():.1f} s', flush=True)
    r.crossfade(0.4)
    r.hold(sp + 0.4)
    H.pages_card(r, [sheet], 'out/harness/harness.png: the routes from above and from the left')
    r.crossfade(0.5)
    r.hold(max(2.5, r.speech_left() + 0.8))
    r.finish_speech(1.0)
