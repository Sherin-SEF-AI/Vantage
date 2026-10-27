"""46 Calibration observability and fusion (episode v020, chapter 6): Calibrate ▸ Calibration observability on the
sample's data rig and its eight planned floor boards. The method (corners through each lens, the corner noise, boards
placed by hand as the joint solve models them), the plan, the table of 1σ, the front camera picked (its boards, its
weakest direction, the reason in plain words, the suggestions tried and measured), "Use ±5 mm" applied and undone,
camera pairs, lidar–camera pairs, corner reflectors for the radars, the Monte Carlo check through the real solves, the
camera–lidar fusion maps and table, issues, limits, batch, export and the assistant.

Figures read from the app while recording (r.expect): 8 boards; corner noise 0.30 px, placement ±20 mm and ±1.00°;
F yaw ±0.131° (1.31× the 0.1° requirement; the loosest QR 1.34×), F counts T2 T3 T1 with 75 corners, shares 34/34/31 %,
without T1 1.44×, weakest direction y +8.5 mm; corners alone ±0.012° (said as 0.01°); the suggestions 1.3× -> 1.2× (a
board added) and 1.3× -> 0.5× (±5 mm and ±0.25°); applied: 6 of 6 within, loosest 0.58×; the F–ML pair through T2,
yaw 0.050°, 0.50×; F–L1 z 16.4 mm, 1.64×, the camera's corners limit it; lidar–camera 0 of 6 within; two reflectors
(the second 3 m aside): RD1 yaw ±2.535° (2.5°), 18.13×, "Add a reflector farther out or at another bearing"; the Monte
Carlo check with 30 trials: holds, every σ within ×1.59, 0 of 30 joint solves stopped short, ratios 0.72 to 1.30; F's
fusion map 22 % with lidar, 1 point per 37 px; a pedestrian at 20 m 48 points on 13 lines, at 50 m 8 points (0 %
enough); the batch run's lines and exit code; the assistant tool get_calibration_observability exists.
"""

META = {
    'id': 'new-observability',
    'part': 2,
    'title': 'Calibration observability and fusion',
    'promise': 'How firmly the planned boards will pin every sensor, before you capture.',
    'menu': 'Calibrate ▸ Calibration observability',
}

LINES = [
    # 0
    ("A calibration can only be as good as what its boards let it measure. Calibration observability tells you how "
     "firmly the planned boards will pin each sensor, before anything is printed or captured.",
     "A calibration can only be as good as what its boards let it measure. Calibration observability tells you how "
     "firmly the planned boards will pin each sensor, before anything is printed, or captured."),
    # 1
    ("Calibrate, Calibration observability: on the data rig, with its eight planned floor boards.",
     "Calibrate, Calibration observability: on the data rig, with its eight planned floor boards."),
    # 2
    ("For every camera it takes the corners of each board it sees whole, through the camera's own lens model, each "
     "found to 0.3 pixels.",
     "For every camera, it takes the corners of each board it sees whole, through the camera's own lens model, each "
     "found to zero point three pixels."),
    # 3
    ("Boards laid by hand may sit 20 millimetres and 1 degree off their plan, as the installation check's joint solve "
     "assumes. The requirement is 0.1 degrees and 10 millimetres.",
     "Boards laid by hand may sit twenty millimetres and one degree off their plan, as the installation check's joint "
     "solve assumes. The requirement is zero point one degrees, and ten millimetres."),
    # 4
    ("The plan shows the boards, shaded by how many cameras count them, and each camera, ringed by its verdict.",
     "The plan shows the boards, shaded by how many cameras count them, and each camera, ringed by its verdict."),
    # 5
    ("The table gives each camera's one sigma in yaw, pitch and roll, and in position. Every camera's yaw comes out "
     "near 0.13 degrees: 1.3 times the requirement.",
     "The table gives each camera's one sigma, in yaw, pitch, and roll, and in position. Every camera's yaw comes out "
     "near zero point one three degrees: one point three times the requirement."),
    # 6
    ("Pick a camera. F counts three boards, 75 corners, each board carrying about a third; without T1 it would be "
     "1.44 times. Its weakest direction is a sideways shift of 8.5 millimetres.",
     "Pick a camera. F counts three boards, seventy-five corners, each board carrying about a third; without T one, "
     "it would be one point four four times. Its weakest direction is a sideways shift of eight point five "
     "millimetres."),
    # 7
    ("The reason, in plain words: yaw rests on where the boards lie. With the boards exactly where planned, F's corners "
     "alone would give 0.01 degrees. The placement is the loose part, not the camera.",
     "The reason, in plain words: yaw rests on where the boards lie. With the boards exactly where planned, F's "
     "corners alone would give zero point zero one degrees. The placement is the loose part, not the camera."),
    # 8
    ("So the suggestions are tried and measured. Another board in F's view gains little, 1.3 to 1.2 times. Laying the "
     "boards to 5 millimetres and a quarter degree, against a jig, takes F to 0.5.",
     "So the suggestions are tried, and measured. Another board in F's view gains little: one point three to one "
     "point two times. Laying the boards to five millimetres and a quarter degree, against a jig, takes F to zero "
     "point five."),
    # 9
    ("Use it, and the study runs again: all six cameras within the requirement, the loosest at 0.58 times. Undo takes "
     "it back, and the study runs once more on the boards as planned.",
     "Use it, and the study runs again: all six cameras within the requirement, the loosest at zero point five eight "
     "times. Undo takes it back, and the study runs once more on the boards as planned."),
    # 10
    ("Two cameras that see one board are tied by it. F and ML share T2: their relative yaw is pinned to 0.05 degrees, "
     "half the requirement, far firmer than either camera alone.",
     "Two cameras that see one board are tied by it. F and M L share T two: their relative yaw is pinned to zero point "
     "zero five degrees, half the requirement, far firmer than either camera alone."),
    # 11
    ("The lidar–camera tab uses the boards the lidar–camera calibration plans. For F and L1, the lidar's height is "
     "pinned to 16.4 millimetres, 1.6 times its tolerance.",
     "The lidar camera tab uses the boards the lidar camera calibration plans. For F and L one, the lidar's height is "
     "pinned to sixteen point four millimetres, one point six times its tolerance."),
    # 12
    ("The camera's corners limit it: where F puts each board, its distance above all. This count leaves out the board "
     "edges and the reflectivity checker, which only improve it.",
     "The camera's corners limit it: where F puts each board, its distance above all. This count leaves out the board "
     "edges and the reflectivity checker, which only improve it."),
    # 13
    ("Radars are pinned by corner reflectors you place, each with its size and its radar cross-section at 77 "
     "gigahertz. Add two, and move one 3 metres aside: with typical radar noise they hold RD1's yaw only to 2.5 "
     "degrees.",
     "Radars are pinned by corner reflectors you place, each with its size, and its radar cross-section at "
     "seventy-seven gigahertz. Add two, and move one three metres aside: with typical radar noise, they hold R D one's "
     "yaw only to two point five degrees."),
    # 14
    ("The finding asks for a reflector farther out, or at another bearing. The radar noise figures are typical: enter "
     "your radar's own. Undo removes the reflectors again.",
     "The finding asks for a reflector farther out, or at another bearing. The radar noise figures are typical: enter "
     "your radar's own. Undo removes the reflectors again."),
    # 15
    ("These are predictions. Verify checks them: 30 simulated captures, with noisy corners and boards laid off their "
     "plan, solved by the installation check's own joint solve and the lidar–camera calibration's.",
     "These are predictions. Verify checks them: thirty simulated captures, with noisy corners, and boards laid off "
     "their plan, solved by the installation check's own joint solve, and the lidar camera calibration's."),
    # 16
    ("The spread matches the prediction: every ratio between 0.72 and 1.30, within the band, and the joint solve "
     "reached its optimum in all 30 trials.",
     "The spread matches the prediction: every ratio between zero point seven two and one point three, within the "
     "band, and the joint solve reached its optimum in all thirty trials."),
    # 17
    ("The Fusion tab shows where lidar points land in each camera's image, on a scene 20 metres out. F: 22 per cent of "
     "its image gets a lidar point, one for every 37 pixels.",
     "The fusion tab shows where lidar points land in each camera's image, on a scene twenty metres out. F: "
     "twenty-two per cent of its image gets a lidar point, one for every thirty-seven pixels."),
    # 18
    ("Black is where no point lands: below the lidar's lowest beam, and on the near ground.",
     "Black is where no point lands: below the lidar's lowest beam, and on the near ground."),
    # 19
    ("The table counts lidar points and scan lines on each object: a pedestrian 20 metres ahead of F gets 48 points on "
     "13 lines; at 50 metres, only 8, fewer than a detector needs.",
     "The table counts lidar points, and scan lines, on each object: a pedestrian twenty metres ahead of F gets "
     "forty-eight points on thirteen lines; at fifty metres, only eight, fewer than a detector needs."),
    # 20
    ("The Issues list flags a camera that sees no board whole, one beyond twice the requirement, and a radar short of "
     "reflectors.",
     "The issues list flags a camera that sees no board whole, one beyond twice the requirement, and a radar short of "
     "reflectors."),
    # 21
    ("The limits: lens models are taken as exact, corner noise as equal everywhere, boards lie flat on the floor, and "
     "the radar noise figures are typical.",
     "The limits: lens models are taken as exact, corner noise as equal everywhere, boards lie flat on the floor, and "
     "the radar noise figures are typical."),
    # 22
    ("In batch mode, --observability writes JSON, CSV and pictures, and --verify adds the Monte Carlo check. It exits "
     "with code 2 when a camera sees no board whole.",
     "In batch mode, observability writes Jason, C S V, and pictures, and verify adds the Monte Carlo check. It exits "
     "with code two when a camera sees no board whole."),
    # 23
    ("The assistant reads the same study with its calibration observability tool, and can run the check as well.",
     "The assistant reads the same study with its calibration observability tool, and can run the check as well."),
]

ROOT_RUN = 'run_vantage.py'


def scene(stage, r):
    import math
    import os
    import site
    import sys
    import numpy as np
    from PyQt6.QtCore import QPointF, QRectF
    from PyQt6.QtWidgets import QPushButton
    import helpers_tourc as H
    from vantage.core import samples

    # ---- the batch run (real, not filmed), in the throw-away home
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
    c_ob, l_ob, s_ob = run('vantage data-rig.vantage --observability obs/ --verify 30')
    r.expect(c_ob, 0, 'exit code of --observability')
    r.expect(any(ln.strip() == 'Monte Carlo: the predictions hold' for _, ln in l_ob), True, 'the batch verdict')
    c_ls, l_ls, _ = run('ls obs/')
    r.expect(sorted(ln for _, ln in l_ls), ['fusion_density.png', 'observability.csv', 'observability.json',
                                            'observability.png'], 'the files written')
    from vantage.assistant import tools as AT
    r.expect('get_calibration_observability' in AT.BY_NAME, True, 'the assistant tool')
    noise = (lambda ln: not ln.startswith(('This plugin does not', '[ WARN')))

    # ---- the window: the data rig
    w = stage._rig_window()
    r.win = w
    vp = stage.vp
    vp.cam.az, vp.cam.el = -38.0, 34.0
    w.viewarea.frame_all()
    stage.settle(2)
    d0 = vp.cam.dist
    r.cur = np.array([1150.0, 560.0])
    r.cursor_on = False
    r.expect(len(w.store.project.targets), 8, 'the planned boards')

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

    def row_of(t, text):
        return next(i for i in range(t.rowCount()) if t.item(i, 0) is not None and t.item(i, 0).text() == text)

    def click_tab(tabs, title):
        bar = tabs.tabBar()
        i = next(k for k in range(tabs.count()) if tabs.tabText(k).startswith(title))
        r.click(bar, QPointF(bar.tabRect(i).center()), 0.5, real=False)
        tabs.setCurrentIndex(i)
        r.hold(0.2)

    def show(widget):
        """Scroll the scroll area holding a widget so that the widget is in view."""
        from PyQt6.QtWidgets import QScrollArea
        a = widget.parentWidget()
        while a is not None and not isinstance(a, QScrollArea):
            a = a.parentWidget()
        if a is not None:
            a.ensureWidgetVisible(widget, 20, 40)
            r.tick(1)

    def calm_wait(cond, timeout=120, hz=4.0):
        """Record in real time until cond(), composing a new frame only hz times a second (the frames in between
        repeat it): the recorder leaves the computer to the study, which otherwise runs about 3x slower than it does
        for a user while frames are grabbed at twice the pixels. The video keeps pace with the wall clock."""
        import time
        t0, f0 = time.time(), r.f
        img, last = None, 0.0
        while not cond():
            now = time.time()
            if now - t0 > timeout:
                raise TimeoutError(f'calm_wait: still waiting after {timeout} s')
            r.app.processEvents()
            if img is None or now - last >= 1.0 / hz:
                r._advance()
                img = r.compose()
                last = now
            want = int((time.time() - t0) * r.fps)
            while r.f - f0 < want:
                r._write(img)
                r.f += 1
            time.sleep(0.01)
        r.events.append({'kind': 'wait', 'start': round(f0 / r.fps, 3), 'video_s': round((r.f - f0) / r.fps, 3),
                         'wall_s': round(time.time() - t0, 3), 'speed': 1})
        r.tick()

    def fresh(timeout=120):
        """Wait until the study shows the current inputs. The page re-runs by itself 0.4 s after an edit when its last
        run took under 3 s (about 2 s on the data rig); recording at twice the pixels slows the study past that, so
        when it has not started after 1.2 s, start the same quiet re-run the page's timer would."""
        t_try = r.now() + 1.2
        calm_wait(lambda: pg.busy or (pg.result is not None and not pg.stale) or r.now() >= t_try, timeout=20)
        for _ in range(5):
            if pg.result is not None and not pg.busy and not pg.stale:
                return
            if not pg.busy:
                print('re-run started by the scene (the last run took over 3 s while recording)', flush=True)
                pg.run(quiet=True)
            calm_wait(lambda: not pg.busy, timeout=timeout)
        raise RuntimeError('the study did not catch up with the edits')

    def undo(n=1):
        r.key_combo('Ctrl+Z', dur=1.2 + 0.4 * n)
        for _ in range(n):
            w.actions_by_id['undo'].trigger()
            r.hold(0.35)

    # ---- 0: the problem, over the 3D view with the boards shown
    r.lower_third()
    r.say(0)
    r.hold(0.6)
    r.cursor_on = True
    r.menu('View', 'Overlays', 'Calibration targets')
    r.cursor_on = False
    stage.fly(r, vp, az=-20.0, el=48.0, dist=d0 * 0.95, dur=2.5)
    stage.orbit(r, vp, 30, max(2.5, r.speech_left() - 0.2), d_el=-4)
    r.finish_speech(0.1)

    # ---- 1: the page
    r.say(1)
    r.cursor_on = True
    r.hold(0.5)
    r.menu('Calibrate', 'Calibration observability')
    pg = w.page('observability')
    r.crossfade(0.4)
    calm_wait(lambda: pg.result is not None and not pg.busy, timeout=90)
    r.tick(2)
    r.finish_speech(0.2)
    res = pg.result
    cams = pg.cams

    # ---- 2, 3: the method and its settings
    r.expect(pg.fields['pixel_sigma'].value(), 0.3, 'corner noise')
    r.expect((pg.fields['board_sigma_mm'].value(), pg.fields['board_sigma_deg'].value()), (20.0, 1.0), 'placement')
    r.expect((pg.fields['need_deg'].value(), pg.fields['need_mm'].value()), (0.1, 10.0), 'the requirement')
    r.say(2)
    lf = r.wrect(pg.fields['pixel_sigma'])
    r.zoom(QRectF(lf.x() - 140, lf.y() - 120, 390, 420), 1.0)
    r.until_word(2, 'zero point three')
    r.callout(grown(pg.fields['pixel_sigma'], 128), '', dur=max(1.6, r.speech_left() + 0.3), pad=2, dim=0.25)
    r.finish_speech(0.2)
    r.say(3)
    r.callout(rect_u(pg.toggles['boards_placed'], pg.fields['board_sigma_deg']), '', dur=3.6, pad=3, dim=0.25)
    r.until_word(3, 'The requirement')
    r.callout(grown(rect_u(pg.fields['need_deg'], pg.fields['need_mm']), 128), '', dur=max(1.6, r.speech_left() + 0.3),
              pad=2, dim=0.25)
    r.finish_speech(0.2)
    r.zoom(None, 0.8)

    # ---- 4: the plan
    plan = pg.plan
    m = res._model

    def board_box(name, rad=34):
        def f():
            if not plan.isVisible() or plan._tf is None:
                return None
            b = next(b for b in m.boards if b.t.name == name)
            q_ = plan.w2s(float(b.centre[0]), float(b.centre[1]))
            return r.local_rect(plan, QRectF(q_.x() - rad, q_.y() - rad, 2 * rad, 2 * rad))()
        return f
    r.say(4)
    pr = r.wrect(plan)
    r.zoom(QRectF(pr.x(), pr.y() + pr.height() * 0.22, pr.width(), pr.height() * 0.62), 1.0)
    r.callout(board_box('T1'), '', dur=1.6, style='circle', dim=0.2)
    r.until_word(4, 'and each camera')
    pts = [c[0] for c in plan._cams]
    if pts:
        xs, ys = [p_.x() for p_ in pts], [p_.y() for p_ in pts]
        r.callout(r.local_rect(plan, QRectF(min(xs) - 20, min(ys) - 20, max(xs) - min(xs) + 40, max(ys) - min(ys) + 40)),
                  '', dur=max(1.8, r.speech_left() + 0.3), pad=4, dim=0.2)
    r.finish_speech(0.2)
    r.zoom(None, 0.8)

    # ---- 5: the table of 1σ
    iF = row_of(cams, 'F')
    r.expect((cams.item(iF, 2).text(), cams.item(iF, 8).text()), ('0.131', '1.31×'), "F's yaw")
    r.expect(all(0.125 <= float(cams.item(i, 2).text()) <= 0.135 for i in range(cams.rowCount())), True,
             'every yaw near 0.13°')
    r.expect((pg.tiles['worst'].value, pg.tiles['worst'].note), ('1.34×', 'QR · yaw'), 'the loosest')
    r.expect((pg.tiles['cams'].value, pg.tiles['cams'].unit), ('0', 'of 6'), 'cameras within')
    r.say(5)
    cr = r.wrect(cams)
    tl = r.wrect(pg.tiles['cams'])
    r.zoom(QRectF(cr.x(), tl.y() - 6, cr.width(), cr.y() + cams.rowHeight(0) * 7 + 30 - tl.y()), 1.0)
    r.callout(span(cams, 0, 2, 5, 7), '', dur=3.2, pad=3, dim=0.25)
    r.until_word(5, 'Every camera')
    r.callout(span(cams, 0, 2, 5, 2), '', dur=2.2, pad=3, dim=0.25)
    r.until_word(5, 'one point three times')
    r.callout(pg.tiles['worst'], '', dur=max(1.6, r.speech_left() + 0.3), pad=3, dim=0.25)
    r.finish_speech(0.2)

    # ---- 6: F picked
    uidF = next(c['uid'] for c in res.cameras if c['tag'] == 'F')
    db = pg.det_boards
    r.say(6)
    rr = cams.visualRect(cams.model().index(iF, 0))
    r.click(cams.viewport(), QPointF(rr.center()), 0.6, real=False)
    pg.pick(uidF)
    r.tick(2)
    r.expect(cams.item(iF, 1).text(), 'T2 T3 T1 · 75', "F's boards")
    r.expect([(db.item(i, 0).text(), db.item(i, 4).text(), db.item(i, 5).text()) for i in range(db.rowCount())],
             [('T2', '34 %', '1.58×'), ('T3', '34 %', '1.58×'), ('T1', '31 %', '1.44×')], "F's boards' shares")
    r.expect(pg.det_weak.text(), 'Weakest direction (1σ): y (left) +8.5 mm · condition number 2,108',
             "F's weakest direction")
    dw = r.wrect(pg.det_head.parentWidget())
    r.zoom(QRectF(dw.x(), dw.y(), dw.width(), 200), 1.0)
    r.callout(span(db, 0, 0, 2, 1), '', dur=2.4, pad=3, dim=0.25)
    r.until_word(6, 'each board carrying')
    r.callout(span(db, 0, 4, 2, 4), '', dur=1.6, pad=3, dim=0.25)
    r.until_word(6, 'without T one')
    r.callout(span(db, 2, 0, 2, 5), '', dur=1.8, pad=3, dim=0.25)
    r.until_word(6, 'Its weakest')
    r.callout(pg.det_weak, '', dur=max(1.6, r.speech_left() + 0.3), pad=3, dim=0.25)
    r.finish_speech(0.2)

    # ---- 7: the reason
    why = pg._why[0]
    r.expect(why.text.text().startswith('yaw rests on where the boards lie (placed to ±20 mm, ±1°): the corners alone '
                                        'give 0.01°.'), True, 'the reason')
    r.expect(cams.item(iF, 2).toolTip(), 'yaw: ±0.131 °; from the corners alone (boards exact) ±0.012', 'corners alone')
    r.say(7)
    wr = r.wrect(why)
    r.zoom(QRectF(dw.x(), wr.y() - 80, dw.width(), 220), 0.8)
    r.callout(why, '', dur=max(3.0, r.speech_left() + 0.3), pad=3, dim=0.2)
    r.finish_speech(0.2)

    # ---- 8: the suggestions
    rows_s = pg._sugg_buttons
    texts = [next(l for l in row.findChildren(type(pg.det_weak)) if l.text()).text() for row in rows_s]
    r.expect(texts[0].startswith("Lay the boards to ±5 mm and ±0.25° (a jig, or measure where they lie): F's worst axis "
                                 "1.3× → 0.5× the requirement"), True, 'the placement suggestion')
    r.expect(texts[1].startswith("Add a 6 × 6 board in F's view: F's worst axis 1.3× → 1.2× the requirement"), True,
             'the board suggestion')
    r.say(8)
    r.zoom(QRectF(dw.x(), wr.y() - 40, dw.width(), 200), 0.8)
    r.until_word(8, "Another board")
    r.move_to(rows_s[1], QPointF(rows_s[1].width() * 0.4, rows_s[1].height() / 2), 0.6)
    pg._hover_sugg(1)
    r.callout(rows_s[1], '', dur=1.6, pad=3, dim=0.2)
    r.zoom(QRectF(pr.x() + pr.width() * 0.35, pr.y() + pr.height() * 0.25, pr.width() * 0.65, pr.height() * 0.55),
           0.8)
    r.hold(0.8)
    pg._hover_sugg(-1)
    r.until_word(8, 'Laying the boards')
    r.zoom(QRectF(dw.x(), wr.y() - 40, dw.width(), 200), 0.8)
    r.callout(rows_s[0], '', dur=max(1.6, r.speech_left() + 0.3), pad=3, dim=0.2)
    r.finish_speech(0.2)

    # ---- 9: Use ±5 mm, and undo
    r.say(9)
    btn = rows_s[0].button
    before = pg.result
    r.click(btn, None, 0.6)
    fresh()
    r.tick(2)
    r.expect((pg.tiles['cams'].value, pg.tiles['cams'].unit), ('6', 'of 6'), 'all within with ±5 mm')
    r.expect(pg.tiles['worst'].value, '0.58×', 'the loosest with ±5 mm')
    r.zoom(QRectF(cr.x(), tl.y() - 6, cr.width(), cr.y() + cams.rowHeight(0) * 7 + 30 - tl.y()), 0.8)
    r.until_word(9, 'all six')
    r.figure('Cameras within', 6, 'of 6', 0, dur=2.4, src=pg.tiles['cams'])
    r.until_word(9, 'the loosest')
    r.callout(pg.tiles['worst'], '', dur=1.6, pad=3, dim=0.2)
    r.until_word(9, 'Undo')
    before = pg.result
    undo()
    fresh()
    r.expect(pg.tiles['worst'].value, '1.34×', 'back to the plan')
    r.finish_speech(0.2)
    r.zoom(None, 0.7, wait=False)

    # ---- 10: camera pairs
    pt = pg.pairs
    ip = row_of(pt, 'F–ML')
    r.expect((pt.item(ip, 1).text(), pt.item(ip, 4).text(), pt.item(ip, 8).text()), ('T2', '0.050', '0.50×'),
             'the F–ML pair')
    r.say(10)
    click_tab(pg.tabs, 'Pairs')
    ptr = r.wrect(pt)
    r.zoom(QRectF(ptr.x(), ptr.y() - 30, ptr.width(), ptr.height() + 30), 1.0)
    r.callout(span(pt, ip, 0, ip, 1), '', dur=2.4, pad=3, dim=0.2)
    r.until_word(10, 'zero point zero five')
    r.callout(span(pt, ip, 4, ip, 4), '', dur=1.8, pad=3, dim=0.2)
    r.until_word(10, 'half the requirement')
    r.callout(span(pt, ip, 8, ip, 8), '', dur=max(1.6, r.speech_left() + 0.3), pad=3, dim=0.2)
    r.finish_speech(0.2)

    # ---- 11, 12: lidar–camera pairs
    lt = pg.lidars
    r.say(11)
    click_tab(pg.tabs, 'Lidar–camera')
    il = row_of(lt, 'F–L1')
    r.expect((lt.item(il, 7).text(), lt.item(il, 8).text()), ('16.4', '1.64×'), 'F–L1 z')
    r.expect((pg.tiles['lidar'].value, pg.tiles['lidar'].unit), ('0', 'of 6'), 'lidar pairs within')
    tip = lt.item(il, 0).toolTip()
    r.expect("The camera's corners limit it: where F puts each board (its distance above all) is the loose part." in tip,
             True, 'what limits F–L1')
    ltr = r.wrect(lt)
    r.zoom(QRectF(ltr.x(), ltr.y() - 30, ltr.width(), ltr.height() + 30), 0.8)
    r.until_word(11, 'For F and L one')
    r.callout(span(lt, il, 0, il, 8), '', dur=1.8, pad=3, dim=0.2)
    r.until_word(11, 'sixteen point four')
    r.figure("Lidar's height, 1σ", 16.4, 'mm', 1, dur=max(2.4, r.speech_left() + 0.6), src=H.cell(r, lt, il, 7),
             note='1.64× its tolerance')
    r.finish_speech(0.2)
    r.say(12)
    fl = next(f for f in pg.findings if f.text.text().startswith('F–L1: '))
    r.expect(fl.text.text().startswith("F–L1: the planned lidar–camera boards pin the lidar's z to 1.6× the tolerance. "
                                       "The camera's corners limit it"), True, 'the F–L1 finding')
    click_tab(pg.tabs, 'Findings')
    r.tick(2)
    show(fl)
    flr = r.wrect(fl)
    r.zoom(QRectF(flr.x() - 6, flr.y() - 60, flr.width() + 12, flr.height() + 120), 0.8)
    r.callout(fl, '', dur=3.0, pad=3, dim=0.2)
    r.until_word(12, 'This count leaves out')
    r.zoom(None, 0.7)
    r.callout(pg.assumptions, '', dur=max(1.8, r.speech_left() + 0.3), pad=3, dim=0.3)
    r.finish_speech(0.2)

    # ---- 13, 14: corner reflectors for the radars
    rt = pg.radars
    r.say(13)
    rf = pg.refl
    show(rf)
    rfr = r.wrect(rf)
    r.zoom(QRectF(rfr.x() - 6, rfr.y() - 40, 360, 160), 0.8)
    for _ in range(2):
        r.click(pg.b_add_refl, None, 0.5)
        r.hold(0.4)
    before = pg.result
    it = rf.item(1, 2)
    rect = rf.visualItemRect(it)
    r.move_to(rf.viewport(), QPointF(rect.center()), 0.5)
    r.ripple()
    rf.editItem(it)
    r.tick(3)
    from PyQt6.QtWidgets import QLineEdit
    from PyQt6.QtTest import QTest
    from PyQt6.QtCore import Qt
    ed = next((e for e in rf.viewport().findChildren(QLineEdit) if e.isVisible()), None)
    if ed is None:
        it.setText('3.0')
    else:
        ed.selectAll()
        r.hold(0.2)
        QTest.keyClicks(ed, '3')
        r.hold(0.3)
        QTest.keyClick(ed, Qt.Key.Key_Return)
    fresh()
    r.expect([rf.item(i, 2).text() for i in range(rf.rowCount())], ['0.00', '3.00'], "the reflectors' y")
    click_tab(pg.tabs, 'Radars')
    ir = 0
    r.expect((rt.item(ir, 0).text(), rt.item(ir, 1).text(), rt.item(ir, 2).text(), rt.item(ir, 5).text()),
             ('RD1 · Long-range radar · typical', 'CR1 CR2', '2.535', '18.13×'), "RD1 on two reflectors")
    rtr = r.wrect(rt)
    r.zoom(QRectF(rtr.x(), rtr.y() - 30, rtr.width(), rtr.height() + 30), 0.8)
    r.until_word(13, 'two point five')
    r.callout(span(rt, ir, 0, ir, 5), '', dur=max(1.8, r.speech_left() + 0.3), pad=3, dim=0.2)
    r.finish_speech(0.2)
    r.say(14)
    fr_ = next(f for f in pg.findings if f.text.text().startswith('RD1: '))
    r.expect(fr_.text.text(), 'RD1: the reflectors pin its y to 18.1× the requirement. Add a reflector farther out or at '
                              'another bearing.', "RD1's finding")
    click_tab(pg.tabs, 'Findings')
    r.tick(2)
    show(fr_)
    frr = r.wrect(fr_)
    r.zoom(QRectF(frr.x() - 6, frr.y() - 60, frr.width() + 12, frr.height() + 120), 0.8)
    r.callout(fr_, '', dur=2.4, pad=3, dim=0.2)
    r.until_word(14, 'enter your')
    r.zoom(None, 0.6, wait=False)
    before = pg.result
    undo(3)
    fresh()
    r.expect(len(pg.settings.reflectors), 0, 'the reflectors removed')
    r.finish_speech(0.2)

    # ---- 15, 16: Verify
    r.say(15)
    show(pg.b_verify)
    vr_ = r.wrect(pg.b_verify)
    r.zoom(QRectF(vr_.x() - 200, vr_.y() - 140, 330, 190), 0.8)
    r.click(pg.b_verify, None, 0.6)
    t_cams = r.wrect(pg.tiles['cams'])
    r.zoom(QRectF(t_cams.x() - 6, 28, 1920 - t_cams.x() + 6, t_cams.bottom() - 18), 0.8)
    done_mc = (lambda: not pg.mc_busy and pg.result is not None and pg.result.mc is not None)
    t_end = r.speech_end + 0.3                  # real time while the line is spoken, then a marked time-lapse
    calm_wait(lambda: done_mc() or r.now() >= t_end, timeout=600)
    if not done_mc():
        r.wait_for(done_mc, timeout=600, speed=4.0)
    mc = pg.result.mc
    r.expect((mc['agree'], mc['trials']), (True, 30), 'the Monte Carlo verdict')
    r.expect(pg.tiles['mc'].value, 'Holds', 'the Monte Carlo tile')
    note = pg.mc_note.text()
    r.expect('The predictions hold: every σ within ×1.59' in note, True, 'the band')
    r.expect(note.endswith('The joint solve stopped short of its optimum in 0 of 30 trials.'), True, 'no stall')
    mt = pg.mc_table
    ratios = [float(mt.item(i, 5).text()) for i in range(mt.rowCount())]
    r.expect((f'{min(ratios):.2f}', f'{max(ratios):.2f}'), ('0.72', '1.30'), 'the ratio range')
    r.finish_speech(0.2)
    r.say(16)
    click_tab(pg.tabs, 'Verify')
    mtr = r.wrect(mt)
    r.zoom(QRectF(mtr.x() - 6, mtr.y() - 60, mtr.width() + 12, mtr.height() + 60), 1.0)
    r.callout(span(mt, 0, 2, min(8, mt.rowCount() - 1), 5), '', dur=2.6, pad=3, dim=0.2)
    r.until_word(16, 'within the')
    r.callout(pg.mc_note, '', dur=max(2.4, r.speech_left() + 0.3), pad=3, dim=0.2)
    r.finish_speech(0.2)
    r.zoom(None, 0.7)

    # ---- 17, 18: the fusion maps
    fm = pg.fusion_maps
    r.say(17)
    click_tab(pg.view_tabs, 'Fusion')
    r.tick(2)
    mF = next(x for x in fm.maps if x[0] == 'F')
    r.expect(f'{100 * mF[3]:.0f} % with lidar · 1 point per {1 / mF[4]:.0f} px', '22 % with lidar · 1 point per 37 px',
             "F's fusion map")
    rect0 = next(x[0] for x in fm._rects if x[1] == 'F') if fm._rects else None
    fmr = r.wrect(fm)
    r.zoom(QRectF(fmr.x(), fmr.y(), fmr.width(), fmr.height()), 1.0)
    r.until_word(17, 'F: ')
    if rect0 is not None:
        r.callout(r.local_rect(fm, QRectF(rect0.x(), rect0.y() - 18, rect0.width(), rect0.height() + 18)), '',
                  dur=max(2.4, r.speech_left() + 0.3), pad=3, dim=0.25)
    r.finish_speech(0.2)
    r.say(18)
    if rect0 is not None:
        r.zoom(r.local_rect(fm, QRectF(rect0.x() - 10, rect0.y() - 24, rect0.width() + 20, rect0.height() + 34)), 1.0)
        r.callout(r.local_rect(fm, QRectF(rect0.x(), rect0.y() + rect0.height() * 0.45, rect0.width(),
                                          rect0.height() * 0.55)), '', dur=max(2.0, r.speech_left() + 0.3), pad=2,
                  dim=0.2)
    r.finish_speech(0.2)

    # ---- 19: the fusion table
    ft = pg.fusion_table

    def frow(cam, obj, rng):
        return next(i for i in range(ft.rowCount()) if (ft.item(i, 0).text(), ft.item(i, 1).text(),
                                                         ft.item(i, 2).text()) == (cam, obj, rng))
    i20, i50 = frow('F', 'Pedestrian', '20'), frow('F', 'Pedestrian', '50')
    r.expect((ft.item(i20, 3).text(), ft.item(i20, 4).text()), ('48', '13'), 'a pedestrian at 20 m')
    r.expect((ft.item(i50, 3).text(), ft.item(i50, 6).text()), ('8', '0 %'), 'a pedestrian at 50 m')
    r.say(19)
    ft.scrollToItem(ft.item(i20, 0))
    r.tick(2)
    ftr = r.wrect(ft)
    r.zoom(QRectF(ftr.x(), ftr.y() - 30, ftr.width(), ftr.height() + 30), 1.0)
    r.until_word(19, 'a pedestrian twenty')
    r.callout(span(ft, i20, 0, i20, 6), '', dur=2.4, pad=2, dim=0.2)
    r.until_word(19, 'at fifty')
    r.callout(span(ft, i50, 0, i50, 6), '', dur=max(1.8, r.speech_left() + 0.3), pad=2, dim=0.2)
    r.finish_speech(0.2)
    r.zoom(None, 0.7)

    # ---- 20, 21: issues and limits
    r.say(20)
    click_tab(pg.view_tabs, 'Plan')
    click_tab(pg.tabs, 'Findings')
    fh = r.wrect(pg.find_host.parentWidget())
    r.zoom(QRectF(fh.x(), fh.y() - 30, fh.width(), fh.height() + 30), 1.0)
    r.hold(max(1.0, r.speech_left() - 0.6))
    r.finish_speech(0.2)
    r.say(21)
    show(pg.assumptions)
    asr = r.wrect(pg.assumptions)
    r.zoom(QRectF(asr.x() - 6, asr.y() - 30, asr.width() + 12, min(asr.height(), 300) + 40), 1.0)
    r.callout(pg.assumptions, '', dur=max(2.5, r.speech_left() + 0.3), pad=3, dim=0.2)
    r.finish_speech(0.2)
    r.zoom(None, 0.7)

    # ---- 22: batch
    def term(cmd, code, lines):
        return [('cmd', cmd)] + H.timed_output(lines, speed=max(1.0, s_ob / 4.0), keep=noise, tones=True) + \
            [('wait', 0.4), ('cmd', 'echo $?'), ('out', str(code), 'warn' if code else 'ok'), ('wait', 0.3),
             ('cmd', 'ls obs/')] + H.timed_output(l_ls)
    r.say(22)
    sp = H.terminal(r, term('vantage data-rig.vantage --observability obs/ --verify 30', c_ob, l_ob),
                    title='~/projects')
    print(f'terminal 22: {sp:.1f} s against {r.speech_left():.1f} s; the run took {s_ob:.1f} s', flush=True)
    if s_ob / 4.0 > 1.05:
        r.timelapse(s_ob / 4.0)
    r.crossfade(0.5)
    r.hold(max(sp + 0.5, r.speech_left() + 0.2))
    r.timelapse(None)
    r.finish_speech(0.2)

    # ---- 23: the assistant; back to the plan
    r.say(23)
    r.card = None
    r.crossfade(0.6)
    r.cursor_on = False
    r.hold(max(2.0, r.speech_left() + 0.3))
    r.finish_speech(1.0)
