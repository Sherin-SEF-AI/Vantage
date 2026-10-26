"""27 Calibration targets: Calibrate ▸ Target planner on the sample (6-cam · v3): the floor plan by redundancy with the
eight boards, the board settings, the table of who sees each board, Check and Apply suggestions, a board dragged out
of a camera's view and back (Ctrl+Z), Auto-place, the boards in 3D, the Add target and Export menus.

Figures, read from the app while recording (r.expect): 8 boards, checkerboard 6 x 6 squares of 100 mm; a board counts
for a camera at >= 90 % in view (core.targets.evaluate_targets) and >= 3 px/cm (Density on board), >= 2 cameras per
board; T1 is seen by F alone and T8 by R alone; after Apply suggestions: "All 8 targets are seen by 2+ cameras."; T2
dragged forward is seen by F alone; Auto-place gives 6 boards, all seen by 2+ cameras.
"""

META = {
    'id': 'targets',
    'part': 4,
    'title': 'Calibration targets',
    'promise': 'Where the boards go on the floor, so every pair of cameras shares one.',
    'menu': 'Calibrate ▸ Target planner  ·  Auto-place  ·  Check',
}

LINES = [
    ("Calibrating a surround rig on the vehicle starts on the workshop floor: printed boards around the car, each seen "
     "by the cameras that share it. The target planner places them.",
     "Calibrating a surround rig on the vehicle starts on the workshop floor: printed boards around the car, each seen "
     "by the cameras that share it. The target planner places them."),
    ("The floor plan shades the ground by how many cameras see it, and shows the sample's eight boards: checkerboards "
     "of six by six squares, 100 millimetres each.",
     "The floor plan shades the ground by how many cameras see it, and shows the sample's eight boards: checkerboards "
     "of six by six squares, a hundred millimetres each."),
    ("A board counts for a camera when at least 90 per cent of it is in view, at 3 pixels per centimetre or more. Each "
     "board should be seen by two cameras, so the cameras can be chained pair by pair.",
     "A board counts for a camera when at least ninety per cent of it is in view, at three pixels per centimetre or "
     "more. Each board should be seen by two cameras, so the cameras can be chained, pair by pair."),
    ("The table lists each board, where it lies, and the cameras that see it. T1 in front and T8 behind are seen by "
     "one camera only, and are marked in red.",
     "The table lists each board, where it lies, and the cameras that see it. T one in front, and T eight behind, are "
     "seen by one camera only, and are marked in red."),
    ("Check looks for a better place for each: a spot that two neighbouring cameras both see well. The suggestions "
     "are drawn dashed on the plan.",
     "Check looks for a better place for each: a spot that two neighbouring cameras both see well. The suggestions "
     "are drawn dashed on the plan."),
    ("Apply suggestions moves both boards in one undo step: all eight are now seen by two cameras or more.",
     "Apply suggestions moves both boards, in one undo step: all eight are now seen by two cameras or more."),
    ("Boards can be dragged on the plan, and the check follows. Pull T2 forward, and the mirror camera no longer sees "
     "it well enough. Ctrl+Z puts it back.",
     "Boards can be dragged on the plan, and the check follows. Pull T two forward, and the mirror camera no longer "
     "sees it well enough. Control Z puts it back."),
    ("Auto-place starts from scratch: one board for each pair of neighbouring cameras, at the spot both see best. Six "
     "boards for this layout.",
     "Auto-place starts from scratch: one board for each pair of neighbouring cameras, at the spot both see best. Six "
     "boards, for this layout."),
    ("In 3D the boards lie on the floor around the car, each where the views of two neighbouring cameras overlap.",
     "In 3D, the boards lie on the floor around the car, each where the views of two neighbouring cameras overlap."),
    ("Boards can be checkerboards, ChArUco boards or AprilGrids. Export target positions writes them as CSV or JSON in "
     "the project's frame, to lay them out for the installation check.",
     "Boards can be checkerboards, charuco boards, or april grids. Export target positions writes them as C S V or "
     "Jason, in the project's frame, to lay them out for the installation check."),
]


def scene(stage, r):
    from PyQt6.QtCore import QPointF, QRectF
    import helpers_tourc as H

    w = stage.window()
    r.win = w
    st = w.store
    p = st.project
    r.expect(len(p.targets), 8, 'boards in the sample')
    ts = p.target_settings
    r.expect((ts.pattern, ts.rows, ts.cols, round(ts.square_mm)), ('checkerboard', 6, 6, 100), 'board pattern')
    r.expect((ts.cams_per_target, ts.min_density), (2, 3.0), 'cameras per board, density on board')
    vp = stage.vp

    # 0 - why: the boards; open the planner
    r.lower_third()
    r.say(0)
    r.cursor_on = False
    stage.orbit(r, vp, 20, 3.4, d_el=6)
    r.cursor_on = True
    r.until_word(0, 'The target planner')
    r.menu('Calibrate', 'Target planner')
    stage.settle(5)
    pg = w.page('targets')
    plan = pg.plan
    ip = w.inspector.pages['targets']
    r.crossfade(0.4)
    r.finish_speech(0.2)

    def board(name):
        t = next(t for t in st.project.targets if t.name == name)
        wd, ht = t.size_m
        a = plan.w2s(t.x - wd / 2, t.y + ht / 2)
        b = plan.w2s(t.x + wd / 2, t.y - ht / 2)
        return r.local_rect(plan, QRectF(a, b).normalized().adjusted(-10, -10, 10, 10))

    def legend():
        lr = plan.legend_rect
        return None if lr is None else QRectF(r.wpos(plan, lr.topLeft()), lr.size())

    # 1 - the plan: redundancy, the eight boards; their pattern
    r.say(1)
    r.move(r.wpos(plan, QPointF(plan.width() * 0.62, plan.height() * 0.3)), 0.9)
    r.callout(legend, 'Redundancy', dur=2.6, side='left')
    r.hold(2.8)
    insp = r.wrect(w.docks['inspector'])
    sect = H.union(r.wrect(ip.pattern.parentWidget()), r.wrect(ip.dob.parentWidget()))
    r.zoom(QRectF(insp.x() - 40, sect.y() - 60, insp.width() + 60, sect.height() + 150), 1.0)
    r.until_word(1, 'checkerboards')
    r.callout(lambda: H.union(r.wrect(ip.pattern.parentWidget()), r.wrect(ip.square.parentWidget())), dur=max(2.2, r.speech_left()),
              pad=3)
    r.finish_speech(0.2)
    # 2 - when a board counts: in view, density; cameras per board
    r.say(2)
    r.until_word(2, 'at three pixels')
    r.callout(ip.dob.parentWidget(), dur=2.4, pad=3)
    r.until_word(2, 'Each board should')
    r.callout(ip.cpt.parentWidget(), dur=max(2.2, r.speech_left()), pad=3)
    r.finish_speech(0.2)
    # 3 - the table; T1 and T8 seen by one camera
    t = ip.t
    seen = {t.item(i, 0).text(): t.item(i, 3).text() for i in range(t.rowCount())}
    print('seen by:', seen, '| status:', ip.status.text.text(), flush=True)
    r.expect(seen['T1'], 'F', 'T1 seen by')
    r.expect(seen['T8'], 'R', 'T8 seen by')
    r.say(3)
    tab = H.union(r.wrect(t), r.wrect(ip.status))
    r.zoom(QRectF(insp.x() - 40, tab.y() - 30, insp.width() + 60, tab.height() + 60), 1.0)
    r.until_word(3, 'T one in front')
    r.callout(H.cell(r, t, 0), dur=1.8, pad=2)
    r.until_word(3, 'T eight behind')
    r.callout(H.cell(r, t, t.rowCount() - 1), dur=1.8, pad=2)
    r.until_word(3, 'marked in red')
    r.zoom(None, 0.8)
    r.callout(board('T1'), dur=max(1.6, r.speech_left() + 0.4), style='box', dim=0.25)
    r.finish_speech(0.2)
    # 4 - Check: dashed suggestions
    b_check = H.button(ip, 'Check')
    r.say(4)
    r.click(b_check, real=False)
    b_check.click()
    w.inspector.scroll.verticalScrollBar().setValue(0)
    stage.settle(5)
    r.expect(len(ip.sugg), 2, 'suggestions')
    print('suggestions:', ip.sugg, flush=True)
    pr = r.wrect(plan)
    r.zoom(QRectF(pr.x() + pr.width() * 0.12, pr.y() + pr.height() * 0.1, pr.width() * 0.76, pr.height() * 0.8), 1.0)
    r.hold(max(1.5, r.speech_left() - 1.2))
    r.zoom(None, 0.8)
    r.finish_speech(0.2)
    # 5 - Apply suggestions: all eight seen by two
    r.say(5)
    r.click(ip.b_apply, real=False)
    ip.b_apply.click()                                # no focus change: the Inspector keeps its scroll position
    w.inspector.scroll.verticalScrollBar().setValue(0)
    stage.wait_analysis(r)
    stage.settle(3)
    r.hold(0.3)
    msg = ip.status.text.text()
    print('status after apply:', msg, flush=True)
    r.expect(msg, 'All 8 targets are seen by 2+ cameras.', 'status after Apply')
    r.zoom(QRectF(insp.x() - 40, tab.y() - 30, insp.width() + 60, tab.height() + 60), 0.9)
    r.callout(ip.status, dur=max(2.0, r.speech_left() + 0.3), pad=4)
    r.finish_speech(0.3)
    r.zoom(None, 0.8)
    # 6 - drag T2 forward: ML loses it; Ctrl+Z
    t2 = next(x for x in st.project.targets if x.name == 'T2')
    x0, y0 = t2.x, t2.y
    r.say(6)
    pr = r.wrect(plan)
    a = plan.w2s(x0, y0)
    b = plan.w2s(x0 + 0.62, y0)
    c = r.wpos(plan, a)
    r.zoom(QRectF(c.x() - 420, c.y() - 240, 840, 472), 1.0, wait=False)
    r.hold(0.4)
    H.drag(r, plan, a, b, 1.6)
    stage.wait_analysis(r)
    stage.settle(3)
    r.hold(0.3)
    row2 = next(i for i in range(t.rowCount()) if t.item(i, 0).text() == 'T2')
    r.expect(t.item(row2, 3).text(), 'F', 'T2 seen by after the drag')
    r.callout(board('T2'), dur=2.2, dim=0.2)
    r.hold(1.6)
    r.until_word(6, 'Control Z')
    r.key_combo('Ctrl+Z', dur=1.4)
    r.hold(0.3)
    st.undo.undo()
    stage.wait_analysis(r)
    stage.settle(3)
    r.expect(t.item(row2, 3).text(), 'F ML', 'T2 seen by after the undo')
    r.finish_speech(0.3)
    r.zoom(None, 0.8)
    # 7 - Auto-place: six boards
    b_auto = H.button(ip, 'Auto-place')
    r.say(7)
    r.click(b_auto, real=False)
    b_auto.click()
    w.inspector.scroll.verticalScrollBar().setValue(0)
    stage.wait_analysis(r)
    stage.settle(3)
    r.expect(len(st.project.targets), 6, 'boards after Auto-place')
    msg = ip.status.text.text()
    print('status after auto-place:', msg, flush=True)
    r.expect(msg, 'All 6 targets are seen by 2+ cameras.', 'status after Auto-place')
    r.hold(1.0)
    r.callout(ip.t, dur=max(2.2, r.speech_left() + 0.3), pad=4, dim=0.25)
    r.finish_speech(0.2)
    # 8 - the boards in 3D
    r.say(8)
    r.toolbar('view_persp')
    stage.settle(5)
    vp = stage.vp
    w.inspector.scroll.verticalScrollBar().setValue(0)
    vp.cam.az, vp.cam.el = -62.0, 46.0
    w.viewarea.frame_all()
    vp.cam.dist *= 0.86
    vp.update()
    r.crossfade(0.5)
    r.cursor_on = False
    r.zoom(w.viewarea, 1.2, wait=False, margin=-30)
    stage.orbit(r, vp, 50, max(4.0, r.speech_left() + 0.3), d_el=-4)
    r.finish_speech(0.1)
    # 9 - patterns; export positions
    r.say(9)
    r.zoom(None, 0.8, wait=False)
    r.cursor_on = True
    r.peek_menu('Calibrate', 'Add target', 'Checkerboard', hold=1.8)
    r.until_word(9, 'Export target positions')
    r.peek_menu('Calibrate', 'Export target positions', hold=max(1.0, r.speech_left() - 1.2))
    r.cursor_on = False
    r.finish_speech(1.0)
