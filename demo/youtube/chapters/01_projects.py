"""01 Projects and the window: the start window, the Tigor sample, the window's parts, selecting a camera, variants,
the command palette (Ctrl+K) running the light theme, workspaces, history and shortcuts, the .vantage file.

Set-up (not filmed): two projects saved in the throw-away home folder (~/Documents/Vantage) so the start window lists
real recent files, shown from ~. Figures: the sample has 5 variants; the data rig carries a roof lidar and 5 radars;
6 cameras, a Jetson AGX Orin (read from the project while recording, r.expect).
"""

META = {
    'id': 'projects',
    'part': 1,
    'title': 'Projects and the window',
    'promise': 'The start window, the sample car, and where everything lives.',
    'menu': 'File  ·  Window ▸ Workspaces  ·  Ctrl+K',
}

LINES = [
    ("Vantage opens on the start window: recent projects on the left, each with a thumbnail and what it holds; a "
     "blank project, Open, and templates on the right.",
     "Vantage opens on the start window: recent projects on the left, each with a thumbnail and what it holds. A "
     "blank project, open, and templates, on the right."),
    ("Templates bring a placeholder body, mount zones and a camera preset. The Tata Tigor EV is a real car, measured "
     "from a 19-frame studio photo spin.",
     "Templates bring a placeholder body, mount zones, and a camera preset. The Tata Tee-gor E V is a real car, "
     "measured from a nineteen frame studio photo spin."),
    ("It carries six e-con Systems STURDeCAM31 cameras with fisheye lenses, wired to a Jetson AGX Orin.",
     "It carries six e-con Systems Sturdy-Cam thirty-one cameras, with fisheye lenses, wired to a Jetson A G X "
     "Orin."),
    ("The Outliner lists everything in the project, the Inspector edits what is selected, the 3D view sits in the "
     "middle, and the analysis panels below update as you work.",
     "The outliner lists everything in the project, the inspector edits what is selected, the 3D view sits in the "
     "middle, and the analysis panels below update as you work."),
    ("Select a camera, in the view or the Outliner, and the Inspector shows its module, lens, pose and mount, while "
     "the camera view shows what it sees.",
     "Select a camera, in the view or the outliner, and the inspector shows its module, lens, pose, and mount, while "
     "the camera view shows what it sees."),
    ("A project keeps alternative layouts as variants. The sample has five, from four cameras to a full data rig with "
     "a roof lidar and five radars.",
     "A project keeps alternative layouts as variants. The sample has five, from four cameras, to a full data rig, "
     "with a roof lidar and five radars."),
    ("Ctrl+K opens the command palette: every command and every object, found by name.",
     "Control K opens the command palette: every command, and every object, found by name."),
    ("Type a few letters and press Enter. Here, the light theme.",
     "Type a few letters, and press enter. Here, the light theme."),
    ("Dark or light, with a UI scale in Preferences for any screen. Back to dark from the View menu.",
     "Dark or light, with a U I scale in preferences, for any screen. Back to dark, from the view menu."),
    ("Workspaces rearrange the window for a task: Analysis splits the view in four, with the Coverage panel open; "
     "Placement brings back the single view. Your own layouts can be saved too.",
     "Workspaces rearrange the window for a task. Analysis splits the view in four, with the coverage panel open. "
     "Placement brings back the single view. Your own layouts can be saved too."),
    ("Every edit can be undone: Ctrl+Z puts the six-camera layout back, and the full history has its own panel. "
     "Press the question mark for every keyboard shortcut.",
     "Every edit can be undone: control Z puts the six camera layout back, and the full history has its own panel. "
     "Press the question mark, for every keyboard shortcut."),
    ("A project is one .vantage file, with the mesh, every variant and a thumbnail inside, ready to share.",
     "A project is one dot vantage file, with the mesh, every variant, and a thumbnail inside, ready to share."),
]


def scene(stage, r):
    from PyQt6.QtCore import QPointF, QRectF, Qt
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QLabel, QPushButton

    # ---- set-up, not filmed: two saved projects with thumbnails, for the recent list
    w0 = stage.window()
    stage.settle(5)
    vp0 = stage.vp
    vp0.cam.az, vp0.cam.el = -38.0, 22.0
    w0.viewarea.frame_all()
    stage.settle(2)
    p1 = stage.save_project('tigor-ev-svm', thumb_from=vp0)

    def rig(p):
        p.active_uid = next(v.uid for v in p.variants if v.sensors)
    w0.switch_variant(next(v.uid for v in w0.store.project.variants if v.sensors))
    stage.settle(10)
    p2 = stage.save_project('tigor-data-rig', edit=rig, thumb_from=vp0)
    stage.close_window()

    # ---- 0: the start window
    sw = stage.start_window([p2, p1])
    stage.film_centred(r, sw)
    r.zoom(sw, 0, margin=36)                          # the start window fills the frame
    r.cursor_on = True
    r.cur = __import__('numpy').array([1500.0, 860.0])
    r.lower_third(dur=4.2)
    r.hold(0.3)
    r.say(0)
    item = sw.list.visualItemRect(sw.list.item(0))
    r.move(r.wpos(sw.list.viewport(), QPointF(item.center().x() - 120, item.center().y())), 1.0)
    r.callout(r.local_rect(sw.list.viewport(), QRectF(sw.list.visualItemRect(sw.list.item(0)).united(
        sw.list.visualItemRect(sw.list.item(1))))), dur=2.6, pad=2, dim=0.25)
    r.until_word(0, 'A blank project')
    tigor = next(b for b in sw.findChildren(QPushButton) if any('Tigor' in lb.text() for lb in b.findChildren(QLabel)))
    rows = [b for b in sw.findChildren(QPushButton) if b.property('role') == 'ghost' and b.findChildren(QLabel)]
    top = min(rows, key=lambda b: r.wpos(b).y())
    bottom_ = max(rows, key=lambda b: r.wpos(b).y())
    r.move(r.wpos(top), 0.9)
    r.callout(lambda: r.wrect(top).united(r.wrect(bottom_)), dur=max(2.2, r.speech_left() + 1.0), pad=4, dim=0.25)
    r.finish_speech(0.2)
    # 1: templates; the Tigor
    r.say(1)
    r.hold(1.0)
    r.move(r.wpos(tigor), 1.0)
    r.until_word(1, 'The Tata')
    r.callout(tigor, '', dur=max(2.0, r.speech_left()), pad=3, dim=0.25)
    r.finish_speech(0.1)
    r.ripple()
    r.hold(0.3)

    # ---- 2: the main window with the sample
    sw.hide()
    w = stage.window()
    stage.film_window(r, w)
    st = w.store
    r.expect(len(st.variant.cameras), 6, 'cameras')
    r.expect(st.project.ecu.name, 'Jetson AGX Orin', 'ECU')
    r.zoom(None, 0)
    r.crossfade(0.7)
    r.cursor_on = False
    vp = stage.vp
    r.say(2)
    stage.orbit(r, vp, 55, r.line(2) + 0.6, d_el=4)
    # 3: the window's parts
    r.say(3)
    r.callout(w.docks['outliner'], 'Outliner', dur=2.4, pad=-2, side='right')
    r.until_word(3, 'the inspector')
    r.callout(w.docks['inspector'], 'Inspector', dur=2.2, pad=-2, side='left')
    r.until_word(3, 'the 3D view')
    r.callout(w.viewarea, '3D view', dur=2.0, pad=-4, side='auto')
    r.until_word(3, 'the analysis panels')
    r.callout(w.docks['bottom'], 'Analysis panels', dur=max(2.0, r.speech_left() + 0.6), pad=-2, side='above')
    r.finish_speech(0.2)
    # 4: select a camera
    r.cursor_on = True
    r.say(4)
    ml_row = r.tree_item(w.outliner, 'ML')
    rr = ml_row()
    if rr is not None:
        r.move(QPointF(rr.x() + 60, rr.center().y()), 0.9)
        r.ripple()
    st.select_cams([st.variant.by_tag('ML').uid])
    stage.settle(3)
    r.hold(0.6)
    insp = r.wrect(w.docks['inspector'])
    r.zoom(QRectF(insp.x() - 300, insp.y(), insp.width() + 300, 560), 1.0, wait=False)
    r.move(QPointF(insp.x() + 160, insp.y() + 200), 0.9)
    r.until_word(4, 'while the camera view')
    r.zoom(None, 0.8)
    r.callout(w.docks['camview'], 'Camera view', dur=max(2.2, r.speech_left() + 0.5), pad=-2, side='left')
    r.finish_speech(0.2)
    # 5: variants: the data rig
    r.say(5)
    variants = st.project.variants
    r.expect(len(variants), 5, 'variants')
    data = next(v for v in variants if v.sensors)
    r.expect(sum(1 for s in data.sensors if s.kind == 'radar'), 5, 'radars')
    r.expect(sum(1 for s in data.sensors if s.kind == 'lidar'), 1, 'lidar')
    combo = w.variant_combo
    r.click(combo, None, 0.8, real=False, hold=0.1)
    combo.showPopup()
    r.hold(1.6)
    view = combo.view()
    i = combo.findData(data.uid)
    ir = view.visualRect(view.model().index(i, 0))
    r.move_to(view.viewport(), QPointF(ir.center().x() * 0.6, ir.center().y()), 0.6)
    view.setCurrentIndex(view.model().index(i, 0))
    r.hold(0.4)
    r.ripple()
    r.hold(0.15)
    combo.hidePopup()
    combo.setCurrentIndex(i)
    combo.activated.emit(i)
    stage.wait_analysis(r)
    r.cursor_on = False
    stage.fly(r, vp, az=-36.0, el=17.0, dist=vp.cam.dist * 1.1, dur=1.6)
    stage.orbit(r, vp, 40, max(2.0, r.speech_left() + 0.4), d_el=3)
    r.finish_speech(0.1)
    # 6-7: the command palette runs the light theme
    from vantage.ui.dialogs.palette import CommandPalette
    r.say(6)
    r.key_combo('Ctrl+K', dur=1.6)
    r.hold(0.5)
    pal = CommandPalette(w.ctx, w, '')
    pal.show()
    r.cursor_on = False
    r.hold(0.6)
    r.zoom(QRectF(r.wrect(pal).x() - 60, r.wrect(pal).y() - 40, r.wrect(pal).width() + 120,
                  r.wrect(pal).height() + 80), 0.9, wait=False)
    r.finish_speech(0.1)
    r.say(7)
    r.type(pal.edit, 'light', cps=7)
    r.hold(0.7)
    r.key_combo('Enter', dur=1.0)
    r.hold(0.25)
    QTest.keyClick(pal.edit, Qt.Key.Key_Return)
    r.app.processEvents()
    if pal.isVisible():
        pal.hide()
    stage.settle(3)
    r.zoom(None, 0.8, wait=False)
    r.crossfade(0.4)
    r.hold(0.8)
    stage.orbit(r, vp, 30, max(1.6, r.speech_left() + 0.3), d_el=2)
    r.finish_speech(0.1)
    # 8: back to dark from the View menu
    r.say(8)
    r.cursor_on = True
    r.hold(1.6)
    r.until(r.at_word(8, 'Back to dark') - 0.9)
    r.menu('View', 'Theme', 'Dark')
    stage.settle(2)
    r.crossfade(0.4)
    r.hold(0.4)
    r.finish_speech(0.1)
    # 9: workspaces
    r.say(9)
    r.menu('Window', 'Workspaces', 'Analysis')
    stage.settle(5)
    r.crossfade(0.4)
    r.hold(0.6)
    r.until_word(9, 'Placement brings')
    r.menu('Window', 'Workspaces', 'Placement')
    stage.settle(3)
    r.crossfade(0.4)
    r.finish_speech(0.1)
    # 10: undo, the history, the shortcuts
    r.say(10)
    r.until_word(10, 'control Z')
    r.key_combo('Ctrl+Z', dur=1.4)
    r.hold(0.3)
    st.undo.undo()
    stage.wait_analysis(r)
    r.expect(st.variant.name, '6-cam \u00b7 v3', 'the layout after the undo')
    r.crossfade(0.3)
    r.until_word(10, 'the full history')
    w.bottom.show_tab('history')
    stage.settle(1)
    hist = r.wrect(w.docks['bottom'])
    r.zoom(QRectF(hist.x(), hist.y() - 20, hist.width() * 0.5, hist.height() + 20), 0.9)
    r.hold(1.4)
    r.zoom(None, 0.7)
    w.bottom.show_tab('coverage')
    r.until_word(10, 'Press the question mark')
    from vantage.ui.dialogs.misc import ShortcutsDialog
    r.key_combo('?', dur=1.2)
    r.hold(0.4)
    sd = ShortcutsDialog(w, w)
    sd.show()
    sd.move(w.mapToGlobal(w.rect().center()) - sd.rect().center())
    stage.extra.append(sd)
    r.crossfade(0.3)
    r.hold(max(1.8, r.speech_left() + 0.8))
    sd.hide()
    r.crossfade(0.3)
    r.finish_speech(0.1)
    # 11: the .vantage file
    r.say(11)
    r.peek_menu('File', 'Save as', hold=max(1.0, r.speech_left() - 1.4))
    r.cursor_on = False
    r.finish_speech(1.0)
