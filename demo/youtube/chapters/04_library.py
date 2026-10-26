"""04 Camera modules and lenses: the e-con Systems library (17 modules, 20 lenses), search, the module editor on the
sample's STURDeCAM31 + H05R2 (datasheet fields, the Source block, the r(theta) plot, datasheet-vs-model field of view,
image circle and derived values), the Preset list (STURDeCAM88 + H04R1, OmniVision OX08B40), Import, and Camera ▸
Change lens on the front camera with the analysis following, then undone.

Figures read from the app while recording (r.expect): 17 modules and 20 lenses in the library; F is STURDeCAM31 + H05R2;
largest datasheet-vs-model difference 0.4 deg; 15.24 px/deg at the centre and 10.28 at the horizontal edge; the
STURDeCAM88 is an OmniVision OX08B40 at 3840 x 2160; the editor offers to switch 6 cameras (cancelled); F on the narrow
H03R2: blind ground in the ROI 0.010 -> 1.130 m^2, minimum density 3.1 -> 2.5 px/cm; Ctrl+Z restores 0.010.
"""

META = {
    'id': 'library',
    'part': 1,
    'title': 'Camera modules and lenses: the e-con library',
    'promise': 'Every module and lens, each figure with its source.',
    'menu': 'Library  ·  Camera ▸ Edit module…  ·  Camera ▸ Change lens',
}

LINES = [
    ("Cameras come from a built-in library of e-con Systems modules: 17 modules with 20 lenses, from the enclosed "
     "STURDeCAM automotive cameras to the NileCAM GMSL2 board cameras and the FPD-Link NeduCAMs.",
     "Cameras come from a built-in library of e-con Systems modules: seventeen modules, with twenty lenses, from the "
     "enclosed Sturdy-Cam automotive cameras, to the Nile-Cam G M S L two board cameras, and the F P D link "
     "Nedu-Cams."),
    ("Search finds them by name, sensor or interface: here, the NileCAMs.",
     "Search finds them by name, sensor, or interface: here, the Nile-Cams."),
    ("Each lens sits under its module, with its field of view and focal length. The sample's cameras are "
     "STURDeCAM31s with the wide H05R2 lens.",
     "Each lens sits under its module, with its field of view and focal length. The sample's cameras are Sturdy-Cam "
     "thirty-ones, with the wide H zero five R two lens."),
    ("Camera ▸ Edit module opens the editor. On the left, the datasheet as e-con's pages state it: sensor, "
     "resolution, pixel pitch, frame rate, shutter, dynamic range, IP rating, temperature and power.",
     "Camera, edit module, opens the editor. On the left, the datasheet as e-con's pages state it: sensor, "
     "resolution, pixel pitch, frame rate, shutter, dynamic range, I P rating, temperature, and power."),
    ("Below it, the source: the pages the figures were read from, the date they were read, and which figures are not "
     "on e-con's pages.",
     "Below it, the source: the pages the figures were read from, the date they were read, and which figures are not "
     "on e-con's pages."),
    ("e-con does not publish focal lengths, so the lens model is fitted to the stated field of view. The plot shows "
     "image height against ray angle, beside the ideal projections.",
     "e-con does not publish focal lengths, so the lens model is fitted to the stated field of view. The plot shows "
     "image height against ray angle, beside the ideal projections."),
    ("It reproduces the datasheet's horizontal, vertical and diagonal field of view, here within 0.4 degrees.",
     "It reproduces the datasheet's horizontal, vertical, and diagonal field of view, here within zero point four "
     "degrees."),
    ("On the right, the image circle on the sensor, and derived values: 15.2 pixels per degree at the centre, 10.3 at "
     "the edge of the horizontal field.",
     "On the right, the image circle on the sensor, and derived values: fifteen point two pixels per degree at the "
     "centre, ten point three at the edge of the horizontal field."),
    ("The Preset list holds the whole library. Pick the STURDeCAM88 with its wide H04R1 lens: an 8-megapixel "
     "OmniVision sensor, with its own figures and source.",
     "The preset list holds the whole library. Pick the Sturdy-Cam eighty-eight, with its wide H zero four R one "
     "lens: an eight megapixel OmniVision sensor, with its own figures, and source."),
    ("Modules also import from JSON, CSV, or an OpenCV, ROS or Kalibr YAML. Applying would switch all six cameras; "
     "here it is cancelled.",
     "Modules also import from JSON, C S V, or an Open C V, ROS, or Kalibr YAML. Applying would switch all six "
     "cameras; here, it is cancelled."),
    ("Camera ▸ Change lens switches the selected cameras: the front camera on the narrow H03R2, and the blind ground "
     "in the region jumps from 0.01 to 1.13 square metres.",
     "Camera, change lens, switches the selected cameras: the front camera on the narrow H zero three R two, and the "
     "blind ground in the region jumps from zero point zero one, to one point one three square metres."),
    ("A narrow lens is for range, not for the ground beside the car. Ctrl+Z puts the wide lens back.",
     "A narrow lens is for range, not for the ground beside the car. Control Z puts the wide lens back."),
]

PICK = 'sturdecam88_h04r1'


def scene(stage, r):
    from PyQt6.QtCore import QPoint, QPointF, QRectF, Qt
    from PyQt6.QtWidgets import QApplication, QMenu, QScrollArea

    from vantage.core.lens import builtin_lenses, builtin_modules
    from vantage.ui.dialogs.module_editor import ModuleEditor

    w = stage.window()
    r.win = w
    st = w.store
    app = stage.app
    rows = w.bottom.coverage.rows

    def shown(k):
        return rows[k].v.text()

    def pump(n=4):
        for _ in range(n):
            app.processEvents()
    # set-up, not filmed: a taller, wider Library dock (the docks are the user's to size)
    lib_dock = w.docks['library']
    w.resizeDocks([w.docks['outliner'], lib_dock], [250, 700], Qt.Orientation.Vertical)
    w.resizeDocks([lib_dock], [330], Qt.Orientation.Horizontal)
    stage.settle(3)
    from vantage.ui.outliner import LibraryPanel
    panel = lib_dock.findChild(LibraryPanel)
    tree = panel.tree
    r.expect(len(builtin_modules()), 17, 'modules in the library')
    r.expect(len(builtin_lenses()), 20, 'lenses in the library')
    cam_f = st.variant.by_tag('F')
    r.expect((cam_f.module_id, cam_f.lens_id), ('sturdecam31', 'h05r2'), "F's module and lens")
    r.expect(shown('Blind in ROI'), '0.010', 'blind before')

    def lib_rect():
        return r.wrect(lib_dock)

    # ---- 0: the library
    r.cursor_on = True
    r.lower_third()
    r.say(0)
    r.hold(1.0)
    r.zoom(lambda: QRectF(lib_rect().x(), lib_rect().y() - 60, lib_rect().width() + 120, lib_rect().height() + 60),
           1.0, margin=10)
    r.move_to(tree.viewport(), QPointF(150, 80), 0.8)
    r.callout(tree, '', dur=max(3.0, r.speech_left() - 0.5), pad=2, dim=0.2)
    r.finish_speech(0.2)

    # ---- 1: search
    r.say(1)
    r.click(panel.search, None, 0.6)
    r.type(panel.search, 'nile', cps=8)
    pump()
    r.hold(max(1.4, r.speech_left() + 0.4))
    names = [tree.topLevelItem(i).text(0) for i in range(tree.topLevelItemCount())]
    print('search nile:', names, flush=True)
    r.expect(sum(1 for n in names if n.startswith('NileCAM')), 4, 'NileCAMs found')
    panel.search.clear()
    pump()
    r.hold(0.4)

    # ---- 2: lenses under their modules; the sample's STURDeCAM31 + H05R2
    r.say(2)

    def lens_row(prefix):
        def find():
            for i in range(tree.topLevelItemCount()):
                g = tree.topLevelItem(i)
                if g.text(0).startswith('STURDeCAM31 '):
                    for k in range(g.childCount()):
                        it = g.child(k)
                        if it.text(0).startswith(prefix):
                            rr = tree.visualItemRect(it)
                            tl = r.wpos(tree.viewport(), QPointF(0, rr.top()))
                            return QRectF(tl.x(), tl.y(), tree.viewport().width(), rr.height())
            return None
        return find
    r.hold(0.8)
    r.callout(lens_row('H03R2'), '', dur=2.0, pad=1, dim=0.25)
    r.until_word(2, "The sample's cameras")
    rr = lens_row('H05R2')()
    r.move(QPointF(rr.x() + 120, rr.center().y()), 0.6)
    r.callout(lens_row('H05R2'), '', dur=max(2.0, r.speech_left()), pad=1, dim=0.25)
    r.finish_speech(0.2)
    r.zoom(None, 0.8, wait=False)

    # ---- 3: Camera ▸ Edit module: the datasheet
    st.select_cams([cam_f.uid])
    stage.settle(2)
    r.say(3)
    mb = w.menuBar()
    m = next(a.menu() for a in mb.actions() if a.menu() and a.text().replace('&', '') == 'Camera')
    rect = mb.actionGeometry(m.menuAction())
    r.move_to(mb, QPointF(rect.center()), 0.7)
    r.ripple()
    m.popup(mb.mapToGlobal(rect.bottomLeft()))
    r.hold(0.35)
    act = next(a for a in m.actions() if a.text().replace('&', '').startswith('Edit module'))
    ar = m.actionGeometry(act)
    r.move_to(m, QPointF(ar.center().x() - ar.width() * 0.2, ar.center().y()), 0.55)
    m.setActiveAction(act)
    r.hold(0.3)
    r.ripple()
    r.hold(0.15)
    m.hide()
    d = ModuleEditor(w.ctx, cam_f.module_id, cam_f.lens_id, w)
    stage.extra.append(d)
    d.resize(min(1500, w.width() - 60), min(900, w.height() - 30))
    d.show()
    pump(6)
    d.move(w.mapToGlobal(QPoint(max(0, (w.width() - d.width()) // 2), max(0, (w.height() - d.height()) // 2))))
    pump(4)
    r.crossfade(0.4)
    sa = d.source.parentWidget()
    while sa is not None and not isinstance(sa, QScrollArea):
        sa = sa.parentWidget()
    r.hold(0.6)
    col = r.wrect(sa)
    r.zoom(QRectF(col.x() - 10, col.y() - 30, col.width() + 200, col.height() * 0.62), 1.0)
    r.move_to(d.f['m_sensor'], None, 0.6)
    r.until_word(3, 'pixel pitch')
    r.move_to(d.f['m_pixel'], None, 0.5)
    r.until_word(3, 'dynamic range')
    r.move_to(d.f['m_dr'], None, 0.5)
    r.until_word(3, 'temperature')
    r.move_to(d.f['m_temp'], None, 0.5)
    r.finish_speech(0.2)

    # ---- 4: the Source block
    r.say(4)
    bar = sa.verticalScrollBar()
    v0, v1 = bar.value(), bar.maximum()
    r.move_to(sa.viewport(), QPointF(sa.viewport().width() * 0.6, sa.viewport().height() * 0.7), 0.5)
    n = round(0.9 * stage.fps)
    for k in range(1, n + 1):
        t = k / n
        bar.setValue(round(v0 + (v1 - v0) * t * t * (3 - 2 * t)))
        r.tick()
    src_txt = d.source.text()
    r.expect('(read 2026-10-24)' in src_txt and 'not on e-con' in src_txt, True, 'the Source block')
    src = r.wrect(d.source)
    r.zoom(QRectF(src.x() - 10, src.y() - 30, src.width() + 20, src.height() + 40), 0.9, margin=8)
    r.callout(d.source, '', dur=max(2.4, r.speech_left()), pad=4, dim=0.2)
    r.finish_speech(0.2)
    r.zoom(None, 0.8, wait=False)

    # ---- 5: the projection plot
    r.say(5)
    r.hold(0.8)
    r.zoom(QRectF(r.wrect(d.plot).x() - 10, r.wrect(d.plot).y() - 50, r.wrect(d.plot).width() + 20,
                  r.wrect(d.plot).height() + 80), 1.0)
    r.move_to(d.plot, QPointF(d.plot.width() * 0.55, d.plot.height() * 0.45), 0.8)
    r.finish_speech(0.2)
    r.zoom(None, 0.8, wait=False)

    # ---- 6: datasheet against model
    deltas = [abs(float(d.cmp.item(i, 3).text().replace('−', '-').replace('+', ''))) for i in range(3)]
    print('datasheet vs model:', [[d.cmp.item(i, j).text() for j in range(4)] for i in range(3)], flush=True)
    r.expect(f'{max(deltas):.1f}', '0.4', 'largest datasheet-vs-model difference')
    r.say(6)
    cmp = r.wrect(d.cmp)
    r.zoom(QRectF(cmp.x() - 20, cmp.y() - 40, cmp.width() + 40, cmp.height() + 70), 1.0)
    r.callout(d.cmp, '', dur=max(2.6, r.speech_left()), pad=3, dim=0.25)
    r.finish_speech(0.2)

    # ---- 7: image circle and derived values
    der = {k: v[1].text() for k, v in d.der_v.items()}
    print('derived:', der, flush=True)
    r.expect(der['px/° at centre'], '15.24', 'px/deg at the centre')
    r.expect(der['px/° at H edge'], '10.28', 'px/deg at the H edge')
    r.say(7)
    circ = r.wrect(d.circle)
    r.zoom(QRectF(circ.x() - 10, circ.y() - 30, circ.width() + 20, 470), 1.0)
    r.until_word(7, 'derived values')
    a_, v_ = d.der_v['px/° at centre']
    r.callout(lambda: r.wrect(a_).united(r.wrect(v_)), '', dur=1.8, pad=3, dim=0.25)
    r.until_word(7, 'ten point three')
    a2, v2 = d.der_v['px/° at H edge']
    r.callout(lambda: r.wrect(a2).united(r.wrect(v2)), '', dur=max(1.8, r.speech_left() + 0.3), pad=3, dim=0.25)
    r.finish_speech(0.2)
    r.zoom(None, 0.8)

    # ---- 8: the Preset list: STURDeCAM88 + H04R1
    r.say(8)
    bar.setValue(0)
    r.click(d.preset, dur=0.6, real=False)
    d.preset.showPopup()
    r.hold(1.8)
    i = d.preset.findData(PICK)
    view = d.preset.view()
    view.setCurrentIndex(view.model().index(i, 0))
    view.scrollTo(view.model().index(i, 0))
    r.hold(0.6)
    ir = view.visualRect(view.model().index(i, 0))
    r.move_to(view.viewport(), QPointF(ir.center().x() * 0.5, ir.center().y()), 0.5)
    r.ripple()
    r.hold(0.2)
    d.preset.hidePopup()
    d.preset.setCurrentIndex(i)
    d.apply_preset(PICK)
    pump()
    r.expect(d.f['m_sensor'].text(), 'OmniVision OX08B40', "the STURDeCAM88's sensor")
    r.expect(d.f['m_res'].text(), '3840 × 2160', "the STURDeCAM88's resolution")
    r.hold(0.4)
    r.zoom(QRectF(col.x() - 10, col.y() - 30, col.width() + 200, col.height() * 0.62), 0.9)
    r.callout(lambda: r.wrect(d.f['m_name']).united(r.wrect(d.f['m_res'])), '', dur=max(2.0, r.speech_left()),
              pad=4, dim=0.25)
    r.finish_speech(0.2)
    r.zoom(None, 0.8, wait=False)

    # ---- 9: Import; Apply would switch six cameras; cancelled
    r.say(9)
    imp = d.b_imp.menu()
    r.click(d.b_imp, None, 0.7, real=False)
    imp.popup(d.b_imp.mapToGlobal(QPoint(0, d.b_imp.height())))
    r.hold(max(1.6, r.at_word(9, 'Applying') - r.now() - 0.3))
    imp.hide()
    r.until_word(9, 'Applying')
    r.expect(d.ok.text(), 'Switch 6 cameras to STURDeCAM88 + H04R1', 'the apply button')
    r.move_to(d.ok, None, 0.7)
    r.callout(d.ok, '', dur=2.0, pad=4, dim=0.25)
    r.until_word(9, 'it is cancelled')
    cancel = next(b for b in d.findChildren(type(d.ok)) if b.text() == 'Cancel')
    r.click(cancel, None, 0.6, real=False)
    d.reject()
    pump()
    r.crossfade(0.35)
    r.finish_speech(0.2)

    # ---- 10: Camera ▸ Change lens on F: the analysis follows
    r.say(10)
    stage.mark_result()
    r.menu('Camera', 'Change lens', 'STURDeCAM31 ·', 'H03R2')
    stage.wait_analysis(r)
    r.expect(st.variant.by_tag('F').lens_id, 'h03r2', "F's lens after the change")
    r.expect(shown('Blind in ROI'), '1.130', 'blind with the narrow lens')
    r.cursor_on = False
    r.figure('Blind in ROI', float(shown('Blind in ROI')), 'm²', 3, dur=max(3.2, r.speech_left() + 0.8),
             src=rows['Blind in ROI'], start=0.010, note='F on the narrow H03R2')
    r.finish_speech(0.4)

    # ---- 11: Ctrl+Z
    r.say(11)
    r.hold(max(0.5, r.at_word(11, 'Control Z') - r.now()))
    r.key_combo('Ctrl+Z', dur=1.4)
    r.hold(0.3)
    st.undo.undo()
    stage.wait_analysis(r)
    r.expect(st.variant.by_tag('F').lens_id, 'h05r2', "F's lens after the undo")
    r.expect(shown('Blind in ROI'), '0.010', 'blind after the undo')
    r.figure('Blind in ROI', float(shown('Blind in ROI')), 'm²', 3, dur=max(2.6, r.speech_left() + 0.6),
             src=rows['Blind in ROI'], start=1.130, note='wide H05R2 back')
    r.finish_speech(1.0)
