"""24 Mount brackets: Analysis ▸ Mount brackets on the sample (6-cam · v3). Which cameras can take a printed bracket;
the left mirror camera moved out to the standoff a bracket needs (its mirrored partner follows); the settings; Generate
at the default 0.35 mm grid; the bracket orbited in the preview; its figures, checks and notes; the exports.

Figures read from the app while recording (r.expect): F needs 41 mm standoff (16 mm planned, its housing would run
23 mm into the base); ML 15 mm, needs 45; after Move out ML and MR are Ready; Generate: 4 brackets, 2 cameras without;
ML's bracket: 26.0 cm³, 28 g of ASA; optical centre 0.00 mm and plate 0.00° from the plan; image blocked 0.0 %;
thinnest wall 3.0 mm.
"""

META = {
    'id': 'brackets',
    'part': 3,
    'title': 'Mount brackets',
    'promise': 'A printable bracket that puts each camera exactly where it was planned.',
    'menu': 'Analysis ▸ Mount brackets',
}

LINES = [
    ("A camera ends up where the plan says only if its bracket puts it there. Mount brackets designs a 3D-printable "
     "bracket for each camera, holding it at its planned optical centre and angles.",
     "A camera ends up where the plan says, only if its bracket puts it there. Mount brackets designs a 3D printable "
     "bracket for each camera, holding it at its planned optical centre and angles."),
    ("First it checks each camera against the surface under it. Most of the sample's cameras sit too close: the "
     "front camera's housing would run 23 mm into a bracket's base, so it needs 41 mm of standoff.",
     "First, it checks each camera against the surface under it. Most of the sample's cameras sit too close: the "
     "front camera's housing would run twenty-three millimetres into a bracket's base, so it needs forty-one "
     "millimetres of standoff."),
    ("Select the left mirror camera: 15 mm off its mirror housing, where a bracket needs 45. Move out sets that "
     "standoff as one undo step, and its mirrored partner follows.",
     "Select the left mirror camera: fifteen millimetres off its mirror housing, where a bracket needs forty-five. "
     "Move out sets that standoff as one undo step, and its mirrored partner follows."),
    ("The settings: which face of the housing the plate goes on, the hole pattern, the material, and the fixing: "
     "two or four countersunk screws, a VHB pad, or a clamp on a rig bar.",
     "The settings: which face of the housing the plate goes on, the hole pattern, the material, and the fixing: two "
     "or four countersunk screws, a V H B pad, or a clamp on a rig bar."),
    ("Generate meshes every bracket that can be made: a plate on the housing, a filleted support, and a base fitted "
     "to the body under it.",
     "Generate meshes every bracket that can be made: a plate on the housing, a filleted support, and a base fitted "
     "to the body under it."),
    ("Four brackets; the front and rear cameras stay without one until they are moved out too, or the panel gets a "
     "cut-out. Orbit the bracket in the preview.",
     "Four brackets. The front and rear cameras stay without one, until they are moved out too, or the panel gets a "
     "cut-out. Orbit the bracket in the preview."),
    ("Then the checks, measured back from the mesh: the optical centre lands 0.00 mm from the plan, the bracket stays "
     "out of the image, and the thinnest wall is 3 mm.",
     "Then the checks, measured back from the mesh: the optical centre lands zero point zero zero millimetres from the "
     "plan, the bracket stays out of the image, and the thinnest wall is three millimetres."),
    ("Each comes with print and fixing notes, and exports as an STL in millimetres, with a JSON of its frame, "
     "parameters and checks, and a sheet of dimensioned views.",
     "Each comes with print and fixing notes, and exports as an S T L in millimetres, with a Jason of its frame, "
     "parameters, and checks, and a sheet of dimensioned views."),
]


def scene(stage, r):
    import numpy as np
    from PyQt6.QtCore import QPointF, QRectF

    w = stage.window()
    r.win = w
    st = w.store
    r.cursor_on = False
    vp = stage.vp
    vp.cam.az, vp.cam.el = -28.0, 16.0
    w.viewarea.frame_all()
    stage.settle(2)
    r.cur = np.array([1150.0, 560.0])

    # ---- 0: the opener; the page from its menu
    r.lower_third()
    r.say(0)
    stage.orbit(r, vp, 30, 3.0, d_el=3)
    r.cursor_on = True
    r.menu('Analysis', 'Mount brackets')
    pg = w.page('brackets')
    ctl = pg.ctl
    n = len([c for c in st.variant.cameras if c.visible])
    r.crossfade(0.4)
    r.wait_for(lambda: len(ctl.pre) == n, timeout=60)
    stage.settle(1)
    r.finish_speech(0.1)
    t = pg.table

    def row(tag):
        return next(i for i in range(t.rowCount()) if t.item(i, 0).text() == tag)

    def cells(r0, c0, r1, c1):
        return lambda: QRectF(r.local_rect(t.viewport(), QRectF(t.visualItemRect(t.item(r0, c0))))()).united(
            r.local_rect(t.viewport(), QRectF(t.visualItemRect(t.item(r1, c1))))())

    # ---- 1: the statuses; the front camera
    r.say(1)
    r.expect(t.item(row('F'), 3).text(), '16 → 41', "F's standoff")
    tr = r.wrect(t)
    r.zoom(QRectF(tr.x(), tr.y() - 26, tr.width(), 26 + 23 * (n + 1)), 1.0)
    r.callout(cells(0, 2, n - 1, 3), '', dur=2.6, pad=3, dim=0.2)
    r.until_word(1, "the front camera's housing")
    r.click(t.viewport(), QPointF(t.visualItemRect(t.item(row('F'), 1)).center()), 0.6)
    stage.settle(1)
    print('F state:', pg.state.text.text(), flush=True)
    r.expect('runs 23 mm into the base' in pg.state.text.text(), True, "F's state")
    det = r.wrect(pg.state)
    r.zoom(QRectF(det.x() - 10, det.y() - 40, det.width() + 20, 140), 0.9)
    r.callout(pg.state, '', dur=max(2.2, r.speech_left() + 0.4), pad=3, dim=0.2)
    r.finish_speech(0.2)

    # ---- 2: the left mirror camera, moved out
    r.say(2)
    r.zoom(QRectF(tr.x(), tr.y() - 26, tr.width(), 26 + 23 * (n + 1)), 0.8)
    i_ml = row('ML')
    r.expect(t.item(i_ml, 3).text(), '15 → 45', "ML's standoff")
    r.click(t.viewport(), QPointF(t.visualItemRect(t.item(i_ml, 1)).center()), 0.6)
    stage.settle(1)
    r.callout(cells(i_ml, 0, i_ml, 3), '', dur=2.4, pad=3, dim=0.2)
    r.until_word(2, 'Move out sets')
    r.zoom(QRectF(det.x() - 10, det.y() - 40, det.width() + 20, 140), 0.8)
    ml = st.variant.by_tag('ML')
    r.click(pg.b_move, None, 0.7)
    r.wait_for(lambda: ctl.bracket(ml.uid) is not None and ctl.bracket(ml.uid).status == 'ok', timeout=60)
    stage.settle(1)
    r.zoom(QRectF(tr.x(), tr.y() - 26, tr.width(), 26 + 23 * (n + 1)), 0.8)
    r.expect((t.item(i_ml, 2).text(), t.item(row('MR'), 2).text()), ('Ready', 'Ready'), 'ML and MR after Move out')
    r.callout(cells(i_ml, 0, row('MR'), 3), '', dur=max(2.0, r.speech_left() + 0.4), pad=3, dim=0.2)
    r.finish_speech(0.2)

    # ---- 3: the settings
    r.say(3)
    fr = r.wrect(pg.face)
    lo = r.wrect(pg.fixing)
    r.zoom(QRectF(fr.x() - 20, fr.y() - 40, 380, lo.bottom() - fr.y() + 120), 1.0)
    r.callout(pg.face, '', dur=1.6, pad=3, dim=0.2)
    r.until_word(3, 'the hole pattern')
    r.callout(pg.pattern, '', dur=1.4, pad=3, dim=0.2)
    r.until_word(3, 'the material')
    r.callout(pg.material, '', dur=1.4, pad=3, dim=0.2)
    r.until_word(3, 'and the fixing')
    r.callout(pg.fixing, '', dur=max(1.8, r.speech_left() + 0.3), pad=3, dim=0.2)
    r.finish_speech(0.2)

    # ---- 4: Generate (a background job: a marked time-lapse)
    r.say(4)
    r.zoom(None, 0.8, wait=False)
    r.click(pg.b_gen, None, 0.7)
    r.wait_for(lambda: not ctl.busy and ctl.result is not None, timeout=300, speed=3)
    stage.settle(1)
    r.finish_speech(0.2)

    # ---- 5: four brackets; orbit one
    r.say(5)
    res = ctl.result
    built = [b for b in res.brackets if b.generated]
    r.expect(len(built), 4, 'brackets generated')
    r.click(t.viewport(), QPointF(t.visualItemRect(t.item(i_ml, 1)).center()), 0.6)
    stage.settle(1)
    r.callout(cells(0, 2, n - 1, 2), '', dur=2.6, pad=3, dim=0.2)
    r.hold(2.6)
    v = pg.view
    r.zoom(r.wrect(v), 1.0, margin=8, wait=False)
    cx, cy = v.width() / 2, v.height() / 2
    r.drag(v, QPointF(cx + 110, cy + 20), QPointF(cx - 70, cy + 34), dur=1.8)
    r.move_to(v, QPointF(v.width() * 0.88, v.height() * 0.86), 0.4)
    r.hold(max(0.4, r.speech_left()))
    r.finish_speech(0.1)

    # ---- 6: the checks
    r.say(6)
    b = ctl.bracket(ml.uid)
    fl = pg.fig_labels
    r.expect(fl['fov'].text(), '0.0 %', 'image blocked')
    r.expect(fl['wall'].text(), '3.0 mm', 'thinnest wall')
    r.expect(fl['solid'].text().startswith('26.0 cm³ · 28 g'), True, 'solid')
    checks = [c.text.text() for c in pg.check_lines]
    print('ML checks:', checks, flush=True)
    r.expect(any('optical centre lands 0.00 mm and the plate 0.00°' in c for c in checks), True, 'measured pose')
    sa = r.wrect(pg.check_host.parentWidget().parentWidget())
    r.zoom(QRectF(sa.x() - 10, sa.y() - 30, sa.width() + 20, 230), 1.0)
    r.callout(pg.check_lines[1], '', dur=2.4, pad=3, dim=0.2)
    r.until_word(6, 'stays out of the image')
    r.callout(pg.check_lines[2], '', dur=1.8, pad=3, dim=0.2)
    r.until_word(6, 'thinnest wall')
    r.callout(pg.check_lines[4], '', dur=max(1.6, r.speech_left() + 0.4), pad=3, dim=0.2)
    r.finish_speech(0.2)

    # ---- 7: notes and exports
    r.say(7)
    nr = r.wrect(pg.notes)
    er = r.wrect(pg.b_all)
    r.zoom(QRectF(nr.x() - 10, nr.y() - 20, nr.width() + 20, er.bottom() - nr.y() + 30), 1.0)
    r.callout(pg.notes, '', dur=2.6, pad=3, dim=0.2)
    r.until_word(7, 'exports as')
    r.callout(lambda: r.wrect(pg.b_stl).united(r.wrect(pg.b_all)), '', dur=max(1.8, r.speech_left() + 0.3), pad=4,
              dim=0.2)
    r.finish_speech(0.3)
    r.zoom(None, 0.9)
    r.cursor_on = False
    r.hold(0.8)
