"""16 Issues and rules: the Issues list on the sample, the front camera pushed 10 cm off the bumper (critical: off the
body; serious: the routed cable longer than the module's; the blind ground grows), the row clicked (the camera
selected and framed), Snap to surface, the GMSL2 link budget at 32 bit/px (a deserializer over its CSI-2 output; the
recorder short of write speed) and undone, the minimum density raised to 4 px/cm, and the other rules.

Figures read from the app while recording (r.expect): three issues on the sample (blind 0.010 m², the disk set, the
0.9 m child at ML | QL); F +10 cm: "F is 115 mm off the body", "Routed cable 3.06 m is longer than the 3.00 m module
cable", blind 0.070 m²; at 32 bit/px deserializer A 12.46 of 10.0 Gbps (125 %), the recorder 2,187 MB/s against
1,500 MB/s; minimum 4 px/cm: below it on 9.2 % of the ROI.
"""

META = {
    'id': 'issues',
    'part': 2,
    'title': 'Issues and rules',
    'promise': 'Every rule checked on every edit, with the fix one click away.',
    'menu': 'Issues panel  ·  Inspector ▸ Compute ▸ Link budget  ·  Analysis ▸ Density thresholds…',
}

LINES = [
    ("The Issues list checks the design against its rules after every edit. On the sample, three: the blind ground in "
     "the region, a disk set that fills before the shift ends, and a child that can vanish at a seam.",
     "The issues list checks the design against its rules after every edit. On the sample, three: the blind ground in "
     "the region, a disk set that fills before the shift ends, and a child that can vanish at a seam."),
    ("Push the front camera 10 centimetres forward, off the bumper.",
     "Push the front camera ten centimetres forward, off the bumper."),
    ("At once it is critical, 115 millimetres off the body; its routed cable, 3.06 metres, is longer than the "
     "module's 3-metre cable; and the blind ground grows to 0.07 square metres.",
     "At once, it is critical: a hundred and fifteen millimetres off the body. Its routed cable, three point zero six "
     "metres, is longer than the module's three metre cable. And the blind ground grows to zero point zero seven "
     "square metres."),
    ("Click an issue and its camera is selected and framed. The Inspector offers the fix: Snap to surface, or keep "
     "the standoff as a bracket.",
     "Click an issue, and its camera is selected and framed. The inspector offers the fix: snap to surface, or keep "
     "the standoff as a bracket."),
    ("Snapped back onto the bumper, the camera's two issues are gone, and the blind ground is back to 0.01 square "
     "metres.",
     "Snapped back onto the bumper, the camera's two issues are gone, and the blind ground is back to zero point zero "
     "one square metres."),
    ("The GMSL2 link budget is a rule too. In the Compute inspector, set 32 bits a pixel: deserializer A then carries "
     "12.46 gigabits a second, over its 10-gigabit CSI-2 output.",
     "The G M S L two link budget is a rule too. In the compute inspector, set thirty-two bits a pixel: deserializer A "
     "then carries twelve point four six gigabits a second, over its ten gigabit C S I two output."),
    ("And the recorder can no longer write the data: 2,187 megabytes a second against 1,500. Undo, and both are "
     "gone.",
     "And the recorder can no longer write the data: two thousand one hundred and eighty-seven megabytes a second, "
     "against fifteen hundred. Undo, and both are gone."),
    ("Thresholds are rules as well: raise the minimum density to 4 pixels per centimetre, and the ground below it is "
     "flagged, 9.2 per cent of the region.",
     "Thresholds are rules as well: raise the minimum density to four pixels per centimetre, and the ground below it "
     "is flagged: nine point two per cent of the region."),
    ("Other rules check the lidars, radars and antennas, the power budget, GNSS sky, dirt, fleet tolerances and the "
     "scorecard's must requirements, all on every edit.",
     "Other rules check the lidars, radars, and antennas, the power budget, G N S S sky, dirt, fleet tolerances, and "
     "the scorecard's must requirements, all on every edit."),
]


def scene(stage, r):
    from PyQt6.QtCore import QPointF, QRectF, Qt

    w = stage.window()
    r.win = w
    st = w.store
    app = stage.app
    iss = w.bottom.issues
    rows = w.bottom.coverage.rows

    def shown(k):
        return rows[k].v.text()

    def pump(n=4):
        for _ in range(n):
            app.processEvents()

    def issue_rows():
        iss.refresh()
        return [(iss.t.item(i, 0).text(), iss.t.item(i, 1).text(), iss.t.item(i, 2).text())
                for i in range(iss.t.rowCount())]

    def row_rect(i):
        def f():
            a = iss.t.visualItemRect(iss.t.item(i, 0))
            b = iss.t.visualItemRect(iss.t.item(i, 2))
            tl = r.wpos(iss.t.viewport(), QPointF(a.topLeft()))
            return QRectF(tl.x(), tl.y(), b.right() - a.left(), a.height())
        return f
    w.resizeDocks([w.docks['bottom']], [380], Qt.Orientation.Vertical)    # set-up: room for the Issues list
    stage.settle(2)
    bb = r.wrect(w.docks['bottom'])
    zone = QRectF(bb.x(), bb.y() - 10, min(bb.width() * 0.6, 1150), bb.height() + 10)

    # ---- 0: the list on the sample
    r.cursor_on = True
    r.lower_third()
    r.say(0)
    r.hold(0.8)
    tabs = w.bottom.tabs
    r.click(tabs, QPointF(tabs.tabRect(w.bottom.keys.index('issues')).center()), 0.7, real=False)
    w.bottom.show_tab('issues')
    pump()
    rws = issue_rows()
    print('issues:', rws, flush=True)
    r.expect(len(rws), 3, 'issues on the sample')
    r.expect(rws[0][1].startswith('0.010 m² of ground in the ROI is blind'), True, 'the blind ground issue')
    r.zoom(zone, 1.0)
    r.cursor_on = False
    for i, word in enumerate(('the blind ground', 'a disk set', 'and a child')):
        r.until_word(0, word)
        r.callout(row_rect(i), '', dur=1.8 if i < 2 else max(1.8, r.speech_left() + 0.2), pad=2, dim=0.25)
    r.finish_speech(0.2)
    r.zoom(None, 0.8)

    # ---- 1: F pushed 10 cm forward in the Inspector
    r.say(1)
    r.cursor_on = True
    f = st.variant.by_tag('F')
    st.select_cams([f.uid])
    stage.settle(2)
    page = w.inspector.pages['camera']
    insp = r.wrect(w.docks['inspector'])
    r.zoom(QRectF(insp.x() - 40, insp.y(), insp.width() + 40, 300), 0.8, wait=False)
    stage.mark_result()
    x0 = float(f.x)
    r.enter(page.px, f'{x0:.3f}+0.1', cps=12)
    if abs(st.cam(f.uid).x - (x0 + 0.1)) > 1e-4:
        page.px.committed.emit(round(x0 + 0.1, 4))
    r.expect(f'{st.cam(f.uid).x:.3f}', f'{x0 + 0.1:.3f}', "F's new x")
    stage.wait_analysis(r)
    r.zoom(None, 0.7)
    r.finish_speech(0.1)

    # ---- 2: the issues that follow
    rws = issue_rows()
    print('after the push:', rws, flush=True)
    msgs = [x[1] for x in rws]
    r.expect(any(m.startswith('F is 115 mm off the body') for m in msgs), True, 'F off the body')
    r.expect(any(m.startswith('Routed cable 3.06 m is longer than the 3.00 m module cable') for m in msgs), True,
             'the routed cable')
    r.expect(shown('Blind in ROI'), '0.070', 'blind after the push')
    i_off = next(i for i, m in enumerate(msgs) if m.startswith('F is 115 mm'))
    i_cab = next(i for i, m in enumerate(msgs) if m.startswith('Routed cable'))
    i_bl = next(i for i, m in enumerate(msgs) if 'of ground in the ROI is blind' in m)
    r.say(2)
    r.zoom(zone, 1.0)
    r.cursor_on = False
    r.callout(row_rect(i_off), '', dur=2.2, pad=2, dim=0.25)
    r.until_word(2, 'Its routed cable')
    r.callout(row_rect(i_cab), '', dur=2.6, pad=2, dim=0.25)
    r.until_word(2, 'And the blind ground')
    r.callout(row_rect(i_bl), '', dur=max(2.0, r.speech_left() + 0.3), pad=2, dim=0.25)
    r.finish_speech(0.2)

    # ---- 3: click the issue; the fix in the Inspector
    r.say(3)
    r.cursor_on = True
    rr = iss.t.visualItemRect(iss.t.item(i_off, 1))
    r.click(iss.t.viewport(), QPointF(rr.x() + 140, rr.center().y()), 0.7)
    pump()
    if st.selected is None or st.selected.uid != f.uid:
        iss._show(i_off, 0)
    pump()
    print('selected after the click:', [st.cam(u).tag for u in st.sel_cams], flush=True)
    r.expect([st.cam(u).tag for u in st.sel_cams], ['F'], 'the issue selects F')
    r.zoom(None, 0.8)
    r.until_word(3, 'The inspector offers')
    stage.settle(1)
    r.zoom(QRectF(insp.x() - 40, insp.y(), insp.width() + 40, insp.height() * 0.6), 0.9)
    r.callout(page.float_box, '', dur=max(2.4, r.speech_left()), pad=4, dim=0.25)
    r.finish_speech(0.2)

    # ---- 4: Snap to surface
    r.say(4)
    stage.mark_result()
    r.click(page.b_snap, None, 0.7)
    stage.wait_analysis(r)
    r.zoom(None, 0.7)
    rws = issue_rows()
    print('after the snap:', rws, flush=True)
    r.expect(len(rws), 3, 'issues after the snap')
    r.expect(shown('Blind in ROI'), '0.010', 'blind after the snap')
    r.zoom(zone, 0.9)
    r.hold(max(1.0, r.speech_left()))
    r.finish_speech(0.2)

    # ---- 5: the link budget at 32 bit/px
    r.say(5)
    r.zoom(None, 0.6)
    row = r.tree_item(w.outliner, 'Compute')
    rr = row()
    if rr is not None:
        r.move(QPointF(rr.x() + 80, rr.center().y()), 0.7)
        r.ripple()
    st.select_obj(('ecu',))
    stage.settle(2)
    ep = w.inspector.pages['ecu']
    sa = ep.parentWidget()
    from PyQt6.QtWidgets import QScrollArea
    while sa is not None and not isinstance(sa, QScrollArea):
        sa = sa.parentWidget()
    if sa is not None:
        sa.ensureWidgetVisible(ep.budget_note, 0, 40)
        pump()
    r.zoom(QRectF(insp.x() - 40, insp.y(), insp.width() + 40, insp.height()), 0.8)
    stage.mark_result()
    r.enter(ep.bpp, '32', cps=8)
    if st.project.ecu.bpp != 32:
        ep.bpp.committed.emit(32.0)
    stage.wait_analysis(r)
    pump()
    bt = [[ep.bt.item(i, j).text() for j in range(4)] for i in range(ep.bt.rowCount())]
    print('budget at 32:', bt, ep.budget_note.text(), flush=True)
    r.expect(bt[0][2:], ['12.46 / 10', '125 %'], 'deserializer A at 32 bit/px')
    r.callout(ep.bt, '', dur=max(2.4, r.speech_left()), pad=3, dim=0.25)
    r.finish_speech(0.2)

    # ---- 6: the recorder too; undo
    rws = issue_rows()
    print('issues at 32:', rws, flush=True)
    msgs = [x[1] for x in rws]
    i_rec = next(i for i, m in enumerate(msgs) if m.startswith('The recorder cannot write the data: 2,187 MB/s against '
                                                                 '1,500 MB/s'))
    i_des = next(i for i, m in enumerate(msgs) if m.startswith('Deserializer A carries 12.46 Gbps'))
    r.say(6)
    r.zoom(zone, 1.0)
    r.cursor_on = False
    r.callout(row_rect(i_des), '', dur=1.8, pad=2, dim=0.25)
    r.hold(1.6)
    r.callout(row_rect(i_rec), '', dur=2.4, pad=2, dim=0.25)
    r.until_word(6, 'Undo')
    r.key_combo('Ctrl+Z', dur=1.2)
    stage.mark_result()
    st.undo.undo()
    stage.wait_analysis(r)
    rws = issue_rows()
    print('after the undo:', rws, flush=True)
    r.expect(len(rws), 3, 'issues after the undo')
    r.hold(max(0.8, r.speech_left()))
    r.finish_speech(0.2)
    r.zoom(None, 0.7)

    # ---- 7: the minimum density threshold
    r.say(7)
    r.cursor_on = True
    r.menu('Analysis', 'Density thresholds')
    stage.settle(2)
    roi = w.inspector.pages['roi']
    sa2 = roi.parentWidget()
    while sa2 is not None and not isinstance(sa2, QScrollArea):
        sa2 = sa2.parentWidget()
    if sa2 is not None:
        sa2.ensureWidgetVisible(roi.t_tgt, 0, 60)
        pump()
    r.zoom(QRectF(insp.x() - 40, insp.y(), insp.width() + 40, insp.height()), 0.8)
    stage.mark_result()
    r.enter(roi.t_min, '4', cps=8)
    if abs(st.project.analysis.min_density - 4.0) > 1e-6:
        roi.t_min.committed.emit(4.0)
    stage.wait_analysis(r)
    rws = issue_rows()
    print('min 4:', rws, flush=True)
    r.expect(any(x[1] == 'Minimum density 3.1 px/cm is below 4.0 px/cm on 9.2 % of the ROI.' for x in rws), True,
             'the density issue')
    i_d = next(i for i, x in enumerate(rws) if x[1].startswith('Minimum density'))
    r.zoom(zone, 0.9)
    r.cursor_on = False
    r.callout(row_rect(i_d), '', dur=max(2.2, r.speech_left() + 0.3), pad=2, dim=0.25)
    r.finish_speech(0.2)
    r.zoom(None, 0.7)

    # ---- 8: the other rules
    r.say(8)
    w.bottom.show_tab('coverage')
    st.select_cams([])
    stage.settle(2)
    vp = stage.vp
    r.crossfade(0.4)
    stage.orbit(r, vp, 50, max(4.0, r.speech_left() + 0.6), d_el=5)
    r.finish_speech(1.0)
