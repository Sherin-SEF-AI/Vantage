"""17 Tolerance, optimiser and variants: the Monte Carlo tolerance study on the sample's 6-cam layout (1000 samples
within ±5° and ±10 mm: the histogram, the tornado, the worst cases pinned in the right-hand views), then the
'4-cam, mirror cameras 5° off' variant through the placement optimiser (least blind ground, scored under tolerance,
10 generations of 60 layouts), its best candidate applied as a new variant, and Layout ▸ Compare variants.

Figures read from the app while recording (r.expect): tolerance study on 6-cam · v3, seed 1729, 1000 samples:
nominal 0.010 m², P95 0.053 m², max 0.075 m²; the tornado's two longest bars are F θ and R θ. The mirror-5°-off
variant: blind in ROI 0.623 m². Optimiser (seed 7, 10 cm cells, 24 tolerance samples per candidate): C1 turns
ML/MR from 110.0° to 101.9° and R's pitch from 60.0° to 48.6°; the worst case under tolerance 0.940 -> 0.360 m².
Compare (5 cm cells): blind 0.623 -> 0.130 m², worst single-camera error 0.910 -> 0.360 m².
"""

META = {
    'id': 'tolerance',
    'part': 3,
    'title': 'Tolerance, optimiser and variants',
    'promise': 'How sure is the layout, and is there a better one?',
    'menu': 'Analysis ▸ Tolerance study  ·  Layout ▸ Optimise placement  ·  Layout ▸ Compare variants',
}

LINES = [
    ("Brackets are never perfect: every camera ends up a little off the pose it was planned at. The tolerance study "
     "measures what that costs.",
     "Brackets are never perfect: every camera ends up a little off the pose it was planned at. The tolerance study "
     "measures what that costs."),
    ("It moves every camera at random within the tolerances you set, here ±5° in yaw, pitch and roll and ±10 mm in "
     "position, a thousand times.",
     "It moves every camera at random, within the tolerances you set: here, plus or minus five degrees in yaw, pitch "
     "and roll, and ten millimetres in position, a thousand times."),
    ("Run it: each sample is a full coverage analysis, spread over every processor core.",
     "Run it. Each sample is a full coverage analysis, spread over every processor core."),
    ("The histogram counts the blind ground in the region over the thousand samples: 0.010 m² as planned, 0.053 at "
     "the 95th percentile, and 0.075 at worst.",
     "The histogram counts the blind ground in the region, over the thousand samples: zero point zero one square "
     "metres as planned, zero point zero five three at the ninety-fifth percentile, and zero point zero seven five at "
     "worst."),
    ("The tornado moves one parameter at a time to its limit. The front and rear cameras' pitch matter most.",
     "The tornado moves one parameter at a time to its limit. The front and rear cameras' pitch matter most."),
    ("The worst cases are listed with the camera that moved, by how much, and where the ground went blind. Pin one "
     "to see it in the top view and in 3D.",
     "The worst cases are listed with the camera that moved, by how much, and where the ground went blind. Pin one, "
     "to see it in the top view and in 3D."),
    ("Can a better layout be found? This variant has its mirror cameras turned five degrees too far: 0.62 m² of the "
     "region is blind.",
     "Can a better layout be found? This variant has its mirror cameras turned five degrees too far: zero point six "
     "two square metres of the region is blind."),
    ("Layout ▸ Optimise placement runs an evolutionary search. The objective here is the least blind ground, scored "
     "under tolerance: each candidate is perturbed 24 times.",
     "Optimise placement, from the layout menu, runs an evolutionary search. The objective here is the least blind "
     "ground, scored under tolerance: each candidate is perturbed twenty-four times."),
    ("Each camera gets a range to search: the front and rear slide along the bumper and tilt, the mirror cameras turn "
     "and tilt, their partners mirrored. Constraints keep mount zones, heights and cable lengths.",
     "Each camera gets a range to search. The front and rear slide along the bumper and tilt, the mirror cameras turn "
     "and tilt, with their partners mirrored. Constraints keep the mount zones, heights, and cable lengths."),
    ("Ten generations of sixty layouts: every layout evaluated is a dot, its worst blind area against its minimum "
     "density, and the Pareto front is drawn through the best.",
     "Ten generations of sixty layouts. Every layout evaluated is a dot, its worst blind area against its minimum "
     "density, and the Pareto front is drawn through the best."),
    ("The best candidate turns the mirror cameras back by eight degrees and tilts the rear camera up by eleven. On "
     "the optimiser's grid, the worst case falls from 0.94 to 0.36 m².",
     "The best candidate turns the mirror cameras back by eight degrees, and tilts the rear camera up by eleven. On "
     "the optimiser's grid, the worst case falls from zero point nine four, to zero point three six square metres."),
    ("On the right, its camera footprints and where the density changed. Apply it as a new variant: the original "
     "stays as it was.",
     "On the right, its camera footprints, and where the density changed. Apply it as a new variant: the original "
     "stays as it was."),
    ("Layout ▸ Compare variants puts layouts side by side, maps the density difference between two of them, and "
     "tables every variant in the project.",
     "Compare variants puts layouts side by side, maps the density difference between two of them, and tables every "
     "variant in the project."),
    ("On the full five-centimetre grid, the optimised layout cuts the blind ground from 0.623 to 0.130 m², and the "
     "worst single-camera error from 0.910 to 0.360.",
     "On the full five centimetre grid, the optimised layout cuts the blind ground from zero point six two three, to "
     "zero point one three square metres, and the worst single camera error from zero point nine one, to zero point "
     "three six."),
]

MIRROR = '4-cam, mirror cameras 5° off'


def scene(stage, r):
    from PyQt6.QtCore import QPointF, QRectF

    w = stage.window()
    r.win = w
    st = w.store
    p = st.project
    rows = w.bottom.coverage.rows
    r.cursor_on = False
    r.cur = __import__('numpy').array([1180.0, 640.0])

    # ---- 0: the opener, the tolerance page from its menu
    r.lower_third()
    r.say(0)
    r.hold(1.2)
    r.cursor_on = True
    r.menu('Analysis', 'Tolerance study')
    pg = w.page('tolerance')
    stage.settle(2)
    r.crossfade(0.4)
    r.finish_speech(0.1)

    # ---- 1: the tolerances and the sample count
    r.say(1)
    left = pg.yaw.parentWidget()
    lrect = r.wrect(left)
    r.zoom(QRectF(lrect.x(), lrect.y(), lrect.width() + 120, 560), 1.0, wait=False)
    r.move_to(pg.yaw, None, 0.8)
    r.until_word(1, 'plus or minus')
    r.callout(lambda: r.wrect(pg.yaw).united(r.wrect(pg.pos)).adjusted(-130, 0, 0, 0), '', dur=3.2, pad=4, dim=0.25)
    r.until_word(1, 'a thousand times')
    r.enter(pg.samples, '1000')
    if int(pg.samples.value()) != 1000:
        pg.samples.set_value(1000)
    r.hold(0.4)
    r.finish_speech(0.1)
    r.zoom(None, 0.8, wait=False)

    # ---- 2: run it (a background job; a marked time-lapse)
    r.say(2)
    r.click(pg.b_run, None, 0.8)
    r.wait_for(lambda: 'tolerance' not in w._jobs and pg.result is not None, timeout=300, speed=6)
    res = pg.result
    r.expect(len(res.blind), 1000, 'samples run')
    stats = pg.stats.text()
    print('tolerance stats:', stats, flush=True)
    r.expect(f'{res.nominal_blind:.3f}', '0.010', 'nominal blind')
    r.expect(f'{res.pct(95):.3f}', '0.053', 'P95')
    r.expect(f'{res.blind.max():.3f}', '0.075', 'max')
    r.finish_speech(0.1)

    # ---- 3: the histogram and its percentiles
    r.say(3)
    hr = r.wrect(pg.hist)
    sr = r.wrect(pg.stats)
    r.zoom(hr.united(sr), 1.0, margin=16)
    r.until_word(3, 'zero point zero five three')
    r.figure('Blind in ROI · P95', res.pct(95), 'm²', 3, dur=2.6, src=pg.stats, start=res.nominal_blind)
    r.until_word(3, 'at worst')
    r.figure('Blind in ROI · worst', float(res.blind.max()), 'm²', 3, dur=max(2.4, r.speech_left() + 0.8),
             src=pg.stats, start=res.pct(95))
    r.finish_speech(0.2)

    # ---- 4: the tornado
    r.say(4)
    top2 = sorted(res.sensitivity, key=lambda s: -max(abs(s[3]), abs(s[4])))[:2]
    print('tornado top:', [s[0] for s in top2], flush=True)
    r.expect(sorted(s[0] for s in top2), ['F θ', 'R θ'], 'the two longest tornado bars')
    r.zoom(pg.tornado, 1.0, margin=16)
    r.move_to(pg.tornado, QPointF(pg.tornado.width() * 0.5, pg.tornado.height() * 0.3), 0.8)
    r.finish_speech(0.4)

    # ---- 5: the worst cases, one pinned
    r.say(5)
    tb = pg.table
    trect = r.wrect(tb)
    r.zoom(QRectF(trect.x(), trect.y() - 30, trect.width(), min(trect.height(), 230) + 30), 1.0)
    r.until_word(5, 'Pin one')
    pin = tb.cellWidget(2, 5)
    r.click(pin, None, 0.7)
    stage.settle(2)
    r.hold(0.5)
    r.zoom(None, 0.9)
    right = r.wrect(pg.pin_plan).united(r.wrect(pg.pin_3d))
    r.callout(right, '', dur=max(2.2, r.speech_left() + 0.8), pad=-2, dim=0.3)
    r.finish_speech(0.3)

    # ---- 6: the mirror-5°-off variant
    r.say(6)
    vm = next(v for v in p.variants if v.name == MIRROR)
    stage.mark_result()
    r.choose(w.variant_combo, w.variant_combo.findData(vm.uid))
    stage.wait_analysis(r)
    r.expect(rows['Blind in ROI'].v.text(), '0.623', 'blind in the mirror-off variant')
    r.until_word(6, 'zero point six two')
    r.figure('Blind in ROI', float(rows['Blind in ROI'].v.text()), 'm²', 3, dur=max(2.6, r.speech_left() + 0.6),
             src=rows['Blind in ROI'], note=MIRROR)
    r.finish_speech(0.2)

    # ---- 7: the optimiser and its objective
    r.say(7)
    r.menu('Layout', 'Optimise placement')
    op = w.page('optimiser')
    stage.settle(2)
    r.crossfade(0.4)
    r.hold(0.4)
    orect = r.wrect(op.obj.buttons()[0]).united(r.wrect(op.tol_note))
    r.zoom(QRectF(orect.x() - 8, orect.y() - 70, 520, orect.height() + 110), 1.0)
    r.until_word(7, 'least blind')
    r.callout(op.obj.buttons()[0], '', dur=1.8, pad=4, dim=0.2)
    r.until_word(7, 'scored under')
    r.callout(lambda: r.wrect(op.under_tol).united(r.wrect(op.tol_note)), '', dur=max(2.0, r.speech_left() + 0.4),
              pad=4, dim=0.2)
    r.finish_speech(0.2)

    # ---- 8: the free variables and the constraints
    r.say(8)
    vt = r.wrect(op.vars_t)
    r.zoom(QRectF(vt.x() - 8, vt.y() - 40, max(vt.width(), 300) + 16, vt.height() + 70), 1.0)
    r.move_to(op.vars_t.viewport(), QPointF(op.vars_t.viewport().width() * 0.7, 10), 0.8)
    r.until_word(8, 'Constraints')
    cons = r.wrect(op.zones).united(r.wrect(op.module))
    r.zoom(QRectF(cons.x() - 130, cons.y() - 20, cons.width() + 160, cons.height() + 40), 0.8)
    r.callout(lambda: r.wrect(op.zones).united(r.wrect(op.cable)).adjusted(-120, 0, 0, 0), '',
              dur=max(1.8, r.speech_left() + 0.3), pad=4, dim=0.2)
    r.finish_speech(0.1)

    # ---- 9: ten generations (a marked time-lapse)
    r.say(9)
    r.zoom(QRectF(r.wrect(op.gens).x() - 40, r.wrect(op.gens).y() - 120, 420, 240), 0.8)
    r.enter(op.gens, '10')
    if int(op.gens.value()) != 10:
        op.gens.set_value(10)
    r.hold(0.2)
    r.zoom(None, 0.7, wait=False)
    r.click(op.b_run, None, 0.7)
    r.hold(0.3)
    r.zoom(QRectF(r.wrect(op.scatter).x(), r.wrect(op.scatter).y() - 50, r.wrect(op.scatter).width(),
                  r.wrect(op.scatter).height() + 50), 1.0, wait=False)
    r.wait_for(lambda: 'optimiser' not in w._jobs, timeout=900, speed=6)
    stage.settle(2)
    r.hold(0.6)
    r.finish_speech(0.2)

    # ---- 10: the best candidate
    r.say(10)
    r.expect(op.table.item(0, 0).text(), 'C1', 'first candidate')
    changes = op.table.item(0, 6).text()
    print('C1 changes:', changes, '|', op.summary.text(), flush=True)
    r.expect('ML/MR ψ 110.0→101.9' in changes and 'R θ 60.0→48.6' in changes, True, 'C1 changes')
    nom, best = op.opt.nominal_cand.worst, op.cands[0].worst
    r.expect(f'{nom:.3f}/{best:.3f}', '0.940/0.360', 'worst under tolerance')
    tr = r.wrect(op.table)
    r.zoom(QRectF(tr.x(), tr.y() - 34, tr.width(), 34 + 6 * 23), 1.0)
    r.callout(r.local_rect(op.table.viewport(), QRectF(op.table.visualItemRect(op.table.item(0, 0)).united(
        op.table.visualItemRect(op.table.item(0, 6))))), '', dur=3.4, pad=3, dim=0.25)
    r.until_word(10, 'the worst case falls')
    r.zoom(None, 0.8)
    r.figure('Worst case under tolerance', best, 'm²', 2, dur=max(2.8, r.speech_left() + 0.8), src=op.summary,
             start=nom, note='C1 · optimiser grid')
    r.finish_speech(0.2)

    # ---- 11: its maps; apply it as a new variant
    r.say(11)
    rr = r.wrect(op.plan).united(r.wrect(op.delta))
    r.zoom(rr, 1.0, margin=8)
    r.hold(max(1.5, r.at_word(11, 'Apply') - r.now() - 1.0))
    r.zoom(None, 0.8)
    n0 = len(p.variants)
    stage.mark_result()
    r.click(op.b_apply, None, 0.7)
    stage.wait_analysis(r)
    r.expect(len(p.variants), n0 + 1, 'variants after Apply')
    r.callout(w.variant_combo, '', dur=max(2.0, r.speech_left() + 0.4), pad=4, dim=0.25)
    r.finish_speech(0.2)

    # ---- 12: compare variants
    r.say(12)
    r.menu('Layout', 'Compare variants')
    cp = w.page('compare')
    stage.settle(2)
    r.crossfade(0.4)
    r.hold(0.3)
    combo0 = cp.cols[0]['combo']
    r.choose(combo0, combo0.findData(vm.uid))
    r.wait_for(lambda: all(v.uid in cp.worsts for v in p.variants), timeout=180)
    stage.settle(1)
    r.hold(0.4)
    top = r.wrect(cp.cols[0]['plan']).united(r.wrect(cp.cols[1]['plan']))
    r.zoom(QRectF(top.x(), top.y() - 30, top.width(), top.height() + 30), 1.0, wait=False)
    r.until_word(12, 'maps the density')
    r.zoom(None, 0.8, wait=False)
    r.callout(lambda: r.wrect(cp.delta).adjusted(0, -24, 0, 30), '', dur=2.4, pad=2, dim=0.25)
    r.until_word(12, 'tables every')
    r.callout(cp.table, '', dur=max(1.8, r.speech_left() + 0.4), pad=4, dim=0.25)
    r.finish_speech(0.2)

    # ---- 13: the figures on the full grid
    r.say(13)
    t = cp.table
    rm = next(i for i in range(t.rowCount()) if t.item(i, 0).text().startswith(MIRROR) and '· C1' not in t.item(i, 0).text())
    rc = next(i for i in range(t.rowCount()) if '· C1' in t.item(i, 0).text())
    r.expect(t.item(rm, 2).text(), '0.623', 'blind before')
    r.expect(t.item(rc, 2).text(), '0.130', 'blind after')
    r.expect(t.item(rm, 6).text(), '0.910', 'worst single-camera error before')
    r.expect(t.item(rc, 6).text(), '0.360', 'worst single-camera error after')

    def cell(row, c):
        return r.local_rect(t.viewport(), QRectF(t.visualItemRect(t.item(row, c))))
    tw = r.wrect(t)
    r.zoom(QRectF(tw.x(), tw.y() - 40, tw.width(), tw.height() + 120), 1.0)
    r.figure('Blind in ROI', float(t.item(rc, 2).text()), 'm²', 3, dur=3.0, src=cell(rc, 2), start=0.623,
             note='optimised · 5 cm grid')
    r.until_word(13, 'worst single camera')
    r.figure('Worst single-camera error', float(t.item(rc, 6).text()), 'm²', 3, dur=max(2.6, r.speech_left() + 0.6),
             src=cell(rc, 6), start=0.910, note='optimised · 5 cm grid')
    r.finish_speech(0.4)
    r.zoom(None, 0.9)
    r.cursor_on = False
    r.hold(0.8)
