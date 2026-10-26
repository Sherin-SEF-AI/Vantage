"""34 Fleet: Calibrate ▸ Fleet on the sample's data rig built on five vehicles and measured four times over six months
(demo/clip_fleet.build_fleet, seed 7): each vehicle's sensors a few millimetres and a tenth of a degree off the plan
with measurement noise; on V2 the front camera pitches down 0.2 deg a month; on V4 the left mirror camera is mounted
2 deg yawed. The records are simulated and labelled so ('simulated check N', note 'simulated for the demo'); the
narration and an on-screen note say so.

Figures read from the page while recording (r.expect), the same as core.fleet.run gives in a light run: 5 vehicles,
20 records, 12 sensors; 58 of 60 measured sensors within 10 mm / 1 deg; V4: ML yawed 2.1 deg left of the fleet median
(robust z 8.6); V2: F drifts +0.20 deg in pitch per 30 days, already out of tolerance (1.1 deg off).
"""

META = {
    'id': 'fleet',
    'part': 4,
    'title': 'Fleet',
    'promise': 'The same rig on every vehicle, measured over months: what drifts, and who is different.',
    'menu': 'Calibrate ▸ Fleet',
}

LINES = [
    ("A design gets built on many vehicles, and each one is measured again and again. Fleet keeps those measurements "
     "and compares them with the plan, over time and across the fleet.",
     "A design gets built on many vehicles, and each one is measured again and again. Fleet keeps those measurements, "
     "and compares them with the plan, over time, and across the fleet."),
    ("Here, five vehicles with the data rig, each measured four times over six months. The records are simulated for "
     "this video: small build offsets and noise, one camera slowly drifting, one mounted differently.",
     "Here, five vehicles with the data rig, each measured four times over six months. The records are simulated for "
     "this video: small build offsets and noise, one camera slowly drifting, one mounted differently."),
    ("Records come from an installation check report, Vantage's extrinsics in any frame and unit, nuScenes tables, a "
     "Kalibr camchain or a variant, or are typed in; sensors are matched to the plan by tag.",
     "Records come from an installation check report, Vantage's extrinsics in any frame and unit, new scenes tables, "
     "a calibre cam chain or a variant, or are typed in. Sensors are matched to the plan by tag."),
    ("The matrix puts every vehicle against every sensor, coloured by its share of the tolerance in the latest record. "
     "58 of 60 are within 10 millimetres and 1 degree.",
     "The matrix puts every vehicle against every sensor, coloured by its share of the tolerance in the latest record. "
     "Fifty-eight of sixty are within ten millimetres and one degree."),
    ("Across the fleet, a robust score against the median finds the vehicle built differently: V4's left mirror "
     "camera is yawed 2.1 degrees left of the rest.",
     "Across the fleet, a robust score against the median finds the vehicle built differently: V four's left mirror "
     "camera is yawed two point one degrees left of the rest."),
    ("Over time, each sensor gets a trend per 30 days. V2's front camera pitches down 0.2 degrees a month: already "
     "out of tolerance, and still moving.",
     "Over time, each sensor gets a trend per thirty days. V two's front camera pitches down zero point two degrees a "
     "month: already out of tolerance, and still moving."),
    ("The drift chart shows each record, the tolerance band, and the trend carried on over the drift window; a sudden "
     "step off the earlier course is flagged as a knock or a loose mount.",
     "The drift chart shows each record, the tolerance band, and the trend carried on over the drift window. A sudden "
     "step off the earlier course is flagged, as a knock or a loose mount."),
    ("When most vehicles share the same offset, the plan is what is wrong: Variant from the fleet median makes the "
     "as-built plan in one step. Export writes JSON, CSV and the chart.",
     "When most vehicles share the same offset, the plan is what is wrong: variant from the fleet median makes the "
     "as-built plan in one step. Export writes Jason, C S V, and the chart."),
]


def scene(stage, r):
    from PyQt6.QtCore import QPointF, QRectF
    import clip_fleet as CF
    import helpers_tourc as H

    w = stage._rig_window(CF.build_fleet)
    r.win = w
    st = w.store
    stage.tilde(w.msg_label)
    vp = stage.vp

    # 0 - open the page
    r.lower_third()
    r.say(0)
    r.cursor_on = False
    stage.orbit(r, vp, 20, 3.4, d_el=4)
    r.cursor_on = True
    r.until_word(0, 'Fleet keeps')
    r.menu('Calibrate', 'Fleet')
    pg = w.page('fleet')
    stage.settle(2)
    r.wait_for(lambda: pg.result is not None and not pg.busy, timeout=60)
    r.crossfade(0.4)
    r.finish_speech(0.2)
    res = pg.result
    vid = {v['name']: v['id'] for v in res.vehicles}

    # 1 - five vehicles, simulated records
    note = r.note('Fleet records simulated for this video', dur=10.0)
    r.say(1)
    vt = r.wrect(pg.vtable)
    r.zoom(QRectF(vt.x() - 10, vt.y() - 40, vt.width() + 20, vt.height() + 60), 1.0)
    r.callout(pg.vtable, dur=2.6, pad=4, dim=0.2)
    r.until_word(1, 'The records are simulated')
    rt = r.wrect(pg.rtable)
    r.zoom(QRectF(rt.x() - 10, rt.y() - 40, rt.width() + 20, rt.height() + 60), 1.0)
    r.callout(pg.rtable, dur=max(2.0, r.speech_left()), pad=4, dim=0.2)
    r.finish_speech(0.2)

    # 2 - where records come from
    r.say(2)
    b0, b1 = r.wrect(pg.b_import), r.wrect(pg.b_new_record)
    r.zoom(QRectF(b0.x() - 40, b0.y() - 120, b1.right() - b0.x() + 120, 300), 1.0)
    r.callout(pg.b_import, dur=3.0, pad=4)
    r.until_word(2, 'or a variant')
    r.callout(lambda: H.union(r.wrect(pg.b_from_variant), r.wrect(pg.b_new_record)), dur=2.4, pad=4)
    r.finish_speech(0.2)
    r.zoom(None, 0.8)

    # 3 - the matrix; 58 of 60 within
    T = pg.tiles
    print('tiles:', {k: (t.value, t.unit, t.note) for k, t in T.items()}, flush=True)
    r.expect((T['within'].value, T['within'].unit), ('58', 'of 60'), 'within tolerance')
    r.say(3)
    r.zoom(r.wrect(pg.matrix_scroll), 1.0, margin=40)
    r.hold(max(1.5, r.speech_left() - 3.2))
    r.zoom(None, 0.8)
    r.figure('Within 10 mm / 1°', int(T['within'].value), T['within'].unit, 0, dur=3.0, src=T['within'])
    r.finish_speech(0.4)

    # 4 - the outlier vehicle: V4's ML
    r.say(4)
    r.zoom(r.wrect(pg.matrix_scroll), 1.0, margin=40, wait=False)
    cell = pg.matrix.cell_rect(vid['V4'], 'ML')
    r.click(pg.matrix, cell.center(), 0.8)
    r.move(r.wpos(pg.matrix, QPointF(cell.right() + cell.width() * 0.6, cell.bottom() + 14)), 0.4)
    stage.settle(1)
    det = pg.detail.text.text()
    print('detail V4 ML:', det, flush=True)
    finds = [f.text.text() for f in pg.findings]
    print('findings:', finds, flush=True)
    r.expect(any(f.startswith('V4: ML is yawed 2.1° left of the fleet median') for f in finds), True, 'the outlier')
    r.hold(0.6)
    r.zoom(None, 0.8)
    fr = r.wrect(pg.find_host)
    r.zoom(QRectF(fr.x() - 10, fr.y() - 30, fr.width() + 20, fr.height() + 40), 1.0)
    k = next(i for i, f in enumerate(finds) if f.startswith('V4: ML is yawed'))
    r.callout(lambda: r.wrect(pg.findings[k]), dur=max(2.0, r.speech_left() + 0.3), pad=4, dim=0.2)
    r.finish_speech(0.2)
    r.zoom(None, 0.8)

    # 5 - drift: V2's F
    r.expect(any(f.startswith('V2: F drifts +0.20° in pitch per 30 days') for f in finds), True, 'the drift')
    r.say(5)
    r.zoom(r.wrect(pg.matrix_scroll), 1.0, margin=40)
    c2 = pg.matrix.cell_rect(vid['V2'], 'F')
    r.click(pg.matrix, c2.center(), 0.8)
    stage.settle(1)
    r.move(r.wpos(pg.chart, QPointF(pg.chart.width() * 0.62, pg.chart.height() * 0.9)), 0.5)
    r.zoom(None, 0.8)
    k2 = next(i for i, f in enumerate(finds) if f.startswith('V2: F drifts'))
    fr = r.wrect(pg.find_host)
    r.zoom(QRectF(fr.x() - 10, fr.y() - 30, fr.width() + 20, fr.height() + 40), 1.0)
    r.callout(lambda: r.wrect(pg.findings[k2]), dur=max(2.0, r.speech_left() + 0.3), pad=4, dim=0.2)
    r.finish_speech(0.2)
    r.zoom(None, 0.8)

    # 6 - the drift chart
    r.say(6)
    r.cursor_on = False
    r.zoom(r.wrect(pg.chart).united(r.wrect(pg.detail)), 1.0, margin=16)
    r.hold(max(1.6, r.speech_left() - 0.4))
    r.finish_speech(0.2)
    r.zoom(None, 0.8)

    # 7 - variant from the fleet median; export
    r.say(7)
    r.cursor_on = True
    bm = r.wrect(pg.b_median)
    r.zoom(QRectF(bm.x() - 40, bm.y() - 160, 640, 360), 1.0)
    r.move_to(pg.b_median, None, 0.8)
    r.callout(pg.b_median, dur=3.0, pad=4)
    r.until_word(7, 'Export writes')
    r.callout(pg.b_export, dur=max(1.8, r.speech_left() + 0.3), pad=4)
    r.cursor_on = False
    r.finish_speech(0.4)
    r.close_layer(note)
    r.zoom(None, 0.8)
    r.hold(0.6)
