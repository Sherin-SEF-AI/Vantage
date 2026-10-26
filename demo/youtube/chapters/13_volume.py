"""13 3D blind volume: Analysis ▸ Blind volume (the page works it out when it opens, shown in real time), the result
(blind volume and its share of the space, ground covered up to a height), the height slider through the levels with
the 3D cut-away following, objects standing around the vehicle, the smallest object seen, the reach rose, and the
export and 3D overlay.

Figures read from the app while recording (r.expect): 551 280 points; blind volume 30.62 m³, 44.4 % of 68.9 m³;
covered up to 0.30 m; blind at 0.03 m: 0.01 m², at 0.43 m: 0.73 m², at 1.03 m: 14.09 m²; pole 0.5 m, child 0.9 m and
adult 1.8 m each unseen on 0.005 m², at most 0.05 m from the body.
"""

META = {
    'id': 'volume',
    'part': 2,
    'title': '3D blind volume',
    'promise': 'Not just the ground: where can something stand unseen?',
    'menu': 'Analysis ▸ Blind volume…  ·  View ▸ Overlays ▸ Blind volume',
}

LINES = [
    ("The ground is only part of it. Analysis ▸ Blind volume asks where, around the vehicle, something can stand "
     "unseen by every camera, at every height.",
     "The ground is only part of it. Analysis, blind volume, asks where, around the vehicle, something can stand "
     "unseen by every camera, at every height."),
    ("The region's columns are tested every 5 centimetres up to 2 metres, with the same test as the ground: lens "
     "model, image edges, the body, the rig and the housings. 551,280 points.",
     "The region's columns are tested every five centimetres, up to two metres, with the same test as the ground: "
     "lens model, image edges, the body, the rig, and the housings. Five hundred and fifty-one thousand points."),
    ("The result: 30.6 of the 68.9 cubic metres around the car are blind, 44 per cent, most of it high up, above "
     "where the cameras look.",
     "The result: thirty point six, of the sixty-eight point nine cubic metres around the car, are blind: forty-four "
     "per cent, most of it high up, above where the cameras look."),
    ("Low down, almost nothing hides: the space is covered up to 0.30 metres.",
     "Low down, almost nothing hides: the space is covered up to zero point three metres."),
    ("The height slider cuts through it: at 3 centimetres, 0.01 square metres are blind; at 43 centimetres, 0.73; at "
     "a metre, 14. The 3D view cuts away the blind space up to the same height.",
     "The height slider cuts through it: at three centimetres, zero point zero one square metres are blind; at forty-"
     "three centimetres, zero point seven three; at a metre, fourteen. The 3D view cuts away the blind space up to "
     "the same height."),
    ("For a dataset, what matters is whether objects are seen. A half-metre pole, a child and an adult are each "
     "missed on just 0.005 square metres, within 5 centimetres of the body.",
     "For a dataset, what matters is whether objects are seen. A half metre pole, a child, and an adult are each "
     "missed on just zero point zero zero five square metres, within five centimetres of the body."),
    ("Smallest object seen maps the shortest object a camera still catches on each cell: on this layout, 5 "
     "centimetres, on 99.98 per cent of the region.",
     "Smallest object seen maps the shortest object a camera still catches on each cell: on this layout, five "
     "centimetres, on ninety-nine point nine eight per cent of the region."),
    ("The reach rose shows, in each direction, how far from the body something of each height can hide.",
     "The reach rose shows, in each direction, how far from the body something of each height can hide."),
    ("The blind space exports as an OBJ surface with its figures in JSON, and shows in the main 3D view as an "
     "overlay.",
     "The blind space exports as an O B J surface, with its figures in JSON, and shows in the main 3D view as an "
     "overlay."),
]


def scene(stage, r):
    from PyQt6.QtCore import QPointF, QRectF

    from helpers_tour_a import drag_sent

    w = stage.window()
    r.win = w
    st = w.store
    app = stage.app
    ctl = w.ctx.volume

    def pump(n=4):
        for _ in range(n):
            app.processEvents()

    # ---- 0: Analysis ▸ Blind volume
    r.cursor_on = True
    r.lower_third()
    r.say(0)
    r.hold(1.0)
    r.menu('Analysis', 'Blind volume')
    pg = w.page('volume')
    r.crossfade(0.4)
    r.wait_for(lambda: ctl.result is not None and not ctl.busy, timeout=120)
    pump()
    r.cursor_on = False
    r.finish_speech(0.2)

    # ---- 1: what is tested
    r.say(1)
    left = pg.fields['height'][0].parentWidget()
    lw = r.wrect(left)
    r.zoom(QRectF(lw.x(), lw.y(), lw.width(), 300), 1.0)
    r.callout(lambda: r.wrect(pg.fields['height'][0]).united(r.wrect(pg.fields['cell'][0])).adjusted(-120, 0, 0, 0),
              '', dur=2.6, pad=4, dim=0.25)
    r.until_word(1, 'Five hundred')
    stt = pg.status.text()
    print('status:', stt, flush=True)
    r.expect(stt.startswith('551 280 points'), True, 'points tested')
    r.zoom(QRectF(lw.x(), lw.y() + lw.height() - 170, lw.width(), 170), 0.8)
    r.callout(pg.status, '', dur=max(1.8, r.speech_left() + 0.3), pad=4, dim=0.25)
    r.finish_speech(0.2)

    # ---- 2: the result
    met = {k: v.text() for k, v in pg.metric.items()}
    print('metrics:', met, flush=True)
    r.expect(met['blind'], '30.62 m³', 'blind volume')
    r.expect(met['pct'], '44.4 % of 68.9 m³', 'share of the space')
    r.expect(met['covered'], '0.30 m', 'covered up to')
    r.say(2)
    mb = r.wrect(pg.metric['blind'])
    r.zoom(QRectF(lw.x(), mb.y() - 60, lw.width(), 260), 0.9)
    r.figure('Blind volume', 30.62, 'm³', 2, dur=max(3.2, r.speech_left() - 1.5), src=pg.metric['blind'])
    r.hold(max(0.5, r.speech_left() - 1.6))
    r.zoom(None, 0.8)
    r.finish_speech(0.2)

    # ---- 3: covered up to 0.30 m
    r.say(3)
    r.callout(lambda: r.wrect(pg.metric['covered']).adjusted(-180, 0, 40, 0), '', dur=max(2.4, r.speech_left() + 0.3),
              pad=4, dim=0.3)
    r.finish_speech(0.3)

    # ---- 4: the height slider; the 3D cut-away follows
    vr = ctl.result
    lv = {round(float(z), 2): round(float(b), 2) for z, b in zip(vr.z, vr.level_blind_m2)}
    print('levels:', lv, flush=True)
    r.expect((lv[0.03], lv[0.43], lv[1.03]), (0.01, 0.73, 14.09), 'blind area at three heights')
    sl = pg.slider

    def sx(k):
        from PyQt6.QtWidgets import QStyle, QStyleOptionSlider
        opt = QStyleOptionSlider()
        sl.initStyleOption(opt)
        hr = sl.style().subControlRect(QStyle.ComplexControl.CC_Slider, opt, QStyle.SubControl.SC_SliderHandle, sl)
        return hr.width() / 2 + (sl.width() - hr.width()) * (k - sl.minimum()) / max(1, sl.maximum() - sl.minimum())
    y = sl.height() / 2
    k_of = {round(float(z), 2): i for i, z in enumerate(vr.z)}
    r.say(4)
    r.cursor_on = True
    right = pg.map.parentWidget()
    r.zoom(QRectF(r.wrect(right).x(), r.wrect(sl).y() - 30, r.wrect(right).width(), r.wrect(right).height() * 0.7),
           0.9, wait=False)
    drag_sent(r, sl, QPointF(sx(sl.value()), y), QPointF(sx(k_of[0.03]), y), 1.0)
    r.expect(pg.z_label.text(), '0.03 m · 0.01 m²', 'slider at 3 cm')
    r.until_word(4, 'at forty-')
    drag_sent(r, sl, QPointF(sx(sl.value()), y), QPointF(sx(k_of[0.43]), y), 1.0)
    r.expect(pg.z_label.text(), '0.43 m · 0.73 m²', 'slider at 43 cm')
    r.until_word(4, 'at a metre')
    drag_sent(r, sl, QPointF(sx(sl.value()), y), QPointF(sx(k_of[1.03]), y), 1.0)
    r.expect(pg.z_label.text(), '1.03 m · 14.09 m²', 'slider at 1.03 m')
    r.until_word(4, 'The 3D view')
    r.cursor_on = False
    vw = r.wrect(pg.view)
    r.zoom(QRectF(vw.x(), vw.y() + vw.height() * 0.18, vw.width(), vw.height() * 0.62), 1.0)
    stage.orbit(r, pg.view, 50, max(2.4, r.speech_left() + 0.4), d_el=4)
    r.finish_speech(0.2)
    r.zoom(None, 0.8)

    # ---- 5: objects standing around the vehicle
    rows = [[pg.table.item(i, j).text() for j in range(4)] for i in range(pg.table.rowCount())]
    print('objects:', rows, flush=True)
    r.expect(rows, [['Pole 0.5 m', '0.50', '0.005', '0.05'], ['Child 0.9 m', '0.90', '0.005', '0.05'],
                    ['Adult 1.8 m', '1.80', '0.005', '0.05']], 'objects table')
    r.say(5)
    tb = r.wrect(pg.table)
    r.zoom(QRectF(lw.x(), tb.y() - 40, lw.width(), tb.height() + 80), 1.0)
    r.callout(pg.table, '', dur=max(2.8, r.speech_left() - 0.4), pad=3, dim=0.25)
    r.finish_speech(0.2)
    r.zoom(None, 0.8)

    # ---- 6: smallest object seen
    r.say(6)
    r.cursor_on = True
    import numpy as np
    vr = ctl.result
    cols = vr.roi & ~vr.body
    share = float(np.mean(vr.detect_h[cols] <= 0.05 + 1e-6))
    print('seen at 5 cm on', share, flush=True)
    r.expect(f'{100 * share:.2f}', '99.98', 'share of the region where 5 cm is seen')
    r.click(pg.mode.buttons['detect'], None, 0.7)
    pump()
    r.zoom(right, 0.9, margin=6)
    r.hold(max(1.6, r.speech_left()))
    r.finish_speech(0.2)

    # ---- 7: the reach rose
    r.say(7)
    r.click(pg.mode.buttons['slice'], None, 0.5)
    r.cursor_on = False
    r.zoom(pg.rose.parentWidget(), 1.0, margin=6)
    r.hold(max(1.6, r.speech_left()))
    r.finish_speech(0.2)
    r.zoom(None, 0.8)

    # ---- 8: export; the overlay in the main view
    r.say(8)
    r.cursor_on = True
    r.move_to(pg.b_export, None, 0.7)
    r.callout(pg.b_export, '', dur=1.8, pad=4, dim=0.25)
    r.hold(1.6)
    r.menu('View', 'Back to viewports')
    stage.settle(2)
    if not w.ctx.overlays.get('volume'):
        r.menu('View', 'Overlays', 'Blind volume')
    stage.settle(3)
    r.crossfade(0.5)
    vp = w.viewarea.panes[0].vp3d
    r.cursor_on = False
    stage.orbit(r, vp, 40, max(2.4, r.speech_left() + 0.4), d_el=5)
    r.finish_speech(1.0)
