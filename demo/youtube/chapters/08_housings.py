"""08 Housings block cameras: on the sample's data rig the front long-range radar RD1 sits just ahead of the bumper,
below camera F, and its housing fills the bottom of F's fisheye image.

Figures read from the app while recording (r.expect): RD1 covers 13.7 % of F's image (Inspector ▸ Mount ▸ Housings);
blind ground in the ROI 0.240 m² with housings, 0.010 m² with the body and rig alone; F pitched 40° down. RD1 moved
5 cm back behind the fascia: F's image is clear (checked with core.housings in demo/clip_housings.py). The project is
never saved.
"""

META = {
    'id': 'housings',
    'part': 1,
    'title': 'Housings block cameras',
    'promise': 'Lidars, radars and the other cameras get in each other’s way. See where, and by how much.',
    'menu': 'Analysis ▸ Housings block cameras  ·  Inspector ▸ Mount ▸ Housings',
}

LINES = [
    ("On a data rig, the cameras share the vehicle with lidars, radars and antennas. Here the front long-range radar "
     "sits just ahead of the bumper, right below the front camera.",
     "On a data rig, the cameras share the vehicle with lidars, radars, and antennas. Here, the front long range "
     "radar sits just ahead of the bumper, right below the front camera."),
    ("The front camera looks 40 degrees down through its fisheye lens. On the right is its view, with everything that "
     "blocks it in blue: the bumper, and the radar's housing.",
     "The front camera looks forty degrees down, through its fisheye lens. On the right is its view, with everything "
     "that blocks it in blue: the bumper, and the radar's housing."),
    ("The Inspector puts a number on it: the radar covers 13.7 per cent of the front camera's image.",
     "The inspector puts a number on it: the radar covers thirteen point seven per cent of the front camera's image."),
    ("And the Issues list flags it: a housing taking more than ten per cent of an image is serious.",
     "And the issues list flags it: a housing taking more than ten per cent of an image is serious."),
    ("Housings block each camera exactly like the body does. Switch them off to see the body and rig alone: the blind "
     "ground in the region drops from 0.24 to 0.01 square metres.",
     "Housings block each camera exactly like the body does. Switch them off, to see the body and rig alone: the "
     "blind ground in the region drops from zero point two four, to zero point zero one square metres."),
    ("Back on, the radar is back in the picture. So move it: five centimetres back, behind the bumper's fascia.",
     "Back on, the radar is back in the picture. So, move it: five centimetres back, behind the bumper's fascia."),
    ("The front camera's view is clear again, the Inspector agrees, and the blind ground is back to 0.01 square "
     "metres.",
     "The front camera's view is clear again, the inspector agrees, and the blind ground is back to zero point zero "
     "one square metres."),
    ("Every lidar, radar, GNSS antenna and IMU, and every other camera's module, is checked against every camera this "
     "way: in the coverage, the blind volume, perception range, the synthetic images and the installation check.",
     "Every lidar, radar, G N S S antenna and I M U, and every other camera's module, is checked against every camera "
     "this way: in the coverage, the blind volume, perception range, the synthetic images, and the installation "
     "check."),
]

BACK_M = 0.05


def scene(stage, r):
    import numpy as np
    from PyQt6.QtCore import QPointF, QRectF
    from vantage.ui.sensorview import set_props

    w = stage._rig_window()
    st = w.store
    v = st.variant
    cam_f = v.by_tag('F')
    rd1 = next(s for s in v.sensors if s.tag == 'RD1')
    rows = w.bottom.coverage.rows

    def shown(k):
        return rows[k].v.text()
    w.set_split('side')
    stage.settle(3)
    left, right = w.viewarea.panes[0], w.viewarea.panes[1]
    right.set_kind('through')
    right.cam.uid = cam_f.uid                         # the right view keeps looking through F whatever is selected
    vp = left.vp3d
    vp.cam.target = np.array([cam_f.x - 0.25, 0.0, 0.45])
    vp.cam.dist, vp.cam.az, vp.cam.el = 3.6, 30.0, 18.0
    vp.update()
    w.bottom.show_tab('coverage')
    st.select_cams([cam_f.uid])
    stage.settle(5)
    stage.mark_result()
    r.win = w
    r.cursor_on = False
    page = w.inspector.pages['camera']
    r.expect(f'{cam_f.pitch:.0f}', '40', "F's pitch")
    r.expect(shown('Blind in ROI'), '0.240', 'blind with housings')
    house = page.house_v.v.text()
    print('Inspector housings row:', house, flush=True)
    r.expect('13.7' in house, True, 'RD1 share of F in the Inspector')

    def through():                                     # the lower part of the view through F: the body and RD1
        g = r.wrect(right)
        return QRectF(g.x(), g.y() + g.height() * 0.40, g.width(), g.height() * 0.60)

    def house_row():
        return r.wrect(page.house_v)

    # 0 - the radar ahead of the bumper, below F
    r.lower_third()
    r.say(0)
    stage.orbit(r, vp, 16, 3.6, d_el=4, d_dist=-0.08)
    s, _, ok = vp.project([rd1.pos()])
    rd_pt = r.wpos(vp, QPointF(*s[0]))
    r.callout((rd_pt.x() - 46, rd_pt.y() - 34, 92, 68), 'RD1 · long-range radar', dur=3.0, style='circle',
              side='above')
    r.finish_speech(0.1)
    # 1 - F looks 40 deg down: its view on the right, blockers in blue
    r.say(1)
    r.hold(1.2)
    r.zoom(right, 1.0, margin=8)
    r.until_word(1, 'the bumper')
    r.callout(through, '', dur=max(2.0, r.speech_left() + 0.3), dim=0.0, pad=-4)
    r.finish_speech(0.2)
    r.zoom(None, 0.8)
    # 2 - the Inspector: 13.7 %
    r.say(2)
    row = house_row()
    r.move(QPointF(row.x() + row.width() * 0.5, row.y() + row.height() / 2), 0.7)
    r.zoom(QRectF(row.x() - 230, row.y() - 170, 540, 300), 1.0)
    r.callout(page.house_v, '', dur=max(2.4, r.speech_left()), pad=4)
    r.finish_speech(0.3)
    r.zoom(None, 0.8, wait=False)
    # 3 - the Issues list
    r.say(3)
    w.bottom.show_tab('issues')
    stage.settle(2)
    issues = w.bottom.issues if hasattr(w.bottom, 'issues') else None
    bottom = r.wrect(w.docks['bottom'])
    r.zoom(QRectF(bottom.x(), bottom.y(), bottom.width() * 0.62, bottom.height()), 1.0)
    if issues is not None:
        r.callout(issues, '', dur=max(2.0, r.speech_left()), dim=0.0, pad=-6)
    r.finish_speech(0.3)
    r.zoom(None, 0.8, wait=False)
    w.bottom.show_tab('coverage')
    stage.settle(1)
    # 4 - housings off: the body and rig alone
    r.say(4)
    r.hold(0.8)
    r.menu('Analysis', 'Housings block cameras')
    stage.wait_analysis(r)
    r.expect(shown('Blind in ROI'), '0.010', 'blind without housings')
    r.zoom(through, 0.8, margin=4)
    r.hold(1.6)
    r.zoom(None, 0.7)
    r.figure('Blind in ROI', float(shown('Blind in ROI')), 'm²', 3, dur=max(2.8, r.speech_left() + 0.8),
             src=rows['Blind in ROI'], start=0.240, note='body and rig alone')
    r.finish_speech(0.4)
    # 5 - back on; RD1 moved 5 cm back in the Inspector
    r.say(5)
    r.menu('Analysis', 'Housings block cameras')
    stage.wait_analysis(r)
    r.expect(shown('Blind in ROI'), '0.240', 'blind with housings again')
    r.hold(0.6)
    s, _, ok = vp.project([rd1.pos()])
    r.click(vp, QPointF(*s[0]), 0.7, real=False)
    st.select_obj(('sensor', rd1.uid))
    stage.settle(2)
    spage = w.inspector.pages['sensor']
    fx = r.wrect(spage.px)
    r.zoom(QRectF(fx.x() - 260, fx.y() - 140, 560, 320), 0.9, wait=False)
    r.move(QPointF(fx.x() + fx.width() * 0.7, fx.y() + fx.height() / 2), 0.6)
    x0 = rd1.x
    n = round(2.2 * stage.fps)
    for k in range(1, n + 1):
        e = stage_ease(k / n)
        set_props(w.ctx, rd1.uid, 'Moved RD1 back', merge=True, x=round(x0 - BACK_M * e, 4))
        r.cur = np.array([fx.x() + fx.width() * (0.7 - 0.35 * e), fx.y() + fx.height() / 2])
        r.tick()
    stage.wait_analysis(r)
    r.finish_speech(0.2)
    r.zoom(None, 0.8)
    # 6 - F is clear: its view and the Inspector
    r.say(6)
    st.select_cams([cam_f.uid])
    stage.settle(2)
    r.zoom(through, 0.8, margin=4)
    r.hold(1.4)
    r.zoom(None, 0.7)
    house2 = page.house_v.v.text()
    print('Inspector housings row after the move:', house2, '| blind', shown('Blind in ROI'), flush=True)
    r.expect(house2, 'none', "F's housings after the move")
    r.expect(shown('Blind in ROI'), '0.010', 'blind after the move')
    r.callout(page.house_v, '', dur=2.2, pad=4)
    r.figure('Blind in ROI', float(shown('Blind in ROI')), 'm²', 3, dur=max(2.8, r.speech_left() + 0.4), src=rows['Blind in ROI'],
             start=0.240, note='RD1 5 cm back')
    r.finish_speech(0.4)
    # 7 - everything, everywhere
    r.say(7)
    w.set_split('single')
    stage.settle(2)
    vp1 = w.viewarea.panes[0].vp3d
    vp1.cam.az, vp1.cam.el = -30.0, 22.0
    w.viewarea.frame_all()
    r.crossfade(0.5)
    stage.orbit(r, vp1, 70, max(4.0, r.speech_left() + 0.6), d_el=6)
    r.finish_speech(0.8)


def stage_ease(t):
    return t * t * (3 - 2 * t)
