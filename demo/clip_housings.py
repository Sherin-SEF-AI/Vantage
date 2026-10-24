"""Website clip 'housings': sensor housings block the cameras.

On the sample's data rig the front long-range radar RD1 sits just in front of the bumper, below camera F, and F looks
40° down through its H05R2 fisheye: the radar's housing fills the bottom of F's image (13.7 % of it) and hides ground
ahead of the bumper (the blind ground in the ROI goes from 0.010 m² with the body and rig alone to 0.240 m²).

The clip: a close-up of the bumper beside the view through F (the body and what blocks F in blue), the Inspector's
Housings row, Analysis ▸ Housings block cameras switched off (the body and rig alone: the radar leaves F's view, the
blind ground drops) and on again, then RD1 selected and moved 5 cm back behind the fascia in the Inspector until F is
clear. The project is never saved.
"""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QPointF, QRectF

BACK_M = 0.05           # RD1 moved this far back: F's image is clear (13.7 % → none, checked with core.housings)


def clip(stage, r):
    w = stage._rig_window()
    st = w.store
    v = st.variant
    cam_f = v.by_tag('F')
    rd1 = next(s for s in v.sensors if s.tag == 'RD1')
    w.set_split('side')
    stage.settle(3)
    left, right = w.viewarea.panes[0], w.viewarea.panes[1]
    right.set_kind('through')
    right.cam.uid = cam_f.uid                         # the right view keeps looking through F whatever is selected
    vp = left.vp3d
    vp.cam.target = np.array([cam_f.x - 0.25, 0.0, 0.45])
    vp.cam.dist, vp.cam.az, vp.cam.el = 3.4, 38.0, 20.0
    vp.update()
    w.bottom.show_tab('coverage')
    st.select_cams([cam_f.uid])
    stage.settle(4)
    for _ in range(10):
        stage.app.processEvents()
    r.win = w
    r.cursor_on = False
    page = w.inspector.pages['camera']
    rows = w.bottom.coverage.rows

    def through():                                     # the lower part of the view through F: the body and RD1
        g = r.wrect(right)
        return QRectF(g.x(), g.y() + g.height() * 0.35, g.width(), g.height() * 0.65)

    def figures():
        f = r.wrect(rows['Blind in ROI']).united(r.wrect(rows['Seen by ≥ 2 cameras']))
        return QRectF(f.x(), f.y() - 30, max(f.width(), 360), f.height() + 40)

    def inspector_row():
        row = r.wrect(page.house_v)
        return QRectF(row.x() - 170, row.y() - 120, 420, 240).united(r.wrect(right))

    r.chapter('Housings', 'Sensor housings block the cameras')
    r.hold(1.2)
    # the Inspector: RD1 covers 13.7 % of F's image (the blue bump at the bottom of the view through F)
    row = r.wrect(page.house_v)
    r.move(QPointF(row.x() + row.width() * 0.5, row.y() + row.height() / 2), 0.7)
    r.zoom(inspector_row(), 0.8)
    r.hold(1.6)
    r.zoom(None, 0.6)
    # housings off: the body and the rig alone block F; the blind ground drops to 0.010 m²
    r.menu('Analysis', 'Housings block cameras')
    stage.settle(4)
    r.zoom(through(), 0.6, margin=4)
    r.hold(1.2)
    r.zoom(figures(), 0.6)
    r.hold(1.2)
    r.zoom(None, 0.5)
    # and on again
    r.menu('Analysis', 'Housings block cameras')
    stage.settle(4)
    r.zoom(through(), 0.6, margin=4)
    r.hold(1.0)
    r.zoom(None, 0.5)
    # RD1 moved 5 cm back behind the fascia in the Inspector: F is clear
    s, _, ok = vp.project([rd1.pos()])
    r.click(vp, QPointF(*s[0]), 0.7, real=False)
    st.select_obj(('sensor', rd1.uid))
    stage.settle(2)
    r.hold(0.4)
    spage = w.inspector.pages['sensor']
    fx = r.wrect(spage.px)
    r.move(QPointF(fx.x() + fx.width() * 0.7, fx.y() + fx.height() / 2), 0.6)
    from vantage.ui.sensorview import set_props
    x0 = rd1.x
    n = round(2.2 * stage.fps)
    for k in range(1, n + 1):
        t = k / n
        e = t * t * (3 - 2 * t)
        set_props(w.ctx, rd1.uid, 'Moved RD1 back', merge=True, x=round(x0 - BACK_M * e, 4))
        r.cur = np.array([fx.x() + fx.width() * (0.7 - 0.35 * e), fx.y() + fx.height() / 2])
        r.tick()
    stage.settle(4)
    r.hold(0.8)
    r.zoom(figures(), 0.6)
    r.hold(1.1)
    r.zoom(None, 0.5)
    # F again: no housing in its image
    st.select_cams([cam_f.uid])
    stage.settle(2)
    row = r.wrect(page.house_v)
    r.move(QPointF(row.x() + row.width() * 0.5, row.y() + row.height() / 2), 0.6)
    r.zoom(inspector_row(), 0.7)
    r.hold(1.6)
    r.zoom(None, 0.5)
    r.cursor_on = False
    r.hold(0.6)
