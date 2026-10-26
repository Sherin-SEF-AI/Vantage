"""Website clip 'terrain': road and body motion on the sample (six cameras).

Analysis ▸ Road and body motion…: a crest added 5 m ahead (a 300 m vertical curve into an 8 % fall): the road ahead
stays in view to about 20 m instead of the 60 m of the check, the side-on profile red beyond the crest; a speed bump
added (80 mm, 2 m ahead): first resolved 18 m ahead behind the crest; hard braking pitches the body 2.25° and moves
F's horizon 36 px (cornering rolls it 2.55°); then the road with the coverage heat map in the main 3D view.
The project is never saved.
"""
from __future__ import annotations

from PyQt6.QtCore import QRectF


def clip(stage, r):
    w = stage.window()
    r.win = w
    r.chapter('Road and body motion', 'The rig on a real road: crests, speed bumps, braking and cornering')
    r.menu('Analysis', 'Road and body motion')
    pg = w.page('terrain')
    ctl = w.ctx.terrain_ctl
    stage.settle(2)
    r.hold(0.5)

    def done(n):
        res = ctl.result
        return res is not None and not ctl.busy and len(res.features) == n and not ctl.timer.isActive()

    # a crest ahead: the road beyond it drops out of view
    r.click(pg.add_buttons['crest'])
    r.wait_for(lambda: done(1), timeout=90)
    r.hold(0.4)
    ch = r.wrect(pg.chart)
    r.zoom(QRectF(ch.x(), ch.y() - 30, ch.width(), ch.height() + 30), 0.8, margin=8)
    r.hold(2.2)
    r.zoom(None, 0.6)
    # a speed bump before it: first seen as the car approaches
    r.click(pg.add_buttons['bump'])
    r.wait_for(lambda: done(2), timeout=90)
    r.hold(0.3)
    stage.settle(2)
    pg.tabs.setCurrentIndex(1)
    tl = r.wrect(pg.tiles['blind']).united(r.wrect(pg.tiles['corner']))
    tb = r.wrect(pg.tabs)
    r.zoom(tl.united(QRectF(tb.x(), tb.y(), tb.width(), tb.height() * 0.5)), 0.8, margin=8)
    r.hold(2.4)
    r.zoom(None, 0.6)
    # body motion: hard braking against the calibration load, each forward camera's horizon
    pg.tabs.setCurrentIndex(0)
    btn = next(b for b in pg.findChildren(type(pg.b_run)) if b.text() == 'Hard braking')
    r.click(btn)
    stage.settle(1)
    r.hold(0.3)
    r.zoom(tl.united(r.wrect(pg.tabs)), 0.8, margin=8)
    r.hold(2.4)
    r.zoom(None, 0.6)
    # the road in the main 3D view, the coverage draped on it
    w.set_page('viewports')
    stage.settle(3)
    vp = stage.vp
    vp.cam.az, vp.cam.el = 25.0, 20.0
    vp.cam.dist *= 1.25
    vp.update()
    r.cursor_on = False
    stage.orbit(r, vp, 55, 4.0, d_el=6)
    r.hold(0.4)
