"""The 3D view's research tools (offscreen): the legend and live metrics card, the camera picture-in-picture, Alt-drag
of seams and guideline markers, the BEV-feature and stereo-sigma heat maps, the lidar–camera fusion rings, the
lidar timing wedges, the motion-warped footprints and the probe tool."""
import math

import numpy as np
import pytest
from PyQt6.QtCore import QPointF

from vantage.core import samples
from test_gui import app, win, pump, settle  # noqa: F401  (fixtures)


def _scene(win):
    return win.viewarea.panes[0].scene


def _data_rig(win, app):
    p = samples.tigor_sample_project()
    p.active_uid = next(v.uid for v in p.variants if v.sensors)
    win.store.set_project(p)
    settle(app, win)
    return p


class _VP:
    """The bits of a viewport the scene's Alt-drag reads: a ground point under a screen point."""

    def __init__(self, g):
        self.g = g

    def ground_at(self, x, y):
        return self.g


def test_legend_and_live_metrics(app, win):  # noqa: F811
    _data_rig(win, app)
    sc = _scene(win)
    mt = sc.hud_metrics()
    res = sc.result
    assert mt['blind_m2'] == pytest.approx(res.metrics['blind_m2'])
    assert mt['blind_pct'] == pytest.approx(100 * res.metrics['blind_m2'] / res.metrics['roi_area'])
    cov = res.grid.roi & (res.count > 0)
    assert mt['median_density'] == pytest.approx(float(np.median(res.best_dens[cov])))
    texts = [t for _, _, t in sc.hud_legend()]
    assert any('blind' in t for t in texts) and any('ROI' in t for t in texts)
    assert any('lidar rings' in t for t in texts)                    # the data rig has sensors
    win.ctx.set_heatmap('bevfeat')
    assert any('BEV pillar' in t for t in (t for _, _, t in sc.hud_legend()))
    win.ctx.set_heatmap('none')
    # the header folds the card (remembered)
    vp = win.viewarea.panes[0].vp3d
    vp.repaint()
    pump(app, 0.1)
    head = sc._hud_head
    assert head is not None
    was = sc._hud_open()
    assert sc.hud_click(head.center()) and sc._hud_open() is not was
    sc.hud_click(head.center())


def test_picture_in_picture_and_its_crosshair(app, win):  # noqa: F811
    _data_rig(win, app)
    sc = _scene(win)
    cam = win.store.variant.by_tag('F')
    img, intr = sc.pip_image(cam)
    m = win.store.project.module_for(cam)
    assert img.width() == sc.PIP_W and img.height() == round(sc.PIP_W * m.height_px / m.width_px)
    # a point straight down the optical axis lands at the principal point (scaled)
    from vantage.core.analysis import camera_frames
    R, W2O = camera_frames(cam)
    ahead = cam.pos() + W2O.T @ np.array([0.0, 0.0, 3.0])
    q = sc.pip_point(cam, intr, ahead, img.width(), img.height())
    assert q == pytest.approx((intr.cx * img.width() / m.width_px, intr.cy * img.height() / m.height_px), abs=0.6)
    assert sc.pip_point(cam, intr, cam.pos() - W2O.T @ np.array([0.0, 0.0, 3.0]), img.width(), img.height()) is None
    # pinned through its label; the pin and the cross unpin and close
    from PyQt6.QtCore import QRectF
    sc._label_rects = [(QRectF(10, 10, 30, 14), cam.uid)]
    assert sc.pip_click(QPointF(20, 16)) and sc.pip_pinned == cam.uid and win.store.sel_cams == [cam.uid]
    vp = win.viewarea.panes[0].vp3d
    vp.repaint()
    pump(app, 0.1)
    assert sc._pip_close is not None and sc.pip_click(sc._pip_close.center()) and sc.pip_pinned is None


def test_alt_drag_moves_a_seam_and_a_guideline_marker(app, win):  # noqa: F811
    _data_rig(win, app)
    sc = _scene(win)
    res = sc.result
    s = res.seams[0]
    cx, cy = res.centre
    r = 1.5
    g = (cx + r * math.cos(math.radians(s.angle)), cy + r * math.sin(math.radians(s.angle)), 0.0)
    d = sc._alt_pick(_VP(g), QPointF(0, 0))
    assert d is not None and d['kind'] == 'seam' and d['uid'] == f'{s.a}|{s.b}'
    d['hud0'] = sc.hud_metrics()
    sc.drag = d
    win.store.dragging = True
    a1 = s.angle + 4.0
    sc._alt_move(_VP((cx + r * math.cos(math.radians(a1)), cy + r * math.sin(math.radians(a1)), 0.0)), d,
                 QPointF(5, 5))
    assert win.store.variant.seams[d['uid']] == pytest.approx(round(a1, 1), abs=0.11)
    assert d['readout'].startswith(f'seam {d["tags"]} +4.0°')
    # a slow analysis waits for the release, and the badge says so rather than showing a stale +0
    live = win.ctx.analysis.drag_live_ms() is not None
    assert ('update on release' in d['readout']) is (not live) and ('blind' in d['readout'])
    sc.mouse_release(None, None)
    win.store.undo.undo()
    settle(app, win)
    assert d['uid'] not in win.store.variant.seams
    # a guideline distance marker
    from vantage.ui import guides
    win.ctx.set_overlay('guidelines', True)
    settle(app, win)
    gc = guides.controller(win.ctx)
    gs = gc.ground()
    ln = next(ln for ln in gs.lines if ln.kind == 'distance' and '2' in ln.label)
    g = (float(ln.pts[len(ln.pts) // 2, 0]), float(ln.pts[len(ln.pts) // 2, 1]), 0.0)
    d = sc._alt_pick(_VP(g), QPointF(0, 0))
    assert d is not None and d['kind'] == 'guide'
    sc.drag = d
    d0 = d['d0']
    sc._alt_move(_VP((g[0] + 0.3 * d['sign'], g[1], 0.0)), d, QPointF(5, 5))
    assert round(d0 + 0.3, 2) in gc.stored().distances and 'm →' in d['readout']
    sc.mouse_release(None, None)
    win.store.undo.undo()
    win.ctx.set_overlay('guidelines', False)
    settle(app, win)


def test_a_quick_analysis_keeps_up_during_a_drag(app, win, monkeypatch):  # noqa: F811
    _data_rig(win, app)
    an = win.ctx.analysis
    monkeypatch.setattr(an.result, 'ms', 40.0)
    assert an.drag_live_ms() == 40.0
    win.store.dragging = True
    try:
        an.timer.stop()
        an.request()
        assert an.timer.isActive() and an.timer.interval() == 120          # throttled: 2 × 40 ms, at least 120
        monkeypatch.setattr(an.result, 'ms', 900.0)
        an.timer.stop()
        an.request()
        assert an.drag_live_ms() is None and not an.timer.isActive() and an.stale
    finally:
        win.store.dragging = False
    an.request()
    assert an.timer.interval() == 30
    settle(app, win)


def test_research_heat_maps_and_overlays(app, win):  # noqa: F811
    _data_rig(win, app)
    sc = _scene(win)
    win.ctx.set_heatmap('bevfeat')
    gd = sc.ground()
    bm = sc.research('bev')
    assert gd.extent == pytest.approx((bm.x0, bm.y0, bm.x0 + bm.n * bm.cell_m, bm.y0 + bm.n * bm.cell_m))
    assert gd.image.width() == 2 * bm.n
    win.ctx.set_heatmap('stereo')
    from vantage.ui import maps
    gd = sc.ground()
    assert gd.extent == pytest.approx(maps.extent(sc.result))
    win.ctx.set_heatmap('none')
    vp = win.viewarea.panes[0].vp3d
    win.ctx.set_overlay('fusion', True)
    n_f = len(sc.research_lines())
    assert n_f > 10
    win.ctx.set_overlay('fusion', False)
    win.ctx.set_overlay('wedges', True)
    tr = sc.research_tris()
    assert tr and all(t.colors.shape == (len(t.tri), 4) for t in tr)
    win.ctx.set_overlay('wedges', False)
    # motion: the toolbar's speed warps the footprints (dashed, per camera); parked: nothing extra
    assert sc.research_lines() == []
    win.motion_speed.setValue(15.0)
    win.motion_yaw.setValue(10.0)
    pump(app, 0.05)
    assert win.ctx.motion == {'speed': 15.0, 'yaw': 10.0}
    lines = sc.research_lines()
    assert len(lines) > 6
    warp = sc.research('warp')
    e = next(iter(warp.values()))
    assert e['speed_mps'] == 15.0 and e['max_shift_m'] > 0
    win.motion_speed.setValue(0.0)
    win.motion_yaw.setValue(0.0)
    pump(app, 0.05)
    assert sc.research_lines() == []
    assert len(vp.provider.lines(vp)) > 0


def test_probe_tool_pins_and_reads_every_camera(app, win):  # noqa: F811
    from PyQt6.QtCore import QEvent, Qt
    from PyQt6.QtGui import QMouseEvent
    _data_rig(win, app)
    sc = _scene(win)
    sc.clear_probes()
    win.ctx.set_tool('probe')
    try:
        pr = sc.add_probe((5.0, 0.0), 'pedestrian')
        assert 'F' in pr['seen_by']
        row = next(r for r in pr['cameras'] if r['tag'] == 'F')
        assert row['bbox_px'][1] > row['bbox_px'][0] > 0          # a standing person: taller than wide
        from vantage.ui import research_overlays as RO
        card = RO.probe_card_lines(pr)
        assert card[0].startswith(pr['about'] if 'about' in pr else 'pedestrian') and any(' F ' in r or r.lstrip()
                                                                                          .startswith('F ')
                                                                                          for r in card[1:])
        lines = sc.research_lines()
        assert any(np.allclose(l.pts[0], win.store.variant.by_tag('F').pos(), atol=1e-6) for l in lines
                   if len(l.pts) == 2)
        # the probe follows a camera moved after it was pinned
        f = win.store.variant.by_tag('F')
        win.store.set_cam(f.uid, yaw=f.yaw + 30.0)
        settle(app, win)
        new = sc._probes_current()[0]['readout']
        assert next(r for r in new['cameras'] if r['tag'] == 'F')['bbox'] != row['bbox']
        win.store.undo.undo()
        settle(app, win)
        for k in ('box', 'board'):
            sc.add_probe((-4.0, 1.0), k)
        assert [p['kind'] for p in sc.probes][-2:] == ['box', 'board'] and len(sc.probes) <= 4
    finally:
        sc.clear_probes()
        win.ctx.set_tool('select')
    assert 'tool_probe' in win.actions_by_id
