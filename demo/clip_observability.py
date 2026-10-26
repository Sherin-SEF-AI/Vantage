"""Website clip 'observability': how firmly the planned boards pin each sensor of the sample's data rig.

Calibrate ▸ Calibration observability…: the eight floor boards in plan with each camera ringed by its verdict, the
table of 1σ per axis (every camera's yaw ±0.13°, 1.3× the 0.1° requirement, because the boards are laid by hand),
F picked: its weakest direction, each board's share and the reasons; the suggestion "Use ±5 mm" (the boards laid to
a jig) and the run that follows with every camera within the requirement; then the fusion maps of lidar points per
pixel. The project is never saved.
"""
from __future__ import annotations

from PyQt6.QtCore import QPointF, QRectF


def click_tab(r, tabs, title):
    bar = tabs.tabBar()
    i = next(k for k in range(tabs.count()) if tabs.tabText(k) == title)
    r.click(bar, QPointF(bar.tabRect(i).center()), 0.5, real=False)
    tabs.setCurrentIndex(i)
    r.hold(0.2)


def clip(stage, r):
    w = stage._rig_window()
    r.win = w
    r.chapter('Calibration observability', 'How firmly the planned boards will pin each sensor')
    r.menu('Calibrate', 'Calibration observability')
    pg = w.page('observability')
    r.wait_for(lambda: pg.result is not None and not pg.busy, timeout=90)
    if pg.result is None:
        raise RuntimeError('the observability study never finished')
    r.hold(1.1)                                 # the boards in plan, each camera ringed by its verdict
    # the table: 1σ per axis against the requirement
    t = r.wrect(pg.cams)
    tl = r.wrect(pg.tiles['cams'])
    r.zoom(QRectF(t.x(), tl.y(), t.width(), t.bottom() - tl.y()), 0.8, margin=10)
    r.hold(1.9)
    # F: its weakest direction, the boards and the reasons
    row = next(i for i in range(pg.cams.rowCount()) if pg.cams.item(i, 0).text().startswith('F'))
    vp = pg.cams.viewport()
    rect = pg.cams.visualRect(pg.cams.model().index(row, 0))
    r.click(vp, QPointF(rect.center()), 0.6, real=False)
    pg.cams.selectRow(row)
    stage.settle(1)
    d = r.wrect(pg.det_head.parentWidget())
    r.zoom(d, 0.8, margin=10)
    r.hold(2.3)
    # the boards laid to a jig: Use ±5 mm
    btn = next((b for b in pg.why_host.findChildren(type(pg.b_run)) if b.text().startswith('Use ±')), None)
    if btn is None:
        from PyQt6.QtWidgets import QPushButton
        btn = next(b for b in pg.why_host.findChildren(QPushButton) if b.text().startswith('Use ±'))
    before = pg.result
    r.click(btn)
    r.wait_for(lambda: pg.result is not before and pg.result is not None and not pg.busy, timeout=60)
    r.hold(0.4)
    r.zoom(QRectF(t.x(), tl.y(), t.width(), t.bottom() - tl.y()), 0.8, margin=10)
    r.hold(1.9)
    r.zoom(None, 0.6)
    # the fusion maps: where lidar points land in each image
    click_tab(r, pg.view_tabs, 'Fusion')
    stage.settle(1)
    r.hold(0.4)
    r.zoom(r.wrect(pg.view_tabs), 0.8, margin=8)
    r.hold(1.7)
    r.zoom(None, 0.6)
    r.cursor_on = False
    r.hold(0.4)
