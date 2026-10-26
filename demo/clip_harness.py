"""Website clip 'harness': every cable of the sample's data rig routed along the body.

Analysis ▸ Cable harness…: the 18 routes from above (56.2 m in all), the front camera F's run picked in the table
(3.64 m over the bonnet to the bulkhead grommet, longer than its 3 m cable; the estimate along the axes said 2.95 m),
the side view and the 3D view turning, then the findings. The project is never saved.
"""
from __future__ import annotations

from PyQt6.QtCore import QPointF, QRectF
from tour import ease


def clip(stage, r):
    w = stage._rig_window()
    r.win = w
    r.chapter('Cable harness', 'Every cable routed along the body, against its link\'s reach')
    r.menu('Analysis', 'Cable harness')
    pg = w.page('harness')
    ctl = pg.ctl
    r.wait_for(lambda: ctl.result is not None and not ctl.busy and pg.table.rowCount() > 0, timeout=90)
    if ctl.result is None:
        raise RuntimeError('the harness never routed')
    r.hold(1.0)
    # the tiles: cables, total length against the estimate, longest and tightest run
    t0, t1 = r.wrect(pg.tiles['cables']), r.wrect(pg.tiles['entries'])
    r.zoom(QRectF(t0.x(), t0.y(), t1.right() - t0.x(), t0.height()), 0.7, margin=12)
    r.hold(2.0)
    r.zoom(None, 0.5)
    # F's run in the table: its route lights up on the body
    row = next(i for i in range(pg.table.rowCount()) if pg.table.item(i, 0).text().startswith('F'))
    vp = pg.table.viewport()
    rect = pg.table.visualRect(pg.table.model().index(row, 0))
    r.click(vp, QPointF(rect.center()), 0.6, real=False)
    pg.table.selectRow(row)
    r.hold(0.4)
    tv = r.wrect(pg.view).united(r.wrect(pg.table))
    r.zoom(tv, 0.8, margin=8)
    r.hold(2.4)
    r.zoom(None, 0.6)
    # from the side, then in 3D
    r.click(pg.mode.buttons['side'])
    r.hold(1.6)
    r.click(pg.mode.buttons['3d'])
    r.hold(0.3)
    v = pg.view                                 # turned as a drag turns it (drawn fast while it moves)
    r.cursor_on = False
    y0, n = v.yaw, round(2.4 * stage.fps)
    v._dragging = True
    for k in range(1, n + 1):
        v.yaw = y0 - 40.0 * ease(k / n)        # towards the side: the front camera's run over the bonnet
        v.update()
        r.tick()
    v._refine()
    r.hold(0.8)
    r.cursor_on = True
    # the findings: F's cable too short, every link within its limit
    fh = r.wrect(pg.find_host.parentWidget())
    r.zoom(QRectF(fh.x(), fh.y() - 34, fh.width(), fh.height() + 34), 0.7, margin=10)
    r.hold(2.4)
    r.zoom(None, 0.6)
    r.cursor_on = False
    r.hold(0.4)
