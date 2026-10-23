"""Website clip 'brackets': Analysis ▸ Mount brackets on the sample. Which cameras can take a printed bracket, the
left mirror camera moved out to the standoff a bracket needs (its mirrored partner follows), Generate at the default
0.35 mm grid, the bracket orbited in the preview, and its checks."""
from PyQt6.QtCore import QPointF, QRectF


def _row(pg, tag):
    t = pg.table
    return next(r for r in range(t.rowCount()) if t.item(r, 0).text() == tag)


def clip(stage, r):
    w = stage.window()
    r.win = w
    r.chapter('Brackets', 'A printable bracket for every camera')
    r.menu('Analysis', 'Mount brackets')
    pg = w.page('brackets')
    ctl = pg.ctl
    n = len([c for c in w.store.variant.cameras if c.visible])
    r.wait_for(lambda: len(ctl.pre) == n, timeout=60)
    r.hold(1.0)
    # the statuses: most of the sample's cameras sit too close to the body for a surface bracket
    t = r.wrect(pg.table)
    r.zoom(QRectF(t.x(), t.y(), t.width(), 26 + 22 * (n + 1)), 0.8, margin=40)
    r.hold(1.6)
    # the left mirror camera: select it, read why, move it out (the right one follows)
    tb = pg.table
    rect = tb.visualItemRect(tb.item(_row(pg, 'ML'), 1))
    r.click(tb.viewport(), QPointF(rect.center()))
    r.hold(0.5)
    det = r.wrect(pg.state)
    r.zoom(QRectF(det.x(), det.y() - 30, det.width(), 120), 0.7, margin=30)
    r.hold(1.6)
    r.click(pg.b_move)
    ml = w.store.variant.by_tag('ML')
    r.wait_for(lambda: ctl.bracket(ml.uid) is not None and ctl.bracket(ml.uid).status == 'ok', timeout=30)
    r.hold(0.6)
    r.zoom(None, 0.6)
    # generate: the cameras that can take one are meshed (a job with progress; shown faster than real time)
    r.click(pg.b_gen)
    r.wait_for(lambda: not ctl.busy and ctl.result is not None, timeout=180, speed=5)
    r.hold(0.8)
    # orbit the bracket in the preview
    v = pg.view
    cx, cy = v.width() / 2, v.height() / 2
    r.drag(v, QPointF(cx + 90, cy + 20), QPointF(cx - 50, cy + 30), dur=1.4)
    r.move_to(v, QPointF(v.width() * 0.88, v.height() * 0.84), 0.4)
    r.hold(1.4)                                                       # settled (the poster frame falls here)
    # its checks
    sa = r.wrect(pg.check_host.parentWidget().parentWidget())        # the checks' scroll area
    r.zoom(QRectF(sa.x(), sa.y() - 24, sa.width(), 190), 0.8, margin=24)
    r.hold(2.4)
    r.zoom(None, 0.6)
    r.move_to(v, QPointF(v.width() * 0.82, v.height() * 0.8), 0.5)
    r.hold(0.8)
