"""Website clip 'requirements': the data rig (six cameras, the roof lidar, five radars, two GNSS antennas and a
dual-antenna INS) signed off against written requirements. Analysis ▸ Requirements scorecard; Add from the plan (the
perception range's required ranges, the recording shift, the blur limit, the power runtime …) and a template; Run
computes each study once; the failing "camera–lidar misalignment at most 1 ms" row's evidence (one trigger line leaves
a camera 12.4 ms off the lidar sweep; the suggested lines 0.94 ms); the sync and timing page opened from the evidence,
its suggested trigger lines applied; back on the scorecard the row passes at once (a cheap metric); a second run
brings the studies up to date: the verdict tiles and the status chips. Called by demo/clips.py (--clips
requirements)."""
from __future__ import annotations

from PyQt6.QtCore import QPoint, QPointF, QRectF


def _template(r, pg, start):
    """Open the Templates menu like a user (without its blocking exec) and pick the entry that starts with start."""
    r.click(pg.b_templates, real=False)
    pg._fill_templates()
    m = pg.tmenu
    m.popup(pg.b_templates.mapToGlobal(QPoint(0, pg.b_templates.height())))
    r.hold(0.5)
    act = next(a for a in m.actions() if a.text().startswith(start))
    ar = m.actionGeometry(act)
    r.move_to(m, QPointF(ar.center().x() - ar.width() * 0.2, ar.center().y()), 0.6)
    m.setActiveAction(act)
    r.hold(0.35)
    r.ripple()
    r.hold(0.15)
    m.hide()
    act.trigger()
    r.hold(0.3)


def _row_of(pg, metric):
    return next(k for k, x in enumerate(pg.res.items) if x.item['metric'] == metric)


def _click_row(r, pg, row):
    """Click the row's status cell (inside the zoomed view of the row and its evidence)."""
    t = pg.table
    t.scrollToItem(t.item(row, 1))
    r.tick(2)
    rect = t.visualItemRect(t.item(row, 6))
    r.click(t.viewport(), QPointF(rect.center().x(), rect.center().y()), 0.6)


def _row_rect(r, pg, row):
    """The row's value, margin and status cells in window coordinates."""
    t = pg.table
    vp = r.wrect(t.viewport())
    a, b = t.visualItemRect(t.item(row, 3)), t.visualItemRect(t.item(row, 6))
    return QRectF(vp.x() + a.x(), vp.y() + a.y(), b.right() - a.x(), a.height())


def _evidence_rect(r, pg):
    """The evidence panel: its head, the study's figures and the detail line."""
    top = r.wrect(pg.b_open)
    tab = r.wrect(pg.ev_table)
    return QRectF(tab.x(), top.y(), tab.width(), tab.y() + 150 - top.y())


def _run(r, pg, speed=1):
    before = pg.full
    r.click(pg.b_run)
    r.wait_for(lambda: not pg.busy and pg.full is not None and pg.full is not before, timeout=180, speed=speed)


def clip(stage, r):
    w = stage._nav_window()
    r.win = w
    r.chapter('Requirements', 'The rig signed off against the programme\'s requirements')
    r.menu('Analysis', 'Requirements scorecard')
    pg = w.page('requirements')
    r.hold(0.2)
    # requirements implied by the plan's own settings, then a typical one from the templates
    r.click(pg.b_plan)
    r.hold(0.4)
    _template(r, pg, 'Pedestrians resolved to 40 m')
    r.hold(0.2)
    # every study a requirement needs, once, in the background
    _run(r, pg, speed=3)
    r.hold(0.4)
    # a failing requirement and the figures behind its value: one trigger line leaves a camera 12 ms off the sweep
    k = _row_of(pg, 'timing.misalignment_ms')
    _click_row(r, pg, k)
    r.zoom(_row_rect(r, pg, k).united(_evidence_rect(r, pg)), 0.6, margin=10)
    r.hold(1.6)
    r.zoom(None, 0.5)
    # the design change, from the evidence's own link: the trigger lines the sync plan suggests
    r.click(pg.b_open, dur=0.6)
    tm = w.page('timing')
    stage.settle(1)
    r.hold(0.2)
    r.click(tm.b_suggest, dur=0.6)
    r.hold(0.4)
    r.click(tm.b_apply, dur=0.6)
    stage.settle(1)
    r.hold(0.3)
    # back on the scorecard: the row passes at once (a cheap metric); the study values wait for a run
    r.menu('Analysis', 'Requirements scorecard')
    r.hold(0.4)
    r.zoom(_row_rect(r, pg, k).united(_evidence_rect(r, pg)), 0.6, margin=10)
    r.hold(1.5)
    r.zoom(None, 0.5)
    _run(r, pg, speed=3)
    r.hold(0.3)
    tiles = r.wrect(pg.tiles['verdict']).united(r.wrect(pg.tiles['open']))
    table = r.wrect(pg.table)
    r.zoom(QRectF(tiles.x(), tiles.y(), tiles.width(), table.height() * 0.55 + (table.y() - tiles.y())), 0.6,
           margin=10)
    r.hold(1.6)
    r.zoom(None, 0.5)
    r.hold(0.4)
