"""Website clip 'critical': critical targets and mounting envelopes on the sample's data rig.

Analysis ▸ Critical targets…: a child 2 m ahead, a pedestrian 10 m ahead, an overhead traffic light 15 m ahead and a
kerb by the right front wheel added from the presets; the page measures them by itself. The pedestrian: only F
resolves it (153 px, 146 lidar points), and F's envelope against the ±10 mm, ±1° install tolerance (pitched 8.3°
further down it loses the pedestrian; moved back it meets the bumper after 17 mm). The traffic light: in no camera's
image, a serious finding. The project is never saved.
"""
from __future__ import annotations

from PyQt6.QtCore import QPointF, QRectF
from PyQt6.QtWidgets import QApplication, QMenu

TARGETS = ('Child 2 m ahead', 'Pedestrian 10 m ahead', 'Overhead traffic light 15 m ahead',
           'Kerb by the right front wheel')


def pick_from_menu(r, btn, menu, text):
    """Press a button that opens a menu (without its blocking exec), move to the entry and trigger it."""
    r.click(btn, None, 0.45, real=False, hold=0.1)
    menu.popup(btn.mapToGlobal(btn.rect().bottomLeft()))
    r.hold(0.3)
    act = next(a for a in menu.actions() if a.text() == text)
    ar = menu.actionGeometry(act)
    r.move_to(menu, QPointF(ar.center().x() - ar.width() * 0.2, ar.center().y()), 0.4)
    menu.setActiveAction(act)
    r.hold(0.2)
    r.ripple()
    r.hold(0.1)
    for m in QApplication.topLevelWidgets():
        if isinstance(m, QMenu) and m.isVisible():
            m.hide()
    act.trigger()
    r.hold(0.15)


def select_row(r, table, row):
    vp = table.viewport()
    rect = table.visualRect(table.model().index(row, 1))
    r.click(vp, QPointF(rect.center()), 0.5, real=False)
    table.selectRow(row)
    r.hold(0.2)


def clip(stage, r):
    w = stage._rig_window()
    r.win = w
    r.chapter('Critical targets', 'What the rig must never miss, and how far each mount may be off')
    r.menu('Analysis', 'Critical targets')
    pg = w.page('critical')
    ctl = pg.ctl
    stage.settle(2)
    r.hold(0.2)
    for text in TARGETS:
        pick_from_menu(r, pg.b_add, pg.add_menu, text)
    r.wait_for(lambda: ctl.result is not None and not ctl.busy and len(ctl.result.targets) == len(TARGETS)
               and not ctl.timer.isActive(), timeout=90)
    if ctl.result is None or len(ctl.result.targets) != len(TARGETS):
        raise RuntimeError('the critical targets were never measured')
    r.click(pg.b_fit)                           # the plan framed on the car and every target
    r.hold(0.6)
    left = r.wrect(pg.targets)
    r.zoom(QRectF(left.x(), left.y() - 40, left.width(), left.height() * 0.6 + 40), 0.6, margin=12)
    r.hold(1.3)
    r.zoom(None, 0.5)
    # the pedestrian: only F resolves it, and F's mounting envelope
    names = [t['name'] for t in ctl.settings.targets]
    select_row(r, pg.targets, names.index('Pedestrian 10 m ahead'))
    stage.settle(1)
    r.hold(0.4)
    lk, bars = r.wrect(pg.looks), r.wrect(pg.bars)
    r.zoom(QRectF(lk.x(), lk.y() - 30, lk.width(), bars.bottom() - lk.y() + 70), 0.7, margin=8)
    r.hold(2.2)
    r.zoom(None, 0.5)
    # the overhead traffic light: in no camera's image
    select_row(r, pg.targets, names.index('Overhead traffic light 15 m ahead'))
    stage.settle(1)
    r.hold(0.4)
    fh = r.wrect(pg.find_host.parentWidget())
    r.zoom(QRectF(lk.x(), lk.y() - 30, lk.width(), lk.height() + 40).united(QRectF(fh.x(), fh.y(), fh.width(), 120)),
           0.7, margin=8)
    r.hold(1.8)
    r.zoom(None, 0.5)
    r.cursor_on = False
    r.hold(0.3)
