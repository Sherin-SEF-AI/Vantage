"""Website clip 'power': the power budget of the data rig (six cameras, the roof lidar, five radars, two GNSS antennas
and a dual-antenna INS). The page opens from Analysis ▸ Power budget; a 5 A fuse typed on the compute circuit is
over its current (critical); the 24 V bus carries the same power at half the current, and the fuse holds."""
from __future__ import annotations

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QLineEdit


def _type_cell(r, table, item, text):
    """Double-click a table cell like a user, type into its editor and press Return."""
    vp = table.viewport()
    rect = table.visualItemRect(item)
    r.click(vp, QPointF(rect.center()), 0.7, real=False)
    table.editItem(item)
    r.tick(3)
    ed = next((e for e in vp.findChildren(QLineEdit) if e.isVisible()), None)
    if ed is None:                                   # no editor (it never happens on screen): set the text
        item.setText(text)
        r.hold(0.4)
        return
    ed.selectAll()
    r.hold(0.3)
    for ch in text:
        QTest.keyClicks(ed, ch)
        r.tick(4)
    QTest.keyClick(ed, Qt.Key.Key_Return)
    r.tick(2)


def clip(stage, r):
    w = stage._nav_window()
    r.win = w
    r.chapter('Power', 'What the rig draws, and whether the supply carries it')
    r.menu('Analysis', 'Power budget')
    pg = w.page('power')
    stage.settle(2)
    r.hold(2.2)
    # a 5 A fuse on the compute circuit: the ECU and the six cameras it powers draw 5.75 A
    row = next(i for i in range(pg.circ.rowCount()) if pg.circ.item(i, 0).text() == 'Compute')
    _type_cell(r, pg.circ, pg.circ.item(row, 2), '5')
    r.wait_for(lambda: pg.result.circuit('compute')['fuse_a'] == 5.0, timeout=10)
    r.move_to(pg.circ, QPointF(pg.circ.width() * 0.5, pg.circ.height() - 12), 0.4)    # off the figures
    low =r.wrect(pg.circ.parentWidget()).united(r.wrect(pg.find_host.parentWidget().parentWidget()))
    r.zoom(low, 0.9, margin=12)
    r.hold(2.4)
    r.zoom(None, 0.6)
    # the same power on a 24 V bus: half the current, and the fuse holds
    r.click(pg.bus.buttons['24'])
    r.wait_for(lambda: pg.result.settings.bus_v == 24.0, timeout=10)
    r.hold(1.2)
    top = r.wrect(pg.tiles['load']).united(r.wrect(pg.tiles['heat'])).united(r.wrect(pg.bar))
    r.zoom(QRectF(top.x(), top.y(), top.width(), top.height() + 150), 0.9, margin=12)
    r.hold(2.4)
    r.zoom(None, 0.6)
    r.zoom(low, 0.9, margin=12)
    r.hold(2.2)
    r.zoom(None, 0.6)
    r.hold(0.5)
