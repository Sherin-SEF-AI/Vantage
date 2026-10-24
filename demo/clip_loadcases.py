"""Website clip 'loadcases': Analysis ▸ Load cases… on the sample's data rig, with the suspension's typical values
(a compact sedan scaled to the model: the dialog says so).

The clip: the cases (rig only, the driver, driver and operator, four occupants, full with luggage) with each corner's
drop, the pitch and roll and the worst move; the Loads tab (seats, boot, roof, equipment and the rig's own components);
the Suspension tab (springs, motion ratios, tyres, bump stops per corner); back on Cases, the driver alone rolls the
body and every camera and sensor moves against the calibration load; Run adds the ground figures (the stitched
ground's shift and the seam ghost); the full load moves the rear camera most.
"""
from __future__ import annotations

from PyQt6.QtCore import QPoint, QPointF, QRectF


def _row(r, table, row, dur=0.6):
    """Click a table row like a user."""
    rect = table.visualItemRect(table.item(row, 0))
    r.click(table.viewport(), QPointF(rect.center().x(), rect.center().y()), dur)


def _find(table, text):
    return next(i for i in range(table.rowCount()) if table.item(i, 0) and table.item(i, 0).text().startswith(text))


def clip(stage, r):
    w = stage._rig_window()
    r.win = w
    r.chapter('Load cases', 'From the suspension and the real loads')
    r.menu('Analysis', 'Load cases')
    d = w._load_dialog
    d.move(w.mapToGlobal(QPoint((w.width() - d.width()) // 2, max(36, (w.height() - d.height()) // 2))))
    for _ in range(6):
        stage.app.processEvents()
    r.hold(1.3)
    bar = d.tabs.tabBar()
    # the loads: seats, boot, roof, equipment, and the rig itself
    r.click(bar, bar.tabRect(1).center(), dur=0.5)
    r.hold(0.5)
    r.zoom(r.wrect(d.itable).united(r.wrect(d.rtable)), 0.6, margin=12)
    r.hold(1.5)
    r.zoom(None, 0.4)
    # the suspension: each corner's spring, motion ratio, tyre and bump stop (typical values until you enter yours)
    r.click(bar, bar.tabRect(2).center(), dur=0.5)
    r.hold(0.4)
    top = r.wrect(d.fields['kerb_mass']).united(r.wrect(d.stable))
    r.zoom(QRectF(top.x() - 80, top.y() - 30, top.width() + 160, top.height() + 40), 0.6, margin=8)
    r.hold(1.7)
    r.zoom(None, 0.4)
    # the driver alone: the body rolls and every camera and sensor moves against the calibration load
    r.click(bar, bar.tabRect(0).center(), dur=0.5)
    r.hold(0.3)
    _row(r, d.ctable, _find(d.ctable, 'Driver'), 0.5)
    r.hold(0.3)
    r.zoom(r.wrect(d.dtable), 0.6, margin=12)
    r.hold(1.6)
    r.zoom(None, 0.4)
    # Run: the ground figures of every case
    r.click(d.b_run, dur=0.5)
    r.wait_for(lambda: not d.busy and d.sres is not None, timeout=120, speed=1)
    r.hold(0.4)
    r.zoom(r.wrect(d.ctable), 0.6, margin=10)
    r.hold(1.9)
    r.zoom(None, 0.4)
    # the full load: the rear camera moves most
    _row(r, d.ctable, _find(d.ctable, 'Full'), 0.5)
    r.hold(0.3)
    r.zoom(r.wrect(d.dtable), 0.6, margin=12)
    r.hold(1.7)
    r.zoom(None, 0.4)
    r.cursor_on = False
    r.hold(0.4)
    d.accept()
