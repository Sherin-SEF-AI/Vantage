"""Website clip 'weather': rain, fog, snow and night on the perception range page (the sample's data rig).

Analysis ▸ Weather and night… opens the perception range page on its Weather block. Night on an unlit road shrinks
the envelopes inside the dashed clear ones (pedestrians resolved to 19 m ahead instead of 66 m, nothing behind), the
panel compares every condition; heavy rain and its Sensors tab: the long-range radar 250 → 195 m, the roof lidar
90 → 76 m. The project is never saved.
"""
from __future__ import annotations

from PyQt6.QtCore import QPointF, QRectF, Qt


def pick_combo(r, combo, text, dur=0.6):
    """Open a combo box's list like a user, move to the item (its data, else the start of its text) and pick it."""
    r.click(combo, None, dur, real=False)
    combo.showPopup()
    r.hold(0.45)
    view = combo.view()
    i = combo.findData(text)
    if i < 0:
        i = next(k for k in range(combo.count()) if combo.itemText(k).startswith(text))
    rect = view.visualRect(view.model().index(i, 0))
    r.move_to(view.viewport(), QPointF(rect.center().x() - rect.width() * 0.2, rect.center().y()), 0.5)
    view.setCurrentIndex(view.model().index(i, 0))
    r.hold(0.3)
    r.ripple()
    combo.setCurrentIndex(i)
    combo.hidePopup()
    r.hold(0.2)


def click_tab(r, tabs, title):
    bar = tabs.tabBar()
    i = next(k for k in range(tabs.count()) if tabs.tabText(k) == title)
    r.click(bar, QPointF(bar.tabRect(i).center()), 0.5, real=False)
    tabs.setCurrentIndex(i)
    r.hold(0.2)


def pick_row(r, table, cid):
    """Click the comparison table's row of a condition (its sensors show in the Sensors tab)."""
    row = next(i for i in range(table.rowCount()) if table.item(i, 0).data(Qt.ItemDataRole.UserRole) == cid)
    rect = table.visualRect(table.model().index(row, 0))
    r.click(table.viewport(), QPointF(rect.center()), 0.5, real=False)
    table.setCurrentCell(row, 0)
    r.hold(0.3)


def clip(stage, r):
    w = stage._rig_window()
    r.win = w
    r.chapter('Weather and night', 'How far the rig still sees in rain, fog and at night')
    r.menu('Analysis', 'Weather and night')
    pg = w.page('perception')

    def ready(cid):
        from vantage.core import perception as PR
        res = pg.ctl.result
        return (res is not None and not pg.ctl.busy and PR._condition(res).id == cid and pg.wres is not None
                and pg.wres.selected == cid and not pg._wbusy)

    r.wait_for(lambda: ready('clear'), timeout=90)
    if not ready('clear'):
        raise RuntimeError('the perception range never finished')
    r.hold(0.6)

    def need(cid):
        r.wait_for(lambda: ready(cid), timeout=90)
        if not ready(cid):
            raise RuntimeError(f'{cid} never computed')
        r.hold(0.3)

    def panel():
        t = r.wrect(pg.wtabs)
        return QRectF(t.x(), t.y() - 30, t.width(), t.height() + 40)

    # night on an unlit road: the envelopes shrink inside the dashed clear ones
    pick_combo(r, pg.cond, 'night')
    need('night')
    r.zoom(r.wrect(pg.map), 0.8, margin=8)
    r.hold(1.6)
    # every condition compared, straight ahead
    r.zoom(panel(), 0.8, margin=10)
    r.hold(2.0)
    r.zoom(None, 0.6)
    # heavy rain: its row, then each lidar and radar against clear
    pick_combo(r, pg.cond, 'heavy_rain')
    need('heavy_rain')
    pick_row(r, pg.wcond, 'heavy_rain')
    click_tab(r, pg.wtabs, 'Sensors')
    r.zoom(panel(), 0.8, margin=10)
    r.hold(2.0)
    r.zoom(None, 0.6)
    r.cursor_on = False
    r.hold(0.4)
