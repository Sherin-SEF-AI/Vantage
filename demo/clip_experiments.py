"""Website clip 'experiments': a designed experiment on the sample, run in the app.

Analysis ▸ Experiments… with F's pitch (25 … 45°, 3 levels) and F's two lenses as factors and the blind ground and
the ROI seen by two or more cameras as responses: a full factorial of 6 points. Its surface: with the narrow lens
(h03r2) 1.0–1.6 m² of the ROI is blind at every pitch, with the fisheye 0.01–0.135 m². Then the lens is removed and
R's pitch added from the Add menu, both pitches drawn normal 40° ± 1° (a mounting tolerance), and the design switched
to Sobol (Saltelli, N = 128: 512 points): R's pitch carries about three quarters of the blind ground's variance, F's a
quarter, each with its bootstrap interval. The best point is added as a variant (one undo step). The project is never
saved.
"""
from __future__ import annotations

from PyQt6.QtCore import QPointF, QRectF
from PyQt6.QtWidgets import QApplication, QMenu


def setup(p):
    """The experiment as it stands when the page opens: F's pitch and lens, two responses, a full factorial, on the
    6-cam variant itself (so it stays on it when its best point becomes the active variant)."""
    from vantage.core import experiments as EX
    v = p.variant
    F = v.by_tag('F')
    f1 = EX.new_factor(p, v, 'camera', F.uid, 'pitch', [])
    f1.update(lo=25.0, hi=45.0, levels=3)
    f2 = EX.new_factor(p, v, 'camera', F.uid, 'lens', [f1['id']])
    p.analysis.experiments = EX.ExperimentSettings.from_dict({
        'name': 'Pitch study', 'variant': v.uid, 'factors': [f1, f2], 'design': 'factorial', 'levels': 3,
        'sobol_n': 128, 'responses': [{'metric': 'coverage.blind_m2'}, {'metric': 'coverage.multi_pct'}]}).to_dict()


def hide_menus():
    for m in QApplication.topLevelWidgets():
        if isinstance(m, QMenu) and m.isVisible():
            m.hide()


def pick_from_submenu(r, btn, menu, group, text):
    """Press a button that opens a menu (without its blocking exec), walk into a submenu and trigger an entry."""
    r.click(btn, None, 0.45, real=False, hold=0.1)
    menu.popup(btn.mapToGlobal(btn.rect().bottomLeft()))
    r.hold(0.3)
    g = next(a for a in menu.actions() if a.text() == group)
    gr = menu.actionGeometry(g)
    r.move_to(menu, QPointF(gr.center().x() - gr.width() * 0.2, gr.center().y()), 0.4)
    menu.setActiveAction(g)
    sub = g.menu()
    sub.popup(menu.mapToGlobal(gr.topRight()))
    r.hold(0.3)
    act = next(a for a in sub.actions() if a.text() == text)
    ar = sub.actionGeometry(act)
    r.move_to(sub, QPointF(ar.center().x() - ar.width() * 0.2, ar.center().y()), 0.4)
    sub.setActiveAction(act)
    r.hold(0.2)
    r.ripple()
    r.hold(0.1)
    hide_menus()
    act.trigger()
    r.hold(0.15)


def pick_combo(r, combo, data, dur=0.6):
    """Open a combo box's list like a user, move to the item with this data and pick it (activated, as a user)."""
    r.click(combo, None, dur, real=False)
    combo.showPopup()
    r.hold(0.4)
    view = combo.view()
    i = combo.findData(data)
    rect = view.visualRect(view.model().index(i, 0))
    r.move_to(view.viewport(), QPointF(rect.center().x() - rect.width() * 0.2, rect.center().y()), 0.45)
    view.setCurrentIndex(view.model().index(i, 0))
    r.hold(0.25)
    r.ripple()
    combo.setCurrentIndex(i)
    combo.hidePopup()
    combo.activated.emit(i)
    r.hold(0.2)


def click_tab(r, tabs, title):
    bar = tabs.tabBar()
    i = next(k for k in range(tabs.count()) if tabs.tabText(k) == title)
    r.click(bar, QPointF(bar.tabRect(i).center()), 0.5, real=False)
    tabs.setCurrentIndex(i)
    r.hold(0.2)


def select_row(r, table, row):
    rect = table.visualRect(table.model().index(row, 0))
    r.click(table.viewport(), QPointF(rect.center()), 0.5, real=False)
    table.selectRow(row)
    r.hold(0.2)


def clip(stage, r):
    w = stage.window(setup)
    r.win = w
    pg = w.page('experiments')                  # built before it shows, its results on the Surface tab
    pg.tabs.setCurrentIndex(next(k for k in range(pg.tabs.count()) if pg.tabs.tabText(k) == 'Surface'))
    r.chapter('Experiments', 'Which design inputs drive the figures, and the best design found')
    r.menu('Analysis', 'Experiments')
    stage.settle(2)
    r.hold(0.4)

    def ran(prev):
        return pg.result is not None and pg.result is not prev and not pg.busy and not pg.est_busy

    def results():
        a, b = r.wrect(pg.tabs), r.wrect(pg.tiles['points'])
        return QRectF(a.x(), b.y(), a.width(), a.bottom() - b.y())

    # F's pitch × F's lens: a full factorial of 6 points, and its surface
    r.click(pg.b_run)
    r.wait_for(lambda: ran(None), timeout=120)
    if pg.result is None or pg.result.n != 6:
        raise RuntimeError('the factorial never finished')
    r.zoom(results(), 0.6, margin=8)
    r.hold(1.5)
    r.zoom(None, 0.4)
    # the lens out, R's pitch in: both pitches drawn normal 40° ± 1°, and Sobol over them
    lens = next(i for i, f in enumerate(pg.settings.factors) if f['attr'] == 'lens')
    select_row(r, pg.ftable, lens)
    r.click(pg.b_del_f)
    pick_from_submenu(r, pg.b_add_f, pg.fmenu, 'Camera R', 'R · pitch')
    pick_combo(r, pg.design, 'sobol', dur=0.5)
    first = pg.result
    r.click(pg.b_run)
    r.wait_for(lambda: ran(first), timeout=240, speed=5)
    if pg.result is first or pg.result is None or pg.result.design.method != 'sobol':
        raise RuntimeError('the Sobol run never finished')
    r.zoom(results(), 0.6, margin=8)
    click_tab(r, pg.tabs, 'Sensitivity')
    r.hold(1.9)
    # the best point found (picked after the run), added as a variant
    click_tab(r, pg.tabs, 'Pareto')
    r.click(pg.b_variant)
    r.hold(0.3)
    m = r.wrect(w.msg_label)
    r.zoom(QRectF(m.x(), m.y() - 120, max(m.width(), 640), m.height() + 125), 0.6, margin=8)
    r.hold(1.2)
    r.zoom(None, 0.4)
    r.cursor_on = False
    r.hold(0.4)
