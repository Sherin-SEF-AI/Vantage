"""The window after the 0.10.0 hardening pass: pages follow a switch of the active variant (the optimiser, the tolerance
study, robust perception, imaging radar), Duplicate variant keeps the hand-set seams, and the pages that drew controls
over each other at 1366 x 768 scroll instead."""
import time

import pytest
from PyQt6.QtCore import QPoint, QRect
from PyQt6.QtWidgets import (QAbstractButton, QAbstractSpinBox, QComboBox, QLabel, QLineEdit, QTableView, QWidget)

from test_gui import app, win, pump, settle, _wait  # noqa: F401 (fixtures)


def _variant(win, name):
    return next(v for v in win.store.project.variants if v.name == name)


def _switch(app, win, name):
    win.switch_variant(_variant(win, name).uid)
    settle(app, win)


def test_duplicate_variant_keeps_the_hand_set_seams(app, win):
    v = win.store.project.variant
    a, b = v.cameras[0].uid, v.cameras[1].uid
    win.store.edit(f'variant:{v.uid}', lambda p: p.variant.seams.__setitem__(f'{a}|{b}', 33.0), 'seam')
    win.duplicate_variant()
    new = win.store.project.variant
    assert new.uid != v.uid
    tags = {c.uid: c.tag for c in new.cameras}
    (key, ang), = new.seams.items()
    assert ang == 33.0 and [tags.get(u) for u in key.split('|')] == [v.cameras[0].tag, v.cameras[1].tag]


def test_the_optimiser_follows_the_active_variant(app, win):
    from vantage.core.optimiser import Optimiser, OptSettings, default_vars
    win.set_page('optimiser')
    pg = win.page('optimiser')
    pump(app, 0.3)
    _switch(app, win, '4-cam')
    v = win.store.project.variant
    assert pg.from_lbl.text() == 'from 4-cam'
    assert {x.uid for x in pg.vars} <= {c.uid for c in v.cameras} and pg.vars
    assert int(pg.c_hi.value()) == len(v.cameras)
    # a list of another variant's free variables is not taken as this one's (it varied nothing)
    other = _variant(win, '6-cam · v3')
    stale = default_vars(win.store.project, other)
    opt = Optimiser(win.ctx.analysis.engine.share(), win.store.project.snapshot(), OptSettings(vars=stale))
    assert set(opt.vars) == {c.uid for c in v.cameras} & set(opt.vars) and opt.vars
    win.set_page('viewports')


def test_the_tolerance_result_is_marked_stale_on_another_variant(app, win):
    from vantage.core.tolerance import TolSettings, run_monte_carlo
    win.set_page('tolerance')
    pg = win.page('tolerance')
    pump(app, 0.3)
    r = run_monte_carlo(win.ctx.analysis.engine.share(), win.store.project.snapshot(), TolSettings(samples=10, seed=1))
    pg.show_result(r)
    assert not pg.stale() and pg.last.text().startswith('Last run')
    v3 = win.store.project.active_uid
    _switch(app, win, '4-cam')
    assert pg.stale() and 'run again' in pg.last.text()
    assert all(not pg.table.cellWidget(i, 5).isEnabled() for i in range(pg.table.rowCount()))
    before = pg.pinned_result
    pg.preview(0, pin=True)                         # the old variant's perturbations are not shown on this one
    assert pg.pinned_result is before
    win.switch_variant(v3)
    settle(app, win)
    assert not pg.stale() and pg.last.text().startswith('Last run')
    win.set_page('viewports')


def test_radar_and_robust_pages_follow_the_variant(app, win):
    from vantage.core import imaging_radar as IR
    p = win.store.project
    keys = set()
    for name in ('6-cam · v3', '4-cam'):
        q = p.snapshot()
        q.active_uid = _variant(win, name).uid
        keys.add(IR.cache_key(q))
    assert len(keys) == 2                            # two variants without radars are two keys (the page names them)
    win.set_page('robust_perception')
    pg = win.page('robust_perception')
    pump(app, 0.3)
    g0 = pg._src_gen if hasattr(pg, '_src_gen') else 0
    _switch(app, win, '4-cam')
    assert pg._src_gen == g0 + 1                     # the sources of the variant before are dropped
    win.set_page('viewports')


def _overlaps(page):
    leaf = (QLabel, QAbstractButton, QComboBox, QLineEdit, QAbstractSpinBox, QTableView)

    def vis(x):
        r = QRect(x.mapTo(page, QPoint(0, 0)), x.size())
        q = x.parentWidget()
        while q is not None and q is not page:
            r = r.intersected(QRect(q.mapTo(page, QPoint(0, 0)), q.size()))
            q = q.parentWidget()
        return r

    def anc(a, b):
        q = b.parentWidget()
        while q is not None:
            if q is a:
                return True
            q = q.parentWidget()
        return False
    xs = [x for x in page.findChildren(QWidget) if isinstance(x, leaf) and x.isVisible()
          and not isinstance(x.parentWidget(), (QAbstractSpinBox, QComboBox))
          and not isinstance(x.parentWidget(), QTableView)
          and not isinstance(getattr(x.parentWidget(), 'parentWidget', lambda: None)(), QTableView)]
    rs = [(x, vis(x)) for x in xs]
    rs = [(x, r) for x, r in rs if r.width() > 2 and r.height() > 2]
    bad = []
    for i, (a, ra) in enumerate(rs):
        for b, rb in rs[i + 1:]:
            it = ra.intersected(rb)
            if it.width() > 3 and it.height() > 3 and not anc(a, b) and not anc(b, a):
                bad.append(f'{type(a).__name__} {getattr(a, "text", lambda: "")()!r} / '
                           f'{type(b).__name__} {getattr(b, "text", lambda: "")()!r}')
    return bad


@pytest.mark.parametrize('page', ['camviews', 'optimiser', 'event_camera', 'volume'])
def test_no_controls_drawn_over_each_other_on_a_small_screen(app, win, page):
    win.resize(1366, 768)
    try:
        win.set_page(page)
        pump(app, 0.6)
        assert not _overlaps(win.page(page))
    finally:
        win.set_page('viewports')
        win.resize(1600, 1000)
