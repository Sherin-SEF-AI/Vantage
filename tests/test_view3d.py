"""The 3D view's chrome and floor: the view bar (view, shading, overlays, heat map), the floor and grid with lighting
off, the ground map's anti-aliased rim, and the label plates."""
import numpy as np
import pytest
from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QImage, QPainter

from test_gui import app, win, pump, settle, vp_of   # noqa: F401  (fixtures)


def test_view_bar_lists_the_view_shading_overlays_and_heat_map(app, win):
    vp = vp_of(win)
    keys = [k for k, *_ in vp.bar.layout()]
    assert keys == ['view', 'shading', 'overlays', 'heat']
    texts = {k: t for k, _, t, *_ in vp.bar.layout()}
    assert texts['shading'] == 'Solid'
    n_on = sum(1 for v in win.ctx.overlays.values() if v)
    assert texts['overlays'] == f'Overlays {n_on}'
    r = vp.bar.layout()[0][1]
    assert vp.nav.hit(r.center()) == ('bar', 'view')
    assert vp.nav.hit(QPointF(r.right() + 300, r.center().y() + 200)) is None


def test_view_bar_menus_hold_the_windows_own_actions(app, win):
    from vantage.ui.navigation import StayOpenMenu
    vp = vp_of(win)
    a = win.actions_by_id
    for key, must in (('view', ['view_persp', 'view_top', 'frame_all', 'render_image']),
                      ('shading', ['shade_solid', 'shade_wire', 'shade_xray']),
                      ('overlays', ['ov_grid', 'ov_labels', 'ov_shadows', 'ov_camlight']),
                      ('heat', ['heat_none', 'heat_coverage', 'heat_density'])):
        m = vp.provider.view_bar_menu(vp, key)
        assert m is not None
        acts = m.actions()
        for i in must:
            assert a[i] in acts, (key, i)
        assert isinstance(m, StayOpenMenu) == (key == 'overlays')
        m.close()
    m = vp.bar.popup('heat')
    pump(app, 0.1)
    assert m is not None and vp.bar.open == 'heat'
    m.hide()
    pump(app, 0.1)
    assert vp.bar.open is None


def test_view_bar_follows_the_shading_and_heat_map(app, win):
    vp = vp_of(win)
    a = win.actions_by_id
    a['shade_xray'].trigger()
    a['heat_density'].trigger()
    pump(app, 0.2)
    texts = {k: t for k, _, t, *_ in vp.bar.layout()}
    assert texts['shading'] == 'X-ray' and texts['heat'] == 'Heat map · Density'
    a['shade_solid'].trigger()
    a['heat_none'].trigger()
    pump(app, 0.2)
    assert {k: t for k, _, t, *_ in vp.bar.layout()}['heat'] == 'Heat map'


def test_overlays_menu_toggles_without_closing(app, win):
    """A check box in the overlays menu toggles the overlay and leaves the menu open for the next one."""
    from PyQt6.QtTest import QTest
    vp = vp_of(win)
    m = vp.bar.popup('overlays')
    pump(app, 0.2)
    act = win.actions_by_id['ov_labels']
    before = act.isChecked()
    m.setActiveAction(act)
    QTest.mouseClick(m, Qt.MouseButton.LeftButton, pos=m.actionGeometry(act).center())
    pump(app, 0.1)
    assert act.isChecked() != before and win.ctx.overlays['labels'] == act.isChecked()
    assert m.isVisible()
    act.trigger()                          # back as it was
    m.hide()
    pump(app, 0.1)


def test_a_render_has_the_plain_label_not_the_buttons(app, win):
    vp = vp_of(win)
    img = vp.render_image(1.0)
    assert not img.isNull()
    assert vp.provider.hud_label(vp).startswith('Perspective')


def test_lighting_off_keeps_the_floor_and_backdrop(app, win):
    vp = vp_of(win)
    a = win.actions_by_id['ov_shadows']
    if not a.isChecked():
        a.trigger()
    a.trigger()                            # off
    pump(app, 0.2)
    st = vp.provider.render_style(vp)
    assert not st.shadows and not st.ao and st.ground_plane and st.backdrop
    a.trigger()                            # on again
    pump(app, 0.2)


def test_grid_moves_to_the_ground_shader_with_a_shader_renderer(app, win):
    vp = vp_of(win)
    sc = vp.provider

    def grid_lines():
        from vantage.ui.theme import THEME
        return [l for l in sc.lines(vp) if l.color in (THEME.color('grid1'), THEME.color('grid2'))]
    assert not sc.render_style(vp).grid and grid_lines()           # no shader renderer: the line grid
    had = 'renderer' in vp.__dict__
    vp.renderer = object()                                         # stands in for the GL renderer
    try:
        st = sc.render_style(vp)
        assert st.grid and st.axes and not grid_lines()
        o = sc.project.origin_offset()
        assert st.origin == (pytest.approx(o[0]), pytest.approx(o[1]))
        win.actions_by_id['ov_grid'].trigger()                     # grid off: neither
        pump(app, 0.1)
        assert not sc.render_style(vp).grid and not grid_lines()
        win.actions_by_id['ov_grid'].trigger()
        pump(app, 0.1)
    finally:
        if not had:
            del vp.renderer


def test_ground_map_rim_is_anti_aliased(app, win):
    """The footprints' outer edge is cut along a smooth line (partial alpha at the rim), not in whole cells; inside
    the analysis domain the map is untouched."""
    from vantage.ui import maps
    res = win.ctx.analysis.result
    img = maps.make(res, 'footprints', k=4)
    a = img[..., 3].astype(int)
    from scipy import ndimage
    g = res.grid
    dom = ndimage.binary_fill_holes(g.roi | g.body)[::-1]
    up = np.repeat(np.repeat(dom, 4, 0), 4, 1)
    inner = ndimage.binary_erosion(up, iterations=6)
    full = img[..., 3][inner & (img[..., 3] > 0)]
    assert full.size and full.min() >= 20                  # the tint is kept inside
    rim = up & ~ndimage.binary_erosion(up, iterations=3)
    ramp = set(np.unique(a[rim])) - set(np.unique(a[inner])) - {0}
    assert len(ramp) >= 2                                  # partial alpha along the edge, not just whole cells
    assert (a[~ndimage.binary_dilation(up, iterations=3)] == 0).all()


def test_label_plates_leave_out_overlaps(app, win):
    vp = vp_of(win)
    sc = vp.provider
    img = QImage(300, 200, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(0)
    p = QPainter(img)
    placed = []
    r1 = sc._chip(p, 50, 50, 'ML', placed=placed)
    r2 = sc._chip(p, 55, 52, 'QL', placed=placed)          # on top of the first: left out
    r3 = sc._chip(p, 50, 120, 'T4', placed=placed)
    p.end()
    assert r1 is not None and r2 is None and r3 is not None and len(placed) == 2
    assert img.pixelColor(52, 50).alpha() > 0


def test_camera_tags_follow_the_labels_overlay(app, win):
    vp = vp_of(win)
    sc = vp.provider
    drawn = []
    real = sc._chip
    sc._chip = lambda p, x, y, text, **kw: drawn.append(text) or real(p, x, y, text, **kw)
    try:
        img = QImage(vp.width(), vp.height(), QImage.Format.Format_ARGB32_Premultiplied)
        p = QPainter(img)
        sc.overlay(p, vp)
        p.end()
        tags = {c.tag for c in sc.cameras() if c.visible}
        assert set(drawn) & tags
        win.actions_by_id['ov_labels'].trigger()
        pump(app, 0.1)
        drawn.clear()
        p = QPainter(img)
        sc.overlay(p, vp)
        p.end()
        assert not set(drawn) & tags
        win.actions_by_id['ov_labels'].trigger()
        pump(app, 0.1)
    finally:
        del sc._chip
