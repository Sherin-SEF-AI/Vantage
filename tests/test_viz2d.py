"""2D visualizations: colour scales and their validator, smooth heatmaps with iso-lines and vector outlines, the plan
view's legend placement and hover read-out, chart tooltips and the bar <-> plan camera highlight."""
import math
import time
from types import SimpleNamespace

import numpy as np
import pytest
from PyQt6.QtCore import QEvent, QPointF, Qt
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtWidgets import QApplication

from test_gui import app, win, pump, settle  # noqa: F401 (fixtures)
from vantage.ui import colormaps as C


def _move(widget, q):
    QApplication.sendEvent(widget, QMouseEvent(QEvent.Type.MouseMove, QPointF(q), widget.mapToGlobal(QPointF(q)),
                                               Qt.MouseButton.NoButton, Qt.MouseButton.NoButton,
                                               Qt.KeyboardModifier.NoModifier))


def _leave(widget):
    QApplication.sendEvent(widget, QEvent(QEvent.Type.Leave))


# ============================================================================ colour scales
def test_colour_maths_known_values():
    assert C.contrast('#000000', '#ffffff') == pytest.approx(21.0, abs=1e-6)
    assert C.contrast('#777777', '#777777') == pytest.approx(1.0)
    lab = C.oklab(C.hex_rgb('#ffffff'))
    assert lab[0] == pytest.approx(1.0, abs=1e-3) and abs(lab[1]) < 1e-3 and abs(lab[2]) < 1e-3
    rng = np.random.default_rng(1)
    x = rng.random((200, 3))
    assert np.abs(C.oklab_to_rgb(C.oklab(x)) - x).max() < 1e-9           # round trip
    assert C.rgb_hex(C.hex_rgb('#2a78d6')) == '#2a78d6'
    # a grey stays grey under both colour-vision simulations
    g = C.hex_rgb('#808080')
    for kind in ('protan', 'deutan'):
        assert np.abs(C.simulate(g, kind) - g).max() < 0.02


@pytest.mark.parametrize('theme', ['dark', 'light'])
def test_every_scale_passes_the_validator(theme):
    rep = C.validate(theme)
    assert rep == {'density': [], 'redundancy': [], 'diverging': [], 'below_min': []}, rep


def test_validator_catches_bad_scales():
    surf = C.SURFACE['light']
    assert any('monotone' in m for m in C.check_sequential(['#cde2fb', '#0d366b', '#86b6ef'], surf))
    assert any('span' in m for m in C.check_sequential(['#cde2fb', '#9ec5f4'], surf))
    assert any('farthest' in m for m in C.check_sequential(C.BLUE[::-1], surf))      # dark end on the light surface
    probs = C.check_steps(['#86b6ef', '#6da7ec', '#104281'], surf)
    assert any('lightness gap' in m for m in probs) and any('dE' in m for m in probs)
    assert any('contrast' in m for m in C.check_steps(['#e8f0fb', '#86b6ef', '#104281'], surf))
    # red/green steps of equal lightness collapse for deuteranopes
    assert any('deutan' in m for m in C.check_steps(['#c05050', '#4f8f3f'], surf, min_dl=0, min_contrast=1))
    assert any('neutral' in m for m in C.check_diverging(['#9c390c', '#e89d83', '#86b6ef', '#3987e5', '#1c5cab'],
                                                         surf))


@pytest.mark.parametrize('theme', ['dark', 'light'])
def test_density_scale(theme):
    sc = C.density_scale(3.0, 5.0, theme)
    assert (sc.lo, sc.hi) == (3.0, 16.0)
    assert float(sc.t(3.0)) == 0.0 and float(sc.t(16.0)) == 1.0 and float(sc.t(40.0)) == 1.0
    assert float(sc.t(5.0)) == pytest.approx(math.log(5 / 3) / math.log(16 / 3))
    v = np.geomspace(3.0, 16.0, 40)
    L = C.oklab(sc.rgb(v) / 255.0)[:, 0]
    d = np.diff(L)
    surf_L = C.lightness(C.SURFACE[theme])
    # more pixels = farther from the surface in lightness, strictly
    assert (np.sign(d) == np.sign(L[-1] - surf_L)).all() and (np.abs(d) > 0).all()
    assert sc.hex(2.9) == C.BELOW_MIN[theme] and sc.hex(3.0) != C.BELOW_MIN[theme]
    assert sc.ticks()[0] == 3.0 and sc.ticks()[-1] == 16.0 and 5 in sc.ticks()
    s2 = C.density_scale(4.5, 8.0, theme)
    assert s2.lo == 4.5 and s2.hi == 25                     # a round top above 5 × the minimum
    assert C.density_scale(float('nan'), None, theme).lo == 3.0


@pytest.mark.parametrize('theme', ['dark', 'light'])
def test_redundancy_steps_are_ordered_and_distinct(theme):
    steps = C.redundancy(theme)
    L = [C.lightness(h) for h in steps]
    surf = C.lightness(C.SURFACE[theme])
    assert all(abs(b - surf) > abs(a - surf) for a, b in zip(L, L[1:]))        # 3+ stands out most
    for a, b in zip(steps, steps[1:]):
        for kind in (None, 'protan', 'deutan'):
            assert C.delta_e(a, b, kind) >= 8
    assert all(C.contrast(h, C.SURFACE[theme]) >= 2 for h in steps)


# ============================================================================ maps on a synthetic field
def _fake(field, roi=None, cell=0.05, x0=-2.0, y0=-2.0):
    ny, nx = field.shape
    roi = np.ones_like(field, bool) if roi is None else roi
    count = np.where(field > 0, 1, 0).astype(np.int8)
    g = SimpleNamespace(x0=x0, y0=y0, cell=cell, nx=nx, ny=ny, roi=roi, body=np.zeros_like(roi))
    return SimpleNamespace(grid=g, best_dens=np.where(roi, field, 0).astype(np.float32), count=count, cams=[],
                           cam_index=lambda uid: -1)


def _radial(n=80, cell=0.05):
    """Density 24 / (1 + r) px/cm around the origin on an n x n grid of cell size `cell` (x0 = y0 = -n/2 cell)."""
    x0 = -n / 2 * cell
    c = x0 + (np.arange(n) + 0.5) * cell
    X, Y = np.meshgrid(c, c)
    return 24.0 / (1.0 + np.hypot(X, Y)), x0


def test_iso_lines_sit_on_the_thresholds():
    from vantage.ui import maps
    f, x0 = _radial()
    res = _fake(f, x0=x0, y0=x0)
    lines = maps.density_contours(res, (12.0, 16.0))
    for lv in (12.0, 16.0):
        assert len(lines[lv]) == 1                      # one closed ring each
        P = lines[lv][0]
        assert np.allclose(P[0], P[-1], atol=1e-6)
        r = np.hypot(P[:, 0], P[:, 1])
        r_true = 24.0 / lv - 1.0
        assert np.abs(r - r_true).max() < 0.03, (lv, np.abs(r - r_true).max())  # within 0.6 of a 5 cm cell


def test_iso_lines_do_not_trace_the_roi_edge_or_blind_cells():
    from vantage.ui import maps
    f, x0 = _radial()
    n = f.shape[0]
    jj, ii = np.mgrid[0:n, 0:n]
    roi = np.hypot(ii - n / 2 + 0.5, jj - n / 2 + 0.5) < 30         # ROI disc of 1.5 m
    res = _fake(np.full_like(f, 9.0), roi=roi, x0=x0, y0=x0)
    assert maps.density_contours(res, (5.0,))[5.0] == []            # flat field: no line along the edge
    f2 = f.copy()
    f2[38:42, 50:54] = 0                                             # a blind patch inside a 12+ px/cm area
    res2 = _fake(f2, roi=roi, x0=x0, y0=x0)
    pts = np.vstack(maps.density_contours(res2, (5.0, 12.0))[12.0])
    assert np.abs(np.hypot(pts[:, 0], pts[:, 1]) - 1.0).max() < 0.03   # still only the true r = 1 m ring


def test_smooth_density_never_bleeds_across_edges():
    from vantage.ui import maps
    n = 60
    jj, ii = np.mgrid[0:n, 0:n]
    roi = np.hypot(ii - 30, jj - 30) < 22
    f = np.full((n, n), 8.0)
    f[28:32, 28:32] = 0                                              # blind cells in the middle
    res = _fake(f, roi=roi)
    sc = C.density_scale(3.0, 5.0)
    img = maps.make(res, 'density', k=4, shadow=False, blind=False, thresholds=(3.0, 5.0))
    assert img.shape == (n * 4, n * 4, 4) and img.dtype == np.uint8
    opaque = img[..., 3] > 0
    want = sc.rgb(8.0)
    assert (img[opaque, :3] == want).all()                          # no darker or paler fringe anywhere
    assert not opaque[118:122, 118:122].any()                       # the blind cells stay open
    roi_up = np.repeat(np.repeat(roi[::-1], 4, 0), 4, 1)
    assert (opaque & ~roi_up).sum() < 0.03 * roi_up.sum()           # the smoothed edge stays at the ROI's edge
    assert (roi_up & ~opaque).sum() < 0.03 * roi_up.sum() + 16 * 16
    # a below-minimum field is the neutral grey (hatched in the 3D texture)
    low = maps.make(_fake(np.full((n, n), 2.0), roi=roi), 'density', k=4, shadow=False, blind=True,
                    thresholds=(3.0, 5.0))
    cols = {tuple(int(v) for v in c) for c in low[low[..., 3] > 0, :3]}
    grey = tuple(int(v) for v in np.round(C.hex_rgb(sc.below_hex) * 255))
    assert grey in cols and len(cols) == 2                           # grey + the hatch lines over it
    # without thresholds (callers that do not pass them) nothing is flagged
    old = maps.make(_fake(np.full((n, n), 2.0), roi=roi), 'density', k=2, shadow=False, blind=False)
    assert {tuple(c) for c in old[old[..., 3] > 0, :3]} == {tuple(sc.lut[0])}


def _qpath(loops):
    from PyQt6.QtGui import QPainterPath
    path = QPainterPath()
    path.setFillRule(Qt.FillRule.OddEvenFill)
    for P in loops:
        path.moveTo(*P[0])
        for x, y in P[1:]:
            path.lineTo(float(x), float(y))
        path.closeSubpath()
    return path


def _area(loops):                             # separate blobs (no holes): add each loop's area
    return sum(abs(0.5 * (P[:-1, 0] * P[1:, 1] - P[1:, 0] * P[:-1, 1]).sum()) for P in loops)


def test_below_minimum_and_blind_outlines_enclose_the_right_ground():
    from vantage.ui import maps
    f, x0 = _radial()
    n = f.shape[0]
    f[30:34, 60:66] = 0                                              # a blind patch where density is ~8
    res = _fake(f, x0=x0, y0=x0)
    below = _qpath(maps.below_min_outline(res, 8.0)).intersected(_qpath(maps.valid_outline(res)))
    c = x0 + (np.arange(n) + 0.5) * 0.05
    inside = np.array([[below.contains(QPointF(float(c[i]), float(c[j]))) for i in range(n)] for j in range(n)])
    want = (f < 8.0) & (f > 0)                                       # seen and below the minimum
    assert (inside == want).mean() > 0.995, (inside != want).sum()
    assert not inside[f == 0].any()                                  # blind cells are not 'below minimum'
    blind = maps.mask_outline(f == 0, res.grid)
    assert _area(blind) == pytest.approx(24 * 0.05 ** 2 - 4 * 0.5 * 0.025 ** 2, rel=0.005)   # corners cut
    assert maps.below_min_outline(res, 0.5) == []


def test_make_keeps_its_interface_on_a_real_result():
    from vantage.core import samples
    from vantage.core.analysis import Engine
    from vantage.ui import maps
    p = samples.tigor_sample_project()
    res = Engine().analyse(p)
    g = res.grid
    for mode in ('density', 'redundancy', 'coverage', 'bev', 'footprints', 'blank'):
        img = maps.make(res, mode, k=4)
        assert img.shape == (g.ny * 4, g.nx * 4, 4) and img.dtype == np.uint8, mode
    assert maps.make(None, 'density') is None
    t0 = time.perf_counter()
    for mode in ('density', 'coverage', 'footprints'):
        maps.make(res, mode, k=4)                                  # the 3D ground texture
    assert (time.perf_counter() - t0) / 3 < 0.25
    # the redundancy map uses the ordered steps: cells seen by 2 cameras get step 2
    img = maps.make(res, 'redundancy', k=1, shadow=False, blind=False)[::-1]
    two = g.roi & (res.count == 2)
    assert (img[two, :3] == np.round(C.hex_rgb(C.redundancy()[1]) * 255)).all()
    up = maps.unique_outline(res, res.cams[0].uid)
    assert up and all(np.allclose(P[0], P[-1]) for P in up)


# ============================================================================ plan view
def _top(app, win, heat):
    win.set_view('top')
    settle(app, win)
    win.set_heatmap(heat)
    pump(app, 0.2)
    plan = win.viewarea.panes[0].plan
    plan.fit()
    plan.grab()
    return plan


def test_plan_legend_present_and_clear_of_the_data(app, win):
    for heat in ('density', 'redundancy', 'coverage'):
        plan = _top(app, win, heat)
        res = win.ctx.analysis.result
        lr = plan.legend_rect
        assert lr is not None, heat
        assert lr.left() >= 0 and lr.top() >= 0 and lr.right() <= plan.width() and lr.bottom() <= plan.height()
        data = plan._data_rect(res)
        assert not lr.intersects(data), (heat, lr, data)           # fitted view: the legend hides no data
        assert not plan._legend_over
    L = plan._legend_obj
    assert [t for _, t in L.chips] == [c.tag for c in res.cams]     # best-camera legend names every camera
    # zoomed into the data: the legend takes the corner that covers least, drawn on its card
    plan = _top(app, win, 'density')
    plan.scale *= 4
    plan._user_view = True
    plan.grab()
    assert plan.legend_rect is not None and plan._legend_over
    plan.legend = False
    plan.grab()
    assert plan.legend_rect is None
    plan.legend = True


def test_plan_iso_lines_at_the_project_thresholds(app, win):
    plan = _top(app, win, 'density')
    a = win.store.project.analysis
    assert set(plan._ov['iso']) == {a.min_density, a.target_density}
    assert not plan._ov['iso'][a.target_density][0].isEmpty()       # the sample meets 5 px/cm only near the car

    def strict(pj):
        pj.analysis.min_density = 4.5
    win.store.edit('analysis', strict, 'strict minimum')
    settle(app, win)
    plan.grab()
    assert 4.5 in plan._ov['iso'] and not plan._ov['below'][0].isEmpty()
    assert plan._legend_obj.bar[1].lo == 4.5
    win.store.undo.undo()
    settle(app, win)


def test_plan_hover_readout_matches_the_analysis(app, win):
    plan = _top(app, win, 'density')
    res = win.ctx.analysis.result
    g = res.grid
    x, y = 4.6, 1.9
    _move(plan, plan.w2s(x, y))
    pump(app, 0.05)
    i, j = g.cell_of(x, y)
    rows = plan.readout()
    q = res.query(x, y)
    assert rows[0] == ('head', f'{float(res.best_dens[j, i]):.1f} px/cm')
    best = res.cams[int(res.best[j, i])].tag
    assert rows[1][0] == 'sub' and rows[1][1].startswith(f'best {best} · ')
    cams = [v for r, v in rows if r == 'cam']
    assert [c.tag for c, *_ in cams] == [ce.tag for ce, _ in q['cams'][:4]]
    assert [d for _, d, _, _ in cams] == [f'{dd:.1f}' for _, dd in q['cams'][:4]]
    cam0 = cams[0][0]
    assert cams[0][2] == f'{math.dist((x, y, 0.0), cam0.pos()):.2f}'      # range from the camera, m
    assert rows[-1][0] == 'foot' and 'click selects' in rows[-1][1]
    img = plan.grab().toImage()
    assert img.width() > 100
    # redundancy: the head is the number of cameras seeing the cell
    win.set_heatmap('redundancy')
    pump(app, 0.1)
    _move(plan, plan.w2s(x, y))
    n = int(res.count[j, i])
    assert plan.readout()[0] == ('head', f'{n} camera' + ('' if n == 1 else 's'))
    # a blind cell
    jj, ii = np.nonzero(g.roi & (res.count == 0))
    for b in range(len(jj)):
        bx, by = g.x0 + (ii[b] + 0.5) * g.cell, g.y0 + (jj[b] + 0.5) * g.cell
        q = plan.w2s(bx, by)
        if plan._pick_camera(q, r=14) is None:
            _move(plan, q)
            assert plan.readout()[:2] == [('crit', 'Blind'), ('sub', 'no camera sees this point')]
            break
    _leave(plan)
    assert plan.readout() is None


def test_plan_hover_stays_fast(app, win):
    plan = _top(app, win, 'density')
    plan.repaint()
    t0 = time.perf_counter()
    for k in range(20):
        _move(plan, plan.w2s(4.0 + k * 0.02, 1.5 + k * 0.01))
        plan.repaint()
    dt = (time.perf_counter() - t0) / 20
    assert dt < 0.05, dt                     # ~6 ms on the dev machine: the heatmap layers are only blitted


def test_bar_hover_lights_up_the_camera_and_back(app, win):
    from vantage.ui.charts import CAMERA_HOVER
    plan = _top(app, win, 'density')
    win.bottom.show_tab('coverage')
    bars = win.bottom.coverage.bars
    bars.resize(520, 180)
    res = win.ctx.analysis.result
    uid, slot, tag, val = bars.rows[1]
    _move(bars, QPointF(200, 22 + 22 * 1 + 10))
    pump(app, 0.05)
    assert bars.hover == 1 and CAMERA_HOVER.uid == uid and plan.hl_cam == uid
    assert bars.tip_text().startswith(f'{val:.3f} m² · {tag} · seen only by this camera')
    plan.grab()
    assert plan._uniq[0][0]() is res and plan._uniq[1] == uid and not plan._uniq[2].isEmpty()
    _leave(bars)
    assert CAMERA_HOVER.uid == '' and plan.hl_cam == '' and bars.tip is None
    # a camera hovered in the plan lights up its row
    cam = win.store.variant.by_tag(tag)
    _move(plan, plan.w2s(cam.x, cam.y))
    assert bars.ext_hover == cam.uid
    _leave(plan)
    assert bars.ext_hover == ''
    # click still selects
    got = []
    bars.picked.connect(got.append)
    from PyQt6.QtTest import QTest
    QTest.mouseClick(bars, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPointF(200, 22 + 10).toPoint())
    assert got == [bars.rows[0][0]]


# ============================================================================ charts
def test_nice_ticks():
    from vantage.ui.charts import nice_ticks
    for v in (0.0007, 0.83, 3.5, 7, 10.4, 99, 1234.5):
        for n in (3, 4):
            t = nice_ticks(v, n)
            st = np.diff(t)
            assert t[0] == 0 and t[-1] >= v and np.allclose(st, st[0]) and len(t) <= n + 3, (v, n, t)
            assert t[-2] < v                                     # no empty step at the top
    assert nice_ticks(0) == [0.0, 1.0] and nice_ticks(float('nan')) == [0.0, 1.0]


def test_density_histogram_tooltip_and_markers(app):
    from vantage.core.analysis import HIST_EDGES
    from vantage.ui.charts import DensityHistogram
    h = DensityHistogram()
    h.resize(640, 190)
    areas = np.array([0.0, 3.1, 10.4, 6.9, 5.0, 3.6, 2.3, 1.3, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.4])
    h.set_data(HIST_EDGES, areas, 3.0, 5.0)
    h.show()
    pump(app, 0.05)
    r = h._plot()
    bw = r.width() / len(areas)
    assert h.x_of(3.0) == pytest.approx(r.left() + bw) and h.x_of(5.0) == pytest.approx(r.left() + 3 * bw)
    assert h.x_of(3.5) == pytest.approx(r.left() + 1.5 * bw)
    _move(h, QPointF(r.left() + 2.5 * bw, r.bottom() - 5))
    assert h.hover == 2
    assert h.tip_text() == f'{areas[2]:.3f} m² · 4–5 px/cm · {areas[2] / areas.sum() * 100:.1f} % of the seen ROI'
    _move(h, QPointF(r.left() + 0.5 * bw, r.bottom() - 5))
    assert h.tip_text().startswith('0.000 m² · below 3 px/cm') and 'below the minimum' in h.tip_text()
    _move(h, QPointF(r.left() + 14.5 * bw, r.bottom() - 5))
    assert '≥ 16 px/cm' in h.tip_text() and 'meets the target' in h.tip_text()
    img = h.grab().toImage()
    assert img.width() == 640
    _leave(h)
    assert h.tip is None and h.hover is None
    h.set_data(None, None, 3.0, 5.0)
    h.grab()


def test_other_charts_tooltips(app):
    from vantage.ui.charts import Histogram, Tornado, Scatter, ProjectionPlot
    hg = Histogram(log=True)
    hg.resize(500, 200)
    vals = np.r_[np.zeros(50), np.linspace(0.01, 0.2, 50)]
    hg.set_data(vals, bins=10, markers=[(0.15, 'P95', True), (0.2, 'max', False)], xlabel='blind area, m²')
    hg.show()
    r = hg._plot()
    _move(hg, QPointF(r.left() + 0.5 * r.width() / 10, r.bottom() - 3))
    assert hg.tip_text().startswith(f'{int(hg.counts[0])} samples · 0.0000–')
    hg.grab()
    hg.set_data([], bins=10)
    assert hg.counts is None and hg.tip is None
    hg.grab()
    t = Tornado()
    t.resize(500, 200)
    t.set_rows([('F yaw', -0.012, 0.034), ('R z', 0.001, -0.002)])
    _move(t, QPointF(250, 40 + 20 + 10))
    assert t.tip_text().startswith('+0.0010 / -0.0020 m² · R z')
    t.grab()
    s = Scatter()
    s.resize(500, 300)
    s.set_data([[0.1, 4.0], [0.2, 5.0]], [[0.1, 4.0], [0.2, 5.0]], [(0.1, 4.0, 'A')], current=(0.2, 5.0))
    _move(s, s._map(0.1, 4.0))
    assert s.tip_text().startswith('0.100 m² · 4.0 px/cm · A')
    _move(s, s._map(0.2, 5.0))
    assert 'current layout' in s.tip_text()
    s.grab()
    pp = ProjectionPlot()
    pp.resize(400, 300)
    pp.set_data([('eq', np.linspace(0, 90, 10), np.linspace(0, 3, 10), 'sel')], [(30, 1)], [(2.0, 'H')], (40, 'HFOV/2'),
                90, 5)
    pp.grab()


def test_plan_without_cameras_and_theme_switch(app, win):
    from vantage.ui.theme import THEME
    st = win.store

    def hide(pj):
        for c in pj.variant.cameras:
            c.visible = False
    st.edit('variant', hide, 'hide all')
    settle(app, win)
    res = win.ctx.analysis.result
    assert len(res.cams) == 0
    for heat in ('density', 'redundancy', 'coverage', 'stitched', 'none'):
        plan = _top(app, win, heat)
        plan.grab()
    plan = _top(app, win, 'density')
    assert plan.legend_rect is not None
    assert all(path.isEmpty() and longest is None for path, longest in plan._ov['iso'].values())   # no data, no lines
    _move(plan, plan.w2s(4.6, 1.9))
    assert plan.readout()[0] == ('crit', 'Blind')
    _leave(plan)
    st.undo.undo()
    settle(app, win)
    # the scales follow the theme
    plan.grab()
    dark_img = plan._img
    THEME.apply(app, 'light')
    try:
        pump(app, 0.1)
        plan.grab()
        assert plan._img is not dark_img and plan._legend_obj.bar[1].theme == 'light'
    finally:
        THEME.apply(app, 'dark')
        pump(app, 0.1)
