"""Standing objects in the stitched bird's-eye view (core.stitched_objects): the smear geometry, agreement with the
remap tables, the seam advice, the export and batch report, and the views that show it."""
import copy
import json
import math
import sys
import time

import numpy as np
import pytest
from PyQt6.QtCore import Qt, QPointF, QEvent
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from vantage.core import samples, stitched_objects as so
from vantage.core import exporters as ex
from vantage.core.analysis import Engine
from vantage.core.project import AnalysisSettings
from vantage.core.stitch import lut_for_points

from test_gui import app, win, pump, _wait     # noqa: F401  (fixtures)


@pytest.fixture(scope='module')
def sample():
    p = samples.tigor_sample_project()
    e = Engine()
    return p, e, e.analyse(p)


@pytest.fixture(scope='module')
def child(sample):
    p, e, r = sample
    return so.analyse(p, e, r, height=0.9)


# ----------------------------------------------------------------------------- geometry
def test_smear_geometry_by_hand(sample):
    """A pole point at height z lands on the ground at P + (P - C)·z / (C_z - z), nowhere above the camera."""
    p, e, r = sample
    f = p.variant.by_tag('F')
    C = f.pos()
    P = np.array([4.5, 0.0])
    sm = {d['tag']: d for d in so.smears(p, e, r, P[0], P[1], height=0.9, diameter=0.3, n=9)}
    d = sm['F']
    face = P + (C[:2] - P) / np.linalg.norm(C[:2] - P) * 0.15            # the side of the pole facing F
    assert d['foot'] == pytest.approx(face)
    zs = 0.9 * (np.arange(9) + 0.5) / 9
    for z, g in zip(zs, d['points']):
        if z < C[2]:
            assert g == pytest.approx(face + (face - C[:2]) * z / (C[2] - z))
        else:
            assert np.isnan(g).all()
    # by hand for one height: F at x 3.23 m, 0.62 m up; the face at x 4.35; z = 0.31 m lands 1.12 m further out
    assert C[0] == pytest.approx(3.23) and C[2] == pytest.approx(0.62)
    g = face + (face - C[:2]) * 0.31 / (0.62 - 0.31)
    assert g[0] == pytest.approx(4.35 + 1.12) and g[1] == pytest.approx(0.0)


def test_a_short_pole_in_front_is_shown_whole(sample):
    """In the middle of the front camera's sector, below the camera and well inside the view: all of it is shown."""
    p, e, r = sample
    s = so.analyse(p, e, r, height=0.3)
    shown, raw, by = s.at(3.9, 0.0)
    assert raw == pytest.approx(1.0) and shown == pytest.approx(1.0)
    F = p.variant.by_tag('F').uid
    assert by[F] == pytest.approx(1.0) and len(by) == 1
    # with only that camera the same pole is still whole, and a pole round the side is not seen at all
    q = copy.deepcopy(p)
    for c in q.variant.cameras:
        c.visible = c.uid == F
    rq = e.analyse(q)
    s1 = so.analyse(q, e, rq, height=0.3)
    assert s1.at(3.9, 0.0)[0] == pytest.approx(1.0)
    assert s1.at(1.2, 2.2) is None or s1.at(1.2, 2.2)[1] < 0.25


def test_a_pole_at_a_seam_vanishes_although_both_cameras_see_it(sample, child):
    """The classic surround-view loss: at the seam each camera's smear runs into the other camera's sector."""
    p, e, r = sample
    h = child.hotspots[0]
    seam = child.seams[h.seam]
    shown, raw, _ = child.at(h.x, h.y)
    assert raw >= 0.25 and shown < 0.25
    tags = set(seam['tags'])
    assert tags & set(h.seen_by), (tags, h.seen_by)
    sm = {d['tag']: d for d in so.smears(p, e, r, h.x, h.y)}
    for t in tags & set(h.seen_by):
        d = sm[t]
        assert d['seen'].any()
        assert d['weight'][d['seen']].mean() < 0.25        # what it draws lands where the other camera owns the view
    assert child.vanish.sum() > 0 and child.metrics['vanish_m2'] > so.ISSUE_M2
    # every vanish cell is nearest one of the seams, and the per-seam areas add up
    assert sum(s['vanish_m2'] for s in child.seams) == pytest.approx(child.metrics['vanish_m2'])


def test_edge_losses_are_not_blamed_on_the_seams(sample, child):
    """Straight ahead of the 0.62 m front camera a 0.9 m child is cut off (view edge, above the camera), not lost at a seam."""
    shown, raw, _ = child.at(5.2, 0.0)
    i, j = int((5.2 - child.x0) // child.cell), int((0.0 - child.y0) // child.cell)
    assert raw >= 0.25 and shown < 0.25
    assert child.cut[j, i] and not child.vanish[j, i]
    assert child.metrics['cut_m2'] > child.metrics['vanish_m2']


def test_shown_fractions_follow_the_remap_tables(sample, child):
    """Each camera's share at a cell is its remap-table weight where it draws the object (stitch.lut_for_points)."""
    p, e, r = sample
    g = r.grid
    picks = [(child.hotspots[0].x, child.hotspots[0].y), (3.9, 0.3), (0.5, -1.6), (-1.3, 0.9), (2.0, 2.1)]
    checked = 0
    for x, y in picks:
        i, j = int((x - child.x0) // child.cell), int((y - child.y0) // child.cell)
        if not child.considered[j, i]:
            continue
        cx, cy = child.x0 + (i + 0.5) * child.cell, child.y0 + (j + 0.5) * child.cell
        for d in so.smears(p, e, r, cx, cy, n=so.HEIGHTS):
            ok = d['seen'] & np.isfinite(d['points']).all(1)
            G = d['points'][ok]
            inside = ((G[:, 0] >= g.x0) & (G[:, 0] < g.x0 + g.nx * g.cell) & (G[:, 1] >= g.y0)
                      & (G[:, 1] < g.y0 + g.ny * g.cell))
            G = G[inside]
            if not len(G):
                continue
            gi = np.clip(((G[:, 0] - g.x0) / g.cell).astype(int), 0, g.nx - 1)
            gj = np.clip(((G[:, 1] - g.y0) / g.cell).astype(int), 0, g.ny - 1)
            lut = lut_for_points(p, e, r, np.c_[G, np.zeros(len(G))][None], g.body[gj, gi][None])
            assert lut.weight[d['uid']][0] == pytest.approx(d['weight'][ok][inside], abs=1e-6)
            assert child.by_cam[d['uid']][j, i] == pytest.approx(d['weight'].mean(), abs=1e-5)
            checked += 1
    assert checked >= 6


# ----------------------------------------------------------------------------- advice
def test_seam_advice_makes_fewer_objects_vanish(sample):
    p, e, r = sample
    a = so.advise(p, e, r)
    assert a.changes and a.vanish_after < a.vanish_now - 0.05
    assert a.ground_low_after <= a.ground_low_now + 0.01 + 1e-9
    for s in a.seams:
        assert abs(((s.best - s.angle) + 180) % 360 - 180) <= 15.0 + 1e-9
    q = copy.deepcopy(p)
    q.variant.seams.update(a.changes)
    rq = e.analyse(q)
    before = so.analyse(p, e, r, cell=0.1).metrics['vanish_m2']
    after = so.analyse(q, e, rq, cell=0.1).metrics['vanish_m2']
    assert after == pytest.approx(a.vanish_after, abs=0.021) and after < before     # angles saved to 0.1°
    assert [round(s.angle, 1) for s in rq.seams if f'{s.a}|{s.b}' in a.changes] == \
        [a.changes[f'{s.a}|{s.b}'] for s in rq.seams if f'{s.a}|{s.b}' in a.changes]


def test_settings_are_saved_and_sane():
    st = AnalysisSettings.from_dict(AnalysisSettings(stitched={'height': 1.8, 'threshold': 0.5}).to_dict())
    assert so.Settings.from_dict(st.stitched).height == 1.8 and so.Settings.from_dict(st.stitched).label == '1.8 m adult'
    bad = so.Settings.from_dict({'height': float('nan'), 'threshold': 7, 'diameter': 'x'})
    assert bad.height == 0.9 and bad.threshold == 0.99 and bad.diameter == 0.3


# ----------------------------------------------------------------------------- export and batch
def test_export_and_batch_report(sample, child, tmp_path):
    p, e, r = sample
    paths = ex.export(['stitched_objects'], p, r, ex.ExportOptions(out_dir=str(tmp_path), engine=e, units='mm'))
    names = sorted(x.replace(str(tmp_path) + '/', '') for x in paths)
    assert names == ['stitched_objects/stitched_objects.json', 'stitched_objects/stitched_objects.png']
    doc = json.loads((tmp_path / 'stitched_objects' / 'stitched_objects.json').read_text())
    assert doc['schema'] == so.SCHEMA and doc['units'] == 'mm' and doc['object']['height'] == 900
    assert doc['metrics']['vanish_m2'] == pytest.approx(child.metrics['vanish_m2'], abs=1e-3)
    h0 = doc['hotspots'][0]
    assert h0['x'] == pytest.approx(child.hotspots[0].x * 1000, abs=0.6)
    assert doc['seam_advice']['vanish_m2_after'] < doc['seam_advice']['vanish_m2_now']
    import cv2
    img = cv2.imread(str(tmp_path / 'stitched_objects' / 'stitched_objects.png'), cv2.IMREAD_UNCHANGED)
    assert img.shape[2] == 4 and img.shape[0] == child.ny * 4 and img.shape[1] == child.nx * 4
    path, text, size = ex.preview('stitched_objects', p, r, ex.ExportOptions(engine=e))
    assert path.endswith('stitched_objects.json') and 'vanishes on' in text and size > 0
    from vantage import batch
    rep, issues = batch.analysis_doc(p, r, e)
    assert rep['stitched_objects']['metrics']['vanish_m2'] == pytest.approx(child.metrics['vanish_m2'], abs=1e-3)
    assert any('stitched bird' in i['message'] for i in rep['issues'])
    assert [i.severity for i in issues] == sorted((i.severity for i in issues),
                                                  key=lambda s: {'critical': 0, 'serious': 1, 'warning': 2}.get(s, 9))


# ----------------------------------------------------------------------------- the views
@pytest.fixture()
def errors(monkeypatch):
    out = []
    monkeypatch.setattr(sys, 'excepthook', lambda t, v, tb: out.append(v))
    return out


def _stitched_ready(app, win):
    _wait(app, lambda: win.ctx.stitched.result is not None and not win.ctx.stitched._busy
          and not win.ctx.stitched.timer.isActive(), 60)
    return win.ctx.stitched.result


def test_objects_heatmap_hover_and_issue(app, win, errors):
    sres = _stitched_ready(app, win)
    assert any(i.action == 'stitched' for i in win.ctx.analysis.issues)
    row = win.bottom.coverage.rows['Object vanishes at seams']
    assert row.v.text() == f'{sres.metrics["vanish_m2"]:.2f}'
    pane = win.viewarea.panes[0]
    pane.set_kind('top')
    win.actions_by_id['heat_stitched'].trigger()
    pump(app, 0.3)
    plan = pane.plan
    assert plan.mode == 'stitched' and 'Objects in the stitched view' in plan.label
    h = sres.hotspots[0]
    q = plan.w2s(h.x, h.y)
    QApplication.sendEvent(plan, QMouseEvent(QEvent.Type.MouseMove, QPointF(q), plan.mapToGlobal(QPointF(q)),
                                             Qt.MouseButton.NoButton, Qt.MouseButton.NoButton,
                                             Qt.KeyboardModifier.NoModifier))
    pump(app, 0.2)
    (shown, raw, by), s2, _ = plan.hover['stitched']
    assert s2 is sres and raw >= 0.25 and shown < 0.25
    assert plan._smears and all('points' in d for d in plan._smears)
    img = plan.grab().toImage()
    assert img.width() > 100
    # the 3D view's ground follows the heatmap too
    g = pane.scene.ground()
    assert g is not None and tuple(g.extent) == pytest.approx((sres.x0, sres.y0, sres.x0 + sres.nx * sres.cell,
                                                                sres.y0 + sres.ny * sres.cell))
    win.actions_by_id['heat_none'].trigger()
    pump(app, 0.1)
    assert not errors, errors


def test_object_choice_and_seam_advice_are_undoable(app, win, errors):
    sres = _stitched_ready(app, win)
    win.set_page('camviews')
    pg = win._page_widgets['camviews']
    st = win.store
    # the object: Adult 1.8 m from the combo, one undo step
    i = next(k for k in range(pg.obj_combo.count()) if pg.obj_combo.itemData(k) == 1.8)
    pg.obj_combo.activated.emit(i)
    _wait(app, lambda: win.ctx.stitched.result is not None and win.ctx.stitched.result.settings.height == 1.8, 60)
    assert st.project.analysis.stitched['height'] == 1.8
    st.undo.undo()
    _wait(app, lambda: win.ctx.stitched.result.settings.height == 0.9, 60)
    assert pg.obj_combo.currentData() == 0.9
    # bird's-eye overlay
    pg.obj_check.setChecked(True)
    pump(app, 0.2)
    assert pg.bev.show_objects
    pg.bev.grab()
    # advice: Suggest, then Apply as one undo step
    seams0 = dict(st.variant.seams)
    QTest.mouseClick(pg.b_suggest, Qt.MouseButton.LeftButton)
    _wait(app, lambda: win.ctx.stitched.advice is not None, 60)
    ch = win.ctx.stitched.advice.changes
    assert ch and pg.b_apply.isEnabled() and '−' in pg.adv_note.text()
    assert any(pg.table.item(4, j).text() != '—' for j in range(pg.table.columnCount()))
    QTest.mouseClick(pg.b_apply, Qt.MouseButton.LeftButton)
    pump(app, 0.2)
    assert all(st.variant.seams.get(k) == v for k, v in ch.items())
    assert win.ctx.stitched.advice is None and not pg.b_apply.isEnabled()
    st.undo.undo()
    pump(app, 0.2)
    assert dict(st.variant.seams) == seams0
    pg.obj_check.setChecked(False)
    assert not errors, errors


def test_one_camera_and_no_cameras(sample):
    p, e, r = sample
    q = copy.deepcopy(p)
    for c in q.variant.cameras[1:]:
        c.visible = False
    rq = e.analyse(q)
    s = so.analyse(q, e, rq)
    assert not rq.seams and not s.vanish.any() and s.seams == []
    assert so.suggest_seams(s).changes == {} and so.issue(s) is None
    for c in q.variant.cameras:
        c.visible = False
    rq = e.analyse(q)
    s = so.analyse(q, e, rq)
    assert s.blind[s.considered].all() and s.metrics['vanish_m2'] == 0 and s.hotspots == []
    assert so.smears(q, e, rq, 4.0, 0.0) == []
