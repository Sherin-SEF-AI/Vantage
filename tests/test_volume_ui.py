"""The Blind volume page and the 3D overlay, driven like a user (offscreen, software 3D renderer)."""
import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest

from test_gui import app, win, pump, settle, vp_of, _wait     # noqa: F401  (fixtures)


def _open(app, win):
    win.actions_by_id['blind_volume'].trigger()
    pump(app, 0.2)
    pg = win.page('volume')
    assert win.page_name() == 'volume' and pg.isVisible()
    return pg


def test_page_runs_and_shows_the_result(app, win, tmp_path):
    pg = _open(app, win)
    ctl = win.ctx.volume
    QTest.mouseClick(pg.b_run, Qt.MouseButton.LeftButton)
    _wait(app, lambda: ctl.result is not None and not ctl.busy, 60)
    pump(app, 0.2)
    vr = ctl.result
    assert pg.table.rowCount() == len(vr.objects) and pg.table.item(0, 0).text() == vr.objects[0][0]
    assert pg.metric['blind'].text().endswith('m³') and 'points' in pg.status.text()
    assert pg.rose.series and len(pg.rose.sector_deg) == 72
    # the height slider drives the plan slice and the 3D cut-away
    k = pg.slider.value()
    assert abs(vr.z[k] - 0.9) < vr.dz                       # starts at child height
    k = vr.level_index(0.6)
    assert vr.level_blind_m2[k] > 0.5
    pg.slider.setValue(k)
    pump(app, 0.1)
    assert pg.map.level == k and ctl.cut_z == pytest.approx(vr.z[k]) and ctl.cut_level() == k + 1
    td = ctl.tri_draw(gl=False)
    assert td is not None and td.tri[:, :, 2].max() <= (k + 1) * vr.dz + 1e-6
    assert any(t is td for t in pg.scene.tris_extra(pg.view))
    pg.mode.buttons['detect'].click()
    pump(app, 0.1)
    assert pg.map.mode == 'detect' and not pg.slider_row.isVisible()
    img = pg.grab().toImage()
    assert img.width() > 0
    paths = pg.export(str(tmp_path))
    assert all((tmp_path / n).exists() for n in ('blind_volume.json', 'blind_volume.obj')) and len(paths) == 2


def test_settings_are_undoable_and_rerun_by_themselves(app, win):
    pg = _open(app, win)
    ctl = win.ctx.volume
    _wait(app, lambda: ctl.result is not None and not ctl.busy, 60)    # the page runs a quick study when shown
    f, _ = pg.fields['height']
    f.set_value(1.2)
    f.committed.emit(1.2)
    pump(app, 0.1)
    assert win.store.project.analysis.volume['height'] == pytest.approx(1.2)
    _wait(app, lambda: ctl.result is not None and ctl.result.height == pytest.approx(1.2) and not ctl.busy, 60)
    assert not ctl.stale and pg.slider.maximum() == len(ctl.result.z) - 1
    win.store.undo.undo()
    pump(app, 0.1)
    assert win.store.project.analysis.volume.get('height', 2.0) == pytest.approx(2.0)
    assert f.value() == pytest.approx(2.0)                                # the field follows undo
    _wait(app, lambda: ctl.result.height == pytest.approx(2.0) and not ctl.busy, 60)
    # an edit to the cameras makes the shown result stale until it has run again
    before = ctl.result
    c = win.store.variant.by_tag('F')
    win.store.set_cam(c.uid, pitch=c.pitch + 8.0)
    settle(app, win)
    _wait(app, lambda: ctl.result is not before and not ctl.busy, 60)
    assert not ctl.stale


def test_overlay_in_the_main_3d_view_and_a_new_project(app, win):
    ctl = win.ctx.volume
    vp = vp_of(win)
    assert not any(t is ctl.tri_draw(False) for t in vp.provider.tris_extra(vp) if ctl.result is not None)
    win.actions_by_id['ov_volume'].trigger()                    # View ▸ Overlays ▸ Blind volume
    _wait(app, lambda: ctl.result is not None and not ctl.busy, 60)
    pump(app, 0.2)
    td = ctl.tri_draw(gl=False)
    assert td is not None and any(t is td for t in vp.provider.tris_extra(vp))
    assert (td.colors[:, 3] < 1).all()                           # translucent
    assert 'blind space up to' in vp.provider.hud_label(vp)
    vp.grab()
    win.actions_by_id['ov_volume'].trigger()
    pump(app, 0.1)
    assert not any(t is td for t in vp.provider.tris_extra(vp))
    win.store.project_replaced.emit()                            # File ▸ New / Open: never the old project's volume
    assert ctl.result is None
