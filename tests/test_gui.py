"""GUI tests driven with real mouse/keyboard events on the offscreen platform (QPainter 3D renderer)."""
import os
import tempfile
import time

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['XDG_CONFIG_HOME'] = tempfile.mkdtemp(prefix='vantage-test-cfg-')

import numpy as np
import pytest
from PyQt6.QtCore import Qt, QPoint, QPointF
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication


@pytest.fixture(scope='module')
def app():
    from vantage.ui.viewport3d import configure_default_format
    configure_default_format()
    a = QApplication.instance() or QApplication([])
    from vantage.ui.theme import THEME, load_fonts
    load_fonts()
    THEME.apply(a, 'dark')
    return a


def pump(app, s=0.3):
    t = time.time()
    while time.time() - t < s:
        app.processEvents()
        time.sleep(0.01)


def settle(app, w, timeout=60):
    pump(app, 0.2)
    t = time.time()
    while (w.ctx.analysis._busy or w.ctx.analysis.timer.isActive()) and time.time() - t < timeout:
        pump(app, 0.05)
    pump(app, 0.2)


@pytest.fixture()
def win(app):
    from vantage.core import samples
    from vantage.ui.mainwindow import MainWindow
    w = MainWindow(samples.tigor_sample_project(), soft_3d=True)
    w.resize(1600, 1000)
    w.show()
    t = time.time()
    while w.ctx.analysis.result is None and time.time() - t < 60:
        pump(app, 0.05)
    settle(app, w)
    yield w
    settle(app, w)          # a run a test's last edit started must not emit into a deleted window
    w.store.undo.setClean()
    w.cancel_jobs()
    w.close()
    w.deleteLater()
    pump(app, 0.1)


def vp_of(w):
    return w.viewarea.panes[0].vp3d


def screen_of(vp, P):
    s, _, ok = vp.project([P])
    assert ok[0]
    return QPoint(int(round(s[0][0])), int(round(s[0][1])))


def test_click_selects_camera(app, win):
    vp = vp_of(win)
    f = win.store.variant.by_tag('F')
    QTest.mouseClick(vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, screen_of(vp, f.pos()))
    pump(app)
    assert win.store.selected.uid == f.uid


def test_gizmo_drag_moves_linked_pair_and_undo(app, win):
    vp = vp_of(win)
    st = win.store
    ml = st.variant.by_tag('ML')
    st.select_cams([ml.uid])
    win.ctx.set_tool('move')
    pump(app)
    scene = win.viewarea.panes[0].scene
    pos, L, axes = scene._gizmo_axes(vp, ml)
    a = screen_of(vp, pos + axes[0] * L * 0.8)
    b = screen_of(vp, pos + axes[0] * L * 1.6)
    QTest.mousePress(vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, a)
    for k in range(1, 6):
        QTest.mouseMove(vp, a + (b - a) * k / 5)
        pump(app, 0.02)
    QTest.mouseRelease(vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, b)
    settle(app, win)
    ml2 = st.cam(ml.uid)
    mr2 = st.cam(ml.link_uid)
    assert ml2.x > 1.9, ml2.x
    assert abs(mr2.x - ml2.x) < 1e-9 and abs(mr2.y + ml2.y) < 1e-9
    st.undo.undo()
    settle(app, win)
    assert abs(st.cam(ml.uid).x - 1.88) < 1e-9


def test_place_tool_adds_camera(app, win):
    vp = vp_of(win)
    st = win.store
    n = len(st.variant.cameras)
    win.ctx.set_tool('place')
    pump(app)
    # a point on the left body side, rear door
    pt = screen_of(vp, np.array([0.9, 0.84, 0.7]))
    QTest.mouseMove(vp, pt)
    pump(app)
    QTest.mouseClick(vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, pt)
    settle(app, win)
    assert len(st.variant.cameras) == n + 1
    c = st.variant.cameras[-1]
    assert c.y > 0.8 and 0.3 < c.z < 1.2
    st.undo.undo()
    settle(app, win)
    assert len(st.variant.cameras) == n


def test_measure_tool(app, win):
    vp = vp_of(win)
    win.ctx.set_tool('measure')
    pump(app)
    sc = win.viewarea.panes[0].scene
    for P in ((5.0, -2.5, 0.0), (5.0, 2.5, 0.0)):
        QTest.mouseClick(vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, screen_of(vp, np.array(P)))
        pump(app, 0.1)
    assert len(sc.measure) == 2
    d = np.linalg.norm(sc.measure[1] - sc.measure[0])
    assert abs(d - 5.0) < 0.15


def test_plan_view_drag_camera(app, win):
    st = win.store
    win.set_view('top')
    settle(app, win)
    plan = win.viewarea.panes[0].plan
    qr = st.variant.by_tag('QR')
    a = plan.w2s(qr.x, qr.y)
    a = QPoint(int(a.x()), int(a.y()))
    b = a + QPoint(25, 0)
    QTest.mousePress(plan, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, a)
    for k in range(1, 6):
        QTest.mouseMove(plan, a + (b - a) * k / 5)
        pump(app, 0.02)
    QTest.mouseRelease(plan, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, b)
    settle(app, win)
    q2 = st.cam(qr.uid)
    assert q2.x > 0.05, q2.x
    # surface snap keeps it on the body side
    assert abs(abs(q2.y) - 0.88) < 0.1
    st.undo.undo()
    settle(app, win)


def test_bev_seam_handle_drag(app, win):
    st = win.store
    win.set_page('camviews')
    settle(app, win)
    bev = win.page('camviews').bev
    bev.fit()
    pump(app)
    res = win.ctx.analysis.result
    s0 = res.seams[0]
    a_, b_ = bev._seam_segment(res, s0.angle)
    a = bev.w2s(*b_)
    a = QPoint(int(a.x()), int(a.y()))
    b = a + QPoint(0, 30)
    QTest.mousePress(bev, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, a)
    QTest.mouseMove(bev, b)
    pump(app, 0.05)
    QTest.mouseRelease(bev, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, b)
    settle(app, win)
    assert f'{s0.a}|{s0.b}' in st.variant.seams
    assert any(not s.auto for s in win.ctx.analysis.result.seams)


def test_inspector_number_field(app, win):
    st = win.store
    ml = st.variant.by_tag('ML')
    st.select_cams([ml.uid])
    settle(app, win)
    page = win.inspector.pages['camera']
    page.px.lineEdit().setText('1.88+0.05')
    page.px._on_finished()
    settle(app, win)
    assert abs(st.cam(ml.uid).x - 1.93) < 1e-6
    page.ry.lineEdit().setText('70deg')
    page.ry._on_finished()
    settle(app, win)
    assert st.cam(ml.uid).yaw == 70.0 and st.cam(ml.link_uid).yaw == -70.0


def test_cameras_table_paste(app, win):
    st = win.store
    tab = win.bottom.cameras
    win.bottom.show_tab('cameras')
    pump(app)
    QApplication.clipboard().setText('Tag\tName\tX\tY\tZ\nF\tFront\t3.25\t0\t0.63\n')
    tab.paste()
    settle(app, win)
    f = st.variant.by_tag('F')
    assert abs(f.x - 3.25) < 1e-6 and abs(f.z - 0.63) < 1e-6


def test_command_palette_runs_action(app, win):
    from vantage.ui.dialogs.palette import CommandPalette
    cp = CommandPalette(win.ctx, win, '2×2')
    cp.show()
    pump(app)
    cp.run(cp.list.currentItem())
    pump(app)
    assert win.viewarea.mode == 'quad'


def test_export_everything(app, win, tmp_path):
    from vantage.ui.dialogs.export_dialog import ExportDialog
    from vantage.core import exporters as ex
    d = ExportDialog(win.ctx, win)
    d.out.setText(str(tmp_path))
    d.open_after.setChecked(False)
    for f in ex.FORMATS:
        d.rows[f.id].cb.setChecked(True)
    d.do_export()
    files = [os.path.join(r, f) for r, _, fs in os.walk(tmp_path) for f in fs]
    assert len(files) >= 20
    names = {os.path.basename(f) for f in files}
    for n in ('mount-sheet.pdf', 'report.html', 'extrinsics.json', 'vantage_cameras.urdf', 'jetson_cameras.json',
              'mount-points.dxf', 'mount-points.step', 'birds-eye.png', 'viewport.png', 'bev_lut.json', 'F_map.bin'):
        assert n in names, n


def test_import_wizard_obj(app, win, tmp_path):
    from vantage.ui.dialogs.import_wizard import ImportWizard
    st = win.store
    m = st.project.vehicle.mesh
    path = str(tmp_path / 'body.obj')
    with open(path, 'w') as fh:
        for x, y, z in m.vertices:              # glTF axes: +Y up, +Z forward, mm
            fh.write(f'v {y * 1000:.2f} {z * 1000:.2f} {x * 1000:.2f}\n')
        for i, p in enumerate(m.parts):
            fh.write(f'g {p.name}\nusemtl {"glass" if p.name == "Glass" else "paint"}\n')
            for f in m.faces[m.face_part == i]:
                fh.write(f'f {f[0] + 1} {f[1] + 1} {f[2] + 1}\n')
    wiz = ImportWizard(win.ctx, win, path=path)
    wiz.show()
    t = time.time()
    while wiz.lm is None and time.time() - t < 30:
        pump(app, 0.05)
    assert wiz.units == 'mm' and wiz.up == '+Y' and wiz.fwd == '+Z'
    wiz.go(2)
    pump(app)
    meas = wiz.measured()
    assert abs(meas['Length'] - 3.993) < 0.01 and abs(meas['Wheelbase'] - 2.45) < 0.01
    wiz.go(4)
    wiz.finish()
    settle(app, win)
    assert st.project.vehicle.mesh is not m
    assert abs(st.project.vehicle.measured()['Height'] - 1.532) < 0.01
    st.undo.undo()
    settle(app, win)
    assert st.project.vehicle.mesh is m


def test_theme_switch_and_pages(app, win):
    for page in ('camviews', 'compare', 'photo', 'targets', 'tolerance', 'optimiser', 'install', 'viewports'):
        win.set_page(page)
        pump(app, 0.2)
    assert win.minimumSizeHint().width() < 1366
    win.set_theme('light')
    pump(app, 0.3)
    win.set_theme('dark')
    pump(app, 0.3)


def _wait(app, cond, timeout=120):
    t = time.time()
    while not cond() and time.time() - t < timeout:
        pump(app, 0.05)
    assert cond()


def test_installation_check_page(app, win, tmp_path):
    win.actions_by_id['install'].trigger()
    pump(app, 0.2)
    assert win.page_name() == 'install'
    pg = win.page('install')
    pg.sim_res.setCurrentIndex(1)                    # half resolution keeps the test quick
    pg.sim_ang.set_value(1.5)
    pg.sim_pos.set_value(8)
    QTest.mouseClick(pg.b_run, Qt.MouseButton.LeftButton)
    _wait(app, lambda: not pg.busy and pg.results)
    cams = win.store.variant.cameras
    assert len(pg.results) == len(cams)
    for c in cams:
        r = pg.results[c.uid]
        assert r.pos is not None, (c.tag, r.message)
        d = pg.truth[c.uid]
        assert np.abs(r.dang_deg - d[:3]).max() < 0.35, c.tag
    assert pg.table.rowCount() == len(cams) and 'recovered' in pg.summary.text()
    assert set(pg.stitched) == {'planned', 'measured'} and pg.bev.img is not None
    pg.which.buttons['planned'].click()
    assert pg.bev.img is pg.stitched['planned']
    # double-click a tile enlarges it, again restores the grid
    tile = pg.tiles[cams[2].uid]
    QTest.mouseDClick(tile, Qt.MouseButton.LeftButton)
    pump(app, 0.1)
    assert pg.single and pg.selected == cams[2].uid
    QTest.mouseDClick(tile, Qt.MouseButton.LeftButton)
    assert not pg.single
    paths = pg.export_luts(str(tmp_path))
    assert any(x.endswith('bev_lut.json') for x in paths)
    pg.save_report(str(tmp_path / 'r.json'))
    import json
    rep_ = json.loads((tmp_path / 'r.json').read_text('utf-8'))
    assert len(rep_['cameras']) == len(cams) and 'measured' in rep_['cameras'][0]
    pg.save_captures(str(tmp_path / 'caps'))
    assert len(os.listdir(tmp_path / 'caps')) == len(cams)
    # the saved captures load back as file sources and check the same
    first = {c.tag: pg.results[c.uid] for c in cams}
    pg.load_folder(str(tmp_path / 'caps'))
    assert all(pg.sources[c.uid]['kind'] == 'file' for c in cams) and not pg.results
    pg.run(check=True)
    _wait(app, lambda: not pg.busy and pg.results)
    measured = {c.tag: pg.results[c.uid] for c in cams}
    for tag, r in measured.items():
        assert np.allclose(r.pos, first[tag].pos, atol=1e-4) and r.yaw == pytest.approx(first[tag].yaw, abs=1e-2)
    n_var = len(win.store.project.variants)
    pg.apply_as_built()
    pump(app, 0.2)
    assert len(win.store.project.variants) == n_var + 1
    v = win.store.variant
    assert v.name.endswith('as-built')
    ml = v.by_tag('ML')
    assert ml.yaw == pytest.approx(measured['ML'].yaw, abs=1e-3) and ml.link_uid == ''
    # the page follows the new variant and keeps the loaded files
    assert list(pg.tiles) == [c.uid for c in v.cameras]
    assert all(pg.sources[c.uid]['kind'] == 'file' for c in v.cameras)
    win.store.undo.undo()
    pump(app, 0.2)
    assert len(win.store.project.variants) == n_var


def test_installation_live_source_error(app, win):
    win.set_page('install')
    pg = win.page('install')
    uid = win.store.variant.cameras[0].uid
    pg.set_source(uid, 'live')
    combo, edit, _ = pg.rows[uid]
    assert combo.currentData() == 'live' and edit.isEnabled()
    edit.setText('/dev/video-missing')
    pg._edit_done(uid, edit.text())
    _wait(app, lambda: pg.live.get(uid) is not None and pg.live[uid].error, 10)
    pg.set_source(uid, 'synthetic')
    assert uid not in pg.live
    win.set_page('viewports')


def test_installation_check_joint_and_exposure(app, win, tmp_path):
    win.set_page('install')
    pg = win.page('install')
    pg.sim_res.setCurrentIndex(1)
    pg.sim_board.set_value(20)                       # boards laid up to 20 mm off
    pg.sim_expo.set_value(30)
    assert pg.joint_cb.isChecked() and pg.balance_cb.isChecked()
    pg.run(check=True)
    _wait(app, lambda: not pg.busy and pg.results)
    jr = pg.joint
    assert jr is not None and jr.poses and jr.seam_after_mm < jr.seam_before_mm
    assert 'Joint solve' in pg.joint_lbl.text() and pg.btable.rowCount() == len(win.store.project.targets)
    assert set(pg.board_truth) == {t.uid for t in win.store.project.targets}
    assert 'simulated' in pg.btable.item(0, 2).toolTip()
    assert pg.gains and 'gains' in pg.bev.caption
    balanced = pg.stitched['measured'].copy()
    pg.balance_cb.setChecked(False)
    assert not pg.gains and not np.array_equal(balanced, pg.stitched['measured'])
    pg.blend.buttons['multiband'].click()
    pump(app, 0.1)
    assert 'multi-band' in pg.bev.caption
    pg.save_report(str(tmp_path / 'r.json'))
    import json
    rep = json.loads((tmp_path / 'r.json').read_text('utf-8'))
    assert rep['joint']['shared_corners'] > 0 and len(rep['joint']['boards']) == len(win.store.project.targets)
    assert all('own_solve' in r for r in rep['cameras'])
    # without the joint solve every camera stands alone
    pg.joint_cb.setChecked(False)
    pg.run(check=True)
    _wait(app, lambda: not pg.busy and pg.results)
    assert pg.joint is None and pg.btable.isHidden() and not any(r.joint for r in pg.results.values())
    pg.blend.buttons['linear'].click()
    pg.balance_cb.setChecked(True)
    pg.joint_cb.setChecked(True)


def test_lens_calibration_dialog(app, win, tmp_path):
    st = win.store
    f = st.variant.by_tag('F')
    st.select_cams([f.uid])
    pump(app, 0.1)
    insp = win.inspector.pages['camera'] if hasattr(win.inspector, 'pages') and 'camera' in win.inspector.pages else None
    win.actions_by_id['lens_calib'].trigger()
    d = win._calib_dialog
    pump(app, 0.1)
    assert d.cam.currentData() == f.uid and not d.b_run.isEnabled() and not d.b_apply.isEnabled()
    d.source.buttons['synthetic'].click()
    d.syn_n.set_value(12)
    d.render_synthetic()
    _wait(app, lambda: not d.busy and len(d.views) == 12)
    assert d.truth is not None and d.vtable.rowCount() == 12 and d.preview.qimg is not None
    d.run()
    _wait(app, lambda: not d.busy and d.result is not None)
    assert d.result.ok, d.result.message
    assert d.ptable.rowCount() >= 8 and 'simulated truth' in d.diff.text() and d.b_apply.isEnabled()
    assert all(d.vtable.item(i, 3).text() == 'used' for i in range(12))
    d.serial.setText('SN-GUI')
    d.apply()
    pump(app, 0.2)
    cal = st.variant.by_tag('F').calib
    assert cal['serial'] == 'SN-GUI' and cal['views'] == 12
    assert st.project.lens_for(st.variant.by_tag('F')).label == 'unit'
    if insp is not None:
        assert 'SN-GUI' in insp.calib_v.v.text() and not insp.proj.isEnabled()
    d.save(str(tmp_path / 'F.yaml'))
    win.store.undo.undo()
    pump(app, 0.2)
    assert not st.variant.by_tag('F').calib
    d.close()
    # the saved file imports back onto F by its camera name
    imp = win.import_calibrations([str(tmp_path / 'F.yaml')])
    pump(app, 0.1)
    assert imp.combos[0].currentData() == f.uid and 'fits' in imp.table.item(0, 7).text()
    imp.apply()
    pump(app, 0.2)
    assert st.variant.by_tag('F').calib['serial'] == 'SN-GUI'
    out = win.export_calibrations(str(tmp_path / 'units.json'))
    assert out and 'SN-GUI' in (tmp_path / 'units.json').read_text('utf-8')
    st.select_cams([f.uid])
    win.clear_calibration()
    pump(app, 0.2)
    assert not st.variant.by_tag('F').calib


def test_load_cases_dialog(app, win):
    d = win.load_cases()
    pump(app, 0.1)
    assert d.table.rowCount() == 4
    d.table.item(1, 2).setText('40')
    d.run()
    _wait(app, lambda: d.results is not None)
    assert len(d.results) == 4 and d.results[1].rear_mm == 40
    assert win.store.project.analysis.load_cases[1][2] == 40
    assert d.table.item(3, 9).text()
    assert d.results[3].seam_ghost_cm > d.results[0].seam_ghost_cm
    d.close()


def test_aim_tool_in_3d_and_plan(app, win):
    st = win.store
    vp = vp_of(win)
    f = st.variant.by_tag('F')
    st.select_cams([f.uid])
    win.actions_by_id['aim_ground'].trigger()
    assert win.ctx.tool == 'aim'
    target = np.array([f.x + 2.5, f.y + 1.0, 0.0])
    QTest.mouseClick(vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, screen_of(vp, target))
    pump(app)
    assert win.ctx.tool == 'select'
    from vantage.core.mathutil import rot_zyx
    c = st.variant.by_tag('F')
    fwd = rot_zyx(c.yaw, c.pitch, c.roll)[:, 0]
    d = target - c.pos()
    assert np.degrees(np.arccos(fwd @ d / np.linalg.norm(d))) < 1.5
    # Esc cancels without aiming
    win.actions_by_id['aim_ground'].trigger()
    QTest.keyClick(vp, Qt.Key.Key_Escape)
    assert win.ctx.tool == 'select'
    # nothing selected: the tool does not start
    st.select_cams([])
    win.actions_by_id['aim_ground'].trigger()
    assert win.ctx.tool == 'select'


def test_ecu_link_budget(app, win):
    st = win.store
    st.select_obj(('ecu',))
    pump(app, 0.2)
    pg = win.inspector.pages['ecu']
    pg.refresh()
    assert pg.bt.rowCount() == 2
    assert pg.bt.item(0, 3).text() == '62 %' and pg.bt.item(0, 2).text() == '6.23 / 10'
    pg.csi.committed.emit(5.0)
    pump(app, 0.1)
    pg.refresh()
    assert st.project.ecu.csi_gbps_per_deser == 5.0
    assert pg.bt.item(0, 3).text() == '125 %' and 'Deserializer A' in pg.budget_note.text()
    st.undo.undo()


def test_photo_matched_shading_main_view(app, win):
    from vantage.core import photos as ph
    ps = win.store.project.photos
    solved = [k for k in range(len(ps.frames)) if ps.frames[k].pose is not None]
    if not solved:
        pytest.skip('sample photos have no solved pose')
    win.set_shading('photo')
    pane = win.viewarea.active
    assert win.ctx.shading == 'photo' and pane.vp3d.cam.custom is not None
    assert pane.scene.photo_frame in solved and pane.scene.background_image(pane.vp3d) is not None
    if len(solved) > 1:
        k0 = ps.current
        win.actions_by_id['photo_next'].trigger()
        assert ps.current != k0 and pane.scene.photo_frame == ps.current
    win.set_view('persp')
    assert win.ctx.shading == 'solid' and pane.vp3d.cam.custom is None and pane.scene.photo_frame is None
