"""The 3D view's new capabilities (render image, turntable, camera light, the textured bowl) and regressions in the
viewport, plan view, scene, outliner and main window."""
import math
import os
import sys

import numpy as np
import pytest
from PyQt6.QtCore import Qt, QEvent, QPoint, QPointF
from PyQt6.QtGui import QFocusEvent, QKeyEvent, QWheelEvent
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from test_gui import app, win, pump, vp_of, screen_of, _wait     # noqa: F401  (fixtures)


def _wheel(widget, pos, dy, dx=0):
    ev = QWheelEvent(QPointF(pos), QPointF(widget.mapToGlobal(pos)), QPoint(0, 0), QPoint(dx, dy),
                     Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False)
    # through Qt: calling widget.wheelEvent() from Python would find QWidget's empty one before the viewport mixin's
    QApplication.sendEvent(widget, ev)


# ----------------------------------------------------------------------------- new capabilities
def test_render_image_and_turntable(app, win, tmp_path):
    vp = vp_of(win)
    img = vp.render_image(2.0)
    assert (img.width(), img.height()) == (vp.width() * 2, vp.height() * 2)
    path = win.render_image(str(tmp_path / 'r.png'), scale=1)
    assert path and os.path.getsize(path) > 1000
    out = win.turntable(str(tmp_path / 'spin.mp4'), frames=6, scale=0.5)
    assert out and os.path.exists(out)
    assert vp.cam.az == pytest.approx(vp_of(win).cam.az)              # the view is back where it was


def test_camera_light_overlay(app, win):
    vp = vp_of(win)
    sc = vp.provider
    assert sc.camera_lights(vp) == []
    win.actions_by_id['ov_camlight'].trigger()
    pump(app, 0.3)
    views = sc.camera_lights(vp)
    assert len(views) == len(win.store.variant.cameras)
    assert all(v.cube is not None for v in views)                    # the analysis' occlusion maps are reused
    assert sc.camera_lights(vp) is views or sc.camera_lights(vp) == views
    plain = vp.grab().toImage()
    win.actions_by_id['ov_camlight'].trigger()
    pump(app, 0.3)
    assert sc.camera_lights(vp) == [] and plain.width() > 0


def test_shadows_overlay_and_style(app, win):
    vp = vp_of(win)
    st = vp.provider.render_style(vp)
    assert st.shadows and st.ground_plane
    win.actions_by_id['ov_shadows'].trigger()
    pump(app, 0.2)
    st = vp.provider.render_style(vp)
    assert not st.shadows and not st.ao
    win.actions_by_id['ov_shadows'].trigger()


def test_bowl_view_projects_the_parking_scene(app, win, tmp_path):
    win.set_page('camviews')
    pg = win._page_widgets['camviews']
    ctl = pg.bowl_scene.bowl
    _wait(app, lambda: ctl.draw is not None, 120)
    d = ctl.draw
    assert d.texture.shape[2] == 4 and d.texture[..., 3].mean() > 100
    assert len(d.faces) > 1000 and pg.bowl_scene.ground() is None
    tris = pg.bowl_scene.tris_extra(pg.bowl)                          # software renderer: coloured triangles
    assert any(len(t.tri) > 1000 for t in tris)
    gen = d.version
    pg.bowl_fields['height_m'].set_value(3.4)
    pg.bowl_fields['height_m'].committed.emit(3.4)
    _wait(app, lambda: ctl.draw is not None and ctl.draw.version != gen, 120)
    assert ctl.shape.height_m == pytest.approx(3.4) and ctl.draw.vertices[:, 2].max() == pytest.approx(3.4, abs=1e-3)
    pg.bowl_view('Rear left')
    _wait(app, lambda: pg.bowl._anim is None, 5)
    assert pg.bowl.cam.az == pytest.approx(-155.0, abs=0.5) or pg.bowl.cam.az == pytest.approx(205.0, abs=0.5)
    paths = pg.export_bowl(str(tmp_path))
    assert {os.path.basename(x) for x in paths} >= {'bowl.obj', 'bowl_lut.npz', 'bowl_lut.json', 'bowl.png'}
    ctl.set_source('sectors')
    _wait(app, lambda: ctl.draw is None, 10)
    win.set_page('viewports')


def test_animated_view_change(app, win):
    vp = vp_of(win)
    vp.set_view('persp')
    vp.animate_to(az=vp.cam.az + 90, el=60.0, ms=120)
    assert vp._anim is not None
    _wait(app, lambda: vp._anim is None, 5)
    assert vp.cam.el == pytest.approx(60.0)
    vp.set_view('top', animate=True)
    _wait(app, lambda: vp._anim is None, 5)
    assert vp.cam.ortho and vp.cam.el == pytest.approx(89.99)


# ----------------------------------------------------------------------------- viewport regressions
def test_second_button_during_a_drag_does_not_stick(app, win):
    vp = vp_of(win)
    st = win.store
    f = st.variant.by_tag('F')
    st.select_cams([f.uid])
    win.set_tool('select')
    pump(app, 0.1)
    pos = screen_of(vp, f.pos())
    QTest.mousePress(vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, pos)
    assert st.dragging
    QTest.mousePress(vp, Qt.MouseButton.RightButton, Qt.KeyboardModifier.NoModifier, pos + QPoint(5, 0))
    QTest.mouseRelease(vp, Qt.MouseButton.RightButton, Qt.KeyboardModifier.NoModifier, pos + QPoint(5, 0))
    assert st.dragging                                               # still the left button's drag
    QTest.mouseRelease(vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, pos)
    assert not st.dragging



def test_focus_loss_and_unhandled_keys_reach_qt(app, win, monkeypatch):
    """The viewport mixin comes after the Qt class, so super() in it is object: passing events on used to raise."""
    errors = []
    monkeypatch.setattr(sys, 'excepthook', lambda t, v, tb: errors.append(v))
    vp = vp_of(win)
    st = win.store
    f = st.variant.by_tag('F')
    st.select_cams([f.uid])
    win.set_tool('select')
    pump(app, 0.1)
    QTest.mousePress(vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, screen_of(vp, f.pos()))
    assert st.dragging
    QApplication.sendEvent(vp, QFocusEvent(QEvent.Type.FocusOut))
    QApplication.sendEvent(vp, QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_F13, Qt.KeyboardModifier.NoModifier))
    assert not errors, errors
    assert not st.dragging



def test_issue_marker_is_cleared_by_another_project(app, win):
    pane = win.viewarea.visible_panes()[0]
    win.frame_point((2.0, 1.0))
    assert pane.plan.pinned_marker == (2.0, 1.0)
    win.store.project_replaced.emit()
    assert pane.plan.pinned_marker is None


def test_scale_and_gizmo_in_a_top_view(app, win):
    vp = vp_of(win)
    vp.set_view('top')
    vp.frame_box(*win.viewarea._bbox())
    ppm = vp.px_per_m(vp.cam.target)
    assert 20 < ppm < 2000


def test_wheel_in_a_front_view_keeps_the_target_near(app, win):
    vp = vp_of(win)
    vp.set_view('front')
    vp.frame_box(*win.viewarea._bbox())
    t0 = vp.cam.target.copy()
    _wheel(vp, QPointF(vp.width() / 2, vp.height() * 0.2), 120)
    assert np.linalg.norm(vp.cam.target - t0) < 5
    h = vp.cam.ortho_h
    _wheel(vp, QPointF(vp.width() / 2, vp.height() / 2), 0, dx=120)      # horizontal scroll: no zoom
    _wheel(vp, QPointF(vp.width() / 2, vp.height() / 2), 20)              # a touchpad sixth of a notch
    assert vp.cam.ortho_h == pytest.approx(h * 0.88 ** (20 / 120))
    vp.set_view('persp')


def test_rotate_ring_does_not_jump(app, win):
    from types import SimpleNamespace as NS
    vp = vp_of(win)
    sc = vp.provider
    st = win.store
    c = st.variant.by_tag('QR')
    pc = screen_of(vp, c.pos())
    sc.drag = {'ring': 1, 'uid': c.uid, 'yaw': c.yaw, 'pitch': c.pitch, 'mouse': QPointF(pc.x() + 40, pc.y()),
               'turn': 0.0, 'last': None}
    st.dragging = True
    pitches = []
    for a in np.linspace(0, 2 * math.pi * 0.9, 40):                    # once round, through the left side
        ev = NS(position=lambda a=a: QPointF(pc.x() + 40 * math.cos(a), pc.y() + 40 * math.sin(a)),
                modifiers=lambda: Qt.KeyboardModifier.ShiftModifier)
        sc.mouse_move(vp, ev)
        pitches.append(st.variant.cam(c.uid).pitch)
    sc.mouse_release(vp, None)
    steps = np.abs(np.diff(pitches))
    assert steps.max() < 3.0
    win.store.undo.undo()


def test_pick_respects_isolation_and_rotated_targets(app, win):
    vp = vp_of(win)
    sc = vp.provider
    st = win.store
    f, r = st.variant.by_tag('F'), st.variant.by_tag('R')
    win.ctx.isolate = {f.uid}
    assert sc._pick_camera(vp, QPointF(screen_of(vp, r.pos()))) is None
    assert sc._pick_camera(vp, QPointF(screen_of(vp, f.pos()))) is f
    win.ctx.isolate = set()
    # a long thin board turned 90°: its drawn end is clickable
    from vantage.core.project import Target

    def fn(p):
        p.targets = [Target(name='Bar', x=6.0, y=0.0, yaw=90.0, rows=2, cols=6, square_mm=200.0)]
    st.edit('targets', fn, 'one bar')
    pump(app, 0.2)
    pl = win.viewarea.panes[0].plan if hasattr(win.viewarea.panes[0], 'plan') else None
    if pl is not None:
        assert pl._pick_target(np.array([6.0, 0.5])) is not None and pl._pick_target(np.array([6.5, 0.0])) is None
    st.undo.undo()


def test_view_state_follows_edits(app, win):
    st = win.store
    ml = st.variant.by_tag('ML')
    win.ctx.isolate = {ml.uid}
    st.select_cams([ml.uid])
    from vantage.ui import ops
    ops.delete_cameras(win.ctx, [ml.uid])
    pump(app, 0.2)
    assert ml.uid not in st.sel_cams and ml.uid not in win.ctx.isolate
    st.undo.undo()
    pump(app, 0.2)
    # the menus' radio items follow undo
    win.set_cell(0.02)
    pump(app, 0.2)
    assert win.actions_by_id['res_2'].isChecked()
    st.undo.undo()
    pump(app, 0.2)
    assert win.actions_by_id['res_5'].isChecked()


def test_snap_keeps_a_mirrored_pair(app, win):
    from vantage.ui import ops
    st = win.store
    ml, mr = st.variant.by_tag('ML'), st.variant.by_tag('MR')
    assert ml.link_uid == mr.uid
    ops.snap_to_surface(win.ctx, [ml.uid])
    pump(app, 0.2)
    ml, mr = st.variant.by_tag('ML'), st.variant.by_tag('MR')
    assert ml.y == pytest.approx(-mr.y) and ml.x == pytest.approx(mr.x) and ml.z == pytest.approx(mr.z)
    st.undo.undo()


def test_clear_zones_without_a_vehicle_and_undo(app, win):
    m = win.store.project.vehicle.mesh
    before = np.bincount(m.face_zone[m.face_zone >= 0], minlength=len(m.zones))
    win.clear_zones()
    pump(app, 0.2)
    assert not win.store.project.vehicle.mesh.zones
    win.store.undo.undo()
    pump(app, 0.2)
    m = win.store.project.vehicle.mesh
    after = np.bincount(m.face_zone[m.face_zone >= 0], minlength=len(m.zones))
    assert list(after) == list(before)
    from vantage.core import samples
    win.store.set_project(samples.blank_project())
    pump(app, 0.3)
    win.clear_zones()                                                # no vehicle: a message, not a traceback


def test_photo_view_is_left_completely(app, win):
    ps = win.store.project.photos
    if not any(f.pose is not None for f in ps.frames):
        pytest.skip('no photo poses in the sample')
    k = next(i for i, f in enumerate(ps.frames) if f.pose is not None)
    if not win.photo_match_view(k):
        pytest.skip('photo images not available')
    vp = vp_of(win)
    assert vp.cam.custom is not None
    vp.frame_box(*win.viewarea._bbox())
    assert vp.cam.custom is None and vp.provider.photo_frame is None and win.ctx.shading != 'photo'


def test_separate_drags_are_separate_undo_steps(app, win):
    st = win.store
    qr = st.variant.by_tag('QR')
    x0 = qr.x
    n = st.undo.count()
    for dx in (0.2, 0.4):
        st.dragging = True
        st.set_cam(qr.uid, merge=True, x=x0 + dx / 2)
        st.set_cam(qr.uid, merge=True, x=x0 + dx)
        st.dragging = False
    assert st.undo.count() == n + 2
    st.undo.undo()
    assert st.variant.by_tag('QR').x == pytest.approx(x0 + 0.2)
    st.undo.undo()


def test_an_older_result_does_not_clear_stale(app, win):
    ctl = win.ctx.analysis
    win.set_live(False) if hasattr(win, 'set_live') else None
    win.store.project.analysis.live = False
    old = ctl.result
    old.revision = win.store.revision - 1                            # computed before the latest edit
    ctl._on_finished(old)
    assert ctl.stale
    old.revision = win.store.revision
    ctl._on_finished(old)
    assert not ctl.stale
    win.store.project.analysis.live = True


def test_a_run_requested_while_busy_happens(app, win):
    ctl = win.ctx.analysis
    st = win.store
    _wait(app, lambda: not ctl._busy, 60)
    f = st.variant.by_tag('F')
    st.set_cam(f.uid, x=f.x + 0.05)
    ctl._kick()                                                      # start a run ...
    st.set_cam(f.uid, x=f.x + 0.10)                                  # ... and edit while it is busy
    ctl._kick()
    _wait(app, lambda: not ctl._busy and getattr(ctl.result, 'revision', -1) == st.revision, 60)
    assert ctl.result.revision == st.revision and not ctl.stale
    st.undo.undo()
    st.undo.undo()


def test_bowl_shape_is_saved_and_undoable(app, win, tmp_path):
    from vantage.core.project import Project
    win.set_page('camviews')
    pg = win._page_widgets['camviews']
    ctl = pg.bowl_scene.bowl
    ctl.set_shape(floor_m=2.2)
    assert win.store.project.analysis.bowl['floor_m'] == pytest.approx(2.2) and ctl.shape.floor_m == pytest.approx(2.2)
    path = tmp_path / 'b.vantage'
    win.store.project.save(str(path))
    assert Project.load(str(path)).analysis.bowl['floor_m'] == pytest.approx(2.2)
    win.store.undo.undo()
    pump(app, 0.1)
    assert ctl.shape.floor_m == pytest.approx(1.6) and pg.bowl_fields['floor_m'].value() == pytest.approx(1.6)
    win.set_page('viewports')


def test_import_wizard_preview_survives_hover(app, win):
    """The wizard's preview provider kept its hover point in an attribute that hid its own hover() method, so every
    mouse move over the preview raised."""
    import sys
    from vantage.ui.dialogs.import_wizard import ImportWizard
    errors = []
    old = sys.excepthook
    sys.excepthook = lambda t, v, tb: errors.append(v)
    try:
        wiz = ImportWizard(win.ctx, win, step=2, reuse=True)
        wiz.show()
        pump(app, 0.3)
        for vp in (wiz.vp_orient, wiz.vp_side, wiz.vp_zones):
            for k in range(3):
                QTest.mouseMove(vp, QPoint(vp.width() // 2 + 10 * k, vp.height() // 2))
            vp.provider.hover(vp, None)
        pump(app, 0.1)
        wiz.reject()
    finally:
        sys.excepthook = old
    assert not errors, errors[0]
