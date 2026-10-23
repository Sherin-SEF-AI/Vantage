"""Getting around the 3D view (ui.navigation, ui.viewport3d) and moving things in it (ui.scene's handles): orbiting
about the point under the cursor, zooming towards it, the on-screen navigation control, touchpad, pinch and keyboard
navigation, and the move/rotate handles on cameras, sensors and rig parts with their read-out, snapping and Esc."""
import math

import numpy as np
import pytest
from PyQt6.QtCore import Qt, QEvent, QPoint, QPointF
from PyQt6.QtGui import QKeyEvent, QMouseEvent, QNativeGestureEvent, QPointingDevice, QWheelEvent
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from test_gui import app, win, pump, settle, vp_of, screen_of   # noqa: F401  (fixtures)

from vantage.ui.navigation import EL_MAX, EL_MIN, orbit_about
from vantage.ui.viewport3d import OrbitCam

NOMOD = Qt.KeyboardModifier.NoModifier
LEFT, MID = Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton


def _wheel(vp, pos, dy, dx=0, pixel=None, phase=Qt.ScrollPhase.NoScrollPhase, mods=NOMOD):
    px = QPoint(*pixel) if pixel else QPoint(0, 0)
    ev = QWheelEvent(QPointF(pos), QPointF(vp.mapToGlobal(pos)), px, QPoint(dx, dy), Qt.MouseButton.NoButton, mods,
                     phase, False)
    QApplication.sendEvent(vp, ev)           # through Qt: the viewport's handler is on the mixin


def _mouse(vp, kind, pos, button=LEFT, buttons=None, mods=NOMOD):
    """A mouse event with modifiers (QTest.mouseMove has none)."""
    t = {'press': QEvent.Type.MouseButtonPress, 'move': QEvent.Type.MouseMove,
         'release': QEvent.Type.MouseButtonRelease}[kind]
    b = Qt.MouseButton.NoButton if kind == 'move' else button
    bs = buttons if buttons is not None else (button if kind != 'release' else Qt.MouseButton.NoButton)
    pos = QPointF(pos)
    QApplication.sendEvent(vp, QMouseEvent(t, pos, QPointF(vp.mapToGlobal(pos)), b, bs, mods))


def _drag(app, vp, a, b, button=LEFT, mods=NOMOD, steps=6):
    a, b = QPointF(a), QPointF(b)
    _mouse(vp, 'press', a, button, mods=mods)
    for k in range(1, steps + 1):
        _mouse(vp, 'move', a + (b - a) * (k / steps), button, buttons=button, mods=mods)
        pump(app, 0.02)
    _mouse(vp, 'release', b, button, mods=mods)
    pump(app, 0.05)


def _key(vp, key, mods=NOMOD):
    QApplication.sendEvent(vp, QKeyEvent(QEvent.Type.KeyPress, key, mods))


def _screen(vp, P):
    s, _, ok = vp.project([P])
    assert ok[0]
    return np.array(s[0], float)


def _settle_anim(app, vp):
    for _ in range(60):
        if vp._anim is None:
            return
        pump(app, 0.03)


# ======================================================================= the orbit
@pytest.mark.parametrize('daz,de', [(30, 0), (0, 25), (-70, -20), (15, 80)])
def test_orbit_about_a_point_keeps_it_where_it_is_in_the_view(daz, de):
    cam = OrbitCam()
    cam.target, cam.dist, cam.az, cam.el = np.array([1.0, 0.2, 0.6]), 7.0, 40.0, 25.0
    P = np.array([2.1, 0.7, 1.3])
    before = cam.view() @ np.r_[P, 1.0]
    el = cam.el
    orbit_about(cam, P, daz, de)
    assert cam.view() @ np.r_[P, 1.0] == pytest.approx(before, abs=1e-9)     # same place in the camera's frame
    assert cam.el == pytest.approx(min(max(el + de, EL_MIN), EL_MAX), abs=1e-6)
    assert cam.dist == pytest.approx(7.0, rel=1e-9)


def test_orbit_about_the_target_is_the_plain_orbit():
    cam = OrbitCam()
    cam.target, cam.dist, cam.az, cam.el = np.array([1.0, 0.0, 0.5]), 6.0, 10.0, 30.0
    orbit_about(cam, cam.target.copy(), 25.0, -10.0)
    assert (cam.az, cam.el) == (pytest.approx(35.0), pytest.approx(20.0))
    assert cam.target == pytest.approx([1.0, 0.0, 0.5])


def test_middle_drag_orbits_about_the_body_under_the_cursor(app, win):
    vp = vp_of(win)
    vp.nav_style = 'blender'
    at = QPointF(vp.width() * 0.45, vp.height() * 0.5)
    P = vp.provider.pick_point(vp, at)
    assert P is not None                                   # the car is in the middle of the view
    s0 = _screen(vp, P)
    az0 = vp.cam.az
    _drag(app, vp, at, at + QPointF(120, 40), button=MID)
    assert vp.cam.az != pytest.approx(az0)
    assert np.abs(_screen(vp, P) - s0).max() < 0.5         # the point grabbed stays under where the drag began


def test_wheel_zooms_towards_the_surface_under_the_cursor(app, win):
    vp = vp_of(win)
    at = QPointF(vp.width() * 0.4, vp.height() * 0.45)
    P = vp.provider.pick_point(vp, at)
    assert P is not None
    s0, d0 = _screen(vp, P), np.linalg.norm(vp.cam.eye() - P)
    _wheel(vp, at, 120)
    assert np.abs(_screen(vp, P) - s0).max() < 0.5
    assert np.linalg.norm(vp.cam.eye() - P) == pytest.approx(d0 * 0.88, rel=1e-6)
    # a plain mouse wheel still zooms, a horizontal one does nothing
    dist = vp.cam.dist
    _wheel(vp, at, 0, dx=120)
    assert vp.cam.dist == dist


# ======================================================================= touchpad, pinch, keys
def test_touchpad_two_fingers_orbit_shift_pans_ctrl_zooms(app, win):
    vp = vp_of(win)
    at = QPointF(vp.width() * 0.45, vp.height() * 0.5)
    cam = vp.cam
    az, dist, tgt = cam.az, cam.dist, cam.target.copy()
    _wheel(vp, at, 0, pixel=(40, 0), phase=Qt.ScrollPhase.ScrollBegin)
    _wheel(vp, at, 0, pixel=(40, 0), phase=Qt.ScrollPhase.ScrollUpdate)
    _wheel(vp, at, 0, pixel=(0, 0), phase=Qt.ScrollPhase.ScrollEnd)
    assert cam.az != pytest.approx(az) and cam.dist == pytest.approx(dist, rel=1e-6)
    az = cam.az
    tgt = cam.target.copy()
    _wheel(vp, at, 0, pixel=(30, 20), phase=Qt.ScrollPhase.ScrollUpdate, mods=Qt.KeyboardModifier.ShiftModifier)
    assert cam.az == pytest.approx(az) and np.linalg.norm(cam.target - tgt) > 1e-3
    dist = cam.dist
    _wheel(vp, at, 60, pixel=(0, 30), phase=Qt.ScrollPhase.ScrollUpdate, mods=Qt.KeyboardModifier.ControlModifier)
    assert cam.dist < dist


def test_pinch_zooms(app, win):
    vp = vp_of(win)
    dist = vp.cam.dist
    at = QPointF(vp.width() / 2, vp.height() / 2)
    ev = QNativeGestureEvent(Qt.NativeGestureType.ZoomNativeGesture, QPointingDevice.primaryPointingDevice(), 2, at, at,
                             QPointF(vp.mapToGlobal(at)), 0.25, QPointF(0, 0))
    QApplication.sendEvent(vp, ev)
    assert vp.cam.dist < dist * 0.85


def test_keyboard_views_orbit_pan_and_zoom(app, win):
    vp = vp_of(win)
    vp.setFocus()
    cam = vp.cam
    _key(vp, Qt.Key.Key_7)                                  # the number row: a laptop has no numpad
    _settle_anim(app, vp)
    assert cam.el == pytest.approx(89.99, abs=0.05) and cam.ortho
    _key(vp, Qt.Key.Key_0)
    _settle_anim(app, vp)
    assert not cam.ortho and cam.el == pytest.approx(34.0, abs=0.05)
    az = cam.az
    _key(vp, Qt.Key.Key_Left)
    assert cam.az == pytest.approx(az + 15.0)
    tgt = cam.target.copy()
    _key(vp, Qt.Key.Key_Left, Qt.KeyboardModifier.ShiftModifier)
    assert cam.az == pytest.approx(az + 15.0) and np.linalg.norm(cam.target - tgt) > 1e-3
    dist = cam.dist
    _key(vp, Qt.Key.Key_Plus)
    assert cam.dist == pytest.approx(dist * 0.8)
    _key(vp, Qt.Key.Key_Minus)
    assert cam.dist == pytest.approx(dist)
    _key(vp, Qt.Key.Key_5)
    assert cam.ortho


# ======================================================================= the navigation control
def test_navigation_control_clicks_and_drags(app, win):
    vp = vp_of(win)
    nav = vp.nav
    assert nav.enabled()
    c = nav.centre()
    # hovering lights it up and says what it does
    QTest.mouseMove(vp, c.toPoint())
    pump(app, 0.05)
    assert nav.hot == 'ball' and 'orbit' in nav.tooltip('ball')
    # the top bubble looks straight down
    top = next(q for _, k, s, q in nav.bubbles() if (k, s) == (2, 1))
    assert nav.hit(top) == ('axis', 2, 1)
    QTest.mouseClick(vp, LEFT, NOMOD, top.toPoint())
    _settle_anim(app, vp)
    assert vp.cam.el == pytest.approx(89.99, abs=0.05)
    # dragging the ball orbits, about the view's centre
    vp.set_view('persp')
    az, tgt = vp.cam.az, vp.cam.target.copy()
    _drag(app, vp, c, c + QPointF(-60, 0))
    assert vp.cam.az == pytest.approx(az + 60 * 0.4) and vp.cam.target == pytest.approx(tgt)
    # the magnifier zooms, the hand pans
    btn = {k: nav.button_centre(i) for i, (k, _, _) in enumerate(nav.BUTTONS)}
    assert all(nav.hit(q) == k for k, q in btn.items())
    dist = vp.cam.dist
    _drag(app, vp, btn['zoom'], btn['zoom'] + QPointF(0, -80))
    assert vp.cam.dist < dist * 0.6
    tgt = vp.cam.target.copy()
    _drag(app, vp, btn['pan'], btn['pan'] + QPointF(-50, 30))
    assert np.linalg.norm(vp.cam.target - tgt) > 0.05
    # the frame button brings the whole car back; the cube switches to orthographic
    QTest.mouseClick(vp, LEFT, NOMOD, btn['frame'].toPoint())
    _settle_anim(app, vp)
    lo, hi, _ = vp.provider.home_box(vp)
    assert vp.cam.target == pytest.approx((np.asarray(lo) + np.asarray(hi)) / 2, abs=1e-6)
    QTest.mouseClick(vp, LEFT, NOMOD, btn['ortho'].toPoint())
    assert vp.cam.ortho
    # clicks on the control never reach the scene (no selection change)
    win.store.select_cams([])
    QTest.mouseClick(vp, LEFT, NOMOD, c.toPoint())
    pump(app, 0.05)
    assert win.store.selected is None


def test_navigation_control_stays_out_of_rendered_images(app, win, monkeypatch):
    vp = vp_of(win)
    calls = []
    monkeypatch.setattr(vp.nav, 'paint', lambda p: calls.append(1))
    vp.render_image(1.0)
    assert not calls
    vp.grab()
    assert calls


# ======================================================================= the handles
def _arrow_points(sc, vp, obj, k=0):
    pos, L, axes = sc._gizmo_axes(vp, obj)
    return _screen(vp, pos + axes[k] * L * 0.7), _screen(vp, pos + axes[k] * L * 1.5)


def test_handles_light_up_under_the_pointer_and_grab_from_further(app, win):
    vp = vp_of(win)
    st = win.store
    ml = st.variant.by_tag('ML')
    st.select_cams([ml.uid])
    win.ctx.set_tool('move')
    pump(app, 0.1)
    sc = vp.provider
    a, b = _arrow_points(sc, vp, ml)
    d = (b - a) / np.linalg.norm(b - a)
    off = a + np.array([-d[1], d[0]]) * 8.0                    # 8 px beside the arrow still grabs it
    QTest.mouseMove(vp, QPoint(*np.round(off).astype(int)))
    pump(app, 0.05)
    assert sc.gizmo_hot == ('axis', 0)
    assert vp.cursor().shape() == Qt.CursorShape.SizeAllCursor
    # after grabbing it and letting go, moving off the handles gives the plain cursor back
    _drag(app, vp, QPointF(*off), QPointF(*(off + d * 30)))
    QTest.mouseMove(vp, QPoint(5, vp.height() - 5))
    pump(app, 0.05)
    assert sc.gizmo_hot is None
    assert vp.cursor().shape() == Qt.CursorShape.ArrowCursor
    win.store.undo.undo()


def test_handle_drag_reads_out_snaps_with_ctrl_and_esc_puts_it_back(app, win):
    vp = vp_of(win)
    st = win.store
    ml = st.variant.by_tag('ML')
    x0 = ml.x
    st.select_cams([ml.uid])
    win.ctx.set_tool('move')
    pump(app, 0.1)
    sc = vp.provider
    a, b = _arrow_points(sc, vp, ml)
    n_undo = st.undo.count()
    _mouse(vp, 'press', QPointF(*a))
    _mouse(vp, 'move', QPointF(*((a + b) / 2)), buttons=LEFT, mods=Qt.KeyboardModifier.ControlModifier)
    _mouse(vp, 'move', QPointF(*b), buttons=LEFT, mods=Qt.KeyboardModifier.ControlModifier)
    pump(app, 0.05)
    x = st.cam(ml.uid).x
    assert x > x0 + 0.02
    assert round((x - x0) / 0.01, 6) == pytest.approx(round((x - x0) / 0.01)), x - x0     # whole centimetres
    assert sc.drag['readout'].startswith('Δx +') and 'mm' in sc.drag['readout']
    vp.grab()                                               # the read-out paints
    _key(vp, Qt.Key.Key_Escape)
    pump(app, 0.05)
    assert st.cam(ml.uid).x == pytest.approx(x0)
    assert st.cam(ml.link_uid).x == pytest.approx(x0)       # the mirrored partner too
    assert st.undo.count() == n_undo and sc.drag is None
    _mouse(vp, 'release', QPointF(*b))
    pump(app, 0.05)
    assert st.cam(ml.uid).x == pytest.approx(x0) and st.undo.count() == n_undo
    settle(app, win)


def test_a_selected_sensor_moves_with_the_handles_and_by_dragging_it(app, win):
    from vantage.ui import sensorview as SV
    vp = vp_of(win)
    st = win.store
    lid = SV.add_sensor(win.ctx, 'lidar')
    win.ctx.set_tool('select')
    settle(app, win)
    sc = vp.provider
    assert sc._gizmo_target()[:2] == ('sensor', lid.uid)
    s0 = st.variant.sensor(lid.uid)
    x0, y0, z0 = s0.x, s0.y, s0.z
    a, b = _arrow_points(sc, vp, s0, k=2)                    # up
    n = st.undo.count()
    _drag(app, vp, QPointF(*a), QPointF(*b))
    s1 = st.variant.sensor(lid.uid)
    assert s1.z > z0 + 0.05 and (s1.x, s1.y) == (pytest.approx(x0), pytest.approx(y0))
    assert st.undo.count() == n + 1                          # one drag, one undo step
    # pressing on the selected sensor itself drags it across, at its height
    p1 = s1.pos()                                            # (the sensor object is edited in place)
    at = _screen(vp, p1)
    _drag(app, vp, QPointF(*at), QPointF(*(at + [70, 25])))
    p2 = st.variant.sensor(lid.uid).pos()
    assert math.hypot(*(p2 - p1)[:2]) > 0.05 and p2[2] == pytest.approx(p1[2])
    assert st.undo.count() == n + 2
    st.undo.undo()
    st.undo.undo()
    s3 = st.variant.sensor(lid.uid)
    assert (s3.x, s3.y, s3.z) == (pytest.approx(x0), pytest.approx(y0), pytest.approx(z0))
    settle(app, win)


def test_a_rig_part_turns_about_z_only(app, win):
    from vantage.ui import rigview as RV
    vp = vp_of(win)
    st = win.store
    part = RV.add_preset(win.ctx, 'enclosure')[0]
    win.ctx.set_tool('rotate')
    settle(app, win)
    sc = vp.provider
    tg = sc._gizmo_target()
    assert tg[:2] == ('rig', part.uid) and sc._rings(tg) == (0,)
    ring = sc._ring(vp, tg[2], 0)
    q, _, ok = vp.project(ring)
    i = int(np.argmax(ok))
    a = q[i]
    b = q[(i + 8) % len(q)]
    assert sc._gizmo_pick(vp, QPointF(*a), tg[2], (0,)) == ('ring', 0)
    yaw0 = part.yaw
    _drag(app, vp, QPointF(*a), QPointF(*b))
    p1 = next(r for r in st.project.rig if r.uid == part.uid)
    assert abs(p1.yaw - yaw0) > 10
    assert (p1.x, p1.y, p1.z) == (pytest.approx(part.x), pytest.approx(part.y), pytest.approx(part.z))
    settle(app, win)
