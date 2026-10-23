"""What the 3D view shows of the rig (ui.overlays3d, ui.scene, ui.sensorview): sensor and camera models at their real
sizes and poses, the simulated lidar sweep, radar volumes, GNSS sky domes, generated brackets, the overlay toggles
and picking."""
import math

import numpy as np
import pytest

from vantage.core import sensors as S
from vantage.core.mathutil import rot_zyx
from vantage.core.project import Obstacle, RigPart, Sensor

from test_gui import app, win, pump, settle, _wait, vp_of, screen_of   # noqa: F401  (fixtures)


@pytest.fixture(scope='module')
def O3(app):
    from vantage.ui import overlays3d
    return overlays3d


def _local(s, T):
    """Vehicle-frame points in the sensor's own frame (origin at its optical centre)."""
    return (np.asarray(T, float).reshape(-1, 3) - s.pos()) @ S.rotation(s)


# ======================================================================= models
def test_housings_have_their_model_size_and_pose(O3):
    rgb = np.array([0.1, 0.6, 0.7])
    for m in S.builtin_sensor_models():
        s = Sensor(kind=m.kind, model_id=m.id, x=1.2, y=-0.4, z=1.9, yaw=33.0, pitch=4.0, roll=-2.5)
        T, N, C = O3.sensor_model(s, m, rgb).arrays()
        P = T.reshape(-1, 3) @ S.rotation(s).T + s.pos()          # posed as the view poses it
        q = _local(s, P)
        dx, dy, h = (v / 1000.0 for v in m.size_mm)
        oz = m.optical_z_mm / 1000.0
        assert np.allclose(np.linalg.norm(N, axis=-1), 1.0, atol=1e-6)
        if m.kind == 'imu':                                         # the box, then the axis arrows beyond it
            big = max(dx, dy, h)
            assert q[:, 0].max() == pytest.approx(dx / 2 + 0.55 * big, abs=1e-6)
            assert q[:, 1].max() == pytest.approx(dy / 2 + 0.55 * big, abs=1e-6)
            assert q[:, 2].max() == pytest.approx(h / 2 + 0.55 * big, abs=1e-6)
            assert q[:, 2].min() == pytest.approx(-oz, abs=1e-6)
            reds = (np.abs(C - O3.AXIS_RGB[0]).sum(1) < 1e-9)
            tip = q.reshape(-1, 3, 3)[reds].reshape(-1, 3)
            assert np.allclose(tip[np.argmax(tip[:, 0])][1:], 0, atol=1e-9)   # the x arrow on the sensor's x axis
            continue
        # every vertex inside the housing core.sensors gives the sensor (its body_box), and reaching its extent
        c, half, R, rnd = S.body_box(s, m)
        b = (P - c) @ R
        lift = 0.0006                                               # the cap's recess ring
        if rnd:
            assert np.hypot(b[:, 0], b[:, 1]).max() == pytest.approx(half[0], abs=1e-6)
        else:
            assert np.abs(b[:, :2]).max(0) == pytest.approx(half[:2], abs=1e-6)
        assert b[:, 2].min() == pytest.approx(-half[2], abs=1e-6) and b[:, 2].max() <= half[2] + lift
        assert q[:, 2].min() == pytest.approx(-oz, abs=1e-6)
        if m.spinning:                                              # the window passes the beams at its radius
            w0, w1 = O3.window_band(m, half[0])
            lo, hi = m.v_fov()
            dark = np.abs(C - O3.WINDOW_RGB).sum(1) < 1e-9
            zw = q.reshape(-1, 3, 3)[dark][..., 2]
            assert zw.min() == pytest.approx(w0) and zw.max() == pytest.approx(w1)
            assert w0 <= half[0] * math.tan(math.radians(lo)) and w1 >= half[0] * math.tan(math.radians(hi))


def test_camera_module_model(O3):
    from vantage.core import samples
    p = samples.tigor_sample_project()
    cam = p.variant.cameras[0]
    cam.yaw, cam.pitch, cam.roll = 25.0, 12.0, -4.0
    mod = p.module_for(cam)
    w, hh, L = (v / 1000.0 for v in mod.size_mm)
    T, N, C = O3.camera_parts(p, cam, True)
    q = (T.reshape(-1, 3) - cam.pos()) @ rot_zyx(cam.yaw, cam.pitch, cam.roll)     # camera body frame
    assert q[:, 0].min() == pytest.approx(-L) and q[:, 0].max() == pytest.approx(0.0, abs=1e-9)
    assert np.abs(q[:, 1]).max() == pytest.approx(w / 2) and np.abs(q[:, 2]).max() == pytest.approx(hh / 2)
    from vantage.ui.theme import THEME
    ring = np.abs(C - np.array(THEME.cam(cam.slot).getRgbF()[:3])).sum(1) < 1e-6
    qr = q.reshape(-1, 3, 3)[ring].reshape(-1, 3)
    assert ring.any() and qr[:, 0].min() >= -0.1 * L and qr[:, 0].max() == pytest.approx(0.0, abs=1e-9)
    # the lens on the optical axis: the ring is centred on it
    assert np.hypot(qr[:, 1], qr[:, 2]).max() == pytest.approx(0.38 * min(w, hh))
    assert abs(qr[:, 1].mean()) < 1e-3 and abs(qr[:, 2].mean()) < 1e-3


def test_radar_volume_lies_inside_its_zone(O3):
    m = next(x for x in S.builtin_sensor_models() if x.id == 'radar_long')
    s = Sensor(kind='radar', model_id=m.id, x=3.3, y=0.2, z=0.5, yaw=10.0, pitch=3.0, roll=1.0)
    vols = O3.radar_volume(s, m, far=15.0)
    assert len(vols) == len(m.zones)
    for (T, N, frac, lines), (rng, ah, eh) in zip(vols, m.zones):
        # outward normals: the far end radially out, the ground footprint down, the faces wound to match
        n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
        assert np.allclose(n / np.linalg.norm(n, axis=1, keepdims=True), N, atol=1e-9)
        ctr = T.mean(axis=1)
        end = np.abs(np.linalg.norm(ctr - s.pos(), axis=1) - min(rng, 15.0)) < 0.02
        assert end.any() and (np.einsum('ij,ij->i', N[end], ctr[end] - s.pos()) > 0).all()
        ground = np.abs(T[..., 2]).max(1) < 1e-9
        assert ground.any() and np.allclose(N[ground], [0, 0, -1], atol=1e-6)
        V = T.reshape(-1, 3)
        V = V[np.linalg.norm(V - s.pos(), axis=1) > 1e-6]                      # not the apex
        r, az, el = S.angles(s, V)
        assert np.abs(az).max() <= ah + 1e-6 and np.abs(el).max() <= eh + 1e-6
        assert r.max() <= min(rng, 15.0) + 1e-6 and V[:, 2].min() >= -1e-9
        assert np.abs(az).max() == pytest.approx(ah) and r.max() == pytest.approx(min(rng, 15.0))
        shells = np.arange(1, 7) * min(rng, 15.0) / 6                         # the shells it fades along
        on_shell = np.abs(r[:, None] - shells[None]).min(1) < 1e-6
        on_ground = np.abs(V[:, 2]) < 1e-9                                      # or cut short by the ground
        assert (on_shell | on_ground).all() and (on_ground & ~on_shell).any()
        assert len(frac) == len(T) and 0 <= frac.min() and frac.max() <= 1
        assert np.all(np.diff(O3.radar_alpha(np.linspace(0, 1, 5), False)) < 0)  # fading with range
        rim = lines[0]
        assert np.allclose(rim[0], rim[-1]) and len(lines) == 5


# ======================================================================= lidar sweeps
def _bare_project():
    from vantage.core import samples
    p = samples.tigor_sample_project()
    p.obstacles = []
    p.variant.cameras = []
    p.variant.sensors = []
    return p


def test_lidar_sweep_count_on_open_ground(O3):
    p = _bare_project()
    m = p.sensor_models['velodyne_vlp16']
    s = Sensor(kind='lidar', model_id=m.id, tag='L1', x=0.0, y=0.0, z=1.8)
    p.variant.sensors = [s]
    sw = O3.lidar_sweep(p, None, s, far=30.0)
    E = m.elevations()
    J = min(m.columns(), max(32, O3.LIDAR_RAYS // len(E)))
    assert sw.channels == 16 and sw.columns == J and sw.rays == 16 * J
    # independent count: every downward beam meeting the ground within its nearest range and 30 m
    t = 1.8 / np.sin(np.radians(-E[E < 0]))
    assert len(sw.points) == int(((t >= m.range_min_m) & (t <= 30.0)).sum()) * J
    assert np.allclose(sw.points[:, 2], O3.GROUND_LIFT, atol=1e-6) and not len(sw.self_points)
    assert np.allclose(np.linalg.norm(sw.points - [0, 0, 1.8 + O3.GROUND_LIFT], axis=1), sw.rng, atol=1e-3)
    assert sw.blocked == 0.0
    # drawn: viridis by range, far fading towards the background
    pd = O3.sweep_draws(sw, dark=True)
    assert len(pd) == 1 and pd[0].points.shape == sw.points.shape and pd[0].colors.shape == (len(sw.points), 4)
    near, far = O3.ramp(np.array([0.0, 1.0]), True)
    assert near.sum() > far.sum()                                    # dark theme: near bright, far dark
    near, far = O3.ramp(np.array([0.0, 1.0]), False)
    assert near.sum() < far.sum()


def test_lidar_sweep_flags_a_pole_and_the_roof(O3):
    from vantage.core import samples, sensor_coverage as SC
    from vantage.core.analysis import Engine
    p = samples.tigor_sample_project()
    p.active_uid = next(v for v in p.variants if v.sensors).uid
    s = next(x for x in p.variant.sensors if x.tag == 'L1')
    p.obstacles = [Obstacle(name='Pole', x=s.x + 4.0, y=s.y + 1.0, height=3.0, radius=0.12)]
    eng = Engine()
    sw = O3.lidar_sweep(p, eng, s)
    pole = sw.points[sw.hit == O3.HIT_OBSTACLE]
    assert len(pole) > 20
    assert np.allclose(np.hypot(pole[:, 0] - (s.x + 4.0), pole[:, 1] - (s.y + 1.0)), 0.12, atol=2e-3)
    # nothing on the ground behind the pole (its shadow), seen from the lidar
    g = sw.points[sw.hit == O3.HIT_GROUND]
    a = np.degrees(np.arctan2(g[:, 1] - s.y, g[:, 0] - s.x) - math.atan2(1.0, 4.0))
    a = (a + 180) % 360 - 180
    d = np.hypot(g[:, 0] - s.x, g[:, 1] - s.y)
    assert not ((np.abs(a) < 0.8) & (d > math.hypot(4, 1) + 0.2)).any()
    # the roof under the lidar: flagged, on the body, the share the coverage study finds
    roof = sw.self_points[sw.self_hit == O3.HIT_VEHICLE]
    lo, hi = p.vehicle.mesh.bbox
    assert len(roof) > 1000 and roof[:, 2].min() > 1.0 and roof[:, 2].max() <= s.z + 1e-3
    assert (roof[:, 0] >= lo[0] - 0.01).all() and (roof[:, 0] <= hi[0] + 0.01).all()
    vg = eng.vehicle_geom(p)
    ref = SC._Lidar(s, p.sensor_model_for(s), eng.cube(s.pos(), vg)).blocked.mean()
    assert sw.blocked == pytest.approx(ref, abs=0.02)
    assert len(O3.sweep_draws(sw, True)) == 2                         # the returns, and the blocked beams in red


def test_lidar_sweep_sees_a_nearby_housing(O3):
    p = _bare_project()
    lid = Sensor(kind='lidar', model_id='ouster_os1_128', tag='L1', x=0.0, y=0.0, z=2.0)
    ant = Sensor(kind='gnss', model_id='gnss_survey', tag='GN1', x=0.4, y=0.0, z=2.03)
    p.variant.sensors = [lid, ant]
    sw = O3.lidar_sweep(p, None, lid)
    hit = sw.self_points[sw.self_hit == O3.HIT_HOUSING]
    c, half, R, _ = S.body_box(ant, p.sensor_model_for(ant))
    assert len(hit) > 10 and np.abs(np.hypot(hit[:, 0] - c[0], hit[:, 1] - c[1])).max() <= half[0] + 0.02


# ======================================================================= GNSS sky
def test_gnss_dome_cells_match_the_sky_view(O3):
    from vantage.core import gnss as G
    from vantage.core.analysis import Engine
    p = _bare_project()
    ant = Sensor(kind='gnss', model_id='gnss_survey', tag='GN1', **S.default_pose(
        'gnss', p.sensor_models['gnss_survey'], p.vehicle.mesh, p.rig, []))
    p.variant.sensors = [ant]
    p.rig = [RigPart(name='Mast', kind='cylinder', x=ant.x + 0.35, y=ant.y, z=ant.z + 0.3, radius=0.06, height=1.0)]
    vg = Engine().vehicle_geom(p)
    sv = G.sky_view(p, vg, ant)
    assert 0.0 < sv.visible < 1.0
    cells = O3.sky_cells(sv)
    blocked = np.asarray(sv.who) >= 0
    # the drawn cells partition the sky cells: their shares add up to what core.gnss blocks
    A = len(sv.az)
    E = len(sv.el_edges) - 1
    n = [(round((c[1] - c[0]) / (360 / A)), int(np.searchsorted(sv.el_edges, c[3] - 1e-9) -
                                                np.searchsorted(sv.el_edges, c[2] + 1e-9) + 1)) for c in cells]
    assert sum(a * e for a, e in n) == A * E
    assert sum(c[4] * a * e for c, (a, e) in zip(cells, n)) == pytest.approx(blocked.sum())
    # a wholly blocked or open cell agrees with the sky view at its centre; nothing straddles the mask
    for a0, a1, e0, e1, share, low in cells:
        who, _ = sv.at((a0 + a1) / 2, (e0 + e1) / 2)
        if share == 1.0:
            assert who is not None
        elif share == 0.0:
            assert who is None
        assert (e1 <= sv.mask_deg + 1e-9) == bool(low)
    assert any(c[4] == 1.0 and not c[5] for c in cells)               # the mast, above the mask
    T, rgba, lines = O3.sky_dome(sv, True)
    V = T.reshape(-1, 3)
    assert np.allclose(np.linalg.norm(V - sv.pos, axis=1), O3.SKY_R, atol=1e-9) and V[:, 2].min() >= sv.pos[2] - 1e-9
    assert rgba[:, 3].max() > 0.6 and rgba[:, 3].min() < 0.2         # blocked cells stand out
    assert len(lines) == 2                                            # the horizon and the mask


# ======================================================================= in the app
def _ov3d(win):
    from vantage.ui import overlays3d as O3
    return O3.controller(win.ctx)


def _draws_with(tris, n):
    return [t for t in tris if len(t.tri) == n]


def test_sensors_and_cameras_drawn_outlined_and_picked(app, win):
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    from vantage.ui import sensorview as SV, overlays3d as O3
    st = win.store
    lid = SV.add_sensor(win.ctx, 'lidar')
    rad = SV.add_sensor(win.ctx, 'radar')
    imu = SV.add_sensor(win.ctx, 'imu')
    st.select_obj(())
    settle(app, win)
    vp = vp_of(win)
    sc = vp.provider
    n = {x.uid: len(O3.sensor_parts(st.project, x, SV._rgb(x.kind), False)[0]) for x in (lid, rad, imu)}
    tris = sc.tris_extra(vp)
    assert _draws_with(tris, sum(n.values()))                        # every housing, one draw
    cams = [c for c in st.variant.cameras if c.visible]
    ncam = sum(len(O3.camera_parts(st.project, c, False)[0]) for c in cams)
    st.select_cams([])
    pump(app, 0.1)
    assert _draws_with(sc.tris_extra(vp), ncam)                       # every camera module
    # selecting a sensor gives it its own draw with the selection outline
    st.select_obj(('sensor', lid.uid))
    pump(app, 0.1)
    tris = sc.tris_extra(vp)
    own = _draws_with(tris, n[lid.uid])
    assert own and _draws_with(tris, n[rad.uid] + n[imu.uid])
    if O3.has_outline():
        assert tuple(own[0].outline) == pytest.approx(O3.sel_outline())
    # and a camera
    f = st.variant.by_tag('F')
    st.select_cams([f.uid])
    pump(app, 0.1)
    one = len(O3.camera_parts(st.project, f, False)[0])
    sel = _draws_with(sc.tris_extra(vp), one)
    assert sel and (not O3.has_outline() or getattr(sel[-1], 'outline', None) is not None)
    # the radar's field as a translucent volume (after the opaque draws), its rim as a line
    tris = sc.tris_extra(vp)
    brown = SV._rgb('radar')
    vol = [i for i, t in enumerate(tris) if len(t.colors) and t.colors[:, 3].max() < 0.5
           and np.allclose(t.colors[0, :3], brown, atol=1e-6)]
    opaque = [i for i, t in enumerate(tris) if len(t.colors) and t.colors[:, 3].min() > 0.99]
    assert vol and opaque and max(opaque) < min(vol)                  # translucent after every opaque draw
    assert any(ld.color.name() == SV.color('radar').name() for ld in sc.lines(vp))
    # picking: a click on the lidar housing selects it, a click on a camera selects the camera
    st.select_cams([])
    vp.cam.az, vp.cam.el = -30.0, 35.0
    vp.frame_box(lid.pos() - 1.2, lid.pos() + 1.2)
    pump(app, 0.1)
    c = S.body_box(lid, st.project.sensor_model_for(lid))[0]
    QTest.mouseClick(vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, screen_of(vp, c))
    pump(app, 0.1)
    assert st.sel_obj == ('sensor', lid.uid)
    win.viewarea.frame_all()
    pump(app, 0.1)
    QTest.mouseClick(vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, screen_of(vp, f.pos()))
    pump(app, 0.1)
    assert st.selected is not None and st.selected.uid == f.uid
    vp.grab()                                                         # the software renderer draws it all


def test_mesh_materials(app, win):
    from vantage.ui import viewport3d as V3
    from vantage.core.mesh import ROLE_GLASS, ROLE_WHEELS
    sc = vp_of(win).provider
    md = sc.mesh()
    if 'materials' not in V3.MeshDraw.__dataclass_fields__:
        pytest.skip('renderer without materials')
    m = sc.project.vehicle.mesh
    assert md.materials is not None and len(md.materials) == len(md.tri)
    from vantage.ui.scene import mesh_materials_for
    mats = mesh_materials_for(m)
    roles = np.array([pt.role for pt in m.parts])[m.face_part]
    assert (mats[roles == ROLE_GLASS] == V3.MAT_GLASS).all() and (mats[roles == ROLE_WHEELS] == V3.MAT_TYRE).all()
    glass_zone = [i for i, z in enumerate(m.zones) if 'glass' in z.name.lower()]
    assert (mats[np.isin(m.face_zone, glass_zone)] == V3.MAT_GLASS).all()


def test_overlays_toggle_through_the_menu(app, win):
    from vantage.ui import sensorview as SV
    from vantage.ui.context import OVERLAYS
    keys = {k for k, *_ in OVERLAYS}
    assert {'lidarpts', 'gnsssky', 'brackets', 'sensors'} <= keys
    menus = {a.text().replace('&', ''): a.menu() for a in win.menuBar().actions()}
    ov = next(a.menu() for a in menus['View'].actions() if a.menu() and a.text().replace('&', '') == 'Overlays')
    texts = [a.text() for a in ov.actions()]
    assert 'Lidar points' in texts and 'GNSS sky' in texts and 'Mount brackets' in texts
    lid = SV.add_sensor(win.ctx, 'lidar')
    SV.add_sensor(win.ctx, 'gnss')
    settle(app, win)
    vp = vp_of(win)
    sc = vp.provider
    act = win.actions_by_id['ov_lidarpts']
    if act.isChecked():
        act.trigger()
    assert not win.ctx.overlays['lidarpts'] and sc.points(vp) == []
    act.trigger()
    assert win.ctx.overlays['lidarpts']
    ctl = _ov3d(win)
    vp.grab()
    _wait(app, lambda: not ctl.busy() and sc.points(vp), timeout=60)
    pts = sc.points(vp)
    assert sum(len(q.points) for q in pts) == sum(len(sw.points) + len(sw.self_points) for sw in ctl.fresh_sweeps())
    # the sky domes: translucent draws over each antenna, labelled with the open share
    act = win.actions_by_id['ov_gnsssky']
    if not act.isChecked():
        act.trigger()
    vp.grab()
    _wait(app, lambda: not ctl.busy() and ctl.fresh_skies(), timeout=60)
    domes, rings = ctl.dome_draws(False)
    assert domes and all(any(t is d for t in sc.tris_extra(vp)) for d in domes)
    assert rings and any(any(l is r for l in sc.lines(vp)) for r in rings)
    vp.grab()
    act.trigger()
    assert not any(any(t is d for t in sc.tris_extra(vp)) for d in domes)
    win.actions_by_id['ov_lidarpts'].trigger()
    assert sc.points(vp) == [] and win.ctx.overlays['lidarpts'] is False
    assert lid.uid in ctl.sweeps


def test_lidar_points_follow_a_moved_lidar(app, win):
    from vantage.ui import sensorview as SV
    lid = SV.add_sensor(win.ctx, 'lidar')
    win.ctx.set_overlay('lidarpts', True)
    try:
        settle(app, win)
        vp = vp_of(win)
        sc = vp.provider
        ctl = _ov3d(win)
        sc.points(vp)
        _wait(app, lambda: not ctl.busy() and sc.points(vp), timeout=60)
        sw0 = ctl.sweeps[lid.uid]
        assert np.allclose(sw0.pos, lid.pos())
        # an edit that does not touch the lidar's inputs keeps its sweep
        cam = win.store.variant.cameras[-1]
        win.store.edit('variant', lambda p: setattr(p.variant.cam(cam.uid), 'name', 'renamed'), 'Rename')
        sc.points(vp)
        assert not ctl.busy() and ctl.sweeps[lid.uid] is sw0
        # moving it: the old points go at once, new ones from the new place come after the debounce
        SV.set_props(win.ctx, lid.uid, 'Raised L1', z=round(lid.z + 0.5, 3))
        new = win.store.variant.sensor(lid.uid)
        assert sc.points(vp) == [] and ctl.busy()
        _wait(app, lambda: not ctl.busy() and sc.points(vp), timeout=60)
        sw1 = ctl.sweeps[lid.uid]
        assert sw1 is not sw0 and np.allclose(sw1.pos, new.pos()) and sw1.blocked < sw0.blocked
        # undo: back to a sweep from the old place
        win.store.undo.undo()
        sc.points(vp)
        _wait(app, lambda: not ctl.busy() and sc.points(vp), timeout=60)
        assert np.allclose(ctl.sweeps[lid.uid].pos, sw0.pos)
    finally:
        win.ctx.set_overlay('lidarpts', False)


def test_brackets_drawn_after_generation(app, win):
    from vantage.ui.workspaces import brackets_ws as BW
    from vantage.ui import overlays3d as O3
    st = win.store
    ctl = BW.controller(win.ctx)
    ctl.set('resolution', res_mm=1.0)
    f = st.variant.by_tag('F')
    ctl.refresh_status()
    _wait(app, lambda: ctl.bracket(f.uid) is not None)
    assert ctl.move_out([f.uid])
    vp = vp_of(win)
    sc = vp.provider
    assert O3.bracket_draws(win.ctx, False) == []
    ctl.generate(quiet=True)
    _wait(app, lambda: not ctl.busy and ctl.result is not None, timeout=300)
    b = ctl.result.by_uid(f.uid)
    assert b.generated
    st.select_cams([])
    pump(app, 0.1)
    draws = O3.bracket_draws(win.ctx, False)
    nf = len((b.display_fast or b.display)[1])
    assert len(draws) == 1 and len(draws[0].tri) == sum(len(O3.bracket_parts(x, 0)[0]) for x in ctl.result.brackets
                                                         if x.generated)
    assert any(len(t.tri) == len(draws[0].tri) for t in sc.tris_extra(vp)) and nf > 100
    # on the vehicle where the plan puts it: the bracket's base touches its contact point
    V = O3.bracket_parts(b, f.slot)[0].reshape(-1, 3)
    assert np.linalg.norm(V - np.asarray(b.contact), axis=1).min() < 0.01
    # in the camera's colour (lit), and gone when the overlay is off or the camera moves away from it
    from vantage.ui.theme import THEME
    rgb = np.array(THEME.cam(f.slot).getRgbF()[:3]) * 0.9
    fb = O3.draw_parts([O3.bracket_parts(b, f.slot)], True)
    assert np.allclose(fb.colors[:, :3], rgb, atol=1e-6)
    win.ctx.set_overlay('brackets', False)
    assert not any(len(t.tri) == len(draws[0].tri) for t in sc.tris_extra(vp))
    win.ctx.set_overlay('brackets', True)
    st.set_cam(f.uid, x=round(f.x + 0.05, 4))
    assert not any(O3._bracket_fits(x, st.variant.cam(x.uid)) for x in ctl.result.brackets if x.uid == f.uid)
    assert all(len(t.tri) != nf for t in O3.bracket_draws(win.ctx, False))
    vp.grab()


def test_volumes_face_the_view_and_bad_poses_draw_nothing(app, win):
    from vantage.ui import sensorview as SV, overlays3d as O3
    st = win.store
    rad = SV.add_sensor(win.ctx, 'radar')
    lid = SV.add_sensor(win.ctx, 'lidar')
    st.select_obj(())
    settle(app, win)
    vp = vp_of(win)
    sc = vp.provider
    vp.cam.az, vp.cam.el = -40.0, 50.0
    vp.update()
    vols = SV._volumes(rad, st.project.sensor_model_for(rad), SV.RING_FAR, False)     # its zones, as drawn
    T = np.concatenate([v[0] for v in vols])
    N = np.concatenate([v[1] for v in vols])
    [td] = SV.radar_draws(win.ctx, False, vp=vp)
    assert len(td.tri) < len(T)                                       # the faces turned away are left out
    k = O3.facing(T, N, vp)
    assert len(td.tri) == k.sum() and not (np.abs(T[k][..., 2]).max(1) < 1e-9).any()   # no footprint from above
    # a hand-edited NaN pose: nothing of it is drawn and nothing breaks
    SV.set_props(win.ctx, lid.uid, 'NaN', x=float('nan'))
    win.ctx.set_overlay('lidarpts', True)
    try:
        for t in sc.tris_extra(vp):
            assert np.isfinite(t.tri).all()
        for ld in sc.lines(vp):
            assert np.isfinite(np.asarray(ld.pts, float)).all()
        assert sc.points(vp) == []
        vp.grab()
    finally:
        win.ctx.set_overlay('lidarpts', False)
        st.undo.undo()
