"""Mount brackets: the mesher (closed, consistently wound, manifold on random fields; volumes; the narrow band), binary
STL, and the brackets themselves measured back from their meshes independently of the generator: the plate's normal
against the camera's axis, the optical centre against the plan, hole diameters from a cross-section, the tube clamp's
bore; image blocking, a camera too far out, inside a surface or needing more standoff, cancel, exports, the page."""
import copy
import json
import math
import os

import numpy as np
import pytest

from vantage.core import brackets as BR, meshing as M, rig, samples
from vantage.core.analysis import Engine
from vantage.core.lens import resolve
from vantage.core.mathutil import BODY_TO_OPTICAL, rot_zyx
from vantage.core.project import Camera

ANGLES = (0, 30, 60, 89)            # optical axis from the mount surface's normal, degrees (back face)


# =========================================================================== fixtures
@pytest.fixture(scope='module')
def base():
    """The sample car with a sensor enclosure on the roof (a flat box top to mount on) and a crossbar behind it."""
    p = samples.tigor_sample_project()
    encl = rig.preset('enclosure', p.vehicle.mesh)[0]
    bar = rig.preset('bar', p.vehicle.mesh)[0]
    bar.x = encl.x + 0.6
    p.rig = [encl, bar]
    return p


def top_of(p):
    e = p.rig[0]
    return e.x, e.z + e.height / 2


def cam(tag, x, y, z, yaw, pitch, roll=0.0, **kw):
    return Camera(tag=tag, name=tag, x=x, y=y, z=z, yaw=yaw, pitch=pitch, roll=roll, standoff_mm=60, **kw)


@pytest.fixture(scope='module')
def built(base):
    """Back-face brackets with the optical axis 0/30/60/89° from the enclosure's normal, a bottom-face one, and a tube
    clamp on the crossbar; meshed at 0.35 mm."""
    p = copy.deepcopy(base)
    x, top = top_of(p)
    v = p.variant
    v.cameras = [cam(f'A{a}', x - 0.2 + 0.13 * i, 0.0, top + 0.060, 0.0, -(90 - a)) for i, a in enumerate(ANGLES)]
    v.cameras.append(cam('BOT', x, 0.15, top + 0.06, 30.0, 25.0))
    bar = p.rig[1]
    v.cameras.append(cam('CL', bar.x, 0.3, bar.z + bar.radius + 0.05, 0.0, 20.0))
    st = BR.BracketSettings.from_dict({'face': 'back', 'connector_mm': 0, 'res_mm': 0.35})
    eng = Engine()
    res = BR.run(p, eng, st, cams=[f'A{a}' for a in ANGLES])
    st2 = BR.BracketSettings.from_dict({'res_mm': 0.35})
    res2 = BR.run(p, eng, st2, cams=['BOT', 'CL'])
    return p, eng, res, res2


# =========================================================================== settings
def test_settings_are_validated():
    st = BR.BracketSettings.from_dict({'wall_mm': 0.5, 'res_mm': 'x', 'margin_mm': float('nan'), 'face': 'side',
                                       'pattern': 'nope', 'material': 'PA-CF', 'fixing': 'vhb', 'countersink': 'no',
                                       'custom_count': 3, 'reach_mm': 1e9, 'clearance_mm': -1, 'junk': 1})
    assert st.wall_mm == 2.0 and st.res_mm == 0.35 and st.margin_mm == 3.0 and st.face == 'auto'
    assert st.pattern == '2xM2-16' and st.material == 'PA-CF' and st.fixing == 'vhb' and st.countersink is False
    assert st.custom_count == 4 and st.reach_mm == 500.0 and st.clearance_mm == 0.0
    assert BR.BracketSettings.from_dict(st.to_dict()) == st
    assert BR.BracketSettings.from_dict(None) == BR.BracketSettings()
    assert BR.BracketSettings.from_dict({'pattern': 'custom', 'custom_count': 1, 'custom_d_mm': 5}).holes()[3] == 1


# =========================================================================== the mesher
def _vertex_manifold(F):
    link = {}
    for a, b, c in F:
        for v, e in ((a, (b, c)), (b, (c, a)), (c, (a, b))):
            link.setdefault(v, []).append(e)
    for v, es in link.items():
        nxt = dict(es)
        if len(nxt) != len(es):
            return False
        x, n = es[0][0], 0
        while True:
            x = nxt.get(x)
            n += 1
            if x is None:
                return False
            if x == es[0][0]:
                break
        if n != len(es):
            return False
    return True


def test_mesher_is_closed_and_manifold_on_random_fields():
    rng = np.random.default_rng(3)
    seen = 0
    for _ in range(600):
        n = int(rng.integers(2, 7))
        vals = (rng.random((n + 2,) * 3) - rng.random()).astype(np.float32)
        for ax in range(3):
            idx = [slice(None)] * 3
            for end in (0, -1):
                idx[ax] = end
                vals[tuple(idx)] = 1.0
        V, F = M.march(vals, np.zeros(3), 1.0)
        if not len(F):
            continue
        seen += 1
        c = M.check(V, F)
        assert c['watertight'] and c['consistent'] and c['degenerate'] == 0 and c['volume_mm3'] > 0, c
        assert _vertex_manifold(F.tolist())
    assert seen > 400


def test_mesher_volume_band_and_cancel():
    box = M.box([0, 0, 0], np.eye(3), [10, 8, 3])
    s = M.subtract(box, M.rod([0, 0, -10], [0, 0, 10], 2.0))
    V, F = M.mesh(s, 0.3)
    c = M.check(V, F)
    exact = 20 * 16 * 6 - math.pi * 4 * 6
    assert c['ok'] and c['shells'] == 1 and c['genus'] == 1
    assert c['volume_mm3'] == pytest.approx(exact, rel=0.01)
    lo, hi = s.lo - 1, s.hi + 1
    full, o = M.grid_eval(s, lo, hi, 0.3, coarse=1)
    band, _ = M.grid_eval(s, lo, hi, 0.3, coarse=4)
    Vf, Ff = M.march(full, o, 0.3)
    Vb, Fb = M.march(band, o, 0.3)
    assert np.array_equal(Ff, Fb) and np.allclose(Vf, Vb)            # sampling only near the surface changes nothing
    assert M.mesh(s, 0.3, cancel=lambda: True) == (None, None)


def test_raytrace_and_section():
    b = M.box([0, 0, 0], np.eye(3), [5, 5, 5])
    hit, t = M.raytrace(b, [[-20, 0, 0], [-20, 9, 0]], [[1, 0, 0], [1, 0, 0]], 60)
    assert hit.tolist() == [True, False] and t[0] == pytest.approx(15, abs=0.02)
    A, _, Imin, _ = M.section(b, [0, 0, 0], np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), (7, 7), h=0.1)
    assert A == pytest.approx(100, rel=0.02) and Imin == pytest.approx(10 ** 4 / 12, rel=0.03)


def test_stl_binary_round_trip():
    s = M.subtract(M.box([0, 0, 0], np.eye(3), [6, 4, 3], r=1), M.rod([0, 0, -9], [0, 0, 9], 1.5))
    V, F = M.mesh(s, 0.4)
    data = M.stl_bytes(V, F, 'solid test part')
    assert len(data) == 84 + 50 * len(F)
    assert not data[:80].lower().startswith(b'solid')                 # not mistaken for an ASCII STL
    assert int.from_bytes(data[80:84], 'little') == len(F)
    V2, F2, N, head = M.read_stl(data)
    assert 'test part' in head and len(F2) == len(F)
    assert np.allclose(np.linalg.norm(N, axis=1), 1, atol=1e-5)
    tri = V2[F2]
    cr = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    assert np.all(np.einsum('ij,ij->i', cr, N) > 0)                  # normals agree with the winding
    c = M.check(V2, F2)
    assert c['ok'] and c['volume_mm3'] == pytest.approx(M.check(V, F)['volume_mm3'], rel=1e-5)


# =========================================================================== brackets, measured back
def _section(V, F, p0, n):
    """Segments where the plane through p0 with normal n cuts the mesh."""
    d = (V - p0) @ n
    tri, dt = V[F], d[F]
    cut = (dt.min(axis=1) < 0) & (dt.max(axis=1) > 0)
    segs = []
    for t, dd in zip(tri[cut], dt[cut]):
        pts = []
        for i in range(3):
            j = (i + 1) % 3
            if (dd[i] < 0) != (dd[j] < 0):
                pts.append(t[i] + (t[j] - t[i]) * dd[i] / (dd[i] - dd[j]))
        segs.append(pts)
    return np.array(segs)


def _circle(P, e1, e2, c0):
    x, y = (P - c0) @ e1, (P - c0) @ e2
    A = np.column_stack([x, y, np.ones_like(x)])
    sol, *_ = np.linalg.lstsq(A, x * x + y * y, rcond=None)
    cx, cy = sol[0] / 2, sol[1] / 2
    return c0 + cx * e1 + cy * e2, 2 * math.sqrt(sol[2] + cx * cx + cy * cy)


def _measure(p, b, face):
    """From the plan (the camera's pose, the module's size, the hole pattern) and the mesh in the vehicle frame (mm):
    the contact plane's normal, the optical centre rebuilt from the holes' centres, the holes' diameters."""
    c = p.variant.by_tag(b.tag)
    Rc = rot_zyx(c.yaw, c.pitch, c.roll)
    a, ey, ez = Rc[:, 0], Rc[:, 1], Rc[:, 2]
    _, h, L = p.module_for(c).size_mm
    st = BR.BracketSettings.from_dict({'face': face})
    _, d, head, count, pa, _ = st.holes()
    P = c.pos() * 1000.0
    if face == 'back':
        nrm, Fc, e1, e2, depth = a, P - a * L, ey, ez, lambda C, n: C + n * L
    else:
        nrm, Fc, e1, e2 = ez, P - a * L / 2 - ez * h / 2, a, ey
        depth = lambda C, n: C + n * h / 2 + a * L / 2                           # noqa: E731
    offs = {1: [(0, 0)], 2: [(-pa / 2, 0), (pa / 2, 0)]}[count]
    V, F = BR.vehicle_mesh(b)
    V = V * 1000.0
    tri = V[F]
    cen = tri.mean(axis=1)
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    fn /= np.linalg.norm(fn, axis=1, keepdims=True)
    rel = cen - Fc
    s1, s2, sn = rel @ e1, rel @ e2, rel @ nrm
    far = np.ones(len(F), bool)
    for o1, o2 in offs:
        far &= np.hypot(s1 - o1, s2 - o2) > head / 2 + 1.0
    face_sel = (fn @ nrm > 0.99) & (np.abs(sn) < 1.0) & (np.abs(s1) < 9) & (np.abs(s2) < 9) & far
    pts = tri[face_sel].reshape(-1, 3)
    c0 = pts.mean(axis=0)
    n_meas = np.linalg.svd(pts - c0, full_matrices=False)[2][2]
    n_meas = n_meas if n_meas @ nrm > 0 else -n_meas
    # holes: the mesh cut half way through the plate, circles fitted to the cut around each hole
    t = 3.0
    segs = _section(V, F, Fc - nrm * t / 2, nrm).reshape(-1, 3)
    centres, diam = [], []
    for o1, o2 in offs:
        hc = Fc - nrm * t / 2 + e1 * o1 + e2 * o2
        near = segs[np.linalg.norm(segs - hc, axis=1) < (d + 0.4) / 2 + 0.6]
        cc, dd = _circle(near, e1, e2, hc)
        centres.append(cc)
        diam.append(dd)
    C = np.mean(centres, axis=0) + nrm * t / 2                       # back onto the contact face
    C = C - n_meas * ((C - c0) @ n_meas)
    return n_meas, depth(C, n_meas) / 1000.0, diam, nrm


@pytest.mark.parametrize('angle', ANGLES)
def test_back_face_brackets_hold_the_planned_pose(built, angle):
    p, _, res, _ = built
    b = res.by_tag(f'A{angle}')
    assert b.generated and b.face == 'back' and b.kind == 'flat'
    chk = M.check(b.V.astype(float), b.F.astype(np.int64))
    assert chk['watertight'] and chk['consistent'] and chk['degenerate'] == 0 and chk['volume_mm3'] > 0
    assert chk['shells'] == 1 and b.mesh['ok']
    n_meas, oc, diam, axis = _measure(p, b, 'back')
    c = p.variant.by_tag(b.tag)
    ang = math.degrees(math.acos(np.clip(n_meas @ axis, -1, 1)))
    assert ang < 0.5                                                  # the plate is square to the optical axis
    assert np.linalg.norm(oc - c.pos()) * 1000 < 0.5                 # the optical centre lands on the plan
    for dmeas in diam:                                                # M2 + 0.4 mm clearance, within the grid
        assert abs(dmeas - 2.4) < 0.35
    assert b.params['plate_angle_deg'] == pytest.approx(angle, abs=0.6)    # flat enclosure top: plate vs base
    assert b.params['fov_blocked_pct'] == 0.0 and b.check('fov')['severity'] == 'good'
    assert b.check('placement')['severity'] == 'good'
    # the bracket frame: z the surface normal (the enclosure's top), the STL's base on z = 0
    assert b.R[:, 2] @ np.array([0, 0, 1.0]) > 0.999
    assert b.V[:, 2].min() == pytest.approx(0.0, abs=0.36)
    Vv, _ = BR.vehicle_mesh(b)
    assert Vv[:, 2].min() == pytest.approx(top_of(p)[1], abs=0.0005)   # the base sits on the enclosure


def test_bottom_face_and_tube_clamp(built):
    p, _, _, res = built
    b = res.by_tag('BOT')
    assert b.generated and b.face == 'bottom'                         # auto: the bottom face keeps the connector free
    n_meas, oc, _, ez = _measure(p, b, 'bottom')
    assert math.degrees(math.acos(np.clip(n_meas @ ez, -1, 1))) < 0.5
    assert np.linalg.norm(oc - p.variant.by_tag('BOT').pos()) * 1000 < 0.5
    cl = res.by_tag('CL')
    assert cl.generated and cl.kind == 'clamp' and cl.surface == 'Crossbar'
    assert cl.mesh['ok'] and cl.mesh['shells'] == 2                  # saddle and cap
    bar = p.rig[1]
    # the bore, measured on the mesh: vertices within 2 mm of the tube's surface lie at radius + clearance
    V, _ = BR.vehicle_mesh(cl)
    yaw = math.radians(bar.yaw)
    axis = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
    rel = V - np.array([bar.x, bar.y, bar.z])
    radial = np.linalg.norm(rel - np.outer(rel @ axis, axis), axis=1) * 1000
    bore = radial[np.abs(radial - bar.radius * 1000) < 2.0]
    assert len(bore) > 50 and np.median(bore) == pytest.approx(bar.radius * 1000 + 0.2, abs=0.3)
    assert cl.params['clamp']['tube_radius_mm'] == pytest.approx(bar.radius * 1000)
    assert 'nuts' in ' '.join(cl.notes)


def test_fov_intrusion_is_detected(built):
    p, _, res, _ = built
    b = res.by_tag('A30')
    c = p.variant.by_tag('A30')
    intr = resolve(p.module_for(c), p.lens_for(c))
    R_lo = b.R.T @ rot_zyx(c.yaw, c.pitch, c.roll) @ BODY_TO_OPTICAL.T
    share, mask = BR.fov_blocked(b.solid, b.oc, R_lo, intr)
    assert share == 0.0 and (mask == 2).sum() == 0
    # the same bracket put 30 mm in front of the lens fills the middle of the image
    ahead = M.translated(b.solid, b.oc + b.axis * 30 - (b.solid.lo + b.solid.hi) / 2)
    share, mask = BR.fov_blocked(ahead, b.oc, R_lo, intr)
    assert share > 0.2
    hh, ww = mask.shape
    assert mask[hh // 2, ww // 2] == 2                               # the image centre looks at it
    # and a camera looking down past its clamp sees it: a warning with the share
    cl = built[3].by_tag('CL')
    assert cl.check('fov')['severity'] in ('warning', 'serious') and cl.params['fov_blocked_pct'] > 0


def test_cameras_without_a_printed_bracket(base):
    p = copy.deepcopy(base)
    x, top = top_of(p)
    v = p.variant
    v.cameras = [cam('FAR', x, 0.0, top + 0.4, 0, 30), cam('IN', x, 0.0, top - 0.05, 0, 30),
                 cam('NONE', x, 0.1, top + 0.06, 0, -60, bracket='None'),
                 cam('NAN', x, 0.1, float('nan'), 0, 30), cam('UP', x + 0.1, 0.0, top + 0.06, 0, 60),
                 cam('HID', x, 0.0, top + 0.06, 0, -60, visible=False)]
    res = BR.run(p, Engine(), {'face': 'back', 'res_mm': 1.0})
    by = {b.tag: b for b in res.brackets}
    assert 'HID' not in by                                            # hidden cameras get none
    assert by['FAR'].status == 'floating' and not by['FAR'].generated and '400 mm' in by['FAR'].message
    assert by['IN'].status == 'inside' and not by['IN'].generated
    assert by['NONE'].status == 'skipped' and 'None' in by['NONE'].message
    assert by['NAN'].status == 'skipped'
    # looking down at 60° the back face turns away from the enclosure: say which face to use
    assert by['UP'].status == 'face' and 'bottom' in by['UP'].message
    assert BR.files(res, p)[0][0] == 'brackets/brackets.json'        # no STL; the JSON and sheet still list them


def test_standoff_needed_then_moved_out(base):
    p = copy.deepcopy(base)
    p.rig = []
    v = p.variant
    eng = Engine()
    pre = {b.tag: b for b in BR.preflight(p, eng)}
    f = pre['F']                                                      # 16 mm off the grille: the housing goes in
    assert f.status == 'standoff' and f.required_mm > f.gap_mm + 10 and not f.generated
    row = next(r for r in BR.summary(BR.plan(p, eng))['per_camera'] if r['tag'] == 'F')
    assert row['required_standoff_mm'] == f.required_mm and 'base_mm' not in row      # no figures of a misfit
    c = v.by_tag('F')
    c.x, c.y, c.z = (f.contact[i] + f.normal[i] * f.required_mm / 1000 for i in range(3))
    b = BR.run(p, eng, {'res_mm': 0.6}, cams=['F']).brackets[0]
    assert b.generated and b.status in ('ok', 'warning') and b.gap_mm == pytest.approx(f.required_mm, abs=0.5)
    ready = next(x for x in BR.preflight(p, eng, BR.BracketSettings.from_dict({'res_mm': 0.6})) if x.tag == 'F')
    assert ready.status == 'ok' and ready.params['base']['rect'] == pytest.approx(b.params['base']['rect'])  # as made
    assert b.check('placement')['severity'] == 'good'
    assert np.allclose(b.origin + b.R @ (b.oc / 1000.0), c.pos(), atol=1e-9)
    base_gap = b.check('base_gap')
    assert base_gap is not None and base_gap['value'] >= 0             # the grille is curved: gaps are reported
    p.variant.cameras = []
    assert BR.run(p, eng).brackets == [] and BR.preflight(p, eng) == []


def test_cancel_and_progress(base):
    p = copy.deepcopy(base)
    x, top = top_of(p)
    p.variant.cameras = [cam('A', x, 0, top + 0.06, 0, -60), cam('B', x + 0.1, 0, top + 0.06, 0, -60)]
    seen = []

    def progress(f, text=''):
        seen.append(f)
    assert BR.run(p, Engine(), {'res_mm': 1.0}, progress=progress, cancel=lambda: len(seen) > 3) is None
    seen.clear()
    res = BR.run(p, Engine(), {'res_mm': 1.0}, progress=progress)
    assert res is not None and seen[-1] == 1.0 and all(b.generated for b in res.brackets)


def test_exports(built, tmp_path):
    p, _, res, _ = built
    out = BR.files(res, p)
    names = [n for n, _ in out]
    assert names[:4] == [f'brackets/A{a}.stl' for a in ANGLES]
    assert names[-2:] == ['brackets/brackets.json', 'brackets/brackets_sheet.png']
    doc = json.loads(dict(out)['brackets/brackets.json'])
    assert doc['schema'] == BR.SCHEMA and 'ISO 8855' in doc['vehicle_frame'] and doc['assumptions']
    d0 = doc['brackets'][0]
    assert d0['tag'] == 'A0' and d0['stl'] == 'brackets/A0.stl' and len(d0['T_vehicle_bracket']) == 4
    T = np.array(d0['T_vehicle_bracket'])
    oc = T[:3, :3] @ (np.array(d0['optical_centre_mm']) / 1000) + T[:3, 3]
    assert np.allclose(oc, p.variant.by_tag('A0').pos(), atol=1e-4)
    assert d0['parameters']['base']['screws']['size'] == 'M4' and d0['mesh']['watertight'] is True
    assert {c['code'] for c in d0['checks']} >= {'mesh', 'placement', 'fov', 'overhang', 'wall'}
    import cv2
    img = cv2.imdecode(np.frombuffer(dict(out)['brackets/brackets_sheet.png'], np.uint8), cv2.IMREAD_COLOR)
    assert img is not None and img.shape[1] > 1000 and img.shape[0] > 4 * 270
    V, F, _, head = M.read_stl(dict(out)['brackets/A30.stl'])
    assert 'A30' in head and M.check(V, F)['ok']
    paths = BR.export(res, p, str(tmp_path), tags=['A60'])
    assert [os.path.relpath(x, tmp_path) for x in paths] == [os.path.join('brackets', 'A60.stl')]
    s = BR.summary(res)
    assert s['generated'] == 4 and s['per_camera'][1]['plate_angle_deg'] == pytest.approx(30, abs=0.6)
    json.dumps(s)


# =========================================================================== the page
from test_gui import app, win, pump, _wait   # noqa: E402,F401  (fixtures)


def _open(win, name, cls):
    if name not in win._page_widgets:
        pg = cls(win.ctx)
        win.pages.addWidget(pg)
        win._page_widgets[name] = pg
    win.set_page(name)
    return win._page_widgets[name]


def _type(app, f, text):
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    QTest.mouseClick(f.lineEdit(), Qt.MouseButton.LeftButton)
    QTest.keyClick(f.lineEdit(), Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
    QTest.keyClicks(f.lineEdit(), text)
    QTest.keyClick(f.lineEdit(), Qt.Key.Key_Return)
    pump(app, 0.1)


def test_brackets_page(app, win, tmp_path):
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    from vantage.ui.workspaces.brackets_ws import BracketsPage
    store = win.store
    pg = _open(win, 'brackets', BracketsPage)
    assert win.page_name() == 'brackets' and pg.isVisible()
    ctl = pg.ctl
    _wait(app, lambda: len(ctl.pre) == len(store.variant.cameras))
    assert pg.table.rowCount() == len(store.variant.cameras)
    words = [pg.table.item(r, 2).text() for r in range(pg.table.rowCount())]
    assert 'Needs standoff' in words                                  # the sample's cameras sit close to the body
    # a coarser grid for speed, typed into its field: saved with the project
    _type(app, pg.fields['res_mm'], '1.0')
    assert store.project.analysis.brackets['res_mm'] == 1.0
    # select F, move it out to the standoff a printed bracket needs: one undo step
    f = store.variant.by_tag('F')
    row = next(r for r in range(pg.table.rowCount()) if pg.table.item(r, 0).text() == 'F')
    pg.table.selectRow(row)
    pump(app, 0.1)
    assert pg.sel_uid == f.uid and pg.b_move.isVisible()
    need = ctl.movable(f.uid).required_mm
    QTest.mouseClick(pg.b_move, Qt.MouseButton.LeftButton)
    pump(app, 0.2)
    assert store.variant.by_tag('F').standoff_mm == need
    _wait(app, lambda: ctl.bracket(f.uid) is not None and ctl.bracket(f.uid).status == 'ok')
    # generate (a main-window job) and see F's bracket in the preview
    QTest.mouseClick(pg.b_gen, Qt.MouseButton.LeftButton)
    _wait(app, lambda: not ctl.busy and ctl.result is not None)
    b = ctl.bracket(f.uid)
    assert b.generated and pg.view.mesh is not None and pg.b_stl.isEnabled() and not ctl.stale
    assert pg.fig_labels['base'].text().endswith('2 screws') and pg.check_lines
    # the wall thicker: undoable, and the result goes stale until generated again
    _type(app, pg.fields['wall_mm'], '4')
    assert store.project.analysis.brackets['wall_mm'] == 4.0 and ctl.stale
    assert ctl.bracket(f.uid) is not b and ctl.meshed(f.uid) is b and pg.view.mesh is not None
    store.undo.undo()
    pump(app, 0.2)
    assert 'wall_mm' not in store.project.analysis.brackets or store.project.analysis.brackets['wall_mm'] == 3.0
    assert pg.fields['wall_mm'].value() == 3.0 and not ctl.stale and ctl.bracket(f.uid) is b
    # export the selected STL and everything
    path = pg.export_selected(str(tmp_path / 'F.stl'))
    V, F, _, _ = M.read_stl(open(path, 'rb').read())
    assert M.check(V, F)['ok']
    paths = pg.export_all(str(tmp_path))
    assert any(x.endswith('brackets.json') for x in paths) and any(x.endswith('brackets_sheet.png') for x in paths)
    assert os.path.exists(tmp_path / 'brackets' / 'F.stl')
    # undo the move: F needs its standoff again
    while store.undo.canUndo() and store.variant.by_tag('F').standoff_mm == need:
        store.undo.undo()
    pump(app, 0.2)
    _wait(app, lambda: ctl.bracket(f.uid) is not None and ctl.bracket(f.uid).status == 'standoff')
