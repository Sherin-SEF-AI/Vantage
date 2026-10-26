"""Road geometry and body motion (core.terrain), its hooks in the coverage engine (core.analysis), the suspension's
load transfer and road inputs (core.suspension), the horizon moves (core.loadcases) and the page (Road and body motion).

Independent checks: flat ground against a copy of the engine's per-camera code as it was before the road (bit for
bit), each primitive against its closed form and its gradient against finite differences, the ray march against a
dense brute-force march, the hidden road beyond a crest against the tangent from the camera, the stance on a plane and
on a crowned road against the plane's slope, the suspension on a twisted road against the least-squares plane, brake
dive and cornering roll against hand calculations (springs and tyres in series, roll centres, anti-dive), the moment
balance, and the horizon against a rectilinear camera's tan law."""
import json
import math
import os
import time

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
import pytest

from vantage.core import loadcases as LC, samples, suspension as SU, terrain as TR
from vantage.core.analysis import Engine, eval_points, iso_segments
from vantage.core.lens import Intrinsics, resolve
from vantage.core.mathutil import BODY_TO_OPTICAL, rot_zyx

from test_gui import app, win, pump, settle, _wait   # noqa: F401 (fixtures)

G = SU.G


@pytest.fixture(scope='module')
def proj():
    return samples.tigor_sample_project()


@pytest.fixture(scope='module')
def engine():
    return Engine()


def _with(p, features, **kw):
    """A snapshot of p with a road."""
    q = p.snapshot()
    q.analysis.terrain = dict({'features': features}, **kw)
    return q


# ============================================================================ flat ground: as before, bit for bit
def _old_cams(engine, project, variant):
    """The engine's per-camera loop as it was before the road (a copy): visibility, density, focus and field."""
    from scipy import ndimage
    grid = engine.roi_grid(project, None, variant)
    vg = engine.vehicle_geom(project, variant)
    P = grid.points(0.0).reshape(-1, 3)
    shape = (grid.ny, grid.nx)
    out = []
    for cam in [c for c in variant.cameras if c.visible]:
        intr = resolve(project.module_for(cam), project.lens_for(cam))
        cube = engine.cube(cam.pos(), vg)
        vis, dens, focus, fld = eval_points(P, cam, intr, cube)
        vis = vis.reshape(shape) & ~grid.body
        dens = np.where(vis, dens.reshape(shape), 0).astype(np.float32)
        fld = np.where(grid.body, -1.0, fld.reshape(shape))
        segs = iso_segments(ndimage.gaussian_filter(np.clip(fld, -0.3, 0.3), 0.9), 0.0, grid.x0, grid.y0, grid.cell)
        if len(segs):
            mid = segs.mean(axis=1)
            ii = np.clip(((mid[:, 0] - grid.x0) / grid.cell).astype(int), 0, grid.nx - 1)
            jj = np.clip(((mid[:, 1] - grid.y0) / grid.cell).astype(int), 0, grid.ny - 1)
            segs = segs[grid.dist[jj, ii] <= (project.roi.distance if project.roi.mode == 'offset' else 1e9)
                        + grid.cell]
        out.append((cam.uid, vis, dens, focus.reshape(shape), fld, segs))
    return out


@pytest.mark.parametrize('settings', [None, {}, {'enabled': True, 'features': []},
                                      {'enabled': False, 'features': [{'kind': 'bump', 'start': 0.3}]},
                                      {'features': 'junk', 'motion': {'long_g': 'x'}}])
def test_flat_ground_is_bit_for_bit_the_analysis_before_the_road(proj, engine, settings):
    for v in proj.variants:
        q = proj.snapshot()
        if settings is not None:
            q.analysis.terrain = settings
        res = engine.analyse(q, variant=v)
        assert res.terrain is None
        old = _old_cams(engine, q, v)
        assert [c.uid for c in res.cams] == [o[0] for o in old]
        for ce, (_, vis, dens, focus, fld, segs) in zip(res.cams, old):
            assert np.array_equal(ce.vis, vis) and np.array_equal(ce.dens, dens) and np.array_equal(ce.focus, focus)
            assert np.array_equal(ce.field, fld) and np.array_equal(ce.segments, segs)
        flat = engine.analyse(q, variant=v, terrain=False)
        assert flat.metrics == res.metrics and np.array_equal(flat.best_dens, res.best_dens)
        assert flat.obstacles == res.obstacles and [s.angle for s in flat.seams] == [s.angle for s in res.seams]


def test_old_projects_and_tolerant_settings(proj, tmp_path):
    from vantage.core.project import AnalysisSettings, Project
    a = AnalysisSettings.from_dict({'cell_m': 0.05})                  # a project from before this round
    assert a.terrain == {} and TR.TerrainSettings.from_dict(a.terrain).features == []
    st = TR.TerrainSettings.from_dict({
        'enabled': 'no', 'range_m': 1e9, 'min_px': -4,
        'features': [{'kind': 'bump', 'height': 9, 'length': 'x', 'start': float('nan'), 'where': 'sideways'},
                     {'kind': 'tunnel'}, 'junk', {'kind': 'KERB', 'side': 'Left', 'offset': -7}] + [{'kind': 'ramp'}] * 20,
        'motion': {'long_g': -9, 'lat_g': float('inf'), 'anti_dive': 250, 'drive': 'rocket', 'cg_height': -1,
                   'load_case': 3}})
    assert not st.enabled and st.range_m == TR.LIMITS['range_m'][1] and st.min_px == TR.LIMITS['min_px'][0]
    b = st.features[0]
    assert b == {'kind': 'bump', 'height': 0.5, 'length': 0.6, 'start': 2.0, 'where': 'ahead'}
    assert st.features[1] == {'kind': 'kerb', 'height': 0.12, 'offset': -1.0, 'side': 'left'}
    assert len(st.features) == TR.MAX_FEATURES
    assert st.long_g == -1.5 and st.lat_g == 0.0 and st.dynamics.anti_dive == 100.0 and st.dynamics.drive == 'front'
    assert st.dynamics.cg_height == 0.0 and st.load_case == ''
    assert TR.TerrainSettings.from_dict(st.to_dict()).to_dict() == st.to_dict()        # round trip
    # saved with the project
    q = proj.snapshot()
    q.analysis.terrain = {'features': [{'kind': 'crest', 'radius': 250, 'grade': -6, 'start': 7}]}
    path = str(tmp_path / 'p.vantage')
    q.save(path)
    r = Project.load(path)
    assert TR.settings_of(r).features == [{'kind': 'crest', 'radius': 250.0, 'grade': -6.0, 'start': 7.0,
                                           'where': 'ahead'}]


# ============================================================================ the height field
REF = {'front_x': 3.0, 'rear_x': -1.0, 'left_y': 0.9, 'right_y': -0.9, 'width': 1.8}


def _closed_form(f, x, y):
    """Each primitive's height from its definition, written out again."""
    k = f['kind']
    if k == 'crown':
        return -f['crown'] / 100 * abs(y - f['offset'])
    if k == 'kerb' and f['side'] in ('left', 'right'):
        s = (y - (REF['left_y'] + f['offset'])) if f['side'] == 'left' else ((REF['right_y'] - f['offset']) - y)
        return f['height'] * min(max(s / TR.FACE_M, 0.0), 1.0)
    side = f.get('where', f.get('side'))
    dist = f.get('start', f.get('offset'))
    s = x - (REF['front_x'] + dist) if side == 'ahead' else (REF['rear_x'] - dist) - x
    if s <= 0:
        return 0.0
    if k == 'ramp':
        return f['grade'] / 100 * s
    if k == 'kerb':
        return f['height'] * min(s / TR.FACE_M, 1.0)
    if k == 'bump':
        return f['height'] * math.sin(math.pi * s / f['length']) ** 2 if s < f['length'] else 0.0
    if k == 'garage':
        g, L = f['grade'] / 100, f['transition']
        return g / 2 * s if s < L else g / 2 * L + g * (s - L)
    g, R = f['grade'] / 100, f['radius']                       # crest: the integral of a grade falling as s / R
    Lc = R * abs(g)
    return math.copysign(1, g) * s * s / (2 * R) if s < Lc else math.copysign(1, g) * Lc * Lc / (2 * R) + g * (s - Lc)


FEATS = [{'kind': 'ramp', 'grade': 7.0, 'start': 2.0, 'where': 'ahead'},
         {'kind': 'crest', 'radius': 150.0, 'grade': -9.0, 'start': 1.0, 'where': 'behind'},
         {'kind': 'crest', 'radius': 80.0, 'grade': 5.0, 'start': 4.0, 'where': 'ahead'},
         {'kind': 'garage', 'grade': -16.0, 'transition': 2.0, 'start': 1.5, 'where': 'ahead'},
         {'kind': 'bump', 'height': 0.09, 'length': 0.7, 'start': 1.2, 'where': 'behind'},
         {'kind': 'kerb', 'height': 0.13, 'offset': 0.6, 'side': 'left'},
         {'kind': 'kerb', 'height': 0.10, 'offset': 0.4, 'side': 'right'},
         {'kind': 'kerb', 'height': 0.15, 'offset': 3.0, 'side': 'ahead'},
         {'kind': 'crown', 'crown': 2.5, 'offset': 1.2}]


@pytest.mark.parametrize('f', FEATS, ids=[f['kind'] + str(i) for i, f in enumerate(FEATS)])
def test_each_primitive_matches_its_closed_form_and_gradient(f):
    ter = TR.Terrain([f], REF)
    rng = np.random.default_rng(3)
    x, y = rng.uniform(-25, 30, 400), rng.uniform(-6, 6, 400)
    h = ter.height(x, y)
    assert np.allclose(h, [_closed_form(f, a, b) for a, b in zip(x, y)], atol=1e-12)
    # the gradient against central differences, away from the kinks
    e = 1e-5
    gx, gy = ter.gradient(x, y)
    nx = (ter.height(x + e, y) - ter.height(x - e, y)) / (2 * e)
    ny = (ter.height(x, y + e) - ter.height(x, y - e)) / (2 * e)
    q = TR.Feature(f, REF)
    s = q.s(x, y)
    smooth = np.all([np.abs(s - b) > 1e-3 for b in q.breaks()], axis=0) & (np.abs(q.q(x, y)) > 1e-3)
    assert smooth.sum() > 300
    assert np.allclose(gx[smooth], nx[smooth], atol=1e-6) and np.allclose(gy[smooth], ny[smooth], atol=1e-6)
    # the box bounds hold the field (sampled densely)
    lo, hi = ter.bounds(-5, -2, 12, 2)
    X, Y = np.meshgrid(np.linspace(-5, 12, 801), np.linspace(-2, 2, 81))
    H = ter.height(X, Y)
    assert H.min() >= lo - 1e-9 and H.max() <= hi + 1e-9


def test_primitives_add_up():
    ter = TR.Terrain(FEATS, REF)
    x, y = np.linspace(-20, 20, 97), np.linspace(-5, 5, 97)
    assert np.allclose(ter.height(x, y), sum(TR.Terrain([f], REF).height(x, y) for f in FEATS), atol=1e-12)


def _brute_clear(ter, C, P, n=4000):
    t = np.linspace(0, 1, n + 1)[1:-1]
    out = np.zeros(len(P), bool)
    for a in range(0, len(P), 100):
        d = P[a:a + 100] - C
        L = np.linalg.norm(d, axis=1)
        X, Y, Z = (C[k] + t[None] * d[:, k:k + 1] for k in range(3))
        far = (1 - t[None]) * L[:, None] > 0.012
        out[a:a + 100] = ((Z < ter.height(X, Y) - 0.002 - 0.001 * (1 - t[None]) * L[:, None]) & far).any(1)
    gx, gy = ter.gradient(P[:, 0], P[:, 1])
    v = C - P
    return ~out & (-gx * v[:, 0] - gy * v[:, 1] + v[:, 2] > 0)


@pytest.mark.parametrize('cam', [(3.2, 0.0, 0.62), (-0.9, 0.3, 0.55), (1.9, 0.95, 0.95), (1.0, -1.0, 0.3)])
def test_ray_march_matches_a_dense_brute_force_march(cam):
    ter = TR.Terrain(FEATS[:1] + FEATS[3:], REF)
    rng = np.random.default_rng(abs(int(cam[0] * 100)) + 7)
    P = np.c_[rng.uniform(-12, 20, 3000), rng.uniform(-5, 5, 3000), np.zeros(3000)]
    P[:, 2] = ter.height(P[:, 0], P[:, 1])
    C = np.array(cam, float)
    C[2] += float(ter.height(C[0], C[1]))
    got = ter.clear(C, P)
    ref = _brute_clear(ter, C, P)
    assert 20 < (~ref).sum() < len(P) - 20            # the case is not trivial
    assert (got != ref).sum() <= 2                    # (a grazing ray may land either side of the tolerance)


def test_hidden_road_beyond_a_crest_matches_the_tangent_from_the_camera(proj, engine):
    R, start = 250.0, 6.0
    q = _with(proj, [{'kind': 'crest', 'radius': R, 'grade': -10.0, 'start': start}])
    ter = TR.build(q)
    views = TR.Views(q, engine)
    views.cams = [v for v in views.cams if v[0].tag == 'F']        # the front camera alone
    pr = TR.profile(ter, views, ter.ref, 'ahead', 60.0)
    assert pr['seen_to_flat_m'] == pytest.approx(60.0) and pr['hidden']
    # the line from the camera tangent to the curve: beyond the tangent point the road is out of sight
    c = views.cams[0][0]
    a = ter.ref['front_x'] + start - c.x                 # camera to the curve's start, along x
    s_t = -a + math.sqrt(a * a + 2 * R * c.z)            # s² + 2 a s - 2 R z = 0 (the parabola -s²/2R)
    assert pr['hidden'][0][0] == pytest.approx(start + s_t, abs=0.3)      # the profile samples every 0.25 m
    assert pr['seen_to_m'] == pytest.approx(start + s_t, abs=0.3)
    # a narrow bump hides the ground behind it where the camera's ray passes under its top (and only there)
    ref = TR.references(q)
    H = 0.15
    x_c = ref['front_x'] + 1.0 + 0.05
    wall = TR.Terrain([{'kind': 'bump', 'height': H, 'length': 0.1, 'start': 1.0, 'where': 'ahead'}], ref)
    C = np.array([ref['front_x'] - 0.2, 0.0, 0.62])
    xs = x_c + np.linspace(0.2, 3.0, 60)
    z_at = C[2] * (xs - x_c) / (xs - C[0])           # the ray's height over the bump's top
    got = wall.clear(C, np.c_[xs, np.zeros(60), np.zeros(60)])
    assert got[z_at > 1.2 * H].all() and not got[z_at < 0.8 * H].any()
    assert (z_at > 1.2 * H).any() and (z_at < 0.8 * H).any()


# ============================================================================ the car standing on the road
def test_stance_on_a_plane_follows_its_slope(proj):
    for grade in (-15.0, 6.0, 12.0):
        q = _with(proj, [{'kind': 'ramp', 'grade': grade, 'start': -8.0}])        # the whole car on the grade
        ter = TR.build(q)
        st = ter.stance
        assert st['pitch_deg'] == pytest.approx(-math.degrees(math.atan(grade / 100)), abs=1e-6)
        assert st['roll_deg'] == pytest.approx(0.0, abs=1e-9) and not st['lifted']
        # every tyre rests on the road: the wheel centre one radius above the contact under it
        r = SU.wheel_radius(q)
        for c, (x, y) in SU.corner_positions(q, SU.settings_for(q)).items():
            w = ter.R @ np.array([x, y, r]) + ter.t
            assert w[2] - r == pytest.approx(TR._contact_height(ter, w[0], w[1], r), abs=1e-6)
    crown = 3.0
    q = _with(proj, [{'kind': 'crown', 'crown': crown, 'offset': 1.75}])           # in the right-hand lane
    st = TR.build(q).stance
    assert st['roll_deg'] == pytest.approx(math.degrees(math.atan(crown / 100)), abs=1e-6)   # right side lower
    assert st['pitch_deg'] == pytest.approx(0.0, abs=1e-9)
    # nothing under the tyres: as on flat ground, no pose at all
    ter = TR.build(_with(proj, [{'kind': 'bump', 'start': 1.0}]))
    assert ter.R is None and ter.pose() is None and ter.stance['flat']


def test_suspension_takes_a_twist_as_the_least_squares_plane():
    st = SU.SuspensionSettings()
    for c in SU.CORNERS:
        st.corners[c] = SU.Corner(spring_rate=25.0, motion_ratio=1.0, tyre_rate=200.0, kerb_load=300.0,
                                  bump_rate=0.0, bump_curve=[])
    st.arb_front = st.arb_rear = 0.0
    xy = {'FL': (2.5, 0.75), 'FR': (2.5, -0.75), 'RL': (0.0, 0.75), 'RR': (0.0, -0.75)}
    g = {'FL': 30.0, 'FR': 0.0, 'RL': 0.0, 'RR': 0.0}
    sv = SU.solve(st, xy, [], ground=g)
    A = np.array([[1, xy[c][0], xy[c][1]] for c in SU.CORNERS])
    plane = A @ np.linalg.lstsq(A, np.array([g[c] for c in SU.CORNERS]), rcond=None)[0]
    for i, c in enumerate(SU.CORNERS):
        assert sv.corners[c].drop_mm == pytest.approx(-plane[i], abs=1e-6)
    # with anti-roll bars the share differs, but the forces still balance and the body stays a plane
    st.arb_front = 15.0
    sv = SU.solve(st, xy, [], ground=g)
    F = np.array([sv.corners[c].added_n for c in SU.CORNERS])
    assert abs(F.sum()) < 1e-6 and abs((F * A[:, 1]).sum()) < 1e-6 and abs((F * A[:, 2]).sum()) < 1e-6
    # a road that is one plane loads no spring
    sv = SU.solve(st, xy, [], ground={c: 5.0 + 20.0 * xy[c][0] - 8.0 * xy[c][1] for c in SU.CORNERS})
    assert all(abs(sv.corners[c].travel_mm) < 1e-9 for c in SU.CORNERS)
    assert sv.pitch_deg == pytest.approx(-math.degrees(math.atan(0.020)), abs=1e-9)
    assert sv.roll_deg == pytest.approx(math.degrees(math.atan(-0.008)), abs=1e-9)


def test_garage_ramp_under_the_front_wheels(proj):
    q = _with(proj, [{'kind': 'garage', 'grade': -15.0, 'transition': 2.4, 'start': -1.5}])
    ter = TR.build(q)
    xy = SU.corner_positions(q, SU.settings_for(q))
    s = xy['FL'][0] - (ter.ref['front_x'] - 1.5)
    assert 0 < s < 2.4                                                   # the front tyres on the transition
    drop = -0.075 * s
    L = xy['FL'][0] - xy['RL'][0]
    assert ter.stance['pitch_deg'] == pytest.approx(math.degrees(math.atan(-drop / L)), abs=0.03)
    assert ter.stance['contact_mm']['RL'] == pytest.approx(0.0, abs=1e-9)


def test_clearance_and_driving_over(proj):
    q = _with(proj, [{'kind': 'bump', 'height': 0.30, 'length': 0.6, 'start': 1.0}])
    ter = TR.build(q)
    under = TR.underside(q.vehicle.mesh)
    assert len(under) > 500
    gc = q.vehicle.measured()['Ground clearance']
    assert float(under[:, 2].min()) == pytest.approx(gc, abs=0.01)     # the lowest body point: the clearance
    cl = TR.clearance(ter, q)
    assert cl['body_m'] == pytest.approx(gc, abs=0.01)                   # standing in front of it: untouched
    dr = TR.drive_over(ter, q, 0)
    assert dr['body_m'] < 0                                              # a 30 cm hump hits the underbody
    res = TR.run(q, None, cancel=None)
    assert any(f['kind'] == 'touch' and f['severity'] == 'serious' for f in res.findings)
    low = TR.drive_over(TR.build(_with(proj, [{'kind': 'bump', 'height': 0.05, 'length': 0.6, 'start': 1.0}])), q, 0)
    assert 0.05 < low['body_m'] < gc


# ============================================================================ braking, accelerating, cornering
def _square(k_w=25.0, k_t=200.0, arb=0.0, kerb=300.0):
    st = SU.SuspensionSettings()
    for c in SU.CORNERS:
        st.corners[c] = SU.Corner(spring_rate=k_w, motion_ratio=1.0, tyre_rate=k_t, kerb_load=kerb, bump_rate=0.0,
                                  bump_curve=[])
    st.arb_front = st.arb_rear = arb
    return st


L_, T_ = 2.5, 1.5
XY = {'FL': (L_, T_ / 2), 'FR': (L_, -T_ / 2), 'RL': (0.0, T_ / 2), 'RR': (0.0, -T_ / 2)}


def _series(a, b):
    return a * b / (a + b)


@pytest.mark.parametrize('ad,al', [(0.0, 0.0), (30.0, 10.0), (100.0, 100.0)])
def test_brake_dive_against_the_hand_calculation(ad, al):
    st = _square()
    dyn = SU.Dynamics(cg_height=0.55, anti_dive=ad, anti_lift=al, unsprung_kg=0.0)
    sv, tr = SU.dynamic_solve(st, XY, [], -0.8, 0.0, dyn)
    M = st.kerb_mass
    dF = M * 0.8 * G * 0.55 / L_                       # the front axle gains, the rear loses (N)
    front = (1 - ad / 100) * dF / 2 / 25.0 + dF / 2 / 200.0       # mm: spring (less the linkage's share) + tyre
    rear = -((1 - al / 100) * dF / 2 / 25.0 + dF / 2 / 200.0)
    assert sv.corners['FL'].drop_mm == pytest.approx(front, rel=1e-6)
    assert sv.corners['RR'].drop_mm == pytest.approx(rear, rel=1e-6)
    assert sv.pitch_deg == pytest.approx(math.degrees(math.atan((front - rear) / 1000 / L_)), rel=1e-6)
    assert sv.pitch_deg > 0                            # nose down
    F = np.array([sv.corners[c].added_n for c in SU.CORNERS])
    assert abs(F.sum()) < 1e-6
    assert (F * np.array([XY[c][0] for c in SU.CORNERS])).sum() == pytest.approx(tr['moment'][0], rel=1e-9)
    acc, _ = SU.dynamic_solve(st, XY, [], 0.3, 0.0, dyn)
    assert acc.pitch_deg < 0                            # squat: nose up


@pytest.mark.parametrize('arb,rc', [(0.0, 0.0), (12.0, 0.0), (0.0, 0.2)])
def test_cornering_roll_against_the_hand_calculation(arb, rc):
    st = _square(arb=arb)
    h = 0.55
    dyn = SU.Dynamics(cg_height=h, rc_front=rc, rc_rear=rc, unsprung_kg=0.0)
    sv, tr = SU.dynamic_solve(st, XY, [], 0.0, 0.5, dyn)
    M = st.kerb_mass
    # per corner: the geometric share M/4·a·rc/T reaches the tyre through the links, the elastic share deflects the
    # spring and its anti-roll bar (in pure roll the bar adds its rate to the wheel's), all of it the tyre
    a = 0.5 * G
    geo = M / 2 * a * rc / T_                           # per corner: the axle's M/2·a·rc over its track
    ela = M * a * (h - rc) / 2 / T_                     # per corner: half the elastic moment per axle over the track
    drop = ela / (25.0 + arb) + (geo + ela) / 200.0     # mm on the right side, the left rises as much
    assert sv.corners['FR'].drop_mm == pytest.approx(drop, rel=1e-6)
    assert sv.corners['FL'].drop_mm == pytest.approx(-drop, rel=1e-6)
    assert sv.roll_deg == pytest.approx(math.degrees(math.atan(2 * drop / 1000 / T_)), rel=1e-6)
    assert sv.roll_deg > 0                              # a left turn: right side down
    F = np.array([sv.corners[c].added_n for c in SU.CORNERS])
    assert (F * np.array([XY[c][1] for c in SU.CORNERS])).sum() == pytest.approx(-M * a * h, rel=1e-9)
    right, _ = SU.dynamic_solve(st, XY, [], 0.0, -0.5, dyn)
    assert right.roll_deg == pytest.approx(-sv.roll_deg, rel=1e-9)


def test_loads_move_the_centre_of_gravity_and_the_unsprung_share():
    st = _square()
    dyn = SU.Dynamics(cg_height=0.5, unsprung_kg=40.0, anti_dive=0.0, anti_lift=0.0)
    tr = SU.transfer(st, XY, [('roof', 100.0, 1.25, 0.0, 1.6)], -1.0, 0.0, dyn, wheel_r=0.3)
    M = st.kerb_mass + 100.0
    assert tr['mass_kg'] == pytest.approx(M)
    assert tr['cog'][2] == pytest.approx((st.kerb_mass * 0.5 + 100 * 1.6) / M)
    assert tr['moment'][0] == pytest.approx(M * G * tr['cog'][2])
    # with no anti features the links carry only the unsprung masses' transfer
    lf = tr['link']['FL'] + tr['link']['FR']
    assert lf == pytest.approx(4 * 40.0 * G * 0.3 / L_)
    assert tr['sprung_cg_z'] == pytest.approx((M * tr['cog'][2] - 160.0 * 0.3) / (M - 160.0))


def test_motion_cases_on_the_sample(proj):
    q = proj.snapshot()
    q.analysis.terrain = {'motion': {'long_g': -0.8, 'lat_g': 0.0}}
    mc = TR.motion_cases(q)
    cases = {c['name']: c for c in mc['cases']}
    rest = cases['At rest']
    assert rest['rel']['pitch_deg'] == pytest.approx(0.0, abs=1e-9)       # driving load = the calibration load
    hb = cases['Hard braking']
    assert 1.0 < hb['rel']['pitch_deg'] < 4.0 and abs(hb['rel']['roll_deg']) < 1e-6
    assert cases['Sliders']['rel']['pitch_deg'] == pytest.approx(hb['rel']['pitch_deg'], rel=1e-9)
    assert cases['Cornering left']['rel']['roll_deg'] > 1.0 > abs(cases['Cornering left']['rel']['pitch_deg'])
    assert cases['Full acceleration']['rel']['pitch_deg'] < 0
    f = next(h for h in hb['horizon'] if h['tag'] == 'F')
    assert f['facing'] == 'forward' and f['dv_px'] < 0                     # nose down: the horizon rises
    assert f['d_deg'] == pytest.approx(hb['rel']['pitch_deg'], abs=0.02)
    # every camera moves rigidly with the body: the same rotation for all
    turns = {round(d['turn_deg'], 9) for d in hb['devices']}
    assert len(turns) == 1
    assert mc['info']['cg_estimated'] and mc['info']['kerb_cg_height_m'] == pytest.approx(
        SU.TYP_CG_FRACTION * q.vehicle.measured()['Height'], abs=1e-3)


# ============================================================================ the horizon
def test_horizon_of_a_rectilinear_camera_follows_the_tan_law():
    f_mm, pix = 4.0, 0.003
    intr = Intrinsics(width=1920, height=1280, pixel_mm=pix, f_mm=f_mm, projection='rectilinear', kb=(),
                      cx=960.0, cy=640.0, max_theta=math.radians(80), focus_near=0.0, focus_far=math.inf)
    for cam_pitch in (0.0, 10.0, 25.0):
        for body_pitch in (0.0, 1.5, -2.0):
            R = rot_zyx(0.0, body_pitch, 0.0) @ rot_zyx(0.0, cam_pitch, 0.0)
            h = LC.horizon_at(intr, BODY_TO_OPTICAL @ R.T)
            e = math.radians(cam_pitch + body_pitch)
            assert h['elev_deg'] == pytest.approx(cam_pitch + body_pitch, abs=1e-9)
            assert h['v'] == pytest.approx(640.0 - f_mm / pix * math.tan(e), abs=1e-6)
            assert h['tilt_deg'] == pytest.approx(0.0, abs=1e-9)
    for roll in (2.0, -3.0):
        h = LC.horizon_at(intr, BODY_TO_OPTICAL @ rot_zyx(0.0, 0.0, roll).T)
        assert abs(h['tilt_deg']) == pytest.approx(abs(roll), abs=1e-6)
    # looking straight down the horizon is 90° off the axis in every direction (out of this lens' view); rolled 90°
    # the image columns are level and the horizon runs along the centre column: no single crossing
    h = LC.horizon_at(intr, BODY_TO_OPTICAL @ rot_zyx(0.0, 90.0, 0.0).T)
    assert h['elev_deg'] == pytest.approx(90.0) and not h['in_view']
    assert LC.horizon_at(intr, BODY_TO_OPTICAL @ rot_zyx(0.0, 0.0, 90.0).T) is None


# ============================================================================ the road in the coverage analysis
def test_road_in_the_coverage_analysis(proj, engine):
    q = _with(proj, [{'kind': 'bump', 'height': 0.12, 'length': 0.5, 'start': 0.6},
                     {'kind': 'kerb', 'side': 'left', 'offset': 0.5, 'height': 0.15}])
    flat = engine.analyse(q, terrain=False)
    road = engine.analyse(q)
    assert road.terrain is not None and road.terrain.key == TR.active(q).key
    g = road.grid
    # the bump's back face (it turns away from the front camera) is hidden; on flat ground that ground is seen
    k = road.cam_index(next(c.uid for c in q.variant.cameras if c.tag == 'F'))
    x_back = road.terrain.ref['front_x'] + 0.6 + 0.4
    i = int((x_back - g.x0) / g.cell)
    j = int((0.0 - g.y0) / g.cell)
    assert flat.cams[k].vis[j, i] and not road.cams[k].vis[j, i]
    assert road.metrics['blind_m2'] >= flat.metrics['blind_m2']
    # cell for cell (a sample of the region): seen = in the image and clear of the body at the road point (the flat
    # analysis' own test, there) and clear of the road by an independent dense march
    ter = road.terrain
    assert ter.R is None                                   # nothing under the tyres: the car stands as on flat ground
    rng = np.random.default_rng(5)
    jj, ii = np.nonzero(g.roi & ~g.body)
    pick = rng.choice(len(jj), 700, replace=False)
    jj, ii = jj[pick], ii[pick]
    P = np.c_[g.xs[ii], g.ys[jj], np.zeros(len(ii))]
    P[:, 2] = ter.height(P[:, 0], P[:, 1])
    for ce in road.cams:
        cam = next(c for c in q.variant.cameras if c.uid == ce.uid)
        intr = resolve(q.module_for(cam), q.lens_for(cam))
        vis, _, _, _ = eval_points(P, cam, intr, engine.cube(cam.pos(), engine.vehicle_geom(q)))
        want = vis & _brute_clear(ter, cam.pos(), P)
        assert (ce.vis[jj, ii] != want).sum() <= 2, ce.tag
    # the density on the road: the distance to the raised point (independent: the lens' figure at that range)
    cam = next(c for c in q.variant.cameras if c.uid == road.cams[k].uid)
    on = g.roi & road.cams[k].vis & (ter.height(*np.meshgrid(g.xs, g.ys)) > 0.05)
    jj, ii = np.nonzero(on)
    intr = resolve(q.module_for(cam), q.lens_for(cam))
    P = np.c_[g.xs[ii], g.ys[jj], ter.height(g.xs[ii], g.ys[jj])]
    d = P - cam.pos()
    _, W2O = (None, BODY_TO_OPTICAL @ rot_zyx(cam.yaw, cam.pitch, cam.roll).T)
    _, _, th, _ = intr.project_dirs(d @ W2O.T)
    assert np.allclose(road.cams[k].dens[jj, ii], intr.density_px_per_cm(th, np.linalg.norm(d, axis=1)), rtol=1e-5)
    # obstacles stand on the road: a short pole beyond a steep crest is hidden; on flat ground the front camera sees it
    q = _with(proj, [{'kind': 'crest', 'radius': 50.0, 'grade': -15.0, 'start': 3.0}])
    ob = q.obstacles[0]
    ob.x, ob.y, ob.height = TR.references(q)['front_x'] + 20.0, 0.0, 0.3
    f_uid = next(c.uid for c in q.variant.cameras if c.tag == 'F')
    assert f_uid in engine.analyse(q, terrain=False).obstacles[ob.uid]
    assert f_uid not in engine.analyse(q).obstacles[ob.uid]


def test_body_pose_composes_with_the_stance(proj, engine):
    q = _with(proj, [{'kind': 'ramp', 'grade': 8.0, 'start': -8.0}])
    ter = TR.active(q)
    R0, t0 = ter.R, ter.t
    Rb, tb = LC.pose(10.0, 0.5, -0.3)
    R, t = ter.pose((Rb, tb))
    p = np.array([2.0, 0.5, 0.7])
    assert np.allclose(R @ p + t, R0 @ (Rb @ p + tb) + t0)
    assert engine.analyse(q, body_pose=(Rb, tb)) is not None


# ============================================================================ the study
def test_run_on_the_sample_and_exports(proj, engine, tmp_path):
    q = _with(proj, [{'kind': 'bump', 'height': 0.08, 'length': 0.6, 'start': 3.0},
                     {'kind': 'kerb', 'side': 'ahead', 'offset': 6.0, 'height': 0.12},
                     {'kind': 'crest', 'radius': 300.0, 'grade': -8.0, 'start': 8.0}])
    steps = []
    res = TR.run(q, engine, progress=steps.append)
    assert steps and steps[-1] == 1.0 and res.road and res.coverage is not None
    assert len(res.sweeps) == 2 and all(s['first_seen_m'] is not None for s in res.sweeps)
    assert res.profiles['ahead']['hidden'] and res.profiles['ahead']['seen_to_m'] < 30
    assert len(res.motion['cases']) == len(TR.MOTIONS) + 1 and all(c['image'] is not None
                                                                    for c in res.motion['cases'])
    # a taller bump is seen from farther; a stricter pixel minimum sees it later
    tall = TR.run(_with(proj, [{'kind': 'bump', 'height': 0.16, 'length': 0.6, 'start': 3.0}]), engine)
    low = TR.run(_with(proj, [{'kind': 'bump', 'height': 0.08, 'length': 0.6, 'start': 3.0}]), engine)
    strict = TR.run(_with(proj, [{'kind': 'bump', 'height': 0.08, 'length': 0.6, 'start': 3.0}], min_px=8), engine)
    assert tall.sweeps[0]['first_seen_m'] > low.sweeps[0]['first_seen_m'] > strict.sweeps[0]['first_seen_m']
    # the sweep's pixel height at one distance, from the lens directly
    d0, cam_tag, px = next(r for r in low.sweeps[0]['distances'] if r[0] == 5.0)
    cam = next(c for c in q.variant.cameras if c.tag == cam_tag)
    intr = resolve(q.module_for(cam), q.lens_for(cam))
    W2O = BODY_TO_OPTICAL @ rot_zyx(cam.yaw, cam.pitch, cam.roll).T
    x = TR.references(q)['front_x'] + 5.0 + 0.3
    ys = np.linspace(-TR.references(q)['width'] / 2, TR.references(q)['width'] / 2, 9)
    top = np.c_[np.full(9, x), ys, np.full(9, 0.08)] - cam.pos()
    base = np.c_[np.full(9, x), ys, np.zeros(9)] - cam.pos()
    ut, vt, _, _ = intr.project_dirs(top @ W2O.T)
    ub, vb, _, _ = intr.project_dirs(base @ W2O.T)
    assert px == pytest.approx(float(np.median(np.hypot(ut - ub, vt - vb))), rel=0.05)
    # exports
    names = [n for n, _ in TR.files(res, q)]
    assert names == ['terrain.json', 'terrain_motion.csv', 'terrain_profile.csv', 'terrain_profile.png']
    paths = TR.export(res, q, str(tmp_path / 'out'))
    doc = json.loads(open(paths[0], encoding='utf-8').read())
    assert doc['schema'] == TR.SCHEMA and doc['frame'].startswith('vehicle') and doc['road']['used']
    assert len(doc['road']['first_seen']) == 2 and doc['motion']['cases'][1]['name'] == 'Hard braking'
    assert doc['assumptions'] == TR.ASSUMPTIONS
    rows = open(paths[1], encoding='utf-8').read().splitlines()
    assert rows[0].split(',') == list(TR.MOTION_COLUMNS) and len(rows) > len(TR.MOTIONS)
    import cv2
    img = cv2.imread(paths[3])
    assert img is not None and img.shape[1] == 1200
    s = TR.summary(res)
    json.dumps(s, allow_nan=False)
    json.dumps(TR.to_doc(res, q), allow_nan=False)            # plain JSON types (batch writes it as is)
    assert s['road_seen_to_m']['ahead']['flat'] == pytest.approx(60.0) and len(s['motion']) == len(TR.MOTIONS) + 1


def test_run_edge_cases(proj, engine):
    # cancel mid-run
    q = _with(proj, [{'kind': 'bump', 'start': 2.0}])
    calls = []
    assert TR.run(q, engine, progress=calls.append, cancel=lambda: len(calls) >= 2) is None
    # flat ground: motion only, a note, files without the profile
    res = TR.run(proj.snapshot(), engine)
    assert not res.road and res.coverage is None and res.motion['cases']
    assert any(f['kind'] == 'flat' for f in res.findings)
    assert [n for n, _ in TR.files(res, proj)] == ['terrain.json', 'terrain_motion.csv']
    # switched off: kept, not used
    res = TR.run(_with(proj, [{'kind': 'bump'}], enabled=False), engine)
    assert not res.road and any(f['kind'] == 'off' for f in res.findings)
    # a variant without cameras, hidden cameras
    q = _with(proj, [{'kind': 'bump', 'start': 1.0}])
    for c in q.variant.cameras:
        c.visible = False
    res = TR.run(q, engine)
    assert res.road and res.coverage is None and not res.sweeps and res.motion['cases'] and \
        all(c['horizon'] == [] and c['image'] is None for c in res.motion['cases'])
    # no vehicle model
    q = _with(proj, [{'kind': 'bump'}])
    q.vehicle.mesh = None
    res = TR.run(q, engine)
    assert not res.road and res.motion['cases']
    assert TR.issues(q) == []


def test_issues_are_gated_cached_and_cheap(proj):
    from vantage.core import rules
    assert TR.issues(proj.snapshot()) == []
    q = _with(proj, [{'kind': 'bump', 'height': 0.30, 'length': 0.6, 'start': 1.0}])
    t0 = time.perf_counter()
    out = TR.issues(q)
    first = time.perf_counter() - t0
    t0 = time.perf_counter()
    again = TR.issues(q)
    assert time.perf_counter() - t0 < 0.03 and first < 5.0
    assert out == again and out and all(isinstance(i, rules.Issue) for i in out)
    assert any(i.severity == 'serious' and 'touch' in i.message for i in out) and all(i.action == 'terrain'
                                                                                     for i in out)
    assert TR.issues(_with(proj, [{'kind': 'bump', 'height': 0.05, 'start': 1.0}])) == []


# ============================================================================ the page
def _open(win, name, cls):
    if name not in win._page_widgets:
        pg = cls(win.ctx)
        win.pages.addWidget(pg)
        win._page_widgets[name] = pg
    win.set_page(name)
    return win._page_widgets[name]


def test_terrain_page_drives_the_road(app, win, tmp_path):
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    from vantage.ui.workspaces.terrain_ws import TerrainPage
    st = win.store
    pg = _open(win, 'terrain', TerrainPage)
    pump(app, 0.3)
    assert pg.rows == [] and pg.no_rows.isVisible()
    n0 = st.undo.index()
    # add a speed bump: the coverage analysis uses it at once, the views draw it
    pg.add_buttons['bump'].click()
    pump(app, 0.2)
    assert TR.settings_of(st.project).features == [TR.default_feature('bump')] and len(pg.rows) == 1
    assert st.undo.index() == n0 + 1
    settle(app, win)
    assert win.ctx.analysis.result.terrain is not None
    # type a height into its field (mm): one undo step, stored in metres
    fl, scale = pg.rows[0].fields['height']
    fl.setFocus()
    fl.lineEdit().selectAll()
    QTest.keyClicks(fl.lineEdit(), '120')
    QTest.keyClick(fl.lineEdit(), Qt.Key.Key_Return)
    pump(app, 0.2)
    assert TR.settings_of(st.project).features[0]['height'] == pytest.approx(0.12)
    assert 'Speed bump 120 mm' in pg.rows[0].desc.text()
    st.undo.undo()
    pump(app, 0.2)
    assert TR.settings_of(st.project).features[0]['height'] == pytest.approx(0.08)
    assert pg.rows[0].fields['height'][0].value() == pytest.approx(80.0)
    st.undo.redo()
    pump(app, 0.2)
    # the side of a kerb
    pg.add_buttons['kerb'].click()
    pump(app, 0.2)
    pg.rows[1].choice.buttons['ahead'].click()
    pump(app, 0.2)
    assert TR.settings_of(st.project).features[1]['side'] == 'ahead'
    # the study runs (a main-window job), fills the tiles and tables
    pg.b_run.click()
    _wait(app, lambda: pg.ctl.result is not None and not pg.ctl.busy, 120)
    pump(app, 0.3)
    res = pg.ctl.result
    assert res.road and len(res.sweeps) == 2
    assert pg.t_road.rowCount() == 2 and pg.t_motion.rowCount() == len(TR.MOTIONS) + 1
    assert pg.t_cams.rowCount() == len([c for c in st.project.variant.cameras if c.visible])
    assert pg.tiles['seen'].value not in ('—', '') and pg.tiles['brake'].value.startswith('+')
    assert pg.chart.res is res and pg.findings
    # the plan view draws the road's edges; the 3D scene drapes the heat map on the road
    pg.plan.repaint()
    pump(app, 0.2)
    segs, lines = pg.plan._ter_layer
    assert any(style == 'dash' for _, style, _ in lines) and any(style == 'face' for _, style, _ in lines)
    assert len(segs)                                     # height contours over the bump
    scene = win.viewarea.active.scene

    class _VP:
        renderer = None
    assert scene.ground() is None and scene._road() is not None
    td = scene._road_draw(_VP(), scene._road())
    assert len(td.tri) > 100 and td.colors.shape == (len(td.tri), 4)
    assert any(l.dash for l in scene._road_edges(scene._road()))
    # the sliders: a live read-out, saved when let go (one undo step)
    n1 = st.undo.index()
    pg.sl_long.setValue(-60)
    pump(app, 0.1)
    assert 'Pitch +' in pg.readout.text() and 'F horizon' in pg.readout.text()
    _wait(app, lambda: TR.settings_of(st.project).long_g == pytest.approx(-0.6), 10)
    assert st.undo.index() == n1 + 1
    # a body-motion input and the drive
    f = pg.dyn['anti_dive']
    f.setFocus()
    f.lineEdit().selectAll()
    QTest.keyClicks(f.lineEdit(), '40')
    QTest.keyClick(f.lineEdit(), Qt.Key.Key_Return)
    pump(app, 0.2)
    assert TR.settings_of(st.project).dynamics.anti_dive == 40.0
    pg.drive.buttons['rear'].click()
    pump(app, 0.2)
    assert TR.settings_of(st.project).dynamics.drive == 'rear'
    # export
    paths = pg.export(str(tmp_path / 'terrain'))
    assert [os.path.basename(x) for x in paths] == ['terrain.json', 'terrain_motion.csv', 'terrain_profile.csv',
                                                    'terrain_profile.png']
    # switch the road off: flat ground in the coverage analysis, the road kept
    pg.enabled.click()
    pump(app, 0.2)
    assert not TR.settings_of(st.project).enabled and TR.settings_of(st.project).features
    settle(app, win)
    assert win.ctx.analysis.result.terrain is None
    # remove a primitive
    pg.enabled.click()
    pump(app, 0.2)
    pg.rows[0].b_remove.click()
    pump(app, 0.2)
    assert [f['kind'] for f in TR.settings_of(st.project).features] == ['kerb'] and len(pg.rows) == 1
    # a new project: the page follows, the old result goes; flat ground again
    st.set_project(samples.tigor_sample_project())
    pump(app, 0.3)
    assert pg.rows == [] and pg.no_rows.isVisible() and pg.ctl.result is None or pg.ctl.busy
    _wait(app, lambda: not pg.ctl.busy, 120)
    assert pg.ctl.result is None or not pg.ctl.result.road
    settle(app, win)
    assert win.ctx.analysis.result.terrain is None and scene.ground() is not None
