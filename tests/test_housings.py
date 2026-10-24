"""Sensor and camera housings block the cameras (core.housings, the composited cube maps of core.occlusion): exact
geometry, the analysis on the sample's data rig against an independent computation, per variant, performance (no
static cube rebuilt when a sensor moves), the Issues list and the Inspector line, the camera light, the images."""
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import pytest

from vantage.core import gnss as G
from vantage.core import housings as H
from vantage.core import occlusion as O
from vantage.core import samples
from vantage.core.analysis import Engine, eval_points
from vantage.core.lens import resolve
from vantage.core.project import Camera, Variant, new_uid

HERE = os.path.dirname(os.path.abspath(__file__))


def _rig_project():
    """The sample with its data rig active (6 cameras, a roof OS1-128, five radars)."""
    p = samples.tigor_sample_project()
    v = next(v for v in p.variants if v.sensors)
    p.active_uid = v.uid
    return p, v


def _sensor(v, tag):
    return next(s for s in v.sensors if s.tag == tag)


def _free_camera(v, x=10.0, y=0.0, z=1.0, yaw=0.0, pitch=0.0, tag='T'):
    """A camera in free space ahead of the car (the body only behind it)."""
    ref = v.cameras[0]
    c = Camera(tag=tag, x=x, y=y, z=z, yaw=yaw, pitch=pitch, roll=0.0, module_id=ref.module_id, lens_id=ref.lens_id,
               slot=7)
    v.cameras.append(c)
    return c


def _seg_hits_cylinder(C, P, cx, cy, r, z0, z1):
    """Independent closed form: whether segments C -> P (P (K, 3)) pass through the vertical cylinder."""
    d = P - C
    ox, oy = C[0] - cx, C[1] - cy
    a = d[:, 0] ** 2 + d[:, 1] ** 2
    b = 2 * (ox * d[:, 0] + oy * d[:, 1])
    c = ox * ox + oy * oy - r * r
    disc = b * b - 4 * a * c
    sq = np.sqrt(np.maximum(disc, 0))
    t1, t2 = (-b - sq) / (2 * a), (-b + sq) / (2 * a)
    with np.errstate(divide='ignore', invalid='ignore'):
        za, zb = (z0 - C[2]) / d[:, 2], (z1 - C[2]) / d[:, 2]
    lo = np.maximum.reduce([t1, np.minimum(za, zb), np.zeros(len(P))])
    hi = np.minimum.reduce([t2, np.maximum(za, zb), np.ones(len(P))])
    return (disc >= 0) & (lo <= hi)


# ============================================================================ settings and solids
def test_settings_are_tolerant_and_clamped():
    s = H.HousingSettings.from_dict({'enabled': 'no', 'cameras': 0, 'kinds': 'radar, lidar, bogus', 'margin_mm': 1e9})
    assert s.enabled is False and s.cameras is False and s.kinds == ('lidar', 'radar')
    assert s.margin_mm == H.LIMITS['margin_mm'][1]
    for junk in (None, [], 'x', {'margin_mm': 'abc', 'kinds': 5, 'enabled': 2.5}, {'margin_mm': float('nan')}):
        d = H.HousingSettings.from_dict(junk)
        assert d.enabled and d.cameras and d.kinds == H.KINDS and d.margin_mm == 0.0
    assert H.HousingSettings.from_dict({'margin_mm': -4}).margin_mm == 0.0
    r = H.HousingSettings.from_dict(H.HousingSettings(True, False, ('gnss',), 7.5).to_dict())
    assert (r.enabled, r.cameras, r.kinds, r.margin_mm) == (True, False, ('gnss',), 7.5)
    assert json.loads(json.dumps(r.to_dict()))['kinds'] == ['gnss']
    p, v = _rig_project()
    p.analysis.housings = {'enabled': False}
    assert H.solids(p) == [] and H.key(p) is None and H.housing_set(p) is None


def test_solids_match_the_gnss_sky_check():
    """One source of truth: the same boxes and cylinders the GNSS sky check traces (camera bodies included)."""
    p, v = _rig_project()
    _free_camera(v, 1.0, 0.3, 1.7, 30.0, 10.0)
    sol = {h.uid: h for h in H.solids(p)}
    ref = G._housings(p, v, '')
    assert len(ref) == len(sol) == len(v.sensors) + len(v.cameras)
    for idn, c, half, R, rnd in ref:
        h = sol[idn['uid']]
        assert np.allclose(h.centre, c) and np.allclose(h.half, half) and np.allclose(h.R, R) and h.round == rnd
    lid = sol[_sensor(v, 'L1').uid]
    assert lid.round and lid.what == 'lidar' and lid.name == 'lidar L1' and np.allclose(lid.anchor, _sensor(v, 'L1').pos())
    f = v.by_tag('F')
    cf = sol[f.uid]
    # a camera's module: its front face centred on the lens, its length behind it along the optical axis
    front = cf.centre + cf.R[:, 0] * cf.half[0]
    assert np.allclose(front, f.pos()) and np.allclose(cf.half * 2 * 1000, [43.8, 25.0, 25.0])
    # margin, kinds, hidden objects
    p.analysis.housings = {'margin_mm': 10, 'kinds': ['radar'], 'cameras': False}
    sol2 = H.solids(p)
    assert {h.what for h in sol2} == {'radar'}
    rd = next(h for h in sol2 if h.tag == 'RD1')
    assert np.allclose(rd.half, sol[rd.uid].half + 0.01)
    _sensor(v, 'RD2').visible = False
    assert 'RD2' not in {h.tag for h in H.solids(p)}
    _sensor(v, 'RD3').x = float('nan')
    assert 'RD3' not in {h.tag for h in H.solids(p)}


def test_ray_matches_gnss_and_gives_outward_normals():
    p, v = _rig_project()
    rng = np.random.default_rng(3)
    for h in H.solids(p):
        O_ = h.centre + rng.normal(size=3) * 0.4
        D = (h.centre + rng.uniform(-1.2, 1.2, (500, 3)) * h.half) - O_
        D /= np.linalg.norm(D, axis=1, keepdims=True)
        t, n = H.ray(O_, D, h, normals=True)
        assert np.array_equal(t, G._ray_housing(O_, D, h.centre, h.half, h.R, h.round))
        hit = np.isfinite(t) & (t > 0)
        assert hit.sum() > 50
        P = O_ + D[hit] * t[hit, None]
        assert np.allclose(np.linalg.norm(n[hit], axis=1), 1.0)
        assert (np.einsum('ij,ij->i', n[hit], P - h.centre) > 0).all()           # outward
        assert (np.einsum('ij,ij->i', n[hit], D[hit]) <= 1e-9).all()             # facing the ray
        assert all(H.contains(h, q, 1e-6) for q in P[:20])
    h = H.solids(p)[0]
    assert (H.ray(h.centre, rng.normal(size=(20, 3)), h) == 0).all()             # from inside: at once


# ============================================================================ cube maps
def test_box_and_cylinder_ahead_give_the_exact_depth():
    p, v = _rig_project()
    cam = _free_camera(v)
    rd1, l1 = _sensor(v, 'RD1'), _sensor(v, 'L1')
    rd1.x, rd1.y, rd1.z, rd1.yaw = 12.0, 0.0, 1.0, 0.0
    l1.x, l1.y, l1.z = 10.0, 3.0, 1.0
    e = Engine()
    cube = e.cube(cam.pos(), e.vehicle_geom(p))
    assert cube.static is not cube and cube.base is not None
    dirs = np.array([[1.0, 0.0, 0.0], [1.0, 0.004, -0.003], [0.0, 1.0, 0.0], [0.002, 1.0, 0.004], [-1.0, 0, 0]])
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    depth, shade, hid = cube.lookup_ids(dirs)
    # each lookup is the exact distance along its texel's centre ray
    face, py, px = O.texel_index(dirs, O.FACE_N)
    centre = np.array([O.texel_dirs(f, y, y, x, x, O.FACE_N)[0, 0] for f, y, x in zip(face, py, px)])
    half_x = next(h for h in H.solids(p) if h.tag == 'RD1').half[0]
    # the radar's flat front face 2 m ahead (independent: the plane x = 12 - half depth)
    for k in (0, 1):
        assert abs(depth[k] - (2.0 - half_x) / centre[k, 0]) < 1e-5
        assert abs(depth[k] - (2.0 - half_x) / dirs[k, 0]) < 1e-3           # within a texel of the query ray
        assert cube.housings[hid[k]].tag == 'RD1'
    # the lidar cylinder 3 m to the left (independent: the circle of radius 43.5 mm about (10, 3))
    r = p.sensor_model_for(l1).size_mm[0] / 2000.0

    def circle(d):
        a, b, c = d[0] ** 2 + d[1] ** 2, -6.0 * d[1], 9.0 - r * r
        return (-b - math.sqrt(b * b - 4 * a * c)) / (2 * a)
    for k in (2, 3):
        assert abs(depth[k] - circle(centre[k])) < 1e-5
        assert abs(depth[k] - circle(dirs[k])) < 0.01 and cube.housings[hid[k]].tag == 'L1'
    # dark plastic, lit: present, and darker than any body texel
    assert (shade[:4] > 0).all() and shade[:4].max() <= 104
    body = cube.static.shade[cube.static.shade > 0]
    assert body.min() >= 31 and np.median(body) > 104
    # behind the camera the body, as in the static map (the camera's own module is not in its own map)
    assert hid[4] == -1 and depth[4] == cube.static.lookup(dirs[4:5])[0][0] and np.isfinite(depth[4])
    assert cam.uid not in {h.uid for h in cube.housings}


def test_a_housing_behind_the_camera_changes_nothing_it_sees():
    p, v = _rig_project()
    cam = _free_camera(v, pitch=30.0)
    rd1 = _sensor(v, 'RD1')
    rd1.x, rd1.y, rd1.z = 9.2, 0.0, 1.0                    # 0.8 m behind the lens
    intr = resolve(p.module_for(cam), p.lens_for(cam))
    X, Y = np.meshgrid(np.linspace(10.2, 16, 60), np.linspace(-4, 4, 60))
    P = np.c_[X.ravel(), Y.ravel(), np.zeros(X.size)]
    e = Engine()
    on = eval_points(P, cam, intr, e.cube(cam.pos(), e.vehicle_geom(p)))[0]
    p.analysis.housings = {'enabled': False}
    off = eval_points(P, cam, intr, e.cube(cam.pos(), e.vehicle_geom(p)))[0]
    assert on.sum() > 1000 and np.array_equal(on, off)


def test_own_housing_left_out_and_inside_another_sees_nothing():
    p, v = _rig_project()
    e = Engine()
    vg = e.vehicle_geom(p)
    rng = np.random.default_rng(0)
    D = rng.normal(size=(4000, 3))
    for o in list(v.sensors) + list(v.cameras):
        cube = e.cube(o.pos(), vg)
        assert o.uid not in {h.uid for h in cube.housings}
        assert len(cube.housings) == len(v.sensors) + len(v.cameras) - 1
        assert (cube.lookup(D)[0] > 0).all()               # never the inside of its own box
    # a slightly moved camera (a tolerance study) still leaves its own module out
    f = v.by_tag('F')
    assert f.uid not in {h.uid for h in e.cube(f.pos() + [0.0, 0.06, -0.01], vg).housings}
    # with camera modules switched off, F (12 cm above RD1's optical centre) still sees RD1: F's own is F's
    p.analysis.housings = {'cameras': False}
    assert 'RD1' in {h.tag for h in e.cube(f.pos(), e.vehicle_geom(p)).housings}
    p.analysis.housings = {}
    # a camera whose lens is inside the roof lidar sees nothing at all
    l1 = _sensor(v, 'L1')
    cam = _free_camera(v, l1.x + 0.01, 0.0, l1.z, pitch=20.0)
    vg = e.vehicle_geom(p)
    cube = e.cube(cam.pos(), vg)
    assert (cube.lookup(D)[0] < 0.05).all() and (cube.lookup(D)[0] == 0).mean() > 0.9
    intr = resolve(p.module_for(cam), p.lens_for(cam))
    P = np.c_[rng.uniform(1, 9, 500), rng.uniform(-4, 4, 500), np.zeros(500)]
    assert not eval_points(P, cam, intr, cube)[0].any()


def test_disabled_gives_the_old_cubes_bit_for_bit():
    p, v = _rig_project()
    p.analysis.housings = {'enabled': False}
    e = Engine()
    vg = e.vehicle_geom(p)
    assert vg.housings is None and vg.static is vg and not isinstance(vg.occ, O.HousedOccluder)
    for tag in ('F', 'ML', 'QR'):
        c = v.by_tag(tag)
        cube = e.cube(c.pos())
        old = O.build_cube(vg.occ, c.pos())
        assert np.array_equal(cube.depth, old.depth) and np.array_equal(cube.shade, old.shade)
        assert cube.hid is None and cube.housings == [] and cube.base is None
    res_off = e.analyse(p)
    # with housings on the static maps underneath are the same arrays
    p.analysis.housings = {}
    e2 = Engine()
    res_on = e2.analyse(p)
    for c in v.cameras:
        assert np.array_equal(res_on.cubes[c.uid].static.depth, res_off.cubes[c.uid].depth)
        assert np.array_equal(res_on.cubes[c.uid].static.shade, res_off.cubes[c.uid].shade)
    # variants without sensors whose cameras do not see each other's modules are unchanged
    for q in p.variants:
        if q.sensors:
            continue
        p.analysis.housings = {'enabled': False}
        a = Engine().analyse(p, variant=q)
        p.analysis.housings = {}
        b = Engine().analyse(p, variant=q)
        assert a.metrics['blind_m2'] == b.metrics['blind_m2'] and a.metrics['multi_pct'] == b.metrics['multi_pct']


# ============================================================================ the analysis
def test_camera_behind_the_roof_lidar_loses_its_shadow_on_the_ground():
    """A camera 0.3 m behind the roof lidar looking forward and down: the ground it loses is the lidar's shadow, as
    an independent closed-form test of each cell's line of sight against the lidar cylinder finds it. The roof's
    front edge hides the ground nearer than ~10 m from up there, so the ROI reaches 15 m (10 cm cells)."""
    p, v = _rig_project()
    l1 = _sensor(v, 'L1')
    m = p.sensor_model_for(l1)
    cam = _free_camera(v, l1.x - 0.30, 0.0, l1.z + 0.02, pitch=22.0, tag='RL')
    p.roi.distance = 15.0
    p.analysis.housings = {'enabled': False}
    off = Engine().analyse(p, cell=0.1)
    p.analysis.housings = {}
    on = Engine().analyse(p, cell=0.1)
    i_off, i_on = off.cam_index(cam.uid), on.cam_index(cam.uid)
    vis_off, vis_on = off.cams[i_off].vis, on.cams[i_on].vis
    assert not (vis_on & ~vis_off).any()                   # housings only take ground away
    lost = vis_off & ~vis_on
    seen_off = off.per_cam[cam.uid]['seen_m2']
    seen_on = on.per_cam[cam.uid]['seen_m2']
    assert seen_on < seen_off - 5.0
    # independent: the lidar cylinder (radius, base and top from the library size) between the lens and each cell
    g = on.grid
    P = g.points(0.0).reshape(-1, 3)
    r = m.size_mm[0] / 2000.0
    base = l1.z - m.optical_z_mm / 1000.0
    shadow = _seg_hits_cylinder(cam.pos(), P, l1.x, l1.y, r, base, base + m.size_mm[2] / 1000.0).reshape(lost.shape)
    expect = vis_off & shadow
    assert expect.sum() > 400
    diff = lost ^ expect
    from scipy import ndimage
    edge = ndimage.binary_dilation(expect, iterations=2) & ~ndimage.binary_erosion(expect, iterations=2)
    assert (diff & ~edge).sum() == 0                       # every disagreement on the shadow's edge
    assert abs(int(lost.sum()) - int(expect.sum())) <= 0.05 * expect.sum()
    roi = g.roi
    assert abs((lost & roi).sum() * g.cell ** 2 - (seen_off - seen_on)) < 1e-6
    # the other cameras see the camera's own module now (it sits on the roof): never more ground than before
    for c in v.cameras:
        if c.uid != cam.uid:
            assert on.per_cam[c.uid]['seen_m2'] <= off.per_cam[c.uid]['seen_m2'] + 1e-9


def test_each_variant_uses_its_own_housings():
    p, v = _rig_project()
    v2 = Variant.from_dict(v.to_dict())
    v2.uid, v2.name = new_uid(), 'Data rig, lidar on a bumper bar'
    for c in v2.cameras:
        c.uid = new_uid()
    for s in v2.sensors:
        s.uid = new_uid()
    f2 = v2.by_tag('F')
    lid = _sensor(v2, 'L1')
    lid.x, lid.y, lid.z = f2.x + 0.30, 0.0, f2.z - 0.25     # a lidar on a bumper bar ahead of camera F
    p.variants.append(v2)
    p.active_uid = p.variants[0].uid                        # neither is the active one
    e = Engine()
    a = e.analyse(p, variant=v)
    b = e.analyse(p, variant=v2)
    a2 = e.analyse(p, variant=v)                           # the same engine, back again
    sa, sb = a.per_cam[v.by_tag('F').uid]['seen_m2'], b.per_cam[f2.uid]['seen_m2']
    assert sb < sa - 0.05 and a2.per_cam[v.by_tag('F').uid]['seen_m2'] == sa
    assert b.metrics['blind_m2'] > a.metrics['blind_m2']
    # the same as analysing each with it active
    for var, res in ((v, a), (v2, b)):
        p.active_uid = var.uid
        ref = Engine().analyse(p)
        assert ref.metrics['blind_m2'] == res.metrics['blind_m2']
        assert all(ref.per_cam[u]['seen_m2'] == res.per_cam[u]['seen_m2'] for u in ref.per_cam)
        assert e.vehicle_geom(p, var).housings.key == H.key(p, var)


def test_the_geometry_keeps_its_version_and_identity():
    """The renderer's shadow and sky maps and the ROI are keyed on the body + rig version: a housing move must not
    change it; an unchanged project gets the very same geometry object (caches keyed on it stay valid)."""
    p, v = _rig_project()
    e = Engine()
    vg = e.vehicle_geom(p)
    assert e.vehicle_geom(p) is vg and e.vehicle_geom(p.snapshot()) is vg
    grid = e.roi_grid(p)
    _sensor(v, 'RD2').x += 0.1
    vg2 = e.vehicle_geom(p)
    assert vg2 is not vg and vg2.version == vg.version and vg2.mesh is vg.mesh and vg2.body is vg.body
    assert vg2.occ.tri is vg.occ.tri and vg2.static is vg.static and e.roi_grid(p) is grid


# ============================================================================ performance
def test_moving_a_sensor_rebuilds_no_static_cube(monkeypatch):
    p, v = _rig_project()
    calls = []
    real = O.build_cube

    def counting(*a, **k):
        calls.append(1)
        return real(*a, **k)
    monkeypatch.setattr(O, 'build_cube', counting)
    e = Engine()
    e.analyse(p)
    assert len(calls) == len(v.cameras)
    t0 = time.perf_counter()
    e.analyse(p)
    warm = time.perf_counter() - t0
    f = v.by_tag('F').uid
    before = e.analyse(p).per_cam[f]['seen_m2']
    rd1 = _sensor(v, 'RD1')
    rd1.x -= 0.06                                           # behind the bumper fascia
    t0 = time.perf_counter()
    res = e.analyse(p)
    moved = time.perf_counter() - t0
    assert len(calls) == len(v.cameras)                     # no body map rasterised again
    assert res.per_cam[f]['seen_m2'] > before + 0.1         # but F sees past the radar now
    assert moved < max(6 * warm, 0.5)
    # all six composites from scratch (static maps cached) and a camera drag step
    e.cubes._comp.clear()
    e.cubes._layers.clear()
    vg = e.vehicle_geom(p)
    t0 = time.perf_counter()
    for c in v.cameras:
        e.cube(c.pos(), vg)
    assert time.perf_counter() - t0 < 1.0
    ml = v.by_tag('ML')
    ml.x += 0.01
    n = len(calls)
    t0 = time.perf_counter()
    e.analyse(p)
    assert len(calls) == n + 1 and time.perf_counter() - t0 < 2.0     # only the moved camera's own body map


# ============================================================================ issues, inspector
def test_issues_name_the_housing_and_flag_a_camera_inside_one():
    p, v = _rig_project()
    e = Engine()
    vg = e.vehicle_geom(p)
    H.clear_cache()
    t0 = time.perf_counter()
    iss = H.issues(p, vg)
    cold = time.perf_counter() - t0
    t0 = time.perf_counter()
    assert [i.message for i in H.issues(p, vg)] == [i.message for i in iss]
    assert time.perf_counter() - t0 < 0.03 and cold < 0.3
    f = v.by_tag('F')
    rd1 = _sensor(v, 'RD1')
    mine = [i for i in iss if f.uid in i.uids]
    assert len(mine) == 1 and mine[0].severity == 'serious' and rd1.uid in mine[0].uids
    assert mine[0].message.startswith("Camera F: radar RD1's housing covers ") and 'of its image, bottom' in mine[0].message
    iv = H.view_shares(p, f, vg)
    assert iv.shares[0]['tag'] == 'RD1' and iv.shares[0]['share'] > H.SERIOUS_SHARE
    txt, sev = H.in_view_text(iv)
    assert txt.startswith('RD1 ') and txt.endswith('%') and sev == 'serious'
    # the share counts the radar's pixels: the analysis's own cube map, sampled on the same grid, agrees
    cube = e.cube(f.pos(), vg)
    intr = resolve(p.module_for(f), p.lens_for(f))
    gh, gw = iv.mask.shape
    U, V = np.meshgrid((np.arange(gw) + 0.5) / gw * intr.width, (np.arange(gh) + 0.5) / gh * intr.height)
    d_opt, th = intr.unproject(U.ravel(), V.ravel())
    from vantage.core.mathutil import BODY_TO_OPTICAL, rot_zyx
    dw = d_opt @ (BODY_TO_OPTICAL @ rot_zyx(f.yaw, f.pitch, f.roll).T)
    _, _, hid = cube.lookup_ids(dw)
    ok = th <= intr.max_theta
    from_cube = float(((hid >= 0) & ok).sum()) / ok.sum()
    assert abs(from_cube - iv.total) < 0.02
    # a radar behind the fascia does not show: no issue
    rd1.x -= 0.06
    assert not [i for i in H.issues(p, e.vehicle_geom(p)) if f.uid in i.uids]
    assert H.in_view_text(H.view_shares(p, f, e.vehicle_geom(p)))[0] == 'none'
    # a camera inside the lidar: critical
    l1 = _sensor(v, 'L1')
    r = v.by_tag('R')
    r.x, r.y, r.z = l1.x, l1.y, l1.z
    crit = [i for i in H.issues(p, e.vehicle_geom(p)) if i.severity == 'critical']
    assert len(crit) == 1 and crit[0].uids == [r.uid] and 'inside lidar L1' in crit[0].message
    assert H.in_view_text(H.view_shares(p, r, e.vehicle_geom(p))) == ('inside lidar L1: no view', 'critical')
    # through the Issues list, and gone when housings are off or the camera is hidden
    from vantage.core import rules
    assert any(i.message == crit[0].message for i in rules.collect_issues(p, None, e.vehicle_geom(p)))
    r.visible = False
    assert not [i for i in H.issues(p, e.vehicle_geom(p)) if i.severity == 'critical']
    p.analysis.housings = {'enabled': False}
    assert H.issues(p, e.vehicle_geom(p)) == []


def test_nearest_names_what_blocks_a_ray():
    p, v = _rig_project()
    f = v.by_tag('F')
    rd1 = _sensor(v, 'RD1')
    d = rd1.pos() - f.pos()
    t, who = H.nearest(p, f.pos(), np.array([d, -d]))
    assert who[0] is not None and who[0].tag == 'RD1' and 0 < t[0] < np.linalg.norm(d) and who[1] is None
    t, who = H.nearest(p, f.pos(), np.array([[-1.0, 0.0, 0.0]]))
    assert who[0] is None                                  # its own module is left out


# ============================================================================ camera light, images
def test_camera_light_atlas_and_cpu_twin_carry_the_housing():
    from vantage.core.visibility import cam_views, seen_by
    from vantage.ui.render_gl import encode_cubes
    p, v = _rig_project()
    e = Engine()
    e.analyse(p)
    f = v.by_tag('F')
    views = cam_views(p, e, [f], [(1.0, 0.0, 0.0)], 'cached')
    cube = views[0].cube
    assert cube is not None and cube.housings and (cube.hid >= 0).any()
    atlas, N = encode_cubes([cube.depth, cube.static.depth])
    face, py, px = np.nonzero(cube.hid >= 0)
    k = len(face) // 2
    f0, y0, x0 = face[k], py[k], px[k]
    mm = lambda row, col: int(atlas[row, col, 0]) * 256 + int(atlas[row, col, 1])        # noqa: E731
    assert mm(y0, f0 * N + x0) == round(float(cube.depth[f0, y0, x0]) * 1000) and atlas[y0, f0 * N + x0, 2] == 255
    st = cube.static.depth[f0, y0, x0]
    assert mm(N + y0, f0 * N + x0) == (round(float(st) * 1000) if np.isfinite(st) else 0)
    # the whole atlas against a plain per-face encoding
    ref = np.zeros((2 * N, 6 * N, 4), np.uint8)
    for i, cb in enumerate((cube.depth, cube.static.depth)):
        for fc in range(6):
            ok = np.isfinite(cb[fc])
            q = np.where(ok, np.clip(np.round(np.where(ok, cb[fc], 0) * 1000), 0, 65535), 0).astype(np.uint32)
            ref[i * N:(i + 1) * N, fc * N:(fc + 1) * N] = np.stack([q >> 8, q & 255, ok * 255, np.full_like(q, 255)], -1)
    assert np.array_equal(atlas, ref)
    # the CPU twin: ground in the radar's shadow is seen through the static map, not through the composite
    X, Y = np.meshgrid(np.linspace(3.25, 3.7, 46), np.linspace(-0.4, 0.4, 41))
    P = np.c_[X.ravel(), Y.ravel(), np.zeros(X.size)]
    static_view = cam_views(p, e, [f], [(1.0, 0.0, 0.0)], None)[0]
    static_view.cube = cube.static
    lost = seen_by(P, [static_view])[:, 0] & ~seen_by(P, views)[:, 0]
    assert lost.sum() > 100
    # the 3D view's camera light gets the composited map, and a radar move changes it without a body rebuild
    rd1 = _sensor(v, 'RD1')
    rd1.x -= 0.06
    c2 = cam_views(p, e, [f], [(1.0, 0.0, 0.0)], 'cached')[0].cube
    assert c2 is not cube and c2.static is cube.static
    rd = [k for k, h in enumerate(c2.housings) if h.tag == 'RD1']
    assert len(rd) == 1 and not np.isin(c2.hid, rd).any()  # behind the fascia now: the bumper hides it


GL_SCRIPT = r'''
import json, os, sys
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
sys.path.insert(0, sys.argv[1])
sys.path.insert(0, sys.argv[2])
from gl_harness import egl_context
try:
    egl_context()
except Exception as e:
    print(json.dumps({'skip': str(e)}))
    raise SystemExit(0)
import numpy as np
from PyQt6.QtGui import QGuiApplication
app = QGuiApplication([])
from vantage.core import samples
from vantage.core.analysis import Engine
from vantage.core.visibility import cam_views, seen_by
from vantage.ui.render_gl import GLRenderer, Frame, Style
from vantage.ui.viewport3d import OrbitCam
p = samples.tigor_sample_project()
v = next(v for v in p.variants if v.sensors)
p.active_uid = v.uid
e = Engine()
e.analyse(p)
r = GLRenderer()
r.init()
if not r.ok:
    print(json.dumps({'skip': r.error or 'no renderer'}))
    raise SystemExit(0)
W, H = 640, 400
top = OrbitCam()
top.target, top.ortho, top.az, top.el, top.ortho_h = np.array([3.45, 0, 0]), True, -90.0, 89.99, 1.6
def frame(**kw):
    return Frame(top.view(), top.proj(W / H), W, H, target=top.target.copy(), clear=(0.12, 0.12, 0.13, 1.0), **kw)
f = v.by_tag('F')
views = cam_views(p, e, [f], [(1.0, 0.0, 0.0)], 'cached')
st = Style(shadows=False, ao=False)
none = r.render_offscreen(frame(style=st))
lit = r.render_offscreen(frame(cams=views, style=st))
rng = np.random.default_rng(0)
pts = np.c_[rng.uniform(3.25, 3.75, 4000), rng.uniform(-0.45, 0.45, 4000), np.zeros(4000)]
V, P = top.view(), top.proj(W / H)
q = np.c_[pts, np.ones(len(pts))] @ (P @ V).T
sx = ((q[:, 0] / q[:, 3] + 1) * 0.5 * W).astype(int)
sy = ((1 - q[:, 1] / q[:, 3]) * 0.5 * H).astype(int)
ok = (sx >= 2) & (sx < W - 2) & (sy >= 2) & (sy < H - 2)
cpu = seen_by(pts, views)[:, 0]
sv = cam_views(p, e, [f], [(1.0, 0.0, 0.0)], None)[0]
sv.cube = views[0].cube.static
cpu_static = seen_by(pts, [sv])[:, 0]
a, b = lit[sy[ok], sx[ok], :3].astype(int), none[sy[ok], sx[ok], :3].astype(int)
gpu = (a[:, 0] - a[:, 1]) > (b[:, 0] - b[:, 1]) + 12
shadow = cpu_static[ok] & ~cpu[ok]
r.release()
print(json.dumps({'agree': float((gpu == cpu[ok]).mean()), 'shadow': int(shadow.sum()),
                  'shadow_dark': float((~gpu[shadow]).mean()) if shadow.any() else -1.0}))
'''


def test_camera_light_on_the_gpu_shows_the_housing_shadow(tmp_path):
    script = tmp_path / 'housing_gl.py'
    script.write_text(GL_SCRIPT)
    r = subprocess.run([sys.executable, str(script), HERE, os.path.join(HERE, '..')], capture_output=True, text=True,
                       timeout=240)
    line = [ln for ln in r.stdout.splitlines() if ln.startswith('{')]
    assert line, r.stderr[-3000:]
    out = json.loads(line[-1])
    if 'skip' in out:
        pytest.skip(out['skip'])
    assert out['shadow'] > 50 and out['shadow_dark'] > 0.95 and out['agree'] > 0.97, out


def test_synthetic_captures_show_the_housing():
    from vantage.core import synth
    p, v = _rig_project()
    f = v.by_tag('F')
    e = Engine()
    img = synth.render_capture(p, e, f, 320, 200, noise=0)
    world = synth.render_world(p, e, f, synth.parking_world(p), 320, 200, noise=0)
    p.analysis.housings = {'enabled': False}
    img0 = synth.render_capture(p, e, f, 320, 200, noise=0)
    world0 = synth.render_world(p, e, f, synth.parking_world(p), 320, 200, noise=0)
    # the radar's pixels, from the same map the analysis uses
    intr = resolve(p.module_for(f), p.lens_for(f))
    d, th = synth.pixel_rays(intr, 320, 200)
    from vantage.core.mathutil import BODY_TO_OPTICAL, rot_zyx
    p.analysis.housings = {}
    cube = e.cube(f.pos(), e.vehicle_geom(p))
    _, _, hid = cube.lookup_ids(d @ (BODY_TO_OPTICAL @ rot_zyx(f.yaw, f.pitch, f.roll).T))
    rad = ((hid >= 0) & (th <= intr.max_theta)).reshape(200, 320)
    assert rad.mean() > 0.05
    assert img[rad].mean() < 60 and img0[rad].mean() > 90            # dark housing where the floor was
    assert (img[~rad] == img0[~rad]).all()
    assert world[rad].mean() < 50 and np.abs(world[rad].astype(int) - world0[rad]).mean() > 20
    assert (world[~rad] == world0[~rad]).all()


def test_the_sample_dataset_camera_shows_the_housing():
    from vantage.core import dataset as DSET
    from vantage.core import drivescene as DS
    p, v = _rig_project()
    f = v.by_tag('F')
    e = Engine()
    st = DSET.DatasetSettings()
    scene = DSET.scene_for(p, st)
    cs = DS.CameraSim(p, e, f, 192, 120)
    img, ids = cs.render(scene, 0.5, noise=0.0, want_ids=True)
    assert cs.house is not None
    rad = np.zeros(192 * 120, bool)
    rad[cs.pix[cs.house >= 0]] = True
    rad = rad.reshape(120, 192)
    assert rad.mean() > 0.05 and (ids[rad] == DS.P_BODY).all()        # ground truth: the ego vehicle
    p.analysis.housings = {'enabled': False}
    cs0 = DS.CameraSim(p, Engine(), f, 192, 120)
    img0 = cs0.render(scene, 0.5, noise=0.0)
    assert cs0.house is None
    lum = img.astype(float).mean(axis=2)
    assert lum[rad].mean() < 45 and img0.astype(float).mean(axis=2)[rad].mean() > lum[rad].mean() + 30


# ============================================================================ the report
def test_report_with_and_without_housings():
    p, v = _rig_project()
    e = Engine()
    steps = []
    res = H.run(p, e, progress=steps.append)
    assert steps[-1] == 1.0 and len(res.cameras) == len(v.cameras) and len(res.solids) == 12
    on, off = res.coverage['on'], res.coverage['off']
    assert on['per_camera_seen_m2']['F'] < off['per_camera_seen_m2']['F'] - 0.1
    assert on['blind_m2'] >= off['blind_m2'] and on['multi_pct'] <= off['multi_pct']
    s = H.summary(res)
    json.dumps(s)
    assert s['coverage']['seen_m2']['F']['housings'] < s['coverage']['seen_m2']['F']['body_only']
    assert s['cameras'][0]['tag'] == 'F' and s['cameras'][0]['in_view'][0]['housing'] == 'RD1'
    doc = H.to_doc(res, p)
    assert doc['schema'] == 'vantage.housings/1' and doc['frame'].startswith('vehicle') and doc['variant'] == v.name
    lid = next(h for h in doc['housings'] if h['tag'] == 'L1')
    assert lid['shape'] == 'cylinder' and abs(lid['half_size'][0] - 0.0435) < 1e-6
    assert doc['coverage']['change']['per_camera_seen_m2']['F'] < -0.1
    names = [n for n, _ in H.files(res, p)]
    assert names == ['housings.json', 'housings.csv']
    csv_text = dict(H.files(res, p))['housings.csv'].decode()
    assert csv_text.splitlines()[0].split(',') == H.CSV_COLUMNS and ',RD1,radar,' in csv_text
    # the project itself is untouched (its settings, its active variant)
    assert p.analysis.housings == {} and p.variant is v
    assert H.run(p, e, cancel=lambda: True) is None
    # another variant, without sensors
    q = p.variants[0]
    r2 = H.run(p, e, variant=q, coverage=False)
    assert r2.variant_uid == q.uid and r2.coverage is None and len(r2.solids) == len(q.cameras)


def test_export_writes_the_files(tmp_path):
    p, v = _rig_project()
    res = H.run(p, Engine(), coverage=False)
    paths = H.export(res, p, str(tmp_path / 'out'))
    assert [os.path.basename(x) for x in paths] == ['housings.json', 'housings.csv']
    doc = json.load(open(paths[0]))
    assert doc['cameras'][0]['in_view'][0]['housing'] == 'RD1' and 'coverage' not in doc


# ============================================================================ GUI: the Inspector line
from test_gui import app, win, pump, settle  # noqa: E402,F401  (fixtures)


def test_inspector_shows_the_housings_in_view(app, win):  # noqa: F811
    st = win.store
    rig = next(v for v in st.project.variants if v.sensors)
    win.switch_variant(rig.uid)
    settle(app, win)
    f = st.variant.by_tag('F')
    st.select_cams([f.uid])
    settle(app, win)
    page = win.inspector.pages['camera']
    assert win.inspector.stack.currentWidget() is page
    assert page.house_v.v.text().startswith('RD1 ') and page.house_v.v.text().endswith('%')
    assert any('RD1' in i.message and 'Camera F' in i.message for i in win.ctx.analysis.issues)
    seen_before = win.ctx.analysis.result.per_cam[f.uid]['seen_m2']
    # move the radar back behind the fascia in its own Inspector page, as a user types it
    rd1 = next(s for s in st.variant.sensors if s.tag == 'RD1')
    st.select_obj(('sensor', rd1.uid))
    settle(app, win)
    sp = win.inspector.pages['sensor']
    sp.px.lineEdit().setText(f'{rd1.x - 0.06:.4f}')
    sp.px._on_finished()
    settle(app, win)
    assert abs(st.variant.sensor(rd1.uid).x - (3.234 - 0.06)) < 1e-6
    st.select_cams([f.uid])
    settle(app, win)
    assert page.house_v.v.text() == 'none'
    assert win.ctx.analysis.result.per_cam[f.uid]['seen_m2'] > seen_before + 0.1
    assert not any('RD1' in i.message and 'Camera F' in i.message for i in win.ctx.analysis.issues)
    st.undo.undo()
    settle(app, win)
    win.inspector.refresh()
    assert page.house_v.v.text().startswith('RD1 ')
    assert abs(win.ctx.analysis.result.per_cam[f.uid]['seen_m2'] - seen_before) < 1e-9
