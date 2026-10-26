"""Critical targets and inverse mounting tolerances (core.critical, Analysis ▸ Critical targets, the optimiser's
critical-target objective): checked against closed forms on a pinhole camera, against the cube maps and the
perception range on the sample, and by moving the camera to the limits it reports."""
import copy
import json
import math
import os

import numpy as np
import pytest

from vantage.core import critical as CR
from vantage.core import samples
from vantage.core.analysis import Engine
from vantage.core.lens import Lens, resolve
from vantage.core.mathutil import BODY_TO_OPTICAL, rot_zyx
from vantage.core.project import Camera, Project, Sensor, Variant


# =========================================================================== fixtures
def _rig_project():
    p = samples.tigor_sample_project()
    v = next(v for v in p.variants if v.sensors)
    p.active_uid = v.uid
    return p


def _with_targets(p, labels):
    ts = []
    for lab in labels:
        ts.append(CR.preset(p, lab, ts))
    p.analysis.critical = {'targets': ts}
    return ts


@pytest.fixture(scope='module')
def rig():
    """The data rig with six critical targets and one full run (envelopes included)."""
    p = _rig_project()
    _with_targets(p, ['Child 2 m ahead', 'Pedestrian 10 m ahead', 'Overhead traffic light 15 m ahead',
                      'Sign 15 m ahead, 3 m left', 'Kerb by the right front wheel', 'Lane marking on a 50 m curve'])
    eng = Engine()
    res = CR.run(p, eng)
    return p, eng, res


PIN_F_MM = 4.0


def _pinhole_project(cam_z=1.2, target=None, **st):
    """No vehicle: one rectilinear camera at (0, 0, cam_z) looking along +x, one target."""
    p = Project()
    lens = Lens(id='pin', name='Pinhole', focal_mm=PIN_F_MM, projection='rectilinear', max_half_deg=80.0,
                focus_near_m=0.0)
    p.lenses['pin'] = lens
    c = Camera(tag='P', name='Pinhole', x=0.0, y=0.0, z=cam_z, yaw=0.0, pitch=0.0, roll=0.0, module_id='sturdecam31',
               lens_id='pin')
    v = Variant(name='Pinhole', cameras=[c])
    p.variants = [v]
    p.active_uid = v.uid
    t = CR.clean_target(dict({'kind': 'pedestrian', 'name': 'Ped', 'x': 10.0, 'y': 0.0}, **(target or {})))
    p.analysis.critical = dict({'targets': [t], 'install_mm': 10.0, 'install_deg': 1.0}, **st)
    return p, c, t


def _f_px(p, c):
    intr = resolve(p.module_for(c), p.lens_for(c))
    return intr, intr.f_mm / intr.pixel_mm


# =========================================================================== settings
def test_settings_are_tolerant_clamped_and_round_trip():
    junk = {'targets': [
        {'kind': 'child', 'name': 'A', 'x': 'nan', 'y': float('inf'), 'height': -3, 'min_px': '12', 'yaw': 540},
        {'kind': 'unicorn', 'name': 'A', 'width': 1e9, 'min_share': 7, 'enabled': 'no', 'uid': 'u1'},
        {'kind': 'lane', 'radius': 0.5, 'uid': 'u1'},
        'not a target', None, 7],
        'install_mm': -5, 'install_deg': 'x', 'search_deg': 1000, 'search_mm': 1, 'joint': 'off', 'lidars': 0}
    st = CR.CriticalSettings.from_dict(junk)
    a, b, c = st.targets
    assert (a['x'], a['y']) == (0.0, 0.0) and a['height'] == 0.9 and a['min_px'] == 12.0
    assert a['yaw'] == pytest.approx(180.0) or a['yaw'] == pytest.approx(-180.0)
    assert b['kind'] == 'object' and b['name'] == 'A 2' and b['width'] == 50.0 and b['min_share'] == 1.0
    assert b['enabled'] is False
    assert c['radius'] == CR.MIN_RADIUS and c['uid'] != 'u1' and len({a['uid'], b['uid'], c['uid']}) == 3
    assert st.install_mm == 0.0 and st.install_deg == 0.0 and st.search_deg == 90.0 and st.search_mm == 5.0
    assert st.joint is False and st.lidars is False
    again = CR.CriticalSettings.from_dict(json.loads(json.dumps(st.to_dict())))
    assert again.to_dict() == st.to_dict()
    many = CR.CriticalSettings.from_dict({'targets': [{'kind': 'sign'}] * 50})
    assert len(many.targets) == CR.MAX_TARGETS
    assert CR.CriticalSettings.from_dict(None).targets == [] and CR.CriticalSettings.from_dict([1, 2]).targets == []


def test_install_tolerance_follows_the_project_until_set():
    p = _rig_project()
    tol = CR.install_tolerance(p)
    assert (tol['mm'], tol['deg'], tol['source']) == (10.0, 1.0, 'project (fleet)')
    p.fleet = {'settings': {'pos_tol_mm': 4.0, 'ang_tol_deg': 0.5}}
    assert (CR.install_tolerance(p)['mm'], CR.install_tolerance(p)['deg']) == (4.0, 0.5)
    p.analysis.critical = {'install_mm': 2.0, 'install_deg': 0.25}
    tol = CR.install_tolerance(p)
    assert (tol['mm'], tol['deg'], tol['source']) == (2.0, 0.25, 'critical targets')


def test_presets_stand_where_their_names_say():
    p = _rig_project()
    b = CR.vehicle_box(p)
    meas = p.vehicle.measured()
    assert b['front_x'] == pytest.approx(meas['front_x']) and b['half_w'] == pytest.approx(meas['Width'] / 2)
    ts = _with_targets(p, [lab for lab, *_ in CR.PRESET_PLACES])
    by = {t['name']: t for t in ts}
    child = by['Child 2 m ahead']
    assert child['x'] == pytest.approx(b['front_x'] + 2.0, abs=1e-3) and child['height'] == 0.9
    assert CR.describe_place(p, child) == '2.0 m ahead of the front bumper'
    assert CR.describe_place(p, by['Child 1 m behind']) == '1.0 m behind the rear bumper'
    assert CR.describe_place(p, by['Child beside the front door']) == '0.5 m left of the body'
    assert CR.describe_place(p, by['Sign 15 m ahead, 3 m left']) == '15.0 m ahead of the front bumper, 3.0 m left'
    kerb = by['Kerb by the right front wheel']
    fr = p.vehicle.wheel_centres['FR']
    assert kerb['y'] == pytest.approx(-(b['half_w'] + 0.3), abs=1e-3)
    assert kerb['x'] + kerb['length'] / 2 == pytest.approx(fr[0], abs=1e-3)
    light = by['Traffic light at the stop line']
    assert light['base'] == 2.5 and light['kind'] == 'traffic_light'
    again = CR.preset(p, 'Child 2 m ahead', ts)
    assert again['name'] == 'Child 2 m ahead 2'
    assert CR.preset(p, 'nonsense', ts)['kind'] == 'object'


# =========================================================================== the measurements
def test_pinhole_pixels_match_the_closed_form():
    p, c, t = _pinhole_project()
    intr, f = _f_px(p, c)
    res = CR.run(p, Engine(), envelopes=False)
    lk = res.targets[0].looks[0]
    assert lk.in_view == pytest.approx(1.0) and lk.seen == pytest.approx(1.0) and lk.met
    assert lk.px == pytest.approx(f * 1.7 / 10.0, rel=1e-6)                 # a fronto-parallel segment: f h / Z
    assert lk.px_width == pytest.approx(f * 0.5 / 10.0, rel=1e-6)
    assert lk.dist == pytest.approx(math.hypot(10.0, 0.85 - 1.2), rel=1e-9)
    # pitched 28° down the image's top edge (v = 0) cuts the target: what is seen is its bottom up to that edge
    c.pitch = 28.0
    W2O = BODY_TO_OPTICAL @ rot_zyx(c.yaw, c.pitch, c.roll).T
    lk = CR.run(p, Engine(), envelopes=False).targets[0].looks[0]

    def proj(P):
        o = (np.asarray(P, float) - c.pos()) @ W2O.T
        return intr.cx + f * o[0] / o[2], intr.cy + f * o[1] / o[2], o
    best = 0.0
    for yj in (np.arange(5) + 0.5) / 5 * 0.5 - 0.25:              # the five columns across the 0.5 m width
        ub, vb, _ = proj([10.0, yj, 0.0])
        ut, vt, _ = proj([10.0, yj, 1.7])
        assert vb > 0 > vt
        s_ = vb / (vb - vt)                                       # a straight image line: cut where v = 0
        best = max(best, math.hypot(ut - ub, vt - vb) * s_)
    assert lk.px == pytest.approx(best, rel=1e-6)
    # the share seen: the height z* at which the column meets the edge (Y/Z = -cy/f, linear in z)
    _, _, o0 = proj([10.0, 0.0, 0.0])
    _, _, o1 = proj([10.0, 0.0, 1.0])
    k = -intr.cy / f
    z_star = (k * o0[2] - o0[1]) / ((o1[1] - o0[1]) - k * (o1[2] - o0[2]))
    assert lk.seen == pytest.approx(z_star / 1.7, abs=2e-3) and lk.in_view == pytest.approx(lk.seen)
    assert not lk.met and lk.fail == 'seen' or lk.seen >= 0.5


def test_pinhole_envelopes_match_closed_forms():
    """Pitch down until the top leaves the image; down in z until it does; back in x until the pixels run out."""
    D, zc, h = 10.0, 1.2, 1.7
    need_back = 0.5
    p, c, t = _pinhole_project(target={'min_share': 1.0, 'min_px': 0.0}, search_mm=1000.0, search_deg=40.0)
    intr, f = _f_px(p, c)
    st = CR.settings_of(p)
    st.targets[0]['min_px'] = f * h / (D + need_back)                     # exactly met 0.5 m further back
    p.analysis.critical = st.to_dict()
    res = CR.run(p, Engine())
    env = res.targets[0].looks[0].envelope
    half_v = math.degrees(math.atan(intr.cy / f))
    top = math.degrees(math.atan((h - zc) / D))
    bottom = math.degrees(math.atan(zc / D))
    assert env.hi['pitch'] == pytest.approx(half_v - top, abs=0.02)        # pitch down: the top leaves
    assert env.lo['pitch'] == pytest.approx(half_v - bottom, abs=0.02)     # pitch up: the feet leave
    assert env.hi_why['pitch'] == env.lo_why['pitch'] == 'seen'
    for side in ('lo', 'hi'):                    # up or down, the target stays whole beyond the 1 m search
        assert getattr(env, side)['z'] == pytest.approx(1.0) and getattr(env, side + '_why')['z'] == 'open'
    assert env.lo['x'] == pytest.approx(need_back, abs=2e-4) and env.lo_why['x'] == 'pixels'
    assert env.hi_why['x'] == 'open' and env.hi['x'] == pytest.approx(1.0)     # closer: more pixels, no loss
    half_h = math.degrees(math.atan((intr.width / 2) / f))
    edge = math.degrees(math.atan(0.2 / D))                                # the outer columns at ±0.4 × 0.5 m
    assert env.lo['yaw'] == pytest.approx(half_h - edge, abs=0.05) and env.hi['yaw'] == pytest.approx(env.lo['yaw'],
                                                                                                         abs=0.02)
    assert env.joint is not None and 0.0 < env.joint < 1.0 and env.tol_ok is True
    tol = res.install
    assert (tol['mm'], tol['deg']) == (10.0, 1.0)
    # pitched down so the top sits 1.5° inside the image: 1.5° more loses it, and so does lowering the camera by
    # D tan(1.5° + the top's elevation) - (h - zc); all six axes at ± the tolerance (1°, 10 mm) keep it
    st = CR.settings_of(p)
    st.targets[0]['min_px'] = 0.1
    p.analysis.critical = st.to_dict()
    c.pitch = half_v - top - 1.5
    env = CR.run(p, Engine()).targets[0].looks[0].envelope
    assert env.hi['pitch'] == pytest.approx(1.5, abs=0.02) and env.hi_why['pitch'] == 'seen'
    assert env.lo['z'] == pytest.approx(D * math.tan(math.radians(top + 1.5)) - (h - zc), abs=2e-4)
    assert env.lo_why['z'] == 'seen' and env.tol_ok is True and env.joint < 1.0
    # 1° inside: exactly the tolerance, so the grid's corner at +1° pitch with the camera 10 mm lower loses it
    c.pitch = half_v - top - 1.0
    env = CR.run(p, Engine()).targets[0].looks[0].envelope
    assert env.hi['pitch'] == pytest.approx(1.0, abs=0.02) and env.tol_ok is False
    w = env.tol_worst
    assert w['pose'][1] == pytest.approx(1.0) and w['seen'] < 1.0


def test_fisheye_pixels_agree_with_the_perception_range(rig):
    from vantage.core import perception as PR
    p, eng, res = rig
    r = res.target('Pedestrian 10 m ahead')
    lk = next(x for x in r.looks if x.tag == 'F')
    c = p.variant.by_tag('F')
    intr = resolve(p.module_for(c), p.lens_for(c))
    t = r.target
    h, _, seen = PR.object_pixels(c, intr, np.array([[t['x'], t['y']]]),
                                  PR.ObjectClass('Pedestrian', t['height'], t['width'], 24.0))
    assert seen[0] and lk.seen == pytest.approx(1.0)
    # the perception range's chord between the projected bottom and top; ours the projected length (≥ the chord)
    assert h[0] <= lk.px + 1e-6 and lk.px == pytest.approx(h[0], rel=0.02)


def test_line_of_sight_agrees_with_the_cube_maps(rig):
    p, eng, res = rig
    vg = eng.vehicle_geom(p)
    occ = CR.Occluders(vg, p, p.variant)
    agree = total = 0
    hidden_somewhere = False
    for c in p.variant.cameras:
        cube = eng.cube(c.pos(), vg)
        for r in res.targets:
            sh = CR.shape(r.target, c.pos())
            P = sh.flat
            blocked, _ = occ.los(c.pos(), P, c.uid)
            d = P - c.pos()
            dist = np.linalg.norm(d, axis=1)
            depth, _ = cube.lookup(d)
            cube_blocked = ~(depth > dist - (0.02 + 0.01 * dist))
            agree += int((blocked == cube_blocked).sum())
            total += len(P)
            hidden_somewhere |= bool(blocked.any() and cube_blocked.any())
    assert hidden_somewhere and agree / total > 0.97


def test_a_housing_in_the_way_hides_the_target_and_is_named():
    p = _rig_project()
    _with_targets(p, ['Pedestrian 10 m ahead'])
    eng = Engine()
    before = next(x for x in CR.run(p, eng, envelopes=False).targets[0].looks if x.tag == 'F')
    assert before.seen == pytest.approx(1.0) and before.hidden == {}
    f = p.variant.by_tag('F')
    t = CR.settings_of(p).targets[0]
    # a lidar 0.6 m in front of F, on its line of sight to the pedestrian's middle
    d = np.array([t['x'], t['y'], 0.85]) - f.pos()
    q = f.pos() + d / np.linalg.norm(d) * 0.6
    p.variant.sensors.append(Sensor(tag='LX', name='Blocker', kind='lidar', model_id='ouster_os1_128',
                                    x=float(q[0]), y=float(q[1]), z=float(q[2])))
    after = next(x for x in CR.run(p, eng, envelopes=False).targets[0].looks if x.tag == 'F')
    assert after.seen < before.seen - 0.1
    assert any('LX' in k for k in after.hidden) and after.in_view == pytest.approx(before.in_view)


# =========================================================================== envelopes on the sample
def _met_by(p, tag, uid):
    res = CR.run(p, Engine(), envelopes=False)
    return next(x for x in res.target(uid).looks if x.tag == tag).met


def test_reported_limits_hold_when_the_camera_is_moved_there(rig):
    """Injected install errors just inside a reported limit keep the target; just outside lose it."""
    p, eng, res = rig
    r = res.target('Child 2 m ahead')
    lk = next(x for x in r.looks if x.tag == 'F')
    env = lk.envelope
    checked = 0
    for ax, side in (('pitch', 1), ('pitch', -1), ('yaw', 1), ('z', 1), ('y', -1), ('x', 1)):
        lim, why = env.limit(ax, side), env.why(ax, side)
        if why in ('open', 'surface'):
            continue
        k = CR.AXES.index(ax)
        eps = 0.05 if k < 3 else 0.002
        for off, expect in ((lim - eps, True), (lim + eps, False)):
            q = copy.deepcopy(p)
            c = q.variant.by_tag('F')
            if k < 3:
                setattr(c, ax, getattr(c, ax) + side * off)
            else:
                setattr(c, ax, getattr(c, ax) + side * off)
            assert _met_by(q, 'F', r.target['uid']) is expect, (ax, side, lim, off)
        checked += 1
    for c in res.cameras:                       # every requirement-limited side of every camera's envelope
        for ax in c.lo:
            for side, (lim, name, why) in ((-1, c.lo[ax]), (1, c.hi[ax])):
                if why in ('open', 'surface') or checked > 8:
                    continue
                k = CR.AXES.index(ax)
                eps = 0.05 if k < 3 else 0.002
                tgt = res.target(name)
                q = copy.deepcopy(p)
                cam = q.variant.by_tag(c.tag)
                setattr(cam, ax, getattr(cam, ax) + side * (lim + eps))
                assert not _met_by(q, c.tag, tgt.target['uid'])
                checked += 1
    assert checked >= 2


def test_surface_limits_are_the_distance_to_the_body(rig):
    p, eng, res = rig
    vg = eng.vehicle_geom(p)
    lk = next(x for x in res.target('Child 2 m ahead').looks if x.tag == 'F')
    env = lk.envelope
    assert env.lo_why['x'] == 'surface'                      # backwards into the front bumper
    f = p.variant.by_tag('F')
    t, face = vg.occ_mesh.raycast(f.pos()[None], np.array([[-1.0, 0.0, 0.0]]))
    assert face[0] >= 0 and env.lo['x'] == pytest.approx(float(t[0]), abs=1e-3)
    c = res.camera('F')
    assert c.lo['x'][2] in ('surface', 'open', 'pixels', 'seen', 'focus')


def test_joint_share_and_tolerance_grid_are_consistent(rig):
    p, eng, res = rig
    tol = res.install
    for r in res.targets:
        for lk in r.looks:
            env = lk.envelope
            if env is None:
                assert not lk.met
                continue
            assert lk.met and 0.0 <= env.joint <= 1.0
            single_tight = any(env.limit(ax, s) < (tol['deg'] if ax in CR.ANGLE_AXES else tol['mm'] / 1000) - 1e-9
                               and env.why(ax, s) not in ('surface', 'open') for ax in CR.AXES for s in (-1, 1))
            if single_tight:
                assert env.tol_ok is False
            w = env.tol_worst
            assert set(w) == {'px', 'seen', 'pose'} and len(w['pose']) == 6
    # an independent check of one joint grid corner at the reported share: the camera moved on all six axes
    r = res.target('Child 2 m ahead')
    lk = next(x for x in r.looks if x.tag == 'F')
    env = lk.envelope
    k = env.joint * 0.98
    q = copy.deepcopy(p)
    c = q.variant.by_tag('F')
    for ax in CR.AXES:
        lim = env.hi[ax] * (CR.SURFACE_KEEP if env.hi_why[ax] == 'surface' else 1.0)
        setattr(c, ax, getattr(c, ax) + k * lim)
    assert _met_by(q, 'F', r.target['uid'])


def test_camera_envelopes_binding_target_and_findings(rig):
    p, eng, res = rig
    f = res.camera('F')
    assert 'Child 2 m ahead' in f.targets and 'Pedestrian 10 m ahead' in f.targets
    for ax in CR.AXES:                                   # the camera's envelope is the tightest of its targets'
        for side, (lim, name, why) in ((-1, f.lo[ax]), (1, f.hi[ax])):
            per = [x.envelope.limit(ax, side) for r in res.targets for x in r.looks
                   if x.tag == 'F' and x.envelope is not None and x.envelope.why(ax, side) not in ('surface', 'open')]
            if per:
                assert lim == pytest.approx(min(per))
    assert f.binding in f.targets and f.margin is not None and f.margin > 0
    light = res.target('Overhead traffic light 15 m ahead')
    assert not light.met and light.met_by == []
    msgs = [x for x in res.findings if x['kind'] == 'unresolved']
    assert any('Overhead traffic light' in x['message'] and x['severity'] == 'serious' for x in msgs)
    assert res.findings[0]['severity'] in ('critical', 'serious')
    order = [x['severity'] for x in res.findings]
    rank = {'critical': 0, 'serious': 1, 'warning': 2, 'good': 3}
    assert order == sorted(order, key=rank.get)


def test_a_tight_mount_is_flagged_when_only_that_camera_resolves_the_target():
    """A pinhole camera that only just sees a target's top: its pitch envelope is tighter than 1°."""
    p, c, t = _pinhole_project(target={'min_share': 1.0, 'min_px': 10.0})
    intr, f = _f_px(p, c)
    half_v = math.degrees(math.atan(intr.cy / f))
    c.pitch = half_v - math.degrees(math.atan(0.5 / 10.0)) - 0.4            # the top 0.4° inside the image edge
    res = CR.run(p, Engine())
    cam = res.camera('P')
    assert cam.sole == ['Ped'] and any(ax == 'pitch' and side == 1 and sole for ax, side, _, _, sole in cam.tight)
    tight = [x for x in res.findings if x['kind'] == 'tight']
    assert tight and 'P alone resolves "Ped"' in tight[0]['message'] and 'pitch +0.4' in tight[0]['message']
    iss = CR.issues(p, None)                         # the cached full run: the Issues list shows it too
    assert any(i.severity == 'warning' and 'alone resolves' in i.message for i in iss)
    assert any(i.uids == [c.uid] for i in iss)


# =========================================================================== lidar and radar
def test_lidar_points_match_the_coverage_study(rig):
    from vantage.core import sensor_coverage as SC
    p, eng, res = rig
    r = res.target('Pedestrian 10 m ahead')
    t = r.target
    L1 = next(s for s in p.variant.sensors if s.tag == 'L1')
    cls = SC.CoverageClass('Pedestrian', t['height'], t['width'], t['base'])
    n = SC.object_points(p, eng, L1, cls, np.array([[t['x'], t['y']]]))[0]
    assert n > 10 and r.lidar['per']['L1'] == pytest.approx(n) and r.lidar['points'] == pytest.approx(n)
    # the lane marking: every unblocked ground point of the lidar inside the strip, counted without the bounding box
    lane = res.target('Lane marking on a 50 m curve')
    Lc = SC._Lidar(L1, p.sensor_model_for(L1), eng.cube(L1.pos(), eng.vehicle_geom(p)))
    P, ok = Lc.ground()
    G = P[ok][:, :2]
    tl = lane.target
    s = np.linspace(0, tl['length'], 4001)
    C, _ = CR._arc(tl, s)
    from scipy.spatial import cKDTree
    dmin, _ = cKDTree(C).query(G)
    inside = (dmin <= tl['width'] / 2)
    assert lane.lidar['points'] == pytest.approx(inside.sum() * Lc.weight, rel=0.05, abs=Lc.weight * 2)
    assert r.radar is not None and 'RD1' in r.radar['by']


def test_a_lidar_requirement_counts_toward_resolved():
    p = _rig_project()
    ts = _with_targets(p, ['Child 2 m ahead', 'Car 30 m ahead'])
    ts[0]['min_points'] = 8.0                                   # the roof lidar cannot see a child at the bumper
    ts[1]['min_points'] = 5.0
    p.analysis.critical = {'targets': ts}
    res = CR.run(p, Engine(), envelopes=False)
    child, car = res.targets
    assert child.cameras_ok and not child.lidar_ok and not child.met
    assert car.met and car.lidar['points'] >= 5.0
    lid = [x for x in res.findings if x['kind'] == 'lidar']
    assert len(lid) == 1 and 'Child 2 m ahead' in lid[0]['message'] and lid[0]['severity'] == 'warning'
    p.variant.sensors = [s for s in p.variant.sensors if s.kind != 'lidar']
    res = CR.run(p, Engine(), envelopes=False)
    assert res.targets[0].lidar.get('no_lidar') and 'No lidar' in [x for x in res.findings
                                                                    if x['kind'] == 'lidar'][0]['message']


# =========================================================================== edge cases
def test_hidden_cameras_empty_variants_nan_poses_and_cancel():
    p = _rig_project()
    _with_targets(p, ['Child 2 m ahead'])
    p.variant.by_tag('F').visible = False
    p.variant.by_tag('ML').x = float('nan')
    res = CR.run(p, Engine(), envelopes=False)
    tags = [x.tag for x in res.targets[0].looks]
    assert 'F' not in tags and 'ML' not in tags and len(tags) == 4
    p.variant.cameras = []
    res = CR.run(p, Engine())
    assert res.targets[0].looks == [] and not res.targets[0].met
    assert 'no camera' in res.findings[0]['message']
    q = _rig_project()
    assert CR.run(q, Engine()).targets == [] and CR.issues(q) == []
    _with_targets(q, ['Child 2 m ahead', 'Pedestrian 10 m ahead'])
    calls = []
    assert CR.run(q, Engine(), cancel=lambda: len(calls) > 0, progress=calls.append) is None
    four = next(v for v in q.variants if v.name == '4-cam')               # another variant than the active one
    r4 = CR.run(q, Engine(), variant=four, envelopes=False)
    assert r4.variant == '4-cam' and [x.tag for x in r4.targets[0].looks] == ['F', 'R', 'ML', 'MR']
    assert q.active_uid != four.uid and CR.plan_png(r4, q)[:4] == b'\x89PNG'
    off = CR.settings_of(q)
    off.targets[0]['enabled'] = False
    q.analysis.critical = off.to_dict()
    assert [r.name for r in CR.run(q, Engine(), envelopes=False).targets] == ['Pedestrian 10 m ahead']


def test_issues_are_cached_and_cheap(rig):
    import time
    p, eng, res = rig
    q = copy.deepcopy(p)
    CR.clear_cache()
    vg = Engine().vehicle_geom(q)
    iss = CR.issues(q, vg)
    assert [i.severity for i in iss].count('serious') >= 1
    assert any('Overhead traffic light' in i.message for i in iss)
    light = next(i for i in iss if 'Overhead traffic light' in i.message)
    t = CR.settings_of(q).target(next(x['uid'] for x in CR.settings_of(q).targets
                                      if x['name'].startswith('Overhead')))
    assert light.where == (t['x'], t['y']) and 'critical target' in light.objects
    t0 = time.perf_counter()
    CR.issues(q, vg)
    assert time.perf_counter() - t0 < 0.01
    q.variant.by_tag('R').pitch += 1.0                     # one camera moved: its looks only are measured again
    t0 = time.perf_counter()
    CR.issues(q, vg)
    assert time.perf_counter() - t0 < 0.5


# =========================================================================== outputs
def test_exports_summary_and_sheets(rig, tmp_path):
    import cv2
    p, eng, res = rig
    paths = CR.export(res, p, str(tmp_path))
    names = sorted(os.path.basename(x) for x in paths)
    assert names == ['critical_targets.csv', 'critical_targets.json', 'critical_targets.png',
                     'mounting_envelopes.csv', 'mounting_envelopes.png']
    doc = json.loads((tmp_path / 'critical_targets.json').read_text())
    assert doc['schema'] == CR.SCHEMA and doc['frame'].startswith('vehicle')
    assert 'mm' in doc['units']['envelope_positions']
    assert len(doc['targets']) == 6 and doc['install_tolerance']['mm'] == 10.0
    ped = next(t for t in doc['targets'] if t['name'] == 'Pedestrian 10 m ahead')
    fcam = next(c for c in ped['cameras'] if c['camera'] == 'F')
    env = res.target('Pedestrian 10 m ahead').looks[0].envelope
    assert fcam['met'] and fcam['envelope']['plus']['pitch'] == pytest.approx(round(env.hi['pitch'], 3))
    assert fcam['envelope']['minus']['x'] == pytest.approx(round(env.lo['x'] * 1000, 2))
    rows = (tmp_path / 'critical_targets.csv').read_text().strip().splitlines()
    assert len(rows) == 1 + 6 * len(p.variant.cameras) and rows[0].startswith('target,kind')
    sheet = (tmp_path / 'mounting_envelopes.csv').read_text().strip().splitlines()
    n_cams = sum(1 for c in res.cameras if c.targets)
    assert len(sheet) == 1 + 6 * n_cams and sheet[1].split(',')[1] == 'yaw'
    for name in ('mounting_envelopes.png', 'critical_targets.png'):
        img = cv2.imdecode(np.frombuffer((tmp_path / name).read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        assert img is not None and img.shape[0] > 200 and img.shape[1] > 200
    sm = CR.summary(res)
    json.dumps(sm)
    assert sm['targets'] == 6 and sm['resolved'] == sum(r.met for r in res.targets)
    assert {x['camera'] for x in sm['per_camera']} == {c.tag for c in res.cameras if c.targets}


# =========================================================================== the optimiser
def _optimiser_project():
    """The sample's 4-camera variant with a target only F can see, and only pitched up from its 40°."""
    p = samples.tigor_sample_project()
    v = next(v for v in p.variants if v.name == '4-cam')
    p.active_uid = v.uid
    b = CR.vehicle_box(p)
    t = CR.clean_target({'kind': 'sign', 'name': 'High sign', 'x': b['front_x'] + 6.0, 'y': 0.0, 'base': 1.6,
                         'height': 0.5, 'width': 0.5, 'min_px': 12.0, 'min_share': 0.8})
    p.analysis.critical = {'targets': [t]}
    return p, t


def test_target_check_agrees_with_the_study():
    p, t = _optimiser_project()
    res = CR.run(p, Engine(), envelopes=False)
    tc = CR.TargetCheck(p, Engine().vehicle_geom(p))
    assert bool(tc.met(p.variant.cameras)[0]) == res.targets[0].met
    f = p.variant.by_tag('F')
    met = []
    for pitch in (40.0, 25.0):
        q = copy.deepcopy(p)
        q.variant.by_tag('F').pitch = pitch
        met.append(CR.run(q, Engine(), envelopes=False).targets[0].met)
        R = rot_zyx(f.yaw, pitch, f.roll)
        assert bool(tc.met(p.variant.cameras, {f.uid: (R, f.pos())})[0]) == met[-1]
    assert met == [False, True]


def test_optimiser_requires_and_weighs_critical_targets():
    from vantage.core.optimiser import OptSettings, Optimiser
    p, t = _optimiser_project()
    base = dict(generations=6, population=12, under_tolerance=False, tol_samples=0, cell=0.2, seed=3)
    o = Optimiser(Engine(), p, OptSettings(critical='require', **base))
    o.run()
    assert o.crit is not None and o.front
    assert all(c.crit_unmet == 0 and o.feasible(c) for c in o.front)
    cands = o.candidates(3)
    assert cands and all(o.targets_text(c) == '1/1' for c in cands)
    for c in cands[:2]:                           # independently: the candidate as a variant resolves the target
        v = o.to_variant(c, f'cand {c.label}')
        q = copy.deepcopy(p)
        q.variants.append(v)
        q.active_uid = v.uid
        assert CR.run(q, Engine(), envelopes=False).targets[0].met
        assert 'critical targets resolved 1/1' in v.note
    nom = o.nominal_cand
    assert nom.crit_unmet == 1 and not o.feasible(nom) and nom not in o.front
    assert 'misses High sign' in o.changes(nom)
    # weighed: an unresolved target costs its weight in the objective
    w = Optimiser(Engine(), p, OptSettings(critical='weigh', critical_weight=2.5, **base))
    cand = w.evaluate(w.nominal())
    assert w.objectives(cand)[0] == pytest.approx(cand.worst + 2.5)
    # off: no checks, objectives as before
    off = Optimiser(Engine(), p, OptSettings(**base))
    cand = off.evaluate(off.nominal())
    assert off.crit is None and off.objectives(cand) == (cand.worst, -cand.min_density, cand.cams)
    assert off.targets_text(cand) == '' and cand.crit_unmet == 0


def test_optimiser_under_tolerance_counts_targets_lost_in_a_sample():
    from vantage.core.optimiser import OptSettings, Optimiser
    p, t = _optimiser_project()
    q = copy.deepcopy(p)
    q.variant.by_tag('F').pitch = 33.0                     # resolved, but only just
    o = Optimiser(Engine(), q, OptSettings(critical='require', under_tolerance=True, tol_samples=24, tol_deg=8.0,
                                           cell=0.2, generations=1, population=6))
    cand = o.evaluate(o.nominal())
    assert cand.crit_unmet == 0 and cand.crit_tol == 1 and o.crit_penalty(cand) == 1 and not o.feasible(cand)
    assert 'loses 1 critical target under tolerance' in o.changes(cand)


# =========================================================================== the page
from test_gui import app, win, pump, settle, _wait   # noqa: E402,F401  (fixtures)


def _open(win, name, cls):
    if name not in win._page_widgets:
        pg = cls(win.ctx)
        win.pages.addWidget(pg)
        win._page_widgets[name] = pg
    win.set_page(name)
    return win._page_widgets[name]


def _type_into(app, w, text):
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    le = w.lineEdit() if hasattr(w, 'lineEdit') else w
    QTest.mouseClick(le, Qt.MouseButton.LeftButton)
    QTest.keyClick(le, Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
    QTest.keyClicks(le, text)
    QTest.keyClick(le, Qt.Key.Key_Return)
    pump(app, 0.1)


def test_page_adds_runs_edits_drags_undoes_and_exports(app, win, tmp_path):
    from PyQt6.QtCore import QPoint, Qt
    from PyQt6.QtTest import QTest
    from vantage.ui.workspaces.critical_ws import CriticalPage, scene_tris, scene_lines, OVERLAY
    store = win.store
    rig = next(v for v in store.project.variants if v.sensors)
    store.edit('variants', lambda pj: setattr(pj, 'active_uid', rig.uid), f'Switched to {rig.name}')
    pump(app, 0.2)
    pg = _open(win, 'critical', CriticalPage)
    ctl = win.ctx.critical
    assert pg.isVisible() and not pg.b_run.isEnabled() and 'Add a critical target' in pg.status.text()
    n0 = store.undo.index()
    acts = {a.text(): a for a in pg.add_menu.actions()}
    acts['Child 2 m ahead'].trigger()
    acts['Overhead traffic light 15 m ahead'].trigger()
    pump(app, 0.1)
    st = CR.settings_of(store.project)
    assert [t['name'] for t in st.targets] == ['Child 2 m ahead', 'Overhead traffic light 15 m ahead']
    assert store.undo.index() == n0 + 2 and ctl.sel == st.targets[1]['uid'] and pg.targets.rowCount() == 2
    _wait(app, lambda: ctl.result is not None and not ctl.busy and not ctl.stale, 90)    # runs by itself
    pump(app, 0.2)
    res = ctl.result
    assert pg.targets.item(0, 3).text() == 'F' and pg.targets.item(1, 3).text() == 'unresolved'
    assert '1 of 2 targets resolved' in pg.status.text()
    assert any('Overhead traffic light' in w.text.text() for w in pg.findings)
    # pick the child: its cameras, F's envelope
    QTest.mouseClick(pg.targets.viewport(), Qt.MouseButton.LeftButton,
                     pos=pg.targets.visualItemRect(pg.targets.item(0, 1)).center())
    pump(app, 0.1)
    assert ctl.sel == st.targets[0]['uid'] and pg.look_head.text() == 'Cameras on Child 2 m ahead'
    assert pg.looks.rowCount() == 6
    f_row = next(i for i in range(6) if pg.looks.item(i, 0).text() == 'F')
    assert pg.looks.item(f_row, 6).text() == 'meets'
    QTest.mouseClick(pg.looks.viewport(), Qt.MouseButton.LeftButton,
                     pos=pg.looks.visualItemRect(pg.looks.item(f_row, 0)).center())
    pump(app, 0.1)
    rows, title, note = pg.envelope_rows()
    assert title == 'F · Child 2 m ahead' and [r[0] for r in rows] == list(CR.AXES)
    env = next(x for x in res.target('Child 2 m ahead').looks if x.tag == 'F').envelope
    assert rows[1][2][0] == pytest.approx(env.hi['pitch'])
    pg.scope_ctl.buttons['all'].click()
    pump(app, 0.05)
    assert pg.envelope_rows()[1] == 'F · every target it resolves'
    pg.bars.grab()
    for fx in (0.3, 0.5, 0.7):
        QTest.mouseMove(pg.bars, QPoint(int(pg.bars.width() * fx), pg.bars.TOP + pg.bars.ROW // 2))
        pump(app, 0.02)
    # type a new distance: 3 m ahead; one undo step, the study runs again, undo puts it back
    b = CR.vehicle_box(store.project)
    n1 = store.undo.index()
    _type_into(app, pg.fields['x'][0], f'{b["front_x"] + 3.0:.3f}')
    t0 = CR.settings_of(store.project).targets[0]
    assert t0['x'] == pytest.approx(b['front_x'] + 3.0, abs=1e-3) and store.undo.index() == n1 + 1
    assert pg.place_lbl.text() == '3.0 m ahead of the front bumper.'
    _wait(app, lambda: not ctl.busy and not ctl.stale and ctl.result is not res, 90)
    store.undo.undo()
    pump(app, 0.1)
    assert CR.settings_of(store.project).targets[0]['x'] == pytest.approx(b['front_x'] + 2.0, abs=1e-3)
    # a value out of range is clamped and shown as stored
    _type_into(app, pg.fields['min_share'][0], '250')
    assert CR.settings_of(store.project).targets[0]['min_share'] == 1.0
    assert pg.fields['min_share'][0].value() == pytest.approx(100.0)
    store.undo.undo()
    pump(app, 0.05)
    # untick the traffic light: left out of the study
    it = pg.targets.item(1, 0)
    it.setCheckState(Qt.CheckState.Unchecked)
    pump(app, 0.05)
    assert CR.settings_of(store.project).targets[1]['enabled'] is False
    store.undo.undo()
    pump(app, 0.05)
    assert CR.settings_of(store.project).targets[1]['enabled'] is True
    # drag the child 1 m to the left in the plan: one undo step however many moves
    _wait(app, lambda: not ctl.busy, 90)
    plan = pg.plan
    plan.fit_targets()
    pump(app, 0.05)
    t0 = CR.settings_of(store.project).targets[0]
    a = plan.w2s(t0['x'], t0['y'])
    bpt = plan.w2s(t0['x'], t0['y'] + 1.0)
    n2 = store.undo.index()
    QTest.mousePress(plan, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, a.toPoint())
    for k in range(1, 6):
        QTest.mouseMove(plan, (a + (bpt - a) * (k / 5)).toPoint())
        pump(app, 0.01)
    QTest.mouseRelease(plan, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, bpt.toPoint())
    pump(app, 0.05)
    t1 = CR.settings_of(store.project).targets[0]
    assert t1['y'] == pytest.approx(t0['y'] + 1.0, abs=0.05) and store.undo.index() == n2 + 1
    store.undo.undo()
    pump(app, 0.05)
    assert CR.settings_of(store.project).targets[0]['y'] == pytest.approx(t0['y'])
    # place: click where the selected target goes
    pg.b_place.click()
    assert ctl.placing
    goal = plan.w2s(t0['x'] + 1.0, -1.0)
    QTest.mouseClick(plan, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, goal.toPoint())
    pump(app, 0.05)
    t2 = CR.settings_of(store.project).targets[0]
    assert not ctl.placing and t2['x'] == pytest.approx(t0['x'] + 1.0, abs=0.05) and t2['y'] == pytest.approx(-1.0,
                                                                                                          abs=0.05)
    store.undo.undo()
    pump(app, 0.05)
    # the 3D view: the targets are drawn; a click on the child picks it
    pg.view_mode.buttons['3d'].click()
    pump(app, 0.2)
    assert pg.stack.currentWidget() is pg.view3d
    gl = getattr(pg.view3d, 'renderer', None) is not None
    tris = scene_tris(win.ctx, gl, ctl, force=True)
    assert tris and len(tris[0].tri) > 12
    # both renderers: one colour and one normal per face (the shader path broke on every frame: one colour row)
    for g in (False, True):
        d = scene_tris(win.ctx, g, ctl, force=True)[0]
        assert len(d.colors) == len(d.tri) and (d.normals is None) != g and (not g or len(d.normals) == len(d.tri))
    _wait(app, lambda: not ctl.busy and not ctl.stale, 90)
    assert scene_lines(win.ctx, ctl, force=True)
    assert scene_tris(win.ctx, gl) == [] or win.ctx.overlays.get(OVERLAY)       # main views: only with the overlay
    pg.view3d.grab()
    ctl.select(CR.settings_of(store.project).targets[1]['uid'])
    t0 = CR.settings_of(store.project).targets[0]
    s, _, ok = pg.view3d.project(np.array([[t0['x'], t0['y'], t0['height'] / 2]]))
    assert ok[0]
    QTest.mouseClick(pg.view3d, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                     QPoint(int(s[0][0]), int(s[0][1])))
    pump(app, 0.05)
    assert ctl.sel == t0['uid']
    pg.view_mode.buttons['plan'].click()
    pump(app, 0.05)
    # the main views' hooks (View ▸ Overlays ▸ Critical targets): the targets in the main plan and 3D scene, with
    # the verdicts of the newest result for this design even when it is not the page's
    from PyQt6.QtGui import QImage, QPainter
    from vantage.ui.workspaces.critical_ws import paint_plan, _result
    main = win.viewarea.panes[0].plan
    img = QImage(400, 300, QImage.Format.Format_ARGB32)
    img.fill(0)
    qp = QPainter(img)
    paint_plan(main, qp)                                 # overlay off: nothing
    qp.end()
    assert img.pixel(200, 150) == 0 and not win.ctx.overlays.get(OVERLAY)
    main.resize(400, 300)
    main.fit_targets = lambda: None
    qp = QPainter(img)
    paint_plan(main, qp, force=True)
    qp.end()
    assert _result(win.ctx, ctl) is ctl.result
    # export
    paths = pg.export(str(tmp_path))
    assert len(paths) == 5 and (tmp_path / 'mounting_envelopes.csv').exists()
    # remove the traffic light; undo brings it back
    ctl.select(CR.settings_of(store.project).targets[1]['uid'])
    pg.b_remove.click()
    pump(app, 0.05)
    assert [t['name'] for t in CR.settings_of(store.project).targets] == ['Child 2 m ahead']
    store.undo.undo()
    pump(app, 0.05)
    assert len(CR.settings_of(store.project).targets) == 2
    # the install tolerance: typed, then 0 goes back to the project's
    _type_into(app, pg.tol_deg, '0.25')
    assert CR.settings_of(store.project).install_deg == 0.25 and '±0.25°' in pg.tol_lbl.text()
    _type_into(app, pg.tol_deg, '0')
    assert '±1°' in pg.tol_lbl.text() and 'project' in pg.tol_lbl.text()
    # the Issues list's function (rules.collect_issues calls it): the traffic light no camera resolves
    _wait(app, lambda: not ctl.busy, 90)
    vg = win.ctx.analysis.engine.vehicle_geom(store.project)
    assert any('Overhead traffic light' in i.message and i.severity == 'serious'
               for i in CR.issues(store.project, vg))
    # a new project: the result and the selection go
    win.store.set_project(samples.tigor_sample_project())
    pump(app, 0.2)
    assert ctl.result is None and ctl.sel == '' and pg.targets.rowCount() == 0
    pg.grab()


def test_optimiser_page_requires_targets_and_adds_the_best_three(app, win):
    from vantage.ui.workspaces.optimiser_ws import OptimiserPage
    store = win.store
    p, t = _optimiser_project()
    v = next(v for v in store.project.variants if v.name == '4-cam')

    def fn(pj):
        pj.active_uid = v.uid
        pj.analysis.critical = {'targets': [t]}
    store.edit(['variants', 'analysis'], fn, 'critical target for the optimiser')
    pump(app, 0.1)
    win.set_page('optimiser')
    pg = win.page('optimiser')
    pg.activate()
    assert pg.crit.isEnabled() and '1 target' in pg.crit_note.text()
    pg.crit.setCurrentIndex(pg.crit.findData('require'))
    assert not pg.crit_w.isEnabled()
    pg.under_tol.setChecked(False)
    pg.gens.set_value(3)
    pg.pop.set_value(8)
    st = pg.settings()
    assert st.critical == 'require' and st.generations == 3
    pg.toggle()
    _wait(app, lambda: 'optimiser' not in win._jobs, 180)
    pump(app, 0.3)
    assert not pg.table.isColumnHidden(6) and pg.cands
    assert all(pg.table.item(i, 6).text() == '1/1' for i in range(pg.table.rowCount()))
    n_var = len(store.project.variants)
    k = min(3, len(pg.cands))
    assert pg.b_three.isEnabled()
    pg.b_three.click()
    pump(app, 0.1)
    assert len(store.project.variants) == n_var + k and store.project.active_uid == v.uid
    names = [x.name for x in store.project.variants[-k:]]
    assert names[0] == f'{v.name} · C1' and all('critical targets resolved 1/1' in x.note
                                                 for x in store.project.variants[-k:])
    store.undo.undo()
    pump(app, 0.05)
    assert len(store.project.variants) == n_var
    pg.crit.setCurrentIndex(pg.crit.findData('weigh'))
    assert pg.crit_w.isEnabled() and pg.settings().critical == 'weigh'
