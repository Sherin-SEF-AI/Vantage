"""Cable harness (core.harness, ui.workspaces.harness_ws) and the deserializer rule of the timing plan.

Independent checks: shortest paths on surfaces with known geodesics (a plane, a box unfolded across an edge), glass
and moving parts avoided by a computable detour, run lengths rebuilt from their parts by hand (Manhattan inside ×
slack, service loop, extra), the RF budget and the power drop recomputed from their formulas, link limits crossed by
moving the compute box, entry points switched off or over capacity, and the routes on the sample's data rig bounded
below by the straight distance; the deserializer rule on stereo pairs built for it and its fixes applied and
re-checked."""
from __future__ import annotations

import csv
import json
import math

import numpy as np
import pytest

from vantage.core import harness as HN
from vantage.core import samples, sensors, timing as TM
from vantage.core.mesh import Mesh, Part, Zone, ROLE_GLASS, ROLE_MOUNTABLE
from vantage.core.project import Camera, Project, Sensor, Variant


# ----------------------------------------------------------------------------- synthetic bodies
def box_mesh(lo, hi, n=(40, 20, 12), glass=None, zone=None):
    """A closed axis-aligned box as a grid of quads (two triangles each), outward faces. glass / zone: a predicate
    f(centroid) marking faces as glass (part 1) or in mount zone 0 ('Door shut lines')."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    V, F = [], []

    def face(origin, u, v, nu, nv, flip):
        base = len(V)
        for j in range(nv + 1):
            for i in range(nu + 1):
                V.append(origin + u * i / nu + v * j / nv)
        for j in range(nv):
            for i in range(nu):
                a = base + j * (nu + 1) + i
                b, c, d = a + 1, a + nu + 2, a + nu + 1
                F.extend([[a, c, b], [a, d, c]] if flip else [[a, b, c], [a, c, d]])
    dx, dy, dz = hi - lo
    ex, ey, ez = np.array([dx, 0, 0]), np.array([0, dy, 0]), np.array([0, 0, dz])
    nx, ny, nz = n
    face(np.array([lo[0], lo[1], hi[2]]), ex, ey, nx, ny, False)        # top (+z)
    face(lo, ex, ey, nx, ny, True)                                       # bottom (-z)
    face(np.array([hi[0], lo[1], lo[2]]), ey, ez, ny, nz, False)        # front (+x)
    face(lo, ey, ez, ny, nz, True)                                       # rear (-x)
    face(np.array([lo[0], hi[1], lo[2]]), ex, ez, nx, nz, True)         # left (+y)
    face(lo, ex, ez, nx, nz, False)                                      # right (-y)
    V, F = np.array(V, float), np.array(F, np.int64)
    c = V[F].mean(axis=1)
    fp = np.zeros(len(F), np.int32)
    parts = [Part('Body', 'paint', role=ROLE_MOUNTABLE)]
    if glass is not None:
        parts.append(Part('Glass', 'glass', role=ROLE_GLASS))
        fp[np.array([bool(glass(q)) for q in c])] = 1
    fz = np.full(len(F), -1, np.int16)
    zones = []
    if zone is not None:
        zones = [Zone('Door shut lines', 'forbidden')]
        fz[np.array([bool(zone(q)) for q in c])] = 0
    m = Mesh(V, F, fp, parts, fz, zones)
    n_ = m.normals                                  # outward check: the face normals point away from the centre
    assert (np.einsum('ij,ij->i', n_, c - (lo + hi) / 2) > 0).all()
    return m


BOX_LO, BOX_HI = (-0.5, -0.9, 0.3), (3.5, 0.9, 1.5)


def box_project(mesh=None, harness=None, ecu=(1.0, 0.0, 0.6)):
    p = Project()
    p.vehicle.mesh = mesh if mesh is not None else box_mesh(BOX_LO, BOX_HI)
    v = Variant(name='Box rig')
    p.variants = [v]
    p.active_uid = v.uid
    p.ecu.x, p.ecu.y, p.ecu.z = ecu
    p.analysis.harness = dict(harness or {})
    return p


def cam(tag, x, y, z, port=1, cable=15.0, **kw):
    c = Camera(tag=tag, name=tag, x=x, y=y, z=z, port=port, cable_m=cable)
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def sensor(p, kind, model_id, tag, x, y, z, **kw):
    s = Sensor(kind=kind, model_id=model_id, tag=tag, x=x, y=y, z=z)
    for k, v in kw.items():
        setattr(s, k, v)
    p.variant.sensors.append(s)
    return s


@pytest.fixture(autouse=True)
def _fresh_cache():
    HN.clear_cache()
    yield
    HN.clear_cache()


def _graph(mesh, **st):
    p = box_project(mesh, st)
    return HN.route_graph(p, HN.settings_of(p))


def _route(g, a, b):
    """The surface route between the vertices nearest a and b: (what it crosses with the pulled length, the two
    vertices, the pulled path)."""
    va, _ = g.nearest(a)
    vb, _ = g.nearest(b)
    path = g.path(va, vb)
    m = g.measure(path)
    pulled = g.pull(path)
    m['length'] = float(np.linalg.norm(np.diff(g.V[pulled], axis=0), axis=1).sum())
    m['found'] = g.measure(path)['length']
    return m, g.V[va], g.V[vb], pulled


# ----------------------------------------------------------------------------- shortest paths on known surfaces
def test_path_on_a_plane_is_the_straight_line():
    g = _graph(box_mesh(BOX_LO, BOX_HI))
    assert g.stats['shells'] == 1 and g.components == 1
    m, a, b, _ = _route(g, (0.2, -0.6, 1.5), (3.1, 0.5, 1.5))
    straight = float(np.linalg.norm(b - a))
    assert straight > 3.0
    assert straight - 1e-9 <= m['length'] <= straight * 1.005     # pulled straight on a plane
    assert m['found'] > m['length']                              # the mesh path zig-zags before it is pulled
    assert m['glass'] == 0 and m['moving'] == 0 and m['free'] == 0


def test_path_over_an_edge_is_the_unfolded_geodesic():
    g = _graph(box_mesh(BOX_LO, BOX_HI))
    # from the top to the front face: unfolded about the edge x = 3.5, z = 1.5 the path is straight
    m, a, b, _ = _route(g, (1.0, -0.3, 1.5), (3.5, 0.4, 0.6))
    unfolded = math.hypot((3.5 - a[0]) + (1.5 - b[2]), b[1] - a[1])
    assert unfolded - 1e-9 <= m['length'] <= unfolded * 1.015
    # from the top to the left side, across the long edge y = 0.9
    m, a, b, path = _route(g, (0.5, -0.5, 1.5), (2.8, 0.9, 0.7))
    unfolded = math.hypot((0.9 - a[1]) + (1.5 - b[2]), b[0] - a[0])
    assert unfolded - 1e-9 <= m['length'] <= unfolded * 1.015
    # every straight piece lies on the box's surface (no corner cut through it)
    mesh = box_mesh(BOX_LO, BOX_HI)
    for i in range(len(path) - 1):
        for t in np.linspace(0, 1, 9):
            q = g.V[path[i]] * (1 - t) + g.V[path[i + 1]] * t
            assert mesh.closest_point(q)[2] < 1e-6


def test_glass_is_avoided_by_the_detour_round_it():
    patch = lambda c: c[2] > 1.49 and 1.5 < c[0] < 2.5 and abs(c[1]) < 0.5        # noqa: E731
    mesh = box_mesh(BOX_LO, BOX_HI, glass=patch)
    a, b = (1.0, 0.0, 1.5), (3.0, 0.0, 1.5)
    m1, va, vb, _ = _route(_graph(mesh, glass_cost=1.0), a, b)          # glass costs as much as paint: straight over
    assert m1['glass'] > 0.9 and m1['length'] == pytest.approx(2.0, rel=0.02)
    HN.clear_cache()
    m25, _, _, path = _route(_graph(mesh, glass_cost=25.0), a, b)
    assert m25['glass'] == 0.0                                           # round the patch, not over it
    detour = 2 * math.hypot(0.5, 0.5) + 1.0                              # A → patch corner → corner → B
    assert detour <= m25['length'] <= detour * 1.15
    # the costs differ, the cache keeps one graph per cost set
    assert HN.route_graph(box_project(mesh, {'glass_cost': 1.0})) is not HN.route_graph(box_project(mesh))
    assert HN.route_graph(box_project(mesh, {'glass_cost': 25.0})) is HN.route_graph(box_project(mesh))


def test_shut_lines_cost_more_and_are_reported():
    band = lambda c: 1.95 < c[0] < 2.05                                 # noqa: E731  a shut line round the body
    mesh = box_mesh(BOX_LO, BOX_HI, zone=band)
    p = box_project(mesh)
    g = HN.route_graph(p, HN.settings_of(p))
    m, *_ = _route(g, (1.0, 0.0, 1.5), (3.0, 0.0, 1.5))
    assert 0.09 <= m['moving'] <= 0.25                                   # no way round: it crosses once
    assert m['length'] == pytest.approx(2.0, rel=0.03)
    vflag = g.vflag[g.nearest((2.0, 0.0, 1.5))[0]]
    assert vflag & HN.MOVING and not g.clean[g.nearest((2.0, 0.0, 1.5))[0]]
    # an entry point asked for on the shut line sits beside it
    v, d = g.nearest((2.0, 0.0, 1.5), clean=True)
    assert g.clean[v] and 0.0 < d < 0.2


def test_separate_shells_are_joined_across_the_gap():
    body = box_mesh(BOX_LO, BOX_HI)
    pod = box_mesh((1.0, -0.1, 1.62), (1.3, 0.1, 1.82), n=(4, 3, 3))  # 12 cm above the roof
    from vantage.core.rig import merge
    g = _graph(merge(body, pod))
    assert g.stats['shells'] == 2
    m, a, b, _ = _route(g, (1.15, 0.0, 1.82), (0.2, 0.0, 1.5))
    assert 0.11 <= m['free'] <= 0.2 and m['length'] >= np.linalg.norm(b - a)


# ----------------------------------------------------------------------------- runs, rebuilt by hand
ENTRY = {'uid': 'g1', 'name': 'Roof grommet', 'kind': 'roof', 'x': 0.0, 'y': 0.0, 'z': 1.5, 'enabled': True}


def test_a_run_is_its_parts_and_the_inside_is_manhattan():
    p = box_project(harness={'entries': [ENTRY], 'slack': 1.1})
    c = cam('F', 3.0, 0.0, 1.52, route_extra_m=0.3)
    p.variant.cameras.append(c)
    p.ecu.route_factor, p.ecu.service_loop_m = 1.2, 0.4
    res = HN.run(p)
    assert res.routed and len(res.runs) == 1
    r = res.runs[0]
    e = res.entry('g1')
    assert r['mode'] == 'out_in' and r['entry'] == 'g1' and e['snap_m'] < 1e-9
    assert r['parts']['inside'] == pytest.approx(HN.manhattan(e['pos'], (1.0, 0.0, 0.6)) * 1.2, abs=1e-9)
    assert r['parts']['inside'] == pytest.approx((0.9 + 1.0) * 1.2, abs=1e-6)        # down 0.9, along 1.0
    straight = 3.0                                                                    # (3, 0) → (0, 0) on the roof
    assert straight * 1.1 <= r['parts']['outside'] + r['parts']['drop'] <= (straight + 0.06) * 1.1 * 1.03
    assert r['parts']['drop'] == pytest.approx(0.02 * 1.1, abs=0.035)
    assert r['parts']['service_loop'] == 0.4 and r['parts']['extra'] == 0.3
    assert r['length_m'] == pytest.approx(sum(r['parts'].values()), abs=1e-9)
    # the old estimate the app used: Manhattan from the camera × the slack + the loop + the extra
    assert r['manhattan_m'] == pytest.approx((2.0 + 0.0 + 0.92) * 1.2 + 0.4 + 0.3)
    # the path: device first, compute box last, the inside part axis-aligned at the end
    P, (i0, i1) = r['path'], r['inside_span']
    assert np.allclose(P[0], c.pos()) and np.allclose(P[-1], (1.0, 0.0, 0.6)) and i1 == len(P) - 1
    assert np.allclose(P[i0], e['pos'])
    assert all(np.count_nonzero(np.abs(P[k + 1] - P[k]) > 1e-12) == 1 for k in range(i0, i1))


def test_devices_inside_and_a_compute_box_outside():
    p = box_project(harness={'entries': [ENTRY]})
    imu = sensor(p, 'imu', 'imu_mems', 'IMU1', 0.2, 0.1, 0.5)
    lid = sensor(p, 'lidar', 'hesai_xt32', 'L1', 2.0, 0.0, 1.55)
    res = HN.run(p)
    r = res.data_run(imu.uid)
    assert r['device_inside'] and r['mode'] == 'in_in' and r['entry'] is None
    assert r['length_m'] == pytest.approx(HN.manhattan(imu.pos(), (1.0, 0.0, 0.6)) * 1.08 + 0.25, abs=1e-9)
    assert res.data_run(lid.uid)['mode'] == 'out_in'
    # the compute box on the roof (outside): the lidar reaches it along the roof, the IMU goes out through the grommet
    p.ecu.x, p.ecu.y, p.ecu.z = 1.0, 0.0, 1.55
    res = HN.run(p)
    assert not res.compute['inside']
    rl, ri = res.data_run(lid.uid), res.data_run(imu.uid)
    assert rl['mode'] == 'out_out' and rl['entry'] is None
    assert rl['length_m'] == pytest.approx((0.05 + 1.0 + 0.05) * 1.05 + 0.25, abs=0.06)
    assert ri['mode'] == 'in_out' and ri['entry'] == 'g1' and ri['inside_span'][0] == 0
    assert ri['parts']['inside'] == pytest.approx(HN.manhattan(imu.pos(), res.entry('g1')['pos']) * 1.08)
    # 'inside' set by hand overrides what the body says
    p.analysis.harness = {'entries': [ENTRY], 'compute': 'inside'}
    assert HN.run(p).compute['inside']


def test_the_cheapest_entry_is_chosen_and_an_assignment_is_kept():
    front = {'uid': 'g2', 'name': 'Front grommet', 'kind': 'bulkhead', 'x': 3.0, 'y': 0.0, 'z': 1.5}
    p = box_project(harness={'entries': [ENTRY, front]})
    c = cam('F', 3.2, 0.0, 1.52)
    p.variant.cameras.append(c)
    r = HN.run(p).data_run(c.uid)
    assert r['entry'] == 'g2'
    lengths = {}
    for uid in ('g1', 'g2'):
        p.analysis.harness = {'entries': [ENTRY, front], 'assign': {c.uid: uid}}
        r2 = HN.run(p).data_run(c.uid)
        assert r2['entry'] == uid
        lengths[uid] = r2['length_m']
    assert lengths['g2'] < lengths['g1'] and r['length_m'] == pytest.approx(lengths['g2'])
    # the front grommet switched off: the roof one; every one off: the Manhattan estimate and a serious finding
    p.analysis.harness = {'entries': [ENTRY, dict(front, enabled=False)]}
    assert HN.run(p).data_run(c.uid)['entry'] == 'g1'
    p.analysis.harness = {'entries': [dict(ENTRY, enabled=False), dict(front, enabled=False)]}
    res = HN.run(p)
    r = res.data_run(c.uid)
    assert r['entry'] is None and r['fallback'] == 'noentry' and r['severity'] == 'serious'
    assert r['length_m'] == pytest.approx(r['manhattan_m'])
    assert any(f['kind'] == 'entry' and f['severity'] == 'serious' for f in res.findings)


def test_an_entry_off_the_body_is_moved_onto_it_and_capacity_is_checked():
    far = dict(ENTRY, uid='g3', name='Floating', x=1.0, y=0.0, z=2.0, capacity=1)
    p = box_project(harness={'entries': [far], 'enabled': True})
    p.variant.cameras += [cam('A', 3.0, 0.3, 1.52), cam('B', 3.0, -0.3, 1.52, port=2)]
    res = HN.run(p)
    e = res.entry('g3')
    assert e['snap_m'] == pytest.approx(0.5, abs=0.01) and e['pos'][2] == pytest.approx(1.5)
    assert any(f['kind'] == 'entry_snap' for f in res.findings)
    cap = [f for f in res.findings if f['kind'] == 'capacity']
    assert len(e['cables']) == 2 and cap and cap[0]['severity'] == 'warning' and set(cap[0]['tags']) == {'A', 'B'}
    assert any(i.message == cap[0]['message'] for i in HN.issues(p))


def test_default_entries_on_the_sample():
    p = samples.tigor_sample_project()
    ents = HN.default_entries(p)
    assert [e['uid'] for e in ents] == ['roof', 'bulkhead', 'door_fl', 'door_fr', 'door_rl', 'door_rr', 'tailgate',
                                        'rear_panel']
    g = HN.route_graph(p)
    by = {e['uid']: np.array([e['x'], e['y'], e['z']]) for e in ents}
    for e in ents:                                      # on the body, off glass and shut lines
        v, d = g.nearest(by[e['uid']])
        assert d < 1e-3 and g.clean[v]
    assert by['roof'][2] > 1.45 and by['bulkhead'][0] > 2.0 and by['tailgate'][0] < 0 and by['rear_panel'][0] < -0.7
    assert by['door_fl'][1] == pytest.approx(-by['door_fr'][1]) and by['door_fl'][1] > 0.8
    assert by['door_rl'][0] < by['door_fl'][0]
    # left and right mirror each other on a symmetric body
    assert by['door_rl'][0] == pytest.approx(by['door_rr'][0]) and by['door_fl'][2] == pytest.approx(by['door_fr'][2])


# ----------------------------------------------------------------------------- the sample's data rig
@pytest.fixture(scope='module')
def rig():
    HN.clear_cache()
    p = samples.tigor_sample_project()
    p.active_uid = next(v for v in p.variants if v.sensors).uid
    res = HN.run(p)
    return p, res


def test_sample_rig_routes_every_device(rig):
    p, res = rig
    v = p.variant
    assert res.routed and res.variant_uid == v.uid
    data = [r for r in res.runs if r['what'] != 'power']
    assert {r['uid'] for r in data} == {c.uid for c in v.cameras} | {s.uid for s in v.sensors}
    assert {r['uid'] for r in res.runs if r['what'] == 'power'} == {s.uid for s in v.sensors
                                                                   if s.kind in ('lidar', 'radar', 'imu')}
    ecu = np.array([p.ecu.x, p.ecu.y, p.ecu.z])
    for r in data:                     # a cable is never shorter than the straight line between its ends + the loop
        assert r['length_m'] >= np.linalg.norm(r['path'][0] - ecu) + p.ecu.service_loop_m - 1e-9
        assert r['glass_m'] == 0 and r['moving_m'] == 0
        assert r['length_m'] == pytest.approx(sum(r['parts'].values()))
    by = {r['tag']: r for r in data}
    assert by['L1']['entry'] == 'roof' and by['ML']['entry'] == 'door_fl' and by['MR']['entry'] == 'door_fr'
    assert by['QL']['entry'] == 'door_rl' and by['R']['entry'] == 'rear_panel' and by['F']['entry'] == 'bulkhead'
    assert by['RD2']['device_inside'] and by['RD2']['mode'] == 'in_in'          # behind the bumper fascia
    assert by['L1']['cable'] == '1000base_t' and by['RD1']['cable'] == '100base_t1' and by['F']['cable'] == 'gmsl2'
    # ML and MR mirror each other
    assert by['ML']['length_m'] == pytest.approx(by['MR']['length_m'], rel=0.02)
    # the front camera's 3 m cable does not reach along the body (the Manhattan estimate said it did)
    assert by['F']['manhattan_m'] < 3.0 < by['F']['length_m']
    assert any(f['kind'] == 'cable' and f['tags'] == ['F'] and 'use the 15 m cable' in f['message']
               for f in res.findings)
    assert res.totals['runs'] == len(res.runs) and res.totals['length_m'] == pytest.approx(
        sum(r['length_m'] for r in res.runs))
    s = HN.summary(res)
    json.dumps(s)
    assert s['routed'] and len(s['runs']) == len(res.runs) and s['longest']['tag'] == res.totals['longest']['tag']
    assert s['tightest']['tag'] == 'F' and s['tightest']['use_pct'] == pytest.approx(by['F']['length_m'] / 3 * 100,
                                                                                    abs=0.1)


def test_power_pairs_match_the_power_budget(rig):
    from vantage.core import power as PW
    p, res = rig
    st = PW.PowerSettings.from_dict(p.analysis.power)
    for r in (r for r in res.runs if r['what'] == 'power'):
        s = p.variant.sensor(r['uid'])
        m = p.sensor_model_for(s)
        chk = PW.cable_check(st, m.power_w, max(st.peak_of(m.kind, m.power_w), m.power_w), r['length_m'], st.cable_awg)
        assert r['drop_pct'] == pytest.approx(chk['drop_pct']) and r['awg'] == st.cable_awg
        assert r['length_m'] == res.data_run(r['uid'])['length_m']
    # a 40 W lidar on 28 AWG over its routed length: the drop is flagged, with the gauge that fixes it
    p2 = samples.tigor_sample_project()
    p2.active_uid = next(v for v in p2.variants if v.sensors).uid
    p2.analysis.power = {'cable_awg': 28}
    bad = [r for r in HN.run(p2).runs if r['what'] == 'power' and r['severity'] != 'good']
    assert bad and all('AWG' in r['problems'][0][2] for r in bad)


# ----------------------------------------------------------------------------- link limits and the RF budget
def test_link_limits_by_interface():
    lib = {m.id: m for m in sensors.builtin_sensor_models()}
    assert HN.sensor_cable(lib['ouster_os1_128']) == '1000base_t'
    assert HN.sensor_cable(lib['hesai_ot128']) == '1000base_t1'
    assert HN.sensor_cable(lib['radar_long']) == '100base_t1'            # 'CAN-FD / 100BASE-T1': the Ethernet
    assert HN.sensor_cable(lib['imu_mems']) == 'can_fd'
    assert HN.sensor_cable(lib['velodyne_vlp16']) == '100base_tx'
    assert HN.sensor_cable(lib['gnss_patch'], 'rf_lmr240') == 'rf_lmr240'
    assert HN.CABLES['1000base_t1'].limit_m == 15.0 and HN.CABLES['1000base_t'].limit_m == 100.0

    class Mod:
        interface = 'FPD-Link IV'
    assert HN.camera_cable(Mod) == 'fpdlink4'
    Mod.interface = 'GMSL2'
    assert HN.camera_cable(Mod) == 'gmsl2'


def test_a_long_t1_run_breaks_its_limit():
    p = box_project(harness={'entries': [ENTRY]})
    lid = sensor(p, 'lidar', 'hesai_ot128', 'L1', 2.0, 0.0, 1.56)         # 1000BASE-T1: 15 m
    res = HN.run(p)
    r = res.data_run(lid.uid)
    assert r['limit_m'] == 15.0 and r['severity'] == 'good'
    # the compute box moved inside a long trailer of a body: the inside run grows past 90 % and then past 15 m
    for x_ecu, sev in ((-10.2, 'warning'), (-12.5, 'serious')):
        mesh = box_mesh((-14.0, -0.9, 0.3), BOX_HI, n=(80, 20, 12))
        q = box_project(mesh, {'entries': [ENTRY]}, ecu=(x_ecu, 0.0, 0.6))
        q.variant.sensors = [lid]
        r = HN.run(q).data_run(lid.uid)
        assert r['severity'] == sev and r['share'] == pytest.approx(r['length_m'] / 15.0)
        assert any(k == 'limit' and s == sev for s, k, _ in r['problems'])
    iss = [i for i in HN.issues(q) if i.uids == [lid.uid]]
    assert iss and iss[0].severity == 'serious' and 'Manhattan estimate' in iss[0].message     # routing off: estimate
    q.analysis.harness = {'entries': [ENTRY], 'enabled': True}
    iss = [i for i in HN.issues(q) if i.uids == [lid.uid]]
    assert iss and 'Manhattan' not in iss[0].message


def test_gnss_rf_budget_and_the_ins_destination():
    p = box_project(harness={'entries': [ENTRY]})
    ant = sensor(p, 'gnss', 'gnss_survey', 'GN1', 2.5, 0.0, 1.55)
    res = HN.run(p)
    r = res.data_run(ant.uid)
    st = res.settings
    assert r['what'] == 'rf' and r['cable'] == 'rf_rg58' and r['to'].endswith('GNSS receiver')
    loss = r['length_m'] * 0.7 + 2 * st.connector_db
    assert r['loss_db'] == pytest.approx(loss)
    assert r['margin_db'] == pytest.approx(st.gnss_lna_db - loss - st.gnss_min_db)
    # an INS with antenna inputs takes the antenna's coax
    ins = sensor(p, 'imu', 'ins_dual', 'INS', 0.3, 0.0, 0.5)
    r = HN.run(p).data_run(ant.uid)
    assert r['to_uid'] == ins.uid and np.allclose(r['path'][-1], ins.pos())
    q = box_project(harness={'entries': [ENTRY], 'enabled': True})
    q.variant.sensors = p.variant.sensors
    assert HN.route_text(q, ant) == 'GN1 → Roof grommet → INS (GNSS/INS)'
    assert HN.routed_length(q, ant) == pytest.approx(HN.run(q).data_run(ant.uid)['length_m'])
    # a weak LNA: no margin left (serious); the low-loss cable for this run fixes it
    p.analysis.harness = {'entries': [ENTRY], 'gnss_lna_db': 17.0}
    r = HN.run(p).data_run(ant.uid)
    assert r['margin_db'] < 0 and r['severity'] == 'serious' and any(k == 'rf' for _, k, _ in r['problems'])
    assert r['use'] == pytest.approx(r['loss_db'] / (17.0 - 15.0)) and r['use'] > 1
    p.analysis.harness = {'entries': [ENTRY], 'gnss_lna_db': 17.0, 'cables': {ant.uid: 'rf_lmr400'}}
    r2 = HN.run(p).data_run(ant.uid)
    assert r2['cable'] == 'rf_lmr400' and r2['margin_db'] > 0
    assert r2['loss_db'] == pytest.approx(r2['length_m'] * 0.17 + 2 * 0.1)


def test_camera_cable_options():
    p = box_project(harness={'entries': [ENTRY]})
    c = cam('F', 3.2, 0.0, 1.52, cable=3.0)
    p.variant.cameras.append(c)
    r = HN.run(p).data_run(c.uid)
    assert r['options'] == [3.0, 15.0] and r['fitted_m'] == 3.0
    assert r['length_m'] > 3.0 and r['option_m'] == 15.0
    assert r['use'] == pytest.approx(r['length_m'] / 3.0) and r['use_of'] == 'its 3 m cable'     # the cable binds
    assert any(k == 'cable' and 'use the 15 m cable' in m for _, k, m in r['problems'])
    c.cable_m = 15.0
    r = HN.run(p).data_run(c.uid)
    assert not [k for _, k, _ in r['problems'] if k in ('cable', 'option')] and r['severity'] == 'good'
    p.modules[c.module_id].cables_m = (3.0,)
    r = HN.run(p).data_run(c.uid)
    assert r['option_m'] is None and any(k == 'option' for _, k, _ in r['problems'])


# ----------------------------------------------------------------------------- the rest of the app
def test_routed_length_only_when_switched_on(rig):
    from vantage.core import power as PW
    from vantage.core import rules
    p, res = rig
    c = p.variant.by_tag('F')
    lid = next(s for s in p.variant.sensors if s.tag == 'L1')
    assert p.analysis.harness == {} and HN.routed_length(p, c) is None and HN.route_text(p, c) is None
    e = p.ecu
    assert rules.route_length(p, c) == pytest.approx((abs(c.x - e.x) + abs(c.y - e.y) + abs(c.z - e.z))
                                                      * e.route_factor + e.service_loop_m + c.route_extra_m)
    q = p.snapshot()
    q.analysis.harness = {'enabled': True}
    assert HN.routed_length(q, q.variant.by_tag('F')) == pytest.approx(res.data_run(c.uid)['length_m'])
    assert HN.routed_length(q, next(s for s in q.variant.sensors if s.tag == 'L1')) == pytest.approx(
        res.data_run(lid.uid)['length_m'])
    assert HN.route_text(q, q.variant.by_tag('F')) == 'F → Bulkhead grommet → Jetson AGX Orin'
    # one device routed on its own matches the whole harness, also after it moved; a hidden one keeps the estimate
    qf = q.variant.by_tag('F')
    qf.z += 0.05
    assert HN.routed_length(q, qf) == pytest.approx(HN.run(q).data_run(qf.uid)['length_m'])
    qf.visible = False
    assert HN.routed_length(q, qf) is None
    qf.visible = True
    # a camera of another variant that shares the uid: its own variant's route
    other = next(v for v in q.variants if v.uid != q.active_uid and v.by_tag('F') is not None)
    f2 = other.by_tag('F')
    f2.z += 0.2
    assert HN.routed_length(q, f2) != pytest.approx(HN.routed_length(q, q.variant.by_tag('F')))
    assert PW.route_length(q, lid) > 0                     # the power budget's own estimate stays callable


def test_manhattan_mode_and_no_body():
    p = samples.tigor_sample_project()
    p.active_uid = next(v for v in p.variants if v.sensors).uid
    res = HN.run(p, routed=False)
    assert not res.routed and res.entries == []
    for r in res.runs:
        assert r['mode'] == 'in_in' and r['length_m'] == pytest.approx(r['manhattan_m'])
    p.vehicle.mesh = None
    res = HN.run(p)
    assert not res.routed and any(f['kind'] == 'nomesh' for f in res.findings)
    assert HN.routed_length(p, p.variant.cameras[0]) is None
    assert HN.issues(p) == [] or all(i.severity in ('critical', 'serious', 'warning') for i in HN.issues(p))


def test_hidden_nan_and_empty():
    p = box_project(harness={'entries': [ENTRY]})
    a = cam('A', 3.0, 0.0, 1.52)
    b = cam('B', 2.0, 0.0, 1.52, visible=False)
    n = cam('N', float('nan'), 0.0, 1.52)
    p.variant.cameras += [a, b, n]
    res = HN.run(p)
    assert [r['tag'] for r in res.runs] == ['A']
    assert any(f['kind'] == 'skipped' and f['tags'] == ['N'] for f in res.findings)
    p.variant.cameras = []
    res = HN.run(p)
    assert res.runs == [] and res.totals['runs'] == 0 and res.totals['longest'] is None
    assert res.worst() in ('good', 'note')
    names = [n for n, _ in HN.files(res, p)]
    assert names == ['harness.json', 'harness.csv', 'harness.png']


def test_cancel_and_progress():
    p = box_project(harness={'entries': [ENTRY]})
    p.variant.cameras += [cam('A', 3.0, 0.0, 1.52), cam('B', 2.0, 0.5, 1.52, port=2)]
    seen = []
    assert HN.run(p, progress=seen.append) is not None and seen[-1] == 1.0 and seen == sorted(seen)
    assert HN.run(p, cancel=lambda: True) is None
    calls = []
    assert HN.run(p, cancel=lambda: len(calls) > 1 or calls.append(1)) is None


def test_settings_are_tolerant_and_round_trip():
    st = HN.HarnessSettings.from_dict({
        'enabled': 'yes', 'slack': 9, 'glass_cost': 'x', 'moving_cost': float('nan'), 'free_cost': 0.0,
        'compute': 'roof', 'gnss_cable': 'rf_lmr400', 'gnss_lna_db': -5, 'warn_share': 0.2,
        'entries': [{'uid': 'a', 'x': 1, 'y': 0, 'z': 1.5, 'kind': 'spaceship', 'capacity': 250},
                    {'uid': 'a', 'x': 2, 'y': 0, 'z': 1.5, 'name': '  '}, {'x': 'nope', 'y': 0, 'z': 0}, 'junk',
                    {'x': 1e9, 'y': 0, 'z': 0, 'enabled': 'off'}],
        'assign': {'u1': 'a', 'u2': 3}, 'cables': {'u1': 'rf_rg58', 'u2': 'warp'}, 'extra': {'u3': -1, 'u4': 'x'}})
    assert st.enabled and st.slack == 2.0 and st.glass_cost == 25.0 and st.moving_cost == 10.0 and st.free_cost == 1.0
    assert st.compute == 'auto' and st.gnss_cable == 'rf_lmr400' and st.gnss_lna_db == 0.0 and st.warn_share == 0.5
    assert [e['uid'] for e in st.entries] == ['a', 'a_', 'e5']
    assert st.entries[0]['kind'] == 'other' and st.entries[0]['capacity'] == 99
    assert st.entries[1]['name'] == 'Entry point' and st.entries[2]['x'] == 50.0 and not st.entries[2]['enabled']
    assert st.assign == {'u1': 'a'} and st.cables == {'u1': 'rf_rg58'} and st.extra == {'u3': 0.0}
    assert HN.HarnessSettings.from_dict(st.to_dict()) == st
    assert HN.HarnessSettings.from_dict(None) == HN.HarnessSettings() == HN.HarnessSettings.from_dict('junk')
    # an old project without the harness settings: the defaults, routing off
    p = samples.tigor_sample_project()
    p.analysis.harness = {}
    assert HN.settings_of(p) == HN.HarnessSettings() and not HN.settings_of(p).enabled


def test_exports(rig, tmp_path):
    import cv2
    p, res = rig
    paths = HN.export(res, p, str(tmp_path / 'out'))
    assert [n.split('/')[-1] for n in paths] == ['harness.json', 'harness.csv', 'harness.png']
    doc = json.loads((tmp_path / 'out' / 'harness.json').read_text())
    assert doc['schema'] == HN.SCHEMA and doc['frame'].startswith('vehicle') and doc['routed']
    assert len(doc['runs']) == len(res.runs) and len(doc['entry_points']) == len(res.entries)
    d0 = doc['runs'][0]
    r0 = res.runs[0]
    assert d0['length_m'] == pytest.approx(r0['length_m'], abs=1e-3) and d0['cable'] == r0['cable']
    assert np.allclose(d0['path'], r0['path'], atol=1e-4) and d0['connectors'] == r0['connectors']
    assert doc['totals']['length_m'] == pytest.approx(res.totals['length_m'], abs=1e-3)
    assert set(doc['cable_types']) == set(res.totals['by_cable'])
    with open(tmp_path / 'out' / 'harness.csv', newline='') as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == len(res.runs) + 1 and rows[-1]['run'] == 'total'
    for row, r in zip(rows, res.runs):
        assert row['run'] == r['id'] and float(row['length_m']) == pytest.approx(r['length_m'], abs=1e-3)
        pts = np.array([[float(v) for v in q.split()] for q in row['path'].split(';')])
        assert np.allclose(pts, r['path'], atol=1e-4)
    img = cv2.imread(str(tmp_path / 'out' / 'harness.png'))
    assert img is not None and img.shape[1] == 1400 and img.shape[0] > 400
    assert (img < 200).any()                                   # something is drawn on the light background


def test_rig_parts_carry_the_cables():
    p = samples.tigor_sample_project()
    p.active_uid = next(v for v in p.variants if v.sensors).uid
    from vantage.core.rig import preset
    p.rig = preset('sensor_mast', p.vehicle.mesh)
    head = p.rig[1]
    lid = next(s for s in p.variant.sensors if s.tag == 'L1')
    lid.x, lid.y, lid.z = head.x, head.y, head.z + head.height / 2 + 0.04
    r = HN.run(p).data_run(lid.uid)
    assert r['drop_m'] < 0.1                                    # it reaches the mast head, not the roof 0.6 m below
    assert r['free_m'] < 0.05                                   # down the mast, which touches the roof
    assert r['parts']['outside'] >= 0.5 * 1.05                  # at least the mast's height along it
    assert not any(k == 'free' for _, k, _ in r['problems'])
    p.rig = []
    r = HN.run(p).data_run(lid.uid)
    assert r['drop_m'] > HN.FREE_NOTE_M and any(k == 'free' for _, k, _ in r['problems'])


def test_a_large_body_is_simplified_first():
    m = box_mesh(BOX_LO, BOX_HI, n=(300, 150, 90))
    assert m.n_tris > 300_000
    p = box_project(m, {'entries': [ENTRY]})
    p.variant.cameras.append(cam('F', 3.0, 0.0, 1.52))
    g = HN.route_graph(p)
    assert g.stats['simplified'] and g.stats['triangles'] < 3 * HN.ROUTE_TRIS
    r = HN.run(p).runs[0]
    straight = 3.0                                              # along the roof to the grommet at x = 0
    assert r['parts']['outside'] == pytest.approx(straight * 1.05, rel=0.02)
    assert r['parts']['inside'] == pytest.approx((0.9 + 1.0) * 1.08, abs=0.02)


def test_issue_kinds_and_cache():
    p = box_project(harness={'entries': [dict(ENTRY, capacity=1)], 'enabled': True})
    p.variant.cameras += [cam('A', 3.0, 0.0, 1.52), cam('B', 2.0, 0.5, 1.52, port=2, cable=3.0)]
    iss = HN.issues(p)
    assert [i.severity for i in iss] == ['warning'] and 'carries 2 cables' in iss[0].message
    assert HN.issues(p) == iss                                 # the same inputs: the cached result
    r1 = HN.cached(p)
    assert HN.cached(p) is r1
    p.variant.cameras[0].x = 2.9
    assert HN.cached(p) is not r1


# ----------------------------------------------------------------------------- timing: the deserializer rule
def _rig_cams(p, *cams_):
    p.variant.cameras = list(cams_)
    return p


def test_stereo_pairs_found_by_geometry():
    p = box_project()
    a = cam('SL', 3.0, 0.15, 1.2, port=1, sync='A', yaw=2.0, pitch=5.0)
    b = cam('SR', 3.0, -0.15, 1.2, port=5, sync='B', yaw=-2.0, pitch=5.0)
    behind = cam('S2', 2.5, 0.15, 1.2, port=2, sync='A', yaw=2.0, pitch=5.0)        # in line with SL, not beside it
    other = cam('X', 3.0, -0.6, 1.2, port=3, sync='A', yaw=40.0)
    _rig_cams(p, a, b, behind, other)
    pairs = TM.stereo_pairs(p)
    found = {(x.tag, y.tag) for x, y, _, _ in pairs}
    assert ('SL', 'SR') in found and ('SL', 'S2') not in found and not any('X' in f for f in found)
    ang, base = next((g, L) for x, y, g, L in pairs if (x.tag, y.tag) == ('SL', 'SR'))
    fwd = lambda yaw, pitch: np.array([math.cos(math.radians(pitch)) * math.cos(math.radians(yaw)),   # noqa: E731
                                       math.cos(math.radians(pitch)) * math.sin(math.radians(yaw)),
                                       -math.sin(math.radians(pitch))])
    assert ang == pytest.approx(math.degrees(math.acos(fwd(2, 5) @ fwd(-2, 5))), abs=1e-9) and 3.9 < ang < 4.0
    assert base == pytest.approx(0.3)
    b.lens_id = 'h12r2'                                          # other optics: not a pair
    assert ('SL', 'SR') not in {(x.tag, y.tag) for x, y, _, _ in TM.stereo_pairs(p)}


def test_stereo_pair_on_two_deserializers_is_a_warning_with_a_fix():
    p = box_project()
    a = cam('SL', 3.0, 0.15, 1.2, port=1, sync='A')
    b = cam('SR', 3.0, -0.15, 1.2, port=5, sync='B')
    _rig_cams(p, a, b)
    out = TM.deserializer_check(p)
    st = [d for d in out if d['kind'] == 'stereo']
    assert len(st) == 1 and st[0]['severity'] == 'warning' and st[0]['uids'] == [a.uid, b.uid]
    assert 'different deserializers' in st[0]['message'] and 'different trigger lines' in st[0]['message']
    assert 'Move SR to port 2' in st[0]['message'] and 'put both on line A' in st[0]['message']
    iss = TM.issues(p)
    assert [i.severity for i in iss] == ['warning'] and iss[0].uids == [a.uid, b.uid]
    # the fix applied: nothing left
    b.port, b.sync = 2, 'A'
    assert TM.deserializer_check(p) == [] and TM.issues(p) == []
    res = TM.plan(p)
    assert not any('Stereo pair' in n for n in res.notes)
    b.port = 6
    assert any('Stereo pair SL–SR' in n for n in TM.plan(p).notes)


def test_trigger_line_across_deserializers():
    p = box_project()
    cams_ = [cam(t, 3.0 - i * 0.7, (-1) ** i * 0.9, 1.0, port=port, sync='C', yaw=90 * (-1) ** i, pitch=20 + 20 * i)
             for i, (t, port) in enumerate((('A1', 1), ('A2', 2), ('B1', 5)))]
    assert TM.stereo_pairs(p.__class__()) == [] and TM.stereo_pairs(_rig_cams(box_project(), *cams_)) == []
    _rig_cams(p, *cams_)
    out = TM.deserializer_check(p)
    assert len(out) == 1 and out[0]['kind'] == 'trigger' and out[0]['severity'] == 'warning'
    msg = out[0]['message']
    assert 'Trigger line C' in msg and 'A: A1 A2; B: B1' in msg and 'Move B1 to port 3' in msg
    cams_[2].port = 3                                           # the suggested move
    assert TM.deserializer_check(p) == []
    # more cameras than one deserializer takes: a note (wire the line to both), never an issue
    more = [cam(f'C{i}', 3.0 - i * 0.4, 0.9, 1.0, port=i + 1, sync='D', yaw=90.0, pitch=10 + 15 * i)
            for i in range(6)]
    _rig_cams(p, *more)
    out = TM.deserializer_check(p)
    assert [(d['kind'], d['severity']) for d in out] == [('trigger', 'note')] and TM.issues(p) == []
    # a free-running camera (no trigger line) or one without a port is not in a line
    more[3].sync = ''
    more[4].port = 0
    out = TM.deserializer_check(p)
    assert len(out) == 1 and out[0]['severity'] == 'warning' and out[0]['tags'] == ['C0', 'C1', 'C2', 'C5']
    assert "Swap ports so the line's cameras share deserializer A" in out[0]['message']     # port 4 is C3's


def test_the_sample_raises_no_new_issue():
    p = samples.tigor_sample_project()
    for v in p.variants:
        p.active_uid = v.uid
        assert TM.issues(p) == [] and TM.stereo_pairs(p) == []
        notes = TM.deserializer_check(p)
        assert all(d['severity'] == 'note' for d in notes)
        assert len(notes) == (1 if len(v.cameras) > 4 else 0)
        assert HN.issues(p) == []                              # routing off: the Manhattan estimate fits every link


# ----------------------------------------------------------------------------- the page, driven like a user
from test_gui import app, win, pump, settle, _wait     # noqa: E402,F401  (fixtures)


def _open(win, name, cls):
    if name not in win._page_widgets:
        pg = cls(win.ctx)
        win.pages.addWidget(pg)
        win._page_widgets[name] = pg
    win.set_page(name)
    return win._page_widgets[name]


def _type_into_field(app, f, text):
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    QTest.mouseClick(f.lineEdit(), Qt.MouseButton.LeftButton)
    QTest.keyClick(f.lineEdit(), Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
    QTest.keyClicks(f.lineEdit(), text)
    QTest.keyClick(f.lineEdit(), Qt.Key.Key_Return)
    pump(app, 0.1)


def _row(table, rid):
    from vantage.ui.workspaces.harness_ws import KEY
    return next(i for i in range(table.rowCount()) if table.item(i, 0) and table.item(i, 0).data(KEY) == rid)


def _fresh(app, pg):
    """Let a deferred edit land, then wait until the harness is routed again for it."""
    pump(app, 0.15)
    _wait(app, lambda: pg.ctl.fresh() and not pg.ctl.busy and not pg.ctl.timer.isActive())
    pump(app, 0.05)


def test_page_routes_edits_undoes_and_exports(app, win, tmp_path):
    from PyQt6.QtCore import QPoint, Qt
    from PyQt6.QtTest import QTest
    from vantage.ui.workspaces.harness_ws import C_ENTRY, C_EXTRA, E_CAP, E_NAME, E_ON, E_X, HarnessPage
    store = win.store
    rig = next(v for v in store.project.variants if v.sensors)
    store.edit('variants', lambda q: setattr(q, 'active_uid', rig.uid), 'Switched to the data rig')
    pg = _open(win, 'harness', HarnessPage)
    _fresh(app, pg)
    res = pg.ctl.result
    assert win.page_name() == 'harness' and pg.isVisible() and res.variant_uid == rig.uid
    assert pg.table.rowCount() == len(res.runs) == 18 and pg.entries.rowCount() == 8
    assert pg.tiles['cables'].value == '18' and pg.tiles['total'].value == f'{res.totals["length_m"]:.1f}'
    assert pg.tiles['tightest'].sev == 'serious' and pg.tiles['tightest'].note.startswith('F ·')     # its 3 m cable
    assert len(pg.findings) == len(res.findings) and pg.view.res is res and pg.view.display is not None
    F = rig.by_tag('F')
    # routed lengths everywhere: the toggle, undoable
    pg.enabled.click()
    pump(app, 0.1)
    assert store.project.analysis.harness == {'enabled': True} or store.project.analysis.harness['enabled']
    assert HN.routed_length(store.project, F) == pytest.approx(res.data_run(F.uid)['length_m'])
    store.undo.undo()
    pump(app, 0.1)
    assert HN.routed_length(store.project, F) is None and not pg.enabled.isChecked()
    # a typed cost; nonsense is clamped
    _type_into_field(app, pg.fields['slack'][0], '1.2')
    _fresh(app, pg)
    assert store.project.analysis.harness['slack'] == 1.2
    r = pg.ctl.result.data_run(F.uid)
    assert r['parts']['outside'] == pytest.approx(res.data_run(F.uid)['parts']['outside'] / 1.05 * 1.2)
    _type_into_field(app, pg.fields['slack'][0], '7')
    assert store.project.analysis.harness['slack'] == 2.0 and pg.fields['slack'][0].value() == 2.0
    store.undo.undo()
    store.undo.undo()
    _fresh(app, pg)
    assert store.project.analysis.harness == {}
    # the front camera sent in through the roof grommet, from its Entry picker
    i = _row(pg.table, f'{F.uid}:data')
    cb = pg.table.cellWidget(i, C_ENTRY)
    cb.setCurrentIndex(cb.findData('roof'))
    _fresh(app, pg)
    assert store.project.analysis.harness['assign'] == {F.uid: 'roof'}
    assert pg.ctl.result.data_run(F.uid)['entry'] == 'roof'
    assert pg.ctl.result.data_run(F.uid)['length_m'] > res.data_run(F.uid)['length_m']
    # the lidar's cable picked by hand
    L1 = next(s for s in rig.sensors if s.tag == 'L1')
    from vantage.ui.workspaces.harness_ws import C_CABLE
    cb = pg.table.cellWidget(_row(pg.table, f'{L1.uid}:data'), C_CABLE)
    cb.setCurrentIndex(cb.findData('1000base_t1'))
    _fresh(app, pg)
    assert store.project.analysis.harness['cables'] == {L1.uid: '1000base_t1'}
    assert pg.ctl.result.data_run(L1.uid)['limit_m'] == 15.0
    # extra length: a camera's own route extra; a sensor's in the harness settings
    pg.table.item(_row(pg.table, f'{F.uid}:data'), C_EXTRA).setText('0.5')
    _fresh(app, pg)
    assert store.project.variant.by_tag('F').route_extra_m == 0.5
    pg.table.item(_row(pg.table, f'{L1.uid}:data'), C_EXTRA).setText('1.25')
    _fresh(app, pg)
    assert store.project.analysis.harness['extra'] == {L1.uid: 1.25}
    assert pg.ctl.result.data_run(L1.uid)['parts']['extra'] == 1.25
    for _ in range(4):
        store.undo.undo()
    _fresh(app, pg)
    assert store.project.analysis.harness == {} and store.project.variant.by_tag('F').route_extra_m == 0.0
    # entry points: the front-left door seal switched off, so ML goes in elsewhere; renamed; a capacity of 1
    ML = rig.by_tag('ML')
    k = next(j for j in range(pg.entries.rowCount()) if pg.entries.item(j, E_NAME).text() == 'Door seal front left')
    pg.entries.item(k, E_ON).setCheckState(Qt.CheckState.Unchecked)
    _fresh(app, pg)
    ents = store.project.analysis.harness['entries']
    assert len(ents) == 8 and not next(e for e in ents if e['uid'] == 'door_fl')['enabled']
    assert pg.ctl.result.data_run(ML.uid)['entry'] != 'door_fl'
    pg.entries.item(k, E_NAME).setText('A-pillar boot L')
    _fresh(app, pg)
    assert next(e for e in store.project.analysis.harness['entries'] if e['uid'] == 'door_fl')['name'] == \
        'A-pillar boot L'
    j = next(j for j in range(pg.entries.rowCount()) if pg.entries.item(j, E_NAME).text() == 'Roof grommet')
    pg.entries.item(j, E_CAP).setText('1')
    _fresh(app, pg)
    assert any(f['kind'] == 'capacity' for f in pg.ctl.result.findings) and pg.tiles['entries'].sev == 'warning'
    # moved along x (typed in the project's frame)
    x0 = float(pg.entries.item(j, E_X).text())
    pg.entries.item(j, E_X).setText(f'{x0 + 0.3:.3f}')
    _fresh(app, pg)
    roof = next(e for e in store.project.analysis.harness['entries'] if e['uid'] == 'roof')
    assert roof['x'] == pytest.approx(store.project.from_display([x0 + 0.3, 0, 0])[0])
    # add one, remove it, back to the body's
    pg.b_add.click()
    _fresh(app, pg)
    assert len(store.project.analysis.harness['entries']) == 9 and pg.entries.rowCount() == 9
    pg.entries.selectRow(8)
    pg.b_remove.click()
    _fresh(app, pg)
    assert len(store.project.analysis.harness['entries']) == 8
    pg.b_reset.click()
    _fresh(app, pg)
    assert 'entries' not in store.project.analysis.harness or store.project.analysis.harness['entries'] == []
    assert pg.ctl.result.data_run(ML.uid)['entry'] == 'door_fl'
    # the view: each mode; a click on a route selects its row, a click on nothing clears it
    for mode in ('side', '3d', 'top'):
        pg.mode.buttons[mode].click()
        pump(app, 0.1)
        assert pg.view.mode == mode
    pump(app, 0.2)
    rid = f'{L1.uid}:data'
    xy = pg.view._screen[rid]
    r1 = pg.ctl.result.run(rid)
    k = max(range(r1['inside_span'][0]), key=lambda i: np.linalg.norm(xy[i + 1] - xy[i]))   # its longest stretch on
    mid = (xy[k] + xy[k + 1]) / 2                                                         # the roof, in the middle
    QTest.mouseClick(pg.view, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                     QPoint(int(round(mid[0])), int(round(mid[1]))))
    pump(app, 0.1)
    assert pg.selected == rid and pg.view.selected == rid
    assert pg.table.item(pg.table.currentRow(), 0).data(Qt.ItemDataRole.UserRole) == rid
    assert 'L1 →' in (pg.view.tip_text() or 'L1 →') and pg.view.pick(QPoint(*map(int, mid)).toPointF())
    QTest.mouseClick(pg.view, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(3, 3))
    pump(app, 0.1)
    assert pg.selected == ''
    # export
    paths = pg.export(str(tmp_path))
    assert len(paths) == 3 and json.loads((tmp_path / 'harness.json').read_text())['schema'] == HN.SCHEMA
    # leave the page: an edit is not routed until it shows again
    win.set_page('viewports')
    pump(app, 0.3)
    assert not pg.ctl.active
    shown = pg.ctl.result
    store.edit('ecu', lambda q: setattr(q.ecu, 'x', 0.5), 'Moved the compute box')
    pump(app, 0.4)
    assert pg.ctl.result is shown and pg.ctl.stale
    _open(win, 'harness', HarnessPage)
    _fresh(app, pg)
    assert pg.ctl.result is not shown and np.allclose(pg.ctl.result.compute['pos'][0], 0.5)
    # the project replaced while the page shows: it follows; the light theme repaints
    store.set_project(samples.tigor_sample_project())
    _fresh(app, pg)
    assert pg.ctl.result.variant_uid == store.project.variant.uid
    assert pg.table.rowCount() == len(pg.ctl.result.runs) == 6
    from vantage.ui.theme import THEME
    THEME.apply(app, 'light')
    pump(app, 0.2)
    pg.view.grab()
    THEME.apply(app, 'dark')
    pump(app, 0.1)


def test_main_view_drawings(app, win):
    from vantage.ui import harnessview as HV
    from vantage.ui.viewport3d import LineDraw
    from vantage.ui.workspaces.harness_ws import HarnessPage
    assert HV.scene_draws(win.ctx) == []                     # nothing until the harness is routed
    pg = _open(win, 'harness', HarnessPage)
    _fresh(app, pg)
    draws = HV.scene_draws(win.ctx)
    assert draws and all(isinstance(d, LineDraw) for d in draws)
    assert any(d.dash for d in draws)                        # the inside parts
    win.set_page('viewports')
    pump(app, 0.2)
    from PyQt6.QtGui import QColor, QImage, QPainter
    pv = win.viewarea.panes[0].plan
    pv.resize(800, 500)
    pv.fit()
    img = QImage(800, 500, QImage.Format.Format_ARGB32)
    img.fill(QColor(0, 0, 0, 0))
    p = QPainter(img)
    HV.paint_plan(pv, p)                                     # the routes and entry points seen from above
    p.end()
    a = np.frombuffer(img.constBits().asstring(img.sizeInBytes()), np.uint8).reshape(500, -1, 4)[:, :800]
    assert (a[:, :, 3] > 0).sum() > 500
    # View ▸ Overlays ▸ Cable harness (a key the window adds): kept up to date off the page, the views repainted
    seen = []
    win.ctx.overlays_changed.connect(lambda: seen.append(1))
    win.ctx.overlays['harness'] = True
    win.store.edit('ecu', lambda q: setattr(q.ecu, 'x', 0.8), 'Moved the compute box')
    pump(app, 0.15)
    _wait(app, lambda: pg.ctl.fresh() and not pg.ctl.busy)
    assert np.allclose(pg.ctl.result.compute['pos'][0], 0.8) and seen
    assert HV.scene_draws(win.ctx)
    win.ctx.overlays['harness'] = False
    win.store.edit('ecu', lambda q: setattr(q.ecu, 'x', 0.9), 'Moved the compute box')
    pump(app, 0.5)
    assert pg.ctl.stale and not pg.ctl.busy and HV.scene_draws(win.ctx) == []      # off: not routed any more


