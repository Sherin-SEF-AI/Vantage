"""Clock tree and time sync: holdover against ppm·t (+ ½·a·t²), FSYNC tick quantisation against a uniform
distribution's tick/√12, the projection error against an independent pinhole closed form, agreement with the sync
plan's motion figures, BMCA switchover, link asymmetry, missed and late FSYNC pulses as one frame of error, a
free-running FSYNC timer, determinism, settings, the export, the overlays and the page (offscreen)."""
import json
import math
import os

import numpy as np
import pytest

from vantage.core import clock_tree as CT
from vantage.core import samples
from vantage.core import timing as TM
from vantage.core.lens import Lens, Module
from vantage.core.project import Camera, Project


@pytest.fixture(scope='module')
def rig():
    """The sample's data rig (6 cameras on deserializers A and B, an OS1-128, five PTP radars)."""
    p = samples.tigor_sample_project()
    p.active_uid = next(v.uid for v in p.variants if v.name.startswith('Data rig'))
    return p


def _st(**kw):
    """Settings for a test: no events, a short timeline, quiet links unless asked."""
    d = dict(events=[], runs=40, duration_s=60.0, seed=3)
    d.update(kw)
    return CT.ClockTreeSettings.from_dict(d)


QUIET = {'gnss': {'pps_ns': 0}, 'compute': {'jitter_ns': 0, 'asym_ns': 0}, 'switch': {'jitter_ns': 0, 'asym_ns': 0}}


def _cam_nodes(topo, deser):
    return [n for n in topo.nodes if n.kind == 'camera' and n.parent == deser]


# =========================================================================== the tree
def test_default_tree_of_the_data_rig(rig):
    topo = CT.build_topology(rig)
    kinds = [n.kind for n in topo.nodes]
    assert kinds.count('camera') == 6 and kinds.count('radar') == 5 and kinds.count('lidar') == 1
    assert topo.gm == 'server' and topo.node('server').parent == 'gnss'          # power budget lists a time server
    assert topo.node('compute').parent == 'switch' and topo.node('switch').parent == 'server'
    assert {n.parent for n in topo.nodes if n.kind in ('lidar', 'radar')} == {'switch'}
    assert [n.info['tag'] for n in _cam_nodes(topo, 'deser:A')] == ['F', 'R', 'ML', 'MR']       # GMSL ports 1-4
    assert [n.info['tag'] for n in _cam_nodes(topo, 'deser:B')] == ['QL', 'QR']
    assert topo.lidar.startswith('sensor:') and topo.node(topo.lidar).kind == 'lidar'
    assert [n.id for n in topo.gm_capable()] == ['server', 'compute']
    assert any('no GNSS antenna' in x for x in topo.notes)
    # every parameter has a typical value with its source, and the 802.1AS defaults are the stated ones
    for n in topo.nodes:
        for k in n.params():
            assert k in n.typical and n.typical[k][1], (n.id, k)
    assert topo.node('switch').p['sync_ms'] == 125.0 and '802.1AS' in topo.node('switch').typical['sync_ms'][1]
    assert topo.node('server').p['priority'] == 246 and topo.node('compute').p['priority'] == 248
    # the camera's readout comes from the sync plan's settings
    assert _cam_nodes(topo, 'deser:A')[0].p['readout_ms'] == TM.TimingSettings().readout_ms
    # edits over the typical values; a lidar stamped from the PPS hangs off the GNSS receiver
    lid = topo.lidar
    t2 = CT.build_topology(rig, settings=_st(nodes={lid: {'stamping': 'pps', 'ppm': 2}, 'server': {'ppm': 'x'}}))
    assert t2.node(lid).parent == 'gnss' and t2.node(lid).p['ppm'] == 2 and set(t2.node(lid).edited) == {
        'stamping', 'ppm'}
    assert t2.node('server').p['ppm'] == 0.5 and not t2.node('server').edited


def test_without_a_time_server_the_compute_is_the_grandmaster(rig):
    import copy
    p = copy.deepcopy(rig)
    p.analysis.power = {'ptp_w': 0.0, 'switch_w': 0.0}
    topo = CT.build_topology(p)
    assert topo.gm == 'compute' and topo.node('compute').parent == 'gnss' and topo.node('server') is None
    assert topo.node('switch') is None and {n.parent for n in topo.nodes if n.kind == 'radar'} == {'compute'}
    assert any('only grandmaster-capable' in x for x in topo.notes)


# =========================================================================== clocks
def test_holdover_drift_is_ppm_times_t(rig):
    """With a perfect PPS the grandmaster's error at the end of a dropout is the run's frequency draw × t exactly
    (y·t), plus ½·a·t² with aging; over the runs |error| at its 95th percentile is 1.96·σ·t."""
    st = _st(runs=300, nodes=dict(QUIET, server={'aging_ppb_day': 0}),
             events=[{'kind': 'pps_dropout', 't_s': 10.0, 'dur_s': 40.0}])
    sim = CT.simulate(CT.build_topology(rig, settings=st), st)
    k0, k1 = int(round(10.0 / sim.dt)), int(round(50.0 / sim.dt))
    x = sim.X['server']
    assert np.all(x[:, :k0 + 1] == 0.0)                                  # locked to a perfect PPS: no error
    y = sim.draws['server']['y']
    assert np.allclose(x[:, k1], y * 40.0, rtol=1e-9, atol=1e-15)        # x = y·t at the end of the dropout
    assert np.allclose(x[:, k0 + 80], y * 80 * sim.dt, rtol=1e-9, atol=1e-15)
    sigma = 0.5e-6                                                       # the time server's typical TCXO, 1σ
    assert np.percentile(np.abs(x[:, k1]), 95) == pytest.approx(1.96 * sigma * 40.0, rel=0.15)
    assert sim.holdover['server'] == [[10.0, 50.0]]
    # aging adds ½·a·t² (a fixed rate of a sign drawn per run)
    st2 = _st(runs=20, nodes=dict(QUIET, server={'aging_ppb_day': 5000.0}),
              events=[{'kind': 'pps_dropout', 't_s': 10.0, 'dur_s': 40.0}])
    sim2 = CT.simulate(CT.build_topology(rig, settings=st2), st2)
    a = sim2.draws['server']['a']
    assert np.allclose(np.abs(a), 5000e-9 / 86400.0)
    assert np.allclose(sim2.X['server'][:, k1], sim2.draws['server']['y'] * 40 + 0.5 * a * 40 ** 2, rtol=1e-9,
                       atol=1e-15)
    # after the PPS returns the slew policy brings it back at most 500 ppm (no step)
    tail = sim2.X['server'][:, k1:]
    assert np.all(np.abs(np.diff(tail, axis=1)) <= 500e-6 * sim2.dt * 1.001 + 1e-12)
    # the holdover time the finding uses: 1.96·σ·t = budget
    assert CT.holdover_time(0.5, 0.0, 100e-6) == pytest.approx(100e-6 / (1.96 * 0.5e-6))
    t = CT.holdover_time(0.5, 3.0, 100e-6)
    a3 = 3e-9 / 86400
    assert 1.96 * 0.5e-6 * t + 0.5 * a3 * t * t == pytest.approx(100e-6)


def test_quantisation_is_uniform_tick_over_sqrt12(rig):
    """Every FSYNC edge on the nearest 10 µs tick of a perfect clock: the stamp error is uniform over a tick, its
    standard deviation tick/√12 and its magnitude at most half a tick."""
    topo0 = CT.build_topology(rig)
    st = _st(runs=60, nodes=dict(QUIET, **{'deser:A': {'tick_us': 10.0, 'rephase_s': 0.0}}))
    topo = CT.build_topology(rig, settings=st)
    sim = CT.simulate(topo, st)
    assert np.all(sim.X['compute'] == 0.0)
    cam = _cam_nodes(topo0, 'deser:A')[0].id
    _, s = CT.stamps(sim, cam)
    tick = 10e-6
    assert np.abs(s).max() <= tick / 2 + 1e-15
    assert s.std() == pytest.approx(tick / math.sqrt(12), rel=0.02)
    assert abs(s.mean()) < 0.02 * tick
    # a timer counting whole ticks per frame drifts by (N·tick − T) a frame from each re-phase
    times = np.arange(90) / 30.0
    q = CT.quantisation(times, 1 / 30.0, tick, 1.0, np.array([0.3]))
    N = round((1 / 30.0) / tick)
    for j0 in (0, 30, 60):
        j = np.arange(j0, j0 + 30)
        assert np.allclose(q[0, j] - q[0, j0], (j - j0) * (N * tick - 1 / 30.0), atol=1e-15)
    assert CT.quantisation(times, 1 / 30.0, 0.0, 0.0, np.zeros(3)).shape == (3, 90)


def test_a_missed_fsync_pulse_is_one_frame_of_error(rig):
    topo0 = CT.build_topology(rig)
    st = _st(runs=12, events=[{'kind': 'fsync_missed', 't_s': 30.0, 'node': 'deser:A'},
                              {'kind': 'fsync_late', 't_s': 45.0, 'node': _cam_nodes(topo0, 'deser:B')[0].id,
                               'value': 2.0}])
    topo = CT.build_topology(rig, settings=st)
    sim = CT.simulate(topo, st)
    for n in _cam_nodes(topo, 'deser:A'):
        t, s = CT.stamps(sim, n.id)
        T = 1.0 / n.info['fps']
        off = np.abs(s) > T / 2
        assert np.all(off.sum(axis=1) == 1)                           # one frame in every run
        j = int(np.flatnonzero(off[0])[0])
        assert t[j] >= 30.0 > t[j] - T and np.allclose(s[:, j], -T, atol=1e-4)
    qb, qr = _cam_nodes(topo, 'deser:B')
    t, s = CT.stamps(sim, qb.id)
    late = np.abs(s) > 1e-3
    assert np.all(late.sum(axis=1) == 1) and np.allclose(s[late], -2e-3, atol=1e-4)
    assert not (np.abs(CT.stamps(sim, qr.id)[1]) > 1e-3).any()
    res = CT.run(rig, settings=st)
    assert {s['tag']: s['frames_off'] for s in res.sensors if s['kind'] == 'camera'} == {
        'F': 1.0, 'R': 1.0, 'ML': 1.0, 'MR': 1.0, 'QL': 0.0, 'QR': 0.0}
    msg = next(f for f in res.findings if f['kind'] == 'fsync' and f['severity'] == 'warning')['message']
    assert 'one frame 33.3 ms off' in msg and 'F, R, ML, MR' in msg
    # stamped on arrival, a missed pulse only drops a frame
    st2 = _st(runs=4, events=st.events, nodes={n.id: {'stamping': 'arrival'} for n in _cam_nodes(topo, 'deser:A')})
    sim2 = CT.simulate(CT.build_topology(rig, settings=st2), st2)
    assert not (np.abs(CT.stamps(sim2, _cam_nodes(topo, 'deser:A')[0].id)[1]) > 1e-3).any()


def test_grandmaster_loss_hands_over_after_the_announce_timeout(rig):
    st = _st(runs=30, bmca_s=3.0, events=[{'kind': 'gm_loss', 't_s': 20.0, 'dur_s': 0.0}])
    topo = CT.build_topology(rig, settings=st)
    sim = CT.simulate(topo, st)
    gm = np.array(sim.gm)
    assert set(gm[sim.t < 20.0]) == {'server'} and set(gm[(sim.t >= 20.0) & (sim.t < 23.0)]) == {''}
    assert set(gm[sim.t >= 23.0]) == {'compute'}
    assert sim.holdover[topo.lidar][0] == [20.0, 23.0]                 # the slaves hold over through the timeout
    assert sim.holdover['compute'][0][0] == 20.0                       # and the compute stays in holdover as GM
    # afterwards the lidar follows the compute, not the lost server
    lid = sim.X[topo.lidar][:, -1]
    assert np.abs(lid - sim.X['compute'][:, -1]).max() < 2e-6
    # the server back after 10 s takes over at once
    st2 = _st(runs=4, events=[{'kind': 'gm_loss', 't_s': 20.0, 'dur_s': 10.0}])
    sim2 = CT.simulate(CT.build_topology(rig, settings=st2), st2)
    gm2 = np.array(sim2.gm)
    assert set(gm2[(sim2.t >= 30.0)]) == {'server'} and set(gm2[(sim2.t >= 23.0) & (sim2.t < 30.0)]) == {'compute'}
    res = CT.run(rig, settings=st)
    assert res.metrics['gm'] == 'Time server'


def test_link_asymmetry_offsets_the_clocks_behind_it_by_half(rig):
    topo0 = CT.build_topology(rig)
    lid = topo0.lidar
    rd = next(n.id for n in topo0.nodes if n.kind == 'radar')
    st = _st(runs=60, duration_s=60.0, nodes={'gnss': {'pps_ns': 0}},
             events=[{'kind': 'asymmetry', 't_s': 10.0, 'node': lid, 'value': 1000.0}])
    sim = CT.simulate(CT.build_topology(rig, settings=st), st)
    base = (10.0 + 10.0) / 2 * 1e-9                                     # its own and the switch's typical 10 ns
    off_l = (sim.X[lid][:, -1] - sim.X['server'][:, -1]).mean()
    off_r = (sim.X[rd][:, -1] - sim.X['server'][:, -1]).mean()
    assert off_l == pytest.approx(base + 500e-9, abs=15e-9)
    assert off_r == pytest.approx(base, abs=15e-9)
    assert topo0.path_links(lid, 'server') == [lid, 'switch']
    assert topo0.path_links(lid, 'compute') == [lid, 'compute']


def test_a_free_running_fsync_timer_drifts_at_its_ppm(rig):
    topo0 = CT.build_topology(rig)
    st = _st(runs=50, nodes=dict(QUIET, **{'deser:B': {'fsync': 'free', 'tick_us': 0.0, 'aging_ppb_day': 0.0}}))
    topo = CT.build_topology(rig, settings=st)
    sim = CT.simulate(topo, st)
    n = _cam_nodes(topo, 'deser:B')[0]
    t, s = CT.stamps(sim, n.id)
    y = sim.draws['deser:B']['y']
    assert np.allclose(s, y[:, None] * t[None, :], rtol=1e-9, atol=1e-15)
    res = CT.run(rig, settings=st)
    f = next(f for f in res.findings if f['kind'] == 'free_fsync')
    assert 'Deserializer B' in f['message'] and set(f['uids']) == {c.uids[0] for c in _cam_nodes(topo0, 'deser:B')}
    rig.analysis.clock_tree = st.to_dict()
    try:
        assert any('runs free' in i.message for i in CT.issues(rig))
    finally:
        rig.analysis.clock_tree = {}


# =========================================================================== projection
def _pinhole_rig(x=1.8, y=0.4, z=1.3, yaw=20.0, pitch=8.0):
    p = Project()
    p.modules['m'] = Module(id='m', name='Test module', width_px=1920, height_px=1536, pixel_um=3.0, fps=30.0)
    p.lenses['pin'] = Lens(id='pin', name='Pinhole 6 mm', focal_mm=6.0, projection='rectilinear')
    cam = Camera(tag='C', x=x, y=y, z=z, yaw=yaw, pitch=pitch, roll=0.0, module_id='m', lens_id='pin')
    p.variant.cameras.append(cam)
    return p, cam


def _closed_form(cam, u, v, r, dt, speed, yaw_dps, f=2000.0, cx=960.0, cy=768.0):
    """An independent pinhole: the camera's forward, left and up axes from its yaw and pitch (positive down), the
    point on the pixel's ray r from the camera, the vehicle's arc over dt, and the point projected again."""
    ps, ph = math.radians(cam.yaw), math.radians(cam.pitch)
    fwd = np.array([math.cos(ph) * math.cos(ps), math.cos(ph) * math.sin(ps), -math.sin(ph)])
    left = np.array([-math.sin(ps), math.cos(ps), 0.0])
    up = np.cross(fwd, left)
    c = cam.pos()
    d = fwd - left * (u - cx) / f - up * (v - cy) / f
    P = c + r * d / np.linalg.norm(d)
    th = math.radians(yaw_dps) * dt
    T = np.array([speed / math.radians(yaw_dps) * math.sin(th), speed / math.radians(yaw_dps) * (1 - math.cos(th)),
                  0.0]) if yaw_dps else np.array([speed * dt, 0.0, 0.0])
    Q = P - T
    Q = np.array([math.cos(th) * Q[0] + math.sin(th) * Q[1], -math.sin(th) * Q[0] + math.cos(th) * Q[1], Q[2]])
    q = Q - c
    return math.hypot(cx - f * (q @ left) / (q @ fwd) - u, cy - f * (q @ up) / (q @ fwd) - v)


def test_projection_error_against_a_pinhole_closed_form():
    p, cam = _pinhole_rig()
    dt, v, w, r = 4e-3, 15.0, 25.0, 12.0
    pe = CT.projection_error(p, cam, dt, v, w, 'range', r, grid=(9, 7))
    for i in range(len(pe['u'])):
        assert pe['err'][i] == pytest.approx(_closed_form(cam, pe['u'][i], pe['v'][i], r, dt, v, w), rel=1e-6,
                                             abs=1e-9)
    # a camera on the yaw axis, level and straight ahead, turning only: every point shifts f·tan(ω·Δt)
    p2, c2 = _pinhole_rig(0.0, 0.0, 1.0, 0.0, 0.0)
    pe2 = CT.projection_error(p2, c2, 5e-3, 0.0, 20.0, 'range', 30.0, grid=(5, 5))
    assert pe2['centre_px'] == pytest.approx(2000.0 * math.tan(math.radians(0.1)), rel=1e-9)
    assert pe2['du'][0] > 0 and abs(pe2['dv'][0]) < 1e-9                # turning left: the scene moves right
    # driving straight: the centre does not move, a point off the axis moves by f·y·(1/(r − vΔt) − 1/r)
    pe3 = CT.projection_error(p2, c2, 10e-3, 20.0, 0.0, 'range', 10.0, grid=(5, 5))
    assert pe3['centre_px'] == pytest.approx(0.0, abs=1e-9)
    i = 1 + 2 * 5                                                       # left edge, middle row
    u = pe3['u'][i]
    y_off = 10.0 * (960.0 - u) / math.hypot(2000.0, 960.0 - u)          # lateral offset of the point, r along the ray
    xr = 10.0 * 2000.0 / math.hypot(2000.0, 960.0 - u)
    assert pe3['err'][i] == pytest.approx(2000.0 * y_off * (1 / (xr - 0.2) - 1 / xr), rel=1e-9)
    # on the ground the rays meet z = 0; the ones above the horizon have no point
    pe4 = CT.projection_error(p, cam, 1e-3, 10.0, 0.0, 'ground', grid=(9, 7))
    ok = np.isfinite(pe4['err'])
    assert ok.any() and not ok.all() and np.allclose(pe4['P'][ok][:, 2], 0.0, atol=1e-9)


def test_agrees_with_the_sync_plans_motion_figures(rig):
    """Driving straight, a small time error makes the image move by core.timing's image speed × Δt over the same
    image samples (its blur figure is that speed × the exposure); the plan's travel is v·Δt."""
    plan = TM.plan(rig, settings=TM.TimingSettings(speed_kmh=50.0, object_m=5.0))
    var = rig.variant
    for c in var.cameras[:3]:
        pc = plan.cam(c.uid)
        pe = CT.projection_error(rig, c, 1e-4, 50 / 3.6, 0.0, 'range', 5.0)
        assert pe['centre_px'] == pytest.approx(pc['rate_centre_px_s'] * 1e-4, rel=2e-3)
        assert pe['worst_px'] == pytest.approx(pc['rate_worst_px_s'] * 1e-4, rel=2e-3)
        sl = CT.slip(rig, c, abs(pc['misalign_ms']), 50.0, 0.0)
        assert sl['translation_mm'] / 10 == pytest.approx(pc['travel_cm'], rel=1e-9)
    res = CT.run(rig, settings=_st(runs=4, speed_kmh=50.0, yaw_rate_dps=0.0))
    tp = TM.plan(rig)
    for row in res.projection:
        assert row['plan_misalign_ms'] == pytest.approx(tp.cam(row['uid'])['misalign_ms'])
        assert row['plan_travel_cm'] == pytest.approx(tp.cam(row['uid'])['travel_cm'])


def test_the_example_slip_at_fs_image_centre(rig):
    """1.5 ms in a 25°/s turn: ω·Δt = 0.0375°; at F's image centre it is a fraction of a pixel 20 m out, more with
    speed, and much more on the ground close to the car (parallax)."""
    F = rig.variant.by_tag('F')
    turn = CT.slip(rig, F, 1.5, 0.0, 25.0)
    assert turn['rotation_deg'] == pytest.approx(0.0375) and turn['translation_mm'] == 0.0
    both = CT.slip(rig, F, 1.5, 30.0, 25.0)
    ground = CT.slip(rig, F, 1.5, 30.0, 25.0, 'ground')
    assert 0.1 < turn['centre_px'] < 2.0 and both['centre_px'] > turn['centre_px']
    assert ground['centre_px'] > 5 * both['centre_px']
    # the turn alone scales with Δt
    assert CT.slip(rig, F, 3.0, 0.0, 25.0)['centre_px'] == pytest.approx(2 * turn['centre_px'], rel=1e-3)


# =========================================================================== the study
def test_run_findings_and_determinism(rig):
    st = CT.ClockTreeSettings.from_dict({'runs': 30})                  # the default urban canyon scenario
    a = CT.run(rig, settings=st)
    b = CT.run(rig, settings=st)
    assert json.dumps(CT.to_doc(a, rig)) == json.dumps(CT.to_doc(b, rig))
    fa, fb = dict(CT.files(a, rig)), dict(CT.files(b, rig))
    assert fa['clock_tree_sensors.csv'] == fb['clock_tree_sensors.csv']
    c = CT.run(rig, settings=CT.ClockTreeSettings.from_dict({'runs': 30, 'seed': 2}))
    assert CT.to_doc(c, rig)['sensors'] != CT.to_doc(a, rig)['sensors']
    kinds = {f['kind'] for f in a.findings}
    assert {'projection', 'holdover', 'fsync'} <= kinds
    sev = [f['severity'] for f in a.findings]
    assert sev == sorted(sev, key=CT.SEVERITIES.index)
    assert a.metrics['gm'] == 'Time server' and a.metrics['holdover_s'] == pytest.approx(
        CT.holdover_time(0.5, 3.0, 100e-6))
    lid = a.sensor(a.topology.lidar)
    assert lid['vs_lidar'] is None and lid['true']['p95'] > 0
    # radars on the same switch are within a microsecond of the lidar (802.1AS's goal)
    assert all(s['vs_lidar']['p95'] < 1e-6 for s in a.sensors if s['kind'] == 'radar')
    # a tight holdover budget and a large projection limit: holdover is serious, projection good
    d = CT.run(rig, settings=CT.ClockTreeSettings.from_dict({'runs': 10, 'time_budget_us': 5, 'threshold_px': 500}))
    assert any(f['kind'] == 'holdover' and f['severity'] == 'serious' for f in d.findings)
    assert any(f['kind'] == 'projection' and f['severity'] == 'good' for f in d.findings)
    # a slow lidar-camera path: the p95 projection error over the limit is serious and reaches the Issues list
    slow = {'runs': 10, 'nodes': {'deser:A': {'tick_us': 2000.0, 'rephase_s': 0.0}}}
    rig.analysis.clock_tree = slow
    try:
        e = CT.run(rig)
        assert any(f['kind'] == 'projection' and f['severity'] == 'serious' for f in e.findings)
        assert any('lidar points land' in i.message and i.severity == 'serious' for i in CT.issues(rig))
    finally:
        rig.analysis.clock_tree = {}
    assert CT.run(rig, settings=st, cancel=lambda: True) is None


def test_settings_are_tolerant():
    s = CT.ClockTreeSettings.from_dict({'runs': 1e9, 'seed': 'x', 'duration_s': -4, 'surface': 'sky',
                                        'events': [{'kind': 'pps_dropout', 't_s': 5}, {'kind': 'nope'}, 3,
                                                   {'kind': 'fsync_late', 't_s': 1e9, 'value': -5}],
                                        'nodes': {'server': {'ppm': -3, 'policy': 'jump', 'servo_hz': 'fast',
                                                             'bogus': 1}, 'x': 'y'},
                                        'rows_compensated': 'no', 'yaw_rate_dps': 999, 'unknown': 1})
    assert s.runs == 400 and s.seed == 1 and s.duration_s == 5.0 and s.surface == 'range'
    assert [e['kind'] for e in s.events] == ['pps_dropout', 'fsync_late']
    assert s.events[0]['dur_s'] == 30.0 and s.events[1]['t_s'] == 5.0 and s.events[1]['value'] == 2.0
    assert s.nodes == {'server': {'ppm': 0.0}} and s.rows_compensated is False and s.yaw_rate_dps == 180.0
    d = CT.ClockTreeSettings().to_dict()
    assert CT.ClockTreeSettings.from_dict(d).to_dict() == d
    assert CT.ClockTreeSettings.from_dict(None).to_dict() == d
    assert [e['kind'] for e in d['events']] == ['pps_dropout', 'fsync_missed']


def test_presets_and_event_words(rig):
    topo = CT.build_topology(rig)
    assert CT.preset_events('nominal', topo) == []
    allf = CT.preset_events('all', topo, 120.0)
    assert [e['kind'] for e in allf] == ['pps_dropout', 'gm_loss', 'asymmetry', 'fsync_missed', 'fsync_late']
    assert CT.describe_event(allf[0]) == 'PPS dropout at 20 s for 60 s'
    assert CT.describe_event(allf[1]).endswith('for good')
    assert 'L1' in CT.describe_event(allf[2], topo) and 'Deserializer A' in CT.describe_event(allf[3], topo)
    res = CT.run(rig, settings=CT.ClockTreeSettings.from_dict({'runs': 6, 'events': allf}))
    assert any(f['kind'] == 'step' for f in res.findings) or any(c['steps_per_run'] for c in res.clocks)


def test_export_files_and_doc(rig, tmp_path):
    res = CT.run(rig, settings=CT.ClockTreeSettings.from_dict({'runs': 8}))
    paths = CT.export(res, rig, str(tmp_path))
    assert [os.path.basename(x) for x in paths] == ['clock_tree.json', 'clock_tree_sensors.csv',
                                                     'clock_tree_projection.csv', 'clock_tree.png']
    doc = json.loads((tmp_path / 'clock_tree.json').read_text())
    assert doc['schema'] == CT.SCHEMA and doc['variant'].startswith('Data rig') and 'Limits' not in doc['limits']
    assert doc['settings'] == res.settings.to_dict() and doc['settings']['runs'] == 8
    sw = next(n for n in doc['topology']['nodes'] if n['id'] == 'switch')
    assert sw['params']['sync_ms'] == 125.0 and '802.1AS' in sw['typical']['sync_ms']['source']
    assert len(doc['sensors']) == 12 and len(doc['projection']) == 6
    assert doc['events'][0]['description'] == 'PPS dropout at 20 s for 60 s'
    csv_ = (tmp_path / 'clock_tree_projection.csv').read_text().splitlines()
    assert csv_[0].startswith('camera,against,dt_p50_ms') and len(csv_) == 7
    assert (tmp_path / 'clock_tree.png').read_bytes()[:4] == b'\x89PNG'
    assert CT.summary(res)['grandmaster'] == 'Time server'


def test_overlays(rig):
    st = CT.ClockTreeSettings.from_dict({'runs': 4, 'yaw_rate_dps': 0.0, 'speed_kmh': 36.0})
    res = CT.run(rig, settings=st)
    F = rig.variant.by_tag('F')
    segs = CT.overlay_segments(rig, res, F.uid, dt_s=0.01)
    assert len(segs) > 50
    for p0, p1, rgba in segs:                                            # straight ahead: v·Δt along +x
        d = np.subtract(p1, p0)
        assert d == pytest.approx([0.1, 0.0, 0.0], abs=1e-9)
        assert rgba[:3] == pytest.approx(CT.CAM_RGBA[F.slot % 8]) and rgba[3] == 0.9
    big = CT.overlay_segments(rig, res, F.uid, dt_s=0.01, scale=10.0)
    assert np.subtract(big[0][1], big[0][0]) == pytest.approx([1.0, 0.0, 0.0], abs=1e-9)
    every = CT.overlay_segments(rig, res)
    assert len(every) > len(segs) and len({s[2] for s in every}) >= 4
    img = CT.image_segments(rig, res, F.uid, dt_s=0.01, scale=3.0)
    assert img and all(math.hypot(b[0] - a[0], b[1] - a[1]) == pytest.approx(3.0 * e, rel=1e-9) for a, b, e in img)
    assert CT.overlay_segments(rig, None, dt_s=0.0)[0][0] == pytest.approx(CT.overlay_segments(rig, None,
                                                                                              dt_s=0.0)[0][1])


# =========================================================================== the page
from test_gui import app, pump, settle, win  # noqa: E402,F401


def test_page(app, win, tmp_path):
    from PyQt6.QtCore import QPoint, Qt
    from PyQt6.QtTest import QTest
    from vantage.ui.workspaces.clock_tree_ws import ClockTreePage
    st = win.store
    win.switch_variant(next(v.uid for v in st.project.variants if v.name.startswith('Data rig')))
    settle(app, win)
    st.edit('analysis', lambda p: setattr(p.analysis, 'clock_tree', {'runs': 12}), 'fewer runs')
    pg = ClockTreePage(win.ctx)
    pg.resize(1700, 1000)
    pg.show()
    try:
        pump(app, 0.2)
        assert pg.wait(60) and pg.res is not None
        res = pg.res
        assert res.settings.runs == 12 and pg.nodes.rowCount() == len(res.topology.nodes)
        assert pg.proj.rowCount() == 6 and pg.events.rowCount() == 2 and 'Grandmaster Time server' in pg.head.text()
        assert pg.findings_box.count() == len(res.findings)
        for w in (pg.tree, pg.dist, pg.timeline, pg.heat):
            w.grab()                                                  # paints without errors
        # pick the switch in the tree: its parameters with their typical values
        pump(app, 0.1)
        box = pg.tree.boxes['switch']
        QTest.mouseClick(pg.tree, Qt.MouseButton.LeftButton, pos=QPoint(int(box.center().x()), int(box.center().y())))
        pump(app, 0.1)
        assert pg.sel == 'switch' and pg.params.rowCount() == 4 and pg.params.item(0, 0).text() == 'Sync interval'
        assert '802.1AS' in pg.params.item(0, 2).text()
        # edit a parameter: one undo step, saved with the project, the run follows
        n = st.undo.count()
        assert pg.set_param('switch', 'sync_ms', 31.25)
        assert st.undo.count() == n + 1 and st.project.analysis.clock_tree['nodes'] == {'switch': {'sync_ms': 31.25}}
        assert pg.wait(60) and pg.res.topology.node('switch').p['sync_ms'] == 31.25
        assert pg.params.item(0, 1).text() == '31.25 ms' and pg.b_reset.isEnabled()
        assert pg.reset_node('switch') and 'switch' not in st.project.analysis.clock_tree['nodes']
        st.undo.undo()
        st.undo.undo()
        pump(app, 0.1)
        assert 'nodes' not in st.project.analysis.clock_tree or not st.project.analysis.clock_tree['nodes']
        # a scenario preset and a fault added by hand
        assert pg.apply_preset('all')
        assert [e['kind'] for e in pg.settings.events][:2] == ['pps_dropout', 'gm_loss']
        assert pg.add_event('fsync_late') and pg.settings.preset == '' and pg.events.rowCount() == 6
        assert pg.set_event(5, t_s=12.0) and pg.settings.events[5]['t_s'] == 12.0
        assert pg.remove_event(5) and pg.events.rowCount() == 5
        assert pg.wait(60) and [e['kind'] for e in pg.res.events][:2] == ['pps_dropout', 'gm_loss']
        # settings fields and the slip calculator
        assert pg.set_option('yaw_rate_dps', 10.0) and pg.settings.yaw_rate_dps == 10.0
        assert '10°/s' in pg.slip_text.text()
        pg.map_which.set_value('slip')
        pg._fill_map()
        assert 'slip 1.5 ms' in pg.heat.caption
        pg.dist_mode.changed.emit('true')
        assert pg.dist.subtitle == 'against true time'
        # export what is shown
        assert pg.wait(60)
        paths = pg.export(str(tmp_path))
        assert [os.path.basename(x) for x in paths][:4] == ['clock_tree.json', 'clock_tree_sensors.csv',
                                                            'clock_tree_projection.csv', 'clock_tree.png']
        assert json.loads((tmp_path / 'clock_tree.json').read_text())['settings']['yaw_rate_dps'] == 10.0
    finally:
        pg.cancel()
        pg.hide()
        pg.deleteLater()
        pump(app, 0.1)
