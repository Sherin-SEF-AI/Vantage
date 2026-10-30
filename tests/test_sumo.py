"""SUMO traffic co-simulated with CARLA (vantage.sim.sumo_cosim, the bridge's --traffic-source sumo, the rig's
traffic_source, the CARLA page's Traffic section), against tests/fake_carla and tests/fake_sumo (a fake traci, sumolib
and sumo package, a fake SUMO process that holds the TraCI socket, a stand-in for CARLA's sumo_integration package):

* the settings: tolerant, clamped, checked against the rig; a CARLA-traffic rig and run are what they were;
* the bridge steps SUMO once per synchronous CARLA tick, before it; SUMO's vehicles are spawned (role vantage_sumo,
  physics on), moved to SUMO's pose (closed forms of CARLA's conversions written out here) with SUMO's velocity and yaw
  rate as their CARLA velocity (so CARLA's radar sees their motion) and destroyed when they arrive; the ego is mirrored
  into SUMO after each tick; traffic lights both ways;
* SUMO's vehicles are in the ground truth and run.json names them; the traffic block validates against
  docs/sim_run.schema.json;
* the same seed gives the same SUMO vehicles on the same paths, another seed other ones;
* tidy-up after a cancel, a SUMO crash (run.json says why) and a SUMO that does not start; SUMO missing in CARLA's
  Python refuses the run before CARLA is touched;
* the net: CARLA's bundled one, one generated once and then taken from the cache, the user's .net.xml and .sumocfg;
* the app's check of CARLA's Python (never importing SUMO), and the page.

A test with CARLA's real sumo_integration package runs when a CARLA folder with Co-Simulation/Sumo is found
(VANTAGE_CARLA_ROOT or the usual places); it is skipped otherwise."""
import contextlib
import json
import math
import os
import signal
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
FAKE_SUMO = os.path.join(HERE, 'fake_sumo')
FAKE_ROOT = os.path.join(FAKE_SUMO, 'carla_root')
OFFSET = (100.0, 50.0)                    # the fake nets' netOffset (tests/fake_sumo/.../util/netconvert_carla.py)
DATA_RIG = 'Data rig · 6-cam + lidar + radars'


def _forget():
    for m in list(sys.modules):
        if m in ('traci', 'sumolib', 'sumo', 'carla_ext') or m.startswith(('traci.', 'sumolib.', 'sumo_integration')):
            del sys.modules[m]


@contextlib.contextmanager
def fake_sumo(no_landmarks=False, real_root=None):
    """The fake carla module as `carla` (CARLA's co-simulation imports it by name), the fake SUMO packages on
    sys.path, CARLA's co-simulation API added to the fake carla; all undone after."""
    from test_sim import _fake
    mp = pytest.MonkeyPatch()
    _forget()
    try:
        fc = _fake()
        fc.reset()
        mp.syspath_prepend(FAKE_SUMO)
        mp.setitem(sys.modules, 'carla', fc)
        for k in ('SUMO_HOME', 'FAKE_SUMO_FAIL', 'FAKE_SUMO_CRASH_AT_STEP', 'CARLA_ROOT'):
            mp.delenv(k, raising=False)
        import carla_ext
        carla_ext.extend(fc, mp, no_landmarks)
        import traci
        traci.LOG.clear()
        traci._S = None
        yield fc
    finally:
        try:
            import traci
            if traci._S is not None:
                traci.close()
        except Exception:
            pass
        mp.undo()
        _forget()


def _project(cams=('F',), lidar=True):
    from test_sim import _project as P
    p = P(lidar_z=2.0, extra=False)
    v = p.variant
    v.cameras[:] = [c for c in v.cameras if c.tag in cams]
    v.sensors[:] = [s for s in v.sensors if s.kind == 'lidar' and lidar]
    return p


def _bridge(out, settings=None, cams=('F',), cancel=None, **kw):
    """A bridge on a small rig (settings over a SUMO drive's defaults), not run yet: (bridge, rig doc)."""
    from test_sim import _args
    from vantage.core import simrig
    from vantage.sim import carla_bridge as CB
    p = _project(cams)
    d = {'image_scale': 0.08, 'duration_s': 1.0, 'traffic': 5, 'warmup_s': 0.25, 'seed': 4, 'traffic_source': 'sumo'}
    d.update(settings or {})
    rig = simrig.build(p, simrig.SimSettings.from_dict(d))
    out = str(out)
    simrig.export(rig, p, os.path.join(out, 'rig'))
    doc, remaps = CB.load_rig(os.path.join(out, 'rig', 'sim_rig.json'))
    kw.setdefault('carla_root', FAKE_ROOT)
    kw.setdefault('sumo_cache', os.path.join(os.path.dirname(out), 'sumo-cache'))
    import carla
    return CB.Bridge(carla, doc, remaps, out, _args(doc, out, **kw), None, cancel), doc


def _tick_log(fc):
    """World.tick logged into traci's LOG (one list: the order of SUMO steps and CARLA ticks)."""
    import traci
    real = fc.World.tick

    def tick(self, seconds=10.0):
        traci.LOG.append(('world_tick',))
        return real(self, seconds)
    return real, tick


def sumo_drive(out, settings=None, cams=('F',), **kw):
    """A SUMO drive through the bridge (inside fake_sumo()): {'report', 'bridge', 'rig', 'traci', 'carla_log', 'out'}."""
    import carla
    import traci
    real, tick = _tick_log(carla)
    carla.World.tick = tick
    try:
        b, doc = _bridge(out, settings, cams, **kw)
        rep = b.run()
    finally:
        carla.World.tick = real
    return {'report': rep, 'bridge': b, 'rig': doc, 'traci': list(traci.LOG), 'carla_log': list(carla.LOG),
            'out': str(out)}


def _main(argv):
    """The bridge's command line in this process (its signal mask put back after)."""
    from vantage.sim import carla_bridge as CB
    old = signal.pthread_sigmask(signal.SIG_BLOCK, [])
    try:
        return CB.main(argv)
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, old)


# ======================================================================= closed forms of CARLA's conversions
def carla_pose_of(x, y, angle, length):
    """A SUMO vehicle's CARLA pose: its front bumper (x, y) moved back half its length along its heading (SUMO's angle
    is clockwise from north: heading (sin a, cos a)), the net offset taken off, y mirrored; CARLA yaw = angle − 90."""
    a = math.radians(angle)
    cx, cy = x - math.sin(a) * length / 2, y - math.cos(a) * length / 2
    return cx - OFFSET[0], -(cy - OFFSET[1]), angle - 90.0


def sumo_pose_of(x, y, yaw, length):
    """A CARLA vehicle's SUMO pose: its centre moved forward half its length (CARLA's yaw towards +y), y mirrored, the
    offset added; SUMO angle = yaw + 90."""
    r = math.radians(yaw)
    fx, fy = x + math.cos(r) * length / 2, y + math.sin(r) * length / 2
    return fx + OFFSET[0], -fy + OFFSET[1], yaw + 90.0


def _wrap(a):
    return (a + 180.0) % 360.0 - 180.0


# ======================================================================= settings and the rig
def test_settings_tolerant_clamped_and_checked_against_the_rig(tmp_path):
    from test_sim_contract import RIG_SCHEMA, _load, _validate
    from vantage.core import samples
    from vantage.core import simrig as SR
    from vantage.sim import sumo_cosim as SUMO
    st = SR.SimSettings.from_dict({})
    assert st.traffic_source == 'carla' and st.sumo == SUMO.default()
    junk = SR.SimSettings.from_dict({'traffic_source': 'bus', 'sumo': {'net': 'x.txt', 'demand': 'many', 'period_s': 1e9,
                                                                       'classes': ['truck', 'boat'], 'tls': 'none',
                                                                       'binary': '', 'extra': 1}})
    assert junk.traffic_source == 'carla'
    assert junk.sumo == dict(SUMO.default(), period_s=60.0, classes=['truck'])
    so, bad = SUMO.clean_sumo({'net': 'x.txt', 'demand': 'many', 'classes': ['boat'], 'period_s': 'soon'})
    assert so == SUMO.default() and len(bad) == 4
    assert SUMO.clean_sumo({'net': '/a/b.sumocfg', 'period_s': 0.0})[0]['period_s'] == 0.1
    assert SR.SimSettings.from_dict({'traffic_source': 'sumo', 'sumo': {'net': '/a/b.net.xml'}}).sumo['net'] == \
        '/a/b.net.xml'
    # what the net will be
    assert SUMO.net_kind({'net': 'auto'}, 'Town04') == 'bundled' and SUMO.net_kind({'net': 'auto'}, 'Town10HD_Opt') == \
        'generated'
    assert SUMO.net_kind({'net': 'generate'}, 'Town04') == 'generated'
    assert SUMO.net_kind({'net': 'a.sumocfg'}, 'Town04') == 'user_config'
    assert 'uv pip install --python /x/py eclipse-sumo==1.23.1 traci==1.23.1 sumolib==1.23.1' == \
        SUMO.install_command('/x/py')
    # the rig: a CARLA-traffic rig has no traffic block and does not change with SUMO's settings
    p = samples.tigor_sample_project()
    p.active_uid = next(v.uid for v in p.variants if v.name == DATA_RIG)
    base = {'image_scale': 0.1}
    plain = SR.build(p, SR.SimSettings.from_dict(base), tables=False).doc
    other = SR.build(p, SR.SimSettings.from_dict(dict(base, traffic_source='carla', sumo={'net': 'generate',
                                                                                          'tls': 'sumo'})),
                     tables=False).doc
    assert 'traffic_source' not in plain
    plain.pop('created'), other.pop('created')
    assert plain == other
    # SUMO's block, checked against the rig
    schema = _load(RIG_SCHEMA)
    for town, kind in (('Town04', 'bundled'), ('Town10HD', 'generated')):
        doc = SR.build(p, SR.SimSettings.from_dict(dict(base, traffic_source='sumo', town=town, walkers=2, traffic=12)),
                       tables=False)
        tb = doc.doc['traffic_source']
        assert tb['source'] == 'sumo' and tb['net_kind'] == kind and tb['vehicles'] == 12
        assert doc.doc['simulation']['dt_s'] == 0.0125 and tb['step_s'] == 0.025 and tb['seed'] == 1   # whole ms
        assert any('every 25 ms (2 simulation steps' in n for n in doc.notes)
        assert any('SUMO traffic: 12 vehicles at once' in n for n in doc.notes)
        assert any('walker AI' in n for n in doc.notes)
        assert (any('no traffic lights' in n for n in doc.notes)) == (kind == 'bundled')
        assert doc.doc['simulation']['bridge_memory_gb'] == pytest.approx(
            plain['simulation']['bridge_memory_gb'] + SR.SUMO_MEMORY_GB)
        assert _validate(json.loads(json.dumps(doc.doc)), schema) == []
    actors = SR.build(p, SR.SimSettings.from_dict(dict(base, traffic_source='sumo', actors=[
        {'id': 'c', 'kind': 'cone', 'pose': {'x': 5.0, 'y': 0.0}}])), tables=False)
    assert actors.doc['traffic_source']['vehicles'] == 0 and any('not started' in n for n in actors.notes)
    gone = SR.build(p, SR.SimSettings.from_dict(dict(base, traffic_source='sumo', sumo={'net': str(tmp_path / 'no.net.xml')})),
                    tables=False)
    assert any('does not exist' in n and 'refuses' in n for n in gone.notes)
    assert SUMO.problems(gone.doc['traffic_source']['sumo'])
    # the bridge's command line over the rig's
    from vantage.sim import carla_bridge as CB
    rig = dict(plain)
    a = CB.resolve_args(CB.parser().parse_args(['r', 'o', '--traffic-source', 'sumo']), rig)
    assert a.traffic_source == 'sumo' and a.sumo == SUMO.default()
    a = CB.resolve_args(CB.parser().parse_args(['r', 'o', '--sumo-net', 'my.net.xml']), rig)
    assert a.traffic_source == 'sumo' and a.sumo['net'] == os.path.abspath('my.net.xml')
    a = CB.resolve_args(CB.parser().parse_args(['r', 'o']), rig)
    assert a.traffic_source == 'carla' and a.sumo is None
    with pytest.raises(ValueError):
        CB.resolve_args(CB.parser().parse_args(['r', 'o', '--sumo-net', 'my.txt']), rig)


# ======================================================================= one SUMO drive through the bridge
@pytest.fixture(scope='module')
def drive(tmp_path_factory):
    """A 1 s drive of a camera and the mast lidar with ground truth, 5 SUMO vehicles on a net generated from the map,
    SUMO running the traffic lights, two walkers."""
    base = tmp_path_factory.mktemp('sumo_drive')
    with fake_sumo() as fc:
        r = sumo_drive(base / 'run', {'sumo': {'tls': 'sumo'}, 'walkers': 2})
        r['world'] = fc.STATE['world']
        r['proc'] = r['bridge'].cosim.proc
        yield r


def test_sumo_steps_once_before_every_carla_tick(drive):
    """At 80 Hz (12.5 ms: not whole milliseconds, SUMO's clock) SUMO steps 25 ms before every second CARLA tick, and
    never during the tidy-up."""
    log = [e for e in drive['traci'] if e[0] in ('init', 'step', 'world_tick', 'close')]
    i0 = next(i for i, e in enumerate(log) if e[0] == 'init')
    i1 = next(i for i, e in enumerate(log) if e[0] == 'close')
    seq = [e[0] for e in log[i0 + 1:i1]]
    rep = drive['report']
    k = rep['traffic_source']['steps_per_sumo_step']
    assert k == 2
    n = 0
    for a, b in zip(seq, seq[1:] + ['end']):
        if a == 'world_tick':
            n += 1
        else:                                                     # a SUMO step: right before a tick, every k-th
            assert b == 'world_tick' and (n + 1) % k == 0
    assert seq.count('step') == n // k and n > 50
    assert not any(e[0] == 'step' for e in log[i1:])
    argv = log[i0][2]
    dt = drive['rig']['simulation']['dt_s']
    assert float(argv[argv.index('--step-length') + 1]) == k * dt == rep['traffic_source']['step_s'] == 0.025
    assert argv[argv.index('--seed') + 1] == '4' and argv[argv.index('--max-num-vehicles') + 1] == '6'
    assert os.path.basename(argv[0]) == 'sumo'                   # headless SUMO, never sumo-gui
    # the SUMO process ended, the files it ran from stay with the drive
    assert drive['proc'].poll() is not None
    assert sorted(os.listdir(os.path.join(drive['out'], 'sumo'))) == ['demand.rou.xml', 'netconvert.log',
                                                                     'randomTrips.log', 'run.sumocfg', 'sumo.log']


def test_sumo_vehicles_spawned_moved_destroyed_and_the_ego_mirrored(drive):
    rep, clog = drive['report'], drive['carla_log']
    tr = rep['traffic_source']
    rows = {a['sumo_id']: a for a in tr['actors']}
    assert tr['vehicles_spawned'] == len(rows) == 5 and rep['traffic'] == 5
    spawns = {e[1]: e for e in clog if e[0] == 'spawn'}
    batch = [e for e in clog if e[0] == 'batch']
    assert sorted(e[1] for e in batch) == sorted(a['actor_id'] for a in rows.values())
    for a in rows.values():
        sp = spawns[a['actor_id']]
        assert sp[2] == a['blueprint'] == a['sumo_type'] and sp[3]['role_name'] == 'vantage_sumo'
        assert sp[4][2] > 20.0                                    # spawned high (SPAWN_OFFSET_Z), then moved
    destroyed = {e[1] for e in clog if e[0] == 'destroy'}
    assert {a['actor_id'] for a in rows.values()} <= destroyed and not drive['world']._actors
    # the ego mirrored into SUMO: added once, moved after every tick
    mir = tr['mirrored']
    assert len(mir) == 1 and mir[0]['role'] == 'hero' and mir[0]['sumo_id'] == 'carla0'
    t = drive['traci']
    adds = [e for e in t if e[0] == 'add']
    assert adds == [('add', 'carla0', 'carla_route_passenger', mir[0]['sumo_type'])]
    moves = [e for e in t if e[0] == 'move' and e[1] == 'carla0']
    assert abs(len(moves) - sum(1 for e in t if e[0] == 'step')) <= 1 and all(e[5] == 2 for e in moves)
    # traffic lights: SUMO's states set on CARLA's light (frozen for it, given back after)
    tl = [e for e in clog if e[0] in ('tl_state', 'tl_freeze')]
    assert tl[0] == ('tl_freeze', 50, True) and tl[-1] == ('tl_freeze', 50, False)
    assert {e[2] for e in tl if e[0] == 'tl_state'} >= {'Green'}
    assert tr['tls'] == {'owner': 'sumo', 'linked': 1, 'frozen_in_carla': True}
    assert tr['vehicles_in_carla']['peak'] == 5 and 0 < tr['vehicles_in_carla']['mean'] <= 5


def test_sumo_vehicles_are_in_the_ground_truth_and_the_report(drive):
    from test_sim_contract import RUN_SCHEMA, _load, _validate
    rep, out = drive['report'], drive['out']
    ids = {a['actor_id'] for a in rep['traffic_source']['actors']}
    gt = _load(os.path.join(out, 'ground_truth', 'index.json'))
    seen = {o.get('actor_id') for c in gt['cameras'].values() for f in c['frames'] for o in f.get('objects', [])}
    assert seen & ids                                             # SUMO's vehicles are objects of the camera
    run = _load(os.path.join(out, 'run.json'))
    assert run['status'] == 'ok' and _validate(run, _load(RUN_SCHEMA)) == []
    tr = run['traffic_source']
    assert tr['net']['kind'] == 'generated' and tr['net']['town'] == 'Town10HD' and tr['net']['file'].endswith('.net.xml')
    assert tr['demand']['mode'] == 'count' and tr['demand']['vehicles'] == 5 and tr['demand']['trips'] > 5
    assert tr['sumo_version'] == '1.23.1' and tr['package_version'] == '1.23.1'
    assert any('walker AI' in n for n in run['notes'])
    ds = _load(os.path.join(out, 'dataset.json'))
    assert ds['carla']['traffic_source']['vehicles_spawned'] == 5
    assert 'Eclipse SUMO' in open(os.path.join(out, 'README.md'), encoding='utf-8').read()
    # annotated: every SUMO vehicle near the ego has a sample_annotation, with SUMO's speed as its velocity
    b = drive['bridge']
    ann = _load(os.path.join(out, 'nuscenes', 'v1.0-vantage', 'sample_annotation.json'))
    assert len(ann) >= 5
    # the velocity of a vehicle CARLA does not simulate: SUMO's speed along its heading
    sid = rep['traffic_source']['actors'][0]['sumo_id']
    b.cosim.speed[sid] = 7.0
    b.cosim.sumo2carla[sid] = 12345
    vx, vy, vz = b.cosim.velocity(12345, 30.0)
    assert (vx, vy, vz) == pytest.approx((7 * math.cos(math.radians(30)), 7 * math.sin(math.radians(30)), 0.0))


def test_the_same_seed_gives_the_same_sumo_vehicles_on_the_same_paths(tmp_path):
    """Same seed: the same SUMO vehicles, types, blueprints and every pose they were given in CARLA (the fake world's
    LOG has each spawn; their annotations hold every keyframe pose); another seed: other ones."""
    out = {}
    for name, seed in (('a', 3), ('b', 3), ('c', 8)):
        with fake_sumo() as fc:
            r = sumo_drive(tmp_path / name, {'seed': seed}, cams=())
            ann = json.load(open(tmp_path / name / 'nuscenes' / 'v1.0-vantage' / 'sample_annotation.json'))
            out[name] = (r['report']['traffic_source']['actors'],
                         sorted((tuple(a['translation']), tuple(a['rotation'])) for a in ann),
                         [e for e in r['traci'] if e[0] == 'move'])
    strip = [[{k: v for k, v in a.items() if k != 'actor_id'} for a in out[n][0]] for n in 'abc']
    assert strip[0] == strip[1] and out['a'][1] == out['b'][1] and out['a'][2] == out['b'][2]
    assert strip[0] != strip[2] and out['a'][1] != out['c'][1]


# ======================================================================= the co-simulation step by step
@contextlib.contextmanager
def cosim(tmp, sumo=None, town='Town10HD_Opt', seed=5, n=4, ego=True, dt=0.05, **kw):
    """A CoSim on a fresh fake world of the town with an ego on the autopilot (inside fake_sumo())."""
    import carla
    from vantage.sim import sumo_cosim as SUMO
    c = carla.Client('127.0.0.1', 2000)
    w = c.load_world(town)
    s = w.get_settings()
    s.synchronous_mode, s.fixed_delta_seconds = True, dt
    w.apply_settings(s)
    vehicles = []
    if ego:
        bp = w.get_blueprint_library().find('vehicle.tesla.model3')
        e = w.spawn_actor(bp, carla.Transform(carla.Location(-20.0, 3.5, 0.3), carla.Rotation(0.0, 0.0, 0.0)))
        e.set_autopilot(True)
        vehicles.append(e)
    events = {'spawn': [], 'gone': []}
    notes = []
    kw.setdefault('note', notes.append)
    kw.setdefault('carla_root', FAKE_ROOT)
    cs = SUMO.CoSim(carla, c, w, SUMO.clean_sumo(sumo or {})[0], seed=seed, dt=dt, vehicles=n, horizon_s=40.0,
                    work_dir=str(tmp / 'sumo'), cache_dir=str(tmp.parent / 'cache'),
                    on_spawn=lambda a, sid: events['spawn'].append((a.id, sid)),
                    on_destroy=lambda aid: events['gone'].append(aid),
                    carla_vehicles=lambda: list(vehicles), **kw)
    cs.events, cs.w, cs.vehicles, cs.notes_seen = events, w, vehicles, notes
    try:
        yield cs
    finally:
        cs.close()


def _step(cs, k=1):
    for _ in range(k):
        cs.before_tick()
        cs.w.tick()
        cs.after_tick(True)


def test_poses_both_ways_follow_carlas_conversions(tmp_path):
    with fake_sumo():
        import traci
        with cosim(tmp_path / 'p') as cs:
            cs.start()
            _step(cs, 20)                                         # (one trip every 0.25 s)
            sim = traci._S
            assert len(cs.sumo2carla) == 4
            # each CARLA actor holds SUMO's pose of this step after CARLA's tick (SUMO has not stepped since): it was
            # put one step behind it with SUMO's velocity, and the tick carried it there
            for sid, cid in cs.sumo2carla.items():
                v = sim.veh[sid]
                x, y, yaw = carla_pose_of(v['x'], v['y'], v['angle'], sim.types[v['type']]['length'])
                tr = cs.w.get_actor(cid).get_transform()
                assert (tr.location.x, tr.location.y) == pytest.approx((x, y), abs=1e-6)
                assert _wrap(tr.rotation.yaw - yaw) == pytest.approx(0.0, abs=1e-6)
                assert cs.w.get_actor(cid).physics is True
            # the ego in SUMO: at its CARLA pose converted, after the tick
            ego = cs.vehicles[0]
            tr = ego.get_transform()
            m = [e for e in traci.LOG if e[0] == 'move' and e[1] == 'carla0'][-1]
            x, y, ang = sumo_pose_of(tr.location.x, tr.location.y, tr.rotation.yaw, 2 * ego.bounding_box.extent.x)
            assert (m[2], m[3]) == pytest.approx((x, y), abs=1e-3) and _wrap(m[4] - ang) == pytest.approx(0, abs=1e-3)
            assert m[5] == 2                                      # keepRoute=2: anywhere, as CARLA's co-simulation


def test_between_sumo_steps_vehicles_move_on_at_sumos_speed(tmp_path):
    """CARLA at 80 Hz: SUMO steps every 25 ms; on the step between, each SUMO vehicle is moved on along its heading by
    its SUMO speed × 12.5 ms, and at SUMO's next step it is at SUMO's pose again."""
    from vantage.sim import sumo_cosim as SUMO
    assert [SUMO.sumo_substeps(1 / hz) for hz in (20, 40, 50, 80, 100, 120)] == [1, 1, 1, 2, 1, 3]
    with fake_sumo():
        import traci
        with cosim(tmp_path / 'd', dt=0.0125) as cs:
            assert cs.k == 2 and cs.sumo_dt == 0.025
            cs.start()
            _step(cs, 40)                                         # ends on a SUMO step (40 ticks = 20 SUMO steps)
            assert cs.ticks % 2 == 0 and cs.sumo2carla
            sim = traci._S
            before = {sid: cs.w.get_actor(cid).get_transform() for sid, cid in cs.sumo2carla.items()}
            _step(cs, 1)                                          # no SUMO step: dead reckoning
            for sid, cid in cs.sumo2carla.items():
                tr, b = cs.w.get_actor(cid).get_transform(), before[sid]
                y = math.radians(b.rotation.yaw)
                d = sim.veh[sid]['speed'] * 0.0125
                assert (tr.location.x, tr.location.y) == pytest.approx(
                    (b.location.x + d * math.cos(y), b.location.y + d * math.sin(y)), abs=1e-9)
            _step(cs, 1)                                          # SUMO steps: SUMO's pose
            for sid, cid in cs.sumo2carla.items():
                v = sim.veh[sid]
                x, y_, _ = carla_pose_of(v['x'], v['y'], v['angle'], sim.types[v['type']]['length'])
                tr = cs.w.get_actor(cid).get_transform()
                assert (tr.location.x, tr.location.y) == pytest.approx((x, y_), abs=1e-6)
            assert cs.report()['steps_per_sumo_step'] == 2 and cs.report()['step_s'] == 0.025


def test_carla_sees_sumo_vehicles_move_with_sumos_velocity(tmp_path):
    """CARLA 0.9.16 reports no velocity for a vehicle with physics off moved by set_transform, and its radar takes a hit's
    radial velocity from the hit actor's velocity (measured): SUMO's vehicles keep physics on and get SUMO's speed along
    their heading and their yaw rate as CARLA target velocities on every tick (SUMO's steps and the dead-reckoned ones
    between, at 80 Hz), put one tick behind SUMO's pose so CARLA's step carries them onto it. Then CARLA's velocity of
    each is SUMO's, and a radar's radial velocity on one is SUMO's velocity along the line of sight."""
    with fake_sumo() as fc:
        import carla
        import traci
        with cosim(tmp_path / 'v', dt=0.0125) as cs:
            assert cs.k == 2
            cs.start()
            _step(cs, 40)
            sim = traci._S
            for k in range(4):                                    # two SUMO steps, two dead-reckoned ticks
                prev = {sid: cs.base[sid].rotation.yaw for sid in cs.sumo2carla if sid in cs.base}
                fc.LOG.clear()
                sumo_step = (cs.ticks + 1) % cs.k == 0
                cs.before_tick()
                targets = {e[1]: e[2] for e in fc.LOG if e[0] == 'target_w'}
                cs.w.tick()
                snap = cs.w.get_snapshot()
                for sid, cid in cs.sumo2carla.items():
                    a = cs.w.get_actor(cid)
                    assert a.physics is True
                    yaw = cs.base[sid].rotation.yaw                   # the heading at SUMO's last step
                    sp = sim.veh[sid]['speed']
                    v = snap.find(cid).get_velocity()
                    assert sp > 1.0 and (v.x, v.y, v.z) == pytest.approx(
                        (sp * math.cos(math.radians(yaw)), sp * math.sin(math.radians(yaw)), 0.0), abs=1e-9)
                    assert cs.velocity(cid, yaw) == pytest.approx((v.x, v.y, v.z), abs=1e-12)   # (the annotations')
                    # the yaw rate: SUMO's heading change over its last step (deg/s about CARLA's z)
                    if sumo_step and sid in prev:
                        assert targets[cid] == pytest.approx(_wrap(yaw - prev[sid]) / cs.sumo_dt, abs=1e-9)
                    assert a.get_angular_velocity().z == pytest.approx(cs.yaw_rate[sid], abs=1e-9)
                    # and it stands on the pose SUMO's (or the dead reckoning's) was, after the tick
                    if sumo_step:
                        vv = sim.veh[sid]
                        x, y_, yw = carla_pose_of(vv['x'], vv['y'], vv['angle'], sim.types[vv['type']]['length'])
                        tr = a.get_transform()
                        assert (tr.location.x, tr.location.y) == pytest.approx((x, y_), abs=1e-6)
                        assert _wrap(tr.rotation.yaw - yw) == pytest.approx(0.0, abs=1e-6)
                cs.after_tick(True)
            # a radar looking at one: its radial velocity is SUMO's velocity along the line of sight (fake CARLA's
            # radar takes it from the hit actor's velocity, as CARLA's)
            sid, cid = sorted(cs.sumo2carla.items())[0]
            car = cs.w.get_actor(cid).get_transform()
            yaw = math.radians(car.rotation.yaw)
            pos = carla.Location(car.location.x - 15 * math.cos(yaw) + 3 * math.sin(yaw),
                                 car.location.y - 15 * math.sin(yaw) - 3 * math.cos(yaw), car.location.z + 0.8)
            look = math.degrees(math.atan2(car.location.y - pos.y, car.location.x - pos.x))
            bp = cs.w.get_blueprint_library().find('sensor.other.radar')
            bp.set_attribute('horizontal_fov', '16')
            bp.set_attribute('vertical_fov', '10')
            bp.set_attribute('range', '40')
            bp.set_attribute('points_per_second', '8000')
            lib = cs.w.get_blueprint_library()
            mast = cs.w.spawn_actor(lib.find('static.prop.trafficcone01'),      # (the fake's sensors need a parent)
                                    carla.Transform(carla.Location(pos.x, pos.y, 0.0), carla.Rotation(0.0, look, 0.0)))
            radar = cs.w.spawn_actor(bp, carla.Transform(carla.Location(0.0, 0.0, pos.z + 0.5)), attach_to=mast)
            got = []
            radar.listen(got.append)
            _step(cs, 3)
            radar.stop()
            m = got[-1]
            a = cs.w.get_actor(cid)
            Mi = np.linalg.inv(np.array(a.get_transform().get_matrix()))
            bb = a.bounding_box
            sp = sim.veh[sid]['speed']
            yaw = math.radians(cs.base[sid].rotation.yaw)
            vt = np.array([sp * math.cos(yaw), sp * math.sin(yaw), 0.0])
            R = np.array(m.transform.get_matrix())
            on_car = []
            for d in m:
                u = R[:3, :3] @ np.array([math.cos(d.altitude) * math.cos(d.azimuth),
                                          math.cos(d.altitude) * math.sin(d.azimuth), math.sin(d.altitude)])
                q = Mi @ np.r_[R[:3, 3] + u * d.depth, 1.0]
                if np.all(np.abs(q[:3] - [bb.location.x, bb.location.y, bb.location.z]) <=
                          [bb.extent.x + 0.05, bb.extent.y + 0.05, bb.extent.z + 0.05]):
                    on_car.append((d.velocity, float(vt @ u)))
            assert len(on_car) >= 3 and min(abs(e) for _, e in on_car) > 1.0
            assert [v for v, _ in on_car] == pytest.approx([e for _, e in on_car], abs=1e-9)
            radar.destroy()
            mast.destroy()


def test_arrived_vehicles_are_destroyed_in_carla(tmp_path):
    """A user net of short edges: SUMO's vehicles end their trips within seconds and leave CARLA."""
    sys.path.insert(0, os.path.join(FAKE_ROOT, 'Co-Simulation', 'Sumo', 'util'))
    try:
        import netconvert_carla as NC
        text = NC.net_xml(False, 'short')
    finally:
        sys.path.pop(0)
        sys.modules.pop('netconvert_carla', None)
    text = text.replace('-40.00', '80.00').replace('240.00', '120.00')
    net = tmp_path / 'short.net.xml'
    net.write_text(text)
    with fake_sumo() as fc:
        with cosim(tmp_path / 'a', {'net': str(net)}) as cs:
            cs.start()
            assert cs.info['net']['kind'] == 'user'
            _step(cs, 200)
            rep = cs.report()
            gone = cs.events['gone']
            assert rep['arrived'] >= 1 and len(gone) == rep['arrived']
            assert set(gone) <= {e[1] for e in fc.LOG if e[0] == 'destroy'}
            assert all(cs.w.get_actor(a) is None for a in gone)
            arrived = [a for a in rep['actors'] if a['arrived_t_s'] is not None]
            assert len(arrived) == rep['arrived'] and all(a['arrived_t_s'] > a['spawned_t_s'] for a in arrived)


def test_net_bundled_generated_cached_and_the_users(tmp_path):
    with fake_sumo():
        with cosim(tmp_path / 'b', town='Town01') as cs:      # CARLA bundles a net for Town01: no lights in it
            cs.start()
            n = cs.info['net']
            assert n['kind'] == 'bundled' and n['file'] == 'Town01.net.xml' and n['cached'] is None
            assert cs.common_tls == [] and any('no traffic lights' in x for x in cs.notes_seen)
        calls = tmp_path / 'cache' / 'netconvert_calls.txt'
        got = []
        for k in range(2):                                     # Town10HD: generated once, then taken from the cache
            with cosim(tmp_path / f'g{k}') as cs:
                cs.start()
                got.append(dict(cs.info['net']))
        assert got[0]['kind'] == got[1]['kind'] == 'generated'
        assert (got[0]['cached'], got[1]['cached']) == (False, True)
        assert got[0]['sha256'] == got[1]['sha256'] and got[0]['opendrive_sha256'] == got[1]['opendrive_sha256']
        assert len(calls.read_text().splitlines()) == 1
        assert not [f for f in os.listdir(tmp_path / 'cache') if f.endswith(('.xodr', '.tmp.net.xml'))]
        with cosim(tmp_path / 'b2', {'net': 'generate'}, town='Town01') as cs:   # a bundled town generated on request
            cs.start()
            assert cs.info['net']['kind'] == 'generated' and cs.common_tls == ['1000']
        with cosim(tmp_path / 'b3', town='Town04') as cs:      # a town CARLA would bundle, missing here: generated
            cs.start()
            assert cs.info['net']['kind'] == 'generated'
        # the user's .net.xml, then the user's .sumocfg with its own demand
        user = tmp_path / 'mine.net.xml'
        user.write_bytes((tmp_path / 'cache' / got[0]['file']).read_bytes())
        with cosim(tmp_path / 'u', {'net': str(user)}) as cs:
            cs.start()
            assert cs.info['net']['kind'] == 'user' and cs.info['net']['file'] == 'mine.net.xml'
            assert cs.info['demand']['mode'] == 'count'
        (tmp_path / 'my.rou.xml').write_text(
            '<routes><vType id="vehicle.audi.a2" vClass="passenger" length="3.7" width="1.8" height="1.5"/>'
            '<trip id="mine0" depart="0" from="east" to="east" type="vehicle.audi.a2"/></routes>')
        (tmp_path / 'my.sumocfg').write_text('<configuration><input><net-file value="mine.net.xml"/>'
                                             '<route-files value="my.rou.xml"/></input></configuration>')
        with cosim(tmp_path / 'c', {'net': str(tmp_path / 'my.sumocfg')}) as cs:
            cs.start()
            assert cs.info['net']['kind'] == 'user_config' and cs.info['demand']['mode'] == 'user'
            assert cs.info['demand']['config'] == 'my.sumocfg'
            _step(cs, 3)
            assert [a['sumo_id'] for a in cs.report()['actors']] == ['mine0']


def test_sumo_crash_fails_the_run_with_its_reason_and_tidies_up(tmp_path, monkeypatch):
    with fake_sumo() as fc:
        b, doc = _bridge(tmp_path / 'x', cams=())
        rig_json = os.path.join(str(tmp_path / 'x'), 'rig', 'sim_rig.json')
        monkeypatch.setenv('FAKE_SUMO_CRASH_AT_STEP', '20')
        code = _main([rig_json, str(tmp_path / 'crash'), '--carla-root', FAKE_ROOT, '--sumo-cache',
                      str(tmp_path / 'cache')])
        rep = json.load(open(tmp_path / 'crash' / 'run.json'))
        assert code == 2 and rep['status'] == 'failed'
        assert rep['error'].startswith('RuntimeError: SUMO stopped stepping at') and 'exit 139' in rep['error']
        assert 'Segmentation fault' in rep['error']                # SUMO's own log, quoted
        w = fc.STATE['world']
        assert not w._actors and w.applied[-1] == (False, None)
        assert not os.path.exists(tmp_path / 'crash' / 'nuscenes')
        assert sorted(os.listdir(tmp_path / 'crash' / 'sumo')) == ['sumo.log']     # its log stays, for the reason
    with fake_sumo() as fc:                                    # a SUMO that does not start at all
        monkeypatch.setenv('FAKE_SUMO_FAIL', '1')
        b, doc = _bridge(tmp_path / 'y', cams=())
        with pytest.raises(RuntimeError, match='SUMO did not start .exit 1.: Error: fake SUMO told to fail'):
            b.run()
        assert not fc.STATE['world']._actors and fc.STATE['world'].applied[-1] == (False, None)


def test_cancel_tidies_up_sumo_and_carla(tmp_path):
    from vantage.sim import carla_bridge as CB
    with fake_sumo() as fc:
        import traci
        n = {'k': 0}

        def cancel():
            n['k'] += 1
            return n['k'] > 60
        b, doc = _bridge(tmp_path / 'c', {'sumo': {'tls': 'sumo'}}, cams=(), cancel=cancel)
        with pytest.raises(CB.Cancelled):
            b.run()
        b.rec.cleanup()
        assert b.cosim.started and b.cosim.proc.poll() is not None          # SUMO ended
        assert traci.LOG[-1] == ('close',) and traci._S is None
        w = fc.STATE['world']
        assert not w._actors and w.applied[-1] == (False, None)
        assert [e for e in fc.LOG if e[0] == 'tl_freeze'][-1] == ('tl_freeze', 50, False)   # CARLA's lights back
        assert sorted(os.listdir(tmp_path / 'c' / 'sumo')) == ['sumo.log']


def test_sumo_missing_is_refused_before_carla_is_touched(tmp_path, monkeypatch):
    """No SUMO in CARLA's Python: the bridge refuses before it connects to CARLA, run.json says what to install."""
    from test_sim import _fake
    from vantage.sim import carla_bridge as CB
    from vantage.sim import sumo_cosim as SUMO
    with fake_sumo() as fc:
        b, doc = _bridge(tmp_path / 'm', cams=())
    rig_json = os.path.join(str(tmp_path / 'm'), 'rig', 'sim_rig.json')
    fc = _fake()
    fc.reset()
    monkeypatch.setitem(sys.modules, 'carla', fc)
    for m in ('traci', 'sumolib', 'sumo'):
        monkeypatch.setitem(sys.modules, m, None)                # importing them fails, as in a Python without SUMO
    from test_sim import _args
    a = _args(doc, str(tmp_path / 'm'), carla_root=FAKE_ROOT)
    with pytest.raises(SUMO.SumoMissing, match='uv pip install --python .* eclipse-sumo==1.23.1'):
        CB.Bridge(fc, doc, {}, str(tmp_path / 'm'), a).run()
    assert fc.LOG == []                                          # CARLA never touched
    code = _main([rig_json, str(tmp_path / 'm2'), '--carla-root', FAKE_ROOT])
    rep = json.load(open(tmp_path / 'm2' / 'run.json'))
    assert code == 2 and rep['status'] == 'failed' and 'SumoMissing' in rep['error'] and 'traci' in rep['error']
    assert fc.LOG == []


def test_carla_traffic_writes_what_it_wrote_before(tmp_path):
    """traffic_source carla: no traffic block in the rig or the run, the traffic manager's vehicles, SUMO never
    started (the outputs were compared file by file with the code before SUMO: identical; see the note)."""
    from test_sim import _mini_run
    r = _mini_run(tmp_path / 'c', {'duration_s': 0.5, 'traffic': 3, 'sumo': {'net': 'generate', 'tls': 'sumo'}},
                  cams=())
    assert 'traffic_source' not in r['rig'] and 'traffic_source' not in r['report']
    assert r['report']['traffic'] == 3 and r['bridge'].cosim is None
    assert not os.path.exists(tmp_path / 'c' / 'sumo')
    assert all(e[3].get('role_name') != 'vantage_sumo' for e in r['log'] if e[0] == 'spawn')


# ======================================================================= CARLA's own package, when it is here
def _real_cosim():
    from vantage.sim import server as SV
    from vantage.sim import sumo_cosim as SUMO
    roots = [os.environ.get('VANTAGE_CARLA_ROOT', '')] + SV.candidate_roots()
    for r in roots:
        d = SUMO.find_cosim(r) if r else ''
        if d and os.path.isfile(os.path.join(d, 'sumo_integration', 'bridge_helper.py')) and \
                os.path.abspath(d) != os.path.abspath(os.path.join(FAKE_ROOT, 'Co-Simulation', 'Sumo')):
            return d
    return ''


def test_with_carlas_real_sumo_integration(tmp_path):
    """The co-simulation on CARLA's real sumo_integration package (its BridgeHelper, SumoSimulation, CarlaSimulation)
    over the fake traci and the fake carla: the same spawns, poses and report. Needs lxml (CARLA's package reads the
    configuration with it) and a CARLA folder."""
    real = _real_cosim()
    if not real:
        pytest.skip('no CARLA folder with Co-Simulation/Sumo here')
    pytest.importorskip('lxml')
    root = os.path.dirname(os.path.dirname(real))
    with fake_sumo() as fc:
        # the fake net and vTypes in a copy of the real folder's layout: CARLA's own code, the fake's data
        import shutil
        tree = tmp_path / 'carla' / 'Co-Simulation' / 'Sumo'
        shutil.copytree(os.path.join(real, 'sumo_integration'), tree / 'sumo_integration',
                        ignore=shutil.ignore_patterns('__pycache__'))
        for d in ('data', 'examples', 'util'):
            shutil.copytree(os.path.join(FAKE_ROOT, 'Co-Simulation', 'Sumo', d), tree / d,
                            ignore=shutil.ignore_patterns('__pycache__'))
        with cosim(tmp_path / 'r', carla_root=str(tmp_path / 'carla')) as cs:
            assert root
            cs.start()
            _step(cs, 30)
            rep = cs.report()
            assert rep['vehicles_spawned'] >= 1 and rep['mirrored'][0]['sumo_id'] == 'carla0'
            import traci
            sim = traci._S
            for sid, cid in cs.sumo2carla.items():
                cs.before_tick()
                cs.w.tick()                                       # (put a step behind, the tick carries it there)
                v = sim.veh.get(sid)
                if v is None:
                    continue
                tr = cs.w.get_actor(cid).get_transform()
                x, y, yaw = carla_pose_of(v['x'], v['y'], v['angle'], sim.types[v['type']]['length'])
                assert (tr.location.x, tr.location.y) == pytest.approx((x, y), abs=1e-4)
                cs.after_tick(True)


# ======================================================================= the app side
def test_the_app_asks_carlas_python_for_sumo_and_caches_the_answer(tmp_path, monkeypatch):
    from vantage.sim import server as SV
    no = SV.check_sumo(sys.executable, refresh=True)           # this Python has no SUMO
    assert not no['ok'] and {'sumo', 'traci', 'sumolib'} <= set(no['missing'])
    assert no['install'] == f'uv pip install --python {sys.executable} eclipse-sumo==1.23.1 traci==1.23.1 ' \
                           'sumolib==1.23.1' and no['install'] in no['text']
    # a "CARLA Python" that sees the fake SUMO packages (a wrapper: check_sumo runs it with -I -c CODE)
    venv = tmp_path / 'venv'
    (venv / 'bin').mkdir(parents=True)
    (venv / 'lib' / 'python3.12' / 'site-packages').mkdir(parents=True)
    py = venv / 'bin' / 'python'
    py.write_text('#!/bin/sh\nexec "%s" -c "import sys; sys.path.insert(0, %r); exec(compile(sys.argv[1], \'<c>\', '
                  '\'exec\'))" "$3"\n' % (sys.executable, FAKE_SUMO))
    py.chmod(0o755)
    calls = []
    real = SV.subprocess.run
    monkeypatch.setattr(SV.subprocess, 'run', lambda *a, **k: calls.append(a) or real(*a, **k))
    yes = SV.check_sumo(str(py))
    assert yes['ok'] and yes['version'] == '1.23.1' and yes['binary'].endswith(os.path.join('bin', 'sumo')), yes
    assert 'SUMO 1.23.1' in yes['text']
    assert SV.check_sumo(str(py)) == yes and len(calls) == 1         # cached
    os.utime(venv / 'lib' / 'python3.12' / 'site-packages', ns=(1, 1))   # something was installed: asked again
    SV.check_sumo(str(py))
    assert len(calls) == 2
    assert 'sumo' not in sys.modules and 'traci' not in sys.modules     # never imported into Vantage
    assert SV.sumo_bundled_towns(FAKE_ROOT) == ['Town01'] and SV.cosim_folder(FAKE_ROOT)


def test_page_traffic_section_undo_status_and_refusal(app, win, tmp_path, monkeypatch):
    from test_gui import pump
    from test_sim import _open
    from vantage.sim import server as SV
    from vantage.ui.workspaces.sim_ws import SimPage
    p = win.store.project
    v = next(v for v in p.variants if v.sensors)

    def activate(pj):
        pj.active_uid = v.uid
    win.store.edit('variants', activate, 'data rig')
    pump(app, 0.2)
    answer = {'ok': False, 'missing': ['sumo', 'traci'], 'install': SV.sumo_install_command('/opt/carla/py'),
              'text': 'missing'}
    monkeypatch.setattr(SV, 'check_sumo', lambda py, timeout=60.0, refresh=False: dict(answer))
    pg = _open(win, 'sim', SimPage)
    pg.python = '/opt/carla/py'
    assert not pg.sumo_box.isVisible() and pg.settings.traffic_source == 'carla'
    pg.t_source.buttons['sumo'].click()
    pump(app, 0.3)
    assert pg.settings.traffic_source == 'sumo' and pg.sumo_box.isVisible()
    assert pg.sumo_install.isVisible() and pg.sumo_install.text().startswith('uv pip install --python /opt/carla/py')
    assert 'Eclipse SUMO drives the traffic' in pg.t_about.text()
    # net, demand, period, lights: each an undo step
    pg.sumo_net.setCurrentIndex(pg.sumo_net.findData('generate'))
    pg.sumo_demand.buttons['period'].click()
    pg.sumo_period.set_value(5.0)
    pg.sumo_period.committed.emit(5.0)
    pg.sumo_tls.buttons['sumo'].click()
    pump(app, 0.2)
    assert pg.settings.sumo == dict(pg.settings.sumo, net='generate', demand='period', period_s=5.0, tls='sumo')
    assert pg.sumo_period.isEnabled() and 'generated from its OpenDRIVE' in pg.sumo_net_note.text()
    assert '--traffic-source sumo --sumo-net generate' in pg.command_line()
    for _ in range(4):
        win.store.undo.undo()
    pump(app, 0.2)
    assert pg.settings.sumo == SR_default() and pg.settings.traffic_source == 'sumo'
    win.store.undo.undo()
    pump(app, 0.2)
    assert pg.settings.traffic_source == 'carla' and not pg.sumo_box.isVisible()
    win.store.undo.redo()
    pump(app, 0.2)
    # a user net that is not there, and SUMO missing: the run is refused before CARLA is asked anything
    asked = []
    monkeypatch.setattr(SV, 'port_open', lambda *a, **k: asked.append(a) or True)
    assert not pg.run(str(tmp_path / 'never'))
    assert 'SUMO is not in CARLA\'s Python' in pg.g_status.text() and 'uv pip install' in pg.g_status.text()
    assert not asked and not os.path.exists(tmp_path / 'never')
    answer.update(ok=True, missing=[], text='SUMO 1.23.1 in CARLA\'s Python (traci 1.23.1)')
    pg.set_sumo('net', str(tmp_path / 'gone.net.xml'))
    pump(app, 0.2)
    assert 'does not exist' in pg.sumo_net_note.text()
    assert not pg.run(str(tmp_path / 'never')) and 'does not exist' in pg.g_status.text() and not asked
    pg._sumo = None
    pg.check_sumo()
    _wait(app, lambda: pg._sumo is not None and pg._sumo.get('ok'))
    assert not pg.sumo_install.isVisible() and 'SUMO 1.23.1' in pg.sumo_status.text.text()
    # a finished SUMO drive's report on the page
    doc = {'scene': {}, 'traffic_source': {'sumo_version': '1.23.1', 'net': {'kind': 'bundled', 'file': 'Town04.net.xml'},
                                           'vehicles_spawned': 31, 'vehicles_in_carla': {'mean': 29.5, 'peak': 30},
                                           'teleports': 0, 'collisions': 1, 'tls': {'owner': 'carla'},
                                           'timing': {'net_s': 0.1, 'step_s': 0.4}}}
    assert 'Traffic: SUMO 1.23.1 on the bundled net Town04.net.xml · 31 vehicles spawned, 29.5 on average' in \
        pg.report_text(doc)


from test_gui import app, win  # noqa: E402,F401  (fixtures)


def SR_default():
    from vantage.sim import sumo_cosim as SUMO
    return SUMO.default()


def _wait(app, cond, timeout=30.0):
    import time
    from test_gui import pump
    t0 = time.time()
    while not cond():
        pump(app, 0.05)
        assert time.time() - t0 < timeout
