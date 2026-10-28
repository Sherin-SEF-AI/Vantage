"""Critical targets as CARLA scenes (core.critical_carla and its page): the actor chosen for each kind, checked by brute
force against the measured tables; scenes that keep targets a camera would see overlap apart; the measurement on CARLA's
ground truth against an analytic pinhole frame (an occluder hiding the lower part, the ground's tolerance); the
explanations and the Wilson interval against published values; the whole comparison through the bridge against the
fake carla (sizes, agreement, pixels against the study's, a camera inside CARLA's car explained, export, re-analysis,
cancel and resume); and the page driven like a user."""
import importlib.util
import json
import math
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
FAKE_DIR = os.path.join(HERE, 'fake_carla')


def _fake():
    name = 'fake_carla_for_tests'
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, os.path.join(FAKE_DIR, 'carla', '__init__.py'))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _project(labels, cams=('F', 'R', 'ML'), actors=None, extra_targets=()):
    """The sample's data rig with only these cameras, its lidar on a mast (no self-hits), no radars; the critical
    targets from these preset labels (stable uids), small images."""
    from vantage.core import critical as CR
    from vantage.core import samples
    p = samples.tigor_sample_project()
    v = next(v for v in p.variants if v.sensors)
    p.active_uid = v.uid
    v.cameras[:] = [c for c in v.cameras if c.tag in cams]
    v.sensors[:] = [s for s in v.sensors if s.kind == 'lidar']
    v.sensors[0].z = 2.0
    targets = [CR.preset(p, lab) for lab in labels] + list(extra_targets)
    for i, t in enumerate(targets):
        t['uid'] = f't{i}'
    p.analysis.critical = CR.CriticalSettings.from_dict({'targets': targets}).to_dict()
    p.analysis.sim = {'image_scale': 0.08, 'warmup_s': 0.25}
    p.analysis.critical_carla = {'image_scale': 0.08, 'spawn_points': [3], 'frames': 2, 'actors': actors or {}}
    return p


def _runner(calls):
    from vantage.sim import carla_bridge as CB
    fc = _fake()

    def runner(rig_json, out, progress, cancel):
        calls.append(out)
        fc.reset()
        doc, remaps = CB.load_rig(rig_json)
        a = CB.resolve_args(CB.parser().parse_args([rig_json, out]), doc)
        CB.Bridge(fc, doc, remaps, out, a, None, cancel).run()
    return runner


# ======================================================================= actors
def _dang(a, b) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def test_actor_for_each_kind_is_the_nearest_in_size():
    from vantage.core import critical as CR
    from vantage.core import critical_carla as CC
    from vantage.core import samples, simrig
    p = samples.tigor_sample_project()
    box = CR.vehicle_box(p)
    centre = (box['mid_x'], 0.0)

    def score(h, w, H, W):
        return abs(h - H) / H + 0.5 * abs(w - W) / W
    # a child: the child walker whose rendered height is nearest, facing the vehicle
    t = CR.preset(p, 'Child 2 m ahead')
    a = CC.actor_for(p, t)
    best = min((r[2], r[3], k) for k, r in CC.WALKERS.items() if r[0] == 'child')
    want = min((score(r[2], r[3], t['height'], t['width']), k) for k, r in CC.WALKERS.items() if r[0] == 'child')[1]
    assert a['spawnable'] and a['family'] == 'walker' and a['blueprint'] == f'walker.pedestrian.{want}' == \
        'walker.pedestrian.0009' and best[0] == pytest.approx(1.095)
    assert a['kind'] == 'blueprint:walker.pedestrian.0009' and a['expected']['rendered_height'] == pytest.approx(1.095)
    assert a['pose']['x'] == pytest.approx(t['x']) and a['pose']['z'] == 0.0
    assert _dang(a['pose']['yaw'], math.degrees(math.atan2(centre[1] - t['y'], centre[0] - t['x']))) < 1e-6
    b = CC.actor_for(p, CR.preset(p, 'Child beside the front door'))
    tb = CR.preset(p, 'Child beside the front door')
    assert _dang(b['pose']['yaw'], math.degrees(math.atan2(-tb['y'], centre[0] - tb['x']))) < 1e-3
    # an adult for a pedestrian
    tp = CR.preset(p, 'Pedestrian 10 m ahead')
    ap = CC.actor_for(p, tp)
    want = min((score(r[2], r[3], tp['height'], tp['width']), k) for k, r in CC.WALKERS.items() if r[0] == 'adult')[1]
    assert ap['blueprint'] == f'walker.pedestrian.{want}' and CC.WALKERS[want][0] == 'adult'
    # a car: the car of CARLA's table nearest in height and width (92 % of the box's: the mirrors)
    tc = CR.preset(p, 'Car 30 m ahead')
    ac = CC.actor_for(p, tc)
    want = min((score(r[3], 0.92 * r[2], tc['height'], tc['width']), k) for k, r in simrig.CARLA_VEHICLES.items()
               if r[0] == 'car')[1]
    assert ac['blueprint'] == want and ac['family'] == 'vehicle' and _dang(ac['pose']['yaw'], tc['yaw']) < 1e-6
    # a sign: a board stands in, its board's bottom at the sign's height, its face to the vehicle
    ts = CR.preset(p, 'Sign 15 m ahead, 3 m left')
    asg = CC.actor_for(p, ts)
    name = asg['blueprint'].split('static.prop.')[1]
    bd = CC.BOARDS[name]
    assert asg['stand_in'] and asg['family'] == 'board' and asg['board'] == bd
    assert asg['pose']['z'] == pytest.approx(ts['base'] - bd['z0'])
    bearing = math.degrees(math.atan2(centre[1] - ts['y'], centre[0] - ts['x']))
    normal = asg['pose']['yaw'] + 90.0                          # the board spans x, its face looks along y
    assert abs(math.cos(math.radians(normal - bearing))) == pytest.approx(1.0, abs=1e-6)
    # a traffic light: not spawnable, unless a stand-in prop is asked for (at the head's height)
    tl = CR.preset(p, 'Overhead traffic light 15 m ahead')
    a0 = CC.actor_for(p, tl)
    assert not a0['spawnable'] and 'traffic-light blueprint' in a0['why']
    a1 = CC.actor_for(p, tl, 'standin')
    want = min((score(r[2], r[0], tl['height'], tl['width']), k) for k, r in CC.PROPS.items())[1]
    assert a1['spawnable'] and a1['stand_in'] and a1['blueprint'] == f'static.prop.{want}'
    assert a1['pose']['z'] == pytest.approx(tl['base'])
    # kerb, lane: part of the road
    for lab in ('Kerb by the right front wheel', 'Lane marking on a 50 m curve'):
        a2 = CC.actor_for(p, CR.preset(p, lab))
        assert not a2['spawnable'] and 'road' in a2['why']
    # objects by their name, else the prop nearest in size
    cy = CR.clean_target({'kind': 'object', 'name': 'Cyclist crossing', 'x': 8.0, 'y': -2.0, 'height': 1.7,
                          'width': 0.6})
    acy = CC.actor_for(p, cy)
    want = min((score(r[4], r[2], 1.7, 0.6), k) for k, r in CC.TWO_WHEELERS.items() if r[0] == 'bicycle')[1]
    assert acy['family'] == 'two_wheeler' and acy['blueprint'] == want
    cone = CC.actor_for(p, CR.clean_target({'kind': 'object', 'name': 'Traffic cone', 'x': 6.0, 'y': 0.0,
                                            'height': 0.6, 'width': 0.35}))
    assert cone['blueprint'] == 'static.prop.constructioncone'
    box_t = CR.clean_target({'kind': 'object', 'name': 'Bollard', 'x': 6.0, 'y': 1.0, 'height': 1.0, 'width': 0.5})
    want = min((score(r[2], r[0], 1.0, 0.5), k) for k, r in CC.PROPS.items())[1]
    assert CC.actor_for(p, box_t)['blueprint'] == f'static.prop.{want}'
    # choices: none, a blueprint
    assert not CC.actor_for(p, t, 'none')['spawnable']
    teen = CC.actor_for(p, tp, 'blueprint:walker.pedestrian.0050')
    assert teen['family'] == 'walker' and teen['expected']['rendered_height'] == pytest.approx(1.203)
    assert CC.actor_for(p, tc, 'blueprint:vehicle.nissan.micra')['expected']['height'] == pytest.approx(
        simrig.CARLA_VEHICLES['vehicle.nissan.micra'][3])


def test_settings_tolerant_and_clamped():
    from vantage.core import critical_carla as CC
    s = CC.CriticalCarlaSettings.from_dict({'spawn_points': '3, x, -5, 3 ;12', 'duration_s': 99, 'image_scale': -1,
                                            'frames': 2.6, 'seed': 'q', 'weather': 'Snowstorm', 'group': 'yes',
                                            'vehicle': 'tank', 'actors': {'a': 'standin', 'b': 'blueprint:x y',
                                                                          'c': 'blueprint:walker.pedestrian.0009',
                                                                          'd': 'auto', 'e': 7}})
    assert s.spawn_points == [3, -1, 12] and s.duration_s == 10.0 and s.image_scale == 0.1 and s.frames == 3
    assert s.seed == 1 and s.weather == 'ClearNoon' and s.group is True and s.vehicle == 'auto'
    assert s.actors == {'a': 'standin', 'c': 'blueprint:walker.pedestrian.0009'}
    assert CC.CriticalCarlaSettings.from_dict(s.to_dict()).to_dict() == s.to_dict()
    assert CC.CriticalCarlaSettings.from_dict(None).spawn_points == [-1]
    assert CC.CriticalCarlaSettings.from_dict({'spawn_points': list(range(20))}).spawn_points == list(range(8))


# ======================================================================= scenes
def test_scenes_keep_targets_a_camera_sees_overlap_apart():
    from vantage.core import critical as CR
    from vantage.core import critical_carla as CC
    p = _project(['Child 2 m ahead', 'Pedestrian 10 m ahead', 'Car 30 m ahead', 'Child 1 m behind', 'Kerb by the '
                  'right front wheel'], cams=('F', 'R', 'ML', 'MR', 'QL', 'QR'))
    pl = CC.plan(p, CC.CriticalCarlaSettings.from_dict({'spawn_points': [-1, 12]}))
    names = {r['uid']: r['name'] for r in pl.targets}
    ahead = {'Child 2 m ahead', 'Pedestrian 10 m ahead', 'Car 30 m ahead'}
    for g in pl.groups:                                  # the three on F's axis never share a scene
        assert len(ahead & {names[u] for u in g}) <= 1
    assert sum(len(g) for g in pl.groups) == 4 and 'Kerb by the right front wheel' not in \
        {names[u] for g in pl.groups for u in g}
    assert [d['key'] for d in pl.drives][:2] == ['scene1-spauto', 'scene1-sp12'] and \
        len(pl.drives) == 2 * len(pl.groups)
    # independently: F sees the child and the pedestrian overlap (their rectangles' image boxes)
    v = p.variant
    cm = CR._Cam(p, next(c for c in v.cameras if c.tag == 'F'))
    ta, tb = (CR.settings_of(p).target(u) for u in ('t0', 't1'))
    import itertools
    boxes = []
    for t in (ta, tb):
        sh = CR.shape(t, cm.pos)
        u, vv, _, ok = cm.intr.project_dirs((sh.flat - cm.pos) @ (CR.BODY_TO_OPTICAL @ cm.R.T).T)
        boxes.append((u[ok].min(), vv[ok].min(), u[ok].max(), vv[ok].max()))
    (a, b), = itertools.combinations(boxes, 2)
    assert a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]
    # the child behind shares a scene with one of them (no camera sees both)
    g_behind = next(g for g in pl.groups if 't3' in g)
    assert len(g_behind) == 2
    # one target per drive when asked; targets within a metre of each other never together
    pl1 = CC.plan(p, CC.CriticalCarlaSettings.from_dict({'group': False}))
    assert [len(g) for g in pl1.groups] == [1, 1, 1, 1]
    near = CR.clean_target(dict(ta, uid='near', name='Close by', y=0.6))
    p.analysis.critical = CR.CriticalSettings.from_dict({'targets': [ta, near]}).to_dict()
    assert len(CC.plan(p).groups) == 2
    acts = pl.actors_json(0)
    assert all(a['kind'].startswith('blueprint:') and set(a['pose']) == {'x', 'y', 'z', 'yaw'} for a in acts)
    assert {a['id'] for a in acts} == set(pl.groups[0])


# ======================================================================= the measurement
def _pinhole(W=400, H=300, f=300.0, z=1.0):
    K = [[f, 0.0, W / 2 - 0.5], [0.0, f, H / 2 - 0.5], [0.0, 0.0, 1.0]]
    return {'tag': 'X', 'uid': 'x', 'pose_vehicle': {'x': 0.0, 'y': 0.0, 'z': z, 'yaw': 0.0, 'pitch': 0.0,
                                                     'roll': 0.0},
            'lens': {'model': 'pinhole', 'K': K, 'max_theta_deg': 80.0, 'image_size': [W, H]}}


def test_measurement_on_an_analytic_pinhole_frame():
    """A 1.5 m × 0.6 m actor 10 m ahead of a level pinhole camera 1 m up (f = 300 px): its rows run from
    v = 150 − 300·0.5/10 = 135 to 180. Something 3 m away (class ego) covers the rows from 167 down (z below
    1 − 17·10/300 = 0.433 m): seen 1 − 0.433/1.5, the actor's own pixels from 0.433 to 1.5 m = 300·1.067/10 = 32 px."""
    from vantage.core import critical_carla as CC
    from vantage.core import groundtruth as GT
    cam = _pinhole()
    W, H = 400, 300
    inst = np.zeros((H, W), np.uint16)
    sem = np.full((H, W), 11, np.uint8)                         # sky
    rng = np.zeros((H, W), np.float32)
    sem[150:] = 1                                               # road beyond, 50 m
    rng[150:] = 50.0
    inst[135:180, 191:209] = 7                                  # the actor: u = 200 ∓ 300·0.3/10
    sem[135:180, 191:209] = 12
    rng[135:180, 191:209] = 10.0
    ego = np.zeros((H, W), bool)
    ego[167:] = True
    inst[ego], sem[ego], rng[ego] = 0, GT.EGO_CLASS, 3.0
    rect = {'x': 10.0, 'y': 0.0, 'base': 0.0, 'height': 1.5, 'width': 0.6}
    m = CC.classify(cam, rect, inst, sem, rng, 7)
    z_cut = 1.0 - 17 * 10 / 300
    assert m['in_view'] == 1.0
    assert m['seen'] == pytest.approx(1 - z_cut / 1.5, abs=0.02)
    assert m['hidden'] == {'ego': pytest.approx(z_cut / 1.5, abs=0.02)}
    assert m['px_image'] == pytest.approx(300 * (1.5 - z_cut) / 10, abs=1.0)
    assert m['z_own'][0] == pytest.approx(z_cut, abs=0.03) and m['z_own'][1] == pytest.approx(1.5, abs=0.02)
    # not in the frame's objects: no pixels of its own, the line of sight the same
    m0 = CC.classify(cam, rect, inst, sem, rng, None)
    assert m0['px_image'] == 0.0 and m0['seen'] == pytest.approx(m['seen'])
    # the ground a little nearer than a node (25 cm) does not hide it; a pole as near does
    inst2, sem2, rng2 = inst.copy(), sem.copy(), rng.copy()
    inst2[ego], sem2[ego], rng2[ego] = 7, 12, 10.0               # no occluder now
    feet = np.zeros((H, W), bool)
    feet[176:180, 191:209] = True
    d_feet = math.hypot(10.0, 0.95)
    inst2[feet], sem2[feet], rng2[feet] = 0, 1, d_feet - 0.25
    m1 = CC.classify(cam, rect, inst2, sem2, rng2, 7)
    assert m1['seen'] == 1.0 and m1['hidden'] == {}
    sem2[feet] = 6                                               # a pole
    m2 = CC.classify(cam, rect, inst2, sem2, rng2, 7)
    assert m2['hidden'].get('pole', 0) > 0.03 and m2['seen'] < 1.0
    # the verdict under the study's rule
    t = {'min_px': 24.0, 'min_share': 0.5, 'focus': True}
    assert CC.verdict({'seen': 0.7, 'px': 32.0, 'focus': True}, t) == (True, '')
    assert CC.verdict({'seen': 0.4, 'px': 32.0, 'focus': True}, t) == (False, 'seen')
    assert CC.verdict({'seen': 0.7, 'px': 20.0, 'focus': True}, t) == (False, 'pixels')
    assert CC.verdict({'seen': 0.7, 'px': 32.0, 'focus': False}, t) == (False, 'focus')
    assert CC.verdict({'seen': 0.7, 'px': 32.0, 'focus': False}, dict(t, focus=False)) == (True, '')
    # the box's width across the line of sight
    assert CC.apparent_width(4.0, 1.8, 0.0, (0.0, 0.0), (10.0, 0.0)) == pytest.approx(1.8)
    assert CC.apparent_width(4.0, 1.8, 90.0, (0.0, 0.0), (10.0, 0.0)) == pytest.approx(4.0)
    assert CC.apparent_width(4.0, 1.8, 45.0, (0.0, 0.0), (10.0, 0.0)) == pytest.approx((4.0 + 1.8) / math.sqrt(2))


def test_rect_nodes_are_the_study_shape():
    from vantage.core import critical as CR
    from vantage.core import critical_carla as CC
    t = CR.clean_target({'kind': 'child', 'x': 5.0, 'y': 1.0})
    cam = np.array([3.0, -0.5, 0.6])
    sh = CR.shape(t, cam)
    P = CC.rect_nodes(t['x'], t['y'], t['base'], t['height'], t['width'], cam, 5, 33)
    assert np.allclose(P, sh.nodes)
    loop = CC.rect_outline({'x': 5.0, 'y': 1.0, 'base': 0.0, 'height': 0.9, 'width': 0.3}, cam, 10)
    assert np.allclose(loop[0], sh.edges[0] - [0, 0, sh.edges[0][2]]) and loop[:, 2].max() == pytest.approx(0.9)


def test_explanations_and_the_wilson_interval():
    from vantage.core import critical_carla as CC
    t = {'min_px': 24.0, 'min_share': 0.5}
    base = {'camera': 'F', 'ego': 'vehicle.audi.a2', 'placement': {},
            'vantage_actor': {'px': 100.0, 'seen': 1.0, 'in_view': 1.0, 'hidden': {}, 'met': True},
            'carla': {'px': 60.0, 'seen': 0.6, 'in_view': 1.0, 'hidden': {'ego': 0.4}, 'met': False,
                      'camera_inside_m': 0.0, 'z_own': [0.4, 1.0], 'rect': {'height': 1.1}}}
    code, text, allr = CC.explain(base, t)
    assert code == 'body' and '40 %' in text and 'audi' in text
    code, text, _ = CC.explain(dict(base, carla=dict(base['carla'], camera_inside_m=0.02)), t)
    assert code == 'inside' and '2.0 cm inside' in text
    code, text, _ = CC.explain(dict(base, carla=dict(base['carla'], hidden={'pole': 0.45})), t)
    assert code == 'scene' and 'pole' in text
    code, _, _ = CC.explain(dict(base, carla=dict(base['carla'], hidden={}, seen=1.0, px=70.0)), t)
    assert code == 'shape'
    code, _, _ = CC.explain(dict(base, carla=dict(base['carla'], hidden={}, seen=1.0, px=23.0),
                                 vantage_actor=dict(base['vantage_actor'], px=25.0)), t)
    assert code == 'threshold'
    code, _, _ = CC.explain(dict(base, placement={'error_m': 0.4}, carla=dict(base['carla'], hidden={}, seen=1.0,
                                                                              px=100.0)), t)
    assert code == 'placement'
    # Wilson 95 % for 8 of 10: 0.4902–0.9433; 0 of 10: 0–0.2775 (Newcombe 1998, method 3)
    lo, hi = CC.wilson(8, 10)
    assert lo == pytest.approx(0.4902, abs=1e-4) and hi == pytest.approx(0.9433, abs=1e-4)
    lo, hi = CC.wilson(0, 10)
    assert lo == 0.0 and hi == pytest.approx(0.2775, abs=1e-4)
    assert CC.wilson(0, 0) is None


# ======================================================================= the whole comparison on the fake carla
LABELS = ['Child 2 m ahead', 'Child 1 m behind', 'Child beside the front door', 'Pedestrian 10 m ahead',
          'Car 30 m ahead', 'Overhead traffic light 15 m ahead', 'Kerb by the right front wheel']


@pytest.fixture(scope='module')
def fake_result(tmp_path_factory):
    from vantage.core import critical_carla as CC
    p = _project(LABELS, actors={'t4': 'blueprint:vehicle.nissan.micra',
                                 't5': 'blueprint:static.prop.trafficcone01'})
    out = str(tmp_path_factory.mktemp('cc') / 'drives')
    calls = []
    res = CC.run(p, out_dir=out, runner=_runner(calls))
    return {'p': p, 'res': res, 'out': out, 'calls': calls}


def test_comparison_against_the_fake_carla(fake_result, tmp_path):
    from vantage.core import critical_carla as CC
    res, p = fake_result['res'], fake_result['p']
    pl = res.plan
    assert len(fake_result['calls']) == len(pl.drives) == len(res.drives) >= 2
    assert all(not d['reused'] for d in res.drives) and all(d['spawn_point'] == 3 for d in res.drives)
    tg = {t['name']: t for t in res.targets}
    assert not tg['Kerb by the right front wheel']['actor']['spawnable']
    assert all(r['target'] != 'Kerb by the right front wheel' for r in res.rows)
    # every actor spawned where planned, its size as CARLA (the fake) reports it
    fc = _fake()
    for name in ('Child 2 m ahead', 'Pedestrian 10 m ahead', 'Car 30 m ahead'):
        sp = tg[name]['spawned']
        assert sp and sp[0]['actor_id'] is not None and sp[0]['error_m'] < 1e-3, sp
        bp = sp[0]['blueprint']
        assert bp == tg[name]['actor']['blueprint']
        if bp in fc.WALKERS:
            ext = fc.WALKERS[bp][1]
            assert sp[0]['size_lwh'] == pytest.approx([2 * e for e in ext], abs=1e-3)
    assert tg['Car 30 m ahead']['spawned'][0]['blueprint'] == 'vehicle.nissan.micra'
    assert tg['Overhead traffic light 15 m ahead']['actor']['stand_in'] is False      # chosen, not automatic
    # the rows: every spawned target in some camera; pixels where both see it against the study's at the actor's size
    cnt = res.rows_of(counted=True)
    assert {r['target'] for r in cnt} >= {'Child 2 m ahead', 'Child 1 m behind', 'Child beside the front door',
                                          'Pedestrian 10 m ahead'}
    big = [r for r in cnt if r['vantage_actor']['px'] >= 150]
    assert len(big) >= 4, [(r['target'], r['camera']) for r in big]
    for r in big:                                            # 1 image px = 12.5 sensor px at this scale
        c, va = r['carla'], r['vantage_actor']
        assert abs(c['px'] - va['px']) <= 0.1 * va['px'] + 25, (r['target'], r['camera'], c['px'], va['px'])
        assert abs(c['seen'] - va['seen']) <= 0.06, (r['target'], r['camera'], c['seen'], va['seen'])
    f_child = next(r for r in cnt if r['target'] == 'Child 2 m ahead' and r['camera'] == 'F')
    assert f_child['carla']['met'] and f_child['vantage_actor']['met'] and f_child['agree']
    assert f_child['carla']['rect']['height'] == pytest.approx(1.095)       # the walker's rendered height
    # agreement and its interval; every disagreement explained
    m = res.metrics
    assert m['pairs'] == len(cnt) and m['agree'] == sum(1 for r in cnt if r['agree'])
    assert m['agreement'] >= 0.8 and m['agreement_ci'] == CC.wilson(m['agree'], m['pairs'])
    for r in cnt:
        if not r['agree']:
            assert r['reason'] in CC.REASONS and r['why']
    assert m['px_error'] is not None and m['px_error']['n'] >= 3
    # the summary and the export
    s = CC.summary(res)
    assert s['camera_views_compared'] == m['pairs'] and s['stood_in_carla'] == 6 and json.dumps(s)
    paths = CC.export(res, p, str(tmp_path / 'export'))
    names = {os.path.relpath(x, str(tmp_path / 'export')) for x in paths}
    assert {'critical_carla.json', 'critical_carla.csv', 'critical_carla_targets.csv', 'critical_carla.png'} <= names
    crops = [n for n in names if n.startswith('crops' + os.sep)]
    assert len(crops) >= len(big)
    doc = json.load(open(tmp_path / 'export' / 'critical_carla.json'))
    assert doc['schema'] == CC.SCHEMA and doc['frame'] == CC.FRAME and len(doc['rows']) == len(res.rows)
    assert doc['metrics']['pairs'] == m['pairs'] and doc['targets'][0]['actor']['blueprint']
    import csv
    rows = list(csv.reader(open(tmp_path / 'export' / 'critical_carla.csv')))
    assert tuple(rows[0]) == CC.CSV_COLUMNS and len(rows) == len(res.rows) + 1
    import cv2
    im = cv2.imread(str(tmp_path / 'export' / 'critical_carla.png'))
    assert im is not None and im.shape[0] > 400
    crop = CC.crop(res, f_child)
    assert crop is not None and crop.shape[0] == 220 + 34
    # the drives compared again without CARLA: the same figures
    res2 = CC.analyse(p, CC.drives_in(fake_result['out']))
    assert res2.metrics['pairs'] == m['pairs'] and res2.metrics['agree'] == m['agree']
    assert [r['carla']['px'] for r in res2.rows_of(counted=True)] == pytest.approx(
        [r['carla']['px'] for r in cnt])



def test_drives_found_at_any_depth_and_keyed_apart(fake_result, tmp_path):
    """Open drives… on a folder of per-seed subfolders whose drives share their names (seed1/scene1-sp3,
    seed2/scene1-sp3): every drive is found at any depth, keyed by its path under the folder so no two collide, and
    compared as a drive of its own (twice the views of the flat folder, the same answers); hidden folders, a failed
    drive's own folder (run.json, no ground truth) and a link back up the tree are not searched; a flat folder keeps
    its drives' folder names as keys."""
    import shutil
    from vantage.core import critical_carla as CC
    p, res = fake_result['p'], fake_result['res']
    flat = CC.drives_in(fake_result['out'])
    assert len(flat) == len(res.drives) >= 2
    assert CC.drive_keys(flat) == [os.path.basename(d) for d in flat] == [d['key'] for d in res.drives]
    assert CC.drive_keys(flat[:1]) == [os.path.basename(flat[0])]
    nest = tmp_path / 'campaign'
    for seed in ('seed1', 'seed2'):
        for d in flat:
            shutil.copytree(d, nest / 'per-seed' / seed / os.path.basename(d))
    shutil.copytree(flat[0], nest / '.trash' / os.path.basename(flat[0]))            # hidden: skipped
    failed = nest / 'per-seed' / 'failed'                                             # a failed drive's folder
    failed.mkdir()
    (failed / 'run.json').write_text('{"schema": "vantage.sim_run/1", "status": "failed"}')
    shutil.copytree(flat[0], failed / 'inner')                                        # not looked into
    os.symlink(str(nest), str(nest / 'per-seed' / 'seed1' / 'up'))                    # a loop: followed once, no more
    found = CC.drives_in(str(nest))
    keys = CC.drive_keys(found)
    names = [os.path.basename(d) for d in flat]
    assert keys == [f'seed{s}/{n}' for s in (1, 2) for n in names]
    assert len({os.path.basename(d) for d in found}) < len(found)                     # folder names alone collide
    assert CC.drives_in(str(nest), depth=2) == []                                     # (two levels hold none)
    res2 = CC.analyse(p, found)
    assert [d['key'] for d in res2.drives] == keys
    m, m2 = res.metrics, res2.metrics
    assert m2['drives'] == 2 * m['drives'] and m2['pairs'] == 2 * m['pairs'] and m2['agree'] == 2 * m['agree']
    for k in keys:
        rows = [r for r in res2.rows_of(counted=True) if r['drive'] == k]
        base = [r for r in res.rows_of(counted=True) if r['drive'] == k.split('/', 1)[1]]
        assert [(r['uid'], r['camera'], r['carla']['px'], r['agree']) for r in rows] == \
            [(r['uid'], r['camera'], r['carla']['px'], r['agree']) for r in base]
    crops = [n for n, _ in CC.files(res2, p) if n.startswith('crops/')]
    assert len(crops) == len(set(crops)) and any('seed2-' in n for n in crops)
    # keys made unique even when given alike (two subfolders' drives handed over by their names)
    res3 = CC.analyse(p, [{'dir': d, 'key': os.path.basename(d), 'group': None, 'spawn_point': None}
                          for d in found])
    assert [d['key'] for d in res3.drives] == keys

def test_a_camera_inside_carla_s_car_is_explained(tmp_path):
    """The rear camera inside a CARLA truck (its box reaches 1.4 m behind the rear axle, the camera is 0.8 m behind
    it): the study sees the child 1 m behind, CARLA's camera sees the inside of the truck; the disagreement says so."""
    from vantage.core import critical_carla as CC
    p = _project(['Child 1 m behind'], cams=('R',))
    p.analysis.critical_carla = dict(p.analysis.critical_carla, vehicle='vehicle.carlamotors.carlacola')
    res = CC.run(p, out_dir=str(tmp_path / 'd'), runner=_runner([]))
    r = next(r for r in res.rows_of(counted=True) if r['camera'] == 'R')
    assert r['vantage_actor']['met'] and not r['carla']['met'] and not r['agree']
    assert r['reason'] == 'inside' and 'inside CARLA\'s car' in r['why'] and r['carla']['camera_inside_m'] > 0.005
    assert res.drives[0]['mounts_inside'] == ['R'] and res.drives[0]['ego'] == 'vehicle.carlamotors.carlacola'
    assert any(f['kind'] == 'disagree' and f['severity'] == 'serious' for f in res.findings)


def test_cancel_and_resume(tmp_path):
    from vantage.core import critical_carla as CC
    p = _project(['Child 2 m ahead', 'Pedestrian 10 m ahead'], cams=('F',))
    out = str(tmp_path / 'drives')
    calls, done = [], []
    inner = _runner(calls)

    def runner(*a):
        inner(*a)
        done.append(a[1])
    # cancelled during the first drive: nothing kept of it (the bridge removes what it wrote)
    res = CC.run(p, out_dir=out, runner=_runner(calls), cancel=lambda: len(calls) >= 1)
    assert res is None and len(calls) == 1 and not CC.is_drive(calls[0])
    # cancelled after the first drive: it is kept, and reused when the run is started again
    res = CC.run(p, out_dir=out, runner=runner, cancel=lambda: len(done) >= 1)
    assert res is None and len(calls) == 2 and CC.is_drive(done[0])
    res = CC.run(p, out_dir=out, runner=runner)
    assert res is not None and len(calls) == 3 and calls[2] != calls[1]
    assert [d['reused'] for d in res.drives] == [True, False]
    with pytest.raises(ValueError):
        CC.run(_project(['Kerb by the right front wheel']), out_dir=str(tmp_path / 'k'), runner=_runner([]))
    with pytest.raises(ValueError, match='no visible camera'):
        CC.run(_project(['Child 2 m ahead'], cams=()), out_dir=str(tmp_path / 'n'), runner=_runner([]))
    q = _project(['Child 2 m ahead'])
    q.analysis.critical = {}
    with pytest.raises(ValueError, match='no critical target'):
        CC.run(q, out_dir=str(tmp_path / 'e'), runner=_runner([]))


# ======================================================================= the page
from test_gui import app, win, pump, settle, _wait   # noqa: E402,F401  (fixtures)


def _open(win, name, cls):
    if name not in win._page_widgets:
        pg = cls(win.ctx)
        win.pages.addWidget(pg)
        win._page_widgets[name] = pg
    win.set_page(name)
    return win._page_widgets[name]


def test_page_runs_shows_crops_exports_and_undoes(app, win, tmp_path):
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    from vantage.core import critical as CR
    from vantage.core import critical_carla as CC
    from vantage.ui.workspaces.critical_carla_ws import CriticalCarlaPage
    p = win.store.project
    v = next(v for v in p.variants if v.sensors)

    def setup(pj):
        pj.active_uid = v.uid
        v.cameras[:] = [c for c in v.cameras if c.tag == 'F']
        v.sensors[:] = [s for s in v.sensors if s.kind == 'lidar']
        v.sensors[0].z = 2.0
        pj.analysis.critical = CR.CriticalSettings.from_dict({'targets': [
            dict(CR.preset(pj, 'Child 2 m ahead'), uid='kid'),
            dict(CR.preset(pj, 'Kerb by the right front wheel'), uid='kerb')]}).to_dict()
        pj.analysis.sim = {'image_scale': 0.08, 'warmup_s': 0.25}
    win.store.edit(['variants', 'analysis'], setup, 'test setup')
    pump(app, 0.3)
    pg = _open(win, 'critical_carla', CriticalCarlaPage)
    pump(app, 0.3)
    assert pg.tg_table.rowCount() == 2 and pg.tg_table.item(1, 3).text() == 'not in CARLA'
    assert '1 of 2 targets' in pg.plan_lbl.text()
    # spawn points typed as a user does; undo, redo
    pg.spawns.selectAll()
    QTest.keyClicks(pg.spawns, '3')
    QTest.keyClick(pg.spawns, Qt.Key.Key_Return)
    pump(app, 0.2)
    assert CC.settings_of(win.store.project).spawn_points == [3]
    win.store.undo.undo()
    pump(app, 0.2)
    assert CC.settings_of(win.store.project).spawn_points == [-1] and pg.spawns.text() == '-1'
    win.store.undo.redo()
    pump(app, 0.2)
    assert pg.spawns.text() == '3'
    # the image scale field, typed
    f = pg.fields['image_scale']
    f.lineEdit().selectAll()
    QTest.keyClicks(f.lineEdit(), '0.08')
    QTest.keyClick(f.lineEdit(), Qt.Key.Key_Return)
    pump(app, 0.2)
    assert CC.settings_of(win.store.project).image_scale == pytest.approx(0.1)          # clamped to its limit
    # the child's actor: a taller child walker from the combo; undo puts Automatic back
    combo = pg.tg_table.cellWidget(0, 2)
    k = combo.findData('blueprint:walker.pedestrian.0012')
    assert k > 0
    combo.setCurrentIndex(k)
    pump(app, 0.2)
    assert CC.settings_of(win.store.project).actors == {'kid': 'blueprint:walker.pedestrian.0012'}
    win.store.undo.undo()
    pump(app, 0.2)
    assert CC.settings_of(win.store.project).actors == {}
    assert pg.tg_table.cellWidget(0, 2).currentData() == 'auto'
    # run against the fake carla, as a user presses Run
    calls = []
    pg.runner = _runner(calls)
    pg.out_dir = str(tmp_path / 'drives')
    QTest.mouseClick(pg.b_run, Qt.MouseButton.LeftButton)
    pump(app, 0.2)
    assert pg.busy
    _wait(app, lambda: not pg.busy, 600)
    res = pg.result
    assert res is not None, pg.status.text()
    assert len(calls) == 1 and pg.rows.rowCount() == len(res.rows_of(counted=True)) >= 1
    assert pg.tiles['agree'].value != '—' and pg.tiles['drives'].value == '1'
    assert pg.res_table.rowCount() == 2
    pg.tabs.setCurrentIndex(2)
    pump(app, 0.2)
    assert pg.crop_target.currentData() == 'kid' and len(pg.crops.items) >= 1
    assert pg.chart.pts and 'Pixel error' in pg.notes.toPlainText()
    pg.only_dis.setChecked(True)
    pump(app, 0.1)
    assert pg.rows.rowCount() == len([r for r in res.rows_of(counted=True) if not r['agree'] or r['reason']])
    paths = pg.export(str(tmp_path / 'export'))
    assert any(x.endswith('critical_carla.json') for x in paths) and any('crops' in x for x in paths)
    assert pg.summary()['camera_views_compared'] == res.metrics['pairs']
    # a setting changed after the run: the result is stale; Run again reuses the drive
    pg._set('weather', weather='CloudyNoon')
    pump(app, 0.2)
    assert pg.stale and 'changed since the run' in pg.status.text()
    win.store.undo.undo()
    pump(app, 0.2)
    assert pg.run()
    _wait(app, lambda: not pg.busy, 600)
    assert len(calls) == 1 and pg.result.drives[0]['reused']
    # open the drives again without CARLA
    assert pg.open_drives(str(tmp_path / 'drives'))
    _wait(app, lambda: not pg.busy, 300)
    assert pg.result is not None and pg.result.metrics['pairs'] == res.metrics['pairs']
    # per-seed subfolders whose drives share their names: found at any depth, kept apart, each in the crop picker
    import shutil
    from vantage.core import critical_carla as CC
    for seed in ('seed1', 'seed2'):
        for d in CC.drives_in(str(tmp_path / 'drives')):
            shutil.copytree(d, tmp_path / 'nested' / seed / os.path.basename(d))
    assert pg.open_drives(str(tmp_path / 'nested'))
    _wait(app, lambda: not pg.busy, 300)
    keys = [d['key'] for d in pg.result.drives]
    assert len(keys) == len(set(keys)) == 2 * len(res.drives) and all(k.startswith('seed') for k in keys)
    assert pg.result.metrics['pairs'] == 2 * res.metrics['pairs'] and pg.crop_drive.count() == 1 + len(keys)
    # a new project: the result goes
    from vantage.core import samples
    win.store.set_project(samples.tigor_sample_project())
    pump(app, 0.3)
    assert pg.result is None
