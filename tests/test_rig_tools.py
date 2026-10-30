"""Whole-rig design for the assistant (assistant/rig_tools.py: optimise_rig, add_camera_options) and the optimiser
options it relies on (core.optimiser: targets as constraints, units, the slide quantum).

Every search runs with a fixed seed and a fixed number of generations inside a generous budget, so the results do not
depend on the machine's speed. The independent checks: an option's changes applied to a copy with apply_changes and
analysed reproduce the figures the tool reported; no option dominates another; a camera tilted away from its optimum
is brought back; a redundant mirrored pair is dropped as a pair; a gap left by removed cameras is closed by a camera
the add search places in an allowed zone."""
import copy
import math
import threading
import time

import numpy as np
import pytest

from vantage.assistant import tools as T
from vantage.assistant import rig_tools as R
from vantage.core import rules, samples
from vantage.core.analysis import Engine
from vantage.core.optimiser import FreeVar, OptSettings, Optimiser, default_vars
from vantage.core.project import MAX_CAMERAS, new_uid

FAST = dict(generations=6, population=12, budget_s=90)


@pytest.fixture(scope='module')
def eng():
    return Engine()


def _data_rig():
    p = samples.tigor_sample_project()
    p.active_uid = next(v for v in p.variants if v.sensors).uid
    return p


def _v3():
    return samples.tigor_sample_project()            # active: '6-cam · v3', no lidar or radars


def _tc(p, eng, **kw):
    return T.ToolContext(p.snapshot(), None, eng.share(), **kw)


def _cams(p):
    return [c.to_dict() for c in p.variant.cameras]


def _apply_and_measure(tc, changes):
    """The project-resolution metrics of the snapshot with the changes applied by apply_changes (independently of the
    tool's own re-check)."""
    trial = tc.project.snapshot()
    _, targets, _, errors = T.apply_changes(trial, copy.deepcopy(changes))
    assert not errors and targets
    return tc.engine.analyse(trial).metrics, trial


@pytest.fixture(scope='module')
def data_run(eng):
    """The Data rig (6 cameras, a roof lidar, five radars) with its quarter cameras tilted 12° up: blind ground a
    better layout closes. (As shipped, the Data rig is at a local optimum within the default ranges: see
    test_rig_search_leaves_the_project_alone_and_keeps_the_budget.)"""
    p = _data_rig()
    for tag in ('QL', 'QR'):
        p.variant.by_tag(tag).pitch -= 12
    before = _cams(p)
    tc = _tc(p, eng)
    t0 = time.perf_counter()
    out = R.optimise_rig(tc, objective='blind', seed=3, generations=10, population=14, budget_s=90)
    return p, tc, out, time.perf_counter() - t0, before


# =========================================================================== optimise_rig
def test_rig_options_are_valid_and_reproduce_their_figures(data_run):
    p, tc, out, secs, _ = data_run
    assert 'error' not in out, out
    assert out['options'], out
    assert out['search']['stopped'] == 'finished' and out['search']['generations'] == 10
    tags = {c.tag for c in p.variant.cameras}
    for o in out['options']:
        assert o['changes'] and all(ch['camera'] in tags for ch in o['changes'])
        m, trial = _apply_and_measure(tc, o['changes'])
        for k in ('blind_m2', 'min_density', 'covered_pct', 'multi_pct'):
            assert m[k] == pytest.approx(o['metrics'][k][1], abs=2e-3), (o['option'], k)
            assert o['metrics'][k][0] == pytest.approx(out['current'].get(
                {'blind_m2': 'blind_m2', 'min_density': 'min_density_px_cm'}.get(k, k), o['metrics'][k][0]), abs=6e-3)
        # mirrored pairs stay mirror images
        for c in trial.variant.cameras:
            q = trial.variant.cam(c.link_uid) if c.link_uid else None
            if q is not None and c.visible and q.visible:
                assert c.y == pytest.approx(-q.y, abs=4e-3) and c.yaw == pytest.approx(-q.yaw, abs=0.06)
                assert c.pitch == pytest.approx(q.pitch, abs=0.06) and c.x == pytest.approx(q.x, abs=4e-3)
        assert o['meets_all_targets'] and o['trade_off'].startswith(f"{o['cameras']['count']} camera")
        assert o['cameras']['count'] == 6 and not o['cameras']['removed']
    # the figures propose_changes shows are the same
    seen = []
    tc2 = T.ToolContext(tc.project, None, tc.engine, on_proposal=seen.append)
    best = out['options'][0]
    prop = T.propose_changes(tc2, changes=copy.deepcopy(best['changes']), title='Rig option A')
    assert prop['proposal'] == 1 and seen
    eff = {e['key']: e for e in seen[0].effects}
    assert eff['blind_m2']['after'] == pytest.approx(best['metrics']['blind_m2'][1], abs=2e-3)


def test_rig_options_are_a_pareto_set_better_than_now(data_run):
    _, _, out, _, _ = data_run
    rows = [(o['metrics']['blind_m2'][1], -o['metrics']['min_density'][1], o['cameras']['count'],
             o['worst_blind_under_tolerance_m2'][1]) for o in out['options']]
    for i, a in enumerate(rows):
        for j, b in enumerate(rows):
            if i != j:
                assert not (all(x <= y for x, y in zip(b, a)) and any(x < y for x, y in zip(b, a))), (i, j, rows)
    cur = out['current']
    for o in out['options']:                         # each improves on the current rig on the objective
        gain = cur['blind_m2'] - o['metrics']['blind_m2'][1]
        tol = cur['worst_blind_under_tolerance_m2'] - o['worst_blind_under_tolerance_m2'][1]
        assert gain >= 0.002 or tol >= 0.005, o


def test_rig_search_leaves_the_project_alone_and_keeps_the_budget(data_run, eng):
    p, tc, out, secs, before = data_run
    assert _cams(p) == before and _cams(tc.project) == before
    assert secs < 90 and out['search']['seconds'] <= 90
    quick = R.optimise_rig(_tc(_data_rig(), eng), vary=['pitch'], budget_s=8, generations=2000)
    assert 'error' not in quick and quick['search']['stopped'] == 'budget'
    assert quick['search']['seconds'] <= 8.0 + 1.5, quick['search']
    # the shipped Data rig: either a real gain (out of sample) or an honest "nothing better"
    assert quick['options'] or 'no rig found' in quick['result']
    for o in quick['options']:
        assert (o['metrics']['blind_m2'][1] <= o['metrics']['blind_m2'][0] - 0.002
                or o['worst_blind_under_tolerance_m2'][1] <= o['worst_blind_under_tolerance_m2'][0] - 0.005)


def test_locked_cameras_and_their_partners_stay(eng):
    p = _v3()
    p.variant.by_tag('F').locked = True
    p.variant.by_tag('ML').locked = True             # MR is its mirrored partner: it stays too
    out = R.optimise_rig(_tc(p, eng), vary=['pitch', 'yaw'], seed=5, **FAST)
    assert 'error' not in out
    assert set(out['fixed_cameras']) == {'F', 'ML', 'MR'}
    for o in out['options']:
        assert not {ch['camera'] for ch in o['changes']} & {'F', 'ML', 'MR'}, o['changes']
    only = R.optimise_rig(_tc(p, eng), cameras=['QL'], vary=['pitch'], seed=5, **FAST)
    assert set(only['fixed_cameras']) == {'F', 'R', 'ML', 'MR'}        # QR comes along with QL
    for o in only['options']:
        assert {ch['camera'] for ch in o['changes']} <= {'QL', 'QR'}
    p.variant.by_tag('QL').locked = p.variant.by_tag('QR').locked = p.variant.by_tag('R').locked = True
    assert 'locked' in R.optimise_rig(_tc(p, eng), **FAST)['error']


def test_targets_are_constraints_and_misses_are_flagged(eng):
    p = _v3()
    for tag in ('QL', 'QR'):
        p.variant.by_tag(tag).pitch -= 12            # blind patches the search can close
    tc = _tc(p, eng)
    easy = R.optimise_rig(tc, vary=['pitch'], min_density=1.0, max_blind_m2=5.0, seed=2, **FAST)
    assert easy['targets'] == {'min_density_px_cm': 1.0, 'max_blind_m2': 5.0} and easy['options']
    assert all(o['meets_all_targets'] and o['meets_targets']['min_density'] and o['meets_targets']['max_blind_m2']
               for o in easy['options'])
    hard = R.optimise_rig(tc, vary=['pitch'], min_density=40.0, seed=2, **FAST)
    assert all(not o['meets_all_targets'] and not o['meets_targets']['min_density'] for o in hard['options'])
    assert 'meets' in hard['result'] and ('misses' in hard['options'][0]['trade_off'] if hard['options'] else True)


def test_the_cameras_objective_drops_a_redundant_pair_as_a_pair(eng):
    p = _v3()
    v = p.variant
    ml, mr = v.by_tag('ML'), v.by_tag('MR')
    a, b = copy.deepcopy(ml), copy.deepcopy(mr)
    a.uid, b.uid = new_uid(), new_uid()
    a.tag, b.tag = 'NL', 'NR'
    a.link_uid, b.link_uid = b.uid, a.uid
    a.x = b.x = ml.x - 0.05
    a.slot, b.slot = 6, 7
    v.cameras += [a, b]
    tc = _tc(p, eng)
    out = R.optimise_rig(tc, objective='cameras', count=[6, 8], vary=['pitch'], seed=4, generations=10, population=14,
                         budget_s=90)
    assert out['options'], out
    best = out['options'][0]
    assert best['cameras']['count'] == 6 and len(best['cameras']['removed']) == 2
    gone = {t for t in best['cameras']['removed']}
    assert gone in ({'NL', 'NR'}, {'ML', 'MR'}, {'QL', 'QR'})                  # a mirrored pair, never one of two
    m, trial = _apply_and_measure(tc, best['changes'])
    assert sum(c.visible for c in trial.variant.cameras) == 6
    assert m['blind_m2'] <= out['current']['blind_m2'] + 0.01                  # dropping it costs no blind ground
    assert best['meets_targets']['count']


def test_no_fewer_cameras_is_explained(eng):
    p = _v3()
    out = R.optimise_rig(_tc(p, eng), objective='cameras', count=[4, 6], min_density=3.0, vary=['pitch'], seed=1,
                         **FAST)
    assert 'error' not in out
    if not any(o['cameras']['count'] < 6 for o in out['options']):
        assert out['fewer_cameras'] and all(f['cameras'] < 6 and f['misses'] for f in out['fewer_cameras'])


def test_same_seed_same_options(eng):
    p = _v3()
    p.variant.by_tag('QL').pitch -= 12
    p.variant.by_tag('QR').pitch -= 12
    a = R.optimise_rig(_tc(p, eng), vary=['pitch'], seed=9, **FAST)
    b = R.optimise_rig(_tc(p, eng), vary=['pitch'], seed=9, **FAST)
    assert [o['changes'] for o in a['options']] == [o['changes'] for o in b['options']] and a['options']


def test_a_camera_tilted_away_from_its_optimum_is_brought_back(eng):
    p = _v3()
    good = eng.analyse(p).metrics['blind_m2']
    for tag in ('QL', 'QR'):
        p.variant.by_tag(tag).pitch -= 12
    bad = eng.analyse(p).metrics['blind_m2']
    assert bad > good + 0.03
    out = R.optimise_rig(_tc(p, eng), cameras=['QL'], vary=['pitch'], under_tolerance=False, seed=6, generations=15,
                         population=12, budget_s=90)
    best = out['options'][0]
    after = best['metrics']['blind_m2'][1]
    assert after <= good + 0.3 * (bad - good), (good, bad, after)            # most of the loss recovered
    ch = next(c for c in best['changes'] if c['camera'] in ('QL', 'QR'))
    assert ch['pitch'] > p.variant.by_tag(ch['camera']).pitch                # tilted back down
    assert ch['mirror_partner'] is True and len(best['changes']) == 1          # the pair moves as one change


def test_cancel_stops_the_search_early(eng):
    p = _data_rig()
    ev = threading.Event()
    calls = []

    def status(text):
        calls.append(text)
        if 'generation' in text:
            ev.set()
    tc = _tc(p, eng, on_status=status)
    tc.cancel = ev
    t0 = time.perf_counter()
    out = R.optimise_rig(tc, generations=500, budget_s=60)
    assert out['error'] == 'cancelled' and out['search']['generations'] <= 2
    assert time.perf_counter() - t0 < 30
    ev2 = threading.Event()
    ev2.set()
    tc2 = _tc(_v3(), eng)
    tc2.cancel = ev2
    assert R.add_camera_options(tc2)['error'] == 'cancelled'


@pytest.mark.parametrize('kw', [
    {'objective': 'speed'}, {'vary': ['spin']}, {'vary': 'pitch'}, {'vary': []}, {'ranges': 'wide'},
    {'ranges': {'roll_deg': 5}}, {'ranges': {'yaw_deg': 'lots'}}, {'count': [0, 0]}, {'count': '4-6'},
    {'count': [4]}, {'count': [True, 6]}, {'count': [9, 12]}, {'min_density': 'high'}, {'max_blind_m2': float('nan')},
    {'budget_s': 'soon'}, {'cameras': ['ZZ']}, {'cameras': 'F'}, {'cameras': []}, {'critical': 'maybe'},
    {'options': 'all'}, {'tolerance_deg': [1]}, {'count': [float('inf'), 6]}, {'under_tolerance': 'yes'}])
def test_junk_arguments_give_errors(eng, kw):
    out = R.optimise_rig(_tc(_v3(), eng), **kw)
    assert isinstance(out, dict) and 'error' in out and isinstance(out['error'], str), (kw, out)


def test_hidden_cameras_stay_off_and_a_project_without_cameras_errors(eng):
    p = _v3()
    p.variant.by_tag('R').visible = False
    out = R.optimise_rig(_tc(p, eng), vary=['pitch'], seed=2, **FAST)
    assert 'error' not in out and out['current']['cameras'] == 5
    for o in out['options']:
        assert 'R' not in {ch['camera'] for ch in o['changes']} and 'R' not in o['cameras']['used']
    assert 'error' in R.optimise_rig(_tc(p, eng), cameras=['R'])
    for c in p.variant.cameras:
        c.visible = False
    assert 'error' in R.optimise_rig(_tc(p, eng))


# =========================================================================== add_camera_options
@pytest.fixture(scope='module')
def gap(eng):
    """The 6-cam sample without its quarter cameras: some 6 m² of blind ground along the sides."""
    p = _v3()
    for tag in ('QL', 'QR'):
        p.variant.cameras.remove(p.variant.by_tag(tag))
    before = _cams(p)
    tc = _tc(p, eng)
    t0 = time.perf_counter()
    out = R.add_camera_options(tc, budget_s=60)
    return p, tc, out, time.perf_counter() - t0, before


def test_add_closes_the_gap_from_an_allowed_zone(gap, eng):
    p, tc, out, secs, before = gap
    assert 'error' not in out and out['objective'] == 'blind' and out['gap']['m2'] > 3.0
    assert out['options'] and secs < 60
    best = out['options'][0]
    b0, b1 = best['metrics']['blind_m2']
    assert b1 < 0.2 * b0                                                       # most of the gap closed
    assert any(not o['change']['mirror_partner'] for o in out['options']) or len(out['options']) < 3
    existing = p.variant.tags()
    zones = {z.name: z.kind for z in p.vehicle.mesh.zones}
    for o in out['options']:
        ch = o['change']
        assert ch['add'] is True and ch['camera'] not in existing
        trial = tc.project.snapshot()
        rows, made, errors = R.apply_add(trial, ch)
        assert not errors and len(made) == (2 if ch['mirror_partner'] else 1)
        assert [r['camera'] for r in rows] == o['adds'] and not set(o['adds']) & existing
        m = eng.analyse(trial).metrics                                        # reproduces the reported figures
        assert m['blind_m2'] == pytest.approx(o['metrics']['blind_m2'][1], abs=2e-3)
        assert m['min_density'] == pytest.approx(o['metrics']['min_density'][1], abs=2e-3)
        vg = eng.vehicle_geom(trial)
        for c in made:                                                         # on the body, in an allowed zone
            mi = rules.mount_info(trial, c, vg)
            assert mi is not None and mi.zone_kind == 'allowed' and mi.gap_mm < 30, (c.tag, mi)
            assert zones.get(mi.surface) == 'allowed'
        if ch['mirror_partner']:
            a, b = made
            assert a.link_uid == b.uid and b.link_uid == a.uid                 # added as a mirrored pair
            assert b.y == pytest.approx(-a.y) and b.yaw == pytest.approx(-a.yaw)
    assert _cams(p) == before and _cams(tc.project) == before


def test_an_add_change_never_moves_an_existing_camera(gap):
    """Until apply_changes understands an add, the change it is given must fail rather than move a camera."""
    p, tc, out, _, _ = gap
    snap = tc.project.snapshot()
    rows, targets, _, errors = T.apply_changes(snap, [copy.deepcopy(out['options'][0]['change'])])
    if errors:
        assert not targets and _cams(snap) == _cams(tc.project)
    else:
        assert len(snap.variant.cameras) > len(tc.project.variant.cameras)    # wired: it adds


def test_add_where_unit_and_limits(gap, eng):
    p, tc, _, _, _ = gap
    meas = p.vehicle.measured()
    rear = R.add_camera_options(tc, where='rear', pair=False, options=2, budget_s=40)
    assert 'error' not in rear, rear
    for o in rear['options']:
        assert o['change']['x'] < meas['rear_x'] + 0.6 and not o['change']['mirror_partner']
    lens = next(l for l in p.lenses.values() if l.module_id and l.id != p.variant.cameras[0].lens_id)
    other = R.add_camera_options(tc, lens=lens.id, pair=False, options=1, budget_s=40)
    assert 'error' not in other
    if other['options']:
        assert other['options'][0]['change']['lens'] == lens.id
        assert other['options'][0]['change']['module'] == lens.module_id
    full = _v3()
    while len(full.variant.cameras) < MAX_CAMERAS:
        c = copy.deepcopy(full.variant.cameras[0])
        c.uid, c.tag, c.link_uid = new_uid(), f'X{len(full.variant.cameras)}', ''
        full.variant.cameras.append(c)
    assert 'GMSL' in R.add_camera_options(_tc(full, eng))['error']


@pytest.mark.parametrize('kw', [
    {'where': 'top'}, {'objective': 'redundancy'}, {'module': 'no such module'}, {'lens': 'zzz'}, {'pair': 'yes'},
    {'zones': 'some'}, {'budget_s': 'later'}, {'min_density': 'x'}, {'options': float('inf')}])
def test_add_junk_arguments_give_errors(gap, kw):
    out = R.add_camera_options(gap[1], **kw)
    assert isinstance(out, dict) and isinstance(out.get('error'), str), (kw, out)


def test_apply_add_validates(gap):
    snap = gap[1].project.snapshot()
    n = len(snap.variant.cameras)
    for bad in ({'camera': 'Z', 'add': True}, {'camera': 'Z', 'add': True, 'x': 1, 'y': 0, 'z': 'high', 'yaw': 0},
                {'camera': 'Z', 'add': True, 'x': 1, 'y': 0, 'z': 9.0, 'yaw': 0, 'pitch': 40},
                {'camera': 'Z', 'add': True, 'x': math.nan, 'y': 0, 'z': 1, 'yaw': 0, 'pitch': 40},
                {'camera': 'Z', 'add': True, 'x': 1, 'y': 0, 'z': 1, 'yaw': 0, 'pitch': 40, 'lens': 'zzz'}):
        rows, made, errors = R.apply_add(snap, bad)
        assert errors and not made and len(snap.variant.cameras) == n
    rows, made, errors = R.apply_add(snap, {'camera': 'F', 'add': True, 'x': 1.0, 'y': 0.9, 'z': 0.9, 'yaw': 90,
                                            'pitch': 45})
    assert not errors and made[0].tag != 'F' and made[0].tag.startswith('F')   # a free tag, never an existing one
    assert made[0].port not in {c.port for c in snap.variant.cameras if c is not made[0] and c.port}


# =========================================================================== registry and schemas
def test_tools_are_registered_once_with_strict_schemas():
    from vantage.assistant import claude as CL
    names = [t.name for t in R.TOOLS_RIG]
    assert names == ['optimise_rig', 'add_camera_options']
    R.register()
    R.register()
    assert [t.name for t in T.TOOLS].count('optimise_rig') == 1 and T.BY_NAME['optimise_rig'].fn is R.optimise_rig
    for t in R.TOOLS_RIG:
        schema = T.input_schema(t)
        assert CL._closed(schema), t.name                        # every object lists its properties: strict
        strict = CL._strict_schema(schema)

        def closed(o):
            if isinstance(o, dict):
                if o.get('type') == 'object':
                    assert o.get('additionalProperties') is False and isinstance(o.get('properties'), dict)
                for x in o.values():
                    closed(x)
        closed(strict)
        assert t.kind == 'compute' and len(t.description) < 1200
    defs = {d['name']: d for d in CL.tool_defs()}
    assert defs['optimise_rig'].get('strict') and defs['add_camera_options'].get('strict')
    ok = {'objective': 'cameras', 'count': [4, 6], 'min_density': 3, 'ranges': {'yaw_deg': 10}, 'budget_s': 30,
          'cameras': ['QL'], 'vary': ['pitch'], 'critical': 'require', 'options': 3}
    assert T.validate_args('optimise_rig', ok) == []
    assert T.validate_args('optimise_rig', {'count': [4.5, 6]}) and T.validate_args('optimise_rig', {'nope': 1})
    assert T.validate_args('optimise_rig', {'ranges': {'roll_deg': 3}})
    assert T.validate_args('add_camera_options', {'where': 'rear', 'pair': True, 'zones': 'any'}) == []
    assert 'optimise_camera' in T.BY_NAME['optimise_rig'].description           # says when to use which
    assert 'Nothing changes until the user accepts' in T.BY_NAME['optimise_rig'].description


def test_tools_run_turns_junk_into_an_error_message(eng):
    tc = _tc(_v3(), eng)
    text, ok = T.run(tc, 'optimise_rig', {'objective': 'speed'})
    assert not ok and 'objective' in text
    text, ok = T.run(tc, 'add_camera_options', {'where': 'moon'})
    assert not ok and 'where' in text


# =========================================================================== core.optimiser additions
def _pairs_variant():
    p = _v3()
    return p, p.variant


def test_units_switch_pairs_together_and_never_a_kept_or_hidden_camera(eng):
    p, v = _pairs_variant()
    v.by_tag('R').visible = False
    fvars = [fv for fv in default_vars(p, v) if fv.tag != 'R']
    for fv in fvars:
        if fv.tag == 'F':
            fv.keep = True
    st = OptSettings(objective='cameras', count=(2, 5), units=True, vars=fvars, generations=1, population=6,
                     tol_samples=0, under_tolerance=False, seed=11)
    o = Optimiser(eng, p, st)
    uid = {c.tag: c.uid for c in v.cameras}
    assert sorted(len(u) for u in o._units) == [2, 2]           # ML+MR and QL+QR; F is kept, R hidden
    nom = o.nominal()
    assert nom.genes[uid['R']][3] == 0.0
    seen = set()
    for _ in range(300):
        g = o.mutate(nom, 1.5).genes
        on = {t for t, u in uid.items() if g[u][3] > 0.5}
        assert 'F' in on and 'R' not in on and 2 <= len(on) <= 5
        assert ('ML' in on) == ('MR' in on) and ('QL' in on) == ('QR' in on)
        seen.add(len(on))
    assert seen == {3, 5}                                       # both pairs get dropped at some point (never 1 of 2)


def test_targets_rank_and_filter_candidates(eng):
    p = _v3()
    o = Optimiser(eng, p, OptSettings(need_density=3.0, need_blind=0.05, tol_samples=0, under_tolerance=False))
    from vantage.core.optimiser import Candidate
    good = Candidate({}, worst=0.02, min_density=3.2, cams=6)
    thin = Candidate({}, worst=0.0, min_density=2.4, cams=6)
    blind = Candidate({}, worst=0.30, min_density=3.5, cams=5)
    assert o.shortfall(good) == 0 and o.shortfall(thin) == pytest.approx(0.2) and o.shortfall(blind) == pytest.approx(0.25)
    assert o.feasible(good) and not o.feasible(thin) and not o.feasible(blind)
    assert sorted([blind, thin, good], key=o.objectives)[0] is good
    assert o._front([good, thin, blind]) == [good]
    assert o._front([thin, blind]) == [thin]                    # none meets: the closest
    plain = Optimiser(eng, p, OptSettings(tol_samples=0))
    assert len(plain.objectives(good)) == 3 and plain.feasible(thin) and plain.shortfall(thin) == 0


def test_slide_quantum(eng):
    p = _v3()
    fv = [FreeVar(c.uid, c.tag, axis='x' if c.tag == 'QL' else '', move=0.3 if c.tag == 'QL' else 0.0,
                  yaw=(c.yaw, c.yaw), pitch=(c.pitch, c.pitch), step=0.05) for c in p.variant.cameras]
    o = Optimiser(eng, p, OptSettings(vars=fv, count=(6, 6), seed=1, tol_samples=0))
    uid = p.variant.by_tag('QL').uid
    vals = {o.mutate(o.nominal(), 1.5).genes[uid][0] for _ in range(200)}
    assert len(vals) > 3 and all(abs(s) <= 0.3 + 1e-9 and abs(s / 0.05 - round(s / 0.05)) < 1e-6 for s in vals)
