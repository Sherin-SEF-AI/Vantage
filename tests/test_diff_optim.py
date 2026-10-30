"""Gradient refinement of a rig (core.diff_optim) and its section of the Optimise page.

References: the torch lens and rotation against core.lens / core.mathutil (numpy, independent code paths); the
gradient against central finite differences of the same surrogate; the surrogate at a sharp τ against the exact
engine (Engine.analyse) on the sample."""
import copy
import json
import math
import os
import time

import numpy as np
import pytest

from vantage.core import diff_optim as DO
from vantage.core import critical as CR
from vantage.core import rules, samples
from vantage.core.analysis import Engine
from vantage.core.lens import r_of_theta, dr_dtheta
from vantage.core.mathutil import rot_zyx

FAST = dict(stages=3, iterations=25)


@pytest.fixture(scope='module', autouse=True)
def _torch_one_thread():
    """The gradient checks call the surrogate directly: one intra-op thread, as DO.run uses."""
    import torch
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


@pytest.fixture(scope='module')
def eng():
    return Engine()


def _sample(name='4-cam'):
    p = samples.tigor_sample_project()
    v = next(v for v in p.variants if v.name == name)
    p.active_uid = v.uid
    return p, v


@pytest.fixture(scope='module')
def refined(eng):
    """The sample's 4-cam variant refined with the default weights (three stages)."""
    p, v = _sample()
    st = DO.DiffSettings(**FAST)
    return p, v, st, DO.run(p, eng, st, v)


# ------------------------------------------------------------------ settings
def test_settings_are_tolerant_and_clamped():
    s = DO.DiffSettings.from_dict({'start': 'nonsense', 'w_blind': 'x', 'w_density': -3, 'max_move_mm': 1e9,
                                   'stages': 2.6, 'iterations': float('nan'), 'move': 'off', 'seams': 1,
                                   'unknown': 5, 'max_turn_deg': True})
    d = DO.DiffSettings()
    assert s.start == 'variant' and s.w_blind == d.w_blind and s.w_density == 0.0
    assert s.max_move_mm == DO.LIMITS['max_move_mm'][1] and s.stages == 3 and isinstance(s.stages, int)
    assert s.iterations == d.iterations and s.move is False and s.seams is True and s.max_turn_deg == d.max_turn_deg
    assert DO.DiffSettings.from_dict(None) == d and DO.DiffSettings.from_dict(d.to_dict()) == d
    assert set(d.to_dict()) == set(DO.DiffSettings.__dataclass_fields__)
    p, _ = _sample()
    p.analysis.diff_optim = {'start': 'pareto', 'candidates': 9}
    assert DO.settings_of(p).start == 'pareto' and DO.settings_of(p).candidates == DO.MAX_STARTS
    sh = DO.DiffSettings(stages=4, sharp_end=0.1).sharpness()
    assert sh[0] == 1.0 and abs(sh[-1] - 0.1) < 1e-12 and all(abs(sh[i + 1] / sh[i] - sh[1]) < 1e-12
                                                              for i in range(3))


# ------------------------------------------------------------------ the maths against numpy references
@pytest.mark.parametrize('proj', ['rectilinear', 'equidistant', 'equisolid', 'stereographic', 'orthographic',
                                  'kannala_brandt'])
def test_torch_lens_matches_the_lens_model(proj):
    import torch
    kb = (0.02, -0.01, 0.003, -0.0004)
    th = np.linspace(0.01, 1.4, 50)
    t = torch.tensor(th, dtype=torch.float64)
    assert np.allclose(DO._r_t(torch, t, 1.7, proj, kb).numpy(), r_of_theta(th, 1.7, proj, kb), rtol=1e-12)
    assert np.allclose(DO._dr_t(torch, t, 1.7, proj, kb).numpy(), dr_dtheta(th, 1.7, proj, kb), rtol=1e-12)


def test_torch_rotation_matches_rot_zyx():
    import torch
    for ypr in ((0, 0, 0), (65, 45, 0), (-105, 55, 3), (180, 40, -7.5)):
        R = DO._rot_t(torch, *(torch.tensor(float(a), dtype=torch.float64) for a in ypr)).numpy()
        assert np.allclose(R, rot_zyx(*ypr), atol=1e-14)
    assert DO.mirror((1, 0.9, 0.5, 65, 45, 2)) == (1, -0.9, 0.5, -65, 45, -2)


def test_the_surrogate_at_a_sharp_tau_agrees_with_the_exact_engine(eng):
    p, v = _sample()
    sur, x = DO.surrogate_at(p, eng, v, DO.DiffSettings(), sharp=0.05)
    got = sur.parts(x)
    ex = DO.exact_check(p, eng, v, v.cameras, DO.DiffSettings(), [], DO._objects(p, []))
    # the exact engine on 5 cm cells; the surrogate on 10 cm cells refined to 5 cm around the blind and thin ground
    assert abs(got['blind_m2'] - ex['blind_m2']) < 0.05
    assert abs(got['low_m2'] - ex['low_m2']) < 0.1 * ex['low_m2']
    assert abs(sur.Ga.sum() - ex['result'].metrics['roi_area']) < 1e-6        # the sample covers the ROI once


def _with_target_on_a_seam(p, v, eng):
    """A child (min 10 px) 3 m out on the 4-cam variant's front-right seam; the poles of the sample stay."""
    r0 = eng.analyse(p, variant=v)
    s = min(r0.seams, key=lambda s: abs(s.angle + 24))
    a = math.radians(s.angle)
    t = CR.preset(p, 'child')
    t.update(x=r0.centre[0] + 3.0 * math.cos(a), y=r0.centre[1] + 3.0 * math.sin(a), min_px=10.0)
    p.analysis.critical = {'targets': [t]}
    return t


@pytest.mark.parametrize('sharp,seams', [(1.0, False), (0.2, True)])
def test_the_gradient_matches_finite_differences(eng, sharp, seams):
    """Every term (ground, a critical target, the seam proximity, seam variables, the surface penalty) against
    central differences of the same surrogate."""
    p, v = _sample()
    _with_target_on_a_seam(p, v, eng)
    st = DO.DiffSettings(seams=seams, w_seam=0.5, w_critical=1.0)
    sur, x0 = DO.surrogate_at(p, eng, v, st, sharp=sharp)
    assert sur.targets and sur.objects and (not seams or any(s['mode'] == 'var' for s in sur.seams))
    rng = np.random.default_rng(3)
    scale = np.r_[np.tile([4.0, 4.0, 4.0, 1.0, 1.0, 0.5], len(sur.units)), np.full(len(x0) - sur.n_unit, 2.0)]
    x = rng.uniform(-1, 1, len(x0)) * scale
    f, g = sur.value_and_grad(x)
    assert math.isfinite(f) and np.isfinite(g).all() and np.abs(g).max() > 0
    h = 1e-5
    E = np.eye(len(x))
    fd = np.array([(sur.value(x + h * E[i]) - sur.value(x - h * E[i])) / (2 * h) for i in range(len(x))])
    assert np.linalg.norm(g - fd) <= 1e-5 * np.linalg.norm(fd)
    assert np.allclose(g, fd, rtol=1e-4, atol=1e-7)


# ------------------------------------------------------------------ refining the sample
def test_refining_the_4cam_variant_improves_the_exact_check(refined, eng):
    p, v, st, res = refined
    r = res.refinements[0]
    b, a = r.exact_before, r.exact_after
    assert r.improved and a['objective'] < b['objective']
    assert a['blind_m2'] < b['blind_m2'] - 1e-9 or a['low_m2'] < b['low_m2'] - 1e-9
    assert any(f['severity'] == 'good' and f['kind'] == 'improved' for f in res.findings)
    # the exact figures are the engine's for the refined layout
    nv = DO.to_variant(r, v, 'refined')
    again = eng.analyse(p, variant=nv)
    assert abs(again.metrics['blind_m2'] - a['blind_m2']) < 1e-9
    assert r.checks[0]['objective'] == b['objective'] and r.kept >= 1
    assert r.history and all(h['objective'] == h['objective'] for h in r.history)


def test_the_refined_layout_keeps_bounds_mount_zones_and_mirrored_pairs(refined, eng):
    p, v, st, res = refined
    r = res.refinements[0]
    lims = [st.max_move_mm] * 3 + [st.max_turn_deg] * 2 + [st.max_roll_deg]
    for d in r.deltas():
        for val, m in zip(d['delta'], lims):
            assert abs(val) <= m + 1e-6
    after = {c.uid: c for c in r.cams_after}
    ml, mr = v.by_tag('ML'), v.by_tag('MR')
    a, b = after[ml.uid], after[mr.uid]
    assert DO.mirror(DO.pose_of(a)) == pytest.approx(DO.pose_of(b), abs=1e-12)
    assert any(abs(x) > 0.05 for x in next(d for d in r.deltas() if d['tag'] == 'ML')['delta'])
    vg = eng.vehicle_geom(p, v)
    for c in v.cameras:                                  # on the same mount zone, at the same distance off it
        m0, m1 = rules.mount_info(p, c, vg), rules.mount_info(p, after[c.uid], vg)
        assert m1.surface == m0.surface and m1.zone_kind == 'allowed'
        assert abs(m1.gap_mm - m0.gap_mm) < 0.5
    nv = DO.to_variant(r, v, 'refined')
    tags = {c.uid: c.tag for c in nv.cameras}
    assert tags[nv.by_tag('ML').link_uid] == 'MR' and tags[nv.by_tag('MR').link_uid] == 'ML'
    assert {c.uid for c in nv.cameras}.isdisjoint({c.uid for c in v.cameras}) and nv.uid != v.uid


def test_refinement_is_deterministic(refined):
    p, v, st, res = refined
    again = DO.run(p, Engine(), st, v)
    r0, r1 = res.refinements[0], again.refinements[0]
    assert [DO.pose_of(c) for c in r0.cams_after] == [DO.pose_of(c) for c in r1.cams_after]
    assert r0.history == r1.history and r0.checks == r1.checks
    d0, d1 = DO.to_doc(res, p), DO.to_doc(again, p)
    for d in (d0, d1):
        d.pop('seconds')
        for x in d['refinements']:
            x.pop('seconds')
    assert json.dumps(d0, sort_keys=True) == json.dumps(d1, sort_keys=True)


def test_locked_and_hidden_cameras_stay_and_unpaired_links_move_apart(eng):
    p, v = _sample()
    v.by_tag('F').locked = True
    v.by_tag('R').visible = False
    units, notes = DO.units_of(v.cameras)
    assert [u.fixed for u in units] == [True, False] and units[1].follow == v.by_tag('MR').uid and not notes
    res = DO.run(p, eng, DO.DiffSettings(stages=1, iterations=10), v)
    r = res.refinements[0]
    roles = {d['tag']: d['role'] for d in r.deltas()}
    assert roles['F'] == 'locked' and roles['MR'] == 'mirrors'
    f_after = next(c for c in r.cams_after if c.tag == 'F')
    assert DO.pose_of(f_after) == DO.pose_of(v.by_tag('F'))
    r_after = next(c for c in r.cams_after if c.tag == 'R')
    assert DO.pose_of(r_after) == DO.pose_of(v.by_tag('R')) and not r_after.visible
    q = copy.deepcopy(v)
    q.by_tag('MR').yaw = -100.0                              # no longer a mirror image
    units, notes = DO.units_of(q.cameras)
    assert len(units) == 3 and notes and 'move apart' in notes[0]


def test_seam_angles_as_variables_move_a_seam_off_a_critical_target(eng):
    p, v = _sample()
    _with_target_on_a_seam(p, v, eng)
    st = DO.DiffSettings(seams=True, w_seam=0.5, **FAST)
    r = DO.run(p, eng, st, v).refinements[0]
    assert r.exact_before['seam_near'] >= 1 and r.exact_after['seam_near'] < r.exact_before['seam_near']
    assert r.seams_after and all(abs(DO._wrap(r.seams_after[k] - s['angle'])) <= st.seam_span_deg + 1e-6
                                 for s in r.exact_before['seams'] for k in r.seams_after if k == f'{s["a"]}|{s["b"]}')
    nv = DO.to_variant(r, v, 'seams')
    uids = {c.uid for c in nv.cameras}
    assert len(nv.seams) == len(r.seams_after) and all(set(k.split('|')) <= uids for k in nv.seams)
    assert all(not s.auto for s in eng.analyse(p, variant=nv).seams)
    # without objects the seams stay automatic (and say so)
    p2, v2 = _sample()
    p2.obstacles = []
    r2 = DO.run(p2, eng, DO.DiffSettings(seams=True, stages=1, iterations=3), v2).refinements[0]
    assert not r2.seams_after and any('seams stay automatic' in n for n in r2.notes)


def test_pareto_candidates_are_refined_and_never_made_worse(eng):
    from vantage.core.optimiser import Optimiser, OptSettings
    p, v = _sample()
    o = Optimiser(eng, p, OptSettings(generations=2, population=8, tol_samples=0, under_tolerance=False))
    o.run()
    cands = o.candidates(2)
    starts = DO.starts_from_optimiser(o, cands)
    assert [s[0] for s in starts] == [c.label for c in cands]
    res = DO.run(p, eng, DO.DiffSettings(stages=2, iterations=12), v, starts=starts)
    assert [r.label for r in res.refinements] == [c.label for c in cands]
    for r in res.refinements:
        assert r.exact_after['objective'] <= r.exact_before['objective'] + 1e-12


def test_cancel_stops_and_says_so(eng):
    import torch
    p, v = _sample()
    torch.set_num_threads(3)
    DO.run(p, eng, DO.DiffSettings(), v, cancel=lambda: True)
    assert torch.get_num_threads() == 3                       # run() puts torch's thread count back
    torch.set_num_threads(1)
    res = DO.run(p, eng, DO.DiffSettings(), v, cancel=lambda: True)
    assert res.cancelled and not res.refinements and res.findings[0]['kind'] == 'cancelled'
    n = [0]

    def later():
        n[0] += 1
        return n[0] > 4
    res = DO.run(p, eng, DO.DiffSettings(), v, cancel=later)
    assert res.cancelled and res.refinements[0].stages[-1]['stop'] == 'cancelled'
    assert res.refinements[0].exact_after['objective'] <= res.refinements[0].exact_before['objective']


def test_to_doc_files_and_batch(refined, eng):
    p, v, st, res = refined
    doc = DO.to_doc(res, p)
    assert doc['schema'] == DO.SCHEMA and doc['frame'].startswith('vehicle')
    assert doc['settings'] == st.to_dict() and set(doc['settings']) == set(DO.DiffSettings.__dataclass_fields__)
    assert doc['solver']['step_tolerance'] == {'mm': 0.1, 'deg': 0.01} and 'not the accuracy' in doc['solver']['note']
    assert doc['limits'] and 'units' in doc and doc['refinements'][0]['exact']['after']['blind_m2'] is not None
    json.dumps(doc)
    names = dict(DO.files(res, p))
    assert set(names) == {'diff_optim.json', 'diff_optim_cameras.csv', 'diff_optim_history.csv'}
    rows = names['diff_optim_cameras.csv'].decode().splitlines()
    assert rows[0].split(',') == list(DO.CAMERA_COLUMNS) and len(rows) == 1 + len(v.cameras)
    assert json.loads(names['diff_optim.json'])['schema'] == DO.SCHEMA
    doc, code = DO.batch(p, eng, 'no such variant')
    assert code == 4 and 'error' in doc
    p.analysis.diff_optim = {'stages': 1, 'iterations': 3}
    doc, code = DO.batch(p, eng, '4-cam')
    assert code == 0 and doc['variant'] == '4-cam' and doc['settings']['iterations'] == 3


def test_the_module_has_no_qt_and_no_unseeded_randomness():
    src = open(DO.__file__, encoding='utf-8').read()
    assert 'PyQt' not in src and 'np.random' not in src and 'import random' not in src


# ------------------------------------------------------------------ the page
from test_gui import app, win, pump, settle  # noqa: E402,F401  (fixtures)


def _wait(app, cond, timeout):
    t = time.time()
    while not cond() and time.time() - t < timeout:
        pump(app, 0.05)
    return cond()


def test_the_optimiser_page_refines_with_gradients(app, win, tmp_path):
    from vantage.ui.workspaces.optimiser_ws import OptimiserPage
    store = win.store
    v = next(v for v in store.project.variants if v.name == '4-cam')

    def fn(pj):
        pj.active_uid = v.uid
    store.edit(['variants'], fn, 'use the 4-cam variant')
    settle(app, win)
    pg = OptimiserPage(win.ctx)
    pg.activate()
    rp = pg.refiner
    assert pg.tabs.count() == 2 and pg.tabs.tabText(1) == 'Refine with gradients'
    # settings edits are one undo step each and come back with undo
    rp.lim['stages'].set_value(2)
    rp.lim['stages'].committed.emit(2.0)
    rp.lim['iterations'].set_value(12)
    rp.lim['iterations'].committed.emit(12.0)
    assert store.project.analysis.diff_optim['stages'] == 2 and store.project.analysis.diff_optim['iterations'] == 12
    store.undo.undo()
    pump(app, 0.05)
    assert DO.settings_of(store.project).iterations == DO.DiffSettings().iterations and rp.lim['iterations'].value() == 40
    store.undo.redo()
    pump(app, 0.05)
    assert rp.settings().iterations == 12
    # Pareto start without a search: refused with a message
    rp.start.set_value('pareto')
    rp._start_note()
    assert not rp.start_run() and 'Run the evolutionary search' in rp.start_note.text()
    rp.start.set_value('variant')
    assert rp.start_run()
    assert rp.b_run.text() == 'Cancel'
    assert _wait(app, lambda: 'diff_optim' not in win._jobs, 180)
    pump(app, 0.3)
    res = rp.result
    assert res is not None and res.refinements and rp.which.count() == 1
    assert rp.chart.hist and rp.chart.checks and rp.b_run.text() == 'Refine'
    assert rp.metrics.item(0, 1).text() == f'{res.refinements[0].exact_before["blind_m2"]:.3f}'
    assert rp.cams.rowCount() == len(v.cameras) and rp.cams.item(0, 0).text() == 'F'
    assert rp._find_rows and rp.b_export.isEnabled()
    rp.chart.resize(500, 200)
    rp.chart.grab()
    paths = rp.export(str(tmp_path))
    assert {os.path.basename(x) for x in paths} >= {'diff_optim.json', 'diff_optim_cameras.csv'}
    if res.refinements[0].improved:
        n = len(store.project.variants)
        nv = rp.add_variant()
        assert nv is not None and len(store.project.variants) == n + 1 and store.project.active_uid == v.uid
        assert nv.name == '4-cam · refined'
        store.undo.undo()
        pump(app, 0.05)
        assert len(store.project.variants) == n
    # a new project drops the result
    store.set_project(samples.tigor_sample_project())
    settle(app, win)
    assert rp.result is None and rp.which.count() == 0 and not rp.b_add.isEnabled()


def test_refining_selected_pareto_candidates_from_the_page(app, win):
    from vantage.ui.workspaces.optimiser_ws import OptimiserPage
    store = win.store
    v = next(v for v in store.project.variants if v.name == '4-cam')

    def fn(pj):
        pj.active_uid = v.uid
        pj.analysis.diff_optim = {'stages': 1, 'iterations': 6, 'start': 'pareto'}
    store.edit(['variants', 'analysis'], fn, 'small refinement')
    settle(app, win)
    win.set_page('optimiser')
    pg = win.page('optimiser')
    pg.activate()
    pg.under_tol.setChecked(False)
    pg.gens.set_value(2)
    pg.pop.set_value(8)
    pg.toggle()
    assert _wait(app, lambda: 'optimiser' not in win._jobs, 180)
    pump(app, 0.3)
    assert pg.cands
    rp = pg.refiner
    rp.load()
    assert rp.start.value() == 'pareto' and rp.chosen() == [pg.cands[0]]     # the search selects its first
    pg.table.clearSelection()
    assert rp.chosen() == pg.cands[:3]
    pg.table.selectRow(0)
    assert rp.chosen() == [pg.cands[0]]
    assert rp.start_run()
    assert _wait(app, lambda: 'diff_optim' not in win._jobs, 180)
    pump(app, 0.3)
    assert [r.label for r in rp.result.refinements] == [pg.cands[0].label]
    win.set_page('viewports')


def test_nothing_to_refine_is_a_clear_error():
    """A variant with no visible camera (or no body) is refused with a plain reason, not a crash deep in the solver."""
    from vantage.core import samples, diff_optim as DO
    p = samples.tigor_sample_project()
    for c in p.variant.cameras:
        c.visible = False
    with pytest.raises(ValueError, match='needs a vehicle body and a visible camera'):
        DO.run(p)
