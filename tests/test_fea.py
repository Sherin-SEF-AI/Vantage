"""core.fea: the T10 element, Euler–Bernoulli beams, Leissa's plate, mesh convergence, the equivalent tubes, tied
joints against a closed form, the frame of the sample with the road and the wind, the page."""
import copy
import json
import math

import numpy as np
import pytest

from vantage.core import fea as FE
from vantage.core import framing as FR
from test_framing import scene


# ------------------------------------------------------------------ the element
def test_t10_mass_reference():
    M = FE.M_REF
    assert M.sum() == pytest.approx(1.0, abs=1e-12)            # Σ N = 1
    assert np.allclose(M, M.T)
    assert np.linalg.eigvalsh(M).min() > 0
    # corner-corner 6/420, corner-edge(adjacent) −4/420 … the known T10 consistent mass (×V)
    assert M[0, 0] == pytest.approx(6 / 420) and M[4, 4] == pytest.approx(32 / 420)


def test_t10_stiffness_rigid_modes():
    X4 = np.array([[[0, 0, 0], [1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]]) * 0.01
    K, Me = FE.element_matrices(X4, 70e9, 0.3, 2700.0)
    ev = np.linalg.eigvalsh(K[0])
    assert np.sum(np.abs(ev) < 1e-6 * ev.max()) == 6
    assert Me[0].sum() / 3 == pytest.approx(2700 * 1e-6 / 6)  # mass = ρV (three directions)
    # a uniform strain field: the internal forces of the element sum to zero
    nodes = np.vstack([X4[0], [(X4[0][i] + X4[0][j]) / 2 for i, j in FE._EDGES]])
    u = (nodes @ np.array([[1e-3, 0, 0], [0, -3e-4, 0], [0, 0, 2e-4]])).ravel()
    f = K[0] @ u
    assert np.allclose(f.reshape(-1, 3).sum(axis=0), 0, atol=1e-6 * np.abs(f).max())


def test_shape_functions_partition_and_nodes():
    L = np.random.default_rng(1).dirichlet(np.ones(4), size=50)
    assert np.allclose(FE.shape10(L).sum(axis=1), 1)
    corners = np.eye(4)
    assert np.allclose(FE.shape10(corners)[:, :4], np.eye(4))


# ------------------------------------------------------------------ closed forms
def test_cantilever_and_free_beam_against_euler_bernoulli():
    r = FE.validate_beams()
    for fea_f, ref, err in r['cantilever']:
        assert abs(err) < 2.0
    assert abs(r['free'][0][2]) < 2.0
    assert r['rigid_modes'] == 6


def test_plate_against_leissa():
    for fea_f, ref, err in FE.validate_plate():
        assert abs(err) < 3.0


def test_mesh_convergence():
    c = FE.convergence_beam((4, 8, 16))
    errs = [abs(e) for _, _, e in c]
    assert errs[0] > errs[1] > errs[2] and errs[2] < 1.0


@pytest.mark.parametrize('pid', sorted(FR.PROFILES))
def test_equivalent_tube_matches_catalogue(pid):
    pr = FR.PROFILES[pid]
    bo, do, t = FE.equivalent_tube(pr)
    bi, di = bo - 2 * t, do - 2 * t
    assert bi > 0 and di > 0
    assert bo * do - bi * di == pytest.approx(pr.area, rel=5e-3)
    assert (bo * do ** 3 - bi * di ** 3) / 12 == pytest.approx(pr.i_d, rel=5e-3)
    assert (do * bo ** 3 - di * bi ** 3) / 12 == pytest.approx(pr.i_b, rel=5e-3)


def test_member_block_cantilever():
    pr = FR.PROFILES['mm-4040']
    m = FR.Member('M1', 'post', 'post', 'mm-4040', np.zeros(3), np.array([0, 0, 0.6]), np.array([1.0, 0, 0]))
    b = FE.member_block(m, 30)
    mesh = FE.Mesh([b])
    K, M = FE.assemble(mesh)
    fixed = np.nonzero(mesh.X[:, 2] < 1e-9)[0]
    free = np.setdiff1d(np.arange(K.shape[0]), (3 * fixed[:, None] + np.arange(3)).ravel())
    f, _ = FE.eigen(K, M, 2, free)
    ref = FE.euler_bernoulli(FE.EB_BETA_CANTILEVER[0], 0.6, pr.E, pr.i_d, pr.rho, pr.area)
    assert f[0] == pytest.approx(ref, rel=0.015)
    assert M.sum() / 3 == pytest.approx(pr.kg_m * 0.6, rel=1e-6)


def test_tied_post_and_plate_against_rayleigh():
    """A post with its top plate tied on, fixed at its foot: the FE frequency sits just under Rayleigh's upper bound
    (a tie that locked the plate's rotation would double the stiffness)."""
    p = scene()
    fr = FR.synthesise(p)
    post = next(m for m in fr.members if m.role == 'post')
    f3 = copy.copy(fr)
    f3.members, f3.feet, f3.mounts = [post], [], []
    f3.joints = [j for j in fr.joints if j['kind'] == 'plate']
    mdl = FE.build_model(p, f3, FE.FEASettings(mesh_mm=30))
    X = mdl.mesh.X
    b = mdl.mesh.blocks[0]
    fixed = b.nodes[np.abs(X[b.nodes, 2] - post.p0[2]) < 1e-9]
    free = np.setdiff1d(np.arange(mdl.n_dof), (3 * fixed[:, None] + np.arange(3)).ravel())
    f, _ = FE.eigen(mdl.K, mdl.M, 2, free)
    pr = post.prof
    plate = fr.plates[0]
    from vantage.core.thermo_vib import cantilever_f1
    ray = cantilever_f1(pr.E, pr.i_b, post.length, plate.mass, pr.kg_m * post.length)
    assert 0.95 * ray < f[0] <= 1.01 * ray


def test_settings_tolerant():
    s = FE.FEASettings.from_dict({'mesh_mm': -5, 'modes': 'x', 'bc': 'glue', 'zeta': 2})
    assert s.mesh_mm == 5.0 and s.modes == 12 and s.bc == 'fixed' and s.zeta == 0.3


# ------------------------------------------------------------------ the frame
@pytest.fixture(scope='module')
def frame_res():
    p = scene()
    from vantage.core import aero as AE
    wind = AE.run(p)
    return p, FE.run(p, settings=FE.FEASettings(mesh_mm=60, plate_cells_mm=40, modes=6), wind=wind)


def test_frame_modes(frame_res):
    p, r = frame_res
    f = [m['f_hz'] for m in r.modes]
    assert len(f) == 6 and f == sorted(f) and f[0] > 5
    for m in r.modes:
        assert all(0 <= v <= 100.0001 for v in m['eff_mass_pct'].values())
        assert 'L1' in m['devices']
    assert all(v <= 100.0001 for v in r.stats['cum_eff_mass_pct'].values())
    assert r.modes[0]['part'] and r.modes[0]['motion'] in ('fore-aft', 'side to side', 'up and down', 'twisting')
    # the lidar rides on the mast: the first two modes sway it most
    assert max(r.modes[0]['devices']['L1']['move_abs_mm'], r.modes[1]['devices']['L1']['move_abs_mm']) > 0.8
    assert len(r.stats['ties']) == 8 and all(t['nodes'] > 0 for t in r.stats['ties'])


def test_frame_road_jitter_beside_beam(frame_res):
    p, r = frame_res
    d = next(x for x in r.devices if x['tag'] == 'L1')
    assert d['flex_deg_rms'] > 0 and d['angle_deg_rms'] >= d['flex_deg_rms'] * 0.5
    assert d['beam']['flex_deg_rms'] > 0 and d['ratio_flex'] > 0
    assert r.beams and r.beams[0]['beam_f1_hz'] > 0


def test_frame_wind_case(frame_res):
    p, r = frame_res
    w = r.wind['devices']['L1']
    assert w['move_abs_mm'] > 0
    # the crosswind blows from the left and the car drives forward: L1 is pushed back and to the right
    assert w['move_mm'][0] < 0 and w['move_mm'][1] < 0


def test_static_load_linear(frame_res):
    p, r = frame_res
    fr = FR.synthesise(p)
    mdl = FE.build_model(p, fr, FE.FEASettings(mesh_mm=60, plate_cells_mm=40))
    uid = next(b.uid for b in mdl.bodies if b.tag == 'L1')
    a = FE.static_wind(mdl, {uid: (10.0, 0, 0)})['devices']['L1']
    b = FE.static_wind(mdl, {uid: (20.0, 0, 0)})['devices']['L1']
    assert b['move_mm'][0] == pytest.approx(2 * a['move_mm'][0], rel=1e-6) and a['move_mm'][0] > 0


def test_outputs(frame_res, tmp_path):
    p, r = frame_res
    paths = FE.export(r, p, str(tmp_path))
    doc = json.load(open(tmp_path / 'fea.json'))
    assert doc['schema'] == FE.SCHEMA and len(doc['modes']) == 6
    assert (tmp_path / 'fea_mode1.png').read_bytes()[:4] == b'\x89PNG'
    assert len(open(tmp_path / 'fea_modes.csv').read().splitlines()) == 7
    s = FE.summary(r)
    assert s['modes_hz'][0] == pytest.approx(r.modes[0]['f_hz'], abs=0.01)
    json.dumps(s)
    assert FE.cached(p) is r


def test_cancel_and_empty():
    p = scene()
    assert FE.run(p, cancel=lambda: True) is None
    from vantage.core import samples
    q = samples.tigor_sample_project()
    r = FE.run(q)
    assert r.modes == [] and r.findings[0]['kind'] == 'empty'


def test_spring_feet_lower_the_modes():
    p = scene()
    fr = FR.synthesise(p)
    a = FE.run(p, settings=FE.FEASettings(mesh_mm=80, plate_cells_mm=50, modes=2), frame=fr)
    b = FE.run(p, settings=FE.FEASettings(mesh_mm=80, plate_cells_mm=50, modes=2, bc='spring', k_foot_z=50, k_foot_xy=20),
               frame=fr)
    assert b.modes[0]['f_hz'] < a.modes[0]['f_hz']


def test_validation_checks():
    for c in FE.validation_checks():
        out = c['fn']()
        assert out['error'] <= c['tolerance'] and out.get('ok', True), (c['id'], out)


# ------------------------------------------------------------------ the page
from test_gui import app, win, pump, settle, _wait   # noqa: E402,F401  (fixtures)


def _open(win, name, cls):
    if name not in win._page_widgets:
        pg = cls(win.ctx)
        win.pages.addWidget(pg)
        win._page_widgets[name] = pg
    win.set_page(name)
    return win._page_widgets[name]


def test_page(app, win, tmp_path):
    from vantage.core.rig import apply_fit, fit_plan
    from vantage.ui.workspaces.fea_ws import FEAPage
    s = win.ctx.store

    def fit(p):
        p.active_uid = next(v for v in p.variants if v.sensors).uid
        for k in ('roof_rack', 'sensor_mast'):
            apply_fit(p, fit_plan(p, k))
        head = next(r for r in p.rig if r.name.endswith('head'))
        L1 = next(x for x in p.variant.sensors if x.tag == 'L1')
        L1.x, L1.y = head.x, head.y
        L1.z = round(head.z + head.height / 2 + p.sensor_model_for(L1).optical_z_mm / 1000.0, 4)
    s.edit(['rig', 'variant'], fit, 'fit a rack and a mast, L1 on its head')
    pg = _open(win, 'fea_test', FEAPage)
    pg.activate()
    pg.set_option('mesh_mm', 80.0)
    pg.set_option('plate_cells_mm', 50.0)
    pg.set_option('modes', 4)
    assert FE.settings_of(s.project).modes == 4
    s.undo.undo()
    assert FE.settings_of(s.project).modes == 12
    pg.set_option('modes', 4)
    pg.sw_wind.setChecked(False)
    pg.run()
    _wait(app, lambda: pg.result is not None and not pg.busy, 300)
    assert pg.t_modes.rowCount() == 4 and pg.t_jit.rowCount() >= 1
    pg.t_modes.selectRow(1)
    pump(app, 0.2)
    assert pg.mode == 1 and 'Mode 2' in pg.mode_text.text()
    img = pg.view.render_image(300, 220)
    assert img is not None and not img.isNull()
    # an edit makes it stale
    pg.set_option('zeta', 0.03)
    pump(app, 0.1)
    assert pg.stale and 'out of date' in pg.status.text()
    paths = pg.export(str(tmp_path))
    assert any(x.endswith('fea.json') for x in paths)
