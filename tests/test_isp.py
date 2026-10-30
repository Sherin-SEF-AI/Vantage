"""ISP auto-exposure metering masks and the photometric seam model: the mask's horizon against a pinhole closed form,
the body excluded where the synthetic capture draws it, the CIE clear sky against its classic formula, the AE
against the metered mean, the stitcher's gain solve against core.stitch, the grid and AE rectangles, a tunnel exit's
EV step against an independent ray count, determinism, the export, the Issues check and the page (offscreen)."""
import copy
import json
import math
import types

import cv2
import numpy as np
import pytest

from vantage.core import isp_photometrics as I
from vantage.core import samples, stitch, synth
from vantage.core.analysis import Engine
from vantage.core.lens import Lens, Module
from vantage.core.project import Camera, Project

FAST = {'mask_scale': 8, 'day_step': 120, 'minutes': 90.0, 'heading_step': 90.0}


def _data_rig():
    p = samples.tigor_sample_project()
    p.active_uid = next(v for v in p.variants if v.name.startswith('Data rig')).uid
    return p


@pytest.fixture(scope='module')
def rig():
    p = _data_rig()
    eng = Engine()
    return p, eng


@pytest.fixture(scope='module')
def fast_res(rig):
    p, eng = rig
    return I.run(p, eng, I.IspSettings.from_dict(FAST))


# ----------------------------------------------------------------------------- settings
def test_settings_are_tolerant_and_round_trip():
    s = I.IspSettings.from_dict({'grid_cols': '8', 'grid_rows': 1000, 'horizon_margin_deg': -4, 'metering': 'spot',
                                 'scenario': 'tunnel', 'glare_k': float('nan'), 'mask_scale': 2.6, 'bogus': 1,
                                 'body_albedo': True, 'night_condition': '  night  '})
    assert (s.grid_cols, s.grid_rows, s.horizon_margin_deg, s.metering, s.scenario) == (8, 64, 0.0, 'mean_log',
                                                                                         'tunnel')
    assert s.glare_k == 1.0 and s.mask_scale == 3 and s.body_albedo == 0.30 and s.night_condition == 'night'
    assert I.IspSettings.from_dict(s.to_dict()) == s
    assert I.IspSettings.from_dict(None) == I.IspSettings() == I.IspSettings.from_dict('junk')
    p = Project()
    p.analysis.isp = {'grid_cols': 32}
    assert I.settings_of(p).grid_cols == 32


# ----------------------------------------------------------------------------- mask geometry
def _pinhole(pitch=10.0, z=1.0, yaw=0.0):
    p = Project()
    p.modules['m'] = Module(id='m', name='Test module', width_px=1920, height_px=1536, pixel_um=3.0)
    p.lenses['pin'] = Lens(id='pin', name='Pinhole 6 mm', focal_mm=6.0, projection='rectilinear')
    p.variant.cameras.append(Camera(tag='F', slot=0, x=0.0, y=0.0, z=z, yaw=yaw, pitch=pitch, roll=0.0, module_id='m',
                                    lens_id='pin'))
    return p, p.variant.cameras[0]


@pytest.mark.parametrize('pitch,margin', [(10.0, 2.0), (25.0, 5.0), (-5.0, 0.0)])
def test_sky_boundary_row_matches_the_pinhole_closed_form(pitch, margin):
    """A level pinhole camera pitched down by p: the horizon is the row cy − f·tan p, and the mask's sky ends on the
    row cy − f·tan(p − margin) (closed form, f = 6 mm / 3 µm = 2000 px)."""
    p, cam = _pinhole(pitch)
    m = I.camera_mask(p, None, cam, margin)
    assert m.shape == (1536, 1920)
    f, cy = 2000.0, 768.0
    col = m[:, 960]
    first = int(np.argmax(col == I.METERED))
    assert (col[:first] == I.SKY).all() and (col[first:] == I.METERED).all()
    want = cy - f * math.tan(math.radians(pitch - margin))          # the boundary, in pixel-edge coordinates
    assert abs(first - want) <= 1.0
    # the horizon polyline as the rig health view draws it
    from vantage.core.lens import resolve
    _, W2O = I._cam_frames(cam)
    hz = I.horizon_lines(resolve(p.module_for(cam), p.lens_for(cam)), W2O)
    v_h = cy - f * math.tan(math.radians(pitch))
    if 0 <= v_h <= 1536:
        pts = np.vstack(hz)
        k = np.argmin(np.abs(pts[:, 0] - 960.0))
        assert pts[k, 1] == pytest.approx(v_h, abs=0.5)
    else:
        assert hz == []                                            # above the image: no horizon drawn
    # without a body nothing is body; every class sums to the frame
    g = I.camera_geometry(p, None, cam, I.IspSettings(horizon_margin_deg=margin, mask_scale=4))
    assert g.shares['body'] == g.shares['housing'] == g.shares['outside'] == 0.0
    assert sum(g.shares.values()) == pytest.approx(1.0)


def test_body_pixels_are_excluded_where_the_synthetic_capture_draws_the_body(rig):
    """core.synth's capture with the vehicle against the same capture without it: every pixel the vehicle changes
    is masked as body, rig or housing, and the masked body pixels are where the capture draws it."""
    p, eng = rig
    nobody = types.SimpleNamespace(vehicle_geom=lambda *a, **k: None)
    vg = eng.vehicle_geom(p, p.variant)
    for tag in ('ML', 'F'):
        cam = p.variant.by_tag(tag)
        w, h = 240, 192
        with_body = synth.render_capture(p, eng, cam, w, h, targets=[], noise=0).astype(int)
        bare = synth.render_capture(p, nobody, cam, w, h, targets=[], noise=0).astype(int)
        drawn = with_body != bare
        m = I.camera_mask(p, eng, cam, 2.0, w, h, vg)
        veh = (m == I.BODY) | (m == I.HOUSING)
        assert drawn.sum() > 0.03 * drawn.size
        assert (veh & drawn).sum() / drawn.sum() > 0.995          # all the body the capture shows is masked
        assert (veh & drawn).sum() / veh.sum() > 0.95             # (a few body pixels match the bare grey by chance)
        assert not (m[veh] == I.METERED).any()


# ----------------------------------------------------------------------------- photometry
def _cie_1973(Z, chi, Zs):
    """The CIE (1973) clear sky in its classic printed form (Kittler)."""
    num = (0.91 + 10 * np.exp(-3 * chi) + 0.45 * np.cos(chi) ** 2) * (1 - np.exp(-0.32 / np.cos(Z)))
    den = (0.91 + 10 * np.exp(-3 * Zs) + 0.45 * np.cos(Zs) ** 2) * (1 - np.exp(-0.32))
    return num / den


def test_cie_clear_sky_against_its_formula():
    rng = np.random.default_rng(3)
    Z = rng.uniform(0.0, 1.5, 200)
    chi = rng.uniform(0.0, math.pi, 200)
    Zs = rng.uniform(0.05, 1.5, 200)
    assert np.allclose(I.cie_relative(Z, chi, Zs), _cie_1973(Z, chi, Zs), rtol=2e-3)
    zs = math.radians(40.0)
    assert float(I.cie_relative(0.0, zs, zs)) == pytest.approx(1.0)          # the zenith is L_z
    # sky_luminance: the zenith's luminance is Karayel's, nothing below the horizon
    sun = np.array([math.cos(math.radians(50)), 0.0, math.sin(math.radians(50))])
    L = I.sky_luminance(np.array([[0.0, 0.0, 1.0], [1.0, 0.0, -0.1]]), sun, 50.0, 3.0)
    assert L[0] == pytest.approx(((1.376 * 3 - 1.81) * math.tan(math.radians(50)) + 0.38) * 1000.0) and L[1] == 0.0
    # the hemisphere integration: a uniform sky gives π, cos-weighted
    els, tab = I._diffuse_table((0.0, -1.0, 0.0, -1.0, 0.0))
    assert np.allclose(tab, math.pi, rtol=2e-4)
    # air mass (Kasten & Young) and the IES direct illuminance
    assert float(I.air_mass(90.0)) == pytest.approx(1.0, abs=1e-3)
    assert float(I.air_mass(30.0)) == pytest.approx(1.0 / (0.5 + 0.50572 * 36.07995 ** -1.6364))
    assert float(I.direct_normal_lx(90.0)) == pytest.approx(127500 * math.exp(-0.21 * float(I.air_mass(90.0))))
    d = I.daylight(np.array([10.0, 45.0]))
    assert (d['E_dh'] > 3000).all() and (d['E_dh'] < 20000).all()             # typical clear-sky diffuse
    assert d['E_h'][1] == pytest.approx(d['E_dn'][1] * math.sin(math.radians(45)) + d['E_dh'][1])


def test_ae_exposure_against_the_metered_mean():
    # a uniform scene: L_eff is its luminance
    L = np.full((1, 10), 800.0)
    assert I.ae_solve(L, np.ones(10))[0] == pytest.approx(math.log2(800.0), abs=1e-9)
    # two luminances: the mean of logs (geometric mean), or the linear mean, weighted
    L = np.array([[100.0, 100.0, 100.0, 6400.0]])
    w = np.array([1.0, 1.0, 1.0, 1.0])
    assert I.ae_solve(L, w, 'mean_log')[0] == pytest.approx((3 * math.log2(100) + math.log2(6400)) / 4)
    wc = np.array([0.5, 1.0, 1.0, 2.0])
    assert I.ae_solve(L, wc, 'centre')[0] == pytest.approx(math.log2((0.5 * 100 + 200 + 2 * 6400) / 4.5))
    # masked rays (weight 0) do not count
    assert I.ae_solve(L, np.array([1.0, 1.0, 1.0, 0.0]))[0] == pytest.approx(math.log2(100.0))
    # clipping: a sun-bright ray saturates at the HDR top; the solved exposure meets the target on the clipped values
    L = np.array([[1000.0, 1000.0, 1000.0, 1e9]])
    u = I.ae_solve(L, np.ones(4), 'mean_log', lo=1e-4, hi=16.0)[0]
    y = np.clip(math.log2(0.18) + np.log2(L[0]) - u, math.log2(1e-4), 4.0)
    assert y.mean() == pytest.approx(math.log2(0.18), abs=1e-9) and u < (3 * math.log2(1000) + math.log2(1e9)) / 4
    # EV100 and the exposure: t·g = N²K/(L·S) while within the module's limits
    st = I.IspSettings()
    e = I.exposure_setting(np.log2([2000.0, 1.0, 0.1]), 2.0, 30.0, 1536, st)
    H = 4.0 * 12.5 / (np.array([2000.0, 1.0, 0.1]) * 100.0)
    assert np.allclose(e['t_s'][:2] * e['gain'][:2], H[:2]) and e['gain'][0] == 1.0
    assert np.all(e['short_ev'][:2] == 0.0)
    assert e['t_s'][1] == pytest.approx(1 / 30) and e['gain'][1] == pytest.approx(15.0)          # gain makes up
    assert e['gain'][2] == pytest.approx(10 ** 1.5)                                             # the gain limit
    assert e['short_ev'][2] == pytest.approx(math.log2(H[2] / (10 ** 1.5 / 30)))
    assert e['log2_leff'][2] == pytest.approx(math.log2(0.1) + e['short_ev'][2])                # darker by that
    assert float(I.ev100(math.log2(1000.0))) == pytest.approx(math.log2(1000 * 100 / 12.5))


def test_balance_gains_is_core_stitch_estimate_gains():
    """The vectorised solve against core.stitch.estimate_gains on a stitched grid whose overlaps are uniform
    blocks of the same grey levels (one block per camera pair)."""
    rng = np.random.default_rng(11)
    n = 5
    pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 2)]
    for trial in range(6):
        Ia = rng.uniform(20, 240, len(pairs))
        Ib = rng.uniform(20, 240, len(pairs))
        if trial == 5:
            Ia[0], Ib[0] = 250.0, 30.0                       # one pair saturated: both leave it out
        blk = 16
        uids = [f'c{i}' for i in range(n)]
        warped = {u: np.zeros((blk, blk * len(pairs), 3), np.float32) for u in uids}
        valid = {u: np.zeros((blk, blk * len(pairs)), bool) for u in uids}
        for k, (i, j) in enumerate(pairs):
            sl = (slice(None), slice(k * blk, (k + 1) * blk))
            warped[uids[i]][sl] = Ia[k]
            warped[uids[j]][sl] = Ib[k]
            valid[uids[i]][sl] = True
            valid[uids[j]][sl] = True
        lut = types.SimpleNamespace(uids=uids, body=np.zeros((blk, blk * len(pairs)), bool))
        ref = stitch.estimate_gains(lut, warped=(warped, valid))
        N = (blk // 2) ** 2                                    # overlap_means' every-second-pixel count
        g, ok = I.balance_gains(Ia[None], Ib[None], [(i, j, N) for i, j in pairs], n)
        assert np.allclose(g[0], [ref[u][0] for u in uids], rtol=1e-5)
        assert ok[0, 0] == (trial != 5)
    assert I.GAIN_RANGE == (0.4, 2.5)
    g, _ = I.balance_gains(np.array([[240.0]]), np.array([[10.0]]), [(0, 1, 100)], 2)
    assert g[0, 0] == pytest.approx(0.4) or g[0, 1] == pytest.approx(2.5)


def test_grid_and_rectangles_of_a_known_mask():
    mask = np.full((96, 128), I.SKY, np.uint8)
    mask[48:, :] = I.METERED
    mask[80:, 32:96] = I.BODY                                  # the bumper in the bottom middle (cells 2–5)
    grid, frac = I.metering_grid(mask, 8, 6, 15)
    assert grid.shape == (6, 8) and (grid[:3] == 0).all() and (grid[3:5] == 15).all()
    assert grid[5, 0] == 15 and grid[5, 3] == 0
    rects = I.regions(grid, frac, (1920, 1440), 15, 4)
    assert rects[0] == {'left': 0, 'top': 720, 'right': 1920, 'bottom': 1200, 'weight': 1.0, 'cells': [0, 3, 8, 5]}
    assert all(r[k] % 2 == 0 for r in rects for k in ('left', 'top', 'right', 'bottom'))
    cov = I.rect_cover(mask, rects, (1920, 1440))
    assert cov['metered_inside'] == pytest.approx(1.0) and cov['spill'] == pytest.approx(0.0)
    # centre weighting scales the weights down away from the centre
    cw = np.ones((6, 8))
    cw[:, 0] = 0.5
    g2, _ = I.metering_grid(mask, 8, 6, 15, cw)
    assert g2[4, 0] == 8 and g2[4, 3] == 15


# ----------------------------------------------------------------------------- scenes
def test_tunnel_exit_gives_the_expected_ev_step(rig):
    """No veil, mean-log metering: the rear camera sees only the tunnel, so it exposes for the road's luminance; the
    front camera's step over it is the metered share of its rays that leave the portal onto the daylit road times
    log2(L_out / L_in) (an independent ray count here)."""
    p, eng = rig
    st = I.IspSettings.from_dict(dict(FAST, veil_index=0.0, glare_k=0.0, interior_cd=20.0, wall_cd=20.0, exit_m=4.0))
    res = I.run(p, eng, st)
    tun = res.scenarios['tunnel']['cams']
    F, R = res.cam('F'), res.cam('R')
    assert tun[R['uid']]['ev_wanted'] == pytest.approx(float(I.ev100(math.log2(20.0))), abs=1e-6)
    g = res.geoms[F['uid']]
    vg = eng.vehicle_geom(p, p.variant)
    exit_x = float(vg.bbox[1][0]) + 4.0
    d, pos = g.dirs, g.pos
    met = g.cls == I.METERED
    with np.errstate(divide='ignore', invalid='ignore'):
        te = (exit_x - pos[0]) / d[:, 0]
    y_e, z_e = pos[1] + te * d[:, 1], pos[2] + te * d[:, 2]
    out = met & (d[:, 0] > 0) & (te < g.tg) & (np.abs(y_e) < 5.0) & (z_e > 0) & (z_e < 5.5)
    share = out.sum() / met.sum()
    assert 0.05 < share < 0.95
    E_h = float(I.daylight(40.0, st.turbidity)['E_h'])
    L_out = st.ground_albedo * E_h / math.pi
    want = share * math.log2(L_out / 20.0)
    got = tun[F['uid']]['ev_wanted'] - tun[R['uid']]['ev_wanted']
    assert got == pytest.approx(want, abs=1e-6)
    assert got > 0.3                                             # without the veil: half an EV from the road alone


def test_veil_and_shadow_drive_the_low_sun_exposures(fast_res):
    res = fast_res
    ls = res.scenarios['low_sun']['cams']
    F, R = res.cam('F'), res.cam('R')
    assert ls[F['uid']]['veil_pct'] > 5 * ls[R['uid']]['veil_pct'] > 0           # the sun ahead veils the front
    assert ls[F['uid']]['ev'] > ls[R['uid']]['ev']                               # and darkens its exposure
    assert ls[R['uid']]['shadow_share'] > ls[F['uid']]['shadow_share']            # the long shadow lies behind
    night = res.scenarios['night']['cams']
    assert night[F['uid']]['ev'] > night[R['uid']]['ev']                         # the headlamps light the front
    assert all(x['gain_db'] <= res.settings.max_gain_db + 1e-9 for x in night.values())
    # metering the whole frame: the dark night sky brightens the ground (−), the daylit sky darkens it (+)
    assert night[F['uid']]['unmasked_off_ev'] < -0.3
    assert res.cam('R')['campaign']['unmasked_off_mean_ev'] > 0.1


# ----------------------------------------------------------------------------- the whole study
def test_run_on_the_sample_rig(rig, fast_res):
    res = fast_res
    assert [c['tag'] for c in res.cams] == ['F', 'R', 'ML', 'MR', 'QL', 'QR'] and len(res.seams) == 6
    for c in res.cams:
        g = res.geoms[c['uid']]
        assert g.mask.shape == (1536 // 8, 1920 // 8)
        assert 0.1 < c['removed_share'] < 0.6 and c['shares']['body'] + c['shares']['housing'] > 0.02
        assert np.asarray(c['grid']).shape == (16, 16) and np.asarray(c['grid']).max() == 15
        assert 1 <= len(c['rects']) <= 8 and c['rect_cover']['metered_inside'] > 0.8
        assert c['rect_cover']['spill'] < 0.1
        cm = c['campaign']
        assert 8 < cm['ev_min'] <= cm['ev_median'] <= cm['ev_max'] < 17      # daylight on asphalt
    for s in res.seams:
        cm = s['campaign']
        assert cm['after_max_ev'] <= cm['before_max_ev'] + 1e-9
        assert len(s['curve_after']) == res.campaign['slots'] == len(res.campaign['hours'])
        assert cm['worst']['text'] and abs(cm['worst']['after_ev']) == pytest.approx(cm['after_max_ev'])
    assert res.campaign['samples'] > 0 and res.campaign['worst']
    kinds = {f['kind'] for f in res.findings}
    assert kinds and all(f['severity'] in ('good', 'note', 'warning', 'serious', 'critical') for f in res.findings)
    # determinism: the same inputs give the same numbers, bit for bit
    p = rig[0]                                    # (the sample makes new uids each time it is built)
    again = I.run(p, Engine(), I.IspSettings.from_dict(FAST))
    a, b = I.to_doc(res, p), I.to_doc(again, p)
    a['metrics'].pop('seconds')
    b['metrics'].pop('seconds')
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_to_doc_and_files(fast_res, tmp_path):
    res = fast_res
    p = _data_rig()
    doc = I.to_doc(res, p)
    assert doc['schema'] == I.SCHEMA and doc['frame'] == I.FRAME and 'limits' in doc and doc['units']['luminance']
    assert doc['settings'] == res.settings.to_dict() and set(doc['scenarios']) == {'low_sun', 'tunnel', 'night'}
    json.dumps(doc)
    names = [n for n, _ in I.files(res, p)]
    assert names[:4] == ['isp.json', 'isp_grids.json', 'isp_regions.json', 'isp_seams.csv']
    assert {f'isp_mask_{t}.png' for t in ('F', 'R', 'ML', 'MR', 'QL', 'QR')} <= set(names)
    paths = I.export(res, p, str(tmp_path))
    m = cv2.imread(str(tmp_path / 'isp_mask_F.png'), cv2.IMREAD_UNCHANGED)
    assert m.shape == res.geoms[res.cam('F')['uid']].mask.shape and set(np.unique(m)) <= {0, 255}
    assert (m == 255).mean() == pytest.approx(res.cam('F')['shares']['metered'])
    grids = json.loads((tmp_path / 'isp_grids.json').read_text())
    assert grids['cols'] == 16 and len(grids['cameras']['F']['weights']) == 16 and 'V4L2' in grids['note']
    regs = json.loads((tmp_path / 'isp_regions.json').read_text())
    assert 'setAeRegions' in regs['libargus'] and 'AcRegion' in regs['libargus']
    r0 = regs['cameras']['F']['regions'][0]
    assert 0 <= r0['left'] < r0['right'] <= 1920 and 0 <= r0['top'] < r0['bottom'] <= 1536 and 0 < r0['weight'] <= 1
    rows = (tmp_path / 'isp_seams.csv').read_text().splitlines()
    assert rows[0] == 'seam,local_hour,worst_before_ev,worst_after_ev' and len(rows) == 1 + 6 * res.campaign['slots']
    assert len(paths) == len(names)


def test_issues_flag_a_camera_the_body_dominates(rig):
    p, eng = rig
    q = copy.deepcopy(p)
    assert I.issues(q, eng) == []                                 # the sample's cameras see at most ~17 % body
    q.analysis.isp = {'body_limit': 0.1}
    out = I.issues(q, eng)
    assert {f['message'].split(':')[0] for f in out} >= {'ML', 'MR', 'QL', 'QR'}
    assert all(f['severity'] == 'warning' and f['kind'] == 'isp_body_metering' for f in out)


# ----------------------------------------------------------------------------- the page
from test_gui import app, win, pump, _wait     # noqa: E402,F401  (fixtures)


def test_page_runs_shows_edits_and_exports(app, win, tmp_path):
    from vantage.ui.workspaces.isp_ws import IspPage
    store = win.store
    rig_v = next(v for v in store.project.variants if v.name.startswith('Data rig'))
    store.edit('variants', lambda pj: setattr(pj, 'active_uid', rig_v.uid), f'Switched to {rig_v.name}')
    store.edit('analysis', lambda pj: setattr(pj.analysis, 'isp', dict(FAST)), 'Fast ISP study')
    pump(app, 0.1)
    pg = IspPage(win.ctx)
    pg.resize(1500, 900)
    pg.show()
    _wait(app, lambda: pg.res is not None and not pg.busy, 120)
    pump(app, 0.1)
    res = pg.res
    assert pg.table.rowCount() == 6 and pg.table.item(0, 0).text() == 'F'
    assert pg.table.item(0, 1).text() == f'{100 * res.cam("F")["removed_share"]:.0f}'
    assert len(pg.findings) == len(res.findings) and set(res.previews) == {c['uid'] for c in res.cams}
    assert res.previews[res.cams[0]['uid']]['rgb'].shape[1] == 320
    assert pg.gallery.images and pg.grid_view.cam['tag'] == 'F'
    ser = pg.chart.series()
    assert len(ser) == 6 and len(ser[0][2]) == res.campaign['slots']
    assert 'daylight samples' in pg.status.text()
    # the scenario selector: an undoable setting; the chart and the table follow without a re-run
    pg.scenario.buttons['tunnel'].click()
    pump(app, 0.1)
    assert store.project.analysis.isp['scenario'] == 'tunnel' and pg.res is res and not pg.stale
    ser = pg.chart.series()
    assert ser[0][2][0][1] == pytest.approx(abs(res.scenarios['tunnel']['seams'][0]['before_ev']))
    assert pg.table.item(0, 5).text() == f'{res.scenarios["tunnel"]["cams"][res.cam("F")["uid"]]["exposure_ms"]:.2f}'
    store.undo.undo()
    pump(app, 0.1)
    assert 'scenario' not in store.project.analysis.isp and pg.scenario.value() == 'campaign'
    # a setting the study reads: saved, re-run by itself, undone
    pg.set_option('horizon_margin_deg', 6.0)
    assert store.project.analysis.isp['horizon_margin_deg'] == 6.0
    _wait(app, lambda: pg.res is not res and not pg.busy and not pg.stale, 120)
    pump(app, 0.1)
    assert pg.res.settings.horizon_margin_deg == 6.0
    assert pg.res.cam('F')['shares']['sky'] > res.cam('F')['shares']['sky']
    assert pg.fields['horizon_margin_deg'][0].value() == pytest.approx(6.0)
    store.undo.undo()
    assert store.project.analysis.isp.get('horizon_margin_deg') is None
    _wait(app, lambda: not pg.busy and not pg.stale and pg.res.settings.horizon_margin_deg == 2.0, 120)
    # select a camera from the gallery
    pg.select(pg.res.cam('QL')['uid'])
    assert pg.grid_view.cam['tag'] == 'QL' and pg.gallery.sel == pg.res.cam('QL')['uid']
    # cancel a run
    pg.run()
    assert pg.busy
    pg.cancel()
    assert not pg.busy and 'Cancelled' in pg.status.text()
    pump(app, 0.3)
    # export
    paths = pg.export(str(tmp_path))
    assert any(x.endswith('isp.json') for x in paths) and (tmp_path / 'isp_mask_QL.png').exists()
    pg.close()
    pg.deleteLater()
    pump(app, 0.1)
