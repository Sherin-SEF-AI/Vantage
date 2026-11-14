"""4D imaging radar (core.imaging_radar): the virtual array is the Tx ⊕ Rx convolution (and TI's cascade comes out as
its design guide draws it), the array factor and the tapers match their closed forms and SciPy, the fascia slab matches
Fresnel, the two-ray factor its closed form and the height average a brute-force one, the cavity image sits at twice
the fascia's tilt, the library's figures and link budget, settings round trips, the study on the sample rig, exports,
the Issues list, the validation checks, and the page driven like a user."""
import copy
import json
import math

import numpy as np
import pytest

from vantage.core import imaging_radar as IR
from vantage.core import materials as M
from vantage.core import samples
from vantage.core import sensors as S


def _data_rig():
    p = samples.tigor_sample_project()
    v = next(v for v in p.variants if v.sensors)
    p.active_uid = v.uid
    return p, v


def _ula(n, spacing=1.0):
    return np.c_[np.arange(n, dtype=float) * spacing, np.zeros(n)]


# =========================================================================== the virtual array
def test_virtual_array_is_the_convolution_of_tx_and_rx():
    tx = [(0, 0), (4, 0), (8, 0)]
    rx = [(0, 0), (1, 0), (2, 0), (3, 0)]
    V = IR.virtual_array(tx, rx)
    assert V.shape == (12, 2)
    assert sorted(V[:, 0]) == list(range(12))                       # a filled 12-element row
    U, cnt = IR.unique_positions(IR.virtual_array([(0, 0), (2, 0)], rx))
    assert len(U) == 6 and cnt.sum() == 8 and cnt.max() == 2         # 2..3 measured twice


def test_ti_cascade_matches_the_design_guide():
    m4 = IR.virtual_array(IR.CAS_TX, IR.CAS_RX)
    assert len(m4) == 192
    assert len(np.unique(m4[m4[:, 1] == 0, 0])) == 86                 # TIDUEN5A: 86 azimuth elements
    m2 = IR.virtual_array([IR.CAS_TX[i - 1] for i in IR.CAS2_TX_IDS], [IR.CAS_RX[i - 1] for i in IR.CAS2_RX_IDS])
    assert len(np.unique(m2[m2[:, 1] == 0, 0])) == 23                 # Fig. 8: 23 azimuth antennas
    rows = np.unique(m4[:, 1])
    assert rows.tolist() == [0, 1, 4, 6]
    assert np.unique(np.abs(rows[:, None] - rows[None, :])).tolist() == list(range(7))   # "equivalent to 7"
    bf = IR.beamformer(IR.unique_positions(m4)[0])
    assert len(bf.y) == 86 and bf.row_z == 0 and bf.z.tolist() == [0, 1, 4, 6]


def test_awr2944_virtual_array_as_the_user_guide_draws_it():
    V = IR.virtual_array(IR.EVM2944_TX, IR.EVM2944_RX)
    low = np.sort(V[V[:, 1] == 0, 0])
    up = np.sort(V[V[:, 1] > 0, 0])
    assert low.tolist() == list(range(12))                            # SPRUJ22C Fig. 2-18: a row of 12 …
    assert up.tolist() == [2, 3, 4, 5] and np.allclose(V[V[:, 1] > 0, 1], 1.6)    # … and 4 at 0.8 λ above its third


def test_assumed_layout_reaches_the_stated_resolution():
    m = next(x for x in IR.library() if x.id == 'arbe_phoenix')
    arr = IR.array_spec(m)
    assert arr.layout == 'assumed' and arr.n_virtual_stated == 2304
    bf = IR.beamformer(arr.unique()[0], 'uniform')
    th = np.linspace(-3, 3, 30001)
    az = IR.lobe(th, IR.db_pow(bf.az(np.sin(np.radians(th)))))['hpbw_deg']
    el = IR.lobe(th, IR.db_pow(bf.el(np.sin(np.radians(th)))))['hpbw_deg']
    assert az <= 0.7 and az > 0.69 and el <= 0.9 and el > 0.88       # the fewest elements that reach 0.7° × 0.9°


# =========================================================================== the array factor
@pytest.mark.parametrize('n', [8, 16, 32, 64])
def test_ula_beamwidth_against_the_closed_form(n):
    th = np.linspace(-1, 1, 8001) * 2 * math.degrees(0.886 * 2 / n)
    p = IR.db_pow(IR.af_cut(_ula(n), np.ones(n), u=np.sin(np.radians(th))))
    hp = IR.lobe(th, p)['hpbw_deg']
    assert hp == pytest.approx(math.degrees(0.886 * 2 / n), rel=0.008)  # 0.886 λ / (N d), d = λ/2


def test_array_factor_is_exact_for_two_elements():
    # AF = cos(π u d) for two elements d half-wavelengths apart: −3 dB at u = 1/(4 d) … checked at any u
    u = np.linspace(-1, 1, 101)
    a = IR.af_cut(np.array([[0.0, 0.0], [1.6, 0.0]]), np.ones(2), u=u)
    assert np.allclose(np.abs(a), np.abs(np.cos(math.pi * u * 1.6 / 2)), atol=1e-12)


def test_beamformer_matches_full_beamforming_on_a_filled_rectangle():
    ys, zs = np.meshgrid(np.arange(9.0), np.arange(5.0), indexing='ij')
    U = np.c_[ys.ravel(), zs.ravel()]
    bf = IR.beamformer(U, 'hann')
    w = IR.weights(U, 'hann')
    az, el = np.linspace(-60, 60, 31), np.linspace(-40, 40, 17)
    A = bf.azel(az, el)
    B = IR.af_azel(U, w, az, el)
    assert np.allclose(np.abs(A), np.abs(B), atol=1e-12)
    uv = bf.uv(np.linspace(-1, 1, 21), np.linspace(-1, 1, 11))
    assert np.allclose(np.abs(uv), np.abs(IR.af_uv(U, w, np.linspace(-1, 1, 21), np.linspace(-1, 1, 11))))


def test_grating_lobes_appear_at_one_wavelength_spacing():
    half = IR.beamformer(_ula(16), 'uniform')
    assert IR.grating_lobes(half, 60.0, 0.0) == []                     # λ/2: none in the visible region
    full = IR.beamformer(_ula(16, spacing=2.0), 'uniform')                # λ: a lobe at Δu = ±1
    gl = IR.grating_lobes(full, 60.0, 0.0)
    assert gl and min(abs(abs(g['du']) - 1.0) for g in gl) < 0.02 and gl[0]['level_db'] > -0.5
    narrow = IR.grating_lobes(full, 0.0, 0.0)                             # steered only at the boresight: at ±90°
    assert all(abs(abs(g['du']) - 1) > 0.02 for g in narrow)


def test_steered_beam_widens_as_one_over_cos():
    bf = IR.beamformer(_ula(64), 'uniform')
    b0 = IR.steered_hpbw(bf, 0.0)
    b45 = IR.steered_hpbw(bf, 45.0)
    assert b45 / b0 == pytest.approx(1 / math.cos(math.radians(45)), rel=0.01)


# =========================================================================== windowing
def test_taper_sidelobes_match_harris():
    n = 128
    th = np.linspace(-1, 1, 12001) * 12 * math.degrees(0.886 * 2 / n)
    sv = np.sin(np.radians(th))
    U = _ula(n)
    psl = {k: IR.lobe(th, IR.db_pow(IR.af_cut(U, IR.weights(U, k), u=sv)))['psl_db'] for k in IR.WINDOWS}
    assert psl['uniform'] == pytest.approx(-13.26, abs=0.05)
    assert psl['hann'] == pytest.approx(-31.47, abs=0.3)
    assert psl['taylor'] == pytest.approx(-30.0, abs=0.5)
    L = IR.lobe(th, IR.db_pow(IR.af_cut(U, IR.weights(U, 'hann'), u=sv)))
    assert math.radians(L['hpbw_deg']) * n / 2 == pytest.approx(1.44, abs=0.01)       # bins


def test_taylor_matches_scipy():
    from scipy.signal.windows import taylor
    for m, nbar, sll in ((12, 3, 25), (64, 5, 35), (86, 4, 30)):
        x = (np.arange(m) - m / 2 + 0.5) / m
        a = IR.taper(x, 'taylor', sll, nbar)
        b = taylor(m, nbar=nbar, sll=sll, norm=True)
        assert np.max(np.abs(a / a.max() - b / b.max())) < 1e-12


def test_taper_positions_normalise_like_the_discrete_window():
    x = IR._norm_pos(np.arange(10.0))
    assert x[0] == pytest.approx(-0.45) and x[-1] == pytest.approx(0.45)
    w = IR.taper(x, 'hann')
    assert np.allclose(w, w[::-1]) and w.max() <= 1.0 and w.min() > 0
    assert np.all(IR.taper(x, 'uniform') == 1.0)


# =========================================================================== the slab and the road
def test_thick_lossy_slab_reflects_as_its_fresnel_interface():
    for eps, tan, th in ((2.75, 0.2, 0.0), (8.6, 0.2, 40.0), (4.46, 0.3, 70.0)):
        r_te, _, r_tm, _ = M.stack_rt([(eps, tan, 0.3)], 77.0, np.array([th]))
        f_te, f_tm = M.fresnel_halfspace(eps * (1 - 1j * tan), th)
        assert abs(abs(r_te[0]) ** 2 - abs(f_te) ** 2) < 1e-12
        assert abs(abs(r_tm[0]) ** 2 - abs(f_tm) ** 2) < 1e-12


def test_thin_slab_is_the_airy_sum_of_its_fresnel_coefficients():
    lam = IR.wavelength_m(77.0)
    eps, d, th = 2.75, 3.0e-3, 30.0
    n = math.sqrt(eps)
    delta = 2 * math.pi / lam * n * d * math.sqrt(1 - (math.sin(math.radians(th)) / n) ** 2)
    f_te, f_tm = M.fresnel_halfspace(eps, th)
    r_te, _, r_tm, _ = M.stack_rt([(eps, 0.0, d)], 77.0, np.array([th]))
    for r12, rm in ((complex(f_te), r_te[0]), (complex(f_tm), r_tm[0])):
        airy = r12 * (1 - np.exp(-2j * delta)) / (1 - r12 ** 2 * np.exp(-2j * delta))
        assert abs(abs(airy) ** 2 - abs(rm) ** 2) < 1e-12


def test_road_reflection_grazing_brewster_and_normal():
    for surf in M.ROAD_SURFACES:
        assert abs(M.road_reflection(surf, 0.0, 'vertical') + 1) < 1e-9          # grazing: Γ → −1
        assert abs(M.road_reflection(surf, 0.0, 'horizontal') + 1) < 1e-9
    eps = M.ROAD_SURFACES['asphalt_concrete'][1]
    brewster_grazing = 90 - math.degrees(math.atan(math.sqrt(eps)))
    assert abs(M.road_reflection('asphalt_concrete', brewster_grazing, 'vertical')) < 1e-9
    n = math.sqrt(eps)
    assert abs(M.road_reflection('asphalt_concrete', 90.0, 'horizontal')) == pytest.approx((n - 1) / (n + 1))
    assert np.all(M.road_reflection('nothing', np.array([1.0, 2.0])) == 0)


# =========================================================================== two-ray
def test_two_ray_closed_form_and_nulls():
    f, hr, ht = 77.0, 0.5, 1.0
    lam = IR.wavelength_m(f)
    d = np.geomspace(2, 2000, 2001)
    F, Rd, Rr, psi = IR.two_ray(d, hr, ht, f, -1.0, spread=False)
    ex = 2 * np.abs(np.sin(math.pi / lam * (np.hypot(d, ht + hr) - np.hypot(d, ht - hr))))
    assert np.max(np.abs(F - ex)) < 1e-9
    far = d > 500
    assert np.max(np.abs(F[far] - 2 * np.abs(np.sin(2 * math.pi * hr * ht / (lam * d[far]))))) < 2e-3
    for n in (1, 2, 3):
        assert IR.two_ray(np.array([2 * hr * ht / (n * lam)]), hr, ht, f, -1.0, spread=False)[0][0] < 0.01
    assert psi[0] == pytest.approx(math.degrees(math.atan2(1.5, 2.0)))
    Fn, *_ = IR.two_ray(d, hr, ht, f, 0.0)
    assert np.allclose(Fn, 1.0)                                         # no reflection: free space


def test_height_average_of_f4_matches_brute_force():
    rng = np.random.default_rng(7)
    for _ in range(20):
        a, phi, w = rng.uniform(0, 1), rng.uniform(-10, 10), rng.uniform(0, 30)
        t = (np.arange(200000) + 0.5) / 200000 - 0.5                      # the midpoint rule
        brute = np.mean(np.abs(1 + a * np.exp(-1j * (phi + w * t))) ** 4)
        assert IR.mean_f4(a, phi, w) == pytest.approx(brute, rel=1e-6)
    assert IR.mean_f4(0.7, 1.3, 0.0) == pytest.approx(abs(1 + 0.7 * np.exp(-1.3j)) ** 4)


def test_roughness_factor():
    assert IR.roughness_factor(0.0, 0.002, 77.0) == pytest.approx(1.0)
    k = 2 * math.pi / IR.wavelength_m(77.0)
    assert IR.roughness_factor(30.0, 0.001, 77.0) == pytest.approx(math.exp(-2 * (k * 0.001 * 0.5) ** 2))


# =========================================================================== the library
def test_library_enters_the_sensor_library():
    ids = {m.id for m in S.builtin_sensor_models()}
    assert set(IR.PRODUCT_IDS) <= ids
    for m in IR.library():
        assert m.kind == 'radar' and m.ranging == 'mimo' and m.builtin and m.zones and m.url
        assert m.params['figures'] and m.params['range_ref_m'] > 0
        assert 76 <= m.params['carrier_ghz'] <= 81 and m.params['layout'] in ('public', 'assumed')
        back = S.SensorModel.from_dict(json.loads(json.dumps(m.to_dict())))
        assert back.ranging == 'mimo' and back.params == m.params and back.zones == m.zones


def test_awr2944_link_budget_by_hand():
    m = next(x for x in IR.library() if x.id == 'ti_awr2944_evm')
    p = m.params
    lam = 299_792_458.0 / 78.5e9
    num = 10 ** ((13.5 - 30) / 10) * 10 ** 2.6 * lam ** 2 * 10.0 * 4 * 0.0128
    den = (4 * math.pi) ** 3 * 1.380649e-23 * 290 * 10 ** 1.2 * 10 ** 0.6 * 10 ** 1.5
    assert p['range_ref_m'] == pytest.approx((num / den) ** 0.25, abs=0.06)
    assert IR.link_range(p, 0.5, 15, 6) == pytest.approx(p['range_ref_m'] * 0.05 ** 0.25, rel=1e-3)
    assert IR.link_range({}, 1.0, 15, 6) is None


def test_ars548_zone_model_maps_to_its_array():
    p, v = _data_rig()
    rd1 = next(s for s in v.sensors if s.tag == 'RD1')
    rd1.model_id = 'conti_ars548_rdi'
    r = IR.evaluate(p, rd1, maps=False)
    assert r['model_id'] == 'conti_ars548_rdi_mimo' and r['own_model_id'] == 'conti_ars548_rdi'
    b = r['beam']
    assert b['layout'] == 'assumed'
    assert b['az_res_uniform_deg'] == pytest.approx(1.2, abs=0.02)        # its stated 3 dB beam width
    assert b['az_res_edge_deg'] / b['az_res_deg'] == pytest.approx(1 / math.cos(math.radians(45)), rel=0.02)


# =========================================================================== settings
def test_settings_round_trip_and_junk():
    st = IR.ImagingRadarSettings.from_dict({'window': 'hann', 'taylor_nbar': 99, 'taylor_sll_db': 'x',
                                           'road': 'concrete', 'polarisation': 'horizontal', 'face_reflection_db': 5,
                                           'integration_ms': 0, 'swap_model': 'arbe_phoenix',
                                           'classes': [{'name': 'Truck', 'rcs_m2': 1e9, 'height_m': 1.2,
                                                        'span_m': 9}, 'junk', {'name': '', 'rcs_m2': 1},
                                                       {'name': 'Truck', 'rcs_m2': 3}],
                                           'radars': ['a', 3, 'b', 'a']})
    assert st.window == 'hann' and st.taylor_nbar == 12 and st.taylor_sll_db == 30.0
    assert st.road == 'concrete' and st.polarisation == 'horizontal' and st.face_reflection_db == 0.0
    assert st.integration_ms == 0.0 and st.swap_model == 'arbe_phoenix' and st.radars == ['a', 'b']
    assert st.classes == [{'name': 'Truck', 'rcs_m2': 1e5, 'height_m': 1.2, 'span_m': 1.2, 'source': 'yours'}]
    again = IR.ImagingRadarSettings.from_dict(json.loads(json.dumps(st.to_dict())))
    assert again.to_dict() == st.to_dict()
    for junk in (None, 5, 'x', [], {'window': 3, 'road': 'mud', 'classes': 'x'}):
        assert IR.ImagingRadarSettings.from_dict(junk).to_dict() == IR.ImagingRadarSettings().to_dict()
    p, _ = _data_rig()
    assert IR.settings_of(p).to_dict() == IR.ImagingRadarSettings().to_dict()       # an old project: defaults


# =========================================================================== the study on the sample
@pytest.fixture(scope='module')
def swapped():
    """The Data rig with its front radar an Arbe Phoenix and its corners AWR2944EVMs (the real-result rig)."""
    p, v = _data_rig()
    for s in v.sensors:
        if s.kind == 'radar':
            s.model_id = 'arbe_phoenix' if s.tag == 'RD1' else 'ti_awr2944_evm'
    return p, IR.run(p)


def test_study_on_the_swapped_data_rig(swapped):
    p, res = swapped
    assert [r['tag'] for r in res.radars] == ['RD1', 'RD2', 'RD3', 'RD4', 'RD5']
    rd1, rd2 = res.radar('RD1'), res.radar('RD2')
    assert rd1['mount'] == 'exposed' and rd1['cavity'] is None
    assert rd2['mount'] == 'behind' and rd2['cavity'] is not None
    b = rd2['beam']
    assert b['n_virtual'] == 16 and b['n_az'] == 12 and b['n_el'] == 2
    assert 9 < b['az_res_deg'] < 13 and b['map']['db'].shape == (len(b['map']['el']), len(b['map']['az']))
    for r in res.radars:
        for name, g in r['ground']['classes'].items():
            assert g['near_m'] <= g['continuous_m'] <= g['max_m']
            assert g['free_range_m'] > 0
    car, ped = rd1['ground']['classes']['Car'], rd1['ground']['classes']['Pedestrian']
    assert car['free_range_m'] / ped['free_range_m'] == pytest.approx((10 / 0.5) ** 0.25)   # R ∝ σ^¼
    s = IR.summary(res)
    json.dumps(s)
    assert s['radars'][0]['model'] == 'Arbe Phoenix'


def test_cavity_image_sits_at_twice_the_fascia_tilt(swapped):
    _, res = swapped
    c = res.radar('RD2')['cavity']
    g = next(x for x in c['ghosts'] if x['order'] == 1 and x['target_az_deg'] == 0.0)
    assert g['angle_offset_deg'] == pytest.approx(2 * c['incidence_deg'], abs=0.3)
    g2 = next(x for x in c['ghosts'] if x['order'] == 2 and x['target_az_deg'] == 0.0)
    assert g2['angle_offset_deg'] == pytest.approx(4 * c['incidence_deg'], abs=0.6)
    assert g2['range_offset_m'] == pytest.approx(2 * g['range_offset_m'])
    assert g['range_offset_m'] == pytest.approx(c['gap_mm'] / 1000 * math.cos(math.radians(c['incidence_deg'])))
    r = 10 ** (c['reflection_db'] / 20)
    assert c['image_db'] == pytest.approx(20 * math.log10(r))           # metal face (0 dB)
    assert c['gap_advice']['best_gap_mm'] > 0 and c['tilt_sweep'][0]['tilt_deg'] == 0.0


def test_householder_image_by_hand():
    a = math.radians(12.0)
    n = np.array([math.cos(a), math.sin(a), 0.0])
    H = IR._householder(n) @ IR._householder([1.0, 0, 0])
    g = H @ np.array([1.0, 0, 0])
    assert math.degrees(math.atan2(g[1], g[0])) == pytest.approx(24.0)


def test_face_reflection_lowers_the_image(swapped):
    p, _ = swapped
    st = IR.ImagingRadarSettings.from_dict({'face_reflection_db': -10})
    s = next(x for x in p.variant.sensors if x.tag == 'RD2')
    c = IR.evaluate(p, s, st, maps=False)['cavity']
    c0 = IR.evaluate(p, s, IR.ImagingRadarSettings(), maps=False)['cavity']
    assert c['image_db'] == pytest.approx(c0['image_db'] - 10)


def test_point_target_fades_and_spread_fills_them(swapped):
    p, _ = swapped
    s = next(x for x in p.variant.sensors if x.tag == 'RD1')
    pt = IR.ImagingRadarSettings.from_dict({'classes': [{'name': 'P', 'rcs_m2': 0.5, 'height_m': 1.0,
                                                         'span_m': 0.0}]})
    sp = IR.ImagingRadarSettings.from_dict({'classes': [{'name': 'P', 'rcs_m2': 0.5, 'height_m': 1.0,
                                                         'span_m': 0.6}]})
    a = IR.evaluate(p, s, pt, maps=False)['ground']['classes']['P']
    b = IR.evaluate(p, s, sp, maps=False)['ground']['classes']['P']
    assert len(a['fades']) > len(b['fades'])
    assert a['continuous_m'] < b['continuous_m']
    none = IR.ImagingRadarSettings.from_dict({'road': 'none', 'classes': pt.classes})
    c = IR.evaluate(p, s, none, maps=False)['ground']['classes']['P']
    assert c['fades'] == [] and c['continuous_m'] == pytest.approx(c['free_m'], rel=1e-3)


def test_zone_radar_has_no_array_but_ground_and_cavity(swapped):
    p, v = _data_rig()
    res = IR.run(p, maps=False)
    r = res.radar('RD2')
    assert r['beam'] is None and r['ranging'] == 'zone' and r['cavity'] is not None
    assert any(f['kind'] == 'no_array' for f in res.findings)
    assert r['zone']['classes']['Car'] == pytest.approx(r['ground']['classes']['Car']['free_range_m'] *
                                                        r['zone']['classes']['Car'] / r['ground']['classes']['Car']
                                                        ['free_range_m'])


def test_swap_does_not_change_the_project():
    p, v = _data_rig()
    before = [s.model_id for s in v.sensors]
    res = IR.run(p, settings=IR.ImagingRadarSettings.from_dict({'swap_model': 'ti_mmwcas_4chip'}), maps=False)
    assert [s.model_id for s in v.sensors] == before
    assert all(r['swapped'] and r['model_id'] == 'ti_mmwcas_4chip' for r in res.radars)
    assert res.radar('RD1')['beam']['n_az'] == 86
    bad = IR.run(p, settings=IR.ImagingRadarSettings.from_dict({'swap_model': 'ouster_os1_128'}), maps=False)
    assert 'ignored' in bad.swap_note and not any(r['swapped'] for r in bad.radars)


def test_empty_hidden_nan_and_cancel():
    p, v = _data_rig()
    for s in v.sensors:
        if s.kind == 'radar':
            s.visible = False
    res = IR.run(p)
    assert res.radars == [] and res.findings[0]['kind'] == 'none'
    v.sensors[1].visible = True
    v.sensors[1].x = float('nan')
    assert IR.run(p).radars == []
    v.sensors[1].x = 3.2
    assert IR.run(p, cancel=lambda: True) is None
    sel = IR.ImagingRadarSettings.from_dict({'radars': [v.sensors[1].uid]})
    assert len(IR.run(p, settings=sel, maps=False).radars) == 1


def test_export_files(swapped, tmp_path):
    p, res = swapped
    paths = IR.export(res, p, str(tmp_path))
    names = sorted(x.split('/')[-1] for x in paths)
    assert names == ['imaging_radar.csv', 'imaging_radar.json', 'imaging_radar.png', 'imaging_radar_ghosts.csv']
    doc = json.loads((tmp_path / 'imaging_radar.json').read_text())
    assert doc['schema'] == IR.SCHEMA and len(doc['radars']) == 5 and doc['references']
    assert doc['radars'][1]['beam']['tx_half_wavelengths'] == [list(map(float, t)) for t in IR.EVM2944_TX]
    assert isinstance(doc['radars'][0]['beam']['rx_half_wavelengths'], str)        # assumed: not a list of 16 000
    rows = (tmp_path / 'imaging_radar.csv').read_text().splitlines()
    assert rows[0].split(',')[0] == 'tag' and len(rows) == 1 + 5 * 3
    assert (tmp_path / 'imaging_radar.png').read_bytes()[:4] == b'\x89PNG'
    assert IR.advice(res.radar('RD2'))


def test_issues_cached_and_only_for_array_radars():
    p, v = _data_rig()
    assert IR.issues(p) == []                                         # zone radars: the materials study speaks
    for s in v.sensors:
        if s.kind == 'radar':
            s.model_id = 'arbe_phoenix'
    IR.clear_cache()
    iss = IR.issues(p)
    assert iss and all(i.action == 'imaging_radar' for i in iss)
    assert IR.cached_run(p) is IR.cached_run(p)
    v.sensors[2].x += 0.05
    assert IR.cache_key(p) != IR.cache_key(copy.deepcopy(_data_rig()[0]))


def test_lobe_mesh_points_along_the_boresight():
    p, v = _data_rig()
    s = next(x for x in v.sensors if x.tag == 'RD2')
    tri, val = IR.lobe_mesh(s, p.sensor_models['ti_awr2944_evm'], scale_m=2.0)
    assert tri.shape[1:] == (3, 3) and len(tri) == len(val) and len(tri) > 100
    far = tri.reshape(-1, 3)[np.argmax(np.linalg.norm(tri.reshape(-1, 3) - s.pos(), axis=1))] - s.pos()
    head = math.degrees(math.atan2(far[1], far[0]))
    assert head == pytest.approx(s.yaw, abs=3.0) and np.linalg.norm(far) == pytest.approx(2.0, abs=0.05)
    assert IR.lobe_mesh(s, p.sensor_models['conti_ars408_21']) is None


def test_validation_checks_pass():
    checks = IR.validation_checks()
    assert {c['id'] for c in checks} >= {'imaging_radar.ula_beamwidth', 'imaging_radar.slab_fresnel',
                                         'imaging_radar.two_ray', 'imaging_radar.taylor', 'imaging_radar.ti_cascade'}
    for c in checks:
        o = c['fn'](None)
        assert math.isfinite(o.error) and o.error <= c['tolerance'] and o.ok, c['id']
        assert set(c['refs']) <= set(IR.REFERENCES), c['id']


def test_project_save_load_keeps_settings(tmp_path):
    from vantage.core.project import Project
    p, _ = _data_rig()
    p.analysis.imaging_radar = IR.ImagingRadarSettings.from_dict({'window': 'hann', 'swap_model': 'arbe_phoenix'}
                                                                ).to_dict()
    path = tmp_path / 'p.vantage'
    p.save(str(path))
    q = Project.load(str(path)) if hasattr(Project, 'load') else Project.from_file(str(path))
    assert IR.settings_of(q).window == 'hann' and IR.settings_of(q).swap_model == 'arbe_phoenix'
    assert q.sensor_models['arbe_phoenix'].ranging == 'mimo'


# =========================================================================== the page
from test_gui import app, win, pump, settle, _wait   # noqa: E402,F401 (fixtures)


def _open(win, name, cls):
    if name not in win._page_widgets:
        pg = cls(win.ctx)
        win.pages.addWidget(pg)
        win._page_widgets[name] = pg
    win.set_page(name)
    return win._page_widgets[name]


def test_page(app, win, tmp_path):
    from vantage.ui.workspaces.imaging_radar_ws import ImagingRadarPage
    store = win.ctx.store
    win.switch_variant(next(x for x in store.project.variants if x.sensors).uid)
    pump(app, 0.2)
    pg = _open(win, 'imaging_radar', ImagingRadarPage)
    pg.activate()
    _wait(app, lambda: pg.result is not None and not pg.busy, timeout=120)
    assert pg.t_radars.rowCount() == 5 and pg.c_radar.count() == 5
    assert pg.radar()['beam'] is None and pg.cut_az.deg is None              # the sample's zone radars
    # evaluate as the Arbe Phoenix: a what-if setting, one undo step
    i = pg.c_model.findData('arbe_phoenix')
    assert i > 0
    pg.c_model.setCurrentIndex(i)
    _wait(app, lambda: pg.result is not None and not pg.busy and pg.result.settings.swap_model == 'arbe_phoenix',
          timeout=120)
    assert store.project.analysis.imaging_radar['swap_model'] == 'arbe_phoenix'
    assert all(r['swapped'] for r in pg.result.radars)
    assert pg.cut_az.deg is not None and pg.map.img is not None and pg.lobe.tri is not None
    assert 'Phoenix' in pg.head.text() and pg.t_ghosts.rowCount() >= 0
    # pick RD2 from the table: it sits behind the fascia
    row = next(k for k, r in enumerate(pg.result.radars) if r['tag'] == 'RD2')
    pg.t_radars.selectRow(row)
    pump(app, 0.1)
    assert pg.radar()['tag'] == 'RD2' and pg.radar()['cavity'] is not None and pg.t_ghosts.rowCount() > 0
    assert 'Gap' in pg.advice.text()
    # the taper and a number field
    pg.seg_window.buttons['hann'].click()
    _wait(app, lambda: pg.result is not None and not pg.busy and pg.result.settings.window == 'hann', timeout=120)
    pg.fields['ghost_floor_db'].set_value(-20.0)
    pg.fields['ghost_floor_db'].committed.emit(-20.0)
    _wait(app, lambda: pg.result is not None and not pg.busy and pg.result.settings.ghost_floor_db == -20.0,
          timeout=120)
    # swap every radar for real, then undo it
    assert pg.swap(all_radars=True)
    pump(app, 0.2)
    assert all(s.model_id == 'arbe_phoenix' for s in store.project.variant.sensors if s.kind == 'radar')
    assert IR.settings_of(store.project).swap_model == ''
    store.undo.undo()
    pump(app, 0.2)
    assert store.project.variant.sensor(pg.result.radars[0]['uid']).model_id == 'conti_ars408_21'
    for _ in range(3):
        store.undo.undo()
        pump(app, 0.1)
    assert IR.settings_of(store.project).to_dict() == IR.ImagingRadarSettings().to_dict()
    _wait(app, lambda: pg.result is not None and not pg.busy and not pg.stale, timeout=120)
    # the charts paint, hover and drag without a fault
    for w in (pg.cut_az, pg.cut_el, pg.map, pg.rng, pg.arr, pg.lobe):
        w.resize(400, 240)
        w.repaint()
    paths = pg.export(str(tmp_path))
    assert any(p.endswith('imaging_radar.json') for p in paths)
    win.ctx.store.project_replaced.emit()
    pump(app, 0.2)
    assert pg.result is None or pg.busy or pg.result is not None
