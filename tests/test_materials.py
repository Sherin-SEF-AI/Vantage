"""Materials, radar through the fascia, cameras behind glass and the wipers: the layer physics against closed forms
and a published measurement, the geometry against the mesh's own ray caster, injected faults found and fixed, the
settings, the Part fields, the Issues, the export and the page (offscreen)."""
import cmath
import copy
import json
import math

import numpy as np
import pytest

from vantage.core import exposure as E, materials as M, rules, samples, wipers as W
from vantage.core.analysis import Engine
from vantage.core.mesh import Mesh, Part, ROLE_GLASS
from vantage.core.project import Camera, Project

F = 76.5


@pytest.fixture(scope='module')
def base():
    p = samples.tigor_sample_project()
    p.active_uid = next(v for v in p.variants if v.sensors).uid
    return p


@pytest.fixture()
def p(base):
    return copy.deepcopy(base)


def windscreen_camera(**kw):
    d = dict(tag='FW', name='Windscreen', x=1.80, y=0.0, z=1.36, yaw=0.0, pitch=5.0, slot=6, port=7,
             module_id='sturdecam21', lens_id='sturdecam21_std')
    d.update(kw)
    return Camera(**d)


# =========================================================================== layer physics
def _airy(eps, tan, d, theta_deg, te=True):
    """|r|², |t|² of one slab in air from the interface Fresnel coefficients and the sum of its internal bounces."""
    N = cmath.sqrt(eps * (1 - 1j * tan))
    th = math.radians(theta_deg)
    ci = math.cos(th)
    ct = cmath.sqrt(1 - (math.sin(th) / N) ** 2)
    if te:
        r01 = (ci - N * ct) / (ci + N * ct)
        t01, t10 = 2 * ci / (ci + N * ct), 2 * N * ct / (N * ct + ci)
    else:
        r01 = (N * ci - ct) / (N * ci + ct)
        t01, t10 = 2 * ci / (N * ci + ct), 2 * N * ct / (ct + N * ci)
    r10 = -r01
    beta = 2 * math.pi / M.wavelength_m(F) * N * d * ct
    ph = cmath.exp(-1j * beta)
    t = sum(t01 * t10 * r10 ** (2 * k) * ph ** (2 * k + 1) for k in range(400))
    r = r01 + sum(t01 * t10 * r10 ** (2 * k + 1) * ph ** (2 * k + 2) for k in range(400))
    return abs(r) ** 2, abs(t) ** 2


@pytest.mark.parametrize('eps, tan, d_mm, theta', [(2.75, 0.005, 3.0, 0.0), (2.75, 0.005, 3.0, 35.0),
                                                     (6.31, 0.045, 2.1, 60.0), (3.2, 0.02, 1.7, 75.0)])
def test_one_slab_matches_the_sum_of_its_bounces(eps, tan, d_mm, theta):
    r_te, t_te, r_tm, t_tm = M.stack_rt([(eps, tan, d_mm / 1000)], F, np.array([theta]))
    for te, r, t in ((True, r_te, t_te), (False, r_tm, t_tm)):
        R, T = _airy(eps, tan, d_mm / 1000, theta, te)
        assert abs(r[0]) ** 2 == pytest.approx(R, abs=1e-9)
        assert abs(t[0]) ** 2 == pytest.approx(T, abs=1e-9)


def test_half_wave_quarter_wave_brewster_and_energy():
    lam = M.wavelength_m(F)
    n = math.sqrt(2.75)
    T, R, _, _ = M.transmission([(2.75, 0.0, lam / (2 * n))], F, np.array([0.0]))
    assert T[0] == pytest.approx(1.0, abs=1e-12) and R[0] == pytest.approx(0.0, abs=1e-12)    # half wave: invisible
    T, R, _, _ = M.transmission([(2.75, 0.0, lam / (4 * n))], F, np.array([0.0]))
    assert R[0] == pytest.approx(((n * n - 1) / (n * n + 1)) ** 2, rel=1e-9)                # quarter wave
    brewster = math.degrees(math.atan(n))
    for d in (0.7e-3, 2.3e-3, 5.1e-3):                                                   # TM at Brewster: no reflection
        _, t_te, r_tm, t_tm = M.stack_rt([(2.75, 0.0, d)], F, np.array([brewster]))
        assert abs(r_tm[0]) == pytest.approx(0.0, abs=1e-9) and abs(t_tm[0]) == pytest.approx(1.0, abs=1e-9)
    stack = [(2.75, 0.0, 3e-3), (2.8, 0.0, 5e-6), (43.0, 0.0, 15e-6), (3.5, 0.0, 40e-6)]
    th = np.array([0.0, 20.0, 45.0, 70.0, 85.0])
    r_te, t_te, r_tm, t_tm = M.stack_rt(stack, F, th)
    assert np.allclose(np.abs(r_te) ** 2 + np.abs(t_te) ** 2, 1.0, atol=1e-10)           # lossless: R + T = 1
    assert np.allclose(np.abs(r_tm) ** 2 + np.abs(t_tm) ** 2, 1.0, atol=1e-10)
    lossy = [(e, 0.02, d) for e, _, d in stack]
    T, R, _, _ = M.transmission(lossy, F, th)
    single = np.exp(-2 * math.pi * math.sqrt(2.75) * 0.02 * 3e-3 / M.wavelength_m(F))  # one pass through the substrate
    assert np.all(T + R < 1.0) and np.all(1 - T - R > 0.3 * (1 - single))              # absorbs, about one pass's worth
    assert np.all(1 - T - R < 2.5 * (1 - single))


@pytest.mark.parametrize('eps_sub', [2.9, 3.0, 3.1])
def test_metallic_basecoat_plate_reproduces_emilssons_measurement(eps_sub):
    """Emilsson (Chalmers 2017), sample 11: a 3.416 mm talc-filled PP plate with a factory metallic basecoat (fitted
    48.94 − j0.005, 15 µm) under a 40 µm clear coat measured |S21| −2.34 / −1.98 dB and |S11| −4.86 / −5.75 dB at
    77 GHz; his substrates were 2.2–3.25. The model with the library's primer and clear coat lands within the plates'
    spread."""
    stack = [(eps_sub, 0.01, 3.416e-3), (2.8, 0.011, 5e-6), (48.94, 0.005 / 48.94, 15e-6), (3.5, 0.02, 40e-6)]
    r_te, t_te, _, _ = M.stack_rt(stack, 77.0, np.array([0.0]))
    s21, s11 = 20 * math.log10(abs(t_te[0])), 20 * math.log10(abs(r_te[0]))
    assert -2.7 < s21 < -1.6 and -6.6 < s11 < -3.9
    bare = M.stack_rt(stack[:1], 77.0, np.array([0.0]))[1]
    assert 20 * math.log10(abs(bare[0])) > -1.0                                        # the paint does it


def test_glass_conductivity_law_and_range_factor():
    sigma = 0.0036 * F ** 1.3394                                                        # ITU-R P.2040-3, glass
    assert M.tan_delta('glass', F) == pytest.approx(sigma / (2 * math.pi * F * 1e9 * 8.8541878128e-12 * 6.31))
    assert 0.03 < M.tan_delta('glass', F) < 0.06
    for loss_db in (0.0, 3.0, 6.0, 10.0):                                               # range ∝ P^(1/4)
        assert M.range_factor(10 ** (-loss_db / 10)) == pytest.approx(10 ** (-loss_db / 40))


def test_cavity_is_periodic_in_the_gap_and_the_finishes_are_ordered():
    lam = M.wavelength_m(F)
    fas = M.CLASSES['fascia']
    for th in (0.0, 30.0):
        _, _, r_te, r_tm = M.transmission(M.layers_of(fas, 'solid', F), F, np.array([th]))
        g = np.linspace(0.005, 0.02, 40)
        a = M.cavity_buildup(r_te[0], r_tm[0], g, th, F)
        b = M.cavity_buildup(r_te[0], r_tm[0], g + lam / (2 * math.cos(math.radians(th))), th, F)
        assert np.allclose(a, b, rtol=1e-9)
        R = 0.5 * (abs(r_te[0]) ** 2 + abs(r_tm[0]) ** 2)
        assert a.max() <= 1 / (1 - math.sqrt(R)) ** 2 * 1.05 and a.min() >= 1 / (1 + math.sqrt(R)) ** 2 * 0.95
    loss = {k: float(-2 * M.db(M.transmission(M.layers_of(fas, k, F, 3.6), F, np.array([0.0]))[0][0]))
            for k in M.FINISHES}
    assert M.class_figures(fas, 'metallic', F, 3.6)['two_way_loss_db'] == pytest.approx(loss['metallic'])
    assert M.class_figures(M.CLASSES['steel'], 'solid', F)['two_way_loss_db'] == float('inf')
    assert loss['none'] < loss['solid'] < loss['metallic'] < loss['repaired']
    assert loss['metallic'] - loss['none'] > 1.0                                        # metallic paint costs dB


def test_glass_slab_shift_transmission_and_energy():
    th = np.array([0.0, 30.0, 60.0, 75.0])
    o = M.glass_optics(M.CLASSES['side_glass'], th)
    d, n = 3.5e-3, 1.52
    t = np.radians(th)
    expect = d * np.sin(t) * (1 - np.cos(t) / np.sqrt(n * n - np.sin(t) ** 2))          # the textbook plate shift
    assert np.allclose(o['shift_m'], expect, rtol=1e-9, atol=1e-15)
    assert o['T'][0] == pytest.approx(M.CLASSES['side_glass'].t_vis, abs=1e-9)         # calibrated at normal incidence
    assert np.all(np.diff(o['T']) < 0) and np.all(np.diff(o['R']) > 0)        # less light, more glare when raked
    R0 = ((1.5 - 1) / 2.5) ** 2
    clear = M.MaterialClass('t', 't', 'glass', optical=((1.5, 3.0),), t_vis=(1 - R0) / (1 + R0))
    o = M.glass_optics(clear, np.linspace(0, 85, 18))
    assert np.allclose(o['T'] + o['R'], 1.0, atol=1e-9)                                # no absorption: R + T = 1
    lam = M.glass_optics(M.CLASSES['windscreen'], np.array([60.0]))
    exp = sum(dm / 1000 * (math.tan(math.radians(60)) - (math.sin(math.radians(60)) / nn)
                           / math.sqrt(1 - (math.sin(math.radians(60)) / nn) ** 2))
              for nn, dm in M.CLASSES['windscreen'].optical)
    assert lam['slide_m'][0] == pytest.approx(exp)


# =========================================================================== settings and the Part fields
def test_settings_are_tolerant_and_clamped():
    s = M.MaterialSettings.from_dict({'paint': 'gold', 'radar_ghz': 'x', 'loss_warn_db': 1e9, 'loss_serious_db': 1.0,
                                      'refl_warn_db': float('nan'), 'auto_regions': 'no', 'glare_shield': [1, 'a', 'a'],
                                      'wipers': 'junk'})
    assert s.paint == 'solid' and s.radar_ghz == 76.5 and s.loss_warn_db == 30.0
    assert s.loss_serious_db == 30.0 and s.refl_warn_db == -10.0 and s.auto_regions is False
    assert s.glare_shield == ['a'] and s.wipers == {}
    assert M.MaterialSettings.from_dict(None) == M.MaterialSettings()
    assert M.MaterialSettings.from_dict(M.MaterialSettings().to_dict()) == M.MaterialSettings()
    w = W.WiperSettings.from_dict({'layout': 'custom', 'arms': [{'arm_m': 99, 'blade_m': 'q', 'park_deg': 1000}, 7]})
    assert w.layout == 'custom' and len(w.arms) == 1 and w.arms[0]['arm_m'] == 2.0 and w.arms[0]['park_deg'] == 360.0
    assert W.WiperSettings.from_dict({'layout': 'custom'}).layout == 'tandem'           # custom without arms


def test_part_material_round_trips_and_old_files_stay_as_they_were(tmp_path):
    a = Part('Bumper', 'paint', 1.0, (1, 1, 1))
    assert 'surface' not in a.to_dict() and 'surface_mm' not in a.to_dict()             # unset: the file is unchanged
    b = Part.from_dict(dict(a.to_dict(), surface='fascia', surface_mm=3.4))
    assert (b.surface, b.surface_mm) == ('fascia', 3.4) and Part.from_dict(b.to_dict()) == b
    junk = Part.from_dict({'name': 'x', 'surface': 5, 'surface_mm': 'thick'})
    assert junk.surface == '' and junk.surface_mm == 0.0
    assert Part.from_dict({'name': 'x', 'surface_mm': -3}).surface_mm == 0.0
    p = samples.tigor_sample_project()
    p.vehicle.mesh.parts[0].surface, p.vehicle.mesh.parts[0].surface_mm = 'steel', 1.2
    assert p.vehicle.mesh.copy().parts[0].surface == 'steel'
    path = str(tmp_path / 'm.vantage')
    p.save(path)
    q = Project.load(path)
    assert (q.vehicle.mesh.parts[0].surface, q.vehicle.mesh.parts[0].surface_mm) == ('steel', 1.2)
    assert q.vehicle.mesh.parts[1].surface == ''
    assert q.analysis.materials == {} and M.settings_of(q) == M.MaterialSettings()


# =========================================================================== which material where
def test_classes_from_role_name_and_position(p):
    mesh = p.vehicle.mesh
    fm = M.face_materials(mesh, mesh.parts, M.MaterialSettings())
    by = {pi['name']: pi for pi in fm.parts}
    assert by['Wheels']['class'] == 'tyre' and by['Glass']['class'] == 'side_glass'
    assert by['Mirrors']['class'] == 'pc_abs' and by['Body']['why'] == 'body'
    assert set(by['Body']['classes']) == {'steel', 'fascia', 'windscreen', 'side_glass'}
    c = mesh.centroids
    fas = fm.cls == M.CLASS_IDS.index('fascia')
    lo, hi = mesh.bbox
    assert fas.any() and np.all(np.minimum(c[fas, 0] - lo[0], hi[0] - c[fas, 0]) <= 0.36)
    ws = fm.cls == M.CLASS_IDS.index('windscreen')
    assert np.all(fm.normals[ws, 0] > 0.39) and np.all(c[ws, 2] > 0.95)                 # facing forward, above the belt
    off = M.face_materials(mesh, mesh.parts, M.MaterialSettings(auto_regions=False))
    assert set(np.unique(off.cls[mesh.face_part == 0])) == {M.CLASS_IDS.index('steel')}
    named = [Part('Front bumper'), Part('Chrome grille'), Part('Windshield', role=ROLE_GLASS), Part('Rear lamp'),
             Part('Rack', 'rig'), Part('Body', surface='cfrp'), Part('Hood', surface='nonsense')]
    assert [M._part_rule(q)[0] for q in named] == ['fascia', 'chrome', 'windscreen', 'lamp', 'aluminium', 'cfrp', None]


# =========================================================================== radars on the sample
def _report(p, tag, settings=None):
    res = M.run(p, Engine(), settings)
    return res, res.radar(tag)


def test_sample_radars_against_the_mesh_ray_caster(p):
    res = M.run(p, Engine())
    assert res.metrics['radars'] == 5 and len(res.radars) == 5
    rd1 = res.radar('RD1')
    assert rd1['mount'] == 'exposed' and rd1['zones'][0]['factor'] == 1.0 and not rd1['inside']
    mesh = p.vehicle.mesh
    for tag in ('RD2', 'RD3', 'RD4', 'RD5'):
        r = res.radar(tag)
        s = p.variant.by_tag(tag) or next(x for x in p.variant.sensors if x.tag == tag)
        from vantage.core import sensors as S
        rot = S.rotation(s)
        t, f = mesh.raycast(s.pos()[None], rot[:, 0][None])                             # the mesh's own caster
        assert r['mount'] == 'behind' and r['class'] == 'fascia' and r['paint'] == 'solid'
        assert r['gap_mm'] == pytest.approx(t[0] * 1000, abs=1e-6)
        inc = math.degrees(math.acos(abs(float(rot[:, 0] @ mesh.normals[f[0]]))))
        assert r['incidence_deg'] == pytest.approx(inc, abs=1e-6)
        T, R, _, _ = M.transmission(M.layers_of(M.CLASSES['fascia'], 'solid', F), F, np.array([inc]))
        assert r['boresight_loss_db'] == pytest.approx(float(-2 * M.db(T[0])), abs=1e-9)
        assert r['reflection_db'] == pytest.approx(float(M.db(R[0])), abs=1e-9)
        z = r['zones'][0]
        assert z['factor'] == pytest.approx(10 ** (-z['loss_db'] / 40), rel=1e-9)
        assert z['derated_m'] == pytest.approx(z['range_m'] * z['factor'])
        assert M.radar_range_factors(p, s) == pytest.approx([z['factor']])
    assert M.radar_range_factors(p, next(x for x in p.variant.sensors if x.tag == 'RD1')) == [1.0, 1.0]
    assert M.issues(p, Engine().vehicle_geom(p)) == []          # the sample's corner radars pass at 2.7 dB


def test_paint_and_repairs_cost_range_and_reach_the_issues(p):
    vg = Engine().vehicle_geom(p)
    base = M.run(p, Engine()).radar('RD2')['zones'][0]
    p.analysis.materials = {'paint': 'metallic'}
    r = M.run(p, Engine()).radar('RD2')
    assert r['zones'][0]['loss_db'] > base['loss_db'] + 1.0 and r['zones'][0]['derated_m'] < base['derated_m']
    p.analysis.materials = {'paint': 'repaired'}
    worse = M.run(p, Engine()).radar('RD2')['zones'][0]['loss_db']
    assert worse > r['zones'][0]['loss_db']
    st = M.settings_of(p)
    iss = [i for i in rules.collect_issues(p, None, vg) if 'two-way loss' in i.message]
    assert len(iss) == 4 and all(i.severity == ('serious' if worse >= st.loss_serious_db else 'warning') for i in iss)
    assert 'no effect paint over the radar' in iss[0].message and iss[0].uids
    p.analysis.materials = {}
    assert not [i for i in rules.collect_issues(p, None, vg) if 'two-way loss' in i.message]


def test_a_radar_behind_steel_is_blocked_and_recovered(p):
    vg = Engine().vehicle_geom(p)
    p.vehicle.mesh.parts[0].surface = 'steel'                  # the whole body steel: no plastic bumper
    res = M.run(p, Engine())
    r = res.radar('RD2')
    assert r['mount'] == 'blocked' and r['blocked_share'] > 0.9 and r['blocked_by'].get('steel', 0) > 0.9
    assert r['risk'] == 'high' and 'metal in the beam' in r['reasons'][0]
    iss = [i for i in M.issues(p, vg) if i.objects == 'RD2']
    assert any(i.severity == 'critical' and 'does not pass' in i.message for i in iss)
    p.vehicle.mesh.parts[0].surface = ''
    assert M.run(p, Engine()).radar('RD2')['mount'] == 'behind'


def test_the_suggested_fascia_thickness_cuts_reflection_and_loss(p):
    r = M.run(p, Engine()).radar('RD2')
    bt = r['best_thickness']
    assert bt is not None and bt['reflection_db'] < -20 and bt['loss_db'] < r['boresight_loss_db']
    p.vehicle.mesh.parts[0].surface_mm = round(bt['thickness_mm'], 3)
    r2 = M.run(p, Engine()).radar('RD2')
    assert r2['thickness_mm'] == pytest.approx(bt['thickness_mm'], abs=1e-3)
    assert r2['reflection_db'] < -20 and r2['boresight_loss_db'] == pytest.approx(bt['loss_db'], abs=0.02)
    assert r2['risk'] in ('low', 'moderate') and not any('reflects' in x for x in r2['reasons'])


def test_a_resonant_gap_raises_the_multipath_risk(p):
    s = next(x for x in p.variant.sensors if x.tag == 'RD2')
    from vantage.core import sensors as S
    fwd = S.rotation(s)[:, 0]
    r = M.run(p, Engine()).radar('RD2')
    cav = r['cavity']
    lam = cav['wavelength_mm']
    # walk the radar back along its boresight to the worst spacing within half a wavelength
    worst = None
    for step in np.linspace(0, lam / 2, 21):
        q = copy.deepcopy(p)
        sq = next(x for x in q.variant.sensors if x.tag == 'RD2')
        sq.x, sq.y, sq.z = (s.pos() - fwd * step / 1000).tolist()
        rr = M.run(q, Engine()).radar('RD2')
        if worst is None or rr['cavity']['resonance'] > worst[1]['cavity']['resonance']:
            worst = (step, rr, q)
    rr = worst[1]
    assert rr['cavity']['resonance'] > 0.9 and rr['risk'] == 'high'
    assert any('near a resonance' in x for x in rr['reasons'])
    assert rr['cavity']['buildup_db'] == pytest.approx(rr['cavity']['max_db'], abs=0.3)
    assert abs(rr['cavity']['move_mm']) < lam / 2
    wq = worst[2]
    iss = [i for i in M.issues(wq, Engine().vehicle_geom(wq)) if 'multipath' in i.message]
    assert len(iss) == 1 and iss[0].severity == 'warning' and 'between resonances' in iss[0].message
    # the suggested move puts it between resonances: lowest build-up, the risk back to the reflection alone
    q = copy.deepcopy(p)
    sq = next(x for x in q.variant.sensors if x.tag == 'RD2')
    sq.x, sq.y, sq.z = (s.pos() - fwd * (worst[0] + rr['cavity']['move_mm']) / 1000).tolist()
    fixed = M.run(q, Engine()).radar('RD2')
    assert fixed['cavity']['resonance'] < 0.05 and fixed['risk'] != 'high'
    assert fixed['cavity']['buildup_db'] == pytest.approx(fixed['cavity']['min_db'], abs=0.1)


# =========================================================================== cameras behind glass
def _plate_mesh(incline_deg=30.0, width=4.0, length=4.0, x0=0.10, name='Windscreen'):
    """One flat glass plate rising towards the back at incline_deg, crossing the x axis at x0 (the camera looks at it
    from the origin)."""
    a = math.radians(incline_deg)
    up = np.array([-math.cos(a), 0.0, math.sin(a)])         # along the plate, upwards and backwards
    c = np.array([x0, 0.0, 0.0])
    V = np.array([c - up * length / 2 - [0, width / 2, 0], c - up * length / 2 + [0, width / 2, 0],
                  c + up * length / 2 + [0, width / 2, 0], c + up * length / 2 - [0, width / 2, 0]])
    F_ = np.array([[0, 1, 2], [0, 2, 3]])
    m = Mesh(V, F_, None, [Part(name, 'glass', 0.3, (0.2, 0.2, 0.2), ROLE_GLASS)])
    n = m.normals[0]
    if n[0] < 0:
        m = Mesh(V, F_[:, ::-1], None, [Part(name, 'glass', 0.3, (0.2, 0.2, 0.2), ROLE_GLASS)])
    return m


def _plate_project(lens=None):
    p = Project()
    p.vehicle.mesh = _plate_mesh()
    v = p.variant
    if lens is not None:
        p.lenses[lens.id] = lens
    v.cameras = [Camera(tag='C', x=0.0, y=0.0, z=0.0, yaw=0.0, pitch=0.0, module_id='sturdecam21',
                        lens_id=lens.id if lens is not None else 'sturdecam21_std')]
    return p


def test_a_narrow_camera_behind_a_tilted_plate_shifts_by_the_plate_formula():
    from vantage.core.lens import Lens
    tele = Lens('tele_test', 'Tele', module_id='sturdecam21', focal_mm=25.0, projection='rectilinear',
                max_half_deg=10.0)
    p = _plate_project(tele)
    res = M.run(p, None, M.MaterialSettings(wipers={'enabled': False}))
    assert len(res.cameras) == 1
    c = res.cameras[0]
    assert c['class'] == 'windscreen' and c['incidence_deg'] == pytest.approx(60.0, abs=1e-6)
    assert c['gap_mm'] == pytest.approx(100.0, abs=1e-6) and c['glass_share'] == 1.0
    o = M.glass_optics(M.CLASSES['windscreen'], np.array([60.0]))
    assert c['transmission'] == pytest.approx(o['T'][0], abs=1e-9)
    # the plate turns no ray, it slides them up its slope: across the axis the centre moves up by slide × cos 60°
    # (along the axis the fit takes up the shift's change over the field, which no single sideways move can)
    slide = o['slide_m'][0]
    assert c['apparent_shift_mm'][1] == pytest.approx(0.0, abs=1e-6)
    assert c['apparent_shift_mm'][2] == pytest.approx(slide * 0.5 * 1000, rel=0.03)
    from vantage.core.lens import resolve
    f_px = resolve(p.module_for(p.variant.cameras[0]), p.lens_for(p.variant.cameras[0])).f_px
    assert c['errors']['1']['naive_rms_px'] == pytest.approx(f_px * slide * 0.5 / 1.0, rel=0.08)   # parallax at 1 m
    assert c['errors']['1']['naive_rms_px'] == pytest.approx(3 * c['errors']['3']['naive_rms_px'], rel=0.05)
    for d in ('1', '3', '10'):
        e = c['errors'][d]
        assert e['fitted_rms_px'] < 0.1 * e['naive_rms_px']                             # one shift explains it
    assert c['dash_share'] == 1.0 and c['dash_reflectance'] == pytest.approx(o['R'][0], rel=0.02)
    assert c['intrinsics_equivalent'] is not None


def test_windscreen_camera_on_the_sample(p):
    vg = Engine().vehicle_geom(p)
    assert not any(c['tag'] == 'FW' for c in M.run(p, Engine()).cameras)
    p.variant.cameras.append(windscreen_camera())
    res = M.run(p, Engine())
    c = res.camera('FW')
    assert c is not None and c['class'] == 'windscreen'
    mesh = p.vehicle.mesh
    cam = p.variant.by_tag('FW')
    from vantage.core.mathutil import rot_zyx
    fwd = rot_zyx(cam.yaw, cam.pitch, cam.roll)[:, 0]
    t, f = mesh.raycast(cam.pos()[None], fwd[None])
    assert c['gap_mm'] == pytest.approx(t[0] * 1000, abs=1e-6)
    inc = math.degrees(math.acos(abs(float(fwd @ mesh.normals[f[0]]))))
    assert abs(c['incidence_deg'] - inc) < 8.0                                          # the image centre's sample
    assert 0.6 < c['transmission'] < 0.88 and c['stops_lost'] > 0
    assert 1.0 < c['apparent_shift_norm_mm'] < 8.0
    # the glass reflects the dashboard into it: a glare warning, gone with a shield
    assert any(i.severity == 'warning' and 'glare shield' in i.message for i in M.issues(p, vg))
    p.analysis.materials = {'glare_shield': [cam.uid]}
    assert not any('glare shield' in i.message for i in M.issues(p, vg))
    assert M.run(p, Engine()).camera('FW')['shield']


# =========================================================================== wipers
def test_arm_sector_membership_and_the_wiped_area():
    rng = np.random.default_rng(3)
    a = W.Arm('A', 0.1, -0.05, 0.5, 0.6, 180.0, 100.0)
    Q = rng.uniform(-1, 1, (4000, 2))
    r = np.hypot(Q[:, 0] - 0.1, Q[:, 1] + 0.05)
    ang = np.degrees(np.arctan2(Q[:, 1] + 0.05, Q[:, 0] - 0.1)) % 360
    expect = (r >= 0.2) & (r <= 0.8) & (ang >= 80) & (ang <= 180)
    assert np.array_equal(a.contains(Q[:, 0], Q[:, 1]), expect)
    assert W.Arm('B', park_deg=0.0, sweep_deg=90.0).interval == (0.0, 90.0)
    # a half disc of radius 1 on a 2 x 1 m glass: π/4 of it wiped
    nw, nh = 200, 100
    ws = W.Windscreen(np.zeros(0, int), 'test', np.zeros(3), np.array([0, 1.0, 0]), np.array([0, 0, 1.0]),
                      np.array([1.0, 0, 0]), 2.0, 1.0, 2.0, 90.0, (-1.0, 0.0, 0.01, nw, nh),
                      np.ones((nh, nw), bool), [])
    m = W.wiped_mask(ws, [W.Arm('H', 0.0, 0.0, 0.5, 1.0, 0.0, 180.0)])
    assert m.sum() / m.size == pytest.approx(math.pi / 4, rel=0.01)


def test_windscreen_found_on_a_flat_plate():
    mesh = _plate_mesh(incline_deg=30.0, width=1.4, length=0.8)
    fm = M.face_materials(mesh, mesh.parts, M.MaterialSettings())
    ws = W.windscreen(mesh, fm)
    assert ws is not None and ws.source == 'glass part'
    assert ws.incline_deg == pytest.approx(30.0, abs=1e-6)
    assert ws.width_m == pytest.approx(1.4, abs=0.02) and ws.height_m == pytest.approx(0.8, abs=0.02)
    assert ws.area_m2 == pytest.approx(1.12)
    bottom = mesh.vertices[[0, 1]].mean(axis=0)
    assert np.allclose(ws.origin, bottom, atol=0.012)
    assert np.allclose(ws.to_plane(ws.to_world([[0.3, 0.2]])), [[0.3, 0.2]])
    arms = W.default_arms(ws, 'tandem', 'lhd')
    assert arms[0].pivot_w > 0 > arms[1].pivot_w and arms[0].park_deg == 180.0
    rh = W.default_arms(ws, 'tandem', 'rhd')
    assert rh[0].pivot_w == pytest.approx(-arms[0].pivot_w) and rh[0].park_deg == 0.0
    top_middle = [[0.0, 0.92 * ws.height_m]]
    assert W.in_sweep(arms, top_middle)[0] and W.in_sweep(rh, top_middle)[0]          # where cameras sit


def test_wipers_on_the_sample_find_an_unwiped_camera_and_recover(p):
    vg = Engine().vehicle_geom(p)
    p.variant.cameras.append(windscreen_camera())
    res = M.run(p, Engine())
    wr = res.wipers
    assert wr.windscreen is not None and wr.windscreen.source.startswith('body region')
    assert 20 < wr.windscreen.incline_deg < 40 and abs(wr.windscreen.origin[1]) < 0.05
    fw = next(c for c in wr.cameras if c['tag'] == 'FW')
    assert fw['share_wiped'] == 1.0 and fw['dirt_unwiped'] is None
    # short blades leave the camera's view on unwiped glass
    arms = [dict(a.to_dict(), arm_m=0.25, blade_m=0.2) for a in wr.arms]
    p.analysis.materials = {'wipers': {'layout': 'custom', 'arms': arms}}
    res2 = M.run(p, Engine())
    fw2 = next(c for c in res2.wipers.cameras if c['tag'] == 'FW')
    assert fw2['share_wiped'] < 0.5 and res2.wipers.wiped_share < wr.wiped_share
    iss = [i for i in M.issues(p, vg) if 'wipers do not clear' in i.message]
    assert len(iss) == 1 and iss[0].severity == 'serious' and iss[0].uids == [p.variant.by_tag('FW').uid]
    assert W.issues(p, vg) == iss
    # the dirt on the unwiped glass is the exposure model's score there
    cam = p.variant.by_tag('FW')
    mesh, _ = M._geometry(p)
    Q, P, N = W.footprint(p, cam, mesh, res2.wipers.windscreen)
    rest = ~W.in_sweep(res2.wipers.arms, Q)
    wheels, _ = E.wheel_centres(p, p.vehicle.mesh)
    sc = E.dirt_score(E.dirt_factors(P[rest], [N[rest]], N[rest], wheels, E._body_info(p.vehicle.mesh)))
    assert fw2['dirt_unwiped']['score'] == pytest.approx(float(sc.mean()))
    p.analysis.materials = {}
    assert not [i for i in M.issues(p, vg) if 'wipers do not clear' in i.message]
    p.analysis.materials = {'wipers': {'enabled': False}}
    assert M.run(p, Engine()).wipers is None


# =========================================================================== the study's edges
def test_cancel_empty_hidden_and_broken_inputs(p):
    assert M.run(p, Engine(), cancel=lambda: True) is None
    q = copy.deepcopy(p)
    for s in q.variant.sensors:
        if s.tag == 'RD2':
            s.visible = False
        if s.tag == 'RD3':
            s.yaw = float('nan')
    res = M.run(q, Engine())
    assert {r['tag'] for r in res.radars} == {'RD1', 'RD4', 'RD5'}
    far = copy.deepcopy(next(x for x in q.variant.sensors if x.tag == 'RD4'))
    far.uid, far.tag, far.x, far.z = 'faraway', 'RD9', 9.0, 0.5            # nothing of the vehicle near it
    q.variant.sensors.append(far)
    r9 = M.run(q, Engine()).radar('RD9')
    assert r9['mount'] == 'exposed' and r9['zones'][0]['loss_db'] == 0.0 and r9['blocked_share'] == 0.0
    q.variant.sensors = []
    q.variant.cameras = []
    res = M.run(q, Engine())
    assert res.radars == [] and res.cameras == [] and M.issues(q) == []
    empty = Project()
    res = M.run(empty)
    assert res.parts == [] and res.radars == [] and res.wipers is None and M.issues(empty) == []
    other = next(v for v in p.variants if not v.sensors)
    res = M.run(p, Engine(), variant=other)
    assert res.variant_uid == other.uid and res.radars == []


def test_the_issue_cache_and_the_default_layout(base):
    q = copy.deepcopy(base)
    q.active_uid = q.variants[0].uid                  # the 6-camera layout: no radar, no camera behind glass
    vg = Engine().vehicle_geom(q)
    assert M.issues(q, vg) == []
    import time
    p = copy.deepcopy(base)
    vg = Engine().vehicle_geom(p)
    M.issues(p, vg)
    t = time.perf_counter()
    for _ in range(20):
        M.issues(p, vg)
    assert (time.perf_counter() - t) / 20 < 0.01                                        # cached on the inputs


def test_export(p, tmp_path):
    import cv2
    p.variant.cameras.append(windscreen_camera())
    res = M.run(p, Engine())
    paths = M.export(res, p, str(tmp_path))
    assert sorted(x.rsplit('/', 1)[1] for x in paths) == ['materials.json', 'materials.png', 'materials_glass.csv',
                                                         'materials_radars.csv']
    doc = json.loads((tmp_path / 'materials.json').read_text())
    assert doc['schema'] == M.SCHEMA and doc['frame'].startswith('vehicle') and doc['assumptions']
    assert len(doc['radars']) == 5 and doc['radars'][1]['looks_through'] == 'Bumper fascia (PP, painted)'
    assert doc['cameras_behind_glass'][0]['tag'] == 'FW' and doc['wipers']['cameras'][0]['tag'] == 'FW'
    assert {c['id'] for c in doc['library']['classes']} == set(M.CLASSES) and doc['library']['sources']['glass']
    rows = (tmp_path / 'materials_radars.csv').read_text().strip().splitlines()
    assert len(rows) == 1 + 2 + 4                                                       # RD1 has two zones
    g = (tmp_path / 'materials_glass.csv').read_text().strip().splitlines()
    assert len(g) == 2 and g[1].startswith('FW,')
    img = cv2.imdecode(np.frombuffer((tmp_path / 'materials.png').read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    assert img is not None and img.shape[1] == 1400
    s = M.summary(res)
    json.dumps(s)
    assert s['radars'][1]['two_way_loss_db'] == pytest.approx(res.radars[1]['zones'][0]['loss_db'], abs=0.01)
    assert s['wipers']['cameras'][0]['wiped_share'] == 1.0


# =========================================================================== the page
from test_gui import app, win, pump, settle, _wait     # noqa: E402,F401  (fixtures)


def _open(win, name, cls):
    if name not in win._page_widgets:
        pg = cls(win.ctx)
        win.pages.addWidget(pg)
        win._page_widgets[name] = pg
    win.set_page(name)
    return win._page_widgets[name]


def _ready(pg):
    c = pg.ctl
    return c.result is not None and not c.busy and not c.stale and not c.timer.isActive() and pg._shown is c.result


def test_page_edits_undoes_and_follows_the_project(app, win, tmp_path):
    from PyQt6.QtWidgets import QComboBox
    from vantage.ui.workspaces.materials_ws import MaterialsPage, scene_lines
    store = win.store
    rig = next(v for v in store.project.variants if v.sensors)

    def fn(p_):
        p_.active_uid = rig.uid
    store.edit('variants', fn, 'Data rig')
    pg = _open(win, 'materials', MaterialsPage)
    pump(app, 0.2)
    assert pg.isVisible()
    _wait(app, lambda: _ready(pg))
    res = pg.ctl.result
    assert pg.radars.rowCount() == 5 and pg.radars.item(1, 0).text() == 'RD2'
    assert pg.radars.item(1, 5).text() == f'{res.radar("RD2")["zones"][0]["loss_db"]:.2f} dB'
    assert pg.beam.rep is not None and pg.parts.rowCount() == len(store.project.vehicle.mesh.parts)
    assert pg.cams.rowCount() == 0 and len(pg.findings) == len(res.findings)
    loss = res.radar('RD2')['zones'][0]['loss_db']
    # metallic paint, a click: saved, computed again, undoable
    pg.paint.buttons['metallic'].click()
    pump(app, 0.1)
    assert store.project.analysis.materials['paint'] == 'metallic'
    _wait(app, lambda: _ready(pg) and pg.ctl.result.settings.paint == 'metallic')
    assert pg.ctl.result.radar('RD2')['zones'][0]['loss_db'] > loss + 1.0
    store.undo.undo()
    _wait(app, lambda: _ready(pg) and pg.ctl.result.settings.paint == 'solid')
    assert pg.paint.value() == 'solid'
    # the body's material set to steel in the Parts table: the corner radars are blocked; undo brings the fascia back
    pg.tabs.setCurrentIndex(2)
    box = pg.parts.cellWidget(0, 3)
    assert isinstance(box, QComboBox)
    box.setCurrentIndex(box.findData('steel'))
    pump(app, 0.1)
    assert store.project.vehicle.mesh.parts[0].surface == 'steel'
    _wait(app, lambda: _ready(pg) and pg.ctl.result.radar('RD2')['mount'] == 'blocked')
    assert pg.radars.item(1, 5).text() == 'blocked'
    store.undo.undo()
    assert store.project.vehicle.mesh.parts[0].surface == ''
    _wait(app, lambda: _ready(pg) and pg.ctl.result.radar('RD2')['mount'] == 'behind')
    # a thickness typed into the Parts table
    pg.parts.item(0, 4).setText('3.6')
    pump(app, 0.1)
    assert store.project.vehicle.mesh.parts[0].surface_mm == 3.6
    store.undo.undo()
    assert store.project.vehicle.mesh.parts[0].surface_mm == 0.0
    # a windscreen camera: the glass table, the glass view and the wipers
    cam = windscreen_camera()

    def add(p_):
        p_.variant.cameras.append(cam)
    store.edit('variant', add, 'Add FW')
    pg.tabs.setCurrentIndex(1)
    _wait(app, lambda: _ready(pg) and pg.cams.rowCount() == 1)
    assert pg.cams.item(0, 0).text() == 'FW' and pg.cams.item(0, 7).text() == '100 %'
    assert pg.glass.res is not None and pg.glass.res.windscreen is not None and pg.arms.rowCount() == 2
    pg.cams.cellWidget(0, 8).setChecked(True)                                         # glare shield fitted
    pump(app, 0.1)
    assert store.project.analysis.materials['glare_shield'] == [cam.uid]
    # an arm made shorter in the table: custom wipers, the camera no longer wiped
    _wait(app, lambda: _ready(pg))
    pg.arms.item(0, 4).setText('0.2')
    pump(app, 0.1)
    w = store.project.analysis.materials['wipers']
    assert w['layout'] == 'custom' and w['arms'][0]['blade_m'] == 0.2
    _wait(app, lambda: _ready(pg) and pg.layout_seg.value() == 'custom')
    store.undo.undo()
    _wait(app, lambda: _ready(pg) and pg.layout_seg.value() == 'tandem')
    # the 3D view's wiper lines from the result
    lines = scene_lines(win.ctx)
    assert len(lines) == 4 and all(len(ln.pts) >= 2 for ln in lines)
    # export
    paths = pg.export(str(tmp_path))
    assert len(paths) == 4 and (tmp_path / 'materials.json').exists()
    # a new project while the page shows
    win.store.set_project(samples.tigor_sample_project())
    _wait(app, lambda: _ready(pg))
    assert pg.ctl.result.radars == [] and pg.radars.rowCount() == 0 and pg.paint.value() == 'solid'
    win.set_page('viewports')
    pump(app, 0.1)
    assert not pg.ctl.active
