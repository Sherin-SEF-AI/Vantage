"""Thermal and vibration drift (core.thermo_vib): closed forms (cantilever frequency, thermal bow, ISO 8608 RMS,
quarter-car and single-mode transmissibility), the energy balance, the chain of deformations, resonance flags, events,
determinism, the exports and the page."""
import copy
import json
import math

import numpy as np
import pytest

from vantage.core import samples, suspension as SU, thermo_vib as TV
from vantage.core.project import RigPart


def _data_rig():
    p = samples.tigor_sample_project()
    p.active_uid = next(v.uid for v in p.variants if v.name.startswith('Data rig'))
    return p


@pytest.fixture(scope='module')
def sample_result():
    return TV.run(_data_rig())


# =========================================================================== closed forms
def test_cantilever_frequency_closed_form():
    E, I, L, mt, mb = 68.9e9, 2.2e-7, 0.5, 3.2, 0.7
    hand = math.sqrt(3 * E * I / ((mt + 0.24 * mb) * L ** 3)) / (2 * math.pi)
    assert TV.cantilever_f1(E, I, L, mt, mb) == pytest.approx(hand, rel=1e-12)
    # Rayleigh's 0.24 against the exact Euler-Bernoulli first mode of a bare cantilever (β1·L = 1.8751): within 1 %
    m = 1.2
    exact = 1.8751 ** 2 / (2 * math.pi) * math.sqrt(E * I / (m / L * L ** 4))
    assert TV.cantilever_f1(E, I, L, 0.0, m) == pytest.approx(exact, rel=0.01)
    # a mast part's mode is that formula with the tube's I and what it carries
    p = _data_rig()
    p.rig = [RigPart(name='Mast', kind='cylinder', x=1.0, y=0.0, z=1.532 + 0.3, height=0.6, radius=0.025)]
    rf = TV.roof_box(p)
    tree = TV.rig_tree(p, rf)
    assert tree[0]['supports'] == ['roof']
    st = TV.ThermoVibSettings()
    md = TV.part_modes(tree, {tree[0]['uid']: (2.0, 0.5)}, st)[0]
    assert md.f1 == pytest.approx(TV.cantilever_f1(68.9e9, TV.tube_I(0.025), 0.6, 2.0, 0.5), rel=1e-12)
    assert TV.tube_I(0.025) == pytest.approx(math.pi / 4 * (0.025 ** 4 - 0.022 ** 4))


def _warp_one(p, part, dT=0.0, depth=0.04, mean_rise=0.0, g=(1.0, 0.0, 0.0), feet='sliding', roof_rise=0.0):
    p.rig = [part]
    rf = TV.roof_box(p)
    tree = TV.rig_tree(p, rf)
    st = TV.ThermoVibSettings(feet=feet)
    cal = st.t_cal_c
    temps = {part.uid: {'mean': cal + mean_rise, 'dT': dT, 'depth': depth, 'g': np.array(g, float),
                        'T': [cal, cal], 'tau_s': 0.0, 'h': 0.0}}
    return TV.Warp(tree, rf, st, cal, cal + roof_rise, temps, cal), tree


def test_thermal_bow_against_a_hand_calculation():
    p = _data_rig()
    top = TV.roof_box(p)['top']
    # an upright mast, its +x side 2 K warmer across 40 mm: κ = 23.6e-6 · 2 / 0.04 = 1.18e-3 /m
    mast = RigPart(name='Mast', kind='cylinder', x=1.0, y=0.0, z=top + 0.25, height=0.5, radius=0.02)
    w, _ = _warp_one(p, mast, dT=2.0, depth=0.04)
    kappa = 23.6e-6 * 2.0 / 0.04
    assert w.info[mast.uid]['kappa'] == pytest.approx(kappa)
    U, W = w.motion(0, np.array([1.0, 0.0, top + 0.5]))            # the mast's top
    assert U[0] == pytest.approx(-kappa * 0.5 ** 2 / 2, rel=1e-9)  # it bends away from the warm side: 0.1475 mm
    assert abs(U[0]) * 1000 == pytest.approx(0.1475, abs=1e-4)
    assert W[1] == pytest.approx(-kappa * 0.5, rel=1e-9)            # and leans back: 5.9e-4 rad
    # 0.1 m above the top, the lever adds slope × 0.1
    U2, _ = w.motion(0, np.array([1.0, 0.0, top + 0.6]))
    assert U2[0] == pytest.approx(-kappa * 0.5 ** 2 / 2 - kappa * 0.5 * 0.1, rel=1e-9)
    # a crossbar between feet 1.2 m apart, its top 3 K warmer across 20 mm: the middle rises κ·s²/8
    bar = RigPart(name='Bar', kind='bar', x=1.0, y=0.0, z=top + 0.07, length=1.32, radius=0.016)
    w, _ = _warp_one(p, bar, dT=3.0, depth=0.02, g=(0, 0, 1))
    kappa = 23.6e-6 * 3.0 / 0.02
    s = 1.32 - 2 * TV.FEET_INSET_M
    U, W = w.motion(0, np.array([1.0, 0.0, top + 0.07 + 0.016]))
    assert U[2] == pytest.approx(kappa * s * s / 8, rel=1e-9)
    assert np.allclose(W, 0.0, atol=1e-15)
    U, W = w.motion(0, np.array([1.0, 0.3, top + 0.07 + 0.016]))  # 0.3 m left: slope −κη about x
    assert U[2] == pytest.approx(kappa / 2 * (s * s / 4 - 0.09), rel=1e-9)
    assert W[0] == pytest.approx(-kappa * 0.3, rel=1e-9)


def test_rack_feet_sliding_and_clamped():
    p = _data_rig()
    top = TV.roof_box(p)['top']
    bar = RigPart(name='Bar', kind='bar', x=1.0, y=0.0, z=top + 0.07, length=1.32, radius=0.016)
    s = 1.32 - 2 * TV.FEET_INSET_M
    al, st_ = TV.MATERIALS['al6061'], TV.MATERIALS['steel']
    # sliding: the bar warms 30 K over the roof, which stays at the calibration: one foot slips (α_al·30)·s
    w, _ = _warp_one(p, bar, mean_rise=30.0)
    assert w.info[bar.uid]['mismatch_mm'] == pytest.approx(al['alpha'] * 30 * s * 1000)
    U, _ = w.motion(0, np.array([1.0, 0.5, top + 0.086]))
    assert U[1] == pytest.approx(al['alpha'] * 30 * 0.5, rel=1e-6)   # free expansion about the middle
    # clamped: the bar's length follows the roof (here not moving), the mismatch is an axial force E·A·α·ΔT
    w, _ = _warp_one(p, bar, mean_rise=10.0, feet='clamped')
    I, A = TV.tube_I(0.016), TV.tube_area(0.016)
    N = al['E'] * A * al['alpha'] * 10
    d = w.info[bar.uid]
    assert d['N'] == pytest.approx(N)
    assert d['stress_mpa'] == pytest.approx(-al['E'] * al['alpha'] * 10 / 1e6)
    assert d['N_cr'] == pytest.approx(math.pi ** 2 * al['E'] * I / s ** 2)
    assert d['amp'] == pytest.approx(1 / (1 - N / d['N_cr']))            # the bow grows under the compression
    # 30 K: σ = EαΔT ≈ 49 MPa against π²E/(L/r)² ≈ 51 MPa for this slender bar: at the buckling load, capped
    w, _ = _warp_one(p, bar, mean_rise=30.0, feet='clamped')
    assert w.info[bar.uid]['N'] / w.info[bar.uid]['N_cr'] > 0.95 and w.info[bar.uid]['amp'] == 20.0
    U, _ = w.motion(0, np.array([1.0, 0.5, top + 0.086]))
    assert abs(U[1]) < 1e-12                                       # held to the (cold) roof's length
    # a warm roof drags the clamped bar's points along with the steel
    w, _ = _warp_one(p, bar, mean_rise=30.0, feet='clamped', roof_rise=30.0)
    assert w.info[bar.uid]['mismatch_mm'] == pytest.approx((al['alpha'] - st_['alpha']) * 30 * s * 1000)


def test_iso8608_rms_against_analytic_integration():
    for cls in 'ACEH':
        gd = TV.iso8608_gd(cls)
        assert gd == pytest.approx(16e-6 * 4 ** 'ABCDEFGH'.index(cls))
        for n1, n2 in ((0.011, 2.83), (0.05, 1.0)):
            analytic = math.sqrt(gd * 0.1 ** 2 * (1 / n1 - 1 / n2))
            assert TV.road_rms(cls, n1, n2) == pytest.approx(analytic, rel=1e-5)
    # in time at a speed: the same variance, and a white velocity PSD (2π·n0)²·Gd(n0)·V
    v = 50 / 3.6
    f = np.geomspace(0.011 * v, 2.83 * v, 20000)
    G = TV.road_psd_temporal(f, 'C', v)
    assert math.sqrt(TV.integrate(f, G)) == pytest.approx(TV.road_rms('C'), rel=1e-5)
    vel = (2 * np.pi * f) ** 2 * G
    assert np.allclose(vel, (2 * math.pi * 0.1) ** 2 * TV.iso8608_gd('C') * v, rtol=1e-9)
    assert TV.road_psd_spatial(np.array([0.005, 3.0]), 'C').tolist() == [0.0, 0.0]


def test_quarter_car_against_the_textbook_formula():
    ms, mu, ks, cs, kt = 300.0, 35.0, 23000.0, 1500.0, 200000.0
    f = np.geomspace(0.3, 40, 120)
    w = 2 * np.pi * f
    # Gillespie (1992) eq. 5-12 form: z_s/z_r = kt·(ks + iωc) / [(ks + iωc − ms ω²)(ks + kt + iωc − mu ω²) − (ks + iωc)²]
    num = kt * (ks + 1j * w * cs)
    den = (ks + 1j * w * cs - ms * w ** 2) * (ks + kt + 1j * w * cs - mu * w ** 2) - (ks + 1j * w * cs) ** 2
    assert np.allclose(TV.quarter_car(f, ms, mu, ks, cs, kt), num / den, rtol=1e-10)
    # a massless wheel on a rigid tyre: the single-mode base-excitation transmissibility
    zeta = 0.3
    fn = math.sqrt(ks / ms) / (2 * math.pi)
    one = TV.quarter_car(f, ms, 1e-6, ks, 2 * zeta * math.sqrt(ks * ms), 1e12)
    assert np.allclose(np.abs(one), TV.transmissibility(f / fn, zeta), rtol=1e-6)
    # the mount mode: absolute tip acceleration over base = 1 − ω²·(relative per base acceleration)
    Hr = TV.sdof_relative(f, 12.0, 0.05)
    assert np.allclose(np.abs(1 - w ** 2 * Hr), TV.transmissibility(f / 12.0, 0.05), rtol=1e-10)
    assert TV.transmissibility(np.array([1.0]), 0.05)[0] == pytest.approx(math.sqrt(1 + 0.01) / 0.1)


def test_half_car_follows_the_road_slowly_and_has_car_like_modes():
    p = _data_rig()
    hc = TV.half_car(p, TV.ThermoVibSettings())
    X = hc.response(np.array([0.005]), 50 / 3.6)
    assert abs(X[0, 0]) == pytest.approx(1.0, abs=1e-3)               # heave follows a long wave
    tau = hc.wheelbase / (50 / 3.6)
    assert abs(X[0, 1]) == pytest.approx(2 * math.pi * 0.005 * tau / hc.wheelbase, rel=0.02)   # pitch: the grade
    m = hc.modes()
    assert 0.9 < m['bounce_hz'] < 2.0 and 0.8 < m['pitch_hz'] < 2.0
    assert all(9.0 < h < 16.0 for h in m['hop_hz'])
    su = SU.settings_for(p)
    assert hc.k_f == pytest.approx((su.corners['FL'].wheel_rate + su.corners['FR'].wheel_rate) * 1000)


def test_resonance_flagged_when_the_first_mode_is_placed_in_band():
    assert TV.resonance(1.5) == ['body']
    assert TV.resonance(12.0) == ['wheel hop']
    assert TV.resonance(30.0, [('excitation peak', 28.0)]) == ['excitation peak']
    assert TV.resonance(40.0, [('excitation peak', 28.0)]) == []
    # a thin mast tuned to 12.5 Hz with the lidar's mass: flagged, a serious finding and an issue
    p = _data_rig()
    top = TV.roof_box(p)['top']
    mast = RigPart(name='Thin mast', kind='cylinder', x=1.0, y=0.0, z=top + 0.3, height=0.6, radius=0.012)
    p.rig = [mast]
    L1 = next(s for s in p.variant.sensors if s.tag == 'L1')
    L1.x, L1.y, L1.z = 1.0, 0.0, top + 0.6 + 0.04
    k = 3 * TV.MATERIALS['al6061']['E'] * TV.tube_I(0.012) / 0.6 ** 3
    m_tip = k / (2 * math.pi * 12.5) ** 2 - 0.24 * SU.rig_part_mass(mast)
    p.analysis.suspension = {'loads': {'rig_masses': {'L1': m_tip}}}
    res = TV.run(p, events=False)
    d = res.device(L1.uid)
    assert d['on'] == 'Thin mast'
    mm = next(m for m in d['modes'] if m['kind'] == 'mast')
    assert mm['f1_hz'] == pytest.approx(12.5, rel=1e-6)
    assert any('wheel hop' in r['why'] for r in d['resonance'])
    f = [f for f in res.findings if f['kind'] == 'resonance' and L1.uid in f['uids']]
    assert f and f[0]['severity'] == 'serious' and '12.5 Hz' in f[0]['message'] and 'wheel-hop' in f[0]['message']
    assert res.findings[0]['severity'] == 'serious'
    iss = TV.issues(p)
    assert any(i.severity == 'serious' and L1.uid in i.uids for i in iss)
    # the vibration there dwarfs the sample's stiff roof mount
    ref = next(x for x in TV.run(_data_rig(), events=False).devices if x['tag'] == 'L1')
    assert d['vib']['flex_deg_rms'] > 20 * ref['vib']['flex_deg_rms']
    assert not ref['resonance']


# =========================================================================== heat
def test_energy_balance_and_sun():
    plate = RigPart(name='Plate', kind='box', x=0, y=0, z=2, length=1.0, width=1.0, height=0.02)
    tp = TV.thermal_part(plate, TV.MATERIALS['al6061'], TV.FINISHES['black_anodised'])
    q = TV.absorbed(tp, np.array([0.0, 0.0, 1.0]), 1000.0, 100.0, 1100.0, 0.2)
    # top (DNI + DHI), bottom (ground-reflected), four 20 mm sides (half the sky, half the ground)
    assert q.sum() == pytest.approx(0.88 * (1100 + 0.2 * 1100 + 4 * 0.02 * (100 / 2 + 0.2 * 1100 / 2)), rel=1e-9)
    top = TV.absorbed(tp, np.array([0.0, 0.0, 1.0]), 1000.0, 100.0, 1100.0, 0.2)[0]
    assert top == pytest.approx(0.88 * 1.0 * (1000 + 100))
    still = TV.solve_part(tp, q, 30.0, 0.0)
    windy = TV.solve_part(tp, q, 30.0, 15.0)
    assert 30 + 30 < still['mean'] < 30 + 70                       # a black plate in still air: 60-100 °C
    assert 30 < windy['mean'] < still['mean']                      # wind cools it
    assert still['dT'] > 0                                          # the sun-lit top is warmer
    dark = TV.solve_part(tp, q * 0, 30.0, 0.0)
    assert dark['mean'] < 30.0                                      # a clear night sky cools it below the air
    # one node, no radiation: absorbed = h·A·ΔT with the h it reports
    one = TV.ThermalPart('x', np.array([[0, 0, 1.0]]), np.array([[0, 0, 2.0]]), np.array([2.0]), np.array([0]), 0.0,
                         ('plate', 1.0), 1.0, 0.0, 1000.0, False, np.array([0, 0, 1.0]))
    r = TV.solve_part(one, np.array([600.0]), 25.0, 10.0)
    assert 600.0 == pytest.approx(r['h'] * 2.0 * (r['mean'] - 25.0), rel=1e-6)
    # Meinel at the zenith: 1361·0.7^1 W/m² direct (Kasten-Young air mass 1 there)
    dni, dhi, ghi = TV.clear_sky(90.0, 1.0)
    assert dni == pytest.approx(1361 * 0.7 ** (float(TV.air_mass(90.0)) ** 0.678))
    assert float(TV.air_mass(90.0)) == pytest.approx(1.0, abs=1e-3)
    assert TV.clear_sky(-1.0) == (0.0, 0.0, 0.0)
    st = TV.ThermoVibSettings()
    assert TV.ambient(st.peak_hour, st) == pytest.approx(st.t_max_c)
    assert TV.ambient(st.peak_hour + 12, st) == pytest.approx(st.t_max_c - st.daily_range_c)


def test_hot_day_and_hours_follow_the_exposure_settings():
    p = _data_rig()
    day = TV.day_hours(p, TV.ThermoVibSettings())
    assert day['date'] in ('2026-06-20', '2026-06-21', '2026-06-22')   # the solstice at Munich, the exposure default
    assert len(day['local_h']) == 10 and day['local_h'][0] == pytest.approx(8.5)
    assert max(day['el']) > 60
    p.analysis.exposure = {'lat': -33.9, 'lon': 151.2, 'utc_offset_h': 10, 'hour_from': 9, 'hour_to': 12}
    day = TV.day_hours(p, TV.ThermoVibSettings())
    assert day['date'][5:7] == '12' and len(day['local_h']) == 3     # Sydney: December
    day = TV.day_hours(p, TV.ThermoVibSettings(date='2026-03-20'))
    assert day['date'] == '2026-03-20' and not day['auto_date']


def test_sample_drift_body_expansion_by_hand(sample_result):
    res = sample_result
    assert res.day['date'] == '2026-06-21' and res.heading_auto
    by = {d['tag']: d for d in res.devices}
    # a body-mounted radar: the body at air temperature, α_steel·(T_air − T_cal)·its distance from the origin
    rd1 = by['RD1']
    assert rd1['on'] == 'body'
    k = int(np.argmax([h['t_air_c'] for h in res.hours]))
    dT = res.hours[k]['t_air_c'] - res.settings.t_cal_c
    s = rd1['thermal'][k]
    P = next(x for x in _data_rig().variant.sensors if x.tag == 'RD1').pos()
    assert s['dx_mm'] == pytest.approx(11.7e-6 * dT * P[0] * 1000, rel=1e-9)
    assert s['turn_deg'] == pytest.approx(0.0, abs=1e-12)
    assert by['L1']['on'] == 'roof'
    # body-mounted pairs do not turn apart: only the parallax of their moves is left
    pr = next(pr for pr in res.pairs if pr['label'] == 'F–L1')
    assert pr['thermal_worst']['deg'] == pytest.approx(0.0, abs=1e-12)
    assert pr['thermal_worst']['px_rot'] == pytest.approx(0.0, abs=1e-9)
    assert 0 < pr['thermal_worst']['px'] < 0.2
    assert res.findings[-1]['kind'] in ('typical', 'no_rig', 'ok')
    assert any(f['kind'] == 'ok' and f['severity'] == 'good' for f in res.findings)


def test_rig_on_the_roof_warps_and_flexes():
    from vantage.core import rig
    p = _data_rig()
    p.rig = rig.preset('roof_rack', p.vehicle.mesh) + rig.preset('sensor_mast', p.vehicle.mesh)
    head = p.rig[-1]
    L1 = next(s for s in p.variant.sensors if s.tag == 'L1')
    L1.x, L1.y, L1.z = head.x, head.y, head.z + head.height / 2 + 0.04
    res = TV.run(p, settings=TV.ThermoVibSettings(heading_deg=90.0))
    assert res.heading_deg == 90.0 and not res.heading_auto
    parts = {pt['name']: pt for pt in res.parts}
    assert parts['Roof rack · platform']['on'] == ['Roof rack · bar 1', 'Roof rack · bar 2']
    assert parts['Sensor mast · head']['on'] == ['Sensor mast']
    assert parts['Roof rack · bar 1']['mode']['kind'] == 'bar'
    d = res.device(L1.uid)
    assert d['chain'] == ['Sensor mast', 'Sensor mast · head']
    assert d['thermal_worst']['deg'] > 0                         # the sun bows the mast
    assert [m['kind'] for m in d['modes']] == ['mast', 'bracket']
    assert d['f1_hz'] == pytest.approx(d['modes'][0]['f1_hz'])
    pk = max(res.hours, key=lambda h: h['t_air_c'])
    assert pk['parts'][parts['Roof rack · platform']['uid']]['mean_c'] > pk['t_air_c']
    assert any(f['kind'] == 'rack_mismatch' for f in res.findings)
    pr = next(pr for pr in res.pairs if pr['label'] == 'F–L1')
    assert pr['thermal_worst']['deg'] > 0 and pr['thermal_worst']['px_rot'] > 0


# =========================================================================== lens, blur, events
def test_pixels_through_the_lens_and_blur_scales_with_exposure():
    p = _data_rig()
    devs = TV.devices(p)
    F = next(d for d in devs if d['tag'] == 'F')
    from vantage.core.mathutil import BODY_TO_OPTICAL
    W2O = BODY_TO_OPTICAL @ F['R'].T
    ang = 1e-4
    axis = F['R'][:, 2]                                           # about the camera's up axis: a pan
    c, worst = TV.px_shift(F['intr'], W2O, axis * ang)
    assert c == pytest.approx(TV.px_per_rad(F['intr']) * ang, rel=1e-3)
    assert worst >= c
    assert TV.px_shift(F['intr'], W2O, np.zeros(3)) == (0.0, 0.0)
    a = TV.run(p, events=False).device(F['uid'])['vib']['blur_px']
    p.analysis.timing = {'exposure_ms': 4.0}
    b = TV.run(p, events=False).device(F['uid'])['vib']['blur_px']
    assert b == pytest.approx(2 * a, rel=1e-9)                    # ω_rms × exposure × px/rad
    from vantage.core.timing import TimingSettings
    assert TimingSettings().exposure_ms == 2.0


def test_events_bump_and_joint(sample_result):
    res = sample_result
    ev = res.events
    assert ev['joint']['profile'] == 'step' and ev['bump']['profile'] == 'half-sine'
    assert ev['joint']['peak_heave_mm'] >= res.settings.joint_mm            # a step overshoots its height
    assert ev['bump']['peak_heave_mm'] > 0 and ev['bump']['peak_pitch_deg'] > 0
    assert ev['cobble']['sett_hz'] == pytest.approx(30 / 3.6 / 0.12)
    for d in res.devices:
        e = d['events']['bump']
        md = min(d['modes'], key=lambda m: m['f1_hz'])
        assert e['mount_ring_down_s'] == pytest.approx(math.log(10) / (md['zeta'] * 2 * math.pi * md['f1_hz']))
        assert e['peak_angle_deg'] >= e['peak_flex_deg']
        assert 0 < e['body_ring_down_s'] < TV.TAIL_S


# =========================================================================== settings, determinism, output
def test_settings_are_tolerant():
    s = TV.ThermoVibSettings.from_dict({'t_cal_c': 'warm', 't_max_c': 400, 'road_class': 'e', 'feet': 'glued',
                                        'heading_deg': 370, 'date': 'tomorrow', 'material': 'cfrp', 'junk': 1,
                                        'speed_kmh': float('nan'), 'tol_source': 'observability'})
    assert s.t_cal_c == 15.0 and s.t_max_c == 60.0 and s.road_class == 'E' and s.feet == 'sliding'
    assert s.heading_deg == 10.0 and s.date == '' and s.material == 'cfrp' and s.speed_kmh == 50.0
    assert s.tol_source == 'observability'
    assert TV.ThermoVibSettings.from_dict(None).to_dict() == TV.ThermoVibSettings().to_dict()
    assert TV.ThermoVibSettings.from_dict(s.to_dict()).to_dict() == s.to_dict()
    p = _data_rig()
    p.analysis.observability = {'need_deg': 0.2, 'need_mm': 5.0}
    t = TV.tolerance(p, s)
    assert (t['deg'], t['mm']) == (0.2, 5.0)
    assert TV.tolerance(p, TV.ThermoVibSettings())['deg'] == 1.0


def test_deterministic_and_exports(sample_result, tmp_path):
    p = _data_rig()
    a = TV.to_doc(TV.run(p), p)
    b = TV.to_doc(TV.run(p), p)
    a.pop('seconds')
    b.pop('seconds')
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    assert a['schema'] == TV.SCHEMA and a['frame'].startswith('vehicle')
    assert set(a['settings']) == set(TV.ThermoVibSettings().to_dict())
    assert 'Out of scope' in a['limits'] and a['assumptions']
    assert len(a['hours']) == 10 and len(a['devices']) == 12
    assert a['psd']['third_octave_hz'] and len(a['psd']['road_m2_hz']) == len(a['psd']['third_octave_hz'])
    names = [n for n, _ in TV.files(sample_result, _data_rig())]
    assert names == ['thermo_vib.json', 'thermo_vib_thermal.csv', 'thermo_vib_pairs.csv', 'thermo_vib_vibration.csv',
                     'thermo_vib_psd.csv', 'thermo_vib.png']
    paths = TV.export(sample_result, _data_rig(), str(tmp_path))
    data = {n: open(pth, 'rb').read() for n, pth in zip(names, paths)}
    assert json.loads(data['thermo_vib.json'])['schema'] == TV.SCHEMA
    assert data['thermo_vib.png'][:8] == b'\x89PNG\r\n\x1a\n'
    assert data['thermo_vib_thermal.csv'].decode().splitlines()[0].startswith('hour,device')
    assert len(data['thermo_vib_thermal.csv'].decode().splitlines()) == 1 + 10 * 12
    sm = TV.summary(sample_result)
    assert sm['lowest_mode']['device'] in ('QL', 'QR') and sm['resonances'] == []


def test_no_devices_and_cancel():
    p = _data_rig()
    for c in p.variant.cameras:
        c.visible = False
    for s in p.variant.sensors:
        s.visible = False
    res = TV.run(p)
    assert res.devices == [] and res.findings[0]['kind'] == 'empty'
    assert TV.issues(p) == []
    assert TV.run(_data_rig(), cancel=lambda: True) is None


# =========================================================================== the page
from test_gui import app, win, pump  # noqa: E402,F401  (fixtures)


def _wait(app, pg, t=30.0):
    import time
    t0 = time.time()
    while (pg.busy or pg.result is None) and time.time() - t0 < t:
        pump(app, 0.05)


def test_page_runs_edits_and_exports(app, win, tmp_path):
    from vantage.ui.workspaces.thermo_vib_ws import ThermoVibPage
    p = win.ctx.store.project
    vid = next(v.uid for v in p.variants if v.name.startswith('Data rig'))
    p.active_uid = vid
    pg = ThermoVibPage(win.ctx)
    pg.resize(1500, 900)
    pg.run()
    _wait(app, pg)
    assert pg.result is not None and not pg.busy
    assert pg.dev_table.rowCount() == 12
    assert pg.pair_table.rowCount() == 6
    assert pg.find_box.count() > 1
    assert 'heading' in pg.head.text()
    for k in ('move', 'pairs', 'angle'):
        pg.metric.buttons[k].click()
        assert pg.day_chart.series
    pg.tabs.setCurrentIndex(1)
    pg.dev_table.selectRow(6)
    assert pg.c_dev.currentData() == pg.result.devices[6]['uid']
    assert pg.psd_chart.series and pg.tf_chart.series
    pg.grab()                                              # paints every chart
    before = copy.deepcopy(p.analysis.thermo_vib)
    assert pg.set_option('road_class', 'E')
    assert TV.settings_of(win.ctx.store.project).road_class == 'E'
    assert not pg.set_option('road_class', 'E')            # nothing changes
    pg.run()
    _wait(app, pg)
    assert pg.result.settings.road_class == 'E'
    win.ctx.store.undo.undo()
    assert win.ctx.store.project.analysis.thermo_vib == before
    paths = pg.export(str(tmp_path))
    assert any(x.endswith('thermo_vib.json') for x in paths)
    pg.set_option('heading_deg', 180.0)
    assert TV.settings_of(win.ctx.store.project).heading_deg == 180.0
    pg.run()
    pg.cancel()
    assert not pg.busy
