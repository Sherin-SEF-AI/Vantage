"""Calibration perturbation and OOD stress tests (core/stress.py): reproducible, the identity at zero severity, knocks
as steps, bounded thermal drift, effects against the drift-shift closed form, and the exported perturbed
calibrations moved from the plan by exactly the schedule's pose."""
import json
import math

import numpy as np
import pytest

from vantage.core import samples
from vantage.core import stress as SX
from vantage.core.mathutil import rot_zyx, rot_y, quat_from_matrix, BODY_TO_OPTICAL
from vantage.core.research_maps import drift_shift_cm


@pytest.fixture(scope='module')
def rig():
    p = samples.tigor_sample_project()
    p.active_uid = next(v.uid for v in p.variants if v.sensors)
    return p


def test_same_seed_same_schedule_and_settings_are_tolerant(rig):
    a = SX.schedule(rig, {'severity': 'severe', 'seed': 3, 'duration_s': 20})
    b = SX.schedule(rig, {'severity': 'severe', 'seed': 3, 'duration_s': 20})
    c = SX.schedule(rig, {'severity': 'severe', 'seed': 4, 'duration_s': 20})
    for ea, eb in zip(a.sensors, b.sensors):
        assert np.array_equal(ea['R'], eb['R']) and np.array_equal(ea['p'], eb['p'])
    assert a.knocks == b.knocks
    assert any(not np.allclose(ea['mount_rot_deg'], ec['mount_rot_deg']) for ea, ec in zip(a.sensors, c.sensors))
    st = SX.StressSettings.from_dict({'severity': 'BOGUS', 'rate_hz': 'x', 'duration_s': -5, 'sets': 999,
                                      'knock_deg': [0.5, 2], 'thermal_dT': float('nan')})
    assert st.severity == 'moderate' and st.rate_hz == 10.0 and st.duration_s == 0.1 and st.sets == 50
    assert st.level('knock_deg') == (0.5, 2.0) and st.level('thermal_dT') == SX.SEVERITY['moderate']['thermal_dT']
    assert len(a.t) == 200 and a.t[1] == pytest.approx(0.1)


def test_zero_severity_is_the_plan(rig):
    sc = SX.schedule(rig, {'severity': 'none', 'duration_s': 5})
    assert not sc.knocks and np.all(sc.body['pitch_deg'] == 0)
    for e in sc.sensors:
        assert np.allclose(e['R'], e['R0'][None]) and np.allclose(e['p'], e['p0'][None])
        if 'effects' in e:
            assert all(np.allclose(v, 0.0) for v in e['effects']['ground_err_cm'].values())
            assert np.allclose(e['effects']['far_px'], 0.0, atol=1e-6)
    q = SX.apply(rig, sc, 2.0)
    for a, b in zip(q.variant.cameras, rig.variant.cameras):
        assert (a.yaw, a.pitch, a.roll, a.x, a.y, a.z) == pytest.approx((b.yaw, b.pitch, b.roll, b.x, b.y, b.z),
                                                                         abs=1e-9)
    assert q is not rig and q.variant is not rig.variant


def test_a_knock_is_a_step_at_its_time(rig):
    sc = SX.schedule(rig, {'severity': 'severe', 'seed': 11, 'duration_s': 60, 'thermal_dT': 0, 'vib_rms_deg': 0,
                           'vib_rms_mm': 0, 'brake_g': 0, 'accel_g': 0, 'corner_g': 0})
    assert sc.knocks
    k = sc.knocks[0]
    e = sc.sensor(k['tag'])
    assert e['kind'] == 'camera'                                   # knocks hit single cameras
    mag = np.linalg.norm(e['parts']['knock_rot_deg'], axis=1)
    i = int(np.searchsorted(sc.t, k['t_s']))
    assert np.all(mag[:i] == 0) and mag[i] == pytest.approx(k['deg'], rel=1e-9)
    later = [q for q in sc.knocks if q['tag'] == k['tag'] and q['t_s'] > k['t_s']]
    j = int(np.searchsorted(sc.t, later[0]['t_s'])) if later else len(sc.t)
    assert np.allclose(mag[i:j], k['deg'])                         # it stays until the next knock on that camera
    lo, hi = SX.SEVERITY['severe']['knock_deg']
    assert all(lo <= q['deg'] <= hi for q in sc.knocks)
    others = [s for s in sc.sensors if s['tag'] not in {q['tag'] for q in sc.knocks}]
    assert all(np.all(s['parts']['knock_rot_deg'] == 0) for s in others)


def test_thermal_drift_is_bounded_and_follows_the_warm_up(rig):
    st = {'severity': 'severe', 'duration_s': 600, 'rate_hz': 2, 'knock_rate_per_min': 0, 'vib_rms_deg': 0,
          'vib_rms_mm': 0, 'brake_g': 0, 'accel_g': 0, 'corner_g': 0}
    sc = SX.schedule(rig, st)
    lev = SX.SEVERITY['severe']
    bound_deg = lev['thermal_dT'] * lev['thermal_deg_per_C']
    bound_mm = lev['thermal_dT'] * lev['thermal_mm_per_C']
    for e in sc.sensors:
        r = np.linalg.norm(e['parts']['thermal_rot_deg'], axis=1)
        m = np.linalg.norm(e['parts']['thermal_tr_mm'], axis=1)
        assert r.max() <= bound_deg + 1e-12 and m.max() <= bound_mm + 1e-12
        assert r[0] == 0 and np.all(np.diff(r) >= -1e-12) and r[-1] > 0.3 * bound_deg    # it warms up, never back


def test_effects_match_the_drift_shift_closed_form(rig):
    cam = rig.variant.by_tag('F')
    R0, C0 = rot_zyx(cam.yaw, cam.pitch, cam.roll), np.asarray(cam.pos(), float)
    a = 0.2
    for d in (10.0, 20.0):
        az = math.atan2(R0[1, 0], R0[0, 0])
        P = np.array([C0[0] + d * math.cos(az), C0[1] + d * math.sin(az), 0.0])
        ref = drift_shift_cm(C0, R0, P, a)['pitch']                  # the worst of ±0.2° about its own pitch axis
        got = max(SX.ground_error_m(C0, R0, C0, R0 @ rot_y(math.radians(s * a)), d)[0] * 100 for s in (1, -1))
        assert got == pytest.approx(ref, rel=1e-9)
        # and the closed form r·δθ / sin(depression) for a pitch small against the depression (0.01° here: at 10 m
        # F sees the ground only 3.4° down, so 0.2° is already second-order)
        dep = math.atan2(C0[2], d)
        r = math.hypot(d, C0[2])
        small = SX.ground_error_m(C0, R0, C0, R0 @ rot_y(math.radians(0.01)), d)[0] * 100
        assert small == pytest.approx(r * math.radians(0.01) / math.sin(dep) * 100, rel=0.01)
    # a pure translation shifts the ground point by its horizontal part
    sh = SX.ground_error_m(C0, R0, C0 + np.array([0.01, 0.02, 0.0]), R0, 10.0)[0]
    assert sh == pytest.approx(math.hypot(0.01, 0.02), rel=1e-9)
    # the schedule's own effects equal ground_error_m of its poses; BEV cells are that over the cell size
    sc = SX.schedule(rig, {'severity': 'moderate', 'duration_s': 10})
    e = sc.sensor('F')
    g20 = SX.ground_error_m(C0, R0, e['p'], e['R'], 20.0) * 100
    assert np.allclose(np.nan_to_num(e['effects']['ground_err_cm']['20'], posinf=-1),
                       np.nan_to_num(g20, posinf=-1))
    assert np.allclose(e['effects']['bev_cells']['20'][np.isfinite(g20)], g20[np.isfinite(g20)] / 100 / 0.5)
    # the far-point shift is the lens's angular sampling × the rotation for a small turn about the optical axis' normal
    from vantage.core.lens import resolve
    intr = resolve(rig.module_for(cam), rig.lens_for(cam))
    sc0 = SX.schedule(rig, {'severity': 'none', 'duration_s': 1})
    e0 = sc0.sensor('F')
    R1 = (R0 @ rot_y(math.radians(0.1)))[None]
    far = SX.camera_effects(rig, cam, R1, C0[None], sc0.settings)['far_px'][0]
    pxr = intr.f_px                                                  # r = f·θ near the axis
    assert far == pytest.approx(pxr * math.radians(0.1), rel=0.02)
    assert e0['effects']['far_px'][0] == pytest.approx(0.0, abs=1e-6)


def test_body_motion_turns_every_sensor_together(rig):
    sc = SX.schedule(rig, {'severity': 'severe', 'duration_s': 30, 'thermal_dT': 0, 'knock_rate_per_min': 0,
                           'vib_rms_deg': 0, 'vib_rms_mm': 0})
    i = int(np.argmax(sc.body['pitch_deg']))                         # hardest braking: nose down
    assert sc.body['pitch_deg'][i] > 0 and sc.long_g[i] < 0 and sc.body['pitch_per_g'] < 0
    for e in sc.sensors:
        dR = e['R'][i] @ e['R0'].T
        assert np.allclose(dR, sc.sensors[0]['R'][i] @ sc.sensors[0]['R0'].T, atol=1e-9)


def test_export_writes_shifted_calibrations(rig, tmp_path):
    paths = SX.export_stress(rig, None, str(tmp_path), {'severity': 'severe', 'duration_s': 20, 'sets': 3})
    doc = json.loads((tmp_path / 'stress_schedule.json').read_text())
    assert doc['schema'] == SX.SCHEMA and doc['settings']['severity'] == 'severe' and len(doc['t_s']) == 200
    assert (tmp_path / 'stress_schedule.csv').read_text().splitlines()[0].startswith('t_s,tag,kind,d_yaw_deg')
    assert 'illustrative' in (tmp_path / 'stress_report.md').read_text()
    sets = doc['calibration_sets']
    assert 2 <= len(sets) <= 3 and sum(s['worst'] for s in sets) == 1
    sc = SX.schedule(rig, {'severity': 'severe', 'duration_s': 20})
    for s in sets:
        cal = json.loads((tmp_path / s['dir'] / 'nuscenes' / 'calibrated_sensor.json').read_text())
        q = SX.apply(rig, sc, s['t_s'])
        cam = q.variant.by_tag('F')
        R = rot_zyx(cam.yaw, cam.pitch, cam.roll)
        O2V = (BODY_TO_OPTICAL @ R.T).T                               # optical -> vehicle, as nuScenes stores it
        x, y, z, w = quat_from_matrix(O2V)
        row = next(r for r in cal if r['token'] and r['camera_intrinsic'] and
                   r['vantage_distortion']['camera'] == 'F')
        qq = np.array(row['rotation'])
        assert min(np.abs(qq - [w, x, y, z]).max(), np.abs(qq + [w, x, y, z]).max()) < 1e-8
        assert row['translation'] == pytest.approx([cam.x, cam.y, cam.z], abs=2e-6)
        # moved from the plan by the schedule's delta at that frame
        e = sc.sensor('F')
        i = s['frame']
        assert np.linalg.norm(np.array(row['translation']) - e['p0']) * 1000 == pytest.approx(
            np.linalg.norm(e['delta_xyz_mm'][i]), abs=0.01)
    assert all(p.startswith(str(tmp_path)) for p in paths)
