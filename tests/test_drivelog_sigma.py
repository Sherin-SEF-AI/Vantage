"""Drive-log check: the rotation estimate's standard error. The error model on its own (drivelog._stderr: the
frames' own optima's scatter pooled with a prior, plus the scene error frames close together along the drive share) checked by
simulation against its own assumptions; the travel the frames are spread over (odometry, a speed, nothing); the
synthetic drive of test_drivelog (each checked camera within three standard errors of the truth, the frames'
independent worth stated); and, when VANTAGE_CARLA_DRIVES names a folder of recorded CARLA drives (each with
nuscenes/ and run.json, as Vantage's bridge writes them), the coverage of successive window estimates and of each
drive's estimate against CARLA's ground truth on yaw and roll (pitch carries a systematic error on CARLA's scenes,
see round7-notes/reports12/errbars.md)."""
import dataclasses
import glob
import json
import math
import os

import numpy as np
import pytest

from vantage.core import drivelog as DL


def _sim(n, spacing_m, rng, reps=4000, noise_deg=0.0):
    """reps draws of n frames spaced spacing_m apart: each frame's error a Gaussian of SCENE_DEG correlated by
    exp(−distance / SCENE_M) plus independent noise (the frames' own optima); the joint estimate their mean. Returns
    z = error / stderr."""
    s = np.arange(n) * spacing_m
    C = DL.SCENE_DEG ** 2 * np.exp(-np.abs(s[:, None] - s[None, :]) / DL.SCENE_M) + noise_deg ** 2 * np.eye(n)
    L = np.linalg.cholesky(C + 1e-12 * np.eye(n))
    z = []
    for _ in range(reps):
        e = L @ rng.standard_normal((n, 3))
        est = e.mean(0)
        se, _ = DL._stderr(e, s, None)
        z.append(est / se)
    return np.concatenate(z)


@pytest.mark.parametrize('n,spacing,noise', [(4, 2.5, 0.0), (20, 4.0, 0.0), (8, 0.5, 0.0), (20, 0.0, 0.0),
                                             (20, 4.0, 0.2), (3, 6.0, 0.05)])
def test_the_error_model_covers_its_own_assumptions(n, spacing, noise):
    """Simulated under the model's own scene correlation (and with frame noise the prior does not know of): ±1σ and
    ±2σ hold about 68 % and 95 % of the errors, never far below; frames of a parked vehicle (spacing 0) too."""
    z = np.abs(_sim(n, spacing, np.random.default_rng(n * 100 + int(spacing * 10)), 1500, noise))
    c1, c2 = float(np.mean(z < 1)), float(np.mean(z < 2))
    assert 0.6 <= c1 <= 0.85 and 0.9 <= c2 <= 0.995, (c1, c2)


def test_one_view_of_one_scene_keeps_the_scene_error_and_far_frames_average_it():
    rng = np.random.default_rng(3)
    per = rng.normal(0, 0.02, (20, 3))
    parked, n_p = DL._stderr(per, np.zeros(20), None)
    driving, n_d = DL._stderr(per, np.arange(20) * 50.0, None)
    assert n_p == pytest.approx(1.0) and n_d == pytest.approx(20.0, rel=1e-6)
    assert np.all(parked == pytest.approx(DL.SCENE_DEG, rel=0.1))         # about one frame's scene error
    assert np.all(driving < 0.5 * DL.SCENE_DEG)                           # twenty independent views
    one, n1 = DL._stderr(np.zeros((0, 3)), None, [])
    assert n1 == 0.0 and np.all(one == DL.SCENE_DEG)
    # without travel the frames' times stand in: SCENE_S apart share as much as SCENE_M apart
    a, _ = DL._stderr(per[:4], None, np.arange(4) * DL.SCENE_S)
    b, _ = DL._stderr(per[:4], np.arange(4) * DL.SCENE_M, None)
    assert a == pytest.approx(b)


def test_a_frame_that_disagrees_widens_the_error_without_swamping_it():
    rng = np.random.default_rng(5)
    base = rng.normal(0, 0.05, (12, 3))
    odd = base.copy()
    odd[4, 0] = 2.0                                      # one frame caught at an optimum 2° away in yaw
    far = np.arange(12) * 10.0
    s0, _ = DL._stderr(base, far, None)
    s1, _ = DL._stderr(odd, far, None)
    plain = math.sqrt((DL.SCENE_NU * DL.SCENE_DEG ** 2 + 11 * odd[:, 0].var(ddof=1)) / (DL.SCENE_NU + 11) / 12)
    assert s0[0] < s1[0] < 0.5 * plain                   # winsorised at three robust standard deviations
    assert s1[1] == pytest.approx(s0[1]) and s1[2] == pytest.approx(s0[2])


def test_travel_from_odometry_from_a_speed_and_from_nothing():
    t = np.linspace(0.0, 2.0, 21)
    pos = np.stack([3.0 * t, np.zeros_like(t), np.zeros_like(t)], 1)
    odo = DL.Motion('odometry', t, pos, np.zeros((len(t), 3)), speed=(t, np.full(len(t), 3.0)))
    assert DL._travel(odo, [0.5, 1.5]) == pytest.approx([1.5, 4.5])
    sp = DL.Motion('imu', t, omega=np.zeros((len(t), 3)), speed=(t, 2.0 * t))      # accelerating at 2 m/s²
    assert DL._travel(sp, [1.0, 2.0]) == pytest.approx([1.0, 4.0], abs=0.02)
    assert DL._travel(DL.Motion('none'), [0.0, 1.0]) is None


def test_the_synthetic_drive_lies_within_its_standard_errors(tmp_path_factory):
    """test_drivelog's drive (ML rendered 0.6° off in yaw): every checked camera's pinned-down axes within three
    standard errors of the truth, and its frames counted by their independent worth (the vehicle stands for half the
    drive: fewer than the frames)."""
    import test_drivelog as TD
    from vantage.core.analysis import Engine
    p, v = TD.make_project()
    eng = Engine()
    path = tmp_path_factory.mktemp('sigma') / 'rig_drive'
    TD.make_drive(path, p, v, eng)
    res = DL.run(p, str(path), DL.DriveLogSettings(camera_stamp='trigger', frames_per_camera=TD.N_FRAMES), engine=eng)
    checked = [r for r in res.extrinsics if r.get('delta_deg') is not None]
    assert {r['tag'] for r in checked} >= {'F', 'R', 'ML', 'MR'}
    for r in checked:
        truth = [TD.YAW_ERR.get(r['tag'], 0.0), 0.0, 0.0]
        for k in range(3):
            if r['observable'][k]:
                assert abs(r['delta_deg'][k] - truth[k]) <= 3.0 * r['stderr_deg'][k], (r['tag'], k, r['delta_deg'],
                                                                                       r['stderr_deg'])
        assert 1.0 <= r['frames_independent'] < r['frames']
        assert min(r['stderr_deg']) >= 0.3 * DL.SCENE_DEG
    ml = res.extrinsic('ML')
    assert ml['verdict'] == 'rotated' and ml['confidence'] in ('high', 'medium')
    doc = DL.to_doc(res, p)
    row = next(c for c in doc['extrinsics']['cameras'] if c['camera'] == 'ML')
    assert row['independent_frames'] == pytest.approx(ml['frames_independent'], abs=0.01)


# ---------------------------------------------------------------------------- recorded CARLA drives (optional)
CARLA = os.environ.get('VANTAGE_CARLA_DRIVES', '')


def _carla_drives():
    if not CARLA:
        return []
    return sorted(d for d in glob.glob(os.path.join(CARLA, '**', 'run.json'), recursive=True)
                  if os.path.isdir(os.path.join(os.path.dirname(d), 'nuscenes')))[:int(os.environ.get(
                      'VANTAGE_CARLA_DRIVES_MAX', '4'))]


@pytest.mark.skipif(not _carla_drives(), reason='set VANTAGE_CARLA_DRIVES to a folder of recorded CARLA drives')
def test_recorded_carla_drives_window_and_truth_coverage():
    """Each drive cut into 1.25 s windows of four frames (as online calibration cuts it): successive window estimates
    of a camera differ within their combined standard error as often as a Gaussian would (±1σ 60–80 %, ±2σ ≥ 88 %; a
    scene that fits a wrong rotation in all four frames of a window is what remains beyond 3σ); each drive's own
    estimate (20 frames) against CARLA's truth on yaw and roll likewise; and no standard error absurdly large (the
    median under 0.1°)."""
    from vantage.core import samples, timing as TM
    from vantage.core.analysis import Engine
    eng = Engine()
    zw, zt, ses = [], [], []
    for rj in _carla_drives():
        d = os.path.dirname(rj)
        run = json.load(open(rj))
        truth = {f['tag']: f['pose_delta_deg'] for f in run.get('faults_injected') or [] if f['kind'] == 'rotation'}
        p = samples.tigor_sample_project()
        var = next((v for v in p.variants if v.name == run.get('variant')), None)
        if var is None:
            continue
        path = os.path.join(d, 'nuscenes')
        res = DL.run(p, path, DL.DriveLogSettings(frames_per_camera=20), engine=eng, variant=var)
        for r in res.extrinsics:
            if r.get('delta_deg') is None or r['confidence'] not in ('high', 'medium'):
                continue
            t = truth.get(r['tag'], [0.0, 0.0, 0.0])
            zt += [(r['delta_deg'][k] - t[k]) / r['stderr_deg'][k] for k in (0, 2) if r['observable'][k]]
        n = int(round(res.scan.duration_s / 0.3125))
        scan = DL.scan_log(path, n, 20)
        st = DL.DriveLogSettings(path=path)
        mapping = DL.mapping_for(var, scan, {})
        rev = {g: tp for tp, g in mapping.items()}
        plan = TM.plan(p, var)
        tm = TM.TimingSettings.from_dict(getattr(p.analysis, 'timing', None))
        conv = DL.stamp_offset_ms(st.camera_stamp, tm.readout_ms, tm.exposure_ms)
        lid = var.sensor(plan.lidar.uid)
        motion = DL.Motion.from_scan(p, var, scan, mapping)
        for c in [c for c in var.cameras if c.visible and c.tag in rev]:
            prev = None
            for k0 in range(0, len(scan.targets) - 3, 4):
                sub = dataclasses.replace(scan, targets=scan.targets[k0:k0 + 4])
                r = DL._extrinsic(p, var, c, rev[c.tag], lid, rev[lid.tag], sub, motion, st, conv, eng, {})
                ok = r.get('delta_deg') is not None and r['confidence'] in ('high', 'medium')
                if ok:
                    ses += [s for s, o in zip(r['stderr_deg'], r['observable']) if o]
                if ok and prev is not None:
                    for k in range(3):
                        if r['observable'][k] and prev['observable'][k]:
                            zw.append((r['delta_deg'][k] - prev['delta_deg'][k])
                                      / math.hypot(r['stderr_deg'][k], prev['stderr_deg'][k]))
                prev = r if ok else None
    zw, zt = np.abs(zw), np.abs(zt)
    assert len(zw) >= 30
    assert 0.6 <= np.mean(zw < 1) <= 0.8 and np.mean(zw < 2) >= 0.88, (np.mean(zw < 1), np.mean(zw < 2))
    if len(zt) >= 10:
        assert np.mean(zt < 2) >= 0.8, np.mean(zt < 2)
    assert float(np.median(ses)) < 0.1
