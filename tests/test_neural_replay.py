"""Tests for high-fidelity neural log replay (core.neural_replay, the Unscented Transform in core.splat_raster, and its
hooks in core.splat_reproject and core.redrive).

References: the UT against a Monte Carlo of the Gaussian pushed through the same fisheye; the 3D response against a
brute-force search along each pixel's ray; the closed form on an equidistant lens's axis; gradients against finite
differences (gradcheck, double precision); the rolling shutter against the exact row-time solution of a camera moving
sideways; the two opacities on a glass pane in front of a wall; the skeleton against Winter's proportions and against
a synthetic walk with known joint angles; the traffic light's lamps over time against the stated states. A 1 s
synthetic drive of the sample's data rig (core.dataset, images at 15 %) runs the study, the log replay and the
re-drive export on the CPU.
Run with:  QT_QPA_PLATFORM=offscreen python3 -m pytest tests/test_neural_replay.py -q
"""
from __future__ import annotations

import json
import math
import os
import sys

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np  # noqa: E402
import pytest  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch = pytest.importorskip('torch')
cv2 = pytest.importorskip('cv2')

from vantage.core import dataset as DSET  # noqa: E402
from vantage.core import neural_replay as NR  # noqa: E402
from vantage.core import splat_raster as SR  # noqa: E402
from vantage.core import splat_reproject as SP  # noqa: E402
from vantage.core.analysis import Engine  # noqa: E402
from vantage.core.mathutil import BODY_TO_OPTICAL  # noqa: E402
from vantage.core.samples import tigor_sample_project  # noqa: E402

from test_gui import app, win, pump, _wait  # noqa: E402,F401  (fixtures)

D64 = torch.float64
CPU = torch.device('cpu')


# ----------------------------------------------------------------------------- settings
def test_settings_are_tolerant():
    s = NR.NeuralSettings.from_dict({'projection': 'pinhole', 'walkers': 'rigid', 'lidar_steps': -5, 'seed': 'x',
                                     'fit_steps': 1e9, 'relight': {'condition': 'snow', 'strength': 9},
                                     'relight_conditions': ['night', 'fog', 'night', 'none'], 'smpl_accept': 'yes',
                                     'gpu_mem_gb': 40})
    assert s.projection == 'ut' and s.walkers == 'rigid' and s.lidar_steps == 0 and s.seed == 0
    assert s.fit_steps == 5000 and s.relight['condition'] == 'snow' and s.relight['strength'] == 1.0
    assert s.relight_conditions == ['night'] and s.smpl_accept is False and s.gpu_mem_gb == 5.5
    assert NR.NeuralSettings.from_dict(None) == NR.NeuralSettings()
    assert NR.NeuralSettings.from_dict(s.to_dict()) == s
    p = tigor_sample_project()
    assert NR.settings_of(p) == NR.NeuralSettings() and SP.projection_of(p) == 'ut'
    p.analysis.neural_replay = {'projection': 'ewa'}
    assert SP.projection_of(p) == 'ewa' and SP.projection_of(None) == 'ewa'


# ----------------------------------------------------------------------------- the Unscented Transform
def test_ut_against_a_monte_carlo_through_a_fisheye():
    """The UT's projected mean and covariance against 400 000 samples pushed through the CARLA drives' fisheye: the
    mean within 0.5 px where the affine projection is off by more than a pixel."""
    r = NR.ut_vs_monte_carlo([1.5, 0.3, 1.2], [0.3, 0.2, 0.25])
    assert r['samples_in_field'] > 399000
    assert r['ut_mean_err_px'] < 0.5 < 1.0 < r['ewa_mean_err_px']
    assert r['ut_cov_rel_err'] < 0.15
    on_axis = NR.ut_vs_monte_carlo([0.0, 0.0, 10.0], [0.1, 0.1, 0.1])     # nearly linear: both agree
    assert on_axis['ut_mean_err_px'] < 0.05 and on_axis['ewa_mean_err_px'] < 0.05
    assert NR.check_ut_monte_carlo()['passed']


def test_ut_weights_are_3dguts():
    wm, wc, c = SR.ut_weights()
    assert wm[0] == 0.0 and wc[0] == pytest.approx(2.0) and all(w == pytest.approx(1 / 6) for w in wm[1:])
    assert c == pytest.approx(math.sqrt(3.0)) and sum(wm) == pytest.approx(1.0)


def test_ut_response_is_the_peak_along_each_ray():
    """Every pixel's alpha is the opacity times the Gaussian's maximum along that pixel's ray (no dilation here),
    found independently by a dense search along the ray, for a large rotated Gaussian on a fisheye's rim."""
    m = SP.CamModel(48, 40, 16.0, 16.0, 24.0, 20.0, 'kannala_brandt', (0.007, 0.132, -0.193, 0.057),
                    math.radians(87.0))
    P = np.array([[1.6, 0.2, 0.3]])
    R = SR.quat_to_rot_np(np.array([[0.9, 0.2, -0.3, 0.25]]))
    s = np.array([[0.35, 0.12, 0.05]])
    op = 0.8
    D, okr = SR.projector_rays(m.projector(), 48, 40, dtype=D64)
    out = SR.splat_ut(torch.tensor(P, dtype=D64), torch.tensor(R, dtype=D64), torch.tensor(s, dtype=D64),
                      torch.ones(1, 1, dtype=D64), torch.tensor([op], dtype=D64), m.projector(), 48, 40, D,
                      lowpass=0.0)
    a = out['alpha'].numpy()
    Sinv = np.linalg.inv(R[0] @ np.diag(s[0] ** 2) @ R[0].T)
    t = np.linspace(0.05, 6.0, 20001)
    Dn = D.numpy()
    ref = np.zeros(len(Dn))
    for k in range(len(Dn)):
        X = t[:, None] * Dn[k] - P[0]
        q = np.einsum('ni,ij,nj->n', X, Sinv, X)
        ref[k] = op * math.exp(-0.5 * q.min())
    ref = np.where(ref >= SR.ALPHA_MIN, np.minimum(ref, SR.ALPHA_MAX), 0.0)
    good = okr.numpy() & ((ref > 0.01) | (a > 0.01))
    assert good.sum() > 30
    assert np.abs(a[good] - ref[good]).max() < 2e-3


def test_ut_closed_form_on_the_axis():
    assert NR.check_ut_axis()['passed']


def test_ut_gradients_match_finite_differences():
    m = SP.CamModel(24, 20, 12.0, 12.0, 12.0, 10.0, 'kannala_brandt', (0.01, 0.0, 0.0, 0.0), math.radians(100.0))
    proj = m.projector()
    D, _ = SR.projector_rays(proj, 24, 20, dtype=D64)
    P0 = torch.tensor([[0.3, -0.1, 2.0], [-0.4, 0.2, 3.0]], dtype=D64)
    R = SR.quat_to_rot(torch.tensor([[1.0, 0.1, 0.0, 0.0], [0.9, 0.0, 0.2, 0.1]], dtype=D64))
    f = torch.tensor([[1.0, 0.2], [0.1, 0.7]], dtype=D64)

    def fn(P, s, op):       # no dilation: it is a filter of the pixel size at the centre, not differentiated
        return SR.splat_ut(P, R, s, f, op, proj, 24, 20, D, lowpass=0.0)['image']
    args = (P0.clone().requires_grad_(True), torch.tensor([[0.2, 0.15, 0.1], [0.3, 0.2, 0.25]], dtype=D64,
                                                          requires_grad=True),
            torch.tensor([0.7, 0.5], dtype=D64, requires_grad=True))
    assert torch.autograd.gradcheck(fn, args, eps=1e-6, atol=1e-5, rtol=1e-3)


def _one_gaussian_scene(P, sigma=0.05, colour=(1.0, 1.0, 1.0)):
    P = np.asarray(P, np.float32).reshape(-1, 3)
    n = len(P)
    return SP.Scene(P, np.full((n, 3), sigma, np.float32), np.tile([1, 0, 0, 0], (n, 1)).astype(np.float32),
                    np.full(n, 0.95, np.float32), np.tile(colour, (n, 1)).astype(np.float32),
                    np.full(n, -1, np.int32), np.zeros(n, np.int32), [], np.zeros((SP.SKY_H, SP.SKY_W, 3), np.float32),
                    np.zeros(3), 'lidar', {}, 'ut')


def test_rolling_shutter_is_exact_per_row():
    """A camera moving sideways at 20 m/s with a 30 ms readout: a point 5 m ahead at image row v₀ appears shifted by
    f·v·(v₀/h − ½)·readout / z (the camera at the instant that row is read), and a vertical line of points slants by
    the same law row by row."""
    w, h, f = 96, 80, 60.0
    m = SP.CamModel(w, h, f, f, w / 2, h / 2, 'rectilinear', (0, 0, 0, 0), math.radians(80))
    v_e, ro, z = 20.0, 0.03, 5.0
    traj = SP.Trajectory([-1.0, 1.0], [[0.0, -v_e, 0.0], [0.0, v_e, 0.0]], [[1, 0, 0, 0], [1, 0, 0, 0]])
    R_cv = BODY_TO_OPTICAL.T                      # optical -> vehicle, looking along +x
    t_cv = np.zeros(3)
    zs = np.linspace(-1.2, 1.2, 9)
    sc = _one_gaussian_scene([[z, 0.0, zz] for zz in zs], sigma=0.01)
    ts = SP.TorchScene(sc, None, CPU)
    out = SP.render_view_ut(ts, traj, m, R_cv, t_cv, 0.0, ro)
    al = out['alpha'].numpy()
    for zz in zs:
        v0 = h / 2 - f * zz / z                                         # its row (no vertical motion)
        tr = (v0 / h - 0.5) * ro
        u_exp = w / 2 + f * (v_e * tr) / z                              # the camera moved left by v·t: the point right
        r = int(v0)
        row = al[max(r - 1, 0):r + 2].sum(0)
        u_got = float((row * (np.arange(w) + 0.5)).sum() / row.sum())
        assert u_got == pytest.approx(u_exp, abs=0.15)
    still = SP.render_view_ut(ts, traj, m, R_cv, t_cv, 0.0, 0.0)['alpha'].numpy()
    assert abs(float((still.sum(0) * (np.arange(w) + 0.5)).sum() / still.sum()) - w / 2) < 0.1


def test_lidar_beams_range_exactly_through_the_ut():
    """A wall 10 m ahead: the UT lidar's ranges are the exact distances along each beam (1 / cos of its angles)."""
    yy, zz = np.meshgrid(np.arange(-6, 6.001, 0.05), np.arange(-1.5, 3.5, 0.05))
    P = np.stack([np.full(yy.size, 10.0), yy.ravel(), zz.ravel()], 1)
    sc = _one_gaussian_scene(P, sigma=0.04)
    sc.scales[:, 0] = 0.004
    ts = SP.TorchScene(sc, None, CPU)
    el = np.linspace(-8, 8, 9)
    C = 256
    k = np.arange(C)
    spec = SP.LidarSpec('L', 'L', np.eye(3), np.array([0.0, 0.0, 1.0]), el, (k * 360.0 / C + 180) % 360 - 180,
                        k / C, 0.1, True, 0.5, 120.0)
    traj = SP.Trajectory([0.0], [[0, 0, 0]], [[1, 0, 0, 0]])
    sw = SP.render_lidar(ts, traj, spec, 0.0, SP.SplatSettings())
    rng = sw['range']
    E, A = np.meshgrid(np.radians(el), np.radians(spec.az), indexing='ij')
    exact = 10.0 / (np.cos(E) * np.cos(A))
    hit = np.isfinite(rng) & (np.abs(np.degrees(A)) < 25)
    assert hit.sum() > 60
    assert np.abs(rng[hit] - exact[hit]).max() < 0.02


# ----------------------------------------------------------------------------- two opacities
def test_two_opacities_glass_is_seen_through_by_the_camera_and_hit_by_the_lidar():
    yy, zz = np.meshgrid(np.arange(-5, 5.001, 0.06), np.arange(-1.0, 3.0, 0.06))
    wall = np.stack([np.full(yy.size, 10.0), yy.ravel(), zz.ravel()], 1)
    glass = np.stack([np.full(yy.size, 5.0), yy.ravel(), zz.ravel()], 1)
    sc = _one_gaussian_scene(np.concatenate([wall, glass]), sigma=0.05)
    sc.scales[:, 0] = 0.005
    n = len(wall)
    sc.opacity[n:] = 0.003                                          # the camera sees through the glass
    sc.colours[:n] = [0.2, 0.8, 0.2]
    op_l = np.full(sc.n, 0.95, np.float32)                         # the lidar returns from it
    sc.neural = NR.NeuralExt([], np.full(sc.n, -1, np.int16), np.full(sc.n, -1, np.int16), [],
                             np.full(sc.n, -1, np.int8), np.full(sc.n, -1, np.int32), op_l, {})
    ts = SP.TorchScene(sc, None, CPU)
    m = SP.CamModel(64, 48, 40.0, 40.0, 32.0, 24.0, 'rectilinear', (0, 0, 0, 0), math.radians(80))
    traj = SP.Trajectory([0.0], [[0, 0, 1.0]], [[1, 0, 0, 0]])
    r = SP.render_view(ts, traj, m, BODY_TO_OPTICAL.T, np.zeros(3), 0.0, depth=True)
    c = r['depth'].numpy()[24, 32]
    assert c == pytest.approx(10.0, abs=0.2) and r['rgb'].numpy()[24, 32, 1] > 0.6     # the wall behind, green
    el = np.linspace(-4, 4, 5)
    C = 128
    k = np.arange(C)
    spec = SP.LidarSpec('L', 'L', np.eye(3), np.zeros(3), el, (k * 360.0 / C + 180) % 360 - 180, k / C, 0.1, True,
                        0.5, 120.0)
    st = SP.SplatSettings()
    front = np.abs(spec.az) < 10
    sw = SP.render_lidar(ts, traj, spec, 0.0, st)
    assert np.nanmedian(sw['range'][:, front]) == pytest.approx(5.0, abs=0.1)     # the glass for the lidar
    ts.op_lidar_logit = None                                                        # one opacity: the camera's
    sw1 = SP.render_lidar(ts, traj, spec, 0.0, st)
    assert np.nanmedian(sw1['range'][:, front]) == pytest.approx(10.0, abs=0.2)


# ----------------------------------------------------------------------------- the body
def test_skeleton_proportions_are_winters_and_bones_keep_their_length():
    H, hb = 1.8, 1.8
    Rs, os_, kp = NR.forward_kinematics(torch.zeros(1, NR.NDOF, dtype=D64), H, hb)
    k = {n: kp[0, i].numpy() + np.array([0, 0, hb / 2]) for i, n in enumerate(NR.KEYPOINTS)}
    W = NR.WINTER
    assert k['l_shoulder'][2] == pytest.approx(W['shoulder'] * H, abs=1e-9)
    assert k['l_elbow'][2] == pytest.approx(W['elbow'] * H, abs=1e-9)
    assert k['l_wrist'][2] == pytest.approx(W['wrist'] * H, abs=1e-9)
    assert k['l_knee'][2] == pytest.approx(W['knee'] * H, abs=1e-9)
    assert k['l_ankle'][2] == pytest.approx(W['ankle'] * H, abs=1e-9)
    assert k['l_shoulder'][1] - k['r_shoulder'][1] == pytest.approx(W['shoulder_w'] * H, abs=1e-9)
    assert k['l_eye'][2] == pytest.approx(W['eye'] * H, abs=1e-9)
    rng = np.random.default_rng(3)
    th = torch.tensor(rng.uniform(NR.DOF_LO, NR.DOF_HI, (5, NR.NDOF)), dtype=D64)
    _, _, kp2 = NR.forward_kinematics(th, H, hb)
    for a, b in (('l_shoulder', 'l_elbow'), ('l_elbow', 'l_wrist'), ('r_hip', 'r_knee'), ('r_knee', 'r_ankle')):
        i, j = NR.KEYPOINTS.index(a), NR.KEYPOINTS.index(b)
        L0 = np.linalg.norm(kp[0, i].numpy() - kp[0, j].numpy())
        L = np.linalg.norm(kp2[:, i].numpy() - kp2[:, j].numpy(), axis=1)
        assert np.allclose(L, L0, atol=1e-9)
    # flexion moves a limb forward, a knee's backward
    th = torch.zeros(1, NR.NDOF, dtype=D64)
    th[0, NR.D_['l_hip_flex']] = 0.5
    th[0, NR.D_['r_knee']] = 0.8
    _, _, kp3 = NR.forward_kinematics(th, H, hb)
    assert kp3[0, NR.KEYPOINTS.index('l_knee'), 0] > 0.1
    assert kp3[0, NR.KEYPOINTS.index('r_ankle'), 0] < -0.1


def test_gait_prior_walks_and_stands():
    t = np.arange(0, 3, 0.1)
    walk = NR.gait_prior(t, np.stack([1.4 * t, 0 * t, 0 * t], 1), 0 * t, 1.8)
    stand = NR.gait_prior(t, np.zeros((len(t), 3)), 0 * t, 1.8)
    l_, r_ = walk[:, NR.D_['l_hip_flex']], walk[:, NR.D_['r_hip_flex']]
    assert np.degrees(l_.max() - l_.min()) > 30 and np.corrcoef(l_, r_)[0, 1] < -0.9      # half a cycle apart
    assert np.corrcoef(l_, walk[:, NR.D_['l_sh_flex']])[0, 1] < -0.9                     # the arm swings against
    assert np.degrees(np.abs(stand[:, NR.D_['l_hip_flex']]).max()) < 1e-9
    # the cycle's length: stride 0.83 H at 1.4 m/s
    x = (l_ - l_.mean())
    crossings = np.flatnonzero((x[:-1] < 0) & (x[1:] >= 0))
    if len(crossings) >= 2:
        assert 1.4 * 0.1 * np.mean(np.diff(crossings)) == pytest.approx(0.83 * 1.8, rel=0.1)


def test_skeleton_fit_recovers_a_synthetic_walk():
    r = NR.check_skeleton_fit()
    assert r['passed'] and r['median_joint_error_deg'] < r['prior_median_joint_error_deg']
    assert r['stats']['median_px_fit'] < 2.5 < r['stats']['median_px_prior']


def test_body_gaussians_cover_the_segments():
    seg, loc, sc, qs = NR.body_gaussians(1.75, 1500)
    assert 1300 < len(seg) < 1800 and set(np.unique(seg)) == set(range(NR.NSEG))
    R = SR.quat_to_rot_np(qs)
    assert np.allclose(np.linalg.det(R), 1.0, atol=1e-5) and (sc[:, 2] < sc[:, 0]).all()


def test_smpl_is_opt_in_and_binds_to_segments(tmp_path):
    st = NR.NeuralSettings(body_model='smpl')
    assert 'non-commercial' in NR.smpl_usable(st)
    st.smpl_accept = True
    assert 'Choose' in NR.smpl_usable(st)
    # an SMPL-format template (not SMPL itself): points around the joints, each weighted to one SMPL joint
    _, os_, _ = NR.forward_kinematics(torch.zeros(1, NR.NDOF, dtype=D64), 1.7, 1.7)
    rng = np.random.default_rng(0)
    seg_of = {v: k for k, v in enumerate(NR.SEGMENTS)}
    vs, ws = [], []
    for j, name in NR.SMPL_TO_SEG.items():
        c = os_[0, seg_of[name]].numpy() + np.array([0, 0, 0.85])
        p = c + rng.normal(0, 0.03, (20, 3))
        vs.append(p[:, [1, 2, 0]])                       # SMPL frame: x left, y up, z forward
        w = np.zeros((20, 24))
        w[:, j] = 1.0
        ws.append(w)
    v = np.concatenate(vs)
    f = np.stack([np.arange(len(v) - 2), np.arange(1, len(v) - 1), np.arange(2, len(v))], 1)
    path = str(tmp_path / 'smpl_like.npz')
    np.savez(path, v_template=v, weights=np.concatenate(ws), f=f)
    st.smpl_path = path
    assert NR.smpl_usable(st) is None
    seg, loc, sc, qs = NR.smpl_gaussians(NR.load_smpl(path), 1.7, 10000)
    assert 0.8 * len(v) <= len(seg) <= len(v)                 # one per cube of the surface spacing
    assert set(np.unique(seg)) == {seg_of[n] for n in NR.SMPL_TO_SEG.values()}
    # each Gaussian sits on its own segment's rest pose: near that segment's joint (the points' spread is 3 cm)
    Rs, os2, _ = NR.forward_kinematics(torch.zeros(1, NR.NDOF, dtype=D64), 1.7, 1.7)
    P = np.einsum('nij,nj->ni', Rs[0].numpy()[seg], loc) + os2[0].numpy()[seg]
    assert np.median(np.linalg.norm(P - os2[0].numpy()[seg], axis=1)) < 0.25
    np.savez(str(tmp_path / 'bad.npz'), v_template=v)
    st.smpl_path = str(tmp_path / 'bad.npz')
    assert 'cannot be read' in NR.smpl_usable(st)


def test_anny_bones_map_to_segments_and_a_wrong_download_is_refused(tmp_path, monkeypatch):
    m = NR._anny_segment
    assert m('upperarm01.L') == 'l_upperarm' and m('lowerarm02.R') == 'r_forearm' and m('finger3-2.L') == 'l_forearm'
    assert m('upperleg02.R') == 'r_thigh' and m('lowerleg01.L') == 'l_shank' and m('toe1-1.R') == 'r_foot'
    assert m('head') == 'head' and m('neck02') == 'head' and m('spine03') == 'torso' and m('root') == 'pelvis'
    assert m('clavicle.L') == 'torso'

    class R:
        def __init__(self, n):
            self.data, self.k = b'v 0 0 0\n' * (n // 8 + 1), 0

        def read(self, n):
            b = self.data[self.k:self.k + n]
            self.k += n
            return b

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False
    with pytest.raises(ValueError):
        NR.download_anny(folder=str(tmp_path), fetch=lambda url, timeout: R(NR.ANNY_FILES[0][2]))
    assert not os.listdir(os.path.join(str(tmp_path), 'anny_base'))
    # a tiny template in the same layout: the body group, triangulated, weights grouped into segments
    d = tmp_path / 'anny_base'
    (d / 'base.obj').write_text('v 0 0 0\nv 1 0 0\nv 0 1 0\nv 1 1 0\nv 5 5 5\ng helper-skirt\nf 5 5 5\n'
                                'g body\nf 1 2 4 3\n')
    (d / 'weights.default.json').write_text(json.dumps({'weights': {'upperleg01.L': [[0, 1.0], [1, 1.0]],
                                                                    'head': [[2, 1.0], [3, 0.5]]}}))
    monkeypatch.setattr(NR, 'anny_status', lambda folder=None, verify=True: {'verified': True, 'path': str(d)})
    mdl = NR.load_anny()
    assert mdl['v'].shape == (4, 3) and mdl['f'].tolist() == [[0, 1, 3], [0, 3, 2]]
    assert mdl['segw'].argmax(1).tolist() == [NR.SEGMENTS.index('l_thigh')] * 2 + [NR.SEGMENTS.index('head')] * 2


# ----------------------------------------------------------------------------- traffic lights
def _light(states=('red', 'amber', 'green'), times=(0.0, 1.0, 2.0)):
    return NR.Light(5000, 'a1', np.array([10.0, 0.0, 4.0]), np.eye(3), (0.37, 0.40, 1.19),
                    np.array([-1.0, 0.0, 0.0]), np.array([0.3, 0.3, 0.1]), {}, np.asarray(times, float),
                    list(states), 'ground truth')


def test_lamp_colours_follow_the_state_over_time():
    L = _light()
    for t, k in ((0.1, 0), (0.9, 1), (2.4, 2)):
        c = NR.lamp_colours(L, t)
        assert np.allclose(c[k], NR.STATE_RGB[NR.STATES[k]])
        assert all(c[j].max() < 0.2 for j in range(3) if j != k)
    lc = NR.lamp_centres(L)
    assert lc[0][2] > lc[1][2] > lc[2][2] and lc[0][0] < 10.0          # top to bottom, on the face towards −x
    unknown = _light((), ())
    assert NR.lamp_colours(unknown, 1.0).max() < 0.2


def test_emissive_plane_renders_the_state():
    L = _light()
    g = NR.light_gaussians([L])
    sc = _one_gaussian_scene(np.zeros((0, 3)))
    sc, ex = NR._append(sc, [g])
    sc.neural = NR.NeuralExt([], ex['wid'], ex['wseg'], [L], ex['lamp'], ex['light'], None, {})
    ts = SP.TorchScene(sc, None, CPU)
    w, h, f = 96, 160, 220.0
    m = SP.CamModel(w, h, f, f, w / 2, h / 2, 'rectilinear', (0, 0, 0, 0), math.radians(80))
    traj = SP.Trajectory([0.0], [[0, 0, 4.0]], [[1, 0, 0, 0]])
    lc = NR.lamp_centres(L)
    seen = {}
    for proj in ('ewa', 'ut'):
        ts.projection = proj
        for t in (0.0, 1.0, 2.0):
            img = SP.render_view(ts, traj, m, BODY_TO_OPTICAL.T, np.zeros(3), t)['rgb'].numpy()
            vals = []
            for c in lc:
                d = c - np.array([0, 0, 4.0])
                u = w / 2 - f * d[1] / d[0]
                v = h / 2 - f * d[2] / d[0]
                vals.append(img[int(v), int(u)])
            vals = np.array(vals)
            k = int(np.argmax(vals.sum(1)))
            seen[(proj, t)] = k
            assert vals[k].sum() > 2 * np.sort(vals.sum(1))[-2]
    assert seen == {('ewa', 0.0): 0, ('ewa', 1.0): 1, ('ewa', 2.0): 2, ('ut', 0.0): 0, ('ut', 1.0): 1,
                    ('ut', 2.0): 2}


def test_lamp_reading_picks_the_lit_lamp():
    img = np.full((60, 30, 3), 0.25)
    for j, y in enumerate((10, 30, 50)):
        img[y - 4:y + 5, 11:20] = (0.12, 0.12, 0.1) if j != 2 else (0.3, 0.95, 0.5)
    sc = [NR._lamp_score(img, 15, y, 4)[0] for y in (10, 30, 50)]
    assert int(np.argmax(sc)) == 2 and sc[2] - sorted(sc)[1] > 0.3


# ----------------------------------------------------------------------------- a synthetic drive
@pytest.fixture(scope='module')
def drive(tmp_path_factory):
    p = tigor_sample_project()
    v = next(v for v in p.variants if v.sensors)
    p.active_uid = v.uid
    engine = Engine()
    out = str(tmp_path_factory.mktemp('drive'))
    assert DSET.generate(p, engine, DSET.DatasetSettings.from_dict({'duration_s': 1.0, 'image_scale': 0.15}),
                         out) is not None
    cache = str(tmp_path_factory.mktemp('cache'))
    p.analysis.splat = {'drive': out, 'device': 'cpu', 'max_gaussians': 6000, 'voxel_m': 0.1, 'iterations': 4,
                        'train_scale': 0.25, 'holdout_every': 4, 'eval_instants': 1, 'render_scale': 0.12,
                        'colour_step': 3, 'time_budget_s': 60, 'val_frames': 6, 'val_sweeps': 1}
    p.analysis.neural_replay = {'lidar_steps': 8, 'fit_steps': 10, 'eval_instants': 2,
                                'relight_conditions': ['night'], 'walker_gaussians': 400}
    d = SP.open_drive(out)
    sst = SP.settings_of(p)
    base = SP.build_scene(d, sst, p, engine, cache_root=cache, neural=False)
    return {'p': p, 'engine': engine, 'out': out, 'cache': cache, 'drive': d, 'base': base}


class _FakePose:
    """A stand-in keypoint model: the gait prior's keypoints of the walker in the box, through the camera."""
    calls = 0

    def keypoints(self, img, boxes):
        _FakePose.calls += 1
        out = np.zeros((len(boxes), 17, 3))
        for i, b in enumerate(boxes):
            x0, y0, x1, y1 = b
            ys = np.linspace(y0 + 0.1 * (y1 - y0), y1 - 0.05 * (y1 - y0), 17)
            out[i, :, 0] = (x0 + x1) / 2
            out[i, :, 1] = ys
            out[i, :, 2] = 0.9
        return out

    def describe(self):
        return {'id': 'fake'}


def test_walkers_replace_the_rigid_pedestrians_and_articulate(drive):
    d, base, p = drive['drive'], drive['base'], drive['p']
    peds = NR.pedestrian_tracks(d, base)
    assert peds, 'the synthetic drive has walking pedestrians'
    sst = SP.settings_of(p)
    nst = NR.NeuralSettings.from_dict(dict(p.analysis.neural_replay, traffic_lights=False, lidar_opacity=False))
    sc = NR.augment(base, d, sst, p, drive['engine'], folder=None, settings=nst)
    ext = sc.neural
    assert ext is not None and len(ext.walkers) == len(peds) and ext.info['walkers']['count'] == len(peds)
    for i, slot in peds:
        assert (sc.actor == slot).any() and bool((ext.wid[sc.actor == slot] >= 0).all())   # all now articulated
    ts = SP.TorchScene(sc, d, CPU)
    w = ext.walkers[0]
    tr = d.tracks[w.track]
    rows = np.flatnonzero(ext.wid == 0)
    t0, t1 = float(tr.t[0]), float(tr.t[-1])
    P0 = ts.posed(t0)[0].detach().numpy()[rows]
    P1 = ts.posed(t1)[0].detach().numpy()[rows]
    for t, P in ((t0, P0), (t1, P1)):
        Rb, cb = tr.pose(t)
        q = (P - cb) @ Rb
        assert np.all(np.abs(q) <= np.asarray(tr.size) / 2 + np.array([0.6, 0.6, 0.15]))      # in (about) its box
    # the limbs move relative to the box between the two instants (the gait), the head stays on top
    q0 = (P0 - tr.pose(t0)[1]) @ tr.pose(t0)[0]
    q1 = (P1 - tr.pose(t1)[1]) @ tr.pose(t1)[0]
    if np.linalg.norm(tr.centre[-1] - tr.centre[0]) > 0.3:
        assert np.abs(q1 - q0).max() > 0.02
    head = ext.wseg[rows] == NR.SEGMENTS.index('head')
    assert q0[head, 2].mean() > tr.size[2] / 2 - 0.35


def test_walkers_fit_keypoints_when_a_pose_model_is_given(drive):
    d, base, p = drive['drive'], drive['base'], drive['p']
    sst = SP.settings_of(p)
    nst = NR.NeuralSettings.from_dict(dict(p.analysis.neural_replay, traffic_lights=False, lidar_opacity=False,
                                           min_walker_px=10))
    _FakePose.calls = 0
    sc = NR.augment(base, d, sst, p, drive['engine'], folder=None, settings=nst, pose_model=_FakePose())
    assert _FakePose.calls > 0
    fits = sc.neural.info['walkers']['fits']
    assert any(f['method'] == 'keypoints' and f['observations'] > 0 for f in fits)


def test_traffic_lights_from_a_detector_when_the_drive_has_no_boxes(drive):
    """A stand-in detector reporting a 'traffic light' in the upper middle of every image: each detection is lifted
    to the reconstruction's surface along its ray, the detections of a camera merge into heads, and the heads become
    emissive lights that are drawn but not scored (their ids are not the drive's)."""
    from vantage.core import detection as DT
    d, base = drive['drive'], drive['base']

    def det(img):
        h, w = img.shape[:2]
        return {'boxes': np.array([[w / 2 - 20, 0.2 * h - 20, w / 2 + 20, 0.2 * h + 20]], float),
                'scores': np.array([0.9]),
                'cls': np.array([DT.CLASSES.index('traffic_light')]), 'coco': np.array([9])}
    heads = NR.lights_from_detections(d, base, det, NR.NeuralSettings(), step=1)
    assert heads and all(i >= NR.DETECTED_ID0 and i not in d.tracks for i in heads)
    from scipy.spatial import cKDTree
    tree = cKDTree(base.means[base.actor < 0])
    for tr in heads.values():
        assert tr.cls == 'traffic_light' and not tr.dynamic
        assert tree.query(tr.centre[0])[0] < 1.5                 # on the rebuilt surface (within the merge radius)
    lights, info = NR.build_lights(base, d, NR.NeuralSettings(), tracks=heads)
    assert info['count'] == len(heads)
    p = drive['p']
    sst = SP.settings_of(p)
    nst = NR.NeuralSettings.from_dict(dict(p.analysis.neural_replay, walkers='rigid', lidar_opacity=False))
    sc = NR.augment(base, d, sst, p, folder=None, settings=nst, light_detector=det)
    assert sc.neural.info['lights']['heads_from'] == 'detector' and sc.neural.info['lights']['count'] == len(heads)


def test_extension_is_cached_and_round_trips(drive, tmp_path):
    d, base, p = drive['drive'], drive['base'], drive['p']
    sst = SP.settings_of(p)
    nst = NR.settings_of(p)
    folder = str(tmp_path / 'scene')
    a = NR.augment(base, d, sst, p, drive['engine'], folder=folder, settings=nst)
    assert a.neural.info['cached'] is False and a.neural.opacity_lidar is not None
    b = NR.augment(base, d, sst, p, drive['engine'], folder=folder, settings=nst)
    assert b.neural.info.get('cached') is True and b.n == a.n
    assert np.array_equal(b.neural.wid, a.neural.wid) and np.allclose(b.neural.opacity_lidar, a.neural.opacity_lidar)
    assert [w.track for w in b.neural.walkers] == [w.track for w in a.neural.walkers]
    assert np.allclose(b.neural.walkers[0].theta, a.neural.walkers[0].theta)


def test_lidar_opacity_fit_learns_where_the_lidar_returns(drive):
    """Gaussians the camera sees through (opacity 0.02) but the recorded beams hit: the lidar opacity rises."""
    d, base, p = drive['drive'], drive['base'], drive['p']
    sst = SP.settings_of(p)
    sc = SP.Scene(*(getattr(base, k).copy() for k in ('means', 'scales', 'quats', 'opacity', 'colours', 'actor',
                                                       'obj')), list(base.actor_ids), base.sky, base.origin)
    rng = np.random.default_rng(0)
    pick = rng.random(sc.n) < 0.3
    sc.opacity[pick] = 0.02
    ts = SP.TorchScene(sc, d, CPU)
    op, stats = NR.fit_lidar_opacity(ts, d, sst, NR.NeuralSettings(lidar_steps=40, lidar_lr=0.2))
    assert stats['steps'] == 40 and stats['loss_last'] < stats['loss_first']
    assert float(op[pick].mean()) > 0.02 + 0.05
    assert float(op[~pick].mean()) > 0.5                     # the rest stays where the camera had it


def test_study_runs_and_exports(drive, tmp_path):
    p = drive['p']
    det = _grid_detector
    res = NR.run(p, drive['engine'], detector=det, cache_root=drive['cache'], drive=drive['drive'])
    assert res is not None
    assert set(res.projection) == {'ewa', 'ut'} and all(r['psnr_db'] for r in res.projection.values())
    assert res.lidar['camera_opacity'] and res.lidar['lidar_opacity']
    pc = res.perception
    assert pc['rigid'] and pc['neural'] and pc['detected']
    assert 'night' in res.relight and res.relight['night']['label'].startswith('relit')
    doc = NR.to_doc(res, p)
    json.dumps(doc, default=float)
    assert doc['schema'] == NR.SCHEMA and doc['relit_not_recorded'] is True
    s = NR.summary(res)
    assert s['walkers'] and s['psnr_ut_db'] is not None
    paths = NR.export(res, p, str(tmp_path / 'exp'))
    names = {os.path.basename(x) for x in paths}
    assert {'neural_replay.json', 'neural_replay_perception.csv', 'neural_replay_projection.csv'} <= names
    assert any(f['kind'] == 'projection' for f in res.findings)


def _grid_detector(img):
    h, w = img.shape[:2]
    boxes, scores = [], []
    for i in range(4):
        for j in range(4):
            x0, x1, y0, y1 = j * w // 4, (j + 1) * w // 4, i * h // 4, (i + 1) * h // 4
            boxes.append([x0, y0, x1, y1])
            scores.append(float(img[y0:y1, x0:x1].mean()) / 255.0)
    return {'boxes': np.array(boxes, float), 'scores': np.array(scores), 'cls': np.full(16, 3),
            'coco': np.full(16, 2)}


def test_log_replay_and_redrive_carry_the_extension(drive, tmp_path):
    """The log replay's run uses the extension (findings, the scene's doc), relit renders say so, and a re-drive
    export records it."""
    from vantage.core import redrive as RD
    p = drive['p']
    p.analysis.neural_replay = dict(p.analysis.neural_replay, relight={'condition': 'wet'})
    try:
        res = SP.run(p, drive['engine'], detector=_grid_detector, cache_root=drive['cache'], drive=drive['drive'])
        assert res.scene['projection'] == 'ut' and res.scene['neural']['walkers']['count'] > 0
        assert any('articulated' in f['message'] for f in res.findings)
        doc = SP.to_doc(res, p)
        assert doc['neural']['relight']['condition'] == 'wet'
        nb = RD.neural_block(res._scene)
        assert nb['relit'].startswith('relit to wet') and nb['projection'] == 'ut'
    finally:
        p.analysis.neural_replay = {k: v for k, v in p.analysis.neural_replay.items() if k != 'relight'}


def test_rigid_setting_keeps_the_log_replay(drive):
    p = drive['p']
    old = dict(p.analysis.neural_replay)
    p.analysis.neural_replay = {'projection': 'ewa', 'walkers': 'rigid', 'traffic_lights': False,
                                'lidar_opacity': False}
    try:
        sc = SP.build_scene(drive['drive'], SP.settings_of(p), p, drive['engine'], cache_root=drive['cache'])
        assert sc.neural is None and sc.projection == 'ewa'
    finally:
        p.analysis.neural_replay = old


def test_validation_checks_are_specs():
    from vantage.validation.suite import Spec
    for c in NR.validation_checks():
        sp = Spec(**c)
        if sp.cost == 'cheap':
            out = sp.fn(None)
            assert out.ok and out.error <= sp.tolerance


def test_no_unseeded_randomness():
    for mod in (NR, SR):
        src = open(mod.__file__, encoding='utf-8').read()
        assert 'np.random.rand' not in src and 'uuid4' not in src and 'np.random.normal' not in src


# ----------------------------------------------------------------------------- the page
def _open(win, name, cls):
    if name not in win._page_widgets:
        pg = cls(win.ctx)
        win.pages.addWidget(pg)
        win._page_widgets[name] = pg
    win.set_page(name)
    return win._page_widgets[name]


def test_page_neural_controls_are_undoable(app, win):
    from vantage.ui.workspaces.splat_ws import SplatPage
    win.store.set_project(tigor_sample_project())
    pump(app)
    pg = _open(win, 'splat', SplatPage)
    pump(app)
    before = pg.nr_settings().projection
    assert before == 'ut'
    pg.nr_projection.buttons['ewa'].click()                       # as a user does
    pump(app)
    assert pg.nr_settings().projection == 'ewa'
    assert win.store.project.analysis.neural_replay['projection'] == 'ewa'
    win.store.undo.undo()
    pump(app)
    assert pg.nr_settings().projection == 'ut'
    pg.nr_relight.buttons['snow'].click()
    pump(app)
    assert pg.nr_settings().relight['condition'] == 'snow'
    assert pg.nr_status_text()
    win.store.undo.undo()
    pump(app)
    assert NR.settings_of(win.store.project).relight.get('condition', 'none') == 'none'
    pg.nr_body.buttons['anny'].click()
    pump(app)
    assert pg.nr_settings().body_model == 'anny' and 'Anny' in pg.nr_status_text()
    win.store.set_project(tigor_sample_project())               # another project: the tab follows it
    pump(app)
    assert pg.nr_projection.value() == 'ut' and pg.nr_result is None


# ----------------------------------------------------------------------------- real models and drives
REAL_DRIVE = os.environ.get('VANTAGE_CARLA_DRIVE', '')          # a finished CARLA drive (the data rig's), if any


@pytest.mark.skipif(not NR.pose_status(verify=False)['verified'] or not os.path.isdir(REAL_DRIVE),
                    reason='needs the ViTPose-B download and the CARLA drive')
def test_real_vitpose_keypoints_on_a_carla_frame():
    d = SP.open_drive(REAL_DRIVE)
    peds = {i: tr for i, tr in d.tracks.items() if tr.cls == 'pedestrian'}
    views = NR.walker_views(d, peds, NR.NeuralSettings(min_walker_px=400))
    assert views
    pm = NR.PoseModel()
    cam, fr, boxes = views[len(views) // 2]
    img = (SP._load_rgb(fr[1]) * 255).astype(np.uint8)
    k = pm.keypoints(img, [NR._crop_box(b, img.shape[1], img.shape[0]) for _, b in boxes])
    b = boxes[0][1]
    good = k[0, :, 2] > 0.3
    assert good.sum() >= 8
    assert np.all((k[0, good, 0] > b[0] - 15) & (k[0, good, 0] < b[2] + 15) & (k[0, good, 1] > b[1] - 15) &
                  (k[0, good, 1] < b[3] + 15))
