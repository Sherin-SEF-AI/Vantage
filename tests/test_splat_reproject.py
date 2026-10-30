"""Tests for log replay through a proposed rig (core.splat_reproject, core.splat_raster) and its page
(ui.workspaces.splat_ws).

References: the rasteriser against the closed form of a Gaussian's footprint (EWA splatting with the 0.3 px² dilation)
and of front-to-back alpha compositing, its gradients against finite differences (torch.autograd.gradcheck in double
precision), the camera model against core.lens (Intrinsics.project_dirs and unproject), the lidar against the exact
ranges to a plane, the .ply against its own writer (a round trip) and the INRIA layout's conventions (sigmoid opacity,
exp scales, SH DC colour). A 1.2 s synthetic drive of the sample's data rig (core.dataset, images at 15 %) runs the
whole study on the CPU, as a nuScenes folder and as a ROS 2 bag.
Run with:  QT_QPA_PLATFORM=offscreen python3 -m pytest tests/test_splat_reproject.py -q
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
from vantage.core import splat_raster as SR  # noqa: E402
from vantage.core import splat_reproject as SP  # noqa: E402
from vantage.core.analysis import Engine  # noqa: E402
from vantage.core.lens import resolve  # noqa: E402
from vantage.core.project import Variant  # noqa: E402
from vantage.core.samples import tigor_sample_project  # noqa: E402

from test_gui import app, win, pump, _wait  # noqa: E402,F401  (fixtures)

CPU = torch.device('cpu')


def _one(P, sigma, colour, opacity, dtype=torch.float64):
    P = torch.tensor(P, dtype=dtype).reshape(-1, 3)
    n = len(P)
    R = torch.eye(3, dtype=dtype).repeat(n, 1, 1)
    s = torch.tensor(sigma, dtype=dtype).reshape(n, -1).expand(n, 3)
    return P, SR.covariance(R, s), torch.tensor(colour, dtype=dtype).reshape(n, -1), \
        torch.tensor(opacity, dtype=dtype).reshape(n)


# ----------------------------------------------------------------------------- the rasteriser
def test_footprint_matches_the_closed_form():
    """One isotropic Gaussian on the axis of an equidistant lens: alpha = o·exp(−½ dᵀ Σ₂⁻¹ d), Σ₂ = (fσ/z)² I +
    0.3 I (the projection's Jacobian at the axis is f/z), at every pixel centre."""
    f, z, sig, op = 100.0, 10.0, 0.08, 0.7
    proj = SR.LensProjector(f, f, 32.0, 32.0, 'equidistant', max_theta=math.radians(80))
    P, cov, col, o = _one([0, 0, z], [sig], [1.0], [op])
    out = SR.splat(P, cov, col, o, proj, 64, 64)
    a = out['alpha'].reshape(64, 64).numpy()
    s2 = (f * sig / z) ** 2 + SR.LOWPASS
    yy, xx = np.mgrid[0:64, 0:64] + 0.5
    ref = op * np.exp(-0.5 * ((xx - 32) ** 2 + (yy - 32) ** 2) / s2)
    ref = np.where(ref >= SR.ALPHA_MIN, np.minimum(ref, SR.ALPHA_MAX), 0.0)
    assert np.max(np.abs(a - ref)) < 2e-4
    assert out['image'].reshape(64, 64)[32, 32].item() == pytest.approx(a[32, 32], abs=1e-9)


def test_compositing_is_front_to_back_whatever_the_input_order():
    proj = SR.LensProjector(50, 50, 16, 16, 'kannala_brandt', (0.01, 0.0, 0.0, 0.0), math.radians(100))
    near = ([0, 0, 5.0], [0.3], [1.0, 0.0, 0.0], [0.6])
    far = ([0, 0, 10.0], [0.6], [0.0, 1.0, 0.0], [0.8])
    outs = []
    for order in ((near, far), (far, near)):
        P, cov, col, o = (torch.cat(x) for x in zip(*[_one(*g) for g in order]))
        bg = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64).repeat(32 * 32, 1)
        outs.append(SR.splat(P, cov, col, o, proj, 32, 32, bg=bg)['image'].reshape(32, 32, 3))
    assert torch.allclose(outs[0], outs[1])
    # the centre pixel against the closed form: C = α₁ c₁ + (1 − α₁) α₂ c₂ + (1 − α₁)(1 − α₂) bg
    d2 = 0.5 ** 2 * 2

    def al(o, s, zz):
        return o * math.exp(-0.5 * d2 / ((50 * s / zz) ** 2 + SR.LOWPASS))
    a1, a2 = al(0.6, 0.3, 5.0), al(0.8, 0.6, 10.0)
    c = outs[0][16, 16].numpy()
    assert c == pytest.approx([a1, (1 - a1) * a2, (1 - a1) * (1 - a2)], abs=2e-4)


def test_median_depth_passes_a_faint_surface():
    """A faint Gaussian (opacity 0.3) in front of a dense one: the median depth is the dense one's range, the
    expected depth their weighted mean (α₁ r₁ + (1 − α₁) α₂ r₂) / (α₁ + (1 − α₁) α₂)."""
    proj = SR.LensProjector(50, 50, 16, 16, 'equidistant', max_theta=math.radians(80))
    P, cov, _, o = (torch.cat(x) for x in zip(_one([0, 0, 5.0], [0.5], [5.0], [0.3]),
                                              _one([0, 0, 10.0], [1.0], [10.0], [0.95])))
    feats = torch.linalg.norm(P, dim=1, keepdim=True)
    with torch.no_grad():
        out = SR.splat(P, cov, feats, o, proj, 32, 32, depth_col=0)
    med = out['median'].reshape(32, 32)[16, 16].item()
    a1 = 0.3 * math.exp(-0.5 * 0.5 / ((50 * 0.5 / 5) ** 2 + SR.LOWPASS))
    a2 = 0.95 * math.exp(-0.5 * 0.5 / ((50 * 1.0 / 10) ** 2 + SR.LOWPASS))
    expected = (a1 * 5 + (1 - a1) * a2 * 10) / (a1 + (1 - a1) * a2)
    assert med == pytest.approx(10.0, abs=1e-6)
    assert (out['image'][:, 0] / out['alpha']).reshape(32, 32)[16, 16].item() == pytest.approx(expected, rel=1e-4)


def test_gradients_match_finite_differences():
    proj = SR.LensProjector(40, 40, 12, 12, 'equisolid', max_theta=math.radians(90))
    P0 = torch.tensor([[0.2, -0.1, 4.0], [-0.3, 0.2, 6.0]], dtype=torch.float64)
    q = torch.tensor([[1.0, 0.1, 0.2, 0.0], [0.9, 0.0, -0.3, 0.2]], dtype=torch.float64)
    col0 = torch.tensor([[0.8, 0.2, 0.1], [0.1, 0.5, 0.9]], dtype=torch.float64)

    def f(P, log_s, op, col):
        cov = SR.covariance(SR.quat_to_rot(q), torch.exp(log_s))
        return SR.splat(P, cov, col, op, proj, 24, 24)['image'].sum(0)
    args = (P0.clone().requires_grad_(), torch.log(torch.tensor([[0.2, 0.3, 0.1], [0.4, 0.2, 0.3]],
                                                                dtype=torch.float64)).requires_grad_(),
            torch.tensor([0.5, 0.7], dtype=torch.float64, requires_grad=True), col0.clone().requires_grad_())
    assert torch.autograd.gradcheck(f, args, eps=1e-6, atol=1e-5, rtol=1e-3)


def test_camera_model_is_vantages_lens():
    p = tigor_sample_project()
    c = p.variant.cameras[0]
    intr = resolve(p.module_for(c), p.lens_for(c))
    m = SP.CamModel.from_intrinsics(intr, intr.width // 4, intr.height // 4)
    rng = np.random.default_rng(3)
    D = rng.normal(size=(500, 3))
    D[:, 2] = np.abs(D[:, 2]) + 0.2
    u, v, th, ok = m.project(D)
    U, V, TH, OK = intr.project_dirs(D)
    assert np.allclose(u, U / 4) and np.allclose(v, V / 4) and np.allclose(th, TH)
    rays, okr = m.rays()
    uu, vv, _, _ = m.project(rays[okr])
    yy, xx = np.nonzero(okr)
    assert np.max(np.abs(uu - (xx + 0.5))) < 1e-6 and np.max(np.abs(vv - (yy + 0.5))) < 1e-6
    # the torch projector is the same model
    tu, tv, _ = m.projector()(torch.tensor(D[ok]))
    assert np.allclose(tu.numpy(), u[ok]) and np.allclose(tv.numpy(), v[ok])
    # an OpenCV fisheye calibration (pixel centres on integers) shifts by half a pixel
    K = np.array([[200.0, 0, 159.5], [0, 200.0, 119.5], [0, 0, 1]])
    mo = SP.CamModel.from_opencv(K, [0.01, -0.002, 0.0, 0.0], 320, 240)
    pu, pv, _, _ = mo.project(np.array([[0.0, 0.0, 1.0]]))
    assert pu[0] == pytest.approx(160.0) and pv[0] == pytest.approx(120.0)
    ref, _ = cv2.fisheye.projectPoints(np.array([[[0.3, -0.2, 1.0]]]), np.zeros(3), np.zeros(3), K,
                                       np.array([0.01, -0.002, 0.0, 0.0]))
    pu, pv, _, _ = mo.project(np.array([[0.3, -0.2, 1.0]]))
    assert pu[0] - 0.5 == pytest.approx(ref[0, 0, 0], abs=1e-6) and pv[0] - 0.5 == pytest.approx(ref[0, 0, 1],
                                                                                                abs=1e-6)


def _wall(x=10.0, half=6.0, step=0.05, normal_sigma=0.004, half_z=None):
    """Flat Gaussians tiling the plane x = const (|y| ≤ half, |z| ≤ half_z or half), in a static scene."""
    g = np.arange(-half, half + 1e-9, step)
    gz = np.arange(-(half_z or half), (half_z or half) + 1e-9, step)
    yy, zz = np.meshgrid(g, gz)
    P = np.stack([np.full(yy.size, x), yy.ravel(), zz.ravel()], 1).astype(np.float32)
    n = len(P)
    R = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])        # columns: y, z along the wall, x across
    q = SR.rot_to_quat_np(np.repeat(R[None], n, 0)).astype(np.float32)
    sc = np.tile([step * 0.6, step * 0.6, normal_sigma], (n, 1)).astype(np.float32)
    return SP.Scene(P, sc, q, np.full(n, 0.95, np.float32), np.full((n, 3), 0.5, np.float32),
                    np.full(n, -1, np.int32), np.zeros(n, np.int32), [], np.full((16, 64, 3), 0.6, np.float32),
                    np.zeros(3))


def test_lidar_ranges_to_a_wall_and_dropouts():
    scene = _wall(half=40.0, half_z=3.0, step=0.15)
    traj = SP.Trajectory([0.0, 1.0], np.zeros((2, 3)), np.tile([1.0, 0, 0, 0], (2, 1)))
    el = np.linspace(-5, 5, 9)
    C = 360
    az = (np.arange(C) * -1.0 + 180.0) % 360.0 - 180.0
    spec = SP.LidarSpec('L', 'L', np.eye(3), np.zeros(3), el, az, np.arange(C) / C, 0.1, True, 0.5, 200.0)
    st = SP.SplatSettings(lidar_max_incidence_deg=60.0)
    ts = SP.TorchScene(scene, None, CPU)
    sw = SP.render_lidar(ts, traj, spec, 0.0, st)
    E, A = np.meshgrid(np.radians(el), np.radians(az), indexing='ij')
    exact = 10.0 / (np.cos(E) * np.cos(A))
    inc = np.degrees(np.arccos(np.cos(E) * np.cos(A)))
    hit = (np.abs(A) < math.radians(50)) & (inc < 55)
    assert np.isfinite(sw['range'][hit]).all()
    assert np.nanmax(np.abs(sw['range'][hit] - exact[hit])) < 0.03
    # behind the sensor: nothing; beyond 60° incidence: dropped as grazing
    assert np.isnan(sw['range'][:, np.abs(az) > 100]).all()
    graze = (np.abs(A) > math.radians(62)) & (np.abs(A) < math.radians(80))
    assert np.isnan(sw['range'][graze]).all() and sw['dropped_incidence'] > 0
    # the points lie on the wall, in the sensor frame (within 3 cm, 6 cm for beams nearly at the incidence limit)
    Q = sw['points'][:, :3]
    inc_p = np.degrees(np.arccos(Q[:, 0] / np.linalg.norm(Q, axis=1)))
    assert np.all(np.abs(Q[inc_p < 55, 0] - 10.0) < 0.03) and np.all(np.abs(Q[:, 0] - 10.0) < 0.06)
    # a beam on the own vehicle is a self hit: no return
    spec.body = np.zeros((len(el), C), bool)
    spec.body[4, C // 2] = True
    sw2 = SP.render_lidar(ts, traj, spec, 0.0, st)
    assert np.isnan(sw2['range'][4, C // 2]) and sw2['self_hits'] == 1


def test_camera_render_of_the_wall_has_its_depth():
    scene = _wall(x=8.0, half=6.0, step=0.1)
    traj = SP.Trajectory([0.0, 1.0], np.zeros((2, 3)), np.tile([1.0, 0, 0, 0], (2, 1)))
    from vantage.core.mathutil import BODY_TO_OPTICAL
    m = SP.CamModel(64, 48, 30.0, 30.0, 32.0, 24.0, 'equidistant', (0, 0, 0, 0), math.radians(95))
    ts = SP.TorchScene(scene, None, CPU)
    with torch.no_grad():
        r = SP.render_view(ts, traj, m, BODY_TO_OPTICAL.T, np.zeros(3), 0.0, depth=True, want_index=True)
    D, ok = m.rays()
    exact = 8.0 / D[..., 2]
    centre = ok & (D[..., 2] > math.cos(math.radians(30)))
    assert np.max(np.abs(r['depth'].numpy()[centre] - exact[centre])) < 0.05
    assert (r['alpha'].numpy()[centre] > 0.99).all() and (r['index'].numpy()[centre] >= 0).all()
    # a rolling shutter while the ego moves sideways bends the wall's edge
    traj2 = SP.Trajectory([0.0, 1.0], np.array([[0, 0, 0], [0, 20.0, 0]]), np.tile([1.0, 0, 0, 0], (2, 1)))
    with torch.no_grad():
        a = SP.render_view(ts, traj2, m, BODY_TO_OPTICAL.T, np.zeros(3), 0.5, 0.0)['alpha'].numpy()
        b = SP.render_view(ts, traj2, m, BODY_TO_OPTICAL.T, np.zeros(3), 0.5, 0.05)['alpha'].numpy()
    assert not np.allclose(a, b)


def test_trajectory_and_tracks_interpolate():
    q0 = [1.0, 0, 0, 0]
    q1 = [math.cos(math.radians(45)), 0, 0, math.sin(math.radians(45))]           # 90° about z
    tr = SP.Trajectory([0.0, 1.0], np.array([[0, 0, 0], [10.0, 0, 0]]), np.array([q0, q1]))
    R, p = tr.pose(0.5)
    assert p == pytest.approx([5.0, 0, 0])
    assert math.degrees(math.atan2(R[1, 0], R[0, 0])) == pytest.approx(45.0, abs=1e-6)
    R, p = tr.pose(2.0)                                                            # held at the end
    assert p == pytest.approx([10.0, 0, 0])
    k = SP.Track(1, 'car', 'vehicle.car', True, (4, 2, 1.5), np.array([0.0, 0.5, 2.0]),
                 np.array([[0, 0, 0], [1.0, 0, 0], [5.0, 0, 0]]), np.tile(q0, (3, 1)))
    assert k.pose(0.25)[1] == pytest.approx([0.5, 0, 0])
    assert k.pose(1.2) is None                                                     # a 1.5 s gap
    assert k.pose(2.1)[1] == pytest.approx([5.0, 0, 0]) and k.pose(2.2) is None
    parked = SP.Track(2, 'car', 'vehicle.car', False, (4, 2, 1.5), np.array([1.0]), np.array([[3.0, 1, 0]]),
                      np.array([q0]))
    assert parked.pose(-5.0)[1] == pytest.approx([3.0, 1, 0]) and parked.pose(50.0) is not None


def test_settings_are_tolerant():
    d = SP.SplatSettings.from_dict({'max_gaussians': 1e9, 'voxel_m': 'x', 'source': 'nonsense', 'optimise': 'yes',
                                    'iterations': -5, 'ply_xyz': [1, 2, 'a'], 'ply_ypr': [10, 20, 30], 'junk': 1,
                                    'variant_a': '', 'rolling_shutter': 'off', 'device': 'cpu'})
    assert d.max_gaussians == 300000 and d.voxel_m == 0.05 and d.source == 'lidar' and d.optimise is True
    assert d.iterations == 0 and d.ply_xyz == [0.0, 0.0, 0.0] and d.ply_ypr == [10.0, 20.0, 30.0]
    assert d.variant_a == SP.RECORDED and d.rolling_shutter == 'off' and d.device == 'cpu'
    assert SP.SplatSettings.from_dict(d.to_dict()) == d
    assert SP.SplatSettings.from_dict(None) == SP.SplatSettings()
    p = tigor_sample_project()
    p.analysis.splat = {'eval_instants': 3}
    assert SP.settings_of(p).eval_instants == 3


def test_ply_round_trip_and_alignment(tmp_path):
    sc = _wall(half=0.5, step=0.25)
    sc.colours[:] = np.linspace(0.1, 0.9, sc.n)[:, None]
    sc.opacity[:] = np.linspace(0.2, 0.9, sc.n)
    sc.origin = np.array([100.0, 200.0, 0.0])
    path = tmp_path / 'scene.ply'
    path.write_bytes(SP.ply_bytes(sc))
    v = SP.read_ply(str(path))
    assert len([k for k in v if k.startswith('f_rest_')]) == 45                  # the INRIA trainer's degree-3 layout
    assert v['x'][0] == pytest.approx(sc.means[0, 0] + 100.0)
    st = SP.SplatSettings()
    back = SP.ply_scene(str(path), st, sc.origin)
    assert np.allclose(back.means, sc.means, atol=1e-4) and np.allclose(back.colours, sc.colours, atol=1e-5)
    assert np.allclose(back.opacity, sc.opacity, atol=1e-5) and np.allclose(back.scales, sc.scales, rtol=1e-5)
    Rb, Ra = SR.quat_to_rot_np(back.quats), SR.quat_to_rot_np(sc.quats)
    assert np.allclose(Rb, Ra, atol=1e-5)
    # placed 5 m forward, turned 90° left and doubled
    st = SP.SplatSettings(ply_xyz=[5.0, 0.0, 0.0], ply_ypr=[90.0, 0.0, 0.0], ply_scale=2.0)
    moved = SP.ply_scene(str(path), st, np.zeros(3))
    g = sc.means[0] + sc.origin
    assert moved.means[0] == pytest.approx([5.0 - 2 * g[1], 2 * g[0], 2 * g[2]], abs=1e-2)
    assert moved.scales[0] == pytest.approx(2 * sc.scales[0], rel=1e-5)


# ----------------------------------------------------------------------------- a synthetic drive
@pytest.fixture(scope='module')
def drive(tmp_path_factory):
    p = tigor_sample_project()
    v = next(v for v in p.variants if v.sensors)
    p.active_uid = v.uid
    b = Variant.from_dict(v.to_dict())
    b.uid, b.name = 'proposed-b', 'Data rig · F 0.3 m higher'
    b.by_tag('F').z += 0.3
    p.variants.append(b)
    engine = Engine()
    out = str(tmp_path_factory.mktemp('drive'))
    st = DSET.DatasetSettings.from_dict({'duration_s': 1.0, 'image_scale': 0.15})
    assert DSET.generate(p, engine, st, out) is not None
    cache = str(tmp_path_factory.mktemp('splat_cache'))
    sst = SP.SplatSettings.from_dict({'drive': out, 'device': 'cpu', 'max_gaussians': 6000, 'voxel_m': 0.1,
                                      'iterations': 4, 'train_scale': 0.25, 'holdout_every': 4, 'eval_instants': 1,
                                      'render_scale': 0.12, 'colour_step': 3, 'time_budget_s': 60, 'val_frames': 6,
                                      'val_sweeps': 1})
    return {'p': p, 'engine': engine, 'out': out, 'cache': cache, 'st': sst, 'b': b}


def test_drive_reads_as_nuscenes_and_as_a_bag(drive):
    d = SP.open_drive(drive['out'])
    assert d.source == 'nuscenes' and d.variant == 'Data rig · 6-cam + lidar + radars'
    assert set(d.cameras) == {'F', 'R', 'ML', 'MR', 'QL', 'QR'} and set(d.lidars) == {'L1'}
    assert d.box_source == 'sample_annotation' and d.tracks
    bag = os.path.join(drive['out'], 'rosbag', os.listdir(os.path.join(drive['out'], 'rosbag'))[0])
    with pytest.raises(ValueError):
        SP.open_drive(bag)                                                         # a bag needs a cache folder
    b = SP.open_drive(bag, drive['cache'])
    assert b.source == 'rosbag' and set(b.cameras) == set(d.cameras) and b.box_source == 'bag boxes'
    for tag, c in d.cameras.items():
        cb = b.cameras[tag]
        assert np.allclose(c.R, cb.R, atol=1e-6) and np.allclose(c.t, cb.t, atol=1e-6)
        assert (c.model.width, c.model.height) == (cb.model.width, cb.model.height)
        assert c.model.fx == pytest.approx(cb.model.fx) and c.model.kb == pytest.approx(cb.model.kb)
        assert len(c.frames) == len(cb.frames)
    # the same ego path; the lidar's rings and turning sense measured alike from either
    for t in (0.1, 0.5, 1.0):
        assert np.allclose(d.traj.pose(t)[1], b.traj.pose(t)[1], atol=1e-3)
    Ld, Lb = d.lidars['L1'], b.lidars['L1']
    assert Ld.sense == Lb.sense == -1                                                 # the OS1 turns clockwise
    model = drive['p'].sensor_model_for(next(s for s in drive['p'].variant.sensors if s.kind == 'lidar'))
    assert np.allclose(Ld.elevations, model.elevations(), atol=0.05)
    # every point fired within its sweep, before its stamp
    sw = Ld.sweeps[3]
    tf = SP.firing_times(Ld, sw, SP.load_sweep(sw))
    assert tf.min() >= sw[0] - Ld.period_s - 1e-9 and tf.max() <= sw[0] + 1e-9


def test_reconstruction_is_deterministic_and_validates(drive):
    st = SP.SplatSettings.from_dict(dict(drive['st'].to_dict(), optimise=False))
    d = SP.open_drive(drive['out'])
    a = SP.reconstruct(d, st, drive['p'], drive['engine'])
    b = SP.reconstruct(d, st, drive['p'], drive['engine'])
    for k in ('means', 'scales', 'quats', 'opacity', 'colours', 'actor', 'obj', 'sky'):
        assert np.array_equal(getattr(a, k), getattr(b, k)), k
    assert 1000 < a.n <= st.max_gaussians and a.info['coloured_share'] > 0.1
    assert a.actor_ids and set(a.actor_ids) <= {i for i, t in d.tracks.items() if t.dynamic}
    assert np.all(a.colours >= 0) and np.all(a.colours <= 1)
    # cached: the same scene comes back, a changed build setting builds again
    s1 = SP.build_scene(d, st, drive['p'], drive['engine'], cache_root=drive['cache'])
    s2 = SP.build_scene(d, st, drive['p'], drive['engine'], cache_root=drive['cache'])
    assert s2.info['cached'] and np.array_equal(s1.means, s2.means)
    ts = SP.TorchScene(a, d, CPU)
    val = SP.validate(ts, d, st, drive['p'], drive['engine'])
    assert val['cameras']['psnr'] > 8.0 and 0.0 < val['cameras']['ssim'] < 1.0
    lv = val['lidar']
    assert lv['compared_points'] > 1000 and lv['median_abs_error_m'] < 0.5 and lv['return_agreement'] > 0.3


def test_optimisation_runs_its_steps_and_scores_held_out_frames(drive):
    d = SP.open_drive(drive['out'])
    st = SP.SplatSettings.from_dict(dict(drive['st'].to_dict(), iterations=8, time_budget_s=3600))
    sc = SP.reconstruct(d, st, drive['p'], drive['engine'])
    o = sc.info['optimisation']
    assert o['iterations'] == 8 and o['device'] == 'cpu'
    assert o['held_out_psnr_after_db'] is not None and o['held_out_psnr_before_db'] is not None


def test_ab_run_doc_and_files(drive):
    st = SP.SplatSettings.from_dict(dict(drive['st'].to_dict(), variant_b=drive['b'].uid, optimise=False))
    calls = []

    def nothing(img):
        calls.append(img.shape)
        return {'boxes': np.zeros((0, 4)), 'scores': np.zeros(0), 'cls': np.zeros(0, int), 'coco': np.zeros(0, int)}
    res = SP.run(drive['p'], drive['engine'], st, progress=lambda f, t='': None, detector=nothing,
                 cache_root=drive['cache'])
    assert res is not None and calls
    assert res.rig_a['key'] == SP.RECORDED and res.rig_b['name'] == drive['b'].name
    fa = next(c for c in res.rig_a['cameras'] if c['tag'] == 'F')
    fb = next(c for c in res.rig_b['cameras'] if c['tag'] == 'F')
    assert fb['position_m'][2] == pytest.approx(fa['position_m'][2] + 0.3, abs=1e-3)
    assert len(res.ab['instants']) == 1 and res.ab['lidar']['a']['L1']['returns'] > 0
    assert res.ab['recall']
    for r in res.ab['recall']:
        assert r['recall_a'] == 0.0 and r['recall_b'] == 0.0 and r['objects'] >= 1
    kinds = {f['kind'] for f in res.findings}
    assert {'validation', 'dynamic', 'caveat'} <= kinds
    # measured on a CARLA drive against its real frames: re-render recall is a little lower on the same objects, but
    # far, small and thin objects drop out of the scored set, so the study's absolute recall can come out higher;
    # the caveat must not claim it is simply lower (round 11)
    caveat = next(f['message'] for f in res.findings if f['kind'] == 'caveat')
    assert 'not scored' in caveat and 'absolute recall on them is lower' not in caveat
    assert 'not scored' in SP.ASSUMPTIONS
    doc = SP.to_doc(res, drive['p'])
    json.dumps(doc)
    assert doc['schema'] == SP.SCHEMA and doc['settings']['max_gaussians'] == 6000 and doc['limits']
    names = [n for n, _ in SP.files(res, drive['p'])]
    assert names[:3] == ['log_replay.json', 'log_replay_recall.csv', 'log_replay_validation.csv']
    assert 'log_replay_cameras.png' in names and 'log_replay_lidar.png' in names and 'log_replay_scene.ply' in names
    # cancelled: None
    assert SP.run(drive['p'], drive['engine'], st, cancel=lambda: True, cache_root=drive['cache']) is None


def _grid_detector(img):
    """A stand-in detector whose scores follow the image: a 4 × 4 grid of 'car' boxes scored by their mean value, so
    two renders that differ give different detections."""
    h, w = img.shape[:2]
    boxes, scores = [], []
    for i in range(4):
        for j in range(4):
            x0, x1, y0, y1 = j * w // 4, (j + 1) * w // 4, i * h // 4, (i + 1) * h // 4
            boxes.append([x0, y0, x1, y1])
            scores.append(float(img[y0:y1, x0:x1].mean()) / 255.0)
    return {'boxes': np.array(boxes, float), 'scores': np.array(scores), 'cls': np.full(16, 3),
            'coco': np.full(16, 2)}


def test_identical_rig_compares_exactly_with_default_settings(drive):
    """The drive's own variant as rig B, default rolling shutter: the recorded rig (a drive whose frames are single
    instants) is rendered with the same module readout as the variant, so the images agree to the calibration's
    rounding and every row of the A/B is the same for both rigs (round 11: rig A was rendered as a global shutter and
    single objects flipped)."""
    p = drive['p']
    st = SP.SplatSettings.from_dict(dict(drive['st'].to_dict(), variant_b=p.variant.uid, optimise=False))
    assert st.rolling_shutter == 'module'
    res = SP.run(p, drive['engine'], st, detector=_grid_detector, cache_root=drive['cache'])
    ro_a = {c['tag']: c['readout_ms'] for c in res.rig_a['cameras']}
    ro_b = {c['tag']: c['readout_ms'] for c in res.rig_b['cameras']}
    assert ro_a == ro_b and all(v > 0 for v in ro_a.values())
    for pv in res.ab['previews']:
        for tag, a in pv['a'].items():
            assert np.abs(a.astype(int) - pv['b'][tag].astype(int)).max() <= 2
    assert res.ab['recall']
    for r in res.ab['recall']:
        assert r['visible_a'] == r['visible_b'] and r['recall_a'] == r['recall_b'] and r['delta'] == 0.0
    ca = {c['camera']: (c['objects'], c['detected']) for c in res.ab['cameras']['a']}
    assert ca == {c['camera']: (c['objects'], c['detected']) for c in res.ab['cameras']['b']}
    assert not any(f['kind'] == 'recall' for f in res.findings)
    assert any('rolling shutter' in n for n in res.notes)


def test_recorded_readout_rule():
    """The recorded rig's readout: off → none; the recording's own when the drive states one; else the variant's."""
    p = tigor_sample_project()
    v = next(v for v in p.variants if v.sensors)
    vc = v.by_tag('F')
    m = SP.CamModel.from_intrinsics(resolve(p.module_for(vc), p.lens_for(vc)), 64, 48)
    cam = SP.DriveCamera('F', 'CAM_F', m, np.eye(3), np.zeros(3), [], 0.0)
    st = SP.SplatSettings()
    assert SP._readout(p, vc, st) > 0
    assert SP.recorded_readout(cam, vc, p, st) == pytest.approx(SP._readout(p, vc, st))
    assert SP.recorded_readout(cam, None, p, st) == 0.0
    cam.readout_s = 0.031                                          # a real log stating its rolling shutter
    assert SP.recorded_readout(cam, vc, p, st) == pytest.approx(0.031)
    assert SP.recorded_readout(cam, vc, p, SP.SplatSettings.from_dict({'rolling_shutter': 'off'})) == 0.0


def test_traffic_lights_are_not_assessed():
    """With traffic lights evaluated, a finding says they are mostly not rebuilt and not assessed (with the recording's
    and the re-render's counts when the drive has per-frame ground truth), and no recall finding is drawn from them."""
    de_st = SP.DE.DetectionSettings()
    st = SP.SplatSettings()
    d = SP.Drive.__new__(SP.Drive)
    d.tracks = {}
    sc = SP.Scene.__new__(SP.Scene)
    sc.actor_ids = []
    rig = SP.RigSpec(SP.RECORDED, 'Recorded rig', [], [])
    rows = [{'class': 'traffic_light', 'range': '10–20 m', 'bin': 1, 'objects': 20, 'visible_a': 2, 'visible_b': 20,
             'visible_share_a': 0.1, 'visible_share_b': 1.0, 'recall_a': 0.0, 'recall_b': 0.5, 'delta': 0.5,
             'delta_ci': [0.3, 0.7]}]
    cov = [{'class': 'traffic_light', 'range': '10–20 m', 'bin': 1, 'recorded': 44, 'rendered': 2, 'both': 2},
           {'class': 'vehicle', 'range': '40+ m', 'bin': 3, 'recorded': 128, 'rendered': 78, 'both': 77}]
    ab = {'recall': rows, 'coverage': cov, 'classes': ['vehicle', 'traffic_light'], 'lidar': {}}
    out = SP._findings(d, sc, {}, rig, rig, ab, st, de_st, True)
    tl = [f['message'] for f in out if f['kind'] == 'coverage' and 'Traffic lights' in f['message']]
    assert len(tl) == 1 and 'not assessed' in tl[0] and '2 of the 44' in tl[0]
    assert any(f['kind'] == 'coverage' and 'Vehicle' in f['message'] and '78 of the 128 at 40+ m' in f['message']
               for f in out)
    assert not any(f['kind'] in ('recall', 'visible') for f in out)


def test_unreadable_drive_is_refused(tmp_path):
    with pytest.raises(ValueError):
        SP.open_drive(str(tmp_path))
    with pytest.raises(ValueError):
        SP.open_drive('')
    assert not SP.is_drive(str(tmp_path))


def test_no_unseeded_randomness():
    for mod in (SP, SR):
        src = open(mod.__file__, encoding='utf-8').read()
        assert 'np.random.rand' not in src and 'np.random.seed' not in src and 'random.random(' not in src


# ----------------------------------------------------------------------------- the page
def test_page_builds_previews_runs_and_exports(app, win, drive, tmp_path):
    from vantage.ui.workspaces.splat_ws import SplatPage
    p = win.store.project
    v = next(v for v in p.variants if v.sensors)
    b = Variant.from_dict(drive['b'].to_dict())
    p.variants.append(b)
    pg = SplatPage(win.ctx)
    pg.cache_root = str(tmp_path / 'cache')
    pg.detector = lambda img: {'boxes': np.zeros((0, 4)), 'scores': np.zeros(0), 'cls': np.zeros(0, int),
                               'coco': np.zeros(0, int)}
    pump(app, 0.1)
    assert not pg.b_run.isEnabled() and 'Open a recorded drive' in pg.status.text()
    assert not pg.open_drive(str(tmp_path))
    assert pg.open_drive(drive['out'])
    pump(app, 0.1)
    assert pg.settings.drive == os.path.abspath(drive['out']) and pg.b_run.isEnabled()
    # settings typed into their fields, undoable
    for k, val in drive['st'].to_dict().items():
        if k in pg.fields:
            pg.fields[k].set_value(val)
            pg.fields[k].committed.emit(float(val))
    pg.device.set_value('cpu')
    pg.device.changed.emit('cpu')
    pg.optimise.setChecked(False)
    pump(app, 0.1)
    assert pg.settings.max_gaussians == 6000 and pg.settings.device == 'cpu' and not pg.settings.optimise
    win.store.undo.undo()
    pump(app, 0.1)
    assert pg.settings.optimise
    pg.optimise.setChecked(False)
    pg.rig_b.setCurrentIndex(pg.rig_b.findData(b.uid))
    pump(app, 0.1)
    assert pg.settings.variant_b == b.uid and pg.rig_a.currentData() == SP.RECORDED
    assert pg.build()
    _wait(app, lambda: not pg.busy, 240)
    assert pg.scene is not None and pg.scene.n > 1000, pg.status.text()
    pg.slider.setValue(500)
    assert pg.preview()
    _wait(app, lambda: not pg.busy, 240)
    assert pg.preview_out is not None and pg.view_a.img is not None and pg.view_b.img is not None
    assert pg.cam_pick.count() == 6
    assert pg.run()
    _wait(app, lambda: not pg.busy, 300)
    assert pg.result is not None, pg.status.text()
    assert pg.val_table.rowCount() >= 1 and pg.ab_table.rowCount() == len(pg.result.ab['recall']) >= 1
    assert pg.summary()['rig_b'] == b.name
    paths = pg.export(str(tmp_path / 'out'))
    assert any(x.endswith('log_replay.json') for x in paths)
    pg.cancel()
    pg.close()
