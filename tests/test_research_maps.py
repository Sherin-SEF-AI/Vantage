"""Research maps (core/research_maps.py) against hand calculations: a pinhole camera's pixel footprint in closed form,
stereo σ ∝ 1 / baseline, the drift shift r·δθ / sin(depression), the BEV grid's feature cells, the lidar ground
density, and the probe's readout on the sample."""
import copy
import math

import numpy as np
import pytest

from vantage.core import research_maps as RM
from vantage.core import samples
from vantage.core.analysis import Engine, camera_frames
from vantage.core.lens import Lens, Module, resolve
from vantage.core.project import Camera


@pytest.fixture(scope='module')
def eng():
    return Engine()


@pytest.fixture(scope='module')
def sample(eng):
    p = samples.tigor_sample_project()
    return p, eng.analyse(p)


def _pinhole(f_mm=4.0):
    m = Module(id='m', width_px=1920, height_px=1536, pixel_um=3.0)
    ln = Lens(id='p', focal_mm=f_mm, projection='rectilinear')
    return m, ln, resolve(m, ln)


# ---------------------------------------------------------------- footprint and angular sampling
def test_pinhole_ground_footprint_closed_form():
    m, ln, intr = _pinhole()
    cam = Camera(tag='T', x=0.0, y=0.0, z=2.0, yaw=0.0, pitch=90.0, roll=0.0)   # straight down
    fpx = intr.f_px
    # a ground square of side h at the nadir, 2 m below: (f_px · h / 2)² pixels
    for h in (0.05, 0.2):
        a = RM.ground_px_per_m2(cam, intr, np.array([[0.0, 0.0, 0.0]]), h=h)[0]
        assert a == pytest.approx((fpx / 2.0) ** 2, rel=1e-6)
    # the ground is parallel to this pinhole's image plane: its image is a uniform scale, the same off the nadir
    a = RM.ground_px_per_m2(cam, intr, np.array([[1.0, 0.4, 0.0]]), h=0.01)[0]
    assert a == pytest.approx((fpx / 2.0) ** 2, rel=1e-6)
    # tilted 40° down at 2 m, the point on the axis: f² sin(40°)³ / h_cam² (the plane foreshortened along the axis,
    # and distance 2 / sin 40° from the camera)
    tilt = Camera(tag='T', x=0.0, y=0.0, z=2.0, yaw=0.0, pitch=40.0, roll=0.0)
    xa = 2.0 / math.tan(math.radians(40.0))
    a = RM.ground_px_per_m2(tilt, intr, np.array([[xa, 0.0, 0.0]]), h=0.002)[0]
    assert a == pytest.approx(fpx ** 2 * math.sin(math.radians(40.0)) ** 3 / 4.0, rel=2e-3)
    # angular sampling of a pinhole at its axis is f in pixels
    assert RM.px_per_rad(intr, 1e-4) == pytest.approx(fpx, rel=1e-4)


def test_drift_shift_matches_r_dtheta_over_sin_depression():
    C = np.array([0.0, 0.0, 1.5])
    for x in (3.0, 6.0, 12.0):
        P = np.array([x, 0.0, 0.0])
        dep = math.atan2(1.5, x)
        cam = Camera(tag='T', x=0.0, y=0.0, z=1.5, yaw=0.0, pitch=math.degrees(dep), roll=0.0)
        R, _ = camera_frames(cam)
        d = RM.drift_shift_cm(C, R, P, 0.2)
        r = math.hypot(x, 1.5)
        assert d['pitch'] == pytest.approx(100 * r * math.radians(0.2) / math.sin(dep), rel=0.03)
        # yaw about the camera's own (tilted) up axis swings the ray sideways by its slant range × δθ
        assert d['yaw'] == pytest.approx(100 * r * math.radians(0.2), rel=0.02)
        assert d['worst'] == max(d['pitch'], d['yaw'], d['roll'])


# ---------------------------------------------------------------- stereo
def test_pair_sigma_scales_as_one_over_the_baseline():
    P = np.array([[10.0, 0.0, 0.0]])
    sig = []
    for b in (0.03, 0.3, 1.8):
        s, ra, rb, g = RM.pair_sigma([0, -b / 2, 1.5], [0, b / 2, 1.5], P, 1000.0, 1000.0, 0.5)
        sig.append(s[0])
        r = math.hypot(10.0, 1.5)
        # far away: γ ≈ b / r, so σ ≈ √2 · (0.5 / 1000) · r / (b / r)
        assert s[0] == pytest.approx(math.sqrt(2) * 0.5e-3 * r * r / b, rel=0.02)
    assert sig[0] / sig[2] == pytest.approx(1.8 / 0.03, rel=0.02)


def test_stereo_map_tells_a_short_baseline_from_a_long_one(eng):
    base = samples.tigor_sample_project()
    out = {}
    for b in (0.03, 0.6):
        p = copy.deepcopy(base)
        v = p.variant
        f = v.by_tag('F')
        f2 = Camera.from_dict(f.to_dict())
        f2.uid, f2.tag, f2.y, f2.port = 'f2uid', 'F2', f.y + b, ''
        v.cameras = [f, f2]
        res = eng.analyse(p)
        sm = RM.stereo_depth_sigma(p, eng, result=res)
        ok = np.isfinite(sm.sigma_m)
        assert ok.sum() > 100 and sm.pairs == [('F', 'F2')]
        assert np.nanmedian(sm.baseline_m) == pytest.approx(b, abs=1e-3)
        out[b] = float(np.nanmedian(sm.sigma_m))
    assert out[0.03] > 10 * out[0.6]


def test_stereo_on_the_sample(sample, eng):
    p, res = sample
    sm = RM.stereo_depth_sigma(p, eng, result=res)
    ok = np.isfinite(sm.sigma_m)
    assert ok.sum() == int(((res.count >= 2) & res.grid.roi).sum())
    s = sm.summary()
    assert s['cells'] == ok.sum() and 0 < s['median_sigma_m'] < 0.05 and s['pairs']
    # every reported baseline is a real distance between two of the variant's cameras
    cams = {c.tag: c.pos() for c in p.variant.cameras}
    for row in s['pairs']:
        a, b = row['pair'].split('·')
        assert row['baseline_m'] == pytest.approx(float(np.linalg.norm(cams[a] - cams[b])), abs=1e-3)


# ---------------------------------------------------------------- BEV grid
def test_bev_feature_map(sample, eng):
    p, res = sample
    b = RM.bev_feature_map(p, eng, result=res)
    assert b.n == 200 and b.cell_m == 0.5 and b.stride == 16 and b.feats.shape == (200, 200)
    assert b.ms < 2000
    out = ~b.body
    assert (b.rays[b.body] == 0).all() and (b.feats[b.body] == 0).all()
    assert np.allclose(sum(b.per_cam.values()), b.feats, rtol=1e-5)
    assert (b.best_feats <= b.feats + 1e-6).all() and ((b.best >= 0) == (b.rays > 0)).all()
    assert (b.starved == ((b.feats < 1.0) & out)).all()
    # the support falls off with range: near cells get many feature cells, far ones are starved
    r = np.hypot(*np.meshgrid(b.xs - (b.x0 + b.extent_m), b.ys - (b.y0 + b.extent_m)))
    near = out & (r < 5) & (b.rays > 0)
    far = out & (r > 40)
    assert np.median(b.feats[near]) > 10 * np.median(b.feats[far])
    assert b.starved[far].mean() > b.starved[near].mean()
    # stride: feature cells scale as 1 / stride²
    b32 = RM.bev_feature_map(p, eng, result=res, stride=32)
    assert np.allclose(b32.feats * 4, b.feats, rtol=1e-5)
    s = b.summary()
    assert s['cells'] == 40000 and 0 <= s['starved_pct'] <= 100


def test_bev_feature_cells_of_a_pinhole_closed_form(eng, sample):
    """One downward pinhole over open ground: a 0.5 m cell under it covers (f_px · 0.5 / 2)² / 16² feature cells."""
    p = copy.deepcopy(sample[0])
    v = p.variant
    m, ln, intr = _pinhole()
    p.modules[m.id] = m
    p.lenses[ln.id] = ln
    ln.module_id = m.id
    cam = Camera(tag='T', x=12.0, y=0.0, z=2.0, yaw=0.0, pitch=90.0, roll=0.0, module_id=m.id, lens_id=ln.id)
    v.cameras = [cam]
    b = RM.bev_feature_map(p, eng, extent_m=20.0, cell_m=0.5, stride=16)
    assert b.rays.max() == 1
    k = np.unravel_index(np.argmax(b.feats), b.feats.shape)
    assert b.feats[k] == pytest.approx((intr.f_px * 0.5 / 2.0) ** 2 / 256, rel=0.05)


# ---------------------------------------------------------------- lidar–camera association
def test_lidar_camera_association_on_the_data_rig(eng):
    p = samples.tigor_sample_project()
    p.active_uid = next(v for v in p.variants if v.sensors).uid
    res = eng.analyse(p)
    L = RM.lidar_camera_association(p, eng, result=res)
    lids = [s for s in p.variant.sensors if s.kind == 'lidar' and s.visible]
    assert len(L) == len(lids) and L[0]['tag'] == lids[0].tag
    a = L[0]
    m = p.sensor_model_for(lids[0])
    assert len(a['points']) > 1000 and np.allclose(a['points'][:, 2], 0.0, atol=1e-9)
    assert set(np.unique(a['ring'])) <= set(range(m.n_channels()))
    # the lidar's ground density at a point by hand: 1 / (radial ring spacing × range · h_res)
    pos = lids[0].pos()
    rho = np.hypot(a['points'][:, 0] - pos[0], a['points'][:, 1] - pos[1])
    assert (a['lidar_pts_m2'] > 0).all() and np.corrcoef(np.log(rho), np.log(a['lidar_pts_m2']))[0, 1] < -0.8
    seen = a['cam_px_m2'] > 0
    assert seen.mean() > 0.5 and np.allclose(a['ratio'][seen], a['cam_px_m2'][seen] / a['lidar_pts_m2'][seen])
    assert set(a['camera'][seen]) <= {c.tag for c in p.variant.cameras}
    assert [s['sector'] for s in a['sectors']] == list(RM.SECTORS) and sum(s['points'] for s in a['sectors']) == len(
        a['points'])


# ---------------------------------------------------------------- probe
@pytest.mark.parametrize('kind', list(RM.PROBES))
def test_probe_readout(sample, eng, kind):
    import time
    p, res = sample
    t = time.perf_counter()
    r = RM.probe_readout(p, eng, (6.0, 0.5), kind=kind, result=res)
    ms = (time.perf_counter() - t) * 1000
    assert ms < 100 and r['kind'] == kind and r['seen_by']
    assert 'F' in r['seen_by']
    f = next(c for c in r['cameras'] if c['tag'] == 'F')
    w, h = f['bbox_px']
    assert w > 5 and h > 5 and f['rays'] and all(np.allclose(a, p.variant.by_tag('F').pos()) for a, _ in f['rays'])
    C = p.variant.by_tag('F').pos()
    d = r['point'] - C
    assert f['incidence_deg'] == pytest.approx(math.degrees(math.acos(abs(d[2]) / np.linalg.norm(d))), abs=0.01)
    assert f['dr_dtheta_px_per_rad'] > 100 and f['drift_cm']['worst'] > 0
    # a probe behind the car: the front camera does not see it
    back = RM.probe_readout(p, eng, (-6.0, 0.0), kind=kind, result=res)
    assert 'F' not in back['seen_by'] and 'R' in back['seen_by']
    with pytest.raises(ValueError):
        RM.probe_readout(p, eng, (1, 1), kind='tree')
