"""3D occupancy-grid support (core/occupancy.py): a voxel's projected area against a pinhole camera's closed form, a
single-beam lidar's samples through a voxel against counting its real beam directions through the box, the stereo
sigma against research_maps.pair_sigma, the Occ3D grid on the sample's data rig (shape, ground, body, the summary),
and the export's files."""
import dataclasses
import json
import math

import numpy as np
import pytest

from vantage.core import occupancy as OC
from vantage.core import research_maps as RM
from vantage.core import samples
from vantage.core import sensors as S
from vantage.core.analysis import Engine
from vantage.core.lens import Lens, Module, resolve
from vantage.core.project import Camera, Project, Sensor


@pytest.fixture(scope='module')
def eng():
    return Engine()


@pytest.fixture(scope='module')
def rig(eng):
    p = samples.tigor_sample_project()
    p.active_uid = next(v.uid for v in p.variants if v.sensors)
    return p, eng.analyse(p)


@pytest.fixture(scope='module')
def vd(rig, eng):
    p, res = rig
    return OC.voxel_density(p, eng, result=res)


def test_a_voxel_seen_face_on_by_a_pinhole_camera():
    p = Project()
    p.modules = {'pm': Module(id='pm', width_px=1280, height_px=960, pixel_um=3.0)}
    p.lenses = {'pl': Lens(id='pl', module_id='pm', focal_mm=6.0, projection='rectilinear')}
    cam = Camera(tag='P', module_id='pm', lens_id='pl', x=0.0, y=0.0, z=1.0, yaw=0.0, pitch=0.0, roll=0.0)   # along +x
    intr = resolve(p.module_for(cam), p.lens_for(cam))
    a, r = 0.4, 12.0
    P = np.array([[r, 0.0, 1.0]])
    area = OC.voxel_area_px(cam, intr, P, a)[0]
    assert area == pytest.approx((a * intr.f_px / r) ** 2, rel=1e-3)               # a² f² / r², face on
    # seen along a diagonal of the cube the silhouette is √3 larger (|n_x| + |n_y| + |n_z| = √3)
    k = r / math.sqrt(3)
    diag = OC.voxel_area_px(cam, intr, np.array([[k, k, 1.0 + k]]), a)[0]
    _, _, th, _ = RM._project(cam, intr, np.array([[k, k, 1.0 + k]]))
    pr = RM.px_per_rad(intr, th)[0]
    assert diag == pytest.approx(a * a * math.sqrt(3) * (pr / r) ** 2, rel=1e-6)


def test_a_single_beam_lidar_against_its_real_beams(eng):
    base = S.builtin_sensor_models()
    m = dataclasses.replace(next(x for x in base if x.id == 'velodyne_vlp16'), beams=[0.0], channels=1, h_res=0.2,
                            range_max_m=100.0, range_m=100.0, range_min_m=0.3)
    s = Sensor(kind='lidar', model_id=m.id, x=0.0, y=0.0, z=1.0)
    a = 0.4
    for P in (np.array([10.0, 0.0, 1.0]), np.array([0.0, 20.0, 1.1]), np.array([-7.0, -7.0, 0.95])):
        n = OC.lidar_samples(s, m, s.pos().astype(float), P[None], a)[0]
        # count the real beam directions (one channel, every h_res) whose ray enters the box
        D = S.beam_dirs(s, m, m.h_res).reshape(-1, 3)
        o = s.pos().astype(float)
        lo, hi = P - a / 2, P + a / 2
        with np.errstate(divide='ignore', invalid='ignore'):
            t1, t2 = (lo - o) / D, (hi - o) / D
        tmin = np.nanmax(np.minimum(t1, t2), axis=1)
        tmax = np.nanmin(np.maximum(t1, t2), axis=1)
        hits = int(((tmax >= tmin) & (tmax > 0)).sum())
        assert n == pytest.approx(hits, abs=1.5), (P, n, hits)
    # out of range, below the field
    assert OC.lidar_samples(s, m, s.pos().astype(float), np.array([[150.0, 0, 1.0]]), a)[0] == 0.0
    assert OC.lidar_samples(s, m, s.pos().astype(float), np.array([[5.0, 0, 3.0]]), a)[0] == 0.0


def test_the_occupancy_grid_on_the_data_rig(vd, rig):
    p, res = rig
    assert vd.shape == (16, 200, 200) and vd.voxel_m == 0.4 and vd.z0 == -1.0
    xs, ys, zs = vd.centres()
    assert xs[0] == pytest.approx(-39.8) and zs[-1] == pytest.approx(5.2)
    # under the ground and inside the vehicle nothing is seen
    assert vd.below[zs < 0].all() and (vd.cam_rays[vd.below] == 0).all() and (vd.lidar_pts[vd.below] == 0).all()
    assert vd.body.any() and (vd.cam_rays[vd.body] == 0).all()
    # the roof lidar gives most free voxels at least one sample; the far field starves the cameras' features
    s = vd.summary()
    assert s['all']['lidar_sparse_pct'] < 10 and s['lidars'] == ['L1']
    rings = s['range_rings']
    assert rings[-1]['feature_starved_pct'] > rings[1]['feature_starved_pct']
    assert rings[0]['median_lidar_pts_where_hit'] > rings[-1]['median_lidar_pts_where_hit']
    assert s['height_bands'][0]['voxels'] == 0 and s['height_bands'][1]['camera_seen_pct'] > 90
    assert vd.ms < 20000
    json.dumps(s)


def test_the_stereo_sigma_is_the_best_pair(vd, rig, eng):
    p, res = rig
    k, j, i = [int(x[0]) for x in np.nonzero(np.isfinite(vd.sigma_m) & (vd.cam_rays >= 2))]
    xs, ys, zs = vd.centres()
    P = np.array([[xs[i], ys[j], zs[k]]])
    cams = RM._cams(p)
    seen = []
    from vantage.core.analysis import eval_points
    for c in cams:
        intr = RM._intr(p, c)
        vis, _, _, _ = eval_points(P, c, intr, RM._cube(eng, p, c, res))
        if vis[0]:
            _, _, th, _ = RM._project(c, intr, P)
            seen.append((c, RM.px_per_rad(intr, th)))
    best = min(RM.pair_sigma(a.pos(), b.pos(), P, pa, pb)[0][0] for n_, (a, pa) in enumerate(seen)
               for (b, pb) in seen[n_ + 1:])
    assert vd.sigma_m[k, j, i] == pytest.approx(best, rel=1e-4)
    assert len(seen) == vd.cam_rays[k, j, i]


def test_export(rig, eng, tmp_path):
    p, res = rig
    paths = OC.export_occupancy(p, eng, str(tmp_path), voxel_m=0.5, extent_m=20.0, result=res)
    names = [x.split('/')[-1] for x in paths]
    assert names[:2] == ['occupancy_density.npz', 'occupancy_density.json'] and any(n.endswith('.png') for n in names)
    z = np.load(paths[0])
    assert z['feats'].shape == (13, 80, 80) and z['lidar_pts'].dtype == np.float32
    meta = json.loads(open(paths[1]).read())
    assert meta['schema'] == OC.SCHEMA and meta['summary']['grid']['voxel_m'] == 0.5
    with pytest.raises(ValueError):
        OC.voxel_density(p, eng, centre='moon')
