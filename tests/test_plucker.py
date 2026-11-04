"""Plücker rays (core/plucker.py): unit directions and m · d = 0, the moment the same along the line, a pinhole camera's
directions against K⁻¹[u, v, 1], the incidence of a downward camera against its closed form, validity outside the
image circle, the ego mask on the sample, the frames, and the export's files."""
import json
import math

import numpy as np
import pytest

from vantage.core import plucker as PL
from vantage.core import samples
from vantage.core.analysis import Engine, camera_frames
from vantage.core.lens import Lens, Module, resolve
from vantage.core.project import Camera, Project


@pytest.fixture(scope='module')
def eng():
    return Engine()


@pytest.fixture(scope='module')
def sample():
    return samples.tigor_sample_project()


def _pinhole_project(pitch=0.0, z=1.5, f_mm=4.0):
    p = Project()
    m = Module(id='pm', width_px=640, height_px=480, pixel_um=3.0)
    ln = Lens(id='pl', module_id='pm', focal_mm=f_mm, projection='rectilinear')
    p.modules = {'pm': m}
    p.lenses = {'pl': ln}
    cam = Camera(tag='P', module_id='pm', lens_id='pl', x=1.0, y=0.5, z=z, yaw=0.0, pitch=pitch)
    p.variant.cameras = [cam]
    return p, cam


def test_unit_directions_and_moments(eng, sample):
    cam = sample.variant.by_tag('F')
    f = PL.ray_field(sample, eng, cam, stride=16)
    d, m = f.d.reshape(-1, 3).astype(float), f.m.reshape(-1, 3).astype(float)
    assert np.allclose(np.linalg.norm(d, axis=1), 1.0, atol=1e-5)
    assert np.abs(np.sum(m * d, axis=1)).max() < 1e-5                  # m ⊥ d
    # the moment is the line's: o + t d gives the same m for any t
    o = f.origin
    for t in (0.5, 3.0, 17.0):
        q = o + t * d
        assert np.allclose(np.cross(q, d), m, atol=2e-5)
    # [d, m] and [m, d]
    pk = f.plucker('dm')
    assert pk.shape == f.d.shape[:2] + (6,) and np.allclose(pk[..., :3], f.d) and np.allclose(f.plucker('md')[..., :3], f.m)
    with pytest.raises(ValueError):
        f.plucker('xx')


def test_pinhole_directions_match_the_inverse_intrinsics(eng):
    p, cam = _pinhole_project()
    f = PL.ray_field(p, eng, cam, stride=8, frame='camera')
    intr = resolve(p.module_for(cam), p.lens_for(cam))
    U, V = np.meshgrid(f.u, f.v)
    K = np.array([[intr.f_px, 0, intr.cx], [0, intr.f_px * intr.aspect, intr.cy], [0, 0, 1.0]])
    ray = np.linalg.solve(K, np.stack([U.ravel(), V.ravel(), np.ones(U.size)]))
    ray /= np.linalg.norm(ray, axis=0)
    assert np.allclose(f.d.reshape(-1, 3), ray.T, atol=1e-6)
    assert np.allclose(f.m, 0.0) and f.frame == 'camera'                # the optical frame: o = 0
    # pixel centres at i + ½ on the stride grid
    assert f.u[0] == pytest.approx(4.0) and f.v[1] == pytest.approx(12.0) and len(f.u) == 80 and len(f.v) == 60
    # in the vehicle frame the directions are the same rays turned by the camera's rotation
    g = PL.ray_field(p, eng, cam, stride=8, frame='vehicle')
    _, W2O = camera_frames(cam)
    assert np.allclose(g.d.reshape(-1, 3), f.d.reshape(-1, 3) @ W2O, atol=1e-6)
    assert np.allclose(g.origin, cam.pos())


def test_incidence_and_ground_range_of_a_downward_camera(eng):
    # pitched 90° down (pitch positive down), 1.5 m up: a ray at angle a off the axis meets the ground at incidence a
    # and range h / cos a
    p, cam = _pinhole_project(pitch=90.0, z=1.5)
    f = PL.ray_field(p, eng, cam, stride=16)
    a = f.theta.astype(float)
    assert np.allclose(f.incidence_deg, np.degrees(a), atol=1e-3)
    assert np.allclose(f.ground_range, 1.5 / np.cos(a), rtol=1e-4)
    assert f.valid.all() and not f.ego.any()


def test_validity_outside_the_image_circle(eng):
    p, cam = _pinhole_project()
    p.lenses['pl'] = Lens(id='pl', module_id='pm', focal_mm=1.2, projection='equidistant', max_half_deg=40.0)
    f = PL.ray_field(p, eng, cam, stride=8)
    intr = resolve(p.module_for(cam), p.lens_for(cam))
    out = f.theta > intr.max_theta + 1e-6
    assert out.any() and (~f.in_lens[out]).all() and (~f.valid[out]).all()
    assert f.in_lens[~out].all()
    with pytest.raises(ValueError):
        PL.ray_field(p, eng, cam, frame='world')


def test_the_sample_cameras_see_the_body_in_their_ego_mask(eng, sample):
    cam = sample.variant.by_tag('ML')                    # the mirror camera looks down along the body
    f = PL.ray_field(sample, eng, cam, stride=16)
    assert f.ego.any() and np.isfinite(f.body_depth[f.ego]).all() and (f.body_depth[f.ego] < 5).all()
    assert np.isnan(f.ground_range[f.ego]).all() and not f.valid[f.ego].any()
    assert 0.0 < f.ego.mean() < 0.6


def test_export_files(eng, sample, tmp_path):
    cams = [sample.variant.by_tag(t) for t in ('F', 'R')]
    paths = PL.export_plucker(sample, eng, str(tmp_path), stride=32, cameras=cams)
    assert [p.split('/')[-1] for p in paths] == ['plucker_F.npz', 'plucker_R.npz', 'plucker_manifest.json']
    z = np.load(paths[0])
    assert z['plucker'].dtype == np.float16 and z['plucker'].shape[-1] == 6
    assert z['valid'].dtype == bool and z['u'].shape[0] == z['plucker'].shape[1]
    man = json.loads(open(paths[-1]).read())
    assert man['schema'] == PL.SCHEMA and man['order'] == 'dm' and man['frame'] == 'vehicle'
    row = man['cameras'][0]
    assert row['camera'] == 'F' and row['shape'] == list(z['plucker'].shape) and 0 < row['valid_share'] <= 1
    assert row['intrinsics']['projection'] and math.isfinite(row['intrinsics']['f_px'])
    f32 = PL.export_plucker(sample, eng, str(tmp_path / 'b'), stride=64, dtype='float32', cameras=cams[:1])
    assert np.load(f32[0])['plucker'].dtype == np.float32
