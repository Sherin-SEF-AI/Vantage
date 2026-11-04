"""Virtual cameras (core/rectify.py): a rectilinear lens rectifies to itself, a ray through the physical fisheye and the
remap lands where the virtual pinhole and cylinder project it (sub-pixel), every valid map pixel is inside the
physical image and its image circle, the canonical camera has one K for all, levelling removes roll and pitch, and
the files read back with OpenCV."""
import math

import cv2
import numpy as np
import pytest

from vantage.core import rectify as RC
from vantage.core import samples
from vantage.core.analysis import camera_frames
from vantage.core.lens import Lens, Module, resolve


@pytest.fixture(scope='module')
def proj():
    return samples.tigor_sample_project()


def _bilinear(m, x, y):
    x0, y0 = int(math.floor(x)), int(math.floor(y))
    fx, fy = x - x0, y - y0
    return ((1 - fx) * (1 - fy) * m[y0, x0] + fx * (1 - fy) * m[y0, x0 + 1] + (1 - fx) * fy * m[y0 + 1, x0]
            + fx * fy * m[y0 + 1, x0 + 1])


def test_a_rectilinear_lens_rectifies_to_itself():
    m = Module(id='m', width_px=1280, height_px=800, pixel_um=3.0)
    ln = Lens(id='p', focal_mm=4.0, projection='rectilinear')
    intr = resolve(m, ln)
    vc = RC.virtual_pinhole(None, None, intr=intr)
    assert vc.fx == pytest.approx(intr.f_px, rel=1e-6)
    # the inscribed rectangle is the whole physical image (to the search's last step)
    assert abs(vc.width - 1280) <= 2 and abs(vc.height - 800) <= 2
    jj, ii = np.mgrid[0:vc.height, 0:vc.width]
    off_x = (intr.cx - 0.5) - vc.cx
    off_y = (intr.cy - 0.5) - vc.cy
    assert np.abs(vc.map_x - (ii + off_x)).max() < 1e-3 and np.abs(vc.map_y - (jj + off_y)).max() < 1e-3


@pytest.mark.parametrize('mode', ['pinhole', 'cylindrical', 'level'])
def test_a_ray_lands_where_both_cameras_put_it(proj, mode):
    cam = proj.variant.by_tag('F')
    intr = resolve(proj.module_for(cam), proj.lens_for(cam))
    if mode == 'cylindrical':
        vc = RC.virtual_cylindrical(proj, cam)
    else:
        vc = RC.virtual_pinhole(proj, cam, level='roll' if mode == 'level' else None)
    rng = np.random.default_rng(1)
    for _ in range(40):
        # a random virtual pixel inside the image, its ray, through the physical lens
        u, v = rng.uniform(2, vc.width - 3), rng.uniform(2, vc.height - 3)
        if mode == 'cylindrical':
            phi, t = (u - vc.cx) / vc.fx, (v - vc.cy) / vc.fy
            d = np.array([math.sin(phi), t, math.cos(phi)])
        else:
            d = np.array([(u - vc.cx) / vc.fx, (v - vc.cy) / vc.fy, 1.0])
        d_phys = vc.R_vp @ (d / np.linalg.norm(d))
        up, vp_, _, ok = intr.project_dirs(d_phys[None])
        assert ok[0]
        assert _bilinear(vc.map_x, u, v) == pytest.approx(up[0] - 0.5, abs=0.05)
        assert _bilinear(vc.map_y, u, v) == pytest.approx(vp_[0] - 0.5, abs=0.05)


def test_valid_pixels_are_inside_the_image_circle(proj):
    cam = proj.variant.by_tag('ML')
    intr = resolve(proj.module_for(cam), proj.lens_for(cam))
    vc = RC.virtual_pinhole(proj, cam, max_half_deg=80.0, max_side=1600)
    assert max(vc.width, vc.height) <= 1600 and 'lowered' in vc.note           # kept within max_side
    assert vc.valid.all() and vc.coverage == 1.0                     # the inscribed rectangle: every pixel valid
    x, y = vc.map_x[vc.valid] + 0.5, vc.map_y[vc.valid] + 0.5
    assert x.min() >= 0 and x.max() <= intr.width and y.min() >= 0 and y.max() <= intr.height
    d, th = intr.unproject(x, y)
    assert th.max() <= intr.max_theta + 1e-6
    # a pinhole wider than the lens reaches: forced by a large given extent, the outside is flagged invalid
    big = RC._make('ML', 'pinhole', intr, np.eye(3), 60.0, math.tan(math.radians(85)), math.tan(math.radians(80)),
                   None)
    assert 0 < big.coverage < 1
    xb, yb = big.map_x[big.valid] + 0.5, big.map_y[big.valid] + 0.5
    assert intr.unproject(xb, yb)[1].max() <= intr.max_theta + 1e-6


def test_canonical_camera_is_one_camera(proj):
    cams = proj.variant.cameras[:3]
    vcs = RC.canonical(proj, cams)
    K0 = vcs[0].K
    assert all(np.allclose(v.K, K0) and (v.width, v.height) == (vcs[0].width, vcs[0].height) for v in vcs)
    assert all(v.coverage == 1.0 for v in vcs)
    own = [RC.virtual_pinhole(proj, c) for c in cams]
    assert vcs[0].width <= min(o.width for o in own) + 2 and vcs[0].fx <= min(o.fx for o in own) + 1e-9


def test_levelling_removes_roll_and_pitch(proj):
    cam = proj.variant.by_tag('F')
    cam2 = type(cam).from_dict(cam.to_dict())
    cam2.roll = 7.0
    R_roll = RC.level_rotation(cam2, 'roll')
    R_hor = RC.level_rotation(cam2, 'horizon')
    _, W2O = camera_frames(cam2)
    O2W = W2O.T
    # the virtual camera's x axis (image right) is horizontal after either; 'horizon' also puts its axis level
    for R in (R_roll, R_hor):
        x_world = O2W @ (R @ np.array([1.0, 0, 0]))
        assert abs(x_world[2]) < 1e-9
    z_world = O2W @ (R_hor @ np.array([0, 0, 1.0]))
    assert abs(z_world[2]) < 1e-9
    assert np.allclose(R_roll @ R_roll.T, np.eye(3)) and np.allclose(RC.level_rotation(cam2, None), np.eye(3))


@pytest.mark.parametrize('mode', ['pinhole', 'cylindrical', 'canonical'])
def test_files(proj, tmp_path, mode):
    cams = proj.variant.cameras[:2]
    paths = RC.export_rectify(proj, str(tmp_path), mode=mode, cams=cams)
    assert sorted(p.rsplit('/', 1)[1] for p in paths) == sorted(
        [f'{c.tag}_virtual.yaml' for c in cams] + [f'{c.tag}_maps.npz' for c in cams])
    fs = cv2.FileStorage(str(tmp_path / f'{cams[0].tag}_virtual.yaml'), cv2.FILE_STORAGE_READ)
    K = fs.getNode('camera_matrix').mat()
    R = fs.getNode('R_virtual_from_physical').mat()
    w = int(fs.getNode('image_width').real())
    fs.release()
    z = np.load(tmp_path / f'{cams[0].tag}_maps.npz')
    assert z['map_x'].shape == (int(z['map_y'].shape[0]), w) and z['map_x'].dtype == np.float32
    assert K.shape == (3, 3) and K[0, 0] > 0 and np.allclose(R @ R.T, np.eye(3), atol=1e-9)
    img = np.zeros((proj.module_for(cams[0]).height_px, proj.module_for(cams[0]).width_px, 3), np.uint8)
    out = cv2.remap(img, z['map_x'], z['map_y'], cv2.INTER_LINEAR)
    assert out.shape[:2] == z['map_x'].shape
