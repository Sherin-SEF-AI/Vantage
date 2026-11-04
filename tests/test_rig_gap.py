"""Cross-rig domain gap (core/rig_gap.py): zero against itself and symmetric, each component moving with the change it
measures (camera height, a narrower lens, turned cameras, a missing camera), the nuScenes and KITTI calibration
readers recovering the exported cameras exactly, the Hungarian camera matches, the JS distance and 1-Wasserstein
against closed forms, and the files."""
import copy
import json
import math
import os

import numpy as np
import pytest

from vantage.core import exporters as ex
from vantage.core import rig_gap as RG
from vantage.core import samples
from vantage.core.analysis import Engine


@pytest.fixture(scope='module')
def proj():
    p = samples.tigor_sample_project()
    p.active_uid = next(v.uid for v in p.variants if v.sensors)
    return p


def _rig(p, edit=None):
    q = copy.deepcopy(p)
    if edit:
        edit(q.variant)
    return RG.rig_from_variant(q)


def test_closed_forms():
    assert RG.js_distance([1, 0], [0, 1]) == pytest.approx(1.0)
    assert RG.js_distance([1, 2, 3], [2, 4, 6]) == pytest.approx(0.0, abs=1e-12)
    p, q = np.array([0.5, 0.5]), np.array([0.9, 0.1])
    m = (p + q) / 2
    js = 0.5 * np.sum(p * np.log2(p / m)) + 0.5 * np.sum(q * np.log2(q / m))
    assert RG.js_distance(p, q) == pytest.approx(math.sqrt(js))
    assert RG.wasserstein1([0, 1, 2], [0.5, 1.5, 2.5]) == pytest.approx(0.5, abs=1e-9)
    D = RG.fibonacci_sphere(2000)
    assert np.allclose(np.linalg.norm(D, axis=1), 1) and abs(D[:, 2].mean()) < 1e-3


def test_zero_against_itself_symmetric_and_occlusion(proj):
    A = RG.rig_from_variant(proj, engine=Engine())
    assert A.occlusion and len(A.cams) == 6
    g = RG.rig_gap(A, A, n_dirs=2048)
    assert g.total == pytest.approx(0.0, abs=1e-12) and all(v == pytest.approx(0, abs=1e-12)
                                                           for v in g.components.values())
    assert [(m['a'], m['b']) for m in g.matches] == [(c.tag, c.tag) for c in A.cams]
    B = _rig(proj, lambda v: [setattr(c, 'yaw', c.yaw + 20.0) for c in v.cameras])
    g1, g2 = RG.rig_gap(A, B, n_dirs=2048), RG.rig_gap(B, A, n_dirs=2048)
    for k in g1.components:
        assert g1.components[k] == pytest.approx(g2.components[k], abs=1e-9), k
    assert 0 < g1.total < 1 and set(g1.components) == set(RG.WEIGHTS)
    # the body hides what it hides: without occlusion the same rig sees more directions
    free = RG.rig_from_variant(proj)
    seen_occ = (RG.direction_profile(A, RG.fibonacci_sphere(2048))['count'] > 0).mean()
    seen_free = (RG.direction_profile(free, RG.fibonacci_sphere(2048))['count'] > 0).mean()
    assert seen_free >= seen_occ


def test_each_component_follows_its_change(proj):
    A = _rig(proj)
    up = RG.rig_gap(A, _rig(proj, lambda v: [setattr(c, 'z', c.z + 0.5) for c in v.cameras]), n_dirs=2048)
    assert up.figures['height_w1_m'] == pytest.approx(0.5, abs=1e-6)
    assert up.components['height_pitch'] == pytest.approx(0.25, abs=1e-6)     # 0.5 × (0.5 m / 1 m) + 0 pitch
    small = RG.rig_gap(A, _rig(proj, lambda v: [setattr(c, 'yaw', c.yaw + 5) for c in v.cameras]), n_dirs=2048)
    large = RG.rig_gap(A, _rig(proj, lambda v: [setattr(c, 'yaw', c.yaw + 45) for c in v.cameras]), n_dirs=2048)
    assert large.components['coverage'] > small.components['coverage'] > 0
    # a narrower lens on every camera: fewer directions, more pixels per radian where both see
    narrow = RG.rig_gap(A, _rig(proj, lambda v: [setattr(c, 'lens_id', 'h03r2') for c in v.cameras]), n_dirs=2048)
    assert narrow.components['resolution'] > 0.1 and narrow.figures['resolution_median_ratio_b_over_a'] > 1.2
    assert narrow.components['fov'] > 0.3
    # a missing camera
    fewer = RG.rig_gap(A, _rig(proj, lambda v: setattr(v, 'cameras', v.cameras[:5])), n_dirs=2048)
    assert fewer.figures['cameras'] == [6, 5] and fewer.components['topology'] >= 1 / 12
    assert len(fewer.matches) == 5 and fewer.transfer_band in ('small', 'moderate', 'large')


def test_readers_recover_the_exported_cameras(proj, tmp_path):
    ex.export(['nuscenes_calib', 'kitti_calib'], proj, None, ex.ExportOptions(out_dir=str(tmp_path)))
    A = RG.rig_from_variant(proj)
    for B in (RG.read_nuscenes_rig(str(tmp_path / 'nuscenes')), RG.read_kitti_rig(str(tmp_path / 'kitti')),
              RG.load_rig(str(tmp_path))):
        assert [c.tag for c in B.cams] == [c.tag for c in A.cams], B.source
        for a, b in zip(A.cams, B.cams):
            assert np.allclose(a.C, b.C, atol=1e-5) and np.allclose(a.R, b.R, atol=1e-6), (B.source, a.tag)
            assert b.intr.f_px == pytest.approx(a.intr.f_px, rel=1e-5)
            assert (b.intr.cx, b.intr.cy) == pytest.approx((a.intr.cx, a.intr.cy), abs=1e-4)
        g = RG.rig_gap(A, B, n_dirs=2048)
        assert g.total < 0.03 and g.components['height_pitch'] == pytest.approx(0, abs=1e-6), (B.source, g.components)


def test_kitti_raw_layout_through_the_velodyne(tmp_path):
    """KITTI's own raw files: camera 00 from calib_velo_to_cam, camera 02 offset by R_02/T_02, the Velodyne frame
    lowered by its 1.73 m height."""
    K = '7.215377e+02 0 6.095593e+02 0 7.215377e+02 1.728540e+02 0 0 1'
    (tmp_path / 'calib_cam_to_cam.txt').write_text(
        f'S_00: 1.392e+03 5.12e+02\nK_00: {K}\nD_00: 0 0 0 0 0\nR_00: 1 0 0 0 1 0 0 0 1\nT_00: 0 0 0\n'
        f'S_02: 1.392e+03 5.12e+02\nK_02: {K}\nD_02: 0 0 0 0 0\nR_02: 1 0 0 0 1 0 0 0 1\nT_02: 0.06 0 0\n')
    # velodyne x forward, y left, z up -> camera x right (= -y), y down (= -z), z forward (= x); camera 0.27 m ahead
    (tmp_path / 'calib_velo_to_cam.txt').write_text('R: 0 -1 0 0 0 -1 1 0 0\nT: 0 -0.08 -0.27\n')
    B = RG.read_kitti_rig(str(tmp_path))
    c0, c2 = B.cams
    assert np.allclose(c0.R[:, 2], [1, 0, 0], atol=1e-9) and np.allclose(c0.C, [0.27, 0.0, 1.73 - 0.08], atol=1e-9)
    assert np.allclose(c2.C - c0.C, [0.0, 0.06, 0.0], atol=1e-9)          # X_02 = X_00 + T_02: 0.06 m to the left
    assert c0.intr.projection == 'rectilinear' and c0.intr.f_px == pytest.approx(721.5377)


def test_document_and_files(proj, tmp_path):
    A = _rig(proj)
    B = _rig(proj, lambda v: [setattr(c, 'pitch', c.pitch + 10) for c in v.cameras])
    paths = RG.export_rig_gap(A, B, str(tmp_path), n_dirs=1024)
    assert [os.path.basename(x) for x in paths] == ['rig_gap.json', 'rig_gap.png']
    doc = json.loads((tmp_path / 'rig_gap.json').read_text())
    assert doc['schema'] == RG.SCHEMA and 'not a predicted mAP' in doc['transfer_band_note']
    assert doc['figures']['depression_w1_deg'] == pytest.approx(10.0, abs=1e-6)
    assert len(doc['rig_a']['per_camera']) == 6 and doc['findings']
    import cv2
    img = cv2.imread(str(tmp_path / 'rig_gap.png'))
    assert img is not None and img.shape[1] == 1400
