"""Lidar–camera calibration: the frames and the solve's maths against known values, the pose plan, synthetic captures
of the data rig's camera F and roof lidar L1 with an injected mounting error recovered, a plan-only capture (no error),
a degenerate set of boards flagged, a board the lidar does not see and a moved board set aside, every point-cloud
format written and read back, the real-data path from a folder of pairs, apply and the calibrated variant, the export,
and the page driven like a user (offscreen)."""
import copy
import json
import math
import os
import struct

import numpy as np
import pytest

from vantage.core import lidar_calib as LC, samples, sensors as S

ERR = dict(err_x_mm=15.0, err_y_mm=-10.0, err_z_mm=8.0, err_yaw_deg=0.6, err_pitch_deg=-0.4, err_roll_deg=0.3)
POS_MM, ANG_DEG = 3.0, 0.08


def _rig():
    """The sample project with its data rig (6 cameras, roof lidar L1, radars) active."""
    p = samples.tigor_sample_project()
    p.active_uid = next(v for v in p.variants if v.sensors).uid
    return p


def _pair(p):
    return p.variant.by_tag('F'), next(s for s in p.variant.sensors if s.tag == 'L1')


@pytest.fixture(scope='module')
def eng():
    from vantage.core.analysis import Engine
    return Engine()


@pytest.fixture(scope='module')
def rig():
    return _rig()


def _wrong_pose(p):
    """A board low in front of the bumper, facing camera F: in its image, but L1 on the roof does not see it."""
    cam, _ = _pair(p)
    C = cam.pos()
    c = np.array([C[0] + 0.75, 0.0, 0.3])
    return LC.BoardPose(c, LC._facing(c, C), (0.0, 0.0, 0.0), float(np.linalg.norm(c - C)))


@pytest.fixture(scope='module')
def main(rig, eng):
    """The six planned poses and the wrong-pairing board, captured with L1 mounted off its plan by ERR."""
    st = LC.LidarCalibSettings.from_dict(dict(ERR, camera='F', lidar='L1', seed=4))
    cam, lid = _pair(rig)
    poses, notes = LC.plan_poses(rig, cam, lid, st, eng)
    res = LC.run(rig, eng, st, poses=poses + [_wrong_pose(rig)])
    return st, poses, res


def _truth_error(p, res, st):
    cam, lid = _pair(p)
    return LC._delta(cam, res.T, LC.plan_T(cam, LC.true_lidar(lid, st)))


# =========================================================================== maths
def test_settings_are_tolerant_and_clamped():
    st = LC.LidarCalibSettings.from_dict({'cols': '9', 'rows': 1, 'square_mm': 'junk', 'margin_mm': -5, 'poses': 50,
                                          'near_m': 9.0, 'far_m': 4.0, 'source': 'FOLDER', 'scene': 'moon',
                                          'bin_fields': 5, 'edges': 'off', 'reflectivity': 0, 'err_yaw_deg': 99,
                                          'camera': '  F ', 'seed': 3.6, 'pos_tol_mm': float('nan')})
    assert (st.cols, st.rows, st.square_mm, st.margin_mm, st.poses) == (9, 2, 100.0, 0.0, 20)
    assert (st.near_m, st.far_m) == (4.0, 9.0)                       # swapped into order
    assert (st.source, st.scene, st.bin_fields) == ('folder', 'plain', '5')
    assert not st.edges and not st.reflectivity and st.err_yaw_deg == 20.0 and st.camera == 'F' and st.seed == 4
    assert st.pos_tol_mm == 10.0
    assert LC.LidarCalibSettings.from_dict(None).to_dict() == LC.LidarCalibSettings().to_dict()
    assert LC.LidarCalibSettings.from_dict(st.to_dict()) == st
    # 7 × 5 inner corners of 100 mm and a 50 mm margin: 8 × 6 squares, 0.9 × 0.7 m
    assert LC.LidarCalibSettings().board_size() == pytest.approx((0.9, 0.7))


def test_frames_round_trip(rig):
    cam, lid = _pair(rig)
    T = LC.plan_T(cam, lid)
    # T maps the lidar's origin to where the vehicle frame puts it, seen from the camera's optical frame
    W2O = LC.world_to_optical(cam)
    assert T[:3, 3] == pytest.approx(W2O @ (lid.pos() - cam.pos()))
    P = np.array([[5.0, 1.0, -0.3], [12.0, -3.0, 2.0]])                      # lidar-frame points
    world = P @ S.rotation(lid).T + lid.pos()
    assert P @ T[:3, :3].T + T[:3, 3] == pytest.approx((world - cam.pos()) @ W2O.T)
    R, pos = LC.lidar_pose(cam, T)
    d = LC.pose_dict(R, pos)
    assert (d['x'], d['y'], d['z'], d['yaw'], d['pitch'], d['roll']) == pytest.approx(
        (lid.x, lid.y, lid.z, lid.yaw, lid.pitch, lid.roll), abs=1e-9)
    assert LC._inv(T) @ T == pytest.approx(np.eye(4), abs=1e-12)
    assert LC._exp([0, 0, math.pi / 2]) @ [1, 0, 0] == pytest.approx([0, 1, 0])
    # a lidar mounted off its plan by ERR
    st = LC.LidarCalibSettings.from_dict(ERR)
    t = LC.true_lidar(lid, st)
    assert (t.x - lid.x, t.y - lid.y, t.z - lid.z) == pytest.approx((0.015, -0.010, 0.008))
    assert (t.yaw - lid.yaw, t.pitch - lid.pitch, t.roll - lid.roll) == pytest.approx((0.6, -0.4, 0.3))


def test_known_values_of_the_residuals():
    # signed distance to a 2 × 1 rectangle
    sdf = LC._box_sdf(np.array([0.0, 1.5, 0.0, 1.3, 0.9]), np.array([0.0, 0.0, 0.8, 0.9, 0.4]), 1.0, 0.5)
    assert sdf == pytest.approx([-0.5, 0.5, 0.3, math.hypot(0.3, 0.4), -0.1])
    # LZF: a literal run, then a back reference that copies over itself
    assert LC._lzf_decompress(bytes([2]) + b'abc' + bytes([0x80, 2]), 9) == b'abcabcabc'
    with pytest.raises(ValueError):
        LC._lzf_decompress(bytes([2]) + b'abc', 9)
    # the checker: a dark point in the dark (0, 0) square is right; in the white (1, 0) square it is 20 mm from the
    # nearest inner edge; a bright point on the margin is right, a dark one there must reach the checker
    st = LC.LidarCalibSettings()
    sq = 0.1
    a0, b0 = -4 * sq, -3 * sq                                              # the checker's (−x, −y) corner
    pts = np.array([[a0 + 0.05, b0 + 0.05, 0], [a0 + 0.12, b0 + 0.05, 0], [a0 - 0.03, b0 + 0.05, 0],
                    [a0 - 0.03, b0 + 0.05, 0]])
    dark = np.array([True, True, False, True])
    miss = LC._checker_miss(pts, dark, np.eye(3), np.zeros(3), np.eye(3), np.zeros(3), st, 0)
    assert miss == pytest.approx([0.0, 0.02, 0.0, 0.03])
    # the other colouring flips which squares are dark
    assert LC._checker_miss(pts[:2], dark[:2], np.eye(3), np.zeros(3), np.eye(3), np.zeros(3), st, 1)[0] > 0


def test_closed_form_recovers_a_known_transform():
    """Exact planes: the camera's board planes and the lidar's, related by a known T."""
    T_true = LC._T(LC._exp([0.02, -0.05, 0.3]), np.array([0.3, -1.2, 2.1]))
    rng = np.random.default_rng(0)
    obs = []
    for n_dir in ([0.2, 0.1, -1.0], [-0.5, 0.0, -1.0], [0.1, 0.6, -1.0], [0.4, -0.4, -1.0]):
        n = np.asarray(n_dir) / np.linalg.norm(n_dir)
        c = -n * rng.uniform(4, 8) + rng.normal(0, 0.5, 3)
        u = np.cross(n, [0, 1.0, 0])
        u /= np.linalg.norm(u)
        R = np.stack([u, np.cross(n, u), n], 1)
        cam = LC.CamBoard(np.zeros((1, 2)), np.zeros((1, 3)), R, c, 0.1, np.eye(6))
        Pc = c + (rng.uniform(-0.4, 0.4, (200, 1)) * R[:, 0] + rng.uniform(-0.3, 0.3, (200, 1)) * R[:, 1])
        Pl = (Pc - T_true[:3, 3]) @ T_true[:3, :3]                         # into the lidar frame
        nl, dl = LC._fit_plane(Pl)
        lid = LC.LidarBoard(Pl, nl, dl, 0.0, np.zeros((0, 3)), np.zeros((0, 3)), 0)
        obs.append(LC.PoseObs('p', cam=cam, lid=lid))
    T = LC._closed_form(obs)
    assert T == pytest.approx(T_true, abs=1e-9)
    assert LC._closed_form(obs[:2]) is None                                 # two planes do not fix it


def test_pairs_are_ranked_and_picked(rig, eng):
    ranked = LC.pairs(rig, None, eng)
    assert len(ranked) == 6 and {r['lidar'] for r in ranked} == {'L1'}
    assert all(0 < r['share'] <= 1 for r in ranked)
    assert [r['share'] for r in ranked] == sorted([r['share'] for r in ranked], reverse=True)
    cam, lid, share = LC.pick_pair(rig, None, LC.LidarCalibSettings.from_dict({'camera': 'F', 'lidar': 'L1'}), eng)
    assert (cam.tag, lid.tag) == ('F', 'L1') and share == next(r['share'] for r in ranked if r['camera'] == 'F')
    best = LC.pick_pair(rig, None, LC.LidarCalibSettings(), eng)
    assert best[0].tag == ranked[0]['camera']
    with pytest.raises(ValueError, match='No camera tagged Q9'):
        LC.pick_pair(rig, None, LC.LidarCalibSettings.from_dict({'camera': 'Q9'}))
    p = _rig()
    p.active_uid = p.variants[0].uid                                        # cameras only
    with pytest.raises(ValueError, match='has no lidar'):
        LC.pick_pair(p)


def _check_plan(p, cam, lid, poses, st):
    """Independent checks of planned poses: the whole board in the camera's image and in the lidar's field, enough
    channels across it, within the distances, above the ground, apart from each other."""
    from vantage.core.lens import resolve
    m = p.sensor_model_for(lid)
    intr = resolve(p.module_for(cam), p.lens_for(cam))
    W, H = st.board_size()
    E = m.elevations()
    lo, hi = m.v_fov()
    for b in poses:
        out = b.centre + LC.outline_local(W, H, 4) @ b.R.T
        _, _, _, ok = intr.project_dirs((out - cam.pos()) @ LC.world_to_optical(cam).T)
        assert ok.all()
        r, az, el = S.angles(lid, out)
        assert ((el > lo) & (el < hi)).all() and (r < m.range_m).all()
        top = b.centre + b.R @ [0, H / 2, 0]
        bottom = b.centre + b.R @ [0, -H / 2, 0]
        e_top, e_bottom = S.angles(lid, np.array([top, bottom]))[2]
        assert ((E >= min(e_top, e_bottom)) & (E <= max(e_top, e_bottom))).sum() >= LC.MIN_RINGS
        assert st.near_m - 0.5 <= np.linalg.norm(b.centre - cam.pos()) <= st.far_m + 0.5
        assert out[:, 2].min() >= LC.BOARD_MIN_Z - 1e-9 and np.isclose(np.linalg.norm(b.R[:, 2]), 1)
    for i in range(len(poses)):
        for j in range(i):
            assert np.linalg.norm(poses[i].centre - poses[j].centre) >= LC.MIN_BOARD_GAP_M - 1e-9


def test_the_plan_is_whole_in_both_and_varied(rig, eng):
    cam, lid = _pair(rig)
    st = LC.LidarCalibSettings()
    poses, notes = LC.plan_poses(rig, cam, lid, st, eng)
    assert len(poses) == 6
    _check_plan(rig, cam, lid, poses, st)
    N = np.array([b.R[:, 2] for b in poses])
    assert np.linalg.svd(N, compute_uv=False)[-1] > 0.1                     # the normals span three directions
    assert len({b.tilt for b in poses}) >= 4
    # the planes alone, each as the lidar's points and as the camera's corners (0.3 px) measure it: the camera's
    # distance to a board 4–8 m out limits z to about 16 mm (it took the camera as good to 1 mm and said 3.4 mm; the
    # spread of the solve's answers is 16.6 mm, tests/test_calib_accuracy.py); the angles to a few tenths of a degree
    pr = LC.predicted_sigma(rig, cam, lid, poses, st)
    assert max(pr['x_mm'], pr['y_mm']) < 10 and 10 < pr['z_mm'] < 25, pr
    assert max(pr['roll_deg'], pr['pitch_deg'], pr['yaw_deg']) < 0.3, pr
    # a coarser lidar (2° between channels against 0.35°): the boards come nearer so that four channels still cross
    # them (camera F tilted up here: looking 40° down it sees too little above the horizon for any board to be
    # crossed by four of a VLP-16's channels, and the plan says so)
    with pytest.raises(ValueError, match='4 or more of L1'):
        q = _rig()
        _pair(q)[1].model_id = 'velodyne_vlp16'
        LC.plan_poses(q, *_pair(q), st, eng)
    far = {}
    for model in ('ouster_os1_128', 'velodyne_vlp16'):
        q = _rig()
        qc, ql = _pair(q)
        qc.pitch, ql.model_id = 10.0, model
        ps, nt = LC.plan_poses(q, qc, ql, st, eng)
        _check_plan(q, qc, ql, ps, st)
        far[model] = max(np.linalg.norm(b.centre - ql.pos()) for b in ps)
    assert far['velodyne_vlp16'] < far['ouster_os1_128'] - 2.0 and 'No board fits beyond' in nt[0]
    # four channels 2° apart span at least 6°: no farther than half the board's height over tan 3°
    assert far['velodyne_vlp16'] <= st.board_size()[1] / 2 / math.tan(math.radians(3.0)) + 0.5
    # more poses when asked
    assert len(LC.plan_poses(rig, cam, lid, LC.LidarCalibSettings.from_dict({'poses': 9}), eng)[0]) == 9


# =========================================================================== the synthetic calibration
def test_injected_mounting_error_is_recovered(main, rig):
    st, poses, res = main
    assert res.verdict == 'warning' and len(res.used) == 6                  # 15 mm off: within 3 × the tolerance
    e = _truth_error(rig, res, st)
    assert max(abs(e['vehicle'][k]) for k in ('x_mm', 'y_mm', 'z_mm')) <= POS_MM
    assert e['rotation_deg'] <= ANG_DEG
    assert res.truth['error_position_mm'] == pytest.approx(e['position_mm'])
    # Δ against the plan is the injected error, in the vehicle frame
    d = res.delta['vehicle']
    assert (d['x_mm'], d['y_mm'], d['z_mm']) == pytest.approx((15.0, -10.0, 8.0), abs=POS_MM)
    assert (d['yaw_deg'], d['pitch_deg'], d['roll_deg']) == pytest.approx((0.6, -0.4, 0.3), abs=ANG_DEG)
    # ... and in the camera frame it is the same move seen from the camera
    cam, lid = _pair(rig)
    W2O = LC.world_to_optical(cam)
    assert np.array([res.delta['camera'][k] for k in ('x_mm', 'y_mm', 'z_mm')]) == pytest.approx(
        W2O @ np.array([d['x_mm'], d['y_mm'], d['z_mm']]), abs=0.5)
    # the 1σ is a sensible size and on the safe side of the actual error
    sg = res.sigma['vehicle']
    assert 0 < max(sg['x_mm'], sg['y_mm'], sg['z_mm']) < st.pos_tol_mm and res.sigma['angle_deg'] < st.ang_tol_deg
    assert not res.obs['degenerate'] and res.fit['poses_with_reflectivity_checker'] == 6
    for o in res.used:
        assert o.cam.rms_px < 0.3 and o.lid.rms_m < 0.02 and len(o.lid.points) > 80 and o.lid.rings >= 4
        assert len(o.lid.edge_in) and len(o.lid.edge_out) and o.stats['edge_mm'] < 2.0


def test_a_board_out_of_the_lidars_sight_is_handled(main, rig):
    st, poses, res = main
    o = res.poses[-1]
    assert o.status == 'no_lidar' and o.cam is not None and 'L1 did not see the board' in o.message
    # independently: the scan has no point on that board
    b = res.captures[-1].board
    t = LC.true_lidar(_pair(rig)[1], st)
    P = res.captures[-1].points[:, :3] @ S.rotation(t).T + t.pos()
    q = (P - b.centre) @ b.R
    W, H = st.board_size()
    assert not ((np.abs(q[:, 2]) < 0.05) & (np.abs(q[:, 0]) < W / 2) & (np.abs(q[:, 1]) < H / 2)).any()
    assert any('pose07' in f['message'] for f in res.findings)


def test_the_plan_alone_measures_no_error(rig, eng, main):
    st = LC.LidarCalibSettings.from_dict({'camera': 'F', 'lidar': 'L1', 'seed': 4})
    res = LC.run(rig, eng, st, poses=main[1])
    assert res.verdict == 'good'
    assert res.delta['position_mm'] <= POS_MM and res.delta['rotation_deg'] <= ANG_DEG
    assert res.measured['x'] == pytest.approx(_pair(rig)[1].x, abs=POS_MM / 1000)


def test_boards_all_tilted_the_same_way_are_unobservable(rig, eng, main):
    poses = main[1]
    same = [LC.BoardPose(b.centre.copy(), poses[0].R.copy(), poses[0].tilt, b.distance) for b in poses]
    st = LC.LidarCalibSettings.from_dict(dict(ERR, camera='F', lidar='L1', seed=4))
    res = LC.run(rig, eng, st, poses=same)
    assert res.verdict == 'unobservable' and res.obs['degenerate']
    assert 'tilted the same way' in res.message and 'not determined' in res.message
    assert any(w['level'] == 'none' and w['kind'] == 'translation' for w in res.obs['weak'])
    assert res.findings[0]['severity'] == 'critical'
    s = LC.summary(res)
    assert s['verdict'] == 'unobservable' and s['not_determined']


def test_a_moved_board_is_set_aside(main, rig):
    st, poses, res0 = main
    caps = [copy.copy(c) for c in res0.captures[:6]]
    caps[2].points = caps[2].points.copy()
    caps[2].points[:, 0] += 0.08                                            # the board moved 8 cm between the frames
    cam, lid = _pair(rig)
    res = LC.calibrate(rig, cam, lid, caps, st, poses)
    assert [o.status for o in res.poses].count('outlier') == 1 and res.poses[2].status == 'outlier'
    assert 'Set aside' in res.poses[2].message
    e = _truth_error(rig, res, st)
    assert max(abs(e['vehicle'][k]) for k in ('x_mm', 'y_mm', 'z_mm')) <= 2 * POS_MM and e['rotation_deg'] <= ANG_DEG


def test_cancel_and_an_empty_variant(rig, eng):
    assert LC.run(rig, eng, LC.LidarCalibSettings.from_dict({'camera': 'F', 'lidar': 'L1'}),
                  cancel=lambda: True) is None
    p = _rig()
    p.active_uid = p.variants[0].uid
    with pytest.raises(ValueError, match='has no lidar'):
        LC.run(p, eng)


# =========================================================================== point clouds
def _lzf_literal(raw: bytes) -> bytes:
    """LZF with literal runs only (valid LZF; what a minimal compressor may write)."""
    out = bytearray()
    for i in range(0, len(raw), 32):
        chunk = raw[i:i + 32]
        out.append(len(chunk) - 1)
        out += chunk
    return bytes(out)


def test_every_point_cloud_format_round_trips(tmp_path):
    rng = np.random.default_rng(1)
    P = np.c_[rng.uniform(-20, 20, (301, 3)), rng.uniform(0, 100, 301)].astype(np.float32).astype(float)
    for name, binary in (('a.pcd', True), ('b.pcd', False), ('c.bin', True), ('d.pcd.bin', True), ('e.ply', True),
                         ('f.ply', False), ('g.npy', True), ('h.csv', True), ('i.txt', True)):
        path = str(tmp_path / name)
        LC.write_cloud(path, P, binary)
        Q = LC.read_cloud(path)
        assert Q.shape == (301, 4), name
        assert Q == pytest.approx(P, abs=1e-5), name
    # binary_compressed PCD (field by field, LZF) with an extra unsigned field
    ring = np.arange(301, dtype='<u2')
    soa = b''.join(P[:, i].astype('<f4').tobytes() for i in range(4)) + ring.tobytes()
    head = ('VERSION 0.7\nFIELDS x y z intensity ring\nSIZE 4 4 4 4 2\nTYPE F F F F U\nCOUNT 1 1 1 1 1\n'
            'WIDTH 301\nHEIGHT 1\nPOINTS 301\nDATA binary_compressed\n').encode()
    comp = _lzf_literal(soa)
    (tmp_path / 'j.pcd').write_bytes(head + struct.pack('<II', len(comp), len(soa)) + comp)
    assert LC.read_cloud(str(tmp_path / 'j.pcd')) == pytest.approx(P, abs=1e-5)
    # binary PCD with mixed field types and padding, an organised cloud's empty returns (NaN, 0 0 0) dropped
    dt = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('_', '<u1', (4,)), ('reflectivity', '<u2')])
    rec = np.zeros(4, dt)
    rec['x'], rec['y'], rec['z'] = [1, np.nan, 0, 3], [2, 1, 0, 4], [3, 1, 0, 5]
    rec['reflectivity'] = [10, 20, 30, 40]
    head = ('FIELDS x y z _ reflectivity\nSIZE 4 4 4 1 2\nTYPE F F F U U\nCOUNT 1 1 1 4 1\nWIDTH 2\nHEIGHT 2\n'
            'DATA binary\n').encode()
    (tmp_path / 'k.pcd').write_bytes(head + rec.tobytes())
    assert LC.read_cloud(str(tmp_path / 'k.pcd')) == pytest.approx(np.array([[1, 2, 3, 10], [3, 4, 5, 40.0]]))
    # PLY big-endian with a face element after the vertices
    body = np.array([(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)], dtype=[('x', '>f4'), ('y', '>f4'), ('z', '>f4')])
    head = (b'ply\nformat binary_big_endian 1.0\nelement vertex 2\nproperty float x\nproperty float y\n'
            b'property float z\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n')
    (tmp_path / 'l.ply').write_bytes(head + body.tobytes() + bytes([3]) + np.array([0, 1, 1], '>i4').tobytes())
    assert LC.read_cloud(str(tmp_path / 'l.ply')) == pytest.approx(np.array([[1, 2, 3], [4, 5, 6.0]]))
    # a structured .npy, a whitespace table without a header, nuScenes points without the .pcd.bin name
    st = np.zeros(3, [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('intensity', 'f4')])
    st['x'], st['y'], st['z'], st['intensity'] = [1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 1, 2]
    np.save(tmp_path / 'm.npy', st)
    assert LC.read_cloud(str(tmp_path / 'm.npy'))[:, :3] == pytest.approx(np.array([[1, 4, 7], [2, 5, 8], [3, 6, 9.0]]))
    np.savetxt(tmp_path / 'n.txt', P[:5, :3], fmt='%.5f')
    assert LC.read_cloud(str(tmp_path / 'n.txt')) == pytest.approx(P[:5, :3], abs=1e-4)
    np.c_[P[:3], np.zeros(3)].astype('<f4').tofile(tmp_path / 'o.bin')      # 3 × 20 bytes: not KITTI
    assert LC.read_cloud(str(tmp_path / 'o.bin')) == pytest.approx(P[:3], abs=1e-5)


def test_unreadable_clouds_say_why(tmp_path):
    cases = {'a.pcd': b'VERSION 0.7\nFIELDS x y z\n', 'b.bin': b'\0' * 7, 'c.xyz': b'1 2 3\n',
             'd.ply': b'ply\nformat ascii 1.0\nelement vertex 1\nproperty list uchar int idx\nend_header\n3 0 1 2\n',
             'e.csv': b'a,b,c\nx,y,z\n', 'g.pcd': b'FIELDS a b c\nSIZE 4 4 4\nTYPE F F F\nWIDTH 1\nHEIGHT 1\n'
                                                b'DATA ascii\n1 2 3\n'}
    msgs = {'a.pcd': 'no DATA line', 'b.bin': 'not a whole number', 'c.xyz': 'not a point cloud format',
            'd.ply': 'list properties', 'e.csv': 'not a table of numbers', 'g.pcd': 'no x/y/z field'}
    for name, data in cases.items():
        (tmp_path / name).write_bytes(data)
        with pytest.raises(ValueError, match=msgs[name]):
            LC.read_cloud(str(tmp_path / name))
    np.save(tmp_path / 'f.npy', np.zeros(5))
    with pytest.raises(ValueError, match='expected an N × 3'):
        LC.read_cloud(str(tmp_path / 'f.npy'))
    np.zeros((3, 5), '<f4').tofile(tmp_path / 'h.bin')                    # 60 bytes read as KITTI
    with pytest.raises(ValueError, match='4-float'):
        LC.read_bin(str(tmp_path / 'h.bin'), '4')
    with pytest.raises(ValueError, match='No image \\+ point cloud pairs'):
        LC.load_folder(str(tmp_path))


def test_a_folder_of_real_pairs_calibrates(main, rig, tmp_path):
    """The real-data path: the synthetic pairs written as files in several formats and calibrated with no plan to
    say where the boards are."""
    st, poses, res0 = main
    LC.save_captures(res0.captures, str(tmp_path), '.pcd')
    for k, ext in ((2, '.bin'), (3, '.ply'), (4, '.npy'), (5, '.csv'), (6, '.pcd.bin')):
        name = f'pose{k:02d}'
        os.remove(tmp_path / (name + '.pcd'))
        LC.write_cloud(str(tmp_path / (name + ext)), res0.captures[k - 1].points)
    (tmp_path / 'notes.png').write_bytes((tmp_path / 'pose01.png').read_bytes())     # no partner: noted
    caps, notes = LC.load_folder(str(tmp_path), st)
    assert [c.name for c in caps] == [f'pose{k:02d}' for k in range(1, 8)] and 'notes' in notes[0]
    assert all(c.error == '' and c.image.shape == (1536, 1920, 3) for c in caps)
    res = LC.calibrate_folder(rig, 'F', 'L1', str(tmp_path), st)
    assert res.source == str(tmp_path) and len(res.used) == 6 and res.poses[6].status == 'no_lidar'
    assert all(o.cam.how in ('whole image', 'view search') for o in res.used)
    e = _truth_error(rig, res, st)
    assert max(abs(e['vehicle'][k]) for k in ('x_mm', 'y_mm', 'z_mm')) <= POS_MM and e['rotation_deg'] <= ANG_DEG
    # a pair that cannot be read is kept with its error
    (tmp_path / 'pose01.pcd').write_bytes(b'junk')
    caps, _ = LC.load_folder(str(tmp_path), st)
    assert 'PCD header' in caps[0].error


# =========================================================================== apply, export
def test_apply_and_a_calibrated_variant(main):
    st, poses, res = main
    p = _rig()
    cam, lid = _pair(p)
    before = p.variant
    assert LC.apply_to_variant(p.variant, res)
    m = res.measured
    assert (lid.x, lid.y, lid.z) == pytest.approx((m['x'], m['y'], m['z']), abs=1e-4)
    true = LC.true_lidar(_pair(_rig())[1], st)
    assert (lid.x, lid.y, lid.z) == pytest.approx((true.x, true.y, true.z), abs=POS_MM / 1000)
    assert (lid.yaw, lid.pitch, lid.roll) == pytest.approx((true.yaw, true.pitch, true.roll), abs=ANG_DEG)
    q = _rig()                       # another project made from the sample: the lidar is found by its tag
    n = len(q.variants)
    old = q.variant
    v = LC.calibrated_variant(q, res)
    assert len(q.variants) == n + 1 and q.active_uid == v.uid and v.name == f'{old.name} · lidar calibrated'
    assert not ({c.uid for c in v.cameras} & {c.uid for c in old.cameras})
    assert not ({s.uid for s in v.sensors} & {s.uid for s in old.sensors})
    uids = {c.uid for c in v.cameras}
    assert all(c.link_uid in uids for c in v.cameras if c.link_uid)
    nl = next(s for s in v.sensors if s.tag == 'L1')
    ol = next(s for s in old.sensors if s.tag == 'L1')
    assert nl.x == pytest.approx(m['x'], abs=1e-4) and ol.x == _pair(_rig())[1].x       # the original untouched
    failed = copy.copy(res)
    failed.T, failed.verdict = None, 'failed'
    assert not LC.apply_to_variant(before, failed) and LC.calibrated_variant(_rig(), failed) is None


def test_doc_files_and_export(main, rig, tmp_path):
    st, poses, res = main
    doc = json.loads(json.dumps(LC.to_doc(res, rig)))
    assert doc['schema'] == LC.SCHEMA and doc['frame'].startswith('vehicle') and 'OpenCV' in doc['camera_frame']
    assert doc['pair'] == {**doc['pair'], 'camera': 'F', 'lidar': 'L1'} and len(doc['poses']) == 7
    T = np.array(doc['solution']['T_camera_lidar']['matrix'])
    Ti = np.array(doc['solution']['T_lidar_camera']['matrix'])
    assert T @ Ti == pytest.approx(np.eye(4), abs=1e-6)
    assert np.linalg.norm(doc['solution']['T_camera_lidar']['quaternion_xyzw']) == pytest.approx(1)
    assert doc['solution']['delta_vs_plan']['vehicle_frame']['x_mm'] == pytest.approx(15, abs=POS_MM)
    assert doc['synthetic_truth']['injected_error']['err_x_mm'] == 15.0 and doc['verdict'] == 'warning'
    assert doc['poses'][0]['lidar']['edge_points_inside'] > 0 and doc['poses'][6]['status'] == 'no_lidar'
    assert len(np.array(doc['solution']['covariance']['matrix'])) == 6
    names = [n for n, _ in LC.files(res, rig)]
    assert names == ['lidar_camera_calibration.json'] + [f'pose{k:02d}_overlay.png' for k in range(1, 8)]
    import cv2
    png = dict(LC.files(res, rig))['pose01_overlay.png']
    img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
    assert img.shape == (542, 1030, 3)
    # the solved overlay puts the board's points on the board the camera saw; the plan's (15 mm, 0.8° off) less so
    full = LC.overlay_image(rig, res, 0, 'solved', crop=False)
    assert full.shape == res.captures[0].image.shape
    paths = LC.export(res, rig, str(tmp_path))
    assert len(paths) == 8 and all(os.path.getsize(x) > 0 for x in paths)
    s = LC.summary(res)
    assert len(json.dumps(s)) < 4000 and s['poses']['used'] == 6 and s['verdict'] == 'warning'
    assert abs(s['synthetic_error']['position_mm']) <= POS_MM


def _inside(pts, poly):
    """Even-odd point-in-polygon for many points."""
    x, y = pts[:, 0], pts[:, 1]
    inside = np.zeros(len(pts), bool)
    for i in range(len(poly)):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % len(poly)]
        with np.errstate(divide='ignore', invalid='ignore'):
            inside ^= ((y1 > y) != (y2 > y)) & (x < x1 + (y - y1) * (x2 - x1) / (y2 - y1))
    return inside


def test_the_lidar_board_lands_on_the_cameras_board(main, rig):
    """Independent of the solve's residuals: each board's lidar points, put through the solved extrinsic and the lens
    model, fall inside the board's outline as the camera measured it in the image; through the plan (15 mm, 0.8° off)
    clearly fewer do. The overlay's projection agrees."""
    st, poses, res = main
    from vantage.core.lens import resolve
    cam, _ = _pair(rig)
    intr = resolve(rig.module_for(cam), rig.lens_for(cam))
    W, H = st.board_size()
    share = {}
    for name, T in (('plan', res.T_plan), ('solved', res.T)):
        n_in = n_all = 0
        for o in res.used:
            Q = o.lid.points @ T[:3, :3].T + T[:3, 3]
            u, v, _, ok = intr.project_dirs(Q)
            ou, ov, _, _ = intr.project_dirs(o.cam.c + LC.outline_local(W, H, 40) @ o.cam.R.T)
            n_in += int(_inside(np.c_[u, v][ok], np.c_[ou, ov]).sum())
            n_all += len(Q)
        share[name] = n_in / n_all
    assert share['solved'] > 0.98 and share['plan'] < share['solved'] - 0.05
    uv, rng, idx = LC.project_points(rig, res, 0, 'solved')
    h, w = res.captures[0].image.shape[:2]
    assert len(uv) > 1000 and (uv >= 0).all() and (uv[:, 0] <= w).all() and (uv[:, 1] <= h).all()
    assert rng == pytest.approx(np.linalg.norm(res.captures[0].points[idx, :3], axis=1))


# =========================================================================== the page
from test_gui import app, win, pump, settle, _wait     # noqa: E402,F401  (fixtures)


def _open(win, name, cls):
    if name not in win._page_widgets:
        pg = cls(win.ctx)
        win.pages.addWidget(pg)
        win._page_widgets[name] = pg
    win.set_page(name)
    return win._page_widgets[name]


def _type_into_field(app, f, text):
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    QTest.mouseClick(f.lineEdit(), Qt.MouseButton.LeftButton)
    QTest.keyClick(f.lineEdit(), Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
    QTest.keyClicks(f.lineEdit(), text)
    QTest.keyClick(f.lineEdit(), Qt.Key.Key_Return)
    pump(app, 0.1)


def _until(app, cond, timeout=240):
    import time
    t = time.time()
    while not cond() and time.time() - t < timeout:
        pump(app, 0.1)
    return cond()


def test_page_captures_solves_applies_and_undoes(app, win, tmp_path):
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    from vantage.ui.workspaces.lidarcalib_ws import LidarCalibPage
    store = win.store
    win.switch_variant(next(v for v in store.project.variants if v.sensors).uid)
    pump(app, 0.2)
    pg = _open(win, 'lidar_calib', LidarCalibPage)
    assert pg.isVisible() and store.project.analysis.lidar_calib == {}
    # the pair, picked from the lists
    for box, tag in ((pg.cam_box, 'F'), (pg.lid_box, 'L1')):
        i = box.findData(tag)
        assert i > 0
        box.setCurrentIndex(i)
        box.activated.emit(i)
        pump(app, 0.05)
    assert store.project.analysis.lidar_calib['camera'] == 'F' and store.project.analysis.lidar_calib['lidar'] == 'L1'
    # the injected mounting error and the seed, typed into their fields
    for key, text in (('err_x_mm', '15'), ('err_y_mm', '-10'), ('err_z_mm', '8'), ('err_yaw_deg', '0.6'),
                      ('err_pitch_deg', '-0.4'), ('err_roll_deg', '0.3'), ('seed', '4')):
        _type_into_field(app, pg.fields[key], text)
    st = LC.LidarCalibSettings.from_dict(store.project.analysis.lidar_calib)
    assert (st.err_x_mm, st.err_pitch_deg, st.seed) == (15.0, -0.4, 4)
    # nonsense is clamped; undo takes it back
    _type_into_field(app, pg.fields['poses'], '1')
    assert store.project.analysis.lidar_calib['poses'] == 3 and pg.fields['poses'].value() == 3
    store.undo.undo()
    pump(app, 0.1)
    assert LC.LidarCalibSettings.from_dict(store.project.analysis.lidar_calib).poses == 6
    # the plan shows (planned in the background)
    assert _until(app, lambda: pg.plan is not None and len(pg.plan.get('poses', [])) == 6, 60)
    assert pg.plan_view.poses and 'pin L1' in pg.plan_note.text()
    pg.plan_view.grab()
    # capture and solve: a main-window job
    QTest.mouseClick(pg.b_run, Qt.MouseButton.LeftButton)
    assert _until(app, lambda: pg.result is not None and not pg.busy)
    res = pg.result
    assert res.verdict == 'warning' and len(res.used) == 6
    e = _truth_error(store.project, res, st)
    assert max(abs(e['vehicle'][k]) for k in ('x_mm', 'y_mm', 'z_mm')) <= POS_MM and e['rotation_deg'] <= ANG_DEG
    assert pg.table.rowCount() == 6 and pg.table.item(0, 1).text() == 'Used'
    assert pg.tiles['verdict'].sev == 'warning' and pg.view.value() == 'overlay' and pg.overlay.isVisible()
    assert 'T_camera_lidar' in pg.matrix.text() and len(pg.findings) == len(res.findings)
    pg.pose_pick.buttons['2'].click()
    pg.zoom.buttons['image'].click()
    pg.mode.buttons['plan'].click()
    pump(app, 0.1)
    assert pg.overlay.k == 2 and pg.overlay.mode == 'plan' and pg.overlay.zoom == 'image'
    pg.overlay.grab()
    # apply the measured pose: one undo step
    lid = next(s for s in store.variant.sensors if s.tag == 'L1')
    planned = (lid.x, lid.y, lid.z, lid.yaw)
    assert pg.b_apply.isEnabled() and pg.b_apply.text() == 'Apply to L1'
    QTest.mouseClick(pg.b_apply, Qt.MouseButton.LeftButton)
    pump(app, 0.1)
    lid = next(s for s in store.variant.sensors if s.tag == 'L1')
    assert (lid.x, lid.y, lid.z) == pytest.approx((res.measured['x'], res.measured['y'], res.measured['z']), abs=1e-4)
    assert pg.stale and 'stale' in pg.status.text()
    store.undo.undo()
    pump(app, 0.1)
    lid = next(s for s in store.variant.sensors if s.tag == 'L1')
    assert (lid.x, lid.y, lid.z, lid.yaw) == planned and not pg.stale
    # ... or as a new variant
    n = len(store.project.variants)
    QTest.mouseClick(pg.b_variant, Qt.MouseButton.LeftButton)
    pump(app, 0.1)
    assert len(store.project.variants) == n + 1 and store.variant.name.endswith('· lidar calibrated')
    assert not pg.b_apply.isEnabled()                       # the result belongs to the variant it was made for
    store.undo.undo()
    pump(app, 0.1)
    assert len(store.project.variants) == n and store.variant.uid == res.variant_uid
    # export and the captures as a folder of pairs
    paths = pg.export(str(tmp_path / 'export'))
    assert json.loads(open(paths[0]).read())['schema'] == LC.SCHEMA and len(paths) == 7
    saved = pg.save_captures(str(tmp_path / 'pairs'))
    assert len(saved) == 12 and LC.load_folder(str(tmp_path / 'pairs'))[0][0].name == 'pose01'
    # back to the defaults; the project replaced drops the result
    pg.b_defaults.click()
    pump(app, 0.1)
    assert store.project.analysis.lidar_calib == {} and pg.cam_box.currentData() == ''
    store.set_project(_rig())
    pump(app, 0.2)
    assert pg.result is None and pg.table.rowCount() == 0
