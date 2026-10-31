"""The board pre-check (calib.board_possible): a frame with no board in it (sensor noise at any level, black, the
calibration coach's empty room, at up to 3840 × 2160) is turned away in well under a second by the lens calibration's
detector (calib.find_board) and the lidar–camera calibration's (lidar_calib.detect_board without a plan, the real-data
path), where their searches took 2–20 s and more; every board those detectors found before is found again with
bit-identical corners and pose (compared with the pre-check switched off, i.e. the code before it); degraded boards
(blur, dark exposure, haze, gamma, clipping, noise) still pass with margin.

Measured offline when the thresholds were chosen (rendered boards with the pre-check off, the detector's answer as
truth): 321 lens boards found (8 library lenses, 1/4 to full resolution, 9 × 6 and 7 × 5) and 98 lidar boards found
(cameras F, R, ML of the data rig, plain and parking scenes, 2.5–11 m, tilted, near the fisheye edge), each also
degraded at random (gain 0.12–2 with clipping, gamma 0.6–1.6, haze up to 60 grey levels, Gaussian blur up to 4 px,
noise up to σ 8, a grey patch over part of the frame): the pre-check rejected one, a 480 × 384 frame blurred by
about 0.4 of a square; in the realistic range (gain ≥ 0.25, haze ≤ 30, blur ≤ 2.5 px at half resolution) the
smallest corner group was 1.4 × the board's inner corners against the 0.5 × threshold. Cluttered board-free scenes
(the sample's parking and plain scenes, with aliased edges, striped posts and tiled walls) often pass and are searched.

The search then takes the first grid that is a board: its squares alternate like a print (Weber contrast ≥ 0.15, ≥ 80 %
of neighbours in order) and its fitted pose misses its corners by ≤ 2 px. The old search reported the first grid it
put together: in 29 of 48 board-free sample frames a grid of floor grain or clutter (rms 1.1–30 px), and in 11 of 98
frames with a board a wrong grid instead of the board. Measured: board-free false boards 29 → 0 (11 → 0 with floor
targets); the 87 true boards found before are identical; 2 boards the old search missed are now found; true boards
fit to ≤ 0.29 px (≤ 1.22 px when the frames come from a unit whose lens differs by up to 5 % / 50 px / 5 % from the
model used, where the old search returned wrong grids 11 times and the new one the board). Floor grain reaches
Weber ≤ 0.13; true boards ≥ 0.18 (haze 60 at gain 0.12), ≥ 0.5 in the realistic range. Search time on cluttered
board-free 1920 × 1536 frames: detect_board median 4.1 → 0.75 s, worst 13.9 → 4.1 s; find_board median
1.7 → 1.0 s, worst 3.1 → 2.1 s.

The classic detector on the whole image (1–2 s a call on clutter, the 4.1 s worst case) now runs only once a view has
found a board, and its board is then taken first as before: on the same render set the 89 true boards are identical
(5 of them would otherwise come from SB in a view instead: same board, error to the true corners 0.202–0.207 px rms
either way), still 0 false boards in 72 board-free frames, and detect_board on the 48 board-free frames takes a
median 0.62–0.77 s and at worst 1.4–1.7 s (was 4.1 s); frames with a board it cannot find, worst 4.2 → 1.3–1.6 s."""
import time

import cv2
import numpy as np
import pytest

from vantage.core import calib, lidar_calib as LC, samples
from vantage.core.analysis import Engine
from vantage.core.lens import builtin_lenses, builtin_modules, resolve

MOD = builtin_modules()[0]
WIDE = builtin_lenses()[0]
LENS_PAT = (9, 6)
CLUTTER_S = 1.5            # a cluttered board-free frame at 1920 × 1536 (measured: see the module docstring)
LIMIT_S = 2.0              # generous bound for a rejected frame (measured 0.02–0.15 s; target 0.3 s at 1920 × 1080)


def _rig():
    """The sample project with its data rig active and no floor targets (they are chessboards themselves)."""
    p = samples.tigor_sample_project()
    p.active_uid = next(v for v in p.variants if v.sensors).uid
    p.targets = []
    return p


@pytest.fixture(scope='module')
def rig():
    p = _rig()
    cam = p.variant.by_tag('F')
    lid = next(s for s in p.variant.sensors if s.tag == 'L1')
    return p, Engine(), cam, lid


def _old_code(monkeypatch):
    """The detectors as they were before the pre-check and the faster search: no pre-check, every view searched,
    install._find (both pattern orientations, the classic detector without its fast check), the first grid taken
    whatever its fit or look."""
    from vantage.core import install
    monkeypatch.setattr(calib, 'board_possible', lambda *a, **k: True)
    monkeypatch.setattr(calib, 'groups_in_view', lambda *a, **k: True)
    monkeypatch.setattr(LC, '_find_quick', lambda gray, pattern, **k: install._find(gray, pattern))
    monkeypatch.setattr(LC, 'WHOLE_CLASSIC_LATE', False)
    monkeypatch.setattr(LC, 'MAX_BOARD_RMS_PX', float('inf'))
    monkeypatch.setattr(LC, '_checker_like', lambda *a, **k: True)


def _noisy(img, sigma, seed):
    f = img.astype(np.float32) + np.random.default_rng(seed).normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(f, 0, 255).astype(np.uint8)


def _room(intr, w, h, seed=1):
    """The calibration coach's empty frame: its softly textured room through the camera's lens, sensor noise."""
    from vantage.core.synth import pixel_rays
    d, th = pixel_rays(intr, w, h)
    img = np.full(h * w, 40.0)
    inside = th <= intr.max_theta
    az = np.arctan2(d[:, 1], d[:, 0])
    img[inside] = 120 + 18 * np.sin(az[inside] * 3) * np.cos(th[inside] * 5)
    return _noisy(img.reshape(h, w), 2.0, seed)


def _empty_frames(intr):
    """Board-free frames: flat grey with sensor noise at several sizes, dark and noisy, the coach's room, black."""
    out = []
    for (w, h) in ((1920, 1080), (1920, 1536), (3840, 2160)):
        out.append((f'grey+noise {w}x{h}', _noisy(np.full((h, w), 128, np.uint8), 8.0, w)))
    out.append(('dark+noise 1920x1080', _noisy(np.full((1080, 1920), 12, np.uint8), 3.0, 2)))
    out.append(('coach room 1920x1536', _room(intr, 1920, 1536)))
    out.append(('black 1920x1080', np.zeros((1080, 1920), np.uint8)))
    return out


# =========================================================================== speed
def test_board_free_frames_are_rejected_fast(rig, record_property):
    p, eng, cam, lid = rig
    st = LC.LidarCalibSettings.from_dict(dict(camera='F', lidar='L1'))
    intr = resolve(p.module_for(cam), p.lens_for(cam))
    calib.board_possible(np.zeros((64, 64), np.uint8), LENS_PAT)            # first-call imports out of the timing
    times = {}
    for name, img in _empty_frames(intr):
        t0 = time.perf_counter()
        assert calib.find_board(img, LENS_PAT, intr) is None
        t1 = time.perf_counter()
        assert LC.detect_board(p, cam, img, st, None) is None
        t2 = time.perf_counter()
        times[name] = (t1 - t0, t2 - t1)
        record_property(f'find_board {name} s', round(t1 - t0, 4))
        record_property(f'detect_board {name} s', round(t2 - t1, 4))
    print('rejected in (find_board, detect_board) s:', {k: (round(a, 3), round(b, 3)) for k, (a, b) in times.items()})
    assert max(max(v) for v in times.values()) < LIMIT_S


def test_a_planned_board_that_is_not_there_is_rejected_fast(rig):
    """With a hint (the plan) but no board at it, the fallback whole-image search is pre-checked too."""
    p, eng, cam, lid = rig
    st = LC.LidarCalibSettings.from_dict(dict(camera='F', lidar='L1'))
    poses, _ = LC.plan_poses(p, cam, lid, st, eng)
    img = _room(resolve(p.module_for(cam), p.lens_for(cam)), 1920, 1536)
    t0 = time.perf_counter()
    assert LC.detect_board(p, cam, img, st, poses[0]) is None
    assert time.perf_counter() - t0 < LIMIT_S


def test_cluttered_board_free_scenes_give_no_board_in_bounded_time(rig, record_property, monkeypatch):
    """The sample's scenes (aliased edges, striped posts, a tiled wall: chessboard-like clutter the pre-check lets
    through) at the camera's full 1920 × 1536: no board (the old search assembled grids from clutter and reported
    them, e.g. 25 px rms in the plain scene through camera F) and a bounded search time. detect_board never runs the
    classic detector on the whole image there (1–2 s a call on clutter): it runs only once a view has found a board."""
    from vantage.core.synth import render_world
    p, eng, cam, lid = rig
    calls = []
    find_classic = LC._find_classic
    monkeypatch.setattr(LC, '_find_classic', lambda g, pat: calls.append(g.shape) or find_classic(g, pat))
    for scene, tag in (('plain', 'F'), ('parking', 'F'), ('parking', 'MR')):
        c = p.variant.by_tag(tag)
        intr = resolve(p.module_for(c), p.lens_for(c))
        st = LC.LidarCalibSettings.from_dict(dict(camera=tag))
        img = render_world(p, eng, c, LC.make_world(p, scene, 8.0), 1920, 1536, noise=2.0)
        t0 = time.perf_counter()
        assert LC.detect_board(p, c, img, st, None) is None, (scene, tag)
        t1 = time.perf_counter()
        assert calib.find_board(img, (st.cols, st.rows), intr) is None
        t2 = time.perf_counter()
        assert not calls, (scene, tag)
        record_property(f'detect_board {scene} {tag} s', round(t1 - t0, 3))
        record_property(f'find_board {scene} {tag} s', round(t2 - t1, 3))
        print(f'{scene} {tag}: detect_board {t1 - t0:.2f} s, find_board {t2 - t1:.2f} s')
        assert t1 - t0 < 3 * CLUTTER_S and t2 - t1 < 3 * CLUTTER_S


# =========================================================================== no regression
def _same(a, b):
    return (a is None and b is None) or (a is not None and b is not None and np.array_equal(a, b))


def test_lens_boards_are_found_with_the_same_corners(monkeypatch):
    intr = resolve(MOD, WIDE)
    views = calib.board_views(MOD, WIDE, None, LENS_PAT, 60.0, n=6, seed=1)
    rng = np.random.default_rng(11)
    poses = calib._random_poses(intr, LENS_PAT, 60.0, 40, rng, 960, 768)
    edge = max(poses, key=lambda P: np.linalg.norm(P[1][:2]) / P[1][2])       # the most off-axis board
    views.append(('edge', calib.render_board(intr, edge[0], edge[1], LENS_PAT, 60.0, 960, 768)))
    small = calib.board_views(MOD, WIDE, None, LENS_PAT, 60.0, n=3, seed=4, scale=0.25)
    with_pre = [calib.find_board(img, LENS_PAT, intr) for _, img in views + small]
    with_pre_raw = [calib.find_board(img, LENS_PAT) for _, img in views[:2]]
    _old_code(monkeypatch)
    before = [calib.find_board(img, LENS_PAT, intr) for _, img in views + small]
    before_raw = [calib.find_board(img, LENS_PAT) for _, img in views[:2]]
    assert all(c is not None for c in before)
    assert all(_same(a, b) for a, b in zip(with_pre, before))
    assert all(_same(a, b) for a, b in zip(with_pre_raw, before_raw))


def test_lidar_boards_are_found_with_the_same_pose(rig, monkeypatch):
    """detect_board without a hint (the real-data path) on the planned boards of camera F."""
    p, eng, cam, lid = rig
    st = LC.LidarCalibSettings.from_dict(dict(camera='F', lidar='L1', seed=4))
    poses, _ = LC.plan_poses(p, cam, lid, st, eng)
    caps = LC.simulate(p, eng, cam, lid, [poses[0], poses[-1]], st)
    now = [LC.detect_board(p, cam, c.image, st, None) for c in caps]
    _old_code(monkeypatch)
    before = [LC.detect_board(p, cam, c.image, st, None) for c in caps]
    for a, b in zip(now, before):
        assert a is not None and b is not None
        assert a.how == b.how and a.phase == b.phase and a.rms_px == b.rms_px
        for k in ('uv', 'rays', 'R', 'c', 'cov'):
            assert np.array_equal(getattr(a, k), getattr(b, k)), k


def test_whole_image_classic_corners_run_late_give_the_same_board(rig, monkeypatch):
    """When SB finds nothing on the whole image, the classic detector's whole-image corners, tried only once a view has
    found the board (WHOLE_CLASSIC_LATE), give the board the classic-first search gave: same corners, pose, how. (SB
    is made to miss on the whole image so the classic detector is what finds it there.)"""
    p, eng, cam, lid = rig
    st = LC.LidarCalibSettings.from_dict(dict(camera='F', lidar='L1', seed=4))
    poses, _ = LC.plan_poses(p, cam, lid, st, eng)
    caps = LC.simulate(p, eng, cam, lid, [poses[0], poses[3]], st)      # boards the classic detector finds whole
    find_quick = LC._find_quick

    def sb_misses_whole(gray, pattern, precheck=True, classic=True):
        if not precheck:                                    # the whole-image call
            return LC._find_classic(gray, pattern) if classic else None
        return find_quick(gray, pattern, precheck=precheck, classic=classic)
    monkeypatch.setattr(LC, '_find_quick', sb_misses_whole)
    late = [LC.detect_board(p, cam, c.image, st, None) for c in caps]
    monkeypatch.setattr(LC, 'WHOLE_CLASSIC_LATE', False)
    first = [LC.detect_board(p, cam, c.image, st, None) for c in caps]
    for a, b in zip(late, first):
        assert a is not None and b is not None and a.how == b.how == 'whole image'
        assert a.phase == b.phase and a.rms_px == b.rms_px
        for k in ('uv', 'rays', 'R', 'c', 'cov'):
            assert np.array_equal(getattr(a, k), getattr(b, k)), k


# =========================================================================== the pre-check itself
def _degrade(img, blur=0.0, gain=1.0, haze=0.0, gamma=1.0, noise=2.0, seed=0):
    f = img.astype(np.float32)
    if blur:
        f = cv2.GaussianBlur(f, (0, 0), blur)
    f = np.clip(f * gain + haze, 0, 255)
    f = 255.0 * (f / 255.0) ** gamma
    return _noisy(f, noise, seed)


def test_degraded_boards_pass_with_margin():
    intr = resolve(MOD, WIDE)
    rng = np.random.default_rng(3)
    poses = calib._random_poses(intr, LENS_PAT, 60.0, 6, rng, 960, 768)
    need = calib.PRE_FRAC * LENS_PAT[0] * LENS_PAT[1]
    for Rb, C in poses:
        clean = calib.render_board(intr, Rb, C, LENS_PAT, 60.0, 960, 768, noise=0.0)
        for kw in (dict(), dict(blur=2.5), dict(gain=0.25), dict(gain=0.25, haze=30.0), dict(gamma=0.6),
                   dict(gamma=1.6), dict(noise=8.0), dict(gain=2.0), dict(blur=1.5, gain=0.5, noise=4.0)):
            img = _degrade(clean, **kw)
            assert calib.board_lattice(img) >= 1.5 * need, kw


def test_noise_at_any_level_and_exposure_fails_the_precheck():
    for mean in (8, 40, 128, 230):
        for sigma in (1.0, 4.0, 12.0, 25.0):
            img = _noisy(np.full((768, 960), mean, np.uint8), sigma, int(mean * 100 + sigma))
            assert calib.board_lattice(img) < calib.PRE_FRAC * 35, (mean, sigma)


def test_precheck_edge_cases_and_determinism():
    img = calib.board_views(MOD, WIDE, None, LENS_PAT, 60.0, n=1, seed=2)[0][1]
    rgb = np.repeat(img[..., None], 3, -1)
    assert calib.board_lattice(img) == calib.board_lattice(img) == calib.board_lattice(rgb)
    assert calib.board_possible(rgb, LENS_PAT)
    for tiny in (np.zeros((10, 12), np.uint8), np.full((40, 300), 200, np.uint8), np.zeros((0, 0), np.uint8)):
        assert not calib.board_possible(tiny, LENS_PAT)
